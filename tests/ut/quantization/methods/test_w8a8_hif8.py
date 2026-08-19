#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Unit tests for the W8A8_HIF8 per_group_median scale algorithm.

per_group_median: for each group of group_size elements, the average of the
two middle sorted |x| values (median_avg) is anchored to 1.0, with
amax/HIF8_MAX as the lower bound so truncation never occurs:
    scale = max(median_avg, amax / 49152)
"""

import torch

from tests.ut.base import TestBase
from vllm_ascend.quantization.methods.w8a8_hif8 import _HIF8_MAX, _hif8_fake_quant, _quant_hif8


def _reference_scale(tensor: torch.Tensor, group_size: int) -> torch.Tensor:
    """Independent reference computation of the per_group_median scale.

    Mirrors the production formula but written as a straightforward
    reference: sort |x| per group, average the middle slice (indices
    gs//2-1, gs//2 for even gs), take max against amax/HIF8_MAX.
    """
    t = tensor.float()
    dim_size = t.shape[-1]
    pad = (group_size - dim_size % group_size) % group_size
    if pad:
        t = torch.nn.functional.pad(t, (0, pad))
    blocks = t.unflatten(-1, (-1, group_size))
    abs_blocks = blocks.abs()
    amax = abs_blocks.amax(dim=-1, keepdim=True)
    abs_med = tensor.float().abs()
    if pad:
        abs_med = torch.nn.functional.pad(abs_med, (0, pad), value=float("inf"))
    sorted_vals = abs_med.unflatten(-1, (-1, group_size)).sort(dim=-1).values
    lo = (group_size - 1) // 2
    hi = group_size // 2 + 1
    median_avg = sorted_vals[..., lo:hi].mean(dim=-1, keepdim=True)
    median_avg = torch.nan_to_num(median_avg, nan=0.0, posinf=0.0)
    return torch.maximum(median_avg, amax / _HIF8_MAX).clamp(min=1e-12)


class TestPerGroupMedianScale(TestBase):
    """per_group_median scale computation and invariants."""

    def test_scale_matches_reference(self):
        torch.manual_seed(0)
        for shape in [(4, 64), (3, 128), (1, 32)]:
            t = torch.randn(*shape) * 10
            group_size = 32
            ref_scale = _reference_scale(t, group_size)
            dim_size = t.shape[-1]
            pad = (group_size - dim_size % group_size) % group_size
            t_pad = torch.nn.functional.pad(t.float(), (0, pad)) if pad else t.float()
            blocks = t_pad.unflatten(-1, (-1, group_size))
            expected = (_quant_hif8(blocks / ref_scale) * ref_scale).flatten(-2, -1)
            if pad:
                expected = expected[..., :dim_size]
            out = _hif8_fake_quant(t, "per_group_median", group_size)
            self.assertEqual(out.shape, t.shape)
            self.assertEqual(out.dtype, t.dtype)
            torch.testing.assert_close(out.float(), expected, rtol=0.0, atol=0.0)

    def test_no_truncation_invariant(self):
        torch.manual_seed(1)
        group_size = 32
        for _ in range(5):
            t = torch.randn(8, 96) * 100
            # inject outliers to stress the lower bound
            t.view(8, -1, group_size)[:, :, -1] = torch.rand(8, 3) * 40000
            scale = _reference_scale(t, group_size)
            dim_size = t.shape[-1]
            pad = (group_size - dim_size % group_size) % group_size
            t_pad = torch.nn.functional.pad(t.float(), (0, pad)) if pad else t.float()
            blocks = t_pad.unflatten(-1, (-1, group_size))
            # scale >= amax/HIF8_MAX by construction => no element exceeds HIF8_MAX
            self.assertTrue((blocks / scale).abs().max().item() <= _HIF8_MAX + 1e-3)

    def test_median_maps_to_one_exactly(self):
        # Block whose two middle sorted values both equal 16 => median_avg = 16.
        # amax = 31, so the median dominates the lower bound: scale = 16 and
        # the middle elements map to exactly 1.0 (zero roundtrip error).
        block = torch.arange(32, dtype=torch.float32)
        block[15] = 16.0
        block[16] = 16.0
        out = _hif8_fake_quant(block.unsqueeze(0), "per_group_median", 32)
        self.assertEqual(out[0, 15].item(), 16.0)
        self.assertEqual(out[0, 16].item(), 16.0)

    def test_lower_bound_dominates_with_outlier(self):
        # 31 zeros + one huge outlier: median_avg = 0, scale falls back to
        # amax/HIF8_MAX — identical to the plain per_group (amax) mode, and
        # the outlier maps to exactly the format max.
        group_size = 32
        t = torch.zeros(1, group_size)
        outlier = 40000.0
        t[0, -1] = outlier
        out_median = _hif8_fake_quant(t, "per_group_median", group_size)
        out_amax = _hif8_fake_quant(t, "per_group", group_size)
        torch.testing.assert_close(out_median, out_amax, rtol=0.0, atol=0.0)
        self.assertAlmostEqual(out_median[0, -1].item(), outlier, delta=outlier * 1e-5)

    def test_pad_tail_block(self):
        # Last dim not divisible by group_size: shape preserved, finite
        # values, and the partial tail block falls back to the amax bound.
        torch.manual_seed(2)
        t = torch.randn(2, 33) * 5
        out = _hif8_fake_quant(t, "per_group_median", 32)
        self.assertEqual(out.shape, t.shape)
        self.assertTrue(torch.isfinite(out).all())
        ref_scale = _reference_scale(t, 32)
        dim_size = t.shape[-1]
        pad = (32 - dim_size % 32) % 32
        t_pad = torch.nn.functional.pad(t.float(), (0, pad)) if pad else t.float()
        blocks = t_pad.unflatten(-1, (-1, 32))
        expected = (_quant_hif8(blocks / ref_scale) * ref_scale).flatten(-2, -1)[..., :dim_size]
        torch.testing.assert_close(out.float(), expected, rtol=0.0, atol=0.0)

    def test_odd_group_size(self):
        # Odd group_size: single middle element (index gs//2) anchors to 1.0.
        group_size = 15
        block = torch.arange(group_size, dtype=torch.float32)
        # sorted ascending: index 7 == 7, amax == 14 => scale == 7
        out = _hif8_fake_quant(block.unsqueeze(0), "per_group_median", group_size)
        self.assertEqual(out[0, 7].item(), 7.0)
        ref_scale = _reference_scale(block.unsqueeze(0), group_size)
        expected = (_quant_hif8(block.unsqueeze(0) / ref_scale) * ref_scale).flatten(-2, -1)
        torch.testing.assert_close(out.float(), expected, rtol=0.0, atol=0.0)

    def test_zero_block(self):
        # All zeros: scale clamps to the 1e-12 floor, output stays zero.
        t = torch.zeros(2, 64)
        out = _hif8_fake_quant(t, "per_group_median", 32)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue((out == 0).all())

    def test_matches_per_group_when_median_small(self):
        # When median_avg < amax/HIF8_MAX in every group (extreme outliers),
        # per_group_median output equals plain per_group output.
        torch.manual_seed(3)
        group_size = 32
        t = torch.zeros(4, 64)
        t[:, 31] = torch.tensor([1e4, 2e4, 3e4, 4e4])
        out_median = _hif8_fake_quant(t, "per_group_median", group_size)
        out_amax = _hif8_fake_quant(t, "per_group", group_size)
        torch.testing.assert_close(out_median, out_amax, rtol=0.0, atol=0.0)
