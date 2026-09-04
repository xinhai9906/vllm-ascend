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

"""W8A8_HIF8 pseudo-quantization scheme for Ascend NPU.

Configurable-granularity fake quant matching verl QAT:
  scale = amax / 49152
  pseudo = _quant_hif8(tensor / scale) * scale

Granularity modes (per_tensor / per_channel / per_group / per_group_median /
per_channel_median) share the same _quant_hif8 tapered-precision kernel as
training QAT. per_group_median and per_channel_median anchor the median of
|x| to 1.0 with amax/49152 as the lower bound, so no truncation occurs.

When rotation is enabled, activations are rotated by a block-Hadamard-sign
matrix before quantisation; weights arrive pre-rotated+quantised from the
verl training side via ZMQ, so no further rotation is needed on the weight
path.
"""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ops.fused_moe.moe_stage_params import MoERotationParams

from ..block_rotation import apply_block_rotation
from .base import AscendLinearScheme, AscendMoEScheme, QuantType, get_moe_num_logical_experts
from .registry import register_scheme


# HiF8 max: 2^15 × 1.5 = 49152 (Dot=4, E=±15, M=1bit)
_HIF8_MAX: float = 49152.0


def _quant_hif8(x: torch.Tensor) -> torch.Tensor:
    """Raw HiF8 quantization with tapered precision (per element).
    Mantissa bits by exponent magnitude:
      |e| <= 3  → 3 bits (Dot=2)
      |e| <= 7  → 2 bits (Dot=3)
      |e| <= 15 → 1 bit (Dot=4)
    """
    x_unsigned = x.abs()
    sign = x.sign()
    eps = x_unsigned.amax().clamp(min=1e-30) * 1e-8
    e = torch.floor(torch.log2(x_unsigned + eps))
    abse = e.abs()
    mant_bits = torch.where(abse <= 3, 3.0,
                   torch.where(abse <= 7, 2.0,
                   torch.where(abse <= 15, 1.0, 0.0)))
    q = torch.floor(x_unsigned * 2.0 ** (-e + mant_bits) + 0.5)
    return q * 2.0 ** (e - mant_bits) * sign


def _hif8_fake_quant(
    tensor: torch.Tensor, granularity: str = "per_tensor", group_size: int = 32
) -> torch.Tensor:
    """HiF8 fake quant matching verl QAT, with configurable granularity.
    granularity='per_tensor':  one scale per tensor
    granularity='per_channel': one scale per output channel (weight) / per token (act)
    granularity='per_group':   one scale per group of group_size elements
    granularity='per_group_median': one scale per group, anchored to the median
        of |x| (average of the two middle sorted values for even group_size,
        single middle for odd) mapped to 1.0, with amax/HIF8_MAX as the lower
        bound so truncation never occurs.
    granularity='per_channel_median': one scale per row, anchored to the median
        of |x| over the whole row, with amax/HIF8_MAX as the lower bound so
        truncation never occurs.
    """
    if granularity in ("per_group", "per_group_median"):
        t = tensor.float()
        dim_size = t.shape[-1]
        pad = (group_size - dim_size % group_size) % group_size
        if pad:
            t = F.pad(t, (0, pad))            # zeros, for amax + quant
        t_blocks = t.unflatten(-1, (-1, group_size))
        amax = t_blocks.abs().amax(dim=-1, keepdim=True)
        if granularity == "per_group_median":
            # Median of |x| per group: average of the two middle sorted values
            # (indices gs//2-1, gs//2 for even gs; single middle for odd gs).
            # Pad positions are imputed +inf so they sort to the tail and
            # cannot distort the valid median; a tail block with gs//2 or
            # fewer valid elements has +inf inside the slice, which
            # posinf=0.0 maps to the amax-based lower bound. amax stays on
            # the zero-padded t_blocks so a padded tail never reads +inf.
            abs_med = tensor.float().abs()
            if pad:
                abs_med = F.pad(abs_med, (0, pad), value=float("inf"))
            sorted_vals = abs_med.unflatten(-1, (-1, group_size)).sort(dim=-1).values
            lo = (group_size - 1) // 2
            hi = group_size // 2 + 1
            median_avg = sorted_vals[..., lo:hi].mean(dim=-1, keepdim=True)
            median_avg = torch.nan_to_num(median_avg, nan=0.0, posinf=0.0)
            scale = torch.maximum(median_avg, amax / _HIF8_MAX).clamp(min=1e-12)
        else:
            scale = (amax / _HIF8_MAX).clamp(min=1e-12)
        q_blocks = _quant_hif8(t_blocks / scale) * scale
        result = q_blocks.flatten(-2, -1)
        if pad:
            result = result[..., :dim_size]
        return result.to(tensor.dtype)
    elif granularity == "per_channel_median":
        # Median of |x| per channel (row): average of the two middle sorted
        # values (indices dim//2-1, dim//2 for even dim, single middle for
        # odd), anchored to 1.0, with amax/HIF8_MAX as the lower bound so
        # truncation never occurs.  No padding — the whole row is sorted, so
        # there is no tail-group edge case (unlike per_group_median).
        t = tensor.float()
        abs_t = t.abs()
        sorted_vals = abs_t.sort(dim=-1).values
        dim_size = t.shape[-1]
        lo = (dim_size - 1) // 2
        hi = dim_size // 2 + 1
        median_avg = sorted_vals[..., lo:hi].mean(dim=-1, keepdim=True)
        median_avg = torch.nan_to_num(median_avg, nan=0.0)
        amax = abs_t.amax(dim=-1, keepdim=True)
        scale = torch.maximum(median_avg, amax / _HIF8_MAX).clamp(min=1e-12)
        return (_quant_hif8(t / scale) * scale).to(tensor.dtype)
    elif granularity == "per_channel":
        amax = tensor.float().abs().amax(dim=-1, keepdim=True)
    else:
        amax = tensor.float().abs().max()
    scale = (amax / _HIF8_MAX).clamp(min=1e-12)
    return (_quant_hif8(tensor.float() / scale) * scale).to(tensor.dtype)


@register_scheme("W8A8_HIF8", "linear")
class AscendW8A8HiF8LinearMethod(AscendLinearScheme):
    """Per-tensor/per-channel/per-group pseudo-quant matching verl QAT."""

    def __init__(
        self,
        granularity: str = "per_tensor",
        group_size: int = 32,
        rotation_enable: bool = False,
        rotation_block_size: int = 32,
        rotation_seed: int = 0,
    ):
        self.granularity = granularity
        self.group_size = group_size
        self.rotation_enable = rotation_enable
        self.rotation_block_size = rotation_block_size
        self.rotation_seed = rotation_seed

    def get_weight(
        self, input_size: int, output_size: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        """Weight stored in model dtype (bf16/fp16)."""
        return {"weight": torch.empty(output_size, input_size, dtype=params_dtype)}

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        """Pseudo-quant: activation quant every forward, weight already
        pre-quantized in process_weights_after_loading (called at init and
        after every verl weight sync).

        When rotation is enabled, the activation is rotated by Q before
        quantisation so that quant noise matches the training distribution.
        Weights arrive pre-rotated+quantised from verl, so no weight-side
        rotation is needed here.
        """
        x_dtype = x.dtype
        # Apply block rotation to activation BEFORE quantisation.
        x_rotated = apply_block_rotation(
            x,
            enable=self.rotation_enable,
            block_size=self.rotation_block_size,
            seed=self.rotation_seed,
        )
        x_fq = _hif8_fake_quant(
            x_rotated, self.granularity, self.group_size
        ).contiguous()
        output = F.linear(x_fq.to(x_dtype), layer.weight, bias=bias)
        return output

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Weights are pre-quantized on the verl side before ZMQ send.
        Nothing to do here — apply() uses layer.weight directly."""


@register_scheme("W8A8_HIF8", "moe")
class AscendW8A8HiF8FusedMoEMethod(AscendMoEScheme):
    """FusedMoE pseudo-quantization: bf16→hifloat8 each forward."""

    quant_type: QuantType = QuantType.W8A8HIF8

    def __init__(
        self,
        granularity: str = "per_tensor",
        group_size: int = 32,
        rotation_enable: bool = False,
        rotation_block_size: int = 32,
        rotation_seed: int = 0,
    ):
        from vllm.config import CompilationMode, get_current_vllm_config

        vllm_config = get_current_vllm_config()
        ascend_config = get_ascend_config()
        self.granularity = granularity
        self.group_size = group_size
        self.rotation_enable = rotation_enable
        self.rotation_block_size = rotation_block_size
        self.rotation_seed = rotation_seed
        self.use_aclgraph = (
            vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
            and not vllm_config.model_config.enforce_eager
        )
        self.multistream_overlap_gate = ascend_config.multistream_overlap_gate
        self.dynamic_eplb = ascend_config.eplb_config.dynamic_eplb
        self.in_dtype = vllm_config.model_config.dtype
        self.supports_eplb = True

        try:
            from vllm_ascend.distributed.parallel_state import get_mc2_group

            device_group = get_mc2_group().device_group
            local_rank = torch.distributed.get_rank(group=device_group)
            backend = device_group._get_backend(torch.device("npu"))
            self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)
        except (AttributeError, RuntimeError):
            from vllm.logger import logger

            logger.warning_once(
                "[vllm-ascend/W8A8_HIF8] MC2 group metadata unavailable, "
                "falling back to empty moe_all_to_all_group_name."
            )
            self.moe_all_to_all_group_name = ""

    def get_weight(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        """MoE expert weights in model dtype (bf16/fp16)."""
        return {
            "w13_weight": torch.empty(
                num_experts, 2 * intermediate_size_per_partition, hidden_sizes,
                dtype=params_dtype,
            ),
            "w2_weight": torch.empty(
                num_experts, hidden_sizes, intermediate_size_per_partition,
                dtype=params_dtype,
            ),
        }

    def get_dynamic_quant_param(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Dummy scale ones — kernel requires but pseudo-quant ignores."""
        return {
            "w13_weight_scale": torch.ones(num_experts, 1, dtype=torch.float32),
            "w2_weight_scale": torch.ones(num_experts, 1, dtype=torch.float32),
        }

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor | None = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: Any | None = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        mc2_mask: torch.Tensor | None = None,
        tid2eid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """MoE forward — delegates to fused experts path."""
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
        from vllm_ascend.flash_common3_context import get_flash_common3_context
        from vllm_ascend.ops.fused_moe.experts_selector import select_experts, zero_experts_compute
        from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input

        zero_expert_num = getattr(layer, "zero_expert_num", 0)
        zero_expert_type = getattr(layer, "zero_expert_type", None)
        n_shared_experts = getattr(layer, "n_shared_experts", 0) or 0
        mix_placement = getattr(layer, "mix_placement", False)

        num_logical_experts = get_moe_num_logical_experts(
            layer, num_experts,
            global_redundant_expert_num=global_redundant_expert_num,
            num_shared_experts=n_shared_experts,
        )

        if self.multistream_overlap_gate:
            fc3_context = get_flash_common3_context()
            assert fc3_context is not None
            topk_weights = fc3_context.topk_weights
            topk_ids = fc3_context.topk_ids
        else:
            topk_weights, topk_ids = select_experts(
                hidden_states=x, router_logits=router_logits, top_k=top_k,
                use_grouped_topk=use_grouped_topk, renormalize=renormalize,
                topk_group=topk_group, num_expert_group=num_expert_group,
                custom_routing_function=custom_routing_function,
                scoring_func=scoring_func,
                routed_scaling_factor=routed_scaling_factor,
                e_score_correction_bias=e_score_correction_bias,
                mix_placement=mix_placement,
                num_logical_experts=router_logits.shape[1],
                num_shared_experts=n_shared_experts,
                num_experts=num_logical_experts,
                tid2eid=tid2eid,
            )

        if zero_expert_num > 0 and zero_expert_type is not None:
            topk_ids, topk_weights, zero_expert_result = zero_experts_compute(
                expert_indices=topk_ids, expert_scales=topk_weights,
                num_experts=num_logical_experts, zero_expert_type=zero_expert_type,
                hidden_states=x,
            )

        if enable_force_load_balance:
            random_matrix = torch.rand(
                topk_ids.size(0), num_logical_experts, device=topk_ids.device
            )
            topk_ids = torch.argsort(random_matrix, dim=1)[:, : topk_ids.size(1)].to(topk_ids.dtype)

        topk_weights = topk_weights.to(self.in_dtype)

        moe_comm_method = _EXTRA_CTX.moe_comm_method

        # Block rotation: rotate the MoE input (hidden dim) by the same Q that
        # pre-rotated the w13 weights (done by the verl weight sync), so the
        # Q·Qᵀ cancellation holds and only the quant noise changes.  The
        # router gate consumes the unrotated x with unrotated weights, so
        # routing logits are identical to training (which rotates both).
        # The intermediate activation is rotated inside unquant_apply_mlp,
        # driven by the rotation params passed to build_fused_experts_input.
        if self.rotation_enable:
            if n_shared_experts > 0:
                raise NotImplementedError(
                    "[W8A8_HIF8] Block rotation is not supported for MoE "
                    "blocks with shared experts: the training side runs them "
                    "unrotated, so rotating here would desynchronize the two "
                    "sides. Disable rotation for shared-expert models."
                )
            if apply_router_weight_on_input:
                raise NotImplementedError(
                    "[W8A8_HIF8] Block rotation is not supported when "
                    "apply_router_weight_on_input=True (routing is computed "
                    "inside the fused kernel from the rotated input). "
                    "Disable rotation for such models."
                )
            moe_comm_type = getattr(_EXTRA_CTX, "moe_comm_type", None)
            if moe_comm_type in (MoECommType.MC2, MoECommType.FUSED_MC2):
                raise NotImplementedError(
                    "[W8A8_HIF8] Block rotation is not supported with the MC2 "
                    "MoE comm path (the fused dispatch kernel cannot rotate "
                    "the intermediate activation). Disable rotation or use "
                    "all-to-all/all-gather MoE comm."
                )
            x = apply_block_rotation(
                x,
                enable=True,
                block_size=self.rotation_block_size,
                seed=self.rotation_seed,
            )
        x = _hif8_fake_quant(x, self.granularity, self.group_size).contiguous()

        # Expert weights arrive pre-rotated+pre-quantized from the verl weight
        # sync (rotation folded into the weights there); process_weights_after_loading
        # only transposes layouts.  Use directly.
        if self.dynamic_eplb:
            w1 = layer.w13_weight_list
            w2 = layer.w2_weight_list
        else:
            w1 = layer.w13_weight
            w2 = layer.w2_weight

        final_hidden_states = moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x, topk_weights=topk_weights, topk_ids=topk_ids,
                w1=w1, w2=w2, quant_type=QuantType.NONE,
                dynamic_eplb=self.dynamic_eplb, expert_map=expert_map,
                global_redundant_expert_num=global_redundant_expert_num,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                log2phy=log2phy, pertoken_scale=pertoken_scale,
                activation=activation,
                swiglu_limit=layer.swiglu_limit,
                w1_scale=[layer.w13_weight_scale],
                w2_scale=[layer.w2_weight_scale],
                rotation=MoERotationParams(
                    enable=self.rotation_enable,
                    block_size=self.rotation_block_size,
                    seed=self.rotation_seed,
                ),
            )
        )

        if zero_expert_num > 0 and zero_expert_type is not None:
            final_hidden_states += zero_expert_result

        return final_hidden_states

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2).contiguous()
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2).contiguous()

        if self.dynamic_eplb:
            layer.w13_weight_list = [
                weight.clone() for weight in layer.w13_weight.data.unbind(dim=0)
            ]
            layer.w2_weight_list = [
                weight.clone() for weight in layer.w2_weight.data.unbind(dim=0)
            ]
            del layer.w13_weight
            del layer.w2_weight
            torch.npu.empty_cache()
