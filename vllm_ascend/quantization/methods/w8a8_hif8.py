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

"""W8A8_HIF8 quantization scheme for Ascend NPU (per-tensor native + per-tensor scale).

HiF8 is Huawei's native 8-bit float with tapered precision.
NPU hardware handles Dot/Exponent/Mantissa encoding per tensor.
Scale uses tensorwise formula: scale = amax / F8max.

  - Weight:  per-tensor HiF8 (native Dot encoding) + per-tensor fp32 scale
  - Activation: per-tensor native HiF8 (no external scale)
  - MoE: grouped matmul with per-tensor HiF8 weights
"""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
import torch_npu

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.utils import maybe_trans_nz

from .base import AscendLinearScheme, AscendMoEScheme, QuantType, get_moe_num_logical_experts
from .registry import register_scheme


def _decode_hif8(b: torch.Tensor) -> torch.Tensor:
    """Decode HiF8 uint8 bytes back to float32 values.

    Bit layout per Dot band (exponent is sign-magnitude):
      Dot=4: S(7) | 11(6,5) | Es,Em(4..1) | M(0)
      Dot=3: S(7) | 10(6,5) | Es,Em(4..2) | M(1,0)
      Dot=2: S(7) | 01(6,5) | Es,Em(4,3) | M(2,1,0)
      Dot=1: S(7) | 001(6,5,4) | Es(3) | M(2,1,0)
      Dot=0: S(7) | 0001(6,5,4,3) | M(2,1,0)
    """
    bi = b.int()
    S = 1.0 - 2.0 * ((bi >> 7) & 1).float()

    # Dot prefix (after Sign at bit 7)
    t2 = (bi >> 5) & 0b3
    t3 = (bi >> 4) & 0x7
    t4 = (bi >> 3) & 0xF

    d4 = (t2 == 0b11)
    d3 = (t2 == 0b10)
    d2 = (t2 == 0b01)
    d1 = (t3 == 0b001)
    d0 = (t4 == 0b0001)

    # E field = [Es][Emag]. Extract Es and Emag.
    e_mag_bits = d4.int() * 3 + d3.int() * 2 + d2.int() * 1
    e_field = (d4.int() * ((bi >> 1) & 0xF)
               + d3.int() * ((bi >> 2) & 0x7)
               + d2.int() * ((bi >> 3) & 0x3)
               + d1.int() * ((bi >> 3) & 0x1)).int()
    Es = (e_field >> e_mag_bits) & 1  # exponent sign
    e_mag = e_field & (torch.pow(2.0, e_mag_bits.float()).int() - 1)

    e_implicit = d4.int() * 8 + d3.int() * 4 + d2.int() * 2 + d1.int() * 1
    e = (1 - 2 * Es.float()) * (e_implicit.float() + e_mag.float())  # signed exponent

    # Mantissa
    m_bits = d4.int() * 1 + d3.int() * 2 + (d2 | d1 | d0).int() * 3
    m_raw = (d4.int() * (bi & 0x1)
             + d3.int() * (bi & 0x3)
             + (d2 | d1 | d0).int() * (bi & 0x7)).float()

    return S * (1.0 + m_raw / torch.pow(2.0, m_bits.float())) * torch.pow(2.0, e)


@register_scheme("W8A8_HIF8", "linear")
class AscendW8A8HiF8LinearMethod(AscendLinearScheme):
    """Linear method for W8A8_HIF8: per-tensor native HiF8 + per-tensor scale.

    Weight: uint8 (HiF8 bytes) → _decode_hif8 → *scale → hifloat8 in post-load.
    Activation: per-tensor native via x.to(hifloat8) (NPU Dot encoding).
    """

    def __init__(self):
        pass

    def get_weight(
        self, input_size: int, output_size: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        """Weight stored as uint8 (encoded HiF8 bytes), decoded in post-load."""
        return {"weight": torch.empty(output_size, input_size, dtype=torch.uint8)}

    def get_pertensor_param(self, params_dtype: torch.dtype, **kwargs: Any) -> dict[str, Any]:
        """Per-tensor weight scale: one scalar fp32 for the whole weight tensor."""
        return {"weight_scale": torch.empty(1, dtype=torch.float32)}

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        """Forward: per-tensor native HiF8, matmul, cast back."""
        x_dtype = x.dtype
        x_hif8 = x.to(torch_npu.hifloat8)
        output = F.linear(x_hif8, layer.weight, bias=None).to(x_dtype)
        if bias is not None:
            output = output + bias
        return output

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Post-load: uint8→decode→×scale→hifloat8, transpose, NZ format."""
        weight_fp = _decode_hif8(layer.weight.data) * layer.weight_scale.data.float()
        layer.weight.data = weight_fp.to(torch_npu.hifloat8)
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight.data = maybe_trans_nz(layer.weight.data)


@register_scheme("W8A8_HIF8", "moe")
class AscendW8A8HiF8FusedMoEMethod(AscendMoEScheme):
    """FusedMoE method for W8A8_HIF8: per-tensor native HiF8 + per-expert scale."""

    quant_type: QuantType = QuantType.W8A8HIF8

    def __init__(self):
        from vllm.config import CompilationMode, get_current_vllm_config

        vllm_config = get_current_vllm_config()
        ascend_config = get_ascend_config()
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
        """MoE expert weights: uint8 (encoded HiF8 bytes), decoded in post-load."""
        return {
            "w13_weight": torch.empty(
                num_experts, 2 * intermediate_size_per_partition, hidden_sizes,
                dtype=torch.uint8,
            ),
            "w2_weight": torch.empty(
                num_experts, hidden_sizes, intermediate_size_per_partition,
                dtype=torch.uint8,
            ),
        }

    def get_dynamic_quant_param(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Per-tensor scales for MoE: one fp32 scalar per expert weight tensor."""
        return {
            "w13_weight_scale": torch.empty(num_experts, 1, dtype=torch.float32),
            "w2_weight_scale": torch.empty(num_experts, 1, dtype=torch.float32),
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
        """MoE forward with per-tensor HiF8 — delegates to fused experts path."""
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

        if self.dynamic_eplb:
            w1 = layer.w13_weight_list
            w2 = layer.w2_weight_list
        else:
            w1 = [layer.w13_weight]
            w2 = [layer.w2_weight]

        final_hidden_states = moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x, topk_weights=topk_weights, topk_ids=topk_ids,
                w1=w1, w2=w2, quant_type=self.quant_type,
                dynamic_eplb=self.dynamic_eplb, expert_map=expert_map,
                global_redundant_expert_num=global_redundant_expert_num,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=apply_router_weight_on_input,
                log2phy=log2phy, pertoken_scale=pertoken_scale,
                activation=activation,
                swiglu_limit=layer.swiglu_limit,
            )
        )

        if zero_expert_num > 0 and zero_expert_type is not None:
            final_hidden_states += zero_expert_result

        return final_hidden_states

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Post-load: uint8→decode→×scale→hifloat8, transpose, NZ format."""
        from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

        w13_scale = layer.w13_weight_scale.data.float()
        w2_scale = layer.w2_weight_scale.data.float()
        layer.w13_weight.data = (
            _decode_hif8(layer.w13_weight.data) * w13_scale.unsqueeze(-1)
        ).to(torch_npu.hifloat8)
        layer.w2_weight.data = (
            _decode_hif8(layer.w2_weight.data) * w2_scale.unsqueeze(-1)
        ).to(torch_npu.hifloat8)

        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2).contiguous()
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2).contiguous()

        layer.w13_weight.data = torch_npu.npu_format_cast(
            layer.w13_weight.data, ACL_FORMAT_FRACTAL_NZ
        )
        layer.w2_weight.data = torch_npu.npu_format_cast(
            layer.w2_weight.data, ACL_FORMAT_FRACTAL_NZ
        )

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
