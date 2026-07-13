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

"""W8A8_HIF8 quantization scheme for Ascend NPU (per-element, native).

HiF8 is Huawei's native 8-bit floating point format. This scheme uses
per-element quantization without any external scale tensors:
  - Weight: stored directly in HiF8 dtype (hifloat8), no per-channel scale
  - Activation: dynamically converted to HiF8 at runtime, per-element
  - MoE: grouped matmul with native HiF8 dtype

Each element is independently quantized by the HiF8 float format's
representable precision — no shared exponents, no blocks, no scales.
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


def _get_hif8_dtype() -> torch.dtype:
    """Get the native HiF8 dtype from torch_npu."""
    from vllm_ascend.device.mxfp_compat import HIFLOAT8_DTYPE

    if HIFLOAT8_DTYPE is not None:
        return HIFLOAT8_DTYPE
    # Fallback: float8_e4m3fn for environments without full HiF8 support
    return torch.float8_e4m3fn


@register_scheme("W8A8_HIF8", "linear")
class AscendW8A8HiF8LinearMethod(AscendLinearScheme):
    """Linear method for per-element W8A8_HIF8.

    No external scale tensors — weights and activations are independently
    quantized per-element via native hifloat8 dtype conversion.
    """

    def __init__(self):
        pass

    def get_weight(
        self, input_size: int, output_size: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        """Weight stored directly in HiF8 dtype."""
        weight_dtype = _get_hif8_dtype()
        return {"weight": torch.empty(output_size, input_size, dtype=weight_dtype)}

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        """Forward pass: convert both inputs to HiF8, matmul, cast back.

        Per-element quantization — no dynamic scale computation, each element
        independently quantized by the hifloat8 format.
        """
        weight_dtype = _get_hif8_dtype()
        x_dtype = x.dtype

        # Convert activation to HiF8
        x_hif8 = x.to(weight_dtype)

        # Matmul in HiF8, cast output back to original dtype
        output = F.linear(x_hif8, layer.weight, bias=None).to(x_dtype)

        if bias is not None:
            output = output + bias

        return output

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Post-loading: transpose to NPU layout and cast to NZ format."""
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight.data = maybe_trans_nz(layer.weight.data)


@register_scheme("W8A8_HIF8", "moe")
class AscendW8A8HiF8FusedMoEMethod(AscendMoEScheme):
    """FusedMoE method for per-element W8A8_HIF8.

    MoE expert weights stored directly in HiF8 dtype, no external scales.
    """

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
        """MoE expert weights in native HiF8 dtype."""
        weight_dtype = _get_hif8_dtype()
        return {
            "w13_weight": torch.empty(
                num_experts, 2 * intermediate_size_per_partition, hidden_sizes,
                dtype=weight_dtype,
            ),
            "w2_weight": torch.empty(
                num_experts, hidden_sizes, intermediate_size_per_partition,
                dtype=weight_dtype,
            ),
        }

    def get_dynamic_quant_param(
        self,
        num_experts: int,
        intermediate_size_per_partition: int,
        hidden_sizes: int,
        params_dtype: torch.dtype,
    ) -> dict[str, Any]:
        """No external scale params needed for per-element HiF8."""
        return {}

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
        """MoE forward with per-element HiF8 — delegates to fused experts path."""
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
        """Post-loading: transpose and NZ format for MoE weights."""
        from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

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
