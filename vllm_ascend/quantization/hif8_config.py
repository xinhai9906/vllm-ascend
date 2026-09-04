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

"""HiF8 pseudo-quantization config for vLLM Ascend.
Registers the "ascend-hif8" quantization method that routes to the
W8A8_HIF8 scheme for linear and MoE layers.
Config JSON format:
{
    "quant_method": "ascend-hif8",
    "granularity": "per_tensor",
    "group_size": 32,
    "ignore": ["lm_head", "embed_tokens"]
}
"""

from typing import Any, Optional, cast

import torch
from vllm.logger import logger
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    check_equal_or_regex_match,
)

from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.23.0"):
    from vllm.model_executor.layers.fused_moe import FusedMoE
else:
    from vllm.model_executor.layers.fused_moe import MoERunner

from .block_rotation import _parse_rotation_config
from .methods import get_scheme_class

ASCEND_HIF8_METHOD = "ascend-hif8"


def _is_fused_moe_layer(layer: torch.nn.Module) -> bool:
    """Check if a layer is a FusedMoE layer."""
    if vllm_version_is("0.23.0"):
        return isinstance(layer, FusedMoE)
    else:
        return isinstance(layer, MoERunner)


@register_quantization_config(ASCEND_HIF8_METHOD)
class AscendHiF8Config(QuantizationConfig):
    """Quantization config for Ascend HiF8 (W8A8_HIF8 pseudo-quant).
    Uses the same _quant_hif8 tapered-precision software quant as verl QAT.
    Supports per_tensor, per_channel, per_group, per_group_median, and
    per_channel_median granularity (per_group_median and per_channel_median
    anchor the median of |x| to 1.0 with amax/49152 as the lower bound).
    """

    def __init__(
        self,
        ignore: list[str],
        granularity: str = "per_tensor",
        group_size: int = 32,
        config: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.ignore = ignore
        self.granularity = granularity
        self.group_size = group_size
        self.quant_description = config if config is not None else {}

    def __repr__(self) -> str:
        return f"AscendHiF8Config({self.granularity} pseudo-quant HiF8)"

    @classmethod
    def get_name(cls) -> str:
        return ASCEND_HIF8_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError(
            "Ascend hardware does not support 'get_min_capability' feature."
        )

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AscendHiF8Config":
        ignore: list[str] = cast(list[str], config.get("ignore", []))
        granularity: str = cast(str, config.get("granularity", "per_tensor"))
        group_size: int = cast(int, config.get("group_size", 32))

        return cls(
            ignore=ignore,
            granularity=granularity,
            group_size=group_size,
            config=config,
        )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
        tid2eid=None,
    ) -> Optional["QuantizeMethodBase"]:
        from .method_adapters import (
            AscendFusedMoEMethod,
            AscendLinearMethod,
        )

        rotation = _parse_rotation_config(self.quant_description)

        if isinstance(layer, LinearBase):
            if prefix.endswith(".gate") or check_equal_or_regex_match(prefix, self.ignore):
                # Ignored layers (e.g. lm_head) and MoE router gates are
                # excluded from QAT on the verl side: their weights arrive
                # unrotated and unquantized, and the training forward runs
                # them in the original basis.  Wrapping them here would
                # rotate/quantize their inputs (with no matching weight
                # rotation) and desynchronize the computation from training.
                # Use an unquantized pass-through method so they run in the
                # original dtype (vllm-ascend's ReplicatedLinear asserts a
                # quant method is present).
                from vllm.model_executor.layers.linear import UnquantizedLinearMethod

                return UnquantizedLinearMethod()
            layer.ascend_quant_method = ASCEND_HIF8_METHOD

            scheme_cls = get_scheme_class("W8A8_HIF8", "linear")
            if scheme_cls is not None:
                scheme = scheme_cls(
                    granularity=self.granularity,
                    group_size=self.group_size,
                    rotation_enable=rotation["enable"],
                    rotation_block_size=rotation["block_size"],
                    rotation_seed=rotation["seed"],
                )
                return AscendLinearMethod(scheme)

        if _is_fused_moe_layer(layer):
            if check_equal_or_regex_match(prefix, self.ignore):
                # Mixed-precision fallback: ignored MoE blocks (e.g. sensitive
                # layers 45-47) run completely unquantized, matching the
                # training-side MoE QAT which skips the same blocks.  Use the
                # Ascend-specific unquantized method — it implements the
                # router_logits-based apply() signature that vllm-ascend's
                # FusedMoE wrapper (fused_moe_0_23_0) dispatches through.
                from vllm_ascend.ops.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod

                return AscendUnquantizedFusedMoEMethod(layer.moe_config, tid2eid=tid2eid)
            layer.ascend_quant_method = ASCEND_HIF8_METHOD
            scheme_cls = get_scheme_class("W8A8_HIF8", "moe")
            if scheme_cls is not None:
                scheme = scheme_cls(
                    granularity=self.granularity,
                    group_size=self.group_size,
                    rotation_enable=rotation["enable"],
                    rotation_block_size=rotation["block_size"],
                    rotation_seed=rotation["seed"],
                )
                return AscendFusedMoEMethod(scheme, layer.moe_config, tid2eid=tid2eid)

        logger.warning_once(
            f"[vllm-ascend/HiF8] No scheme found for layer type: {type(layer).__name__}"
        )
        return None
