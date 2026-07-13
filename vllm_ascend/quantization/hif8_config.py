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

"""HiF8 quantization config for vLLM Ascend (per-element, native).

Registers the "ascend-hif8" quantization method that routes to the
W8A8_HIF8 scheme for linear and MoE layers.

Config JSON format:
{
    "quant_method": "ascend-hif8",
    "ignore": ["lm_head", "embed_tokens"]
}
"""

from typing import Any, Optional, cast

import torch
from vllm.logger import logger
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase

from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.23.0"):
    from vllm.model_executor.layers.fused_moe import FusedMoE
else:
    from vllm.model_executor.layers.fused_moe import MoERunner

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
    """Quantization config for Ascend HiF8 (W8A8_HIF8, per-element native).

    HiF8 is Huawei's native 8-bit floating point format. Each element
    is independently quantized — no external scales, no blocks.
    """

    def __init__(
        self,
        ignore: list[str],
        config: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.ignore = ignore
        self.quant_description = config if config is not None else {}

    def __repr__(self) -> str:
        return "AscendHiF8Config(per-element native HiF8)"

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

        return cls(
            ignore=ignore,
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

        if isinstance(layer, LinearBase):
            layer.ascend_quant_method = ASCEND_HIF8_METHOD

            scheme_cls = get_scheme_class("W8A8_HIF8", "linear")
            if scheme_cls is not None:
                scheme = scheme_cls()
                return AscendLinearMethod(scheme)

        if _is_fused_moe_layer(layer):
            layer.ascend_quant_method = ASCEND_HIF8_METHOD

            scheme_cls = get_scheme_class("W8A8_HIF8", "moe")
            if scheme_cls is not None:
                scheme = scheme_cls()
                return AscendFusedMoEMethod(scheme, layer.moe_config, tid2eid=tid2eid)

        logger.warning_once(
            f"[vllm-ascend/HiF8] No scheme found for layer type: {type(layer).__name__}"
        )
        return None
