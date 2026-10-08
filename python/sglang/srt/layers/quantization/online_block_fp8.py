"""Load-time block FP8 for linear layers a checkpoint ships in BF16.

The ModelOpt NVFP4 GLM-5.3-Flash checkpoint keeps the KDA/MLA projections and the
shared experts in BF16, ~19% of a 4k-token prefill step on B300. This method keeps
the BF16 weight and adds an e4m3 copy with 128x128 block scales (the layout of
GLM's own FP8 release); batches of at least ``min_tokens`` rows run the block-FP8
GEMM (DeepGEMM on SM100), smaller ones (decode, verify) the BF16 weight, where
FP8 does not pay.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.nn import Module, Parameter

from sglang.srt.layers.quantization.base_config import LinearMethodBase
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

BLOCK = 128
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def quantize_block_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] (multiples of 128) -> e4m3 [N, K] and fp32 scale_inv [N/128, K/128]."""
    n, k = weight.shape
    blocks = weight.float().view(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    scale = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12) / FP8_MAX
    q = (blocks / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX)
    return q.to(torch.float8_e4m3fn).view(n, k), scale


def can_quantize(layer: Module) -> bool:
    w = layer.weight
    return w.dim() == 2 and w.shape[0] % BLOCK == 0 and w.shape[1] % BLOCK == 0


class OnlineBlockFp8LinearMethod(LinearMethodBase):
    def __init__(self, min_tokens: int):
        self.min_tokens = min_tokens
        self._bf16 = UnquantizedLinearMethod()
        self._fp8 = Fp8LinearMethod(
            Fp8Config(
                is_checkpoint_fp8_serialized=True,
                activation_scheme="dynamic",
                weight_block_size=[BLOCK, BLOCK],
            )
        )

    def create_weights(self, layer: Module, *args, **kwargs) -> None:
        self._bf16.create_weights(layer, *args, **kwargs)

    def process_weights_after_loading(self, layer: Module) -> None:
        self._bf16.process_weights_after_loading(layer)
        if not can_quantize(layer):
            layer.fp8_block = None
            return
        q, scale = quantize_block_fp8(layer.weight.data)
        # The FP8 path reads its own holder so the BF16 weight stays layer.weight.
        holder = Module()
        holder.weight = Parameter(q, requires_grad=False)
        holder.weight_scale_inv = Parameter(scale, requires_grad=False)
        holder.input_scale = None
        holder.orig_dtype = layer.weight.dtype
        holder.logical_widths = [q.shape[0]]
        holder.input_size_per_partition = q.shape[1]
        holder.output_size_per_partition = q.shape[0]
        self._fp8.process_weights_after_loading(holder)
        layer.fp8_block = holder

    def apply(
        self,
        layer: Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if layer.fp8_block is None or x.dim() != 2 or x.shape[0] < self.min_tokens:
            return self._bf16.apply(layer, x, bias)
        return self._fp8.apply(layer.fp8_block, x, bias)
