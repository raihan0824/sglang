"""Load-time block FP8 for BF16 checkpoint linears: the e4m3 copy must follow the
128x128 block layout (its scale is each block's amax / 448), and which layers get
it is a decision the ModelOpt NVFP4 config makes from the module prefix."""

import pytest
import torch

from sglang.srt.layers.quantization.modelopt_quant import _is_online_fp8_projection
from sglang.srt.layers.quantization.online_block_fp8 import (
    FP8_MAX,
    quantize_block_fp8,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_block_scales_follow_128x128_amax():
    torch.manual_seed(0)
    w = torch.randn(256, 384) * 0.02
    w[0, 0] = 3.0  # one outlier must only coarsen its own block
    q, scale = quantize_block_fp8(w.to(torch.bfloat16))
    assert q.dtype == torch.float8_e4m3fn and scale.shape == (2, 3)
    assert torch.isclose(scale[0, 0], torch.tensor(3.0) / FP8_MAX, rtol=1e-2)
    assert scale[1, 2] < scale[0, 0] / 20
    deq = q.float().view(2, 128, 3, 128) * scale[:, None, :, None]
    err = (deq.view(256, 384) - w.to(torch.bfloat16).float()).abs()
    # e4m3 keeps 3 mantissa bits: half an ulp of the block's largest value bounds it.
    bound = (scale * FP8_MAX / 16).repeat_interleave(128, 0).repeat_interleave(128, 1)
    assert (err <= bound + 1e-6).all()


@pytest.mark.parametrize(
    "prefix, expected",
    [
        ("model.language_model.layers.4.self_attn.fused_qkvbfg_a_proj", True),
        ("model.language_model.layers.3.self_attn.o_proj", True),
        ("model.language_model.layers.5.mlp.shared_experts.gate_up_proj", True),
        ("model.language_model.layers.3.self_attn.indexer.wk", False),
        ("model.language_model.layers.5.mlp.gate", False),
        ("model.visual.blocks.0.attn.qkv", False),
        ("lm_head", False),
    ],
)
def test_online_fp8_layer_selection(prefix, expected):
    assert _is_online_fp8_projection(prefix) is expected
