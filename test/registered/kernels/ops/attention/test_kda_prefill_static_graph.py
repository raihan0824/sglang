"""KDA prefill in the breakable prefill graph: a padded static-shape extend, captured
once and replayed with other batch layouts, must match the plain varlen extend on
the real rows, real state slots and prefix-cache snapshots, and leave every other
slot except the padding slot untouched."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.fla.l2norm import (
    kda_prefill_qkv_l2norm_prepare,
    l2norm_fwd,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn
from sglang.srt.layers.attention.linear.kda_prefill_graph import (
    KDAPrefillGraphState,
    build_static_layout,
    bucket_num_seqs,
)
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

H, D, CONV, SLOTS, PAD_SLOT = 32, 128, 4, 24, 0
QKV = 3 * H * D


class _Pool:
    def __init__(self, gen):
        self.conv = [
            torch.randn(SLOTS, CONV - 1, QKV, generator=gen, device="cuda").to(
                torch.bfloat16
            )
        ]
        self.temporal = 0.1 * torch.randn(
            SLOTS, H, D, D, generator=gen, device="cuda", dtype=torch.float32
        )

    def clone(self):
        c = object.__new__(_Pool)
        c.conv = [self.conv[0].clone()]
        c.temporal = self.temporal.clone()
        return c


def _fake_backend(pool, max_tokens, max_bs, max_track):
    req_pool = SimpleNamespace(
        get_mamba_indices=lambda r: torch.full_like(r, PAD_SLOT),
        translate_mamba_indices=lambda x: x,
        mamba_pool=SimpleNamespace(
            mamba_cache=SimpleNamespace(temporal=pool.temporal.unsqueeze(0))
        ),
        mamba2_layer_cache=lambda layer_id: pool,
    )
    backend = SimpleNamespace(
        req_to_token_pool=req_pool,
        device="cuda",
        conv_states_shape=(SLOTS, QKV, CONV - 1),
        kernel_dispatcher=SimpleNamespace(triton_kernel=TritonKDAKernel()),
    )
    state = KDAPrefillGraphState(backend, max_tokens, max_bs, max_track)
    return backend, state


def _layer(gen):
    return SimpleNamespace(
        layer_id=0,
        conv_weights=0.3
        * torch.randn(QKV, CONV, generator=gen, device="cuda").to(torch.bfloat16),
        bias=None,
        q_dim=H * D,
        k_dim=H * D,
        v_dim=H * D,
        head_q_dim=D,
        head_k_dim=D,
        head_v_dim=D,
        A_log=torch.randn(H, generator=gen, device="cuda"),
        dt_bias=torch.randn(H * D, generator=gen, device="cuda"),
        lower_bound=-5.0,
    )


def _inputs(gen, num_tokens):
    mixed = torch.randn(num_tokens, QKV, generator=gen, device="cuda").to(
        torch.bfloat16
    )
    a = torch.randn(1, num_tokens, H * D, generator=gen, device="cuda").to(
        torch.bfloat16
    )
    b = torch.randn(1, num_tokens, H, generator=gen, device="cuda").to(torch.bfloat16)
    return mixed, a, b


def _reference(pool, layer, mixed, a, b, lens, prefix, slots, track_chunk):
    """The eager varlen extend over the real tokens only."""
    t = sum(lens)
    qsl = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device="cuda").to(
        torch.int32
    )
    idx = torch.tensor(slots, device="cuda", dtype=torch.int32)
    qkv = causal_conv1d_fn(
        mixed[:t].transpose(0, 1),
        layer.conv_weights,
        None,
        conv_states=pool.conv[0].transpose(-1, -2),
        query_start_loc=qsl,
        seq_lens_cpu=list(lens),
        cache_indices=idx,
        has_initial_state=torch.tensor([p > 0 for p in prefix], device="cuda"),
        activation="silu",
    ).transpose(0, 1)
    q, k, v = (
        x.unflatten(-1, (-1, D)).unsqueeze(0) for x in qkv.split([H * D] * 3, dim=-1)
    )
    track = torch.empty(len(lens), H, D, D, device="cuda", dtype=torch.float32)
    out = TritonKDAKernel().extend(
        q,
        k,
        v,
        a[:, :t].unflatten(-1, (-1, D)),
        b[:, :t],
        ssm_states=pool.temporal,
        cache_indices=idx,
        query_start_loc=qsl,
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        lower_bound=-5.0,
        beta_is_raw=True,
        track_state=track,
        track_chunk_idx=torch.tensor(track_chunk, device="cuda", dtype=torch.int32),
    )
    return out, track


class TestKdaPrefillStaticGraph(CustomTestCase):
    def test_layout(self):
        qsl, chunks, offsets, blocks = build_static_layout([100, 37, 700], 1024, max_bs=8)
        n = bucket_num_seqs(8, 1024)
        self.assertEqual(qsl.tolist()[:6], [0, 100, 137, 837, 1024, 1024])
        self.assertEqual(qsl.shape[0], n + 1)
        self.assertEqual(offsets.tolist()[:5], [0, 2, 3, 14, 17])
        # Padding chunks and conv blocks are block 0 of the last, empty sequence.
        self.assertEqual(chunks[-1].tolist(), [n - 1, 0])
        self.assertEqual(blocks[-1].tolist(), [n - 1, 0])
        self.assertEqual(blocks[12].tolist(), [0, 12])  # ceil(100 / 8) blocks for seq 0
        self.assertEqual(blocks[13].tolist(), [1, 0])
        self.assertEqual(int(qsl[-1] - qsl[-2]), 0)

    @torch.inference_mode()
    def test_packed_qkv_prepare_is_bitwise_l2norm(self):
        """The fused prepare replaces l2norm_fwd(q.contiguous()) etc. in chunk_kda;
        any difference would change every KDA prefill output."""
        if not torch.cuda.is_available():
            self.skipTest("needs CUDA")
        gen = torch.Generator(device="cuda").manual_seed(3)
        packed = torch.randn(777, QKV, generator=gen, device="cuda").to(torch.bfloat16)
        q, k, v = (x.unflatten(-1, (-1, D)) for x in packed.split([H * D] * 3, dim=-1))
        qn, kn, vc = kda_prefill_qkv_l2norm_prepare(q, k, v)
        self.assertTrue(torch.equal(qn, l2norm_fwd(q.contiguous())))
        self.assertTrue(torch.equal(kn, l2norm_fwd(k.contiguous())))
        self.assertTrue(torch.equal(vc, v.contiguous()))

    @torch.inference_mode()
    def test_capture_once_replay_layouts(self):
        if not torch.cuda.is_available():
            self.skipTest("needs CUDA")
        gen = torch.Generator(device="cuda").manual_seed(0)
        bucket, max_bs = 1024, 8
        layer = _layer(gen)
        base_pool = _Pool(gen)
        pool = base_pool.clone()
        backend, state = _fake_backend(pool, bucket, max_bs, max_track=4)
        mixed_s, a_s, b_s = _inputs(gen, bucket)
        out_s = torch.empty(1, bucket, H, D, device="cuda", dtype=torch.bfloat16)

        # (lens, prefix lens, slots, track chunk index per request)
        layouts = [
            ([1000], [0], [3], [-1]),
            ([100, 37, 700], [5, 0, 64], [4, 7, 9], [1, -1, 10]),
            ([1, 1, 1, 1], [3, 0, 2, 9], [11, 12, 13, 14], [-1, -1, -1, -1]),
            ([64, 128, 63, 65, 500], [0, 1, 0, 0, 7], [1, 2, 5, 6, 8], [0, 1, -1, 0, 7]),
        ]

        def fill(lens, prefix, slots, track_chunk):
            fb = SimpleNamespace(
                batch_size=len(lens),
                extend_seq_lens_cpu=lens,
                extend_prefix_lens_cpu=prefix,
            )
            md = SimpleNamespace(
                mamba_cache_indices=torch.tensor(slots, device="cuda", dtype=torch.int64),
                has_mamba_track_mask=True,
                track_chunk_idx=torch.tensor(track_chunk, device="cuda", dtype=torch.int32),
                track_conv_indices=None,
                track_ssm_h_src=None,
                track_ssm_final_src=None,
            )
            state.fill(fb, md, bucket)

        def run():
            state.forward_extend(
                backend=backend, layer=layer, mixed_qkv=mixed_s, a=a_s, b=b_s, output=out_s
            )

        fill(*layouts[0])
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            run()  # warm up Triton autotune outside capture
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()

        for i, (lens, prefix, slots, track_chunk) in enumerate(layouts):
            g = torch.Generator(device="cuda").manual_seed(100 + i)
            mixed, a, b = _inputs(g, bucket)
            for sl, p in zip(slots, prefix):
                if p == 0:  # new requests start from a zeroed slot
                    base_pool.temporal[sl].zero_()
            ref_pool = base_pool.clone()
            ref_out, ref_track = _reference(
                ref_pool, layer, mixed, a, b, lens, prefix, slots, track_chunk
            )
            pool.conv[0].copy_(base_pool.conv[0])
            pool.temporal.copy_(base_pool.temporal)
            mixed_s.copy_(mixed)
            a_s.copy_(a)
            b_s.copy_(b)
            fill(lens, prefix, slots, track_chunk)
            graph.replay()
            torch.cuda.synchronize()
            t = sum(lens)
            torch.testing.assert_close(
                out_s[:, :t].float(), ref_out.float(), rtol=2e-2, atol=2e-2
            )
            self.assertTrue(torch.isfinite(out_s[:, t:]).all())
            others = [x for x in range(SLOTS) if x not in slots and x != PAD_SLOT]
            torch.testing.assert_close(
                pool.temporal[slots], ref_pool.temporal[slots], rtol=2e-2, atol=2e-2
            )
            torch.testing.assert_close(
                pool.conv[0][slots], ref_pool.conv[0][slots], rtol=0, atol=0
            )
            self.assertTrue(torch.equal(pool.temporal[others], base_pool.temporal[others]))
            self.assertTrue(torch.equal(pool.conv[0][others], base_pool.conv[0][others]))
            for row, c in enumerate(track_chunk):
                if c >= 0:
                    torch.testing.assert_close(
                        state.h_track_buf[row], ref_track[row], rtol=2e-2, atol=2e-2
                    )


if __name__ == "__main__":
    unittest.main()
