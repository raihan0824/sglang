"""KDA prefill inside the breakable prefill CUDA graph.

By default every KDA layer of a prefill step is an eager break in the breakable
graph: ~9 Triton launches plus metadata ops per layer, ~1.2 ms of host time each,
so a small prefill step is bound by host launches. This module keeps the KDA
extend metadata in static buffers sized per token bucket, so the KDA layers can
be captured with the rest of the step.

Static layout for a bucket of ``T`` tokens (``bs_cap = min(max_bs, T)``):

* sequences ``[0, bs)`` are the real requests,
* sequence ``bs`` is a pad sequence covering the bucket padding ``[T_real, T)``,
* sequences ``(bs, bs_cap + 2)`` have zero length,

so there are always ``bs_cap + 2`` sequences and at least one empty one. The pad
sequence uses the reserved padding mamba slot, whose SSM state is zeroed before
each layer; empty sequences carry slot -1, which the kernels skip. The chunk and
conv block lists are padded to a static length with block 0 of the last (empty)
sequence, a no-op in every kernel. Prefix cache snapshots are padded to
``max_track`` copies into the padding slot.
Batches beyond the static limits run the whole step eager
(``can_run_prefill_cuda_graph``).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    CONV_FWD_BLOCK_M,
    causal_conv1d_fn,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
    from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

CHUNK_SIZE = 64


def bucket_bs_cap(max_bs: int, num_tokens: int) -> int:
    return min(max_bs, num_tokens)


def bucket_num_seqs(max_bs: int, num_tokens: int) -> int:
    return bucket_bs_cap(max_bs, num_tokens) + 2


def bucket_num_chunks(max_bs: int, num_tokens: int, chunk: int = CHUNK_SIZE) -> int:
    # Real chunks <= T_real/chunk + bs, pad chunks <= pad/chunk + 1.
    return num_tokens // chunk + bucket_bs_cap(max_bs, num_tokens) + 2


def _block_list(lens: list[int], block: int, total: int) -> list[tuple[int, int]]:
    out = [(i, t) for i, n in enumerate(lens) for t in range(math.ceil(n / block))]
    assert len(out) <= total, (len(out), total)
    return out + [(len(lens) - 1, 0)] * (total - len(out))


def build_static_layout(
    extend_seq_lens: list[int],
    num_tokens: int,
    max_bs: int,
):
    """CPU part of the static metadata: int32 ``qsl[N+1]``, ``chunk_indices[NT, 2]``,
    ``chunk_offsets[N+1]`` and conv ``block_indices[NB, 2]`` for the bucket."""
    bs = len(extend_seq_lens)
    bs_cap = bucket_bs_cap(max_bs, num_tokens)
    n_seqs = bs_cap + 2
    n_chunks = bucket_num_chunks(max_bs, num_tokens)
    t_real = sum(extend_seq_lens)
    assert bs <= bs_cap and t_real <= num_tokens, (bs, bs_cap, t_real, num_tokens)
    lens = list(extend_seq_lens) + [num_tokens - t_real] + [0] * (n_seqs - bs - 1)
    qsl = [0]
    offsets = [0]
    for n in lens:
        qsl.append(qsl[-1] + n)
        offsets.append(offsets[-1] + math.ceil(n / CHUNK_SIZE))
    chunks = _block_list(lens, CHUNK_SIZE, n_chunks)
    blocks = _block_list(
        lens, CONV_FWD_BLOCK_M, bucket_num_chunks(max_bs, num_tokens, CONV_FWD_BLOCK_M)
    )
    return (
        torch.tensor(qsl, dtype=torch.int32),
        torch.tensor(chunks, dtype=torch.int32),
        torch.tensor(offsets, dtype=torch.int32),
        torch.tensor(blocks, dtype=torch.int32),
    )


class KDAPrefillGraphState:
    """Static KDA extend metadata shared by every captured prefill bucket.

    Buffers are allocated for the largest bucket; a bucket reads fixed-size
    leading slices, so a captured graph keeps reading the same addresses.
    """

    def __init__(
        self,
        backend: KDAAttnBackend,
        max_num_tokens: int,
        max_bs: int,
        max_track: int,
    ):
        pool = backend.req_to_token_pool
        device = backend.device
        self.max_bs = max_bs
        self.max_track = max_track
        self.max_num_tokens = max_num_tokens
        self.device = device
        n_seqs = bucket_num_seqs(max_bs, max_num_tokens)
        n_chunks = bucket_num_chunks(max_bs, max_num_tokens)
        # Padded rows carry req-pool row 0 -> the reserved mamba padding slot.
        pad_slot = pool.translate_mamba_indices(
            pool.get_mamba_indices(torch.zeros(1, dtype=torch.int64, device=device))
        )
        self.pad_slot = int(pad_slot.item())
        idx_dtype = pad_slot.dtype
        conv_len = backend.conv_states_shape[-1]

        self.qsl = torch.zeros(n_seqs + 1, dtype=torch.int32, device=device)
        self.cache_indices = torch.full(
            (n_seqs,), self.pad_slot, dtype=idx_dtype, device=device
        )
        self.has_initial_state = torch.zeros(n_seqs, dtype=torch.bool, device=device)
        self.chunk_indices = torch.zeros(n_chunks, 2, dtype=torch.int32, device=device)
        self.chunk_offsets = torch.zeros(n_seqs + 1, dtype=torch.int32, device=device)
        self.conv_blocks = torch.zeros(
            bucket_num_chunks(max_bs, max_num_tokens, CONV_FWD_BLOCK_M),
            2,
            dtype=torch.int32,
            device=device,
        )
        self.track_chunk_idx = torch.full(
            (n_seqs,), -1, dtype=torch.int32, device=device
        )
        self.row_mask = torch.zeros(max_num_tokens, dtype=torch.bfloat16, device=device)
        self.conv_dst = torch.full(
            (max_track,), self.pad_slot, dtype=torch.int64, device=device
        )
        self.conv_tok = torch.zeros(max_track, conv_len, dtype=torch.int64, device=device)
        self.h_dst = torch.full((max_track,), self.pad_slot, dtype=torch.int64, device=device)
        self.h_src = torch.full((max_track,), n_seqs - 1, dtype=torch.int64, device=device)
        self.final_src = torch.full(
            (max_track,), self.pad_slot, dtype=torch.int64, device=device
        )
        self.final_dst = torch.full(
            (max_track,), self.pad_slot, dtype=torch.int64, device=device
        )
        ssm_shape = pool.mamba_pool.mamba_cache.temporal.shape[2:]
        # fp32 snapshot rows the chunk kernel writes for tracked sequences.
        self.h_track_buf = torch.empty(
            (n_seqs, *ssm_shape), dtype=torch.float32, device=device
        )
        self.num_fallbacks = 0
        logger.info(
            "KDA prefill in graph: max_bs=%d max_track=%d, %d seqs / %d chunks "
            "at %d tokens, track buffer %.0f MB",
            max_bs,
            max_track,
            n_seqs,
            n_chunks,
            max_num_tokens,
            self.h_track_buf.numel() * 4 / 2**20,
        )

    # ------------------------------------------------------------------ host side
    def can_run(self, forward_batch: ForwardBatch) -> bool:
        lens = forward_batch.extend_seq_lens_cpu
        prefix = forward_batch.extend_prefix_lens_cpu
        bs = forward_batch.batch_size
        ok = (
            lens is not None
            and prefix is not None
            and len(lens) == bs
            and bs <= self.max_bs
            and sum(lens) <= self.max_num_tokens
        )
        if ok and forward_batch.mamba_track_mask is not None:
            track_cpu = forward_batch.mamba_prefill_track_mask_cpu
            ok = (
                track_cpu is not None
                and len(track_cpu) == bs
                and sum(1 for t in track_cpu if t) <= self.max_track
            )
        if not ok:
            self.num_fallbacks += 1
            if self.num_fallbacks in (1, 10, 100) or self.num_fallbacks % 1000 == 0:
                logger.info(
                    "KDA prefill in graph: eager step #%d (bs=%d)",
                    self.num_fallbacks,
                    bs,
                )
        return ok

    def fill(self, forward_batch: ForwardBatch, metadata, num_tokens: int) -> None:
        """Copy this step's metadata into the static buffers for bucket
        ``num_tokens``. Runs eagerly before the graph replays (stream-ordered)."""
        bs = forward_batch.batch_size
        lens = list(forward_batch.extend_seq_lens_cpu)
        prefix = forward_batch.extend_prefix_lens_cpu
        qsl, chunks, offsets, blocks = build_static_layout(
            lens, num_tokens, self.max_bs
        )
        n_seqs = qsl.shape[0] - 1
        t_real = sum(lens)
        nb = True
        self.qsl[: n_seqs + 1].copy_(qsl, non_blocking=nb)
        self.chunk_indices[: chunks.shape[0]].copy_(chunks, non_blocking=nb)
        self.chunk_offsets[: n_seqs + 1].copy_(offsets, non_blocking=nb)
        self.conv_blocks[: blocks.shape[0]].copy_(blocks, non_blocking=nb)
        self.has_initial_state[:n_seqs].copy_(
            torch.tensor([p > 0 for p in prefix] + [False] * (n_seqs - bs)),
            non_blocking=nb,
        )
        self.cache_indices[:bs].copy_(metadata.mamba_cache_indices[:bs])
        self.cache_indices[bs].fill_(self.pad_slot)
        self.cache_indices[bs + 1 : n_seqs].fill_(-1)
        self.row_mask[:t_real].fill_(1)
        self.row_mask[t_real:num_tokens].zero_()

        self.track_chunk_idx[:n_seqs].fill_(-1)
        n_conv = n_h = n_fin = 0
        if metadata.has_mamba_track_mask:
            if metadata.track_chunk_idx is not None:
                self.track_chunk_idx[:bs].copy_(metadata.track_chunk_idx[:bs])
            if metadata.track_conv_indices is not None:
                n_conv = metadata.track_conv_indices.shape[0]
                self.conv_tok[:n_conv].copy_(metadata.track_conv_indices)
                self.conv_dst[:n_conv].copy_(metadata.conv_states_mask_indices)
            if metadata.track_ssm_h_src is not None:
                n_h = metadata.track_ssm_h_src.shape[0]
                if n_h:
                    self.h_dst[:n_h].copy_(metadata.track_ssm_h_dst)
                    self.h_src[:n_h].copy_(metadata.track_ssm_h_batch_src)
            if metadata.track_ssm_final_src is not None:
                n_fin = metadata.track_ssm_final_src.shape[0]
                if n_fin:
                    self.final_src[:n_fin].copy_(metadata.track_ssm_final_src)
                    self.final_dst[:n_fin].copy_(metadata.track_ssm_final_dst)
        assert max(n_conv, n_h, n_fin) <= self.max_track
        self.conv_tok[n_conv:].zero_()
        self.conv_dst[n_conv:].fill_(self.pad_slot)
        self.h_dst[n_h:].fill_(self.pad_slot)
        self.h_src[n_h:].fill_(n_seqs - 1)
        self.final_src[n_fin:].fill_(self.pad_slot)
        self.final_dst[n_fin:].fill_(self.pad_slot)

    # ---------------------------------------------------------------- device side
    def forward_extend(
        self,
        backend: KDAAttnBackend,
        layer: RadixLinearAttention,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """KDA extend over the whole bucket with static shapes; writes ``output``
        (zero on padding rows). Captured into the prefill graph."""
        num_tokens = mixed_qkv.shape[0]
        n_seqs = bucket_num_seqs(self.max_bs, num_tokens)
        n_chunks = bucket_num_chunks(self.max_bs, num_tokens)
        qsl = self.qsl[: n_seqs + 1]
        cache_indices = self.cache_indices[:n_seqs]

        layer_cache = backend.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_pool = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        # The pad and empty sequences start from the padding slot's state.
        ssm_states[self.pad_slot].zero_()
        # Prefix-cache conv snapshot (padded copies land in the padding slot).
        conv_pool[self.conv_dst] = mixed_qkv[self.conv_tok]

        qkv = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),
            layer.conv_weights,
            layer.bias,
            conv_states=conv_pool.transpose(-1, -2),
            query_start_loc=qsl,
            # The Triton conv only sizes its grid from these.
            seq_lens_cpu=[num_tokens] + [0] * (n_seqs - 1),
            cache_indices=cache_indices,
            has_initial_state=self.has_initial_state[:n_seqs],
            activation="silu",
            block_indices=self.conv_blocks[
                : bucket_num_chunks(self.max_bs, num_tokens, CONV_FWD_BLOCK_M)
            ],
        ).transpose(0, 1)
        q, k, v = qkv.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)
        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)
        gate_was_flat = a.ndim == 3
        if gate_was_flat:
            a = a.unflatten(-1, (-1, layer.head_k_dim))

        track_buf = self.h_track_buf[:n_seqs]
        core_attn_out = backend.kernel_dispatcher.triton_kernel.extend(
            q,
            k,
            v,
            a,
            b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=qsl,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            lower_bound=layer.lower_bound,
            beta_is_raw=gate_was_flat,
            return_intermediate_states=False,
            track_state=track_buf,
            track_chunk_idx=self.track_chunk_idx[:n_seqs],
            chunk_indices=self.chunk_indices[:n_chunks],
            chunk_offsets=self.chunk_offsets[: n_seqs + 1],
        )
        # Prefix-cache SSM snapshots: unaligned rows from the fp32 chunk-boundary
        # buffer, aligned rows from the final state.
        ssm_states[self.h_dst] = track_buf[self.h_src].to(ssm_states.dtype)
        ssm_states[self.final_dst] = ssm_states[self.final_src]
        torch.mul(
            core_attn_out,
            self.row_mask[:num_tokens].view(1, num_tokens, 1, 1),
            out=output,
        )


def maybe_create_prefill_graph_state(
    backend: KDAAttnBackend, max_num_tokens: Optional[int]
) -> Optional[KDAPrefillGraphState]:
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel

    if not envs.SGLANG_OPT_KDA_PREFILL_IN_GRAPH.get() or not max_num_tokens:
        return None
    reason = None
    if not isinstance(backend.kernel_dispatcher.extend_kernel, TritonKDAKernel):
        reason = "the prefill kernel is not Triton"
    elif backend.accept_lens_pool is not None:
        reason = "fused-accept staging is on"
    if reason is not None:
        logger.warning("KDA prefill in graph disabled: %s", reason)
        return None
    return KDAPrefillGraphState(
        backend,
        max_num_tokens=max_num_tokens,
        max_bs=envs.SGLANG_OPT_KDA_PREFILL_IN_GRAPH_MAX_BS.get(),
        max_track=envs.SGLANG_OPT_KDA_PREFILL_IN_GRAPH_MAX_TRACK.get(),
    )
