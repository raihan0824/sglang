# Design: verify-preserving mixed step for DFLASH (SGLang v0.5.19)

Goal: when a prefill chunk is scheduled, running requests keep their draft block (verify) instead of
degrading to a 1-token decode (today's `--enable-mixed-chunk`), so no decode work is lost while prompts
are read. Expected: recovers most of the decode throughput lost during prefill steps (~25% of GPU time on
the standard mix at rate 8, more with 32k prompts).

## Existing plumbing (v0.5.19)
- Scheduler mixed branch (scheduler.py ~3808): `new_batch.mix_with_running(running_batch)` converts running
  rows to 1-token extends (prefix = committed len, extend_len 1), `input_ids=None`, sets
  `mix_running_indices`; input ids are gathered from `future_map.output_tokens_buf` at forward entry
  (overlap_utils.resolve_forward_inputs); under overlap the tails are re-bound from the published
  `new_seq_lens_buf` (resolve_mixed_spec_tails).
- Result: `process_batch_result_prefill` treats `decoding_reqs` as 1 new token (`kv_committed_len += 1`).
- DFLASH worker extend path: target forward with FULL hidden capture, draft KV append for all extend
  tokens, `next_draft_input` = bonus tokens for every row.
- GDN backend: `forward_extend` (chunk kernel, state advanced in place) vs `forward_decode`/verify (ring
  write, `disable_state_update`, commit replay). No mixed split; Mamba2 (`layers/attention/mamba/mamba.py`)
  has one (prefill segment + decode/verify segment) to copy.

## Changes
1. ScheduleBatch.mix_with_running (spec + flag): running rows get extend_len = block, prefix = committed
   len; out_cache_loc for the block = the spec reservation (`DFlashDraftInputV2.prepare_for_decode`
   already reserved 2*block per row); `input_ids` of the tail rows are filled by the worker (draft tokens),
   so `resolve_forward_inputs` must skip the tail gather when `batch.mixed_spec_block` is set. Keep the
   running batch's `spec_info` (bonus tokens, nxt_kv_lens) reachable from the merged batch.
2. DFLASH worker: new path when `batch.forward_mode.is_mixed() and batch.mixed_spec_block`:
   a. run the draft for the tail rows exactly as in decode (bonus from spec_info / future map) ->
      draft_tokens [n, block], verify locs, ring locs;
   b. build ONE ForwardBatch(EXTEND, capture_hidden_mode=FULL): input_ids = [prefill ids, draft block
      ids], positions from prefix lens, out_cache_loc = [prefill locs, verify locs],
      `mixed_spec_rows = n` marker for the GDN backend;
   c. logits: prefill rows -> `next_token_logits` (last positions) as today; tail rows -> gather the
      block hidden states from `hidden_states` (FULL) and apply the target `lm_head` (n*block x vocab,
      ~0.2 ms) -> `_accept_block` (existing) -> commit_lens, out_tokens, bonus, new_seq_lens;
   d. mamba commit for the tail rows only (`update_mamba_state_after_mtp_verify` with the tail's slot
      indices: forward_metadata.mamba_cache_indices[-n:]);
   e. draft KV: prefill rows -> window append (ring); tail rows -> accepted-block append (ring);
   f. result: prefill rows as an extend result (next token), tail rows as a spec result (accept_lens,
      out_tokens stride block); `next_draft_input` bonus = [sampled tokens, accept bonus], new_seq_lens
      combined.
3. GDN backend mixed split (gdn_backend.forward_extend, when `forward_batch.mixed_spec_rows > 0`):
   rows [0, bs-n) -> chunk kernel path (as now, with cu_seqlens of the prefill rows only);
   rows [bs-n, bs) -> verify path: conv `causal_conv1d_update` with `intermediate_conv_window`
   (cache_steps=block) and the ring-writing recurrent kernel (`_replayssm_fold_target_verify`,
   disable_state_update) on the [n*block] token slice. Mamba2Metadata.prepare_mixed already carries
   num_prefills/num_decodes; extend it with `draft_token_num=block, is_target_verify=True` for the tail.
4. Result processor: `process_batch_result_prefill` with `result.mixed_spec`: tail reqs use
   `accept_lens`/`next_token_ids` (stride block) like the spec-v2 decode path (`_resolve_spec_v2_tokens`
   on the tail slice), `kv_committed_len += accept_len`; prefill reqs unchanged. Overlap relay: publish
   the tail rows' new_seq_lens to `new_seq_lens_buf` (already keyed by req_pool_indices).
5. Guards: flag `--enable-mixed-chunk` + DFLASH + linear chain; fall back to the 1-token mixed step for
   grammar / logprob / retracted rows in the tail.

## Risks
- Full-attention prefill backend (FA3/FA4 extend) with block rows: positions/out_cache_loc are explicit, so
  ragged extend covers it; mask is causal, which is exactly the DFLASH verify mask.
- Overlap late-binding (resolve_mixed_spec_tails) must rebuild block locs from the published lengths.
- Prefill graph runner: mixed-spec batches run eager (they already do for MIXED).

Effort: 2-3 days incl. debugging on the shared GPU. Measure on sweep (standard mix) and long32k.
