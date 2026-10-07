import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
    resolve_system_prompt_checkpoint_token_id,
    system_prompt_checkpoints,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    DecLockRefResult,
    IncLockRefResult,
)
from sglang.srt.mem_cache.prefill_budget import PrefillBudget
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils.common import Range
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

USER = 7  # stands in for the <｜User｜> token id
PAGE = 256
CHUNK = 8192


def prompt(system_len: int, user_len: int, system_seed: int = 1, user_seed: int = 2):
    """Token ids of a chat prompt: a system block, then the user turn token, then the turn."""
    system = [1000 + (system_seed * 7919 + i) % 50000 for i in range(system_len)]
    turn = [1000 + (user_seed * 104729 + i) % 50000 for i in range(user_len)]
    return system + [USER] + turn


class TestSystemPromptCheckpoints(CustomTestCase):
    def test_early_point_and_block_end(self):
        ids = prompt(5000, 600)
        self.assertEqual(system_prompt_checkpoints(ids, USER, PAGE, 2048), (2048, 4864))

    def test_array_ids(self):
        ids = array("q", prompt(5000, 600))
        self.assertEqual(system_prompt_checkpoints(ids, USER, PAGE, 2048), (2048, 4864))

    def test_short_block_keeps_only_the_end(self):
        self.assertEqual(system_prompt_checkpoints(prompt(1000, 50), USER, PAGE, 2048), (768,))
        # early point and end on the same page: one checkpoint
        self.assertEqual(system_prompt_checkpoints(prompt(2100, 50), USER, PAGE, 2048), (2048,))

    def test_early_point_off(self):
        self.assertEqual(system_prompt_checkpoints(prompt(5000, 600), USER, PAGE, 0), (4864,))

    def test_no_system_block(self):
        self.assertEqual(system_prompt_checkpoints(prompt(3, 900), USER, PAGE, 2048), ())
        self.assertEqual(system_prompt_checkpoints(list(range(10, 5000)), USER, PAGE, 2048), ())
        self.assertEqual(system_prompt_checkpoints([], USER, PAGE, 2048), ())

    def test_block_past_scan_bound(self):
        self.assertEqual(system_prompt_checkpoints(prompt(70000, 10), USER, PAGE, 2048), ())

    def test_ids_without_index_method(self):
        ids = np.array(prompt(5000, 600))
        self.assertEqual(system_prompt_checkpoints(ids, USER, PAGE, 2048), ())


class TestResolveToken(CustomTestCase):
    def tokenizer(self, token_id, unk=0):
        tok = MagicMock()
        tok.convert_tokens_to_ids.return_value = token_id
        tok.unk_token_id = unk
        return tok

    def test_off_by_default(self):
        self.assertIsNone(resolve_system_prompt_checkpoint_token_id(self.tokenizer(128803)))

    def test_on(self):
        with envs.SGLANG_ENABLE_SYSTEM_PROMPT_CHECKPOINT.override(True):
            self.assertEqual(resolve_system_prompt_checkpoint_token_id(self.tokenizer(128803)), 128803)
            self.assertIsNone(resolve_system_prompt_checkpoint_token_id(self.tokenizer(0, unk=0)))
            self.assertIsNone(resolve_system_prompt_checkpoint_token_id(None))
            broken = MagicMock()
            broken.convert_tokens_to_ids.side_effect = RuntimeError("no vocab")
            self.assertIsNone(resolve_system_prompt_checkpoint_token_id(broken))


class _AdderFixture(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        self.tree_cache = MagicMock()
        for name in ("full_evictable_size", "swa_evictable_size", "evictable_size"):
            getattr(self.tree_cache, name).return_value = 0
        self.tree_cache.disable = False
        self.tree_cache.is_tree_cache.return_value = False
        self.tree_cache.sliding_window_size = 128
        self.tree_cache.inc_lock_ref.return_value = IncLockRefResult()
        self.tree_cache.dec_lock_ref.return_value = DecLockRefResult()
        self.tree_cache.buffer_pipeline = None
        self.tree_cache.supports_mamba.return_value = False
        self.allocator = MagicMock()
        self.allocator.swa_req_ring = False
        self.allocator.page_size = PAGE
        self.allocator.create_prefill_budget.side_effect = lambda tree_cache, **kwargs: (
            PrefillBudget(self.allocator, tree_cache, **kwargs)
        )
        for name in ("full_available_size", "swa_available_size", "available_size"):
            getattr(self.allocator, name).return_value = 10_000_000
        self.allocator.size_swa = 10_000_000

    def adder(self, token_id=USER, hybrid_swa=False):
        batch = MagicMock()
        batch.reqs = []
        adder = PrefillAdder(
            page_size=PAGE,
            tree_cache=self.tree_cache,
            token_to_kv_pool_allocator=self.allocator,
            running_batch=batch,
            new_token_ratio=1.0,
            rem_input_tokens=16384,
            rem_chunk_tokens=CHUNK,
            system_prompt_checkpoint_token_id=token_id,
        )
        adder.is_hybrid_swa = hybrid_swa
        return adder

    def req(self, rid, ids, prefix_len=0, max_new_tokens=256):
        req = MagicMock(spec=Req)
        req.rid = rid
        req.cache_request_handle = CacheRequestHandle(rid, 0)
        req.finished.return_value = False
        req.origin_input_ids = ids
        req.full_untruncated_fill_ids = list(ids)
        req.prefix_indices = list(range(prefix_len))
        req.system_prompt_checkpoints = None
        req.output_ids = []
        req.sampling_params = SimpleNamespace(max_new_tokens=max_new_tokens, ignore_eos=False)
        req.retracted_stain = False
        req.host_hit_length = 0
        req.swa_host_hit_length = 0
        req.storage_hit_length = 0
        req.storage_hit_start = None
        req.host_hit_is_storage = False
        req.host_loaded_length = 0
        req.materialized_host_hit_len.return_value = 0
        req.fulfilled_storage_hit_len.return_value = 0
        req.needs_host_load_back.return_value = False
        req.last_node = MagicMock()
        req.set_extend_range = MagicMock(
            side_effect=lambda start, end: setattr(req, "extend_range", Range(start, end))
        )
        return req

    def add(self, adder, req, has_chunked_req=False):
        return adder.add_one_req(req, has_chunked_req=has_chunked_req, truncation_align_size=None)


class TestAdderCheckpoints(_AdderFixture):
    def test_cold_request_stops_at_the_early_point_and_closes_the_batch(self):
        adder = self.adder()
        a = self.req("a", prompt(5000, 600))
        self.assertIs(self.add(adder, a), AddReqResult.OTHER)
        self.assertEqual((a.extend_range.start, a.extend_range.end), (0, 2048))
        self.assertIs(adder.new_chunked_req, a)

    def test_hybrid_swa_path_cuts_the_same_way(self):
        adder = self.adder(hybrid_swa=True)
        a = self.req("a", prompt(5000, 600))
        self.assertIs(self.add(adder, a), AddReqResult.OTHER)
        self.assertEqual(a.extend_range.end, 2048)

    def test_continuing_chunk_stops_at_the_block_end_and_closes_the_batch(self):
        adder = self.adder()
        a = self.req("a", prompt(5000, 600), prefix_len=2048)
        self.assertIs(adder.add_chunked_req(a), a)  # still chunked
        self.assertEqual((a.extend_range.start, a.extend_range.end), (2048, 4864))
        self.assertEqual(adder.rem_chunk_tokens, 0)
        # nothing else joins this batch, so no second chunked request appears
        other = self.req("other", prompt(9000, 400, system_seed=3))
        self.assertIs(self.add(adder, other, has_chunked_req=True), AddReqResult.OTHER)
        self.assertNotIn(other, adder.can_run_list)
        self.assertIsNone(adder.new_chunked_req)

    def test_last_chunk_after_the_block_end_finishes_the_prefill(self):
        adder = self.adder()
        a = self.req("a", prompt(5000, 600), prefix_len=4864)
        self.assertIsNone(adder.add_chunked_req(a))
        self.assertEqual(a.extend_range.end, len(a.full_untruncated_fill_ids))

    def test_request_cached_to_the_block_end_is_not_cut(self):
        adder = self.adder()
        b = self.req("b", prompt(5000, 400, user_seed=9), prefix_len=4864)
        self.assertIs(self.add(adder, b), AddReqResult.CONTINUE)
        self.assertEqual(b.extend_range.end, len(b.full_untruncated_fill_ids))
        self.assertIsNone(adder.new_chunked_req)

    def test_request_cached_to_the_early_point_is_cut_at_the_block_end(self):
        adder = self.adder()
        b = self.req("b", prompt(5000, 400, user_seed=9), prefix_len=2048)
        self.assertIs(self.add(adder, b), AddReqResult.OTHER)
        self.assertEqual((b.extend_range.start, b.extend_range.end), (2048, 4864))

    def test_checkpoint_in_the_last_page_is_skipped(self):
        # block end 4864 but the prompt ends 100 tokens later: only the early point cuts
        adder = self.adder()
        a = self.req("a", prompt(5000, 100))
        self.assertIs(self.add(adder, a), AddReqResult.OTHER)
        self.assertEqual(a.extend_range.end, 2048)
        adder = self.adder()
        a2 = self.req("a2", prompt(5000, 100), prefix_len=2048)
        self.assertIsNone(adder.add_chunked_req(a2))  # the rest in one chunk
        self.assertEqual(a2.extend_range.end, len(a2.full_untruncated_fill_ids))

    def test_no_cut_while_another_request_is_chunked(self):
        adder = self.adder()
        a = self.req("a", prompt(5000, 600))
        result = self.add(adder, a, has_chunked_req=True)
        self.assertEqual(a.extend_range.end, len(a.full_untruncated_fill_ids))
        self.assertIs(result, AddReqResult.CONTINUE)

    def test_no_cut_after_a_new_chunked_request_in_the_batch(self):
        adder = self.adder()
        long_req = self.req("long", prompt(3, 20000))  # no system block, chunked by size
        self.add(adder, long_req)
        self.assertIs(adder.new_chunked_req, long_req)
        a = self.req("a", prompt(5000, 600))
        self.add(adder, a)
        self.assertIs(adder.new_chunked_req, long_req)

    def test_feature_off_keeps_the_old_behaviour(self):
        adder = self.adder(token_id=None)
        a = self.req("a", prompt(5000, 600))
        self.assertIs(self.add(adder, a), AddReqResult.CONTINUE)
        self.assertEqual(a.extend_range.end, len(a.full_untruncated_fill_ids))
        self.assertIsNone(adder.new_chunked_req)

    def test_long_block_cuts_inside_later_chunks(self):
        # 12k-token block: 2048, then a full chunk, then the block end, then the rest
        ids = prompt(12000, 3000)
        adder = self.adder()
        a = self.req("a", ids)
        self.add(adder, a)
        self.assertEqual(a.extend_range.end, 2048)
        ends = [2048]
        while True:
            adder = self.adder()
            a.prefix_indices = list(range(ends[-1]))
            done = adder.add_chunked_req(a) is None
            ends.append(a.extend_range.end)
            if done:
                break
        self.assertEqual(ends, [2048, 2048 + CHUNK, 11776, len(ids)])


class _BoundaryCache:
    """Prefix cache that, like the SWA radix cache, can resume only at positions where
    an earlier chunk or request ended (page-aligned)."""

    def __init__(self):
        self.prefixes = set()

    def insert(self, ids, end):
        end = end // PAGE * PAGE
        if end > 0:
            self.prefixes.add(tuple(ids[:end]))

    def match(self, ids):
        best = 0
        for end in range(PAGE, len(ids) + 1, PAGE):
            if tuple(ids[:end]) in self.prefixes:
                best = end
        return best


class TestSchedulerLoop(_AdderFixture):
    """The scheduler's admission loop over a mixed queue: one chunked request at a time,
    every request finishes its prefill, and a new conversation with a known system
    prompt starts from the cached block end."""

    def run_loop(self, reqs, token_id=USER, max_iters=200):
        cache = _BoundaryCache()
        waiting = list(reqs)
        chunked = None
        first_prefix = {}
        for _ in range(max_iters):
            if not waiting and chunked is None:
                return cache, first_prefix
            adder = self.adder(token_id=token_id)
            if chunked is not None:
                chunked = adder.add_chunked_req(chunked)
            for req in waiting:
                if req.rid not in first_prefix:
                    req.prefix_indices = list(range(cache.match(req.full_untruncated_fill_ids)))
                res = adder.add_one_req(
                    req, has_chunked_req=chunked is not None, truncation_align_size=None
                )
                if res is not AddReqResult.CONTINUE:
                    break
            if adder.new_chunked_req is not None:
                self.assertIsNone(chunked, "a second chunked request")
                chunked = adder.new_chunked_req
            for req in adder.can_run_list:
                first_prefix.setdefault(req.rid, req.extend_range.start)
                self.assertGreater(req.extend_range.length, 0)
                cache.insert(req.full_untruncated_fill_ids, req.extend_range.end)
                req.prefix_indices = list(range(req.extend_range.end))
            waiting = [r for r in waiting if r not in adder.can_run_list]
        self.fail("the loop did not drain")

    def test_mixed_queue(self):
        a = self.req("a", prompt(5000, 600, user_seed=1))
        b = self.req("b", prompt(5000, 400, user_seed=2))  # same system prompt as a
        c = self.req("c", prompt(12000, 3000, system_seed=5))
        d = self.req("d", prompt(3, 20000, user_seed=4))  # no system block, long
        e = self.req("e", prompt(1500, 50, system_seed=6))  # short block
        f = self.req("f", prompt(12000, 800, system_seed=5, user_seed=8))  # same block as c
        cache, first_prefix = self.run_loop([a, b, c, d, e, f])
        for r in (a, b, c, d, e, f):
            self.assertEqual(r.prefix_indices, list(range(len(r.full_untruncated_fill_ids))))
        self.assertEqual(first_prefix["b"], 4864)
        self.assertEqual(first_prefix["f"], 11776)

    def test_same_queue_without_checkpoints(self):
        a = self.req("a", prompt(5000, 600, user_seed=1))
        b = self.req("b", prompt(5000, 400, user_seed=2))
        cache, first_prefix = self.run_loop([a, b], token_id=None)
        self.assertEqual(first_prefix["b"], 0)  # today's miss


if __name__ == "__main__":
    unittest.main()
