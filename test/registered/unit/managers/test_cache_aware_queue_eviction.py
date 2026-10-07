"""Cache-aware queue-full eviction (SGLANG_ENABLE_CACHE_AWARE_QUEUE_EVICTION).

When the waiting queue is full, the scheduler refuses the request with the
smallest share of its prompt in the prefix cache, the arriving one or one
already waiting, so a follow-up turn is not refused in favor of a cold prompt.
The pick is bookkeeping over the queue and the radix tree -- no model, no GPU --
so it is driven here directly.
"""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle, InsertParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _FakeReq:
    def __init__(self, rid, *, output_ids=(), to_finish=None, holds_kv=False):
        self.rid = rid
        self.cache_request_handle = CacheRequestHandle(rid, 0)
        self.to_finish = to_finish
        self.output_ids = list(output_ids)
        self.session = None
        self.kv = SimpleNamespace(holds_kv=holds_kv, holds_mamba=False)
        self.beam_group = None
        self.weight_version_events = []
        self.time_stats = SimpleNamespace(
            wait_queue_entry_time=0.0, trace_ctx=MagicMock()
        )


def _scheduler(waiting_queue, *, max_queued, tree_cache=None, shares=None):
    s = Scheduler.__new__(Scheduler)
    s.disaggregation_mode = DisaggregationMode.NULL
    s.waiting_queue = list(waiting_queue)
    s.max_queued_requests = max_queued
    s.enable_priority_scheduling = False
    s.schedule_low_priority_values_first = False
    s.enable_hierarchical_cache = False
    s.enable_hicache_storage = False
    s.enable_unified_cache_external_linker = False
    s.ipc_channels = SimpleNamespace(send_to_tokenizer=MagicMock())
    s.beam_coordinator = MagicMock()
    s.metrics_collector = MagicMock()
    s.tree_cache = tree_cache if tree_cache is not None else MagicMock()
    if shares is not None:
        s._cached_prompt_share = MagicMock(side_effect=lambda req: shares[req.rid])
    return s


def _refused(s):
    """(rid, status, message) of every request answered with an abort."""
    return [
        (
            call.args[0].rid,
            call.args[0].finished_reason["status_code"],
            call.args[0].finished_reason["message"],
        )
        for call in s.ipc_channels.send_to_tokenizer.send_output.call_args_list
    ]


class _PatchedServing(CustomTestCase):
    def setUp(self):
        patcher = patch(
            "sglang.srt.managers.scheduler.get_serving",
            return_value=SimpleNamespace(weight_version="v0"),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _abort_on_queued_limit(self, s, arriving, *, enabled=True):
        with envs.SGLANG_ENABLE_CACHE_AWARE_QUEUE_EVICTION.override(enabled):
            return s._abort_on_queued_limit(arriving)


class TestPick(_PatchedServing):
    def test_follow_up_displaces_the_cold_waiting_request(self):
        shares = {"cold": 0.0, "warm": 0.9, "follow_up": 0.97}
        s = _scheduler(
            [_FakeReq("cold"), _FakeReq("warm")], max_queued=2, shares=shares
        )

        arriving_refused = self._abort_on_queued_limit(s, _FakeReq("follow_up"))

        self.assertFalse(arriving_refused)
        self.assertEqual([r.rid for r in s.waiting_queue], ["warm"])
        [(rid, status, message)] = _refused(s)
        self.assertEqual((rid, status.value), ("cold", 429))
        self.assertIn("more of its prompt cached", message)
        s.beam_coordinator.retire_group.assert_called_once()
        s.metrics_collector.increment_rejected_requests.assert_called_once_with(
            reason="less_cached"
        )

    def test_cold_arrival_is_refused_without_matching_the_queue(self):
        shares = {"a": 0.0, "b": 0.0, "cold": 0.1}
        s = _scheduler([_FakeReq("a"), _FakeReq("b")], max_queued=2, shares=shares)

        self.assertTrue(self._abort_on_queued_limit(s, _FakeReq("cold")))

        self.assertEqual([r.rid for r in s.waiting_queue], ["a", "b"])
        self.assertEqual([r[0] for r in _refused(s)], ["cold"])
        self.assertEqual(s._cached_prompt_share.call_count, 1)
        s.metrics_collector.increment_rejected_requests.assert_called_once_with(
            reason="queue_full"
        )

    def test_gap_below_the_minimum_keeps_the_waiting_request(self):
        shares = {"a": 0.8, "b": 0.9, "follow_up": 0.99}
        s = _scheduler([_FakeReq("a"), _FakeReq("b")], max_queued=2, shares=shares)
        self.assertTrue(self._abort_on_queued_limit(s, _FakeReq("follow_up")))
        self.assertEqual(len(s.waiting_queue), 2)

        s = _scheduler([_FakeReq("a"), _FakeReq("b")], max_queued=2, shares=shares)
        with envs.SGLANG_CACHE_AWARE_QUEUE_EVICTION_MIN_GAP.override(0.1):
            self.assertFalse(self._abort_on_queued_limit(s, _FakeReq("follow_up")))
        self.assertEqual([r[0] for r in _refused(s)], ["a"])

    def test_tie_refuses_the_latest_queued(self):
        shares = {"first": 0.0, "second": 0.0, "follow_up": 0.95}
        s = _scheduler(
            [_FakeReq("first"), _FakeReq("second")], max_queued=2, shares=shares
        )
        self.assertFalse(self._abort_on_queued_limit(s, _FakeReq("follow_up")))
        self.assertEqual([r.rid for r in s.waiting_queue], ["first"])

    def test_started_requests_are_never_displaced(self):
        started = [
            _FakeReq("streamed", output_ids=[7]),
            _FakeReq("pending_error", to_finish=object()),
            _FakeReq("holds_kv", holds_kv=True),
        ]
        shares = {"follow_up": 0.99}
        s = _scheduler(started, max_queued=3, shares=shares)

        self.assertTrue(self._abort_on_queued_limit(s, _FakeReq("follow_up")))

        self.assertEqual(len(s.waiting_queue), 3)
        self.assertEqual([r[0] for r in _refused(s)], ["follow_up"])
        self.assertEqual(s._cached_prompt_share.call_count, 1)

    def test_arriving_request_with_a_pending_error_is_refused_as_before(self):
        s = _scheduler([_FakeReq("cold")], max_queued=1, shares={"cold": 0.0})
        arriving = _FakeReq("bad", to_finish=object())
        self.assertTrue(self._abort_on_queued_limit(s, arriving))
        s._cached_prompt_share.assert_not_called()

    def test_off_by_default(self):
        self.assertFalse(envs.SGLANG_ENABLE_CACHE_AWARE_QUEUE_EVICTION.get())
        shares = {"cold": 0.0, "follow_up": 0.99}
        s = _scheduler([_FakeReq("cold")], max_queued=1, shares=shares)
        self.assertTrue(
            self._abort_on_queued_limit(s, _FakeReq("follow_up"), enabled=False)
        )
        self.assertEqual([r.rid for r in s.waiting_queue], ["cold"])
        s._cached_prompt_share.assert_not_called()

    def test_queue_with_room_refuses_nothing(self):
        s = _scheduler([_FakeReq("cold")], max_queued=2, shares={})
        self.assertFalse(self._abort_on_queued_limit(s, _FakeReq("follow_up")))
        s.ipc_channels.send_to_tokenizer.send_output.assert_not_called()

    def test_priority_scheduling_keeps_its_own_eviction(self):
        cold = _FakeReq("cold")
        cold.priority = 0
        follow_up = _FakeReq("follow_up")
        follow_up.priority = 0
        s = _scheduler([cold], max_queued=1, shares={})
        s.enable_priority_scheduling = True
        self.assertTrue(self._abort_on_queued_limit(s, follow_up))
        s._cached_prompt_share.assert_not_called()


def _req(rid, ids):
    sampling_params = SamplingParams(max_new_tokens=1)
    sampling_params.normalize(None)
    return Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", ids),
        sampling_params=sampling_params,
        vocab_size=100_000,
    )


class TestWithRadixTree(_PatchedServing):
    """Real Req objects matched against a radix tree holding one earlier turn."""

    TURN = list(range(1, 801))

    def setUp(self):
        super().setUp()
        self.tree = RadixCache.create_simulated()
        self.tree.insert(
            InsertParams(
                key=RadixKey(array("q", self.TURN)),
                value=torch.empty(len(self.TURN), dtype=torch.bool),
            )
        )

    def test_share_counts_the_cached_turn(self):
        s = _scheduler([], max_queued=1, tree_cache=self.tree)
        follow_up = _req("follow_up", self.TURN + list(range(2001, 2201)))
        cold = _req("cold", list(range(5001, 5501)))
        self.assertAlmostEqual(s._cached_prompt_share(follow_up), 0.8)
        self.assertEqual(s._cached_prompt_share(cold), 0.0)

    def test_follow_up_takes_the_place_of_a_cold_prompt(self):
        cold_a = _req("cold_a", list(range(5001, 5501)))
        cold_b = _req("cold_b", list(range(6001, 6501)))
        s = _scheduler([cold_a, cold_b], max_queued=2, tree_cache=self.tree)
        follow_up = _req("follow_up", self.TURN + list(range(2001, 2101)))

        self.assertFalse(self._abort_on_queued_limit(s, follow_up))

        self.assertEqual([r.rid for r in s.waiting_queue], ["cold_a"])
        self.assertEqual([r[0] for r in _refused(s)], ["cold_b"])

    def test_cold_prompt_still_refused_when_the_queue_is_warm(self):
        warm = _req("warm", self.TURN + list(range(3001, 3101)))
        s = _scheduler([warm], max_queued=1, tree_cache=self.tree)
        self.assertTrue(
            self._abort_on_queued_limit(s, _req("cold", list(range(5001, 5501))))
        )
        self.assertEqual([r.rid for r in s.waiting_queue], ["warm"])


if __name__ == "__main__":
    unittest.main()
