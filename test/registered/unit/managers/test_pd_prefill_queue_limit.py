"""--max-queued-requests on a disaggregated (PD) prefill server.

The limit used to apply only in NULL mode, so a PD deployment queued without bound and never answered 429. On a
prefill server it now counts every request not yet handed to a decode server: the waiting queue, the bootstrap queue
(waiting for a decode server to take the request) and the inflight queue (KV still being sent).
"""

import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestOutcome
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestPrefillServerQueueLimit(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def _scheduler(self, mode, max_queued, waiting=0, bootstrap=0, inflight=0, priority=False):
        s = Scheduler.__new__(Scheduler)
        s.disaggregation_mode = mode
        s.enable_priority_scheduling = priority
        s.schedule_low_priority_values_first = False
        s.abort_on_priority_when_disabled = False
        s.enable_hierarchical_cache = False
        s.enable_hicache_storage = False
        s.enable_unified_cache_external_linker = False
        s.processed_tokens_counter = 0
        s.max_queued_requests = max_queued
        s.waiting_queue = [self._req(f"w{i}") for i in range(waiting)]
        s._prefetch_kvcache = MagicMock()
        s.tree_cache = MagicMock(spec=["finish"])
        s.model_config = SimpleNamespace(num_key_value_heads=1)
        s.disagg_prefill_bootstrap_queue = MagicMock()
        s.disagg_prefill_bootstrap_queue.queue = [self._req(f"b{i}") for i in range(bootstrap)]
        s.disagg_prefill_inflight_queue = [self._req(f"i{i}") for i in range(inflight)]
        s.disagg_decode_prealloc_queue = MagicMock()
        s.ipc_channels = MagicMock()
        s.metrics_collector = MagicMock()
        s.beam_coordinator = MagicMock()
        return s

    def _req(self, rid="incoming", priority=None):
        req = MagicMock()
        req.rid = rid
        req.priority = priority
        req.output_ids = []
        req.weight_version_events = []
        req.time_stats = MagicMock()
        req.time_stats.trace_ctx = MagicMock()
        return req

    def _add(self, s, req):
        with patch(
            "sglang.srt.managers.scheduler.get_serving",
            return_value=SimpleNamespace(weight_version="v0"),
        ):
            s._add_request_to_queue(req)

    def _sent_abort(self, s):
        abort_req = s.ipc_channels.send_to_tokenizer.send_output.call_args.args[0]
        return abort_req.finished_reason

    def test_full_prefill_server_answers_429_before_bootstrap(self):
        # 1 waiting + 1 in bootstrap + 1 sending KV = 3 = the limit.
        s = self._scheduler(DisaggregationMode.PREFILL, max_queued=3, waiting=1, bootstrap=1, inflight=1)
        incoming = self._req()
        self._add(s, incoming)
        s.disagg_prefill_bootstrap_queue.add.assert_not_called()
        s._prefetch_kvcache.assert_not_called()
        s.tree_cache.finish.assert_called_once_with(incoming.cache_request_handle, CacheRequestOutcome.ABORT)
        reason = self._sent_abort(s)
        self.assertEqual(reason["status_code"], HTTPStatus.TOO_MANY_REQUESTS)
        s.metrics_collector.increment_rejected_requests.assert_called_once_with(reason="queue_full")

    def test_prefill_server_below_the_limit_admits(self):
        s = self._scheduler(DisaggregationMode.PREFILL, max_queued=4, waiting=1, bootstrap=1, inflight=1)
        incoming = self._req()
        self._add(s, incoming)
        s.disagg_prefill_bootstrap_queue.add.assert_called_once_with(incoming, 1)
        s.ipc_channels.send_to_tokenizer.send_output.assert_not_called()

    def test_each_prefill_queue_counts(self):
        for waiting, bootstrap, inflight in ((2, 0, 0), (0, 2, 0), (0, 0, 2)):
            with self.subTest(waiting=waiting, bootstrap=bootstrap, inflight=inflight):
                s = self._scheduler(DisaggregationMode.PREFILL, 2, waiting, bootstrap, inflight)
                self._add(s, self._req())
                s.disagg_prefill_bootstrap_queue.add.assert_not_called()

    def test_priority_scheduling_never_evicts_on_a_prefill_server(self):
        # The waiting queue is empty, the bootstrap queue is full: reject the arrival, do not touch the queues.
        s = self._scheduler(DisaggregationMode.PREFILL, max_queued=1, bootstrap=1, priority=True)
        incoming = self._req(priority=100)
        self._add(s, incoming)
        self.assertEqual(len(s.disagg_prefill_bootstrap_queue.queue), 1)
        s.disagg_prefill_bootstrap_queue.add.assert_not_called()
        self.assertEqual(self._sent_abort(s)["status_code"], HTTPStatus.TOO_MANY_REQUESTS)

    def test_no_limit_admits_everything(self):
        s = self._scheduler(DisaggregationMode.PREFILL, max_queued=None, waiting=50, bootstrap=50, inflight=50)
        incoming = self._req()
        self._add(s, incoming)
        s.disagg_prefill_bootstrap_queue.add.assert_called_once()

    def test_decode_server_is_not_limited(self):
        s = self._scheduler(DisaggregationMode.DECODE, max_queued=1, waiting=5)
        incoming = self._req()
        self._add(s, incoming)
        s.disagg_decode_prealloc_queue.add.assert_called_once_with(incoming, is_retracted=False)
        s.ipc_channels.send_to_tokenizer.send_output.assert_not_called()

    def test_colocated_server_counts_only_its_waiting_queue(self):
        # NULL mode keeps its old behaviour; the PD queues are not consulted.
        s = self._scheduler(DisaggregationMode.NULL, max_queued=2, waiting=1, bootstrap=5, inflight=5)
        incoming = self._req()
        self._add(s, incoming)
        self.assertIn(incoming, s.waiting_queue)


if __name__ == "__main__":
    unittest.main()
