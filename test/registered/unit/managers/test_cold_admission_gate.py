"""Cold-admission reserve (SGLANG_COLD_ADMISSION_RESERVE_SLOTS).

While a replica has at most N free running slots (queued requests count as
taken), an arriving request with no more than MAX_CACHED_SHARE of its prompt in
the prefix cache is refused with 429 at once, so the last slots go to requests
that continue conversations cached on this replica. Bookkeeping over the batch,
the queue and the cached share only -- no model, no GPU -- so it is driven here
directly.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _FakeReq:
    def __init__(self, rid, *, output_ids=(), to_finish=None, holds_kv=False):
        self.rid = rid
        self.to_finish = to_finish
        self.output_ids = list(output_ids)
        self.session = None
        self.kv = SimpleNamespace(holds_kv=holds_kv, holds_mamba=False)
        self.beam_group = None
        self.weight_version_events = []
        self.time_stats = SimpleNamespace(
            wait_queue_entry_time=0.0, trace_ctx=MagicMock()
        )


def _scheduler(*, max_running, running, waiting, shares):
    s = Scheduler.__new__(Scheduler)
    s.max_running_requests = max_running
    s.running_batch = SimpleNamespace(reqs=[_FakeReq(f"r{i}") for i in range(running)])
    s.waiting_queue = [_FakeReq(f"w{i}") for i in range(waiting)]
    s.ipc_channels = SimpleNamespace(send_to_tokenizer=MagicMock())
    s.metrics_collector = MagicMock()
    s._cached_prompt_share = MagicMock(side_effect=lambda req: shares[req.rid])
    return s


def _refused(s):
    return [
        (call.args[0].rid, call.args[0].finished_reason["status_code"].value)
        for call in s.ipc_channels.send_to_tokenizer.send_output.call_args_list
    ]


class TestColdAdmissionGate(CustomTestCase):
    def setUp(self):
        patcher = patch(
            "sglang.srt.managers.scheduler.get_serving",
            return_value=SimpleNamespace(weight_version="v0"),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _gate(self, s, req, *, reserve=2, max_share=0.05):
        with envs.SGLANG_COLD_ADMISSION_RESERVE_SLOTS.override(reserve):
            with envs.SGLANG_COLD_ADMISSION_MAX_CACHED_SHARE.override(max_share):
                return s._abort_cold_when_busy(req)

    def test_cold_request_is_refused_when_the_last_slots_are_left(self):
        s = _scheduler(max_running=64, running=62, waiting=0, shares={"cold": 0.0})

        self.assertTrue(self._gate(s, _FakeReq("cold")))

        self.assertEqual(_refused(s), [("cold", 429)])
        s.metrics_collector.increment_rejected_requests.assert_called_once_with(
            reason="cold_when_busy"
        )

    def test_warm_request_takes_a_reserved_slot(self):
        s = _scheduler(max_running=64, running=63, waiting=0, shares={"warm": 0.9})

        self.assertFalse(self._gate(s, _FakeReq("warm")))

        self.assertEqual(_refused(s), [])

    def test_cold_request_is_admitted_while_slots_are_plentiful(self):
        s = _scheduler(max_running=64, running=40, waiting=0, shares={"cold": 0.0})

        self.assertFalse(self._gate(s, _FakeReq("cold")))

        s._cached_prompt_share.assert_not_called()
        self.assertEqual(_refused(s), [])

    def test_queued_requests_count_as_taken_slots(self):
        s = _scheduler(max_running=64, running=59, waiting=3, shares={"cold": 0.0})

        self.assertTrue(self._gate(s, _FakeReq("cold")))

    def test_threshold_is_inclusive(self):
        s = _scheduler(max_running=64, running=63, waiting=0, shares={"a": 0.05, "b": 0.06})

        self.assertTrue(self._gate(s, _FakeReq("a")))
        self.assertFalse(self._gate(s, _FakeReq("b")))

    def test_off_by_default(self):
        s = _scheduler(max_running=64, running=64, waiting=4, shares={"cold": 0.0})

        self.assertFalse(self._gate(s, _FakeReq("cold"), reserve=0))

        s._cached_prompt_share.assert_not_called()

    def test_started_requests_are_never_refused(self):
        s = _scheduler(max_running=64, running=63, waiting=0, shares={"retracted": 0.0})

        self.assertFalse(self._gate(s, _FakeReq("retracted", output_ids=[1, 2])))

        s._cached_prompt_share.assert_not_called()


if __name__ == "__main__":
    unittest.main()
