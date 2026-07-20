import asyncio
import json
import os
import re
import unittest

import aiohttp

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    STDERR_FILENAME,
    STDOUT_FILENAME,
    CustomTestCase,
    popen_launch_server,
    send_concurrent_generate_requests_detailed,
    send_generate_requests,
)

register_cuda_ci(est_time=70, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=90, suite="stage-b-test-1-gpu-small-amd")


class TestMaxQueuedRequests(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.stdout = open(STDOUT_FILENAME, "w")
        cls.stderr = open(STDERR_FILENAME, "w")

        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                "--max-running-requests",  # Enforce max request concurrency is 1
                "1",
                "--max-queued-requests",  # Enforce max queued request number is 1
                "1",
                "--attention-backend",
                "triton",
            ),
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.stdout.close()
        cls.stderr.close()
        os.remove(STDOUT_FILENAME)
        os.remove(STDERR_FILENAME)

    def test_max_queued_requests_validation_with_serial_requests(self):
        """Verify request is not throttled when the max concurrency is 1."""
        status_codes = send_generate_requests(
            self.base_url,
            num_requests=10,
        )

        for status_code in status_codes:
            assert status_code == 200  # request shouldn't be throttled

    def test_max_queued_requests_validation_with_concurrent_requests(self):
        """Verify queue-full requests are throttled with a well-formed HTTP 429."""
        results = asyncio.run(
            send_concurrent_generate_requests_detailed(self.base_url, num_requests=10)
        )
        status_codes = [status for status, _, _ in results]
        self.assertLessEqual(status_codes.count(200), 2)
        self.assertIn(429, status_codes)

        # expected_status_codes = [200, 200, 429, 429, 429, 429, 429, 429, 429, 429]
        for status, headers, body in results:
            if status == 200:
                continue
            self.assertEqual(status, 429)
            # Retry-After: 0 => retry immediately / route elsewhere, don't back off.
            self.assertEqual(headers.get("Retry-After"), "0")
            self.assertEqual(body["type"], "rate_limit_error")
            self.assertEqual(body["code"], 429)

    def test_max_queued_requests_validation_with_concurrent_stream_requests(self):
        """A streaming queue-reject must be a real 429, not HTTP 200 + an SSE frame.

        Routers such as OpenRouter grade on the HTTP status, so the reject has to land
        before the stream commits to 200. Uses /v1/chat/completions: that handler
        pre-pulls the first chunk, which is what makes the pre-200 429 possible. Plain
        /generate streaming has no pre-pull and still emits an in-band frame.
        """

        async def run():
            async def one(session):
                async with session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "user",
                                "content": "What is the capital of France?",
                            }
                        ],
                        "stream": True,
                        "temperature": 0,
                        "max_tokens": 500,
                    },
                ) as response:
                    return (
                        response.status,
                        dict(response.headers),
                        await response.text(),
                    )

            async with aiohttp.ClientSession() as session:
                return await asyncio.gather(*[one(session) for _ in range(10)])

        results = asyncio.run(run())
        status_codes = [status for status, _, _ in results]
        self.assertIn(429, status_codes)

        for status, headers, text in results:
            if status == 200:
                # A 200 must be a genuine stream, never a smuggled-in 429 frame.
                self.assertNotIn('"code": 429', text)
                continue
            self.assertEqual(status, 429)
            self.assertEqual(headers.get("Retry-After"), "0")
            body = json.loads(text)
            self.assertEqual(body["type"], "rate_limit_error")
            self.assertEqual(body["code"], 429)

    def test_max_running_requests_and_max_queued_request_validation(self):
        """Verify running request and queued request numbers based on server logs."""
        rr_pattern = re.compile(r"#running-req:\s*(\d+)")
        qr_pattern = re.compile(r"#queue-req:\s*(\d+)")

        with open(STDERR_FILENAME) as lines:
            for line in lines:
                rr_match, qr_match = rr_pattern.search(line), qr_pattern.search(line)
                if rr_match:
                    assert int(rr_match.group(1)) <= 1
                if qr_match:
                    assert int(qr_match.group(1)) <= 1


if __name__ == "__main__":
    unittest.main()
