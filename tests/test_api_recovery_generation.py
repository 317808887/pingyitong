"""In-flight requests cannot overwrite a newer connection recovery."""

from __future__ import annotations

import threading
from unittest import TestCase, mock

from pingyitong.components.llm.api_engine import ApiEngine, ApiRequestError
from pingyitong.config.schema import ApiConfig


class TestApiRecoveryGeneration(TestCase):
    def setUp(self) -> None:
        self.engine = ApiEngine(
            ApiConfig(enabled=True, base_url="http://example.test/v1", model_name="m")
        )
        self.addCleanup(self.engine.unload)

    def test_old_failures_do_not_cool_down_recovered_engine(self) -> None:
        started = threading.Barrier(4)
        release = threading.Event()

        def chat(*, system: str, user: str, max_tokens: int) -> str:
            if user == "Translate to Chinese: OK":
                return "好的"
            started.wait(timeout=5)
            if not release.wait(timeout=5):
                raise RuntimeError("Old request was not released")
            raise ApiRequestError(503, "Old request failed")

        with mock.patch.object(self.engine, "_chat", side_effect=chat):
            threads = [
                threading.Thread(
                    target=self.engine.translate_batch,
                    args=([f"Old request {index}"], "Translate"),
                )
                for index in range(3)
            ]
            for thread in threads:
                thread.start()
            try:
                started.wait(timeout=5)
                before = self.engine.failure_generation
                ok, message = self.engine.test()
                self.assertTrue(ok, message)
                recovered = self.engine.failure_generation
                self.assertGreater(recovered, before)
                self.assertTrue(self.engine.is_available())
            finally:
                release.set()
                for thread in threads:
                    thread.join(timeout=5)
                    self.assertFalse(thread.is_alive())

        self.assertTrue(self.engine.is_available())
        self.assertEqual(self.engine.failure_generation, recovered)
        self.assertEqual(self.engine.status().total_calls, 4)

    def test_old_success_does_not_clear_new_generation_auth_failure(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def chat(*, system: str, user: str, max_tokens: int) -> str:
            if user == "Old request":
                started.set()
                if not release.wait(timeout=5):
                    raise RuntimeError("Old request was not released")
                return "旧译文"
            if user == "Translate to Chinese: OK":
                return "好的"
            raise ApiRequestError(401, "New request rejected")

        with mock.patch.object(self.engine, "_chat", side_effect=chat):
            thread = threading.Thread(
                target=self.engine.translate_batch, args=(["Old request"], "Translate")
            )
            thread.start()
            try:
                self.assertTrue(started.wait(timeout=5))
                ok, message = self.engine.test()
                self.assertTrue(ok, message)
                self.assertEqual(self.engine.translate_batch(["New request"], "Translate"), [""])
                failed = self.engine.failure_generation
                self.assertFalse(self.engine.is_available())
            finally:
                release.set()
                thread.join(timeout=5)
                self.assertFalse(thread.is_alive())

        self.assertFalse(self.engine.is_available())
        self.assertIn("401", self.engine.status().message)
        self.assertEqual(self.engine.failure_generation, failed)
        self.assertEqual(self.engine.status().total_calls, 3)

    def test_old_failed_probe_does_not_undo_new_successful_probe(self) -> None:
        started = threading.Event()
        release = threading.Event()
        old_result: list[tuple[bool, str]] = []

        def chat(*, system: str, user: str, max_tokens: int) -> str:
            if threading.current_thread().name == "OldProbe":
                started.set()
                if not release.wait(timeout=5):
                    raise RuntimeError("Old probe was not released")
                raise ApiRequestError(401, "Old probe rejected")
            return "好的"

        with mock.patch.object(self.engine, "_chat", side_effect=chat):
            thread = threading.Thread(
                target=lambda: old_result.append(self.engine.test()), name="OldProbe"
            )
            thread.start()
            try:
                self.assertTrue(started.wait(timeout=5))
                ok, message = self.engine.test()
                self.assertTrue(ok, message)
                recovered = self.engine.failure_generation
            finally:
                release.set()
                thread.join(timeout=5)
                self.assertFalse(thread.is_alive())

        self.assertFalse(old_result[0][0])
        self.assertTrue(self.engine.is_available())
        self.assertEqual(self.engine.failure_generation, recovered)
        self.assertEqual(self.engine.status().total_calls, 2)
