"""P2 尾测试：活跃轨注册表、连播推荐、优先级调度器。

对应方案 v1.2 §3.5 / §3.6。关键约束：
- 推荐只建议、不静默切换（D4）；
- 连播只在文件名序列**严格连续**时动作；
- 调度器空闲/网关/队列深度任一不满足即不出队；
- 取消：整页取消对未来入队也生效。
"""

from __future__ import annotations

import threading
import unittest

from pingyitong.services.prefetch.active_track import (
    ActiveTrackRegistry,
    SubtitleTrack,
    next_episode_file,
    similar_title_score,
)
from pingyitong.services.prefetch.scheduler import (
    PrefetchJob,
    PrefetchScheduler,
    Priority,
)


class TestEpisodeMatching(unittest.TestCase):
    """连播：文件名序列严格连续才动作。"""

    def test_sxxexx_next(self) -> None:
        self.assertEqual(
            next_episode_file("Movie.S01E01.1080p.mkv",
                              ["Movie.S01E03.mkv", "Movie.S01E02.mkv"]),
            "Movie.S01E02.mkv",
        )

    def test_chinese_episode_mark(self) -> None:
        self.assertEqual(
            next_episode_file("动画 第3集.mp4", ["动画 第5集.mp4", "动画 第4集.srt"]),
            "动画 第4集.srt",
        )

    def test_gap_is_not_next(self) -> None:
        """跳集（E01 → E03）不算连播，宁可返回 None。"""
        self.assertIsNone(
            next_episode_file("Show.S01E01.mkv", ["Show.S01E03.mkv"])
        )

    def test_no_episode_mark(self) -> None:
        self.assertIsNone(
            next_episode_file("movie.mkv", ["movie2.mkv"])
        )


class TestSimilarity(unittest.TestCase):
    def test_same_title_high_score(self) -> None:
        score = similar_title_score(
            "Movie.S01E01.zh.srt", "Movie.S01E01.1080p.mkv"
        )
        self.assertGreaterEqual(score, 0.5)

    def test_different_title_low_score(self) -> None:
        score = similar_title_score(
            "Another.Show.S01E01.srt", "Movie.S01E01.1080p.mkv"
        )
        self.assertLess(score, 0.5)


class TestActiveTrackRegistry(unittest.TestCase):
    def _track(self, track_id: str, title: str, line_count: int = 100) -> SubtitleTrack:
        return SubtitleTrack(track_id=track_id, path=f"/x/{title}",
                             title=title, line_count=line_count, lang="en")

    def test_first_register_becomes_active(self) -> None:
        registry = ActiveTrackRegistry()
        registry.register(self._track("t1", "A.srt"))
        self.assertEqual(registry.active_id, "t1")
        registry.register(self._track("t2", "B.srt"))
        self.assertEqual(registry.active_id, "t1", "后续注册不得抢占活跃轨")

    def test_unregister_active_clears_active(self) -> None:
        registry = ActiveTrackRegistry()
        registry.register(self._track("t1", "A.srt"))
        registry.unregister("t1")
        self.assertIsNone(registry.active_id)

    def test_manual_switch_requires_existing(self) -> None:
        registry = ActiveTrackRegistry()
        registry.register(self._track("t1", "A.srt"))
        self.assertFalse(registry.set_active("不存在"))
        self.assertTrue(registry.set_active("t1"))

    def test_suggest_respects_episode_and_similarity(self) -> None:
        registry = ActiveTrackRegistry()
        registry.register(self._track("a", "Movie.S01E01.zh.srt"))
        registry.register(self._track("b", "Other.S01E01.srt"))
        best = registry.suggest_for_video("Movie.S01E01.1080p.mkv")
        self.assertIsNotNone(best)
        self.assertEqual(best.track_id, "a")

    def test_suggest_never_switches_silently(self) -> None:
        """D4：推荐只建议，不改变活跃态。"""
        registry = ActiveTrackRegistry()
        registry.register(self._track("a", "A.srt"))
        registry.register(self._track("b", "Movie.S01E01.zh.srt"))
        best = registry.suggest_for_video("Movie.S01E01.1080p.mkv")
        self.assertIsNotNone(best)
        self.assertEqual(registry.active_id, "a", "推荐不得静默切换活跃轨")

    def test_auto_advance_off_by_default(self) -> None:
        registry = ActiveTrackRegistry()
        registry.register(self._track("a", "Show.S01E01.srt"))
        registry.register(self._track("b", "Show.S01E02.srt"))
        self.assertIsNone(
            registry.auto_advance("Show.S01E01.mkv", ["Show.S01E02.mkv"]),
            "自动连播默认关闭",
        )
        self.assertEqual(registry.active_id, "a")


class TestScheduler(unittest.TestCase):
    def _job(self, priority=Priority.BULK_REST, page_key="page1",
             kind="subtitle_line", payload=0):
        return PrefetchJob(
            priority=priority, seq=0, kind=kind,
            track_or_site=page_key, payload=payload, page_key=page_key,
        )

    def test_priority_order(self) -> None:
        """低优先级号先出队（播放当前位置 > 余量）。"""
        order: list[int] = []
        scheduler = PrefetchScheduler(
            idle_check=lambda: True,
            gate_allow=lambda source: True,
            pull=lambda job: order.append(job.priority) or True,
        )
        scheduler.enqueue(self._job(Priority.BULK_REST))
        scheduler.enqueue(self._job(Priority.PLAYBACK_CURRENT))
        scheduler._drain_one()
        scheduler._drain_one()
        self.assertEqual(order, [Priority.PLAYBACK_CURRENT, Priority.BULK_REST])

    def test_idle_gate_blocks(self) -> None:
        """非空闲时不出队（R1：不得抢占实时链路）。"""
        executed: list = []
        scheduler = PrefetchScheduler(
            idle_check=lambda: False,
            gate_allow=lambda source: True,
            pull=lambda job: executed.append(job) or True,
        )
        scheduler.enqueue(self._job())
        self.assertFalse(scheduler._drain_one())
        self.assertEqual(executed, [])

    def test_gate_blocks_per_source(self) -> None:
        """源冷却只屏蔽该源（v1.0 缺陷回归）。"""
        executed: list = []
        scheduler = PrefetchScheduler(
            idle_check=lambda: True,
            gate_allow=lambda source: source != "cooled",
            pull=lambda job: executed.append(job) or True,
        )
        scheduler.enqueue(self._job(page_key="cooled"))
        scheduler.enqueue(self._job(page_key="healthy"))
        scheduler._drain_one()
        scheduler._drain_one()
        self.assertEqual([job.page_key for job in executed], ["healthy"])

    def test_cancel_page_removes_and_blocks_future(self) -> None:
        """取消：既清空待办，也拦住未来同名入队（离开页面立即取消）。"""
        scheduler = PrefetchScheduler(
            idle_check=lambda: True,
            gate_allow=lambda source: True,
            pull=lambda job: True,
        )
        scheduler.enqueue(self._job(page_key="page1"))
        scheduler.cancel_page("page1")
        self.assertEqual(scheduler.pending_count(), 0)
        scheduler.enqueue(self._job(page_key="page1"))
        self.assertEqual(scheduler.pending_count(), 0, "取消后的同名入队应被拦截")

    def test_queue_depth_gate(self) -> None:
        """翻译队列深时后台让出（R1：实时优先）。"""
        executed: list = []
        scheduler = PrefetchScheduler(
            idle_check=lambda: True,
            gate_allow=lambda source: True,
            pull=lambda job: executed.append(job) or True,
            model_queue_depth_max=0,
            queue_depth_probe=lambda: 3,
        )
        scheduler.enqueue(self._job())
        self.assertFalse(scheduler._drain_one())
        self.assertEqual(executed, [])

    def test_failed_task_is_not_reported_as_translated(self) -> None:
        scheduler = PrefetchScheduler(
            idle_check=lambda: True, gate_allow=lambda source: True, pull=lambda job: False,
        )
        scheduler.enqueue(self._job())
        self.assertFalse(scheduler._drain_one())
        self.assertEqual(scheduler.page_progress("page1"), (0, 1))
        self.assertEqual(scheduler.page_failures("page1"), 1)

    def test_success_and_failure_have_separate_counters(self) -> None:
        scheduler = PrefetchScheduler(
            idle_check=lambda: True, gate_allow=lambda source: True,
            pull=lambda job: job.payload == 0,
        )
        scheduler.enqueue(self._job(payload=0))
        scheduler.enqueue(self._job(payload=1))
        self.assertTrue(scheduler._drain_one())
        self.assertFalse(scheduler._drain_one())
        self.assertEqual(scheduler.page_progress("page1"), (1, 2))
        self.assertEqual(scheduler.page_failures("page1"), 1)

    def test_cancelled_tasks_are_not_reported_as_translated(self) -> None:
        scheduler = PrefetchScheduler(
            idle_check=lambda: True, gate_allow=lambda source: True, pull=lambda job: True,
        )
        scheduler.enqueue(self._job())
        scheduler.cancel_page("page1")
        self.assertEqual(scheduler.page_progress("page1"), (0, 1))

    def test_reimported_page_starts_new_progress_counters(self) -> None:
        scheduler = PrefetchScheduler(
            idle_check=lambda: True, gate_allow=lambda source: True,
            pull=lambda job: job.payload == 0,
        )
        scheduler.enqueue(self._job(payload=1))
        scheduler._drain_one()
        scheduler.cancel_page("page1")
        scheduler.clear_cancelled("page1")
        scheduler.enqueue(self._job(payload=0))
        scheduler._drain_one()
        self.assertEqual(scheduler.page_progress("page1"), (1, 1))
        self.assertEqual(scheduler.page_failures("page1"), 0)

    def test_old_inflight_job_does_not_count_toward_reimported_page(self) -> None:
        def pull(job):
            if job.payload == 0:
                scheduler.cancel_page("page1")
                scheduler.clear_cancelled("page1")
                scheduler.enqueue(self._job(payload=1))
            return True
        scheduler = PrefetchScheduler(
            idle_check=lambda: True, gate_allow=lambda source: True, pull=pull,
        )
        scheduler.enqueue(self._job(payload=0))
        scheduler._drain_one()
        self.assertEqual(scheduler.page_progress("page1"), (0, 1))
        scheduler._drain_one()
        self.assertEqual(scheduler.page_progress("page1"), (1, 1))

    def test_progress_callback_failure_does_not_stop_other_jobs(self) -> None:
        scheduler = PrefetchScheduler(
            idle_check=lambda: True, gate_allow=lambda source: True, pull=lambda job: True,
        )
        def broken_progress(*args):
            raise RuntimeError("callback failed")
        scheduler.progress = broken_progress
        scheduler.enqueue(self._job(payload=0))
        scheduler.enqueue(self._job(payload=1))
        self.assertTrue(scheduler._drain_one())
        self.assertTrue(scheduler._drain_one())
        self.assertEqual(scheduler.page_progress("page1"), (2, 2))

    def test_worker_drains_with_gates(self) -> None:
        """worker 线程在条件满足时消费任务。"""
        executed: list = []
        done = threading.Event()
        scheduler = PrefetchScheduler(
            idle_check=lambda: True,
            gate_allow=lambda source: True,
            pull=lambda job: executed.append(job) or done.set(),
        )
        scheduler.start()
        try:
            scheduler.enqueue(self._job())
            self.assertTrue(done.wait(timeout=5), "worker 未在 5s 内消费任务")
        finally:
            scheduler.stop()
        self.assertEqual(len(executed), 1)


if __name__ == "__main__":
    unittest.main()
