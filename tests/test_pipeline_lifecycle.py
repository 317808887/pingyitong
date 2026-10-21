"""流水线生命周期：停止 → 重启不得泄漏"僵尸"工作线程。

对应一个真实缺陷：旧实现的 ``stop()`` 等待线程退出有 2 秒超时，
正在一次数秒推理中的翻译线程超时后仍然存活；而 ``start()`` 会
``clear()`` **共享的**停止事件——僵尸线程的循环条件重新成立，
从此永不退出，与新一代线程并存：双倍推理负载、争抢同一批队列、
暂停期间还在往结果队列里塞过期帧。

修复后的契约（本测试钉住）：

1. 每次启动使用**新一代**停止事件，工作线程持有自己出生时的那个；
2. ``stop()`` 置位后，即使线程正卡在一次长推理里，它完成当前条目
   后也必然退出（检查的是自己那代已置位的事件）；
3. 卡住的线程在被叫停后产出的过期帧**不得**进入结果队列。
"""

from __future__ import annotations

import threading
import time
import unittest
from dataclasses import dataclass

try:
    from PyQt6.QtWidgets import QApplication

    PYQT_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYQT_AVAILABLE = False

from pingyitong.components.capture.geometry import Rect
from pingyitong.components.ocr import RecognizedFrame, TextBlock
from pingyitong.components.render import RenderFrame
from pingyitong.core.pipeline import Pipeline


def _ensure_app():
    """Qt 定时器需要 QApplication 实例。"""
    if not PYQT_AVAILABLE:
        return None
    app = QApplication.instance()
    if app is None:
        try:
            app = QApplication([])
        except Exception:  # noqa: BLE001
            return None
    return app


@dataclass
class _FakeCaptureFrame:
    """捕获帧替身：只需 seq 与时间戳。"""

    seq: int = 1
    timestamp: float = 0.0


class _FakeCapturer:
    """捕获器替身：快速产出序号递增的帧。"""

    def __init__(self) -> None:
        self._seq = 0
        self.last_capture_cost = 0.001

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def capture_all(self) -> _FakeCaptureFrame:
        self._seq += 1
        return _FakeCaptureFrame(seq=self._seq, timestamp=float(self._seq))


class _FakeOcr:
    """OCR 替身：把每帧变成一个含单条文本的识别结果。"""

    def __init__(self) -> None:
        self.configured: dict | None = None

    def configure(self, **kwargs) -> None:
        self.configured = kwargs

    def is_ready(self) -> bool:
        return True

    def recognize_frame(self, frame: _FakeCaptureFrame) -> RecognizedFrame:
        return RecognizedFrame(
            seq=frame.seq,
            origin=Rect(0, 0, 100, 100),
            blocks=[
                TextBlock(text="hello world", box=Rect(0, 0, 60, 20), confidence=0.9)
            ],
            is_full_frame=True,
        )


class _BlockingTranslator:
    """翻译替身：第一次调用阻塞，模拟一次长推理。

    第二次及以后的调用（重启后的新一代线程）立即返回，
    避免测试尾部出现高速空转。被阻塞那次调用的帧序号记入
    :attr:`blocked_seq`，供"过期帧不得入队"的断言使用。
    """

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.blocked_seq: int | None = None
        self._calls_lock = threading.Lock()

    @property
    def last_stats(self):
        class _Stats:
            model_id = ""

        return _Stats()

    def translate_frame(
        self,
        recognized: RecognizedFrame,
        snapshot: object | None = None,
        cpu_sustained_high: bool = False,
    ) -> RenderFrame:
        with self._calls_lock:
            self.calls += 1
            call_index = self.calls
        if call_index == 1:
            self.blocked_seq = recognized.seq
            self.entered.set()
            self.release.wait(timeout=10)
        return RenderFrame(
            seq=recognized.seq,
            origin=recognized.origin,
            items=[],
            is_full_frame=True,
            timings={},
        )


class _SpyQueue:
    """结果队列间谍：转发全部操作并记录入队帧的序号。

    直接检查 ``LatestQueue`` 的内容会受"保留最新"淘汰策略影响——
    过期帧可能刚入队就被更新的帧挤掉，检查因此不确定。间谍在
    ``put`` 入口记录，任何一次入队（含过期帧）都逃不掉。
    """

    def __init__(self, target) -> None:
        self._target = target
        self.put_seqs: list[int] = []

    def put(self, item, seq: int | None = None, **kwargs):
        self.put_seqs.append(seq)
        return self._target.put(item, seq=seq, **kwargs)

    def get(self, timeout: float | None = None):
        return self._target.get(timeout)

    def get_latest(self):
        return self._target.get_latest()

    def wake_all(self) -> None:
        self._target.wake_all()

    def clear(self) -> None:
        self._target.clear()

    def reset_stats(self) -> None:
        self._target.reset_stats()

    def __len__(self) -> int:
        return len(self._target)


class _FakeOverlay:
    """翻译层替身：只记录 clear 调用。"""

    def __init__(self) -> None:
        self.is_visible = False
        self.cleared = 0

    def clear(self) -> None:
        self.cleared += 1


class _CaptureConfig:
    """最小捕获配置：全屏、无差分、高帧率。"""

    def __init__(self) -> None:
        self.fps = 60
        self.mode = "fullscreen"
        self.region = None
        self.monitors: list = []
        self.enable_diff = False


class _Config:
    """最小总配置替身。"""

    def __init__(self) -> None:
        from types import SimpleNamespace

        self.capture = _CaptureConfig()
        self.translation = SimpleNamespace(
            target_language="zh", source_language="auto"
        )
        # apply_config 会透传 OCR 参数；给最小合法值供热更新测试使用。
        self.ocr = SimpleNamespace(
            upscale_factor=1.0,
            enable_preprocess=True,
            min_confidence=0.5,
            min_box_height=8,
            box_padding=2,
            threads=4,
            source_language="auto",
        )


def _make_pipeline() -> tuple[Pipeline, _BlockingTranslator, _FakeOverlay]:
    from pingyitong.core.pipeline import Pipeline

    config = _Config()
    translator = _BlockingTranslator()
    overlay = _FakeOverlay()
    pipeline = Pipeline(
        config=config,
        capturer=_FakeCapturer(),
        differ=object(),
        ocr_engine=_FakeOcr(),
        translator=translator,
        overlay=overlay,
    )
    return pipeline, translator, overlay


@unittest.skipUnless(PYQT_AVAILABLE, "需要 PyQt6")
class TestPipelineLifecycle(unittest.TestCase):
    """stop → start 交替时的工作线程生命周期。"""

    @classmethod
    def setUpClass(cls) -> None:
        if _ensure_app() is None:
            raise unittest.SkipTest("无法创建 QApplication")

    def test_stop_does_not_revive_blocked_thread(self) -> None:
        """停止时卡在推理里的线程，重启后不得复活并继续工作。"""
        pipeline, translator, _overlay = _make_pipeline()

        self.assertTrue(pipeline.start())

        # 等翻译线程进入"长推理"（阻塞）状态
        self.assertTrue(translator.entered.wait(timeout=5), "翻译线程未进入推理")
        threads_before_stop = list(pipeline._threads)

        # 停止：join 有超时上限，卡住的线程是"幸存者"
        pipeline.stop()
        self.assertFalse(pipeline.is_running)
        blocked = [
            thread
            for thread in threads_before_stop
            if thread.is_alive() and thread.name == "TranslateThread"
        ]
        self.assertEqual(len(blocked), 1, "卡住的翻译线程应仍在收尾")

        # 重启：新一代事件。旧线程持有旧事件（已置位）。
        self.assertTrue(pipeline.start())
        self.assertTrue(pipeline.is_running)

        # 换上间谍队列：捕获之后**所有**入队帧的序号
        from pingyitong.core.task_queue import LatestQueue

        spy = _SpyQueue(LatestQueue(maxsize=3, keep_latest_only=True, name="spy"))
        pipeline._result_queue = spy

        # 放行第一次推理：旧线程完成当前条目后必须退出
        translator.release.set()
        blocked[0].join(timeout=5)
        self.assertFalse(blocked[0].is_alive(), "旧线程未退出——僵尸线程复现")

        # 旧线程叫停后正在处理的那一帧不得入队（捕获线程的序号单调
        # 递增，重启后的新一代只会产出更大的序号）
        self.assertNotIn(
            translator.blocked_seq,
            spy.put_seqs,
            "叫停后的过期帧被写入结果队列",
        )

        pipeline.stop()

    def test_normal_stop_leaves_no_survivors(self) -> None:
        """没有卡住的推理时，停止后不应残留任何工作线程。"""
        pipeline, translator, _overlay = _make_pipeline()
        translator.release.set()  # 首次调用也不阻塞：构造"一切正常"的场景

        self.assertTrue(pipeline.start())
        time.sleep(0.2)  # 让各线程跑起来并处理若干帧
        pipeline.stop()

        alive = [thread for thread in pipeline._threads if thread.is_alive()]
        self.assertEqual(alive, [])
        self.assertFalse(pipeline.is_running)


class TestCaptureFpsHotApply(unittest.TestCase):
    """监视频率独立于旧捕获帧率，按监测间隔热更新。"""

    def test_apply_config_updates_limiter_interval(self) -> None:
        pipeline, _translator, _overlay = _make_pipeline()
        from pingyitong.core.task_queue import RateLimiter

        limiter = RateLimiter(pipeline._config.capture.fps)
        pipeline._capture_limiter = limiter

        new_config = _Config()
        new_config.capture.fps = 5
        new_config.capture.monitor_interval_ms = 40
        pipeline.apply_config(new_config)

        self.assertIs(pipeline._config, new_config)
        self.assertAlmostEqual(limiter._interval, 0.04, places=6)

    def test_apply_config_before_capture_loop_is_safe(self) -> None:
        """捕获线程尚未启动（限速器为 None）时改配置不抛异常。"""
        pipeline, _translator, _overlay = _make_pipeline()
        self.assertIsNone(pipeline._capture_limiter)
        pipeline.apply_config(_Config())


if __name__ == "__main__":
    unittest.main()
