"""Explicit, test-home-only evidence of actual application painting.

The saved image is the buffer submitted by paintEvent, not a DWM screenshot.
Personal runs never write source text or images through this module.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import paths

_writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="PaintEvidence")
_lock = threading.Lock()
_identity: dict | None = None


def _directory() -> Path | None:
    raw = os.environ.get("PINGYITONG_ACCEPTANCE_DIR", "")
    home = os.environ.get("PINGYITONG_HOME", "")
    if not raw or not home:
        return None
    destination = Path(raw).resolve()
    root = Path(home).resolve()
    if destination == root or not destination.is_relative_to(root):
        return None
    return destination


def evidence_enabled() -> bool:
    return _directory() is not None


def _hash(path: Path) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_paint(frame, screen: str, buffer, items: list[dict], painted_at: float) -> None:
    destination = _directory()
    if destination is None:
        return
    # Capture mutable values on the GUI thread, perform disk I/O afterwards.
    data = {
        "evidence_kind": "application_submitted_paint_buffer",
        "seq": frame.seq, "scene_generation": getattr(frame, "_scene_generation", None),
        "captured_at": getattr(frame, "_captured_at", None),
        "change_started_at": getattr(frame, "_change_started_at", None),
        "observation_lower_bound": getattr(frame, "_observation_lower_bound", None),
        "painted_at": painted_at, "timings": dict(frame.timings),
        "diagnostics": dict(frame.diagnostics), "items": items,
        "delivery_evidence": getattr(frame, "_delivery_evidence", None),
        "diagnostic_events": list(getattr(frame, "_diagnostic_events", [])),
        "source_lines": list(getattr(frame, "_source_lines", [])),
        "paragraph_provenance": list(getattr(frame, "_paragraph_provenance", [])),
        "refresh_kind": getattr(frame, "_refresh_kind", None),
        "screen": screen, "run_id": os.environ.get("PINGYITONG_ACCEPTANCE_RUN", ""),
        "process_id": os.getpid(),
    }
    _writer.submit(_write, destination, data, buffer.copy())


def save_execution_event(event: dict) -> None:
    destination = _directory()
    if destination is None:
        return
    data = {**event, "run_id": os.environ.get("PINGYITONG_ACCEPTANCE_RUN", ""),
            "evidence_kind": "observed_model_pool_operation"}
    try:
        _writer.submit(_write_execution_event, destination, data)
    except RuntimeError:
        # Application shutdown may already have closed the evidence writer.
        # Instrumentation must never turn a successful model call into failure.
        from .logger import get_logger
        get_logger(__name__).warning("本轮模型证据写入器已关闭，未保存事件")


def _write_execution_event(destination: Path, data: dict) -> None:
    try:
        destination.mkdir(parents=True, exist_ok=True)
        with _lock, (destination / "model_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(data, ensure_ascii=False) + "\n")
    except Exception:
        from .logger import get_logger
        get_logger(__name__).exception("无法保存本轮模型阶段证据")


def _write(destination: Path, data: dict, buffer) -> None:
    global _identity
    from ..version import __version__

    try:
        destination.mkdir(parents=True, exist_ok=True)
        with _lock:
            if _identity is None:
                _identity = {
                    "version": __version__, "executable": sys.executable,
                    "process_id": os.getpid(),
                    "executable_sha256": _hash(Path(sys.executable)),
                    "config_sha256": _hash(paths.app_data_dir() / "config.json"),
                }
                (destination / "identity.json").write_text(
                    json.dumps(_identity, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            suffix = hashlib.sha256(data["screen"].encode()).hexdigest()[:8]
            name = f"paint_{data['seq']}_{suffix}_{data['painted_at']:.6f}"
            buffer.save(str(destination / f"{name}.png"))
            (destination / f"{name}.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    except Exception:
        from .logger import get_logger
        get_logger(__name__).exception("无法保存本轮实机绘制证据")
