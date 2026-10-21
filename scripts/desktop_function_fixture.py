"""Known text for testing the installed desktop app; does not control its UI."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from PyQt6.QtCore import QPoint, Qt, QTimer
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget


class PaintedLabel(QLabel):
    """Acknowledge the actual source-label paint, rather than a control-file write."""

    def __init__(self, text, parent, index, generation, callback):
        super().__init__(text, parent)
        self.index = index
        self.generation = generation
        self.callback = callback

    def paintEvent(self, event):  # noqa: N802
        super().paintEvent(event)
        self.callback(self.index, self.generation, time.perf_counter_ns())


class Fixture(QWidget):
    def __init__(self, control: Path, state: Path, output: Path | None, run_id: str):
        super().__init__()
        self.control, self.state, self.output, self.run_id = control, state, output, run_id
        self.setWindowTitle(f"屏译通功能测试：已知字幕 [{run_id}]")
        self.setGeometry(680, 340, 900, 360)
        self.setStyleSheet("background: #111111; color: white;")
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(28, 28, 28, 28)
        self.previous = ""
        self.labels = []
        self.generation = 0
        self.recorded_generation = -1
        self.painted = {}
        self.command = {}
        self.received_ns = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(20)
        self.poll()

    def poll(self) -> None:
        if not self.control.exists():
            return
        try:
            raw = self.control.read_text(encoding="utf-8-sig")
            if raw == self.previous:
                return
            command = json.loads(raw)
        except (OSError, ValueError):
            return  # A writer may still be replacing its control file.
        if command.get("run_id", self.run_id) != self.run_id:
            return  # Independent runs must never consume each other's scenes.
        self.received_ns = time.perf_counter_ns()
        self.generation += 1
        self.command = command
        self.painted = {}
        for label in self.labels:
            self.layout_.removeWidget(label)
            label.hide()
            label.deleteLater()
        self.labels.clear()
        foreground = QColor(command.get("foreground", "#ffffff"))
        background = QColor(command.get("background", "#111111"))
        if not foreground.isValid() or not background.isValid():
            foreground, background = QColor("white"), QColor("#111111")
        self.setStyleSheet(f"background: {background.name()}; color: {foreground.name()};")
        if "geometry" in command:
            self.setGeometry(*command["geometry"])
        blocks = command.get("blocks", [])
        for index, block in enumerate(blocks or [{"text": command.get("text", "")}]):
            label = PaintedLabel(block["text"], self, index, self.generation, self.on_painted)
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setFont(QFont(command.get("font_family", "Microsoft YaHei"), command.get("font_size", 25)))
            label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            label.setWordWrap(bool(command.get("word_wrap", True)))
            if blocks:
                label.setGeometry(*block["geometry"])
            else:
                self.layout_.addWidget(label)
            label.show()
            self.labels.append(label)
        self.previous = raw
        self.update()

    def on_painted(self, index: int, generation: int, timestamp_ns: int) -> None:
        if generation != self.generation or generation == self.recorded_generation:
            return
        self.painted[index] = timestamp_ns
        if len(self.painted) != len(self.labels):
            return
        self.recorded_generation = generation
        painted_ns = max(self.painted.values())
        result = {"run_id": self.run_id, "scene_id": self.command.get("scene_id", "legacy"),
                  "generation": generation, "text": self.command.get("text", ""),
                  "blocks": self.command.get("blocks", []), "updated_at": time.time(),
                  "received_monotonic_ns": self.received_ns, "paint_completed_monotonic_ns": painted_ns,
                  "clock": "time.perf_counter_ns (same-host monotonic)",
                  "timing_scope": "All visible source labels finished Qt paint; excludes unobservable compositor presentation delay",
                  "control_sha256": hashlib.sha256(self.previous.encode("utf-8")).hexdigest(),
                  "client_origin_logical": [self.mapToGlobal(QPoint(0, 0)).x(), self.mapToGlobal(QPoint(0, 0)).y()],
                  "client_size_logical": [self.width(), self.height()],
                  "device_pixel_ratio": self.devicePixelRatioF(),
                  "geometry": [self.x(), self.y(), self.width(), self.height()]}
        # File I/O runs after painting; it does not move the recorded start time.
        QTimer.singleShot(0, lambda result=result: self.write_evidence(result))

    def write_evidence(self, result: dict) -> None:
        self.state.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state.with_name(self.state.name + ".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.state)
        if self.output:
            self.output.mkdir(parents=True, exist_ok=True)
            with (self.output / "fixture_paints.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts" / "desktop_fixture_control.json")
    parser.add_argument("--state", type=Path)
    parser.add_argument("--output", type=Path, help="Append actual source-paint timestamps here")
    parser.add_argument("--run-id", default="legacy")
    args = parser.parse_args()
    app = QApplication([sys.argv[0]])
    widget = Fixture(args.control, args.state or args.control.with_name("desktop_fixture_state.json"), args.output, args.run_id)
    widget.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
