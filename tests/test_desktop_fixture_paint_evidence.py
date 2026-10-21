from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "desktop_function_fixture.py"
spec = importlib.util.spec_from_file_location("desktop_fixture_evidence", SCRIPT)
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)


def test_scene_time_is_last_actual_label_paint_not_control_receipt(monkeypatch, tmp_path):
    pending = []
    monkeypatch.setattr(fixture_module.QTimer, "singleShot", lambda _, callback: pending.append(callback))
    fake = SimpleNamespace(generation=4, recorded_generation=-1, painted={}, labels=[0, 1],
                           command={"scene_id": "new-scene", "text": "test"}, run_id="test-run",
                           received_ns=100, previous="control", x=lambda: 1, y=lambda: 2,
                           mapToGlobal=lambda _: fixture_module.QPoint(1, 32), devicePixelRatioF=lambda: 1.5,
                           width=lambda: 900, height=lambda: 300, state=tmp_path / "state.json", output=tmp_path / "raw")
    fake.write_evidence = lambda value: fixture_module.Fixture.write_evidence(fake, value)
    fixture_module.Fixture.on_painted(fake, 0, 4, 300)
    assert pending == []
    fixture_module.Fixture.on_painted(fake, 1, 3, 400)  # Old scene cannot finish new scene.
    assert pending == []
    fixture_module.Fixture.on_painted(fake, 1, 4, 500)
    assert len(pending) == 1
    assert not fake.state.exists()
    pending.pop()()
    recorded = json.loads(fake.state.read_text(encoding="utf-8"))
    assert recorded["paint_completed_monotonic_ns"] == 500
    assert recorded["received_monotonic_ns"] == 100
    assert recorded["scene_id"] == "new-scene"
    assert recorded["client_origin_logical"] == [1, 32]
    assert recorded["client_size_logical"] == [900, 300]
    assert recorded["device_pixel_ratio"] == 1.5
    fixture_module.Fixture.on_painted(fake, 0, 4, 900)
    assert pending == []
    assert len((fake.output / "fixture_paints.jsonl").read_text(encoding="utf-8").splitlines()) == 1
