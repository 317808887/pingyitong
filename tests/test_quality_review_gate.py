"""The quality review command must fail when translated output is invalid."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def test_review_rejects_bad_output_and_uses_product_path(monkeypatch, capsys):
    from pingyitong.components import llm
    from pingyitong.services import translation_service

    class Pool:
        closed = False

        def __init__(self, *args, **kwargs):
            pass

        def ensure_loaded(self, *args):
            return True

        def translate(self, **kwargs):
            raise AssertionError("Quality review must inspect final product output")

        def shutdown(self):
            Pool.closed = True

    class Translator:
        def __init__(self, *args, **kwargs):
            pass

        def translate_text(self, source):
            return "<<<SEG>>>"

    monkeypatch.setattr(llm, "ModelPool", Pool)
    monkeypatch.setattr(translation_service, "TranslationService", Translator)
    path = Path(__file__).resolve().parents[1] / "scripts" / "review_translation_quality.py"
    spec = importlib.util.spec_from_file_location("quality_review_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main() == 1
    assert Pool.closed
    assert "分段标记" in capsys.readouterr().out
