"""Offline smoke checks for the subject-pack selector in the settings panel."""

from __future__ import annotations

from types import SimpleNamespace


def test_subject_selector_persists_and_reloads(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from PyQt6.QtWidgets import QApplication

    from pingyitong.config.settings import SettingsManager
    from pingyitong.ui.app import Application
    from pingyitong.ui.design.theme import build_theme
    from pingyitong.ui.main_panel import MainPanel

    qt_app = QApplication.instance() or QApplication([])
    settings = SettingsManager(tmp_path / "config.json")
    calls = []
    controller = SimpleNamespace(
        theme=build_theme(),
        settings=settings,
        reload_glossary=lambda: calls.append("reload"),
    )
    controller.select_glossary_domain = lambda domain: Application.select_glossary_domain(
        controller, domain
    )

    panel = MainPanel(controller)
    assert panel._glossary_domain.count() == 7
    assert panel._glossary_domain.currentData() == "software_ai"

    panel._glossary_domain.setCurrentIndex(1)
    assert settings.config.translation.glossary_domain == "medicine"
    assert calls == ["reload"]

    panel.set_glossary_mode("custom")
    assert not panel._glossary_domain.isEnabled()
    panel.set_glossary_mode("builtin")
    assert panel._glossary_domain.isEnabled()
    panel.close()
    assert qt_app is not None
