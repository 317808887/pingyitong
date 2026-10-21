"""Changing the online translator must replace translations already on screen."""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace
from unittest import TestCase, mock

from pingyitong.config.schema import ApiConfig, AppConfig, TranslationConfig
from pingyitong.ui.app import Application
from pingyitong.ui.main_panel import MainPanel


class TestApiConfigInvalidation(TestCase):
    def test_connection_button_uses_active_model_pool(self) -> None:
        api = ApiConfig(
            enabled=True,
            base_url="https://example.test/v1",
            api_key="test-key",
            model_name="test-model",
        )
        model_pool = mock.Mock()
        model_pool.test_api_connection.return_value = (True, "连接成功")
        finished = mock.Mock()
        panel = SimpleNamespace(
            _persist_api=mock.Mock(),
            _settings=SimpleNamespace(config=SimpleNamespace(api=api, translation=TranslationConfig(local_only=False))),
            _api_test_result=mock.Mock(),
            _api_test_button=mock.Mock(),
            _api_test_holder=SimpleNamespace(finished=finished),
            _controller=SimpleNamespace(model_pool=model_pool),
        )

        with mock.patch("threading.Thread") as thread_class:
            thread_class.return_value.start.side_effect = (
                lambda: thread_class.call_args.kwargs["target"]()
            )
            MainPanel._on_test_api(panel)

        panel._persist_api.assert_called_once_with()
        model_pool.test_api_connection.assert_called_once_with(api)
        tested_api = model_pool.test_api_connection.call_args.args[0]
        self.assertIsNot(tested_api, api)
        finished.emit.assert_called_once_with(asdict(api), True, "连接成功")

    def test_successful_connection_refreshes_active_translation(self) -> None:
        api = ApiConfig(enabled=True, api_key="test-key", model_name="test-model")
        panel = self._test_result_panel(api)

        MainPanel._on_api_test_finished(panel, asdict(api), True, "连接成功")

        panel._api_test_button.setEnabled.assert_called_once_with(True)
        panel._api_test_result.setText.assert_called_once_with("连接成功")
        panel._controller.refresh_api_translation.assert_called_once_with()

    def test_result_for_old_key_does_not_mark_new_key_successful(self) -> None:
        api = ApiConfig(enabled=True, api_key="first-key", model_name="first-model")
        tested_settings = asdict(api)
        panel = self._test_result_panel(api)
        api.api_key = "second-key"
        api.model_name = "second-model"

        MainPanel._on_api_test_finished(panel, tested_settings, True, "连接成功")

        panel._api_test_button.setEnabled.assert_called_once_with(True)
        panel._api_test_result.setText.assert_called_once_with("配置已变化，请重新测试")
        panel._refresh_api_status.assert_called_once_with()
        panel._controller.refresh_api_translation.assert_not_called()

    @staticmethod
    def _test_result_panel(api: ApiConfig) -> SimpleNamespace:
        return SimpleNamespace(
            _settings=SimpleNamespace(config=SimpleNamespace(api=api, translation=TranslationConfig(local_only=False))),
            _api_test_button=mock.Mock(),
            _api_test_result=mock.Mock(),
            _api_status=mock.Mock(),
            _refresh_api_status=mock.Mock(),
            _controller=SimpleNamespace(refresh_api_translation=mock.Mock()),
        )

    def test_recovered_connection_replaces_static_screen_output(self) -> None:
        controller = SimpleNamespace(
            translator=mock.Mock(),
            pipeline=mock.Mock(),
            transcript_panel=SimpleNamespace(is_expanded=True, set_text=mock.Mock()),
        )

        Application.refresh_api_translation(controller)

        controller.pipeline.invalidate.assert_called_once_with(clear_output=True)
        controller.translator.clear_output.assert_called_once_with()
        controller.transcript_panel.set_text.assert_called_once_with("（等待识别…）")

    def test_enabling_and_changing_key_retranslate_static_screen(self) -> None:
        config = AppConfig()
        controller = SimpleNamespace(
            _last_translation_direction=(
                config.translation.source_language,
                config.translation.target_language,
            ),
            _last_api_settings=asdict(config.api),
            overlay=mock.Mock(),
            ocr_engine=mock.Mock(),
            pipeline=mock.Mock(),
            preprocessor=mock.Mock(),
            differ=mock.Mock(),
            scheduler=mock.Mock(),
            model_pool=mock.Mock(),
            cache=mock.Mock(),
            translator=mock.Mock(),
            _refresh_api_runtime_status=mock.Mock(),
            history=mock.Mock(),
            prefetch_gate=mock.Mock(),
            network_monitor=mock.Mock(),
            panel=mock.Mock(),
            transcript_panel=SimpleNamespace(is_expanded=True, set_text=mock.Mock()),
        )

        Application._on_config_changed(controller, config)
        controller.pipeline.invalidate.assert_not_called()
        controller.translator.clear_output.assert_not_called()

        config.api.enabled = True
        config.api.base_url = "https://example.test/v1"
        config.api.model_name = "test-model"
        config.api.api_key = "first-key"
        Application._on_config_changed(controller, config)
        controller.pipeline.invalidate.assert_called_once_with(clear_output=True)
        controller.translator.clear_output.assert_called_once_with()
        controller.transcript_panel.set_text.assert_called_once_with("（等待识别…）")

        controller.pipeline.invalidate.reset_mock()
        config.api.api_key = "second-key"
        Application._on_config_changed(controller, config)
        controller.pipeline.invalidate.assert_called_once_with(clear_output=True)

        controller.pipeline.invalidate.reset_mock()
        Application._on_config_changed(controller, config)
        controller.pipeline.invalidate.assert_not_called()

    def test_api_badge_reports_runtime_route_and_failure(self) -> None:
        api = ApiConfig(enabled=True, model_name="test-model")
        controller = SimpleNamespace(
            settings=SimpleNamespace(config=SimpleNamespace(api=api)),
            model_pool=mock.Mock(),
            panel=mock.Mock(),
        )
        cases = [
            (True, 0, 0.0, "test-model", "已启用 · 待请求 · test-model", "info"),
            (True, 1, 0.3, "test-model", "在线 AI · test-model", "success"),
            (True, 2, 0.0, "test-model", "上次请求失败 · 本地回退", "warning"),
            (False, 3, 0.0, "认证错误 HTTP 401", "本地回退 · 认证错误 HTTP 401", "warning"),
        ]
        for available, calls, latency, message, expected, tone in cases:
            with self.subTest(expected=expected):
                controller.model_pool.api_status.return_value = SimpleNamespace(
                    available=available, total_calls=calls,
                    last_latency=latency, message=message,
                )
                controller.panel.reset_mock()
                Application._refresh_api_runtime_status(controller)
                controller.panel.set_api_status.assert_called_once_with(expected, tone)

        api.enabled = False
        controller.panel.reset_mock()
        Application._refresh_api_runtime_status(controller)
        controller.panel.set_api_status.assert_called_once_with("未启用", "info")

    def test_saving_unchanged_settings_keeps_live_failure_badge(self) -> None:
        controller = SimpleNamespace(_refresh_api_runtime_status=mock.Mock())
        panel = SimpleNamespace(_controller=controller, _api_status=mock.Mock())
        controller.panel = panel

        MainPanel._refresh_api_status(panel)

        controller._refresh_api_runtime_status.assert_called_once_with()
        panel._api_status.set_status.assert_not_called()
