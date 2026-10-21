"""Chinese-adjacent technical names must not trigger endless language correction."""

import pytest

from pingyitong.config.schema import AppConfig
from pingyitong.services.language import has_foreign_prose
from tests.test_mixed_target_language import _Responses
from tests.test_translation_service import block, make_frame, make_service


@pytest.mark.parametrize("source,candidate", [
    ("缓存模型、Qt.Window closeEvent和app.py按钮同步。", "缓存模型、Qt.Window closeEvent和app.py按钮同步。"),
    ("检测snapshot_items与_screen_items的完整快照。", "检测snapshot_items与_screen_items的完整快照。"),
    ("报告请见translation_final.json和final_real_e2e.json。", "报告请见translation_final.json和final_real_e2e.json。"),
    ("Open app.py and inspect closeEvent after saving.", "保存后打开app.py并检查closeEvent。"),
    ("检查app.py以继续。", "运行app.py即可。"),
    ("Open résumé.txt and анализ.json.", "打开résumé.txt和анализ.json。"),
    ("Read snapshot_items and keep foo_bar unchanged.", "读取snapshot_items并保持foo_bar不变。"),
])
def test_source_technical_identifiers_touching_chinese_are_valid(source, candidate):
    assert not has_foreign_prose(candidate, "zh", source)


@pytest.mark.parametrize("source,candidate", [
    ("打开窗口。", "打开窗口，close the window。"),
    ("Save the file and keep this window open.", "保存文件并keep this window open。"),
    ("Inspect app.py after saving.", "保存后检查other.py。"),
    ("检查snapshot_items。", "检查snapshot_items并close the window。"),
])
def test_new_identifiers_and_untranslated_foreign_prose_are_still_rejected(source, candidate):
    assert has_foreign_prose(candidate, "zh", source)


@pytest.mark.parametrize("source", [
    "缓存模型、Qt.Window closeEvent和app.py按钮同步。",
    "检测snapshot_items与screen_items的完整快照。",
    "报告请见translation_final.json和final_real_e2e.json。",
    "中文界面已准备好，API和OCR请求正常。",
])
def test_auto_chinese_technical_content_has_no_retry_regions_or_repeated_inference(source):
    config = AppConfig()
    config.translation.semantic_review = False  # legacy single-pass contract
    config.translation.source_language = "auto"
    config.translation.target_language = "zh"
    pool = _Responses(source)
    service, _ = make_service(config, pool=pool)
    frame = make_frame(block(source, 0, 0))
    first = service.translate_frame(frame)
    assert len(first.items) == 1
    assert first.items[0].text == source
    assert first.retry_regions == []
    assert len(pool.calls) == 1
    second = service.translate_frame(frame)
    assert len(second.items) == 1
    assert second.items[0].from_cache
    assert second.retry_regions == []
    assert len(pool.calls) == 1

