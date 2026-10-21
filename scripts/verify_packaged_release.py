"""Read a PyInstaller EXE and check release code without starting the app.

Example after the final rebuild:
    python -X utf8 scripts/verify_packaged_release.py \
        build/dist/PingYiTong/PingYiTong.exe --version 0.6.13 \
        --output artifacts/packaged_release_0613_verified.json

This never imports product modules, reads settings, or starts the executable.
"""

from __future__ import annotations

import argparse
import dis
import hashlib
import json
import types
from pathlib import Path

from PyInstaller.archive.readers import CArchiveReader

ROOT = Path(__file__).resolve().parents[1]
CHECKS = {
    "pingyitong.__main__": ("main",),
    "pingyitong.selfcheck": ("_check_ocr",),
    "pingyitong.components.llm.prompt": (
        "parse_response", "_strip_code_fence", "_strip_source_suffix", "build_compact_prompt",
    ),
    "pingyitong.services.translation_service": (
        "_parse_single_response", "TranslationService._valid_output",
        "TranslationService._translate_batch", "TranslationService._translate_text_request",
        "TranslationService._cache_key", "TranslationService._run_correction_batches",
        "TranslationService._translate_pending", "TranslationService._retry_without_placeholders",
        "TranslationService._needs_translation",
    ),
    "pingyitong.config.settings": ("SettingsManager._sanitize",),
    "pingyitong.ui.main_panel": ("MainPanel._on_languages_changed",),
    "pingyitong.components.llm.model_pool": ("ModelPool.shutdown", "ModelPool.reset"),
    "pingyitong.services.language": ("has_foreign_prose", "_is_obvious_english_prose"),
    "pingyitong.services.code_filter": ("looks_like_code",),
    "pingyitong.utils.process_guard": (
        "_kernel32", "create_kill_on_close_job", "assign_to_job",
        "_record_owned_server", "_read_ownership_record", "_owner_alive", "reap_stale_servers",
    ),
    "pingyitong.services.history_service": ("_literal_keyword_pattern", "HistoryService.search", "HistoryService.count"),
    "pingyitong.ui.app": (
        "Application.__init__", "Application._start_preload", "Application._begin_preload",
        "Application._warm_ocr_async", "_SerializedOcrEngine.warm_up", "_SerializedOcrEngine.recognize_frame",
        "Application._build_layers", "Application._on_config_changed", "Application._check_dependencies",
        # 区域离屏回退的用户提示（0.6.17）
        "Application._on_skip_notice",
    ),
    "pingyitong.ui.floating_bar": ("FloatingBar._setup_window",),
    "pingyitong.ui.widgets.transcript_panel": ("TranscriptPanel._setup_window",),
    "pingyitong.components.ocr.ocr_engine": (
        "RapidOcrBackend._create_engine", "RapidOcrBackend._recognize_crops",
        "OcrEngine.configure", "OcrEngine.backend", "OcrEngine.preflight_error",
        "OcrEngine.backend_name", "OcrEngine.error_message",
        "_join_segment_results", "_space_separated_word_boundary",
        # 贴边漏检与裁剪缺尾的修复（0.6.16）：字幕行贴屏底/行尾句号被切
        "_pad_edge", "_expand_boxes", "_clamp_rect",
    ),
    "pingyitong.components.ocr.recognizer_profiles": (
        "profile_for_source", "reverse_rtl_prediction",
    ),
    "pingyitong.core.pipeline": (
        "Pipeline.start", "Pipeline._ocr_loop", "Pipeline.apply_config",
        # 区域落在已断开显示器上的静默停摆修复（0.6.17）：回退整屏捕获
        "Pipeline._region_is_offscreen",
    ),
    "pingyitong.services.text_preprocess": (
        "_order_line_blocks", "TextPreprocessor._merge_group", "TextPreprocessor._assemble_text",
    ),
    "pingyitong.services.reading_order": (
        "_order_recursive", "_order_rows", "_Boxed.text", "_Column.text",
    ),
}


def walk_codes(code: types.CodeType):
    yield code
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from walk_codes(const)


def normalized(value):
    """Keep executable structure; ignore file names and line/column positions."""
    if isinstance(value, types.CodeType):
        return {
            "name": value.co_name,
            "qualname": value.co_qualname,
            "args": [value.co_argcount, value.co_posonlyargcount, value.co_kwonlyargcount],
            "flags": value.co_flags,
            "bytecode": value.co_code.hex(),
            "exceptions": value.co_exceptiontable.hex(),
            "names": value.co_names,
            "varnames": value.co_varnames,
            "freevars": value.co_freevars,
            "cellvars": value.co_cellvars,
            "constants": [normalized(item) for item in value.co_consts],
        }
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, tuple):
        return [normalized(item) for item in value]
    if isinstance(value, frozenset):
        return {"frozenset": [normalized(item) for item in sorted(value, key=repr)]}
    if value is Ellipsis:
        return {"ellipsis": True}
    if isinstance(value, complex):
        return {"complex": [value.real, value.imag]}
    return value


def fingerprint(code: types.CodeType) -> str:
    data = json.dumps(normalized(code), ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def embedded_version(code: types.CodeType) -> str | None:
    instructions = list(dis.get_instructions(code))
    for previous, current in zip(instructions, instructions[1:], strict=False):
        if (
            current.opname == "STORE_NAME"
            and current.argval == "__version__"
            and previous.opname == "LOAD_CONST"
            and isinstance(previous.argval, str)
        ):
            return previous.argval
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    exe = args.exe.resolve(strict=True)
    archive = CArchiveReader(str(exe))
    pyz = archive.open_embedded_archive("PYZ.pyz")
    version = embedded_version(pyz.extract("pingyitong.version"))
    with exe.open("rb") as handle:
        exe_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    report = {
        "exe": str(exe),
        "exe_sha256": exe_hash,
        "expected_version": args.version,
        "embedded_version": version,
        "version_matches": version == args.version,
        "functions": [],
    }

    for module, qualnames in CHECKS.items():
        try:
            packed_codes = {code.co_qualname: code for code in walk_codes(pyz.extract(module))}
        except KeyError:
            packed_codes = {}
        path = args.source_root / "src" / Path(*module.split(".")).with_suffix(".py")
        source = compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True, optimize=0)
        source_codes = {code.co_qualname: code for code in walk_codes(source)}
        for qualname in qualnames:
            packed = packed_codes.get(qualname)
            current = source_codes.get(qualname)
            packed_hash = fingerprint(packed) if packed else None
            source_hash = fingerprint(current) if current else None
            report["functions"].append({
                "module": module,
                "function": qualname,
                "present": packed is not None,
                "matches_source": packed is not None and packed_hash == source_hash,
                "packaged_fingerprint": packed_hash,
                "source_fingerprint": source_hash,
            })

    report["passed"] = report["version_matches"] and all(
        item["matches_source"] for item in report["functions"]
    )
    # Validate every product module, including caches and prefetch code outside
    # the older selected-function list. A current version number cannot prove
    # that an installer contains all of the accepted fixes.
    report["modules"] = []
    package_root = args.source_root / "src" / "pingyitong"
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(args.source_root / "src").with_suffix("")
        parts = list(relative.parts)
        if parts[-1] == "__init__":
            parts.pop()
        module = ".".join(parts)
        current = compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True, optimize=0)
        try:
            packaged = pyz.extract(module)
        except KeyError:
            packaged = None
        report["modules"].append({
            "module": module,
            "matches_source": packaged is not None and fingerprint(packaged) == fingerprint(current),
        })
    report["passed"] = report["passed"] and bool(report["modules"]) and all(
        item["matches_source"] for item in report["modules"]
    )
    # ONNX dictionaries are embedded in the recognition weights. Compare the
    # shipped files themselves, including the routing manifest, to prevent a
    # current EXE plus stale/missing assets from passing this release gate.
    bundled_assets = exe.parent / "_internal" / "assets" / "ocr"
    source_assets = args.source_root / "assets" / "ocr"
    report["ocr_assets"] = []
    for source in sorted(source_assets.rglob("*")):
        if not source.is_file():
            continue
        relative_path = source.relative_to(source_assets)
        bundled = bundled_assets / relative_path
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        bundled_hash = hashlib.sha256(bundled.read_bytes()).hexdigest() if bundled.is_file() else None
        report["ocr_assets"].append({
            "name": relative_path.as_posix(), "bytes": source.stat().st_size,
            "source_sha256": source_hash, "packaged_sha256": bundled_hash,
            "matches_source": source_hash == bundled_hash,
        })
    report["passed"] = report["passed"] and bool(report["ocr_assets"]) and all(
        item["matches_source"] for item in report["ocr_assets"]
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
