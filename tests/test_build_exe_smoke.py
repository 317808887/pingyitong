"""Regression tests for the PyInstaller onedir runtime smoke check."""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest

BUILD_SCRIPT = Path(__file__).resolve().parents[1] / "build" / "build_exe.py"
SPEC = importlib.util.spec_from_file_location("pingyitong_build_exe", BUILD_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BUILD_EXE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD_EXE)


def _create_bundle(root: Path, *, include_base_library: bool = True) -> Path:
    internal = root / "_internal"
    internal.mkdir(parents=True)
    (root / "PingYiTong.exe").write_bytes(b"exe")
    (internal / "python311.dll").write_bytes(b"dll")
    if include_base_library:
        with zipfile.ZipFile(internal / "base_library.zip", "w") as archive:
            for name in ("codecs.pyc", "encodings/__init__.pyc", "io.pyc"):
                archive.writestr(name, b"test")
    return root


def test_onedir_smoke_check_accepts_complete_runtime(tmp_path: Path) -> None:
    BUILD_EXE.validate_onedir_bundle(_create_bundle(tmp_path / "PingYiTong"))


def test_onedir_smoke_check_rejects_missing_base_library(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "PingYiTong", include_base_library=False)

    with pytest.raises(RuntimeError, match="base_library.zip"):
        BUILD_EXE.validate_onedir_bundle(bundle)


def test_onedir_smoke_check_rejects_corrupt_base_library(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path / "PingYiTong")
    (bundle / "_internal" / "base_library.zip").write_bytes(b"not a zip archive")

    with pytest.raises(RuntimeError, match="ZIP 完整性检查"):
        BUILD_EXE.validate_onedir_bundle(bundle)
