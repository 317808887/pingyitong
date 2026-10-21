from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


manager_module = load_script("iteration_manager")
scene_module = load_script("generate_acceptance_scenes")


def finish(manager, run):
    manager.complete(run["run_id"], {"outcome": "partial", "metrics": {"latency_p95_ms": 700},
                                      "limitations": ["Native installation not tested"]})


def record_delivery(manager, run):
    """Small identity fixtures; no executable is launched or installed."""
    project = manager.root.parent / "project"
    (project / "src").mkdir(parents=True, exist_ok=True)
    (project / "src" / "main.py").write_text("answer = 42")
    manager.freeze(run["run_id"], project)
    for kind in ("build", "install"):
        evidence = project / f"{kind}-{run['run_id']}.json"
        evidence.write_text(json.dumps({"version": run["version"], "kind": kind}))
        manager.record_files(run["run_id"], kind, [evidence])


def passing_summary():
    return {"outcome": "passed", "native_installed_verified": True, "metrics": {}, "limitations": [],
            "acceptance_gates": dict.fromkeys(manager_module.ACCEPTANCE_GATES, True)}


def test_rounds_are_unique_and_scratch_reset_preserves_evidence(tmp_path):
    manager = manager_module.Iterations(tmp_path / "managed")
    first, second = manager.start("0.6.34", 13), manager.start("0.6.34", 13)
    assert first["run_id"] != second["run_id"]
    folder = Path(first["path"])
    (folder / "scratch" / "test_home" / "cache.sqlite").write_bytes(b"old cache")
    (folder / "reports" / "review.txt").write_text("observed failure")
    (folder / "raw" / "output.txt").write_text("preserved output")
    manager.reset_scratch(first["run_id"])
    assert list((folder / "scratch" / "test_home").iterdir()) == []
    assert (folder / "reports" / "review.txt").read_text() == "observed failure"
    assert (folder / "raw" / "output.txt").exists()


def test_snapshot_is_immutable_and_excludes_models_personal_data(tmp_path):
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / "src" / "main.py").write_text("answer = 42")
    (project / "models").mkdir()
    (project / "models" / "private.bin").write_bytes(b"model")
    (project / "config.json").write_text("private setting")
    manager = manager_module.Iterations(tmp_path / "managed")
    run = manager.start("0.6.34", 3)
    result = manager.freeze(run["run_id"], project)
    assert result["files"] == 1
    snapshot = Path(run["path"]) / "source"
    assert (snapshot / "src" / "main.py").read_text() == "answer = 42"
    assert not (snapshot / "config.json").exists()
    (project / "src" / "main.py").write_text("changed")
    with pytest.raises(ValueError, match="already frozen"):
        manager.freeze(run["run_id"], project)


def test_keep_two_raw_sets_and_ten_round_boundary_keeps_five_directories(tmp_path):
    manager = manager_module.Iterations(tmp_path / "managed")
    unknown = tmp_path / "managed" / "runs" / "personal_unregistered"
    unknown.mkdir(parents=True)
    (unknown / "keep.txt").write_text("personal")
    rounds = []
    for number in range(1, 11):
        run = manager.start(f"0.6.{number}", 23)
        (Path(run["path"]) / "raw" / "result.txt").write_text(f"run {number}")
        record_delivery(manager, run)
        finish(manager, run)
        rounds.append(run)
    active = manager.start("0.6.11", 24)
    plan = manager.retention_plan()
    assert plan["five_round_boundary"]
    assert sum(target["kind"] == "round" for target in plan["targets"]) == 5
    assert sum(target["kind"] == "raw" for target in plan["targets"]) == 3
    manager.prune(plan)
    assert all(not Path(run["path"]).exists() for run in rounds[:5])
    assert all(not (Path(run["path"]) / "raw").exists() for run in rounds[5:8])
    assert all((Path(run["path"]) / "raw" / "result.txt").exists() for run in rounds[8:])
    assert Path(active["path"]).exists()
    assert (unknown / "keep.txt").read_text() == "personal"
    assert len(list((manager.root / "registry" / "summaries").glob("*.json"))) == 10
    assert manager.compare(3)["available_versions"] == 3


def test_stale_or_tampered_cleanup_plan_never_deletes(tmp_path):
    manager = manager_module.Iterations(tmp_path / "managed")
    first = manager.start("0.6.1", 23)
    finish(manager, first)
    for number in range(2, 4):
        finish(manager, manager.start(f"0.6.{number}", 23))
    plan = manager.retention_plan()
    plan["targets"][0]["relative_path"] = "../personal"
    with pytest.raises(ValueError, match="exact managed plan"):
        manager.prune(plan)
    assert (Path(first["path"]) / "raw").exists()
    plan = manager.retention_plan()
    manager.start("0.6.4", 24)
    with pytest.raises(ValueError, match="plan changed"):
        manager.prune(plan)
    assert (Path(first["path"]) / "raw").exists()


def test_linked_descendant_blocks_cleanup_before_any_deletion(tmp_path):
    manager = manager_module.Iterations(tmp_path / "managed")
    rounds = []
    for number in range(4):
        run = manager.start(f"0.6.{number}", 23)
        finish(manager, run)
        rounds.append(run)
    outside = tmp_path / "personal"
    outside.mkdir()
    (outside / "important.txt").write_text("keep")
    link = Path(rounds[1]["path"]) / "raw" / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        # Windows directory junctions are also reparse points and generally do
        # not require Developer Mode. Create only; never delete through the link.
        environment = {**os.environ, "ACCEPTANCE_TEST_LINK": str(link), "ACCEPTANCE_TEST_TARGET": str(outside)}
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                        "$ErrorActionPreference = 'Stop'\nNew-Item -ItemType Junction -Path $env:ACCEPTANCE_TEST_LINK -Target $env:ACCEPTANCE_TEST_TARGET | Out-Null"],
                       env=environment, check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    with pytest.raises(ValueError, match="linked/reparse"):
        manager.retention_plan()
    assert (Path(rounds[0]["path"]) / "raw").exists()
    assert (outside / "important.txt").read_text() == "keep"
    # Unlink the junction itself. No recursive operation is issued on its target.
    if os.name == "nt" and not link.is_symlink():
        os.rmdir(link)
    else:
        link.unlink()


def test_missing_owner_or_traversal_cannot_select_managed_run(tmp_path):
    manager = manager_module.Iterations(tmp_path / "managed")
    run = manager.start("0.6.34", 2)
    with pytest.raises(ValueError, match="Invalid run"):
        manager.reset_scratch("../../personal")
    marker = Path(run["path"]) / "ownership.json"
    marker.write_text(json.dumps({"schema": "wrong"}))
    with pytest.raises(ValueError, match="Ownership"):
        manager.reset_scratch(run["run_id"])


def test_component_results_cannot_be_declared_native_pass(tmp_path):
    manager = manager_module.Iterations(tmp_path / "managed")
    run = manager.start("0.6.34", 2)
    with pytest.raises(ValueError, match="native verification"):
        manager.complete(run["run_id"], {"outcome": "passed", "metrics": {}, "limitations": []})
    finish(manager, run)
    with pytest.raises(ValueError, match="immutable"):
        manager.reset_scratch(run["run_id"])


def test_same_seed_reproduces_directions_and_inputs_across_runs():
    first = scene_module.generate(311, 9, "first")
    second = scene_module.generate(311, 9, "second")
    assert first["corpus_sha256"] == second["corpus_sha256"]
    assert first["corpus_sha256"] != scene_module.generate(312, 9, "first")["corpus_sha256"]
    assert len({s["source_language"] for s in first["scenes"]}) == 6
    assert any(s["source_language"] not in {"en", "zh"} and s["target_language"] not in {"en", "zh"}
               for s in first["scenes"])
    assert {s["category"] for s in first["scenes"]} == {"subtitle", "dense_page", "technical_document"}
    assert all(s["source_language"] != s["target_language"] for s in first["scenes"])
    assert all(row["review"] == "pending" for s in first["scenes"] for row in s["rows"])


def test_passing_comparison_never_invents_missing_versions(tmp_path):
    manager = manager_module.Iterations(tmp_path / "managed")
    for seed in (1, 2):
        finish(manager, manager.start("0.6.34", seed))
    result = manager.compare(3)
    assert result["available_versions"] == 1
    assert result["rounds"][0]["seed"] == 2
    assert result["protocol_comparable"] is False


@pytest.mark.parametrize("missing_gate", [None, *manager_module.ACCEPTANCE_GATES])
def test_native_boolean_alone_cannot_bypass_acceptance_gates(tmp_path, missing_gate):
    manager = manager_module.Iterations(tmp_path / "managed")
    run = manager.start("0.6.37", 2)
    record_delivery(manager, run)
    summary = passing_summary()
    if missing_gate is None:
        del summary["acceptance_gates"]
    else:
        summary["acceptance_gates"][missing_gate] = False
    with pytest.raises(ValueError, match="acceptance gate"):
        manager.complete(run["run_id"], summary)
    assert manager.index()["runs"][0]["state"] == "active"
    summary["outcome"] = "partial"
    manager.complete(run["run_id"], summary)


@pytest.mark.parametrize("value", [1, "true", None])
def test_gate_requires_boolean_true_not_truthy_value(tmp_path, value):
    manager = manager_module.Iterations(tmp_path / "managed")
    run = manager.start("0.6.37", 2)
    record_delivery(manager, run)
    summary = passing_summary()
    summary["acceptance_gates"]["latency"] = value
    with pytest.raises(ValueError, match="acceptance gate"):
        manager.complete(run["run_id"], summary)


def test_omitted_named_gate_and_additional_failed_gate_cannot_hide_failure(tmp_path):
    manager = manager_module.Iterations(tmp_path / "managed")
    run = manager.start("0.6.37", 2)
    record_delivery(manager, run)
    summary = passing_summary()
    del summary["acceptance_gates"]["semantic"]
    with pytest.raises(ValueError, match="acceptance gate"):
        manager.complete(run["run_id"], summary)
    summary = passing_summary()
    summary["acceptance_gates"]["refresh_stability"] = False
    with pytest.raises(ValueError, match="acceptance gate"):
        manager.complete(run["run_id"], summary)


def test_all_explicit_gates_and_identities_allow_pass_without_rewriting_history(tmp_path):
    manager = manager_module.Iterations(tmp_path / "managed")
    legacy = manager.start("0.6.36", 2)
    finish(manager, legacy)
    historical = Path(legacy["path"]) / "reports" / "summary.json"
    previous_bytes = historical.read_bytes()
    run = manager.start("0.6.37", 2)
    with pytest.raises(ValueError, match="source, build and installation"):
        manager.complete(run["run_id"], passing_summary())
    record_delivery(manager, run)
    result = manager.complete(run["run_id"], passing_summary())
    assert result["outcome"] == "passed"
    assert result["delivery_recorded"] is True
    manifest = manager_module.read_json(Path(run["path"]) / "manifest.json")
    assert manifest["state"] == "completed"
    assert manifest["evidence_scope"] == "acceptance_gates_passed"
    assert historical.read_bytes() == previous_bytes


@pytest.mark.parametrize("changed_field", [None, "seed", "corpus_sha256", "model", "protocol_version"])
def test_comparison_requires_same_complete_protocol(tmp_path, changed_field):
    manager = manager_module.Iterations(tmp_path / "managed")
    for number in range(3):
        seed = 17 if number != 2 or changed_field != "seed" else 18
        run = manager.start(f"0.6.{number}", seed)
        summary = {"outcome": "partial", "metrics": {}, "limitations": [],
                   "corpus_sha256": "corpus-17", "model": {"id": "4b", "temperature": 0},
                   "protocol_version": "native-v2"}
        if number == 2 and changed_field and changed_field != "seed":
            summary[changed_field] = "different"
        manager.complete(run["run_id"], summary)
    comparison = manager.compare(3)
    assert comparison["protocol_comparable"] is (changed_field is None)
    assert "stable" not in comparison
    if changed_field:
        assert f"Different comparison field: {changed_field}" in comparison["comparison_issues"]


def test_equal_seeds_without_corpus_model_protocol_cannot_be_compared(tmp_path):
    manager = manager_module.Iterations(tmp_path / "managed")
    for number in range(3):
        finish(manager, manager.start(f"0.6.{number}", 17))
    comparison = manager.compare(3)
    assert comparison["comparable_seed"] is True
    assert comparison["protocol_comparable"] is False
    assert len(comparison["comparison_issues"]) == 3


def test_uninstalled_attempts_and_version_reruns_do_not_advance_five_version_cycle(tmp_path):
    manager = manager_module.Iterations(tmp_path / "managed")
    for number in range(1, 5):
        run = manager.start(f"0.6.{number}", 17)
        record_delivery(manager, run)
        finish(manager, run)
    for _ in range(6):
        finish(manager, manager.start("uninstalled-candidate", 17))
    plan = manager.retention_plan()
    assert plan["completed_delivery_versions"] == 4
    assert plan["five_round_boundary"] is False
    assert all(target["kind"] == "raw" for target in plan["targets"])
    fifth = manager.start("0.6.5", 17)
    record_delivery(manager, fifth)
    finish(manager, fifth)
    assert manager.retention_plan()["five_round_boundary"] is True
    rerun = manager.start("0.6.5", 18)
    record_delivery(manager, rerun)
    finish(manager, rerun)
    plan = manager.retention_plan()
    assert plan["completed_delivery_versions"] == 5
    assert plan["five_round_boundary"] is False
    assert not any(target["run_id"] in {fifth["run_id"], rerun["run_id"]} for target in plan["targets"])


def test_old_version_rerun_in_latest_two_raw_sets_is_protected_at_boundary(tmp_path):
    manager = manager_module.Iterations(tmp_path / "managed")
    for number in range(1, 10):
        run = manager.start(f"0.6.{number}", 17)
        record_delivery(manager, run)
        finish(manager, run)
    rerun = manager.start("0.6.1", 17)
    record_delivery(manager, rerun)
    finish(manager, rerun)
    tenth = manager.start("0.6.10", 17)
    record_delivery(manager, tenth)
    finish(manager, tenth)
    plan = manager.retention_plan()
    assert plan["completed_delivery_versions"] == 10
    assert plan["five_round_boundary"] is True
    assert all(target["run_id"] not in {rerun["run_id"], tenth["run_id"]} for target in plan["targets"])
