"""Manage local acceptance rounds without touching unregistered or personal data.

This is evidence bookkeeping, not a runner or a declaration that tests passed.
Only explicit completion records count towards retention and version comparisons.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SCHEMA = "pingyitong-acceptance-v1"
DIRECTORIES = ("source", "build", "install", "reports", "raw", "scratch")
RUN_PATTERN = re.compile(r"r\d{4,}-[A-Za-z0-9_.-]+-[a-f0-9]{8}\Z")
ACCEPTANCE_GATES = ("semantic", "readability", "completeness", "latency", "installed_identity")
COMPARISON_FIELDS = ("seed", "corpus_sha256", "model", "protocol_version")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: object) -> None:
    reject_links(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    reject_links(temporary)
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_link(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def delivery_records_complete(records: dict) -> bool:
    """Count an installed version only when all three identity records exist."""
    source = records.get("source", {})
    if (not isinstance(source, dict) or not source.get("manifest_sha256")
            or not isinstance(source.get("file_count"), int) or source["file_count"] <= 0):
        return False
    for kind in ("build", "install"):
        record = records.get(kind, {})
        if not isinstance(record, dict) or not record.get("files"):
            return False
        if not all(isinstance(item, dict) and item.get("path") and item.get("sha256")
                   for item in record["files"]):
            return False
    return True


def reject_links(path: Path, *, recursive: bool = False) -> None:
    for entry in (path, *path.parents):
        if (entry.exists() or entry.is_symlink()) and is_link(entry):
            raise ValueError(f"Refusing linked/reparse path: {entry}")
    if recursive and path.is_dir():
        for directory, folders, files in os.walk(path, followlinks=False):
            for name in folders + files:
                entry = Path(directory) / name
                if is_link(entry):
                    raise ValueError(f"Refusing linked/reparse entry: {entry}")


class Iterations:
    def __init__(self, root: Path):
        self.root = root.absolute()
        reject_links(self.root)
        self.registry = self.root / "registry" / "index.json"

    def index(self) -> dict:
        if not self.registry.exists():
            return {"schema": SCHEMA, "runs": []}
        reject_links(self.registry)
        data = read_json(self.registry)
        if data.get("schema") != SCHEMA:
            raise ValueError("Unrecognized iteration registry")
        return data

    def run(self, run_id: str) -> tuple[Path, dict]:
        if not RUN_PATTERN.fullmatch(run_id):
            raise ValueError("Invalid run id")
        entry = next((r for r in self.index()["runs"] if r["run_id"] == run_id), None)
        if entry is None or entry.get("directory_removed"):
            raise ValueError("Run is not registered or its directory was retired")
        path = self.root / "runs" / run_id
        reject_links(path)
        if path.resolve().parent != (self.root / "runs").resolve():
            raise ValueError("Run escaped managed runs directory")
        owner = read_json(path / "ownership.json")
        if owner != {"schema": SCHEMA, "run_id": run_id, "owner_token": entry["owner_token"]}:
            raise ValueError("Ownership marker mismatch")
        return path, entry

    def start(self, version: str, seed: int) -> dict:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", version):
            raise ValueError("Invalid version")
        index = self.index()
        number = max((r["number"] for r in index["runs"]), default=0) + 1
        token = uuid.uuid4().hex
        run_id = f"r{number:04d}-{version}-{token[:8]}"
        path = self.root / "runs" / run_id
        reject_links(path)
        path.mkdir(parents=True, exist_ok=False)
        for name in DIRECTORIES:
            (path / name).mkdir()
        (path / "scratch" / "test_home").mkdir()
        entry = {"run_id": run_id, "number": number, "version": version, "seed": seed,
                 "owner_token": token, "state": "active", "pinned": False, "created_at": utc_now()}
        write_json(path / "ownership.json", {"schema": SCHEMA, "run_id": run_id, "owner_token": token})
        write_json(path / "manifest.json", {**entry, "evidence_scope": "unverified", "records": {}})
        index["runs"].append(entry)
        write_json(self.registry, index)
        return {**entry, "path": str(path)}

    def active(self, run_id: str) -> tuple[Path, dict]:
        path, entry = self.run(run_id)
        if entry["state"] != "active":
            raise ValueError("Completed evidence is immutable; start another round")
        return path, entry

    def reset_scratch(self, run_id: str) -> dict:
        path, _ = self.active(run_id)
        scratch = path / "scratch"
        reject_links(scratch, recursive=True)
        if scratch.resolve().parent != path.resolve():
            raise ValueError("Scratch escaped run directory")
        shutil.rmtree(scratch)
        (scratch / "test_home").mkdir(parents=True)
        return {"run_id": run_id, "cleared": str(scratch), "reports_preserved": True}

    def freeze(self, run_id: str, project: Path) -> dict:
        path, _ = self.active(run_id)
        project = project.absolute()
        reject_links(project)
        destination = path / "source"
        reject_links(destination, recursive=True)
        if any(destination.iterdir()):
            raise ValueError("Source already frozen; a snapshot cannot be overwritten")
        candidates = []
        for folder in ("src", "tests", "scripts"):
            source = project / folder
            if source.exists():
                reject_links(source, recursive=True)
                candidates.extend(p for p in source.rglob("*") if p.is_file())
        for pattern in ("pyproject.toml", "requirements*.txt", "README*", "LICENSE*", "build/*.py", "build/*.iss"):
            candidates.extend(project.glob(pattern))
        entries = []
        for source in sorted(set(candidates)):
            relative = source.relative_to(project)
            if "__pycache__" in relative.parts or source.suffix in {".pyc", ".pyo"}:
                continue
            reject_links(source)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            digest = sha256(target)
            if sha256(source) != digest:
                raise ValueError(f"Source changed during snapshot: {relative}")
            entries.append({"path": relative.as_posix(), "bytes": target.stat().st_size, "sha256": digest})
        if not entries:
            raise ValueError("No source files were found")
        result = {"run_id": run_id, "created_at": utc_now(), "files": entries,
                  "scope": "src, tests, scripts, build scripts and dependency metadata; external models/assets are not copied"}
        write_json(path / "reports" / "source_manifest.json", result)
        self._record(path, "source", {"manifest_sha256": sha256(path / "reports" / "source_manifest.json"),
                                      "file_count": len(entries)})
        return {"run_id": run_id, "files": len(entries), "source": str(destination)}

    def _record(self, path: Path, kind: str, record: dict) -> None:
        manifest = read_json(path / "manifest.json")
        manifest["records"][kind] = record
        write_json(path / "manifest.json", manifest)

    def record_files(self, run_id: str, kind: str, files: list[Path], *, copy: bool = False) -> dict:
        path, _ = self.active(run_id)
        if kind not in {"build", "install"}:
            raise ValueError("Record kind must be build or install")
        if (path / kind / "files.json").exists():
            raise ValueError("Artifact record already exists; evidence cannot be overwritten")
        entries = []
        for source in files:
            source = source.absolute()
            reject_links(source)
            if not source.is_file():
                raise ValueError(f"Not a file: {source}")
            record = {"path": str(source), "bytes": source.stat().st_size, "sha256": sha256(source)}
            if copy:
                target = path / kind / source.name
                reject_links(target)
                if target.exists():
                    raise ValueError(f"Evidence already exists: {target}")
                shutil.copy2(source, target)
                if sha256(target) != record["sha256"]:
                    raise ValueError("Copied artifact changed during recording")
                record["managed_path"] = str(target.relative_to(path))
            entries.append(record)
        result = {"recorded_at": utc_now(), "files": entries}
        write_json(path / kind / "files.json", result)
        self._record(path, kind, result)
        return result

    def complete(self, run_id: str, summary: dict) -> dict:
        path, entry = self.active(run_id)
        if summary.get("outcome") not in {"passed", "failed", "partial"}:
            raise ValueError("Summary requires an honest outcome: passed, failed or partial")
        if not isinstance(summary.get("metrics"), dict) or not isinstance(summary.get("limitations"), list):
            raise ValueError("Summary needs metrics object and limitations array")
        if summary["outcome"] == "passed" and not summary.get("native_installed_verified"):
            raise ValueError("Passed delivery requires explicit installed native verification")
        records = read_json(path / "manifest.json")["records"]
        delivery_recorded = delivery_records_complete(records)
        if summary["outcome"] == "passed":
            gates = summary.get("acceptance_gates")
            if (not isinstance(gates, dict) or any(gates.get(key) is not True for key in ACCEPTANCE_GATES)
                    or any(value is not True for value in gates.values())):
                raise ValueError("Passed delivery requires every explicit acceptance gate to be true; use partial or failed")
            if not delivery_recorded:
                raise ValueError("Passed delivery requires frozen source, build and installation identities")
        result = {**summary, "schema": SCHEMA, "run_id": run_id, "version": entry["version"],
                  "seed": entry["seed"], "delivery_recorded": delivery_recorded, "completed_at": utc_now()}
        write_json(path / "reports" / "summary.json", result)
        write_json(self.root / "registry" / "summaries" / f"{run_id}.json", result)
        manifest = read_json(path / "manifest.json")
        manifest.update(state="completed", outcome=result["outcome"], completed_at=result["completed_at"],
                        delivery_recorded=delivery_recorded,
                        evidence_scope="acceptance_gates_passed" if result["outcome"] == "passed" else "acceptance_incomplete")
        write_json(path / "manifest.json", manifest)
        index = self.index()
        for item in index["runs"]:
            if item["run_id"] == run_id:
                item.update(state="completed", outcome=result["outcome"], completed_at=result["completed_at"],
                            delivery_recorded=delivery_recorded)
        write_json(self.registry, index)
        return result

    def compare(self, count: int = 3) -> dict:
        completed = [r for r in self.index()["runs"] if r["state"] == "completed"]
        latest = []
        versions = set()
        for entry in reversed(completed):
            if entry["version"] not in versions:
                latest.append(read_json(self.root / "registry" / "summaries" / f"{entry['run_id']}.json"))
                versions.add(entry["version"])
            if len(latest) >= count:
                break
        latest.reverse()
        reasons = []
        if len(latest) != count:
            reasons.append("Requested number of distinct versions is unavailable")
        for field in COMPARISON_FIELDS:
            if any(field not in row or row[field] is None or row[field] == ""
                   or (isinstance(row[field], (dict, list)) and not row[field]) for row in latest):
                reasons.append(f"Missing comparison field: {field}")
            elif len({json.dumps(row[field], ensure_ascii=False, sort_keys=True) for row in latest}) != 1:
                reasons.append(f"Different comparison field: {field}")
        return {"requested_versions": count, "available_versions": len(latest), "rounds": latest,
                "comparable_seed": len({r["seed"] for r in latest}) == 1 if latest else False,
                "protocol_comparable": bool(latest) and not reasons, "comparison_issues": reasons,
                "note": "Different scene seeds, models, hardware or metric definitions require separate cohorts; no automatic stability claim."}

    def _delivery_recorded(self, entry: dict) -> bool:
        if "delivery_recorded" in entry:
            return entry["delivery_recorded"] is True
        # Legacy summaries remain immutable. Their manifests supply only the
        # version-retention identity, never an inferred acceptance result.
        if entry.get("directory_removed"):
            return False
        path, _ = self.run(entry["run_id"])
        return delivery_records_complete(read_json(path / "manifest.json").get("records", {}))

    def retention_plan(self) -> dict:
        index = self.index()
        completed = [r for r in index["runs"] if r["state"] == "completed"]
        latest_two = {r["run_id"] for r in completed[-2:]}
        delivered = [entry for entry in completed if self._delivery_recorded(entry)]
        first_deliveries = {}
        for entry in delivered:
            first_deliveries.setdefault(entry["version"], entry["run_id"])
        latest_five_versions = set(list(first_deliveries)[-5:])
        five_round_boundary = (bool(completed) and bool(first_deliveries) and len(first_deliveries) % 5 == 0
                               and completed[-1]["run_id"] == first_deliveries.get(completed[-1]["version"]))
        delivered_ids = {entry["run_id"] for entry in delivered}
        targets = []
        for entry in completed:
            if entry.get("pinned") or entry.get("directory_removed"):
                continue
            path, _ = self.run(entry["run_id"])
            if (five_round_boundary and entry["run_id"] in delivered_ids and entry["run_id"] not in latest_two
                    and entry["version"] not in latest_five_versions):
                target, kind = path, "round"
            elif entry["run_id"] not in latest_two and (path / "raw").exists():
                target, kind = path / "raw", "raw"
            else:
                continue
            reject_links(target, recursive=True)
            targets.append({"run_id": entry["run_id"], "kind": kind,
                            "relative_path": target.relative_to(self.root).as_posix(),
                            "owner_token": entry["owner_token"]})
        return {"schema": SCHEMA, "root": str(self.root), "registry_sha256": sha256(self.registry) if self.registry.exists() else None,
                "completed_rounds": len(completed), "five_round_boundary": five_round_boundary,
                "completed_delivery_versions": len(first_deliveries), "delivery_versions": list(first_deliveries),
                "targets": targets, "summaries_preserved": True}

    def prune(self, plan: dict) -> dict:
        current = self.retention_plan()
        if plan != current:
            raise ValueError("Retention plan changed or is not an exact managed plan; regenerate it")
        # Preflight EVERY target before deleting anything, including all descendants.
        verified = []
        for target in plan["targets"]:
            path, _ = self.run(target["run_id"])
            selected = path if target["kind"] == "round" else path / "raw"
            if selected.relative_to(self.root).as_posix() != target["relative_path"]:
                raise ValueError("Unexpected cleanup target")
            reject_links(selected, recursive=True)
            verified.append((selected, target))
        index = self.index()
        deleted = []
        for selected, target in verified:
            reject_links(selected, recursive=True)
            shutil.rmtree(selected)
            deleted.append(target)
            for entry in index["runs"]:
                if entry["run_id"] == target["run_id"]:
                    entry["directory_removed" if target["kind"] == "round" else "raw_removed"] = True
        write_json(self.registry, index)
        audit = {"at": utc_now(), "targets": deleted, "summaries_preserved": True}
        write_json(self.root / "registry" / f"cleanup-{uuid.uuid4().hex[:12]}.json", audit)
        return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT / "managed_iterations")
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start")
    start.add_argument("--version", required=True)
    start.add_argument("--seed", type=int, required=True)
    for name in ("freeze", "reset-scratch", "record", "complete"):
        command = sub.add_parser(name)
        command.add_argument("--run", required=True)
        if name == "freeze":
            command.add_argument("--project", type=Path, default=PROJECT)
        elif name == "record":
            command.add_argument("--kind", choices=("build", "install"), required=True)
            command.add_argument("--files", type=Path, nargs="+", required=True)
            command.add_argument("--copy", action="store_true")
        elif name == "complete":
            command.add_argument("--summary", type=Path, required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--count", type=int, choices=(3, 5, 10), default=3)
    plan = sub.add_parser("retention-plan")
    plan.add_argument("--output", type=Path, required=True)
    prune = sub.add_parser("prune")
    prune.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    manager = Iterations(args.root)
    if args.command == "start":
        result = manager.start(args.version, args.seed)
    elif args.command == "freeze":
        result = manager.freeze(args.run, args.project)
    elif args.command == "reset-scratch":
        result = manager.reset_scratch(args.run)
    elif args.command == "record":
        result = manager.record_files(args.run, args.kind, args.files, copy=args.copy)
    elif args.command == "complete":
        result = manager.complete(args.run, read_json(args.summary))
    elif args.command == "compare":
        result = manager.compare(args.count)
    elif args.command == "retention-plan":
        result = manager.retention_plan()
        write_json(args.output, result)
    else:
        result = manager.prune(read_json(args.plan))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
