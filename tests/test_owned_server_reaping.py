"""服务器归属回归：模拟 kill 不接触系统进程。"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import psutil as real_psutil

from pingyitong.utils import process_guard


class FakeProcess:
    def __init__(self, pid, executable, created=200.0, owner_pid=101, parent=None):
        self.pid = pid
        self.executable = str(executable)
        self.created = created
        self.owner_pid = owner_pid
        self.parent_value = parent
        self.alive = True
        self.killed = 0
        self.info = {"pid": pid, "exe": self.executable, "name": Path(executable).name}

    def create_time(self):
        return self.created

    def exe(self):
        return self.executable

    def name(self):
        return Path(self.executable).name

    def ppid(self):
        return self.owner_pid

    def parent(self):
        if isinstance(self.parent_value, BaseException):
            raise self.parent_value
        return self.parent_value

    def is_running(self):
        return self.alive

    def kill(self):
        self.killed += 1
        self.alive = False


class TestOwnedServerReaping(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pyt_owned_guard_")
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.runtime = self.home / "runtime_binary"
        self.runtime.mkdir()
        self.executable = self.runtime / "llama-server.exe"
        self.executable.touch()
        self.child = FakeProcess(202, self.executable)
        self.owner = FakeProcess(101, self.home / "python.exe", created=100.0)
        self.processes = {202: self.child, 101: self.owner}

        def lookup(pid):
            value = self.processes.get(pid)
            if value is None:
                raise real_psutil.NoSuchProcess(pid)
            if isinstance(value, BaseException):
                raise value
            return value

        self.fake = types.SimpleNamespace(
            Process=lookup,
            process_iter=lambda attrs: [self.child],
            NoSuchProcess=real_psutil.NoSuchProcess,
            AccessDenied=real_psutil.AccessDenied,
        )
        for context in (
            patch.dict(sys.modules, {"psutil": self.fake}),
            patch.object(process_guard.sys, "platform", "win32"),
            patch.object(process_guard.paths, "app_data_dir", return_value=self.home),
        ):
            context.start()
            self.addCleanup(context.stop)

    def record(self, **overrides):
        directory = process_guard._ownership_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "server.json"
        value = {
            "schema": 1,
            "pid": 202,
            "created": 200.0,
            "executable": str(self.executable.resolve()),
            "owner_pid": 101,
            "owner_created": 100.0,
        }
        value.update(overrides)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def reap(self):
        return process_guard.reap_stale_servers(self.executable)

    def test_same_runtime_owned_server_with_live_owner_survives(self):
        self.record()
        self.child.parent_value = self.owner
        self.assertEqual(self.reap(), [])
        self.assertEqual(self.child.killed, 0)

    def test_unrecorded_manual_server_with_live_parent_survives(self):
        self.child.parent_value = self.owner
        self.assertEqual(self.reap(), [])
        self.assertEqual(self.child.killed, 0)

    def test_unrecorded_detached_manual_server_survives(self):
        self.processes.pop(101)
        self.assertEqual(self.reap(), [])
        self.assertEqual(self.child.killed, 0)

    def test_confirmed_owned_orphan_is_reaped_and_record_removed(self):
        path = self.record()
        self.processes.pop(101)
        self.assertEqual(self.reap(), [202])
        self.assertEqual(self.child.killed, 1)
        self.assertFalse(path.exists())

    def test_reused_child_pid_survives_and_stale_record_is_removed(self):
        path = self.record()
        self.child.created = 300.0
        self.processes.pop(101)
        self.assertEqual(self.reap(), [])
        self.assertEqual(self.child.killed, 0)
        self.assertFalse(path.exists())

    def test_reused_owner_pid_does_not_hide_confirmed_orphan(self):
        self.record()
        self.owner.created = 300.0
        # psutil.parent checks owner creation time > child and returns None.
        self.assertEqual(self.reap(), [202])

    def test_access_denied_owner_identity_keeps_server(self):
        self.record()
        self.processes[101] = real_psutil.AccessDenied(101)
        self.assertEqual(self.reap(), [])
        self.assertEqual(self.child.killed, 0)

    def test_access_denied_current_parent_keeps_server(self):
        self.record()
        self.processes.pop(101)
        self.child.parent_value = real_psutil.AccessDenied(202)
        self.assertEqual(self.reap(), [])
        self.assertEqual(self.child.killed, 0)

    def test_access_denied_child_keeps_record(self):
        path = self.record()
        self.processes[202] = real_psutil.AccessDenied(202)
        self.assertEqual(self.reap(), [])
        self.assertTrue(path.exists())

    def test_unexpected_owner_identity_error_keeps_server_and_record(self):
        path = self.record()
        self.processes[101] = RuntimeError("identity unavailable")
        self.assertEqual(self.reap(), [])
        self.assertEqual(self.child.killed, 0)
        self.assertTrue(path.exists())

    def test_exited_child_record_is_removed(self):
        path = self.record()
        self.processes.pop(202)
        self.assertEqual(self.reap(), [])
        self.assertFalse(path.exists())

    def test_pid_reused_during_final_check_does_not_get_killed(self):
        path = self.record()
        self.processes.pop(101)
        self.child.alive = False
        self.assertEqual(self.reap(), [])
        self.assertEqual(self.child.killed, 0)
        self.assertFalse(path.exists())

    def test_kill_access_denied_keeps_record(self):
        path = self.record()
        self.processes.pop(101)
        with patch.object(self.child, "kill", side_effect=real_psutil.AccessDenied(202)):
            self.assertEqual(self.reap(), [])
        self.assertTrue(path.exists())

    def test_wrong_actual_executable_in_same_runtime_is_not_owned(self):
        self.record()
        self.processes.pop(101)
        self.child.executable = str(self.runtime / "other.exe")
        self.assertEqual(self.reap(), [])
        self.assertEqual(self.child.killed, 0)

    def test_different_runtime_record_survives(self):
        self.record(executable=str(self.home / "other_runtime/llama-server.exe"))
        self.processes.pop(101)
        self.assertEqual(self.reap(), [])
        self.assertEqual(self.child.killed, 0)

    def test_wrong_original_parent_identity_is_not_owned(self):
        self.record()
        self.processes.pop(101)
        self.child.owner_pid = 555
        self.assertEqual(self.reap(), [])
        self.assertEqual(self.child.killed, 0)

    def test_corrupt_partial_unknown_record_never_authorizes_kill(self):
        path = self.record()
        self.processes.pop(101)
        for value in ("{", "{}", '{"schema":99}', '{"schema":1,"pid":true}'):
            path.write_text(value, encoding="utf-8")
            self.assertEqual(self.reap(), [])
            self.assertEqual(self.child.killed, 0)

    def test_invalid_complete_identity_never_authorizes_kill(self):
        self.processes.pop(101)
        for invalid in (
            {"schema": True},
            {"pid": True},
            {"owner_pid": -1},
            {"created": float("nan")},
            {"created": float("inf")},
            {"owner_created": 201.0},
            {"executable": "llama-server.exe"},
        ):
            with self.subTest(invalid=invalid):
                self.record(**invalid)
                self.assertEqual(self.reap(), [])
                self.assertEqual(self.child.killed, 0)

    def test_oversized_record_never_authorizes_kill(self):
        path = self.record()
        self.processes.pop(101)
        path.write_text(path.read_text(encoding="utf-8") + " " * 4096, encoding="utf-8")
        self.assertEqual(self.reap(), [])
        self.assertEqual(self.child.killed, 0)

    def test_record_directory_error_never_authorizes_kill(self):
        self.record()
        self.processes.pop(101)
        with patch.object(process_guard, "_ownership_dir", side_effect=OSError("unavailable")):
            self.assertEqual(self.reap(), [])
        self.assertEqual(self.child.killed, 0)

    def test_non_windows_does_not_reap(self):
        path = self.record()
        self.processes.pop(101)
        with patch.object(process_guard.sys, "platform", "linux"):
            self.assertEqual(self.reap(), [])
        self.assertEqual(self.child.killed, 0)
        self.assertTrue(path.exists())

    def test_parent_appears_between_checks_and_is_protected(self):
        self.record()
        self.processes.pop(101)
        self.child.parent_value = self.owner
        self.assertEqual(self.reap(), [])
        self.assertEqual(self.child.killed, 0)

    def test_owner_recheck_is_conservative(self):
        self.record()
        with patch.object(process_guard, "_owner_alive", side_effect=[False, True]) as verify:
            self.assertEqual(self.reap(), [])
        self.assertEqual(verify.call_count, 2)
        self.assertEqual(self.child.killed, 0)

    def test_records_only_actual_child_of_current_launcher(self):
        self.owner.pid = os.getpid()
        self.child.parent_value = self.owner
        self.child.owner_pid = self.owner.pid
        process_guard._record_owned_server(self.child.pid)
        records = list(process_guard._ownership_dir().glob("*.json"))
        self.assertEqual(len(records), 1)
        value = json.loads(records[0].read_text(encoding="utf-8"))
        self.assertEqual(value["pid"], self.child.pid)
        self.assertEqual(value["owner_pid"], os.getpid())
        self.assertEqual(value["created"], self.child.created)
        self.assertEqual(list(process_guard._ownership_dir().glob("*.tmp")), [])

    def test_other_launcher_child_is_not_recorded(self):
        self.child.parent_value = self.owner
        process_guard._record_owned_server(self.child.pid)
        self.assertEqual(list(process_guard._ownership_dir().glob("*.json")), [])

    def test_non_llama_child_is_not_recorded(self):
        self.owner.pid = os.getpid()
        self.child.parent_value = self.owner
        self.child.executable = str(self.home / "python.exe")
        process_guard._record_owned_server(self.child.pid)
        self.assertEqual(list(process_guard._ownership_dir().glob("*.json")), [])

    def test_job_assignment_failure_still_records_actual_owned_server(self):
        self.owner.pid = os.getpid()
        self.child.parent_value = self.owner
        with patch.object(process_guard, "create_kill_on_close_job", return_value=None):
            self.assertFalse(process_guard.assign_to_job(self.child.pid))
        self.assertEqual(len(list(process_guard._ownership_dir().glob("*.json"))), 1)

    def test_record_write_failure_cannot_break_job_assignment(self):
        self.owner.pid = os.getpid()
        self.child.parent_value = self.owner
        with (
            patch.object(process_guard, "_ownership_dir", side_effect=OSError("read only")),
            patch.object(process_guard, "create_kill_on_close_job", return_value=None),
        ):
            self.assertFalse(process_guard.assign_to_job(self.child.pid))


if __name__ == "__main__":
    unittest.main(verbosity=2)
