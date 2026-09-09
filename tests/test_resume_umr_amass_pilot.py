"""Short resume-controller contract tests; no simulator or retarget workers."""
from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts import resume_umr_amass_pilot as resume


def planned_rows():
    return [{"origin_id": f"KIT/actor/motion{index}", "source": f"/raw/motion{index}.npz",
             "body_model": "/licensed/model.npz"} for index in range(40)]


def prefix_fixture(count=2, *, unfinished=True):
    plan = {"rows": planned_rows()}
    states = [{"origin_id": row["origin_id"], "result": "COMPLETE", "stage": "complete",
               "jobs": [{"stage": stage, "result": "COMPLETE"} for stage in ("prepare", "retarget")]}
              for row in plan["rows"][:count]]
    if unfinished:
        states.append({"origin_id": plan["rows"][count]["origin_id"], "stage": "retarget",
                       "result": "INTERRUPTED_BY_USER", "jobs": [
                           {"stage": "prepare", "result": "COMPLETE"},
                           {"stage": "retarget", "result": "INTERRUPTED_BY_USER"}]})
    return plan, {"result": "PAUSED_BY_USER", "motions_completed": count, "motions_requested": 40,
                  "training_started": False, "automatic_promotion": False, "motions": states,
                  "unit": "bfm-umr-pilot-old.service"}


class CompletedPrefixTests(unittest.TestCase):
    def test_accepts_only_complete_prefix_with_optional_interrupted_tail(self):
        for count, unfinished in ((0, False), (0, True), (2, False), (30, True), (39, True)):
            with self.subTest(count=count, unfinished=unfinished):
                plan, status = prefix_fixture(count, unfinished=unfinished)
                self.assertEqual(resume.completed_prefix(plan, status), count)

    def test_failed_tail_can_be_resumed_without_reclassifying_it_as_complete(self):
        plan, status = prefix_fixture()
        status["result"] = status["motions"][-1]["result"] = "FAIL"
        self.assertEqual(resume.completed_prefix(plan, status), 2)

    def test_rejects_nonprefix_reordered_duplicate_and_extra_origin(self):
        plan, original = prefix_fixture()
        mutations = []
        nonprefix = copy.deepcopy(original)
        nonprefix["motions"][0]["origin_id"] = plan["rows"][4]["origin_id"]
        mutations.append(nonprefix)
        reordered = copy.deepcopy(original)
        reordered["motions"][:2] = reversed(reordered["motions"][:2])
        mutations.append(reordered)
        duplicate = copy.deepcopy(original)
        duplicate["motions"][1]["origin_id"] = duplicate["motions"][0]["origin_id"]
        mutations.append(duplicate)
        extra = copy.deepcopy(original)
        extra["motions"].append(copy.deepcopy(extra["motions"][-1]))
        mutations.append(extra)
        for bad in mutations:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                resume.completed_prefix(plan, bad)

    def test_rejects_incomplete_work_disguised_as_completed_origin(self):
        plan, original = prefix_fixture()
        mutations = []
        for field, value in (("result", "FAIL"), ("stage", "retarget")):
            bad = copy.deepcopy(original)
            bad["motions"][0][field] = value
            mutations.append(bad)
        for jobs in ([{"stage": "prepare", "result": "COMPLETE"}],
                     [{"stage": "retarget", "result": "COMPLETE"}, {"stage": "prepare", "result": "COMPLETE"}],
                     [{"stage": "prepare", "result": "COMPLETE"}, {"stage": "retarget", "result": "RUNNING"}],
                     [{"stage": stage, "result": "COMPLETE"} for stage in ("prepare", "retarget", "retarget")]):
            bad = copy.deepcopy(original)
            bad["motions"][0]["jobs"] = jobs
            mutations.append(bad)
        for bad in mutations:
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "incomplete work"):
                resume.completed_prefix(plan, bad)

    def test_rejects_running_or_complete_unfinished_tail(self):
        for value in ("RUNNING", "COMPLETE", None):
            plan, status = prefix_fixture()
            status["motions"][-1]["result"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "interrupted/failed"):
                resume.completed_prefix(plan, status)

    def test_rejects_invalid_top_level_state_or_training_authority(self):
        plan, original = prefix_fixture()
        for change in ({"motions_completed": True}, {"motions_completed": -1}, {"motions_completed": 40},
                       {"motions_completed": 1.0}, {"motions_requested": 39}, {"result": "RUNNING"},
                       {"result": "COMPLETE"}, {"training_started": True}, {"automatic_promotion": True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                resume.completed_prefix(plan, {**original, **change})


class ResumeServiceTests(unittest.TestCase):
    def test_previous_service_must_be_stopped_and_owned(self):
        for state in ("inactive", "failed"):
            with mock.patch.object(resume.subprocess, "check_output", return_value=f"MainPID=0\nActiveState={state}\n") as call:
                resume.service_stopped("bfm-umr-pilot-old.service")
                self.assertIn("bfm-umr-pilot-old.service", call.call_args.args[0])
        for unit in ("other.service", "bfm-umr-eval-old.service", "bfm-umr-pilot-old.service/extra", None):
            with mock.patch.object(resume.subprocess, "check_output") as call, self.subTest(unit=unit), self.assertRaises(ValueError):
                resume.service_stopped(unit)
            call.assert_not_called()

    def test_previous_service_alive_or_unknown_is_rejected(self):
        for output in ("MainPID=17\nActiveState=inactive\n", "MainPID=0\nActiveState=active\n",
                       "MainPID=0\nActiveState=activating\n", "ActiveState=inactive\n", "MainPID=0\n"):
            with mock.patch.object(resume.subprocess, "check_output", return_value=output), \
                    self.subTest(output=output), self.assertRaisesRegex(ValueError, "still alive"):
                resume.service_stopped("bfm-umr-pilot-old.service")

    def test_execution_requires_owned_finitely_bounded_nonrestarting_unit(self):
        output = "Restart=no\nRuntimeMaxUSec=30min\nMemoryMax=8589934592\nTasksMax=192\n"
        with mock.patch.object(resume.producer, "owned_unit", return_value="bfm-umr-pilot-new.service"), \
                mock.patch.object(resume.subprocess, "check_output", return_value=output):
            self.assertEqual(resume.bounded_unit(), "bfm-umr-pilot-new.service")
        for key in ("RuntimeMaxUSec", "MemoryMax", "TasksMax"):
            for bad in ("infinity", "0", ""):
                values = {"Restart": "no", "RuntimeMaxUSec": "30min", "MemoryMax": "8589934592", "TasksMax": "192"}
                values[key] = bad
                with mock.patch.object(resume.producer, "owned_unit", return_value="bfm-umr-pilot-new.service"), \
                        mock.patch.object(resume.subprocess, "check_output", return_value="\n".join(f"{k}={v}" for k, v in values.items())), \
                        self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    resume.bounded_unit()
        with mock.patch.object(resume.producer, "owned_unit", side_effect=ValueError("not owned")), self.assertRaisesRegex(ValueError, "not owned"):
            resume.bounded_unit()
        with mock.patch.object(resume.producer, "owned_unit", return_value="bfm-umr-pilot-new.service"), \
                mock.patch.object(resume.subprocess, "check_output", return_value=output.replace("Restart=no", "Restart=on-failure")), \
                self.assertRaises(ValueError):
            resume.bounded_unit()


class InterruptedArtifactTests(unittest.TestCase):
    def test_candidates_only_include_explicit_unfinished_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            plan, _ = prefix_fixture()
            (batch / "setup_cache").mkdir()
            completed = batch / "pairs" / plan["rows"][0]["origin_id"] / "keep.pkl"
            unfinished = batch / "pairs" / plan["rows"][2]["origin_id"] / "partial.pkl"
            unrelated = batch / "unrelated.txt"
            for path in (completed, unfinished, unrelated):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(path.name.encode())
            found = resume.interrupted_paths(batch, plan, 2)
            self.assertEqual(found, [unfinished.parent])
            self.assertEqual(completed.read_bytes(), b"keep.pkl")
            self.assertEqual(unrelated.read_bytes(), b"unrelated.txt")

    def test_rejects_symlink_leaf_and_symlink_inside_partial_tree(self):
        for child_link in (False, True):
            with tempfile.TemporaryDirectory() as folder:
                batch = Path(folder)
                plan, _ = prefix_fixture()
                target = batch / "pairs" / plan["rows"][2]["origin_id"]
                outside = batch / "unrelated"
                outside.mkdir()
                (outside / "keep").write_bytes(b"preserved")
                target.parent.mkdir(parents=True)
                if child_link:
                    target.mkdir()
                    (target / "redirect").symlink_to(outside, target_is_directory=True)
                else:
                    target.symlink_to(outside, target_is_directory=True)
                with self.subTest(child_link=child_link), self.assertRaises(ValueError):
                    resume.interrupted_paths(batch, plan, 2)
                self.assertEqual((outside / "keep").read_bytes(), b"preserved")

    def test_rejects_intermediate_symlink_even_when_target_remains_inside_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            plan, _ = prefix_fixture()
            # A parent redirect could otherwise move a completed artifact with
            # the same leaf name into the unfinished origin's history slot.
            safe = batch / "completed"
            safe.mkdir()
            (safe / "motion2").mkdir()
            (safe / "motion2" / "keep.pkl").write_bytes(b"completed")
            parent = batch / "pairs/KIT/actor"
            parent.parent.mkdir(parents=True)
            parent.symlink_to(safe, target_is_directory=True)
            with self.assertRaises(ValueError):
                resume.interrupted_paths(batch, plan, 2)
            self.assertEqual((safe / "motion2/keep.pkl").read_bytes(), b"completed")

    def test_archiving_moves_only_requested_paths_and_preserves_contents(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            log = batch / "logs/30_retarget.log"
            target = batch / "retargeted/CNRS/288/incomplete"
            untouched = batch / "logs/29_retarget.log"
            history = batch / "resume_history/next"
            log.parent.mkdir(parents=True)
            target.mkdir(parents=True)
            log.write_bytes(b"interrupted log\n")
            (target / "partial.npz").write_bytes(b"partial binary\x00\xff")
            untouched.write_bytes(b"completed log\n")
            before = {path: path.stat().st_ino for path in (log, target)}
            moves = resume.archive_attempt([log, target], batch, history)
            self.assertEqual(len(moves), 2)
            for move in moves:
                original, archived = Path(move["from"]), Path(move["to"])
                self.assertFalse(original.exists())
                self.assertEqual(archived.stat().st_ino, before[original])
                self.assertTrue(archived.is_relative_to(history / "interrupted_artifacts"))
            self.assertEqual((history / "interrupted_artifacts/logs/30_retarget.log").read_bytes(), b"interrupted log\n")
            self.assertEqual((history / "interrupted_artifacts/retargeted/CNRS/288/incomplete/partial.npz").read_bytes(), b"partial binary\x00\xff")
            self.assertEqual(untouched.read_bytes(), b"completed log\n")

    def test_archive_existing_target_refused_without_overwriting(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            source = batch / "logs/30_retarget.log"
            history = batch / "resume_history/next"
            target = history / "interrupted_artifacts/logs/30_retarget.log"
            for path, content in ((source, b"new"), (target, b"old")):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            with self.assertRaises(FileExistsError):
                resume.archive_attempt([source], batch, history)
            self.assertEqual(source.read_bytes(), b"new")
            self.assertEqual(target.read_bytes(), b"old")


class ResumeMainTests(unittest.TestCase):
    def mocked_validation(self, batch):
        (batch / "setup_cache").mkdir(exist_ok=True)
        plan, status = prefix_fixture()
        files = SimpleNamespace(files={"verified": "digest"}, recheck=mock.Mock())
        return batch, plan, status, [{"origin_id": row["origin_id"]} for row in plan["rows"][:2]], files

    def test_dry_run_never_writes_archives_locks_or_launches_workers(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            incomplete = batch / "logs/02_retarget.log"
            incomplete.parent.mkdir()
            incomplete.write_bytes(b"incomplete")
            (batch / "setup_cache").mkdir()
            before = sorted(str(path.relative_to(batch)) for path in batch.rglob("*"))
            out = io.StringIO()
            with mock.patch.object(resume, "validate_resume", return_value=self.mocked_validation(batch)), \
                    mock.patch.object(resume, "bounded_unit") as bounded, \
                    mock.patch.object(resume, "archive_attempt") as archive, \
                    mock.patch.object(resume, "resume_lock") as lock, \
                    mock.patch.object(resume.producer, "run_owned_child") as child, \
                    mock.patch.object(resume.producer, "save_json") as save, redirect_stdout(out):
                self.assertEqual(resume.main(["--batch", str(batch), "--run-id", "next"]), 0)
            result = json.loads(out.getvalue())
            self.assertEqual((result["plan_only"], result["verified_completed"], result["remaining"]), (True, 2, 38))
            self.assertEqual(result["archive_paths"], [str(incomplete)])
            self.assertFalse(result["training_started"])
            for call in (bounded, archive, lock, child, save):
                call.assert_not_called()
            self.assertEqual(before, sorted(str(path.relative_to(batch)) for path in batch.rglob("*")))
            self.assertEqual(incomplete.read_bytes(), b"incomplete")

    def test_existing_history_refused_before_archive_or_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            (batch / "resume_history/next").mkdir(parents=True)
            with mock.patch.object(resume, "validate_resume", return_value=self.mocked_validation(batch)), \
                    mock.patch.object(resume, "archive_attempt") as archive, \
                    mock.patch.object(resume.producer, "run_owned_child") as child, \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                resume.main(["--batch", str(batch), "--run-id", "next", "--execute"])
            archive.assert_not_called()
            child.assert_not_called()

    def test_history_symlink_parent_refused_before_any_write(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            batch, outside = root / "batch", root / "outside"
            batch.mkdir()
            outside.mkdir()
            (batch / "resume_history").symlink_to(outside, target_is_directory=True)
            with mock.patch.object(resume, "validate_resume", return_value=self.mocked_validation(batch)), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
                    self.assertRaises((ValueError, SystemExit)):
                resume.main(["--batch", str(batch), "--run-id", "next"])
            self.assertEqual(list(outside.iterdir()), [])

    def test_unsafe_run_id_rejected_before_batch_read(self):
        for value in ("../old", "/outside", ".", "..", "name/child", "x" * 81, ""):
            with mock.patch.object(resume, "validate_resume") as validate, redirect_stderr(io.StringIO()), \
                    self.subTest(value=value), self.assertRaises(SystemExit):
                resume.main(["--batch", "/unread", "--run-id", value])
            validate.assert_not_called()

    def test_execute_checks_bounded_service_before_lock_or_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            with mock.patch.object(resume, "validate_resume", return_value=self.mocked_validation(batch)), \
                    mock.patch.object(resume, "bounded_unit", side_effect=ValueError("unbounded")), \
                    mock.patch.object(resume, "resume_lock") as lock, \
                    mock.patch.object(resume, "archive_attempt") as archive, self.assertRaisesRegex(ValueError, "unbounded"):
                resume.main(["--batch", str(batch), "--run-id", "next", "--execute"])
            lock.assert_not_called()
            archive.assert_not_called()
            self.assertEqual(list(batch.iterdir()), [batch / "setup_cache"])


if __name__ == "__main__":
    unittest.main()
