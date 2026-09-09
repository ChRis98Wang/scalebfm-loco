from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
from queue import Queue
import tempfile
import unittest

from omegaconf import DictConfig

from scaleretarget.loader.base_loader import BaseLoader, load_motion_manifest
from scaleretarget.utils.multi_process import (
    ProducerFailure,
    ensure_all_motions_accounted_for,
    producer,
)


class _ResumeLoader(BaseLoader):
    def _load_sample(self, sample_path: Path):
        return [str(sample_path)], {"source": str(sample_path)}


class _FailingResumeLoader(_ResumeLoader):
    def _load_sample(self, sample_path: Path):
        raise ValueError(f"broken input: {sample_path}")


def _successful_worker(item) -> str:
    save_path, _frames, _extras = item
    return save_path


class RetargetResumeTests(unittest.TestCase):
    def _partial_loader(self, temp_path: Path) -> _ResumeLoader:
        input_dir = temp_path / "motions"
        input_dir.mkdir()
        (input_dir / "a.npz").touch()
        (input_dir / "b.npz").touch()
        output_dir = temp_path / "output"
        cached = output_dir / input_dir.name / "a.pkl"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"cached")
        loader = _ResumeLoader(
            DictConfig(
                {
                    "data_format": ".npz",
                    "overwrite": False,
                    "target_fps": 30,
                    "output_dir": str(output_dir),
                }
            )
        )
        loader.load(str(input_dir))
        return loader

    def test_serial_partial_resume_counts_cached_output_as_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = self._partial_loader(Path(temp_dir))

            attempted = list(loader)

            self.assertEqual(len(attempted), 1)
            self.assertEqual(loader.skipped_num, 1)
            ensure_all_motions_accounted_for(loader, success_num=1, processing_failures=0)

    def test_two_worker_partial_resume_counts_cached_output_as_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = self._partial_loader(Path(temp_dir))
            attempted = list(loader)
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=2, mp_context=context) as executor:
                successes = sum(
                    future.result() is not None
                    for future in [executor.submit(_successful_worker, item) for item in attempted]
                )

            self.assertEqual(successes, 1)
            self.assertEqual(loader.skipped_num, 1)
            ensure_all_motions_accounted_for(
                loader,
                success_num=successes,
                processing_failures=0,
            )

    def test_manifest_excludes_orphaned_motion_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            selected = root / "selected.npz"
            orphan = root / "orphan.npz"
            selected.touch()
            orphan.touch()
            manifest = root / ".bfm_motion_manifest"
            manifest.write_text("selected.npz\n", encoding="utf-8")

            files = load_motion_manifest(
                root,
                manifest,
                allowed_suffixes=(".npz",),
            )

            self.assertEqual(files, [str(selected)])

    def test_loader_failure_is_not_misclassified_as_a_cached_skip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            normal_loader = self._partial_loader(Path(temp_dir))
            loader = _FailingResumeLoader(normal_loader.config)
            loader.load(normal_loader.data_path)

            self.assertEqual(list(loader), [])
            self.assertEqual(loader.skipped_num, 1)
            self.assertEqual(loader.failed_num, 1)
            with self.assertRaisesRegex(RuntimeError, "loader_failed=1"):
                ensure_all_motions_accounted_for(
                    loader,
                    success_num=0,
                    processing_failures=0,
                )

    def test_producer_preserves_an_unexpected_iteration_exception(self) -> None:
        def broken_items():
            yield ("one", [], {})
            raise RuntimeError("producer exploded")

        queue = Queue()
        producer(broken_items(), queue)

        self.assertEqual(queue.get()[0], "one")
        failure = queue.get()
        self.assertIsInstance(failure, ProducerFailure)
        self.assertRegex(str(failure.exception), "producer exploded")
        self.assertIsNone(queue.get())


if __name__ == "__main__":
    unittest.main()
