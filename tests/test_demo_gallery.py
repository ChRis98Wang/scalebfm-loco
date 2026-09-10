"""Integrity checks for published media; no simulator, assets or Pillow required."""
from pathlib import Path
import hashlib
import json
import re
import unittest
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "docs/media/gallery-20260910.json"


class DemoGalleryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads(CATALOG.read_text())
        cls.clips = cls.catalog["clips"]
        cls.readme = (ROOT / "README.md").read_text()

    def test_catalog_counts_and_boundaries(self):
        self.assertEqual(self.catalog["schema"], "bfm.demo_gallery/1")
        self.assertEqual(len(self.clips), 13)
        self.assertEqual(self.catalog["gif_count"], 13)
        self.assertEqual(self.catalog["mp4_count"], 26)
        self.assertEqual(self.catalog["total_media_files"], 52)
        self.assertEqual(self.catalog["training_updates_in_recordings"], 0)
        self.assertFalse(self.catalog["full_bfm_completed"])
        self.assertFalse(self.catalog["formal_multi_initial_state_accepted"])
        self.assertFalse(self.catalog["blanket_redistribution_license_granted"])

    def test_file_paths_headers_sizes_and_hashes(self):
        paths = set()
        total = 0
        for clip in self.clips:
            self.assertEqual(set(clip["files"]), {"gif", "mp4", "raw_mp4", "poster"})
            for kind, info in clip["files"].items():
                with self.subTest(clip=clip["prefix"], kind=kind):
                    path = ROOT / info["path"]
                    self.assertTrue(path.resolve().is_relative_to(ROOT / "docs/media"))
                    self.assertFalse(path.is_symlink())
                    self.assertNotIn(info["path"], paths)
                    paths.add(info["path"])
                    data = path.read_bytes()
                    self.assertEqual(len(data), info["bytes"])
                    self.assertLess(len(data), 50 * 1024 * 1024)
                    self.assertEqual(hashlib.sha256(data).hexdigest(), info["sha256"])
                    total += len(data)
                    if kind == "gif":
                        self.assertIn(data[:6], (b"GIF87a", b"GIF89a"))
                    elif kind == "poster":
                        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
                    else:
                        self.assertIn(b"ftyp", data[:32])
        self.assertEqual(total, self.catalog["total_media_bytes"])

    def test_decode_receipts_preserve_full_windows(self):
        for clip in self.clips:
            with self.subTest(clip=clip["prefix"]):
                self.assertTrue(clip["media_full_decode_pass"])
                self.assertFalse(clip["full_manual_playback_review"])
                for role in ("mp4", "raw_mp4"):
                    info = clip["files"][role]
                    self.assertEqual(info["decoded_frames"], clip["video_frames"])
                    self.assertAlmostEqual(info["decoded_frames"] / info["fps"], clip["video_seconds"])
                gif = clip["files"]["gif"]
                self.assertLessEqual(abs(gif["seconds"] - clip["video_seconds"]), 0.101)
                self.assertEqual(gif["loop"], 0)

    def test_every_gif_and_video_is_reachable_from_readme(self):
        for clip in self.clips:
            for role in ("gif", "mp4", "raw_mp4"):
                self.assertIn(clip["files"][role]["path"], self.readme)
            self.assertIn('src="' + clip["files"]["gif"]["path"] + '"', self.readme)

    def test_relative_documentation_links_resolve(self):
        for name in ("README.md", "docs/DEMO_GALLERY.md", "docs/MEDIA_SOURCES.md", "docs/media/README.md"):
            file = ROOT / name
            text = file.read_text()
            links = re.findall(r'\]\(([^)\s]+)(?:\s+[^)]*)?\)', text)
            links += re.findall(r'(?:href|src)="([^"]+)"', text)
            for link in links:
                parsed = urlsplit(link)
                if parsed.scheme or not parsed.path:
                    continue
                with self.subTest(document=name, link=link):
                    self.assertTrue((file.parent / unquote(parsed.path)).resolve().exists())

    def test_failure_and_empty_handed_labels_are_preserved(self):
        failures = [clip for clip in self.clips if clip["status"] == "failed_trial"]
        self.assertEqual(len(failures), 2)
        self.assertTrue(all(clip["group"] == "failures" for clip in failures))
        self.assertIn("17/19", self.readme)
        self.assertIn("empty-handed", self.readme)
        self.assertIn("1/4", self.readme)
        self.assertIn("Development failures", self.readme)

    def test_amass_summary_stays_concise(self):
        section = self.readme.split("## Local AMASS end-to-end workflow\n", 1)[1].split("\n## ", 1)[0]
        self.assertLessEqual(len(section.split()), 200)
        for term in ("7,174", "1,751", "Transformer", "PPO", "open_motion_browser.sh"):
            self.assertIn(term, section)

    def test_gallery_integrity_is_in_public_cpu_workflow(self):
        workflow = (ROOT / ".github/workflows/cpu-tests.yml").read_text()
        self.assertIn("python -m pytest tests/test_demo_gallery.py -q", workflow)


if __name__ == "__main__":
    unittest.main()
