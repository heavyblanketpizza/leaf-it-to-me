"""Regression checks for approximate Mango curation and retained group links."""

import csv
import hashlib
import io
import json
import tempfile
import unittest
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from leafit.mango import curate_mango
from leafit.prepare import UnionFind, make_row, prepare
from leafit.smoke import make_fixture


class MangoCurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def photo(self, name, *, label="mango_healthy", source="mangoleafbd", capture=None,
              camera=("Maker", "Phone"), software="", size=(20, 30), color=(60, 90, 20)):
        path = self.root / name
        exif = Image.Exif()
        if capture is not None:
            exif[271], exif[272] = camera
            exif[34665] = {36867: capture}
        if software:
            exif[305] = software
        Image.new("RGB", size, color).save(path, exif=exif)
        return {**make_row(path, self.root, source, label), "sha256_pixels": "fixture"}

    def run_curation(self, rows, union=None):
        return curate_mango(rows, union or UnionFind(len(rows)), self.root)

    def audit(self):
        with (self.root / "mango_curation.csv").open(newline="") as handle:
            return list(csv.DictReader(handle))

    def test_same_capture_collapses_changed_pixels_and_preserves_raw_files(self):
        rows = [self.photo("first.jpg", capture="2021:05:06 12:00:00", color=(255, 0, 0)),
                self.photo("first (Custom) Copy (1).jpg", capture="2021:05:06 12:00:00", color=(0, 0, 255))]
        before = {row["image_path"]: hashlib.sha256(Path(row["image_path"]).read_bytes()).hexdigest() for row in rows}
        retained, _, report = self.run_curation(rows)
        self.assertEqual([row["relative_path"] for row in retained], ["first.jpg"])
        self.assertEqual((report["input_images"], report["retained_images"], report["excluded_images"]), (2, 1, 1))
        self.assertEqual(report["capture_families"], 1)
        self.assertEqual({row["action"] for row in self.audit()}, {"keep", "exclude"})
        self.assertEqual(len({row["capture_id"] for row in self.audit()}), 1)
        for path, digest in before.items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), digest)

    def test_missing_or_invalid_capture_is_unique_and_cameras_are_separate(self):
        rows = [self.photo("no-metadata-a.jpg"), self.photo("no-metadata-b.jpg"),
                self.photo("invalid-a.jpg", capture="0000:00:00 00:00:00"),
                self.photo("invalid-b.jpg", capture="0000:00:00 00:00:00"),
                self.photo("phone-a.jpg", capture="2021:05:06 12:00:00"),
                self.photo("phone-b.jpg", capture="2021:05:06 12:00:00", camera=("Maker", "Other phone"))]
        retained, groups, report = self.run_curation(rows)
        self.assertEqual(len(retained), 6)
        self.assertEqual(len({groups.find(i) for i in range(6)}), 6)
        self.assertEqual(report["images_without_capture_metadata"], 4)
        self.assertEqual(report["capture_families"], 2)

    def test_cross_label_family_retains_one_each_without_relabeling(self):
        rows = [self.photo("healthy-a.jpg", capture="2021:05:06 12:00:00"),
                self.photo("healthy-b.jpg", capture="2021:05:06 12:00:00"),
                self.photo("scab-a.jpg", label="mango_dieback", capture="2021:05:06 12:00:00"),
                self.photo("scab-b.jpg", label="mango_dieback", capture="2021:05:06 12:00:00")]
        retained, groups, report = self.run_curation(rows)
        self.assertEqual(len(retained), 2)
        self.assertEqual({row["class_label"] for row in retained}, {"mango_healthy", "mango_dieback"})
        self.assertEqual(groups.find(0), groups.find(1))
        self.assertEqual(report["families"], 1)
        self.assertEqual({item["family_id"] for item in self.audit()}, {self.audit()[0]["family_id"]})

    def test_removed_mango_bridge_keeps_other_dataset_relatives_connected(self):
        rows = [self.photo("apple-a.jpg", source="plantvillage", label="apple_healthy"),
                self.photo("mango-z.jpg", capture="2021:05:06 12:00:00"),
                self.photo("apple-b.jpg", source="cornell", label="apple_scab"),
                self.photo("mango-a.jpg", capture="2021:05:06 12:00:00")]
        groups = UnionFind(len(rows))
        groups.union(0, 1)
        groups.union(1, 2)
        retained, remapped, _ = self.run_curation(rows, groups)
        self.assertEqual([row["relative_path"] for row in retained], ["apple-a.jpg", "apple-b.jpg", "mango-a.jpg"])
        self.assertEqual(len({remapped.find(i) for i in range(3)}), 1)

    def test_representative_preference_and_family_id_ignore_input_order(self):
        rows = [self.photo("a-edited.jpg", software="Windows Photo Editor", size=(400, 600)),
                self.photo("small.jpg", size=(10, 10)), self.photo("best.jpg", size=(100, 100)),
                self.photo("best Copy (2).jpg", size=(100, 100))]
        selected, family_ids, audits = [], [], []
        for ordering in (rows, list(reversed(rows))):
            groups = UnionFind(len(ordering))
            for index in range(1, len(ordering)):
                groups.union(0, index)
            retained, _, _ = self.run_curation(ordering, groups)
            selected.append(retained[0]["relative_path"])
            family_ids.append({row["family_id"] for row in self.audit()})
            audits.append(self.audit())
        self.assertEqual(selected, ["best.jpg", "best.jpg"])
        self.assertEqual(family_ids[0], family_ids[1])
        self.assertEqual(audits[0], audits[1])

    def test_metadata_read_failure_stops_with_filename(self):
        row = self.photo("broken.jpg")
        Path(row["image_path"]).write_bytes(b"broken JPEG")
        with self.assertRaisesRegex(ValueError, "Cannot read Mango capture metadata.*broken.jpg"):
            self.run_curation([row])

    def test_no_mango_preserves_existing_groups_and_writes_empty_audit(self):
        rows = [self.photo("apple-a.jpg", source="cornell", label="apple_healthy"),
                self.photo("apple-b.jpg", source="plantvillage", label="apple_healthy")]
        groups = UnionFind(2)
        groups.union(0, 1)
        retained, remapped, report = self.run_curation(rows, groups)
        self.assertEqual(retained, rows)
        self.assertEqual(remapped.find(0), remapped.find(1))
        self.assertEqual(report["input_images"], 0)
        self.assertEqual(self.audit(), [])

    def test_prepare_deduplicates_and_splits_all_classes_without_claiming_review(self):
        raw = make_fixture(self.root / "raw")
        originals = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in raw.rglob("*") if path.is_file()}
        input_mango = list((raw / "mangoleafbd").rglob("*.png"))
        outputs = [self.root / "prepared-first", self.root / "prepared-again"]
        for output in outputs:
            with redirect_stdout(io.StringIO()):
                prepare(SimpleNamespace(
                    plantvillage=raw / "plantvillage", cornell=raw / "cornell",
                    mango=raw / "mangoleafbd", output=output, leaf_map=None,
                    groups=None, seed=42, near_threshold=4, allow_partial=False,
                    exclude_conflicting_duplicates=False, mango_policy="deduplicate",
                ))
            report = json.loads((output / "report.json").read_text())
            labels = json.loads((output / "labels.json").read_text())
            with (output / "manifest.csv").open(newline="") as handle:
                manifest = list(csv.DictReader(handle))
            with (output / "mango_curation.csv").open(newline="") as handle:
                audit = list(csv.DictReader(handle))
            self.assertFalse(report["mango_groups_completely_reviewed"])
            self.assertEqual(report["mango_policy"], "deduplicate")
            self.assertEqual(len(labels), 18)
            expected_classes = {row["label"] for row in labels}
            for split in ("train", "val", "test"):
                self.assertEqual({row["class_label"] for row in manifest if row["split"] == split},
                                 expected_classes)
            mango_rows = [row for row in manifest if row["source_dataset"] == "mangoleafbd"]
            self.assertTrue(all(not row["reviewed_group"] for row in mango_rows))
            self.assertLess(len(mango_rows), len(input_mango))
            self.assertEqual(len({(row["group_id"], row["class_label"]) for row in mango_rows}), len(mango_rows))
            self.assertEqual({row["image_path"] for row in audit}, {str(path) for path in input_mango})
            self.assertEqual(len(audit), len(input_mango))
            self.assertEqual({row["image_path"] for row in audit if row["action"] == "keep"},
                             {row["image_path"] for row in mango_rows})
            group_splits = defaultdict(set)
            for row in manifest:
                group_splits[row["group_id"]].add(row["split"])
            self.assertTrue(all(len(splits) == 1 for splits in group_splits.values()))
        for filename in ("manifest.csv", "labels.json", "mango_curation.csv"):
            self.assertEqual((outputs[0] / filename).read_bytes(), (outputs[1] / filename).read_bytes())
        self.assertEqual(set(originals), {path for path in raw.rglob("*") if path.is_file()})
        for path, digest in originals.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
