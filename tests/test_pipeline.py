"""Small regression checks for leakage protection and training behavior.

Run with: python -m unittest discover -s tests -v
These use temporary generated images, never the downloaded datasets.
"""

import csv
import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import timm
import torch

from orchard.model import freeze_backbone, load_rgb, make_loader, metrics_from_confusion, read_manifest
from orchard.prepare import UnionFind, assign_splits, connect_related, fingerprint, import_cornell, make_row, prepare
from orchard.smoke import make_fixture


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_exif_orientation_rgb_and_original_preserved(self):
        original = self.root / "oriented.jpg"
        image = Image.new("L", (12, 20), 80)
        exif = image.getexif()
        exif[274] = 6  # Camera metadata says rotate ninety degrees clockwise.
        image.save(original, exif=exif)
        before = hashlib.sha256(original.read_bytes()).hexdigest()
        loaded = load_rgb(original)
        self.assertEqual(loaded.mode, "RGB")
        self.assertEqual(loaded.size, (20, 12))
        self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), before)
        normalized = self.root / "normalized.png"
        loaded.save(normalized)
        self.assertEqual(fingerprint(original)[0], fingerprint(normalized)[0])
        self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), before)

    def test_group_split_is_reproducible_and_never_separates_leaf_copies(self):
        rows = [dict(source_dataset="plantvillage", relative_path=f"leaf{group}_copy{copy}.png",
                     class_label="apple_healthy") for group in range(10) for copy in range(3)]
        union = UnionFind(len(rows))
        for group in range(10):
            union.union(group * 3, group * 3 + 1)
            union.union(group * 3, group * 3 + 2)
        first, second = copy.deepcopy(rows), copy.deepcopy(rows)
        distribution = assign_splits(first, union, seed=11, mango_reviewed=False)
        assign_splits(second, union, seed=11, mango_reviewed=False)
        self.assertEqual(first, second)
        self.assertEqual({split: sum(counts.values()) for split, counts in distribution.items()},
                         {"train": 24, "val": 3, "test": 3})
        for group in range(10):
            self.assertEqual(len({row["split"] for row in first[group * 3:group * 3 + 3]}), 1)
            self.assertEqual(len({row["group_id"] for row in first[group * 3:group * 3 + 3]}), 1)

    def test_exact_cross_dataset_copies_share_group(self):
        image = Image.fromarray(np.random.default_rng(7).integers(0, 256, (32, 32, 3), dtype=np.uint8))
        rows = []
        for name, source in (("first.png", "plantvillage"), ("second.png", "cornell")):
            path = self.root / name
            image.save(path)
            rows.append(make_row(path, self.root, source, "apple_healthy"))
        valid, union, stats = connect_related(rows, 0, self.root)
        self.assertEqual(len(valid), 2)
        self.assertEqual(union.find(0), union.find(1))
        self.assertEqual(stats["cross_dataset_duplicate_pairs"], 1)
        self.assertEqual(stats["exact_duplicate_pairs"], 1)

    def test_duplicate_with_conflicting_label_stops_preparation(self):
        image = Image.fromarray(np.random.default_rng(17).integers(0, 256, (32, 32, 3), dtype=np.uint8))
        rows = []
        for index, label in enumerate(("apple_healthy", "apple_scab")):
            path = self.root / f"copy{index}.png"
            image.save(path)
            rows.append(make_row(path, self.root, "plantvillage", label))
        with self.assertRaisesRegex(ValueError, "different labels"):
            connect_related(rows, 0, self.root)
        with (self.root / "conflicts.csv").open(newline="") as handle:
            conflicts = list(csv.DictReader(handle))
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["match_type"], "exact_pixels")

    def test_conflict_exclusion_removes_whole_family_and_preserves_group_bridges(self):
        rng = np.random.default_rng(73)
        ambiguous = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        rows = []
        # The two ambiguous copies bridge different official groups. Removing
        # them must still leave the two surviving relatives in one group.
        specifications = [
            ("ambiguous_a.png", "apple_healthy", "leaf-a", ambiguous),
            ("survivor_a.png", "apple_healthy", "leaf-a", rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)),
            ("ambiguous_b.png", "apple_scab", "leaf-b", ambiguous),
            ("survivor_b.png", "apple_scab", "leaf-b", rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)),
            ("third_copy.png", "apple_healthy", "", ambiguous),
        ]
        for name, label, group, pixels in specifications:
            path = self.root / name
            Image.fromarray(pixels).save(path)
            rows.append(make_row(path, self.root, "plantvillage", label, group))
        originals = {Path(row["image_path"]): Path(row["image_path"]).read_bytes() for row in rows}
        with redirect_stdout(io.StringIO()) as log:
            valid, union, stats = connect_related(rows, 0, self.root, exclude_conflicts=True)
        self.assertEqual([Path(row["image_path"]).name for row in valid], ["survivor_a.png", "survivor_b.png"])
        self.assertEqual(union.find(0), union.find(1))
        self.assertEqual(stats["exact_cross_label_conflict_pairs"], 1)
        self.assertEqual(stats["excluded_conflicting_images"], 3)
        self.assertEqual(stats["excluded_conflicting_pixel_hashes"], 1)
        self.assertIn("Excluded 3 images", log.getvalue())
        with (self.root / "excluded_conflicts.csv").open(newline="") as handle:
            excluded = list(csv.DictReader(handle))
        self.assertEqual({Path(row["image_path"]).name for row in excluded},
                         {"ambiguous_a.png", "ambiguous_b.png", "third_copy.png"})
        self.assertEqual({row["class_label"] for row in excluded}, {"apple_healthy", "apple_scab"})
        self.assertEqual(len({row["sha256_pixels"] for row in excluded}), 1)
        self.assertTrue(all(row["reason"] for row in excluded))
        with (self.root / "conflicts.csv").open(newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 1)
        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content)

    def test_near_candidate_with_different_labels_is_grouped_for_review(self):
        pixels = np.random.default_rng(91).integers(32, 220, (32, 32, 3), dtype=np.uint8)
        rows = []
        for index, label in enumerate(("apple_healthy", "apple_scab")):
            path = self.root / f"near{index}.png"
            Image.fromarray(pixels + index * 10).save(path)
            rows.append(make_row(path, self.root, "plantvillage", label))
        self.assertNotEqual(fingerprint(rows[0]["image_path"])[0], fingerprint(rows[1]["image_path"])[0])
        valid, union, _ = connect_related(rows, 4, self.root)
        self.assertEqual(union.find(0), union.find(1))
        self.assertEqual([row["class_label"] for row in valid], ["apple_healthy", "apple_scab"])
        with (self.root / "review_candidates.csv").open(newline="") as handle:
            candidates = list(csv.DictReader(handle))
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["match_type"], "perceptual_candidate")

    def test_rotated_copy_is_grouped_and_unreadable_image_is_reported(self):
        image = Image.fromarray(np.random.default_rng(27).integers(0, 256, (48, 48, 3), dtype=np.uint8))
        first, second, broken = [self.root / name for name in ("original.png", "rotated.png", "broken.jpg")]
        image.save(first)
        image.transpose(Image.Transpose.ROTATE_90).save(second)
        broken.write_bytes(b"not an image")
        originals = {path: path.read_bytes() for path in (first, second, broken)}
        rows = [make_row(path, self.root, "plantvillage", "apple_healthy") for path in originals]
        valid, union, stats = connect_related(rows, 4, self.root)
        self.assertEqual(len(valid), 2)
        self.assertEqual(union.find(0), union.find(1))
        self.assertEqual(stats["unreadable_images"], 1)
        self.assertEqual(stats["perceptual_candidate_pairs"], 1)
        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content)

    def test_manifest_rejects_group_crossing_splits(self):
        labels = [{"index": 0, "label": "apple_healthy", "tree_type": "apple", "condition": "healthy"}]
        manifest = self.root / "manifest.csv"
        fields = ["image_path", "source_dataset", "tree_type", "condition", "class_label", "group_id", "split"]
        with manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, split in enumerate(("train", "test")):
                path = self.root / f"photo{index}.png"
                Image.new("RGB", (12, 12), (index * 20, 100, 40)).save(path)
                writer.writerow(dict(image_path=path.name, source_dataset="fixture",
                                     tree_type="apple", condition="healthy", class_label="apple_healthy",
                                     group_id="one-physical-leaf", split=split))
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            read_manifest(manifest, labels)

    def test_metrics_count_absent_classes_explicitly(self):
        summary, per_class = metrics_from_confusion(np.array([[3, 1, 0], [1, 1, 0], [0, 0, 0]]))
        self.assertAlmostEqual(summary["accuracy"], 4 / 6)
        self.assertAlmostEqual(summary["macro_f1"], (0.75 + 0.5) / 3)
        self.assertAlmostEqual(summary["macro_f1_supported_classes"], (0.75 + 0.5) / 2)
        self.assertEqual([row["support"] for row in per_class], [4, 2, 0])

    def test_head_step_changes_head_only_and_preserves_batchnorm(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        model = timm.create_model("mobilenetv4_conv_small.e2400_r224_in1k", pretrained=False, num_classes=2)
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        head = freeze_backbone(model)
        optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(torch.randn(2, 3, 224, 224)), torch.tensor([0, 0]))
        loss.backward()
        optimizer.step()
        after = model.state_dict()
        self.assertTrue(any(not torch.equal(after[name], before[name])
                            for name in after if name.startswith("classifier.")))
        for name in after:
            if not name.startswith("classifier."):
                self.assertTrue(torch.equal(after[name], before[name]), name)

    def test_fine_tuning_omits_only_singleton_and_evaluation_keeps_every_image(self):
        torch.set_num_threads(2)
        image = self.root / "photo.png"
        Image.new("RGB", (224, 224), (20, 100, 40)).save(image)
        rows = [dict(image_path=str(image), target=index % 2, source_dataset="fixture") for index in range(5)]
        args = SimpleNamespace(batch_size=2, workers=0, seed=4)
        device = torch.device("cpu")
        model = timm.create_model("mobilenetv4_conv_small.e2400_r224_in1k", pretrained=False, num_classes=2)
        config = timm.data.resolve_model_data_config(model)
        fine_loader = make_loader(rows, config, args, True, device, fine_tuning=True)
        model.train()
        fine_sizes = []
        with torch.inference_mode():
            for images, _, _ in fine_loader:
                fine_sizes.append(len(images))
                self.assertEqual(tuple(model(images).shape), (2, 2))
        self.assertEqual(fine_sizes, [2, 2])
        for training in (True, False):
            loader = make_loader(rows, config, args, training, device)
            self.assertEqual([len(images) for images, _, _ in loader], [2, 2, 1])
        non_singleton = make_loader(rows[:4], config, SimpleNamespace(batch_size=3, workers=0, seed=4),
                                    True, device, fine_tuning=True)
        self.assertEqual([len(images) for images, _, _ in non_singleton], [3])
        remainder_two = make_loader(rows, config, SimpleNamespace(batch_size=3, workers=0, seed=4),
                                    True, device, fine_tuning=True)
        self.assertEqual([len(images) for images, _, _ in remainder_two], [3, 2])
        with self.assertRaisesRegex(ValueError, "at least 2"):
            make_loader(rows, config, SimpleNamespace(batch_size=1, workers=0), True, device, fine_tuning=True)
        with self.assertRaisesRegex(ValueError, "at least two"):
            make_loader(rows[:1], config, args, True, device, fine_tuning=True)

    def test_fixture_cornell_ignores_unlabeled_test_and_mango_requires_reviewed_groups(self):
        raw = make_fixture(self.root / "fixture")
        notes = {}
        cornell = import_cornell(raw / "cornell", notes)
        self.assertEqual(len(cornell), 48)
        self.assertEqual(notes["cornell"]["ignored_test_images"], 1)
        self.assertTrue(all(Path(row["image_path"]).stem.startswith("Train_") for row in cornell))
        # A sample submission is an output example, never a source of ground truth.
        (raw / "cornell/sample_submission.csv").write_text(
            "image_id,healthy,multiple_diseases,rust,scab\nTest_0,1,0,0,0\n"
        )
        self.assertEqual(len(import_cornell(raw / "cornell", {})), 48)
        for reviewed in (False, True):
            output = self.root / ("reviewed" if reviewed else "unreviewed")
            args = SimpleNamespace(output=output, near_threshold=4, plantvillage=raw / "plantvillage",
                                   cornell=raw / "cornell", mango=raw / "mangoleafbd", leaf_map=None,
                                   groups=raw / "groups.csv" if reviewed else None, seed=42, allow_partial=False)
            with redirect_stdout(io.StringIO()):
                prepare(args)
            report = json.loads((output / "report.json").read_text())
            with (output / "manifest.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            mango_splits = {row["split"] for row in rows if row["source_dataset"] == "mangoleafbd"}
            self.assertEqual(mango_splits, {"train", "val", "test"} if reviewed else {"train"})
            self.assertEqual(report["mango_groups_completely_reviewed"], reviewed)
            self.assertEqual(report["datasets"]["cornell"]["public_labeled_rows"], 48)
            self.assertEqual(report["datasets"]["cornell"]["ignored_test_images"], 1)
            self.assertEqual(report["retained_images"], 252)
            self.assertFalse(report["exclude_conflicting_duplicates"])
            self.assertEqual(report["excluded_conflicting_images"], 0)
            by_group = {}
            for row in rows:
                by_group.setdefault(row["group_id"], set()).add(row["split"])
            self.assertTrue(all(len(splits) == 1 for splits in by_group.values()))
            self.assertEqual(len(json.loads((output / "labels.json").read_text())), 18)


if __name__ == "__main__":
    unittest.main()
