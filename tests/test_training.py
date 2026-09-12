"""Regression checks for safe training startup and honest split coverage."""

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import train as train_cli
from leafit.model import train, training_data_summary, validate_training_output


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_portable_default_without_environment_setting(self):
        for environment in ({}, {"LEAFIT_DATA": ""}):
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True):
                self.assertEqual(train_cli.build_parser().parse_args([]).data_root, Path("."))

    def test_environment_root_and_explicit_override_reach_trainer(self):
        for folder in ("storage with spaces", "explicit override"):
            prepared = self.root / folder / "data/prepared"
            prepared.mkdir(parents=True)
            for filename in ("manifest.csv", "labels.json"):
                (prepared / filename).touch()
        with patch.dict(os.environ, {"LEAFIT_DATA": str(self.root / "storage with spaces")}):
            for options, folder in (([], "storage with spaces"),
                                    (["--data-root", str(self.root / "explicit override")], "explicit override")):
                with self.subTest(folder=folder), patch("train.train") as trainer:
                    self.assertEqual(train_cli.main(options), 0)
                    args = trainer.call_args.args[0]
                    prepared = (self.root / folder / "data/prepared").resolve()
                    self.assertEqual(args.manifest, str(prepared / "manifest.csv"))
                    self.assertEqual(args.labels, str(prepared / "labels.json"))

    def test_rerun_preserves_existing_artifacts_and_stops_before_model_creation(self):
        output = self.root / "completed-run"
        output.mkdir()
        originals = {
            "best.pt": b"previous learned weights",
            "metrics.json": b'{"accuracy": 0.8}\n',
            "history.json": b'[{"epoch": 3}]\n',
        }
        for name, content in originals.items():
            (output / name).write_bytes(content)
        args = train_cli.build_parser().parse_args(["--output", str(output)])

        with patch("leafit.model.validate_training_output", wraps=validate_training_output) as guard:
            with patch("leafit.model.timm.create_model") as create_model:
                with self.assertRaisesRegex(ValueError, "new or empty folder"):
                    train(args)
                guard.assert_called_once_with(output)
                create_model.assert_not_called()

        self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, originals)

    def test_data_summary_exposes_unmeasured_classes_with_zero_counts(self):
        labels = [
            {"index": 0, "label": "apple_healthy", "tree_type": "apple", "condition": "healthy"},
            {"index": 1, "label": "apple_scab", "tree_type": "apple", "condition": "scab"},
            {"index": 2, "label": "mango_healthy", "tree_type": "mango", "condition": "healthy"},
        ]
        # Two photos of one training leaf count as two images but one group.
        records = [
            ("train", "apple_healthy", "plantvillage", "apple-train"),
            ("train", "apple_healthy", "plantvillage", "apple-train"),
            ("train", "apple_scab", "cornell", "scab-train"),
            ("train", "mango_healthy", "mangoleafbd", "mango-train"),
            ("val", "apple_healthy", "plantvillage", "apple-val"),
            ("test", "apple_scab", "cornell", "scab-test-1"),
            ("test", "apple_scab", "cornell", "scab-test-2"),
        ]
        rows = [dict(zip(("split", "class_label", "source_dataset", "group_id"), record))
                for record in records]

        summary = training_data_summary(rows, labels)
        self.assertEqual(summary["total_images"], 7)
        self.assertEqual(summary["class_count"], 3)
        training = summary["splits"]["train"]
        self.assertEqual((training["images"], training["groups"], training["supported_classes"]), (4, 3, 3))
        self.assertEqual(training["missing_classes"], [])
        self.assertEqual(training["by_source"], {"plantvillage": 2, "cornell": 1, "mangoleafbd": 1})
        for split, supported_label, count, missing in (
            ("val", "apple_healthy", 1, ["apple_scab", "mango_healthy"]),
            ("test", "apple_scab", 2, ["apple_healthy", "mango_healthy"]),
        ):
            with self.subTest(split=split):
                details = summary["splits"][split]
                self.assertEqual(details["supported_classes"], 1)
                self.assertEqual(details["missing_classes"], missing)
                self.assertEqual(details["by_class"], {
                    label["label"]: count if label["label"] == supported_label else 0
                    for label in labels
                })

    def test_cli_missing_merged_files_stops_before_trainer_runs(self):
        for manifest_exists in (False, True):
            with self.subTest(manifest_exists=manifest_exists):
                data_root = self.root / ("missing-labels" if manifest_exists else "missing-manifest")
                prepared = data_root / "data" / "prepared"
                prepared.mkdir(parents=True)
                if manifest_exists:
                    (prepared / "manifest.csv").write_text("image_path,class_label,split\n", encoding="utf-8")
                output = data_root / "run"
                error_text = io.StringIO()
                with patch("train.train") as trainer, redirect_stderr(error_text):
                    with self.assertRaises(SystemExit) as caught:
                        train_cli.main(["--data-root", str(data_root), "--output", str(output)])
                    trainer.assert_not_called()
                self.assertEqual(caught.exception.code, 2)
                expected_file = "labels.json" if manifest_exists else "manifest.csv"
                self.assertIn(str(prepared / expected_file), error_text.getvalue())
                self.assertIn("merge_datasets.py", error_text.getvalue())
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
