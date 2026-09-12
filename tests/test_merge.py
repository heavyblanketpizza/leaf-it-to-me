"""Check the convenient merge command without scanning real photographs."""

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import merge_datasets


class MergeCommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for source in ("plantvillage", "cornell", "mangoleafbd"):
            (self.root / "data" / "raw" / source).mkdir(parents=True)
        (self.root / "data" / "raw" / "cornell" / "train.csv").write_text(
            "image_id,healthy,multiple_diseases,rust,scab\n", encoding="utf-8"
        )

    def test_default_local_layout_and_conflict_policy(self):
        with patch.dict(os.environ, {}, clear=True):
            args = merge_datasets.build_parser().parse_args([])
        self.assertEqual(args.data_root, Path("."))
        with patch.object(merge_datasets, "prepare") as prepare, redirect_stdout(io.StringIO()):
            self.assertEqual(merge_datasets.main(["--data-root", str(self.root)]), 0)
        prepared = prepare.call_args.args[0]
        self.assertEqual(prepared.plantvillage, self.root.resolve() / "data/raw/plantvillage")
        self.assertEqual(prepared.cornell, self.root.resolve() / "data/raw/cornell")
        self.assertEqual(prepared.mango, self.root.resolve() / "data/raw/mangoleafbd")
        self.assertEqual(prepared.output, self.root.resolve() / "data/prepared")
        self.assertTrue(prepared.exclude_conflicting_duplicates)
        self.assertFalse(prepared.allow_partial)
        self.assertEqual(prepared.seed, 42)
        self.assertEqual(prepared.near_threshold, 4)
        self.assertEqual(prepared.mango_policy, "deduplicate")
        self.assertIsNone(prepared.groups)
        self.assertIsNone(prepared.leaf_map)

    def test_environment_root_and_explicit_override_reach_preparation(self):
        environment_root = self.root / "storage with spaces"
        environment_root.mkdir()
        (self.root / "data").rename(environment_root / "data")
        with patch.dict(os.environ, {"LEAFIT_DATA": str(environment_root)}):
            with patch.object(merge_datasets, "prepare") as prepare, redirect_stdout(io.StringIO()):
                self.assertEqual(merge_datasets.main([]), 0)
            self.assertEqual(prepare.call_args.args[0].output, environment_root.resolve() / "data/prepared")
            (environment_root / "data").rename(self.root / "data")
            with patch.object(merge_datasets, "prepare") as prepare, redirect_stdout(io.StringIO()):
                self.assertEqual(merge_datasets.main(["--data-root", str(self.root)]), 0)
            self.assertEqual(prepare.call_args.args[0].output, self.root.resolve() / "data/prepared")

    def test_custom_options_and_strict_policy_reach_preparation(self):
        groups = self.root / "reviewed.csv"
        groups.write_text("source_dataset,image_path,group_id\n", encoding="utf-8")
        output = self.root / "other prepared"
        with patch.object(merge_datasets, "prepare") as prepare, redirect_stdout(io.StringIO()):
            merge_datasets.main([
                "--data-root", str(self.root), "--output", str(output),
                "--groups", str(groups), "--seed", "19", "--near-threshold", "2",
                "--strict-conflicts", "--mango-policy", "train-only",
            ])
        prepared = prepare.call_args.args[0]
        self.assertEqual(prepared.output, output.resolve())
        self.assertEqual(prepared.groups, groups.resolve())
        self.assertEqual(prepared.seed, 19)
        self.assertEqual(prepared.near_threshold, 2)
        self.assertFalse(prepared.exclude_conflicting_duplicates)
        self.assertEqual(prepared.mango_policy, "train-only")

    def test_missing_source_or_label_table_stops_before_preparation(self):
        for missing in ("data/raw/mangoleafbd", "data/raw/cornell/train.csv"):
            with self.subTest(missing=missing):
                path = self.root / missing
                moved = self.root / "temporarily-moved"
                path.rename(moved)
                try:
                    with patch.object(merge_datasets, "prepare") as prepare, redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as error:
                            merge_datasets.main(["--data-root", str(self.root)])
                    self.assertEqual(error.exception.code, 2)
                    prepare.assert_not_called()
                finally:
                    moved.rename(path)

    def test_invalid_review_file_or_threshold_stops_before_preparation(self):
        for extra in (["--groups", str(self.root / "missing.csv")], ["--near-threshold", "17"]):
            with self.subTest(options=extra):
                with patch.object(merge_datasets, "prepare") as prepare, redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        merge_datasets.main(["--data-root", str(self.root), *extra])
                self.assertEqual(error.exception.code, 2)
                prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
