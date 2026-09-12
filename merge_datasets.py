"""Build one training index from the three downloaded leaf datasets.

Run from this project with: uv run --env-file .env python merge_datasets.py
The photographs stay in their original storage folders. The resulting
manifest.csv tells the model where each photograph is and what it shows.
"""

import argparse
import os
from pathlib import Path

from leafit.prepare import prepare


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path(os.environ.get("LEAFIT_DATA") or "."),
        help="Storage folder containing data/raw (default: LEAFIT_DATA, or the current folder)",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Output folder (default: <data-root>/data/prepared)",
    )
    parser.add_argument(
        "--groups", type=Path,
        help="Optional reviewed leaf groups CSV: source_dataset,image_path,group_id",
    )
    parser.add_argument("--seed", type=int, default=42, help="Reproducible split seed (default: 42)")
    parser.add_argument(
        "--mango-policy", choices=("deduplicate", "train-only"), default="deduplicate",
        help="Mango: keep one representative per detected family/label and split it, or retain the earlier train-only policy",
    )
    parser.add_argument(
        "--near-threshold", type=int, default=4,
        help="Perceptual hash distance for grouping similar copies, 0–16 (default: 4)",
    )
    parser.add_argument(
        "--strict-conflicts", action="store_true",
        help="Stop on identical photos with conflicting labels; default excludes every conflicting copy from the index",
    )
    return parser


def preparation_arguments(args):
    """Check the local layout before the slower image and duplicate checks."""
    root = args.data_root.expanduser().resolve()
    raw = root / "data" / "raw"
    sources = {
        "plantvillage": raw / "plantvillage",
        "cornell": raw / "cornell",
        "mango": raw / "mangoleafbd",
    }
    missing = [str(path) for path in sources.values() if not path.is_dir()]
    if missing:
        raise ValueError(
            "Dataset folders are missing:\n  " + "\n  ".join(missing)
            + "\nCheck that your storage is available. Set LEAFIT_DATA in .env and load it, "
            "or pass --data-root for the folder containing data/raw."
        )
    cornell_tables = sorted(sources["cornell"].rglob("train.csv"))
    if len(cornell_tables) != 1:
        raise ValueError(
            f"Expected exactly one labeled Cornell train.csv under {sources['cornell']}; "
            f"found {len(cornell_tables)}. Check the local Cornell import before merging."
        )
    if not 0 <= args.near_threshold <= 16:
        raise ValueError("--near-threshold must be between 0 and 16.")
    groups = args.groups.expanduser().resolve() if args.groups else None
    if groups is not None and not groups.is_file():
        raise ValueError(f"Reviewed groups CSV does not exist: {groups}")
    output = (args.output or root / "data" / "prepared").expanduser().resolve()
    return argparse.Namespace(
        **sources, output=output, groups=groups, leaf_map=None,
        seed=args.seed, near_threshold=args.near_threshold, allow_partial=False,
        exclude_conflicting_duplicates=not args.strict_conflicts,
        mango_policy=args.mango_policy,
    )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        prepared_args = preparation_arguments(args)
        print(f"Reading local datasets from {prepared_args.plantvillage.parent}")
        print(f"Writing the merged dataset index to {prepared_args.output}")
        if prepared_args.exclude_conflicting_duplicates:
            print("Exact copies with conflicting labels are excluded from the index; original files are preserved.")
        prepare(prepared_args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
