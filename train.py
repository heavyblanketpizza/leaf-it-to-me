"""Train MobileNetV4 on Orchard's merged CSV using plain PyTorch and timm.

Run: uv run --env-file .env python train.py
Quick real-data check: uv run --env-file .env python train.py --smoke
"""

import argparse
import os
from pathlib import Path

# Set the same project-local model and plotting caches as the other commands.
import orchard.__main__  # noqa: F401
from orchard.model import train, training_arguments


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    training_arguments(parser)
    parser.set_defaults(manifest=None, labels=None, output=None)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("ORCHARD_DATA") or "."),
                        help="Storage folder containing data/prepared (default: ORCHARD_DATA, or the current folder)")
    parser.add_argument("--smoke", action="store_true",
                        help="One training batch in each stage; evaluate full validation/test splits")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    prepared = args.data_root.expanduser().resolve() / "data" / "prepared"
    args.manifest = str(Path(args.manifest).expanduser().resolve()) if args.manifest else str(prepared / "manifest.csv")
    args.labels = str(Path(args.labels).expanduser().resolve()) if args.labels else str(prepared / "labels.json")
    args.output = args.output or ("runs/training-smoke" if args.smoke else "runs/weekend")
    args.output = str(Path(args.output).expanduser().resolve())
    if args.smoke:
        args.head_epochs = args.fine_tune_epochs = args.max_batches = 1
    try:
        for name in (args.manifest, args.labels):
            if not Path(name).is_file():
                raise ValueError(f"Missing prepared dataset file: {name}. Check ORCHARD_DATA or --data-root "
                                 "and run merge_datasets.py first.")
        train(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
