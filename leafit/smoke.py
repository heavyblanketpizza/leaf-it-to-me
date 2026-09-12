"""Generate fake pictures to test the plumbing, not plant-recognition quality."""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .labels import PLANTVILLAGE_LABELS, CORNELL_LABELS, MANGO_LABELS


def make_fixture(root):
    """Six distinct fake leaves per source label, two related views of each."""
    root = Path(root).resolve()
    rng = np.random.default_rng(123)
    groups, train_rows = [], []
    pv_map = {}
    sequence = 0
    sources = [
        ("plantvillage", PLANTVILLAGE_LABELS),
        ("cornell", CORNELL_LABELS),
        ("mangoleafbd", MANGO_LABELS),
    ]
    for source, mapping in sources:
        source_root = root / source
        for source_label, label in mapping.items():
            for leaf in range(6):
                # No real leaves! Random pixels make accidental duplicates unlikely.
                pixels = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
                original = Image.fromarray(pixels)
                for view in range(2):
                    sequence += 1
                    if source == "plantvillage":
                        name = f"fixture-{sequence}___photo{label}{leaf}{view}.png"
                        relative = Path("raw/color") / source_label / name
                        pv_map[f"photo{label}{leaf}{view}"] = [f"{source_label}:::{leaf}"]
                    elif source == "cornell":
                        name = f"Train_{sequence}"
                        relative = Path("images") / f"{name}.jpg"
                        train_rows.append({"image_id": name, **{key: int(key == source_label) for key in CORNELL_LABELS}})
                    else:
                        relative = Path(source_label) / f"leaf-{leaf}-view-{view}.png"
                    path = source_root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    (original if view == 0 else ImageOps.mirror(original)).save(path)
                    groups.append({"source_dataset": source, "image_path": str(relative),
                                   "group_id": f"fixture-{source}-{label}-{leaf}"})
    (root / "plantvillage/leaf_grouping").mkdir(exist_ok=True)
    (root / "plantvillage/leaf_grouping/leaf-map.json").write_text(json.dumps(pv_map))
    with (root / "cornell/train.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image_id", *CORNELL_LABELS])
        writer.writeheader()
        writer.writerows(train_rows)
    # A public test image intentionally has no ground truth and must be ignored.
    Image.new("RGB", (32, 32), "red").save(root / "cornell/images/Test_0.jpg")
    (root / "cornell/test.csv").write_text("image_id\nTest_0\n")
    with (root / "groups.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["source_dataset", "image_path", "group_id"])
        writer.writeheader()
        writer.writerows(groups)
    return root


def smoke(args):
    from .__main__ import main
    output = Path(args.output).resolve()
    raw = make_fixture(output / "synthetic-raw")
    prepared = output / "prepared"
    print("SMOKE TEST: synthetic random pixels; scores do not measure disease recognition.", flush=True)
    main(["prepare", "--plantvillage", str(raw / "plantvillage"),
          "--cornell", str(raw / "cornell"), "--mango", str(raw / "mangoleafbd"),
          "--groups", str(raw / "groups.csv"), "--output", str(prepared)])
    train = ["train", "--manifest", str(prepared / "manifest.csv"),
             "--labels", str(prepared / "labels.json"), "--output", str(output / "model"),
             "--head-epochs", "1", "--fine-tune-epochs", "1", "--max-batches", "1",
             "--batch-size", "4", "--device", args.device]
    if args.no_pretrained:
        train.append("--no-pretrained")
    main(train)
    sample = next((raw / "mangoleafbd").rglob("*.png"))
    main(["predict", "--checkpoint", str(output / "model/best.pt"), "--image", str(sample),
          "--device", args.device])
    (output / "SMOKE_ONLY.txt").write_text("Synthetic random-pixel fixture. These metrics are not plant-recognition results.\n")


def add_commands(subparsers):
    parser = subparsers.add_parser("smoke", help="Test all imports and both training stages on synthetic pixels")
    parser.add_argument("--output", default="runs/smoke")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--no-pretrained", action="store_true", help="Offline wiring test with random starting weights")
    parser.set_defaults(func=smoke)
