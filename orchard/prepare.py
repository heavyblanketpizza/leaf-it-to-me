"""Import untouched local photos, find related copies, and write a grouped split.

This deliberately uses local files: authentication/download instructions live in
the README, while preparation works the same way regardless of download method.
"""

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
from PIL import Image, ImageOps

from .labels import (
    BY_LABEL, CORNELL_LABELS, LABELS, MANGO_LABELS, PLANTVILLAGE_LABELS,
    normalize_name,
)
from .mango import CURATION_FIELDS, curate_mango

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MANIFEST_FIELDS = [
    "image_path", "source_dataset", "tree_type", "condition", "class_label",
    "group_id", "split", "relative_path", "sha256_pixels", "official_group",
    "reviewed_group",
]


class UnionFind:
    """Put connected photos in one bucket (a group cannot cross data splits)."""

    def __init__(self, n):
        self.parents = list(range(n))

    def find(self, i):
        while i != self.parents[i]:
            self.parents[i] = self.parents[self.parents[i]]
            i = self.parents[i]
        return i

    def union(self, a, b):
        self.parents[self.find(b)] = self.find(a)


class HashTree:
    """A BK tree avoids comparing every perceptual hash with every other hash."""

    def __init__(self):
        self.root = None

    def add(self, value, index):
        if self.root is None:
            self.root = [value, [index], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(index)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [index], {}]
                return
            node = node[2][distance]

    def query(self, value, radius):
        pending = [self.root] if self.root else []
        while pending:
            node = pending.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                for index in node[1]:
                    yield index, distance
            pending.extend(child for edge, child in node[2].items()
                           if distance - radius <= edge <= distance + radius)


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def find_images(directory):
    return sorted(path for path in directory.rglob("*")
                  if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                  and not any(part.startswith(".") for part in path.relative_to(directory).parts))


def make_row(path, root, source, label, official_group=""):
    entry = BY_LABEL[label]
    return dict(image_path=str(path.resolve()), relative_path=path.relative_to(root).as_posix(),
                source_dataset=source, tree_type=entry["tree_type"], condition=entry["condition"],
                class_label=label, official_group=official_group, reviewed_group="")


def pv_photo_id(filename):
    # The official Hugging Face builder uses this normalization for leaf-map.json.
    name = filename.replace("_final_masked", "").split("___")[-1].split("copy")[0]
    for extension in (".jpg", ".JPG", ".png", ".PNG"):
        name = name.replace(extension, "")
    return name.strip().lower()


def import_plantvillage(root, leaf_map_path, notes):
    root = root.expanduser().resolve()
    candidates = [root / "raw" / "color", root / "color", root]
    color = next((directory for directory in candidates if directory.is_dir() and
                  any((directory / name).is_dir() for name in PLANTVILLAGE_LABELS)), None)
    if color is None or any(part.lower() in {"grayscale", "segmented"} for part in color.parts):
        raise ValueError("PlantVillage must point to the original raw/color directory (or its repo root).")
    if leaf_map_path is None:
        leaf_map_path = next((parent / "leaf_grouping" / "leaf-map.json"
                              for parent in [root, color.parent, color.parent.parent]
                              if (parent / "leaf_grouping" / "leaf-map.json").is_file()), None)
    leaf_map = json.loads(Path(leaf_map_path).read_text()) if leaf_map_path else {}
    rows, missing = [], 0
    for folder_name, label in PLANTVILLAGE_LABELS.items():
        for image in find_images(color / folder_name):
            groups = leaf_map.get(pv_photo_id(image.name), [])
            if isinstance(groups, str):
                groups = [groups]
            candidates = [group for group in groups if folder_name in str(group)]
            if len(groups) == 1:
                candidates = groups
            group = str(candidates[0]) if len(set(candidates)) == 1 else ""
            missing += not bool(group)
            rows.append(make_row(image, color, "plantvillage", label, group))
    notes["plantvillage"] = {
        "import_root": str(color), "leaf_map": str(leaf_map_path) if leaf_map_path else None,
        "images_without_official_leaf_group": missing,
        "expected_full_color_tree_subset": 13241,
        "grouping_limitation": ("Unmapped photos rely on heuristic duplicate grouping; leaf identity is not proven."
                                if missing else "Official leaf groups loaded for all imported files."),
    }
    return rows


def import_cornell(root, notes):
    root = root.expanduser().resolve()
    tables = sorted(root.rglob("train.csv"))
    if len(tables) != 1:
        raise ValueError(f"Expected exactly one Cornell train.csv under {root}; found {len(tables)}.")
    table = tables[0]
    by_stem = defaultdict(list)
    for path in find_images(table.parent):
        by_stem[path.stem].append(path)
    rows, seen = [], set()
    with table.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", *CORNELL_LABELS}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Cornell train.csv needs columns {sorted(required)}.")
        for line, record in enumerate(reader, start=2):
            image_id = record["image_id"]
            if not image_id.startswith("Train_") or image_id in seen:
                raise ValueError(f"Unexpected or repeated publicly labelled image_id at line {line}: {image_id}")
            seen.add(image_id)
            values = {key: float(record[key]) for key in CORNELL_LABELS}
            if any(value not in (0, 1) for value in values.values()) or sum(values.values()) != 1:
                raise ValueError(f"Cornell line {line} is not exactly one public one-hot label.")
            condition = next(key for key, value in values.items() if value == 1)
            matches = by_stem.get(image_id, [])
            if len(matches) != 1:
                raise ValueError(f"Expected one image for Cornell {image_id}; found {len(matches)}.")
            rows.append(make_row(matches[0], table.parent, "cornell", CORNELL_LABELS[condition]))
    notes["cornell"] = {
        "import_root": str(table.parent), "public_labeled_rows": len(rows),
        "ignored_test_images": sum(path.stem.startswith("Test_") for paths in by_stem.values() for path in paths),
        "grouping_limitation": "No public leaf identifiers: exact/perceptual duplicate checks are a heuristic.",
    }
    return rows


def import_mango(root, notes):
    root = root.expanduser().resolve()
    folders = {normalize_name(name): label for name, label in MANGO_LABELS.items()}
    # Archives may add a wrapper directory; infer a label from the nearest category ancestor.
    rows, unrecognized = [], []
    for path in find_images(root):
        matching = [folders[normalize_name(parent.name)] for parent in path.parents
                    if parent != root.parent and root in [parent, *parent.parents]
                    and normalize_name(parent.name) in folders]
        if not matching:
            unrecognized.append(str(path))
            continue
        if len(set(matching)) != 1:
            raise ValueError(f"Conflicting Mango category directories: {path}")
        rows.append(make_row(path, root, "mangoleafbd", matching[0]))
    if unrecognized:
        raise ValueError(f"Unrecognized Mango category for {len(unrecognized)} images, e.g. {unrecognized[0]}")
    notes["mangoleafbd"] = {
        "import_root": str(root), "expected_full_release_images": 4000,
        "grouping_limitation": "4,000 augmented photos represent about 1,800 leaves; no published complete leaf map.",
    }
    return rows


def apply_reviewed_groups(rows, path):
    if not path:
        return
    lookup = defaultdict(list)
    for i, row in enumerate(rows):
        keys = {row["image_path"], row["relative_path"], Path(row["image_path"]).name}
        if row["source_dataset"] == "plantvillage":
            keys.update({"raw/color/" + row["relative_path"], "color/" + row["relative_path"]})
        for key in keys:
            lookup[(row["source_dataset"], key)].append(i)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"source_dataset", "image_path", "group_id"} <= set(reader.fieldnames or []):
            raise ValueError("--groups CSV requires source_dataset,image_path,group_id columns.")
        for record in reader:
            key = (record["source_dataset"].strip(), record["image_path"].strip())
            matches = lookup.get(key, [])
            if len(matches) != 1:
                raise ValueError(f"Group-map image must match exactly once: {key}; found {len(matches)}.")
            row = rows[matches[0]]
            group = record["group_id"].strip()
            if not group or (row["reviewed_group"] and row["reviewed_group"] != group):
                raise ValueError(f"Missing/conflicting reviewed group for {key}.")
            row["reviewed_group"] = group


def fingerprint(path):
    """Hash decoded RGB pixels and all eight right-angle rotation/flip variants."""
    with Image.open(path) as original:
        rgb = ImageOps.exif_transpose(original).convert("RGB")
        rgb.load()
        exact = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        small = rgb.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    transforms = [None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180,
                  Image.Transpose.ROTATE_270, Image.Transpose.FLIP_LEFT_RIGHT,
                  Image.Transpose.FLIP_TOP_BOTTOM, Image.Transpose.TRANSPOSE,
                  Image.Transpose.TRANSVERSE]
    hashes = [int(str(imagehash.phash(small if transform is None else small.transpose(transform))), 16)
              for transform in transforms]
    return exact, hashes


def connect_related(rows, threshold, output, exclude_conflicts=False):
    valid, bad, fingerprints = [], [], []
    for i, row in enumerate(rows):
        try:
            exact, hashes = fingerprint(row["image_path"])
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            bad.append({**row, "error": str(error)})
            continue
        row["sha256_pixels"] = exact
        valid.append(row)
        fingerprints.append(hashes)
        if (i + 1) % 500 == 0:
            print(f"Checked {i + 1:,}/{len(rows):,} photographs", flush=True)
    write_csv(output / "bad_images.csv", bad, ["image_path", "source_dataset", "error"])
    groups, tree, exact_seen = UnionFind(len(valid)), HashTree(), {}
    known = {}
    duplicates, conflicts, review_candidates = [], [], []
    conflicting_hashes = set()
    for i, (row, hashes) in enumerate(zip(valid, fingerprints)):
        # Explicit reviewed groups supplement official groups; they cannot split them apart.
        for field in ("official_group", "reviewed_group"):
            if row[field]:
                namespace = ("global" if field == "reviewed_group" and row[field].startswith("global:")
                             else row["source_dataset"])
                key = (field, namespace, row[field])
                if key in known:
                    groups.union(i, known[key])
                else:
                    known[key] = i
        if row["sha256_pixels"] in exact_seen:
            matches = {exact_seen[row["sha256_pixels"]]: ("exact_pixels", 0)}
        else:
            matches = {}
            for hash_value in set(hashes):
                for other, distance in tree.query(hash_value, threshold):
                    if other not in matches or distance < matches[other][1]:
                        matches[other] = ("perceptual_candidate", distance)
            tree.add(hashes[0], i)
            exact_seen[row["sha256_pixels"]] = i
        for other, (kind, distance) in matches.items():
            previous = valid[other]
            pair = dict(image_path_a=previous["image_path"], image_path_b=row["image_path"],
                        source_a=previous["source_dataset"], source_b=row["source_dataset"],
                        label_a=previous["class_label"], label_b=row["class_label"],
                        match_type=kind, phash_distance=distance)
            duplicates.append(pair)
            groups.union(i, other)
            if previous["class_label"] != row["class_label"]:
                if kind == "exact_pixels":
                    conflicts.append(pair)
                    conflicting_hashes.add(row["sha256_pixels"])
                else:
                    # Similarity is uncertain: preserve both labels and keep
                    # the candidate pair together, without treating it as fact.
                    review_candidates.append(pair)
        if (i + 1) % 500 == 0:
            print(f"Compared duplicate hashes for {i + 1:,}/{len(valid):,} photographs", flush=True)
    fields = ["image_path_a", "image_path_b", "source_a", "source_b", "label_a", "label_b",
              "match_type", "phash_distance"]
    write_csv(output / "duplicate_pairs.csv", duplicates, fields)
    write_csv(output / "conflicts.csv", conflicts, fields)
    write_csv(output / "review_candidates.csv", review_candidates, fields)
    excluded = [dict(row, reason="Exact decoded pixels have conflicting class labels; all copies excluded.")
                for row in valid if exclude_conflicts and row["sha256_pixels"] in conflicting_hashes]
    write_csv(output / "excluded_conflicts.csv", excluded,
              ["image_path", "source_dataset", "class_label", "sha256_pixels", "reason"])
    if conflicts and not exclude_conflicts:
        raise ValueError(
            f"{len(conflicts)} exact decoded-pixel duplicates have different labels; inspect {output / 'conflicts.csv'}. "
            "Review the labels, or use --exclude-conflicting-duplicates to leave every ambiguous copy "
            "out of the manifest while preserving all original files."
        )
    if excluded:
        # Keep connections through excluded photos. They can bridge official leaf
        # groups, and their removal must not send surviving relatives to different splits.
        surviving_indexes = [i for i, row in enumerate(valid)
                             if row["sha256_pixels"] not in conflicting_hashes]
        surviving_groups, old_roots = UnionFind(len(surviving_indexes)), {}
        for new_index, old_index in enumerate(surviving_indexes):
            root = groups.find(old_index)
            if root in old_roots:
                surviving_groups.union(new_index, old_roots[root])
            else:
                old_roots[root] = new_index
        valid = [valid[i] for i in surviving_indexes]
        groups = surviving_groups
        print(f"Excluded {len(excluded):,} images from {len(conflicting_hashes):,} exact-pixel families "
              f"with conflicting labels ({len(conflicts):,} conflict pairs). Originals are unchanged; "
              f"see {output / 'excluded_conflicts.csv'}.", flush=True)
    return valid, groups, {
        "unreadable_images": len(bad), "exact_duplicate_pairs": sum(pair["match_type"] == "exact_pixels" for pair in duplicates),
        "perceptual_candidate_pairs": sum(pair["match_type"] == "perceptual_candidate" for pair in duplicates),
        "cross_label_perceptual_candidates": len(review_candidates),
        "cross_dataset_duplicate_pairs": sum(pair["source_a"] != pair["source_b"] for pair in duplicates),
        "exact_cross_label_conflict_pairs": len(conflicts),
        "excluded_conflicting_images": len(excluded),
        "excluded_conflicting_pixel_hashes": len(conflicting_hashes) if exclude_conflicts else 0,
    }


def assign_splits(rows, union, seed, mango_reviewed=False, *, mango_evaluation_allowed=None):
    # The explicit option also allows exploratory automatic Mango grouping.
    # Keep mango_reviewed as a compatibility argument for earlier practice code.
    if mango_evaluation_allowed is None:
        mango_evaluation_allowed = mango_reviewed
    rng = random.Random(seed)
    components = defaultdict(list)
    for i in range(len(rows)):
        components[union.find(i)].append(i)
    split_names = ("train", "val", "test")
    counts = {split: Counter() for split in split_names}
    totals = Counter(row["class_label"] for row in rows)
    targets = {split: {label: count * fraction for label, count in totals.items()}
               for split, fraction in zip(split_names, (.8, .1, .1))}
    assigned = {}
    histograms = {key: Counter(rows[i]["class_label"] for i in indexes)
                  for key, indexes in components.items()}

    def place(key, split):
        assigned[key] = split
        counts[split].update(histograms[key])

    for key, indexes in components.items():
        if not mango_evaluation_allowed and any(rows[i]["source_dataset"] == "mangoleafbd" for i in indexes):
            place(key, "train")
    available = [key for key in components if key not in assigned]
    rng.shuffle(available)
    # Reserve two small, independent groups for each evaluable class. Classes with
    # fewer than three independent groups cannot support all three splits.
    for label in sorted(totals):
        candidates = sorted((key for key in available if label in histograms[key]),
                            key=lambda key: len(components[key]))
        if len(candidates) < 3:
            continue
        for split in ("test", "val"):
            if counts[split][label]:
                continue
            candidate = next((key for key in candidates if key not in assigned), None)
            if candidate is not None:
                place(candidate, split)
    # Greedy balancing uses image counts, while the indivisible units are groups.
    for key in sorted(available, key=lambda key: -len(components[key])):
        if key in assigned:
            continue
        histogram = histograms[key]

        def cost(split):
            return sum(((counts[split][label] + number - targets[split][label]) ** 2
                        - (counts[split][label] - targets[split][label]) ** 2)
                       / max(targets[split][label], 1)
                       for label, number in histogram.items())

        place(key, min(split_names, key=cost))
    for key, indexes in components.items():
        identifiers = sorted(f"{rows[i]['source_dataset']}:{rows[i]['relative_path']}" for i in indexes)
        group_id = "group_" + hashlib.sha256("\n".join(identifiers).encode()).hexdigest()[:16]
        for i in indexes:
            rows[i].update(group_id=group_id, split=assigned[key])
    return {split: dict(sorted(counts[split].items())) for split in split_names}


def prepare(args):
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not 0 <= args.near_threshold <= 16:
        raise ValueError("--near-threshold must be between 0 and 16 (default 4).")
    mango_policy = getattr(args, "mango_policy", "train-only")
    if mango_policy not in {"deduplicate", "train-only"}:
        raise ValueError("--mango-policy must be deduplicate or train-only.")
    if not any((args.plantvillage, args.cornell, args.mango)):
        raise ValueError("Provide local dataset paths; see README for exact download/import instructions.")
    if not args.allow_partial and not all((args.plantvillage, args.cornell, args.mango)):
        raise ValueError("Supply all three datasets, or use --allow-partial for a small practice run.")
    rows, notes = [], {}
    if args.plantvillage:
        rows.extend(import_plantvillage(args.plantvillage, args.leaf_map, notes))
    if args.cornell:
        rows.extend(import_cornell(args.cornell, notes))
    if args.mango:
        rows.extend(import_mango(args.mango, notes))
    rows.sort(key=lambda row: (row["source_dataset"], row["relative_path"]))
    if not rows:
        raise ValueError("No supported images found. Check the extracted directory paths.")
    apply_reviewed_groups(rows, args.groups)
    imported = len(rows)
    exclude_conflicts = getattr(args, "exclude_conflicting_duplicates", False)
    rows, union, duplicate_stats = connect_related(rows, args.near_threshold, output, exclude_conflicts)
    # Measure review coverage before removing representatives. Automatic links
    # never become human-reviewed leaf identities in the report or manifest.
    mango_before = [row for row in rows if row["source_dataset"] == "mangoleafbd"]
    mango_reviewed = bool(mango_before) and all(row["reviewed_group"] for row in mango_before)
    mango_curation = None
    if mango_policy == "deduplicate" and mango_before:
        rows, union, mango_curation = curate_mango(rows, union, output)
        print(f"Mango: kept {mango_curation['retained_images']:,}/{mango_curation['input_images']:,} "
              "representatives from detected families; originals are unchanged.", flush=True)
    else:
        write_csv(output / "mango_curation.csv", [], CURATION_FIELDS)
    missing = sorted(set(BY_LABEL) - {row["class_label"] for row in rows})
    observed = set(BY_LABEL) - set(missing)
    if missing and not args.allow_partial:
        raise ValueError(f"Missing classes after readability and duplicate checks: {missing}. Use --allow-partial only for practice subsets.")
    if not rows:
        raise ValueError(f"No usable images. See {output / 'bad_images.csv'} and excluded_conflicts.csv.")
    mango_rows = [row for row in rows if row["source_dataset"] == "mangoleafbd"]
    mango_evaluation_allowed = mango_reviewed or mango_policy == "deduplicate"
    if mango_rows:
        notes["mangoleafbd"]["split_policy"] = (
            "Deduplicated representatives; exploratory grouped train/val/test using capture metadata and visual matches."
            if mango_policy == "deduplicate" else
            "All files have explicitly reviewed groups: eligible for grouped train/val/test."
            if mango_reviewed else "Train only: complete reviewed leaf grouping not provided."
        )
    distributions = assign_splits(rows, union, args.seed, mango_evaluation_allowed=mango_evaluation_allowed)
    report = dict(
        imported_images=imported, retained_images=len(rows),
        by_source=dict(sorted(Counter(row["source_dataset"] for row in rows).items())),
        by_class=dict(sorted(Counter(row["class_label"] for row in rows).items())),
        by_split=dict(sorted(Counter(row["split"] for row in rows).items())),
        class_distribution_by_split=distributions,
        groups=len({row["group_id"] for row in rows}),
        groups_by_split={split: len({row["group_id"] for row in rows if row["split"] == split})
                         for split in ("train", "val", "test")},
        missing_classes=missing,
        missing_training_classes=sorted(observed - set(distributions["train"])),
        missing_evaluation_classes={split: sorted(observed - set(distributions[split])) for split in ("val", "test")},
        seed=args.seed, split_target=[.8, .1, .1], near_duplicate_phash_distance=args.near_threshold,
        mango_groups_completely_reviewed=mango_reviewed,
        mango_policy=mango_policy, mango_curation=mango_curation,
        exclude_conflicting_duplicates=exclude_conflicts,
        grouping_warning="Perceptual hashes are heuristics: crops, arbitrary rotations, and separate photographs of one leaf may be missed. Inspect duplicate_pairs.csv; no claim of perfect leaf isolation.",
        duplicate_policy=("Preserve all original files and labels; exclude every copy of exact-pixel families "
                          "with conflicting labels when requested; optionally retain one Mango representative "
                          "per detected family and label; keep all surviving related rows in the same split."),
        perceptual_label_policy="Cross-label perceptual candidates keep their original labels and share a split; inspect review_candidates.csv for possible false matches.",
        datasets=notes, **duplicate_stats,
    )
    # Only publish manifests after validation succeeds.
    write_csv(output / "manifest.csv", rows, MANIFEST_FIELDS)
    observed_labels = [dict(entry, index=index) for index, entry in enumerate(
        entry for entry in LABELS if entry["label"] not in missing
    )]
    (output / "labels.json").write_text(json.dumps(observed_labels, indent=2) + "\n")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Retained {len(rows):,}/{imported:,} readable target images in {report['groups']:,} groups.")
    print("Source counts:", report["by_source"])
    print("Split counts:", report["by_split"])
    print(f"{'class':32s} {'train':>7s} {'val':>7s} {'test':>7s} {'total':>7s}")
    for label in sorted(BY_LABEL):
        values = [distributions[split].get(label, 0) for split in ("train", "val", "test")]
        print(f"{label:32s} {values[0]:7,d} {values[1]:7,d} {values[2]:7,d} {sum(values):7,d}")
    for split, absent in report["missing_evaluation_classes"].items():
        if absent:
            print(f"WARNING: {split} has no examples for: {', '.join(absent)}")
    if report["missing_training_classes"]:
        print("WARNING: train has no examples for: " + ", ".join(report["missing_training_classes"]) +
              ". More independent groups or reviewed split changes are needed before training.")
    if report["cross_label_perceptual_candidates"]:
        print(f"WARNING: {report['cross_label_perceptual_candidates']} cross-label perceptual candidates "
              f"share splits but retain their labels. Inspect {output / 'review_candidates.csv'}.")
    if mango_rows and mango_policy == "deduplicate":
        print("NOTE: Mango evaluation is exploratory. Automatic groups do not verify original leaves "
              "or catch every augmented/re-photographed view; inspect mango_curation.csv.")
    elif mango_rows and not mango_reviewed:
        print("WARNING: Mango is train-only until every image has a reviewed leaf group in --groups.")
    print(f"Wrote {output / 'manifest.csv'} and report.json; originals were not changed.")


def add_commands(subparsers):
    parser = subparsers.add_parser("prepare", help="Import local datasets and create grouped 80/10/10 splits")
    parser.add_argument("--plantvillage", type=Path, help="PlantVillage repository or original raw/color directory")
    parser.add_argument("--cornell", type=Path, help="Extracted FGVC7 directory containing train.csv and images")
    parser.add_argument("--mango", type=Path, help="Extracted MangoLeafBD directory containing category folders")
    parser.add_argument("--output", type=Path, default=Path("data/prepared"))
    parser.add_argument("--leaf-map", type=Path, help="Optional official PlantVillage leaf_grouping/leaf-map.json")
    parser.add_argument("--groups", type=Path, help="Reviewed CSV: source_dataset,image_path,group_id; paths relative to import_root; global: group IDs can unite sources")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mango-policy", choices=("train-only", "deduplicate"), default="train-only",
                        help="Mango: earlier train-only policy, or deduplicated representatives with exploratory grouped splits")
    parser.add_argument("--near-threshold", type=int, default=4, help="Maximum D4 perceptual-hash Hamming distance")
    parser.add_argument("--allow-partial", action="store_true", help="Allow missing datasets/classes for practice subsets")
    parser.add_argument("--exclude-conflicting-duplicates", action="store_true",
                        help="Exclude all copies of exact-pixel families with conflicting labels; preserve originals")
    parser.set_defaults(func=prepare)
