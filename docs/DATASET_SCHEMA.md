# The merged dataset: CSV + JSON + original photos

The finished dataset is a table of photo paths and answers, called a **manifest**. One row means one usable photo. The table brings PlantVillage, Cornell, and MangoLeafBD together while the original photos stay in their source folders.

The training code reads `manifest.csv` to find each photo and its split. It reads `labels.json` to turn a label such as `apple_scab` into the output number the model learns to predict. These two files belong together.

## Build it

From the project folder, after [setting up and loading `.env`](../README.md):

```bash
# Run from the project folder after loading .env (see README).
uv sync --locked
uv run python merge_datasets.py
```

The script reads `ORCHARD_DATA` from the environment, falling back to the current folder when it is unset or empty. `--data-root` overrides that setting. It uses the downloaded folders beneath `data/raw/` and writes the result to:

```text
$ORCHARD_DATA/data/prepared/
├── manifest.csv
├── labels.json
├── report.json
├── bad_images.csv
├── duplicate_pairs.csv
├── conflicts.csv
├── excluded_conflicts.csv
├── review_candidates.csv
└── mango_curation.csv
```

To choose a separate output folder beneath the configured storage root:

```bash
uv run python merge_datasets.py \
  --data-root "$ORCHARD_DATA" \
  --output "$ORCHARD_DATA/data/prepared-experiment" \
  --seed 42
```

The script opens photos with Pillow, checks that they decode, respects the camera's EXIF rotation instruction, and converts them to RGB in memory. It checks exact pixels and visual similarity to group related photos. Original files remain unchanged. Resizing and normalization for MobileNetV4 happen when training loads a photo; preparation only creates small temporary thumbnails for similarity checks.

## `manifest.csv`: the photo table

This is a UTF-8 CSV with a header row and the following **11 columns, in this order**. CSV stores text; there are no numeric training targets in this file. Only `official_group` and `reviewed_group` may be empty.

| Column | Type / allowed values | Meaning and example |
|---|---|---|
| `image_path` | String; absolute file path | Where to open the original photo, e.g. `/path/to/storage/data/raw/cornell/images/example_leaf.jpg` (a placeholder). |
| `source_dataset` | String: `plantvillage`, `cornell`, or `mangoleafbd` | Which dataset supplied this photo. |
| `tree_type` | String: `apple`, `cherry`, `peach`, `orange`, or `mango` | The dataset's tree/crop category. These categories are not precise botanical species identifiers. |
| `condition` | String; condition from the mapping below | The answer for this leaf, e.g. `scab`, `healthy`, or `multiple_diseases`. |
| `class_label` | String; one of the 18 labels below | The combined answer, e.g. `apple_scab`. It equals `tree_type + "_" + condition`. |
| `group_id` | String: `group_` followed by 16 hexadecimal characters | The final group containing this photo and its detected relatives, e.g. `group_0000000000000000`. Every row with this ID has the same split. |
| `split` | String: `train`, `val`, or `test` | `train` teaches the model; `val` helps choose its best checkpoint; `test` measures that checkpoint afterward. |
| `relative_path` | String; path relative to the source's recorded import root | A way to trace the photo within its source, e.g. `images/example_leaf.jpg`. See `report.json` → `datasets` → source → `import_root`. |
| `sha256_pixels` | String; 64 hexadecimal characters | A fingerprint of the image dimensions and decoded, correctly oriented RGB pixels. Equal values identify exact pixel copies, even if file metadata differs. |
| `official_group` | String, or empty | A leaf identifier supplied by the source, e.g. `Apple___Apple_scab:::79.0` from PlantVillage. |
| `reviewed_group` | String, or empty | A leaf identifier you supply through `--groups` after reviewing related photos. |

For example, a Cornell row could look like this (illustrative values):

```json
{
  "image_path": "/path/to/storage/data/raw/cornell/images/example_leaf.jpg",
  "source_dataset": "cornell",
  "tree_type": "apple",
  "condition": "scab",
  "class_label": "apple_scab",
  "group_id": "group_0000000000000000",
  "split": "train",
  "relative_path": "images/example_leaf.jpg",
  "sha256_pixels": "0000000000000000000000000000000000000000000000000000000000000000",
  "official_group": "",
  "reviewed_group": ""
}
```

The example uses JSON so the names and values are easy to read; the actual photo table is `manifest.csv`. `/path/to/storage` is a generic example, not a setting to copy. The real table contains resolved paths, not environment-variable expressions. If the data location changes, update and reload `.env`, then run the merge command again so those paths are updated. Keep generated manifests and reports out of Git because they contain local paths.

## `labels.json`: output numbers and their meanings

This file is a JSON list. Each item has exactly these fields:

| Field | Type | Meaning |
|---|---|---|
| `index` | Integer | Zero-based model output number: `0` through `17` for the complete dataset. |
| `label` | String | The combined label; matches `class_label` in the CSV. |
| `tree_type` | String | The tree/crop category to display. |
| `condition` | String | The condition to display. |

For example, the item for scab is:

```json
{
  "index": 4,
  "label": "apple_scab",
  "tree_type": "apple",
  "condition": "scab"
}
```

The full 18-class mapping is:

| Index | Label | Tree type | Condition |
|---:|---|---|---|
| 0 | `apple_black_rot` | `apple` | `black_rot` |
| 1 | `apple_cedar_apple_rust` | `apple` | `cedar_apple_rust` |
| 2 | `apple_healthy` | `apple` | `healthy` |
| 3 | `apple_multiple_diseases` | `apple` | `multiple_diseases` |
| 4 | `apple_scab` | `apple` | `scab` |
| 5 | `cherry_healthy` | `cherry` | `healthy` |
| 6 | `cherry_powdery_mildew` | `cherry` | `powdery_mildew` |
| 7 | `mango_anthracnose` | `mango` | `anthracnose` |
| 8 | `mango_bacterial_canker` | `mango` | `bacterial_canker` |
| 9 | `mango_cutting_weevil` | `mango` | `cutting_weevil` |
| 10 | `mango_dieback` | `mango` | `dieback` |
| 11 | `mango_gall_midge` | `mango` | `gall_midge` |
| 12 | `mango_healthy` | `mango` | `healthy` |
| 13 | `mango_powdery_mildew` | `mango` | `powdery_mildew` |
| 14 | `mango_sooty_mould` | `mango` | `sooty_mould` |
| 15 | `orange_citrus_greening` | `orange` | `citrus_greening` |
| 16 | `peach_bacterial_spot` | `peach` | `bacterial_spot` |
| 17 | `peach_healthy` | `peach` | `healthy` |

Training looks up `class_label` in this file to get the integer target. Keep the generated mapping with the model checkpoint; the separate practice-subset command can generate a shorter mapping with different indices.

Equivalent apple answers from PlantVillage and Cornell share one label. Cornell's `rust` maps to `apple_cedar_apple_rust`. Its `multiple_diseases` remains one separate class. Orange has only `citrus_greening` examples, so this dataset cannot teach a healthy-versus-diseased orange decision.

## Counts, exclusions, and group splits

`report.json` records your import's source, class, split, and group counts,
unreadable files, conflicting copies, and missing evaluation classes. Counts
are computed from your files rather than assumed from the source totals.

The merge script excludes all members of exact-pixel groups whose labels
conflict, while preserving the source files. `--strict-conflicts` instead stops
preparation when it finds a conflict. Visually similar photos with different
labels keep their labels and share a group for review.

Groups stay together while aiming for 80/10/10. PlantVillage supplies leaf
metadata; Cornell has no public leaf identifiers, so automatic duplicate and
near-duplicate checks cannot guarantee physical-leaf independence.

The default `merge_datasets.py` command uses `--mango-policy deduplicate`. It joins Mango photos that share camera make/model and EXIF capture time with the global exact-pixel, rotated/flipped perceptual-hash, official-leaf, and reviewed-group links. From each detected family it keeps one Mango representative **per class**, so an uncertain visual match across conditions does not erase a condition. All remaining members of a family stay in one split. It then aims for grouped 80/10/10 across Mango as well as the other sources. Downloaded Mango files stay unchanged on disk; only extra rows are omitted from the training manifest.

These are representatives of automatically detected families, not verified originals or a verified count of physical leaves. Similarity and capture timestamps can miss related photos or join separate photos. The report keeps `mango_groups_completely_reviewed` false unless complete manual group coverage was actually supplied. A reviewed `--groups` CSV can improve grouping; the audit explains the remaining limitations in [MANGO_AUDIT.md](MANGO_AUDIT.md).

Use `uv run python merge_datasets.py --mango-policy train-only` to keep Mango in training unless complete reviewed groups are supplied. The lower-level `uv run python -m orchard prepare` command still defaults to `train-only`; pass `--mango-policy deduplicate` when using it to select and split Mango representatives.

## Audit files: how preparation made its choices

| File | Contents |
|---|---|
| `report.json` | Counts by source, class, and split; groups; settings; source import roots; missing classes; grouping limits; duplicate statistics. |
| `bad_images.csv` | `image_path`, `source_dataset`, `error` for photos that failed to decode. A header-only file means no unreadable images were found. |
| `duplicate_pairs.csv` | Exact-copy and visual-similarity candidate pairs across all imported sources, before Mango curation. |
| `conflicts.csv` | Exact pixel-copy pairs whose labels disagree. |
| `excluded_conflicts.csv` | `image_path`, `source_dataset`, `class_label`, `sha256_pixels`, `reason` for each excluded ambiguous copy. |
| `review_candidates.csv` | Visually similar pairs with different labels. Similarity alone does not prove a label is wrong. |
| `mango_curation.csv` | Imported Mango photos and their keep/exclude decisions when using `deduplicate`; header only for the `train-only` policy. |

Pair reports describe the inputs before Mango representatives are selected.
They can include paths omitted from the final manifest. Use `mango_curation.csv`
to trace an excluded photo to its retained representative.

The three pair tables use the same columns: `image_path_a`, `image_path_b`, `source_a`, `source_b`, `label_a`, `label_b`, `match_type`, and `phash_distance`. All are text in CSV; `phash_distance` represents an integer. `match_type` is `exact_pixels` or `perceptual_candidate`. A smaller perceptual hash distance means the small grayscale fingerprints are more alike; it is a similarity clue, not a disease score.

`report.json` uses JSON numbers for counts and settings, booleans for policy flags, objects for named counts and dataset metadata, and lists for missing classes and the split target. It records preparation only. Model scores and evaluation results are written later by the training and evaluation commands.

### `mango_curation.csv`: Mango decisions

This UTF-8 CSV also has 11 text columns, in this order:

| Column | Meaning |
|---|---|
| `image_path` | Absolute path to the imported Mango photo. |
| `relative_path` | Path relative to the Mango import root. |
| `class_label` | Its original combined condition label. |
| `sha256_pixels` | Exact RGB-pixel fingerprint used by preparation. |
| `capture_id` | `capture_` followed by 16 hexadecimal characters, derived from camera and capture time; empty when evidence is incomplete. |
| `capture_time` | EXIF capture time in `YYYY-MM-DDTHH:MM:SS` form, without a timezone; empty when unavailable or invalid. |
| `camera` | Camera make and model separated by ` \| `; may be empty or incomplete. |
| `family_id` | `mango_family_` followed by 16 hexadecimal characters; identifies a connected family before extra Mango rows are removed. |
| `action` | `keep` or `exclude`. |
| `representative_path` | Absolute path to the kept Mango representative for this family and class. A kept row points to itself. |
| `reason` | Plain-language explanation of this row's decision. |

The curation `family_id` and final manifest `group_id` describe different
stages. Family IDs include the imported members before curation; manifest
group IDs identify the retained members used for splitting. Several retained
Mango labels may share one family, and all belong to the same split.

`report.json` includes a `mango_curation` object with input, retained and
excluded counts; capture and connected-family counts; counts by class; the
selection rule; and the grouping limitation. These automatic groups do not
change `mango_groups_completely_reviewed` to true.
