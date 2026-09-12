# Mango curation and grouping

MangoLeafBD includes augmented photos. If a leaf's copies appear in both
training and test, the test becomes easier because the model has already seen
that leaf. This project groups likely relatives and keeps one representative
per detected family and class before creating training, validation, and test
splits.

The [official release](https://data.mendeley.com/datasets/hxsnvwty3r/1) describes
4,000 images derived from about 1,800 distinct leaves. The
[authors' paper, section 2.1.4](https://arxiv.org/html/2209.02377v1), describes
zooming and rotation after image selection. The release contains no
original-to-augmentation list or physical-leaf map. Automatic checks cannot
reliably recover those approximately 1,800 leaves.

## Prepare the dataset

From the project folder, after [setting up and loading `.env`](../README.md):

```bash
uv run python merge_datasets.py
```

The defaults are `--mango-policy deduplicate`, `--seed 42`, and
`--near-threshold 4`. Preparation reads the source photos and writes its manifest
and reports under `$ORCHARD_DATA/data/prepared/`. Source image files remain
unchanged; an excluded photo loses its manifest row, not its file.

The lower-level `uv run python -m orchard prepare` command defaults to
`--mango-policy train-only`. Pass `--mango-policy deduplicate` to that command to
use the same curation policy as the merge helper.

## How curation works

A **family** means photos linked by available evidence. It is an estimate of
related images, not a verified physical-leaf identity.

1. Read each Mango photo's camera make, camera model, and EXIF
   `DateTimeOriginal` at one-second precision. EXIF is information saved inside
   a photo by a camera or editor. Matching valid combinations link likely
   copies from one capture. Missing or invalid capture metadata creates no
   capture-time link.
2. Combine those links with exact RGB-pixel matches, rotated/flipped perceptual
   hash candidates, official leaf groups, and supplied reviewed groups across
   the full imported dataset. A perceptual hash is a small visual fingerprint;
   similar fingerprints are clues that photos may be related. Connections are
   transitive: if A links to B and B links to C, all three belong together.
3. Keep one Mango representative per detected family **and class**. Prefer a
   file without a Photo Editor, Photoshop, or GIMP software marker, then a larger
   pixel area, then fewer filename copy markers, and finally the path for a
   repeatable tie-break. These clues do not prove that a file is an original.
   If a family spans several condition labels, keep one representative of each
   class and preserve their common group. Connections through an excluded photo
   are also preserved.
4. Split whole families reproducibly, aiming for 80% training, 10% validation,
   and 10% test. Groups cannot be divided just to reach an exact percentage.
   Class coverage depends on having enough independent groups.
5. Record each imported Mango photo's keep/exclude decision and selected
   representative in `mango_curation.csv`.

Training adds fresh flips, modest rotations, and mild brightness changes when
loading training photos. Validation and test receive deterministic preprocessing
without random augmentation.

## Review the output

Preparation prints retained counts and class distributions. Its output includes:

| File | What to check |
|---|---|
| `report.json` | Mango input and retained counts, group counts, settings, missing evaluation classes, and grouping limitations |
| `mango_curation.csv` | Capture evidence, detected family, keep/exclude action, representative path, and reason for each Mango photo |
| `duplicate_pairs.csv` | Exact and visual-similarity links, including links through excluded photos |
| `review_candidates.csv` | Visual matches whose labels differ and may need inspection |
| `manifest.csv` | Retained photos with final group identifiers and split assignments |

See [the dataset schema](DATASET_SCHEMA.md) for field definitions and
[the verification guide](VALIDATION.md) for checking a preparation run. Generated
per-image reports contain dataset references and local paths; keep them local.

The report's `mango_groups_completely_reviewed` is **false** unless every usable
Mango photo has a reviewed group supplied through `--groups`. Automatic capture
and visual matches never count as manual review. Family counts and representative
counts can differ because one family can contain several class labels.

## Limitations and reviewed groups

Camera timestamps and visual matches complement each other, but both can fail:

- Distinct photos taken with the same camera in one second can be grouped
  together. False visual matches can also join unrelated leaves.
- Crops, arbitrary rotations, separate views, or missing metadata can hide a
  relationship. Undetected relatives may end up in different splits and make
  evaluation scores too optimistic.
- Editor metadata and filename suffixes do not reliably identify augmentations.
  An edited photo can be the only available example of that leaf. Dropping all
  edited files would lose such examples.

Treat this as an exploratory representative subset. Do not describe it as
verified originals or a guaranteed leaf-independent dataset. Better physical-leaf
ground truth requires reviewed identities or an author-provided leaf map.

You can supply reviewed groups in a CSV with
`source_dataset,image_path,group_id` columns. Use paths relative to each source's
import root; see [the schema guide](DATASET_SCHEMA.md) for details. For example,
with your reviewed file saved under the private data folder:

```bash
uv run python merge_datasets.py --groups "$ORCHARD_DATA/data/reviewed-groups.csv"
```

To retain the conservative fallback:

```bash
uv run python merge_datasets.py --mango-policy train-only
```

Without complete reviewed Mango grouping, this fallback keeps Mango and any
connected relatives in training, so it cannot measure Mango performance on
validation or test. With complete reviewed coverage, the pipeline permits
Mango groups in all three splits while keeping every usable Mango photo.
