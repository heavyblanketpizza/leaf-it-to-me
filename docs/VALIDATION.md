# Verify your setup and experiment

This guide describes checks you can run after the [README setup](../README.md).
It does not report a bundled model's performance. Downloaded datasets,
checkpoints, and local experiment logs are excluded from the repository.

Run commands from the project folder. Install the locked dependencies first:

```bash
uv sync --locked
```

## 1. Run the automated tests

```bash
uv run python -m unittest discover -s tests -v
```

The suite uses temporary synthetic files and randomly initialized model weights.
It does not need downloaded plant photos or a pretrained checkpoint. A successful
run ends with `OK`; read any failure before training on your dataset.

The tests check:

- Correct EXIF orientation and RGB conversion while preserving original bytes.
- Reproducible group splits, rejection of groups crossing splits, and grouping
  exact copies and detected rotated or near-duplicate copies across sources.
- Handling of contradictory exact-copy labels and preservation of grouping
  links through excluded copies.
- Cornell import selection and exclusion of unlabeled competition test images.
- Mango representative selection, preservation of raw files, reproducible
  curation records, and honest separation of inferred and reviewed groups.
- A head update changing classifier weights while leaving backbone weights and
  running statistics unchanged.
- Metric arithmetic, visibility of missing classes, and keeping every evaluation
  image even when training must skip a final singleton batch.
- Portable configuration, missing-input errors, and protection against training
  over an existing nonempty output folder.

These checks verify program behavior. Passing them does not establish accuracy
on real leaves or prove that automatic duplicate detection finds every relative.

## 2. Run a synthetic smoke test without downloads

A **smoke test** is a small end-to-end run that checks whether the pieces connect.
After dependencies are installed, this command needs no dataset or weight
downloads:

```bash
uv run python -m orchard smoke --no-pretrained --device cpu \
  --output runs/smoke-offline
```

The command creates random-pixel images in the three supported dataset layouts,
prepares a grouped manifest covering all 18 labels, and runs one training batch
in each stage. It checks the complete synthetic validation split after each
stage, selects a checkpoint using validation, evaluates the complete test split,
and reloads that checkpoint for a prediction.

Look for the frozen-running-statistics verification message and the final
prediction JSON. Artifacts are written under the chosen output folder, including
`model/best.pt`, evaluation reports, and `SMOKE_ONLY.txt`. These random-pixel
scores say nothing about tree or disease recognition. Choose a new output
folder, such as `runs/smoke-offline-2`, when repeating the check.

## 3. Inspect prepared data and optionally smoke-test real photos

Follow the [README import and preparation commands](../README.md), then load
your private settings:

```bash
set -a
source .env
set +a
```

Inspect `report.json` beside your prepared `manifest.csv`. Check actual retained
counts and class distributions, unreadable-image reports, label conflicts,
Mango curation decisions, and grouping limitations before training. The split
target is about 80/10/10, with keeping detected families together taking priority.
See [DATASET_SCHEMA.md](DATASET_SCHEMA.md) for file fields and
[MANGO_AUDIT.md](MANGO_AUDIT.md) for Mango's approximate grouping method.

For a short check using your actual photos and the pretrained MobileNetV4
checkpoint:

```bash
uv run python train.py --smoke --output runs/training-smoke
```

The first pretrained run may download weights. This command makes one training
update per stage and still evaluates **all validation and test rows**. Its runtime
therefore depends on your dataset and device, not just two training batches.
Use a fresh output folder if that name already contains a run.

The trainer rejects invalid labels, missing image paths, repeated image paths,
groups assigned to multiple splits, empty splits, and mapped classes with no
training examples. It prints split sizes and class coverage and saves them in
`data_summary.json`. Missing validation or test classes are reported; inspect
these before interpreting any scores. A smoke-test score checks that the
pipeline runs and is not a finished model's performance.

## 4. Evaluate a completed training run

Use the full training commands in [TRAINING.md](TRAINING.md). Validation selects
the best checkpoint; the test split is assessed afterward. To evaluate that
saved checkpoint again, use the same prepared manifest and a separate output
folder:

```bash
uv run python -m orchard evaluate --checkpoint runs/weekend/best.pt \
  --manifest "${ORCHARD_DATA:-.}/data/prepared/manifest.csv" \
  --output runs/weekend-evaluation
```

Compare `metrics.json`, `per_class.csv`, and the confusion matrix with your
run's `data_summary.json`. The test count should match the manifest's test rows;
confusion-matrix counts should sum to that count. Check macro F1 and per-class
recall alongside overall accuracy so a common class cannot hide a weak one.
The saved manifest fingerprint helps identify which preparation a run used.
Keep the same seed, inputs, settings, and dependencies when comparing repeated
runs; accelerator differences can still affect results.

A held-out result measures this dataset split. Mango's automatic groups may miss
related leaves, and orange contains citrus greening examples without healthy
orange examples. Real orchard photos can also differ from the training photos.
Keep these limitations with any results you choose to share. Dataset provenance
and license information are in [SOURCES.md](SOURCES.md).
