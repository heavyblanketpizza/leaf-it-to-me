# Leaf It to Me

English | [한국어](README.ko.md)

![Apple, cherry, mango, orange, and peach: outdoor tree photographs with matching colorful fruit icons.](docs/assets/leaf-it-to-me-banner.webp)

A Python experiment that learns to identify **apple, cherry, mango, orange,
and peach trees from photographs of their leaves**, then classify the leaf's
condition. It uses MobileNetV4, a small image-classification model, with
PyTorch and timm.

You run it from a terminal: download labeled leaf photos, prepare the data,
train a model, and try a photo of your own. Run the package commands with
`python -m leafit`.

## Fruit trees and conditions

The model chooses from 18 combinations of fruit-tree type and leaf condition:

| Fruit tree | Supported leaf conditions |
|---|---|
| **Apple** | Healthy, scab, black rot, cedar apple rust, multiple diseases |
| **Cherry** | Healthy, powdery mildew |
| **Mango** | Healthy, anthracnose, bacterial canker, cutting weevil, dieback, gall midge, powdery mildew, sooty mould |
| **Orange** | Citrus greening only (no healthy-orange class) |
| **Peach** | Healthy, bacterial spot |

A prediction returns the tree type, condition, combined label, and model score.
For example (**illustrative output, not a measured result**):

```json
{
  "tree_type": "Apple",
  "condition": "scab",
  "class_label": "apple_scab",
  "model_score": 0.73
}
```

The score expresses the model's preference among its known labels. It is not a
73% probability of a correct diagnosis. This is a learning experiment; see the
[limitations](#limitations) before interpreting results.

## 1. Set up

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then open
a terminal in the project folder containing `pyproject.toml`. Run all commands
below from that folder. The examples use Bash or Zsh; on Windows, use WSL.
Initial setup and downloads need an internet connection. A GPU is optional.

```bash
uv sync --locked
cp -n .env.example .env
```

`uv` installs the project's Python 3.12.12 environment and pinned dependencies
in `.venv`. The copy command creates `.env` without replacing an existing file.
Edit these settings in your private `.env`:

| Setting | Value |
|---|---|
| `LEAFIT_DATA` | Folder containing `data/raw` and `data/prepared`. Use `"."` for the project folder, or enter your own storage location. |
| `LEAFIT_IMAGE` | Photo to predict after training, such as `"./data/my-leaf.jpg"`. |

Keep paths quoted. Load the settings in each new terminal and after editing them:

```bash
set -a
source .env
set +a
```

This makes the settings available to Python and to shell expressions such as
`"$LEAFIT_DATA"`. Keep external storage connected while using its datasets.

## 2. Check the setup

Run a small test before downloading the datasets:

```bash
uv run python -m leafit smoke --no-pretrained --device cpu --output runs/quick-check
```

It creates synthetic pictures and checks preparation, training, saving, loading,
and prediction. After dependency setup, it needs no downloads. Its scores measure
no real leaf-recognition ability. Use a new output folder when repeating it.

## 3. Download the datasets

The experiment uses three sources. These counts describe the selected source
images before preparation filters unreadable or related copies:

| Dataset | Fruit-tree leaves used | Selected source images | Source terms |
|---|---|---:|---|
| [PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage) | Original color apple, cherry, peach, and orange leaves; nine categories | 13,241 | CC BY-SA 3.0 |
| [Plant Pathology 2020 / FGVC7 (Cornell)](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7/data) | Apple leaves with public `train.csv` labels | 1,821 | Apache 2.0 plus the [competition rules](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7/rules), which take precedence |
| [MangoLeafBD v1](https://data.mendeley.com/datasets/hxsnvwty3r/1) | Mango leaves in eight categories, including supplied augmentations | 4,000 | CC BY-NC 3.0 |

Download the original releases below. The [source guide](docs/SOURCES.md)
includes credits, pinned revisions, and manual-download alternatives.

With `.env` loaded, create the storage folders and download PlantVillage:

```bash
mkdir -p "$LEAFIT_DATA/data/raw" "$LEAFIT_DATA/downloads"
uv run python -m leafit download-plantvillage --output "$LEAFIT_DATA/data/raw/plantvillage"
```

For Cornell, sign in to the competition page, accept its rules, and download
`plant-pathology-2020-fgvc7.zip`. Save it in the `downloads` folder under
`LEAFIT_DATA`, then import the labeled photos:

```bash
uv run python scripts/import_cornell_zip.py \
  --archive "$LEAFIT_DATA/downloads/plant-pathology-2020-fgvc7.zip" \
  --output "$LEAFIT_DATA/data/raw/cornell"
```

The importer excludes the unlabeled competition test photos and keeps the ZIP.
To remove the ZIP after verification, add `--delete-archive`.

Download MangoLeafBD:

```bash
uv run python scripts/download_mangoleafbd.py --output "$LEAFIT_DATA/data/raw/mangoleafbd"
```

This script removes its temporary ZIP after verifying the extracted files.
If the download is blocked, use the manual import in the [source guide](docs/SOURCES.md).

## 4. Prepare and train

Create the photo list and label mapping from all three datasets:

```bash
uv run python merge_datasets.py
```

The script checks the images, groups detected copies, and writes
`manifest.csv`, `labels.json`, and audit reports under
`"$LEAFIT_DATA/data/prepared"`. Original photos stay in place.

Preparation reports the retained counts and class distributions for your data.
It excludes exact copies with contradictory labels and keeps one Mango
representative per detected family and class. Whole groups stay together while
splitting toward these targets:

| Split | Target | Purpose |
|---|---:|---|
| Training | About 80% | Teach the model |
| Validation | About 10% | Select the best saved model |
| Test | About 10% | Assess that model after selection |

Group sizes and filtering affect the final counts. Check `report.json` for
class coverage and grouping limitations before training. See the
[dataset schema](docs/DATASET_SCHEMA.md) for the file format and the
[Mango grouping guide](docs/MANGO_AUDIT.md) for the curation method.

First, check training on a small number of real-data batches:

```bash
uv run python train.py --smoke
```

This performs two training updates and evaluates the full validation and test
sets. After it succeeds, run normal training:

```bash
uv run python train.py
```

Training starts from timm's ImageNet-1k pretrained checkpoint. It teaches the
final classification layer for two **epochs**—passes through the training
photos—then adjusts the whole model for three more.

The short check saves to `runs/training-smoke`; normal training saves to
`runs/weekend`. Every run needs a new or empty output folder. For another run
with lower memory use:

```bash
uv run python train.py --output runs/second-try --batch-size 8
```

Device selection tries CUDA, then Apple Silicon MPS, then CPU. Use
`--device cpu` to request CPU. The [training guide](docs/TRAINING.md) explains
all settings. `--data-root` overrides `LEAFIT_DATA`.

## 5. Predict and review results

Set `LEAFIT_IMAGE` in `.env` to your leaf photo and reload the settings. Use the
saved model to predict:

```bash
uv run python -m leafit predict --checkpoint runs/weekend/best.pt --image "$LEAFIT_IMAGE"
```

To turn the saved training results into a report and learning-curve chart:

```bash
uv run python scripts/summarize_training.py --run runs/weekend
```

Open `runs/weekend/results.md`. The report uses the saved results without
training again. Other useful files are:

| File | Contents |
|---|---|
| `best.pt` | Model selected using validation results, with its labels and preprocessing |
| `metrics.json` | Test scores, including accuracy and macro F1 |
| `per_class.csv` | Results for each fruit-tree and condition label |
| `confusion_matrix.png` | Which labels the model confuses |
| `history.json` | Training and validation results over time |

**Accuracy** is the fraction of correct predictions. **Macro F1** gives each
label equal weight while accounting for missed examples and wrong predictions.

To evaluate the saved model again:

```bash
uv run python -m leafit evaluate --checkpoint runs/weekend/best.pt \
  --manifest "$LEAFIT_DATA/data/prepared/manifest.csv" --output runs/evaluation
```

If you move the datasets, update `.env`, reload it, and rebuild the manifest
with `merge_datasets.py`: it stores absolute photo paths. Prediction only needs
the saved model and the chosen photo.

## First full training run

Completed September 12, 2026, using all 18 labels across the five fruit trees.
The split contained **14,929 training**, **1,867 validation**, and
**1,866 test** images.

| Stage | Epochs | Learning rate |
|---|---:|---:|
| Train the classification layer; keep the backbone frozen | 2 | 0.001 |
| Fine-tune the whole model | 3 | 0.0001 |

Configuration: ImageNet-pretrained `mobilenetv4_conv_small.e2400_r224_in1k`,
**224 × 224 RGB** input, **batch size 16**, AdamW with weight decay **0.0001**,
seed **42**, and the **MPS** backend. All training batches were used.

Validation selected **epoch 4**, during fine-tuning, with macro F1 **0.9201**.
That checkpoint achieved **97.96% test accuracy** (1,828 of 1,866 images)
and **0.9417 test macro F1** across all 18 labels.

These figures come from the saved `runs/weekend` reports. They describe the
prepared dataset split; the Mango grouping limitations below still apply.
Raw reports, datasets, and model weights remain excluded from Git.

## Limitations

- The model always chooses a known label, including for unfamiliar trees or
  non-leaf images. It has no "unknown" category.
- Orange has only citrus-greening examples, so this dataset cannot teach the
  model to recognize healthy orange leaves.
- Mango grouping uses visual similarity and camera metadata. Related photos
  may still cross splits, making test results optimistic.
- Scores on these datasets do not establish performance under different
  growing conditions. A model score is not a confirmed diagnosis.

## License and scope

This personal learning project shares original code and documentation under
[MIT](LICENSE). Training uses datasets and pretrained weights under their own
terms.
The README banner uses photos and icons licensed through a paid stock
subscription and is excluded from this repository's MIT license.
Demonstrating predictions is a separate use from distributing those materials;
follow the terms for any source images shown. See the
[source guide](docs/SOURCES.md#license-scope-and-attribution), including
[timm's ImageNet caveat](docs/SOURCES.md#pretrained-model-and-dependencies).

Datasets, model weights, private settings, and raw run outputs stay local via
[`.gitignore`](.gitignore). The [repository guide](docs/REPOSITORY.md) lists
what belongs in Git.

## Development

Start with `merge_datasets.py` and `train.py`. The `leafit/` package contains
preparation, training, evaluation, and prediction; `scripts/` contains import
helpers and the results summarizer. Run the automated checks with:

```bash
uv run python -m unittest discover -s tests -v
```

The [verification guide](docs/VALIDATION.md) explains the checks and how to run them.
