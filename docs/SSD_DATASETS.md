# Store datasets locally or on an external drive

The code and Python environment live in the project folder. Dataset storage can
be in the same folder or on another drive. Set `LEAFIT_DATA` in your private
`.env` to the folder containing `data/`, then load it as described in the
[README](../README.md). The default template uses the current project folder.

The expected layout under `LEAFIT_DATA` is:

| Folder | Contents |
|---|---|
| `downloads/` | Dataset archives waiting to be imported |
| `data/raw/plantvillage/` | Original color photos and official leaf-group metadata |
| `data/raw/cornell/` | Public `train.csv` and its labeled photos |
| `data/raw/mangoleafbd/` | Photos from all eight MangoLeafBD categories |
| `data/prepared/` | The generated photo list, labels, and preparation reports |

Keep external storage connected during preparation, training, and evaluation.
Prediction only needs the saved model and your photo. Download instructions and
source terms are in [SOURCES.md](SOURCES.md).

## Prepare and train

From the project folder, after setup:

```bash
set -a
source .env
set +a
uv run python merge_datasets.py
uv run python train.py
```

`merge_datasets.py` reads all three raw dataset folders and writes the prepared
files beneath the same storage root. `train.py` reads those files and saves its
results in the project folder's `runs/weekend`. Every training run needs a new
or empty output folder; choose another with `--output runs/second-try`.

If you move the dataset storage, update `.env`, reload it, and rebuild the
prepared files. The generated photo list contains absolute paths to the images.
See the [schema guide](DATASET_SCHEMA.md) for the file format and the
[training guide](TRAINING.md) for run settings.

## Inspect prepared data and training outputs

Preparation writes `report.json` with imported and retained counts, class
coverage, split sizes, and grouping limitations. Inspect it before training;
the actual counts depend on your source files and preparation settings.
The [Mango grouping guide](MANGO_AUDIT.md) explains the limits of automatic
family detection. The [verification guide](VALIDATION.md) describes checks
for the prepared data and training commands.

Training saves checkpoints and evaluation reports in the chosen output folder.
These generated files and the source datasets are excluded from this repository.
See [TRAINING.md](TRAINING.md) for commands and output descriptions.

## What stays out of Git

The repository ignores datasets, archives, generated indexes, model weights,
and training outputs. Manifests and reports may contain local folder paths.
Keep private settings in `.env`, raw photos in `data/`, and results in `runs/`
or `outputs/`. Review any artifact separately before sharing it.
