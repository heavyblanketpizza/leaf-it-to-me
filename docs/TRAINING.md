# Train the leaf classifier

Use **PyTorch + timm**. PyTorch does the learning, and timm supplies our exact
pretrained MobileNetV4 model. A short Python training loop makes each step easy
to inspect while using your prepared CSV and label mapping.

We start from `mobilenetv4_conv_small.e2400_r224_in1k` and give it one output
for each combined label in `labels.json`. The complete three-source mapping
has 18 labels. With the project's pinned timm version, the loading API is:

```python
model = timm.create_model(
    "mobilenetv4_conv_small.e2400_r224_in1k",
    pretrained=True,
    num_classes=len(labels),
)
preprocessing = timm.data.resolve_model_data_config(model)
```

See the [official model card](https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k)
for the upstream checkpoint.

## Why three splits?

Preparation writes the train/validation/test assignments into the merged CSV's
`split` column. `train.py` uses those assignments. Even a pretrained model needs
separate evaluation photos so we can check whether it learns patterns that work
beyond the pictures used to adjust its weights.

| Split | Approximate target | Job |
|---|---:|---|
| Training | 80% | Adjust the model's weights |
| Validation | 10% | Choose the best saved checkpoint |
| Test | 10% | Assess that checkpoint after selection |

Keeping related photos together takes priority over exact percentages. Read
`report.json` beside your manifest for the actual retained counts and split
sizes; training also prints counts and saves them in `data_summary.json`.

Known leaf groups and detected copies stay together. The default merge command
keeps one Mango representative per detected family and class, then includes
Mango in the grouped split. These automatically detected families are an
approximation: some relatives may still be missed. See
[MANGO_AUDIT.md](MANGO_AUDIT.md) for the method and limitations.

## What the model learns

The **backbone** is the part that recognizes visual patterns such as edges,
shapes, and textures. Its existing weights are numbers learned from ImageNet
pictures. The **head** is the final layer that turns those patterns into one
score per answer, such as `apple_scab`.

We **freeze** the backbone by keeping its weights and remembered running
statistics unchanged while the new head learns. Then we **fine-tune** by letting
the whole model adjust with smaller steps. These are the two standard transfer
learning approaches described in the [PyTorch tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

An **epoch** is one trip through the training photos. A **batch** is a small
group processed together. The **learning rate** controls the size of each
update. Cross-entropy loss measures how badly the predicted scores disagree
with the correct label; the AdamW optimizer uses that loss to adjust weights.

## 1. Check setup and run a smoke test

Follow the [README setup](../README.md), prepare the datasets, and make sure
their storage is available. Run from the project folder. Load the private `.env`
you created during setup so commands can use its paths:

```bash
set -a
source .env
set +a
uv sync --locked
uv run python train.py --smoke
```

The script reads `data/prepared/manifest.csv` and the adjacent `labels.json`
under `LEAFIT_DATA`. If that variable is unset or empty, it uses the current
folder. The CSV points to the original photos, so their storage must stay
available. `--data-root` overrides the storage setting; `--manifest` and
`--labels` can select individual prepared files. For a one-off training command
without loading settings into your shell, use
`uv run --env-file .env python train.py --smoke`.

This smoke test uses the real merged dataset and pretrained weights. It makes
**two training updates**: one batch with the head learning and one with the
whole model learning. Validation checks the complete validation split after
each stage; the final test checks the complete test split. Expect it to take
longer than two training batches alone. Its scores do not measure a properly
trained classifier. For a check without downloaded photos or pretrained
weights, use the synthetic smoke test in [VALIDATION.md](VALIDATION.md).

Results go to `runs/training-smoke`. Each training run needs a new or empty
output folder. For a repeat, choose a new name:

```bash
uv run python train.py --smoke --output runs/training-smoke-2
```

The first pretrained run may download weights; later runs use the local cache.
`--device auto` chooses CUDA, then Apple Silicon MPS, then CPU, according to
availability. MPS uses a compatible Apple GPU. `--device cpu` forces CPU.

## 2. Train the full experiment

```bash
uv run python train.py
```

The defaults use all training rows, batch size **16**, two head epochs at
learning rate **0.001**, then three fine-tuning epochs at **0.0001**. Results go
to `runs/weekend`. These are starting settings for learning, not a promise of
accuracy or a particular runtime.

To change the settings for another run, use a fresh output folder:

```bash
uv run python train.py --head-epochs 2 --fine-tune-epochs 5 \
  --lr-head 0.001 --lr-fine 0.0001 --batch-size 16 \
  --workers 0 --device auto --seed 42 --output runs/weekend-2
```

`--workers 0` loads photos in the main process and is a simple starting point.
If memory runs out, try `--batch-size 8`. Fine-tuning requires at least two
images per batch. A final leftover batch of one image is skipped and reported
during fine-tuning because this model's BatchNorm layer needs a larger batch.
Head training, validation, and test keep their final image. The default
`--max-batches 0` means unlimited training batches; a positive value limits
training only. `uv run python train.py --help` lists all options.

The loader turns each photo into correctly oriented RGB pixels, then looks up
its numeric answer in `labels.json`. It gets resizing, cropping, and
normalization settings from the pretrained model. Normalization puts pixel
numbers on the scale the model expects. Training adds flips, rotations up to
10 degrees, and brightness changes up to 15%; validation and test use the
saved preprocessing without random changes. Original photos stay unchanged.

Preparation and training default to seed **42**. A seed fixes the starting point
for random choices. Keeping the same inputs, settings, and dependency versions
helps reproduce an experiment, but different devices or some accelerator
operations can still produce different results.

## 3. Read the results

After every epoch, validation measures how well learning transfers to held-out
photos. The script saves the checkpoint with the highest validation macro F1,
then reloads that checkpoint for the final test. Use validation results to
choose training settings and reserve test results for the final assessment.

Training saves checkpoints and evaluation reports in the chosen output folder.
Datasets, model weights, and generated run files are excluded from this
repository; a fresh checkout does not include a trained model.

| Saved file | What it tells you |
|---|---|
| `best.pt` | Best learned weights, with labels and preprocessing needed to reload them |
| `labels.json`, `preprocessing.json` | The output mapping and photo-loading settings |
| `run_config.json` | Settings, selected hardware, and library versions |
| `data_summary.json` | Counts and class coverage for each split, plus the manifest's SHA-256 fingerprint |
| `history.json` | Training and validation results for every epoch |
| `metrics.json` | Final test accuracy, macro F1, and results by source dataset |
| `per_class.csv` | Each label's image count, precision, recall, and F1 |
| `confusion_matrix.csv`, `confusion_matrix.png` | True labels in rows, predicted labels in columns |

To turn a completed run's saved numbers into a readable report and learning curves:

```bash
uv run python scripts/summarize_training.py --run runs/weekend
```

This adds `results.md` and `learning_curves.png` without training or evaluating
again. The report includes each condition's scores and the most common mistakes.

**Accuracy** is the fraction of photos classified correctly. **F1** combines
how often a label's predictions are right with how many actual examples of
that label the model finds. **Macro F1** gives each class an equal vote.

Check class coverage in `data_summary.json`: a label without held-out photos
cannot be assessed. `macro_f1` averages all mapped classes, counting any missing
classes as zero; `macro_f1_supported_classes` averages only classes with
held-out examples. Both are reported so a run's coverage stays visible.

Automatically detected Mango families do not prove that every held-out leaf is
new: capture metadata and visual similarity can miss relatives. Treat scores as
a weekend experiment with that limitation recorded. All orange photos show
citrus greening, so the model cannot learn healthy orange diagnosis. The
[SOURCES.md](SOURCES.md) guide records dataset sources and terms.

## 4. Evaluate or predict again

Training evaluates the best checkpoint automatically. After you have completed
a run, you can reproduce its test report using the same manifest:

```bash
uv run python -m leafit evaluate --checkpoint runs/weekend/best.pt \
  --manifest "${LEAFIT_DATA:-.}/data/prepared/manifest.csv" \
  --output runs/weekend-evaluation
```

To predict from your own close-up leaf photo, set `LEAFIT_IMAGE` in `.env`,
reload it using the commands in section 1, and run:

```bash
uv run python -m leafit predict --checkpoint runs/weekend/best.pt \
  --image "$LEAFIT_IMAGE"
```

The JSON result includes tree type, condition, combined label, and model score.
The score is the model's preference among known answers. Real garden photos
can differ from the datasets, and that score is not a calibrated probability
of a correct diagnosis. See [VALIDATION.md](VALIDATION.md) for repeatable checks.
