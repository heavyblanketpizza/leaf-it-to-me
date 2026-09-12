"""Turn an Orchard training run's saved numbers into a report and learning curves.

Run from the project folder: uv run python scripts/summarize_training.py
This reads saved results only; it never trains a model or predicts an image.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_results(run):
    """Require completed, mutually consistent artifacts before reporting results."""
    names = ("history.json", "metrics.json", "run_config.json", "data_summary.json",
             "per_class.csv", "confusion_matrix.csv")
    missing = [name for name in names if not (run / name).is_file()]
    if missing:
        raise ValueError("The run is incomplete or missing results: " + ", ".join(missing))
    history, metrics, config, data = (read_json(run / name) for name in names[:4])
    with (run / "per_class.csv").open(newline="", encoding="utf-8") as handle:
        classes = list(csv.DictReader(handle))
    with (run / "confusion_matrix.csv").open(newline="", encoding="utf-8") as handle:
        matrix_rows = list(csv.reader(handle))
    labels = [row["label"] for row in classes]
    if not history or not labels or metrics.get("split") != "test":
        raise ValueError("Expected nonempty training history, classes, and final test metrics.")
    epochs = [entry["epoch"] for entry in history]
    expected_epochs = config["head_epochs"] + config["fine_tune_epochs"]
    if epochs != list(range(1, expected_epochs + 1)):
        raise ValueError("History does not contain every configured training epoch.")
    if not matrix_rows or matrix_rows[0][1:] != labels or [row[0] for row in matrix_rows[1:]] != labels:
        raise ValueError("Confusion matrix class order disagrees with per_class.csv.")
    matrix = [[int(value) for value in row[1:]] for row in matrix_rows[1:]]
    if any(len(row) != len(labels) or any(value < 0 for value in row) for row in matrix):
        raise ValueError("Confusion matrix must be square with nonnegative counts.")
    if sum(map(sum, matrix)) != metrics["count"] or metrics["count"] != data["splits"]["test"]["images"]:
        raise ValueError("Test image counts disagree across saved results.")
    if [sum(row) for row in matrix] != [int(row["support"]) for row in classes]:
        raise ValueError("Per-class supports disagree with the confusion matrix.")
    best = max(history, key=lambda entry: entry["val"]["macro_f1"])
    if best["epoch"] != metrics["best_epoch"] or not math.isclose(
            best["val"]["macro_f1"], metrics["validation_macro_f1"], abs_tol=1e-12):
        raise ValueError("Selected checkpoint disagrees with validation history.")
    return history, metrics, config, data, classes, matrix


def plot_curves(history, best_epoch, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [entry["epoch"] for entry in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(epochs, [entry["train"]["loss"] for entry in history], "o-", label="Training loss")
    axes[0].plot(epochs, [entry["val"]["loss"] for entry in history], "o-", label="Validation loss")
    axes[0].set(title="Mistake penalty (lower is better)", ylabel="Cross-entropy loss")
    axes[1].plot(epochs, [entry["val"]["accuracy"] for entry in history], "o-", label="Validation accuracy")
    axes[1].plot(epochs, [entry["val"]["macro_f1"] for entry in history], "o-", label="Validation macro F1")
    axes[1].set(title="Validation scores (higher is better)", ylabel="Score", ylim=(0, 1.02))
    first_fine = next((entry["epoch"] for entry in history if entry["stage"] == "fine_tune"), None)
    for axis in axes:
        axis.axvline(best_epoch, color="#777777", linestyle=":", label="Chosen epoch")
        if first_fine and first_fine > epochs[0]:
            axis.axvline(first_fine - 0.5, color="#8c6239", linestyle="--", label="Fine-tuning starts")
        axis.set(xlabel="Epoch (one pass through training data)", xticks=epochs)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("Orchard learning curves")
    figure.tight_layout()
    figure.savefig(output / "learning_curves.png", dpi=160)
    plt.close(figure)


def make_report(run, history, metrics, config, data, classes, matrix):
    best = next(entry for entry in history if entry["epoch"] == metrics["best_epoch"])
    smoke = bool(config.get("max_batches", 0) or config.get("smoke", False))
    lines = ["# Orchard training results", "",
             f"Run: `{run}`", "",
             "**Smoke test only:** training was limited to a few batches. These results check the code; "
             "they do not describe a fully trained model." if smoke else
             f"Completed {len(history)} training epochs using the saved dataset split.", "",
             f"The chosen model scored **{metrics['accuracy']:.2%} test accuracy** and "
             f"**{metrics['macro_f1']:.4f} macro F1** on **{metrics['count']:,} test photographs**.", "",
             f"Overall supported-class F1: **{metrics['macro_f1_supported_classes']:.4f}** "
             f"across {metrics['supported_classes']}/{data['class_count']} tested classes.", "",
             "Accuracy is the fraction of answers that were correct. F1 balances finding examples of a class "
             "with avoiding false alarms. Macro F1 gives every mapped class equal weight; a class with no "
             "test examples scores zero in that average. These scores range from 0 to 1, with 1 best.", "",
             "## Data and settings", "",
             f"The run's saved data summary contains {data['total_images']:,} images across {data['class_count']} combined labels. "
             "These counts come from this run, even if the current dataset has since changed.", "",
             "| Split | Images | Groups | Classes represented |",
             "|---|---:|---:|---:|"]
    for name in ("train", "val", "test"):
        split = data["splits"][name]
        lines.append(f"| {name} | {split['images']:,} | {split['groups']:,} | {split['supported_classes']}/{data['class_count']} |")
    lines += ["", f"Model: `{config['model_name']}`. Device: `{config['resolved_device']}`. "
              f"Pretrained weights: {'yes' if metrics['pretrained'] else 'no'}. "
              f"Batch size: {config['batch_size']}; seed: {config['seed']}; data workers: {config['workers']}.", "",
              f"Head training: {config['head_epochs']} epochs at learning rate {config['lr_head']:g}. "
              f"Fine-tuning: {config['fine_tune_epochs']} epochs at learning rate {config['lr_fine']:g}.", "",
              "The backbone finds visual patterns; the head chooses a label from them. Head training freezes "
              "the backbone. Fine-tuning lets the whole model adjust with smaller steps.", "",
              f"Manifest SHA-256: `{data['manifest_sha256']}`.", "",
              "## Learning and checkpoint selection", "",
              f"Epoch **{metrics['best_epoch']}** ({metrics['best_stage']}) had the highest validation macro F1: "
              f"**{best['val']['macro_f1']:.4f}**. Validation chose the checkpoint before the final test evaluation.", ""]
    if metrics.get("frozen_buffers_verified") is True:
        lines += ["The trainer verified that the frozen backbone's running statistics stayed unchanged during head training.", ""]
    lines += ["| Epoch | Stage | Training images processed | Train loss | Validation loss | Validation accuracy | Validation macro F1 |",
              "|---:|---|---:|---:|---:|---:|---:|"]
    for entry in history:
        lines.append(f"| {entry['epoch']} | {entry['stage']} | {entry['train']['count']:,} | "
                     f"{entry['train']['loss']:.4f} | {entry['val']['loss']:.4f} | "
                     f"{entry['val']['accuracy']:.2%} | {entry['val']['macro_f1']:.4f} |")
    if any(entry["train"].get("dropped_singleton_images", 0) for entry in history):
        lines += ["", "Fine-tuning omits the final one-image batch when necessary because BatchNorm needs at least two "
                  "training examples in that batch. Validation and test keep every image."]
    lines += ["", "Loss measures the penalty for mistakes; lower is better. Training images have random augmentation, "
              "so training and validation losses are measured under different conditions.", "",
              "![Training loss and validation scores by epoch](learning_curves.png)", "",
              "## Test results by dataset", "",
              "Supported-class F1 averages only classes with test examples in that dataset. Different datasets cover "
              "different classes, so these averages are not directly interchangeable.", "",
              "| Dataset | Test images | Accuracy | Supported classes | Supported-class F1 |",
              "|---|---:|---:|---:|---:|"]
    sources = sorted({source for split in data["splits"].values() for source in split["by_source"]})
    for source in sources:
        values = metrics["by_source"].get(source)
        lines.append(f"| {source} | {values['count']:,} | {values['accuracy']:.2%} | {values['supported_classes']} | "
                     f"{values['macro_f1_supported_classes']:.4f} |" if values else
                     f"| {source} | 0 | Not measured | 0 | Not measured |")
    lines += ["", "## Test results by class", "",
              "Support is how many test photographs have that label. Precision asks how often predictions of "
              "that label were right; recall asks how many actual examples were found. Zero-support classes were not tested.", "",
              "| Label | Support | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|"]
    for row in classes:
        lines.append(f"| {row['label']} | {int(row['support']):,} | {float(row['precision']):.4f} | "
                     f"{float(row['recall']):.4f} | {float(row['f1']):.4f} |")
    errors = sorted(((count, classes[i]["label"], classes[j]["label"]) for i, row in enumerate(matrix)
                     for j, count in enumerate(row) if i != j and count), key=lambda item: (-item[0], item[1], item[2]))
    lines += ["", "## Most common test mistakes", ""]
    if errors:
        lines += ["Up to ten nonzero off-diagonal cells, sorted by image count. Frequent classes have more chances "
                  "to appear here.", "", "| Actual label | Predicted label | Images |", "|---|---|---:|"]
        lines += [f"| {actual} | {predicted} | {count:,} |" for count, actual, predicted in errors[:10]]
    else:
        lines += ["There were no off-diagonal errors in the saved test confusion matrix."]
    lines += ["", "## What these results can tell us", "",
              "This is a personal learning experiment on close-up leaf photographs. The model chooses among known "
              "tree/crop and condition labels. It has no unknown-tree or unknown-condition category.", "",
              "Orange has only citrus greening examples, so this dataset cannot teach healthy-versus-diseased orange diagnosis. "
              "Mango grouping is approximate: camera metadata and visual similarity cannot certify that all photographs "
              "of the same physical leaf were found. When Mango appears in validation or test, treat those scores as exploratory.", "",
              "A model score is a softmax score among known labels, not a calibrated probability or a confirmed diagnosis. "
              "Performance on these test photos does not establish performance on new orchard conditions.", "",
              "Source artifacts: `history.json`, `metrics.json`, `per_class.csv`, `confusion_matrix.csv`, "
              "`run_config.json`, and `data_summary.json` in the run folder above. This report performs no new training or inference.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/weekend"), help="Completed training run folder")
    parser.add_argument("--output", type=Path, help="Report destination; defaults to the run folder")
    args = parser.parse_args(argv)
    run = args.run.expanduser().resolve()
    output = (args.output or run).expanduser().resolve()
    try:
        history, metrics, config, data, classes, matrix = load_results(run)
        report = make_report(run, history, metrics, config, data, classes, matrix)
        output.mkdir(parents=True, exist_ok=True)
        plot_curves(history, metrics["best_epoch"], output)
        (output / "results.md").write_text(report, encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"Wrote {output / 'results.md'} and {output / 'learning_curves.png'}")


if __name__ == "__main__":
    main()
