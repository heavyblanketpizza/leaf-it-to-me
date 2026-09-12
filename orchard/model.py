"""A small two-stage image classifier; run through ``python -m orchard``.

The backbone learns visual patterns. The head turns those patterns into one
score for each combined tree/condition label. Imports here require the ML extras.
"""

import csv
import hashlib
import json
import os
import platform
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


MODEL_NAME = "mobilenetv4_conv_small.e2400_r224_in1k"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def validate_labels(labels):
    if not isinstance(labels, list) or not labels:
        raise ValueError("labels.json must contain a nonempty list of label objects.")
    if [entry["index"] for entry in labels] != list(range(len(labels))):
        raise ValueError("Label indices must be consecutive, ordered from zero.")
    if len({entry["label"] for entry in labels}) != len(labels):
        raise ValueError("Each combined label must appear exactly once.")
    for entry in labels:
        for key in ("label", "tree_type", "condition"):
            if not isinstance(entry[key], str) or not entry[key]:
                raise ValueError(f"Each label needs a nonempty {key}.")
    return labels


def read_manifest(path, labels):
    """Refuse a split that accidentally lets the same leaf appear on both sides."""
    path = Path(path).resolve()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "source_dataset", "tree_type", "condition",
                    "class_label", "group_id", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest needs these columns: {sorted(required)}")
        rows = list(reader)
    label_by_name = {entry["label"]: entry for entry in labels}
    groups, paths = {}, set()
    for row in rows:
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Unexpected split: {row['split']!r}")
        label = label_by_name.get(row["class_label"])
        if label is None:
            raise ValueError(f"Unknown class: {row['class_label']}")
        if (row["tree_type"], row["condition"]) != (label["tree_type"], label["condition"]):
            raise ValueError(f"Tree/condition disagrees with label: {row['class_label']}")
        group = row["group_id"]
        if not group or not row["source_dataset"]:
            raise ValueError("Every image needs a group_id and source_dataset.")
        if groups.setdefault(group, row["split"]) != row["split"]:
            raise ValueError(f"Data leakage: group {group!r} crosses splits.")
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if str(image_path) in paths:
            raise ValueError(f"Duplicate image path in manifest: {image_path}")
        paths.add(str(image_path))
        row["image_path"] = str(image_path)
        row["target"] = label["index"]
    if not rows:
        raise ValueError("Manifest is empty. Prepare the datasets first.")
    return rows


def load_rgb(path):
    # Some cameras store rotation in metadata instead of rotating the pixels.
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def make_transform(preprocessing, training=False):
    # timm supplies the pretrained model's exact size, crop, mean, and std.
    evaluation = timm.data.create_transform(**preprocessing, is_training=False)
    if not training:
        return evaluation
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15),
        evaluation,
    ])


class LeafDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return self.transform(load_rgb(row["image_path"])), row["target"], row["source_dataset"]


def choose_device(requested="auto"):
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if mps_available else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Use --device auto or --device cpu.")
    if requested == "mps" and not mps_available:
        raise ValueError("MPS is not available. Use --device auto or --device cpu.")
    print(f"Device: {requested}", flush=True)
    return torch.device(requested)


def seed_everything(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Some accelerators cannot guarantee identical results for every operation.
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(rows, preprocessing, args, training, device, fine_tuning=False):
    if fine_tuning and (args.batch_size < 2 or len(rows) < 2):
        raise ValueError("Fine-tuning needs --batch-size at least 2 and at least two training images for BatchNorm.")
    # The pooled BatchNorm layer cannot learn from a one-image batch. Only omit
    # the final singleton during fine-tuning; head training and evaluation keep it.
    drop_singleton = fine_tuning and len(rows) % args.batch_size == 1
    generator = torch.Generator().manual_seed(getattr(args, "seed", 42))
    return DataLoader(
        LeafDataset(rows, make_transform(preprocessing, training)),
        batch_size=args.batch_size, shuffle=training, num_workers=args.workers,
        pin_memory=device.type == "cuda", worker_init_fn=seed_worker, generator=generator,
        drop_last=drop_singleton,
    )


def metrics_from_confusion(confusion):
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    correct = np.diag(confusion)
    precision = np.divide(correct, predicted, out=np.zeros_like(correct, dtype=float), where=predicted > 0)
    recall = np.divide(correct, support, out=np.zeros_like(correct, dtype=float), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=precision + recall > 0)
    count = int(support.sum())
    summary = {
        "count": count,
        "accuracy": float(correct.sum() / count) if count else 0.0,
        "macro_f1": float(f1.mean()),
        "macro_f1_supported_classes": float(f1[support > 0].mean()) if count else 0.0,
        "supported_classes": int((support > 0).sum()),
        "macro_f1_definition": "Unweighted mean over all mapped classes; unsupported classes score zero.",
    }
    per_class = [{"index": i, "support": int(support[i]), "precision": float(precision[i]),
                  "recall": float(recall[i]), "f1": float(f1[i])} for i in range(len(support))]
    return summary, per_class


def run_epoch(model, loader, device, class_count, optimizer=None, max_batches=0, description=""):
    """One pass through the data. Only training may stop early for a smoke test."""
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    source_confusions = {}
    loss_sum, count = 0.0, 0
    criterion = nn.CrossEntropyLoss()
    context = torch.enable_grad() if optimizer is not None else torch.inference_mode()
    batches = min(len(loader), max_batches) if optimizer is not None and max_batches else len(loader)
    with context:
        for batch_index, (images, targets, sources) in enumerate(loader):
            if optimizer is not None and max_batches and batch_index >= max_batches:
                break
            images, targets = images.to(device), targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Training produced a non-finite loss; inspect the images and learning rate.")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            truth = targets.detach().cpu().numpy()
            predictions = logits.detach().argmax(dim=1).cpu().numpy()
            np.add.at(confusion, (truth, predictions), 1)
            for target, prediction, source in zip(truth, predictions, sources):
                source_confusions.setdefault(source, np.zeros_like(confusion))[target, prediction] += 1
            loss_sum += loss.item() * len(targets)
            count += len(targets)
            if description and (batch_index == 0 or (batch_index + 1) % 50 == 0 or batch_index + 1 == batches):
                print(f"{description}: batch {batch_index + 1}/{batches}, loss={loss_sum / count:.4f}", flush=True)
    if not count:
        raise ValueError("Cannot evaluate or train an empty split.")
    summary, per_class = metrics_from_confusion(confusion)
    summary["loss"] = loss_sum / count
    if optimizer is not None:
        summary["dropped_singleton_images"] = 1 if loader.drop_last else 0
        summary["images_not_processed"] = len(loader.dataset) - count
    summary["by_source"] = {source: metrics_from_confusion(matrix)[0]
                            for source, matrix in sorted(source_confusions.items())}
    return summary, per_class, confusion


def freeze_backbone(model):
    """Freeze weights AND BatchNorm's remembered averages during head training."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    classifier = model.get_classifier()
    for parameter in classifier.parameters():
        parameter.requires_grad_(True)
    model.eval()
    classifier.train()
    return classifier


def save_reports(output, summary, per_class, confusion, labels):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary["missing_classes"] = [label["label"] for label, values in zip(labels, per_class)
                                  if not values["support"]]
    write_json(output / "metrics.json", summary)
    with (output / "per_class.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "label", "tree_type", "condition",
                                                   "support", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows({**label, **values} for label, values in zip(labels, per_class))
    names = [entry["label"] for entry in labels]
    with (output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true / predicted", *names])
        writer.writerows([name, *row.tolist()] for name, row in zip(names, confusion))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(12, 10))
    plot = axis.imshow(confusion, cmap="Blues")
    axis.set_xticks(range(len(names)), names, rotation=90, fontsize=8)
    axis.set_yticks(range(len(names)), names, fontsize=8)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(f"Test confusion matrix · accuracy {summary['accuracy']:.1%} · macro F1 {summary['macro_f1']:.3f}")
    figure.colorbar(plot, ax=axis, label="Images")
    figure.tight_layout()
    figure.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(figure)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != MODEL_NAME:
        raise ValueError("Checkpoint uses a different model architecture.")
    labels = validate_labels(checkpoint["labels"])
    # All learned weights are already inside the checkpoint: no network needed.
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(labels))
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint


def training_data_summary(rows, labels):
    """Count what each split can teach or measure before training begins."""
    names = [entry["label"] for entry in labels]
    result = {"total_images": len(rows), "class_count": len(labels), "splits": {}}
    for split in ("train", "val", "test"):
        selected = [row for row in rows if row["split"] == split]
        counts = Counter(row["class_label"] for row in selected)
        result["splits"][split] = {
            "images": len(selected), "groups": len({row["group_id"] for row in selected}),
            "by_class": {label: counts[label] for label in names},
            "by_source": dict(sorted(Counter(row["source_dataset"] for row in selected).items())),
            "supported_classes": len(counts), "missing_classes": [label for label in names if not counts[label]],
        }
    return result


def validate_training_output(output):
    """Keep a new experiment from replacing a previous checkpoint or report."""
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Training output must be a new or empty folder: {output}. "
                         "Choose a fresh --output, for example runs/weekend-2.")


def train(args):
    if args.head_epochs < 0 or args.fine_tune_epochs < 0 or args.head_epochs + args.fine_tune_epochs < 1:
        raise ValueError("Request at least one epoch; epoch counts cannot be negative.")
    if args.max_batches < 0:
        raise ValueError("--max-batches must be zero (unlimited) or positive.")
    validate_runtime_args(args)
    output = Path(args.output)
    validate_training_output(output)
    seed_everything(args.seed, args.threads)
    device = choose_device(args.device)
    labels = validate_labels(json.loads(Path(args.labels).read_text(encoding="utf-8")))
    rows = read_manifest(args.manifest, labels)
    splits = {split: [row for row in rows if row["split"] == split] for split in ("train", "val", "test")}
    if any(not values for values in splits.values()):
        raise ValueError("Training needs nonempty train, val, and test splits. Check the preparation report.json.")
    if args.fine_tune_epochs and len(splits["train"]) < 2:
        raise ValueError("Fine-tuning needs at least two training images for BatchNorm.")
    absent_training = sorted({entry["label"] for entry in labels} - {row["class_label"] for row in splits["train"]})
    if absent_training:
        raise ValueError(f"These labels have no training images: {absent_training}. Rebuild the mapping/splits.")
    print("Split sizes:", {split: len(values) for split, values in splits.items()}, flush=True)
    print("Class counts:", dict(sorted(Counter(row["class_label"] for row in rows).items())), flush=True)
    data_summary = training_data_summary(rows, labels)
    data_summary["manifest_sha256"] = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    for split, details in data_summary["splits"].items():
        print(f"{split}: {details['images']:,} photos, {details['supported_classes']}/{len(labels)} classes.", flush=True)
        if details["missing_classes"]:
            print("  No examples for: " + ", ".join(details["missing_classes"]), flush=True)
    if args.max_batches:
        print(f"Smoke/practice run: at most {args.max_batches} training batches per epoch; "
              "validation and test still use their complete splits. Scores are not a finished model's results.", flush=True)
    model = timm.create_model(MODEL_NAME, pretrained=not args.no_pretrained, num_classes=len(labels))
    preprocessing = timm.data.resolve_model_data_config(model)
    model.to(device)
    loaders = {split: make_loader(values, preprocessing, args, split == "train", device)
               for split, values in splits.items()}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "data_summary.json", data_summary)
    write_json(output / "labels.json", labels)
    write_json(output / "preprocessing.json", preprocessing)
    run_config = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items() if not callable(value)}
    run_config.update(resolved_device=str(device), model_name=MODEL_NAME,
                      python_version=platform.python_version(), torch_version=str(torch.__version__),
                      timm_version=timm.__version__)
    write_json(output / "run_config.json", run_config)
    history, best_score, total_epoch = [], -1.0, 0
    frozen_buffers_verified = None
    for stage, epochs, learning_rate in (("head", args.head_epochs, args.lr_head),
                                          ("fine_tune", args.fine_tune_epochs, args.lr_fine)):
        if not epochs:
            continue
        if stage == "head":
            classifier = freeze_backbone(model)
            # This model's linear classifier has no buffers; every buffer belongs to its backbone.
            frozen_buffers = {name: value.detach().cpu().clone() for name, value in model.named_buffers()}
        else:
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            loaders["train"] = make_loader(splits["train"], preprocessing, args, True, device, fine_tuning=True)
            if loaders["train"].drop_last:
                print("Fine-tuning omits 1 shuffled training image per epoch to avoid a one-image BatchNorm batch.", flush=True)
        optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                      lr=learning_rate, weight_decay=1e-4)
        for stage_epoch in range(1, epochs + 1):
            total_epoch += 1
            if stage == "head":
                model.eval()
                classifier.train()
            else:
                model.train()
            training, _, _ = run_epoch(model, loaders["train"], device, len(labels), optimizer, args.max_batches,
                                       description=f"Epoch {total_epoch} ({stage}) train")
            model.eval()
            validation, _, _ = run_epoch(model, loaders["val"], device, len(labels), description="Validation")
            history.append({"epoch": total_epoch, "stage": stage, "stage_epoch": stage_epoch,
                            "learning_rate": learning_rate, "train": training, "val": validation})
            print(f"Epoch {total_epoch} ({stage}) train loss={training['loss']:.4f}; "
                  f"val accuracy={validation['accuracy']:.3f}, macro F1={validation['macro_f1']:.3f}, "
                  f"supported-class F1={validation['macro_f1_supported_classes']:.3f}", flush=True)
            if validation["macro_f1"] > best_score:
                best_score = validation["macro_f1"]
                torch.save({"model_name": MODEL_NAME,
                            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                            "labels": labels, "preprocessing": preprocessing,
                            "epoch": total_epoch, "stage": stage, "validation": validation,
                            "pretrained": not args.no_pretrained, "seed": args.seed,
                            "training_max_batches": args.max_batches}, output / "best.pt")
            write_json(output / "history.json", history)
        if stage == "head":
            changed = [name for name, value in model.named_buffers()
                       if not torch.equal(value.detach().cpu(), frozen_buffers[name])]
            if changed:
                raise RuntimeError(f"Frozen backbone running statistics changed: {changed}")
            frozen_buffers_verified = True
            print("Verified: frozen backbone running statistics stayed unchanged.", flush=True)
    # Looking at test scores only after validation has chosen the winner avoids tuning to the exam.
    model, checkpoint = load_checkpoint(output / "best.pt", device)
    summary, per_class, confusion = run_epoch(model, loaders["test"], device, len(labels), description="Final test")
    summary.update({"split": "test", "best_epoch": checkpoint["epoch"], "best_stage": checkpoint["stage"],
                    "validation_macro_f1": best_score, "frozen_buffers_verified": frozen_buffers_verified,
                    "pretrained": not args.no_pretrained, "training_max_batches": args.max_batches})
    save_reports(output, summary, per_class, confusion, labels)
    print(f"Test accuracy={summary['accuracy']:.3f}, macro F1={summary['macro_f1']:.3f}, "
          f"supported-class F1={summary['macro_f1_supported_classes']:.3f}; saved to {output.resolve()}", flush=True)


def evaluate(args):
    validate_runtime_args(args)
    seed_everything(42, args.threads)
    device = choose_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    labels = checkpoint["labels"]
    rows = read_manifest(args.manifest, labels)
    rows = [row for row in rows if row["split"] == "test"]
    loader = make_loader(rows, checkpoint["preprocessing"], args, False, device)
    summary, per_class, confusion = run_epoch(model, loader, device, len(labels))
    summary.update({"split": "test", "checkpoint": str(Path(args.checkpoint).resolve()),
                    "pretrained": checkpoint["pretrained"], "training_max_batches": checkpoint["training_max_batches"]})
    save_reports(args.output, summary, per_class, confusion, labels)
    print(json.dumps(summary, indent=2))


def predict(args):
    if args.threads < 1:
        raise ValueError("--threads must be positive.")
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    image = make_transform(checkpoint["preprocessing"])(load_rgb(args.image)).unsqueeze(0).to(device)
    with torch.inference_mode():
        scores = model(image).softmax(dim=1)[0]
    index = int(scores.argmax().item())
    label = checkpoint["labels"][index]
    result = {"tree_type": label["tree_type"].replace("_", " ").title(),
              "condition": label["condition"].replace("_", " "), "class_label": label["label"],
              "model_score": float(scores[index].item()),
              "score_note": "Softmax score among known labels; not a calibrated probability or a diagnosis."}
    print(json.dumps(result, indent=2))


def validate_runtime_args(args):
    if args.batch_size < 1 or args.workers < 0 or args.threads < 1:
        raise ValueError("Batch size and threads must be positive; workers cannot be negative.")
    if hasattr(args, "lr_head") and (args.lr_head <= 0 or args.lr_fine <= 0):
        raise ValueError("Learning rates must be positive.")
    if getattr(args, "fine_tune_epochs", 0) > 0 and args.batch_size < 2:
        raise ValueError("Fine-tuning needs --batch-size at least 2 for BatchNorm; head-only training can use 1.")


def add_commands(subparsers):
    parser = subparsers.add_parser("train", help="Train the head, fine-tune, then test the best checkpoint")
    training_arguments(parser)
    parser.set_defaults(func=train)
    parser = subparsers.add_parser("evaluate", help="Evaluate a saved checkpoint on the full test split")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="data/prepared/manifest.csv")
    parser.add_argument("--output", default="runs/evaluation")
    runtime_arguments(parser)
    parser.set_defaults(func=evaluate)
    parser = subparsers.add_parser("predict", help="Predict a combined label from one local RGB photograph")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.set_defaults(func=predict)


def training_arguments(parser):
    """Share the same training options between train.py and python -m orchard."""
    parser.add_argument("--manifest", default="data/prepared/manifest.csv")
    parser.add_argument("--labels", default="data/prepared/labels.json")
    parser.add_argument("--output", default="runs/weekend")
    parser.add_argument("--head-epochs", type=int, default=2)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-fine", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batches", type=int, default=0,
                        help="Training batches per epoch for smoke tests; 0=all. Validation/test always complete.")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Use random weights explicitly for an offline plumbing test; not real fine-tuning")
    runtime_arguments(parser)


def runtime_arguments(parser):
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
