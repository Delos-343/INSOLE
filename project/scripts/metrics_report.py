"""
Extended metrics report — Dr. Marizuana's review request.

Produces:
  * Per-class precision, recall, F1
  * Macro-averaged precision, recall, F1
  * Overall accuracy
  * On BOTH the training split and the test split (so over-fit is visible)
  * For BOTH the neural-network classifier (full inputs) and the
    deterministic arch-height rule (the path the delivered system uses)

Run from the project root:
    docker compose exec backend python scripts/metrics_report.py \
        --checkpoint backend/model/checkpoints/best.pt

Optionally redirect to file for the handover record:
    docker compose exec backend python scripts/metrics_report.py \
        --checkpoint backend/model/checkpoints/best.pt \
        > metrics_report_$(date +%Y-%m-%d).txt 2>&1
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from loguru import logger

# Imports follow the same pattern as diagnose_model.py / predictor.py
from backend.model.config import ARCH_HEIGHT_BANDS, CLASS_NAMES
from backend.model.architectures.classifier import build_classifier
from backend.model.config import ModelConfig, TrainingConfig
from backend.model.data.dataloader import build_dataloaders
from backend.model.data.measurement_lookup import classify_by_arch_height_cm
from backend.model.utils.checkpoint import load_checkpoint


# ---------------------------------------------------------------------------
# Metric computation (no sklearn dependency — explicit, auditable)
# ---------------------------------------------------------------------------
def confusion_counts(y_true: list[int], y_pred: list[int], n_classes: int):
    """Per-class TP, FP, FN."""
    tp = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    return tp, fp, fn


def per_class_metrics(y_true: list[int], y_pred: list[int], n_classes: int):
    """Returns list of dicts per class with precision, recall, F1, support."""
    tp, fp, fn = confusion_counts(y_true, y_pred, n_classes)
    out = []
    for c in range(n_classes):
        support = tp[c] + fn[c]
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        recall = tp[c] / support if support > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        out.append({
            "class": CLASS_NAMES[c],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        })
    return out


def macro_average(rows: list[dict]):
    """Unweighted mean over classes — gives minority classes equal weight."""
    n = len(rows)
    return {
        "precision": sum(r["precision"] for r in rows) / n,
        "recall": sum(r["recall"] for r in rows) / n,
        "f1": sum(r["f1"] for r in rows) / n,
    }


def overall_accuracy(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_block(title: str, y_true: list[int], y_pred: list[int]):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

    n_classes = len(CLASS_NAMES)
    rows = per_class_metrics(y_true, y_pred, n_classes)
    macro = macro_average(rows)
    acc = overall_accuracy(y_true, y_pred)

    print(f"\n   Samples: {len(y_true)}")
    print(f"   Overall accuracy: {acc * 100:.2f}%\n")

    print(f"   {'Class':<20}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Support':>12}")
    print(f"   {'-'*20}{'-'*12}{'-'*12}{'-'*12}{'-'*12}")
    for r in rows:
        print(
            f"   {r['class']:<20}"
            f"{r['precision']*100:>11.2f}%"
            f"{r['recall']*100:>11.2f}%"
            f"{r['f1']*100:>11.2f}%"
            f"{r['support']:>12d}"
        )
    print(f"   {'-'*20}{'-'*12}{'-'*12}{'-'*12}{'-'*12}")
    print(
        f"   {'macro avg':<20}"
        f"{macro['precision']*100:>11.2f}%"
        f"{macro['recall']*100:>11.2f}%"
        f"{macro['f1']*100:>11.2f}%"
    )


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_model_predictions(model, loader, device):
    """Run the model in full-input mode (images + measurements) and collect
    (true, pred) labels."""
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        lat = batch["lateral"].to(device)
        top = batch["top"].to(device)
        back = batch["back"].to(device)
        meas = batch["measurements"].to(device)
        mask = batch["measurement_mask"].to(device)
        out = model(lateral=lat, top=top, back=back,
                    measurements=meas, measurement_mask=mask)
        preds = out["logits"].argmax(dim=1).cpu().numpy().tolist()
        y_pred.extend(preds)
        y_true.extend(batch["label"].cpu().numpy().tolist())
    return y_true, y_pred


def collect_rule_predictions(loader):
    """Apply the deterministic arch-height rule to the ground-truth arch height
    in each batch. This is what the delivered classification system does."""
    y_true, y_pred = [], []
    name_to_idx = {n: i for i, n in enumerate(CLASS_NAMES)}
    # measurements tensor is [calc_incl, heel_angle, arch_cm, kite, mt_talus]
    arch_idx_in_tensor = 2
    for batch in loader:
        meas = batch["measurements"].cpu().numpy()
        labels = batch["label"].cpu().numpy().tolist()
        for i, true_label in enumerate(labels):
            arch_cm = float(meas[i, arch_idx_in_tensor])
            predicted_class_name = classify_by_arch_height_cm(arch_cm)
            y_pred.append(name_to_idx[predicted_class_name])
            y_true.append(true_label)
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)

    print("=" * 78)
    print("EXTENDED METRICS REPORT")
    print("=" * 78)
    print("Per-class precision / recall / F1 and macro averages on the")
    print("training AND test splits, for both classification paths.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state, meta = load_checkpoint(args.checkpoint, map_location=device)
    model_cfg = ModelConfig(**meta["model_cfg"]) if meta.get("model_cfg") else ModelConfig()
    model = build_classifier(model_cfg).to(device)
    model.load_state_dict(state, strict=False)
    print(f"\nLoaded checkpoint: {args.checkpoint}")
    print(f"Device: {device}")

    loaders = build_dataloaders(TrainingConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=0,
    ))
    train_loader, val_loader, test_loader = loaders.train, loaders.val, loaders.test
    print(f"Train batches: {len(train_loader)}  "
          f"Val batches: {len(val_loader)}  "
          f"Test batches: {len(test_loader)}")

    # ---------- Path 1: deterministic rule (what the system uses) ----------
    y_true_tr, y_pred_tr = collect_rule_predictions(train_loader)
    print_block(
        "RULE PATH — DETERMINISTIC ARCH-HEIGHT BANDS — TRAIN SPLIT",
        y_true_tr, y_pred_tr,
    )
    y_true_te, y_pred_te = collect_rule_predictions(test_loader)
    print_block(
        "RULE PATH — DETERMINISTIC ARCH-HEIGHT BANDS — TEST SPLIT",
        y_true_te, y_pred_te,
    )

    # ---------- Path 2: neural-network classifier (for transparency) ----------
    y_true_tr, y_pred_tr = collect_model_predictions(model, train_loader, device)
    print_block(
        "MODEL PATH — NEURAL NETWORK (full inputs) — TRAIN SPLIT",
        y_true_tr, y_pred_tr,
    )
    y_true_te, y_pred_te = collect_model_predictions(model, test_loader, device)
    print_block(
        "MODEL PATH — NEURAL NETWORK (full inputs) — TEST SPLIT",
        y_true_te, y_pred_te,
    )

    print("\n" + "=" * 78)
    print("INTERPRETATION")
    print("=" * 78)
    print(
        "  * The RULE PATH is the delivered classification system. Its metrics"
        "\n    reflect the system's clinical accuracy on patients present in"
        "\n    the consolidated records."
        "\n"
        "\n  * The MODEL PATH is the neural-network classifier head. It is NOT"
        "\n    used to produce the headline classification in the delivered"
        "\n    system. Its metrics are included for transparency and for any"
        "\n    future image-driven product based on this codebase."
        "\n"
        "\n  * Compare per-class F1 across the two paths to see why the system"
        "\n    is designed measurement-first. The rule path's metrics are"
        "\n    near-100% by construction; the model path's metrics reflect"
        "\n    genuine learned-from-data performance with all its limits."
    )


if __name__ == "__main__":
    main()
