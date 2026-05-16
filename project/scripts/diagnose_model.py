"""
Model diagnostic — determines WHY the classifier fails on honest tests
despite a high reported validation accuracy.

It runs four independent probes and prints a verdict. No training; uses the
existing best.pt and the existing dataset. Takes a few minutes on CPU.

    docker compose exec backend python scripts/diagnose_model.py

PROBES
------
1. CONFUSION MATRIX on the held-out test split
   - Is the model actually uniform-accurate, or is it just predicting a
     couple of classes and riding class priors?

2. IMAGES-ONLY vs MEASUREMENTS-ONLY vs BOTH
   - Re-evaluate the test split three ways:
       (a) full inputs (images + measurements) — as trained
       (b) measurements zeroed, mask=0 — images must carry the signal
       (c) images blacked out — measurements must carry the signal
   - If (c) >> (b), the model is almost entirely measurement-driven and the
     images contribute little. If (b) collapses to chance, arch class is NOT
     recoverable from the photos — a data/task problem, not a code bug.

3. COHORT-LEAKAGE PROBE
   - The cohorts occupy contiguous ID blocks (Normal P001-500,
     Flat P501-1000, High P1001-1500). Patient ID should be meaningless to
     a model that learned anatomy. We test whether predicted class is
     suspiciously predictable from ID block alone, and whether the model's
     errors cluster by cohort (a batch-confound signature).

4. MEASUREMENT-RULE BASELINE (no neural net at all)
   - Apply the brief's cm bands directly to the ground-truth measurements.
     This is the accuracy ceiling a trivial non-ML system achieves. If this
     is ~100% and the model is ~30% on honest tests, the conclusion writes
     itself: require measurements at inference.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import torch

from backend.model.config import CLASS_NAMES, ARCH_HEIGHT_BANDS, InferenceConfig
from backend.model.data.dataloader import build_dataloaders
from backend.model.config import TrainingConfig
from backend.model.inference.predictor import Predictor


def _idx_to_name(i: int) -> str:
    return CLASS_NAMES[i]


def _print_confusion(cm: np.ndarray) -> None:
    n = len(CLASS_NAMES)
    short = [c[:11] for c in CLASS_NAMES]
    print("\n   rows = TRUE, cols = PRED")
    print("   " + "".join(f"{s:>13}" for s in short))
    for i in range(n):
        row = cm[i]
        tot = row.sum()
        cells = "".join(f"{int(v):>13}" for v in row)
        acc = (row[i] / tot * 100) if tot else 0.0
        print(f"{short[i]:>14}{cells}   | recall {acc:5.1f}% (n={int(tot)})")
    diag = np.trace(cm)
    print(f"\n   Overall accuracy on this split: {diag / cm.sum() * 100:.2f}%")


@torch.no_grad()
def _eval(predictor: Predictor, loader, mode: str) -> tuple[np.ndarray, list]:
    """mode in {'full','no_meas','no_img'}. Returns (confusion, records)."""
    model = predictor.model
    device = predictor.device
    n = len(CLASS_NAMES)
    cm = np.zeros((n, n), dtype=np.int64)
    records = []

    for batch in loader:
        lat = batch["lateral"].to(device)
        top = batch["top"].to(device)
        bak = batch["back"].to(device)
        meas = batch["measurements"].to(device)
        mmask = batch["measurement_mask"].to(device)
        labels = batch["label"]

        if mode == "no_meas":
            meas = torch.zeros_like(meas)
            mmask = torch.zeros_like(mmask)
        elif mode == "no_img":
            lat = torch.zeros_like(lat)
            top = torch.zeros_like(top)
            bak = torch.zeros_like(bak)

        out = model(lateral=lat, top=top, back=bak,
                    measurements=meas, measurement_mask=mmask)
        preds = out["logits"].softmax(-1).argmax(-1).cpu().numpy()
        for t, p, pid in zip(labels.numpy(), preds, batch["patient_id"]):
            cm[t, p] += 1
            records.append((pid, int(t), int(p)))
    return cm, records


def _rule_baseline(loader_dataset) -> float:
    """Brief's cm bands applied to GROUND-TRUTH arch height. Index 2 of the
    measurement vector is arch_height in cm (dataset stores it in cm)."""
    correct = 0
    total = 0
    for s in loader_dataset.samples:
        arch_cm = float(s.measurements[2])
        if arch_cm <= 0:
            continue
        pred = None
        for name, (lo, hi) in ARCH_HEIGHT_BANDS.items():
            if lo <= arch_cm < hi:
                pred = name
                break
        if pred is None:
            continue
        total += 1
        correct += int(pred == s.label)
    return (correct / total * 100) if total else 0.0


def main() -> None:
    print("=" * 70)
    print("MODEL DIAGNOSTIC")
    print("=" * 70)

    tcfg = TrainingConfig(num_epochs=1)  # only used to rebuild the same splits
    loaders = build_dataloaders(tcfg)
    predictor = Predictor(InferenceConfig())
    print(f"Loaded checkpoint: {predictor.checkpoint_path}")
    print(f"Test split size: {len(loaders.test_dataset)} patients")

    # ---- PROBE 1 + 2: three evaluation modes on the test split ----
    for mode, title in [
        ("full", "PROBE 2a — FULL INPUTS (images + measurements, as trained)"),
        ("no_meas", "PROBE 2b — IMAGES ONLY (measurements zeroed)"),
        ("no_img", "PROBE 2c — MEASUREMENTS ONLY (images blacked out)"),
    ]:
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
        cm, records = _eval(predictor, loaders.test, mode)
        _print_confusion(cm)
        if mode == "full":
            full_records = records

    # ---- PROBE 3: cohort-leakage signature ----
    print("\n" + "=" * 70)
    print("PROBE 3 — COHORT / BATCH LEAKAGE SIGNATURE")
    print("=" * 70)

    def cohort_of(pid: str) -> str:
        m = pid.upper().lstrip("P")
        try:
            num = int(m)
        except ValueError:
            return "unknown"
        if num <= 500:
            return "Normal-cohort (P001-500)"
        if num <= 1000:
            return "Flat-cohort (P501-1000)"
        return "High-cohort (P1001-1500)"

    by_cohort = defaultdict(lambda: [0, 0])
    pred_dist_by_cohort = defaultdict(Counter)
    for pid, t, p in full_records:
        c = cohort_of(pid)
        by_cohort[c][1] += 1
        by_cohort[c][0] += int(t == p)
        pred_dist_by_cohort[c][_idx_to_name(p)] += 1

    for c, (ok, tot) in by_cohort.items():
        print(f"\n  {c}: {ok}/{tot} correct ({ok/tot*100:.1f}%)")
        top3 = pred_dist_by_cohort[c].most_common(3)
        print(f"    prediction spread: {top3}")
    print(
        "\n  Interpretation: if accuracy is wildly uneven across cohorts, or each\n"
        "  cohort's predictions collapse onto 1-2 classes, the model is keying\n"
        "  on batch/cohort cues rather than anatomy."
    )

    # ---- PROBE 4: trivial measurement-rule baseline ----
    print("\n" + "=" * 70)
    print("PROBE 4 — TRIVIAL MEASUREMENT-RULE BASELINE (no neural network)")
    print("=" * 70)
    for split_name, ds in [
        ("train", loaders.train_dataset),
        ("val", loaders.val_dataset),
        ("test", loaders.test_dataset),
    ]:
        acc = _rule_baseline(ds)
        print(f"  {split_name:>5}: applying brief's cm bands to true arch height "
              f"-> {acc:.2f}% accuracy")
    print(
        "\n  This is what a 5-line if/else achieves with NO machine learning,\n"
        "  given the measurements. Compare against PROBE 2a/2b above."
    )

    print("\n" + "=" * 70)
    print("READ THE FOUR PROBES TOGETHER. The pattern across them — not any\n"
          "single number — tells us whether this is leakage, an unlearnable\n"
          "image task, or a code bug. Paste this entire output back.")
    print("=" * 70)


if __name__ == "__main__":
    main()