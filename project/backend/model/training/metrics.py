"""Metric computation for classification."""

from __future__ import annotations

import numpy as np
import torch


class ClassificationMetrics:
    """Accumulates predictions across a full epoch and produces metrics."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self._preds: list[int] = []
        self._labels: list[int] = []
        self._max_probs: list[float] = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        with torch.no_grad():
            probs = logits.softmax(dim=-1)
            top_p, top_i = probs.max(dim=-1)
            self._preds.extend(top_i.cpu().tolist())
            self._labels.extend(labels.cpu().tolist())
            self._max_probs.extend(top_p.cpu().tolist())

    # ---------------------------------------------------------- aggregation
    def accuracy(self) -> float:
        if not self._preds:
            return 0.0
        correct = sum(p == y for p, y in zip(self._preds, self._labels))
        return correct / len(self._preds)

    def per_class_accuracy(self) -> dict[int, float]:
        out: dict[int, float] = {}
        for c in range(self.num_classes):
            mask = [y == c for y in self._labels]
            n = sum(mask)
            if n == 0:
                out[c] = float("nan")
                continue
            correct = sum(p == c for p, y in zip(self._preds, self._labels) if y == c)
            out[c] = correct / n
        return out

    def confusion_matrix(self) -> np.ndarray:
        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for p, y in zip(self._preds, self._labels):
            cm[y, p] += 1
        return cm

    def macro_f1(self) -> float:
        cm = self.confusion_matrix()
        f1s = []
        for c in range(self.num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)
        return float(np.mean(f1s)) if f1s else 0.0

    def mean_confidence(self) -> float:
        return float(np.mean(self._max_probs)) if self._max_probs else 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy(),
            "macro_f1": self.macro_f1(),
            "mean_confidence": self.mean_confidence(),
        }
