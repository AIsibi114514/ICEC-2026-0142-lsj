"""评价指标：混淆矩阵、每类 P/R/F1、Wilson 置信区间。

在 25 个盲测样本上，1 个错就是 4 个百分点。所以任何一个准确率数字
都必须带置信区间，否则"96% vs 100%"这种差别其实读不出意义。
"""

from __future__ import annotations

import math

import numpy as np


def confusion_matrix(y_true: list[str], y_pred: list[str], classes: list[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    M = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[idx[t], idx[p]] += 1
    return M


def per_class_report(M: np.ndarray, classes: list[str]) -> list[dict]:
    rows = []
    for i, c in enumerate(classes):
        tp = int(M[i, i])
        fp = int(M[:, i].sum() - tp)
        fn = int(M[i, :].sum() - tp)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append(
            {
                "class": c,
                "support": int(M[i, :].sum()),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
            }
        )
    return rows


def macro_f1(M: np.ndarray, classes: list[str]) -> float:
    rows = per_class_report(M, classes)
    return float(np.mean([r["f1"] for r in rows])) if rows else 0.0


def accuracy(M: np.ndarray) -> float:
    total = M.sum()
    return float(np.trace(M) / total) if total else 0.0


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 区间。比正态近似更稳，小样本/极端比例（0% 或 100%）也能算。"""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def summarize(y_true: list[str], y_pred: list[str], classes: list[str]) -> dict:
    M = confusion_matrix(y_true, y_pred, classes)
    n = len(y_true)
    k = int(np.trace(M))
    lo, hi = wilson_ci(k, n)
    return {
        "n": n,
        "correct": k,
        "accuracy": round(accuracy(M), 4),
        "accuracy_ci95": [round(lo, 4), round(hi, 4)],
        "macro_f1": round(macro_f1(M, classes), 4),
        "confusion": M.tolist(),
        "classes": classes,
        "per_class": per_class_report(M, classes),
    }


def format_confusion(M: np.ndarray, classes: list[str]) -> str:
    w = max(len(c) for c in classes) + 2
    head = "true\\pred".ljust(w) + "".join(c[:14].rjust(16) for c in classes)
    lines = [head]
    for i, c in enumerate(classes):
        lines.append(c.ljust(w) + "".join(str(v).rjust(16) for v in M[i]))
    return "\n".join(lines)
