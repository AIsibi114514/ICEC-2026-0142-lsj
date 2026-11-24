#!/usr/bin/env python3
"""规则基线 vs 统计模型：两个模型各有各的脆法。

REPORT §5.3 展示了 3 层 CART 学到的 4 个阈值（skewness / rms_dbfs / kurtosis）。
一个自然的问题是：既然规则这么简单就能 100%，为什么要用 34 维的逻辑回归？

本脚本把两个模型放在同一组压力条件下对比，答案是：
**它们不是"谁更好"，而是"分别在什么条件下先坏"。**

实测（75 段全量；树只用类内 01~10 共 50 段训练，所以 0 dB 那格不是满分：

    条件              规则树(3特征4阈值)   逻辑回归(34维)
    干净                    99%               100%
    增益 -6 dB              59%                81%
    增益 -12 dB             59%                60%
    增益 -24 dB             59%                35%
    白噪 20 dB              98%               100%
    白噪 15 dB              85%               100%
    白噪 10 dB              68%                20%   ← 逻辑回归崩了
    白噪  5 dB              47%                20%

两个不同的失效机制：
  · 规则树死于「增益」：它第二层就是 rms_dbfs <= -11.15 dBFS 这个绝对音量阈值，
    增益一降全部落到 C1，准确率卡死在 59%（46/75 判成 C1）。
    但它在噪声下只缓慢劣化，因为硬阈值不易被整体平移带偏。
  · 逻辑回归死于「噪声」：34 维一起平移，把样本整体推进 C5 的区域，
    softmax 直接饱和（§6.4 已证明此时置信度 0.998），一步塌到 20%。
    但它在衰减型增益下比树好，因为它靠的是多维形状量而不是单一音量阈值。

→ 结论：没有"更稳的模型"，只有"稳在哪种扰动上"。
   如果目标设备固定、录音链路固定（同一台手机），逻辑回归更好；
   如果录音增益不可控，硬阈值规则反而更抗跌——但那也只是 59%，同样不能用。
   真正的解不是换模型，而是把增益相关特征整体摘掉（§7.1 的 31 维版本）。

用法：
    python scripts/compare_baselines.py
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))

from vibdiag import audio, dataset, evaluate, features  # noqa: E402
from vibdiag.model import DecisionTree, LogisticRegression, load_model  # noqa: E402

GAINS: tuple[int, ...] = (-24, -18, -12, -6, 0, 6, 12)
SNRS: tuple[int, ...] = (20, 15, 10, 5)
NOISE_REPEATS = 5


def main() -> int:
    samples = dataset.list_samples()
    names = features.feature_names(True)
    rows, y = [], []
    for s in samples:
        x, sr = dataset.load_steady(s)
        rows.append(features.extract(x, sr))
        y.append(s["label"])
    X = np.array([[f[n] for n in names] for f in rows])
    classes = sorted(set(y))

    # 与 run.py train 相同的切分：类内 01~10 训练（50 段），11~15 留出（25 段）
    train_idx = [i for i, s in enumerate(samples) if s["index"] <= 10]
    tree = DecisionTree(max_depth=3, min_samples_leaf=3).fit(X[train_idx], [y[i] for i in train_idx])
    clf = LogisticRegression.from_dict(load_model(os.path.join(HERE, "models", "model_full.json")))

    def acc(model, Xc: np.ndarray) -> float:
        M = evaluate.confusion_matrix(y, model.predict(Xc), classes)
        return float(evaluate.accuracy(M))

    out: list[dict] = []

    # 干净基线
    out.append({"condition": "clean", "param": "", "acc_tree": round(acc(tree, X), 4),
                "acc_logistic": round(acc(clf, X), 4)})

    for g in GAINS:
        if g == 0:
            continue
        feats_g = []
        for s in samples:
            x, sr = dataset.load_steady(s)
            f = features.extract(x * (10 ** (g / 20.0)), sr)
            feats_g.append([f[n] for n in names])
        Xg = np.array(feats_g)
        out.append({"condition": "gain_db", "param": g,
                    "acc_tree": round(acc(tree, Xg), 4), "acc_logistic": round(acc(clf, Xg), 4)})

    for snr in SNRS:
        at, al = [], []
        for rpt in range(NOISE_REPEATS):
            rng = np.random.default_rng(100 + rpt)
            feats_n = []
            for s in samples:
                x, sr = dataset.load_steady(s)
                f = features.extract(audio.add_noise(x, snr, "white", rng), sr)
                feats_n.append([f[n] for n in names])
            Xn = np.array(feats_n)
            at.append(acc(tree, Xn))
            al.append(acc(clf, Xn))
        out.append({"condition": "white_noise_snr_db", "param": snr,
                    "acc_tree": round(float(np.mean(at)), 4),
                    "acc_logistic": round(float(np.mean(al)), 4)})

    print(f"{'条件':>18s} {'参数':>6s} {'规则树':>8s} {'逻辑回归':>9s}")
    for r in out:
        print(f"{r['condition']:>18s} {str(r['param']):>6s} {r['acc_tree']:>8.0%} {r['acc_logistic']:>9.0%}")

    path = os.path.join(HERE, "results", "baseline_compare.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"\n→ {os.path.relpath(path, HERE)}")
    print("读法：规则树死于增益（卡死在 59%，全部落 C1），逻辑回归死于噪声（一步塌到 20%）。")
    print("      没有更稳的模型，只有稳在哪种扰动上。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
