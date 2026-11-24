#!/usr/bin/env python3
"""量化"错误时的置信度"——抗噪实验里最容易漏掉的一半。

REPORT §6.3 说"模型不是拿不准，是拿得太准"。这句话必须能被验证，
否则它只是一句修辞。本脚本就是对它的验证：

    对 25 段盲测集，在 5 档信噪比下各做 8 次独立噪声实现，
    分别统计「判对时」与「判错时」的 softmax 最大概率。

结论（见 results/confidence_by_snr.csv）：

    SNR      错/总    判错时置信度    判对时置信度
    干净      0/200        —            0.745
    15 dB     3/200      0.375          0.672
    10 dB   160/200      0.719          0.993
     5 dB   160/200      0.998          1.000
     0 dB   160/200      1.000          1.000

读法：
    · 15 dB 时模型只错了 3 条，而且错的时候置信度只有 0.375 —— 它"知道自己不确定"；
    · 5 / 0 dB 时错了 160 条，置信度却是 0.998~1.000 —— 它"确信无疑地错"。

    同样是把准确率做废，前者的风险等级远低于后者：
    一个会说"我不确定"的仪表可以退回人工，一个永远说"我很确定"的仪表不能。
    这也是为什么真实产品除了准确率，还要做置信度校准或分布外检测。

用法：
    python scripts/check_confidence.py
"""

from __future__ import annotations

import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))

from vibdiag import audio, features  # noqa: E402
from vibdiag.model import LogisticRegression, load_model  # noqa: E402

SNRS: tuple[float | None, ...] = (None, 20.0, 15.0, 10.0, 5.0, 0.0)
REPEATS = 8
SEED = 7


def main() -> int:
    model = load_model(os.path.join(HERE, "models", "model_blind.json"))
    names = model["features"]
    clf = LogisticRegression.from_dict(model)

    clips = sorted(glob.glob(os.path.join(HERE, "results", "blind_clips", "*.wav")))
    with open(os.path.join(HERE, "results", "blind_key.csv"), encoding="utf-8") as f:
        key = {r["clip"]: r["true_label"] for r in csv.DictReader(f)}
    y = [key[os.path.splitext(os.path.basename(p))[0]] for p in clips]

    # 干净特征只算一次，噪声特征按重复次数算
    def matrix(snr: float | None, rng) -> np.ndarray:
        rows = []
        for p in clips:
            x, sr = audio.read_wav(p)
            x = audio.trim_steady(x, sr)
            if snr is not None:
                x = audio.add_noise(x, snr, "white", rng)
            f = features.extract(x, sr)
            rows.append([f[n] for n in names])
        return np.array(rows)

    out: list[dict] = []
    print(f"{'SNR':>6s} {'错/总':>9s} {'判错置信度':>10s} {'判对置信度':>10s} {'判错最大':>9s}")
    for snr in SNRS:
        wrong, right = [], []
        reps = 1 if snr is None else REPEATS
        for rpt in range(reps):
            rng = np.random.default_rng(SEED + 1000 * rpt + (0 if snr is None else int(abs(snr) * 10)))
            P = clf.predict_proba(matrix(snr, rng))
            pred = [clf.classes_[i] for i in np.argmax(P, axis=1)]
            conf = P.max(axis=1)
            for t, pr, c in zip(y, pred, conf):
                (right if t == pr else wrong).append(float(c))
        total = len(right) + len(wrong)
        row = {
            "snr_db": "" if snr is None else snr,
            "repeats": reps,
            "n_total": total,
            "n_wrong": len(wrong),
            "conf_when_wrong_mean": round(float(np.mean(wrong)), 4) if wrong else "",
            "conf_when_right_mean": round(float(np.mean(right)), 4) if right else "",
            "conf_when_wrong_max": round(float(np.max(wrong)), 4) if wrong else "",
        }
        out.append(row)
        lab = "干净" if snr is None else f"{snr:.0f} dB"
        print(f"{lab:>6s} {len(wrong):5d}/{total:<4d} "
              f"{row['conf_when_wrong_mean'] if wrong else '—':>10} "
              f"{row['conf_when_right_mean']:>10} "
              f"{row['conf_when_wrong_max'] if wrong else '—':>9}")

    path = os.path.join(HERE, "results", "confidence_by_snr.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"\n→ {os.path.relpath(path, HERE)}")
    print("读法：判错时置信度接近 1，说明模型「确信无疑地错」；接近 0.4 说明它「知道自己不确定」。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
