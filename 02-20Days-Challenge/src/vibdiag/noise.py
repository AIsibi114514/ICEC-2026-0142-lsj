"""抗噪实验。

评测的是一个"部署后才会遇到的问题"：真实车间/宿舍里不可能只有这台机器在响。
所以把留出样本按指定 SNR 叠加白噪声与粉红噪声，再看准确率掉到哪。

三个变量：
    噪声类型  white（电路/风道底噪，能量均匀） vs pink（空调/人流/远处车流，能量偏低频）
    SNR       20 / 15 / 10 / 5 / 0 / -5 dB
    前端      none（直接送特征）
              narrow（带通 80~3500 Hz，砍掉大部分宽带噪声）
              wide  （带通 80~7000 Hz，只砍次声与 8 kHz 以上）

**两个真实结论，都比"准确率 100%"重要：**

① 白噪声 10 dB 时，25 段全被判成 C5（擦碰），准确率从 100% 直接砸到 20%
   （= 5/25，等于瞎猜）。原因不是信号被淹，而是白噪声把 2~8 kHz 抬高，
   正好模仿了擦碰的谱形。这类"结构化误判"比随机出错更危险：
   模型不是在乱猜，是在自信地按一个错误的理由归类。
   逐条预测分布记在 noise_sweep.csv 的 pred_spread 列，可自行核对。

② 消噪并非越狠越好。narrow 前端在干净数据上就把准确率从 100% 掉到 80%，
   因为它把 C5 赖以区分的 4~8 kHz 一起砍掉了——**消掉噪声的同时也消掉了证据**。
   wide 前端只砍次声，干净数据不掉点，抗噪改善有限但不伤判别力。
"""

from __future__ import annotations

import csv
import glob
import os

import numpy as np

from . import audio, evaluate, features
from .model import LogisticRegression

# 前端档位：(名称, 传给 features.extract 的 denoise 参数)
FRONTENDS: dict[str, bool | tuple[float, float]] = {
    "none": False,
    "narrow_80_3500": (80.0, 3500.0),
    "wide_80_7000": (80.0, 7000.0),
}


def _clip_paths(root: str) -> list[str]:
    return sorted(glob.glob(os.path.join(root, "results", "blind_clips", "*.wav")))


def _load_clips(root: str) -> list[tuple[str, np.ndarray, int]]:
    out = []
    for p in _clip_paths(root):
        x, sr = audio.read_wav(p)
        out.append((os.path.splitext(os.path.basename(p))[0], audio.trim_steady(x, sr), sr))
    return out


def _labels_from_key(root: str) -> dict[str, str]:
    with open(os.path.join(root, "results", "blind_key.csv"), encoding="utf-8") as f:
        return {r["clip"]: r["true_label"] for r in csv.DictReader(f)}


def _feature_matrix(clips, names, denoise, snr: float | None = None, kind: str = "white", rng=None):
    """一次性抽完全部特征。

    注意：`features.extract` 必须放在按样本的循环里调用**一次**，
    不能写进按特征名的内层——否则每条音频会被重复提取 34 次，
    并且每次都会消耗新的噪声抽样，同一行特征其实来自不同的噪声实现。
    （这个 bug 在初版里真实存在过，把噪声扫描从 19 s 拖到 9 min。）
    """
    rows = []
    for _, x, sr in clips:
        sig = x if snr is None else audio.add_noise(x, snr, kind, rng)
        f = features.extract(sig, sr, denoise=denoise)
        rows.append([f[n] for n in names])
    return np.array(rows)


def _spread(preds: list[str]) -> str:
    uniq, cnt = np.unique(preds, return_counts=True)
    order = np.argsort(-cnt)
    return ";".join(f"{uniq[i]}:{int(cnt[i])}" for i in order)


def sweep(
    root: str,
    model: dict,
    names: list[str],
    snrs: tuple[float, ...] = (20.0, 15.0, 10.0, 5.0, 0.0, -5.0),
    kinds: tuple[str, ...] = ("white", "pink"),
    repeats: int = 8,
    seed: int = 7,
) -> list[dict]:
    clf = LogisticRegression.from_dict(model)
    clips = _load_clips(root)
    key = _labels_from_key(root)
    y_true = [key[c] for c, _, _ in clips]
    classes = sorted(set(y_true))
    n = len(y_true)

    rows: list[dict] = []
    for fe_name, denoise in FRONTENDS.items():
        preds = clf.predict(_feature_matrix(clips, names, denoise))
        s = evaluate.summarize(y_true, preds, classes)
        rows.append(
            {
                "frontend": fe_name,
                "noise": "clean",
                "snr_db": "",
                "n": n,
                "correct": s["correct"],
                "accuracy": s["accuracy"],
                "ci_lo": s["accuracy_ci95"][0],
                "ci_hi": s["accuracy_ci95"][1],
                "macro_f1": s["macro_f1"],
                "repeats": 1,
                "pred_spread": _spread(preds),
            }
        )
        for kind in kinds:
            for snr in snrs:
                accs, f1s, last = [], [], []
                for rpt in range(repeats):
                    rng = np.random.default_rng(seed + 1000 * rpt + int(abs(snr) * 10))
                    p = clf.predict(_feature_matrix(clips, names, denoise, snr, kind, rng))
                    r = evaluate.summarize(y_true, p, classes)
                    accs.append(r["accuracy"])
                    f1s.append(r["macro_f1"])
                    last = p
                acc = round(float(np.mean(accs)), 4)
                k = int(round(acc * n))
                lo, hi = evaluate.wilson_ci(k, n)
                rows.append(
                    {
                        "frontend": fe_name,
                        "noise": kind,
                        "snr_db": snr,
                        "n": n,
                        "correct": k,
                        "accuracy": acc,
                        "ci_lo": round(lo, 4),
                        "ci_hi": round(hi, 4),
                        "macro_f1": round(float(np.mean(f1s)), 4),
                        "repeats": repeats,
                        "pred_spread": _spread(last),
                    }
                )
    return rows


def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
