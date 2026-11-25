#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""加速度通道体检 + 振动特征提取 + 判据可行性核对。

先说清楚它不做什么：
    · 不复用 features.py 的 34 维（那套频带划到 8 kHz，在 200 Hz 采样下
      100 Hz 以上全是空值，算出来是一堆 0）
    · 不训练分类器。加速度通道的用途是**交叉验证声学结论**，
      尤其"C3 失衡是否在 1× 转频上留下痕迹"，不是再刷一个准确率

输出：
    results/accel_features.csv    逐段特征（RMS/峭度/1×2×3× 相对幅值/冲击计数）
    控制台：采样率体检、按工况的均值对比、判据可行性结论

用法：
    python scripts/accel_check.py
    python scripts/accel_check.py --bpfs 100,181        # 覆盖默认 BPF 假设
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ACCEL = ROOT / "raw_accel"
RESULTS = ROOT / "results"
# 声学侧实测反推：低档 1199 r/min、高档 2170 r/min、5 叶
SHAFT_LOW, SHAFT_HIGH = 20.0, 36.2
BPF_LOW, BPF_HIGH = SHAFT_LOW * 5, SHAFT_HIGH * 5


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        r = csv.DictReader(fh)
        for row in r:
            rows.append([float(row["t_s"]), float(row["x"]), float(row["y"]), float(row["z"])])
    a = np.array(rows)
    return a[:, 0], a[:, 1:4]


def kurtosis(x: np.ndarray) -> float:
    m = x.mean()
    s = x.std()
    return float(((x - m) ** 4).mean() / (s**4 + 1e-20)) if s > 0 else 0.0


def amp_at(freq: np.ndarray, mag: np.ndarray, f0: float, tol: float = 1.5) -> float:
    """取 f0±tol 内的谱峰幅值（tol 要大于频率分辨率）。"""
    m = (freq >= f0 - tol) & (freq <= f0 + tol)
    return float(mag[m].max()) if m.any() else 0.0


def impact_count(mag: np.ndarray, k: float = 4.0) -> int:
    """超过 k 倍标准差的峰个数，作为"冲击密集度"的粗估。"""
    thr = mag.mean() + k * mag.std()
    above = mag > thr
    # 数上升沿，避免一个峰被算成多次
    return int(np.sum(above[1:] & ~above[:-1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpfs", default=f"{BPF_LOW:.0f},{BPF_HIGH:.0f}", help="低/高档 BPF（Hz）")
    ap.add_argument("--dir", default=str(ACCEL), help="数据目录，默认 raw_accel/（自检时可指向别处）")
    ap.add_argument("--out", default=str(RESULTS), help="输出目录，默认 results/")
    args = ap.parse_args()
    bpf_low, bpf_high = [float(v) for v in args.bpfs.split(",")]

    accel = Path(args.dir)
    results = Path(args.out)
    idx_path = accel / "sample_index_accel.csv"
    if not idx_path.exists():
        print(f"[accel] 找不到 {idx_path}\n        先跑：python scripts/import_accel.py --src <导出目录> --plan ...")
        return 1
    rows = list(csv.DictReader(idx_path.open(encoding="utf-8")))
    if not rows:
        print("[accel] 登记表为空")
        return 1

    print(f"[accel] {len(rows)} 段\n")
    feats = []
    for r in rows:
        t, xyz = load(accel / r["file"])
        sr = float(r["sample_rate"])
        mag = np.sqrt((xyz**2).sum(axis=1))
        x = mag - mag.mean()
        n = len(x)
        spec = np.abs(np.fft.rfft(x * np.hanning(n))) / n
        freq = np.fft.rfftfreq(n, 1.0 / sr)

        # 转频：在 10~50 Hz 内找最强峰。找不到就退化为 NaN，不瞎猜。
        band = (freq >= 10) & (freq <= 50)
        f1 = float(freq[band][np.argmax(spec[band])]) if band.any() else float("nan")
        a1 = amp_at(freq, spec, f1) if f1 == f1 else 0.0
        a2 = amp_at(freq, spec, f1 * 2) if f1 == f1 else 0.0
        a3 = amp_at(freq, spec, f1 * 3) if f1 == f1 else 0.0
        ref = float(np.sqrt((spec**2).sum())) + 1e-20
        nyq = sr / 2
        hf = (freq >= 0.7 * nyq)
        feats.append({
            "file": r["file"],
            "label": r["label"],
            "sample_rate": int(sr),
            "nyquist_hz": round(nyq, 1),
            "bpf_visible": int(nyq > (bpf_high if "C2" in r["label"] or "C3" in r["label"] or "C4" in r["label"] or "C5" in r["label"] else bpf_low)),
            "rms_mag": round(float(mag.std()), 4),
            "peak_mag": round(float(np.abs(x).max()), 4),
            "crest": round(float(np.abs(x).max() / (mag.std() + 1e-20)), 3),
            "kurtosis": round(kurtosis(x), 3),
            "f1_hz": round(f1, 2) if f1 == f1 else "",
            # 绝对幅值（m/s²）才是"1× 是否升高"的正确判据；
            # 相对幅值会被"整体频谱一起变大"稀释，只留作参考。
            "a1_abs": round(float(a1), 5),
            "a2_abs": round(float(a2), 5),
            "a1_rel": round(a1 / ref, 4),
            "a2_rel": round(a2 / ref, 4),
            "a3_rel": round(a3 / ref, 4),
            "hf_ratio": round(float(spec[hf].sum() / ref) if hf.any() else 0.0, 5),
            "impacts": impact_count(x),
        })

    results.mkdir(parents=True, exist_ok=True)
    out = results / "accel_features.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(feats[0].keys()))
        w.writeheader()
        w.writerows(feats)

    # ---- 按工况汇总 ----
    labels = sorted({f["label"] for f in feats})
    print(f"{'工况':22s}{'n':>4s}{'RMS':>9s}{'峭度':>8s}{'1×绝对':>10s}{'2×绝对':>10s}{'冲击数':>7s}")
    summary = {}
    for lab in labels:
        g = [f for f in feats if f["label"] == lab]
        def m(key):
            v = [f[key] for f in g if isinstance(f[key], (int, float))]
            return float(np.mean(v)) if v else float("nan")
        summary[lab] = {k: m(k) for k in ("rms_mag", "kurtosis", "a1_abs", "a2_abs", "a1_rel", "impacts")}
        print(f"{lab:22s}{len(g):4d}{m('rms_mag'):9.3f}{m('kurtosis'):8.2f}{m('a1_abs'):10.4f}{m('a2_abs'):10.4f}{m('impacts'):7.1f}")

    # ---- 判据可行性 ----
    print("\n[判据核对]")
    sr = feats[0]["sample_rate"]
    nyq = sr / 2
    verdicts = []
    verdicts.append((
        f"采样率 {sr:.0f} Hz → 奈奎斯特 {nyq:.0f} Hz",
        f"高档 BPF {bpf_high:.0f} Hz " + ("可观测" if nyq > bpf_high else "**不可观测（混叠，不得解释为特征）**"),
    ))
    normal = [summary.get(l, {}).get("a1_abs", float("nan")) for l in labels if l.startswith(("C1", "C2"))]
    imb = summary.get("C3_imbalance", {}).get("a1_abs", float("nan"))
    if normal and imb == imb:
        base = float(np.nanmean(normal))
        ratio = imb / base if base > 0 else float("nan")
        rms_ratio = (summary.get("C3_imbalance", {}).get("rms_mag", float("nan")) /
                     float(np.nanmean([summary[l]["rms_mag"] for l in labels if l.startswith(("C1", "C2"))])))
        verdicts.append(("C3 失衡 → 1× 幅值应升高",
                         f"1× 绝对幅值 {ratio:.2f}×（基线 {base:.4f} m/s²）、RMS {rms_ratio:.2f}×" +
                         ("，判据成立" if ratio > 1.3 else "，**判据不成立，不要硬说成立**")))
    rub = summary.get("C5_rubbing", {}).get("kurtosis", float("nan"))
    nk = float(np.nanmean([summary[l]["kurtosis"] for l in labels if l.startswith(("C1", "C2"))] or [float("nan")]))
    if rub == rub and nk == nk:
        verdicts.append(("C5 擦碰 → 时域冲击/峭度升高", f"峭度 {rub:.2f} vs 正常 {nk:.2f}" +
                         ("，冲击判据可用" if rub > nk * 1.3 else "，**冲击判据不可用：证明高频故障需声学通道**")))
    verdicts.append(("4–8 kHz 擦碰指纹",
                     "物理上不在加速度计带宽内，本通道**永远看不到**，不得声称可替代声学"))
    for a, b in verdicts:
        print(f"  · {a}\n      {b}")

    print(f"\n[accel] 逐段特征 → {out}")
    print("[accel] 结论怎么写：只写上面成立的判据；不成立的照实写，见 docs/加速度通道补采方案.md 第五节")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
