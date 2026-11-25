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
# 转频：2026-11-25 加速度通道直接实测（1 档 27.2 Hz / 2 档 33.3 Hz），
# 取代此前"声学侧反推 1199/2170 r/min"的估计（实测 1632/2000 r/min，两者都不符）。
SHAFT_LOW, SHAFT_HIGH = 27.2, 33.3
# 叶片数：2026-11-25 由用户实数 **3 叶**（此前按 5 叶占位是错的）。
# 注意由此产生的重合：高档 BPF = 3 × 33.3 = 99.9 Hz，与 2× 电源电磁分量（100.0 Hz）
# 只差 0.1 Hz，**高档的 BPF 判据永远不可分离**；只有低档能分开（81.6 vs 100.0）。
BLADES = 3
BPF_LOW, BPF_HIGH = SHAFT_LOW * BLADES, SHAFT_HIGH * BLADES


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
    ap.add_argument("--shaft-low", type=float, default=SHAFT_LOW,
                    help=f"低档转频先验（Hz），默认 {SHAFT_LOW}（声学反推）")
    ap.add_argument("--shaft-high", type=float, default=SHAFT_HIGH,
                    help=f"高档转频先验（Hz），默认 {SHAFT_HIGH}（声学反推）")
    ap.add_argument("--dir", default=str(ACCEL), help="数据目录，默认 raw_accel/（自检时可指向别处）")
    ap.add_argument("--out", default=str(RESULTS), help="输出目录，默认 results/")
    args = ap.parse_args()
    bpf_low, bpf_high = [float(v) for v in args.bpfs.split(",")]
    shaft_low, shaft_high = args.shaft_low, args.shaft_high

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
    deviate: list[tuple[str, float, float]] = []
    weak1x: list[tuple[str, float, float, float]] = []
    for r in rows:
        t, xyz = load(accel / r["file"])
        sr = float(r["sample_rate"])
        mag = np.sqrt((xyz**2).sum(axis=1))
        x = mag - mag.mean()
        n = len(x)
        freq = np.fft.rfftfreq(n, 1.0 / sr)

        # ⚠️ 频域判据必须用「逐轴谱」，不能用合矢量 |a| 的谱。
        # |a| = √(x²+y²+z²) 在信号过零附近是非线性的：取模会把单频成分整流，
        # 实测（2026-11-26 试采模拟）z 轴 20 Hz 幅值 0.0074，取模后只剩 0.0031，
        # 而 40 Hz 反而从 0.0022 涨到 0.0058 —— 1× 被压、2× 被抬。
        # 由于 import_accel.py 已逐轴去直流，三轴都是纯交流，整流效应无法回避。
        # 时域指标（RMS/峭度/峰值因子/冲击数）仍用合矢量，这是振动烈度的常规口径。
        axis_specs = np.array([
            np.abs(np.fft.rfft((xyz[:, k] - xyz[:, k].mean()) * np.hanning(n))) / n
            for k in range(xyz.shape[1])
        ])
        spec_max = axis_specs.max(axis=0)          # 三轴取最强，取向无关

        def amp_axes(f0: float, tol: float = 1.5) -> float:
            """三轴分别取值后取最大——物理上就是"该频率在三轴中最强的那个"。"""
            return float(max(amp_at(freq, s, f0, tol) for s in axis_specs))

        # 转频：用「按档位的先验」，不是每段各自找最强峰。
        # 为什么必须这样：C1 是低档(≈20 Hz)、C2~C5 是高档(≈36.2 Hz)。
        # 若每段各自取 argmax，C3 完全可能锁到与 C1/C2 不同的峰上，
        # a1_abs 就失去可比性——而"C3 的 1× 是否高于正常"这条判据正是靠它。
        band = (freq >= 10) & (freq <= 50)
        f1_auto = float(freq[band][np.argmax(spec_max[band])]) if band.any() else float("nan")
        f1 = shaft_low if r["label"].startswith("C1") else shaft_high
        a1 = amp_axes(f1)
        a2 = amp_axes(f1 * 2)
        a3 = amp_axes(f1 * 3)
        if f1_auto == f1_auto and abs(f1_auto - f1) > 0.25 * f1:
            deviate.append((r["file"], round(f1_auto, 2), f1))
        # 转频先验的硬性自检：若该频率上的幅值比整体 RMS 还低 20 倍，
        # 说明这个先验根本不在这台机器的谱上，"1× 是否升高"就无从谈起。
        # 2026-11-25 实测即撞上此情形：先验 20 Hz → 1× 0.0008，而 RMS 0.200。
        rms_now = float(mag.std())
        if rms_now > 0 and a1 * 20 < rms_now:
            weak1x.append((r["file"], round(a1, 6), round(rms_now, 4), f1))
        ref = float(np.sqrt((spec_max**2).sum())) + 1e-20
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
            "f1_hz": round(f1, 2),
            "f1_auto": round(f1_auto, 2) if f1_auto == f1_auto else "",
            # 绝对幅值（m/s²）才是"1× 是否升高"的正确判据；
            # 相对幅值会被"整体频谱一起变大"稀释，只留作参考。
            "a1_abs": round(float(a1), 5),
            "a2_abs": round(float(a2), 5),
            "a1_rel": round(a1 / ref, 4),
            "a2_rel": round(a2 / ref, 4),
            "a3_rel": round(a3 / ref, 4),
            "hf_ratio": round(float(spec_max[hf].sum() / ref) if hf.any() else 0.0, 5),
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

    if weak1x:
        print(f"\n[严重] {len(weak1x)} 段的 1× 幅值比 RMS 低 20 倍以上，转频先验很可能不适用于这台机器：")
        for fn, a1v, rmsv, prior in weak1x[:6]:
            print(f"  · {fn}: 1×({prior:.1f} Hz)={a1v} m/s², RMS={rmsv} → 低 {rmsv/max(a1v,1e-12):.0f} 倍")
        print("    先验证转频再谈 1× 判据；此时任何 C3/C2 的比值都是噪声之比，不得作为结论。")

    if deviate:
        print(f"\n[注意] {len(deviate)} 段的最强峰与转频先验相差 >25%：")
        for fn, auto, prior in deviate[:6]:
            print(f"  · {fn}: 自动最强峰 {auto} Hz / 先验 {prior} Hz"
                  f"{'（先验可能不适用于该工况，或该段非稳态）'}")

    # ---- 判据可行性 ----
    print("\n[判据核对]")
    sr = feats[0]["sample_rate"]
    nyq = sr / 2
    verdicts = []
    verdicts.append((
        f"采样率 {sr:.0f} Hz → 奈奎斯特 {nyq:.0f} Hz",
        f"高档 BPF {bpf_high:.0f} Hz " + ("可观测" if nyq > bpf_high else "**不可观测（混叠，不得解释为特征）**")
        + ("；但它与 100 Hz 电磁分量相差 <2 Hz，**实际不可分离**" if abs(bpf_high - 100) < 2 else ""),
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
    verdicts.append(("谱中 100 Hz 的成分",
                     "2026-11-25 断电惰转实测：断电后 0.25 s 内衰减 10 倍，先于整体振动消失 "
                     "→ 判定为 2× 电源频率电磁分量，**不得作为任何机械判据**"))
    verdicts.append(("叶片数 / BPF",
                     f"实数 {BLADES} 叶 → BPF = {BPF_LOW:.1f} / {BPF_HIGH:.1f} Hz。"
                     f"高档 BPF 与 100 Hz 电磁分量仅差 {abs(BPF_HIGH - 100):.1f} Hz，**不可分离**；"
                     f"BPF 判据只能在低档做"))
    verdicts.append(("4–8 kHz 擦碰指纹",
                     "物理上不在加速度计带宽内，本通道**永远看不到**，不得声称可替代声学"))
    for a, b in verdicts:
        print(f"  · {a}\n      {b}")

    print(f"\n[accel] 逐段特征 → {out}")
    print("[accel] 结论怎么写：只写上面成立的判据；不成立的照实写，见 docs/加速度通道补采方案.md 第五节")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
