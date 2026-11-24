#!/usr/bin/env python3
"""生成全部图表（fig1~fig8）。

每张图都回答一个问题，不摆装饰：
    fig1  五项工况长什么样           → 波形 + 频谱图并列，人眼先看一眼
    fig2  频谱差在哪                  → 5 条谱线叠在一起，看谐波与高频拖尾
    fig3  哪些特征真的分得开          → 6 个关键特征的箱线图
    fig4  叶片数是怎么反推出来的      → 次谐波信噪比柱状图
    fig5  盲测错在哪                  → 混淆矩阵
    fig6  噪声加到多大就崩            → 准确率-SNR 曲线（白/粉 × 有无带通）
    fig7  模型在看哪几个特征          → 逻辑回归权重条图
    fig8  换台手机还行不行            → 有/无绝对电平特征的 CV 对比

字体用 Microsoft YaHei（Windows 自带），中文不会变成方框。
"""

from __future__ import annotations

import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vibdiag import CONDITIONS, CONDITION_ZH, dataset, derivations, features  # noqa: E402

FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")

PALETTE = {
    "C1_normal_low": "#1f77b4",
    "C2_normal_high": "#2ca02c",
    "C3_imbalance": "#ff7f0e",
    "C4_obstruction": "#d62728",
    "C5_rubbing": "#9467bd",
}


def _short(label: str) -> str:
    return CONDITION_ZH[label]


def _first_sample(samples: list[dict], label: str) -> dict:
    return next(s for s in samples if s["label"] == label)


def fig1_overview(samples: list[dict]) -> str:
    fig, axes = plt.subplots(len(CONDITIONS), 2, figsize=(12.5, 10.5))
    for i, label in enumerate(CONDITIONS):
        s = _first_sample(samples, label)
        x, sr = dataset.load_steady(s)
        t = np.arange(len(x)) / sr
        ax = axes[i, 0]
        seg = int(0.15 * sr)
        ax.plot(t[:seg], x[:seg], lw=0.7, color=PALETTE[label])
        ax.set_ylim(-1, 1)
        ax.set_ylabel("幅值", fontsize=9)
        ax.set_title(f"{label} · {_short(label)}（稳态段前 150 ms）", fontsize=9, loc="left")
        ax.grid(alpha=0.3)

        ax2 = axes[i, 1]
        ax2.specgram(x, NFFT=1024, Fs=sr, noverlap=768, cmap="magma", vmin=-125, vmax=-25)
        ax2.set_ylim(0, 2000)
        ax2.set_title(f"{label} · 频谱图 0~2 kHz", fontsize=9, loc="left")
        ax2.set_ylabel("Hz", fontsize=9)
        if i == len(CONDITIONS) - 1:
            axes[i, 0].set_xlabel("时间 / s")
            ax2.set_xlabel("时间 / s")
    fig.suptitle("图 1  五种工况的时域波形与频谱图（各取同一台 180 mm 台扇的首段录音）", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(FIG, "fig1_waveform_spectrogram.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig2_spectra(samples: list[dict]) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    # 每个类别的 15 条谱，用于 (b) 的平均
    per_class: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {c: [] for c in CONDITIONS}
    for label in CONDITIONS:
        s = _first_sample(samples, label)
        x, sr = dataset.load_steady(s)
        sp = features.spectrum(x, sr)
        m = sp.freq > 20
        ax1.semilogx(sp.freq[m], 20 * np.log10(sp.mag[m] + 1e-20), lw=0.7,
                     color=PALETTE[label], label=f"{label} {_short(label)}", alpha=0.9)
    for s in samples:
        x, sr = dataset.load_steady(s)
        sp = features.spectrum(x, sr)
        per_class[s["label"]].append((sp.freq, sp.mag))

    ax1.set_xlim(20, 8000)
    ax1.set_ylim(-105, -10)
    ax1.set_xlabel("频率 / Hz（对数轴）")
    ax1.set_ylabel("幅度 / dB")
    ax1.set_title("(a) 全频段谱形：擦碰把 2~8 kHz 抬起来，遮挡把 100~800 Hz 抬起来", fontsize=9, loc="left")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(fontsize=8, loc="lower left")

    # (b) 15 条谱的平均（已做 5 Hz 平滑，否则 0.5 Hz 分辨率下全是尖刺看不清）
    k = 9  # ≈4.5 Hz 的滑动平均
    kernel = np.ones(k) / k
    for label in CONDITIONS:
        stack = []
        freqs = None
        for freq, mag in per_class[label]:
            keep = (freq >= 60) & (freq <= 700)
            freqs = freq[keep]
            stack.append(20 * np.log10(mag[keep] + 1e-20))
        arr = np.array(stack)
        # 用 'valid' 卷积避免两端被零填充拉出假的尖峰
        sm = np.array([np.convolve(row, kernel, mode="valid") for row in arr]).mean(axis=0)
        fc = freqs[k // 2 : len(freqs) - k // 2]
        bpf = features.find_bpf(features.spectrum(*dataset.load_steady(_first_sample(samples, label))))
        ax2.plot(fc, sm, lw=1.4, color=PALETTE[label],
                 label=f"{label[:2]} {_short(label)}  BPF≈{bpf:.0f} Hz")
    ax2.set_xlim(60, 700)
    ax2.set_xlabel("频率 / Hz")
    ax2.set_ylabel("幅度 / dB（15 段平均）")
    ax2.set_title("(b) 高档四类的转速几乎相同（BPF 177~184 Hz），差异不在速度而在别处", fontsize=9, loc="left")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    out = os.path.join(FIG, "fig2_spectrum_compare.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig3_features(samples: list[dict]) -> str:
    keys = ["bpf_hz", "spec_centroid", "hf_noise_ratio", "band_300_800", "crest_factor", "spec_flatness"]
    titles = {
        "bpf_hz": "BPF 叶片通过频率 / Hz",
        "spec_centroid": "谱质心 / Hz",
        "hf_noise_ratio": "2~8 kHz 能量占比",
        "band_300_800": "300~800 Hz 能量占比",
        "crest_factor": "峰值因子（峰值/RMS）",
        "spec_flatness": "谱平坦度",
    }
    data: dict[str, dict[str, list[float]]] = {k: {c: [] for c in CONDITIONS} for k in keys}
    for s in samples:
        x, sr = dataset.load_steady(s)
        f = features.extract(x, sr)
        for k in keys:
            data[k][s["label"]].append(f[k])

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    for ax, k in zip(axes.ravel(), keys):
        vals = [data[k][c] for c in CONDITIONS]
        bp = ax.boxplot(vals, patch_artist=True, widths=0.6, showfliers=False)
        for patch, c in zip(bp["boxes"], CONDITIONS):
            patch.set_facecolor(PALETTE[c])
            patch.set_alpha(0.55)
        for med in bp["medians"]:
            med.set_color("black")
        ax.set_xticklabels([c[:2] for c in CONDITIONS])
        ax.set_title(titles[k], fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("图 3  6 个关键特征在五类工况上的分布（每类 15 组；C1 是低档，其余为高档）", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(FIG, "fig3_feature_boxplots.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig4_blade_count(samples: list[dict]) -> str:
    profiles = []
    for s in samples:
        x, sr = dataset.load_steady(s)
        profiles.append(derivations.subharmonic_profile(x, sr))
    n_est, med = derivations.estimate_blade_count(profiles)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    ns = sorted(med)
    ax1.bar([str(n) for n in ns], [med[n] for n in ns],
            color=["#d62728" if n == n_est else "#9db4c0" for n in ns])
    ax1.axhline(2.0, color="gray", ls="--", lw=1)
    ax1.text(len(ns) - 1, 2.2, "判据：>2 才算有谱线", fontsize=8, ha="right", color="gray")
    ax1.set_xlabel("假设的叶片数 N")
    ax1.set_ylabel("BPF/N 处 峰值/局部底噪")
    ax1.set_title(f"(a) 次谐波检验（75 段中位数）→ 推定 N={n_est}", fontsize=9, loc="left")
    ax1.grid(alpha=0.3, axis="y")

    # 低档样例在 15~60 Hz 的实测谱，标出轴频位置
    s = _first_sample(samples, "C1_normal_low")
    x, sr = dataset.load_steady(s)
    sp = features.spectrum(x, sr)
    m = (sp.freq >= 10) & (sp.freq <= 70)
    ax2.plot(sp.freq[m], 20 * np.log10(sp.mag[m] + 1e-20), lw=1.0, color="#1f77b4")
    bpf = features.find_bpf(sp)
    ax2.axvline(bpf / 5, color="#d62728", ls="--", lw=1)
    ax2.axvline(50.0, color="#2ca02c", ls=":", lw=1.2)
    ylo, yhi = ax2.get_ylim()
    ax2.set_ylim(ylo, yhi + (yhi - ylo) * 0.12)
    ax2.annotate(f"BPF/5 = {bpf/5:.1f} Hz\n推定轴频", (bpf / 5, 0.93), xycoords=("data", "axes fraction"),
                 fontsize=8, color="#d62728", ha="center", va="top")
    ax2.annotate("50 Hz 市电", (50.0, 0.93), xycoords=("data", "axes fraction"),
                 fontsize=8, color="#2ca02c", ha="center", va="top")
    ax2.set_xlabel("频率 / Hz")
    ax2.set_ylabel("幅度 / dB")
    ax2.set_title("(b) 低档样例 10~70 Hz：轴频处确有独立谱线", fontsize=9, loc="left")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIG, "fig4_blade_count.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig5_confusion() -> str:
    with open(os.path.join(RES, "blind_score.json"), encoding="utf-8") as f:
        sc = json.load(f)
    M = np.array(sc["confusion"])
    classes = sc["classes"]
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels([f"{c[:2]}\n{_short(c)}" for c in classes], fontsize=8)
    ax.set_yticklabels([f"{c[:2]} {_short(c)}" for c in classes], fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=11,
                    color="white" if M[i, j] > M.max() / 2 else "black")
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    ax.set_title(
        f"图 5  盲测混淆矩阵（{sc['correct']}/{sc['n']} = {sc['accuracy']:.1%}，"
        f"95%CI {sc['accuracy_ci95'][0]:.0%}~{sc['accuracy_ci95'][1]:.0%}）",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    out = os.path.join(FIG, "fig5_blind_confusion.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig6_noise() -> str:
    with open(os.path.join(RES, "noise_sweep.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    frontends = []
    for r in rows:
        if r["frontend"] not in frontends:
            frontends.append(r["frontend"])
    fe_title = {
        "none": "(a) 无前端：原样送特征",
        "narrow_80_3500": "(b) 窄带 80~3500 Hz",
        "wide_80_7000": "(c) 宽带 80~7000 Hz",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    flat = axes.ravel()
    for ax, fe in zip(flat[:3], frontends):
        for kind, marker, col in (("white", "o", "#1f77b4"), ("pink", "s", "#d62728")):
            pts = [r for r in rows if r["frontend"] == fe and r["noise"] == kind and r["snr_db"] != ""]
            xs = [float(r["snr_db"]) for r in pts]
            ys = [float(r["accuracy"]) for r in pts]
            lo = [float(r["ci_lo"]) for r in pts]
            hi = [float(r["ci_hi"]) for r in pts]
            ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(lo), np.array(hi) - np.array(ys)],
                        marker=marker, color=col, capsize=3, lw=1.4,
                        label="白噪声" if kind == "white" else "粉红噪声")
        clean = [r for r in rows if r["frontend"] == fe and r["snr_db"] == ""]
        if clean:
            ax.axhline(float(clean[0]["accuracy"]), color="gray", ls="--", lw=1, label="干净基线")
        ax.axhline(0.2, color="black", ls=":", lw=0.8)
        ax.text(20, 0.225, "瞎猜水平 20%", fontsize=7, color="black")
        ax.set_xlabel("信噪比 SNR / dB（越小越吵）")
        ax.invert_xaxis()
        ax.set_ylim(-0.03, 1.08)
        ax.grid(alpha=0.3)
        ax.set_title(fe_title.get(fe, fe), fontsize=9, loc="left")
        ax.legend(fontsize=7, loc="lower left")

    # (d) 准确率崩了，但"判错时的置信度"反而涨到 1.0
    ax = flat[3]
    with open(os.path.join(RES, "confidence_by_snr.csv"), encoding="utf-8") as f:
        conf_rows = [r for r in csv.DictReader(f) if r["snr_db"] != ""]
    # 置信度表与抗噪表覆盖的 SNR 档位不完全相同（前者没有 -5 dB），按 SNR 对齐再画
    acc_by_snr = {
        float(r["snr_db"]): float(r["accuracy"])
        for r in rows
        if r["frontend"] == "none" and r["noise"] == "white" and r["snr_db"] != ""
    }
    xs = [float(r["snr_db"]) for r in conf_rows]
    acc = [acc_by_snr[x] for x in xs]
    xs_w = [float(r["snr_db"]) for r in conf_rows if r["conf_when_wrong_mean"]]
    wrong = [float(r["conf_when_wrong_mean"]) for r in conf_rows if r["conf_when_wrong_mean"]]
    right = [float(r["conf_when_right_mean"]) for r in conf_rows]
    ax.plot(xs, acc, "o-", color="#1f77b4", label="准确率（无前端·白噪）")
    ax.plot(xs_w, wrong, "s-", color="#d62728", label="判错时的置信度")
    ax.plot(xs, right, "^--", color="#2ca02c", lw=1, label="判对时的置信度")
    ax.set_xlabel("信噪比 SNR / dB（越小越吵）")
    ax.set_ylabel("比例 / 置信度")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{int(x)}" for x in xs])
    ax.invert_xaxis()
    ax.set_ylim(-0.03, 1.08)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="center left")
    ax.set_title("(d) 准确率崩到 20%，判错时的置信度却涨到 1.000——「自信地错」", fontsize=9, loc="left")

    flat[0].set_ylabel("准确率")
    flat[2].set_ylabel("准确率")

    fig.suptitle("图 6  抗噪：消噪前端不是越狠越好，而最危险的不是错，是错了还很有把握", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(FIG, "fig6_noise_robustness.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig7_importance() -> str:
    rows = []
    with open(os.path.join(RES, "feature_importance.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:18]
    names = [r["feature"] for r in rows][::-1]
    vals = [float(r["max_abs_weight"]) for r in rows][::-1]
    fig, ax = plt.subplots(figsize=(8.2, 6))
    bars = ax.barh(names, vals, color="#4c72b0")
    ax.set_xlabel("|权重| 最大值（跨 5 类）")
    ax.set_title("图 7  逻辑回归看重的特征（在全部 75 段上训练）", fontsize=10)
    ax.grid(alpha=0.3, axis="x")
    for b, v in zip(bars, vals):
        ax.text(v + max(vals) * 0.01, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=7)
    fig.tight_layout()
    out = os.path.join(FIG, "fig7_feature_importance.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def fig8_stress() -> str:
    """四项压力测试：增益漂移 / 少样本 / 窗口长度 / 未见故障检出。"""
    def read_csv(name: str) -> list[dict]:
        with open(os.path.join(RES, name), encoding="utf-8") as f:
            return list(csv.DictReader(f))

    gain = read_csv("stress_gain.csv")
    few = read_csv("stress_fewshot.csv")
    win = read_csv("stress_window.csv")
    with open(os.path.join(RES, "stress_novelty.json"), encoding="utf-8") as f:
        nov = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8))

    # (a) 增益漂移
    ax = axes[0, 0]
    g = [float(r["gain_db"]) for r in gain]
    ax.plot(g, [float(r["acc_with_level"]) for r in gain], "o-", color="#d62728", label="含绝对电平特征")
    ax.plot(g, [float(r["acc_no_level"]) for r in gain], "s-", color="#2ca02c", label="去掉绝对电平")
    ax.axhline(0.2, color="black", ls=":", lw=0.8)
    ax.set_xlabel("整体增益偏移 / dB")
    ax.set_ylabel("75 段全量准确率")
    ax.set_ylim(0, 1.08)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower center")
    ax.set_title("(a) 换手机/改距离：含电平特征在 -18 dB 就崩了", fontsize=9, loc="left")

    # (b) 少样本
    ax = axes[0, 1]
    ns = [int(r["n_per_class"]) for r in few]
    mean = [float(r["acc_mean"]) for r in few]
    sd = [float(r["acc_sd"]) for r in few]
    ax.errorbar(ns, mean, yerr=sd, marker="o", color="#1f77b4", capsize=3)
    for x, m in zip(ns, mean):
        ax.annotate(f"{m:.1%}", (x, m), textcoords="offset points", xytext=(0, 8), fontsize=7, ha="center")
    ax.set_xlabel("每类训练样本数")
    ax.set_ylabel("准确率（30 次重复均值 ± 1σ）")
    ax.set_ylim(0.75, 1.03)
    ax.grid(alpha=0.3)
    ax.set_title("(b) 少样本：每类 3 段就够，1 段时最差掉到 81%", fontsize=9, loc="left")

    # (c) 窗口长度：干净 vs 较吵
    ax = axes[1, 0]
    ws = [float(r["window_s"]) for r in win]
    ax.plot(ws, [float(r["acc_clean"]) for r in win], "o-", color="#4c72b0", label="干净输入")
    ax.plot(ws, [float(r["acc_noisy"]) for r in win], "s-", color="#d62728",
            label=f'{float(win[0]["noise_snr"]):.0f} dB 白噪声')
    ax.set_xlabel("分析窗长 / s")
    ax.set_ylabel("窗口级准确率")
    ax.set_ylim(0.96, 1.005)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.annotate("干净：所有窗长都 100%", (0.25, 1.0035), fontsize=7.5, va="top")
    ax.set_title("(c) 窗长：注意纵轴只有 4 个百分点——窗长的收益很小，成本却是 8 倍延迟",
                 fontsize=9, loc="left")

    # (d) 未见故障：马氏距离分布
    ax = axes[1, 1]
    dist = nov["distance_samples"]
    order = ["normal", *nov["anomalies"].keys()]
    pos = np.arange(len(order))
    data = [dist[k] for k in order]
    vp = ax.violinplot(data, positions=pos, showmedians=True, widths=0.75)
    for i, (body, k) in enumerate(zip(vp["bodies"], order)):
        body.set_facecolor("#9db4c0" if k == "normal" else ["#ff7f0e", "#d62728", "#9467bd"][i - 1])
        body.set_alpha(0.7)
    ax.axhline(nov["threshold_mahalanobis"], color="black", ls="--", lw=1.2)
    ax.text(0.0, nov["threshold_mahalanobis"] * 1.03, "判定阈值（正常样本 95 分位）", fontsize=7.5)
    ax.set_xticks(pos)
    ax.set_xticklabels(["正常\n(建模用)"] + [_short(k) for k in nov["anomalies"]], fontsize=8)
    ax.set_ylabel("马氏距离（越大越不像正常）")
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("(d) 未见故障：只学过'正常'，三类新故障都落在阈值之外", fontsize=9, loc="left")

    fig.suptitle("图 8  四项压力测试：把「100% 准确率」逼到真实边界", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = os.path.join(FIG, "fig8_stress_tests.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def build_all(root: str = ROOT) -> list[str]:
    os.makedirs(FIG, exist_ok=True)
    samples = dataset.list_samples()
    made = [
        fig1_overview(samples),
        fig2_spectra(samples),
        fig3_features(samples),
        fig4_blade_count(samples),
        fig5_confusion(),
        fig6_noise(),
        fig7_importance(),
        fig8_stress(),
    ]
    return made


if __name__ == "__main__":
    for p in build_all():
        print("wrote", os.path.basename(p))
