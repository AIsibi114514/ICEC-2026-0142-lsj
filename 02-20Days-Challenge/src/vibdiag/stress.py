"""压力测试：把"100% 准确率"这件事逼到它真实的边界上。

干净数据上 5 类线性可分，这是事实，但它不是一个可以用来吹的结论——
它只说明"这批录音里 5 个已知工况彼此差别很大"。
一台真正能用的诊断仪会碰到四类更难的事，这里逐一测：

  A. 窗口长度   只给 0.25 s 还能不能判？（决定实时性）
  B. 少样本     每类只给 1~10 段，剩下的能不能分对？
  C. 未见故障   训练时只见过"失衡"，遇到"擦碰/遮挡"能不能报异常？
  D. 增益漂移   换台手机、挪一下距离，绝对音量变了会不会全错？

这四个数字才是这个项目的真实成色。
"""

from __future__ import annotations

import numpy as np

from . import audio, evaluate, features
from .model import LogisticRegression


# ---------------------------------------------------------------------------
# A. 窗口长度 → 延迟与准确率的权衡
# ---------------------------------------------------------------------------
def window_length_sweep(
    samples: list[dict],
    model: dict,
    lengths: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5, 2.0),
    noise_snr: float = 15.0,
    noise_repeats: int = 3,
    seed: int = 3,
) -> list[dict]:
    """把每段稳态录音切成固定长度窗口，逐窗预测。

    同时测干净与 15 dB 白噪两种条件：干净条件下窗越短越省事，
    但窗越短、参与统计的样本越少，抗噪能力会先掉。
    这两个数字放一起才说得清"实时仪表该用多长的窗"。

    为什么取 15 dB 而不是 10 dB：10 dB 时误判机制是结构性的
    （白噪声整体模仿擦碰谱形），与窗长无关，两条线会一起躺平在 20%，
    反而看不出窗长的影响。15 dB 处在爬坡段，窗长的作用才显出来。
    """
    clf = LogisticRegression.from_dict(model)
    names = model["features"]
    rows = []
    for L in lengths:
        seg_index: list[tuple[str, np.ndarray, int]] = []
        for s in samples:
            x, sr = audio.read_wav(s["path"])
            x = audio.trim_steady(x, sr)
            n = int(L * sr)
            if n < 64:
                continue
            nseg = len(x) // n
            for i in range(nseg):
                seg_index.append((s["label"], x[i * n : (i + 1) * n], sr))
        if not seg_index:
            continue

        def evaluate_condition(snr: float | None, rpt_seed: int) -> tuple[float, float, int]:
            rng = np.random.default_rng(rpt_seed)
            y_true_w, y_pred_w = [], []
            for label, seg, sr in seg_index:
                sig = seg if snr is None else audio.add_noise(seg, snr, "white", rng)
                f = features.extract(sig, sr)
                pred = clf.predict(np.array([[f[k] for k in names]]))[0]
                y_true_w.append(label)
                y_pred_w.append(pred)
            a = evaluate.summarize(y_true_w, y_pred_w, sorted(set(y_true_w)))
            return a["accuracy"], a["macro_f1"], a["n"]

        acc_clean, f1_clean, n_win = evaluate_condition(None, 0)
        noisy = [evaluate_condition(noise_snr, seed + 100 * i) for i in range(noise_repeats)]
        rows.append(
            {
                "window_s": L,
                "n_windows": n_win,
                "acc_clean": round(acc_clean, 4),
                "macro_f1_clean": round(f1_clean, 4),
                "acc_noisy": round(float(np.mean([x[0] for x in noisy])), 4),
                "noise_snr": noise_snr,
                "noise_repeats": noise_repeats,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# B. 少样本
# ---------------------------------------------------------------------------
def few_shot_sweep(
    X: np.ndarray,
    y: list[str],
    names: list[str],
    n_per_class: tuple[int, ...] = (1, 2, 3, 5, 8, 10),
    repeats: int = 30,
    C: float = 0.3,
    seed: int = 11,
) -> list[dict]:
    """每类只用 n 段训练，剩下的测试；重复 repeats 次取均值。"""
    by_class: dict[str, list[int]] = {}
    for i, lab in enumerate(y):
        by_class.setdefault(lab, []).append(i)
    classes = sorted(by_class)
    rows = []
    for n in n_per_class:
        if n >= min(len(v) for v in by_class.values()):
            continue
        accs, f1s = [], []
        for r in range(repeats):
            rng = np.random.default_rng(seed * 977 + r)
            tr, te = [], []
            for c in classes:
                ids = np.array(by_class[c])
                rng.shuffle(ids)
                tr.extend(ids[:n].tolist())
                te.extend(ids[n:].tolist())
            clf = LogisticRegression(C=C, seed=r).fit(X[tr], [y[i] for i in tr])
            s = evaluate.summarize([y[i] for i in te], clf.predict(X[te]), sorted(set(y)))
            accs.append(s["accuracy"])
            f1s.append(s["macro_f1"])
        rows.append(
            {
                "n_per_class": n,
                "n_train": n * len(classes),
                "n_test": len(y) - n * len(classes),
                "acc_mean": round(float(np.mean(accs)), 4),
                "acc_sd": round(float(np.std(accs)), 4),
                "acc_min": round(float(np.min(accs)), 4),
                "macro_f1_mean": round(float(np.mean(f1s)), 4),
                "repeats": repeats,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# C. 未见故障类（异常检测口径）
# ---------------------------------------------------------------------------
def novelty_test(
    X: np.ndarray,
    y: list[str],
    normal_labels: tuple[str, ...] = ("C1_normal_low", "C2_normal_high"),
    anomaly_labels: tuple[str, ...] = ("C3_imbalance", "C4_obstruction", "C5_rubbing"),
    threshold_q: float = 0.95,
) -> dict:
    """用马氏距离给"正常"建模，阈值取正常样本距离的 95 分位；
    然后看三类从未参与建模的故障能不能被判为异常。

    这是诊断仪真正该有的形态：现场绝大多数时候机器是好的，
    我不需要它认出"这是第 3 号故障"，我只需要它说"这台不对劲，去查"。
    """
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd

    normal_idx = [i for i, lab in enumerate(y) if lab in normal_labels]
    Zn = Z[normal_idx]
    cov = np.cov(Zn, rowvar=False)
    cov += np.eye(cov.shape[0]) * 1e-3  # 条件数保护
    inv = np.linalg.pinv(cov)

    def maha(A: np.ndarray) -> np.ndarray:
        D = A - Zn.mean(axis=0)
        return np.sqrt(np.einsum("ij,jk,ik->i", D, inv, D))

    d_normal = maha(Zn)
    thr = float(np.quantile(d_normal, threshold_q))

    per_class = {}
    distance_samples: dict[str, list[float]] = {}
    for lab in anomaly_labels:
        idx = [i for i, v in enumerate(y) if v == lab]
        d = maha(Z[idx])
        distance_samples[lab] = [round(float(x), 3) for x in d]
        per_class[lab] = {
            "n": len(idx),
            "detected": int(np.sum(d > thr)),
            "detect_rate": round(float(np.mean(d > thr)), 4),
            "median_distance": round(float(np.median(d)), 3),
        }

    # 训练集自身的误报率（应该 ≈ 1 - threshold_q）
    false_alarm = float(np.mean(d_normal > thr))
    return {
        "normal_labels": list(normal_labels),
        "threshold_q": threshold_q,
        "threshold_mahalanobis": round(thr, 3),
        "n_normal": len(normal_idx),
        "normal_false_alarm": round(false_alarm, 4),
        "anomalies": per_class,
        "mean_detect_rate": round(float(np.mean([v["detect_rate"] for v in per_class.values()])), 4),
        "distance_samples": {"normal": [round(float(x), 3) for x in d_normal], **distance_samples},
    }


# ---------------------------------------------------------------------------
# D. 增益漂移
# ---------------------------------------------------------------------------
def gain_shift_test(
    samples: list[dict],
    model_level: dict,
    model_nolevel: dict,
    gains_db: tuple[float, ...] = (-24.0, -18.0, -12.0, -6.0, 0.0, 6.0, 12.0),
) -> list[dict]:
    """整体改增益（等价于换手机灵敏度、挪远一点），看两种特征集谁扛得住。

    为什么这是真问题：手机麦克风的自动增益会随环境变化，
    同一台机器在 30 cm 与 3 m 处录到的 dBFS 能差 20 dB 以上。
    如果模型依赖绝对音量，搬个地方就废了。

    实现上刻意**不削波**（浮点直接乘），目的是单独隔离"增益"这一个变量：
    这批录音峰值已经贴近满量程，+12 dB 必然削波，而削波是另一个独立故障，
    混在一起就说不清是哪来的掉点。削波比例单独记在 clipped_fraction 列，
    供读者判断"+12 dB 那一行到底还能不能信"。
    """
    clf_l = LogisticRegression.from_dict(model_level)
    clf_n = LogisticRegression.from_dict(model_nolevel)
    y = [s["label"] for s in samples]
    classes = sorted(set(y))
    rows = []
    for g in gains_db:
        f_l, f_n = [], []
        clip_frac = []
        for s in samples:
            x, sr = audio.read_wav(s["path"])
            x = audio.trim_steady(x, sr)
            y_gain = x * (10 ** (g / 20.0))
            clip_frac.append(float(np.mean(np.abs(y_gain) > 1.0)))
            feats = features.extract(y_gain, sr, include_level=True)
            f_l.append([feats[k] for k in model_level["features"]])
            f_n.append([feats[k] for k in model_nolevel["features"]])
        a = evaluate.summarize(y, clf_l.predict(np.array(f_l)), classes)
        b = evaluate.summarize(y, clf_n.predict(np.array(f_n)), classes)
        rows.append(
            {
                "gain_db": g,
                "acc_with_level": a["accuracy"],
                "acc_no_level": b["accuracy"],
                "clipped_fraction": round(float(np.mean(clip_frac)), 4),
            }
        )
    return rows


def write_csv(path: str, rows: list[dict]) -> None:
    import csv

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
