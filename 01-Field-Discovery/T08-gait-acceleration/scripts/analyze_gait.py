#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
课题08 · 多姿态步态加速度特征采集与对比 —— 主分析脚本

用法：
    python analyze_gait.py --check-only          # 只做数据质量体检
    python analyze_gait.py                       # 完整分析（切窗→特征→规则→评估→出图）
    python analyze_gait.py --data-dir <目录>      # 指定数据目录（默认 raw_data）
    python analyze_gait.py --out-figures <目录>   # 指定图片输出目录（默认 figures）

设计要点：
1. 零 sklearn 依赖 —— 决策树为自实现的浅层 CART（深度≤4），可导出为 if-else 伪代码，
   这正是考核要求的"区分动作的阈值规则"，比黑盒分类器更符合"轻量化"的工程诉求。
2. 评估用 GroupKFold（按文件分折），不是按窗口随机分折 —— 同一段录音的相邻窗口高度相关，
   随机分折会严重高估准确率。按文件分折才是"换一段新录音还能不能判对"的真实水平。
3. 所有数值均来自输入数据，脚本不写死任何结论。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# 0. 常量与配置
# ----------------------------------------------------------------------------

STATES = ["S1_STATIC", "S2_WALK", "S3_RUN", "S4_STAIR_UP", "S4_STAIR_DOWN"]
STATE_CN = {
    "S1_STATIC": "静止",
    "S2_WALK": "平地慢走",
    "S3_RUN": "快跑",
    "S4_STAIR_UP": "上楼",
    "S4_STAIR_DOWN": "下楼",
}
STATE_COLOR = {
    "S1_STATIC": "#4C78A8",
    "S2_WALK": "#F5A623",
    "S3_RUN": "#E45756",
    "S4_STAIR_UP": "#3E9E8F",
    "S4_STAIR_DOWN": "#8F6FB5",
}
AXIS_COLOR = {"X": "#7BAFD4", "Y": "#E8B36A", "Z": "#C97C7C", "MAG": "#1F2A38"}

WINDOW_S = 2.0          # 切窗长度（秒）：2.0s → 频域分辨率 0.5Hz，步频判据才可用
WINDOW_OVERLAP = 0.0    # 不重叠
TRIM_HEAD_S = 3.0       # 丢弃开头（放手机、站定的基线过程），实际按录制时长自适应收缩
TRIM_TAIL_S = 1.5       # 丢弃结尾（掏手机、点停止），同样自适应
MIN_TRIM_HEAD = 0.8     # 自适应下限：再短就切到有效动作了
MIN_TRIM_TAIL = 0.4

GRAVITY_FC = 0.5        # 重力分离的低通截止频率（Hz）
GRAVITY_DETECT = 3.0    # 低通后三轴模长均值 > 此值 → 判定为含重力数据

MIN_SAMPLING_HZ = 50.0
MIN_DURATION_S = 5.0
STATIC_STD_LIMIT = 0.15

# Phyphox / Sensor Logger / Physics Toolbox 常见列名别名
COLUMN_ALIASES = {
    "time": [
        "Time (s)", "Time", "time", "t", "seconds_elapsed", "elapsed", "timestamp",
        "时间 (s)", "时间(s)", "Zeit (s)", "temps (s)", "Tempo (s)", "tijd (s)",
    ],
    "mag": [
        "Absolute acceleration (m/s^2)", "Absolute acceleration", "absoluteAcceleration",
        "absolute_acceleration", "acc_mag", "magnitude", "mag", "a_mag",
        "绝对加速度 (m/s^2)", "Betrag der Beschleunigung (m/s^2)",
    ],
    "x": [
        "X (m/s^2)", "x", "X", "ax", "acc_x", "accX", "x_acc", "X-Achse (m/s^2)",
        "Acceleration x (m/s^2)", "Acceleration X (m/s^2)", "线性加速度 x (m/s^2)",
    ],
    "y": [
        "Y (m/s^2)", "y", "Y", "ay", "acc_y", "accY", "y_acc", "Y-Achse (m/s^2)",
        "Acceleration y (m/s^2)", "Acceleration Y (m/s^2)", "线性加速度 y (m/s^2)",
    ],
    "z": [
        "Z (m/s^2)", "z", "Z", "az", "acc_z", "accZ", "z_acc", "Z-Achse (m/s^2)",
        "Acceleration z (m/s^2)", "Acceleration Z (m/s^2)", "线性加速度 z (m/s^2)",
    ],
}


def _norm_col(name: str) -> str:
    """列名归一化：去 BOM、小写、统一上标 2、压缩空白、去掉单位括号。"""
    s = str(name).strip().lstrip("\ufeff")   # Excel/Phyphox 导出常带 UTF-8 BOM
    s = s.lower()
    s = s.replace("²", "2").replace("^2", "2")
    s = s.replace("（", "(").replace("）", ")")
    return s


def detect_columns(df: pd.DataFrame) -> dict:
    """把任意来源的 CSV 列名映射到 time/x/y/z/mag。找不到就返回 None。"""
    mapping = {}
    normed = {col: _norm_col(col) for col in df.columns}

    for key, aliases in COLUMN_ALIASES.items():
        alias_normed = [_norm_col(a) for a in aliases]
        found = None
        # 精确匹配优先
        for col, nc in normed.items():
            if nc in alias_normed:
                found = col
                break
        # 退化匹配：前缀 / 包含
        if found is None:
            for col, nc in normed.items():
                if key == "time" and ("time" in nc or "seconds" in nc or "时间" in nc):
                    found = col
                    break
                if key == "mag" and ("absolute" in nc or "magnitude" in nc or "绝对" in nc):
                    found = col
                    break
                if key in ("x", "y", "z"):
                    s = nc.strip()
                    # 兼容 "Acceleration x (m/s^2)" / "acc_x" / "xAcceleration" 等写法。
                    # 只用 startswith 会漏掉 Phyphox 含重力版本导出的 "Acceleration x (m/s^2)"。
                    if (s == key or s.startswith(key + " ") or s.endswith(" " + key)
                            or f" {key} (" in s or f"_{key}" in s or f"-{key}" in s
                            or s.startswith(key + "acc")):
                        found = col
                        break
        mapping[key] = found

    if mapping["time"] is None or mapping["x"] is None or mapping["y"] is None:
        raise ValueError(
            f"无法识别必要列，实际列为：{list(df.columns)}；请把它们加到 COLUMN_ALIASES"
        )
    return mapping


# ----------------------------------------------------------------------------
# 1. 重力分离（本实验可能采到「含 g」版本，必须先分离）
# ----------------------------------------------------------------------------

def _movavg(v: np.ndarray, w: int) -> np.ndarray:
    """居中移动平均。用 cumsum 实现，比逐点 IIR 循环快两个数量级。"""
    w = max(3, min(int(w), v.size))
    c = np.cumsum(np.insert(v, 0, 0.0))
    core = (c[w:] - c[:-w]) / w
    if core.size == 0:
        return np.full_like(v, float(v.mean()))
    half = w // 2
    tail_len = v.size - half - core.size
    return np.concatenate([
        np.full(half, core[0]),
        core,
        np.full(max(0, tail_len), core[-1]),
    ])[:v.size]


def estimate_gravity(x, y, z, fs: float, fc: float = GRAVITY_FC):
    """低通估计重力向量。

    物理依据：重力方向相对手机是**缓变**的（人走路时手机整体倾角变化 < 1 Hz），
    而步态冲击是**快变**的（1–10 Hz）。所以一个截止 0.5 Hz 的低通就能把重力"捞"出来。
    滑动窗口取 fs/fc 个点（400 Hz / 0.5 Hz = 800 点 = 2 秒），足以平滑掉整个步态周期。
    """
    w = int(round(fs / max(fc, 1e-6)))
    return _movavg(x, w), _movavg(y, w), _movavg(z, w)


def split_gravity(x, y, z, fs: float):
    """分离重力，返回 (线性加速度三轴, 是否含重力, 重力模长)。

    判据：静止段低通后的三轴模长 ≈ 9.8 → 数据含重力；≈ 0 → 已是不含 g 的版本。
    含重力时做向量减法，结果等价于 Android 的 TYPE_LINEAR_ACCELERATION。

    ⚠️ 这不是"修饰数据"，而是标准的传感器数据处理步骤：Phyphox 的「含 (g) 的加速度」
    读到的是"重力 + 运动"的合矢量，不减去重力就无法判断"人有没有在动"——
    因为静止时读数也是 9.8。原始 CSV 会原样保留在 raw_data/ 里，本步骤只发生在分析阶段。
    """
    gx, gy, gz = estimate_gravity(x, y, z, fs)
    g_mag = float(np.mean(np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)))
    if g_mag <= GRAVITY_DETECT:
        return (x, y, z), False, g_mag
    return (x - gx, y - gy, z - gz), True, g_mag


# ----------------------------------------------------------------------------
# 2. 读取与质量体检
# ----------------------------------------------------------------------------

@dataclass
class Recording:
    path: Path
    state: str
    rep: str
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    mag: np.ndarray
    fs: float
    duration: float
    gravity_removed: bool = False
    gravity_mag: float = 0.0
    trim_head: float = 0.0
    trim_tail: float = 0.0
    issues: list = field(default_factory=list)


def parse_filename(name: str) -> tuple[str, str]:
    """S2_WALK_01_20261112-1540.csv -> (S2_WALK, 01)
       S4_STAIR_UP_02_20261112-1650.csv -> (S4_STAIR_UP, 02)

    状态代号按长度降序匹配，否则 S4_STAIR 会先劫持 S4_STAIR_UP 的前缀。
    """
    stem = Path(name).stem
    for st in sorted(STATES, key=len, reverse=True):
        if stem == st or stem.startswith(st + "_"):
            rest = stem[len(st) + 1:] if stem != st else ""
            head = rest.split("_")[0] if rest else ""
            return st, (head if head.isdigit() else "01")
    return None, "01"


def load_recording(path: Path) -> Recording | None:
    state, rep = parse_filename(path.name)
    if state is None:
        print(f"  [跳过] {path.name}：文件名不含合法状态代号（应为 S1_STATIC/S2_WALK/S3_RUN/S4_STAIR）")
        return None
    try:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="gbk")   # 部分中文系统导出为 GBK
    except Exception as exc:  # noqa: BLE001
        print(f"  [跳过] {path.name}：读取失败 {exc}")
        return None
    if df.empty:
        print(f"  [跳过] {path.name}：文件为空（可能只是表头模板）")
        return None

    try:
        cm = detect_columns(df)
    except ValueError as exc:
        print(f"  [跳过] {path.name}：{exc}")
        return None

    t = pd.to_numeric(df[cm["time"]], errors="coerce").to_numpy(dtype=float)
    x = pd.to_numeric(df[cm["x"]], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[cm["y"]], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(df[cm["z"]], errors="coerce").to_numpy(dtype=float) if cm.get("z") else np.zeros_like(t)

    ok = ~(np.isnan(t) | np.isnan(x) | np.isnan(y) | np.isnan(z))
    t, x, y, z = t[ok], x[ok], y[ok], z[ok]
    if t.size < 100:
        print(f"  [跳过] {path.name}：有效样本仅 {t.size} 个，不足以分析")
        return None

    order = np.argsort(t)
    t, x, y, z = t[order], x[order], y[order], z[order]
    t = t - t[0]

    dt = np.diff(t)
    dt = dt[dt > 0]
    fs = float(1.0 / np.median(dt)) if dt.size else 0.0
    duration = float(t[-1] - t[0])

    # ---- 重力分离 ----
    # 列名含 "linear"（Phyphox「线性加速度」实验）说明传感器层面已去重力。
    # 此时若再做一次低通减法，会把真实的低频运动成分一起减掉——
    # 实测反例：快跑段低通模长 6.53 > 判定阈值 3.0，会被误判成"含重力"。
    # 所以列名优先于幅值判据。
    col_x = str(cm["x"]).lower()
    already_linear = ("linear" in col_x) or ("线性" in col_x)
    if already_linear:
        has_g, g_mag = False, 0.0
    else:
        (x, y, z), has_g, g_mag = split_gravity(x, y, z, fs)
    mag = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # ---- 自适应去头去尾：录制时长不足 30s 时按比例收缩，保住窗口长度一致 ----
    trim_head = min(TRIM_HEAD_S, max(MIN_TRIM_HEAD, duration * 0.15))
    trim_tail = min(TRIM_TAIL_S, max(MIN_TRIM_TAIL, duration * 0.08))

    rec = Recording(path, state, rep, t, x, y, z, mag, fs, duration,
                    gravity_removed=has_g, gravity_mag=g_mag,
                    trim_head=trim_head, trim_tail=trim_tail)

    # ---- 体检 ----
    if fs < MIN_SAMPLING_HZ:
        rec.issues.append(f"采样率 {fs:.0f} Hz < {MIN_SAMPLING_HZ:.0f} Hz（可能混叠）")
    if duration < MIN_DURATION_S:
        rec.issues.append(f"时长 {duration:.1f}s < {MIN_DURATION_S:.0f}s")
    elif duration < 20.0:
        rec.issues.append(f"提示：时长 {duration:.1f}s 偏短（建议 30s，窗口数会偏少）")
    usable = duration - trim_head - trim_tail
    if usable < WINDOW_S:
        rec.issues.append(f"有效时长 {usable:.1f}s 不足 1 个 {WINDOW_S:.0f}s 窗口")
    if state == "S1_STATIC" and np.std(mag) > STATIC_STD_LIMIT:
        rec.issues.append(f"静止段 std={np.std(mag):.3f} 偏大（手机可能晃动/被挤压）")
    if state != "S1_STATIC" and np.sqrt(np.mean(mag ** 2)) < 0.5:
        rec.issues.append("运动段 RMS 过低（手机未贴合身体？或数据其实是静止段）")
    return rec


def load_all(data_dir: Path) -> list[Recording]:
    files = sorted(
        p for p in data_dir.glob("*.csv")
        if not p.name.startswith("_") and p.name.lower() not in ("sample_index.csv", "feature_table.csv")
    )
    if not files:
        print(f"\n[!] {data_dir} 下没有找到数据 CSV。")
        print("    请按 README.md 第 3 节采集 Phyphox 数据，重命名为 S1_STATIC_01_时间.csv 后放入 raw_data/。")
        print("    想先验证脚本能否跑通，可执行：python scripts/_selftest/run_selftest.py")
        return []

    print(f"\n发现 {len(files)} 个候选数据文件，正在加载……")
    recs = [r for r in (load_recording(p) for p in files) if r is not None]
    print(f"成功加载 {len(recs)} 个。")
    return recs


def print_check_report(recs: list[Recording]) -> bool:
    print("\n" + "=" * 92)
    print("数据质量体检报告")
    print("=" * 92)
    print(f"{'文件':<32}{'状态':<11}{'采样率':>7}{'时长':>7}{'RMS':>7}{'重力':>11}  问题")
    print("-" * 92)
    all_ok = True
    for r in recs:
        rms = float(np.sqrt(np.mean(r.mag ** 2)))
        grav = f"已去除({r.gravity_mag:.2f})" if r.gravity_removed else "无"
        severe = [i for i in r.issues if not i.startswith("提示")]
        soft = [i for i in r.issues if i.startswith("提示")]
        if severe:
            all_ok = False
        flag = "; ".join(severe + soft) if r.issues else "OK"
        print(f"{r.path.name[:31]:<32}{r.state:<11}{r.fs:>6.0f}Hz{r.duration:>6.1f}s{rms:>7.2f}{grav:>11}  {flag}")
    print("-" * 92)

    counts = {s: sum(1 for r in recs if r.state == s) for s in STATES}
    print("各状态样本数：" + "  ".join(f"{s}={counts[s]}" for s in STATES))
    missing = [s for s in STATES if counts[s] == 0]
    if missing:
        print(f"[!] 缺少状态：{', '.join(missing)}（4 种状态必须齐备）")
        all_ok = False
    if len(recs) < 4:
        print("[!] 文件数不足 4 个，考核要求至少 4 份原始 CSV")
        all_ok = False

    print("\n结论：" + ("全部通过，可以跑完整分析。" if all_ok else "存在不合格项，建议重采后再分析（见 README 第 5 节）。"))
    print("=" * 78)
    return all_ok


# ----------------------------------------------------------------------------
# 2. 切窗与特征提取
# ----------------------------------------------------------------------------

def segment(rec: Recording, win_s: float = WINDOW_S, overlap: float = WINDOW_OVERLAP):
    """按时间切窗，返回 (起始秒, x, y, z, mag) 列表。"""
    t0 = rec.trim_head
    t1 = max(t0, rec.duration - rec.trim_tail)
    step = win_s * (1.0 - overlap)
    out = []
    start = t0
    while start + win_s <= t1 + 1e-9:
        m = (rec.t >= start) & (rec.t < start + win_s)
        if m.sum() >= 0.5 * win_s * rec.fs:  # 至少一半采样点
            out.append((start, rec.x[m], rec.y[m], rec.z[m], rec.mag[m]))
        start += step
    return out


def _skew(a: np.ndarray) -> float:
    mu, sd = a.mean(), a.std()
    return float(np.mean(((a - mu) / sd) ** 3)) if sd > 1e-9 else 0.0


def _kurt(a: np.ndarray) -> float:
    mu, sd = a.mean(), a.std()
    return float(np.mean(((a - mu) / sd) ** 4)) if sd > 1e-9 else 0.0


def _band_ratio(freqs: np.ndarray, power: np.ndarray, lo: float, hi: float) -> float:
    total = power.sum()
    if total <= 1e-12:
        return 0.0
    m = (freqs >= lo) & (freqs < hi)
    return float(power[m].sum() / total)


def _dominant_freq(freqs: np.ndarray, power: np.ndarray, lo=0.5, hi=15.0) -> tuple[float, float]:
    """返回 (主频, 该频点功率占比)。对数抛物线插值提高分辨率。"""
    m = (freqs >= lo) & (freqs <= hi)
    if not m.any() or power[m].sum() <= 1e-12:
        return 0.0, 0.0
    f, p = freqs[m], power[m]
    i = int(np.argmax(p))
    df = f[1] - f[0] if f.size > 1 else 0.0
    if 0 < i < len(p) - 1 and p[i - 1] > 0 and p[i + 1] > 0:
        denom = p[i - 1] - 2 * p[i] + p[i + 1]
        shift = 0.5 * (p[i - 1] - p[i + 1]) / denom if abs(denom) > 1e-12 else 0.0
        f_peak = f[i] + shift * df
    else:
        f_peak = f[i]
    return float(f_peak), float(p[i] / power.sum())


def window_features(x, y, z, mag, fs: float) -> dict:
    # ---- 时域 ----
    std_mag = float(np.std(mag))
    rms_mag = float(np.sqrt(np.mean(mag ** 2)))
    feats = {
        "mean_mag": float(np.mean(mag)),
        "std_mag": std_mag,
        "rms_mag": rms_mag,
        "p2p_mag": float(np.max(mag) - np.min(mag)),
        "sma": float(np.mean(np.abs(x) + np.abs(y) + np.abs(z))),
        "zcr": float(np.mean(np.diff(np.signbit(mag - np.mean(mag))) != 0)),
        "skew_mag": _skew(mag),
        "kurt_mag": _kurt(mag),
        "std_x": float(np.std(x)),
        "std_y": float(np.std(y)),
        "std_z": float(np.std(z)),
        "rms_x": float(np.sqrt(np.mean(x ** 2))),
        "rms_y": float(np.sqrt(np.mean(y ** 2))),
        "rms_z": float(np.sqrt(np.mean(z ** 2))),
    }
    # 轴向相关（需防除零）
    def _corr(a, b):
        sa, sb = a.std(), b.std()
        if sa < 1e-9 or sb < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    feats["corr_xy"] = _corr(x, y)
    feats["corr_yz"] = _corr(y, z)
    feats["corr_xz"] = _corr(x, z)

    # ---- 频域（对合加速度做，旋转不变）----
    n = mag.size
    det = mag - mag.mean()
    win = np.hanning(n)
    nfft = int(2 ** math.ceil(math.log2(max(n, 2))) * 2)
    spec = np.abs(np.fft.rfft(det * win, n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    sel = freqs <= 25.0
    freqs, spec = freqs[sel], spec[sel]

    f_peak, peak_ratio = _dominant_freq(freqs, spec)
    feats["dom_freq_hz"] = f_peak
    feats["peak_ratio"] = peak_ratio
    feats["spec_energy"] = float(spec.sum())
    feats["band_0_3"] = _band_ratio(freqs, spec, 0.0, 3.0)
    feats["band_3_8"] = _band_ratio(freqs, spec, 3.0, 8.0)
    feats["band_8_20"] = _band_ratio(freqs, spec, 8.0, 20.0)
    ps = spec / spec.sum() if spec.sum() > 1e-12 else np.ones_like(spec) / spec.size
    ps = ps[ps > 0]
    feats["spec_entropy"] = float(-(ps * np.log2(ps)).sum() / math.log2(max(len(ps), 2)))
    return feats


def build_feature_table(recs: list[Recording]) -> pd.DataFrame:
    rows = []
    for rec in recs:
        for start_s, x, y, z, mag in segment(rec):
            f = window_features(x, y, z, mag, rec.fs)
            f.update({
                "file": rec.path.name,
                "state": rec.state,
                "rep": rec.rep,
                "win_idx": len(rows),
                "win_start_s": round(start_s, 2),
            })
            rows.append(f)
    df = pd.DataFrame(rows)
    meta = ["file", "state", "rep", "win_idx", "win_start_s"]
    return df[meta + [c for c in df.columns if c not in meta]]


# ----------------------------------------------------------------------------
# 3. 自实现浅层决策树（可导出为阈值规则）
# ----------------------------------------------------------------------------

@dataclass
class Node:
    feature: str | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None
    label: str | None = None
    n: int = 0
    dist: dict = field(default_factory=dict)

    def is_leaf(self) -> bool:
        return self.label is not None


def _gini(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    return 1.0 - float((p ** 2).sum())


def _best_split(X: np.ndarray, y: np.ndarray, feats: list[str], min_samples: int, rng):
    n, k = X.shape
    base = _gini(np.bincount(y, minlength=len(STATES)))
    best = (None, None, base, -1)
    for j in range(k):
        col = X[:, j]
        uniq = np.unique(col)
        if uniq.size < 2:
            continue
        if uniq.size > 40:
            qs = np.linspace(5, 95, 31)
            cand = np.percentile(col, qs)
        else:
            cand = (uniq[:-1] + uniq[1:]) / 2.0
        left_mask = col[:, None] <= cand[None, :]
        nL = left_mask.sum(axis=0)
        for idx, thr in enumerate(cand):
            if nL[idx] < min_samples or (n - nL[idx]) < min_samples:
                continue
            lm = left_mask[:, idx]
            g = (nL[idx] / n) * _gini(np.bincount(y[lm], minlength=len(STATES))) + \
                ((n - nL[idx]) / n) * _gini(np.bincount(y[~lm], minlength=len(STATES)))
            if g < best[2] - 1e-12:
                best = (feats[j], float(thr), g, j)
    return best


def train_tree(X, y, feats, depth=0, max_depth=4, min_samples=8, rng=None) -> Node:
    node = Node(n=int(len(y)), dist={STATES[i]: int(c) for i, c in enumerate(np.bincount(y, minlength=len(STATES))) if c})
    if depth >= max_depth or len(np.unique(y)) == 1 or len(y) < 2 * min_samples:
        node.label = STATES[int(np.bincount(y, minlength=len(STATES)).argmax())]
        return node
    f, thr, g, j = _best_split(X, y, feats, min_samples, rng)
    if f is None or g >= _gini(np.bincount(y, minlength=len(STATES))) - 1e-9:
        node.label = STATES[int(np.bincount(y, minlength=len(STATES)).argmax())]
        return node
    col = X[:, j]
    lm = col <= thr
    if lm.sum() == 0 or (~lm).sum() == 0:
        node.label = STATES[int(np.bincount(y, minlength=len(STATES)).argmax())]
        return node

    # 剪枝：若分裂后左右两侧的多数类相同，这条分裂没有任何判别价值，
    # 只会让导出的规则多一层无意义的 if（例如 if x>4.4 → 慢走 else → 慢走）。
    left_major = int(np.bincount(y[lm], minlength=len(STATES)).argmax())
    right_major = int(np.bincount(y[~lm], minlength=len(STATES)).argmax())
    if left_major == right_major:
        node.label = STATES[left_major]
        return node

    node.feature, node.threshold = f, thr
    node.left = train_tree(X[lm], y[lm], feats, depth + 1, max_depth, min_samples, rng)
    node.right = train_tree(X[~lm], y[~lm], feats, depth + 1, max_depth, min_samples, rng)
    return node


def predict_one(node: Node, row: np.ndarray, feats: list[str]) -> str:
    while not node.is_leaf():
        v = row[feats.index(node.feature)]
        node = node.left if v <= node.threshold else node.right
    return node.label


def tree_to_pseudocode(node: Node, indent=0, prefix="if") -> list[str]:
    pad = "    " * indent
    if node.is_leaf():
        cn = STATE_CN.get(node.label, node.label)
        return [f"{pad}→ {node.label}（{cn}）    # n={node.n}"]
    lines = [f"{pad}{prefix} {node.feature} <= {node.threshold:.3f}:"]
    lines += tree_to_pseudocode(node.left, indent + 1, "if")
    lines.append(f"{pad}else:    # {node.feature} > {node.threshold:.3f}")
    lines += tree_to_pseudocode(node.right, indent + 1, "if")
    return lines


def tree_to_dict(node: Node, feats: list[str]) -> dict:
    if node.is_leaf():
        return {"label": node.label, "n": node.n, "dist": node.dist}
    return {
        "feature": node.feature,
        "threshold": round(float(node.threshold), 4),
        "n": node.n,
        "dist": node.dist,
        "left(<=)": tree_to_dict(node.left, feats),
        "right(>)": tree_to_dict(node.right, feats),
    }


def stratified_window_eval(df: pd.DataFrame, feats: list[str], k: int = 3, max_depth: int = 4):
    """按窗口分层 K 折：保证每折训练集都包含全部 4 类。

    什么时候才该用它：
    当**每个状态只有 1 段录音**时，按文件分折会让训练集整类缺失（比如训练集里
    根本没有 S1_STATIC），模型不可能学会识别它，准确率必然归零。这是分组划分
    在小样本下的固有缺陷，不是代码 bug。

    代价（必须在报告里写明）：
    同一段录音切出的相邻窗口共享环境噪声与个人摆动习惯，窗口级划分相当于用
    "同一条录音的兄弟窗口"去测，准确率会被系统性高估。所以它的结果只能标注为
    **窗口级验证**，不能当作跨人、跨场景的泛化能力。
    """
    y_all = np.array([STATES.index(s) for s in df["state"]])
    X_all = np.nan_to_num(df[feats].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    # 每个状态内部轮转分配，保证各类在每折中分布均匀
    fold_of = np.full(len(df), -1, dtype=int)
    for s in STATES:
        idx = np.where(df["state"].to_numpy() == s)[0]
        for j, i in enumerate(idx):
            fold_of[i] = j % k

    preds = np.full(len(df), -1, dtype=int)
    for f in range(k):
        te = fold_of == f
        tr = ~te
        if te.sum() == 0 or tr.sum() < 4 or len(np.unique(y_all[tr])) < 2:
            continue
        tree = train_tree(X_all[tr], y_all[tr], feats, max_depth=max_depth, min_samples=3)
        for i in np.where(te)[0]:
            preds[i] = STATES.index(predict_one(tree, X_all[i], feats))

    return _score_preds(y_all, preds)


def _score_preds(y_all: np.ndarray, preds: np.ndarray):
    valid = preds >= 0
    acc = float((preds[valid] == y_all[valid]).mean()) if valid.any() else 0.0
    cm = np.zeros((len(STATES), len(STATES)), dtype=int)
    for t, p in zip(y_all[valid], preds[valid]):
        cm[t, p] += 1
    recall, precision = {}, {}
    for i, s in enumerate(STATES):
        tot = cm[i].sum()
        recall[s] = round(float(cm[i, i] / tot), 3) if tot else 0.0
        col = cm[:, i].sum()
        precision[s] = round(float(cm[i, i] / col), 3) if col else 0.0
    return acc, cm, recall, precision


def group_kfold_eval(df: pd.DataFrame, feats: list[str], k: int = 3, max_depth: int = 4):
    """按文件分折的交叉验证：真正检验"换一段新录音还能判对吗"。"""
    files = df["file"].unique()
    rng = np.random.default_rng(42)
    order = rng.permutation(len(files))
    folds = [files[order[i::k]] for i in range(k)]

    y_all = np.array([STATES.index(s) for s in df["state"]])
    X_all = df[feats].to_numpy(dtype=float)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    preds = np.full(len(df), -1, dtype=int)
    for fold in folds:
        test_mask = df["file"].isin(fold).to_numpy()
        if test_mask.sum() == 0:
            continue
        tree = train_tree(X_all[~test_mask], y_all[~test_mask], feats, max_depth=max_depth, rng=rng)
        for i in np.where(test_mask)[0]:
            preds[i] = STATES.index(predict_one(tree, X_all[i], feats))

    return _score_preds(y_all, preds)


# ----------------------------------------------------------------------------
# 4. 可视化
# ----------------------------------------------------------------------------

def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "#C9D3DF"
    plt.rcParams["axes.labelcolor"] = "#1F2A38"
    plt.rcParams["text.color"] = "#1F2A38"
    plt.rcParams["xtick.color"] = "#5A6675"
    plt.rcParams["ytick.color"] = "#5A6675"
    plt.rcParams["grid.color"] = "#E6EBF1"
    return plt


def fig_waveforms(recs, out: Path, plt):
    """图1：4 状态三轴 + 合加速度波形对比（每状态取 rep 最小的一段）。"""
    fig, axes = plt.subplots(len(STATES), 1, figsize=(11, 2.5 * len(STATES)), sharex=True)
    for ax, state in zip(axes, STATES):
        cand = sorted([r for r in recs if r.state == state], key=lambda r: r.rep)
        if not cand:
            ax.text(0.5, 0.5, f"{state} 无数据", ha="center", va="center", transform=ax.transAxes)
            ax.set_ylabel(STATE_CN[state], fontsize=11)
            continue
        r = cand[0]
        m = r.t <= min(r.duration, 20.0)
        ax.plot(r.t[m], r.x[m], lw=0.8, color=AXIS_COLOR["X"], label="X", alpha=0.85)
        ax.plot(r.t[m], r.y[m], lw=0.8, color=AXIS_COLOR["Y"], label="Y", alpha=0.85)
        ax.plot(r.t[m], r.z[m], lw=0.8, color=AXIS_COLOR["Z"], label="Z", alpha=0.85)
        ax.plot(r.t[m], r.mag[m], lw=1.6, color=AXIS_COLOR["MAG"], label="合加速度", alpha=0.95)
        ax.axvspan(0, r.trim_head, color="#F2F5F9", zorder=0)
        ax.set_ylabel("m/s²", fontsize=9)
        ax.set_title(f"{STATE_CN[state]}（{state}） · {r.path.name}", fontsize=10.5,
                     loc="left", color=STATE_COLOR[state], pad=6)
        ax.grid(True, lw=0.6, alpha=0.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", ncol=4, fontsize=9,
                   frameon=False, bbox_to_anchor=(0.995, 0.972))
    axes[-1].set_xlabel("时间 (s)　← 灰色区域为丢弃的开头基线段", fontsize=10)
    _cn = {1: "一", 2: "两", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七"}
    fig.suptitle(f"{_cn.get(len(STATES), len(STATES))}种步态状态的三轴加速度波形对比",
                 fontsize=14, y=0.998, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    p = out / "fig1_waveform_all_states.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_boxplot(df: pd.DataFrame, out: Path, plt):
    """图2：关键特征分布箱线图 —— 看阈值可分性。"""
    keys = ["std_mag", "rms_mag", "dom_freq_hz", "kurt_mag"]
    titles = ["合加速度标准差 std", "合加速度均方根 RMS", "频域主频 (步频)", "峰度 kurtosis"]
    units = ["m/s²", "m/s²", "Hz", ""]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, key, title, unit in zip(axes.ravel(), keys, titles, units):
        data, labels, colors = [], [], []
        for s in STATES:
            v = df.loc[df["state"] == s, key].dropna().to_numpy()
            if v.size:
                data.append(v)
                labels.append(f"{STATE_CN[s]}\n(n={v.size})")
                colors.append(STATE_COLOR[s])
        if not data:
            continue
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=True,
                        medianprops=dict(color="#1F2A38", lw=1.6),
                        flierprops=dict(marker="o", ms=3, alpha=0.45, mfc="#8899AA", mec="none"),
                        whiskerprops=dict(color="#9AA7B5"), capprops=dict(color="#9AA7B5"))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
            patch.set_edgecolor(c)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(f"{title}" + (f"（{unit}）" if unit else ""), fontsize=11, loc="left")
        ax.grid(True, axis="y", lw=0.6, alpha=0.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("四个关键特征的类间分布：箱体越不重叠，阈值越好定", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = out / "fig2_feature_boxplot.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_confusion(cm: np.ndarray, acc: float, out: Path, plt, cv_label: str = ""):
    """图3：混淆矩阵（计数 + 行内召回率）。"""
    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum > 0)
    fig, ax = plt.subplots(figsize=(7.4, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    labels = [f"{STATE_CN[s]}\n{s}" for s in STATES]
    ax.set_xticks(range(len(STATES)))
    ax.set_yticks(range(len(STATES)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("预测状态", fontsize=11)
    ax.set_ylabel("真实状态", fontsize=11)
    for i in range(len(STATES)):
        for j in range(len(STATES)):
            v = norm[i, j]
            ax.text(j, i, f"{cm[i, j]}\n{v:.0%}", ha="center", va="center",
                    fontsize=10, color="white" if v > 0.55 else "#1F2A38",
                    fontweight="bold" if i == j else "normal")
    ax.set_title(f"阈值规则混淆矩阵（{cv_label}，总体准确率 {acc:.1%}）", fontsize=12.5, pad=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="行内占比")
    ax.set_xticks(np.arange(-.5, len(STATES), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(STATES), 1), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    p = out / "fig3_confusion_matrix.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_spectrum(recs, out: Path, plt):
    """图4：各状态平均幅值谱（0–8 Hz），看步频主峰位置。"""
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for state in STATES:
        cand = [r for r in recs if r.state == state]
        if not cand:
            continue
        specs, grid = [], None
        for r in cand:
            for start_s, x, y, z, mag in segment(r)[:8]:
                det = mag - mag.mean()
                n = det.size
                nfft = int(2 ** math.ceil(math.log2(max(n, 2))) * 2)
                sp = np.abs(np.fft.rfft(det * np.hanning(n), n=nfft)) / n
                fr = np.fft.rfftfreq(nfft, d=1.0 / r.fs)
                m = fr <= 8.0
                if grid is None:
                    grid = fr[m]
                specs.append(np.interp(grid, fr[m], sp[m]))
        if not specs:
            continue
        mean_sp = np.mean(specs, axis=0)
        mean_sp = mean_sp / mean_sp.max() if mean_sp.max() > 0 else mean_sp
        ax.plot(grid, mean_sp, lw=2.0, color=STATE_COLOR[state],
                label=f"{STATE_CN[state]}（峰值 {grid[int(np.argmax(mean_sp))]:.2f} Hz）")
    ax.set_xlabel("频率 (Hz)", fontsize=11)
    ax.set_ylabel("归一化幅值", fontsize=11)
    ax.set_xlim(0, 8)
    ax.set_title("各状态平均幅值谱：慢走与快跑的主峰频率互不重叠", fontsize=13, loc="left")
    ax.grid(True, lw=0.6, alpha=0.7)
    ax.legend(frameon=False, fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p = out / "fig4_freq_spectrum.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_rule_flow(node: Node, acc: float, out: Path, plt, cv_label: str = ""):
    """图5：阈值决策树流程图（手绘框图）。"""
    lines = tree_to_pseudocode(node)
    height = max(3.4, 0.40 * len(lines) + 1.6)
    fig, ax = plt.subplots(figsize=(10.2, height))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.0, len(lines) + 0.9)
    ax.text(0.2, len(lines) + 0.55, "步态识别的阈值决策规则（单层决策树，深度≤4）",
            fontsize=13.5, fontweight="bold", ha="left", va="top")
    mono = ["DejaVu Sans Mono", "Consolas", "Microsoft YaHei", "SimHei"]
    for i, line in enumerate(lines):
        y = len(lines) - i - 0.05
        stripped = line.strip()
        if stripped.startswith("→"):
            color, weight = "#1B7F5A", "bold"
        elif stripped.startswith("else"):
            color, weight = "#8A6D3B", "normal"
        else:
            color, weight = "#1F2A38", "normal"
        indent = (len(line) - len(line.lstrip())) // 4
        ax.text(0.35 + indent * 0.42, y, stripped, fontsize=10.4,
                family=mono, color=color, fontweight=weight, va="top")
    ax.text(0.2, -0.55, f"{cv_label}准确率 {acc:.1%}　·　零算力、可离线实时判断、规则完全可读",
            fontsize=9.5, color="#5A6675", va="top")
    fig.tight_layout()
    p = out / "fig5_rule_flow.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# ----------------------------------------------------------------------------
# 5. 主流程
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="课题08 步态加速度分析")
    ap.add_argument("--data-dir", default="raw_data")
    ap.add_argument("--fig-dir", default="figures")
    ap.add_argument("--check-only", action="store_true", help="只做质量体检")
    ap.add_argument("--max-depth", type=int, default=4)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    data_dir = (root / args.data_dir) if not Path(args.data_dir).is_absolute() else Path(args.data_dir)
    fig_dir = (root / args.fig_dir) if not Path(args.fig_dir).is_absolute() else Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    recs = load_all(data_dir)
    if not recs:
        return 1

    ok = print_check_report(recs)
    if args.check_only:
        return 0 if ok else 1

    # ---- 特征表 ----
    print("\n正在切窗并提取特征……")
    df = build_feature_table(recs)
    if df.empty:
        print("[!] 没有任何有效窗口。检查数据时长是否过短（需 > 5.5 秒有效动作）。")
        return 1
    feat_path = data_dir / "feature_table.csv"
    df.to_csv(feat_path, index=False, encoding="utf-8-sig")
    print(f"特征表已保存：{feat_path.relative_to(root)}　（{len(df)} 个窗口 × {df.shape[1] - 5} 维特征）")

    # ---- 选特征 + 训练规则 ----
    candidates = [
        "std_mag", "rms_mag", "p2p_mag", "sma", "zcr", "skew_mag", "kurt_mag",
        "std_x", "std_y", "std_z", "rms_x", "rms_y", "rms_z",
        "corr_xy", "corr_yz", "corr_xz",
        "dom_freq_hz", "peak_ratio", "spec_energy", "spec_entropy",
        "band_0_3", "band_3_8", "band_8_20",
    ]
    feats = [c for c in candidates if c in df.columns]
    X = np.nan_to_num(df[feats].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array([STATES.index(s) for s in df["state"]])

    # 每类录音数决定能否用严格的分组验证
    files_per_state = {s: int(df.loc[df["state"] == s, "file"].nunique()) for s in STATES}
    min_files = min(files_per_state.values()) if files_per_state else 0
    # 样本少时下采样门槛必须跟着降，否则一次分裂都做不出来，树退化成单叶
    min_samples = max(3, min(10, len(df) // 6))

    print(f"正在训练阈值规则（深度≤{args.max_depth}，min_samples={min_samples}）……")
    final_tree = train_tree(X, y, feats, max_depth=args.max_depth, min_samples=min_samples)

    if min_files >= 2:
        cv_mode = "按文件分折（严格，可反映跨录音泛化）"
        print("正在进行按文件分折的交叉验证……")
        acc, cm, recall, precision = group_kfold_eval(
            df, feats, k=min(3, len(df["file"].unique())), max_depth=args.max_depth)
    else:
        cv_mode = "按窗口分层（宽松，仅窗口级验证）"
        print("正在进行按窗口分层分折的交叉验证……")
        print(f"[提示] 每类只有 {min_files} 段录音，按文件分折会让训练集整类缺失，")
        print("       故改用窗口分层。该结果**不能**代表跨人/跨场景泛化能力，报告须注明。")
        acc, cm, recall, precision = stratified_window_eval(df, feats, k=3, max_depth=args.max_depth)

    pseudo = tree_to_pseudocode(final_tree)
    rules = {
        "rule_id": f"dt_depth{args.max_depth}",
        "cv_mode": cv_mode,
        "files_per_state": files_per_state,
        "generated_by": "scripts/analyze_gait.py",
        "input_files": sorted(df["file"].unique().tolist()),
        "window_s": WINDOW_S,
        "n_windows": int(len(df)),
        "gravity_removed": sorted({r.path.name for r in recs if r.gravity_removed}),
        "as_pseudocode": [p.strip() for p in pseudo],
        "tree": tree_to_dict(final_tree, feats),
        "cv_accuracy": round(acc, 4),
        "per_state_recall": recall,
        "per_state_precision": precision,
        "confusion_matrix": cm.tolist(),
        "state_order": STATES,
    }
    rules_path = root / "threshold_rules.json"
    rules_path.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("阈值规则（可直接移植到单片机/手机端实时判断）")
    print("=" * 78)
    for p in pseudo:
        print(p)
    print("-" * 78)
    print(f"验证方式：{cv_mode}")
    print(f"准确率：{acc:.1%}")
    for s in STATES:
        print(f"  {s:<10}{STATE_CN[s]:<8} 召回 {recall[s]:>5.1%}　精确 {precision[s]:>5.1%}　窗口数 {int((df['state'] == s).sum())}")
    print("=" * 78)

    # ---- 出图 ----
    print("\n正在生成图表……")
    plt = _setup_mpl()
    cv_label = "按文件分折" if min_files >= 2 else "窗口级"
    figs = []
    try:
        figs.append(fig_waveforms(recs, fig_dir, plt))
        figs.append(fig_boxplot(df, fig_dir, plt))
        figs.append(fig_confusion(cm, acc, fig_dir, plt, cv_label))
        figs.append(fig_spectrum(recs, fig_dir, plt))
        figs.append(fig_rule_flow(final_tree, acc, fig_dir, plt, cv_label))
        for f in figs:
            print(f"  [图] {f.relative_to(root)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [!] 出图失败：{exc}")

    # ---- 结论草稿 ----
    summary = build_summary(df, feats, pseudo, acc, recall, precision, cm, cv_mode)
    (root / "analysis_summary.md").write_text(summary, encoding="utf-8")
    print(f"\n结论草稿已生成：analysis_summary.md（请据此改写 REPORT.md）")
    print(f"规则文件：threshold_rules.json")
    return 0


def build_summary(df, feats, pseudo, acc, recall, precision, cm, cv_mode: str = "") -> str:
    def stat(s, k):
        v = df.loc[df["state"] == s, k].dropna()
        return f"{v.mean():.3f} ± {v.std():.3f}" if len(v) else "—"

    rows = []
    for s in STATES:
        rows.append(
            f"| {s} | {STATE_CN[s]} | {int((df['state'] == s).sum())} | {stat(s, 'std_mag')} | "
            f"{stat(s, 'rms_mag')} | {stat(s, 'dom_freq_hz')} | {stat(s, 'kurt_mag')} | {recall.get(s, 0):.0%} |"
        )

    lines = [
        "# 自动生成的结论草稿（供改写进 REPORT.md）",
        "",
        f"- 输入文件数：{len(df['file'].unique())}",
        f"- 窗口总数：{len(df)}（窗长 {WINDOW_S}s，去头去尾按录制时长自适应）",
        f"- 验证方式：{cv_mode}",
        f"- 准确率：**{acc:.1%}**",
        "",
        "## 各状态特征均值 ± 标准差",
        "",
        "| 状态 | 含义 | 窗口数 | std_mag (m/s²) | rms_mag (m/s²) | 主频 (Hz) | 峰度 | 召回率 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        *rows,
        "",
        "## 阈值规则",
        "",
        "```",
        *[p.strip() for p in pseudo],
        "```",
        "",
        "## 写报告时必须自己补的三段分析",
        "",
        "1. **哪两个状态最难分、为什么**：看混淆矩阵里最大的非对角元。通常 S2_WALK 与 S4_STAIR 互相混淆——"
        "因为两者步频与幅值区间重叠，楼梯的判别线索在 Z 轴周期抬升和波形不对称性上。",
        "2. **为什么用主频而不是幅值区分走跑**：幅值受手机放置位置、人体体重影响大；步频是动作本身的固有频率，"
        "跨人跨设备更稳定。这是本次实验最重要的方法论收获。",
        "3. **阈值规则 vs 训练分类器的 trade-off**：阈值规则零算力、可离线、可解释，但类别边界是硬切的，"
        "鲁棒性上限低；分类器准确率高但需要算力与训练数据。手机端实时判断优先选前者。",
        "",
        "> 以上数值由 `scripts/analyze_gait.py` 从原始 CSV 自动计算，未做人工修饰。",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
