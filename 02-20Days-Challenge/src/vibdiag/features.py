"""时域 / 频域特征提取。

对外契约只有一个：`extract()` 返回的 dict 键的集合必须恒等于 `FEATURE_ORDER`。
其它脚本（训练、盲测、抗噪）都只认这个契约，不关心内部怎么算。

三类特征，各自的物理含义先说清楚：

  时域幅值/形状   衡量"有多响、有多尖、有多冲"
                  —— 遮挡会让风阻上升、负载变重（RMS 升、峰均比降）；
                     擦碰会引入短促冲击（峰值因子、峭度升）。

  频域分布        衡量"能量堆在哪个频段"
                  —— 失衡把能量推向 BPF 及其低次谐波；
                     擦碰把能量推向 2~8 kHz 的宽带摩擦噪声；
                     遮挡抬高 100~800 Hz 的风流噪声底。

  谐波结构        衡量"旋转有多规律"
                  —— BPF（叶片通过频率）= 转速 × 叶片数。
                     本次实测 BPF 的 1/5 处稳定出现次谐波，据此推定 5 叶、
                     进而反推转速（REPORT 3.1）。这是全项目最有物理含量的一个判据：
                     它把"频谱里最高的那根线"翻译成了"转多快"。

注意：所有幅值类特征都是**绝对值**（dBFS），会随录音增益变化。
`include_level=False` 可把它们置换为与增益无关的形状量，
用于量化"换一台手机还能不能用"——结果见 REPORT 5.2。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import audio

EPS = 1e-20

# 叶片数：由 BPF 次谐波推定（见 derivations.py / REPORT 3.1）
N_BLADES = 5

FEATURE_ORDER: list[str] = [
    # ---- 时域：幅值 ----
    "rms_dbfs",
    "peak_dbfs",
    # ---- 时域：形状 / 冲击性 ----
    "crest_factor",
    "kurtosis",
    "skewness",
    "zcr",
    "shape_factor",
    "impulse_factor",
    "clearance_factor",
    # ---- 时域：包络（调幅） ----
    "env_cv",
    "env_p2p_ratio",
    # ---- 时域：周期性 ----
    "autocorr_peak",
    # ---- 频域：整体分布 ----
    "spec_centroid",
    "spec_spread",
    "spec_rolloff85",
    "spec_flatness",
    "spec_entropy",
    "spec_flux",
    # ---- 频域：转速 / 谐波 ----
    "bpf_hz",
    "bpf_level_db",
    "bpf_prominence",
    "h2_ratio",
    "h3_ratio",
    "harmonic_energy_ratio",
    "shaft_sideband_ratio",
    # ---- 频域：分频段能量占比 ----
    "band_20_100",
    "band_100_300",
    "band_300_800",
    "band_800_2000",
    "band_2000_4000",
    "band_4000_8000",
    "lf_noise_ratio",
    "hf_noise_ratio",
    "tonal_to_total",
]

# 受录音增益影响的特征（include_level=False 时被剔除）
LEVEL_DEPENDENT = {"rms_dbfs", "peak_dbfs", "bpf_level_db"}

# 全频段能量统计范围（避开直流与 8 kHz 以上）
BAND_LO, BAND_HI = 20.0, 8000.0


@dataclass
class Spectrum:
    mag: np.ndarray  # 单边幅度谱（已按 N/2 归一）
    freq: np.ndarray  # 对应频率轴
    sr: int
    n: int

    @property
    def df(self) -> float:
        return self.sr / self.n


def spectrum(x: np.ndarray, sr: int) -> Spectrum:
    """整段 Hann 窗单边幅度谱。

    整段做一次 FFT（而不是 Welch 平均）是有意的：本次每段 3 s、裁剪后 2 s，
    频率分辨率 0.5 Hz，足以分辨 BPF 与其邻谐波。Welch 平滑会把
    叶片数这种"相邻次谐波差 5 Hz"的结构抹掉。
    """
    n = len(x)
    w = np.hanning(n)
    mag = np.abs(np.fft.rfft(x * w)) / (n / 2.0)
    freq = np.fft.rfftfreq(n, 1.0 / sr)
    return Spectrum(mag=mag, freq=freq, sr=sr, n=n)


def _peak_near(sp: Spectrum, f0: float, half_width: float) -> float:
    m = (sp.freq >= f0 - half_width) & (sp.freq <= f0 + half_width)
    if not m.any():
        return 0.0
    return float(sp.mag[m].max())


def _band_energy(sp: Spectrum, lo: float, hi: float) -> float:
    m = (sp.freq >= lo) & (sp.freq < hi)
    return float(np.sum(sp.mag[m] ** 2))


def find_bpf(sp: Spectrum, lo: float = 60.0, hi: float = 400.0) -> float:
    """叶片通过频率：60~400 Hz 内幅度最大的谱线，抛物线插值细化。

    为什么搜索区间是 60~400 Hz：本次这台 180 mm 台扇低档 100 Hz、高档 181 Hz。
    区间留了三倍余量，换一台小台扇/吹风机也不会跑出去；
    若换到手持小风扇（BPF 可能 300+ Hz）只需改这两个默认参数。
    """
    m = (sp.freq >= lo) & (sp.freq <= hi)
    if not m.any():
        return 0.0
    fm, mm = sp.freq[m], sp.mag[m]
    i = int(np.argmax(mm))
    if 0 < i < len(mm) - 1:
        a, b, c = np.log(mm[i - 1] + EPS), np.log(mm[i] + EPS), np.log(mm[i + 1] + EPS)
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
        return float(fm[i] + delta * (fm[1] - fm[0]))
    return float(fm[i])


def extract(
    x: np.ndarray,
    sr: int,
    *,
    include_level: bool = True,
    denoise: bool | tuple[float, float] = False,
) -> dict[str, float]:
    """从一段稳态信号里抽出 FEATURE_ORDER 全部特征。

    denoise:
        False / True  不滤波 / 窄带 80~3500 Hz
        (lo, hi)      指定带通边界，用于消融"消噪带宽砍到多窄"的取舍

    窄带 80~3500 Hz 在干净数据上反而掉点（C5 的 4~8 kHz 指纹被砍掉了），
    这个代价必须在报告里说清楚——见 REPORT 4.3 与图 6。
    """
    if denoise:
        lo, hi = (80.0, 3500.0) if denoise is True else denoise
        x = audio.bandpass_fft(x, sr, lo, hi)

    x = x - float(np.mean(x))
    n = len(x)
    sp = spectrum(x, sr)
    total_energy = _band_energy(sp, BAND_LO, BAND_HI) + EPS

    # ---------------- 时域 ----------------
    rms = float(np.sqrt(np.mean(x**2)) + EPS)
    peak = float(np.max(np.abs(x)) + EPS)
    mean_abs = float(np.mean(np.abs(x)) + EPS)
    sqrt_mean = float(np.mean(np.sqrt(np.abs(x))) ** 2 + EPS)

    sd = float(np.std(x) + EPS)
    z = (x - np.mean(x)) / sd
    kurt = float(np.mean(z**4))
    skew = float(np.mean(z**3))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x).astype(np.int8)))))

    # 包络：1 ms 帧、0.5 ms 步长
    win = max(2, int(0.001 * sr))
    hop = max(1, win // 2)
    env = audio.frame_rms(x, win, hop)
    env_cv = float(np.std(env) / (np.mean(env) + EPS))
    env_p2p = float((env.max() - env.min()) / (env.max() + env.min() + EPS))

    # 归一化自相关峰：在 5~400 Hz 对应的滞后区间里找最大周期性强弱
    # 用 FFT 算自相关（O(n log n)）。直接 np.correlate 是 O(n²)，
    # 2 s × 16 kHz = 32000 点，逐条要跑十几秒，75 条就要十几分钟，不划算。
    lag_lo = max(1, int(sr / 400.0))
    lag_hi = min(n - 2, int(sr / 5.0))
    nfft = 1 << (2 * n - 1).bit_length()
    Xf = np.fft.rfft(x, nfft)
    ac = np.fft.irfft(Xf * np.conj(Xf), nfft)[:n]
    if lag_hi > lag_lo and ac[0] > 0:
        seg = ac[lag_lo:lag_hi] / ac[0]
        autocorr_peak = float(np.max(seg))
    else:
        autocorr_peak = 0.0

    # ---------------- 频域 ----------------
    mag = sp.mag
    freq = sp.freq
    band_mask = (freq >= BAND_LO) & (freq <= BAND_HI)
    mb, fb = mag[band_mask], freq[band_mask]

    p = mb**2
    p_sum = float(np.sum(p) + EPS)
    centroid = float(np.sum(fb * p) / p_sum)
    spread = float(np.sqrt(np.sum(((fb - centroid) ** 2) * p) / p_sum))
    cum = np.cumsum(p)
    rolloff = float(fb[int(np.searchsorted(cum, 0.85 * cum[-1]))]) if cum[-1] > 0 else 0.0

    flatness = float(np.exp(np.mean(np.log(mb + EPS))) / (np.mean(mb) + EPS))
    pn = p / p_sum
    entropy = float(-np.sum(pn * np.log(pn + EPS)) / np.log(len(pn)))  # 归一化到 0~1

    # 谱通量：STFT 相邻帧幅度谱的 L2 变化（衡量时变剧烈程度）
    win_s = max(64, int(0.032 * sr))
    hop_s = win_s // 2
    fr = audio.frame(x, win_s, hop_s)
    win_h = np.hanning(win_s)
    S = np.abs(np.fft.rfft(fr * win_h, axis=1))
    S = S / (np.sum(S, axis=1, keepdims=True) + EPS)
    flux = float(np.mean(np.sqrt(np.sum(np.diff(S, axis=0) ** 2, axis=1)))) if len(S) > 1 else 0.0

    # ---------------- 转速 / 谐波 ----------------
    bpf = find_bpf(sp)
    df = sp.df
    hw = max(df * 1.5, 1.0)
    bpf_level = float(20 * np.log10(_peak_near(sp, bpf, hw) + EPS)) if bpf > 0 else -120.0
    # 局部噪声底：取 BPF 附近 ±(20%~80%) 区间的中位幅度
    if bpf > 0:
        lo_f, hi_f = bpf * 0.2, bpf * 1.8
        near = mag[(freq >= lo_f) & (freq <= hi_f)]
        floor = float(np.median(near)) if len(near) else EPS
        prominence = float(_peak_near(sp, bpf, hw) / (floor + EPS))
    else:
        prominence = 0.0

    def harm_ratio(k: int) -> float:
        if bpf <= 0:
            return 0.0
        a1 = _peak_near(sp, bpf, hw)
        ak = _peak_near(sp, bpf * k, hw * k)
        return float(ak / (a1 + EPS))

    harmonic_energy = 0.0
    if bpf > 0:
        for k in range(1, 7):
            fk = bpf * k
            if fk > BAND_HI:
                break
            m = (freq >= fk - hw * k) & (freq <= fk + hw * k)
            harmonic_energy += float(np.sum(sp.mag[m] ** 2))
    harmonic_ratio = harmonic_energy / total_energy

    # 轴频边带：失衡时会在 BPF 附近出现间隔 = 轴频 (bpf/5) 的边带群
    sideband_energy = 0.0
    if bpf > 0:
        shaft = bpf / N_BLADES
        for k in (1, 2, 3):
            for sign in (-1, 1):
                fs = bpf + sign * shaft * k
                m = (freq >= fs - hw) & (freq <= fs + hw)
                sideband_energy += float(np.sum(sp.mag[m] ** 2))
    sideband_ratio = sideband_energy / total_energy

    # ---------------- 分频段 ----------------
    bands = {
        "band_20_100": (20, 100),
        "band_100_300": (100, 300),
        "band_300_800": (300, 800),
        "band_800_2000": (800, 2000),
        "band_2000_4000": (2000, 4000),
        "band_4000_8000": (4000, 8000),
    }
    feats: dict[str, float] = {}
    for name, (lo, hi) in bands.items():
        feats[name] = _band_energy(sp, lo, hi) / total_energy

    feats["lf_noise_ratio"] = _band_energy(sp, 20, 80) / total_energy
    feats["hf_noise_ratio"] = _band_energy(sp, 2000, 8000) / total_energy

    # 调性占比：能量最大的 60 根谱线占总能量的比例
    top = np.sort(sp.mag[(freq >= BAND_LO)] ** 2)[::-1][:60]
    feats["tonal_to_total"] = float(np.sum(top) / total_energy)

    feats.update(
        rms_dbfs=20 * np.log10(rms),
        peak_dbfs=20 * np.log10(peak),
        crest_factor=peak / rms,
        kurtosis=kurt,
        skewness=skew,
        zcr=zcr,
        shape_factor=rms / mean_abs,
        impulse_factor=peak / mean_abs,
        clearance_factor=peak / sqrt_mean,
        env_cv=env_cv,
        env_p2p_ratio=env_p2p,
        autocorr_peak=autocorr_peak,
        spec_centroid=centroid,
        spec_spread=spread,
        spec_rolloff85=rolloff,
        spec_flatness=flatness,
        spec_entropy=entropy,
        spec_flux=flux,
        bpf_hz=bpf,
        bpf_level_db=bpf_level,
        bpf_prominence=prominence,
        h2_ratio=harm_ratio(2),
        h3_ratio=harm_ratio(3),
        harmonic_energy_ratio=harmonic_ratio,
        shaft_sideband_ratio=sideband_ratio,
    )

    if not include_level:
        feats.pop("rms_dbfs", None)
        feats.pop("peak_dbfs", None)
        feats.pop("bpf_level_db", None)

    return {k: float(feats[k]) for k in FEATURE_ORDER if k in feats}


def feature_names(include_level: bool = True) -> list[str]:
    return [f for f in FEATURE_ORDER if include_level or f not in LEVEL_DEPENDENT]


def to_matrix(rows: list[dict[str, float]], names: list[str]) -> np.ndarray:
    return np.array([[r[n] for n in names] for r in rows], dtype=np.float64)
