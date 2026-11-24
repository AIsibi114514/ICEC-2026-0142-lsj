"""音频读写、裁剪、分帧、加噪、带通。

全部只用标准库 wave + numpy，不依赖 soundfile / librosa / scipy。

关于采样率：本次 75 段录音统一为 16 kHz / 16-bit / 单声道 / 3.000 s。
16 kHz 的奈奎斯特上限是 8 kHz，风扇的擦碰（friction）能量主要在 2~8 kHz，
所以这个采样率刚好够用——但如果要分析 >8 kHz 的气流啸叫，需要 44.1 kHz。
"""

from __future__ import annotations

import wave

import numpy as np

INT16_FULL_SCALE = 32768.0


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """读 16-bit PCM 单声道 WAV，返回 (-1, 1) 浮点数组与采样率。"""
    with wave.open(path, "rb") as w:
        n_ch, sampwidth, sr, n_frames = (
            w.getnchannels(),
            w.getsampwidth(),
            w.getframerate(),
            w.getnframes(),
        )
        raw = w.readframes(n_frames)
    if sampwidth != 2:
        raise ValueError(f"{path}: only 16-bit PCM supported, got {sampwidth * 8}-bit")
    if n_ch != 1:
        raise ValueError(f"{path}: expected mono, got {n_ch} channels")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / INT16_FULL_SCALE
    return x, sr


def write_wav(path: str, x: np.ndarray, sr: int) -> None:
    """写 16-bit PCM 单声道 WAV；超过 ±1 的值会被硬限幅而不是环绕。"""
    y = np.clip(x, -1.0, 1.0)
    data = np.round(y * (INT16_FULL_SCALE - 1)).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())


# --------------------------------------------------------------------------
# 裁剪
# --------------------------------------------------------------------------
def trim_steady(x: np.ndarray, sr: int, head: float = 0.5, tail: float = 0.5) -> np.ndarray:
    """去掉首尾，只保留稳态段。

    为什么必须裁：75 段录音的第 0~150 ms 都有一段明显的淡入（RMS 从 0.01
    爬到 0.2 以上，见 REPORT 第 2.2 节）。那是录音链路的起音包络，不是风扇
    的工况特征。若不裁，RMS、峰值、峰值因子这类幅值特征里会混进 5% 的
    "起音伪影"，而且五个类别的淡入长度略有差异，等于给分类器开了一条捷径。
    """
    h = int(head * sr)
    t = int(tail * sr)
    if len(x) <= h + t:
        return x
    return x[h : len(x) - t]


def frame(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    """分帧，返回 (n_frames, win)。不足一帧的尾巴丢掉。"""
    if len(x) < win:
        return x[np.newaxis, :]
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[np.newaxis, :] + hop * np.arange(n)[:, np.newaxis]
    return x[idx]


def frame_rms(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    """平方和的短时 RMS 包络。"""
    fr = frame(x, win, hop)
    return np.sqrt(np.mean(fr**2, axis=1) + 1e-20)


# --------------------------------------------------------------------------
# 噪声注入（抗噪实验用）
# --------------------------------------------------------------------------
def _white(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n)


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f 粉红噪声：在频域把白噪声按 1/sqrt(f) 整形再逆变换。"""
    w = rng.standard_normal(n)
    spec = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, d=1.0 / n)
    f[0] = f[1] if len(f) > 1 else 1.0
    spec = spec / np.sqrt(f)
    y = np.fft.irfft(spec, n=n)
    return y


def add_noise(
    x: np.ndarray,
    snr_db: float,
    kind: str = "white",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """按目标 SNR 叠加噪声。

    SNR 的定义是 10*log10(P_signal / P_noise)，其中 P 取整段的均方值。
    这意味着 SNR=0 dB 时噪声功率与信号相等——对 3 s 风扇录音来说
    已经是"旁边同时开着另一台风扇"的程度。
    """
    rng = rng or np.random.default_rng(0)
    n = len(x)
    noise = _white(n, rng) if kind == "white" else _pink(n, rng)
    noise = noise - noise.mean()
    p_sig = float(np.mean(x**2)) + 1e-20
    p_noise = float(np.mean(noise**2)) + 1e-20
    scale = np.sqrt(p_sig / p_noise / (10.0 ** (snr_db / 10.0)))
    return x + noise * scale


# --------------------------------------------------------------------------
# 频域带通（抗噪前端的"简易去噪"，FFT 掩模实现）
# --------------------------------------------------------------------------
def bandpass_fft(x: np.ndarray, sr: int, lo: float, hi: float, roll: float = 20.0) -> np.ndarray:
    """零相位 FFT 带通，边缘用余弦过渡带软切，避免砖墙带来的振铃。

    这不是最优滤波器（没有维纳/谱减），只是一个"低保真前端"：
    风扇的有用信息集中在 60~3500 Hz，把之外的宽带噪声削掉就能改善信噪比，
    在 0 dB 白噪下实测把盲测准确率从 4.6% 拉回到 100%（见 REPORT 4.3）。
    """
    n = len(x)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    gain = np.ones_like(f)
    gain[f < lo - roll] = 0.0
    gain[f > hi + roll] = 0.0
    lo_band = (f >= lo - roll) & (f < lo)
    hi_band = (f > hi) & (f <= hi + roll)
    gain[lo_band] = 0.5 * (1 - np.cos(np.pi * (f[lo_band] - (lo - roll)) / roll))
    gain[hi_band] = 0.5 * (1 + np.cos(np.pi * (f[hi_band] - hi) / roll))
    return np.fft.irfft(spec * gain, n=n)
