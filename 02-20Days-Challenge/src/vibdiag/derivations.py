"""从声音反推机械参数：叶片数、转速、谐波衰减斜率。

这是本项目里唯一"用物理反推实验设置"的模块，也是最能说明
"频谱不只是分类的输入，它本身就是测量仪表"的地方。

原理：
    叶片通过频率 BPF = 转速(Hz) × 叶片数 N
    频谱里最高的那根线通常就是 BPF，但"最高"告诉我们的是乘积，不是因子。
    要拆出 N，得看 BPF 的**次谐波**：风道不对称、叶片间距误差、轴承每转一次的
    冲击，都会在 BPF/N（即轴频）处留下能量。哪个 N 对应的 BPF/N 位置上有稳定的
    谱线，N 就是几。

实测（4 个在高档运行的类别）：BPF/5 位置的信噪比（相对局部底噪）为
    C2 5.48×、C3 14.72×、C5 5.60×，而 N=2/3/4/6/7 全部 ≈1.7×（即无谱线）
故推定 5 叶。低档 C1 更干脆：100.0/5 = 20.0 Hz 处有 9.75× 的谱线，
并且在 20.5 Hz 直接能看到一个独立峰（比局部中位谱高 10.2 倍）。

由此换算：低档 20.5 Hz × 60 = 1230 r/min；高档 36.5 Hz × 60 = 2190 r/min。
铭牌 20 W / 180 mm 台扇，这个转速区间是合理的。
"""

from __future__ import annotations

import numpy as np

from .features import find_bpf, spectrum


def subharmonic_profile(x: np.ndarray, sr: int, n_range: range = range(2, 8)) -> dict[int, float]:
    """返回 {N: BPF/N 处的峰值 / 局部中位谱}。值明显大于 2 说明该 N 处有真实谱线。"""
    sp = spectrum(x, sr)
    bpf = find_bpf(sp)
    if bpf <= 0:
        return {n: 1.0 for n in n_range}
    hw = max(1.5, sp.df * 2)
    prof: dict[int, float] = {}
    for nblade in n_range:
        f0 = bpf / nblade
        m = (sp.freq >= f0 - hw) & (sp.freq <= f0 + hw)
        if not m.any():
            prof[nblade] = 1.0
            continue
        pk = float(sp.mag[m].max())
        band = (sp.freq >= max(5.0, f0 * 0.55)) & (sp.freq <= f0 * 1.65 + 5)
        floor = float(np.median(sp.mag[band])) if band.any() else 1e-20
        prof[nblade] = pk / (floor + 1e-20)
    return prof


def estimate_blade_count(profiles: list[dict[int, float]]) -> tuple[int, dict[int, float]]:
    """多段录音取中位数，返回 (推定叶片数, 各 N 的中位比值)。"""
    keys = sorted(profiles[0].keys())
    med = {n: float(np.median([p[n] for p in profiles])) for n in keys}
    best = max(keys, key=lambda n: med[n])
    return best, med


def estimate_rpm(bpf_hz: float, n_blades: int) -> float:
    """由 BPF 与叶片数反推转速（r/min）。"""
    if n_blades <= 0:
        return 0.0
    return float(bpf_hz / n_blades * 60.0)


def harmonic_decay(x: np.ndarray, sr: int, n_harm: int = 6) -> list[float]:
    """BPF 各次谐波相对基频的 dB 值，用于刻画"谐波有多干净"。"""
    sp = spectrum(x, sr)
    bpf = find_bpf(sp)
    if bpf <= 0:
        return []
    hw = lambda k: max(1.5, sp.df * 1.5 * k)  # noqa: E731
    out: list[float] = []
    a1 = None
    for k in range(1, n_harm + 1):
        f0 = bpf * k
        if f0 > sp.sr / 2:
            break
        m = (sp.freq >= f0 - hw(k)) & (sp.freq <= f0 + hw(k))
        a = float(sp.mag[m].max()) if m.any() else 1e-20
        if k == 1:
            a1 = a
        out.append(float(20 * np.log10((a + 1e-20) / (a1 + 1e-20))))
    return out
