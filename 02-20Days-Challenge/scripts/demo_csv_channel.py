#!/usr/bin/env python3
"""证明特征提取器与声道无关：直接吃一份 CSV 时序，跑出同一套 34 维特征。

这个脚本存在的理由是一个诚实问题：项目要求"麦克风 + 加速度计"两路数据，
但本轮只采到了麦克风。与其在文档里写"代码支持加速度计"这种无法验证的话，
不如提供一个**现在就能跑**的证明：`features.extract()` 的输入只要满足
"一维等间隔序列 + 采样率"，来源是 WAV 还是 CSV、是声压还是加速度，它并不关心。

用法：
    python scripts/demo_csv_channel.py                       # 用导出的 csv_series 演示
    python scripts/demo_csv_channel.py path/to/accel.csv     # 用你自己的表格

CSV 格式要求（两种都支持）：
    · 单列，无表头                     → 每行一个采样值
    · 两列 time,value（可带表头）       → 取第二列，第一列仅用于推断采样率

对加速度计数据的额外提醒（写在这里，是因为补采时最容易踩）：
    ① Phyphox 导出的「线性加速度」已去重力，可直接用；
       导出的「加速度（含 g）」必须先减掉重力分量，否则 1 g 直流会盖住所有交流特征。
    ② 采样率通常在 CSV 的元信息行里（`# Sampling rate: 200 Hz`），
       本文不解析元信息，请用 --rate 显式指定。
    ③ 200 Hz 的采样率意味着奈奎斯特上限只有 100 Hz，远低于声道的 8 kHz，
       所以依赖 2~8 kHz 的擦碰判据在加速度通道上不成立——这是换声道不是换文件格式。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))

from vibdiag import features  # noqa: E402


def _looks_like_header(row: list[str]) -> bool:
    for cell in row:
        try:
            float(cell)
        except ValueError:
            return True
    return False


def load_table(path: str) -> tuple[np.ndarray, float | None]:
    """读取 CSV 时序，返回 (信号, 推断出的采样率或 None)。"""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].lstrip().startswith("#")]
    if not rows:
        raise ValueError(f"{path}: 空文件")

    header = None
    if _looks_like_header(rows[0]):
        header = rows[0]
        rows = rows[1:]

    if len(rows[0]) == 1:
        x = np.array([float(r[0]) for r in rows])
        return x, None

    t = np.array([float(r[0]) for r in rows])
    x = np.array([float(r[1]) for r in rows])
    dt = float(np.median(np.diff(t)))
    sr = 1.0 / dt if dt > 0 else None
    if header:
        print(f"       列名：{header}")
    return x, sr


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="用一份 CSV 时序跑同一套特征提取")
    p.add_argument("path", nargs="?", default=os.path.join(HERE, "csv_series", "C5_rubbing_01.csv"))
    p.add_argument("--rate", type=float, default=None, help="单列 CSV 时必须指定采样率")
    p.add_argument("--no-level", action="store_true", help="只用增益无关的 31 维")
    args = p.parse_args(argv)

    x, sr = load_table(args.path)
    sr = args.rate or sr
    if not sr:
        print("错误：单列 CSV 无法推断采样率，请用 --rate 指定", file=sys.stderr)
        return 1

    print(f"[csv-channel] {os.path.relpath(args.path, HERE)}")
    print(f"              采样点数 {len(x)}，推断采样率 {sr:.1f} Hz，时长 {len(x)/sr:.3f} s")
    print(f"              直流分量（均值）{x.mean():+.6f}  ← 加速度计若未去重力，这里会是 1.0 量级")

    names = features.feature_names(not args.no_level)
    feats = features.extract(x, sr, include_level=not args.no_level)
    print(f"              提取 {len(names)} 维特征（与 WAV 通道完全相同的一套代码）：")
    for n in names:
        print(f"                {n:24s} {feats[n]:12.6f}")
    print("\n[结论] 特征提取器不关心数据来源，只要求一维等间隔序列 + 采样率。")
    print("       但这不等于加速度计通道已经做完——见本文件顶部第 ③ 条提醒。")
    print("       另外：csv_series/ 是 WAV 的 1 kHz 降采样视图（有损），")
    print("       所以这里 band_2000_4000 / band_4000_8000 / hf_noise_ratio 等高频特征")
    print("       必然与 features/feature_table.csv 里的值不同，属预期行为，不是 bug。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
