#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自检用合成数据生成器 —— 仅用于验证 analyze_gait.py 的逻辑是否跑通。

⚠️ 重要：这里生成的是**数学合成信号，不是实测数据**，绝不能用于报告、不能进 figures/。
   它的唯一用途是：在你还没去采集之前，确认脚本本身没有 bug。

用法：
    python scripts/_selftest/run_selftest.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent          # T08-gait-acceleration/
SYN_DIR = HERE / "synthetic_raw"
SYN_FIG = HERE / "synthetic_figures"

FS = 100.0
DUR = 32.0
RNG = np.random.default_rng(7)


def gen(state: str, rep: int) -> np.ndarray:
    """返回 (n, 5) 数组：time, x, y, z, mag"""
    n = int(FS * DUR)
    t = np.arange(n) / FS
    noise = RNG.normal(0, 0.035, (n, 3))

    if state == "S1_STATIC":
        base = RNG.normal(0, 0.02, (n, 3))
        x, y, z = base[:, 0], base[:, 1], base[:, 2]

    elif state in ("S2_WALK", "S3_RUN", "S4_STAIR"):
        f_step = {"S2_WALK": 1.62, "S3_RUN": 2.68, "S4_STAIR": 1.75}[state]
        amp = {"S2_WALK": 2.4, "S3_RUN": 7.2, "S4_STAIR": 2.9}[state]
        # 步态：两倍步频的躯干上下振动 + 单倍步频的左右摆动
        vert = amp * np.sin(2 * np.pi * 2 * f_step * t)
        lateral = amp * 0.45 * np.sin(2 * np.pi * f_step * t + 0.7)
        fore = amp * 0.25 * np.sin(2 * np.pi * f_step * t + 1.9)
        # 落地冲击：每步一个衰减脉冲
        impact = np.zeros(n)
        for k in range(int(DUR * f_step)):
            idx = int(k / f_step * FS)
            width = int(0.06 * FS)
            if idx + width < n:
                impact[idx:idx + width] += np.exp(-np.linspace(0, 5, width)) * amp * 0.5
        if state == "S4_STAIR":
            # 楼梯：垂直位移带来慢变偏置 + 冲击不对称
            vert = vert * 1.2 + 1.1 * np.sin(2 * np.pi * 0.13 * t)
            impact = impact * 1.6
        x = lateral + noise[:, 0]
        y = fore + noise[:, 1]
        z = vert + impact + noise[:, 2]
    else:
        raise ValueError(state)

    mag = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    return np.column_stack([t, x, y, z, mag])


def main() -> int:
    SYN_DIR.mkdir(parents=True, exist_ok=True)
    SYN_FIG.mkdir(parents=True, exist_ok=True)
    for state in ("S1_STATIC", "S2_WALK", "S3_RUN", "S4_STAIR"):
        for rep in (1, 2, 3):
            arr = gen(state, rep)
            path = SYN_DIR / f"{state}_{rep:02d}_20260101-0000.csv"
            with path.open("w", encoding="utf-8", newline="") as fh:
                fh.write("Time (s),Absolute acceleration (m/s^2),X (m/s^2),Y (m/s^2),Z (m/s^2)\n")
                for row in arr:
                    fh.write(f"{row[0]:.4f},{row[4]:.5f},{row[1]:.5f},{row[2]:.5f},{row[3]:.5f}\n")
    print(f"[自检] 已生成 12 个合成样本 -> {SYN_DIR}")
    print("[自检] 注意：这些是数学合成信号，仅用于验证代码，不可用于报告！\n")

    script = ROOT / "scripts" / "analyze_gait.py"

    # analyze_gait.py 会把 threshold_rules.json 与 analysis_summary.md 写到 T08 根目录。
    # 自检用的是合成数据，这两个产物绝不能留在正式目录里被当成实测结论，
    # 所以跑之前先备份、跑完按原状恢复（原本不存在就删掉）。
    artifacts = [ROOT / "threshold_rules.json", ROOT / "analysis_summary.md"]
    before = {p: (p.read_text(encoding="utf-8") if p.exists() else None) for p in artifacts}

    for args in (["--check-only"], []):
        cmd = [sys.executable, str(script), "--data-dir", str(SYN_DIR), "--fig-dir", str(SYN_FIG), *args]
        print(f"[自检] 运行: {' '.join(cmd[1:])}")
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode not in (0, 1):
            print(f"[自检] 失败，退出码 {r.returncode}")
            return r.returncode

    for p, old in before.items():
        if old is None:
            if p.exists():
                p.unlink()
                print(f"[自检] 已清理合成数据产物：{p.name}（避免被误认为实测结果）")
        else:
            p.write_text(old, encoding="utf-8")
            print(f"[自检] 已还原原有产物：{p.name}")

    print("\n[自检] 完成。若上面打印出了规则、准确率与 5 张图，说明脚本逻辑正常。")
    print("[自检] 合成数据不会留在 raw_data/，正式分析前请按 README 采集真实数据。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
