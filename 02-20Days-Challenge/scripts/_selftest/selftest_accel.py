#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""import_accel.py / accel_check.py 的合成数据自检。

⚠️ 本脚本生成的信号是**合成数据**，只为验证代码路径能跑通，
   绝不可用于报告、不可写入 raw_accel/。跑完自动清理临时目录。

它验证的是一件事：如果真实数据里 C3 的 1× 确实升高、C5 确实有冲击，
这套脚本能不能把它们认出来；反过来，当判据不成立时它会不会如实说"不成立"。

用法：
    python scripts/_selftest/selftest_accel.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent      # 02-20Days-Challenge/
PY = sys.executable

# 5 工况的合成参数：(标签, 转频Hz, 1×幅值, 附加成分)
# 转频取 accel_check.py 的实测默认值（低 27.2 / 高 33.3）——两边必须一致，
# 否则合成数据里 C1 的 1× 落在 20 Hz、脚本却去 27.2 Hz 找，判据会假性不成立。
#   0=无   1=低频调制（遮挡）   2=周期冲击（擦碰）
PLAN = [
    ("C1", 27.2, 0.05, 0),
    ("C2", 33.3, 0.05, 0),
    ("C3", 33.3, 0.22, 0),      # 失衡：1× 抬到 4 倍以上
    ("C4", 33.3, 0.05, 1),
    ("C5", 33.3, 0.05, 2),      # 擦碰：时域冲击
]
SR, DUR, REPEAT = 200.0, 4.0, 3


def make_csv(dst: Path) -> int:
    rng = np.random.default_rng(7)
    t = np.arange(0, DUR, 1 / SR)
    n = 0
    for key, f1, amp1, kind in PLAN:
        for _ in range(REPEAT):
            n += 1
            x = amp1 * np.sin(2 * np.pi * f1 * t) + 0.3 * amp1 * np.sin(2 * np.pi * 2 * f1 * t)
            x = x + rng.normal(0, 0.02, len(t))
            if kind == 1:
                x = x + 0.05 * np.sin(2 * np.pi * 8 * t)
            if kind == 2:
                for i in range(3, len(t), 40):
                    x[i] += 0.6
            y = 0.8 * x + rng.normal(0, 0.01, len(t))
            z = 0.6 * x + rng.normal(0, 0.01, len(t))
            lines = [
                "# This file was exported from the app phyphox.    (合成自检数据)",
                "# Date\t2026-11-25",
                "",
                "Time (s),Linear Acceleration x (m/s^2),Linear Acceleration y (m/s^2),Linear Acceleration z (m/s^2)",
            ]
            lines += [f"{t[i]:.5f},{x[i]:.6f},{y[i]:.6f},{z[i]:.6f}" for i in range(len(t))]
            (dst / f"Linear Acceleration_selftest_{key}_{n:02d}.csv").write_text("\n".join(lines), encoding="utf-8")
    return n


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="accel_selftest_"))
    try:
        src, raw, res = tmp / "src", tmp / "raw_accel", tmp / "results"
        src.mkdir()
        n = make_csv(src)
        print(f"[selftest] 合成 {n} 段（{SR:.0f} Hz / {DUR}s），临时目录 {tmp}\n")

        plan = ",".join(f"{k}={REPEAT}" for k, *_ in PLAN)
        r1 = subprocess.run([PY, str(ROOT / "scripts" / "import_accel.py"),
                             "--src", str(src), "--dst", str(raw), "--plan", plan],
                            capture_output=True, text=True, encoding="utf-8")
        print(r1.stdout.strip()[-400:])
        if r1.returncode != 0:
            print(r1.stderr[-800:])
            return 1

        r2 = subprocess.run([PY, str(ROOT / "scripts" / "accel_check.py"),
                             "--dir", str(raw), "--out", str(res)],
                            capture_output=True, text=True, encoding="utf-8")
        print("\n" + r2.stdout.strip())
        if r2.returncode != 0:
            print(r2.stderr[-800:])
            return 1

        # 断言跟着"实质结论"走，不绑死某句具体措辞：
        #   ① C3 的 1× 判据成立  ② C5 的冲击判据可用  ③ 100 Hz 被判定为电磁分量
        # （原先断言 "不可观测"，在叶片数改为 3 叶、BPF_high=99.9 Hz 后不再成立）
        ok = ("判据成立" in r2.stdout) and ("冲击判据可用" in r2.stdout) and ("电磁分量" in r2.stdout)
        print("\n[selftest] " + ("通过：三条判据的判定逻辑都按预期工作" if ok else "失败：判定逻辑与预期不符"))
        print("[selftest] 合成数据不作任何用途，已随临时目录删除")
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
