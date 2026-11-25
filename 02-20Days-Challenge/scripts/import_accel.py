#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 Phyphox 导出的加速度 CSV 规整成与声学同构的 75 段数据集。

为什么需要这个脚本：Phyphox 导出的文件名形如
    Linear Acceleration_2026-11-25_19-30-12.csv
里面既没有工况、也没有 3.000 s 对齐——而声学那 75 段是按
C1_normal_low_01.wav 这种口径命名的。要让两路"同构可比"，
必须先做一次规整，且把"哪个文件是哪个工况"这一步落成可复查的记录。

用法：
    # 按录制顺序分配（先录 C1 十五段，再 C2 十五段……）
    python scripts/import_accel.py --src 导出目录 --plan C1=15,C2=15,C3=15,C4=15,C5=15

    # 更稳妥：给一份手工映射表（filename,label）
    python scripts/import_accel.py --src 导出目录 --map mapping.csv

    # 只看分配结果，不写文件
    python scripts/import_accel.py --src 导出目录 --plan ... --dry-run

输出：
    raw_accel/C{n}_..._{nn}.csv       t_s,x,y,z（米/秒²，已去直流）
    raw_accel/sample_index_accel.csv  每段的采样率/时长/是否含 g/RMS/sha256

诚实约定：
    · 脚本不做任何插值、不去噪、不补点。裁剪只切窗口，不改数值。
    · 采样率由时间列实测，不信文件名与元信息里的声称值。
    · 含 g 的数据只做"减均值"处理，并标记 has_dc；这在手机非水平放置时
      不等于去重力，报告里必须注明（见 docs/加速度通道补采方案.md）。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LABELS = {
    "C1": "C1_normal_low",
    "C2": "C2_normal_high",
    "C3": "C3_imbalance",
    "C4": "C4_obstruction",
    "C5": "C5_rubbing",
}


def parse_phyphox_csv(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """读 Phyphox CSV。返回 (t, xyz[n,3], meta)。

    Phyphox 的格式：# 开头的元信息行 + 一行表头 + 数据。
    表头里时间列通常叫 "Time (s)"，三轴叫 "Acceleration x (m/s^2)" 之类，
    也可能是 "Linear Acceleration x (m/s^2)"。这里按关键字匹配，不写死。
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    meta = {"raw_header": "", "comments": 0}
    rows: list[list[str]] = []
    header: list[str] | None = None
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            meta["comments"] += 1
            continue
        if header is None:
            header = [c.strip() for c in s.split(",")]
            meta["raw_header"] = s
            continue
        rows.append([c.strip() for c in s.split(",")])
    if header is None or not rows:
        raise ValueError(f"{path.name}: 没读到表头或数据行")

    lower = [h.lower() for h in header]
    t_col = next((i for i, h in enumerate(lower) if h.startswith("time")), None)
    axis_cols, axis_names = [], []
    for i, h in enumerate(lower):
        if "acceleration" in h or h.endswith("(m/s^2)") or h.startswith("x (") or h.startswith("y ("):
            if " x" in h or h.endswith("x (m/s^2)") or h.startswith("x ("):
                axis_cols.append(i), axis_names.append("x")
            elif " y" in h or h.startswith("y ("):
                axis_cols.append(i), axis_names.append("y")
            elif " z" in h or h.startswith("z ("):
                axis_cols.append(i), axis_names.append("z")
    if t_col is None or len(axis_cols) < 3:
        raise ValueError(f"{path.name}: 表头里找不到时间列或三轴列：{header}")

    arr = np.array([[float(r[i]) for i in (*[t_col], *axis_cols)] for r in rows if len(r) > max(axis_cols + [t_col])])
    t = arr[:, 0]
    xyz = arr[:, 1:4]
    meta["axes"] = "".join(axis_names)
    meta["has_g"] = "linear" not in meta["raw_header"].lower()
    meta["n_raw"] = int(xyz.shape[0])
    dt = np.diff(t)
    meta["sr"] = float(1.0 / np.median(dt)) if dt.size and np.median(dt) > 0 else 0.0
    meta["jitter_ms"] = float(np.std(dt) * 1000) if dt.size else 0.0
    return t, xyz, meta


def centered_window(n: int, sr: float, seconds: float) -> slice:
    """取中间 seconds 秒。取中间而不是开头，是为了避开按下录制键时的操作冲击。"""
    want = int(round(seconds * sr))
    if want >= n:
        return slice(0, n)
    start = (n - want) // 2
    return slice(start, start + want)


def sha16(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def to_csv_bytes(t: np.ndarray, xyz: np.ndarray) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["t_s", "x", "y", "z"])
    for i in range(len(t)):
        w.writerow([f"{t[i]:.4f}", f"{xyz[i, 0]:.6f}", f"{xyz[i, 1]:.6f}", f"{xyz[i, 2]:.6f}"])
    return buf.getvalue().encode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Phyphox 加速度 CSV → 与声学同构的 raw_accel/")
    ap.add_argument("--src", required=True, help="Phyphox 导出目录（zip 需先解压）")
    ap.add_argument("--dst", default=str(ROOT / "raw_accel"), help="输出目录，默认 raw_accel/")
    ap.add_argument("--plan", default="", help="按录制顺序分配，如 C1=15,C2=15,C3=15")
    ap.add_argument("--map", default="", help="映射表 CSV，两列：filename,label")
    ap.add_argument("--trim", type=float, default=3.0, help="裁剪到多少秒（默认 3.0，与声学同口径）")
    ap.add_argument("--keep-dc", action="store_true", help="保留直流分量（默认减均值）")
    ap.add_argument("--dry-run", action="store_true", help="只打印分配结果，不写文件")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    files = sorted([p for p in src.rglob("*.csv") if p.resolve() != dst.resolve()], key=lambda p: p.stat().st_mtime)
    if not files:
        print(f"[import] {src} 下没有 CSV", file=sys.stderr)
        return 1

    # ---- 建立 filename → label 的分配 ----
    assign: list[tuple[Path, str]] = []
    if args.map:
        m = {}
        with open(args.map, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                m[(row.get("filename") or "").strip()] = (row.get("label") or "").strip()
        for f in files:
            lab = m.get(f.name)
            if lab:
                assign.append((f, lab))
        missing = [f.name for f in files if f.name not in m]
        if missing:
            print(f"[import] 映射表未覆盖 {len(missing)} 个文件，已跳过：{missing[:3]}", file=sys.stderr)
    elif args.plan:
        plan = []
        for part in args.plan.split(","):
            k, v = part.split("=")
            plan.append((k.strip(), int(v)))
        i = 0
        for key, n in plan:
            for _ in range(n):
                if i >= len(files):
                    break
                assign.append((files[i], LABELS[key]))
                i += 1
        if i < len(files):
            print(f"[import] 计划只覆盖 {i}/{len(files)} 个文件，多余的已忽略", file=sys.stderr)
    else:
        print("[import] 必须给 --plan 或 --map（脚本不猜工况）", file=sys.stderr)
        return 1

    if not assign:
        print("[import] 分配结果为空", file=sys.stderr)
        return 1

    print(f"[import] 待处理 {len(assign)} 个文件 → {dst}")
    print(f"{'源文件':52s} {'→ 输出':26s} {'采样率':>7s} {'原始':>6s} {'含g':>4s}")
    for f, lab in assign:
        try:
            _, _, meta = parse_phyphox_csv(f)
            print(f"{f.name[:50]:52s} → {lab:24s} {meta['sr']:7.0f} {meta['n_raw']/max(meta['sr'],1):6.2f}s {'Y' if meta['has_g'] else 'N':>4s}")
        except Exception as e:  # noqa: BLE001
            print(f"{f.name[:50]:52s} !! {e}")
    if args.dry_run:
        print("[import] dry-run，未写文件")
        return 0

    dst.mkdir(parents=True, exist_ok=True)
    counters: dict[str, int] = {}
    index_rows = []
    for f, lab in assign:
        t, xyz, meta = parse_phyphox_csv(f)
        sr = meta["sr"]
        if sr <= 0:
            print(f"[import] 跳过 {f.name}：时间列异常")
            continue
        sl = centered_window(len(t), sr, args.trim)
        t2, xyz2 = t[sl], xyz[sl].copy()
        t2 = t2 - t2[0]
        if not args.keep_dc:
            xyz2 = xyz2 - xyz2.mean(axis=0, keepdims=True)
        counters[lab] = counters.get(lab, 0) + 1
        name = f"{lab}_{counters[lab]:02d}.csv"
        payload = to_csv_bytes(t2, xyz2)
        (dst / name).write_bytes(payload)
        mag = np.sqrt((xyz2**2).sum(axis=1))
        index_rows.append({
            "file": name,
            "label": lab,
            "index": counters[lab],
            "sample_rate": int(round(sr)),
            "n_raw": meta["n_raw"],
            "duration_raw": round(meta["n_raw"] / sr, 3),
            "trimmed_seconds": round(len(t2) / sr, 3),
            "has_dc": int(meta["has_g"]),
            "jitter_ms": round(meta["jitter_ms"], 2),
            "rms_mag": round(float(np.sqrt(np.mean(mag**2))), 4),
            "sha256_16": sha16(payload),
            "source": f.name,
        })

    idx = dst / "sample_index_accel.csv"
    with idx.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(index_rows[0].keys()))
        w.writeheader()
        w.writerows(index_rows)

    by_sr: dict[int, int] = {}
    for r in index_rows:
        by_sr[r["sample_rate"]] = by_sr.get(r["sample_rate"], 0) + 1
    print(f"\n[import] 写入 {len(index_rows)} 段 → {dst}")
    print(f"[import] 登记表：{idx.name}")
    print(f"[import] 采样率分布：{by_sr}")
    if len(by_sr) > 1:
        print("[import] 警告：采样率不一致，阶次分析前必须先说明（或不混用）", file=sys.stderr)
    print("[import] 下一步：python scripts/accel_check.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
