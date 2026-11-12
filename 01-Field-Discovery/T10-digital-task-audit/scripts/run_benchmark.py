#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
课题10 · 耗时基准测试与准确率评估

做三件事：
1. **真实计时** —— 用 perf_counter 实测 AI 流水线处理 10 个样本的耗时（重复 N 轮取中位数，
   排除首次解释器预热与磁盘缓存冷启动的影响）；
2. **真实评估** —— 把提取结果与人工标注基准逐字段比对，算出字段级准确率。
   AI 的准确率不是"看起来还行"，而是量化的；
3. **合并人工数据** —— 如果 manual_run/timing_sheet.csv 已填，自动生成人工 vs AI 对比图
   与摊销曲线（第几次使用之后 AI 开始回本）。

用法：
    python scripts/run_benchmark.py --repeat 50
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

import ai_extract as ax

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
RAW = ROOT / "raw_data"
FIGS = ROOT / "figures"
MANUAL_SHEET = ROOT / "manual_run" / "timing_sheet.csv"
CONFIG = ROOT / "manual_run" / "audit_config.csv"

EVAL_FIELDS = ["course", "date_iso", "time_start", "time_end", "location", "deadline", "action"]
FIELD_CN = {
    "course": "课程名", "date_iso": "日期", "time_start": "开始时间",
    "time_end": "结束时间", "location": "地点", "deadline": "截止时间", "action": "事件类型",
}


# ---------------------------------------------------------------------------
# 计时
# ---------------------------------------------------------------------------

def time_pipeline(repeat: int, anchor, term_start):
    """端到端计时：每次都是「扫描目录 + 读 10 个文件 + 提取」的完整流程。"""
    files = sorted(SAMPLES.glob("N*.txt"))
    ax.run(SAMPLES, anchor, term_start)          # warm-up，不计入

    totals: list[float] = []
    per_file: dict[str, list[float]] = {p.name: [] for p in files}

    for _ in range(repeat):
        t0 = time.perf_counter()
        rows = ax.run(SAMPLES, anchor, term_start)
        totals.append((time.perf_counter() - t0) * 1000.0)
        # run() 内部按文件顺序处理，用文件边界切不出单文件耗时，
        # 所以单文件耗时单独测（见下）。

    for p in files:
        raw = p.read_text(encoding="utf-8")
        for _ in range(repeat):
            t0 = time.perf_counter()
            for ev in ax.split_events(raw):
                ax.extract_one(ev, anchor, term_start)
            per_file[p.name].append((time.perf_counter() - t0) * 1000.0)

    return totals, per_file, len(rows)


def stat_block(v: list[float]) -> dict:
    v = sorted(v)
    return {
        "n": len(v),
        "median_ms": statistics.median(v),
        "mean_ms": statistics.fmean(v),
        "std_ms": statistics.pstdev(v) if len(v) > 1 else 0.0,
        "min_ms": v[0],
        "max_ms": v[-1],
        "p95_ms": v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))],
    }


# ---------------------------------------------------------------------------
# 准确率评估
# ---------------------------------------------------------------------------

def _eq(field: str, gt: str, pred: str) -> bool:
    gt, pred = (gt or "").strip(), (pred or "").strip()
    if field == "deadline":
        return gt[:10] == pred[:10]
    if field in ("time_end", "deadline"):
        if not gt and not pred:
            return True
    return gt == pred


def evaluate(pred_rows: list[dict]) -> tuple[list[dict], dict]:
    gt_path = SAMPLES / "_ground_truth.csv"
    with gt_path.open(encoding="utf-8-sig") as fh:
        gt = [r for r in csv.DictReader(fh) if r["include_in_eval"] == "1"]

    by_file: dict[str, dict] = {}
    for r in pred_rows:
        by_file.setdefault(r["file"], r)

    detail = []
    for g in gt:
        p = by_file.get(g["file"], {})
        row = {"file": g["file"], "include_in_eval": g["include_in_eval"]}
        for f in EVAL_FIELDS:
            ok = _eq(f, g.get(f, ""), p.get(f, ""))
            row[f"{f}_gt"] = g.get(f, "")
            row[f"{f}_pred"] = p.get(f, "")
            row[f"{f}_ok"] = "1" if ok else "0"
        row["n_correct"] = sum(int(row[f"{f}_ok"]) for f in EVAL_FIELDS)
        detail.append(row)

    summary = {}
    for f in EVAL_FIELDS:
        hit = sum(int(r[f"{f}_ok"]) for r in detail)
        summary[f] = {"correct": hit, "total": len(detail), "accuracy": round(hit / len(detail), 4)}
    summary["_overall"] = {
        "correct": sum(r["n_correct"] for r in detail),
        "total": len(detail) * len(EVAL_FIELDS),
        "accuracy": round(sum(r["n_correct"] for r in detail) / (len(detail) * len(EVAL_FIELDS)), 4),
        "samples": len(detail),
    }
    return detail, summary


# ---------------------------------------------------------------------------
# 人工数据
# ---------------------------------------------------------------------------

def load_manual() -> dict | None:
    if not MANUAL_SHEET.exists():
        return None
    with MANUAL_SHEET.open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    per_sample, check_s, ok = {}, {}, True
    for r in rows:
        vals = []
        for k in ("manual_round1_s", "manual_round2_s", "manual_round3_s"):
            v = (r.get(k) or "").strip()
            if not v:
                ok = False
                break
            try:
                vals.append(float(v))
            except ValueError:
                ok = False
                break
        if not ok:
            break
        per_sample[r["sample_id"]] = statistics.median(vals)
        try:
            check_s[r["sample_id"]] = float((r.get("ai_check_s") or 0) or 0)
        except ValueError:
            check_s[r["sample_id"]] = 0.0
    if not ok or not per_sample:
        return None
    return {"per_sample_s": per_sample, "total_s": sum(per_sample.values()),
            "n": len(per_sample), "check_s": check_s,
            "check_total_s": sum(check_s.values())}


def load_config() -> dict:
    cfg = {"ai_build_minutes": 0.0, "manual_setup_minutes": 0.0}
    if CONFIG.exists():
        with CONFIG.open(encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if r["key"] in cfg:
                    try:
                        cfg[r["key"]] = float(r["value"] or 0)
                    except ValueError:
                        pass
    return cfg


# ---------------------------------------------------------------------------
# 出图
# ---------------------------------------------------------------------------

def make_figures(per_file, manual, acc_summary, cfg, ai_per_sample_ms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "#C9D3DF"
    plt.rcParams["grid.color"] = "#E6EBF1"

    FIGS.mkdir(parents=True, exist_ok=True)
    out = []

    # ---- 图1：逐样本耗时对比 ----
    files = sorted(per_file.keys())
    ai_ms = [statistics.median(per_file[f]) for f in files]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = range(len(files))
    w = 0.38
    if manual:
        man_s = [manual["per_sample_s"].get(f.replace(".txt", ""), 0) * 1000 for f in files]
        ax.bar([i - w / 2 for i in x], man_s, w, label="人工处理", color="#C97C7C", alpha=0.85)
        ax.bar([i + w / 2 for i in x], ai_ms, w, label="AI 流水线", color="#3E9E8F", alpha=0.9)
        for i, v in enumerate(man_s):
            ax.text(i - w / 2, v * 1.6, f"{v/1000:.1f}s", ha="center", fontsize=7.5, color="#8A4B4B")
        for i, v in enumerate(ai_ms):
            # 对数轴上用"乘固定倍率"定位，柱高 0.05 与 1 之间的视觉差距才正常
            ax.text(i + w / 2, v * 2.2, f"{v:.3f}ms", ha="center", va="bottom", fontsize=7, color="#22705F")
        ax.set_yscale("log")
        ax.set_ylim(bottom=min(ai_ms) * 0.35)
        ax.set_ylabel("单样本处理耗时（毫秒，对数轴）", fontsize=10.5)
        man_med = statistics.median(man_s)
        ai_med = statistics.median(ai_ms)
        orders = math.log10(man_med / ai_med) if ai_med > 0 else 0
        ax.set_title(f"逐样本耗时对比：AI 与人工差约 {orders:.0f} 个数量级，所以必须用对数轴",
                     fontsize=13, loc="left")
    else:
        ax.bar(list(x), ai_ms, 0.5, label="AI 流水线", color="#3E9E8F", alpha=0.9)
        for i, v in enumerate(ai_ms):
            ax.text(i, v, f"{v:.3f}ms", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("单样本处理耗时（毫秒）", fontsize=10.5)
        ax.set_title("逐样本耗时（AI 流水线）· 人工耗时待填后可对比", fontsize=13, loc="left")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f.replace(".txt", "") for f in files], fontsize=9)
    ax.legend(frameon=False, fontsize=10, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    ax.grid(True, axis="y", lw=0.6, alpha=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p = FIGS / "fig1_per_sample_timing.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    out.append(p)

    # ---- 图2：字段级准确率 ----
    fig, ax = plt.subplots(figsize=(9, 4.6))
    fields = EVAL_FIELDS
    vals = [acc_summary[f]["accuracy"] * 100 for f in fields]
    colors = ["#3E9E8F" if v >= 90 else "#E8A33D" if v >= 60 else "#D9534F" for v in vals]
    bars = ax.barh([FIELD_CN[f] for f in fields], vals, color=colors, alpha=0.88, height=0.62)
    for b, v in zip(bars, vals):
        ax.text(v + 1.2, b.get_y() + b.get_height() / 2, f"{v:.0f}%", va="center", fontsize=10)
    ax.set_xlim(0, 112)
    ax.set_xlabel("字段提取准确率（%）", fontsize=10.5)
    ax.set_title(f"字段级准确率（{acc_summary['_overall']['samples']} 个纳入评估的样本，"
                 f"总体 {acc_summary['_overall']['accuracy']*100:.1f}%）", fontsize=13, loc="left")
    ax.grid(True, axis="x", lw=0.6, alpha=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p = FIGS / "fig2_field_accuracy.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    out.append(p)

    # ---- 图3：摊销曲线（AI 什么时候回本）----
    if manual:
        build_s = cfg["ai_build_minutes"] * 60
        man_per = manual["total_s"] / manual["n"]
        chk_per = manual["check_total_s"] / manual["n"]
        ai_per = chk_per + ai_per_sample_ms / 1000.0   # 含人工校验，才是真实单次成本
        # x 轴要能覆盖回本点，否则图上看起来"AI 永远更慢"，反而误导
        n_max = 40
        if man_per > ai_per:
            n_max = int(max(40, (build_s / (man_per - ai_per)) * 1.35))
        ns = list(range(1, n_max + 1))
        man_curve = [n * man_per / 60 for n in ns]
        ai_curve = [(build_s + n * ai_per) / 60 for n in ns]
        fig, ax = plt.subplots(figsize=(9.4, 5))
        ax.plot(ns, man_curve, lw=2.2, color="#C97C7C", label="人工：随样本数线性增长")
        ax.plot(ns, ai_curve, lw=2.2, color="#3E9E8F",
                label=f"AI：一次性构建 {cfg['ai_build_minutes']:.0f} 分钟后，单次成本极低")
        if man_per > ai_per:
            be = build_s / (man_per - ai_per)
            if 1 <= be <= n_max:
                ax.axvline(be, ls="--", lw=1.4, color="#8899AA")
                ax.annotate(f"回本点 ≈ {be:.0f} 个样本", xy=(be, be * man_per / 60),
                            xytext=(be + 2, be * man_per / 60 * 0.55), fontsize=10,
                            arrowprops=dict(arrowstyle="->", color="#8899AA", lw=1.2))
        ax.set_xlabel("累计处理的样本数", fontsize=10.5)
        ax.set_ylabel("累计耗时（分钟）", fontsize=10.5)
        ax.set_title("摊销曲线：AI 的构建成本是一次性的，样本越多摊得越薄", fontsize=13, loc="left")
        ax.legend(frameon=False, fontsize=10)
        ax.grid(True, lw=0.6, alpha=0.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        p = FIGS / "fig3_amortization.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out.append(p)

    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=50)
    ap.add_argument("--anchor", default=str(ax.DEFAULT_ANCHOR))
    ap.add_argument("--term-start", default=str(ax.DEFAULT_TERM_START))
    args = ap.parse_args()

    from datetime import date
    anchor = date.fromisoformat(args.anchor)
    term_start = date.fromisoformat(args.term_start)

    RAW.mkdir(parents=True, exist_ok=True)
    print(f"样本目录：{SAMPLES}")
    print(f"重复轮次：{args.repeat}（取中位数，剔除预热与冷缓存影响）\n")

    print("① 计时中……")
    totals, per_file, n_rows = time_pipeline(args.repeat, anchor, term_start)
    tot = stat_block(totals)
    per_stats = {f: stat_block(v) for f, v in per_file.items()}
    ai_per_sample_ms = statistics.median([s["median_ms"] for s in per_stats.values()])
    print(f"   10 个样本端到端：中位数 {tot['median_ms']:.2f} ms"
          f"（均值 {tot['mean_ms']:.2f}，p95 {tot['p95_ms']:.2f}，n={tot['n']}）")
    print(f"   单样本中位耗时：{ai_per_sample_ms:.2f} ms")
    print(f"   共产出 {n_rows} 条结构化事件\n")

    print("② 提取并评估……")
    rows = ax.run(SAMPLES, anchor, term_start)
    with (RAW / "ai_extracted.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    detail, acc = evaluate(rows)
    with (RAW / "field_accuracy.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(detail[0].keys()))
        w.writeheader()
        w.writerows(detail)

    print(f"   总体字段准确率：{acc['_overall']['accuracy']*100:.1f}%"
          f"（{acc['_overall']['correct']}/{acc['_overall']['total']} 个字段）")
    for f in EVAL_FIELDS:
        print(f"     {FIELD_CN[f]:<6}{acc[f]['accuracy']*100:>6.1f}%  ({acc[f]['correct']}/{acc[f]['total']})")
    print()

    with (RAW / "ai_timing.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "median_ms", "mean_ms", "std_ms", "min_ms", "max_ms", "p95_ms", "runs"])
        for f in sorted(per_stats):
            s = per_stats[f]
            w.writerow([f, f"{s['median_ms']:.3f}", f"{s['mean_ms']:.3f}", f"{s['std_ms']:.3f}",
                        f"{s['min_ms']:.3f}", f"{s['max_ms']:.3f}", f"{s['p95_ms']:.3f}", s["n"]])

    manual = load_manual()
    cfg = load_config()
    print("③ 人工数据：" + ("已载入 manual_run/timing_sheet.csv" if manual else "未填写，跳过对比（图1/图3 将只显示 AI）"))
    if manual:
        man_per = manual["total_s"] / manual["n"]
        chk_per = manual["check_total_s"] / manual["n"]
        real_per = chk_per + ai_per_sample_ms / 1000.0
        print(f"   人工单样本耗时（中位）：{man_per:.1f} s")
        print(f"   纯计算比：{man_per / (ai_per_sample_ms/1000):,.0f} 倍"
              f"　← 这个数字没有意义，因为 AI 的输出必须有人校验")
        print(f"   AI 路径单样本耗时（机器 {ai_per_sample_ms/1000:.4f}s + 人工校验 {chk_per:.1f}s）：{real_per:.1f} s")
        print(f"   含校验的真实提速：{man_per / real_per:.1f} 倍")

    summary = {
        "generated_by": "scripts/run_benchmark.py",
        "repeat": args.repeat,
        "anchor_date": str(anchor),
        "term_start": str(term_start),
        "samples": len(per_stats),
        "events_extracted": n_rows,
        "ai_total_10samples_ms": tot,
        "ai_per_sample_median_ms": round(ai_per_sample_ms, 3),
        "per_file": {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in per_stats.items()},
        "field_accuracy": acc,
        "manual": ({"per_sample_s": manual["per_sample_s"], "total_s": round(manual["total_s"], 2),
                    "per_sample_mean_s": round(manual["total_s"] / manual["n"], 2)} if manual else None),
        "config": cfg,
    }
    (RAW / "benchmark_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                encoding="utf-8")

    print("\n④ 出图中……")
    for p in make_figures(per_file, manual, acc, cfg, ai_per_sample_ms):
        print(f"   [图] {p.relative_to(ROOT)}")

    print(f"\n结果已写入 {RAW.relative_to(ROOT)}/：ai_extracted.csv / ai_timing.csv / "
          f"field_accuracy.csv / benchmark_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
