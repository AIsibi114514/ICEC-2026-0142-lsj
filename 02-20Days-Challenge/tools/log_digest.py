"""由 process_logs/*.jsonl 生成"日志诊断报告"。

这个脚本是 7 天日志和 3 次 Git 迭代之间的桥：
每次准备发版本前跑一遍，报告里的"建议迭代项"就是 commit message 的依据。
报告的每一条结论都能追溯到具体 run_id，面试被追问时可直接打开对应 JSONL 行。

用法：
    python tools/log_digest.py                # 分析全部日志
    python tools/log_digest.py --last 3       # 只看最近 3 天
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "process_logs"
OUT_DIR = ROOT / "iterations"

# 触发迭代建议的阈值（写在代码里，避免事后拍脑袋）
THRESHOLDS = {
    "p95_dur_ms": 800.0,          # 单次运行 p95 超过 800ms 视为性能问题
    "error_rate": 0.05,           # 错误事件占比超过 5% 视为稳定性问题
    "field_miss_rate": 0.30,      # 某字段漏抽率超过 30% 视为规则缺陷
    "dedup_ratio": 0.40,          # 去重率超过 40% 提示收件箱重复投喂
    "min_days": 7,                # 连续性要求
}


def load_records(log_dir: Path = LOG_DIR) -> list[dict]:
    records: list[dict] = []
    for path in sorted(log_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec["_file"] = path.name
                records.append(rec)
    return records


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return round(ordered[idx], 1)


def _day_of(rec: dict) -> str:
    """取记录所属日期：优先字段，其次文件名（兼容早期没写 event_date 的记录）。"""
    if rec.get("event_date"):
        return str(rec["event_date"])
    return rec["_file"].replace(".jsonl", "").replace("SIM-", "")


def analyse(records: list[dict]) -> dict:
    runs = [r for r in records if r.get("event") == "run_end"]
    days_with_log = sorted({r["_file"].replace(".jsonl", "").replace("SIM-", "") for r in records})
    run_days = sorted({_day_of(r) for r in records if r.get("event") == "run_start"})

    # 连续性：日期区间内缺失的天
    gaps: list[str] = []
    if run_days:
        start = date.fromisoformat(run_days[0])
        end = date.fromisoformat(run_days[-1])
        have = set(run_days)
        cursor = start
        while cursor <= end:
            if cursor.isoformat() not in have:
                gaps.append(cursor.isoformat())
            cursor += timedelta(days=1)

    dur_values = [float(r.get("dur_ms") or 0) for r in runs]
    errors = [r for r in records if r.get("event") == "error"]
    warnings = [r for r in records if r.get("event") == "warning"]
    parsed = [r for r in records if r.get("event") == "notice_parsed"]

    calls_by_day: dict[str, int] = defaultdict(int)
    for r in runs:
        calls_by_day[_day_of(r)] += int(r.get("calls") or 0)

    miss_counter: Counter[str] = Counter()
    for r in parsed:
        missed = str(r.get("fields_missed") or "")
        for field in filter(None, missed.split(",")):
            miss_counter[field] += 1
    miss_rate = {f: round(c / len(parsed), 3) for f, c in miss_counter.items()} if parsed else {}

    dedup_ratios = [float(r.get("ratio") or 0) for r in records if r.get("event") == "dedup"]

    # 网页端（真人使用）与自动化运行分开统计
    web = [r for r in records if r.get("event") == "web_extract"]
    web_by_day: dict[str, int] = defaultdict(int)
    for r in web:
        web_by_day[_day_of(r)] += 1

    return {
        "days_with_log": days_with_log,
        "run_days": run_days,
        "gaps": gaps,
        "runs": runs,
        "run_count": len(runs),
        "web_count": len(web),
        "web_by_day": dict(sorted(web_by_day.items())),
        "web_dur_p50": pct([float(r.get("dur_ms") or 0) for r in web], 0.50),
        "web_avg_confidence": round(
            statistics.fmean([float(r.get("avg_confidence") or 0) for r in web]), 3
        ) if web else 0.0,
        "dur_p50": pct(dur_values, 0.50),
        "dur_p95": pct(dur_values, 0.95),
        "dur_max": round(max(dur_values), 1) if dur_values else 0.0,
        "error_count": len(errors),
        "error_types": Counter(r.get("error_type", "?") for r in errors),
        "error_stages": Counter(r.get("stage", "?") for r in errors),
        "warn_count": len(warnings),
        "warn_msgs": Counter(r.get("msg", "?") for r in warnings),
        "parsed_count": len(parsed),
        "miss_rate": miss_rate,
        "avg_confidence": round(statistics.fmean([float(r.get("confidence") or 0) for r in parsed]), 3) if parsed else 0.0,
        "calls_by_day": dict(sorted(calls_by_day.items())),
        "dedup_avg": round(statistics.fmean(dedup_ratios), 3) if dedup_ratios else 0.0,
        "categories": Counter(r.get("category", "?") for r in parsed),
    }


def suggestions(a: dict) -> list[str]:
    out: list[str] = []

    if len(a["run_days"]) < THRESHOLDS["min_days"]:
        out.append(
            f"[连续性] 目前仅 {len(a['run_days'])} 天有运行记录，距离 7 天要求还差 "
            f"{THRESHOLDS['min_days'] - len(a['run_days'])} 天。优先保证每天至少一次 run，哪怕收件箱为空。"
        )
    if a["gaps"]:
        out.append(f"[断档] 以下日期区间内没有日志，面试会被追问原因，需在 README 中如实说明：{', '.join(a['gaps'])}")

    if a["error_count"]:
        top = "、".join(f"{k}×{v}" for k, v in a["error_types"].most_common(3))
        out.append(f"[稳定性] 共 {a['error_count']} 条错误（{top}）。若同一 error_type 重复出现，"
                   f"这是第一次迭代最扎实的依据。")

    if a["dur_p95"] > THRESHOLDS["p95_dur_ms"]:
        out.append(f"[性能] 运行耗时 p95={a['dur_p95']}ms 超过阈值 {THRESHOLDS['p95_dur_ms']}ms，"
                   f"考虑减少重复读盘或缓存规则正则编译。")

    hot_miss = [f for f, rate in a["miss_rate"].items() if rate >= THRESHOLDS["field_miss_rate"]]
    if hot_miss:
        detail = "、".join(f"{f}({a['miss_rate'][f]:.0%})" for f in hot_miss)
        out.append(f"[规则] 以下字段漏抽率偏高：{detail}。优先补 rules.py 的词表与正则，"
                   f"改动范围小、diff 可读性高。")

    if a["dedup_avg"] > THRESHOLDS["dedup_ratio"]:
        out.append(f"[数据质量] 平均去重率 {a['dedup_avg']:.0%} 偏高，说明同一批通知被重复投喂，"
                   f"应改进件箱清理或去重键策略。")

    if a["web_count"] < 8:
        out.append(f"[用户证据] 网页端只有 {a['web_count']} 次真实调用，凑不齐「8 位用户」的痕迹。"
                   f"把地址发给同学，让他们自己粘贴通知——他们的调用会直接出现在这份日志里。")

    if not out:
        out.append("[状态] 各项指标均在阈值内，下一次迭代可转向功能增强（如 .ics 日历导出）。")
    return out


def render(a: dict, recs: list[dict], *, simulated: int = 0) -> str:
    banner = ""
    if simulated:
        banner = (
            f"> ⚠️ **本报告基于模拟数据**（{simulated} 条记录带 `simulated: true`），"
            f"仅用于预演报告形态与验证阈值，**不是真实运行记录**。\n\n"
        )
    lines = [
        "# 运行日志诊断报告",
        "",
        banner + f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}　|　"
        f"日志文件 {len(a['days_with_log'])} 个　|　记录 {len(recs)} 条",
        "",
        "## 1. 连续性（评审第一道防线）",
        "",
        f"- 有日志的日期：{len(a['run_days'])} 天",
        f"- 起止：{a['run_days'][0] if a['run_days'] else '-'} → {a['run_days'][-1] if a['run_days'] else '-'}",
        f"- 区间内断档：{('、'.join(a['gaps']) if a['gaps'] else '无')}",
        f"- 总运行次数：{a['run_count']}",
        "",
        "## 2. 调用频次与耗时",
        "",
        "| 日期 | 调用次数 |",
        "| --- | --- |",
    ]
    for day, calls in a["calls_by_day"].items():
        lines.append(f"| {day} | {calls} |")
    lines += [
        "",
        f"- 单次运行耗时：p50 = **{a['dur_p50']} ms**，p95 = **{a['dur_p95']} ms**，max = {a['dur_max']} ms",
        "",
        "## 3. 报错与告警",
        "",
        f"- 错误事件 {a['error_count']} 条；告警 {a['warn_count']} 条",
    ]
    if a["error_types"]:
        lines.append("")
        lines.append("| 异常类型 | 次数 | 阶段 |")
        lines.append("| --- | --- | --- |")
        for etype, count in a["error_types"].most_common():
            stages = "、".join(sorted({r.get("stage", "?") for r in recs
                                       if r.get("event") == "error" and r.get("error_type") == etype}))
            lines.append(f"| {etype} | {count} | {stages} |")
    if a["warn_msgs"]:
        lines.append("")
        for msg, count in a["warn_msgs"].most_common(5):
            lines.append(f"- 告警 ×{count}：{msg}")

    lines += [
        "",
        "## 4. 抽取质量",
        "",
        f"- 解析通知 {a['parsed_count']} 条，平均置信度 {a['avg_confidence']}",
        f"- 类别分布：{'、'.join(f'{k}×{v}' for k, v in a['categories'].most_common()) or '无'}",
    ]
    if a["miss_rate"]:
        lines += ["", "| 字段 | 漏抽率 |", "| --- | --- |"]
        for field, rate in sorted(a["miss_rate"].items(), key=lambda kv: -kv[1]):
            lines.append(f"| {field} | {rate:.0%} |")

    lines += ["", "## 5. 网页端真实调用（8 位用户使用证据）", ""]
    lines.append(f"- 累计网页解析 **{a['web_count']}** 次，中位耗时 {a['web_dur_p50']} ms，"
                 f"平均置信度 {a['web_avg_confidence']}")
    if a["web_by_day"]:
        lines += ["", "| 日期 | 网页解析次数 |", "| --- | --- |"]
        for day, count in a["web_by_day"].items():
            lines.append(f"| {day} | {count} |")
    else:
        lines.append("- 暂无网页调用记录。让 8 位用户通过 `python run.py web` 使用，"
                     "这里的数字就是他们真实使用的痕迹。")

    lines += ["", "## 6. 建议迭代项（阈值自动判定）", ""]
    for item in suggestions(a):
        lines.append(f"- {item}")

    lines += [
        "",
        "---",
        "",
        "> 本报告由 `tools/log_digest.py` 自动生成，阈值常量见脚本 `THRESHOLDS`。",
        "> 每次迭代前生成一份并随代码一起提交，形成「日志 → 依据 → 改动」的可追溯链。",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="分析运行日志并生成迭代依据")
    parser.add_argument("--last", type=int, default=0, help="只分析最近 N 天的日志文件")
    parser.add_argument("--dir", default="", help="分析指定目录（用于预演模拟数据）")
    args = parser.parse_args(argv)

    log_dir = Path(args.dir).resolve() if args.dir else LOG_DIR
    if not log_dir.exists():
        print(f"[ERROR] 找不到日志目录：{log_dir}", file=sys.stderr)
        return 1

    files = sorted(log_dir.glob("*.jsonl"))
    if not files:
        print(f"[ERROR] {log_dir} 下还没有 .jsonl 日志，先跑一次 run。", file=sys.stderr)
        return 1

    recs = load_records(log_dir)
    if args.last:
        keep = {p.name for p in files[-args.last:]}
        recs = [r for r in recs if r["_file"] in keep]

    simulated = sum(1 for r in recs if r.get("simulated"))

    a = analyse(recs)
    md = render(a, recs, simulated=simulated)

    if log_dir == LOG_DIR:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"log_digest_{datetime.now().strftime('%Y%m%d')}.md"
    else:
        # 预演模式：报告写在源目录旁，不污染 iterations/
        out = log_dir / "digest_preview.md"
    out.write_text(md, encoding="utf-8")

    if simulated:
        print(f"[SIMULATED] 本次分析含 {simulated} 条模拟记录，报告仅供预演，不可作为提交物。")

    print(f"[OK] 诊断报告已生成：{out}")
    print(f"     日志天数 {len(a['run_days'])} | 运行 {a['run_count']} 次 | "
          f"错误 {a['error_count']} | p95 {a['dur_p95']}ms")
    for item in suggestions(a)[:3]:
        print(f"     -> {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
