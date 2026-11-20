"""输出层：Markdown 日报 + CSV 台账。

日报是用户真正每天看的东西，也是"8 位真实用户反馈"最容易产生有效意见的载体。
"""
from __future__ import annotations

import csv
from datetime import date, datetime, timedelta
from pathlib import Path

from . import config, store

CSV_HEADER = ["事件日期", "时间", "标题", "地点", "需携带", "截止时间", "类别", "优先级", "状态", "首次发现"]

_PRIORITY_LABEL = {
    "P0": "🔴 24h 内",
    "P1": "🟠 3 天内",
    "P2": "🟡 本周",
    "P3": "⚪ 无期限",
    "P4": "⚫ 已截止",
}


def _fmt_row(row) -> str:
    bits = [f"**{row['title']}**"]
    when = " ".join(x for x in [row["event_date"] or "", row["event_time"] or ""] if x).strip()
    meta = []
    if when:
        meta.append(f"🕒 {when}")
    if row["place"]:
        meta.append(f"📍 {row['place']}")
    if row["bring"]:
        meta.append(f"🎒 带 {row['bring']}")
    if row["deadline"]:
        meta.append(f"⏳ 截止 {row['deadline']}")
    meta.append(_PRIORITY_LABEL.get(row["priority"], row["priority"]))
    if meta:
        bits.append(" · ".join(meta))
    return "- " + "\n  ".join(bits)


def build_markdown(rows, *, day: date, run_meta: dict) -> str:
    today = day
    upcoming_end = today + timedelta(days=config.UPCOMING_DAYS)
    week_end = today + timedelta(days=7)

    buckets: dict[str, list] = {"today": [], "upcoming": [], "week": [], "later": [], "nodate": [], "overdue": []}
    for row in rows:
        if not row["event_date"]:
            buckets["nodate"].append(row)
            continue
        try:
            ed = datetime.strptime(row["event_date"], "%Y-%m-%d").date()
        except ValueError:
            buckets["nodate"].append(row)
            continue
        if ed < today:
            buckets["overdue"].append(row)
        elif ed == today:
            buckets["today"].append(row)
        elif ed <= upcoming_end:
            buckets["upcoming"].append(row)
        elif ed <= week_end:
            buckets["week"].append(row)
        else:
            buckets["later"].append(row)

    lines = [
        f"# 今日待办 · {today.isoformat()}",
        "",
        f"> 由 NoticeCatcher v{config.APP_VERSION} 自动生成于 {datetime.now().strftime('%H:%M:%S')}",
        f"> 本次运行：解析 {run_meta.get('notices', 0)} 条通知，新增 {run_meta.get('items_new', 0)} 条，"
        f"去重 {run_meta.get('items_dup', 0)} 条，耗时 {run_meta.get('dur_ms', 0):.0f} ms",
        "",
    ]

    def section(title: str, key: str, hint: str = "") -> None:
        items = buckets[key]
        if not items:
            return
        lines.append(f"## {title}（{len(items)}）")
        if hint:
            lines.append(f"_{hint}_")
        lines.append("")
        for row in items:
            lines.append(_fmt_row(row))
        lines.append("")

    section("今天", "today")
    section(f"未来 {config.UPCOMING_DAYS} 天", "upcoming")
    section("本周稍后", "week")
    section("需要人工确认时间", "nodate", "程序没能从原文里找到明确日期，建议手动补一下")
    section("已过期未处理", "overdue", "如果已经完成，可忽略；如果是漏了，值得复盘")

    total = len(rows)
    lines += [
        "---",
        "",
        f"共 {total} 条未完成待办。原始通知全文不入库，仅保留已脱敏摘录（见 `output/excerpts/`）。",
    ]
    return "\n".join(lines)


def write_daily(rows, *, day: date, run_meta: dict) -> Path:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.OUTPUT_DIR / f"{day.isoformat()}.md"
    path.write_text(build_markdown(rows, day=day, run_meta=run_meta), encoding="utf-8")
    return path


def write_ledger(conn) -> Path:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(conn.execute("SELECT * FROM items ORDER BY event_date, event_time"))
    with config.LEDGER_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)
        for r in rows:
            writer.writerow([
                r["event_date"] or "", r["event_time"] or "", r["title"], r["place"] or "",
                r["bring"] or "", r["deadline"] or "", r["category"], r["priority"],
                r["status"], (r["first_seen"] or "")[:10],
            ])
    return config.LEDGER_CSV


def write_excerpts(conn, day: date) -> Path:
    """脱敏原文摘录——过程留痕用，不含手机号/学号/本机路径。"""
    folder = config.OUTPUT_DIR / "excerpts"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{day.isoformat()}.md"
    rows = list(conn.execute(
        "SELECT title, raw_excerpt, source FROM items WHERE first_seen LIKE ? ORDER BY first_seen",
        (f"{day.isoformat()}%",),
    ))
    lines = [f"# 输入摘录（已脱敏）· {day.isoformat()}", ""]
    if not rows:
        lines.append("_今日无新增通知。_")
    for i, r in enumerate(rows, 1):
        lines += [f"## {i}. {r['title']}", "", f"来源：{r['source']}", "", f"> {r['raw_excerpt']}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
