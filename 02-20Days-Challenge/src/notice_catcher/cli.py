"""NoticeCatcher 命令行入口。

    python -m notice_catcher run       # 每日一键：收件箱 -> 抽取 -> 入库 -> 日报 -> 日志
    python -m notice_catcher status    # 看历史运行与待办概览
    python -m notice_catcher digest    # 由日志生成诊断报告（迭代依据）
    python -m notice_catcher selftest  # 抽取规则自测，不写生产日志
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import unicodedata
from datetime import date, datetime
from pathlib import Path

from . import config, ocr, report, store
from .extractor import explain_extraction, extract_one, split_notices
from .telemetry import Telemetry, now_iso, sanitize


def _safe_print(text: str) -> None:
    """Windows 控制台编码不可靠时，宁可丢字符也不要崩掉整次运行。"""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("gbk", "replace").decode("gbk", "replace"))


def _width(text: str) -> int:
    """终端显示宽度：中文按 2 格算，用于表格对齐。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _width(text))


# ---------------------------------------------------------------- run

def cmd_run(args: argparse.Namespace) -> int:
    day = date.fromisoformat(args.date) if args.date else date.today()
    tel = Telemetry(day=day.isoformat())
    tel.start()

    started = time.perf_counter()
    stats = {"notices": 0, "items_new": 0, "items_dup": 0, "images": 0, "images_failed": 0}
    notices = []
    conn = None

    try:
        config.INBOX_DIR.mkdir(parents=True, exist_ok=True)
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        sources = sorted(
            p for p in config.INBOX_DIR.iterdir()
            if p.is_file() and not p.name.startswith("_") and p.suffix.lower() in {".txt", ".md", ".png", ".jpg", ".jpeg"}
        )

        if not sources:
            tel.warn("inbox_scan", "收件箱为空，本次仅更新日报与日志", files=0)
            _safe_print("[WARN] 收件箱为空 —— 仍会生成本次运行日志，保证 7 天连续性。")
        else:
            tel.event("inbox_scan", files=len(sources))

        conn = store.connect()

        for path in sources:
            try:
                if ocr.is_image(path):
                    stats["images"] += 1
                    with tel.span("ocr", file=path.name):
                        blob = ocr.image_to_text(path)
                    source_kind = "screenshot"
                else:
                    with tel.span("read_text", file=path.name, bytes=path.stat().st_size):
                        blob = path.read_text(encoding="utf-8", errors="replace")
                    source_kind = "text"

                chunks = split_notices(blob)
                tel.event("split_notices", file=path.name, notices=len(chunks))

                for chunk in chunks:
                    with tel.span("extract_notice", source=source_kind):
                        notice = extract_one(chunk, base=day, source=source_kind)
                    notices.append(notice)
                    tel.event(
                        "notice_parsed",
                        category=notice.category,
                        priority=notice.priority,
                        confidence=notice.confidence,
                        fields_found=len(notice.fields_found),
                        fields_missed=",".join(notice.fields_missed),
                        item_key=notice.key,
                    )

                if not args.keep:
                    target_dir = config.PROCESSED_DIR / day.isoformat()
                    target_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(path), str(target_dir / path.name))

            except ocr.EngineMissing as exc:
                # span 已经写过一条 error，这里只留一条处置记录，避免同一次失败被计两次
                stats["images_failed"] += 1
                tel.event("ocr_skipped", status="warn", file=path.name, fallback="text_path_only")
                _safe_print(f"[WARN] 截图通路不可用，已跳过 {path.name}（详情见日志）")
            except Exception as exc:  # noqa: BLE001 - 单文件失败不应中断整天运行
                tel.error("process_file", exc, file=path.name)
                _safe_print(f"[ERROR] 处理 {path.name} 失败：{type(exc).__name__}")

        stats["notices"] = len(notices)

        if notices:
            with tel.span("store_upsert", count=len(notices)):
                new, dup = store.upsert_notices(conn, notices)
            stats["items_new"], stats["items_dup"] = new, dup
            tel.event("dedup", new=new, dup=dup, ratio=round(dup / max(new + dup, 1), 3))

        rows = store.open_items(conn)
        run_meta = {**stats, "dur_ms": (time.perf_counter() - started) * 1000}

        with tel.span("write_report"):
            md_path = report.write_daily(rows, day=day, run_meta=run_meta)
        with tel.span("write_ledger"):
            csv_path = report.write_ledger(conn)
        with tel.span("write_excerpts"):
            report.write_excerpts(conn, day)

        dur = (time.perf_counter() - started) * 1000
        store.log_run(conn, run_id=tel.run_id, day=day.isoformat(), dur_ms=dur,
                      notices=stats["notices"], items_new=stats["items_new"],
                      items_dup=stats["items_dup"], errors=tel.errors, warnings=tel.warnings)

        tel.finish(exit_code=0, **stats)
        _print_summary(stats, rows, md_path, csv_path, tel, dur)
        return 0

    except Exception as exc:  # noqa: BLE001
        tel.error("run", exc)
        tel.finish(exit_code=1, **stats)
        _safe_print(f"[FATAL] 运行失败：{type(exc).__name__}: {exc}")
        return 1
    finally:
        if conn is not None:
            conn.close()


def _print_summary(stats, rows, md_path, csv_path, tel: Telemetry, dur: float) -> None:
    _safe_print("")
    _safe_print("=" * 56)
    _safe_print(f" NoticeCatcher v{config.APP_VERSION}  run_id={tel.run_id}")
    _safe_print("=" * 56)
    _safe_print(f" 通知 {stats['notices']} 条 -> 新增 {stats['items_new']} / 去重 {stats['items_dup']}")
    _safe_print(f" 未完成待办 {len(rows)} 条    耗时 {dur:.0f} ms    错误 {tel.errors} / 警告 {tel.warnings}")
    _safe_print(f" 日报 {md_path}")
    _safe_print(f" 台账 {csv_path}")
    _safe_print(f" 日志 {tel.path}")
    _safe_print("=" * 56)


# ---------------------------------------------------------------- status

def cmd_status(args: argparse.Namespace) -> int:
    conn = store.connect()
    s = store.stats(conn)
    rows = store.open_items(conn)
    _safe_print(f"累计待办 {s['total']} 条（未完成 {s['open']}，已完成 {s['done']}），历史运行 {s['runs']} 次")
    _safe_print("")
    if not rows:
        _safe_print("（暂无未完成待办，先往 inbox/ 放一条通知试试）")
        conn.close()
        return 0
    for r in rows[:15]:
        when = " ".join(x for x in [r["event_date"] or "--", r["event_time"] or ""] if x).strip()
        _safe_print(f" [{r['priority']}] {when} | {r['title'][:26]} | {r['place'] or '-'}")
    if len(rows) > 15:
        _safe_print(f" ... 另有 {len(rows) - 15} 条")
    conn.close()
    return 0


# ---------------------------------------------------------------- digest

def cmd_digest(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    import log_digest  # type: ignore

    return log_digest.main([])


# ---------------------------------------------------------------- demo

def cmd_demo(args: argparse.Namespace) -> int:
    """把抽取过程摊开给人看：每个字段命中了哪段原文、走的哪条规则。

    不写任何日志、不落库，纯粹用于演示与排障。
    """
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8", errors="replace")
    elif args.text:
        text = " ".join(args.text)
    else:
        text = sys.stdin.read()

    chunks = split_notices(text)
    if not chunks:
        _safe_print("[WARN] 没有解析到有效通知（太短或为空）")
        return 1

    base = date.fromisoformat(args.date) if args.date else date.today()
    _safe_print("")
    _safe_print(f"以 {base.isoformat()}（周{'一二三四五六日'[base.weekday()]}）为基准解析 {len(chunks)} 条通知")
    _safe_print("=" * 78)

    for i, chunk in enumerate(chunks, 1):
        notice, rows = explain_extraction(chunk, base=base)
        _safe_print("")
        _safe_print(f"[通知 {i}] {notice.title}")
        _safe_print("-" * 78)
        for row in rows:
            line = f"  {_pad(row['field'], 10)}{_pad(row['value'], 22)}"
            if row["evidence"]:
                line += f"← {sanitize(row['evidence'], 34)}"
            _safe_print(line.rstrip())
            if row["rule"]:
                _safe_print(f"  {_pad('', 10)}{_pad('', 22)}  规则 {row['rule']}")
        _safe_print(
            f"  置信度 {notice.confidence:.1f}（{len(notice.fields_found)}/5 字段）"
            f"　类别 {notice.category}　优先级 {notice.priority}"
        )

    _safe_print("")
    _safe_print("=" * 78)
    _safe_print("演示模式不写入日志与数据库。要真正入库请用 python run.py run")
    return 0


# ---------------------------------------------------------------- web

def cmd_web(args: argparse.Namespace) -> int:
    """启动本地网页界面（与 run 共用同一套抽取代码）。"""
    sys.path.insert(0, str(config.PROJECT_ROOT / "web"))
    import server  # type: ignore

    argv: list[str] = []
    if args.port:
        argv += ["--port", str(args.port)]
    if args.no_browser:
        argv.append("--no-browser")
    return int(server.main(argv))


# ---------------------------------------------------------------- selftest

def cmd_selftest(args: argparse.Namespace) -> int:
    cases = [
        ("【补课通知】因调课安排，11月22日 14:30 在信通学院431实训室补上《电路分析》，请携带实验报告。",
         {"event_date": "2026-11-22", "event_time": "14:30", "place": "信通学院431实训室", "bring": "实验报告"}),
        ("社团例会改到本周五晚上7点，地点：学生活动中心302，请于11月21日 18:00前在群里接龙报名。",
         {"event_time": "19:00", "deadline": "2026-11-21"}),
        ("明天下午3点前把大作业交到助教邮箱，过期不候。", {"deadline": "2026-11-21 15:00"}),
    ]
    ok = 0
    base = date(2026, 11, 20)
    for text, expect in cases:
        n = extract_one(text, base=base, now=datetime(2026, 11, 20, 12, 0))
        hits = []
        for key, want in expect.items():
            got = getattr(n, key)
            if want is None or (got and str(got).startswith(str(want))):
                hits.append(key)
        passed = len(hits) == len(expect)
        ok += passed
        _safe_print(f"[{'PASS' if passed else 'FAIL'}] conf={n.confidence} "
                    f"{n.event_date} {n.event_time or '--:--'} @ {n.place or '-'} "
                    f"| 截止 {n.deadline or '-'} | {n.category}/{n.priority}")
        if not passed:
            _safe_print(f"       未命中 {[k for k in expect if k not in hits]} 原文：{sanitize(text, 60)}")
    _safe_print(f"\n{ok}/{len(cases)} 通过")
    return 0 if ok == len(cases) else 1


# ---------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="notice_catcher", description="校园通知提取与待办归档工具")
    sub = p.add_subparsers(dest="cmd")

    pr = sub.add_parser("run", help="执行每日流程")
    pr.add_argument("--date", help="按指定日期运行（YYYY-MM-DD），默认今天")
    pr.add_argument("--keep", action="store_true", help="处理完不移动收件箱文件")
    pr.set_defaults(func=cmd_run)

    ps = sub.add_parser("status", help="查看待办与运行概览")
    ps.set_defaults(func=cmd_status)

    pd = sub.add_parser("digest", help="由 process_logs 生成诊断报告")
    pd.set_defaults(func=cmd_digest)

    pm = sub.add_parser("demo", help="演示抽取过程，显示每个字段命中的原文片段")
    pm.add_argument("text", nargs="*", help="直接给出通知原文（可多条）")
    pm.add_argument("--file", help="从文件读取通知原文")
    pm.add_argument("--date", help="解析相对日期的基准日（YYYY-MM-DD）")
    pm.set_defaults(func=cmd_demo)

    pw = sub.add_parser("web", help="启动本地网页界面")
    pw.add_argument("--port", type=int, default=8765)
    pw.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    pw.set_defaults(func=cmd_web)

    pt = sub.add_parser("selftest", help="抽取规则自测")
    pt.set_defaults(func=cmd_selftest)
    return p


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        args = parser.parse_args(["run"])
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
