"""七天之窗的日志模拟器（仅供预演，不是真实遥测）。

用途有三个，都是工程上的正经需求：
  1. 预演 11-26 那份 7 天在 `log_digest.py` 里长什么样（阈值、p95、断档检测有没有问题）；
  2. 给 README / 答辩截图提供"数据量级"的视觉参考；
  3. 验证指标算法在异常分布下不会崩（比如某天 0 通知、某天耗时尖刺）。

**它不是遥测数据，也不能变成遥测数据。** 三重标记：
  - 文件名前缀 `SIM-`，不匹配 `YYYY-MM-DD.jsonl` 的命名规则；
  - 每条记录带 `"simulated": true`；
  - 默认输出到 `docs/simulated_logs/`，且该目录在 .gitignore 中。

生成器本身也拒绝写入 process_logs/ —— 见 main() 里的断言。

用法：
    python tools/simulate_logs.py                       # 7 天，止于今天
    python tools/simulate_logs.py --days 7 --end 2026-11-26
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from datetime import date, datetime, time, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "simulated_logs"   # 02-20Days-Challenge/docs/simulated_logs
REAL_LOG_DIR = ROOT / "process_logs"

RUN_TIMES = ((8, 0), (12, 30), (21, 30))
PLACES = ("信通学院431实训室", "学生活动中心302", "图书馆三楼", "429机房", "西门")
CATEGORIES = ("作业", "报名", "活动", "会议", "讲座", "考试")


def _notice_count(day: date) -> int:
    """工作日通知多、周末少 —— 真实日志一定有这种波动，平均分布的日志反而可疑。"""
    if day.weekday() >= 5:
        return random.randint(2, 5)
    return random.randint(7, 14)


def _record(run_id: str, day: date, ts: datetime, event: str, **fields) -> dict:
    rec = {
        "schema": "nc.telemetry/1.0",
        "simulated": True,          # ← 关键标记
        "run_id": run_id,
        "app_version": "0.1.0",
        "ts": ts.astimezone().isoformat(timespec="seconds"),
        "event": event,
        "status": "ok",
    }
    rec.update(fields)
    return rec


def build_day(day: date, rng: random.Random) -> list[dict]:
    """造一天的记录。故意留下可解释的"瑕疵"，用来检验诊断报告能不能发现它们。"""
    records: list[dict] = []
    total_notices = _notice_count(day)

    for idx, (hour, minute) in enumerate(RUN_TIMES):
        ts = datetime.combine(day, time(hour, minute))
        run_id = uuid.uuid4().hex[:8]
        records.append(_record(run_id, day, ts, "run_start",
                               event_date=day.isoformat(), host_hash="simulate",
                               py="3.13.14", tz="CST"))

        # 第 3 天的中午那次故意空跑：模拟"那天确实没新通知"
        empty_run = (idx == 1 and day.day % 7 == 3)
        batch = 0 if empty_run else max(1, total_notices // len(RUN_TIMES))
        if idx == len(RUN_TIMES) - 1 and not empty_run:
            batch += total_notices % len(RUN_TIMES)

        if batch == 0:
            records.append(_record(run_id, day, ts, "inbox_scan", files=0))
            records.append(_record(run_id, day, ts, "warning", status="warn",
                                   stage="inbox_scan", msg="收件箱为空，本次仅更新日报与日志"))
        else:
            records.append(_record(run_id, day, ts, "inbox_scan", files=1))
            records.append(_record(run_id, day, ts, "read_text", file=f"{day.isoformat()}.txt",
                                   bytes=rng.randint(300, 1600), dur_ms=round(rng.uniform(0.1, 0.6), 3)))

            # 第 2 天截图通路缺依赖 —— 这就是 v0.2.0 的真实由头
            if day.day == 21 and idx == 0:
                records.append(_record(run_id, day, ts, "error", status="error", stage="ocr",
                                       error_type="EngineMissing",
                                       msg="pytesseract/Pillow 未安装，截图通路跳过",
                                       file="shot.png", dur_ms=round(rng.uniform(0.2, 0.5), 3)))

            records.append(_record(run_id, day, ts, "split_notices", file=f"{day.isoformat()}.txt",
                                   notices=batch))

            for _ in range(batch):
                dur = round(rng.uniform(0.08, 0.35), 3)
                records.append(_record(run_id, day, ts, "extract_notice", dur_ms=dur, source="text"))
                # 前 3 天 deadline 漏抽偏多（规则没补全），第 4 天起下降
                miss_rate = 0.45 if day.day <= 23 else 0.15
                missed = [f for f in ("deadline", "bring", "place", "event_time")
                          if rng.random() < miss_rate]
                found = 5 - len(missed)
                records.append(_record(
                    run_id, day, ts, "notice_parsed",
                    category=rng.choice(CATEGORIES),
                    priority=rng.choice(("P0", "P1", "P1", "P2", "P3")),
                    confidence=round(found / 5, 3),
                    fields_found=found,
                    fields_missed=",".join(missed),
                    item_key=uuid.uuid4().hex[:12],
                ))

            new = max(1, int(batch * rng.uniform(0.55, 0.8)))
            dup = batch - new
            records.append(_record(run_id, day, ts, "store_upsert",
                                   count=batch, dur_ms=round(rng.uniform(2.5, 7.5), 3)))
            records.append(_record(run_id, day, ts, "dedup", new=new, dup=dup,
                                   ratio=round(dup / batch, 3)))

        # 耗时基线 ~80ms；第 6 天有个尖刺，用来验证 p95 报警
        base = 300.0 if (day.day == 25 and idx == 2) else rng.uniform(60, 110)
        records.append(_record(run_id, day, ts, "write_report", dur_ms=round(rng.uniform(20, 45), 1)))
        records.append(_record(run_id, day, ts, "write_ledger", dur_ms=round(rng.uniform(3, 8), 1)))
        records.append(_record(run_id, day, ts, "write_excerpts", dur_ms=round(rng.uniform(2, 6), 1)))

        total_dur = round(base + rng.uniform(5, 20), 1)
        calls = 4 + batch * 2 + (2 if batch else 1)
        records.append(_record(
            run_id, day, ts, "run_end",
            dur_ms=total_dur, calls=calls,
            errors=1 if (day.day == 21 and idx == 0) else 0,
            warnings=1 if batch == 0 else 0,
            counters={"inbox_scan": 1, "notice_parsed": batch, "extract_notice": batch},
            exit_code=0,
        ))
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成带 simulated 标记的日志样例（不可作为提交物）")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--end", default=date.today().isoformat(), help="最后一天，YYYY-MM-DD")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=20261120, help="固定随机种子，便于复现")
    args = parser.parse_args(argv)

    out_dir = Path(args.out).resolve()
    # 硬约束：绝不写进真实日志目录
    if out_dir == REAL_LOG_DIR.resolve() or REAL_LOG_DIR.resolve() in out_dir.parents:
        print("[REFUSED] 拒绝写入 process_logs/ —— 模拟数据不能混进真实遥测目录。", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    end = date.fromisoformat(args.end)
    start = end - timedelta(days=args.days - 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for offset in range(args.days):
        day = start + timedelta(days=offset)
        records = build_day(day, rng)
        path = out_dir / f"SIM-{day.isoformat()}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        total += len(records)
        print(f"  {path.name}  {len(records):>3} 条")

    (out_dir / "README.md").write_text(
        "# 模拟日志（非真实遥测）\n\n"
        f"由 `tools/simulate_logs.py --days {args.days} --end {args.end} --seed {args.seed}` 生成，"
        f"覆盖 {start} ~ {end}，共 {total} 条记录。\n\n"
        "**这些文件不是运行数据，不能作为任何提交物。**\n\n"
        "- 文件名带 `SIM-` 前缀，不匹配 `YYYY-MM-DD.jsonl`；\n"
        "- 每条记录含 `\"simulated\": true`；\n"
        "- 本目录已在 `.gitignore` 中，不会进入仓库。\n\n"
        "用途：预演 `log_digest.py` 在 7 天数据下的报告形态、验证阈值与算法。\n",
        encoding="utf-8",
    )
    print(f"\n[OK] {args.days} 天 / {total} 条 -> {out_dir}")
    print("     这些是模拟数据，不能作为提交物。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
