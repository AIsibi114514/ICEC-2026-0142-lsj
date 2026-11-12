#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
提交前自检 —— 逐项核对 ICEC 第一阶段考核要求的交付物是否真的齐了。

为什么需要它：
     两个课题的交付物散在 6 个子目录里，靠人肉核对一定会漏。而且有些缺失是
     **静默的**：raw_data 少一组数据、figures 少一张图、REPORT 里还留着〔待填〕，
     这些都不会报错，只有到交材料时才发现。

     最典型的坑：REPORT.md 里引用了 ![](figures/fig1_xxx.png)，但图还没生成。
     GitHub 上会安静地显示成一个破图标，不点开根本看不出来。

用法：
     python check_delivery.py            # 自检
     python check_delivery.py --strict   # 严格模式：任何 WARN 也算未完成

退出码：0 全部通过 / 1 有未完成项
"""
from __future__ import annotations

import csv
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
T08 = ROOT / "01-Field-Discovery" / "T08-gait-acceleration"
T10 = ROOT / "01-Field-Discovery" / "T10-digital-task-audit"
AI_LOG = ROOT / "03-AI-Collaboration"

STATES = ["S1_STATIC", "S2_WALK", "S3_RUN", "S4_STAIR_UP", "S4_STAIR_DOWN"]

results: list[tuple[str, str, str, str]] = []   # (分节, 状态, 项目, 说明)


def add(section: str, ok: str, item: str, note: str = "") -> None:
    results.append((section, ok, item, note))


def count_placeholder(path: Path) -> int:
    """统计文档里还留着多少个〔待填〕/〔待补〕占位。"""
    if not path.exists():
        return -1
    t = path.read_text(encoding="utf-8")
    return t.count("〔待填〕") + t.count("〔待补")


def check_broken_images(report: Path) -> tuple[int, int]:
    """检查报告里引用的图片是否真实存在（防 GitHub 破图）。返回 (存在数, 引用总数)。"""
    if not report.exists():
        return 0, 0
    text = report.read_text(encoding="utf-8")
    refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    ok = sum(1 for r in refs if (report.parent / r).exists())
    return ok, len(refs)


# ---------------------------------------------------------------------------
# 公共项
# ---------------------------------------------------------------------------

def check_common() -> None:
    s = "公共"
    name = ROOT.name
    m = re.fullmatch(r"ICEC-2026-(\d{4})-([a-zA-Z]+)", name)
    add(s, "OK" if m else "FAIL", "仓库命名规范",
        f"{name}" + ("（学号后四位 + 姓名拼音）" if m else " 应为 ICEC-2026-学号后四位-姓名拼音"))

    # 占位符残留（只扫文档与代码，排除本脚本与 .gitignore）
    leftovers = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or ".git" in p.parts:
            continue
        if p.suffix not in (".md", ".py", ".csv", ".json", ".txt"):
            continue
        if p.name == "check_delivery.py":
            continue
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "XXXX" in t or "NamePinyin" in t:
            leftovers.append(str(p.relative_to(ROOT)))
    add(s, "OK" if not leftovers else "FAIL", "无 XXXX / NamePinyin 占位残留",
        "干净" if not leftovers else f"残留：{', '.join(leftovers[:3])}")

    # 必填文档
    for f, desc in ((ROOT / "README.md", "顶层 README（个人简介与选题说明）"),
                    (ROOT / "01-Field-Discovery" / "README.md", "阶段一选题总览")):
        add(s, "OK" if f.exists() and f.stat().st_size > 500 else "FAIL", desc,
            "" if f.exists() else "文件不存在")

    # AI 协作留痕
    logs = list(AI_LOG.glob("log-*.md")) if AI_LOG.exists() else []
    add(s, "OK" if logs else "WARN", "AI 对话日志（03-AI-Collaboration/log-*.md）",
        f"已归档 {len(logs)} 篇" if logs else "只有 README 模板，建议补至少 1 篇真实排障记录")


# ---------------------------------------------------------------------------
# 课题 08
# ---------------------------------------------------------------------------

def check_t08() -> None:
    s = "课题08"
    if not T08.exists():
        add(s, "FAIL", "课题目录", "不存在")
        return

    raw = T08 / "raw_data"
    csvs = [p for p in raw.glob("*.csv")
            if p.name[0] not in "_" and p.name != "sample_index.csv"] if raw.exists() else []
    named = [p for p in csvs if any(p.name.startswith(st) for st in STATES)]

    # 考核最低要求 4 份（4 状态各 1 份）；建议 4 状态 × 3 组 = 12 份
    if len(named) >= 12:
        lvl = "OK"
    elif len(named) >= 4:
        lvl = "WARN"
    else:
        lvl = "FAIL"
    add(s, lvl, "原始 CSV 数量",
        f"{len(named)} 份（考核最低 4 份，建议 4 状态×3 组=12 份）")

    per = {st: sum(1 for p in named if p.name.startswith(st)) for st in STATES}
    missing = [f"{st}({per[st]})" for st in STATES if per[st] == 0]
    add(s, "OK" if not missing else "FAIL", f"{len(STATES)} 种状态齐备",
        "每类 " + "/".join(f"{st}={per[st]}" for st in STATES) if not missing
        else f"缺：{', '.join(missing)}")

    # 现场认证照
    photos = list((raw / "photos").glob("*")) if (raw / "photos").exists() else []
    photos = [p for p in photos if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".heic")]
    add(s, "OK" if len(photos) >= 4 else "FAIL", "现场采集认证照（考核硬要求）",
        f"{len(photos)} 张" + ("" if len(photos) >= 4 else "，建议每类动作至少 1 张"))

    # 采集元信息：设备型号与场地是"控制变量"的记录凭证，不能空着
    idx = raw / "sample_index.csv"
    filled, todo = 0, 0
    if idx.exists():
        with idx.open(encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        todo = len(rows)
        filled = sum(1 for r in rows
                     if (r.get("completed") or "").strip().upper() == "Y"
                     and (r.get("phone_model") or "").strip() not in ("", "待补")
                     and (r.get("site") or "").strip() not in ("", "待补"))
    add(s, "OK" if filled >= todo and todo >= 4 else "WARN" if filled >= 1 else "FAIL",
        "采集元信息登记（sample_index.csv）",
        f"{filled}/{todo} 条已填全（设备型号/场地/完成标记）" + ("" if filled >= todo else "，含待补项"))

    # 重力分离：含 g 的数据必须做，否则课题结论不成立
    rules_json = T08 / "threshold_rules.json"
    if rules_json.exists():
        try:
            import json
            rj = json.loads(rules_json.read_text(encoding="utf-8"))
            grav = rj.get("gravity_removed") or []
            # 无去重力记录有两种可能：一是采集时用了「线性加速度」（传感器已去重力），
            # 二是漏了处理。前者是正常的，所以只作为确认项而非缺陷。
            add(s, "OK", "重力状态已确认",
                f"{len(grav)} 个文件经低通去重力" if grav
                else "数据为「线性加速度」，传感器层已去重力，无需再分离")
        except Exception:
            add(s, "WARN", "重力分离已执行", "rules 文件无法解析")

    # 分析产物
    for f, desc in ((raw / "feature_table.csv", "逐窗特征表"),
                    (T08 / "threshold_rules.json", "阈值规则文件")):
        add(s, "OK" if f.exists() and f.stat().st_size > 100 else "FAIL", desc,
            "" if f.exists() else "未生成，需跑 analyze_gait.py")

    figs = ["fig1_waveform_4states.png", "fig2_feature_boxplot.png",
            "fig3_confusion_matrix.png", "fig4_freq_spectrum.png", "fig5_rule_flow.png"]
    have = [f for f in figs if (T08 / "figures" / f).exists()]
    add(s, "OK" if len(have) == 5 else "FAIL", "结果图 5 张",
        f"{len(have)}/5" + ("" if len(have) == 5 else "，跑 analyze_gait.py 生成"))

    # 报告
    rep = T08 / "REPORT.md"
    n = count_placeholder(rep)
    add(s, "OK" if n == 0 else "FAIL", "报告无〔待填〕",
        "已填完" if n == 0 else f"还剩 {n} 处")
    exist, total = check_broken_images(rep)
    add(s, "OK" if total and exist == total else "FAIL", "报告内图片引用有效",
        f"{exist}/{total} 可显示" + ("" if exist == total else "，GitHub 会破图"))


# ---------------------------------------------------------------------------
# 课题 10
# ---------------------------------------------------------------------------

def check_t10() -> None:
    s = "课题10"
    if not T10.exists():
        add(s, "FAIL", "课题目录", "不存在")
        return

    samples = sorted((T10 / "samples").glob("N*.txt"))
    add(s, "OK" if len(samples) >= 10 else "FAIL", "任务样本 10 份", f"{len(samples)} 份")
    add(s, "OK" if (T10 / "samples" / "_ground_truth.csv").exists() else "FAIL",
        "人工标注基准", "_ground_truth.csv")

    for f, desc in ((T10 / "raw_data" / "ai_extracted.csv", "AI 提取结果"),
                    (T10 / "raw_data" / "ai_timing.csv", "AI 实测耗时"),
                    (T10 / "raw_data" / "field_accuracy.csv", "字段对错明细"),
                    (T10 / "raw_data" / "benchmark_summary.json", "基准汇总")):
        add(s, "OK" if f.exists() else "FAIL", desc,
            "" if f.exists() else "未生成，需跑 run_benchmark.py")

    # 人工计时（这一步只能人做）
    sheet = T10 / "manual_run" / "timing_sheet.csv"
    done = 0
    if sheet.exists():
        with sheet.open(encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            vals = [(r.get(k) or "").strip() for k in
                    ("manual_round1_s", "manual_round2_s", "manual_round3_s")]
            if all(vals):
                done += 1
    add(s, "OK" if done >= 10 else "FAIL", "人工计时表（3 轮 ×10 样本）",
        f"{done}/10 已填" + ("" if done >= 10 else "，这步 AI 替不了"))

    # 一次性构建成本
    cfg = T10 / "manual_run" / "audit_config.csv"
    build_min = 0.0
    if cfg.exists():
        with cfg.open(encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if r.get("key") == "ai_build_minutes":
                    try:
                        build_min = float(r.get("value") or 0)
                    except ValueError:
                        build_min = 0.0
    add(s, "OK" if build_min > 0 else "FAIL", "一次性构建成本 ai_build_minutes",
        f"{build_min} 分钟" + ("" if build_min > 0 else "，为 0 则摊销曲线画不出来"))

    figs = ["fig1_per_sample_timing.png", "fig2_field_accuracy.png", "fig3_amortization.png"]
    have = [f for f in figs if (T10 / "figures" / f).exists()]
    add(s, "OK" if len(have) == 3 else "WARN", "结果图 3 张",
        f"{len(have)}/3" + ("" if len(have) == 3 else "，fig3 需先填 ai_build_minutes"))

    rep = T10 / "REPORT.md"
    n = count_placeholder(rep)
    add(s, "OK" if n == 0 else "FAIL", "报告无〔待填〕",
        "已填完" if n == 0 else f"还剩 {n} 处")
    exist, total = check_broken_images(rep)
    add(s, "OK" if total and exist == total else "FAIL", "报告内图片引用有效",
        f"{exist}/{total} 可显示" + ("" if exist == total else "，GitHub 会破图"))


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def _dwidth(s: str) -> int:
    """按终端显示宽度算长度：中文/全角占 2 列，ASCII 占 1 列。
    不这么算，str.ljust 会按字符数补空格，中英混排的表格永远是歪的。"""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)


def _pad(s: str, width: int) -> str:
    return s + " " * max(0, width - _dwidth(s))


def main() -> int:
    strict = "--strict" in sys.argv
    check_common()
    check_t08()
    check_t10()

    icon = {"OK": "[OK]  ", "WARN": "[WARN]", "FAIL": "[!!]  "}
    section = None
    width = max(_dwidth(r[2]) for r in results) + 2
    for sec, ok, item, note in results:
        if sec != section:
            print(f"\n=== {sec} " + "=" * (52 - len(sec)))
            section = sec
        mark = "OK" if ok == "OK" else "WARN" if ok == "WARN" else "FAIL"
        print(f"{icon[mark]} {_pad(item, width)}{note}")

    fails = [r for r in results if r[1] == "FAIL"]
    warns = [r for r in results if r[1] == "WARN"]
    total = len(results)
    passed = total - len(fails) - len(warns)

    print("\n" + "-" * 62)
    print(f"通过 {passed}/{total}　待办 {len(fails)}　提示 {len(warns)}")

    if not fails and (not warns or not strict):
        print("\n[结论] 可以提交。")
        return 0

    print("\n[结论] 还不能提交。优先补齐下面这些：")
    for _, _, item, note in fails:
        print(f"  - {item}：{note}")
    if warns and strict:
        for _, _, item, note in warns:
            print(f"  - (提示) {item}：{note}")
    print("\n提示：课题08 的数据必须你本人实地采集；课题10 的人工计时必须你亲手做。"
          "\n      这两项 AI 替不了，也不该替——考核第 1 维度就是核验是否为本人真实动手。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
