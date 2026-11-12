#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
课题10 · 课程通知 → 结构化表格：AI 批处理提取器

一条纯本地规则流水线，零网络请求、零模型推理、毫秒级。
它被 run_benchmark.py 计时，用于与人工处理 10 个样本的耗时做对比。

为什么不用"调个大模型 API"来做对照？
因为本课题审计的是**日常高频任务的自动化可行性**。调 API 会引入网络抖动、限流、
按量计费和隐私上传，这些因素会把"耗时"变成一个不可控的随机变量，审计结论就失去意义。
本地规则流水线的耗时是**确定性的、可复现的**，这才是一份合格的耗时审计应有的基准。

用法：
    python scripts/ai_extract.py                     # 输出到 stdout
    python scripts/ai_extract.py --out result.csv    # 输出到 CSV
    python scripts/ai_extract.py --anchor 2026-11-12 --term-start 2026-09-07
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import date, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# 可配置基准（教学周与相对时间都依赖它）
# ---------------------------------------------------------------------------
DEFAULT_ANCHOR = date(2026, 11, 12)      # 「今天」
DEFAULT_TERM_START = date(2026, 9, 7)    # 第 1 周周一

WEEKDAY = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}

# 节次 → (开始, 结束)。各校不同，这里用信通学院常用作息。
SECTIONS = {
    1: ("08:20", "09:00"), 2: ("09:05", "09:45"),
    3: ("10:00", "10:40"), 4: ("10:45", "11:25"),
    5: ("14:00", "14:40"), 6: ("14:45", "15:25"),
    7: ("15:40", "16:20"), 8: ("16:25", "17:05"),
    9: ("19:00", "19:40"), 10: ("19:45", "20:25"),
}

# 没有书名号时的课程词典（校园常见课程 / 活动）
COURSE_DICT = [
    "高等数学", "线性代数", "大学物理", "大学英语", "体育", "形势与政策",
    "通信原理", "数字信号处理", "模拟电子技术实验", "模拟电子技术", "数字电子技术",
    "单片机原理与应用", "嵌入式系统设计", "传感器技术", "C语言程序设计",
    "人工智能导论", "数据结构", "计算机网络", "操作系统", "班会", "年级大会",
]

CN_DIGIT = {"零": 0, "〇": 0, "一": 1, "两": 2, "二": 2, "三": 3, "四": 4,
            "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def cn_num(s: str) -> int | None:
    """把「十」「十一」「二十三」这类中文数字转成 int（只需覆盖 0–59）。"""
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    if s in CN_DIGIT:
        return CN_DIGIT[s]
    if "十" in s:
        head, _, tail = s.partition("十")
        tens = CN_DIGIT.get(head, 1) if head else 1
        ones = CN_DIGIT.get(tail, 0) if tail else 0
        return tens * 10 + ones
    return None


def normalize(s: str) -> str:
    """全角→半角、统一分隔符，让后续正则只处理一种形态。"""
    out = []
    for ch in s:
        o = ord(ch)
        if 0xFF01 <= o <= 0xFF5E:
            out.append(chr(o - 0xFEE0))
        elif ch == "\u3000":
            out.append(" ")
        else:
            out.append(ch)
    t = "".join(out)
    for a, b in (("：", ":"), ("－", "-"), ("—", "-"), ("–", "-"),
                 ("～", "~"), ("至", "到"), ("礼拜", "周"), ("星期", "周")):
        t = t.replace(a, b)
    return t


# ---------------------------------------------------------------------------
# 日期解析
# ---------------------------------------------------------------------------

def resolve_date(text: str, anchor: date, term_start: date, mode: str = "last") -> tuple[str, str]:
    """返回 (yyyy-mm-dd, 原始表达)。解析不出返回 ("", "")。

    两个关键启发式，都是被真实样本逼出来的：

    1. **mode="last"**：一般通知里「原定A 调整至B」生效的是 B，所以同一篇里多个日期时取最后一个。
    2. **mode="first"**：但在「调整至」之后的片段里，第一个日期就是生效日期，再往后取会撞上
       文末落款日期（如「2026年11月10日」）——这是初次实现时踩到的坑。
    """
    def _pick(cands):
        if not cands:
            return None
        cands.sort(key=lambda x: x[0])
        return cands[0] if mode == "first" else cands[-1]

    # ① 绝对日期：11月20日 / 11月20号
    cands = []
    for m in re.finditer(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]", text):
        try:
            cands.append((m.start(), date(anchor.year, int(m.group(1)), int(m.group(2))), m.group(0)))
        except ValueError:
            continue
    hit = _pick(cands)
    if hit:
        return hit[1].isoformat(), hit[2]

    # ② 教学周：第12周周二
    cands = []
    for m in re.finditer(r"第\s*(\d{1,2})\s*周\s*周\s*([一二三四五六日天])", text):
        cands.append((m.start(),
                      term_start + timedelta(weeks=int(m.group(1)) - 1, days=WEEKDAY[m.group(2)]),
                      m.group(0)))
    hit = _pick(cands)
    if hit:
        return hit[1].isoformat(), hit[2]

    # ③ 相对周：下周三 / 这个周五 / 周三
    cands = []
    for m in re.finditer(r"(下下|下|本|这|上)?\s*周\s*([一二三四五六日天])", text):
        prefix, wd_ch = m.group(1), m.group(2)
        this_mon = anchor - timedelta(days=anchor.weekday())
        offset = {"下下": 2, "下": 1, "本": 0, "这": 0, "上": -1}.get(prefix or "", 0)
        cands.append((m.start(), this_mon + timedelta(weeks=offset, days=WEEKDAY[wd_ch]), m.group(0)))
    hit = _pick(cands)
    if hit:
        return hit[1].isoformat(), hit[2]

    # ④ 明天 / 今天 / 后天
    cands = []
    for m in re.finditer(r"(大后天|后天|明天|今天|昨天)", text):
        delta = {"大后天": 3, "后天": 2, "明天": 1, "今天": 0, "昨天": -1}[m.group(1)]
        cands.append((m.start(), anchor + timedelta(days=delta), m.group(0)))
    hit = _pick(cands)
    if hit:
        return hit[1].isoformat(), hit[2]

    return "", ""


# ---------------------------------------------------------------------------
# 时间解析
# ---------------------------------------------------------------------------

def _ampm_before(text: str, pos: int) -> str:
    """看时间表达前面有没有 上午/下午/晚上 修饰。

    窗口取 25 字符而不是 8：语音转写和闲聊体里「下午」离「三点半」隔着十几个字
    （「明天下午的班会改到教学楼441 三点半开始」），窗口太小就漏判，把 15:30 记成 03:30。
    """
    window = text[max(0, pos - 25):pos]
    base = max(0, pos - 25)
    # 必须取**离时间表达最近**的那个修饰词，不能按关键词表顺序取第一个命中。
    # 反例（N07 真实踩过）：「本来上午的 现在改成下午三点开始」——窗口里「上午」在前、
    # 「下午」在后，按固定顺序返回「上午」就把 15:00 记成了 03:00。
    best: tuple[int, str] | None = None
    for kw in ("傍晚", "晚上", "晚间", "凌晨", "早上", "上午", "中午", "下午"):
        i = window.rfind(kw)
        if i >= 0:
            end = base + i + len(kw)
            if best is None or end > best[0]:
                best = (end, kw)
    return best[1] if best else ""


def _apply_ampm(hour: int, minute: int, ampm: str) -> tuple[int, int]:
    if ampm in ("下午", "傍晚", "晚上", "晚间") and hour < 12:
        hour += 12
    if ampm == "中午" and hour < 12 and hour != 12:
        hour += 12 if hour <= 2 else 0
    if ampm == "晚上" and hour == 12:
        return 23, 59          # 「晚上12点」按当天结束处理
    if hour == 24:
        return 23, 59
    return hour % 24, minute


def _fmt(h: int, m: int) -> str:
    return f"{h:02d}:{m:02d}"


def resolve_time(text: str) -> tuple[str, str, str]:
    """返回 (start, end, 原始片段)。按优先级择一，避免多处时间混取。"""
    # ① 明确的 HH:MM - HH:MM
    m = re.search(r"(\d{1,2}):(\d{2})\s*[-~到]\s*(\d{1,2}):(\d{2})", text)
    if m:
        ampm = _ampm_before(text, m.start())
        h1, m1 = _apply_ampm(int(m.group(1)), int(m.group(2)), ampm)
        h2, m2 = _apply_ampm(int(m.group(3)), int(m.group(4)), ampm)
        return _fmt(h1, m1), _fmt(h2, m2), m.group(0)

    # ② 中文时段：下午两点到三点四十
    m = re.search(
        r"([零〇一两二三四五六七八九十]{1,3})\s*点\s*(?:([零〇一两二三四五六七八九十]{1,3})\s*分?|半)?"
        r"\s*[-~到]\s*"
        r"([零〇一两二三四五六七八九十]{1,3})\s*点\s*(?:([零〇一两二三四五六七八九十]{1,3})\s*分?|半)?",
        text)
    if m:
        ampm = _ampm_before(text, m.start())
        h1 = cn_num(m.group(1))
        m1 = 30 if m.group(2) == "半" else (cn_num(m.group(2)) or 0)
        h2 = cn_num(m.group(3))
        m2 = 30 if m.group(4) == "半" else (cn_num(m.group(4)) or 0)
        if h1 is not None and h2 is not None:
            H1, M1 = _apply_ampm(h1, m1, ampm)
            H2, M2 = _apply_ampm(h2, m2, ampm)
            # 「下午两点到三点」：结束时刻小于开始时刻时补 12 小时
            if (H2, M2) <= (H1, M1) and ampm in ("下午", "晚上", "傍晚"):
                H2 = (H2 + 12) % 24
            return _fmt(H1, M1), _fmt(H2, M2), m.group(0)

    # ③ 单个 HH:MM
    m = re.search(r"(\d{1,2}):(\d{2})", text)
    if m:
        ampm = _ampm_before(text, m.start())
        h, mi = _apply_ampm(int(m.group(1)), int(m.group(2)), ampm)
        return _fmt(h, mi), "", m.group(0)

    # ④ 中文单点：下午三点半 / 三点
    m = re.search(r"([零〇一两二三四五六七八九十]{1,3})\s*点\s*(半|([零〇一两二三四五六七八九十]{1,3})\s*分)?", text)
    if m:
        ampm = _ampm_before(text, m.start())
        h = cn_num(m.group(1))
        mi = 30 if m.group(2) and m.group(2).startswith("半") else (cn_num(m.group(3)) or 0 if m.group(3) else 0)
        if h is not None:
            H, M = _apply_ampm(h, mi, ampm)
            return _fmt(H, M), "", m.group(0)

    # ⑤ 节次：3-4节 / 第5-6节 / 第3节
    #    取最后一个：「第5-6节课调整至…第3-4节」生效的是后者。
    secs = [(m.start(), int(m.group(1)), int(m.group(2)))
            for m in re.finditer(r"(\d{1,2})\s*[-~到]\s*(\d{1,2})\s*节", text)]
    if secs:
        _, a, b = secs[-1]
        if a in SECTIONS and b in SECTIONS:
            return SECTIONS[a][0], SECTIONS[b][1], f"{a}-{b}节"
    singles = [(m.start(), int(m.group(1))) for m in re.finditer(r"第?\s*(\d{1,2})\s*节", text)]
    if singles:
        _, a = singles[-1]
        if a in SECTIONS:
            return SECTIONS[a][0], SECTIONS[a][1], f"{a}节"

    return "", "", ""


# ---------------------------------------------------------------------------
# 地点 / 课程 / 动作 / 截止时间
# ---------------------------------------------------------------------------

LOC_PATTERNS = [
    re.compile(r"(教学楼|信通楼|实验楼|综合楼|图书馆|体育馆|食堂|操场|实训楼)\s*([A-Za-z]{0,2}\s*\d{2,3})?"),
    re.compile(r"(?<![A-Za-z\u4e00-\u9fa5])(信|教|实)\s*([A-Za-z]?\d{2,3})"),
    re.compile(r"(\d{3})\s*(实践工坊|工坊|报告厅|机房|教室|实验室)"),
]


def resolve_location(text: str) -> str:
    """取**最后**一个地点：改地点场景里生效的是后出现的那个。

    两个坑：
    1. 括注干扰 —— 「教学楼441（426工坊旁）」里的「426工坊」是方位说明，不是场地。
       所以匹配前先删掉括号内容。
    2. 短匹配抢占 —— 「教学楼441报告厅」会被 `(\\d{3})报告厅` 二次命中，导致丢掉「教学楼」前缀。
       若短匹配紧跟在「楼」字后面，就跳过它。
    """
    cleaned = re.sub(r"[（(][^）)]*[）)]", "", text)
    found: list[tuple[int, str]] = []

    for m in LOC_PATTERNS[0].finditer(cleaned):
        token = re.sub(r"\s+", "", m.group(1) + (m.group(2) or "")).upper()
        found.append((m.start(), token))

    for m in LOC_PATTERNS[1].finditer(cleaned):
        token = re.sub(r"\s+", "", m.group(1) + m.group(2)).upper()
        found.append((m.start(), token))

    for m in LOC_PATTERNS[2].finditer(cleaned):
        prev = cleaned[max(0, m.start() - 1):m.start()]
        if prev in ("楼", "馆"):
            continue                      # 已被长匹配覆盖，跳过
        found.append((m.start(), re.sub(r"\s+", "", m.group(0)).upper()))

    if not found:
        return ""
    found.sort()
    return found[-1][1]


def resolve_course(text: str) -> str:
    for m in re.finditer(r"《([^》]{1,24})》", text):
        name = m.group(1).strip()
        if 1 < len(name) <= 24:
            return name
    # 无书名号 → 词典兜底，取文中最早出现的一个
    hits = [(text.find(c), c) for c in COURSE_DICT if c in text]
    hits = [h for h in hits if h[0] >= 0]
    if hits:
        hits.sort()
        return hits[0][1]
    return ""


def resolve_action(text: str) -> str:
    # ⚠️ 「会议」不能用裸关键词，否则「改为线上（腾讯会议）」会被误判成会议通知。
    rules = [
        ("停课", ("停课", "取消", "暂停")),
        ("补考", ("补考", "重修考试")),
        ("考试", ("考试", "测验", "考核", "笔试")),
        ("讲座", ("讲座", "报告会", "分享会", "宣讲")),
        ("会议", ("班会", "年级大会", "开班会", "召开会议", "会议通知", "开会")),
        ("调课", ("调整", "调课", "改到", "改为", "改成", "顺延", "换成", "调到", "变更")),
        ("上课", ("上课", "照常", "正常进行", "正常")),
    ]
    for label, kws in rules:
        if any(k in text for k in kws):
            return label
    return "未识别"


DEADLINE_HINT = ("截止", "前交", "之前", "前扫描", "前提交", "前上传", "前完成", "ddl")


def resolve_deadline(text: str, anchor: date, term_start: date) -> str:
    for kw in DEADLINE_HINT:
        idx = text.find(kw)
        if idx < 0:
            continue
        # 窗口取「前 40 + 后 30」：中文句式常把日期放在关键词之后
        # （「作业截止时间也往后推一天 就周日晚上吧」），后窗口太窄会取不到日期。
        seg = text[max(0, idx - 40): idx + 30]
        d, _ = resolve_date(seg, anchor, term_start, mode="last")
        if not d:
            continue
        if re.search(r"\d{1,2}:\d{2}", seg):
            t_start, _, _ = resolve_time(seg.replace("截止", " "))
            if t_start and t_start != "23:59":
                return f"{d} {t_start}"
        return d
    return ""


# ---------------------------------------------------------------------------
# 事件拆分与主流程
# ---------------------------------------------------------------------------

def split_events(text: str) -> list[str]:
    """编号列表（1. 2. 3.）视为独立事件；否则整篇一个事件。"""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    numbered = [ln for ln in lines if re.match(r"^\d{1,2}\s*[.、)）]", ln)]
    if len(numbered) >= 2:
        return numbered
    return ["\n".join(lines)]


# 「截止/报名」类句子只影响 deadline 字段，不应当污染事件本身的日期与地点。
# 例：N10 的活动在 11月24日，文末「报名截止 11月20日」如果混进来，主日期就取错了。
DEADLINE_HINT_LOCAL = ("截止", "前交", "之前", "前扫描", "前提交", "前上传", "前完成", "ddl")

# 「调整至 / 顺延至 / 改到」之后的文本才是生效信息
EFFECTIVE_VERB = re.compile(
    r"(?:调整到|调整至|调整成|调整为|改到|改为|改成|更改为|顺延到|顺延至|调到|定于|搬至|搬至)")


def strip_deadline_clauses(text: str) -> str:
    parts = re.split(r"(?<=[。!?\n])", text)
    keep = [p for p in parts if p.strip() and not any(k in p for k in DEADLINE_HINT_LOCAL)]
    return "".join(keep) if keep else text


def extract_one(raw: str, anchor: date, term_start: date) -> dict:
    text = normalize(raw)
    main = strip_deadline_clauses(text)

    # 生效片段：优先在「调整至…」之后取第一个日期（那里才是改期后的时间）
    m = EFFECTIVE_VERB.search(main)
    after = main[m.end():] if m else ""
    d_iso, d_raw = ("", "")
    if after:
        d_iso, d_raw = resolve_date(after, anchor, term_start, mode="first")
    if not d_iso:
        d_iso, d_raw = resolve_date(main, anchor, term_start, mode="last")

    t_start, t_end, t_raw = resolve_time(main)
    if not t_start and after:
        t_start, t_end, t_raw = resolve_time(after)

    return {
        "course": resolve_course(main),
        "date_iso": d_iso,
        "date_raw": d_raw,
        "time_start": t_start,
        "time_end": t_end,
        "time_raw": t_raw,
        "location": resolve_location(main),
        "deadline": resolve_deadline(text, anchor, term_start),
        "action": resolve_action(main),
    }


def run(samples_dir: Path, anchor: date, term_start: date) -> list[dict]:
    rows = []
    for p in sorted(samples_dir.glob("N*.txt")):
        raw = p.read_text(encoding="utf-8")
        events = split_events(raw)
        for i, ev in enumerate(events, 1):
            rec = {"file": p.name, "evt": i}
            rec.update(extract_one(ev, anchor, term_start))
            rows.append(rec)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--samples-dir", default=str(root / "samples"))
    ap.add_argument("--out", default="")
    ap.add_argument("--anchor", default=str(DEFAULT_ANCHOR))
    ap.add_argument("--term-start", default=str(DEFAULT_TERM_START))
    args = ap.parse_args()

    anchor = date.fromisoformat(args.anchor)
    term_start = date.fromisoformat(args.term_start)
    rows = run(Path(args.samples_dir), anchor, term_start)

    cols = ["file", "evt", "course", "date_iso", "date_raw", "time_start", "time_end",
            "time_raw", "location", "deadline", "action"]
    if args.out:
        with open(args.out, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"已写出 {len(rows)} 条 -> {args.out}")
    else:
        w = csv.DictWriter(sys.stdout, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
