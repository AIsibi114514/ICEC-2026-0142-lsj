"""相对时间归一化："本周五晚7点" -> 2026-11-20 19:00。

这是校园通知里最容易出错的一环，也是日志里 date/时间 抽取失败率的主要来源。
所有函数都是纯函数（输入 base 日期），便于单元自测。
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta

from . import rules

_CN_WEEKDAY = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
_REL_OFFSET = {
    "今天": 0, "今日": 0, "今晚": 0, "今晚": 0,
    "明天": 1, "明日": 1, "明晚": 1, "明早": 1,
    "后天": 2, "大后天": 3,
}


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_date_token(token: str, base: date) -> date | None:
    """解析单个已匹配的日期片段。"""
    token = token.strip()

    m = re.search(rules.DATE_YMD, token)
    if m:
        if m.group("y"):
            return _safe_date(int(m.group("y")), int(m.group("m")), int(m.group("d")))
        candidate = _safe_date(base.year, int(m.group("m")), int(m.group("d")))
        # 通知里不带年份且已过去超过 60 天，视为明年
        if candidate and (base - candidate).days > 60:
            candidate = _safe_date(base.year + 1, int(m.group("m")), int(m.group("d")))
        return candidate

    m = re.search(rules.DATE_SLASH, token)
    if m:
        candidate = _safe_date(base.year, int(m.group("m")), int(m.group("d")))
        if candidate and (base - candidate).days > 60:
            candidate = _safe_date(base.year + 1, int(m.group("m")), int(m.group("d")))
        return candidate

    m = re.search(rules.DATE_WEEK, token)
    if m:
        target = _CN_WEEKDAY[m.group("w")]
        prefix = m.group("wk") or ""
        delta = (target - base.weekday()) % 7
        if prefix == "下":
            delta = delta + 7 if delta > 0 else 7
        elif prefix in ("本", "这") and delta == 0:
            delta = 0
        return base + timedelta(days=delta)

    m = re.search(rules.DATE_REL, token)
    if m:
        return base + timedelta(days=_REL_OFFSET.get(m.group("rel"), 0))

    return None


def parse_time(token: str) -> str | None:
    """把时间片段规范化为 HH:MM（24 小时制）。"""
    token = token.strip()

    m = re.search(rules.TIME_HM, token)
    if m:
        hour, minute = int(m.group("h")), int(m.group("mi"))
        hour = _apply_ampm(hour, m.group("ap"))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}"

    m = re.search(rules.TIME_H, token)
    if m:
        hour = int(m.group("h"))
        raw_min = m.group("mi")
        minute = 30 if raw_min == "半" else int(raw_min or 0)
        hour = _apply_ampm(hour, m.group("ap"))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}"

    return None


def _apply_ampm(hour: int, ap: str | None) -> int:
    if ap in ("下午", "晚上", "傍晚") and hour < 12:
        return hour + 12
    if ap == "中午" and hour < 11:
        return hour + 12
    if ap == "凌晨" and hour == 12:
        return 0
    if ap == "上午" and hour == 12:
        return 0
    return hour


def to_datetime(d: date | None, hhmm: str | None) -> datetime | None:
    if d is None:
        return None
    if not hhmm:
        return datetime(d.year, d.month, d.day, 0, 0)
    hh, mm = map(int, hhmm.split(":"))
    return datetime(d.year, d.month, d.day, hh, mm)
