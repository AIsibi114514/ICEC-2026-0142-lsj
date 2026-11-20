"""规则抽取器：一条通知原文 -> 一个结构化待办。

返回值中的 `fields_found` / `fields_missed` 会写进 telemetry，
这是后续判断"哪个字段最需要改规则"的唯一依据。
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime

from . import normalizer, rules
from .telemetry import digest, sanitize

CORE_FIELDS = ("event_date", "event_time", "place", "deadline", "bring")


@dataclass
class Notice:
    """一条被结构化后的校园通知。"""

    title: str
    event_date: str | None = None       # YYYY-MM-DD
    event_time: str | None = None       # HH:MM
    place: str | None = None
    bring: str | None = None
    deadline: str | None = None         # YYYY-MM-DD HH:MM
    category: str = "其他"
    priority: str = "P3"
    raw_excerpt: str = ""
    source: str = "text"
    fields_found: list[str] = field(default_factory=list)
    fields_missed: list[str] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def key(self) -> str:
        """去重键：标题 + 日期 + 地点，避免同一通知每天被重复计入。"""
        basis = "|".join([self.title, self.event_date or "", self.place or ""])
        return digest(basis)

    def to_row(self) -> dict:
        return asdict(self)


def _clean(text: str) -> str:
    lines = []
    noise = [re.compile(p) for p in rules.NOISE_PATTERNS]
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(p.match(stripped) for p in noise):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def _date_candidates(text: str, base: date) -> list[tuple[int, int, date, str]]:
    found: list[tuple[int, int, date, str]] = []
    seen: set[tuple[int, int]] = set()
    for pattern in (rules.DATE_YMD, rules.DATE_WEEK, rules.DATE_REL, rules.DATE_SLASH):
        for m in re.finditer(pattern, text):
            span = (m.start(), m.end())
            if any(not (span[1] <= s or span[0] >= e) for s, e in seen):
                continue
            parsed = normalizer.parse_date_token(m.group(0), base)
            if parsed:
                seen.add(span)
                found.append((m.start(), m.end(), parsed, m.group(0)))
    return sorted(found)


def _time_candidates(text: str) -> list[tuple[int, int, str]]:
    found: list[tuple[int, int, str]] = []
    seen: list[tuple[int, int]] = []
    for pattern in (rules.TIME_HM, rules.TIME_H):
        for m in re.finditer(pattern, text):
            span = (m.start(), m.end())
            if any(not (span[1] <= s or span[0] >= e) for s, e in seen):
                continue
            parsed = normalizer.parse_time(m.group(0))
            if parsed:
                seen.append(span)
                found.append((m.start(), m.end(), parsed))
    return sorted(found)


def _is_deadline(text: str, start: int, end: int) -> bool:
    left = text[max(0, start - rules.DEADLINE_WINDOW_BEFORE):start]
    right = text[end:end + rules.DEADLINE_WINDOW_AFTER]
    window = left + right
    if any(cue in window for cue in rules.DEADLINE_CUES):
        return True
    return bool(re.search(rules.DEADLINE_TIME_SUFFIX, window))


def _pick_category(text: str) -> str:
    best, best_hits = "其他", 0
    for name, keywords in rules.CATEGORY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text)
        if hits > best_hits:
            best, best_hits = name, hits
    return best


def _pick_priority(deadline: datetime | None, now: datetime) -> str:
    if deadline is None:
        return "P3"
    hours = (deadline - now).total_seconds() / 3600
    if hours < 0:
        return "P4"  # 已截止
    if hours <= rules.PRIORITY_P0_HOURS:
        return "P0"
    if hours <= rules.PRIORITY_P1_HOURS:
        return "P1"
    if hours <= rules.PRIORITY_P2_HOURS:
        return "P2"
    return "P3"


def _pick_title(text: str) -> str:
    bracket = re.search(r"[【\[]([^】\]]{2,40})[】\]]", text)
    if bracket and bracket.group(1).strip() not in rules.TITLE_TAG_WORDS:
        return bracket.group(1).strip()[:40]

    # 【紧急】【通知】这类是标签，跳过方括号本身，从后面的正文取标题
    body = text
    for m in re.finditer(r"[【\[]([^】\]]{1,40})[】\]]", text):
        if m.group(1).strip() in rules.TITLE_TAG_WORDS:
            body = text[m.end():]
            break

    first = next((ln.strip(" -—·:：") for ln in body.splitlines() if ln.strip()), "")
    # 群通知常常一整段没有换行，标题取到第一个句读为止，避免把整段正文当标题
    cut = re.search(r"[。！？；：，,;]", first)
    if cut and cut.start() >= 4:
        first = first[:cut.start()]
    return (first or "未命名通知")[:40]


def _pick_place(text: str) -> str | None:
    m = re.search(rules.PLACE_LABEL, text)
    if m:
        value = _strip_place(m.group(1))
        if value:
            return value[:30]
    # 裸地名容易误抓（"图书馆闭馆通知"），必须带介词或房号才认
    for m in re.finditer(rules.PLACE_ROOM, text):
        raw = m.group(1)
        value = _strip_place(raw)
        if not value:
            continue
        left = text[max(0, m.start() - 3):m.start()]
        has_number = any(ch.isdigit() for ch in value)
        has_prep = any(p in raw for p in rules.PLACE_PREPOSITIONS) or \
                   any(p in left for p in rules.PLACE_PREPOSITIONS)
        if has_number or has_prep:
            return value[:30]
    return None


def _strip_place(value: str) -> str:
    """清洗地名候选。

    两个典型脏数据：
      「在信通学院431实训室」 -> 剥掉开头介词 -> 「信通学院431实训室」
      「本周六上午9点在西门」 -> 候选横跨了时间，取最后一个介词之后 -> 「西门」
    """
    cleaned = value.strip(" ：:，,。、")

    cut_at = -1
    for prep in rules.PLACE_LEADING_NOISE + ("集合",):
        idx = cleaned.rfind(prep)
        if idx > 0:
            cut_at = max(cut_at, idx + len(prep))
    if cut_at > 0:
        cleaned = cleaned[cut_at:].strip(" ：:，,。、")

    changed = True
    while changed:
        changed = False
        for prefix in rules.PLACE_LEADING_NOISE:
            if cleaned.startswith(prefix) and len(cleaned) > len(prefix) + 1:
                cleaned = cleaned[len(prefix):].strip(" ：:，,。、")
                changed = True
    return cleaned


def extract_one(text: str, base: date | None = None, now: datetime | None = None,
                source: str = "text") -> Notice:
    """抽取单条通知。base 为解析相对日期的基准日（默认为今天）。"""
    today = base or date.today()
    current = now or datetime.now()
    cleaned = _clean(text)

    dates = _date_candidates(cleaned, today)
    times = _time_candidates(cleaned)

    deadline_match = next(((s, e, d) for s, e, d, _ in dates if _is_deadline(cleaned, s, e)), None)
    deadline_dt: datetime | None = None
    if deadline_match:
        s, e, d = deadline_match
        # 取距离截止日最近的时间点，而不是文本里最先出现的时间点
        nearby = sorted(
            (t for t in times if min(abs(t[0] - s), abs(t[0] - e)) <= 20),
            key=lambda t: min(abs(t[0] - s), abs(t[0] - e)),
        )
        hhmm = nearby[0][2] if nearby else None
        deadline_dt = normalizer.to_datetime(d, hhmm)

    event_date = next((d for s, e, d, _ in dates if not (deadline_match and d == deadline_match[2])), None)
    if event_date is None and dates:
        event_date = dates[0][2]

    event_hhmm: str | None = None
    if event_date is not None:
        anchor = next(((s, e) for s, e, d, _ in dates if d == event_date), None)
        if anchor:
            times_sorted = sorted(times, key=lambda t: min(abs(t[0] - anchor[0]), abs(t[0] - anchor[1])))
            if times_sorted:
                event_hhmm = times_sorted[0][2]
    elif times:
        event_hhmm = times[0][2]

    bring_match = re.search(rules.BRING, cleaned)
    bring = bring_match.group(1).strip(" ：:，,。") if bring_match else None

    notice = Notice(
        title=_pick_title(cleaned),
        event_date=event_date.isoformat() if event_date else None,
        event_time=event_hhmm,
        place=_pick_place(cleaned),
        bring=bring,
        deadline=deadline_dt.strftime("%Y-%m-%d %H:%M") if deadline_dt else None,
        category=_pick_category(cleaned),
        priority=_pick_priority(deadline_dt, current),
        raw_excerpt=sanitize(cleaned, 240),
        source=source,
    )

    notice.fields_found = [f for f in CORE_FIELDS if getattr(notice, f)]
    notice.fields_missed = [f for f in CORE_FIELDS if not getattr(notice, f)]
    notice.confidence = round(len(notice.fields_found) / len(CORE_FIELDS), 3)
    return notice


def split_notices(blob: str) -> list[str]:
    """把每天的收件箱文本切成独立通知。

    约定：以空行或 `---` 分隔；单条通知内部允许换行。
    """
    parts = re.split(r"\n\s*---+\s*\n|\n\s*\n", blob)
    return [p.strip() for p in parts if len(p.strip()) >= 8]


# ---------------------------------------------------------------- 可解释性

FIELD_LABELS = {
    "event_date": "活动日期",
    "event_time": "活动时间",
    "place": "地点",
    "bring": "需携带",
    "deadline": "截止时间",
}


def _which_date_rule(token: str) -> str:
    """判断命中日期片段的是哪一条规则，便于排障时定位。"""
    if re.search(r"月\s*\d{1,2}\s*[日号]", token):
        return "DATE_YMD"
    if re.search(r"周[一二三四五六日天]", token):
        return "DATE_WEEK"
    if re.fullmatch(r"\d{1,2}\s*[/\-]\s*\d{1,2}", token.strip()):
        return "DATE_SLASH"
    return "DATE_REL"


def explain_extraction(text: str, base: date | None = None, now: datetime | None = None) -> tuple[Notice, list[dict]]:
    """抽取并给出每个字段的"证据"——命中的原文片段与规则名。

    用途：`python run.py demo` 时把黑盒变成白盒，也方便排查漏抽究竟是
    规则没命中，还是命中了但值解析错了。
    """
    today = base or date.today()
    current = now or datetime.now()
    cleaned = _clean(text)
    notice = extract_one(text, base=today, now=current)

    dates = _date_candidates(cleaned, today)
    times = _time_candidates(cleaned)
    evidence: dict[str, tuple[str, str]] = {}
    raw_map: dict[str, str] = {}

    if notice.event_date:
        target = date.fromisoformat(notice.event_date)
        for start, end, parsed, token in dates:
            if parsed == target:
                evidence["event_date"] = (token, _which_date_rule(token))
                break

    if notice.deadline:
        dl = datetime.strptime(notice.deadline, "%Y-%m-%d %H:%M")
        for start, end, parsed, token in dates:
            if parsed != dl.date():
                continue
            window = cleaned[max(0, start - rules.DEADLINE_WINDOW_BEFORE):end + rules.DEADLINE_WINDOW_AFTER]
            if any(cue in window for cue in rules.DEADLINE_CUES):
                cue = next(cue for cue in rules.DEADLINE_CUES if cue in window)
                evidence["deadline"] = (f"{token} + 「{cue}」", "DEADLINE_CUES")
            else:
                evidence["deadline"] = (f"{token} + 时间后缀「前」", "DEADLINE_TIME_SUFFIX")
            # 高亮范围尽量覆盖完整的截止表达（"今晚22:00前"），而不是只有"今晚"
            suffix = re.search(rules.DEADLINE_TIME_SUFFIX, cleaned[end:end + 14])
            if suffix:
                span = cleaned[start:end + suffix.end()].strip()
                if span:
                    raw_map["deadline"] = span
            break

    if notice.event_time:
        for start, end, hhmm in times:
            if hhmm == notice.event_time:
                raw = cleaned[start:end].strip()
                rule = "TIME_HM" if (":" in raw or "：" in raw) else "TIME_H"
                if raw != hhmm:
                    evidence["event_time"] = (f"{raw} → {hhmm}", rule)
                else:
                    evidence["event_time"] = (raw, rule)
                break

    if notice.place:
        idx = cleaned.find(notice.place)
        if idx >= 0:
            # 只回带紧邻的介词（"在信通学院431实训室"），不要把前一句的时间数字卷进来
            prefix = ""
            j = idx - 1
            while j >= 0 and idx - j <= 3 and cleaned[j] in "在于到至去":
                prefix = cleaned[j] + prefix
                j -= 1
            evidence["place"] = (prefix + notice.place, "PLACE_LABEL/PLACE_ROOM")
    else:
        # 抽不到时，指出是被什么规则挡下来的（这是最有用的排障信息）
        for m in re.finditer(rules.PLACE_ROOM, cleaned):
            raw = m.group(1)
            evidence["place"] = (raw, "已命中地名但缺介词/房号，被丢弃")
            break

    if notice.bring:
        bring_match = re.search(rules.BRING, cleaned)
        if bring_match:
            # 从触发词本身开始展示（"请携带实验报告"），比从逗号开始更容易看懂
            evidence["bring"] = (cleaned[bring_match.start():bring_match.end(1)], "BRING")
        else:
            evidence["bring"] = (notice.bring, "BRING")

    rows: list[dict] = []
    for field in ("event_date", "event_time", "place", "bring", "deadline"):
        value = getattr(notice, field)
        token, rule = evidence.get(field, ("", ""))
        rows.append({
            "key": field,
            "field": FIELD_LABELS[field],
            "value": value or "—",
            "raw": raw_map.get(field) or _strip_decoration(token, value),
            "evidence": token,
            "rule": rule if value or token else "",
        })
    return notice, rows


def _strip_decoration(token: str, value: str | None) -> str:
    """从展示用的证据串里还原出"原文片段"，供网页高亮。

    "11月21日 + 「请于」"  -> "11月21日"
    "下午4点 → 16:00"      -> "下午4点"
    """
    if not token:
        return ""
    for sep in (" + ", " → ", "←"):
        if sep in token:
            token = token.split(sep)[0]
    return token.strip()
