"""Telemetry（运行日志）埋点模块。

设计要点（对应评审项"含脱敏时间戳、调用频次、耗时与报错记录"）：

1. 脱敏时间戳：ISO8601 本地时区，精确到秒；不记录毫秒级行为指纹。
2. 调用频次：每条事件带 event 名，run_end 汇总 counters / calls。
3. 耗时：每个 span 记录 dur_ms（浮点毫秒），可直接算 p50 / p95。
4. 报错：error 事件带 error_type（异常类名）+ msg（已脱敏）+ 阶段名。
5. 隐私：正文一律不入日志，只留 sha256 前 12 位与字符数；手机号/学号/QQ/
   本机绝对路径在落盘前替换为占位符。

日志格式：process_logs/YYYY-MM-DD.jsonl，一行一个 JSON，append 写入。
"""
from __future__ import annotations

import hashlib
import json
import re
import socket
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from . import config

# ---------------------------------------------------------------- 脱敏

_REDACTORS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s\"']+"), "[HOME]"),
    (re.compile(r"[A-Za-z]:[\\/]+Documents and Settings[\\/]+[^\\/\s\"']+"), "[HOME]"),
    (re.compile(r"\b1[3-9]\d{9}\b"), "[PHONE]"),
    (re.compile(r"\b\d{17}[\dXx]\b"), "[IDCARD]"),
    (re.compile(r"\b[1-9]\d{4,11}\b"), "[NUMID]"),      # 学号 / QQ 号
    (re.compile(r"(?i)\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[EMAIL]"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}"), "Bearer [TOKEN]"),
]


def sanitize(text: Any, limit: int = 160) -> str:
    """把任意文本压成可安全落盘的短串。"""
    s = "" if text is None else str(text)
    for pattern, repl in _REDACTORS:
        s = pattern.sub(repl, s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > limit:
        s = s[: limit - 3] + "..."
    return s


def digest(text: str) -> str:
    """内容指纹：只保留前 12 位，用于去重与溯源，不可逆推原文。"""
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:12]


_HOST_HASH = hashlib.sha256(socket.gethostname().encode("utf-8")).hexdigest()[:8]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


# ---------------------------------------------------------------- Telemetry


class Telemetry:
    """一次运行对应一个 run_id；所有事件写入当天的 JSONL。"""

    def __init__(self, day: str | None = None, run_id: str | None = None) -> None:
        self.day = day or today_str()
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.path = config.PROCESS_LOG_DIR / f"{self.day}.jsonl"
        self._t0 = time.perf_counter()
        self.counters: dict[str, int] = {}
        self.errors = 0
        self.warnings = 0
        self.notes: list[str] = []
        config.PROCESS_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # -- 底层写入 ---------------------------------------------------
    def _append(self, record: dict[str, Any]) -> None:
        payload = {
            "schema": config.SCHEMA,
            "run_id": self.run_id,
            "app_version": config.APP_VERSION,
            "ts": now_iso(),
            **record,
        }
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    # -- 对外接口 ---------------------------------------------------
    def event(self, name: str, status: str = "ok", dur_ms: float | None = None, **fields: Any) -> None:
        self.counters[name] = self.counters.get(name, 0) + 1
        record: dict[str, Any] = {"event": name, "status": status}
        if dur_ms is not None:
            record["dur_ms"] = round(dur_ms, 3)
        for key, value in fields.items():
            if isinstance(value, str):
                record[key] = sanitize(value)
            else:
                record[key] = value
        self._append(record)

    def error(self, stage: str, exc: BaseException, **fields: Any) -> None:
        self.errors += 1
        self._append(
            {
                "event": "error",
                "status": "error",
                "stage": stage,
                "error_type": type(exc).__name__,
                "msg": sanitize(exc),
                **{k: sanitize(v) if isinstance(v, str) else v for k, v in fields.items()},
            }
        )
        self.counters["error"] = self.counters.get("error", 0) + 1

    def warn(self, stage: str, msg: str, **fields: Any) -> None:
        self.warnings += 1
        self._append({"event": "warning", "status": "warn", "stage": stage, "msg": sanitize(msg), **fields})

    @contextmanager
    def span(self, name: str, **fields: Any):
        """计时上下文：正常结束记 ok，抛异常记 error 后继续向上抛。"""
        t0 = time.perf_counter()
        try:
            yield
        except Exception as exc:  # noqa: BLE001 - 日志模块必须兜住所有异常
            self.error(name, exc, dur_ms=round((time.perf_counter() - t0) * 1000, 3), **fields)
            raise
        else:
            self.event(name, dur_ms=(time.perf_counter() - t0) * 1000, **fields)

    # -- 生命周期 ---------------------------------------------------
    def start(self) -> None:
        self._append(
            {
                "event": "run_start",
                "status": "ok",
                "event_date": self.day,
                "host_hash": _HOST_HASH,
                "py": ".".join(map(str, __import__("sys").version_info[:3])),
                "tz": datetime.now().astimezone().tzname(),
                "log_path": str(self.path.relative_to(config.PROJECT_ROOT)),
            }
        )

    def finish(self, exit_code: int = 0, **fields: Any) -> None:
        dur = (time.perf_counter() - self._t0) * 1000
        self._append(
            {
                "event": "run_end",
                "status": "ok" if exit_code == 0 and self.errors == 0 else ("error" if exit_code else "warn"),
                "event_date": self.day,
                "dur_ms": round(dur, 3),
                "calls": sum(v for k, v in self.counters.items() if k != "error"),
                "errors": self.errors,
                "warnings": self.warnings,
                "counters": dict(sorted(self.counters.items())),
                "exit_code": exit_code,
                **{k: sanitize(v) if isinstance(v, str) else v for k, v in fields.items()},
            }
        )
