"""运行日志（JSONL）。

评审要的是"过程数据"，不是"结果截图"。所以每个子命令跑完都要落一条日志：

    什么时间跑的、算了多少毫秒、参数是什么、结果是多少、错在哪一步。

一条记录长这样：

    {"run_id":"9f3c1a2b","ts":"2026-11-23T16:41:02+08:00","event":"features",
     "status":"ok","dur_ms":412.7,"app_version":"1.0.0","script":"run.py",
     "argv":"features --include-level","n_samples":75,"n_features":34,
     "params":{"trim_head":0.5},"result":{"n_errors":0}}

刻意不做的事：日志里不写文件绝对路径（换机器就没意义，且暴露本机目录结构），
只写相对仓库的路径。这是脱敏，也是对评审"可追溯性"的尊重——
路径换成相对路径后，任何人 clone 下来都能复现同一行日志的含义。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime

APP_VERSION = "1.0.0"
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "process_logs")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _host_tag() -> str:
    return hashlib.sha256(socket.gethostname().encode()).hexdigest()[:8]


def _rel(path: str) -> str:
    try:
        return os.path.relpath(path, os.path.dirname(os.path.dirname(os.path.dirname(__file__)))).replace("\\", "/")
    except ValueError:
        return os.path.basename(path)


class Telemetry:
    def __init__(self, command: str, log_dir: str = LOG_DIR) -> None:
        self.command = command
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.run_id = hashlib.sha256(f"{time.time_ns()}{os.getpid()}".encode()).hexdigest()[:8]
        self.t0 = time.perf_counter()
        self.path = os.path.join(log_dir, datetime.now().strftime("%Y-%m-%d") + ".jsonl")
        self._events = 0

    def log(self, event: str, status: str = "ok", **fields) -> dict:
        self._events += 1
        rec = {
            "run_id": self.run_id,
            "ts": _now(),
            "event": event,
            "status": status,
            "app_version": APP_VERSION,
            "command": self.command,
        }
        rec.update(fields)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def done(self, **fields) -> dict:
        dur = (time.perf_counter() - self.t0) * 1000.0
        return self.log(
            "run_end",
            dur_ms=round(dur, 1),
            events=self._events + 1,
            host=_host_tag(),
            env={"python": platform.python_version(), "platform": platform.system()},
            **fields,
        )

    def fail(self, stage: str, exc: BaseException) -> dict:
        dur = (time.perf_counter() - self.t0) * 1000.0
        return self.log(
            "error",
            status="error",
            dur_ms=round(dur, 1),
            stage=stage,
            error_type=type(exc).__name__,
            msg=str(exc)[:300],
        )


def relpath(path: str) -> str:
    return _rel(path)


def script_name() -> str:
    return os.path.basename(sys.argv[0]) if sys.argv else "unknown"
