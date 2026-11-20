"""NoticeCatcher 全局配置。

目录约定（对应 02-20Days-Challenge/ 下的提交结构）：
    inbox/          每日待处理的通知原文（.txt / 截图）
    process_logs/   自动化运行日志（JSONL，每日一份，必须提交）
    output/         日报 Markdown、CSV 台账、脱敏摘录
    data/           本地 SQLite（不入库）
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

APP_NAME = "NoticeCatcher"
APP_VERSION = "0.1.0"
SCHEMA = "nc.telemetry/1.0"


def _pick_root() -> Path:
    """定位运行根目录。

    本地运行时就是 02-20Days-Challenge/；
    在只读沙箱里部署时（如云端预览环境），回退到临时目录，保证程序仍能跑起来。
    """
    candidate = Path(__file__).resolve().parents[2]
    try:
        probe = candidate / ".write_probe"
        probe.write_text("x", encoding="utf-8")
        probe.unlink()
        return candidate
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "notice-catcher-runtime"
        (fallback / "process_logs").mkdir(parents=True, exist_ok=True)
        (fallback / "output").mkdir(parents=True, exist_ok=True)
        return fallback


# config.py 位于 02-20Days-Challenge/src/notice_catcher/，向上三级即项目根
PROJECT_ROOT = _pick_root()

INBOX_DIR = PROJECT_ROOT / "inbox"
PROCESSED_DIR = INBOX_DIR / "_processed"
PROCESS_LOG_DIR = PROJECT_ROOT / "process_logs"
OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_DIR = PROJECT_ROOT / "data"
ITERATION_DIR = PROJECT_ROOT / "iterations"

DB_PATH = DATA_DIR / "notices.db"
LEDGER_CSV = OUTPUT_DIR / "ledger.csv"

# 日报里"未来 N 天"的窗口
UPCOMING_DAYS = 3
# 单条通知原文在数据库/日报中保留的摘录长度（已脱敏）
EXCERPT_LIMIT = 240
# 图片 OCR 是否默认启用（缺失依赖时自动降级并记录 EngineMissing）
OCR_ENABLED = True
