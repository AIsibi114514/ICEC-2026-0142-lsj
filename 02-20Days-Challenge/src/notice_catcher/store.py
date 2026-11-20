"""SQLite 存储层。表结构刻意扁平，方便面试时直接 sqlite3 查数据自证。"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from . import config
from .extractor import Notice

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS items (
    item_key    TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    event_date  TEXT,
    event_time  TEXT,
    place       TEXT,
    bring       TEXT,
    deadline    TEXT,
    category    TEXT,
    priority    TEXT,
    raw_excerpt TEXT,
    source      TEXT,
    first_seen  TEXT NOT NULL,
    last_seen   TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'open'
);

CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    day        TEXT NOT NULL,
    started    TEXT NOT NULL,
    dur_ms     REAL,
    notices    INTEGER,
    items_new  INTEGER,
    items_dup  INTEGER,
    errors     INTEGER,
    warnings   INTEGER
);

CREATE INDEX IF NOT EXISTS idx_items_event ON items(event_date, event_time);
CREATE INDEX IF NOT EXISTS idx_items_status ON items(status);
"""


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    return conn


def upsert_notices(conn: sqlite3.Connection, notices: list[Notice]) -> tuple[int, int]:
    """写入待办，按 item_key 去重。返回 (新增数, 重复数)。"""
    now = datetime.now().isoformat(timespec="seconds")
    new = dup = 0
    for n in notices:
        exists = conn.execute("SELECT 1 FROM items WHERE item_key = ?", (n.key,)).fetchone()
        if exists:
            conn.execute(
                "UPDATE items SET last_seen = ?, priority = ?, status = 'open' WHERE item_key = ?",
                (now, n.priority, n.key),
            )
            dup += 1
        else:
            conn.execute(
                """INSERT INTO items (item_key, title, event_date, event_time, place, bring,
                       deadline, category, priority, raw_excerpt, source, first_seen, last_seen)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (n.key, n.title, n.event_date, n.event_time, n.place, n.bring,
                 n.deadline, n.category, n.priority, n.raw_excerpt, n.source, now, now),
            )
            new += 1
    conn.commit()
    return new, dup


def log_run(conn: sqlite3.Connection, *, run_id: str, day: str, dur_ms: float,
            notices: int, items_new: int, items_dup: int, errors: int, warnings: int) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO runs
           (run_id, day, started, dur_ms, notices, items_new, items_dup, errors, warnings)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (run_id, day, datetime.now().isoformat(timespec="seconds"), dur_ms,
         notices, items_new, items_dup, errors, warnings),
    )
    conn.commit()


def open_items(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """SELECT * FROM items WHERE status = 'open'
               ORDER BY (event_date IS NULL), event_date, event_time, priority"""
        )
    )


def stats(conn: sqlite3.Connection) -> dict:
    row = conn.execute(
        """SELECT COUNT(*) AS total,
                  SUM(CASE WHEN status='open' THEN 1 ELSE 0 END) AS open,
                  SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) AS done
           FROM items"""
    ).fetchone()
    runs = conn.execute("SELECT COUNT(*) AS n FROM runs").fetchone()
    return {"total": row["total"] or 0, "open": row["open"] or 0,
            "done": row["done"] or 0, "runs": runs["n"] or 0}
