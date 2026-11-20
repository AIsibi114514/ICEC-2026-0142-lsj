"""本地网页界面（零第三方依赖：http.server + 原生 JS）。

定位：**这是给人用的入口，CLI 是给定时任务用的入口。**
两者跑的是同一套抽取代码（`explain_extraction`），不存在"网页版是另写的"问题。

    python run.py web            # 启动并自动开浏览器
    python run.py web --port 8765 --no-browser

网页上的每次解析都会写一条 `web_extract` 事件到 process_logs/，
这样"8 位真实用户使用"就有日志可查，而不是只靠截图自证。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import webbrowser
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

WEB_DIR = Path(__file__).resolve().parent
PROJECT = WEB_DIR.parent  # 02-20Days-Challenge
sys.path.insert(0, str(PROJECT / "src"))

from notice_catcher import config, report, store  # noqa: E402
from notice_catcher.extractor import explain_extraction, split_notices  # noqa: E402
from notice_catcher.telemetry import Telemetry, sanitize  # noqa: E402

MAX_BODY = 200_000  # 单次粘贴上限，防止误传大文件


# ---------------------------------------------------------------- 业务

def do_extract(text: str, save: bool = False) -> dict:
    """解析一段文本，返回可直接渲染的结构。"""
    started = time.perf_counter()
    chunks = split_notices(text)
    tel = Telemetry(day=date.today().isoformat())

    notices: list[dict] = []
    notice_objs = []
    for chunk in chunks:
        notice, rows = explain_extraction(chunk)
        notice_objs.append(notice)
        notices.append({
            "title": notice.title,
            "category": notice.category,
            "priority": notice.priority,
            "confidence": notice.confidence,
            "fields_found": len(notice.fields_found),
            "rows": rows,
        })

    saved_new = saved_dup = 0
    if save and notice_objs:
        conn = store.connect()
        try:
            saved_new, saved_dup = store.upsert_notices(conn, notice_objs)
            rows_now = store.open_items(conn)
            dur = (time.perf_counter() - started) * 1000
            report.write_daily(rows_now, day=date.today(),
                               run_meta={"notices": len(notice_objs), "items_new": saved_new,
                                         "items_dup": saved_dup, "dur_ms": dur})
            report.write_ledger(conn)
            report.write_excerpts(conn, date.today())
        finally:
            conn.close()

    dur_ms = round((time.perf_counter() - started) * 1000, 1)
    tel.event(
        "web_extract",
        status="ok",
        dur_ms=dur_ms,
        source="web",
        notices=len(notices),
        saved="true" if save else "false",
        items_new=saved_new,
        items_dup=saved_dup,
        avg_confidence=round(sum(n["confidence"] for n in notices) / len(notices), 3) if notices else 0.0,
        fields_missed=",".join(sorted({r["key"] for n in notices for r in n["rows"] if r["value"] == "—"})),
    )
    return {
        "notices": notices,
        "count": len(notices),
        "dur_ms": dur_ms,
        "saved_new": saved_new,
        "saved_dup": saved_dup,
    }


def build_board() -> dict:
    if not config.DB_PATH.exists():
        return {"items": [], "stats": {"open": 0, "total": 0}}
    conn = store.connect()
    try:
        items = [dict(r) for r in store.open_items(conn)]
        stats = store.stats(conn)
    finally:
        conn.close()
    today = date.today().isoformat()
    for it in items:
        if not it.get("event_date"):
            it["bucket"] = "nodate"
        elif it["event_date"] < today:
            it["bucket"] = "overdue"
        elif it["event_date"] == today:
            it["bucket"] = "today"
        else:
            it["bucket"] = "later"
    return {"items": items, "stats": stats}


def build_meta() -> dict:
    log_dir = config.PROCESS_LOG_DIR
    files = sorted(log_dir.glob("*.jsonl")) if log_dir.exists() else []
    today_runs = 0
    today_web = 0
    today_path = log_dir / f"{date.today().isoformat()}.jsonl"
    if today_path.exists():
        for line in today_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "run_start":
                today_runs += 1
            elif rec.get("event") == "web_extract":
                today_web += 1
    return {
        "version": config.APP_VERSION,
        "today": date.today().isoformat(),
        "log_days": len(files),
        "today_runs": today_runs,
        "today_web": today_web,
    }


def recent_logs(limit: int = 12) -> list[dict]:
    path = config.PROCESS_LOG_DIR / f"{date.today().isoformat()}.jsonl"
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        out.append({
            "time": str(rec.get("ts", ""))[11:19],
            "event": rec.get("event", "?"),
            "status": rec.get("status", "ok"),
            "dur_ms": rec.get("dur_ms"),
            "extra": sanitize(
                rec.get("msg") or rec.get("fallback") or rec.get("fields_missed") or "", 60
            ),
        })
    return out


# ---------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    server_version = "NoticeCatcherWeb"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:  # 静音默认访问日志
        pass

    # -- 工具方法 --
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: dict, code: int = 200) -> None:
        self._send(code, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    # -- 路由 --
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            html = (WEB_DIR / "index.html").read_bytes()
            self._send(200, html, "text/html; charset=utf-8")
            return

        if path == "/api/meta":
            self._json(build_meta())
            return

        if path == "/api/board":
            self._json(build_board())
            return

        if path == "/api/logs":
            qs = parse_qs(parsed.query)
            limit = min(int(qs.get("n", ["12"])[0] or 12), 60)
            self._json({"logs": recent_logs(limit)})
            return

        self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/extract":
            self._json({"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                self._json({"error": "文本过长"}, 413)
                return
            payload = json.loads(self.rfile.read(length) or b"{}")
            text = str(payload.get("text") or "")
            if not text.strip():
                self._json({"error": "请先粘贴通知原文"}, 400)
                return
            result = do_extract(text, save=bool(payload.get("save")))
            self._json(result)
        except json.JSONDecodeError:
            self._json({"error": "请求格式错误"}, 400)
        except Exception as exc:  # noqa: BLE001
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> int:
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print("")
    print("  NoticeCatcher 网页界面已启动")
    print(f"  地址：{url}")
    print("  提示：解析记录会写入 process_logs/（已脱敏），Ctrl+C 停止")
    print("")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  已停止。")
    finally:
        httpd.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="启动 NoticeCatcher 本地网页界面")
    # 云端部署时平台会注入 PORT，并要求绑定 0.0.0.0；本地运行默认只监听本机
    env_port = os.environ.get("PORT")
    parser.add_argument("--host", default="0.0.0.0" if env_port else "127.0.0.1",
                        help="监听地址，默认仅本机可访问")
    parser.add_argument("--port", type=int, default=int(env_port) if env_port else 8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    return serve(args.host, args.port, not args.no_browser)


if __name__ == "__main__":
    raise SystemExit(main())
