"""守住「模拟数据不得冒充真实遥测」这条线。

两条断言，任何一条被破坏都说明仓库里混进了不该有的东西：

  1. `tests/fixtures/simulated_logs/` 里的每个文件都是 `SIM-` 前缀、每条记录都带
     `"simulated": true`、`host_hash` 为 `simulate`；
  2. `process_logs/` 里每个文件都是 `YYYY-MM-DD.jsonl`、且**没有任何**记录带
     `simulated` 字段——真实遥测目录不接受合成数据。

用法：
    python tests/test_fixture_integrity.py
或由 pytest 收集：
    pytest tests/test_fixture_integrity.py
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "simulated_logs"
REAL_LOG_DIR = ROOT / "process_logs"

DATE_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}\.jsonl$")
SIM_NAME = re.compile(r"^SIM-\d{4}-\d{2}-\d{2}\.jsonl$")
EXPECTED_DAYS = [date(2026, 11, 20) + timedelta(days=i) for i in range(7)]


def _records(path: Path) -> list[dict]:
    out = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:  # noqa: PERF203
            raise AssertionError(f"{path.name}:{lineno} 不是合法 JSON：{exc}") from exc
    return out


def test_fixtures_are_marked_synthetic() -> None:
    files = sorted(FIXTURE_DIR.glob("*.jsonl"))
    assert files, f"{FIXTURE_DIR} 下没有夹具，检查是不是被误删了"

    seen: list[date] = []
    for path in files:
        assert SIM_NAME.match(path.name), f"夹具必须带 SIM- 前缀：{path.name}"
        assert not DATE_NAME.match(path.name), f"夹具不得使用 YYYY-MM-DD.jsonl 命名：{path.name}"

        recs = _records(path)
        assert recs, f"{path.name} 是空文件"

        for rec in recs:
            assert rec.get("simulated") is True, f"{path.name} 有记录缺 simulated: true"
            if "host_hash" in rec:
                assert rec["host_hash"] == "simulate", (
                    f"{path.name} 的 host_hash 应为 simulate，实际 {rec['host_hash']!r}"
                )

        day = date.fromisoformat(path.name[len("SIM-"):-len(".jsonl")])
        seen.append(day)

    assert sorted(seen) == EXPECTED_DAYS, f"夹具应覆盖 {EXPECTED_DAYS}，实际 {sorted(seen)}"


def test_real_logs_contain_no_synthetic_records() -> None:
    files = sorted(REAL_LOG_DIR.glob("*.jsonl"))
    assert files, f"{REAL_LOG_DIR} 下没有真实日志"

    for path in files:
        assert DATE_NAME.match(path.name), (
            f"process_logs/ 只接受 YYYY-MM-DD.jsonl，发现 {path.name}——"
            "模拟数据不能混进真实遥测目录"
        )
        for rec in _records(path):
            assert "simulated" not in rec, (
                f"{path.name} 出现 simulated 字段：真实遥测目录里不该有合成记录"
            )


def main() -> int:
    failed = 0
    for fn in (test_fixtures_are_marked_synthetic, test_real_logs_contain_no_synthetic_records):
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"[FAIL] {fn.__name__}: {exc}", file=sys.stderr)
        else:
            print(f"[ OK ] {fn.__name__}")
    if failed:
        return 1
    print("\n夹具与真实日志边界完好。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
