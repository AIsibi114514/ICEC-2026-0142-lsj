"""守住「process_logs/ 只存真实运行记录」这条线。

本仓库不包含任何合成/预演日志。这道检查负责防止以后混进来：

  1. `process_logs/` 下的文件名必须是 `YYYY-MM-DD.jsonl`；
  2. 任何记录都不得带 `simulated` 字段；
  3. 不得出现 `SIM-` 前缀的文件。

为什么值得单独写一个测试：日志是评审 25 分项的证据，一旦有一条合成记录混进来，
整份数据就不再是「过程产生的数据」。靠人眼检查 7 天 × 每天几十条记录不可靠。

用法：
    python tests/test_log_purity.py
或由 pytest 收集：
    pytest tests/test_log_purity.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REAL_LOG_DIR = ROOT / "process_logs"

DATE_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}\.jsonl$")


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


def test_real_logs_contain_no_synthetic_records() -> None:
    files = sorted(REAL_LOG_DIR.glob("*.jsonl"))
    assert files, f"{REAL_LOG_DIR} 下没有真实日志"

    for path in files:
        assert DATE_NAME.match(path.name), (
            f"process_logs/ 只接受 YYYY-MM-DD.jsonl，发现 {path.name}"
        )
        for rec in _records(path):
            assert "simulated" not in rec, (
                f"{path.name} 出现 simulated 字段：真实遥测目录里不该有合成记录"
            )


def test_no_simulated_directory_in_repo() -> None:
    """仓库内不应存在模拟日志目录（本机预演产物也不该留在提交范围里）。"""
    offenders = [
        p for p in ROOT.rglob("*simulated*")
        if p.is_dir() and ".git" not in p.parts and "__pycache__" not in p.parts
    ]
    assert not offenders, "发现模拟数据目录：" + "、".join(str(p) for p in offenders)


def main() -> int:
    failed = 0
    for fn in (test_real_logs_contain_no_synthetic_records, test_no_simulated_directory_in_repo):
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"[FAIL] {fn.__name__}: {exc}", file=sys.stderr)
        else:
            print(f"[ OK ] {fn.__name__}")
    if failed:
        return 1
    print("\n真实日志纯净，仓库内无模拟数据目录。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
