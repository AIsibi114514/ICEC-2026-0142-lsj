"""项目根启动器：把 src/ 加进 sys.path，让 python run.py 直接可用。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from notice_catcher.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
