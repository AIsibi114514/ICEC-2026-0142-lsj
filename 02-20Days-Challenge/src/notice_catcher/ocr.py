"""截图 OCR 兜底通路。

策略：文本通路上是主路；当日收件箱里出现 .png/.jpg 时走本模块。
- 未安装 pytesseract / tesseract 时，不崩、不伪造，直接抛 EngineMissing，
  由 cli 捕获并写入 error 事件（这会成为第一次真实迭代的依据）。
- 装了引擎后无需改代码，同一份日志里引擎字段会自动从 missing 变成可用。
"""
from __future__ import annotations

from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class EngineMissing(RuntimeError):
    """OCR 引擎不可用（依赖未安装）。"""


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def available_engines() -> list[str]:
    engines: list[str] = []
    try:
        import pytesseract  # noqa: F401
        from PIL import Image  # noqa: F401

        engines.append("pytesseract")
    except Exception:  # noqa: BLE001
        pass
    return engines


def image_to_text(path: Path, lang: str = "chi_sim+eng") -> str:
    """把截图转成文本；不可用时抛 EngineMissing。"""
    try:
        import pytesseract
        from PIL import Image
    except Exception as exc:  # noqa: BLE001
        raise EngineMissing(
            "pytesseract/Pillow 未安装，截图通路跳过（pip install pytesseract pillow）"
        ) from exc

    try:
        with Image.open(path) as img:
            return pytesseract.image_to_string(img, lang=lang)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"OCR 失败: {type(exc).__name__}") from exc
