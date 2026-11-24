"""数据集清单。

约定的文件名格式：C{类别号}_{条件名}_{两位序号}.wav
    C1_normal_low_01.wav
    C5_rubbing_15.wav

类别标签以目录 + 文件名为准，不另设标签表——
少一处需要手工同步的地方，就少一处会写错的地方。
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np

from . import CONDITIONS, CONDITION_ZH, audio

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(ROOT, "raw_audio")

PATTERN = re.compile(r"^(C\d)_([a-z_]+)_(\d{2})\.wav$")

TRIM_HEAD = 0.5
TRIM_TAIL = 0.5


def list_samples(raw_dir: str = RAW_DIR) -> list[dict]:
    """按类别、序号排序返回全部样本。"""
    out: list[dict] = []
    for path in glob.glob(os.path.join(raw_dir, "*.wav")):
        name = os.path.basename(path)
        m = PATTERN.match(name)
        if not m:
            continue  # ALL_75_samples.wav 这类汇总文件不算样本
        cid, cond, idx = m.group(1), m.group(2), int(m.group(3))
        label = f"{cid}_{cond}"
        if label not in CONDITIONS:
            raise ValueError(f"unknown condition label in {name}: {label}")
        out.append(
            {
                "sample_id": f"{label}_{idx:02d}",
                "label": label,
                "label_zh": CONDITION_ZH[label],
                "condition_class": cid,
                "index": idx,
                "file": name,
                "path": path,
            }
        )
    out.sort(key=lambda s: (s["label"], s["index"]))
    return out


def load_steady(sample: dict, trim_head: float = TRIM_HEAD, trim_tail: float = TRIM_TAIL):
    """读入并裁剪到稳态段。"""
    x, sr = audio.read_wav(sample["path"])
    return audio.trim_steady(x, sr, trim_head, trim_tail), sr


def load_all(samples: list[dict]) -> list[tuple[dict, np.ndarray, int]]:
    return [(s, *load_steady(s)) for s in samples]
