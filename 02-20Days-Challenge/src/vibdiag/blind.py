"""盲测协议：封存标签 → 匿名化音频 → 预测 → 启封打分。

为什么要走这么麻烦的流程：
  自己给自己的数据打分，最容易被"我大概记得第 12 条是擦碰"污染。
  所以把 25 段留出样本重命名成 B01~B25.wav（打乱顺序），
  预测阶段只允许读 blind_clips/ 与 model_blind.json，看不到任何标签；
  答案键 blind_key.csv 立刻算出 sha256 并写进 manifest，
  连我自己改一个字符都能被发现。

三步对应三个子命令，中间产物全部提交进仓库：

    python run.py blind seal       → results/blind_clips/ + results/blind_manifest.json
    python run.py blind predict    → results/blind_predictions.csv
    python run.py blind score      → results/blind_score.json（此时才读盲键）

`blind verify` 会重新校验 key 的 sha256，确认打分用的键就是封存时的那一份。
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import shutil

import numpy as np

from . import CONDITION_ZH, audio, features

CLIP_PREFIX = "B"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def split_train_test(samples: list[dict], n_test_per_class: int = 5) -> tuple[list[dict], list[dict]]:
    """按类别内序号切分：前 (15 - n) 条训练，最后 n 条留出。

    刻意不做随机划分：序号是采集顺序，最后 5 条是同一场采集的尾段，
    时间上更接近"之后新录的数据"，比随机打散更接近真实部署场景。
    """
    train, test = [], []
    by_class: dict[str, list[dict]] = {}
    for s in samples:
        by_class.setdefault(s["label"], []).append(s)
    for label in sorted(by_class):
        items = sorted(by_class[label], key=lambda x: x["index"])
        cut = len(items) - n_test_per_class
        train.extend(items[:cut])
        test.extend(items[cut:])
    return train, test


def seal(test: list[dict], out_dir: str, seed: int = 20261123) -> dict:
    """把留出样本匿名化拷入 blind_clips/，并生成封存的答案键。"""
    clips_dir = os.path.join(out_dir, "blind_clips")
    os.makedirs(clips_dir, exist_ok=True)
    for old in glob.glob(os.path.join(clips_dir, "*.wav")):
        os.remove(old)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(test))
    rows = []
    for pos, i in enumerate(order, start=1):
        s = test[int(i)]
        clip_id = f"{CLIP_PREFIX}{pos:02d}"
        dst = os.path.join(clips_dir, f"{clip_id}.wav")
        shutil.copyfile(s["path"], dst)
        rows.append(
            {
                "clip": clip_id,
                "true_label": s["label"],
                "true_label_zh": s["label_zh"],
                "source_file": s["file"],
                "sha256": sha256_file(dst),
            }
        )

    key_path = os.path.join(out_dir, "blind_key.csv")
    with open(key_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "true_label", "true_label_zh", "source_file", "sha256"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["clip"]))

    manifest = {
        "seed": seed,
        "n_clips": len(rows),
        "clip_dir": "results/blind_clips",
        "key_file": "results/blind_key.csv",
        "key_sha256": sha256_file(key_path),
        "candidate_files": sorted(r["source_file"] for r in rows),
        "protocol": "clip 重命名并打乱顺序；predict 阶段只读 blind_clips/ 与 model_blind.json；score 阶段才读 blind_key.csv",
    }
    with open(os.path.join(out_dir, "blind_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    return manifest


def verify(out_dir: str) -> tuple[bool, str, str]:
    """校验答案键是否被改动过。返回 (是否一致, 记录值, 实际值)。"""
    with open(os.path.join(out_dir, "blind_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    actual = sha256_file(os.path.join(out_dir, "blind_key.csv"))
    ok = actual == manifest["key_sha256"]
    return ok, manifest["key_sha256"], actual


def predict(clips_dir: str, model: dict, names: list[str], out_dir: str) -> list[dict]:
    """对盲测片段逐条预测。此函数不接触任何标签。"""
    from .model import LogisticRegression

    clf = LogisticRegression.from_dict(model)
    rows = []
    for path in sorted(glob.glob(os.path.join(clips_dir, "*.wav"))):
        x, sr = audio.read_wav(path)
        x = audio.trim_steady(x, sr)
        feats = features.extract(x, sr, include_level=True)
        vec = np.array([[feats[n] for n in names]])
        proba = clf.predict_proba(vec)[0]
        order = np.argsort(proba)[::-1]
        rows.append(
            {
                "clip": os.path.splitext(os.path.basename(path))[0],
                "pred_label": clf.classes_[order[0]],
                "pred_label_zh": CONDITION_ZH[clf.classes_[order[0]]],
                "confidence": round(float(proba[order[0]]), 4),
                "second_label": clf.classes_[order[1]],
                "second_prob": round(float(proba[order[1]]), 4),
                "bpf_hz": round(feats["bpf_hz"], 2),
                "hf_noise_ratio": round(feats["hf_noise_ratio"], 4),
            }
        )
    path_out = os.path.join(out_dir, "blind_predictions.csv")
    with open(path_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def score(out_dir: str) -> dict:
    """启封打分。"""
    from . import evaluate

    with open(os.path.join(out_dir, "blind_predictions.csv"), encoding="utf-8") as f:
        preds = list(csv.DictReader(f))
    with open(os.path.join(out_dir, "blind_key.csv"), encoding="utf-8") as f:
        keys = {r["clip"]: r for r in csv.DictReader(f)}

    y_true = [keys[p["clip"]]["true_label"] for p in preds]
    y_pred = [p["pred_label"] for p in preds]
    classes = sorted(set(y_true))
    summary = evaluate.summarize(y_true, y_pred, classes)

    # 逐条对照表，方便人工抽查
    detail = [
        {
            "clip": p["clip"],
            "true": keys[p["clip"]]["true_label"],
            "pred": p["pred_label"],
            "hit": int(keys[p["clip"]]["true_label"] == p["pred_label"]),
            "confidence": p["confidence"],
            "source_file": keys[p["clip"]]["source_file"],
        }
        for p in preds
    ]
    summary["detail"] = detail
    with open(os.path.join(out_dir, "blind_score.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    return summary
