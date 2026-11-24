#!/usr/bin/env python3
"""声振诊断仪 · 命令行入口。

    python run.py index            体检：清点 75 段录音并写 sample_index.csv
    python run.py features         提取全部特征 → features/feature_table.csv
    python run.py train            交叉验证 + 训练两个模型 + 导出决策树规则
    python run.py blind seal       封存盲测集（匿名化 25 段留出样本）
    python run.py blind predict    盲测预测（只读 blind_clips/ 与 model_blind.json）
    python run.py blind score      启封打分
    python run.py blind verify     校验答案键未被改动
    python run.py noise            抗噪扫描 → results/noise_sweep.csv
    python run.py figures          生成 figures/fig1~fig8
    python run.py export-csv       导出 75 份时序 CSV → csv_series/
    python run.py all              按顺序跑完整条流水线

每次运行都会往 process_logs/<日期>.jsonl 追加真实日志。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover - 极老的终端
        pass

import numpy as np  # noqa: E402

from vibdiag import CONDITIONS, CONDITION_ZH, blind, dataset, evaluate, features, noise  # noqa: E402
from vibdiag.model import DecisionTree, LogisticRegression, load_model, save_model  # noqa: E402
from vibdiag.telemetry import Telemetry, relpath  # noqa: E402

ROOT = HERE
RESULTS = os.path.join(ROOT, "results")
MODELS = os.path.join(ROOT, "models")
FEATDIR = os.path.join(ROOT, "features")
FIGDIR = os.path.join(ROOT, "figures")
CSVDIR = os.path.join(ROOT, "csv_series")
LOGSDIR = os.path.join(ROOT, "process_logs")


def ensure_dirs() -> None:
    for d in (RESULTS, MODELS, FEATDIR, FIGDIR, CSVDIR, LOGSDIR):
        os.makedirs(d, exist_ok=True)


class _NullTelemetry:
    """--no-telemetry 用：接口与 Telemetry 一致，但不落盘。

    刻意不叫 Telemetry 的子类——子类的 run_id / 路径逻辑会被继承，
    而这里的语义是"根本没有日志"，不该留下任何痕迹。
    """

    def __init__(self, command: str) -> None:
        self.command = command
        self.path = "(未启用)"

    def log(self, event: str, status: str = "ok", **fields) -> dict:
        return {}

    def done(self, **fields) -> dict:
        return {}

    def fail(self, stage: str, exc: BaseException) -> dict:
        return {}


# ---------------------------------------------------------------------------
# 交叉验证
# ---------------------------------------------------------------------------
def stratified_folds(labels: list[str], k: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    by_class: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        by_class.setdefault(lab, []).append(i)
    for lab in sorted(by_class):
        idxs = np.array(sorted(by_class[lab]))
        rng.shuffle(idxs)
        for j, i in enumerate(idxs):
            folds[j % k].append(int(i))
    return folds


def cross_validate(X: np.ndarray, y: list[str], k: int = 5, C: float = 0.3, seed: int = 42) -> dict:
    folds = stratified_folds(y, k, seed)
    y_true: list[str] = []
    y_pred: list[str] = []
    for j in range(k):
        te = folds[j]
        hold = set(te)
        tr = [i for i in range(len(y)) if i not in hold]
        clf = LogisticRegression(C=C, seed=seed)
        clf.fit(X[tr], [y[i] for i in tr])
        y_true.extend([y[i] for i in te])
        y_pred.extend(clf.predict(X[te]))
    return evaluate.summarize(y_true, y_pred, sorted(set(y)))


# ---------------------------------------------------------------------------
# 子命令
# ---------------------------------------------------------------------------
def cmd_index(tm: Telemetry) -> dict:
    t0 = time.perf_counter()
    samples = dataset.list_samples()
    if not samples:
        raise FileNotFoundError("raw_audio/ 下没有找到符合 C{n}_{name}_{nn}.wav 的文件")
    rows = []
    sr = 0
    for s in samples:
        x, sr = dataset.load_steady(s)
        rows.append(
            {
                "sample_id": s["sample_id"],
                "label": s["label"],
                "label_zh": s["label_zh"],
                "index": s["index"],
                "file": s["file"],
                "sample_rate": sr,
                "trimmed_samples": len(x),
                "trimmed_seconds": round(len(x) / sr, 3),
                "rms_dbfs": round(float(20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-20)), 2),
                "sha256_16": blind.sha256_file(s["path"])[:16],
            }
        )
    path = os.path.join(ROOT, "sample_index.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    counts = {c: sum(1 for r in rows if r["label"] == c) for c in CONDITIONS}
    dur = (time.perf_counter() - t0) * 1000
    print(f"[index] {len(rows)} 段录音 → {relpath(path)}  ({dur:.0f} ms)")
    for c in CONDITIONS:
        print(f"        {c:16s} {CONDITION_ZH[c]:8s} {counts[c]:2d} 组")
    tm.log("index", n_samples=len(rows), counts=counts, dur_ms=round(dur, 1),
           out=relpath(path), sample_rate=sr, trimmed_seconds=round(len(x) / sr, 3))
    return counts


def _feature_rows(samples: list[dict], names: list[str], include_level: bool) -> list[dict]:
    rows = []
    for s in samples:
        x, sr = dataset.load_steady(s)
        feats = features.extract(x, sr, include_level=include_level)
        rows.append({"sample_id": s["sample_id"], "label": s["label"],
                     **{n: round(feats[n], 6) for n in names}})
    return rows


def cmd_features(tm: Telemetry, include_level: bool = True) -> dict:
    t0 = time.perf_counter()
    samples = dataset.list_samples()
    names = features.feature_names(include_level)
    rows = _feature_rows(samples, names, include_level)
    path = os.path.join(FEATDIR, "feature_table.csv" if include_level else "feature_table_no_level.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 每类均值/标准差汇总，写报告时直接引用，避免手抄出错
    summary = []
    for n in names:
        row = {"feature": n}
        for c in CONDITIONS:
            vals = np.array([r[n] for r in rows if r["label"] == c])
            row[f"{c}_mean"] = round(float(vals.mean()), 6)
            row[f"{c}_sd"] = round(float(vals.std()), 6)
        allv = np.array([r[n] for r in rows])
        row["all_min"] = round(float(allv.min()), 6)
        row["all_max"] = round(float(allv.max()), 6)
        # 类间分离度：各类均值的极差 / 类内平均标准差
        means = np.array([row[f"{c}_mean"] for c in CONDITIONS])
        sds = np.array([row[f"{c}_sd"] for c in CONDITIONS]) + 1e-12
        row["separation"] = round(float((means.max() - means.min()) / sds.mean()), 3)
        summary.append(row)
    spath = os.path.join(FEATDIR, "feature_summary.csv")
    with open(spath, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    dur = (time.perf_counter() - t0) * 1000
    print(f"[features] {len(rows)} 样本 × {len(names)} 维 → {relpath(path)}  ({dur:.0f} ms)")
    print(f"[features] 每类均值/标准差 → {relpath(spath)}")
    tm.log("features", n_samples=len(rows), n_features=len(names),
           include_level=int(include_level), dur_ms=round(dur, 1), out=relpath(path),
           trim={"head_s": dataset.TRIM_HEAD, "tail_s": dataset.TRIM_TAIL})
    return {"path": path, "n_features": len(names)}


def cmd_train(tm: Telemetry, C: float | None = None) -> dict:
    t0 = time.perf_counter()
    samples = dataset.list_samples()
    names = features.feature_names(True)
    names_nolevel = features.feature_names(False)

    raw_rows, y = [], []
    for s in samples:
        x, sr = dataset.load_steady(s)
        raw_rows.append(features.extract(x, sr))
        y.append(s["label"])
    col = {n: i for i, n in enumerate(features.FEATURE_ORDER)}
    Xfull = np.array([[r[n] for n in features.FEATURE_ORDER] for r in raw_rows], dtype=float)
    X = Xfull[:, [col[n] for n in names]]
    Xn = Xfull[:, [col[n] for n in names_nolevel]]

    # 正则强度扫描：只看 5 折 CV，不看盲测
    sweep_rows = []
    for c in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        cv = cross_validate(X, y, C=c)
        sweep_rows.append({"C": c, "cv_accuracy": cv["accuracy"], "cv_macro_f1": cv["macro_f1"]})
    best_C = max(sweep_rows, key=lambda r: (r["cv_accuracy"], -r["C"]))["C"]
    chosen_C = C if C is not None else best_C

    cv_full = cross_validate(X, y, C=chosen_C)
    cv_nolevel = cross_validate(Xn, y, C=chosen_C)

    train, test = blind.split_train_test(samples, n_test_per_class=5)
    print(f"[train] 训练/留出：{len(train)} / {len(test)}   正则 C={chosen_C}（5 折 CV 选出）")

    idx_of = {s["sample_id"]: i for i, s in enumerate(samples)}
    tr_i = [idx_of[s["sample_id"]] for s in train]
    clf_blind = LogisticRegression(C=chosen_C).fit(X[tr_i], [y[i] for i in tr_i])
    save_model(os.path.join(MODELS, "model_blind.json"), clf_blind.to_dict(names))

    # 去掉绝对电平的版本：用来做"换手机/换距离"的增益漂移实验
    Xn_tr = Xn[tr_i]
    clf_blind_nl = LogisticRegression(C=chosen_C).fit(Xn_tr, [y[i] for i in tr_i])
    save_model(os.path.join(MODELS, "model_blind_nolevel.json"), clf_blind_nl.to_dict(names_nolevel))

    clf_full = LogisticRegression(C=chosen_C).fit(X, y)
    save_model(os.path.join(MODELS, "model_full.json"), clf_full.to_dict(names))

    tree = DecisionTree(max_depth=3, min_samples_leaf=3).fit(X[tr_i], [y[i] for i in tr_i])
    with open(os.path.join(RESULTS, "rules.txt"), "w", encoding="utf-8") as f:
        f.write("# CART(max_depth=3) 在 50 段训练样本上学到的判据（可直接移植为 if-else）\n")
        f.write(f"# 特征口径：raw_audio 裁剪 {dataset.TRIM_HEAD}s ~ {dataset.TRIM_TAIL}s 稳态段\n\n")
        f.write(tree.rules_text(names) + "\n")
    y_tr = [y[i] for i in tr_i]
    tree_acc = evaluate.accuracy(evaluate.confusion_matrix(y_tr, tree.predict(X[tr_i]), sorted(set(y))))

    imp = clf_full.feature_importance(names)
    with open(os.path.join(RESULTS, "feature_importance.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "feature", "max_abs_weight"])
        for r, (n, v) in enumerate(imp, start=1):
            w.writerow([r, n, round(v, 6)])

    keep = ("n", "correct", "accuracy", "accuracy_ci95", "macro_f1", "confusion", "classes", "per_class")
    summary = {
        "n_samples": len(samples),
        "n_features": len(names),
        "n_features_no_level": len(names_nolevel),
        "C_sweep": sweep_rows,
        "chosen_C": chosen_C,
        "cv_5fold": {k: cv_full[k] for k in keep},
        "cv_5fold_no_level": {k: cv_nolevel[k] for k in ("n", "correct", "accuracy", "accuracy_ci95", "macro_f1", "confusion")},
        "decision_tree_train_accuracy": round(tree_acc, 4),
        "train_n": len(train),
        "test_n": len(test),
        "top_features": [{"feature": n, "weight": round(v, 4)} for n, v in imp[:12]],
    }
    with open(os.path.join(RESULTS, "cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    dur = (time.perf_counter() - t0) * 1000
    print(f"[train] 5 折 CV {cv_full['accuracy']:.1%}"
          f"（95%CI {cv_full['accuracy_ci95'][0]:.1%}~{cv_full['accuracy_ci95'][1]:.1%}）"
          f"  macro-F1 {cv_full['macro_f1']:.3f}  ({dur:.0f} ms)")
    print(f"[train] 去掉绝对电平特征后 CV {cv_nolevel['accuracy']:.1%}（{len(names_nolevel)} 维）")
    print(f"[train] 决策树（深 3）训练集准确率 {tree_acc:.1%} → results/rules.txt")
    print("[train] 权重前 8：" + ", ".join(f"{n}({v:.2f})" for n, v in imp[:8]))
    tm.log("train", n_train=len(train), n_test=len(test), C=chosen_C,
           cv_accuracy=cv_full["accuracy"], cv_macro_f1=cv_full["macro_f1"],
           cv_accuracy_no_level=cv_nolevel["accuracy"], tree_train_accuracy=round(tree_acc, 4),
           dur_ms=round(dur, 1))
    return summary


def cmd_blind(tm: Telemetry, action: str) -> dict:
    t0 = time.perf_counter()
    samples = dataset.list_samples()

    if action == "seal":
        _, test = blind.split_train_test(samples, 5)
        manifest = blind.seal(test, RESULTS)
        dur = (time.perf_counter() - t0) * 1000
        print(f"[blind:seal] {manifest['n_clips']} 段匿名化 → results/blind_clips/  ({dur:.0f} ms)")
        print(f"[blind:seal] 答案键 sha256={manifest['key_sha256'][:32]}… 已封存")
        tm.log("blind_seal", n_clips=manifest["n_clips"], seed=manifest["seed"],
               key_sha256=manifest["key_sha256"], dur_ms=round(dur, 1))
        return manifest

    if action == "predict":
        model = load_model(os.path.join(MODELS, "model_blind.json"))
        rows = blind.predict(os.path.join(RESULTS, "blind_clips"), model, model["features"], RESULTS)
        dur = (time.perf_counter() - t0) * 1000
        spread: dict[str, int] = {}
        for r in rows:
            spread[r["pred_label"]] = spread.get(r["pred_label"], 0) + 1
        print(f"[blind:predict] {len(rows)} 段完成预测（全程未接触标签）  ({dur:.0f} ms)")
        print(f"[blind:predict] 预测分布：{spread}")
        tm.log("blind_predict", n=len(rows), prediction_spread=spread, dur_ms=round(dur, 1),
               out="results/blind_predictions.csv")
        return {"n": len(rows)}

    if action == "score":
        summary = blind.score(RESULTS)
        dur = (time.perf_counter() - t0) * 1000
        print(f"[blind:score] {summary['correct']}/{summary['n']} = {summary['accuracy']:.1%}"
              f"（95%CI {summary['accuracy_ci95'][0]:.1%}~{summary['accuracy_ci95'][1]:.1%}）"
              f"  macro-F1 {summary['macro_f1']:.3f}  ({dur:.0f} ms)")
        print(evaluate.format_confusion(np.array(summary["confusion"]), summary["classes"]))
        tm.log("blind_score", accuracy=summary["accuracy"], correct=summary["correct"],
               n=summary["n"], macro_f1=summary["macro_f1"], dur_ms=round(dur, 1))
        return summary

    if action == "verify":
        ok, recorded, actual = blind.verify(RESULTS)
        dur = (time.perf_counter() - t0) * 1000
        print(f"[blind:verify] 答案键{'一致 OK' if ok else '被改动 MISMATCH'}  "
              f"recorded={recorded[:16]}… actual={actual[:16]}…")
        tm.log("blind_verify", ok=int(ok), dur_ms=round(dur, 1))
        if not ok:
            raise RuntimeError("blind_key.csv 已被改动，打分结果不可信")
        return {"ok": ok}

    raise ValueError(f"unknown blind action: {action}")


def cmd_noise(tm: Telemetry) -> dict:
    t0 = time.perf_counter()
    model = load_model(os.path.join(MODELS, "model_blind.json"))
    rows = noise.sweep(ROOT, model, model["features"])
    path = os.path.join(RESULTS, "noise_sweep.csv")
    noise.write_csv(path, rows)
    dur = (time.perf_counter() - t0) * 1000
    print(f"[noise] {len(rows)} 组条件 → {relpath(path)}  ({dur:.0f} ms)")
    frontends = []
    for r in rows:
        if r["frontend"] not in frontends:
            frontends.append(r["frontend"])
    for fe in frontends:
        clean = next(r for r in rows if r["frontend"] == fe and r["snr_db"] == "")
        print(f"        {fe:16s} 干净基线 {clean['accuracy']:.0%}   预测分布 {clean['pred_spread']}")
        for kind in ("white", "pink"):
            cells = [
                f"{int(r['snr_db']):>4}dB:{r['accuracy']:.0%}"
                for r in rows
                if r["frontend"] == fe and r["noise"] == kind and r["snr_db"] != ""
            ]
            print(f"        {fe:16s} {kind:5s} " + "  ".join(cells))
    tm.log("noise", n_conditions=len(rows), dur_ms=round(dur, 1), out=relpath(path))
    return {"path": path, "rows": len(rows)}


def cmd_export_csv(tm: Telemetry, rate: int = 1000) -> dict:
    """把 75 段 WAV 导出为时序 CSV（默认降采样到 1 kHz）。

    CSV 是 WAV 的**有损视图**：便于用 Excel 打开、手算 RMS 与峰峰值，
    但不能替代 WAV 做高频频谱分析（1 kHz 的奈奎斯特上限只有 500 Hz）。
    一切分析都读 raw_audio/*.wav。
    """
    t0 = time.perf_counter()
    samples = dataset.list_samples()
    os.makedirs(CSVDIR, exist_ok=True)
    n_rows = 0
    for s in samples:
        x, sr = dataset.load_steady(s)
        step = max(1, sr // rate)
        t = np.arange(0, len(x), step) / sr
        y = x[::step]
        with open(os.path.join(CSVDIR, f"{s['sample_id']}.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "amplitude"])
            for ti, yi in zip(t, y):
                w.writerow([f"{ti:.4f}", f"{yi:.6f}"])
        n_rows += len(y)
    dur = (time.perf_counter() - t0) * 1000
    print(f"[export-csv] {len(samples)} 份 / {n_rows} 行 → csv_series/  ({rate} Hz, {dur:.0f} ms)")
    tm.log("export_csv", n_files=len(samples), n_rows=n_rows, rate=rate, dur_ms=round(dur, 1))
    return {"files": len(samples), "rows": n_rows}


def cmd_figures(tm: Telemetry) -> dict:
    t0 = time.perf_counter()
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import make_figures

    made = make_figures.build_all(ROOT)
    dur = (time.perf_counter() - t0) * 1000
    print(f"[figures] {len(made)} 张 → figures/  ({dur:.0f} ms)")
    for m in made:
        print(f"          {os.path.basename(m)}")
    tm.log("figures", n_figures=len(made), files=[relpath(m) for m in made], dur_ms=round(dur, 1))
    return {"n": len(made)}


def cmd_stress(tm: Telemetry) -> dict:
    """跑四项压力测试：窗口长度 / 少样本 / 未见故障 / 增益漂移。"""
    t0 = time.perf_counter()
    from vibdiag import stress

    samples = dataset.list_samples()
    names = features.feature_names(True)
    names_nl = features.feature_names(False)
    col = {n: i for i, n in enumerate(features.FEATURE_ORDER)}
    rows_feat, y = [], []
    for s in samples:
        x, sr = dataset.load_steady(s)
        rows_feat.append(features.extract(x, sr))
        y.append(s["label"])
    Xall = np.array([[r[n] for n in features.FEATURE_ORDER] for r in rows_feat])
    X = Xall[:, [col[n] for n in names]]

    model_l = load_model(os.path.join(MODELS, "model_blind.json"))
    model_nl = load_model(os.path.join(MODELS, "model_blind_nolevel.json"))

    w = stress.window_length_sweep(samples, model_l)
    stress.write_csv(os.path.join(RESULTS, "stress_window.csv"), w)
    print("[stress] A 窗口长度：")
    for r in w:
        print(f"          {r['window_s']:.2f}s  {r['n_windows']:3d} 窗   干净 {r['acc_clean']:.1%}"
              f"   {r['noise_snr']:.0f}dB 白噪 {r['acc_noisy']:.1%}")

    f = stress.few_shot_sweep(X, y, names)
    stress.write_csv(os.path.join(RESULTS, "stress_fewshot.csv"), f)
    print("[stress] B 少样本（每类 n 段训练）：")
    for r in f:
        print(f"          每类 {r['n_per_class']:2d} 段 → 准确率 {r['acc_mean']:.1%} "
              f"± {r['acc_sd']:.1%}（最差 {r['acc_min']:.1%}）")

    nv = stress.novelty_test(X, y)
    with open(os.path.join(RESULTS, "stress_novelty.json"), "w", encoding="utf-8") as fh:
        json.dump(nv, fh, ensure_ascii=False, indent=1)
    print(f"[stress] C 未见故障：正常误报 {nv['normal_false_alarm']:.0%}，"
          f"三类故障平均检出 {nv['mean_detect_rate']:.0%}")
    for k, v in nv["anomalies"].items():
        print(f"          {k:16s} 检出 {v['detected']}/{v['n']} = {v['detect_rate']:.0%}")

    g = stress.gain_shift_test(samples, model_l, model_nl)
    stress.write_csv(os.path.join(RESULTS, "stress_gain.csv"), g)
    print("[stress] D 增益漂移（浮点直乘，不削波，以隔离增益单一变量）：")
    for r in g:
        print(f"          {r['gain_db']:+5.0f} dB   含电平 {r['acc_with_level']:>4.0%}"
              f"   去掉电平 {r['acc_no_level']:>4.0%}   削波比例 {r['clipped_fraction']:.1%}")

    dur = (time.perf_counter() - t0) * 1000
    tm.log("stress", window=w, fewshot=f, novelty=nv, gain=g, dur_ms=round(dur, 1))
    return {"window": w, "fewshot": f, "novelty": nv, "gain": g}


def cmd_confidence(tm: Telemetry) -> dict:
    """统计"判错时的置信度"——抗噪实验里最容易漏掉的一半。"""
    t0 = time.perf_counter()
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import check_confidence

    check_confidence.main()
    dur = (time.perf_counter() - t0) * 1000
    tm.log("confidence", dur_ms=round(dur, 1), out="results/confidence_by_snr.csv")
    return {"ok": True}


def cmd_all(tm: Telemetry) -> dict:
    cmd_index(tm)
    cmd_features(tm)
    cmd_train(tm)
    cmd_blind(tm, "seal")
    cmd_blind(tm, "predict")
    cmd_blind(tm, "score")
    cmd_blind(tm, "verify")
    cmd_noise(tm)
    cmd_confidence(tm)
    cmd_stress(tm)
    cmd_export_csv(tm)
    cmd_figures(tm)
    return {"ok": True}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="手机声振诊断仪：特征提取 / 训练 / 盲测 / 抗噪")
    p.add_argument("--no-telemetry", action="store_true",
                   help="不写运行日志（仅调试用；正式跑请勿加，process_logs 是过程证据）")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    pf = sub.add_parser("features")
    pf.add_argument("--no-level", action="store_true", help="剔除与录音增益相关的绝对电平特征")
    pt = sub.add_parser("train")
    pt.add_argument("--C", type=float, default=None, help="跳过自动选择，直接指定正则强度倒数")
    pb = sub.add_parser("blind")
    pb.add_argument("action", choices=["seal", "predict", "score", "verify"])
    sub.add_parser("noise")
    sub.add_parser("confidence")
    sub.add_parser("stress")
    pe = sub.add_parser("export-csv")
    pe.add_argument("--rate", type=int, default=1000)
    sub.add_parser("figures")
    sub.add_parser("all")

    args = p.parse_args(argv)
    ensure_dirs()
    command = args.cmd if args.cmd != "blind" else f"blind_{args.action}"
    tm = _NullTelemetry(command) if args.no_telemetry else Telemetry(command, log_dir=LOGSDIR)
    stage = args.cmd
    try:
        if args.cmd == "index":
            cmd_index(tm)
        elif args.cmd == "features":
            cmd_features(tm, include_level=not args.no_level)
        elif args.cmd == "train":
            cmd_train(tm, C=args.C)
        elif args.cmd == "blind":
            cmd_blind(tm, args.action)
        elif args.cmd == "noise":
            cmd_noise(tm)
        elif args.cmd == "confidence":
            cmd_confidence(tm)
        elif args.cmd == "stress":
            cmd_stress(tm)
        elif args.cmd == "export-csv":
            cmd_export_csv(tm, rate=args.rate)
        elif args.cmd == "figures":
            cmd_figures(tm)
        elif args.cmd == "all":
            cmd_all(tm)
        tm.done(script="run.py")
        return 0
    except Exception as exc:  # noqa: BLE001
        tm.fail(stage, exc)
        print(f"[error] {stage}: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
