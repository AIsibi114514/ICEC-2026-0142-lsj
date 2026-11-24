"""分类器：标准化 + 多项逻辑回归 + CART 决策树。纯 numpy 实现。

为什么不用现成框架：
  ① 评审要在没有 sklearn 的机器上复现，少一个依赖少一个坑；
  ② 75 个样本、5 个类别，模型复杂度必须压低到能被"读出权重"的程度——
     面试时如果被问"凭什么判成 C3"，我要能指着某个特征的权重说话，
     而不是说"随机森林这么认为"。

两个模型互补：
  LogisticRegression  准确率的主力，输出可解释的概率与特征权重；
  DecisionTree        3 层 if-else，能直接印在纸上当"规则基线"，
                      用来验证"这件事有没有简单解"，也是嵌入式移植的形态。
"""

from __future__ import annotations

import json

import numpy as np

EPS = 1e-12


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=1, keepdims=True) + EPS)


class Standardizer:
    """按列 z-score。std 为 0 的列（常数特征）保留原值。"""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean = X.mean(axis=0)
        s = X.std(axis=0)
        s[s < 1e-9] = 1.0
        self.std = s
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.std is not None
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class LogisticRegression:
    """多项逻辑回归，L2 正则，Adam 优化。

    C 是正则强度的倒数（C 越小正则越强）。本次 75 样本 / 34 特征，
    选 C=0.3——在 5 折交叉验证里 C=0.1~3 的差别 <2 个百分点，
    取偏小的一侧以降低过拟合风险（取值扫描见 results/cv_summary.json）。
    """

    def __init__(self, C: float = 0.3, lr: float = 0.08, epochs: int = 1500, seed: int = 0) -> None:
        self.C = C
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.scaler = Standardizer()
        self.classes_: list[str] = []
        self.loss_history: list[float] = []

    def fit(self, X: np.ndarray, y: list[str]) -> "LogisticRegression":
        self.classes_ = sorted(set(y))
        idx = {c: i for i, c in enumerate(self.classes_)}
        Y = np.zeros((len(y), len(self.classes_)))
        for i, lab in enumerate(y):
            Y[i, idx[lab]] = 1.0
        Xs = self.scaler.fit_transform(X)
        n, d = Xs.shape
        k = len(self.classes_)
        rng = np.random.default_rng(self.seed)
        W = rng.normal(0, 0.01, size=(d, k))
        b = np.zeros(k)
        mW, vW = np.zeros_like(W), np.zeros_like(W)
        mb, vb = np.zeros_like(b), np.zeros_like(b)
        b1, b2 = 0.9, 0.999
        lam = 1.0 / self.C
        for t in range(1, self.epochs + 1):
            P = softmax(Xs @ W + b)
            loss = -np.sum(Y * np.log(P + EPS)) / n + lam / (2 * n) * np.sum(W**2)
            self.loss_history.append(float(loss))
            G = (P - Y) / n
            gW = Xs.T @ G + lam / n * W
            gb = G.sum(axis=0)
            mW = b1 * mW + (1 - b1) * gW
            vW = b2 * vW + (1 - b2) * gW**2
            mb = b1 * mb + (1 - b1) * gb
            vb = b2 * vb + (1 - b2) * gb**2
            mWh, vWh = mW / (1 - b1**t), vW / (1 - b2**t)
            mbh, vbh = mb / (1 - b1**t), vb / (1 - b2**t)
            W -= self.lr * mWh / (np.sqrt(vWh) + 1e-8)
            b -= self.lr * mbh / (np.sqrt(vbh) + 1e-8)
        self.W, self.b = W, b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.W is not None and self.b is not None
        return softmax(self.scaler.transform(X) @ self.W + self.b)

    def predict(self, X: np.ndarray) -> list[str]:
        P = self.predict_proba(X)
        return [self.classes_[i] for i in np.argmax(P, axis=1)]

    # ---- 可解释性 ----
    def feature_importance(self, names: list[str]) -> list[tuple[str, float]]:
        """每列权重在各类别上的绝对值最大值，降序。"""
        assert self.W is not None
        score = np.max(np.abs(self.W), axis=1)
        order = np.argsort(score)[::-1]
        return [(names[i], float(score[i])) for i in order]

    def class_weights(self, names: list[str], cls: str) -> list[tuple[str, float]]:
        """某一类的权重（正=推向该类）。"""
        assert self.W is not None
        j = self.classes_.index(cls)
        w = self.W[:, j]
        order = np.argsort(np.abs(w))[::-1]
        return [(names[i], float(w[i])) for i in order]

    # ---- 序列化 ----
    def to_dict(self, names: list[str]) -> dict:
        return {
            "type": "logistic",
            "features": names,
            "classes": self.classes_,
            "C": self.C,
            "epochs": self.epochs,
            "mean": self.scaler.mean.tolist(),
            "std": self.scaler.std.tolist(),
            "W": self.W.tolist(),
            "b": self.b.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LogisticRegression":
        m = cls(C=d["C"], epochs=d.get("epochs", 0))
        m.classes_ = d["classes"]
        m.scaler.mean = np.array(d["mean"])
        m.scaler.std = np.array(d["std"])
        m.W = np.array(d["W"])
        m.b = np.array(d["b"])
        return m


class DecisionTree:
    """CART 分类树，gini 不纯度，可导出为 if-else 文本。

    故意限制 max_depth=3：一个 75 样本的树如果长到 6 层，
    它记住的是噪声而不是规律。
    """

    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 3, max_splits: int = 24) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_splits = max_splits
        self.tree: dict | None = None
        self.classes_: list[str] = []

    # ---- 内部 ----
    @staticmethod
    def _gini(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return float(1.0 - np.sum(p**2))

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        best = (None, None, 1e9)
        for j in range(X.shape[1]):
            col = X[:, j]
            uniq = np.unique(col)
            if len(uniq) < 2:
                continue
            if len(uniq) > self.max_splits:
                qs = np.quantile(col, np.linspace(0.05, 0.95, self.max_splits))
                cands = np.unique(qs)
            else:
                cands = (uniq[:-1] + uniq[1:]) / 2
            for t in cands:
                left = col <= t
                nl, nr = int(left.sum()), int((~left).sum())
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                g = (nl * self._gini(y[left]) + nr * self._gini(y[~left])) / (nl + nr)
                if g < best[2] - 1e-12:
                    best = (j, float(t), g)
        return best

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        node = {"n": int(len(y)), "dist": self._dist(y)}
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            node["leaf"] = True
            node["pred"] = self.classes_[int(np.argmax(node["dist"]))]
            return node
        j, t, g = self._best_split(X, y)
        if j is None:
            node["leaf"] = True
            node["pred"] = self.classes_[int(np.argmax(node["dist"]))]
            return node
        node.update(leaf=False, feature=int(j), threshold=float(t), gini=float(g))
        left = X[:, j] <= t
        node["left"] = self._build(X[left], y[left], depth + 1)
        node["right"] = self._build(X[~left], y[~left], depth + 1)
        return node

    def _dist(self, y: np.ndarray) -> list[float]:
        counts = np.array([np.sum(y == c) for c in self.classes_], dtype=float)
        return (counts / (counts.sum() + EPS)).tolist()

    # ---- 对外 ----
    def fit(self, X: np.ndarray, y: list[str]) -> "DecisionTree":
        self.classes_ = sorted(set(y))
        self.tree = self._build(X, np.array(y), 0)
        return self

    def _leaf(self, node: dict, x: np.ndarray) -> dict:
        while not node.get("leaf"):
            node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        return node

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.tree is not None
        return np.array([self._leaf(self.tree, x)["dist"] for x in X])

    def predict(self, X: np.ndarray) -> list[str]:
        P = self.predict_proba(X)
        return [self.classes_[i] for i in np.argmax(P, axis=1)]

    def rules_text(self, names: list[str]) -> str:
        """把树打印成可直接移植到单片机的 if-else。"""
        assert self.tree is not None
        lines: list[str] = []

        def walk(node: dict, depth: int, cond: str) -> None:
            pad = "    " * depth
            if node.get("leaf"):
                dist = ", ".join(f"{c}:{p:.0%}" for c, p in zip(self.classes_, node["dist"]))
                lines.append(f"{pad}→ {node['pred']}   (n={node['n']}  {dist})")
                return
            f = names[node["feature"]]
            th = node["threshold"]
            lines.append(f"{pad}{cond}if {f} <= {th:.4f}:")
            walk(node["left"], depth + 1, "")
            lines.append(f"{pad}else:  # {f} > {th:.4f}")
            walk(node["right"], depth + 1, "")

        walk(self.tree, 0, "")
        return "\n".join(lines)

    def to_dict(self, names: list[str]) -> dict:
        return {
            "type": "cart",
            "features": names,
            "classes": self.classes_,
            "max_depth": self.max_depth,
            "tree": self.tree,
        }


def save_model(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)


def load_model(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
