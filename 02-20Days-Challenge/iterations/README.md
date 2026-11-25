# iterations/ —— 阶段二的三次实质迭代

> 目录规范（大纲 Step 0）要求 `02-20Days-Challenge/` 下具备 `iterations/`。此前这一层是缺的：
> 迭代只散落在 git commit 里，评审要自己 `git log` 才能拼出来。这里把它补成可读的记录。

## 收录标准

只收**改变了结论或改变了交付物**的迭代。改错别字、调格式、补注释不算。

| # | 迭代 | 触发问题 | 结论变化 | commit | 证据 |
| --- | --- | --- | --- | --- | --- |
| 01 | [单步试跑 → 首轮全流程](iteration-01-pipeline-first-run.md) | 一次 `all` 跑完不知道错在哪一步 | 拆成 9 个可单独重跑的子命令 | `7e57694` | 28 次运行日志（12:30–14:18） |
| 02 | [置信度实测](iteration-02-confidence-verification.md) | 报告初稿写了"失效时置信度会很高"，但**没测过** | 从"感觉"变成"判错时置信度从 0.375 涨到 1.000" | `811e2b3` | `results/confidence_by_snr.csv` |
| 03 | [规则树 vs 逻辑回归](iteration-03-tree-vs-logistic.md) | "4 个阈值就够，为什么要 34 维特征？" | 34 维的价值不在干净数据，在**退化条件** | `12e6eaf` | `results/baseline_compare.csv` |

另有一次不计入迭代的修正：`43e2983` 校正 README 的代码量（2876 行）与子命令数、并清掉 4 个文件误加的 BOM。它不改变结论，只让文档与代码对齐——但正是这类"数字对不上"最容易在面试现场被戳穿，所以一并留痕。

## 三次迭代的共同点

都不是"发现问题 → 改代码 → 变好"，而是**"发现一句话没有证据 → 补做实验 → 结果推翻了我原来的说法"**。

第 02 次尤其典型：初稿里那句"失效时置信度会很高"读起来非常合理，写进报告会让论证更顺。补做之后发现真实情况比我设想的极端得多（判错时置信度能到 1.000），于是 §6 从"一句推测"变成了一个可被追问的实测结论。

## 怎么核对

```bash
cd 02-20Days-Challenge

# 三次迭代的原始 diff
git show --stat 811e2b3
git show --stat 12e6eaf

# 迭代 02 的数据
cat results/confidence_by_snr.csv

# 迭代 03 的数据
cat results/baseline_compare.csv

# 迭代 01 的过程：28 次运行的时间线
python -c "import json;[print(json.loads(l)['ts'][11:19], json.loads(l)['run_id'], json.loads(l)['event']) for l in open('process_logs/2026-11-24.jsonl',encoding='utf-8')]" | head -30
```
