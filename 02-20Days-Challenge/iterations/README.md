# iterations/ —— 迭代留痕目录

本目录存放「日志 → 依据 → 改动」的完整证据链，对应评审第 ② 项（20 分）。

## 每次迭代归档三样东西

1. `log_digest_YYYYMMDD.md`：改动**前**由 `python run.py digest` 自动生成的诊断报告，
   里面明确写了"哪个指标异常、建议改什么"；
2. `0X-vA-to-vB.diff`：`git diff vA vB` 的完整输出，评审可直接看改了哪几行；
3. commit message 中引用具体日期与 `run_id`，例如：
   `fix(v0.3.0): 补全截止时间规则 (依据: 11-23 日志 deadline 漏抽 42%)`

## 为什么 diff 单独存一份

GitHub 的 commit 页面能看到 diff，但评审要在手机上翻阅时，单独的 `.diff` 文件更容易打开；
同时也证明这些改动是**分次**产生的，而非最后一次性 push 全部代码。

## 查看方式

```bash
# 看某次迭代改了什么
cat iterations/02-v0.1.0-to-v0.2.0.diff

# 在仓库里还原当时的完整状态
git checkout v0.2.0
```
