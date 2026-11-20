# 合成日志夹具（非真实遥测）

本目录是**测试夹具**，不是运行记录。7 个 `SIM-*.jsonl` 覆盖 2026-11-20 ~ 2026-11-26，
共 316 条记录，由 `tools/simulate_logs.py` 生成后冻结在此，用于回归测试。

## 为什么它值得进仓库

它验证的是**代码**，不是**过程**：

- `tools/log_digest.py` 在 7 天形态下的报告结构（断档检测、p95、阈值告警）；
- 异常分布下不崩：某次空跑（`收件箱为空`）、某个阶段缺依赖（`stage=ocr / EngineMissing`）、
  单次耗时尖刺（11-25 的 300ms 基线）；
- 改动规则或指标算法后，可以直接 `--dir` 指向本目录对比报告差异，不必等 7 天。

## 三重标记（任何一条都足以识别）

| 标记 | 值 |
| --- | --- |
| 文件名 | `SIM-YYYY-MM-DD.jsonl`，不匹配 `process_logs/` 的 `YYYY-MM-DD.jsonl` |
| 记录字段 | 每条都带 `"simulated": true` |
| `host_hash` | `"simulate"`（真实日志是本机短哈希，如 `b7ac7fba`） |
| 生成时间 | 每次运行固定在 `08:00 / 12:30 / 21:30`，`tz` 为 `CST` |

## 用法

```bash
# 用夹具预演诊断报告（报告写在本目录的 digest_preview.md，不污染 iterations/）
python tools/log_digest.py --dir tests/fixtures/simulated_logs
```

一次真实运行的结果对照：

```bash
python tools/log_digest.py                      # 只读 process_logs/*.jsonl
```

## 三条硬约束

1. **不得改名搬进 `process_logs/`。** 那里只放程序实际运行生成的文件，文件名必须是
   `YYYY-MM-DD.jsonl`。`tools/simulate_logs.py` 对此有硬断言，`tests/test_fixture_integrity.py`
   会检查 `process_logs/` 里没有任何 `simulated` 记录。
2. **不得计入覆盖率。** 「真实天数 1 / 7」只看 `process_logs/`，与夹具无关。同一天
   （11-20）在两边都存在，`SIM-2026-11-20.jsonl` 是预演，`process_logs/2026-11-20.jsonl`
   才是那次 12:32 的真实运行。
3. **不得摘掉 `simulated` 标记。** 去掉标记不会让数据变真，只会让数据变假且不可辨识。

## 关于可复现性

生成器里的 `uuid4()` 未受 seed 控制，**重新生成不会得到逐字节相同的文件**
（`run_id` / `item_key` 会变）。所以这里放的是**冻结快照**：改夹具请有意为之并写清原因，
不要用一次新的生成结果去刷 diff。

生成命令（历史记录，不是让你重跑）：

```bash
python tools/simulate_logs.py --days 7 --end 2026-11-26 --seed 20261120
```

`docs/simulated_logs/2026-11-16_22/` 是更早一版预演（11-16 ~ 11-22，与项目窗口不符），
已被本目录取代，仍留在本机且被 `.gitignore` 排除，未提交。
