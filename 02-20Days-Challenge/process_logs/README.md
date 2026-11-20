# 运行日志规范（Telemetry Logs）

本目录是评审第一项（25 分）的物证。规则：**每次运行追加写入当天文件**，一天一个 `YYYY-MM-DD.jsonl`，
一行一个 JSON，跨天不覆盖、不手工编辑。

## 为什么这样设计

| 评审要求 | 本目录如何满足 |
| --- | --- |
| 连续 ≥7 天 | 一天一个文件，`git log` 的提交时间与文件日期互相印证 |
| 自动化运行 | 由 Windows 任务计划每天定时触发（见 `tools/register_task.ps1`），不靠手动点 |
| 脱敏时间戳 | `ts` 为 ISO8601 本地时区、精确到秒；不含毫秒级行为指纹；主机名只存 hash |
| 调用频次 | 每条事件带 `event` 名；`run_end` 汇总 `calls` 与 `counters` |
| 耗时 | 每个耗时事件带 `dur_ms`（浮点毫秒），可直接算 p50/p95 |
| 报错记录 | `error` 事件带 `stage` / `error_type` / `msg`，单文件失败不中断整天运行 |

## 事件字典

| `event` | 出现时机 | 关键字段 |
| --- | --- | --- |
| `run_start` | 每次运行开始 | `event_date` `host_hash` `py` `tz` |
| `inbox_scan` | 扫描收件箱 | `files` |
| `read_text` | 读取一份文本 | `file` `bytes` `dur_ms` |
| `ocr` | 截图上走 OCR | `dur_ms`；失败时另发 `error` |
| `ocr_skipped` | OCR 不可用被跳过 | `fallback` |
| `split_notices` | 一条文本切成 N 条通知 | `notices` |
| `extract_notice` | 单条通知抽取 | `dur_ms` `source` |
| `notice_parsed` | 抽取结果摘要 | `category` `priority` `confidence` `fields_found` `fields_missed` |
| `store_upsert` | 写入数据库 | `count` `dur_ms` |
| `dedup` | 去重统计 | `new` `dup` `ratio` |
| `write_report` / `write_ledger` / `write_excerpts` | 产出文件 | `dur_ms` |
| `warning` | 非致命异常（如收件箱为空） | `stage` `msg` |
| `error` | 致命或已降级的异常 | `stage` `error_type` `msg` |
| `run_end` | 每次运行结束 | `dur_ms` `calls` `errors` `warnings` `counters` `exit_code` |
| `web_extract` | 网页端解析一次（无 `run_start`/`run_end`） | `source=web` `notices` `saved` `dur_ms` `avg_confidence` |

## 网页端的调用怎么记

网页（`python run.py web`）上的每次解析只写一条 `web_extract`，**不写** `run_start`/`run_end`。
原因是这两类数据要分开看：

- `run_start` / `run_end` = **程序自动化运行**，7 天连续性的证据，一天应当只有 1~2 次；
- `web_extract` = **真人点击**，8 位用户真实使用的证据，一天可能十几次。

混在一起会让"自动化运行次数"虚高，评审一眼就能看出不对。
`tools/log_digest.py` 把两者分别统计，报告里第 2 节是自动化，第 6 节是网页调用。

## 脱敏做了什么

1. **正文不入库**：日志只记录 `sha256(原文)[:12]`、字符数和字段统计，不记录通知原文。
2. **正则替换**：手机号 → `[PHONE]`，身份证 → `[IDCARD]`，8–12 位数字（学号/QQ）→ `[NUMID]`，
   邮箱 → `[EMAIL]`，`C:\Users\xxx` 路径 → `[HOME]`，Bearer Token → `[TOKEN]`。
3. **身份脱敏**：不记录用户名，主机名只存 8 位 hash。
4. **时间精度**：保留到秒，够还原操作时序，但不构成行为画像。

## 怎么自证没摆拍

```bash
# 每天的日志文件日期应与提交时间一致
git log --format="%ad %s" --date=iso -- process_logs/

# 某天日志的记录条数、调用次数
python tools/log_digest.py

# 抽查某次运行的完整事件链（把 run_id 换成日志里的任意值）
grep '"run_id": "a1b2c3d4"' process_logs/*.jsonl
```

如果某天确实没有新通知，日志里会有 `inbox_scan` + `warning: 收件箱为空` + `run_end`，
这比伪造数据可信得多——**空跑也是运行记录**。
