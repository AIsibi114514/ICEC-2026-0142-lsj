# 三次迭代的执行方案（对应评审第 ②项，20 分）

核心原则：**每次改动的触发条件都来自日志里的具体字段，不是"我想加个功能"。**
下面的方案是提前规划的骨架，真正动手时以当天日志的实际数字为准（数字可能和预估不同，
不同反而更好——说明是数据驱动而不是套模板）。

## 迭代总览

| 版本 | 计划日期 | 触发信号（来自日志） | 改动范围 | 预期日志变化 |
| --- | --- | --- | --- | --- |
| v0.1.0 | 11-20 | —（基线） | 文本通路 + 全链路埋点 + 日报 | 首次产生 `run_start`…`run_end` |
| v0.2.0 | 11-21~22 | `stage=ocr` 出现 `error_type=EngineMissing` ≥2 次 | `ocr.py` + 依赖 | `ocr` 事件从 error 转 ok，新增识别字数 |
| v0.3.0 | 11-23~24 | `log_digest` 报告 `miss_rate[deadline] ≥ 30%` | `rules.py` + 时间锚点 | `fields_missed` 中 deadline 占比下降 |
| v0.4.0 | 11-25~26 | 反馈中"还要手抄进日历" ≥3 人，或 `p95 > 800ms` | 新增 `ics.py` | 新增 `write_ics` 事件与耗时 |

---

## v0.2.0：打通截图通路

**依据**：`process_logs/` 里出现
```json
{"event":"error","stage":"ocr","error_type":"EngineMissing","file":"shot.png"}
```
出现两次以上说明不是偶发，而是环境缺失——这是最典型、最该先修的缺陷。

**改动**
- `pip install pytesseract pillow`，安装 tesseract 主程序（含 `chi_sim` 语言包）
- `ocr.py`：`lang` 参数降级链 `chi_sim+eng` → `eng`，语言包缺失时不再整体失败
- `cli.py`：新增 `--no-ocr`，便于对比"开/关截图通路"的日志差异

**验收标准**：改造后同一天的日志中 `event=ocr` 且 `status=ok` 至少 1 条，
`run_end.errors` 相比改造前下降。

---

## v0.3.0：补规则 + 修正去重键

**依据**：`python tools/log_digest.py` 输出的第 4 节表格。若
`deadline` 漏抽率 ≥ 30%，说明 `rules.py` 的截止提示词覆盖不足（真实群通知里
"今晚八点前""赶在周日前"这类表达很多）。

**改动**
- `rules.py`：增补 `DEADLINE_CUES`（"赶在""务必在""不晚于"等）；时间词表补中文数字（"七点半"）
- `extractor.py`：时间锚点从"文本中最先出现"改为"距离日期最近"（v0.1.0 已修，v0.3.0 复审）
- `store.py`：去重键加入 `deadline` 归一化，避免同一通知改期后被当成两条

**验收标准**：改造前后各跑一次 digest，把两张"字段漏抽率"表并列放进本文件，
deadline 漏抽率下降 ≥ 10 个百分点。

---

## v0.4.0：输出到日历 + 收尾

**依据**：8 位用户的反馈表里，若"看日报还是要手动抄进手机日历"出现 ≥3 次，
这就是最高频的抱怨，而不是我凭空想的"加个导出功能"。

**改动**
- 新增 `ics.py`：把 `deadline` 转成 VALARM 提前 2 小时提醒的 `.ics`
- `report.py`：日报顶部加一行"日历已导出：output/today.ics"
- `cli.py`：`run` 末尾自动导出

**验收标准**：`.ics` 能被手机日历导入；日志新增 `write_ics` 事件与耗时；
至少 3 位用户确认导入成功（截图存 `docs/feedback/`）。

---

## 每次迭代的 Git 流程（必须按顺序）

```bash
# 0) 改动前：先出一份诊断报告，作为"改动依据"存档
python run.py digest

# 1) 改代码……

# 2) 本地验证
python run.py selftest
python run.py run

# 3) 提交（commit message 必须引用日志证据）
git add -A
git commit -m "feat(v0.2.0): 启用截图OCR通路

依据: 11-21~11-22 日志中 stage=ocr / EngineMissing 共 4 次
改动: ocr.py 增加 lang 降级链; cli.py 新增 --no-ocr
效果: 11-23 日志 ocr 事件 status=ok x3, run_end.errors 4->0"

# 4) 打标签，导出 diff 归档
git tag v0.2.0
git diff v0.1.0 v0.2.0 > iterations/0X-v0.1.0-to-v0.2.0.diff
git add iterations/ && git commit -m "docs: 归档 v0.1.0 -> v0.2.0 diff"
```

**注意**：先 `git tag` 再导出 diff，标签顺序不能乱；否则 diff 内容会混入无关文档改动，
看起来就不像"针对某个问题的一次改动"。

## 归档文件命名

```
iterations/
├── README.md
├── iteration-plan.md                    # 本文件
├── 01-log-digest-before-v0.2.0.md       # 改动前的诊断报告
├── 02-v0.1.0-to-v0.2.0.diff             # 第一次迭代完整 diff
├── 03-log-digest-before-v0.3.0.md
├── 04-v0.2.0-to-v0.3.0.diff
├── 05-log-digest-before-v0.4.0.md
└── 06-v0.3.0-to-v0.4.0.diff
```

这样评审打开 `iterations/` 就能看到「诊断 → 改动 → 效果」三件套，
而不是一堆没有前因后果的代码。
