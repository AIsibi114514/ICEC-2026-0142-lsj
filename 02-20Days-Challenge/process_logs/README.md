# process_logs —— 运行日志

> 目录里每一个 `YYYY-MM-DD.jsonl` 都是**脚本实际运行时写出来的**，没有一条是手写或补录的。
> 写入逻辑见 [`src/vibdiag/telemetry.py`](../src/vibdiag/telemetry.py)。

---

## 为什么要有这个目录

评审要的是"过程数据"，不是"结果截图"。截图能证明"某个时刻屏幕上显示过 100%"，但证明不了：
这条结果是哪次跑出来的、跑了多久、用的什么参数、中间有没有报错。

所以每个子命令跑完都会落一条 `run_end`（失败则落 `error`），把上面这些一次性记下来。

---

## 文件与字段

一天一个文件，每次运行 **append，从不覆盖**。

```json
{"run_id":"9f3c1a2b","ts":"2026-11-24T12:41:02+08:00","event":"train","status":"ok",
 "app_version":"1.0.0","command":"train","n_train":50,"n_test":25,"C":0.03,
 "cv_accuracy":1.0,"cv_macro_f1":1.0,"cv_accuracy_no_level":1.0,
 "tree_train_accuracy":1.0,"dur_ms":5137.2}
```

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `run_id` | str(8) | 本次运行的 id，`sha256(纳秒时间 + pid)` 前 8 位 |
| `ts` | ISO8601 | 本地时区时间戳，精确到秒 |
| `event` | str | 事件名：`index` / `features` / `train` / `blind_seal` / `blind_predict` / `blind_score` / `blind_verify` / `noise` / `stress` / `export_csv` / `figures` / `run_end` / `error` |
| `status` | str | `ok` 或 `error` |
| `app_version` | str | 代码版本 |
| `command` | str | 触发它的子命令 |
| `dur_ms` | float | 本步骤耗时（毫秒） |
| 其它 | — | 各命令自己的参数与结果（样本数、特征数、准确率、输出路径……） |

`run_end` 额外带 `events`（本次一共写了多少条）、`host`（主机名 sha256 前 8 位）、`env`（Python 版本与操作系统）。
`error` 带 `stage`、`error_type`、`msg`（截断到 300 字符）。

---

## 脱敏做了什么

| 项 | 处理 |
| --- | --- |
| 主机名 | 只存 `sha256(hostname)[:8]`，不存明文 |
| 文件路径 | **一律转成相对仓库根的路径**（`raw_audio/...`），不写 `C:\Users\...` |
| 音频内容 | 不记任何波形/样本值，只记统计量与文件名 |
| 个人信息 | 全程不涉及（没有用户数据、没有通知正文、没有同学姓名） |

相对路径这件事不只是脱敏，也是可复现性：换成任何一台机器 clone 下来，同一行日志的含义完全一样。

---

## 核对方法

```bash
# 今天跑了哪些命令、各花多久
python -c "import json,glob;[print(json.loads(l)['ts'], json.loads(l)['command'], json.loads(l)['dur_ms']) for f in glob.glob('process_logs/*.jsonl') for l in open(f,encoding='utf-8')]"

# 只看 train
python -c "import json,glob;[print(l.strip()) for f in glob.glob('process_logs/*.jsonl') for l in open(f,encoding='utf-8') if json.loads(l)['event']=='train']"
```

日志里的 `cv_accuracy` 必须与 `results/cv_summary.json` 的 `cv_5fold.accuracy` 一致；`blind_score` 的 `accuracy` 必须与 `results/blind_score.json` 一致。**对不上说明要么日志被改过，要么结果被改过。**

---

## 一个诚实的说明：本项目的日志量级与项目四不同

项目四（校园通知工具）要求"连续 7 天自动运行日志"，那是**按日历建仓、按天提交**的形态。

本项目是**离线实验型**，日志的性质不一样：它记录的是"每次分析跑了多久、用了什么参数、得到什么结果"，而不是"每天几点被定时任务唤醒"。

所以这里的日志：

- ✅ 能证明每份结果都是真跑出来的（时间戳、耗时、参数、run_id 齐全）；
- ✅ 能证明结果没被手工改过（日志与 results/ 双向可对照）；
- ❌ **不能**证明"连续 7 天在运行"——因为它本来就不是那个形态，我也没打算伪装成那个形态。

如果评审需要的是"过程连续性"，本项目提供的是**完整可复现的流水线**：`python run.py all` 一条命令即可重新生成 `results/` 与 `figures/` 下的全部产物。
