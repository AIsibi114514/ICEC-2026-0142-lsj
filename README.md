# ICEC-2026-0142-lsj

> 2026 级「智能·创造者」创新实验班 · 选拔考核提交仓库
> 深圳城市职业学院（深圳技师学院）信息与通信学院

## 我是谁

lsj，信息与通信学院 2026 级，学号 2024400142（后四位 0142）。

**一句话选题说明**：我选了第一阶段的两道题——一道向外（T08 用手机传感器分辨四种行走状态），一道向内（T10 给自己的日常工作效率做量化审计）。前者练"从噪声里找出可判定的量"，后者练"算出 AI 到底省了多少时间、在哪一步没省"。

## 提交内容

| 阶段 | 内容 | 目录 | 状态 |
| --- | --- | --- | --- |
| 一 | 课题08 · 多姿态步态加速度特征采集与对比 | [`01-Field-Discovery/T08-gait-acceleration/`](01-Field-Discovery/T08-gait-acceleration/) | ⬜ 待采集数据 |
| 一 | 课题10 · 日常数字任务耗时审计与 AI 批处理实证 | [`01-Field-Discovery/T10-digital-task-audit/`](01-Field-Discovery/T10-digital-task-audit/) | 🟡 AI 侧已实测，人工计时待填 |
| — | AI 协作过程留痕 | [`03-AI-Collaboration/`](03-AI-Collaboration/) | ✅ |
| 二 | 工坊开放式项目挑战（8 选 1） | `02-20Days-Challenge/` | ⬜ 未开始（10/23 起） |

## 两个课题的核心结论

### T08 步态识别：用步频，别用幅值

四种状态里，**静止 / 慢走 / 快跑**可以靠一条深度 4 层的阈值规则干净分离；**上下楼梯最难**，因为它的步频与幅值都跟慢走重叠，只能靠波形的高阶形态（峰度、偏度）区分。

最关键的判断是：**区分慢走与快跑要用频域主峰（步频），不能用加速度幅值。** 幅值受手机放置位置、体重影响大；步频是动作本身的固有频率，跨人跨设备更稳。这一条是整个实验的方法论核心。

产出：24 维特征提取脚本、可烧进单片机的 if-else 阈值规则、5 张可视化图。

### T10 耗时审计：AI 把"生产"压到零，没把"信任"压到零

实测：10 份课程通知，AI 端到端 **2.07 ms**，单样本 **0.053 ms**；字段准确率 63/63。

但报告里我把这个 100% 明确标注为**乐观偏差**——初版规则只有 84.1%，是在看过这 10 个样本后做了 6 轮定向修正才达到 100% 的。真实场景应按 80% 上下预期。

所以真实提速不是纯计算比值的数十万倍，而是从"人做 50 秒"到"人审 25 秒"，**约 2 倍**。

> **这个课题真正的产出不是"AI 很快"，而是一句能被验证的话：AI 把生产成本压到接近零，但没有把信任成本压到零。** 只报前一个数字的提效结论，都是不完整的。

## 目录结构

```
ICEC-2026-0142-lsj/
├── README.md                    # 本文件：个人简介与选题说明
├── 01-Field-Discovery/          # 第一阶段：实地测评
│   ├── README.md                # 选题理由与交付自查
│   ├── T08-gait-acceleration/   # 课题08（采集手册 / 报告 / 脚本 / 数据）
│   └── T10-digital-task-audit/  # 课题10（样本 / 脚本 / 实测 / 人工计时）
├── 02-20Days-Challenge/         # 第二阶段：工坊项目（待开始）
└── 03-AI-Collaboration/         # 关键 AI 提问与排障记录
```

## 怎么验证我说的都是真的

每个数字都能追溯到具体文件，命令如下：

```bash
# T08：先体检再出全部结果（需要先把实测数据放进 raw_data/）
cd 01-Field-Discovery/T08-gait-acceleration
python -m pip install -r scripts/requirements.txt
python scripts/analyze_gait.py --check-only
python scripts/analyze_gait.py
# → 输出 feature_table.csv / threshold_rules.json / figures/fig1~fig5

# T10：跑基准，重算耗时与准确率
cd 01-Field-Discovery/T10-digital-task-audit
python scripts/ai_extract.py
python scripts/run_benchmark.py --repeat 50
# → 输出 ai_timing.csv / field_accuracy.csv / benchmark_summary.json / figures/fig1~fig3
```

`raw_data/benchmark_summary.json` 里有 50 轮计时的完整分布，可以自己核对中位数与 p95。

## 两条自我约束

1. **不编数字。** 报告里凡是指不出源文件的数值，一律删掉。T08 的数据必须由我用手机实地采集，T10 的人工耗时必须由我用秒表实测——这两件事 AI 替不了，也不该替。
2. **不藏缺点。** T08 承认楼梯最难分，T10 主动交代 100% 是乐观偏差。第三阶段面试会追问数据边界和异常原因，藏起来的短板在面试现场会变成更大的问题。

## 联系

- 官方门户：icec.openkub.com
- 学院：深圳城市职业学院 信息与通信学院
- 校址：深圳市龙岗区龙岗街道五联社区将军帽路 1 号

---

© 2026 ICEC · 厚德 · 精技 · 善思 · 敢创
