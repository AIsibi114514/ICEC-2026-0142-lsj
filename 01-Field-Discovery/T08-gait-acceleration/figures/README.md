# 图表目录

本目录的 5 张图**全部由 `scripts/analyze_gait.py` 自动生成**，没有任何手工绘制或修饰。

| 文件 | 内容 | 看什么 |
| --- | --- | --- |
| `fig1_waveform_all_states.png` | 5 种状态的三轴 + 合加速度波形 | 下楼的正负对称冲击尖峰 vs 上楼的单向推动 |
| `fig2_feature_boxplot.png` | 4 个关键特征的类间分布 | **慢走与上楼的箱体几乎完全重叠**——这就是它们只有 56% 召回的直接原因 |
| `fig3_confusion_matrix.png` | 阈值规则的混淆矩阵 | 上楼有 4 个窗口被判成慢走 |
| `fig4_freq_spectrum.png` | 各状态平均幅值谱 | 慢走 2.77 Hz 与快跑 2.85 Hz 几乎重合，步频判据失效 |
| `fig5_rule_flow.png` | 阈值决策树流程图 | 4 层规则、4 个特征，可直接移植到设备端 |

重新生成：

```bash
cd .. && python scripts/analyze_gait.py
```

图会覆盖本目录同名文件，`REPORT.md` 中的引用随之更新。
