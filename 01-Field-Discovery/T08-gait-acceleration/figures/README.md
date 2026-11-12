# 图表目录

本目录的 5 张图**全部由脚本自动生成**，没有任何手工绘制或修饰：

| 文件 | 内容 | 看什么 |
| --- | --- | --- |
| `fig1_waveform_4states.png` | 四状态三轴+合加速度波形 | 跑步的包络密度明显高于慢走（步频证据） |
| `fig2_feature_boxplot.png` | 4 个关键特征的类间分布 | 箱体重叠越少，阈值越好定 |
| `fig3_confusion_matrix.png` | 阈值规则的混淆矩阵 | 哪两类最容易互相混淆 |
| `fig4_freq_spectrum.png` | 各状态平均幅值谱 | 慢走与快跑的主峰是否分离 |
| `fig5_rule_flow.png` | 阈值决策规则流程图 | 规则能否直接移植到设备端 |

**当前为空是正常的**：本仓库尚未放入实测数据，脚本没有可分析的对象。
按 `README.md` 第 3 节采集完 12 份 CSV 后执行：

```bash
python scripts/analyze_gait.py
```

5 张图即自动出现在本目录，`REPORT.md` 中的图片引用也随之生效。
