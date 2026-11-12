# 图表目录

本目录的图由脚本自动生成，不手工制作：

- `fig1_per_sample_timing.png` —— 逐样本耗时对比（人工数据填进 `manual_run/timing_sheet.csv` 后自动变成双系列）
- `fig2_field_accuracy.png` —— 字段级准确率
- `fig3_amortization.png` —— 摊销曲线（需先填 `manual_run/audit_config.csv`）

重新生成：`python scripts/run_benchmark.py --repeat 50`
