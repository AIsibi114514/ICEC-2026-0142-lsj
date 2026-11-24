"""vibdiag —— 手机声振诊断仪核心库。

五个模块各管一件事，互不越界：

    audio.py       读写 WAV、裁剪稳定段、分帧、加噪、带通
    features.py    时域 + 频域特征提取（唯一对外契约：FEATURE_ORDER）
    model.py       标准化器 / 多项逻辑回归 / CART 决策树（纯 numpy）
    evaluate.py    混淆矩阵、每类 P-R-F1、Wilson 置信区间
    telemetry.py   运行日志（JSONL），每个子命令跑完都留痕

设计上刻意不引入 scipy / sklearn：
本机只有 Python 3.10+ 与 numpy，少一个依赖，评审就少一步装不上的风险。
"""

__version__ = "1.0.0"

CONDITIONS = [
    "C1_normal_low",
    "C2_normal_high",
    "C3_imbalance",
    "C4_obstruction",
    "C5_rubbing",
]

CONDITION_ZH = {
    "C1_normal_low": "正常·低档",
    "C2_normal_high": "正常·高档",
    "C3_imbalance": "扇叶轻微失衡",
    "C4_obstruction": "进风侧遮挡",
    "C5_rubbing": "网罩擦碰",
}
