import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle
import numpy as np

# ==========================================
# 配置区域
# ==========================================
# 设置输出图片的文件名
OUTPUT_FILENAME = 'mine_water_framework_en.png'

# 定义专业配色方案 (使用十六进制颜色码)
colors = {
    'layer_bg_text': '#546e7a',  # 层级标签文字颜色 (深蓝灰)
    'box_data': '#e3f2fd',  # 数据层框体颜色 (浅蓝)
    'box_proc': '#fff9c4',  # 处理层框体颜色 (浅黄)
    'core_bg': '#f1f8e9',  # 核心层大背景颜色 (浅绿)
    'core_box': '#ffffff',  # 核心算法小框颜色 (纯白)
    'latent_box': '#eeeeee',  # 潜在特征框颜色 (浅灰)
    'box_eval': '#ffe0b2',  # 评估层框体颜色 (浅橙)
    'box_out': '#ffcc80',  # 输出层框体颜色 (深橙)
    'text_main': '#263238',  # 主要文字颜色 (深灰)
    'text_sub': '#455a64',  # 次要文字颜色 (稍浅灰)
    'arrow': '#37474f',  # 箭头颜色 (深灰)
    'core_border': '#66bb6a'  # 核心层边框颜色 (绿色)
}


# ==========================================
# 绘图辅助函数定义
# ==========================================
def add_box(ax, x, y, w, h, text, title=None, color='#ffffff', fontsize=9, title_fontsize=11):
    """
    辅助函数：在画布上绘制一个带有标题和文本内容的圆角矩形框。
    参数:
    - ax: matplotlib的坐标轴对象
    - x, y: 框左下角的坐标
    - w, h: 框的宽度和高度
    - text: 框内的主要文本内容 (英文)
    - title: 框的标题 (英文, 可选)
    - color: 框的填充颜色
    """
    # 1. 绘制阴影效果 (稍微偏移的灰色框)
    shadow = FancyBboxPatch((x + 0.08, y - 0.08), w, h, boxstyle="round,pad=0.2",
                            fc='#cfd8dc', ec='none', alpha=0.5, mutation_scale=1, zorder=1)
    ax.add_patch(shadow)

    # 2. 绘制主框体
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                         fc=color, ec='#90a4ae', lw=1.5, mutation_scale=1, zorder=2)
    ax.add_patch(box)

    # 3. 计算文本中心位置
    cx = x + w / 2
    cy = y + h / 2

    # 4. 添加文本
    if title:
        # 如果有标题，标题靠上加粗，正文靠下
        ax.text(cx, y + h - 0.2, title, ha='center', va='top',
                fontsize=title_fontsize, fontweight='bold', color=colors['text_main'], zorder=3)
        ax.text(cx, y + h / 2 - 0.1, text, ha='center', va='center',
                fontsize=fontsize, color=colors['text_sub'], linespacing=1.3, zorder=3)
    else:
        # 如果没标题，正文居中显示
        ax.text(cx, cy, text, ha='center', va='center',
                fontsize=fontsize + 1, fontweight='bold', color=colors['text_main'], zorder=3)


def add_arrow(ax, x_start, y_start, x_end, y_end, lw=2):
    """
    辅助函数：绘制连接箭头。
    """
    # 使用 FancyArrowPatch 创建样式更好的箭头
    arrow = patches.FancyArrowPatch(
        (x_start, y_start), (x_end, y_end),
        arrowstyle='-|>,head_length=10,head_width=8',  # 箭头样式
        color=colors['arrow'], lw=lw, zorder=1  # 箭头颜色和线宽
    )
    ax.add_patch(arrow)


# ==========================================
# 主绘图函数
# ==========================================
def draw_framework_en():
    # 1. 创建画布
    # figsize设置图片比例 (宽13, 高16)
    fig, ax = plt.subplots(figsize=(13, 16))
    # 设置坐标轴范围 (用于定位元素)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 16)
    ax.axis('off')  # 隐藏坐标轴刻度

    # ====================
    # 绘制左侧层级标签 (Layer Labels)
    # ====================
    # 定义5个层级的Y坐标和英文名称
    layer_configs = [
        (14.5, "L1: Data Perception Layer"),  # 数据感知层
        (12.0, "L2: Data Preprocessing Layer"),  # 数据预处理层
        (8.5, "L3: Core Algorithm Layer (TAE-SOM)"),  # 核心算法层
        (4.5, "L4: Decision & Evaluation Layer"),  # 决策评估层
        (1.5, "L5: Application Output Layer")  # 应用输出层
    ]

    for y_pos, name in layer_configs:
        # 添加层级名称文本
        ax.text(0.2, y_pos, name, ha='left', va='center', fontsize=12,
                fontweight='bold', color=colors['layer_bg_text'], style='italic')
        # 添加层级分割虚线
        ax.plot([0.2, 12.8], [y_pos - 0.4, y_pos - 0.4], color='#cfd8dc', lw=1.5, ls='--', zorder=0)

    # ====================
    # L1: 数据感知层 (Data Perception)
    # ====================
    # 绘制数据源框
    add_box(ax, 4.5, 13.8, 4.0, 1.4,
            "Raw Hydrochemical Dataset (67 Samples)\nFeatures: Na+, Ca2+, Mg2+, Cl-, \nSO4 2-, HCO3-, pH, Hardness",
            title="Data Acquisition", color=colors['box_data'])

    # 向下的箭头 (L1 -> L2)
    add_arrow(ax, 6.5, 13.7, 6.5, 13.2)

    # ====================
    # L2: 数据预处理层 (Data Preprocessing)
    # ====================
    # 1. 数据集划分框 (Split)
    add_box(ax, 1.5, 11.5, 4.0, 1.4,
            "Train Set: 40 samples\nTest Set: 27 samples\n(Random Split 6:4)",
            title="Data Partitioning", color=colors['box_proc'])

    # 2. 数据标准化框 (Z-score)
    add_box(ax, 7.5, 11.5, 4.0, 1.4,
            "Z-score Standardization\n(Zero Mean, Unit Variance)\nRemoving Scale Bias",
            title="Feature Scaling", color=colors['box_proc'])

    # L2层内箭头：数据源 -> 划分 -> 标准化
    add_arrow(ax, 6.5, 13.1, 3.5, 12.9)  # 源 -> 划分
    add_arrow(ax, 5.6, 12.2, 7.4, 12.2)  # 划分 -> 标准化

    # 向下的箭头 (L2 -> L3)
    add_arrow(ax, 9.5, 11.4, 9.5, 10.8)

    # ====================
    # L3: 核心算法引擎层 (Core Algorithm - TAE+SOM) --- 重点!
    # ====================
    # 1. 绘制核心层大背景框 (绿色虚线框)
    core_bg_patch = FancyBboxPatch((1.0, 5.5), 11.0, 5.2, boxstyle="round,pad=0.2",
                                   ec=colors['core_border'], fc=colors['core_bg'], lw=2.5, linestyle='--', zorder=0)
    ax.add_patch(core_bg_patch)
    # 核心层大标题
    ax.text(6.5, 10.3, "Deep Coupled Model Engine (TAE feature extraction + SOM clustering)",
            ha='center', fontsize=13, fontweight='bold', color='#2e7d32')

    # 2. 绘制 Transformer 自编码器 (TAE) 模块
    # 左侧：编码器 (Encoder)
    add_box(ax, 1.5, 6.5, 2.8, 3.0,
            "Multi-head Attention\nPosition-wise FFN\nLayer Normalization",
            title="TAE Encoder\n(Feature Capture)", color=colors['core_box'])

    # 中间：潜在特征空间 (Latent Space) - 你的代码核心交互点
    add_box(ax, 5.0, 7.5, 1.8, 1.2,
            "Latent\nVectors\n(Low-dim)", color=colors['latent_box'], fontsize=10)

    # 右侧：解码器 (Decoder) - 用于重构损失计算
    add_box(ax, 1.5, 8.3, 2.8, 0.8, "(Reconstruction Loss)", color=colors['core_box'], fontsize=8)  # 简化显示

    # TAE内部箭头
    add_arrow(ax, 4.4, 8.0, 4.9, 8.0)  # Encoder -> Latent

    # 3. 绘制 SOM 自组织映射模块
    add_box(ax, 7.5, 6.5, 4.0, 3.0,
            "Competitive Learning Layer\nTopological Neighborhood (e.g., 6x6 grid)\nBMU (Best Matching Unit) Search",
            title="SOM Neural Network\n(Topological Clustering)", color=colors['core_box'])

    # 核心层间箭头：潜在特征 -> SOM
    add_arrow(ax, 6.9, 8.0, 7.4, 8.0)

    # 向下的箭头 (L3 -> L4)
    add_arrow(ax, 9.5, 6.4, 9.5, 5.5)

    # ====================
    # L4: 决策与评估层 (Decision & Evaluation)
    # ====================
    # 1. 性能指标框
    add_box(ax, 1.5, 3.5, 4.5, 1.6,
            "Confusion Matrix Analysis\nAccuracy / Precision / Recall / F1-Score\n(Train vs Test Performance)",
            title="Model Evaluation Metrics", color=colors['box_eval'])

    # 2. 预测推理框
    add_box(ax, 7.0, 3.5, 4.5, 1.6,
            "Mapping new samples to SOM Topology\nDetermining Water Source Category\nCalculating Confidence Level",
            title="Predictive Inference", color=colors['box_eval'])

    # L4层内箭头：推理 -> 指标计算
    add_arrow(ax, 6.9, 4.3, 6.1, 4.3)

    # 向下的箭头 (L4 -> L5)
    add_arrow(ax, 9.25, 3.4, 9.25, 2.5)

    # ====================
    # L5: 应用输出层 (Application Output)
    # ====================
    # 最终结果终端框
    add_box(ax, 4.0, 0.5, 5.0, 1.6,
            "> Result: [Ordovician Limestone Water]\n> Confidence: [High (92%)]\n> Action: [Trigger Alarm / Normal]",
            title="Identification Terminal Output", color=colors['box_out'], fontsize=10)

    # ====================
    # 添加总标题 (顶部)
    # ====================
    ax.set_title("Overall Framework of Mine Water Inrush Source Identification System Based on TAE-SOM Coupled Model",
                 fontsize=15, fontweight='bold', pad=30, color='#1a237e')

    # 保存图片并显示
    # dpi=300 保证高清晰度用于文档打印
    plt.savefig(OUTPUT_FILENAME, dpi=300, bbox_inches='tight')
    print(f"Success! Diagram saved as: {OUTPUT_FILENAME}")
    plt.show()


# 运行绘图函数
if __name__ == '__main__':
    draw_framework_en()