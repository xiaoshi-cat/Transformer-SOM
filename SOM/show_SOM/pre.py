import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# 设置风格
plt.style.use('default')

# 创建画布和子图布局
fig = plt.figure(figsize=(16, 9), facecolor='#f0f2f5') # 模拟软件背景色
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.5, 1.5], height_ratios=[1, 1])

# --- 1. 左侧：数据输入面板 (Input Panel) ---
ax_input = fig.add_subplot(gs[:, 0])
ax_input.set_facecolor('white')
ax_input.set_title("Step 1: Water Quality Data Entry", fontsize=14, pad=20, loc='left', fontweight='bold')

# 模拟输入框字段
features = ['Ca (mg/L)', 'Mg (mg/L)', 'Na (mg/L)', 'HCO3 (mg/L)', 'Cl (mg/L)', 'SO4 (mg/L)', 'TH (mg/L)', 'TA (mg/L)', 'PH']
# 模拟填入的一些数值
mock_values = ['55.2', '34.1', '8.5', '75.0', '9.2', '14.5', '250.1', '230.5', '8.2']

y_pos = np.linspace(0.85, 0.15, len(features))
for i, (feat, val) in enumerate(zip(features, mock_values)):
    # 标签
    ax_input.text(0.05, y_pos[i], feat, fontsize=11, va='center')
    # 模拟输入框 (画一个矩形框)
    rect = patches.Rectangle((0.45, y_pos[i]-0.03), 0.5, 0.06, linewidth=1, edgecolor='#d9d9d9', facecolor='#fafafa')
    ax_input.add_patch(rect)
    # 模拟填入的数字
    ax_input.text(0.47, y_pos[i], val, fontsize=11, va='center', color='#333333')

# 模拟按钮 - 使用 FancyBboxPatch 实现圆角
# 注意：boxstyle="round,pad=0.02" 控制圆角和内边距
button_rect = patches.FancyBboxPatch((0.05, 0.02), 0.9, 0.08, boxstyle="round,pad=0.02", linewidth=0, facecolor='#1890ff', edgecolor='none')
ax_input.add_patch(button_rect)
ax_input.text(0.5, 0.06, "Analyze / Identify Source", fontsize=13, color='white', va='center', ha='center', fontweight='bold')

ax_input.set_xlim(0, 1)
ax_input.set_ylim(0, 1)
ax_input.axis('off')


# --- 2. 右侧上部：主要结论区 (Main Result) ---
ax_result = fig.add_subplot(gs[0, 1:])
ax_result.set_facecolor('white')
ax_result.set_title("Step 2: Identification Result (Primary Prediction)", fontsize=14, pad=15, loc='left', fontweight='bold')

# 模拟主要结论框
# 根据预测结果改变颜色 (例如奥灰水用高危红色)
result_color = '#ff4d4f' # 红色高危
result_text = "Ordovician Limestone Water\n"

# 使用 FancyBboxPatch 替换 Rectangle 以支持圆角
res_rect = patches.FancyBboxPatch((0.05, 0.2), 0.9, 0.6, boxstyle="round,pad=0.05", linewidth=0, facecolor=result_color, alpha=0.1)
ax_result.add_patch(res_rect)
# 左侧强调色条
bar_rect = patches.Rectangle((0.05, 0.2), 0.02, 0.6 + 0.05, linewidth=0, facecolor=result_color)
ax_result.add_patch(bar_rect)

ax_result.text(0.1, 0.5, result_text, fontsize=24, color=result_color, va='center', fontweight='bold')
ax_result.text(0.1, 0.25, "Risk Level: High Risk (Requires Immediate Attention)", fontsize=12, color='#666666', va='center')

ax_result.set_xlim(0, 1)
ax_result.set_ylim(0, 1)
ax_result.axis('off')


# --- 3. 右侧下部左：概率详情 (Probability Breakdown) ---
ax_prob = fig.add_subplot(gs[1, 1])
ax_prob.set_facecolor('white')
ax_prob.set_title("Detailed Probability Analysis (Confidence)", fontsize=12, pad=10, loc='left', fontweight='bold')

# 模拟概率数据
labels = ['Ordovician', 'Old Goaf', '13-Coal', '12-Coal']
probs = [0.885, 0.092, 0.015, 0.008]
colors = ['#ff4d4f', '#faad14', '#1890ff', '#52c41a']

y_pos_prob = np.arange(len(labels))

# 绘制水平条形图
bars = ax_prob.barh(y_pos_prob, probs, color=colors, height=0.5, alpha=0.8)
ax_prob.set_yticks(y_pos_prob)
ax_prob.set_yticklabels(labels, fontsize=11)
ax_prob.set_xlim(0, 1.1)
ax_prob.set_xlabel("Probability Score (0.0 - 1.0)")
ax_prob.invert_yaxis() # 让最大的在上面

# 在条形图末尾添加百分比文字
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax_prob.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{probs[i]*100:.1f}%',
                 va='center', fontsize=10, fontweight='bold', color=colors[i])

ax_prob.spines['right'].set_visible(False)
ax_prob.spines['top'].set_visible(False)
ax_prob.grid(axis='x', linestyle='--', alpha=0.5)


# --- 4. 右侧下部右：SOM 可视化定位 (SOM Visualization) ---
ax_som = fig.add_subplot(gs[1, 2])
ax_som.set_facecolor('white')
ax_som.set_title("Topological Mapping (Visual Evidence)", fontsize=12, pad=10, loc='left', fontweight='bold')

# 模拟 SOM 底图 (用一个简单的梯度代替实际的 U-Matrix 背景)
# 实际项目中这里应放置你的 som_cluster_map.png 作为背景
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))
ax_som.imshow(gradient, aspect='auto', cmap='viridis', extent=[0, 15, 0, 15], alpha=0.3)

# 模拟不同水源的区域中心 (在底图上画几个圈表示势力范围)
circles = [
    patches.Circle((3, 12), 2.5, color='#ff4d4f', alpha=0.4, label='Ordovician Zone'),
    patches.Circle((12, 3), 2.5, color='#faad14', alpha=0.4, label='Old Goaf Zone'),
    patches.Circle((8, 8), 3, color='#1890ff', alpha=0.3, label='Sandstone Zone'),
]
for circ in circles:
    ax_som.add_patch(circ)

# 关键：标出当前样本的位置 (模拟投射结果)
# 假设这个样本被识别为奥灰水，它应该落在奥灰水区域附近
sample_x, sample_y = 3.5, 11.5
ax_som.plot(sample_x, sample_y, 'rX', markersize=18, markeredgewidth=3, label='Current Sample')
ax_som.text(sample_x+0.5, sample_y-0.5, "You are here", color='red', fontweight='bold')

ax_som.set_xlim(0, 15)
ax_som.set_ylim(0, 15)
ax_som.set_xticks([])
ax_som.set_yticks([])
ax_som.legend(loc='lower right', fontsize=9)
ax_som.set_xlabel("SOM Grid X")
ax_som.set_ylabel("SOM Grid Y")

# 调整整体布局
plt.tight_layout()
plt.subplots_adjust(wspace=0.2, hspace=0.3)

# 保存并显示
plt.savefig('system_prototype_mockup.png', dpi=150, bbox_inches='tight')
plt.show()