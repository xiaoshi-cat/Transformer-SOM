import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.manifold import TSNE
from minisom import MiniSom
import os
from matplotlib.patches import Patch

# 配置
plt.style.use('seaborn-v0_8-whitegrid')
# 解决中文显示问题（可选，视环境而定）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = "Ablation_SOM_Results"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def load_data():
    # 检查文件是否存在
    if not os.path.exists('new_train_data.csv') or not os.path.exists('new_test_data.csv'):
        print("❌ 错误: 未找到数据文件，请确保 new_train_data.csv 和 new_test_data.csv 在当前目录下。")
        return None, None, None, None, None

    train_df = pd.read_csv('new_train_data.csv')
    test_df = pd.read_csv('new_test_data.csv')

    feature_cols = ['Ca', 'Mg', 'Na', 'HCO3', 'Cl', 'SO4', 'TH', 'TA', 'PH']
    target_col = 'Label'

    # 标签编码
    le = LabelEncoder()
    full_labels = pd.concat([train_df[target_col], test_df[target_col]], axis=0)
    le.fit(full_labels)
    y_train = le.transform(train_df[target_col])
    y_test = le.transform(test_df[target_col])

    # Z-score 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)

    return X_train, y_train, X_test, y_test, le.classes_


def calculate_metrics(y_true, y_pred, set_name):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    print(f"[{set_name}] Acc: {acc:.4f}, Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
    return acc, p, r, f1


def plot_som_activation(som, data, filename):
    """绘制 SOM 激活频率图 (Hit Map) - 统一风格"""
    plt.figure(figsize=(8, 7))
    frequencies = som.activation_response(data)
    sns.heatmap(frequencies.T, cmap='Blues', linewidths=0.5, annot=True, fmt='.0f', cbar_kws={'label': 'Hit Count'})
    plt.title('SOM Activation Map (Sample Hits)', fontsize=14)
    plt.xlabel('SOM X')
    plt.ylabel('SOM Y')
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 激活图已保存: {filename}")


def plot_som_labels(node_map, som_shape, class_names, filename):
    """绘制 SOM 标签分布图 (多数投票) - 统一风格"""
    label_grid = np.full((som_shape[1], som_shape[0]), -1)
    for (x, y), label in node_map.items():
        label_grid[y, x] = label

    # 使用自定义配色：粉、绿、蓝、紫 (对应类索引 0,1,2,3)
    custom_colors = ['#e377c2', '#2ca02c', '#1f77b4', '#9467bd']  # Pink, Green, Blue, Purple
    # 如果类别超过4种，后续颜色使用默认
    if len(class_names) > 4:
        custom_colors += sns.color_palette("husl", len(class_names) - 4).as_hex()

    cmap = sns.color_palette(custom_colors[:len(class_names)])
    mask = (label_grid == -1)

    plt.figure(figsize=(9, 7))
    ax = sns.heatmap(label_grid, mask=mask, cmap=cmap, linewidths=0.5, linecolor='gray',
                     cbar=False, annot=True, fmt='d')

    # 自定义图例标签 A, B, C, D
    custom_labels = ['A', 'B', 'C', 'D']
    legend_elements = [Patch(facecolor=custom_colors[i], edgecolor='w',
                             label=f'{custom_labels[i]} ({class_names[i]})' if i < 4 else f'{class_names[i]}')
                       for i in range(len(class_names))]

    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', title="Classes")

    plt.title('SOM Predicted Labels (Majority Voting)', fontsize=14)
    plt.xlabel('SOM X')
    plt.ylabel('SOM Y')
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 标签分布图已保存: {filename}")


def plot_custom_tsne(data, labels, class_names, title, filename):
    """
    绘制 t-SNE，要求：
    1. 粉绿蓝紫 -> A, B, C, D
    2. 图例在右上角
    """
    print(f"正在计算 t-SNE ({title}) ...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_embedded = tsne.fit_transform(data)

    # 自定义颜色: Pink, Green, Blue, Purple
    color_map = ['#e377c2', '#2ca02c', '#1f77b4', '#9467bd']
    label_map = ['A', 'B', 'C', 'D']

    plt.figure(figsize=(10, 8))

    # 循环绘制每一类，确保图例正确
    for i in range(len(class_names)):
        idx = (labels == i)
        if np.sum(idx) > 0:
            c = color_map[i] if i < 4 else None
            l = label_map[i] if i < 4 else str(class_names[i])
            plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1],
                        c=c, label=l, s=80, alpha=0.8, edgecolors='w')

    plt.title(title, fontsize=15)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    # 图例右上角
    plt.legend(title='Types', loc='upper right', frameon=True, fancybox=True, framealpha=0.9)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ t-SNE 分布图已保存: {save_path}")


def main():
    data = load_data()
    if data[0] is None: return
    X_train, y_train, X_test, y_test, class_names = data

    # 1. 训练 SOM
    print("正在训练 SOM ...")
    som_size = 6  # 保持与其他代码一致
    som = MiniSom(x=som_size, y=som_size, input_len=X_train.shape[1], sigma=1.0, learning_rate=0.5, random_seed=42)
    som.train_random(X_train, 10000)

    # 2. 建立映射 (分类逻辑)
    winmap = som.labels_map(X_train, y_train)
    node_map = {}
    for position, label_list in winmap.items():
        from collections import Counter
        counts = Counter(label_list)
        node_map[position] = counts.most_common(1)[0][0]

    def predict(data):
        preds = []
        for x in data:
            w = som.winner(x)
            if w in node_map:
                preds.append(node_map[w])
            else:
                dists = []
                for kw in node_map:
                    d = np.linalg.norm(np.array(w) - np.array(kw))
                    dists.append((d, node_map[kw]))
                if dists:
                    preds.append(min(dists, key=lambda x: x[0])[1])
                else:
                    preds.append(0)
        return np.array(preds)

    # 3. 预测与评估
    print("\n=== SOM (Raw Data) 实验结果 ===")
    y_pred_train = predict(X_train)
    y_pred_test = predict(X_test)

    calculate_metrics(y_train, y_pred_train, "训练集")
    calculate_metrics(y_test, y_pred_test, "测试集")

    # 4. 绘图 (统一风格)
    plot_som_activation(som, X_train, 'SOM_Activation.png')
    plot_som_labels(node_map, (som_size, som_size), class_names, 'SOM_Labels.png')
    plot_custom_tsne(X_train, y_train, class_names, 'tSNE_Distribution_Raw.png', 'tSNE_Distribution_Raw.png')


if __name__ == "__main__":
    main()