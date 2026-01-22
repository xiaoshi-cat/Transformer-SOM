import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.manifold import TSNE
from minisom import MiniSom
import os
import shutil

# ==========================================
# 0. 配置与初始化
# ==========================================
# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
# 如果系统有中文字体可取消注释
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# 创建结果保存目录
OUTPUT_DIR = "Ablation_MLP_SOM_Results"
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)  # 清空旧结果
os.makedirs(OUTPUT_DIR)
print(f"📁 结果将保存至: {OUTPUT_DIR}")


# ==========================================
# 1. 数据加载与预处理
# ==========================================
def load_data():
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
    class_names = le.classes_

    # Z-score 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)

    return X_train, y_train, X_test, y_test, class_names


# ==========================================
# 2. MLP Autoencoder 模型
# ==========================================
class MLPAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super(MLPAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent


# ==========================================
# 3. 辅助函数: 绘图与评估
# ==========================================
def calculate_metrics(y_true, y_pred):
    """计算 Acc, Precision, Recall, F1"""
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return {'acc': acc, 'p': p, 'r': r, 'f1': f1}


def plot_som_activation(som, data, filename):
    """绘制 SOM 激活频率图 (Hit Map)"""
    plt.figure(figsize=(8, 7))
    frequencies = som.activation_response(data)
    sns.heatmap(frequencies.T, cmap='Blues', linewidths=0.5, annot=True, fmt='.0f', cbar_kws={'label': 'Hit Count'})
    plt.title('SOM Activation Map (Sample Hits)', fontsize=14)
    plt.xlabel('SOM X')
    plt.ylabel('SOM Y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()
    print(f"✅ 激活图已保存: {filename}")


def plot_som_labels(node_map, som_shape, class_names, filename):
    """绘制 SOM 标签分布图 (多数投票结果)"""
    label_grid = np.full((som_shape[1], som_shape[0]), -1)
    for (x, y), label in node_map.items():
        label_grid[y, x] = label

    plt.figure(figsize=(9, 7))
    cmap = sns.color_palette("husl", len(class_names))
    mask = (label_grid == -1)

    ax = sns.heatmap(label_grid, mask=mask, cmap=cmap, linewidths=0.5, linecolor='gray',
                     cbar=False, annot=True, fmt='d')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=cmap[i], edgecolor='w', label=f'{i}: {name}')
                       for i, name in enumerate(class_names)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', title="Classes")

    plt.title('SOM Predicted Labels (Majority Voting)', fontsize=14)
    plt.xlabel('SOM X')
    plt.ylabel('SOM Y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()
    print(f"✅ 标签分布图已保存: {filename}")


def plot_metrics_comparison(tr_metrics, te_metrics, filename):
    """绘制训练集 vs 测试集 指标对比图"""
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    train_vals = [tr_metrics['acc'], tr_metrics['p'], tr_metrics['r'], tr_metrics['f1']]
    test_vals = [te_metrics['acc'], te_metrics['p'], te_metrics['r'], te_metrics['f1']]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    rects1 = plt.bar(x - width / 2, train_vals, width, label='Train', color='#3498db', alpha=0.9)
    rects2 = plt.bar(x + width / 2, test_vals, width, label='Test', color='#e74c3c', alpha=0.9)

    plt.ylabel('Score')
    plt.title('Model Performance Comparison (MLP+SOM)', fontsize=14)
    plt.xticks(x, labels)
    plt.ylim(0, 1.15)
    plt.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()
    print(f"✅ 性能对比图已保存: {filename}")


def plot_tsne(data, labels, class_names, title, filename):
    """
    [新增功能] 绘制 t-SNE 散点图
    :param data: 输入特征数据 (numpy array)
    :param labels: 标签 (numpy array)
    :param class_names: 类别名称列表
    :param title: 图表标题
    :param filename: 保存文件名
    """
    print(f"    正在计算 t-SNE ({title})...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    data_embedded = tsne.fit_transform(data)

    plt.figure(figsize=(10, 8))
    # 使用 seaborn 绘制散点图
    sns.scatterplot(
        x=data_embedded[:, 0],
        y=data_embedded[:, 1],
        hue=class_names[labels],  # 将数字标签转为文字标签
        palette="husl",
        s=80,
        alpha=0.8,
        edgecolor='w'
    )

    plt.title(title, fontsize=15)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Water Source', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ t-SNE 图已保存: {filename}")


# ==========================================
# 4. 主程序
# ==========================================
def main():
    # 1. 加载数据
    data = load_data()
    if data[0] is None: return
    X_train, y_train, X_test, y_test, class_names = data

    # 转换为 Tensor
    train_tensor = torch.FloatTensor(X_train)
    test_tensor = torch.FloatTensor(X_test)

    input_dim = X_train.shape[1]
    latent_dim = 32

    # 2. 训练 MLP
    print(">>> [1/5] 正在训练 MLP Autoencoder...")
    model = MLPAutoencoder(input_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.MSELoss()

    epochs = 100
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon, _ = model(train_tensor)
        loss = criterion(recon, train_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f}")

    # 3. 提取特征
    model.eval()
    with torch.no_grad():
        _, train_features = model(train_tensor)
        _, test_features = model(test_tensor)
        train_features = train_features.numpy()
        test_features = test_features.numpy()

    # 4. 训练 SOM
    print(">>> [2/5] 正在训练 SOM (6x6)...")
    som_size = 6
    som = MiniSom(x=som_size, y=som_size, input_len=latent_dim, sigma=1.0, learning_rate=0.5, random_seed=42)
    som.train_random(train_features, 5000)

    # 5. 建立映射与预测
    print(">>> [3/5] 计算映射与指标...")
    # 建立节点映射 (兼容性写法)
    winmap = som.labels_map(train_features, y_train)
    node_map = {}
    for position, label_data in winmap.items():
        if hasattr(label_data, 'most_common'):
            node_map[position] = label_data.most_common(1)[0][0]
        else:
            node_map[position] = max(set(label_data), key=label_data.count)

    # 预测函数
    def predict_som_safe(data):
        preds = []
        for x in data:
            w = som.winner(x)
            if w in node_map:
                preds.append(node_map[w])
            else:
                # 寻找最近邻
                dists = []
                for kw in node_map:
                    d = np.linalg.norm(np.array(w) - np.array(kw))
                    dists.append((d, node_map[kw]))
                preds.append(min(dists, key=lambda x: x[0])[1] if dists else 0)
        return np.array(preds)

    y_pred_tr = predict_som_safe(train_features)
    y_pred_te = predict_som_safe(test_features)

    # 计算指标
    metrics_tr = calculate_metrics(y_train, y_pred_tr)
    metrics_te = calculate_metrics(y_test, y_pred_te)

    # 打印结果
    print("\n" + "=" * 40)
    print("   MLP+SOM 实验结果")
    print("=" * 40)
    print(f"训练集 | Acc: {metrics_tr['acc']:.4f}, F1: {metrics_tr['f1']:.4f}")
    print(f"测试集 | Acc: {metrics_te['acc']:.4f}, F1: {metrics_te['f1']:.4f}")
    print("=" * 40)

    # 6. 绘图
    print(">>> [4/5] 生成 SOM 相关图表...")
    plot_som_activation(som, train_features, 'chart_1_som_activation.png')
    plot_som_labels(node_map, (som_size, som_size), class_names, 'chart_2_som_labels.png')
    plot_metrics_comparison(metrics_tr, metrics_te, 'chart_3_metrics_comparison.png')

    # 7. 绘制 t-SNE 对比
    print(">>> [5/5] 生成 t-SNE 特征分布图 (原始 vs 升维)...")
    # 7.1 原始特征 t-SNE
    plot_tsne(X_train, y_train, class_names,
              title='t-SNE of Raw Data (Input Features)',
              filename='chart_4_tsne_raw.png')

    # 7.2 升维特征 t-SNE
    plot_tsne(train_features, y_train, class_names,
              title='t-SNE of MLP Latent Features',
              filename='chart_5_tsne_mlp_latent.png')

    print(f"\n🎉 所有结果已保存至文件夹: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()