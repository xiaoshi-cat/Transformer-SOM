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
torch.manual_seed(42)
np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid')

OUTPUT_DIR = "Ablation_MLP_SOM_Results"
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)
print(f"📁 结果将保存至: {OUTPUT_DIR}")


# ==========================================
# 1. 数据加载与预处理
# ==========================================
def load_data():
    if not os.path.exists('new_train_data.csv') or not os.path.exists('new_test_data.csv'):
        print("❌ 错误: 未找到数据文件。")
        return None, None, None, None, None

    train_df = pd.read_csv('new_train_data.csv')
    test_df = pd.read_csv('new_test_data.csv')

    feature_cols = ['Ca', 'Mg', 'Na', 'HCO3', 'Cl', 'SO4', 'TH', 'TA', 'PH']
    target_col = 'Label'

    le = LabelEncoder()
    full_labels = pd.concat([train_df[target_col], test_df[target_col]], axis=0)
    le.fit(full_labels)
    y_train = le.transform(train_df[target_col])
    y_test = le.transform(test_df[target_col])
    class_names = le.classes_

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
# 3. 辅助函数
# ==========================================
def calculate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return {'acc': acc, 'p': p, 'r': r, 'f1': f1}


def plot_som_activation(som, data, filename):
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
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()
    print(f"✅ 性能对比图已保存: {filename}")


def plot_tsne_custom(data, labels, title, filename):
    """
    [修改] 绘制 t-SNE 散点图
    要求: 标注在右上角，颜色粉绿蓝紫对应A,B,C,D
    """
    print(f"    正在计算 t-SNE ({title})...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    data_embedded = tsne.fit_transform(data)

    # 自定义颜色: A(Pink), B(Green), C(Blue), D(Purple)
    custom_colors = ['#FF69B4', '#008000', '#0000FF', '#800080']
    custom_labels = ['A', 'B', 'C', 'D']

    n_classes = len(np.unique(labels))

    # 构建 DataFrame
    df_plot = pd.DataFrame(data_embedded, columns=['d1', 'd2'])
    df_plot['label_idx'] = labels
    label_map = {i: custom_labels[i] for i in range(min(n_classes, 4))}
    df_plot['Type'] = df_plot['label_idx'].map(label_map)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_plot,
        x='d1', y='d2',
        hue='Type',
        hue_order=custom_labels[:n_classes],
        palette=dict(zip(custom_labels[:n_classes], custom_colors[:n_classes])),
        s=80, alpha=0.8, edgecolor='w'
    )

    plt.title(title, fontsize=15)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    # 强制右上角
    plt.legend(title='Types', loc='upper right', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ t-SNE 图已保存: {filename}")


# ==========================================
# 4. 主程序
# ==========================================
def main():
    data = load_data()
    if data[0] is None: return
    X_train, y_train, X_test, y_test, class_names = data

    train_tensor = torch.FloatTensor(X_train)
    test_tensor = torch.FloatTensor(X_test)

    input_dim = X_train.shape[1]
    latent_dim = 32

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

    model.eval()
    with torch.no_grad():
        _, train_features = model(train_tensor)
        _, test_features = model(test_tensor)
        train_features = train_features.numpy()
        test_features = test_features.numpy()

    print(">>> [2/5] 正在训练 SOM (6x6)...")
    som_size = 6
    som = MiniSom(x=som_size, y=som_size, input_len=latent_dim, sigma=1.0, learning_rate=0.5, random_seed=42)
    som.train_random(train_features, 5000)

    print(">>> [3/5] 计算映射与指标...")
    winmap = som.labels_map(train_features, y_train)
    node_map = {}
    for position, label_data in winmap.items():
        if hasattr(label_data, 'most_common'):
            node_map[position] = label_data.most_common(1)[0][0]
        else:
            node_map[position] = max(set(label_data), key=label_data.count)

    def predict_som_safe(data):
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
                preds.append(min(dists, key=lambda x: x[0])[1] if dists else 0)
        return np.array(preds)

    y_pred_tr = predict_som_safe(train_features)
    y_pred_te = predict_som_safe(test_features)

    metrics_tr = calculate_metrics(y_train, y_pred_tr)
    metrics_te = calculate_metrics(y_test, y_pred_te)

    print(f"训练集 | Acc: {metrics_tr['acc']:.4f}")
    print(f"测试集 | Acc: {metrics_te['acc']:.4f}")

    print(">>> [4/5] 生成 SOM 相关图表...")
    plot_som_activation(som, train_features, 'chart_1_som_activation.png')
    plot_som_labels(node_map, (som_size, som_size), class_names, 'chart_2_som_labels.png')
    plot_metrics_comparison(metrics_tr, metrics_te, 'chart_3_metrics_comparison.png')

    print(">>> [5/5] 生成 t-SNE 特征分布图 (修改版)...")
    plot_tsne_custom(X_train, y_train, 't-SNE of Raw Data (Input Features)', 'chart_4_tsne_raw.png')
    plot_tsne_custom(train_features, y_train, 't-SNE of MLP Latent Features', 'chart_5_tsne_mlp_latent.png')

    print(f"\n🎉 所有结果已保存至文件夹: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()