import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from minisom import MiniSom
import os
import shutil
import copy

# ==========================================
# 配置
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
# 解决中文乱码 (如果需要)
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = "Experiment_Results_SmallData_Zscore"
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)


# ==========================================
# 1. 微型 Transformer (针对小样本优化)
# ==========================================
class InterpretableEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.last_attn_weights = None

    def forward(self, src):
        src2, weights = self.self_attn(src, src, src, need_weights=True, average_attn_weights=True)
        self.last_attn_weights = weights
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class SupervisedTransformerAE(nn.Module):
    def __init__(self, num_features, num_classes, d_model=16, nhead=2, num_layers=1):
        super(SupervisedTransformerAE, self).__init__()
        self.feature_embedding = nn.Linear(1, d_model)
        self.column_embedding = nn.Parameter(torch.randn(1, num_features, d_model))

        self.layers = nn.ModuleList([
            InterpretableEncoderLayer(d_model, nhead, dim_feedforward=32, dropout=0.2)
            for _ in range(num_layers)
        ])

        self.decoder = nn.Sequential(nn.Linear(d_model, 16), nn.ReLU(), nn.Linear(16, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features * d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x_emb = self.feature_embedding(x.unsqueeze(-1)) + self.column_embedding
        latent = x_emb
        for layer in self.layers:
            latent = layer(latent)
        reconstruction = self.decoder(latent).squeeze(-1)
        class_logits = self.classifier(latent)
        return reconstruction, latent, class_logits


# ==========================================
# 2. 绘图函数
# ==========================================
def plot_attention_heatmap(model, data_tensor, feature_names):
    print(">>> 正在绘制 [自注意力热力图 (红色系)]...")
    model.eval()
    with torch.no_grad():
        model(data_tensor)

    attn_weights = model.layers[0].last_attn_weights.mean(dim=0).cpu().numpy()

    plt.figure(figsize=(10, 9))
    sns.heatmap(attn_weights, xticklabels=feature_names, yticklabels=feature_names,
                cmap='Reds', annot=False, square=True,
                cbar_kws={'label': 'Attention Weight (Importance)', 'shrink': 0.8})

    plt.title('Self-Attention Heatmap', fontsize=14, fontweight='bold')
    plt.xlabel('Source Feature', fontsize=12)
    plt.ylabel('Target Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Advanced_1_Attention_Heatmap_Red.png'), dpi=300)


def plot_u_matrix(som, data, y, class_names):
    print(">>> 正在绘制 [U-Matrix]...")
    plt.figure(figsize=(10, 10))
    u_matrix = som.distance_map()

    sns.heatmap(u_matrix, cmap='coolwarm', annot=False,
                cbar_kws={'label': 'Euclidean Distance', 'shrink': 0.8},
                square=True)

    markers = ['o', 's', 'D', '^']
    colors = sns.color_palette("husl", len(class_names))

    w_x, w_y = [], []
    for x in data:
        w = som.winner(x)
        w_x.append(w[0]);
        w_y.append(w[1])
    w_x = np.array(w_x);
    w_y = np.array(w_y)

    for i, name in enumerate(class_names):
        idx = np.where(y == i)[0]
        jitter_x = np.random.rand(len(idx)) * 0.6 - 0.3
        jitter_y = np.random.rand(len(idx)) * 0.6 - 0.3
        plt.scatter(w_y[idx] + 0.5 + jitter_y, w_x[idx] + 0.5 + jitter_x,
                    label=name, s=50, color=colors[i], marker=markers[i % len(markers)],
                    edgecolors='white', linewidth=0.8, alpha=0.9)

    plt.title('U-Matrix with Sample Distribution', fontsize=15, fontweight='bold', pad=20)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
               fancybox=True, shadow=True, ncol=4, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Advanced_4_U_Matrix.png'), dpi=300)


def plot_som_component_planes(som, latent_features, raw_features_data, feature_names):
    """
    修改点：支持直接传入标准化后的 numpy array
    """
    print(">>> 正在绘制 [SOM 组件平面图 (Z-score)]...")
    weights_shape = som.get_weights().shape[:2]
    component_planes = np.zeros((weights_shape[0], weights_shape[1], len(feature_names)))
    counts = np.zeros(weights_shape)

    # 兼容处理：如果是DataFrame取values，如果是numpy直接用
    if isinstance(raw_features_data, pd.DataFrame):
        raw_values = raw_features_data.values
    else:
        raw_values = raw_features_data  # 这里传入的就是 Z-score 后的数据

    for i, x in enumerate(latent_features):
        w = som.winner(x)
        component_planes[w] += raw_values[i]
        counts[w] += 1
    global_means = np.mean(raw_values, axis=0)
    for r in range(weights_shape[0]):
        for c in range(weights_shape[1]):
            if counts[r, c] > 0:
                component_planes[r, c] /= counts[r, c]
            else:
                component_planes[r, c] = global_means

    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    axes = axes.flatten()
    for i, name in enumerate(feature_names):
        if i >= len(axes): break
        # Label 改为 Z-score
        sns.heatmap(component_planes[:, :, i], ax=axes[i], cmap='coolwarm',
                    annot=False, cbar=True, square=True,
                    cbar_kws={'label': 'Z-score', 'shrink': 0.8})
        axes[i].set_title(f'{name}', fontsize=12, fontweight='bold')
        axes[i].axis('off')
    plt.suptitle('SOM Component Planes (Z-score Standardized)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, 'Advanced_2_SOM_Components_Zscore.png'), dpi=300)


def plot_latent_space_comparison(X_raw, X_latent, y, class_names):
    print(">>> 正在绘制 [隐空间对比图]...")
    pca = PCA(n_components=2)
    X_raw_pca = pca.fit_transform(X_raw)
    X_latent_pca = pca.fit_transform(X_latent)

    tsne = TSNE(n_components=2, perplexity=min(10, len(X_raw) - 1), random_state=42)
    X_latent_tsne = tsne.fit_transform(X_latent)

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    colors = sns.color_palette("husl", len(class_names))

    for ax, data, title in zip(axes, [X_raw_pca, X_latent_pca, X_latent_tsne],
                               ['Raw Data (PCA)', 'Latent (PCA)', 'Latent (t-SNE)']):
        for i, name in enumerate(class_names):
            mask = (y == i)
            ax.scatter(data[mask, 0], data[mask, 1], label=name, color=colors[i], s=60, alpha=0.8, edgecolors='white')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Advanced_3_Latent_Space.png'), dpi=300)


# ==========================================
# 3. 主程序
# ==========================================
def main():
    print(">>> [1/5] 读取数据...")
    if not os.path.exists('new_train_data.csv'):
        print("未找到文件");
        return

    train_df = pd.read_csv('new_train_data.csv')
    test_df = pd.read_csv('new_test_data.csv')
    feature_cols = ['Ca', 'Mg', 'Na', 'HCO3', 'Cl', 'SO4', 'TH', 'TA', 'PH']
    target_col = 'Label'

    full_labels = pd.concat([train_df[target_col], test_df[target_col]], axis=0)
    le = LabelEncoder();
    le.fit(full_labels)
    class_names = le.classes_
    y_train = le.transform(train_df[target_col])

    # === 关键步骤：Z-score 标准化 ===
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)

    print(f"    - 样本数: {len(X_train)} (Small Data Mode Activated)")

    print(">>> [2/5] 训练微型 Supervised Transformer...")
    train_tensor = torch.FloatTensor(X_train)
    train_labels = torch.LongTensor(y_train)

    model = SupervisedTransformerAE(len(feature_cols), len(class_names),
                                    d_model=16, nhead=2, num_layers=1)

    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)
    criterion_recon = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()

    epochs = 150
    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        recon, _, logits = model(train_tensor)
        loss_r = criterion_recon(recon, train_tensor)
        loss_c = criterion_class(logits, train_labels)
        total_loss = loss_r + 0.8 * loss_c
        total_loss.backward()
        optimizer.step()
        if (ep + 1) % 50 == 0: print(f"    Epoch {ep + 1} | Loss: {total_loss.item():.4f}")

    print(">>> [3/5] 提取 Latent Features...")
    model.eval()
    with torch.no_grad():
        _, tr_lat, _ = model(train_tensor)
        feat_train = tr_lat.reshape(len(X_train), -1).numpy()

    print(">>> [4/5] 训练 SOM...")
    som = MiniSom(6, 6, feat_train.shape[1], sigma=1.0, learning_rate=0.5, random_seed=42)
    som.train_random(feat_train, 5000)

    print(">>> [5/5] 生成图表 (Z-score Version)...")

    # 1. 红色系热力图
    plot_attention_heatmap(model, train_tensor, feature_cols)

    # 2. SOM 组件图 (关键修改：传入 X_train 也就是 Z-score 后的数据)
    plot_som_component_planes(som, feat_train, X_train, feature_cols)

    # 3. Latent Space
    plot_latent_space_comparison(X_train, feat_train, y_train, class_names)

    # 4. U-Matrix
    plot_u_matrix(som, feat_train, y_train, class_names)

    print(f"\n🎉 优化完成！结果在 '{OUTPUT_DIR}'。")


if __name__ == "__main__":
    main()

