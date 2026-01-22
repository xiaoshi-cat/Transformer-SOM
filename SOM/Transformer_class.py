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
from sklearn.metrics import accuracy_score
from minisom import MiniSom
import os
import shutil
import warnings

# ==========================================
# 0. 全局配置
# ==========================================
# 忽略一些不必要的警告
warnings.filterwarnings("ignore")
# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
# 解决中文显示问题 (如果环境支持)
plt.rcParams['axes.unicode_minus'] = False

# 输出目录
OUTPUT_DIR = "Experiment_Results_Final"
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)
print(f"文件夹 '{OUTPUT_DIR}' 已创建，所有结果将保存在此。")


# ==========================================
# 1. 模型定义: 微型 Supervised Transformer
# ==========================================
class InterpretableEncoderLayer(nn.Module):
    """
    可解释的编码层：在 Forward 过程中保存 Attention 权重
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        # average_attn_weights=True: 返回 [Batch, Seq, Seq]
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.last_attn_weights = None  # 用于存储注意力权重

    def forward(self, src):
        # src: [Batch, Seq, Feature_Dim]
        src2, weights = self.self_attn(src, src, src, need_weights=True, average_attn_weights=True)
        self.last_attn_weights = weights

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class SupervisedTransformerAE(nn.Module):
    """
    针对小样本优化的 Supervised Autoencoder
    """

    def __init__(self, num_features, num_classes, d_model=16, nhead=2, num_layers=1):
        super(SupervisedTransformerAE, self).__init__()
        # 特征嵌入层
        self.feature_embedding = nn.Linear(1, d_model)
        self.column_embedding = nn.Parameter(torch.randn(1, num_features, d_model))

        # 编码器 (层数减少到1，防止过拟合)
        self.layers = nn.ModuleList([
            InterpretableEncoderLayer(d_model, nhead, dim_feedforward=32, dropout=0.2)
            for _ in range(num_layers)
        ])

        # 解码器 (重建任务)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # 分类头 (监督任务，消融实验 Baseline)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features * d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.3),  # 高 Dropout 增加鲁棒性
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # [Batch, Features] -> [Batch, Features, 1]
        x_emb = self.feature_embedding(x.unsqueeze(-1)) + self.column_embedding

        latent = x_emb
        for layer in self.layers:
            latent = layer(latent)

        # 1. 重建
        reconstruction = self.decoder(latent).squeeze(-1)
        # 2. 分类 (Logits)
        class_logits = self.classifier(latent)

        return reconstruction, latent, class_logits


# ==========================================
# 2. 核心绘图函数
# ==========================================

def plot_attention_heatmap(model, data_tensor, feature_names):
    """绘制自注意力热力图 (红色系)"""
    print(">>> 正在绘制 [自注意力热力图]...")
    model.eval()
    with torch.no_grad():
        model(data_tensor)

    # 获取第一层的平均注意力权重
    attn_weights = model.layers[0].last_attn_weights.mean(dim=0).cpu().numpy()

    plt.figure(figsize=(10, 9))
    sns.heatmap(attn_weights, xticklabels=feature_names, yticklabels=feature_names,
                cmap='Reds', annot=False, square=True,
                cbar_kws={'label': 'Attention Weight (Importance)', 'shrink': 0.8})

    plt.title('Self-Attention Heatmap (Global Interpretability)', fontsize=14, fontweight='bold')
    plt.xlabel('Source Feature (Key)', fontsize=12)
    plt.ylabel('Target Feature (Query)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Advanced_1_Attention_Heatmap.png'), dpi=300)


def plot_som_component_planes(som, latent_features, raw_features_data, feature_names):
    """
    绘制 SOM 组件平面图
    关键点：传入的是 Z-score 标准化后的数据 (raw_features_data)
    """
    print(">>> 正在绘制 [SOM 组件平面图 (Z-score)]...")
    weights_shape = som.get_weights().shape[:2]  # (6, 6)

    # 准备容器
    component_planes = np.zeros((weights_shape[0], weights_shape[1], len(feature_names)))
    counts = np.zeros(weights_shape)

    # 兼容处理: 确保是 numpy array
    if isinstance(raw_features_data, pd.DataFrame):
        raw_values = raw_features_data.values
    else:
        raw_values = raw_features_data

        # 累加每个节点对应的样本特征值
    for i, x in enumerate(latent_features):
        w = som.winner(x)
        component_planes[w] += raw_values[i]
        counts[w] += 1

    # 求平均
    global_means = np.mean(raw_values, axis=0)
    for r in range(weights_shape[0]):
        for c in range(weights_shape[1]):
            if counts[r, c] > 0:
                component_planes[r, c] /= counts[r, c]
            else:
                component_planes[r, c] = global_means  # 空节点填均值

    # 绘图 3x3
    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    axes = axes.flatten()

    for i, name in enumerate(feature_names):
        if i >= len(axes): break
        # 使用 coolwarm, 颜色条标签设为 Z-score
        sns.heatmap(component_planes[:, :, i], ax=axes[i], cmap='coolwarm',
                    annot=False, cbar=True, square=True,
                    cbar_kws={'label': 'Z-score', 'shrink': 0.8})
        axes[i].set_title(f'{name} Distribution', fontsize=12, fontweight='bold')
        axes[i].axis('off')

    plt.suptitle('SOM Component Planes (Z-score Standardized)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, 'Advanced_2_SOM_Components_Zscore.png'), dpi=300)


def plot_latent_space_comparison(X_raw, X_latent, y, class_names):
    """绘制隐空间对比图 (Raw PCA vs Latent PCA vs Latent t-SNE)"""
    print(">>> 正在绘制 [隐空间分布对比图]...")

    # 1. Raw Data PCA
    pca = PCA(n_components=2)
    X_raw_pca = pca.fit_transform(X_raw)

    # 2. Latent Data PCA
    X_latent_pca = pca.fit_transform(X_latent)

    # 3. Latent Data t-SNE (适配小样本的 perplexity)
    tsne = TSNE(n_components=2, perplexity=min(10, len(X_raw) - 1), random_state=42)
    X_latent_tsne = tsne.fit_transform(X_latent)

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    colors = sns.color_palette("husl", len(class_names))

    plot_data = [
        (X_raw_pca, 'Raw Data (PCA)'),
        (X_latent_pca, 'Latent Space (PCA)'),
        (X_latent_tsne, 'Latent Space (t-SNE)')
    ]

    for ax_idx, (data, title) in enumerate(plot_data):
        ax = axes[ax_idx]
        for i, name in enumerate(class_names):
            mask = (y == i)
            ax.scatter(data[mask, 0], data[mask, 1], label=name,
                       color=colors[i], s=60, alpha=0.8, edgecolors='white')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Advanced_3_Latent_Space_Comparison.png'), dpi=300)


def plot_u_matrix(som, data, y, class_names):
    """绘制 U-Matrix (图例在底部)"""
    print(">>> 正在绘制 [U-Matrix 聚类图]...")
    plt.figure(figsize=(10, 10))

    u_matrix = som.distance_map()

    # 背景距离热力图
    sns.heatmap(u_matrix, cmap='coolwarm', annot=False,
                cbar_kws={'label': 'Euclidean Distance (Blue=Center, Red=Boundary)', 'shrink': 0.8},
                square=True)

    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*']
    colors = sns.color_palette("husl", len(class_names))

    w_x, w_y = [], []
    for x in data:
        w = som.winner(x)
        w_x.append(w[0]);
        w_y.append(w[1])
    w_x = np.array(w_x);
    w_y = np.array(w_y)

    # 绘制散点 (带抖动防止重叠)
    for i, name in enumerate(class_names):
        idx = np.where(y == i)[0]
        jitter_x = np.random.rand(len(idx)) * 0.6 - 0.3
        jitter_y = np.random.rand(len(idx)) * 0.6 - 0.3

        plt.scatter(w_y[idx] + 0.5 + jitter_y, w_x[idx] + 0.5 + jitter_x,
                    label=name, s=60, color=colors[i], marker=markers[i % len(markers)],
                    edgecolors='white', linewidth=1.0, alpha=0.9)

    plt.title('U-Matrix with Sample Distribution', fontsize=15, fontweight='bold', pad=20)
    # 图例放底部
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
               fancybox=True, shadow=True, ncol=4, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Advanced_4_U_Matrix.png'), dpi=300)


def plot_ablation_study(trans_acc, som_acc):
    """绘制消融实验对比图"""
    print(">>> 正在绘制 [消融实验对比图]...")
    plt.figure(figsize=(7, 6))
    methods = ['Transformer Head\n(Baseline)', 'Transformer + SOM\n(Ours)']
    accs = [trans_acc, som_acc]
    colors = ['gray', '#e74c3c']  # 灰色对比红色

    bars = plt.bar(methods, accs, color=colors, width=0.6, alpha=0.9)

    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Ablation Study: Classifier vs. SOM', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 标数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{height:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Advanced_5_Ablation_Study.png'), dpi=300)


# ==========================================
# 3. 主程序流程
# ==========================================
def main():
    print(">>> [1/7] 读取数据...")
    if not os.path.exists('new_train_data.csv'):
        print("❌ 错误: 当前目录下未找到 'new_train_data.csv'")
        return

    train_df = pd.read_csv('new_train_data.csv')
    test_df = pd.read_csv('new_test_data.csv')  # 如果有的话

    feature_cols = ['Ca', 'Mg', 'Na', 'HCO3', 'Cl', 'SO4', 'TH', 'TA', 'PH']
    target_col = 'Label'

    # 1. 标签编码
    full_labels = pd.concat([train_df[target_col], test_df[target_col]], axis=0)
    le = LabelEncoder()
    le.fit(full_labels)
    class_names = le.classes_

    y_train = le.transform(train_df[target_col])

    # 2. 特征标准化 (Z-score)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)

    print(f"    - 训练集样本数: {len(X_train)} (Small Data Mode)")
    print(f"    - 特征数: {len(feature_cols)}")
    print(f"    - 类别数: {len(class_names)}")

    # 准备 Tensor
    train_tensor = torch.FloatTensor(X_train)
    train_labels = torch.LongTensor(y_train)

    print(">>> [2/7] 训练 Supervised Transformer (Micro版)...")
    # 初始化模型: 16维, 1层, 2头
    model = SupervisedTransformerAE(len(feature_cols), len(class_names),
                                    d_model=16, nhead=2, num_layers=1)

    # 优化器: 加入 Weight Decay 防止过拟合
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)

    # 损失函数: 重建 + 分类
    criterion_recon = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()

    epochs = 150
    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()

        recon, _, logits = model(train_tensor)

        loss_r = criterion_recon(recon, train_tensor)
        loss_c = criterion_class(logits, train_labels)

        # 联合损失: 加大分类权重 (0.8) 强迫分离
        total_loss = loss_r + 0.8 * loss_c

        total_loss.backward()
        optimizer.step()

        if (ep + 1) % 50 == 0:
            print(f"    Epoch {ep + 1}/{epochs} | Loss: {total_loss.item():.4f}")

    print(">>> [3/7] 提取 Latent Features...")
    model.eval()
    with torch.no_grad():
        _, tr_lat, tr_logits = model(train_tensor)
        feat_train = tr_lat.reshape(len(X_train), -1).numpy()

        # --- 消融实验数据准备: 计算 Transformer 自带分类头的准确率 ---
        trans_preds = torch.argmax(tr_logits, dim=1).numpy()
        trans_acc = accuracy_score(y_train, trans_preds)

    print(">>> [4/7] 训练 SOM (拓扑聚类)...")
    som = MiniSom(6, 6, feat_train.shape[1], sigma=1.0, learning_rate=0.5, random_seed=42)
    som.train_random(feat_train, 5000)

    # --- 消融实验数据准备: 计算 SOM 的准确率 ---
    # 建立映射: 节点 -> 类别标签 (多数投票)
    node_map = {}
    for i, x in enumerate(feat_train):
        w = som.winner(x)
        if w not in node_map: node_map[w] = []
        node_map[w].append(y_train[i])
    for w in node_map: node_map[w] = max(set(node_map[w]), key=node_map[w].count)

    # 预测
    som_preds = []
    for x in feat_train:
        w = som.winner(x)
        som_preds.append(node_map.get(w, 0))  # 默认0类以防万一
    som_acc = accuracy_score(y_train, som_preds)

    print(f"    [对比结果] Transformer Head Acc: {trans_acc:.2%} | SOM Acc: {som_acc:.2%}")

    print(">>> [5/7] 生成核心可视化图表...")

    # 图 1: 自注意力热力图 (Reds)
    plot_attention_heatmap(model, train_tensor, feature_cols)

    # 图 2: SOM 组件平面图 (使用 Z-score 数据 X_train)
    plot_som_component_planes(som, feat_train, X_train, feature_cols)

    # 图 3: 隐空间对比 (t-SNE)
    plot_latent_space_comparison(X_train, feat_train, y_train, class_names)

    # 图 4: U-Matrix (布局优化)
    plot_u_matrix(som, feat_train, y_train, class_names)

    print(">>> [6/7] 生成消融实验对比图...")
    # 图 5: 消融对比
    plot_ablation_study(trans_acc, som_acc)

    print(f"\n🎉🎉🎉 全部完成！\n请打开文件夹 '{OUTPUT_DIR}' 查看你的 5 张论文配图。")


if __name__ == "__main__":
    main()