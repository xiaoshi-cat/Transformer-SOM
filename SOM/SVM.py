import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from minisom import MiniSom  # 确保已安装: pip install minisom


# ==========================================
# 1. 深度学习模型: Transformer Encoder
# ==========================================
class TabularTransformerAutoencoder(nn.Module):
    def __init__(self, num_features, d_model=32, nhead=4, num_layers=2, dim_feedforward=64):
        super(TabularTransformerAutoencoder, self).__init__()
        # 特征嵌入: 数值 -> 向量
        self.feature_embedding = nn.Linear(1, d_model)
        # 列位置编码: 区分不同离子
        self.column_embedding = nn.Parameter(torch.randn(1, num_features, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder (仅用于训练重建)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # x shape: [batch, features]
        x_emb = self.feature_embedding(x.unsqueeze(-1)) + self.column_embedding
        latent_seq = self.transformer_encoder(x_emb)  # [batch, features, d_model]
        reconstruction = self.decoder(latent_seq).squeeze(-1)
        return reconstruction, latent_seq


# ==========================================
# 2. 辅助函数: 绘图与评估
# ==========================================
def plot_confusion_matrix(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def plot_class_metrics(y_true, y_pred, classes, title):
    # 计算每个类别的指标
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1 Score')

    plt.xlabel('Water Source Type')
    plt.ylabel('Score')
    plt.title(title)
    plt.xticks(x, classes)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


# ==========================================
# 3. 主程序
# ==========================================
def main():
    # --- A. 数据加载与预处理 ---
    print("正在加载数据...")
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data.csv')

    # 定义列名 (请根据实际情况调整)
    feature_cols = ['Ca', 'Mg', 'Na', 'HCO3', 'Cl', 'SO4', 'TH', 'TA', 'PH']
    target_col = 'Label'

    # 标签编码 (String -> Int)
    le = LabelEncoder()
    y_train = le.fit_transform(train_df[target_col])
    y_test = le.transform(test_df[target_col])
    class_names = le.classes_

    # [关键步骤] 数据标准化 (Z-Score)
    print("正在进行数据标准化...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)

    # 转为 PyTorch Tensor
    train_tensor = torch.FloatTensor(X_train)
    test_tensor = torch.FloatTensor(X_test)

    # --- B. 训练 Transformer ---
    print("\n[Step 1] 训练 Transformer Feature Extractor...")
    model = TabularTransformerAutoencoder(num_features=len(feature_cols))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    epochs = 100
    loss_history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        recon, _ = model(train_tensor)
        loss = criterion(recon, train_tensor)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    # 画 Loss 曲线
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history)
    plt.title('Transformer Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
    plt.show()

    # --- C. 提取深度特征 ---
    print("\n[Step 2] 提取深度特征...")
    model.eval()
    with torch.no_grad():
        _, train_latent = model(train_tensor)
        _, test_latent = model(test_tensor)
        # 展平: [Batch, 9, 32] -> [Batch, 288]
        train_features = train_latent.reshape(len(X_train), -1).numpy()
        test_features = test_latent.reshape(len(X_test), -1).numpy()

    # --- D. 训练 SOM (使用 MiniSom) ---
    print("\n[Step 3] 训练 SOM 聚类...")
    som_dim = 10  # 10x10 网格
    input_len = train_features.shape[1]

    som = MiniSom(som_dim, som_dim, input_len, sigma=1.0, learning_rate=0.5)
    som.train_random(train_features, 1000)  # 训练1000次

    # --- E. 建立 SOM 节点与标签的映射 (Labeling) ---
    # 逻辑: 统计落入每个节点的训练样本的标签，取众数作为该节点的标签
    node_label_map = {}
    train_bmus = [som.winner(x) for x in train_features]  # BMU: Best Matching Unit

    # 统计每个节点的样本标签分布
    bmu_to_labels = {}
    for i, bmu in enumerate(train_bmus):
        if bmu not in bmu_to_labels:
            bmu_to_labels[bmu] = []
        bmu_to_labels[bmu].append(y_train[i])

    # 多数投票
    for bmu, labels in bmu_to_labels.items():
        node_label_map[bmu] = max(set(labels), key=labels.count)

    # --- F. 预测与评估 ---
    def predict(features):
        preds = []
        for x in features:
            bmu = som.winner(x)
            # 如果该节点在训练中未被激活过，默认预测为第0类 (或可改为 -1 表示未知)
            preds.append(node_label_map.get(bmu, 0))
        return np.array(preds)

    print("\n正在评估模型性能...")

    # 预测
    y_pred_train = predict(train_features)
    y_pred_test = predict(test_features)

    # 计算指标
    def print_metrics(y_true, y_pred, split_name):
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        print(f"\n>>> {split_name} Set Metrics:")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {p:.4f}")
        print(f"Recall   : {r:.4f}")
        print(f"F1 Score : {f1:.4f}")

    print_metrics(y_train, y_pred_train, "Training")
    print_metrics(y_test, y_pred_test, "Test")

    # --- G. 绘图 ---
    print("\n正在生成图表...")

    # 1. 混淆矩阵
    plot_confusion_matrix(y_train, y_pred_train, class_names, 'Confusion Matrix (Train)')
    plot_confusion_matrix(y_test, y_pred_test, class_names, 'Confusion Matrix (Test)')

    # 2. 各类型水源详细指标 (柱状图替代曲线)
    plot_class_metrics(y_train, y_pred_train, class_names, 'Per-Class Metrics (Train)')
    plot_class_metrics(y_test, y_pred_test, class_names, 'Per-Class Metrics (Test)')


if __name__ == "__main__":
    main()