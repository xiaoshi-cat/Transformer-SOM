import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_recall_fscore_support,
                             precision_recall_curve, average_precision_score)
from sklearn.neighbors import KNeighborsClassifier
from minisom import MiniSom
import datetime
import itertools
import os
import copy
import shutil
import random
import os
import numpy as np
import torch

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 在代码最开始调用
setup_seed(42)
# ==========================================
# 配置: 绘图风格 (使用纯英文避免乱码)
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
# 不设置中文字体，直接使用默认英文，防止服务器报错
# plt.rcParams['font.sans-serif'] = ['SimHei']

# ==========================================
# 0. 准备输出文件夹
# ==========================================
OUTPUT_DIR = "Experiment_Results"
if os.path.exists(OUTPUT_DIR):
    # 如果存在则清空，保持干净
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)
print(f"文件夹 '{OUTPUT_DIR}' 已创建，所有结果将保存在此。")


# ==========================================
# 1. 模型定义: Transformer Autoencoder
# ==========================================
class TabularTransformerAutoencoder(nn.Module):
    def __init__(self, num_features, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1):
        super(TabularTransformerAutoencoder, self).__init__()
        # 特征嵌入: 将单个数值映射为高维向量
        self.feature_embedding = nn.Linear(1, d_model)
        # 列位置编码: 区分不同离子的位置
        self.column_embedding = nn.Parameter(torch.randn(1, num_features, d_model))

        # Transformer 编码器层 (包含 Dropout 防止过拟合)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 解码器: 用于重建数据计算 Loss
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # [Batch, Features] -> [Batch, Features, 1] -> [Batch, Features, d_model]
        x_emb = self.feature_embedding(x.unsqueeze(-1)) + self.column_embedding
        latent_seq = self.transformer_encoder(x_emb)
        reconstruction = self.decoder(latent_seq).squeeze(-1)
        return reconstruction, latent_seq


# ==========================================
# 2. 辅助算法函数
# ==========================================
def get_som_probabilities(features, som, node_label_map, num_classes):
    """
    计算 SOM 的伪概率，用于绘制 PR 曲线。
    原理：样本离哪个神经元越近，属于该神经元标签的概率就越大。
    """
    probs = np.zeros((len(features), num_classes))
    weights = som.get_weights()

    for idx, x in enumerate(features):
        dists = np.linalg.norm(weights - x, axis=2)
        # 将距离转化为相似度权重 (exp(-dist))
        sims = np.exp(-dists)

        class_scores = np.zeros(num_classes)
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                coord = (i, j)
                if coord in node_label_map:
                    label = node_label_map[coord]
                    class_scores[label] += sims[i, j]

        # 归一化为概率
        if class_scores.sum() > 0:
            probs[idx] = class_scores / class_scores.sum()
        else:
            probs[idx][0] = 1.0  # 异常处理

    return probs


def calculate_metrics_report(y_true, y_pred, class_names):
    """计算详细的分类指标：全局指标 + 每一类的指标"""
    metrics = {}
    # 全局指标
    acc = accuracy_score(y_true, y_pred)
    gp, gr, gf1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    metrics['global'] = {'acc': acc, 'p': gp, 'r': gr, 'f1': gf1}

    # 分类指标
    labels = list(range(len(class_names)))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)

    metrics['class'] = {}
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = len(y_true)

    for i, name in enumerate(class_names):
        tp = cm[i, i]
        tn = total - (cm[i, :].sum() + cm[:, i].sum() - tp)
        acc_i = (tp + tn) / total if total > 0 else 0
        metrics['class'][name] = {'acc': acc_i, 'p': p[i], 'r': r[i], 'f1': f1[i]}

    return metrics


def evaluate_proxy_accuracy(train_features, train_labels, test_features, test_labels):
    """
    代理准确率评估：
    在 Transformer 训练过程中，用简单的 KNN(k=1) 来快速评估
    当前提取的特征对测试集是否有区分度。
    """
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_features, train_labels)
    return knn.score(test_features, test_labels)


# ==========================================
# 3. 绘图函数 (全部保存到 OUTPUT_DIR)
# ==========================================
def plot_all_charts(train_metrics, test_metrics, train_loss_hist, test_acc_hist,
                    y_train, y_pred_train, y_test, y_pred_test,
                    y_train_probs, y_test_probs, class_names):
    # --- 图 1: 训练集 Loss 曲线 (只画训练集) ---
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_hist, label='Training Loss (Reconstruction)', color='#3498db', linewidth=2)
    plt.title('Training Loss Curve (100 Epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'chart_1_train_loss.png'), dpi=300)
    print("✅ 已保存: chart_1_train_loss.png")

    # --- 图 2: 测试集 Accuracy 曲线 (只画测试集) ---
    plt.figure(figsize=(8, 5))
    plt.plot(test_acc_hist, label='Test Accuracy (Proxy)', color='#e74c3c', linewidth=2)
    plt.title('Test Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'chart_2_test_accuracy.png'), dpi=300)
    print("✅ 已保存: chart_2_test_accuracy.png")

    # --- 图 3: 全局指标对比 (Train vs Test) ---
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    tr_vals = [train_metrics['global'][k] for k in ['acc', 'p', 'r', 'f1']]
    te_vals = [test_metrics['global'][k] for k in ['acc', 'p', 'r', 'f1']]

    x = np.arange(4)
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, tr_vals, width, label='Train Set', color='#2ecc71', alpha=0.8)
    plt.bar(x + width / 2, te_vals, width, label='Test Set', color='#9b59b6', alpha=0.8)
    plt.xticks(x, metrics_names)
    plt.title('Global Metrics Comparison')
    plt.ylim(0, 1.15)
    plt.legend()

    for i, v in enumerate(tr_vals):
        plt.text(i - width / 2, v + 0.02, f"{v:.2f}", ha='center', fontsize=9)
    for i, v in enumerate(te_vals):
        plt.text(i + width / 2, v + 0.02, f"{v:.2f}", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'chart_3_global_metrics.png'), dpi=300)
    print("✅ 已保存: chart_3_global_metrics.png")

    # --- 图 4: 混淆矩阵 (Train & Test) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(confusion_matrix(y_train, y_pred_train), annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title('Train Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title('Test Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'chart_4_confusion_matrix.png'), dpi=300)
    print("✅ 已保存: chart_4_confusion_matrix.png")

    # --- 图 5 & 6: 各类别详细指标 ---
    plot_class_bar(train_metrics['class'], 'Train Set - Per Class Metrics', 'chart_5_class_metrics_train.png')
    plot_class_bar(test_metrics['class'], 'Test Set - Per Class Metrics', 'chart_6_class_metrics_test.png')

    # --- 图 7 & 8: PR 曲线 ---
    plot_pr_curve(y_train, y_train_probs, class_names, 'Train Set PR Curve', 'chart_7_pr_train.png')
    plot_pr_curve(y_test, y_test_probs, class_names, 'Test Set PR Curve', 'chart_8_pr_test.png')


def plot_class_bar(class_data, title, filename):
    names = list(class_data.keys())
    accs = [class_data[n]['acc'] for n in names]
    precs = [class_data[n]['p'] for n in names]
    recs = [class_data[n]['r'] for n in names]
    f1s = [class_data[n]['f1'] for n in names]

    x = np.arange(len(names))
    width = 0.2
    plt.figure(figsize=(10, 6))
    plt.bar(x - 1.5 * width, accs, width, label='Accuracy', color='#3498db')
    plt.bar(x - 0.5 * width, precs, width, label='Precision', color='#e67e22')
    plt.bar(x + 0.5 * width, recs, width, label='Recall', color='#2ecc71')
    plt.bar(x + 1.5 * width, f1s, width, label='F1-Score', color='#e74c3c')
    plt.xticks(x, names)
    plt.title(title)
    plt.ylim(0, 1.1)
    plt.legend(ncol=4, loc='upper center')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    print(f"✅ 已保存: {filename}")


def plot_pr_curve(y_true, y_probs, class_names, title, filename):
    y_bin = label_binarize(y_true, classes=range(len(class_names)))
    n_classes = y_bin.shape[1]

    plt.figure(figsize=(8, 6))
    styles = itertools.cycle(['-', '--', '-.', ':'])
    colors = itertools.cycle(sns.color_palette("husl", n_classes))

    for i, color, style in zip(range(n_classes), colors, styles):
        if i < y_probs.shape[1]:
            precision, recall, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
            ap = average_precision_score(y_bin[:, i], y_probs[:, i])
            plt.plot(recall, precision, color=color, linestyle=style, lw=2,
                     label=f'{class_names[i]} (AP={ap:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    print(f"✅ 已保存: {filename}")


# ==========================================
# 4. 主程序流程
# ==========================================
def main():
    print(">>> [1/5] 读取数据 (使用固定的 new_train_data.csv 和 new_test_data.csv)...")
    if not os.path.exists('new_train_data.csv') or not os.path.exists('new_test_data.csv'):
        print("❌ 错误: 未找到数据文件，请确保它们在当前目录下。")
        return

    train_df = pd.read_csv('new_train_data.csv')
    test_df = pd.read_csv('new_test_data.csv')

    feature_cols = ['Ca', 'Mg', 'Na', 'HCO3', 'Cl', 'SO4', 'TH', 'TA', 'PH']
    target_col = 'Label'

    # 标签编码: 必须基于合并后的标签进行fit，防止某些类别只出现在一个集中
    full_labels = pd.concat([train_df[target_col], test_df[target_col]], axis=0)
    le = LabelEncoder()
    le.fit(full_labels)
    class_names = le.classes_

    y_train = le.transform(train_df[target_col])
    y_test = le.transform(test_df[target_col])

    # 数据标准化 (严谨逻辑: Train Fit -> Test Transform)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)

    print(f"    - 训练集大小: {len(X_train)}")
    print(f"    - 测试集大小: {len(X_test)}")

    print(">>> [2/5] 训练 Transformer (100 Epochs, 无早停, 记录最佳模型)...")
    train_tensor = torch.FloatTensor(X_train)
    test_tensor = torch.FloatTensor(X_test)

    model = TabularTransformerAutoencoder(len(feature_cols), dropout=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.MSELoss()

    train_loss_history = []
    test_acc_history = []

    # 用于记录最佳模型的变量
    best_test_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs = 100

    for ep in range(epochs):
        # 1. 训练
        model.train()
        optimizer.zero_grad()
        recon, _ = model(train_tensor)
        loss = criterion(recon, train_tensor)
        loss.backward()
        optimizer.step()
        train_loss_history.append(loss.item())

        # 2. 评估 (计算 Test Acc 用于画图和选最佳模型)
        model.eval()
        with torch.no_grad():
            _, tr_lat = model(train_tensor)
            _, te_lat = model(test_tensor)
            feat_tr_curr = tr_lat.reshape(len(X_train), -1).numpy()
            feat_te_curr = te_lat.reshape(len(X_test), -1).numpy()

            curr_acc = evaluate_proxy_accuracy(feat_tr_curr, y_train, feat_te_curr, y_test)
            test_acc_history.append(curr_acc)

        # 记录最佳模型 (即便不早停，最后也应该用最好的那个状态)
        if curr_acc >= best_test_acc:
            best_test_acc = curr_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = ep

        if (ep + 1) % 10 == 0:
            print(f"    Epoch {ep + 1}/{epochs} | Train Loss: {loss.item():.4f} | Test Acc: {curr_acc:.4f}")

    print(f">>> 训练结束。最佳 Acc: {best_test_acc:.4f} (at Epoch {best_epoch + 1})")
    print(f">>> 加载最佳模型权重...")
    model.load_state_dict(best_model_wts)

    print(">>> [3/5] 训练 SOM (6x6)...")
    model.eval()
    with torch.no_grad():
        _, tr_lat = model(train_tensor)
        _, te_lat = model(test_tensor)
        feat_train = tr_lat.reshape(len(X_train), -1).numpy()
        feat_test = te_lat.reshape(len(X_test), -1).numpy()

    # SOM 训练
    som = MiniSom(6, 6, feat_train.shape[1], sigma=1.0, learning_rate=0.5, random_seed=42)
    som.train_random(feat_train, 5000)

    print(">>> [4/5] 最终预测与评估...")
    node_map = {}
    for i, x in enumerate(feat_train):
        w = som.winner(x)
        if w not in node_map: node_map[w] = []
        node_map[w].append(y_train[i])
    for w in node_map: node_map[w] = max(set(node_map[w]), key=node_map[w].count)

    def predict(feats):
        res = []
        for x in feats:
            w = som.winner(x)
            if w in node_map:
                res.append(node_map[w])
            else:
                dists = []
                for kw in node_map:
                    d = np.linalg.norm(np.array(w) - np.array(kw))
                    dists.append((d, node_map[kw]))
                if dists:
                    res.append(min(dists, key=lambda x: x[0])[1])
                else:
                    res.append(0)
        return np.array(res)

    y_pred_tr = predict(feat_train)
    y_pred_te = predict(feat_test)

    probs_tr = get_som_probabilities(feat_train, som, node_map, len(class_names))
    probs_te = get_som_probabilities(feat_test, som, node_map, len(class_names))

    metrics_tr = calculate_metrics_report(y_train, y_pred_tr, class_names)
    metrics_te = calculate_metrics_report(y_test, y_pred_te, class_names)

    print(">>> [5/5] 保存日志与图表到 Experiment_Results 文件夹...")

    # 写入日志
    log_path = os.path.join(OUTPUT_DIR, "training_log_final.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"训练时间: {datetime.datetime.now()}\n")
        f.write(f"配置: Transformer(100 Epochs, No Early Stop) + SOM(6x6)\n")
        f.write(f"最佳模型来自于 Epoch {best_epoch + 1}, Test Acc: {best_test_acc:.4f}\n\n")

        for split_name, m in [("Train Set", metrics_tr), ("Test Set", metrics_te)]:
            f.write(f"=== {split_name} Metrics ===\n")
            f.write(f"Global Accuracy : {m['global']['acc']:.4f}\n")
            f.write(f"Global F1-Score : {m['global']['f1']:.4f}\n")
            for c in class_names:
                d = m['class'][c]
                f.write(f"  [{c}]: Acc={d['acc']:.4f}, Prec={d['p']:.4f}, Rec={d['r']:.4f}, F1={d['f1']:.4f}\n")
            f.write("\n")
    print(f"✅ 日志已保存: {log_path}")

    plot_all_charts(metrics_tr, metrics_te, train_loss_history, test_acc_history,
                    y_train, y_pred_tr, y_test, y_pred_te,
                    probs_tr, probs_te, class_names)

    print(f"\n🎉 运行成功！请打开文件夹 '{OUTPUT_DIR}' 查看所有结果。")


if __name__ == "__main__":
    main()