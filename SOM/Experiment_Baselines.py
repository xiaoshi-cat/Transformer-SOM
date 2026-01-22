import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import logging
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# ==========================================
# 0. 全局配置与日志设置
# ==========================================
OUTPUT_DIR = "Experiment_Results_Baselines"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 设置日志
log_file_path = os.path.join(OUTPUT_DIR, "baseline_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')


# ==========================================
# 1. 辅助函数: 绘图与评估
# ==========================================
def calculate_and_log_metrics(y_true, y_pred, method_name, set_name):
    """
    计算指标，打印日志，并返回 metrics 字典
    """
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    log_msg = (f"[{method_name}] - {set_name} Set Results:\n"
               f"   Accuracy : {acc:.4f}\n"
               f"   Precision: {p:.4f}\n"
               f"   Recall   : {r:.4f}\n"
               f"   F1-Score : {f1:.4f}")
    logger.info(log_msg)
    return acc, p, r, f1


def plot_confusion_matrix(y_true, y_pred, class_names, method_name, set_name):
    """
    绘制并保存混淆矩阵
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.title(f'{method_name} - {set_name} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    # 保存文件
    filename = f"{method_name}_{set_name}_CM.png"
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"   📊 混淆矩阵已保存: {filename}")


# ==========================================
# 2. 数据准备
# ==========================================
def load_data():
    logger.info(">>> Loading Data...")
    if not os.path.exists('new_train_data.csv') or not os.path.exists('new_test_data.csv'):
        logger.error("❌ 错误: 数据文件不存在！")
        return None

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

    # 标准化 (Z-score)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)

    logger.info(f"Data Loaded. Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"Classes: {le.classes_}")

    return X_train, y_train, X_test, y_test, le.classes_


# ==========================================
# 3. 模型定义与执行
# ==========================================

# --- SVM ---
def run_svm(X_train, y_train, X_test, y_test, classes):
    method = "SVM"
    logger.info(f"\n{'=' * 20} Running {method} {'=' * 20}")

    clf = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
    clf.fit(X_train, y_train)

    # 预测
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    # 评估与记录
    calculate_and_log_metrics(y_train, y_pred_train, method, "Train")
    calculate_and_log_metrics(y_test, y_pred_test, method, "Test")

    # 绘图
    plot_confusion_matrix(y_train, y_pred_train, classes, method, "Train")
    plot_confusion_matrix(y_test, y_pred_test, classes, method, "Test")


# --- Random Forest ---
def run_rf(X_train, y_train, X_test, y_test, classes):
    method = "RandomForest"
    logger.info(f"\n{'=' * 20} Running {method} {'=' * 20}")

    clf = RandomForestClassifier(n_estimators=15, random_state=42)
    clf.fit(X_train, y_train)

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    calculate_and_log_metrics(y_train, y_pred_train, method, "Train")
    calculate_and_log_metrics(y_test, y_pred_test, method, "Test")

    plot_confusion_matrix(y_train, y_pred_train, classes, method, "Train")
    plot_confusion_matrix(y_test, y_pred_test, classes, method, "Test")


# --- CNN ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # 简化版: 只有 1 层卷积，卷积核数量减少到 8
        self.conv1 = nn.Conv1d(1, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()

        # 9 -> pool(2) -> 4.   8 channels * 4 = 32
        self.fc = nn.Linear(32, num_classes)  # 直接输出，去掉中间的 hidden layer

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x


def run_cnn(X_train, y_train, X_test, y_test, classes):
    method = "CNN"
    logger.info(f"\n{'=' * 20} Running {method} {'=' * 20}")

    # 准备数据
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)  # 用于评估

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=16, shuffle=True)

    # 模型设置
    model = SimpleCNN(len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # 训练
    epochs = 100
    logger.info(f"Starting CNN training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            logger.info(f"   Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(train_loader):.4f}")

    # 预测
    model.eval()
    with torch.no_grad():
        # Train Preds
        out_tr = model(X_train_t)
        _, pred_tr = torch.max(out_tr, 1)
        y_pred_train = pred_tr.numpy()

        # Test Preds
        out_te = model(X_test_t)
        _, pred_te = torch.max(out_te, 1)
        y_pred_test = pred_te.numpy()

    calculate_and_log_metrics(y_train, y_pred_train, method, "Train")
    calculate_and_log_metrics(y_test, y_pred_test, method, "Test")

    plot_confusion_matrix(y_train, y_pred_train, classes, method, "Train")
    plot_confusion_matrix(y_test, y_pred_test, classes, method, "Test")


# ==========================================
# 4. 主程序
# ==========================================
def main():
    logger.info(">>> Baseline Experiments Started")

    # 1. 加载数据
    data = load_data()
    if data is None: return
    X_train, y_train, X_test, y_test, classes = data

    # 2. 运行 SVM
    run_svm(X_train, y_train, X_test, y_test, classes)

    # 3. 运行 RF
    run_rf(X_train, y_train, X_test, y_test, classes)

    # 4. 运行 CNN
    run_cnn(X_train, y_train, X_test, y_test, classes)

    logger.info(f"\n🎉 所有实验完成！结果已保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()