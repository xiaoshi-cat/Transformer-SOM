import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from minisom import MiniSom
import os
import shutil

# ================= 配置 =================
OUTPUT_DIR = "Ablation_Study_Results"
if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['axes.unicode_minus'] = False


# ================= 1. 定义不同的模型 =================

# 模型 A: 纯 SOM (不需要 PyTorch 模型，直接用原始数据)

# 模型 B: MLP Autoencoder (普通全连接，无 Attention)
class SimpleMLP_AE(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super(SimpleMLP_AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent


# 模型 C: 你的 Supervised Transformer (完整版)
# (为了节省篇幅，这里复用你之前的类定义，略微简化)
class Transformer_Full(nn.Module):
    # ... (使用上一版代码的 SupervisedTransformerAE 类) ...
    pass


# 模型 D: Unsupervised Transformer (去掉分类头)
class Transformer_Unsup(nn.Module):
    # ... (和 Full 一样，但 forward 里不返回 class_logits，也不算分类 Loss) ...
    pass


# ================= 2. 统一训练与评估函数 =================
def run_experiment(experiment_name, X, y, model_type="som_only"):
    print(f"\n>>> Running Experiment: {experiment_name}")

    feat_train = X  # 默认特征就是原始数据

    # 如果不是纯 SOM，就需要先训练特征提取器
    if model_type != "som_only":
        input_dim = X.shape[1]
        num_classes = len(np.unique(y))

        # 初始化模型
        if model_type == "mlp":
            model = SimpleMLP_AE(input_dim)
            criterion = nn.MSELoss()
        elif model_type == "trans_unsup":
            model = Transformer_Unsup(...)  # 实例化
            criterion = nn.MSELoss()
        elif model_type == "trans_full":
            model = Transformer_Full(...)  # 实例化
            criterion1 = nn.MSELoss()
            criterion2 = nn.CrossEntropyLoss()

        # 训练过程 (简略)
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        tensor_x = torch.FloatTensor(X)
        tensor_y = torch.LongTensor(y)

        for ep in range(150):
            optimizer.zero_grad()
            if model_type == "trans_full":
                recon, lat, logits = model(tensor_x)
                loss = criterion1(recon, tensor_x) + 0.8 * criterion2(logits, tensor_y)
            else:
                recon, lat = model(tensor_x)
                loss = criterion(recon, tensor_x)
            loss.backward()
            optimizer.step()

        # 提取隐特征
        with torch.no_grad():
            _, lat_tensor = model(tensor_x)[:2]  # 取前两个返回值
            feat_train = lat_tensor.numpy()

    # --- 统一部分：SOM 训练 ---
    # 无论前面特征怎么来的，最后都进 SOM
    som = MiniSom(6, 6, feat_train.shape[1], sigma=1.0, learning_rate=0.5)
    som.train_random(feat_train, 5000)

    # --- 评估：计算 Purity 或 Accuracy ---
    # 这里写一个简单的基于 SOM 投票的 Accuracy 计算函数
    # ... (参考之前的 predict 函数) ...
    acc = calculate_som_accuracy(som, feat_train, y)
    print(f"   -> Accuracy: {acc:.4f}")

    return acc, feat_train


# ================= 3. 主流程与绘图 =================
def main():
    # ... 读取数据，标准化 ...

    results = {}

    # 实验 1: Baseline (Raw Data + SOM)
    acc1, _ = run_experiment("1_Raw_SOM", X_train, y_train, "som_only")
    results['Raw Data + SOM'] = acc1

    # 实验 2: MLP + SOM (验证 Attention 的作用)
    acc2, _ = run_experiment("2_MLP_SOM", X_train, y_train, "mlp")
    results['MLP + SOM'] = acc2

    # 实验 3: Unsupervised Transformer + SOM (验证监督信号的作用)
    acc3, _ = run_experiment("3_Unsup_Trans_SOM", X_train, y_train, "trans_unsup")
    results['Unsup Trans + SOM'] = acc3

    # 实验 4: Ours (Supervised Transformer + SOM)
    acc4, _ = run_experiment("4_Ours_Full", X_train, y_train, "trans_full")
    results['Ours (Supervised Trans)'] = acc4

    # --- 画柱状图对比 ---
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results.keys(), results.values(), color=['gray', 'skyblue', 'orange', 'red'])
    plt.title("Ablation Study: Accuracy Comparison", fontsize=14)
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.1)

    # 在柱子上标数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f"{yval:.1%}", ha='center', va='bottom')

    plt.savefig(os.path.join(OUTPUT_DIR, "Ablation_Chart.png"), dpi=300)
    print("消融实验图表已保存！")


if __name__ == "__main__":
    main()