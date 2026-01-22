import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import time

# 忽略警告
warnings.filterwarnings("ignore")

# --- 配置参数 ---
SEED = 42
AUGMENT_FACTOR = 50  # 数据增强倍数 (针对小样本极其重要)
SOM_SIZE = 15  # SOM 网格大小
EPOCHS = 50  # Transformer 训练轮数
DEVICE = torch.device('cpu')  # 工程部署建议用 CPU 保证稳定性

# 设置随机种子
np.random.seed(SEED)
torch.manual_seed(SEED)


# --- 1. 化学特征工程模块 ---
class ChemicalPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.imputer = KNNImputer(n_neighbors=3)  # 用于填充无法计算的缺失值

    def process(self, df):
        # A. 化学计算补全 (基于守恒定律)
        # 1. 补全总硬度 (TH) = 2.497*Ca + 4.118*Mg
        mask_th = df['TH'].isna() & df['Ca'].notna() & df['Mg'].notna()
        df.loc[mask_th, 'TH'] = 2.497 * df.loc[mask_th, 'Ca'] + 4.118 * df.loc[mask_th, 'Mg']

        # 2. 补全 TDS (简单加和估算)
        # TDS approx = Na+K + Ca + Mg + Cl + SO4 + HCO3 + CO3
        ions = ['Na+K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'CO3']
        # 先把离子的NaN填为0用于计算TDS (仅用于此步计算)
        temp_sum = df[ions].fillna(0).sum(axis=1)
        mask_tds = df['TDS'].isna()
        df.loc[mask_tds, 'TDS'] = temp_sum[mask_tds]

        # 3. 补全总碱度 (TA) approx HCO3 + 2*CO3
        mask_ta = df['TA'].isna()
        # 假设 CO3 缺失为 0
        co3_fill = df['CO3'].fillna(0)
        hco3_fill = df['HCO3'].fillna(0)
        df.loc[mask_ta, 'TA'] = hco3_fill[mask_ta] + 2 * co3_fill[mask_ta]

        # B. 衍生特征计算 (比值特征)
        # 注意防止除零错误，分母加一个极小值 1e-5
        # 1. rNa/rCl (钠氯比) - 摩尔比，需先除以原子量: Na=23, Cl=35.5
        # 这里简化直接用质量比作为特征，或者严格转摩尔。为了模型效果，直接相比即可，模型会学习关系。
        df['rNa/rCl'] = df['Na+K'] / (df['Cl'] + 1e-5)

        # 2. rSO4/rCl (脱硫系数相关)
        df['rSO4/rCl'] = df['SO4'] / (df['Cl'] + 1e-5)

        # 3. rCa/rMg (钙镁比)
        df['rCa/rMg'] = df['Ca'] / (df['Mg'] + 1e-5)

        # C. 剩余缺失值处理 (KNN插值)
        # 只选取数值列
        numeric_cols = ['Na+K', 'Ca', 'Mg', 'Cl', 'SO4', 'HCO3', 'CO3', 'TDS', 'TH', 'TA', 'PH', 'rNa/rCl', 'rSO4/rCl',
                        'rCa/rMg']
        df_numeric = df[numeric_cols]

        # 执行 KNN 插值
        df_filled = pd.DataFrame(self.imputer.fit_transform(df_numeric), columns=numeric_cols)

        # D. 归一化
        X = self.scaler.fit_transform(df_filled)

        return X, df['Label'].values, numeric_cols


# --- 2. Transformer 模型 ---
class FeatureTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, num_classes):
        super(FeatureTransformer, self).__init__()
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, input_dim, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model * input_dim, num_classes)

    def forward(self, x):
        b, f = x.shape
        x = x.unsqueeze(-1)  # (Batch, Features, 1)
        x = self.embedding(x) + self.pos_encoder
        x = self.transformer_encoder(x)
        features = x.reshape(b, -1)
        output = self.fc_out(features)
        return output, features


# --- 3. SOM 概率模型 ---
class ProbabilisticSOM:
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5):
        self.x = x
        self.y = y
        self.weights = np.random.random((x, y, input_len)) * 2 - 1
        self._neigx = np.arange(x)
        self._neigy = np.arange(y)
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.node_labels = None
        self.node_probs = None

    def _activate(self, x):
        s = np.subtract(x, self.weights)
        return np.linalg.norm(s, axis=-1)

    def train(self, data, num_iteration):
        for t in range(num_iteration):
            x = data[np.random.randint(0, len(data))]
            dists = self._activate(x)
            win = np.unravel_index(dists.argmin(), (self.x, self.y))

            # Update
            eta = self.learning_rate * (1 - t / num_iteration)
            sig = self.sigma * (1 - t / num_iteration)

            ax, ay = win
            dist_sq = (self._neigx - ax) ** 2 + (self._neigy - ay)[:, np.newaxis] ** 2
            dist_sq = dist_sq.T  # Shape fix

            h = np.exp(-dist_sq / (2 * sig ** 2 + 1e-10))
            self.weights += eta * h[:, :, np.newaxis] * (x - self.weights)

    def map_labels(self, data, labels, num_classes):
        self.node_labels = np.zeros((self.x, self.y, num_classes))
        for i, x in enumerate(data):
            dists = self._activate(x)
            win = np.unravel_index(dists.argmin(), (self.x, self.y))
            self.node_labels[win][labels[i]] += 1

        sums = self.node_labels.sum(axis=2, keepdims=True)
        self.node_probs = np.divide(self.node_labels, sums, out=np.zeros_like(self.node_labels), where=sums != 0)

    def predict_proba(self, x, inference_sigma=2.0):
        dists = self._activate(x)
        # 核心概率逻辑：距离转相似度
        similarity = np.exp(-dists ** 2 / (2 * inference_sigma ** 2))
        sim_flat = similarity.flatten()
        probs_flat = self.node_probs.reshape(-1, self.node_probs.shape[-1])
        weighted_probs = np.dot(sim_flat, probs_flat)
        total = np.sum(weighted_probs)
        return weighted_probs / total if total > 0 else np.ones(self.node_probs.shape[-1]) / self.node_probs.shape[-1]


# --- 4. 主程序 (交叉验证) ---
def main():
    print(">>> 正在初始化系统...")

    # 1. 读取数据
    try:
        # 优先尝试 'gbk' 编码 (Excel/Windows生成的CSV常用此编码)
        df = pd.read_csv('train_data.csv', encoding='gbk')
        print("成功使用 GBK 编码读取文件。")
    except UnicodeDecodeError:
        try:
            # 如果失败，尝试 'utf-8' (标准编码)
            df = pd.read_csv('train_data.csv', encoding='utf-8')
            print("成功使用 UTF-8 编码读取文件。")
        except UnicodeDecodeError:
            # 如果还失败，尝试 'gb18030' (超大字符集)
            df = pd.read_csv('train_data.csv', encoding='gb18030')
            print("成功使用 GB18030 编码读取文件。")
    except Exception as e:
        print(f"严重错误：无法读取 train_data.csv。原因: {e}")
        return

    # 2. 预处理
    print(">>> 正在进行化学特征补全与计算...")
    preprocessor = ChemicalPreprocessor()
    X, y_raw, feature_names = preprocessor.process(df)

    # 标签编码
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_
    num_classes = len(class_names)
    print(f"检测到 {num_classes} 类水源：{class_names}")

    # 3. 留一法交叉验证 (Leave-One-Out CV)
    n_samples = len(X)
    print(f"\n>>> 开始留一法交叉验证 (共 {n_samples} 轮)...")

    results = []
    y_true_all = []
    y_pred_all = []

    start_time = time.time()

    for i in range(n_samples):
        # A. 数据划分
        X_test = X[i:i + 1]
        y_test = y[i:i + 1]

        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)

        # B. 数据增强 (Seed Generation)
        X_train_aug = []
        y_train_aug = []
        for j in range(len(X_train)):
            X_train_aug.append(X_train[j])
            y_train_aug.append(y_train[j])
            # 生成虚拟样本
            for _ in range(AUGMENT_FACTOR):
                noise = np.random.normal(0, 0.03, X_train.shape[1])  # 3% 的扰动
                new_sample = np.clip(X_train[j] + noise, 0, 1)
                X_train_aug.append(new_sample)
                y_train_aug.append(y_train[j])

        X_train_aug = np.array(X_train_aug)
        y_train_aug = np.array(y_train_aug)

        # C. 训练 Transformer
        model = FeatureTransformer(input_dim=len(feature_names), d_model=16, nhead=4, num_layers=2,
                                   num_classes=num_classes).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=0.002)
        criterion = nn.CrossEntropyLoss()

        train_ds = TensorDataset(torch.FloatTensor(X_train_aug), torch.LongTensor(y_train_aug))
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

        model.train()
        for epoch in range(EPOCHS):
            for xb, yb in train_loader:
                optimizer.zero_grad()
                out, _ = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

        # D. 提取特征 & 训练 SOM
        model.eval()
        with torch.no_grad():
            _, feats_train = model(torch.FloatTensor(X_train_aug))
            _, feats_test = model(torch.FloatTensor(X_test))
            feats_train = feats_train.numpy()
            feats_test = feats_test.numpy()

        som = ProbabilisticSOM(SOM_SIZE, SOM_SIZE, feats_train.shape[1], sigma=2.0)
        som.train(feats_train, num_iteration=1000)
        som.map_labels(feats_train, y_train_aug, num_classes)

        # E. 预测
        probs = som.predict_proba(feats_test[0], inference_sigma=3.0)  # Sigma=3.0 避免盲目自信
        pred_label = np.argmax(probs)

        # F. 记录结果
        y_true_all.append(y_test[0])
        y_pred_all.append(pred_label)

        # 格式化概率输出
        prob_dict = {class_names[k]: v for k, v in enumerate(probs)}
        sorted_probs = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
        prob_str = ", ".join([f"{p:.1%} {n}" for n, p in sorted_probs if p > 0.01])

        res_entry = {
            'SampleID': df.iloc[i]['SampleID'],
            'True_Label': class_names[y_test[0]],
            'Predicted_Label': class_names[pred_label],
            'Is_Correct': pred_label == y_test[0],
            'Probability_Detail': prob_str
        }
        results.append(res_entry)

        if (i + 1) % 5 == 0:
            print(f"进度: {i + 1}/{n_samples} 已完成...")

    # 4. 生成报告
    print("\n>>> 验证完成，正在生成报告...")
    results_df = pd.DataFrame(results)

    # 计算指标
    acc = accuracy_score(y_true_all, y_pred_all)
    prec = precision_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
    rec = recall_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0)

    # 混淆矩阵 (使用英文标签防止乱码)
    # 为了绘图美观，将长中文名映射为短编码 (A, B, C...)
    short_labels = [f"Type_{k}" for k in range(len(class_names))]
    label_map_str = "\n".join([f"Type_{k}: {name}" for k, name in enumerate(class_names)])

    cm = confusion_matrix(y_true_all, y_pred_all)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=short_labels, yticklabels=short_labels)
    plt.title('Confusion Matrix (Leave-One-Out CV)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')

    # 日志内容
    log_content = f"""
    ================================================
    Water Source Identification Model - Validation Log
    ================================================
    Date: {time.strftime("%Y-%m-%d %H:%M:%S")}
    Validation Strategy: Leave-One-Out Cross-Validation (LOOCV)
    Total Samples: {n_samples}

    [Overall Metrics]
    Accuracy : {acc:.4f}
    Precision: {prec:.4f}
    Recall   : {rec:.4f}
    F1 Score : {f1:.4f}

    [Label Mapping]
    {label_map_str}

    [Detailed Classification Report]
    {classification_report(y_true_all, y_pred_all, target_names=class_names, zero_division=0)}

    [Misclassified Samples]
    {results_df[~results_df['Is_Correct']][['SampleID', 'True_Label', 'Predicted_Label', 'Probability_Detail']].to_string()}
    """

    with open('result_log.txt', 'w', encoding='utf-8') as f:
        f.write(log_content)

    results_df.to_csv('validation_results.csv', index=False, encoding='utf_8_sig')

    print(f"总耗时: {time.time() - start_time:.1f}秒")
    print(f"最终准确率 (Accuracy): {acc:.2%}")
    print("结果文件已生成：")
    print("1. validation_results.csv (包含每条数据的概率)")
    print("2. result_log.txt (详细指标与误判记录)")
    print("3. confusion_matrix.png (混淆矩阵图)")


if __name__ == "__main__":
    main()