import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
import warnings

# ==========================================
# 0. 配置日志与环境
# ==========================================
warnings.filterwarnings('ignore')
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# 配置日志：同时输出到控制台和文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler("training_log.txt", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()


# ==========================================
# 1. MiniSom 实现 (保持不变)
# ==========================================
class MiniSom:
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5, random_seed=None):
        if random_seed:
            np.random.seed(random_seed)
        self._weights = np.random.rand(x, y, input_len) * 2 - 1
        self._weights /= np.linalg.norm(self._weights, axis=-1, keepdims=True)
        self._activation_map = np.zeros((x, y))
        self._neigx = np.arange(x)
        self._neigy = np.arange(y)
        self._sigma = sigma
        self._learning_rate = learning_rate

    def _activate(self, x):
        x = x / (np.linalg.norm(x) + 1e-9)
        s = np.subtract(x, self._weights)
        self._activation_map = np.linalg.norm(s, axis=-1)
        return self._activation_map

    def winner(self, x):
        self._activate(x)
        return np.unravel_index(self._activation_map.argmin(), self._activation_map.shape)

    def update(self, x, win, t, max_iter):
        eta = self._learning_rate * np.exp(-t / max_iter)
        sig = self._sigma * np.exp(-t / max_iter)
        g, h = win
        dist = np.square(self._neigx[:, np.newaxis] - g) + np.square(self._neigy[np.newaxis, :] - h)
        influence = np.exp(-dist / (2 * sig * sig + 1e-9))
        x = x / (np.linalg.norm(x) + 1e-9)
        self._weights += eta * influence[:, :, np.newaxis] * (x - self._weights)
        self._weights /= (np.linalg.norm(self._weights, axis=-1, keepdims=True) + 1e-9)

    def train(self, data, num_iteration):
        for t in range(num_iteration):
            idx = t % len(data)
            x = data[idx]
            win = self.winner(x)
            self.update(x, win, t, num_iteration)

    def distance_map(self):
        um = np.zeros((self._weights.shape[0], self._weights.shape[1]))
        it = np.nditer(um, flags=['multi_index'])
        while not it.finished:
            for ii in range(it.multi_index[0] - 1, it.multi_index[0] + 2):
                for jj in range(it.multi_index[1] - 1, it.multi_index[1] + 2):
                    if ii >= 0 and ii < self._weights.shape[0] and jj >= 0 and jj < self._weights.shape[1]:
                        w_1 = self._weights[it.multi_index]
                        w_2 = self._weights[ii, jj]
                        um[it.multi_index] += np.linalg.norm(w_1 - w_2)
            it.iternext()
        um = um / (um.max() + 1e-9)
        return um


# ==========================================
# 2. 数据处理与增强
# ==========================================
def load_data():
    logger.info("正在加载数据...")
    train_df = pd.read_csv('new_train_data.csv')
    test_df = pd.read_csv('new_test_data.csv')

    # 删除 ID
    if 'SampleID' in train_df.columns:
        train_df = train_df.drop('SampleID', axis=1)
        test_df = test_df.drop('SampleID', axis=1)

    # 标签编码
    le = LabelEncoder()
    all_labels = pd.concat([train_df['Label'], test_df['Label']])
    le.fit(all_labels)
    y_train_raw = le.transform(train_df['Label'])
    y_test = le.transform(test_df['Label'])

    # 建立映射: 0->A, 1->B...
    label_map = {i: chr(65 + i) for i in range(len(le.classes_))}
    logger.info(f"类别映射关系: {dict(zip(le.classes_, label_map.values()))}")

    X_train_raw = train_df.drop('Label', axis=1).values
    X_test = test_df.drop('Label', axis=1).values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test)

    # 数据增强 (仅用于训练)
    X_augmented = []
    y_augmented = []
    aug_factor = 40
    noise_level = 0.08

    for i in range(len(X_train_scaled)):
        X_augmented.append(X_train_scaled[i])
        y_augmented.append(y_train_raw[i])
        for _ in range(aug_factor):
            noise = np.random.normal(0, noise_level, X_train_scaled.shape[1])
            X_augmented.append(X_train_scaled[i] + noise)
            y_augmented.append(y_train_raw[i])

    X_train_aug = np.array(X_augmented)
    y_train_aug = np.array(y_augmented)
    X_train_aug, y_train_aug = shuffle(X_train_aug, y_train_aug, random_state=SEED)

    logger.info(f"原始训练集: {len(X_train_scaled)}, 增强后训练集: {len(X_train_aug)}")

    return X_train_aug, y_train_aug, X_train_scaled, y_train_raw, X_test_scaled, y_test, label_map


# ==========================================
# 3. Transformer 模型
# ==========================================
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=32, num_classes=4):
        super(TransformerModel, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1) + self.pos_enc
        feat = self.encoder(x).squeeze(1)
        return self.classifier(feat), feat


# ==========================================
# 4. 主程序
# ==========================================
X_train_aug, y_train_aug, X_train_real, y_train_real, X_test, y_test, label_map = load_data()

# Tensor 转换
X_train_t = torch.FloatTensor(X_train_aug)
y_train_t = torch.LongTensor(y_train_aug)
X_test_t = torch.FloatTensor(X_test)
X_train_real_t = torch.FloatTensor(X_train_real)
y_train_real_t = torch.LongTensor(y_train_real)

# 模型初始化
model = TransformerModel(X_train_aug.shape[1], 32, num_classes=len(label_map))
optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

logger.info("开始 Transformer 预训练...")
for epoch in range(120):
    model.train()
    optimizer.zero_grad()
    logits, _ = model(X_train_t)
    loss = criterion(logits, y_train_t)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            l_real, _ = model(X_train_real_t)
            acc_real = (l_real.argmax(1) == y_train_real_t).float().mean()
            logger.info(f"Epoch {epoch + 1}/120 | Loss: {loss.item():.4f} | Real Train Acc: {acc_real:.2%}")

# 特征提取与 SOM
logger.info("训练 SOM...")
model.eval()
with torch.no_grad():
    _, feat_aug = model(X_train_t)
    _, feat_real = model(X_train_real_t)
    _, feat_test = model(X_test_t)

som = MiniSom(10, 10, 32, sigma=1.5, learning_rate=0.5, random_seed=SEED)
som.train(feat_aug.numpy(), 5000)

# SOM 标记
map_labels = {}
for i, x in enumerate(feat_aug.numpy()):
    w = som.winner(x)
    if w not in map_labels: map_labels[w] = []
    map_labels[w].append(y_train_aug[i])

grid_labels = np.zeros((10, 10)) - 1
for i in range(10):
    for j in range(10):
        if (i, j) in map_labels:
            l = map_labels[(i, j)]
            grid_labels[i, j] = max(set(l), key=l.count)


def som_predict(feats):
    preds = []
    for x in feats:
        w = som.winner(x)
        l = grid_labels[w]
        if l == -1:  # 简单填充
            preds.append(0)
        else:
            preds.append(l)
    return np.array(preds)


y_pred_train = som_predict(feat_real.numpy())
y_pred_test = som_predict(feat_test.numpy())


# 指标计算
def get_metrics(y_true, y_pred):
    return [
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, average='weighted', zero_division=0),
        recall_score(y_true, y_pred, average='weighted', zero_division=0),
        f1_score(y_true, y_pred, average='weighted', zero_division=0)
    ]


m_train = get_metrics(y_train_real, y_pred_train)
m_test = get_metrics(y_test, y_pred_test)

logger.info("=" * 30)
logger.info(f"最终测试集准确率: {m_test[0]:.4f}")
logger.info(f"Precision: {m_test[1]:.4f} | Recall: {m_test[2]:.4f} | F1: {m_test[3]:.4f}")
logger.info("=" * 30)

# ==========================================
# 5. 绘图与保存 (分开保存)
# ==========================================
# 设置绘图风格
plt.style.use('default')

# 图1: 混淆矩阵
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_train_real, y_pred_train), annot=True, fmt='d', cmap='Blues')
plt.title('Train Confusion Matrix')
plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, fmt='d', cmap='Greens')
plt.title('Test Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()

# 图2: 指标对比
plt.figure(figsize=(8, 6))
x = np.arange(4)
plt.bar(x - 0.2, m_train, 0.4, label='Train', color='#4c72b0', alpha=0.8)
plt.bar(x + 0.2, m_test, 0.4, label='Test', color='#55a868', alpha=0.8)
plt.xticks(x, ['Acc', 'Pre', 'Rec', 'F1'])
plt.legend()
plt.title('Metrics Comparison')
plt.savefig('metrics_comparison.png', dpi=300)
plt.close()

# 图3: SOM 可视化
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("SOM U-Matrix (Distance)")
plt.pcolor(som.distance_map().T, cmap='bone_r')
plt.colorbar()
plt.subplot(1, 2, 2)
plt.title("SOM Grid Labels")
sns.heatmap(grid_labels, annot=True, fmt='.0f', cmap='tab10', cbar=False)
plt.tight_layout()
plt.savefig('som_visualization.png', dpi=300)
plt.close()

# 图4: 特征分布图 (t-SNE) - 带 ABCD 标签
logger.info("正在绘制 t-SNE...")
tsne = TSNE(n_components=2, perplexity=min(20, len(X_test) - 1), random_state=SEED)
tsne_res = tsne.fit_transform(feat_test.numpy())

plt.figure(figsize=(8, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 对应 A, B, C, D
markers = ['o', 's', '^', 'D']

for cls_idx in range(len(label_map)):
    idx = (y_test == cls_idx)
    plt.scatter(tsne_res[idx, 0], tsne_res[idx, 1],
                c=colors[cls_idx],
                label=label_map[cls_idx],  # 这里使用 A, B, C, D
                marker=markers[cls_idx],
                s=100, edgecolors='k', alpha=0.8)

plt.legend(loc='upper right', title="Type", fontsize=12)
plt.title('Feature Distribution (t-SNE)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_distribution.png', dpi=300)
plt.close()

logger.info("所有图像已保存。程序结束。")