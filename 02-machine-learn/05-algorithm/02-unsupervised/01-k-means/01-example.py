import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 1. 生成模拟数据 ----------
X, y_true = make_blobs(
    n_samples=300,          # 样本数
    n_features=2,           # 特征数（二维，便于可视化）
    centers=3,              # 实际簇数
    cluster_std=0.60,       # 簇的标准差
    random_state=0
)

# ---------- 2. 数据标准化（极其重要！） ----------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 查看原始数据分布
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], s=30)
plt.title("标准化后的数据")

# ---------- 3. 肘部法则选择最佳 K ----------
sse = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=0)
    km.fit(X_scaled)
    sse.append(km.inertia_)       # inertia_ 即簇内平方和 SSE

plt.subplot(1, 3, 2)
plt.plot(K_range, sse, marker='o')
plt.xlabel('K')
plt.ylabel('SSE')
plt.title('肘部法则')

# ---------- 4. 使用 K-means++ 聚类（K=3） ----------
best_k = 3   # 根据肘部法则观察拐点
km = KMeans(
    n_clusters=best_k,
    init='k-means++',      # 优化初始化
    n_init=10,             # 用不同随机初始值运行次数，选 SSE 最小的结果
    max_iter=300,
    random_state=0
)
y_km = km.fit_predict(X_scaled)

# 轮廓系数评估
sil_score = silhouette_score(X_scaled, y_km)
print(f"轮廓系数 (Silhouette Score): {sil_score:.3f}")

# 可视化聚类结果
plt.subplot(1, 3, 3)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_km, s=30, cmap='viridis')
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    c='red', marker='x', s=200, linewidths=3, label='Centroids'
)
plt.title(f'K-means 聚类结果 (K={best_k})')
plt.legend()
plt.tight_layout()
plt.show()

# ---------- 5. 从零实现 K-means（含 K-means++） ----------
class KMeansFromScratch:
    def __init__(self, n_clusters=3, max_iters=300, tol=1e-4, init='k-means++'):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.init = init
        self.centroids = None
        self.labels_ = None

    def _init_centroids(self, X):
        """用随机或 K-means++ 方法初始化质心"""
        if self.init == 'random':
            idx = np.random.choice(len(X), self.n_clusters, replace=False)
            self.centroids = X[idx]
        elif self.init == 'k-means++':
            # 1. 随机选第一个中心
            self.centroids = [X[np.random.randint(len(X))]]
            for _ in range(1, self.n_clusters):
                # 2. 计算每个样本到最近中心的距离平方
                dist_sq = np.array([
                    min([np.linalg.norm(x - c) ** 2 for c in self.centroids])
                    for x in X
                ])
                # 3. 按距离平方比例的概率选择下一个中心
                probs = dist_sq / dist_sq.sum()
                cumprobs = probs.cumsum()
                r = np.random.rand()
                idx = np.searchsorted(cumprobs, r)
                self.centroids.append(X[idx])
            self.centroids = np.array(self.centroids)
        else:
            raise ValueError("init must be 'random' or 'k-means++'")

    def fit(self, X):
        self._init_centroids(X)
        for i in range(self.max_iters):
            # ---- 分配样本到最近的质心 ----
            # 计算距离矩阵: (n_samples, n_clusters)
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            # ---- 更新质心 ----
            new_centroids = np.array([
                X[labels == j].mean(axis=0) for j in range(self.n_clusters)
            ])

            # ---- 检查收敛 ----
            shift = np.linalg.norm(self.centroids - new_centroids)
            self.centroids = new_centroids
            if shift < self.tol:
                break

        self.labels_ = labels
        return self

# 使用自实现版本聚类并可视化
custom_km = KMeansFromScratch(n_clusters=3, init='k-means++')
custom_km.fit(X_scaled)

plt.figure(figsize=(6, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=custom_km.labels_, s=30, cmap='viridis')
plt.scatter(
    custom_km.centroids[:, 0], custom_km.centroids[:, 1],
    c='red', marker='x', s=200, linewidths=3, label='Centroids'
)
plt.title("从零实现 K-means 聚类")
plt.legend()
plt.show()