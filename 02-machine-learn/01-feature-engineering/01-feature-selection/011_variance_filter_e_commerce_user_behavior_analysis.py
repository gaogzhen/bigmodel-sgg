import pandas as pd
from sklearn.feature_selection import VarianceThreshold

# 创建示例数据
data = {
    'user_id': [1, 2, 3, 4, 5],
    'age': [25, 30, 35, 40, 45],
    'gender': [0, 1, 0, 1, 0],  # 0: 男, 1: 女
    'purchase_count': [10, 15, 12, 18, 20],
    'constant_feature': [1, 1, 1, 1, 1],  # 常数特征
    'low_variance': [0.1, 0.2, 0.1, 0.2, 0.1]  # 低方差特征
}

df = pd.DataFrame(data)

# 分离特征和标签
X = df.drop('user_id', axis=1)
y = df['purchase_count']

# 创建低方差过滤器（阈值设为0.1）
selector = VarianceThreshold(threshold=0.1)

# 应用过滤器
X_filtered = selector.fit_transform(X)

print("原特征数量:", X.shape[1])
print("过滤后特征数量:", X_filtered.shape[1])
print("保留的特征索引:", selector.get_support(indices=True))
print("各特征方差:", selector.variances_)