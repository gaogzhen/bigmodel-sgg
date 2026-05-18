import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 加载房价数据
# df = pd.read_csv('housing.csv')

# 假设数据
np.random.seed(42)
df = pd.DataFrame({
    'area': np.random.randint(50, 200, 100),
    'bedrooms': np.random.randint(1, 5, 100),
    'age': np.random.randint(0, 50, 100),
    'distance_to_city': np.random.randint(1, 20, 100),
    'price': np.random.randint(100000, 500000, 100)
})

# 添加一些相关性
df['price'] = df['area'] * 2000 + df['bedrooms'] * 10000 - df['age'] * 500

# 计算相关系数
correlations = []
for feature in df.columns[:-1]:
    corr, p_value = pearsonr(df[feature], df['price'])
    correlations.append({
        'feature': feature,
        'correlation': corr,
        'p_value': p_value
    })

corr_df = pd.DataFrame(correlations)
print("房价预测特征相关性分析：")
print(corr_df.sort_values('correlation', ascending=False))

# 可视化
plt.figure(figsize=(12, 5))

# 相关性条形图
plt.subplot(1, 2, 1)
plt.barh(corr_df['feature'], corr_df['correlation'])
plt.xlabel('相关系数')
plt.title('特征与房价的相关性')
plt.grid(True, alpha=0.3)

# 散点图矩阵
plt.subplot(1, 2, 2)
sns.scatterplot(data=df, x='area', y='price', alpha=0.6)
plt.xlabel('面积')
plt.ylabel('房价')
plt.title('面积与房价的关系')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()