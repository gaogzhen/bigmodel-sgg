import os
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import config

# 加载数据集
data_path = os.path.join(Path(config.__file__).resolve().parent, 'data','heart_disease.csv')
data = pd.read_csv(data_path)
# print(data.info())

# 处理缺失值
data.dropna(inplace=True)

# 数据集划分
X = data.drop(columns=['是否患有心脏病'], axis=1)
y = data['是否患有心脏病']

# 特征工程：特征转换
## 数值型特征
numerical_features = ['年龄', '静息血压', '胆固醇', '最大心率', '运动后的ST下降', '主血管数量']
## 类别特征
categorical_features = ['胸痛类型', '静息心电图结果', '峰值ST段的斜率', '地中海贫血']
## 二分类特征
binary_features = ['性别', '空腹血糖', '运动性心绞痛']

## 创建转换器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features),
        ('binary', 'passthrough', binary_features),
    ]
)
## 执行特征转换
X_standardized = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.2, random_state=42, stratify=y)

# 定义多个评分指标
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1'
}

# 多指标网格搜索
grid_search_multi = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid={'n_neighbors': list(range(3, 21, 2))},
    cv=5,
    scoring=scoring,
    refit='f1',  # 以F1分数作为最终选择标准
    return_train_score=True,
    n_jobs=-1
)

grid_search_multi.fit(X_train, y_train)

print("\n多指标评估结果:")
print(f"最佳参数: {grid_search_multi.best_params_}")
print(f"最佳F1分数: {grid_search_multi.best_score_:.4f}")

# 查看不同指标下的最佳参数
for metric in scoring.keys():
    best_idx = grid_search_multi.cv_results_[f'rank_test_{metric}'].argmin()
    print(f"\n按{metric}排序的最佳参数:")
    print(f"  参数: {grid_search_multi.cv_results_['params'][best_idx]}")
    print(f"  {metric}分数: {grid_search_multi.cv_results_[f'mean_test_{metric}'][best_idx]:.4f}")