import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix, roc_curve, auc
from scipy.stats import randint
import config
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================
# 1. 数据加载与预处理
# ============================================

# 加载数据集
data_path = os.path.join(Path(config.__file__).resolve().parent, 'data', 'heart_disease.csv')
data = pd.read_csv(data_path)

print("=" * 60)
print("📊 原始数据集信息")
print("=" * 60)
print(f"数据集形状: {data.shape}")
print(f"\n{data.info()}")
print(f"\n{data.describe()}")

# 处理缺失值
data.dropna(inplace=True)
print(f"\n处理缺失值后数据集形状: {data.shape}")

# 数据集划分
X = data.drop(columns=['是否患有心脏病'], axis=1)
y = data['是否患有心脏病']

print(f"\n特征数量: {X.shape[1]}")
print(f"样本数量: {X.shape[0]}")
print(f"\n目标变量分布:")
print(y.value_counts())
print(f"类别比例: {y.value_counts(normalize=True)}")

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
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
        ('binary', 'passthrough', binary_features),
    ]
)

## 执行特征转换
X_standardized = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_standardized, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

# ============================================
# 2. 早停策略：两阶段搜索（粗搜 + 精搜）
# ============================================

print("\n" + "=" * 60)
print("🔍 早停策略：两阶段网格搜索")
print("=" * 60)

# 定义多个评分指标
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1'
}

# ==================== 第一阶段：粗略搜索 ====================
print("\n【第一阶段】粗略搜索：大范围快速筛选")

param_grid_coarse = {
    'n_neighbors': list(range(3, 31, 2)),  # K值：3,5,7,...,29
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_search_coarse = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid_coarse,
    cv=5,
    scoring=scoring,
    refit='f1',
    return_train_score=True,
    n_jobs=-1,
    verbose=1
)

grid_search_coarse.fit(X_train, y_train)

print(f"\n第一阶段完成！")
print(f"最佳参数: {grid_search_coarse.best_params_}")
print(f"最佳F1分数: {grid_search_coarse.best_score_:.4f}")

# 获取第一阶段结果用于可视化
results_coarse = pd.DataFrame(grid_search_coarse.cv_results_)
best_k_coarse = grid_search_coarse.best_params_['n_neighbors']

# ==================== 第二阶段：精细搜索 ====================
print("\n【第二阶段】精细搜索：在最佳值附近深入优化")

# 在最佳K值附近进行精细搜索
k_fine_range = range(max(1, best_k_coarse - 2), best_k_coarse + 3)
param_grid_fine = {
    'n_neighbors': list(k_fine_range),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1, 2]  # 仅当metric='minkowski'时有效
}

grid_search_fine = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid_fine,
    cv=5,
    scoring=scoring,
    refit='f1',
    return_train_score=True,
    n_jobs=-1,
    verbose=1
)

grid_search_fine.fit(X_train, y_train)

print(f"\n第二阶段完成！")
print(f"最佳参数: {grid_search_fine.best_params_}")
print(f"最佳F1分数: {grid_search_fine.best_score_:.4f}")

# ============================================
# 3. 替代方案：RandomizedSearchCV（更快）
# ============================================

print("\n" + "=" * 60)
print("🎲 替代方案：随机搜索（更快的早停策略）")
print("=" * 60)

param_dist = {
    'n_neighbors': randint(3, 30),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1, 2]
}

random_search = RandomizedSearchCV(
    estimator=KNeighborsClassifier(),
    param_distributions=param_dist,
    n_iter=30,  # 只搜索30个随机组合
    cv=5,
    scoring='f1',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train, y_train)

print(f"\n随机搜索完成！")
print(f"最佳参数: {random_search.best_params_}")
print(f"最佳F1分数: {random_search.best_score_:.4f}")

# 选择最终模型（使用精细搜索的结果）
best_model = grid_search_fine.best_estimator_
print(f"\n✅ 最终选择的模型: {best_model}")

# ============================================
# 4. 数据预测
# ============================================

print("\n" + "=" * 60)
print("🎯 模型预测与评估")
print("=" * 60)

# 在测试集上进行预测
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # 正类概率

# 计算各项指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n测试集评估结果:")
print(f"准确率 (Accuracy):  {accuracy:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"F1分数 (F1):        {f1:.4f}")

print(f"\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['无心脏病', '有心脏病']))

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# ============================================
# 5. 数据可视化
# ============================================

# 创建可视化目录
output_dir = os.path.join(Path(config.__file__).resolve().parent, 'output')
os.makedirs(output_dir, exist_ok=True)

print("\n" + "=" * 60)
print("📈 数据可视化")
print("=" * 60)

# ==================== 可视化1：目标变量分布 ====================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.countplot(x=y, palette='Set2')
plt.title('原始数据集目标变量分布', fontsize=14)
plt.xlabel('是否患有心脏病', fontsize=12)
plt.ylabel('样本数量', fontsize=12)
plt.xticks([0, 1], ['无心脏病', '有心脏病'])

plt.subplot(1, 2, 2)
y_train_counts = pd.Series(y_train).value_counts()
y_test_counts = pd.Series(y_test).value_counts()
x_labels = ['训练集-无心脏病', '训练集-有心脏病', '测试集-无心脏病', '测试集-有心脏病']
heights = [y_train_counts[0], y_train_counts[1], y_test_counts[0], y_test_counts[1]]
colors = ['skyblue', 'lightcoral', 'skyblue', 'lightcoral']

plt.bar(x_labels, heights, color=colors, edgecolor='black', linewidth=1.5)
plt.title('训练集与测试集分布对比', fontsize=14)
plt.ylabel('样本数量', fontsize=12)
plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '01_目标变量分布.png'), dpi=300, bbox_inches='tight')
print("✅ 保存: 目标变量分布图")

# ==================== 可视化2：第一阶段K值搜索结果 ====================
plt.figure(figsize=(14, 6))

# 提取K值和对应的F1分数
k_values_coarse = results_coarse['param_n_neighbors'].astype(int)
f1_scores_coarse = results_coarse['mean_test_f1']

plt.subplot(1, 2, 1)
plt.plot(k_values_coarse, f1_scores_coarse, 'bo-', linewidth=2, markersize=8, alpha=0.7)
plt.axvline(x=best_k_coarse, color='r', linestyle='--', linewidth=2,
            label=f'最佳K={best_k_coarse}')
plt.xlabel('K值', fontsize=12)
plt.ylabel('F1分数', fontsize=12)
plt.title('第一阶段：K值与F1分数关系', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

# 显示所有可能的K值
plt.subplot(1, 2, 2)
unique_k = sorted(results_coarse['param_n_neighbors'].unique())
k_f1_means = [results_coarse[results_coarse['param_n_neighbors'] == k]['mean_test_f1'].mean()
              for k in unique_k]
plt.bar(unique_k, k_f1_means, color='skyblue', edgecolor='black', alpha=0.7)
plt.axhline(y=max(k_f1_means), color='r', linestyle='--', linewidth=2,
            label=f'最高F1={max(k_f1_means):.4f}')
plt.xlabel('K值', fontsize=12)
plt.ylabel('平均F1分数', fontsize=12)
plt.title('第一阶段：K值性能对比', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '02_第一阶段K值搜索结果.png'), dpi=300, bbox_inches='tight')
print("✅ 保存: 第一阶段K值搜索结果图")

# ==================== 可视化3：混淆矩阵 ====================
plt.figure(figsize=(10, 8))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['预测:无心脏病', '预测:有心脏病'],
            yticklabels=['真实:无心脏病', '真实:有心脏病'],
            annot_kws={"size": 14, "weight": "bold"})

plt.title('混淆矩阵', fontsize=16, fontweight='bold')
plt.xlabel('预测值', fontsize=14)
plt.ylabel('真实值', fontsize=14)
plt.xticks(rotation=0)
plt.yticks(rotation=0)

# 添加性能指标文本
plt.text(0.5, -0.3,
         f'准确率: {accuracy:.4f}\n精确率: {precision:.4f}\n召回率: {recall:.4f}\nF1分数: {f1:.4f}',
         ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '03_混淆矩阵.png'), dpi=300, bbox_inches='tight')
print("✅ 保存: 混淆矩阵图")

# ==================== 可视化4：ROC曲线与AUC ====================
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))

plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率 (FPR)', fontsize=12)
plt.ylabel('真阳性率 (TPR)', fontsize=12)
plt.title('ROC曲线', fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '04_ROC曲线.png'), dpi=300, bbox_inches='tight')
print("✅ 保存: ROC曲线图")

# ==================== 可视化5：不同K值的性能对比 ====================
# 使用交叉验证评估不同K值的性能
k_range = range(1, 31)
cv_scores = []
cv_std = []

for k in k_range:
    # 从最佳参数中复制，然后更新K值
    params = grid_search_fine.best_params_.copy()
    params['n_neighbors'] = k
    knn = KNeighborsClassifier(**params)

    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='f1')
    cv_scores.append(scores.mean())
    cv_std.append(scores.std())

plt.figure(figsize=(12, 6))

plt.errorbar(k_range, cv_scores, yerr=cv_std, fmt='o-', capsize=5,
             linewidth=2, markersize=8, label='F1分数 ± 标准差', color='blue', alpha=0.7)

best_k_final = k_range[np.argmax(cv_scores)]
plt.axvline(x=best_k_final, color='r', linestyle='--', linewidth=2,
            label=f'最佳K={best_k_final}\nF1={max(cv_scores):.4f}')

plt.xlabel('K值', fontsize=12)
plt.ylabel('交叉验证F1分数', fontsize=12)
plt.title('不同K值的交叉验证性能', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '05_K值性能对比.png'), dpi=300, bbox_inches='tight')
print("✅ 保存: K值性能对比图")

# ==================== 可视化6：预测概率分布 ====================
plt.figure(figsize=(12, 6))

plt.hist(y_pred_proba[y_test == 0], bins=20, alpha=0.7, label='无心脏病', color='skyblue', edgecolor='black')
plt.hist(y_pred_proba[y_test == 1], bins=20, alpha=0.7, label='有心脏病', color='lightcoral', edgecolor='black')

plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='分类阈值 (0.5)')
plt.xlabel('预测概率', fontsize=12)
plt.ylabel('样本数量', fontsize=12)
plt.title('预测概率分布', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '06_预测概率分布.png'), dpi=300, bbox_inches='tight')
print("✅ 保存: 预测概率分布图")

# ==================== 可视化7：特征重要性（基于距离权重） ====================
# KNN没有直接的特征重要性，但我们可以通过分析来近似
# 这里我们使用一个简单的代理：计算每个特征的标准差（标准化后的特征）
# 标准差大的特征对距离计算影响更大

# 获取特征名称
feature_names = []
# 数值特征
feature_names.extend(numerical_features)
# 独热编码后的类别特征
ohe = preprocessor.named_transformers_['cat']
if hasattr(ohe, 'get_feature_names_out'):
    cat_feature_names = ohe.get_feature_names_out(categorical_features)
else:
    cat_feature_names = [f"{cat}_{i}" for cat in categorical_features for i in range(len(data[cat].unique()) - 1)]
feature_names.extend(cat_feature_names)
# 二分类特征
feature_names.extend(binary_features)

# 计算标准化后特征的标准差（作为重要性代理）
feature_std = np.std(X_standardized, axis=0)

# 创建特征重要性DataFrame
feature_importance = pd.DataFrame({
    '特征': feature_names,
    '标准差': feature_std
})
feature_importance = feature_importance.sort_values('标准差', ascending=False).head(15)  # 取前15个

plt.figure(figsize=(14, 8))

colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(feature_importance)))
bars = plt.barh(range(len(feature_importance)), feature_importance['标准差'],
                color=colors, edgecolor='black', linewidth=1.5)

plt.yticks(range(len(feature_importance)), feature_importance['特征'], fontsize=11)
plt.xlabel('标准差（特征重要性代理）', fontsize=12)
plt.title('特征重要性排名（前15）', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()  # 重要性从高到低

# 添加数值标签
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
             f'{width:.3f}', ha='left', va='center', fontsize=10)

plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '07_特征重要性.png'), dpi=300, bbox_inches='tight')
print("✅ 保存: 特征重要性图")

# ==================== 可视化8：多指标对比 ====================
# 提取最终模型在不同指标下的表现
metrics_data = {
    '指标': ['准确率', '精确率', '召回率', 'F1分数'],
    '分数': [accuracy, precision, recall, f1]
}

plt.figure(figsize=(10, 6))

colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
bars = plt.bar(metrics_data['指标'], metrics_data['分数'],
               color=colors, edgecolor='black', linewidth=1.5)

plt.ylim(0, 1.05)
plt.ylabel('分数', fontsize=12)
plt.title('模型性能多指标对比', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
             f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '08_多指标对比.png'), dpi=300, bbox_inches='tight')
print("✅ 保存: 多指标对比图")

# ============================================
# 6. 保存预测结果
# ============================================

# 创建预测结果DataFrame
predictions_df = pd.DataFrame({
    '真实标签': y_test.values,
    '预测标签': y_pred,
    '预测概率_有心脏病': y_pred_proba,
    '预测正确': (y_test.values == y_pred)
})

# 保存预测结果
predictions_path = os.path.join(output_dir, '预测结果.csv')
predictions_df.to_csv(predictions_path, index=False, encoding='utf-8-sig')
print(f"\n✅ 保存: 预测结果到 {predictions_path}")

# ============================================
# 7. 生成总结报告
# ============================================

report_path = os.path.join(output_dir, '模型评估报告.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("KNN心脏病预测模型评估报告\n")
    f.write("=" * 60 + "\n\n")

    f.write("【数据集信息】\n")
    f.write(f"总样本数: {len(data)}\n")
    f.write(f"特征数: {X.shape[1]}\n")
    f.write(f"训练集: {len(X_train)} 样本\n")
    f.write(f"测试集: {len(X_test)} 样本\n\n")

    f.write("【模型配置】\n")
    f.write(f"算法: KNN (K-Nearest Neighbors)\n")
    f.write(f"最佳参数: {grid_search_fine.best_params_}\n")
    f.write(f"交叉验证: 5折\n")
    f.write(f"评估指标: F1分数 (主), 准确率, 精确率, 召回率\n\n")

    f.write("【测试集性能】\n")
    f.write(f"准确率 (Accuracy):  {accuracy:.4f}\n")
    f.write(f"精确率 (Precision): {precision:.4f}\n")
    f.write(f"召回率 (Recall):    {recall:.4f}\n")
    f.write(f"F1分数 (F1):        {f1:.4f}\n")
    f.write(f"AUC:                {roc_auc:.4f}\n\n")

    f.write("【混淆矩阵】\n")
    f.write(f"真正例 (TP): {cm[1, 1]}\n")
    f.write(f"假正例 (FP): {cm[0, 1]}\n")
    f.write(f"真反例 (TN): {cm[0, 0]}\n")
    f.write(f"假反例 (FN): {cm[1, 0]}\n\n")

    f.write("【早停策略】\n")
    f.write("采用两阶段网格搜索:\n")
    f.write("  1. 第一阶段: 粗略搜索 (K=3-29, 步长=2)\n")
    f.write("  2. 第二阶段: 精细搜索 (最佳K值±2范围内)\n")
    f.write(f"第一阶段最佳K: {best_k_coarse}\n")
    f.write(f"第二阶段最佳K: {grid_search_fine.best_params_['n_neighbors']}\n\n")

    f.write("【可视化文件】\n")
    f.write("已生成以下可视化图表:\n")
    f.write("  01_目标变量分布.png\n")
    f.write("  02_第一阶段K值搜索结果.png\n")
    f.write("  03_混淆矩阵.png\n")
    f.write("  04_ROC曲线.png\n")
    f.write("  05_K值性能对比.png\n")
    f.write("  06_预测概率分布.png\n")
    f.write("  07_特征重要性.png\n")
    f.write("  08_多指标对比.png\n\n")

    f.write("【预测结果】\n")
    f.write(f"预测结果已保存至: 预测结果.csv\n")
    f.write(f"总预测数: {len(predictions_df)}\n")
    f.write(f"正确预测: {predictions_df['预测正确'].sum()} ({predictions_df['预测正确'].mean() * 100:.2f}%)\n")
    f.write(f"错误预测: {(~predictions_df['预测正确']).sum()} ({(~predictions_df['预测正确']).mean() * 100:.2f}%)\n\n")

    f.write("=" * 60 + "\n")
    f.write("报告生成时间: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
    f.write("=" * 60 + "\n")

print(f"\n✅ 保存: 模型评估报告到 {report_path}")

# ============================================
# 8. 打印最终总结
# ============================================

print("\n" + "=" * 60)
print("🎉 模型训练与评估完成！")
print("=" * 60)
print(f"\n📊 最终模型性能:")
print(f"   准确率:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"   精确率:  {precision:.4f} ({precision * 100:.2f}%)")
print(f"   召回率:  {recall:.4f} ({recall * 100:.2f}%)")
print(f"   F1分数:  {f1:.4f} ({f1 * 100:.2f}%)")
print(f"   AUC:     {roc_auc:.4f} ({roc_auc * 100:.2f}%)")
print(f"\n⚙️  最佳超参数:")
for key, value in grid_search_fine.best_params_.items():
    print(f"   {key}: {value}")
print(f"\n📁 输出文件:")
print(f"   可视化图表: {output_dir}/")
print(f"   预测结果: {predictions_path}")
print(f"   评估报告: {report_path}")
print("\n" + "=" * 60)

# 显示所有图表
plt.show()