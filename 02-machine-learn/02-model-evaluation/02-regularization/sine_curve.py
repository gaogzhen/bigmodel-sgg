"""
欠拟合、过拟合与正则化 —— 完整演示代码
==========================================
本脚本包含：
1. 生成带噪声的非线性数据（正弦曲线）
2. 对比一阶（欠拟合）、三阶（良好）、15阶（过拟合）多项式回归
3. 展示 L2 正则化（Ridge）与 L1 正则化（Lasso）如何抑制过拟合
4. 绘制学习曲线，分析偏差-方差权衡
5. 查看多项式系数，理解 L1 的稀疏性
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# 设置中文字体（如果系统支持）及随机种子
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
np.random.seed(42)

# ==================== 1. 生成模拟数据 ====================
X = np.linspace(0, 1, 50).reshape(-1, 1)               # 50 个样本点，一维特征
true_signal = np.sin(2 * np.pi * X).ravel()            # 真实函数：sin(2πx)
noise = np.random.normal(0, 0.2, size=X.shape[0])      # 高斯噪声，标准差 0.2
y = true_signal + noise                                # 带噪声的目标值

# 分割训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# 用于绘制平滑曲线的密集点
X_smooth = np.linspace(0, 1, 200).reshape(-1, 1)

# ==================== 2. 欠拟合、良好拟合、过拟合对比 ====================
print("=" * 50)
print("第一部分：不同复杂度模型的拟合效果")
print("=" * 50)

# 定义三个模型
models_complexity = {
    "欠拟合 (线性, 1阶)": LinearRegression(),
    "良好拟合 (3阶多项式)": make_pipeline(PolynomialFeatures(3), LinearRegression()),
    "过拟合 (15阶多项式)": make_pipeline(PolynomialFeatures(15), LinearRegression())
}

plt.figure(figsize=(15, 4))
for idx, (name, model) in enumerate(models_complexity.items(), 1):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    plt.subplot(1, 3, idx)
    plt.scatter(X_train, y_train, color='blue', s=20, label='训练集')
    plt.scatter(X_test, y_test, color='green', s=20, label='测试集')
    plt.plot(X_smooth, model.predict(X_smooth), color='red', lw=2, label='拟合曲线')
    plt.title(f"{name}\n训练MSE: {train_mse:.3f} | 测试MSE: {test_mse:.3f}")
    plt.ylim(-2, 2)
    plt.legend()
plt.tight_layout()
plt.show()

# ==================== 3. 正则化对抗过拟合 ====================
print("\n" + "=" * 50)
print("第二部分：正则化如何抑制过拟合")
print("=" * 50)

# 固定使用容易过拟合的 15 阶多项式，对比不同正则化手段
degree = 15
models_reg = {
    "无正则化 (过拟合)": make_pipeline(
        PolynomialFeatures(degree), StandardScaler(), LinearRegression()
    ),
    "L2 正则化 (Ridge α=0.01)": make_pipeline(
        PolynomialFeatures(degree), StandardScaler(), Ridge(alpha=0.01)
    ),
    "L2 正则化 (Ridge α=1.0)": make_pipeline(
        PolynomialFeatures(degree), StandardScaler(), Ridge(alpha=1.0)
    ),
    "L1 正则化 (Lasso α=0.001)": make_pipeline(
        PolynomialFeatures(degree), StandardScaler(), Lasso(alpha=0.001, max_iter=10000)
    )
}

plt.figure(figsize=(16, 10))
for idx, (name, model) in enumerate(models_reg.items(), 1):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    plt.subplot(2, 2, idx)
    plt.scatter(X_train, y_train, color='blue', s=20, label='训练集')
    plt.scatter(X_test, y_test, color='green', s=20, label='测试集')
    plt.plot(X_smooth, model.predict(X_smooth), color='red', lw=2, label='拟合曲线')
    plt.title(f"{name}\n训练MSE: {train_mse:.3f} | 测试MSE: {test_mse:.3f}")
    plt.ylim(-2, 2)
    plt.legend()
plt.tight_layout()
plt.show()

# ==================== 4. 学习曲线分析 ====================
print("\n" + "=" * 50)
print("第三部分：学习曲线 —— 偏差与方差诊断")
print("=" * 50)

def plot_learning_curve(estimator, title, X, y):
    """绘制单一模型的学习曲线"""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5,
        scoring='neg_mean_squared_error',
        train_sizes=np.linspace(0.3, 1.0, 10),
        random_state=1
    )
    train_errors = -train_scores.mean(axis=1)
    test_errors = -test_scores.mean(axis=1)

    plt.plot(train_sizes, train_errors, 'o-', color='blue', label='训练误差')
    plt.plot(train_sizes, test_errors, 'o-', color='red', label='验证误差')
    plt.title(title)
    plt.xlabel('训练样本数')
    plt.ylabel('均方误差 (MSE)')
    plt.legend()
    plt.grid(True)

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plot_learning_curve(LinearRegression(), "欠拟合 (线性回归)", X, y)

plt.subplot(1, 3, 2)
plot_learning_curve(
    make_pipeline(PolynomialFeatures(3), LinearRegression()),
    "良好拟合 (3阶多项式)", X, y
)

plt.subplot(1, 3, 3)
plot_learning_curve(
    make_pipeline(PolynomialFeatures(15), LinearRegression()),
    "过拟合 (15阶多项式)", X, y
)
plt.tight_layout()
plt.show()

# ==================== 5. 系数稀疏性对比 (L1 vs L2) ====================
print("\n" + "=" * 50)
print("第四部分：L1 与 L2 正则化的系数稀疏性对比")
print("=" * 50)

# 手动提取特征并标准化，便于查看系数
poly = PolynomialFeatures(degree, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
feature_names = poly.get_feature_names_out(['x'])

# 训练三个模型（均在标准化后的数据上）
lr = LinearRegression().fit(X_train_scaled, y_train)
ridge = Ridge(alpha=1.0).fit(X_train_scaled, y_train)
lasso = Lasso(alpha=0.001, max_iter=10000).fit(X_train_scaled, y_train)

# 绘制系数数值
plt.figure(figsize=(14, 5))
plt.plot(lr.coef_, 'o-', label='无正则化', markersize=4)
plt.plot(ridge.coef_, 's-', label='Ridge (L2)', markersize=4)
plt.plot(lasso.coef_, '^-', label='Lasso (L1)', markersize=4)
plt.axhline(0, color='black', linewidth=0.5)
plt.xlabel('特征索引（多项式项）')
plt.ylabel('系数数值')
plt.title('不同正则化下的多项式系数对比')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 打印 Lasso 非零系数统计
non_zero = np.sum(lasso.coef_ != 0)
print(f"Lasso 非零系数个数: {non_zero} / {len(lasso.coef_)}")
print("解释：L1 正则化将大部分系数压缩为 0，实现了自动特征选择。")

# ==================== 6. 正则化强度 α 的影响 ====================
print("\n" + "=" * 50)
print("第五部分：正则化强度 α 的影响")
print("=" * 50)

alphas = np.logspace(-4, 2, 50)  # α 从 10^-4 到 10^2
train_errors, test_errors = [], []

for alpha in alphas:
    model = make_pipeline(
        PolynomialFeatures(degree), StandardScaler(),
        Ridge(alpha=alpha)
    )
    model.fit(X_train, y_train)
    train_errors.append(mean_squared_error(y_train, model.predict(X_train)))
    test_errors.append(mean_squared_error(y_test, model.predict(X_test)))

plt.figure(figsize=(10, 5))
plt.semilogx(alphas, train_errors, 'b-o', label='训练误差')
plt.semilogx(alphas, test_errors, 'r-s', label='测试误差')
plt.xlabel('正则化强度 α')
plt.ylabel('均方误差 (MSE)')
plt.title('Ridge 回归：α 对训练/测试误差的影响')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 找到最佳 α（根据测试误差最小）
best_alpha = alphas[np.argmin(test_errors)]
print(f"最佳 α (基于测试集): {best_alpha:.4f}，对应测试 MSE: {min(test_errors):.4f}")

print("\n演示完成。")