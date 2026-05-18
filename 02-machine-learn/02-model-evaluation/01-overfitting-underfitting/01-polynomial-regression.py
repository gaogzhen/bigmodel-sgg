import numpy as np
import matplotlib.pyplot as plt
from  sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 生成数据
np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1) * 0.8

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建不同复杂的的模型
degrees = [1, 2, 10]
colors = ['blue', 'red', 'green']
titles = ['欠拟合（一次多项式）', '正常拟合（二次多项式）', '过拟合（十次多项式）']

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

plt.figure(figsize=(15, 5))

for i, degree in enumerate(degrees):
    # 创建多项式特征
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    x_poly_train = poly_features.fit_transform(X_train)
    x_poly_test = poly_features.transform(X_test)

    # 训练模型
    model = LinearRegression()
    model.fit(x_poly_train, y_train)

    # 预测
    X_poly_full = poly_features.transform(X)
    y_pred_full = model.predict(X_poly_full)
    y_pred_train = model.predict(x_poly_train)
    y_pred_test = model.predict(x_poly_test)

    # 计算误差
    train_error = mean_squared_error(y_train, y_pred_train)
    test_error = mean_squared_error(y_test, y_pred_test)

    # 可视化
    plt.subplot(1, 3, i + 1)
    plt.scatter(X_train, y_train, color='black', label='训练数据')
    plt.scatter(X_test, y_test, color='orange', label='测试数据')
    plt.plot(X, y_pred_full, color=colors[i], linewidth=2, label=f'预测曲线')
    plt.title(f'{titles[i]}\nTrain MSE: {train_error:.2f}\nTest MSE: {test_error:.2f}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
