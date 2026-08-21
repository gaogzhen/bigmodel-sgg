# 一、激活函数

# 阶跃函数
def step_function0(x):
    if x > 0:
        return 1
    else:
        return 0

import numpy as np
def step_function(x):
    return np.array(x > 0, dtype=int)

# Sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))

# ReLU
def relu(x):
    return np.maximum(0, x)

# Softmax
def softmax0(x):
    return np.exp(x) / np.sum(np.exp(x))

# 数据溢出的对策
def softmax1(x):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

# 考虑二维矩阵的输入
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

# 恒等函数
def identity(x):
    return x

# 二、损失函数
# 均方误差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

# 交叉熵误差
def cross_entropy_error(y, t):
    # 对于一条数据的情况，转换为二维结果
    if y.ndim == 1:
        t = t.reshape(1, -1)
        y = y.reshape(1, -1)
    # 如果t是独热编码，就转换为正确的解标签
    if t.size == y.size:
        t = np.argmax(t, axis=1)
    n = y.shape[0]
    return -np.sum( np.log(y[np.arange(n), t] + 1e-7) ) / n

if __name__ == '__main__':
    x = np.array([1000, 900, 2, 3, 4, 5, -1, -2, -3, -4, -5])
    x2 = np.array([[1, 2, 3], [-4, 5, -6], [-1, -2, -3], [3, -4, -5]])
    print(step_function(x))
    print(sigmoid(x))
    print(np.tanh(x))
    print(relu(x))
    print(x2)
    print(softmax(x2))
    print(identity(x2))