import numpy as np
from   common.functions import sigmoid, identity

# 初始化网络
def init_network():
    network = {}
    # 隐藏层第一层参数
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    # 隐藏层第二层参数
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    # 输出层
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network

# 前向传播
def forward(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = x @ w1 + b1
    z1 = sigmoid(a1)
    a2 = z1 @ w2 + b2
    z2 = sigmoid(a2)
    a3 = z2 @ w3 + b3
    y = identity(a3)
    return y

# 主流程
# 1. 定义数据
x = np.array([1.0, 0.5])

# 2. 定义神经网络
network = init_network()

# 3. 预测：前向传播
y = forward(network, x)

print(y)

