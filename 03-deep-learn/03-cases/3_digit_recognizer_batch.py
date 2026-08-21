import joblib
import pandas as pd
from sklearn.model_selection import train_test_split  # 划分数据集
from sklearn.preprocessing import MinMaxScaler  # 归一化

from common.functions import *
from common.utils.data_import import data_import, data_absolute_path


# 读取数据
def get_data():
    # 1. 加载数据
    data = data_import('data/train.csv')

    # 2. 划分数据集
    X = data.drop(columns='label', axis=1)
    y = data['label']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 3. 特征转换：归一化
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_test, y_test

# 初始化神经网络
def init_network():
    # 从文件中直接加载训练好的模型
    network = joblib.load(data_absolute_path('data/nn_sample'))
    return network

# 前向传播（预测）
def forward(network, X):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = X @ w1 + b1
    z1 = sigmoid(a1)
    a2 = z1 @ w2 + b2
    z2 = sigmoid(a2)
    a3 = z2 @ w3 + b3
    y = softmax(a3)
    return y

# 主流程
# 1. 获取测试数据
x_test, y_test = get_data()
print(x_test.shape)
print(y_test.shape)

# 2. 创建模型
network = init_network()
print(network['W1'].shape)
print(network['W2'].shape)
print(network['W3'].shape)

# 定义一些变量
batch_size = 100
n = x_test.shape[0]
acc_cnt = 0

# 3. 循环迭代：分批预测，前向传播
for i in range( 0, n, batch_size):
    # 3.1 取出当前批次的测试数据
    x_batch = x_test[i:i+batch_size]
    y_batch = y_test[i:i+batch_size]

    # 3.2 前向传播，预测分类概率
    y_proba = forward(network, x_batch)

    # 3.3 得到预测分类标签
    y_pred = np.argmax(y_proba, axis=1)

    # 3.4 累计预测准确个数
    acc_cnt += np.sum(y_pred == y_batch)

print("Accuracy: ", acc_cnt / n)