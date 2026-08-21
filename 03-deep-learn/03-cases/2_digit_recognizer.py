import joblib
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

# 3. 预测分类概率（前向传播）
y_proba = forward(network, x_test)
print(y_proba)
print(y_proba.shape)

# 得到预测分类标签
y_pred = np.argmax(y_proba, axis=1)
print(y_pred)
print(y_pred.shape)

# 4. 计算准确率
# 统计预测准确个数
acc_cnt = np.sum(y_pred == y_test)
n = x_test.shape[0]
print("Accuracy: ", acc_cnt / n)