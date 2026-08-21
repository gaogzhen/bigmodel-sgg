import numpy as np
def softmax1(x):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

# 考虑二维矩阵
def softmax(x):
    # 二维数据
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    # 向量或者标量
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

if __name__ == '__main__':
    x = np.array([-11, -9, -2, 1 ,3, 4, 500])
    x2 = np.array([[-5, -4, -3], [2,9, 9], [-100, 0 , 100]])
    print(softmax(x2))