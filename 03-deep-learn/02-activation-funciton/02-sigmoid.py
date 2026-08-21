import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))

if __name__ == '__main__':
    x = np.array([-11, -9, -2, 1 ,3, 4, 5])
    print(sigmoid(x))