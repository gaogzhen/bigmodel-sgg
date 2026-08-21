import numpy as np
def tanh(x):
    return np.tanh(x)

if __name__ == '__main__':
    x = np.array([-11, -9, -2, 1 ,3, 4, 5])
    print(tanh(x))