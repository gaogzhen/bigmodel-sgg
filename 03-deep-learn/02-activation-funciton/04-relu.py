import numpy as np
def relu(x):
    return np.maximum(0, x)

if __name__ == '__main__':
    x = np.array([-11, -9, -2, 1 ,3, 4, 5])
    print(relu(x))