def step_function0(x):
    if x < 0:
        return 0
    else:
        return 1

import numpy as np
def step_function(x):
    return np.array(x > 0, dtype=int)

if __name__ == '__main__':
    x = np.array([-11, -2, 3, 4, 5])
    print(step_function(x))