import numpy as np

M = np.array([[4, 1],
              [2, 3]])

v = np.array([3, 2])

Mv = M @ v
result = v @ Mv
print("M @ v:\n", Mv)
print("v' @ M @ v:",result)