# TODO: test the model here
# TODO: pick one of the episode from the

import numpy as np

A = np.array([[1, 2, 3],
              [1, 3, 5],
              [2, 4, 6],
              [1, 2, 3]])
B = np.array([[100, 200, 300],
              [100, 200, 300],
              [100, 200, 300],
              [100, 100, 100]])

print("A=", A.shape)
print("B=", B.shape)
print("B*A=", B * A)