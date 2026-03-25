
import numpy as np

test_1 = np.array([1, 2, 3])
test_2 = np.array([2, 1, 4])
test_3 = np.array([0, 0, 0])

print(np.where(test_1 > test_2, test_2, test_3))