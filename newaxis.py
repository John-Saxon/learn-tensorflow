import numpy as np

x1 = np.arange(6)

# print(x1)

# print(x1[:, np.newaxis])
# print(x1[np.newaxis, :])


b = x1.reshape((2, -1))
print(b)

print(b[:, np.newaxis])
print(b[np.newaxis, :])
print(b[:, np.newaxis, :])
print(b[np.newaxis, :, :])
print(b[:, :, np.newaxis])
