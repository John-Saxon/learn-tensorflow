import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 100)
fig = plt.figure()


ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(x, x)
ax3 = fig.add_subplot(224)
ax3.plot(x, x**2)
plt.show()
