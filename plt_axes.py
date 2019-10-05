import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 100)

fig, axes = plt.subplots(2, 2)
ax1 = axes[0, 0]
ax2 = axes[0, 1]
ax3 = axes[1, 0]
ax4 = axes[1, 1]

ax1.plot(x, x)
ax2.plot(x, -x)
ax3.plot(x, x**2)
ax4.plot(x, np.log(x))

plt.show()
