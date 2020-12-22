import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax1 = fig.add_axes([0., 0.5, 1., 0.5],
                   xticklabels=[], ylim=(-1.2, 1.2))
ax2 = fig.add_axes([0., 0., 1., 0.5],
                   ylim=(-1.2, 1.2))

x = np.linspace(0, 10, 100)

ax1.plot(np.sin(x))
ax2.plot(np.cos(x))

plt.show()
