import numpy as np

x = np.arange(-10, 10, .1)
y = x**3 + x**2 + x + 1

import matplotlib.pyplot as plt

noise = 100 * np.random.normal(size=len(x))
y = y + noise
plt.scatter(x, y)
plt.show()
