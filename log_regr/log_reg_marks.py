import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('marks.csv', header=None)
x = df.iloc[:, :-1]
y = df.iloc[:, -1]
admitted = df.loc[y == 1]
not_admitted = df.loc[y == 0]

# plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], label='Admitted')
# plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], label='Not Admitted')
# plt.show()

