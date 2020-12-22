import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)

df = pd.read_csv("automobile_data.csv")

models = ["toyota","nissan","mazda", "honda", "mitsubishi", "subaru", "volkswagen", "volvo"]
df = df[df.make.isin(models)]

sns.heatmap(pd.crosstab([df.make, df['num-doors']], [df['body-style'], df['drive-wheels']]),
            cmap="YlGnBu", annot=True, cbar=True)
plt.show()