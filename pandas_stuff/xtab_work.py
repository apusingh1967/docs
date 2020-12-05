import pandas as pd
import seaborn as sns


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)

df = pd.read_csv("Automobile_data.csv")
print(df['engine-type'].unique())
print(df.head(5))
