import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)

df = pd.read_excel('./sales.xlsx')
df['Status'] = df['Status'].astype('category')
df['Status'].cat.set_categories(['won', 'pending', 'presented', 'declined'], inplace=True)
print(pd.pivot_table(df, index=['Manager', 'Status'], values=['Price', 'Quantity'], aggfunc=[np.sum], columns=['Product'], fill_value=0, margins=True))
