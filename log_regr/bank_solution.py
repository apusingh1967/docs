import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns


sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)

X_cont = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'euribor3m', 'nr.employed']
X_cat = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']

df = pd.read_csv('bank-additional-full.csv', sep=';')

df.loc[df['education'] == 'basic.4y', ['education']] = 'Basic'
df.loc[df['education'] == 'basic.6y', ['education']] = 'Basic'
df.loc[df['education'] == 'basic.9y', ['education']] = 'Basic'

cont_df = df[X_cont]
cat_df = df[X_cat]

X = cont_df.join(pd.get_dummies(cat_df))
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

log_mdl = LogisticRegression(random_state=0)
clf = log_mdl.fit(X_train, y_train)
score = clf.score(X_test, y_test)

print(score)


