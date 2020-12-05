import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)

df = pd.read_csv('titanic.csv')
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Name_wiki', 'Age_wiki', 'Hometown', 'Destination', 'WikiId', 'Boarded', 'Lifeboat', 'Body', 'Class'],
        'columns', inplace=True)
df = df.dropna()
print(df.head(5))

# fig, ax = plt.subplots(figsize=(12, 8))
# plt.scatter(df['Fare'], df['Survived'])
# plt.xlabel('Fare')
# plt.ylabel('Survived')
# plt.show()

xtab = pd.crosstab(df['Embarked'], df['Survived'])
print(xtab)

lbl_enc = preprocessing.LabelEncoder()
df['Sex'] = lbl_enc.fit_transform(df['Sex'].astype(str))
df = pd.get_dummies(df, columns=['Embarked'])
print(df.head(3))
print(df.corr())
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv('titanic_processed.csv', index=False)
