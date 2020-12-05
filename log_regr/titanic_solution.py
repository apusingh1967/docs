import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)

df = pd.read_csv('titanic_processed.csv')
print(df.head(3))
print(df.shape)

x = df.drop(['Survived'], axis=1)
y = df['Survived']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear').fit(x_train, y_train)

y_pred = model.predict(x_test)
results = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Precision: ', precision_score(y_test, y_pred))
print('Recall: ', recall_score(y_test, y_pred))
