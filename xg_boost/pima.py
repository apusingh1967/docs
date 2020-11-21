import pandas as pd
from numpy import loadtxt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
print(pd)
data = loadtxt('pima_indians_dataset', delimiter=',')
X = data[:, 0:8]
y = data[:, 8]

model = XGBClassifier()
kfold = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print(results)
print(results.mean())
print(results.std())
