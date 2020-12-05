import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from log_regr import titanic_helper


df = pd.read_csv('titanic_processed.csv')
fn = LogisticRegression(solver='liblinear', penalty='l2', C=10)
y_col = 'Survived'
x_col = list(df.columns[1:])
results = titanic_helper.build_model(fn, y_col, x_col, df)
titanic_helper.print_results(results)

