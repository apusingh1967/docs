import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

data = pd.read_csv('titanic-train.csv')[['Pclass', 'Sex', 'Age', 'Survived']]
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data = data.dropna()
X = data.drop('Survived', axis=1)
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)

model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
pred = [round(value) for value in y_pred]

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, pred)
print(accuracy)

