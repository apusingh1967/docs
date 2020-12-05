import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score

data = pd.read_csv("../FuelConsumptionCo2.csv")
data = data[["ENGINESIZE", "CYLINDERS", "CO2EMISSIONS"]]

train = data[: (int(len(data) * 0.8))]
test = data[(int(len(data) * 0.8)) :]

train_x = np.array(train[["ENGINESIZE", "CYLINDERS"]])
train_y = np.array(train[["CO2EMISSIONS"]])

model = linear_model.LinearRegression()
model.fit(train_x, train_y)
print(model.coef_, model.intercept_)

plt.scatter(data.iloc[:, 0], data.iloc[:, 1], color="blue")
plt.plot(train_x, model.intercept_ + model.coef_ * train_x, "-r")
plt.xlabel("engine size")
plt.ylabel("co2 emission")
# plt.show()

test_x = np.array(test[["ENGINESIZE", "CYLINDERS"]])
test_y = np.array(test[["CO2EMISSIONS"]])
test_y_pred = model.predict(test_x)

print("Mean: %.2f" % np.mean(test_y - test_y_pred))
print("R2Score: %.2f" % r2_score(test_y_pred, test_y))
