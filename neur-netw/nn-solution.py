import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

mat = loadmat('nn-solution-data.mat')
X = mat['X']
y = mat['y']

# import matplotlib.image as mpimg


def sample():
    fig, axis = plt.subplots(10, 10, figsize=(8, 8))
    for i in range(10):
        for j in range(10):
            axis[i, j] \
                .imshow(X[np.random.randint(0, 5001), :]
                        .reshape(20, 20, order='F'), cmap='magma')
            axis[i, j].axis('off')

    plt.show()


# sample()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def log_reg_cost_function(theta, X, y, lmbda):
    m = len(y)
    z = X @ theta
    preds = sigmoid(z)
    error = (-y * np.log(preds)) - ((1 - y) * np.log(1 - preds))
    cost = 1/m * sum(error)
    regularization_err = lmbda/(2*m) * sum(theta[1:]**2)
    regCost = cost + regularization_err
    j_0 = 1/m * (X.transpose() @ (preds - y))[0]
    j_1 = 1/m * (X.transpose() @ (preds - y))[1:] + (lmbda/m) * theta[1:]
    grad = np.vstack((j_0[:, np.newaxis], j_1))

    return regCost[0], grad


theta_t = np.array([-2, -1, 1, 2]).reshape(4, 1)
X_t = np.array([np.linspace(0.1, 1.5, 15)]).reshape(3, 5).T
X_t = np.hstack((np.ones((5, 1)), X_t))
y_t = np.array([1, 0, 1, 0, 1]).reshape(5, 1)
lambda_t = 3
J, grad_t = log_reg_cost_function(theta_t, X_t, y_t, lambda_t)
print(J)
print(grad_t)

