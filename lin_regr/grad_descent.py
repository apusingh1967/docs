import numpy as np
import matplotlib.pyplot as plt


def grad_desc(x, y):
    m_curr = b_curr = 0
    iters = 500000
    lr = 0.00001
    n = len(x)

    for i in range(iters):
        y_pred = b_curr + m_curr * x
        cost = (1 / n) * sum([val ** 2 for val in (y - y_pred)])
        md = (-2 / n) * sum(x * (y - y_pred))
        bd = (-2 / n) * sum(y - y_pred)
        m_curr = m_curr - lr * md
        b_curr = b_curr - lr * bd

    print("m {}, b {}, cost {}".format(m_curr, b_curr, cost))

    plt.style.use("seaborn-whitegrid")

    fig = plt.figure()
    ax = plt.axes()

    ax.plot(x, y)
    ax.plot(x, b_curr + m_curr * x)
    plt.show()


x = np.array([1, 2, 3, 4, 5])
y = np.array([5.2, 7.5, 8.4, 11.3, 12.4])

grad_desc(x, y)
