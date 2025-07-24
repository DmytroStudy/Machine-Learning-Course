import copy, math
import numpy as np
import matplotlib.pyplot as plt
from utils.lab_utils_common import  dlc, plot_data, plt_tumor_data, sigmoid, compute_cost_logistic
from utils.plt_quad_logistic import plt_quad_logistic, plt_prob
from unsupervised_learning.cost_function import logistic_cost
plt.style.use('../utils/deeplearning.mplstyle')

# Compute one gradient descent step
def compute_gradient(x_train, y_train, w_train, b, alpha):
    m, n = x_train.shape
    w_new = np.copy(w_train)
    cost = 0

    # Update ws
    for j in range(n):
        cost = 0
        for i in range(m):
            z = np.dot(x_train[i], w_train) + b
            f_wb = 1 / (1 + np.exp(-z))
            cost += (f_wb - y_train[i]) * x_train[i, j]

        cost = cost / m
        w_new[j] = w_train[j] - alpha * cost

    cost = 0

    # Update b
    for i in range(m):
        z = np.dot(x_train[i], w_train) + b
        f_wb = 1 / (1 + np.exp(-z))
        cost += (f_wb - y_train[i])

    cost = cost / m
    b_new = b - alpha * cost

    return w_new, b_new

# Compute all gradient descent
def gradient_descent(x_train, y_train, w_train, b, alpha, iters):
    w_new = np.copy(w_train)
    b_new = b
    J_history = []

    for i in range(iters):
        w_new, b_new = compute_gradient(x_train, y_train, w_new, b_new, alpha)

        if i < 100000:  # prevent resource exhaustion
            J_history.append(logistic_cost (x_train, y_train, w_new, b))

            # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]} ")

    return w_new, b_new, J_history



# Loading datasets
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])
w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.
alph = 0.1
iters = 10000

w_out, b_out, J_history = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters)
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")

