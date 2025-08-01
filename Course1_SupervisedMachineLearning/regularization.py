import numpy as np
import matplotlib.pyplot as plt
from utils.plt_overfit import overfit_example, output
from utils.lab_utils_common import sigmoid
import math
np.set_printoptions(precision=8)



def regularized_linear_cost(x_train, y_train, w_train, b, lambda_):
    m, n = x_train.shape
    cost, cost_reg = 0 , 0

    for i in range(m):
        f_wb = np.dot(x_train[i], w_train) + b
        cost += (f_wb - y_train[i]) ** 2
    cost /= (2*m)

    for j in range(n):
        cost_reg += (w_train[j] ** 2)
    cost_reg *= lambda_ / (2*m)

    cost += cost_reg
    return cost

def regularized_logistic_cost(x_train, y_train, w_train, b, lambda_):
    m, n = x_train.shape
    cost, cost_reg = 0 , 0

    for i in range(m):
        z = np.dot(x_train[i], w_train) + b
        f_wb = 1/(1+np.exp(-z))
        cost += (y_train[i]*np.log(f_wb)) + ((1-y_train[i])*np.log(1-f_wb))
    cost /= -m

    for j in range(n):
        cost_reg += (w_train[j] ** 2)
    cost_reg *= lambda_ / (2*m)

    cost += cost_reg
    return cost

def compute_linear_gradient(x_train, y_train, w_train, b, alpha, lambda_):
    m, n = x_train.shape
    w_new = np.zeros((n,))
    b_new = 0

    for j in range(n):
        for i in range(m):
            f_wb = np.dot(x_train[i], w_train) + b
            w_new[j] += (f_wb - y_train[i]) * x_train[i,j]
        w_new[j] /= m
        w_new[j] += (lambda_/m)*w_train[j]

    for j in range(n):
        w_new[j] = w_train[j]-alpha*w_new[j]

    for i in range(m):
        f_wb = np.dot(x_train[i], w_train) + b
        b_new += (f_wb - y_train[i])

    b_new /= m
    b_new = b-alpha*b_new

    return w_new, b_new

def regularized_linear_gradient_descent(x_train, y_train, w_train, b, alpha, lambda_, iters):
    J_history = []
    w_new, b_new = np.copy(w_train), np.copy(b)
    for i in range(iters):
        w_new, b_new = compute_linear_gradient(x_train, y_train, w_new, b_new, alpha, lambda_)

        # Save cost J at each iteration
        if i < 100000:
            J_history.append(regularized_linear_cost(x_train, y_train, w_new, b_new, 0.7))

            # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(iters / 10) == 0:
            print(f"Iteration {i}: Cost {J_history[-1]} ")

    return w_new, b_new, J_history

def compute_logistic_gradient(x_train, y_train, w_train, b, alpha, lambda_):
    m, n = x_train.shape
    w_new = np.zeros((n,))
    b_new = 0

    for j in range(n):
        for i in range(m):
            z = np.dot(x_train[i], w_train) + b
            f_wb = 1/(1+np.exp(-z))
            w_new[j] += (f_wb - y_train[i]) * x_train[i,j]
        w_new[j] /= m
        w_new[j] += (lambda_/m)*w_train[j]

    for j in range(n):
        w_new[j] = w_train[j]-alpha*w_new[j]

    for i in range(m):
        z = np.dot(x_train[i], w_train) + b
        f_wb = 1/(1+np.exp(-z))
        b_new += (f_wb - y_train[i])

    b_new /= m
    b_new = b-alpha*b_new

    return w_new, b_new

def regularized_logistic_gradient_descent(x_train, y_train, w_train, b, alpha, lambda_, iters):
    J_history = []
    w_new, b_new = np.copy(w_train), np.copy(b)
    for i in range(iters):
        w_new, b_new = compute_logistic_gradient(x_train, y_train, w_new, b_new, alpha, lambda_)

        # Save cost J at each iteration
        if i < 100000:
            J_history.append(regularized_logistic_cost(x_train, y_train, w_new, b_new, 0.7))

            # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(iters / 10) == 0:
            print(f"Iteration {i}: Cost {J_history[-1]} ")

    return w_new, b_new, J_history


# Testing regularized linear cost
np.random.seed(1)
X_tmp = np.random.rand(5,6)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
cost_tmp = regularized_linear_cost(X_tmp, y_tmp, w_tmp, 0.5, 0.7)
print(f"Regularized linear cost: {cost_tmp}\n")


# Testing regularized logistic cost
X_tmp = np.random.rand(5,6)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
b_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = regularized_logistic_cost(X_tmp, y_tmp, w_tmp, 0.5, 0.7)
print(f"Regularized logisitc cost:{cost_tmp}\n")


# Testing regularized linear gradient descent
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
w_new, b_new, _ =  regularized_linear_gradient_descent(X_tmp, y_tmp, w_tmp, 0.5, 0.8 , 0.7, 100)
print(f"\nRegularized linear w, b: {w_new, b_new}")
print(f"Linear cost: {regularized_linear_cost(X_tmp, y_tmp, w_new, b_new,0.7)}\n")


# Testing regularized logistic gradient descent
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
w_new, b_new, _ = regularized_logistic_gradient_descent(X_tmp, y_tmp, w_tmp, 0.5, 0.8, 0.7, 100)
print(f"\nRegularized logistic w, b: {w_new, b_new}")
print(f"Logistic cost: {regularized_logistic_cost(X_tmp, y_tmp, w_new, b_new,0.7)}")