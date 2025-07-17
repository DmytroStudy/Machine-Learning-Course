import copy, math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./utils/deeplearning.mplstyle')
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

# Single predict using linear regression
def predict(x, w, b):
    p = np.dot(x, w) + b
    return p


# Cost function for data set
def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.0

    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost = cost + (f_wb_i - y[i]) ** 2
    cost = cost / (2 * m)
    return cost


# Computes single gradient for linear regression
def compute_gradient(X, y, w, b):
    m, n = X.shape
    w_new = np.zeros((n,)) # initialize weights
    b_new = 0.

    for i in range(m):
        error = (np.dot(X[i], w) + b) - y[i]

        for j in range(n):
            w_new[j] = w_new[j] + error * X[i, j]
        b_new = b_new + error

    w_new = w_new / m
    b_new = b_new / m

    return w_new, b_new

# Batch gradient descent to learn w and b
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  # avoid modifying global w within function
    b = b_in

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_dw, dj_db = gradient_function(X, y, w, b)  ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000: J_history.append(cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}")

    return w, b, J_history  # final w,b and J history for graphing


# size, bedrooms, floors, age
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178]) # price
# optimal weights abd bias
w = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
b = 785.1811367994083


# Prediction With Multiple Variables
x_vec = X_train[0,:]
f_wb = predict(x_vec,w, b)
print(f"Prediction for {x_vec}: {f_wb}$")


# Cost Function With Multiple Variables
cost = compute_cost(X_train, y_train, w, b)
print(f'Cost at optimal w and b: {cost}')


# Gradient Descent With Multiple Variables
test_w = np.zeros_like(w)
test_b = 0.

w_final, b_final, J_hist = gradient_descent(X_train, y_train, test_w, test_b, compute_cost, compute_gradient, 5.0e-7, 1000)
print(f"\nb,w found by gradient descent: {b_final:0.2f},{w_final} \n")

m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

# Plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration"); ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost'); ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step'); ax2.set_xlabel('iteration step')
plt.show()