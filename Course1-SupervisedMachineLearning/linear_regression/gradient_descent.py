import math, copy
import vectorization as np
import matplotlib.pyplot as plt
from utils.lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients
plt.style.use('../utils/deeplearning.mplstyle')

def cost_function(x, y, w, b):
    m = x.shape[0] # length of 1st array column
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i]) ** 2
    total_cost = 1 / (2 * m) * cost

    return total_cost

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    tmp_w = 0
    tmp_b = 0

    for i in range(m):
        f_wb = w * x[i] + b
        tmp_w_i = (f_wb - y[i]) * x[i]
        tmp_b_i = f_wb - y[i]
        tmp_w += tmp_w_i
        tmp_b += tmp_b_i
    tmp_w /= m
    tmp_b /= m

    return tmp_w, tmp_b

def gradient_descent(x, y, w, b, a, num_iters, cost_function, gradient_function):
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []

    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        tmp_w, tmp_b = gradient_function(x, y, w, b)
        b = b - a * tmp_b
        w = w - a * tmp_w

        # Save cost J at each iteration
        if i < 100000:
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])
        # Print cost
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e}")

    return w, b, J_history, p_history  # return w, b and J history for graphing

x_train = np.array([1.0, 2.0]) # input values
y_train = np.array([300.0, 500.0]) # output values

# Plot cost function and gradient
plt_gradients(x_train,y_train, cost_function, compute_gradient)
plt.show()

# Gradient descent with alpha = 0.1
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, 0, 0, 0.1, 10000, cost_function, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

# Plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step')
plt.show()

# Contour plot of the cost
fig, ax = plt.subplots(1,1, figsize=(12, 4))
plt_contour_wgrad(x_train, y_train, p_hist, ax, w_range=[180, 220, 0.5], b_range=[80, 120, 0.5], contours=[1,5,10,20],resolution=0.5)

# Gradient descent with alpha = 0.8
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, 0, 0, 0.8,10, cost_function, compute_gradient)

# Plot steps on gradient descent
plt_divergence(p_hist, J_hist,x_train, y_train)
plt.show()
