import numpy as np
import matplotlib.pyplot as plt
from utils.lab_utils_common import  plot_data, sigmoid, dlc
plt.style.use('../utils/deeplearning.mplstyle')


def logistic_cost(x_train, y_train, w, b):
    m = y_train.shape[0]
    cost = 0

    for i in range(m):
        z = np.dot(x_train[i], w) + b
        f_wb=1 / (1+np.exp(-z))

        if y_train[i] == 0: cost += -(np.log(1-f_wb))
        else: cost += -(np.log(f_wb))

    return cost / m


# Loading data
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])
w_tmp = np.array([1,1])

# Testing logistic cost function for different b
print("If b=-2:", logistic_cost(X_train, y_train, w_tmp, -2))
print("If b=-3:", logistic_cost(X_train, y_train, w_tmp, -3))
print("If b=-4:", logistic_cost(X_train, y_train, w_tmp, -4))



# Plot decision boundary for each b
x0 = np.arange(0,6)

x1_1 = 2 - x0
x1_2 = 3 - x0
x1_3 = 4 - x0

fig,ax = plt.subplots(1, 1, figsize=(4,4))
# Plot the decision boundary
ax.plot(x0,x1_1, c=dlc["dlblue"], label="$b$=-2")
ax.plot(x0,x1_2, c="green", label="$b$=-3")
ax.plot(x0,x1_3, c=dlc["dlmagenta"], label="$b$=-4")
ax.axis([0, 4, 0, 4])

# Plot the original data
plot_data(X_train,y_train,ax)
ax.axis([0, 4, 0, 4])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.legend(loc="upper right")
plt.title("Decision Boundary")
plt.show()

