import numpy as np
import matplotlib.pyplot as plt
from utils.lab_utils_common import plot_data, sigmoid, draw_vthresh
plt.style.use('../utils/deeplearning.mplstyle')



# Loading data sets
X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1)



# Calculate when f(x1,x2) = 0.5
    # f(x)=g(−3+x0+x1)  ->  1/1+e^-(−3+x0+x1)=0.5  ->  e^-(−3+x0+x1)=1  ->  −3+x0+x1=0  -> !!x0 = 3-x1!!
x0 = np.array([0, 6])
x1 = 3 - x0


# Plot the decision boundary
fig,ax = plt.subplots(1,1,figsize=(5,4))
ax.plot(x0,x1, c="b")
ax.axis([0, 4, 0, 3.5])
# Fill the region below the line
ax.fill_between(x0,x1, alpha=0.2)
# Plot the original data
plot_data(X,y,ax)
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')
plt.show()