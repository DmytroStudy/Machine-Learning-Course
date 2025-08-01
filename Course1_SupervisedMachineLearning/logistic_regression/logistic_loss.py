import numpy as np
import matplotlib.pyplot as plt
from utils.plt_logistic_loss import  plt_logistic_cost, plt_two_logistic_loss_curves, plt_simple_example
from utils.plt_logistic_loss import soup_bowl, plt_logistic_squared_error
plt.style.use('../../utils/deeplearning.mplstyle')

# Loading simple dataset for classification
x_train = np.array([0., 1, 2, 3, 4, 5],dtype=np.longdouble)
y_train = np.array([0,  0, 0, 1, 1, 1],dtype=np.longdouble)
plt_simple_example(x_train, y_train) # plot values

# Plot linear regression cost function for logistic regression
plt_logistic_squared_error(x_train,y_train)
plt.show()

# Plot logistic regression cost function for logistic regression
plt_logistic_cost(x_train,y_train)
plt.show()