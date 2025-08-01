import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.activations import sigmoid
from utils.lab_utils_common import dlc
from utils.lab_neurons_utils import plt_prob_1d, sigmoidnp, plt_linear, plt_logistic
plt.style.use('../utils/deeplearning.mplstyle')
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)




# Loading dataset
X_train1 = np.array([[1.0], [2.0]], dtype=np.float32)    #(size in 1000 square feet)
Y_train1 = np.array([[300.0], [500.0]], dtype=np.float32)    #(price in 1000 of dollars)


#Neuron without activation - Regression/Linear Model

# Creating layer with one neuron(unit)
linear_layer = tf.keras.layers.Dense(units=1, activation = 'linear')

# Testing model with dataset
a1 = linear_layer(X_train1)

# Getting weights
w1, b1= linear_layer.get_weights()
print(f"\nw1 = {w1}, b1={b1}")

# Setting our own weights
set_w1 = np.array([[200]])
set_b1 = np.array([100])
linear_layer.set_weights([set_w1, set_b1])

# Testing model with dataset
a2 = linear_layer(X_train1)
print(a2)

# Plotting two models
plt_linear(X_train1, Y_train1, a1, a2)



# Neuron with Sigmoid activation

# Loading dataset
X_train2 = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1,1)  # 2-D Matrix
Y_train2 = np.array([0,  0, 0, 1, 1, 1], dtype=np.float32).reshape(-1,1)  # 2-D Matrix

# Creating layer with one neuron(unit)
model = Sequential([tf.keras.layers.Dense(1, input_dim=1,  activation = 'sigmoid', name='L1')])
model.summary() # shows the layers and number of parameters in the model

# Getting layer
logistic_layer = model.get_layer('L1')

# Getting weights
w3,b3 = logistic_layer.get_weights()
print(w3,b3)

# Setting our own weights
set_w2 = np.array([[2]])
set_b2 = np.array([-4.5])
logistic_layer.set_weights([set_w2, set_b2])

# Testing model with dataset
a3 = model.predict(X_train2[0].reshape(1,1))
print(a3)

# Plotting two models
plt_logistic(X_train2, Y_train2, model, set_w2, set_b2, Y_train2 == 1, Y_train2 == 0)