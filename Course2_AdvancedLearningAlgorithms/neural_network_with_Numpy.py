import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils._repr_html import params
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from utils.lab_utils_common import dlc, sigmoid
from utils.lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
import logging
tf.autograph.set_verbosity(0)
plt.style.use('../utils/deeplearning.mplstyle')



def my_dense(g, a_in, w_train, b):
    units = w_train.shape[1]
    a_out = np.zeros(units)

    for i in range(units):
        z = np.dot(a_in, w_train[:,i]) + b[i]
        a_out[i] = g(z)
    return a_out


def my_sequential(g, x_train, *params):
    assert len(params) % 2 == 0 # Number of parameters must be even

    l = len(params) // 2
    a_out=x_train

    for i in range(l):
        w_train = params[i*2]
        b = params[2*i + 1]
        a_out = my_dense(g, a_out, w_train, b)
    return a_out


def my_predict(g, x_train, *params):
    assert len(params) % 2 == 0

    m = x_train.shape[0]
    p = np.zeros((m,1))

    for i in range(m):
        p[i,0] = my_sequential(g, x_train[i,:],*params)
    return p



# Loading dataset of coffee roasting(x: temperature and duration, y: roast quality)
X,Y = load_coffee_data()
plt_roast(X,Y)

# Normalizing data
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X)


# Building model using Numpy
W1_tmp = np.array( [[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]] )
b1_tmp = np.array( [-9.82, -9.28,  0.96] )
W2_tmp = np.array( [[-31.18], [-27.59], [-32.56]] )
b2_tmp = np.array( [15.41] )

X_tst = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example
X_tstn = norm_l(X_tst)  # remember to normalize
predictions = my_predict(sigmoid, X_tstn, W1_tmp, b1_tmp, W2_tmp, b2_tmp)
print(f"Predictions: \n{predictions}")