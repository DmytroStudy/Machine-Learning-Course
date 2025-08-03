import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from utils.lab_utils_common import dlc
from utils.lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
import logging
tf.autograph.set_verbosity(0)
plt.style.use('../utils/deeplearning.mplstyle')


# Loading dataset of coffee roasting(x: temperature and duration, y: roast quality)
X,Y = load_coffee_data()
plt_roast(X,Y)

# Normalizing data
print(f"Temperature Min, Max pre normalization: {np.min(X[:,0]):0.2f}, {np.max(X[:,0]):0.2f}")
print(f"Duration    Min, Max pre normalization: {np.min(X[:,1]):0.2f}, {np.max(X[:,1]):0.2f}")

norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X)

print(f"Temperature Min, Max post normalization: {np.min(Xn[:,0]):0.2f}, {np.max(Xn[:,0]):0.2f}")
print(f"Duration    Min, Max post normalization: {np.min(Xn[:,1]):0.2f}, {np.max(Xn[:,1]):0.2f}")



# Building Tensorflow model
model = Sequential([
    tf.keras.Input(shape=(2,)),
    Dense(3, activation='sigmoid', name="layer1"),
    Dense(1, activation='sigmoid', name="layer2")
])

model.summary()

# Getting weights
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)\

# Running model
model.compile(  loss = tf.keras.losses.BinaryCrossentropy(),
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),)

model.fit(Xn,Y,epochs=100,)

# Getting updated weights
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print("Updated W1:\n", W1, "\nb1:", b1)
print("Updated W2:\n", W2, "\nb2:", b2)

# Making prediction
X_test = np.array([
    [200,1],  # positive example
    [200,17]])   # negative example
X_testn = norm_l(X_test)
predictions = model.predict(X_testn)
print("predictions = \n", predictions)