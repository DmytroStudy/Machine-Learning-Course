import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from utils.autils import *
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


# Loading dataset(1000 examples of handwritten 0 or 1)
X, y = load_data()
print ('The shape of X is: ' + str(X.shape))
print ('The shape of y is: ' + str(y.shape))




model = Sequential(
    [
        tf.keras.Input(shape=(400,)),           # Specify input size
        Dense(units=25, activation="sigmoid"),  # Layer 1
        Dense(units=15, activation="sigmoid"),  # Layer 2
        Dense(units=1, activation="sigmoid")    # Layer 3
    ], name = "my_model"
)

model.compile(  loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(0.001),)

model.fit(X, y, epochs=20)

prediction = model.predict(X[0].reshape(1,400))  # a zero
print(f" predicting a zero: {prediction}")
prediction = model.predict(X[500].reshape(1,400))  # a one
print(f" predicting a one:  {prediction}")