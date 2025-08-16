import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
np.set_printoptions(precision=2)
from utils.lab_utils_multiclass_TF import *
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


# Creating 4-class dataset for classification
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train = make_blobs(n_samples=100, centers=centers, cluster_std=1.0, random_state=30)

plt_mc(X_train,y_train, 4, centers, 1.0) # plotting data


# Creating model
my_model = Sequential([
    Dense(2, activation='relu'),
    Dense(4, activation='linear'),])

my_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01),)

my_model.fit(X_train, y_train, epochs=200)

plt_cat_mc(X_train, y_train, my_model, 4) # plotting classification



