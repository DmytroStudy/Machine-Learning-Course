import numpy as np
import matplotlib.pyplot as plt
plt.style.use('../utils/deeplearning.mplstyle')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from IPython.display import display, Markdown, Latex
from sklearn.datasets import make_blobs
from matplotlib.widgets import Slider
from utils.lab_utils_common import dlc
from utils.lab_utils_softmax import plt_softmax
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)



def my_softmax(z):
    ez = np.exp(z)
    sum = np.sum(ez)
    return ez / sum


# make  dataset for example
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]] # centers of clusters
X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0,random_state=30)


# Creating model
my_model = Sequential([
    Dense(25, activation='relu'),
    Dense(15, activation='relu'),
    Dense(4, activation='linear'),])

my_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # softmax activation
    optimizer=tf.keras.optimizers.Adam(0.001),
)

model_fit = my_model.fit(X_train, y_train,epochs=10)


# Using model
predictions = my_model.predict(X_train) # predictions
print(f"\nTwo example output vectors:\n {predictions[:2]}")

predictions_probs = tf.nn.softmax(predictions).numpy() # predictions as probabilities
print(f"\nTwo example output vectors as probabilities:\n {predictions_probs[:2]}")

print("\nFive example output prefered categories:")
for i in range(5): # predictions of the most likely category
    print(f"{predictions[i]}, category: {np.argmax(predictions[i])}")