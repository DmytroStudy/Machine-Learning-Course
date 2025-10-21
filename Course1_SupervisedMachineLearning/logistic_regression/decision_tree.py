import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *

def entropy(p):
    if p <= 0 or p >= 1:
        return 0
    return -p * np.log2(p) - (1-p)*np.log2(1-p)


def split_indexes(X, index_feature):
    left_indexes, right_indexes = [], []\

    for i, x in enumerate(X):
        if x[index_feature] == 1:left_indexes.append(i)
        else:right_indexes.append(i)

    return left_indexes, right_indexes

def weighted_entropy(X, y, left_indexes, right_indexes):
    w_left = len(left_indexes)/len(X)
    w_right = len(right_indexes)/len(X)

    p_left = sum(y[left_indexes])/len(left_indexes)
    p_right = sum(y[right_indexes])/len(right_indexes)

    w_entropy = w_left * entropy(p_left) + w_right * entropy(p_right)
    return w_entropy

def information_gain(X, y, left_indexes, right_indexes):
    p_node = sum(y)/len(y)
    h_node = entropy(p_node)
    w_entropy = weighted_entropy(X, y, left_indexes, right_indexes)
    return h_node - w_entropy

def split_tree(X, y):
    inf_gain = 0

    for i in range(len(X[0])):
        left_indexes, right_indexes = split_indexes(X, i)
        gain = information_gain(X, y, left_indexes, right_indexes)
        print(gain)
        if gain > inf_gain: inf_gain = gain; left_node, right_node = left_indexes, right_indexes

    return left_node, right_node


# Ear Shape (1 - pointy, 0 - otherwise)
# Face Shape (1 - round, 0 - otherwise)
# Whiskers (1 - present, 0 - otherwise)

X_train = np.array([[1, 1, 1],
[0, 0, 1],
[0, 1, 0],
[1, 0, 1],
[1, 1, 1],
[1, 1, 0],
[0, 0, 0],
[1, 1, 0],
[0, 1, 0],
[0, 1, 0]])

# Cat (1 - yes, 0 - no)
y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])

print(f"{i, x}" for i, x in enumerate(X_train[0]))

left_node, right_node = split_tree(X_train, y_train)
print(f"{left_node, right_node}")