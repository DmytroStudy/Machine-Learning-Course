import numpy as np
import matplotlib.pyplot as plt
from utils import *


def compute_entropy(y):
    entropy = 0.
    if len(y) < 1: return entropy

    p1 = sum(y) / len(y)
    p2 = 1 - p1

    if p1 > 0 and p2 > 0: entropy = (-p1 * np.log2(p1)) - (p2 * np.log2(p2))
    return entropy


def split_dataset(X, node_indices, feature):
    left_indices = []
    right_indices = []

    for i in node_indices:
        if X[i][feature] == 1: left_indices.append(i)
        else: right_indices.append(i)

    return left_indices, right_indices


def compute_information_gain(X, y, node_indices, feature):
    # Split dataset
    left_indices, right_indices = split_dataset(X, node_indices, feature)

    # Some useful variables
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]

    if len(y_left) == 0 or len(y_right) == 0 or len(y_node) == 0: return 0

    w_left = len(y_left) / len(y_node)
    w_right = len(y_right) / len(y_node)

    node_entropy = compute_entropy(y_node)
    left_entropy = compute_entropy(y_left)
    right_entropy = compute_entropy(y_right)

    information_gain = node_entropy - (w_left * left_entropy + w_right * right_entropy)
    return information_gain


def get_best_split(X, y, node_indices):
    num_features = X.shape[1]

    best_feature = -1
    best_gain = 0

    for i in range(num_features):
        gain = compute_information_gain(X, y, node_indices, i)
        if gain > best_gain: best_gain = gain; best_feature = i

    return best_feature


TREE = []
def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
    # Maximum depth reached - stop splitting
    if current_depth == max_depth:
        formatting = " " * current_depth + "-" * current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return

    # Otherwise, get best split and split the data
    # Get the best feature and threshold at this node
    best_feature = get_best_split(X, y, node_indices)

    formatting = "-" * current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))

    # Split the dataset at the best feature
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    TREE.append((left_indices, right_indices, best_feature))

    # continue splitting the left and the right child. Increment current depth
    build_tree_recursive(X, y, left_indices, "left", max_depth, current_depth + 1)
    build_tree_recursive(X, y, right_indices, "right", max_depth, current_depth + 1)

# Mushrooms dataset (brown cap, tapering stalk shape, solitary), (edible)
X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])

build_tree_recursive(X_train, y_train, range(10), "Root", max_depth=2, current_depth=0)




