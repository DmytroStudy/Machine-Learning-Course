import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from utils.lab_utils_multi import  load_house_data
from utils.lab_utils_common import dlc
np.set_printoptions(precision=2)
plt.style.use('../utils/deeplearning.mplstyle')



# Load data set
X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']

# Normalize data
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)


# Create and fit the regression model
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")


# View parameters
b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters: w: {w_norm}, b:{b_norm}")



# Make a prediction
y_pred_sgd = sgdr.predict(X_norm)
print(f"Prediction on training set:\n{y_pred_sgd[:4]}" )
print(f"Target values \n{y_train[:4]}")