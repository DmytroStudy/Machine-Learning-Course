import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import utils
np.set_printoptions(precision=2)
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

# Going through m degrees and finding best model
def evaluate_models(X, Y, m):
    # Initializing lists to save data
    train_mses, cv_mses, models, polys = [], [], [], []

    # Splitting dataset into training(60%/30), cross-validation(20%/10), test(20%/10) sets
    x_train, x_, y_train, y_ = train_test_split(X, Y, test_size=0.4, random_state=1)
    x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.2, random_state=1)
    del x_, y_  # deleting temporary variables

    for degree in range(1, m):
        # Add polynomial features to the training set
        poly = PolynomialFeatures(degree, include_bias=False)
        X_train_mapped = poly.fit_transform(x_train)

        # Create and train the model
        model = LinearRegression()
        model.fit(X_train_mapped, y_train)

        # Compute the training MSE
        yhat = model.predict(X_train_mapped)
        train_mse = mean_squared_error(y_train, yhat) / 2

        # Add polynomial features and scale the cross-validation set
        X_cv_mapped = poly.transform(x_cv)

        # Compute the cross-validation MSE
        yhat = model.predict(X_cv_mapped)
        cv_mse = mean_squared_error(y_cv, yhat) / 2

        polys.append(poly)
        models.append(model)
        train_mses.append(train_mse)
        cv_mses.append(cv_mse)

    print(f"\nTraining MSEs (1-10): {train_mses}")
    print(f"Cross-validation MSEs (1-10): {cv_mses}\n")

    # Finding model with the lowest cross-validation MSE
    best_index = np.argmin(cv_mses)
    best_poly = polys[best_index]

    x_test_mapped = best_poly.fit_transform(x_test)
    yhat = models[best_index].predict(x_test_mapped)

    test_mse = mean_squared_error(y_test, yhat) / 2
    print(f"Test MSE of the best model (degree: {best_index+1}): {test_mse}")

    return models[best_index]



# Load the dataset (50 examples)
data = np.loadtxt('../utils/data/data_w3_ex1.csv', delimiter=',')
X = data[:,0]
Y = data[:,1]
X = np.expand_dims(X, axis=1) # Convert 1D arrays into 2D
Y = np.expand_dims(Y, axis=1)

# Scaling dataset with z-score
X_scaled, Y_scaled = StandardScaler().fit_transform(X), StandardScaler().fit_transform(Y)

# Splitting dataset into training(60%/30), cross-validation(20%/10), test(20%/10) sets
x_train, x_, y_train, y_ = train_test_split(X_scaled,Y_scaled, test_size=0.4, random_state=1)
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.2, random_state=1)
del x_, y_ # deleting temporary variables


# Training the model
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)

# Evaluating the Model
yhat = linear_model.predict(x_train)
print(f"Training MSE1: {mean_squared_error(y_train, yhat)}")

yhat_cv = linear_model.predict(x_cv)
print(f"Cross-validation MSE1: {mean_squared_error(y_cv, yhat_cv)}")


# Creating model with 2nd polynomial (x+x^2)
poly = PolynomialFeatures(degree=2, include_bias=False)
x_train2 = poly.fit_transform(x_train)
x_cv2 = poly.fit_transform(x_cv)

# Training the model
model2 = LinearRegression()
model2.fit(x_train2, y_train)

# Evaluating the Model
yhat2 = model2.predict(x_train2)
print(f"Training MSE2: {mean_squared_error(y_train, yhat2)/2}")

yhat_cv2 = model2.predict(x_cv2)
print(f"Cross-validation MSE2: {mean_squared_error(y_cv, yhat_cv2)/2}")

# Finding the best degree(1-10) for the model
best_model = evaluate_models(x_train, y_train, 10)


