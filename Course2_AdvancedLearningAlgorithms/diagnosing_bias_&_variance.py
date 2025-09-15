from sklearn.linear_model import LinearRegression, Ridge
from utils.utils import prepare_dataset, train_plot_poly, train_plot_reg_params, train_plot_diff_datasets, train_plot_learning_curve

# Fixing High Bias (underfitting)
# Train and plot polynomial regression models
x_train, y_train, x_cv, y_cv, x_test, y_test = prepare_dataset('../utils/data/c2w3_lab2_data1.csv')
model = LinearRegression()
train_plot_poly(model, x_train, y_train, x_cv, y_cv, max_degree=10, baseline=250)

# Adding second feature
x_train, y_train, x_cv, y_cv, x_test, y_test = prepare_dataset('../utils/data/c2w3_lab2_data2.csv')
model = LinearRegression()
train_plot_poly(model, x_train, y_train, x_cv, y_cv, max_degree=6, baseline=250)

# Adding regularization
lambdas = [10, 5, 2, 1, 0.5, 0.2, 0.1] # lambdas
train_plot_reg_params(lambdas, x_train, y_train, x_cv, y_cv, degree= 4, baseline=250)



# Fixing high variance (overfitting)
# Increasing lambda
lambdas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
train_plot_reg_params(lambdas, x_train, y_train, x_cv, y_cv, degree= 4, baseline=250)

# Deleting unsuitable features
x_train, y_train, x_cv, y_cv, x_test, y_test = prepare_dataset('../utils/data/c2w3_lab2_data3.csv') # dataset with unsuitable features
x_train, y_train, x_cv, y_cv, x_test, y_test = prepare_dataset('../utils/data/c2w3_lab2_data2.csv') # dataset without unsuitable features
model = LinearRegression()
file1 = {'filename':'../utils/data/c2w3_lab2_data3.csv', 'label': '3 features', 'linestyle': 'dotted'}
file2 = {'filename':'../utils/data/c2w3_lab2_data2.csv', 'label': '2 features', 'linestyle': 'solid'}
train_plot_diff_datasets(model, [file1, file2], max_degree=4, baseline=250)

# Getting more training examples
x_train, y_train, x_cv, y_cv, x_test, y_test = prepare_dataset('../utils/data/c2w3_lab2_data4.csv')
model = LinearRegression()
train_plot_learning_curve(model, x_train, y_train, x_cv, y_cv, degree= 4, baseline=250)