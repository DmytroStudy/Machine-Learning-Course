import vectorization as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0]) # input variable (size in 1000 square feet)
y_train = np.array([300.0, 500.0]) #target (price in 1000 of dollars)

m = x_train.shape[0] # number of training examples
i = 1 # index

def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m) # array of m zeros
    for i in range(m):
        f_wb[i] = w * x[i] + b # using function

    return f_wb

tmp_f_wb = compute_model_output(x_train, 200, 100) # w-weight, b-bias

# Plot model prediction
plt.plot(x_train, tmp_f_wb, color = 'blue', label = 'Our Prediction')
# Plot the data points
plt.scatter(x_train, y_train, marker = 'x', color = 'red',label = 'Actual Values')

plt.title("Housing Prices") # title
plt.ylabel('Price (in 1000s of dollars)') # y-axis label
plt.xlabel('Size (1000 sqft)') # x-axis label

plt.legend()
plt.show()