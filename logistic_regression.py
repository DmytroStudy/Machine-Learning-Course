import numpy as np
import matplotlib.pyplot as plt
from utils.plt_one_addpt_onclick import plt_one_addpt_onclick
from utils.lab_utils_common import draw_vthresh
plt.style.use('./utils/deeplearning.mplstyle')
np.set_printoptions(precision=2)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))



# Get sigmoid values
z_tmp = np.arange(-10,10) # array from -10 to 10 (input values)
y = sigmoid(z_tmp) # output values

print("Input (z), Output (sigmoid(z))")
print(np.c_[z_tmp, y])



# Plot function
fig,ax = plt.subplots(1,1,figsize=(5,3))
ax.plot(z_tmp, y, c="b")

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')

draw_vthresh(ax,0)
plt.show()