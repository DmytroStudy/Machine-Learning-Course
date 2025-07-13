import numpy as np
import time

# NumPy routines which allocate memory and fill arrays with value
a = np.zeros(4);                print(f"np.zeros(4) : a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.arange(4.);              print(f"np.arange(4.): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

a = np.array([5,4.2,3,2]);      print(f"np.array([5,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

a = np.random.random_sample(4); print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")



# Vector indexing operations on 1-D vectors
a = np.arange(10)

print(f"\na[2].shape: {a[2].shape} a[2]  = {a[2]}")

print(f"a[-1] = {a[-1]}") # access the last element, negative indexes count from the end

try: # indexes must be within the range of the vector, or they will produce and error
    c = a[10]
except Exception as e:
    print("The error message you'll see is:")
    print(e)



# Vector slicing operations
a = np.arange(10);  print(f"\na = {a}")

c = a[2:7:1];       print("a[2:7:1] = ", c) # access 5 consecutive elements (start:stop:step)
c = a[3:];          print("a[3:] = ", c) # access all elements index 3 and above
c = a[:3];          print("a[:3] = ", c) # access all elements below index 3
c = a[:];           print("a[:] = ", c) # access all elements


#Single vector operations
a = np.array([1,2,3,4, 5])

b = -a;             print(f"\nb = -a : {b}") # negate elements of a
b = np.sum(a);      print(f"b = np.sum(a) : {b}") # sum all elements
b = np.mean(a);     print(f"b = np.mean(a) : {b}") # mean of all elements
b = a**2;           print(f"b = a**2 : {b}") # all elements squared



# Vector-Vector element-wise operations
a = np.array([ 1, 2, 3, 4])
b = np.array([-1,-2, 3, 4])
print(f"\nBinary operators work element wise: {a + b}")

c = np.array([1, 2])
try: #mismatched vector operation
    d = a + c
except Exception as e:
    print("The error message you'll see is:")
    print(e)

a = np.array([1, 2, 3, 4])
b = 5 * a # multiply a by a scalar
print(f"b = 5 * a : {b}")



# Dot product
a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
c = np.dot(a, b)
print(f"\nNumPy 1-D np.dot(a, b) = {c}, np.dot(a, b).shape = {c.shape} ")



# Matrix Creation
a = np.zeros((2, 5)); print(f"\na shape = {a.shape}, \na = {a}")

# NumPy routines which allocate memory and fill with user specified values
a = np.array([[5], [4], [3]]);   print(f"a shape = {a.shape}, np.array: a = {a}")




# Operations on matrices
#[row, column]
a = np.arange(6).reshape(3, 2)   #reshape is a convenient way to create matrices
a.reshape(-1, 2) # -1 calculates the right number of rows
print(f"\na.shape: {a.shape}, a= {a}")

#access an element
print(f"a[2,0].shape: {a[2, 0].shape}, a[2,0] = {a[2, 0]}, type(a[2,0]) = {type(a[2, 0])} Accessing an element returns a scalar")
#access a row
print(f"a[2].shape: {a[2].shape}, a[2] = {a[2]}, type(a[2]) = {type(a[2])}")

# Slicing matrices
a = np.arange(20).reshape(-1, 10)

#access 5 consecutive elements (start:stop:step)
print("\na[0, 2:7:1] = ", a[0, 2:7:1], ", a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")
#access 5 consecutive elements (start:stop:step) in all rows
print("a[:, 2:7:1] = \n", a[:, 2:7:1], ", a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")
