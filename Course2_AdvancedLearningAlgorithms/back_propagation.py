from sympy import *
import numpy as np
import re


# Inputs and parameters
x, y = 2, 1
w, b = -2, 8

# Step values
c = w * x
a = c + b
d = a - y
J = (d**2)/2
print(f"J={J}, d={d}, a={a}, c={c}")



# Back propagation using sympy
sx,sw,sb,sy,sJ = symbols('x,w,b,y,J')
sa, sc, sd = symbols('a,c,d')

# Calculation steps
exp_c = sw * sx
exp_a = sc + sb
exp_d = sa - sy
exp_J = (sd**2)/2

# Derivatives
d_Jd = diff(exp_J, sd)
d_da = diff(exp_d,sa)
d_ab = diff(exp_a,sb)
d_ac = diff(exp_a,sc)
d_cw = diff(exp_c,sw)

d_Ja = d_Jd * d_da
d_Jb = d_Ja * d_ab
d_Jc = d_Ja * d_ac
d_Jw = d_Jc * d_cw

print(f"Back propagation using sympy: {d_Jw} \n") # result (dJ/dd)



# Back propagation using arithmetics
d_epsilon = d + 0.001
J_epsilon = d_epsilon**2/2
k_Jd = (J_epsilon - J)/0.001

a_epsilon = a + 0.001
d_epsilon = a_epsilon - y
k_da = (d_epsilon - d)/0.001

b_epsilon = b + 0.001
a_epsilon = c + b_epsilon
k_ab = (a_epsilon - a)/0.001

c_epsilon = c + 0.001
a_epsilon = c_epsilon + b
k_ac = (a_epsilon - a)/0.001


w_epsilon = w + 0.001
c_epsilon = w_epsilon * x
k_cw = (c_epsilon - c)/0.001


k_Jw = k_Jd * k_da * k_ac * k_cw
print(f"Back propagation using arithmetics: {k_Jw} = d*x(3*2)\n") # result (dJ/dd)







