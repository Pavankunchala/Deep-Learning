from math import *
import matplotlib.pyplot as plt
import numpy as np

def f(mu,sigma2,x):
    coefficent = 1.0/sqrt(2*pi* sigma2)
    exponetial = exp(-0.5*(x- mu)**2/sigma2)
    
    return coefficent * exponetial
gauss_1 = f(10,4,8)
print(gauss_1)

#plotting the gausian

mu = 10
sigma2 = 4

# define a range of x values
x_axis = np.arange(0, 20, 0.1)

# create a corresponding list of gaussian values
g = []
for x in x_axis:
    g.append(f(mu, sigma2, x))

# plot the result 
plt.plot(x_axis, g)
plt.show()
