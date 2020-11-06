from math import *
import matplotlib.pyplot as plt
import numpy as np

def f(mu,sigma2,x):
    coefficent = 1.0/sqrt(2*pi* sigma2)
    exponetial = exp(-0.5*(x- mu)**2/sigma2)
    
    return coefficent * exponetial

#if we have two gausian graph and we want to find the mean of the vars

def update(mean1,var1,mean2,var2):
    
    new_mean = (var2*mean1 + var1*mean2)/(var2 + var1)
    new_var = 1/(1/var2 + 1/var1)
    
    return [new_mean , new_var]

# test your implementation
new_params = update(10, 4, 12, 4)
print(new_params)

mu = new_params[0]
sigma2 = new_params[1]

# define a range of x values
x_axis = np.arange(0, 20, 0.1)

# create a corresponding list of gaussian values
g = []
for x in x_axis:
    g.append(f(mu, sigma2, x))

# plot the result 

plt.plot(x_axis, g)
plt.show()


