"""
Numerical implementation of derivatives:
* Forward differences: f'(x) = (f(x+h) - f(x)) / h as 'h' goes to zero (limit)
* Backwards differences: f'(x) = (f(x) - f(x-h)) / (-h) as 'h' goes to zero (limit)
* Forward differences: f'(x) = (f(x+h) - f(x-h)) / (2h) as 'h' goes to zero (limit)
"""

import numpy as np
import matplotlib.pyplot as plt
# Just for macOS plots
import matplotlib as mpl
mpl.use('macosx')


# Function to be derived
def f(x):
    return np.sin(x)*x - x**3 / 100


x_array = np.linspace(-10, 10, 201)
y_array = f(x_array)

#########################
#  Analytic Derivative  #
#########################
analytical_derivative = x_array * np.cos(x_array) + np.sin(x_array) - 3 * x_array**2 / 100

##########################
#  Numerical Derivative  #
##########################
h = 0.1
forward_array = (f(x_array+h) - f(x_array))/h
backward_array = (f(x_array) - f(x_array-h))/h
central_array = (f(x_array+h) - f(x_array-h))/(2*h)

"""
RICHARDSON METHOD:
It takes two O(h^2) approximation to create an O(h^4) approximation
"""


def d1_richardson(f, x, h):
    """
    :param f: The function itself
    :param x: Argument of 'f'
    :param h: Step size
    :return: First derivative using Richardson Method with 1 Iteration, ie, O(h^4)
    """
    return 1 / (12*h) * (f(x-2*h) - 8 * f(x-h) + 8 * f(x+h) - f(x+2*h))


richardson_array = d1_richardson(f, x_array, h)

# Iteration formulae to compute dn_richardson:
# D_{n+1} = [2^{2n}D_{n}(h) - D_{n}(2h)] / [2^{2n} -1]


def d1n_richardson(n, f, x, h):
    """
    :param n: Order of iteration for 1st derivative.
    :param f: Function to be derived
    :param x: Argument of function
    :param h: Step Size
    :return: First derivative using Richardson Method with n iterations
    """
    d0 = np.array([d1_richardson(f, x, h*2**i) for i in range(n)])
    for j in range(1, n):
        d = np.array([(2**(2*j) * d0[k] - d0[k+1]) / (2**(2*j) - 1) for k in range(len(d0)-1)])
        print(d)
    return d


d1n_richardson(3, f, x_array, h)
####################
# PLOTTING SECTION #
####################

# Set Background style
plt.style.use('dark_background')
fig, axs = plt.subplots(1, 2)

# Set labels names and title
axs[0].set_title(r'Function: $x \cdot sin(x) - x^{3}/100$')
axs[0].set_xlabel('x')
axs[0].set_ylabel('f(x)')

axs[1].set_title('Differences Analytical - Numerical Method')
axs[1].set_xlabel('x')
axs[1].set_ylabel("f'(x)")

# Plot
axs[0].plot(x_array, y_array, color='#7090FF')

axs[1].plot(x_array, analytical_derivative - forward_array, label='Analytical - Forwards Differences')
axs[1].plot(x_array, analytical_derivative - backward_array, label='Analytical - Backwards Differences')
axs[1].plot(x_array, analytical_derivative - central_array, label='Analytical - Central Differences')
axs[1].plot(x_array, analytical_derivative - richardson_array, label='Analytical - Richardson Method')
plt.legend(fontsize=13)
plt.show()
