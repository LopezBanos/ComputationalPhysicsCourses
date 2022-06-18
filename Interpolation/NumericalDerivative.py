"""
Numerical Derivative:
-----------------
Compute numerical derivative from definition:
f'(x) = lim    f(x+h) - f(x) / h
          h->0
"""
import numpy as np
import matplotlib.pyplot as plt
# Just for macOS plots
import matplotlib as mpl
mpl.use('macosx')


def numerical_derivative(f, x, h):
    """
    :param f: Function
    :param x: Argument
    :param h: Step size
    :return: The first derivative of function f
    """
    return (f(x+h) - f(x)) / h


def n_derivative(f, x0, h, n_max):
    """
    :param f: Function
    :param x0: Argument
    :param h: Step size
    :param n_max: Derivative order
    :return: The n-th derivative of function f
    """
    derivative = 0
    for i in range(n_max+1):
        numerator = (-1)**(i+n_max) * np.math.factorial(n_max) * f(x0+i*h)
        denominator = np.math.factorial(i) * np.math.factorial(n_max-i)
        derivative = derivative + numerator / denominator

    return derivative/h**n_max


def func(x):
    return 2*np.sin(x)**2 + x


# Example:
dh = 0.01
xo = 10.5

print('Value of the function at x0: ', func(xo))
print('First order derivative: ', numerical_derivative(func, xo, dh))
print('n-order derivative: ', n_derivative(func, xo, dh, 2))

"""
General Taylor Expansion:
-----------------
Taylor Expansion of any function up to any n-th power 
of the corresponding polynomial
"""


def taylor(f, x, x0, n_max, h):
    """
    :param f: Function
    :param x: Argument
    :param x0: Point at which derivatives will be calculated
    :param n_max: n-th power at which the series will end
    :param h: Step size
    :return:
    """
    serie = 0
    for i in range(n_max+1):
        serie = serie + n_derivative(f, x0, h, i) * (x-x0)**i / np.math.factorial(i)
    return serie


#####################################
#  Example: General function of x   #
#####################################

x_array = np.linspace(-5, 5, 200)
nmax = 5
dh = 0.01

# Set Background style
plt.style.use('dark_background')
fig, axs = plt.subplots(1, 1)

# Set labels names and title
plt.title('Taylor Expansion of general function')
plt.xlabel('x')
plt.ylabel('f(x) - Taylor expanded')

# Axis Limits
plt.ylim([-5, 8])

# Plot func() taylor expanded
plt.scatter(x_array, func(x_array), label='Real Function')
plt.plot(x_array, taylor(func, x_array, 0, nmax, dh), color='white', label='Taylor Expanded - xo=0 - n=5')
plt.plot(x_array, taylor(func, x_array, 2, nmax, dh), color='#12FFFB', label='Taylor Expanded - xo=2 - n=5')
plt.plot(x_array, taylor(func, x_array, -3, nmax, dh), color='#FFC912', label='Taylor Expanded - xo=-3 - n=5')
plt.legend(fontsize=13)
plt.show()
