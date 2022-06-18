"""
Taylor Expansion:
-----------------
We can expand a function as a polynomial,
f(x) = Σ (1/n!) f^{n}(xo)(x-xo)^{n}
f^{n}(xo) : n-th derivativa of the f(x) function evaluate in xo.
"""

import numpy as np
import matplotlib.pyplot as plt
# Just for macOS plots
import matplotlib as mpl
mpl.use('macosx')


########################################
#  Example: Exponential function of x  #
########################################
def taylor_exp(x, x0, n_max):
    """
    :param x: Argument
    :param x0: Argument at which derivatives will be calculated
    :param n_max: n at which the series will terminate
    :return: Return taylor expansion of the exponential function evaluated at x0.
    """
    taylor_expansion = 0
    for i in range(n_max+1):
        taylor_expansion = taylor_expansion + np.exp(x0) * (x-x0)**i / np.math.factorial(i)
    return taylor_expansion


print(taylor_exp(1, 0, 10))

x_array = np.linspace(-10, 10, 200)

# Set Background style
plt.style.use('dark_background')
fig, axs = plt.subplots(1, 1)

# Set labels names and title
plt.title('Taylor Expansion of exponential function')
plt.xlabel('x')
plt.ylabel('f(x) - Taylor expanded')

# Plot numpy exp() function versus the Taylor expanded
plt.plot(x_array, np.exp(x_array), color='red', label='Numpy exp() function')
plt.scatter(x_array, taylor_exp(x_array, 0, 3), color='white', label='Taylor Expanded - n=3')
plt.scatter(x_array, taylor_exp(x_array, 0, 5), color='#12FFFB', label='Taylor Expanded - n=5')
plt.scatter(x_array, taylor_exp(x_array, 0, 10), color='#FFC912', label='Taylor Expanded - n=10')
plt.legend(fontsize=13)
plt.show()


#########################################
#  Example: sin() function of x at x0=0 #
#########################################
def taylor_sin(x, x0, n_max):
    """
    :param x: Argument
    :param x0: Argument at which derivatives will be calculated
    :param n_max: n at which the series will terminate
    :return: Return taylor expansion of the sin function evaluated at x0.
    """
    taylor_expansion = 0
    for i in range(n_max+1):
        taylor_expansion = taylor_expansion + (-1)**i * (x-x0)**(2*i+1) / np.math.factorial(2*i+1)
    return taylor_expansion


# Set labels names and title
plt.title('Taylor Expansion of sin() function')
plt.xlabel('x')
plt.ylabel('f(x) - Taylor expanded')

# Set axis limits
plt.ylim(-3, 3)

# Plot numpy sin() function versus the Taylor expanded
plt.plot(x_array, np.sin(x_array), color='red', label='Numpy sin() function')
plt.scatter(x_array, taylor_sin(x_array, 0, 3), color='white', label='Taylor Expanded - n=3')
plt.scatter(x_array, taylor_sin(x_array, 0, 5), color='#12FFFB', label='Taylor Expanded - n=5')
plt.scatter(x_array, taylor_sin(x_array, 0, 10), color='#FFC912', label='Taylor Expanded - n=10')
plt.legend(fontsize=13)
plt.show()
