"""
INTEGRALS NUMERICAL IMPLEMENTATION:

There are different method to face the problem of numerical integration. Depending on the dimension of the problem,
if the integral is time-based, singularities and so on. Some methods are:

• Gauss-Kronrod Adaptive quadrature: (No singularities)
https://www.ams.org/journals/mcom/2000-69-231/S0025-5718-00-01174-1/S0025-5718-00-01174-1.pdf
• Tanh-Sinh Quadrature: (Singularities at the boundaries)
https://arxiv.org/pdf/2007.15057.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
# scipy.integrate.quad_vec: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad_vec.html
import scipy.integrate as integrate
# Just for macOS plots
import matplotlib as mpl
mpl.use('macosx')

# Define the functions
x_array = np.linspace(-3.5, 3.5, 101)


def func(x):
    return 0.5 + 0.1 * x + 0.2 * x**2 + 0.03 * x**3


# Data of that function (lab data)
x_points = np.linspace(-3, 3, 13)
data = np.array([x_points,func(x_points)])

# Integral of the function

integral_value_GK_21 = integrate.quad_vec(func, -3, 3, quadrature='gk21')
integral_value_GK_15 = integrate.quad_vec(func, -3, 3, quadrature='gk15')
print('Gauss-Kronrod 21-point rule', integral_value_GK_21)
print('Gauss-Kronrod 15-point rule', integral_value_GK_15)
# -------------- #
#  Plot Section
# -------------- #

# Set Background style
plt.style.use('dark_background')

plt.plot(x_array, func(x_array))
plt.scatter(data[0], data[1])
plt.title(r'Function: $0.03 \cdot x^{3} + 0.2\cdot x^{2} + 0.1\cdot x + 0.5$', fontsize=15)
plt.xlabel('x', fontsize=15)
plt.ylabel('f(x)', fontsize=15)
plt.show()
