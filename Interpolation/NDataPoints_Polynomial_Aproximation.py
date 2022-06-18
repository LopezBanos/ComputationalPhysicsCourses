"""
Polynomial Approximation:
-----------------
Given a data set, approximate smoothly and perfectly the data points to a curve.
Theory: If we have 'n' data points we need a polynomial of order (n-1)

    yi = a0 + a1xi + a2xi^2 + ... + a_{n-1}xi^(n-1)

for each value i-th of data point.
"""
# Import package
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
# Just for macOS plots
import matplotlib as mpl
mpl.use('macosx')

##############################################
#             IMPORTING DATA                 #
# Effect of string length on pendulum period #
##############################################
#  [[length_string], [Time (s) for 3 oscillations]]
pendulum_data = [[26.4, 35.4, 38.7, 46.6, 52.2, 57.3, 63.2, 72.4, 86.5],
                 [3.27, 3.98, 3.99, 4.38, 4.91, 4.65, 4.91, 5.55, 5.76]]


############################
# POLYNOMIAL APPROXIMATION #
############################


def polynomial_approximation(data, x_lower_lim=0, x_max_lim=1, number_points=100):
    """
    :param data: List of list [[x_data_list],[y_data_list]]
    :param x_lower_lim: Lower limit of x_value to be plotted
    :param x_max_lim: Max Value of x_value to be plotted
    :param number_points: Number of points to be plotted with the polynomial Approx
    :return: [[x_value_list],[y_polynomial_fit_values]]
    """
    x_matrix = []
    y_matrix = data[1]

    # Loop for calculating the x_matrix
    for i in range(len(data[0])):
        i_array = np.array(data[0]) ** i
        x_matrix.append(i_array)
    x_matrix = np.transpose(x_matrix)

    # Solving the problem with np.linalg.solve(x_matrix, y_matrix)
    a_matrix = np.linalg.solve(x_matrix, y_matrix)

    # Inserting in the polynomial expression:
    x_list = np.linspace(x_lower_lim, x_max_lim, number_points)
    y_list = 0 # Initialize y values to zero
    for i in range(len(a_matrix)):
        y_list += a_matrix[i]*x_list**i

    return [x_list,y_list]


pendulum_polynomial_fit = polynomial_approximation(pendulum_data, 25, 87, 100)

####################
# PLOTTING SECTION #
####################
# Set Background style
plt.style.use('dark_background')
fig, axs = plt.subplots(1, 1)

# Set labels names and title
plt.title('Fitting Data with polynomial - General Method')
plt.xlabel('Pendulum String Length')
plt.ylabel('Time (s) for 3 oscillations')

# Plot
plt.scatter(pendulum_data[0], pendulum_data[1], label='Pendulum Lab Data')
plt.plot(pendulum_polynomial_fit[0], pendulum_polynomial_fit[1], color='#BFA7FF', label='Pendulum Polynomial Fit')
plt.legend(fontsize=13)
plt.show()
