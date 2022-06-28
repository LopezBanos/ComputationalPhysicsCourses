"""
Author: Sergio Lopez  Banos
GitHub: LopezBanos
Exercise Title: RADIOACTIVE DECAY
Date: 28 June 2022
Description: In this exercise we need to solve a first order differential equation.
"""
import time
start_time = time.time()
import numpy as np
import matplotlib.pyplot as plt
# Just for macOS plots
import matplotlib as mpl
mpl.use('macosx')

# PARAMETERS
n_max = 200       # Numbers of iterations
h = 0.1          # Step Size
yo = 1           # Initial Condition
y = 1            # Initialize the variable
test_t = np.linspace(0, n_max*h, n_max)
t_values = [0]   # The first value of t is zero.
y_values = [1]   # The first value of y is yo.
# Analytical solution:


def rad_decay_theo(initial_condition, time):
    """
    :param initial_condition: Initial condition of the diferential equation problem
    :param time: Argument of the function
    :return: The solution of y' = -y
    """
    return initial_condition * np.exp(-time)


# Numerical Solution (Euler Method):
for i in range(1, n_max + 1):
    f = -y
    y = y + f * h
    t_values.append(i*h)
    y_values.append(y)

print("--- %s seconds ---" % (time.time() - start_time))
# -------------- #
#  Plot Section
# -------------- #
# Set Background style
plt.style.use('dark_background')
fig = plt.figure()

# Analytical and Numerical Solution
plt.title('Radioactive Decay', fontsize=14)
plt.xlabel('Time [s]', fontsize=14)
plt.ylabel("Solution of differential equation: y' = -y ", fontsize=14)
plt.scatter(t_values, y_values, label='Numerical Solution')
plt.plot(test_t, rad_decay_theo(yo, test_t), color='#FB5959', label='Analytical Solution')
plt.legend(fontsize=14)

plt.show()
