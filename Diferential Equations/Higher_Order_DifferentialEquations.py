"""
Author: Sergio Lopez  Banos
GitHub: LopezBanos
Exercise Title: Higher Order Differential Equation
Date: 30 June 2022
Description: Transform a second order differential equation into a first order differential equation
by using a matricial approach of system of equations. This idea can be applied recursively.
"""
import numpy as np
import matplotlib.pyplot as plt
# Just for macOS plots
import matplotlib as mpl
mpl.use('macosx')


########################
# Analytical solution  #
########################
n_max = 100      # Numbers of iterations
h = 0.1          # Step Size
y0 = 10          # Initial Condition
y1 = 50          # Initial Derivative Condition
test_t = np.linspace(0, n_max*h, n_max)
g = 9.81  # [m / s**2]


def free_fall_theo(initial_condition_0, initial_condition_1, time):
    """
    :param initial_condition_0: Initial condition of the differential equation problem
    :param initial_condition_1: Initial condition of the derivative
    :param time: Argument of the function
    :return: The solution of y' = -y
    """
    return -(g / 2) * time**2 + initial_condition_1 * time + initial_condition_0


#####################################
# Numerical Solution (Euler Method) #
#####################################


def euler_ode2(func, init_time, initial_condition_0, initial_condition_1, n_max, step_size):
    """
    :param func: Function containing the ODE
    :param init_time: Initial time
    :param initial_condition_0: Initial value of the function
    :param initial_condition_1: Initial value of the function derivative at t0
    :param n_max: Number of points to be calculated
    :param step_size: Step-size
    :return:
    """
    y0 = initial_condition_0
    y1 = initial_condition_1
    time = init_time
    t_values = [time]
    y0_values = [y0]
    y1_values = [y1]
    for i in range(1, n_max + 1):
        y0 = y0 + y1 * step_size
        y1 = y1 + func(time, y0, y1) * step_size
        time += step_size
        t_values.append(i * step_size)
        y0_values.append(y0)
        y1_values.append(y1)
    return np.array([t_values, y0_values, y1_values])


# Defining Differential Equation
def func(time, y0, y1):
    return -g


# Call the Euler Method
solution = euler_ode2(func, 0, y0, y1, n_max, h)


# -------------- #
#  Plot Section  #
# -------------- #
# Set Background style
plt.style.use('dark_background')
fig = plt.figure()

# Analytical and Numerical Solution
plt.title('Free Fall', fontsize=14)
plt.xlabel('Time [s]', fontsize=14)
plt.ylabel(r"Solution of differential equation: $y^{''} = -g$ ", fontsize=14)
plt.scatter(solution[0], solution[1], label='Numerical Solution')
plt.plot(test_t, free_fall_theo(y0, y1, test_t), color='#FB5959', label='Analytical Solution')
plt.legend(fontsize=14)

plt.show()
