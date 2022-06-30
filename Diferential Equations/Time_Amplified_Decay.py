"""
Author: Sergio Lopez  Banos
GitHub: LopezBanos
Exercise Title: Time-Amplified Decay
Date: 30 June 2022
"""
import numpy as np
import matplotlib.pyplot as plt
# Just for macOS plots
import matplotlib as mpl
mpl.use('macosx')

########################
# Analytical solution  #
########################
n_max = 300      # Numbers of iterations
h = 0.1          # Step Size
yo = 1           # Initial Condition
test_t = np.linspace(0, n_max*h, n_max)
parameter = 0.01


def rad_decay_theo(initial_condition, time):
    """
    :param initial_condition: Initial condition of the differential equation problem
    :param time: Argument of the function
    :return: The solution of y' = -y
    """
    return initial_condition * np.exp(-(parameter / 2) * time**2)


#####################################
# Numerical Solution (Euler Method) #
#####################################
def euler_ode(func, init_time, initial_condition, n_max, step_size):
    """
    :param func: Function containing the ODE
    :param init_time: Initial time
    :param initial_condition: Initial value of the function
    :param n_max: Number of points to be calculated
    :param step_size: Step-size
    :return:
    """
    y = initial_condition
    time = init_time
    t_values = [time]
    y_values = [y]
    for i in range(1, n_max + 1):
        y = y + func(time, y) * step_size
        time += step_size
        t_values.append(i * step_size)
        y_values.append(y)
    return np.array([t_values, y_values])


# Define Function
def f(time, y):
    return - parameter * y * time


# Call the Euler Method
solution = euler_ode(f, 0, 1, n_max, h)


# -------------- #
#  Plot Section  #
# -------------- #
# Set Background style
plt.style.use('dark_background')
fig = plt.figure()

# Analytical and Numerical Solution
plt.title('Time-Amplified Radioactive Decay', fontsize=14)
plt.xlabel('Time [s]', fontsize=14)
plt.ylabel(r"Solution of differential equation: $y' = -a\cdot t \cdot y$", fontsize=14)
plt.scatter(solution[0], solution[1], label='Numerical Solution')
plt.plot(test_t, rad_decay_theo(yo, test_t), color='#FB5959', label='Analytical Solution')
plt.legend(fontsize=14)

plt.show()