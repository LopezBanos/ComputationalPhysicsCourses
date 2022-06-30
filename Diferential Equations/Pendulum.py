"""
Author: Sergio Lopez  Banos
GitHub: LopezBanos
Exercise Title: Pendulum
Date: 30 June 2022
Description: Study the pendulum system of differential equations using different approaches.
θ''(t) + b * θ'(t) + c * sin(θ(t)) = 0
b : Damping parameter
c = g / l
"""
import numpy as np
import matplotlib.pyplot as plt
# Just for macOS plots
import matplotlib as mpl
mpl.use('macosx')

# Pendulum Geometry
g = 9.81    # [m / s**2]
length = 2  # [m
c = g / length

# Damping
b = 1

# Parameters
n_max = 200          # Numbers of iterations
h = 0.1             # Step Size
theta_0 = 0.2          # Initial Condition
theta_1 = 0          # Initial Derivative Condition
test_t = np.linspace(0, n_max*h, n_max)
g = 9.81  # [m / s**2]

########################
# Analytical solution  #
########################


def pendulum_theo(theta_0, time):
    """
    :param theta_0: Initial condition of the differential equation problem
    :param time: Argument of the function
    :return: The solution of y' = -y
    """
    return theta_0 * np.cos(np.sqrt(9.81 / length) * time)


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
def func(time, theta_0, theta_1):
    return -b * theta_1 - c * np.sin(theta_0)


# Call the Euler Method
solution = euler_ode2(func, 0, theta_0, theta_1, n_max, h)


# -------------- #
#  Plot Section  #
# -------------- #
# Set Background style
plt.style.use('dark_background')
fig = plt.figure()

# Analytical and Numerical Solution
plt.title(r'Pendulum: $\theta = 0.2 [rads]$ ', fontsize=14)
plt.xlabel('Time [s]', fontsize=14)
plt.ylabel(r"Solution to $\theta^{''}(t) + b * \theta^{'}(t) + c * sin(\theta (t)) = 0$ ",
           fontsize=14)
plt.scatter(solution[0], solution[1], color='#D7BDE2', label='Numerical Solution')
plt.legend(fontsize=14)

plt.show()
