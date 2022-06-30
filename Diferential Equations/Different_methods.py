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
from scipy import integrate
import matplotlib.pyplot as plt
# Just for macOS plots
import matplotlib as mpl
mpl.use('macosx')

# Pendulum geometry
length = 2
c = 9.81/length

# Damping
b = 0.1
d = -1.0
omega = 1.0

theta_0 = 2.0
theta_1 = 0.0
n_max = 200          # Numbers of iterations
h = 0.1             # Step Size

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


def f_ODE(t, theta_0, theta_1):
    return -b*theta_1 - c*np.sin(theta_0) - d*np.sin(omega*t)


# Call the Euler Method
solution = euler_ode2(f_ODE, 0, theta_0, theta_1, n_max, h)


def f_ODE(t, theta):
    return theta[1], -b*theta[1] - c*np.sin(theta[0]) - d*np.sin(omega*t)


t_array = np.linspace(0, 100, 201)

solution_RK45 = integrate.solve_ivp(f_ODE, [0, 100], [theta_0, theta_1], method='RK45', t_eval=np.linspace(0, 100, 201))
solution_RK23 = integrate.solve_ivp(f_ODE, [0, 100], [theta_0, theta_1], method='RK23', t_eval=t_array)
solution_DOP853 = integrate.solve_ivp(f_ODE, [0, 100], [theta_0, theta_1], method='DOP853', t_eval=t_array)
solution_Radau = integrate.solve_ivp(f_ODE, [0, 100], [theta_0, theta_1], method='Radau', t_eval=t_array)
solution_BDF = integrate.solve_ivp(f_ODE, [0, 100], [theta_0, theta_1], method='BDF', t_eval=t_array)
solution_LSODA = integrate.solve_ivp(f_ODE, [0, 100], [theta_0, theta_1], method='LSODA', t_eval=t_array)

# -------------- #
#  Plot Section  #
# -------------- #
# Set Background style
plt.style.use('dark_background')
fig = plt.figure()
plt.title('Different Methods for Solving Differential Equations', fontsize=14)
plt.xlabel('Time [s]', fontsize=14)
plt.ylabel('Error comparison', fontsize=14)
plt.xlim(10, 30)
plt.ylim(-0.1, 0.1)

#plt.plot(solution_RK45.t, solution_RK45.y[0] - solution[1], 'white', label='RK45 vs EulerMethod')
plt.plot(solution_RK45.t, solution_RK45.y[0] - solution_RK23.y[0], 'blue', label='RK45 vs RK23')
plt.plot(solution_RK45.t, solution_RK45.y[0] - solution_DOP853.y[0], 'red', label='RK45 vs DOP853')
plt.plot(solution_RK45.t, solution_RK45.y[0] - solution_Radau.y[0], 'green', label='RK45 vs Radau')
plt.plot(solution_RK45.t, solution_RK45.y[0] - solution_BDF.y[0], 'orange', label='RK45 vs BDF')
plt.plot(solution_RK45.t, solution_RK45.y[0] - solution_LSODA.y[0], 'purple', label='RK45 vs LSODA')
plt.legend()

plt.show()
