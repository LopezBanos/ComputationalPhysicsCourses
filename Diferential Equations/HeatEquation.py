"""
Author: Sergio Lopez  Banos
GitHub: LopezBanos
Exercise Title: HeatEquation
Date: 15 July 2022
Description: Solve the rolling ball second order differential equation in a given potential.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
# Just for macOS plots
import matplotlib as mpl
mpl.use('macosx')

# PARAMETERS
a = 1.0
dx = 1.0
dy = 1.0

def f_1D(time,u):
    """
    :param time:
    :param u:
    :return:
    """
    u_new = np.zeros(len(u))
    u_new[1:-1] = u[2:] - 2 * u[1:-1] + u[:-2]  # To exclude the edges
    return u_new * a / (dx**2)

size_x = 100
size_y = 100
def f_2D(time, u):
    """
    :param time:
    :param u:
    :return:
    """
    u = u.reshape(size_x, size_y)
    u_new = np.zeros([size_x, size_y])
    u_new[1:-1, 1:-1] = ((u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) * a / (dx**2) +
                         (u[1:-1, 2: ] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) * a / (dy**2))  # To exclude the edges
    return u_new.flatten()



tStart = 0
tEnd = 10000

# One Dimensional Solution
size = 100
u0_1D = np.zeros([size]) # Initial Condition
u0_1D[0] = 1
solution_1D = integrate.solve_ivp(f_1D, [tStart, tEnd], u0_1D, method="RK45", t_eval= np.linspace(tStart, tEnd, 10001))

t_list, x_list = np.meshgrid(solution_1D.t, np.arange(size))

# Two Dimensional Solution
u0_2D = np.zeros([size, size])
u0_2D[0, :] = 1  # Heat one side
u0_2D[:, 0] = 1  # Heat one side
solution_2D = integrate.solve_ivp(f_2D, [tStart, tEnd], u0_2D.flatten(),
                                  method="RK45", t_eval= np.linspace(tStart, tEnd, 10001))
t_list_2D, x_list_2D = np.meshgrid(np.arange(size_x), np.arange(size_y))

# -------------- #
#  Plot Section  #
# -------------- #
# Set Background style
plt.style.use('dark_background')

fig = plt.figure()
plt.title('Heat Equation', fontsize=14)
plt.xlabel('Time [s]', fontsize=14)
plt.ylabel('Temperature [K]', fontsize=14)

index = size//3
plt.plot(solution_1D.t, solution_1D.y[index], color="#a4f576", label="Element Number: #" + str(index))
index = size//2
plt.plot(solution_1D.t, solution_1D.y[index], color="#c073ff", label="Element Number: #" + str(index))
plt.show()

plt.contourf(t_list, x_list, solution_1D.y)
plt.title('Heat Equation', fontsize=14)
plt.xlabel('Time [s]', fontsize=14)
plt.ylabel('Cell Index', fontsize=14)
plt.colorbar()
plt.show()

tIndex = tEnd // 2
plt.contourf(t_list_2D, x_list_2D, solution_2D.y[:, tIndex].reshape(size_x, size_y))
plt.title('2D Heat Equation at t = ' + str(solution_2D.t[tIndex]) + " [s]", fontsize=14)
plt.xlabel('Coordinate X', fontsize=14)
plt.ylabel('Coordinate Y', fontsize=14)
plt.colorbar()
plt.show()