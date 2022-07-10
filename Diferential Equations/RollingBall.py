"""
Author: Sergio Lopez  Banos
GitHub: LopezBanos
Exercise Title: RollingBall
Date: 7 July 2022
Description: Solve the rolling ball second order differential equation in a given potential
Potential U = x**2 + y**2
b : Damping parameter
c = g / l
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
# Just for macOS plots
import matplotlib as mpl
mpl.use('macosx')

##############
# PARAMETERS #
##############
n_points = 201
x, y = np.meshgrid(np.linspace(-2, 2, n_points), np.linspace(-2, 2, n_points))
z = x**2 + y**2

m = 1
U0 = 1
epsilon = 0.1

#############
# FUNCTIONS #
#############
def f_ODE(t, r):
    x, y = r[0:2]
    vx, vy = r[2:4]
    return [vx, vy, -epsilon/m*vx - 2*U0/m*x, -epsilon/m*vy - 2*U0/m*y]

def f_ODE_Forces(t, r):
    x, y = r[0:2]
    vx, vy = r[2:4]
    return [vx, vy, -epsilon/m*vx - 2*U0/m*x, -epsilon/m*vy - 2*U0/m*y]



x0 = 2
y0 = 0
vx0 = 0
vy0 = 1

solution_RK45 = integrate.solve_ivp(f_ODE, [0, 100], [x0, y0, vx0, vy0], method='RK45', t_eval=np.linspace(0, 100, 1001))

# -------------- #
#  Plot Section  #
# -------------- #
# Set Background style
plt.style.use('dark_background')
plotproj = plt.axes(projection='3d')
# make the panes transparent
plotproj.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
plotproj.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
plotproj.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
plt.axis('auto')

plotproj.set_title('Potential Energy')
plotproj.set_xlabel('x coordinate')
plotproj.set_ylabel('y coordinate')
plotproj.axes.set_xlim3d(left=-2, right=2)
plotproj.axes.set_ylim3d(bottom=-2, top=2)
plotproj.axes.set_zlim3d(bottom=0, top=4)

plotproj.contour3D(x, y, z, 300)
plotproj.scatter3D(solution_RK45.y[0], solution_RK45.y[1], solution_RK45.y[0]**2 + solution_RK45.y[1]**2,
                   color='#FC433B',  marker="o")
plotproj.plot(solution_RK45.y[0], solution_RK45.y[1], color='red')  # Plane Projection of Ball Positions
plt.show()


# Analytical
plt.ylabel('Radial Coordinate')
plt.xlabel('Time [s]')
plt.title('Analytical Solution: No external Forces')
plt.plot(solution_RK45.t, solution_RK45.y[0], color='#D7BDE2', label='X Position')
plt.plot(solution_RK45.t, solution_RK45.y[1], color='orange', label='Y Position')
plt.legend()

# Numerical (Presence of Forces)
plt.show()