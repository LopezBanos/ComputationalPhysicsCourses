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

A0 = 4
TOsc = 50
phi = 45 * np.pi / 180

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
    return [vx, vy, -epsilon/m*vx - 2*U0/m*x + A0*np.sin(2*np.pi*t/TOsc)*np.cos(phi),
            -epsilon/m*vy - 2*U0/m*y + A0*np.sin(2*np.pi*t/TOsc)*np.sin(phi)]



x0 = 2
y0 = 0
vx0 = 0
vy0 = 1

solution_RK45 = integrate.solve_ivp(f_ODE, [0, 100], [x0, y0, vx0, vy0], method='RK45',
                                    t_eval=np.linspace(0, 100, 1001))
solution_RK45_Forces = integrate.solve_ivp(f_ODE_Forces, [0, 100], [x0, y0, vx0, vy0], method='RK45',
                                           t_eval=np.linspace(0, 100, 1001))

# -------------- #
#  Plot Section  #
# -------------- #
# Set Background style
plt.style.use('dark_background')

#############################################
#  ABSENCE OF FORCES: ROLLING BALL SOLUTION #
#############################################
fig1 = plt.figure(figsize=(10, 10))
fig1.suptitle('Trajectory in Absence of Forces', fontsize=16)
plotproj3D = fig1.add_subplot(121, projection='3d')
plotproj2D = fig1.add_subplot(122)
plotproj2D.set_aspect(1)
# make the panes transparent
plotproj3D.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
plotproj3D.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
plotproj3D.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
plt.axis('auto')

plotproj3D.set_title('Potential Energy')
plotproj3D.set_xlabel('x coordinate')
plotproj3D.set_ylabel('y coordinate')
plotproj3D.axes.set_xlim3d(left=-2, right=2)
plotproj3D.axes.set_ylim3d(bottom=-2, top=2)
plotproj3D.axes.set_zlim3d(bottom=0, top=4)

plotproj3D.contour3D(x, y, z, 300)
plotproj3D.scatter3D(solution_RK45.y[0], solution_RK45.y[1], solution_RK45.y[0]**2 + solution_RK45.y[1]**2,
                   color='#FC433B',  marker="o")
plotproj3D.plot(solution_RK45.y[0], solution_RK45.y[1], color='red')  # Plane Projection of Ball Positions
plotproj2D.plot(solution_RK45.y[0], solution_RK45.y[1], color='red')
plotproj2D.set_title('Trajectory in Absence of Forces')
plotproj2D.set_xlabel('x coordinate')
plotproj2D.set_ylabel('y coordinate')
plt.show()


# Analytical
plt.ylabel('Radial Coordinate')
plt.xlabel('Time [s]')
plt.title('Analytical Solution: No external Forces')
plt.plot(solution_RK45.t, solution_RK45.y[0], color='#D7BDE2', label='X Position')
plt.plot(solution_RK45.t, solution_RK45.y[1], color='orange', label='Y Position')
plt.legend()

##############################################
#  PRESENCE OF FORCES: ROLLING BALL SOLUTION #
##############################################
fig2 = plt.figure(figsize=(10, 10))
fig2.suptitle('Trajectory in Presence of Forces', fontsize=16)
plotproj3D = fig2.add_subplot(121, projection='3d')
plotproj2D = fig2.add_subplot(122)
plotproj2D.set_aspect(1)
# make the panes transparent
plotproj3D.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
plotproj3D.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
plotproj3D.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
plt.axis('auto')

plotproj3D.set_title('Potential Energy')
plotproj3D.set_xlabel('x coordinate')
plotproj3D.set_ylabel('y coordinate')
plotproj3D.axes.set_xlim3d(left=-2, right=2)
plotproj3D.axes.set_ylim3d(bottom=-2, top=2)
plotproj3D.axes.set_zlim3d(bottom=0, top=4)

plotproj3D.contour3D(x, y, z, 300)
plotproj3D.scatter3D(solution_RK45_Forces.y[0], solution_RK45_Forces.y[1],
                     solution_RK45_Forces.y[0]**2 + solution_RK45_Forces.y[1]**2,color='#FC433B',  marker="o")
plotproj3D.plot(solution_RK45_Forces.y[0], solution_RK45_Forces.y[1], color='red')  # Plane Projection of Ball Positions
plotproj2D.plot(solution_RK45_Forces.y[0], solution_RK45_Forces.y[1], color='red')
plotproj2D.set_title('Trajectory in Presence of Forces')
plotproj2D.set_xlabel('x coordinate')
plotproj2D.set_ylabel('y coordinate')
plt.show()


# Analytical
plt.ylabel('Radial Coordinate')
plt.xlabel('Time [s]')
plt.title('Numerical Solution: With Forces')
plt.plot(solution_RK45_Forces.t, solution_RK45_Forces.y[0], color='#D7BDE2', label='X Position')
plt.plot(solution_RK45_Forces.t, solution_RK45_Forces.y[1], color='orange', label='Y Position')
plt.legend()

plt.show()