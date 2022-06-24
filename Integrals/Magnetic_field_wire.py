"""
Magnetic field wire, project proposed by Dr. Börge Göbel.
    STRAIGHT WIRE ALONG THE Z-AXIS

• Length: [-lo,lo] (We only consider the xy plane, because all other planes behave equally)
• Thin: Radius ro (Basically non-zero for the point [x=0,y=0]
"""
import numpy as np
import matplotlib.pyplot as plt
# Constants
from scipy.constants import mu_0, pi
import scipy.integrate as integrate
# Just for macOS plots
import matplotlib as mpl
mpl.use('macosx')

##############
# PARAMETERS #
##############
j0 = 1               # Charge Density: Ampere / meter^2
r0 = 0.001           # [m]
l0 = 1000            # [m]
mu0 = 1              # Arbitrary units for permeability
coord_max = 4.9      # Range for x and y coordinates
number_points = 100  # Number of points to be calculated


#############
# FUNCTIONS #
#############
A = []


def current_density(r_array):
    if np.sqrt(r_array[0]**2 + r_array[1]**2) > r0:
        return np.array([0.0, 0.0, 0.0])
    else:
        return np.array([0.0, 0.0, j0])


"""
We can compute each component of the vector potential A. In our case, the vector potential has only z-coordinate
"""
for i in np.linspace(-coord_max, coord_max, number_points):  # x-Coordinate
    # A_x = []
    # A_y = []
    A_z = []  # Initialize the list for each x-value on the mesh
    for j in np.linspace(-coord_max, coord_max, number_points):  # y-Coordinate

        def func(z):
            """
            :param z: Argument
            :return: Function to be integrated
            """
            return mu0 / (4 * pi) * j0 / np.sqrt((i**2 + j**2 + z**2))

        def vector_potential(z_lower_limit, z_upper_limit):
            return integrate.quad(func, z_lower_limit, z_upper_limit)

        A_z.append(vector_potential(-l0, l0)[0])   # vector_potential(-l0, l0)[0]: Value of A in that position (i,j,0)
    A.append(np.array(A_z))                        # vector_potential(-l0, l0)[1: Error associated with (i,j,0)

# -------------- #
#  Plot Section
# -------------- #
# Coordinates Standard indexing yxz, fix by indexing using 'ij'
x3, y3 = np.meshgrid(np.linspace(-coord_max, coord_max, number_points),
                     np.linspace(-coord_max, coord_max, number_points), indexing='ij')
z3 = np.array(A)

# Set Background style
plt.style.use('dark_background')
fig = plt.figure()

# Contour plot
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect(1)
plt.xlabel('x - coordinate', fontsize=13)
plt.ylabel('y - coordinate', fontsize=13)

plt.contourf(x3, y3, z3)
plt.title('Vector Potential of Infinite straight wire with Null cross section')
cbar = plt.colorbar()
cbar.set_label(r'Vector Potential $\vec{A} = A_{z} \hat{z}$', fontsize=13)

plt.show()

# Line x = 0 plot:
y_list = np.linspace(-coord_max, coord_max, number_points)
plt.title('Line x=0', fontsize=13)
plt.xlabel('x - coordinate', fontsize=13)
plt.ylabel('y - coordinate', fontsize=13)
plt.scatter(y_list, z3[50], label='Points from numerical integration')
plt.plot(y_list, mu0/(2*pi) * j0 * np.log(2*l0 / np.sqrt(y_list**2)), color='#eb5e5e', label='Theoretical Values')
plt.show()
