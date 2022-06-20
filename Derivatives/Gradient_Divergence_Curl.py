import numpy as np
import matplotlib.pyplot as plt
# Just for macOS plots
import matplotlib as mpl
mpl.use('macosx')


# Example of scalar and vectorial functions


def f_exp(r_array):
    return np.exp(-r_array[0]**2 - r_array[1]**4)


def g_vector(r_array):
    return r_array / np.linalg.norm(r_array)

########################
#       GRADIENT       #
########################


def gradient(f, r_array, h):
    x, y, z = r_array
    partial_x = (f(np.array([x + h, y, z])) - f(np.array([x - h, y, z])))/(2 * h)
    partial_y = (f(np.array([x, y + h, z])) - f(np.array([x, y - h, z]))) / (2 * h)
    partial_z = (f(np.array([x, y, z + h])) - f(np.array([x, y, z - h]))) / (2 * h)
    return np.array([partial_x, partial_y, partial_z])


r = np.array([0.5, -1.2, -8])
step_size = 0.001
gradient(f_exp, r, step_size)
########################
#       DIVERGENCE     #
########################


def divergence(g, r_array, h):
    x, y, z = r_array
    dg_x_dx = (g(np.array([x+h, y, z]))[0] - g(np.array([x-h, y, z]))[0])
    dg_y_dy = (g(np.array([x, y + h, z]))[0] - g(np.array([x - h, y - h, z]))[0])
    dg_z_dz = (g(np.array([x, y, z + h]))[0] - g(np.array([x - h, y, z - h]))[0])
    return dg_x_dx + dg_y_dy + dg_z_dz


########################
#         CURL         #
########################


def curl(g, r_array, h):
    x, y, z = r_array
    dg_x_dy = (g(np.array([x, y + h, z]))[0] - g(np.array([x, y - h, z]))[0]) / (2 * h)
    dg_x_dz = (g(np.array([x, y, z + h]))[0] - g(np.array([x, y, z - h]))[0]) / (2 * h)
    dg_y_dx = (g(np.array([x + h, y, z]))[1] - g(np.array([x - h, y, z]))[1]) / (2 * h)
    dg_y_dz = (g(np.array([x, y, z + h]))[1] - g(np.array([x, y, z - h]))[1]) / (2 * h)
    dg_z_dx = (g(np.array([x + h, y, z]))[2] - g(np.array([x - h, y, z]))[2]) / (2 * h)
    dg_z_dy = (g(np.array([x, y + h, z]))[2] - g(np.array([x, y - h, z]))[2]) / (2 * h)
    return np.array([dg_z_dy - dg_y_dz, dg_x_dz - dg_z_dx, dg_y_dx - dg_x_dy])


# -------------- #
#  Plot Section
# -------------- #
x3, y3 = np.meshgrid(np.linspace(-2, 2, 201), np.linspace(-2, 2, 201))
z3 = f_exp(np.array([x3, y3]))

# Set Background style
plt.style.use('dark_background')
plotproj = plt.axes(projection='3d')
plotproj.set_title(r'Scalar Function: $-x^{2} - y^{4}$')
# make the panes transparent
plotproj.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
plotproj.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
plotproj.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
plotproj.contour3D(x3, y3, z3, 100, cmap='plasma')
plt.axis('auto')

plt.show()

# Vectorial Function r_array / np.linalg.norm(r_array)
x3, y3, z3 = np.meshgrid(np.linspace(-2, 2, 11), np.linspace(-2, 2, 11), np.linspace(-2, 2, 11))
values = g_vector(np.array([x3, y3, z3]))
arrow_plot = plt.axes(projection='3d')
arrow_plot.set_title(r'Vectorial Function: $\vec{r}/|\vec{r}|$')
arrow_plot.axis(False)

scale = 7
arrow_plot.quiver(x3, y3, z3, values[0] * scale, values[1] * scale, values[2] * scale)
plt.show()
