"""
Rotational Motion project propose by Dr. Börge Göbel.

• Mass - SI Unit: [m] = kg
• Radius - [r] = m
• Angular Velocity - [ω] = 1/s
"""
import numpy as np
import matplotlib.pyplot as plt
# Just for macOS plots
import matplotlib as mpl
mpl.use('macosx')

# PARAMETERS AND ARRAYS
m = 1  # Mass
r = 1  # Radius
omega = 1  # Angular Velocity

t_array = np.linspace(0, 2*np.pi, 100)
x_array = r * np.cos(omega * t_array)
y_array = r * np.sin(omega * t_array)


# -------------- #
#  PLOT SECTION
# -------------- #
# Set Background style
plt.style.use('dark_background')
plt3D = plt.axes(projection='3d')
# make the panes transparent
plt3D.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
plt3D.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
plt3D.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
plt.axis('auto')

plt3D.set_title('Position vs Time')
plt3D.set_xlabel('x coordinate [m]')
plt3D.set_ylabel('y coordinate [m]')
plt3D.set_zlabel('Time [s]')
plt3D.scatter3D(x_array, y_array, t_array, color='#CB70FF')
plt.show()
