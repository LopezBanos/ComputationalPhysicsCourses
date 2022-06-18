"""
Interpolation:
-----------------
Find the best function to fit a given data set.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
# Just for macOS plots
import matplotlib as mpl
mpl.use('macosx')


def correct_function(x):
    return 15 + 2.4*x - 0.5*x**2 - 0.35*x**3


n_points = 21
x_array = np.linspace(-5, 5, n_points)

# Data points
data0 = np.array([x_array, correct_function(x_array)])

# Adding Noise
data_noise = data0[1] + 10 * (2 * np.random.rand(n_points) - 1)

# Linear and cubic splines (To approximate the noise data)
spline_linear0 = interpolate.interp1d(x_array, data_noise, kind='linear')  # Create a function bases on the given data
spline_cubic = interpolate.interp1d(x_array, data_noise, kind='cubic')

# Fitting data with common curves/functions that appears in physics
spline_smooth = interpolate.UnivariateSpline(data0[0], data_noise)
spline_smooth.set_smoothing_factor(500)

# Set Background style
plt.style.use('dark_background')
fig, axs = plt.subplots(1, 1)

# Set labels names and title
plt.title('Fitting Data with spline method')
plt.xlabel('x')
plt.ylabel('y')

# Plot
x_array = np.linspace(-5, 5, 901)
plt.plot(x_array, correct_function(x_array), color='red', label='correct_function')
plt.plot(x_array, spline_linear0(x_array), color='#12FFFB', label='Spline Linear')
plt.plot(x_array, spline_cubic(x_array), color='#FFC912', label='Spline Qubic')
plt.plot(x_array, spline_smooth(x_array), color='#12FF41', label='Spline Smooth')
#  plt.scatter(data0[0], data0[1], label='Clear Data')
plt.scatter(data0[0], data_noise, label='Noise Data')
plt.legend(fontsize=13)
plt.show()
