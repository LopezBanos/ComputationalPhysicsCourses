"""
Fitting a model function:
-----------------
Choose ideal parameters of a (physically motivated) model function such that error is minimized.
"""
import numpy as np
import matplotlib.pyplot as plt
# Just for macOS plots
import matplotlib as mpl
mpl.use('macosx')

##############################################
#             IMPORTING DATA                 #
# Effect of string length on pendulum period #
##############################################
#  [[length_string], [Time (s) for 3 oscillations]]
x_weird_data = [0, 1, 2, 3]  # Issue: It must fit the number of data with the number of coefficients
y_weird_data = []
for x in x_weird_data:
    y_weird_data.append(-0.35 * x**3 - 0.5 * x**2 + 2.4 * x + 15)
weird_data = [x_weird_data,y_weird_data]

############################
#     POLYNOMIAL MODEL     #
############################


def polynomial_model(x, A):
    """
    :param x: x_array
    :param A: Matrix A, that contain the coefficients we want to optimize
    :return: The polynomial according to that coefficients
    """
    t = 0
    for i in range(len(A)):
        t = t + A[i] * x**i
    return t


true_coefficients = np.array([15, 2.4, -0.5, -0.35])
guess_coefficients = 2 * np.random.rand(4) - 1  # 4 Coefficients

##################################################
#             ERROR FUNCTION                     #
# Error function fot the adjustment coefficients #
#           ∆ = ∑ (yi - f(x))**2                 #
##################################################


def error_fit(f, coefficients, data):
    """
    :param f: The function
    :param coefficients: ai that we try to optimize
    :param data: the data we try to fit
    :return: The error with that coefficients
    """
    error = 0
    for i in range(len(data[0])):
        error = error + (data[1][i] - f(data[0][i], coefficients))**2
    return error

########################################################
#    GRADIENT DESCENT METHOD FOR ERROR FUNCTION        #
# Update the coefficients to reduce the error function #
# We need the partial derivative with respect the      #
# coefficients,i.e, the gradient.                      #
########################################################


def gradient_descent_method(f, coefficients, data):
    """
    :param f: The function
    :param coefficients: ai that we try to optimize
    :param data: the data we try to fit
    :return: The gradient of the error function with the given coefficients
    """
    return -2 * np.array([np.sum(np.array([(data[1][i] - f(data[0][i], coefficients)) * data[0][i]**j
                          for i in range(len(data[0]))]))
                          for j in range(len(coefficients))])


# Loop for reducing the error in the coefficients
iterations = 100000
h = 0.00001
print('True Coefficients:', true_coefficients)
print('Initial Guess Coefficients', guess_coefficients)

for i in range(iterations):
    # Minus sign comes from the fact we are moving opposite to gradient
    grad_descent = gradient_descent_method(polynomial_model, guess_coefficients, weird_data)
    guess_coefficients = guess_coefficients - h * grad_descent
print('Final Guess Coefficients', guess_coefficients)

####################
# PLOTTING SECTION #
####################
x_list = np.linspace(0, 10, 100)
# Set Background style
plt.style.use('dark_background')
fig, axs = plt.subplots(1, 1)

# Set labels names and title
plt.title('Fitting Data with polynomial - General Method')
plt.xlabel('x')
plt.ylabel('y')

# Plot
plt.scatter(weird_data[0], weird_data[1], label='Weird Data')
plt.plot(x_list, polynomial_model(x_list, guess_coefficients), color='#BFA7FF', label='Weird Data Fit')
plt.legend(fontsize=13)
plt.show()
