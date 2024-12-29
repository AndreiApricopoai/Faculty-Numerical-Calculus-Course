import numpy as np
import numpy.polynomial.polynomial as poly

# Function definition for the second example.
def f(x):
    return x**4 - 12*x**3 + 30*x**2 + 12

# Function to calculate finite differences using Aitken's scheme.
def finite_differences(y):
    n = len(y)
    delta = y.copy()
    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            delta[j] = delta[j] - delta[j - 1]
    return delta

# Function to approximate using Newton's progressive formula.
def newton_progressive_approximation(x, y, x_bar):
    h = x[1] - x[0]  # Calculate the step size.
    t = (x_bar - x[0]) / h  # Calculate the normalized point for approximation.
    delta = finite_differences(y)  # Calculate the finite differences.
    approx = delta[0]  # Initialize the approximation with the first value.
    t_prod = 1
    for i in range(1, len(x)):
        t_prod *= (t - i + 1) / i  # Update the product term in the Newton formula.
        approx += t_prod * delta[i]  # Update the approximation sum.
    return approx

# Example 1: Initialize nodes and values, then perform approximations.
x_nodes_1 = np.array([0, 1, 2, 3, 4, 5])
y_values_1 = np.array([50, 47, -2, -121, -310, -545])
x_bar_1 = 1.5  # The point at which the function is approximated.
newton_approx_1 = newton_progressive_approximation(x_nodes_1, y_values_1, x_bar_1)
# Least Squares Approximation
A_1 = np.vander(x_nodes_1, 3, increasing=True)
coefficients_1 = np.linalg.lstsq(A_1, y_values_1, rcond=None)[0]
least_squares_approx_1 = poly.polyval(x_bar_1, coefficients_1)

print("Example 1:")
print("Newton Approximation: ", newton_approx_1)
print("Least Squares Approximation: ", least_squares_approx_1)

# Example 2: Define nodes and values based on the function, then perform approximations.
x_nodes_2 = np.array([1, 2, 3, 4, 5])
y_values_2 = f(x_nodes_2)
x_bar_2 = 1.5
newton_approx_2 = newton_progressive_approximation(x_nodes_2, y_values_2, x_bar_2)
# Least Squares Approximation
A_2 = np.vander(x_nodes_2, 3, increasing=True)
coefficients_2 = np.linalg.lstsq(A_2, y_values_2, rcond=None)[0]
least_squares_approx_2 = poly.polyval(x_bar_2, coefficients_2)

print("\nExample 2:")
print("Newton Approximation: ", newton_approx_2)
print("Least Squares Approximation: ", least_squares_approx_2)
