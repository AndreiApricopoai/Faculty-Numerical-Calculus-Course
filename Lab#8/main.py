import numpy as np

# Defining the functions and their gradients
def F1(x, y):
    # Function F1 definition
    return x ** 2 + y ** 2 - 2 * x - 4 * y - 1

def grad_F1(x, y):
    # Gradient of F1
    return np.array([2 * x - 2, 2 * y - 4])

def F2(x, y):
    # Function F2 definition
    return 3 * x ** 2 - 12 * x + 2 * y ** 2 + 16 * y - 10

def grad_F2(x, y):
    # Gradient of F2
    return np.array([6 * x - 12, 4 * y + 16])

def F3(x, y):
    # Function F3 definition
    return x ** 2 - 4 * x * y + 5 * y ** 2 - 4 * y + 3

def grad_F3(x, y):
    # Gradient of F3
    return np.array([2 * x - 4 * y, -4 * x + 10 * y - 4])

def F4(x, y):
    # Function F4 definition
    return x ** 2 * y - 2 * x * y ** 2 + 3 * x * y + 4

def grad_F4(x, y):
    # Gradient of F4 with overflow handling
    try:
        return np.array([2 * x * y - 2 * y ** 2 + 3 * y, x ** 2 - 4 * x * y + 3 * x])
    except OverflowError:
        # Return a large but finite vector in case of overflow
        return np.array([1e100, 1e100])

# Implementation of the gradient descent method
def gradient_descent(f, grad_f, x0, y0, eta=0.01, precision=1e-5, max_iterations=30000):
    # Initialize starting point
    x, y = x0, y0
    for i in range(max_iterations):
        # Compute the gradient at the current point
        gradient = grad_f(x, y)

        # Check for overflow in the gradient
        if np.any(np.abs(gradient) > 1e100):
            print("Overflow detected. Stopping iterations.")
            break

        # Update the point using the gradient
        x_new, y_new = x - eta * gradient[0], y - eta * gradient[1]

        # Check for convergence (if the change is below a certain precision)
        if np.linalg.norm(np.array([x_new, y_new]) - np.array([x, y])) <= precision:
            break

        # Update the current point
        x, y = x_new, y_new

    # Return the point that approximates the minimum, along with the number of iterations
    return x, y, i + 1

# Apply the method to all defined functions
initial_points = [(np.random.rand(), np.random.rand()) for _ in range(4)]  # Random initial points
functions = [F1, F2, F3, F4]  # List of functions
gradients = [grad_F1, grad_F2, grad_F3, grad_F4]  # Corresponding gradients

# Loop through each function and its gradient
for f, grad_f, (x0, y0) in zip(functions, gradients, initial_points):
    # Apply gradient descent
    x_min, y_min, iterations = gradient_descent(f, grad_f, x0, y0)
    # Print the results
    print(f"Minim pentru f: x : {x_min}, y : {y_min} în {iterations} iterații")
