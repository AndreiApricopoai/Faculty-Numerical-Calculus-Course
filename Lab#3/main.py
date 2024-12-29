import numpy as np


def compute_b(A, s):
    n = A.shape[0]
    b = np.zeros(n)
    for i in range(n):
        for j in range(n):
            b[i] += s[j] * A[i, j]
    return b


def qr_decomposition_householder(A):
    n = A.shape[0]
    Q = np.eye(n)
    R = A.copy()

    for r in range(n - 1):
        x = R[r:, r]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x)
        u = x - e
        v = u / np.linalg.norm(u)

        Q_r = np.eye(n)
        Q_r[r:, r:] -= 2.0 * np.outer(v, v)

        R = np.dot(Q_r, R)
        Q = np.dot(Q, Q_r.T)

    return Q, R


def solve_qr(Q, R, b):
    y = np.dot(Q.T, b)
    x = np.zeros_like(y)
    # Back substitution
    for i in reversed(range(R.shape[0])):
        x[i] = y[i]
        for j in range(i+1, R.shape[1]):
            x[i] -= R[i, j] * x[j]
        x[i] /= R[i, i]
    return x


def inverse_via_qr(Q, R):
    n = R.shape[0]
    inv_A = np.zeros((n, n))
    for col in range(n):
        e_col = np.zeros(n)
        e_col[col] = 1
        inv_A[:, col] = solve_qr(Q, R, e_col)
    return inv_A


#n_init = 4
#A_init = np.random.rand(n_init, n_init)
#s_init = np.random.rand(n_init)
A_init = np.array([[1, 0, 0, 0],
                   [0, 2, 0, 0],
                   [0, 0, 3, 0],
                   [0, 0, 0, 4]
                   ])

s_init = np.array([3, 2, 1, 1])

b_init = compute_b(A_init, s_init)
print("b_init: ", b_init)

# Using our Householder QR decomposition
Q_init, R_init = qr_decomposition_householder(A_init)
print("Q_init: ", Q_init)
print("R_init: ", R_init)

xHouseholder = solve_qr(Q_init, R_init, b_init)

# Using NumPy's QR decomposition for comparison
Q_np, R_np = np.linalg.qr(A_init)
xQR = solve_qr(Q_np, R_np, b_init)

# Displaying the results for comparison
print("Solution using custom Householder QR decomposition:", xHouseholder)
print("Solution using NumPy's QR decomposition:", xQR)

# Compare the solutions
print("Difference between the solutions:", np.linalg.norm(xHouseholder - xQR))

# Errors required by the task
error_xHouseholder = np.linalg.norm(np.dot(A_init, xHouseholder) - b_init)
error_xQR = np.linalg.norm(np.dot(A_init, xQR) - b_init)
error_sHouseholder = np.linalg.norm(xHouseholder - s_init)
error_sQR = np.linalg.norm(xQR - s_init)
norm_s = np.linalg.norm(s_init)

# Display the errors
print("Error ||A*x_Householder - b_init||_2:", error_xHouseholder)
print("Error ||A*x_QR - b_init||_2:", error_xQR)
print("Error ||x_Householder - s||_2 / ||s||_2:", error_sHouseholder / norm_s)
print("Error ||x_QR - s||_2 / ||s||_2:", error_sQR / norm_s)

# Ensure errors are below the threshold
epsilon = 1e-6
print("Are all errors below epsilon?", error_xHouseholder < epsilon, error_xQR < epsilon,
      (error_sHouseholder / norm_s) < epsilon, (error_sQR / norm_s) < epsilon)

# Calculate the inverse using the Householder QR decomposition
A_inv_Householder = inverse_via_qr(Q_init, R_init)
print("A_inv_Householder: ", A_inv_Householder)

# Calculate the inverse using NumPy's built-in function
A_inv_np = np.linalg.inv(A_init)
print("A_inv_np: ", A_inv_np)

# Compute the norm of the difference between the two inverses
inverse_diff_norm = np.linalg.norm(A_inv_Householder - A_inv_np)

# Display the norm of the difference
print("Norm of the difference between the inverses:", inverse_diff_norm)