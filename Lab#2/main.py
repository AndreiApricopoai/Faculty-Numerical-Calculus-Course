import numpy as np


def can_do_lu_decomposition(A):
    # Check if A is square
    n, m = A.shape
    if n != m:
        return False

    # Check leading principal minors
    for i in range(1, n + 1):
        minor = A[:i, :i]
        if np.linalg.det(minor) == 0:
            return False

    return True


# Pasul 1: Descompunerea LU
def lu_decomposition(A):
    """
    Această funcție realizează descompunerea LU a matricei A.
    Returnează matricile L și U.
    """
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        L[i][i] = 1
        for j in range(i, n):
            sum_u = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = A[i][j] - sum_u

        for j in range(i + 1, n):
            sum_l = sum(L[j][k] * U[k][i] for k in range(i))
            L[j][i] = (A[j][i] - sum_l) / U[i][i]

    return L, U


# Pasul 2: Calculul determinantului folosind U
def determinant(L, U):
    """
    Calculează determinantul corect folosind matricile L și U din descompunerea LU.
    Determinantul lui A este produsul determinantelor lui L și U.
    """
    det_L = np.prod(np.diag(L))
    det_U = np.prod(np.diag(U))
    return det_L * det_U


# Pasul 3: Soluționarea sistemului Ax = b folosind descompunerea LU
def lu_solve(L, U, b):
    """
    Soluționează sistemul Ax = b folosind descompunerea LU și metoda substituției.
    """
    n = L.shape[0]
    # Substituție directă pentru a găsi y în Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    # Substituție inversă pentru a găsi x în Ux = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x


# Continuăm cu Pasul 4: Verificarea soluției
def verify_solution(A_init, x_LU, b_init):
    """
    Verifică soluția calculată prin calcularea normei euclidiene a diferenței
    dintre produsul Ax și vectorul b.
    """
    residual = np.dot(A_init, x_LU) - b_init
    norm = np.linalg.norm(residual, 2)
    return norm


# Exemplificăm utilizarea funcțiilor
A = np.array([[1, 2, 1, 1],
              [1, 4, -1, 7],
              [4, 9, 5, 11],
              [1,0,6,4]
              ])

if can_do_lu_decomposition(A):
    print("Matricea A poate fi descompusă LU.")

    b = np.array([0, 20, 18, 1])

    L, U = lu_decomposition(A)

    print("L = ", L)
    print("U = ", U)

    # Calculăm determinantul corectat
    det_A = determinant(L, U)
    print("Determinantul lui A este: ", det_A)

    x = lu_solve(L, U, b)
    print("Soluția sistemului Ax = b este: ", x)

    # Datele inițiale sunt A și b, iar x_LU este soluția sistemului obținută anterior
    A_init = A.copy()
    b_init = b.copy()
    norm_residual = verify_solution(A_init, x, b_init)

    print("Norma reziduală este: ", norm_residual)
else:
    print("Matricea A nu poate fi descompusă LU.")


'''
A = np.random.rand(100, 100)
A = np.dot(A, A.T)  # make A symmetric
b = np.random.rand(100)

if can_do_lu_decomposition(A):
    print("Matricea A poate fi descompusă LU.")

    L, U = lu_decomposition(A)

    # Calculăm determinantul corectat
    det_A = determinant(L, U)
    print("Determinantul lui A este: ", det_A)

    x = lu_solve(L, U, b)
    print("Soluția sistemului Ax = b este: ", x)

    # Datele inițiale sunt A și b, iar x_LU este soluția sistemului obținută anterior
    A_init = A.copy()
    b_init = b.copy()
    norm_residual = verify_solution(A_init, x, b_init)

    print("Norma reziduală este: ", norm_residual)
'''