import math
import numpy as np
import copy

K_MAX = 1000


def jacobi_method(A, n):
    k = 0

    U = np.eye(n)

    p = compute_p_and_q(A, n)[0]
    q = compute_p_and_q(A, n)[1]

    teta = compute_teta(A, p, q)
    c = math.cos(teta)
    s = math.sin(teta)

    while is_diagonal_matrix(A) is True and k <= K_MAX:
        R_pq = calculate_R_pq_Rotatie(n, p, q, c, s)
        A = np.array(A)

        R_pq = np.array(R_pq)
        R_pq_T = np.transpose(R_pq)

        inm_aux = np.dot(R_pq, A)

        A = np.dot(inm_aux, R_pq_T)
        U = np.dot(U, R_pq_T)

        p = compute_p_and_q(A, n)[0]
        q = compute_p_and_q(A, n)[1]

        teta = compute_teta(A, p, q)

        c = math.cos(teta)
        s = math.sin(teta)

        k = k + 1

    return A, U


def is_diagonal_matrix(A):
    for i in range(len(A)):
        for j in range(len(A)):
            if i != j and abs(A[i][j]) > 10 ** (-8):
                return True
    return False


def compute_p_and_q(A, n):
    maximum = 0
    p = 0
    q = 0
    for i in range(n):
        for j in range(n):
            if i < j and abs(A[i][j]) >= maximum:
                maximum = abs(A[i][j])
                p = i
                q = j
    return p, q


def compute_teta(A, p, q):
    if A[p][p] != A[q][q]:
        return 1 / 2 * math.tan(2 * A[p][q] / (A[p][p] - A[q][q]))
    else:
        if A[p][q] > 0:
            teta_factor = 1
        else:
            teta_factor = -1
    return teta_factor * math.pi / 4


def calculate_R_pq_Rotatie(n, p, q, c, s):
    R_pq = [[0] * n for _ in range(n)]
    for i in range(len(R_pq)):
        for j in range(len(R_pq)):
            if i == j != p and i == j != q:
                R_pq[i][j] = 1
            elif i == j == p or i == j == q:
                R_pq[i][j] = c
            elif i == p and j == q:
                R_pq[i][j] = s
            elif i == q and j == p:
                R_pq[i][j] = -s
            else:
                R_pq[i][j] = 0
    return R_pq


def get_lambda(A, n):
    lamba_diag = [[0] * n for i in range(n)]
    for i in range(n):
        lamba_diag[i][i] = A[i][i]
    return lamba_diag


def choleski_fact(A):

    L_np = np.linalg.cholesky(A)
    L_np_Transpose = np.transpose(L_np)
    # print(L_np, L_np_Transpose)

    A_0 = np.dot(L_np, L_np_Transpose)
    A_1 = np.dot(L_np_Transpose, L_np)

    L_np = np.linalg.cholesky(A_1)
    L_np_Transpose = np.transpose(L_np)

    for k in range(K_MAX):
        if np.linalg.norm(A_1 - A_0) <= 10 ** (-8):
            break
        else:
            A_1 = np.dot(L_np, L_np_Transpose)
            A_0 = np.dot(L_np_Transpose, L_np)

            L_np = np.linalg.cholesky(A_1)
            L_np_Transpose = np.transpose(L_np)

    return A_1


def main():
    # --------------------------- 1. Metoda Jacobi --------------------------------

    A = np.array([[0, 0, 1],
                  [0, 0, 1],
                  [1, 1, 1]])
    # A = np.random.rand(3, 3)
    n = A.shape[0]
    A_init = copy.deepcopy(A)

    A, U = jacobi_method(A, n)
    U_T = np.transpose(U)
    inm = np.dot(U_T, np.array(A_init))
    A_final = np.dot(inm, U)

    lamba_diag = get_lambda(A, n)
    a = np.dot(A_init, U)
    b = np.dot(U, lamba_diag)
    norma = np.linalg.norm(a - b)

    print("\nMetoda Jacobi",
          "\nMatricea initiala:\n " + str(A_init),
          "\nMatricea finala:\n " + str(A_final), "\nNorma: " + str(norma))

    # --------------------------- 2. Metoda Choleski ------------------------------

    A = np.array([[4, 2, 8],
                  [2, 10, 10],
                  [8, 10, 21]])
    A_factorizat = choleski_fact(A)
    print("\n\nFactorizarea Choleski",
          "\nMatricea initiala:\n " + str(A),
          "\nMatricea obtinuta:\n " + str(A_factorizat))

    # --------------------------- 3. SVD ------------------------------------------

    A = np.array([[4, 2, 8],
                  [2, 10, 10],
                  [8, 10, 21],
                  [6, 8, 18]])

    U, S, V = np.linalg.svd(A, full_matrices=True)

    rang = np.linalg.matrix_rank(A)
    cond = np.linalg.cond(A)
    Moore_Penrose = np.linalg.pinv(A)
    A_Transpose = np.transpose(A)
    A_patrate = np.dot(np.linalg.inv(np.dot(A_Transpose, A)), A_Transpose)

    print("\n\nSingular Value Decomposition",
          "\nMatricea initiala:\n" + str(A),
          "\nNumarul de conditionare al matricei A: " + str(cond),
          "\nRangul matricei A: " + str(rang),
          "\nValorile singulare ale matricei(S):  " + str(S),
          "\nMoore-Penrose:\n " + str(Moore_Penrose),
          "\nCele mai mici patrate:\n " + str(A_patrate),
          "\nNorma Moore_Penrose - A_patrate : " + str(np.linalg.norm(Moore_Penrose - A_patrate)))


if __name__ == "__main__":
    main()