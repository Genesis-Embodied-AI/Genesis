import taichi as ti

import genesis as gs


@ti.func
def mat_mul(A, B, res, n, m, l, i_b):
    """
    Performs matrix multiplication between matrices A and B and stores the result in res.

    Args:
        A (ti.field): The first matrix of shape (n, m, B).
        B (ti.field): The second matrix of shape (m, l, B).
        res (ti.field): The result matrix of shape (n, l, B).
        n (int): The number of rows in matrix A and res.
        m (int): The number of columns in matrix A and rows in matrix B.
        l (int): The number of columns in matrix B and res.
        i_b (int): batch index.
    """
    for i in range(n):
        for j in range(l):
            res[i, j, i_b] = 0
            for k in range(m):
                res[i, j, i_b] += A[i, k, i_b] * B[k, j, i_b]


@ti.func
def mat_mul_vec(mat, vec, res, n, m, i_b):
    for i in range(n):
        res[i, i_b] = 0
        for j in range(m):
            res[i, i_b] += mat[i, j, i_b] * vec[j, i_b]
    return res


@ti.func
def mat_inverse(mat, L, U, y, res, n, i_b):
    """
    Inverse via LU decomposition
    """
    # LU decomposition
    for i in range(n):
        L[i, i, i_b] = 1
        for j in range(i, n):
            tmp = mat[i, j, i_b]
            for k in range(i):
                tmp -= L[i, k, i_b] * U[k, j, i_b]
            U[i, j, i_b] = tmp

        for j in range(i + 1, n):
            tmp = mat[j, i, i_b]
            for k in range(i):
                tmp -= L[j, k, i_b] * U[k, i, i_b]
            L[j, i, i_b] = tmp / U[i, i, i_b]

    # Forward and backward substitution for each column k of the identity matrix
    for k in range(n):
        # Forward substitution
        for i in range(n):
            tmp = gs.ti_float(0.0)
            for j in range(i):
                tmp += L[i, j, i_b] * y[k, j, i_b]
            if i == k:
                y[k, i, i_b] = 1 - tmp
            else:
                y[k, i, i_b] = -tmp

        # Backward substitution
        for i_ in range(n):
            i = n - 1 - i_
            tmp = gs.ti_float(0.0)
            for j in range(i + 1, n):
                tmp += U[i, j, i_b] * res[j, k, i_b]
            res[i, k, i_b] = (y[k, i, i_b] - tmp) / U[i, i, i_b]


@ti.func
def mat_add(A, B, n, m, i_b):
    for i in range(n):
        for k in range(m):
            A[i, k, i_b] += B[i, k, i_b]


@ti.func
def mat_transpose(A, B, n, m, i_b):
    for i in range(n):
        for k in range(m):
            B[k, i, i_b] = A[i, k, i_b]


@ti.func
def mat_add_eye(A, x, n, i_b):
    for i in range(n):
        A[i, i, i_b] += x


@ti.func
def mat_mask(A, mask, n, m, i_b):
    for i in range(n):
        for j in range(m):
            A[i, j, i_b] *= mask[i]
