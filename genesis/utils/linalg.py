import quadrants as qd

import genesis as gs


@qd.func
def mat_mul(A: qd.Tensor, B: qd.Tensor, res: qd.Tensor, n, m, l, i_b):
    """
    Performs matrix multiplication between matrices A and B and stores the result in res.

    Args:
        A (qd.field): The first matrix of shape (n, m, B).
        B (qd.field): The second matrix of shape (m, l, B).
        res (qd.field): The result matrix of shape (n, l, B).
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


@qd.func
def mat_mul_vec(mat: qd.Tensor, vec: qd.Tensor, res: qd.Tensor, n, m, i_b):
    for i in range(n):
        res[i, i_b] = 0
        for j in range(m):
            res[i, i_b] += mat[i, j, i_b] * vec[j, i_b]
    return res


@qd.func
def mat_inverse(mat: qd.Tensor, L: qd.Tensor, U: qd.Tensor, y: qd.Tensor, res: qd.Tensor, n, i_b):
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
            tmp = gs.qd_float(0.0)
            for j in range(i):
                tmp += L[i, j, i_b] * y[k, j, i_b]
            if i == k:
                y[k, i, i_b] = 1 - tmp
            else:
                y[k, i, i_b] = -tmp

        # Backward substitution
        for i_ in range(n):
            i = n - 1 - i_
            tmp = gs.qd_float(0.0)
            for j in range(i + 1, n):
                tmp += U[i, j, i_b] * res[j, k, i_b]
            res[i, k, i_b] = (y[k, i, i_b] - tmp) / U[i, i, i_b]


@qd.func
def mat_add(A, B, n, m, i_b):
    for i in range(n):
        for k in range(m):
            A[i, k, i_b] += B[i, k, i_b]


@qd.func
def mat_transpose(A: qd.Tensor, B: qd.Tensor, n, m, i_b):
    for i in range(n):
        for k in range(m):
            B[k, i, i_b] = A[i, k, i_b]


@qd.func
def mat_add_eye(A: qd.Tensor, x, n, i_b):
    for i in range(n):
        A[i, i, i_b] += x


@qd.func
def mat_mask(A, mask, n, m, i_b):
    for i in range(n):
        for j in range(m):
            A[i, j, i_b] *= mask[i]


# Most chunks the terms of one batch entry split into for a fixed-order sum (see get_sum_fields). It bounds the serial
# sum over the partial sums that closes the sum, while leaving enough threads to the chunks for many thousand items.
SUM_MAX_CHUNKS = 256


def get_sum_fields(n_batches: int, n_items: int, n_sums: int):
    """Return the fields of the terms of fixed-order sums over the items of each batch entry and of their partial sums.

    A sum runs in up to three steps, so that it is bitwise reproducible whatever the scheduling of the threads. Each
    item writes its term of sum k into terms[i_b, i_item, k], func_reduce_terms sums the terms over contiguous chunks
    of the items into partials[i_b, i_chunk, k], and func_sum_chunks sums the partials of a batch entry in order.
    Items no more numerous than SUM_MAX_CHUNKS are their own chunks: the two fields are then the same, and
    func_reduce_terms does nothing, sparing its launch.
    """
    terms = qd.field(dtype=gs.qd_float, shape=(n_batches, max(n_items, 1), n_sums))
    if n_items <= SUM_MAX_CHUNKS:
        return terms, terms
    return terms, qd.field(dtype=gs.qd_float, shape=(n_batches, SUM_MAX_CHUNKS, n_sums))


@qd.func
def func_chunk_range(i_start: int, i_chunk: int, n_items: int, n_chunks: int):
    """Return the bounds of one of the contiguous chunks a range of items splits into for a fixed-order sum.

    The items i_start to i_start + n_items split into n_chunks chunks of equal size but the last, chunk i_chunk being
    the items between the returned bounds.
    """
    chunk_size = (n_items + n_chunks - 1) // n_chunks
    return i_start + qd.min(i_chunk * chunk_size, n_items), i_start + qd.min((i_chunk + 1) * chunk_size, n_items)


@qd.func
def func_reduce_terms(terms: qd.template(), partials: qd.template(), mask: qd.template()):
    """Sum in order the terms of each batch entry whose mask is set over contiguous chunks (see get_sum_fields).

    Must be called at the top level of a kernel, the chunks being summed in parallel.
    """
    if qd.static(terms.shape[1] > partials.shape[1]):
        n_items = qd.static(terms.shape[1])
        n_chunks = qd.static(partials.shape[1])
        for i_b, i_chunk, k in qd.ndrange(terms.shape[0], n_chunks, terms.shape[2]):
            if mask[i_b]:
                value_sum = gs.qd_float(0.0)
                i_start, i_end = func_chunk_range(0, i_chunk, n_items, n_chunks)
                for i in range(i_start, i_end):
                    value_sum += terms[i_b, i, k]
                partials[i_b, i_chunk, k] = value_sum


@qd.func
def func_sum_chunks(i_b: int, k: int, partials: qd.template()):
    """Sum in order the partial sums of the chunks of one batch entry (see get_sum_fields).

    The partial sums are partials[i_b, :, k], those of sum k of batch entry i_b.
    """
    value_sum = gs.qd_float(0.0)
    for i_chunk in range(partials.shape[1]):
        value_sum += partials[i_b, i_chunk, k]
    return value_sum
