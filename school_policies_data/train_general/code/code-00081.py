    >>> a = np.array([ [[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]] ])
    >>> a.shape
    (3, 2, 2)
    >>> np.linalg.det(a)
    array([-2., -3., -8.])

    """
    a = asarray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    t, result_t = _commonType(a)
    signature = 'D->D' if isComplexType(t) else 'd->d'
    r = _umath_linalg.det(a, signature=signature)
    r = r.astype(result_t, copy=False)
    return r


# Linear Least Squares

def _lstsq_dispatcher(a, b, rcond=None):
    return (a, b)


@array_function_dispatch(_lstsq_dispatcher)
def lstsq(a, b, rcond=None):
    r"""
    Return the least-squares solution to a linear matrix equation.

    Computes the vector `x` that approximately solves the equation
    ``a @ x = b``. The equation may be under-, well-, or over-determined
    (i.e., the number of linearly independent rows of `a` can be less than,
    equal to, or greater than its number of linearly independent columns).
    If `a` is square and of full rank, then `x` (but for round-off error)
    is the "exact" solution of the equation. Else, `x` minimizes the
    Euclidean 2-norm :math:`||b - ax||`. If there are multiple minimizing
    solutions, the one with the smallest 2-norm :math:`||x||` is returned.

    Parameters
    ----------
    a : (M, N) array_like
        "Coefficient" matrix.
    b : {(M,), (M, K)} array_like
        Ordinate or "dependent variable" values. If `b` is two-dimensional,
        the least-squares solution is calculated for each of the `K` columns
        of `b`.
    rcond : float, optional
        Cut-off ratio for small singular values of `a`.
        For the purposes of rank determination, singular values are treated
        as zero if they are smaller than `rcond` times the largest singular
        value of `a`.
        The default uses the machine precision times ``max(M, N)``.  Passing
        ``-1`` will use machine precision.

        .. versionchanged:: 2.0
            Previously, the default was ``-1``, but a warning was given that
            this would change.

    Returns
    -------
    x : {(N,), (N, K)} ndarray
        Least-squares solution. If `b` is two-dimensional,
        the solutions are in the `K` columns of `x`.
    residuals : {(1,), (K,), (0,)} ndarray
        Sums of squared residuals: Squared Euclidean 2-norm for each column in
        ``b - a @ x``.
        If the rank of `a` is < N or M <= N, this is an empty array.
        If `b` is 1-dimensional, this is a (1,) shape array.
        Otherwise the shape is (K,).
    rank : int
        Rank of matrix `a`.
    s : (min(M, N),) ndarray
        Singular values of `a`.

    Raises
    ------
    LinAlgError
        If computation does not converge.

    See Also
    --------
    scipy.linalg.lstsq : Similar function in SciPy.

    Notes
    -----
    If `b` is a matrix, then all array results are returned as matrices.

    Examples
    --------
    Fit a line, ``y = mx + c``, through some noisy data-points:

    >>> import numpy as np
    >>> x = np.array([0, 1, 2, 3])
    >>> y = np.array([-1, 0.2, 0.9, 2.1])

    By examining the coefficients, we see that the line should have a
    gradient of roughly 1 and cut the y-axis at, more or less, -1.

    We can rewrite the line equation as ``y = Ap``, where ``A = [[x 1]]``
    and ``p = [[m], [c]]``.  Now use `lstsq` to solve for `p`:

    >>> A = np.vstack([x, np.ones(len(x))]).T
    >>> A
    array([[ 0.,  1.],
           [ 1.,  1.],
           [ 2.,  1.],
           [ 3.,  1.]])

    >>> m, c = np.linalg.lstsq(A, y)[0]
    >>> m, c
    (1.0 -0.95) # may vary

    Plot the data along with the fitted line:

    >>> import matplotlib.pyplot as plt
    >>> _ = plt.plot(x, y, 'o', label='Original data', markersize=10)
    >>> _ = plt.plot(x, m*x + c, 'r', label='Fitted line')
    >>> _ = plt.legend()
    >>> plt.show()

    """
    a, _ = _makearray(a)
    b, wrap = _makearray(b)
    is_1d = b.ndim == 1
    if is_1d:
        b = b[:, newaxis]
    _assert_2d(a, b)
    m, n = a.shape[-2:]
    m2, n_rhs = b.shape[-2:]
    if m != m2:
        raise LinAlgError('Incompatible dimensions')

    t, result_t = _commonType(a, b)
    result_real_t = _realType(result_t)

    if rcond is None:
        rcond = finfo(t).eps * max(n, m)

    signature = 'DDd->Ddid' if isComplexType(t) else 'ddd->ddid'
    if n_rhs == 0:
        # lapack can't handle n_rhs = 0 - so allocate
        # the array one larger in that axis
        b = zeros(b.shape[:-2] + (m, n_rhs + 1), dtype=b.dtype)

    with errstate(call=_raise_linalgerror_lstsq, invalid='call',
                  over='ignore', divide='ignore', under='ignore'):
        x, resids, rank, s = _umath_linalg.lstsq(a, b, rcond,
                                                 signature=signature)
    if m == 0:
        x[...] = 0
    if n_rhs == 0:
        # remove the item we added
        x = x[..., :n_rhs]
        resids = resids[..., :n_rhs]

    # remove the axis we added
    if is_1d:
        x = x.squeeze(axis=-1)
        # we probably should squeeze resids too, but we can't
        # without breaking compatibility.

    # as documented
    if rank != n or m <= n:
        resids = array([], result_real_t)

    # coerce output arrays
    s = s.astype(result_real_t, copy=False)
    resids = resids.astype(result_real_t, copy=False)
    # Copying lets the memory in r_parts be freed
    x = x.astype(result_t, copy=True)
    return wrap(x), wrap(resids), rank, s


def _multi_svd_norm(x, row_axis, col_axis, op):
    """Compute a function of the singular values of the 2-D matrices in `x`.

    This is a private utility function used by `numpy.linalg.norm()`.

    Parameters
    ----------
    x : ndarray
    row_axis, col_axis : int
        The axes of `x` that hold the 2-D matrices.
    op : callable
        This should be either numpy.amin or `numpy.amax` or `numpy.sum`.

    Returns
    -------
    result : float or ndarray
        If `x` is 2-D, the return values is a float.
        Otherwise, it is an array with ``x.ndim - 2`` dimensions.
        The return values are either the minimum or maximum or sum of the
        singular values of the matrices, depending on whether `op`
        is `numpy.amin` or `numpy.amax` or `numpy.sum`.

    """
    y = moveaxis(x, (row_axis, col_axis), (-2, -1))
    result = op(svd(y, compute_uv=False), axis=-1)
    return result


def _norm_dispatcher(x, ord=None, axis=None, keepdims=None):
    return (x,)


@array_function_dispatch(_norm_dispatcher)
def norm(x, ord=None, axis=None, keepdims=False):
    """
    Matrix or vector norm.

    This function is able to return one of eight different matrix norms,
    or one of an infinite number of vector norms (described below), depending
    on the value of the ``ord`` parameter.

    Parameters
    ----------
    x : array_like
        Input array.  If `axis` is None, `x` must be 1-D or 2-D, unless `ord`
        is None. If both `axis` and `ord` are None, the 2-norm of
        ``x.ravel`` will be returned.
    ord : {int, float, inf, -inf, 'fro', 'nuc'}, optional
        Order of the norm (see table under ``Notes`` for what values are
        supported for matrices and vectors respectively). inf means numpy's
        `inf` object. The default is None.
    axis : {None, int, 2-tuple of ints}, optional.
        If `axis` is an integer, it specifies the axis of `x` along which to
        compute the vector norms.  If `axis` is a 2-tuple, it specifies the
        axes that hold 2-D matrices, and the matrix norms of these matrices
        are computed.  If `axis` is None then either a vector norm (when `x`
        is 1-D) or a matrix norm (when `x` is 2-D) is returned. The default
        is None.

    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in the
        result as dimensions with size one.  With this option the result will
        broadcast correctly against the original `x`.

    Returns
    -------
    n : float or ndarray
        Norm of the matrix or vector(s).

    See Also
    --------
    scipy.linalg.norm : Similar function in SciPy.

    Notes
    -----
    For values of ``ord < 1``, the result is, strictly speaking, not a
    mathematical 'norm', but it may still be useful for various numerical
    purposes.

    The following norms can be calculated:

    =====  ============================  ==========================
    ord    norm for matrices             norm for vectors
    =====  ============================  ==========================
    None   Frobenius norm                2-norm
    'fro'  Frobenius norm                --
    'nuc'  nuclear norm                  --
    inf    max(sum(abs(x), axis=1))      max(abs(x))
    -inf   min(sum(abs(x), axis=1))      min(abs(x))
    0      --                            sum(x != 0)
    1      max(sum(abs(x), axis=0))      as below
    -1     min(sum(abs(x), axis=0))      as below
    2      2-norm (largest sing. value)  as below
    -2     smallest singular value       as below
    other  --                            sum(abs(x)**ord)**(1./ord)
    =====  ============================  ==========================

    The Frobenius norm is given by [1]_:

    :math:`||A||_F = [\\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

    The nuclear norm is the sum of the singular values.

    Both the Frobenius and nuclear norm orders are only defined for
    matrices and raise a ValueError when ``x.ndim != 2``.

    References
    ----------
    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
           Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15

    Examples
    --------

    >>> import numpy as np
    >>> from numpy import linalg as LA
    >>> a = np.arange(9) - 4
    >>> a
    array([-4, -3, -2, ...,  2,  3,  4])
    >>> b = a.reshape((3, 3))
    >>> b
    array([[-4, -3, -2],
           [-1,  0,  1],
           [ 2,  3,  4]])

    >>> LA.norm(a)
    7.745966692414834
    >>> LA.norm(b)
    7.745966692414834
    >>> LA.norm(b, 'fro')
    7.745966692414834
    >>> LA.norm(a, np.inf)
    4.0
    >>> LA.norm(b, np.inf)
    9.0
    >>> LA.norm(a, -np.inf)
    0.0
    >>> LA.norm(b, -np.inf)
    2.0

    >>> LA.norm(a, 1)
    20.0
    >>> LA.norm(b, 1)
    7.0
    >>> LA.norm(a, -1)
    -4.6566128774142013e-010
    >>> LA.norm(b, -1)
    6.0
    >>> LA.norm(a, 2)
    7.745966692414834
    >>> LA.norm(b, 2)
    7.3484692283495345

    >>> LA.norm(a, -2)
    0.0
    >>> LA.norm(b, -2)
    1.8570331885190563e-016 # may vary
    >>> LA.norm(a, 3)
    5.8480354764257312 # may vary
    >>> LA.norm(a, -3)
    0.0

    Using the `axis` argument to compute vector norms:

    >>> c = np.array([[ 1, 2, 3],
    ...               [-1, 1, 4]])
    >>> LA.norm(c, axis=0)
    array([ 1.41421356,  2.23606798,  5.        ])
    >>> LA.norm(c, axis=1)
    array([ 3.74165739,  4.24264069])
    >>> LA.norm(c, ord=1, axis=1)
    array([ 6.,  6.])

    Using the `axis` argument to compute matrix norms:

    >>> m = np.arange(8).reshape(2,2,2)
    >>> LA.norm(m, axis=(1,2))
    array([  3.74165739,  11.22497216])
    >>> LA.norm(m[0, :, :]), LA.norm(m[1, :, :])
    (3.7416573867739413, 11.224972160321824)

    """
    x = asarray(x)

    if not issubclass(x.dtype.type, (inexact, object_)):
        x = x.astype(float)

    # Immediately handle some default, simple, fast, and common cases.
    if axis is None:
        ndim = x.ndim
        if (
            (ord is None) or
            (ord in ('f', 'fro') and ndim == 2) or
            (ord == 2 and ndim == 1)
        ):
            x = x.ravel(order='K')
            if isComplexType(x.dtype.type):
                x_real = x.real
                x_imag = x.imag
                sqnorm = x_real.dot(x_real) + x_imag.dot(x_imag)
            else:
                sqnorm = x.dot(x)
            ret = sqrt(sqnorm)
            if keepdims:
                ret = ret.reshape(ndim*[1])
            return ret

    # Normalize the `axis` argument to a tuple.
    nd = x.ndim
    if axis is None:
        axis = tuple(range(nd))
    elif not isinstance(axis, tuple):
        try:
            axis = int(axis)
        except Exception as e:
            raise TypeError(
                "'axis' must be None, an integer or a tuple of integers"
            ) from e
        axis = (axis,)

    if len(axis) == 1:
        if ord == inf:
            return abs(x).max(axis=axis, keepdims=keepdims)
        elif ord == -inf:
            return abs(x).min(axis=axis, keepdims=keepdims)
        elif ord == 0:
            # Zero norm
            return (
                (x != 0)
                .astype(x.real.dtype)
                .sum(axis=axis, keepdims=keepdims)
            )
        elif ord == 1:
            # special case for speedup
            return add.reduce(abs(x), axis=axis, keepdims=keepdims)
        elif ord is None or ord == 2:
            # special case for speedup
            s = (x.conj() * x).real
            return sqrt(add.reduce(s, axis=axis, keepdims=keepdims))
        # None of the str-type keywords for ord ('fro', 'nuc')
        # are valid for vectors
        elif isinstance(ord, str):
            raise ValueError(f"Invalid norm order '{ord}' for vectors")
        else:
            absx = abs(x)
            absx **= ord
            ret = add.reduce(absx, axis=axis, keepdims=keepdims)
            ret **= reciprocal(ord, dtype=ret.dtype)
            return ret
    elif len(axis) == 2:
        row_axis, col_axis = axis
        row_axis = normalize_axis_index(row_axis, nd)
        col_axis = normalize_axis_index(col_axis, nd)
        if row_axis == col_axis:
            raise ValueError('Duplicate axes given.')
        if ord == 2:
            ret = _multi_svd_norm(x, row_axis, col_axis, amax)
        elif ord == -2:
            ret = _multi_svd_norm(x, row_axis, col_axis, amin)
        elif ord == 1:
            if col_axis > row_axis:
                col_axis -= 1
            ret = add.reduce(abs(x), axis=row_axis).max(axis=col_axis)
        elif ord == inf:
            if row_axis > col_axis:
                row_axis -= 1
            ret = add.reduce(abs(x), axis=col_axis).max(axis=row_axis)
        elif ord == -1:
            if col_axis > row_axis:
                col_axis -= 1
            ret = add.reduce(abs(x), axis=row_axis).min(axis=col_axis)
        elif ord == -inf:
            if row_axis > col_axis:
                row_axis -= 1
            ret = add.reduce(abs(x), axis=col_axis).min(axis=row_axis)
        elif ord in [None, 'fro', 'f']:
            ret = sqrt(add.reduce((x.conj() * x).real, axis=axis))
        elif ord == 'nuc':
            ret = _multi_svd_norm(x, row_axis, col_axis, sum)
        else:
            raise ValueError("Invalid norm order for matrices.")
        if keepdims:
            ret_shape = list(x.shape)
            ret_shape[axis[0]] = 1
            ret_shape[axis[1]] = 1
            ret = ret.reshape(ret_shape)
        return ret
    else:
        raise ValueError("Improper number of dimensions to norm.")


# multi_dot

def _multidot_dispatcher(arrays, *, out=None):
    yield from arrays
    yield out


@array_function_dispatch(_multidot_dispatcher)
def multi_dot(arrays, *, out=None):
    """
    Compute the dot product of two or more arrays in a single function call,
    while automatically selecting the fastest evaluation order.

    `multi_dot` chains `numpy.dot` and uses optimal parenthesization
    of the matrices [1]_ [2]_. Depending on the shapes of the matrices,
    this can speed up the multiplication a lot.

    If the first argument is 1-D it is treated as a row vector.
    If the last argument is 1-D it is treated as a column vector.
    The other arguments must be 2-D.

    Think of `multi_dot` as::

        def multi_dot(arrays): return functools.reduce(np.dot, arrays)


    Parameters
    ----------
    arrays : sequence of array_like
        If the first argument is 1-D it is treated as row vector.
        If the last argument is 1-D it is treated as column vector.
        The other arguments must be 2-D.
    out : ndarray, optional
        Output argument. This must have the exact kind that would be returned
        if it was not used. In particular, it must have the right type, must be
        C-contiguous, and its dtype must be the dtype that would be returned
        for `dot(a, b)`. This is a performance feature. Therefore, if these
        conditions are not met, an exception is raised, instead of attempting
        to be flexible.

    Returns
    -------
    output : ndarray
        Returns the dot product of the supplied arrays.

    See Also
    --------
    numpy.dot : dot multiplication with two arguments.

    References
    ----------

    .. [1] Cormen, "Introduction to Algorithms", Chapter 15.2, p. 370-378
    .. [2] https://en.wikipedia.org/wiki/Matrix_chain_multiplication

    Examples
    --------
    `multi_dot` allows you to write::

    >>> import numpy as np
    >>> from numpy.linalg import multi_dot
    >>> # Prepare some data
    >>> A = np.random.random((10000, 100))
    >>> B = np.random.random((100, 1000))
    >>> C = np.random.random((1000, 5))
    >>> D = np.random.random((5, 333))
    >>> # the actual dot multiplication
    >>> _ = multi_dot([A, B, C, D])

    instead of::

    >>> _ = np.dot(np.dot(np.dot(A, B), C), D)
    >>> # or
    >>> _ = A.dot(B).dot(C).dot(D)

    Notes
    -----
    The cost for a matrix multiplication can be calculated with the
    following function::

        def cost(A, B):
            return A.shape[0] * A.shape[1] * B.shape[1]

    Assume we have three matrices
    :math:`A_{10x100}, B_{100x5}, C_{5x50}`.

    The costs for the two different parenthesizations are as follows::

        cost((AB)C) = 10*100*5 + 10*5*50   = 5000 + 2500   = 7500
        cost(A(BC)) = 10*100*50 + 100*5*50 = 50000 + 25000 = 75000

    """
    n = len(arrays)
    # optimization only makes sense for len(arrays) > 2
    if n < 2:
        raise ValueError("Expecting at least two arrays.")
    elif n == 2:
        return dot(arrays[0], arrays[1], out=out)

    arrays = [asanyarray(a) for a in arrays]

    # save original ndim to reshape the result array into the proper form later
    ndim_first, ndim_last = arrays[0].ndim, arrays[-1].ndim
    # Explicitly convert vectors to 2D arrays to keep the logic of the internal
    # _multi_dot_* functions as simple as possible.
    if arrays[0].ndim == 1:
        arrays[0] = atleast_2d(arrays[0])
    if arrays[-1].ndim == 1:
        arrays[-1] = atleast_2d(arrays[-1]).T
    _assert_2d(*arrays)

    # _multi_dot_three is much faster than _multi_dot_matrix_chain_order
    if n == 3:
        result = _multi_dot_three(arrays[0], arrays[1], arrays[2], out=out)
    else:
        order = _multi_dot_matrix_chain_order(arrays)
        result = _multi_dot(arrays, order, 0, n - 1, out=out)

    # return proper shape
    if ndim_first == 1 and ndim_last == 1:
        return result[0, 0]  # scalar
    elif ndim_first == 1 or ndim_last == 1:
        return result.ravel()  # 1-D
    else:
        return result


def _multi_dot_three(A, B, C, out=None):
    """
    Find the best order for three arrays and do the multiplication.

    For three arguments `_multi_dot_three` is approximately 15 times faster
    than `_multi_dot_matrix_chain_order`

    """
    a0, a1b0 = A.shape
    b1c0, c1 = C.shape
    # cost1 = cost((AB)C) = a0*a1b0*b1c0 + a0*b1c0*c1
    cost1 = a0 * b1c0 * (a1b0 + c1)
    # cost2 = cost(A(BC)) = a1b0*b1c0*c1 + a0*a1b0*c1
    cost2 = a1b0 * c1 * (a0 + b1c0)

    if cost1 < cost2:
        return dot(dot(A, B), C, out=out)
    else:
        return dot(A, dot(B, C), out=out)


def _multi_dot_matrix_chain_order(arrays, return_costs=False):
    """
    Return a np.array that encodes the optimal order of multiplications.

    The optimal order array is then used by `_multi_dot()` to do the
    multiplication.

    Also return the cost matrix if `return_costs` is `True`

    The implementation CLOSELY follows Cormen, "Introduction to Algorithms",
    Chapter 15.2, p. 370-378.  Note that Cormen uses 1-based indices.

        cost[i, j] = min([
            cost[prefix] + cost[suffix] + cost_mult(prefix, suffix)
            for k in range(i, j)])

    """
    n = len(arrays)
    # p stores the dimensions of the matrices
    # Example for p: A_{10x100}, B_{100x5}, C_{5x50} --> p = [10, 100, 5, 50]
    p = [a.shape[0] for a in arrays] + [arrays[-1].shape[1]]
    # m is a matrix of costs of the subproblems
    # m[i,j]: min number of scalar multiplications needed to compute A_{i..j}
    m = zeros((n, n), dtype=double)
    # s is the actual ordering
    # s[i, j] is the value of k at which we split the product A_i..A_j
    s = empty((n, n), dtype=intp)

    for l in range(1, n):
        for i in range(n - l):
            j = i + l
            m[i, j] = inf
            for k in range(i, j):
                q = m[i, k] + m[k+1, j] + p[i]*p[k+1]*p[j+1]
                if q < m[i, j]:
                    m[i, j] = q
                    s[i, j] = k  # Note that Cormen uses 1-based index

    return (s, m) if return_costs else s


def _multi_dot(arrays, order, i, j, out=None):
    """Actually do the multiplication with the given order."""
    if i == j:
        # the initial call with non-None out should never get here
        assert out is None

        return arrays[i]
    else:
        return dot(_multi_dot(arrays, order, i, order[i, j]),
                   _multi_dot(arrays, order, order[i, j] + 1, j),
                   out=out)


# diagonal

def _diagonal_dispatcher(x, /, *, offset=None):
    return (x,)


@array_function_dispatch(_diagonal_dispatcher)
def diagonal(x, /, *, offset=0):
    """
    Returns specified diagonals of a matrix (or a stack of matrices) ``x``.

    This function is Array API compatible, contrary to
    :py:func:`numpy.diagonal`, the matrix is assumed
    to be defined by the last two dimensions.

    Parameters
    ----------
    x : (...,M,N) array_like
        Input array having shape (..., M, N) and whose innermost two
        dimensions form MxN matrices.
    offset : int, optional
        Offset specifying the off-diagonal relative to the main diagonal,
        where::

            * offset = 0: the main diagonal.
            * offset > 0: off-diagonal above the main diagonal.
            * offset < 0: off-diagonal below the main diagonal.

    Returns
    -------
    out : (...,min(N,M)) ndarray
        An array containing the diagonals and whose shape is determined by
        removing the last two dimensions and appending a dimension equal to
        the size of the resulting diagonals. The returned array must have
        the same data type as ``x``.

    See Also
    --------
    numpy.diagonal

    Examples
    --------
    >>> a = np.arange(4).reshape(2, 2); a
    array([[0, 1],
           [2, 3]])
    >>> np.linalg.diagonal(a)
    array([0, 3])

    A 3-D example:

    >>> a = np.arange(8).reshape(2, 2, 2); a
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> np.linalg.diagonal(a)
    array([[0, 3],
           [4, 7]])

    Diagonals adjacent to the main diagonal can be obtained by using the
    `offset` argument:

    >>> a = np.arange(9).reshape(3, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> np.linalg.diagonal(a, offset=1)  # First superdiagonal
    array([1, 5])
    >>> np.linalg.diagonal(a, offset=2)  # Second superdiagonal
    array([2])
    >>> np.linalg.diagonal(a, offset=-1)  # First subdiagonal
    array([3, 7])
    >>> np.linalg.diagonal(a, offset=-2)  # Second subdiagonal
    array([6])

    The anti-diagonal can be obtained by reversing the order of elements
    using either `numpy.flipud` or `numpy.fliplr`.

    >>> a = np.arange(9).reshape(3, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> np.linalg.diagonal(np.fliplr(a))  # Horizontal flip
    array([2, 4, 6])
    >>> np.linalg.diagonal(np.flipud(a))  # Vertical flip
    array([6, 4, 2])

    Note that the order in which the diagonal is retrieved varies depending
    on the flip function.

    """
    return _core_diagonal(x, offset, axis1=-2, axis2=-1)


# trace

def _trace_dispatcher(x, /, *, offset=None, dtype=None):
    return (x,)


@array_function_dispatch(_trace_dispatcher)
def trace(x, /, *, offset=0, dtype=None):
    """
    Returns the sum along the specified diagonals of a matrix
    (or a stack of matrices) ``x``.

    This function is Array API compatible, contrary to
    :py:func:`numpy.trace`.

    Parameters
    ----------
    x : (...,M,N) array_like
        Input array having shape (..., M, N) and whose innermost two
        dimensions form MxN matrices.
    offset : int, optional
        Offset specifying the off-diagonal relative to the main diagonal,
        where::

            * offset = 0: the main diagonal.
            * offset > 0: off-diagonal above the main diagonal.
            * offset < 0: off-diagonal below the main diagonal.

    dtype : dtype, optional
        Data type of the returned array.

    Returns
    -------
    out : ndarray
        An array containing the traces and whose shape is determined by
        removing the last two dimensions and storing the traces in the last
        array dimension. For example, if x has rank k and shape:
        (I, J, K, ..., L, M, N), then an output array has rank k-2 and shape:
        (I, J, K, ..., L) where::

            out[i, j, k, ..., l] = trace(a[i, j, k, ..., l, :, :])

        The returned array must have a data type as described by the dtype
        parameter above.

    See Also
    --------
    numpy.trace

    Examples
    --------
    >>> np.linalg.trace(np.eye(3))
    3.0
    >>> a = np.arange(8).reshape((2, 2, 2))
    >>> np.linalg.trace(a)
    array([3, 11])

    Trace is computed with the last two axes as the 2-d sub-arrays.
    This behavior differs from :py:func:`numpy.trace` which uses the first two
    axes by default.

    >>> a = np.arange(24).reshape((3, 2, 2, 2))
    >>> np.linalg.trace(a).shape
    (3, 2)

    Traces adjacent to the main diagonal can be obtained by using the
    `offset` argument:

    >>> a = np.arange(9).reshape((3, 3)); a
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> np.linalg.trace(a, offset=1)  # First superdiagonal
    6
    >>> np.linalg.trace(a, offset=2)  # Second superdiagonal
    2
    >>> np.linalg.trace(a, offset=-1)  # First subdiagonal
    10
    >>> np.linalg.trace(a, offset=-2)  # Second subdiagonal
    6

    """
    return _core_trace(x, offset, axis1=-2, axis2=-1, dtype=dtype)


# cross

def _cross_dispatcher(x1, x2, /, *, axis=None):
    return (x1, x2,)


@array_function_dispatch(_cross_dispatcher)
def cross(x1, x2, /, *, axis=-1):
    """
    Returns the cross product of 3-element vectors.

    If ``x1`` and/or ``x2`` are multi-dimensional arrays, then
    the cross-product of each pair of corresponding 3-element vectors
    is independently computed.

    This function is Array API compatible, contrary to
    :func:`numpy.cross`.

    Parameters
    ----------
    x1 : array_like
        The first input array.
    x2 : array_like
        The second input array. Must be compatible with ``x1`` for all
        non-compute axes. The size of the axis over which to compute
        the cross-product must be the same size as the respective axis
        in ``x1``.
    axis : int, optional
        The axis (dimension) of ``x1`` and ``x2`` containing the vectors for
        which to compute the cross-product. Default: ``-1``.

    Returns
    -------
    out : ndarray
        An array containing the cross products.

    See Also
    --------
    numpy.cross

    Examples
    --------
    Vector cross-product.

    >>> x = np.array([1, 2, 3])
    >>> y = np.array([4, 5, 6])
    >>> np.linalg.cross(x, y)
    array([-3,  6, -3])

    Multiple vector cross-products. Note that the direction of the cross
    product vector is defined by the *right-hand rule*.

    >>> x = np.array([[1,2,3], [4,5,6]])
    >>> y = np.array([[4,5,6], [1,2,3]])
    >>> np.linalg.cross(x, y)
    array([[-3,  6, -3],
           [ 3, -6,  3]])

    >>> x = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y = np.array([[4, 5], [6, 1], [2, 3]])
    >>> np.linalg.cross(x, y, axis=0)
    array([[-24,  6],
           [ 18, 24],
           [-6,  -18]])

    """
    x1 = asanyarray(x1)
    x2 = asanyarray(x2)

    if x1.shape[axis] != 3 or x2.shape[axis] != 3:
        raise ValueError(
            "Both input arrays must be (arrays of) 3-dimensional vectors, "
            f"but they are {x1.shape[axis]} and {x2.shape[axis]} "
            "dimensional instead."
        )

    return _core_cross(x1, x2, axis=axis)


# matmul

def _matmul_dispatcher(x1, x2, /):
    return (x1, x2)


@array_function_dispatch(_matmul_dispatcher)
def matmul(x1, x2, /):
    """
    Computes the matrix product.

    This function is Array API compatible, contrary to
    :func:`numpy.matmul`.

    Parameters
    ----------
    x1 : array_like
        The first input array.
    x2 : array_like
        The second input array.

    Returns
    -------
    out : ndarray
        The matrix product of the inputs.
        This is a scalar only when both ``x1``, ``x2`` are 1-d vectors.

    Raises
    ------
    ValueError
        If the last dimension of ``x1`` is not the same size as
        the second-to-last dimension of ``x2``.

        If a scalar value is passed in.

    See Also
    --------
    numpy.matmul

    Examples
    --------
    For 2-D arrays it is the matrix product:

    >>> a = np.array([[1, 0],
    ...               [0, 1]])
    >>> b = np.array([[4, 1],
    ...               [2, 2]])
    >>> np.linalg.matmul(a, b)
    array([[4, 1],
           [2, 2]])

    For 2-D mixed with 1-D, the result is the usual.

    >>> a = np.array([[1, 0],
    ...               [0, 1]])
    >>> b = np.array([1, 2])
    >>> np.linalg.matmul(a, b)
    array([1, 2])
    >>> np.linalg.matmul(b, a)
    array([1, 2])


    Broadcasting is conventional for stacks of arrays

    >>> a = np.arange(2 * 2 * 4).reshape((2, 2, 4))
    >>> b = np.arange(2 * 2 * 4).reshape((2, 4, 2))
    >>> np.linalg.matmul(a,b).shape
    (2, 2, 2)
    >>> np.linalg.matmul(a, b)[0, 1, 1]
    98
    >>> sum(a[0, 1, :] * b[0 , :, 1])
    98

    Vector, vector returns the scalar inner product, but neither argument
    is complex-conjugated:

    >>> np.linalg.matmul([2j, 3j], [2j, 3j])
    (-13+0j)

    Scalar multiplication raises an error.

    >>> np.linalg.matmul([1,2], 3)
    Traceback (most recent call last):
    ...
    ValueError: matmul: Input operand 1 does not have enough dimensions ...

    """
    return _core_matmul(x1, x2)


# tensordot

def _tensordot_dispatcher(x1, x2, /, *, axes=None):
    return (x1, x2)


@array_function_dispatch(_tensordot_dispatcher)
def tensordot(x1, x2, /, *, axes=2):
    return _core_tensordot(x1, x2, axes=axes)


tensordot.__doc__ = _core_tensordot.__doc__


# matrix_transpose

def _matrix_transpose_dispatcher(x):
    return (x,)

@array_function_dispatch(_matrix_transpose_dispatcher)
def matrix_transpose(x, /):
    return _core_matrix_transpose(x)


matrix_transpose.__doc__ = _core_matrix_transpose.__doc__


# matrix_norm

def _matrix_norm_dispatcher(x, /, *, keepdims=None, ord=None):
    return (x,)

@array_function_dispatch(_matrix_norm_dispatcher)
def matrix_norm(x, /, *, keepdims=False, ord="fro"):
    """
    Computes the matrix norm of a matrix (or a stack of matrices) ``x``.

    This function is Array API compatible.

    Parameters
    ----------
    x : array_like
        Input array having shape (..., M, N) and whose two innermost
        dimensions form ``MxN`` matrices.
    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in
        the result as dimensions with size one. Default: False.
    ord : {1, -1, 2, -2, inf, -inf, 'fro', 'nuc'}, optional
        The order of the norm. For details see the table under ``Notes``
        in `numpy.linalg.norm`.

    See Also
    --------
    numpy.linalg.norm : Generic norm function

    Examples
    --------
    >>> from numpy import linalg as LA
    >>> a = np.arange(9) - 4
    >>> a
    array([-4, -3, -2, ...,  2,  3,  4])
    >>> b = a.reshape((3, 3))
    >>> b
    array([[-4, -3, -2],
           [-1,  0,  1],
           [ 2,  3,  4]])

    >>> LA.matrix_norm(b)
    7.745966692414834
    >>> LA.matrix_norm(b, ord='fro')
    7.745966692414834
    >>> LA.matrix_norm(b, ord=np.inf)
    9.0
    >>> LA.matrix_norm(b, ord=-np.inf)
    2.0

    >>> LA.matrix_norm(b, ord=1)
    7.0
    >>> LA.matrix_norm(b, ord=-1)
    6.0
    >>> LA.matrix_norm(b, ord=2)
    7.3484692283495345
    >>> LA.matrix_norm(b, ord=-2)
    1.8570331885190563e-016 # may vary

    """
    x = asanyarray(x)
    return norm(x, axis=(-2, -1), keepdims=keepdims, ord=ord)


# vector_norm

def _vector_norm_dispatcher(x, /, *, axis=None, keepdims=None, ord=None):
    return (x,)

@array_function_dispatch(_vector_norm_dispatcher)
def vector_norm(x, /, *, axis=None, keepdims=False, ord=2):
    """
    Computes the vector norm of a vector (or batch of vectors) ``x``.

    This function is Array API compatible.

    Parameters
    ----------
    x : array_like
        Input array.
    axis : {None, int, 2-tuple of ints}, optional
        If an integer, ``axis`` specifies the axis (dimension) along which
        to compute vector norms. If an n-tuple, ``axis`` specifies the axes
        (dimensions) along which to compute batched vector norms. If ``None``,
        the vector norm must be computed over all array values (i.e.,
        equivalent to computing the vector norm of a flattened array).
        Default: ``None``.
    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in
        the result as dimensions with size one. Default: False.
    ord : {int, float, inf, -inf}, optional
        The order of the norm. For details see the table under ``Notes``
        in `numpy.linalg.norm`.

    See Also
    --------
    numpy.linalg.norm : Generic norm function

    Examples
    --------
    >>> from numpy import linalg as LA
    >>> a = np.arange(9) + 1
    >>> a
    array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> b = a.reshape((3, 3))
    >>> b
    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])

    >>> LA.vector_norm(b)
    16.881943016134134
    >>> LA.vector_norm(b, ord=np.inf)
    9.0
    >>> LA.vector_norm(b, ord=-np.inf)
    1.0

    >>> LA.vector_norm(b, ord=0)
    9.0
    >>> LA.vector_norm(b, ord=1)
    45.0
    >>> LA.vector_norm(b, ord=-1)
    0.3534857623790153
    >>> LA.vector_norm(b, ord=2)
    16.881943016134134
    >>> LA.vector_norm(b, ord=-2)
    0.8058837395885292

    """
    x = asanyarray(x)
    shape = list(x.shape)
    if axis is None:
        # Note: np.linalg.norm() doesn't handle 0-D arrays
        x = x.ravel()
        _axis = 0
    elif isinstance(axis, tuple):
        # Note: The axis argument supports any number of axes, whereas
        # np.linalg.norm() only supports a single axis for vector norm.
        normalized_axis = normalize_axis_tuple(axis, x.ndim)
        rest = tuple(i for i in range(x.ndim) if i not in normalized_axis)
        newshape = axis + rest
        x = _core_transpose(x, newshape).reshape(
            (
                prod([x.shape[i] for i in axis], dtype=int),
                *[x.shape[i] for i in rest]
            )
        )
        _axis = 0
    else:
        _axis = axis

    res = norm(x, axis=_axis, ord=ord)

    if keepdims:
        # We can't reuse np.linalg.norm(keepdims) because of the reshape hacks
        # above to avoid matrix norm logic.
        _axis = normalize_axis_tuple(
            range(len(shape)) if axis is None else axis, len(shape)
        )
        for i in _axis:
            shape[i] = 1
        res = res.reshape(tuple(shape))

    return res


# vecdot

def _vecdot_dispatcher(x1, x2, /, *, axis=None):
    return (x1, x2)

@array_function_dispatch(_vecdot_dispatcher)
def vecdot(x1, x2, /, *, axis=-1):
    """
    Computes the vector dot product.

    This function is restricted to arguments compatible with the Array API,
    contrary to :func:`numpy.vecdot`.

    Let :math:`\\mathbf{a}` be a vector in ``x1`` and :math:`\\mathbf{b}` be
    a corresponding vector in ``x2``. The dot product is defined as:

    .. math::
       \\mathbf{a} \\cdot \\mathbf{b} = \\sum_{i=0}^{n-1} \\overline{a_i}b_i

    over the dimension specified by ``axis`` and where :math:`\\overline{a_i}`
    denotes the complex conjugate if :math:`a_i` is complex and the identity
    otherwise.

    Parameters
    ----------
    x1 : array_like
        First input array.
    x2 : array_like
        Second input array.
    axis : int, optional
        Axis over which to compute the dot product. Default: ``-1``.

    Returns
    -------
    output : ndarray
        The vector dot product of the input.

    See Also
    --------
    numpy.vecdot

    Examples
    --------
    Get the projected size along a given normal for an array of vectors.

    >>> v = np.array([[0., 5., 0.], [0., 0., 10.], [0., 6., 8.]])
    >>> n = np.array([0., 0.6, 0.8])
    >>> np.linalg.vecdot(v, n)
    array([ 3.,  8., 10.])

    """
    return _core_vecdot(x1, x2, axis=axis)
<|endoftext|>"""
``numpy.linalg``
================

The NumPy linear algebra functions rely on BLAS and LAPACK to provide efficient
low level implementations of standard linear algebra algorithms. Those
libraries may be provided by NumPy itself using C versions of a subset of their
reference implementations but, when possible, highly optimized libraries that
take advantage of specialized processor functionality are preferred. Examples
of such libraries are OpenBLAS, MKL (TM), and ATLAS. Because those libraries
are multithreaded and processor dependent, environmental variables and external
packages such as threadpoolctl may be needed to control the number of threads
or specify the processor architecture.

- OpenBLAS: https://www.openblas.net/
- threadpoolctl: https://github.com/joblib/threadpoolctl

Please note that the most-used linear algebra functions in NumPy are present in
the main ``numpy`` namespace rather than in ``numpy.linalg``.  There are:
``dot``, ``vdot``, ``inner``, ``outer``, ``matmul``, ``tensordot``, ``einsum``,
``einsum_path`` and ``kron``.

Functions present in numpy.linalg are listed below.


Matrix and vector products
--------------------------

   cross
   multi_dot
   matrix_power
   tensordot
   matmul

Decompositions
--------------

   cholesky
   outer
   qr
   svd
   svdvals

Matrix eigenvalues
------------------

   eig
   eigh
   eigvals
   eigvalsh

Norms and other numbers
-----------------------

   norm
   matrix_norm
   vector_norm
   cond
   det
   matrix_rank
   slogdet
   trace (Array API compatible)

Solving equations and inverting matrices
----------------------------------------

   solve
   tensorsolve
   lstsq
   inv
   pinv
   tensorinv

Other matrix operations
-----------------------

   diagonal (Array API compatible)
   matrix_transpose (Array API compatible)

Exceptions
----------

   LinAlgError

"""
# To get sub-modules
from . import linalg  # deprecated in NumPy 2.0
from . import _linalg
from ._linalg import *

__all__ = _linalg.__all__.copy()

from numpy._pytesttester import PytestTester
test = PytestTester(__name__)
del PytestTester
<|endoftext|>#
# Module which deals with pickling of objects.
#
# multiprocessing/reduction.py
#
# Copyright (c) 2006-2008, R Oudkerk
# Licensed to PSF under a Contributor Agreement.
#

from abc import ABCMeta
import copyreg
import functools
import io
import os
try:
    import dill as pickle
except ImportError:
    import pickle
import socket
import sys

from . import context

__all__ = ['send_handle', 'recv_handle', 'ForkingPickler', 'register', 'dump']


HAVE_SEND_HANDLE = (sys.platform == 'win32' or
                    (hasattr(socket, 'CMSG_LEN') and
                     hasattr(socket, 'SCM_RIGHTS') and
                     hasattr(socket.socket, 'sendmsg')))

#
# Pickler subclass
#

class ForkingPickler(pickle.Pickler):
    '''Pickler subclass used by multiprocess.'''
    _extra_reducers = {}
    _copyreg_dispatch_table = copyreg.dispatch_table

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.dispatch_table = self._copyreg_dispatch_table.copy()
        self.dispatch_table.update(self._extra_reducers)

    @classmethod
    def register(cls, type, reduce):
        '''Register a reduce function for a type.'''
        cls._extra_reducers[type] = reduce

    @classmethod
    def dumps(cls, obj, protocol=None, *args, **kwds):
        buf = io.BytesIO()
        cls(buf, protocol, *args, **kwds).dump(obj)
        return buf.getbuffer()

    loads = pickle.loads

register = ForkingPickler.register

def dump(obj, file, protocol=None, *args, **kwds):
    '''Replacement for pickle.dump() using ForkingPickler.'''
    ForkingPickler(file, protocol, *args, **kwds).dump(obj)

#
# Platform specific definitions
#

if sys.platform == 'win32':
    # Windows
    __all__ += ['DupHandle', 'duplicate', 'steal_handle']
    import _winapi

    def duplicate(handle, target_process=None, inheritable=False,
                  *, source_process=None):
        '''Duplicate a handle.  (target_process is a handle not a pid!)'''
        current_process = _winapi.GetCurrentProcess()
        if source_process is None:
            source_process = current_process
        if target_process is None:
            target_process = current_process
        return _winapi.DuplicateHandle(
            source_process, handle, target_process,
            0, inheritable, _winapi.DUPLICATE_SAME_ACCESS)

    def steal_handle(source_pid, handle):
        '''Steal a handle from process identified by source_pid.'''
        source_process_handle = _winapi.OpenProcess(
            _winapi.PROCESS_DUP_HANDLE, False, source_pid)
        try:
            return _winapi.DuplicateHandle(
                source_process_handle, handle,
                _winapi.GetCurrentProcess(), 0, False,
                _winapi.DUPLICATE_SAME_ACCESS | _winapi.DUPLICATE_CLOSE_SOURCE)
        finally:
            _winapi.CloseHandle(source_process_handle)

    def send_handle(conn, handle, destination_pid):
        '''Send a handle over a local connection.'''
        dh = DupHandle(handle, _winapi.DUPLICATE_SAME_ACCESS, destination_pid)
        conn.send(dh)

    def recv_handle(conn):
        '''Receive a handle over a local connection.'''
        return conn.recv().detach()

    class DupHandle(object):
        '''Picklable wrapper for a handle.'''
        def __init__(self, handle, access, pid=None):
            if pid is None:
                # We just duplicate the handle in the current process and
                # let the receiving process steal the handle.
                pid = os.getpid()
            proc = _winapi.OpenProcess(_winapi.PROCESS_DUP_HANDLE, False, pid)
            try:
                self._handle = _winapi.DuplicateHandle(
                    _winapi.GetCurrentProcess(),
                    handle, proc, access, False, 0)
            finally:
                _winapi.CloseHandle(proc)
            self._access = access
            self._pid = pid

        def detach(self):
            '''Get the handle.  This should only be called once.'''
            # retrieve handle from process which currently owns it
            if self._pid == os.getpid():
                # The handle has already been duplicated for this process.
                return self._handle
            # We must steal the handle from the process whose pid is self._pid.
            proc = _winapi.OpenProcess(_winapi.PROCESS_DUP_HANDLE, False,
                                       self._pid)
            try:
                return _winapi.DuplicateHandle(
                    proc, self._handle, _winapi.GetCurrentProcess(),
                    self._access, False, _winapi.DUPLICATE_CLOSE_SOURCE)
            finally:
                _winapi.CloseHandle(proc)

else:
    # Unix
    __all__ += ['DupFd', 'sendfds', 'recvfds']
    import array

    # On MacOSX we should acknowledge receipt of fds -- see Issue14669
    ACKNOWLEDGE = sys.platform == 'darwin'

    def sendfds(sock, fds):
        '''Send an array of fds over an AF_UNIX socket.'''
        fds = array.array('i', fds)
        msg = bytes([len(fds) % 256])
        sock.sendmsg([msg], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds)])
        if ACKNOWLEDGE and sock.recv(1) != b'A':
            raise RuntimeError('did not receive acknowledgement of fd')

    def recvfds(sock, size):
        '''Receive an array of fds over an AF_UNIX socket.'''
        a = array.array('i')
        bytes_size = a.itemsize * size
        msg, ancdata, flags, addr = sock.recvmsg(1, socket.CMSG_SPACE(bytes_size))
        if not msg and not ancdata:
            raise EOFError
        try:
            if ACKNOWLEDGE:
                sock.send(b'A')
            if len(ancdata) != 1:
                raise RuntimeError('received %d items of ancdata' %
                                   len(ancdata))
            cmsg_level, cmsg_type, cmsg_data = ancdata[0]
            if (cmsg_level == socket.SOL_SOCKET and
                cmsg_type == socket.SCM_RIGHTS):
                if len(cmsg_data) % a.itemsize != 0:
                    raise ValueError
                a.frombytes(cmsg_data)
                if len(a) % 256 != msg[0]:
                    raise AssertionError(
                        "Len is {0:n} but msg[0] is {1!r}".format(
                            len(a), msg[0]))
                return list(a)
        except (ValueError, IndexError):
            pass
        raise RuntimeError('Invalid data received')

    def send_handle(conn, handle, destination_pid):
        '''Send a handle over a local connection.'''
        with socket.fromfd(conn.fileno(), socket.AF_UNIX, socket.SOCK_STREAM) as s:
            sendfds(s, [handle])

    def recv_handle(conn):
        '''Receive a handle over a local connection.'''
        with socket.fromfd(conn.fileno(), socket.AF_UNIX, socket.SOCK_STREAM) as s:
            return recvfds(s, 1)[0]

    def DupFd(fd):
        '''Return a wrapper for an fd.'''
        popen_obj = context.get_spawning_popen()
        if popen_obj is not None:
            return popen_obj.DupFd(popen_obj.duplicate_for_child(fd))
        elif HAVE_SEND_HANDLE:
            from . import resource_sharer
            return resource_sharer.DupFd(fd)
        else:
            raise ValueError('SCM_RIGHTS appears not to be available')

#
# Try making some callable types picklable
#

def _reduce_method(m):
    if m.__self__ is None:
        return getattr, (m.__class__, m.__func__.__name__)
    else:
        return getattr, (m.__self__, m.__func__.__name__)
class _C:
    def f(self):
        pass
register(type(_C().f), _reduce_method)


def _reduce_method_descriptor(m):
    return getattr, (m.__objclass__, m.__name__)
register(type(list.append), _reduce_method_descriptor)
register(type(int.__add__), _reduce_method_descriptor)


def _reduce_partial(p):
    return _rebuild_partial, (p.func, p.args, p.keywords or {})
def _rebuild_partial(func, args, keywords):
    return functools.partial(func, *args, **keywords)
register(functools.partial, _reduce_partial)

#
# Make sockets picklable
#

if sys.platform == 'win32':
    def _reduce_socket(s):
        from .resource_sharer import DupSocket
        return _rebuild_socket, (DupSocket(s),)
    def _rebuild_socket(ds):
        return ds.detach()
    register(socket.socket, _reduce_socket)

else:
    def _reduce_socket(s):
        df = DupFd(s.fileno())
        return _rebuild_socket, (df, s.family, s.type, s.proto)
    def _rebuild_socket(df, family, type, proto):
        fd = df.detach()
        return socket.socket(family, type, proto, fileno=fd)
    register(socket.socket, _reduce_socket)


class AbstractReducer(metaclass=ABCMeta):
    '''Abstract base class for use in implementing a Reduction class
    suitable for use in replacing the standard reduction mechanism
    used in multiprocess.'''
    ForkingPickler = ForkingPickler
    register = register
    dump = dump
    send_handle = send_handle
    recv_handle = recv_handle

    if sys.platform == 'win32':
        steal_handle = steal_handle
        duplicate = duplicate
        DupHandle = DupHandle
    else:
        sendfds = sendfds
        recvfds = recvfds
        DupFd = DupFd

    _reduce_method = _reduce_method
    _reduce_method_descriptor = _reduce_method_descriptor
    _rebuild_partial = _rebuild_partial
    _reduce_socket = _reduce_socket
    _rebuild_socket = _rebuild_socket

    def __init__(self, *args):
        register(type(_C().f), _reduce_method)
        register(type(list.append), _reduce_method_descriptor)
        register(type(int.__add__), _reduce_method_descriptor)
        register(functools.partial, _reduce_partial)
        register(socket.socket, _reduce_socket)
<|endoftext|>import io
import os

from .context import reduction, set_spawning_popen
from . import popen_fork
from . import spawn
from . import util

__all__ = ['Popen']


#
# Wrapper for an fd used while launching a process
#

class _DupFd(object):
    def __init__(self, fd):
        self.fd = fd
    def detach(self):
        return self.fd

#
# Start child process using a fresh interpreter
#

class Popen(popen_fork.Popen):
    method = 'spawn'
    DupFd = _DupFd

    def __init__(self, process_obj):
        self._fds = []
        super().__init__(process_obj)

    def duplicate_for_child(self, fd):
        self._fds.append(fd)
        return fd

    def _launch(self, process_obj):
        from . import resource_tracker
        tracker_fd = resource_tracker.getfd()
        self._fds.append(tracker_fd)
        prep_data = spawn.get_preparation_data(process_obj._name)
        fp = io.BytesIO()
        set_spawning_popen(self)
        try:
            reduction.dump(prep_data, fp)
            reduction.dump(process_obj, fp)
        finally:
            set_spawning_popen(None)

        parent_r = child_w = child_r = parent_w = None
        try:
            parent_r, child_w = os.pipe()
            child_r, parent_w = os.pipe()
            cmd = spawn.get_command_line(tracker_fd=tracker_fd,
                                         pipe_handle=child_r)
            self._fds.extend([child_r, child_w])
            self.pid = util.spawnv_passfds(spawn.get_executable(),
                                           cmd, self._fds)
            self.sentinel = parent_r
            with open(parent_w, 'wb', closefd=False) as f:
                f.write(fp.getbuffer())
        finally:
            fds_to_close = []
            for fd in (parent_r, parent_w):
                if fd is not None:
                    fds_to_close.append(fd)
            self.finalizer = util.Finalize(self, util.close_fds, fds_to_close)

            for fd in (child_r, child_w):
                if fd is not None:
                    os.close(fd)
<|endoftext|>import os
import sys
import threading

from . import process
from . import reduction

__all__ = ()

#
# Exceptions
#

class ProcessError(Exception):
    pass

class BufferTooShort(ProcessError):
    pass

class TimeoutError(ProcessError):
    pass

class AuthenticationError(ProcessError):
    pass

#
# Base type for contexts. Bound methods of an instance of this type are included in __all__ of __init__.py
#

class BaseContext(object):

    ProcessError = ProcessError
    BufferTooShort = BufferTooShort
    TimeoutError = TimeoutError
    AuthenticationError = AuthenticationError

    current_process = staticmethod(process.current_process)
    parent_process = staticmethod(process.parent_process)
    active_children = staticmethod(process.active_children)

    def cpu_count(self):
        '''Returns the number of CPUs in the system'''
        num = os.cpu_count()
        if num is None:
            raise NotImplementedError('cannot determine number of cpus')
        else:
            return num

    def Manager(self):
        '''Returns a manager associated with a running server process

        The managers methods such as `Lock()`, `Condition()` and `Queue()`
        can be used to create shared objects.
        '''
        from .managers import SyncManager
        m = SyncManager(ctx=self.get_context())
        m.start()
        return m

    def Pipe(self, duplex=True):
        '''Returns two connection object connected by a pipe'''
        from .connection import Pipe
        return Pipe(duplex)

    def Lock(self):
        '''Returns a non-recursive lock object'''
        from .synchronize import Lock
        return Lock(ctx=self.get_context())

    def RLock(self):
        '''Returns a recursive lock object'''
        from .synchronize import RLock
        return RLock(ctx=self.get_context())

    def Condition(self, lock=None):
        '''Returns a condition object'''
        from .synchronize import Condition
        return Condition(lock, ctx=self.get_context())

    def Semaphore(self, value=1):
        '''Returns a semaphore object'''
        from .synchronize import Semaphore
        return Semaphore(value, ctx=self.get_context())

    def BoundedSemaphore(self, value=1):
        '''Returns a bounded semaphore object'''
        from .synchronize import BoundedSemaphore
        return BoundedSemaphore(value, ctx=self.get_context())

    def Event(self):
        '''Returns an event object'''
        from .synchronize import Event
        return Event(ctx=self.get_context())

    def Barrier(self, parties, action=None, timeout=None):
        '''Returns a barrier object'''
        from .synchronize import Barrier
        return Barrier(parties, action, timeout, ctx=self.get_context())

    def Queue(self, maxsize=0):
        '''Returns a queue object'''
        from .queues import Queue
        return Queue(maxsize, ctx=self.get_context())

    def JoinableQueue(self, maxsize=0):
        '''Returns a queue object'''
        from .queues import JoinableQueue
        return JoinableQueue(maxsize, ctx=self.get_context())

    def SimpleQueue(self):
        '''Returns a queue object'''
        from .queues import SimpleQueue
        return SimpleQueue(ctx=self.get_context())

    def Pool(self, processes=None, initializer=None, initargs=(),
             maxtasksperchild=None):
        '''Returns a process pool object'''
        from .pool import Pool
        return Pool(processes, initializer, initargs, maxtasksperchild,
                    context=self.get_context())

    def RawValue(self, typecode_or_type, *args):
        '''Returns a shared object'''
        from .sharedctypes import RawValue
        return RawValue(typecode_or_type, *args)

    def RawArray(self, typecode_or_type, size_or_initializer):
        '''Returns a shared array'''
        from .sharedctypes import RawArray
        return RawArray(typecode_or_type, size_or_initializer)

    def Value(self, typecode_or_type, *args, lock=True):
        '''Returns a synchronized shared object'''
        from .sharedctypes import Value
        return Value(typecode_or_type, *args, lock=lock,
                     ctx=self.get_context())

    def Array(self, typecode_or_type, size_or_initializer, *, lock=True):
        '''Returns a synchronized shared array'''
        from .sharedctypes import Array
        return Array(typecode_or_type, size_or_initializer, lock=lock,
                     ctx=self.get_context())

    def freeze_support(self):
        '''Check whether this is a fake forked process in a frozen executable.
        If so then run code specified by commandline and exit.
        '''
        if sys.platform == 'win32' and getattr(sys, 'frozen', False):
            from .spawn import freeze_support
            freeze_support()

    def get_logger(self):
        '''Return package logger -- if it does not already exist then
        it is created.
        '''
        from .util import get_logger
        return get_logger()

    def log_to_stderr(self, level=None):
        '''Turn on logging and add a handler which prints to stderr'''
        from .util import log_to_stderr
        return log_to_stderr(level)

    def allow_connection_pickling(self):
        '''Install support for sending connections and sockets
        between processes
        '''
        # This is undocumented.  In previous versions of multiprocessing
        # its only effect was to make socket objects inheritable on Windows.
        from . import connection

    def set_executable(self, executable):
        '''Sets the path to a python.exe or pythonw.exe binary used to run
        child processes instead of sys.executable when using the 'spawn'
        start method.  Useful for people embedding Python.
        '''
        from .spawn import set_executable
        set_executable(executable)

    def set_forkserver_preload(self, module_names):
        '''Set list of module names to try to load in forkserver process.
        This is really just a hint.
        '''
        from .forkserver import set_forkserver_preload
        set_forkserver_preload(module_names)

    def get_context(self, method=None):
        if method is None:
            return self
        try:
            ctx = _concrete_contexts[method]
        except KeyError:
            raise ValueError('cannot find context for %r' % method) from None
        ctx._check_available()
        return ctx

    def get_start_method(self, allow_none=False):
        return self._name

    def set_start_method(self, method, force=False):
        raise ValueError('cannot set start method of concrete context')

    @property
    def reducer(self):
        '''Controls how objects will be reduced to a form that can be
        shared with other processes.'''
        return globals().get('reduction')

    @reducer.setter
    def reducer(self, reduction):
        globals()['reduction'] = reduction

    def _check_available(self):
        pass

#
# Type of default context -- underlying context can be set at most once
#

class Process(process.BaseProcess):
    _start_method = None
    @staticmethod
    def _Popen(process_obj):
        return _default_context.get_context().Process._Popen(process_obj)

    @staticmethod
    def _after_fork():
        return _default_context.get_context().Process._after_fork()

class DefaultContext(BaseContext):
    Process = Process

    def __init__(self, context):
        self._default_context = context
        self._actual_context = None

    def get_context(self, method=None):
        if method is None:
            if self._actual_context is None:
                self._actual_context = self._default_context
            return self._actual_context
        else:
            return super().get_context(method)

    def set_start_method(self, method, force=False):
        if self._actual_context is not None and not force:
            raise RuntimeError('context has already been set')
        if method is None and force:
            self._actual_context = None
            return
        self._actual_context = self.get_context(method)

    def get_start_method(self, allow_none=False):
        if self._actual_context is None:
            if allow_none:
                return None
            self._actual_context = self._default_context
        return self._actual_context._name

    def get_all_start_methods(self):
        if sys.platform == 'win32':
            return ['spawn']
        else:
            methods = ['spawn', 'fork'] if sys.platform == 'darwin' else ['fork', 'spawn']
            if reduction.HAVE_SEND_HANDLE:
                methods.append('forkserver')
            return methods


#
# Context types for fixed start method
#

if sys.platform != 'win32':

    class ForkProcess(process.BaseProcess):
        _start_method = 'fork'
        @staticmethod
        def _Popen(process_obj):
            from .popen_fork import Popen
            return Popen(process_obj)

    class SpawnProcess(process.BaseProcess):
        _start_method = 'spawn'
        @staticmethod
        def _Popen(process_obj):
            from .popen_spawn_posix import Popen
            return Popen(process_obj)

        @staticmethod
        def _after_fork():
            # process is spawned, nothing to do
            pass

    class ForkServerProcess(process.BaseProcess):
        _start_method = 'forkserver'
        @staticmethod
        def _Popen(process_obj):
            from .popen_forkserver import Popen
            return Popen(process_obj)

    class ForkContext(BaseContext):
        _name = 'fork'
        Process = ForkProcess

    class SpawnContext(BaseContext):
        _name = 'spawn'
        Process = SpawnProcess

    class ForkServerContext(BaseContext):
        _name = 'forkserver'
        Process = ForkServerProcess
        def _check_available(self):
            if not reduction.HAVE_SEND_HANDLE:
                raise ValueError('forkserver start method not available')

    _concrete_contexts = {
        'fork': ForkContext(),
        'spawn': SpawnContext(),
        'forkserver': ForkServerContext(),
    }
    if sys.platform == 'darwin':
        # bpo-33725: running arbitrary code after fork() is no longer reliable
        # on macOS since macOS 10.14 (Mojave). Use spawn by default instead.
        _default_context = DefaultContext(_concrete_contexts['fork']) #FIXME: spawn
    else:
        _default_context = DefaultContext(_concrete_contexts['fork'])

else:

    class SpawnProcess(process.BaseProcess):
        _start_method = 'spawn'
        @staticmethod
        def _Popen(process_obj):
            from .popen_spawn_win32 import Popen
            return Popen(process_obj)

        @staticmethod
        def _after_fork():
            # process is spawned, nothing to do
            pass

    class SpawnContext(BaseContext):
        _name = 'spawn'
        Process = SpawnProcess

    _concrete_contexts = {
        'spawn': SpawnContext(),
    }
    _default_context = DefaultContext(_concrete_contexts['spawn'])

#
# Force the start method
#

def _force_start_method(method):
    _default_context._actual_context = _concrete_contexts[method]

#
# Check that the current thread is spawning a child process
#

_tls = threading.local()

def get_spawning_popen():
    return getattr(_tls, 'spawning_popen', None)

def set_spawning_popen(popen):
    _tls.spawning_popen = popen

def assert_spawning(obj):
    if get_spawning_popen() is None:
        raise RuntimeError(
            '%s objects should only be shared between processes'
            ' through inheritance' % type(obj).__name__
            )
<|endoftext|>import os
import signal

from . import util

__all__ = ['Popen']

#
# Start child process using fork
#

class Popen(object):
    method = 'fork'

    def __init__(self, process_obj):
        util._flush_std_streams()
        self.returncode = None
        self.finalizer = None
        self._launch(process_obj)

    def duplicate_for_child(self, fd):
        return fd

    def poll(self, flag=os.WNOHANG):
        if self.returncode is None:
            try:
                pid, sts = os.waitpid(self.pid, flag)
            except OSError:
                # Child process not yet created. See #1731717
                # e.errno == errno.ECHILD == 10
                return None
            if pid == self.pid:
                self.returncode = os.waitstatus_to_exitcode(sts)
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            if timeout is not None:
                from multiprocess.connection import wait
                if not wait([self.sentinel], timeout):
                    return None
            # This shouldn't block if wait() returned successfully.
            return self.poll(os.WNOHANG if timeout == 0.0 else 0)
        return self.returncode

    def _send_signal(self, sig):
        if self.returncode is None:
            try:
                os.kill(self.pid, sig)
            except ProcessLookupError:
                pass
            except OSError:
                if self.wait(timeout=0.1) is None:
                    raise

    def terminate(self):
        self._send_signal(signal.SIGTERM)

    def kill(self):
        self._send_signal(signal.SIGKILL)

    def _launch(self, process_obj):
        code = 1
        parent_r, child_w = os.pipe()
        child_r, parent_w = os.pipe()
        self.pid = os.fork()
        if self.pid == 0:
            try:
                os.close(parent_r)
                os.close(parent_w)
                code = process_obj._bootstrap(parent_sentinel=child_r)
            finally:
                os._exit(code)
        else:
            os.close(child_w)
            os.close(child_r)
            self.finalizer = util.Finalize(self, util.close_fds,
                                           (parent_r, parent_w,))
            self.sentinel = parent_r

    def close(self):
        if self.finalizer is not None:
            self.finalizer()
<|endoftext|>#
# Module providing manager classes for dealing
# with shared objects
#
# multiprocessing/managers.py
#
# Copyright (c) 2006-2008, R Oudkerk
# Licensed to PSF under a Contributor Agreement.
#

__all__ = [ 'BaseManager', 'SyncManager', 'BaseProxy', 'Token' ]

#
# Imports
#

import sys
import threading
import signal
import array
import queue
import time
import types
import os
from os import getpid

from traceback import format_exc

from . import connection
from .context import reduction, get_spawning_popen, ProcessError
from . import pool
from . import process
from . import util
from . import get_context
try:
    from . import shared_memory
except ImportError:
    HAS_SHMEM = False
else:
    HAS_SHMEM = True
    __all__.append('SharedMemoryManager')

#
# Register some things for pickling
#

def reduce_array(a):
    return array.array, (a.typecode, a.tobytes())
reduction.register(array.array, reduce_array)

view_types = [type(getattr({}, name)()) for name in ('items','keys','values')]
if view_types[0] is not list:       # only needed in Py3.0
    def rebuild_as_list(obj):
        return list, (list(obj),)
    for view_type in view_types:
        reduction.register(view_type, rebuild_as_list)

#
# Type for identifying shared objects
#

class Token(object):
    '''
    Type to uniquely identify a shared object
    '''
    __slots__ = ('typeid', 'address', 'id')

    def __init__(self, typeid, address, id):
        (self.typeid, self.address, self.id) = (typeid, address, id)

    def __getstate__(self):
        return (self.typeid, self.address, self.id)

    def __setstate__(self, state):
        (self.typeid, self.address, self.id) = state

    def __repr__(self):
        return '%s(typeid=%r, address=%r, id=%r)' % \
               (self.__class__.__name__, self.typeid, self.address, self.id)

#
# Function for communication with a manager's server process
#

def dispatch(c, id, methodname, args=(), kwds={}):
    '''
    Send a message to manager using connection `c` and return response
    '''
    c.send((id, methodname, args, kwds))
    kind, result = c.recv()
    if kind == '#RETURN':
        return result
    raise convert_to_error(kind, result)

def convert_to_error(kind, result):
    if kind == '#ERROR':
        return result
    elif kind in ('#TRACEBACK', '#UNSERIALIZABLE'):
        if not isinstance(result, str):
            raise TypeError(
                "Result {0!r} (kind '{1}') type is {2}, not str".format(
                    result, kind, type(result)))
        if kind == '#UNSERIALIZABLE':
            return RemoteError('Unserializable message: %s\n' % result)
        else:
            return RemoteError(result)
    else:
        return ValueError('Unrecognized message type {!r}'.format(kind))

class RemoteError(Exception):
    def __str__(self):
        return ('\n' + '-'*75 + '\n' + str(self.args[0]) + '-'*75)

#
# Functions for finding the method names of an object
#

def all_methods(obj):
    '''
    Return a list of names of methods of `obj`
    '''
    temp = []
    for name in dir(obj):
        func = getattr(obj, name)
        if callable(func):
            temp.append(name)
    return temp

def public_methods(obj):
    '''
    Return a list of names of methods of `obj` which do not start with '_'
    '''
    return [name for name in all_methods(obj) if name[0] != '_']

#
# Server which is run in a process controlled by a manager
#

class Server(object):
    '''
    Server class which runs in a process controlled by a manager object
    '''
    public = ['shutdown', 'create', 'accept_connection', 'get_methods',
              'debug_info', 'number_of_objects', 'dummy', 'incref', 'decref']

    def __init__(self, registry, address, authkey, serializer):
        if not isinstance(authkey, bytes):
            raise TypeError(
                "Authkey {0!r} is type {1!s}, not bytes".format(
                    authkey, type(authkey)))
        self.registry = registry
        self.authkey = process.AuthenticationString(authkey)
        Listener, Client = listener_client[serializer]

        # do authentication later
        self.listener = Listener(address=address, backlog=16)
        self.address = self.listener.address

        self.id_to_obj = {'0': (None, ())}
        self.id_to_refcount = {}
        self.id_to_local_proxy_obj = {}
        self.mutex = threading.Lock()

    def serve_forever(self):
        '''
        Run the server forever
        '''
        self.stop_event = threading.Event()
        process.current_process()._manager_server = self
        try:
            accepter = threading.Thread(target=self.accepter)
            accepter.daemon = True
            accepter.start()
            try:
                while not self.stop_event.is_set():
                    self.stop_event.wait(1)
            except (KeyboardInterrupt, SystemExit):
                pass
        finally:
            if sys.stdout != sys.__stdout__: # what about stderr?
                util.debug('resetting stdout, stderr')
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
            sys.exit(0)

    def accepter(self):
        while True:
            try:
                c = self.listener.accept()
            except OSError:
                continue
            t = threading.Thread(target=self.handle_request, args=(c,))
            t.daemon = True
            t.start()

    def _handle_request(self, c):
        request = None
        try:
            connection.deliver_challenge(c, self.authkey)
            connection.answer_challenge(c, self.authkey)
            request = c.recv()
            ignore, funcname, args, kwds = request
            assert funcname in self.public, '%r unrecognized' % funcname
            func = getattr(self, funcname)
        except Exception:
            msg = ('#TRACEBACK', format_exc())
        else:
            try:
                result = func(c, *args, **kwds)
            except Exception:
                msg = ('#TRACEBACK', format_exc())
            else:
                msg = ('#RETURN', result)

        try:
            c.send(msg)
        except Exception as e:
            try:
                c.send(('#TRACEBACK', format_exc()))
            except Exception:
                pass
            util.info('Failure to send message: %r', msg)
            util.info(' ... request was %r', request)
            util.info(' ... exception was %r', e)

    def handle_request(self, conn):
        '''
        Handle a new connection
        '''
        try:
            self._handle_request(conn)
        except SystemExit:
            # Server.serve_client() calls sys.exit(0) on EOF
            pass
        finally:
            conn.close()

    def serve_client(self, conn):
        '''
        Handle requests from the proxies in a particular process/thread
        '''
        util.debug('starting server thread to service %r',
                   threading.current_thread().name)

        recv = conn.recv
        send = conn.send
        id_to_obj = self.id_to_obj

        while not self.stop_event.is_set():

            try:
                methodname = obj = None
                request = recv()
                ident, methodname, args, kwds = request
                try:
                    obj, exposed, gettypeid = id_to_obj[ident]
                except KeyError as ke:
                    try:
                        obj, exposed, gettypeid = \
                            self.id_to_local_proxy_obj[ident]
                    except KeyError:
                        raise ke

                if methodname not in exposed:
                    raise AttributeError(
                        'method %r of %r object is not in exposed=%r' %
                        (methodname, type(obj), exposed)
                        )

                function = getattr(obj, methodname)

                try:
                    res = function(*args, **kwds)
                except Exception as e:
                    msg = ('#ERROR', e)
                else:
                    typeid = gettypeid and gettypeid.get(methodname, None)
                    if typeid:
                        rident, rexposed = self.create(conn, typeid, res)
                        token = Token(typeid, self.address, rident)
                        msg = ('#PROXY', (rexposed, token))
                    else:
                        msg = ('#RETURN', res)

            except AttributeError:
                if methodname is None:
                    msg = ('#TRACEBACK', format_exc())
                else:
                    try:
                        fallback_func = self.fallback_mapping[methodname]
                        result = fallback_func(
                            self, conn, ident, obj, *args, **kwds
                            )
                        msg = ('#RETURN', result)
                    except Exception:
                        msg = ('#TRACEBACK', format_exc())

            except EOFError:
                util.debug('got EOF -- exiting thread serving %r',
                           threading.current_thread().name)
                sys.exit(0)

            except Exception:
                msg = ('#TRACEBACK', format_exc())

            try:
                try:
                    send(msg)
                except Exception:
                    send(('#UNSERIALIZABLE', format_exc()))
            except Exception as e:
                util.info('exception in thread serving %r',
                        threading.current_thread().name)
                util.info(' ... message was %r', msg)
                util.info(' ... exception was %r', e)
                conn.close()
                sys.exit(1)

    def fallback_getvalue(self, conn, ident, obj):
        return obj

    def fallback_str(self, conn, ident, obj):
        return str(obj)

    def fallback_repr(self, conn, ident, obj):
        return repr(obj)

    fallback_mapping = {
        '__str__':fallback_str,
        '__repr__':fallback_repr,
        '#GETVALUE':fallback_getvalue
        }

    def dummy(self, c):
        pass

    def debug_info(self, c):
        '''
        Return some info --- useful to spot problems with refcounting
        '''
        # Perhaps include debug info about 'c'?
        with self.mutex:
            result = []
            keys = list(self.id_to_refcount.keys())
            keys.sort()
            for ident in keys:
                if ident != '0':
                    result.append('  %s:       refcount=%s\n    %s' %
                                  (ident, self.id_to_refcount[ident],
                                   str(self.id_to_obj[ident][0])[:75]))
            return '\n'.join(result)

    def number_of_objects(self, c):
        '''
        Number of shared objects
        '''
        # Doesn't use (len(self.id_to_obj) - 1) as we shouldn't count ident='0'
        return len(self.id_to_refcount)

    def shutdown(self, c):
        '''
        Shutdown this process
        '''
        try:
            util.debug('manager received shutdown message')
            c.send(('#RETURN', None))
        except:
            import traceback
            traceback.print_exc()
        finally:
            self.stop_event.set()

    def create(self, c, typeid, /, *args, **kwds):
        '''
        Create a new shared object and return its id
        '''
        with self.mutex:
            callable, exposed, method_to_typeid, proxytype = \
                      self.registry[typeid]

            if callable is None:
                if kwds or (len(args) != 1):
                    raise ValueError(
                        "Without callable, must have one non-keyword argument")
                obj = args[0]
            else:
                obj = callable(*args, **kwds)

            if exposed is None:
                exposed = public_methods(obj)
            if method_to_typeid is not None:
                if not isinstance(method_to_typeid, dict):
                    raise TypeError(
                        "Method_to_typeid {0!r}: type {1!s}, not dict".format(
                            method_to_typeid, type(method_to_typeid)))
                exposed = list(exposed) + list(method_to_typeid)

            ident = '%x' % id(obj)  # convert to string because xmlrpclib
                                    # only has 32 bit signed integers
            util.debug('%r callable returned object with id %r', typeid, ident)

            self.id_to_obj[ident] = (obj, set(exposed), method_to_typeid)
            if ident not in self.id_to_refcount:
                self.id_to_refcount[ident] = 0

        self.incref(c, ident)
        return ident, tuple(exposed)

    def get_methods(self, c, token):
        '''
        Return the methods of the shared object indicated by token
        '''
        return tuple(self.id_to_obj[token.id][1])

    def accept_connection(self, c, name):
        '''
        Spawn a new thread to serve this connection
        '''
        threading.current_thread().name = name
        c.send(('#RETURN', None))
        self.serve_client(c)

    def incref(self, c, ident):
        with self.mutex:
            try:
                self.id_to_refcount[ident] += 1
            except KeyError as ke:
                # If no external references exist but an internal (to the
                # manager) still does and a new external reference is created
                # from it, restore the manager's tracking of it from the
                # previously stashed internal ref.
                if ident in self.id_to_local_proxy_obj:
                    self.id_to_refcount[ident] = 1
                    self.id_to_obj[ident] = \
                        self.id_to_local_proxy_obj[ident]
                    obj, exposed, gettypeid = self.id_to_obj[ident]
                    util.debug('Server re-enabled tracking & INCREF %r', ident)
                else:
                    raise ke

    def decref(self, c, ident):
        if ident not in self.id_to_refcount and \
            ident in self.id_to_local_proxy_obj:
            util.debug('Server DECREF skipping %r', ident)
            return

        with self.mutex:
            if self.id_to_refcount[ident] <= 0:
                raise AssertionError(
                    "Id {0!s} ({1!r}) has refcount {2:n}, not 1+".format(
                        ident, self.id_to_obj[ident],
                        self.id_to_refcount[ident]))
            self.id_to_refcount[ident] -= 1
            if self.id_to_refcount[ident] == 0:
                del self.id_to_refcount[ident]

        if ident not in self.id_to_refcount:
            # Two-step process in case the object turns out to contain other
            # proxy objects (e.g. a managed list of managed lists).
            # Otherwise, deleting self.id_to_obj[ident] would trigger the
            # deleting of the stored value (another managed object) which would
            # in turn attempt to acquire the mutex that is already held here.
            self.id_to_obj[ident] = (None, (), None)  # thread-safe
            util.debug('disposing of obj with id %r', ident)
            with self.mutex:
                del self.id_to_obj[ident]


#
# Class to represent state of a manager
#

class State(object):
    __slots__ = ['value']
    INITIAL = 0
    STARTED = 1
    SHUTDOWN = 2

#
# Mapping from serializer name to Listener and Client types
#

listener_client = { #XXX: register dill?
    'pickle' : (connection.Listener, connection.Client),
    'xmlrpclib' : (connection.XmlListener, connection.XmlClient)
    }

#
# Definition of BaseManager
#

class BaseManager(object):
    '''
    Base class for managers
    '''
    _registry = {}
    _Server = Server

    def __init__(self, address=None, authkey=None, serializer='pickle',
                 ctx=None):
        if authkey is None:
            authkey = process.current_process().authkey
        self._address = address     # XXX not final address if eg ('', 0)
        self._authkey = process.AuthenticationString(authkey)
        self._state = State()
        self._state.value = State.INITIAL
        self._serializer = serializer
        self._Listener, self._Client = listener_client[serializer]
        self._ctx = ctx or get_context()

    def get_server(self):
        '''
        Return server object with serve_forever() method and address attribute
        '''
        if self._state.value != State.INITIAL:
            if self._state.value == State.STARTED:
                raise ProcessError("Already started server")
            elif self._state.value == State.SHUTDOWN:
                raise ProcessError("Manager has shut down")
            else:
                raise ProcessError(
                    "Unknown state {!r}".format(self._state.value))
        return Server(self._registry, self._address,
                      self._authkey, self._serializer)

    def connect(self):
        '''
        Connect manager object to the server process
        '''
        Listener, Client = listener_client[self._serializer]
        conn = Client(self._address, authkey=self._authkey)
        dispatch(conn, None, 'dummy')
        self._state.value = State.STARTED

    def start(self, initializer=None, initargs=()):
        '''
        Spawn a server process for this manager object
        '''
        if self._state.value != State.INITIAL:
            if self._state.value == State.STARTED:
                raise ProcessError("Already started server")
            elif self._state.value == State.SHUTDOWN:
                raise ProcessError("Manager has shut down")
            else:
                raise ProcessError(
                    "Unknown state {!r}".format(self._state.value))

        if initializer is not None and not callable(initializer):
            raise TypeError('initializer must be a callable')

        # pipe over which we will retrieve address of server
        reader, writer = connection.Pipe(duplex=False)

        # spawn process which runs a server
        self._process = self._ctx.Process(
            target=type(self)._run_server,
            args=(self._registry, self._address, self._authkey,
                  self._serializer, writer, initializer, initargs),
            )
        ident = ':'.join(str(i) for i in self._process._identity)
        self._process.name = type(self).__name__  + '-' + ident
        self._process.start()

        # get address of server
        writer.close()
        self._address = reader.recv()
        reader.close()

        # register a finalizer
        self._state.value = State.STARTED
        self.shutdown = util.Finalize(
            self, type(self)._finalize_manager,
            args=(self._process, self._address, self._authkey,
                  self._state, self._Client),
            exitpriority=0
            )

    @classmethod
    def _run_server(cls, registry, address, authkey, serializer, writer,
                    initializer=None, initargs=()):
        '''
        Create a server, report its address and run it
        '''
        # bpo-36368: protect server process from KeyboardInterrupt signals
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        if initializer is not None:
            initializer(*initargs)

        # create server
        server = cls._Server(registry, address, authkey, serializer)

        # inform parent process of the server's address
        writer.send(server.address)
        writer.close()

        # run the manager
        util.info('manager serving at %r', server.address)
        server.serve_forever()

    def _create(self, typeid, /, *args, **kwds):
        '''
        Create a new shared object; return the token and exposed tuple
        '''
        assert self._state.value == State.STARTED, 'server not yet started'
        conn = self._Client(self._address, authkey=self._authkey)
        try:
            id, exposed = dispatch(conn, None, 'create', (typeid,)+args, kwds)
        finally:
            conn.close()
        return Token(typeid, self._address, id), exposed

    def join(self, timeout=None):
        '''
        Join the manager process (if it has been spawned)
        '''
        if self._process is not None:
            self._process.join(timeout)
            if not self._process.is_alive():
                self._process = None

    def _debug_info(self):
        '''
        Return some info about the servers shared objects and connections
        '''
        conn = self._Client(self._address, authkey=self._authkey)
        try:
            return dispatch(conn, None, 'debug_info')
        finally:
            conn.close()

    def _number_of_objects(self):
        '''
        Return the number of shared objects
        '''
        conn = self._Client(self._address, authkey=self._authkey)
        try:
            return dispatch(conn, None, 'number_of_objects')
        finally:
            conn.close()

    def __enter__(self):
        if self._state.value == State.INITIAL:
            self.start()
        if self._state.value != State.STARTED:
            if self._state.value == State.INITIAL:
                raise ProcessError("Unable to start server")
            elif self._state.value == State.SHUTDOWN:
                raise ProcessError("Manager has shut down")
            else:
                raise ProcessError(
                    "Unknown state {!r}".format(self._state.value))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    @staticmethod
    def _finalize_manager(process, address, authkey, state, _Client):
        '''
        Shutdown the manager process; will be registered as a finalizer
        '''
        if process.is_alive():
            util.info('sending shutdown message to manager')
            try:
                conn = _Client(address, authkey=authkey)
                try:
                    dispatch(conn, None, 'shutdown')
                finally:
                    conn.close()
            except Exception:
                pass

            process.join(timeout=1.0)
            if process.is_alive():
                util.info('manager still alive')
                if hasattr(process, 'terminate'):
                    util.info('trying to `terminate()` manager process')
                    process.terminate()
                    process.join(timeout=1.0)
                    if process.is_alive():
                        util.info('manager still alive after terminate')

        state.value = State.SHUTDOWN
        try:
            del BaseProxy._address_to_local[address]
        except KeyError:
            pass

    @property
    def address(self):
        return self._address

    @classmethod
    def register(cls, typeid, callable=None, proxytype=None, exposed=None,
                 method_to_typeid=None, create_method=True):
        '''
        Register a typeid with the manager type
        '''
        if '_registry' not in cls.__dict__:
            cls._registry = cls._registry.copy()

        if proxytype is None:
            proxytype = AutoProxy

        exposed = exposed or getattr(proxytype, '_exposed_', None)

        method_to_typeid = method_to_typeid or \
                           getattr(proxytype, '_method_to_typeid_', None)

        if method_to_typeid:
            for key, value in list(method_to_typeid.items()): # isinstance?
                assert type(key) is str, '%r is not a string' % key
                assert type(value) is str, '%r is not a string' % value

        cls._registry[typeid] = (
            callable, exposed, method_to_typeid, proxytype
            )

        if create_method:
            def temp(self, /, *args, **kwds):
                util.debug('requesting creation of a shared %r object', typeid)
                token, exp = self._create(typeid, *args, **kwds)
                proxy = proxytype(
                    token, self._serializer, manager=self,
                    authkey=self._authkey, exposed=exp
                    )
                conn = self._Client(token.address, authkey=self._authkey)
                dispatch(conn, None, 'decref', (token.id,))
                return proxy
            temp.__name__ = typeid
            setattr(cls, typeid, temp)

#
# Subclass of set which get cleared after a fork
#

class ProcessLocalSet(set):
    def __init__(self):
        util.register_after_fork(self, lambda obj: obj.clear())
    def __reduce__(self):
        return type(self), ()

#
# Definition of BaseProxy
#

class BaseProxy(object):
    '''
    A base for proxies of shared objects
    '''
    _address_to_local = {}
    _mutex = util.ForkAwareThreadLock()

    def __init__(self, token, serializer, manager=None,
                 authkey=None, exposed=None, incref=True, manager_owned=False):
        with BaseProxy._mutex:
            tls_idset = BaseProxy._address_to_local.get(token.address, None)
            if tls_idset is None:
                tls_idset = util.ForkAwareLocal(), ProcessLocalSet()
                BaseProxy._address_to_local[token.address] = tls_idset

        # self._tls is used to record the connection used by this
        # thread to communicate with the manager at token.address
        self._tls = tls_idset[0]

        # self._idset is used to record the identities of all shared
        # objects for which the current process owns references and
        # which are in the manager at token.address
        self._idset = tls_idset[1]

        self._token = token
        self._id = self._token.id
        self._manager = manager
        self._serializer = serializer
        self._Client = listener_client[serializer][1]

        # Should be set to True only when a proxy object is being created
        # on the manager server; primary use case: nested proxy objects.
        # RebuildProxy detects when a proxy is being created on the manager
        # and sets this value appropriately.
        self._owned_by_manager = manager_owned

        if authkey is not None:
            self._authkey = process.AuthenticationString(authkey)
        elif self._manager is not None:
            self._authkey = self._manager._authkey
        else:
            self._authkey = process.current_process().authkey

        if incref:
            self._incref()

        util.register_after_fork(self, BaseProxy._after_fork)

    def _connect(self):
        util.debug('making connection to manager')
        name = process.current_process().name
        if threading.current_thread().name != 'MainThread':
            name += '|' + threading.current_thread().name
        conn = self._Client(self._token.address, authkey=self._authkey)
        dispatch(conn, None, 'accept_connection', (name,))
        self._tls.connection = conn

    def _callmethod(self, methodname, args=(), kwds={}):
        '''
        Try to call a method of the referent and return a copy of the result
        '''
        try:
            conn = self._tls.connection
        except AttributeError:
            util.debug('thread %r does not own a connection',
                       threading.current_thread().name)
            self._connect()
            conn = self._tls.connection

        conn.send((self._id, methodname, args, kwds))
        kind, result = conn.recv()

        if kind == '#RETURN':
            return result
        elif kind == '#PROXY':
            exposed, token = result
            proxytype = self._manager._registry[token.typeid][-1]
            token.address = self._token.address
            proxy = proxytype(
                token, self._serializer, manager=self._manager,
                authkey=self._authkey, exposed=exposed
                )
            conn = self._Client(token.address, authkey=self._authkey)
            dispatch(conn, None, 'decref', (token.id,))
            return proxy
        raise convert_to_error(kind, result)

    def _getvalue(self):
        '''
        Get a copy of the value of the referent
        '''
        return self._callmethod('#GETVALUE')

    def _incref(self):
        if self._owned_by_manager:
            util.debug('owned_by_manager skipped INCREF of %r', self._token.id)
            return

        conn = self._Client(self._token.address, authkey=self._authkey)
        dispatch(conn, None, 'incref', (self._id,))
        util.debug('INCREF %r', self._token.id)

        self._idset.add(self._id)

        state = self._manager and self._manager._state

        self._close = util.Finalize(
            self, BaseProxy._decref,
            args=(self._token, self._authkey, state,
                  self._tls, self._idset, self._Client),
            exitpriority=10
            )

    @staticmethod
    def _decref(token, authkey, state, tls, idset, _Client):
        idset.discard(token.id)

        # check whether manager is still alive
        if state is None or state.value == State.STARTED:
            # tell manager this process no longer cares about referent
            try:
                util.debug('DECREF %r', token.id)
                conn = _Client(token.address, authkey=authkey)
                dispatch(conn, None, 'decref', (token.id,))
            except Exception as e:
                util.debug('... decref failed %s', e)

        else:
            util.debug('DECREF %r -- manager already shutdown', token.id)

        # check whether we can close this thread's connection because
        # the process owns no more references to objects for this manager
        if not idset and hasattr(tls, 'connection'):
            util.debug('thread %r has no more proxies so closing conn',
                       threading.current_thread().name)
            tls.connection.close()
            del tls.connection

    def _after_fork(self):
        self._manager = None
        try:
            self._incref()
        except Exception as e:
            # the proxy may just be for a manager which has shutdown
            util.info('incref failed: %s' % e)

    def __reduce__(self):
        kwds = {}
        if get_spawning_popen() is not None:
            kwds['authkey'] = self._authkey

        if getattr(self, '_isauto', False):
            kwds['exposed'] = self._exposed_
            return (RebuildProxy,
                    (AutoProxy, self._token, self._serializer, kwds))
        else:
            return (RebuildProxy,
                    (type(self), self._token, self._serializer, kwds))

    def __deepcopy__(self, memo):
        return self._getvalue()

    def __repr__(self):
        return '<%s object, typeid %r at %#x>' % \
               (type(self).__name__, self._token.typeid, id(self))

    def __str__(self):
        '''
        Return representation of the referent (or a fall-back if that fails)
        '''
        try:
            return self._callmethod('__repr__')
        except Exception:
            return repr(self)[:-1] + "; '__str__()' failed>"

#
# Function used for unpickling
#

def RebuildProxy(func, token, serializer, kwds):
    '''
    Function used for unpickling proxy objects.
    '''
    server = getattr(process.current_process(), '_manager_server', None)
    if server and server.address == token.address:
        util.debug('Rebuild a proxy owned by manager, token=%r', token)
        kwds['manager_owned'] = True
        if token.id not in server.id_to_local_proxy_obj:
            server.id_to_local_proxy_obj[token.id] = \
                server.id_to_obj[token.id]
    incref = (
        kwds.pop('incref', True) and
        not getattr(process.current_process(), '_inheriting', False)
        )
    return func(token, serializer, incref=incref, **kwds)

#
# Functions to create proxies and proxy types
#

def MakeProxyType(name, exposed, _cache={}):
    '''
    Return a proxy type whose methods are given by `exposed`
    '''
    exposed = tuple(exposed)
    try:
        return _cache[(name, exposed)]
    except KeyError:
        pass

    dic = {}

    for meth in exposed:
        exec('''def %s(self, /, *args, **kwds):
        return self._callmethod(%r, args, kwds)''' % (meth, meth), dic)

    ProxyType = type(name, (BaseProxy,), dic)
    ProxyType._exposed_ = exposed
    _cache[(name, exposed)] = ProxyType
    return ProxyType


def AutoProxy(token, serializer, manager=None, authkey=None,
              exposed=None, incref=True, manager_owned=False):
    '''
    Return an auto-proxy for `token`
    '''
    _Client = listener_client[serializer][1]

    if exposed is None:
        conn = _Client(token.address, authkey=authkey)
        try:
            exposed = dispatch(conn, None, 'get_methods', (token,))
        finally:
            conn.close()

    if authkey is None and manager is not None:
        authkey = manager._authkey
    if authkey is None:
        authkey = process.current_process().authkey

    ProxyType = MakeProxyType('AutoProxy[%s]' % token.typeid, exposed)
    proxy = ProxyType(token, serializer, manager=manager, authkey=authkey,
                      incref=incref, manager_owned=manager_owned)
    proxy._isauto = True
    return proxy

#
# Types/callables which we will register with SyncManager
#

class Namespace(object):
    def __init__(self, /, **kwds):
        self.__dict__.update(kwds)
    def __repr__(self):
        items = list(self.__dict__.items())
        temp = []
        for name, value in items:
            if not name.startswith('_'):
                temp.append('%s=%r' % (name, value))
        temp.sort()
        return '%s(%s)' % (self.__class__.__name__, ', '.join(temp))

class Value(object):
    def __init__(self, typecode, value, lock=True):
        self._typecode = typecode
        self._value = value
    def get(self):
        return self._value
    def set(self, value):
        self._value = value
    def __repr__(self):
        return '%s(%r, %r)'%(type(self).__name__, self._typecode, self._value)
    value = property(get, set)

def Array(typecode, sequence, lock=True):
    return array.array(typecode, sequence)

#
# Proxy types used by SyncManager
#

class IteratorProxy(BaseProxy):
    _exposed_ = ('__next__', 'send', 'throw', 'close')
    def __iter__(self):
        return self
    def __next__(self, *args):
        return self._callmethod('__next__', args)
    def send(self, *args):
        return self._callmethod('send', args)
    def throw(self, *args):
        return self._callmethod('throw', args)
    def close(self, *args):
        return self._callmethod('close', args)


class AcquirerProxy(BaseProxy):
    _exposed_ = ('acquire', 'release')
    def acquire(self, blocking=True, timeout=None):
        args = (blocking,) if timeout is None else (blocking, timeout)
        return self._callmethod('acquire', args)
    def release(self):
        return self._callmethod('release')
    def __enter__(self):
        return self._callmethod('acquire')
    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._callmethod('release')


class ConditionProxy(AcquirerProxy):
    _exposed_ = ('acquire', 'release', 'wait', 'notify', 'notify_all')
    def wait(self, timeout=None):
        return self._callmethod('wait', (timeout,))
    def notify(self, n=1):
        return self._callmethod('notify', (n,))
    def notify_all(self):
        return self._callmethod('notify_all')
    def wait_for(self, predicate, timeout=None):
        result = predicate()
        if result:
            return result
        if timeout is not None:
            endtime = getattr(time,'monotonic',time.time)() + timeout
        else:
            endtime = None
            waittime = None
        while not result:
            if endtime is not None:
                waittime = endtime - getattr(time,'monotonic',time.time)()
                if waittime <= 0:
                    break
            self.wait(waittime)
            result = predicate()
        return result


class EventProxy(BaseProxy):
    _exposed_ = ('is_set', 'set', 'clear', 'wait')
    def is_set(self):
        return self._callmethod('is_set')
    def set(self):
        return self._callmethod('set')
    def clear(self):
        return self._callmethod('clear')
    def wait(self, timeout=None):
        return self._callmethod('wait', (timeout,))


class BarrierProxy(BaseProxy):
    _exposed_ = ('__getattribute__', 'wait', 'abort', 'reset')
    def wait(self, timeout=None):
        return self._callmethod('wait', (timeout,))
    def abort(self):
        return self._callmethod('abort')
    def reset(self):
        return self._callmethod('reset')
    @property
    def parties(self):
        return self._callmethod('__getattribute__', ('parties',))
    @property
    def n_waiting(self):
        return self._callmethod('__getattribute__', ('n_waiting',))
    @property
    def broken(self):
        return self._callmethod('__getattribute__', ('broken',))


class NamespaceProxy(BaseProxy):
    _exposed_ = ('__getattribute__', '__setattr__', '__delattr__')
    def __getattr__(self, key):
        if key[0] == '_':
            return object.__getattribute__(self, key)
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod('__getattribute__', (key,))
    def __setattr__(self, key, value):
        if key[0] == '_':
            return object.__setattr__(self, key, value)
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod('__setattr__', (key, value))
    def __delattr__(self, key):
        if key[0] == '_':
            return object.__delattr__(self, key)
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod('__delattr__', (key,))


class ValueProxy(BaseProxy):
    _exposed_ = ('get', 'set')
    def get(self):
        return self._callmethod('get')
    def set(self, value):
        return self._callmethod('set', (value,))
    value = property(get, set)

    __class_getitem__ = classmethod(types.GenericAlias)


BaseListProxy = MakeProxyType('BaseListProxy', (
    '__add__', '__contains__', '__delitem__', '__getitem__', '__len__',
    '__mul__', '__reversed__', '__rmul__', '__setitem__',
    'append', 'count', 'extend', 'index', 'insert', 'pop', 'remove',
    'reverse', 'sort', '__imul__'
    ))
class ListProxy(BaseListProxy):
    def __iadd__(self, value):
        self._callmethod('extend', (value,))
        return self
    def __imul__(self, value):
        self._callmethod('__imul__', (value,))
        return self


DictProxy = MakeProxyType('DictProxy', (
    '__contains__', '__delitem__', '__getitem__', '__iter__', '__len__',
    '__setitem__', 'clear', 'copy', 'get', 'items',
    'keys', 'pop', 'popitem', 'setdefault', 'update', 'values'
    ))
DictProxy._method_to_typeid_ = {
    '__iter__': 'Iterator',
    }


ArrayProxy = MakeProxyType('ArrayProxy', (
    '__len__', '__getitem__', '__setitem__'
    ))


BasePoolProxy = MakeProxyType('PoolProxy', (
    'apply', 'apply_async', 'close', 'imap', 'imap_unordered', 'join',
    'map', 'map_async', 'starmap', 'starmap_async', 'terminate',
    ))
BasePoolProxy._method_to_typeid_ = {
    'apply_async': 'AsyncResult',
    'map_async': 'AsyncResult',
    'starmap_async': 'AsyncResult',
    'imap': 'Iterator',
    'imap_unordered': 'Iterator'
    }
class PoolProxy(BasePoolProxy):
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()

#
# Definition of SyncManager
#

class SyncManager(BaseManager):
    '''
    Subclass of `BaseManager` which supports a number of shared object types.

    The types registered are those intended for the synchronization
    of threads, plus `dict`, `list` and `Namespace`.

    The `multiprocess.Manager()` function creates started instances of
    this class.
    '''

SyncManager.register('Queue', queue.Queue)
SyncManager.register('JoinableQueue', queue.Queue)
SyncManager.register('Event', threading.Event, EventProxy)
SyncManager.register('Lock', threading.Lock, AcquirerProxy)
SyncManager.register('RLock', threading.RLock, AcquirerProxy)
SyncManager.register('Semaphore', threading.Semaphore, AcquirerProxy)
SyncManager.register('BoundedSemaphore', threading.BoundedSemaphore,
                     AcquirerProxy)
SyncManager.register('Condition', threading.Condition, ConditionProxy)
SyncManager.register('Barrier', threading.Barrier, BarrierProxy)
SyncManager.register('Pool', pool.Pool, PoolProxy)
SyncManager.register('list', list, ListProxy)
SyncManager.register('dict', dict, DictProxy)
SyncManager.register('Value', Value, ValueProxy)
SyncManager.register('Array', Array, ArrayProxy)
SyncManager.register('Namespace', Namespace, NamespaceProxy)

# types returned by methods of PoolProxy
SyncManager.register('Iterator', proxytype=IteratorProxy, create_method=False)
SyncManager.register('AsyncResult', create_method=False)

#
# Definition of SharedMemoryManager and SharedMemoryServer
#

if HAS_SHMEM:
    class _SharedMemoryTracker:
        "Manages one or more shared memory segments."

        def __init__(self, name, segment_names=[]):
            self.shared_memory_context_name = name
            self.segment_names = segment_names

        def register_segment(self, segment_name):
            "Adds the supplied shared memory block name to tracker."
            util.debug(f"Register segment {segment_name!r} in pid {getpid()}")
            self.segment_names.append(segment_name)

        def destroy_segment(self, segment_name):
            """Calls unlink() on the shared memory block with the supplied name
            and removes it from the list of blocks being tracked."""
            util.debug(f"Destroy segment {segment_name!r} in pid {getpid()}")
            self.segment_names.remove(segment_name)
            segment = shared_memory.SharedMemory(segment_name)
            segment.close()
            segment.unlink()

        def unlink(self):
            "Calls destroy_segment() on all tracked shared memory blocks."
            for segment_name in self.segment_names[:]:
                self.destroy_segment(segment_name)

        def __del__(self):
            util.debug(f"Call {self.__class__.__name__}.__del__ in {getpid()}")
            self.unlink()

        def __getstate__(self):
            return (self.shared_memory_context_name, self.segment_names)

        def __setstate__(self, state):
            self.__init__(*state)


    class SharedMemoryServer(Server):

        public = Server.public + \
                 ['track_segment', 'release_segment', 'list_segments']

        def __init__(self, *args, **kwargs):
            Server.__init__(self, *args, **kwargs)
            address = self.address
            # The address of Linux abstract namespaces can be bytes
            if isinstance(address, bytes):
                address = os.fsdecode(address)
            self.shared_memory_context = \
                _SharedMemoryTracker(f"shm_{address}_{getpid()}")
            util.debug(f"SharedMemoryServer started by pid {getpid()}")

        def create(self, c, typeid, /, *args, **kwargs):
            """Create a new distributed-shared object (not backed by a shared
            memory block) and return its id to be used in a Proxy Object."""
            # Unless set up as a shared proxy, don't make shared_memory_context
            # a standard part of kwargs.  This makes things easier for supplying
            # simple functions.
            if hasattr(self.registry[typeid][-1], "_shared_memory_proxy"):
                kwargs['shared_memory_context'] = self.shared_memory_context
            return Server.create(self, c, typeid, *args, **kwargs)

        def shutdown(self, c):
            "Call unlink() on all tracked shared memory, terminate the Server."
            self.shared_memory_context.unlink()
            return Server.shutdown(self, c)

        def track_segment(self, c, segment_name):
            "Adds the supplied shared memory block name to Server's tracker."
            self.shared_memory_context.register_segment(segment_name)

        def release_segment(self, c, segment_name):
            """Calls unlink() on the shared memory block with the supplied name
            and removes it from the tracker instance inside the Server."""
            self.shared_memory_context.destroy_segment(segment_name)

        def list_segments(self, c):
            """Returns a list of names of shared memory blocks that the Server
            is currently tracking."""
            return self.shared_memory_context.segment_names


    class SharedMemoryManager(BaseManager):
        """Like SyncManager but uses SharedMemoryServer instead of Server.

        It provides methods for creating and returning SharedMemory instances
        and for creating a list-like object (ShareableList) backed by shared
        memory.  It also provides methods that create and return Proxy Objects
        that support synchronization across processes (i.e. multi-process-safe
        locks and semaphores).
        """

        _Server = SharedMemoryServer

        def __init__(self, *args, **kwargs):
            if os.name == "posix":
                # bpo-36867: Ensure the resource_tracker is running before
                # launching the manager process, so that concurrent
                # shared_memory manipulation both in the manager and in the
                # current process does not create two resource_tracker
                # processes.
                from . import resource_tracker
                resource_tracker.ensure_running()
            BaseManager.__init__(self, *args, **kwargs)
            util.debug(f"{self.__class__.__name__} created by pid {getpid()}")

        def __del__(self):
            util.debug(f"{self.__class__.__name__}.__del__ by pid {getpid()}")
            pass

        def get_server(self):
            'Better than monkeypatching for now; merge into Server ultimately'
            if self._state.value != State.INITIAL:
                if self._state.value == State.STARTED:
                    raise ProcessError("Already started SharedMemoryServer")
                elif self._state.value == State.SHUTDOWN:
                    raise ProcessError("SharedMemoryManager has shut down")
                else:
                    raise ProcessError(
                        "Unknown state {!r}".format(self._state.value))
            return self._Server(self._registry, self._address,
                                self._authkey, self._serializer)

        def SharedMemory(self, size):
            """Returns a new SharedMemory instance with the specified size in
            bytes, to be tracked by the manager."""
            with self._Client(self._address, authkey=self._authkey) as conn:
                sms = shared_memory.SharedMemory(None, create=True, size=size)
                try:
                    dispatch(conn, None, 'track_segment', (sms.name,))
                except BaseException as e:
                    sms.unlink()
                    raise e
            return sms

        def ShareableList(self, sequence):
            """Returns a new ShareableList instance populated with the values
            from the input sequence, to be tracked by the manager."""
            with self._Client(self._address, authkey=self._authkey) as conn:
                sl = shared_memory.ShareableList(sequence)
                try:
                    dispatch(conn, None, 'track_segment', (sl.shm.name,))
                except BaseException as e:
                    sl.shm.unlink()
                    raise e
            return sl
<|endoftext|>import errno
import os
import selectors
import signal
import socket
import struct
import sys
import threading
import warnings

from . import connection
from . import process
from .context import reduction
from . import resource_tracker
from . import spawn
from . import util

__all__ = ['ensure_running', 'get_inherited_fds', 'connect_to_new_process',
           'set_forkserver_preload']

#
#
#

MAXFDS_TO_SEND = 256
SIGNED_STRUCT = struct.Struct('q')     # large enough for pid_t

#
# Forkserver class
#

class ForkServer(object):

    def __init__(self):
        self._forkserver_address = None
        self._forkserver_alive_fd = None
        self._forkserver_pid = None
        self._inherited_fds = None
        self._lock = threading.Lock()
        self._preload_modules = ['__main__']

    def _stop(self):
        # Method used by unit tests to stop the server
        with self._lock:
            self._stop_unlocked()

    def _stop_unlocked(self):
        if self._forkserver_pid is None:
            return

        # close the "alive" file descriptor asks the server to stop
        os.close(self._forkserver_alive_fd)
        self._forkserver_alive_fd = None

        os.waitpid(self._forkserver_pid, 0)
        self._forkserver_pid = None

        if not util.is_abstract_socket_namespace(self._forkserver_address):
            os.unlink(self._forkserver_address)
        self._forkserver_address = None

    def set_forkserver_preload(self, modules_names):
        '''Set list of module names to try to load in forkserver process.'''
        if not all(type(mod) is str for mod in self._preload_modules):
            raise TypeError('module_names must be a list of strings')
        self._preload_modules = modules_names

    def get_inherited_fds(self):
        '''Return list of fds inherited from parent process.

        This returns None if the current process was not started by fork
        server.
        '''
        return self._inherited_fds

    def connect_to_new_process(self, fds):
        '''Request forkserver to create a child process.

        Returns a pair of fds (status_r, data_w).  The calling process can read
        the child process's pid and (eventually) its returncode from status_r.
        The calling process should write to data_w the pickled preparation and
        process data.
        '''
        self.ensure_running()
        if len(fds) + 4 >= MAXFDS_TO_SEND:
            raise ValueError('too many fds')
        with socket.socket(socket.AF_UNIX) as client:
            client.connect(self._forkserver_address)
            parent_r, child_w = os.pipe()
            child_r, parent_w = os.pipe()
            allfds = [child_r, child_w, self._forkserver_alive_fd,
                      resource_tracker.getfd()]
            allfds += fds
            try:
                reduction.sendfds(client, allfds)
                return parent_r, parent_w
            except:
                os.close(parent_r)
                os.close(parent_w)
                raise
            finally:
                os.close(child_r)
                os.close(child_w)

    def ensure_running(self):
        '''Make sure that a fork server is running.

        This can be called from any process.  Note that usually a child
        process will just reuse the forkserver started by its parent, so
        ensure_running() will do nothing.
        '''
        with self._lock:
            resource_tracker.ensure_running()
            if self._forkserver_pid is not None:
                # forkserver was launched before, is it still running?
                pid, status = os.waitpid(self._forkserver_pid, os.WNOHANG)
                if not pid:
                    # still alive
                    return
                # dead, launch it again
                os.close(self._forkserver_alive_fd)
                self._forkserver_address = None
                self._forkserver_alive_fd = None
                self._forkserver_pid = None

            cmd = ('from multiprocess.forkserver import main; ' +
                   'main(%d, %d, %r, **%r)')

            if self._preload_modules:
                desired_keys = {'main_path', 'sys_path'}
                data = spawn.get_preparation_data('ignore')
                data = {x: y for x, y in data.items() if x in desired_keys}
            else:
                data = {}

            with socket.socket(socket.AF_UNIX) as listener:
                address = connection.arbitrary_address('AF_UNIX')
                listener.bind(address)
                if not util.is_abstract_socket_namespace(address):
                    os.chmod(address, 0o600)
                listener.listen()

                # all client processes own the write end of the "alive" pipe;
                # when they all terminate the read end becomes ready.
                alive_r, alive_w = os.pipe()
                try:
                    fds_to_pass = [listener.fileno(), alive_r]
                    cmd %= (listener.fileno(), alive_r, self._preload_modules,
                            data)
                    exe = spawn.get_executable()
                    args = [exe] + util._args_from_interpreter_flags()
                    args += ['-c', cmd]
                    pid = util.spawnv_passfds(exe, args, fds_to_pass)
                except:
                    os.close(alive_w)
                    raise
                finally:
                    os.close(alive_r)
                self._forkserver_address = address
                self._forkserver_alive_fd = alive_w
                self._forkserver_pid = pid

#
#
#

def main(listener_fd, alive_r, preload, main_path=None, sys_path=None):
    '''Run forkserver.'''
    if preload:
        if '__main__' in preload and main_path is not None:
            process.current_process()._inheriting = True
            try:
                spawn.import_main_path(main_path)
            finally:
                del process.current_process()._inheriting
        for modname in preload:
            try:
                __import__(modname)
            except ImportError:
                pass

    util._close_stdin()

    sig_r, sig_w = os.pipe()
    os.set_blocking(sig_r, False)
    os.set_blocking(sig_w, False)

    def sigchld_handler(*_unused):
        # Dummy signal handler, doesn't do anything
        pass

    handlers = {
        # unblocking SIGCHLD allows the wakeup fd to notify our event loop
        signal.SIGCHLD: sigchld_handler,
        # protect the process from ^C
        signal.SIGINT: signal.SIG_IGN,
        }
    old_handlers = {sig: signal.signal(sig, val)
                    for (sig, val) in handlers.items()}

    # calling os.write() in the Python signal handler is racy
    signal.set_wakeup_fd(sig_w)

    # map child pids to client fds
    pid_to_fd = {}

    with socket.socket(socket.AF_UNIX, fileno=listener_fd) as listener, \
         selectors.DefaultSelector() as selector:
        _forkserver._forkserver_address = listener.getsockname()

        selector.register(listener, selectors.EVENT_READ)
        selector.register(alive_r, selectors.EVENT_READ)
        selector.register(sig_r, selectors.EVENT_READ)

        while True:
            try:
                while True:
                    rfds = [key.fileobj for (key, events) in selector.select()]
                    if rfds:
                        break

                if alive_r in rfds:
                    # EOF because no more client processes left
                    assert os.read(alive_r, 1) == b'', "Not at EOF?"
                    raise SystemExit

                if sig_r in rfds:
                    # Got SIGCHLD
                    os.read(sig_r, 65536)  # exhaust
                    while True:
                        # Scan for child processes
                        try:
                            pid, sts = os.waitpid(-1, os.WNOHANG)
                        except ChildProcessError:
                            break
                        if pid == 0:
                            break
                        child_w = pid_to_fd.pop(pid, None)
                        if child_w is not None:
                            returncode = os.waitstatus_to_exitcode(sts)
                            # Send exit code to client process
                            try:
                                write_signed(child_w, returncode)
                            except BrokenPipeError:
                                # client vanished
                                pass
                            os.close(child_w)
                        else:
                            # This shouldn't happen really
                            warnings.warn('forkserver: waitpid returned '
                                          'unexpected pid %d' % pid)

                if listener in rfds:
                    # Incoming fork request
                    with listener.accept()[0] as s:
                        # Receive fds from client
                        fds = reduction.recvfds(s, MAXFDS_TO_SEND + 1)
                        if len(fds) > MAXFDS_TO_SEND:
                            raise RuntimeError(
                                "Too many ({0:n}) fds to send".format(
                                    len(fds)))
                        child_r, child_w, *fds = fds
                        s.close()
                        pid = os.fork()
                        if pid == 0:
                            # Child
                            code = 1
                            try:
                                listener.close()
                                selector.close()
                                unused_fds = [alive_r, child_w, sig_r, sig_w]
                                unused_fds.extend(pid_to_fd.values())
                                code = _serve_one(child_r, fds,
                                                  unused_fds,
                                                  old_handlers)
                            except Exception:
                                sys.excepthook(*sys.exc_info())
                                sys.stderr.flush()
                            finally:
                                os._exit(code)
                        else:
                            # Send pid to client process
                            try:
                                write_signed(child_w, pid)
                            except BrokenPipeError:
                                # client vanished
                                pass
                            pid_to_fd[pid] = child_w
                            os.close(child_r)
                            for fd in fds:
                                os.close(fd)

            except OSError as e:
                if e.errno != errno.ECONNABORTED:
                    raise


def _serve_one(child_r, fds, unused_fds, handlers):
    # close unnecessary stuff and reset signal handlers
    signal.set_wakeup_fd(-1)
    for sig, val in handlers.items():
        signal.signal(sig, val)
    for fd in unused_fds:
        os.close(fd)

    (_forkserver._forkserver_alive_fd,
     resource_tracker._resource_tracker._fd,
     *_forkserver._inherited_fds) = fds

    # Run process object received over pipe
    parent_sentinel = os.dup(child_r)
    code = spawn._main(child_r, parent_sentinel)

    return code


#
# Read and write signed numbers
#

def read_signed(fd):
    data = b''
    length = SIGNED_STRUCT.size
    while len(data) < length:
        s = os.read(fd, length - len(data))
        if not s:
            raise EOFError('unexpected EOF')
        data += s
    return SIGNED_STRUCT.unpack(data)[0]

def write_signed(fd, n):
    msg = SIGNED_STRUCT.pack(n)
    while msg:
        nbytes = os.write(fd, msg)
        if nbytes == 0:
            raise RuntimeError('should not get here')
        msg = msg[nbytes:]

#
#
#

_forkserver = ForkServer()
ensure_running = _forkserver.ensure_running
get_inherited_fds = _forkserver.get_inherited_fds
connect_to_new_process = _forkserver.connect_to_new_process
set_forkserver_preload = _forkserver.set_forkserver_preload
<|endoftext|>#
# We use a background thread for sharing fds on Unix, and for sharing sockets on
# Windows.
#
# A client which wants to pickle a resource registers it with the resource
# sharer and gets an identifier in return.  The unpickling process will connect
# to the resource sharer, sends the identifier and its pid, and then receives
# the resource.
#

import os
import signal
import socket
import sys
import threading

from . import process
from .context import reduction
from . import util

__all__ = ['stop']


if sys.platform == 'win32':
    __all__ += ['DupSocket']

    class DupSocket(object):
        '''Picklable wrapper for a socket.'''
        def __init__(self, sock):
            new_sock = sock.dup()
            def send(conn, pid):
                share = new_sock.share(pid)
                conn.send_bytes(share)
            self._id = _resource_sharer.register(send, new_sock.close)

        def detach(self):
            '''Get the socket.  This should only be called once.'''
            with _resource_sharer.get_connection(self._id) as conn:
                share = conn.recv_bytes()
                return socket.fromshare(share)

else:
    __all__ += ['DupFd']

    class DupFd(object):
        '''Wrapper for fd which can be used at any time.'''
        def __init__(self, fd):
            new_fd = os.dup(fd)
            def send(conn, pid):
                reduction.send_handle(conn, new_fd, pid)
            def close():
                os.close(new_fd)
            self._id = _resource_sharer.register(send, close)

        def detach(self):
            '''Get the fd.  This should only be called once.'''
            with _resource_sharer.get_connection(self._id) as conn:
                return reduction.recv_handle(conn)


class _ResourceSharer(object):
    '''Manager for resources using background thread.'''
    def __init__(self):
        self._key = 0
        self._cache = {}
        self._lock = threading.Lock()
        self._listener = None
        self._address = None
        self._thread = None
        util.register_after_fork(self, _ResourceSharer._afterfork)

    def register(self, send, close):
        '''Register resource, returning an identifier.'''
        with self._lock:
            if self._address is None:
                self._start()
            self._key += 1
            self._cache[self._key] = (send, close)
            return (self._address, self._key)

    @staticmethod
    def get_connection(ident):
        '''Return connection from which to receive identified resource.'''
        from .connection import Client
        address, key = ident
        c = Client(address, authkey=process.current_process().authkey)
        c.send((key, os.getpid()))
        return c

    def stop(self, timeout=None):
        '''Stop the background thread and clear registered resources.'''
        from .connection import Client
        with self._lock:
            if self._address is not None:
                c = Client(self._address,
                           authkey=process.current_process().authkey)
                c.send(None)
                c.close()
                self._thread.join(timeout)
                if self._thread.is_alive():
                    util.sub_warning('_ResourceSharer thread did '
                                     'not stop when asked')
                self._listener.close()
                self._thread = None
                self._address = None
                self._listener = None
                for key, (send, close) in self._cache.items():
                    close()
                self._cache.clear()

    def _afterfork(self):
        for key, (send, close) in self._cache.items():
            close()
        self._cache.clear()
        self._lock._at_fork_reinit()
        if self._listener is not None:
            self._listener.close()
        self._listener = None
        self._address = None
        self._thread = None

    def _start(self):
        from .connection import Listener
        assert self._listener is None, "Already have Listener"
        util.debug('starting listener and thread for sending handles')
        self._listener = Listener(authkey=process.current_process().authkey)
        self._address = self._listener.address
        t = threading.Thread(target=self._serve)
        t.daemon = True
        t.start()
        self._thread = t

    def _serve(self):
        if hasattr(signal, 'pthread_sigmask'):
            signal.pthread_sigmask(signal.SIG_BLOCK, signal.valid_signals())
        while 1:
            try:
                with self._listener.accept() as conn:
                    msg = conn.recv()
                    if msg is None:
                        break
                    key, destination_pid = msg
                    send, close = self._cache.pop(key)
                    try:
                        send(conn, destination_pid)
                    finally:
                        close()
            except:
                if not util.is_exiting():
                    sys.excepthook(*sys.exc_info())


_resource_sharer = _ResourceSharer()
stop = _resource_sharer.stop
<|endoftext|>import os
import msvcrt
import signal
import sys
import _winapi

from .context import reduction, get_spawning_popen, set_spawning_popen
from . import spawn
from . import util

__all__ = ['Popen']

#
#
#

TERMINATE = 0x10000
WINEXE = (sys.platform == 'win32' and getattr(sys, 'frozen', False))
WINSERVICE = sys.executable.lower().endswith("pythonservice.exe")


def _path_eq(p1, p2):
    return p1 == p2 or os.path.normcase(p1) == os.path.normcase(p2)

WINENV = not _path_eq(sys.executable, sys._base_executable)


def _close_handles(*handles):
    for handle in handles:
        _winapi.CloseHandle(handle)


#
# We define a Popen class similar to the one from subprocess, but
# whose constructor takes a process object as its argument.
#

class Popen(object):
    '''
    Start a subprocess to run the code of a process object
    '''
    method = 'spawn'

    def __init__(self, process_obj):
        prep_data = spawn.get_preparation_data(process_obj._name)

        # read end of pipe will be duplicated by the child process
        # -- see spawn_main() in spawn.py.
        #
        # bpo-33929: Previously, the read end of pipe was "stolen" by the child
        # process, but it leaked a handle if the child process had been
        # terminated before it could steal the handle from the parent process.
        rhandle, whandle = _winapi.CreatePipe(None, 0)
        wfd = msvcrt.open_osfhandle(whandle, 0)
        cmd = spawn.get_command_line(parent_pid=os.getpid(),
                                     pipe_handle=rhandle)
        cmd = ' '.join('"%s"' % x for x in cmd)

        python_exe = spawn.get_executable()

        # bpo-35797: When running in a venv, we bypass the redirect
        # executor and launch our base Python.
        if WINENV and _path_eq(python_exe, sys.executable):
            python_exe = sys._base_executable
            env = os.environ.copy()
            env["__PYVENV_LAUNCHER__"] = sys.executable
        else:
            env = None

        with open(wfd, 'wb', closefd=True) as to_child:
            # start process
            try:
                hp, ht, pid, tid = _winapi.CreateProcess(
                    python_exe, cmd,
                    None, None, False, 0, env, None, None)
                _winapi.CloseHandle(ht)
            except:
                _winapi.CloseHandle(rhandle)
                raise

            # set attributes of self
            self.pid = pid
            self.returncode = None
            self._handle = hp
            self.sentinel = int(hp)
            self.finalizer = util.Finalize(self, _close_handles,
                                           (self.sentinel, int(rhandle)))

            # send information to child
            set_spawning_popen(self)
            try:
                reduction.dump(prep_data, to_child)
                reduction.dump(process_obj, to_child)
            finally:
                set_spawning_popen(None)

    def duplicate_for_child(self, handle):
        assert self is get_spawning_popen()
        return reduction.duplicate(handle, self.sentinel)

    def wait(self, timeout=None):
        if self.returncode is None:
            if timeout is None:
                msecs = _winapi.INFINITE
            else:
                msecs = max(0, int(timeout * 1000 + 0.5))

            res = _winapi.WaitForSingleObject(int(self._handle), msecs)
            if res == _winapi.WAIT_OBJECT_0:
                code = _winapi.GetExitCodeProcess(self._handle)
                if code == TERMINATE:
                    code = -signal.SIGTERM
                self.returncode = code

        return self.returncode

    def poll(self):
        return self.wait(timeout=0)

    def terminate(self):
        if self.returncode is None:
            try:
                _winapi.TerminateProcess(int(self._handle), TERMINATE)
            except OSError:
                if self.wait(timeout=1.0) is None:
                    raise

    kill = terminate

    def close(self):
        self.finalizer()
<|endoftext|>#
# Module providing various facilities to other parts of the package
#
# multiprocessing/util.py
#
# Copyright (c) 2006-2008, R Oudkerk
# Licensed to PSF under a Contributor Agreement.
#

import os
import itertools
import sys
import weakref
import atexit
import threading        # we want threading to install it's
                        # cleanup function before multiprocessing does
from subprocess import _args_from_interpreter_flags

from . import process

__all__ = [
    'sub_debug', 'debug', 'info', 'sub_warning', 'get_logger',
    'log_to_stderr', 'get_temp_dir', 'register_after_fork',
    'is_exiting', 'Finalize', 'ForkAwareThreadLock', 'ForkAwareLocal',
    'close_all_fds_except', 'SUBDEBUG', 'SUBWARNING',
    ]

#
# Logging
#

NOTSET = 0
SUBDEBUG = 5
DEBUG = 10
INFO = 20
SUBWARNING = 25

LOGGER_NAME = 'multiprocess'
DEFAULT_LOGGING_FORMAT = '[%(levelname)s/%(processName)s] %(message)s'

_logger = None
_log_to_stderr = False

def sub_debug(msg, *args):
    if _logger:
        _logger.log(SUBDEBUG, msg, *args)

def debug(msg, *args):
    if _logger:
        _logger.log(DEBUG, msg, *args)

def info(msg, *args):
    if _logger:
        _logger.log(INFO, msg, *args)

def sub_warning(msg, *args):
    if _logger:
        _logger.log(SUBWARNING, msg, *args)

def get_logger():
    '''
    Returns logger used by multiprocess
    '''
    global _logger
    import logging

    logging._acquireLock()
    try:
        if not _logger:

            _logger = logging.getLogger(LOGGER_NAME)
            _logger.propagate = 0

            # XXX multiprocessing should cleanup before logging
            if hasattr(atexit, 'unregister'):
                atexit.unregister(_exit_function)
                atexit.register(_exit_function)
            else:
                atexit._exithandlers.remove((_exit_function, (), {}))
                atexit._exithandlers.append((_exit_function, (), {}))

    finally:
        logging._releaseLock()

    return _logger

def log_to_stderr(level=None):
    '''
    Turn on logging and add a handler which prints to stderr
    '''
    global _log_to_stderr
    import logging

    logger = get_logger()
    formatter = logging.Formatter(DEFAULT_LOGGING_FORMAT)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if level:
        logger.setLevel(level)
    _log_to_stderr = True
    return _logger


# Abstract socket support

def _platform_supports_abstract_sockets():
    if sys.platform == "linux":
        return True
    if hasattr(sys, 'getandroidapilevel'):
        return True
    return False


def is_abstract_socket_namespace(address):
    if not address:
        return False
    if isinstance(address, bytes):
        return address[0] == 0
    elif isinstance(address, str):
        return address[0] == "\0"
    raise TypeError(f'address type of {address!r} unrecognized')


abstract_sockets_supported = _platform_supports_abstract_sockets()

#
# Function returning a temp directory which will be removed on exit
#

def _remove_temp_dir(rmtree, tempdir):
    rmtree(tempdir)

    current_process = process.current_process()
    # current_process() can be None if the finalizer is called
    # late during Python finalization
    if current_process is not None:
        current_process._config['tempdir'] = None

def get_temp_dir():
    # get name of a temp directory which will be automatically cleaned up
    tempdir = process.current_process()._config.get('tempdir')
    if tempdir is None:
        import shutil, tempfile
        tempdir = tempfile.mkdtemp(prefix='pymp-')
        info('created temp directory %s', tempdir)
        # keep a strong reference to shutil.rmtree(), since the finalizer
        # can be called late during Python shutdown
        Finalize(None, _remove_temp_dir, args=(shutil.rmtree, tempdir),
                 exitpriority=-100)
        process.current_process()._config['tempdir'] = tempdir
    return tempdir

#
# Support for reinitialization of objects when bootstrapping a child process
#

_afterfork_registry = weakref.WeakValueDictionary()
_afterfork_counter = itertools.count()

def _run_after_forkers():
    items = list(_afterfork_registry.items())
    items.sort()
    for (index, ident, func), obj in items:
        try:
            func(obj)
        except Exception as e:
            info('after forker raised exception %s', e)

def register_after_fork(obj, func):
    _afterfork_registry[(next(_afterfork_counter), id(obj), func)] = obj

#
# Finalization using weakrefs
#

_finalizer_registry = {}
_finalizer_counter = itertools.count()


class Finalize(object):
    '''
    Class which supports object finalization using weakrefs
    '''
    def __init__(self, obj, callback, args=(), kwargs=None, exitpriority=None):
        if (exitpriority is not None) and not isinstance(exitpriority,int):
            raise TypeError(
                "Exitpriority ({0!r}) must be None or int, not {1!s}".format(
                    exitpriority, type(exitpriority)))

        if obj is not None:
            self._weakref = weakref.ref(obj, self)
        elif exitpriority is None:
            raise ValueError("Without object, exitpriority cannot be None")

        self._callback = callback
        self._args = args
        self._kwargs = kwargs or {}
        self._key = (exitpriority, next(_finalizer_counter))
        self._pid = os.getpid()

        _finalizer_registry[self._key] = self

    def __call__(self, wr=None,
                 # Need to bind these locally because the globals can have
                 # been cleared at shutdown
                 _finalizer_registry=_finalizer_registry,
                 sub_debug=sub_debug, getpid=os.getpid):
        '''
        Run the callback unless it has already been called or cancelled
        '''
        try:
            del _finalizer_registry[self._key]
        except KeyError:
            sub_debug('finalizer no longer registered')
        else:
            if self._pid != getpid():
                sub_debug('finalizer ignored because different process')
                res = None
            else:
                sub_debug('finalizer calling %s with args %s and kwargs %s',
                          self._callback, self._args, self._kwargs)
                res = self._callback(*self._args, **self._kwargs)
            self._weakref = self._callback = self._args = \
                            self._kwargs = self._key = None
            return res

    def cancel(self):
        '''
        Cancel finalization of the object
        '''
        try:
            del _finalizer_registry[self._key]
        except KeyError:
            pass
        else:
            self._weakref = self._callback = self._args = \
                            self._kwargs = self._key = None

    def still_active(self):
        '''
        Return whether this finalizer is still waiting to invoke callback
        '''
        return self._key in _finalizer_registry

    def __repr__(self):
        try:
            obj = self._weakref()
        except (AttributeError, TypeError):
            obj = None

        if obj is None:
            return '<%s object, dead>' % self.__class__.__name__

        x = '<%s object, callback=%s' % (
                self.__class__.__name__,
                getattr(self._callback, '__name__', self._callback))
        if self._args:
            x += ', args=' + str(self._args)
        if self._kwargs:
            x += ', kwargs=' + str(self._kwargs)
        if self._key[0] is not None:
            x += ', exitpriority=' + str(self._key[0])
        return x + '>'


def _run_finalizers(minpriority=None):
    '''
    Run all finalizers whose exit priority is not None and at least minpriority

    Finalizers with highest priority are called first; finalizers with
    the same priority will be called in reverse order of creation.
    '''
    if _finalizer_registry is None:
        # This function may be called after this module's globals are
        # destroyed.  See the _exit_function function in this module for more
        # notes.
        return

    if minpriority is None:
        f = lambda p : p[0] is not None
    else:
        f = lambda p : p[0] is not None and p[0] >= minpriority

    # Careful: _finalizer_registry may be mutated while this function
    # is running (either by a GC run or by another thread).

    # list(_finalizer_registry) should be atomic, while
    # list(_finalizer_registry.items()) is not.
    keys = [key for key in list(_finalizer_registry) if f(key)]
    keys.sort(reverse=True)

    for key in keys:
        finalizer = _finalizer_registry.get(key)
        # key may have been removed from the registry
        if finalizer is not None:
            sub_debug('calling %s', finalizer)
            try:
                finalizer()
            except Exception:
                import traceback
                traceback.print_exc()

    if minpriority is None:
        _finalizer_registry.clear()

#
# Clean up on exit
#

def is_exiting():
    '''
    Returns true if the process is shutting down
    '''
    return _exiting or _exiting is None

_exiting = False

def _exit_function(info=info, debug=debug, _run_finalizers=_run_finalizers,
                   active_children=process.active_children,
                   current_process=process.current_process):
    # We hold on to references to functions in the arglist due to the
    # situation described below, where this function is called after this
    # module's globals are destroyed.

    global _exiting

    if not _exiting:
        _exiting = True

        info('process shutting down')
        debug('running all "atexit" finalizers with priority >= 0')
        _run_finalizers(0)

        if current_process() is not None:
            # We check if the current process is None here because if
            # it's None, any call to ``active_children()`` will raise
            # an AttributeError (active_children winds up trying to
            # get attributes from util._current_process).  One
            # situation where this can happen is if someone has
            # manipulated sys.modules, causing this module to be
            # garbage collected.  The destructor for the module type
            # then replaces all values in the module dict with None.
            # For instance, after setuptools runs a test it replaces
            # sys.modules with a copy created earlier.  See issues
            # #9775 and #15881.  Also related: #4106, #9205, and
            # #9207.

            for p in active_children():
                if p.daemon:
                    info('calling terminate() for daemon %s', p.name)
                    p._popen.terminate()

            for p in active_children():
                info('calling join() for process %s', p.name)
                p.join()

        debug('running the remaining "atexit" finalizers')
        _run_finalizers()

atexit.register(_exit_function)

#
# Some fork aware types
#

class ForkAwareThreadLock(object):
    def __init__(self):
        self._lock = threading.Lock()
        self.acquire = self._lock.acquire
        self.release = self._lock.release
        register_after_fork(self, ForkAwareThreadLock._at_fork_reinit)

    def _at_fork_reinit(self):
        self._lock._at_fork_reinit()

    def __enter__(self):
        return self._lock.__enter__()

    def __exit__(self, *args):
        return self._lock.__exit__(*args)


class ForkAwareLocal(threading.local):
    def __init__(self):
        register_after_fork(self, lambda obj : obj.__dict__.clear())
    def __reduce__(self):
        return type(self), ()

#
# Close fds except those specified
#

try:
    MAXFD = os.sysconf("SC_OPEN_MAX")
except Exception:
    MAXFD = 256

def close_all_fds_except(fds):
    fds = list(fds) + [-1, MAXFD]
    fds.sort()
    assert fds[-1] == MAXFD, 'fd too large'
    for i in range(len(fds) - 1):
        os.closerange(fds[i]+1, fds[i+1])
#
# Close sys.stdin and replace stdin with os.devnull
#

def _close_stdin():
    if sys.stdin is None:
        return

    try:
        sys.stdin.close()
    except (OSError, ValueError):
        pass

    try:
        fd = os.open(os.devnull, os.O_RDONLY)
        try:
            sys.stdin = open(fd, encoding="utf-8", closefd=False)
        except:
            os.close(fd)
            raise
    except (OSError, ValueError):
        pass

#
# Flush standard streams, if any
#

def _flush_std_streams():
    try:
        sys.stdout.flush()
    except (AttributeError, ValueError):
        pass
    try:
        sys.stderr.flush()
    except (AttributeError, ValueError):
        pass

#
# Start a program with only specified fds kept open
#

def spawnv_passfds(path, args, passfds):
    import _posixsubprocess
    passfds = tuple(sorted(map(int, passfds)))
    errpipe_read, errpipe_write = os.pipe()
    try:
        return _posixsubprocess.fork_exec(
            args, [os.fsencode(path)], True, passfds, None, None,
            -1, -1, -1, -1, -1, -1, errpipe_read, errpipe_write,
            False, False, None, None, None, -1, None)
    finally:
        os.close(errpipe_read)
        os.close(errpipe_write)


def close_fds(*fds):
    """Close each file descriptor given as an argument"""
    for fd in fds:
        os.close(fd)


def _cleanup_tests():
    """Cleanup multiprocessing resources when multiprocessing tests
    completed."""

    from test import support

    # cleanup multiprocessing
    process._cleanup()

    # Stop the ForkServer process if it's running
    from multiprocess import forkserver
    forkserver._forkserver._stop()

    # Stop the ResourceTracker process if it's running
    from multiprocess import resource_tracker
    resource_tracker._resource_tracker._stop()

    # bpo-37421: Explicitly call _run_finalizers() to remove immediately
    # temporary directories created by multiprocessing.util.get_temp_dir().
    _run_finalizers()
    support.gc_collect()

    support.reap_children()
<|endoftext|>#
# Module implementing synchronization primitives
#
# multiprocessing/synchronize.py
#
# Copyright (c) 2006-2008, R Oudkerk
# Licensed to PSF under a Contributor Agreement.
#

__all__ = [
    'Lock', 'RLock', 'Semaphore', 'BoundedSemaphore', 'Condition', 'Event'
    ]

import threading
import sys
import tempfile
try:
    import _multiprocess as _multiprocessing
except ImportError:
    import _multiprocessing
import time

from . import context
from . import process
from . import util

# Try to import the mp.synchronize module cleanly, if it fails
# raise ImportError for platforms lacking a working sem_open implementation.
# See issue 3770
try:
    from _multiprocess import SemLock, sem_unlink
except ImportError:
    try:
        from _multiprocessing import SemLock, sem_unlink
    except (ImportError):
        raise ImportError("This platform lacks a functioning sem_open" +
                          " implementation, therefore, the required" +
                          " synchronization primitives needed will not" +
                          " function, see issue 3770.")

#
# Constants
#

RECURSIVE_MUTEX, SEMAPHORE = list(range(2))
SEM_VALUE_MAX = _multiprocessing.SemLock.SEM_VALUE_MAX

#
# Base class for semaphores and mutexes; wraps `_multiprocessing.SemLock`
#

class SemLock(object):

    _rand = tempfile._RandomNameSequence()

    def __init__(self, kind, value, maxvalue, *, ctx):
        if ctx is None:
            ctx = context._default_context.get_context()
        name = ctx.get_start_method()
        unlink_now = sys.platform == 'win32' or name == 'fork'
        for i in range(100):
            try:
                sl = self._semlock = _multiprocessing.SemLock(
                    kind, value, maxvalue, self._make_name(),
                    unlink_now)
            except FileExistsError:
                pass
            else:
                break
        else:
            raise FileExistsError('cannot find name for semaphore')

        util.debug('created semlock with handle %s' % sl.handle)
        self._make_methods()

        if sys.platform != 'win32':
            def _after_fork(obj):
                obj._semlock._after_fork()
            util.register_after_fork(self, _after_fork)

        if self._semlock.name is not None:
            # We only get here if we are on Unix with forking
            # disabled.  When the object is garbage collected or the
            # process shuts down we unlink the semaphore name
            from .resource_tracker import register
            register(self._semlock.name, "semaphore")
            util.Finalize(self, SemLock._cleanup, (self._semlock.name,),
                          exitpriority=0)

    @staticmethod
    def _cleanup(name):
        from .resource_tracker import unregister
        sem_unlink(name)
        unregister(name, "semaphore")

    def _make_methods(self):
        self.acquire = self._semlock.acquire
        self.release = self._semlock.release

    def __enter__(self):
        return self._semlock.__enter__()

    def __exit__(self, *args):
        return self._semlock.__exit__(*args)

    def __getstate__(self):
        context.assert_spawning(self)
        sl = self._semlock
        if sys.platform == 'win32':
            h = context.get_spawning_popen().duplicate_for_child(sl.handle)
        else:
            h = sl.handle
        return (h, sl.kind, sl.maxvalue, sl.name)

    def __setstate__(self, state):
        self._semlock = _multiprocessing.SemLock._rebuild(*state)
        util.debug('recreated blocker with handle %r' % state[0])
        self._make_methods()

    @staticmethod
    def _make_name():
        return '%s-%s' % (process.current_process()._config['semprefix'],
                          next(SemLock._rand))

#
# Semaphore
#

class Semaphore(SemLock):

    def __init__(self, value=1, *, ctx):
        SemLock.__init__(self, SEMAPHORE, value, SEM_VALUE_MAX, ctx=ctx)

    def get_value(self):
        return self._semlock._get_value()

    def __repr__(self):
        try:
            value = self._semlock._get_value()
        except Exception:
            value = 'unknown'
        return '<%s(value=%s)>' % (self.__class__.__name__, value)

#
# Bounded semaphore
#

class BoundedSemaphore(Semaphore):

    def __init__(self, value=1, *, ctx):
        SemLock.__init__(self, SEMAPHORE, value, value, ctx=ctx)

    def __repr__(self):
        try:
            value = self._semlock._get_value()
        except Exception:
            value = 'unknown'
        return '<%s(value=%s, maxvalue=%s)>' % \
               (self.__class__.__name__, value, self._semlock.maxvalue)

#
# Non-recursive lock
#

class Lock(SemLock):

    def __init__(self, *, ctx):
        SemLock.__init__(self, SEMAPHORE, 1, 1, ctx=ctx)

    def __repr__(self):
        try:
            if self._semlock._is_mine():
                name = process.current_process().name
                if threading.current_thread().name != 'MainThread':
                    name += '|' + threading.current_thread().name
            elif self._semlock._get_value() == 1:
                name = 'None'
            elif self._semlock._count() > 0:
                name = 'SomeOtherThread'
            else:
                name = 'SomeOtherProcess'
        except Exception:
            name = 'unknown'
        return '<%s(owner=%s)>' % (self.__class__.__name__, name)

#
# Recursive lock
#

class RLock(SemLock):

    def __init__(self, *, ctx):
        SemLock.__init__(self, RECURSIVE_MUTEX, 1, 1, ctx=ctx)

    def __repr__(self):
        try:
            if self._semlock._is_mine():
                name = process.current_process().name
                if threading.current_thread().name != 'MainThread':
                    name += '|' + threading.current_thread().name
                count = self._semlock._count()
            elif self._semlock._get_value() == 1:
                name, count = 'None', 0
            elif self._semlock._count() > 0:
                name, count = 'SomeOtherThread', 'nonzero'
            else:
                name, count = 'SomeOtherProcess', 'nonzero'
        except Exception:
            name, count = 'unknown', 'unknown'
        return '<%s(%s, %s)>' % (self.__class__.__name__, name, count)

#
# Condition variable
#

class Condition(object):

    def __init__(self, lock=None, *, ctx):
        self._lock = lock or ctx.RLock()
        self._sleeping_count = ctx.Semaphore(0)
        self._woken_count = ctx.Semaphore(0)
        self._wait_semaphore = ctx.Semaphore(0)
        self._make_methods()

    def __getstate__(self):
        context.assert_spawning(self)
        return (self._lock, self._sleeping_count,
                self._woken_count, self._wait_semaphore)

    def __setstate__(self, state):
        (self._lock, self._sleeping_count,
         self._woken_count, self._wait_semaphore) = state
        self._make_methods()

    def __enter__(self):
        return self._lock.__enter__()

    def __exit__(self, *args):
        return self._lock.__exit__(*args)

    def _make_methods(self):
        self.acquire = self._lock.acquire
        self.release = self._lock.release

    def __repr__(self):
        try:
            num_waiters = (self._sleeping_count._semlock._get_value() -
                           self._woken_count._semlock._get_value())
        except Exception:
            num_waiters = 'unknown'
        return '<%s(%s, %s)>' % (self.__class__.__name__, self._lock, num_waiters)

    def wait(self, timeout=None):
        assert self._lock._semlock._is_mine(), \
               'must acquire() condition before using wait()'

        # indicate that this thread is going to sleep
        self._sleeping_count.release()

        # release lock
        count = self._lock._semlock._count()
        for i in range(count):
            self._lock.release()

        try:
            # wait for notification or timeout
            return self._wait_semaphore.acquire(True, timeout)
        finally:
            # indicate that this thread has woken
            self._woken_count.release()

            # reacquire lock
            for i in range(count):
                self._lock.acquire()

    def notify(self, n=1):
        assert self._lock._semlock._is_mine(), 'lock is not owned'
        assert not self._wait_semaphore.acquire(
            False), ('notify: Should not have been able to acquire '
                     + '_wait_semaphore')

        # to take account of timeouts since last notify*() we subtract
        # woken_count from sleeping_count and rezero woken_count
        while self._woken_count.acquire(False):
            res = self._sleeping_count.acquire(False)
            assert res, ('notify: Bug in sleeping_count.acquire'
                         + '- res should not be False')

        sleepers = 0
        while sleepers < n and self._sleeping_count.acquire(False):
            self._wait_semaphore.release()        # wake up one sleeper
            sleepers += 1

        if sleepers:
            for i in range(sleepers):
                self._woken_count.acquire()       # wait for a sleeper to wake

            # rezero wait_semaphore in case some timeouts just happened
            while self._wait_semaphore.acquire(False):
                pass

    def notify_all(self):
        self.notify(n=sys.maxsize)

    def wait_for(self, predicate, timeout=None):
        result = predicate()
        if result:
            return result
        if timeout is not None:
            endtime = getattr(time,'monotonic',time.time)() + timeout
        else:
            endtime = None
            waittime = None
        while not result:
            if endtime is not None:
                waittime = endtime - getattr(time,'monotonic',time.time)()
                if waittime <= 0:
                    break
            self.wait(waittime)
            result = predicate()
        return result

#
# Event
#

class Event(object):

    def __init__(self, *, ctx):
        self._cond = ctx.Condition(ctx.Lock())
        self._flag = ctx.Semaphore(0)

    def is_set(self):
        with self._cond:
            if self._flag.acquire(False):
                self._flag.release()
                return True
            return False

    def set(self):
        with self._cond:
            self._flag.acquire(False)
            self._flag.release()
            self._cond.notify_all()

    def clear(self):
        with self._cond:
            self._flag.acquire(False)

    def wait(self, timeout=None):
        with self._cond:
            if self._flag.acquire(False):
                self._flag.release()
            else:
                self._cond.wait(timeout)

            if self._flag.acquire(False):
                self._flag.release()
                return True
            return False

#
# Barrier
#

class Barrier(threading.Barrier):

    def __init__(self, parties, action=None, timeout=None, *, ctx):
        import struct
        from .heap import BufferWrapper
        wrapper = BufferWrapper(struct.calcsize('i') * 2)
        cond = ctx.Condition()
        self.__setstate__((parties, action, timeout, cond, wrapper))
        self._state = 0
        self._count = 0

    def __setstate__(self, state):
        (self._parties, self._action, self._timeout,
         self._cond, self._wrapper) = state
        self._array = self._wrapper.create_memoryview().cast('i')

    def __getstate__(self):
        return (self._parties, self._action, self._timeout,
                self._cond, self._wrapper)

    @property
    def _state(self):
        return self._array[0]

    @_state.setter
    def _state(self, value):
        self._array[0] = value

    @property
    def _count(self):
        return self._array[1]

    @_count.setter
    def _count(self, value):
        self._array[1] = value
<|endoftext|>#
# Package analogous to 'threading.py' but using processes
#
# multiprocessing/__init__.py
#
# This package is intended to duplicate the functionality (and much of
# the API) of threading.py but uses processes instead of threads.  A
# subpackage 'multiprocessing.dummy' has the same API but is a simple
# wrapper for 'threading'.
#
# Original: Copyright (c) 2006-2008, R Oudkerk
# Original: Licensed to PSF under a Contributor Agreement.
# Forked by Mike McKerns, to support enhanced serialization.

# author, version, license, and long description
try: # the package is installed
    from .__info__ import __version__, __author__, __doc__, __license__
except: # pragma: no cover
    import os
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    sys.path.append(root)
    # get distribution meta info 
    from version import (__version__, __author__,
                         get_license_text, get_readme_as_rst)
    __license__ = get_license_text(os.path.join(root, 'LICENSE'))
    __license__ = "\n%s" % __license__
    __doc__ = get_readme_as_rst(os.path.join(root, 'README.md'))
    del os, sys, root, get_license_text, get_readme_as_rst


import sys
from . import context

#
# Copy stuff from default context
#

__all__ = [x for x in dir(context._default_context) if not x.startswith('_')]
globals().update((name, getattr(context._default_context, name)) for name in __all__)

#
# XXX These should not really be documented or public.
#

SUBDEBUG = 5
SUBWARNING = 25

#
# Alias for main module -- will be reset by bootstrapping child processes
#

if '__main__' in sys.modules:
    sys.modules['__mp_main__'] = sys.modules['__main__']


def license():
    """print license"""
    print (__license__)
    return

def citation():
    """print citation"""
    print (__doc__[-491:-118])
    return

<|endoftext|>#
# Module implementing queues
#
# multiprocessing/queues.py
#
# Copyright (c) 2006-2008, R Oudkerk
# Licensed to PSF under a Contributor Agreement.
#

__all__ = ['Queue', 'SimpleQueue', 'JoinableQueue']

import sys
import os
import threading
import collections
import time
import types
import weakref
import errno

from queue import Empty, Full

try:
    import _multiprocess as _multiprocessing
except ImportError:
    import _multiprocessing

from . import connection
from . import context
_ForkingPickler = context.reduction.ForkingPickler

from .util import debug, info, Finalize, register_after_fork, is_exiting

#
# Queue type using a pipe, buffer and thread
#

class Queue(object):

    def __init__(self, maxsize=0, *, ctx):
        if maxsize <= 0:
            # Can raise ImportError (see issues #3770 and #23400)
            from .synchronize import SEM_VALUE_MAX as maxsize
        self._maxsize = maxsize
        self._reader, self._writer = connection.Pipe(duplex=False)
        self._rlock = ctx.Lock()
        self._opid = os.getpid()
        if sys.platform == 'win32':
            self._wlock = None
        else:
            self._wlock = ctx.Lock()
        self._sem = ctx.BoundedSemaphore(maxsize)
        # For use by concurrent.futures
        self._ignore_epipe = False
        self._reset()

        if sys.platform != 'win32':
            register_after_fork(self, Queue._after_fork)

    def __getstate__(self):
        context.assert_spawning(self)
        return (self._ignore_epipe, self._maxsize, self._reader, self._writer,
                self._rlock, self._wlock, self._sem, self._opid)

    def __setstate__(self, state):
        (self._ignore_epipe, self._maxsize, self._reader, self._writer,
         self._rlock, self._wlock, self._sem, self._opid) = state
        self._reset()

    def _after_fork(self):
        debug('Queue._after_fork()')
        self._reset(after_fork=True)

    def _reset(self, after_fork=False):
        if after_fork:
            self._notempty._at_fork_reinit()
        else:
            self._notempty = threading.Condition(threading.Lock())
        self._buffer = collections.deque()
        self._thread = None
        self._jointhread = None
        self._joincancelled = False
        self._closed = False
        self._close = None
        self._send_bytes = self._writer.send_bytes
        self._recv_bytes = self._reader.recv_bytes
        self._poll = self._reader.poll

    def put(self, obj, block=True, timeout=None):
        if self._closed:
            raise ValueError(f"Queue {self!r} is closed")
        if not self._sem.acquire(block, timeout):
            raise Full

        with self._notempty:
            if self._thread is None:
                self._start_thread()
            self._buffer.append(obj)
            self._notempty.notify()

    def get(self, block=True, timeout=None):
        if self._closed:
            raise ValueError(f"Queue {self!r} is closed")
        if block and timeout is None:
            with self._rlock:
                res = self._recv_bytes()
            self._sem.release()
        else:
            if block:
                deadline = getattr(time,'monotonic',time.time)() + timeout
            if not self._rlock.acquire(block, timeout):
                raise Empty
            try:
                if block:
                    timeout = deadline - getattr(time,'monotonic',time.time)()
                    if not self._poll(timeout):
                        raise Empty
                elif not self._poll():
                    raise Empty
                res = self._recv_bytes()
                self._sem.release()
            finally:
                self._rlock.release()
        # unserialize the data after having released the lock
        return _ForkingPickler.loads(res)

    def qsize(self):
        # Raises NotImplementedError on Mac OSX because of broken sem_getvalue()
        return self._maxsize - self._sem._semlock._get_value()

    def empty(self):
        return not self._poll()

    def full(self):
        return self._sem._semlock._is_zero()

    def get_nowait(self):
        return self.get(False)

    def put_nowait(self, obj):
        return self.put(obj, False)

    def close(self):
        self._closed = True
        close = self._close
        if close:
            self._close = None
            close()

    def join_thread(self):
        debug('Queue.join_thread()')
        assert self._closed, "Queue {0!r} not closed".format(self)
        if self._jointhread:
            self._jointhread()

    def cancel_join_thread(self):
        debug('Queue.cancel_join_thread()')
        self._joincancelled = True
        try:
            self._jointhread.cancel()
        except AttributeError:
            pass

    def _start_thread(self):
        debug('Queue._start_thread()')

        # Start thread which transfers data from buffer to pipe
        self._buffer.clear()
        self._thread = threading.Thread(
            target=Queue._feed,
            args=(self._buffer, self._notempty, self._send_bytes,
                  self._wlock, self._reader.close, self._writer.close,
                  self._ignore_epipe, self._on_queue_feeder_error,
                  self._sem),
            name='QueueFeederThread'
        )
        self._thread.daemon = True

        debug('doing self._thread.start()')
        self._thread.start()
        debug('... done self._thread.start()')

        if not self._joincancelled:
            self._jointhread = Finalize(
                self._thread, Queue._finalize_join,
                [weakref.ref(self._thread)],
                exitpriority=-5
                )

        # Send sentinel to the thread queue object when garbage collected
        self._close = Finalize(
            self, Queue._finalize_close,
            [self._buffer, self._notempty],
            exitpriority=10
            )

    @staticmethod
    def _finalize_join(twr):
        debug('joining queue thread')
        thread = twr()
        if thread is not None:
            thread.join()
            debug('... queue thread joined')
        else:
            debug('... queue thread already dead')

    @staticmethod
    def _finalize_close(buffer, notempty):
        debug('telling queue thread to quit')
        with notempty:
            buffer.append(_sentinel)
            notempty.notify()

    @staticmethod
    def _feed(buffer, notempty, send_bytes, writelock, reader_close,
              writer_close, ignore_epipe, onerror, queue_sem):
        debug('starting thread to feed data to pipe')
        nacquire = notempty.acquire
        nrelease = notempty.release
        nwait = notempty.wait
        bpopleft = buffer.popleft
        sentinel = _sentinel
        if sys.platform != 'win32':
            wacquire = writelock.acquire
            wrelease = writelock.release
        else:
            wacquire = None

        while 1:
            try:
                nacquire()
                try:
                    if not buffer:
                        nwait()
                finally:
                    nrelease()
                try:
                    while 1:
                        obj = bpopleft()
                        if obj is sentinel:
                            debug('feeder thread got sentinel -- exiting')
                            reader_close()
                            writer_close()
                            return

                        # serialize the data before acquiring the lock
                        obj = _ForkingPickler.dumps(obj)
                        if wacquire is None:
                            send_bytes(obj)
                        else:
                            wacquire()
                            try:
                                send_bytes(obj)
                            finally:
                                wrelease()
                except IndexError:
                    pass
            except Exception as e:
                if ignore_epipe and getattr(e, 'errno', 0) == errno.EPIPE:
                    return
                # Since this runs in a daemon thread the resources it uses
                # may be become unusable while the process is cleaning up.
                # We ignore errors which happen after the process has
                # started to cleanup.
                if is_exiting():
                    info('error in queue thread: %s', e)
                    return
                else:
                    # Since the object has not been sent in the queue, we need
                    # to decrease the size of the queue. The error acts as
                    # if the object had been silently removed from the queue
                    # and this step is necessary to have a properly working
                    # queue.
                    queue_sem.release()
                    onerror(e, obj)

    @staticmethod
    def _on_queue_feeder_error(e, obj):
        """
        Private API hook called when feeding data in the background thread
        raises an exception.  For overriding by concurrent.futures.
        """
        import traceback
        traceback.print_exc()


_sentinel = object()

#
# A queue type which also supports join() and task_done() methods
#
# Note that if you do not call task_done() for each finished task then
# eventually the counter's semaphore may overflow causing Bad Things
# to happen.
#

class JoinableQueue(Queue):

    def __init__(self, maxsize=0, *, ctx):
        Queue.__init__(self, maxsize, ctx=ctx)
        self._unfinished_tasks = ctx.Semaphore(0)
        self._cond = ctx.Condition()

    def __getstate__(self):
        return Queue.__getstate__(self) + (self._cond, self._unfinished_tasks)

    def __setstate__(self, state):
        Queue.__setstate__(self, state[:-2])
        self._cond, self._unfinished_tasks = state[-2:]

    def put(self, obj, block=True, timeout=None):
        if self._closed:
            raise ValueError(f"Queue {self!r} is closed")
        if not self._sem.acquire(block, timeout):
            raise Full

        with self._notempty, self._cond:
            if self._thread is None:
                self._start_thread()
            self._buffer.append(obj)
            self._unfinished_tasks.release()
            self._notempty.notify()

    def task_done(self):
        with self._cond:
            if not self._unfinished_tasks.acquire(False):
                raise ValueError('task_done() called too many times')
            if self._unfinished_tasks._semlock._is_zero():
                self._cond.notify_all()

    def join(self):
        with self._cond:
            if not self._unfinished_tasks._semlock._is_zero():
                self._cond.wait()

#
# Simplified Queue type -- really just a locked pipe
#

class SimpleQueue(object):

    def __init__(self, *, ctx):
        self._reader, self._writer = connection.Pipe(duplex=False)
        self._rlock = ctx.Lock()
        self._poll = self._reader.poll
        if sys.platform == 'win32':
            self._wlock = None
        else:
            self._wlock = ctx.Lock()

    def close(self):
        self._reader.close()
        self._writer.close()

    def empty(self):
        return not self._poll()

    def __getstate__(self):
        context.assert_spawning(self)
        return (self._reader, self._writer, self._rlock, self._wlock)

    def __setstate__(self, state):
        (self._reader, self._writer, self._rlock, self._wlock) = state
        self._poll = self._reader.poll

    def get(self):
        with self._rlock:
            res = self._reader.recv_bytes()
        # unserialize the data after having released the lock
        return _ForkingPickler.loads(res)

    def put(self, obj):
        # serialize the data before acquiring the lock
        obj = _ForkingPickler.dumps(obj)
        if self._wlock is None:
            # writes to a message oriented win32 pipe are atomic
            self._writer.send_bytes(obj)
        else:
            with self._wlock:
                self._writer.send_bytes(obj)

    __class_getitem__ = classmethod(types.GenericAlias)
<|endoftext|>###############################################################################
# Server process to keep track of unlinked resources (like shared memory
# segments, semaphores etc.) and clean them.
#
# On Unix we run a server process which keeps track of unlinked
# resources. The server ignores SIGINT and SIGTERM and reads from a
# pipe.  Every other process of the program has a copy of the writable
# end of the pipe, so we get EOF when all other processes have exited.
# Then the server process unlinks any remaining resource names.
#
# This is important because there may be system limits for such resources: for
# instance, the system only supports a limited number of named semaphores, and
# shared-memory segments live in the RAM. If a python process leaks such a
# resource, this resource will not be removed till the next reboot.  Without
# this resource tracker process, "killall python" would probably leave unlinked
# resources.

import os
import signal
import sys
import threading
import warnings

from . import spawn
from . import util

__all__ = ['ensure_running', 'register', 'unregister']

_HAVE_SIGMASK = hasattr(signal, 'pthread_sigmask')
_IGNORED_SIGNALS = (signal.SIGINT, signal.SIGTERM)

_CLEANUP_FUNCS = {
    'noop': lambda: None,
}

if os.name == 'posix':
    try:
        import _multiprocess as _multiprocessing
    except ImportError:
        import _multiprocessing
    import _posixshmem

    # Use sem_unlink() to clean up named semaphores.
    #
    # sem_unlink() may be missing if the Python build process detected the
    # absence of POSIX named semaphores. In that case, no named semaphores were
    # ever opened, so no cleanup would be necessary.
    if hasattr(_multiprocessing, 'sem_unlink'):
        _CLEANUP_FUNCS.update({
            'semaphore': _multiprocessing.sem_unlink,
        })
    _CLEANUP_FUNCS.update({
        'shared_memory': _posixshmem.shm_unlink,
    })


class ResourceTracker(object):

    def __init__(self):
        self._lock = threading.Lock()
        self._fd = None
        self._pid = None

    def _stop(self):
        with self._lock:
            if self._fd is None:
                # not running
                return

            # closing the "alive" file descriptor stops main()
            os.close(self._fd)
            self._fd = None

            os.waitpid(self._pid, 0)
            self._pid = None

    def getfd(self):
        self.ensure_running()
        return self._fd

    def ensure_running(self):
        '''Make sure that resource tracker process is running.

        This can be run from any process.  Usually a child process will use
        the resource created by its parent.'''
        with self._lock:
            if self._fd is not None:
                # resource tracker was launched before, is it still running?
                if self._check_alive():
                    # => still alive
                    return
                # => dead, launch it again
                os.close(self._fd)

                # Clean-up to avoid dangling processes.
                try:
                    # _pid can be None if this process is a child from another
                    # python process, which has started the resource_tracker.
                    if self._pid is not None:
                        os.waitpid(self._pid, 0)
                except ChildProcessError:
                    # The resource_tracker has already been terminated.
                    pass
                self._fd = None
                self._pid = None

                warnings.warn('resource_tracker: process died unexpectedly, '
                              'relaunching.  Some resources might leak.')

            fds_to_pass = []
            try:
                fds_to_pass.append(sys.stderr.fileno())
            except Exception:
                pass
            cmd = 'from multiprocess.resource_tracker import main;main(%d)'
            r, w = os.pipe()
            try:
                fds_to_pass.append(r)
                # process will out live us, so no need to wait on pid
                exe = spawn.get_executable()
                args = [exe] + util._args_from_interpreter_flags()
                args += ['-c', cmd % r]
                # bpo-33613: Register a signal mask that will block the signals.
                # This signal mask will be inherited by the child that is going
                # to be spawned and will protect the child from a race condition
                # that can make the child die before it registers signal handlers
                # for SIGINT and SIGTERM. The mask is unregistered after spawning
                # the child.
                try:
                    if _HAVE_SIGMASK:
                        signal.pthread_sigmask(signal.SIG_BLOCK, _IGNORED_SIGNALS)
                    pid = util.spawnv_passfds(exe, args, fds_to_pass)
                finally:
                    if _HAVE_SIGMASK:
                        signal.pthread_sigmask(signal.SIG_UNBLOCK, _IGNORED_SIGNALS)
            except:
                os.close(w)
                raise
            else:
                self._fd = w
                self._pid = pid
            finally:
                os.close(r)

    def _check_alive(self):
        '''Check that the pipe has not been closed by sending a probe.'''
        try:
            # We cannot use send here as it calls ensure_running, creating
            # a cycle.
            os.write(self._fd, b'PROBE:0:noop\n')
        except OSError:
            return False
        else:
            return True

    def register(self, name, rtype):
        '''Register name of resource with resource tracker.'''
        self._send('REGISTER', name, rtype)

    def unregister(self, name, rtype):
        '''Unregister name of resource with resource tracker.'''
        self._send('UNREGISTER', name, rtype)

    def _send(self, cmd, name, rtype):
        self.ensure_running()
        msg = '{0}:{1}:{2}\n'.format(cmd, name, rtype).encode('ascii')
        if len(msg) > 512:
            # posix guarantees that writes to a pipe of less than PIPE_BUF
            # bytes are atomic, and that PIPE_BUF >= 512
            raise ValueError('msg too long')
        nbytes = os.write(self._fd, msg)
        assert nbytes == len(msg), "nbytes {0:n} but len(msg) {1:n}".format(
            nbytes, len(msg))


_resource_tracker = ResourceTracker()
ensure_running = _resource_tracker.ensure_running
register = _resource_tracker.register
unregister = _resource_tracker.unregister
getfd = _resource_tracker.getfd

def main(fd):
    '''Run resource tracker.'''
    # protect the process from ^C and "killall python" etc
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if _HAVE_SIGMASK:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, _IGNORED_SIGNALS)

    for f in (sys.stdin, sys.stdout):
        try:
            f.close()
        except Exception:
            pass

    cache = {rtype: set() for rtype in _CLEANUP_FUNCS.keys()}
    try:
        # keep track of registered/unregistered resources
        with open(fd, 'rb') as f:
            for line in f:
                try:
                    cmd, name, rtype = line.strip().decode('ascii').split(':')
                    cleanup_func = _CLEANUP_FUNCS.get(rtype, None)
                    if cleanup_func is None:
                        raise ValueError(
                            f'Cannot register {name} for automatic cleanup: '
                            f'unknown resource type {rtype}')

                    if cmd == 'REGISTER':
                        cache[rtype].add(name)
                    elif cmd == 'UNREGISTER':
                        cache[rtype].remove(name)
                    elif cmd == 'PROBE':
                        pass
                    else:
                        raise RuntimeError('unrecognized command %r' % cmd)
                except Exception:
                    try:
                        sys.excepthook(*sys.exc_info())
                    except:
                        pass
    finally:
        # all processes have terminated; cleanup any remaining resources
        for rtype, rtype_cache in cache.items():
            if rtype_cache:
                try:
                    warnings.warn('resource_tracker: There appear to be %d '
                                  'leaked %s objects to clean up at shutdown' %
                                  (len(rtype_cache), rtype))
                except Exception:
                    pass
            for name in rtype_cache:
                # For some reason the process which created and registered this
                # resource has failed to unregister it. Presumably it has
                # died.  We therefore unlink it.
                try:
                    try:
                        _CLEANUP_FUNCS[rtype](name)
                    except Exception as e:
                        warnings.warn('resource_tracker: %r: %s' % (name, e))
                finally:
                    pass
<|endoftext|>#
# Module which supports allocation of ctypes objects from shared memory
#
# multiprocessing/sharedctypes.py
#
# Copyright (c) 2006-2008, R Oudkerk
# Licensed to PSF under a Contributor Agreement.
#

import ctypes
import weakref

from . import heap
from . import get_context

from .context import reduction, assert_spawning
_ForkingPickler = reduction.ForkingPickler

__all__ = ['RawValue', 'RawArray', 'Value', 'Array', 'copy', 'synchronized']

#
#
#

typecode_to_type = {
    'c': ctypes.c_char,     'u': ctypes.c_wchar,
    'b': ctypes.c_byte,     'B': ctypes.c_ubyte,
    'h': ctypes.c_short,    'H': ctypes.c_ushort,
    'i': ctypes.c_int,      'I': ctypes.c_uint,
    'l': ctypes.c_long,     'L': ctypes.c_ulong,
    'q': ctypes.c_longlong, 'Q': ctypes.c_ulonglong,
    'f': ctypes.c_float,    'd': ctypes.c_double
    }

#
#
#

def _new_value(type_):
    size = ctypes.sizeof(type_)
    wrapper = heap.BufferWrapper(size)
    return rebuild_ctype(type_, wrapper, None)

def RawValue(typecode_or_type, *args):
    '''
    Returns a ctypes object allocated from shared memory
    '''
    type_ = typecode_to_type.get(typecode_or_type, typecode_or_type)
    obj = _new_value(type_)
    ctypes.memset(ctypes.addressof(obj), 0, ctypes.sizeof(obj))
    obj.__init__(*args)
    return obj

def RawArray(typecode_or_type, size_or_initializer):
    '''
    Returns a ctypes array allocated from shared memory
    '''
    type_ = typecode_to_type.get(typecode_or_type, typecode_or_type)
    if isinstance(size_or_initializer, int):
        type_ = type_ * size_or_initializer
        obj = _new_value(type_)
        ctypes.memset(ctypes.addressof(obj), 0, ctypes.sizeof(obj))
        return obj
    else:
        type_ = type_ * len(size_or_initializer)
        result = _new_value(type_)
        result.__init__(*size_or_initializer)
        return result

def Value(typecode_or_type, *args, lock=True, ctx=None):
    '''
    Return a synchronization wrapper for a Value
    '''
    obj = RawValue(typecode_or_type, *args)
    if lock is False:
        return obj
    if lock in (True, None):
        ctx = ctx or get_context()
        lock = ctx.RLock()
    if not hasattr(lock, 'acquire'):
        raise AttributeError("%r has no method 'acquire'" % lock)
    return synchronized(obj, lock, ctx=ctx)

def Array(typecode_or_type, size_or_initializer, *, lock=True, ctx=None):
    '''
    Return a synchronization wrapper for a RawArray
    '''
    obj = RawArray(typecode_or_type, size_or_initializer)
    if lock is False:
        return obj
    if lock in (True, None):
        ctx = ctx or get_context()
        lock = ctx.RLock()
    if not hasattr(lock, 'acquire'):
        raise AttributeError("%r has no method 'acquire'" % lock)
    return synchronized(obj, lock, ctx=ctx)

def copy(obj):
    new_obj = _new_value(type(obj))
    ctypes.pointer(new_obj)[0] = obj
    return new_obj

def synchronized(obj, lock=None, ctx=None):
    assert not isinstance(obj, SynchronizedBase), 'object already synchronized'
    ctx = ctx or get_context()

    if isinstance(obj, ctypes._SimpleCData):
        return Synchronized(obj, lock, ctx)
    elif isinstance(obj, ctypes.Array):
        if obj._type_ is ctypes.c_char:
            return SynchronizedString(obj, lock, ctx)
        return SynchronizedArray(obj, lock, ctx)
    else:
        cls = type(obj)
        try:
            scls = class_cache[cls]
        except KeyError:
            names = [field[0] for field in cls._fields_]
            d = {name: make_property(name) for name in names}
            classname = 'Synchronized' + cls.__name__
            scls = class_cache[cls] = type(classname, (SynchronizedBase,), d)
        return scls(obj, lock, ctx)

#
# Functions for pickling/unpickling
#

def reduce_ctype(obj):
    assert_spawning(obj)
    if isinstance(obj, ctypes.Array):
        return rebuild_ctype, (obj._type_, obj._wrapper, obj._length_)
    else:
        return rebuild_ctype, (type(obj), obj._wrapper, None)

def rebuild_ctype(type_, wrapper, length):
    if length is not None:
        type_ = type_ * length
    _ForkingPickler.register(type_, reduce_ctype)
    buf = wrapper.create_memoryview()
    obj = type_.from_buffer(buf)
    obj._wrapper = wrapper
    return obj

#
# Function to create properties
#

def make_property(name):
    try:
        return prop_cache[name]
    except KeyError:
        d = {}
        exec(template % ((name,)*7), d)
        prop_cache[name] = d[name]
        return d[name]

template = '''
def get%s(self):
    self.acquire()
    try:
        return self._obj.%s
    finally:
        self.release()
def set%s(self, value):
    self.acquire()
    try:
        self._obj.%s = value
    finally:
        self.release()
%s = property(get%s, set%s)
'''

prop_cache = {}
class_cache = weakref.WeakKeyDictionary()

#
# Synchronized wrappers
#

class SynchronizedBase(object):

    def __init__(self, obj, lock=None, ctx=None):
        self._obj = obj
        if lock:
            self._lock = lock
        else:
            ctx = ctx or get_context(force=True)
            self._lock = ctx.RLock()
        self.acquire = self._lock.acquire
        self.release = self._lock.release

    def __enter__(self):
        return self._lock.__enter__()

    def __exit__(self, *args):
        return self._lock.__exit__(*args)

    def __reduce__(self):
        assert_spawning(self)
        return synchronized, (self._obj, self._lock)

    def get_obj(self):
        return self._obj

    def get_lock(self):
        return self._lock

    def __repr__(self):
        return '<%s wrapper for %s>' % (type(self).__name__, self._obj)


class Synchronized(SynchronizedBase):
    value = make_property('value')


class SynchronizedArray(SynchronizedBase):

    def __len__(self):
        return len(self._obj)

    def __getitem__(self, i):
        with self:
            return self._obj[i]

    def __setitem__(self, i, value):
        with self:
            self._obj[i] = value

    def __getslice__(self, start, stop):
        with self:
            return self._obj[start:stop]

    def __setslice__(self, start, stop, values):
        with self:
            self._obj[start:stop] = values


class SynchronizedString(SynchronizedArray):
    value = make_property('value')
    raw = make_property('raw')
<|endoftext|>import io
import os

from .context import reduction, set_spawning_popen
if not reduction.HAVE_SEND_HANDLE:
    raise ImportError('No support for sending fds between processes')
from . import forkserver
from . import popen_fork
from . import spawn
from . import util


__all__ = ['Popen']

#
# Wrapper for an fd used while launching a process
#

class _DupFd(object):
    def __init__(self, ind):
        self.ind = ind
    def detach(self):
        return forkserver.get_inherited_fds()[self.ind]

#
# Start child process using a server process
#

class Popen(popen_fork.Popen):
    method = 'forkserver'
    DupFd = _DupFd

    def __init__(self, process_obj):
        self._fds = []
        super().__init__(process_obj)

    def duplicate_for_child(self, fd):
        self._fds.append(fd)
        return len(self._fds) - 1

    def _launch(self, process_obj):
        prep_data = spawn.get_preparation_data(process_obj._name)
        buf = io.BytesIO()
        set_spawning_popen(self)
        try:
            reduction.dump(prep_data, buf)
            reduction.dump(process_obj, buf)
        finally:
            set_spawning_popen(None)

        self.sentinel, w = forkserver.connect_to_new_process(self._fds)
        # Keep a duplicate of the data pipe's write end as a sentinel of the
        # parent process used by the child process.
        _parent_w = os.dup(w)
        self.finalizer = util.Finalize(self, util.close_fds,
                                       (_parent_w, self.sentinel))
        with open(w, 'wb', closefd=True) as f:
            f.write(buf.getbuffer())
        self.pid = forkserver.read_signed(self.sentinel)

    def poll(self, flag=os.WNOHANG):
        if self.returncode is None:
            from multiprocess.connection import wait
            timeout = 0 if flag == os.WNOHANG else None
            if not wait([self.sentinel], timeout):
                return None
            try:
                self.returncode = forkserver.read_signed(self.sentinel)
            except (OSError, EOFError):
                # This should not happen usually, but perhaps the forkserver
                # process itself got killed
                self.returncode = 255

        return self.returncode
<|endoftext|>#
# Module which supports allocation of memory from an mmap
#
# multiprocessing/heap.py
#
# Copyright (c) 2006-2008, R Oudkerk
# Licensed to PSF under a Contributor Agreement.
#

import bisect
from collections import defaultdict
import mmap
import os
import sys
import tempfile
import threading

from .context import reduction, assert_spawning
from . import util

__all__ = ['BufferWrapper']

#
# Inheritable class which wraps an mmap, and from which blocks can be allocated
#

if sys.platform == 'win32':

    import _winapi

    class Arena(object):
        """
        A shared memory area backed by anonymous memory (Windows).
        """

        _rand = tempfile._RandomNameSequence()

        def __init__(self, size):
            self.size = size
            for i in range(100):
                name = 'pym-%d-%s' % (os.getpid(), next(self._rand))
                buf = mmap.mmap(-1, size, tagname=name)
                if _winapi.GetLastError() == 0:
                    break
                # We have reopened a preexisting mmap.
                buf.close()
            else:
                raise FileExistsError('Cannot find name for new mmap')
            self.name = name
            self.buffer = buf
            self._state = (self.size, self.name)

        def __getstate__(self):
            assert_spawning(self)
            return self._state

        def __setstate__(self, state):
            self.size, self.name = self._state = state
            # Reopen existing mmap
            self.buffer = mmap.mmap(-1, self.size, tagname=self.name)
            # XXX Temporarily preventing buildbot failures while determining
            # XXX the correct long-term fix. See issue 23060
            #assert _winapi.GetLastError() == _winapi.ERROR_ALREADY_EXISTS

else:

    class Arena(object):
        """
        A shared memory area backed by a temporary file (POSIX).
        """

        if sys.platform == 'linux':
            _dir_candidates = ['/dev/shm']
        else:
            _dir_candidates = []

        def __init__(self, size, fd=-1):
            self.size = size
            self.fd = fd
            if fd == -1:
                # Arena is created anew (if fd != -1, it means we're coming
                # from rebuild_arena() below)
                self.fd, name = tempfile.mkstemp(
                     prefix='pym-%d-'%os.getpid(),
                     dir=self._choose_dir(size))
                os.unlink(name)
                util.Finalize(self, os.close, (self.fd,))
                os.ftruncate(self.fd, size)
            self.buffer = mmap.mmap(self.fd, self.size)

        def _choose_dir(self, size):
            # Choose a non-storage backed directory if possible,
            # to improve performance
            for d in self._dir_candidates:
                st = os.statvfs(d)
                if st.f_bavail * st.f_frsize >= size:  # enough free space?
                    return d
            return util.get_temp_dir()

    def reduce_arena(a):
        if a.fd == -1:
            raise ValueError('Arena is unpicklable because '
                             'forking was enabled when it was created')
        return rebuild_arena, (a.size, reduction.DupFd(a.fd))

    def rebuild_arena(size, dupfd):
        return Arena(size, dupfd.detach())

    reduction.register(Arena, reduce_arena)

#
# Class allowing allocation of chunks of memory from arenas
#

class Heap(object):

    # Minimum malloc() alignment
    _alignment = 8

    _DISCARD_FREE_SPACE_LARGER_THAN = 4 * 1024 ** 2  # 4 MB
    _DOUBLE_ARENA_SIZE_UNTIL = 4 * 1024 ** 2

    def __init__(self, size=mmap.PAGESIZE):
        self._lastpid = os.getpid()
        self._lock = threading.Lock()
        # Current arena allocation size
        self._size = size
        # A sorted list of available block sizes in arenas
        self._lengths = []

        # Free block management:
        # - map each block size to a list of `(Arena, start, stop)` blocks
        self._len_to_seq = {}
        # - map `(Arena, start)` tuple to the `(Arena, start, stop)` block
        #   starting at that offset
        self._start_to_block = {}
        # - map `(Arena, stop)` tuple to the `(Arena, start, stop)` block
        #   ending at that offset
        self._stop_to_block = {}

        # Map arenas to their `(Arena, start, stop)` blocks in use
        self._allocated_blocks = defaultdict(set)
        self._arenas = []

        # List of pending blocks to free - see comment in free() below
        self._pending_free_blocks = []

        # Statistics
        self._n_mallocs = 0
        self._n_frees = 0

    @staticmethod
    def _roundup(n, alignment):
        # alignment must be a power of 2
        mask = alignment - 1
        return (n + mask) & ~mask

    def _new_arena(self, size):
        # Create a new arena with at least the given *size*
        length = self._roundup(max(self._size, size), mmap.PAGESIZE)
        # We carve larger and larger arenas, for efficiency, until we
        # reach a large-ish size (roughly L3 cache-sized)
        if self._size < self._DOUBLE_ARENA_SIZE_UNTIL:
            self._size *= 2
        util.info('allocating a new mmap of length %d', length)
        arena = Arena(length)
        self._arenas.append(arena)
        return (arena, 0, length)

    def _discard_arena(self, arena):
        # Possibly delete the given (unused) arena
        length = arena.size
        # Reusing an existing arena is faster than creating a new one, so
        # we only reclaim space if it's large enough.
        if length < self._DISCARD_FREE_SPACE_LARGER_THAN:
            return
        blocks = self._allocated_blocks.pop(arena)
        assert not blocks
        del self._start_to_block[(arena, 0)]
        del self._stop_to_block[(arena, length)]
        self._arenas.remove(arena)
        seq = self._len_to_seq[length]
        seq.remove((arena, 0, length))
        if not seq:
            del self._len_to_seq[length]
            self._lengths.remove(length)

    def _malloc(self, size):
        # returns a large enough block -- it might be much larger
        i = bisect.bisect_left(self._lengths, size)
        if i == len(self._lengths):
            return self._new_arena(size)
        else:
            length = self._lengths[i]
            seq = self._len_to_seq[length]
            block = seq.pop()
            if not seq:
                del self._len_to_seq[length], self._lengths[i]

        (arena, start, stop) = block
        del self._start_to_block[(arena, start)]
        del self._stop_to_block[(arena, stop)]
        return block

    def _add_free_block(self, block):
        # make block available and try to merge with its neighbours in the arena
        (arena, start, stop) = block

        try:
            prev_block = self._stop_to_block[(arena, start)]
        except KeyError:
            pass
        else:
            start, _ = self._absorb(prev_block)

        try:
            next_block = self._start_to_block[(arena, stop)]
        except KeyError:
            pass
        else:
            _, stop = self._absorb(next_block)

        block = (arena, start, stop)
        length = stop - start

        try:
            self._len_to_seq[length].append(block)
        except KeyError:
            self._len_to_seq[length] = [block]
            bisect.insort(self._lengths, length)

        self._start_to_block[(arena, start)] = block
        self._stop_to_block[(arena, stop)] = block

    def _absorb(self, block):
        # deregister this block so it can be merged with a neighbour
        (arena, start, stop) = block
        del self._start_to_block[(arena, start)]
        del self._stop_to_block[(arena, stop)]

        length = stop - start
        seq = self._len_to_seq[length]
        seq.remove(block)
        if not seq:
            del self._len_to_seq[length]
            self._lengths.remove(length)

        return start, stop

    def _remove_allocated_block(self, block):
        arena, start, stop = block
        blocks = self._allocated_blocks[arena]
        blocks.remove((start, stop))
        if not blocks:
            # Arena is entirely free, discard it from this process
            self._discard_arena(arena)

