    >>> np.allclose(a, np.dot(q4, r4) @ P)
    True
    >>> np.allclose(a @ P.T, np.dot(q4, r4))
    True
    >>> q4.shape, r4.shape, p4.shape
    ((9, 9), (9, 6), (6,))

    >>> q5, r5, p5 = linalg.qr(a, mode='economic', pivoting=True)
    >>> q5.shape, r5.shape, p5.shape
    ((9, 6), (6, 6), (6,))
    >>> P = np.eye(6)[:, p5]
    >>> np.allclose(a @ P, np.dot(q5, r5))
    True

    """
    # 'qr' was the old default, equivalent to 'full'. Neither 'full' nor
    # 'qr' are used below.
    # 'raw' is used internally by qr_multiply
    if mode not in ['full', 'qr', 'r', 'economic', 'raw']:
        raise ValueError("Mode argument should be one of ['full', 'r', "
                         "'economic', 'raw']")

    if check_finite:
        a1 = np.asarray_chkfinite(a)
    else:
        a1 = np.asarray(a)
    if len(a1.shape) != 2:
        raise ValueError("expected a 2-D array")

    M, N = a1.shape

    # accommodate empty arrays
    if a1.size == 0:
        K = min(M, N)

        if mode not in ['economic', 'raw']:
            Q = np.empty_like(a1, shape=(M, M))
            Q[...] = np.identity(M)
            R = np.empty_like(a1)
        else:
            Q = np.empty_like(a1, shape=(M, K))
            R = np.empty_like(a1, shape=(K, N))

        if pivoting:
            Rj = R, np.arange(N, dtype=np.int32)
        else:
            Rj = R,

        if mode == 'r':
            return Rj
        elif mode == 'raw':
            qr = np.empty_like(a1, shape=(M, N))
            tau = np.zeros_like(a1, shape=(K,))
            return ((qr, tau),) + Rj
        return (Q,) + Rj

    overwrite_a = overwrite_a or (_datacopied(a1, a))

    if pivoting:
        geqp3, = get_lapack_funcs(('geqp3',), (a1,))
        qr, jpvt, tau = safecall(geqp3, "geqp3", a1, overwrite_a=overwrite_a)
        jpvt -= 1  # geqp3 returns a 1-based index array, so subtract 1
    else:
        geqrf, = get_lapack_funcs(('geqrf',), (a1,))
        qr, tau = safecall(geqrf, "geqrf", a1, lwork=lwork,
                           overwrite_a=overwrite_a)

    if mode not in ['economic', 'raw'] or M < N:
        R = np.triu(qr)
    else:
        R = np.triu(qr[:N, :])

    if pivoting:
        Rj = R, jpvt
    else:
        Rj = R,

    if mode == 'r':
        return Rj
    elif mode == 'raw':
        return ((qr, tau),) + Rj

    gor_un_gqr, = get_lapack_funcs(('orgqr',), (qr,))

    if M < N:
        Q, = safecall(gor_un_gqr, "gorgqr/gungqr", qr[:, :M], tau,
                      lwork=lwork, overwrite_a=1)
    elif mode == 'economic':
        Q, = safecall(gor_un_gqr, "gorgqr/gungqr", qr, tau, lwork=lwork,
                      overwrite_a=1)
    else:
        t = qr.dtype.char
        qqr = np.empty((M, M), dtype=t)
        qqr[:, :N] = qr
        Q, = safecall(gor_un_gqr, "gorgqr/gungqr", qqr, tau, lwork=lwork,
                      overwrite_a=1)

    return (Q,) + Rj


def qr_multiply(a, c, mode='right', pivoting=False, conjugate=False,
                overwrite_a=False, overwrite_c=False):
    """
    Calculate the QR decomposition and multiply Q with a matrix.

    Calculate the decomposition ``A = Q R`` where Q is unitary/orthogonal
    and R upper triangular. Multiply Q with a vector or a matrix c.

    Parameters
    ----------
    a : (M, N), array_like
        Input array
    c : array_like
        Input array to be multiplied by ``q``.
    mode : {'left', 'right'}, optional
        ``Q @ c`` is returned if mode is 'left', ``c @ Q`` is returned if
        mode is 'right'.
        The shape of c must be appropriate for the matrix multiplications,
        if mode is 'left', ``min(a.shape) == c.shape[0]``,
        if mode is 'right', ``a.shape[0] == c.shape[1]``.
    pivoting : bool, optional
        Whether or not factorization should include pivoting for rank-revealing
        qr decomposition, see the documentation of qr.
    conjugate : bool, optional
        Whether Q should be complex-conjugated. This might be faster
        than explicit conjugation.
    overwrite_a : bool, optional
        Whether data in a is overwritten (may improve performance)
    overwrite_c : bool, optional
        Whether data in c is overwritten (may improve performance).
        If this is used, c must be big enough to keep the result,
        i.e. ``c.shape[0]`` = ``a.shape[0]`` if mode is 'left'.

    Returns
    -------
    CQ : ndarray
        The product of ``Q`` and ``c``.
    R : (K, N), ndarray
        R array of the resulting QR factorization where ``K = min(M, N)``.
    P : (N,) ndarray
        Integer pivot array. Only returned when ``pivoting=True``.

    Raises
    ------
    LinAlgError
        Raised if QR decomposition fails.

    Notes
    -----
    This is an interface to the LAPACK routines ``?GEQRF``, ``?ORMQR``,
    ``?UNMQR``, and ``?GEQP3``.

    .. versionadded:: 0.11.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import qr_multiply, qr
    >>> A = np.array([[1, 3, 3], [2, 3, 2], [2, 3, 3], [1, 3, 2]])
    >>> qc, r1, piv1 = qr_multiply(A, 2*np.eye(4), pivoting=1)
    >>> qc
    array([[-1.,  1., -1.],
           [-1., -1.,  1.],
           [-1., -1., -1.],
           [-1.,  1.,  1.]])
    >>> r1
    array([[-6., -3., -5.            ],
           [ 0., -1., -1.11022302e-16],
           [ 0.,  0., -1.            ]])
    >>> piv1
    array([1, 0, 2], dtype=int32)
    >>> q2, r2, piv2 = qr(A, mode='economic', pivoting=1)
    >>> np.allclose(2*q2 - qc, np.zeros((4, 3)))
    True

    """
    if mode not in ['left', 'right']:
        raise ValueError("Mode argument can only be 'left' or 'right' but "
                         f"not '{mode}'")
    c = np.asarray_chkfinite(c)
    if c.ndim < 2:
        onedim = True
        c = np.atleast_2d(c)
        if mode == "left":
            c = c.T
    else:
        onedim = False

    a = np.atleast_2d(np.asarray(a))  # chkfinite done in qr
    M, N = a.shape

    if mode == 'left':
        if c.shape[0] != min(M, N + overwrite_c*(M-N)):
            raise ValueError('Array shapes are not compatible for Q @ c'
                             f' operation: {a.shape} vs {c.shape}')
    else:
        if M != c.shape[1]:
            raise ValueError('Array shapes are not compatible for c @ Q'
                             f' operation: {c.shape} vs {a.shape}')

    raw = qr(a, overwrite_a, None, "raw", pivoting)
    Q, tau = raw[0]

    # accommodate empty arrays
    if c.size == 0:
        return (np.empty_like(c),) + raw[1:]

    gor_un_mqr, = get_lapack_funcs(('ormqr',), (Q,))
    if gor_un_mqr.typecode in ('s', 'd'):
        trans = "T"
    else:
        trans = "C"

    Q = Q[:, :min(M, N)]
    if M > N and mode == "left" and not overwrite_c:
        if conjugate:
            cc = np.zeros((c.shape[1], M), dtype=c.dtype, order="F")
            cc[:, :N] = c.T
        else:
            cc = np.zeros((M, c.shape[1]), dtype=c.dtype, order="F")
            cc[:N, :] = c
            trans = "N"
        if conjugate:
            lr = "R"
        else:
            lr = "L"
        overwrite_c = True
    elif c.flags["C_CONTIGUOUS"] and trans == "T" or conjugate:
        cc = c.T
        if mode == "left":
            lr = "R"
        else:
            lr = "L"
    else:
        trans = "N"
        cc = c
        if mode == "left":
            lr = "L"
        else:
            lr = "R"
    cQ, = safecall(gor_un_mqr, "gormqr/gunmqr", lr, trans, Q, tau, cc,
                   overwrite_c=overwrite_c)
    if trans != "N":
        cQ = cQ.T
    if mode == "right":
        cQ = cQ[:, :min(M, N)]
    if onedim:
        cQ = cQ.ravel()

    return (cQ,) + raw[1:]


def rq(a, overwrite_a=False, lwork=None, mode='full', check_finite=True):
    """
    Compute RQ decomposition of a matrix.

    Calculate the decomposition ``A = R Q`` where Q is unitary/orthogonal
    and R upper triangular.

    Parameters
    ----------
    a : (M, N) array_like
        Matrix to be decomposed
    overwrite_a : bool, optional
        Whether data in a is overwritten (may improve performance)
    lwork : int, optional
        Work array size, lwork >= a.shape[1]. If None or -1, an optimal size
        is computed.
    mode : {'full', 'r', 'economic'}, optional
        Determines what information is to be returned: either both Q and R
        ('full', default), only R ('r') or both Q and R but computed in
        economy-size ('economic', see Notes).
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    R : float or complex ndarray
        Of shape (M, N) or (M, K) for ``mode='economic'``. ``K = min(M, N)``.
    Q : float or complex ndarray
        Of shape (N, N) or (K, N) for ``mode='economic'``. Not returned
        if ``mode='r'``.

    Raises
    ------
    LinAlgError
        If decomposition fails.

    Notes
    -----
    This is an interface to the LAPACK routines sgerqf, dgerqf, cgerqf, zgerqf,
    sorgrq, dorgrq, cungrq and zungrq.

    If ``mode=economic``, the shapes of Q and R are (K, N) and (M, K) instead
    of (N,N) and (M,N), with ``K=min(M,N)``.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import linalg
    >>> rng = np.random.default_rng()
    >>> a = rng.standard_normal((6, 9))
    >>> r, q = linalg.rq(a)
    >>> np.allclose(a, r @ q)
    True
    >>> r.shape, q.shape
    ((6, 9), (9, 9))
    >>> r2 = linalg.rq(a, mode='r')
    >>> np.allclose(r, r2)
    True
    >>> r3, q3 = linalg.rq(a, mode='economic')
    >>> r3.shape, q3.shape
    ((6, 6), (6, 9))

    """
    if mode not in ['full', 'r', 'economic']:
        raise ValueError(
                 "Mode argument should be one of ['full', 'r', 'economic']")

    if check_finite:
        a1 = np.asarray_chkfinite(a)
    else:
        a1 = np.asarray(a)
    if len(a1.shape) != 2:
        raise ValueError('expected matrix')

    M, N = a1.shape

    # accommodate empty arrays
    if a1.size == 0:
        K = min(M, N)

        if not mode == 'economic':
            R = np.empty_like(a1)
            Q = np.empty_like(a1, shape=(N, N))
            Q[...] = np.identity(N)
        else:
            R = np.empty_like(a1, shape=(M, K))
            Q = np.empty_like(a1, shape=(K, N))

        if mode == 'r':
            return R
        return R, Q

    overwrite_a = overwrite_a or (_datacopied(a1, a))

    gerqf, = get_lapack_funcs(('gerqf',), (a1,))
    rq, tau = safecall(gerqf, 'gerqf', a1, lwork=lwork,
                       overwrite_a=overwrite_a)
    if not mode == 'economic' or N < M:
        R = np.triu(rq, N-M)
    else:
        R = np.triu(rq[-M:, -M:])

    if mode == 'r':
        return R

    gor_un_grq, = get_lapack_funcs(('orgrq',), (rq,))

    if N < M:
        Q, = safecall(gor_un_grq, "gorgrq/gungrq", rq[-N:], tau, lwork=lwork,
                      overwrite_a=1)
    elif mode == 'economic':
        Q, = safecall(gor_un_grq, "gorgrq/gungrq", rq, tau, lwork=lwork,
                      overwrite_a=1)
    else:
        rq1 = np.empty((N, N), dtype=rq.dtype)
        rq1[-M:] = rq
        Q, = safecall(gor_un_grq, "gorgrq/gungrq", rq1, tau, lwork=lwork,
                      overwrite_a=1)

    return R, Q
<|endoftext|>"""Schur decomposition functions."""
import numpy as np
from numpy import asarray_chkfinite, single, asarray, array
from numpy.linalg import norm


# Local imports.
from ._misc import LinAlgError, _datacopied
from .lapack import get_lapack_funcs
from ._decomp import eigvals

__all__ = ['schur', 'rsf2csf']

_double_precision = ['i', 'l', 'd']


def schur(a, output='real', lwork=None, overwrite_a=False, sort=None,
          check_finite=True):
    """
    Compute Schur decomposition of a matrix.

    The Schur decomposition is::

        A = Z T Z^H

    where Z is unitary and T is either upper-triangular, or for real
    Schur decomposition (output='real'), quasi-upper triangular. In
    the quasi-triangular form, 2x2 blocks describing complex-valued
    eigenvalue pairs may extrude from the diagonal.

    Parameters
    ----------
    a : (M, M) array_like
        Matrix to decompose
    output : {'real', 'complex'}, optional
        When the dtype of `a` is real, this specifies whether to compute
        the real or complex Schur decomposition.
        When the dtype of `a` is complex, this argument is ignored, and the
        complex Schur decomposition is computed.
    lwork : int, optional
        Work array size. If None or -1, it is automatically computed.
    overwrite_a : bool, optional
        Whether to overwrite data in a (may improve performance).
    sort : {None, callable, 'lhp', 'rhp', 'iuc', 'ouc'}, optional
        Specifies whether the upper eigenvalues should be sorted. A callable
        may be passed that, given an eigenvalue, returns a boolean denoting
        whether the eigenvalue should be sorted to the top-left (True).

        - If ``output='complex'`` OR the dtype of `a` is complex, the callable
          should have one argument: the eigenvalue expressed as a complex number.
        - If ``output='real'`` AND the dtype of `a` is real, the callable should have
          two arguments: the real and imaginary parts of the eigenvalue, respectively.

        Alternatively, string parameters may be used::

            'lhp'   Left-hand plane (real(eigenvalue) < 0.0)
            'rhp'   Right-hand plane (real(eigenvalue) >= 0.0)
            'iuc'   Inside the unit circle (abs(eigenvalue) <= 1.0)
            'ouc'   Outside the unit circle (abs(eigenvalue) > 1.0)

        Defaults to None (no sorting).
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    T : (M, M) ndarray
        Schur form of A. It is real-valued for the real Schur decomposition.
    Z : (M, M) ndarray
        An unitary Schur transformation matrix for A.
        It is real-valued for the real Schur decomposition.
    sdim : int
        If and only if sorting was requested, a third return value will
        contain the number of eigenvalues satisfying the sort condition.
        Note that complex conjugate pairs for which the condition is true
        for either eigenvalue count as 2.

    Raises
    ------
    LinAlgError
        Error raised under three conditions:

        1. The algorithm failed due to a failure of the QR algorithm to
           compute all eigenvalues.
        2. If eigenvalue sorting was requested, the eigenvalues could not be
           reordered due to a failure to separate eigenvalues, usually because
           of poor conditioning.
        3. If eigenvalue sorting was requested, roundoff errors caused the
           leading eigenvalues to no longer satisfy the sorting condition.

    See Also
    --------
    rsf2csf : Convert real Schur form to complex Schur form

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import schur, eigvals
    >>> A = np.array([[0, 2, 2], [0, 1, 2], [1, 0, 1]])
    >>> T, Z = schur(A)
    >>> T
    array([[ 2.65896708,  1.42440458, -1.92933439],
           [ 0.        , -0.32948354, -0.49063704],
           [ 0.        ,  1.31178921, -0.32948354]])
    >>> Z
    array([[0.72711591, -0.60156188, 0.33079564],
           [0.52839428, 0.79801892, 0.28976765],
           [0.43829436, 0.03590414, -0.89811411]])

    >>> T2, Z2 = schur(A, output='complex')
    >>> T2
    array([[ 2.65896708, -1.22839825+1.32378589j,  0.42590089+1.51937378j], # may vary
           [ 0.        , -0.32948354+0.80225456j, -0.59877807+0.56192146j],
           [ 0.        ,  0.                    , -0.32948354-0.80225456j]])
    >>> eigvals(T2)
    array([2.65896708, -0.32948354+0.80225456j, -0.32948354-0.80225456j])   # may vary

    A custom eigenvalue-sorting condition that sorts by positive imaginary part
    is satisfied by only one eigenvalue.

    >>> _, _, sdim = schur(A, output='complex', sort=lambda x: x.imag > 1e-15)
    >>> sdim
    1

    When ``output='real'`` and the array `a` is real, the `sort` callable must accept
    the real and imaginary parts as separate arguments. Note that now the complex
    eigenvalues ``-0.32948354+0.80225456j`` and ``-0.32948354-0.80225456j`` will be
    treated as a complex conjugate pair, and according to the `sdim` documentation,
    complex conjugate pairs for which the condition is True for *either* eigenvalue
    increase `sdim` by *two*.

    >>> _, _, sdim = schur(A, output='real', sort=lambda x, y: y > 1e-15)
    >>> sdim
    2

    """
    if output not in ['real', 'complex', 'r', 'c']:
        raise ValueError("argument must be 'real', or 'complex'")
    if check_finite:
        a1 = asarray_chkfinite(a)
    else:
        a1 = asarray(a)
    if np.issubdtype(a1.dtype, np.integer):
        a1 = asarray(a, dtype=np.dtype("long"))
    if len(a1.shape) != 2 or (a1.shape[0] != a1.shape[1]):
        raise ValueError('expected square matrix')

    typ = a1.dtype.char
    if output in ['complex', 'c'] and typ not in ['F', 'D']:
        if typ in _double_precision:
            a1 = a1.astype('D')
        else:
            a1 = a1.astype('F')

    # accommodate empty matrix
    if a1.size == 0:
        t0, z0 = schur(np.eye(2, dtype=a1.dtype))
        if sort is None:
            return (np.empty_like(a1, dtype=t0.dtype),
                    np.empty_like(a1, dtype=z0.dtype))
        else:
            return (np.empty_like(a1, dtype=t0.dtype),
                    np.empty_like(a1, dtype=z0.dtype), 0)

    overwrite_a = overwrite_a or (_datacopied(a1, a))
    gees, = get_lapack_funcs(('gees',), (a1,))
    if lwork is None or lwork == -1:
        # get optimal work array
        result = gees(lambda x: None, a1, lwork=-1)
        lwork = result[-2][0].real.astype(np.int_)

    if sort is None:
        sort_t = 0
        def sfunction(x, y=None):
            return None
    else:
        sort_t = 1
        if callable(sort):
            sfunction = sort
        elif sort == 'lhp':
            def sfunction(x, y=None):
                return x.real < 0.0
        elif sort == 'rhp':
            def sfunction(x, y=None):
                return x.real >= 0.0
        elif sort == 'iuc':
            def sfunction(x, y=None):
                z = x if y is None else x + y*1j
                return abs(z) <= 1.0
        elif sort == 'ouc':
            def sfunction(x, y=None):
                z = x if y is None else x + y*1j
                return abs(z) > 1.0
        else:
            raise ValueError("'sort' parameter must either be 'None', or a "
                             "callable, or one of ('lhp','rhp','iuc','ouc')")

    result = gees(sfunction, a1, lwork=lwork, overwrite_a=overwrite_a,
                  sort_t=sort_t)

    info = result[-1]
    if info < 0:
        raise ValueError(f'illegal value in {-info}-th argument of internal gees')
    elif info == a1.shape[0] + 1:
        raise LinAlgError('Eigenvalues could not be separated for reordering.')
    elif info == a1.shape[0] + 2:
        raise LinAlgError('Leading eigenvalues do not satisfy sort condition.')
    elif info > 0:
        raise LinAlgError("Schur form not found. Possibly ill-conditioned.")

    if sort is None:
        return result[0], result[-3]
    else:
        return result[0], result[-3], result[1]


eps = np.finfo(float).eps
feps = np.finfo(single).eps

_array_kind = {'b': 0, 'h': 0, 'B': 0, 'i': 0, 'l': 0,
               'f': 0, 'd': 0, 'F': 1, 'D': 1}
_array_precision = {'i': 1, 'l': 1, 'f': 0, 'd': 1, 'F': 0, 'D': 1}
_array_type = [['f', 'd'], ['F', 'D']]


def _commonType(*arrays):
    kind = 0
    precision = 0
    for a in arrays:
        t = a.dtype.char
        kind = max(kind, _array_kind[t])
        precision = max(precision, _array_precision[t])
    return _array_type[kind][precision]


def _castCopy(type, *arrays):
    cast_arrays = ()
    for a in arrays:
        if a.dtype.char == type:
            cast_arrays = cast_arrays + (a.copy(),)
        else:
            cast_arrays = cast_arrays + (a.astype(type),)
    if len(cast_arrays) == 1:
        return cast_arrays[0]
    else:
        return cast_arrays


def rsf2csf(T, Z, check_finite=True):
    """
    Convert real Schur form to complex Schur form.

    Convert a quasi-diagonal real-valued Schur form to the upper-triangular
    complex-valued Schur form.

    Parameters
    ----------
    T : (M, M) array_like
        Real Schur form of the original array
    Z : (M, M) array_like
        Schur transformation matrix
    check_finite : bool, optional
        Whether to check that the input arrays contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    T : (M, M) ndarray
        Complex Schur form of the original array
    Z : (M, M) ndarray
        Schur transformation matrix corresponding to the complex form

    See Also
    --------
    schur : Schur decomposition of an array

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import schur, rsf2csf
    >>> A = np.array([[0, 2, 2], [0, 1, 2], [1, 0, 1]])
    >>> T, Z = schur(A)
    >>> T
    array([[ 2.65896708,  1.42440458, -1.92933439],
           [ 0.        , -0.32948354, -0.49063704],
           [ 0.        ,  1.31178921, -0.32948354]])
    >>> Z
    array([[0.72711591, -0.60156188, 0.33079564],
           [0.52839428, 0.79801892, 0.28976765],
           [0.43829436, 0.03590414, -0.89811411]])
    >>> T2 , Z2 = rsf2csf(T, Z)
    >>> T2
    array([[2.65896708+0.j, -1.64592781+0.743164187j, -1.21516887+1.00660462j],
           [0.+0.j , -0.32948354+8.02254558e-01j, -0.82115218-2.77555756e-17j],
           [0.+0.j , 0.+0.j, -0.32948354-0.802254558j]])
    >>> Z2
    array([[0.72711591+0.j,  0.28220393-0.31385693j,  0.51319638-0.17258824j],
           [0.52839428+0.j,  0.24720268+0.41635578j, -0.68079517-0.15118243j],
           [0.43829436+0.j, -0.76618703+0.01873251j, -0.03063006+0.46857912j]])

    """
    if check_finite:
        Z, T = map(asarray_chkfinite, (Z, T))
    else:
        Z, T = map(asarray, (Z, T))

    for ind, X in enumerate([Z, T]):
        if X.ndim != 2 or X.shape[0] != X.shape[1]:
            raise ValueError(f"Input '{'ZT'[ind]}' must be square.")

    if T.shape[0] != Z.shape[0]:
        message = f"Input array shapes must match: Z: {Z.shape} vs. T: {T.shape}"
        raise ValueError(message)
    N = T.shape[0]
    t = _commonType(Z, T, array([3.0], 'F'))
    Z, T = _castCopy(t, Z, T)

    for m in range(N-1, 0, -1):
        if abs(T[m, m-1]) > eps*(abs(T[m-1, m-1]) + abs(T[m, m])):
            mu = eigvals(T[m-1:m+1, m-1:m+1]) - T[m, m]
            r = norm([mu[0], T[m, m-1]])
            c = mu[0] / r
            s = T[m, m-1] / r
            G = array([[c.conj(), s], [-s, c]], dtype=t)

            T[m-1:m+1, m-1:] = G.dot(T[m-1:m+1, m-1:])
            T[:m+1, m-1:m+1] = T[:m+1, m-1:m+1].dot(G.conj().T)
            Z[:, m-1:m+1] = Z[:, m-1:m+1].dot(G.conj().T)

        T[m, m-1] = 0.0
    return T, Z
<|endoftext|>"""
Low-level LAPACK functions (:mod:`scipy.linalg.lapack`)
=======================================================

This module contains low-level functions from the LAPACK library.

.. versionadded:: 0.12.0

.. note::

    The common ``overwrite_<>`` option in many routines, allows the
    input arrays to be overwritten to avoid extra memory allocation.
    However this requires the array to satisfy two conditions
    which are memory order and the data type to match exactly the
    order and the type expected by the routine.

    As an example, if you pass a double precision float array to any
    ``S....`` routine which expects single precision arguments, f2py
    will create an intermediate array to match the argument types and
    overwriting will be performed on that intermediate array.

    Similarly, if a C-contiguous array is passed, f2py will pass a
    FORTRAN-contiguous array internally. Please make sure that these
    details are satisfied. More information can be found in the f2py
    documentation.

.. warning::

   These functions do little to no error checking.
   It is possible to cause crashes by mis-using them,
   so prefer using the higher-level routines in `scipy.linalg`.

Finding functions
-----------------

.. autosummary::
   :toctree: generated/

   get_lapack_funcs

All functions
-------------

.. autosummary::
   :toctree: generated/

   sgbsv
   dgbsv
   cgbsv
   zgbsv

   sgbtrf
   dgbtrf
   cgbtrf
   zgbtrf

   sgbtrs
   dgbtrs
   cgbtrs
   zgbtrs

   sgebal
   dgebal
   cgebal
   zgebal

   sgecon
   dgecon
   cgecon
   zgecon

   sgeequ
   dgeequ
   cgeequ
   zgeequ

   sgeequb
   dgeequb
   cgeequb
   zgeequb

   sgees
   dgees
   cgees
   zgees

   sgeev
   dgeev
   cgeev
   zgeev

   sgeev_lwork
   dgeev_lwork
   cgeev_lwork
   zgeev_lwork

   sgehrd
   dgehrd
   cgehrd
   zgehrd

   sgehrd_lwork
   dgehrd_lwork
   cgehrd_lwork
   zgehrd_lwork

   sgejsv
   dgejsv

   sgels
   dgels
   cgels
   zgels

   sgels_lwork
   dgels_lwork
   cgels_lwork
   zgels_lwork

   sgelsd
   dgelsd
   cgelsd
   zgelsd

   sgelsd_lwork
   dgelsd_lwork
   cgelsd_lwork
   zgelsd_lwork

   sgelss
   dgelss
   cgelss
   zgelss

   sgelss_lwork
   dgelss_lwork
   cgelss_lwork
   zgelss_lwork

   sgelsy
   dgelsy
   cgelsy
   zgelsy

   sgelsy_lwork
   dgelsy_lwork
   cgelsy_lwork
   zgelsy_lwork

   sgeqp3
   dgeqp3
   cgeqp3
   zgeqp3

   sgeqrf
   dgeqrf
   cgeqrf
   zgeqrf

   sgeqrf_lwork
   dgeqrf_lwork
   cgeqrf_lwork
   zgeqrf_lwork

   sgeqrfp
   dgeqrfp
   cgeqrfp
   zgeqrfp

   sgeqrfp_lwork
   dgeqrfp_lwork
   cgeqrfp_lwork
   zgeqrfp_lwork

   sgerqf
   dgerqf
   cgerqf
   zgerqf

   sgesdd
   dgesdd
   cgesdd
   zgesdd

   sgesdd_lwork
   dgesdd_lwork
   cgesdd_lwork
   zgesdd_lwork

   sgesv
   dgesv
   cgesv
   zgesv

   sgesvd
   dgesvd
   cgesvd
   zgesvd

   sgesvd_lwork
   dgesvd_lwork
   cgesvd_lwork
   zgesvd_lwork

   sgesvx
   dgesvx
   cgesvx
   zgesvx

   sgetrf
   dgetrf
   cgetrf
   zgetrf

   sgetc2
   dgetc2
   cgetc2
   zgetc2

   sgetri
   dgetri
   cgetri
   zgetri

   sgetri_lwork
   dgetri_lwork
   cgetri_lwork
   zgetri_lwork

   sgetrs
   dgetrs
   cgetrs
   zgetrs

   sgesc2
   dgesc2
   cgesc2
   zgesc2

   sgges
   dgges
   cgges
   zgges

   sggev
   dggev
   cggev
   zggev

   sgglse
   dgglse
   cgglse
   zgglse

   sgglse_lwork
   dgglse_lwork
   cgglse_lwork
   zgglse_lwork

   sgtsv
   dgtsv
   cgtsv
   zgtsv

   sgtsvx
   dgtsvx
   cgtsvx
   zgtsvx

   chbevd
   zhbevd

   chbevx
   zhbevx

   checon
   zhecon

   cheequb
   zheequb

   cheev
   zheev

   cheev_lwork
   zheev_lwork

   cheevd
   zheevd

   cheevd_lwork
   zheevd_lwork

   cheevr
   zheevr

   cheevr_lwork
   zheevr_lwork

   cheevx
   zheevx

   cheevx_lwork
   zheevx_lwork

   chegst
   zhegst

   chegv
   zhegv

   chegv_lwork
   zhegv_lwork

   chegvd
   zhegvd

   chegvx
   zhegvx

   chegvx_lwork
   zhegvx_lwork

   chesv
   zhesv

   chesv_lwork
   zhesv_lwork

   chesvx
   zhesvx

   chesvx_lwork
   zhesvx_lwork

   chetrd
   zhetrd

   chetrd_lwork
   zhetrd_lwork

   chetrf
   zhetrf

   chetrf_lwork
   zhetrf_lwork

   chetrs
   zhetrs

   chfrk
   zhfrk

   slamch
   dlamch

   slange
   dlange
   clange
   zlange

   slantr
   dlantr
   clantr
   zlantr

   slarf
   dlarf
   clarf
   zlarf

   slarfg
   dlarfg
   clarfg
   zlarfg

   slartg
   dlartg
   clartg
   zlartg

   slasd4
   dlasd4

   slaswp
   dlaswp
   claswp
   zlaswp

   slauum
   dlauum
   clauum
   zlauum

   sorcsd
   dorcsd
   sorcsd_lwork
   dorcsd_lwork

   sorghr
   dorghr
   sorghr_lwork
   dorghr_lwork

   sorgqr
   dorgqr

   sorgrq
   dorgrq

   sormqr
   dormqr

   sormrz
   dormrz

   sormrz_lwork
   dormrz_lwork

   spbsv
   dpbsv
   cpbsv
   zpbsv

   spbtrf
   dpbtrf
   cpbtrf
   zpbtrf

   spbtrs
   dpbtrs
   cpbtrs
   zpbtrs

   spftrf
   dpftrf
   cpftrf
   zpftrf

   spftri
   dpftri
   cpftri
   zpftri

   spftrs
   dpftrs
   cpftrs
   zpftrs

   spocon
   dpocon
   cpocon
   zpocon

   spstrf
   dpstrf
   cpstrf
   zpstrf

   spstf2
   dpstf2
   cpstf2
   zpstf2

   sposv
   dposv
   cposv
   zposv

   sposvx
   dposvx
   cposvx
   zposvx

   spotrf
   dpotrf
   cpotrf
   zpotrf

   spotri
   dpotri
   cpotri
   zpotri

   spotrs
   dpotrs
   cpotrs
   zpotrs

   sppcon
   dppcon
   cppcon
   zppcon

   sppsv
   dppsv
   cppsv
   zppsv

   spptrf
   dpptrf
   cpptrf
   zpptrf

   spptri
   dpptri
   cpptri
   zpptri

   spptrs
   dpptrs
   cpptrs
   zpptrs

   sptsv
   dptsv
   cptsv
   zptsv

   sptsvx
   dptsvx
   cptsvx
   zptsvx

   spttrf
   dpttrf
   cpttrf
   zpttrf

   spttrs
   dpttrs
   cpttrs
   zpttrs

   spteqr
   dpteqr
   cpteqr
   zpteqr

   crot
   zrot

   ssbev
   dsbev

   ssbevd
   dsbevd

   ssbevx
   dsbevx

   ssfrk
   dsfrk

   sstebz
   dstebz

   sstein
   dstein

   sstemr
   dstemr

   sstemr_lwork
   dstemr_lwork

   ssterf
   dsterf

   sstev
   dstev

   ssycon
   dsycon
   csycon
   zsycon

   ssyconv
   dsyconv
   csyconv
   zsyconv

   ssyequb
   dsyequb
   csyequb
   zsyequb

   ssyev
   dsyev

   ssyev_lwork
   dsyev_lwork

   ssyevd
   dsyevd

   ssyevd_lwork
   dsyevd_lwork

   ssyevr
   dsyevr

   ssyevr_lwork
   dsyevr_lwork

   ssyevx
   dsyevx

   ssyevx_lwork
   dsyevx_lwork

   ssygst
   dsygst

   ssygv
   dsygv

   ssygv_lwork
   dsygv_lwork

   ssygvd
   dsygvd

   ssygvx
   dsygvx

   ssygvx_lwork
   dsygvx_lwork

   ssysv
   dsysv
   csysv
   zsysv

   ssysv_lwork
   dsysv_lwork
   csysv_lwork
   zsysv_lwork

   ssysvx
   dsysvx
   csysvx
   zsysvx

   ssysvx_lwork
   dsysvx_lwork
   csysvx_lwork
   zsysvx_lwork

   ssytf2
   dsytf2
   csytf2
   zsytf2

   ssytrd
   dsytrd

   ssytrd_lwork
   dsytrd_lwork

   ssytrf
   dsytrf
   csytrf
   zsytrf

   ssytrf_lwork
   dsytrf_lwork
   csytrf_lwork
   zsytrf_lwork

   ssytrs
   dsytrs
   csytrs
   zsytrs

   stbtrs
   dtbtrs
   ctbtrs
   ztbtrs

   stfsm
   dtfsm
   ctfsm
   ztfsm

   stfttp
   dtfttp
   ctfttp
   ztfttp

   stfttr
   dtfttr
   ctfttr
   ztfttr

   stgexc
   dtgexc
   ctgexc
   ztgexc

   stgsen
   dtgsen
   ctgsen
   ztgsen

   stgsen_lwork
   dtgsen_lwork
   ctgsen_lwork
   ztgsen_lwork

   stgsyl
   dtgsyl

   stpttf
   dtpttf
   ctpttf
   ztpttf

   stpttr
   dtpttr
   ctpttr
   ztpttr

   strcon
   dtrcon
   ctrcon
   ztrcon

   strexc
   dtrexc
   ctrexc
   ztrexc

   strsen
   dtrsen
   ctrsen
   ztrsen

   strsen_lwork
   dtrsen_lwork
   ctrsen_lwork
   ztrsen_lwork

   strsyl
   dtrsyl
   ctrsyl
   ztrsyl

   strtri
   dtrtri
   ctrtri
   ztrtri

   strtrs
   dtrtrs
   ctrtrs
   ztrtrs

   strttf
   dtrttf
   ctrttf
   ztrttf

   strttp
   dtrttp
   ctrttp
   ztrttp

   stzrzf
   dtzrzf
   ctzrzf
   ztzrzf

   stzrzf_lwork
   dtzrzf_lwork
   ctzrzf_lwork
   ztzrzf_lwork

   cunghr
   zunghr

   cunghr_lwork
   zunghr_lwork

   cungqr
   zungqr

   cungrq
   zungrq

   cunmqr
   zunmqr

   sgeqrt
   dgeqrt
   cgeqrt
   zgeqrt

   sgemqrt
   dgemqrt
   cgemqrt
   zgemqrt

   sgttrf
   dgttrf
   cgttrf
   zgttrf

   sgttrs
   dgttrs
   cgttrs
   zgttrs

   sgtcon
   dgtcon
   cgtcon
   zgtcon

   stpqrt
   dtpqrt
   ctpqrt
   ztpqrt

   stpmqrt
   dtpmqrt
   ctpmqrt
   ztpmqrt

   cuncsd
   zuncsd

   cuncsd_lwork
   zuncsd_lwork

   cunmrz
   zunmrz

   cunmrz_lwork
   zunmrz_lwork

   ilaver

"""
#
# Author: Pearu Peterson, March 2002
#

import numpy as np
from .blas import _get_funcs, _memoize_get_funcs
from scipy.linalg import _flapack
from re import compile as regex_compile
try:
    from scipy.linalg import _clapack
except ImportError:
    _clapack = None

try:
    from scipy.linalg import _flapack_64
    HAS_ILP64 = True
except ImportError:
    HAS_ILP64 = False
    _flapack_64 = None


# Expose all functions (only flapack --- clapack is an implementation detail)
empty_module = None
from scipy.linalg._flapack import *  # noqa: E402, F403
del empty_module

__all__ = ['get_lapack_funcs']

# some convenience alias for complex functions
_lapack_alias = {
    'corghr': 'cunghr', 'zorghr': 'zunghr',
    'corghr_lwork': 'cunghr_lwork', 'zorghr_lwork': 'zunghr_lwork',
    'corgqr': 'cungqr', 'zorgqr': 'zungqr',
    'cormqr': 'cunmqr', 'zormqr': 'zunmqr',
    'corgrq': 'cungrq', 'zorgrq': 'zungrq',
}


# Place guards against docstring rendering issues with special characters
p1 = regex_compile(r'with bounds (?P<b>.*?)( and (?P<s>.*?) storage){0,1}\n')
p2 = regex_compile(r'Default: (?P<d>.*?)\n')


def backtickrepl(m):
    if m.group('s'):
        return (f"with bounds ``{m.group('b')}`` with ``{m.group('s')}`` storage\n")
    else:
        return f"with bounds ``{m.group('b')}``\n"


for routine in [ssyevr, dsyevr, cheevr, zheevr,
                ssyevx, dsyevx, cheevx, zheevx,
                ssygvd, dsygvd, chegvd, zhegvd]:
    if routine.__doc__:
        routine.__doc__ = p1.sub(backtickrepl, routine.__doc__)
        routine.__doc__ = p2.sub('Default ``\\1``\n', routine.__doc__)
    else:
        continue

del regex_compile, p1, p2, backtickrepl


@_memoize_get_funcs
def get_lapack_funcs(names, arrays=(), dtype=None, ilp64=False):
    """Return available LAPACK function objects from names.

    Arrays are used to determine the optimal prefix of LAPACK routines.

    Parameters
    ----------
    names : str or sequence of str
        Name(s) of LAPACK functions without type prefix.

    arrays : sequence of ndarrays, optional
        Arrays can be given to determine optimal prefix of LAPACK
        routines. If not given, double-precision routines will be
        used, otherwise the most generic type in arrays will be used.

    dtype : str or dtype, optional
        Data-type specifier. Not used if `arrays` is non-empty.

    ilp64 : {True, False, 'preferred'}, optional
        Whether to return ILP64 routine variant.
        Choosing 'preferred' returns ILP64 routine if available, and
        otherwise the 32-bit routine. Default: False

    Returns
    -------
    funcs : list
        List containing the found function(s).

    Notes
    -----
    This routine automatically chooses between Fortran/C
    interfaces. Fortran code is used whenever possible for arrays with
    column major order. In all other cases, C code is preferred.

    In LAPACK, the naming convention is that all functions start with a
    type prefix, which depends on the type of the principal
    matrix. These can be one of {'s', 'd', 'c', 'z'} for the NumPy
    types {float32, float64, complex64, complex128} respectively, and
    are stored in attribute ``typecode`` of the returned functions.

    Examples
    --------
    Suppose we would like to use '?lange' routine which computes the selected
    norm of an array. We pass our array in order to get the correct 'lange'
    flavor.

    >>> import numpy as np
    >>> import scipy.linalg as LA
    >>> rng = np.random.default_rng()

    >>> a = rng.random((3,2))
    >>> x_lange = LA.get_lapack_funcs('lange', (a,))
    >>> x_lange.typecode
    'd'
    >>> x_lange = LA.get_lapack_funcs('lange',(a*1j,))
    >>> x_lange.typecode
    'z'

    Several LAPACK routines work best when its internal WORK array has
    the optimal size (big enough for fast computation and small enough to
    avoid waste of memory). This size is determined also by a dedicated query
    to the function which is often wrapped as a standalone function and
    commonly denoted as ``###_lwork``. Below is an example for ``?sysv``

    >>> a = rng.random((1000, 1000))
    >>> b = rng.random((1000, 1)) * 1j
    >>> # We pick up zsysv and zsysv_lwork due to b array
    ... xsysv, xlwork = LA.get_lapack_funcs(('sysv', 'sysv_lwork'), (a, b))
    >>> opt_lwork, _ = xlwork(a.shape[0])  # returns a complex for 'z' prefix
    >>> udut, ipiv, x, info = xsysv(a, b, lwork=int(opt_lwork.real))

    """
    if isinstance(ilp64, str):
        if ilp64 == 'preferred':
            ilp64 = HAS_ILP64
        else:
            raise ValueError("Invalid value for 'ilp64'")

    if not ilp64:
        return _get_funcs(names, arrays, dtype,
                          "LAPACK", _flapack, _clapack,
                          "flapack", "clapack", _lapack_alias,
                          ilp64=False)
    else:
        if not HAS_ILP64:
            raise RuntimeError("LAPACK ILP64 routine requested, but Scipy "
                               "compiled only with 32-bit BLAS")
        return _get_funcs(names, arrays, dtype,
                          "LAPACK", _flapack_64, None,
                          "flapack_64", None, _lapack_alias,
                          ilp64=True)


_int32_max = np.iinfo(np.int32).max
_int64_max = np.iinfo(np.int64).max


def _compute_lwork(routine, *args, **kwargs):
    """
    Round floating-point lwork returned by lapack to integer.

    Several LAPACK routines compute optimal values for LWORK, which
    they return in a floating-point variable. However, for large
    values of LWORK, single-precision floating point is not sufficient
    to hold the exact value --- some LAPACK versions (<= 3.5.0 at
    least) truncate the returned integer to single precision and in
    some cases this can be smaller than the required value.

    Examples
    --------
    >>> from scipy.linalg import lapack
    >>> n = 5000
    >>> s_r, s_lw = lapack.get_lapack_funcs(('sysvx', 'sysvx_lwork'))
    >>> lwork = lapack._compute_lwork(s_lw, n)
    >>> lwork
    32000

    """
    dtype = getattr(routine, 'dtype', None)
    int_dtype = getattr(routine, 'int_dtype', None)
    ret = routine(*args, **kwargs)
    if ret[-1] != 0:
        raise ValueError("Internal work array size computation failed: "
                         "%d" % (ret[-1],))

    if len(ret) == 2:
        return _check_work_float(ret[0].real, dtype, int_dtype)
    else:
        return tuple(_check_work_float(x.real, dtype, int_dtype)
                     for x in ret[:-1])


def _check_work_float(value, dtype, int_dtype):
    """
    Convert LAPACK-returned work array size float to integer,
    carefully for single-precision types.
    """

    if dtype == np.float32 or dtype == np.complex64:
        # Single-precision routine -- take next fp value to work
        # around possible truncation in LAPACK code
        value = np.nextafter(value, np.inf, dtype=np.float32)

    value = int(value)
    if int_dtype.itemsize == 4:
        if value < 0 or value > _int32_max:
            raise ValueError("Too large work array required -- computation "
                             "cannot be performed with standard 32-bit"
                             " LAPACK.")
    elif int_dtype.itemsize == 8:
        if value < 0 or value > _int64_max:
            raise ValueError("Too large work array required -- computation"
                             " cannot be performed with standard 64-bit"
                             " LAPACK.")
    return value
<|endoftext|>import numpy as np


class _FakeMatrix:
    def __init__(self, data):
        self._data = data
        self.__array_interface__ = data.__array_interface__


class _FakeMatrix2:
    def __init__(self, data):
        self._data = data

    def __array__(self, dtype=None, copy=None):
        if copy:
            return self._data.copy()
        return self._data


def _get_array(shape, dtype):
    """
    Get a test array of given shape and data type.
    Returned NxN matrices are posdef, and 2xN are banded-posdef.

    """
    if len(shape) == 2 and shape[0] == 2:
        # yield a banded positive definite one
        x = np.zeros(shape, dtype=dtype)
        x[0, 1:] = -1
        x[1] = 2
        return x
    elif len(shape) == 2 and shape[0] == shape[1]:
        # always yield a positive definite matrix
        x = np.zeros(shape, dtype=dtype)
        j = np.arange(shape[0])
        x[j, j] = 2
        x[j[:-1], j[:-1]+1] = -1
        x[j[:-1]+1, j[:-1]] = -1
        return x
    else:
        np.random.seed(1234)
        return np.random.randn(*shape).astype(dtype)


def _id(x):
    return x


def assert_no_overwrite(call, shapes, dtypes=None):
    """
    Test that a call does not overwrite its input arguments
    """

    if dtypes is None:
        dtypes = [np.float32, np.float64, np.complex64, np.complex128]

    for dtype in dtypes:
        for order in ["C", "F"]:
            for faker in [_id, _FakeMatrix, _FakeMatrix2]:
                orig_inputs = [_get_array(s, dtype) for s in shapes]
                inputs = [faker(x.copy(order)) for x in orig_inputs]
                call(*inputs)
                msg = f"call modified inputs [{dtype!r}, {faker!r}]"
                for a, b in zip(inputs, orig_inputs):
                    np.testing.assert_equal(a, b, err_msg=msg)
<|endoftext|>#  ******************************************************************************
#   Copyright (C) 2013 Kenneth L. Ho
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#
#   Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer. Redistributions in binary
#   form must reproduce the above copyright notice, this list of conditions and
#   the following disclaimer in the documentation and/or other materials
#   provided with the distribution.
#
#   None of the names of the copyright holders may be used to endorse or
#   promote products derived from this software without specific prior written
#   permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#   POSSIBILITY OF SUCH DAMAGE.
#  ******************************************************************************

r"""
======================================================================
Interpolative matrix decomposition (:mod:`scipy.linalg.interpolative`)
======================================================================

.. versionadded:: 0.13

.. versionchanged:: 1.15.0
    The underlying algorithms have been ported to Python from the original Fortran77
    code. See references below for more details.

.. currentmodule:: scipy.linalg.interpolative

An interpolative decomposition (ID) of a matrix :math:`A \in
\mathbb{C}^{m \times n}` of rank :math:`k \leq \min \{ m, n \}` is a
factorization

.. math::
  A \Pi =
  \begin{bmatrix}
   A \Pi_{1} & A \Pi_{2}
  \end{bmatrix} =
  A \Pi_{1}
  \begin{bmatrix}
   I & T
  \end{bmatrix},

where :math:`\Pi = [\Pi_{1}, \Pi_{2}]` is a permutation matrix with
:math:`\Pi_{1} \in \{ 0, 1 \}^{n \times k}`, i.e., :math:`A \Pi_{2} =
A \Pi_{1} T`. This can equivalently be written as :math:`A = BP`,
where :math:`B = A \Pi_{1}` and :math:`P = [I, T] \Pi^{\mathsf{T}}`
are the *skeleton* and *interpolation matrices*, respectively.

If :math:`A` does not have exact rank :math:`k`, then there exists an
approximation in the form of an ID such that :math:`A = BP + E`, where
:math:`\| E \| \sim \sigma_{k + 1}` is on the order of the :math:`(k +
1)`-th largest singular value of :math:`A`. Note that :math:`\sigma_{k
+ 1}` is the best possible error for a rank-:math:`k` approximation
and, in fact, is achieved by the singular value decomposition (SVD)
:math:`A \approx U S V^{*}`, where :math:`U \in \mathbb{C}^{m \times
k}` and :math:`V \in \mathbb{C}^{n \times k}` have orthonormal columns
and :math:`S = \mathop{\mathrm{diag}} (\sigma_{i}) \in \mathbb{C}^{k
\times k}` is diagonal with nonnegative entries. The principal
advantages of using an ID over an SVD are that:

- it is cheaper to construct;
- it preserves the structure of :math:`A`; and
- it is more efficient to compute with in light of the identity submatrix of :math:`P`.

Routines
========

Main functionality:

.. autosummary::
   :toctree: generated/

   interp_decomp
   reconstruct_matrix_from_id
   reconstruct_interp_matrix
   reconstruct_skel_matrix
   id_to_svd
   svd
   estimate_spectral_norm
   estimate_spectral_norm_diff
   estimate_rank

Following support functions are deprecated and will be removed in SciPy 1.17.0:

.. autosummary::
   :toctree: generated/

   seed
   rand


References
==========

This module uses the algorithms found in ID software package [1]_ by Martinsson,
Rokhlin, Shkolnisky, and Tygert, which is a Fortran library for computing IDs using
various algorithms, including the rank-revealing QR approach of [2]_ and the more
recent randomized methods described in [3]_, [4]_, and [5]_.

We advise the user to consult also the documentation for the `ID package
<http://tygert.com/software.html>`_.

.. [1] P.G. Martinsson, V. Rokhlin, Y. Shkolnisky, M. Tygert. "ID: a
    software package for low-rank approximation of matrices via interpolative
    decompositions, version 0.2." http://tygert.com/id_doc.4.pdf.

.. [2] H. Cheng, Z. Gimbutas, P.G. Martinsson, V. Rokhlin. "On the
    compression of low rank matrices." *SIAM J. Sci. Comput.* 26 (4): 1389--1404,
    2005. :doi:`10.1137/030602678`.

.. [3] E. Liberty, F. Woolfe, P.G. Martinsson, V. Rokhlin, M.
    Tygert. "Randomized algorithms for the low-rank approximation of matrices."
    *Proc. Natl. Acad. Sci. U.S.A.* 104 (51): 20167--20172, 2007.
    :doi:`10.1073/pnas.0709640104`.

.. [4] P.G. Martinsson, V. Rokhlin, M. Tygert. "A randomized
    algorithm for the decomposition of matrices." *Appl. Comput. Harmon. Anal.* 30
    (1): 47--68,  2011. :doi:`10.1016/j.acha.2010.02.003`.

.. [5] F. Woolfe, E. Liberty, V. Rokhlin, M. Tygert. "A fast
    randomized algorithm for the approximation of matrices." *Appl. Comput.
    Harmon. Anal.* 25 (3): 335--366, 2008. :doi:`10.1016/j.acha.2007.12.002`.


Tutorial
========

Initializing
------------

The first step is to import :mod:`scipy.linalg.interpolative` by issuing the
command:

>>> import scipy.linalg.interpolative as sli

Now let's build a matrix. For this, we consider a Hilbert matrix, which is well
know to have low rank:

>>> from scipy.linalg import hilbert
>>> n = 1000
>>> A = hilbert(n)

We can also do this explicitly via:

>>> import numpy as np
>>> n = 1000
>>> A = np.empty((n, n), order='F')
>>> for j in range(n):
...     for i in range(n):
...         A[i,j] = 1. / (i + j + 1)

Note the use of the flag ``order='F'`` in :func:`numpy.empty`. This
instantiates the matrix in Fortran-contiguous order and is important for
avoiding data copying when passing to the backend.

We then define multiplication routines for the matrix by regarding it as a
:class:`scipy.sparse.linalg.LinearOperator`:

>>> from scipy.sparse.linalg import aslinearoperator
>>> L = aslinearoperator(A)

This automatically sets up methods describing the action of the matrix and its
adjoint on a vector.

Computing an ID
---------------

We have several choices of algorithm to compute an ID. These fall largely
according to two dichotomies:

1. how the matrix is represented, i.e., via its entries or via its action on a
   vector; and
2. whether to approximate it to a fixed relative precision or to a fixed rank.

We step through each choice in turn below.

In all cases, the ID is represented by three parameters:

1. a rank ``k``;
2. an index array ``idx``; and
3. interpolation coefficients ``proj``.

The ID is specified by the relation
``np.dot(A[:,idx[:k]], proj) == A[:,idx[k:]]``.

From matrix entries
...................

We first consider a matrix given in terms of its entries.

To compute an ID to a fixed precision, type:

>>> eps = 1e-3
>>> k, idx, proj = sli.interp_decomp(A, eps)

where ``eps < 1`` is the desired precision.

To compute an ID to a fixed rank, use:

>>> idx, proj = sli.interp_decomp(A, k)

where ``k >= 1`` is the desired rank.

Both algorithms use random sampling and are usually faster than the
corresponding older, deterministic algorithms, which can be accessed via the
commands:

>>> k, idx, proj = sli.interp_decomp(A, eps, rand=False)

and:

>>> idx, proj = sli.interp_decomp(A, k, rand=False)

respectively.

From matrix action
..................

Now consider a matrix given in terms of its action on a vector as a
:class:`scipy.sparse.linalg.LinearOperator`.

To compute an ID to a fixed precision, type:

>>> k, idx, proj = sli.interp_decomp(L, eps)

To compute an ID to a fixed rank, use:

>>> idx, proj = sli.interp_decomp(L, k)

These algorithms are randomized.

Reconstructing an ID
--------------------

The ID routines above do not output the skeleton and interpolation matrices
explicitly but instead return the relevant information in a more compact (and
sometimes more useful) form. To build these matrices, write:

>>> B = sli.reconstruct_skel_matrix(A, k, idx)

for the skeleton matrix and:

>>> P = sli.reconstruct_interp_matrix(idx, proj)

for the interpolation matrix. The ID approximation can then be computed as:

>>> C = np.dot(B, P)

This can also be constructed directly using:

>>> C = sli.reconstruct_matrix_from_id(B, idx, proj)

without having to first compute ``P``.

Alternatively, this can be done explicitly as well using:

>>> B = A[:,idx[:k]]
>>> P = np.hstack([np.eye(k), proj])[:,np.argsort(idx)]
>>> C = np.dot(B, P)

Computing an SVD
----------------

An ID can be converted to an SVD via the command:

>>> U, S, V = sli.id_to_svd(B, idx, proj)

The SVD approximation is then:

>>> approx = U @ np.diag(S) @ V.conj().T

The SVD can also be computed "fresh" by combining both the ID and conversion
steps into one command. Following the various ID algorithms above, there are
correspondingly various SVD algorithms that one can employ.

From matrix entries
...................

We consider first SVD algorithms for a matrix given in terms of its entries.

To compute an SVD to a fixed precision, type:

>>> U, S, V = sli.svd(A, eps)

To compute an SVD to a fixed rank, use:

>>> U, S, V = sli.svd(A, k)

Both algorithms use random sampling; for the deterministic versions, issue the
keyword ``rand=False`` as above.

From matrix action
..................

Now consider a matrix given in terms of its action on a vector.

To compute an SVD to a fixed precision, type:

>>> U, S, V = sli.svd(L, eps)

To compute an SVD to a fixed rank, use:

>>> U, S, V = sli.svd(L, k)

Utility routines
----------------

Several utility routines are also available.

To estimate the spectral norm of a matrix, use:

>>> snorm = sli.estimate_spectral_norm(A)

This algorithm is based on the randomized power method and thus requires only
matrix-vector products. The number of iterations to take can be set using the
keyword ``its`` (default: ``its=20``). The matrix is interpreted as a
:class:`scipy.sparse.linalg.LinearOperator`, but it is also valid to supply it
as a :class:`numpy.ndarray`, in which case it is trivially converted using
:func:`scipy.sparse.linalg.aslinearoperator`.

The same algorithm can also estimate the spectral norm of the difference of two
matrices ``A1`` and ``A2`` as follows:

>>> A1, A2 = A**2, A
>>> diff = sli.estimate_spectral_norm_diff(A1, A2)

This is often useful for checking the accuracy of a matrix approximation.

Some routines in :mod:`scipy.linalg.interpolative` require estimating the rank
of a matrix as well. This can be done with either:

>>> k = sli.estimate_rank(A, eps)

or:

>>> k = sli.estimate_rank(L, eps)

depending on the representation. The parameter ``eps`` controls the definition
of the numerical rank.

Finally, the random number generation required for all randomized routines can
be controlled via providing NumPy pseudo-random generators with a fixed seed. See
:class:`numpy.random.Generator` and :func:`numpy.random.default_rng` for more details.

Remarks
-------

The above functions all automatically detect the appropriate interface and work
with both real and complex data types, passing input arguments to the proper
backend routine.

"""

import scipy.linalg._decomp_interpolative as _backend
import numpy as np
import warnings

__all__ = [
    'estimate_rank',
    'estimate_spectral_norm',
    'estimate_spectral_norm_diff',
    'id_to_svd',
    'interp_decomp',
    'rand',
    'reconstruct_interp_matrix',
    'reconstruct_matrix_from_id',
    'reconstruct_skel_matrix',
    'seed',
    'svd',
]

_DTYPE_ERROR = ValueError("invalid input dtype (input must be float64 or complex128)")
_TYPE_ERROR = TypeError("invalid input type (must be array or LinearOperator)")


def _C_contiguous_copy(A):
    """
    Same as np.ascontiguousarray, but ensure a copy
    """
    A = np.asarray(A)
    if A.flags.c_contiguous:
        A = A.copy()
    else:
        A = np.ascontiguousarray(A)
    return A


def _is_real(A):
    try:
        if A.dtype == np.complex128:
            return False
        elif A.dtype == np.float64:
            return True
        else:
            raise _DTYPE_ERROR
    except AttributeError as e:
        raise _TYPE_ERROR from e


def seed(seed=None):
    """
    This function, historically, used to set the seed of the randomization algorithms
    used in the `scipy.linalg.interpolative` functions written in Fortran77.

    The library has been ported to Python and now the functions use the native NumPy
    generators and this function has no content and returns None. Thus this function
    should not be used and will be removed in SciPy version 1.17.0.
    """
    warnings.warn("`scipy.linalg.interpolative.seed` is deprecated and will be "
                  "removed in SciPy 1.17.0.", DeprecationWarning, stacklevel=3)


def rand(*shape):
    """
    This function, historically, used to generate uniformly distributed random number
    for the randomization algorithms used in the `scipy.linalg.interpolative` functions
    written in Fortran77.

    The library has been ported to Python and now the functions use the native NumPy
    generators. Thus this function should not be used and will be removed in the
    SciPy version 1.17.0.

    If pseudo-random numbers are needed, NumPy pseudo-random generators should be used
    instead.

    Parameters
    ----------
    *shape
        Shape of output array

    """
    warnings.warn("`scipy.linalg.interpolative.rand` is deprecated and will be "
                  "removed in SciPy 1.17.0.", DeprecationWarning, stacklevel=3)
    rng = np.random.default_rng()
    return rng.uniform(low=0., high=1.0, size=shape)


def interp_decomp(A, eps_or_k, rand=True, rng=None):
    """
    Compute ID of a matrix.

    An ID of a matrix `A` is a factorization defined by a rank `k`, a column
    index array `idx`, and interpolation coefficients `proj` such that::

        numpy.dot(A[:,idx[:k]], proj) = A[:,idx[k:]]

    The original matrix can then be reconstructed as::

        numpy.hstack([A[:,idx[:k]],
                                    numpy.dot(A[:,idx[:k]], proj)]
                                )[:,numpy.argsort(idx)]

    or via the routine :func:`reconstruct_matrix_from_id`. This can
    equivalently be written as::

        numpy.dot(A[:,idx[:k]],
                            numpy.hstack([numpy.eye(k), proj])
                          )[:,np.argsort(idx)]

    in terms of the skeleton and interpolation matrices::

        B = A[:,idx[:k]]

    and::

        P = numpy.hstack([numpy.eye(k), proj])[:,np.argsort(idx)]

    respectively. See also :func:`reconstruct_interp_matrix` and
    :func:`reconstruct_skel_matrix`.

    The ID can be computed to any relative precision or rank (depending on the
    value of `eps_or_k`). If a precision is specified (`eps_or_k < 1`), then
    this function has the output signature::

        k, idx, proj = interp_decomp(A, eps_or_k)

    Otherwise, if a rank is specified (`eps_or_k >= 1`), then the output
    signature is::

        idx, proj = interp_decomp(A, eps_or_k)

    ..  This function automatically detects the form of the input parameters
        and passes them to the appropriate backend. For details, see
        :func:`_backend.iddp_id`, :func:`_backend.iddp_aid`,
        :func:`_backend.iddp_rid`, :func:`_backend.iddr_id`,
        :func:`_backend.iddr_aid`, :func:`_backend.iddr_rid`,
        :func:`_backend.idzp_id`, :func:`_backend.idzp_aid`,
        :func:`_backend.idzp_rid`, :func:`_backend.idzr_id`,
        :func:`_backend.idzr_aid`, and :func:`_backend.idzr_rid`.

    Parameters
    ----------
    A : :class:`numpy.ndarray` or :class:`scipy.sparse.linalg.LinearOperator` with `rmatvec`
        Matrix to be factored
    eps_or_k : float or int
        Relative error (if ``eps_or_k < 1``) or rank (if ``eps_or_k >= 1``) of
        approximation.
    rand : bool, optional
        Whether to use random sampling if `A` is of type :class:`numpy.ndarray`
        (randomized algorithms are always used if `A` is of type
        :class:`scipy.sparse.linalg.LinearOperator`).
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.
        If `rand` is ``False``, the argument is ignored.

    Returns
    -------
    k : int
        Rank required to achieve specified relative precision if
        ``eps_or_k < 1``.
    idx : :class:`numpy.ndarray`
        Column index array.
    proj : :class:`numpy.ndarray`
        Interpolation coefficients.
    """  # numpy/numpydoc#87  # noqa: E501
    from scipy.sparse.linalg import LinearOperator
    rng = np.random.default_rng(rng)
    real = _is_real(A)

    if isinstance(A, np.ndarray):
        A = _C_contiguous_copy(A)
        if eps_or_k < 1:
            eps = eps_or_k
            if rand:
                if real:
                    k, idx, proj = _backend.iddp_aid(A, eps, rng=rng)
                else:
                    k, idx, proj = _backend.idzp_aid(A, eps, rng=rng)
            else:
                if real:
                    k, idx, proj = _backend.iddp_id(A, eps)
                else:
                    k, idx, proj = _backend.idzp_id(A, eps)
            return k, idx, proj
        else:
            k = int(eps_or_k)
            if rand:
                if real:
                    idx, proj = _backend.iddr_aid(A, k, rng=rng)
                else:
                    idx, proj = _backend.idzr_aid(A, k, rng=rng)
            else:
                if real:
                    idx, proj = _backend.iddr_id(A, k)
                else:
                    idx, proj = _backend.idzr_id(A, k)
            return idx, proj
    elif isinstance(A, LinearOperator):

        if eps_or_k < 1:
            eps = eps_or_k
            if real:
                k, idx, proj = _backend.iddp_rid(A, eps, rng=rng)
            else:
                k, idx, proj = _backend.idzp_rid(A, eps, rng=rng)
            return k, idx, proj
        else:
            k = int(eps_or_k)
            if real:
                idx, proj = _backend.iddr_rid(A, k, rng=rng)
            else:
                idx, proj = _backend.idzr_rid(A, k, rng=rng)
            return idx, proj
    else:
        raise _TYPE_ERROR


def reconstruct_matrix_from_id(B, idx, proj):
    """
    Reconstruct matrix from its ID.

    A matrix `A` with skeleton matrix `B` and ID indices and coefficients `idx`
    and `proj`, respectively, can be reconstructed as::

        numpy.hstack([B, numpy.dot(B, proj)])[:,numpy.argsort(idx)]

    See also :func:`reconstruct_interp_matrix` and
    :func:`reconstruct_skel_matrix`.

    ..  This function automatically detects the matrix data type and calls the
        appropriate backend. For details, see :func:`_backend.idd_reconid` and
        :func:`_backend.idz_reconid`.

    Parameters
    ----------
    B : :class:`numpy.ndarray`
        Skeleton matrix.
    idx : :class:`numpy.ndarray`
        Column index array.
    proj : :class:`numpy.ndarray`
        Interpolation coefficients.

    Returns
    -------
    :class:`numpy.ndarray`
        Reconstructed matrix.
    """
    if _is_real(B):
        return _backend.idd_reconid(B, idx, proj)
    else:
        return _backend.idz_reconid(B, idx, proj)


def reconstruct_interp_matrix(idx, proj):
    """
    Reconstruct interpolation matrix from ID.

    The interpolation matrix can be reconstructed from the ID indices and
    coefficients `idx` and `proj`, respectively, as::

        P = numpy.hstack([numpy.eye(proj.shape[0]), proj])[:,numpy.argsort(idx)]

    The original matrix can then be reconstructed from its skeleton matrix ``B``
    via ``A = B @ P``

    See also :func:`reconstruct_matrix_from_id` and
    :func:`reconstruct_skel_matrix`.

    ..  This function automatically detects the matrix data type and calls the
        appropriate backend. For details, see :func:`_backend.idd_reconint` and
        :func:`_backend.idz_reconint`.

    Parameters
    ----------
    idx : :class:`numpy.ndarray`
        1D column index array.
    proj : :class:`numpy.ndarray`
        Interpolation coefficients.

    Returns
    -------
    :class:`numpy.ndarray`
        Interpolation matrix.
    """
    n, krank = len(idx), proj.shape[0]
    if _is_real(proj):
        p = np.zeros([krank, n], dtype=np.float64)
    else:
        p = np.zeros([krank, n], dtype=np.complex128)

    for ci in range(krank):
        p[ci, idx[ci]] = 1.0
    p[:, idx[krank:]] = proj[:, :]

    return p


def reconstruct_skel_matrix(A, k, idx):
    """
    Reconstruct skeleton matrix from ID.

    The skeleton matrix can be reconstructed from the original matrix `A` and its
    ID rank and indices `k` and `idx`, respectively, as::

        B = A[:,idx[:k]]

    The original matrix can then be reconstructed via::

        numpy.hstack([B, numpy.dot(B, proj)])[:,numpy.argsort(idx)]

    See also :func:`reconstruct_matrix_from_id` and
    :func:`reconstruct_interp_matrix`.

    ..  This function automatically detects the matrix data type and calls the
        appropriate backend. For details, see :func:`_backend.idd_copycols` and
        :func:`_backend.idz_copycols`.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Original matrix.
    k : int
        Rank of ID.
    idx : :class:`numpy.ndarray`
        Column index array.

    Returns
    -------
    :class:`numpy.ndarray`
        Skeleton matrix.
    """
    return A[:, idx[:k]]


def id_to_svd(B, idx, proj):
    """
    Convert ID to SVD.

    The SVD reconstruction of a matrix with skeleton matrix `B` and ID indices and
    coefficients `idx` and `proj`, respectively, is::

        U, S, V = id_to_svd(B, idx, proj)
        A = numpy.dot(U, numpy.dot(numpy.diag(S), V.conj().T))

    See also :func:`svd`.

    ..  This function automatically detects the matrix data type and calls the
        appropriate backend. For details, see :func:`_backend.idd_id2svd` and
        :func:`_backend.idz_id2svd`.

    Parameters
    ----------
    B : :class:`numpy.ndarray`
        Skeleton matrix.
    idx : :class:`numpy.ndarray`
        1D column index array.
    proj : :class:`numpy.ndarray`
        Interpolation coefficients.

    Returns
    -------
    U : :class:`numpy.ndarray`
        Left singular vectors.
    S : :class:`numpy.ndarray`
        Singular values.
    V : :class:`numpy.ndarray`
        Right singular vectors.
    """
    B = _C_contiguous_copy(B)
    if _is_real(B):
        U, S, V = _backend.idd_id2svd(B, idx, proj)
    else:
        U, S, V = _backend.idz_id2svd(B, idx, proj)

    return U, S, V


def estimate_spectral_norm(A, its=20, rng=None):
    """
    Estimate spectral norm of a matrix by the randomized power method.

    ..  This function automatically detects the matrix data type and calls the
        appropriate backend. For details, see :func:`_backend.idd_snorm` and
        :func:`_backend.idz_snorm`.

    Parameters
    ----------
    A : :class:`scipy.sparse.linalg.LinearOperator`
        Matrix given as a :class:`scipy.sparse.linalg.LinearOperator` with the
        `matvec` and `rmatvec` methods (to apply the matrix and its adjoint).
    its : int, optional
        Number of power method iterations.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.
        If `rand` is ``False``, the argument is ignored.

    Returns
    -------
    float
        Spectral norm estimate.
    """
    from scipy.sparse.linalg import aslinearoperator
    rng = np.random.default_rng(rng)
    A = aslinearoperator(A)

    if _is_real(A):
        return _backend.idd_snorm(A, its=its, rng=rng)
    else:
        return _backend.idz_snorm(A, its=its, rng=rng)


def estimate_spectral_norm_diff(A, B, its=20, rng=None):
    """
    Estimate spectral norm of the difference of two matrices by the randomized
    power method.

    ..  This function automatically detects the matrix data type and calls the
        appropriate backend. For details, see :func:`_backend.idd_diffsnorm` and
        :func:`_backend.idz_diffsnorm`.

    Parameters
    ----------
    A : :class:`scipy.sparse.linalg.LinearOperator`
        First matrix given as a :class:`scipy.sparse.linalg.LinearOperator` with the
        `matvec` and `rmatvec` methods (to apply the matrix and its adjoint).
    B : :class:`scipy.sparse.linalg.LinearOperator`
        Second matrix given as a :class:`scipy.sparse.linalg.LinearOperator` with
        the `matvec` and `rmatvec` methods (to apply the matrix and its adjoint).
    its : int, optional
        Number of power method iterations.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.
        If `rand` is ``False``, the argument is ignored.

    Returns
    -------
    float
        Spectral norm estimate of matrix difference.
    """
    from scipy.sparse.linalg import aslinearoperator
    rng = np.random.default_rng(rng)
    A = aslinearoperator(A)
    B = aslinearoperator(B)

    if _is_real(A):
        return _backend.idd_diffsnorm(A, B, its=its, rng=rng)
    else:
        return _backend.idz_diffsnorm(A, B, its=its, rng=rng)


def svd(A, eps_or_k, rand=True, rng=None):
    """
    Compute SVD of a matrix via an ID.

    An SVD of a matrix `A` is a factorization::

        A = U @ np.diag(S) @ V.conj().T

    where `U` and `V` have orthonormal columns and `S` is nonnegative.

    The SVD can be computed to any relative precision or rank (depending on the
    value of `eps_or_k`).

    See also :func:`interp_decomp` and :func:`id_to_svd`.

    ..  This function automatically detects the form of the input parameters and
        passes them to the appropriate backend. For details, see
        :func:`_backend.iddp_svd`, :func:`_backend.iddp_asvd`,
        :func:`_backend.iddp_rsvd`, :func:`_backend.iddr_svd`,
        :func:`_backend.iddr_asvd`, :func:`_backend.iddr_rsvd`,
        :func:`_backend.idzp_svd`, :func:`_backend.idzp_asvd`,
        :func:`_backend.idzp_rsvd`, :func:`_backend.idzr_svd`,
        :func:`_backend.idzr_asvd`, and :func:`_backend.idzr_rsvd`.

    Parameters
    ----------
    A : :class:`numpy.ndarray` or :class:`scipy.sparse.linalg.LinearOperator`
        Matrix to be factored, given as either a :class:`numpy.ndarray` or a
        :class:`scipy.sparse.linalg.LinearOperator` with the `matvec` and
        `rmatvec` methods (to apply the matrix and its adjoint).
    eps_or_k : float or int
        Relative error (if ``eps_or_k < 1``) or rank (if ``eps_or_k >= 1``) of
        approximation.
    rand : bool, optional
        Whether to use random sampling if `A` is of type :class:`numpy.ndarray`
        (randomized algorithms are always used if `A` is of type
        :class:`scipy.sparse.linalg.LinearOperator`).
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.
        If `rand` is ``False``, the argument is ignored.

    Returns
    -------
    U : :class:`numpy.ndarray`
        2D array of left singular vectors.
    S : :class:`numpy.ndarray`
        1D array of singular values.
    V : :class:`numpy.ndarray`
        2D array right singular vectors.
    """
    from scipy.sparse.linalg import LinearOperator
    rng = np.random.default_rng(rng)

    real = _is_real(A)

    if isinstance(A, np.ndarray):
        A = _C_contiguous_copy(A)
        if eps_or_k < 1:
            eps = eps_or_k
            if rand:
                if real:
                    U, S, V = _backend.iddp_asvd(A, eps, rng=rng)
                else:
                    U, S, V = _backend.idzp_asvd(A, eps, rng=rng)
            else:
                if real:
                    U, S, V = _backend.iddp_svd(A, eps)
                    V = V.T.conj()
                else:
                    U, S, V = _backend.idzp_svd(A, eps)
                    V = V.T.conj()
        else:
            k = int(eps_or_k)
            if k > min(A.shape):
                raise ValueError(f"Approximation rank {k} exceeds min(A.shape) = "
                                 f" {min(A.shape)} ")
            if rand:
                if real:
                    U, S, V = _backend.iddr_asvd(A, k, rng=rng)
                else:
                    U, S, V = _backend.idzr_asvd(A, k, rng=rng)
            else:
                if real:
                    U, S, V = _backend.iddr_svd(A, k)
                    V = V.T.conj()
                else:
                    U, S, V = _backend.idzr_svd(A, k)
                    V = V.T.conj()
    elif isinstance(A, LinearOperator):
        if eps_or_k < 1:
            eps = eps_or_k
            if real:
                U, S, V = _backend.iddp_rsvd(A, eps, rng=rng)
            else:
                U, S, V = _backend.idzp_rsvd(A, eps, rng=rng)
        else:
            k = int(eps_or_k)
            if real:
                U, S, V = _backend.iddr_rsvd(A, k, rng=rng)
            else:
                U, S, V = _backend.idzr_rsvd(A, k, rng=rng)
    else:
        raise _TYPE_ERROR
    return U, S, V


def estimate_rank(A, eps, rng=None):
    """
    Estimate matrix rank to a specified relative precision using randomized
    methods.

    The matrix `A` can be given as either a :class:`numpy.ndarray` or a
    :class:`scipy.sparse.linalg.LinearOperator`, with different algorithms used
    for each case. If `A` is of type :class:`numpy.ndarray`, then the output
    rank is typically about 8 higher than the actual numerical rank.

    ..  This function automatically detects the form of the input parameters and
        passes them to the appropriate backend. For details,
        see :func:`_backend.idd_estrank`, :func:`_backend.idd_findrank`,
        :func:`_backend.idz_estrank`, and :func:`_backend.idz_findrank`.

    Parameters
    ----------
    A : :class:`numpy.ndarray` or :class:`scipy.sparse.linalg.LinearOperator`
        Matrix whose rank is to be estimated, given as either a
        :class:`numpy.ndarray` or a :class:`scipy.sparse.linalg.LinearOperator`
        with the `rmatvec` method (to apply the matrix adjoint).
    eps : float
        Relative error for numerical rank definition.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.
        If `rand` is ``False``, the argument is ignored.

    Returns
    -------
    int
        Estimated matrix rank.
    """
    from scipy.sparse.linalg import LinearOperator

    rng = np.random.default_rng(rng)
    real = _is_real(A)

    if isinstance(A, np.ndarray):
        A = _C_contiguous_copy(A)
        if real:
            rank, _ = _backend.idd_estrank(A, eps, rng=rng)
        else:
            rank, _ = _backend.idz_estrank(A, eps, rng=rng)
        if rank == 0:
            # special return value for nearly full rank
            rank = min(A.shape)
        return rank
    elif isinstance(A, LinearOperator):
        if real:
            return _backend.idd_findrank(A, eps, rng=rng)[0]
        else:
            return _backend.idz_findrank(A, eps, rng=rng)[0]
    else:
        raise _TYPE_ERROR
<|endoftext|>"""Frechet derivative of the matrix exponential."""
import numpy as np
import scipy.linalg

__all__ = ['expm_frechet', 'expm_cond']


def expm_frechet(A, E, method=None, compute_expm=True, check_finite=True):
    """
    Frechet derivative of the matrix exponential of A in the direction E.

    Parameters
    ----------
    A : (N, N) array_like
        Matrix of which to take the matrix exponential.
    E : (N, N) array_like
        Matrix direction in which to take the Frechet derivative.
    method : str, optional
        Choice of algorithm. Should be one of

        - `SPS` (default)
        - `blockEnlarge`

    compute_expm : bool, optional
        Whether to compute also `expm_A` in addition to `expm_frechet_AE`.
        Default is True.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    expm_A : ndarray
        Matrix exponential of A.
    expm_frechet_AE : ndarray
        Frechet derivative of the matrix exponential of A in the direction E.
    For ``compute_expm = False``, only `expm_frechet_AE` is returned.

    See Also
    --------
    expm : Compute the exponential of a matrix.

    Notes
    -----
    This section describes the available implementations that can be selected
    by the `method` parameter. The default method is *SPS*.

    Method *blockEnlarge* is a naive algorithm.

    Method *SPS* is Scaling-Pade-Squaring [1]_.
    It is a sophisticated implementation which should take
    only about 3/8 as much time as the naive implementation.
    The asymptotics are the same.

    .. versionadded:: 0.13.0

    References
    ----------
    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2009)
           Computing the Frechet Derivative of the Matrix Exponential,
           with an application to Condition Number Estimation.
           SIAM Journal On Matrix Analysis and Applications.,
           30 (4). pp. 1639-1657. ISSN 1095-7162

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import linalg
    >>> rng = np.random.default_rng()

    >>> A = rng.standard_normal((3, 3))
    >>> E = rng.standard_normal((3, 3))
    >>> expm_A, expm_frechet_AE = linalg.expm_frechet(A, E)
    >>> expm_A.shape, expm_frechet_AE.shape
    ((3, 3), (3, 3))

    Create a 6x6 matrix containing [[A, E], [0, A]]:

    >>> M = np.zeros((6, 6))
    >>> M[:3, :3] = A
    >>> M[:3, 3:] = E
    >>> M[3:, 3:] = A

    >>> expm_M = linalg.expm(M)
    >>> np.allclose(expm_A, expm_M[:3, :3])
    True
    >>> np.allclose(expm_frechet_AE, expm_M[:3, 3:])
    True

    """
    if check_finite:
        A = np.asarray_chkfinite(A)
        E = np.asarray_chkfinite(E)
    else:
        A = np.asarray(A)
        E = np.asarray(E)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be a square matrix')
    if E.ndim != 2 or E.shape[0] != E.shape[1]:
        raise ValueError('expected E to be a square matrix')
    if A.shape != E.shape:
        raise ValueError('expected A and E to be the same shape')
    if method is None:
        method = 'SPS'
    if method == 'SPS':
        expm_A, expm_frechet_AE = expm_frechet_algo_64(A, E)
    elif method == 'blockEnlarge':
        expm_A, expm_frechet_AE = expm_frechet_block_enlarge(A, E)
    else:
        raise ValueError(f'Unknown implementation {method}')
    if compute_expm:
        return expm_A, expm_frechet_AE
    else:
        return expm_frechet_AE


def expm_frechet_block_enlarge(A, E):
    """
    This is a helper function, mostly for testing and profiling.
    Return expm(A), frechet(A, E)
    """
    n = A.shape[0]
    M = np.vstack([
        np.hstack([A, E]),
        np.hstack([np.zeros_like(A), A])])
    expm_M = scipy.linalg.expm(M)
    return expm_M[:n, :n], expm_M[:n, n:]


"""
Maximal values ell_m of ||2**-s A|| such that the backward error bound
does not exceed 2**-53.
"""
ell_table_61 = (
        None,
        # 1
        2.11e-8,
        3.56e-4,
        1.08e-2,
        6.49e-2,
        2.00e-1,
        4.37e-1,
        7.83e-1,
        1.23e0,
        1.78e0,
        2.42e0,
        # 11
        3.13e0,
        3.90e0,
        4.74e0,
        5.63e0,
        6.56e0,
        7.52e0,
        8.53e0,
        9.56e0,
        1.06e1,
        1.17e1,
        )


# The b vectors and U and V are copypasted
# from scipy.sparse.linalg.matfuncs.py.
# M, Lu, Lv follow (6.11), (6.12), (6.13), (3.3)

def _diff_pade3(A, E, ident):
    b = (120., 60., 12., 1.)
    A2 = A.dot(A)
    M2 = np.dot(A, E) + np.dot(E, A)
    U = A.dot(b[3]*A2 + b[1]*ident)
    V = b[2]*A2 + b[0]*ident
    Lu = A.dot(b[3]*M2) + E.dot(b[3]*A2 + b[1]*ident)
    Lv = b[2]*M2
    return U, V, Lu, Lv


def _diff_pade5(A, E, ident):
    b = (30240., 15120., 3360., 420., 30., 1.)
    A2 = A.dot(A)
    M2 = np.dot(A, E) + np.dot(E, A)
    A4 = np.dot(A2, A2)
    M4 = np.dot(A2, M2) + np.dot(M2, A2)
    U = A.dot(b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = b[4]*A4 + b[2]*A2 + b[0]*ident
    Lu = (A.dot(b[5]*M4 + b[3]*M2) +
            E.dot(b[5]*A4 + b[3]*A2 + b[1]*ident))
    Lv = b[4]*M4 + b[2]*M2
    return U, V, Lu, Lv


def _diff_pade7(A, E, ident):
    b = (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.)
    A2 = A.dot(A)
    M2 = np.dot(A, E) + np.dot(E, A)
    A4 = np.dot(A2, A2)
    M4 = np.dot(A2, M2) + np.dot(M2, A2)
    A6 = np.dot(A2, A4)
    M6 = np.dot(A4, M2) + np.dot(M4, A2)
    U = A.dot(b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    Lu = (A.dot(b[7]*M6 + b[5]*M4 + b[3]*M2) +
            E.dot(b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident))
    Lv = b[6]*M6 + b[4]*M4 + b[2]*M2
    return U, V, Lu, Lv


def _diff_pade9(A, E, ident):
    b = (17643225600., 8821612800., 2075673600., 302702400., 30270240.,
            2162160., 110880., 3960., 90., 1.)
    A2 = A.dot(A)
    M2 = np.dot(A, E) + np.dot(E, A)
    A4 = np.dot(A2, A2)
    M4 = np.dot(A2, M2) + np.dot(M2, A2)
    A6 = np.dot(A2, A4)
    M6 = np.dot(A4, M2) + np.dot(M4, A2)
    A8 = np.dot(A4, A4)
    M8 = np.dot(A4, M4) + np.dot(M4, A4)
    U = A.dot(b[9]*A8 + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident)
    V = b[8]*A8 + b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
    Lu = (A.dot(b[9]*M8 + b[7]*M6 + b[5]*M4 + b[3]*M2) +
            E.dot(b[9]*A8 + b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident))
    Lv = b[8]*M8 + b[6]*M6 + b[4]*M4 + b[2]*M2
    return U, V, Lu, Lv


def expm_frechet_algo_64(A, E):
    n = A.shape[0]
    s = None
    ident = np.identity(n)
    A_norm_1 = scipy.linalg.norm(A, 1)
    m_pade_pairs = (
            (3, _diff_pade3),
            (5, _diff_pade5),
            (7, _diff_pade7),
            (9, _diff_pade9))
    for m, pade in m_pade_pairs:
        if A_norm_1 <= ell_table_61[m]:
            U, V, Lu, Lv = pade(A, E, ident)
            s = 0
            break
    if s is None:
        # scaling
        s = max(0, int(np.ceil(np.log2(A_norm_1 / ell_table_61[13]))))
        A = A * 2.0**-s
        E = E * 2.0**-s
        # pade order 13
        A2 = np.dot(A, A)
        M2 = np.dot(A, E) + np.dot(E, A)
        A4 = np.dot(A2, A2)
        M4 = np.dot(A2, M2) + np.dot(M2, A2)
        A6 = np.dot(A2, A4)
        M6 = np.dot(A4, M2) + np.dot(M4, A2)
        b = (64764752532480000., 32382376266240000., 7771770303897600.,
                1187353796428800., 129060195264000., 10559470521600.,
                670442572800., 33522128640., 1323241920., 40840800., 960960.,
                16380., 182., 1.)
        W1 = b[13]*A6 + b[11]*A4 + b[9]*A2
        W2 = b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*ident
        Z1 = b[12]*A6 + b[10]*A4 + b[8]*A2
        Z2 = b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*ident
        W = np.dot(A6, W1) + W2
        U = np.dot(A, W)
        V = np.dot(A6, Z1) + Z2
        Lw1 = b[13]*M6 + b[11]*M4 + b[9]*M2
        Lw2 = b[7]*M6 + b[5]*M4 + b[3]*M2
        Lz1 = b[12]*M6 + b[10]*M4 + b[8]*M2
        Lz2 = b[6]*M6 + b[4]*M4 + b[2]*M2
        Lw = np.dot(A6, Lw1) + np.dot(M6, W1) + Lw2
        Lu = np.dot(A, Lw) + np.dot(E, W)
        Lv = np.dot(A6, Lz1) + np.dot(M6, Z1) + Lz2
    # factor once and solve twice
    lu_piv = scipy.linalg.lu_factor(-U + V)
    R = scipy.linalg.lu_solve(lu_piv, U + V)
    L = scipy.linalg.lu_solve(lu_piv, Lu + Lv + np.dot((Lu - Lv), R))
    # squaring
    for k in range(s):
        L = np.dot(R, L) + np.dot(L, R)
        R = np.dot(R, R)
    return R, L


def vec(M):
    """
    Stack columns of M to construct a single vector.

    This is somewhat standard notation in linear algebra.

    Parameters
    ----------
    M : 2-D array_like
        Input matrix

    Returns
    -------
    v : 1-D ndarray
        Output vector

    """
    return M.T.ravel()


def expm_frechet_kronform(A, method=None, check_finite=True):
    """
    Construct the Kronecker form of the Frechet derivative of expm.

    Parameters
    ----------
    A : array_like with shape (N, N)
        Matrix to be expm'd.
    method : str, optional
        Extra keyword to be passed to expm_frechet.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    K : 2-D ndarray with shape (N*N, N*N)
        Kronecker form of the Frechet derivative of the matrix exponential.

    Notes
    -----
    This function is used to help compute the condition number
    of the matrix exponential.

    See Also
    --------
    expm : Compute a matrix exponential.
    expm_frechet : Compute the Frechet derivative of the matrix exponential.
    expm_cond : Compute the relative condition number of the matrix exponential
                in the Frobenius norm.

    """
    if check_finite:
        A = np.asarray_chkfinite(A)
    else:
        A = np.asarray(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected a square matrix')

    n = A.shape[0]
    ident = np.identity(n)
    cols = []
    for i in range(n):
        for j in range(n):
            E = np.outer(ident[i], ident[j])
            F = expm_frechet(A, E,
                    method=method, compute_expm=False, check_finite=False)
            cols.append(vec(F))
    return np.vstack(cols).T


def expm_cond(A, check_finite=True):
    """
    Relative condition number of the matrix exponential in the Frobenius norm.

    Parameters
    ----------
    A : 2-D array_like
        Square input matrix with shape (N, N).
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    kappa : float
        The relative condition number of the matrix exponential
        in the Frobenius norm

    See Also
    --------
    expm : Compute the exponential of a matrix.
    expm_frechet : Compute the Frechet derivative of the matrix exponential.

    Notes
    -----
    A faster estimate for the condition number in the 1-norm
    has been published but is not yet implemented in SciPy.

    .. versionadded:: 0.14.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import expm_cond
    >>> A = np.array([[-0.3, 0.2, 0.6], [0.6, 0.3, -0.1], [-0.7, 1.2, 0.9]])
    >>> k = expm_cond(A)
    >>> k
    1.7787805864469866

    """
    if check_finite:
        A = np.asarray_chkfinite(A)
    else:
        A = np.asarray(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected a square matrix')

    X = scipy.linalg.expm(A)
    K = expm_frechet_kronform(A, check_finite=False)

    # The following norm choices are deliberate.
    # The norms of A and X are Frobenius norms,
    # and the norm of K is the induced 2-norm.
    A_norm = scipy.linalg.norm(A, 'fro')
    X_norm = scipy.linalg.norm(X, 'fro')
    K_norm = scipy.linalg.norm(K, 2)

    kappa = (K_norm * A_norm) / X_norm
    return kappa
<|endoftext|>import warnings

import numpy as np
from numpy import asarray_chkfinite
from ._misc import LinAlgError, _datacopied, LinAlgWarning
from .lapack import get_lapack_funcs


__all__ = ['qz', 'ordqz']

_double_precision = ['i', 'l', 'd']


def _select_function(sort):
    if callable(sort):
        # assume the user knows what they're doing
        sfunction = sort
    elif sort == 'lhp':
        sfunction = _lhp
    elif sort == 'rhp':
        sfunction = _rhp
    elif sort == 'iuc':
        sfunction = _iuc
    elif sort == 'ouc':
        sfunction = _ouc
    else:
        raise ValueError("sort parameter must be None, a callable, or "
                         "one of ('lhp','rhp','iuc','ouc')")

    return sfunction


def _lhp(x, y):
    out = np.empty_like(x, dtype=bool)
    nonzero = (y != 0)
    # handles (x, y) = (0, 0) too
    out[~nonzero] = False
    out[nonzero] = (np.real(x[nonzero]/y[nonzero]) < 0.0)
    return out


def _rhp(x, y):
    out = np.empty_like(x, dtype=bool)
    nonzero = (y != 0)
    # handles (x, y) = (0, 0) too
    out[~nonzero] = False
    out[nonzero] = (np.real(x[nonzero]/y[nonzero]) > 0.0)
    return out


def _iuc(x, y):
    out = np.empty_like(x, dtype=bool)
    nonzero = (y != 0)
    # handles (x, y) = (0, 0) too
    out[~nonzero] = False
    out[nonzero] = (abs(x[nonzero]/y[nonzero]) < 1.0)
    return out


def _ouc(x, y):
    out = np.empty_like(x, dtype=bool)
    xzero = (x == 0)
    yzero = (y == 0)
    out[xzero & yzero] = False
    out[~xzero & yzero] = True
    out[~yzero] = (abs(x[~yzero]/y[~yzero]) > 1.0)
    return out


def _qz(A, B, output='real', lwork=None, sort=None, overwrite_a=False,
        overwrite_b=False, check_finite=True):
    if sort is not None:
        # Disabled due to segfaults on win32, see ticket 1717.
        raise ValueError("The 'sort' input of qz() has to be None and will be "
                         "removed in a future release. Use ordqz instead.")

    if output not in ['real', 'complex', 'r', 'c']:
        raise ValueError("argument must be 'real', or 'complex'")

    if check_finite:
        a1 = asarray_chkfinite(A)
        b1 = asarray_chkfinite(B)
    else:
        a1 = np.asarray(A)
        b1 = np.asarray(B)

    a_m, a_n = a1.shape
    b_m, b_n = b1.shape
    if not (a_m == a_n == b_m == b_n):
        raise ValueError("Array dimensions must be square and agree")

    typa = a1.dtype.char
    if output in ['complex', 'c'] and typa not in ['F', 'D']:
        if typa in _double_precision:
            a1 = a1.astype('D')
            typa = 'D'
        else:
            a1 = a1.astype('F')
            typa = 'F'
    typb = b1.dtype.char
    if output in ['complex', 'c'] and typb not in ['F', 'D']:
        if typb in _double_precision:
            b1 = b1.astype('D')
            typb = 'D'
        else:
            b1 = b1.astype('F')
            typb = 'F'

    overwrite_a = overwrite_a or (_datacopied(a1, A))
    overwrite_b = overwrite_b or (_datacopied(b1, B))

    gges, = get_lapack_funcs(('gges',), (a1, b1))

    if lwork is None or lwork == -1:
        # get optimal work array size
        result = gges(lambda x: None, a1, b1, lwork=-1)
        lwork = result[-2][0].real.astype(int)

    def sfunction(x):
        return None
    result = gges(sfunction, a1, b1, lwork=lwork, overwrite_a=overwrite_a,
                  overwrite_b=overwrite_b, sort_t=0)

    info = result[-1]
    if info < 0:
        raise ValueError(f"Illegal value in argument {-info} of gges")
    elif info > 0 and info <= a_n:
        warnings.warn("The QZ iteration failed. (a,b) are not in Schur "
                      "form, but ALPHAR(j), ALPHAI(j), and BETA(j) should be "
                      f"correct for J={info-1},...,N", LinAlgWarning,
                      stacklevel=3)
    elif info == a_n+1:
        raise LinAlgError("Something other than QZ iteration failed")
    elif info == a_n+2:
        raise LinAlgError("After reordering, roundoff changed values of some "
                          "complex eigenvalues so that leading eigenvalues "
                          "in the Generalized Schur form no longer satisfy "
                          "sort=True. This could also be due to scaling.")
    elif info == a_n+3:
        raise LinAlgError("Reordering failed in <s,d,c,z>tgsen")

    return result, gges.typecode


def qz(A, B, output='real', lwork=None, sort=None, overwrite_a=False,
       overwrite_b=False, check_finite=True):
    """
    QZ decomposition for generalized eigenvalues of a pair of matrices.

    The QZ, or generalized Schur, decomposition for a pair of n-by-n
    matrices (A,B) is::

        (A,B) = (Q @ AA @ Z*, Q @ BB @ Z*)

    where AA, BB is in generalized Schur form if BB is upper-triangular
    with non-negative diagonal and AA is upper-triangular, or for real QZ
    decomposition (``output='real'``) block upper triangular with 1x1
    and 2x2 blocks. In this case, the 1x1 blocks correspond to real
    generalized eigenvalues and 2x2 blocks are 'standardized' by making
    the corresponding elements of BB have the form::

        [ a 0 ]
        [ 0 b ]

    and the pair of corresponding 2x2 blocks in AA and BB will have a complex
    conjugate pair of generalized eigenvalues. If (``output='complex'``) or
    A and B are complex matrices, Z' denotes the conjugate-transpose of Z.
    Q and Z are unitary matrices.

    Parameters
    ----------
    A : (N, N) array_like
        2-D array to decompose
    B : (N, N) array_like
        2-D array to decompose
    output : {'real', 'complex'}, optional
        Construct the real or complex QZ decomposition for real matrices.
        Default is 'real'.
    lwork : int, optional
        Work array size. If None or -1, it is automatically computed.
    sort : {None, callable, 'lhp', 'rhp', 'iuc', 'ouc'}, optional
        NOTE: THIS INPUT IS DISABLED FOR NOW. Use ordqz instead.

        Specifies whether the upper eigenvalues should be sorted. A callable
        may be passed that, given a eigenvalue, returns a boolean denoting
        whether the eigenvalue should be sorted to the top-left (True). For
        real matrix pairs, the sort function takes three real arguments
        (alphar, alphai, beta). The eigenvalue
        ``x = (alphar + alphai*1j)/beta``. For complex matrix pairs or
        output='complex', the sort function takes two complex arguments
        (alpha, beta). The eigenvalue ``x = (alpha/beta)``.  Alternatively,
        string parameters may be used:

            - 'lhp'   Left-hand plane (x.real < 0.0)
            - 'rhp'   Right-hand plane (x.real > 0.0)
            - 'iuc'   Inside the unit circle (x*x.conjugate() < 1.0)
            - 'ouc'   Outside the unit circle (x*x.conjugate() > 1.0)

        Defaults to None (no sorting).
    overwrite_a : bool, optional
        Whether to overwrite data in a (may improve performance)
    overwrite_b : bool, optional
        Whether to overwrite data in b (may improve performance)
    check_finite : bool, optional
        If true checks the elements of `A` and `B` are finite numbers. If
        false does no checking and passes matrix through to
        underlying algorithm.

    Returns
    -------
    AA : (N, N) ndarray
        Generalized Schur form of A.
    BB : (N, N) ndarray
        Generalized Schur form of B.
    Q : (N, N) ndarray
        The left Schur vectors.
    Z : (N, N) ndarray
        The right Schur vectors.

    See Also
    --------
    ordqz

    Notes
    -----
    Q is transposed versus the equivalent function in Matlab.

    .. versionadded:: 0.11.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import qz

    >>> A = np.array([[1, 2, -1], [5, 5, 5], [2, 4, -8]])
    >>> B = np.array([[1, 1, -3], [3, 1, -1], [5, 6, -2]])

    Compute the decomposition.  The QZ decomposition is not unique, so
    depending on the underlying library that is used, there may be
    differences in the signs of coefficients in the following output.

    >>> AA, BB, Q, Z = qz(A, B)
    >>> AA
    array([[-1.36949157, -4.05459025,  7.44389431],
           [ 0.        ,  7.65653432,  5.13476017],
           [ 0.        , -0.65978437,  2.4186015 ]])  # may vary
    >>> BB
    array([[ 1.71890633, -1.64723705, -0.72696385],
           [ 0.        ,  8.6965692 , -0.        ],
           [ 0.        ,  0.        ,  2.27446233]])  # may vary
    >>> Q
    array([[-0.37048362,  0.1903278 ,  0.90912992],
           [-0.90073232,  0.16534124, -0.40167593],
           [ 0.22676676,  0.96769706, -0.11017818]])  # may vary
    >>> Z
    array([[-0.67660785,  0.63528924, -0.37230283],
           [ 0.70243299,  0.70853819, -0.06753907],
           [ 0.22088393, -0.30721526, -0.92565062]])  # may vary

    Verify the QZ decomposition.  With real output, we only need the
    transpose of ``Z`` in the following expressions.

    >>> Q @ AA @ Z.T  # Should be A
    array([[ 1.,  2., -1.],
           [ 5.,  5.,  5.],
           [ 2.,  4., -8.]])
    >>> Q @ BB @ Z.T  # Should be B
    array([[ 1.,  1., -3.],
           [ 3.,  1., -1.],
           [ 5.,  6., -2.]])

    Repeat the decomposition, but with ``output='complex'``.

    >>> AA, BB, Q, Z = qz(A, B, output='complex')

    For conciseness in the output, we use ``np.set_printoptions()`` to set
    the output precision of NumPy arrays to 3 and display tiny values as 0.

    >>> np.set_printoptions(precision=3, suppress=True)
    >>> AA
    array([[-1.369+0.j   ,  2.248+4.237j,  4.861-5.022j],
           [ 0.   +0.j   ,  7.037+2.922j,  0.794+4.932j],
           [ 0.   +0.j   ,  0.   +0.j   ,  2.655-1.103j]])  # may vary
    >>> BB
    array([[ 1.719+0.j   , -1.115+1.j   , -0.763-0.646j],
           [ 0.   +0.j   ,  7.24 +0.j   , -3.144+3.322j],
           [ 0.   +0.j   ,  0.   +0.j   ,  2.732+0.j   ]])  # may vary
    >>> Q
    array([[ 0.326+0.175j, -0.273-0.029j, -0.886-0.052j],
           [ 0.794+0.426j, -0.093+0.134j,  0.402-0.02j ],
           [-0.2  -0.107j, -0.816+0.482j,  0.151-0.167j]])  # may vary
    >>> Z
    array([[ 0.596+0.32j , -0.31 +0.414j,  0.393-0.347j],
           [-0.619-0.332j, -0.479+0.314j,  0.154-0.393j],
           [-0.195-0.104j,  0.576+0.27j ,  0.715+0.187j]])  # may vary

    With complex arrays, we must use ``Z.conj().T`` in the following
    expressions to verify the decomposition.

    >>> Q @ AA @ Z.conj().T  # Should be A
    array([[ 1.-0.j,  2.-0.j, -1.-0.j],
           [ 5.+0.j,  5.+0.j,  5.-0.j],
           [ 2.+0.j,  4.+0.j, -8.+0.j]])
    >>> Q @ BB @ Z.conj().T  # Should be B
    array([[ 1.+0.j,  1.+0.j, -3.+0.j],
           [ 3.-0.j,  1.-0.j, -1.+0.j],
           [ 5.+0.j,  6.+0.j, -2.+0.j]])

    """
    # output for real
    # AA, BB, sdim, alphar, alphai, beta, vsl, vsr, work, info
    # output for complex
    # AA, BB, sdim, alpha, beta, vsl, vsr, work, info
    result, _ = _qz(A, B, output=output, lwork=lwork, sort=sort,
                    overwrite_a=overwrite_a, overwrite_b=overwrite_b,
                    check_finite=check_finite)
    return result[0], result[1], result[-4], result[-3]


def ordqz(A, B, sort='lhp', output='real', overwrite_a=False,
          overwrite_b=False, check_finite=True):
    """QZ decomposition for a pair of matrices with reordering.

    Parameters
    ----------
    A : (N, N) array_like
        2-D array to decompose
    B : (N, N) array_like
        2-D array to decompose
    sort : {callable, 'lhp', 'rhp', 'iuc', 'ouc'}, optional
        Specifies whether the upper eigenvalues should be sorted. A
        callable may be passed that, given an ordered pair ``(alpha,
        beta)`` representing the eigenvalue ``x = (alpha/beta)``,
        returns a boolean denoting whether the eigenvalue should be
        sorted to the top-left (True). For the real matrix pairs
        ``beta`` is real while ``alpha`` can be complex, and for
        complex matrix pairs both ``alpha`` and ``beta`` can be
        complex. The callable must be able to accept a NumPy
        array. Alternatively, string parameters may be used:

            - 'lhp'   Left-hand plane (x.real < 0.0)
            - 'rhp'   Right-hand plane (x.real > 0.0)
            - 'iuc'   Inside the unit circle (x*x.conjugate() < 1.0)
            - 'ouc'   Outside the unit circle (x*x.conjugate() > 1.0)

        With the predefined sorting functions, an infinite eigenvalue
        (i.e., ``alpha != 0`` and ``beta = 0``) is considered to lie in
        neither the left-hand nor the right-hand plane, but it is
        considered to lie outside the unit circle. For the eigenvalue
        ``(alpha, beta) = (0, 0)``, the predefined sorting functions
        all return `False`.
    output : str {'real','complex'}, optional
        Construct the real or complex QZ decomposition for real matrices.
        Default is 'real'.
    overwrite_a : bool, optional
        If True, the contents of A are overwritten.
    overwrite_b : bool, optional
        If True, the contents of B are overwritten.
    check_finite : bool, optional
        If true checks the elements of `A` and `B` are finite numbers. If
        false does no checking and passes matrix through to
        underlying algorithm.

    Returns
    -------
    AA : (N, N) ndarray
        Generalized Schur form of A.
    BB : (N, N) ndarray
        Generalized Schur form of B.
    alpha : (N,) ndarray
        alpha = alphar + alphai * 1j. See notes.
    beta : (N,) ndarray
        See notes.
    Q : (N, N) ndarray
        The left Schur vectors.
    Z : (N, N) ndarray
        The right Schur vectors.

    See Also
    --------
    qz

    Notes
    -----
    On exit, ``(ALPHAR(j) + ALPHAI(j)*i)/BETA(j), j=1,...,N``, will be the
    generalized eigenvalues.  ``ALPHAR(j) + ALPHAI(j)*i`` and
    ``BETA(j),j=1,...,N`` are the diagonals of the complex Schur form (S,T)
    that would result if the 2-by-2 diagonal blocks of the real generalized
    Schur form of (A,B) were further reduced to triangular form using complex
    unitary transformations. If ALPHAI(j) is zero, then the jth eigenvalue is
    real; if positive, then the ``j``\\ th and ``(j+1)``\\ st eigenvalues are a
    complex conjugate pair, with ``ALPHAI(j+1)`` negative.

    .. versionadded:: 0.17.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import ordqz
    >>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
    >>> B = np.array([[0, 6, 0, 0], [5, 0, 2, 1], [5, 2, 6, 6], [4, 7, 7, 7]])
    >>> AA, BB, alpha, beta, Q, Z = ordqz(A, B, sort='lhp')

    Since we have sorted for left half plane eigenvalues, negatives come first

    >>> (alpha/beta).real < 0
    array([ True,  True, False, False], dtype=bool)

    """
    (AA, BB, _, *ab, Q, Z, _, _), typ = _qz(A, B, output=output, sort=None,
                                            overwrite_a=overwrite_a,
                                            overwrite_b=overwrite_b,
                                            check_finite=check_finite)

    if typ == 's':
        alpha, beta = ab[0] + ab[1]*np.complex64(1j), ab[2]
    elif typ == 'd':
        alpha, beta = ab[0] + ab[1]*1.j, ab[2]
    else:
        alpha, beta = ab

    sfunction = _select_function(sort)
    select = sfunction(alpha, beta)

    tgsen = get_lapack_funcs('tgsen', (AA, BB))
    # the real case needs 4n + 16 lwork
    lwork = 4*AA.shape[0] + 16 if typ in 'sd' else 1
    AAA, BBB, *ab, QQ, ZZ, _, _, _, _, info = tgsen(select, AA, BB, Q, Z,
                                                    ijob=0,
                                                    lwork=lwork, liwork=1)

    # Once more for tgsen output
    if typ == 's':
        alpha, beta = ab[0] + ab[1]*np.complex64(1j), ab[2]
    elif typ == 'd':
        alpha, beta = ab[0] + ab[1]*1.j, ab[2]
    else:
        alpha, beta = ab

    if info < 0:
        raise ValueError(f"Illegal value in argument {-info} of tgsen")
    elif info == 1:
        raise ValueError("Reordering of (A, B) failed because the transformed"
                         " matrix pair (A, B) would be too far from "
                         "generalized Schur form; the problem is very "
                         "ill-conditioned. (A, B) may have been partially "
                         "reordered.")

    return AAA, BBB, alpha, beta, QQ, ZZ
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.linalg` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'LinAlgError', 'LinAlgWarning', 'norm', 'get_blas_funcs',
    'get_lapack_funcs'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="linalg", module="misc",
                                   private_modules=["_misc"], all=__all__,
                                   attribute=name)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.linalg` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'eig', 'eigvals', 'eigh', 'eigvalsh',
    'eig_banded', 'eigvals_banded',
    'eigh_tridiagonal', 'eigvalsh_tridiagonal', 'hessenberg', 'cdf2rdf',
    'LinAlgError', 'norm', 'get_lapack_funcs'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="linalg", module="decomp",
                                   private_modules=["_decomp"], all=__all__,
                                   attribute=name)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.linalg` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'solve', 'solve_triangular', 'solveh_banded', 'solve_banded',
    'solve_toeplitz', 'solve_circulant', 'inv', 'det', 'lstsq',
    'pinv', 'pinvh', 'matrix_balance', 'matmul_toeplitz',
    'get_lapack_funcs', 'LinAlgError', 'LinAlgWarning',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="linalg", module="basic",
                                   private_modules=["_basic"], all=__all__,
                                   attribute=name)
<|endoftext|>"""
====================================
Linear algebra (:mod:`scipy.linalg`)
====================================

.. currentmodule:: scipy.linalg

.. toctree::
   :hidden:

   linalg.blas
   linalg.cython_blas
   linalg.cython_lapack
   linalg.interpolative
   linalg.lapack

Linear algebra functions.

.. eventually, we should replace the numpy.linalg HTML link with just `numpy.linalg`

.. seealso::

   `numpy.linalg <https://www.numpy.org/devdocs/reference/routines.linalg.html>`__
   for more linear algebra functions. Note that
   although `scipy.linalg` imports most of them, identically named
   functions from `scipy.linalg` may offer more or slightly differing
   functionality.


Basics
======

.. autosummary::
   :toctree: generated/

   inv - Find the inverse of a square matrix
   solve - Solve a linear system of equations
   solve_banded - Solve a banded linear system
   solveh_banded - Solve a Hermitian or symmetric banded system
   solve_circulant - Solve a circulant system
   solve_triangular - Solve a triangular matrix
   solve_toeplitz - Solve a toeplitz matrix
   matmul_toeplitz - Multiply a Toeplitz matrix with an array.
   det - Find the determinant of a square matrix
   norm - Matrix and vector norm
   lstsq - Solve a linear least-squares problem
   pinv - Pseudo-inverse (Moore-Penrose) using lstsq
   pinvh - Pseudo-inverse of hermitian matrix
   kron - Kronecker product of two arrays
   khatri_rao - Khatri-Rao product of two arrays
   orthogonal_procrustes - Solve an orthogonal Procrustes problem
   matrix_balance - Balance matrix entries with a similarity transformation
   subspace_angles - Compute the subspace angles between two matrices
   bandwidth - Return the lower and upper bandwidth of an array
   issymmetric - Check if a square 2D array is symmetric
   ishermitian - Check if a square 2D array is Hermitian
   LinAlgError
   LinAlgWarning

Eigenvalue Problems
===================

.. autosummary::
   :toctree: generated/

   eig - Find the eigenvalues and eigenvectors of a square matrix
   eigvals - Find just the eigenvalues of a square matrix
   eigh - Find the e-vals and e-vectors of a Hermitian or symmetric matrix
   eigvalsh - Find just the eigenvalues of a Hermitian or symmetric matrix
   eig_banded - Find the eigenvalues and eigenvectors of a banded matrix
   eigvals_banded - Find just the eigenvalues of a banded matrix
   eigh_tridiagonal - Find the eigenvalues and eigenvectors of a tridiagonal matrix
   eigvalsh_tridiagonal - Find just the eigenvalues of a tridiagonal matrix

Decompositions
==============

.. autosummary::
   :toctree: generated/

   lu - LU decomposition of a matrix
   lu_factor - LU decomposition returning unordered matrix and pivots
   lu_solve - Solve Ax=b using back substitution with output of lu_factor
   svd - Singular value decomposition of a matrix
   svdvals - Singular values of a matrix
   diagsvd - Construct matrix of singular values from output of svd
   orth - Construct orthonormal basis for the range of A using svd
   null_space - Construct orthonormal basis for the null space of A using svd
   ldl - LDL.T decomposition of a Hermitian or a symmetric matrix.
   cholesky - Cholesky decomposition of a matrix
   cholesky_banded - Cholesky decomp. of a sym. or Hermitian banded matrix
   cho_factor - Cholesky decomposition for use in solving a linear system
   cho_solve - Solve previously factored linear system
   cho_solve_banded - Solve previously factored banded linear system
   polar - Compute the polar decomposition.
   qr - QR decomposition of a matrix
   qr_multiply - QR decomposition and multiplication by Q
   qr_update - Rank k QR update
   qr_delete - QR downdate on row or column deletion
   qr_insert - QR update on row or column insertion
   rq - RQ decomposition of a matrix
   qz - QZ decomposition of a pair of matrices
   ordqz - QZ decomposition of a pair of matrices with reordering
   schur - Schur decomposition of a matrix
   rsf2csf - Real to complex Schur form
   hessenberg - Hessenberg form of a matrix
   cdf2rdf - Complex diagonal form to real diagonal block form
   cossin - Cosine sine decomposition of a unitary or orthogonal matrix

.. seealso::

   `scipy.linalg.interpolative` -- Interpolative matrix decompositions


Matrix Functions
================

.. autosummary::
   :toctree: generated/

   expm - Matrix exponential
   logm - Matrix logarithm
   cosm - Matrix cosine
   sinm - Matrix sine
   tanm - Matrix tangent
   coshm - Matrix hyperbolic cosine
   sinhm - Matrix hyperbolic sine
   tanhm - Matrix hyperbolic tangent
   signm - Matrix sign
   sqrtm - Matrix square root
   funm - Evaluating an arbitrary matrix function
   expm_frechet - Frechet derivative of the matrix exponential
   expm_cond - Relative condition number of expm in the Frobenius norm
   fractional_matrix_power - Fractional matrix power


Matrix Equation Solvers
=======================

.. autosummary::
   :toctree: generated/

   solve_sylvester - Solve the Sylvester matrix equation
   solve_continuous_are - Solve the continuous-time algebraic Riccati equation
   solve_discrete_are - Solve the discrete-time algebraic Riccati equation
   solve_continuous_lyapunov - Solve the continuous-time Lyapunov equation
   solve_discrete_lyapunov - Solve the discrete-time Lyapunov equation


Sketches and Random Projections
===============================

.. autosummary::
   :toctree: generated/

   clarkson_woodruff_transform - Applies the Clarkson Woodruff Sketch (a.k.a CountMin Sketch)

Special Matrices
================

.. autosummary::
   :toctree: generated/

   block_diag - Construct a block diagonal matrix from submatrices
   circulant - Circulant matrix
   companion - Companion matrix
   convolution_matrix - Convolution matrix
   dft - Discrete Fourier transform matrix
   fiedler - Fiedler matrix
   fiedler_companion - Fiedler companion matrix
   hadamard - Hadamard matrix of order 2**n
   hankel - Hankel matrix
   helmert - Helmert matrix
   hilbert - Hilbert matrix
   invhilbert - Inverse Hilbert matrix
   leslie - Leslie matrix
   pascal - Pascal matrix
   invpascal - Inverse Pascal matrix
   toeplitz - Toeplitz matrix

Low-level routines
==================

.. autosummary::
   :toctree: generated/

   get_blas_funcs
   get_lapack_funcs
   find_best_blas_type

.. seealso::

   `scipy.linalg.blas` -- Low-level BLAS functions

   `scipy.linalg.lapack` -- Low-level LAPACK functions

   `scipy.linalg.cython_blas` -- Low-level BLAS functions for Cython

   `scipy.linalg.cython_lapack` -- Low-level LAPACK functions for Cython

"""  # noqa: E501

from ._misc import *
from ._cythonized_array_utils import *
from ._basic import *
from ._decomp import *
from ._decomp_lu import *
from ._decomp_ldl import *
from ._decomp_cholesky import *
from ._decomp_qr import *
from ._decomp_qz import *
from ._decomp_svd import *
from ._decomp_schur import *
from ._decomp_polar import *
from ._matfuncs import *
from .blas import *
from .lapack import *
from ._special_matrices import *
from ._solvers import *
from ._procrustes import *
from ._decomp_update import *
from ._sketches import *
from ._decomp_cossin import *

# Deprecated namespaces, to be removed in v2.0.0
from . import (
    decomp, decomp_cholesky, decomp_lu, decomp_qr, decomp_svd, decomp_schur,
    basic, misc, special_matrices, matfuncs,
)

__all__ = [s for s in dir() if not s.startswith('_')]


from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
<|endoftext|>import numpy as np
from numpy.linalg import LinAlgError
from .blas import get_blas_funcs
from .lapack import get_lapack_funcs

__all__ = ['LinAlgError', 'LinAlgWarning', 'norm']


class LinAlgWarning(RuntimeWarning):
    """
    The warning emitted when a linear algebra related operation is close
    to fail conditions of the algorithm or loss of accuracy is expected.
    """
    pass


def norm(a, ord=None, axis=None, keepdims=False, check_finite=True):
    """
    Matrix or vector norm.

    This function is able to return one of eight different matrix norms,
    or one of an infinite number of vector norms (described below), depending
    on the value of the ``ord`` parameter. For tensors with rank different from
    1 or 2, only `ord=None` is supported.

    Parameters
    ----------
    a : array_like
        Input array. If `axis` is None, `a` must be 1-D or 2-D, unless `ord`
        is None. If both `axis` and `ord` are None, the 2-norm of
        ``a.ravel`` will be returned.
    ord : {int, inf, -inf, 'fro', 'nuc', None}, optional
        Order of the norm (see table under ``Notes``). inf means NumPy's
        `inf` object.
    axis : {int, 2-tuple of ints, None}, optional
        If `axis` is an integer, it specifies the axis of `a` along which to
        compute the vector norms. If `axis` is a 2-tuple, it specifies the
        axes that hold 2-D matrices, and the matrix norms of these matrices
        are computed. If `axis` is None then either a vector norm (when `a`
        is 1-D) or a matrix norm (when `a` is 2-D) is returned.
    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in the
        result as dimensions with size one. With this option the result will
        broadcast correctly against the original `a`.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    n : float or ndarray
        Norm of the matrix or vector(s).

    Notes
    -----
    For values of ``ord <= 0``, the result is, strictly speaking, not a
    mathematical 'norm', but it may still be useful for various numerical
    purposes.

    The following norms can be calculated:

    =====  ============================  ==========================
    ord    norm for matrices             norm for vectors
    =====  ============================  ==========================
    None   Frobenius norm                2-norm
    'fro'  Frobenius norm                --
    'nuc'  nuclear norm                  --
    inf    max(sum(abs(a), axis=1))      max(abs(a))
    -inf   min(sum(abs(a), axis=1))      min(abs(a))
    0      --                            sum(a != 0)
    1      max(sum(abs(a), axis=0))      as below
    -1     min(sum(abs(a), axis=0))      as below
    2      2-norm (largest sing. value)  as below
    -2     smallest singular value       as below
    other  --                            sum(abs(a)**ord)**(1./ord)
    =====  ============================  ==========================

    The Frobenius norm is given by [1]_:

        :math:`||A||_F = [\\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

    The nuclear norm is the sum of the singular values.

    Both the Frobenius and nuclear norm orders are only defined for
    matrices.

    References
    ----------
    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
           Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import norm
    >>> a = np.arange(9) - 4.0
    >>> a
    array([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
    >>> b = a.reshape((3, 3))
    >>> b
    array([[-4., -3., -2.],
           [-1.,  0.,  1.],
           [ 2.,  3.,  4.]])

    >>> norm(a)
    7.745966692414834
    >>> norm(b)
    7.745966692414834
    >>> norm(b, 'fro')
    7.745966692414834
    >>> norm(a, np.inf)
    4.0
    >>> norm(b, np.inf)
    9.0
    >>> norm(a, -np.inf)
    0.0
    >>> norm(b, -np.inf)
    2.0

    >>> norm(a, 1)
    20.0
    >>> norm(b, 1)
    7.0
    >>> norm(a, -1)
    -4.6566128774142013e-010
    >>> norm(b, -1)
    6.0
    >>> norm(a, 2)
    7.745966692414834
    >>> norm(b, 2)
    7.3484692283495345

    >>> norm(a, -2)
    0.0
    >>> norm(b, -2)
    1.8570331885190563e-016
    >>> norm(a, 3)
    5.8480354764257312
    >>> norm(a, -3)
    0.0

    """
    # Differs from numpy only in non-finite handling and the use of blas.
    if check_finite:
        a = np.asarray_chkfinite(a)
    else:
        a = np.asarray(a)

    if a.size and a.dtype.char in 'fdFD' and axis is None and not keepdims:

        if ord in (None, 2) and (a.ndim == 1):
            # use blas for fast and stable euclidean norm
            nrm2 = get_blas_funcs('nrm2', dtype=a.dtype, ilp64='preferred')
            return nrm2(a)

        if a.ndim == 2:
            # Use lapack for a couple fast matrix norms.
            # For some reason the *lange frobenius norm is slow.
            lange_args = None
            # Make sure this works if the user uses the axis keywords
            # to apply the norm to the transpose.
            if ord == 1:
                if np.isfortran(a):
                    lange_args = '1', a
                elif np.isfortran(a.T):
                    lange_args = 'i', a.T
            elif ord == np.inf:
                if np.isfortran(a):
                    lange_args = 'i', a
                elif np.isfortran(a.T):
                    lange_args = '1', a.T
            if lange_args:
                lange = get_lapack_funcs('lange', dtype=a.dtype, ilp64='preferred')
                return lange(*lange_args)

    # fall back to numpy in every other case
    return np.linalg.norm(a, ord=ord, axis=axis, keepdims=keepdims)


def _datacopied(arr, original):
    """
    Strict check for `arr` not sharing any data with `original`,
    under the assumption that arr = asarray(original)

    """
    if arr is original:
        return False
    if not isinstance(original, np.ndarray) and hasattr(original, '__array__'):
        return False
    return arr.base is None
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.linalg` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'qr', 'qr_multiply', 'rq', 'get_lapack_funcs'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="linalg", module="decomp_qr",
                                   private_modules=["_decomp_qr"], all=__all__,
                                   attribute=name)
<|endoftext|>"""SVD decomposition functions."""
import numpy as np
from numpy import zeros, r_, diag, dot, arccos, arcsin, where, clip

# Local imports.
from ._misc import LinAlgError, _datacopied
from .lapack import get_lapack_funcs, _compute_lwork
from ._decomp import _asarray_validated

__all__ = ['svd', 'svdvals', 'diagsvd', 'orth', 'subspace_angles', 'null_space']


def svd(a, full_matrices=True, compute_uv=True, overwrite_a=False,
        check_finite=True, lapack_driver='gesdd'):
    """
    Singular Value Decomposition.

    Factorizes the matrix `a` into two unitary matrices ``U`` and ``Vh``, and
    a 1-D array ``s`` of singular values (real, non-negative) such that
    ``a == U @ S @ Vh``, where ``S`` is a suitably shaped matrix of zeros with
    main diagonal ``s``.

    Parameters
    ----------
    a : (M, N) array_like
        Matrix to decompose.
    full_matrices : bool, optional
        If True (default), `U` and `Vh` are of shape ``(M, M)``, ``(N, N)``.
        If False, the shapes are ``(M, K)`` and ``(K, N)``, where
        ``K = min(M, N)``.
    compute_uv : bool, optional
        Whether to compute also ``U`` and ``Vh`` in addition to ``s``.
        Default is True.
    overwrite_a : bool, optional
        Whether to overwrite `a`; may improve performance.
        Default is False.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    lapack_driver : {'gesdd', 'gesvd'}, optional
        Whether to use the more efficient divide-and-conquer approach
        (``'gesdd'``) or general rectangular approach (``'gesvd'``)
        to compute the SVD. MATLAB and Octave use the ``'gesvd'`` approach.
        Default is ``'gesdd'``.

    Returns
    -------
    U : ndarray
        Unitary matrix having left singular vectors as columns.
        Of shape ``(M, M)`` or ``(M, K)``, depending on `full_matrices`.
    s : ndarray
        The singular values, sorted in non-increasing order.
        Of shape (K,), with ``K = min(M, N)``.
    Vh : ndarray
        Unitary matrix having right singular vectors as rows.
        Of shape ``(N, N)`` or ``(K, N)`` depending on `full_matrices`.

    For ``compute_uv=False``, only ``s`` is returned.

    Raises
    ------
    LinAlgError
        If SVD computation does not converge.

    See Also
    --------
    svdvals : Compute singular values of a matrix.
    diagsvd : Construct the Sigma matrix, given the vector s.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import linalg
    >>> rng = np.random.default_rng()
    >>> m, n = 9, 6
    >>> a = rng.standard_normal((m, n)) + 1.j*rng.standard_normal((m, n))
    >>> U, s, Vh = linalg.svd(a)
    >>> U.shape,  s.shape, Vh.shape
    ((9, 9), (6,), (6, 6))

    Reconstruct the original matrix from the decomposition:

    >>> sigma = np.zeros((m, n))
    >>> for i in range(min(m, n)):
    ...     sigma[i, i] = s[i]
    >>> a1 = np.dot(U, np.dot(sigma, Vh))
    >>> np.allclose(a, a1)
    True

    Alternatively, use ``full_matrices=False`` (notice that the shape of
    ``U`` is then ``(m, n)`` instead of ``(m, m)``):

    >>> U, s, Vh = linalg.svd(a, full_matrices=False)
    >>> U.shape, s.shape, Vh.shape
    ((9, 6), (6,), (6, 6))
    >>> S = np.diag(s)
    >>> np.allclose(a, np.dot(U, np.dot(S, Vh)))
    True

    >>> s2 = linalg.svd(a, compute_uv=False)
    >>> np.allclose(s, s2)
    True

    """
    a1 = _asarray_validated(a, check_finite=check_finite)
    if len(a1.shape) != 2:
        raise ValueError('expected matrix')
    m, n = a1.shape

    # accommodate empty matrix
    if a1.size == 0:
        u0, s0, v0 = svd(np.eye(2, dtype=a1.dtype))

        s = np.empty_like(a1, shape=(0,), dtype=s0.dtype)
        if full_matrices:
            u = np.empty_like(a1, shape=(m, m), dtype=u0.dtype)
            u[...] = np.identity(m)
            v = np.empty_like(a1, shape=(n, n), dtype=v0.dtype)
            v[...] = np.identity(n)
        else:
            u = np.empty_like(a1, shape=(m, 0), dtype=u0.dtype)
            v = np.empty_like(a1, shape=(0, n), dtype=v0.dtype)
        if compute_uv:
            return u, s, v
        else:
            return s

    overwrite_a = overwrite_a or (_datacopied(a1, a))

    if not isinstance(lapack_driver, str):
        raise TypeError('lapack_driver must be a string')
    if lapack_driver not in ('gesdd', 'gesvd'):
        message = f'lapack_driver must be "gesdd" or "gesvd", not "{lapack_driver}"'
        raise ValueError(message)

    if compute_uv:
        # XXX: revisit int32 when ILP64 lapack becomes a thing
        max_mn, min_mn = (m, n) if m > n else (n, m)
        if full_matrices:
            if max_mn*max_mn > np.iinfo(np.int32).max:
                raise ValueError(f"Indexing a matrix size {max_mn} x {max_mn} "
                                  "would incur integer overflow in LAPACK. "
                                  "Try using numpy.linalg.svd instead.")
        else:
            sz = max(m * min_mn, n * min_mn)
            if max(m * min_mn, n * min_mn) > np.iinfo(np.int32).max:
                raise ValueError(f"Indexing a matrix of {sz} elements would "
                                  "incur an in integer overflow in LAPACK. "
                                  "Try using numpy.linalg.svd instead.")

    funcs = (lapack_driver, lapack_driver + '_lwork')
    # XXX: As of 1.14.1 it isn't possible to build SciPy with ILP64,
    # so the following line always yields a LP64 (32-bit pointer size) variant
    gesXd, gesXd_lwork = get_lapack_funcs(funcs, (a1,), ilp64="preferred")

    # compute optimal lwork
    lwork = _compute_lwork(gesXd_lwork, a1.shape[0], a1.shape[1],
                           compute_uv=compute_uv, full_matrices=full_matrices)

    # perform decomposition
    u, s, v, info = gesXd(a1, compute_uv=compute_uv, lwork=lwork,
                          full_matrices=full_matrices, overwrite_a=overwrite_a)

    if info > 0:
        raise LinAlgError("SVD did not converge")
    if info < 0:
        raise ValueError('illegal value in %dth argument of internal gesdd'
                         % -info)
    if compute_uv:
        return u, s, v
    else:
        return s


def svdvals(a, overwrite_a=False, check_finite=True):
    """
    Compute singular values of a matrix.

    Parameters
    ----------
    a : (M, N) array_like
        Matrix to decompose.
    overwrite_a : bool, optional
        Whether to overwrite `a`; may improve performance.
        Default is False.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    s : (min(M, N),) ndarray
        The singular values, sorted in decreasing order.

    Raises
    ------
    LinAlgError
        If SVD computation does not converge.

    See Also
    --------
    svd : Compute the full singular value decomposition of a matrix.
    diagsvd : Construct the Sigma matrix, given the vector s.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import svdvals
    >>> m = np.array([[1.0, 0.0],
    ...               [2.0, 3.0],
    ...               [1.0, 1.0],
    ...               [0.0, 2.0],
    ...               [1.0, 0.0]])
    >>> svdvals(m)
    array([ 4.28091555,  1.63516424])

    We can verify the maximum singular value of `m` by computing the maximum
    length of `m.dot(u)` over all the unit vectors `u` in the (x,y) plane.
    We approximate "all" the unit vectors with a large sample. Because
    of linearity, we only need the unit vectors with angles in [0, pi].

    >>> t = np.linspace(0, np.pi, 2000)
    >>> u = np.array([np.cos(t), np.sin(t)])
    >>> np.linalg.norm(m.dot(u), axis=0).max()
    4.2809152422538475

    `p` is a projection matrix with rank 1. With exact arithmetic,
    its singular values would be [1, 0, 0, 0].

    >>> v = np.array([0.1, 0.3, 0.9, 0.3])
    >>> p = np.outer(v, v)
    >>> svdvals(p)
    array([  1.00000000e+00,   2.02021698e-17,   1.56692500e-17,
             8.15115104e-34])

    The singular values of an orthogonal matrix are all 1. Here, we
    create a random orthogonal matrix by using the `rvs()` method of
    `scipy.stats.ortho_group`.

    >>> from scipy.stats import ortho_group
    >>> orth = ortho_group.rvs(4)
    >>> svdvals(orth)
    array([ 1.,  1.,  1.,  1.])

    """
    return svd(a, compute_uv=0, overwrite_a=overwrite_a,
               check_finite=check_finite)


def diagsvd(s, M, N):
    """
    Construct the sigma matrix in SVD from singular values and size M, N.

    Parameters
    ----------
    s : (M,) or (N,) array_like
        Singular values
    M : int
        Size of the matrix whose singular values are `s`.
    N : int
        Size of the matrix whose singular values are `s`.

    Returns
    -------
    S : (M, N) ndarray
        The S-matrix in the singular value decomposition

    See Also
    --------
    svd : Singular value decomposition of a matrix
    svdvals : Compute singular values of a matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import diagsvd
    >>> vals = np.array([1, 2, 3])  # The array representing the computed svd
    >>> diagsvd(vals, 3, 4)
    array([[1, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 0]])
    >>> diagsvd(vals, 4, 3)
    array([[1, 0, 0],
           [0, 2, 0],
           [0, 0, 3],
           [0, 0, 0]])

    """
    part = diag(s)
    typ = part.dtype.char
    MorN = len(s)
    if MorN == M:
        return np.hstack((part, zeros((M, N - M), dtype=typ)))
    elif MorN == N:
        return r_[part, zeros((M - N, N), dtype=typ)]
    else:
        raise ValueError("Length of s must be M or N.")


# Orthonormal decomposition

def orth(A, rcond=None):
    """
    Construct an orthonormal basis for the range of A using SVD

    Parameters
    ----------
    A : (M, N) array_like
        Input array
    rcond : float, optional
        Relative condition number. Singular values ``s`` smaller than
        ``rcond * max(s)`` are considered zero.
        Default: floating point eps * max(M,N).

    Returns
    -------
    Q : (M, K) ndarray
        Orthonormal basis for the range of A.
        K = effective rank of A, as determined by rcond

    See Also
    --------
    svd : Singular value decomposition of a matrix
    null_space : Matrix null space

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import orth
    >>> A = np.array([[2, 0, 0], [0, 5, 0]])  # rank 2 array
    >>> orth(A)
    array([[0., 1.],
           [1., 0.]])
    >>> orth(A.T)
    array([[0., 1.],
           [1., 0.],
           [0., 0.]])

    """
    u, s, vh = svd(A, full_matrices=False)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s, initial=0.) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = u[:, :num]
    return Q


def null_space(A, rcond=None, *, overwrite_a=False, check_finite=True,
               lapack_driver='gesdd'):
    """
    Construct an orthonormal basis for the null space of A using SVD

    Parameters
    ----------
    A : (M, N) array_like
        Input array
    rcond : float, optional
        Relative condition number. Singular values ``s`` smaller than
        ``rcond * max(s)`` are considered zero.
        Default: floating point eps * max(M,N).
    overwrite_a : bool, optional
        Whether to overwrite `a`; may improve performance.
        Default is False.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    lapack_driver : {'gesdd', 'gesvd'}, optional
        Whether to use the more efficient divide-and-conquer approach
        (``'gesdd'``) or general rectangular approach (``'gesvd'``)
        to compute the SVD. MATLAB and Octave use the ``'gesvd'`` approach.
        Default is ``'gesdd'``.

    Returns
    -------
    Z : (N, K) ndarray
        Orthonormal basis for the null space of A.
        K = dimension of effective null space, as determined by rcond

    See Also
    --------
    svd : Singular value decomposition of a matrix
    orth : Matrix range

    Examples
    --------
    1-D null space:

    >>> import numpy as np
    >>> from scipy.linalg import null_space
    >>> A = np.array([[1, 1], [1, 1]])
    >>> ns = null_space(A)
    >>> ns * np.copysign(1, ns[0,0])  # Remove the sign ambiguity of the vector
    array([[ 0.70710678],
           [-0.70710678]])

    2-D null space:

    >>> from numpy.random import default_rng
    >>> rng = default_rng()
    >>> B = rng.random((3, 5))
    >>> Z = null_space(B)
    >>> Z.shape
    (5, 2)
    >>> np.allclose(B.dot(Z), 0)
    True

    The basis vectors are orthonormal (up to rounding error):

    >>> Z.T.dot(Z)
    array([[  1.00000000e+00,   6.92087741e-17],
           [  6.92087741e-17,   1.00000000e+00]])

    """
    u, s, vh = svd(A, full_matrices=True, overwrite_a=overwrite_a,
                   check_finite=check_finite, lapack_driver=lapack_driver)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s, initial=0.) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:,:].T.conj()
    return Q


def subspace_angles(A, B):
    r"""
    Compute the subspace angles between two matrices.

    Parameters
    ----------
    A : (M, N) array_like
        The first input array.
    B : (M, K) array_like
        The second input array.

    Returns
    -------
    angles : ndarray, shape (min(N, K),)
        The subspace angles between the column spaces of `A` and `B` in
        descending order.

    See Also
    --------
    orth
    svd

    Notes
    -----
    This computes the subspace angles according to the formula
    provided in [1]_. For equivalence with MATLAB and Octave behavior,
    use ``angles[0]``.

    .. versionadded:: 1.0

    References
    ----------
    .. [1] Knyazev A, Argentati M (2002) Principal Angles between Subspaces
           in an A-Based Scalar Product: Algorithms and Perturbation
           Estimates. SIAM J. Sci. Comput. 23:2008-2040.

    Examples
    --------
    An Hadamard matrix, which has orthogonal columns, so we expect that
    the suspace angle to be :math:`\frac{\pi}{2}`:

    >>> import numpy as np
    >>> from scipy.linalg import hadamard, subspace_angles
    >>> rng = np.random.default_rng()
    >>> H = hadamard(4)
    >>> print(H)
    [[ 1  1  1  1]
     [ 1 -1  1 -1]
     [ 1  1 -1 -1]
     [ 1 -1 -1  1]]
    >>> np.rad2deg(subspace_angles(H[:, :2], H[:, 2:]))
    array([ 90.,  90.])

    And the subspace angle of a matrix to itself should be zero:

    >>> subspace_angles(H[:, :2], H[:, :2]) <= 2 * np.finfo(float).eps
    array([ True,  True], dtype=bool)

    The angles between non-orthogonal subspaces are in between these extremes:

    >>> x = rng.standard_normal((4, 3))
    >>> np.rad2deg(subspace_angles(x[:, :2], x[:, [2]]))
    array([ 55.832])  # random
    """
    # Steps here omit the U and V calculation steps from the paper

    # 1. Compute orthonormal bases of column-spaces
    A = _asarray_validated(A, check_finite=True)
    if len(A.shape) != 2:
        raise ValueError(f'expected 2D array, got shape {A.shape}')
    QA = orth(A)
    del A

    B = _asarray_validated(B, check_finite=True)
    if len(B.shape) != 2:
        raise ValueError(f'expected 2D array, got shape {B.shape}')
    if len(B) != len(QA):
        raise ValueError('A and B must have the same number of rows, got '
                         f'{QA.shape[0]} and {B.shape[0]}')
    QB = orth(B)
    del B

    # 2. Compute SVD for cosine
    QA_H_QB = dot(QA.T.conj(), QB)
    sigma = svdvals(QA_H_QB)

    # 3. Compute matrix B
    if QA.shape[1] >= QB.shape[1]:
        B = QB - dot(QA, QA_H_QB)
    else:
        B = QA - dot(QB, QA_H_QB.T.conj())
    del QA, QB, QA_H_QB

    # 4. Compute SVD for sine
    mask = sigma ** 2 >= 0.5
    if mask.any():
        mu_arcsin = arcsin(clip(svdvals(B, overwrite_a=True), -1., 1.))
    else:
        mu_arcsin = 0.

    # 5. Compute the principal angles
    # with reverse ordering of sigma because smallest sigma belongs to largest
    # angle theta
    theta = where(mask, mu_arcsin, arccos(clip(sigma[::-1], -1., 1.)))
    return theta
<|endoftext|>""" Sketching-based Matrix Computations """

# Author: Jordi Montes <jomsdev@gmail.com>
# August 28, 2017

import numpy as np

from scipy._lib._util import (check_random_state, rng_integers,
                              _transition_to_rng)
from scipy.sparse import csc_matrix

__all__ = ['clarkson_woodruff_transform']


def cwt_matrix(n_rows, n_columns, rng=None):
    r"""
    Generate a matrix S which represents a Clarkson-Woodruff transform.

    Given the desired size of matrix, the method returns a matrix S of size
    (n_rows, n_columns) where each column has all the entries set to 0
    except for one position which has been randomly set to +1 or -1 with
    equal probability.

    Parameters
    ----------
    n_rows : int
        Number of rows of S
    n_columns : int
        Number of columns of S
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.


    Returns
    -------
    S : (n_rows, n_columns) csc_matrix
        The returned matrix has ``n_columns`` nonzero entries.

    Notes
    -----
    Given a matrix A, with probability at least 9/10,
    .. math:: \|SA\| = (1 \pm \epsilon)\|A\|
    Where the error epsilon is related to the size of S.
    """
    rng = check_random_state(rng)
    rows = rng_integers(rng, 0, n_rows, n_columns)
    cols = np.arange(n_columns+1)
    signs = rng.choice([1, -1], n_columns)
    S = csc_matrix((signs, rows, cols), shape=(n_rows, n_columns))
    return S


@_transition_to_rng("seed", position_num=2)
def clarkson_woodruff_transform(input_matrix, sketch_size, rng=None):
    r"""
    Applies a Clarkson-Woodruff Transform/sketch to the input matrix.

    Given an input_matrix ``A`` of size ``(n, d)``, compute a matrix ``A'`` of
    size (sketch_size, d) so that

    .. math:: \|Ax\| \approx \|A'x\|

    with high probability via the Clarkson-Woodruff Transform, otherwise
    known as the CountSketch matrix.

    Parameters
    ----------
    input_matrix : array_like
        Input matrix, of shape ``(n, d)``.
    sketch_size : int
        Number of rows for the sketch.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.

    Returns
    -------
    A' : array_like
        Sketch of the input matrix ``A``, of size ``(sketch_size, d)``.

    Notes
    -----
    To make the statement

    .. math:: \|Ax\| \approx \|A'x\|

    precise, observe the following result which is adapted from the
    proof of Theorem 14 of [2]_ via Markov's Inequality. If we have
    a sketch size ``sketch_size=k`` which is at least

    .. math:: k \geq \frac{2}{\epsilon^2\delta}

    Then for any fixed vector ``x``,

    .. math:: \|Ax\| = (1\pm\epsilon)\|A'x\|

    with probability at least one minus delta.

    This implementation takes advantage of sparsity: computing
    a sketch takes time proportional to ``A.nnz``. Data ``A`` which
    is in ``scipy.sparse.csc_matrix`` format gives the quickest
    computation time for sparse input.

    >>> import numpy as np
    >>> from scipy import linalg
    >>> from scipy import sparse
    >>> rng = np.random.default_rng()
    >>> n_rows, n_columns, density, sketch_n_rows = 15000, 100, 0.01, 200
    >>> A = sparse.rand(n_rows, n_columns, density=density, format='csc')
    >>> B = sparse.rand(n_rows, n_columns, density=density, format='csr')
    >>> C = sparse.rand(n_rows, n_columns, density=density, format='coo')
    >>> D = rng.standard_normal((n_rows, n_columns))
    >>> SA = linalg.clarkson_woodruff_transform(A, sketch_n_rows) # fastest
    >>> SB = linalg.clarkson_woodruff_transform(B, sketch_n_rows) # fast
    >>> SC = linalg.clarkson_woodruff_transform(C, sketch_n_rows) # slower
    >>> SD = linalg.clarkson_woodruff_transform(D, sketch_n_rows) # slowest

    That said, this method does perform well on dense inputs, just slower
    on a relative scale.

    References
    ----------
    .. [1] Kenneth L. Clarkson and David P. Woodruff. Low rank approximation
           and regression in input sparsity time. In STOC, 2013.
    .. [2] David P. Woodruff. Sketching as a tool for numerical linear algebra.
           In Foundations and Trends in Theoretical Computer Science, 2014.

    Examples
    --------
    Create a big dense matrix ``A`` for the example:

    >>> import numpy as np
    >>> from scipy import linalg
    >>> n_rows, n_columns  = 15000, 100
    >>> rng = np.random.default_rng()
    >>> A = rng.standard_normal((n_rows, n_columns))

    Apply the transform to create a new matrix with 200 rows:

    >>> sketch_n_rows = 200
    >>> sketch = linalg.clarkson_woodruff_transform(A, sketch_n_rows, seed=rng)
    >>> sketch.shape
    (200, 100)

    Now with high probability, the true norm is close to the sketched norm
    in absolute value.

    >>> linalg.norm(A)
    1224.2812927123198
    >>> linalg.norm(sketch)
    1226.518328407333

    Similarly, applying our sketch preserves the solution to a linear
    regression of :math:`\min \|Ax - b\|`.

    >>> b = rng.standard_normal(n_rows)
    >>> x = linalg.lstsq(A, b)[0]
    >>> Ab = np.hstack((A, b.reshape(-1, 1)))
    >>> SAb = linalg.clarkson_woodruff_transform(Ab, sketch_n_rows, seed=rng)
    >>> SA, Sb = SAb[:, :-1], SAb[:, -1]
    >>> x_sketched = linalg.lstsq(SA, Sb)[0]

    As with the matrix norm example, ``linalg.norm(A @ x - b)`` is close
    to ``linalg.norm(A @ x_sketched - b)`` with high probability.

    >>> linalg.norm(A @ x - b)
    122.83242365433877
    >>> linalg.norm(A @ x_sketched - b)
    166.58473879945151

    """
    S = cwt_matrix(sketch_size, input_matrix.shape[0], rng=rng)
    return S.dot(input_matrix)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.linalg` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'cholesky', 'cho_factor', 'cho_solve', 'cholesky_banded',
    'cho_solve_banded', 'LinAlgError', 'get_lapack_funcs'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="linalg", module="decomp_cholesky",
                                   private_modules=["_decomp_cholesky"], all=__all__,
                                   attribute=name)
<|endoftext|>"""Cholesky decomposition functions."""

import numpy as np
from numpy import asarray_chkfinite, asarray, atleast_2d, empty_like

# Local imports
from ._misc import LinAlgError, _datacopied
from .lapack import get_lapack_funcs

__all__ = ['cholesky', 'cho_factor', 'cho_solve', 'cholesky_banded',
           'cho_solve_banded']


def _cholesky(a, lower=False, overwrite_a=False, clean=True,
              check_finite=True):
    """Common code for cholesky() and cho_factor()."""

    a1 = asarray_chkfinite(a) if check_finite else asarray(a)
    a1 = atleast_2d(a1)

    # Dimension check
    if a1.ndim != 2:
        raise ValueError(f'Input array needs to be 2D but received a {a1.ndim}d-array.')
    # Squareness check
    if a1.shape[0] != a1.shape[1]:
        raise ValueError('Input array is expected to be square but has '
                         f'the shape: {a1.shape}.')

    # Quick return for square empty array
    if a1.size == 0:
        dt = cholesky(np.eye(1, dtype=a1.dtype)).dtype
        return empty_like(a1, dtype=dt), lower

    overwrite_a = overwrite_a or _datacopied(a1, a)
    potrf, = get_lapack_funcs(('potrf',), (a1,))
    c, info = potrf(a1, lower=lower, overwrite_a=overwrite_a, clean=clean)
    if info > 0:
        raise LinAlgError("%d-th leading minor of the array is not positive "
                          "definite" % info)
    if info < 0:
        raise ValueError(f'LAPACK reported an illegal value in {-info}-th argument'
                         'on entry to "POTRF".')
    return c, lower


def cholesky(a, lower=False, overwrite_a=False, check_finite=True):
    """
    Compute the Cholesky decomposition of a matrix.

    Returns the Cholesky decomposition, :math:`A = L L^*` or
    :math:`A = U^* U` of a Hermitian positive-definite matrix A.

    Parameters
    ----------
    a : (M, M) array_like
        Matrix to be decomposed
    lower : bool, optional
        Whether to compute the upper- or lower-triangular Cholesky
        factorization. During decomposition, only the selected half of the
        matrix is referenced. Default is upper-triangular.
    overwrite_a : bool, optional
        Whether to overwrite data in `a` (may improve performance).
    check_finite : bool, optional
        Whether to check that the entire input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    c : (M, M) ndarray
        Upper- or lower-triangular Cholesky factor of `a`.

    Raises
    ------
    LinAlgError : if decomposition fails.

    Notes
    -----
    During the finiteness check (if selected), the entire matrix `a` is
    checked. During decomposition, `a` is assumed to be symmetric or Hermitian
    (as applicable), and only the half selected by option `lower` is referenced.
    Consequently, if `a` is asymmetric/non-Hermitian, `cholesky` may still
    succeed if the symmetric/Hermitian matrix represented by the selected half
    is positive definite, yet it may fail if an element in the other half is
    non-finite.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import cholesky
    >>> a = np.array([[1,-2j],[2j,5]])
    >>> L = cholesky(a, lower=True)
    >>> L
    array([[ 1.+0.j,  0.+0.j],
           [ 0.+2.j,  1.+0.j]])
    >>> L @ L.T.conj()
    array([[ 1.+0.j,  0.-2.j],
           [ 0.+2.j,  5.+0.j]])

    """
    c, lower = _cholesky(a, lower=lower, overwrite_a=overwrite_a, clean=True,
                         check_finite=check_finite)
    return c


def cho_factor(a, lower=False, overwrite_a=False, check_finite=True):
    """
    Compute the Cholesky decomposition of a matrix, to use in cho_solve

    Returns a matrix containing the Cholesky decomposition,
    ``A = L L*`` or ``A = U* U`` of a Hermitian positive-definite matrix `a`.
    The return value can be directly used as the first parameter to cho_solve.

    .. warning::
        The returned matrix also contains random data in the entries not
        used by the Cholesky decomposition. If you need to zero these
        entries, use the function `cholesky` instead.

    Parameters
    ----------
    a : (M, M) array_like
        Matrix to be decomposed
    lower : bool, optional
        Whether to compute the upper or lower triangular Cholesky factorization.
        During decomposition, only the selected half of the matrix is referenced.
        (Default: upper-triangular)
    overwrite_a : bool, optional
        Whether to overwrite data in a (may improve performance)
    check_finite : bool, optional
        Whether to check that the entire input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    c : (M, M) ndarray
        Matrix whose upper or lower triangle contains the Cholesky factor
        of `a`. Other parts of the matrix contain random data.
    lower : bool
        Flag indicating whether the factor is in the lower or upper triangle

    Raises
    ------
    LinAlgError
        Raised if decomposition fails.

    See Also
    --------
    cho_solve : Solve a linear set equations using the Cholesky factorization
                of a matrix.

    Notes
    -----
    During the finiteness check (if selected), the entire matrix `a` is
    checked. During decomposition, `a` is assumed to be symmetric or Hermitian
    (as applicable), and only the half selected by option `lower` is referenced.
    Consequently, if `a` is asymmetric/non-Hermitian, `cholesky` may still
    succeed if the symmetric/Hermitian matrix represented by the selected half
    is positive definite, yet it may fail if an element in the other half is
    non-finite.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import cho_factor
    >>> A = np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]])
    >>> c, low = cho_factor(A)
    >>> c
    array([[3.        , 1.        , 0.33333333, 1.66666667],
           [3.        , 2.44948974, 1.90515869, -0.27216553],
           [1.        , 5.        , 2.29330749, 0.8559528 ],
           [5.        , 1.        , 2.        , 1.55418563]])
    >>> np.allclose(np.triu(c).T @ np. triu(c) - A, np.zeros((4, 4)))
    True

    """
    c, lower = _cholesky(a, lower=lower, overwrite_a=overwrite_a, clean=False,
                         check_finite=check_finite)
    return c, lower


def cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True):
    """Solve the linear equations A x = b, given the Cholesky factorization of A.

    Parameters
    ----------
    (c, lower) : tuple, (array, bool)
        Cholesky factorization of a, as given by cho_factor
    b : array
        Right-hand side
    overwrite_b : bool, optional
        Whether to overwrite data in b (may improve performance)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    x : array
        The solution to the system A x = b

    See Also
    --------
    cho_factor : Cholesky factorization of a matrix

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import cho_factor, cho_solve
    >>> A = np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]])
    >>> c, low = cho_factor(A)
    >>> x = cho_solve((c, low), [1, 1, 1, 1])
    >>> np.allclose(A @ x - [1, 1, 1, 1], np.zeros(4))
    True

    """
    (c, lower) = c_and_lower
    if check_finite:
        b1 = asarray_chkfinite(b)
        c = asarray_chkfinite(c)
    else:
        b1 = asarray(b)
        c = asarray(c)

    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("The factored matrix c is not square.")
    if c.shape[1] != b1.shape[0]:
        raise ValueError(f"incompatible dimensions ({c.shape} and {b1.shape})")

    # accommodate empty arrays
    if b1.size == 0:
        dt = cho_solve((np.eye(2, dtype=b1.dtype), True),
                        np.ones(2, dtype=c.dtype)).dtype
        return empty_like(b1, dtype=dt)

    overwrite_b = overwrite_b or _datacopied(b1, b)

    potrs, = get_lapack_funcs(('potrs',), (c, b1))
    x, info = potrs(c, b1, lower=lower, overwrite_b=overwrite_b)
    if info != 0:
        raise ValueError('illegal value in %dth argument of internal potrs'
                         % -info)
    return x


def cholesky_banded(ab, overwrite_ab=False, lower=False, check_finite=True):
    """
    Cholesky decompose a banded Hermitian positive-definite matrix

    The matrix a is stored in ab either in lower-diagonal or upper-
    diagonal ordered form::

        ab[u + i - j, j] == a[i,j]        (if upper form; i <= j)
        ab[    i - j, j] == a[i,j]        (if lower form; i >= j)

    Example of ab (shape of a is (6,6), u=2)::

        upper form:
        *   *   a02 a13 a24 a35
        *   a01 a12 a23 a34 a45
        a00 a11 a22 a33 a44 a55

        lower form:
        a00 a11 a22 a33 a44 a55
        a10 a21 a32 a43 a54 *
        a20 a31 a42 a53 *   *

    Parameters
    ----------
    ab : (u + 1, M) array_like
        Banded matrix
    overwrite_ab : bool, optional
        Discard data in ab (may enhance performance)
    lower : bool, optional
        Is the matrix in the lower form. (Default is upper form)
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    c : (u + 1, M) ndarray
        Cholesky factorization of a, in the same banded format as ab

    See Also
    --------
    cho_solve_banded :
        Solve a linear set equations, given the Cholesky factorization
        of a banded Hermitian.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import cholesky_banded
    >>> from numpy import allclose, zeros, diag
    >>> Ab = np.array([[0, 0, 1j, 2, 3j], [0, -1, -2, 3, 4], [9, 8, 7, 6, 9]])
    >>> A = np.diag(Ab[0,2:], k=2) + np.diag(Ab[1,1:], k=1)
    >>> A = A + A.conj().T + np.diag(Ab[2, :])
    >>> c = cholesky_banded(Ab)
    >>> C = np.diag(c[0, 2:], k=2) + np.diag(c[1, 1:], k=1) + np.diag(c[2, :])
    >>> np.allclose(C.conj().T @ C - A, np.zeros((5, 5)))
    True

    """
    if check_finite:
        ab = asarray_chkfinite(ab)
    else:
        ab = asarray(ab)

    # accommodate square empty matrices
    if ab.size == 0:
        dt = cholesky_banded(np.array([[0, 0], [1, 1]], dtype=ab.dtype)).dtype
        return empty_like(ab, dtype=dt)

    pbtrf, = get_lapack_funcs(('pbtrf',), (ab,))
    c, info = pbtrf(ab, lower=lower, overwrite_ab=overwrite_ab)
    if info > 0:
        raise LinAlgError("%d-th leading minor not positive definite" % info)
    if info < 0:
        raise ValueError('illegal value in %d-th argument of internal pbtrf'
                         % -info)
    return c


def cho_solve_banded(cb_and_lower, b, overwrite_b=False, check_finite=True):
    """
    Solve the linear equations ``A x = b``, given the Cholesky factorization of
    the banded Hermitian ``A``.

    Parameters
    ----------
    (cb, lower) : tuple, (ndarray, bool)
        `cb` is the Cholesky factorization of A, as given by cholesky_banded.
        `lower` must be the same value that was given to cholesky_banded.
    b : array_like
        Right-hand side
    overwrite_b : bool, optional
        If True, the function will overwrite the values in `b`.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    x : array
        The solution to the system A x = b

    See Also
    --------
    cholesky_banded : Cholesky factorization of a banded matrix

    Notes
    -----

    .. versionadded:: 0.8.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import cholesky_banded, cho_solve_banded
    >>> Ab = np.array([[0, 0, 1j, 2, 3j], [0, -1, -2, 3, 4], [9, 8, 7, 6, 9]])
    >>> A = np.diag(Ab[0,2:], k=2) + np.diag(Ab[1,1:], k=1)
    >>> A = A + A.conj().T + np.diag(Ab[2, :])
    >>> c = cholesky_banded(Ab)
    >>> x = cho_solve_banded((c, False), np.ones(5))
    >>> np.allclose(A @ x - np.ones(5), np.zeros(5))
    True

    """
    (cb, lower) = cb_and_lower
    if check_finite:
        cb = asarray_chkfinite(cb)
        b = asarray_chkfinite(b)
    else:
        cb = asarray(cb)
        b = asarray(b)

    # Validate shapes.
    if cb.shape[-1] != b.shape[0]:
        raise ValueError("shapes of cb and b are not compatible.")

    # accommodate empty arrays
    if b.size == 0:
        m = cholesky_banded(np.array([[0, 0], [1, 1]], dtype=cb.dtype))
        dt = cho_solve_banded((m, True), np.ones(2, dtype=b.dtype)).dtype
        return empty_like(b, dtype=dt)

    pbtrs, = get_lapack_funcs(('pbtrs',), (cb, b))
    x, info = pbtrs(cb, b, lower=lower, overwrite_b=overwrite_b)
    if info > 0:
        raise LinAlgError("%dth leading minor not positive definite" % info)
    if info < 0:
        raise ValueError('illegal value in %dth argument of internal pbtrs'
                         % -info)
    return x
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.linalg` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'lu', 'lu_solve', 'lu_factor',
    'LinAlgWarning', 'get_lapack_funcs',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="linalg", module="decomp_lu",
                                   private_modules=["_decomp_lu"], all=__all__,
                                   attribute=name)
<|endoftext|>"""
Low-level BLAS functions (:mod:`scipy.linalg.blas`)
===================================================

This module contains low-level functions from the BLAS library.

.. versionadded:: 0.12.0

.. note::

   The common ``overwrite_<>`` option in many routines, allows the
   input arrays to be overwritten to avoid extra memory allocation.
   However this requires the array to satisfy two conditions
   which are memory order and the data type to match exactly the
   order and the type expected by the routine.

   As an example, if you pass a double precision float array to any
   ``S....`` routine which expects single precision arguments, f2py
   will create an intermediate array to match the argument types and
   overwriting will be performed on that intermediate array.

   Similarly, if a C-contiguous array is passed, f2py will pass a
   FORTRAN-contiguous array internally. Please make sure that these
   details are satisfied. More information can be found in the f2py
   documentation.

.. warning::

   These functions do little to no error checking.
   It is possible to cause crashes by mis-using them,
   so prefer using the higher-level routines in `scipy.linalg`.

Finding functions
-----------------

.. autosummary::
   :toctree: generated/

   get_blas_funcs
   find_best_blas_type

BLAS Level 1 functions
----------------------

.. autosummary::
   :toctree: generated/

   caxpy
   ccopy
   cdotc
   cdotu
   crotg
   cscal
   csrot
   csscal
   cswap
   dasum
   daxpy
   dcopy
   ddot
   dnrm2
   drot
   drotg
   drotm
   drotmg
   dscal
   dswap
   dzasum
   dznrm2
   icamax
   idamax
   isamax
   izamax
   sasum
   saxpy
   scasum
   scnrm2
   scopy
   sdot
   snrm2
   srot
   srotg
   srotm
   srotmg
   sscal
   sswap
   zaxpy
   zcopy
   zdotc
   zdotu
   zdrot
   zdscal
   zrotg
   zscal
   zswap

BLAS Level 2 functions
----------------------

.. autosummary::
   :toctree: generated/

   sgbmv
   sgemv
   sger
   ssbmv
   sspr
   sspr2
   ssymv
   ssyr
   ssyr2
   stbmv
   stpsv
   strmv
   strsv
   dgbmv
   dgemv
   dger
   dsbmv
   dspr
   dspr2
   dsymv
   dsyr
   dsyr2
   dtbmv
   dtpsv
   dtrmv
   dtrsv
   cgbmv
   cgemv
   cgerc
   cgeru
   chbmv
   chemv
   cher
   cher2
   chpmv
   chpr
   chpr2
   ctbmv
   ctbsv
   ctpmv
   ctpsv
   ctrmv
   ctrsv
   csyr
   zgbmv
   zgemv
   zgerc
   zgeru
   zhbmv
   zhemv
   zher
   zher2
   zhpmv
   zhpr
   zhpr2
   ztbmv
   ztbsv
   ztpmv
   ztrmv
   ztrsv
   zsyr

BLAS Level 3 functions
----------------------

.. autosummary::
   :toctree: generated/

   sgemm
   ssymm
   ssyr2k
   ssyrk
   strmm
   strsm
   dgemm
   dsymm
   dsyr2k
   dsyrk
   dtrmm
   dtrsm
   cgemm
   chemm
   cher2k
   cherk
   csymm
   csyr2k
   csyrk
   ctrmm
   ctrsm
   zgemm
   zhemm
   zher2k
   zherk
   zsymm
   zsyr2k
   zsyrk
   ztrmm
   ztrsm

"""
#
# Author: Pearu Peterson, March 2002
#         refactoring by Fabian Pedregosa, March 2010
#

__all__ = ['get_blas_funcs', 'find_best_blas_type']

import numpy as np
import functools

from scipy.linalg import _fblas
try:
    from scipy.linalg import _cblas
except ImportError:
    _cblas = None

try:
    from scipy.linalg import _fblas_64
    HAS_ILP64 = True
except ImportError:
    HAS_ILP64 = False
    _fblas_64 = None

# Expose all functions (only fblas --- cblas is an implementation detail)
empty_module = None
from scipy.linalg._fblas import *  # noqa: E402, F403
del empty_module

# all numeric dtypes '?bBhHiIlLqQefdgFDGO' that are safe to be converted to

# single precision float   : '?bBhH!!!!!!ef!!!!!!'
# double precision float   : '?bBhHiIlLqQefdg!!!!'
# single precision complex : '?bBhH!!!!!!ef!!F!!!'
# double precision complex : '?bBhHiIlLqQefdgFDG!'

_type_score = {x: 1 for x in '?bBhHef'}
_type_score.update({x: 2 for x in 'iIlLqQd'})

# Handle float128(g) and complex256(G) separately in case non-Windows systems.
# On Windows, the values will be rewritten to the same key with the same value.
_type_score.update({'F': 3, 'D': 4, 'g': 2, 'G': 4})

# Final mapping to the actual prefixes and dtypes
_type_conv = {1: ('s', np.dtype('float32')),
              2: ('d', np.dtype('float64')),
              3: ('c', np.dtype('complex64')),
              4: ('z', np.dtype('complex128'))}

# some convenience alias for complex functions
_blas_alias = {'cnrm2': 'scnrm2', 'znrm2': 'dznrm2',
               'cdot': 'cdotc', 'zdot': 'zdotc',
               'cger': 'cgerc', 'zger': 'zgerc',
               'sdotc': 'sdot', 'sdotu': 'sdot',
               'ddotc': 'ddot', 'ddotu': 'ddot'}


def find_best_blas_type(arrays=(), dtype=None):
    """Find best-matching BLAS/LAPACK type.

    Arrays are used to determine the optimal prefix of BLAS routines.

    Parameters
    ----------
    arrays : sequence of ndarrays, optional
        Arrays can be given to determine optimal prefix of BLAS
        routines. If not given, double-precision routines will be
        used, otherwise the most generic type in arrays will be used.
    dtype : str or dtype, optional
        Data-type specifier. Not used if `arrays` is non-empty.

    Returns
    -------
    prefix : str
        BLAS/LAPACK prefix character.
    dtype : dtype
        Inferred Numpy data type.
    prefer_fortran : bool
        Whether to prefer Fortran order routines over C order.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.linalg.blas as bla
    >>> rng = np.random.default_rng()
    >>> a = rng.random((10,15))
    >>> b = np.asfortranarray(a)  # Change the memory layout order
    >>> bla.find_best_blas_type((a,))
    ('d', dtype('float64'), False)
    >>> bla.find_best_blas_type((a*1j,))
    ('z', dtype('complex128'), False)
    >>> bla.find_best_blas_type((b,))
    ('d', dtype('float64'), True)

    """
    dtype = np.dtype(dtype)
    max_score = _type_score.get(dtype.char, 5)
    prefer_fortran = False

    if arrays:
        # In most cases, single element is passed through, quicker route
        if len(arrays) == 1:
            max_score = _type_score.get(arrays[0].dtype.char, 5)
            prefer_fortran = arrays[0].flags['FORTRAN']
        else:
            # use the most generic type in arrays
            scores = [_type_score.get(x.dtype.char, 5) for x in arrays]
            max_score = max(scores)
            ind_max_score = scores.index(max_score)
            # safe upcasting for mix of float64 and complex64 --> prefix 'z'
            if max_score == 3 and (2 in scores):
                max_score = 4

            if arrays[ind_max_score].flags['FORTRAN']:
                # prefer Fortran for leading array with column major order
                prefer_fortran = True

    # Get the LAPACK prefix and the corresponding dtype if not fall back
    # to 'd' and double precision float.
    prefix, dtype = _type_conv.get(max_score, ('d', np.dtype('float64')))

    return prefix, dtype, prefer_fortran


def _get_funcs(names, arrays, dtype,
               lib_name, fmodule, cmodule,
               fmodule_name, cmodule_name, alias,
               ilp64=False):
    """
    Return available BLAS/LAPACK functions.

    Used also in lapack.py. See get_blas_funcs for docstring.
    """

    funcs = []
    unpack = False
    dtype = np.dtype(dtype)
    module1 = (cmodule, cmodule_name)
    module2 = (fmodule, fmodule_name)

    if isinstance(names, str):
        names = (names,)
        unpack = True

    prefix, dtype, prefer_fortran = find_best_blas_type(arrays, dtype)

    if prefer_fortran:
        module1, module2 = module2, module1

    for name in names:
        func_name = prefix + name
        func_name = alias.get(func_name, func_name)
        func = getattr(module1[0], func_name, None)
        module_name = module1[1]
        if func is None:
            func = getattr(module2[0], func_name, None)
            module_name = module2[1]
        if func is None:
            raise ValueError(
                f'{lib_name} function {func_name} could not be found')
        func.module_name, func.typecode = module_name, prefix
        func.dtype = dtype
        if not ilp64:
            func.int_dtype = np.dtype(np.intc)
        else:
            func.int_dtype = np.dtype(np.int64)
        func.prefix = prefix  # Backward compatibility
        funcs.append(func)

    if unpack:
        return funcs[0]
    else:
        return funcs


def _memoize_get_funcs(func):
    """
    Memoized fast path for _get_funcs instances
    """
    memo = {}
    func.memo = memo

    @functools.wraps(func)
    def getter(names, arrays=(), dtype=None, ilp64=False):
        key = (names, dtype, ilp64)
        for array in arrays:
            # cf. find_blas_funcs
            key += (array.dtype.char, array.flags.fortran)

        try:
            value = memo.get(key)
        except TypeError:
            # unhashable key etc.
            key = None
            value = None

        if value is not None:
            return value

        value = func(names, arrays, dtype, ilp64)

        if key is not None:
            memo[key] = value

        return value

    return getter


@_memoize_get_funcs
def get_blas_funcs(names, arrays=(), dtype=None, ilp64=False):
    """Return available BLAS function objects from names.

    Arrays are used to determine the optimal prefix of BLAS routines.

    Parameters
    ----------
    names : str or sequence of str
        Name(s) of BLAS functions without type prefix.

    arrays : sequence of ndarrays, optional
        Arrays can be given to determine optimal prefix of BLAS
        routines. If not given, double-precision routines will be
        used, otherwise the most generic type in arrays will be used.

    dtype : str or dtype, optional
        Data-type specifier. Not used if `arrays` is non-empty.

    ilp64 : {True, False, 'preferred'}, optional
        Whether to return ILP64 routine variant.
        Choosing 'preferred' returns ILP64 routine if available,
        and otherwise the 32-bit routine. Default: False

    Returns
    -------
    funcs : list
        List containing the found function(s).


    Notes
    -----
    This routine automatically chooses between Fortran/C
    interfaces. Fortran code is used whenever possible for arrays with
    column major order. In all other cases, C code is preferred.

    In BLAS, the naming convention is that all functions start with a
    type prefix, which depends on the type of the principal
    matrix. These can be one of {'s', 'd', 'c', 'z'} for the NumPy
    types {float32, float64, complex64, complex128} respectively.
    The code and the dtype are stored in attributes `typecode` and `dtype`
    of the returned functions.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.linalg as LA
    >>> rng = np.random.default_rng()
    >>> a = rng.random((3,2))
    >>> x_gemv = LA.get_blas_funcs('gemv', (a,))
    >>> x_gemv.typecode
    'd'
    >>> x_gemv = LA.get_blas_funcs('gemv',(a*1j,))
    >>> x_gemv.typecode
    'z'

    """
    if isinstance(ilp64, str):
        if ilp64 == 'preferred':
            ilp64 = HAS_ILP64
        else:
            raise ValueError("Invalid value for 'ilp64'")

    if not ilp64:
        return _get_funcs(names, arrays, dtype,
                          "BLAS", _fblas, _cblas, "fblas", "cblas",
                          _blas_alias, ilp64=False)
    else:
        if not HAS_ILP64:
            raise RuntimeError("BLAS ILP64 routine requested, but Scipy "
                               "compiled only with 32-bit BLAS")
        return _get_funcs(names, arrays, dtype,
                          "BLAS", _fblas_64, None, "fblas_64", None,
                          _blas_alias, ilp64=True)
<|endoftext|>"""
Matrix functions that use Pade approximation with inverse scaling and squaring.

"""
import warnings

import numpy as np

from scipy.linalg._matfuncs_sqrtm import SqrtmError, _sqrtm_triu
from scipy.linalg._decomp_schur import schur, rsf2csf
from scipy.linalg._matfuncs import funm
from scipy.linalg import svdvals, solve_triangular
from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse.linalg import onenormest
import scipy.special


class LogmRankWarning(UserWarning):
    pass


class LogmExactlySingularWarning(LogmRankWarning):
    pass


class LogmNearlySingularWarning(LogmRankWarning):
    pass


class LogmError(np.linalg.LinAlgError):
    pass


class FractionalMatrixPowerError(np.linalg.LinAlgError):
    pass


#TODO renovate or move this class when scipy operators are more mature
class _MatrixM1PowerOperator(LinearOperator):
    """
    A representation of the linear operator (A - I)^p.
    """

    def __init__(self, A, p):
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError('expected A to be like a square matrix')
        if p < 0 or p != int(p):
            raise ValueError('expected p to be a non-negative integer')
        self._A = A
        self._p = p
        self.ndim = A.ndim
        self.shape = A.shape

    def _matvec(self, x):
        for i in range(self._p):
            x = self._A.dot(x) - x
        return x

    def _rmatvec(self, x):
        for i in range(self._p):
            x = x.dot(self._A) - x
        return x

    def _matmat(self, X):
        for i in range(self._p):
            X = self._A.dot(X) - X
        return X

    def _adjoint(self):
        return _MatrixM1PowerOperator(self._A.T, self._p)


#TODO renovate or move this function when SciPy operators are more mature
def _onenormest_m1_power(A, p,
        t=2, itmax=5, compute_v=False, compute_w=False):
    """
    Efficiently estimate the 1-norm of (A - I)^p.

    Parameters
    ----------
    A : ndarray
        Matrix whose 1-norm of a power is to be computed.
    p : int
        Non-negative integer power.
    t : int, optional
        A positive parameter controlling the tradeoff between
        accuracy versus time and memory usage.
        Larger values take longer and use more memory
        but give more accurate output.
    itmax : int, optional
        Use at most this many iterations.
    compute_v : bool, optional
        Request a norm-maximizing linear operator input vector if True.
    compute_w : bool, optional
        Request a norm-maximizing linear operator output vector if True.

    Returns
    -------
    est : float
        An underestimate of the 1-norm of the sparse matrix.
    v : ndarray, optional
        The vector such that ||Av||_1 == est*||v||_1.
        It can be thought of as an input to the linear operator
        that gives an output with particularly large norm.
    w : ndarray, optional
        The vector Av which has relatively large 1-norm.
        It can be thought of as an output of the linear operator
        that is relatively large in norm compared to the input.

    """
    return onenormest(_MatrixM1PowerOperator(A, p),
            t=t, itmax=itmax, compute_v=compute_v, compute_w=compute_w)


def _unwindk(z):
    """
    Compute the scalar unwinding number.

    Uses Eq. (5.3) in [1]_, and should be equal to (z - log(exp(z)) / (2 pi i).
    Note that this definition differs in sign from the original definition
    in equations (5, 6) in [2]_.  The sign convention is justified in [3]_.

    Parameters
    ----------
    z : complex
        A complex number.

    Returns
    -------
    unwinding_number : integer
        The scalar unwinding number of z.

    References
    ----------
    .. [1] Nicholas J. Higham and Lijing lin (2011)
           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
           SIAM Journal on Matrix Analysis and Applications,
           32 (3). pp. 1056-1078. ISSN 0895-4798

    .. [2] Robert M. Corless and David J. Jeffrey,
           "The unwinding number." Newsletter ACM SIGSAM Bulletin
           Volume 30, Issue 2, June 1996, Pages 28-35.

    .. [3] Russell Bradford and Robert M. Corless and James H. Davenport and
           David J. Jeffrey and Stephen M. Watt,
           "Reasoning about the elementary functions of complex analysis"
           Annals of Mathematics and Artificial Intelligence,
           36: 303-318, 2002.

    """
    return int(np.ceil((z.imag - np.pi) / (2*np.pi)))


def _briggs_helper_function(a, k):
    """
    Computes r = a^(1 / (2^k)) - 1.

    This is algorithm (2) of [1]_.
    The purpose is to avoid a danger of subtractive cancellation.
    For more computational efficiency it should probably be cythonized.

    Parameters
    ----------
    a : complex
        A complex number.
    k : integer
        A nonnegative integer.

    Returns
    -------
    r : complex
        The value r = a^(1 / (2^k)) - 1 computed with less cancellation.

    Notes
    -----
    The algorithm as formulated in the reference does not handle k=0 or k=1
    correctly, so these are special-cased in this implementation.
    This function is intended to not allow `a` to belong to the closed
    negative real axis, but this constraint is relaxed.

    References
    ----------
    .. [1] Awad H. Al-Mohy (2012)
           "A more accurate Briggs method for the logarithm",
           Numerical Algorithms, 59 : 393--402.

    """
    if k < 0 or int(k) != k:
        raise ValueError('expected a nonnegative integer k')
    if k == 0:
        return a - 1
    elif k == 1:
        return np.sqrt(a) - 1
    else:
        k_hat = k
        if np.angle(a) >= np.pi / 2:
            a = np.sqrt(a)
            k_hat = k - 1
        z0 = a - 1
        a = np.sqrt(a)
        r = 1 + a
        for j in range(1, k_hat):
            a = np.sqrt(a)
            r = r * (1 + a)
        r = z0 / r
        return r


def _fractional_power_superdiag_entry(l1, l2, t12, p):
    """
    Compute a superdiagonal entry of a fractional matrix power.

    This is Eq. (5.6) in [1]_.

    Parameters
    ----------
    l1 : complex
        A diagonal entry of the matrix.
    l2 : complex
        A diagonal entry of the matrix.
    t12 : complex
        A superdiagonal entry of the matrix.
    p : float
        A fractional power.

    Returns
    -------
    f12 : complex
        A superdiagonal entry of the fractional matrix power.

    Notes
    -----
    Care has been taken to return a real number if possible when
    all of the inputs are real numbers.

    References
    ----------
    .. [1] Nicholas J. Higham and Lijing lin (2011)
           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
           SIAM Journal on Matrix Analysis and Applications,
           32 (3). pp. 1056-1078. ISSN 0895-4798

    """
    if l1 == l2:
        f12 = t12 * p * l1**(p-1)
    elif abs(l2 - l1) > abs(l1 + l2) / 2:
        f12 = t12 * ((l2**p) - (l1**p)) / (l2 - l1)
    else:
        # This is Eq. (5.5) in [1].
        z = (l2 - l1) / (l2 + l1)
        log_l1 = np.log(l1)
        log_l2 = np.log(l2)
        arctanh_z = np.arctanh(z)
        tmp_a = t12 * np.exp((p/2)*(log_l2 + log_l1))
        tmp_u = _unwindk(log_l2 - log_l1)
        if tmp_u:
            tmp_b = p * (arctanh_z + np.pi * 1j * tmp_u)
        else:
            tmp_b = p * arctanh_z
        tmp_c = 2 * np.sinh(tmp_b) / (l2 - l1)
        f12 = tmp_a * tmp_c
    return f12


def _logm_superdiag_entry(l1, l2, t12):
    """
    Compute a superdiagonal entry of a matrix logarithm.

    This is like Eq. (11.28) in [1]_, except the determination of whether
    l1 and l2 are sufficiently far apart has been modified.

    Parameters
    ----------
    l1 : complex
        A diagonal entry of the matrix.
    l2 : complex
        A diagonal entry of the matrix.
    t12 : complex
        A superdiagonal entry of the matrix.

    Returns
    -------
    f12 : complex
        A superdiagonal entry of the matrix logarithm.

    Notes
    -----
    Care has been taken to return a real number if possible when
    all of the inputs are real numbers.

    References
    ----------
    .. [1] Nicholas J. Higham (2008)
           "Functions of Matrices: Theory and Computation"
           ISBN 978-0-898716-46-7

    """
    if l1 == l2:
        f12 = t12 / l1
    elif abs(l2 - l1) > abs(l1 + l2) / 2:
        f12 = t12 * (np.log(l2) - np.log(l1)) / (l2 - l1)
    else:
        z = (l2 - l1) / (l2 + l1)
        u = _unwindk(np.log(l2) - np.log(l1))
        if u:
            f12 = t12 * 2 * (np.arctanh(z) + np.pi*1j*u) / (l2 - l1)
        else:
            f12 = t12 * 2 * np.arctanh(z) / (l2 - l1)
    return f12


def _inverse_squaring_helper(T0, theta):
    """
    A helper function for inverse scaling and squaring for Pade approximation.

    Parameters
    ----------
    T0 : (N, N) array_like upper triangular
        Matrix involved in inverse scaling and squaring.
    theta : indexable
        The values theta[1] .. theta[7] must be available.
        They represent bounds related to Pade approximation, and they depend
        on the matrix function which is being computed.
        For example, different values of theta are required for
        matrix logarithm than for fractional matrix power.

    Returns
    -------
    R : (N, N) array_like upper triangular
        Composition of zero or more matrix square roots of T0, minus I.
    s : non-negative integer
        Number of square roots taken.
    m : positive integer
        The degree of the Pade approximation.

    Notes
    -----
    This subroutine appears as a chunk of lines within
    a couple of published algorithms; for example it appears
    as lines 4--35 in algorithm (3.1) of [1]_, and
    as lines 3--34 in algorithm (4.1) of [2]_.
    The instances of 'goto line 38' in algorithm (3.1) of [1]_
    probably mean 'goto line 36' and have been interpreted accordingly.

    References
    ----------
    .. [1] Nicholas J. Higham and Lijing Lin (2013)
           "An Improved Schur-Pade Algorithm for Fractional Powers
           of a Matrix and their Frechet Derivatives."

    .. [2] Awad H. Al-Mohy and Nicholas J. Higham (2012)
           "Improved Inverse Scaling and Squaring Algorithms
           for the Matrix Logarithm."
           SIAM Journal on Scientific Computing, 34 (4). C152-C169.
           ISSN 1095-7197

    """
    if len(T0.shape) != 2 or T0.shape[0] != T0.shape[1]:
        raise ValueError('expected an upper triangular square matrix')
    n, n = T0.shape
    T = T0

    # Find s0, the smallest s such that the spectral radius
    # of a certain diagonal matrix is at most theta[7].
    # Note that because theta[7] < 1,
    # this search will not terminate if any diagonal entry of T is zero.
    s0 = 0
    tmp_diag = np.diag(T)
    if np.count_nonzero(tmp_diag) != n:
        raise Exception('Diagonal entries of T must be nonzero')
    while np.max(np.absolute(tmp_diag - 1), initial=0.) > theta[7]:
        tmp_diag = np.sqrt(tmp_diag)
        s0 += 1

    # Take matrix square roots of T.
    for i in range(s0):
        T = _sqrtm_triu(T)

    # Flow control in this section is a little odd.
    # This is because I am translating algorithm descriptions
    # which have GOTOs in the publication.
    s = s0
    k = 0
    d2 = _onenormest_m1_power(T, 2) ** (1/2)
    d3 = _onenormest_m1_power(T, 3) ** (1/3)
    a2 = max(d2, d3)
    m = None
    for i in (1, 2):
        if a2 <= theta[i]:
            m = i
            break
    while m is None:
        if s > s0:
            d3 = _onenormest_m1_power(T, 3) ** (1/3)
        d4 = _onenormest_m1_power(T, 4) ** (1/4)
        a3 = max(d3, d4)
        if a3 <= theta[7]:
            j1 = min(i for i in (3, 4, 5, 6, 7) if a3 <= theta[i])
            if j1 <= 6:
                m = j1
                break
            elif a3 / 2 <= theta[5] and k < 2:
                k += 1
                T = _sqrtm_triu(T)
                s += 1
                continue
        d5 = _onenormest_m1_power(T, 5) ** (1/5)
        a4 = max(d4, d5)
        eta = min(a3, a4)
        for i in (6, 7):
            if eta <= theta[i]:
                m = i
                break
        if m is not None:
            break
        T = _sqrtm_triu(T)
        s += 1

    # The subtraction of the identity is redundant here,
    # because the diagonal will be replaced for improved numerical accuracy,
    # but this formulation should help clarify the meaning of R.
    R = T - np.identity(n)

    # Replace the diagonal and first superdiagonal of T0^(1/(2^s)) - I
    # using formulas that have less subtractive cancellation.
    # Skip this step if the principal branch
    # does not exist at T0; this happens when a diagonal entry of T0
    # is negative with imaginary part 0.
    has_principal_branch = all(x.real > 0 or x.imag != 0 for x in np.diag(T0))
    if has_principal_branch:
        for j in range(n):
            a = T0[j, j]
            r = _briggs_helper_function(a, s)
            R[j, j] = r
        p = np.exp2(-s)
        for j in range(n-1):
            l1 = T0[j, j]
            l2 = T0[j+1, j+1]
            t12 = T0[j, j+1]
            f12 = _fractional_power_superdiag_entry(l1, l2, t12, p)
            R[j, j+1] = f12

    # Return the T-I matrix, the number of square roots, and the Pade degree.
    if not np.array_equal(R, np.triu(R)):
        raise Exception('R is not upper triangular')
    return R, s, m


def _fractional_power_pade_constant(i, t):
    # A helper function for matrix fractional power.
    if i < 1:
        raise ValueError('expected a positive integer i')
    if not (-1 < t < 1):
        raise ValueError('expected -1 < t < 1')
    if i == 1:
        return -t
    elif i % 2 == 0:
        j = i // 2
        return (-j + t) / (2 * (2*j - 1))
    elif i % 2 == 1:
        j = (i - 1) // 2
        return (-j - t) / (2 * (2*j + 1))
    else:
        raise Exception(f'unnexpected value of i, i = {i}')


def _fractional_power_pade(R, t, m):
    """
    Evaluate the Pade approximation of a fractional matrix power.

    Evaluate the degree-m Pade approximation of R
    to the fractional matrix power t using the continued fraction
    in bottom-up fashion using algorithm (4.1) in [1]_.

    Parameters
    ----------
    R : (N, N) array_like
        Upper triangular matrix whose fractional power to evaluate.
    t : float
        Fractional power between -1 and 1 exclusive.
    m : positive integer
        Degree of Pade approximation.

    Returns
    -------
    U : (N, N) array_like
        The degree-m Pade approximation of R to the fractional power t.
        This matrix will be upper triangular.

    References
    ----------
    .. [1] Nicholas J. Higham and Lijing lin (2011)
           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
           SIAM Journal on Matrix Analysis and Applications,
           32 (3). pp. 1056-1078. ISSN 0895-4798

    """
    if m < 1 or int(m) != m:
        raise ValueError('expected a positive integer m')
    if not (-1 < t < 1):
        raise ValueError('expected -1 < t < 1')
    R = np.asarray(R)
    if len(R.shape) != 2 or R.shape[0] != R.shape[1]:
        raise ValueError('expected an upper triangular square matrix')
    n, n = R.shape
    ident = np.identity(n)
    Y = R * _fractional_power_pade_constant(2*m, t)
    for j in range(2*m - 1, 0, -1):
        rhs = R * _fractional_power_pade_constant(j, t)
        Y = solve_triangular(ident + Y, rhs)
    U = ident + Y
    if not np.array_equal(U, np.triu(U)):
        raise Exception('U is not upper triangular')
    return U


def _remainder_matrix_power_triu(T, t):
    """
    Compute a fractional power of an upper triangular matrix.

    The fractional power is restricted to fractions -1 < t < 1.
    This uses algorithm (3.1) of [1]_.
    The Pade approximation itself uses algorithm (4.1) of [2]_.

    Parameters
    ----------
    T : (N, N) array_like
        Upper triangular matrix whose fractional power to evaluate.
    t : float
        Fractional power between -1 and 1 exclusive.

    Returns
    -------
    X : (N, N) array_like
        The fractional power of the matrix.

    References
    ----------
    .. [1] Nicholas J. Higham and Lijing Lin (2013)
           "An Improved Schur-Pade Algorithm for Fractional Powers
           of a Matrix and their Frechet Derivatives."

    .. [2] Nicholas J. Higham and Lijing lin (2011)
           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
           SIAM Journal on Matrix Analysis and Applications,
           32 (3). pp. 1056-1078. ISSN 0895-4798

    """
    m_to_theta = {
            1: 1.51e-5,
            2: 2.24e-3,
            3: 1.88e-2,
            4: 6.04e-2,
            5: 1.24e-1,
            6: 2.00e-1,
            7: 2.79e-1,
            }
    n, n = T.shape
    T0 = T
    T0_diag = np.diag(T0)
    if np.array_equal(T0, np.diag(T0_diag)):
        U = np.diag(T0_diag ** t)
    else:
        R, s, m = _inverse_squaring_helper(T0, m_to_theta)

        # Evaluate the Pade approximation.
        # Note that this function expects the negative of the matrix
        # returned by the inverse squaring helper.
        U = _fractional_power_pade(-R, t, m)

        # Undo the inverse scaling and squaring.
        # Be less clever about this
        # if the principal branch does not exist at T0;
        # this happens when a diagonal entry of T0
        # is negative with imaginary part 0.
        eivals = np.diag(T0)
        has_principal_branch = all(x.real > 0 or x.imag != 0 for x in eivals)
        for i in range(s, -1, -1):
            if i < s:
                U = U.dot(U)
            else:
                if has_principal_branch:
                    p = t * np.exp2(-i)
                    U[np.diag_indices(n)] = T0_diag ** p
                    for j in range(n-1):
                        l1 = T0[j, j]
                        l2 = T0[j+1, j+1]
                        t12 = T0[j, j+1]
                        f12 = _fractional_power_superdiag_entry(l1, l2, t12, p)
                        U[j, j+1] = f12
    if not np.array_equal(U, np.triu(U)):
        raise Exception('U is not upper triangular')
    return U


def _remainder_matrix_power(A, t):
    """
    Compute the fractional power of a matrix, for fractions -1 < t < 1.

    This uses algorithm (3.1) of [1]_.
    The Pade approximation itself uses algorithm (4.1) of [2]_.

    Parameters
    ----------
    A : (N, N) array_like
        Matrix whose fractional power to evaluate.
    t : float
        Fractional power between -1 and 1 exclusive.

    Returns
    -------
    X : (N, N) array_like
        The fractional power of the matrix.

    References
    ----------
    .. [1] Nicholas J. Higham and Lijing Lin (2013)
           "An Improved Schur-Pade Algorithm for Fractional Powers
           of a Matrix and their Frechet Derivatives."

    .. [2] Nicholas J. Higham and Lijing lin (2011)
           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
           SIAM Journal on Matrix Analysis and Applications,
           32 (3). pp. 1056-1078. ISSN 0895-4798

    """
    # This code block is copied from numpy.matrix_power().
    A = np.asarray(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('input must be a square array')

    # Get the number of rows and columns.
    n, n = A.shape

    # Triangularize the matrix if necessary,
    # attempting to preserve dtype if possible.
    if np.array_equal(A, np.triu(A)):
        Z = None
        T = A
    else:
        if np.isrealobj(A):
            T, Z = schur(A)
            if not np.array_equal(T, np.triu(T)):
                T, Z = rsf2csf(T, Z)
        else:
            T, Z = schur(A, output='complex')

    # Zeros on the diagonal of the triangular matrix are forbidden,
    # because the inverse scaling and squaring cannot deal with it.
    T_diag = np.diag(T)
    if np.count_nonzero(T_diag) != n:
        raise FractionalMatrixPowerError(
                'cannot use inverse scaling and squaring to find '
                'the fractional matrix power of a singular matrix')

    # If the triangular matrix is real and has a negative
    # entry on the diagonal, then force the matrix to be complex.
    if np.isrealobj(T) and np.min(T_diag) < 0:
        T = T.astype(complex)

    # Get the fractional power of the triangular matrix,
    # and de-triangularize it if necessary.
    U = _remainder_matrix_power_triu(T, t)
    if Z is not None:
        ZH = np.conjugate(Z).T
        return Z.dot(U).dot(ZH)
    else:
        return U


def _fractional_matrix_power(A, p):
    """
    Compute the fractional power of a matrix.

    See the fractional_matrix_power docstring in matfuncs.py for more info.

    """
    A = np.asarray(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected a square matrix')
    if p == int(p):
        return np.linalg.matrix_power(A, int(p))
    # Compute singular values.
    s = svdvals(A)
    # Inverse scaling and squaring cannot deal with a singular matrix,
    # because the process of repeatedly taking square roots
    # would not converge to the identity matrix.
    if s[-1]:
        # Compute the condition number relative to matrix inversion,
        # and use this to decide between floor(p) and ceil(p).
        k2 = s[0] / s[-1]
        p1 = p - np.floor(p)
        p2 = p - np.ceil(p)
        if p1 * k2 ** (1 - p1) <= -p2 * k2:
            a = int(np.floor(p))
            b = p1
        else:
            a = int(np.ceil(p))
            b = p2
        try:
            R = _remainder_matrix_power(A, b)
            Q = np.linalg.matrix_power(A, a)
            return Q.dot(R)
        except np.linalg.LinAlgError:
            pass
    # If p is negative then we are going to give up.
    # If p is non-negative then we can fall back to generic funm.
    if p < 0:
        X = np.empty_like(A)
        X.fill(np.nan)
        return X
    else:
        p1 = p - np.floor(p)
        a = int(np.floor(p))
        b = p1
        R, info = funm(A, lambda x: pow(x, b), disp=False)
        Q = np.linalg.matrix_power(A, a)
        return Q.dot(R)


def _logm_triu(T):
    """
    Compute matrix logarithm of an upper triangular matrix.

    The matrix logarithm is the inverse of
    expm: expm(logm(`T`)) == `T`

    Parameters
    ----------
    T : (N, N) array_like
        Upper triangular matrix whose logarithm to evaluate

    Returns
    -------
    logm : (N, N) ndarray
        Matrix logarithm of `T`

    References
    ----------
    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2012)
           "Improved Inverse Scaling and Squaring Algorithms
           for the Matrix Logarithm."
           SIAM Journal on Scientific Computing, 34 (4). C152-C169.
           ISSN 1095-7197

    .. [2] Nicholas J. Higham (2008)
           "Functions of Matrices: Theory and Computation"
           ISBN 978-0-898716-46-7

    .. [3] Nicholas J. Higham and Lijing lin (2011)
           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
           SIAM Journal on Matrix Analysis and Applications,
           32 (3). pp. 1056-1078. ISSN 0895-4798

    """
    T = np.asarray(T)
    if len(T.shape) != 2 or T.shape[0] != T.shape[1]:
        raise ValueError('expected an upper triangular square matrix')
    n, n = T.shape

    # Construct T0 with the appropriate type,
    # depending on the dtype and the spectrum of T.
    T_diag = np.diag(T)
    keep_it_real = np.isrealobj(T) and np.min(T_diag, initial=0.) >= 0
    if keep_it_real:
        T0 = T
    else:
        T0 = T.astype(complex)

    # Define bounds given in Table (2.1).
    theta = (None,
            1.59e-5, 2.31e-3, 1.94e-2, 6.21e-2,
            1.28e-1, 2.06e-1, 2.88e-1, 3.67e-1,
            4.39e-1, 5.03e-1, 5.60e-1, 6.09e-1,
            6.52e-1, 6.89e-1, 7.21e-1, 7.49e-1)

    R, s, m = _inverse_squaring_helper(T0, theta)

    # Evaluate U = 2**s r_m(T - I) using the partial fraction expansion (1.1).
    # This requires the nodes and weights
    # corresponding to degree-m Gauss-Legendre quadrature.
    # These quadrature arrays need to be transformed from the [-1, 1] interval
    # to the [0, 1] interval.
    nodes, weights = scipy.special.p_roots(m)
    nodes = nodes.real
    if nodes.shape != (m,) or weights.shape != (m,):
        raise Exception('internal error')
    nodes = 0.5 + 0.5 * nodes
    weights = 0.5 * weights
    ident = np.identity(n)
    U = np.zeros_like(R)
    for alpha, beta in zip(weights, nodes):
        U += solve_triangular(ident + beta*R, alpha*R)
    U *= np.exp2(s)

    # Skip this step if the principal branch
    # does not exist at T0; this happens when a diagonal entry of T0
    # is negative with imaginary part 0.
    has_principal_branch = all(x.real > 0 or x.imag != 0 for x in np.diag(T0))
    if has_principal_branch:

        # Recompute diagonal entries of U.
        U[np.diag_indices(n)] = np.log(np.diag(T0))

        # Recompute superdiagonal entries of U.
        # This indexing of this code should be renovated
        # when newer np.diagonal() becomes available.
        for i in range(n-1):
            l1 = T0[i, i]
            l2 = T0[i+1, i+1]
            t12 = T0[i, i+1]
            U[i, i+1] = _logm_superdiag_entry(l1, l2, t12)

    # Return the logm of the upper triangular matrix.
    if not np.array_equal(U, np.triu(U)):
        raise Exception('U is not upper triangular')
    return U


def _logm_force_nonsingular_triangular_matrix(T, inplace=False):
    # The input matrix should be upper triangular.
    # The eps is ad hoc and is not meant to be machine precision.
    tri_eps = 1e-20
    abs_diag = np.absolute(np.diag(T))
    if np.any(abs_diag == 0):
        exact_singularity_msg = 'The logm input matrix is exactly singular.'
        warnings.warn(exact_singularity_msg, LogmExactlySingularWarning, stacklevel=3)
        if not inplace:
            T = T.copy()
        n = T.shape[0]
        for i in range(n):
            if not T[i, i]:
                T[i, i] = tri_eps
    elif np.any(abs_diag < tri_eps):
        near_singularity_msg = 'The logm input matrix may be nearly singular.'
        warnings.warn(near_singularity_msg, LogmNearlySingularWarning, stacklevel=3)
    return T


def _logm(A):
    """
    Compute the matrix logarithm.

    See the logm docstring in matfuncs.py for more info.

    Notes
    -----
    In this function we look at triangular matrices that are similar
    to the input matrix. If any diagonal entry of such a triangular matrix
    is exactly zero then the original matrix is singular.
    The matrix logarithm does not exist for such matrices,
    but in such cases we will pretend that the diagonal entries that are zero
    are actually slightly positive by an ad-hoc amount, in the interest
    of returning something more useful than NaN. This will cause a warning.

    """
    A = np.asarray(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected a square matrix')

    # If the input matrix dtype is integer then copy to a float dtype matrix.
    if issubclass(A.dtype.type, np.integer):
        A = np.asarray(A, dtype=float)

    keep_it_real = np.isrealobj(A)
    try:
        if np.array_equal(A, np.triu(A)):
            A = _logm_force_nonsingular_triangular_matrix(A)
            if np.min(np.diag(A), initial=0.) < 0:
                A = A.astype(complex)
            return _logm_triu(A)
        else:
            if keep_it_real:
                T, Z = schur(A)
                if not np.array_equal(T, np.triu(T)):
                    T, Z = rsf2csf(T, Z)
            else:
                T, Z = schur(A, output='complex')
            T = _logm_force_nonsingular_triangular_matrix(T, inplace=True)
            U = _logm_triu(T)
            ZH = np.conjugate(Z).T
            return Z.dot(U).dot(ZH)
    except (SqrtmError, LogmError):
        X = np.empty_like(A)
        X.fill(np.nan)
        return X
<|endoftext|>"""Matrix equation solver routines"""
# Author: Jeffrey Armstrong <jeff@approximatrix.com>
# February 24, 2012

# Modified: Chad Fulton <ChadFulton@gmail.com>
# June 19, 2014

# Modified: Ilhan Polat <ilhanpolat@gmail.com>
# September 13, 2016

import warnings
import numpy as np
from numpy.linalg import inv, LinAlgError, norm, cond, svd

from ._basic import solve, solve_triangular, matrix_balance
from .lapack import get_lapack_funcs
from ._decomp_schur import schur
from ._decomp_lu import lu
from ._decomp_qr import qr
from ._decomp_qz import ordqz
from ._decomp import _asarray_validated
from ._special_matrices import block_diag

__all__ = ['solve_sylvester',
           'solve_continuous_lyapunov', 'solve_discrete_lyapunov',
           'solve_lyapunov',
           'solve_continuous_are', 'solve_discrete_are']


def solve_sylvester(a, b, q):
    """
    Computes a solution (X) to the Sylvester equation :math:`AX + XB = Q`.

    Parameters
    ----------
    a : (M, M) array_like
        Leading matrix of the Sylvester equation
    b : (N, N) array_like
        Trailing matrix of the Sylvester equation
    q : (M, N) array_like
        Right-hand side

    Returns
    -------
    x : (M, N) ndarray
        The solution to the Sylvester equation.

    Raises
    ------
    LinAlgError
        If solution was not found

    Notes
    -----
    Computes a solution to the Sylvester matrix equation via the Bartels-
    Stewart algorithm. The A and B matrices first undergo Schur
    decompositions. The resulting matrices are used to construct an
    alternative Sylvester equation (``RY + YS^T = F``) where the R and S
    matrices are in quasi-triangular form (or, when R, S or F are complex,
    triangular form). The simplified equation is then solved using
    ``*TRSYL`` from LAPACK directly.

    .. versionadded:: 0.11.0

    Examples
    --------
    Given `a`, `b`, and `q` solve for `x`:

    >>> import numpy as np
    >>> from scipy import linalg
    >>> a = np.array([[-3, -2, 0], [-1, -1, 3], [3, -5, -1]])
    >>> b = np.array([[1]])
    >>> q = np.array([[1],[2],[3]])
    >>> x = linalg.solve_sylvester(a, b, q)
    >>> x
    array([[ 0.0625],
           [-0.5625],
           [ 0.6875]])
    >>> np.allclose(a.dot(x) + x.dot(b), q)
    True

    """
    # Accommodate empty a
    if a.size == 0 or b.size == 0:
        tdict = {'s': np.float32, 'd': np.float64,
                 'c': np.complex64, 'z': np.complex128}
        func, = get_lapack_funcs(('trsyl',), arrays=(a, b, q))
        return np.empty(q.shape, dtype=tdict[func.typecode])

    # Compute the Schur decomposition form of a
    r, u = schur(a, output='real')

    # Compute the Schur decomposition of b
    s, v = schur(b.conj().transpose(), output='real')

    # Construct f = u'*q*v
    f = np.dot(np.dot(u.conj().transpose(), q), v)

    # Call the Sylvester equation solver
    trsyl, = get_lapack_funcs(('trsyl',), (r, s, f))
    if trsyl is None:
        raise RuntimeError('LAPACK implementation does not contain a proper '
                           'Sylvester equation solver (TRSYL)')
    y, scale, info = trsyl(r, s, f, tranb='C')

    y = scale*y

    if info < 0:
