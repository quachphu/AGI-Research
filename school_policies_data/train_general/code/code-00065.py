           [-4.5, -2. ]])

    """
    c = np.array(c, ndmin=1, copy=None)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
    if isinstance(x, np.ndarray) and tensor:
        c = c.reshape(c.shape + (1,)*x.ndim)

    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - (c1*(nd - 1))/nd
            c1 = tmp + (c1*((2*nd - 1) - x))/nd
    return c0 + c1*(1 - x)


def lagval2d(x, y, c):
    """
    Evaluate a 2-D Laguerre series at points (x, y).

    This function returns the values:

    .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * L_i(x) * L_j(y)

    The parameters `x` and `y` are converted to arrays only if they are
    tuples or a lists, otherwise they are treated as a scalars and they
    must have the same shape after conversion. In either case, either `x`
    and `y` or their elements must support multiplication and addition both
    with themselves and with the elements of `c`.

    If `c` is a 1-D array a one is implicitly appended to its shape to make
    it 2-D. The shape of the result will be c.shape[2:] + x.shape.

    Parameters
    ----------
    x, y : array_like, compatible objects
        The two dimensional series is evaluated at the points ``(x, y)``,
        where `x` and `y` must have the same shape. If `x` or `y` is a list
        or tuple, it is first converted to an ndarray, otherwise it is left
        unchanged and if it isn't an ndarray it is treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficient of the term
        of multi-degree i,j is contained in ``c[i,j]``. If `c` has
        dimension greater than two the remaining indices enumerate multiple
        sets of coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional polynomial at points formed with
        pairs of corresponding values from `x` and `y`.

    See Also
    --------
    lagval, laggrid2d, lagval3d, laggrid3d

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagval2d
    >>> c = [[1, 2],[3, 4]]
    >>> lagval2d(1, 1, c)
    1.0
    """
    return pu._valnd(lagval, c, x, y)


def laggrid2d(x, y, c):
    """
    Evaluate a 2-D Laguerre series on the Cartesian product of x and y.

    This function returns the values:

    .. math:: p(a,b) = \\sum_{i,j} c_{i,j} * L_i(a) * L_j(b)

    where the points ``(a, b)`` consist of all pairs formed by taking
    `a` from `x` and `b` from `y`. The resulting points form a grid with
    `x` in the first dimension and `y` in the second.

    The parameters `x` and `y` are converted to arrays only if they are
    tuples or a lists, otherwise they are treated as a scalars. In either
    case, either `x` and `y` or their elements must support multiplication
    and addition both with themselves and with the elements of `c`.

    If `c` has fewer than two dimensions, ones are implicitly appended to
    its shape to make it 2-D. The shape of the result will be c.shape[2:] +
    x.shape + y.shape.

    Parameters
    ----------
    x, y : array_like, compatible objects
        The two dimensional series is evaluated at the points in the
        Cartesian product of `x` and `y`.  If `x` or `y` is a list or
        tuple, it is first converted to an ndarray, otherwise it is left
        unchanged and, if it isn't an ndarray, it is treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficient of the term of
        multi-degree i,j is contained in ``c[i,j]``. If `c` has dimension
        greater than two the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional Chebyshev series at points in the
        Cartesian product of `x` and `y`.

    See Also
    --------
    lagval, lagval2d, lagval3d, laggrid3d

    Examples
    --------
    >>> from numpy.polynomial.laguerre import laggrid2d
    >>> c = [[1, 2], [3, 4]]
    >>> laggrid2d([0, 1], [0, 1], c)
    array([[10.,  4.],
           [ 3.,  1.]])

    """
    return pu._gridnd(lagval, c, x, y)


def lagval3d(x, y, z, c):
    """
    Evaluate a 3-D Laguerre series at points (x, y, z).

    This function returns the values:

    .. math:: p(x,y,z) = \\sum_{i,j,k} c_{i,j,k} * L_i(x) * L_j(y) * L_k(z)

    The parameters `x`, `y`, and `z` are converted to arrays only if
    they are tuples or a lists, otherwise they are treated as a scalars and
    they must have the same shape after conversion. In either case, either
    `x`, `y`, and `z` or their elements must support multiplication and
    addition both with themselves and with the elements of `c`.

    If `c` has fewer than 3 dimensions, ones are implicitly appended to its
    shape to make it 3-D. The shape of the result will be c.shape[3:] +
    x.shape.

    Parameters
    ----------
    x, y, z : array_like, compatible object
        The three dimensional series is evaluated at the points
        ``(x, y, z)``, where `x`, `y`, and `z` must have the same shape.  If
        any of `x`, `y`, or `z` is a list or tuple, it is first converted
        to an ndarray, otherwise it is left unchanged and if it isn't an
        ndarray it is  treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficient of the term of
        multi-degree i,j,k is contained in ``c[i,j,k]``. If `c` has dimension
        greater than 3 the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the multidimensional polynomial on points formed with
        triples of corresponding values from `x`, `y`, and `z`.

    See Also
    --------
    lagval, lagval2d, laggrid2d, laggrid3d

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagval3d
    >>> c = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    >>> lagval3d(1, 1, 2, c)
    -1.0

    """
    return pu._valnd(lagval, c, x, y, z)


def laggrid3d(x, y, z, c):
    """
    Evaluate a 3-D Laguerre series on the Cartesian product of x, y, and z.

    This function returns the values:

    .. math:: p(a,b,c) = \\sum_{i,j,k} c_{i,j,k} * L_i(a) * L_j(b) * L_k(c)

    where the points ``(a, b, c)`` consist of all triples formed by taking
    `a` from `x`, `b` from `y`, and `c` from `z`. The resulting points form
    a grid with `x` in the first dimension, `y` in the second, and `z` in
    the third.

    The parameters `x`, `y`, and `z` are converted to arrays only if they
    are tuples or a lists, otherwise they are treated as a scalars. In
    either case, either `x`, `y`, and `z` or their elements must support
    multiplication and addition both with themselves and with the elements
    of `c`.

    If `c` has fewer than three dimensions, ones are implicitly appended to
    its shape to make it 3-D. The shape of the result will be c.shape[3:] +
    x.shape + y.shape + z.shape.

    Parameters
    ----------
    x, y, z : array_like, compatible objects
        The three dimensional series is evaluated at the points in the
        Cartesian product of `x`, `y`, and `z`.  If `x`, `y`, or `z` is a
        list or tuple, it is first converted to an ndarray, otherwise it is
        left unchanged and, if it isn't an ndarray, it is treated as a
        scalar.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree i,j are contained in ``c[i,j]``. If `c` has dimension
        greater than two the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional polynomial at points in the Cartesian
        product of `x` and `y`.

    See Also
    --------
    lagval, lagval2d, laggrid2d, lagval3d

    Examples
    --------
    >>> from numpy.polynomial.laguerre import laggrid3d
    >>> c = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    >>> laggrid3d([0, 1], [0, 1], [2, 4], c)
    array([[[ -4., -44.],
            [ -2., -18.]],
           [[ -2., -14.],
            [ -1.,  -5.]]])

    """
    return pu._gridnd(lagval, c, x, y, z)


def lagvander(x, deg):
    """Pseudo-Vandermonde matrix of given degree.

    Returns the pseudo-Vandermonde matrix of degree `deg` and sample points
    `x`. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., i] = L_i(x)

    where ``0 <= i <= deg``. The leading indices of `V` index the elements of
    `x` and the last index is the degree of the Laguerre polynomial.

    If `c` is a 1-D array of coefficients of length ``n + 1`` and `V` is the
    array ``V = lagvander(x, n)``, then ``np.dot(V, c)`` and
    ``lagval(x, c)`` are the same up to roundoff. This equivalence is
    useful both for least squares fitting and for the evaluation of a large
    number of Laguerre series of the same degree and sample points.

    Parameters
    ----------
    x : array_like
        Array of points. The dtype is converted to float64 or complex128
        depending on whether any of the elements are complex. If `x` is
        scalar it is converted to a 1-D array.
    deg : int
        Degree of the resulting matrix.

    Returns
    -------
    vander : ndarray
        The pseudo-Vandermonde matrix. The shape of the returned matrix is
        ``x.shape + (deg + 1,)``, where The last index is the degree of the
        corresponding Laguerre polynomial.  The dtype will be the same as
        the converted `x`.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial.laguerre import lagvander
    >>> x = np.array([0, 1, 2])
    >>> lagvander(x, 3)
    array([[ 1.        ,  1.        ,  1.        ,  1.        ],
           [ 1.        ,  0.        , -0.5       , -0.66666667],
           [ 1.        , -1.        , -1.        , -0.33333333]])

    """
    ideg = pu._as_int(deg, "deg")
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = np.array(x, copy=None, ndmin=1) + 0.0
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype
    v = np.empty(dims, dtype=dtyp)
    v[0] = x*0 + 1
    if ideg > 0:
        v[1] = 1 - x
        for i in range(2, ideg + 1):
            v[i] = (v[i-1]*(2*i - 1 - x) - v[i-2]*(i - 1))/i
    return np.moveaxis(v, 0, -1)


def lagvander2d(x, y, deg):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points ``(x, y)``. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (deg[1] + 1)*i + j] = L_i(x) * L_j(y),

    where ``0 <= i <= deg[0]`` and ``0 <= j <= deg[1]``. The leading indices of
    `V` index the points ``(x, y)`` and the last index encodes the degrees of
    the Laguerre polynomials.

    If ``V = lagvander2d(x, y, [xdeg, ydeg])``, then the columns of `V`
    correspond to the elements of a 2-D coefficient array `c` of shape
    (xdeg + 1, ydeg + 1) in the order

    .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...

    and ``np.dot(V, c.flat)`` and ``lagval2d(x, y, c)`` will be the same
    up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 2-D Laguerre
    series of the same degrees and sample points.

    Parameters
    ----------
    x, y : array_like
        Arrays of point coordinates, all of the same shape. The dtypes
        will be converted to either float64 or complex128 depending on
        whether any of the elements are complex. Scalars are converted to
        1-D arrays.
    deg : list of ints
        List of maximum degrees of the form [x_deg, y_deg].

    Returns
    -------
    vander2d : ndarray
        The shape of the returned matrix is ``x.shape + (order,)``, where
        :math:`order = (deg[0]+1)*(deg[1]+1)`.  The dtype will be the same
        as the converted `x` and `y`.

    See Also
    --------
    lagvander, lagvander3d, lagval2d, lagval3d

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial.laguerre import lagvander2d
    >>> x = np.array([0])
    >>> y = np.array([2])
    >>> lagvander2d(x, y, [2, 1])
    array([[ 1., -1.,  1., -1.,  1., -1.]])

    """
    return pu._vander_nd_flat((lagvander, lagvander), (x, y), deg)


def lagvander3d(x, y, z, deg):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points ``(x, y, z)``. If `l`, `m`, `n` are the given degrees in `x`, `y`, `z`,
    then The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = L_i(x)*L_j(y)*L_k(z),

    where ``0 <= i <= l``, ``0 <= j <= m``, and ``0 <= j <= n``.  The leading
    indices of `V` index the points ``(x, y, z)`` and the last index encodes
    the degrees of the Laguerre polynomials.

    If ``V = lagvander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns
    of `V` correspond to the elements of a 3-D coefficient array `c` of
    shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order

    .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...

    and  ``np.dot(V, c.flat)`` and ``lagval3d(x, y, z, c)`` will be the
    same up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 3-D Laguerre
    series of the same degrees and sample points.

    Parameters
    ----------
    x, y, z : array_like
        Arrays of point coordinates, all of the same shape. The dtypes will
        be converted to either float64 or complex128 depending on whether
        any of the elements are complex. Scalars are converted to 1-D
        arrays.
    deg : list of ints
        List of maximum degrees of the form [x_deg, y_deg, z_deg].

    Returns
    -------
    vander3d : ndarray
        The shape of the returned matrix is ``x.shape + (order,)``, where
        :math:`order = (deg[0]+1)*(deg[1]+1)*(deg[2]+1)`.  The dtype will
        be the same as the converted `x`, `y`, and `z`.

    See Also
    --------
    lagvander, lagvander3d, lagval2d, lagval3d

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial.laguerre import lagvander3d
    >>> x = np.array([0])
    >>> y = np.array([2])
    >>> z = np.array([0])
    >>> lagvander3d(x, y, z, [2, 1, 3])
    array([[ 1.,  1.,  1.,  1., -1., -1., -1., -1.,  1.,  1.,  1.,  1., -1.,
            -1., -1., -1.,  1.,  1.,  1.,  1., -1., -1., -1., -1.]])

    """
    return pu._vander_nd_flat((lagvander, lagvander, lagvander), (x, y, z), deg)


def lagfit(x, y, deg, rcond=None, full=False, w=None):
    """
    Least squares fit of Laguerre series to data.

    Return the coefficients of a Laguerre series of degree `deg` that is the
    least squares fit to the data values `y` given at points `x`. If `y` is
    1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple
    fits are done, one for each column of `y`, and the resulting
    coefficients are stored in the corresponding columns of a 2-D return.
    The fitted polynomial(s) are in the form

    .. math::  p(x) = c_0 + c_1 * L_1(x) + ... + c_n * L_n(x),

    where ``n`` is `deg`.

    Parameters
    ----------
    x : array_like, shape (M,)
        x-coordinates of the M sample points ``(x[i], y[i])``.
    y : array_like, shape (M,) or (M, K)
        y-coordinates of the sample points. Several data sets of sample
        points sharing the same x-coordinates can be fitted at once by
        passing in a 2D-array that contains one dataset per column.
    deg : int or 1-D array_like
        Degree(s) of the fitting polynomials. If `deg` is a single integer
        all terms up to and including the `deg`'th term are included in the
        fit. For NumPy versions >= 1.11.0 a list of integers specifying the
        degrees of the terms to include may be used instead.
    rcond : float, optional
        Relative condition number of the fit. Singular values smaller than
        this relative to the largest singular value will be ignored. The
        default value is len(x)*eps, where eps is the relative precision of
        the float type, about 2e-16 in most cases.
    full : bool, optional
        Switch determining nature of return value. When it is False (the
        default) just the coefficients are returned, when True diagnostic
        information from the singular value decomposition is also returned.
    w : array_like, shape (`M`,), optional
        Weights. If not None, the weight ``w[i]`` applies to the unsquared
        residual ``y[i] - y_hat[i]`` at ``x[i]``. Ideally the weights are
        chosen so that the errors of the products ``w[i]*y[i]`` all have the
        same variance.  When using inverse-variance weighting, use
        ``w[i] = 1/sigma(y[i])``.  The default value is None.

    Returns
    -------
    coef : ndarray, shape (M,) or (M, K)
        Laguerre coefficients ordered from low to high. If `y` was 2-D,
        the coefficients for the data in column *k*  of `y` are in column
        *k*.

    [residuals, rank, singular_values, rcond] : list
        These values are only returned if ``full == True``

        - residuals -- sum of squared residuals of the least squares fit
        - rank -- the numerical rank of the scaled Vandermonde matrix
        - singular_values -- singular values of the scaled Vandermonde matrix
        - rcond -- value of `rcond`.

        For more details, see `numpy.linalg.lstsq`.

    Warns
    -----
    RankWarning
        The rank of the coefficient matrix in the least-squares fit is
        deficient. The warning is only raised if ``full == False``.  The
        warnings can be turned off by

        >>> import warnings
        >>> warnings.simplefilter('ignore', np.exceptions.RankWarning)

    See Also
    --------
    numpy.polynomial.polynomial.polyfit
    numpy.polynomial.legendre.legfit
    numpy.polynomial.chebyshev.chebfit
    numpy.polynomial.hermite.hermfit
    numpy.polynomial.hermite_e.hermefit
    lagval : Evaluates a Laguerre series.
    lagvander : pseudo Vandermonde matrix of Laguerre series.
    lagweight : Laguerre weight function.
    numpy.linalg.lstsq : Computes a least-squares fit from the matrix.
    scipy.interpolate.UnivariateSpline : Computes spline fits.

    Notes
    -----
    The solution is the coefficients of the Laguerre series ``p`` that
    minimizes the sum of the weighted squared errors

    .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,

    where the :math:`w_j` are the weights. This problem is solved by
    setting up as the (typically) overdetermined matrix equation

    .. math:: V(x) * c = w * y,

    where ``V`` is the weighted pseudo Vandermonde matrix of `x`, ``c`` are the
    coefficients to be solved for, `w` are the weights, and `y` are the
    observed values.  This equation is then solved using the singular value
    decomposition of ``V``.

    If some of the singular values of `V` are so small that they are
    neglected, then a `~exceptions.RankWarning` will be issued. This means that
    the coefficient values may be poorly determined. Using a lower order fit
    will usually get rid of the warning.  The `rcond` parameter can also be
    set to a value smaller than its default, but the resulting fit may be
    spurious and have large contributions from roundoff error.

    Fits using Laguerre series are probably most useful when the data can
    be approximated by ``sqrt(w(x)) * p(x)``, where ``w(x)`` is the Laguerre
    weight. In that case the weight ``sqrt(w(x[i]))`` should be used
    together with data values ``y[i]/sqrt(w(x[i]))``. The weight function is
    available as `lagweight`.

    References
    ----------
    .. [1] Wikipedia, "Curve fitting",
           https://en.wikipedia.org/wiki/Curve_fitting

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial.laguerre import lagfit, lagval
    >>> x = np.linspace(0, 10)
    >>> rng = np.random.default_rng()
    >>> err = rng.normal(scale=1./10, size=len(x))
    >>> y = lagval(x, [1, 2, 3]) + err
    >>> lagfit(x, y, 2)
    array([1.00578369, 1.99417356, 2.99827656]) # may vary

    """
    return pu._fit(lagvander, x, y, deg, rcond, full, w)


def lagcompanion(c):
    """
    Return the companion matrix of c.

    The usual companion matrix of the Laguerre polynomials is already
    symmetric when `c` is a basis Laguerre polynomial, so no scaling is
    applied.

    Parameters
    ----------
    c : array_like
        1-D array of Laguerre series coefficients ordered from low to high
        degree.

    Returns
    -------
    mat : ndarray
        Companion matrix of dimensions (deg, deg).

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagcompanion
    >>> lagcompanion([1, 2, 3])
    array([[ 1.        , -0.33333333],
           [-1.        ,  4.33333333]])

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    if len(c) < 2:
        raise ValueError('Series must have maximum degree of at least 1.')
    if len(c) == 2:
        return np.array([[1 + c[0]/c[1]]])

    n = len(c) - 1
    mat = np.zeros((n, n), dtype=c.dtype)
    top = mat.reshape(-1)[1::n+1]
    mid = mat.reshape(-1)[0::n+1]
    bot = mat.reshape(-1)[n::n+1]
    top[...] = -np.arange(1, n)
    mid[...] = 2.*np.arange(n) + 1.
    bot[...] = top
    mat[:, -1] += (c[:-1]/c[-1])*n
    return mat


def lagroots(c):
    """
    Compute the roots of a Laguerre series.

    Return the roots (a.k.a. "zeros") of the polynomial

    .. math:: p(x) = \\sum_i c[i] * L_i(x).

    Parameters
    ----------
    c : 1-D array_like
        1-D array of coefficients.

    Returns
    -------
    out : ndarray
        Array of the roots of the series. If all the roots are real,
        then `out` is also real, otherwise it is complex.

    See Also
    --------
    numpy.polynomial.polynomial.polyroots
    numpy.polynomial.legendre.legroots
    numpy.polynomial.chebyshev.chebroots
    numpy.polynomial.hermite.hermroots
    numpy.polynomial.hermite_e.hermeroots

    Notes
    -----
    The root estimates are obtained as the eigenvalues of the companion
    matrix, Roots far from the origin of the complex plane may have large
    errors due to the numerical instability of the series for such
    values. Roots with multiplicity greater than 1 will also show larger
    errors as the value of the series near such points is relatively
    insensitive to errors in the roots. Isolated roots near the origin can
    be improved by a few iterations of Newton's method.

    The Laguerre series basis polynomials aren't powers of `x` so the
    results of this function may seem unintuitive.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagroots, lagfromroots
    >>> coef = lagfromroots([0, 1, 2])
    >>> coef
    array([  2.,  -8.,  12.,  -6.])
    >>> lagroots(coef)
    array([-4.4408921e-16,  1.0000000e+00,  2.0000000e+00])

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    if len(c) <= 1:
        return np.array([], dtype=c.dtype)
    if len(c) == 2:
        return np.array([1 + c[0]/c[1]])

    # rotated companion matrix reduces error
    m = lagcompanion(c)[::-1,::-1]
    r = la.eigvals(m)
    r.sort()
    return r


def laggauss(deg):
    """
    Gauss-Laguerre quadrature.

    Computes the sample points and weights for Gauss-Laguerre quadrature.
    These sample points and weights will correctly integrate polynomials of
    degree :math:`2*deg - 1` or less over the interval :math:`[0, \\inf]`
    with the weight function :math:`f(x) = \\exp(-x)`.

    Parameters
    ----------
    deg : int
        Number of sample points and weights. It must be >= 1.

    Returns
    -------
    x : ndarray
        1-D ndarray containing the sample points.
    y : ndarray
        1-D ndarray containing the weights.

    Notes
    -----
    The results have only been tested up to degree 100 higher degrees may
    be problematic. The weights are determined by using the fact that

    .. math:: w_k = c / (L'_n(x_k) * L_{n-1}(x_k))

    where :math:`c` is a constant independent of :math:`k` and :math:`x_k`
    is the k'th root of :math:`L_n`, and then scaling the results to get
    the right value when integrating 1.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import laggauss
    >>> laggauss(2)
    (array([0.58578644, 3.41421356]), array([0.85355339, 0.14644661]))

    """
    ideg = pu._as_int(deg, "deg")
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    # first approximation of roots. We use the fact that the companion
    # matrix is symmetric in this case in order to obtain better zeros.
    c = np.array([0]*deg + [1])
    m = lagcompanion(c)
    x = la.eigvalsh(m)

    # improve roots by one application of Newton
    dy = lagval(x, c)
    df = lagval(x, lagder(c))
    x -= dy/df

    # compute the weights. We scale the factor to avoid possible numerical
    # overflow.
    fm = lagval(x, c[1:])
    fm /= np.abs(fm).max()
    df /= np.abs(df).max()
    w = 1/(fm * df)

    # scale w to get the right value, 1 in this case
    w /= w.sum()

    return x, w


def lagweight(x):
    """Weight function of the Laguerre polynomials.

    The weight function is :math:`exp(-x)` and the interval of integration
    is :math:`[0, \\inf]`. The Laguerre polynomials are orthogonal, but not
    normalized, with respect to this weight function.

    Parameters
    ----------
    x : array_like
       Values at which the weight function will be computed.

    Returns
    -------
    w : ndarray
       The weight function at `x`.

    Examples
    --------
    >>> from numpy.polynomial.laguerre import lagweight
    >>> x = np.array([0, 1, 2])
    >>> lagweight(x)
    array([1.        , 0.36787944, 0.13533528])

    """
    w = np.exp(-x)
    return w

#
# Laguerre series class
#

class Laguerre(ABCPolyBase):
    """A Laguerre series class.

    The Laguerre class provides the standard Python numerical methods
    '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the
    attributes and methods listed below.

    Parameters
    ----------
    coef : array_like
        Laguerre coefficients in order of increasing degree, i.e,
        ``(1, 2, 3)`` gives ``1*L_0(x) + 2*L_1(X) + 3*L_2(x)``.
    domain : (2,) array_like, optional
        Domain to use. The interval ``[domain[0], domain[1]]`` is mapped
        to the interval ``[window[0], window[1]]`` by shifting and scaling.
        The default value is [0., 1.].
    window : (2,) array_like, optional
        Window, see `domain` for its use. The default value is [0., 1.].
    symbol : str, optional
        Symbol used to represent the independent variable in string
        representations of the polynomial expression, e.g. for printing.
        The symbol must be a valid Python identifier. Default value is 'x'.

        .. versionadded:: 1.24

    """
    # Virtual Functions
    _add = staticmethod(lagadd)
    _sub = staticmethod(lagsub)
    _mul = staticmethod(lagmul)
    _div = staticmethod(lagdiv)
    _pow = staticmethod(lagpow)
    _val = staticmethod(lagval)
    _int = staticmethod(lagint)
    _der = staticmethod(lagder)
    _fit = staticmethod(lagfit)
    _line = staticmethod(lagline)
    _roots = staticmethod(lagroots)
    _fromroots = staticmethod(lagfromroots)

    # Virtual properties
    domain = np.array(lagdomain)
    window = np.array(lagdomain)
    basis_name = 'L'
<|endoftext|>""":mod:`numpy.ma..mrecords`

Defines the equivalent of :class:`numpy.recarrays` for masked arrays,
where fields can be accessed as attributes.
Note that :class:`numpy.ma.MaskedArray` already supports structured datatypes
and the masking of individual fields.

.. moduleauthor:: Pierre Gerard-Marchant

"""
#  We should make sure that no field is called '_mask','mask','_fieldmask',
#  or whatever restricted keywords.  An idea would be to no bother in the
#  first place, and then rename the invalid fields with a trailing
#  underscore. Maybe we could just overload the parser function ?

import warnings

import numpy as np
import numpy.ma as ma


_byteorderconv = np._core.records._byteorderconv


_check_fill_value = ma.core._check_fill_value


__all__ = [
    'MaskedRecords', 'mrecarray', 'fromarrays', 'fromrecords',
    'fromtextfile', 'addfield',
]

reserved_fields = ['_data', '_mask', '_fieldmask', 'dtype']


def _checknames(descr, names=None):
    """
    Checks that field names ``descr`` are not reserved keywords.

    If this is the case, a default 'f%i' is substituted.  If the argument
    `names` is not None, updates the field names to valid names.

    """
    ndescr = len(descr)
    default_names = ['f%i' % i for i in range(ndescr)]
    if names is None:
        new_names = default_names
    else:
        if isinstance(names, (tuple, list)):
            new_names = names
        elif isinstance(names, str):
            new_names = names.split(',')
        else:
            raise NameError(f'illegal input names {names!r}')
        nnames = len(new_names)
        if nnames < ndescr:
            new_names += default_names[nnames:]
    ndescr = []
    for (n, d, t) in zip(new_names, default_names, descr.descr):
        if n in reserved_fields:
            if t[0] in reserved_fields:
                ndescr.append((d, t[1]))
            else:
                ndescr.append(t)
        else:
            ndescr.append((n, t[1]))
    return np.dtype(ndescr)


def _get_fieldmask(self):
    mdescr = [(n, '|b1') for n in self.dtype.names]
    fdmask = np.empty(self.shape, dtype=mdescr)
    fdmask.flat = tuple([False] * len(mdescr))
    return fdmask


class MaskedRecords(ma.MaskedArray):
    """

    Attributes
    ----------
    _data : recarray
        Underlying data, as a record array.
    _mask : boolean array
        Mask of the records. A record is masked when all its fields are
        masked.
    _fieldmask : boolean recarray
        Record array of booleans, setting the mask of each individual field
        of each record.
    _fill_value : record
        Filling values for each field.

    """

    def __new__(cls, shape, dtype=None, buf=None, offset=0, strides=None,
                formats=None, names=None, titles=None,
                byteorder=None, aligned=False,
                mask=ma.nomask, hard_mask=False, fill_value=None, keep_mask=True,
                copy=False,
                **options):

        self = np.recarray.__new__(cls, shape, dtype=dtype, buf=buf, offset=offset,
                                   strides=strides, formats=formats, names=names,
                                   titles=titles, byteorder=byteorder,
                                   aligned=aligned,)

        mdtype = ma.make_mask_descr(self.dtype)
        if mask is ma.nomask or not np.size(mask):
            if not keep_mask:
                self._mask = tuple([False] * len(mdtype))
        else:
            mask = np.array(mask, copy=copy)
            if mask.shape != self.shape:
                (nd, nm) = (self.size, mask.size)
                if nm == 1:
                    mask = np.resize(mask, self.shape)
                elif nm == nd:
                    mask = np.reshape(mask, self.shape)
                else:
                    msg = "Mask and data not compatible: data size is %i, " + \
                          "mask size is %i."
                    raise ma.MAError(msg % (nd, nm))
            if not keep_mask:
                self.__setmask__(mask)
                self._sharedmask = True
            else:
                if mask.dtype == mdtype:
                    _mask = mask
                else:
                    _mask = np.array([tuple([m] * len(mdtype)) for m in mask],
                                     dtype=mdtype)
                self._mask = _mask
        return self

    def __array_finalize__(self, obj):
        # Make sure we have a _fieldmask by default
        _mask = getattr(obj, '_mask', None)
        if _mask is None:
            objmask = getattr(obj, '_mask', ma.nomask)
            _dtype = np.ndarray.__getattribute__(self, 'dtype')
            if objmask is ma.nomask:
                _mask = ma.make_mask_none(self.shape, dtype=_dtype)
            else:
                mdescr = ma.make_mask_descr(_dtype)
                _mask = np.array([tuple([m] * len(mdescr)) for m in objmask],
                               dtype=mdescr).view(np.recarray)
        # Update some of the attributes
        _dict = self.__dict__
        _dict.update(_mask=_mask)
        self._update_from(obj)
        if _dict['_baseclass'] == np.ndarray:
            _dict['_baseclass'] = np.recarray
        return

    @property
    def _data(self):
        """
        Returns the data as a recarray.

        """
        return np.ndarray.view(self, np.recarray)

    @property
    def _fieldmask(self):
        """
        Alias to mask.

        """
        return self._mask

    def __len__(self):
        """
        Returns the length

        """
        # We have more than one record
        if self.ndim:
            return len(self._data)
        # We have only one record: return the nb of fields
        return len(self.dtype)

    def __getattribute__(self, attr):
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            # attr must be a fieldname
            pass
        fielddict = np.ndarray.__getattribute__(self, 'dtype').fields
        try:
            res = fielddict[attr][:2]
        except (TypeError, KeyError) as e:
            raise AttributeError(
                f'record array has no attribute {attr}') from e
        # So far, so good
        _localdict = np.ndarray.__getattribute__(self, '__dict__')
        _data = np.ndarray.view(self, _localdict['_baseclass'])
        obj = _data.getfield(*res)
        if obj.dtype.names is not None:
            raise NotImplementedError("MaskedRecords is currently limited to"
                                      "simple records.")
        # Get some special attributes
        # Reset the object's mask
        hasmasked = False
        _mask = _localdict.get('_mask', None)
        if _mask is not None:
            try:
                _mask = _mask[attr]
            except IndexError:
                # Couldn't find a mask: use the default (nomask)
                pass
            tp_len = len(_mask.dtype)
            hasmasked = _mask.view((bool, ((tp_len,) if tp_len else ()))).any()
        if (obj.shape or hasmasked):
            obj = obj.view(ma.MaskedArray)
            obj._baseclass = np.ndarray
            obj._isfield = True
            obj._mask = _mask
            # Reset the field values
            _fill_value = _localdict.get('_fill_value', None)
            if _fill_value is not None:
                try:
                    obj._fill_value = _fill_value[attr]
                except ValueError:
                    obj._fill_value = None
        else:
            obj = obj.item()
        return obj

    def __setattr__(self, attr, val):
        """
        Sets the attribute attr to the value val.

        """
        # Should we call __setmask__ first ?
        if attr in ['mask', 'fieldmask']:
            self.__setmask__(val)
            return
        # Create a shortcut (so that we don't have to call getattr all the time)
        _localdict = object.__getattribute__(self, '__dict__')
        # Check whether we're creating a new field
        newattr = attr not in _localdict
        try:
            # Is attr a generic attribute ?
            ret = object.__setattr__(self, attr, val)
        except Exception:
            # Not a generic attribute: exit if it's not a valid field
            fielddict = np.ndarray.__getattribute__(self, 'dtype').fields or {}
            optinfo = np.ndarray.__getattribute__(self, '_optinfo') or {}
            if not (attr in fielddict or attr in optinfo):
                raise
        else:
            # Get the list of names
            fielddict = np.ndarray.__getattribute__(self, 'dtype').fields or {}
            # Check the attribute
            if attr not in fielddict:
                return ret
            if newattr:
                # We just added this one or this setattr worked on an
                # internal attribute.
                try:
                    object.__delattr__(self, attr)
                except Exception:
                    return ret
        # Let's try to set the field
        try:
            res = fielddict[attr][:2]
        except (TypeError, KeyError) as e:
            raise AttributeError(
                f'record array has no attribute {attr}') from e

        if val is ma.masked:
            _fill_value = _localdict['_fill_value']
            if _fill_value is not None:
                dval = _localdict['_fill_value'][attr]
            else:
                dval = val
            mval = True
        else:
            dval = ma.filled(val)
            mval = ma.getmaskarray(val)
        obj = np.ndarray.__getattribute__(self, '_data').setfield(dval, *res)
        _localdict['_mask'].__setitem__(attr, mval)
        return obj

    def __getitem__(self, indx):
        """
        Returns all the fields sharing the same fieldname base.

        The fieldname base is either `_data` or `_mask`.

        """
        _localdict = self.__dict__
        _mask = np.ndarray.__getattribute__(self, '_mask')
        _data = np.ndarray.view(self, _localdict['_baseclass'])
        # We want a field
        if isinstance(indx, str):
            # Make sure _sharedmask is True to propagate back to _fieldmask
            # Don't use _set_mask, there are some copies being made that
            # break propagation Don't force the mask to nomask, that wreaks
            # easy masking
            obj = _data[indx].view(ma.MaskedArray)
            obj._mask = _mask[indx]
            obj._sharedmask = True
            fval = _localdict['_fill_value']
            if fval is not None:
                obj._fill_value = fval[indx]
            # Force to masked if the mask is True
            if not obj.ndim and obj._mask:
                return ma.masked
            return obj
        # We want some elements.
        # First, the data.
        obj = np.asarray(_data[indx]).view(mrecarray)
        obj._mask = np.asarray(_mask[indx]).view(np.recarray)
        return obj

    def __setitem__(self, indx, value):
        """
        Sets the given record to value.

        """
        ma.MaskedArray.__setitem__(self, indx, value)
        if isinstance(indx, str):
            self._mask[indx] = ma.getmaskarray(value)

    def __str__(self):
        """
        Calculates the string representation.

        """
        if self.size > 1:
            mstr = [f"({','.join([str(i) for i in s])})"
                    for s in zip(*[getattr(self, f) for f in self.dtype.names])]
            return f"[{', '.join(mstr)}]"
        else:
            mstr = [f"{','.join([str(i) for i in s])}"
                    for s in zip([getattr(self, f) for f in self.dtype.names])]
            return f"({', '.join(mstr)})"

    def __repr__(self):
        """
        Calculates the repr representation.

        """
        _names = self.dtype.names
        fmt = "%%%is : %%s" % (max([len(n) for n in _names]) + 4,)
        reprstr = [fmt % (f, getattr(self, f)) for f in self.dtype.names]
        reprstr.insert(0, 'masked_records(')
        reprstr.extend([fmt % ('    fill_value', self.fill_value),
                        '              )'])
        return str("\n".join(reprstr))

    def view(self, dtype=None, type=None):
        """
        Returns a view of the mrecarray.

        """
        # OK, basic copy-paste from MaskedArray.view.
        if dtype is None:
            if type is None:
                output = np.ndarray.view(self)
            else:
                output = np.ndarray.view(self, type)
        # Here again.
        elif type is None:
            try:
                if issubclass(dtype, np.ndarray):
                    output = np.ndarray.view(self, dtype)
                else:
                    output = np.ndarray.view(self, dtype)
            # OK, there's the change
            except TypeError:
                dtype = np.dtype(dtype)
                # we need to revert to MaskedArray, but keeping the possibility
                # of subclasses (eg, TimeSeriesRecords), so we'll force a type
                # set to the first parent
                if dtype.fields is None:
                    basetype = self.__class__.__bases__[0]
                    output = self.__array__().view(dtype, basetype)
                    output._update_from(self)
                else:
                    output = np.ndarray.view(self, dtype)
                output._fill_value = None
        else:
            output = np.ndarray.view(self, dtype, type)
        # Update the mask, just like in MaskedArray.view
        if (getattr(output, '_mask', ma.nomask) is not ma.nomask):
            mdtype = ma.make_mask_descr(output.dtype)
            output._mask = self._mask.view(mdtype, np.ndarray)
            output._mask.shape = output.shape
        return output

    def harden_mask(self):
        """
        Forces the mask to hard.

        """
        self._hardmask = True

    def soften_mask(self):
        """
        Forces the mask to soft

        """
        self._hardmask = False

    def copy(self):
        """
        Returns a copy of the masked record.

        """
        copied = self._data.copy().view(type(self))
        copied._mask = self._mask.copy()
        return copied

    def tolist(self, fill_value=None):
        """
        Return the data portion of the array as a list.

        Data items are converted to the nearest compatible Python type.
        Masked values are converted to fill_value. If fill_value is None,
        the corresponding entries in the output list will be ``None``.

        """
        if fill_value is not None:
            return self.filled(fill_value).tolist()
        result = np.array(self.filled().tolist(), dtype=object)
        mask = np.array(self._mask.tolist())
        result[mask] = None
        return result.tolist()

    def __getstate__(self):
        """Return the internal state of the masked array.

        This is for pickling.

        """
        state = (1,
                 self.shape,
                 self.dtype,
                 self.flags.fnc,
                 self._data.tobytes(),
                 self._mask.tobytes(),
                 self._fill_value,
                 )
        return state

    def __setstate__(self, state):
        """
        Restore the internal state of the masked array.

        This is for pickling.  ``state`` is typically the output of the
        ``__getstate__`` output, and is a 5-tuple:

        - class name
        - a tuple giving the shape of the data
        - a typecode for the data
        - a binary string for the data
        - a binary string for the mask.

        """
        (ver, shp, typ, isf, raw, msk, flv) = state
        np.ndarray.__setstate__(self, (shp, typ, isf, raw))
        mdtype = np.dtype([(k, np.bool) for (k, _) in self.dtype.descr])
        self.__dict__['_mask'].__setstate__((shp, mdtype, isf, msk))
        self.fill_value = flv

    def __reduce__(self):
        """
        Return a 3-tuple for pickling a MaskedArray.

        """
        return (_mrreconstruct,
                (self.__class__, self._baseclass, (0,), 'b',),
                self.__getstate__())


def _mrreconstruct(subtype, baseclass, baseshape, basetype,):
    """
    Build a new MaskedArray from the information stored in a pickle.

    """
    _data = np.ndarray.__new__(baseclass, baseshape, basetype).view(subtype)
    _mask = np.ndarray.__new__(np.ndarray, baseshape, 'b1')
    return subtype.__new__(subtype, _data, mask=_mask, dtype=basetype,)

mrecarray = MaskedRecords


###############################################################################
#                             Constructors                                    #
###############################################################################


def fromarrays(arraylist, dtype=None, shape=None, formats=None,
               names=None, titles=None, aligned=False, byteorder=None,
               fill_value=None):
    """
    Creates a mrecarray from a (flat) list of masked arrays.

    Parameters
    ----------
    arraylist : sequence
        A list of (masked) arrays. Each element of the sequence is first converted
        to a masked array if needed. If a 2D array is passed as argument, it is
        processed line by line
    dtype : {None, dtype}, optional
        Data type descriptor.
    shape : {None, integer}, optional
        Number of records. If None, shape is defined from the shape of the
        first array in the list.
    formats : {None, sequence}, optional
        Sequence of formats for each individual field. If None, the formats will
        be autodetected by inspecting the fields and selecting the highest dtype
        possible.
    names : {None, sequence}, optional
        Sequence of the names of each field.
    fill_value : {None, sequence}, optional
        Sequence of data to be used as filling values.

    Notes
    -----
    Lists of tuples should be preferred over lists of lists for faster processing.

    """
    datalist = [ma.getdata(x) for x in arraylist]
    masklist = [np.atleast_1d(ma.getmaskarray(x)) for x in arraylist]
    _array = np.rec.fromarrays(datalist,
                               dtype=dtype, shape=shape, formats=formats,
                               names=names, titles=titles, aligned=aligned,
                               byteorder=byteorder).view(mrecarray)
    _array._mask.flat = list(zip(*masklist))
    if fill_value is not None:
        _array.fill_value = fill_value
    return _array


def fromrecords(reclist, dtype=None, shape=None, formats=None, names=None,
                titles=None, aligned=False, byteorder=None,
                fill_value=None, mask=ma.nomask):
    """
    Creates a MaskedRecords from a list of records.

    Parameters
    ----------
    reclist : sequence
        A list of records. Each element of the sequence is first converted
        to a masked array if needed. If a 2D array is passed as argument, it is
        processed line by line
    dtype : {None, dtype}, optional
        Data type descriptor.
    shape : {None,int}, optional
        Number of records. If None, ``shape`` is defined from the shape of the
        first array in the list.
    formats : {None, sequence}, optional
        Sequence of formats for each individual field. If None, the formats will
        be autodetected by inspecting the fields and selecting the highest dtype
        possible.
    names : {None, sequence}, optional
        Sequence of the names of each field.
    fill_value : {None, sequence}, optional
        Sequence of data to be used as filling values.
    mask : {nomask, sequence}, optional.
        External mask to apply on the data.

    Notes
    -----
    Lists of tuples should be preferred over lists of lists for faster processing.

    """
    # Grab the initial _fieldmask, if needed:
    _mask = getattr(reclist, '_mask', None)
    # Get the list of records.
    if isinstance(reclist, np.ndarray):
        # Make sure we don't have some hidden mask
        if isinstance(reclist, ma.MaskedArray):
            reclist = reclist.filled().view(np.ndarray)
        # Grab the initial dtype, just in case
        if dtype is None:
            dtype = reclist.dtype
        reclist = reclist.tolist()
    mrec = np.rec.fromrecords(reclist, dtype=dtype, shape=shape, formats=formats,
                          names=names, titles=titles,
                          aligned=aligned, byteorder=byteorder).view(mrecarray)
    # Set the fill_value if needed
    if fill_value is not None:
        mrec.fill_value = fill_value
    # Now, let's deal w/ the mask
    if mask is not ma.nomask:
        mask = np.asarray(mask)
        maskrecordlength = len(mask.dtype)
        if maskrecordlength:
            mrec._mask.flat = mask
        elif mask.ndim == 2:
            mrec._mask.flat = [tuple(m) for m in mask]
        else:
            mrec.__setmask__(mask)
    if _mask is not None:
        mrec._mask[:] = _mask
    return mrec


def _guessvartypes(arr):
    """
    Tries to guess the dtypes of the str_ ndarray `arr`.

    Guesses by testing element-wise conversion. Returns a list of dtypes.
    The array is first converted to ndarray. If the array is 2D, the test
    is performed on the first line. An exception is raised if the file is
    3D or more.

    """
    vartypes = []
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[0]
    elif arr.ndim > 2:
        raise ValueError("The array should be 2D at most!")
    # Start the conversion loop.
    for f in arr:
        try:
            int(f)
        except (ValueError, TypeError):
            try:
                float(f)
            except (ValueError, TypeError):
                try:
                    complex(f)
                except (ValueError, TypeError):
                    vartypes.append(arr.dtype)
                else:
                    vartypes.append(np.dtype(complex))
            else:
                vartypes.append(np.dtype(float))
        else:
            vartypes.append(np.dtype(int))
    return vartypes


def openfile(fname):
    """
    Opens the file handle of file `fname`.

    """
    # A file handle
    if hasattr(fname, 'readline'):
        return fname
    # Try to open the file and guess its type
    try:
        f = open(fname)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"No such file: '{fname}'") from e
    if f.readline()[:2] != "\\x":
        f.seek(0, 0)
        return f
    f.close()
    raise NotImplementedError("Wow, binary file")


def fromtextfile(fname, delimiter=None, commentchar='#', missingchar='',
                 varnames=None, vartypes=None,
                 *, delimitor=np._NoValue):  # backwards compatibility
    """
    Creates a mrecarray from data stored in the file `filename`.

    Parameters
    ----------
    fname : {file name/handle}
        Handle of an opened file.
    delimiter : {None, string}, optional
        Alphanumeric character used to separate columns in the file.
        If None, any (group of) white spacestring(s) will be used.
    commentchar : {'#', string}, optional
        Alphanumeric character used to mark the start of a comment.
    missingchar : {'', string}, optional
        String indicating missing data, and used to create the masks.
    varnames : {None, sequence}, optional
        Sequence of the variable names. If None, a list will be created from
        the first non empty line of the file.
    vartypes : {None, sequence}, optional
        Sequence of the variables dtypes. If None, it will be estimated from
        the first non-commented line.


    Ultra simple: the varnames are in the header, one line"""
    if delimitor is not np._NoValue:
        if delimiter is not None:
            raise TypeError("fromtextfile() got multiple values for argument "
                            "'delimiter'")
        # NumPy 1.22.0, 2021-09-23
        warnings.warn("The 'delimitor' keyword argument of "
                      "numpy.ma.mrecords.fromtextfile() is deprecated "
                      "since NumPy 1.22.0, use 'delimiter' instead.",
                      DeprecationWarning, stacklevel=2)
        delimiter = delimitor

    # Try to open the file.
    ftext = openfile(fname)

    # Get the first non-empty line as the varnames
    while True:
        line = ftext.readline()
        firstline = line[:line.find(commentchar)].strip()
        _varnames = firstline.split(delimiter)
        if len(_varnames) > 1:
            break
    if varnames is None:
        varnames = _varnames

    # Get the data.
    _variables = ma.masked_array([line.strip().split(delimiter) for line in ftext
                                 if line[0] != commentchar and len(line) > 1])
    (_, nfields) = _variables.shape
    ftext.close()

    # Try to guess the dtype.
    if vartypes is None:
        vartypes = _guessvartypes(_variables[0])
    else:
        vartypes = [np.dtype(v) for v in vartypes]
        if len(vartypes) != nfields:
            msg = "Attempting to %i dtypes for %i fields!"
            msg += " Reverting to default."
            warnings.warn(msg % (len(vartypes), nfields), stacklevel=2)
            vartypes = _guessvartypes(_variables[0])

    # Construct the descriptor.
    mdescr = list(zip(varnames, vartypes))
    mfillv = [ma.default_fill_value(f) for f in vartypes]

    # Get the data and the mask.
    # We just need a list of masked_arrays. It's easier to create it like that:
    _mask = (_variables.T == missingchar)
    _datalist = [ma.masked_array(a, mask=m, dtype=t, fill_value=f)
                 for (a, m, t, f) in zip(_variables.T, _mask, vartypes, mfillv)]

    return fromarrays(_datalist, dtype=mdescr)


def addfield(mrecord, newfield, newfieldname=None):
    """Adds a new field to the masked record array

    Uses `newfield` as data and `newfieldname` as name. If `newfieldname`
    is None, the new field name is set to 'fi', where `i` is the number of
    existing fields.

    """
    _data = mrecord._data
    _mask = mrecord._mask
    if newfieldname is None or newfieldname in reserved_fields:
        newfieldname = 'f%i' % len(_data.dtype)
    newfield = ma.array(newfield)
    # Get the new data.
    # Create a new empty recarray
    newdtype = np.dtype(_data.dtype.descr + [(newfieldname, newfield.dtype)])
    newdata = np.recarray(_data.shape, newdtype)
    # Add the existing field
    [newdata.setfield(_data.getfield(*f), *f)
     for f in _data.dtype.fields.values()]
    # Add the new field
    newdata.setfield(newfield._data, *newdata.dtype.fields[newfieldname])
    newdata = newdata.view(MaskedRecords)
    # Get the new mask
    # Create a new empty recarray
    newmdtype = np.dtype([(n, np.bool) for n in newdtype.names])
    newmask = np.recarray(_data.shape, newmdtype)
    # Add the old masks
    [newmask.setfield(_mask.getfield(*f), *f)
     for f in _mask.dtype.fields.values()]
    # Add the mask of the new field
    newmask.setfield(ma.getmaskarray(newfield),
                     *newmask.dtype.fields[newfieldname])
    newdata._mask = newmask
    return newdata
<|endoftext|>"""
Masked arrays add-ons.

A collection of utilities for `numpy.ma`.

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu
:version: $Id: extras.py 3473 2007-10-29 15:18:13Z jarrod.millman $

"""
__all__ = [
    'apply_along_axis', 'apply_over_axes', 'atleast_1d', 'atleast_2d',
    'atleast_3d', 'average', 'clump_masked', 'clump_unmasked', 'column_stack',
    'compress_cols', 'compress_nd', 'compress_rowcols', 'compress_rows',
    'count_masked', 'corrcoef', 'cov', 'diagflat', 'dot', 'dstack', 'ediff1d',
    'flatnotmasked_contiguous', 'flatnotmasked_edges', 'hsplit', 'hstack',
    'isin', 'in1d', 'intersect1d', 'mask_cols', 'mask_rowcols', 'mask_rows',
    'masked_all', 'masked_all_like', 'median', 'mr_', 'ndenumerate',
    'notmasked_contiguous', 'notmasked_edges', 'polyfit', 'row_stack',
    'setdiff1d', 'setxor1d', 'stack', 'unique', 'union1d', 'vander', 'vstack',
    ]

import itertools
import warnings

from . import core as ma
from .core import (
    MaskedArray, MAError, add, array, asarray, concatenate, filled, count,
    getmask, getmaskarray, make_mask_descr, masked, masked_array, mask_or,
    nomask, ones, sort, zeros, getdata, get_masked_subclass, dot
    )

import numpy as np
from numpy import ndarray, array as nxarray
from numpy.lib.array_utils import normalize_axis_index, normalize_axis_tuple
from numpy.lib._function_base_impl import _ureduce
from numpy.lib._index_tricks_impl import AxisConcatenator


def issequence(seq):
    """
    Is seq a sequence (ndarray, list or tuple)?

    """
    return isinstance(seq, (ndarray, tuple, list))


def count_masked(arr, axis=None):
    """
    Count the number of masked elements along the given axis.

    Parameters
    ----------
    arr : array_like
        An array with (possibly) masked elements.
    axis : int, optional
        Axis along which to count. If None (default), a flattened
        version of the array is used.

    Returns
    -------
    count : int, ndarray
        The total number of masked elements (axis=None) or the number
        of masked elements along each slice of the given axis.

    See Also
    --------
    MaskedArray.count : Count non-masked elements.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(9).reshape((3,3))
    >>> a = np.ma.array(a)
    >>> a[1, 0] = np.ma.masked
    >>> a[1, 2] = np.ma.masked
    >>> a[2, 1] = np.ma.masked
    >>> a
    masked_array(
      data=[[0, 1, 2],
            [--, 4, --],
            [6, --, 8]],
      mask=[[False, False, False],
            [ True, False,  True],
            [False,  True, False]],
      fill_value=999999)
    >>> np.ma.count_masked(a)
    3

    When the `axis` keyword is used an array is returned.

    >>> np.ma.count_masked(a, axis=0)
    array([1, 1, 1])
    >>> np.ma.count_masked(a, axis=1)
    array([0, 2, 1])

    """
    m = getmaskarray(arr)
    return m.sum(axis)


def masked_all(shape, dtype=float):
    """
    Empty masked array with all elements masked.

    Return an empty masked array of the given shape and dtype, where all the
    data are masked.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the required MaskedArray, e.g., ``(2, 3)`` or ``2``.
    dtype : dtype, optional
        Data type of the output.

    Returns
    -------
    a : MaskedArray
        A masked array with all data masked.

    See Also
    --------
    masked_all_like : Empty masked array modelled on an existing array.

    Notes
    -----
    Unlike other masked array creation functions (e.g. `numpy.ma.zeros`,
    `numpy.ma.ones`, `numpy.ma.full`), `masked_all` does not initialize the
    values of the array, and may therefore be marginally faster. However,
    the values stored in the newly allocated array are arbitrary. For
    reproducible behavior, be sure to set each element of the array before
    reading.

    Examples
    --------
    >>> import numpy as np
    >>> np.ma.masked_all((3, 3))
    masked_array(
      data=[[--, --, --],
            [--, --, --],
            [--, --, --]],
      mask=[[ True,  True,  True],
            [ True,  True,  True],
            [ True,  True,  True]],
      fill_value=1e+20,
      dtype=float64)

    The `dtype` parameter defines the underlying data type.

    >>> a = np.ma.masked_all((3, 3))
    >>> a.dtype
    dtype('float64')
    >>> a = np.ma.masked_all((3, 3), dtype=np.int32)
    >>> a.dtype
    dtype('int32')

    """
    a = masked_array(np.empty(shape, dtype),
                     mask=np.ones(shape, make_mask_descr(dtype)))
    return a


def masked_all_like(arr):
    """
    Empty masked array with the properties of an existing array.

    Return an empty masked array of the same shape and dtype as
    the array `arr`, where all the data are masked.

    Parameters
    ----------
    arr : ndarray
        An array describing the shape and dtype of the required MaskedArray.

    Returns
    -------
    a : MaskedArray
        A masked array with all data masked.

    Raises
    ------
    AttributeError
        If `arr` doesn't have a shape attribute (i.e. not an ndarray)

    See Also
    --------
    masked_all : Empty masked array with all elements masked.

    Notes
    -----
    Unlike other masked array creation functions (e.g. `numpy.ma.zeros_like`,
    `numpy.ma.ones_like`, `numpy.ma.full_like`), `masked_all_like` does not
    initialize the values of the array, and may therefore be marginally
    faster. However, the values stored in the newly allocated array are
    arbitrary. For reproducible behavior, be sure to set each element of the
    array before reading.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.zeros((2, 3), dtype=np.float32)
    >>> arr
    array([[0., 0., 0.],
           [0., 0., 0.]], dtype=float32)
    >>> np.ma.masked_all_like(arr)
    masked_array(
      data=[[--, --, --],
            [--, --, --]],
      mask=[[ True,  True,  True],
            [ True,  True,  True]],
      fill_value=np.float64(1e+20),
      dtype=float32)

    The dtype of the masked array matches the dtype of `arr`.

    >>> arr.dtype
    dtype('float32')
    >>> np.ma.masked_all_like(arr).dtype
    dtype('float32')

    """
    a = np.empty_like(arr).view(MaskedArray)
    a._mask = np.ones(a.shape, dtype=make_mask_descr(a.dtype))
    return a


#####--------------------------------------------------------------------------
#---- --- Standard functions ---
#####--------------------------------------------------------------------------
class _fromnxfunction:
    """
    Defines a wrapper to adapt NumPy functions to masked arrays.


    An instance of `_fromnxfunction` can be called with the same parameters
    as the wrapped NumPy function. The docstring of `newfunc` is adapted from
    the wrapped function as well, see `getdoc`.

    This class should not be used directly. Instead, one of its extensions that
    provides support for a specific type of input should be used.

    Parameters
    ----------
    funcname : str
        The name of the function to be adapted. The function should be
        in the NumPy namespace (i.e. ``np.funcname``).

    """

    def __init__(self, funcname):
        self.__name__ = funcname
        self.__qualname__ = funcname
        self.__doc__ = self.getdoc()

    def getdoc(self):
        """
        Retrieve the docstring and signature from the function.

        The ``__doc__`` attribute of the function is used as the docstring for
        the new masked array version of the function. A note on application
        of the function to the mask is appended.

        Parameters
        ----------
        None

        """
        npfunc = getattr(np, self.__name__, None)
        doc = getattr(npfunc, '__doc__', None)
        if doc:
            sig = ma.get_object_signature(npfunc)
            doc = ma.doc_note(doc, "The function is applied to both the _data "
                                   "and the _mask, if any.")
            if sig:
                sig = self.__name__ + sig + "\n\n"
            return sig + doc
        return

    def __call__(self, *args, **params):
        pass


class _fromnxfunction_single(_fromnxfunction):
    """
    A version of `_fromnxfunction` that is called with a single array
    argument followed by auxiliary args that are passed verbatim for
    both the data and mask calls.
    """
    def __call__(self, x, *args, **params):
        func = getattr(np, self.__name__)
        if isinstance(x, ndarray):
            _d = func(x.__array__(), *args, **params)
            _m = func(getmaskarray(x), *args, **params)
            return masked_array(_d, mask=_m)
        else:
            _d = func(np.asarray(x), *args, **params)
            _m = func(getmaskarray(x), *args, **params)
            return masked_array(_d, mask=_m)


class _fromnxfunction_seq(_fromnxfunction):
    """
    A version of `_fromnxfunction` that is called with a single sequence
    of arrays followed by auxiliary args that are passed verbatim for
    both the data and mask calls.
    """
    def __call__(self, x, *args, **params):
        func = getattr(np, self.__name__)
        _d = func(tuple([np.asarray(a) for a in x]), *args, **params)
        _m = func(tuple([getmaskarray(a) for a in x]), *args, **params)
        return masked_array(_d, mask=_m)


class _fromnxfunction_args(_fromnxfunction):
    """
    A version of `_fromnxfunction` that is called with multiple array
    arguments. The first non-array-like input marks the beginning of the
    arguments that are passed verbatim for both the data and mask calls.
    Array arguments are processed independently and the results are
    returned in a list. If only one array is found, the return value is
    just the processed array instead of a list.
    """
    def __call__(self, *args, **params):
        func = getattr(np, self.__name__)
        arrays = []
        args = list(args)
        while len(args) > 0 and issequence(args[0]):
            arrays.append(args.pop(0))
        res = []
        for x in arrays:
            _d = func(np.asarray(x), *args, **params)
            _m = func(getmaskarray(x), *args, **params)
            res.append(masked_array(_d, mask=_m))
        if len(arrays) == 1:
            return res[0]
        return res


class _fromnxfunction_allargs(_fromnxfunction):
    """
    A version of `_fromnxfunction` that is called with multiple array
    arguments. Similar to `_fromnxfunction_args` except that all args
    are converted to arrays even if they are not so already. This makes
    it possible to process scalars as 1-D arrays. Only keyword arguments
    are passed through verbatim for the data and mask calls. Arrays
    arguments are processed independently and the results are returned
    in a list. If only one arg is present, the return value is just the
    processed array instead of a list.
    """
    def __call__(self, *args, **params):
        func = getattr(np, self.__name__)
        res = []
        for x in args:
            _d = func(np.asarray(x), **params)
            _m = func(getmaskarray(x), **params)
            res.append(masked_array(_d, mask=_m))
        if len(args) == 1:
            return res[0]
        return res


atleast_1d = _fromnxfunction_allargs('atleast_1d')
atleast_2d = _fromnxfunction_allargs('atleast_2d')
atleast_3d = _fromnxfunction_allargs('atleast_3d')

vstack = row_stack = _fromnxfunction_seq('vstack')
hstack = _fromnxfunction_seq('hstack')
column_stack = _fromnxfunction_seq('column_stack')
dstack = _fromnxfunction_seq('dstack')
stack = _fromnxfunction_seq('stack')

hsplit = _fromnxfunction_single('hsplit')

diagflat = _fromnxfunction_single('diagflat')


#####--------------------------------------------------------------------------
#----
#####--------------------------------------------------------------------------
def flatten_inplace(seq):
    """Flatten a sequence in place."""
    k = 0
    while (k != len(seq)):
        while hasattr(seq[k], '__iter__'):
            seq[k:(k + 1)] = seq[k]
        k += 1
    return seq


def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    (This docstring should be overwritten)
    """
    arr = array(arr, copy=False, subok=True)
    nd = arr.ndim
    axis = normalize_axis_index(axis, nd)
    ind = [0] * (nd - 1)
    i = np.zeros(nd, 'O')
    indlist = list(range(nd))
    indlist.remove(axis)
    i[axis] = slice(None, None)
    outshape = np.asarray(arr.shape).take(indlist)
    i.put(indlist, ind)
    res = func1d(arr[tuple(i.tolist())], *args, **kwargs)
    #  if res is a number, then we have a smaller output array
    asscalar = np.isscalar(res)
    if not asscalar:
        try:
            len(res)
        except TypeError:
            asscalar = True
    # Note: we shouldn't set the dtype of the output from the first result
    # so we force the type to object, and build a list of dtypes.  We'll
    # just take the largest, to avoid some downcasting
    dtypes = []
    if asscalar:
        dtypes.append(np.asarray(res).dtype)
        outarr = zeros(outshape, object)
        outarr[tuple(ind)] = res
        Ntot = np.prod(outshape)
        k = 1
        while k < Ntot:
            # increment the index
            ind[-1] += 1
            n = -1
            while (ind[n] >= outshape[n]) and (n > (1 - nd)):
                ind[n - 1] += 1
                ind[n] = 0
                n -= 1
            i.put(indlist, ind)
            res = func1d(arr[tuple(i.tolist())], *args, **kwargs)
            outarr[tuple(ind)] = res
            dtypes.append(asarray(res).dtype)
            k += 1
    else:
        res = array(res, copy=False, subok=True)
        j = i.copy()
        j[axis] = ([slice(None, None)] * res.ndim)
        j.put(indlist, ind)
        Ntot = np.prod(outshape)
        holdshape = outshape
        outshape = list(arr.shape)
        outshape[axis] = res.shape
        dtypes.append(asarray(res).dtype)
        outshape = flatten_inplace(outshape)
        outarr = zeros(outshape, object)
        outarr[tuple(flatten_inplace(j.tolist()))] = res
        k = 1
        while k < Ntot:
            # increment the index
            ind[-1] += 1
            n = -1
            while (ind[n] >= holdshape[n]) and (n > (1 - nd)):
                ind[n - 1] += 1
                ind[n] = 0
                n -= 1
            i.put(indlist, ind)
            j.put(indlist, ind)
            res = func1d(arr[tuple(i.tolist())], *args, **kwargs)
            outarr[tuple(flatten_inplace(j.tolist()))] = res
            dtypes.append(asarray(res).dtype)
            k += 1
    max_dtypes = np.dtype(np.asarray(dtypes).max())
    if not hasattr(arr, '_mask'):
        result = np.asarray(outarr, dtype=max_dtypes)
    else:
        result = asarray(outarr, dtype=max_dtypes)
        result.fill_value = ma.default_fill_value(result)
    return result
apply_along_axis.__doc__ = np.apply_along_axis.__doc__


def apply_over_axes(func, a, axes):
    """
    (This docstring will be overwritten)
    """
    val = asarray(a)
    N = a.ndim
    if array(axes).ndim == 0:
        axes = (axes,)
    for axis in axes:
        if axis < 0:
            axis = N + axis
        args = (val, axis)
        res = func(*args)
        if res.ndim == val.ndim:
            val = res
        else:
            res = ma.expand_dims(res, axis)
            if res.ndim == val.ndim:
                val = res
            else:
                raise ValueError("function is not returning "
                        "an array of the correct shape")
    return val


if apply_over_axes.__doc__ is not None:
    apply_over_axes.__doc__ = np.apply_over_axes.__doc__[
        :np.apply_over_axes.__doc__.find('Notes')].rstrip() + \
    """

    Examples
    --------
    >>> import numpy as np
    >>> a = np.ma.arange(24).reshape(2,3,4)
    >>> a[:,0,1] = np.ma.masked
    >>> a[:,1,:] = np.ma.masked
    >>> a
    masked_array(
      data=[[[0, --, 2, 3],
             [--, --, --, --],
             [8, 9, 10, 11]],
            [[12, --, 14, 15],
             [--, --, --, --],
             [20, 21, 22, 23]]],
      mask=[[[False,  True, False, False],
             [ True,  True,  True,  True],
             [False, False, False, False]],
            [[False,  True, False, False],
             [ True,  True,  True,  True],
             [False, False, False, False]]],
      fill_value=999999)
    >>> np.ma.apply_over_axes(np.ma.sum, a, [0,2])
    masked_array(
      data=[[[46],
             [--],
             [124]]],
      mask=[[[False],
             [ True],
             [False]]],
      fill_value=999999)

    Tuple axis arguments to ufuncs are equivalent:

    >>> np.ma.sum(a, axis=(0,2)).reshape((1,-1,1))
    masked_array(
      data=[[[46],
             [--],
             [124]]],
      mask=[[[False],
             [ True],
             [False]]],
      fill_value=999999)
    """


def average(a, axis=None, weights=None, returned=False, *,
            keepdims=np._NoValue):
    """
    Return the weighted average of array over the given axis.

    Parameters
    ----------
    a : array_like
        Data to be averaged.
        Masked entries are not taken into account in the computation.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to average `a`.  The default,
        `axis=None`, will average over all of the elements of the input array.
        If axis is a tuple of ints, averaging is performed on all of the axes
        specified in the tuple instead of a single axis or all the axes as
        before.
    weights : array_like, optional
        An array of weights associated with the values in `a`. Each value in
        `a` contributes to the average according to its associated weight.
        The array of weights must be the same shape as `a` if no axis is
        specified, otherwise the weights must have dimensions and shape
        consistent with `a` along the specified axis.
        If `weights=None`, then all data in `a` are assumed to have a
        weight equal to one.
        The calculation is::

            avg = sum(a * weights) / sum(weights)

        where the sum is over all included elements.
        The only constraint on the values of `weights` is that `sum(weights)`
        must not be 0.
    returned : bool, optional
        Flag indicating whether a tuple ``(result, sum of weights)``
        should be returned as output (True), or just the result (False).
        Default is False.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `a`.
        *Note:* `keepdims` will not work with instances of `numpy.matrix`
        or other classes whose methods do not support `keepdims`.

        .. versionadded:: 1.23.0

    Returns
    -------
    average, [sum_of_weights] : (tuple of) scalar or MaskedArray
        The average along the specified axis. When returned is `True`,
        return a tuple with the average as the first element and the sum
        of the weights as the second element. The return type is `np.float64`
        if `a` is of integer type and floats smaller than `float64`, or the
        input data-type, otherwise. If returned, `sum_of_weights` is always
        `float64`.

    Raises
    ------
    ZeroDivisionError
        When all weights along axis are zero. See `numpy.ma.average` for a
        version robust to this type of error.
    TypeError
        When `weights` does not have the same shape as `a`, and `axis=None`.
    ValueError
        When `weights` does not have dimensions and shape consistent with `a`
        along specified `axis`.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.ma.array([1., 2., 3., 4.], mask=[False, False, True, True])
    >>> np.ma.average(a, weights=[3, 1, 0, 0])
    1.25

    >>> x = np.ma.arange(6.).reshape(3, 2)
    >>> x
    masked_array(
      data=[[0., 1.],
            [2., 3.],
            [4., 5.]],
      mask=False,
      fill_value=1e+20)
    >>> data = np.arange(8).reshape((2, 2, 2))
    >>> data
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> np.ma.average(data, axis=(0, 1), weights=[[1./4, 3./4], [1., 1./2]])
    masked_array(data=[3.4, 4.4],
             mask=[False, False],
       fill_value=1e+20)
    >>> np.ma.average(data, axis=0, weights=[[1./4, 3./4], [1., 1./2]])
    Traceback (most recent call last):
        ...
    ValueError: Shape of weights must be consistent
    with shape of a along specified axis.

    >>> avg, sumweights = np.ma.average(x, axis=0, weights=[1, 2, 3],
    ...                                 returned=True)
    >>> avg
    masked_array(data=[2.6666666666666665, 3.6666666666666665],
                 mask=[False, False],
           fill_value=1e+20)

    With ``keepdims=True``, the following result has shape (3, 1).

    >>> np.ma.average(x, axis=1, keepdims=True)
    masked_array(
      data=[[0.5],
            [2.5],
            [4.5]],
      mask=False,
      fill_value=1e+20)
    """
    a = asarray(a)
    m = getmask(a)

    if axis is not None:
        axis = normalize_axis_tuple(axis, a.ndim, argname="axis")

    if keepdims is np._NoValue:
        # Don't pass on the keepdims argument if one wasn't given.
        keepdims_kw = {}
    else:
        keepdims_kw = {'keepdims': keepdims}

    if weights is None:
        avg = a.mean(axis, **keepdims_kw)
        scl = avg.dtype.type(a.count(axis))
    else:
        wgt = asarray(weights)

        if issubclass(a.dtype.type, (np.integer, np.bool)):
            result_dtype = np.result_type(a.dtype, wgt.dtype, 'f8')
        else:
            result_dtype = np.result_type(a.dtype, wgt.dtype)

        # Sanity checks
        if a.shape != wgt.shape:
            if axis is None:
                raise TypeError(
                    "Axis must be specified when shapes of a and weights "
                    "differ.")
            if wgt.shape != tuple(a.shape[ax] for ax in axis):
                raise ValueError(
                    "Shape of weights must be consistent with "
                    "shape of a along specified axis.")

            # setup wgt to broadcast along axis
            wgt = wgt.transpose(np.argsort(axis))
            wgt = wgt.reshape(tuple((s if ax in axis else 1)
                                    for ax, s in enumerate(a.shape)))

        if m is not nomask:
            wgt = wgt*(~a.mask)
            wgt.mask |= a.mask

        scl = wgt.sum(axis=axis, dtype=result_dtype, **keepdims_kw)
        avg = np.multiply(a, wgt,
                          dtype=result_dtype).sum(axis, **keepdims_kw) / scl

    if returned:
        if scl.shape != avg.shape:
            scl = np.broadcast_to(scl, avg.shape).copy()
        return avg, scl
    else:
        return avg


def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    """
    Compute the median along the specified axis.

    Returns the median of the array elements.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : int, optional
        Axis along which the medians are computed. The default (None) is
        to compute the median along a flattened version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type will be cast if necessary.
    overwrite_input : bool, optional
        If True, then allow use of memory of input array (a) for
        calculations. The input array will be modified by the call to
        median. This will save memory when you do not need to preserve
        the contents of the input array. Treat the input as undefined,
        but it will probably be fully or partially sorted. Default is
        False. Note that, if `overwrite_input` is True, and the input
        is not already an `ndarray`, an error will be raised.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

    Returns
    -------
    median : ndarray
        A new array holding the result is returned unless out is
        specified, in which case a reference to out is returned.
        Return data-type is `float64` for integers and floats smaller than
        `float64`, or the input data-type, otherwise.

    See Also
    --------
    mean

    Notes
    -----
    Given a vector ``V`` with ``N`` non masked values, the median of ``V``
    is the middle value of a sorted copy of ``V`` (``Vs``) - i.e.
    ``Vs[(N-1)/2]``, when ``N`` is odd, or ``{Vs[N/2 - 1] + Vs[N/2]}/2``
    when ``N`` is even.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.ma.array(np.arange(8), mask=[0]*4 + [1]*4)
    >>> np.ma.median(x)
    1.5

    >>> x = np.ma.array(np.arange(10).reshape(2, 5), mask=[0]*6 + [1]*4)
    >>> np.ma.median(x)
    2.5
    >>> np.ma.median(x, axis=-1, overwrite_input=True)
    masked_array(data=[2.0, 5.0],
                 mask=[False, False],
           fill_value=1e+20)

    """
    if not hasattr(a, 'mask'):
        m = np.median(getdata(a, subok=True), axis=axis,
                      out=out, overwrite_input=overwrite_input,
                      keepdims=keepdims)
        if isinstance(m, np.ndarray) and 1 <= m.ndim:
            return masked_array(m, copy=False)
        else:
            return m

    return _ureduce(a, func=_median, keepdims=keepdims, axis=axis, out=out,
                    overwrite_input=overwrite_input)


def _median(a, axis=None, out=None, overwrite_input=False):
    # when an unmasked NaN is present return it, so we need to sort the NaN
    # values behind the mask
    if np.issubdtype(a.dtype, np.inexact):
        fill_value = np.inf
    else:
        fill_value = None
    if overwrite_input:
        if axis is None:
            asorted = a.ravel()
            asorted.sort(fill_value=fill_value)
        else:
            a.sort(axis=axis, fill_value=fill_value)
            asorted = a
    else:
        asorted = sort(a, axis=axis, fill_value=fill_value)

    if axis is None:
        axis = 0
    else:
        axis = normalize_axis_index(axis, asorted.ndim)

    if asorted.shape[axis] == 0:
        # for empty axis integer indices fail so use slicing to get same result
        # as median (which is mean of empty slice = nan)
        indexer = [slice(None)] * asorted.ndim
        indexer[axis] = slice(0, 0)
        indexer = tuple(indexer)
        return np.ma.mean(asorted[indexer], axis=axis, out=out)

    if asorted.ndim == 1:
        idx, odd = divmod(count(asorted), 2)
        mid = asorted[idx + odd - 1:idx + 1]
        if np.issubdtype(asorted.dtype, np.inexact) and asorted.size > 0:
            # avoid inf / x = masked
            s = mid.sum(out=out)
            if not odd:
                s = np.true_divide(s, 2., casting='safe', out=out)
            s = np.lib._utils_impl._median_nancheck(asorted, s, axis)
        else:
            s = mid.mean(out=out)

        # if result is masked either the input contained enough
        # minimum_fill_value so that it would be the median or all values
        # masked
        if np.ma.is_masked(s) and not np.all(asorted.mask):
            return np.ma.minimum_fill_value(asorted)
        return s

    counts = count(asorted, axis=axis, keepdims=True)
    h = counts // 2

    # duplicate high if odd number of elements so mean does nothing
    odd = counts % 2 == 1
    l = np.where(odd, h, h-1)

    lh = np.concatenate([l,h], axis=axis)

    # get low and high median
    low_high = np.take_along_axis(asorted, lh, axis=axis)

    def replace_masked(s):
        # Replace masked entries with minimum_full_value unless it all values
        # are masked. This is required as the sort order of values equal or
        # larger than the fill value is undefined and a valid value placed
        # elsewhere, e.g. [4, --, inf].
        if np.ma.is_masked(s):
            rep = (~np.all(asorted.mask, axis=axis, keepdims=True)) & s.mask
            s.data[rep] = np.ma.minimum_fill_value(asorted)
            s.mask[rep] = False

    replace_masked(low_high)

    if np.issubdtype(asorted.dtype, np.inexact):
        # avoid inf / x = masked
        s = np.ma.sum(low_high, axis=axis, out=out)
        np.true_divide(s.data, 2., casting='unsafe', out=s.data)

        s = np.lib._utils_impl._median_nancheck(asorted, s, axis)
    else:
        s = np.ma.mean(low_high, axis=axis, out=out)

    return s


def compress_nd(x, axis=None):
    """Suppress slices from multiple dimensions which contain masked values.

    Parameters
    ----------
    x : array_like, MaskedArray
        The array to operate on. If not a MaskedArray instance (or if no array
        elements are masked), `x` is interpreted as a MaskedArray with `mask`
        set to `nomask`.
    axis : tuple of ints or int, optional
        Which dimensions to suppress slices from can be configured with this
        parameter.
        - If axis is a tuple of ints, those are the axes to suppress slices from.
        - If axis is an int, then that is the only axis to suppress slices from.
        - If axis is None, all axis are selected.

    Returns
    -------
    compress_array : ndarray
        The compressed array.

    Examples
    --------
    >>> import numpy as np
    >>> arr = [[1, 2], [3, 4]]
    >>> mask = [[0, 1], [0, 0]]
    >>> x = np.ma.array(arr, mask=mask)
    >>> np.ma.compress_nd(x, axis=0)
    array([[3, 4]])
    >>> np.ma.compress_nd(x, axis=1)
    array([[1],
           [3]])
    >>> np.ma.compress_nd(x)
    array([[3]])

    """
    x = asarray(x)
    m = getmask(x)
    # Set axis to tuple of ints
    if axis is None:
        axis = tuple(range(x.ndim))
    else:
        axis = normalize_axis_tuple(axis, x.ndim)

    # Nothing is masked: return x
    if m is nomask or not m.any():
        return x._data
    # All is masked: return empty
    if m.all():
        return nxarray([])
    # Filter elements through boolean indexing
    data = x._data
    for ax in axis:
        axes = tuple(list(range(ax)) + list(range(ax + 1, x.ndim)))
        data = data[(slice(None),)*ax + (~m.any(axis=axes),)]
    return data


def compress_rowcols(x, axis=None):
    """
    Suppress the rows and/or columns of a 2-D array that contain
    masked values.

    The suppression behavior is selected with the `axis` parameter.

    - If axis is None, both rows and columns are suppressed.
    - If axis is 0, only rows are suppressed.
    - If axis is 1 or -1, only columns are suppressed.

    Parameters
    ----------
    x : array_like, MaskedArray
        The array to operate on.  If not a MaskedArray instance (or if no array
        elements are masked), `x` is interpreted as a MaskedArray with
        `mask` set to `nomask`. Must be a 2D array.
    axis : int, optional
        Axis along which to perform the operation. Default is None.

    Returns
    -------
    compressed_array : ndarray
        The compressed array.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.ma.array(np.arange(9).reshape(3, 3), mask=[[1, 0, 0],
    ...                                                   [1, 0, 0],
    ...                                                   [0, 0, 0]])
    >>> x
    masked_array(
      data=[[--, 1, 2],
            [--, 4, 5],
            [6, 7, 8]],
      mask=[[ True, False, False],
            [ True, False, False],
            [False, False, False]],
      fill_value=999999)

    >>> np.ma.compress_rowcols(x)
    array([[7, 8]])
    >>> np.ma.compress_rowcols(x, 0)
    array([[6, 7, 8]])
    >>> np.ma.compress_rowcols(x, 1)
    array([[1, 2],
           [4, 5],
           [7, 8]])

    """
    if asarray(x).ndim != 2:
        raise NotImplementedError("compress_rowcols works for 2D arrays only.")
    return compress_nd(x, axis=axis)


def compress_rows(a):
    """
    Suppress whole rows of a 2-D array that contain masked values.

    This is equivalent to ``np.ma.compress_rowcols(a, 0)``, see
    `compress_rowcols` for details.

    Parameters
    ----------
    x : array_like, MaskedArray
        The array to operate on. If not a MaskedArray instance (or if no array
        elements are masked), `x` is interpreted as a MaskedArray with
        `mask` set to `nomask`. Must be a 2D array.

    Returns
    -------
    compressed_array : ndarray
        The compressed array.

    See Also
    --------
    compress_rowcols

    Examples
    --------
    >>> import numpy as np
    >>> a = np.ma.array(np.arange(9).reshape(3, 3), mask=[[1, 0, 0],
    ...                                                   [1, 0, 0],
    ...                                                   [0, 0, 0]])
    >>> np.ma.compress_rows(a)
    array([[6, 7, 8]])

    """
    a = asarray(a)
    if a.ndim != 2:
        raise NotImplementedError("compress_rows works for 2D arrays only.")
    return compress_rowcols(a, 0)


def compress_cols(a):
    """
    Suppress whole columns of a 2-D array that contain masked values.

    This is equivalent to ``np.ma.compress_rowcols(a, 1)``, see
    `compress_rowcols` for details.

    Parameters
    ----------
    x : array_like, MaskedArray
        The array to operate on.  If not a MaskedArray instance (or if no array
        elements are masked), `x` is interpreted as a MaskedArray with
        `mask` set to `nomask`. Must be a 2D array.

    Returns
    -------
    compressed_array : ndarray
        The compressed array.

    See Also
    --------
    compress_rowcols

    Examples
    --------
    >>> import numpy as np
    >>> a = np.ma.array(np.arange(9).reshape(3, 3), mask=[[1, 0, 0],
    ...                                                   [1, 0, 0],
    ...                                                   [0, 0, 0]])
    >>> np.ma.compress_cols(a)
    array([[1, 2],
           [4, 5],
           [7, 8]])

    """
    a = asarray(a)
    if a.ndim != 2:
        raise NotImplementedError("compress_cols works for 2D arrays only.")
    return compress_rowcols(a, 1)


def mask_rowcols(a, axis=None):
    """
    Mask rows and/or columns of a 2D array that contain masked values.

    Mask whole rows and/or columns of a 2D array that contain
    masked values.  The masking behavior is selected using the
    `axis` parameter.

      - If `axis` is None, rows *and* columns are masked.
      - If `axis` is 0, only rows are masked.
      - If `axis` is 1 or -1, only columns are masked.

    Parameters
    ----------
    a : array_like, MaskedArray
        The array to mask.  If not a MaskedArray instance (or if no array
        elements are masked), the result is a MaskedArray with `mask` set
        to `nomask` (False). Must be a 2D array.
    axis : int, optional
        Axis along which to perform the operation. If None, applies to a
        flattened version of the array.

    Returns
    -------
    a : MaskedArray
        A modified version of the input array, masked depending on the value
        of the `axis` parameter.

    Raises
    ------
    NotImplementedError
        If input array `a` is not 2D.

    See Also
    --------
    mask_rows : Mask rows of a 2D array that contain masked values.
    mask_cols : Mask cols of a 2D array that contain masked values.
    masked_where : Mask where a condition is met.

    Notes
    -----
    The input array's mask is modified by this function.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.zeros((3, 3), dtype=int)
    >>> a[1, 1] = 1
    >>> a
    array([[0, 0, 0],
           [0, 1, 0],
           [0, 0, 0]])
    >>> a = np.ma.masked_equal(a, 1)
    >>> a
    masked_array(
      data=[[0, 0, 0],
            [0, --, 0],
            [0, 0, 0]],
      mask=[[False, False, False],
            [False,  True, False],
            [False, False, False]],
      fill_value=1)
    >>> np.ma.mask_rowcols(a)
    masked_array(
      data=[[0, --, 0],
            [--, --, --],
            [0, --, 0]],
      mask=[[False,  True, False],
            [ True,  True,  True],
            [False,  True, False]],
      fill_value=1)

    """
    a = array(a, subok=False)
    if a.ndim != 2:
        raise NotImplementedError("mask_rowcols works for 2D arrays only.")
    m = getmask(a)
    # Nothing is masked: return a
    if m is nomask or not m.any():
        return a
    maskedval = m.nonzero()
    a._mask = a._mask.copy()
    if not axis:
        a[np.unique(maskedval[0])] = masked
    if axis in [None, 1, -1]:
        a[:, np.unique(maskedval[1])] = masked
    return a


def mask_rows(a, axis=np._NoValue):
    """
    Mask rows of a 2D array that contain masked values.

    This function is a shortcut to ``mask_rowcols`` with `axis` equal to 0.

    See Also
    --------
    mask_rowcols : Mask rows and/or columns of a 2D array.
    masked_where : Mask where a condition is met.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.zeros((3, 3), dtype=int)
    >>> a[1, 1] = 1
    >>> a
    array([[0, 0, 0],
           [0, 1, 0],
           [0, 0, 0]])
    >>> a = np.ma.masked_equal(a, 1)
    >>> a
    masked_array(
      data=[[0, 0, 0],
            [0, --, 0],
            [0, 0, 0]],
      mask=[[False, False, False],
            [False,  True, False],
            [False, False, False]],
      fill_value=1)

    >>> np.ma.mask_rows(a)
    masked_array(
      data=[[0, 0, 0],
            [--, --, --],
            [0, 0, 0]],
      mask=[[False, False, False],
            [ True,  True,  True],
            [False, False, False]],
      fill_value=1)

    """
    if axis is not np._NoValue:
        # remove the axis argument when this deprecation expires
        # NumPy 1.18.0, 2019-11-28
        warnings.warn(
            "The axis argument has always been ignored, in future passing it "
            "will raise TypeError", DeprecationWarning, stacklevel=2)
    return mask_rowcols(a, 0)


def mask_cols(a, axis=np._NoValue):
    """
    Mask columns of a 2D array that contain masked values.

    This function is a shortcut to ``mask_rowcols`` with `axis` equal to 1.

    See Also
    --------
    mask_rowcols : Mask rows and/or columns of a 2D array.
    masked_where : Mask where a condition is met.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.zeros((3, 3), dtype=int)
    >>> a[1, 1] = 1
    >>> a
    array([[0, 0, 0],
           [0, 1, 0],
           [0, 0, 0]])
    >>> a = np.ma.masked_equal(a, 1)
    >>> a
    masked_array(
      data=[[0, 0, 0],
            [0, --, 0],
            [0, 0, 0]],
      mask=[[False, False, False],
            [False,  True, False],
            [False, False, False]],
      fill_value=1)
    >>> np.ma.mask_cols(a)
    masked_array(
      data=[[0, --, 0],
            [0, --, 0],
            [0, --, 0]],
      mask=[[False,  True, False],
            [False,  True, False],
            [False,  True, False]],
      fill_value=1)

    """
    if axis is not np._NoValue:
        # remove the axis argument when this deprecation expires
        # NumPy 1.18.0, 2019-11-28
        warnings.warn(
            "The axis argument has always been ignored, in future passing it "
            "will raise TypeError", DeprecationWarning, stacklevel=2)
    return mask_rowcols(a, 1)


#####--------------------------------------------------------------------------
#---- --- arraysetops ---
#####--------------------------------------------------------------------------

def ediff1d(arr, to_end=None, to_begin=None):
    """
    Compute the differences between consecutive elements of an array.

    This function is the equivalent of `numpy.ediff1d` that takes masked
    values into account, see `numpy.ediff1d` for details.

    See Also
    --------
    numpy.ediff1d : Equivalent function for ndarrays.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.ma.array([1, 2, 4, 7, 0])
    >>> np.ma.ediff1d(arr)
    masked_array(data=[ 1,  2,  3, -7],
                 mask=False,
           fill_value=999999)

    """
    arr = ma.asanyarray(arr).flat
    ed = arr[1:] - arr[:-1]
    arrays = [ed]
    #
    if to_begin is not None:
        arrays.insert(0, to_begin)
    if to_end is not None:
        arrays.append(to_end)
    #
    if len(arrays) != 1:
        # We'll save ourselves a copy of a potentially large array in the common
        # case where neither to_begin or to_end was given.
        ed = hstack(arrays)
    #
    return ed


def unique(ar1, return_index=False, return_inverse=False):
    """
    Finds the unique elements of an array.

    Masked values are considered the same element (masked). The output array
    is always a masked array. See `numpy.unique` for more details.

    See Also
    --------
    numpy.unique : Equivalent function for ndarrays.

    Examples
    --------
    >>> import numpy as np
    >>> a = [1, 2, 1000, 2, 3]
    >>> mask = [0, 0, 1, 0, 0]
    >>> masked_a = np.ma.masked_array(a, mask)
    >>> masked_a
    masked_array(data=[1, 2, --, 2, 3],
                mask=[False, False,  True, False, False],
        fill_value=999999)
    >>> np.ma.unique(masked_a)
    masked_array(data=[1, 2, 3, --],
                mask=[False, False, False,  True],
        fill_value=999999)
    >>> np.ma.unique(masked_a, return_index=True)
    (masked_array(data=[1, 2, 3, --],
                mask=[False, False, False,  True],
        fill_value=999999), array([0, 1, 4, 2]))
    >>> np.ma.unique(masked_a, return_inverse=True)
    (masked_array(data=[1, 2, 3, --],
                mask=[False, False, False,  True],
        fill_value=999999), array([0, 1, 3, 1, 2]))
    >>> np.ma.unique(masked_a, return_index=True, return_inverse=True)
    (masked_array(data=[1, 2, 3, --],
                mask=[False, False, False,  True],
        fill_value=999999), array([0, 1, 4, 2]), array([0, 1, 3, 1, 2]))
    """
    output = np.unique(ar1,
                       return_index=return_index,
                       return_inverse=return_inverse)
    if isinstance(output, tuple):
        output = list(output)
        output[0] = output[0].view(MaskedArray)
        output = tuple(output)
    else:
        output = output.view(MaskedArray)
    return output


def intersect1d(ar1, ar2, assume_unique=False):
    """
    Returns the unique elements common to both arrays.

    Masked values are considered equal one to the other.
    The output is always a masked array.

    See `numpy.intersect1d` for more details.

    See Also
    --------
    numpy.intersect1d : Equivalent function for ndarrays.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.ma.array([1, 3, 3, 3], mask=[0, 0, 0, 1])
    >>> y = np.ma.array([3, 1, 1, 1], mask=[0, 0, 0, 1])
    >>> np.ma.intersect1d(x, y)
    masked_array(data=[1, 3, --],
                 mask=[False, False,  True],
           fill_value=999999)

    """
    if assume_unique:
        aux = ma.concatenate((ar1, ar2))
    else:
        # Might be faster than unique( intersect1d( ar1, ar2 ) )?
        aux = ma.concatenate((unique(ar1), unique(ar2)))
    aux.sort()
    return aux[:-1][aux[1:] == aux[:-1]]


def setxor1d(ar1, ar2, assume_unique=False):
    """
    Set exclusive-or of 1-D arrays with unique elements.

    The output is always a masked array. See `numpy.setxor1d` for more details.

    See Also
    --------
    numpy.setxor1d : Equivalent function for ndarrays.

    Examples
    --------
    >>> import numpy as np
    >>> ar1 = np.ma.array([1, 2, 3, 2, 4])
    >>> ar2 = np.ma.array([2, 3, 5, 7, 5])
    >>> np.ma.setxor1d(ar1, ar2)
    masked_array(data=[1, 4, 5, 7],
                 mask=False,
           fill_value=999999)

    """
    if not assume_unique:
        ar1 = unique(ar1)
        ar2 = unique(ar2)

    aux = ma.concatenate((ar1, ar2), axis=None)
    if aux.size == 0:
        return aux
    aux.sort()
    auxf = aux.filled()
#    flag = ediff1d( aux, to_end = 1, to_begin = 1 ) == 0
    flag = ma.concatenate(([True], (auxf[1:] != auxf[:-1]), [True]))
#    flag2 = ediff1d( flag ) == 0
    flag2 = (flag[1:] == flag[:-1])
    return aux[flag2]


def in1d(ar1, ar2, assume_unique=False, invert=False):
    """
    Test whether each element of an array is also present in a second
    array.

    The output is always a masked array. See `numpy.in1d` for more details.

    We recommend using :func:`isin` instead of `in1d` for new code.

    See Also
    --------
    isin       : Version of this function that preserves the shape of ar1.
    numpy.in1d : Equivalent function for ndarrays.

    Examples
    --------
    >>> import numpy as np
    >>> ar1 = np.ma.array([0, 1, 2, 5, 0])
    >>> ar2 = [0, 2]
    >>> np.ma.in1d(ar1, ar2)
    masked_array(data=[ True, False,  True, False,  True],
                 mask=False,
           fill_value=True)

    """
    if not assume_unique:
        ar1, rev_idx = unique(ar1, return_inverse=True)
        ar2 = unique(ar2)

    ar = ma.concatenate((ar1, ar2))
    # We need this to be a stable sort, so always use 'mergesort'
    # here. The values from the first array should always come before
    # the values from the second array.
    order = ar.argsort(kind='mergesort')
    sar = ar[order]
    if invert:
        bool_ar = (sar[1:] != sar[:-1])
    else:
        bool_ar = (sar[1:] == sar[:-1])
    flag = ma.concatenate((bool_ar, [invert]))
    indx = order.argsort(kind='mergesort')[:len(ar1)]

    if assume_unique:
        return flag[indx]
    else:
        return flag[indx][rev_idx]


def isin(element, test_elements, assume_unique=False, invert=False):
    """
    Calculates `element in test_elements`, broadcasting over
    `element` only.

    The output is always a masked array of the same shape as `element`.
    See `numpy.isin` for more details.

    See Also
    --------
    in1d       : Flattened version of this function.
    numpy.isin : Equivalent function for ndarrays.

    Examples
    --------
    >>> import numpy as np
    >>> element = np.ma.array([1, 2, 3, 4, 5, 6])
    >>> test_elements = [0, 2]
    >>> np.ma.isin(element, test_elements)
    masked_array(data=[False,  True, False, False, False, False],
                 mask=False,
           fill_value=True)

    """
    element = ma.asarray(element)
    return in1d(element, test_elements, assume_unique=assume_unique,
                invert=invert).reshape(element.shape)


def union1d(ar1, ar2):
    """
    Union of two arrays.

    The output is always a masked array. See `numpy.union1d` for more details.

    See Also
    --------
    numpy.union1d : Equivalent function for ndarrays.

    Examples
    --------
    >>> import numpy as np
    >>> ar1 = np.ma.array([1, 2, 3, 4])
    >>> ar2 = np.ma.array([3, 4, 5, 6])
    >>> np.ma.union1d(ar1, ar2)
    masked_array(data=[1, 2, 3, 4, 5, 6],
             mask=False,
       fill_value=999999)

    """
    return unique(ma.concatenate((ar1, ar2), axis=None))


def setdiff1d(ar1, ar2, assume_unique=False):
    """
    Set difference of 1D arrays with unique elements.

    The output is always a masked array. See `numpy.setdiff1d` for more
    details.

    See Also
    --------
    numpy.setdiff1d : Equivalent function for ndarrays.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.ma.array([1, 2, 3, 4], mask=[0, 1, 0, 1])
    >>> np.ma.setdiff1d(x, [1, 2])
    masked_array(data=[3, --],
                 mask=[False,  True],
           fill_value=999999)

    """
    if assume_unique:
        ar1 = ma.asarray(ar1).ravel()
    else:
        ar1 = unique(ar1)
        ar2 = unique(ar2)
    return ar1[in1d(ar1, ar2, assume_unique=True, invert=True)]


###############################################################################
#                                Covariance                                   #
###############################################################################


def _covhelper(x, y=None, rowvar=True, allow_masked=True):
    """
    Private function for the computation of covariance and correlation
    coefficients.

    """
    x = ma.array(x, ndmin=2, copy=True, dtype=float)
    xmask = ma.getmaskarray(x)
    # Quick exit if we can't process masked data
    if not allow_masked and xmask.any():
        raise ValueError("Cannot process masked data.")
    #
    if x.shape[0] == 1:
        rowvar = True
    # Make sure that rowvar is either 0 or 1
    rowvar = int(bool(rowvar))
    axis = 1 - rowvar
    if rowvar:
        tup = (slice(None), None)
    else:
        tup = (None, slice(None))
    #
    if y is None:
        # Check if we can guarantee that the integers in the (N - ddof)
        # normalisation can be accurately represented with single-precision
        # before computing the dot product.
        if x.shape[0] > 2 ** 24 or x.shape[1] > 2 ** 24:
            xnm_dtype = np.float64
        else:
            xnm_dtype = np.float32
        xnotmask = np.logical_not(xmask).astype(xnm_dtype)
    else:
        y = array(y, copy=False, ndmin=2, dtype=float)
        ymask = ma.getmaskarray(y)
        if not allow_masked and ymask.any():
            raise ValueError("Cannot process masked data.")
        if xmask.any() or ymask.any():
            if y.shape == x.shape:
                # Define some common mask
                common_mask = np.logical_or(xmask, ymask)
                if common_mask is not nomask:
                    xmask = x._mask = y._mask = ymask = common_mask
                    x._sharedmask = False
                    y._sharedmask = False
        x = ma.concatenate((x, y), axis)
        # Check if we can guarantee that the integers in the (N - ddof)
        # normalisation can be accurately represented with single-precision
        # before computing the dot product.
        if x.shape[0] > 2 ** 24 or x.shape[1] > 2 ** 24:
            xnm_dtype = np.float64
        else:
            xnm_dtype = np.float32
        xnotmask = np.logical_not(np.concatenate((xmask, ymask), axis)).astype(
            xnm_dtype
        )
    x -= x.mean(axis=rowvar)[tup]
    return (x, xnotmask, rowvar)


def cov(x, y=None, rowvar=True, bias=False, allow_masked=True, ddof=None):
    """
    Estimate the covariance matrix.

    Except for the handling of missing data this function does the same as
    `numpy.cov`. For more details and examples, see `numpy.cov`.

    By default, masked values are recognized as such. If `x` and `y` have the
    same shape, a common mask is allocated: if ``x[i,j]`` is masked, then
    ``y[i,j]`` will also be masked.
    Setting `allow_masked` to False will raise an exception if values are
    missing in either of the input arrays.

    Parameters
    ----------
    x : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `x` represents a variable, and each column a single
        observation of all those variables. Also see `rowvar` below.
    y : array_like, optional
        An additional set of variables and observations. `y` has the same
        shape as `x`.
    rowvar : bool, optional
        If `rowvar` is True (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
    bias : bool, optional
        Default normalization (False) is by ``(N-1)``, where ``N`` is the
        number of observations given (unbiased estimate). If `bias` is True,
        then normalization is by ``N``. This keyword can be overridden by
        the keyword ``ddof`` in numpy versions >= 1.5.
    allow_masked : bool, optional
        If True, masked values are propagated pair-wise: if a value is masked
        in `x`, the corresponding value is masked in `y`.
        If False, raises a `ValueError` exception when some values are missing.
    ddof : {None, int}, optional
        If not ``None`` normalization is by ``(N - ddof)``, where ``N`` is
        the number of observations; this overrides the value implied by
        ``bias``. The default value is ``None``.

    Raises
    ------
    ValueError
        Raised if some values are missing and `allow_masked` is False.

    See Also
    --------
    numpy.cov

    Examples
    --------
    >>> import numpy as np
    >>> x = np.ma.array([[0, 1], [1, 1]], mask=[0, 1, 0, 1])
    >>> y = np.ma.array([[1, 0], [0, 1]], mask=[0, 0, 1, 1])
    >>> np.ma.cov(x, y)
    masked_array(
    data=[[--, --, --, --],
          [--, --, --, --],
          [--, --, --, --],
          [--, --, --, --]],
    mask=[[ True,  True,  True,  True],
          [ True,  True,  True,  True],
          [ True,  True,  True,  True],
          [ True,  True,  True,  True]],
    fill_value=1e+20,
    dtype=float64)

    """
    # Check inputs
    if ddof is not None and ddof != int(ddof):
        raise ValueError("ddof must be an integer")
    # Set up ddof
    if ddof is None:
        if bias:
            ddof = 0
        else:
            ddof = 1

    (x, xnotmask, rowvar) = _covhelper(x, y, rowvar, allow_masked)
    if not rowvar:
        fact = np.dot(xnotmask.T, xnotmask) - ddof
        mask = np.less_equal(fact, 0, dtype=bool)
        with np.errstate(divide="ignore", invalid="ignore"):
            data = np.dot(filled(x.T, 0), filled(x.conj(), 0)) / fact
        result = ma.array(data, mask=mask).squeeze()
    else:
        fact = np.dot(xnotmask, xnotmask.T) - ddof
        mask = np.less_equal(fact, 0, dtype=bool)
        with np.errstate(divide="ignore", invalid="ignore"):
            data = np.dot(filled(x, 0), filled(x.T.conj(), 0)) / fact
        result = ma.array(data, mask=mask).squeeze()
    return result


def corrcoef(x, y=None, rowvar=True, bias=np._NoValue, allow_masked=True,
             ddof=np._NoValue):
    """
    Return Pearson product-moment correlation coefficients.

    Except for the handling of missing data this function does the same as
    `numpy.corrcoef`. For more details and examples, see `numpy.corrcoef`.

    Parameters
    ----------
    x : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `x` represents a variable, and each column a single
        observation of all those variables. Also see `rowvar` below.
    y : array_like, optional
        An additional set of variables and observations. `y` has the same
        shape as `x`.
    rowvar : bool, optional
        If `rowvar` is True (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
    bias : _NoValue, optional
        Has no effect, do not use.

        .. deprecated:: 1.10.0
    allow_masked : bool, optional
        If True, masked values are propagated pair-wise: if a value is masked
        in `x`, the corresponding value is masked in `y`.
        If False, raises an exception.  Because `bias` is deprecated, this
        argument needs to be treated as keyword only to avoid a warning.
    ddof : _NoValue, optional
        Has no effect, do not use.

        .. deprecated:: 1.10.0

    See Also
    --------
    numpy.corrcoef : Equivalent function in top-level NumPy module.
    cov : Estimate the covariance matrix.

    Notes
    -----
    This function accepts but discards arguments `bias` and `ddof`.  This is
    for backwards compatibility with previous versions of this function.  These
    arguments had no effect on the return values of the function and can be
    safely ignored in this and previous versions of numpy.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.ma.array([[0, 1], [1, 1]], mask=[0, 1, 0, 1])
    >>> np.ma.corrcoef(x)
    masked_array(
      data=[[--, --],
            [--, --]],
      mask=[[ True,  True],
            [ True,  True]],
      fill_value=1e+20,
      dtype=float64)

    """
    msg = 'bias and ddof have no effect and are deprecated'
    if bias is not np._NoValue or ddof is not np._NoValue:
        # 2015-03-15, 1.10
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
    # Estimate the covariance matrix.
    corr = cov(x, y, rowvar, allow_masked=allow_masked)
    # The non-masked version returns a masked value for a scalar.
    try:
        std = ma.sqrt(ma.diagonal(corr))
    except ValueError:
        return ma.MaskedConstant()
    corr /= ma.multiply.outer(std, std)
    return corr

#####--------------------------------------------------------------------------
#---- --- Concatenation helpers ---
#####--------------------------------------------------------------------------

class MAxisConcatenator(AxisConcatenator):
    """
    Translate slice objects to concatenation along an axis.

    For documentation on usage, see `mr_class`.

    See Also
    --------
    mr_class

    """
    __slots__ = ()

    concatenate = staticmethod(concatenate)

    @classmethod
    def makemat(cls, arr):
        # There used to be a view as np.matrix here, but we may eventually
        # deprecate that class. In preparation, we use the unmasked version
        # to construct the matrix (with copy=False for backwards compatibility
        # with the .view)
        data = super().makemat(arr.data, copy=False)
        return array(data, mask=arr.mask)

    def __getitem__(self, key):
        # matrix builder syntax, like 'a, b; c, d'
        if isinstance(key, str):
            raise MAError("Unavailable for masked array.")

        return super().__getitem__(key)


class mr_class(MAxisConcatenator):
    """
    Translate slice objects to concatenation along the first axis.

    This is the masked array version of `r_`.

    See Also
    --------
    r_

    Examples
    --------
    >>> import numpy as np
    >>> np.ma.mr_[np.ma.array([1,2,3]), 0, 0, np.ma.array([4,5,6])]
    masked_array(data=[1, 2, 3, ..., 4, 5, 6],
                 mask=False,
           fill_value=999999)

    """
    __slots__ = ()

    def __init__(self):
        MAxisConcatenator.__init__(self, 0)

mr_ = mr_class()


#####--------------------------------------------------------------------------
#---- Find unmasked data ---
#####--------------------------------------------------------------------------

def ndenumerate(a, compressed=True):
    """
    Multidimensional index iterator.

    Return an iterator yielding pairs of array coordinates and values,
    skipping elements that are masked. With `compressed=False`,
    `ma.masked` is yielded as the value of masked elements. This
    behavior differs from that of `numpy.ndenumerate`, which yields the
    value of the underlying data array.

    Notes
    -----
    .. versionadded:: 1.23.0

    Parameters
    ----------
    a : array_like
        An array with (possibly) masked elements.
    compressed : bool, optional
        If True (default), masked elements are skipped.

    See Also
    --------
    numpy.ndenumerate : Equivalent function ignoring any mask.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.ma.arange(9).reshape((3, 3))
    >>> a[1, 0] = np.ma.masked
    >>> a[1, 2] = np.ma.masked
    >>> a[2, 1] = np.ma.masked
    >>> a
    masked_array(
      data=[[0, 1, 2],
            [--, 4, --],
            [6, --, 8]],
      mask=[[False, False, False],
            [ True, False,  True],
            [False,  True, False]],
      fill_value=999999)
    >>> for index, x in np.ma.ndenumerate(a):
    ...     print(index, x)
    (0, 0) 0
    (0, 1) 1
    (0, 2) 2
    (1, 1) 4
    (2, 0) 6
    (2, 2) 8

    >>> for index, x in np.ma.ndenumerate(a, compressed=False):
    ...     print(index, x)
    (0, 0) 0
    (0, 1) 1
    (0, 2) 2
    (1, 0) --
    (1, 1) 4
    (1, 2) --
    (2, 0) 6
    (2, 1) --
    (2, 2) 8
    """
    for it, mask in zip(np.ndenumerate(a), getmaskarray(a).flat):
        if not mask:
            yield it
        elif not compressed:
            yield it[0], masked


def flatnotmasked_edges(a):
    """
    Find the indices of the first and last unmasked values.

    Expects a 1-D `MaskedArray`, returns None if all values are masked.

    Parameters
    ----------
    a : array_like
        Input 1-D `MaskedArray`

    Returns
    -------
    edges : ndarray or None
        The indices of first and last non-masked value in the array.
        Returns None if all values are masked.

    See Also
    --------
    flatnotmasked_contiguous, notmasked_contiguous, notmasked_edges
    clump_masked, clump_unmasked

    Notes
    -----
    Only accepts 1-D arrays.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.ma.arange(10)
    >>> np.ma.flatnotmasked_edges(a)
    array([0, 9])

    >>> mask = (a < 3) | (a > 8) | (a == 5)
    >>> a[mask] = np.ma.masked
    >>> np.array(a[~a.mask])
    array([3, 4, 6, 7, 8])

    >>> np.ma.flatnotmasked_edges(a)
    array([3, 8])

    >>> a[:] = np.ma.masked
    >>> print(np.ma.flatnotmasked_edges(a))
    None

    """
    m = getmask(a)
    if m is nomask or not np.any(m):
        return np.array([0, a.size - 1])
    unmasked = np.flatnonzero(~m)
    if len(unmasked) > 0:
        return unmasked[[0, -1]]
    else:
        return None


def notmasked_edges(a, axis=None):
    """
    Find the indices of the first and last unmasked values along an axis.

    If all values are masked, return None.  Otherwise, return a list
    of two tuples, corresponding to the indices of the first and last
    unmasked values respectively.

    Parameters
    ----------
    a : array_like
        The input array.
    axis : int, optional
        Axis along which to perform the operation.
        If None (default), applies to a flattened version of the array.

    Returns
    -------
    edges : ndarray or list
        An array of start and end indexes if there are any masked data in
        the array. If there are no masked data in the array, `edges` is a
        list of the first and last index.

    See Also
    --------
    flatnotmasked_contiguous, flatnotmasked_edges, notmasked_contiguous
    clump_masked, clump_unmasked

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(9).reshape((3, 3))
    >>> m = np.zeros_like(a)
    >>> m[1:, 1:] = 1

    >>> am = np.ma.array(a, mask=m)
    >>> np.array(am[~am.mask])
    array([0, 1, 2, 3, 6])

    >>> np.ma.notmasked_edges(am)
    array([0, 6])

    """
    a = asarray(a)
    if axis is None or a.ndim == 1:
        return flatnotmasked_edges(a)
    m = getmaskarray(a)
    idx = array(np.indices(a.shape), mask=np.asarray([m] * a.ndim))
    return [tuple([idx[i].min(axis).compressed() for i in range(a.ndim)]),
            tuple([idx[i].max(axis).compressed() for i in range(a.ndim)]), ]


def flatnotmasked_contiguous(a):
    """
    Find contiguous unmasked data in a masked array.

    Parameters
    ----------
    a : array_like
        The input array.

    Returns
    -------
    slice_list : list
        A sorted sequence of `slice` objects (start index, end index).

    See Also
    --------
    flatnotmasked_edges, notmasked_contiguous, notmasked_edges
    clump_masked, clump_unmasked

    Notes
    -----
    Only accepts 2-D arrays at most.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.ma.arange(10)
    >>> np.ma.flatnotmasked_contiguous(a)
    [slice(0, 10, None)]

    >>> mask = (a < 3) | (a > 8) | (a == 5)
    >>> a[mask] = np.ma.masked
    >>> np.array(a[~a.mask])
    array([3, 4, 6, 7, 8])

    >>> np.ma.flatnotmasked_contiguous(a)
    [slice(3, 5, None), slice(6, 9, None)]
    >>> a[:] = np.ma.masked
    >>> np.ma.flatnotmasked_contiguous(a)
    []

    """
    m = getmask(a)
    if m is nomask:
        return [slice(0, a.size)]
    i = 0
    result = []
    for (k, g) in itertools.groupby(m.ravel()):
        n = len(list(g))
        if not k:
            result.append(slice(i, i + n))
        i += n
    return result


def notmasked_contiguous(a, axis=None):
    """
    Find contiguous unmasked data in a masked array along the given axis.

    Parameters
    ----------
    a : array_like
        The input array.
    axis : int, optional
        Axis along which to perform the operation.
        If None (default), applies to a flattened version of the array, and this
        is the same as `flatnotmasked_contiguous`.

    Returns
    -------
    endpoints : list
        A list of slices (start and end indexes) of unmasked indexes
        in the array.

        If the input is 2d and axis is specified, the result is a list of lists.

    See Also
    --------
    flatnotmasked_edges, flatnotmasked_contiguous, notmasked_edges
    clump_masked, clump_unmasked

    Notes
    -----
    Only accepts 2-D arrays at most.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(12).reshape((3, 4))
    >>> mask = np.zeros_like(a)
    >>> mask[1:, :-1] = 1; mask[0, 1] = 1; mask[-1, 0] = 0
    >>> ma = np.ma.array(a, mask=mask)
    >>> ma
    masked_array(
      data=[[0, --, 2, 3],
            [--, --, --, 7],
            [8, --, --, 11]],
      mask=[[False,  True, False, False],
            [ True,  True,  True, False],
            [False,  True,  True, False]],
      fill_value=999999)
    >>> np.array(ma[~ma.mask])
    array([ 0,  2,  3,  7, 8, 11])

    >>> np.ma.notmasked_contiguous(ma)
    [slice(0, 1, None), slice(2, 4, None), slice(7, 9, None), slice(11, 12, None)]

    >>> np.ma.notmasked_contiguous(ma, axis=0)
    [[slice(0, 1, None), slice(2, 3, None)], [], [slice(0, 1, None)], [slice(0, 3, None)]]

    >>> np.ma.notmasked_contiguous(ma, axis=1)
    [[slice(0, 1, None), slice(2, 4, None)], [slice(3, 4, None)], [slice(0, 1, None), slice(3, 4, None)]]

    """
    a = asarray(a)
    nd = a.ndim
    if nd > 2:
        raise NotImplementedError("Currently limited to at most 2D array.")
    if axis is None or nd == 1:
        return flatnotmasked_contiguous(a)
    #
    result = []
    #
    other = (axis + 1) % 2
    idx = [0, 0]
    idx[axis] = slice(None, None)
    #
    for i in range(a.shape[other]):
        idx[other] = i
        result.append(flatnotmasked_contiguous(a[tuple(idx)]))
    return result


def _ezclump(mask):
    """
    Finds the clumps (groups of data with the same values) for a 1D bool array.

    Returns a series of slices.
    """
    if mask.ndim > 1:
        mask = mask.ravel()
    idx = (mask[1:] ^ mask[:-1]).nonzero()
    idx = idx[0] + 1

    if mask[0]:
        if len(idx) == 0:
            return [slice(0, mask.size)]

        r = [slice(0, idx[0])]
        r.extend((slice(left, right)
                  for left, right in zip(idx[1:-1:2], idx[2::2])))
    else:
        if len(idx) == 0:
            return []

        r = [slice(left, right) for left, right in zip(idx[:-1:2], idx[1::2])]

    if mask[-1]:
        r.append(slice(idx[-1], mask.size))
    return r


def clump_unmasked(a):
    """
    Return list of slices corresponding to the unmasked clumps of a 1-D array.
    (A "clump" is defined as a contiguous region of the array).

    Parameters
    ----------
    a : ndarray
        A one-dimensional masked array.

    Returns
    -------
    slices : list of slice
        The list of slices, one for each continuous region of unmasked
        elements in `a`.

    See Also
    --------
    flatnotmasked_edges, flatnotmasked_contiguous, notmasked_edges
    notmasked_contiguous, clump_masked

    Examples
    --------
    >>> import numpy as np
    >>> a = np.ma.masked_array(np.arange(10))
    >>> a[[0, 1, 2, 6, 8, 9]] = np.ma.masked
    >>> np.ma.clump_unmasked(a)
    [slice(3, 6, None), slice(7, 8, None)]

    """
    mask = getattr(a, '_mask', nomask)
    if mask is nomask:
        return [slice(0, a.size)]
    return _ezclump(~mask)


def clump_masked(a):
    """
    Returns a list of slices corresponding to the masked clumps of a 1-D array.
    (A "clump" is defined as a contiguous region of the array).

    Parameters
    ----------
    a : ndarray
        A one-dimensional masked array.

    Returns
    -------
    slices : list of slice
        The list of slices, one for each continuous region of masked elements
        in `a`.

    See Also
    --------
    flatnotmasked_edges, flatnotmasked_contiguous, notmasked_edges
    notmasked_contiguous, clump_unmasked

    Examples
    --------
    >>> import numpy as np
    >>> a = np.ma.masked_array(np.arange(10))
    >>> a[[0, 1, 2, 6, 8, 9]] = np.ma.masked
    >>> np.ma.clump_masked(a)
    [slice(0, 3, None), slice(6, 7, None), slice(8, 10, None)]

    """
    mask = ma.getmask(a)
    if mask is nomask:
        return []
    return _ezclump(mask)


###############################################################################
#                              Polynomial fit                                 #
###############################################################################


def vander(x, n=None):
    """
    Masked values in the input array result in rows of zeros.

    """
    _vander = np.vander(x, n)
    m = getmask(x)
    if m is not nomask:
        _vander[m] = 0
    return _vander

vander.__doc__ = ma.doc_note(np.vander.__doc__, vander.__doc__)


def polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False):
    """
    Any masked values in x is propagated in y, and vice-versa.

    """
    x = asarray(x)
    y = asarray(y)

    m = getmask(x)
    if y.ndim == 1:
        m = mask_or(m, getmask(y))
    elif y.ndim == 2:
        my = getmask(mask_rows(y))
        if my is not nomask:
            m = mask_or(m, my[:, 0])
    else:
        raise TypeError("Expected a 1D or 2D array for y!")

    if w is not None:
        w = asarray(w)
        if w.ndim != 1:
            raise TypeError("expected a 1-d array for weights")
        if w.shape[0] != y.shape[0]:
            raise TypeError("expected w and y to have the same length")
        m = mask_or(m, getmask(w))

    if m is not nomask:
        not_m = ~m
        if w is not None:
            w = w[not_m]
        return np.polyfit(x[not_m], y[not_m], deg, rcond, full, w, cov)
    else:
        return np.polyfit(x, y, deg, rcond, full, w, cov)

polyfit.__doc__ = ma.doc_note(np.polyfit.__doc__, polyfit.__doc__)
<|endoftext|>"""
=============
Masked Arrays
=============

Arrays sometimes contain invalid or missing data.  When doing operations
on such arrays, we wish to suppress invalid values, which is the purpose masked
arrays fulfill (an example of typical use is given below).

For example, examine the following array:

>>> x = np.array([2, 1, 3, np.nan, 5, 2, 3, np.nan])

When we try to calculate the mean of the data, the result is undetermined:

>>> np.mean(x)
nan

The mean is calculated using roughly ``np.sum(x)/len(x)``, but since
any number added to ``NaN`` [1]_ produces ``NaN``, this doesn't work.  Enter
masked arrays:

>>> m = np.ma.masked_array(x, np.isnan(x))
>>> m
masked_array(data=[2.0, 1.0, 3.0, --, 5.0, 2.0, 3.0, --],
             mask=[False, False, False, True, False, False, False, True],
      fill_value=1e+20)

Here, we construct a masked array that suppress all ``NaN`` values.  We
may now proceed to calculate the mean of the other values:

>>> np.mean(m)
2.6666666666666665

.. [1] Not-a-Number, a floating point value that is the result of an
       invalid operation.

.. moduleauthor:: Pierre Gerard-Marchant
.. moduleauthor:: Jarrod Millman

"""
from . import core
from .core import *

from . import extras
from .extras import *

__all__ = ['core', 'extras']
__all__ += core.__all__
__all__ += extras.__all__

from numpy._pytesttester import PytestTester
test = PytestTester(__name__)
del PytestTester
<|endoftext|>"""Miscellaneous functions for testing masked arrays and subclasses

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu
:version: $Id: testutils.py 3529 2007-11-13 08:01:14Z jarrod.millman $

"""
import operator

import numpy as np
from numpy import ndarray
import numpy._core.umath as umath
import numpy.testing
from numpy.testing import (
    assert_, assert_allclose, assert_array_almost_equal_nulp,
    assert_raises, build_err_msg
    )
from .core import mask_or, getmask, masked_array, nomask, masked, filled

__all__masked = [
    'almost', 'approx', 'assert_almost_equal', 'assert_array_almost_equal',
    'assert_array_approx_equal', 'assert_array_compare',
    'assert_array_equal', 'assert_array_less', 'assert_close',
    'assert_equal', 'assert_equal_records', 'assert_mask_equal',
    'assert_not_equal', 'fail_if_array_equal',
    ]

# Include some normal test functions to avoid breaking other projects who
# have mistakenly included them from this file. SciPy is one. That is
# unfortunate, as some of these functions are not intended to work with
# masked arrays. But there was no way to tell before.
from unittest import TestCase
__some__from_testing = [
    'TestCase', 'assert_', 'assert_allclose', 'assert_array_almost_equal_nulp',
    'assert_raises'
    ]

__all__ = __all__masked + __some__from_testing


def approx(a, b, fill_value=True, rtol=1e-5, atol=1e-8):
    """
    Returns true if all components of a and b are equal to given tolerances.

    If fill_value is True, masked values considered equal. Otherwise,
    masked values are considered unequal.  The relative error rtol should
    be positive and << 1.0 The absolute error atol comes into play for
    those elements of b that are very small or zero; it says how small a
    must be also.

    """
    m = mask_or(getmask(a), getmask(b))
    d1 = filled(a)
    d2 = filled(b)
    if d1.dtype.char == "O" or d2.dtype.char == "O":
        return np.equal(d1, d2).ravel()
    x = filled(
        masked_array(d1, copy=False, mask=m), fill_value
    ).astype(np.float64)
    y = filled(masked_array(d2, copy=False, mask=m), 1).astype(np.float64)
    d = np.less_equal(umath.absolute(x - y), atol + rtol * umath.absolute(y))
    return d.ravel()


def almost(a, b, decimal=6, fill_value=True):
    """
    Returns True if a and b are equal up to decimal places.

    If fill_value is True, masked values considered equal. Otherwise,
    masked values are considered unequal.

    """
    m = mask_or(getmask(a), getmask(b))
    d1 = filled(a)
    d2 = filled(b)
    if d1.dtype.char == "O" or d2.dtype.char == "O":
        return np.equal(d1, d2).ravel()
    x = filled(
        masked_array(d1, copy=False, mask=m), fill_value
    ).astype(np.float64)
    y = filled(masked_array(d2, copy=False, mask=m), 1).astype(np.float64)
    d = np.around(np.abs(x - y), decimal) <= 10.0 ** (-decimal)
    return d.ravel()


def _assert_equal_on_sequences(actual, desired, err_msg=''):
    """
    Asserts the equality of two non-array sequences.

    """
    assert_equal(len(actual), len(desired), err_msg)
    for k in range(len(desired)):
        assert_equal(actual[k], desired[k], f'item={k!r}\n{err_msg}')
    return


def assert_equal_records(a, b):
    """
    Asserts that two records are equal.

    Pretty crude for now.

    """
    assert_equal(a.dtype, b.dtype)
    for f in a.dtype.names:
        (af, bf) = (operator.getitem(a, f), operator.getitem(b, f))
        if not (af is masked) and not (bf is masked):
            assert_equal(operator.getitem(a, f), operator.getitem(b, f))
    return


def assert_equal(actual, desired, err_msg=''):
    """
    Asserts that two items are equal.

    """
    # Case #1: dictionary .....
    if isinstance(desired, dict):
        if not isinstance(actual, dict):
            raise AssertionError(repr(type(actual)))
        assert_equal(len(actual), len(desired), err_msg)
        for k, i in desired.items():
            if k not in actual:
                raise AssertionError(f"{k} not in {actual}")
            assert_equal(actual[k], desired[k], f'key={k!r}\n{err_msg}')
        return
    # Case #2: lists .....
    if isinstance(desired, (list, tuple)) and isinstance(actual, (list, tuple)):
        return _assert_equal_on_sequences(actual, desired, err_msg='')
    if not (isinstance(actual, ndarray) or isinstance(desired, ndarray)):
        msg = build_err_msg([actual, desired], err_msg,)
        if not desired == actual:
            raise AssertionError(msg)
        return
    # Case #4. arrays or equivalent
    if ((actual is masked) and not (desired is masked)) or \
            ((desired is masked) and not (actual is masked)):
        msg = build_err_msg([actual, desired],
                            err_msg, header='', names=('x', 'y'))
        raise ValueError(msg)
    actual = np.asanyarray(actual)
    desired = np.asanyarray(desired)
    (actual_dtype, desired_dtype) = (actual.dtype, desired.dtype)
    if actual_dtype.char == "S" and desired_dtype.char == "S":
        return _assert_equal_on_sequences(actual.tolist(),
                                          desired.tolist(),
                                          err_msg='')
    return assert_array_equal(actual, desired, err_msg)


def fail_if_equal(actual, desired, err_msg='',):
    """
    Raises an assertion error if two items are equal.

    """
    if isinstance(desired, dict):
        if not isinstance(actual, dict):
            raise AssertionError(repr(type(actual)))
        fail_if_equal(len(actual), len(desired), err_msg)
        for k, i in desired.items():
            if k not in actual:
                raise AssertionError(repr(k))
            fail_if_equal(actual[k], desired[k], f'key={k!r}\n{err_msg}')
        return
    if isinstance(desired, (list, tuple)) and isinstance(actual, (list, tuple)):
        fail_if_equal(len(actual), len(desired), err_msg)
        for k in range(len(desired)):
            fail_if_equal(actual[k], desired[k], f'item={k!r}\n{err_msg}')
        return
    if isinstance(actual, np.ndarray) or isinstance(desired, np.ndarray):
        return fail_if_array_equal(actual, desired, err_msg)
    msg = build_err_msg([actual, desired], err_msg)
    if not desired != actual:
        raise AssertionError(msg)


assert_not_equal = fail_if_equal


def assert_almost_equal(actual, desired, decimal=7, err_msg='', verbose=True):
    """
    Asserts that two items are almost equal.

    The test is equivalent to abs(desired-actual) < 0.5 * 10**(-decimal).

    """
    if isinstance(actual, np.ndarray) or isinstance(desired, np.ndarray):
        return assert_array_almost_equal(actual, desired, decimal=decimal,
                                         err_msg=err_msg, verbose=verbose)
    msg = build_err_msg([actual, desired],
                        err_msg=err_msg, verbose=verbose)
    if not round(abs(desired - actual), decimal) == 0:
        raise AssertionError(msg)


assert_close = assert_almost_equal


def assert_array_compare(comparison, x, y, err_msg='', verbose=True, header='',
                         fill_value=True):
    """
    Asserts that comparison between two masked arrays is satisfied.

    The comparison is elementwise.

    """
    # Allocate a common mask and refill
    m = mask_or(getmask(x), getmask(y))
    x = masked_array(x, copy=False, mask=m, keep_mask=False, subok=False)
    y = masked_array(y, copy=False, mask=m, keep_mask=False, subok=False)
    if ((x is masked) and not (y is masked)) or \
            ((y is masked) and not (x is masked)):
        msg = build_err_msg([x, y], err_msg=err_msg, verbose=verbose,
                            header=header, names=('x', 'y'))
        raise ValueError(msg)
    # OK, now run the basic tests on filled versions
    return np.testing.assert_array_compare(comparison,
                                           x.filled(fill_value),
                                           y.filled(fill_value),
                                           err_msg=err_msg,
                                           verbose=verbose, header=header)


def assert_array_equal(x, y, err_msg='', verbose=True):
    """
    Checks the elementwise equality of two masked arrays.

    """
    assert_array_compare(operator.__eq__, x, y,
                         err_msg=err_msg, verbose=verbose,
                         header='Arrays are not equal')


def fail_if_array_equal(x, y, err_msg='', verbose=True):
    """
    Raises an assertion error if two masked arrays are not equal elementwise.

    """
    def compare(x, y):
        return (not np.all(approx(x, y)))
    assert_array_compare(compare, x, y, err_msg=err_msg, verbose=verbose,
                         header='Arrays are not equal')


def assert_array_approx_equal(x, y, decimal=6, err_msg='', verbose=True):
    """
    Checks the equality of two masked arrays, up to given number odecimals.

    The equality is checked elementwise.

    """
    def compare(x, y):
        "Returns the result of the loose comparison between x and y)."
        return approx(x, y, rtol=10. ** -decimal)
    assert_array_compare(compare, x, y, err_msg=err_msg, verbose=verbose,
                         header='Arrays are not almost equal')


def assert_array_almost_equal(x, y, decimal=6, err_msg='', verbose=True):
    """
    Checks the equality of two masked arrays, up to given number odecimals.

    The equality is checked elementwise.

    """
    def compare(x, y):
        "Returns the result of the loose comparison between x and y)."
        return almost(x, y, decimal)
    assert_array_compare(compare, x, y, err_msg=err_msg, verbose=verbose,
                         header='Arrays are not almost equal')


def assert_array_less(x, y, err_msg='', verbose=True):
    """
    Checks that x is smaller than y elementwise.

    """
    assert_array_compare(operator.__lt__, x, y,
                         err_msg=err_msg, verbose=verbose,
                         header='Arrays are not less-ordered')


def assert_mask_equal(m1, m2, err_msg=''):
    """
    Asserts the equality of two masks.

    """
    if m1 is nomask:
        assert_(m2 is nomask)
    if m2 is nomask:
        assert_(m1 is nomask)
    assert_array_equal(m1, m2, err_msg=err_msg)
<|endoftext|>import functools
import timeit

import numpy as np
import numpy._core.fromnumeric as fromnumeric

from numpy.testing import build_err_msg


pi = np.pi

class ModuleTester:
    def __init__(self, module):
        self.module = module
        self.allequal = module.allequal
        self.arange = module.arange
        self.array = module.array
        self.concatenate = module.concatenate
        self.count = module.count
        self.equal = module.equal
        self.filled = module.filled
        self.getmask = module.getmask
        self.getmaskarray = module.getmaskarray
        self.id = id
        self.inner = module.inner
        self.make_mask = module.make_mask
        self.masked = module.masked
        self.masked_array = module.masked_array
        self.masked_values = module.masked_values
        self.mask_or = module.mask_or
        self.nomask = module.nomask
        self.ones = module.ones
        self.outer = module.outer
        self.repeat = module.repeat
        self.resize = module.resize
        self.sort = module.sort
        self.take = module.take
        self.transpose = module.transpose
        self.zeros = module.zeros
        self.MaskType = module.MaskType
        try:
            self.umath = module.umath
        except AttributeError:
            self.umath = module.core.umath
        self.testnames = []

    def assert_array_compare(self, comparison, x, y, err_msg='', header='',
                         fill_value=True):
        """
        Assert that a comparison of two masked arrays is satisfied elementwise.

        """
        xf = self.filled(x)
        yf = self.filled(y)
        m = self.mask_or(self.getmask(x), self.getmask(y))

        x = self.filled(self.masked_array(xf, mask=m), fill_value)
        y = self.filled(self.masked_array(yf, mask=m), fill_value)
        if (x.dtype.char != "O"):
            x = x.astype(np.float64)
            if isinstance(x, np.ndarray) and x.size > 1:
                x[np.isnan(x)] = 0
            elif np.isnan(x):
                x = 0
        if (y.dtype.char != "O"):
            y = y.astype(np.float64)
            if isinstance(y, np.ndarray) and y.size > 1:
                y[np.isnan(y)] = 0
            elif np.isnan(y):
                y = 0
        try:
            cond = (x.shape == () or y.shape == ()) or x.shape == y.shape
            if not cond:
                msg = build_err_msg([x, y],
                                    err_msg
                                    + f'\n(shapes {x.shape}, {y.shape} mismatch)',
                                    header=header,
                                    names=('x', 'y'))
                assert cond, msg
            val = comparison(x, y)
            if m is not self.nomask and fill_value:
                val = self.masked_array(val, mask=m)
            if isinstance(val, bool):
                cond = val
                reduced = [0]
            else:
                reduced = val.ravel()
                cond = reduced.all()
                reduced = reduced.tolist()
            if not cond:
                match = 100-100.0*reduced.count(1)/len(reduced)
                msg = build_err_msg([x, y],
                                    err_msg
                                    + '\n(mismatch %s%%)' % (match,),
                                    header=header,
                                    names=('x', 'y'))
                assert cond, msg
        except ValueError as e:
            msg = build_err_msg([x, y], err_msg, header=header, names=('x', 'y'))
            raise ValueError(msg) from e

    def assert_array_equal(self, x, y, err_msg=''):
        """
        Checks the elementwise equality of two masked arrays.

        """
        self.assert_array_compare(self.equal, x, y, err_msg=err_msg,
                                  header='Arrays are not equal')

    @np.errstate(all='ignore')
    def test_0(self):
        """
        Tests creation

        """
        x = np.array([1., 1., 1., -2., pi/2.0, 4., 5., -10., 10., 1., 2., 3.])
        m = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        xm = self.masked_array(x, mask=m)
        xm[0]

    @np.errstate(all='ignore')
    def test_1(self):
        """
        Tests creation

        """
        x = np.array([1., 1., 1., -2., pi/2.0, 4., 5., -10., 10., 1., 2., 3.])
        y = np.array([5., 0., 3., 2., -1., -4., 0., -10., 10., 1., 0., 3.])
        m1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        m2 = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1]
        xm = self.masked_array(x, mask=m1)
        ym = self.masked_array(y, mask=m2)
        xf = np.where(m1, 1.e+20, x)
        xm.set_fill_value(1.e+20)

        assert (xm-ym).filled(0).any()
        s = x.shape
        assert xm.size == functools.reduce(lambda x, y: x*y, s)
        assert self.count(xm) == len(m1) - functools.reduce(lambda x, y: x+y, m1)

        for s in [(4, 3), (6, 2)]:
            x.shape = s
            y.shape = s
            xm.shape = s
            ym.shape = s
            xf.shape = s
            assert self.count(xm) == len(m1) - functools.reduce(lambda x, y: x+y, m1)

    @np.errstate(all='ignore')
    def test_2(self):
        """
        Tests conversions and indexing.

        """
        x1 = np.array([1, 2, 4, 3])
        x2 = self.array(x1, mask=[1, 0, 0, 0])
        x3 = self.array(x1, mask=[0, 1, 0, 1])
        x4 = self.array(x1)
        # test conversion to strings, no errors
        str(x2)
        repr(x2)
        # tests of indexing
        assert type(x2[1]) is type(x1[1])
        assert x1[1] == x2[1]
        x1[2] = 9
        x2[2] = 9
        self.assert_array_equal(x1, x2)
        x1[1:3] = 99
        x2[1:3] = 99
        x2[1] = self.masked
        x2[1:3] = self.masked
        x2[:] = x1
        x2[1] = self.masked
        x3[:] = self.masked_array([1, 2, 3, 4], [0, 1, 1, 0])
        x4[:] = self.masked_array([1, 2, 3, 4], [0, 1, 1, 0])
        x1 = np.arange(5)*1.0
        x2 = self.masked_values(x1, 3.0)
        x1 = self.array([1, 'hello', 2, 3], object)
        x2 = np.array([1, 'hello', 2, 3], object)
        # check that no error occurs.
        x1[1]
        x2[1]
        assert x1[1:1].shape == (0,)
        # Tests copy-size
        n = [0, 0, 1, 0, 0]
        m = self.make_mask(n)
        m2 = self.make_mask(m)
        assert(m is m2)
        m3 = self.make_mask(m, copy=1)
        assert(m is not m3)

    @np.errstate(all='ignore')
    def test_3(self):
        """
        Tests resize/repeat

        """
        x4 = self.arange(4)
        x4[2] = self.masked
        y4 = self.resize(x4, (8,))
        assert self.allequal(self.concatenate([x4, x4]), y4)
        assert self.allequal(self.getmask(y4), [0, 0, 1, 0, 0, 0, 1, 0])
        y5 = self.repeat(x4, (2, 2, 2, 2), axis=0)
        self.assert_array_equal(y5, [0, 0, 1, 1, 2, 2, 3, 3])
        y6 = self.repeat(x4, 2, axis=0)
        assert self.allequal(y5, y6)
        y7 = x4.repeat((2, 2, 2, 2), axis=0)
        assert self.allequal(y5, y7)
        y8 = x4.repeat(2, 0)
        assert self.allequal(y5, y8)

    @np.errstate(all='ignore')
    def test_4(self):
        """
        Test of take, transpose, inner, outer products.

        """
        x = self.arange(24)
        y = np.arange(24)
        x[5:6] = self.masked
        x = x.reshape(2, 3, 4)
        y = y.reshape(2, 3, 4)
        assert self.allequal(np.transpose(y, (2, 0, 1)), self.transpose(x, (2, 0, 1)))
        assert self.allequal(np.take(y, (2, 0, 1), 1), self.take(x, (2, 0, 1), 1))
        assert self.allequal(np.inner(self.filled(x, 0), self.filled(y, 0)),
                            self.inner(x, y))
        assert self.allequal(np.outer(self.filled(x, 0), self.filled(y, 0)),
                            self.outer(x, y))
        y = self.array(['abc', 1, 'def', 2, 3], object)
        y[2] = self.masked
        t = self.take(y, [0, 3, 4])
        assert t[0] == 'abc'
        assert t[1] == 2
        assert t[2] == 3

    @np.errstate(all='ignore')
    def test_5(self):
        """
        Tests inplace w/ scalar

        """
        x = self.arange(10)
        y = self.arange(10)
        xm = self.arange(10)
        xm[2] = self.masked
        x += 1
        assert self.allequal(x, y+1)
        xm += 1
        assert self.allequal(xm, y+1)

        x = self.arange(10)
        xm = self.arange(10)
        xm[2] = self.masked
        x -= 1
        assert self.allequal(x, y-1)
        xm -= 1
        assert self.allequal(xm, y-1)

        x = self.arange(10)*1.0
        xm = self.arange(10)*1.0
        xm[2] = self.masked
        x *= 2.0
        assert self.allequal(x, y*2)
        xm *= 2.0
        assert self.allequal(xm, y*2)

        x = self.arange(10)*2
        xm = self.arange(10)*2
        xm[2] = self.masked
        x /= 2
        assert self.allequal(x, y)
        xm /= 2
        assert self.allequal(xm, y)

        x = self.arange(10)*1.0
        xm = self.arange(10)*1.0
        xm[2] = self.masked
        x /= 2.0
        assert self.allequal(x, y/2.0)
        xm /= self.arange(10)
        self.assert_array_equal(xm, self.ones((10,)))

        x = self.arange(10).astype(np.float64)
        xm = self.arange(10)
        xm[2] = self.masked
        x += 1.
        assert self.allequal(x, y + 1.)

    @np.errstate(all='ignore')
    def test_6(self):
        """
        Tests inplace w/ array

        """
        x = self.arange(10, dtype=np.float64)
        y = self.arange(10)
        xm = self.arange(10, dtype=np.float64)
        xm[2] = self.masked
        m = xm.mask
        a = self.arange(10, dtype=np.float64)
        a[-1] = self.masked
        x += a
        xm += a
        assert self.allequal(x, y+a)
        assert self.allequal(xm, y+a)
        assert self.allequal(xm.mask, self.mask_or(m, a.mask))

        x = self.arange(10, dtype=np.float64)
        xm = self.arange(10, dtype=np.float64)
        xm[2] = self.masked
        m = xm.mask
        a = self.arange(10, dtype=np.float64)
        a[-1] = self.masked
        x -= a
        xm -= a
        assert self.allequal(x, y-a)
        assert self.allequal(xm, y-a)
        assert self.allequal(xm.mask, self.mask_or(m, a.mask))

        x = self.arange(10, dtype=np.float64)
        xm = self.arange(10, dtype=np.float64)
        xm[2] = self.masked
        m = xm.mask
        a = self.arange(10, dtype=np.float64)
        a[-1] = self.masked
        x *= a
        xm *= a
        assert self.allequal(x, y*a)
        assert self.allequal(xm, y*a)
        assert self.allequal(xm.mask, self.mask_or(m, a.mask))

        x = self.arange(10, dtype=np.float64)
        xm = self.arange(10, dtype=np.float64)
        xm[2] = self.masked
        m = xm.mask
        a = self.arange(10, dtype=np.float64)
        a[-1] = self.masked
        x /= a
        xm /= a

    @np.errstate(all='ignore')
    def test_7(self):
        "Tests ufunc"
        d = (self.array([1.0, 0, -1, pi/2]*2, mask=[0, 1]+[0]*6),
             self.array([1.0, 0, -1, pi/2]*2, mask=[1, 0]+[0]*6),)
        for f in ['sqrt', 'log', 'log10', 'exp', 'conjugate',
#                  'sin', 'cos', 'tan',
#                  'arcsin', 'arccos', 'arctan',
#                  'sinh', 'cosh', 'tanh',
#                  'arcsinh',
#                  'arccosh',
#                  'arctanh',
#                  'absolute', 'fabs', 'negative',
#                  # 'nonzero', 'around',
#                  'floor', 'ceil',
#                  # 'sometrue', 'alltrue',
#                  'logical_not',
#                  'add', 'subtract', 'multiply',
#                  'divide', 'true_divide', 'floor_divide',
#                  'remainder', 'fmod', 'hypot', 'arctan2',
#                  'equal', 'not_equal', 'less_equal', 'greater_equal',
#                  'less', 'greater',
#                  'logical_and', 'logical_or', 'logical_xor',
                  ]:
            try:
                uf = getattr(self.umath, f)
            except AttributeError:
                uf = getattr(fromnumeric, f)
            mf = getattr(self.module, f)
            args = d[:uf.nin]
            ur = uf(*args)
            mr = mf(*args)
            self.assert_array_equal(ur.filled(0), mr.filled(0), f)
            self.assert_array_equal(ur._mask, mr._mask)

    @np.errstate(all='ignore')
    def test_99(self):
        # test average
        ott = self.array([0., 1., 2., 3.], mask=[1, 0, 0, 0])
        self.assert_array_equal(2.0, self.average(ott, axis=0))
        self.assert_array_equal(2.0, self.average(ott, weights=[1., 1., 2., 1.]))
        result, wts = self.average(ott, weights=[1., 1., 2., 1.], returned=1)
        self.assert_array_equal(2.0, result)
        assert(wts == 4.0)
        ott[:] = self.masked
        assert(self.average(ott, axis=0) is self.masked)
        ott = self.array([0., 1., 2., 3.], mask=[1, 0, 0, 0])
        ott = ott.reshape(2, 2)
        ott[:, 1] = self.masked
        self.assert_array_equal(self.average(ott, axis=0), [2.0, 0.0])
        assert(self.average(ott, axis=1)[0] is self.masked)
        self.assert_array_equal([2., 0.], self.average(ott, axis=0))
        result, wts = self.average(ott, axis=0, returned=1)
        self.assert_array_equal(wts, [1., 0.])
        w1 = [0, 1, 1, 1, 1, 0]
        w2 = [[0, 1, 1, 1, 1, 0], [1, 0, 0, 0, 0, 1]]
        x = self.arange(6)
        self.assert_array_equal(self.average(x, axis=0), 2.5)
        self.assert_array_equal(self.average(x, axis=0, weights=w1), 2.5)
        y = self.array([self.arange(6), 2.0*self.arange(6)])
        self.assert_array_equal(self.average(y, None), np.add.reduce(np.arange(6))*3./12.)
        self.assert_array_equal(self.average(y, axis=0), np.arange(6) * 3./2.)
        self.assert_array_equal(self.average(y, axis=1), [self.average(x, axis=0), self.average(x, axis=0) * 2.0])
        self.assert_array_equal(self.average(y, None, weights=w2), 20./6.)
        self.assert_array_equal(self.average(y, axis=0, weights=w2), [0., 1., 2., 3., 4., 10.])
        self.assert_array_equal(self.average(y, axis=1), [self.average(x, axis=0), self.average(x, axis=0) * 2.0])
        m1 = self.zeros(6)
        m2 = [0, 0, 1, 1, 0, 0]
        m3 = [[0, 0, 1, 1, 0, 0], [0, 1, 1, 1, 1, 0]]
        m4 = self.ones(6)
        m5 = [0, 1, 1, 1, 1, 1]
        self.assert_array_equal(self.average(self.masked_array(x, m1), axis=0), 2.5)
        self.assert_array_equal(self.average(self.masked_array(x, m2), axis=0), 2.5)
        self.assert_array_equal(self.average(self.masked_array(x, m5), axis=0), 0.0)
        self.assert_array_equal(self.count(self.average(self.masked_array(x, m4), axis=0)), 0)
        z = self.masked_array(y, m3)
        self.assert_array_equal(self.average(z, None), 20./6.)
        self.assert_array_equal(self.average(z, axis=0), [0., 1., 99., 99., 4.0, 7.5])
        self.assert_array_equal(self.average(z, axis=1), [2.5, 5.0])
        self.assert_array_equal(self.average(z, axis=0, weights=w2), [0., 1., 99., 99., 4.0, 10.0])

    @np.errstate(all='ignore')
    def test_A(self):
        x = self.arange(24)
        x[5:6] = self.masked
        x = x.reshape(2, 3, 4)


if __name__ == '__main__':
    setup_base = ("from __main__ import ModuleTester \n"
                  "import numpy\n"
                  "tester = ModuleTester(module)\n")
    setup_cur = "import numpy.ma.core as module\n" + setup_base
    (nrepeat, nloop) = (10, 10)

    for i in range(1, 8):
        func = 'tester.test_%i()' % i
        cur = timeit.Timer(func, setup_cur).repeat(nrepeat, nloop*10)
        cur = np.sort(cur)
        print("#%i" % i + 50*'.')
        print(eval("ModuleTester.test_%i.__doc__" % i))
        print(f'core_current : {cur[0]:.3f} - {cur[1]:.3f}')
<|endoftext|>"""
========================
Random Number Generation
========================

Use ``default_rng()`` to create a `Generator` and call its methods.

=============== =========================================================
Generator
--------------- ---------------------------------------------------------
Generator       Class implementing all of the random number distributions
default_rng     Default constructor for ``Generator``
=============== =========================================================

============================================= ===
BitGenerator Streams that work with Generator
--------------------------------------------- ---
MT19937
PCG64
PCG64DXSM
Philox
SFC64
============================================= ===

============================================= ===
Getting entropy to initialize a BitGenerator
--------------------------------------------- ---
SeedSequence
============================================= ===


Legacy
------

For backwards compatibility with previous versions of numpy before 1.17, the
various aliases to the global `RandomState` methods are left alone and do not
use the new `Generator` API.

==================== =========================================================
Utility functions
-------------------- ---------------------------------------------------------
random               Uniformly distributed floats over ``[0, 1)``
bytes                Uniformly distributed random bytes.
permutation          Randomly permute a sequence / generate a random sequence.
shuffle              Randomly permute a sequence in place.
choice               Random sample from 1-D array.
==================== =========================================================

==================== =========================================================
Compatibility
functions - removed
in the new API
-------------------- ---------------------------------------------------------
rand                 Uniformly distributed values.
randn                Normally distributed values.
ranf                 Uniformly distributed floating point numbers.
random_integers      Uniformly distributed integers in a given range.
                     (deprecated, use ``integers(..., closed=True)`` instead)
random_sample        Alias for `random_sample`
randint              Uniformly distributed integers in a given range
seed                 Seed the legacy random number generator.
==================== =========================================================

==================== =========================================================
Univariate
distributions
-------------------- ---------------------------------------------------------
beta                 Beta distribution over ``[0, 1]``.
binomial             Binomial distribution.
chisquare            :math:`\\chi^2` distribution.
exponential          Exponential distribution.
f                    F (Fisher-Snedecor) distribution.
gamma                Gamma distribution.
geometric            Geometric distribution.
gumbel               Gumbel distribution.
hypergeometric       Hypergeometric distribution.
laplace              Laplace distribution.
logistic             Logistic distribution.
lognormal            Log-normal distribution.
logseries            Logarithmic series distribution.
negative_binomial    Negative binomial distribution.
noncentral_chisquare Non-central chi-square distribution.
noncentral_f         Non-central F distribution.
normal               Normal / Gaussian distribution.
pareto               Pareto distribution.
poisson              Poisson distribution.
power                Power distribution.
rayleigh             Rayleigh distribution.
triangular           Triangular distribution.
uniform              Uniform distribution.
vonmises             Von Mises circular distribution.
wald                 Wald (inverse Gaussian) distribution.
weibull              Weibull distribution.
zipf                 Zipf's distribution over ranked data.
==================== =========================================================

==================== ==========================================================
Multivariate
distributions
-------------------- ----------------------------------------------------------
dirichlet            Multivariate generalization of Beta distribution.
multinomial          Multivariate generalization of the binomial distribution.
multivariate_normal  Multivariate generalization of the normal distribution.
==================== ==========================================================

==================== =========================================================
Standard
distributions
-------------------- ---------------------------------------------------------
standard_cauchy      Standard Cauchy-Lorentz distribution.
standard_exponential Standard exponential distribution.
standard_gamma       Standard Gamma distribution.
standard_normal      Standard normal distribution.
standard_t           Standard Student's t-distribution.
==================== =========================================================

==================== =========================================================
Internal functions
-------------------- ---------------------------------------------------------
get_state            Get tuple representing internal state of generator.
set_state            Set state of generator.
==================== =========================================================


"""
__all__ = [
    'beta',
    'binomial',
    'bytes',
    'chisquare',
    'choice',
    'dirichlet',
    'exponential',
    'f',
    'gamma',
    'geometric',
    'get_state',
    'gumbel',
    'hypergeometric',
    'laplace',
    'logistic',
    'lognormal',
    'logseries',
    'multinomial',
    'multivariate_normal',
    'negative_binomial',
    'noncentral_chisquare',
    'noncentral_f',
    'normal',
    'pareto',
    'permutation',
    'poisson',
    'power',
    'rand',
    'randint',
    'randn',
    'random',
    'random_integers',
    'random_sample',
    'ranf',
    'rayleigh',
    'sample',
    'seed',
    'set_state',
    'shuffle',
    'standard_cauchy',
    'standard_exponential',
    'standard_gamma',
    'standard_normal',
    'standard_t',
    'triangular',
    'uniform',
    'vonmises',
    'wald',
    'weibull',
    'zipf',
]

# add these for module-freeze analysis (like PyInstaller)
from . import _pickle
from . import _common
from . import _bounded_integers

from ._generator import Generator, default_rng
from .bit_generator import SeedSequence, BitGenerator
from ._mt19937 import MT19937
from ._pcg64 import PCG64, PCG64DXSM
from ._philox import Philox
from ._sfc64 import SFC64
from .mtrand import *

__all__ += ['Generator', 'RandomState', 'SeedSequence', 'MT19937',
            'Philox', 'PCG64', 'PCG64DXSM', 'SFC64', 'default_rng',
            'BitGenerator']


def __RandomState_ctor():
    """Return a RandomState instance.

    This function exists solely to assist (un)pickling.

    Note that the state of the RandomState returned here is irrelevant, as this
    function's entire purpose is to return a newly allocated RandomState whose
    state pickle can set.  Consequently the RandomState returned by this function
    is a freshly allocated copy with a seed=0.

    See https://github.com/numpy/numpy/issues/4763 for a detailed discussion

    """
    return RandomState(seed=0)


from numpy._pytesttester import PytestTester
test = PytestTester(__name__)
del PytestTester
<|endoftext|>from .bit_generator import BitGenerator
from .mtrand import RandomState
from ._philox import Philox
from ._pcg64 import PCG64, PCG64DXSM
from ._sfc64 import SFC64

from ._generator import Generator
from ._mt19937 import MT19937

BitGenerators = {'MT19937': MT19937,
                 'PCG64': PCG64,
                 'PCG64DXSM': PCG64DXSM,
                 'Philox': Philox,
                 'SFC64': SFC64,
                 }


def __bit_generator_ctor(bit_generator: str | type[BitGenerator] = 'MT19937'):
    """
    Pickling helper function that returns a bit generator object

    Parameters
    ----------
    bit_generator : type[BitGenerator] or str
        BitGenerator class or string containing the name of the BitGenerator

    Returns
    -------
    BitGenerator
        BitGenerator instance
    """
    if isinstance(bit_generator, type):
        bit_gen_class = bit_generator
    elif bit_generator in BitGenerators:
        bit_gen_class = BitGenerators[bit_generator]
    else:
        raise ValueError(
            str(bit_generator) + ' is not a known BitGenerator module.'
        )

    return bit_gen_class()


def __generator_ctor(bit_generator_name="MT19937",
                     bit_generator_ctor=__bit_generator_ctor):
    """
    Pickling helper function that returns a Generator object

    Parameters
    ----------
    bit_generator_name : str or BitGenerator
        String containing the core BitGenerator's name or a
        BitGenerator instance
    bit_generator_ctor : callable, optional
        Callable function that takes bit_generator_name as its only argument
        and returns an instantized bit generator.

    Returns
    -------
    rg : Generator
        Generator using the named core BitGenerator
    """
    if isinstance(bit_generator_name, BitGenerator):
        return Generator(bit_generator_name)
    # Legacy path that uses a bit generator name and ctor
    return Generator(bit_generator_ctor(bit_generator_name))


def __randomstate_ctor(bit_generator_name="MT19937",
                       bit_generator_ctor=__bit_generator_ctor):
    """
    Pickling helper function that returns a legacy RandomState-like object

    Parameters
    ----------
    bit_generator_name : str
        String containing the core BitGenerator's name
    bit_generator_ctor : callable, optional
        Callable function that takes bit_generator_name as its only argument
        and returns an instantized bit generator.

    Returns
    -------
    rs : RandomState
        Legacy RandomState using the named core BitGenerator
    """
    if isinstance(bit_generator_name, BitGenerator):
        return RandomState(bit_generator_name)
    return RandomState(bit_generator_ctor(bit_generator_name))
<|endoftext|>import os


def parse_distributions_h(ffi, inc_dir):
    """
    Parse distributions.h located in inc_dir for CFFI, filling in the ffi.cdef

    Read the function declarations without the "#define ..." macros that will
    be filled in when loading the library.
    """

    with open(os.path.join(inc_dir, 'random', 'bitgen.h')) as fid:
        s = []
        for line in fid:
            # massage the include file
            if line.strip().startswith('#'):
                continue
            s.append(line)
        ffi.cdef('\n'.join(s))

    with open(os.path.join(inc_dir, 'random', 'distributions.h')) as fid:
        s = []
        in_skip = 0
        ignoring = False
        for line in fid:
            # check for and remove extern "C" guards
            if ignoring:
                if line.strip().startswith('#endif'):
                    ignoring = False
                continue
            if line.strip().startswith('#ifdef __cplusplus'):
                ignoring = True

            # massage the include file
            if line.strip().startswith('#'):
                continue

            # skip any inlined function definition
            # which starts with 'static inline xxx(...) {'
            # and ends with a closing '}'
            if line.strip().startswith('static inline'):
                in_skip += line.count('{')
                continue
            elif in_skip > 0:
                in_skip += line.count('{')
                in_skip -= line.count('}')
                continue

            # replace defines with their value or remove them
            line = line.replace('DECLDIR', '')
            line = line.replace('RAND_INT_TYPE', 'int64_t')
            s.append(line)
        ffi.cdef('\n'.join(s))

<|endoftext|>"""
Use cffi to access any of the underlying C functions from distributions.h
"""
import os
import numpy as np
import cffi
from .parse import parse_distributions_h
ffi = cffi.FFI()

inc_dir = os.path.join(np.get_include(), 'numpy')

# Basic numpy types
ffi.cdef('''
    typedef intptr_t npy_intp;
    typedef unsigned char npy_bool;

''')

parse_distributions_h(ffi, inc_dir)

lib = ffi.dlopen(np.random._generator.__file__)

# Compare the distributions.h random_standard_normal_fill to
# Generator.standard_random
bit_gen = np.random.PCG64()
rng = np.random.Generator(bit_gen)
state = bit_gen.state

interface = rng.bit_generator.cffi
n = 100
vals_cffi = ffi.new('double[%d]' % n)
lib.random_standard_normal_fill(interface.bit_generator, n, vals_cffi)

# reset the state
bit_gen.state = state

vals = rng.standard_normal(n)

for i in range(n):
    assert vals[i] == vals_cffi[i]
<|endoftext|>r"""
Building the required library in this example requires a source distribution
of NumPy or clone of the NumPy git repository since distributions.c is not
included in binary distributions.

On *nix, execute in numpy/random/src/distributions

export ${PYTHON_VERSION}=3.8 # Python version
export PYTHON_INCLUDE=#path to Python's include folder, usually \
    ${PYTHON_HOME}/include/python${PYTHON_VERSION}m
export NUMPY_INCLUDE=#path to numpy's include folder, usually \
    ${PYTHON_HOME}/lib/python${PYTHON_VERSION}/site-packages/numpy/_core/include
gcc -shared -o libdistributions.so -fPIC distributions.c \
    -I${NUMPY_INCLUDE} -I${PYTHON_INCLUDE}
mv libdistributions.so ../../_examples/numba/

On Windows

rem PYTHON_HOME and PYTHON_VERSION are setup dependent, this is an example
set PYTHON_HOME=c:\Anaconda
set PYTHON_VERSION=38
cl.exe /LD .\distributions.c -DDLL_EXPORT \
    -I%PYTHON_HOME%\lib\site-packages\numpy\_core\include \
    -I%PYTHON_HOME%\include %PYTHON_HOME%\libs\python%PYTHON_VERSION%.lib
move distributions.dll ../../_examples/numba/
"""
import os

import numba as nb
import numpy as np
from cffi import FFI

from numpy.random import PCG64

ffi = FFI()
if os.path.exists('./distributions.dll'):
    lib = ffi.dlopen('./distributions.dll')
elif os.path.exists('./libdistributions.so'):
    lib = ffi.dlopen('./libdistributions.so')
else:
    raise RuntimeError('Required DLL/so file was not found.')

ffi.cdef("""
double random_standard_normal(void *bitgen_state);
""")
x = PCG64()
xffi = x.cffi
bit_generator = xffi.bit_generator

random_standard_normal = lib.random_standard_normal


def normals(n, bit_generator):
    out = np.empty(n)
    for i in range(n):
        out[i] = random_standard_normal(bit_generator)
    return out


normalsj = nb.jit(normals, nopython=True)

# Numba requires a memory address for void *
# Can also get address from x.ctypes.bit_generator.value
bit_generator_address = int(ffi.cast('uintptr_t', bit_generator))

norm = normalsj(1000, bit_generator_address)
print(norm[:12])
<|endoftext|>import numpy as np
import numba as nb

from numpy.random import PCG64
from timeit import timeit

bit_gen = PCG64()
next_d = bit_gen.cffi.next_double
state_addr = bit_gen.cffi.state_address

def normals(n, state):
    out = np.empty(n)
    for i in range((n + 1) // 2):
        x1 = 2.0 * next_d(state) - 1.0
        x2 = 2.0 * next_d(state) - 1.0
        r2 = x1 * x1 + x2 * x2
        while r2 >= 1.0 or r2 == 0.0:
            x1 = 2.0 * next_d(state) - 1.0
            x2 = 2.0 * next_d(state) - 1.0
            r2 = x1 * x1 + x2 * x2
        f = np.sqrt(-2.0 * np.log(r2) / r2)
        out[2 * i] = f * x1
        if 2 * i + 1 < n:
            out[2 * i + 1] = f * x2
    return out

# Compile using Numba
normalsj = nb.jit(normals, nopython=True)
# Must use state address not state with numba
n = 10000

def numbacall():
    return normalsj(n, state_addr)

rg = np.random.Generator(PCG64())

def numpycall():
    return rg.normal(size=n)

# Check that the functions work
r1 = numbacall()
r2 = numpycall()
assert r1.shape == (n,)
assert r1.shape == r2.shape

t1 = timeit(numbacall, number=1000)
print(f'{t1:.2f} secs for {n} PCG64 (Numba/PCG64) gaussian randoms')
t2 = timeit(numpycall, number=1000)
print(f'{t2:.2f} secs for {n} PCG64 (NumPy/PCG64) gaussian randoms')

# example 2

next_u32 = bit_gen.ctypes.next_uint32
ctypes_state = bit_gen.ctypes.state

@nb.jit(nopython=True)
def bounded_uint(lb, ub, state):
    mask = delta = ub - lb
    mask |= mask >> 1
    mask |= mask >> 2
    mask |= mask >> 4
    mask |= mask >> 8
    mask |= mask >> 16

    val = next_u32(state) & mask
    while val > delta:
        val = next_u32(state) & mask

    return lb + val


print(bounded_uint(323, 2394691, ctypes_state.value))


@nb.jit(nopython=True)
def bounded_uints(lb, ub, n, state):
    out = np.empty(n, dtype=np.uint32)
    for i in range(n):
        out[i] = bounded_uint(lb, ub, state)


bounded_uints(323, 2394691, 10000000, ctypes_state.value)


<|endoftext|>def __getattr__(attr_name):
    import warnings
    from numpy.fft import _helper
    ret = getattr(_helper, attr_name, None)
    if ret is None:
        raise AttributeError(
            f"module 'numpy.fft.helper' has no attribute {attr_name}")
    warnings.warn(
        "The numpy.fft.helper has been made private and renamed to "
        "numpy.fft._helper. All four functions exported by it (i.e. fftshift, "
        "ifftshift, fftfreq, rfftfreq) are available from numpy.fft. "
        f"Please use numpy.fft.{attr_name} instead.",
        DeprecationWarning,
        stacklevel=3
    )
    return ret
<|endoftext|>"""
Discrete Fourier Transform (:mod:`numpy.fft`)
=============================================

.. currentmodule:: numpy.fft

The SciPy module `scipy.fft` is a more comprehensive superset
of ``numpy.fft``, which includes only a basic set of routines.

Standard FFTs
-------------

.. autosummary::
   :toctree: generated/

   fft       Discrete Fourier transform.
   ifft      Inverse discrete Fourier transform.
   fft2      Discrete Fourier transform in two dimensions.
   ifft2     Inverse discrete Fourier transform in two dimensions.
   fftn      Discrete Fourier transform in N-dimensions.
   ifftn     Inverse discrete Fourier transform in N dimensions.

Real FFTs
---------

.. autosummary::
   :toctree: generated/

   rfft      Real discrete Fourier transform.
   irfft     Inverse real discrete Fourier transform.
   rfft2     Real discrete Fourier transform in two dimensions.
   irfft2    Inverse real discrete Fourier transform in two dimensions.
   rfftn     Real discrete Fourier transform in N dimensions.
   irfftn    Inverse real discrete Fourier transform in N dimensions.

Hermitian FFTs
--------------

.. autosummary::
   :toctree: generated/

   hfft      Hermitian discrete Fourier transform.
   ihfft     Inverse Hermitian discrete Fourier transform.

Helper routines
---------------

.. autosummary::
   :toctree: generated/

   fftfreq   Discrete Fourier Transform sample frequencies.
   rfftfreq  DFT sample frequencies (for usage with rfft, irfft).
   fftshift  Shift zero-frequency component to center of spectrum.
   ifftshift Inverse of fftshift.


Background information
----------------------

Fourier analysis is fundamentally a method for expressing a function as a
sum of periodic components, and for recovering the function from those
components.  When both the function and its Fourier transform are
replaced with discretized counterparts, it is called the discrete Fourier
transform (DFT).  The DFT has become a mainstay of numerical computing in
part because of a very fast algorithm for computing it, called the Fast
Fourier Transform (FFT), which was known to Gauss (1805) and was brought
to light in its current form by Cooley and Tukey [CT]_.  Press et al. [NR]_
provide an accessible introduction to Fourier analysis and its
applications.

Because the discrete Fourier transform separates its input into
components that contribute at discrete frequencies, it has a great number
of applications in digital signal processing, e.g., for filtering, and in
this context the discretized input to the transform is customarily
referred to as a *signal*, which exists in the *time domain*.  The output
is called a *spectrum* or *transform* and exists in the *frequency
domain*.

Implementation details
----------------------

There are many ways to define the DFT, varying in the sign of the
exponent, normalization, etc.  In this implementation, the DFT is defined
as

.. math::
   A_k =  \\sum_{m=0}^{n-1} a_m \\exp\\left\\{-2\\pi i{mk \\over n}\\right\\}
   \\qquad k = 0,\\ldots,n-1.

The DFT is in general defined for complex inputs and outputs, and a
single-frequency component at linear frequency :math:`f` is
represented by a complex exponential
:math:`a_m = \\exp\\{2\\pi i\\,f m\\Delta t\\}`, where :math:`\\Delta t`
is the sampling interval.

The values in the result follow so-called "standard" order: If ``A =
fft(a, n)``, then ``A[0]`` contains the zero-frequency term (the sum of
the signal), which is always purely real for real inputs. Then ``A[1:n/2]``
contains the positive-frequency terms, and ``A[n/2+1:]`` contains the
negative-frequency terms, in order of decreasingly negative frequency.
For an even number of input points, ``A[n/2]`` represents both positive and
negative Nyquist frequency, and is also purely real for real input.  For
an odd number of input points, ``A[(n-1)/2]`` contains the largest positive
frequency, while ``A[(n+1)/2]`` contains the largest negative frequency.
The routine ``np.fft.fftfreq(n)`` returns an array giving the frequencies
of corresponding elements in the output.  The routine
``np.fft.fftshift(A)`` shifts transforms and their frequencies to put the
zero-frequency components in the middle, and ``np.fft.ifftshift(A)`` undoes
that shift.

When the input `a` is a time-domain signal and ``A = fft(a)``, ``np.abs(A)``
is its amplitude spectrum and ``np.abs(A)**2`` is its power spectrum.
The phase spectrum is obtained by ``np.angle(A)``.

The inverse DFT is defined as

.. math::
   a_m = \\frac{1}{n}\\sum_{k=0}^{n-1}A_k\\exp\\left\\{2\\pi i{mk\\over n}\\right\\}
   \\qquad m = 0,\\ldots,n-1.

It differs from the forward transform by the sign of the exponential
argument and the default normalization by :math:`1/n`.

Type Promotion
--------------

`numpy.fft` promotes ``float32`` and ``complex64`` arrays to ``float64`` and
``complex128`` arrays respectively. For an FFT implementation that does not
promote input arrays, see `scipy.fftpack`.

Normalization
-------------

The argument ``norm`` indicates which direction of the pair of direct/inverse
transforms is scaled and with what normalization factor.
The default normalization (``"backward"``) has the direct (forward) transforms
unscaled and the inverse (backward) transforms scaled by :math:`1/n`. It is
possible to obtain unitary transforms by setting the keyword argument ``norm``
to ``"ortho"`` so that both direct and inverse transforms are scaled by
:math:`1/\\sqrt{n}`. Finally, setting the keyword argument ``norm`` to
``"forward"`` has the direct transforms scaled by :math:`1/n` and the inverse
transforms unscaled (i.e. exactly opposite to the default ``"backward"``).
`None` is an alias of the default option ``"backward"`` for backward
compatibility.

Real and Hermitian transforms
-----------------------------

When the input is purely real, its transform is Hermitian, i.e., the
component at frequency :math:`f_k` is the complex conjugate of the
component at frequency :math:`-f_k`, which means that for real
inputs there is no information in the negative frequency components that
is not already available from the positive frequency components.
The family of `rfft` functions is
designed to operate on real inputs, and exploits this symmetry by
computing only the positive frequency components, up to and including the
Nyquist frequency.  Thus, ``n`` input points produce ``n/2+1`` complex
output points.  The inverses of this family assumes the same symmetry of
its input, and for an output of ``n`` points uses ``n/2+1`` input points.

Correspondingly, when the spectrum is purely real, the signal is
Hermitian.  The `hfft` family of functions exploits this symmetry by
using ``n/2+1`` complex points in the input (time) domain for ``n`` real
points in the frequency domain.

In higher dimensions, FFTs are used, e.g., for image analysis and
filtering.  The computational efficiency of the FFT means that it can
also be a faster way to compute large convolutions, using the property
that a convolution in the time domain is equivalent to a point-by-point
multiplication in the frequency domain.

Higher dimensions
-----------------

In two dimensions, the DFT is defined as

.. math::
   A_{kl} =  \\sum_{m=0}^{M-1} \\sum_{n=0}^{N-1}
   a_{mn}\\exp\\left\\{-2\\pi i \\left({mk\\over M}+{nl\\over N}\\right)\\right\\}
   \\qquad k = 0, \\ldots, M-1;\\quad l = 0, \\ldots, N-1,

which extends in the obvious way to higher dimensions, and the inverses
in higher dimensions also extend in the same way.

References
----------

.. [CT] Cooley, James W., and John W. Tukey, 1965, "An algorithm for the
        machine calculation of complex Fourier series," *Math. Comput.*
        19: 297-301.

.. [NR] Press, W., Teukolsky, S., Vetterline, W.T., and Flannery, B.P.,
        2007, *Numerical Recipes: The Art of Scientific Computing*, ch.
        12-13.  Cambridge Univ. Press, Cambridge, UK.

Examples
--------

For examples, see the various functions.

"""

from . import _pocketfft, _helper
# TODO: `numpy.fft.helper`` was deprecated in NumPy 2.0. It should
# be deleted once downstream libraries move to `numpy.fft`.
from . import helper
from ._pocketfft import *
from ._helper import *

__all__ = _pocketfft.__all__.copy()
__all__ += _helper.__all__

from numpy._pytesttester import PytestTester
test = PytestTester(__name__)
del PytestTester
<|endoftext|>"""
Discrete Fourier Transforms

Routines in this module:

fft(a, n=None, axis=-1, norm="backward")
ifft(a, n=None, axis=-1, norm="backward")
rfft(a, n=None, axis=-1, norm="backward")
irfft(a, n=None, axis=-1, norm="backward")
hfft(a, n=None, axis=-1, norm="backward")
ihfft(a, n=None, axis=-1, norm="backward")
fftn(a, s=None, axes=None, norm="backward")
ifftn(a, s=None, axes=None, norm="backward")
rfftn(a, s=None, axes=None, norm="backward")
irfftn(a, s=None, axes=None, norm="backward")
fft2(a, s=None, axes=(-2,-1), norm="backward")
ifft2(a, s=None, axes=(-2, -1), norm="backward")
rfft2(a, s=None, axes=(-2,-1), norm="backward")
irfft2(a, s=None, axes=(-2, -1), norm="backward")

i = inverse transform
r = transform of purely real data
h = Hermite transform
n = n-dimensional transform
2 = 2-dimensional transform
(Note: 2D routines are just nD routines with different default
behavior.)

"""
__all__ = ['fft', 'ifft', 'rfft', 'irfft', 'hfft', 'ihfft', 'rfftn',
           'irfftn', 'rfft2', 'irfft2', 'fft2', 'ifft2', 'fftn', 'ifftn']

import functools
import warnings

from numpy.lib.array_utils import normalize_axis_index
from numpy._core import (asarray, empty_like, result_type,
                         conjugate, take, sqrt, reciprocal)
from . import _pocketfft_umath as pfu
from numpy._core import overrides


array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy.fft')


# `inv_norm` is a float by which the result of the transform needs to be
# divided. This replaces the original, more intuitive 'fct` parameter to avoid
# divisions by zero (or alternatively additional checks) in the case of
# zero-length axes during its computation.
def _raw_fft(a, n, axis, is_real, is_forward, norm, out=None):
    if n < 1:
        raise ValueError(f"Invalid number of FFT data points ({n}) specified.")

    # Calculate the normalization factor, passing in the array dtype to
    # avoid precision loss in the possible sqrt or reciprocal.
    if not is_forward:
        norm = _swap_direction(norm)

    real_dtype = result_type(a.real.dtype, 1.0)
    if norm is None or norm == "backward":
        fct = 1
    elif norm == "ortho":
        fct = reciprocal(sqrt(n, dtype=real_dtype))
    elif norm == "forward":
        fct = reciprocal(n, dtype=real_dtype)
    else:
        raise ValueError(f'Invalid norm value {norm}; should be "backward",'
                         '"ortho" or "forward".')

    n_out = n
    if is_real:
        if is_forward:
            ufunc = pfu.rfft_n_even if n % 2 == 0 else pfu.rfft_n_odd
            n_out = n // 2 + 1
        else:
            ufunc = pfu.irfft
    else:
        ufunc = pfu.fft if is_forward else pfu.ifft

    axis = normalize_axis_index(axis, a.ndim)

    if out is None:
        if is_real and not is_forward:  # irfft, complex in, real output.
            out_dtype = real_dtype
        else:  # Others, complex output.
            out_dtype = result_type(a.dtype, 1j)
        out = empty_like(a, shape=a.shape[:axis] + (n_out,) + a.shape[axis+1:],
                         dtype=out_dtype)
    elif ((shape := getattr(out, "shape", None)) is not None
          and (len(shape) != a.ndim or shape[axis] != n_out)):
        raise ValueError("output array has wrong shape.")

    return ufunc(a, fct, axes=[(axis,), (), (axis,)], out=out)


_SWAP_DIRECTION_MAP = {"backward": "forward", None: "forward",
                       "ortho": "ortho", "forward": "backward"}


def _swap_direction(norm):
    try:
        return _SWAP_DIRECTION_MAP[norm]
    except KeyError:
        raise ValueError(f'Invalid norm value {norm}; should be "backward", '
                         '"ortho" or "forward".') from None


def _fft_dispatcher(a, n=None, axis=None, norm=None, out=None):
    return (a, out)


@array_function_dispatch(_fft_dispatcher)
def fft(a, n=None, axis=-1, norm=None, out=None):
    """
    Compute the one-dimensional discrete Fourier Transform.

    This function computes the one-dimensional *n*-point discrete Fourier
    Transform (DFT) with the efficient Fast Fourier Transform (FFT)
    algorithm [CT].

    Parameters
    ----------
    a : array_like
        Input array, can be complex.
    n : int, optional
        Length of the transformed axis of the output.
        If `n` is smaller than the length of the input, the input is cropped.
        If it is larger, the input is padded with zeros.  If `n` is not given,
        the length of the input along the axis specified by `axis` is used.
    axis : int, optional
        Axis over which to compute the FFT.  If not given, the last axis is
        used.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

        .. versionadded:: 1.20.0

            The "backward", "forward" values were added.
    out : complex ndarray, optional
        If provided, the result will be placed in this array. It should be
        of the appropriate shape and dtype.

        .. versionadded:: 2.0.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.

    Raises
    ------
    IndexError
        If `axis` is not a valid axis of `a`.

    See Also
    --------
    numpy.fft : for definition of the DFT and conventions used.
    ifft : The inverse of `fft`.
    fft2 : The two-dimensional FFT.
    fftn : The *n*-dimensional FFT.
    rfftn : The *n*-dimensional FFT of real input.
    fftfreq : Frequency bins for given FFT parameters.

    Notes
    -----
    FFT (Fast Fourier Transform) refers to a way the discrete Fourier
    Transform (DFT) can be calculated efficiently, by using symmetries in the
    calculated terms.  The symmetry is highest when `n` is a power of 2, and
    the transform is therefore most efficient for these sizes.

    The DFT is defined, with the conventions used in this implementation, in
    the documentation for the `numpy.fft` module.

    References
    ----------
    .. [CT] Cooley, James W., and John W. Tukey, 1965, "An algorithm for the
            machine calculation of complex Fourier series," *Math. Comput.*
            19: 297-301.

    Examples
    --------
    >>> import numpy as np
    >>> np.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8))
    array([-2.33486982e-16+1.14423775e-17j,  8.00000000e+00-1.25557246e-15j,
            2.33486982e-16+2.33486982e-16j,  0.00000000e+00+1.22464680e-16j,
           -1.14423775e-17+2.33486982e-16j,  0.00000000e+00+5.20784380e-16j,
            1.14423775e-17+1.14423775e-17j,  0.00000000e+00+1.22464680e-16j])

    In this example, real input has an FFT which is Hermitian, i.e., symmetric
    in the real part and anti-symmetric in the imaginary part, as described in
    the `numpy.fft` documentation:

    >>> import matplotlib.pyplot as plt
    >>> t = np.arange(256)
    >>> sp = np.fft.fft(np.sin(t))
    >>> freq = np.fft.fftfreq(t.shape[-1])
    >>> plt.plot(freq, sp.real, freq, sp.imag)
    [<matplotlib.lines.Line2D object at 0x...>, <matplotlib.lines.Line2D object at 0x...>]
    >>> plt.show()

    """
    a = asarray(a)
    if n is None:
        n = a.shape[axis]
    output = _raw_fft(a, n, axis, False, True, norm, out)
    return output


@array_function_dispatch(_fft_dispatcher)
def ifft(a, n=None, axis=-1, norm=None, out=None):
    """
    Compute the one-dimensional inverse discrete Fourier Transform.

    This function computes the inverse of the one-dimensional *n*-point
    discrete Fourier transform computed by `fft`.  In other words,
    ``ifft(fft(a)) == a`` to within numerical accuracy.
    For a general description of the algorithm and definitions,
    see `numpy.fft`.

    The input should be ordered in the same way as is returned by `fft`,
    i.e.,

    * ``a[0]`` should contain the zero frequency term,
    * ``a[1:n//2]`` should contain the positive-frequency terms,
    * ``a[n//2 + 1:]`` should contain the negative-frequency terms, in
      increasing order starting from the most negative frequency.

    For an even number of input points, ``A[n//2]`` represents the sum of
    the values at the positive and negative Nyquist frequencies, as the two
    are aliased together. See `numpy.fft` for details.

    Parameters
    ----------
    a : array_like
        Input array, can be complex.
    n : int, optional
        Length of the transformed axis of the output.
        If `n` is smaller than the length of the input, the input is cropped.
        If it is larger, the input is padded with zeros.  If `n` is not given,
        the length of the input along the axis specified by `axis` is used.
        See notes about padding issues.
    axis : int, optional
        Axis over which to compute the inverse DFT.  If not given, the last
        axis is used.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

        .. versionadded:: 1.20.0

            The "backward", "forward" values were added.

    out : complex ndarray, optional
        If provided, the result will be placed in this array. It should be
        of the appropriate shape and dtype.

        .. versionadded:: 2.0.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.

    Raises
    ------
    IndexError
        If `axis` is not a valid axis of `a`.

    See Also
    --------
    numpy.fft : An introduction, with definitions and general explanations.
    fft : The one-dimensional (forward) FFT, of which `ifft` is the inverse
    ifft2 : The two-dimensional inverse FFT.
    ifftn : The n-dimensional inverse FFT.

    Notes
    -----
    If the input parameter `n` is larger than the size of the input, the input
    is padded by appending zeros at the end.  Even though this is the common
    approach, it might lead to surprising results.  If a different padding is
    desired, it must be performed before calling `ifft`.

    Examples
    --------
    >>> import numpy as np
    >>> np.fft.ifft([0, 4, 0, 0])
    array([ 1.+0.j,  0.+1.j, -1.+0.j,  0.-1.j]) # may vary

    Create and plot a band-limited signal with random phases:

    >>> import matplotlib.pyplot as plt
    >>> t = np.arange(400)
    >>> n = np.zeros((400,), dtype=complex)
    >>> n[40:60] = np.exp(1j*np.random.uniform(0, 2*np.pi, (20,)))
    >>> s = np.fft.ifft(n)
    >>> plt.plot(t, s.real, label='real')
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.plot(t, s.imag, '--', label='imaginary')
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.legend()
    <matplotlib.legend.Legend object at ...>
    >>> plt.show()

    """
    a = asarray(a)
    if n is None:
        n = a.shape[axis]
    output = _raw_fft(a, n, axis, False, False, norm, out=out)
    return output


@array_function_dispatch(_fft_dispatcher)
def rfft(a, n=None, axis=-1, norm=None, out=None):
    """
    Compute the one-dimensional discrete Fourier Transform for real input.

    This function computes the one-dimensional *n*-point discrete Fourier
    Transform (DFT) of a real-valued array by means of an efficient algorithm
    called the Fast Fourier Transform (FFT).

    Parameters
    ----------
    a : array_like
        Input array
    n : int, optional
        Number of points along transformation axis in the input to use.
        If `n` is smaller than the length of the input, the input is cropped.
        If it is larger, the input is padded with zeros. If `n` is not given,
        the length of the input along the axis specified by `axis` is used.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last axis is
        used.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

        .. versionadded:: 1.20.0

            The "backward", "forward" values were added.

    out : complex ndarray, optional
        If provided, the result will be placed in this array. It should be
        of the appropriate shape and dtype.

        .. versionadded:: 2.0.0

    Returns
    -------
    out : complex ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.
        If `n` is even, the length of the transformed axis is ``(n/2)+1``.
        If `n` is odd, the length is ``(n+1)/2``.

    Raises
    ------
    IndexError
        If `axis` is not a valid axis of `a`.

    See Also
    --------
    numpy.fft : For definition of the DFT and conventions used.
    irfft : The inverse of `rfft`.
    fft : The one-dimensional FFT of general (complex) input.
    fftn : The *n*-dimensional FFT.
    rfftn : The *n*-dimensional FFT of real input.

    Notes
    -----
    When the DFT is computed for purely real input, the output is
    Hermitian-symmetric, i.e. the negative frequency terms are just the complex
    conjugates of the corresponding positive-frequency terms, and the
    negative-frequency terms are therefore redundant.  This function does not
    compute the negative frequency terms, and the length of the transformed
    axis of the output is therefore ``n//2 + 1``.

    When ``A = rfft(a)`` and fs is the sampling frequency, ``A[0]`` contains
    the zero-frequency term 0*fs, which is real due to Hermitian symmetry.

    If `n` is even, ``A[-1]`` contains the term representing both positive
    and negative Nyquist frequency (+fs/2 and -fs/2), and must also be purely
    real. If `n` is odd, there is no term at fs/2; ``A[-1]`` contains
    the largest positive frequency (fs/2*(n-1)/n), and is complex in the
    general case.

    If the input `a` contains an imaginary part, it is silently discarded.

    Examples
    --------
    >>> import numpy as np
    >>> np.fft.fft([0, 1, 0, 0])
    array([ 1.+0.j,  0.-1.j, -1.+0.j,  0.+1.j]) # may vary
    >>> np.fft.rfft([0, 1, 0, 0])
    array([ 1.+0.j,  0.-1.j, -1.+0.j]) # may vary

    Notice how the final element of the `fft` output is the complex conjugate
    of the second element, for real input. For `rfft`, this symmetry is
    exploited to compute only the non-negative frequency terms.

    """
    a = asarray(a)
    if n is None:
        n = a.shape[axis]
    output = _raw_fft(a, n, axis, True, True, norm, out=out)
    return output


@array_function_dispatch(_fft_dispatcher)
def irfft(a, n=None, axis=-1, norm=None, out=None):
    """
    Computes the inverse of `rfft`.

    This function computes the inverse of the one-dimensional *n*-point
    discrete Fourier Transform of real input computed by `rfft`.
    In other words, ``irfft(rfft(a), len(a)) == a`` to within numerical
    accuracy. (See Notes below for why ``len(a)`` is necessary here.)

    The input is expected to be in the form returned by `rfft`, i.e. the
    real zero-frequency term followed by the complex positive frequency terms
    in order of increasing frequency.  Since the discrete Fourier Transform of
    real input is Hermitian-symmetric, the negative frequency terms are taken
    to be the complex conjugates of the corresponding positive frequency terms.

    Parameters
    ----------
    a : array_like
        The input array.
    n : int, optional
        Length of the transformed axis of the output.
        For `n` output points, ``n//2+1`` input points are necessary.  If the
        input is longer than this, it is cropped.  If it is shorter than this,
        it is padded with zeros.  If `n` is not given, it is taken to be
        ``2*(m-1)`` where ``m`` is the length of the input along the axis
        specified by `axis`.
    axis : int, optional
        Axis over which to compute the inverse FFT. If not given, the last
        axis is used.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

        .. versionadded:: 1.20.0

            The "backward", "forward" values were added.

    out : ndarray, optional
        If provided, the result will be placed in this array. It should be
        of the appropriate shape and dtype.

        .. versionadded:: 2.0.0

    Returns
    -------
    out : ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.
        The length of the transformed axis is `n`, or, if `n` is not given,
        ``2*(m-1)`` where ``m`` is the length of the transformed axis of the
        input. To get an odd number of output points, `n` must be specified.

    Raises
    ------
    IndexError
        If `axis` is not a valid axis of `a`.

    See Also
    --------
    numpy.fft : For definition of the DFT and conventions used.
    rfft : The one-dimensional FFT of real input, of which `irfft` is inverse.
    fft : The one-dimensional FFT.
    irfft2 : The inverse of the two-dimensional FFT of real input.
    irfftn : The inverse of the *n*-dimensional FFT of real input.

    Notes
    -----
    Returns the real valued `n`-point inverse discrete Fourier transform
    of `a`, where `a` contains the non-negative frequency terms of a
    Hermitian-symmetric sequence. `n` is the length of the result, not the
    input.

    If you specify an `n` such that `a` must be zero-padded or truncated, the
    extra/removed values will be added/removed at high frequencies. One can
    thus resample a series to `m` points via Fourier interpolation by:
    ``a_resamp = irfft(rfft(a), m)``.

    The correct interpretation of the hermitian input depends on the length of
    the original data, as given by `n`. This is because each input shape could
    correspond to either an odd or even length signal. By default, `irfft`
    assumes an even output length which puts the last entry at the Nyquist
    frequency; aliasing with its symmetric counterpart. By Hermitian symmetry,
    the value is thus treated as purely real. To avoid losing information, the
    correct length of the real input **must** be given.

    Examples
    --------
    >>> import numpy as np
    >>> np.fft.ifft([1, -1j, -1, 1j])
    array([0.+0.j,  1.+0.j,  0.+0.j,  0.+0.j]) # may vary
    >>> np.fft.irfft([1, -1j, -1])
    array([0.,  1.,  0.,  0.])

    Notice how the last term in the input to the ordinary `ifft` is the
    complex conjugate of the second term, and the output has zero imaginary
    part everywhere.  When calling `irfft`, the negative frequencies are not
    specified, and the output array is purely real.

    """
    a = asarray(a)
    if n is None:
        n = (a.shape[axis] - 1) * 2
    output = _raw_fft(a, n, axis, True, False, norm, out=out)
    return output


@array_function_dispatch(_fft_dispatcher)
def hfft(a, n=None, axis=-1, norm=None, out=None):
    """
    Compute the FFT of a signal that has Hermitian symmetry, i.e., a real
    spectrum.

    Parameters
    ----------
    a : array_like
        The input array.
    n : int, optional
        Length of the transformed axis of the output. For `n` output
        points, ``n//2 + 1`` input points are necessary.  If the input is
        longer than this, it is cropped.  If it is shorter than this, it is
        padded with zeros.  If `n` is not given, it is taken to be ``2*(m-1)``
        where ``m`` is the length of the input along the axis specified by
        `axis`.
    axis : int, optional
        Axis over which to compute the FFT. If not given, the last
        axis is used.
    norm : {"backward", "ortho", "forward"}, optional
        Normalization mode (see `numpy.fft`). Default is "backward".
        Indicates which direction of the forward/backward pair of transforms
        is scaled and with what normalization factor.

        .. versionadded:: 1.20.0

            The "backward", "forward" values were added.

    out : ndarray, optional
        If provided, the result will be placed in this array. It should be
        of the appropriate shape and dtype.

        .. versionadded:: 2.0.0

    Returns
    -------
    out : ndarray
        The truncated or zero-padded input, transformed along the axis
        indicated by `axis`, or the last one if `axis` is not specified.
        The length of the transformed axis is `n`, or, if `n` is not given,
