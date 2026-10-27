    ``x`` is an array of five points in three-dimensional space.

    >>> x = np.array([[2, 0, 2], [2, 2, 3], [-2, 4, 5], [0, 1, 9], [2, 2, 4]])

    ``pdist(x)`` with no additional arguments computes the 10 pairwise
    Euclidean distances:

    >>> pdist(x)
    array([2.23606798, 6.40312424, 7.34846923, 2.82842712, 4.89897949,
           6.40312424, 1.        , 5.38516481, 4.58257569, 5.47722558])

    The following computes the pairwise Minkowski distances with ``p = 3.5``:

    >>> pdist(x, metric='minkowski', p=3.5)
    array([2.04898923, 5.1154929 , 7.02700737, 2.43802731, 4.19042714,
           6.03956994, 1.        , 4.45128103, 4.10636143, 5.0619695 ])

    The pairwise city block or Manhattan distances:

    >>> pdist(x, metric='cityblock')
    array([ 3., 11., 10.,  4.,  8.,  9.,  1.,  9.,  7.,  8.])

    """
    # You can also call this as:
    #     Y = pdist(X, 'test_abc')
    # where 'abc' is the metric being tested.  This computes the distance
    # between all pairs of vectors in X using the distance metric 'abc' but
    # with a more succinct, verifiable, but less efficient implementation.

    X = _asarray_validated(X, sparse_ok=False, objects_ok=True, mask_ok=True,
                           check_finite=False)

    s = X.shape
    if len(s) != 2:
        raise ValueError('A 2-dimensional array must be passed.')

    m, n = s

    if callable(metric):
        mstr = getattr(metric, '__name__', 'UnknownCustomMetric')
        metric_info = _METRIC_ALIAS.get(mstr, None)

        if metric_info is not None:
            X, typ, kwargs = _validate_pdist_input(
                X, m, n, metric_info, **kwargs)

        return _pdist_callable(X, metric=metric, out=out, **kwargs)
    elif isinstance(metric, str):
        mstr = metric.lower()
        metric_info = _METRIC_ALIAS.get(mstr, None)

        if metric_info is not None:
            pdist_fn = metric_info.pdist_func
            return pdist_fn(X, out=out, **kwargs)
        elif mstr.startswith("test_"):
            metric_info = _TEST_METRICS.get(mstr, None)
            if metric_info is None:
                raise ValueError(f'Unknown "Test" Distance Metric: {mstr[5:]}')
            X, typ, kwargs = _validate_pdist_input(
                X, m, n, metric_info, **kwargs)
            return _pdist_callable(
                X, metric=metric_info.dist_func, out=out, **kwargs)
        else:
            raise ValueError(f'Unknown Distance Metric: {mstr}')
    else:
        raise TypeError('2nd argument metric must be a string identifier '
                        'or a function.')


def squareform(X, force="no", checks=True):
    """
    Convert a vector-form distance vector to a square-form distance
    matrix, and vice-versa.

    Parameters
    ----------
    X : array_like
        Either a condensed or redundant distance matrix.
    force : str, optional
        As with MATLAB(TM), if force is equal to ``'tovector'`` or
        ``'tomatrix'``, the input will be treated as a distance matrix or
        distance vector respectively.
    checks : bool, optional
        If set to False, no checks will be made for matrix
        symmetry nor zero diagonals. This is useful if it is known that
        ``X - X.T1`` is small and ``diag(X)`` is close to zero.
        These values are ignored any way so they do not disrupt the
        squareform transformation.

    Returns
    -------
    Y : ndarray
        If a condensed distance matrix is passed, a redundant one is
        returned, or if a redundant one is passed, a condensed distance
        matrix is returned.

    Notes
    -----
    1. ``v = squareform(X)``

       Given a square n-by-n symmetric distance matrix ``X``,
       ``v = squareform(X)`` returns a ``n * (n-1) / 2``
       (i.e. binomial coefficient n choose 2) sized vector `v`
       where :math:`v[{n \\choose 2} - {n-i \\choose 2} + (j-i-1)]`
       is the distance between distinct points ``i`` and ``j``.
       If ``X`` is non-square or asymmetric, an error is raised.

    2. ``X = squareform(v)``

       Given a ``n * (n-1) / 2`` sized vector ``v``
       for some integer ``n >= 1`` encoding distances as described,
       ``X = squareform(v)`` returns a n-by-n distance matrix ``X``.
       The ``X[i, j]`` and ``X[j, i]`` values are set to
       :math:`v[{n \\choose 2} - {n-i \\choose 2} + (j-i-1)]`
       and all diagonal elements are zero.

    In SciPy 0.19.0, ``squareform`` stopped casting all input types to
    float64, and started returning arrays of the same dtype as the input.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.spatial.distance import pdist, squareform

    ``x`` is an array of five points in three-dimensional space.

    >>> x = np.array([[2, 0, 2], [2, 2, 3], [-2, 4, 5], [0, 1, 9], [2, 2, 4]])

    ``pdist(x)`` computes the Euclidean distances between each pair of
    points in ``x``.  The distances are returned in a one-dimensional
    array with length ``5*(5 - 1)/2 = 10``.

    >>> distvec = pdist(x)
    >>> distvec
    array([2.23606798, 6.40312424, 7.34846923, 2.82842712, 4.89897949,
           6.40312424, 1.        , 5.38516481, 4.58257569, 5.47722558])

    ``squareform(distvec)`` returns the 5x5 distance matrix.

    >>> m = squareform(distvec)
    >>> m
    array([[0.        , 2.23606798, 6.40312424, 7.34846923, 2.82842712],
           [2.23606798, 0.        , 4.89897949, 6.40312424, 1.        ],
           [6.40312424, 4.89897949, 0.        , 5.38516481, 4.58257569],
           [7.34846923, 6.40312424, 5.38516481, 0.        , 5.47722558],
           [2.82842712, 1.        , 4.58257569, 5.47722558, 0.        ]])

    When given a square distance matrix ``m``, ``squareform(m)`` returns
    the one-dimensional condensed distance vector associated with the
    matrix.  In this case, we recover ``distvec``.

    >>> squareform(m)
    array([2.23606798, 6.40312424, 7.34846923, 2.82842712, 4.89897949,
           6.40312424, 1.        , 5.38516481, 4.58257569, 5.47722558])
    """
    X = np.ascontiguousarray(X)

    s = X.shape

    if force.lower() == 'tomatrix':
        if len(s) != 1:
            raise ValueError("Forcing 'tomatrix' but input X is not a "
                             "distance vector.")
    elif force.lower() == 'tovector':
        if len(s) != 2:
            raise ValueError("Forcing 'tovector' but input X is not a "
                             "distance matrix.")

    # X = squareform(v)
    if len(s) == 1:
        if s[0] == 0:
            return np.zeros((1, 1), dtype=X.dtype)

        # Grab the closest value to the square root of the number
        # of elements times 2 to see if the number of elements
        # is indeed a binomial coefficient.
        d = int(np.ceil(np.sqrt(s[0] * 2)))

        # Check that v is of valid dimensions.
        if d * (d - 1) != s[0] * 2:
            raise ValueError('Incompatible vector size. It must be a binomial '
                             'coefficient n choose 2 for some integer n >= 2.')

        # Allocate memory for the distance matrix.
        M = np.zeros((d, d), dtype=X.dtype)

        # Since the C code does not support striding using strides.
        # The dimensions are used instead.
        X = _copy_array_if_base_present(X)

        # Fill in the values of the distance matrix.
        _distance_wrap.to_squareform_from_vector_wrap(M, X)

        # Return the distance matrix.
        return M
    elif len(s) == 2:
        if s[0] != s[1]:
            raise ValueError('The matrix argument must be square.')
        if checks:
            is_valid_dm(X, throw=True, name='X')

        # One-side of the dimensions is set here.
        d = s[0]

        if d <= 1:
            return np.array([], dtype=X.dtype)

        # Create a vector.
        v = np.zeros((d * (d - 1)) // 2, dtype=X.dtype)

        # Since the C code does not support striding using strides.
        # The dimensions are used instead.
        X = _copy_array_if_base_present(X)

        # Convert the vector to squareform.
        _distance_wrap.to_vector_from_squareform_wrap(X, v)
        return v
    else:
        raise ValueError("The first argument must be one or two dimensional "
                         f"array. A {len(s)}-dimensional array is not permitted")


def is_valid_dm(D, tol=0.0, throw=False, name="D", warning=False):
    """
    Return True if input array is a valid distance matrix.

    Distance matrices must be 2-dimensional numpy arrays.
    They must have a zero-diagonal, and they must be symmetric.

    Parameters
    ----------
    D : array_like
        The candidate object to test for validity.
    tol : float, optional
        The distance matrix should be symmetric. `tol` is the maximum
        difference between entries ``ij`` and ``ji`` for the distance
        metric to be considered symmetric.
    throw : bool, optional
        An exception is thrown if the distance matrix passed is not valid.
    name : str, optional
        The name of the variable to checked. This is useful if
        throw is set to True so the offending variable can be identified
        in the exception message when an exception is thrown.
    warning : bool, optional
        Instead of throwing an exception, a warning message is
        raised.

    Returns
    -------
    valid : bool
        True if the variable `D` passed is a valid distance matrix.

    Notes
    -----
    Small numerical differences in `D` and `D.T` and non-zeroness of
    the diagonal are ignored if they are within the tolerance specified
    by `tol`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.spatial.distance import is_valid_dm

    This matrix is a valid distance matrix.

    >>> d = np.array([[0.0, 1.1, 1.2, 1.3],
    ...               [1.1, 0.0, 1.0, 1.4],
    ...               [1.2, 1.0, 0.0, 1.5],
    ...               [1.3, 1.4, 1.5, 0.0]])
    >>> is_valid_dm(d)
    True

    In the following examples, the input is not a valid distance matrix.

    Not square:

    >>> is_valid_dm([[0, 2, 2], [2, 0, 2]])
    False

    Nonzero diagonal element:

    >>> is_valid_dm([[0, 1, 1], [1, 2, 3], [1, 3, 0]])
    False

    Not symmetric:

    >>> is_valid_dm([[0, 1, 3], [2, 0, 1], [3, 1, 0]])
    False

    """
    D = np.asarray(D, order='c')
    valid = True
    try:
        s = D.shape
        if len(D.shape) != 2:
            if name:
                raise ValueError(f"Distance matrix '{name}' must have shape=2 "
                                 "(i.e. be two-dimensional).")
            else:
                raise ValueError('Distance matrix must have shape=2 (i.e. '
                                 'be two-dimensional).')
        if tol == 0.0:
            if not (D == D.T).all():
                if name:
                    raise ValueError(f"Distance matrix '{name}' must be symmetric.")
                else:
                    raise ValueError('Distance matrix must be symmetric.')
            if not (D[range(0, s[0]), range(0, s[0])] == 0).all():
                if name:
                    raise ValueError(f"Distance matrix '{name}' diagonal must be zero.")
                else:
                    raise ValueError('Distance matrix diagonal must be zero.')
        else:
            if not (D - D.T <= tol).all():
                if name:
                    raise ValueError(f'Distance matrix \'{name}\' must be '
                                     f'symmetric within tolerance {tol:5.5f}.')
                else:
                    raise ValueError('Distance matrix must be symmetric within '
                                     f'tolerance {tol:5.5f}.')
            if not (D[range(0, s[0]), range(0, s[0])] <= tol).all():
                if name:
                    raise ValueError(f'Distance matrix \'{name}\' diagonal must be '
                                     f'close to zero within tolerance {tol:5.5f}.')
                else:
                    raise ValueError(('Distance matrix \'{}\' diagonal must be close '
                                      'to zero within tolerance {:5.5f}.').format(*tol))
    except Exception as e:
        if throw:
            raise
        if warning:
            warnings.warn(str(e), stacklevel=2)
        valid = False
    return valid


def is_valid_y(y, warning=False, throw=False, name=None):
    """
    Return True if the input array is a valid condensed distance matrix.

    Condensed distance matrices must be 1-dimensional numpy arrays.
    Their length must be a binomial coefficient :math:`{n \\choose 2}`
    for some positive integer n.

    Parameters
    ----------
    y : array_like
        The condensed distance matrix.
    warning : bool, optional
        Invokes a warning if the variable passed is not a valid
        condensed distance matrix. The warning message explains why
        the distance matrix is not valid.  `name` is used when
        referencing the offending variable.
    throw : bool, optional
        Throws an exception if the variable passed is not a valid
        condensed distance matrix.
    name : bool, optional
        Used when referencing the offending variable in the
        warning or exception message.

    Returns
    -------
    bool
        True if the input array is a valid condensed distance matrix,
        False otherwise.

    Examples
    --------
    >>> from scipy.spatial.distance import is_valid_y

    This vector is a valid condensed distance matrix.  The length is 6,
    which corresponds to ``n = 4``, since ``4*(4 - 1)/2`` is 6.

    >>> v = [1.0, 1.2, 1.0, 0.5, 1.3, 0.9]
    >>> is_valid_y(v)
    True

    An input vector with length, say, 7, is not a valid condensed distance
    matrix.

    >>> is_valid_y([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7])
    False

    """
    y = np.asarray(y, order='c')
    valid = True
    try:
        if len(y.shape) != 1:
            if name:
                raise ValueError(f"Condensed distance matrix '{name}' must "
                                 "have shape=1 (i.e. be one-dimensional).")
            else:
                raise ValueError('Condensed distance matrix must have shape=1 '
                                 '(i.e. be one-dimensional).')
        n = y.shape[0]
        d = int(np.ceil(np.sqrt(n * 2)))
        if (d * (d - 1) / 2) != n:
            if name:
                raise ValueError(f"Length n of condensed distance matrix '{name}' "
                                 "must be a binomial coefficient, i.e."
                                 "there must be a k such that (k \\choose 2)=n)!")
            else:
                raise ValueError('Length n of condensed distance matrix must '
                                 'be a binomial coefficient, i.e. there must '
                                 'be a k such that (k \\choose 2)=n)!')
    except Exception as e:
        if throw:
            raise
        if warning:
            warnings.warn(str(e), stacklevel=2)
        valid = False
    return valid


def num_obs_dm(d):
    """
    Return the number of original observations that correspond to a
    square, redundant distance matrix.

    Parameters
    ----------
    d : array_like
        The target distance matrix.

    Returns
    -------
    num_obs_dm : int
        The number of observations in the redundant distance matrix.

    Examples
    --------
    Find the number of original observations corresponding
    to a square redundant distance matrix d.
    
    >>> from scipy.spatial.distance import num_obs_dm
    >>> d = [[0, 100, 200], [100, 0, 150], [200, 150, 0]]
    >>> num_obs_dm(d)
    3
    """
    d = np.asarray(d, order='c')
    is_valid_dm(d, tol=np.inf, throw=True, name='d')
    return d.shape[0]


def num_obs_y(Y):
    """
    Return the number of original observations that correspond to a
    condensed distance matrix.

    Parameters
    ----------
    Y : array_like
        Condensed distance matrix.

    Returns
    -------
    n : int
        The number of observations in the condensed distance matrix `Y`.

    Examples
    --------
    Find the number of original observations corresponding to a
    condensed distance matrix Y.
    
    >>> from scipy.spatial.distance import num_obs_y
    >>> Y = [1, 2, 3.5, 7, 10, 4]
    >>> num_obs_y(Y)
    4
    """
    Y = np.asarray(Y, order='c')
    is_valid_y(Y, throw=True, name='Y')
    k = Y.shape[0]
    if k == 0:
        raise ValueError("The number of observations cannot be determined on "
                         "an empty distance matrix.")
    d = int(np.ceil(np.sqrt(k * 2)))
    if (d * (d - 1) / 2) != k:
        raise ValueError("Invalid condensed distance matrix passed. Must be "
                         "some k where k=(n choose 2) for some n >= 2.")
    return d


def _prepare_out_argument(out, dtype, expected_shape):
    if out is None:
        return np.empty(expected_shape, dtype=dtype)

    if out.shape != expected_shape:
        raise ValueError("Output array has incorrect shape.")
    if not out.flags.c_contiguous:
        raise ValueError("Output array must be C-contiguous.")
    if out.dtype != np.float64:
        raise ValueError("Output array must be double type.")
    return out


def _pdist_callable(X, *, out, metric, **kwargs):
    n = X.shape[0]
    out_size = (n * (n - 1)) // 2
    dm = _prepare_out_argument(out, np.float64, (out_size,))
    k = 0
    for i in range(X.shape[0] - 1):
        for j in range(i + 1, X.shape[0]):
            dm[k] = metric(X[i], X[j], **kwargs)
            k += 1
    return dm


def _cdist_callable(XA, XB, *, out, metric, **kwargs):
    mA = XA.shape[0]
    mB = XB.shape[0]
    dm = _prepare_out_argument(out, np.float64, (mA, mB))
    for i in range(mA):
        for j in range(mB):
            dm[i, j] = metric(XA[i], XB[j], **kwargs)
    return dm


def cdist(XA, XB, metric='euclidean', *, out=None, **kwargs):
    """
    Compute distance between each pair of the two collections of inputs.

    See Notes for common calling conventions.

    Parameters
    ----------
    XA : array_like
        An :math:`m_A` by :math:`n` array of :math:`m_A`
        original observations in an :math:`n`-dimensional space.
        Inputs are converted to float type.
    XB : array_like
        An :math:`m_B` by :math:`n` array of :math:`m_B`
        original observations in an :math:`n`-dimensional space.
        Inputs are converted to float type.
    metric : str or callable, optional
        The distance metric to use. If a string, the distance function can be
        'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
        'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
        'kulczynski1', 'mahalanobis', 'matching', 'minkowski',
        'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
        'sokalsneath', 'sqeuclidean', 'yule'.
    **kwargs : dict, optional
        Extra arguments to `metric`: refer to each metric documentation for a
        list of all possible arguments.

        Some possible arguments:

        p : scalar
        The p-norm to apply for Minkowski, weighted and unweighted.
        Default: 2.

        w : array_like
        The weight vector for metrics that support weights (e.g., Minkowski).

        V : array_like
        The variance vector for standardized Euclidean.
        Default: var(vstack([XA, XB]), axis=0, ddof=1)

        VI : array_like
        The inverse of the covariance matrix for Mahalanobis.
        Default: inv(cov(vstack([XA, XB].T))).T

        out : ndarray
        The output array
        If not None, the distance matrix Y is stored in this array.

    Returns
    -------
    Y : ndarray
        A :math:`m_A` by :math:`m_B` distance matrix is returned.
        For each :math:`i` and :math:`j`, the metric
        ``dist(u=XA[i], v=XB[j])`` is computed and stored in the
        :math:`ij` th entry.

    Raises
    ------
    ValueError
        An exception is thrown if `XA` and `XB` do not have
        the same number of columns.

    Notes
    -----
    The following are common calling conventions:

    1. ``Y = cdist(XA, XB, 'euclidean')``

       Computes the distance between :math:`m` points using
       Euclidean distance (2-norm) as the distance metric between the
       points. The points are arranged as :math:`m`
       :math:`n`-dimensional row vectors in the matrix X.

    2. ``Y = cdist(XA, XB, 'minkowski', p=2.)``

       Computes the distances using the Minkowski distance
       :math:`\\|u-v\\|_p` (:math:`p`-norm) where :math:`p > 0` (note
       that this is only a quasi-metric if :math:`0 < p < 1`).

    3. ``Y = cdist(XA, XB, 'cityblock')``

       Computes the city block or Manhattan distance between the
       points.

    4. ``Y = cdist(XA, XB, 'seuclidean', V=None)``

       Computes the standardized Euclidean distance. The standardized
       Euclidean distance between two n-vectors ``u`` and ``v`` is

       .. math::

          \\sqrt{\\sum {(u_i-v_i)^2 / V[x_i]}}.

       V is the variance vector; V[i] is the variance computed over all
       the i'th components of the points. If not passed, it is
       automatically computed.

    5. ``Y = cdist(XA, XB, 'sqeuclidean')``

       Computes the squared Euclidean distance :math:`\\|u-v\\|_2^2` between
       the vectors.

    6. ``Y = cdist(XA, XB, 'cosine')``

       Computes the cosine distance between vectors u and v,

       .. math::

          1 - \\frac{u \\cdot v}
                   {{\\|u\\|}_2 {\\|v\\|}_2}

       where :math:`\\|*\\|_2` is the 2-norm of its argument ``*``, and
       :math:`u \\cdot v` is the dot product of :math:`u` and :math:`v`.

    7. ``Y = cdist(XA, XB, 'correlation')``

       Computes the correlation distance between vectors u and v. This is

       .. math::

          1 - \\frac{(u - \\bar{u}) \\cdot (v - \\bar{v})}
                   {{\\|(u - \\bar{u})\\|}_2 {\\|(v - \\bar{v})\\|}_2}

       where :math:`\\bar{v}` is the mean of the elements of vector v,
       and :math:`x \\cdot y` is the dot product of :math:`x` and :math:`y`.


    8. ``Y = cdist(XA, XB, 'hamming')``

       Computes the normalized Hamming distance, or the proportion of
       those vector elements between two n-vectors ``u`` and ``v``
       which disagree. To save memory, the matrix ``X`` can be of type
       boolean.

    9. ``Y = cdist(XA, XB, 'jaccard')``

       Computes the Jaccard distance between the points. Given two
       vectors, ``u`` and ``v``, the Jaccard distance is the
       proportion of those elements ``u[i]`` and ``v[i]`` that
       disagree where at least one of them is non-zero.

    10. ``Y = cdist(XA, XB, 'jensenshannon')``

        Computes the Jensen-Shannon distance between two probability arrays.
        Given two probability vectors, :math:`p` and :math:`q`, the
        Jensen-Shannon distance is

        .. math::

           \\sqrt{\\frac{D(p \\parallel m) + D(q \\parallel m)}{2}}

        where :math:`m` is the pointwise mean of :math:`p` and :math:`q`
        and :math:`D` is the Kullback-Leibler divergence.

    11. ``Y = cdist(XA, XB, 'chebyshev')``

        Computes the Chebyshev distance between the points. The
        Chebyshev distance between two n-vectors ``u`` and ``v`` is the
        maximum norm-1 distance between their respective elements. More
        precisely, the distance is given by

        .. math::

           d(u,v) = \\max_i {|u_i-v_i|}.

    12. ``Y = cdist(XA, XB, 'canberra')``

        Computes the Canberra distance between the points. The
        Canberra distance between two points ``u`` and ``v`` is

        .. math::

          d(u,v) = \\sum_i \\frac{|u_i-v_i|}
                               {|u_i|+|v_i|}.

    13. ``Y = cdist(XA, XB, 'braycurtis')``

        Computes the Bray-Curtis distance between the points. The
        Bray-Curtis distance between two points ``u`` and ``v`` is


        .. math::

             d(u,v) = \\frac{\\sum_i (|u_i-v_i|)}
                           {\\sum_i (|u_i+v_i|)}

    14. ``Y = cdist(XA, XB, 'mahalanobis', VI=None)``

        Computes the Mahalanobis distance between the points. The
        Mahalanobis distance between two points ``u`` and ``v`` is
        :math:`\\sqrt{(u-v)(1/V)(u-v)^T}` where :math:`(1/V)` (the ``VI``
        variable) is the inverse covariance. If ``VI`` is not None,
        ``VI`` will be used as the inverse covariance matrix.

    15. ``Y = cdist(XA, XB, 'yule')``

        Computes the Yule distance between the boolean
        vectors. (see `yule` function documentation)

    16. ``Y = cdist(XA, XB, 'matching')``

        Synonym for 'hamming'.

    17. ``Y = cdist(XA, XB, 'dice')``

        Computes the Dice distance between the boolean vectors. (see
        `dice` function documentation)

    18. ``Y = cdist(XA, XB, 'kulczynski1')``

        Computes the kulczynski distance between the boolean
        vectors. (see `kulczynski1` function documentation)

        .. deprecated:: 1.15.0
           This metric is deprecated and will be removed in SciPy 1.17.0.
           Replace usage of ``cdist(XA, XB, 'kulczynski1')`` with
           ``1 / cdist(XA, XB, 'jaccard') - 1``.

    19. ``Y = cdist(XA, XB, 'rogerstanimoto')``

        Computes the Rogers-Tanimoto distance between the boolean
        vectors. (see `rogerstanimoto` function documentation)

    20. ``Y = cdist(XA, XB, 'russellrao')``

        Computes the Russell-Rao distance between the boolean
        vectors. (see `russellrao` function documentation)

    21. ``Y = cdist(XA, XB, 'sokalmichener')``

        Computes the Sokal-Michener distance between the boolean
        vectors. (see `sokalmichener` function documentation)

        .. deprecated:: 1.15.0
           This metric is deprecated and will be removed in SciPy 1.17.0.
           Replace usage of ``cdist(XA, XB, 'sokalmichener')`` with
           ``cdist(XA, XB, 'rogerstanimoto')``.

    22. ``Y = cdist(XA, XB, 'sokalsneath')``

        Computes the Sokal-Sneath distance between the vectors. (see
        `sokalsneath` function documentation)

    23. ``Y = cdist(XA, XB, f)``

        Computes the distance between all pairs of vectors in X
        using the user supplied 2-arity function f. For example,
        Euclidean distance between the vectors could be computed
        as follows::

          dm = cdist(XA, XB, lambda u, v: np.sqrt(((u-v)**2).sum()))

        Note that you should avoid passing a reference to one of
        the distance functions defined in this library. For example,::

          dm = cdist(XA, XB, sokalsneath)

        would calculate the pair-wise distances between the vectors in
        X using the Python function `sokalsneath`. This would result in
        sokalsneath being called :math:`{n \\choose 2}` times, which
        is inefficient. Instead, the optimized C version is more
        efficient, and we call it using the following syntax::

          dm = cdist(XA, XB, 'sokalsneath')

    Examples
    --------
    Find the Euclidean distances between four 2-D coordinates:

    >>> from scipy.spatial import distance
    >>> import numpy as np
    >>> coords = [(35.0456, -85.2672),
    ...           (35.1174, -89.9711),
    ...           (35.9728, -83.9422),
    ...           (36.1667, -86.7833)]
    >>> distance.cdist(coords, coords, 'euclidean')
    array([[ 0.    ,  4.7044,  1.6172,  1.8856],
           [ 4.7044,  0.    ,  6.0893,  3.3561],
           [ 1.6172,  6.0893,  0.    ,  2.8477],
           [ 1.8856,  3.3561,  2.8477,  0.    ]])


    Find the Manhattan distance from a 3-D point to the corners of the unit
    cube:

    >>> a = np.array([[0, 0, 0],
    ...               [0, 0, 1],
    ...               [0, 1, 0],
    ...               [0, 1, 1],
    ...               [1, 0, 0],
    ...               [1, 0, 1],
    ...               [1, 1, 0],
    ...               [1, 1, 1]])
    >>> b = np.array([[ 0.1,  0.2,  0.4]])
    >>> distance.cdist(a, b, 'cityblock')
    array([[ 0.7],
           [ 0.9],
           [ 1.3],
           [ 1.5],
           [ 1.5],
           [ 1.7],
           [ 2.1],
           [ 2.3]])

    """
    # You can also call this as:
    #     Y = cdist(XA, XB, 'test_abc')
    # where 'abc' is the metric being tested.  This computes the distance
    # between all pairs of vectors in XA and XB using the distance metric 'abc'
    # but with a more succinct, verifiable, but less efficient implementation.

    XA = np.asarray(XA)
    XB = np.asarray(XB)

    s = XA.shape
    sB = XB.shape

    if len(s) != 2:
        raise ValueError('XA must be a 2-dimensional array.')
    if len(sB) != 2:
        raise ValueError('XB must be a 2-dimensional array.')
    if s[1] != sB[1]:
        raise ValueError('XA and XB must have the same number of columns '
                         '(i.e. feature dimension.)')

    mA = s[0]
    mB = sB[0]
    n = s[1]

    if callable(metric):
        mstr = getattr(metric, '__name__', 'Unknown')
        metric_info = _METRIC_ALIAS.get(mstr, None)
        if metric_info is not None:
            XA, XB, typ, kwargs = _validate_cdist_input(
                XA, XB, mA, mB, n, metric_info, **kwargs)
        return _cdist_callable(XA, XB, metric=metric, out=out, **kwargs)
    elif isinstance(metric, str):
        mstr = metric.lower()
        metric_info = _METRIC_ALIAS.get(mstr, None)
        if metric_info is not None:
            cdist_fn = metric_info.cdist_func
            return cdist_fn(XA, XB, out=out, **kwargs)
        elif mstr.startswith("test_"):
            metric_info = _TEST_METRICS.get(mstr, None)
            if metric_info is None:
                raise ValueError(f'Unknown "Test" Distance Metric: {mstr[5:]}')
            XA, XB, typ, kwargs = _validate_cdist_input(
                XA, XB, mA, mB, n, metric_info, **kwargs)
            return _cdist_callable(
                XA, XB, metric=metric_info.dist_func, out=out, **kwargs)
        else:
            raise ValueError(f'Unknown Distance Metric: {mstr}')
    else:
        raise TypeError('2nd argument metric must be a string identifier '
                        'or a function.')
<|endoftext|>"""
This module provides functions to perform full Procrustes analysis.

This code was originally written by Justin Kucynski and ported over from
scikit-bio by Yoshiki Vazquez-Baeza.
"""

import numpy as np
from scipy.linalg import orthogonal_procrustes


__all__ = ['procrustes']


def procrustes(data1, data2):
    r"""Procrustes analysis, a similarity test for two data sets.

    Each input matrix is a set of points or vectors (the rows of the matrix).
    The dimension of the space is the number of columns of each matrix. Given
    two identically sized matrices, procrustes standardizes both such that:

    - :math:`tr(AA^{T}) = 1`.

    - Both sets of points are centered around the origin.

    Procrustes ([1]_, [2]_) then applies the optimal transform to the second
    matrix (including scaling/dilation, rotations, and reflections) to minimize
    :math:`M^{2}=\sum(data1-data2)^{2}`, or the sum of the squares of the
    pointwise differences between the two input datasets.

    This function was not designed to handle datasets with different numbers of
    datapoints (rows).  If two data sets have different dimensionality
    (different number of columns), simply add columns of zeros to the smaller
    of the two.

    Parameters
    ----------
    data1 : array_like
        Matrix, n rows represent points in k (columns) space `data1` is the
        reference data, after it is standardised, the data from `data2` will be
        transformed to fit the pattern in `data1` (must have >1 unique points).
    data2 : array_like
        n rows of data in k space to be fit to `data1`.  Must be the  same
        shape ``(numrows, numcols)`` as data1 (must have >1 unique points).

    Returns
    -------
    mtx1 : array_like
        A standardized version of `data1`.
    mtx2 : array_like
        The orientation of `data2` that best fits `data1`. Centered, but not
        necessarily :math:`tr(AA^{T}) = 1`.
    disparity : float
        :math:`M^{2}` as defined above.

    Raises
    ------
    ValueError
        If the input arrays are not two-dimensional.
        If the shape of the input arrays is different.
        If the input arrays have zero columns or zero rows.

    See Also
    --------
    scipy.linalg.orthogonal_procrustes
    scipy.spatial.distance.directed_hausdorff : Another similarity test
      for two data sets

    Notes
    -----
    - The disparity should not depend on the order of the input matrices, but
      the output matrices will, as only the first output matrix is guaranteed
      to be scaled such that :math:`tr(AA^{T}) = 1`.

    - Duplicate data points are generally ok, duplicating a data point will
      increase its effect on the procrustes fit.

    - The disparity scales as the number of points per input matrix.

    References
    ----------
    .. [1] Krzanowski, W. J. (2000). "Principles of Multivariate analysis".
    .. [2] Gower, J. C. (1975). "Generalized procrustes analysis".

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.spatial import procrustes

    The matrix ``b`` is a rotated, shifted, scaled and mirrored version of
    ``a`` here:

    >>> a = np.array([[1, 3], [1, 2], [1, 1], [2, 1]], 'd')
    >>> b = np.array([[4, -2], [4, -4], [4, -6], [2, -6]], 'd')
    >>> mtx1, mtx2, disparity = procrustes(a, b)
    >>> round(disparity)
    0

    """
    mtx1 = np.array(data1, dtype=np.float64, copy=True)
    mtx2 = np.array(data2, dtype=np.float64, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s

    # measure the dissimilarity between the two datasets
    disparity = np.sum(np.square(mtx1 - mtx2))

    return mtx1, mtx2, disparity

<|endoftext|>import numpy as np
from scipy._lib.decorator import decorator as _decorator

__all__ = ['delaunay_plot_2d', 'convex_hull_plot_2d', 'voronoi_plot_2d']


@_decorator
def _held_figure(func, obj, ax=None, **kw):
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
        return func(obj, ax=ax, **kw)

    # As of matplotlib 2.0, the "hold" mechanism is deprecated.
    # When matplotlib 1.x is no longer supported, this check can be removed.
    was_held = getattr(ax, 'ishold', lambda: True)()
    if was_held:
        return func(obj, ax=ax, **kw)
    try:
        ax.hold(True)
        return func(obj, ax=ax, **kw)
    finally:
        ax.hold(was_held)


def _adjust_bounds(ax, points):
    margin = 0.1 * np.ptp(points, axis=0)
    xy_min = points.min(axis=0) - margin
    xy_max = points.max(axis=0) + margin
    ax.set_xlim(xy_min[0], xy_max[0])
    ax.set_ylim(xy_min[1], xy_max[1])


@_held_figure
def delaunay_plot_2d(tri, ax=None):
    """
    Plot the given Delaunay triangulation in 2-D

    Parameters
    ----------
    tri : scipy.spatial.Delaunay instance
        Triangulation to plot
    ax : matplotlib.axes.Axes instance, optional
        Axes to plot on

    Returns
    -------
    fig : matplotlib.figure.Figure instance
        Figure for the plot

    See Also
    --------
    Delaunay
    matplotlib.pyplot.triplot

    Notes
    -----
    Requires Matplotlib.

    Examples
    --------

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.spatial import Delaunay, delaunay_plot_2d

    The Delaunay triangulation of a set of random points:

    >>> rng = np.random.default_rng()
    >>> points = rng.random((30, 2))
    >>> tri = Delaunay(points)

    Plot it:

    >>> _ = delaunay_plot_2d(tri)
    >>> plt.show()

    """
    if tri.points.shape[1] != 2:
        raise ValueError("Delaunay triangulation is not 2-D")

    x, y = tri.points.T
    ax.plot(x, y, 'o')
    ax.triplot(x, y, tri.simplices.copy())

    _adjust_bounds(ax, tri.points)

    return ax.figure


@_held_figure
def convex_hull_plot_2d(hull, ax=None):
    """
    Plot the given convex hull diagram in 2-D

    Parameters
    ----------
    hull : scipy.spatial.ConvexHull instance
        Convex hull to plot
    ax : matplotlib.axes.Axes instance, optional
        Axes to plot on

    Returns
    -------
    fig : matplotlib.figure.Figure instance
        Figure for the plot

    See Also
    --------
    ConvexHull

    Notes
    -----
    Requires Matplotlib.


    Examples
    --------

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.spatial import ConvexHull, convex_hull_plot_2d

    The convex hull of a random set of points:

    >>> rng = np.random.default_rng()
    >>> points = rng.random((30, 2))
    >>> hull = ConvexHull(points)

    Plot it:

    >>> _ = convex_hull_plot_2d(hull)
    >>> plt.show()

    """
    from matplotlib.collections import LineCollection

    if hull.points.shape[1] != 2:
        raise ValueError("Convex hull is not 2-D")

    ax.plot(hull.points[:, 0], hull.points[:, 1], 'o')
    line_segments = [hull.points[simplex] for simplex in hull.simplices]
    ax.add_collection(LineCollection(line_segments,
                                     colors='k',
                                     linestyle='solid'))
    _adjust_bounds(ax, hull.points)

    return ax.figure


@_held_figure
def voronoi_plot_2d(vor, ax=None, **kw):
    """
    Plot the given Voronoi diagram in 2-D

    Parameters
    ----------
    vor : scipy.spatial.Voronoi instance
        Diagram to plot
    ax : matplotlib.axes.Axes instance, optional
        Axes to plot on
    show_points : bool, optional
        Add the Voronoi points to the plot.
    show_vertices : bool, optional
        Add the Voronoi vertices to the plot.
    line_colors : string, optional
        Specifies the line color for polygon boundaries
    line_width : float, optional
        Specifies the line width for polygon boundaries
    line_alpha : float, optional
        Specifies the line alpha for polygon boundaries
    point_size : float, optional
        Specifies the size of points

    Returns
    -------
    fig : matplotlib.figure.Figure instance
        Figure for the plot

    See Also
    --------
    Voronoi

    Notes
    -----
    Requires Matplotlib. For degenerate input, including collinearity and
    other violations of general position, it may be preferable to
    calculate the Voronoi diagram with Qhull options ``QJ`` for random
    joggling, or ``Qt`` to enforce triangulated output. Otherwise, some
    Voronoi regions may not be visible.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.spatial import Voronoi, voronoi_plot_2d

    Create a set of points for the example:

    >>> rng = np.random.default_rng()
    >>> points = rng.random((10,2))

    Generate the Voronoi diagram for the points:

    >>> vor = Voronoi(points)

    Use `voronoi_plot_2d` to plot the diagram:

    >>> fig = voronoi_plot_2d(vor)

    Use `voronoi_plot_2d` to plot the diagram again, with some settings
    customized:

    >>> fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',
    ...                       line_width=2, line_alpha=0.6, point_size=2)
    >>> plt.show()

    """
    from matplotlib.collections import LineCollection

    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi diagram is not 2-D")

    if kw.get('show_points', True):
        point_size = kw.get('point_size', None)
        ax.plot(vor.points[:, 0], vor.points[:, 1], '.', markersize=point_size)
    if kw.get('show_vertices', True):
        ax.plot(vor.vertices[:, 0], vor.vertices[:, 1], 'o')

    line_colors = kw.get('line_colors', 'k')
    line_width = kw.get('line_width', 1.0)
    line_alpha = kw.get('line_alpha', 1.0)

    center = vor.points.mean(axis=0)
    ptp_bound = np.ptp(vor.points, axis=0)

    finite_segments = []
    infinite_segments = []
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            finite_segments.append(vor.vertices[simplex])
        else:
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            if (vor.furthest_site):
                direction = -direction
            aspect_factor = abs(ptp_bound.max() / ptp_bound.min())
            far_point = vor.vertices[i] + direction * ptp_bound.max() * aspect_factor

            infinite_segments.append([vor.vertices[i], far_point])

    ax.add_collection(LineCollection(finite_segments,
                                     colors=line_colors,
                                     lw=line_width,
                                     alpha=line_alpha,
                                     linestyle='solid'))
    ax.add_collection(LineCollection(infinite_segments,
                                     colors=line_colors,
                                     lw=line_width,
                                     alpha=line_alpha,
                                     linestyle='dashed'))

    _adjust_bounds(ax, vor.points)

    return ax.figure
<|endoftext|>"""
=============================================================
Spatial algorithms and data structures (:mod:`scipy.spatial`)
=============================================================

.. currentmodule:: scipy.spatial

.. toctree::
   :hidden:

   spatial.distance

Spatial transformations
=======================

These are contained in the `scipy.spatial.transform` submodule.

Nearest-neighbor queries
========================
.. autosummary::
   :toctree: generated/

   KDTree      -- class for efficient nearest-neighbor queries
   cKDTree     -- class for efficient nearest-neighbor queries (faster implementation)
   Rectangle

Distance metrics
================

Distance metrics are contained in the :mod:`scipy.spatial.distance` submodule.

Delaunay triangulation, convex hulls, and Voronoi diagrams
==========================================================

.. autosummary::
   :toctree: generated/

   Delaunay    -- compute Delaunay triangulation of input points
   ConvexHull  -- compute a convex hull for input points
   Voronoi     -- compute a Voronoi diagram hull from input points
   SphericalVoronoi -- compute a Voronoi diagram from input points on the surface of a sphere
   HalfspaceIntersection -- compute the intersection points of input halfspaces

Plotting helpers
================

.. autosummary::
   :toctree: generated/

   delaunay_plot_2d     -- plot 2-D triangulation
   convex_hull_plot_2d  -- plot 2-D convex hull
   voronoi_plot_2d      -- plot 2-D Voronoi diagram

.. seealso:: :ref:`Tutorial <qhulltutorial>`


Simplex representation
======================
The simplices (triangles, tetrahedra, etc.) appearing in the Delaunay
tessellation (N-D simplices), convex hull facets, and Voronoi ridges
(N-1-D simplices) are represented in the following scheme::

    tess = Delaunay(points)
    hull = ConvexHull(points)
    voro = Voronoi(points)

    # coordinates of the jth vertex of the ith simplex
    tess.points[tess.simplices[i, j], :]        # tessellation element
    hull.points[hull.simplices[i, j], :]        # convex hull facet
    voro.vertices[voro.ridge_vertices[i, j], :] # ridge between Voronoi cells

For Delaunay triangulations and convex hulls, the neighborhood
structure of the simplices satisfies the condition:
``tess.neighbors[i,j]`` is the neighboring simplex of the ith
simplex, opposite to the ``j``-vertex. It is -1 in case of no neighbor.

Convex hull facets also define a hyperplane equation::

    (hull.equations[i,:-1] * coord).sum() + hull.equations[i,-1] == 0

Similar hyperplane equations for the Delaunay triangulation correspond
to the convex hull facets on the corresponding N+1-D
paraboloid.

The Delaunay triangulation objects offer a method for locating the
simplex containing a given point, and barycentric coordinate
computations.

Functions
---------

.. autosummary::
   :toctree: generated/

   tsearch
   distance_matrix
   minkowski_distance
   minkowski_distance_p
   procrustes
   geometric_slerp

Warnings / Errors used in :mod:`scipy.spatial`
----------------------------------------------
.. autosummary::
   :toctree: generated/

   QhullError
"""  # noqa: E501

from ._kdtree import *
from ._ckdtree import *  # type: ignore[import-not-found]
from ._qhull import *
from ._spherical_voronoi import SphericalVoronoi
from ._plotutils import *
from ._procrustes import procrustes
from ._geometric_slerp import geometric_slerp

# Deprecated namespaces, to be removed in v2.0.0
from . import ckdtree, kdtree, qhull

__all__ = [s for s in dir() if not s.startswith('_')]

from . import distance, transform

__all__ += ['distance', 'transform']

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.spatial` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'ConvexHull',
    'Delaunay',
    'HalfspaceIntersection',
    'QhullError',
    'Voronoi',
    'tsearch',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="spatial", module="qhull",
                                   private_modules=["_qhull"], all=__all__,
                                   attribute=name)
<|endoftext|>__all__ = ['geometric_slerp']

import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.distance import euclidean

if TYPE_CHECKING:
    import numpy.typing as npt


def _geometric_slerp(start, end, t):
    # create an orthogonal basis using QR decomposition
    basis = np.vstack([start, end])
    Q, R = np.linalg.qr(basis.T)
    signs = 2 * (np.diag(R) >= 0) - 1
    Q = Q.T * signs.T[:, np.newaxis]
    R = R.T * signs.T[:, np.newaxis]

    # calculate the angle between `start` and `end`
    c = np.dot(start, end)
    s = np.linalg.det(R)
    omega = np.arctan2(s, c)

    # interpolate
    start, end = Q
    s = np.sin(t * omega)
    c = np.cos(t * omega)
    return start * c[:, np.newaxis] + end * s[:, np.newaxis]


def geometric_slerp(
    start: "npt.ArrayLike",
    end: "npt.ArrayLike",
    t: "npt.ArrayLike",
    tol: float = 1e-7,
) -> np.ndarray:
    """
    Geometric spherical linear interpolation.

    The interpolation occurs along a unit-radius
    great circle arc in arbitrary dimensional space.

    Parameters
    ----------
    start : (n_dimensions, ) array-like
        Single n-dimensional input coordinate in a 1-D array-like
        object. `n` must be greater than 1.
    end : (n_dimensions, ) array-like
        Single n-dimensional input coordinate in a 1-D array-like
        object. `n` must be greater than 1.
    t : float or (n_points,) 1D array-like
        A float or 1D array-like of doubles representing interpolation
        parameters, with values required in the inclusive interval
        between 0 and 1. A common approach is to generate the array
        with ``np.linspace(0, 1, n_pts)`` for linearly spaced points.
        Ascending, descending, and scrambled orders are permitted.
    tol : float
        The absolute tolerance for determining if the start and end
        coordinates are antipodes.

    Returns
    -------
    result : (t.size, D)
        An array of doubles containing the interpolated
        spherical path and including start and
        end when 0 and 1 t are used. The
        interpolated values should correspond to the
        same sort order provided in the t array. The result
        may be 1-dimensional if ``t`` is a float.

    Raises
    ------
    ValueError
        If ``start`` and ``end`` are antipodes, not on the
        unit n-sphere, or for a variety of degenerate conditions.

    See Also
    --------
    scipy.spatial.transform.Slerp : 3-D Slerp that works with quaternions

    Notes
    -----
    The implementation is based on the mathematical formula provided in [1]_,
    and the first known presentation of this algorithm, derived from study of
    4-D geometry, is credited to Glenn Davis in a footnote of the original
    quaternion Slerp publication by Ken Shoemake [2]_.

    .. versionadded:: 1.5.0

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Slerp#Geometric_Slerp
    .. [2] Ken Shoemake (1985) Animating rotation with quaternion curves.
           ACM SIGGRAPH Computer Graphics, 19(3): 245-254.

    Examples
    --------
    Interpolate four linearly-spaced values on the circumference of
    a circle spanning 90 degrees:

    >>> import numpy as np
    >>> from scipy.spatial import geometric_slerp
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> start = np.array([1, 0])
    >>> end = np.array([0, 1])
    >>> t_vals = np.linspace(0, 1, 4)
    >>> result = geometric_slerp(start,
    ...                          end,
    ...                          t_vals)

    The interpolated results should be at 30 degree intervals
    recognizable on the unit circle:

    >>> ax.scatter(result[...,0], result[...,1], c='k')
    >>> circle = plt.Circle((0, 0), 1, color='grey')
    >>> ax.add_artist(circle)
    >>> ax.set_aspect('equal')
    >>> plt.show()

    Attempting to interpolate between antipodes on a circle is
    ambiguous because there are two possible paths, and on a
    sphere there are infinite possible paths on the geodesic surface.
    Nonetheless, one of the ambiguous paths is returned along
    with a warning:

    >>> opposite_pole = np.array([-1, 0])
    >>> with np.testing.suppress_warnings() as sup:
    ...     sup.filter(UserWarning)
    ...     geometric_slerp(start,
    ...                     opposite_pole,
    ...                     t_vals)
    array([[ 1.00000000e+00,  0.00000000e+00],
           [ 5.00000000e-01,  8.66025404e-01],
           [-5.00000000e-01,  8.66025404e-01],
           [-1.00000000e+00,  1.22464680e-16]])

    Extend the original example to a sphere and plot interpolation
    points in 3D:

    >>> from mpl_toolkits.mplot3d import proj3d
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')

    Plot the unit sphere for reference (optional):

    >>> u = np.linspace(0, 2 * np.pi, 100)
    >>> v = np.linspace(0, np.pi, 100)
    >>> x = np.outer(np.cos(u), np.sin(v))
    >>> y = np.outer(np.sin(u), np.sin(v))
    >>> z = np.outer(np.ones(np.size(u)), np.cos(v))
    >>> ax.plot_surface(x, y, z, color='y', alpha=0.1)

    Interpolating over a larger number of points
    may provide the appearance of a smooth curve on
    the surface of the sphere, which is also useful
    for discretized integration calculations on a
    sphere surface:

    >>> start = np.array([1, 0, 0])
    >>> end = np.array([0, 0, 1])
    >>> t_vals = np.linspace(0, 1, 200)
    >>> result = geometric_slerp(start,
    ...                          end,
    ...                          t_vals)
    >>> ax.plot(result[...,0],
    ...         result[...,1],
    ...         result[...,2],
    ...         c='k')
    >>> plt.show()
    """

    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    t = np.asarray(t)

    if t.ndim > 1:
        raise ValueError("The interpolation parameter "
                         "value must be one dimensional.")

    if start.ndim != 1 or end.ndim != 1:
        raise ValueError("Start and end coordinates "
                         "must be one-dimensional")

    if start.size != end.size:
        raise ValueError("The dimensions of start and "
                         "end must match (have same size)")

    if start.size < 2 or end.size < 2:
        raise ValueError("The start and end coordinates must "
                         "both be in at least two-dimensional "
                         "space")

    if np.array_equal(start, end):
        return np.linspace(start, start, t.size)

    # for points that violate equation for n-sphere
    for coord in [start, end]:
        if not np.allclose(np.linalg.norm(coord), 1.0,
                           rtol=1e-9,
                           atol=0):
            raise ValueError("start and end are not"
                             " on a unit n-sphere")

    if not isinstance(tol, float):
        raise ValueError("tol must be a float")
    else:
        tol = np.fabs(tol)

    coord_dist = euclidean(start, end)

    # diameter of 2 within tolerance means antipodes, which is a problem
    # for all unit n-spheres (even the 0-sphere would have an ambiguous path)
    if np.allclose(coord_dist, 2.0, rtol=0, atol=tol):
        warnings.warn("start and end are antipodes "
                      "using the specified tolerance; "
                      "this may cause ambiguous slerp paths",
                      stacklevel=2)

    t = np.asarray(t, dtype=np.float64)

    if t.size == 0:
        return np.empty((0, start.size))

    if t.min() < 0 or t.max() > 1:
        raise ValueError("interpolation parameter must be in [0, 1]")

    if t.ndim == 0:
        return _geometric_slerp(start,
                                end,
                                np.atleast_1d(t)).ravel()
    else:
        return _geometric_slerp(start,
                                end,
                                t)
<|endoftext|>"""
Spherical Voronoi Code

.. versionadded:: 0.18.0

"""
#
# Copyright (C)  Tyler Reddy, Ross Hemsley, Edd Edmondson,
#                Nikolai Nowaczyk, Joe Pitt-Francis, 2015.
#
# Distributed under the same BSD license as SciPy.
#

import numpy as np
import scipy
from . import _voronoi
from scipy.spatial import cKDTree  # type: ignore[attr-defined]

__all__ = ['SphericalVoronoi']


def calculate_solid_angles(R):
    """Calculates the solid angles of plane triangles. Implements the method of
    Van Oosterom and Strackee [VanOosterom]_ with some modifications. Assumes
    that input points have unit norm."""
    # Original method uses a triple product `R1 . (R2 x R3)` for the numerator.
    # This is equal to the determinant of the matrix [R1 R2 R3], which can be
    # computed with better stability.
    numerator = np.linalg.det(R)
    denominator = 1 + (np.einsum('ij,ij->i', R[:, 0], R[:, 1]) +
                       np.einsum('ij,ij->i', R[:, 1], R[:, 2]) +
                       np.einsum('ij,ij->i', R[:, 2], R[:, 0]))
    return np.abs(2 * np.arctan2(numerator, denominator))


class SphericalVoronoi:
    """ Voronoi diagrams on the surface of a sphere.

    .. versionadded:: 0.18.0

    Parameters
    ----------
    points : ndarray of floats, shape (npoints, ndim)
        Coordinates of points from which to construct a spherical
        Voronoi diagram.
    radius : float, optional
        Radius of the sphere (Default: 1)
    center : ndarray of floats, shape (ndim,)
        Center of sphere (Default: origin)
    threshold : float
        Threshold for detecting duplicate points and
        mismatches between points and sphere parameters.
        (Default: 1e-06)

    Attributes
    ----------
    points : double array of shape (npoints, ndim)
        the points in `ndim` dimensions to generate the Voronoi diagram from
    radius : double
        radius of the sphere
    center : double array of shape (ndim,)
        center of the sphere
    vertices : double array of shape (nvertices, ndim)
        Voronoi vertices corresponding to points
    regions : list of list of integers of shape (npoints, _ )
        the n-th entry is a list consisting of the indices
        of the vertices belonging to the n-th point in points

    Methods
    -------
    calculate_areas
        Calculates the areas of the Voronoi regions. For 2D point sets, the
        regions are circular arcs. The sum of the areas is ``2 * pi * radius``.
        For 3D point sets, the regions are spherical polygons. The sum of the
        areas is ``4 * pi * radius**2``.

    Raises
    ------
    ValueError
        If there are duplicates in `points`.
        If the provided `radius` is not consistent with `points`.

    Notes
    -----
    The spherical Voronoi diagram algorithm proceeds as follows. The Convex
    Hull of the input points (generators) is calculated, and is equivalent to
    their Delaunay triangulation on the surface of the sphere [Caroli]_.
    The Convex Hull neighbour information is then used to
    order the Voronoi region vertices around each generator. The latter
    approach is substantially less sensitive to floating point issues than
    angle-based methods of Voronoi region vertex sorting.

    Empirical assessment of spherical Voronoi algorithm performance suggests
    quadratic time complexity (loglinear is optimal, but algorithms are more
    challenging to implement).

    References
    ----------
    .. [Caroli] Caroli et al. Robust and Efficient Delaunay triangulations of
                points on or close to a sphere. Research Report RR-7004, 2009.

    .. [VanOosterom] Van Oosterom and Strackee. The solid angle of a plane
                     triangle. IEEE Transactions on Biomedical Engineering,
                     2, 1983, pp 125--126.

    See Also
    --------
    Voronoi : Conventional Voronoi diagrams in N dimensions.

    Examples
    --------
    Do some imports and take some points on a cube:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.spatial import SphericalVoronoi, geometric_slerp
    >>> from mpl_toolkits.mplot3d import proj3d
    >>> # set input data
    >>> points = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0],
    ...                    [0, 1, 0], [0, -1, 0], [-1, 0, 0], ])

    Calculate the spherical Voronoi diagram:

    >>> radius = 1
    >>> center = np.array([0, 0, 0])
    >>> sv = SphericalVoronoi(points, radius, center)

    Generate plot:

    >>> # sort vertices (optional, helpful for plotting)
    >>> sv.sort_vertices_of_regions()
    >>> t_vals = np.linspace(0, 1, 2000)
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> # plot the unit sphere for reference (optional)
    >>> u = np.linspace(0, 2 * np.pi, 100)
    >>> v = np.linspace(0, np.pi, 100)
    >>> x = np.outer(np.cos(u), np.sin(v))
    >>> y = np.outer(np.sin(u), np.sin(v))
    >>> z = np.outer(np.ones(np.size(u)), np.cos(v))
    >>> ax.plot_surface(x, y, z, color='y', alpha=0.1)
    >>> # plot generator points
    >>> ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b')
    >>> # plot Voronoi vertices
    >>> ax.scatter(sv.vertices[:, 0], sv.vertices[:, 1], sv.vertices[:, 2],
    ...                    c='g')
    >>> # indicate Voronoi regions (as Euclidean polygons)
    >>> for region in sv.regions:
    ...    n = len(region)
    ...    for i in range(n):
    ...        start = sv.vertices[region][i]
    ...        end = sv.vertices[region][(i + 1) % n]
    ...        result = geometric_slerp(start, end, t_vals)
    ...        ax.plot(result[..., 0],
    ...                result[..., 1],
    ...                result[..., 2],
    ...                c='k')
    >>> ax.azim = 10
    >>> ax.elev = 40
    >>> _ = ax.set_xticks([])
    >>> _ = ax.set_yticks([])
    >>> _ = ax.set_zticks([])
    >>> fig.set_size_inches(4, 4)
    >>> plt.show()

    """
    def __init__(self, points, radius=1, center=None, threshold=1e-06):

        if radius is None:
            raise ValueError('`radius` is `None`. '
                             'Please provide a floating point number '
                             '(i.e. `radius=1`).')

        self.radius = float(radius)
        self.points = np.array(points).astype(np.float64)
        self._dim = self.points.shape[1]
        if center is None:
            self.center = np.zeros(self._dim)
        else:
            self.center = np.array(center, dtype=float)

        # test degenerate input
        self._rank = np.linalg.matrix_rank(self.points - self.points[0],
                                           tol=threshold * self.radius)
        if self._rank < self._dim:
            raise ValueError(f"Rank of input points must be at least {self._dim}")

        if cKDTree(self.points).query_pairs(threshold * self.radius):
            raise ValueError("Duplicate generators present.")

        radii = np.linalg.norm(self.points - self.center, axis=1)
        max_discrepancy = np.abs(radii - self.radius).max()
        if max_discrepancy >= threshold * self.radius:
            raise ValueError("Radius inconsistent with generators.")

        self._calc_vertices_regions()

    def _calc_vertices_regions(self):
        """
        Calculates the Voronoi vertices and regions of the generators stored
        in self.points. The vertices will be stored in self.vertices and the
        regions in self.regions.

        This algorithm was discussed at PyData London 2015 by
        Tyler Reddy, Ross Hemsley and Nikolai Nowaczyk
        """
        # get Convex Hull
        conv = scipy.spatial.ConvexHull(self.points)
        # get circumcenters of Convex Hull triangles from facet equations
        # for 3D input circumcenters will have shape: (2N-4, 3)
        self.vertices = self.radius * conv.equations[:, :-1] + self.center
        self._simplices = conv.simplices
        # calculate regions from triangulation
        # for 3D input simplex_indices will have shape: (2N-4,)
        simplex_indices = np.arange(len(self._simplices))
        # for 3D input tri_indices will have shape: (6N-12,)
        tri_indices = np.column_stack([simplex_indices] * self._dim).ravel()
        # for 3D input point_indices will have shape: (6N-12,)
        point_indices = self._simplices.ravel()
        # for 3D input indices will have shape: (6N-12,)
        indices = np.argsort(point_indices, kind='mergesort')
        # for 3D input flattened_groups will have shape: (6N-12,)
        flattened_groups = tri_indices[indices].astype(np.intp)
        # intervals will have shape: (N+1,)
        intervals = np.cumsum(np.bincount(point_indices + 1))
        # split flattened groups to get nested list of unsorted regions
        groups = [list(flattened_groups[intervals[i]:intervals[i + 1]])
                  for i in range(len(intervals) - 1)]
        self.regions = groups

    def sort_vertices_of_regions(self):
        """Sort indices of the vertices to be (counter-)clockwise ordered.

        Raises
        ------
        TypeError
            If the points are not three-dimensional.

        Notes
        -----
        For each region in regions, it sorts the indices of the Voronoi
        vertices such that the resulting points are in a clockwise or
        counterclockwise order around the generator point.

        This is done as follows: Recall that the n-th region in regions
        surrounds the n-th generator in points and that the k-th
        Voronoi vertex in vertices is the circumcenter of the k-th triangle
        in self._simplices.  For each region n, we choose the first triangle
        (=Voronoi vertex) in self._simplices and a vertex of that triangle
        not equal to the center n. These determine a unique neighbor of that
        triangle, which is then chosen as the second triangle. The second
        triangle will have a unique vertex not equal to the current vertex or
        the center. This determines a unique neighbor of the second triangle,
        which is then chosen as the third triangle and so forth. We proceed
        through all the triangles (=Voronoi vertices) belonging to the
        generator in points and obtain a sorted version of the vertices
        of its surrounding region.
        """
        if self._dim != 3:
            raise TypeError("Only supported for three-dimensional point sets")
        _voronoi.sort_vertices_of_regions(self._simplices, self.regions)

    def _calculate_areas_3d(self):
        self.sort_vertices_of_regions()
        sizes = [len(region) for region in self.regions]
        csizes = np.cumsum(sizes)
        num_regions = csizes[-1]

        # We create a set of triangles consisting of one point and two Voronoi
        # vertices. The vertices of each triangle are adjacent in the sorted
        # regions list.
        point_indices = [i for i, size in enumerate(sizes)
                         for j in range(size)]

        nbrs1 = np.array([r for region in self.regions for r in region])

        # The calculation of nbrs2 is a vectorized version of:
        # np.array([r for region in self.regions for r in np.roll(region, 1)])
        nbrs2 = np.roll(nbrs1, 1)
        indices = np.roll(csizes, 1)
        indices[0] = 0
        nbrs2[indices] = nbrs1[csizes - 1]

        # Normalize points and vertices.
        pnormalized = (self.points - self.center) / self.radius
        vnormalized = (self.vertices - self.center) / self.radius

        # Create the complete set of triangles and calculate their solid angles
        triangles = np.hstack([pnormalized[point_indices],
                               vnormalized[nbrs1],
                               vnormalized[nbrs2]
                               ]).reshape((num_regions, 3, 3))
        triangle_solid_angles = calculate_solid_angles(triangles)

        # Sum the solid angles of the triangles in each region
        solid_angles = np.cumsum(triangle_solid_angles)[csizes - 1]
        solid_angles[1:] -= solid_angles[:-1]

        # Get polygon areas using A = omega * r**2
        return solid_angles * self.radius**2

    def _calculate_areas_2d(self):
        # Find start and end points of arcs
        arcs = self.points[self._simplices] - self.center

        # Calculate the angle subtended by arcs
        d = np.sum((arcs[:, 1] - arcs[:, 0]) ** 2, axis=1)
        theta = np.arccos(1 - (d / (2 * (self.radius ** 2))))

        # Get areas using A = r * theta
        areas = self.radius * theta

        # Correct arcs which go the wrong way (single-hemisphere inputs)
        signs = np.sign(np.einsum('ij,ij->i', arcs[:, 0],
                                              self.vertices - self.center))
        indices = np.where(signs < 0)
        areas[indices] = 2 * np.pi * self.radius - areas[indices]
        return areas

    def calculate_areas(self):
        """Calculates the areas of the Voronoi regions.

        For 2D point sets, the regions are circular arcs. The sum of the areas
        is ``2 * pi * radius``.

        For 3D point sets, the regions are spherical polygons. The sum of the
        areas is ``4 * pi * radius**2``.

        .. versionadded:: 1.5.0

        Returns
        -------
        areas : double array of shape (npoints,)
            The areas of the Voronoi regions.
        """
        if self._dim == 2:
            return self._calculate_areas_2d()
        elif self._dim == 3:
            return self._calculate_areas_3d()
        else:
            raise TypeError("Only supported for 2D and 3D point sets")
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.spatial` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = ["cKDTree"]  # noqa: F822


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="spatial", module="ckdtree",
                                   private_modules=["_ckdtree"], all=__all__,
                                   attribute=name)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.spatial` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'KDTree',
    'Rectangle',
    'cKDTree',
    'distance_matrix',
    'minkowski_distance',
    'minkowski_distance_p',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="spatial", module="kdtree",
                                   private_modules=["_kdtree"], all=__all__,
                                   attribute=name)
<|endoftext|># Copyright Anne M. Archibald 2008
# Released under the scipy license
import numpy as np
from ._ckdtree import cKDTree, cKDTreeNode  # type: ignore[import-not-found]

__all__ = ['minkowski_distance_p', 'minkowski_distance',
           'distance_matrix',
           'Rectangle', 'KDTree']


def minkowski_distance_p(x, y, p=2):
    """Compute the pth power of the L**p distance between two arrays.

    For efficiency, this function computes the L**p distance but does
    not extract the pth root. If `p` is 1 or infinity, this is equal to
    the actual L**p distance.

    The last dimensions of `x` and `y` must be the same length.  Any
    other dimensions must be compatible for broadcasting.

    Parameters
    ----------
    x : (..., K) array_like
        Input array.
    y : (..., K) array_like
        Input array.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.

    Returns
    -------
    dist : ndarray
        pth power of the distance between the input arrays.

    Examples
    --------
    >>> from scipy.spatial import minkowski_distance_p
    >>> minkowski_distance_p([[0, 0], [0, 0]], [[1, 1], [0, 1]])
    array([2., 1.])

    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Find smallest common datatype with float64 (return type of this
    # function) - addresses #10262.
    # Don't just cast to float64 for complex input case.
    common_datatype = np.promote_types(np.promote_types(x.dtype, y.dtype),
                                       'float64')

    # Make sure x and y are NumPy arrays of correct datatype.
    x = x.astype(common_datatype)
    y = y.astype(common_datatype)

    if p == np.inf:
        return np.amax(np.abs(y-x), axis=-1)
    elif p == 1:
        return np.sum(np.abs(y-x), axis=-1)
    else:
        return np.sum(np.abs(y-x)**p, axis=-1)


def minkowski_distance(x, y, p=2):
    """Compute the L**p distance between two arrays.

    The last dimensions of `x` and `y` must be the same length.  Any
    other dimensions must be compatible for broadcasting.

    Parameters
    ----------
    x : (..., K) array_like
        Input array.
    y : (..., K) array_like
        Input array.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.

    Returns
    -------
    dist : ndarray
        Distance between the input arrays.

    Examples
    --------
    >>> from scipy.spatial import minkowski_distance
    >>> minkowski_distance([[0, 0], [0, 0]], [[1, 1], [0, 1]])
    array([ 1.41421356,  1.        ])

    """
    x = np.asarray(x)
    y = np.asarray(y)
    if p == np.inf or p == 1:
        return minkowski_distance_p(x, y, p)
    else:
        return minkowski_distance_p(x, y, p)**(1./p)


class Rectangle:
    """Hyperrectangle class.

    Represents a Cartesian product of intervals.
    """
    def __init__(self, maxes, mins):
        """Construct a hyperrectangle."""
        self.maxes = np.maximum(maxes,mins).astype(float)
        self.mins = np.minimum(maxes,mins).astype(float)
        self.m, = self.maxes.shape

    def __repr__(self):
        return f"<Rectangle {list(zip(self.mins, self.maxes))}>"

    def volume(self):
        """Total volume."""
        return np.prod(self.maxes-self.mins)

    def split(self, d, split):
        """Produce two hyperrectangles by splitting.

        In general, if you need to compute maximum and minimum
        distances to the children, it can be done more efficiently
        by updating the maximum and minimum distances to the parent.

        Parameters
        ----------
        d : int
            Axis to split hyperrectangle along.
        split : float
            Position along axis `d` to split at.

        """
        mid = np.copy(self.maxes)
        mid[d] = split
        less = Rectangle(self.mins, mid)
        mid = np.copy(self.mins)
        mid[d] = split
        greater = Rectangle(mid, self.maxes)
        return less, greater

    def min_distance_point(self, x, p=2.):
        """
        Return the minimum distance between input and points in the
        hyperrectangle.

        Parameters
        ----------
        x : array_like
            Input.
        p : float, optional
            Input.

        """
        return minkowski_distance(
            0, np.maximum(0, np.maximum(self.mins-x, x-self.maxes)),
            p
        )

    def max_distance_point(self, x, p=2.):
        """
        Return the maximum distance between input and points in the hyperrectangle.

        Parameters
        ----------
        x : array_like
            Input array.
        p : float, optional
            Input.

        """
        return minkowski_distance(0, np.maximum(self.maxes-x, x-self.mins), p)

    def min_distance_rectangle(self, other, p=2.):
        """
        Compute the minimum distance between points in the two hyperrectangles.

        Parameters
        ----------
        other : hyperrectangle
            Input.
        p : float
            Input.

        """
        return minkowski_distance(
            0,
            np.maximum(0, np.maximum(self.mins-other.maxes,
                                     other.mins-self.maxes)),
            p
        )

    def max_distance_rectangle(self, other, p=2.):
        """
        Compute the maximum distance between points in the two hyperrectangles.

        Parameters
        ----------
        other : hyperrectangle
            Input.
        p : float, optional
            Input.

        """
        return minkowski_distance(
            0, np.maximum(self.maxes-other.mins, other.maxes-self.mins), p)


class KDTree(cKDTree):
    """kd-tree for quick nearest-neighbor lookup.

    This class provides an index into a set of k-dimensional points
    which can be used to rapidly look up the nearest neighbors of any
    point.

    Parameters
    ----------
    data : array_like, shape (n,m)
        The n data points of dimension m to be indexed. This array is
        not copied unless this is necessary to produce a contiguous
        array of doubles, and so modifying this data will result in
        bogus results. The data are also copied if the kd-tree is built
        with copy_data=True.
    leafsize : positive int, optional
        The number of points at which the algorithm switches over to
        brute-force.  Default: 10.
    compact_nodes : bool, optional
        If True, the kd-tree is built to shrink the hyperrectangles to
        the actual data range. This usually gives a more compact tree that
        is robust against degenerated input data and gives faster queries
        at the expense of longer build time. Default: True.
    copy_data : bool, optional
        If True the data is always copied to protect the kd-tree against
        data corruption. Default: False.
    balanced_tree : bool, optional
        If True, the median is used to split the hyperrectangles instead of
        the midpoint. This usually gives a more compact tree and
        faster queries at the expense of longer build time. Default: True.
    boxsize : array_like or scalar, optional
        Apply a m-d toroidal topology to the KDTree.. The topology is generated
        by :math:`x_i + n_i L_i` where :math:`n_i` are integers and :math:`L_i`
        is the boxsize along i-th dimension. The input data shall be wrapped
        into :math:`[0, L_i)`. A ValueError is raised if any of the data is
        outside of this bound.

    Notes
    -----
    The algorithm used is described in Maneewongvatana and Mount 1999.
    The general idea is that the kd-tree is a binary tree, each of whose
    nodes represents an axis-aligned hyperrectangle. Each node specifies
    an axis and splits the set of points based on whether their coordinate
    along that axis is greater than or less than a particular value.

    During construction, the axis and splitting point are chosen by the
    "sliding midpoint" rule, which ensures that the cells do not all
    become long and thin.

    The tree can be queried for the r closest neighbors of any given point
    (optionally returning only those within some maximum distance of the
    point). It can also be queried, with a substantial gain in efficiency,
    for the r approximate closest neighbors.

    For large dimensions (20 is already large) do not expect this to run
    significantly faster than brute force. High-dimensional nearest-neighbor
    queries are a substantial open problem in computer science.

    Attributes
    ----------
    data : ndarray, shape (n,m)
        The n data points of dimension m to be indexed. This array is
        not copied unless this is necessary to produce a contiguous
        array of doubles. The data are also copied if the kd-tree is built
        with ``copy_data=True``.
    leafsize : positive int
        The number of points at which the algorithm switches over to
        brute-force.
    m : int
        The dimension of a single data-point.
    n : int
        The number of data points.
    maxes : ndarray, shape (m,)
        The maximum value in each dimension of the n data points.
    mins : ndarray, shape (m,)
        The minimum value in each dimension of the n data points.
    size : int
        The number of nodes in the tree.

    """

    class node:
        @staticmethod
        def _create(ckdtree_node=None):
            """Create either an inner or leaf node, wrapping a cKDTreeNode instance"""
            if ckdtree_node is None:
                return KDTree.node(ckdtree_node)
            elif ckdtree_node.split_dim == -1:
                return KDTree.leafnode(ckdtree_node)
            else:
                return KDTree.innernode(ckdtree_node)

        def __init__(self, ckdtree_node=None):
            if ckdtree_node is None:
                ckdtree_node = cKDTreeNode()
            self._node = ckdtree_node

        def __lt__(self, other):
            return id(self) < id(other)

        def __gt__(self, other):
            return id(self) > id(other)

        def __le__(self, other):
            return id(self) <= id(other)

        def __ge__(self, other):
            return id(self) >= id(other)

        def __eq__(self, other):
            return id(self) == id(other)

    class leafnode(node):
        @property
        def idx(self):
            return self._node.indices

        @property
        def children(self):
            return self._node.children

    class innernode(node):
        def __init__(self, ckdtreenode):
            assert isinstance(ckdtreenode, cKDTreeNode)
            super().__init__(ckdtreenode)
            self.less = KDTree.node._create(ckdtreenode.lesser)
            self.greater = KDTree.node._create(ckdtreenode.greater)

        @property
        def split_dim(self):
            return self._node.split_dim

        @property
        def split(self):
            return self._node.split

        @property
        def children(self):
            return self._node.children

    @property
    def tree(self):
        if not hasattr(self, "_tree"):
            self._tree = KDTree.node._create(super().tree)

        return self._tree

    def __init__(self, data, leafsize=10, compact_nodes=True, copy_data=False,
                 balanced_tree=True, boxsize=None):
        data = np.asarray(data)
        if data.dtype.kind == 'c':
            raise TypeError("KDTree does not work with complex data")

        # Note KDTree has different default leafsize from cKDTree
        super().__init__(data, leafsize, compact_nodes, copy_data,
                         balanced_tree, boxsize)

    def query(
            self, x, k=1, eps=0, p=2, distance_upper_bound=np.inf, workers=1):
        r"""Query the kd-tree for nearest neighbors.

        Parameters
        ----------
        x : array_like, last dimension self.m
            An array of points to query.
        k : int or Sequence[int], optional
            Either the number of nearest neighbors to return, or a list of the
            k-th nearest neighbors to return, starting from 1.
        eps : nonnegative float, optional
            Return approximate nearest neighbors; the kth returned value
            is guaranteed to be no further than (1+eps) times the
            distance to the real kth nearest neighbor.
        p : float, 1<=p<=infinity, optional
            Which Minkowski p-norm to use.
            1 is the sum-of-absolute-values distance ("Manhattan" distance).
            2 is the usual Euclidean distance.
            infinity is the maximum-coordinate-difference distance.
            A large, finite p may cause a ValueError if overflow can occur.
        distance_upper_bound : nonnegative float, optional
            Return only neighbors within this distance. This is used to prune
            tree searches, so if you are doing a series of nearest-neighbor
            queries, it may help to supply the distance to the nearest neighbor
            of the most recent point.
        workers : int, optional
            Number of workers to use for parallel processing. If -1 is given
            all CPU threads are used. Default: 1.

            .. versionadded:: 1.6.0

        Returns
        -------
        d : float or array of floats
            The distances to the nearest neighbors.
            If ``x`` has shape ``tuple+(self.m,)``, then ``d`` has shape
            ``tuple+(k,)``.
            When k == 1, the last dimension of the output is squeezed.
            Missing neighbors are indicated with infinite distances.
            Hits are sorted by distance (nearest first).

            .. versionchanged:: 1.9.0
               Previously if ``k=None``, then `d` was an object array of
               shape ``tuple``, containing lists of distances. This behavior
               has been removed, use `query_ball_point` instead.

        i : integer or array of integers
            The index of each neighbor in ``self.data``.
            ``i`` is the same shape as d.
            Missing neighbors are indicated with ``self.n``.

        Examples
        --------

        >>> import numpy as np
        >>> from scipy.spatial import KDTree
        >>> x, y = np.mgrid[0:5, 2:8]
        >>> tree = KDTree(np.c_[x.ravel(), y.ravel()])

        To query the nearest neighbours and return squeezed result, use

        >>> dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=1)
        >>> print(dd, ii, sep='\n')
        [2.         0.2236068]
        [ 0 13]

        To query the nearest neighbours and return unsqueezed result, use

        >>> dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=[1])
        >>> print(dd, ii, sep='\n')
        [[2.        ]
         [0.2236068]]
        [[ 0]
         [13]]

        To query the second nearest neighbours and return unsqueezed result,
        use

        >>> dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=[2])
        >>> print(dd, ii, sep='\n')
        [[2.23606798]
         [0.80622577]]
        [[ 6]
         [19]]

        To query the first and second nearest neighbours, use

        >>> dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=2)
        >>> print(dd, ii, sep='\n')
        [[2.         2.23606798]
         [0.2236068  0.80622577]]
        [[ 0  6]
         [13 19]]

        or, be more specific

        >>> dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=[1, 2])
        >>> print(dd, ii, sep='\n')
        [[2.         2.23606798]
         [0.2236068  0.80622577]]
        [[ 0  6]
         [13 19]]

        """
        x = np.asarray(x)
        if x.dtype.kind == 'c':
            raise TypeError("KDTree does not work with complex data")

        if k is None:
            raise ValueError("k must be an integer or a sequence of integers")

        d, i = super().query(x, k, eps, p, distance_upper_bound, workers)
        if isinstance(i, int):
            i = np.intp(i)
        return d, i

    def query_ball_point(self, x, r, p=2., eps=0, workers=1,
                         return_sorted=None, return_length=False):
        """Find all points within distance r of point(s) x.

        Parameters
        ----------
        x : array_like, shape tuple + (self.m,)
            The point or points to search for neighbors of.
        r : array_like, float
            The radius of points to return, must broadcast to the length of x.
        p : float, optional
            Which Minkowski p-norm to use.  Should be in the range [1, inf].
            A finite large p may cause a ValueError if overflow can occur.
        eps : nonnegative float, optional
            Approximate search. Branches of the tree are not explored if their
            nearest points are further than ``r / (1 + eps)``, and branches are
            added in bulk if their furthest points are nearer than
            ``r * (1 + eps)``.
        workers : int, optional
            Number of jobs to schedule for parallel processing. If -1 is given
            all processors are used. Default: 1.

            .. versionadded:: 1.6.0
        return_sorted : bool, optional
            Sorts returned indices if True and does not sort them if False. If
            None, does not sort single point queries, but does sort
            multi-point queries which was the behavior before this option
            was added.

            .. versionadded:: 1.6.0
        return_length : bool, optional
            Return the number of points inside the radius instead of a list
            of the indices.

            .. versionadded:: 1.6.0

        Returns
        -------
        results : list or array of lists
            If `x` is a single point, returns a list of the indices of the
            neighbors of `x`. If `x` is an array of points, returns an object
            array of shape tuple containing lists of neighbors.

        Notes
        -----
        If you have many points whose neighbors you want to find, you may save
        substantial amounts of time by putting them in a KDTree and using
        query_ball_tree.

        Examples
        --------
        >>> import numpy as np
        >>> from scipy import spatial
        >>> x, y = np.mgrid[0:5, 0:5]
        >>> points = np.c_[x.ravel(), y.ravel()]
        >>> tree = spatial.KDTree(points)
        >>> sorted(tree.query_ball_point([2, 0], 1))
        [5, 10, 11, 15]

        Query multiple points and plot the results:

        >>> import matplotlib.pyplot as plt
        >>> points = np.asarray(points)
        >>> plt.plot(points[:,0], points[:,1], '.')
        >>> for results in tree.query_ball_point(([2, 0], [3, 3]), 1):
        ...     nearby_points = points[results]
        ...     plt.plot(nearby_points[:,0], nearby_points[:,1], 'o')
        >>> plt.margins(0.1, 0.1)
        >>> plt.show()

        """
        x = np.asarray(x)
        if x.dtype.kind == 'c':
            raise TypeError("KDTree does not work with complex data")
        return super().query_ball_point(
            x, r, p, eps, workers, return_sorted, return_length)

    def query_ball_tree(self, other, r, p=2., eps=0):
        """
        Find all pairs of points between `self` and `other` whose distance is
        at most r.

        Parameters
        ----------
        other : KDTree instance
            The tree containing points to search against.
        r : float
            The maximum distance, has to be positive.
        p : float, optional
            Which Minkowski norm to use.  `p` has to meet the condition
            ``1 <= p <= infinity``.
        eps : float, optional
            Approximate search.  Branches of the tree are not explored
            if their nearest points are further than ``r/(1+eps)``, and
            branches are added in bulk if their furthest points are nearer
            than ``r * (1+eps)``.  `eps` has to be non-negative.

        Returns
        -------
        results : list of lists
            For each element ``self.data[i]`` of this tree, ``results[i]`` is a
            list of the indices of its neighbors in ``other.data``.

        Examples
        --------
        You can search all pairs of points between two kd-trees within a distance:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from scipy.spatial import KDTree
        >>> rng = np.random.default_rng()
        >>> points1 = rng.random((15, 2))
        >>> points2 = rng.random((15, 2))
        >>> plt.figure(figsize=(6, 6))
        >>> plt.plot(points1[:, 0], points1[:, 1], "xk", markersize=14)
        >>> plt.plot(points2[:, 0], points2[:, 1], "og", markersize=14)
        >>> kd_tree1 = KDTree(points1)
        >>> kd_tree2 = KDTree(points2)
        >>> indexes = kd_tree1.query_ball_tree(kd_tree2, r=0.2)
        >>> for i in range(len(indexes)):
        ...     for j in indexes[i]:
        ...         plt.plot([points1[i, 0], points2[j, 0]],
        ...             [points1[i, 1], points2[j, 1]], "-r")
        >>> plt.show()

        """
        return super().query_ball_tree(other, r, p, eps)

    def query_pairs(self, r, p=2., eps=0, output_type='set'):
        """Find all pairs of points in `self` whose distance is at most r.

        Parameters
        ----------
        r : positive float
            The maximum distance.
        p : float, optional
            Which Minkowski norm to use.  `p` has to meet the condition
            ``1 <= p <= infinity``.
        eps : float, optional
            Approximate search.  Branches of the tree are not explored
            if their nearest points are further than ``r/(1+eps)``, and
            branches are added in bulk if their furthest points are nearer
            than ``r * (1+eps)``.  `eps` has to be non-negative.
        output_type : string, optional
            Choose the output container, 'set' or 'ndarray'. Default: 'set'

            .. versionadded:: 1.6.0

        Returns
        -------
        results : set or ndarray
            Set of pairs ``(i,j)``, with ``i < j``, for which the corresponding
            positions are close. If output_type is 'ndarray', an ndarry is
            returned instead of a set.

        Examples
        --------
        You can search all pairs of points in a kd-tree within a distance:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from scipy.spatial import KDTree
        >>> rng = np.random.default_rng()
        >>> points = rng.random((20, 2))
        >>> plt.figure(figsize=(6, 6))
        >>> plt.plot(points[:, 0], points[:, 1], "xk", markersize=14)
        >>> kd_tree = KDTree(points)
        >>> pairs = kd_tree.query_pairs(r=0.2)
        >>> for (i, j) in pairs:
        ...     plt.plot([points[i, 0], points[j, 0]],
        ...             [points[i, 1], points[j, 1]], "-r")
        >>> plt.show()

        """
        return super().query_pairs(r, p, eps, output_type)

    def count_neighbors(self, other, r, p=2., weights=None, cumulative=True):
        """Count how many nearby pairs can be formed.

        Count the number of pairs ``(x1,x2)`` can be formed, with ``x1`` drawn
        from ``self`` and ``x2`` drawn from ``other``, and where
        ``distance(x1, x2, p) <= r``.

        Data points on ``self`` and ``other`` are optionally weighted by the
        ``weights`` argument. (See below)

        This is adapted from the "two-point correlation" algorithm described by
        Gray and Moore [1]_.  See notes for further discussion.

        Parameters
        ----------
        other : KDTree
            The other tree to draw points from, can be the same tree as self.
        r : float or one-dimensional array of floats
            The radius to produce a count for. Multiple radii are searched with
            a single tree traversal.
            If the count is non-cumulative(``cumulative=False``), ``r`` defines
            the edges of the bins, and must be non-decreasing.
        p : float, optional
            1<=p<=infinity.
            Which Minkowski p-norm to use.
            Default 2.0.
            A finite large p may cause a ValueError if overflow can occur.
        weights : tuple, array_like, or None, optional
            If None, the pair-counting is unweighted.
            If given as a tuple, weights[0] is the weights of points in
            ``self``, and weights[1] is the weights of points in ``other``;
            either can be None to indicate the points are unweighted.
            If given as an array_like, weights is the weights of points in
            ``self`` and ``other``. For this to make sense, ``self`` and
            ``other`` must be the same tree. If ``self`` and ``other`` are two
            different trees, a ``ValueError`` is raised.
            Default: None

            .. versionadded:: 1.6.0
        cumulative : bool, optional
            Whether the returned counts are cumulative. When cumulative is set
            to ``False`` the algorithm is optimized to work with a large number
            of bins (>10) specified by ``r``. When ``cumulative`` is set to
            True, the algorithm is optimized to work with a small number of
            ``r``. Default: True

            .. versionadded:: 1.6.0

        Returns
        -------
        result : scalar or 1-D array
            The number of pairs. For unweighted counts, the result is integer.
            For weighted counts, the result is float.
            If cumulative is False, ``result[i]`` contains the counts with
            ``(-inf if i == 0 else r[i-1]) < R <= r[i]``

        Notes
        -----
        Pair-counting is the basic operation used to calculate the two point
        correlation functions from a data set composed of position of objects.

        Two point correlation function measures the clustering of objects and
        is widely used in cosmology to quantify the large scale structure
        in our Universe, but it may be useful for data analysis in other fields
        where self-similar assembly of objects also occur.

        The Landy-Szalay estimator for the two point correlation function of
        ``D`` measures the clustering signal in ``D``. [2]_

        For example, given the position of two sets of objects,

        - objects ``D`` (data) contains the clustering signal, and

        - objects ``R`` (random) that contains no signal,

        .. math::

             \\xi(r) = \\frac{<D, D> - 2 f <D, R> + f^2<R, R>}{f^2<R, R>},

        where the brackets represents counting pairs between two data sets
        in a finite bin around ``r`` (distance), corresponding to setting
        `cumulative=False`, and ``f = float(len(D)) / float(len(R))`` is the
        ratio between number of objects from data and random.

        The algorithm implemented here is loosely based on the dual-tree
        algorithm described in [1]_. We switch between two different
        pair-cumulation scheme depending on the setting of ``cumulative``.
        The computing time of the method we use when for
        ``cumulative == False`` does not scale with the total number of bins.
        The algorithm for ``cumulative == True`` scales linearly with the
        number of bins, though it is slightly faster when only
        1 or 2 bins are used. [5]_.

        As an extension to the naive pair-counting,
        weighted pair-counting counts the product of weights instead
        of number of pairs.
        Weighted pair-counting is used to estimate marked correlation functions
        ([3]_, section 2.2),
        or to properly calculate the average of data per distance bin
        (e.g. [4]_, section 2.1 on redshift).

        .. [1] Gray and Moore,
               "N-body problems in statistical learning",
               Mining the sky, 2000,
               https://arxiv.org/abs/astro-ph/0012333

        .. [2] Landy and Szalay,
               "Bias and variance of angular correlation functions",
               The Astrophysical Journal, 1993,
               http://adsabs.harvard.edu/abs/1993ApJ...412...64L

        .. [3] Sheth, Connolly and Skibba,
               "Marked correlations in galaxy formation models",
               Arxiv e-print, 2005,
               https://arxiv.org/abs/astro-ph/0511773

        .. [4] Hawkins, et al.,
               "The 2dF Galaxy Redshift Survey: correlation functions,
               peculiar velocities and the matter density of the Universe",
               Monthly Notices of the Royal Astronomical Society, 2002,
               http://adsabs.harvard.edu/abs/2003MNRAS.346...78H

        .. [5] https://github.com/scipy/scipy/pull/5647#issuecomment-168474926

        Examples
        --------
        You can count neighbors number between two kd-trees within a distance:

        >>> import numpy as np
        >>> from scipy.spatial import KDTree
        >>> rng = np.random.default_rng()
        >>> points1 = rng.random((5, 2))
        >>> points2 = rng.random((5, 2))
        >>> kd_tree1 = KDTree(points1)
        >>> kd_tree2 = KDTree(points2)
        >>> kd_tree1.count_neighbors(kd_tree2, 0.2)
        1

        This number is same as the total pair number calculated by
        `query_ball_tree`:

        >>> indexes = kd_tree1.query_ball_tree(kd_tree2, r=0.2)
        >>> sum([len(i) for i in indexes])
        1

        """
        return super().count_neighbors(other, r, p, weights, cumulative)

    def sparse_distance_matrix(
            self, other, max_distance, p=2., output_type='dok_matrix'):
        """Compute a sparse distance matrix.

        Computes a distance matrix between two KDTrees, leaving as zero
        any distance greater than max_distance.

        Parameters
        ----------
        other : KDTree

        max_distance : positive float

        p : float, 1<=p<=infinity
            Which Minkowski p-norm to use.
            A finite large p may cause a ValueError if overflow can occur.

        output_type : string, optional
            Which container to use for output data. Options: 'dok_matrix',
            'coo_matrix', 'dict', or 'ndarray'. Default: 'dok_matrix'.

            .. versionadded:: 1.6.0

        Returns
        -------
        result : dok_matrix, coo_matrix, dict or ndarray
            Sparse matrix representing the results in "dictionary of keys"
            format. If a dict is returned the keys are (i,j) tuples of indices.
            If output_type is 'ndarray' a record array with fields 'i', 'j',
            and 'v' is returned,

        Examples
        --------
        You can compute a sparse distance matrix between two kd-trees:

        >>> import numpy as np
        >>> from scipy.spatial import KDTree
        >>> rng = np.random.default_rng()
        >>> points1 = rng.random((5, 2))
        >>> points2 = rng.random((5, 2))
        >>> kd_tree1 = KDTree(points1)
        >>> kd_tree2 = KDTree(points2)
        >>> sdm = kd_tree1.sparse_distance_matrix(kd_tree2, 0.3)
        >>> sdm.toarray()
        array([[0.        , 0.        , 0.12295571, 0.        , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.28942611, 0.        , 0.        , 0.2333084 , 0.        ],
           [0.        , 0.        , 0.        , 0.        , 0.        ],
           [0.24617575, 0.29571802, 0.26836782, 0.        , 0.        ]])

        You can check distances above the `max_distance` are zeros:

        >>> from scipy.spatial import distance_matrix
        >>> distance_matrix(points1, points2)
        array([[0.56906522, 0.39923701, 0.12295571, 0.8658745 , 0.79428925],
           [0.37327919, 0.7225693 , 0.87665969, 0.32580855, 0.75679479],
           [0.28942611, 0.30088013, 0.6395831 , 0.2333084 , 0.33630734],
           [0.31994999, 0.72658602, 0.71124834, 0.55396483, 0.90785663],
           [0.24617575, 0.29571802, 0.26836782, 0.57714465, 0.6473269 ]])

        """
        return super().sparse_distance_matrix(
            other, max_distance, p, output_type)


def distance_matrix(x, y, p=2, threshold=1000000):
    """Compute the distance matrix.

    Returns the matrix of all pair-wise distances.

    Parameters
    ----------
    x : (M, K) array_like
        Matrix of M vectors in K dimensions.
    y : (N, K) array_like
        Matrix of N vectors in K dimensions.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.
    threshold : positive int
        If ``M * N * K`` > `threshold`, algorithm uses a Python loop instead
        of large temporary arrays.

    Returns
    -------
    result : (M, N) ndarray
        Matrix containing the distance from every vector in `x` to every vector
        in `y`.

    Examples
    --------
    >>> from scipy.spatial import distance_matrix
    >>> distance_matrix([[0,0],[0,1]], [[1,0],[1,1]])
    array([[ 1.        ,  1.41421356],
           [ 1.41421356,  1.        ]])

    """

    x = np.asarray(x)
    m, k = x.shape
    y = np.asarray(y)
    n, kk = y.shape

    if k != kk:
        raise ValueError(f"x contains {k}-dimensional vectors but y contains "
                         f"{kk}-dimensional vectors")

    if m*n*k <= threshold:
        return minkowski_distance(x[:,np.newaxis,:],y[np.newaxis,:,:],p)
    else:
        result = np.empty((m,n),dtype=float)  # FIXME: figure out the best dtype
        if m < n:
            for i in range(m):
                result[i,:] = minkowski_distance(x[i],y,p)
        else:
            for j in range(n):
                result[:,j] = minkowski_distance(x,y[j],p)
        return result
<|endoftext|>import numpy as np
from scipy.constants import golden as phi


def icosahedral(cls):
    g1 = tetrahedral(cls).as_quat()
    a = 0.5
    b = 0.5 / phi
    c = phi / 2
    g2 = np.array([[+a, +b, +c, 0],
                   [+a, +b, -c, 0],
                   [+a, +c, 0, +b],
                   [+a, +c, 0, -b],
                   [+a, -b, +c, 0],
                   [+a, -b, -c, 0],
                   [+a, -c, 0, +b],
                   [+a, -c, 0, -b],
                   [+a, 0, +b, +c],
                   [+a, 0, +b, -c],
                   [+a, 0, -b, +c],
                   [+a, 0, -b, -c],
                   [+b, +a, 0, +c],
                   [+b, +a, 0, -c],
                   [+b, +c, +a, 0],
                   [+b, +c, -a, 0],
                   [+b, -a, 0, +c],
                   [+b, -a, 0, -c],
                   [+b, -c, +a, 0],
                   [+b, -c, -a, 0],
                   [+b, 0, +c, +a],
                   [+b, 0, +c, -a],
                   [+b, 0, -c, +a],
                   [+b, 0, -c, -a],
                   [+c, +a, +b, 0],
                   [+c, +a, -b, 0],
                   [+c, +b, 0, +a],
                   [+c, +b, 0, -a],
                   [+c, -a, +b, 0],
                   [+c, -a, -b, 0],
                   [+c, -b, 0, +a],
                   [+c, -b, 0, -a],
                   [+c, 0, +a, +b],
                   [+c, 0, +a, -b],
                   [+c, 0, -a, +b],
                   [+c, 0, -a, -b],
                   [0, +a, +c, +b],
                   [0, +a, +c, -b],
                   [0, +a, -c, +b],
                   [0, +a, -c, -b],
                   [0, +b, +a, +c],
                   [0, +b, +a, -c],
                   [0, +b, -a, +c],
                   [0, +b, -a, -c],
                   [0, +c, +b, +a],
                   [0, +c, +b, -a],
                   [0, +c, -b, +a],
                   [0, +c, -b, -a]])
    return cls.from_quat(np.concatenate((g1, g2)))


def octahedral(cls):
    g1 = tetrahedral(cls).as_quat()
    c = np.sqrt(2) / 2
    g2 = np.array([[+c, 0, 0, +c],
                   [0, +c, 0, +c],
                   [0, 0, +c, +c],
                   [0, 0, -c, +c],
                   [0, -c, 0, +c],
                   [-c, 0, 0, +c],
                   [0, +c, +c, 0],
                   [0, -c, +c, 0],
                   [+c, 0, +c, 0],
                   [-c, 0, +c, 0],
                   [+c, +c, 0, 0],
                   [-c, +c, 0, 0]])
    return cls.from_quat(np.concatenate((g1, g2)))


def tetrahedral(cls):
    g1 = np.eye(4)
    c = 0.5
    g2 = np.array([[c, -c, -c, +c],
                   [c, -c, +c, +c],
                   [c, +c, -c, +c],
                   [c, +c, +c, +c],
                   [c, -c, -c, -c],
                   [c, -c, +c, -c],
                   [c, +c, -c, -c],
                   [c, +c, +c, -c]])
    return cls.from_quat(np.concatenate((g1, g2)))


def dicyclic(cls, n, axis=2):
    g1 = cyclic(cls, n, axis).as_rotvec()

    thetas = np.linspace(0, np.pi, n, endpoint=False)
    rv = np.pi * np.vstack([np.zeros(n), np.cos(thetas), np.sin(thetas)]).T
    g2 = np.roll(rv, axis, axis=1)
    return cls.from_rotvec(np.concatenate((g1, g2)))


def cyclic(cls, n, axis=2):
    thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rv = np.vstack([thetas, np.zeros(n), np.zeros(n)]).T
    return cls.from_rotvec(np.roll(rv, axis, axis=1))


def create_group(cls, group, axis='Z'):
    if not isinstance(group, str):
        raise ValueError("`group` argument must be a string")

    permitted_axes = ['x', 'y', 'z', 'X', 'Y', 'Z']
    if axis not in permitted_axes:
        raise ValueError("`axis` must be one of " + ", ".join(permitted_axes))

    if group in ['I', 'O', 'T']:
        symbol = group
        order = 1
    elif group[:1] in ['C', 'D'] and group[1:].isdigit():
        symbol = group[:1]
        order = int(group[1:])
    else:
        raise ValueError("`group` must be one of 'I', 'O', 'T', 'Dn', 'Cn'")

    if order < 1:
        raise ValueError("Group order must be positive")

    axis = 'xyz'.index(axis.lower())
    if symbol == 'I':
        return icosahedral(cls)
    elif symbol == 'O':
        return octahedral(cls)
    elif symbol == 'T':
        return tetrahedral(cls)
    elif symbol == 'D':
        return dicyclic(cls, order, axis=axis)
    elif symbol == 'C':
        return cyclic(cls, order, axis=axis)
    else:
        assert False
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.spatial` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'Rotation',
    'Slerp',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="spatial.transform", module="rotation",
                                   private_modules=["_rotation"], all=__all__,
                                   attribute=name)
<|endoftext|>"""
Spatial Transformations (:mod:`scipy.spatial.transform`)
========================================================

.. currentmodule:: scipy.spatial.transform

This package implements various spatial transformations. For now,
only rotations are supported.

Rotations in 3 dimensions
-------------------------
.. autosummary::
   :toctree: generated/

   Rotation
   Slerp
   RotationSpline
"""
from ._rotation import Rotation, Slerp
from ._rotation_spline import RotationSpline

# Deprecated namespaces, to be removed in v2.0.0
from . import rotation

__all__ = ['Rotation', 'Slerp', 'RotationSpline']

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
<|endoftext|>import numpy as np
from scipy.linalg import solve_banded
from ._rotation import Rotation


def _create_skew_matrix(x):
    """Create skew-symmetric matrices corresponding to vectors.

    Parameters
    ----------
    x : ndarray, shape (n, 3)
        Set of vectors.

    Returns
    -------
    ndarray, shape (n, 3, 3)
    """
    result = np.zeros((len(x), 3, 3))
    result[:, 0, 1] = -x[:, 2]
    result[:, 0, 2] = x[:, 1]
    result[:, 1, 0] = x[:, 2]
    result[:, 1, 2] = -x[:, 0]
    result[:, 2, 0] = -x[:, 1]
    result[:, 2, 1] = x[:, 0]
    return result


def _matrix_vector_product_of_stacks(A, b):
    """Compute the product of stack of matrices and vectors."""
    return np.einsum("ijk,ik->ij", A, b)


def _angular_rate_to_rotvec_dot_matrix(rotvecs):
    """Compute matrices to transform angular rates to rot. vector derivatives.

    The matrices depend on the current attitude represented as a rotation
    vector.

    Parameters
    ----------
    rotvecs : ndarray, shape (n, 3)
        Set of rotation vectors.

    Returns
    -------
    ndarray, shape (n, 3, 3)
    """
    norm = np.linalg.norm(rotvecs, axis=1)
    k = np.empty_like(norm)

    mask = norm > 1e-4
    nm = norm[mask]
    k[mask] = (1 - 0.5 * nm / np.tan(0.5 * nm)) / nm**2
    mask = ~mask
    nm = norm[mask]
    k[mask] = 1/12 + 1/720 * nm**2

    skew = _create_skew_matrix(rotvecs)

    result = np.empty((len(rotvecs), 3, 3))
    result[:] = np.identity(3)
    result[:] += 0.5 * skew
    result[:] += k[:, None, None] * np.matmul(skew, skew)

    return result


def _rotvec_dot_to_angular_rate_matrix(rotvecs):
    """Compute matrices to transform rot. vector derivatives to angular rates.

    The matrices depend on the current attitude represented as a rotation
    vector.

    Parameters
    ----------
    rotvecs : ndarray, shape (n, 3)
        Set of rotation vectors.

    Returns
    -------
    ndarray, shape (n, 3, 3)
    """
    norm = np.linalg.norm(rotvecs, axis=1)
    k1 = np.empty_like(norm)
    k2 = np.empty_like(norm)

    mask = norm > 1e-4
    nm = norm[mask]
    k1[mask] = (1 - np.cos(nm)) / nm ** 2
    k2[mask] = (nm - np.sin(nm)) / nm ** 3

    mask = ~mask
    nm = norm[mask]
    k1[mask] = 0.5 - nm ** 2 / 24
    k2[mask] = 1 / 6 - nm ** 2 / 120

    skew = _create_skew_matrix(rotvecs)

    result = np.empty((len(rotvecs), 3, 3))
    result[:] = np.identity(3)
    result[:] -= k1[:, None, None] * skew
    result[:] += k2[:, None, None] * np.matmul(skew, skew)

    return result


def _angular_acceleration_nonlinear_term(rotvecs, rotvecs_dot):
    """Compute the non-linear term in angular acceleration.

    The angular acceleration contains a quadratic term with respect to
    the derivative of the rotation vector. This function computes that.

    Parameters
    ----------
    rotvecs : ndarray, shape (n, 3)
        Set of rotation vectors.
    rotvecs_dot : ndarray, shape (n, 3)
        Set of rotation vector derivatives.

    Returns
    -------
    ndarray, shape (n, 3)
    """
    norm = np.linalg.norm(rotvecs, axis=1)
    dp = np.sum(rotvecs * rotvecs_dot, axis=1)
    cp = np.cross(rotvecs, rotvecs_dot)
    ccp = np.cross(rotvecs, cp)
    dccp = np.cross(rotvecs_dot, cp)

    k1 = np.empty_like(norm)
    k2 = np.empty_like(norm)
    k3 = np.empty_like(norm)

    mask = norm > 1e-4
    nm = norm[mask]
    k1[mask] = (-nm * np.sin(nm) - 2 * (np.cos(nm) - 1)) / nm ** 4
    k2[mask] = (-2 * nm + 3 * np.sin(nm) - nm * np.cos(nm)) / nm ** 5
    k3[mask] = (nm - np.sin(nm)) / nm ** 3

    mask = ~mask
    nm = norm[mask]
    k1[mask] = 1/12 - nm ** 2 / 180
    k2[mask] = -1/60 + nm ** 2 / 12604
    k3[mask] = 1/6 - nm ** 2 / 120

    dp = dp[:, None]
    k1 = k1[:, None]
    k2 = k2[:, None]
    k3 = k3[:, None]

    return dp * (k1 * cp + k2 * ccp) + k3 * dccp


def _compute_angular_rate(rotvecs, rotvecs_dot):
    """Compute angular rates given rotation vectors and its derivatives.

    Parameters
    ----------
    rotvecs : ndarray, shape (n, 3)
        Set of rotation vectors.
    rotvecs_dot : ndarray, shape (n, 3)
        Set of rotation vector derivatives.

    Returns
    -------
    ndarray, shape (n, 3)
    """
    return _matrix_vector_product_of_stacks(
        _rotvec_dot_to_angular_rate_matrix(rotvecs), rotvecs_dot)


def _compute_angular_acceleration(rotvecs, rotvecs_dot, rotvecs_dot_dot):
    """Compute angular acceleration given rotation vector and its derivatives.

    Parameters
    ----------
    rotvecs : ndarray, shape (n, 3)
        Set of rotation vectors.
    rotvecs_dot : ndarray, shape (n, 3)
        Set of rotation vector derivatives.
    rotvecs_dot_dot : ndarray, shape (n, 3)
        Set of rotation vector second derivatives.

    Returns
    -------
    ndarray, shape (n, 3)
    """
    return (_compute_angular_rate(rotvecs, rotvecs_dot_dot) +
            _angular_acceleration_nonlinear_term(rotvecs, rotvecs_dot))


def _create_block_3_diagonal_matrix(A, B, d):
    """Create a 3-diagonal block matrix as banded.

    The matrix has the following structure:

        DB...
        ADB..
        .ADB.
        ..ADB
        ...AD

    The blocks A, B and D are 3-by-3 matrices. The D matrices has the form
    d * I.

    Parameters
    ----------
    A : ndarray, shape (n, 3, 3)
        Stack of A blocks.
    B : ndarray, shape (n, 3, 3)
        Stack of B blocks.
    d : ndarray, shape (n + 1,)
        Values for diagonal blocks.

    Returns
    -------
    ndarray, shape (11, 3 * (n + 1))
        Matrix in the banded form as used by `scipy.linalg.solve_banded`.
    """
    ind = np.arange(3)
    ind_blocks = np.arange(len(A))

    A_i = np.empty_like(A, dtype=int)
    A_i[:] = ind[:, None]
    A_i += 3 * (1 + ind_blocks[:, None, None])

    A_j = np.empty_like(A, dtype=int)
    A_j[:] = ind
    A_j += 3 * ind_blocks[:, None, None]

    B_i = np.empty_like(B, dtype=int)
    B_i[:] = ind[:, None]
    B_i += 3 * ind_blocks[:, None, None]

    B_j = np.empty_like(B, dtype=int)
    B_j[:] = ind
    B_j += 3 * (1 + ind_blocks[:, None, None])

    diag_i = diag_j = np.arange(3 * len(d))
    i = np.hstack((A_i.ravel(), B_i.ravel(), diag_i))
    j = np.hstack((A_j.ravel(), B_j.ravel(), diag_j))
    values = np.hstack((A.ravel(), B.ravel(), np.repeat(d, 3)))

    u = 5
    l = 5
    result = np.zeros((u + l + 1, 3 * len(d)))
    result[u + i - j, j] = values
    return result


class RotationSpline:
    """Interpolate rotations with continuous angular rate and acceleration.

    The rotation vectors between each consecutive orientation are cubic
    functions of time and it is guaranteed that angular rate and acceleration
    are continuous. Such interpolation are analogous to cubic spline
    interpolation.

    Refer to [1]_ for math and implementation details.

    Parameters
    ----------
    times : array_like, shape (N,)
        Times of the known rotations. At least 2 times must be specified.
    rotations : `Rotation` instance
        Rotations to perform the interpolation between. Must contain N
        rotations.

    Methods
    -------
    __call__

    References
    ----------
    .. [1] `Smooth Attitude Interpolation
            <https://github.com/scipy/scipy/files/2932755/attitude_interpolation.pdf>`_

    Examples
    --------
    >>> from scipy.spatial.transform import Rotation, RotationSpline
    >>> import numpy as np

    Define the sequence of times and rotations from the Euler angles:

    >>> times = [0, 10, 20, 40]
    >>> angles = [[-10, 20, 30], [0, 15, 40], [-30, 45, 30], [20, 45, 90]]
    >>> rotations = Rotation.from_euler('XYZ', angles, degrees=True)

    Create the interpolator object:

    >>> spline = RotationSpline(times, rotations)

    Interpolate the Euler angles, angular rate and acceleration:

    >>> angular_rate = np.rad2deg(spline(times, 1))
    >>> angular_acceleration = np.rad2deg(spline(times, 2))
    >>> times_plot = np.linspace(times[0], times[-1], 100)
    >>> angles_plot = spline(times_plot).as_euler('XYZ', degrees=True)
    >>> angular_rate_plot = np.rad2deg(spline(times_plot, 1))
    >>> angular_acceleration_plot = np.rad2deg(spline(times_plot, 2))

    On this plot you see that Euler angles are continuous and smooth:

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(times_plot, angles_plot)
    >>> plt.plot(times, angles, 'x')
    >>> plt.title("Euler angles")
    >>> plt.show()

    The angular rate is also smooth:

    >>> plt.plot(times_plot, angular_rate_plot)
    >>> plt.plot(times, angular_rate, 'x')
    >>> plt.title("Angular rate")
    >>> plt.show()

    The angular acceleration is continuous, but not smooth. Also note that
    the angular acceleration is not a piecewise-linear function, because
    it is different from the second derivative of the rotation vector (which
    is a piecewise-linear function as in the cubic spline).

    >>> plt.plot(times_plot, angular_acceleration_plot)
    >>> plt.plot(times, angular_acceleration, 'x')
    >>> plt.title("Angular acceleration")
    >>> plt.show()
    """
    # Parameters for the solver for angular rate.
    MAX_ITER = 10
    TOL = 1e-9

    def _solve_for_angular_rates(self, dt, angular_rates, rotvecs):
        angular_rate_first = angular_rates[0].copy()

        A = _angular_rate_to_rotvec_dot_matrix(rotvecs)
        A_inv = _rotvec_dot_to_angular_rate_matrix(rotvecs)
        M = _create_block_3_diagonal_matrix(
            2 * A_inv[1:-1] / dt[1:-1, None, None],
            2 * A[1:-1] / dt[1:-1, None, None],
            4 * (1 / dt[:-1] + 1 / dt[1:]))

        b0 = 6 * (rotvecs[:-1] * dt[:-1, None] ** -2 +
                  rotvecs[1:] * dt[1:, None] ** -2)
        b0[0] -= 2 / dt[0] * A_inv[0].dot(angular_rate_first)
        b0[-1] -= 2 / dt[-1] * A[-1].dot(angular_rates[-1])

        for iteration in range(self.MAX_ITER):
            rotvecs_dot = _matrix_vector_product_of_stacks(A, angular_rates)
            delta_beta = _angular_acceleration_nonlinear_term(
                rotvecs[:-1], rotvecs_dot[:-1])
            b = b0 - delta_beta
            angular_rates_new = solve_banded((5, 5), M, b.ravel())
            angular_rates_new = angular_rates_new.reshape((-1, 3))

            delta = np.abs(angular_rates_new - angular_rates[:-1])
            angular_rates[:-1] = angular_rates_new
            if np.all(delta < self.TOL * (1 + np.abs(angular_rates_new))):
                break

        rotvecs_dot = _matrix_vector_product_of_stacks(A, angular_rates)
        angular_rates = np.vstack((angular_rate_first, angular_rates[:-1]))

        return angular_rates, rotvecs_dot

    def __init__(self, times, rotations):
        from scipy.interpolate import PPoly

        if rotations.single:
            raise ValueError("`rotations` must be a sequence of rotations.")

        if len(rotations) == 1:
            raise ValueError("`rotations` must contain at least 2 rotations.")

        times = np.asarray(times, dtype=float)
        if times.ndim != 1:
            raise ValueError("`times` must be 1-dimensional.")

        if len(times) != len(rotations):
            raise ValueError("Expected number of rotations to be equal to "
                             "number of timestamps given, "
                             f"got {len(rotations)} rotations "
                             f"and {len(times)} timestamps.")

        dt = np.diff(times)
        if np.any(dt <= 0):
            raise ValueError("Values in `times` must be in a strictly "
                             "increasing order.")

        rotvecs = (rotations[:-1].inv() * rotations[1:]).as_rotvec()
        angular_rates = rotvecs / dt[:, None]

        if len(rotations) == 2:
            rotvecs_dot = angular_rates
        else:
            angular_rates, rotvecs_dot = self._solve_for_angular_rates(
                dt, angular_rates, rotvecs)

        dt = dt[:, None]
        coeff = np.empty((4, len(times) - 1, 3))
        coeff[0] = (-2 * rotvecs + dt * angular_rates
                    + dt * rotvecs_dot) / dt ** 3
        coeff[1] = (3 * rotvecs - 2 * dt * angular_rates
                    - dt * rotvecs_dot) / dt ** 2
        coeff[2] = angular_rates
        coeff[3] = 0

        self.times = times
        self.rotations = rotations
        self.interpolator = PPoly(coeff, times)

    def __call__(self, times, order=0):
        """Compute interpolated values.

        Parameters
        ----------
        times : float or array_like
            Times of interest.
        order : {0, 1, 2}, optional
            Order of differentiation:

                * 0 (default) : return Rotation
                * 1 : return the angular rate in rad/sec
                * 2 : return the angular acceleration in rad/sec/sec

        Returns
        -------
        Interpolated Rotation, angular rate or acceleration.
        """
        if order not in [0, 1, 2]:
            raise ValueError("`order` must be 0, 1 or 2.")

        times = np.asarray(times, dtype=float)
        if times.ndim > 1:
            raise ValueError("`times` must be at most 1-dimensional.")

        singe_time = times.ndim == 0
        times = np.atleast_1d(times)

        rotvecs = self.interpolator(times)
        if order == 0:
            index = np.searchsorted(self.times, times, side='right')
            index -= 1
            index[index < 0] = 0
            n_segments = len(self.times) - 1
            index[index > n_segments - 1] = n_segments - 1
            result = self.rotations[index] * Rotation.from_rotvec(rotvecs)
        elif order == 1:
            rotvecs_dot = self.interpolator(times, 1)
            result = _compute_angular_rate(rotvecs, rotvecs_dot)
        elif order == 2:
            rotvecs_dot = self.interpolator(times, 1)
            rotvecs_dot_dot = self.interpolator(times, 2)
            result = _compute_angular_acceleration(rotvecs, rotvecs_dot,
                                                   rotvecs_dot_dot)
        else:
            assert False

        if singe_time:
            result = result[0]

        return result
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.constants` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'Avogadro', 'Boltzmann', 'Btu', 'Btu_IT', 'Btu_th', 'G',
    'Julian_year', 'N_A', 'Planck', 'R', 'Rydberg',
    'Stefan_Boltzmann', 'Wien', 'acre', 'alpha',
    'angstrom', 'arcmin', 'arcminute', 'arcsec',
    'arcsecond', 'astronomical_unit', 'atm',
    'atmosphere', 'atomic_mass', 'atto', 'au', 'bar',
    'barrel', 'bbl', 'blob', 'c', 'calorie',
    'calorie_IT', 'calorie_th', 'carat', 'centi',
    'convert_temperature', 'day', 'deci', 'degree',
    'degree_Fahrenheit', 'deka', 'dyn', 'dyne', 'e',
    'eV', 'electron_mass', 'electron_volt',
    'elementary_charge', 'epsilon_0', 'erg',
    'exa', 'exbi', 'femto', 'fermi', 'fine_structure',
    'fluid_ounce', 'fluid_ounce_US', 'fluid_ounce_imp',
    'foot', 'g', 'gallon', 'gallon_US', 'gallon_imp',
    'gas_constant', 'gibi', 'giga', 'golden', 'golden_ratio',
    'grain', 'gram', 'gravitational_constant', 'h', 'hbar',
    'hectare', 'hecto', 'horsepower', 'hour', 'hp',
    'inch', 'k', 'kgf', 'kibi', 'kilo', 'kilogram_force',
    'kmh', 'knot', 'lambda2nu', 'lb', 'lbf',
    'light_year', 'liter', 'litre', 'long_ton', 'm_e',
    'm_n', 'm_p', 'm_u', 'mach', 'mebi', 'mega',
    'metric_ton', 'micro', 'micron', 'mil', 'mile',
    'milli', 'minute', 'mmHg', 'mph', 'mu_0', 'nano',
    'nautical_mile', 'neutron_mass', 'nu2lambda',
    'ounce', 'oz', 'parsec', 'pebi', 'peta',
    'pi', 'pico', 'point', 'pound', 'pound_force',
    'proton_mass', 'psi', 'pt', 'short_ton',
    'sigma', 'slinch', 'slug', 'speed_of_light',
    'speed_of_sound', 'stone', 'survey_foot',
    'survey_mile', 'tebi', 'tera', 'ton_TNT',
    'torr', 'troy_ounce', 'troy_pound', 'u',
    'week', 'yard', 'year', 'yobi', 'yocto',
    'yotta', 'zebi', 'zepto', 'zero_Celsius', 'zetta'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="constants", module="constants",
                                   private_modules=["_constants"], all=__all__,
                                   attribute=name)
<|endoftext|>r"""
==================================
Constants (:mod:`scipy.constants`)
==================================

.. currentmodule:: scipy.constants

Physical and mathematical constants and units.


Mathematical constants
======================

================  =================================================================
``pi``            Pi
``golden``        Golden ratio
``golden_ratio``  Golden ratio
================  =================================================================


Physical constants
==================
The following physical constants are available as attributes of `scipy.constants`.
All units are `SI <https://en.wikipedia.org/wiki/International_System_of_Units>`_.

===========================  ================================================================  ===============
Attribute                    Quantity                                                          Units
===========================  ================================================================  ===============
``c``                        speed of light in vacuum                                          m s^-1
``speed_of_light``           speed of light in vacuum                                          m s^-1
``mu_0``                     the magnetic constant :math:`\mu_0`                               N A^-2
``epsilon_0``                the electric constant (vacuum permittivity), :math:`\epsilon_0`   F m^-1
``h``                        the Planck constant :math:`h`                                     J Hz^-1
``Planck``                   the Planck constant :math:`h`                                     J Hz^-1
``hbar``                     the reduced Planck constant, :math:`\hbar = h/(2\pi)`             J s
``G``                        Newtonian constant of gravitation                                 m^3 kg^-1 s^-2
``gravitational_constant``   Newtonian constant of gravitation                                 m^3 kg^-1 s^-2
``g``                        standard acceleration of gravity                                  m s^-2
``e``                        elementary charge                                                 C
``elementary_charge``        elementary charge                                                 C
``R``                        molar gas constant                                                J mol^-1 K^-1
``gas_constant``             molar gas constant                                                J mol^-1 K^-1
``alpha``                    fine-structure constant                                           (unitless)
``fine_structure``           fine-structure constant                                           (unitless)
``N_A``                      Avogadro constant                                                 mol^-1
``Avogadro``                 Avogadro constant                                                 mol^-1
``k``                        Boltzmann constant                                                J K^-1
``Boltzmann``                Boltzmann constant                                                J K^-1
``sigma``                    Stefan-Boltzmann constant :math:`\sigma`                          W m^-2 K^-4
``Stefan_Boltzmann``         Stefan-Boltzmann constant :math:`\sigma`                          W m^-2 K^-4
``Wien``                     Wien wavelength displacement law constant                         m K
``Rydberg``                  Rydberg constant                                                  m^-1
``m_e``                      electron mass                                                     kg
``electron_mass``            electron mass                                                     kg
``m_p``                      proton mass                                                       kg
``proton_mass``              proton mass                                                       kg
``m_n``                      neutron mass                                                      kg
``neutron_mass``             neutron mass                                                      kg
===========================  ================================================================  ===============


Constants database
------------------

In addition to the above variables, :mod:`scipy.constants` also contains the
2022 CODATA recommended values [CODATA2022]_ database containing more physical
constants.

.. autosummary::
   :toctree: generated/

   value      -- Value in physical_constants indexed by key
   unit       -- Unit in physical_constants indexed by key
   precision  -- Relative precision in physical_constants indexed by key
   find       -- Return list of physical_constant keys with a given string
   ConstantWarning -- Constant sought not in newest CODATA data set

.. data:: physical_constants

   Dictionary of physical constants, of the format
   ``physical_constants[name] = (value, unit, uncertainty)``.
   The CODATA database uses ellipses to indicate that a value is defined
   (exactly) in terms of others but cannot be represented exactly with the
   allocated number of digits. In these cases, SciPy calculates the derived
   value and reports it to the full precision of a Python ``float``. Although 
   ``physical_constants`` lists the uncertainty as ``0.0`` to indicate that
   the CODATA value is exact, the value in ``physical_constants`` is still
   subject to the truncation error inherent in double-precision representation.

Available constants:

======================================================================  ====
%(constant_names)s
======================================================================  ====


Units
=====

SI prefixes
-----------

============  =================================================================
``quetta``    :math:`10^{30}`
``ronna``     :math:`10^{27}`
``yotta``     :math:`10^{24}`
``zetta``     :math:`10^{21}`
``exa``       :math:`10^{18}`
``peta``      :math:`10^{15}`
``tera``      :math:`10^{12}`
``giga``      :math:`10^{9}`
``mega``      :math:`10^{6}`
``kilo``      :math:`10^{3}`
``hecto``     :math:`10^{2}`
``deka``      :math:`10^{1}`
``deci``      :math:`10^{-1}`
``centi``     :math:`10^{-2}`
``milli``     :math:`10^{-3}`
``micro``     :math:`10^{-6}`
``nano``      :math:`10^{-9}`
``pico``      :math:`10^{-12}`
``femto``     :math:`10^{-15}`
``atto``      :math:`10^{-18}`
``zepto``     :math:`10^{-21}`
``yocto``     :math:`10^{-24}`
``ronto``     :math:`10^{-27}`
``quecto``    :math:`10^{-30}`
============  =================================================================

Binary prefixes
---------------

============  =================================================================
``kibi``      :math:`2^{10}`
``mebi``      :math:`2^{20}`
``gibi``      :math:`2^{30}`
``tebi``      :math:`2^{40}`
``pebi``      :math:`2^{50}`
``exbi``      :math:`2^{60}`
``zebi``      :math:`2^{70}`
``yobi``      :math:`2^{80}`
============  =================================================================

Mass
----

=================  ============================================================
``gram``           :math:`10^{-3}` kg
``metric_ton``     :math:`10^{3}` kg
``grain``          one grain in kg
``lb``             one pound (avoirdupous) in kg
``pound``          one pound (avoirdupous) in kg
``blob``           one inch version of a slug in kg (added in 1.0.0)
``slinch``         one inch version of a slug in kg (added in 1.0.0)
``slug``           one slug in kg (added in 1.0.0)
``oz``             one ounce in kg
``ounce``          one ounce in kg
``stone``          one stone in kg
``grain``          one grain in kg
``long_ton``       one long ton in kg
``short_ton``      one short ton in kg
``troy_ounce``     one Troy ounce in kg
``troy_pound``     one Troy pound in kg
``carat``          one carat in kg
``m_u``            atomic mass constant (in kg)
``u``              atomic mass constant (in kg)
``atomic_mass``    atomic mass constant (in kg)
=================  ============================================================

Angle
-----

=================  ============================================================
``degree``         degree in radians
``arcmin``         arc minute in radians
``arcminute``      arc minute in radians
``arcsec``         arc second in radians
``arcsecond``      arc second in radians
=================  ============================================================


Time
----

=================  ============================================================
``minute``         one minute in seconds
``hour``           one hour in seconds
``day``            one day in seconds
``week``           one week in seconds
``year``           one year (365 days) in seconds
``Julian_year``    one Julian year (365.25 days) in seconds
=================  ============================================================


Length
------

=====================  ============================================================
``inch``               one inch in meters
``foot``               one foot in meters
``yard``               one yard in meters
``mile``               one mile in meters
``mil``                one mil in meters
``pt``                 one point in meters
``point``              one point in meters
``survey_foot``        one survey foot in meters
``survey_mile``        one survey mile in meters
``nautical_mile``      one nautical mile in meters
``fermi``              one Fermi in meters
``angstrom``           one Angstrom in meters
``micron``             one micron in meters
``au``                 one astronomical unit in meters
``astronomical_unit``  one astronomical unit in meters
``light_year``         one light year in meters
``parsec``             one parsec in meters
=====================  ============================================================

Pressure
--------

=================  ============================================================
``atm``            standard atmosphere in pascals
``atmosphere``     standard atmosphere in pascals
``bar``            one bar in pascals
``torr``           one torr (mmHg) in pascals
``mmHg``           one torr (mmHg) in pascals
``psi``            one psi in pascals
=================  ============================================================

Area
----

=================  ============================================================
``hectare``        one hectare in square meters
``acre``           one acre in square meters
=================  ============================================================


Volume
------

===================    ========================================================
``liter``              one liter in cubic meters
``litre``              one liter in cubic meters
``gallon``             one gallon (US) in cubic meters
``gallon_US``          one gallon (US) in cubic meters
``gallon_imp``         one gallon (UK) in cubic meters
``fluid_ounce``        one fluid ounce (US) in cubic meters
``fluid_ounce_US``     one fluid ounce (US) in cubic meters
``fluid_ounce_imp``    one fluid ounce (UK) in cubic meters
``bbl``                one barrel in cubic meters
``barrel``             one barrel in cubic meters
===================    ========================================================

Speed
-----

==================    ==========================================================
``kmh``               kilometers per hour in meters per second
``mph``               miles per hour in meters per second
``mach``              one Mach (approx., at 15 C, 1 atm) in meters per second
``speed_of_sound``    one Mach (approx., at 15 C, 1 atm) in meters per second
``knot``              one knot in meters per second
==================    ==========================================================


Temperature
-----------

=====================  =======================================================
``zero_Celsius``       zero of Celsius scale in Kelvin
``degree_Fahrenheit``  one Fahrenheit (only differences) in Kelvins
=====================  =======================================================

.. autosummary::
   :toctree: generated/

   convert_temperature

Energy
------

====================  =======================================================
``eV``                one electron volt in Joules
``electron_volt``     one electron volt in Joules
``calorie``           one calorie (thermochemical) in Joules
``calorie_th``        one calorie (thermochemical) in Joules
``calorie_IT``        one calorie (International Steam Table calorie, 1956) in Joules
``erg``               one erg in Joules
``Btu``               one British thermal unit (International Steam Table) in Joules
``Btu_IT``            one British thermal unit (International Steam Table) in Joules
``Btu_th``            one British thermal unit (thermochemical) in Joules
``ton_TNT``           one ton of TNT in Joules
====================  =======================================================

Power
-----

====================  =======================================================
``hp``                one horsepower in watts
``horsepower``        one horsepower in watts
====================  =======================================================

Force
-----

====================  =======================================================
``dyn``               one dyne in newtons
``dyne``              one dyne in newtons
``lbf``               one pound force in newtons
``pound_force``       one pound force in newtons
``kgf``               one kilogram force in newtons
``kilogram_force``    one kilogram force in newtons
====================  =======================================================

Optics
------

.. autosummary::
   :toctree: generated/

   lambda2nu
   nu2lambda

References
==========

.. [CODATA2022] CODATA Recommended Values of the Fundamental
   Physical Constants 2022.

   https://physics.nist.gov/cuu/Constants/

"""  # noqa: E501
# Modules contributed by BasSw (wegwerp@gmail.com)
from ._codata import *
from ._constants import *
from ._codata import _obsolete_constants, physical_constants

# Deprecated namespaces, to be removed in v2.0.0
from . import codata, constants

_constant_names_list = [(_k.lower(), _k, _v)
                        for _k, _v in physical_constants.items()
                        if _k not in _obsolete_constants]
_constant_names = "\n".join(["``{}``{}  {} {}".format(_x[1], " "*(66-len(_x[1])),
                                                  _x[2][0], _x[2][1])
                             for _x in sorted(_constant_names_list)])
if __doc__:
    __doc__ = __doc__ % dict(constant_names=_constant_names)

del _constant_names
del _constant_names_list

__all__ = [s for s in dir() if not s.startswith('_')]

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
<|endoftext|>"""
Collection of physical constants and conversion factors.

Most constants are in SI units, so you can do
print '10 mile per minute is', 10*mile/minute, 'm/s or', 10*mile/(minute*knot), 'knots'

The list is not meant to be comprehensive, but just convenient for everyday use.
"""

import math as _math
from typing import TYPE_CHECKING, Any

from ._codata import value as _cd

if TYPE_CHECKING:
    import numpy.typing as npt

from scipy._lib._array_api import array_namespace, _asarray


"""
BasSw 2006
physical constants: imported from CODATA
unit conversion: see e.g., NIST special publication 811
Use at own risk: double-check values before calculating your Mars orbit-insertion burn.
Some constants exist in a few variants, which are marked with suffixes.
The ones without any suffix should be the most common ones.
"""

__all__ = [
    'Avogadro', 'Boltzmann', 'Btu', 'Btu_IT', 'Btu_th', 'G',
    'Julian_year', 'N_A', 'Planck', 'R', 'Rydberg',
    'Stefan_Boltzmann', 'Wien', 'acre', 'alpha',
    'angstrom', 'arcmin', 'arcminute', 'arcsec',
    'arcsecond', 'astronomical_unit', 'atm',
    'atmosphere', 'atomic_mass', 'atto', 'au', 'bar',
    'barrel', 'bbl', 'blob', 'c', 'calorie',
    'calorie_IT', 'calorie_th', 'carat', 'centi',
    'convert_temperature', 'day', 'deci', 'degree',
    'degree_Fahrenheit', 'deka', 'dyn', 'dyne', 'e',
    'eV', 'electron_mass', 'electron_volt',
    'elementary_charge', 'epsilon_0', 'erg',
    'exa', 'exbi', 'femto', 'fermi', 'fine_structure',
    'fluid_ounce', 'fluid_ounce_US', 'fluid_ounce_imp',
    'foot', 'g', 'gallon', 'gallon_US', 'gallon_imp',
    'gas_constant', 'gibi', 'giga', 'golden', 'golden_ratio',
    'grain', 'gram', 'gravitational_constant', 'h', 'hbar',
    'hectare', 'hecto', 'horsepower', 'hour', 'hp',
    'inch', 'k', 'kgf', 'kibi', 'kilo', 'kilogram_force',
    'kmh', 'knot', 'lambda2nu', 'lb', 'lbf',
    'light_year', 'liter', 'litre', 'long_ton', 'm_e',
    'm_n', 'm_p', 'm_u', 'mach', 'mebi', 'mega',
    'metric_ton', 'micro', 'micron', 'mil', 'mile',
    'milli', 'minute', 'mmHg', 'mph', 'mu_0', 'nano',
    'nautical_mile', 'neutron_mass', 'nu2lambda',
    'ounce', 'oz', 'parsec', 'pebi', 'peta',
    'pi', 'pico', 'point', 'pound', 'pound_force',
    'proton_mass', 'psi', 'pt', 'quecto', 'quetta', 'ronna', 'ronto',
    'short_ton', 'sigma', 'slinch', 'slug', 'speed_of_light',
    'speed_of_sound', 'stone', 'survey_foot',
    'survey_mile', 'tebi', 'tera', 'ton_TNT',
    'torr', 'troy_ounce', 'troy_pound', 'u',
    'week', 'yard', 'year', 'yobi', 'yocto',
    'yotta', 'zebi', 'zepto', 'zero_Celsius', 'zetta'
]


# mathematical constants
pi = _math.pi
golden = golden_ratio = (1 + _math.sqrt(5)) / 2

# SI prefixes
quetta = 1e30
ronna = 1e27
yotta = 1e24
zetta = 1e21
exa = 1e18
peta = 1e15
tera = 1e12
giga = 1e9
mega = 1e6
kilo = 1e3
hecto = 1e2
deka = 1e1
deci = 1e-1
centi = 1e-2
milli = 1e-3
micro = 1e-6
nano = 1e-9
pico = 1e-12
femto = 1e-15
atto = 1e-18
zepto = 1e-21
yocto = 1e-24
ronto = 1e-27
quecto = 1e-30

# binary prefixes
kibi = 2**10
mebi = 2**20
gibi = 2**30
tebi = 2**40
pebi = 2**50
exbi = 2**60
zebi = 2**70
yobi = 2**80

# physical constants
c = speed_of_light = _cd('speed of light in vacuum')
mu_0 = _cd('vacuum mag. permeability')
epsilon_0 = _cd('vacuum electric permittivity')
h = Planck = _cd('Planck constant')
hbar = _cd('reduced Planck constant')
G = gravitational_constant = _cd('Newtonian constant of gravitation')
g = _cd('standard acceleration of gravity')
e = elementary_charge = _cd('elementary charge')
R = gas_constant = _cd('molar gas constant')
alpha = fine_structure = _cd('fine-structure constant')
N_A = Avogadro = _cd('Avogadro constant')
k = Boltzmann = _cd('Boltzmann constant')
sigma = Stefan_Boltzmann = _cd('Stefan-Boltzmann constant')
Wien = _cd('Wien wavelength displacement law constant')
Rydberg = _cd('Rydberg constant')

# mass in kg
gram = 1e-3
metric_ton = 1e3
grain = 64.79891e-6
lb = pound = 7000 * grain  # avoirdupois
blob = slinch = pound * g / 0.0254  # lbf*s**2/in (added in 1.0.0)
slug = blob / 12  # lbf*s**2/foot (added in 1.0.0)
oz = ounce = pound / 16
stone = 14 * pound
long_ton = 2240 * pound
short_ton = 2000 * pound

troy_ounce = 480 * grain  # only for metals / gems
troy_pound = 12 * troy_ounce
carat = 200e-6

m_e = electron_mass = _cd('electron mass')
m_p = proton_mass = _cd('proton mass')
m_n = neutron_mass = _cd('neutron mass')
m_u = u = atomic_mass = _cd('atomic mass constant')

# angle in rad
degree = pi / 180
arcmin = arcminute = degree / 60
arcsec = arcsecond = arcmin / 60

# time in second
minute = 60.0
hour = 60 * minute
day = 24 * hour
week = 7 * day
year = 365 * day
Julian_year = 365.25 * day

# length in meter
inch = 0.0254
foot = 12 * inch
yard = 3 * foot
mile = 1760 * yard
mil = inch / 1000
pt = point = inch / 72  # typography
survey_foot = 1200.0 / 3937
survey_mile = 5280 * survey_foot
nautical_mile = 1852.0
fermi = 1e-15
angstrom = 1e-10
micron = 1e-6
au = astronomical_unit = 149597870700.0
light_year = Julian_year * c
parsec = au / arcsec

# pressure in pascal
atm = atmosphere = _cd('standard atmosphere')
bar = 1e5
torr = mmHg = atm / 760
psi = pound * g / (inch * inch)

# area in meter**2
hectare = 1e4
acre = 43560 * foot**2

# volume in meter**3
litre = liter = 1e-3
gallon = gallon_US = 231 * inch**3  # US
# pint = gallon_US / 8
fluid_ounce = fluid_ounce_US = gallon_US / 128
bbl = barrel = 42 * gallon_US  # for oil

gallon_imp = 4.54609e-3  # UK
fluid_ounce_imp = gallon_imp / 160

# speed in meter per second
kmh = 1e3 / hour
mph = mile / hour
# approx value of mach at 15 degrees in 1 atm. Is this a common value?
mach = speed_of_sound = 340.5
knot = nautical_mile / hour

# temperature in kelvin
zero_Celsius = 273.15
degree_Fahrenheit = 1/1.8  # only for differences

# energy in joule
eV = electron_volt = elementary_charge  # * 1 Volt
calorie = calorie_th = 4.184
calorie_IT = 4.1868
erg = 1e-7
Btu_th = pound * degree_Fahrenheit * calorie_th / gram
Btu = Btu_IT = pound * degree_Fahrenheit * calorie_IT / gram
ton_TNT = 1e9 * calorie_th
# Wh = watt_hour

# power in watt
hp = horsepower = 550 * foot * pound * g

# force in newton
dyn = dyne = 1e-5
lbf = pound_force = pound * g
kgf = kilogram_force = g  # * 1 kg

# functions for conversions that are not linear


def convert_temperature(
    val: "npt.ArrayLike",
    old_scale: str,
    new_scale: str,
) -> Any:
    """
    Convert from a temperature scale to another one among Celsius, Kelvin,
    Fahrenheit, and Rankine scales.

    Parameters
    ----------
    val : array_like
        Value(s) of the temperature(s) to be converted expressed in the
        original scale.
    old_scale : str
        Specifies as a string the original scale from which the temperature
        value(s) will be converted. Supported scales are Celsius ('Celsius',
        'celsius', 'C' or 'c'), Kelvin ('Kelvin', 'kelvin', 'K', 'k'),
        Fahrenheit ('Fahrenheit', 'fahrenheit', 'F' or 'f'), and Rankine
        ('Rankine', 'rankine', 'R', 'r').
    new_scale : str
        Specifies as a string the new scale to which the temperature
        value(s) will be converted. Supported scales are Celsius ('Celsius',
        'celsius', 'C' or 'c'), Kelvin ('Kelvin', 'kelvin', 'K', 'k'),
        Fahrenheit ('Fahrenheit', 'fahrenheit', 'F' or 'f'), and Rankine
        ('Rankine', 'rankine', 'R', 'r').

    Returns
    -------
    res : float or array of floats
        Value(s) of the converted temperature(s) expressed in the new scale.

    Notes
    -----
    .. versionadded:: 0.18.0

    Examples
    --------
    >>> from scipy.constants import convert_temperature
    >>> import numpy as np
    >>> convert_temperature(np.array([-40, 40]), 'Celsius', 'Kelvin')
    array([ 233.15,  313.15])

    """
    xp = array_namespace(val)
    _val = _asarray(val, xp=xp, subok=True)
    # Convert from `old_scale` to Kelvin
    if old_scale.lower() in ['celsius', 'c']:
        tempo = _val + zero_Celsius
    elif old_scale.lower() in ['kelvin', 'k']:
        tempo = _val
    elif old_scale.lower() in ['fahrenheit', 'f']:
        tempo = (_val - 32) * 5 / 9 + zero_Celsius
    elif old_scale.lower() in ['rankine', 'r']:
        tempo = _val * 5 / 9
    else:
        raise NotImplementedError(f"{old_scale=} is unsupported: supported scales "
                                   "are Celsius, Kelvin, Fahrenheit, and "
                                   "Rankine")
    # and from Kelvin to `new_scale`.
    if new_scale.lower() in ['celsius', 'c']:
        res = tempo - zero_Celsius
    elif new_scale.lower() in ['kelvin', 'k']:
        res = tempo
    elif new_scale.lower() in ['fahrenheit', 'f']:
        res = (tempo - zero_Celsius) * 9 / 5 + 32
    elif new_scale.lower() in ['rankine', 'r']:
        res = tempo * 9 / 5
    else:
        raise NotImplementedError(f"{new_scale=} is unsupported: supported "
                                   "scales are 'Celsius', 'Kelvin', "
                                   "'Fahrenheit', and 'Rankine'")

    return res


# optics


def lambda2nu(lambda_: "npt.ArrayLike") -> Any:
    """
    Convert wavelength to optical frequency

    Parameters
    ----------
    lambda_ : array_like
        Wavelength(s) to be converted.

    Returns
    -------
    nu : float or array of floats
        Equivalent optical frequency.

    Notes
    -----
    Computes ``nu = c / lambda`` where c = 299792458.0, i.e., the
    (vacuum) speed of light in meters/second.

    Examples
    --------
    >>> from scipy.constants import lambda2nu, speed_of_light
    >>> import numpy as np
    >>> lambda2nu(np.array((1, speed_of_light)))
    array([  2.99792458e+08,   1.00000000e+00])

    """
    xp = array_namespace(lambda_)
    return c / _asarray(lambda_, xp=xp, subok=True)


def nu2lambda(nu: "npt.ArrayLike") -> Any:
    """
    Convert optical frequency to wavelength.

    Parameters
    ----------
    nu : array_like
        Optical frequency to be converted.

    Returns
    -------
    lambda : float or array of floats
        Equivalent wavelength(s).

    Notes
    -----
    Computes ``lambda = c / nu`` where c = 299792458.0, i.e., the
    (vacuum) speed of light in meters/second.

    Examples
    --------
    >>> from scipy.constants import nu2lambda, speed_of_light
    >>> import numpy as np
    >>> nu2lambda(np.array((1, speed_of_light)))
    array([  2.99792458e+08,   1.00000000e+00])

    """
    xp = array_namespace(nu)
    return c / _asarray(nu, xp=xp, subok=True)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.constants` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation

__all__ = [  # noqa: F822
    'physical_constants', 'value', 'unit', 'precision', 'find',
    'ConstantWarning', 'k', 'c',

]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="constants", module="codata",
                                   private_modules=["_codata"], all=__all__,
                                   attribute=name)
<|endoftext|>"""
NetCDF reader/writer module.

This module is used to read and create NetCDF files. NetCDF files are
accessed through the `netcdf_file` object. Data written to and from NetCDF
files are contained in `netcdf_variable` objects. Attributes are given
as member variables of the `netcdf_file` and `netcdf_variable` objects.

This module implements the Scientific.IO.NetCDF API to read and create
NetCDF files. The same API is also used in the PyNIO and pynetcdf
modules, allowing these modules to be used interchangeably when working
with NetCDF files.

Only NetCDF3 is supported here; for NetCDF4 see
`netCDF4-python <http://unidata.github.io/netcdf4-python/>`__,
which has a similar API.

"""

# TODO:
# * properly implement ``_FillValue``.
# * fix character variables.
# * implement PAGESIZE for Python 2.6?

# The Scientific.IO.NetCDF API allows attributes to be added directly to
# instances of ``netcdf_file`` and ``netcdf_variable``. To differentiate
# between user-set attributes and instance attributes, user-set attributes
# are automatically stored in the ``_attributes`` attribute by overloading
#``__setattr__``. This is the reason why the code sometimes uses
#``obj.__dict__['key'] = value``, instead of simply ``obj.key = value``;
# otherwise the key would be inserted into userspace attributes.


__all__ = ['netcdf_file', 'netcdf_variable']


import warnings
import weakref
from operator import mul
from platform import python_implementation

import mmap as mm

import numpy as np
from numpy import frombuffer, dtype, empty, array, asarray
from numpy import little_endian as LITTLE_ENDIAN
from functools import reduce


IS_PYPY = python_implementation() == 'PyPy'

ABSENT = b'\x00\x00\x00\x00\x00\x00\x00\x00'
ZERO = b'\x00\x00\x00\x00'
NC_BYTE = b'\x00\x00\x00\x01'
NC_CHAR = b'\x00\x00\x00\x02'
NC_SHORT = b'\x00\x00\x00\x03'
NC_INT = b'\x00\x00\x00\x04'
NC_FLOAT = b'\x00\x00\x00\x05'
NC_DOUBLE = b'\x00\x00\x00\x06'
NC_DIMENSION = b'\x00\x00\x00\n'
NC_VARIABLE = b'\x00\x00\x00\x0b'
NC_ATTRIBUTE = b'\x00\x00\x00\x0c'
FILL_BYTE = b'\x81'
FILL_CHAR = b'\x00'
FILL_SHORT = b'\x80\x01'
FILL_INT = b'\x80\x00\x00\x01'
FILL_FLOAT = b'\x7C\xF0\x00\x00'
FILL_DOUBLE = b'\x47\x9E\x00\x00\x00\x00\x00\x00'

TYPEMAP = {NC_BYTE: ('b', 1),
           NC_CHAR: ('c', 1),
           NC_SHORT: ('h', 2),
           NC_INT: ('i', 4),
           NC_FLOAT: ('f', 4),
           NC_DOUBLE: ('d', 8)}

FILLMAP = {NC_BYTE: FILL_BYTE,
           NC_CHAR: FILL_CHAR,
           NC_SHORT: FILL_SHORT,
           NC_INT: FILL_INT,
           NC_FLOAT: FILL_FLOAT,
           NC_DOUBLE: FILL_DOUBLE}

REVERSE = {('b', 1): NC_BYTE,
           ('B', 1): NC_CHAR,
           ('c', 1): NC_CHAR,
           ('h', 2): NC_SHORT,
           ('i', 4): NC_INT,
           ('f', 4): NC_FLOAT,
           ('d', 8): NC_DOUBLE,

           # these come from asarray(1).dtype.char and asarray('foo').dtype.char,
           # used when getting the types from generic attributes.
           ('l', 4): NC_INT,
           ('S', 1): NC_CHAR}


class netcdf_file:
    """
    A file object for NetCDF data.

    A `netcdf_file` object has two standard attributes: `dimensions` and
    `variables`. The values of both are dictionaries, mapping dimension
    names to their associated lengths and variable names to variables,
    respectively. Application programs should never modify these
    dictionaries.

    All other attributes correspond to global attributes defined in the
    NetCDF file. Global file attributes are created by assigning to an
    attribute of the `netcdf_file` object.

    Parameters
    ----------
    filename : string or file-like
        string -> filename
    mode : {'r', 'w', 'a'}, optional
        read-write-append mode, default is 'r'
    mmap : None or bool, optional
        Whether to mmap `filename` when reading.  Default is True
        when `filename` is a file name, False when `filename` is a
        file-like object. Note that when mmap is in use, data arrays
        returned refer directly to the mmapped data on disk, and the
        file cannot be closed as long as references to it exist.
    version : {1, 2}, optional
        version of netcdf to read / write, where 1 means *Classic
        format* and 2 means *64-bit offset format*.  Default is 1.  See
        `here <https://docs.unidata.ucar.edu/nug/current/netcdf_introduction.html#select_format>`__
        for more info.
    maskandscale : bool, optional
        Whether to automatically scale and/or mask data based on attributes.
        Default is False.

    Notes
    -----
    The major advantage of this module over other modules is that it doesn't
    require the code to be linked to the NetCDF libraries. This module is
    derived from `pupynere <https://bitbucket.org/robertodealmeida/pupynere/>`_.

    NetCDF files are a self-describing binary data format. The file contains
    metadata that describes the dimensions and variables in the file. More
    details about NetCDF files can be found `here
    <https://www.unidata.ucar.edu/software/netcdf/guide_toc.html>`__. There
    are three main sections to a NetCDF data structure:

    1. Dimensions
    2. Variables
    3. Attributes

    The dimensions section records the name and length of each dimension used
    by the variables. The variables would then indicate which dimensions it
    uses and any attributes such as data units, along with containing the data
    values for the variable. It is good practice to include a
    variable that is the same name as a dimension to provide the values for
    that axes. Lastly, the attributes section would contain additional
    information such as the name of the file creator or the instrument used to
    collect the data.

    When writing data to a NetCDF file, there is often the need to indicate the
    'record dimension'. A record dimension is the unbounded dimension for a
    variable. For example, a temperature variable may have dimensions of
    latitude, longitude and time. If one wants to add more temperature data to
    the NetCDF file as time progresses, then the temperature variable should
    have the time dimension flagged as the record dimension.

    In addition, the NetCDF file header contains the position of the data in
    the file, so access can be done in an efficient manner without loading
    unnecessary data into memory. It uses the ``mmap`` module to create
    Numpy arrays mapped to the data on disk, for the same purpose.

    Note that when `netcdf_file` is used to open a file with mmap=True
    (default for read-only), arrays returned by it refer to data
    directly on the disk. The file should not be closed, and cannot be cleanly
    closed when asked, if such arrays are alive. You may want to copy data arrays
    obtained from mmapped Netcdf file if they are to be processed after the file
    is closed, see the example below.

    Examples
    --------
    To create a NetCDF file:

    >>> from scipy.io import netcdf_file
    >>> import numpy as np
    >>> f = netcdf_file('simple.nc', 'w')
    >>> f.history = 'Created for a test'
    >>> f.createDimension('time', 10)
    >>> time = f.createVariable('time', 'i', ('time',))
    >>> time[:] = np.arange(10)
    >>> time.units = 'days since 2008-01-01'
    >>> f.close()

    Note the assignment of ``arange(10)`` to ``time[:]``.  Exposing the slice
    of the time variable allows for the data to be set in the object, rather
    than letting ``arange(10)`` overwrite the ``time`` variable.

    To read the NetCDF file we just created:

    >>> from scipy.io import netcdf_file
    >>> f = netcdf_file('simple.nc', 'r')
    >>> print(f.history)
    b'Created for a test'
    >>> time = f.variables['time']
    >>> print(time.units)
    b'days since 2008-01-01'
    >>> print(time.shape)
    (10,)
    >>> print(time[-1])
    9

    NetCDF files, when opened read-only, return arrays that refer
    directly to memory-mapped data on disk:

    >>> data = time[:]

    If the data is to be processed after the file is closed, it needs
    to be copied to main memory:

    >>> data = time[:].copy()
    >>> del time
    >>> f.close()
    >>> data.mean()
    4.5

    A NetCDF file can also be used as context manager:

    >>> from scipy.io import netcdf_file
    >>> with netcdf_file('simple.nc', 'r') as f:
    ...     print(f.history)
    b'Created for a test'

    """
    def __init__(self, filename, mode='r', mmap=None, version=1,
                 maskandscale=False):
        """Initialize netcdf_file from fileobj (str or file-like)."""
        if mode not in 'rwa':
            raise ValueError("Mode must be either 'r', 'w' or 'a'.")

        if hasattr(filename, 'seek'):  # file-like
            self.fp = filename
            self.filename = 'None'
            if mmap is None:
                mmap = False
            elif mmap and not hasattr(filename, 'fileno'):
                raise ValueError('Cannot use file object for mmap')
        else:  # maybe it's a string
            self.filename = filename
            omode = 'r+' if mode == 'a' else mode
            self.fp = open(self.filename, f'{omode}b')
            if mmap is None:
                # Mmapped files on PyPy cannot be usually closed
                # before the GC runs, so it's better to use mmap=False
                # as the default.
                mmap = (not IS_PYPY)

        if mode != 'r':
            # Cannot read write-only files
            mmap = False

        self.use_mmap = mmap
        self.mode = mode
        self.version_byte = version
        self.maskandscale = maskandscale

        self.dimensions = {}
        self.variables = {}

        self._dims = []
        self._recs = 0
        self._recsize = 0

        self._mm = None
        self._mm_buf = None
        if self.use_mmap:
            self._mm = mm.mmap(self.fp.fileno(), 0, access=mm.ACCESS_READ)
            self._mm_buf = np.frombuffer(self._mm, dtype=np.int8)

        self._attributes = {}

        if mode in 'ra':
            self._read()

    def __setattr__(self, attr, value):
        # Store user defined attributes in a separate dict,
        # so we can save them to file later.
        try:
            self._attributes[attr] = value
        except AttributeError:
            pass
        self.__dict__[attr] = value

    def close(self):
        """Closes the NetCDF file."""
        if hasattr(self, 'fp') and not self.fp.closed:
            try:
                self.flush()
            finally:
                self.variables = {}
                if self._mm_buf is not None:
                    ref = weakref.ref(self._mm_buf)
                    self._mm_buf = None
                    if ref() is None:
                        # self._mm_buf is gc'd, and we can close the mmap
                        self._mm.close()
                    else:
                        # we cannot close self._mm, since self._mm_buf is
                        # alive and there may still be arrays referring to it
                        warnings.warn(
                            "Cannot close a netcdf_file opened with mmap=True, when "
                            "netcdf_variables or arrays referring to its data still "
                            "exist. All data arrays obtained from such files refer "
                            "directly to data on disk, and must be copied before the "
                            "file can be cleanly closed. "
                            "(See netcdf_file docstring for more information on mmap.)",
                            category=RuntimeWarning, stacklevel=2,
                        )
                self._mm = None
                self.fp.close()
    __del__ = close

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def createDimension(self, name, length):
        """
        Adds a dimension to the Dimension section of the NetCDF data structure.

        Note that this function merely adds a new dimension that the variables can
        reference. The values for the dimension, if desired, should be added as
        a variable using `createVariable`, referring to this dimension.

        Parameters
        ----------
        name : str
            Name of the dimension (Eg, 'lat' or 'time').
        length : int
            Length of the dimension.

        See Also
        --------
        createVariable

        """
        if length is None and self._dims:
            raise ValueError("Only first dimension may be unlimited!")

        self.dimensions[name] = length
        self._dims.append(name)

    def createVariable(self, name, type, dimensions):
        """
        Create an empty variable for the `netcdf_file` object, specifying its data
        type and the dimensions it uses.

        Parameters
        ----------
        name : str
            Name of the new variable.
        type : dtype or str
            Data type of the variable.
        dimensions : sequence of str
            List of the dimension names used by the variable, in the desired order.

        Returns
        -------
        variable : netcdf_variable
            The newly created ``netcdf_variable`` object.
            This object has also been added to the `netcdf_file` object as well.

        See Also
        --------
        createDimension

        Notes
        -----
        Any dimensions to be used by the variable should already exist in the
        NetCDF data structure or should be created by `createDimension` prior to
        creating the NetCDF variable.

        """
        shape = tuple([self.dimensions[dim] for dim in dimensions])
        shape_ = tuple([dim or 0 for dim in shape])  # replace None with 0 for NumPy

        type = dtype(type)
        typecode, size = type.char, type.itemsize
        if (typecode, size) not in REVERSE:
            raise ValueError(f"NetCDF 3 does not support type {type}")

        # convert to big endian always for NetCDF 3
        data = empty(shape_, dtype=type.newbyteorder("B"))
        self.variables[name] = netcdf_variable(
                data, typecode, size, shape, dimensions,
                maskandscale=self.maskandscale)
        return self.variables[name]

    def flush(self):
        """
        Perform a sync-to-disk flush if the `netcdf_file` object is in write mode.

        See Also
        --------
        sync : Identical function

        """
        if hasattr(self, 'mode') and self.mode in 'wa':
            self._write()
    sync = flush

    def _write(self):
        self.fp.seek(0)
        self.fp.write(b'CDF')
        self.fp.write(array(self.version_byte, '>b').tobytes())

        # Write headers and data.
        self._write_numrecs()
        self._write_dim_array()
        self._write_gatt_array()
        self._write_var_array()

    def _write_numrecs(self):
        # Get highest record count from all record variables.
        for var in self.variables.values():
            if var.isrec and len(var.data) > self._recs:
                self.__dict__['_recs'] = len(var.data)
        self._pack_int(self._recs)

    def _write_dim_array(self):
        if self.dimensions:
            self.fp.write(NC_DIMENSION)
            self._pack_int(len(self.dimensions))
            for name in self._dims:
                self._pack_string(name)
                length = self.dimensions[name]
                self._pack_int(length or 0)  # replace None with 0 for record dimension
        else:
            self.fp.write(ABSENT)

    def _write_gatt_array(self):
        self._write_att_array(self._attributes)

    def _write_att_array(self, attributes):
        if attributes:
            self.fp.write(NC_ATTRIBUTE)
            self._pack_int(len(attributes))
            for name, values in attributes.items():
                self._pack_string(name)
                self._write_att_values(values)
        else:
            self.fp.write(ABSENT)

    def _write_var_array(self):
        if self.variables:
            self.fp.write(NC_VARIABLE)
            self._pack_int(len(self.variables))

            # Sort variable names non-recs first, then recs.
            def sortkey(n):
                v = self.variables[n]
                if v.isrec:
                    return (-1,)
                return v._shape
            variables = sorted(self.variables, key=sortkey, reverse=True)

            # Set the metadata for all variables.
            for name in variables:
                self._write_var_metadata(name)
            # Now that we have the metadata, we know the vsize of
            # each record variable, so we can calculate recsize.
            self.__dict__['_recsize'] = sum([
                    var._vsize for var in self.variables.values()
                    if var.isrec])
            # Set the data for all variables.
            for name in variables:
                self._write_var_data(name)
        else:
            self.fp.write(ABSENT)

    def _write_var_metadata(self, name):
        var = self.variables[name]

        self._pack_string(name)
        self._pack_int(len(var.dimensions))
        for dimname in var.dimensions:
            dimid = self._dims.index(dimname)
            self._pack_int(dimid)

        self._write_att_array(var._attributes)

        nc_type = REVERSE[var.typecode(), var.itemsize()]
        self.fp.write(nc_type)

        if not var.isrec:
            vsize = var.data.size * var.data.itemsize
            vsize += -vsize % 4
        else:  # record variable
            try:
                vsize = var.data[0].size * var.data.itemsize
            except IndexError:
                vsize = 0
            rec_vars = len([v for v in self.variables.values()
                            if v.isrec])
            if rec_vars > 1:
                vsize += -vsize % 4
        self.variables[name].__dict__['_vsize'] = vsize
        self._pack_int(vsize)

        # Pack a bogus begin, and set the real value later.
        self.variables[name].__dict__['_begin'] = self.fp.tell()
        self._pack_begin(0)

    def _write_var_data(self, name):
        var = self.variables[name]

        # Set begin in file header.
        the_beguine = self.fp.tell()
        self.fp.seek(var._begin)
        self._pack_begin(the_beguine)
        self.fp.seek(the_beguine)

        # Write data.
        if not var.isrec:
            self.fp.write(var.data.tobytes())
            count = var.data.size * var.data.itemsize
            self._write_var_padding(var, var._vsize - count)
        else:  # record variable
            # Handle rec vars with shape[0] < nrecs.
            if self._recs > len(var.data):
                shape = (self._recs,) + var.data.shape[1:]
                # Resize in-place does not always work since
                # the array might not be single-segment
                try:
                    var.data.resize(shape)
                except ValueError:
                    dtype = var.data.dtype
                    var.__dict__['data'] = np.resize(var.data, shape).astype(dtype)

            pos0 = pos = self.fp.tell()
            for rec in var.data:
                # Apparently scalars cannot be converted to big endian. If we
                # try to convert a ``=i4`` scalar to, say, '>i4' the dtype
                # will remain as ``=i4``.
                if not rec.shape and (rec.dtype.byteorder == '<' or
                        (rec.dtype.byteorder == '=' and LITTLE_ENDIAN)):
                    rec = rec.byteswap()
                self.fp.write(rec.tobytes())
                # Padding
                count = rec.size * rec.itemsize
                self._write_var_padding(var, var._vsize - count)
                pos += self._recsize
                self.fp.seek(pos)
            self.fp.seek(pos0 + var._vsize)

    def _write_var_padding(self, var, size):
        encoded_fill_value = var._get_encoded_fill_value()
        num_fills = size // len(encoded_fill_value)
        self.fp.write(encoded_fill_value * num_fills)

    def _write_att_values(self, values):
        if hasattr(values, 'dtype'):
            nc_type = REVERSE[values.dtype.char, values.dtype.itemsize]
        else:
            types = [(int, NC_INT), (float, NC_FLOAT), (str, NC_CHAR)]

            # bytes index into scalars in py3k. Check for "string" types
            if isinstance(values, (str, bytes)):
                sample = values
            else:
                try:
                    sample = values[0]  # subscriptable?
                except TypeError:
                    sample = values     # scalar

            for class_, nc_type in types:
                if isinstance(sample, class_):
                    break

        typecode, size = TYPEMAP[nc_type]
        dtype_ = f'>{typecode}'
        # asarray() dies with bytes and '>c' in py3k. Change to 'S'
        dtype_ = 'S' if dtype_ == '>c' else dtype_

        values = asarray(values, dtype=dtype_)

        self.fp.write(nc_type)

        if values.dtype.char == 'S':
            nelems = values.itemsize
        else:
            nelems = values.size
        self._pack_int(nelems)

        if not values.shape and (values.dtype.byteorder == '<' or
                (values.dtype.byteorder == '=' and LITTLE_ENDIAN)):
            values = values.byteswap()
        self.fp.write(values.tobytes())
        count = values.size * values.itemsize
        self.fp.write(b'\x00' * (-count % 4))  # pad

    def _read(self):
        # Check magic bytes and version
        magic = self.fp.read(3)
        if not magic == b'CDF':
            raise TypeError(f"Error: {self.filename} is not a valid NetCDF 3 file")
        self.__dict__['version_byte'] = frombuffer(self.fp.read(1), '>b')[0]

        # Read file headers and set data.
        self._read_numrecs()
        self._read_dim_array()
        self._read_gatt_array()
        self._read_var_array()

    def _read_numrecs(self):
        self.__dict__['_recs'] = self._unpack_int()

    def _read_dim_array(self):
        header = self.fp.read(4)
        if header not in [ZERO, NC_DIMENSION]:
            raise ValueError("Unexpected header.")
        count = self._unpack_int()

        for dim in range(count):
            name = self._unpack_string().decode('latin1')
            length = self._unpack_int() or None  # None for record dimension
            self.dimensions[name] = length
            self._dims.append(name)  # preserve order

    def _read_gatt_array(self):
        for k, v in self._read_att_array().items():
            self.__setattr__(k, v)

    def _read_att_array(self):
        header = self.fp.read(4)
        if header not in [ZERO, NC_ATTRIBUTE]:
            raise ValueError("Unexpected header.")
        count = self._unpack_int()

        attributes = {}
        for attr in range(count):
            name = self._unpack_string().decode('latin1')
            attributes[name] = self._read_att_values()
        return attributes

    def _read_var_array(self):
        header = self.fp.read(4)
        if header not in [ZERO, NC_VARIABLE]:
            raise ValueError("Unexpected header.")

        begin = 0
        dtypes = {'names': [], 'formats': []}
        rec_vars = []
        count = self._unpack_int()
        for var in range(count):
            (name, dimensions, shape, attributes,
             typecode, size, dtype_, begin_, vsize) = self._read_var()
            # https://www.unidata.ucar.edu/software/netcdf/guide_toc.html
            # Note that vsize is the product of the dimension lengths
            # (omitting the record dimension) and the number of bytes
            # per value (determined from the type), increased to the
            # next multiple of 4, for each variable. If a record
            # variable, this is the amount of space per record. The
            # netCDF "record size" is calculated as the sum of the
            # vsize's of all the record variables.
            #
            # The vsize field is actually redundant, because its value
            # may be computed from other information in the header. The
            # 32-bit vsize field is not large enough to contain the size
            # of variables that require more than 2^32 - 4 bytes, so
            # 2^32 - 1 is used in the vsize field for such variables.
            if shape and shape[0] is None:  # record variable
                rec_vars.append(name)
                # The netCDF "record size" is calculated as the sum of
                # the vsize's of all the record variables.
                self.__dict__['_recsize'] += vsize
                if begin == 0:
                    begin = begin_
                dtypes['names'].append(name)
                dtypes['formats'].append(str(shape[1:]) + dtype_)

                # Handle padding with a virtual variable.
                if typecode in 'bch':
                    actual_size = reduce(mul, (1,) + shape[1:]) * size
                    padding = -actual_size % 4
                    if padding:
                        dtypes['names'].append('_padding_%d' % var)
                        dtypes['formats'].append('(%d,)>b' % padding)

                # Data will be set later.
                data = None
            else:  # not a record variable
                # Calculate size to avoid problems with vsize (above)
                a_size = reduce(mul, shape, 1) * size
                if self.use_mmap:
                    data = self._mm_buf[begin_:begin_+a_size].view(dtype=dtype_)
                    data.shape = shape
                else:
                    pos = self.fp.tell()
                    self.fp.seek(begin_)
                    data = frombuffer(self.fp.read(a_size), dtype=dtype_
                                      ).copy()
                    data.shape = shape
                    self.fp.seek(pos)

            # Add variable.
            self.variables[name] = netcdf_variable(
                    data, typecode, size, shape, dimensions, attributes,
                    maskandscale=self.maskandscale)

        if rec_vars:
            # Remove padding when only one record variable.
            if len(rec_vars) == 1:
                dtypes['names'] = dtypes['names'][:1]
                dtypes['formats'] = dtypes['formats'][:1]

            # Build rec array.
            if self.use_mmap:
                buf = self._mm_buf[begin:begin+self._recs*self._recsize]
                rec_array = buf.view(dtype=dtypes)
                rec_array.shape = (self._recs,)
            else:
                pos = self.fp.tell()
                self.fp.seek(begin)
                rec_array = frombuffer(self.fp.read(self._recs*self._recsize),
                                       dtype=dtypes).copy()
                rec_array.shape = (self._recs,)
                self.fp.seek(pos)

            for var in rec_vars:
                self.variables[var].__dict__['data'] = rec_array[var]

    def _read_var(self):
        name = self._unpack_string().decode('latin1')
        dimensions = []
        shape = []
        dims = self._unpack_int()

        for i in range(dims):
            dimid = self._unpack_int()
            dimname = self._dims[dimid]
            dimensions.append(dimname)
            dim = self.dimensions[dimname]
            shape.append(dim)
        dimensions = tuple(dimensions)
        shape = tuple(shape)

        attributes = self._read_att_array()
        nc_type = self.fp.read(4)
        vsize = self._unpack_int()
        begin = [self._unpack_int, self._unpack_int64][self.version_byte-1]()

        typecode, size = TYPEMAP[nc_type]
        dtype_ = f'>{typecode}'

        return name, dimensions, shape, attributes, typecode, size, dtype_, begin, vsize

    def _read_att_values(self):
        nc_type = self.fp.read(4)
        n = self._unpack_int()

        typecode, size = TYPEMAP[nc_type]

        count = n*size
        values = self.fp.read(int(count))
        self.fp.read(-count % 4)  # read padding

        if typecode != 'c':
            values = frombuffer(values, dtype=f'>{typecode}').copy()
            if values.shape == (1,):
                values = values[0]
        else:
            values = values.rstrip(b'\x00')
        return values

    def _pack_begin(self, begin):
        if self.version_byte == 1:
            self._pack_int(begin)
        elif self.version_byte == 2:
            self._pack_int64(begin)

    def _pack_int(self, value):
        self.fp.write(array(value, '>i').tobytes())
    _pack_int32 = _pack_int

    def _unpack_int(self):
        return int(frombuffer(self.fp.read(4), '>i')[0])
    _unpack_int32 = _unpack_int

    def _pack_int64(self, value):
        self.fp.write(array(value, '>q').tobytes())

    def _unpack_int64(self):
        return frombuffer(self.fp.read(8), '>q')[0]

    def _pack_string(self, s):
        count = len(s)
        self._pack_int(count)
        self.fp.write(s.encode('latin1'))
        self.fp.write(b'\x00' * (-count % 4))  # pad

    def _unpack_string(self):
        count = self._unpack_int()
        s = self.fp.read(count).rstrip(b'\x00')
        self.fp.read(-count % 4)  # read padding
        return s


class netcdf_variable:
    """
    A data object for netcdf files.

    `netcdf_variable` objects are constructed by calling the method
    `netcdf_file.createVariable` on the `netcdf_file` object. `netcdf_variable`
    objects behave much like array objects defined in numpy, except that their
    data resides in a file. Data is read by indexing and written by assigning
    to an indexed subset; the entire array can be accessed by the index ``[:]``
    or (for scalars) by using the methods `getValue` and `assignValue`.
    `netcdf_variable` objects also have attribute `shape` with the same meaning
    as for arrays, but the shape cannot be modified. There is another read-only
    attribute `dimensions`, whose value is the tuple of dimension names.

    All other attributes correspond to variable attributes defined in
    the NetCDF file. Variable attributes are created by assigning to an
    attribute of the `netcdf_variable` object.

    Parameters
    ----------
    data : array_like
        The data array that holds the values for the variable.
        Typically, this is initialized as empty, but with the proper shape.
    typecode : dtype character code
        Desired data-type for the data array.
    size : int
        Desired element size for the data array.
    shape : sequence of ints
        The shape of the array. This should match the lengths of the
        variable's dimensions.
    dimensions : sequence of strings
        The names of the dimensions used by the variable. Must be in the
        same order of the dimension lengths given by `shape`.
    attributes : dict, optional
        Attribute values (any type) keyed by string names. These attributes
        become attributes for the netcdf_variable object.
    maskandscale : bool, optional
        Whether to automatically scale and/or mask data based on attributes.
        Default is False.


    Attributes
    ----------
    dimensions : list of str
        List of names of dimensions used by the variable object.
    isrec, shape
        Properties

    See also
    --------
    isrec, shape

    """
    def __init__(self, data, typecode, size, shape, dimensions,
                 attributes=None,
                 maskandscale=False):
        self.data = data
        self._typecode = typecode
        self._size = size
        self._shape = shape
        self.dimensions = dimensions
        self.maskandscale = maskandscale

        self._attributes = attributes or {}
        for k, v in self._attributes.items():
            self.__dict__[k] = v

    def __setattr__(self, attr, value):
        # Store user defined attributes in a separate dict,
        # so we can save them to file later.
        try:
            self._attributes[attr] = value
        except AttributeError:
            pass
        self.__dict__[attr] = value

    def isrec(self):
        """Returns whether the variable has a record dimension or not.

        A record dimension is a dimension along which additional data could be
        easily appended in the netcdf data structure without much rewriting of
        the data file. This attribute is a read-only property of the
        `netcdf_variable`.

        """
        return bool(self.data.shape) and not self._shape[0]
    isrec = property(isrec)

    def shape(self):
        """Returns the shape tuple of the data variable.

        This is a read-only attribute and can not be modified in the
        same manner of other numpy arrays.
        """
        return self.data.shape
    shape = property(shape)

    def getValue(self):
        """
        Retrieve a scalar value from a `netcdf_variable` of length one.

        Raises
        ------
        ValueError
            If the netcdf variable is an array of length greater than one,
            this exception will be raised.

        """
        return self.data.item()

    def assignValue(self, value):
        """
        Assign a scalar value to a `netcdf_variable` of length one.

        Parameters
        ----------
        value : scalar
            Scalar value (of compatible type) to assign to a length-one netcdf
            variable. This value will be written to file.

        Raises
        ------
        ValueError
            If the input is not a scalar, or if the destination is not a length-one
            netcdf variable.

        """
        if not self.data.flags.writeable:
            # Work-around for a bug in NumPy.  Calling itemset() on a read-only
            # memory-mapped array causes a seg. fault.
            # See NumPy ticket #1622, and SciPy ticket #1202.
            # This check for `writeable` can be removed when the oldest version
            # of NumPy still supported by scipy contains the fix for #1622.
            raise RuntimeError("variable is not writeable")

        self.data[:] = value

    def typecode(self):
        """
        Return the typecode of the variable.

        Returns
        -------
        typecode : char
            The character typecode of the variable (e.g., 'i' for int).

        """
        return self._typecode

    def itemsize(self):
        """
        Return the itemsize of the variable.

        Returns
        -------
        itemsize : int
            The element size of the variable (e.g., 8 for float64).

        """
        return self._size

    def __getitem__(self, index):
        if not self.maskandscale:
            return self.data[index]

        data = self.data[index].copy()
        missing_value = self._get_missing_value()
        data = self._apply_missing_value(data, missing_value)
        scale_factor = self._attributes.get('scale_factor')
        add_offset = self._attributes.get('add_offset')
        if add_offset is not None or scale_factor is not None:
            data = data.astype(np.float64)
        if scale_factor is not None:
            data = data * scale_factor
        if add_offset is not None:
            data += add_offset

        return data

    def __setitem__(self, index, data):
        if self.maskandscale:
            missing_value = (
                    self._get_missing_value() or
                    getattr(data, 'fill_value', 999999))
            self._attributes.setdefault('missing_value', missing_value)
            self._attributes.setdefault('_FillValue', missing_value)
            data = ((data - self._attributes.get('add_offset', 0.0)) /
                    self._attributes.get('scale_factor', 1.0))
            data = np.ma.asarray(data).filled(missing_value)
            if self._typecode not in 'fd' and data.dtype.kind == 'f':
                data = np.round(data)

        # Expand data for record vars?
        if self.isrec:
            if isinstance(index, tuple):
                rec_index = index[0]
            else:
                rec_index = index
            if isinstance(rec_index, slice):
                recs = (rec_index.start or 0) + len(data)
            else:
                recs = rec_index + 1
            if recs > len(self.data):
                shape = (recs,) + self._shape[1:]
                # Resize in-place does not always work since
                # the array might not be single-segment
                try:
                    self.data.resize(shape)
                except ValueError:
                    dtype = self.data.dtype
                    self.__dict__['data'] = np.resize(self.data, shape).astype(dtype)
        self.data[index] = data

    def _default_encoded_fill_value(self):
        """
        The default encoded fill-value for this Variable's data type.
        """
        nc_type = REVERSE[self.typecode(), self.itemsize()]
        return FILLMAP[nc_type]

    def _get_encoded_fill_value(self):
        """
        Returns the encoded fill value for this variable as bytes.

        This is taken from either the _FillValue attribute, or the default fill
        value for this variable's data type.
        """
        if '_FillValue' in self._attributes:
            fill_value = np.array(self._attributes['_FillValue'],
                                  dtype=self.data.dtype).tobytes()
            if len(fill_value) == self.itemsize():
                return fill_value
            else:
                return self._default_encoded_fill_value()
        else:
            return self._default_encoded_fill_value()

    def _get_missing_value(self):
        """
        Returns the value denoting "no data" for this variable.

        If this variable does not have a missing/fill value, returns None.

        If both _FillValue and missing_value are given, give precedence to
        _FillValue. The netCDF standard gives special meaning to _FillValue;
        missing_value is  just used for compatibility with old datasets.
        """

        if '_FillValue' in self._attributes:
            missing_value = self._attributes['_FillValue']
        elif 'missing_value' in self._attributes:
            missing_value = self._attributes['missing_value']
        else:
            missing_value = None

        return missing_value

    @staticmethod
    def _apply_missing_value(data, missing_value):
        """
        Applies the given missing value to the data array.

        Returns a numpy.ma array, with any value equal to missing_value masked
        out (unless missing_value is None, in which case the original array is
        returned).
        """

        if missing_value is None:
            newdata = data
        else:
            try:
                missing_value_isnan = np.isnan(missing_value)
            except (TypeError, NotImplementedError):
                # some data types (e.g., characters) cannot be tested for NaN
                missing_value_isnan = False

            if missing_value_isnan:
                mymask = np.isnan(data)
            else:
                mymask = (data == missing_value)

            newdata = np.ma.masked_where(mymask, data)

        return newdata


NetCDFFile = netcdf_file
NetCDFVariable = netcdf_variable
<|endoftext|>"""
Module to read / write wav files using NumPy arrays

Functions
---------
`read`: Return the sample rate (in samples/sec) and data from a WAV file.

`write`: Write a NumPy array as a WAV file.

"""
import io
import sys
import numpy as np
import struct
import warnings
from enum import IntEnum


__all__ = [
    'WavFileWarning',
    'read',
    'write'
]


class WavFileWarning(UserWarning):
    pass


class WAVE_FORMAT(IntEnum):
    """
    WAVE form wFormatTag IDs

    Complete list is in mmreg.h in Windows 10 SDK.  ALAC and OPUS are the
    newest additions, in v10.0.14393 2016-07
    """
    UNKNOWN = 0x0000
    PCM = 0x0001
    ADPCM = 0x0002
    IEEE_FLOAT = 0x0003
    VSELP = 0x0004
    IBM_CVSD = 0x0005
    ALAW = 0x0006
    MULAW = 0x0007
    DTS = 0x0008
    DRM = 0x0009
    WMAVOICE9 = 0x000A
    WMAVOICE10 = 0x000B
    OKI_ADPCM = 0x0010
    DVI_ADPCM = 0x0011
    IMA_ADPCM = 0x0011  # Duplicate
    MEDIASPACE_ADPCM = 0x0012
    SIERRA_ADPCM = 0x0013
    G723_ADPCM = 0x0014
    DIGISTD = 0x0015
    DIGIFIX = 0x0016
    DIALOGIC_OKI_ADPCM = 0x0017
    MEDIAVISION_ADPCM = 0x0018
    CU_CODEC = 0x0019
    HP_DYN_VOICE = 0x001A
    YAMAHA_ADPCM = 0x0020
    SONARC = 0x0021
    DSPGROUP_TRUESPEECH = 0x0022
    ECHOSC1 = 0x0023
    AUDIOFILE_AF36 = 0x0024
    APTX = 0x0025
    AUDIOFILE_AF10 = 0x0026
    PROSODY_1612 = 0x0027
    LRC = 0x0028
    DOLBY_AC2 = 0x0030
    GSM610 = 0x0031
    MSNAUDIO = 0x0032
    ANTEX_ADPCME = 0x0033
    CONTROL_RES_VQLPC = 0x0034
    DIGIREAL = 0x0035
    DIGIADPCM = 0x0036
    CONTROL_RES_CR10 = 0x0037
    NMS_VBXADPCM = 0x0038
    CS_IMAADPCM = 0x0039
    ECHOSC3 = 0x003A
    ROCKWELL_ADPCM = 0x003B
    ROCKWELL_DIGITALK = 0x003C
    XEBEC = 0x003D
    G721_ADPCM = 0x0040
    G728_CELP = 0x0041
    MSG723 = 0x0042
    INTEL_G723_1 = 0x0043
    INTEL_G729 = 0x0044
    SHARP_G726 = 0x0045
    MPEG = 0x0050
    RT24 = 0x0052
    PAC = 0x0053
    MPEGLAYER3 = 0x0055
    LUCENT_G723 = 0x0059
    CIRRUS = 0x0060
    ESPCM = 0x0061
    VOXWARE = 0x0062
    CANOPUS_ATRAC = 0x0063
    G726_ADPCM = 0x0064
    G722_ADPCM = 0x0065
    DSAT = 0x0066
    DSAT_DISPLAY = 0x0067
    VOXWARE_BYTE_ALIGNED = 0x0069
    VOXWARE_AC8 = 0x0070
    VOXWARE_AC10 = 0x0071
    VOXWARE_AC16 = 0x0072
    VOXWARE_AC20 = 0x0073
    VOXWARE_RT24 = 0x0074
    VOXWARE_RT29 = 0x0075
    VOXWARE_RT29HW = 0x0076
    VOXWARE_VR12 = 0x0077
    VOXWARE_VR18 = 0x0078
    VOXWARE_TQ40 = 0x0079
    VOXWARE_SC3 = 0x007A
    VOXWARE_SC3_1 = 0x007B
    SOFTSOUND = 0x0080
    VOXWARE_TQ60 = 0x0081
    MSRT24 = 0x0082
    G729A = 0x0083
    MVI_MVI2 = 0x0084
    DF_G726 = 0x0085
    DF_GSM610 = 0x0086
    ISIAUDIO = 0x0088
    ONLIVE = 0x0089
    MULTITUDE_FT_SX20 = 0x008A
    INFOCOM_ITS_G721_ADPCM = 0x008B
    CONVEDIA_G729 = 0x008C
    CONGRUENCY = 0x008D
    SBC24 = 0x0091
    DOLBY_AC3_SPDIF = 0x0092
    MEDIASONIC_G723 = 0x0093
    PROSODY_8KBPS = 0x0094
    ZYXEL_ADPCM = 0x0097
    PHILIPS_LPCBB = 0x0098
    PACKED = 0x0099
    MALDEN_PHONYTALK = 0x00A0
    RACAL_RECORDER_GSM = 0x00A1
    RACAL_RECORDER_G720_A = 0x00A2
    RACAL_RECORDER_G723_1 = 0x00A3
    RACAL_RECORDER_TETRA_ACELP = 0x00A4
    NEC_AAC = 0x00B0
    RAW_AAC1 = 0x00FF
    RHETOREX_ADPCM = 0x0100
    IRAT = 0x0101
    VIVO_G723 = 0x0111
    VIVO_SIREN = 0x0112
    PHILIPS_CELP = 0x0120
    PHILIPS_GRUNDIG = 0x0121
    DIGITAL_G723 = 0x0123
    SANYO_LD_ADPCM = 0x0125
    SIPROLAB_ACEPLNET = 0x0130
    SIPROLAB_ACELP4800 = 0x0131
    SIPROLAB_ACELP8V3 = 0x0132
    SIPROLAB_G729 = 0x0133
    SIPROLAB_G729A = 0x0134
    SIPROLAB_KELVIN = 0x0135
    VOICEAGE_AMR = 0x0136
    G726ADPCM = 0x0140
    DICTAPHONE_CELP68 = 0x0141
    DICTAPHONE_CELP54 = 0x0142
    QUALCOMM_PUREVOICE = 0x0150
    QUALCOMM_HALFRATE = 0x0151
    TUBGSM = 0x0155
    MSAUDIO1 = 0x0160
    WMAUDIO2 = 0x0161
    WMAUDIO3 = 0x0162
    WMAUDIO_LOSSLESS = 0x0163
    WMASPDIF = 0x0164
    UNISYS_NAP_ADPCM = 0x0170
    UNISYS_NAP_ULAW = 0x0171
    UNISYS_NAP_ALAW = 0x0172
    UNISYS_NAP_16K = 0x0173
    SYCOM_ACM_SYC008 = 0x0174
    SYCOM_ACM_SYC701_G726L = 0x0175
    SYCOM_ACM_SYC701_CELP54 = 0x0176
    SYCOM_ACM_SYC701_CELP68 = 0x0177
    KNOWLEDGE_ADVENTURE_ADPCM = 0x0178
    FRAUNHOFER_IIS_MPEG2_AAC = 0x0180
    DTS_DS = 0x0190
    CREATIVE_ADPCM = 0x0200
    CREATIVE_FASTSPEECH8 = 0x0202
    CREATIVE_FASTSPEECH10 = 0x0203
    UHER_ADPCM = 0x0210
    ULEAD_DV_AUDIO = 0x0215
    ULEAD_DV_AUDIO_1 = 0x0216
    QUARTERDECK = 0x0220
    ILINK_VC = 0x0230
    RAW_SPORT = 0x0240
    ESST_AC3 = 0x0241
    GENERIC_PASSTHRU = 0x0249
    IPI_HSX = 0x0250
    IPI_RPELP = 0x0251
    CS2 = 0x0260
    SONY_SCX = 0x0270
    SONY_SCY = 0x0271
    SONY_ATRAC3 = 0x0272
    SONY_SPC = 0x0273
    TELUM_AUDIO = 0x0280
    TELUM_IA_AUDIO = 0x0281
    NORCOM_VOICE_SYSTEMS_ADPCM = 0x0285
    FM_TOWNS_SND = 0x0300
    MICRONAS = 0x0350
    MICRONAS_CELP833 = 0x0351
    BTV_DIGITAL = 0x0400
    INTEL_MUSIC_CODER = 0x0401
    INDEO_AUDIO = 0x0402
    QDESIGN_MUSIC = 0x0450
    ON2_VP7_AUDIO = 0x0500
    ON2_VP6_AUDIO = 0x0501
    VME_VMPCM = 0x0680
    TPC = 0x0681
    LIGHTWAVE_LOSSLESS = 0x08AE
    OLIGSM = 0x1000
    OLIADPCM = 0x1001
    OLICELP = 0x1002
    OLISBC = 0x1003
    OLIOPR = 0x1004
    LH_CODEC = 0x1100
    LH_CODEC_CELP = 0x1101
    LH_CODEC_SBC8 = 0x1102
    LH_CODEC_SBC12 = 0x1103
    LH_CODEC_SBC16 = 0x1104
    NORRIS = 0x1400
    ISIAUDIO_2 = 0x1401
    SOUNDSPACE_MUSICOMPRESS = 0x1500
    MPEG_ADTS_AAC = 0x1600
    MPEG_RAW_AAC = 0x1601
    MPEG_LOAS = 0x1602
    NOKIA_MPEG_ADTS_AAC = 0x1608
    NOKIA_MPEG_RAW_AAC = 0x1609
    VODAFONE_MPEG_ADTS_AAC = 0x160A
    VODAFONE_MPEG_RAW_AAC = 0x160B
    MPEG_HEAAC = 0x1610
    VOXWARE_RT24_SPEECH = 0x181C
    SONICFOUNDRY_LOSSLESS = 0x1971
    INNINGS_TELECOM_ADPCM = 0x1979
    LUCENT_SX8300P = 0x1C07
    LUCENT_SX5363S = 0x1C0C
    CUSEEME = 0x1F03
    NTCSOFT_ALF2CM_ACM = 0x1FC4
    DVM = 0x2000
    DTS2 = 0x2001
    MAKEAVIS = 0x3313
    DIVIO_MPEG4_AAC = 0x4143
    NOKIA_ADAPTIVE_MULTIRATE = 0x4201
    DIVIO_G726 = 0x4243
    LEAD_SPEECH = 0x434C
    LEAD_VORBIS = 0x564C
    WAVPACK_AUDIO = 0x5756
    OGG_VORBIS_MODE_1 = 0x674F
    OGG_VORBIS_MODE_2 = 0x6750
    OGG_VORBIS_MODE_3 = 0x6751
    OGG_VORBIS_MODE_1_PLUS = 0x676F
    OGG_VORBIS_MODE_2_PLUS = 0x6770
    OGG_VORBIS_MODE_3_PLUS = 0x6771
    ALAC = 0x6C61
    _3COM_NBX = 0x7000  # Can't have leading digit
    OPUS = 0x704F
    FAAD_AAC = 0x706D
    AMR_NB = 0x7361
    AMR_WB = 0x7362
    AMR_WP = 0x7363
    GSM_AMR_CBR = 0x7A21
    GSM_AMR_VBR_SID = 0x7A22
    COMVERSE_INFOSYS_G723_1 = 0xA100
    COMVERSE_INFOSYS_AVQSBC = 0xA101
    COMVERSE_INFOSYS_SBC = 0xA102
    SYMBOL_G729_A = 0xA103
    VOICEAGE_AMR_WB = 0xA104
    INGENIENT_G726 = 0xA105
    MPEG4_AAC = 0xA106
    ENCORE_G726 = 0xA107
    ZOLL_ASAO = 0xA108
    SPEEX_VOICE = 0xA109
    VIANIX_MASC = 0xA10A
    WM9_SPECTRUM_ANALYZER = 0xA10B
    WMF_SPECTRUM_ANAYZER = 0xA10C
    GSM_610 = 0xA10D
    GSM_620 = 0xA10E
    GSM_660 = 0xA10F
    GSM_690 = 0xA110
    GSM_ADAPTIVE_MULTIRATE_WB = 0xA111
    POLYCOM_G722 = 0xA112
    POLYCOM_G728 = 0xA113
    POLYCOM_G729_A = 0xA114
    POLYCOM_SIREN = 0xA115
    GLOBAL_IP_ILBC = 0xA116
    RADIOTIME_TIME_SHIFT_RADIO = 0xA117
    NICE_ACA = 0xA118
    NICE_ADPCM = 0xA119
    VOCORD_G721 = 0xA11A
    VOCORD_G726 = 0xA11B
    VOCORD_G722_1 = 0xA11C
    VOCORD_G728 = 0xA11D
    VOCORD_G729 = 0xA11E
    VOCORD_G729_A = 0xA11F
    VOCORD_G723_1 = 0xA120
    VOCORD_LBC = 0xA121
    NICE_G728 = 0xA122
    FRACE_TELECOM_G729 = 0xA123
    CODIAN = 0xA124
    FLAC = 0xF1AC
    EXTENSIBLE = 0xFFFE
    DEVELOPMENT = 0xFFFF


KNOWN_WAVE_FORMATS = {WAVE_FORMAT.PCM, WAVE_FORMAT.IEEE_FLOAT}


def _raise_bad_format(format_tag):
    try:
        format_name = WAVE_FORMAT(format_tag).name
    except ValueError:
        format_name = f'{format_tag:#06x}'
    raise ValueError(f"Unknown wave file format: {format_name}. Supported "
                     "formats: " +
                     ', '.join(x.name for x in KNOWN_WAVE_FORMATS))


def _read_fmt_chunk(fid, is_big_endian):
    """
    Returns
    -------
    size : int
        size of format subchunk in bytes (minus 8 for "fmt " and itself)
    format_tag : int
        PCM, float, or compressed format
    channels : int
        number of channels
    fs : int
        sampling frequency in samples per second
    bytes_per_second : int
        overall byte rate for the file
    block_align : int
        bytes per sample, including all channels
    bit_depth : int
        bits per sample

    Notes
    -----
    Assumes file pointer is immediately after the 'fmt ' id
    """
    if is_big_endian:
        fmt = '>'
    else:
        fmt = '<'

    size = struct.unpack(fmt+'I', fid.read(4))[0]

    if size < 16:
        raise ValueError("Binary structure of wave file is not compliant")

    res = struct.unpack(fmt+'HHIIHH', fid.read(16))
    bytes_read = 16

    format_tag, channels, fs, bytes_per_second, block_align, bit_depth = res

    if format_tag == WAVE_FORMAT.EXTENSIBLE and size >= (16+2):
        ext_chunk_size = struct.unpack(fmt+'H', fid.read(2))[0]
        bytes_read += 2
        if ext_chunk_size >= 22:
            extensible_chunk_data = fid.read(22)
            bytes_read += 22
            raw_guid = extensible_chunk_data[2+4:2+4+16]
            # GUID template {XXXXXXXX-0000-0010-8000-00AA00389B71} (RFC-2361)
            # MS GUID byte order: first three groups are native byte order,
            # rest is Big Endian
            if is_big_endian:
                tail = b'\x00\x00\x00\x10\x80\x00\x00\xAA\x00\x38\x9B\x71'
            else:
                tail = b'\x00\x00\x10\x00\x80\x00\x00\xAA\x00\x38\x9B\x71'
            if raw_guid.endswith(tail):
                format_tag = struct.unpack(fmt+'I', raw_guid[:4])[0]
        else:
            raise ValueError("Binary structure of wave file is not compliant")

    if format_tag not in KNOWN_WAVE_FORMATS:
