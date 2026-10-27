    arrays. The vectorized function evaluates `pyfunc` over successive tuples
    of the input arrays like the python map function, except it uses the
    broadcasting rules of numpy.

    The data type of the output of `vectorized` is determined by calling
    the function with the first element of the input.  This can be avoided
    by specifying the `otypes` argument.

    Parameters
    ----------
    pyfunc : callable, optional
        A python function or method.
        Can be omitted to produce a decorator with keyword arguments.
    otypes : str or list of dtypes, optional
        The output data type. It must be specified as either a string of
        typecode characters or a list of data type specifiers. There should
        be one data type specifier for each output.
    doc : str, optional
        The docstring for the function. If None, the docstring will be the
        ``pyfunc.__doc__``.
    excluded : set, optional
        Set of strings or integers representing the positional or keyword
        arguments for which the function will not be vectorized. These will be
        passed directly to `pyfunc` unmodified.

    cache : bool, optional
        If `True`, then cache the first function call that determines the number
        of outputs if `otypes` is not provided.

    signature : string, optional
        Generalized universal function signature, e.g., ``(m,n),(n)->(m)`` for
        vectorized matrix-vector multiplication. If provided, ``pyfunc`` will
        be called with (and expected to return) arrays with shapes given by the
        size of corresponding core dimensions. By default, ``pyfunc`` is
        assumed to take scalars as input and output.

    Returns
    -------
    out : callable
        A vectorized function if ``pyfunc`` was provided,
        a decorator otherwise.

    See Also
    --------
    frompyfunc : Takes an arbitrary Python function and returns a ufunc

    Notes
    -----
    The `vectorize` function is provided primarily for convenience, not for
    performance. The implementation is essentially a for loop.

    If `otypes` is not specified, then a call to the function with the
    first argument will be used to determine the number of outputs.  The
    results of this call will be cached if `cache` is `True` to prevent
    calling the function twice.  However, to implement the cache, the
    original function must be wrapped which will slow down subsequent
    calls, so only do this if your function is expensive.

    The new keyword argument interface and `excluded` argument support
    further degrades performance.

    References
    ----------
    .. [1] :doc:`/reference/c-api/generalized-ufuncs`

    Examples
    --------
    >>> import numpy as np
    >>> def myfunc(a, b):
    ...     "Return a-b if a>b, otherwise return a+b"
    ...     if a > b:
    ...         return a - b
    ...     else:
    ...         return a + b

    >>> vfunc = np.vectorize(myfunc)
    >>> vfunc([1, 2, 3, 4], 2)
    array([3, 4, 1, 2])

    The docstring is taken from the input function to `vectorize` unless it
    is specified:

    >>> vfunc.__doc__
    'Return a-b if a>b, otherwise return a+b'
    >>> vfunc = np.vectorize(myfunc, doc='Vectorized `myfunc`')
    >>> vfunc.__doc__
    'Vectorized `myfunc`'

    The output type is determined by evaluating the first element of the input,
    unless it is specified:

    >>> out = vfunc([1, 2, 3, 4], 2)
    >>> type(out[0])
    <class 'numpy.int64'>
    >>> vfunc = np.vectorize(myfunc, otypes=[float])
    >>> out = vfunc([1, 2, 3, 4], 2)
    >>> type(out[0])
    <class 'numpy.float64'>

    The `excluded` argument can be used to prevent vectorizing over certain
    arguments.  This can be useful for array-like arguments of a fixed length
    such as the coefficients for a polynomial as in `polyval`:

    >>> def mypolyval(p, x):
    ...     _p = list(p)
    ...     res = _p.pop(0)
    ...     while _p:
    ...         res = res*x + _p.pop(0)
    ...     return res

    Here, we exclude the zeroth argument from vectorization whether it is
    passed by position or keyword.

    >>> vpolyval = np.vectorize(mypolyval, excluded={0, 'p'})
    >>> vpolyval([1, 2, 3], x=[0, 1])
    array([3, 6])
    >>> vpolyval(p=[1, 2, 3], x=[0, 1])
    array([3, 6])

    The `signature` argument allows for vectorizing functions that act on
    non-scalar arrays of fixed length. For example, you can use it for a
    vectorized calculation of Pearson correlation coefficient and its p-value:

    >>> import scipy.stats
    >>> pearsonr = np.vectorize(scipy.stats.pearsonr,
    ...                 signature='(n),(n)->(),()')
    >>> pearsonr([[0, 1, 2, 3]], [[1, 2, 3, 4], [4, 3, 2, 1]])
    (array([ 1., -1.]), array([ 0.,  0.]))

    Or for a vectorized convolution:

    >>> convolve = np.vectorize(np.convolve, signature='(n),(m)->(k)')
    >>> convolve(np.eye(4), [1, 2, 1])
    array([[1., 2., 1., 0., 0., 0.],
           [0., 1., 2., 1., 0., 0.],
           [0., 0., 1., 2., 1., 0.],
           [0., 0., 0., 1., 2., 1.]])

    Decorator syntax is supported.  The decorator can be called as
    a function to provide keyword arguments:

    >>> @np.vectorize
    ... def identity(x):
    ...     return x
    ...
    >>> identity([0, 1, 2])
    array([0, 1, 2])
    >>> @np.vectorize(otypes=[float])
    ... def as_float(x):
    ...     return x
    ...
    >>> as_float([0, 1, 2])
    array([0., 1., 2.])
    """
    def __init__(self, pyfunc=np._NoValue, otypes=None, doc=None,
                 excluded=None, cache=False, signature=None):

        if (pyfunc != np._NoValue) and (not callable(pyfunc)):
            #Splitting the error message to keep
            #the length below 79 characters.
            part1 = "When used as a decorator, "
            part2 = "only accepts keyword arguments."
            raise TypeError(part1 + part2)

        self.pyfunc = pyfunc
        self.cache = cache
        self.signature = signature
        if pyfunc != np._NoValue and hasattr(pyfunc, '__name__'):
            self.__name__ = pyfunc.__name__

        self._ufunc = {}    # Caching to improve default performance
        self._doc = None
        self.__doc__ = doc
        if doc is None and hasattr(pyfunc, '__doc__'):
            self.__doc__ = pyfunc.__doc__
        else:
            self._doc = doc

        if isinstance(otypes, str):
            for char in otypes:
                if char not in typecodes['All']:
                    raise ValueError("Invalid otype specified: %s" % (char,))
        elif iterable(otypes):
            otypes = [_get_vectorize_dtype(_nx.dtype(x)) for x in otypes]
        elif otypes is not None:
            raise ValueError("Invalid otype specification")
        self.otypes = otypes

        # Excluded variable support
        if excluded is None:
            excluded = set()
        self.excluded = set(excluded)

        if signature is not None:
            self._in_and_out_core_dims = _parse_gufunc_signature(signature)
        else:
            self._in_and_out_core_dims = None

    def _init_stage_2(self, pyfunc, *args, **kwargs):
        self.__name__ = pyfunc.__name__
        self.pyfunc = pyfunc
        if self._doc is None:
            self.__doc__ = pyfunc.__doc__
        else:
            self.__doc__ = self._doc

    def _call_as_normal(self, *args, **kwargs):
        """
        Return arrays with the results of `pyfunc` broadcast (vectorized) over
        `args` and `kwargs` not in `excluded`.
        """
        excluded = self.excluded
        if not kwargs and not excluded:
            func = self.pyfunc
            vargs = args
        else:
            # The wrapper accepts only positional arguments: we use `names` and
            # `inds` to mutate `the_args` and `kwargs` to pass to the original
            # function.
            nargs = len(args)

            names = [_n for _n in kwargs if _n not in excluded]
            inds = [_i for _i in range(nargs) if _i not in excluded]
            the_args = list(args)

            def func(*vargs):
                for _n, _i in enumerate(inds):
                    the_args[_i] = vargs[_n]
                kwargs.update(zip(names, vargs[len(inds):]))
                return self.pyfunc(*the_args, **kwargs)

            vargs = [args[_i] for _i in inds]
            vargs.extend([kwargs[_n] for _n in names])

        return self._vectorize_call(func=func, args=vargs)

    def __call__(self, *args, **kwargs):
        if self.pyfunc is np._NoValue:
            self._init_stage_2(*args, **kwargs)
            return self

        return self._call_as_normal(*args, **kwargs)

    def _get_ufunc_and_otypes(self, func, args):
        """Return (ufunc, otypes)."""
        # frompyfunc will fail if args is empty
        if not args:
            raise ValueError('args can not be empty')

        if self.otypes is not None:
            otypes = self.otypes

            # self._ufunc is a dictionary whose keys are the number of
            # arguments (i.e. len(args)) and whose values are ufuncs created
            # by frompyfunc. len(args) can be different for different calls if
            # self.pyfunc has parameters with default values.  We only use the
            # cache when func is self.pyfunc, which occurs when the call uses
            # only positional arguments and no arguments are excluded.

            nin = len(args)
            nout = len(self.otypes)
            if func is not self.pyfunc or nin not in self._ufunc:
                ufunc = frompyfunc(func, nin, nout)
            else:
                ufunc = None  # We'll get it from self._ufunc
            if func is self.pyfunc:
                ufunc = self._ufunc.setdefault(nin, ufunc)
        else:
            # Get number of outputs and output types by calling the function on
            # the first entries of args.  We also cache the result to prevent
            # the subsequent call when the ufunc is evaluated.
            # Assumes that ufunc first evaluates the 0th elements in the input
            # arrays (the input values are not checked to ensure this)
            args = [asarray(arg) for arg in args]
            if builtins.any(arg.size == 0 for arg in args):
                raise ValueError('cannot call `vectorize` on size 0 inputs '
                                 'unless `otypes` is set')

            inputs = [arg.flat[0] for arg in args]
            outputs = func(*inputs)

            # Performance note: profiling indicates that -- for simple
            # functions at least -- this wrapping can almost double the
            # execution time.
            # Hence we make it optional.
            if self.cache:
                _cache = [outputs]

                def _func(*vargs):
                    if _cache:
                        return _cache.pop()
                    else:
                        return func(*vargs)
            else:
                _func = func

            if isinstance(outputs, tuple):
                nout = len(outputs)
            else:
                nout = 1
                outputs = (outputs,)

            otypes = ''.join([asarray(outputs[_k]).dtype.char
                              for _k in range(nout)])

            # Performance note: profiling indicates that creating the ufunc is
            # not a significant cost compared with wrapping so it seems not
            # worth trying to cache this.
            ufunc = frompyfunc(_func, len(args), nout)

        return ufunc, otypes

    def _vectorize_call(self, func, args):
        """Vectorized call to `func` over positional `args`."""
        if self.signature is not None:
            res = self._vectorize_call_with_signature(func, args)
        elif not args:
            res = func()
        else:
            ufunc, otypes = self._get_ufunc_and_otypes(func=func, args=args)

            # Convert args to object arrays first
            inputs = [asanyarray(a, dtype=object) for a in args]

            outputs = ufunc(*inputs)

            if ufunc.nout == 1:
                res = asanyarray(outputs, dtype=otypes[0])
            else:
                res = tuple([asanyarray(x, dtype=t)
                             for x, t in zip(outputs, otypes)])
        return res

    def _vectorize_call_with_signature(self, func, args):
        """Vectorized call over positional arguments with a signature."""
        input_core_dims, output_core_dims = self._in_and_out_core_dims

        if len(args) != len(input_core_dims):
            raise TypeError('wrong number of positional arguments: '
                            'expected %r, got %r'
                            % (len(input_core_dims), len(args)))
        args = tuple(asanyarray(arg) for arg in args)

        broadcast_shape, dim_sizes = _parse_input_dimensions(
            args, input_core_dims)
        input_shapes = _calculate_shapes(broadcast_shape, dim_sizes,
                                         input_core_dims)
        args = [np.broadcast_to(arg, shape, subok=True)
                for arg, shape in zip(args, input_shapes)]

        outputs = None
        otypes = self.otypes
        nout = len(output_core_dims)

        for index in np.ndindex(*broadcast_shape):
            results = func(*(arg[index] for arg in args))

            n_results = len(results) if isinstance(results, tuple) else 1

            if nout != n_results:
                raise ValueError(
                    'wrong number of outputs from pyfunc: expected %r, got %r'
                    % (nout, n_results))

            if nout == 1:
                results = (results,)

            if outputs is None:
                for result, core_dims in zip(results, output_core_dims):
                    _update_dim_sizes(dim_sizes, result, core_dims)

                outputs = _create_arrays(broadcast_shape, dim_sizes,
                                         output_core_dims, otypes, results)

            for output, result in zip(outputs, results):
                output[index] = result

        if outputs is None:
            # did not call the function even once
            if otypes is None:
                raise ValueError('cannot call `vectorize` on size 0 inputs '
                                 'unless `otypes` is set')
            if builtins.any(dim not in dim_sizes
                            for dims in output_core_dims
                            for dim in dims):
                raise ValueError('cannot call `vectorize` with a signature '
                                 'including new output dimensions on size 0 '
                                 'inputs')
            outputs = _create_arrays(broadcast_shape, dim_sizes,
                                     output_core_dims, otypes)

        return outputs[0] if nout == 1 else outputs


def _cov_dispatcher(m, y=None, rowvar=None, bias=None, ddof=None,
                    fweights=None, aweights=None, *, dtype=None):
    return (m, y, fweights, aweights)


@array_function_dispatch(_cov_dispatcher)
def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None,
        aweights=None, *, dtype=None):
    """
    Estimate a covariance matrix, given data and weights.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, :math:`X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element :math:`C_{ij}` is the covariance of
    :math:`x_i` and :math:`x_j`. The element :math:`C_{ii}` is the variance
    of :math:`x_i`.

    See the notes for an outline of the algorithm.

    Parameters
    ----------
    m : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `m` represents a variable, and each column a single
        observation of all those variables. Also see `rowvar` below.
    y : array_like, optional
        An additional set of variables and observations. `y` has the same form
        as that of `m`.
    rowvar : bool, optional
        If `rowvar` is True (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
    bias : bool, optional
        Default normalization (False) is by ``(N - 1)``, where ``N`` is the
        number of observations given (unbiased estimate). If `bias` is True,
        then normalization is by ``N``. These values can be overridden by using
        the keyword ``ddof`` in numpy versions >= 1.5.
    ddof : int, optional
        If not ``None`` the default value implied by `bias` is overridden.
        Note that ``ddof=1`` will return the unbiased estimate, even if both
        `fweights` and `aweights` are specified, and ``ddof=0`` will return
        the simple average. See the notes for the details. The default value
        is ``None``.
    fweights : array_like, int, optional
        1-D array of integer frequency weights; the number of times each
        observation vector should be repeated.
    aweights : array_like, optional
        1-D array of observation vector weights. These relative weights are
        typically large for observations considered "important" and smaller for
        observations considered less "important". If ``ddof=0`` the array of
        weights can be used to assign probabilities to observation vectors.
    dtype : data-type, optional
        Data-type of the result. By default, the return data-type will have
        at least `numpy.float64` precision.

        .. versionadded:: 1.20

    Returns
    -------
    out : ndarray
        The covariance matrix of the variables.

    See Also
    --------
    corrcoef : Normalized covariance matrix

    Notes
    -----
    Assume that the observations are in the columns of the observation
    array `m` and let ``f = fweights`` and ``a = aweights`` for brevity. The
    steps to compute the weighted covariance are as follows::

        >>> m = np.arange(10, dtype=np.float64)
        >>> f = np.arange(10) * 2
        >>> a = np.arange(10) ** 2.
        >>> ddof = 1
        >>> w = f * a
        >>> v1 = np.sum(w)
        >>> v2 = np.sum(w * a)
        >>> m -= np.sum(m * w, axis=None, keepdims=True) / v1
        >>> cov = np.dot(m * w, m.T) * v1 / (v1**2 - ddof * v2)

    Note that when ``a == 1``, the normalization factor
    ``v1 / (v1**2 - ddof * v2)`` goes over to ``1 / (np.sum(f) - ddof)``
    as it should.

    Examples
    --------
    >>> import numpy as np

    Consider two variables, :math:`x_0` and :math:`x_1`, which
    correlate perfectly, but in opposite directions:

    >>> x = np.array([[0, 2], [1, 1], [2, 0]]).T
    >>> x
    array([[0, 1, 2],
           [2, 1, 0]])

    Note how :math:`x_0` increases while :math:`x_1` decreases. The covariance
    matrix shows this clearly:

    >>> np.cov(x)
    array([[ 1., -1.],
           [-1.,  1.]])

    Note that element :math:`C_{0,1}`, which shows the correlation between
    :math:`x_0` and :math:`x_1`, is negative.

    Further, note how `x` and `y` are combined:

    >>> x = [-2.1, -1,  4.3]
    >>> y = [3,  1.1,  0.12]
    >>> X = np.stack((x, y), axis=0)
    >>> np.cov(X)
    array([[11.71      , -4.286     ], # may vary
           [-4.286     ,  2.144133]])
    >>> np.cov(x, y)
    array([[11.71      , -4.286     ], # may vary
           [-4.286     ,  2.144133]])
    >>> np.cov(x)
    array(11.71)

    """
    # Check inputs
    if ddof is not None and ddof != int(ddof):
        raise ValueError(
            "ddof must be integer")

    # Handles complex arrays too
    m = np.asarray(m)
    if m.ndim > 2:
        raise ValueError("m has more than 2 dimensions")

    if y is not None:
        y = np.asarray(y)
        if y.ndim > 2:
            raise ValueError("y has more than 2 dimensions")

    if dtype is None:
        if y is None:
            dtype = np.result_type(m, np.float64)
        else:
            dtype = np.result_type(m, y, np.float64)

    X = array(m, ndmin=2, dtype=dtype)
    if not rowvar and m.ndim != 1:
        X = X.T
    if X.shape[0] == 0:
        return np.array([]).reshape(0, 0)
    if y is not None:
        y = array(y, copy=None, ndmin=2, dtype=dtype)
        if not rowvar and y.shape[0] != 1:
            y = y.T
        X = np.concatenate((X, y), axis=0)

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    # Get the product of frequencies and weights
    w = None
    if fweights is not None:
        fweights = np.asarray(fweights, dtype=float)
        if not np.all(fweights == np.around(fweights)):
            raise TypeError(
                "fweights must be integer")
        if fweights.ndim > 1:
            raise RuntimeError(
                "cannot handle multidimensional fweights")
        if fweights.shape[0] != X.shape[1]:
            raise RuntimeError(
                "incompatible numbers of samples and fweights")
        if any(fweights < 0):
            raise ValueError(
                "fweights cannot be negative")
        w = fweights
    if aweights is not None:
        aweights = np.asarray(aweights, dtype=float)
        if aweights.ndim > 1:
            raise RuntimeError(
                "cannot handle multidimensional aweights")
        if aweights.shape[0] != X.shape[1]:
            raise RuntimeError(
                "incompatible numbers of samples and aweights")
        if any(aweights < 0):
            raise ValueError(
                "aweights cannot be negative")
        if w is None:
            w = aweights
        else:
            w *= aweights

    avg, w_sum = average(X, axis=1, weights=w, returned=True)
    w_sum = w_sum[0]

    # Determine the normalization
    if w is None:
        fact = X.shape[1] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof*sum(w*aweights)/w_sum

    if fact <= 0:
        warnings.warn("Degrees of freedom <= 0 for slice",
                      RuntimeWarning, stacklevel=2)
        fact = 0.0

    X -= avg[:, None]
    if w is None:
        X_T = X.T
    else:
        X_T = (X*w).T
    c = dot(X, X_T.conj())
    c *= np.true_divide(1, fact)
    return c.squeeze()


def _corrcoef_dispatcher(x, y=None, rowvar=None, bias=None, ddof=None, *,
                         dtype=None):
    return (x, y)


@array_function_dispatch(_corrcoef_dispatcher)
def corrcoef(x, y=None, rowvar=True, bias=np._NoValue, ddof=np._NoValue, *,
             dtype=None):
    """
    Return Pearson product-moment correlation coefficients.

    Please refer to the documentation for `cov` for more detail.  The
    relationship between the correlation coefficient matrix, `R`, and the
    covariance matrix, `C`, is

    .. math:: R_{ij} = \\frac{ C_{ij} } { \\sqrt{ C_{ii} C_{jj} } }

    The values of `R` are between -1 and 1, inclusive.

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
    ddof : _NoValue, optional
        Has no effect, do not use.

        .. deprecated:: 1.10.0
    dtype : data-type, optional
        Data-type of the result. By default, the return data-type will have
        at least `numpy.float64` precision.

        .. versionadded:: 1.20

    Returns
    -------
    R : ndarray
        The correlation coefficient matrix of the variables.

    See Also
    --------
    cov : Covariance matrix

    Notes
    -----
    Due to floating point rounding the resulting array may not be Hermitian,
    the diagonal elements may not be 1, and the elements may not satisfy the
    inequality abs(a) <= 1. The real and imaginary parts are clipped to the
    interval [-1,  1] in an attempt to improve on that situation but is not
    much help in the complex case.

    This function accepts but discards arguments `bias` and `ddof`.  This is
    for backwards compatibility with previous versions of this function.  These
    arguments had no effect on the return values of the function and can be
    safely ignored in this and previous versions of numpy.

    Examples
    --------
    >>> import numpy as np

    In this example we generate two random arrays, ``xarr`` and ``yarr``, and
    compute the row-wise and column-wise Pearson correlation coefficients,
    ``R``. Since ``rowvar`` is  true by  default, we first find the row-wise
    Pearson correlation coefficients between the variables of ``xarr``.

    >>> import numpy as np
    >>> rng = np.random.default_rng(seed=42)
    >>> xarr = rng.random((3, 3))
    >>> xarr
    array([[0.77395605, 0.43887844, 0.85859792],
           [0.69736803, 0.09417735, 0.97562235],
           [0.7611397 , 0.78606431, 0.12811363]])
    >>> R1 = np.corrcoef(xarr)
    >>> R1
    array([[ 1.        ,  0.99256089, -0.68080986],
           [ 0.99256089,  1.        , -0.76492172],
           [-0.68080986, -0.76492172,  1.        ]])

    If we add another set of variables and observations ``yarr``, we can
    compute the row-wise Pearson correlation coefficients between the
    variables in ``xarr`` and ``yarr``.

    >>> yarr = rng.random((3, 3))
    >>> yarr
    array([[0.45038594, 0.37079802, 0.92676499],
           [0.64386512, 0.82276161, 0.4434142 ],
           [0.22723872, 0.55458479, 0.06381726]])
    >>> R2 = np.corrcoef(xarr, yarr)
    >>> R2
    array([[ 1.        ,  0.99256089, -0.68080986,  0.75008178, -0.934284  ,
            -0.99004057],
           [ 0.99256089,  1.        , -0.76492172,  0.82502011, -0.97074098,
            -0.99981569],
           [-0.68080986, -0.76492172,  1.        , -0.99507202,  0.89721355,
             0.77714685],
           [ 0.75008178,  0.82502011, -0.99507202,  1.        , -0.93657855,
            -0.83571711],
           [-0.934284  , -0.97074098,  0.89721355, -0.93657855,  1.        ,
             0.97517215],
           [-0.99004057, -0.99981569,  0.77714685, -0.83571711,  0.97517215,
             1.        ]])

    Finally if we use the option ``rowvar=False``, the columns are now
    being treated as the variables and we will find the column-wise Pearson
    correlation coefficients between variables in ``xarr`` and ``yarr``.

    >>> R3 = np.corrcoef(xarr, yarr, rowvar=False)
    >>> R3
    array([[ 1.        ,  0.77598074, -0.47458546, -0.75078643, -0.9665554 ,
             0.22423734],
           [ 0.77598074,  1.        , -0.92346708, -0.99923895, -0.58826587,
            -0.44069024],
           [-0.47458546, -0.92346708,  1.        ,  0.93773029,  0.23297648,
             0.75137473],
           [-0.75078643, -0.99923895,  0.93773029,  1.        ,  0.55627469,
             0.47536961],
           [-0.9665554 , -0.58826587,  0.23297648,  0.55627469,  1.        ,
            -0.46666491],
           [ 0.22423734, -0.44069024,  0.75137473,  0.47536961, -0.46666491,
             1.        ]])

    """
    if bias is not np._NoValue or ddof is not np._NoValue:
        # 2015-03-15, 1.10
        warnings.warn('bias and ddof have no effect and are deprecated',
                      DeprecationWarning, stacklevel=2)
    c = cov(x, y, rowvar, dtype=dtype)
    try:
        d = diag(c)
    except ValueError:
        # scalar covariance
        # nan if incorrect value (nan, inf, 0), 1 otherwise
        return c / c
    stddev = sqrt(d.real)
    c /= stddev[:, None]
    c /= stddev[None, :]

    # Clip real and imaginary parts to [-1, 1].  This does not guarantee
    # abs(a[i,j]) <= 1 for complex arrays, but is the best we can do without
    # excessive work.
    np.clip(c.real, -1, 1, out=c.real)
    if np.iscomplexobj(c):
        np.clip(c.imag, -1, 1, out=c.imag)

    return c


@set_module('numpy')
def blackman(M):
    """
    Return the Blackman window.

    The Blackman window is a taper formed by using the first three
    terms of a summation of cosines. It was designed to have close to the
    minimal leakage possible.  It is close to optimal, only slightly worse
    than a Kaiser window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.

    Returns
    -------
    out : ndarray
        The window, with the maximum value normalized to one (the value one
        appears only if the number of samples is odd).

    See Also
    --------
    bartlett, hamming, hanning, kaiser

    Notes
    -----
    The Blackman window is defined as

    .. math::  w(n) = 0.42 - 0.5 \\cos(2\\pi n/M) + 0.08 \\cos(4\\pi n/M)

    Most references to the Blackman window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function. It is known as a
    "near optimal" tapering function, almost as good (by some measures)
    as the kaiser window.

    References
    ----------
    Blackman, R.B. and Tukey, J.W., (1958) The measurement of power spectra,
    Dover Publications, New York.

    Oppenheim, A.V., and R.W. Schafer. Discrete-Time Signal Processing.
    Upper Saddle River, NJ: Prentice-Hall, 1999, pp. 468-471.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> np.blackman(12)
    array([-1.38777878e-17,   3.26064346e-02,   1.59903635e-01, # may vary
            4.14397981e-01,   7.36045180e-01,   9.67046769e-01,
            9.67046769e-01,   7.36045180e-01,   4.14397981e-01,
            1.59903635e-01,   3.26064346e-02,  -1.38777878e-17])

    Plot the window and the frequency response.

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from numpy.fft import fft, fftshift
        window = np.blackman(51)
        plt.plot(window)
        plt.title("Blackman window")
        plt.ylabel("Amplitude")
        plt.xlabel("Sample")
        plt.show()  # doctest: +SKIP

        plt.figure()
        A = fft(window, 2048) / 25.5
        mag = np.abs(fftshift(A))
        freq = np.linspace(-0.5, 0.5, len(A))
        with np.errstate(divide='ignore', invalid='ignore'):
            response = 20 * np.log10(mag)
        response = np.clip(response, -100, 100)
        plt.plot(freq, response)
        plt.title("Frequency response of Blackman window")
        plt.ylabel("Magnitude [dB]")
        plt.xlabel("Normalized frequency [cycles per sample]")
        plt.axis('tight')
        plt.show()

    """
    # Ensures at least float64 via 0.0.  M should be an integer, but conversion
    # to double is safe for a range.
    values = np.array([0.0, M])
    M = values[1]

    if M < 1:
        return array([], dtype=values.dtype)
    if M == 1:
        return ones(1, dtype=values.dtype)
    n = arange(1-M, M, 2)
    return 0.42 + 0.5*cos(pi*n/(M-1)) + 0.08*cos(2.0*pi*n/(M-1))


@set_module('numpy')
def bartlett(M):
    """
    Return the Bartlett window.

    The Bartlett window is very similar to a triangular window, except
    that the end points are at zero.  It is often used in signal
    processing for tapering a signal, without generating too much
    ripple in the frequency domain.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.

    Returns
    -------
    out : array
        The triangular window, with the maximum value normalized to one
        (the value one appears only if the number of samples is odd), with
        the first and last samples equal to zero.

    See Also
    --------
    blackman, hamming, hanning, kaiser

    Notes
    -----
    The Bartlett window is defined as

    .. math:: w(n) = \\frac{2}{M-1} \\left(
              \\frac{M-1}{2} - \\left|n - \\frac{M-1}{2}\\right|
              \\right)

    Most references to the Bartlett window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  Note that convolution with this window produces linear
    interpolation.  It is also known as an apodization (which means "removing
    the foot", i.e. smoothing discontinuities at the beginning and end of the
    sampled signal) or tapering function. The Fourier transform of the
    Bartlett window is the product of two sinc functions. Note the excellent
    discussion in Kanasewich [2]_.

    References
    ----------
    .. [1] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",
           Biometrika 37, 1-16, 1950.
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics",
           The University of Alberta Press, 1975, pp. 109-110.
    .. [3] A.V. Oppenheim and R.W. Schafer, "Discrete-Time Signal
           Processing", Prentice-Hall, 1999, pp. 468-471.
    .. [4] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function
    .. [5] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
           "Numerical Recipes", Cambridge University Press, 1986, page 429.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> np.bartlett(12)
    array([ 0.        ,  0.18181818,  0.36363636,  0.54545455,  0.72727273, # may vary
            0.90909091,  0.90909091,  0.72727273,  0.54545455,  0.36363636,
            0.18181818,  0.        ])

    Plot the window and its frequency response (requires SciPy and matplotlib).

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from numpy.fft import fft, fftshift
        window = np.bartlett(51)
        plt.plot(window)
        plt.title("Bartlett window")
        plt.ylabel("Amplitude")
        plt.xlabel("Sample")
        plt.show()
        plt.figure()
        A = fft(window, 2048) / 25.5
        mag = np.abs(fftshift(A))
        freq = np.linspace(-0.5, 0.5, len(A))
        with np.errstate(divide='ignore', invalid='ignore'):
            response = 20 * np.log10(mag)
        response = np.clip(response, -100, 100)
        plt.plot(freq, response)
        plt.title("Frequency response of Bartlett window")
        plt.ylabel("Magnitude [dB]")
        plt.xlabel("Normalized frequency [cycles per sample]")
        plt.axis('tight')
        plt.show()

    """
    # Ensures at least float64 via 0.0.  M should be an integer, but conversion
    # to double is safe for a range.
    values = np.array([0.0, M])
    M = values[1]

    if M < 1:
        return array([], dtype=values.dtype)
    if M == 1:
        return ones(1, dtype=values.dtype)
    n = arange(1-M, M, 2)
    return where(less_equal(n, 0), 1 + n/(M-1), 1 - n/(M-1))


@set_module('numpy')
def hanning(M):
    """
    Return the Hanning window.

    The Hanning window is a taper formed by using a weighted cosine.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.

    Returns
    -------
    out : ndarray, shape(M,)
        The window, with the maximum value normalized to one (the value
        one appears only if `M` is odd).

    See Also
    --------
    bartlett, blackman, hamming, kaiser

    Notes
    -----
    The Hanning window is defined as

    .. math::  w(n) = 0.5 - 0.5\\cos\\left(\\frac{2\\pi{n}}{M-1}\\right)
               \\qquad 0 \\leq n \\leq M-1

    The Hanning was named for Julius von Hann, an Austrian meteorologist.
    It is also known as the Cosine Bell. Some authors prefer that it be
    called a Hann window, to help avoid confusion with the very similar
    Hamming window.

    Most references to the Hanning window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.

    References
    ----------
    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
           spectra, Dover Publications, New York.
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics",
           The University of Alberta Press, 1975, pp. 106-108.
    .. [3] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function
    .. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
           "Numerical Recipes", Cambridge University Press, 1986, page 425.

    Examples
    --------
    >>> import numpy as np
    >>> np.hanning(12)
    array([0.        , 0.07937323, 0.29229249, 0.57115742, 0.82743037,
           0.97974649, 0.97974649, 0.82743037, 0.57115742, 0.29229249,
           0.07937323, 0.        ])

    Plot the window and its frequency response.

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from numpy.fft import fft, fftshift
        window = np.hanning(51)
        plt.plot(window)
        plt.title("Hann window")
        plt.ylabel("Amplitude")
        plt.xlabel("Sample")
        plt.show()

        plt.figure()
        A = fft(window, 2048) / 25.5
        mag = np.abs(fftshift(A))
        freq = np.linspace(-0.5, 0.5, len(A))
        with np.errstate(divide='ignore', invalid='ignore'):
            response = 20 * np.log10(mag)
        response = np.clip(response, -100, 100)
        plt.plot(freq, response)
        plt.title("Frequency response of the Hann window")
        plt.ylabel("Magnitude [dB]")
        plt.xlabel("Normalized frequency [cycles per sample]")
        plt.axis('tight')
        plt.show()

    """
    # Ensures at least float64 via 0.0.  M should be an integer, but conversion
    # to double is safe for a range.
    values = np.array([0.0, M])
    M = values[1]

    if M < 1:
        return array([], dtype=values.dtype)
    if M == 1:
        return ones(1, dtype=values.dtype)
    n = arange(1-M, M, 2)
    return 0.5 + 0.5*cos(pi*n/(M-1))


@set_module('numpy')
def hamming(M):
    """
    Return the Hamming window.

    The Hamming window is a taper formed by using a weighted cosine.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.

    Returns
    -------
    out : ndarray
        The window, with the maximum value normalized to one (the value
        one appears only if the number of samples is odd).

    See Also
    --------
    bartlett, blackman, hanning, kaiser

    Notes
    -----
    The Hamming window is defined as

    .. math::  w(n) = 0.54 - 0.46\\cos\\left(\\frac{2\\pi{n}}{M-1}\\right)
               \\qquad 0 \\leq n \\leq M-1

    The Hamming was named for R. W. Hamming, an associate of J. W. Tukey
    and is described in Blackman and Tukey. It was recommended for
    smoothing the truncated autocovariance function in the time domain.
    Most references to the Hamming window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.

    References
    ----------
    .. [1] Blackman, R.B. and Tukey, J.W., (1958) The measurement of power
           spectra, Dover Publications, New York.
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The
           University of Alberta Press, 1975, pp. 109-110.
    .. [3] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function
    .. [4] W.H. Press,  B.P. Flannery, S.A. Teukolsky, and W.T. Vetterling,
           "Numerical Recipes", Cambridge University Press, 1986, page 425.

    Examples
    --------
    >>> import numpy as np
    >>> np.hamming(12)
    array([ 0.08      ,  0.15302337,  0.34890909,  0.60546483,  0.84123594, # may vary
            0.98136677,  0.98136677,  0.84123594,  0.60546483,  0.34890909,
            0.15302337,  0.08      ])

    Plot the window and the frequency response.

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from numpy.fft import fft, fftshift
        window = np.hamming(51)
        plt.plot(window)
        plt.title("Hamming window")
        plt.ylabel("Amplitude")
        plt.xlabel("Sample")
        plt.show()

        plt.figure()
        A = fft(window, 2048) / 25.5
        mag = np.abs(fftshift(A))
        freq = np.linspace(-0.5, 0.5, len(A))
        response = 20 * np.log10(mag)
        response = np.clip(response, -100, 100)
        plt.plot(freq, response)
        plt.title("Frequency response of Hamming window")
        plt.ylabel("Magnitude [dB]")
        plt.xlabel("Normalized frequency [cycles per sample]")
        plt.axis('tight')
        plt.show()

    """
    # Ensures at least float64 via 0.0.  M should be an integer, but conversion
    # to double is safe for a range.
    values = np.array([0.0, M])
    M = values[1]

    if M < 1:
        return array([], dtype=values.dtype)
    if M == 1:
        return ones(1, dtype=values.dtype)
    n = arange(1-M, M, 2)
    return 0.54 + 0.46*cos(pi*n/(M-1))


## Code from cephes for i0

_i0A = [
    -4.41534164647933937950E-18,
    3.33079451882223809783E-17,
    -2.43127984654795469359E-16,
    1.71539128555513303061E-15,
    -1.16853328779934516808E-14,
    7.67618549860493561688E-14,
    -4.85644678311192946090E-13,
    2.95505266312963983461E-12,
    -1.72682629144155570723E-11,
    9.67580903537323691224E-11,
    -5.18979560163526290666E-10,
    2.65982372468238665035E-9,
    -1.30002500998624804212E-8,
    6.04699502254191894932E-8,
    -2.67079385394061173391E-7,
    1.11738753912010371815E-6,
    -4.41673835845875056359E-6,
    1.64484480707288970893E-5,
    -5.75419501008210370398E-5,
    1.88502885095841655729E-4,
    -5.76375574538582365885E-4,
    1.63947561694133579842E-3,
    -4.32430999505057594430E-3,
    1.05464603945949983183E-2,
    -2.37374148058994688156E-2,
    4.93052842396707084878E-2,
    -9.49010970480476444210E-2,
    1.71620901522208775349E-1,
    -3.04682672343198398683E-1,
    6.76795274409476084995E-1
    ]

_i0B = [
    -7.23318048787475395456E-18,
    -4.83050448594418207126E-18,
    4.46562142029675999901E-17,
    3.46122286769746109310E-17,
    -2.82762398051658348494E-16,
    -3.42548561967721913462E-16,
    1.77256013305652638360E-15,
    3.81168066935262242075E-15,
    -9.55484669882830764870E-15,
    -4.15056934728722208663E-14,
    1.54008621752140982691E-14,
    3.85277838274214270114E-13,
    7.18012445138366623367E-13,
    -1.79417853150680611778E-12,
    -1.32158118404477131188E-11,
    -3.14991652796324136454E-11,
    1.18891471078464383424E-11,
    4.94060238822496958910E-10,
    3.39623202570838634515E-9,
    2.26666899049817806459E-8,
    2.04891858946906374183E-7,
    2.89137052083475648297E-6,
    6.88975834691682398426E-5,
    3.36911647825569408990E-3,
    8.04490411014108831608E-1
    ]


def _chbevl(x, vals):
    b0 = vals[0]
    b1 = 0.0

    for i in range(1, len(vals)):
        b2 = b1
        b1 = b0
        b0 = x*b1 - b2 + vals[i]

    return 0.5*(b0 - b2)


def _i0_1(x):
    return exp(x) * _chbevl(x/2.0-2, _i0A)


def _i0_2(x):
    return exp(x) * _chbevl(32.0/x - 2.0, _i0B) / sqrt(x)


def _i0_dispatcher(x):
    return (x,)


@array_function_dispatch(_i0_dispatcher)
def i0(x):
    """
    Modified Bessel function of the first kind, order 0.

    Usually denoted :math:`I_0`.

    Parameters
    ----------
    x : array_like of float
        Argument of the Bessel function.

    Returns
    -------
    out : ndarray, shape = x.shape, dtype = float
        The modified Bessel function evaluated at each of the elements of `x`.

    See Also
    --------
    scipy.special.i0, scipy.special.iv, scipy.special.ive

    Notes
    -----
    The scipy implementation is recommended over this function: it is a
    proper ufunc written in C, and more than an order of magnitude faster.

    We use the algorithm published by Clenshaw [1]_ and referenced by
    Abramowitz and Stegun [2]_, for which the function domain is
    partitioned into the two intervals [0,8] and (8,inf), and Chebyshev
    polynomial expansions are employed in each interval. Relative error on
    the domain [0,30] using IEEE arithmetic is documented [3]_ as having a
    peak of 5.8e-16 with an rms of 1.4e-16 (n = 30000).

    References
    ----------
    .. [1] C. W. Clenshaw, "Chebyshev series for mathematical functions", in
           *National Physical Laboratory Mathematical Tables*, vol. 5, London:
           Her Majesty's Stationery Office, 1962.
    .. [2] M. Abramowitz and I. A. Stegun, *Handbook of Mathematical
           Functions*, 10th printing, New York: Dover, 1964, pp. 379.
           https://personal.math.ubc.ca/~cbm/aands/page_379.htm
    .. [3] https://metacpan.org/pod/distribution/Math-Cephes/lib/Math/Cephes.pod#i0:-Modified-Bessel-function-of-order-zero

    Examples
    --------
    >>> import numpy as np
    >>> np.i0(0.)
    array(1.0)
    >>> np.i0([0, 1, 2, 3])
    array([1.        , 1.26606588, 2.2795853 , 4.88079259])

    """
    x = np.asanyarray(x)
    if x.dtype.kind == 'c':
        raise TypeError("i0 not supported for complex values")
    if x.dtype.kind != 'f':
        x = x.astype(float)
    x = np.abs(x)
    return piecewise(x, [x <= 8.0], [_i0_1, _i0_2])

## End of cephes code for i0


@set_module('numpy')
def kaiser(M, beta):
    """
    Return the Kaiser window.

    The Kaiser window is a taper formed by using a Bessel function.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an
        empty array is returned.
    beta : float
        Shape parameter for window.

    Returns
    -------
    out : array
        The window, with the maximum value normalized to one (the value
        one appears only if the number of samples is odd).

    See Also
    --------
    bartlett, blackman, hamming, hanning

    Notes
    -----
    The Kaiser window is defined as

    .. math::  w(n) = I_0\\left( \\beta \\sqrt{1-\\frac{4n^2}{(M-1)^2}}
               \\right)/I_0(\\beta)

    with

    .. math:: \\quad -\\frac{M-1}{2} \\leq n \\leq \\frac{M-1}{2},

    where :math:`I_0` is the modified zeroth-order Bessel function.

    The Kaiser was named for Jim Kaiser, who discovered a simple
    approximation to the DPSS window based on Bessel functions.  The Kaiser
    window is a very good approximation to the Digital Prolate Spheroidal
    Sequence, or Slepian window, which is the transform which maximizes the
    energy in the main lobe of the window relative to total energy.

    The Kaiser can approximate many other windows by varying the beta
    parameter.

    ====  =======================
    beta  Window shape
    ====  =======================
    0     Rectangular
    5     Similar to a Hamming
    6     Similar to a Hanning
    8.6   Similar to a Blackman
    ====  =======================

    A beta value of 14 is probably a good starting point. Note that as beta
    gets large, the window narrows, and so the number of samples needs to be
    large enough to sample the increasingly narrow spike, otherwise NaNs will
    get returned.

    Most references to the Kaiser window come from the signal processing
    literature, where it is used as one of many windowing functions for
    smoothing values.  It is also known as an apodization (which means
    "removing the foot", i.e. smoothing discontinuities at the beginning
    and end of the sampled signal) or tapering function.

    References
    ----------
    .. [1] J. F. Kaiser, "Digital Filters" - Ch 7 in "Systems analysis by
           digital computer", Editors: F.F. Kuo and J.F. Kaiser, p 218-285.
           John Wiley and Sons, New York, (1966).
    .. [2] E.R. Kanasewich, "Time Sequence Analysis in Geophysics", The
           University of Alberta Press, 1975, pp. 177-178.
    .. [3] Wikipedia, "Window function",
           https://en.wikipedia.org/wiki/Window_function

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> np.kaiser(12, 14)
     array([7.72686684e-06, 3.46009194e-03, 4.65200189e-02, # may vary
            2.29737120e-01, 5.99885316e-01, 9.45674898e-01,
            9.45674898e-01, 5.99885316e-01, 2.29737120e-01,
            4.65200189e-02, 3.46009194e-03, 7.72686684e-06])


    Plot the window and the frequency response.

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from numpy.fft import fft, fftshift
        window = np.kaiser(51, 14)
        plt.plot(window)
        plt.title("Kaiser window")
        plt.ylabel("Amplitude")
        plt.xlabel("Sample")
        plt.show()

        plt.figure()
        A = fft(window, 2048) / 25.5
        mag = np.abs(fftshift(A))
        freq = np.linspace(-0.5, 0.5, len(A))
        response = 20 * np.log10(mag)
        response = np.clip(response, -100, 100)
        plt.plot(freq, response)
        plt.title("Frequency response of Kaiser window")
        plt.ylabel("Magnitude [dB]")
        plt.xlabel("Normalized frequency [cycles per sample]")
        plt.axis('tight')
        plt.show()

    """
    # Ensures at least float64 via 0.0.  M should be an integer, but conversion
    # to double is safe for a range.  (Simplified result_type with 0.0
    # strongly typed.  result-type is not/less order sensitive, but that mainly
    # matters for integers anyway.)
    values = np.array([0.0, M, beta])
    M = values[1]
    beta = values[2]

    if M == 1:
        return np.ones(1, dtype=values.dtype)
    n = arange(0, M)
    alpha = (M-1)/2.0
    return i0(beta * sqrt(1-((n-alpha)/alpha)**2.0))/i0(beta)


def _sinc_dispatcher(x):
    return (x,)


@array_function_dispatch(_sinc_dispatcher)
def sinc(x):
    r"""
    Return the normalized sinc function.

    The sinc function is equal to :math:`\sin(\pi x)/(\pi x)` for any argument
    :math:`x\ne 0`. ``sinc(0)`` takes the limit value 1, making ``sinc`` not
    only everywhere continuous but also infinitely differentiable.

    .. note::

        Note the normalization factor of ``pi`` used in the definition.
        This is the most commonly used definition in signal processing.
        Use ``sinc(x / np.pi)`` to obtain the unnormalized sinc function
        :math:`\sin(x)/x` that is more common in mathematics.

    Parameters
    ----------
    x : ndarray
        Array (possibly multi-dimensional) of values for which to calculate
        ``sinc(x)``.

    Returns
    -------
    out : ndarray
        ``sinc(x)``, which has the same shape as the input.

    Notes
    -----
    The name sinc is short for "sine cardinal" or "sinus cardinalis".

    The sinc function is used in various signal processing applications,
    including in anti-aliasing, in the construction of a Lanczos resampling
    filter, and in interpolation.

    For bandlimited interpolation of discrete-time signals, the ideal
    interpolation kernel is proportional to the sinc function.

    References
    ----------
    .. [1] Weisstein, Eric W. "Sinc Function." From MathWorld--A Wolfram Web
           Resource. https://mathworld.wolfram.com/SincFunction.html
    .. [2] Wikipedia, "Sinc function",
           https://en.wikipedia.org/wiki/Sinc_function

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-4, 4, 41)
    >>> np.sinc(x)
     array([-3.89804309e-17,  -4.92362781e-02,  -8.40918587e-02, # may vary
            -8.90384387e-02,  -5.84680802e-02,   3.89804309e-17,
            6.68206631e-02,   1.16434881e-01,   1.26137788e-01,
            8.50444803e-02,  -3.89804309e-17,  -1.03943254e-01,
            -1.89206682e-01,  -2.16236208e-01,  -1.55914881e-01,
            3.89804309e-17,   2.33872321e-01,   5.04551152e-01,
            7.56826729e-01,   9.35489284e-01,   1.00000000e+00,
            9.35489284e-01,   7.56826729e-01,   5.04551152e-01,
            2.33872321e-01,   3.89804309e-17,  -1.55914881e-01,
           -2.16236208e-01,  -1.89206682e-01,  -1.03943254e-01,
           -3.89804309e-17,   8.50444803e-02,   1.26137788e-01,
            1.16434881e-01,   6.68206631e-02,   3.89804309e-17,
            -5.84680802e-02,  -8.90384387e-02,  -8.40918587e-02,
            -4.92362781e-02,  -3.89804309e-17])

    >>> plt.plot(x, np.sinc(x))
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.title("Sinc Function")
    Text(0.5, 1.0, 'Sinc Function')
    >>> plt.ylabel("Amplitude")
    Text(0, 0.5, 'Amplitude')
    >>> plt.xlabel("X")
    Text(0.5, 0, 'X')
    >>> plt.show()

    """
    x = np.asanyarray(x)
    y = pi * where(x == 0, 1.0e-20, x)
    return sin(y)/y


def _ureduce(a, func, keepdims=False, **kwargs):
    """
    Internal Function.
    Call `func` with `a` as first argument swapping the axes to use extended
    axis on functions that don't support it natively.

    Returns result and a.shape with axis dims set to 1.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    func : callable
        Reduction function capable of receiving a single axis argument.
        It is called with `a` as first argument followed by `kwargs`.
    kwargs : keyword arguments
        additional keyword arguments to pass to `func`.

    Returns
    -------
    result : tuple
        Result of func(a, **kwargs) and a.shape with axis dims set to 1
        which can be used to reshape the result to the same shape a ufunc with
        keepdims=True would produce.

    """
    a = np.asanyarray(a)
    axis = kwargs.get('axis', None)
    out = kwargs.get('out', None)

    if keepdims is np._NoValue:
        keepdims = False

    nd = a.ndim
    if axis is not None:
        axis = _nx.normalize_axis_tuple(axis, nd)

        if keepdims:
            if out is not None:
                index_out = tuple(
                    0 if i in axis else slice(None) for i in range(nd))
                kwargs['out'] = out[(Ellipsis, ) + index_out]

        if len(axis) == 1:
            kwargs['axis'] = axis[0]
        else:
            keep = set(range(nd)) - set(axis)
            nkeep = len(keep)
            # swap axis that should not be reduced to front
            for i, s in enumerate(sorted(keep)):
                a = a.swapaxes(i, s)
            # merge reduced axis
            a = a.reshape(a.shape[:nkeep] + (-1,))
            kwargs['axis'] = -1
    else:
        if keepdims:
            if out is not None:
                index_out = (0, ) * nd
                kwargs['out'] = out[(Ellipsis, ) + index_out]

    r = func(a, **kwargs)

    if out is not None:
        return out

    if keepdims:
        if axis is None:
            index_r = (np.newaxis, ) * nd
        else:
            index_r = tuple(
                np.newaxis if i in axis else slice(None)
                for i in range(nd))
        r = r[(Ellipsis, ) + index_r]

    return r


def _median_dispatcher(
        a, axis=None, out=None, overwrite_input=None, keepdims=None):
    return (a, out)


@array_function_dispatch(_median_dispatcher)
def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    """
    Compute the median along the specified axis.

    Returns the median of the array elements.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : {int, sequence of int, None}, optional
        Axis or axes along which the medians are computed. The default,
        axis=None, will compute the median along a flattened version of
        the array. If a sequence of axes, the array is first flattened
        along the given axes, then the median is computed along the
        resulting flattened axis.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.
    overwrite_input : bool, optional
       If True, then allow use of memory of input array `a` for
       calculations. The input array will be modified by the call to
       `median`. This will save memory when you do not need to preserve
       the contents of the input array. Treat the input as undefined,
       but it will probably be fully or partially sorted. Default is
       False. If `overwrite_input` is ``True`` and `a` is not already an
       `ndarray`, an error will be raised.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.

    Returns
    -------
    median : ndarray
        A new array holding the result. If the input contains integers
        or floats smaller than ``float64``, then the output data-type is
        ``np.float64``.  Otherwise, the data-type of the output is the
        same as that of the input. If `out` is specified, that array is
        returned instead.

    See Also
    --------
    mean, percentile

    Notes
    -----
    Given a vector ``V`` of length ``N``, the median of ``V`` is the
    middle value of a sorted copy of ``V``, ``V_sorted`` - i
    e., ``V_sorted[(N-1)/2]``, when ``N`` is odd, and the average of the
    two middle values of ``V_sorted`` when ``N`` is even.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[10, 7, 4], [3, 2, 1]])
    >>> a
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> np.median(a)
    np.float64(3.5)
    >>> np.median(a, axis=0)
    array([6.5, 4.5, 2.5])
    >>> np.median(a, axis=1)
    array([7.,  2.])
    >>> np.median(a, axis=(0, 1))
    np.float64(3.5)
    >>> m = np.median(a, axis=0)
    >>> out = np.zeros_like(m)
    >>> np.median(a, axis=0, out=m)
    array([6.5,  4.5,  2.5])
    >>> m
    array([6.5,  4.5,  2.5])
    >>> b = a.copy()
    >>> np.median(b, axis=1, overwrite_input=True)
    array([7.,  2.])
    >>> assert not np.all(a==b)
    >>> b = a.copy()
    >>> np.median(b, axis=None, overwrite_input=True)
    np.float64(3.5)
    >>> assert not np.all(a==b)

    """
    return _ureduce(a, func=_median, keepdims=keepdims, axis=axis, out=out,
                    overwrite_input=overwrite_input)


def _median(a, axis=None, out=None, overwrite_input=False):
    # can't be reasonably be implemented in terms of percentile as we have to
    # call mean to not break astropy
    a = np.asanyarray(a)

    # Set the partition indexes
    if axis is None:
        sz = a.size
    else:
        sz = a.shape[axis]
    if sz % 2 == 0:
        szh = sz // 2
        kth = [szh - 1, szh]
    else:
        kth = [(sz - 1) // 2]

    # We have to check for NaNs (as of writing 'M' doesn't actually work).
    supports_nans = np.issubdtype(a.dtype, np.inexact) or a.dtype.kind in 'Mm'
    if supports_nans:
        kth.append(-1)

    if overwrite_input:
        if axis is None:
            part = a.ravel()
            part.partition(kth)
        else:
            a.partition(kth, axis=axis)
            part = a
    else:
        part = partition(a, kth, axis=axis)

    if part.shape == ():
        # make 0-D arrays work
        return part.item()
    if axis is None:
        axis = 0

    indexer = [slice(None)] * part.ndim
    index = part.shape[axis] // 2
    if part.shape[axis] % 2 == 1:
        # index with slice to allow mean (below) to work
        indexer[axis] = slice(index, index+1)
    else:
        indexer[axis] = slice(index-1, index+1)
    indexer = tuple(indexer)

    # Use mean in both odd and even case to coerce data type,
    # using out array if needed.
    rout = mean(part[indexer], axis=axis, out=out)
    if supports_nans and sz > 0:
        # If nans are possible, warn and replace by nans like mean would.
        rout = np.lib._utils_impl._median_nancheck(part, rout, axis)

    return rout


def _percentile_dispatcher(a, q, axis=None, out=None, overwrite_input=None,
                           method=None, keepdims=None, *, weights=None,
                           interpolation=None):
    return (a, q, out, weights)


@array_function_dispatch(_percentile_dispatcher)
def percentile(a,
               q,
               axis=None,
               out=None,
               overwrite_input=False,
               method="linear",
               keepdims=False,
               *,
               weights=None,
               interpolation=None):
    """
    Compute the q-th percentile of the data along the specified axis.

    Returns the q-th percentile(s) of the array elements.

    Parameters
    ----------
    a : array_like of real numbers
        Input array or object that can be converted to an array.
    q : array_like of float
        Percentage or sequence of percentages for the percentiles to compute.
        Values must be between 0 and 100 inclusive.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the percentiles are computed. The
        default is to compute the percentile(s) along a flattened
        version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.
    overwrite_input : bool, optional
        If True, then allow the input array `a` to be modified by intermediate
        calculations, to save memory. In this case, the contents of the input
        `a` after this function completes is undefined.
    method : str, optional
        This parameter specifies the method to use for estimating the
        percentile.  There are many different methods, some unique to NumPy.
        See the notes for explanation.  The options sorted by their R type
        as summarized in the H&F paper [1]_ are:

        1. 'inverted_cdf'
        2. 'averaged_inverted_cdf'
        3. 'closest_observation'
        4. 'interpolated_inverted_cdf'
        5. 'hazen'
        6. 'weibull'
        7. 'linear'  (default)
        8. 'median_unbiased'
        9. 'normal_unbiased'

        The first three methods are discontinuous.  NumPy further defines the
        following discontinuous variations of the default 'linear' (7.) option:

        * 'lower'
        * 'higher',
        * 'midpoint'
        * 'nearest'

        .. versionchanged:: 1.22.0
            This argument was previously called "interpolation" and only
            offered the "linear" default and last four options.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the
        result will broadcast correctly against the original array `a`.

     weights : array_like, optional
        An array of weights associated with the values in `a`. Each value in
        `a` contributes to the percentile according to its associated weight.
        The weights array can either be 1-D (in which case its length must be
        the size of `a` along the given axis) or of the same shape as `a`.
        If `weights=None`, then all data in `a` are assumed to have a
        weight equal to one.
        Only `method="inverted_cdf"` supports weights.
        See the notes for more details.

        .. versionadded:: 2.0.0

    interpolation : str, optional
        Deprecated name for the method keyword argument.

        .. deprecated:: 1.22.0

    Returns
    -------
    percentile : scalar or ndarray
        If `q` is a single percentile and `axis=None`, then the result
        is a scalar. If multiple percentiles are given, first axis of
        the result corresponds to the percentiles. The other axes are
        the axes that remain after the reduction of `a`. If the input
        contains integers or floats smaller than ``float64``, the output
        data-type is ``float64``. Otherwise, the output data-type is the
        same as that of the input. If `out` is specified, that array is
        returned instead.

    See Also
    --------
    mean
    median : equivalent to ``percentile(..., 50)``
    nanpercentile
    quantile : equivalent to percentile, except q in the range [0, 1].

    Notes
    -----
    The behavior of `numpy.percentile` with percentage `q` is
    that of `numpy.quantile` with argument ``q/100``.
    For more information, please see `numpy.quantile`.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[10, 7, 4], [3, 2, 1]])
    >>> a
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> np.percentile(a, 50)
    3.5
    >>> np.percentile(a, 50, axis=0)
    array([6.5, 4.5, 2.5])
    >>> np.percentile(a, 50, axis=1)
    array([7.,  2.])
    >>> np.percentile(a, 50, axis=1, keepdims=True)
    array([[7.],
           [2.]])

    >>> m = np.percentile(a, 50, axis=0)
    >>> out = np.zeros_like(m)
    >>> np.percentile(a, 50, axis=0, out=out)
    array([6.5, 4.5, 2.5])
    >>> m
    array([6.5, 4.5, 2.5])

    >>> b = a.copy()
    >>> np.percentile(b, 50, axis=1, overwrite_input=True)
    array([7.,  2.])
    >>> assert not np.all(a == b)

    The different methods can be visualized graphically:

    .. plot::

        import matplotlib.pyplot as plt

        a = np.arange(4)
        p = np.linspace(0, 100, 6001)
        ax = plt.gca()
        lines = [
            ('linear', '-', 'C0'),
            ('inverted_cdf', ':', 'C1'),
            # Almost the same as `inverted_cdf`:
            ('averaged_inverted_cdf', '-.', 'C1'),
            ('closest_observation', ':', 'C2'),
            ('interpolated_inverted_cdf', '--', 'C1'),
            ('hazen', '--', 'C3'),
            ('weibull', '-.', 'C4'),
            ('median_unbiased', '--', 'C5'),
            ('normal_unbiased', '-.', 'C6'),
            ]
        for method, style, color in lines:
            ax.plot(
                p, np.percentile(a, p, method=method),
                label=method, linestyle=style, color=color)
        ax.set(
            title='Percentiles for different methods and data: ' + str(a),
            xlabel='Percentile',
            ylabel='Estimated percentile value',
            yticks=a)
        ax.legend(bbox_to_anchor=(1.03, 1))
        plt.tight_layout()
        plt.show()

    References
    ----------
    .. [1] R. J. Hyndman and Y. Fan,
       "Sample quantiles in statistical packages,"
       The American Statistician, 50(4), pp. 361-365, 1996

    """
    if interpolation is not None:
        method = _check_interpolation_as_method(
            method, interpolation, "percentile")

    a = np.asanyarray(a)
    if a.dtype.kind == "c":
        raise TypeError("a must be an array of real numbers")

    # Use dtype of array if possible (e.g., if q is a python int or float)
    # by making the divisor have the dtype of the data array.
    q = np.true_divide(q, a.dtype.type(100) if a.dtype.kind == "f" else 100)
    q = asanyarray(q)  # undo any decay that the ufunc performed (see gh-13105)
    if not _quantile_is_valid(q):
        raise ValueError("Percentiles must be in the range [0, 100]")

    if weights is not None:
        if method != "inverted_cdf":
            msg = ("Only method 'inverted_cdf' supports weights. "
                   f"Got: {method}.")
            raise ValueError(msg)
        if axis is not None:
            axis = _nx.normalize_axis_tuple(axis, a.ndim, argname="axis")
        weights = _weights_are_valid(weights=weights, a=a, axis=axis)
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative.")

    return _quantile_unchecked(
        a, q, axis, out, overwrite_input, method, keepdims, weights)


def _quantile_dispatcher(a, q, axis=None, out=None, overwrite_input=None,
                         method=None, keepdims=None, *, weights=None,
                         interpolation=None):
    return (a, q, out, weights)


@array_function_dispatch(_quantile_dispatcher)
def quantile(a,
             q,
             axis=None,
             out=None,
             overwrite_input=False,
             method="linear",
             keepdims=False,
             *,
             weights=None,
             interpolation=None):
    """
    Compute the q-th quantile of the data along the specified axis.

    Parameters
    ----------
    a : array_like of real numbers
        Input array or object that can be converted to an array.
    q : array_like of float
        Probability or sequence of probabilities of the quantiles to compute.
        Values must be between 0 and 1 inclusive.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the quantiles are computed. The default is
        to compute the quantile(s) along a flattened version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape and buffer length as the expected output, but the
        type (of the output) will be cast if necessary.
    overwrite_input : bool, optional
        If True, then allow the input array `a` to be modified by
        intermediate calculations, to save memory. In this case, the
        contents of the input `a` after this function completes is
        undefined.
    method : str, optional
        This parameter specifies the method to use for estimating the
        quantile.  There are many different methods, some unique to NumPy.
        The recommended options, numbered as they appear in [1]_, are:

        1. 'inverted_cdf'
        2. 'averaged_inverted_cdf'
        3. 'closest_observation'
        4. 'interpolated_inverted_cdf'
        5. 'hazen'
        6. 'weibull'
        7. 'linear'  (default)
        8. 'median_unbiased'
        9. 'normal_unbiased'

        The first three methods are discontinuous. For backward compatibility
        with previous versions of NumPy, the following discontinuous variations
        of the default 'linear' (7.) option are available:

        * 'lower'
        * 'higher',
        * 'midpoint'
        * 'nearest'

        See Notes for details.

        .. versionchanged:: 1.22.0
            This argument was previously called "interpolation" and only
            offered the "linear" default and last four options.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the
        result will broadcast correctly against the original array `a`.

    weights : array_like, optional
        An array of weights associated with the values in `a`. Each value in
        `a` contributes to the quantile according to its associated weight.
        The weights array can either be 1-D (in which case its length must be
        the size of `a` along the given axis) or of the same shape as `a`.
        If `weights=None`, then all data in `a` are assumed to have a
        weight equal to one.
        Only `method="inverted_cdf"` supports weights.
        See the notes for more details.

        .. versionadded:: 2.0.0

    interpolation : str, optional
        Deprecated name for the method keyword argument.

        .. deprecated:: 1.22.0

    Returns
    -------
    quantile : scalar or ndarray
        If `q` is a single probability and `axis=None`, then the result
        is a scalar. If multiple probability levels are given, first axis
        of the result corresponds to the quantiles. The other axes are
        the axes that remain after the reduction of `a`. If the input
        contains integers or floats smaller than ``float64``, the output
        data-type is ``float64``. Otherwise, the output data-type is the
        same as that of the input. If `out` is specified, that array is
        returned instead.

    See Also
    --------
    mean
    percentile : equivalent to quantile, but with q in the range [0, 100].
    median : equivalent to ``quantile(..., 0.5)``
    nanquantile

    Notes
    -----
    Given a sample `a` from an underlying distribution, `quantile` provides a
    nonparametric estimate of the inverse cumulative distribution function.

    By default, this is done by interpolating between adjacent elements in
    ``y``, a sorted copy of `a`::

        (1-g)*y[j] + g*y[j+1]

    where the index ``j`` and coefficient ``g`` are the integral and
    fractional components of ``q * (n-1)``, and ``n`` is the number of
    elements in the sample.

    This is a special case of Equation 1 of H&F [1]_. More generally,

    - ``j = (q*n + m - 1) // 1``, and
    - ``g = (q*n + m - 1) % 1``,

    where ``m`` may be defined according to several different conventions.
    The preferred convention may be selected using the ``method`` parameter:

    =============================== =============== ===============
    ``method``                      number in H&F   ``m``
    =============================== =============== ===============
    ``interpolated_inverted_cdf``   4               ``0``
    ``hazen``                       5               ``1/2``
    ``weibull``                     6               ``q``
    ``linear`` (default)            7               ``1 - q``
    ``median_unbiased``             8               ``q/3 + 1/3``
    ``normal_unbiased``             9               ``q/4 + 3/8``
    =============================== =============== ===============

    Note that indices ``j`` and ``j + 1`` are clipped to the range ``0`` to
    ``n - 1`` when the results of the formula would be outside the allowed
    range of non-negative indices. The ``- 1`` in the formulas for ``j`` and
    ``g`` accounts for Python's 0-based indexing.

    The table above includes only the estimators from H&F that are continuous
    functions of probability `q` (estimators 4-9). NumPy also provides the
    three discontinuous estimators from H&F (estimators 1-3), where ``j`` is
    defined as above, ``m`` is defined as follows, and ``g`` is a function
    of the real-valued ``index = q*n + m - 1`` and ``j``.

    1. ``inverted_cdf``: ``m = 0`` and ``g = int(index - j > 0)``
    2. ``averaged_inverted_cdf``: ``m = 0`` and
       ``g = (1 + int(index - j > 0)) / 2``
    3. ``closest_observation``: ``m = -1/2`` and
       ``g = 1 - int((index == j) & (j%2 == 1))``

    For backward compatibility with previous versions of NumPy, `quantile`
    provides four additional discontinuous estimators. Like
    ``method='linear'``, all have ``m = 1 - q`` so that ``j = q*(n-1) // 1``,
    but ``g`` is defined as follows.

    - ``lower``: ``g = 0``
    - ``midpoint``: ``g = 0.5``
    - ``higher``: ``g = 1``
    - ``nearest``: ``g = (q*(n-1) % 1) > 0.5``

    **Weighted quantiles:**
    More formally, the quantile at probability level :math:`q` of a cumulative
    distribution function :math:`F(y)=P(Y \\leq y)` with probability measure
    :math:`P` is defined as any number :math:`x` that fulfills the
    *coverage conditions*

    .. math:: P(Y < x) \\leq q \\quad\\text{and}\\quad P(Y \\leq x) \\geq q

    with random variable :math:`Y\\sim P`.
    Sample quantiles, the result of `quantile`, provide nonparametric
    estimation of the underlying population counterparts, represented by the
    unknown :math:`F`, given a data vector `a` of length ``n``.

    Some of the estimators above arise when one considers :math:`F` as the
    empirical distribution function of the data, i.e.
    :math:`F(y) = \\frac{1}{n} \\sum_i 1_{a_i \\leq y}`.
    Then, different methods correspond to different choices of :math:`x` that
    fulfill the above coverage conditions. Methods that follow this approach
    are ``inverted_cdf`` and ``averaged_inverted_cdf``.

    For weighted quantiles, the coverage conditions still hold. The
    empirical cumulative distribution is simply replaced by its weighted
    version, i.e.
    :math:`P(Y \\leq t) = \\frac{1}{\\sum_i w_i} \\sum_i w_i 1_{x_i \\leq t}`.
    Only ``method="inverted_cdf"`` supports weights.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[10, 7, 4], [3, 2, 1]])
    >>> a
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> np.quantile(a, 0.5)
    3.5
    >>> np.quantile(a, 0.5, axis=0)
    array([6.5, 4.5, 2.5])
    >>> np.quantile(a, 0.5, axis=1)
    array([7.,  2.])
    >>> np.quantile(a, 0.5, axis=1, keepdims=True)
    array([[7.],
           [2.]])
    >>> m = np.quantile(a, 0.5, axis=0)
    >>> out = np.zeros_like(m)
    >>> np.quantile(a, 0.5, axis=0, out=out)
    array([6.5, 4.5, 2.5])
    >>> m
    array([6.5, 4.5, 2.5])
    >>> b = a.copy()
    >>> np.quantile(b, 0.5, axis=1, overwrite_input=True)
    array([7.,  2.])
    >>> assert not np.all(a == b)

    See also `numpy.percentile` for a visualization of most methods.

    References
    ----------
    .. [1] R. J. Hyndman and Y. Fan,
       "Sample quantiles in statistical packages,"
       The American Statistician, 50(4), pp. 361-365, 1996

    """
    if interpolation is not None:
        method = _check_interpolation_as_method(
            method, interpolation, "quantile")

    a = np.asanyarray(a)
    if a.dtype.kind == "c":
        raise TypeError("a must be an array of real numbers")

    # Use dtype of array if possible (e.g., if q is a python int or float).
    if isinstance(q, (int, float)) and a.dtype.kind == "f":
        q = np.asanyarray(q, dtype=a.dtype)
    else:
        q = np.asanyarray(q)

    if not _quantile_is_valid(q):
        raise ValueError("Quantiles must be in the range [0, 1]")

    if weights is not None:
        if method != "inverted_cdf":
            msg = ("Only method 'inverted_cdf' supports weights. "
                   f"Got: {method}.")
            raise ValueError(msg)
        if axis is not None:
            axis = _nx.normalize_axis_tuple(axis, a.ndim, argname="axis")
        weights = _weights_are_valid(weights=weights, a=a, axis=axis)
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative.")

    return _quantile_unchecked(
        a, q, axis, out, overwrite_input, method, keepdims, weights)


def _quantile_unchecked(a,
                        q,
                        axis=None,
                        out=None,
                        overwrite_input=False,
                        method="linear",
                        keepdims=False,
                        weights=None):
    """Assumes that q is in [0, 1], and is an ndarray"""
    return _ureduce(a,
                    func=_quantile_ureduce_func,
                    q=q,
                    weights=weights,
                    keepdims=keepdims,
                    axis=axis,
                    out=out,
                    overwrite_input=overwrite_input,
                    method=method)


def _quantile_is_valid(q):
    # avoid expensive reductions, relevant for arrays with < O(1000) elements
    if q.ndim == 1 and q.size < 10:
        for i in range(q.size):
            if not (0.0 <= q[i] <= 1.0):
                return False
    else:
        if not (q.min() >= 0 and q.max() <= 1):
            return False
    return True


def _check_interpolation_as_method(method, interpolation, fname):
    # Deprecated NumPy 1.22, 2021-11-08
    warnings.warn(
        f"the `interpolation=` argument to {fname} was renamed to "
        "`method=`, which has additional options.\n"
        "Users of the modes 'nearest', 'lower', 'higher', or "
        "'midpoint' are encouraged to review the method they used. "
        "(Deprecated NumPy 1.22)",
        DeprecationWarning, stacklevel=4)
    if method != "linear":
        # sanity check, we assume this basically never happens
        raise TypeError(
            "You shall not pass both `method` and `interpolation`!\n"
            "(`interpolation` is Deprecated in favor of `method`)")
    return interpolation


def _compute_virtual_index(n, quantiles, alpha: float, beta: float):
    """
    Compute the floating point indexes of an array for the linear
    interpolation of quantiles.
    n : array_like
        The sample sizes.
    quantiles : array_like
        The quantiles values.
    alpha : float
        A constant used to correct the index computed.
    beta : float
        A constant used to correct the index computed.

    alpha and beta values depend on the chosen method
    (see quantile documentation)

    Reference:
    Hyndman&Fan paper "Sample Quantiles in Statistical Packages",
    DOI: 10.1080/00031305.1996.10473566
    """
    return n * quantiles + (
            alpha + quantiles * (1 - alpha - beta)
    ) - 1


def _get_gamma(virtual_indexes, previous_indexes, method):
    """
    Compute gamma (a.k.a 'm' or 'weight') for the linear interpolation
    of quantiles.

    virtual_indexes : array_like
        The indexes where the percentile is supposed to be found in the sorted
        sample.
    previous_indexes : array_like
        The floor values of virtual_indexes.
    interpolation : dict
        The interpolation method chosen, which may have a specific rule
        modifying gamma.

    gamma is usually the fractional part of virtual_indexes but can be modified
    by the interpolation method.
    """
    gamma = np.asanyarray(virtual_indexes - previous_indexes)
    gamma = method["fix_gamma"](gamma, virtual_indexes)
    # Ensure both that we have an array, and that we keep the dtype
    # (which may have been matched to the input array).
    return np.asanyarray(gamma, dtype=virtual_indexes.dtype)


def _lerp(a, b, t, out=None):
    """
    Compute the linear interpolation weighted by gamma on each point of
    two same shape array.

    a : array_like
        Left bound.
    b : array_like
        Right bound.
    t : array_like
        The interpolation weight.
    out : array_like
        Output array.
    """
    diff_b_a = subtract(b, a)
    # asanyarray is a stop-gap until gh-13105
    lerp_interpolation = asanyarray(add(a, diff_b_a * t, out=out))
    subtract(b, diff_b_a * (1 - t), out=lerp_interpolation, where=t >= 0.5,
             casting='unsafe', dtype=type(lerp_interpolation.dtype))
    if lerp_interpolation.ndim == 0 and out is None:
        lerp_interpolation = lerp_interpolation[()]  # unpack 0d arrays
    return lerp_interpolation


def _get_gamma_mask(shape, default_value, conditioned_value, where):
    out = np.full(shape, default_value)
    np.copyto(out, conditioned_value, where=where, casting="unsafe")
    return out


def _discrete_interpolation_to_boundaries(index, gamma_condition_fun):
    previous = np.floor(index)
    next = previous + 1
    gamma = index - previous
    res = _get_gamma_mask(shape=index.shape,
                          default_value=next,
                          conditioned_value=previous,
                          where=gamma_condition_fun(gamma, index)
                          ).astype(np.intp)
    # Some methods can lead to out-of-bound integers, clip them:
    res[res < 0] = 0
    return res


def _closest_observation(n, quantiles):
    # "choose the nearest even order statistic at g=0" (H&F (1996) pp. 362).
    # Order is 1-based so for zero-based indexing round to nearest odd index.
    gamma_fun = lambda gamma, index: (gamma == 0) & (np.floor(index) % 2 == 1)
    return _discrete_interpolation_to_boundaries((n * quantiles) - 1 - 0.5,
                                                 gamma_fun)


def _inverted_cdf(n, quantiles):
    gamma_fun = lambda gamma, _: (gamma == 0)
    return _discrete_interpolation_to_boundaries((n * quantiles) - 1,
                                                 gamma_fun)


def _quantile_ureduce_func(
        a: np.array,
        q: np.array,
        weights: np.array,
        axis: int | None = None,
        out=None,
        overwrite_input: bool = False,
        method="linear",
) -> np.array:
    if q.ndim > 2:
        # The code below works fine for nd, but it might not have useful
        # semantics. For now, keep the supported dimensions the same as it was
        # before.
        raise ValueError("q must be a scalar or 1d")
    if overwrite_input:
        if axis is None:
            axis = 0
            arr = a.ravel()
            wgt = None if weights is None else weights.ravel()
        else:
            arr = a
            wgt = weights
    else:
        if axis is None:
            axis = 0
            arr = a.flatten()
            wgt = None if weights is None else weights.flatten()
        else:
            arr = a.copy()
            wgt = weights
    result = _quantile(arr,
                       quantiles=q,
                       axis=axis,
                       method=method,
                       out=out,
                       weights=wgt)
    return result


def _get_indexes(arr, virtual_indexes, valid_values_count):
    """
    Get the valid indexes of arr neighbouring virtual_indexes.
    Note
    This is a companion function to linear interpolation of
    Quantiles

    Returns
    -------
    (previous_indexes, next_indexes): Tuple
        A Tuple of virtual_indexes neighbouring indexes
    """
    previous_indexes = np.asanyarray(np.floor(virtual_indexes))
    next_indexes = np.asanyarray(previous_indexes + 1)
    indexes_above_bounds = virtual_indexes >= valid_values_count - 1
    # When indexes is above max index, take the max value of the array
    if indexes_above_bounds.any():
        previous_indexes[indexes_above_bounds] = -1
        next_indexes[indexes_above_bounds] = -1
    # When indexes is below min index, take the min value of the array
    indexes_below_bounds = virtual_indexes < 0
    if indexes_below_bounds.any():
        previous_indexes[indexes_below_bounds] = 0
        next_indexes[indexes_below_bounds] = 0
    if np.issubdtype(arr.dtype, np.inexact):
        # After the sort, slices having NaNs will have for last element a NaN
        virtual_indexes_nans = np.isnan(virtual_indexes)
        if virtual_indexes_nans.any():
            previous_indexes[virtual_indexes_nans] = -1
            next_indexes[virtual_indexes_nans] = -1
    previous_indexes = previous_indexes.astype(np.intp)
    next_indexes = next_indexes.astype(np.intp)
    return previous_indexes, next_indexes


def _quantile(
        arr: np.array,
        quantiles: np.array,
        axis: int = -1,
        method="linear",
        out=None,
        weights=None,
):
    """
    Private function that doesn't support extended axis or keepdims.
    These methods are extended to this function using _ureduce
    See nanpercentile for parameter usage
    It computes the quantiles of the array for the given axis.
    A linear interpolation is performed based on the `interpolation`.

    By default, the method is "linear" where alpha == beta == 1 which
    performs the 7th method of Hyndman&Fan.
    With "median_unbiased" we get alpha == beta == 1/3
    thus the 8th method of Hyndman&Fan.
    """
    # --- Setup
    arr = np.asanyarray(arr)
    values_count = arr.shape[axis]
    # The dimensions of `q` are prepended to the output shape, so we need the
    # axis being sampled from `arr` to be last.
    if axis != 0:  # But moveaxis is slow, so only call it if necessary.
        arr = np.moveaxis(arr, axis, destination=0)
    supports_nans = (
        np.issubdtype(arr.dtype, np.inexact) or arr.dtype.kind in 'Mm'
    )

    if weights is None:
        # --- Computation of indexes
        # Index where to find the value in the sorted array.
        # Virtual because it is a floating point value, not an valid index.
        # The nearest neighbours are used for interpolation
        try:
            method_props = _QuantileMethods[method]
        except KeyError:
            raise ValueError(
                f"{method!r} is not a valid method. Use one of: "
                f"{_QuantileMethods.keys()}") from None
        virtual_indexes = method_props["get_virtual_index"](values_count,
                                                            quantiles)
        virtual_indexes = np.asanyarray(virtual_indexes)

        if method_props["fix_gamma"] is None:
            supports_integers = True
        else:
            int_virtual_indices = np.issubdtype(virtual_indexes.dtype,
                                                np.integer)
            supports_integers = method == 'linear' and int_virtual_indices

        if supports_integers:
            # No interpolation needed, take the points along axis
            if supports_nans:
                # may contain nan, which would sort to the end
                arr.partition(
                    concatenate((virtual_indexes.ravel(), [-1])), axis=0,
                )
                slices_having_nans = np.isnan(arr[-1, ...])
            else:
                # cannot contain nan
                arr.partition(virtual_indexes.ravel(), axis=0)
                slices_having_nans = np.array(False, dtype=bool)
            result = take(arr, virtual_indexes, axis=0, out=out)
        else:
            previous_indexes, next_indexes = _get_indexes(arr,
                                                          virtual_indexes,
                                                          values_count)
            # --- Sorting
            arr.partition(
                np.unique(np.concatenate(([0, -1],
                                          previous_indexes.ravel(),
                                          next_indexes.ravel(),
                                          ))),
                axis=0)
            if supports_nans:
                slices_having_nans = np.isnan(arr[-1, ...])
            else:
                slices_having_nans = None
            # --- Get values from indexes
            previous = arr[previous_indexes]
            next = arr[next_indexes]
            # --- Linear interpolation
            gamma = _get_gamma(virtual_indexes, previous_indexes, method_props)
            result_shape = virtual_indexes.shape + (1,) * (arr.ndim - 1)
            gamma = gamma.reshape(result_shape)
            result = _lerp(previous,
                        next,
                        gamma,
                        out=out)
    else:
        # Weighted case
        # This implements method="inverted_cdf", the only supported weighted
        # method, which needs to sort anyway.
        weights = np.asanyarray(weights)
        if axis != 0:
            weights = np.moveaxis(weights, axis, destination=0)
        index_array = np.argsort(arr, axis=0, kind="stable")

        # arr = arr[index_array, ...]  # but this adds trailing dimensions of
        # 1.
        arr = np.take_along_axis(arr, index_array, axis=0)
        if weights.shape == arr.shape:
            weights = np.take_along_axis(weights, index_array, axis=0)
        else:
            # weights is 1d
            weights = weights.reshape(-1)[index_array, ...]

        if supports_nans:
            # may contain nan, which would sort to the end
            slices_having_nans = np.isnan(arr[-1, ...])
        else:
            # cannot contain nan
            slices_having_nans = np.array(False, dtype=bool)

        # We use the weights to calculate the empirical cumulative
        # distribution function cdf
        cdf = weights.cumsum(axis=0, dtype=np.float64)
        cdf /= cdf[-1, ...]  # normalization to 1
        # Search index i such that
        #   sum(weights[j], j=0..i-1) < quantile <= sum(weights[j], j=0..i)
        # is then equivalent to
        #   cdf[i-1] < quantile <= cdf[i]
        # Unfortunately, searchsorted only accepts 1-d arrays as first
        # argument, so we will need to iterate over dimensions.

        # Without the following cast, searchsorted can return surprising
        # results, e.g.
        #   np.searchsorted(np.array([0.2, 0.4, 0.6, 0.8, 1.]),
        #                   np.array(0.4, dtype=np.float32), side="left")
        # returns 2 instead of 1 because 0.4 is not binary representable.
        if quantiles.dtype.kind == "f":
            cdf = cdf.astype(quantiles.dtype)
        # Weights must be non-negative, so we might have zero weights at the
        # beginning leading to some leading zeros in cdf. The call to
        # np.searchsorted for quantiles=0 will then pick the first element,
        # but should pick the first one larger than zero. We
        # therefore simply set 0 values in cdf to -1.
        if np.any(cdf[0, ...] == 0):
            cdf[cdf == 0] = -1

        def find_cdf_1d(arr, cdf):
            indices = np.searchsorted(cdf, quantiles, side="left")
            # We might have reached the maximum with i = len(arr), e.g. for
            # quantiles = 1, and need to cut it to len(arr) - 1.
            indices = minimum(indices, values_count - 1)
            result = take(arr, indices, axis=0)
            return result

        r_shape = arr.shape[1:]
        if quantiles.ndim > 0:
            r_shape = quantiles.shape + r_shape
        if out is None:
            result = np.empty_like(arr, shape=r_shape)
        else:
            if out.shape != r_shape:
                msg = (f"Wrong shape of argument 'out', shape={r_shape} is "
                       f"required; got shape={out.shape}.")
                raise ValueError(msg)
            result = out

        # See apply_along_axis, which we do for axis=0. Note that Ni = (,)
        # always, so we remove it here.
        Nk = arr.shape[1:]
        for kk in np.ndindex(Nk):
            result[(...,) + kk] = find_cdf_1d(
                arr[np.s_[:, ] + kk], cdf[np.s_[:, ] + kk]
            )

        # Make result the same as in unweighted inverted_cdf.
        if result.shape == () and result.dtype == np.dtype("O"):
            result = result.item()

    if np.any(slices_having_nans):
        if result.ndim == 0 and out is None:
            # can't write to a scalar, but indexing will be correct
            result = arr[-1]
        else:
            np.copyto(result, arr[-1, ...], where=slices_having_nans)
    return result


def _trapezoid_dispatcher(y, x=None, dx=None, axis=None):
    return (y, x)


@array_function_dispatch(_trapezoid_dispatcher)
def trapezoid(y, x=None, dx=1.0, axis=-1):
    r"""
    Integrate along the given axis using the composite trapezoidal rule.

    If `x` is provided, the integration happens in sequence along its
    elements - they are not sorted.

    Integrate `y` (`x`) along each 1d slice on the given axis, compute
    :math:`\int y(x) dx`.
    When `x` is specified, this integrates along the parametric curve,
    computing :math:`\int_t y(t) dt =
    \int_t y(t) \left.\frac{dx}{dt}\right|_{x=x(t)} dt`.

    .. versionadded:: 2.0.0

    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : array_like, optional
        The sample points corresponding to the `y` values. If `x` is None,
        the sample points are assumed to be evenly spaced `dx` apart. The
        default is None.
    dx : scalar, optional
        The spacing between sample points when `x` is None. The default is 1.
    axis : int, optional
        The axis along which to integrate.

    Returns
    -------
    trapezoid : float or ndarray
        Definite integral of `y` = n-dimensional array as approximated along
        a single axis by the trapezoidal rule. If `y` is a 1-dimensional array,
        then the result is a float. If `n` is greater than 1, then the result
        is an `n`-1 dimensional array.

    See Also
    --------
    sum, cumsum

    Notes
    -----
    Image [2]_ illustrates trapezoidal rule -- y-axis locations of points
    will be taken from `y` array, by default x-axis distances between
    points will be 1.0, alternatively they can be provided with `x` array
    or with `dx` scalar.  Return value will be equal to combined area under
    the red lines.


    References
    ----------
    .. [1] Wikipedia page: https://en.wikipedia.org/wiki/Trapezoidal_rule

    .. [2] Illustration image:
           https://en.wikipedia.org/wiki/File:Composite_trapezoidal_rule_illustration.png

    Examples
    --------
    >>> import numpy as np

    Use the trapezoidal rule on evenly spaced points:

    >>> np.trapezoid([1, 2, 3])
    4.0

    The spacing between sample points can be selected by either the
    ``x`` or ``dx`` arguments:

    >>> np.trapezoid([1, 2, 3], x=[4, 6, 8])
    8.0
    >>> np.trapezoid([1, 2, 3], dx=2)
    8.0

    Using a decreasing ``x`` corresponds to integrating in reverse:

    >>> np.trapezoid([1, 2, 3], x=[8, 6, 4])
    -8.0

    More generally ``x`` is used to integrate along a parametric curve. We can
    estimate the integral :math:`\int_0^1 x^2 = 1/3` using:

    >>> x = np.linspace(0, 1, num=50)
    >>> y = x**2
    >>> np.trapezoid(y, x)
    0.33340274885464394

    Or estimate the area of a circle, noting we repeat the sample which closes
    the curve:

    >>> theta = np.linspace(0, 2 * np.pi, num=1000, endpoint=True)
    >>> np.trapezoid(np.cos(theta), x=np.sin(theta))
    3.141571941375841

    ``np.trapezoid`` can be applied along a specified axis to do multiple
    computations in one call:

    >>> a = np.arange(6).reshape(2, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.trapezoid(a, axis=0)
    array([1.5, 2.5, 3.5])
    >>> np.trapezoid(a, axis=1)
    array([2.,  8.])
    """

    y = asanyarray(y)
    if x is None:
        d = dx
    else:
        x = asanyarray(x)
        if x.ndim == 1:
            d = diff(x)
            # reshape to correct shape
            shape = [1]*y.ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            d = diff(x, axis=axis)
    nd = y.ndim
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    try:
        ret = (d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0).sum(axis)
    except ValueError:
        # Operations didn't work, cast to ndarray
        d = np.asarray(d)
        y = np.asarray(y)
        ret = add.reduce(d * (y[tuple(slice1)]+y[tuple(slice2)])/2.0, axis)
    return ret


@set_module('numpy')
def trapz(y, x=None, dx=1.0, axis=-1):
    """
    `trapz` is deprecated in NumPy 2.0.

    Please use `trapezoid` instead, or one of the numerical integration
    functions in `scipy.integrate`.
    """
    # Deprecated in NumPy 2.0, 2023-08-18
    warnings.warn(
        "`trapz` is deprecated. Use `trapezoid` instead, or one of the "
        "numerical integration functions in `scipy.integrate`.",
        DeprecationWarning,
        stacklevel=2
    )
    return trapezoid(y, x=x, dx=dx, axis=axis)


def _meshgrid_dispatcher(*xi, copy=None, sparse=None, indexing=None):
    return xi


# Based on scitools meshgrid
@array_function_dispatch(_meshgrid_dispatcher)
def meshgrid(*xi, copy=True, sparse=False, indexing='xy'):
    """
    Return a tuple of coordinate matrices from coordinate vectors.

    Make N-D coordinate arrays for vectorized evaluations of
    N-D scalar/vector fields over N-D grids, given
    one-dimensional coordinate arrays x1, x2,..., xn.

    Parameters
    ----------
    x1, x2,..., xn : array_like
        1-D arrays representing the coordinates of a grid.
    indexing : {'xy', 'ij'}, optional
        Cartesian ('xy', default) or matrix ('ij') indexing of output.
        See Notes for more details.
    sparse : bool, optional
        If True the shape of the returned coordinate array for dimension *i*
        is reduced from ``(N1, ..., Ni, ... Nn)`` to
        ``(1, ..., 1, Ni, 1, ..., 1)``.  These sparse coordinate grids are
        intended to be use with :ref:`basics.broadcasting`.  When all
        coordinates are used in an expression, broadcasting still leads to a
        fully-dimensonal result array.

        Default is False.

    copy : bool, optional
        If False, a view into the original arrays are returned in order to
        conserve memory.  Default is True.  Please note that
        ``sparse=False, copy=False`` will likely return non-contiguous
        arrays.  Furthermore, more than one element of a broadcast array
        may refer to a single memory location.  If you need to write to the
        arrays, make copies first.

    Returns
    -------
    X1, X2,..., XN : tuple of ndarrays
        For vectors `x1`, `x2`,..., `xn` with lengths ``Ni=len(xi)``,
        returns ``(N1, N2, N3,..., Nn)`` shaped arrays if indexing='ij'
        or ``(N2, N1, N3,..., Nn)`` shaped arrays if indexing='xy'
        with the elements of `xi` repeated to fill the matrix along
        the first dimension for `x1`, the second for `x2` and so on.

    Notes
    -----
    This function supports both indexing conventions through the indexing
    keyword argument.  Giving the string 'ij' returns a meshgrid with
    matrix indexing, while 'xy' returns a meshgrid with Cartesian indexing.
    In the 2-D case with inputs of length M and N, the outputs are of shape
    (N, M) for 'xy' indexing and (M, N) for 'ij' indexing.  In the 3-D case
    with inputs of length M, N and P, outputs are of shape (N, M, P) for
    'xy' indexing and (M, N, P) for 'ij' indexing.  The difference is
    illustrated by the following code snippet::

        xv, yv = np.meshgrid(x, y, indexing='ij')
        for i in range(nx):
            for j in range(ny):
                # treat xv[i,j], yv[i,j]

        xv, yv = np.meshgrid(x, y, indexing='xy')
        for i in range(nx):
            for j in range(ny):
                # treat xv[j,i], yv[j,i]

    In the 1-D and 0-D case, the indexing and sparse keywords have no effect.

    See Also
    --------
    mgrid : Construct a multi-dimensional "meshgrid" using indexing notation.
    ogrid : Construct an open multi-dimensional "meshgrid" using indexing
            notation.
    :ref:`how-to-index`

    Examples
    --------
    >>> import numpy as np
    >>> nx, ny = (3, 2)
    >>> x = np.linspace(0, 1, nx)
    >>> y = np.linspace(0, 1, ny)
    >>> xv, yv = np.meshgrid(x, y)
    >>> xv
    array([[0. , 0.5, 1. ],
           [0. , 0.5, 1. ]])
    >>> yv
    array([[0.,  0.,  0.],
           [1.,  1.,  1.]])

    The result of `meshgrid` is a coordinate grid:

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(xv, yv, marker='o', color='k', linestyle='none')
    >>> plt.show()

    You can create sparse output arrays to save memory and computation time.

    >>> xv, yv = np.meshgrid(x, y, sparse=True)
    >>> xv
    array([[0. ,  0.5,  1. ]])
    >>> yv
    array([[0.],
           [1.]])

    `meshgrid` is very useful to evaluate functions on a grid. If the
    function depends on all coordinates, both dense and sparse outputs can be
    used.

    >>> x = np.linspace(-5, 5, 101)
    >>> y = np.linspace(-5, 5, 101)
    >>> # full coordinate arrays
    >>> xx, yy = np.meshgrid(x, y)
    >>> zz = np.sqrt(xx**2 + yy**2)
    >>> xx.shape, yy.shape, zz.shape
    ((101, 101), (101, 101), (101, 101))
    >>> # sparse coordinate arrays
    >>> xs, ys = np.meshgrid(x, y, sparse=True)
    >>> zs = np.sqrt(xs**2 + ys**2)
    >>> xs.shape, ys.shape, zs.shape
    ((1, 101), (101, 1), (101, 101))
    >>> np.array_equal(zz, zs)
    True

    >>> h = plt.contourf(x, y, zs)
    >>> plt.axis('scaled')
    >>> plt.colorbar()
    >>> plt.show()
    """
    ndim = len(xi)

    if indexing not in ['xy', 'ij']:
        raise ValueError(
            "Valid values for `indexing` are 'xy' and 'ij'.")

    s0 = (1,) * ndim
    output = [np.asanyarray(x).reshape(s0[:i] + (-1,) + s0[i + 1:])
              for i, x in enumerate(xi)]

    if indexing == 'xy' and ndim > 1:
        # switch first and second axis
        output[0].shape = (1, -1) + s0[2:]
        output[1].shape = (-1, 1) + s0[2:]

    if not sparse:
        # Return the full N-D matrix (not only the 1-D vector)
        output = np.broadcast_arrays(*output, subok=True)

    if copy:
        output = tuple(x.copy() for x in output)

    return output


def _delete_dispatcher(arr, obj, axis=None):
    return (arr, obj)


@array_function_dispatch(_delete_dispatcher)
def delete(arr, obj, axis=None):
    """
    Return a new array with sub-arrays along an axis deleted. For a one
    dimensional array, this returns those entries not returned by
    `arr[obj]`.

    Parameters
    ----------
    arr : array_like
        Input array.
    obj : slice, int, array-like of ints or bools
        Indicate indices of sub-arrays to remove along the specified axis.

        .. versionchanged:: 1.19.0
            Boolean indices are now treated as a mask of elements to remove,
            rather than being cast to the integers 0 and 1.

    axis : int, optional
        The axis along which to delete the subarray defined by `obj`.
        If `axis` is None, `obj` is applied to the flattened array.

    Returns
    -------
    out : ndarray
        A copy of `arr` with the elements specified by `obj` removed. Note
        that `delete` does not occur in-place. If `axis` is None, `out` is
        a flattened array.

    See Also
    --------
    insert : Insert elements into an array.
    append : Append elements at the end of an array.

    Notes
    -----
    Often it is preferable to use a boolean mask. For example:

    >>> arr = np.arange(12) + 1
    >>> mask = np.ones(len(arr), dtype=bool)
    >>> mask[[0,2,4]] = False
    >>> result = arr[mask,...]

    Is equivalent to ``np.delete(arr, [0,2,4], axis=0)``, but allows further
    use of `mask`.

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    >>> arr
    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12]])
    >>> np.delete(arr, 1, 0)
    array([[ 1,  2,  3,  4],
           [ 9, 10, 11, 12]])

    >>> np.delete(arr, np.s_[::2], 1)
    array([[ 2,  4],
           [ 6,  8],
           [10, 12]])
    >>> np.delete(arr, [1,3,5], None)
    array([ 1,  3,  5,  7,  8,  9, 10, 11, 12])

    """
    conv = _array_converter(arr)
    arr, = conv.as_arrays(subok=False)

    ndim = arr.ndim
    arrorder = 'F' if arr.flags.fnc else 'C'
    if axis is None:
        if ndim != 1:
            arr = arr.ravel()
        # needed for np.matrix, which is still not 1d after being ravelled
        ndim = arr.ndim
        axis = ndim - 1
    else:
        axis = normalize_axis_index(axis, ndim)

    slobj = [slice(None)]*ndim
    N = arr.shape[axis]
    newshape = list(arr.shape)

    if isinstance(obj, slice):
        start, stop, step = obj.indices(N)
        xr = range(start, stop, step)
        numtodel = len(xr)

        if numtodel <= 0:
            return conv.wrap(arr.copy(order=arrorder), to_scalar=False)

        # Invert if step is negative:
        if step < 0:
            step = -step
            start = xr[-1]
            stop = xr[0] + 1

        newshape[axis] -= numtodel
        new = empty(newshape, arr.dtype, arrorder)
        # copy initial chunk
        if start == 0:
            pass
        else:
            slobj[axis] = slice(None, start)
            new[tuple(slobj)] = arr[tuple(slobj)]
        # copy end chunk
        if stop == N:
            pass
        else:
            slobj[axis] = slice(stop-numtodel, None)
            slobj2 = [slice(None)]*ndim
            slobj2[axis] = slice(stop, None)
            new[tuple(slobj)] = arr[tuple(slobj2)]
        # copy middle pieces
        if step == 1:
            pass
        else:  # use array indexing.
            keep = ones(stop-start, dtype=bool)
            keep[:stop-start:step] = False
            slobj[axis] = slice(start, stop-numtodel)
            slobj2 = [slice(None)]*ndim
            slobj2[axis] = slice(start, stop)
            arr = arr[tuple(slobj2)]
            slobj2[axis] = keep
            new[tuple(slobj)] = arr[tuple(slobj2)]

        return conv.wrap(new, to_scalar=False)

    if isinstance(obj, (int, integer)) and not isinstance(obj, bool):
        single_value = True
    else:
        single_value = False
        _obj = obj
        obj = np.asarray(obj)
        # `size == 0` to allow empty lists similar to indexing, but (as there)
        # is really too generic:
        if obj.size == 0 and not isinstance(_obj, np.ndarray):
            obj = obj.astype(intp)
        elif obj.size == 1 and obj.dtype.kind in "ui":
            # For a size 1 integer array we can use the single-value path
            # (most dtypes, except boolean, should just fail later).
            obj = obj.item()
            single_value = True

    if single_value:
        # optimization for a single value
        if (obj < -N or obj >= N):
            raise IndexError(
                "index %i is out of bounds for axis %i with "
                "size %i" % (obj, axis, N))
        if (obj < 0):
            obj += N
        newshape[axis] -= 1
        new = empty(newshape, arr.dtype, arrorder)
        slobj[axis] = slice(None, obj)
        new[tuple(slobj)] = arr[tuple(slobj)]
        slobj[axis] = slice(obj, None)
        slobj2 = [slice(None)]*ndim
        slobj2[axis] = slice(obj+1, None)
        new[tuple(slobj)] = arr[tuple(slobj2)]
    else:
        if obj.dtype == bool:
            if obj.shape != (N,):
                raise ValueError('boolean array argument obj to delete '
                                 'must be one dimensional and match the axis '
                                 'length of {}'.format(N))

            # optimization, the other branch is slower
            keep = ~obj
        else:
            keep = ones(N, dtype=bool)
            keep[obj,] = False

        slobj[axis] = keep
        new = arr[tuple(slobj)]

    return conv.wrap(new, to_scalar=False)


def _insert_dispatcher(arr, obj, values, axis=None):
    return (arr, obj, values)


@array_function_dispatch(_insert_dispatcher)
def insert(arr, obj, values, axis=None):
    """
    Insert values along the given axis before the given indices.

    Parameters
    ----------
    arr : array_like
        Input array.
    obj : slice, int, array-like of ints or bools
        Object that defines the index or indices before which `values` is
        inserted.

        .. versionchanged:: 2.1.2
            Boolean indices are now treated as a mask of elements to insert,
            rather than being cast to the integers 0 and 1.

        Support for multiple insertions when `obj` is a single scalar or a
        sequence with one element (similar to calling insert multiple
        times).
    values : array_like
        Values to insert into `arr`. If the type of `values` is different
        from that of `arr`, `values` is converted to the type of `arr`.
        `values` should be shaped so that ``arr[...,obj,...] = values``
        is legal.
    axis : int, optional
        Axis along which to insert `values`.  If `axis` is None then `arr`
        is flattened first.

    Returns
    -------
    out : ndarray
        A copy of `arr` with `values` inserted.  Note that `insert`
        does not occur in-place: a new array is returned. If
        `axis` is None, `out` is a flattened array.

    See Also
    --------
    append : Append elements at the end of an array.
    concatenate : Join a sequence of arrays along an existing axis.
    delete : Delete elements from an array.

    Notes
    -----
    Note that for higher dimensional inserts ``obj=0`` behaves very different
    from ``obj=[0]`` just like ``arr[:,0,:] = values`` is different from
    ``arr[:,[0],:] = values``. This is because of the difference between basic
    and advanced :ref:`indexing <basics.indexing>`.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(6).reshape(3, 2)
    >>> a
    array([[0, 1],
           [2, 3],
           [4, 5]])
    >>> np.insert(a, 1, 6)
    array([0, 6, 1, 2, 3, 4, 5])
    >>> np.insert(a, 1, 6, axis=1)
    array([[0, 6, 1],
           [2, 6, 3],
           [4, 6, 5]])

    Difference between sequence and scalars,
    showing how ``obj=[1]`` behaves different from ``obj=1``:

    >>> np.insert(a, [1], [[7],[8],[9]], axis=1)
    array([[0, 7, 1],
           [2, 8, 3],
           [4, 9, 5]])
    >>> np.insert(a, 1, [[7],[8],[9]], axis=1)
    array([[0, 7, 8, 9, 1],
           [2, 7, 8, 9, 3],
           [4, 7, 8, 9, 5]])
    >>> np.array_equal(np.insert(a, 1, [7, 8, 9], axis=1),
    ...                np.insert(a, [1], [[7],[8],[9]], axis=1))
    True

    >>> b = a.flatten()
    >>> b
    array([0, 1, 2, 3, 4, 5])
    >>> np.insert(b, [2, 2], [6, 7])
    array([0, 1, 6, 7, 2, 3, 4, 5])

    >>> np.insert(b, slice(2, 4), [7, 8])
    array([0, 1, 7, 2, 8, 3, 4, 5])

    >>> np.insert(b, [2, 2], [7.13, False]) # type casting
    array([0, 1, 7, 0, 2, 3, 4, 5])

    >>> x = np.arange(8).reshape(2, 4)
    >>> idx = (1, 3)
    >>> np.insert(x, idx, 999, axis=1)
    array([[  0, 999,   1,   2, 999,   3],
           [  4, 999,   5,   6, 999,   7]])

    """
    conv = _array_converter(arr)
    arr, = conv.as_arrays(subok=False)

    ndim = arr.ndim
    arrorder = 'F' if arr.flags.fnc else 'C'
    if axis is None:
        if ndim != 1:
            arr = arr.ravel()
        # needed for np.matrix, which is still not 1d after being ravelled
        ndim = arr.ndim
        axis = ndim - 1
    else:
        axis = normalize_axis_index(axis, ndim)
    slobj = [slice(None)]*ndim
    N = arr.shape[axis]
    newshape = list(arr.shape)

    if isinstance(obj, slice):
        # turn it into a range object
        indices = arange(*obj.indices(N), dtype=intp)
    else:
        # need to copy obj, because indices will be changed in-place
        indices = np.array(obj)
        if indices.dtype == bool:
            if obj.ndim != 1:
                raise ValueError('boolean array argument obj to insert '
                                'must be one dimensional')
            indices = np.flatnonzero(obj)
        elif indices.ndim > 1:
            raise ValueError(
                "index array argument obj to insert must be one dimensional "
                "or scalar")
    if indices.size == 1:
        index = indices.item()
        if index < -N or index > N:
            raise IndexError(f"index {obj} is out of bounds for axis {axis} "
                             f"with size {N}")
        if (index < 0):
            index += N

        # There are some object array corner cases here, but we cannot avoid
        # that:
        values = array(values, copy=None, ndmin=arr.ndim, dtype=arr.dtype)
        if indices.ndim == 0:
            # broadcasting is very different here, since a[:,0,:] = ... behaves
            # very different from a[:,[0],:] = ...! This changes values so that
            # it works likes the second case. (here a[:,0:1,:])
            values = np.moveaxis(values, 0, axis)
        numnew = values.shape[axis]
        newshape[axis] += numnew
        new = empty(newshape, arr.dtype, arrorder)
        slobj[axis] = slice(None, index)
        new[tuple(slobj)] = arr[tuple(slobj)]
        slobj[axis] = slice(index, index+numnew)
        new[tuple(slobj)] = values
        slobj[axis] = slice(index+numnew, None)
        slobj2 = [slice(None)] * ndim
        slobj2[axis] = slice(index, None)
        new[tuple(slobj)] = arr[tuple(slobj2)]

        return conv.wrap(new, to_scalar=False)

    elif indices.size == 0 and not isinstance(obj, np.ndarray):
        # Can safely cast the empty list to intp
        indices = indices.astype(intp)

    indices[indices < 0] += N

    numnew = len(indices)
    order = indices.argsort(kind='mergesort')   # stable sort
    indices[order] += np.arange(numnew)

    newshape[axis] += numnew
    old_mask = ones(newshape[axis], dtype=bool)
    old_mask[indices] = False

    new = empty(newshape, arr.dtype, arrorder)
    slobj2 = [slice(None)]*ndim
    slobj[axis] = indices
    slobj2[axis] = old_mask
    new[tuple(slobj)] = values
    new[tuple(slobj2)] = arr

    return conv.wrap(new, to_scalar=False)


def _append_dispatcher(arr, values, axis=None):
    return (arr, values)


@array_function_dispatch(_append_dispatcher)
def append(arr, values, axis=None):
    """
    Append values to the end of an array.

    Parameters
    ----------
    arr : array_like
        Values are appended to a copy of this array.
    values : array_like
        These values are appended to a copy of `arr`.  It must be of the
        correct shape (the same shape as `arr`, excluding `axis`).  If
        `axis` is not specified, `values` can be any shape and will be
        flattened before use.
    axis : int, optional
        The axis along which `values` are appended.  If `axis` is not
        given, both `arr` and `values` are flattened before use.

    Returns
    -------
    append : ndarray
        A copy of `arr` with `values` appended to `axis`.  Note that
        `append` does not occur in-place: a new array is allocated and
        filled.  If `axis` is None, `out` is a flattened array.

    See Also
    --------
    insert : Insert elements into an array.
    delete : Delete elements from an array.

    Examples
    --------
    >>> import numpy as np
    >>> np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]])
    array([1, 2, 3, ..., 7, 8, 9])

    When `axis` is specified, `values` must have the correct shape.

    >>> np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)
    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])

    >>> np.append([[1, 2, 3], [4, 5, 6]], [7, 8, 9], axis=0)
    Traceback (most recent call last):
        ...
    ValueError: all the input arrays must have same number of dimensions, but
    the array at index 0 has 2 dimension(s) and the array at index 1 has 1
    dimension(s)

    >>> a = np.array([1, 2], dtype=int)
    >>> c = np.append(a, [])
    >>> c
    array([1., 2.])
    >>> c.dtype
    float64

    Default dtype for empty ndarrays is `float64` thus making the output of dtype
    `float64` when appended with dtype `int64`

    """
    arr = asanyarray(arr)
    if axis is None:
        if arr.ndim != 1:
            arr = arr.ravel()
        values = ravel(values)
        axis = arr.ndim-1
    return concatenate((arr, values), axis=axis)


def _digitize_dispatcher(x, bins, right=None):
    return (x, bins)


@array_function_dispatch(_digitize_dispatcher)
def digitize(x, bins, right=False):
    """
    Return the indices of the bins to which each value in input array belongs.

    =========  =============  ============================
    `right`    order of bins  returned index `i` satisfies
    =========  =============  ============================
    ``False``  increasing     ``bins[i-1] <= x < bins[i]``
    ``True``   increasing     ``bins[i-1] < x <= bins[i]``
    ``False``  decreasing     ``bins[i-1] > x >= bins[i]``
    ``True``   decreasing     ``bins[i-1] >= x > bins[i]``
    =========  =============  ============================

    If values in `x` are beyond the bounds of `bins`, 0 or ``len(bins)`` is
    returned as appropriate.

    Parameters
    ----------
    x : array_like
        Input array to be binned. Prior to NumPy 1.10.0, this array had to
        be 1-dimensional, but can now have any shape.
    bins : array_like
        Array of bins. It has to be 1-dimensional and monotonic.
    right : bool, optional
        Indicating whether the intervals include the right or the left bin
        edge. Default behavior is (right==False) indicating that the interval
        does not include the right edge. The left bin end is open in this
        case, i.e., bins[i-1] <= x < bins[i] is the default behavior for
        monotonically increasing bins.

    Returns
    -------
    indices : ndarray of ints
        Output array of indices, of same shape as `x`.

    Raises
    ------
    ValueError
        If `bins` is not monotonic.
    TypeError
        If the type of the input is complex.

    See Also
    --------
    bincount, histogram, unique, searchsorted

    Notes
    -----
    If values in `x` are such that they fall outside the bin range,
    attempting to index `bins` with the indices that `digitize` returns
    will result in an IndexError.

    .. versionadded:: 1.10.0

    `numpy.digitize` is  implemented in terms of `numpy.searchsorted`.
    This means that a binary search is used to bin the values, which scales
    much better for larger number of bins than the previous linear search.
    It also removes the requirement for the input array to be 1-dimensional.

    For monotonically *increasing* `bins`, the following are equivalent::

        np.digitize(x, bins, right=True)
        np.searchsorted(bins, x, side='left')

    Note that as the order of the arguments are reversed, the side must be too.
    The `searchsorted` call is marginally faster, as it does not do any
    monotonicity checks. Perhaps more importantly, it supports all dtypes.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0.2, 6.4, 3.0, 1.6])
    >>> bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
    >>> inds = np.digitize(x, bins)
    >>> inds
    array([1, 4, 3, 2])
    >>> for n in range(x.size):
    ...   print(bins[inds[n]-1], "<=", x[n], "<", bins[inds[n]])
    ...
    0.0 <= 0.2 < 1.0
    4.0 <= 6.4 < 10.0
    2.5 <= 3.0 < 4.0
    1.0 <= 1.6 < 2.5

    >>> x = np.array([1.2, 10.0, 12.4, 15.5, 20.])
    >>> bins = np.array([0, 5, 10, 15, 20])
    >>> np.digitize(x,bins,right=True)
    array([1, 2, 3, 4, 4])
    >>> np.digitize(x,bins,right=False)
    array([1, 3, 3, 4, 5])
    """
    x = _nx.asarray(x)
    bins = _nx.asarray(bins)

    # here for compatibility, searchsorted below is happy to take this
    if np.issubdtype(x.dtype, _nx.complexfloating):
        raise TypeError("x may not be complex")

    mono = _monotonicity(bins)
    if mono == 0:
        raise ValueError("bins must be monotonically increasing or decreasing")

    # this is backwards because the arguments below are swapped
    side = 'left' if right else 'right'
    if mono == -1:
        # reverse the bins, and invert the results
        return len(bins) - _nx.searchsorted(bins[::-1], x, side=side)
    else:
        return _nx.searchsorted(bins, x, side=side)
<|endoftext|>"""
Introspection helper functions.
"""
import re

__all__ = ['opt_func_info']


def opt_func_info(func_name=None, signature=None):
    """
    Returns a dictionary containing the currently supported CPU dispatched
    features for all optimized functions.

    Parameters
    ----------
    func_name : str (optional)
        Regular expression to filter by function name.

    signature : str (optional)
        Regular expression to filter by data type.

    Returns
    -------
    dict
        A dictionary where keys are optimized function names and values are
        nested dictionaries indicating supported targets based on data types.

    Examples
    --------
    Retrieve dispatch information for functions named 'add' or 'sub' and
    data types 'float64' or 'float32':

    >>> import numpy as np
    >>> dict = np.lib.introspect.opt_func_info(
    ...     func_name="add|abs", signature="float64|complex64"
    ... )
    >>> import json
    >>> print(json.dumps(dict, indent=2))
        {
          "absolute": {
            "dd": {
              "current": "SSE41",
              "available": "SSE41 baseline(SSE SSE2 SSE3)"
            },
            "Ff": {
              "current": "FMA3__AVX2",
              "available": "AVX512F FMA3__AVX2 baseline(SSE SSE2 SSE3)"
            },
            "Dd": {
              "current": "FMA3__AVX2",
              "available": "AVX512F FMA3__AVX2 baseline(SSE SSE2 SSE3)"
            }
          },
          "add": {
            "ddd": {
              "current": "FMA3__AVX2",
              "available": "FMA3__AVX2 baseline(SSE SSE2 SSE3)"
            },
            "FFF": {
              "current": "FMA3__AVX2",
              "available": "FMA3__AVX2 baseline(SSE SSE2 SSE3)"
            }
          }
        }

    """
    from numpy._core._multiarray_umath import (
        __cpu_targets_info__ as targets, dtype
    )

    if func_name is not None:
        func_pattern = re.compile(func_name)
        matching_funcs = {
            k: v for k, v in targets.items()
            if func_pattern.search(k)
        }
    else:
        matching_funcs = targets

    if signature is not None:
        sig_pattern = re.compile(signature)
        matching_sigs = {}
        for k, v in matching_funcs.items():
            matching_chars = {}
            for chars, targets in v.items():
                if any(
                    sig_pattern.search(c) or sig_pattern.search(dtype(c).name)
                    for c in chars
                ):
                    matching_chars[chars] = targets
            if matching_chars:
                matching_sigs[k] = matching_chars
    else:
        matching_sigs = matching_funcs
    return matching_sigs
<|endoftext|>"""
Functions to operate on polynomials.

"""
__all__ = ['poly', 'roots', 'polyint', 'polyder', 'polyadd',
           'polysub', 'polymul', 'polydiv', 'polyval', 'poly1d',
           'polyfit']

import functools
import re
import warnings

from .._utils import set_module
import numpy._core.numeric as NX

from numpy._core import (isscalar, abs, finfo, atleast_1d, hstack, dot, array,
                        ones)
from numpy._core import overrides
from numpy.exceptions import RankWarning
from numpy.lib._twodim_base_impl import diag, vander
from numpy.lib._function_base_impl import trim_zeros
from numpy.lib._type_check_impl import iscomplex, real, imag, mintypecode
from numpy.linalg import eigvals, lstsq, inv


array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


def _poly_dispatcher(seq_of_zeros):
    return seq_of_zeros


@array_function_dispatch(_poly_dispatcher)
def poly(seq_of_zeros):
    """
    Find the coefficients of a polynomial with the given sequence of roots.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    Returns the coefficients of the polynomial whose leading coefficient
    is one for the given sequence of zeros (multiple roots must be included
    in the sequence as many times as their multiplicity; see Examples).
    A square matrix (or array, which will be treated as a matrix) can also
    be given, in which case the coefficients of the characteristic polynomial
    of the matrix are returned.

    Parameters
    ----------
    seq_of_zeros : array_like, shape (N,) or (N, N)
        A sequence of polynomial roots, or a square array or matrix object.

    Returns
    -------
    c : ndarray
        1D array of polynomial coefficients from highest to lowest degree:

        ``c[0] * x**(N) + c[1] * x**(N-1) + ... + c[N-1] * x + c[N]``
        where c[0] always equals 1.

    Raises
    ------
    ValueError
        If input is the wrong shape (the input must be a 1-D or square
        2-D array).

    See Also
    --------
    polyval : Compute polynomial values.
    roots : Return the roots of a polynomial.
    polyfit : Least squares polynomial fit.
    poly1d : A one-dimensional polynomial class.

    Notes
    -----
    Specifying the roots of a polynomial still leaves one degree of
    freedom, typically represented by an undetermined leading
    coefficient. [1]_ In the case of this function, that coefficient -
    the first one in the returned array - is always taken as one. (If
    for some reason you have one other point, the only automatic way
    presently to leverage that information is to use ``polyfit``.)

    The characteristic polynomial, :math:`p_a(t)`, of an `n`-by-`n`
    matrix **A** is given by

    :math:`p_a(t) = \\mathrm{det}(t\\, \\mathbf{I} - \\mathbf{A})`,

    where **I** is the `n`-by-`n` identity matrix. [2]_

    References
    ----------
    .. [1] M. Sullivan and M. Sullivan, III, "Algebra and Trigonometry,
       Enhanced With Graphing Utilities," Prentice-Hall, pg. 318, 1996.

    .. [2] G. Strang, "Linear Algebra and Its Applications, 2nd Edition,"
       Academic Press, pg. 182, 1980.

    Examples
    --------
    Given a sequence of a polynomial's zeros:

    >>> import numpy as np

    >>> np.poly((0, 0, 0)) # Multiple root example
    array([1., 0., 0., 0.])

    The line above represents z**3 + 0*z**2 + 0*z + 0.

    >>> np.poly((-1./2, 0, 1./2))
    array([ 1.  ,  0.  , -0.25,  0.  ])

    The line above represents z**3 - z/4

    >>> np.poly((np.random.random(1)[0], 0, np.random.random(1)[0]))
    array([ 1.        , -0.77086955,  0.08618131,  0.        ]) # random

    Given a square array object:

    >>> P = np.array([[0, 1./3], [-1./2, 0]])
    >>> np.poly(P)
    array([1.        , 0.        , 0.16666667])

    Note how in all cases the leading coefficient is always 1.

    """
    seq_of_zeros = atleast_1d(seq_of_zeros)
    sh = seq_of_zeros.shape

    if len(sh) == 2 and sh[0] == sh[1] and sh[0] != 0:
        seq_of_zeros = eigvals(seq_of_zeros)
    elif len(sh) == 1:
        dt = seq_of_zeros.dtype
        # Let object arrays slip through, e.g. for arbitrary precision
        if dt != object:
            seq_of_zeros = seq_of_zeros.astype(mintypecode(dt.char))
    else:
        raise ValueError("input must be 1d or non-empty square 2d array.")

    if len(seq_of_zeros) == 0:
        return 1.0
    dt = seq_of_zeros.dtype
    a = ones((1,), dtype=dt)
    for zero in seq_of_zeros:
        a = NX.convolve(a, array([1, -zero], dtype=dt), mode='full')

    if issubclass(a.dtype.type, NX.complexfloating):
        # if complex roots are all complex conjugates, the roots are real.
        roots = NX.asarray(seq_of_zeros, complex)
        if NX.all(NX.sort(roots) == NX.sort(roots.conjugate())):
            a = a.real.copy()

    return a


def _roots_dispatcher(p):
    return p


@array_function_dispatch(_roots_dispatcher)
def roots(p):
    """
    Return the roots of a polynomial with coefficients given in p.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    The values in the rank-1 array `p` are coefficients of a polynomial.
    If the length of `p` is n+1 then the polynomial is described by::

      p[0] * x**n + p[1] * x**(n-1) + ... + p[n-1]*x + p[n]

    Parameters
    ----------
    p : array_like
        Rank-1 array of polynomial coefficients.

    Returns
    -------
    out : ndarray
        An array containing the roots of the polynomial.

    Raises
    ------
    ValueError
        When `p` cannot be converted to a rank-1 array.

    See also
    --------
    poly : Find the coefficients of a polynomial with a given sequence
           of roots.
    polyval : Compute polynomial values.
    polyfit : Least squares polynomial fit.
    poly1d : A one-dimensional polynomial class.

    Notes
    -----
    The algorithm relies on computing the eigenvalues of the
    companion matrix [1]_.

    References
    ----------
    .. [1] R. A. Horn & C. R. Johnson, *Matrix Analysis*.  Cambridge, UK:
        Cambridge University Press, 1999, pp. 146-7.

    Examples
    --------
    >>> import numpy as np
    >>> coeff = [3.2, 2, 1]
    >>> np.roots(coeff)
    array([-0.3125+0.46351241j, -0.3125-0.46351241j])

    """
    # If input is scalar, this makes it an array
    p = atleast_1d(p)
    if p.ndim != 1:
        raise ValueError("Input must be a rank-1 array.")

    # find non-zero array entries
    non_zero = NX.nonzero(NX.ravel(p))[0]

    # Return an empty array if polynomial is all zeros
    if len(non_zero) == 0:
        return NX.array([])

    # find the number of trailing zeros -- this is the number of roots at 0.
    trailing_zeros = len(p) - non_zero[-1] - 1

    # strip leading and trailing zeros
    p = p[int(non_zero[0]):int(non_zero[-1])+1]

    # casting: if incoming array isn't floating point, make it floating point.
    if not issubclass(p.dtype.type, (NX.floating, NX.complexfloating)):
        p = p.astype(float)

    N = len(p)
    if N > 1:
        # build companion matrix and find its eigenvalues (the roots)
        A = diag(NX.ones((N-2,), p.dtype), -1)
        A[0,:] = -p[1:] / p[0]
        roots = eigvals(A)
    else:
        roots = NX.array([])

    # tack any zeros onto the back of the array
    roots = hstack((roots, NX.zeros(trailing_zeros, roots.dtype)))
    return roots


def _polyint_dispatcher(p, m=None, k=None):
    return (p,)


@array_function_dispatch(_polyint_dispatcher)
def polyint(p, m=1, k=None):
    """
    Return an antiderivative (indefinite integral) of a polynomial.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    The returned order `m` antiderivative `P` of polynomial `p` satisfies
    :math:`\\frac{d^m}{dx^m}P(x) = p(x)` and is defined up to `m - 1`
    integration constants `k`. The constants determine the low-order
    polynomial part

    .. math:: \\frac{k_{m-1}}{0!} x^0 + \\ldots + \\frac{k_0}{(m-1)!}x^{m-1}

    of `P` so that :math:`P^{(j)}(0) = k_{m-j-1}`.

    Parameters
    ----------
    p : array_like or poly1d
        Polynomial to integrate.
        A sequence is interpreted as polynomial coefficients, see `poly1d`.
    m : int, optional
        Order of the antiderivative. (Default: 1)
    k : list of `m` scalars or scalar, optional
        Integration constants. They are given in the order of integration:
        those corresponding to highest-order terms come first.

        If ``None`` (default), all constants are assumed to be zero.
        If `m = 1`, a single scalar can be given instead of a list.

    See Also
    --------
    polyder : derivative of a polynomial
    poly1d.integ : equivalent method

    Examples
    --------
    The defining property of the antiderivative:

    >>> import numpy as np

    >>> p = np.poly1d([1,1,1])
    >>> P = np.polyint(p)
    >>> P
     poly1d([ 0.33333333,  0.5       ,  1.        ,  0.        ]) # may vary
    >>> np.polyder(P) == p
    True

    The integration constants default to zero, but can be specified:

    >>> P = np.polyint(p, 3)
    >>> P(0)
    0.0
    >>> np.polyder(P)(0)
    0.0
    >>> np.polyder(P, 2)(0)
    0.0
    >>> P = np.polyint(p, 3, k=[6,5,3])
    >>> P
    poly1d([ 0.01666667,  0.04166667,  0.16666667,  3. ,  5. ,  3. ]) # may vary

    Note that 3 = 6 / 2!, and that the constants are given in the order of
    integrations. Constant of the highest-order polynomial term comes first:

    >>> np.polyder(P, 2)(0)
    6.0
    >>> np.polyder(P, 1)(0)
    5.0
    >>> P(0)
    3.0

    """
    m = int(m)
    if m < 0:
        raise ValueError("Order of integral must be positive (see polyder)")
    if k is None:
        k = NX.zeros(m, float)
    k = atleast_1d(k)
    if len(k) == 1 and m > 1:
        k = k[0]*NX.ones(m, float)
    if len(k) < m:
        raise ValueError(
              "k must be a scalar or a rank-1 array of length 1 or >m.")

    truepoly = isinstance(p, poly1d)
    p = NX.asarray(p)
    if m == 0:
        if truepoly:
            return poly1d(p)
        return p
    else:
        # Note: this must work also with object and integer arrays
        y = NX.concatenate((p.__truediv__(NX.arange(len(p), 0, -1)), [k[0]]))
        val = polyint(y, m - 1, k=k[1:])
        if truepoly:
            return poly1d(val)
        return val


def _polyder_dispatcher(p, m=None):
    return (p,)


@array_function_dispatch(_polyder_dispatcher)
def polyder(p, m=1):
    """
    Return the derivative of the specified order of a polynomial.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    Parameters
    ----------
    p : poly1d or sequence
        Polynomial to differentiate.
        A sequence is interpreted as polynomial coefficients, see `poly1d`.
    m : int, optional
        Order of differentiation (default: 1)

    Returns
    -------
    der : poly1d
        A new polynomial representing the derivative.

    See Also
    --------
    polyint : Anti-derivative of a polynomial.
    poly1d : Class for one-dimensional polynomials.

    Examples
    --------
    The derivative of the polynomial :math:`x^3 + x^2 + x^1 + 1` is:

    >>> import numpy as np

    >>> p = np.poly1d([1,1,1,1])
    >>> p2 = np.polyder(p)
    >>> p2
    poly1d([3, 2, 1])

    which evaluates to:

    >>> p2(2.)
    17.0

    We can verify this, approximating the derivative with
    ``(f(x + h) - f(x))/h``:

    >>> (p(2. + 0.001) - p(2.)) / 0.001
    17.007000999997857

    The fourth-order derivative of a 3rd-order polynomial is zero:

    >>> np.polyder(p, 2)
    poly1d([6, 2])
    >>> np.polyder(p, 3)
    poly1d([6])
    >>> np.polyder(p, 4)
    poly1d([0])

    """
    m = int(m)
    if m < 0:
        raise ValueError("Order of derivative must be positive (see polyint)")

    truepoly = isinstance(p, poly1d)
    p = NX.asarray(p)
    n = len(p) - 1
    y = p[:-1] * NX.arange(n, 0, -1)
    if m == 0:
        val = p
    else:
        val = polyder(y, m - 1)
    if truepoly:
        val = poly1d(val)
    return val


def _polyfit_dispatcher(x, y, deg, rcond=None, full=None, w=None, cov=None):
    return (x, y, w)


@array_function_dispatch(_polyfit_dispatcher)
def polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False):
    """
    Least squares polynomial fit.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    Fit a polynomial ``p(x) = p[0] * x**deg + ... + p[deg]`` of degree `deg`
    to points `(x, y)`. Returns a vector of coefficients `p` that minimises
    the squared error in the order `deg`, `deg-1`, ... `0`.

    The `Polynomial.fit <numpy.polynomial.polynomial.Polynomial.fit>` class
    method is recommended for new code as it is more stable numerically. See
    the documentation of the method for more information.

    Parameters
    ----------
    x : array_like, shape (M,)
        x-coordinates of the M sample points ``(x[i], y[i])``.
    y : array_like, shape (M,) or (M, K)
        y-coordinates of the sample points. Several data sets of sample
        points sharing the same x-coordinates can be fitted at once by
        passing in a 2D-array that contains one dataset per column.
    deg : int
        Degree of the fitting polynomial
    rcond : float, optional
        Relative condition number of the fit. Singular values smaller than
        this relative to the largest singular value will be ignored. The
        default value is len(x)*eps, where eps is the relative precision of
        the float type, about 2e-16 in most cases.
    full : bool, optional
        Switch determining nature of return value. When it is False (the
        default) just the coefficients are returned, when True diagnostic
        information from the singular value decomposition is also returned.
    w : array_like, shape (M,), optional
        Weights. If not None, the weight ``w[i]`` applies to the unsquared
        residual ``y[i] - y_hat[i]`` at ``x[i]``. Ideally the weights are
        chosen so that the errors of the products ``w[i]*y[i]`` all have the
        same variance.  When using inverse-variance weighting, use
        ``w[i] = 1/sigma(y[i])``.  The default value is None.
    cov : bool or str, optional
        If given and not `False`, return not just the estimate but also its
        covariance matrix. By default, the covariance are scaled by
        chi2/dof, where dof = M - (deg + 1), i.e., the weights are presumed
        to be unreliable except in a relative sense and everything is scaled
        such that the reduced chi2 is unity. This scaling is omitted if
        ``cov='unscaled'``, as is relevant for the case that the weights are
        w = 1/sigma, with sigma known to be a reliable estimate of the
        uncertainty.

    Returns
    -------
    p : ndarray, shape (deg + 1,) or (deg + 1, K)
        Polynomial coefficients, highest power first.  If `y` was 2-D, the
        coefficients for `k`-th data set are in ``p[:,k]``.

    residuals, rank, singular_values, rcond
        These values are only returned if ``full == True``

        - residuals -- sum of squared residuals of the least squares fit
        - rank -- the effective rank of the scaled Vandermonde
           coefficient matrix
        - singular_values -- singular values of the scaled Vandermonde
           coefficient matrix
        - rcond -- value of `rcond`.

        For more details, see `numpy.linalg.lstsq`.

    V : ndarray, shape (deg + 1, deg + 1) or (deg + 1, deg + 1, K)
        Present only if ``full == False`` and ``cov == True``.  The covariance
        matrix of the polynomial coefficient estimates.  The diagonal of
        this matrix are the variance estimates for each coefficient.  If y
        is a 2-D array, then the covariance matrix for the `k`-th data set
        are in ``V[:,:,k]``


    Warns
    -----
    RankWarning
        The rank of the coefficient matrix in the least-squares fit is
        deficient. The warning is only raised if ``full == False``.

        The warnings can be turned off by

        >>> import warnings
        >>> warnings.simplefilter('ignore', np.exceptions.RankWarning)

    See Also
    --------
    polyval : Compute polynomial values.
    linalg.lstsq : Computes a least-squares fit.
    scipy.interpolate.UnivariateSpline : Computes spline fits.

    Notes
    -----
    The solution minimizes the squared error

    .. math::
        E = \\sum_{j=0}^k |p(x_j) - y_j|^2

    in the equations::

        x[0]**n * p[0] + ... + x[0] * p[n-1] + p[n] = y[0]
        x[1]**n * p[0] + ... + x[1] * p[n-1] + p[n] = y[1]
        ...
        x[k]**n * p[0] + ... + x[k] * p[n-1] + p[n] = y[k]

    The coefficient matrix of the coefficients `p` is a Vandermonde matrix.

    `polyfit` issues a `~exceptions.RankWarning` when the least-squares fit is
    badly conditioned. This implies that the best fit is not well-defined due
    to numerical error. The results may be improved by lowering the polynomial
    degree or by replacing `x` by `x` - `x`.mean(). The `rcond` parameter
    can also be set to a value smaller than its default, but the resulting
    fit may be spurious: including contributions from the small singular
    values can add numerical noise to the result.

    Note that fitting polynomial coefficients is inherently badly conditioned
    when the degree of the polynomial is large or the interval of sample points
    is badly centered. The quality of the fit should always be checked in these
    cases. When polynomial fits are not satisfactory, splines may be a good
    alternative.

    References
    ----------
    .. [1] Wikipedia, "Curve fitting",
           https://en.wikipedia.org/wiki/Curve_fitting
    .. [2] Wikipedia, "Polynomial interpolation",
           https://en.wikipedia.org/wiki/Polynomial_interpolation

    Examples
    --------
    >>> import numpy as np
    >>> import warnings
    >>> x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
    >>> y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
    >>> z = np.polyfit(x, y, 3)
    >>> z
    array([ 0.08703704, -0.81349206,  1.69312169, -0.03968254]) # may vary

    It is convenient to use `poly1d` objects for dealing with polynomials:

    >>> p = np.poly1d(z)
    >>> p(0.5)
    0.6143849206349179 # may vary
    >>> p(3.5)
    -0.34732142857143039 # may vary
    >>> p(10)
    22.579365079365115 # may vary

    High-order polynomials may oscillate wildly:

    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter('ignore', np.exceptions.RankWarning)
    ...     p30 = np.poly1d(np.polyfit(x, y, 30))
    ...
    >>> p30(4)
    -0.80000000000000204 # may vary
    >>> p30(5)
    -0.99999999999999445 # may vary
    >>> p30(4.5)
    -0.10547061179440398 # may vary

    Illustration:

    >>> import matplotlib.pyplot as plt
    >>> xp = np.linspace(-2, 6, 100)
    >>> _ = plt.plot(x, y, '.', xp, p(xp), '-', xp, p30(xp), '--')
    >>> plt.ylim(-2,2)
    (-2, 2)
    >>> plt.show()

    """
    order = int(deg) + 1
    x = NX.asarray(x) + 0.0
    y = NX.asarray(y) + 0.0

    # check arguments.
    if deg < 0:
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    if x.shape[0] != y.shape[0]:
        raise TypeError("expected x and y to have same length")

    # set rcond
    if rcond is None:
        rcond = len(x)*finfo(x.dtype).eps

    # set up least squares equation for powers of x
    lhs = vander(x, order)
    rhs = y

    # apply weighting
    if w is not None:
        w = NX.asarray(w) + 0.0
        if w.ndim != 1:
            raise TypeError("expected a 1-d array for weights")
        if w.shape[0] != y.shape[0]:
            raise TypeError("expected w and y to have the same length")
        lhs *= w[:, NX.newaxis]
        if rhs.ndim == 2:
            rhs *= w[:, NX.newaxis]
        else:
            rhs *= w

    # scale lhs to improve condition number and solve
    scale = NX.sqrt((lhs*lhs).sum(axis=0))
    lhs /= scale
    c, resids, rank, s = lstsq(lhs, rhs, rcond)
    c = (c.T/scale).T  # broadcast scale coefficients

    # warn on rank reduction, which indicates an ill conditioned matrix
    if rank != order and not full:
        msg = "Polyfit may be poorly conditioned"
        warnings.warn(msg, RankWarning, stacklevel=2)

    if full:
        return c, resids, rank, s, rcond
    elif cov:
        Vbase = inv(dot(lhs.T, lhs))
        Vbase /= NX.outer(scale, scale)
        if cov == "unscaled":
            fac = 1
        else:
            if len(x) <= order:
                raise ValueError("the number of data points must exceed order "
                                 "to scale the covariance matrix")
            # note, this used to be: fac = resids / (len(x) - order - 2.0)
            # it was decided that the "- 2" (originally justified by "Bayesian
            # uncertainty analysis") is not what the user expects
            # (see gh-11196 and gh-11197)
            fac = resids / (len(x) - order)
        if y.ndim == 1:
            return c, Vbase * fac
        else:
            return c, Vbase[:,:, NX.newaxis] * fac
    else:
        return c


def _polyval_dispatcher(p, x):
    return (p, x)


@array_function_dispatch(_polyval_dispatcher)
def polyval(p, x):
    """
    Evaluate a polynomial at specific values.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    If `p` is of length N, this function returns the value::

        p[0]*x**(N-1) + p[1]*x**(N-2) + ... + p[N-2]*x + p[N-1]

    If `x` is a sequence, then ``p(x)`` is returned for each element of ``x``.
    If `x` is another polynomial then the composite polynomial ``p(x(t))``
    is returned.

    Parameters
    ----------
    p : array_like or poly1d object
       1D array of polynomial coefficients (including coefficients equal
       to zero) from highest degree to the constant term, or an
       instance of poly1d.
    x : array_like or poly1d object
       A number, an array of numbers, or an instance of poly1d, at
       which to evaluate `p`.

    Returns
    -------
    values : ndarray or poly1d
       If `x` is a poly1d instance, the result is the composition of the two
       polynomials, i.e., `x` is "substituted" in `p` and the simplified
       result is returned. In addition, the type of `x` - array_like or
       poly1d - governs the type of the output: `x` array_like => `values`
       array_like, `x` a poly1d object => `values` is also.

    See Also
    --------
    poly1d: A polynomial class.

    Notes
    -----
    Horner's scheme [1]_ is used to evaluate the polynomial. Even so,
    for polynomials of high degree the values may be inaccurate due to
    rounding errors. Use carefully.

    If `x` is a subtype of `ndarray` the return value will be of the same type.

    References
    ----------
    .. [1] I. N. Bronshtein, K. A. Semendyayev, and K. A. Hirsch (Eng.
       trans. Ed.), *Handbook of Mathematics*, New York, Van Nostrand
       Reinhold Co., 1985, pg. 720.

    Examples
    --------
    >>> import numpy as np
    >>> np.polyval([3,0,1], 5)  # 3 * 5**2 + 0 * 5**1 + 1
    76
    >>> np.polyval([3,0,1], np.poly1d(5))
    poly1d([76])
    >>> np.polyval(np.poly1d([3,0,1]), 5)
    76
    >>> np.polyval(np.poly1d([3,0,1]), np.poly1d(5))
    poly1d([76])

    """
    p = NX.asarray(p)
    if isinstance(x, poly1d):
        y = 0
    else:
        x = NX.asanyarray(x)
        y = NX.zeros_like(x)
    for pv in p:
        y = y * x + pv
    return y


def _binary_op_dispatcher(a1, a2):
    return (a1, a2)


@array_function_dispatch(_binary_op_dispatcher)
def polyadd(a1, a2):
    """
    Find the sum of two polynomials.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    Returns the polynomial resulting from the sum of two input polynomials.
    Each input must be either a poly1d object or a 1D sequence of polynomial
    coefficients, from highest to lowest degree.

    Parameters
    ----------
    a1, a2 : array_like or poly1d object
        Input polynomials.

    Returns
    -------
    out : ndarray or poly1d object
        The sum of the inputs. If either input is a poly1d object, then the
        output is also a poly1d object. Otherwise, it is a 1D array of
        polynomial coefficients from highest to lowest degree.

    See Also
    --------
    poly1d : A one-dimensional polynomial class.
    poly, polyadd, polyder, polydiv, polyfit, polyint, polysub, polyval

    Examples
    --------
    >>> import numpy as np
    >>> np.polyadd([1, 2], [9, 5, 4])
    array([9, 6, 6])

    Using poly1d objects:

    >>> p1 = np.poly1d([1, 2])
    >>> p2 = np.poly1d([9, 5, 4])
    >>> print(p1)
    1 x + 2
    >>> print(p2)
       2
    9 x + 5 x + 4
    >>> print(np.polyadd(p1, p2))
       2
    9 x + 6 x + 6

    """
    truepoly = (isinstance(a1, poly1d) or isinstance(a2, poly1d))
    a1 = atleast_1d(a1)
    a2 = atleast_1d(a2)
    diff = len(a2) - len(a1)
    if diff == 0:
        val = a1 + a2
    elif diff > 0:
        zr = NX.zeros(diff, a1.dtype)
        val = NX.concatenate((zr, a1)) + a2
    else:
        zr = NX.zeros(abs(diff), a2.dtype)
        val = a1 + NX.concatenate((zr, a2))
    if truepoly:
        val = poly1d(val)
    return val


@array_function_dispatch(_binary_op_dispatcher)
def polysub(a1, a2):
    """
    Difference (subtraction) of two polynomials.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    Given two polynomials `a1` and `a2`, returns ``a1 - a2``.
    `a1` and `a2` can be either array_like sequences of the polynomials'
    coefficients (including coefficients equal to zero), or `poly1d` objects.

    Parameters
    ----------
    a1, a2 : array_like or poly1d
        Minuend and subtrahend polynomials, respectively.

    Returns
    -------
    out : ndarray or poly1d
        Array or `poly1d` object of the difference polynomial's coefficients.

    See Also
    --------
    polyval, polydiv, polymul, polyadd

    Examples
    --------
    .. math:: (2 x^2 + 10 x - 2) - (3 x^2 + 10 x -4) = (-x^2 + 2)

    >>> import numpy as np

    >>> np.polysub([2, 10, -2], [3, 10, -4])
    array([-1,  0,  2])

    """
    truepoly = (isinstance(a1, poly1d) or isinstance(a2, poly1d))
    a1 = atleast_1d(a1)
    a2 = atleast_1d(a2)
    diff = len(a2) - len(a1)
    if diff == 0:
        val = a1 - a2
    elif diff > 0:
        zr = NX.zeros(diff, a1.dtype)
        val = NX.concatenate((zr, a1)) - a2
    else:
        zr = NX.zeros(abs(diff), a2.dtype)
        val = a1 - NX.concatenate((zr, a2))
    if truepoly:
        val = poly1d(val)
    return val


@array_function_dispatch(_binary_op_dispatcher)
def polymul(a1, a2):
    """
    Find the product of two polynomials.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    Finds the polynomial resulting from the multiplication of the two input
    polynomials. Each input must be either a poly1d object or a 1D sequence
    of polynomial coefficients, from highest to lowest degree.

    Parameters
    ----------
    a1, a2 : array_like or poly1d object
        Input polynomials.

    Returns
    -------
    out : ndarray or poly1d object
        The polynomial resulting from the multiplication of the inputs. If
        either inputs is a poly1d object, then the output is also a poly1d
        object. Otherwise, it is a 1D array of polynomial coefficients from
        highest to lowest degree.

    See Also
    --------
    poly1d : A one-dimensional polynomial class.
    poly, polyadd, polyder, polydiv, polyfit, polyint, polysub, polyval
    convolve : Array convolution. Same output as polymul, but has parameter
               for overlap mode.

    Examples
    --------
    >>> import numpy as np
    >>> np.polymul([1, 2, 3], [9, 5, 1])
    array([ 9, 23, 38, 17,  3])

    Using poly1d objects:

    >>> p1 = np.poly1d([1, 2, 3])
    >>> p2 = np.poly1d([9, 5, 1])
    >>> print(p1)
       2
    1 x + 2 x + 3
    >>> print(p2)
       2
    9 x + 5 x + 1
    >>> print(np.polymul(p1, p2))
       4      3      2
    9 x + 23 x + 38 x + 17 x + 3

    """
    truepoly = (isinstance(a1, poly1d) or isinstance(a2, poly1d))
    a1, a2 = poly1d(a1), poly1d(a2)
    val = NX.convolve(a1, a2)
    if truepoly:
        val = poly1d(val)
    return val


def _polydiv_dispatcher(u, v):
    return (u, v)


@array_function_dispatch(_polydiv_dispatcher)
def polydiv(u, v):
    """
    Returns the quotient and remainder of polynomial division.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    The input arrays are the coefficients (including any coefficients
    equal to zero) of the "numerator" (dividend) and "denominator"
    (divisor) polynomials, respectively.

    Parameters
    ----------
    u : array_like or poly1d
        Dividend polynomial's coefficients.

    v : array_like or poly1d
        Divisor polynomial's coefficients.

    Returns
    -------
    q : ndarray
        Coefficients, including those equal to zero, of the quotient.
    r : ndarray
        Coefficients, including those equal to zero, of the remainder.

    See Also
    --------
    poly, polyadd, polyder, polydiv, polyfit, polyint, polymul, polysub
    polyval

    Notes
    -----
    Both `u` and `v` must be 0-d or 1-d (ndim = 0 or 1), but `u.ndim` need
    not equal `v.ndim`. In other words, all four possible combinations -
    ``u.ndim = v.ndim = 0``, ``u.ndim = v.ndim = 1``,
    ``u.ndim = 1, v.ndim = 0``, and ``u.ndim = 0, v.ndim = 1`` - work.

    Examples
    --------
    .. math:: \\frac{3x^2 + 5x + 2}{2x + 1} = 1.5x + 1.75, remainder 0.25

    >>> import numpy as np
    >>> x = np.array([3.0, 5.0, 2.0])
    >>> y = np.array([2.0, 1.0])
    >>> np.polydiv(x, y)
    (array([1.5 , 1.75]), array([0.25]))

    """
    truepoly = (isinstance(u, poly1d) or isinstance(v, poly1d))
    u = atleast_1d(u) + 0.0
    v = atleast_1d(v) + 0.0
    # w has the common type
    w = u[0] + v[0]
    m = len(u) - 1
    n = len(v) - 1
    scale = 1. / v[0]
    q = NX.zeros((max(m - n + 1, 1),), w.dtype)
    r = u.astype(w.dtype)
    for k in range(0, m-n+1):
        d = scale * r[k]
        q[k] = d
        r[k:k+n+1] -= d*v
    while NX.allclose(r[0], 0, rtol=1e-14) and (r.shape[-1] > 1):
        r = r[1:]
    if truepoly:
        return poly1d(q), poly1d(r)
    return q, r

_poly_mat = re.compile(r"\*\*([0-9]*)")
def _raise_power(astr, wrap=70):
    n = 0
    line1 = ''
    line2 = ''
    output = ' '
    while True:
        mat = _poly_mat.search(astr, n)
        if mat is None:
            break
        span = mat.span()
        power = mat.groups()[0]
        partstr = astr[n:span[0]]
        n = span[1]
        toadd2 = partstr + ' '*(len(power)-1)
        toadd1 = ' '*(len(partstr)-1) + power
        if ((len(line2) + len(toadd2) > wrap) or
                (len(line1) + len(toadd1) > wrap)):
            output += line1 + "\n" + line2 + "\n "
            line1 = toadd1
            line2 = toadd2
        else:
            line2 += partstr + ' '*(len(power)-1)
            line1 += ' '*(len(partstr)-1) + power
    output += line1 + "\n" + line2
    return output + astr[n:]


@set_module('numpy')
class poly1d:
    """
    A one-dimensional polynomial class.

    .. note::
       This forms part of the old polynomial API. Since version 1.4, the
       new polynomial API defined in `numpy.polynomial` is preferred.
       A summary of the differences can be found in the
       :doc:`transition guide </reference/routines.polynomials>`.

    A convenience class, used to encapsulate "natural" operations on
    polynomials so that said operations may take on their customary
    form in code (see Examples).

    Parameters
    ----------
    c_or_r : array_like
        The polynomial's coefficients, in decreasing powers, or if
        the value of the second parameter is True, the polynomial's
        roots (values where the polynomial evaluates to 0).  For example,
        ``poly1d([1, 2, 3])`` returns an object that represents
        :math:`x^2 + 2x + 3`, whereas ``poly1d([1, 2, 3], True)`` returns
        one that represents :math:`(x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x -6`.
    r : bool, optional
        If True, `c_or_r` specifies the polynomial's roots; the default
        is False.
    variable : str, optional
        Changes the variable used when printing `p` from `x` to `variable`
        (see Examples).

    Examples
    --------
    Construct the polynomial :math:`x^2 + 2x + 3`:

    >>> import numpy as np

    >>> p = np.poly1d([1, 2, 3])
    >>> print(np.poly1d(p))
       2
    1 x + 2 x + 3

    Evaluate the polynomial at :math:`x = 0.5`:

    >>> p(0.5)
    4.25

    Find the roots:

    >>> p.r
    array([-1.+1.41421356j, -1.-1.41421356j])
    >>> p(p.r)
    array([ -4.44089210e-16+0.j,  -4.44089210e-16+0.j]) # may vary

    These numbers in the previous line represent (0, 0) to machine precision

    Show the coefficients:

    >>> p.c
    array([1, 2, 3])

    Display the order (the leading zero-coefficients are removed):

    >>> p.order
    2

    Show the coefficient of the k-th power in the polynomial
    (which is equivalent to ``p.c[-(i+1)]``):

    >>> p[1]
    2

    Polynomials can be added, subtracted, multiplied, and divided
    (returns quotient and remainder):

    >>> p * p
    poly1d([ 1,  4, 10, 12,  9])

    >>> (p**3 + 4) / p
    (poly1d([ 1.,  4., 10., 12.,  9.]), poly1d([4.]))

    ``asarray(p)`` gives the coefficient array, so polynomials can be
    used in all functions that accept arrays:

    >>> p**2 # square of polynomial
    poly1d([ 1,  4, 10, 12,  9])

    >>> np.square(p) # square of individual coefficients
    array([1, 4, 9])

    The variable used in the string representation of `p` can be modified,
    using the `variable` parameter:

    >>> p = np.poly1d([1,2,3], variable='z')
    >>> print(p)
       2
    1 z + 2 z + 3

    Construct a polynomial from its roots:

    >>> np.poly1d([1, 2], True)
    poly1d([ 1., -3.,  2.])

    This is the same polynomial as obtained by:

    >>> np.poly1d([1, -1]) * np.poly1d([1, -2])
    poly1d([ 1, -3,  2])

    """
    __hash__ = None

    @property
    def coeffs(self):
        """ The polynomial coefficients """
        return self._coeffs

    @coeffs.setter
    def coeffs(self, value):
        # allowing this makes p.coeffs *= 2 legal
        if value is not self._coeffs:
            raise AttributeError("Cannot set attribute")

    @property
    def variable(self):
        """ The name of the polynomial variable """
        return self._variable

    # calculated attributes
    @property
    def order(self):
        """ The order or degree of the polynomial """
        return len(self._coeffs) - 1

    @property
    def roots(self):
        """ The roots of the polynomial, where self(x) == 0 """
        return roots(self._coeffs)

    # our internal _coeffs property need to be backed by __dict__['coeffs'] for
    # scipy to work correctly.
    @property
    def _coeffs(self):
        return self.__dict__['coeffs']
    @_coeffs.setter
    def _coeffs(self, coeffs):
        self.__dict__['coeffs'] = coeffs

    # alias attributes
    r = roots
    c = coef = coefficients = coeffs
    o = order

    def __init__(self, c_or_r, r=False, variable=None):
        if isinstance(c_or_r, poly1d):
            self._variable = c_or_r._variable
            self._coeffs = c_or_r._coeffs

            if set(c_or_r.__dict__) - set(self.__dict__):
                msg = ("In the future extra properties will not be copied "
                       "across when constructing one poly1d from another")
                warnings.warn(msg, FutureWarning, stacklevel=2)
                self.__dict__.update(c_or_r.__dict__)

            if variable is not None:
                self._variable = variable
            return
        if r:
            c_or_r = poly(c_or_r)
        c_or_r = atleast_1d(c_or_r)
        if c_or_r.ndim > 1:
            raise ValueError("Polynomial must be 1d only.")
        c_or_r = trim_zeros(c_or_r, trim='f')
        if len(c_or_r) == 0:
            c_or_r = NX.array([0], dtype=c_or_r.dtype)
        self._coeffs = c_or_r
        if variable is None:
            variable = 'x'
        self._variable = variable

    def __array__(self, t=None, copy=None):
        if t:
            return NX.asarray(self.coeffs, t, copy=copy)
        else:
            return NX.asarray(self.coeffs, copy=copy)

    def __repr__(self):
        vals = repr(self.coeffs)
        vals = vals[6:-1]
        return "poly1d(%s)" % vals

    def __len__(self):
        return self.order

    def __str__(self):
        thestr = "0"
        var = self.variable

        # Remove leading zeros
        coeffs = self.coeffs[NX.logical_or.accumulate(self.coeffs != 0)]
        N = len(coeffs)-1

        def fmt_float(q):
            s = '%.4g' % q
            if s.endswith('.0000'):
                s = s[:-5]
            return s

        for k, coeff in enumerate(coeffs):
            if not iscomplex(coeff):
                coefstr = fmt_float(real(coeff))
            elif real(coeff) == 0:
                coefstr = '%sj' % fmt_float(imag(coeff))
            else:
                coefstr = '(%s + %sj)' % (fmt_float(real(coeff)),
                                          fmt_float(imag(coeff)))

            power = (N-k)
            if power == 0:
                if coefstr != '0':
                    newstr = '%s' % (coefstr,)
                else:
                    if k == 0:
                        newstr = '0'
                    else:
                        newstr = ''
            elif power == 1:
                if coefstr == '0':
                    newstr = ''
                elif coefstr == 'b':
                    newstr = var
                else:
                    newstr = '%s %s' % (coefstr, var)
            else:
                if coefstr == '0':
                    newstr = ''
                elif coefstr == 'b':
                    newstr = '%s**%d' % (var, power,)
                else:
                    newstr = '%s %s**%d' % (coefstr, var, power)

            if k > 0:
                if newstr != '':
                    if newstr.startswith('-'):
                        thestr = "%s - %s" % (thestr, newstr[1:])
                    else:
                        thestr = "%s + %s" % (thestr, newstr)
            else:
                thestr = newstr
        return _raise_power(thestr)

    def __call__(self, val):
        return polyval(self.coeffs, val)

    def __neg__(self):
        return poly1d(-self.coeffs)

    def __pos__(self):
        return self

    def __mul__(self, other):
        if isscalar(other):
            return poly1d(self.coeffs * other)
        else:
            other = poly1d(other)
            return poly1d(polymul(self.coeffs, other.coeffs))

    def __rmul__(self, other):
        if isscalar(other):
            return poly1d(other * self.coeffs)
        else:
            other = poly1d(other)
            return poly1d(polymul(self.coeffs, other.coeffs))

    def __add__(self, other):
        other = poly1d(other)
        return poly1d(polyadd(self.coeffs, other.coeffs))

    def __radd__(self, other):
        other = poly1d(other)
        return poly1d(polyadd(self.coeffs, other.coeffs))

    def __pow__(self, val):
        if not isscalar(val) or int(val) != val or val < 0:
            raise ValueError("Power to non-negative integers only.")
        res = [1]
        for _ in range(val):
            res = polymul(self.coeffs, res)
        return poly1d(res)

    def __sub__(self, other):
        other = poly1d(other)
        return poly1d(polysub(self.coeffs, other.coeffs))

    def __rsub__(self, other):
        other = poly1d(other)
        return poly1d(polysub(other.coeffs, self.coeffs))

    def __div__(self, other):
        if isscalar(other):
            return poly1d(self.coeffs/other)
        else:
            other = poly1d(other)
            return polydiv(self, other)

    __truediv__ = __div__

    def __rdiv__(self, other):
        if isscalar(other):
            return poly1d(other/self.coeffs)
        else:
            other = poly1d(other)
            return polydiv(other, self)

    __rtruediv__ = __rdiv__

    def __eq__(self, other):
        if not isinstance(other, poly1d):
            return NotImplemented
        if self.coeffs.shape != other.coeffs.shape:
            return False
        return (self.coeffs == other.coeffs).all()

    def __ne__(self, other):
        if not isinstance(other, poly1d):
            return NotImplemented
        return not self.__eq__(other)


    def __getitem__(self, val):
        ind = self.order - val
        if val > self.order:
            return self.coeffs.dtype.type(0)
        if val < 0:
            return self.coeffs.dtype.type(0)
        return self.coeffs[ind]

    def __setitem__(self, key, val):
        ind = self.order - key
        if key < 0:
            raise ValueError("Does not support negative powers.")
        if key > self.order:
            zr = NX.zeros(key-self.order, self.coeffs.dtype)
            self._coeffs = NX.concatenate((zr, self.coeffs))
            ind = 0
        self._coeffs[ind] = val
        return

    def __iter__(self):
        return iter(self.coeffs)

    def integ(self, m=1, k=0):
        """
        Return an antiderivative (indefinite integral) of this polynomial.

        Refer to `polyint` for full documentation.

        See Also
        --------
        polyint : equivalent function

        """
        return poly1d(polyint(self.coeffs, m=m, k=k))

    def deriv(self, m=1):
        """
        Return a derivative of this polynomial.

        Refer to `polyder` for full documentation.

        See Also
        --------
        polyder : equivalent function

        """
        return poly1d(polyder(self.coeffs, m=m))

# Stuff to do on module import

warnings.simplefilter('always', RankWarning)
<|endoftext|>"""
``numpy.lib`` is mostly a space for implementing functions that don't
belong in core or in another NumPy submodule with a clear purpose
(e.g. ``random``, ``fft``, ``linalg``, ``ma``).

``numpy.lib``'s private submodules contain basic functions that are used by
other public modules and are useful to have in the main name-space.

"""

# Public submodules
# Note: recfunctions and (maybe) format are public too, but not imported
from . import array_utils
from . import introspect
from . import mixins
from . import npyio
from . import scimath
from . import stride_tricks

# Private submodules
# load module names. See https://github.com/networkx/networkx/issues/5838
from . import _type_check_impl
from . import _index_tricks_impl
from . import _nanfunctions_impl
from . import _function_base_impl
from . import _stride_tricks_impl
from . import _shape_base_impl
from . import _twodim_base_impl
from . import _ufunclike_impl
from . import _histograms_impl
from . import _utils_impl
from . import _arraysetops_impl
from . import _polynomial_impl
from . import _npyio_impl
from . import _arrayterator_impl
from . import _arraypad_impl
from . import _version

# numpy.lib namespace members
from ._arrayterator_impl import Arrayterator
from ._version import NumpyVersion
from numpy._core._multiarray_umath import add_docstring, tracemalloc_domain
from numpy._core.function_base import add_newdoc

__all__ = [
    "Arrayterator", "add_docstring", "add_newdoc", "array_utils",
    "introspect", "mixins", "NumpyVersion", "npyio", "scimath",
    "stride_tricks", "tracemalloc_domain"
]

add_newdoc.__module__ = "numpy.lib"

from numpy._pytesttester import PytestTester
test = PytestTester(__name__)
del PytestTester

def __getattr__(attr):
    # Warn for deprecated/removed aliases
    import math
    import warnings

    if attr == "math":
        warnings.warn(
            "`np.lib.math` is a deprecated alias for the standard library "
            "`math` module (Deprecated Numpy 1.25). Replace usages of "
            "`numpy.lib.math` with `math`", DeprecationWarning, stacklevel=2)
        return math
    elif attr == "emath":
        raise AttributeError(
            "numpy.lib.emath was an alias for emath module that was removed "
            "in NumPy 2.0. Replace usages of numpy.lib.emath with "
            "numpy.emath.",
            name=None
        )
    elif attr in (
        "histograms", "type_check", "nanfunctions", "function_base",
        "arraypad", "arraysetops", "ufunclike", "utils", "twodim_base",
        "shape_base", "polynomial", "index_tricks",
    ):
        raise AttributeError(
            f"numpy.lib.{attr} is now private. If you are using a public "
            "function, it should be available in the main numpy namespace, "
            "otherwise check the NumPy 2.0 migration guide.",
            name=None
        )
    elif attr == "arrayterator":
        raise AttributeError(
            "numpy.lib.arrayterator submodule is now private. To access "
            "Arrayterator class use numpy.lib.Arrayterator.",
            name=None
        )
    else:
        raise AttributeError("module {!r} has no attribute "
                             "{!r}".format(__name__, attr))
<|endoftext|>"""
Set operations for arrays based on sorting.

Notes
-----

For floating point arrays, inaccurate results may appear due to usual round-off
and floating point comparison issues.

Speed could be gained in some operations by an implementation of
`numpy.sort`, that can provide directly the permutation vectors, thus avoiding
calls to `numpy.argsort`.

Original author: Robert Cimrman

"""
import functools
import warnings
from typing import NamedTuple

import numpy as np
from numpy._core import overrides
from numpy._core._multiarray_umath import _array_converter


array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


__all__ = [
    "ediff1d", "in1d", "intersect1d", "isin", "setdiff1d", "setxor1d",
    "union1d", "unique", "unique_all", "unique_counts", "unique_inverse",
    "unique_values"
]


def _ediff1d_dispatcher(ary, to_end=None, to_begin=None):
    return (ary, to_end, to_begin)


@array_function_dispatch(_ediff1d_dispatcher)
def ediff1d(ary, to_end=None, to_begin=None):
    """
    The differences between consecutive elements of an array.

    Parameters
    ----------
    ary : array_like
        If necessary, will be flattened before the differences are taken.
    to_end : array_like, optional
        Number(s) to append at the end of the returned differences.
    to_begin : array_like, optional
        Number(s) to prepend at the beginning of the returned differences.

    Returns
    -------
    ediff1d : ndarray
        The differences. Loosely, this is ``ary.flat[1:] - ary.flat[:-1]``.

    See Also
    --------
    diff, gradient

    Notes
    -----
    When applied to masked arrays, this function drops the mask information
    if the `to_begin` and/or `to_end` parameters are used.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 4, 7, 0])
    >>> np.ediff1d(x)
    array([ 1,  2,  3, -7])

    >>> np.ediff1d(x, to_begin=-99, to_end=np.array([88, 99]))
    array([-99,   1,   2, ...,  -7,  88,  99])

    The returned array is always 1D.

    >>> y = [[1, 2, 4], [1, 6, 24]]
    >>> np.ediff1d(y)
    array([ 1,  2, -3,  5, 18])

    """
    conv = _array_converter(ary)
    # Convert to (any) array and ravel:
    ary = conv[0].ravel()

    # enforce that the dtype of `ary` is used for the output
    dtype_req = ary.dtype

    # fast track default case
    if to_begin is None and to_end is None:
        return ary[1:] - ary[:-1]

    if to_begin is None:
        l_begin = 0
    else:
        to_begin = np.asanyarray(to_begin)
        if not np.can_cast(to_begin, dtype_req, casting="same_kind"):
            raise TypeError("dtype of `to_begin` must be compatible "
                            "with input `ary` under the `same_kind` rule.")

        to_begin = to_begin.ravel()
        l_begin = len(to_begin)

    if to_end is None:
        l_end = 0
    else:
        to_end = np.asanyarray(to_end)
        if not np.can_cast(to_end, dtype_req, casting="same_kind"):
            raise TypeError("dtype of `to_end` must be compatible "
                            "with input `ary` under the `same_kind` rule.")

        to_end = to_end.ravel()
        l_end = len(to_end)

    # do the calculation in place and copy to_begin and to_end
    l_diff = max(len(ary) - 1, 0)
    result = np.empty_like(ary, shape=l_diff + l_begin + l_end)

    if l_begin > 0:
        result[:l_begin] = to_begin
    if l_end > 0:
        result[l_begin + l_diff:] = to_end
    np.subtract(ary[1:], ary[:-1], result[l_begin:l_begin + l_diff])

    return conv.wrap(result)


def _unpack_tuple(x):
    """ Unpacks one-element tuples for use as return values """
    if len(x) == 1:
        return x[0]
    else:
        return x


def _unique_dispatcher(ar, return_index=None, return_inverse=None,
                       return_counts=None, axis=None, *, equal_nan=None):
    return (ar,)


@array_function_dispatch(_unique_dispatcher)
def unique(ar, return_index=False, return_inverse=False,
           return_counts=False, axis=None, *, equal_nan=True):
    """
    Find the unique elements of an array.

    Returns the sorted unique elements of an array. There are three optional
    outputs in addition to the unique elements:

    * the indices of the input array that give the unique values
    * the indices of the unique array that reconstruct the input array
    * the number of times each unique value comes up in the input array

    Parameters
    ----------
    ar : array_like
        Input array. Unless `axis` is specified, this will be flattened if it
        is not already 1-D.
    return_index : bool, optional
        If True, also return the indices of `ar` (along the specified axis,
        if provided, or in the flattened array) that result in the unique array.
    return_inverse : bool, optional
        If True, also return the indices of the unique array (for the specified
        axis, if provided) that can be used to reconstruct `ar`.
    return_counts : bool, optional
        If True, also return the number of times each unique item appears
        in `ar`.
    axis : int or None, optional
        The axis to operate on. If None, `ar` will be flattened. If an integer,
        the subarrays indexed by the given axis will be flattened and treated
        as the elements of a 1-D array with the dimension of the given axis,
        see the notes for more details.  Object arrays or structured arrays
        that contain objects are not supported if the `axis` kwarg is used. The
        default is None.

    equal_nan : bool, optional
        If True, collapses multiple NaN values in the return array into one.

        .. versionadded:: 1.24

    Returns
    -------
    unique : ndarray
        The sorted unique values.
    unique_indices : ndarray, optional
        The indices of the first occurrences of the unique values in the
        original array. Only provided if `return_index` is True.
    unique_inverse : ndarray, optional
        The indices to reconstruct the original array from the
        unique array. Only provided if `return_inverse` is True.
    unique_counts : ndarray, optional
        The number of times each of the unique values comes up in the
        original array. Only provided if `return_counts` is True.

    See Also
    --------
    repeat : Repeat elements of an array.
    sort : Return a sorted copy of an array.

    Notes
    -----
    When an axis is specified the subarrays indexed by the axis are sorted.
    This is done by making the specified axis the first dimension of the array
    (move the axis to the first dimension to keep the order of the other axes)
    and then flattening the subarrays in C order. The flattened subarrays are
    then viewed as a structured type with each element given a label, with the
    effect that we end up with a 1-D array of structured types that can be
    treated in the same way as any other 1-D array. The result is that the
    flattened subarrays are sorted in lexicographic order starting with the
    first element.

    .. versionchanged:: 1.21
        Like np.sort, NaN will sort to the end of the values.
        For complex arrays all NaN values are considered equivalent
        (no matter whether the NaN is in the real or imaginary part).
        As the representant for the returned array the smallest one in the
        lexicographical order is chosen - see np.sort for how the lexicographical
        order is defined for complex arrays.

    .. versionchanged:: 2.0
        For multi-dimensional inputs, ``unique_inverse`` is reshaped
        such that the input can be reconstructed using
        ``np.take(unique, unique_inverse, axis=axis)``. The result is
        now not 1-dimensional when ``axis=None``.

        Note that in NumPy 2.0.0 a higher dimensional array was returned also
        when ``axis`` was not ``None``.  This was reverted, but
        ``inverse.reshape(-1)`` can be used to ensure compatibility with both
        versions.

    Examples
    --------
    >>> import numpy as np
    >>> np.unique([1, 1, 2, 2, 3, 3])
    array([1, 2, 3])
    >>> a = np.array([[1, 1], [2, 3]])
    >>> np.unique(a)
    array([1, 2, 3])

    Return the unique rows of a 2D array

    >>> a = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]])
    >>> np.unique(a, axis=0)
    array([[1, 0, 0], [2, 3, 4]])

    Return the indices of the original array that give the unique values:

    >>> a = np.array(['a', 'b', 'b', 'c', 'a'])
    >>> u, indices = np.unique(a, return_index=True)
    >>> u
    array(['a', 'b', 'c'], dtype='<U1')
    >>> indices
    array([0, 1, 3])
    >>> a[indices]
    array(['a', 'b', 'c'], dtype='<U1')

    Reconstruct the input array from the unique values and inverse:

    >>> a = np.array([1, 2, 6, 4, 2, 3, 2])
    >>> u, indices = np.unique(a, return_inverse=True)
    >>> u
    array([1, 2, 3, 4, 6])
    >>> indices
    array([0, 1, 4, 3, 1, 2, 1])
    >>> u[indices]
    array([1, 2, 6, 4, 2, 3, 2])

    Reconstruct the input values from the unique values and counts:

    >>> a = np.array([1, 2, 6, 4, 2, 3, 2])
    >>> values, counts = np.unique(a, return_counts=True)
    >>> values
    array([1, 2, 3, 4, 6])
    >>> counts
    array([1, 3, 1, 1, 1])
    >>> np.repeat(values, counts)
    array([1, 2, 2, 2, 3, 4, 6])    # original order not preserved

    """
    ar = np.asanyarray(ar)
    if axis is None:
        ret = _unique1d(ar, return_index, return_inverse, return_counts,
                        equal_nan=equal_nan, inverse_shape=ar.shape, axis=None)
        return _unpack_tuple(ret)

    # axis was specified and not None
    try:
        ar = np.moveaxis(ar, axis, 0)
    except np.exceptions.AxisError:
        # this removes the "axis1" or "axis2" prefix from the error message
        raise np.exceptions.AxisError(axis, ar.ndim) from None
    inverse_shape = [1] * ar.ndim
    inverse_shape[axis] = ar.shape[0]

    # Must reshape to a contiguous 2D array for this to work...
    orig_shape, orig_dtype = ar.shape, ar.dtype
    ar = ar.reshape(orig_shape[0], np.prod(orig_shape[1:], dtype=np.intp))
    ar = np.ascontiguousarray(ar)
    dtype = [('f{i}'.format(i=i), ar.dtype) for i in range(ar.shape[1])]

    # At this point, `ar` has shape `(n, m)`, and `dtype` is a structured
    # data type with `m` fields where each field has the data type of `ar`.
    # In the following, we create the array `consolidated`, which has
    # shape `(n,)` with data type `dtype`.
    try:
        if ar.shape[1] > 0:
            consolidated = ar.view(dtype)
        else:
            # If ar.shape[1] == 0, then dtype will be `np.dtype([])`, which is
            # a data type with itemsize 0, and the call `ar.view(dtype)` will
            # fail.  Instead, we'll use `np.empty` to explicitly create the
            # array with shape `(len(ar),)`.  Because `dtype` in this case has
            # itemsize 0, the total size of the result is still 0 bytes.
            consolidated = np.empty(len(ar), dtype=dtype)
    except TypeError as e:
        # There's no good way to do this for object arrays, etc...
        msg = 'The axis argument to unique is not supported for dtype {dt}'
        raise TypeError(msg.format(dt=ar.dtype)) from e

    def reshape_uniq(uniq):
        n = len(uniq)
        uniq = uniq.view(orig_dtype)
        uniq = uniq.reshape(n, *orig_shape[1:])
        uniq = np.moveaxis(uniq, 0, axis)
        return uniq

    output = _unique1d(consolidated, return_index,
                       return_inverse, return_counts,
                       equal_nan=equal_nan, inverse_shape=inverse_shape,
                       axis=axis)
    output = (reshape_uniq(output[0]),) + output[1:]
    return _unpack_tuple(output)


def _unique1d(ar, return_index=False, return_inverse=False,
              return_counts=False, *, equal_nan=True, inverse_shape=None,
              axis=None):
    """
    Find the unique elements of an array, ignoring shape.
    """
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse

    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    mask = np.empty(aux.shape, dtype=np.bool)
    mask[:1] = True
    if (equal_nan and aux.shape[0] > 0 and aux.dtype.kind in "cfmM" and
            np.isnan(aux[-1])):
        if aux.dtype.kind == "c":  # for complex all NaNs are considered equivalent
            aux_firstnan = np.searchsorted(np.isnan(aux), True, side='left')
        else:
            aux_firstnan = np.searchsorted(aux, aux[-1], side='left')
        if aux_firstnan > 0:
            mask[1:aux_firstnan] = (
                aux[1:aux_firstnan] != aux[:aux_firstnan - 1])
        mask[aux_firstnan] = True
        mask[aux_firstnan + 1:] = False
    else:
        mask[1:] = aux[1:] != aux[:-1]

    ret = (aux[mask],)
    if return_index:
        ret += (perm[mask],)
    if return_inverse:
        imask = np.cumsum(mask) - 1
        inv_idx = np.empty(mask.shape, dtype=np.intp)
        inv_idx[perm] = imask
        ret += (inv_idx.reshape(inverse_shape) if axis is None else inv_idx,)
    if return_counts:
        idx = np.concatenate(np.nonzero(mask) + ([mask.size],))
        ret += (np.diff(idx),)
    return ret


# Array API set functions

class UniqueAllResult(NamedTuple):
    values: np.ndarray
    indices: np.ndarray
    inverse_indices: np.ndarray
    counts: np.ndarray


class UniqueCountsResult(NamedTuple):
    values: np.ndarray
    counts: np.ndarray


class UniqueInverseResult(NamedTuple):
    values: np.ndarray
    inverse_indices: np.ndarray


def _unique_all_dispatcher(x, /):
    return (x,)


@array_function_dispatch(_unique_all_dispatcher)
def unique_all(x):
    """
    Find the unique elements of an array, and counts, inverse, and indices.

    This function is an Array API compatible alternative to::

        np.unique(x, return_index=True, return_inverse=True,
                  return_counts=True, equal_nan=False)

    but returns a namedtuple for easier access to each output.

    Parameters
    ----------
    x : array_like
        Input array. It will be flattened if it is not already 1-D.

    Returns
    -------
    out : namedtuple
        The result containing:

        * values - The unique elements of an input array.
        * indices - The first occurring indices for each unique element.
        * inverse_indices - The indices from the set of unique elements
          that reconstruct `x`.
        * counts - The corresponding counts for each unique element.

    See Also
    --------
    unique : Find the unique elements of an array.

    Examples
    --------
    >>> import numpy as np
    >>> x = [1, 1, 2]
    >>> uniq = np.unique_all(x)
    >>> uniq.values
    array([1, 2])
    >>> uniq.indices
    array([0, 2])
    >>> uniq.inverse_indices
    array([0, 0, 1])
    >>> uniq.counts
    array([2, 1])
    """
    result = unique(
        x,
        return_index=True,
        return_inverse=True,
        return_counts=True,
        equal_nan=False
    )
    return UniqueAllResult(*result)


def _unique_counts_dispatcher(x, /):
    return (x,)


@array_function_dispatch(_unique_counts_dispatcher)
def unique_counts(x):
    """
    Find the unique elements and counts of an input array `x`.

    This function is an Array API compatible alternative to::

        np.unique(x, return_counts=True, equal_nan=False)

    but returns a namedtuple for easier access to each output.

    Parameters
    ----------
    x : array_like
        Input array. It will be flattened if it is not already 1-D.

    Returns
    -------
    out : namedtuple
        The result containing:

        * values - The unique elements of an input array.
        * counts - The corresponding counts for each unique element.

    See Also
    --------
    unique : Find the unique elements of an array.

    Examples
    --------
    >>> import numpy as np
    >>> x = [1, 1, 2]
    >>> uniq = np.unique_counts(x)
    >>> uniq.values
    array([1, 2])
    >>> uniq.counts
    array([2, 1])
    """
    result = unique(
        x,
        return_index=False,
        return_inverse=False,
        return_counts=True,
        equal_nan=False
    )
    return UniqueCountsResult(*result)


def _unique_inverse_dispatcher(x, /):
    return (x,)


@array_function_dispatch(_unique_inverse_dispatcher)
def unique_inverse(x):
    """
    Find the unique elements of `x` and indices to reconstruct `x`.

    This function is an Array API compatible alternative to::

        np.unique(x, return_inverse=True, equal_nan=False)

    but returns a namedtuple for easier access to each output.

    Parameters
    ----------
    x : array_like
        Input array. It will be flattened if it is not already 1-D.

    Returns
    -------
    out : namedtuple
        The result containing:

        * values - The unique elements of an input array.
        * inverse_indices - The indices from the set of unique elements
          that reconstruct `x`.

    See Also
    --------
    unique : Find the unique elements of an array.

    Examples
    --------
    >>> import numpy as np
    >>> x = [1, 1, 2]
    >>> uniq = np.unique_inverse(x)
    >>> uniq.values
    array([1, 2])
    >>> uniq.inverse_indices
    array([0, 0, 1])
    """
    result = unique(
        x,
        return_index=False,
        return_inverse=True,
        return_counts=False,
        equal_nan=False
    )
    return UniqueInverseResult(*result)


def _unique_values_dispatcher(x, /):
    return (x,)


@array_function_dispatch(_unique_values_dispatcher)
def unique_values(x):
    """
    Returns the unique elements of an input array `x`.

    This function is an Array API compatible alternative to::

        np.unique(x, equal_nan=False)

    Parameters
    ----------
    x : array_like
        Input array. It will be flattened if it is not already 1-D.

    Returns
    -------
    out : ndarray
        The unique elements of an input array.

    See Also
    --------
    unique : Find the unique elements of an array.

    Examples
    --------
    >>> import numpy as np
    >>> np.unique_values([1, 1, 2])
    array([1, 2])

    """
    return unique(
        x,
        return_index=False,
        return_inverse=False,
        return_counts=False,
        equal_nan=False
    )


def _intersect1d_dispatcher(
        ar1, ar2, assume_unique=None, return_indices=None):
    return (ar1, ar2)


@array_function_dispatch(_intersect1d_dispatcher)
def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
    """
    Find the intersection of two arrays.

    Return the sorted, unique values that are in both of the input arrays.

    Parameters
    ----------
    ar1, ar2 : array_like
        Input arrays. Will be flattened if not already 1D.
    assume_unique : bool
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  If True but ``ar1`` or ``ar2`` are not
        unique, incorrect results and out-of-bounds indices could result.
        Default is False.
    return_indices : bool
        If True, the indices which correspond to the intersection of the two
        arrays are returned. The first instance of a value is used if there are
        multiple. Default is False.

    Returns
    -------
    intersect1d : ndarray
        Sorted 1D array of common and unique elements.
    comm1 : ndarray
        The indices of the first occurrences of the common values in `ar1`.
        Only provided if `return_indices` is True.
    comm2 : ndarray
        The indices of the first occurrences of the common values in `ar2`.
        Only provided if `return_indices` is True.

    Examples
    --------
    >>> import numpy as np
    >>> np.intersect1d([1, 3, 4, 3], [3, 1, 2, 1])
    array([1, 3])

    To intersect more than two arrays, use functools.reduce:

    >>> from functools import reduce
    >>> reduce(np.intersect1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2]))
    array([3])

    To return the indices of the values common to the input arrays
    along with the intersected values:

    >>> x = np.array([1, 1, 2, 3, 4])
    >>> y = np.array([2, 1, 4, 6])
    >>> xy, x_ind, y_ind = np.intersect1d(x, y, return_indices=True)
    >>> x_ind, y_ind
    (array([0, 2, 4]), array([1, 0, 2]))
    >>> xy, x[x_ind], y[y_ind]
    (array([1, 2, 4]), array([1, 2, 4]), array([1, 2, 4]))

    """
    ar1 = np.asanyarray(ar1)
    ar2 = np.asanyarray(ar2)

    if not assume_unique:
        if return_indices:
            ar1, ind1 = unique(ar1, return_index=True)
            ar2, ind2 = unique(ar2, return_index=True)
        else:
            ar1 = unique(ar1)
            ar2 = unique(ar2)
    else:
        ar1 = ar1.ravel()
        ar2 = ar2.ravel()

    aux = np.concatenate((ar1, ar2))
    if return_indices:
        aux_sort_indices = np.argsort(aux, kind='mergesort')
        aux = aux[aux_sort_indices]
    else:
        aux.sort()

    mask = aux[1:] == aux[:-1]
    int1d = aux[:-1][mask]

    if return_indices:
        ar1_indices = aux_sort_indices[:-1][mask]
        ar2_indices = aux_sort_indices[1:][mask] - ar1.size
        if not assume_unique:
            ar1_indices = ind1[ar1_indices]
            ar2_indices = ind2[ar2_indices]

        return int1d, ar1_indices, ar2_indices
    else:
        return int1d


def _setxor1d_dispatcher(ar1, ar2, assume_unique=None):
    return (ar1, ar2)


@array_function_dispatch(_setxor1d_dispatcher)
def setxor1d(ar1, ar2, assume_unique=False):
    """
    Find the set exclusive-or of two arrays.

    Return the sorted, unique values that are in only one (not both) of the
    input arrays.

    Parameters
    ----------
    ar1, ar2 : array_like
        Input arrays.
    assume_unique : bool
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation. Default is False.

    Returns
    -------
    setxor1d : ndarray
        Sorted 1D array of unique values that are in only one of the input
        arrays.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1, 2, 3, 2, 4])
    >>> b = np.array([2, 3, 5, 7, 5])
    >>> np.setxor1d(a,b)
    array([1, 4, 5, 7])

    """
    if not assume_unique:
        ar1 = unique(ar1)
        ar2 = unique(ar2)

    aux = np.concatenate((ar1, ar2), axis=None)
    if aux.size == 0:
        return aux

    aux.sort()
    flag = np.concatenate(([True], aux[1:] != aux[:-1], [True]))
    return aux[flag[1:] & flag[:-1]]


def _in1d_dispatcher(ar1, ar2, assume_unique=None, invert=None, *,
                     kind=None):
    return (ar1, ar2)


@array_function_dispatch(_in1d_dispatcher)
def in1d(ar1, ar2, assume_unique=False, invert=False, *, kind=None):
    """
    Test whether each element of a 1-D array is also present in a second array.

    .. deprecated:: 2.0
        Use :func:`isin` instead of `in1d` for new code.

    Returns a boolean array the same length as `ar1` that is True
    where an element of `ar1` is in `ar2` and False otherwise.

    Parameters
    ----------
    ar1 : (M,) array_like
        Input array.
    ar2 : array_like
        The values against which to test each value of `ar1`.
    assume_unique : bool, optional
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  Default is False.
    invert : bool, optional
        If True, the values in the returned array are inverted (that is,
        False where an element of `ar1` is in `ar2` and True otherwise).
        Default is False. ``np.in1d(a, b, invert=True)`` is equivalent
        to (but is faster than) ``np.invert(in1d(a, b))``.
    kind : {None, 'sort', 'table'}, optional
        The algorithm to use. This will not affect the final result,
        but will affect the speed and memory use. The default, None,
        will select automatically based on memory considerations.

        * If 'sort', will use a mergesort-based approach. This will have
          a memory usage of roughly 6 times the sum of the sizes of
          `ar1` and `ar2`, not accounting for size of dtypes.
        * If 'table', will use a lookup table approach similar
          to a counting sort. This is only available for boolean and
          integer arrays. This will have a memory usage of the
          size of `ar1` plus the max-min value of `ar2`. `assume_unique`
          has no effect when the 'table' option is used.
        * If None, will automatically choose 'table' if
          the required memory allocation is less than or equal to
          6 times the sum of the sizes of `ar1` and `ar2`,
          otherwise will use 'sort'. This is done to not use
          a large amount of memory by default, even though
          'table' may be faster in most cases. If 'table' is chosen,
          `assume_unique` will have no effect.

    Returns
    -------
    in1d : (M,) ndarray, bool
        The values `ar1[in1d]` are in `ar2`.

    See Also
    --------
    isin                  : Version of this function that preserves the
                            shape of ar1.

    Notes
    -----
    `in1d` can be considered as an element-wise function version of the
    python keyword `in`, for 1-D sequences. ``in1d(a, b)`` is roughly
    equivalent to ``np.array([item in b for item in a])``.
    However, this idea fails if `ar2` is a set, or similar (non-sequence)
    container:  As ``ar2`` is converted to an array, in those cases
    ``asarray(ar2)`` is an object array rather than the expected array of
    contained values.

    Using ``kind='table'`` tends to be faster than `kind='sort'` if the
    following relationship is true:
    ``log10(len(ar2)) > (log10(max(ar2)-min(ar2)) - 2.27) / 0.927``,
    but may use greater memory. The default value for `kind` will
    be automatically selected based only on memory usage, so one may
    manually set ``kind='table'`` if memory constraints can be relaxed.

    Examples
    --------
    >>> import numpy as np
    >>> test = np.array([0, 1, 2, 5, 0])
    >>> states = [0, 2]
    >>> mask = np.in1d(test, states)
    >>> mask
    array([ True, False,  True, False,  True])
    >>> test[mask]
    array([0, 2, 0])
    >>> mask = np.in1d(test, states, invert=True)
    >>> mask
    array([False,  True, False,  True, False])
    >>> test[mask]
    array([1, 5])
    """

    # Deprecated in NumPy 2.0, 2023-08-18
    warnings.warn(
