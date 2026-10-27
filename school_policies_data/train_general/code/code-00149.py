        Notes
        -----
        Input values `x` must be convertible to `float` values like `int`
        or `float`.

        """
        x, x_shape = self._prepare_x(x)
        y = self._evaluate(x)
        return self._finish_y(y, x_shape)

    def _evaluate(self, x):
        """
        Actually evaluate the value of the interpolator.
        """
        raise NotImplementedError()

    def _prepare_x(self, x):
        """Reshape input x array to 1-D"""
        x = _asarray_validated(x, check_finite=False, as_inexact=True)
        x_shape = x.shape
        return x.ravel(), x_shape

    def _finish_y(self, y, x_shape):
        """Reshape interpolated y back to an N-D array similar to initial y"""
        y = y.reshape(x_shape + self._y_extra_shape)
        if self._y_axis != 0 and x_shape != ():
            nx = len(x_shape)
            ny = len(self._y_extra_shape)
            s = (list(range(nx, nx + self._y_axis))
                 + list(range(nx)) + list(range(nx+self._y_axis, nx+ny)))
            y = y.transpose(s)
        return y

    def _reshape_yi(self, yi, check=False):
        yi = np.moveaxis(np.asarray(yi), self._y_axis, 0)
        if check and yi.shape[1:] != self._y_extra_shape:
            ok_shape = (f"{self._y_extra_shape[-self._y_axis:]!r} + (N,) + "
                        f"{self._y_extra_shape[:-self._y_axis]!r}")
            raise ValueError(f"Data must be of shape {ok_shape}")
        return yi.reshape((yi.shape[0], -1))

    def _set_yi(self, yi, xi=None, axis=None):
        if axis is None:
            axis = self._y_axis
        if axis is None:
            raise ValueError("no interpolation axis specified")

        yi = np.asarray(yi)

        shape = yi.shape
        if shape == ():
            shape = (1,)
        if xi is not None and shape[axis] != len(xi):
            raise ValueError("x and y arrays must be equal in length along "
                             "interpolation axis.")

        self._y_axis = (axis % yi.ndim)
        self._y_extra_shape = yi.shape[:self._y_axis] + yi.shape[self._y_axis+1:]
        self.dtype = None
        self._set_dtype(yi.dtype)

    def _set_dtype(self, dtype, union=False):
        if np.issubdtype(dtype, np.complexfloating) \
               or np.issubdtype(self.dtype, np.complexfloating):
            self.dtype = np.complex128
        else:
            if not union or self.dtype != np.complex128:
                self.dtype = np.float64


class _Interpolator1DWithDerivatives(_Interpolator1D):
    def derivatives(self, x, der=None):
        """
        Evaluate several derivatives of the polynomial at the point `x`

        Produce an array of derivatives evaluated at the point `x`.

        Parameters
        ----------
        x : array_like
            Point or points at which to evaluate the derivatives
        der : int or list or None, optional
            How many derivatives to evaluate, or None for all potentially
            nonzero derivatives (that is, a number equal to the number
            of points), or a list of derivatives to evaluate. This number
            includes the function value as the '0th' derivative.

        Returns
        -------
        d : ndarray
            Array with derivatives; ``d[j]`` contains the jth derivative.
            Shape of ``d[j]`` is determined by replacing the interpolation
            axis in the original array with the shape of `x`.

        Examples
        --------
        >>> from scipy.interpolate import KroghInterpolator
        >>> KroghInterpolator([0,0,0],[1,2,3]).derivatives(0)
        array([1.0,2.0,3.0])
        >>> KroghInterpolator([0,0,0],[1,2,3]).derivatives([0,0])
        array([[1.0,1.0],
               [2.0,2.0],
               [3.0,3.0]])

        """
        x, x_shape = self._prepare_x(x)
        y = self._evaluate_derivatives(x, der)

        y = y.reshape((y.shape[0],) + x_shape + self._y_extra_shape)
        if self._y_axis != 0 and x_shape != ():
            nx = len(x_shape)
            ny = len(self._y_extra_shape)
            s = ([0] + list(range(nx+1, nx + self._y_axis+1))
                 + list(range(1, nx+1)) +
                 list(range(nx+1+self._y_axis, nx+ny+1)))
            y = y.transpose(s)
        return y

    def derivative(self, x, der=1):
        """
        Evaluate a single derivative of the polynomial at the point `x`.

        Parameters
        ----------
        x : array_like
            Point or points at which to evaluate the derivatives

        der : integer, optional
            Which derivative to evaluate (default: first derivative).
            This number includes the function value as 0th derivative.

        Returns
        -------
        d : ndarray
            Derivative interpolated at the x-points. Shape of `d` is
            determined by replacing the interpolation axis in the
            original array with the shape of `x`.

        Notes
        -----
        This may be computed by evaluating all derivatives up to the desired
        one (using self.derivatives()) and then discarding the rest.

        """
        x, x_shape = self._prepare_x(x)
        y = self._evaluate_derivatives(x, der+1)
        return self._finish_y(y[der], x_shape)

    def _evaluate_derivatives(self, x, der=None):
        """
        Actually evaluate the derivatives.

        Parameters
        ----------
        x : array_like
            1D array of points at which to evaluate the derivatives
        der : integer, optional
            The number of derivatives to evaluate, from 'order 0' (der=1)
            to order der-1.  If omitted, return all possibly-non-zero
            derivatives, ie 0 to order n-1.

        Returns
        -------
        d : ndarray
            Array of shape ``(der, x.size, self.yi.shape[1])`` containing
            the derivatives from 0 to der-1
        """
        raise NotImplementedError()


class KroghInterpolator(_Interpolator1DWithDerivatives):
    """
    Interpolating polynomial for a set of points.

    The polynomial passes through all the pairs ``(xi, yi)``. One may
    additionally specify a number of derivatives at each point `xi`;
    this is done by repeating the value `xi` and specifying the
    derivatives as successive `yi` values.

    Allows evaluation of the polynomial and all its derivatives.
    For reasons of numerical stability, this function does not compute
    the coefficients of the polynomial, although they can be obtained
    by evaluating all the derivatives.

    Parameters
    ----------
    xi : array_like, shape (npoints, )
        Known x-coordinates. Must be sorted in increasing order.
    yi : array_like, shape (..., npoints, ...)
        Known y-coordinates. When an xi occurs two or more times in
        a row, the corresponding yi's represent derivative values. The length of `yi`
        along the interpolation axis must be equal to the length of `xi`. Use the
        `axis` parameter to select the correct axis.
    axis : int, optional
        Axis in the `yi` array corresponding to the x-coordinate values. Defaults to
        ``axis=0``.

    Notes
    -----
    Be aware that the algorithms implemented here are not necessarily
    the most numerically stable known. Moreover, even in a world of
    exact computation, unless the x coordinates are chosen very
    carefully - Chebyshev zeros (e.g., cos(i*pi/n)) are a good choice -
    polynomial interpolation itself is a very ill-conditioned process
    due to the Runge phenomenon. In general, even with well-chosen
    x values, degrees higher than about thirty cause problems with
    numerical instability in this code.

    Based on [1]_.

    References
    ----------
    .. [1] Krogh, "Efficient Algorithms for Polynomial Interpolation
        and Numerical Differentiation", 1970.

    Examples
    --------
    To produce a polynomial that is zero at 0 and 1 and has
    derivative 2 at 0, call

    >>> from scipy.interpolate import KroghInterpolator
    >>> KroghInterpolator([0,0,1],[0,2,0])

    This constructs the quadratic :math:`2x^2-2x`. The derivative condition
    is indicated by the repeated zero in the `xi` array; the corresponding
    yi values are 0, the function value, and 2, the derivative value.

    For another example, given `xi`, `yi`, and a derivative `ypi` for each
    point, appropriate arrays can be constructed as:

    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> xi = np.linspace(0, 1, 5)
    >>> yi, ypi = rng.random((2, 5))
    >>> xi_k, yi_k = np.repeat(xi, 2), np.ravel(np.dstack((yi,ypi)))
    >>> KroghInterpolator(xi_k, yi_k)

    To produce a vector-valued polynomial, supply a higher-dimensional
    array for `yi`:

    >>> KroghInterpolator([0,1],[[2,3],[4,5]])

    This constructs a linear polynomial giving (2,3) at 0 and (4,5) at 1.

    """

    def __init__(self, xi, yi, axis=0):
        super().__init__(xi, yi, axis)

        self.xi = np.asarray(xi)
        self.yi = self._reshape_yi(yi)
        self.n, self.r = self.yi.shape

        if (deg := self.xi.size) > 30:
            warnings.warn(f"{deg} degrees provided, degrees higher than about"
                          " thirty cause problems with numerical instability "
                          "with 'KroghInterpolator'", stacklevel=2)

        c = np.zeros((self.n+1, self.r), dtype=self.dtype)
        c[0] = self.yi[0]
        Vk = np.zeros((self.n, self.r), dtype=self.dtype)
        for k in range(1, self.n):
            s = 0
            while s <= k and xi[k-s] == xi[k]:
                s += 1
            s -= 1
            Vk[0] = self.yi[k]/float_factorial(s)
            for i in range(k-s):
                if xi[i] == xi[k]:
                    raise ValueError("Elements of `xi` can't be equal.")
                if s == 0:
                    Vk[i+1] = (c[i]-Vk[i])/(xi[i]-xi[k])
                else:
                    Vk[i+1] = (Vk[i+1]-Vk[i])/(xi[i]-xi[k])
            c[k] = Vk[k-s]
        self.c = c

    def _evaluate(self, x):
        pi = 1
        p = np.zeros((len(x), self.r), dtype=self.dtype)
        p += self.c[0,np.newaxis,:]
        for k in range(1, self.n):
            w = x - self.xi[k-1]
            pi = w*pi
            p += pi[:,np.newaxis] * self.c[k]
        return p

    def _evaluate_derivatives(self, x, der=None):
        n = self.n
        r = self.r

        if der is None:
            der = self.n

        pi = np.zeros((n, len(x)))
        w = np.zeros((n, len(x)))
        pi[0] = 1
        p = np.zeros((len(x), self.r), dtype=self.dtype)
        p += self.c[0, np.newaxis, :]

        for k in range(1, n):
            w[k-1] = x - self.xi[k-1]
            pi[k] = w[k-1] * pi[k-1]
            p += pi[k, :, np.newaxis] * self.c[k]

        cn = np.zeros((max(der, n+1), len(x), r), dtype=self.dtype)
        cn[:n+1, :, :] += self.c[:n+1, np.newaxis, :]
        cn[0] = p
        for k in range(1, n):
            for i in range(1, n-k+1):
                pi[i] = w[k+i-1]*pi[i-1] + pi[i]
                cn[k] = cn[k] + pi[i, :, np.newaxis]*cn[k+i]
            cn[k] *= float_factorial(k)

        cn[n, :, :] = 0
        return cn[:der]


def krogh_interpolate(xi, yi, x, der=0, axis=0):
    """
    Convenience function for polynomial interpolation.

    See `KroghInterpolator` for more details.

    Parameters
    ----------
    xi : array_like
        Interpolation points (known x-coordinates).
    yi : array_like
        Known y-coordinates, of shape ``(xi.size, R)``. Interpreted as
        vectors of length R, or scalars if R=1.
    x : array_like
        Point or points at which to evaluate the derivatives.
    der : int or list or None, optional
        How many derivatives to evaluate, or None for all potentially
        nonzero derivatives (that is, a number equal to the number
        of points), or a list of derivatives to evaluate. This number
        includes the function value as the '0th' derivative.
    axis : int, optional
        Axis in the `yi` array corresponding to the x-coordinate values.

    Returns
    -------
    d : ndarray
        If the interpolator's values are R-D then the
        returned array will be the number of derivatives by N by R.
        If `x` is a scalar, the middle dimension will be dropped; if
        the `yi` are scalars then the last dimension will be dropped.

    See Also
    --------
    KroghInterpolator : Krogh interpolator

    Notes
    -----
    Construction of the interpolating polynomial is a relatively expensive
    process. If you want to evaluate it repeatedly consider using the class
    KroghInterpolator (which is what this function uses).

    Examples
    --------
    We can interpolate 2D observed data using Krogh interpolation:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.interpolate import krogh_interpolate
    >>> x_observed = np.linspace(0.0, 10.0, 11)
    >>> y_observed = np.sin(x_observed)
    >>> x = np.linspace(min(x_observed), max(x_observed), num=100)
    >>> y = krogh_interpolate(x_observed, y_observed, x)
    >>> plt.plot(x_observed, y_observed, "o", label="observation")
    >>> plt.plot(x, y, label="krogh interpolation")
    >>> plt.legend()
    >>> plt.show()
    """

    P = KroghInterpolator(xi, yi, axis=axis)
    if der == 0:
        return P(x)
    elif _isscalar(der):
        return P.derivative(x, der=der)
    else:
        return P.derivatives(x, der=np.amax(der)+1)[der]


def approximate_taylor_polynomial(f,x,degree,scale,order=None):
    """
    Estimate the Taylor polynomial of f at x by polynomial fitting.

    Parameters
    ----------
    f : callable
        The function whose Taylor polynomial is sought. Should accept
        a vector of `x` values.
    x : scalar
        The point at which the polynomial is to be evaluated.
    degree : int
        The degree of the Taylor polynomial
    scale : scalar
        The width of the interval to use to evaluate the Taylor polynomial.
        Function values spread over a range this wide are used to fit the
        polynomial. Must be chosen carefully.
    order : int or None, optional
        The order of the polynomial to be used in the fitting; `f` will be
        evaluated ``order+1`` times. If None, use `degree`.

    Returns
    -------
    p : poly1d instance
        The Taylor polynomial (translated to the origin, so that
        for example p(0)=f(x)).

    Notes
    -----
    The appropriate choice of "scale" is a trade-off; too large and the
    function differs from its Taylor polynomial too much to get a good
    answer, too small and round-off errors overwhelm the higher-order terms.
    The algorithm used becomes numerically unstable around order 30 even
    under ideal circumstances.

    Choosing order somewhat larger than degree may improve the higher-order
    terms.

    Examples
    --------
    We can calculate Taylor approximation polynomials of sin function with
    various degrees:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.interpolate import approximate_taylor_polynomial
    >>> x = np.linspace(-10.0, 10.0, num=100)
    >>> plt.plot(x, np.sin(x), label="sin curve")
    >>> for degree in np.arange(1, 15, step=2):
    ...     sin_taylor = approximate_taylor_polynomial(np.sin, 0, degree, 1,
    ...                                                order=degree + 2)
    ...     plt.plot(x, sin_taylor(x), label=f"degree={degree}")
    >>> plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
    ...            borderaxespad=0.0, shadow=True)
    >>> plt.tight_layout()
    >>> plt.axis([-10, 10, -10, 10])
    >>> plt.show()

    """
    if order is None:
        order = degree

    n = order+1
    # Choose n points that cluster near the endpoints of the interval in
    # a way that avoids the Runge phenomenon. Ensure, by including the
    # endpoint or not as appropriate, that one point always falls at x
    # exactly.
    xs = scale*np.cos(np.linspace(0,np.pi,n,endpoint=n % 1)) + x

    P = KroghInterpolator(xs, f(xs))
    d = P.derivatives(x,der=degree+1)

    return np.poly1d((d/factorial(np.arange(degree+1)))[::-1])


class BarycentricInterpolator(_Interpolator1DWithDerivatives):
    r"""Interpolating polynomial for a set of points.

    Constructs a polynomial that passes through a given set of points.
    Allows evaluation of the polynomial and all its derivatives,
    efficient changing of the y-values to be interpolated,
    and updating by adding more x- and y-values.

    For reasons of numerical stability, this function does not compute
    the coefficients of the polynomial.

    The values `yi` need to be provided before the function is
    evaluated, but none of the preprocessing depends on them, so rapid
    updates are possible.

    Parameters
    ----------
    xi : array_like, shape (npoints, )
        1-D array of x coordinates of the points the polynomial
        should pass through
    yi : array_like, shape (..., npoints, ...), optional
        N-D array of y coordinates of the points the polynomial should pass through.
        If None, the y values will be supplied later via the `set_y` method.
        The length of `yi` along the interpolation axis must be equal to the length
        of `xi`. Use the ``axis`` parameter to select correct axis.
    axis : int, optional
        Axis in the yi array corresponding to the x-coordinate values. Defaults
        to ``axis=0``.
    wi : array_like, optional
        The barycentric weights for the chosen interpolation points `xi`.
        If absent or None, the weights will be computed from `xi` (default).
        This allows for the reuse of the weights `wi` if several interpolants
        are being calculated using the same nodes `xi`, without re-computation.
    rng : {None, int, `numpy.random.Generator`}, optional
        If `rng` is passed by keyword, types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.
        If `rng` is already a ``Generator`` instance, then the provided instance is
        used. Specify `rng` for repeatable interpolation.

        If this argument `random_state` is passed by keyword,
        legacy behavior for the argument `random_state` applies:

        - If `random_state` is None (or `numpy.random`), the `numpy.random.RandomState`
          singleton is used.
        - If `random_state` is an int, a new ``RandomState`` instance is used,
          seeded with `random_state`.
        - If `random_state` is already a ``Generator`` or ``RandomState`` instance then
          that instance is used.

        .. versionchanged:: 1.15.0
            As part of the `SPEC-007 <https://scientific-python.org/specs/spec-0007/>`_
            transition from use of `numpy.random.RandomState` to
            `numpy.random.Generator` this keyword was changed from `random_state` to `rng`.
            For an interim period, both keywords will continue to work (only specify
            one of them). After the interim period using the `random_state` keyword will emit
            warnings. The behavior of the `random_state` and `rng` keywords is outlined above.

    Notes
    -----
    This class uses a "barycentric interpolation" method that treats
    the problem as a special case of rational function interpolation.
    This algorithm is quite stable, numerically, but even in a world of
    exact computation, unless the x coordinates are chosen very
    carefully - Chebyshev zeros (e.g., cos(i*pi/n)) are a good choice -
    polynomial interpolation itself is a very ill-conditioned process
    due to the Runge phenomenon.

    Based on Berrut and Trefethen 2004, "Barycentric Lagrange Interpolation".

    Examples
    --------
    To produce a quintic barycentric interpolant approximating the function
    :math:`\sin x`, and its first four derivatives, using six randomly-spaced
    nodes in :math:`(0, \frac{\pi}{2})`:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.interpolate import BarycentricInterpolator
    >>> rng = np.random.default_rng()
    >>> xi = rng.random(6) * np.pi/2
    >>> f, f_d1, f_d2, f_d3, f_d4 = np.sin, np.cos, lambda x: -np.sin(x), lambda x: -np.cos(x), np.sin
    >>> P = BarycentricInterpolator(xi, f(xi), random_state=rng)
    >>> fig, axs = plt.subplots(5, 1, sharex=True, layout='constrained', figsize=(7,10))
    >>> x = np.linspace(0, np.pi, 100)
    >>> axs[0].plot(x, P(x), 'r:', x, f(x), 'k--', xi, f(xi), 'xk')
    >>> axs[1].plot(x, P.derivative(x), 'r:', x, f_d1(x), 'k--', xi, f_d1(xi), 'xk')
    >>> axs[2].plot(x, P.derivative(x, 2), 'r:', x, f_d2(x), 'k--', xi, f_d2(xi), 'xk')
    >>> axs[3].plot(x, P.derivative(x, 3), 'r:', x, f_d3(x), 'k--', xi, f_d3(xi), 'xk')
    >>> axs[4].plot(x, P.derivative(x, 4), 'r:', x, f_d4(x), 'k--', xi, f_d4(xi), 'xk')
    >>> axs[0].set_xlim(0, np.pi)
    >>> axs[4].set_xlabel(r"$x$")
    >>> axs[4].set_xticks([i * np.pi / 4 for i in range(5)],
    ...                   ["0", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"])
    >>> axs[0].set_ylabel("$f(x)$")
    >>> axs[1].set_ylabel("$f'(x)$")
    >>> axs[2].set_ylabel("$f''(x)$")
    >>> axs[3].set_ylabel("$f^{(3)}(x)$")
    >>> axs[4].set_ylabel("$f^{(4)}(x)$")
    >>> labels = ['Interpolation nodes', 'True function $f$', 'Barycentric interpolation']
    >>> axs[0].legend(axs[0].get_lines()[::-1], labels, bbox_to_anchor=(0., 1.02, 1., .102),
    ...               loc='lower left', ncols=3, mode="expand", borderaxespad=0., frameon=False)
    >>> plt.show()
    """ # numpy/numpydoc#87  # noqa: E501

    @_transition_to_rng("random_state", replace_doc=False)
    def __init__(self, xi, yi=None, axis=0, *, wi=None, rng=None):
        super().__init__(xi, yi, axis)

        rng = check_random_state(rng)

        self.xi = np.asarray(xi, dtype=np.float64)
        self.set_yi(yi)
        self.n = len(self.xi)

        # cache derivative object to avoid re-computing the weights with every call.
        self._diff_cij = None

        if wi is not None:
            self.wi = wi
        else:
            # See page 510 of Berrut and Trefethen 2004 for an explanation of the
            # capacity scaling and the suggestion of using a random permutation of
            # the input factors.
            # At the moment, the permutation is not performed for xi that are
            # appended later through the add_xi interface. It's not clear to me how
            # to implement that and it seems that most situations that require
            # these numerical stability improvements will be able to provide all
            # the points to the constructor.
            self._inv_capacity = 4.0 / (np.max(self.xi) - np.min(self.xi))
            permute = rng.permutation(self.n, )
            inv_permute = np.zeros(self.n, dtype=np.int32)
            inv_permute[permute] = np.arange(self.n)
            self.wi = np.zeros(self.n)

            for i in range(self.n):
                dist = self._inv_capacity * (self.xi[i] - self.xi[permute])
                dist[inv_permute[i]] = 1.0
                prod = np.prod(dist)
                if prod == 0.0:
                    raise ValueError("Interpolation points xi must be"
                                     " distinct.")
                self.wi[i] = 1.0 / prod

    def set_yi(self, yi, axis=None):
        """
        Update the y values to be interpolated

        The barycentric interpolation algorithm requires the calculation
        of weights, but these depend only on the `xi`. The `yi` can be changed
        at any time.

        Parameters
        ----------
        yi : array_like
            The y-coordinates of the points the polynomial will pass through.
            If None, the y values must be supplied later.
        axis : int, optional
            Axis in the `yi` array corresponding to the x-coordinate values.

        """
        if yi is None:
            self.yi = None
            return
        self._set_yi(yi, xi=self.xi, axis=axis)
        self.yi = self._reshape_yi(yi)
        self.n, self.r = self.yi.shape
        self._diff_baryint = None

    def add_xi(self, xi, yi=None):
        """
        Add more x values to the set to be interpolated

        The barycentric interpolation algorithm allows easy updating by
        adding more points for the polynomial to pass through.

        Parameters
        ----------
        xi : array_like
            The x coordinates of the points that the polynomial should pass
            through.
        yi : array_like, optional
            The y coordinates of the points the polynomial should pass through.
            Should have shape ``(xi.size, R)``; if R > 1 then the polynomial is
            vector-valued.
            If `yi` is not given, the y values will be supplied later. `yi`
            should be given if and only if the interpolator has y values
            specified.

        Notes
        -----
        The new points added by `add_xi` are not randomly permuted
        so there is potential for numerical instability,
        especially for a large number of points. If this
        happens, please reconstruct interpolation from scratch instead.
        """
        if yi is not None:
            if self.yi is None:
                raise ValueError("No previous yi value to update!")
            yi = self._reshape_yi(yi, check=True)
            self.yi = np.vstack((self.yi,yi))
        else:
            if self.yi is not None:
                raise ValueError("No update to yi provided!")
        old_n = self.n
        self.xi = np.concatenate((self.xi,xi))
        self.n = len(self.xi)
        self.wi **= -1
        old_wi = self.wi
        self.wi = np.zeros(self.n)
        self.wi[:old_n] = old_wi
        for j in range(old_n, self.n):
            self.wi[:j] *= self._inv_capacity * (self.xi[j]-self.xi[:j])
            self.wi[j] = np.multiply.reduce(
                self._inv_capacity * (self.xi[:j]-self.xi[j])
            )
        self.wi **= -1
        self._diff_cij = None
        self._diff_baryint = None

    def __call__(self, x):
        """Evaluate the interpolating polynomial at the points x

        Parameters
        ----------
        x : array_like
            Point or points at which to evaluate the interpolant.

        Returns
        -------
        y : array_like
            Interpolated values. Shape is determined by replacing
            the interpolation axis in the original array with the shape of `x`.

        Notes
        -----
        Currently the code computes an outer product between `x` and the
        weights, that is, it constructs an intermediate array of size
        ``(N, len(x))``, where N is the degree of the polynomial.
        """
        return _Interpolator1D.__call__(self, x)

    def _evaluate(self, x):
        if x.size == 0:
            p = np.zeros((0, self.r), dtype=self.dtype)
        else:
            c = x[..., np.newaxis] - self.xi
            z = c == 0
            c[z] = 1
            c = self.wi / c
            with np.errstate(divide='ignore'):
                p = np.dot(c, self.yi) / np.sum(c, axis=-1)[..., np.newaxis]
            # Now fix where x==some xi
            r = np.nonzero(z)
            if len(r) == 1:  # evaluation at a scalar
                if len(r[0]) > 0:  # equals one of the points
                    p = self.yi[r[0][0]]
            else:
                p[r[:-1]] = self.yi[r[-1]]
        return p

    def derivative(self, x, der=1):
        """
        Evaluate a single derivative of the polynomial at the point x.

        Parameters
        ----------
        x : array_like
            Point or points at which to evaluate the derivatives
        der : integer, optional
            Which derivative to evaluate (default: first derivative).
            This number includes the function value as 0th derivative.

        Returns
        -------
        d : ndarray
            Derivative interpolated at the x-points. Shape of `d` is
            determined by replacing the interpolation axis in the
            original array with the shape of `x`.
        """
        x, x_shape = self._prepare_x(x)
        y = self._evaluate_derivatives(x, der+1, all_lower=False)
        return self._finish_y(y, x_shape)

    def _evaluate_derivatives(self, x, der=None, all_lower=True):
        # NB: der here is not the order of the highest derivative;
        # instead, it is the size of the derivatives matrix that
        # would be returned with all_lower=True, including the
        # '0th' derivative (the undifferentiated function).
        # E.g. to evaluate the 5th derivative alone, call
        # _evaluate_derivatives(x, der=6, all_lower=False).

        if (not all_lower) and (x.size == 0 or self.r == 0):
            return np.zeros((0, self.r), dtype=self.dtype)

        if (not all_lower) and der == 1:
            return self._evaluate(x)

        if (not all_lower) and (der > self.n):
            return np.zeros((len(x), self.r), dtype=self.dtype)

        if der is None:
            der = self.n

        if all_lower and (x.size == 0 or self.r == 0):
            return np.zeros((der, len(x), self.r), dtype=self.dtype)

        if self._diff_cij is None:
            # c[i,j] = xi[i] - xi[j]
            c = self.xi[:, np.newaxis] - self.xi

            # avoid division by 0 (diagonal entries are so far zero by construction)
            np.fill_diagonal(c, 1)

            # c[i,j] = (w[j] / w[i]) / (xi[i] - xi[j]) (equation 9.4)
            c = self.wi/ (c * self.wi[..., np.newaxis])

            # fill in correct diagonal entries: each column sums to 0
            np.fill_diagonal(c, 0)

            # calculate diagonal
            # c[j,j] = -sum_{i != j} c[i,j] (equation 9.5)
            d = -c.sum(axis=1)
            # c[i,j] = l_j(x_i)
            np.fill_diagonal(c, d)

            self._diff_cij = c

        if self._diff_baryint is None:
            # initialise and cache derivative interpolator and cijs;
            # reuse weights wi (which depend only on interpolation points xi),
            # to avoid unnecessary re-computation
            self._diff_baryint = BarycentricInterpolator(xi=self.xi,
                                                         yi=self._diff_cij @ self.yi,
                                                         wi=self.wi)
            self._diff_baryint._diff_cij = self._diff_cij

        if all_lower:
            # assemble matrix of derivatives from order 0 to order der-1,
            # in the format required by _Interpolator1DWithDerivatives.
            cn = np.zeros((der, len(x), self.r), dtype=self.dtype)
            for d in range(der):
                cn[d, :, :] = self._evaluate_derivatives(x, d+1, all_lower=False)
            return cn

        # recursively evaluate only the derivative requested
        return self._diff_baryint._evaluate_derivatives(x, der-1, all_lower=False)


def barycentric_interpolate(xi, yi, x, axis=0, *, der=0, rng=None):
    """
    Convenience function for polynomial interpolation.

    Constructs a polynomial that passes through a given set of points,
    then evaluates the polynomial. For reasons of numerical stability,
    this function does not compute the coefficients of the polynomial.

    This function uses a "barycentric interpolation" method that treats
    the problem as a special case of rational function interpolation.
    This algorithm is quite stable, numerically, but even in a world of
    exact computation, unless the `x` coordinates are chosen very
    carefully - Chebyshev zeros (e.g., cos(i*pi/n)) are a good choice -
    polynomial interpolation itself is a very ill-conditioned process
    due to the Runge phenomenon.

    Parameters
    ----------
    xi : array_like
        1-D array of x coordinates of the points the polynomial should
        pass through
    yi : array_like
        The y coordinates of the points the polynomial should pass through.
    x : scalar or array_like
        Point or points at which to evaluate the interpolant.
    axis : int, optional
        Axis in the `yi` array corresponding to the x-coordinate values.
    der : int or list or None, optional
        How many derivatives to evaluate, or None for all potentially
        nonzero derivatives (that is, a number equal to the number
        of points), or a list of derivatives to evaluate. This number
        includes the function value as the '0th' derivative.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.

    Returns
    -------
    y : scalar or array_like
        Interpolated values. Shape is determined by replacing
        the interpolation axis in the original array with the shape of `x`.

    See Also
    --------
    BarycentricInterpolator : Barycentric interpolator

    Notes
    -----
    Construction of the interpolation weights is a relatively slow process.
    If you want to call this many times with the same xi (but possibly
    varying yi or x) you should use the class `BarycentricInterpolator`.
    This is what this function uses internally.

    Examples
    --------
    We can interpolate 2D observed data using barycentric interpolation:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.interpolate import barycentric_interpolate
    >>> x_observed = np.linspace(0.0, 10.0, 11)
    >>> y_observed = np.sin(x_observed)
    >>> x = np.linspace(min(x_observed), max(x_observed), num=100)
    >>> y = barycentric_interpolate(x_observed, y_observed, x)
    >>> plt.plot(x_observed, y_observed, "o", label="observation")
    >>> plt.plot(x, y, label="barycentric interpolation")
    >>> plt.legend()
    >>> plt.show()

    """
    P = BarycentricInterpolator(xi, yi, axis=axis, rng=rng)
    if der == 0:
        return P(x)
    elif _isscalar(der):
        return P.derivative(x, der=der)
    else:
        return P.derivatives(x, der=np.amax(der)+1)[der]
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.interpolate` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'BSpline',
    'bisplev',
    'bisplrep',
    'insert',
    'spalde',
    'splantider',
    'splder',
    'splev',
    'splint',
    'splprep',
    'splrep',
    'sproot',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="interpolate", module="fitpack",
                                   private_modules=["_fitpack_py"], all=__all__,
                                   attribute=name)
<|endoftext|>"""
Convenience interface to N-D interpolation

.. versionadded:: 0.9

"""
import numpy as np
from ._interpnd import (LinearNDInterpolator, NDInterpolatorBase,
     CloughTocher2DInterpolator, _ndim_coords_from_arrays)
from scipy.spatial import cKDTree

__all__ = ['griddata', 'NearestNDInterpolator', 'LinearNDInterpolator',
           'CloughTocher2DInterpolator']

#------------------------------------------------------------------------------
# Nearest-neighbor interpolation
#------------------------------------------------------------------------------


class NearestNDInterpolator(NDInterpolatorBase):
    """NearestNDInterpolator(x, y).

    Nearest-neighbor interpolator in N > 1 dimensions.

    .. versionadded:: 0.9

    Methods
    -------
    __call__

    Parameters
    ----------
    x : (npoints, ndims) 2-D ndarray of floats
        Data point coordinates.
    y : (npoints, ) 1-D ndarray of float or complex
        Data values.
    rescale : boolean, optional
        Rescale points to unit cube before performing interpolation.
        This is useful if some of the input dimensions have
        incommensurable units and differ by many orders of magnitude.

        .. versionadded:: 0.14.0
    tree_options : dict, optional
        Options passed to the underlying ``cKDTree``.

        .. versionadded:: 0.17.0

    See Also
    --------
    griddata :
        Interpolate unstructured D-D data.
    LinearNDInterpolator :
        Piecewise linear interpolator in N dimensions.
    CloughTocher2DInterpolator :
        Piecewise cubic, C1 smooth, curvature-minimizing interpolator in 2D.
    interpn : Interpolation on a regular grid or rectilinear grid.
    RegularGridInterpolator : Interpolator on a regular or rectilinear grid
                              in arbitrary dimensions (`interpn` wraps this
                              class).

    Notes
    -----
    Uses ``scipy.spatial.cKDTree``

    .. note:: For data on a regular grid use `interpn` instead.

    Examples
    --------
    We can interpolate values on a 2D plane:

    >>> from scipy.interpolate import NearestNDInterpolator
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> x = rng.random(10) - 0.5
    >>> y = rng.random(10) - 0.5
    >>> z = np.hypot(x, y)
    >>> X = np.linspace(min(x), max(x))
    >>> Y = np.linspace(min(y), max(y))
    >>> X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
    >>> interp = NearestNDInterpolator(list(zip(x, y)), z)
    >>> Z = interp(X, Y)
    >>> plt.pcolormesh(X, Y, Z, shading='auto')
    >>> plt.plot(x, y, "ok", label="input point")
    >>> plt.legend()
    >>> plt.colorbar()
    >>> plt.axis("equal")
    >>> plt.show()

    """

    def __init__(self, x, y, rescale=False, tree_options=None):
        NDInterpolatorBase.__init__(self, x, y, rescale=rescale,
                                    need_contiguous=False,
                                    need_values=False)
        if tree_options is None:
            tree_options = dict()
        self.tree = cKDTree(self.points, **tree_options)
        self.values = np.asarray(y)

    def __call__(self, *args, **query_options):
        """
        Evaluate interpolator at given points.

        Parameters
        ----------
        x1, x2, ... xn : array-like of float
            Points where to interpolate data at.
            x1, x2, ... xn can be array-like of float with broadcastable shape.
            or x1 can be array-like of float with shape ``(..., ndim)``
        **query_options
            This allows ``eps``, ``p``, ``distance_upper_bound``, and ``workers``
            being passed to the cKDTree's query function to be explicitly set.
            See `scipy.spatial.cKDTree.query` for an overview of the different options.

            .. versionadded:: 1.12.0

        """
        # For the sake of enabling subclassing, NDInterpolatorBase._set_xi performs
        # some operations which are not required by NearestNDInterpolator.__call__, 
        # hence here we operate on xi directly, without calling a parent class function.
        xi = _ndim_coords_from_arrays(args, ndim=self.points.shape[1])
        xi = self._check_call_shape(xi)
        xi = self._scale_x(xi)

        # We need to handle two important cases:
        # (1) the case where xi has trailing dimensions (..., ndim), and
        # (2) the case where y has trailing dimensions
        # We will first flatten xi to deal with case (1),
        # do the computation in flattened array while retaining y's dimensionality,
        # and then reshape the interpolated values back to match xi's shape.

        # Flatten xi for the query
        xi_flat = xi.reshape(-1, xi.shape[-1])
        original_shape = xi.shape
        flattened_shape = xi_flat.shape

        # if distance_upper_bound is set to not be infinite,
        # then we need to consider the case where cKDtree
        # does not find any points within distance_upper_bound to return.
        # It marks those points as having infinte distance, which is what will be used
        # below to mask the array and return only the points that were deemed
        # to have a close enough neighbor to return something useful.
        dist, i = self.tree.query(xi_flat, **query_options)
        valid_mask = np.isfinite(dist)

        # create a holder interp_values array and fill with nans.
        if self.values.ndim > 1:
            interp_shape = flattened_shape[:-1] + self.values.shape[1:]
        else:
            interp_shape = flattened_shape[:-1]

        if np.issubdtype(self.values.dtype, np.complexfloating):
            interp_values = np.full(interp_shape, np.nan, dtype=self.values.dtype)
        else:
            interp_values = np.full(interp_shape, np.nan)

        interp_values[valid_mask] = self.values[i[valid_mask], ...]

        if self.values.ndim > 1:
            new_shape = original_shape[:-1] + self.values.shape[1:]
        else:
            new_shape = original_shape[:-1]
        interp_values = interp_values.reshape(new_shape)

        return interp_values


#------------------------------------------------------------------------------
# Convenience interface function
#------------------------------------------------------------------------------


def griddata(points, values, xi, method='linear', fill_value=np.nan,
             rescale=False):
    """
    Interpolate unstructured D-D data.

    Parameters
    ----------
    points : 2-D ndarray of floats with shape (n, D), or length D tuple of 1-D ndarrays with shape (n,).
        Data point coordinates.
    values : ndarray of float or complex, shape (n,)
        Data values.
    xi : 2-D ndarray of floats with shape (m, D), or length D tuple of ndarrays broadcastable to the same shape.
        Points at which to interpolate data.
    method : {'linear', 'nearest', 'cubic'}, optional
        Method of interpolation. One of

        ``nearest``
          return the value at the data point closest to
          the point of interpolation. See `NearestNDInterpolator` for
          more details.

        ``linear``
          tessellate the input point set to N-D
          simplices, and interpolate linearly on each simplex. See
          `LinearNDInterpolator` for more details.

        ``cubic`` (1-D)
          return the value determined from a cubic
          spline.

        ``cubic`` (2-D)
          return the value determined from a
          piecewise cubic, continuously differentiable (C1), and
          approximately curvature-minimizing polynomial surface. See
          `CloughTocher2DInterpolator` for more details.
    fill_value : float, optional
        Value used to fill in for requested points outside of the
        convex hull of the input points. If not provided, then the
        default is ``nan``. This option has no effect for the
        'nearest' method.
    rescale : bool, optional
        Rescale points to unit cube before performing interpolation.
        This is useful if some of the input dimensions have
        incommensurable units and differ by many orders of magnitude.

        .. versionadded:: 0.14.0

    Returns
    -------
    ndarray
        Array of interpolated values.

    See Also
    --------
    LinearNDInterpolator :
        Piecewise linear interpolator in N dimensions.
    NearestNDInterpolator :
        Nearest-neighbor interpolator in N dimensions.
    CloughTocher2DInterpolator :
        Piecewise cubic, C1 smooth, curvature-minimizing interpolator in 2D.
    interpn : Interpolation on a regular grid or rectilinear grid.
    RegularGridInterpolator : Interpolator on a regular or rectilinear grid
                              in arbitrary dimensions (`interpn` wraps this
                              class).

    Notes
    -----

    .. versionadded:: 0.9

    .. note:: For data on a regular grid use `interpn` instead.

    Examples
    --------

    Suppose we want to interpolate the 2-D function

    >>> import numpy as np
    >>> def func(x, y):
    ...     return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2

    on a grid in [0, 1]x[0, 1]

    >>> grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

    but we only know its values at 1000 data points:

    >>> rng = np.random.default_rng()
    >>> points = rng.random((1000, 2))
    >>> values = func(points[:,0], points[:,1])

    This can be done with `griddata` -- below we try out all of the
    interpolation methods:

    >>> from scipy.interpolate import griddata
    >>> grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
    >>> grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
    >>> grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')

    One can see that the exact result is reproduced by all of the
    methods to some degree, but for this smooth function the piecewise
    cubic interpolant gives the best results:

    >>> import matplotlib.pyplot as plt
    >>> plt.subplot(221)
    >>> plt.imshow(func(grid_x, grid_y).T, extent=(0,1,0,1), origin='lower')
    >>> plt.plot(points[:,0], points[:,1], 'k.', ms=1)
    >>> plt.title('Original')
    >>> plt.subplot(222)
    >>> plt.imshow(grid_z0.T, extent=(0,1,0,1), origin='lower')
    >>> plt.title('Nearest')
    >>> plt.subplot(223)
    >>> plt.imshow(grid_z1.T, extent=(0,1,0,1), origin='lower')
    >>> plt.title('Linear')
    >>> plt.subplot(224)
    >>> plt.imshow(grid_z2.T, extent=(0,1,0,1), origin='lower')
    >>> plt.title('Cubic')
    >>> plt.gcf().set_size_inches(6, 6)
    >>> plt.show()

    """ # numpy/numpydoc#87  # noqa: E501

    points = _ndim_coords_from_arrays(points)

    if points.ndim < 2:
        ndim = points.ndim
    else:
        ndim = points.shape[-1]

    if ndim == 1 and method in ('nearest', 'linear', 'cubic'):
        from ._interpolate import interp1d
        points = points.ravel()
        if isinstance(xi, tuple):
            if len(xi) != 1:
                raise ValueError("invalid number of dimensions in xi")
            xi, = xi
        # Sort points/values together, necessary as input for interp1d
        idx = np.argsort(points)
        points = points[idx]
        values = values[idx]
        if method == 'nearest':
            fill_value = 'extrapolate'
        ip = interp1d(points, values, kind=method, axis=0, bounds_error=False,
                      fill_value=fill_value)
        return ip(xi)
    elif method == 'nearest':
        ip = NearestNDInterpolator(points, values, rescale=rescale)
        return ip(xi)
    elif method == 'linear':
        ip = LinearNDInterpolator(points, values, fill_value=fill_value,
                                  rescale=rescale)
        return ip(xi)
    elif method == 'cubic' and ndim == 2:
        ip = CloughTocher2DInterpolator(points, values, fill_value=fill_value,
                                        rescale=rescale)
        return ip(xi)
    else:
        raise ValueError("Unknown interpolation method %r for "
                         "%d dimensional data" % (method, ndim))
<|endoftext|>"""rbf - Radial basis functions for interpolation/smoothing scattered N-D data.

Written by John Travers <jtravs@gmail.com>, February 2007
Based closely on Matlab code by Alex Chirokov
Additional, large, improvements by Robert Hetland
Some additional alterations by Travis Oliphant
Interpolation with multi-dimensional target domain by Josua Sassen

Permission to use, modify, and distribute this software is given under the
terms of the SciPy (BSD style) license. See LICENSE.txt that came with
this distribution for specifics.

NO WARRANTY IS EXPRESSED OR IMPLIED. USE AT YOUR OWN RISK.

Copyright (c) 2006-2007, Robert Hetland <hetland@tamu.edu>
Copyright (c) 2007, John Travers <jtravs@gmail.com>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of Robert Hetland nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import numpy as np

from scipy import linalg
from scipy.special import xlogy
from scipy.spatial.distance import cdist, pdist, squareform

__all__ = ['Rbf']


class Rbf:
    """
    Rbf(*args, **kwargs)

    A class for radial basis function interpolation of functions from
    N-D scattered data to an M-D domain.

    .. legacy:: class

        `Rbf` is legacy code, for new usage please use `RBFInterpolator`
        instead.

    Parameters
    ----------
    *args : arrays
        x, y, z, ..., d, where x, y, z, ... are the coordinates of the nodes
        and d is the array of values at the nodes
    function : str or callable, optional
        The radial basis function, based on the radius, r, given by the norm
        (default is Euclidean distance); the default is 'multiquadric'::

            'multiquadric': sqrt((r/self.epsilon)**2 + 1)
            'inverse': 1.0/sqrt((r/self.epsilon)**2 + 1)
            'gaussian': exp(-(r/self.epsilon)**2)
            'linear': r
            'cubic': r**3
            'quintic': r**5
            'thin_plate': r**2 * log(r)

        If callable, then it must take 2 arguments (self, r). The epsilon
        parameter will be available as self.epsilon. Other keyword
        arguments passed in will be available as well.

    epsilon : float, optional
        Adjustable constant for gaussian or multiquadrics functions
        - defaults to approximate average distance between nodes (which is
        a good start).
    smooth : float, optional
        Values greater than zero increase the smoothness of the
        approximation. 0 is for interpolation (default), the function will
        always go through the nodal points in this case.
    norm : str, callable, optional
        A function that returns the 'distance' between two points, with
        inputs as arrays of positions (x, y, z, ...), and an output as an
        array of distance. E.g., the default: 'euclidean', such that the result
        is a matrix of the distances from each point in ``x1`` to each point in
        ``x2``. For more options, see documentation of
        `scipy.spatial.distances.cdist`.
    mode : str, optional
        Mode of the interpolation, can be '1-D' (default) or 'N-D'. When it is
        '1-D' the data `d` will be considered as 1-D and flattened
        internally. When it is 'N-D' the data `d` is assumed to be an array of
        shape (n_samples, m), where m is the dimension of the target domain.


    Attributes
    ----------
    N : int
        The number of data points (as determined by the input arrays).
    di : ndarray
        The 1-D array of data values at each of the data coordinates `xi`.
    xi : ndarray
        The 2-D array of data coordinates.
    function : str or callable
        The radial basis function. See description under Parameters.
    epsilon : float
        Parameter used by gaussian or multiquadrics functions. See Parameters.
    smooth : float
        Smoothing parameter. See description under Parameters.
    norm : str or callable
        The distance function. See description under Parameters.
    mode : str
        Mode of the interpolation. See description under Parameters.
    nodes : ndarray
        A 1-D array of node values for the interpolation.
    A : internal property, do not use

    See Also
    --------
    RBFInterpolator

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.interpolate import Rbf
    >>> rng = np.random.default_rng()
    >>> x, y, z, d = rng.random((4, 50))
    >>> rbfi = Rbf(x, y, z, d)  # radial basis function interpolator instance
    >>> xi = yi = zi = np.linspace(0, 1, 20)
    >>> di = rbfi(xi, yi, zi)   # interpolated values
    >>> di.shape
    (20,)

    """
    # Available radial basis functions that can be selected as strings;
    # they all start with _h_ (self._init_function relies on that)
    def _h_multiquadric(self, r):
        return np.sqrt((1.0/self.epsilon*r)**2 + 1)

    def _h_inverse_multiquadric(self, r):
        return 1.0/np.sqrt((1.0/self.epsilon*r)**2 + 1)

    def _h_gaussian(self, r):
        return np.exp(-(1.0/self.epsilon*r)**2)

    def _h_linear(self, r):
        return r

    def _h_cubic(self, r):
        return r**3

    def _h_quintic(self, r):
        return r**5

    def _h_thin_plate(self, r):
        return xlogy(r**2, r)

    # Setup self._function and do smoke test on initial r
    def _init_function(self, r):
        if isinstance(self.function, str):
            self.function = self.function.lower()
            _mapped = {'inverse': 'inverse_multiquadric',
                       'inverse multiquadric': 'inverse_multiquadric',
                       'thin-plate': 'thin_plate'}
            if self.function in _mapped:
                self.function = _mapped[self.function]

            func_name = "_h_" + self.function
            if hasattr(self, func_name):
                self._function = getattr(self, func_name)
            else:
                functionlist = [x[3:] for x in dir(self)
                                if x.startswith('_h_')]
                raise ValueError("function must be a callable or one of " +
                                 ", ".join(functionlist))
            self._function = getattr(self, "_h_"+self.function)
        elif callable(self.function):
            allow_one = False
            if hasattr(self.function, 'func_code') or \
               hasattr(self.function, '__code__'):
                val = self.function
                allow_one = True
            elif hasattr(self.function, "__call__"):
                val = self.function.__call__.__func__
            else:
                raise ValueError("Cannot determine number of arguments to "
                                 "function")

            argcount = val.__code__.co_argcount
            if allow_one and argcount == 1:
                self._function = self.function
            elif argcount == 2:
                self._function = self.function.__get__(self, Rbf)
            else:
                raise ValueError("Function argument must take 1 or 2 "
                                 "arguments.")

        a0 = self._function(r)
        if a0.shape != r.shape:
            raise ValueError("Callable must take array and return array of "
                             "the same shape")
        return a0

    def __init__(self, *args, **kwargs):
        # `args` can be a variable number of arrays; we flatten them and store
        # them as a single 2-D array `xi` of shape (n_args-1, array_size),
        # plus a 1-D array `di` for the values.
        # All arrays must have the same number of elements
        self.xi = np.asarray([np.asarray(a, dtype=np.float64).flatten()
                              for a in args[:-1]])
        self.N = self.xi.shape[-1]

        self.mode = kwargs.pop('mode', '1-D')

        if self.mode == '1-D':
            self.di = np.asarray(args[-1]).flatten()
            self._target_dim = 1
        elif self.mode == 'N-D':
            self.di = np.asarray(args[-1])
            self._target_dim = self.di.shape[-1]
        else:
            raise ValueError("Mode has to be 1-D or N-D.")

        if not all([x.size == self.di.shape[0] for x in self.xi]):
            raise ValueError("All arrays must be equal length.")

        self.norm = kwargs.pop('norm', 'euclidean')
        self.epsilon = kwargs.pop('epsilon', None)
        if self.epsilon is None:
            # default epsilon is the "the average distance between nodes" based
            # on a bounding hypercube
            ximax = np.amax(self.xi, axis=1)
            ximin = np.amin(self.xi, axis=1)
            edges = ximax - ximin
            edges = edges[np.nonzero(edges)]
            self.epsilon = np.power(np.prod(edges)/self.N, 1.0/edges.size)

        self.smooth = kwargs.pop('smooth', 0.0)
        self.function = kwargs.pop('function', 'multiquadric')

        # attach anything left in kwargs to self for use by any user-callable
        # function or to save on the object returned.
        for item, value in kwargs.items():
            setattr(self, item, value)

        # Compute weights
        if self._target_dim > 1:  # If we have more than one target dimension,
            # we first factorize the matrix
            self.nodes = np.zeros((self.N, self._target_dim), dtype=self.di.dtype)
            lu, piv = linalg.lu_factor(self.A)
            for i in range(self._target_dim):
                self.nodes[:, i] = linalg.lu_solve((lu, piv), self.di[:, i])
        else:
            self.nodes = linalg.solve(self.A, self.di)

    @property
    def A(self):
        # this only exists for backwards compatibility: self.A was available
        # and, at least technically, public.
        r = squareform(pdist(self.xi.T, self.norm))  # Pairwise norm
        return self._init_function(r) - np.eye(self.N)*self.smooth

    def _call_norm(self, x1, x2):
        return cdist(x1.T, x2.T, self.norm)

    def __call__(self, *args):
        args = [np.asarray(x) for x in args]
        if not all([x.shape == y.shape for x in args for y in args]):
            raise ValueError("Array lengths must be equal")
        if self._target_dim > 1:
            shp = args[0].shape + (self._target_dim,)
        else:
            shp = args[0].shape
        xa = np.asarray([a.flatten() for a in args], dtype=np.float64)
        r = self._call_norm(xa, self.xi)
        return np.dot(self._function(r), self.nodes).reshape(shp)
<|endoftext|>"""
========================================
Interpolation (:mod:`scipy.interpolate`)
========================================

.. currentmodule:: scipy.interpolate

Sub-package for objects used in interpolation.

As listed below, this sub-package contains spline functions and classes,
1-D and multidimensional (univariate and multivariate)
interpolation classes, Lagrange and Taylor polynomial interpolators, and
wrappers for `FITPACK <http://www.netlib.org/dierckx/>`__
and DFITPACK functions.

Univariate interpolation
========================

.. autosummary::
   :toctree: generated/

   interp1d
   BarycentricInterpolator
   KroghInterpolator
   barycentric_interpolate
   krogh_interpolate
   pchip_interpolate
   CubicHermiteSpline
   PchipInterpolator
   Akima1DInterpolator
   CubicSpline
   PPoly
   BPoly
   FloaterHormannInterpolator


Multivariate interpolation
==========================

Unstructured data:

.. autosummary::
   :toctree: generated/

   griddata
   LinearNDInterpolator
   NearestNDInterpolator
   CloughTocher2DInterpolator
   RBFInterpolator
   Rbf
   interp2d

For data on a grid:

.. autosummary::
   :toctree: generated/

   interpn
   RegularGridInterpolator
   RectBivariateSpline

.. seealso::

    `scipy.ndimage.map_coordinates`

Tensor product polynomials:

.. autosummary::
   :toctree: generated/

   NdPPoly
   NdBSpline

1-D Splines
===========

.. autosummary::
   :toctree: generated/

   BSpline
   make_interp_spline
   make_lsq_spline
   make_smoothing_spline
   generate_knots
   make_splrep
   make_splprep

Functional interface to FITPACK routines:

.. autosummary::
   :toctree: generated/

   splrep
   splprep
   splev
   splint
   sproot
   spalde
   splder
   splantider
   insert

Object-oriented FITPACK interface:

.. autosummary::
   :toctree: generated/

   UnivariateSpline
   InterpolatedUnivariateSpline
   LSQUnivariateSpline



2-D Splines
===========

For data on a grid:

.. autosummary::
   :toctree: generated/

   RectBivariateSpline
   RectSphereBivariateSpline

For unstructured data:

.. autosummary::
   :toctree: generated/

   BivariateSpline
   SmoothBivariateSpline
   SmoothSphereBivariateSpline
   LSQBivariateSpline
   LSQSphereBivariateSpline

Low-level interface to FITPACK functions:

.. autosummary::
   :toctree: generated/

   bisplrep
   bisplev

Rational Approximation
======================

.. autosummary::
   :toctree: generated/

   pade
   AAA

Additional tools
================

.. autosummary::
   :toctree: generated/

   lagrange
   approximate_taylor_polynomial

.. seealso::

   `scipy.ndimage.map_coordinates`,
   `scipy.ndimage.spline_filter`,
   `scipy.signal.resample`,
   `scipy.signal.bspline`,
   `scipy.signal.gauss_spline`,
   `scipy.signal.qspline1d`,
   `scipy.signal.cspline1d`,
   `scipy.signal.qspline1d_eval`,
   `scipy.signal.cspline1d_eval`,
   `scipy.signal.qspline2d`,
   `scipy.signal.cspline2d`.

``pchip`` is an alias of `PchipInterpolator` for backward compatibility
(should not be used in new code).
"""
from ._interpolate import *
from ._fitpack_py import *

# New interface to fitpack library:
from ._fitpack2 import *

from ._rbf import Rbf

from ._rbfinterp import *

from ._polyint import *

from ._cubic import *

from ._ndgriddata import *

from ._bsplines import *
from ._fitpack_repro import generate_knots, make_splrep, make_splprep

from ._pade import *

from ._rgi import *

from ._ndbspline import NdBSpline

from ._bary_rational import *

# Deprecated namespaces, to be removed in v2.0.0
from . import fitpack, fitpack2, interpolate, ndgriddata, polyint, rbf, interpnd

__all__ = [s for s in dir() if not s.startswith('_')]

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester

# Backward compatibility
pchip = PchipInterpolator
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.interpolate` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'bispeu',
    'bispev',
    'curfit',
    'dblint',
    'fpchec',
    'fpcurf0',
    'fpcurf1',
    'fpcurfm1',
    'parcur',
    'parder',
    'pardeu',
    'pardtc',
    'percur',
    'regrid_smth',
    'regrid_smth_spher',
    'spalde',
    'spherfit_lsq',
    'spherfit_smth',
    'splder',
    'splev',
    'splint',
    'sproot',
    'surfit_lsq',
    'surfit_smth',
    'types',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="interpolate", module="dfitpack",
                                   private_modules=["_dfitpack"], all=__all__,
                                   attribute=name)
<|endoftext|>""" Replicate FITPACK's logic for constructing smoothing spline functions and curves.

    Currently provides analogs of splrep and splprep python routines, i.e.
    curfit.f and parcur.f routines (the drivers are fpcurf.f and fppara.f, respectively)

    The Fortran sources are from
    https://github.com/scipy/scipy/blob/maintenance/1.11.x/scipy/interpolate/fitpack/

    .. [1] P. Dierckx, "Algorithms for smoothing data with periodic and
        parametric splines, Computer Graphics and Image Processing",
        20 (1982) 171-184.
        :doi:`10.1016/0146-664X(82)90043-0`.
    .. [2] P. Dierckx, "Curve and surface fitting with splines", Monographs on
         Numerical Analysis, Oxford University Press, 1993.
    .. [3] P. Dierckx, "An algorithm for smoothing, differentiation and integration
         of experimental data using spline functions",
         Journal of Computational and Applied Mathematics, vol. I, no 3, p. 165 (1975).
         https://doi.org/10.1016/0771-050X(75)90034-0
"""
import warnings
import operator
import numpy as np

from ._bsplines import (
    _not_a_knot, make_interp_spline, BSpline, fpcheck, _lsq_solve_qr
)
from . import _dierckx      # type: ignore[attr-defined]


#    cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
#    c  part 1: determination of the number of knots and their position     c
#    c  **************************************************************      c
#
# https://github.com/scipy/scipy/blob/maintenance/1.11.x/scipy/interpolate/fitpack/fpcurf.f#L31


# Hardcoded in curfit.f
TOL = 0.001
MAXIT = 20


def _get_residuals(x, y, t, k, w):
    # FITPACK has (w*(spl(x)-y))**2; make_lsq_spline has w*(spl(x)-y)**2
    w2 = w**2

    # inline the relevant part of
    # >>> spl = make_lsq_spline(x, y, w=w2, t=t, k=k)
    # NB:
    #     1. y is assumed to be 2D here. For 1D case (parametric=False),
    #        the call must have been preceded by y = y[:, None] (cf _validate_inputs)
    #     2. We always sum the squares across axis=1:
    #         * For 1D (parametric=False), the last dimension has size one,
    #           so the summation is a no-op.
    #         * For 2D (parametric=True), the summation is actually how the
    #           'residuals' are defined, see Eq. (42) in Dierckx1982
    #           (the reference is in the docstring of `class F`) below.
    _, _, c = _lsq_solve_qr(x, y, t, k, w)
    c = np.ascontiguousarray(c)
    spl = BSpline(t, c, k)
    return _compute_residuals(w2, spl(x), y)


def _compute_residuals(w2, splx, y):
    delta = ((splx - y)**2).sum(axis=1)
    return w2 * delta


def add_knot(x, t, k, residuals):
    """Add a new knot.

    (Approximately) replicate FITPACK's logic:
      1. split the `x` array into knot intervals, ``t(j+k) <= x(i) <= t(j+k+1)``
      2. find the interval with the maximum sum of residuals
      3. insert a new knot into the middle of that interval.

    NB: a new knot is in fact an `x` value at the middle of the interval.
    So *the knots are a subset of `x`*.

    This routine is an analog of
    https://github.com/scipy/scipy/blob/v1.11.4/scipy/interpolate/fitpack/fpcurf.f#L190-L215
    (cf _split function)

    and https://github.com/scipy/scipy/blob/v1.11.4/scipy/interpolate/fitpack/fpknot.f
    """
    new_knot = _dierckx.fpknot(x, t, k, residuals)

    idx_t = np.searchsorted(t, new_knot)
    t_new = np.r_[t[:idx_t], new_knot, t[idx_t:]]
    return t_new


def _validate_inputs(x, y, w, k, s, xb, xe, parametric):
    """Common input validations for generate_knots and make_splrep.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if w is None:
        w = np.ones_like(x, dtype=float)
    else:
        w = np.asarray(w, dtype=float)
        if w.ndim != 1:
            raise ValueError(f"{w.ndim = } not implemented yet.")
        if (w < 0).any():
            raise ValueError("Weights must be non-negative")

    if y.ndim == 0 or y.ndim > 2:
        raise ValueError(f"{y.ndim = } not supported (must be 1 or 2.)")

    parametric = bool(parametric)
    if parametric:
        if y.ndim != 2:
            raise ValueError(f"{y.ndim = } != 2 not supported with {parametric =}.")
    else:
        if y.ndim != 1:
            raise ValueError(f"{y.ndim = } != 1 not supported with {parametric =}.")
        # all _impl functions expect y.ndim = 2
        y = y[:, None]

    if w.shape[0] != x.shape[0]:
        raise ValueError(f"Weights is incompatible: {w.shape =} != {x.shape}.")

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Data is incompatible: {x.shape = } and {y.shape = }.")
    if x.ndim != 1 or (x[1:] < x[:-1]).any():
        raise ValueError("Expect `x` to be an ordered 1D sequence.")

    k = operator.index(k)

    if s < 0:
        raise ValueError(f"`s` must be non-negative. Got {s = }")

    if xb is None:
        xb = min(x)
    if xe is None:
        xe = max(x)

    return x, y, w, k, s, xb, xe


def generate_knots(x, y, *, w=None, xb=None, xe=None, k=3, s=0, nest=None):
    """Replicate FITPACK's constructing the knot vector.

    Parameters
    ----------
    x, y : array_like
        The data points defining the curve ``y = f(x)``.
    w : array_like, optional
        Weights.
    xb : float, optional
        The boundary of the approximation interval. If None (default),
        is set to ``x[0]``.
    xe : float, optional
        The boundary of the approximation interval. If None (default),
        is set to ``x[-1]``.
    k : int, optional
        The spline degree. Default is cubic, ``k = 3``.
    s : float, optional
        The smoothing factor. Default is ``s = 0``.
    nest : int, optional
        Stop when at least this many knots are placed.

    Yields
    ------
    t : ndarray
        Knot vectors with an increasing number of knots.
        The generator is finite: it stops when the smoothing critetion is
        satisfied, or when then number of knots exceeds the maximum value:
        the user-provided `nest` or `x.size + k + 1` --- which is the knot vector
        for the interpolating spline.

    Examples
    --------
    Generate some noisy data and fit a sequence of LSQ splines:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.interpolate import make_lsq_spline, generate_knots
    >>> rng = np.random.default_rng(12345)
    >>> x = np.linspace(-3, 3, 50)
    >>> y = np.exp(-x**2) + 0.1 * rng.standard_normal(size=50)

    >>> knots = list(generate_knots(x, y, s=1e-10))
    >>> for t in knots[::3]:
    ...     spl = make_lsq_spline(x, y, t)
    ...     xs = xs = np.linspace(-3, 3, 201)
    ...     plt.plot(xs, spl(xs), '-', label=f'n = {len(t)}', lw=3, alpha=0.7)
    >>> plt.plot(x, y, 'o', label='data')
    >>> plt.plot(xs, np.exp(-xs**2), '--')
    >>> plt.legend()

    Note that increasing the number of knots make the result follow the data
    more and more closely.

    Also note that a step of the generator may add multiple knots:

    >>> [len(t) for t in knots]
    [8, 9, 10, 12, 16, 24, 40, 48, 52, 54]

    Notes
    -----
    The routine generates successive knots vectors of increasing length, starting
    from ``2*(k+1)`` to ``len(x) + k + 1``, trying to make knots more dense
    in the regions where the deviation of the LSQ spline from data is large.

    When the maximum number of knots, ``len(x) + k + 1`` is reached
    (this happens when ``s`` is small and ``nest`` is large), the generator
    stops, and the last output is the knots for the interpolation with the
    not-a-knot boundary condition.

    Knots are located at data sites, unless ``k`` is even and the number of knots
    is ``len(x) + k + 1``. In that case, the last output of the generator
    has internal knots at Greville sites, ``(x[1:] + x[:-1]) / 2``.

    .. versionadded:: 1.15.0

    """
    if s == 0:
        if nest is not None or w is not None:
            raise ValueError("s == 0 is interpolation only")
        t = _not_a_knot(x, k)
        yield t
        return

    x, y, w, k, s, xb, xe = _validate_inputs(
        x, y, w, k, s, xb, xe, parametric=np.ndim(y) == 2
    )

    yield from _generate_knots_impl(x, y, w=w, xb=xb, xe=xe, k=k, s=s, nest=nest)


def _generate_knots_impl(x, y, *, w=None, xb=None, xe=None, k=3, s=0, nest=None):

    acc = s * TOL
    m = x.size    # the number of data points

    if nest is None:
        # the max number of knots. This is set in _fitpack_impl.py line 274
        # and fitpack.pyf line 198
        nest = max(m + k + 1, 2*k + 3)
    else:
        if nest < 2*(k + 1):
            raise ValueError(f"`nest` too small: {nest = } < 2*(k+1) = {2*(k+1)}.")

    nmin = 2*(k + 1)    # the number of knots for an LSQ polynomial approximation
    nmax = m + k + 1  # the number of knots for the spline interpolation

    # start from no internal knots
    t = np.asarray([xb]*(k+1) + [xe]*(k+1), dtype=float)
    n = t.shape[0]
    fp = 0.0
    fpold = 0.0

    # c  main loop for the different sets of knots. m is a safe upper bound
    # c  for the number of trials.
    for _ in range(m):
        yield t

        # construct the LSQ spline with this set of knots
        fpold = fp
        residuals = _get_residuals(x, y, t, k, w=w)
        fp = residuals.sum()
        fpms = fp - s

        # c  test whether the approximation sinf(x) is an acceptable solution.
        # c  if f(p=inf) < s accept the choice of knots.
        if (abs(fpms) < acc) or (fpms < 0):
            return

        # ### c  increase the number of knots. ###

        # c  determine the number of knots nplus we are going to add.
        if n == nmin:
            # the first iteration
            nplus = 1
        else:
            delta = fpold - fp
            npl1 = int(nplus * fpms / delta) if delta > acc else nplus*2
            nplus = min(nplus*2, max(npl1, nplus//2, 1))

        # actually add knots
        for j in range(nplus):
            t = add_knot(x, t, k, residuals)

            # check if we have enough knots already

            n = t.shape[0]
            # c  if n = nmax, sinf(x) is an interpolating spline.
            # c  if n=nmax we locate the knots as for interpolation.
            if n >= nmax:
                t = _not_a_knot(x, k)
                yield t
                return

            # c  if n=nest we cannot increase the number of knots because of
            # c  the storage capacity limitation.
            if n >= nest:
                yield t
                return

            # recompute if needed
            if j < nplus - 1:
                residuals = _get_residuals(x, y, t, k, w=w)

    # this should never be reached
    return


#   cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
#   c  part 2: determination of the smoothing spline sp(x).                c
#   c  ***************************************************                 c
#   c  we have determined the number of knots and their position.          c
#   c  we now compute the b-spline coefficients of the smoothing spline    c
#   c  sp(x). the observation matrix a is extended by the rows of matrix   c
#   c  b expressing that the kth derivative discontinuities of sp(x) at    c
#   c  the interior knots t(k+2),...t(n-k-1) must be zero. the corres-     c
#   c  ponding weights of these additional rows are set to 1/p.            c
#   c  iteratively we then have to determine the value of p such that      c
#   c  f(p)=sum((w(i)*(y(i)-sp(x(i))))**2) be = s. we already know that    c
#   c  the least-squares kth degree polynomial corresponds to p=0, and     c
#   c  that the least-squares spline corresponds to p=infinity. the        c
#   c  iteration process which is proposed here, makes use of rational     c
#   c  interpolation. since f(p) is a convex and strictly decreasing       c
#   c  function of p, it can be approximated by a rational function        c
#   c  r(p) = (u*p+v)/(p+w). three values of p(p1,p2,p3) with correspond-  c
#   c  ing values of f(p) (f1=f(p1)-s,f2=f(p2)-s,f3=f(p3)-s) are used      c
#   c  to calculate the new value of p such that r(p)=s. convergence is    c
#   c  guaranteed by taking f1>0 and f3<0.                                 c
#   cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc


def prodd(t, i, j, k):
    res = 1.0
    for s in range(k+2):
        if i + s != j:
            res *= (t[j] - t[i+s])
    return res


def disc(t, k):
    """Discontinuity matrix: jumps of k-th derivatives of b-splines at internal knots.

    See Eqs. (9)-(10) of Ref. [1], or, equivalently, Eq. (3.43) of Ref. [2].

    This routine assumes internal knots are all simple (have multiplicity =1).

    Parameters
    ----------
    t : ndarray, 1D, shape(n,)
        Knots.
    k : int
        The spline degree

    Returns
    -------
    disc : ndarray, shape(n-2*k-1, k+2)
        The jumps of the k-th derivatives of b-splines at internal knots,
        ``t[k+1], ...., t[n-k-1]``.
    offset : ndarray, shape(2-2*k-1,)
        Offsets
    nc : int

    Notes
    -----

    The normalization here follows FITPACK:
    (https://github.com/scipy/scipy/blob/maintenance/1.11.x/scipy/interpolate/fitpack/fpdisc.f#L36)

    The k-th derivative jumps are multiplied by a factor::

        (delta / nrint)**k / k!

    where ``delta`` is the length of the interval spanned by internal knots, and
    ``nrint`` is one less the number of internal knots (i.e., the number of
    subintervals between them).

    References
    ----------
    .. [1] Paul Dierckx, Algorithms for smoothing data with periodic and parametric
           splines, Computer Graphics and Image Processing, vol. 20, p. 171 (1982).
           :doi:`10.1016/0146-664X(82)90043-0`

    .. [2] Tom Lyche and Knut Morken, Spline methods,
        http://www.uio.no/studier/emner/matnat/ifi/INF-MAT5340/v05/undervisningsmateriale/

    """
    n = t.shape[0]

    # the length of the base interval spanned by internal knots & the number
    # of subintervas between these internal knots
    delta = t[n - k - 1] - t[k]
    nrint = n - 2*k - 1

    matr = np.empty((nrint - 1, k + 2), dtype=float)
    for jj in range(nrint - 1):
        j = jj + k + 1
        for ii in range(k + 2):
            i = jj + ii
            matr[jj, ii] = (t[i + k + 1] - t[i]) / prodd(t, i, j, k)
        # NB: equivalent to
        # row = [(t[i + k + 1] - t[i]) / prodd(t, i, j, k) for i in range(j-k-1, j+1)]
        # assert (matr[j-k-1, :] == row).all()

    # follow FITPACK
    matr *= (delta/ nrint)**k

    # make it packed
    offset = np.array([i for i in range(nrint-1)], dtype=np.int64)
    nc = n - k - 1
    return matr, offset, nc


class F:
    """ The r.h.s. of ``f(p) = s``.

    Given scalar `p`, we solve the system of equations in the LSQ sense:

        | A     |  @ | c | = | y |
        | B / p |    | 0 |   | 0 |

    where `A` is the matrix of b-splines and `b` is the discontinuity matrix
    (the jumps of the k-th derivatives of b-spline basis elements at knots).

    Since we do that repeatedly while minimizing over `p`, we QR-factorize
    `A` only once and update the QR factorization only of the `B` rows of the
    augmented matrix |A, B/p|.

    The system of equations is Eq. (15) Ref. [1]_, the strategy and implementation
    follows that of FITPACK, see specific links below.

    References
    ----------
    [1] P. Dierckx, Algorithms for Smoothing Data with Periodic and Parametric Splines,
        COMPUTER GRAPHICS AND IMAGE PROCESSING vol. 20, pp 171-184 (1982.)
        https://doi.org/10.1016/0146-664X(82)90043-0

    """
    def __init__(self, x, y, t, k, s, w=None, *, R=None, Y=None):
        self.x = x
        self.y = y
        self.t = t
        self.k = k
        w = np.ones_like(x, dtype=float) if w is None else w
        if w.ndim != 1:
            raise ValueError(f"{w.ndim = } != 1.")
        self.w = w
        self.s = s

        if y.ndim != 2:
            raise ValueError(f"F: expected y.ndim == 2, got {y.ndim = } instead.")

        # ### precompute what we can ###

        # https://github.com/scipy/scipy/blob/maintenance/1.11.x/scipy/interpolate/fitpack/fpcurf.f#L250
        # c  evaluate the discontinuity jump of the kth derivative of the
        # c  b-splines at the knots t(l),l=k+2,...n-k-1 and store in b.
        b, b_offset, b_nc = disc(t, k)

        # the QR factorization of the data matrix, if not provided
        # NB: otherwise, must be consistent with x,y & s, but this is not checked
        if R is None and Y is None:
            R, Y, _ = _lsq_solve_qr(x, y, t, k, w)

        # prepare to combine R and the discontinuity matrix (AB); also r.h.s. (YY)
        # https://github.com/scipy/scipy/blob/maintenance/1.11.x/scipy/interpolate/fitpack/fpcurf.f#L269
        # c  the rows of matrix b with weight 1/p are rotated into the
        # c  triangularised observation matrix a which is stored in g.
        nc = t.shape[0] - k - 1
        nz = k + 1
        if R.shape[1] != nz:
            raise ValueError(f"Internal error: {R.shape[1] =} != {k+1 =}.")

        # r.h.s. of the augmented system
        z = np.zeros((b.shape[0], Y.shape[1]), dtype=float)
        self.YY = np.r_[Y[:nc], z]

        # l.h.s. of the augmented system
        AA = np.zeros((nc + b.shape[0], self.k+2), dtype=float)
        AA[:nc, :nz] = R[:nc, :]
        # AA[nc:, :] = b.a / p  # done in __call__(self, p)
        self.AA  = AA
        self.offset = np.r_[np.arange(nc, dtype=np.int64), b_offset]

        self.nc = nc
        self.b = b

    def __call__(self, p):
        # https://github.com/scipy/scipy/blob/maintenance/1.11.x/scipy/interpolate/fitpack/fpcurf.f#L279
        # c  the row of matrix b is rotated into triangle by givens transformation

        # copy the precomputed matrices over for in-place work
        # R = PackedMatrix(self.AB.a.copy(), self.AB.offset.copy(), nc)
        AB = self.AA.copy()
        offset = self.offset.copy()
        nc = self.nc

        AB[nc:, :] = self.b / p
        QY = self.YY.copy()

        # heavy lifting happens here, in-place
        _dierckx.qr_reduce(AB, offset, nc, QY, startrow=nc)

        # solve for the coefficients
        c = _dierckx.fpback(AB, nc, QY)

        spl = BSpline(self.t, c, self.k)
        residuals = _compute_residuals(self.w**2, spl(self.x), self.y)
        fp = residuals.sum()

        self.spl = spl   # store it

        return fp - self.s


def fprati(p1, f1, p2, f2, p3, f3):
    """The root of r(p) = (u*p + v) / (p + w) given three points and values,
    (p1, f2), (p2, f2) and (p3, f3).

    The FITPACK analog adjusts the bounds, and we do not
    https://github.com/scipy/scipy/blob/maintenance/1.11.x/scipy/interpolate/fitpack/fprati.f

    NB: FITPACK uses p < 0 to encode p=infinity. We just use the infinity itself.
    Since the bracket is ``p1 <= p2 <= p3``, ``p3`` can be infinite (in fact,
    this is what the minimizer starts with, ``p3=inf``).
    """
    h1 = f1 * (f2 - f3)
    h2 = f2 * (f3 - f1)
    h3 = f3 * (f1 - f2)
    if p3 == np.inf:
        return -(p2*h1 + p1*h2) / h3
    return -(p1*p2*h3 + p2*p3*h1 + p1*p3*h2) / (p1*h1 + p2*h2 + p3*h3)


class Bunch:
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)


_iermesg = {
2: """error. a theoretically impossible result was found during
the iteration process for finding a smoothing spline with
fp = s. probably causes : s too small.
there is an approximation returned but the corresponding
weighted sum of squared residuals does not satisfy the
condition abs(fp-s)/s < tol.
""",
3: """error. the maximal number of iterations maxit (set to 20
by the program) allowed for finding a smoothing spline
with fp=s has been reached. probably causes : s too small
there is an approximation returned but the corresponding
weighted sum of squared residuals does not satisfy the
condition abs(fp-s)/s < tol.
"""
}


def root_rati(f, p0, bracket, acc):
    """Solve `f(p) = 0` using a rational function approximation.

    In a nutshell, since the function f(p) is known to be monotonically decreasing, we
       - maintain the bracket (p1, f1), (p2, f2) and (p3, f3)
       - at each iteration step, approximate f(p) by a rational function
         r(p) = (u*p + v) / (p + w)
         and make a step to p_new to the root of f(p): r(p_new) = 0.
         The coefficients u, v and w are found from the bracket values p1..3 and f1...3

    The algorithm and implementation follows
    https://github.com/scipy/scipy/blob/maintenance/1.11.x/scipy/interpolate/fitpack/fpcurf.f#L229
    and
    https://github.com/scipy/scipy/blob/maintenance/1.11.x/scipy/interpolate/fitpack/fppara.f#L290

    Note that the latter is for parametric splines and the former is for 1D spline
    functions. The minimization is indentical though [modulo a summation over the
    dimensions in the computation of f(p)], so we reuse the minimizer for both
    d=1 and d>1.
    """
    # Magic values from
    # https://github.com/scipy/scipy/blob/maintenance/1.11.x/scipy/interpolate/fitpack/fpcurf.f#L27
    con1 = 0.1
    con9 = 0.9
    con4 = 0.04

    # bracketing flags (follow FITPACK)
    # https://github.com/scipy/scipy/blob/maintenance/1.11.x/scipy/interpolate/fitpack/fppara.f#L365
    ich1, ich3 = 0, 0

    (p1, f1), (p3, f3)  = bracket
    p = p0

    for it in range(MAXIT):
        p2, f2 = p, f(p)

        # c  test whether the approximation sp(x) is an acceptable solution.
        if abs(f2) < acc:
            ier, converged = 0, True
            break

        # c  carry out one more step of the iteration process.
        if ich3 == 0:
            if f2 - f3 <= acc:
                # c  our initial choice of p is too large.
                p3 = p2
                f3 = f2
                p = p*con4
                if p <= p1:
                     p = p1*con9 + p2*con1
                continue
            else:
                if f2 < 0:
                    ich3 = 1

        if ich1 == 0:
            if f1 - f2 <= acc:
                # c  our initial choice of p is too small
                p1 = p2
                f1 = f2
                p = p/con4
                if p3 != np.inf and p <= p3:
                     p = p2*con1 + p3*con9
                continue
            else:
                if f2 > 0:
                    ich1 = 1

        # c  test whether the iteration process proceeds as theoretically expected.
        # [f(p) should be monotonically decreasing]
        if f1 <= f2 or f2 <= f3:
            ier, converged = 2, False
            break

        # actually make the iteration step
        p = fprati(p1, f1, p2, f2, p3, f3)

        # c  adjust the value of p1,f1,p3 and f3 such that f1 > 0 and f3 < 0.
        if f2 < 0:
            p3, f3 = p2, f2
        else:
            p1, f1 = p2, f2

    else:
        # not converged in MAXIT iterations
        ier, converged = 3, False

    if ier != 0:
        warnings.warn(RuntimeWarning(_iermesg[ier]), stacklevel=2)

    return Bunch(converged=converged, root=p, iterations=it, ier=ier)


def _make_splrep_impl(x, y, *, w=None, xb=None, xe=None, k=3, s=0, t=None, nest=None):
    """Shared infra for make_splrep and make_splprep.
    """
    acc = s * TOL
    m = x.size    # the number of data points

    if nest is None:
        # the max number of knots. This is set in _fitpack_impl.py line 274
        # and fitpack.pyf line 198
        nest = max(m + k + 1, 2*k + 3)
    else:
        if nest < 2*(k + 1):
            raise ValueError(f"`nest` too small: {nest = } < 2*(k+1) = {2*(k+1)}.")    
        if t is not None:
            raise ValueError("Either supply `t` or `nest`.")

    if t is None:
        gen = _generate_knots_impl(x, y, w=w, k=k, s=s, xb=xb, xe=xe, nest=nest)
        t = list(gen)[-1]
    else:
        fpcheck(x, t, k)

    if t.shape[0] == 2 * (k + 1):
        # nothing to optimize
        _, _, c = _lsq_solve_qr(x, y, t, k, w)
        return BSpline(t, c, k)

    ### solve ###

    # c  initial value for p.
    # https://github.com/scipy/scipy/blob/maintenance/1.11.x/scipy/interpolate/fitpack/fpcurf.f#L253
    R, Y, _ = _lsq_solve_qr(x, y, t, k, w)
    nc = t.shape[0] -k -1
    p = nc / R[:, 0].sum()

    # ### bespoke solver ####
    # initial conditions
    # f(p=inf) : LSQ spline with knots t   (XXX: reuse R, c)
    residuals = _get_residuals(x, y, t, k, w=w)
    fp = residuals.sum()
    fpinf = fp - s

    # f(p=0): LSQ spline without internal knots
    residuals = _get_residuals(x, y, np.array([xb]*(k+1) + [xe]*(k+1)), k, w)
    fp0 = residuals.sum()
    fp0 = fp0 - s

    # solve
    bracket = (0, fp0), (np.inf, fpinf)
    f = F(x, y, t, k=k, s=s, w=w, R=R, Y=Y)
    _ = root_rati(f, p, bracket, acc)

    # solve ALTERNATIVE: is roughly equivalent, gives slightly different results
    # starting from scratch, that would have probably been tolerable;
    # backwards compatibility dictates that we replicate the FITPACK minimizer though.
 #   f = F(x, y, t, k=k, s=s, w=w, R=R, Y=Y)
 #   from scipy.optimize import root_scalar
 #   res_ = root_scalar(f, x0=p, rtol=acc)
 #   assert res_.converged

    # f.spl is the spline corresponding to the found `p` value
    return f.spl


def make_splrep(x, y, *, w=None, xb=None, xe=None, k=3, s=0, t=None, nest=None):
    r"""Find the B-spline representation of a 1D function.

    Given the set of data points ``(x[i], y[i])``, determine a smooth spline
    approximation of degree ``k`` on the interval ``xb <= x <= xe``.

    Parameters
    ----------
    x, y : array_like, shape (m,)
        The data points defining a curve ``y = f(x)``.
    w : array_like, shape (m,), optional
        Strictly positive 1D array of weights, of the same length as `x` and `y`.
        The weights are used in computing the weighted least-squares spline
        fit. If the errors in the y values have standard-deviation given by the
        vector ``d``, then `w` should be ``1/d``.
        Default is ``np.ones(m)``.
    xb, xe : float, optional
        The interval to fit.  If None, these default to ``x[0]`` and ``x[-1]``,
        respectively.
    k : int, optional
        The degree of the spline fit. It is recommended to use cubic splines,
        ``k=3``, which is the default. Even values of `k` should be avoided,
        especially with small `s` values.
    s : float, optional
        The smoothing condition. The amount of smoothness is determined by
        satisfying the conditions::

            sum((w * (g(x)  - y))**2 ) <= s

        where ``g(x)`` is the smoothed fit to ``(x, y)``. The user can use `s`
        to control the tradeoff between closeness to data and smoothness of fit.
        Larger `s` means more smoothing while smaller values of `s` indicate less
        smoothing.
        Recommended values of `s` depend on the weights, `w`. If the weights
        represent the inverse of the standard deviation of `y`, then a good `s`
        value should be found in the range ``(m-sqrt(2*m), m+sqrt(2*m))`` where
        ``m`` is the number of datapoints in `x`, `y`, and `w`.
        Default is ``s = 0.0``, i.e. interpolation.
    t : array_like, optional
        The spline knots. If None (default), the knots will be constructed
        automatically.
        There must be at least ``2*k + 2`` and at most ``m + k + 1`` knots.
    nest : int, optional
        The target length of the knot vector. Should be between ``2*(k + 1)``
        (the minimum number of knots for a degree-``k`` spline), and
        ``m + k + 1`` (the number of knots of the interpolating spline).
        The actual number of knots returned by this routine may be slightly
        larger than `nest`.
        Default is None (no limit, add up to ``m + k + 1`` knots).

    Returns
    -------
    spl : a `BSpline` instance
        For `s=0`,  ``spl(x) == y``.
        For non-zero values of `s` the `spl` represents the smoothed approximation
        to `(x, y)`, generally with fewer knots.

    See Also
    --------
    generate_knots : is used under the hood for generating the knots
    make_splprep : the analog of this routine for parametric curves
    make_interp_spline : construct an interpolating spline (``s = 0``)
    make_lsq_spline : construct the least-squares spline given the knot vector
    splrep : a FITPACK analog of this routine

    References
    ----------
    .. [1] P. Dierckx, "Algorithms for smoothing data with periodic and
        parametric splines, Computer Graphics and Image Processing",
        20 (1982) 171-184.
    .. [2] P. Dierckx, "Curve and surface fitting with splines", Monographs on
        Numerical Analysis, Oxford University Press, 1993.

    Notes
    -----
    This routine constructs the smoothing spline function, :math:`g(x)`, to
    minimize the sum of jumps, :math:`D_j`, of the ``k``-th derivative at the
    internal knots (:math:`x_b < t_i < x_e`), where

    .. math::

        D_i = g^{(k)}(t_i + 0) - g^{(k)}(t_i - 0)

    Specifically, the routine constructs the spline function :math:`g(x)` which
    minimizes

    .. math::

            \sum_i | D_i |^2 \to \mathrm{min}

    provided that

    .. math::

           \sum_{j=1}^m (w_j \times (g(x_j) - y_j))^2 \leqslant s ,

    where :math:`s > 0` is the input parameter.

    In other words, we balance maximizing the smoothness (measured as the jumps
    of the derivative, the first criterion), and the deviation of :math:`g(x_j)`
    from the data :math:`y_j` (the second criterion).

    Note that the summation in the second criterion is over all data points,
    and in the first criterion it is over the internal spline knots (i.e.
    those with ``xb < t[i] < xe``). The spline knots are in general a subset
    of data, see `generate_knots` for details.

    Also note the difference of this routine to `make_lsq_spline`: the latter
    routine does not consider smoothness and simply solves a least-squares
    problem

    .. math::

        \sum w_j \times (g(x_j) - y_j)^2 \to \mathrm{min}

    for a spline function :math:`g(x)` with a _fixed_ knot vector ``t``.

    .. versionadded:: 1.15.0
    """
    if s == 0:
        if t is not None or w is not None or nest is not None:
            raise ValueError("s==0 is for interpolation only")
        return make_interp_spline(x, y, k=k)

    x, y, w, k, s, xb, xe = _validate_inputs(x, y, w, k, s, xb, xe, parametric=False)

    spl = _make_splrep_impl(x, y, w=w, xb=xb, xe=xe, k=k, s=s, t=t, nest=nest)

    # postprocess: squeeze out the last dimension: was added to simplify the internals.
    spl.c = spl.c[:, 0]
    return spl


def make_splprep(x, *, w=None, u=None, ub=None, ue=None, k=3, s=0, t=None, nest=None):
    r"""
    Find a smoothed B-spline representation of a parametric N-D curve.

    Given a list of N 1D arrays, `x`, which represent a curve in
    N-dimensional space parametrized by `u`, find a smooth approximating
    spline curve ``g(u)``.

    Parameters
    ----------
    x : array_like, shape (m, ndim)
        Sampled data points representing the curve in ``ndim`` dimensions.
        The typical use is a list of 1D arrays, each of length ``m``.
    w : array_like, shape(m,), optional
        Strictly positive 1D array of weights.
        The weights are used in computing the weighted least-squares spline
        fit. If the errors in the `x` values have standard deviation given by
        the vector d, then `w` should be 1/d. Default is ``np.ones(m)``.
    u : array_like, optional
        An array of parameter values for the curve in the parametric form.
        If not given, these values are calculated automatically, according to::

            v[0] = 0
            v[i] = v[i-1] + distance(x[i], x[i-1])
            u[i] = v[i] / v[-1]

    ub, ue : float, optional
        The end-points of the parameters interval. Default to ``u[0]`` and ``u[-1]``.
    k : int, optional
        Degree of the spline. Cubic splines, ``k=3``, are recommended.
        Even values of `k` should be avoided especially with a small ``s`` value.
        Default is ``k=3``
    s : float, optional
        A smoothing condition.  The amount of smoothness is determined by
        satisfying the conditions::

            sum((w * (g(u) - x))**2) <= s,

        where ``g(u)`` is the smoothed approximation to ``x``.  The user can
        use `s` to control the trade-off between closeness and smoothness
        of fit.  Larger ``s`` means more smoothing while smaller values of ``s``
        indicate less smoothing.
        Recommended values of ``s`` depend on the weights, ``w``.  If the weights
        represent the inverse of the standard deviation of ``x``, then a good
        ``s`` value should be found in the range ``(m - sqrt(2*m), m + sqrt(2*m))``,
        where ``m`` is the number of data points in ``x`` and ``w``.
    t : array_like, optional
        The spline knots. If None (default), the knots will be constructed
        automatically.
        There must be at least ``2*k + 2`` and at most ``m + k + 1`` knots.
    nest : int, optional
        The target length of the knot vector. Should be between ``2*(k + 1)``
        (the minimum number of knots for a degree-``k`` spline), and
        ``m + k + 1`` (the number of knots of the interpolating spline).
        The actual number of knots returned by this routine may be slightly
        larger than `nest`.
        Default is None (no limit, add up to ``m + k + 1`` knots).

    Returns
    -------
    spl : a `BSpline` instance
        For `s=0`,  ``spl(u) == x``.
        For non-zero values of ``s``, `spl` represents the smoothed approximation
        to ``x``, generally with fewer knots.
    u : ndarray
        The values of the parameters

    See Also
    --------
    generate_knots : is used under the hood for generating the knots
    make_splrep : the analog of this routine 1D functions
    make_interp_spline : construct an interpolating spline (``s = 0``)
    make_lsq_spline : construct the least-squares spline given the knot vector
    splprep : a FITPACK analog of this routine

    Notes
    -----
    Given a set of :math:`m` data points in :math:`D` dimensions, :math:`\vec{x}_j`,
    with :math:`j=1, ..., m` and :math:`\vec{x}_j = (x_{j; 1}, ..., x_{j; D})`,
    this routine constructs the parametric spline curve :math:`g_a(u)` with
    :math:`a=1, ..., D`, to minimize the sum of jumps, :math:`D_{i; a}`, of the
    ``k``-th derivative at the internal knots (:math:`u_b < t_i < u_e`), where

    .. math::

        D_{i; a} = g_a^{(k)}(t_i + 0) - g_a^{(k)}(t_i - 0)

    Specifically, the routine constructs the spline function :math:`g(u)` which
    minimizes

    .. math::

            \sum_i \sum_{a=1}^D | D_{i; a} |^2 \to \mathrm{min}

    provided that

    .. math::

        \sum_{j=1}^m \sum_{a=1}^D (w_j \times (g_a(u_j) - x_{j; a}))^2 \leqslant s

    where :math:`u_j` is the value of the parameter corresponding to the data point
    :math:`(x_{j; 1}, ..., x_{j; D})`, and :math:`s > 0` is the input parameter.

    In other words, we balance maximizing the smoothness (measured as the jumps
    of the derivative, the first criterion), and the deviation of :math:`g(u_j)`
    from the data :math:`x_j` (the second criterion).

    Note that the summation in the second criterion is over all data points,
    and in the first criterion it is over the internal spline knots (i.e.
    those with ``ub < t[i] < ue``). The spline knots are in general a subset
    of data, see `generate_knots` for details.

    .. versionadded:: 1.15.0

    References
    ----------
    .. [1] P. Dierckx, "Algorithms for smoothing data with periodic and
        parametric splines, Computer Graphics and Image Processing",
        20 (1982) 171-184.
    .. [2] P. Dierckx, "Curve and surface fitting with splines", Monographs on
        Numerical Analysis, Oxford University Press, 1993.
    """
    x = np.stack(x, axis=1)

    # construct the default parametrization of the curve
    if u is None:
        dp = (x[1:, :] - x[:-1, :])**2
        u = np.sqrt((dp).sum(axis=1)).cumsum()
        u = np.r_[0, u / u[-1]]

    if s == 0:
        if t is not None or w is not None or nest is not None:
            raise ValueError("s==0 is for interpolation only")
        return make_interp_spline(u, x.T, k=k, axis=1), u

    u, x, w, k, s, ub, ue = _validate_inputs(u, x, w, k, s, ub, ue, parametric=True)

    spl = _make_splrep_impl(u, x, w=w, xb=ub, xe=ue, k=k, s=s, t=t, nest=nest)

    # posprocess: `axis=1` so that spl(u).shape == np.shape(x)
    # when `x` is a list of 1D arrays (cf original splPrep)
    cc = spl.c.T
    spl1 = BSpline(spl.t, cc, spl.k, axis=1)

    return spl1, u

<|endoftext|>import operator
from math import prod

import numpy as np
from scipy._lib._util import normalize_axis_index
from scipy.linalg import (get_lapack_funcs, LinAlgError,
                          cholesky_banded, cho_solve_banded,
                          solve, solve_banded)
from scipy.optimize import minimize_scalar
from . import _dierckx
from . import _fitpack_impl
from scipy.sparse import csr_array
from scipy.special import poch
from itertools import combinations


__all__ = ["BSpline", "make_interp_spline", "make_lsq_spline",
           "make_smoothing_spline"]


def _get_dtype(dtype):
    """Return np.complex128 for complex dtypes, np.float64 otherwise."""
    if np.issubdtype(dtype, np.complexfloating):
        return np.complex128
    else:
        return np.float64


def _as_float_array(x, check_finite=False):
    """Convert the input into a C contiguous float array.

    NB: Upcasts half- and single-precision floats to double precision.
    """
    x = np.ascontiguousarray(x)
    dtyp = _get_dtype(x.dtype)
    x = x.astype(dtyp, copy=False)
    if check_finite and not np.isfinite(x).all():
        raise ValueError("Array must not contain infs or nans.")
    return x


def _dual_poly(j, k, t, y):
    """
    Dual polynomial of the B-spline B_{j,k,t} -
    polynomial which is associated with B_{j,k,t}:
    $p_{j,k}(y) = (y - t_{j+1})(y - t_{j+2})...(y - t_{j+k})$
    """
    if k == 0:
        return 1
    return np.prod([(y - t[j + i]) for i in range(1, k + 1)])


def _diff_dual_poly(j, k, y, d, t):
    """
    d-th derivative of the dual polynomial $p_{j,k}(y)$
    """
    if d == 0:
        return _dual_poly(j, k, t, y)
    if d == k:
        return poch(1, k)
    comb = list(combinations(range(j + 1, j + k + 1), d))
    res = 0
    for i in range(len(comb) * len(comb[0])):
        res += np.prod([(y - t[j + p]) for p in range(1, k + 1)
                        if (j + p) not in comb[i//d]])
    return res


class BSpline:
    r"""Univariate spline in the B-spline basis.

    .. math::

        S(x) = \sum_{j=0}^{n-1} c_j  B_{j, k; t}(x)

    where :math:`B_{j, k; t}` are B-spline basis functions of degree `k`
    and knots `t`.

    Parameters
    ----------
    t : ndarray, shape (n+k+1,)
        knots
    c : ndarray, shape (>=n, ...)
        spline coefficients
    k : int
        B-spline degree
    extrapolate : bool or 'periodic', optional
        whether to extrapolate beyond the base interval, ``t[k] .. t[n]``,
        or to return nans.
        If True, extrapolates the first and last polynomial pieces of b-spline
        functions active on the base interval.
        If 'periodic', periodic extrapolation is used.
        Default is True.
    axis : int, optional
        Interpolation axis. Default is zero.

    Attributes
    ----------
    t : ndarray
        knot vector
    c : ndarray
        spline coefficients
    k : int
        spline degree
    extrapolate : bool
        If True, extrapolates the first and last polynomial pieces of b-spline
        functions active on the base interval.
    axis : int
        Interpolation axis.
    tck : tuple
        A read-only equivalent of ``(self.t, self.c, self.k)``

    Methods
    -------
    __call__
    basis_element
    derivative
    antiderivative
    integrate
    insert_knot
    construct_fast
    design_matrix
    from_power_basis

    Notes
    -----
    B-spline basis elements are defined via

    .. math::

        B_{i, 0}(x) = 1, \textrm{if $t_i \le x < t_{i+1}$, otherwise $0$,}

        B_{i, k}(x) = \frac{x - t_i}{t_{i+k} - t_i} B_{i, k-1}(x)
                 + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B_{i+1, k-1}(x)

    **Implementation details**

    - At least ``k+1`` coefficients are required for a spline of degree `k`,
      so that ``n >= k+1``. Additional coefficients, ``c[j]`` with
      ``j > n``, are ignored.

    - B-spline basis elements of degree `k` form a partition of unity on the
      *base interval*, ``t[k] <= x <= t[n]``.


    Examples
    --------

    Translating the recursive definition of B-splines into Python code, we have:

    >>> def B(x, k, i, t):
    ...    if k == 0:
    ...       return 1.0 if t[i] <= x < t[i+1] else 0.0
    ...    if t[i+k] == t[i]:
    ...       c1 = 0.0
    ...    else:
    ...       c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
    ...    if t[i+k+1] == t[i+1]:
    ...       c2 = 0.0
    ...    else:
    ...       c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
    ...    return c1 + c2

    >>> def bspline(x, t, c, k):
    ...    n = len(t) - k - 1
    ...    assert (n >= k+1) and (len(c) >= n)
    ...    return sum(c[i] * B(x, k, i, t) for i in range(n))

    Note that this is an inefficient (if straightforward) way to
    evaluate B-splines --- this spline class does it in an equivalent,
    but much more efficient way.

    Here we construct a quadratic spline function on the base interval
    ``2 <= x <= 4`` and compare with the naive way of evaluating the spline:

    >>> from scipy.interpolate import BSpline
    >>> k = 2
    >>> t = [0, 1, 2, 3, 4, 5, 6]
    >>> c = [-1, 2, 0, -1]
    >>> spl = BSpline(t, c, k)
    >>> spl(2.5)
    array(1.375)
    >>> bspline(2.5, t, c, k)
    1.375

    Note that outside of the base interval results differ. This is because
    `BSpline` extrapolates the first and last polynomial pieces of B-spline
    functions active on the base interval.

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig, ax = plt.subplots()
    >>> xx = np.linspace(1.5, 4.5, 50)
    >>> ax.plot(xx, [bspline(x, t, c ,k) for x in xx], 'r-', lw=3, label='naive')
    >>> ax.plot(xx, spl(xx), 'b-', lw=4, alpha=0.7, label='BSpline')
    >>> ax.grid(True)
    >>> ax.legend(loc='best')
    >>> plt.show()


    References
    ----------
    .. [1] Tom Lyche and Knut Morken, Spline methods,
        http://www.uio.no/studier/emner/matnat/ifi/INF-MAT5340/v05/undervisningsmateriale/
    .. [2] Carl de Boor, A practical guide to splines, Springer, 2001.

    """

    def __init__(self, t, c, k, extrapolate=True, axis=0):
        super().__init__()

        self.k = operator.index(k)
        self.c = np.asarray(c)
        self.t = np.ascontiguousarray(t, dtype=np.float64)

        if extrapolate == 'periodic':
            self.extrapolate = extrapolate
        else:
            self.extrapolate = bool(extrapolate)

        n = self.t.shape[0] - self.k - 1

        axis = normalize_axis_index(axis, self.c.ndim)

        # Note that the normalized axis is stored in the object.
        self.axis = axis
        if axis != 0:
            # roll the interpolation axis to be the first one in self.c
            # More specifically, the target shape for self.c is (n, ...),
            # and axis !=0 means that we have c.shape (..., n, ...)
            #                                               ^
            #                                              axis
            self.c = np.moveaxis(self.c, axis, 0)

        if k < 0:
            raise ValueError("Spline order cannot be negative.")
        if self.t.ndim != 1:
            raise ValueError("Knot vector must be one-dimensional.")
        if n < self.k + 1:
            raise ValueError("Need at least %d knots for degree %d" %
                             (2*k + 2, k))
        if (np.diff(self.t) < 0).any():
            raise ValueError("Knots must be in a non-decreasing order.")
        if len(np.unique(self.t[k:n+1])) < 2:
            raise ValueError("Need at least two internal knots.")
        if not np.isfinite(self.t).all():
            raise ValueError("Knots should not have nans or infs.")
        if self.c.ndim < 1:
            raise ValueError("Coefficients must be at least 1-dimensional.")
        if self.c.shape[0] < n:
            raise ValueError("Knots, coefficients and degree are inconsistent.")

        dt = _get_dtype(self.c.dtype)
        self.c = np.ascontiguousarray(self.c, dtype=dt)

    @classmethod
    def construct_fast(cls, t, c, k, extrapolate=True, axis=0):
        """Construct a spline without making checks.

        Accepts same parameters as the regular constructor. Input arrays
        `t` and `c` must of correct shape and dtype.
        """
        self = object.__new__(cls)
        self.t, self.c, self.k = t, c, k
        self.extrapolate = extrapolate
        self.axis = axis
        return self

    @property
    def tck(self):
        """Equivalent to ``(self.t, self.c, self.k)`` (read-only).
        """
        return self.t, self.c, self.k

    @classmethod
    def basis_element(cls, t, extrapolate=True):
        """Return a B-spline basis element ``B(x | t[0], ..., t[k+1])``.

        Parameters
        ----------
        t : ndarray, shape (k+2,)
            internal knots
        extrapolate : bool or 'periodic', optional
            whether to extrapolate beyond the base interval, ``t[0] .. t[k+1]``,
            or to return nans.
            If 'periodic', periodic extrapolation is used.
            Default is True.

        Returns
        -------
        basis_element : callable
            A callable representing a B-spline basis element for the knot
            vector `t`.

        Notes
        -----
        The degree of the B-spline, `k`, is inferred from the length of `t` as
        ``len(t)-2``. The knot vector is constructed by appending and prepending
        ``k+1`` elements to internal knots `t`.

        Examples
        --------

        Construct a cubic B-spline:

        >>> import numpy as np
        >>> from scipy.interpolate import BSpline
        >>> b = BSpline.basis_element([0, 1, 2, 3, 4])
        >>> k = b.k
        >>> b.t[k:-k]
        array([ 0.,  1.,  2.,  3.,  4.])
        >>> k
        3

        Construct a quadratic B-spline on ``[0, 1, 1, 2]``, and compare
        to its explicit form:

        >>> t = [0, 1, 1, 2]
        >>> b = BSpline.basis_element(t)
        >>> def f(x):
        ...     return np.where(x < 1, x*x, (2. - x)**2)

        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> x = np.linspace(0, 2, 51)
        >>> ax.plot(x, b(x), 'g', lw=3)
        >>> ax.plot(x, f(x), 'r', lw=8, alpha=0.4)
        >>> ax.grid(True)
        >>> plt.show()

        """
        k = len(t) - 2
        t = _as_float_array(t)
        t = np.r_[(t[0]-1,) * k, t, (t[-1]+1,) * k]
        c = np.zeros_like(t)
        c[k] = 1.
        return cls.construct_fast(t, c, k, extrapolate)

    @classmethod
    def design_matrix(cls, x, t, k, extrapolate=False):
        """
        Returns a design matrix as a CSR format sparse array.

        Parameters
        ----------
        x : array_like, shape (n,)
            Points to evaluate the spline at.
        t : array_like, shape (nt,)
            Sorted 1D array of knots.
        k : int
            B-spline degree.
        extrapolate : bool or 'periodic', optional
            Whether to extrapolate based on the first and last intervals
            or raise an error. If 'periodic', periodic extrapolation is used.
            Default is False.

            .. versionadded:: 1.10.0

        Returns
        -------
        design_matrix : `csr_array` object
            Sparse matrix in CSR format where each row contains all the basis
            elements of the input row (first row = basis elements of x[0],
            ..., last row = basis elements x[-1]).

        Examples
        --------
        Construct a design matrix for a B-spline

        >>> from scipy.interpolate import make_interp_spline, BSpline
        >>> import numpy as np
        >>> x = np.linspace(0, np.pi * 2, 4)
        >>> y = np.sin(x)
        >>> k = 3
        >>> bspl = make_interp_spline(x, y, k=k)
        >>> design_matrix = bspl.design_matrix(x, bspl.t, k)
        >>> design_matrix.toarray()
        [[1.        , 0.        , 0.        , 0.        ],
        [0.2962963 , 0.44444444, 0.22222222, 0.03703704],
        [0.03703704, 0.22222222, 0.44444444, 0.2962963 ],
        [0.        , 0.        , 0.        , 1.        ]]

        Construct a design matrix for some vector of knots

        >>> k = 2
        >>> t = [-1, 0, 1, 2, 3, 4, 5, 6]
        >>> x = [1, 2, 3, 4]
        >>> design_matrix = BSpline.design_matrix(x, t, k).toarray()
        >>> design_matrix
        [[0.5, 0.5, 0. , 0. , 0. ],
        [0. , 0.5, 0.5, 0. , 0. ],
        [0. , 0. , 0.5, 0.5, 0. ],
        [0. , 0. , 0. , 0.5, 0.5]]

        This result is equivalent to the one created in the sparse format

        >>> c = np.eye(len(t) - k - 1)
        >>> design_matrix_gh = BSpline(t, c, k)(x)
        >>> np.allclose(design_matrix, design_matrix_gh, atol=1e-14)
        True

        Notes
        -----
        .. versionadded:: 1.8.0

        In each row of the design matrix all the basis elements are evaluated
        at the certain point (first row - x[0], ..., last row - x[-1]).

        `nt` is a length of the vector of knots: as far as there are
        `nt - k - 1` basis elements, `nt` should be not less than `2 * k + 2`
        to have at least `k + 1` basis element.

        Out of bounds `x` raises a ValueError.
        """
        x = _as_float_array(x, True)
        t = _as_float_array(t, True)

        if extrapolate != 'periodic':
            extrapolate = bool(extrapolate)

        if k < 0:
            raise ValueError("Spline order cannot be negative.")
        if t.ndim != 1 or np.any(t[1:] < t[:-1]):
            raise ValueError(f"Expect t to be a 1-D sorted array_like, but "
                             f"got t={t}.")
        # There are `nt - k - 1` basis elements in a BSpline built on the
        # vector of knots with length `nt`, so to have at least `k + 1` basis
        # elements we need to have at least `2 * k + 2` elements in the vector
        # of knots.
        if len(t) < 2 * k + 2:
            raise ValueError(f"Length t is not enough for k={k}.")

        if extrapolate == 'periodic':
            # With periodic extrapolation we map x to the segment
            # [t[k], t[n]].
            n = t.size - k - 1
            x = t[k] + (x - t[k]) % (t[n] - t[k])
            extrapolate = False
        elif not extrapolate and (
            (min(x) < t[k]) or (max(x) > t[t.shape[0] - k - 1])
        ):
            # Checks from `find_interval` function
            raise ValueError(f'Out of bounds w/ x = {x}.')

        # Compute number of non-zeros of final CSR array in order to determine
        # the dtype of indices and indptr of the CSR array.
        n = x.shape[0]
        nnz = n * (k + 1)
        if nnz < np.iinfo(np.int32).max:
            int_dtype = np.int32
        else:
            int_dtype = np.int64

        # Get the non-zero elements of the design matrix and per-row `offsets`:
        # In row `i`, k+1 nonzero elements are consecutive, and start from `offset[i]`
        data, offsets, _ = _dierckx.data_matrix(x, t, k, np.ones_like(x), extrapolate)
        data = data.ravel()

        if offsets.dtype != int_dtype:
            offsets = offsets.astype(int_dtype)

        # Convert from per-row offsets to the CSR indices/indptr format
        indices = np.repeat(offsets, k+1).reshape(-1, k+1)
        indices = indices + np.arange(k+1, dtype=int_dtype)
        indices = indices.ravel()

        indptr = np.arange(0, (n + 1) * (k + 1), k + 1, dtype=int_dtype)

        return csr_array(
            (data, indices, indptr),
            shape=(x.shape[0], t.shape[0] - k - 1)
        )

    def __call__(self, x, nu=0, extrapolate=None):
        """
        Evaluate a spline function.

        Parameters
        ----------
        x : array_like
            points to evaluate the spline at.
        nu : int, optional
            derivative to evaluate (default is 0).
        extrapolate : bool or 'periodic', optional
            whether to extrapolate based on the first and last intervals
            or return nans. If 'periodic', periodic extrapolation is used.
            Default is `self.extrapolate`.

        Returns
        -------
        y : array_like
            Shape is determined by replacing the interpolation axis
            in the coefficient array with the shape of `x`.

        """
        if extrapolate is None:
            extrapolate = self.extrapolate
        x = np.asarray(x)
        x_shape, x_ndim = x.shape, x.ndim
        x = np.ascontiguousarray(x.ravel(), dtype=np.float64)

        # With periodic extrapolation we map x to the segment
        # [self.t[k], self.t[n]].
        if extrapolate == 'periodic':
            n = self.t.size - self.k - 1
            x = self.t[self.k] + (x - self.t[self.k]) % (self.t[n] -
                                                         self.t[self.k])
            extrapolate = False

        out = np.empty((len(x), prod(self.c.shape[1:])), dtype=self.c.dtype)
        self._ensure_c_contiguous()

        # if self.c is complex, so is `out`; cython code in _bspl.pyx expectes
        # floats though, so make a view---this expands the last axis, and
        # the view is C contiguous if the original is.
        # if c.dtype is complex of shape (n,), c.view(float).shape == (2*n,)
        # if c.dtype is complex of shape (n, m), c.view(float).shape == (n, 2*m)

        cc = self.c.view(float)
        if self.c.ndim == 1 and self.c.dtype.kind == 'c':
            cc = cc.reshape(self.c.shape[0], 2)

        _dierckx.evaluate_spline(self.t, cc.reshape(cc.shape[0], -1),
                              self.k, x, nu, extrapolate, out.view(float))

        out = out.reshape(x_shape + self.c.shape[1:])
        if self.axis != 0:
            # transpose to move the calculated values to the interpolation axis
            l = list(range(out.ndim))
            l = l[x_ndim:x_ndim+self.axis] + l[:x_ndim] + l[x_ndim+self.axis:]
            out = out.transpose(l)
        return out

    def _ensure_c_contiguous(self):
        """
        c and t may be modified by the user. The Cython code expects
        that they are C contiguous.

        """
        if not self.t.flags.c_contiguous:
            self.t = self.t.copy()
        if not self.c.flags.c_contiguous:
            self.c = self.c.copy()

    def derivative(self, nu=1):
        """Return a B-spline representing the derivative.

        Parameters
        ----------
        nu : int, optional
            Derivative order.
            Default is 1.

        Returns
        -------
        b : BSpline object
            A new instance representing the derivative.

        See Also
        --------
        splder, splantider

        """
        c = self.c.copy()
        # pad the c array if needed
        ct = len(self.t) - len(c)
        if ct > 0:
            c = np.r_[c, np.zeros((ct,) + c.shape[1:])]
        tck = _fitpack_impl.splder((self.t, c, self.k), nu)
        return self.construct_fast(*tck, extrapolate=self.extrapolate,
                                   axis=self.axis)

    def antiderivative(self, nu=1):
        """Return a B-spline representing the antiderivative.

        Parameters
        ----------
        nu : int, optional
            Antiderivative order. Default is 1.

        Returns
        -------
        b : BSpline object
            A new instance representing the antiderivative.

        Notes
        -----
        If antiderivative is computed and ``self.extrapolate='periodic'``,
        it will be set to False for the returned instance. This is done because
        the antiderivative is no longer periodic and its correct evaluation
        outside of the initially given x interval is difficult.

        See Also
        --------
        splder, splantider

        """
        c = self.c.copy()
        # pad the c array if needed
        ct = len(self.t) - len(c)
        if ct > 0:
            c = np.r_[c, np.zeros((ct,) + c.shape[1:])]
        tck = _fitpack_impl.splantider((self.t, c, self.k), nu)

        if self.extrapolate == 'periodic':
            extrapolate = False
        else:
            extrapolate = self.extrapolate

        return self.construct_fast(*tck, extrapolate=extrapolate,
                                   axis=self.axis)

    def integrate(self, a, b, extrapolate=None):
        """Compute a definite integral of the spline.

        Parameters
        ----------
        a : float
            Lower limit of integration.
        b : float
            Upper limit of integration.
        extrapolate : bool or 'periodic', optional
            whether to extrapolate beyond the base interval,
            ``t[k] .. t[-k-1]``, or take the spline to be zero outside of the
            base interval. If 'periodic', periodic extrapolation is used.
            If None (default), use `self.extrapolate`.

        Returns
        -------
        I : array_like
            Definite integral of the spline over the interval ``[a, b]``.

        Examples
        --------
        Construct the linear spline ``x if x < 1 else 2 - x`` on the base
        interval :math:`[0, 2]`, and integrate it

        >>> from scipy.interpolate import BSpline
        >>> b = BSpline.basis_element([0, 1, 2])
        >>> b.integrate(0, 1)
        array(0.5)

        If the integration limits are outside of the base interval, the result
        is controlled by the `extrapolate` parameter

        >>> b.integrate(-1, 1)
        array(0.0)
        >>> b.integrate(-1, 1, extrapolate=False)
        array(0.5)

        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.grid(True)
        >>> ax.axvline(0, c='r', lw=5, alpha=0.5)  # base interval
        >>> ax.axvline(2, c='r', lw=5, alpha=0.5)
        >>> xx = [-1, 1, 2]
        >>> ax.plot(xx, b(xx))
        >>> plt.show()

        """
        if extrapolate is None:
            extrapolate = self.extrapolate

        # Prepare self.t and self.c.
        self._ensure_c_contiguous()

        # Swap integration bounds if needed.
        sign = 1
        if b < a:
            a, b = b, a
            sign = -1
        n = self.t.size - self.k - 1

        if extrapolate != "periodic" and not extrapolate:
            # Shrink the integration interval, if needed.
            a = max(a, self.t[self.k])
            b = min(b, self.t[n])

            if self.c.ndim == 1:
                # Fast path: use FITPACK's routine
                # (cf _fitpack_impl.splint).
                integral = _fitpack_impl.splint(a, b, self.tck)
                return np.asarray(integral * sign)

        out = np.empty((2, prod(self.c.shape[1:])), dtype=self.c.dtype)

        # Compute the antiderivative.
        c = self.c
        ct = len(self.t) - len(c)
        if ct > 0:
            c = np.r_[c, np.zeros((ct,) + c.shape[1:])]
        ta, ca, ka = _fitpack_impl.splantider((self.t, c, self.k), 1)

        if extrapolate == 'periodic':
            # Split the integral into the part over period (can be several
            # of them) and the remaining part.

            ts, te = self.t[self.k], self.t[n]
            period = te - ts
            interval = b - a
            n_periods, left = divmod(interval, period)

            if n_periods > 0:
                # Evaluate the difference of antiderivatives.
                x = np.asarray([ts, te], dtype=np.float64)
                _dierckx.evaluate_spline(ta, ca.reshape(ca.shape[0], -1),
                                      ka, x, 0, False, out)
                integral = out[1] - out[0]
                integral *= n_periods
            else:
                integral = np.zeros((1, prod(self.c.shape[1:])),
                                    dtype=self.c.dtype)

            # Map a to [ts, te], b is always a + left.
            a = ts + (a - ts) % period
            b = a + left

            # If b <= te then we need to integrate over [a, b], otherwise
            # over [a, te] and from xs to what is remained.
            if b <= te:
                x = np.asarray([a, b], dtype=np.float64)
                _dierckx.evaluate_spline(ta, ca.reshape(ca.shape[0], -1),
                                      ka, x, 0, False, out)
                integral += out[1] - out[0]
            else:
                x = np.asarray([a, te], dtype=np.float64)
                _dierckx.evaluate_spline(ta, ca.reshape(ca.shape[0], -1),
                                      ka, x, 0, False, out)
                integral += out[1] - out[0]

                x = np.asarray([ts, ts + b - te], dtype=np.float64)
                _dierckx.evaluate_spline(ta, ca.reshape(ca.shape[0], -1),
                                      ka, x, 0, False, out)
                integral += out[1] - out[0]
        else:
            # Evaluate the difference of antiderivatives.
            x = np.asarray([a, b], dtype=np.float64)
            _dierckx.evaluate_spline(ta, ca.reshape(ca.shape[0], -1),
                                  ka, x, 0, extrapolate, out)
            integral = out[1] - out[0]

        integral *= sign
        return integral.reshape(ca.shape[1:])

    @classmethod
    def from_power_basis(cls, pp, bc_type='not-a-knot'):
        r"""
        Construct a polynomial in the B-spline basis
        from a piecewise polynomial in the power basis.

        For now, accepts ``CubicSpline`` instances only.

        Parameters
        ----------
        pp : CubicSpline
            A piecewise polynomial in the power basis, as created
            by ``CubicSpline``
        bc_type : string, optional
            Boundary condition type as in ``CubicSpline``: one of the
            ``not-a-knot``, ``natural``, ``clamped``, or ``periodic``.
            Necessary for construction an instance of ``BSpline`` class.
            Default is ``not-a-knot``.

        Returns
        -------
        b : BSpline object
            A new instance representing the initial polynomial
            in the B-spline basis.

        Notes
        -----
        .. versionadded:: 1.8.0

        Accepts only ``CubicSpline`` instances for now.

        The algorithm follows from differentiation
        the Marsden's identity [1]: each of coefficients of spline
        interpolation function in the B-spline basis is computed as follows:

        .. math::

            c_j = \sum_{m=0}^{k} \frac{(k-m)!}{k!}
                       c_{m,i} (-1)^{k-m} D^m p_{j,k}(x_i)

        :math:`c_{m, i}` - a coefficient of CubicSpline,
        :math:`D^m p_{j, k}(x_i)` - an m-th defivative of a dual polynomial
        in :math:`x_i`.

        ``k`` always equals 3 for now.

        First ``n - 2`` coefficients are computed in :math:`x_i = x_j`, e.g.

        .. math::

            c_1 = \sum_{m=0}^{k} \frac{(k-1)!}{k!} c_{m,1} D^m p_{j,3}(x_1)

        Last ``nod + 2`` coefficients are computed in ``x[-2]``,
        ``nod`` - number of derivatives at the ends.

        For example, consider :math:`x = [0, 1, 2, 3, 4]`,
        :math:`y = [1, 1, 1, 1, 1]` and bc_type = ``natural``

        The coefficients of CubicSpline in the power basis:

        :math:`[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]`

        The knot vector: :math:`t = [0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4]`

        In this case

        .. math::

            c_j = \frac{0!}{k!} c_{3, i} k! = c_{3, i} = 1,~j = 0, ..., 6

        References
        ----------
        .. [1] Tom Lyche and Knut Morken, Spline Methods, 2005, Section 3.1.2

        """
        from ._cubic import CubicSpline
        if not isinstance(pp, CubicSpline):
            raise NotImplementedError(f"Only CubicSpline objects are accepted "
                                      f"for now. Got {type(pp)} instead.")
        x = pp.x
        coef = pp.c
        k = pp.c.shape[0] - 1
        n = x.shape[0]

        if bc_type == 'not-a-knot':
            t = _not_a_knot(x, k)
        elif bc_type == 'natural' or bc_type == 'clamped':
            t = _augknt(x, k)
        elif bc_type == 'periodic':
            t = _periodic_knots(x, k)
        else:
            raise TypeError(f'Unknown boundary condition: {bc_type}')

        nod = t.shape[0] - (n + k + 1)  # number of derivatives at the ends
        c = np.zeros(n + nod, dtype=pp.c.dtype)
        for m in range(k + 1):
            for i in range(n - 2):
                c[i] += poch(k + 1, -m) * coef[m, i]\
                        * np.power(-1, k - m)\
                        * _diff_dual_poly(i, k, x[i], m, t)
            for j in range(n - 2, n + nod):
                c[j] += poch(k + 1, -m) * coef[m, n - 2]\
                        * np.power(-1, k - m)\
                        * _diff_dual_poly(j, k, x[n - 2], m, t)
        return cls.construct_fast(t, c, k, pp.extrapolate, pp.axis)

    def insert_knot(self, x, m=1):
        """Insert a new knot at `x` of multiplicity `m`.

        Given the knots and coefficients of a B-spline representation, create a
        new B-spline with a knot inserted `m` times at point `x`.

        Parameters
        ----------
        x : float
            The position of the new knot
        m : int, optional
            The number of times to insert the given knot (its multiplicity).
            Default is 1.

        Returns
        -------
        spl : BSpline object
            A new BSpline object with the new knot inserted.

        Notes
        -----
        Based on algorithms from [1]_ and [2]_.

        In case of a periodic spline (``self.extrapolate == "periodic"``)
        there must be either at least k interior knots t(j) satisfying
        ``t(k+1)<t(j)<=x`` or at least k interior knots t(j) satisfying
        ``x<=t(j)<t(n-k)``.

        This routine is functionally equivalent to `scipy.interpolate.insert`.

        .. versionadded:: 1.13

        References
        ----------
        .. [1] W. Boehm, "Inserting new knots into b-spline curves.",
            Computer Aided Design, 12, p.199-201, 1980.
            :doi:`10.1016/0010-4485(80)90154-2`.
        .. [2] P. Dierckx, "Curve and surface fitting with splines, Monographs on
            Numerical Analysis", Oxford University Press, 1993.

        See Also
        --------
        scipy.interpolate.insert

        Examples
        --------
        You can insert knots into a B-spline:

        >>> import numpy as np
        >>> from scipy.interpolate import BSpline, make_interp_spline
        >>> x = np.linspace(0, 10, 5)
        >>> y = np.sin(x)
        >>> spl = make_interp_spline(x, y, k=3)
        >>> spl.t
        array([ 0.,  0.,  0.,  0.,  5., 10., 10., 10., 10.])

        Insert a single knot

        >>> spl_1 = spl.insert_knot(3)
        >>> spl_1.t
        array([ 0.,  0.,  0.,  0.,  3.,  5., 10., 10., 10., 10.])

        Insert a multiple knot

        >>> spl_2 = spl.insert_knot(8, m=3)
        >>> spl_2.t
        array([ 0.,  0.,  0.,  0.,  5.,  8.,  8.,  8., 10., 10., 10., 10.])

        """
        if x < self.t[self.k] or x > self.t[-self.k-1]:
            raise ValueError(f"Cannot insert a knot at {x}.")
        if m <= 0:
            raise ValueError(f"`m` must be positive, got {m = }.")

        tt = self.t.copy()
        cc = self.c.copy()

        for _ in range(m):
            tt, cc = _insert(x, tt, cc, self.k, self.extrapolate == "periodic")
        return self.construct_fast(tt, cc, self.k, self.extrapolate, self.axis)


def _insert(xval, t, c, k, periodic=False):
    """Insert a single knot at `xval`."""
    #
    # This is a port of the FORTRAN `insert` routine by P. Dierckx,
    # https://github.com/scipy/scipy/blob/maintenance/1.11.x/scipy/interpolate/fitpack/insert.f
    # which carries the following comment:
    #
    # subroutine insert inserts a new knot x into a spline function s(x)
    # of degree k and calculates the b-spline representation of s(x) with
    # respect to the new set of knots. in addition, if iopt.ne.0, s(x)
    # will be considered as a periodic spline with period per=t(n-k)-t(k+1)
    # satisfying the boundary constraints
    #      t(i+n-2*k-1) = t(i)+per  ,i=1,2,...,2*k+1
    #      c(i+n-2*k-1) = c(i)      ,i=1,2,...,k
    # in that case, the knots and b-spline coefficients returned will also
    # satisfy these boundary constraints, i.e.
    #      tt(i+nn-2*k-1) = tt(i)+per  ,i=1,2,...,2*k+1
    #      cc(i+nn-2*k-1) = cc(i)      ,i=1,2,...,k
    interval = _dierckx.find_interval(t, k, float(xval), k, False)
    if interval < 0:
        # extrapolated values are guarded for in BSpline.insert_knot
        raise ValueError(f"Cannot insert the knot at {xval}.")

    # super edge case: a knot with multiplicity > k+1
    # see https://github.com/scipy/scipy/commit/037204c3e91
    if t[interval] == t[interval + k + 1]:
        interval -= 1

    if periodic:
        if (interval + 1 <= 2*k) and (interval + 1 >= t.shape[0] - 2*k):
            # in case of a periodic spline (iopt.ne.0) there must be
            # either at least k interior knots t(j) satisfying t(k+1)<t(j)<=x
            # or at least k interior knots t(j) satisfying x<=t(j)<t(n-k)
            raise ValueError("Not enough internal knots.")

    # knots
    tt = np.r_[t[:interval+1], xval, t[interval+1:]]

    newshape = (c.shape[0] + 1,) + c.shape[1:]
    cc = np.zeros(newshape, dtype=c.dtype)

    # coefficients
    cc[interval+1:, ...] = c[interval:, ...]

    for i in range(interval, interval-k, -1):
        fac = (xval - tt[i]) / (tt[i+k+1] - tt[i])
        cc[i, ...] = fac*c[i, ...] + (1. - fac)*c[i-1, ...]

    cc[:interval - k+1, ...] = c[:interval - k+1, ...]

    if periodic:
        # c   incorporate the boundary conditions for a periodic spline.
        n = tt.shape[0]
        nk = n - k - 1
        n2k = n - 2*k - 1
        T = tt[nk] - tt[k]   # period

        if interval >= nk - k:
            # adjust the left-hand boundary knots & coefs
            tt[:k] = tt[nk - k:nk] - T
            cc[:k, ...] = cc[n2k:n2k + k, ...]

        if interval <= 2*k-1:
            # adjust the right-hand boundary knots & coefs
            tt[n-k:] = tt[k+1:k+1+k] + T
            cc[n2k:n2k + k, ...] = cc[:k, ...]

    return tt, cc


#################################
#  Interpolating spline helpers #
#################################

def _not_a_knot(x, k):
    """Given data x, construct the knot vector w/ not-a-knot BC.
    cf de Boor, XIII(12).

    For even k, it's a bit ad hoc: Greville sites + omit 2nd and 2nd-to-last
    data points, a la not-a-knot.
    This seems to match what Dierckx does, too:
    https://github.com/scipy/scipy/blob/maintenance/1.11.x/scipy/interpolate/fitpack/fpcurf.f#L63-L80
    """
    x = np.asarray(x)
    if k % 2 == 1:
        k2 = (k + 1) // 2
        t = x.copy()
    else:
        k2 = k // 2
        t = (x[1:] + x[:-1]) / 2

    t = t[k2:-k2]
    t = np.r_[(x[0],)*(k+1), t, (x[-1],)*(k+1)]
    return t


def _augknt(x, k):
    """Construct a knot vector appropriate for the order-k interpolation."""
    return np.r_[(x[0],)*k, x, (x[-1],)*k]


def _convert_string_aliases(deriv, target_shape):
    if isinstance(deriv, str):
        if deriv == "clamped":
            deriv = [(1, np.zeros(target_shape))]
        elif deriv == "natural":
            deriv = [(2, np.zeros(target_shape))]
        else:
            raise ValueError(f"Unknown boundary condition : {deriv}")
    return deriv


def _process_deriv_spec(deriv):
    if deriv is not None:
        try:
            ords, vals = zip(*deriv)
        except TypeError as e:
            msg = ("Derivatives, `bc_type`, should be specified as a pair of "
                   "iterables of pairs of (order, value).")
            raise ValueError(msg) from e
    else:
        ords, vals = [], []
    return np.atleast_1d(ords, vals)


def _woodbury_algorithm(A, ur, ll, b, k):
    '''
    Solve a cyclic banded linear system with upper right
    and lower blocks of size ``(k-1) / 2`` using
    the Woodbury formula

    Parameters
    ----------
    A : 2-D array, shape(k, n)
        Matrix of diagonals of original matrix (see
        ``solve_banded`` documentation).
    ur : 2-D array, shape(bs, bs)
        Upper right block matrix.
    ll : 2-D array, shape(bs, bs)
        Lower left block matrix.
    b : 1-D array, shape(n,)
        Vector of constant terms of the system of linear equations.
    k : int
        B-spline degree.

    Returns
    -------
    c : 1-D array, shape(n,)
        Solution of the original system of linear equations.

    Notes
    -----
    This algorithm works only for systems with banded matrix A plus
    a correction term U @ V.T, where the matrix U @ V.T gives upper right
    and lower left block of A
    The system is solved with the following steps:
        1.  New systems of linear equations are constructed:
            A @ z_i = u_i,
            u_i - column vector of U,
            i = 1, ..., k - 1
        2.  Matrix Z is formed from vectors z_i:
            Z = [ z_1 | z_2 | ... | z_{k - 1} ]
        3.  Matrix H = (1 + V.T @ Z)^{-1}
        4.  The system A' @ y = b is solved
        5.  x = y - Z @ (H @ V.T @ y)
    Also, ``n`` should be greater than ``k``, otherwise corner block
    elements will intersect with diagonals.

    Examples
    --------
    Consider the case of n = 8, k = 5 (size of blocks - 2 x 2).
    The matrix of a system:       U:          V:
      x  x  x  *  *  a  b         a b 0 0     0 0 1 0
      x  x  x  x  *  *  c         0 c 0 0     0 0 0 1
      x  x  x  x  x  *  *         0 0 0 0     0 0 0 0
      *  x  x  x  x  x  *         0 0 0 0     0 0 0 0
      *  *  x  x  x  x  x         0 0 0 0     0 0 0 0
      d  *  *  x  x  x  x         0 0 d 0     1 0 0 0
      e  f  *  *  x  x  x         0 0 e f     0 1 0 0

    References
    ----------
    .. [1] William H. Press, Saul A. Teukolsky, William T. Vetterling
           and Brian P. Flannery, Numerical Recipes, 2007, Section 2.7.3

    '''
    k_mod = k - k % 2
    bs = int((k - 1) / 2) + (k + 1) % 2

    n = A.shape[1] + 1
    U = np.zeros((n - 1, k_mod))
    VT = np.zeros((k_mod, n - 1))  # V transpose

    # upper right block
    U[:bs, :bs] = ur
    VT[np.arange(bs), np.arange(bs) - bs] = 1

    # lower left block
    U[-bs:, -bs:] = ll
    VT[np.arange(bs) - bs, np.arange(bs)] = 1

    Z = solve_banded((bs, bs), A, U)

    H = solve(np.identity(k_mod) + VT @ Z, np.identity(k_mod))

    y = solve_banded((bs, bs), A, b)
    c = y - Z @ (H @ (VT @ y))

    return c


def _periodic_knots(x, k):
    '''
    returns vector of nodes on circle
    '''
    xc = np.copy(x)
    n = len(xc)
    if k % 2 == 0:
        dx = np.diff(xc)
        xc[1: -1] -= dx[:-1] / 2
    dx = np.diff(xc)
    t = np.zeros(n + 2 * k)
    t[k: -k] = xc
    for i in range(0, k):
        # filling first `k` elements in descending order
        t[k - i - 1] = t[k - i] - dx[-(i % (n - 1)) - 1]
        # filling last `k` elements in ascending order
        t[-k + i] = t[-k + i - 1] + dx[i % (n - 1)]
    return t


def _make_interp_per_full_matr(x, y, t, k):
    '''
    Returns a solution of a system for B-spline interpolation with periodic
    boundary conditions. First ``k - 1`` rows of matrix are conditions of
    periodicity (continuity of ``k - 1`` derivatives at the boundary points).
    Last ``n`` rows are interpolation conditions.
    RHS is ``k - 1`` zeros and ``n`` ordinates in this case.

    Parameters
    ----------
    x : 1-D array, shape (n,)
        Values of x - coordinate of a given set of points.
    y : 1-D array, shape (n,)
        Values of y - coordinate of a given set of points.
    t : 1-D array, shape(n+2*k,)
        Vector of knots.
    k : int
        The maximum degree of spline

    Returns
    -------
    c : 1-D array, shape (n+k-1,)
        B-spline coefficients

    Notes
    -----
    ``t`` is supposed to be taken on circle.

    '''

    x, y, t = map(np.asarray, (x, y, t))

    n = x.size
    # LHS: the colocation matrix + derivatives at edges
    matr = np.zeros((n + k - 1, n + k - 1))

    # derivatives at x[0] and x[-1]:
    for i in range(k - 1):
        bb = _dierckx.evaluate_all_bspl(t, k, x[0], k, i + 1)
        matr[i, : k + 1] += bb
        bb = _dierckx.evaluate_all_bspl(t, k, x[-1], n + k - 1, i + 1)[:-1]
        matr[i, -k:] -= bb

    # colocation matrix
    for i in range(n):
        xval = x[i]
        # find interval
        if xval == t[k]:
            left = k
        else:
            left = np.searchsorted(t, xval) - 1

        # fill a row
        bb = _dierckx.evaluate_all_bspl(t, k, xval, left)
        matr[i + k - 1, left-k:left+1] = bb

    # RHS
    b = np.r_[[0] * (k - 1), y]

    c = solve(matr, b)
    return c


def _handle_lhs_derivatives(t, k, xval, ab, kl, ku, deriv_ords, offset=0):
    """ Fill in the entries of the colocation matrix corresponding to known
    derivatives at `xval`.

    The colocation matrix is in the banded storage, as prepared by _coloc.
    No error checking.

    Parameters
    ----------
    t : ndarray, shape (nt + k + 1,)
        knots
    k : integer
        B-spline order
    xval : float
        The value at which to evaluate the derivatives at.
    ab : ndarray, shape(2*kl + ku + 1, nt), Fortran order
        B-spline colocation matrix.
        This argument is modified *in-place*.
    kl : integer
        Number of lower diagonals of ab.
    ku : integer
        Number of upper diagonals of ab.
    deriv_ords : 1D ndarray
        Orders of derivatives known at xval
    offset : integer, optional
        Skip this many rows of the matrix ab.

    """
    # find where `xval` is in the knot vector, `t`
    left = _dierckx.find_interval(t, k, float(xval), k, False)

    # compute and fill in the derivatives @ xval
    for row in range(deriv_ords.shape[0]):
        nu = deriv_ords[row]
        wrk = _dierckx.evaluate_all_bspl(t, k, xval, left, nu)

        # if A were a full matrix, it would be just
        # ``A[row + offset, left-k:left+1] = bb``.
        for a in range(k+1):
            clmn = left - k + a
            ab[kl + ku + offset + row - clmn, clmn] = wrk[a]


def _make_periodic_spline(x, y, t, k, axis):
    '''
    Compute the (coefficients of) interpolating B-spline with periodic
    boundary conditions.

    Parameters
    ----------
    x : array_like, shape (n,)
        Abscissas.
    y : array_like, shape (n,)
        Ordinates.
    k : int
        B-spline degree.
    t : array_like, shape (n + 2 * k,).
        Knots taken on a circle, ``k`` on the left and ``k`` on the right
        of the vector ``x``.

    Returns
    -------
    b : a BSpline object of the degree ``k`` and with knots ``t``.

    Notes
    -----
    The original system is formed by ``n + k - 1`` equations where the first
    ``k - 1`` of them stand for the ``k - 1`` derivatives continuity on the
    edges while the other equations correspond to an interpolating case
    (matching all the input points). Due to a special form of knot vector, it
    can be proved that in the original system the first and last ``k``
    coefficients of a spline function are the same, respectively. It follows
    from the fact that all ``k - 1`` derivatives are equal term by term at ends
    and that the matrix of the original system of linear equations is
    non-degenerate. So, we can reduce the number of equations to ``n - 1``
    (first ``k - 1`` equations could be reduced). Another trick of this
    implementation is cyclic shift of values of B-splines due to equality of
    ``k`` unknown coefficients. With this we can receive matrix of the system
    with upper right and lower left blocks, and ``k`` diagonals.  It allows
    to use Woodbury formula to optimize the computations.

    '''
    n = y.shape[0]

    extradim = prod(y.shape[1:])
    y_new = y.reshape(n, extradim)
    c = np.zeros((n + k - 1, extradim))

    # n <= k case is solved with full matrix
    if n <= k:
        for i in range(extradim):
            c[:, i] = _make_interp_per_full_matr(x, y_new[:, i], t, k)
        c = np.ascontiguousarray(c.reshape((n + k - 1,) + y.shape[1:]))
        return BSpline.construct_fast(t, c, k, extrapolate='periodic', axis=axis)

    nt = len(t) - k - 1

    # size of block elements
    kul = int(k / 2)

    # kl = ku = k
    ab = np.zeros((3 * k + 1, nt), dtype=np.float64, order='F')

    # upper right and lower left blocks
    ur = np.zeros((kul, kul))
    ll = np.zeros_like(ur)

    # `offset` is made to shift all the non-zero elements to the end of the
    # matrix
    # NB: 1. drop the last element of `x` because `x[0] = x[-1] + T` & `y[0] == y[-1]`
    #     2. pass ab.T to _coloc to make it C-ordered; below it'll be fed to banded
    #        LAPACK, which needs F-ordered arrays
    _dierckx._coloc(x[:-1], t, k, ab.T, k)

    # remove zeros before the matrix
    ab = ab[-k - (k + 1) % 2:, :]

    # The least elements in rows (except repetitions) are diagonals
    # of block matrices. Upper right matrix is an upper triangular
    # matrix while lower left is a lower triangular one.
    for i in range(kul):
        ur += np.diag(ab[-i - 1, i: kul], k=i)
        ll += np.diag(ab[i, -kul - (k % 2): n - 1 + 2 * kul - i], k=-i)

    # remove elements that occur in the last point
    # (first and last points are equivalent)
    A = ab[:, kul: -k + kul]

    for i in range(extradim):
        cc = _woodbury_algorithm(A, ur, ll, y_new[:, i][:-1], k)
        c[:, i] = np.concatenate((cc[-kul:], cc, cc[:kul + k % 2]))
    c = np.ascontiguousarray(c.reshape((n + k - 1,) + y.shape[1:]))
    return BSpline.construct_fast(t, c, k, extrapolate='periodic', axis=axis)


def make_interp_spline(x, y, k=3, t=None, bc_type=None, axis=0,
                       check_finite=True):
    """Compute the (coefficients of) interpolating B-spline.

    Parameters
    ----------
    x : array_like, shape (n,)
        Abscissas.
    y : array_like, shape (n, ...)
        Ordinates.
    k : int, optional
        B-spline degree. Default is cubic, ``k = 3``.
    t : array_like, shape (nt + k + 1,), optional.
        Knots.
        The number of knots needs to agree with the number of data points and
        the number of derivatives at the edges. Specifically, ``nt - n`` must
        equal ``len(deriv_l) + len(deriv_r)``.
    bc_type : 2-tuple or None
        Boundary conditions.
        Default is None, which means choosing the boundary conditions
        automatically. Otherwise, it must be a length-two tuple where the first
        element (``deriv_l``) sets the boundary conditions at ``x[0]`` and
        the second element (``deriv_r``) sets the boundary conditions at
        ``x[-1]``. Each of these must be an iterable of pairs
        ``(order, value)`` which gives the values of derivatives of specified
        orders at the given edge of the interpolation interval.
        Alternatively, the following string aliases are recognized:

        * ``"clamped"``: The first derivatives at the ends are zero. This is
           equivalent to ``bc_type=([(1, 0.0)], [(1, 0.0)])``.
        * ``"natural"``: The second derivatives at ends are zero. This is
          equivalent to ``bc_type=([(2, 0.0)], [(2, 0.0)])``.
        * ``"not-a-knot"`` (default): The first and second segments are the
          same polynomial. This is equivalent to having ``bc_type=None``.
        * ``"periodic"``: The values and the first ``k-1`` derivatives at the
          ends are equivalent.

    axis : int, optional
        Interpolation axis. Default is 0.
    check_finite : bool, optional
        Whether to check that the input arrays contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default is True.

    Returns
    -------
    b : a BSpline object of the degree ``k`` and with knots ``t``.

    See Also
    --------
    BSpline : base class representing the B-spline objects
    CubicSpline : a cubic spline in the polynomial basis
    make_lsq_spline : a similar factory function for spline fitting
    UnivariateSpline : a wrapper over FITPACK spline fitting routines
    splrep : a wrapper over FITPACK spline fitting routines

    Examples
    --------

    Use cubic interpolation on Chebyshev nodes:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> def cheb_nodes(N):
    ...     jj = 2.*np.arange(N) + 1
    ...     x = np.cos(np.pi * jj / 2 / N)[::-1]
    ...     return x

    >>> x = cheb_nodes(20)
    >>> y = np.sqrt(1 - x**2)

    >>> from scipy.interpolate import BSpline, make_interp_spline
    >>> b = make_interp_spline(x, y)
    >>> np.allclose(b(x), y)
    True

    Note that the default is a cubic spline with a not-a-knot boundary condition

    >>> b.k
    3

    Here we use a 'natural' spline, with zero 2nd derivatives at edges:

    >>> l, r = [(2, 0.0)], [(2, 0.0)]
    >>> b_n = make_interp_spline(x, y, bc_type=(l, r))  # or, bc_type="natural"
    >>> np.allclose(b_n(x), y)
    True
    >>> x0, x1 = x[0], x[-1]
    >>> np.allclose([b_n(x0, 2), b_n(x1, 2)], [0, 0])
    True

    Interpolation of parametric curves is also supported. As an example, we
    compute a discretization of a snail curve in polar coordinates

    >>> phi = np.linspace(0, 2.*np.pi, 40)
    >>> r = 0.3 + np.cos(phi)
    >>> x, y = r*np.cos(phi), r*np.sin(phi)  # convert to Cartesian coordinates

    Build an interpolating curve, parameterizing it by the angle

    >>> spl = make_interp_spline(phi, np.c_[x, y])

    Evaluate the interpolant on a finer grid (note that we transpose the result
    to unpack it into a pair of x- and y-arrays)

    >>> phi_new = np.linspace(0, 2.*np.pi, 100)
    >>> x_new, y_new = spl(phi_new).T

    Plot the result

    >>> plt.plot(x, y, 'o')
    >>> plt.plot(x_new, y_new, '-')
    >>> plt.show()

    Build a B-spline curve with 2 dimensional y

    >>> x = np.linspace(0, 2*np.pi, 10)
    >>> y = np.array([np.sin(x), np.cos(x)])

    Periodic condition is satisfied because y coordinates of points on the ends
    are equivalent

    >>> ax = plt.axes(projection='3d')
    >>> xx = np.linspace(0, 2*np.pi, 100)
    >>> bspl = make_interp_spline(x, y, k=5, bc_type='periodic', axis=1)
    >>> ax.plot3D(xx, *bspl(xx))
    >>> ax.scatter3D(x, *y, color='red')
    >>> plt.show()

    """
    # convert string aliases for the boundary conditions
    if bc_type is None or bc_type == 'not-a-knot' or bc_type == 'periodic':
        deriv_l, deriv_r = None, None
    elif isinstance(bc_type, str):
        deriv_l, deriv_r = bc_type, bc_type
    else:
        try:
            deriv_l, deriv_r = bc_type
        except TypeError as e:
            raise ValueError(f"Unknown boundary condition: {bc_type}") from e

    y = np.asarray(y)

    axis = normalize_axis_index(axis, y.ndim)

    x = _as_float_array(x, check_finite)
    y = _as_float_array(y, check_finite)

    y = np.moveaxis(y, axis, 0)    # now internally interp axis is zero

    # sanity check the input
    if bc_type == 'periodic' and not np.allclose(y[0], y[-1], atol=1e-15):
        raise ValueError("First and last points does not match while "
                         "periodic case expected")
    if x.size != y.shape[0]:
        raise ValueError(f'Shapes of x {x.shape} and y {y.shape} are incompatible')
    if np.any(x[1:] == x[:-1]):
        raise ValueError("Expect x to not have duplicates")
    if x.ndim != 1 or np.any(x[1:] < x[:-1]):
        raise ValueError("Expect x to be a 1D strictly increasing sequence.")

    # special-case k=0 right away
    if k == 0:
        if any(_ is not None for _ in (t, deriv_l, deriv_r)):
            raise ValueError("Too much info for k=0: t and bc_type can only "
                             "be None.")
        t = np.r_[x, x[-1]]
        c = np.asarray(y)
        c = np.ascontiguousarray(c, dtype=_get_dtype(c.dtype))
        return BSpline.construct_fast(t, c, k, axis=axis)

    # special-case k=1 (e.g., Lyche and Morken, Eq.(2.16))
    if k == 1 and t is None:
        if not (deriv_l is None and deriv_r is None):
            raise ValueError("Too much info for k=1: bc_type can only be None.")
        t = np.r_[x[0], x, x[-1]]
        c = np.asarray(y)
        c = np.ascontiguousarray(c, dtype=_get_dtype(c.dtype))
        return BSpline.construct_fast(t, c, k, axis=axis)

    k = operator.index(k)

    if bc_type == 'periodic' and t is not None:
        raise NotImplementedError("For periodic case t is constructed "
                                  "automatically and can not be passed "
                                  "manually")

    # come up with a sensible knot vector, if needed
    if t is None:
        if deriv_l is None and deriv_r is None:
            if bc_type == 'periodic':
                t = _periodic_knots(x, k)
            else:
                t = _not_a_knot(x, k)
        else:
            t = _augknt(x, k)

    t = _as_float_array(t, check_finite)

    if k < 0:
        raise ValueError("Expect non-negative k.")
    if t.ndim != 1 or np.any(t[1:] < t[:-1]):
        raise ValueError("Expect t to be a 1-D sorted array_like.")
    if t.size < x.size + k + 1:
        raise ValueError('Got %d knots, need at least %d.' %
                         (t.size, x.size + k + 1))
    if (x[0] < t[k]) or (x[-1] > t[-k]):
        raise ValueError(f'Out of bounds w/ x = {x}.')

    if bc_type == 'periodic':
        return _make_periodic_spline(x, y, t, k, axis)

    # Here : deriv_l, r = [(nu, value), ...]
    deriv_l = _convert_string_aliases(deriv_l, y.shape[1:])
    deriv_l_ords, deriv_l_vals = _process_deriv_spec(deriv_l)
    nleft = deriv_l_ords.shape[0]

    deriv_r = _convert_string_aliases(deriv_r, y.shape[1:])
    deriv_r_ords, deriv_r_vals = _process_deriv_spec(deriv_r)
    nright = deriv_r_ords.shape[0]

    if not all(0 <= i <= k for i in deriv_l_ords):
        raise ValueError(f"Bad boundary conditions at {x[0]}.")

    if not all(0 <= i <= k for i in deriv_r_ords):
        raise ValueError(f"Bad boundary conditions at {x[-1]}.")

    # have `n` conditions for `nt` coefficients; need nt-n derivatives
    n = x.size
    nt = t.size - k - 1

    if nt - n != nleft + nright:
        raise ValueError("The number of derivatives at boundaries does not "
                         f"match: expected {nt-n}, got {nleft}+{nright}")

    # bail out if the `y` array is zero-sized
    if y.size == 0:
        c = np.zeros((nt,) + y.shape[1:], dtype=float)
        return BSpline.construct_fast(t, c, k, axis=axis)

    # set up the LHS: the colocation matrix + derivatives at boundaries
    # NB: ab is in F order for banded LAPACK; _coloc needs C-ordered arrays,
    #     this pass ab.T into _coloc
    kl = ku = k
    ab = np.zeros((2*kl + ku + 1, nt), dtype=np.float64, order='F')
    _dierckx._coloc(x, t, k, ab.T, nleft)
    if nleft > 0:
        _handle_lhs_derivatives(t, k, x[0], ab, kl, ku, deriv_l_ords)
    if nright > 0:
        _handle_lhs_derivatives(t, k, x[-1], ab, kl, ku, deriv_r_ords,
                                offset=nt-nright)

    # set up the RHS: values to interpolate (+ derivative values, if any)
    extradim = prod(y.shape[1:])
    rhs = np.empty((nt, extradim), dtype=y.dtype)
    if nleft > 0:
        rhs[:nleft] = deriv_l_vals.reshape(-1, extradim)
    rhs[nleft:nt - nright] = y.reshape(-1, extradim)
    if nright > 0:
        rhs[nt - nright:] = deriv_r_vals.reshape(-1, extradim)

    # solve Ab @ x = rhs; this is the relevant part of linalg.solve_banded
    if check_finite:
        ab, rhs = map(np.asarray_chkfinite, (ab, rhs))
    gbsv, = get_lapack_funcs(('gbsv',), (ab, rhs))
    lu, piv, c, info = gbsv(kl, ku, ab, rhs,
                            overwrite_ab=True, overwrite_b=True)

    if info > 0:
        raise LinAlgError("Colocation matrix is singular.")
    elif info < 0:
        raise ValueError('illegal value in %d-th argument of internal gbsv' % -info)

    c = np.ascontiguousarray(c.reshape((nt,) + y.shape[1:]))
    return BSpline.construct_fast(t, c, k, axis=axis)


def make_lsq_spline(x, y, t, k=3, w=None, axis=0, check_finite=True, *, method="qr"):
    r"""Compute the (coefficients of) an LSQ (Least SQuared) based
    fitting B-spline.

    The result is a linear combination

    .. math::

            S(x) = \sum_j c_j B_j(x; t)

    of the B-spline basis elements, :math:`B_j(x; t)`, which minimizes

    .. math::

        \sum_{j} \left( w_j \times (S(x_j) - y_j) \right)^2

    Parameters
    ----------
    x : array_like, shape (m,)
        Abscissas.
    y : array_like, shape (m, ...)
        Ordinates.
    t : array_like, shape (n + k + 1,).
        Knots.
        Knots and data points must satisfy Schoenberg-Whitney conditions.
    k : int, optional
        B-spline degree. Default is cubic, ``k = 3``.
    w : array_like, shape (m,), optional
        Weights for spline fitting. Must be positive. If ``None``,
        then weights are all equal.
        Default is ``None``.
    axis : int, optional
        Interpolation axis. Default is zero.
    check_finite : bool, optional
        Whether to check that the input arrays contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default is True.
    method : str, optional
        Method for solving the linear LSQ problem. Allowed values are "norm-eq"
        (Explicitly construct and solve the normal system of equations), and
        "qr" (Use the QR factorization of the design matrix).
        Default is "qr".

    Returns
    -------
    b : a BSpline object of the degree ``k`` with knots ``t``.

    See Also
    --------
    BSpline : base class representing the B-spline objects
    make_interp_spline : a similar factory function for interpolating splines
    LSQUnivariateSpline : a FITPACK-based spline fitting routine
    splrep : a FITPACK-based fitting routine

    Notes
    -----
    The number of data points must be larger than the spline degree ``k``.

    Knots ``t`` must satisfy the Schoenberg-Whitney conditions,
    i.e., there must be a subset of data points ``x[j]`` such that
    ``t[j] < x[j] < t[j+k+1]``, for ``j=0, 1,...,n-k-2``.

    Examples
    --------
    Generate some noisy data:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> x = np.linspace(-3, 3, 50)
    >>> y = np.exp(-x**2) + 0.1 * rng.standard_normal(50)

    Now fit a smoothing cubic spline with a pre-defined internal knots.
    Here we make the knot vector (k+1)-regular by adding boundary knots:

    >>> from scipy.interpolate import make_lsq_spline, BSpline
    >>> t = [-1, 0, 1]
    >>> k = 3
    >>> t = np.r_[(x[0],)*(k+1),
    ...           t,
    ...           (x[-1],)*(k+1)]
    >>> spl = make_lsq_spline(x, y, t, k)

    For comparison, we also construct an interpolating spline for the same
    set of data:

    >>> from scipy.interpolate import make_interp_spline
    >>> spl_i = make_interp_spline(x, y)

    Plot both:

    >>> xs = np.linspace(-3, 3, 100)
    >>> plt.plot(x, y, 'ro', ms=5)
    >>> plt.plot(xs, spl(xs), 'g-', lw=3, label='LSQ spline')
    >>> plt.plot(xs, spl_i(xs), 'b-', lw=3, alpha=0.7, label='interp spline')
    >>> plt.legend(loc='best')
    >>> plt.show()

    **NaN handling**: If the input arrays contain ``nan`` values, the result is
    not useful since the underlying spline fitting routines cannot deal with
    ``nan``. A workaround is to use zero weights for not-a-number data points:

    >>> y[8] = np.nan
    >>> w = np.isnan(y)
    >>> y[w] = 0.
    >>> tck = make_lsq_spline(x, y, t, w=~w)

    Notice the need to replace a ``nan`` by a numerical value (precise value
    does not matter as long as the corresponding weight is zero.)

    """
    x = _as_float_array(x, check_finite)
    y = _as_float_array(y, check_finite)
    t = _as_float_array(t, check_finite)
    if w is not None:
        w = _as_float_array(w, check_finite)
    else:
        w = np.ones_like(x)
    k = operator.index(k)

    axis = normalize_axis_index(axis, y.ndim)

    y = np.moveaxis(y, axis, 0)    # now internally interp axis is zero

    if x.ndim != 1:
        raise ValueError("Expect x to be a 1-D sequence.")
    if x.shape[0] < k+1:
        raise ValueError("Need more x points.")
    if k < 0:
        raise ValueError("Expect non-negative k.")
    if t.ndim != 1 or np.any(t[1:] - t[:-1] < 0):
        raise ValueError("Expect t to be a 1D strictly increasing sequence.")
    if x.size != y.shape[0]:
        raise ValueError(f'Shapes of x {x.shape} and y {y.shape} are incompatible')
    if k > 0 and np.any((x < t[k]) | (x > t[-k])):
        raise ValueError(f'Out of bounds w/ x = {x}.')
    if x.size != w.size:
        raise ValueError(f'Shapes of x {x.shape} and w {w.shape} are incompatible')
    if method == "norm-eq" and np.any(x[1:] - x[:-1] <= 0):
        raise ValueError("Expect x to be a 1D strictly increasing sequence.")
    if method == "qr" and any(x[1:] - x[:-1] < 0):
        raise ValueError("Expect x to be a 1D non-decreasing sequence.")

    # number of coefficients
    n = t.size - k - 1

    # complex y: view as float, preserve the length
    was_complex =  y.dtype.kind == 'c'
    yy = y.view(float)
    if was_complex and y.ndim == 1:
        yy = yy.reshape(y.shape[0], 2)

    # multiple r.h.s
    extradim = prod(yy.shape[1:])
    yy = yy.reshape(-1, extradim)

    # complex y: view as float, preserve the length
    was_complex =  y.dtype.kind == 'c'
    yy = y.view(float)
    if was_complex and y.ndim == 1:
        yy = yy.reshape(y.shape[0], 2)

    # multiple r.h.s
    extradim = prod(yy.shape[1:])
    yy = yy.reshape(-1, extradim)

    if method == "norm-eq":
        # construct A.T @ A and rhs with A the colocation matrix, and
        # rhs = A.T @ y for solving the LSQ problem  ``A.T @ A @ c = A.T @ y``
        lower = True
        ab = np.zeros((k+1, n), dtype=np.float64, order='F')
        rhs = np.zeros((n, extradim), dtype=np.float64)
        _dierckx._norm_eq_lsq(x, t, k,
                              yy,
                              w,
                              ab.T, rhs)

        # undo complex -> float and flattening the trailing dims
        if was_complex:
            rhs = rhs.view(complex)

        rhs = rhs.reshape((n,) + y.shape[1:])

        # have observation matrix & rhs, can solve the LSQ problem
        cho_decomp = cholesky_banded(ab, overwrite_ab=True, lower=lower,
                                     check_finite=check_finite)
        c = cho_solve_banded((cho_decomp, lower), rhs, overwrite_b=True,
                             check_finite=check_finite)
    elif method == "qr":
        _, _, c = _lsq_solve_qr(x, yy, t, k, w)

        if was_complex:
            c = c.view(complex)

    else:
        raise ValueError(f"Unknown {method =}.")


    # restore the shape of `c` for both single and multiple r.h.s.
    c = c.reshape((n,) + y.shape[1:])
    c = np.ascontiguousarray(c)
    return BSpline.construct_fast(t, c, k, axis=axis)


######################
# LSQ spline helpers #
######################

def _lsq_solve_qr(x, y, t, k, w):
    """Solve for the LSQ spline coeffs given x, y and knots.

    `y` is always 2D: for 1D data, the shape is ``(m, 1)``.
    `w` is always 1D: one weight value per `x` value.

    """
    assert y.ndim == 2

    y_w = y * w[:, None]
    A, offset, nc = _dierckx.data_matrix(x, t, k, w)
    _dierckx.qr_reduce(A, offset, nc, y_w)         # modifies arguments in-place
    c = _dierckx.fpback(A, nc, y_w)

    return A, y_w, c


#############################
#  Smoothing spline helpers #
#############################

def _compute_optimal_gcv_parameter(X, wE, y, w):
    """
    Returns an optimal regularization parameter from the GCV criteria [1].

    Parameters
    ----------
    X : array, shape (5, n)
        5 bands of the design matrix ``X`` stored in LAPACK banded storage.
    wE : array, shape (5, n)
        5 bands of the penalty matrix :math:`W^{-1} E` stored in LAPACK banded
        storage.
    y : array, shape (n,)
        Ordinates.
    w : array, shape (n,)
        Vector of weights.

    Returns
    -------
    lam : float
        An optimal from the GCV criteria point of view regularization
        parameter.

    Notes
    -----
    No checks are performed.

    References
    ----------
    .. [1] G. Wahba, "Estimating the smoothing parameter" in Spline models
        for observational data, Philadelphia, Pennsylvania: Society for
        Industrial and Applied Mathematics, 1990, pp. 45-65.
        :doi:`10.1137/1.9781611970128`

    """

    def compute_banded_symmetric_XT_W_Y(X, w, Y):
        """
        Assuming that the product :math:`X^T W Y` is symmetric and both ``X``
        and ``Y`` are 5-banded, compute the unique bands of the product.

        Parameters
        ----------
        X : array, shape (5, n)
            5 bands of the matrix ``X`` stored in LAPACK banded storage.
        w : array, shape (n,)
            Array of weights
        Y : array, shape (5, n)
            5 bands of the matrix ``Y`` stored in LAPACK banded storage.

        Returns
        -------
        res : array, shape (4, n)
            The result of the product :math:`X^T Y` stored in the banded way.

        Notes
        -----
        As far as the matrices ``X`` and ``Y`` are 5-banded, their product
        :math:`X^T W Y` is 7-banded. It is also symmetric, so we can store only
        unique diagonals.

        """
        # compute W Y
        W_Y = np.copy(Y)

        W_Y[2] *= w
        for i in range(2):
            W_Y[i, 2 - i:] *= w[:-2 + i]
            W_Y[3 + i, :-1 - i] *= w[1 + i:]

        n = X.shape[1]
        res = np.zeros((4, n))
        for i in range(n):
            for j in range(min(n-i, 4)):
                res[-j-1, i + j] = sum(X[j:, i] * W_Y[:5-j, i + j])
        return res

    def compute_b_inv(A):
        """
        Inverse 3 central bands of matrix :math:`A=U^T D^{-1} U` assuming that
        ``U`` is a unit upper triangular banded matrix using an algorithm
        proposed in [1].

        Parameters
        ----------
        A : array, shape (4, n)
            Matrix to inverse, stored in LAPACK banded storage.

        Returns
        -------
        B : array, shape (4, n)
            3 unique bands of the symmetric matrix that is an inverse to ``A``.
            The first row is filled with zeros.

        Notes
        -----
        The algorithm is based on the cholesky decomposition and, therefore,
        in case matrix ``A`` is close to not positive defined, the function
        raises LinalgError.

        Both matrices ``A`` and ``B`` are stored in LAPACK banded storage.

        References
        ----------
        .. [1] M. F. Hutchinson and F. R. de Hoog, "Smoothing noisy data with
            spline functions," Numerische Mathematik, vol. 47, no. 1,
            pp. 99-106, 1985.
            :doi:`10.1007/BF01389878`

        """

        def find_b_inv_elem(i, j, U, D, B):
            rng = min(3, n - i - 1)
            rng_sum = 0.
            if j == 0:
                # use 2-nd formula from [1]
                for k in range(1, rng + 1):
                    rng_sum -= U[-k - 1, i + k] * B[-k - 1, i + k]
                rng_sum += D[i]
                B[-1, i] = rng_sum
            else:
                # use 1-st formula from [1]
                for k in range(1, rng + 1):
                    diag = abs(k - j)
                    ind = i + min(k, j)
                    rng_sum -= U[-k - 1, i + k] * B[-diag - 1, ind + diag]
                B[-j - 1, i + j] = rng_sum

        U = cholesky_banded(A)
        for i in range(2, 5):
            U[-i, i-1:] /= U[-1, :-i+1]
        D = 1. / (U[-1])**2
        U[-1] /= U[-1]

        n = U.shape[1]

        B = np.zeros(shape=(4, n))
        for i in range(n - 1, -1, -1):
            for j in range(min(3, n - i - 1), -1, -1):
                find_b_inv_elem(i, j, U, D, B)
        # the first row contains garbage and should be removed
        B[0] = [0.] * n
        return B

    def _gcv(lam, X, XtWX, wE, XtE):
        r"""
        Computes the generalized cross-validation criteria [1].

        Parameters
        ----------
        lam : float, (:math:`\lambda \geq 0`)
            Regularization parameter.
        X : array, shape (5, n)
            Matrix is stored in LAPACK banded storage.
        XtWX : array, shape (4, n)
            Product :math:`X^T W X` stored in LAPACK banded storage.
        wE : array, shape (5, n)
            Matrix :math:`W^{-1} E` stored in LAPACK banded storage.
        XtE : array, shape (4, n)
            Product :math:`X^T E` stored in LAPACK banded storage.

        Returns
        -------
        res : float
            Value of the GCV criteria with the regularization parameter
            :math:`\lambda`.

        Notes
        -----
        Criteria is computed from the formula (1.3.2) [3]:

        .. math:

        GCV(\lambda) = \dfrac{1}{n} \sum\limits_{k = 1}^{n} \dfrac{ \left(
        y_k - f_{\lambda}(x_k) \right)^2}{\left( 1 - \Tr{A}/n\right)^2}$.
        The criteria is discussed in section 1.3 [3].

        The numerator is computed using (2.2.4) [3] and the denominator is
        computed using an algorithm from [2] (see in the ``compute_b_inv``
        function).

        References
        ----------
        .. [1] G. Wahba, "Estimating the smoothing parameter" in Spline models
            for observational data, Philadelphia, Pennsylvania: Society for
            Industrial and Applied Mathematics, 1990, pp. 45-65.
            :doi:`10.1137/1.9781611970128`
        .. [2] M. F. Hutchinson and F. R. de Hoog, "Smoothing noisy data with
            spline functions," Numerische Mathematik, vol. 47, no. 1,
            pp. 99-106, 1985.
            :doi:`10.1007/BF01389878`
        .. [3] E. Zemlyanoy, "Generalized cross-validation smoothing splines",
            BSc thesis, 2022. Might be available (in Russian)
            `here <https://www.hse.ru/ba/am/students/diplomas/620910604>`_

        """
        # Compute the numerator from (2.2.4) [3]
        n = X.shape[1]
        c = solve_banded((2, 2), X + lam * wE, y)
        res = np.zeros(n)
        # compute ``W^{-1} E c`` with respect to banded-storage of ``E``
        tmp = wE * c
        for i in range(n):
            for j in range(max(0, i - n + 3), min(5, i + 3)):
                res[i] += tmp[j, i + 2 - j]
        numer = np.linalg.norm(lam * res)**2 / n

        # compute the denominator
        lhs = XtWX + lam * XtE
        try:
            b_banded = compute_b_inv(lhs)
            # compute the trace of the product b_banded @ XtX
            tr = b_banded * XtWX
            tr[:-1] *= 2
            # find the denominator
            denom = (1 - sum(sum(tr)) / n)**2
        except LinAlgError:
            # cholesky decomposition cannot be performed
            raise ValueError('Seems like the problem is ill-posed')

        res = numer / denom

        return res

    n = X.shape[1]

    XtWX = compute_banded_symmetric_XT_W_Y(X, w, X)
    XtE = compute_banded_symmetric_XT_W_Y(X, w, wE)

    def fun(lam):
        return _gcv(lam, X, XtWX, wE, XtE)

    gcv_est = minimize_scalar(fun, bounds=(0, n), method='Bounded')
    if gcv_est.success:
        return gcv_est.x
    raise ValueError(f"Unable to find minimum of the GCV "
                     f"function: {gcv_est.message}")


def _coeff_of_divided_diff(x):
    """
    Returns the coefficients of the divided difference.

    Parameters
    ----------
    x : array, shape (n,)
        Array which is used for the computation of divided difference.

    Returns
    -------
    res : array_like, shape (n,)
        Coefficients of the divided difference.

    Notes
    -----
    Vector ``x`` should have unique elements, otherwise an error division by
    zero might be raised.

    No checks are performed.

    """
    n = x.shape[0]
    res = np.zeros(n)
    for i in range(n):
        pp = 1.
        for k in range(n):
            if k != i:
                pp *= (x[i] - x[k])
        res[i] = 1. / pp
    return res


def make_smoothing_spline(x, y, w=None, lam=None):
    r"""
    Compute the (coefficients of) smoothing cubic spline function using
    ``lam`` to control the tradeoff between the amount of smoothness of the
    curve and its proximity to the data. In case ``lam`` is None, using the
    GCV criteria [1] to find it.

    A smoothing spline is found as a solution to the regularized weighted
    linear regression problem:

    .. math::

        \sum\limits_{i=1}^n w_i\lvert y_i - f(x_i) \rvert^2 +
        \lambda\int\limits_{x_1}^{x_n} (f^{(2)}(u))^2 d u

    where :math:`f` is a spline function, :math:`w` is a vector of weights and
    :math:`\lambda` is a regularization parameter.

    If ``lam`` is None, we use the GCV criteria to find an optimal
    regularization parameter, otherwise we solve the regularized weighted
    linear regression problem with given parameter. The parameter controls
    the tradeoff in the following way: the larger the parameter becomes, the
    smoother the function gets.

    Parameters
    ----------
    x : array_like, shape (n,)
        Abscissas. `n` must be at least 5.
    y : array_like, shape (n,)
        Ordinates. `n` must be at least 5.
    w : array_like, shape (n,), optional
        Vector of weights. Default is ``np.ones_like(x)``.
    lam : float, (:math:`\lambda \geq 0`), optional
        Regularization parameter. If ``lam`` is None, then it is found from
        the GCV criteria. Default is None.

    Returns
    -------
    func : a BSpline object.
        A callable representing a spline in the B-spline basis
        as a solution of the problem of smoothing splines using
        the GCV criteria [1] in case ``lam`` is None, otherwise using the
        given parameter ``lam``.

    Notes
    -----
    This algorithm is a clean room reimplementation of the algorithm
    introduced by Woltring in FORTRAN [2]. The original version cannot be used
    in SciPy source code because of the license issues. The details of the
    reimplementation are discussed here (available only in Russian) [4].

    If the vector of weights ``w`` is None, we assume that all the points are
    equal in terms of weights, and vector of weights is vector of ones.

    Note that in weighted residual sum of squares, weights are not squared:
    :math:`\sum\limits_{i=1}^n w_i\lvert y_i - f(x_i) \rvert^2` while in
    ``splrep`` the sum is built from the squared weights.

    In cases when the initial problem is ill-posed (for example, the product
    :math:`X^T W X` where :math:`X` is a design matrix is not a positive
    defined matrix) a ValueError is raised.

    References
    ----------
    .. [1] G. Wahba, "Estimating the smoothing parameter" in Spline models for
        observational data, Philadelphia, Pennsylvania: Society for Industrial
        and Applied Mathematics, 1990, pp. 45-65.
        :doi:`10.1137/1.9781611970128`
    .. [2] H. J. Woltring, A Fortran package for generalized, cross-validatory
        spline smoothing and differentiation, Advances in Engineering
        Software, vol. 8, no. 2, pp. 104-113, 1986.
        :doi:`10.1016/0141-1195(86)90098-7`
    .. [3] T. Hastie, J. Friedman, and R. Tisbshirani, "Smoothing Splines" in
        The elements of Statistical Learning: Data Mining, Inference, and
        prediction, New York: Springer, 2017, pp. 241-249.
        :doi:`10.1007/978-0-387-84858-7`
    .. [4] E. Zemlyanoy, "Generalized cross-validation smoothing splines",
        BSc thesis, 2022.
        `<https://www.hse.ru/ba/am/students/diplomas/620910604>`_ (in
        Russian)

    Examples
    --------
    Generate some noisy data

    >>> import numpy as np
    >>> np.random.seed(1234)
    >>> n = 200
    >>> def func(x):
    ...    return x**3 + x**2 * np.sin(4 * x)
    >>> x = np.sort(np.random.random_sample(n) * 4 - 2)
    >>> y = func(x) + np.random.normal(scale=1.5, size=n)

    Make a smoothing spline function

    >>> from scipy.interpolate import make_smoothing_spline
    >>> spl = make_smoothing_spline(x, y)

    Plot both

    >>> import matplotlib.pyplot as plt
    >>> grid = np.linspace(x[0], x[-1], 400)
    >>> plt.plot(grid, spl(grid), label='Spline')
    >>> plt.plot(grid, func(grid), label='Original function')
    >>> plt.scatter(x, y, marker='.')
    >>> plt.legend(loc='best')
    >>> plt.show()

    """

    x = np.ascontiguousarray(x, dtype=float)
    y = np.ascontiguousarray(y, dtype=float)

    if any(x[1:] - x[:-1] <= 0):
        raise ValueError('``x`` should be an ascending array')

    if x.ndim != 1 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError('``x`` and ``y`` should be one dimensional and the'
                         ' same size')

    if w is None:
        w = np.ones(len(x))
    else:
        w = np.ascontiguousarray(w)
        if any(w <= 0):
            raise ValueError('Invalid vector of weights')

    t = np.r_[[x[0]] * 3, x, [x[-1]] * 3]
    n = x.shape[0]

    if n <= 4:
        raise ValueError('``x`` and ``y`` length must be at least 5')

    # It is known that the solution to the stated minimization problem exists
    # and is a natural cubic spline with vector of knots equal to the unique
    # elements of ``x`` [3], so we will solve the problem in the basis of
    # natural splines.

    # create design matrix in the B-spline basis
    X_bspl = BSpline.design_matrix(x, t, 3)
    # move from B-spline basis to the basis of natural splines using equations
    # (2.1.7) [4]
    # central elements
    X = np.zeros((5, n))
    for i in range(1, 4):
        X[i, 2: -2] = X_bspl[i: i - 4, 3: -3][np.diag_indices(n - 4)]

    # first elements
    X[1, 1] = X_bspl[0, 0]
    X[2, :2] = ((x[2] + x[1] - 2 * x[0]) * X_bspl[0, 0],
                X_bspl[1, 1] + X_bspl[1, 2])
    X[3, :2] = ((x[2] - x[0]) * X_bspl[1, 1], X_bspl[2, 2])

    # last elements
    X[1, -2:] = (X_bspl[-3, -3], (x[-1] - x[-3]) * X_bspl[-2, -2])
    X[2, -2:] = (X_bspl[-2, -3] + X_bspl[-2, -2],
                 (2 * x[-1] - x[-2] - x[-3]) * X_bspl[-1, -1])
    X[3, -2] = X_bspl[-1, -1]

    # create penalty matrix and divide it by vector of weights: W^{-1} E
    wE = np.zeros((5, n))
    wE[2:, 0] = _coeff_of_divided_diff(x[:3]) / w[:3]
    wE[1:, 1] = _coeff_of_divided_diff(x[:4]) / w[:4]
    for j in range(2, n - 2):
        wE[:, j] = (x[j+2] - x[j-2]) * _coeff_of_divided_diff(x[j-2:j+3])\
                   / w[j-2: j+3]

    wE[:-1, -2] = -_coeff_of_divided_diff(x[-4:]) / w[-4:]
    wE[:-2, -1] = _coeff_of_divided_diff(x[-3:]) / w[-3:]
    wE *= 6

    if lam is None:
        lam = _compute_optimal_gcv_parameter(X, wE, y, w)
    elif lam < 0.:
        raise ValueError('Regularization parameter should be non-negative')

    # solve the initial problem in the basis of natural splines
    c = solve_banded((2, 2), X + lam * wE, y)
    # move back to B-spline basis using equations (2.2.10) [4]
    c_ = np.r_[c[0] * (t[5] + t[4] - 2 * t[3]) + c[1],
               c[0] * (t[5] - t[3]) + c[1],
               c[1: -1],
               c[-1] * (t[-4] - t[-6]) + c[-2],
               c[-1] * (2 * t[-4] - t[-5] - t[-6]) + c[-2]]

    return BSpline.construct_fast(t, c_, 3)


########################
#  FITPACK look-alikes #
########################

def fpcheck(x, t, k):
    """ Check consistency of the data vector `x` and the knot vector `t`.

    Return None if inputs are consistent, raises a ValueError otherwise.
    """
    # This routine is a clone of the `fpchec` Fortran routine,
    # https://github.com/scipy/scipy/blob/main/scipy/interpolate/fitpack/fpchec.f
    # which carries the following comment:
    #
    # subroutine fpchec verifies the number and the position of the knots
    #  t(j),j=1,2,...,n of a spline of degree k, in relation to the number
    #  and the position of the data points x(i),i=1,2,...,m. if all of the
    #  following conditions are fulfilled, the error parameter ier is set
    #  to zero. if one of the conditions is violated ier is set to ten.
    #      1) k+1 <= n-k-1 <= m
    #      2) t(1) <= t(2) <= ... <= t(k+1)
    #         t(n-k) <= t(n-k+1) <= ... <= t(n)
    #      3) t(k+1) < t(k+2) < ... < t(n-k)
    #      4) t(k+1) <= x(i) <= t(n-k)
    #      5) the conditions specified by schoenberg and whitney must hold
    #         for at least one subset of data points, i.e. there must be a
    #         subset of data points y(j) such that
    #             t(j) < y(j) < t(j+k+1), j=1,2,...,n-k-1
    x = np.asarray(x)
    t = np.asarray(t)

    if x.ndim != 1 or t.ndim != 1:
        raise ValueError(f"Expect `x` and `t` be 1D sequences. Got {x = } and {t = }")

    m = x.shape[0]
    n = t.shape[0]
    nk1 = n - k - 1

    # check condition no 1
    # c      1) k+1 <= n-k-1 <= m
    if not (k + 1 <= nk1 <= m):
        raise ValueError(f"Need k+1 <= n-k-1 <= m. Got {m = }, {n = } and {k = }.")

    # check condition no 2
    # c      2) t(1) <= t(2) <= ... <= t(k+1)
    # c         t(n-k) <= t(n-k+1) <= ... <= t(n)
    if (t[:k+1] > t[1:k+2]).any():
        raise ValueError(f"First k knots must be ordered; got {t = }.")

    if (t[nk1:] < t[nk1-1:-1]).any():
        raise ValueError(f"Last k knots must be ordered; got {t = }.")

    # c  check condition no 3
    # c      3) t(k+1) < t(k+2) < ... < t(n-k)
    if (t[k+1:n-k] <= t[k:n-k-1]).any():
        raise ValueError(f"Internal knots must be distinct. Got {t = }.")

    # c  check condition no 4
    # c      4) t(k+1) <= x(i) <= t(n-k)
    # NB: FITPACK's fpchec only checks x[0] & x[-1], so we follow.
    if (x[0] < t[k]) or (x[-1] > t[n-k-1]):
        raise ValueError(f"Out of bounds: {x = } and {t = }.")

    # c  check condition no 5
    # c      5) the conditions specified by schoenberg and whitney must hold
    # c         for at least one subset of data points, i.e. there must be a
    # c         subset of data points y(j) such that
    # c             t(j) < y(j) < t(j+k+1), j=1,2,...,n-k-1
    mesg = f"Schoenberg-Whitney condition is violated with {t = } and {x =}."

    if (x[0] >= t[k+1]) or (x[-1] <= t[n-k-2]):
        raise ValueError(mesg)

    m = x.shape[0]
    l = k+1
    nk3 = n - k - 3
    if nk3 < 2:
        return
    for j in range(1, nk3+1):
        tj = t[j]
        l += 1
        tl = t[l]
        i = np.argmax(x > tj)
        if i >= m-1:
            raise ValueError(mesg)
        if x[i] >= tl:
            raise ValueError(mesg)
    return
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.interpolate` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = ["Rbf"]  # noqa: F822


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="interpolate", module="rbf",
                                   private_modules=["_rbf"], all=__all__,
                                   attribute=name)
<|endoftext|>"""
fitpack (dierckx in netlib) --- A Python-C wrapper to FITPACK (by P. Dierckx).
        FITPACK is a collection of FORTRAN programs for curve and surface
        fitting with splines and tensor product splines.

See
 https://web.archive.org/web/20010524124604/http://www.cs.kuleuven.ac.be:80/cwis/research/nalag/research/topics/fitpack.html
or
 http://www.netlib.org/dierckx/

Copyright 2002 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@cens.ioc.ee>
Permission to use, modify, and distribute this software is given under the
terms of the SciPy (BSD style) license. See LICENSE.txt that came with
this distribution for specifics.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.

TODO: Make interfaces to the following fitpack functions:
    For univariate splines: cocosp, concon, fourco, insert
    For bivariate splines: profil, regrid, parsur, surev
"""

__all__ = ['splrep', 'splprep', 'splev', 'splint', 'sproot', 'spalde',
           'bisplrep', 'bisplev', 'insert', 'splder', 'splantider']

import warnings
import numpy as np
from . import _fitpack
from numpy import (atleast_1d, array, ones, zeros, sqrt, ravel, transpose,
                   empty, iinfo, asarray)

# Try to replace _fitpack interface with
#  f2py-generated version
from . import _dfitpack as dfitpack


dfitpack_int = dfitpack.types.intvar.dtype


def _int_overflow(x, exception, msg=None):
    """Cast the value to an dfitpack_int and raise an OverflowError if the value
    cannot fit.
    """
    if x > iinfo(dfitpack_int).max:
        if msg is None:
            msg = f'{x!r} cannot fit into an {dfitpack_int!r}'
        raise exception(msg)
    return dfitpack_int.type(x)


_iermess = {
    0: ["The spline has a residual sum of squares fp such that "
        "abs(fp-s)/s<=0.001", None],
    -1: ["The spline is an interpolating spline (fp=0)", None],
    -2: ["The spline is weighted least-squares polynomial of degree k.\n"
         "fp gives the upper bound fp0 for the smoothing factor s", None],
    1: ["The required storage space exceeds the available storage space.\n"
        "Probable causes: data (x,y) size is too small or smoothing parameter"
        "\ns is too small (fp>s).", ValueError],
    2: ["A theoretically impossible result when finding a smoothing spline\n"
        "with fp = s. Probable cause: s too small. (abs(fp-s)/s>0.001)",
        ValueError],
    3: ["The maximal number of iterations (20) allowed for finding smoothing\n"
        "spline with fp=s has been reached. Probable cause: s too small.\n"
        "(abs(fp-s)/s>0.001)", ValueError],
    10: ["Error on input data", ValueError],
    'unknown': ["An error occurred", TypeError]
}

_iermess2 = {
    0: ["The spline has a residual sum of squares fp such that "
        "abs(fp-s)/s<=0.001", None],
    -1: ["The spline is an interpolating spline (fp=0)", None],
    -2: ["The spline is weighted least-squares polynomial of degree kx and ky."
         "\nfp gives the upper bound fp0 for the smoothing factor s", None],
    -3: ["Warning. The coefficients of the spline have been computed as the\n"
         "minimal norm least-squares solution of a rank deficient system.",
         None],
    1: ["The required storage space exceeds the available storage space.\n"
        "Probable causes: nxest or nyest too small or s is too small. (fp>s)",
        ValueError],
    2: ["A theoretically impossible result when finding a smoothing spline\n"
        "with fp = s. Probable causes: s too small or badly chosen eps.\n"
        "(abs(fp-s)/s>0.001)", ValueError],
    3: ["The maximal number of iterations (20) allowed for finding smoothing\n"
        "spline with fp=s has been reached. Probable cause: s too small.\n"
        "(abs(fp-s)/s>0.001)", ValueError],
    4: ["No more knots can be added because the number of B-spline\n"
        "coefficients already exceeds the number of data points m.\n"
        "Probable causes: either s or m too small. (fp>s)", ValueError],
    5: ["No more knots can be added because the additional knot would\n"
        "coincide with an old one. Probable cause: s too small or too large\n"
        "a weight to an inaccurate data point. (fp>s)", ValueError],
    10: ["Error on input data", ValueError],
    11: ["rwrk2 too small, i.e., there is not enough workspace for computing\n"
         "the minimal least-squares solution of a rank deficient system of\n"
         "linear equations.", ValueError],
    'unknown': ["An error occurred", TypeError]
}

_parcur_cache = {'t': array([], float), 'wrk': array([], float),
                 'iwrk': array([], dfitpack_int), 'u': array([], float),
                 'ub': 0, 'ue': 1}


def splprep(x, w=None, u=None, ub=None, ue=None, k=3, task=0, s=None, t=None,
            full_output=0, nest=None, per=0, quiet=1):
    # see the docstring of `_fitpack_py/splprep`
    if task <= 0:
        _parcur_cache = {'t': array([], float), 'wrk': array([], float),
                         'iwrk': array([], dfitpack_int), 'u': array([], float),
                         'ub': 0, 'ue': 1}
    x = atleast_1d(x)
    idim, m = x.shape
    if per:
        for i in range(idim):
            if x[i][0] != x[i][-1]:
                if not quiet:
                    warnings.warn(RuntimeWarning('Setting x[%d][%d]=x[%d][0]' %
                                                 (i, m, i)),
                                  stacklevel=2)
                x[i][-1] = x[i][0]
    if not 0 < idim < 11:
        raise TypeError('0 < idim < 11 must hold')
    if w is None:
        w = ones(m, float)
    else:
        w = atleast_1d(w)
    ipar = (u is not None)
    if ipar:
        _parcur_cache['u'] = u
        if ub is None:
            _parcur_cache['ub'] = u[0]
        else:
            _parcur_cache['ub'] = ub
        if ue is None:
            _parcur_cache['ue'] = u[-1]
        else:
            _parcur_cache['ue'] = ue
    else:
        _parcur_cache['u'] = zeros(m, float)
    if not (1 <= k <= 5):
        raise TypeError('1 <= k= %d <=5 must hold' % k)
    if not (-1 <= task <= 1):
        raise TypeError('task must be -1, 0 or 1')
    if (not len(w) == m) or (ipar == 1 and (not len(u) == m)):
        raise TypeError('Mismatch of input dimensions')
    if s is None:
        s = m - sqrt(2*m)
    if t is None and task == -1:
        raise TypeError('Knots must be given for task=-1')
    if t is not None:
        _parcur_cache['t'] = atleast_1d(t)
    n = len(_parcur_cache['t'])
    if task == -1 and n < 2*k + 2:
        raise TypeError('There must be at least 2*k+2 knots for task=-1')
    if m <= k:
        raise TypeError('m > k must hold')
    if nest is None:
        nest = m + 2*k

    if (task >= 0 and s == 0) or (nest < 0):
        if per:
            nest = m + 2*k
        else:
            nest = m + k + 1
    nest = max(nest, 2*k + 3)
    u = _parcur_cache['u']
    ub = _parcur_cache['ub']
    ue = _parcur_cache['ue']
    t = _parcur_cache['t']
    wrk = _parcur_cache['wrk']
    iwrk = _parcur_cache['iwrk']
    t, c, o = _fitpack._parcur(ravel(transpose(x)), w, u, ub, ue, k,
                               task, ipar, s, t, nest, wrk, iwrk, per)
    _parcur_cache['u'] = o['u']
    _parcur_cache['ub'] = o['ub']
    _parcur_cache['ue'] = o['ue']
    _parcur_cache['t'] = t
    _parcur_cache['wrk'] = o['wrk']
    _parcur_cache['iwrk'] = o['iwrk']
    ier = o['ier']
    fp = o['fp']
    n = len(t)
    u = o['u']
    c.shape = idim, n - k - 1
    tcku = [t, list(c), k], u
    if ier <= 0 and not quiet:
        warnings.warn(RuntimeWarning(_iermess[ier][0] +
                                     "\tk=%d n=%d m=%d fp=%f s=%f" %
                                     (k, len(t), m, fp, s)),
                      stacklevel=2)
    if ier > 0 and not full_output:
        if ier in [1, 2, 3]:
            warnings.warn(RuntimeWarning(_iermess[ier][0]), stacklevel=2)
        else:
            try:
                raise _iermess[ier][1](_iermess[ier][0])
            except KeyError as e:
                raise _iermess['unknown'][1](_iermess['unknown'][0]) from e
    if full_output:
        try:
            return tcku, fp, ier, _iermess[ier][0]
        except KeyError:
            return tcku, fp, ier, _iermess['unknown'][0]
    else:
        return tcku


_curfit_cache = {'t': array([], float), 'wrk': array([], float),
                 'iwrk': array([], dfitpack_int)}


def splrep(x, y, w=None, xb=None, xe=None, k=3, task=0, s=None, t=None,
           full_output=0, per=0, quiet=1):
    # see the docstring of `_fitpack_py/splrep`
    if task <= 0:
        _curfit_cache = {}
    x, y = map(atleast_1d, [x, y])
    m = len(x)
    if w is None:
        w = ones(m, float)
        if s is None:
            s = 0.0
    else:
        w = atleast_1d(w)
        if s is None:
            s = m - sqrt(2*m)
    if not len(w) == m:
        raise TypeError('len(w)=%d is not equal to m=%d' % (len(w), m))
    if (m != len(y)) or (m != len(w)):
        raise TypeError('Lengths of the first three arguments (x,y,w) must '
                        'be equal')
    if not (1 <= k <= 5):
        raise TypeError('Given degree of the spline (k=%d) is not supported. '
                        '(1<=k<=5)' % k)
    if m <= k:
        raise TypeError('m > k must hold')
    if xb is None:
        xb = x[0]
    if xe is None:
        xe = x[-1]
    if not (-1 <= task <= 1):
        raise TypeError('task must be -1, 0 or 1')
    if t is not None:
        task = -1
    if task == -1:
        if t is None:
            raise TypeError('Knots must be given for task=-1')
        numknots = len(t)
        _curfit_cache['t'] = empty((numknots + 2*k + 2,), float)
        _curfit_cache['t'][k+1:-k-1] = t
        nest = len(_curfit_cache['t'])
    elif task == 0:
        if per:
            nest = max(m + 2*k, 2*k + 3)
        else:
            nest = max(m + k + 1, 2*k + 3)
        t = empty((nest,), float)
        _curfit_cache['t'] = t
    if task <= 0:
        if per:
            _curfit_cache['wrk'] = empty((m*(k + 1) + nest*(8 + 5*k),), float)
        else:
            _curfit_cache['wrk'] = empty((m*(k + 1) + nest*(7 + 3*k),), float)
        _curfit_cache['iwrk'] = empty((nest,), dfitpack_int)
    try:
        t = _curfit_cache['t']
        wrk = _curfit_cache['wrk']
        iwrk = _curfit_cache['iwrk']
    except KeyError as e:
        raise TypeError("must call with task=1 only after"
                        " call with task=0,-1") from e
    if not per:
        n, c, fp, ier = dfitpack.curfit(task, x, y, w, t, wrk, iwrk,
                                        xb, xe, k, s)
    else:
        n, c, fp, ier = dfitpack.percur(task, x, y, w, t, wrk, iwrk, k, s)
    tck = (t[:n], c[:n], k)
    if ier <= 0 and not quiet:
        _mess = (_iermess[ier][0] + "\tk=%d n=%d m=%d fp=%f s=%f" %
                 (k, len(t), m, fp, s))
        warnings.warn(RuntimeWarning(_mess), stacklevel=2)
    if ier > 0 and not full_output:
        if ier in [1, 2, 3]:
            warnings.warn(RuntimeWarning(_iermess[ier][0]), stacklevel=2)
        else:
            try:
                raise _iermess[ier][1](_iermess[ier][0])
            except KeyError as e:
                raise _iermess['unknown'][1](_iermess['unknown'][0]) from e
    if full_output:
        try:
            return tck, fp, ier, _iermess[ier][0]
        except KeyError:
            return tck, fp, ier, _iermess['unknown'][0]
    else:
        return tck


def splev(x, tck, der=0, ext=0):
    # see the docstring of `_fitpack_py/splev`
    t, c, k = tck
    try:
        c[0][0]
        parametric = True
    except Exception:
        parametric = False
    if parametric:
        return list(map(lambda c, x=x, t=t, k=k, der=der:
                        splev(x, [t, c, k], der, ext), c))
    else:
        if not (0 <= der <= k):
            raise ValueError("0<=der=%d<=k=%d must hold" % (der, k))
        if ext not in (0, 1, 2, 3):
            raise ValueError(f"ext = {ext} not in (0, 1, 2, 3) ")

        x = asarray(x)
        shape = x.shape
        x = atleast_1d(x).ravel()
        if der == 0:
            y, ier = dfitpack.splev(t, c, k, x, ext)
        else:
            y, ier = dfitpack.splder(t, c, k, x, der, ext)

        if ier == 10:
            raise ValueError("Invalid input data")
        if ier == 1:
            raise ValueError("Found x value not in the domain")
        if ier:
            raise TypeError("An error occurred")

        return y.reshape(shape)


def splint(a, b, tck, full_output=0):
    # see the docstring of `_fitpack_py/splint`
    t, c, k = tck
    try:
        c[0][0]
        parametric = True
    except Exception:
        parametric = False
    if parametric:
        return list(map(lambda c, a=a, b=b, t=t, k=k:
                        splint(a, b, [t, c, k]), c))
    else:
        aint, wrk = dfitpack.splint(t, c, k, a, b)
        if full_output:
            return aint, wrk
        else:
            return aint


def sproot(tck, mest=10):
    # see the docstring of `_fitpack_py/sproot`
    t, c, k = tck
    if k != 3:
        raise ValueError("sproot works only for cubic (k=3) splines")
    try:
        c[0][0]
        parametric = True
    except Exception:
        parametric = False
    if parametric:
        return list(map(lambda c, t=t, k=k, mest=mest:
                        sproot([t, c, k], mest), c))
    else:
        if len(t) < 8:
            raise TypeError(f"The number of knots {len(t)}>=8")
        z, m, ier = dfitpack.sproot(t, c, mest)
        if ier == 10:
            raise TypeError("Invalid input data. "
                            "t1<=..<=t4<t5<..<tn-3<=..<=tn must hold.")
        if ier == 0:
            return z[:m]
        if ier == 1:
            warnings.warn(RuntimeWarning("The number of zeros exceeds mest"),
                          stacklevel=2)
            return z[:m]
        raise TypeError("Unknown error")


def spalde(x, tck):
    # see the docstring of `_fitpack_py/spalde`
    t, c, k = tck
    try:
        c[0][0]
        parametric = True
    except Exception:
        parametric = False
    if parametric:
        return list(map(lambda c, x=x, t=t, k=k:
                        spalde(x, [t, c, k]), c))
    else:
        x = atleast_1d(x)
        if len(x) > 1:
            return list(map(lambda x, tck=tck: spalde(x, tck), x))
        d, ier = dfitpack.spalde(t, c, k+1, x[0])
        if ier == 0:
            return d
        if ier == 10:
            raise TypeError("Invalid input data. t(k)<=x<=t(n-k+1) must hold.")
        raise TypeError("Unknown error")

# def _curfit(x,y,w=None,xb=None,xe=None,k=3,task=0,s=None,t=None,
#           full_output=0,nest=None,per=0,quiet=1):


_surfit_cache = {'tx': array([], float), 'ty': array([], float),
                 'wrk': array([], float), 'iwrk': array([], dfitpack_int)}


def bisplrep(x, y, z, w=None, xb=None, xe=None, yb=None, ye=None,
             kx=3, ky=3, task=0, s=None, eps=1e-16, tx=None, ty=None,
             full_output=0, nxest=None, nyest=None, quiet=1):
    """
    Find a bivariate B-spline representation of a surface.

    Given a set of data points (x[i], y[i], z[i]) representing a surface
    z=f(x,y), compute a B-spline representation of the surface. Based on
    the routine SURFIT from FITPACK.

    Parameters
    ----------
    x, y, z : ndarray
        Rank-1 arrays of data points.
    w : ndarray, optional
        Rank-1 array of weights. By default ``w=np.ones(len(x))``.
    xb, xe : float, optional
        End points of approximation interval in `x`.
        By default ``xb = x.min(), xe=x.max()``.
    yb, ye : float, optional
        End points of approximation interval in `y`.
        By default ``yb=y.min(), ye = y.max()``.
    kx, ky : int, optional
        The degrees of the spline (1 <= kx, ky <= 5).
        Third order (kx=ky=3) is recommended.
    task : int, optional
        If task=0, find knots in x and y and coefficients for a given
        smoothing factor, s.
        If task=1, find knots and coefficients for another value of the
        smoothing factor, s.  bisplrep must have been previously called
        with task=0 or task=1.
        If task=-1, find coefficients for a given set of knots tx, ty.
    s : float, optional
        A non-negative smoothing factor. If weights correspond
        to the inverse of the standard-deviation of the errors in z,
