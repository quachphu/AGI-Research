           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    if (floor(nt) != nt) or (nt <= 0) or not isscalar(nt):
        raise ValueError("Argument must be positive scalar integer.")
    return _specfun.fcszo(1, nt)


def fresnels_zeros(nt):
    """Compute nt complex zeros of sine Fresnel integral S(z).

    Parameters
    ----------
    nt : int
        Number of zeros to compute

    Returns
    -------
    fresnels_zeros: ndarray
        Zeros of the sine Fresnel integral

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    if (floor(nt) != nt) or (nt <= 0) or not isscalar(nt):
        raise ValueError("Argument must be positive scalar integer.")
    return _specfun.fcszo(2, nt)


def fresnel_zeros(nt):
    """Compute nt complex zeros of sine and cosine Fresnel integrals S(z) and C(z).

    Parameters
    ----------
    nt : int
        Number of zeros to compute

    Returns
    -------
    zeros_sine: ndarray
        Zeros of the sine Fresnel integral
    zeros_cosine : ndarray
        Zeros of the cosine Fresnel integral

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    if (floor(nt) != nt) or (nt <= 0) or not isscalar(nt):
        raise ValueError("Argument must be positive scalar integer.")
    return _specfun.fcszo(2, nt), _specfun.fcszo(1, nt)


def assoc_laguerre(x, n, k=0.0):
    """Compute the generalized (associated) Laguerre polynomial of degree n and order k.

    The polynomial :math:`L^{(k)}_n(x)` is orthogonal over ``[0, inf)``,
    with weighting function ``exp(-x) * x**k`` with ``k > -1``.

    Parameters
    ----------
    x : float or ndarray
        Points where to evaluate the Laguerre polynomial
    n : int
        Degree of the Laguerre polynomial
    k : int
        Order of the Laguerre polynomial

    Returns
    -------
    assoc_laguerre: float or ndarray
        Associated laguerre polynomial values

    Notes
    -----
    `assoc_laguerre` is a simple wrapper around `eval_genlaguerre`, with
    reversed argument order ``(x, n, k=0.0) --> (n, k, x)``.

    """
    return _ufuncs.eval_genlaguerre(n, k, x)


digamma = psi


def polygamma(n, x):
    r"""Polygamma functions.

    Defined as :math:`\psi^{(n)}(x)` where :math:`\psi` is the
    `digamma` function. See [dlmf]_ for details.

    Parameters
    ----------
    n : array_like
        The order of the derivative of the digamma function; must be
        integral
    x : array_like
        Real valued input

    Returns
    -------
    ndarray
        Function results

    See Also
    --------
    digamma

    References
    ----------
    .. [dlmf] NIST, Digital Library of Mathematical Functions,
        https://dlmf.nist.gov/5.15

    Examples
    --------
    >>> from scipy import special
    >>> x = [2, 3, 25.5]
    >>> special.polygamma(1, x)
    array([ 0.64493407,  0.39493407,  0.03999467])
    >>> special.polygamma(0, x) == special.psi(x)
    array([ True,  True,  True], dtype=bool)

    """
    n, x = asarray(n), asarray(x)
    fac2 = (-1.0)**(n+1) * gamma(n+1.0) * zeta(n+1, x)
    return where(n == 0, psi(x), fac2)


def mathieu_even_coef(m, q):
    r"""Fourier coefficients for even Mathieu and modified Mathieu functions.

    The Fourier series of the even solutions of the Mathieu differential
    equation are of the form

    .. math:: \mathrm{ce}_{2n}(z, q) = \sum_{k=0}^{\infty} A_{(2n)}^{(2k)} \cos 2kz

    .. math:: \mathrm{ce}_{2n+1}(z, q) =
              \sum_{k=0}^{\infty} A_{(2n+1)}^{(2k+1)} \cos (2k+1)z

    This function returns the coefficients :math:`A_{(2n)}^{(2k)}` for even
    input m=2n, and the coefficients :math:`A_{(2n+1)}^{(2k+1)}` for odd input
    m=2n+1.

    Parameters
    ----------
    m : int
        Order of Mathieu functions.  Must be non-negative.
    q : float (>=0)
        Parameter of Mathieu functions.  Must be non-negative.

    Returns
    -------
    Ak : ndarray
        Even or odd Fourier coefficients, corresponding to even or odd m.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html
    .. [2] NIST Digital Library of Mathematical Functions
           https://dlmf.nist.gov/28.4#i

    """
    if not (isscalar(m) and isscalar(q)):
        raise ValueError("m and q must be scalars.")
    if (q < 0):
        raise ValueError("q >=0")
    if (m != floor(m)) or (m < 0):
        raise ValueError("m must be an integer >=0.")

    if (q <= 1):
        qm = 7.5 + 56.1*sqrt(q) - 134.7*q + 90.7*sqrt(q)*q
    else:
        qm = 17.0 + 3.1*sqrt(q) - .126*q + .0037*sqrt(q)*q
    km = int(qm + 0.5*m)
    if km > 251:
        warnings.warn("Too many predicted coefficients.", RuntimeWarning, stacklevel=2)
    kd = 1
    m = int(floor(m))
    if m % 2:
        kd = 2

    a = mathieu_a(m, q)
    fc = _specfun.fcoef(kd, m, q, a)
    return fc[:km]


def mathieu_odd_coef(m, q):
    r"""Fourier coefficients for even Mathieu and modified Mathieu functions.

    The Fourier series of the odd solutions of the Mathieu differential
    equation are of the form

    .. math:: \mathrm{se}_{2n+1}(z, q) =
              \sum_{k=0}^{\infty} B_{(2n+1)}^{(2k+1)} \sin (2k+1)z

    .. math:: \mathrm{se}_{2n+2}(z, q) =
              \sum_{k=0}^{\infty} B_{(2n+2)}^{(2k+2)} \sin (2k+2)z

    This function returns the coefficients :math:`B_{(2n+2)}^{(2k+2)}` for even
    input m=2n+2, and the coefficients :math:`B_{(2n+1)}^{(2k+1)}` for odd
    input m=2n+1.

    Parameters
    ----------
    m : int
        Order of Mathieu functions.  Must be non-negative.
    q : float (>=0)
        Parameter of Mathieu functions.  Must be non-negative.

    Returns
    -------
    Bk : ndarray
        Even or odd Fourier coefficients, corresponding to even or odd m.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    if not (isscalar(m) and isscalar(q)):
        raise ValueError("m and q must be scalars.")
    if (q < 0):
        raise ValueError("q >=0")
    if (m != floor(m)) or (m <= 0):
        raise ValueError("m must be an integer > 0")

    if (q <= 1):
        qm = 7.5 + 56.1*sqrt(q) - 134.7*q + 90.7*sqrt(q)*q
    else:
        qm = 17.0 + 3.1*sqrt(q) - .126*q + .0037*sqrt(q)*q
    km = int(qm + 0.5*m)
    if km > 251:
        warnings.warn("Too many predicted coefficients.", RuntimeWarning, stacklevel=2)
    kd = 4
    m = int(floor(m))
    if m % 2:
        kd = 3

    b = mathieu_b(m, q)
    fc = _specfun.fcoef(kd, m, q, b)
    return fc[:km]


@_deprecated(__DEPRECATION_MSG_1_15.format("lpmn", "assoc_legendre_p_all"))
def lpmn(m, n, z):
    """Sequence of associated Legendre functions of the first kind.

    Computes the associated Legendre function of the first kind of order m and
    degree n, ``Pmn(z)`` = :math:`P_n^m(z)`, and its derivative, ``Pmn'(z)``.
    Returns two arrays of size ``(m+1, n+1)`` containing ``Pmn(z)`` and
    ``Pmn'(z)`` for all orders from ``0..m`` and degrees from ``0..n``.

    This function takes a real argument ``z``. For complex arguments ``z``
    use clpmn instead.

    .. deprecated:: 1.15.0
        This function is deprecated and will be removed in SciPy 1.17.0.
        Please `scipy.special.assoc_legendre_p_all` instead.

    Parameters
    ----------
    m : int
       ``|m| <= n``; the order of the Legendre function.
    n : int
       where ``n >= 0``; the degree of the Legendre function.  Often
       called ``l`` (lower case L) in descriptions of the associated
       Legendre function
    z : array_like
        Input value.

    Returns
    -------
    Pmn_z : (m+1, n+1) array
       Values for all orders 0..m and degrees 0..n
    Pmn_d_z : (m+1, n+1) array
       Derivatives for all orders 0..m and degrees 0..n

    See Also
    --------
    clpmn: associated Legendre functions of the first kind for complex z

    Notes
    -----
    In the interval (-1, 1), Ferrer's function of the first kind is
    returned. The phase convention used for the intervals (1, inf)
    and (-inf, -1) is such that the result is always real.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html
    .. [2] NIST Digital Library of Mathematical Functions
           https://dlmf.nist.gov/14.3

    """

    n = _nonneg_int_or_fail(n, 'n', strict=False)

    if (abs(m) > n):
        raise ValueError("m must be <= n.")

    if np.iscomplexobj(z):
        raise ValueError("Argument must be real. Use clpmn instead.")

    m, n = int(m), int(n)  # Convert to int to maintain backwards compatibility.

    branch_cut = np.where(np.abs(z) <= 1, 2, 3)

    p, pd = assoc_legendre_p_all(n, abs(m), z, branch_cut=branch_cut, diff_n=1)
    p = np.swapaxes(p, 0, 1)
    pd = np.swapaxes(pd, 0, 1)

    if (m >= 0):
        p = p[:(m + 1)]
        pd = pd[:(m + 1)]
    else:
        p = np.insert(p[:(m - 1):-1], 0, p[0], axis=0)
        pd = np.insert(pd[:(m - 1):-1], 0, pd[0], axis=0)

    return p, pd


@_deprecated(__DEPRECATION_MSG_1_15.format("clpmn", "assoc_legendre_p_all"))
def clpmn(m, n, z, type=3):
    """Associated Legendre function of the first kind for complex arguments.

    Computes the associated Legendre function of the first kind of order m and
    degree n, ``Pmn(z)`` = :math:`P_n^m(z)`, and its derivative, ``Pmn'(z)``.
    Returns two arrays of size ``(m+1, n+1)`` containing ``Pmn(z)`` and
    ``Pmn'(z)`` for all orders from ``0..m`` and degrees from ``0..n``.

    .. deprecated:: 1.15.0
        This function is deprecated and will be removed in SciPy 1.17.0.
        Please use `scipy.special.assoc_legendre_p_all` instead.

    Parameters
    ----------
    m : int
       ``|m| <= n``; the order of the Legendre function.
    n : int
       where ``n >= 0``; the degree of the Legendre function.  Often
       called ``l`` (lower case L) in descriptions of the associated
       Legendre function
    z : array_like, float or complex
        Input value.
    type : int, optional
       takes values 2 or 3
       2: cut on the real axis ``|x| > 1``
       3: cut on the real axis ``-1 < x < 1`` (default)

    Returns
    -------
    Pmn_z : (m+1, n+1) array
       Values for all orders ``0..m`` and degrees ``0..n``
    Pmn_d_z : (m+1, n+1) array
       Derivatives for all orders ``0..m`` and degrees ``0..n``

    See Also
    --------
    lpmn: associated Legendre functions of the first kind for real z

    Notes
    -----
    By default, i.e. for ``type=3``, phase conventions are chosen according
    to [1]_ such that the function is analytic. The cut lies on the interval
    (-1, 1). Approaching the cut from above or below in general yields a phase
    factor with respect to Ferrer's function of the first kind
    (cf. `lpmn`).

    For ``type=2`` a cut at ``|x| > 1`` is chosen. Approaching the real values
    on the interval (-1, 1) in the complex plane yields Ferrer's function
    of the first kind.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html
    .. [2] NIST Digital Library of Mathematical Functions
           https://dlmf.nist.gov/14.21

    """

    if (abs(m) > n):
        raise ValueError("m must be <= n.")

    if not (type == 2 or type == 3):
        raise ValueError("type must be either 2 or 3.")

    m, n = int(m), int(n)  # Convert to int to maintain backwards compatibility.

    if not np.iscomplexobj(z):
        z = np.asarray(z, dtype=complex)

    out, out_jac = assoc_legendre_p_all(n, abs(m), z, branch_cut=type, diff_n=1)
    out = np.swapaxes(out, 0, 1)
    out_jac = np.swapaxes(out_jac, 0, 1)

    if (m >= 0):
        out = out[:(m + 1)]
        out_jac = out_jac[:(m + 1)]
    else:
        out = np.insert(out[:(m - 1):-1], 0, out[0], axis=0)
        out_jac = np.insert(out_jac[:(m - 1):-1], 0, out_jac[0], axis=0)

    return out, out_jac


def lqmn(m, n, z):
    """Sequence of associated Legendre functions of the second kind.

    Computes the associated Legendre function of the second kind of order m and
    degree n, ``Qmn(z)`` = :math:`Q_n^m(z)`, and its derivative, ``Qmn'(z)``.
    Returns two arrays of size ``(m+1, n+1)`` containing ``Qmn(z)`` and
    ``Qmn'(z)`` for all orders from ``0..m`` and degrees from ``0..n``.

    Parameters
    ----------
    m : int
       ``|m| <= n``; the order of the Legendre function.
    n : int
       where ``n >= 0``; the degree of the Legendre function.  Often
       called ``l`` (lower case L) in descriptions of the associated
       Legendre function
    z : array_like, complex
        Input value.

    Returns
    -------
    Qmn_z : (m+1, n+1) array
       Values for all orders 0..m and degrees 0..n
    Qmn_d_z : (m+1, n+1) array
       Derivatives for all orders 0..m and degrees 0..n

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    if not isscalar(m) or (m < 0):
        raise ValueError("m must be a non-negative integer.")
    if not isscalar(n) or (n < 0):
        raise ValueError("n must be a non-negative integer.")

    m, n = int(m), int(n)  # Convert to int to maintain backwards compatibility.
    # Ensure neither m nor n == 0
    mm = max(1, m)
    nn = max(1, n)

    z = np.asarray(z)
    if (not np.issubdtype(z.dtype, np.inexact)):
        z = z.astype(np.float64)

    if np.iscomplexobj(z):
        q = np.empty((mm + 1, nn + 1) + z.shape, dtype=np.complex128)
    else:
        q = np.empty((mm + 1, nn + 1) + z.shape, dtype=np.float64)
    qd = np.empty_like(q)
    if (z.ndim == 0):
        _lqmn(z, out=(q, qd))
    else:
        # new axes must be last for the ufunc
        _lqmn(z,
              out=(np.moveaxis(q, (0, 1), (-2, -1)),
                   np.moveaxis(qd, (0, 1), (-2, -1))))

    return q[:(m+1), :(n+1)], qd[:(m+1), :(n+1)]


def bernoulli(n):
    """Bernoulli numbers B0..Bn (inclusive).

    Parameters
    ----------
    n : int
        Indicated the number of terms in the Bernoulli series to generate.

    Returns
    -------
    ndarray
        The Bernoulli numbers ``[B(0), B(1), ..., B(n)]``.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html
    .. [2] "Bernoulli number", Wikipedia, https://en.wikipedia.org/wiki/Bernoulli_number

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import bernoulli, zeta
    >>> bernoulli(4)
    array([ 1.        , -0.5       ,  0.16666667,  0.        , -0.03333333])

    The Wikipedia article ([2]_) points out the relationship between the
    Bernoulli numbers and the zeta function, ``B_n^+ = -n * zeta(1 - n)``
    for ``n > 0``:

    >>> n = np.arange(1, 5)
    >>> -n * zeta(1 - n)
    array([ 0.5       ,  0.16666667, -0.        , -0.03333333])

    Note that, in the notation used in the wikipedia article,
    `bernoulli` computes ``B_n^-`` (i.e. it used the convention that
    ``B_1`` is -1/2).  The relation given above is for ``B_n^+``, so the
    sign of 0.5 does not match the output of ``bernoulli(4)``.

    """
    if not isscalar(n) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    n = int(n)
    if (n < 2):
        n1 = 2
    else:
        n1 = n
    return _specfun.bernob(int(n1))[:(n+1)]


def euler(n):
    """Euler numbers E(0), E(1), ..., E(n).

    The Euler numbers [1]_ are also known as the secant numbers.

    Because ``euler(n)`` returns floating point values, it does not give
    exact values for large `n`.  The first inexact value is E(22).

    Parameters
    ----------
    n : int
        The highest index of the Euler number to be returned.

    Returns
    -------
    ndarray
        The Euler numbers [E(0), E(1), ..., E(n)].
        The odd Euler numbers, which are all zero, are included.

    References
    ----------
    .. [1] Sequence A122045, The On-Line Encyclopedia of Integer Sequences,
           https://oeis.org/A122045
    .. [2] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import euler
    >>> euler(6)
    array([  1.,   0.,  -1.,   0.,   5.,   0., -61.])

    >>> euler(13).astype(np.int64)
    array([      1,       0,      -1,       0,       5,       0,     -61,
                 0,    1385,       0,  -50521,       0, 2702765,       0])

    >>> euler(22)[-1]  # Exact value of E(22) is -69348874393137901.
    -69348874393137976.0

    """
    if not isscalar(n) or (n < 0):
        raise ValueError("n must be a non-negative integer.")
    n = int(n)
    if (n < 2):
        n1 = 2
    else:
        n1 = n
    return _specfun.eulerb(n1)[:(n+1)]


@_deprecated(__DEPRECATION_MSG_1_15.format("lpn", "legendre_p_all"))
def lpn(n, z):
    """Legendre function of the first kind.

    Compute sequence of Legendre functions of the first kind (polynomials),
    Pn(z) and derivatives for all degrees from 0 to n (inclusive).

    See also special.legendre for polynomial class.

    .. deprecated:: 1.15.0
        This function is deprecated and will be removed in SciPy 1.17.0.
        Please use `scipy.special.legendre_p_all` instead.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html
    """

    return legendre_p_all(n, z, diff_n=1)


def lqn(n, z):
    """Legendre function of the second kind.

    Compute sequence of Legendre functions of the second kind, Qn(z) and
    derivatives for all degrees from 0 to n (inclusive).

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    n = _nonneg_int_or_fail(n, 'n', strict=False)
    if (n < 1):
        n1 = 1
    else:
        n1 = n

    z = np.asarray(z)
    if (not np.issubdtype(z.dtype, np.inexact)):
        z = z.astype(float)

    if np.iscomplexobj(z):
        qn = np.empty((n1 + 1,) + z.shape, dtype=np.complex128)
    else:
        qn = np.empty((n1 + 1,) + z.shape, dtype=np.float64)
    qd = np.empty_like(qn)
    if (z.ndim == 0):
        _lqn(z, out=(qn, qd))
    else:
          # new axes must be last for the ufunc
        _lqn(z,
             out=(np.moveaxis(qn, 0, -1),
                  np.moveaxis(qd, 0, -1)))

    return qn[:(n+1)], qd[:(n+1)]


def ai_zeros(nt):
    """
    Compute `nt` zeros and values of the Airy function Ai and its derivative.

    Computes the first `nt` zeros, `a`, of the Airy function Ai(x);
    first `nt` zeros, `ap`, of the derivative of the Airy function Ai'(x);
    the corresponding values Ai(a');
    and the corresponding values Ai'(a).

    Parameters
    ----------
    nt : int
        Number of zeros to compute

    Returns
    -------
    a : ndarray
        First `nt` zeros of Ai(x)
    ap : ndarray
        First `nt` zeros of Ai'(x)
    ai : ndarray
        Values of Ai(x) evaluated at first `nt` zeros of Ai'(x)
    aip : ndarray
        Values of Ai'(x) evaluated at first `nt` zeros of Ai(x)

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    Examples
    --------
    >>> from scipy import special
    >>> a, ap, ai, aip = special.ai_zeros(3)
    >>> a
    array([-2.33810741, -4.08794944, -5.52055983])
    >>> ap
    array([-1.01879297, -3.24819758, -4.82009921])
    >>> ai
    array([ 0.53565666, -0.41901548,  0.38040647])
    >>> aip
    array([ 0.70121082, -0.80311137,  0.86520403])

    """
    kf = 1
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be a positive integer scalar.")
    return _specfun.airyzo(nt, kf)


def bi_zeros(nt):
    """
    Compute `nt` zeros and values of the Airy function Bi and its derivative.

    Computes the first `nt` zeros, b, of the Airy function Bi(x);
    first `nt` zeros, b', of the derivative of the Airy function Bi'(x);
    the corresponding values Bi(b');
    and the corresponding values Bi'(b).

    Parameters
    ----------
    nt : int
        Number of zeros to compute

    Returns
    -------
    b : ndarray
        First `nt` zeros of Bi(x)
    bp : ndarray
        First `nt` zeros of Bi'(x)
    bi : ndarray
        Values of Bi(x) evaluated at first `nt` zeros of Bi'(x)
    bip : ndarray
        Values of Bi'(x) evaluated at first `nt` zeros of Bi(x)

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    Examples
    --------
    >>> from scipy import special
    >>> b, bp, bi, bip = special.bi_zeros(3)
    >>> b
    array([-1.17371322, -3.2710933 , -4.83073784])
    >>> bp
    array([-2.29443968, -4.07315509, -5.51239573])
    >>> bi
    array([-0.45494438,  0.39652284, -0.36796916])
    >>> bip
    array([ 0.60195789, -0.76031014,  0.83699101])

    """
    kf = 2
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be a positive integer scalar.")
    return _specfun.airyzo(nt, kf)


def lmbda(v, x):
    r"""Jahnke-Emden Lambda function, Lambdav(x).

    This function is defined as [2]_,

    .. math:: \Lambda_v(x) = \Gamma(v+1) \frac{J_v(x)}{(x/2)^v},

    where :math:`\Gamma` is the gamma function and :math:`J_v` is the
    Bessel function of the first kind.

    Parameters
    ----------
    v : float
        Order of the Lambda function
    x : float
        Value at which to evaluate the function and derivatives

    Returns
    -------
    vl : ndarray
        Values of Lambda_vi(x), for vi=v-int(v), vi=1+v-int(v), ..., vi=v.
    dl : ndarray
        Derivatives Lambda_vi'(x), for vi=v-int(v), vi=1+v-int(v), ..., vi=v.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html
    .. [2] Jahnke, E. and Emde, F. "Tables of Functions with Formulae and
           Curves" (4th ed.), Dover, 1945
    """
    if not (isscalar(v) and isscalar(x)):
        raise ValueError("arguments must be scalars.")
    if (v < 0):
        raise ValueError("argument must be > 0.")
    n = int(v)
    v0 = v - n
    if (n < 1):
        n1 = 1
    else:
        n1 = n
    v1 = n1 + v0
    if (v != floor(v)):
        vm, vl, dl = _specfun.lamv(v1, x)
    else:
        vm, vl, dl = _specfun.lamn(v1, x)
    return vl[:(n+1)], dl[:(n+1)]


def pbdv_seq(v, x):
    """Parabolic cylinder functions Dv(x) and derivatives.

    Parameters
    ----------
    v : float
        Order of the parabolic cylinder function
    x : float
        Value at which to evaluate the function and derivatives

    Returns
    -------
    dv : ndarray
        Values of D_vi(x), for vi=v-int(v), vi=1+v-int(v), ..., vi=v.
    dp : ndarray
        Derivatives D_vi'(x), for vi=v-int(v), vi=1+v-int(v), ..., vi=v.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 13.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    if not (isscalar(v) and isscalar(x)):
        raise ValueError("arguments must be scalars.")
    n = int(v)
    v0 = v-n
    if (n < 1):
        n1 = 1
    else:
        n1 = n
    v1 = n1 + v0
    dv, dp, pdf, pdd = _specfun.pbdv(v1, x)
    return dv[:n1+1], dp[:n1+1]


def pbvv_seq(v, x):
    """Parabolic cylinder functions Vv(x) and derivatives.

    Parameters
    ----------
    v : float
        Order of the parabolic cylinder function
    x : float
        Value at which to evaluate the function and derivatives

    Returns
    -------
    dv : ndarray
        Values of V_vi(x), for vi=v-int(v), vi=1+v-int(v), ..., vi=v.
    dp : ndarray
        Derivatives V_vi'(x), for vi=v-int(v), vi=1+v-int(v), ..., vi=v.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 13.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    if not (isscalar(v) and isscalar(x)):
        raise ValueError("arguments must be scalars.")
    n = int(v)
    v0 = v-n
    if (n <= 1):
        n1 = 1
    else:
        n1 = n
    v1 = n1 + v0
    dv, dp, pdf, pdd = _specfun.pbvv(v1, x)
    return dv[:n1+1], dp[:n1+1]


def pbdn_seq(n, z):
    """Parabolic cylinder functions Dn(z) and derivatives.

    Parameters
    ----------
    n : int
        Order of the parabolic cylinder function
    z : complex
        Value at which to evaluate the function and derivatives

    Returns
    -------
    dv : ndarray
        Values of D_i(z), for i=0, ..., i=n.
    dp : ndarray
        Derivatives D_i'(z), for i=0, ..., i=n.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 13.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    if not (isscalar(n) and isscalar(z)):
        raise ValueError("arguments must be scalars.")
    if (floor(n) != n):
        raise ValueError("n must be an integer.")
    if (abs(n) <= 1):
        n1 = 1
    else:
        n1 = n
    cpb, cpd = _specfun.cpbdn(n1, z)
    return cpb[:n1+1], cpd[:n1+1]


def ber_zeros(nt):
    """Compute nt zeros of the Kelvin function ber.

    Parameters
    ----------
    nt : int
        Number of zeros to compute. Must be positive.

    Returns
    -------
    ndarray
        First `nt` zeros of the Kelvin function.

    See Also
    --------
    ber

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    return _specfun.klvnzo(nt, 1)


def bei_zeros(nt):
    """Compute nt zeros of the Kelvin function bei.

    Parameters
    ----------
    nt : int
        Number of zeros to compute. Must be positive.

    Returns
    -------
    ndarray
        First `nt` zeros of the Kelvin function.

    See Also
    --------
    bei

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    return _specfun.klvnzo(nt, 2)


def ker_zeros(nt):
    """Compute nt zeros of the Kelvin function ker.

    Parameters
    ----------
    nt : int
        Number of zeros to compute. Must be positive.

    Returns
    -------
    ndarray
        First `nt` zeros of the Kelvin function.

    See Also
    --------
    ker

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    return _specfun.klvnzo(nt, 3)


def kei_zeros(nt):
    """Compute nt zeros of the Kelvin function kei.

    Parameters
    ----------
    nt : int
        Number of zeros to compute. Must be positive.

    Returns
    -------
    ndarray
        First `nt` zeros of the Kelvin function.

    See Also
    --------
    kei

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    return _specfun.klvnzo(nt, 4)


def berp_zeros(nt):
    """Compute nt zeros of the derivative of the Kelvin function ber.

    Parameters
    ----------
    nt : int
        Number of zeros to compute. Must be positive.

    Returns
    -------
    ndarray
        First `nt` zeros of the derivative of the Kelvin function.

    See Also
    --------
    ber, berp

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html


    Examples
    --------
    Compute the first 5 zeros of the derivative of the Kelvin function.

    >>> from scipy.special import berp_zeros
    >>> berp_zeros(5)
    array([ 6.03871081, 10.51364251, 14.96844542, 19.41757493, 23.86430432])

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    return _specfun.klvnzo(nt, 5)


def beip_zeros(nt):
    """Compute nt zeros of the derivative of the Kelvin function bei.

    Parameters
    ----------
    nt : int
        Number of zeros to compute. Must be positive.

    Returns
    -------
    ndarray
        First `nt` zeros of the derivative of the Kelvin function.

    See Also
    --------
    bei, beip

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    return _specfun.klvnzo(nt, 6)


def kerp_zeros(nt):
    """Compute nt zeros of the derivative of the Kelvin function ker.

    Parameters
    ----------
    nt : int
        Number of zeros to compute. Must be positive.

    Returns
    -------
    ndarray
        First `nt` zeros of the derivative of the Kelvin function.

    See Also
    --------
    ker, kerp

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    return _specfun.klvnzo(nt, 7)


def keip_zeros(nt):
    """Compute nt zeros of the derivative of the Kelvin function kei.

    Parameters
    ----------
    nt : int
        Number of zeros to compute. Must be positive.

    Returns
    -------
    ndarray
        First `nt` zeros of the derivative of the Kelvin function.

    See Also
    --------
    kei, keip

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    return _specfun.klvnzo(nt, 8)


def kelvin_zeros(nt):
    """Compute nt zeros of all Kelvin functions.

    Returned in a length-8 tuple of arrays of length nt.  The tuple contains
    the arrays of zeros of (ber, bei, ker, kei, ber', bei', ker', kei').

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("nt must be positive integer scalar.")
    return (_specfun.klvnzo(nt, 1),
            _specfun.klvnzo(nt, 2),
            _specfun.klvnzo(nt, 3),
            _specfun.klvnzo(nt, 4),
            _specfun.klvnzo(nt, 5),
            _specfun.klvnzo(nt, 6),
            _specfun.klvnzo(nt, 7),
            _specfun.klvnzo(nt, 8))


def pro_cv_seq(m, n, c):
    """Characteristic values for prolate spheroidal wave functions.

    Compute a sequence of characteristic values for the prolate
    spheroidal wave functions for mode m and n'=m..n and spheroidal
    parameter c.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    if not (isscalar(m) and isscalar(n) and isscalar(c)):
        raise ValueError("Arguments must be scalars.")
    if (n != floor(n)) or (m != floor(m)):
        raise ValueError("Modes must be integers.")
    if (n-m > 199):
        raise ValueError("Difference between n and m is too large.")
    maxL = n-m+1
    return _specfun.segv(m, n, c, 1)[1][:maxL]


def obl_cv_seq(m, n, c):
    """Characteristic values for oblate spheroidal wave functions.

    Compute a sequence of characteristic values for the oblate
    spheroidal wave functions for mode m and n'=m..n and spheroidal
    parameter c.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    if not (isscalar(m) and isscalar(n) and isscalar(c)):
        raise ValueError("Arguments must be scalars.")
    if (n != floor(n)) or (m != floor(m)):
        raise ValueError("Modes must be integers.")
    if (n-m > 199):
        raise ValueError("Difference between n and m is too large.")
    maxL = n-m+1
    return _specfun.segv(m, n, c, -1)[1][:maxL]


def comb(N, k, *, exact=False, repetition=False):
    """The number of combinations of N things taken k at a time.

    This is often expressed as "N choose k".

    Parameters
    ----------
    N : int, ndarray
        Number of things.
    k : int, ndarray
        Number of elements taken.
    exact : bool, optional
        For integers, if `exact` is False, then floating point precision is
        used, otherwise the result is computed exactly.

        .. deprecated:: 1.14.0
            ``exact=True`` is deprecated for non-integer `N` and `k` and will raise an
            error in SciPy 1.16.0
    repetition : bool, optional
        If `repetition` is True, then the number of combinations with
        repetition is computed.

    Returns
    -------
    val : int, float, ndarray
        The total number of combinations.

    See Also
    --------
    binom : Binomial coefficient considered as a function of two real
            variables.

    Notes
    -----
    - Array arguments accepted only for exact=False case.
    - If N < 0, or k < 0, then 0 is returned.
    - If k > N and repetition=False, then 0 is returned.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import comb
    >>> k = np.array([3, 4])
    >>> n = np.array([10, 10])
    >>> comb(n, k, exact=False)
    array([ 120.,  210.])
    >>> comb(10, 3, exact=True)
    120
    >>> comb(10, 3, exact=True, repetition=True)
    220

    """
    if repetition:
        return comb(N + k - 1, k, exact=exact)
    if exact:
        if int(N) == N and int(k) == k:
            # _comb_int casts inputs to integers, which is safe & intended here
            return _comb_int(N, k)
        # otherwise, we disregard `exact=True`; it makes no sense for
        # non-integral arguments
        msg = ("`exact=True` is deprecated for non-integer `N` and `k` and will raise "
               "an error in SciPy 1.16.0")
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return comb(N, k)
    else:
        k, N = asarray(k), asarray(N)
        cond = (k <= N) & (N >= 0) & (k >= 0)
        vals = binom(N, k)
        if isinstance(vals, np.ndarray):
            vals[~cond] = 0
        elif not cond:
            vals = np.float64(0)
        return vals


def perm(N, k, exact=False):
    """Permutations of N things taken k at a time, i.e., k-permutations of N.

    It's also known as "partial permutations".

    Parameters
    ----------
    N : int, ndarray
        Number of things.
    k : int, ndarray
        Number of elements taken.
    exact : bool, optional
        If ``True``, calculate the answer exactly using long integer arithmetic (`N`
        and `k` must be scalar integers). If ``False``, a floating point approximation
        is calculated (more rapidly) using `poch`. Default is ``False``.

    Returns
    -------
    val : int, ndarray
        The number of k-permutations of N.

    Notes
    -----
    - Array arguments accepted only for exact=False case.
    - If k > N, N < 0, or k < 0, then a 0 is returned.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import perm
    >>> k = np.array([3, 4])
    >>> n = np.array([10, 10])
    >>> perm(n, k)
    array([  720.,  5040.])
    >>> perm(10, 3, exact=True)
    720

    """
    if exact:
        N = np.squeeze(N)[()]  # for backward compatibility (accepted size 1 arrays)
        k = np.squeeze(k)[()]
        if not (isscalar(N) and isscalar(k)):
            raise ValueError("`N` and `k` must scalar integers be with `exact=True`.")

        floor_N, floor_k = int(N), int(k)
        non_integral = not (floor_N == N and floor_k == k)
        if (k > N) or (N < 0) or (k < 0):
            if non_integral:
                msg = ("Non-integer `N` and `k` with `exact=True` is deprecated and "
                       "will raise an error in SciPy 1.16.0.")
                warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return 0
        if non_integral:
            raise ValueError("Non-integer `N` and `k` with `exact=True` is not "
                             "supported.")
        val = 1
        for i in range(floor_N - floor_k + 1, floor_N + 1):
            val *= i
        return val
    else:
        k, N = asarray(k), asarray(N)
        cond = (k <= N) & (N >= 0) & (k >= 0)
        vals = poch(N - k + 1, k)
        if isinstance(vals, np.ndarray):
            vals[~cond] = 0
        elif not cond:
            vals = np.float64(0)
        return vals


# https://stackoverflow.com/a/16327037
def _range_prod(lo, hi, k=1):
    """
    Product of a range of numbers spaced k apart (from hi).

    For k=1, this returns the product of
    lo * (lo+1) * (lo+2) * ... * (hi-2) * (hi-1) * hi
    = hi! / (lo-1)!

    For k>1, it correspond to taking only every k'th number when
    counting down from hi - e.g. 18!!!! = _range_prod(1, 18, 4).

    Breaks into smaller products first for speed:
    _range_prod(2, 9) = ((2*3)*(4*5))*((6*7)*(8*9))
    """
    if lo == 1 and k == 1:
        return math.factorial(hi)

    if lo + k < hi:
        mid = (hi + lo) // 2
        if k > 1:
            # make sure mid is a multiple of k away from hi
            mid = mid - ((mid - hi) % k)
        return _range_prod(lo, mid, k) * _range_prod(mid + k, hi, k)
    elif lo + k == hi:
        return lo * hi
    else:
        return hi


def _factorialx_array_exact(n, k=1):
    """
    Exact computation of factorial for an array.

    The factorials are computed in incremental fashion, by taking
    the sorted unique values of n and multiplying the intervening
    numbers between the different unique values.

    In other words, the factorial for the largest input is only
    computed once, with each other result computed in the process.

    k > 1 corresponds to the multifactorial.
    """
    un = np.unique(n)
    # numpy changed nan-sorting behaviour with 1.21, see numpy/numpy#18070;
    # to unify the behaviour, we remove the nan's here; the respective
    # values will be set separately at the end
    un = un[~np.isnan(un)]

    # Convert to object array if np.int64 can't handle size
    if np.isnan(n).any():
        dt = float
    elif k in _FACTORIALK_LIMITS_64BITS.keys():
        if un[-1] > _FACTORIALK_LIMITS_64BITS[k]:
            # e.g. k=1: 21! > np.iinfo(np.int64).max
            dt = object
        elif un[-1] > _FACTORIALK_LIMITS_32BITS[k]:
            # e.g. k=3: 26!!! > np.iinfo(np.int32).max
            dt = np.int64
        else:
            dt = np.dtype("long")
    else:
        # for k >= 10, we always use object
        dt = object

    out = np.empty_like(n, dtype=dt)

    # Handle invalid/trivial values
    un = un[un > 1]
    out[n < 2] = 1
    out[n < 0] = 0

    # Calculate products of each range of numbers
    # we can only multiply incrementally if the values are k apart;
    # therefore we partition `un` into "lanes", i.e. its residues modulo k
    for lane in range(0, k):
        ul = un[(un % k) == lane] if k > 1 else un
        if ul.size:
            # after np.unique, un resp. ul are sorted, ul[0] is the smallest;
            # cast to python ints to avoid overflow with np.int-types
            val = _range_prod(1, int(ul[0]), k=k)
            out[n == ul[0]] = val
            for i in range(len(ul) - 1):
                # by the filtering above, we have ensured that prev & current
                # are a multiple of k apart
                prev = ul[i]
                current = ul[i + 1]
                # we already multiplied all factors until prev; continue
                # building the full factorial from the following (`prev + 1`);
                # use int() for the same reason as above
                val *= _range_prod(int(prev + 1), int(current), k=k)
                out[n == current] = val

    if np.isnan(n).any():
        out = out.astype(np.float64)
        out[np.isnan(n)] = np.nan
    return out


def _factorialx_array_approx(n, k, extend):
    """
    Calculate approximation to multifactorial for array n and integer k.

    Ensure that values aren't calculated unnecessarily.
    """
    if extend == "complex":
        return _factorialx_approx_core(n, k=k, extend=extend)

    # at this point we are guaranteed that extend='zero' and that k>0 is an integer
    result = zeros(n.shape)
    # keep nans as nans
    place(result, np.isnan(n), np.nan)
    # only compute where n >= 0 (excludes nans), everything else is 0
    cond = (n >= 0)
    n_to_compute = extract(cond, n)
    place(result, cond, _factorialx_approx_core(n_to_compute, k=k, extend=extend))
    return result


def _gamma1p(vals):
    """
    returns gamma(n+1), though with NaN at -1 instead of inf, c.f. #21827
    """
    res = gamma(vals + 1)
    # replace infinities at -1 (from gamma function at 0) with nan
    # gamma only returns inf for real inputs; can ignore complex case
    if isinstance(res, np.ndarray):
        if not _is_subdtype(vals.dtype, "c"):
            res[vals == -1] = np.nan
    elif np.isinf(res) and vals == -1:
        res = np.float64("nan")
    return res


def _factorialx_approx_core(n, k, extend):
    """
    Core approximation to multifactorial for array n and integer k.
    """
    if k == 1:
        # shortcut for k=1; same for both extensions, because we assume the
        # handling of extend == 'zero' happens in _factorialx_array_approx
        result = _gamma1p(n)
        if isinstance(n, np.ndarray):
            # gamma does not maintain 0-dim arrays; fix it
            result = np.array(result)
        return result

    if extend == "complex":
        # see https://numpy.org/doc/stable/reference/generated/numpy.power.html
        p_dtype = complex if (_is_subdtype(type(k), "c") or k < 0) else None
        with warnings.catch_warnings():
            # do not warn about 0 * inf, nan / nan etc.; the results are correct
            warnings.simplefilter("ignore", RuntimeWarning)
            # don't use `(n-1)/k` in np.power; underflows if 0 is of a uintX type
            result = np.power(k, n / k, dtype=p_dtype) * _gamma1p(n / k)
            result *= rgamma(1 / k + 1) / np.power(k, 1 / k, dtype=p_dtype)
        if isinstance(n, np.ndarray):
            # ensure we keep array-ness for 0-dim inputs; already n/k above loses it
            result = np.array(result)
        return result

    # at this point we are guaranteed that extend='zero' and that k>0 is an integer
    n_mod_k = n % k
    # scalar case separately, unified handling would be inefficient for arrays;
    # don't use isscalar due to numpy/numpy#23574; 0-dim arrays treated below
    if not isinstance(n, np.ndarray):
        return (
            np.power(k, (n - n_mod_k) / k)
            * gamma(n / k + 1) / gamma(n_mod_k / k + 1)
            * max(n_mod_k, 1)
        )

    # factor that's independent of the residue class (see factorialk docstring)
    result = np.power(k, n / k) * gamma(n / k + 1)
    # factor dependent on residue r (for `r=0` it's 1, so we skip `r=0`
    # below and thus also avoid evaluating `max(r, 1)`)
    def corr(k, r): return np.power(k, -r / k) / gamma(r / k + 1) * r
    for r in np.unique(n_mod_k):
        if r == 0:
            continue
        # cast to int because uint types break on `-r`
        result[n_mod_k == r] *= corr(k, int(r))
    return result


def _is_subdtype(dtype, dtypes):
    """
    Shorthand for calculating whether dtype is subtype of some dtypes.

    Also allows specifying a list instead of just a single dtype.

    Additionaly, the most important supertypes from
        https://numpy.org/doc/stable/reference/arrays.scalars.html
    can optionally be specified using abbreviations as follows:
        "i": np.integer
        "f": np.floating
        "c": np.complexfloating
        "n": np.number (contains the other three)
    """
    dtypes = dtypes if isinstance(dtypes, list) else [dtypes]
    # map single character abbreviations, if they are in dtypes
    mapping = {
        "i": np.integer,
        "f": np.floating,
        "c": np.complexfloating,
        "n": np.number
    }
    dtypes = [mapping.get(x, x) for x in dtypes]
    return any(np.issubdtype(dtype, dt) for dt in dtypes)


def _factorialx_wrapper(fname, n, k, exact, extend):
    """
    Shared implementation for factorial, factorial2 & factorialk.
    """
    if extend not in ("zero", "complex"):
        raise ValueError(
            f"argument `extend` must be either 'zero' or 'complex', received: {extend}"
        )
    if exact and extend == "complex":
        raise ValueError("Incompatible options: `exact=True` and `extend='complex'`")

    msg_unsup = (
        "Unsupported data type for {vname} in {fname}: {dtype}\n"
    )
    if fname == "factorial":
        msg_unsup += (
            "Permitted data types are integers and floating point numbers, "
            "as well as complex numbers if `extend='complex' is passed."
        )
    else:
        msg_unsup += (
            "Permitted data types are integers, as well as floating point "
            "numbers and complex numbers if `extend='complex' is passed."
        )
    msg_exact_not_possible = (
        "`exact=True` only supports integers, cannot use data type {dtype}"
    )
    msg_needs_complex = (
        "In order to use non-integer arguments, you must opt into this by passing "
        "`extend='complex'`. Note that this changes the result for all negative "
        "arguments (which by default return 0)."
    )

    if fname == "factorial2":
        msg_needs_complex += (" Additionally, it will rescale the values of the double"
                              " factorial at even integers by a factor of sqrt(2/pi).")
    elif fname == "factorialk":
        msg_needs_complex += (" Additionally, it will perturb the values of the"
                              " multifactorial at most positive integers `n`.")
        # check type of k
        if not _is_subdtype(type(k), ["i", "f", "c"]):
            raise ValueError(msg_unsup.format(vname="`k`", fname=fname, dtype=type(k)))
        elif _is_subdtype(type(k), ["f", "c"]) and extend != "complex":
            raise ValueError(msg_needs_complex)
        # check value of k
        if extend == "zero" and k < 1:
            msg = f"For `extend='zero'`, k must be a positive integer, received: {k}"
            raise ValueError(msg)
        elif k == 0:
            raise ValueError("Parameter k cannot be zero!")

    # factorial allows floats also for extend="zero"
    types_requiring_complex = "c" if fname == "factorial" else ["f", "c"]

    # don't use isscalar due to numpy/numpy#23574; 0-dim arrays treated below
    if np.ndim(n) == 0 and not isinstance(n, np.ndarray):
        # scalar cases
        if not _is_subdtype(type(n), ["i", "f", "c", type(None)]):
            raise ValueError(msg_unsup.format(vname="`n`", fname=fname, dtype=type(n)))
        elif _is_subdtype(type(n), types_requiring_complex) and extend != "complex":
            raise ValueError(msg_needs_complex)
        elif n is None or np.isnan(n):
            complexify = (extend == "complex") and _is_subdtype(type(n), "c")
            return np.complex128("nan+nanj") if complexify else np.float64("nan")
        elif extend == "zero" and n < 0:
            return 0 if exact else np.float64(0)
        elif n in {0, 1}:
            return 1 if exact else np.float64(1)
        elif exact and _is_subdtype(type(n), "i"):
            # calculate with integers
            return _range_prod(1, n, k=k)
        elif exact:
            # only relevant for factorial
            raise ValueError(msg_exact_not_possible.format(dtype=type(n)))
        # approximation
        return _factorialx_approx_core(n, k=k, extend=extend)

    # arrays & array-likes
    n = asarray(n)

    if not _is_subdtype(n.dtype, ["i", "f", "c"]):
        raise ValueError(msg_unsup.format(vname="`n`", fname=fname, dtype=n.dtype))
    elif _is_subdtype(n.dtype, types_requiring_complex) and extend != "complex":
        raise ValueError(msg_needs_complex)
    elif exact and _is_subdtype(n.dtype, ["f"]):
        # only relevant for factorial
        raise ValueError(msg_exact_not_possible.format(dtype=n.dtype))

    if n.size == 0:
        # return empty arrays unchanged
        return n
    elif exact:
        # calculate with integers
        return _factorialx_array_exact(n, k=k)
    # approximation
    return _factorialx_array_approx(n, k=k, extend=extend)


def factorial(n, exact=False, extend="zero"):
    """
    The factorial of a number or array of numbers.

    The factorial of non-negative integer `n` is the product of all
    positive integers less than or equal to `n`::

        n! = n * (n - 1) * (n - 2) * ... * 1

    Parameters
    ----------
    n : int or float or complex (or array_like thereof)
        Input values for ``n!``. Complex values require ``extend='complex'``.
        By default, the return value for ``n < 0`` is 0.
    exact : bool, optional
        If ``exact`` is set to True, calculate the answer exactly using
        integer arithmetic, otherwise approximate using the gamma function
        (faster, but yields floats instead of integers).
        Default is False.
    extend : string, optional
        One of ``'zero'`` or ``'complex'``; this determines how values ``n<0``
        are handled - by default they are 0, but it is possible to opt into the
        complex extension of the factorial (see below).

    Returns
    -------
    nf : int or float or complex or ndarray
        Factorial of ``n``, as integer, float or complex (depending on ``exact``
        and ``extend``). Array inputs are returned as arrays.

    Notes
    -----
    For arrays with ``exact=True``, the factorial is computed only once, for
    the largest input, with each other result computed in the process.
    The output dtype is increased to ``int64`` or ``object`` if necessary.

    With ``exact=False`` the factorial is approximated using the gamma
    function (which is also the definition of the complex extension):

    .. math:: n! = \\Gamma(n+1)

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import factorial
    >>> arr = np.array([3, 4, 5])
    >>> factorial(arr, exact=False)
    array([   6.,   24.,  120.])
    >>> factorial(arr, exact=True)
    array([  6,  24, 120])
    >>> factorial(5, exact=True)
    120

    """
    return _factorialx_wrapper("factorial", n, k=1, exact=exact, extend=extend)


def factorial2(n, exact=False, extend="zero"):
    """Double factorial.

    This is the factorial with every second value skipped.  E.g., ``7!! = 7 * 5
    * 3 * 1``.  It can be approximated numerically as::

      n!! = 2 ** (n / 2) * gamma(n / 2 + 1) * sqrt(2 / pi)  n odd
          = 2 ** (n / 2) * gamma(n / 2 + 1)                 n even
          = 2 ** (n / 2) * (n / 2)!                         n even

    The formula for odd ``n`` is the basis for the complex extension.

    Parameters
    ----------
    n : int or float or complex (or array_like thereof)
        Input values for ``n!!``. Non-integer values require ``extend='complex'``.
        By default, the return value for ``n < 0`` is 0.
    exact : bool, optional
        If ``exact`` is set to True, calculate the answer exactly using
        integer arithmetic, otherwise use above approximation (faster,
        but yields floats instead of integers).
        Default is False.
    extend : string, optional
        One of ``'zero'`` or ``'complex'``; this determines how values ``n<0``
        are handled - by default they are 0, but it is possible to opt into the
        complex extension of the double factorial. This also enables passing
        complex values to ``n``.

        .. warning::

           Using the ``'complex'`` extension also changes the values of the
           double factorial for even integers, reducing them by a factor of
           ``sqrt(2/pi) ~= 0.79``, see [1].

    Returns
    -------
    nf : int or float or complex or ndarray
        Double factorial of ``n``, as integer, float or complex (depending on
        ``exact`` and ``extend``). Array inputs are returned as arrays.

    Examples
    --------
    >>> from scipy.special import factorial2
    >>> factorial2(7, exact=False)
    array(105.00000000000001)
    >>> factorial2(7, exact=True)
    105

    References
    ----------
    .. [1] Complex extension to double factorial
            https://en.wikipedia.org/wiki/Double_factorial#Complex_arguments
    """
    return _factorialx_wrapper("factorial2", n, k=2, exact=exact, extend=extend)


def factorialk(n, k, exact=False, extend="zero"):
    """Multifactorial of n of order k, n(!!...!).

    This is the multifactorial of n skipping k values.  For example,

      factorialk(17, 4) = 17!!!! = 17 * 13 * 9 * 5 * 1

    In particular, for any integer ``n``, we have

      factorialk(n, 1) = factorial(n)

      factorialk(n, 2) = factorial2(n)

    Parameters
    ----------
    n : int or float or complex (or array_like thereof)
        Input values for multifactorial. Non-integer values require
        ``extend='complex'``. By default, the return value for ``n < 0`` is 0.
    n : int or float or complex (or array_like thereof)
        Order of multifactorial. Non-integer values require ``extend='complex'``.
    exact : bool, optional
        If ``exact`` is set to True, calculate the answer exactly using
        integer arithmetic, otherwise use an approximation (faster,
        but yields floats instead of integers)
        Default is False.
    extend : string, optional
        One of ``'zero'`` or ``'complex'``; this determines how values ``n<0`` are
        handled - by default they are 0, but it is possible to opt into the complex
        extension of the multifactorial. This enables passing complex values,
        not only to ``n`` but also to ``k``.

        .. warning::

           Using the ``'complex'`` extension also changes the values of the
           multifactorial at integers ``n != 1 (mod k)`` by a factor depending
           on both ``k`` and ``n % k``, see below or [1].

    Returns
    -------
    nf : int or float or complex or ndarray
        Multifactorial (order ``k``) of ``n``, as integer, float or complex (depending
        on ``exact`` and ``extend``). Array inputs are returned as arrays.

    Examples
    --------
    >>> from scipy.special import factorialk
    >>> factorialk(5, k=1, exact=True)
    120
    >>> factorialk(5, k=3, exact=True)
    10
    >>> factorialk([5, 7, 9], k=3, exact=True)
    array([ 10,  28, 162])
    >>> factorialk([5, 7, 9], k=3, exact=False)
    array([ 10.,  28., 162.])

    Notes
    -----
    While less straight-forward than for the double-factorial, it's possible to
    calculate a general approximation formula of n!(k) by studying ``n`` for a given
    remainder ``r < k`` (thus ``n = m * k + r``, resp. ``r = n % k``), which can be
    put together into something valid for all integer values ``n >= 0`` & ``k > 0``::

      n!(k) = k ** ((n - r)/k) * gamma(n/k + 1) / gamma(r/k + 1) * max(r, 1)

    This is the basis of the approximation when ``exact=False``.

    In principle, any fixed choice of ``r`` (ignoring its relation ``r = n%k``
    to ``n``) would provide a suitable analytic continuation from integer ``n``
    to complex ``z`` (not only satisfying the functional equation but also
    being logarithmically convex, c.f. Bohr-Mollerup theorem) -- in fact, the
    choice of ``r`` above only changes the function by a constant factor. The
    final constraint that determines the canonical continuation is ``f(1) = 1``,
    which forces ``r = 1`` (see also [1]).::

      z!(k) = k ** ((z - 1)/k) * gamma(z/k + 1) / gamma(1/k + 1)

    References
    ----------
    .. [1] Complex extension to multifactorial
            https://en.wikipedia.org/wiki/Double_factorial#Alternative_extension_of_the_multifactorial
    """
    return _factorialx_wrapper("factorialk", n, k=k, exact=exact, extend=extend)


def stirling2(N, K, *, exact=False):
    r"""Generate Stirling number(s) of the second kind.

    Stirling numbers of the second kind count the number of ways to
    partition a set with N elements into K non-empty subsets.

    The values this function returns are calculated using a dynamic
    program which avoids redundant computation across the subproblems
    in the solution. For array-like input, this implementation also
    avoids redundant computation across the different Stirling number
    calculations.

    The numbers are sometimes denoted

    .. math::

        {N \brace{K}}

    see [1]_ for details. This is often expressed-verbally-as
    "N subset K".

    Parameters
    ----------
    N : int, ndarray
        Number of things.
    K : int, ndarray
        Number of non-empty subsets taken.
    exact : bool, optional
        Uses dynamic programming (DP) with floating point
        numbers for smaller arrays and uses a second order approximation due to
        Temme for larger entries  of `N` and `K` that allows trading speed for
        accuracy. See [2]_ for a description. Temme approximation is used for
        values ``n>50``. The max error from the DP has max relative error
        ``4.5*10^-16`` for ``n<=50`` and the max error from the Temme approximation
        has max relative error ``5*10^-5`` for ``51 <= n < 70`` and
        ``9*10^-6`` for ``70 <= n < 101``. Note that these max relative errors will
        decrease further as `n` increases.

    Returns
    -------
    val : int, float, ndarray
        The number of partitions.

    See Also
    --------
    comb : The number of combinations of N things taken k at a time.

    Notes
    -----
    - If N < 0, or K < 0, then 0 is returned.
    - If K > N, then 0 is returned.

    The output type will always be `int` or ndarray of `object`.
    The input must contain either numpy or python integers otherwise a
    TypeError is raised.

    References
    ----------
    .. [1] R. L. Graham, D. E. Knuth and O. Patashnik, "Concrete
        Mathematics: A Foundation for Computer Science," Addison-Wesley
        Publishing Company, Boston, 1989. Chapter 6, page 258.

    .. [2] Temme, Nico M. "Asymptotic estimates of Stirling numbers."
        Studies in Applied Mathematics 89.3 (1993): 233-243.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import stirling2
    >>> k = np.array([3, -1, 3])
    >>> n = np.array([10, 10, 9])
    >>> stirling2(n, k)
    array([9330.0, 0.0, 3025.0])

    """
    output_is_scalar = np.isscalar(N) and np.isscalar(K)
    # make a min-heap of unique (n,k) pairs
    N, K = asarray(N), asarray(K)
    if not np.issubdtype(N.dtype, np.integer):
        raise TypeError("Argument `N` must contain only integers")
    if not np.issubdtype(K.dtype, np.integer):
        raise TypeError("Argument `K` must contain only integers")
    if not exact:
        # NOTE: here we allow np.uint via casting to double types prior to
        # passing to private ufunc dispatcher. All dispatched functions
        # take double type for (n,k) arguments and return double.
        return _stirling2_inexact(N.astype(float), K.astype(float))
    nk_pairs = list(
        set([(n.take(0), k.take(0))
             for n, k in np.nditer([N, K], ['refs_ok'])])
    )
    heapify(nk_pairs)
    # base mapping for small values
    snsk_vals = defaultdict(int)
    for pair in [(0, 0), (1, 1), (2, 1), (2, 2)]:
        snsk_vals[pair] = 1
    # for each pair in the min-heap, calculate the value, store for later
    n_old, n_row = 2, [0, 1, 1]
    while nk_pairs:
        n, k = heappop(nk_pairs)
        if n < 2 or k > n or k <= 0:
            continue
        elif k == n or k == 1:
            snsk_vals[(n, k)] = 1
            continue
        elif n != n_old:
            num_iters = n - n_old
            while num_iters > 0:
                n_row.append(1)
                # traverse from back to remove second row
                for j in range(len(n_row)-2, 1, -1):
                    n_row[j] = n_row[j]*j + n_row[j-1]
                num_iters -= 1
            snsk_vals[(n, k)] = n_row[k]
        else:
            snsk_vals[(n, k)] = n_row[k]
        n_old, n_row = n, n_row
    out_types = [object, object, object] if exact else [float, float, float]
    # for each pair in the map, fetch the value, and populate the array
    it = np.nditer(
        [N, K, None],
        ['buffered', 'refs_ok'],
        [['readonly'], ['readonly'], ['writeonly', 'allocate']],
        op_dtypes=out_types,
    )
    with it:
        while not it.finished:
            it[2] = snsk_vals[(int(it[0]), int(it[1]))]
            it.iternext()
        output = it.operands[2]
        # If N and K were both scalars, convert output to scalar.
        if output_is_scalar:
            output = output.take(0)
    return output


def zeta(x, q=None, out=None):
    r"""
    Riemann or Hurwitz zeta function.

    Parameters
    ----------
    x : array_like of float or complex.
        Input data
    q : array_like of float, optional
        Input data, must be real.  Defaults to Riemann zeta. When `q` is
        ``None``, complex inputs `x` are supported. If `q` is not ``None``,
        then currently only real inputs `x` with ``x >= 1`` are supported,
        even when ``q = 1.0`` (corresponding to the Riemann zeta function).

    out : ndarray, optional
        Output array for the computed values.

    Returns
    -------
    out : array_like
        Values of zeta(x).

    See Also
    --------
    zetac

    Notes
    -----
    The two-argument version is the Hurwitz zeta function

    .. math::

        \zeta(x, q) = \sum_{k=0}^{\infty} \frac{1}{(k + q)^x};

    see [dlmf]_ for details. The Riemann zeta function corresponds to
    the case when ``q = 1``.

    For complex inputs with ``q = None``, points with
    ``abs(z.imag) > 1e9`` and ``0 <= abs(z.real) < 2.5`` are currently not
    supported due to slow convergence causing excessive runtime.

    References
    ----------
    .. [dlmf] NIST, Digital Library of Mathematical Functions,
        https://dlmf.nist.gov/25.11#i

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import zeta, polygamma, factorial

    Some specific values:

    >>> zeta(2), np.pi**2/6
    (1.6449340668482266, 1.6449340668482264)

    >>> zeta(4), np.pi**4/90
    (1.0823232337111381, 1.082323233711138)

    First nontrivial zero:

    >>> zeta(0.5 + 14.134725141734695j)
    0 + 0j

    Relation to the `polygamma` function:

    >>> m = 3
    >>> x = 1.25
    >>> polygamma(m, x)
    array(2.782144009188397)
    >>> (-1)**(m+1) * factorial(m) * zeta(m+1, x)
    2.7821440091883969

    """
    if q is None:
        return _ufuncs._riemann_zeta(x, out)
    else:
        return _ufuncs._zeta(x, q, out)


def softplus(x, **kwargs):
    r"""
    Compute the softplus function element-wise.

    The softplus function is defined as: ``softplus(x) = log(1 + exp(x))``.
    It is a smooth approximation of the rectifier function (ReLU).

    Parameters
    ----------
    x : array_like
        Input value.
    **kwargs
        For other keyword-only arguments, see the
        `ufunc docs <https://numpy.org/doc/stable/reference/ufuncs.html>`_.

    Returns
    -------
    softplus : ndarray
        Logarithm of ``exp(0) + exp(x)``.

    Examples
    --------
    >>> from scipy import special

    >>> special.softplus(0)
    0.6931471805599453

    >>> special.softplus([-1, 0, 1])
    array([0.31326169, 0.69314718, 1.31326169])
    """
    return np.logaddexp(0, x, **kwargs)
<|endoftext|>"""Precompute coefficients of several series expansions
of Wright's generalized Bessel function Phi(a, b, x).

See https://dlmf.nist.gov/10.46.E1 with rho=a, beta=b, z=x.
"""
from argparse import ArgumentParser, RawTextHelpFormatter
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, curve_fit
from time import time

try:
    import sympy
    from sympy import EulerGamma, Rational, S, Sum, \
        factorial, gamma, gammasimp, pi, polygamma, symbols, zeta
    from sympy.polys.polyfuncs import horner
except ImportError:
    pass


def series_small_a():
    """Tylor series expansion of Phi(a, b, x) in a=0 up to order 5.
    """
    order = 5
    a, b, x, k = symbols("a b x k")
    A = []  # terms with a
    X = []  # terms with x
    B = []  # terms with b (polygammas)
    # Phi(a, b, x) = exp(x)/gamma(b) * sum(A[i] * X[i] * B[i])
    expression = Sum(x**k/factorial(k)/gamma(a*k+b), (k, 0, S.Infinity))
    expression = gamma(b)/sympy.exp(x) * expression

    # nth term of taylor series in a=0: a^n/n! * (d^n Phi(a, b, x)/da^n at a=0)
    for n in range(0, order+1):
        term = expression.diff(a, n).subs(a, 0).simplify().doit()
        # set the whole bracket involving polygammas to 1
        x_part = (term.subs(polygamma(0, b), 1)
                  .replace(polygamma, lambda *args: 0))
        # sign convention: x part always positive
        x_part *= (-1)**n

        A.append(a**n/factorial(n))
        X.append(horner(x_part))
        B.append(horner((term/x_part).simplify()))

    s = "Tylor series expansion of Phi(a, b, x) in a=0 up to order 5.\n"
    s += "Phi(a, b, x) = exp(x)/gamma(b) * sum(A[i] * X[i] * B[i], i=0..5)\n"
    for name, c in zip(['A', 'X', 'B'], [A, X, B]):
        for i in range(len(c)):
            s += f"\n{name}[{i}] = " + str(c[i])
    return s


# expansion of digamma
def dg_series(z, n):
    """Symbolic expansion of digamma(z) in z=0 to order n.

    See https://dlmf.nist.gov/5.7.E4 and with https://dlmf.nist.gov/5.5.E2
    """
    k = symbols("k")
    return -1/z - EulerGamma + \
        sympy.summation((-1)**k * zeta(k) * z**(k-1), (k, 2, n+1))


def pg_series(k, z, n):
    """Symbolic expansion of polygamma(k, z) in z=0 to order n."""
    return sympy.diff(dg_series(z, n+k), z, k)


def series_small_a_small_b():
    """Tylor series expansion of Phi(a, b, x) in a=0 and b=0 up to order 5.

    Be aware of cancellation of poles in b=0 of digamma(b)/Gamma(b) and
    polygamma functions.

    digamma(b)/Gamma(b) = -1 - 2*M_EG*b + O(b^2)
    digamma(b)^2/Gamma(b) = 1/b + 3*M_EG + b*(-5/12*PI^2+7/2*M_EG^2) + O(b^2)
    polygamma(1, b)/Gamma(b) = 1/b + M_EG + b*(1/12*PI^2 + 1/2*M_EG^2) + O(b^2)
    and so on.
    """
    order = 5
    a, b, x, k = symbols("a b x k")
    M_PI, M_EG, M_Z3 = symbols("M_PI M_EG M_Z3")
    c_subs = {pi: M_PI, EulerGamma: M_EG, zeta(3): M_Z3}
    A = []  # terms with a
    X = []  # terms with x
    B = []  # terms with b (polygammas expanded)
    C = []  # terms that generate B
    # Phi(a, b, x) = exp(x) * sum(A[i] * X[i] * B[i])
    # B[0] = 1
    # B[k] = sum(C[k] * b**k/k!, k=0..)
    # Note: C[k] can be obtained from a series expansion of 1/gamma(b).
    expression = gamma(b)/sympy.exp(x) * \
        Sum(x**k/factorial(k)/gamma(a*k+b), (k, 0, S.Infinity))

    # nth term of taylor series in a=0: a^n/n! * (d^n Phi(a, b, x)/da^n at a=0)
    for n in range(0, order+1):
        term = expression.diff(a, n).subs(a, 0).simplify().doit()
        # set the whole bracket involving polygammas to 1
        x_part = (term.subs(polygamma(0, b), 1)
                  .replace(polygamma, lambda *args: 0))
        # sign convention: x part always positive
        x_part *= (-1)**n
        # expansion of polygamma part with 1/gamma(b)
        pg_part = term/x_part/gamma(b)
        if n >= 1:
            # Note: highest term is digamma^n
            pg_part = pg_part.replace(polygamma,
                                      lambda k, x: pg_series(k, x, order+1+n))
            pg_part = (pg_part.series(b, 0, n=order+1-n)
                       .removeO()
                       .subs(polygamma(2, 1), -2*zeta(3))
                       .simplify()
                       )

        A.append(a**n/factorial(n))
        X.append(horner(x_part))
        B.append(pg_part)

    # Calculate C and put in the k!
    C = sympy.Poly(B[1].subs(c_subs), b).coeffs()
    C.reverse()
    for i in range(len(C)):
        C[i] = (C[i] * factorial(i)).simplify()

    s = "Tylor series expansion of Phi(a, b, x) in a=0 and b=0 up to order 5."
    s += "\nPhi(a, b, x) = exp(x) * sum(A[i] * X[i] * B[i], i=0..5)\n"
    s += "B[0] = 1\n"
    s += "B[i] = sum(C[k+i-1] * b**k/k!, k=0..)\n"
    s += "\nM_PI = pi"
    s += "\nM_EG = EulerGamma"
    s += "\nM_Z3 = zeta(3)"
    for name, c in zip(['A', 'X'], [A, X]):
        for i in range(len(c)):
            s += f"\n{name}[{i}] = "
            s += str(c[i])
    # For C, do also compute the values numerically
    for i in range(len(C)):
        s += f"\n# C[{i}] = "
        s += str(C[i])
        s += f"\nC[{i}] = "
        s += str(C[i].subs({M_EG: EulerGamma, M_PI: pi, M_Z3: zeta(3)})
                 .evalf(17))

    # Does B have the assumed structure?
    s += "\n\nTest if B[i] does have the assumed structure."
    s += "\nC[i] are derived from B[1] alone."
    s += "\nTest B[2] == C[1] + b*C[2] + b^2/2*C[3] + b^3/6*C[4] + .."
    test = sum([b**k/factorial(k) * C[k+1] for k in range(order-1)])
    test = (test - B[2].subs(c_subs)).simplify()
    s += f"\ntest successful = {test==S(0)}"
    s += "\nTest B[3] == C[2] + b*C[3] + b^2/2*C[4] + .."
    test = sum([b**k/factorial(k) * C[k+2] for k in range(order-2)])
    test = (test - B[3].subs(c_subs)).simplify()
    s += f"\ntest successful = {test==S(0)}"
    return s


def asymptotic_series():
    """Asymptotic expansion for large x.

    Phi(a, b, x) ~ Z^(1/2-b) * exp((1+a)/a * Z) * sum_k (-1)^k * C_k / Z^k
    Z = (a*x)^(1/(1+a))

    Wright (1935) lists the coefficients C_0 and C_1 (he calls them a_0 and
    a_1). With slightly different notation, Paris (2017) lists coefficients
    c_k up to order k=3.
    Paris (2017) uses ZP = (1+a)/a * Z  (ZP = Z of Paris) and
    C_k = C_0 * (-a/(1+a))^k * c_k
    """
    order = 8

    class g(sympy.Function):
        """Helper function g according to Wright (1935)

        g(n, rho, v) = (1 + (rho+2)/3 * v + (rho+2)*(rho+3)/(2*3) * v^2 + ...)

        Note: Wright (1935) uses square root of above definition.
        """
        nargs = 3

        @classmethod
        def eval(cls, n, rho, v):
            if not n >= 0:
                raise ValueError("must have n >= 0")
            elif n == 0:
                return 1
            else:
                return g(n-1, rho, v) \
                    + gammasimp(gamma(rho+2+n)/gamma(rho+2)) \
                    / gammasimp(gamma(3+n)/gamma(3))*v**n

    class coef_C(sympy.Function):
        """Calculate coefficients C_m for integer m.

        C_m is the coefficient of v^(2*m) in the Taylor expansion in v=0 of
        Gamma(m+1/2)/(2*pi) * (2/(rho+1))^(m+1/2) * (1-v)^(-b)
            * g(rho, v)^(-m-1/2)
        """
        nargs = 3

        @classmethod
        def eval(cls, m, rho, beta):
            if not m >= 0:
                raise ValueError("must have m >= 0")

            v = symbols("v")
            expression = (1-v)**(-beta) * g(2*m, rho, v)**(-m-Rational(1, 2))
            res = expression.diff(v, 2*m).subs(v, 0) / factorial(2*m)
            res = res * (gamma(m + Rational(1, 2)) / (2*pi)
                         * (2/(rho+1))**(m + Rational(1, 2)))
            return res

    # in order to have nice ordering/sorting of expressions, we set a = xa.
    xa, b, xap1 = symbols("xa b xap1")
    C0 = coef_C(0, xa, b)
    # a1 = a(1, rho, beta)
    s = "Asymptotic expansion for large x\n"
    s += "Phi(a, b, x) = Z**(1/2-b) * exp((1+a)/a * Z) \n"
    s += "               * sum((-1)**k * C[k]/Z**k, k=0..6)\n\n"
    s += "Z      = pow(a * x, 1/(1+a))\n"
    s += "A[k]   = pow(a, k)\n"
    s += "B[k]   = pow(b, k)\n"
    s += "Ap1[k] = pow(1+a, k)\n\n"
    s += "C[0] = 1./sqrt(2. * M_PI * Ap1[1])\n"
    for i in range(1, order+1):
        expr = (coef_C(i, xa, b) / (C0/(1+xa)**i)).simplify()
        factor = [x.denominator() for x in sympy.Poly(expr).coeffs()]
        factor = sympy.lcm(factor)
        expr = (expr * factor).simplify().collect(b, sympy.factor)
        expr = expr.xreplace({xa+1: xap1})
        s += f"C[{i}] = C[0] / ({factor} * Ap1[{i}])\n"
        s += f"C[{i}] *= {str(expr)}\n\n"
    import re
    re_a = re.compile(r'xa\*\*(\d+)')
    s = re_a.sub(r'A[\1]', s)
    re_b = re.compile(r'b\*\*(\d+)')
    s = re_b.sub(r'B[\1]', s)
    s = s.replace('xap1', 'Ap1[1]')
    s = s.replace('xa', 'a')
    # max integer = 2^31-1 = 2,147,483,647. Solution: Put a point after 10
    # or more digits.
    re_digits = re.compile(r'(\d{10,})')
    s = re_digits.sub(r'\1.', s)
    return s


def optimal_epsilon_integral():
    """Fit optimal choice of epsilon for integral representation.

    The integrand of
        int_0^pi P(eps, a, b, x, phi) * dphi
    can exhibit oscillatory behaviour. It stems from the cosine of P and can be
    minimized by minimizing the arc length of the argument
        f(phi) = eps * sin(phi) - x * eps^(-a) * sin(a * phi) + (1 - b) * phi
    of cos(f(phi)).
    We minimize the arc length in eps for a grid of values (a, b, x) and fit a
    parametric function to it.
    """
    def fp(eps, a, b, x, phi):
        """Derivative of f w.r.t. phi."""
        eps_a = np.power(1. * eps, -a)
        return eps * np.cos(phi) - a * x * eps_a * np.cos(a * phi) + 1 - b

    def arclength(eps, a, b, x, epsrel=1e-2, limit=100):
        """Compute Arc length of f.

        Note that the arc length of a function f from t0 to t1 is given by
            int_t0^t1 sqrt(1 + f'(t)^2) dt
        """
        return quad(lambda phi: np.sqrt(1 + fp(eps, a, b, x, phi)**2),
                    0, np.pi,
                    epsrel=epsrel, limit=100)[0]

    # grid of minimal arc length values
    data_a = [1e-3, 0.1, 0.5, 0.9, 1, 2, 4, 5, 6, 8]
    data_b = [0, 1, 4, 7, 10]
    data_x = [1, 1.5, 2, 4, 10, 20, 50, 100, 200, 500, 1e3, 5e3, 1e4]
    data_a, data_b, data_x = np.meshgrid(data_a, data_b, data_x)
    data_a, data_b, data_x = (data_a.flatten(), data_b.flatten(),
                              data_x.flatten())
    best_eps = []
    for i in range(data_x.size):
        best_eps.append(
            minimize_scalar(lambda eps: arclength(eps, data_a[i], data_b[i],
                                                  data_x[i]),
                            bounds=(1e-3, 1000),
                            method='Bounded', options={'xatol': 1e-3}).x
        )
    best_eps = np.array(best_eps)
    # pandas would be nice, but here a dictionary is enough
    df = {'a': data_a,
          'b': data_b,
          'x': data_x,
          'eps': best_eps,
          }

    def func(data, A0, A1, A2, A3, A4, A5):
        """Compute parametric function to fit."""
        a = data['a']
        b = data['b']
        x = data['x']
        return (A0 * b * np.exp(-0.5 * a)
                + np.exp(A1 + 1 / (1 + a) * np.log(x) - A2 * np.exp(-A3 * a)
                         + A4 / (1 + np.exp(A5 * a))))

    func_params = list(curve_fit(func, df, df['eps'], method='trf')[0])

    s = "Fit optimal eps for integrand P via minimal arc length\n"
    s += "with parametric function:\n"
    s += "optimal_eps = (A0 * b * exp(-a/2) + exp(A1 + 1 / (1 + a) * log(x)\n"
    s += "              - A2 * exp(-A3 * a) + A4 / (1 + exp(A5 * a)))\n\n"
    s += "Fitted parameters A0 to A5 are:\n"
    s += ', '.join([f'{x:.5g}' for x in func_params])
    return s


def main():
    t0 = time()
    parser = ArgumentParser(description=__doc__,
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument('action', type=int, choices=[1, 2, 3, 4],
                        help='chose what expansion to precompute\n'
                             '1 : Series for small a\n'
                             '2 : Series for small a and small b\n'
                             '3 : Asymptotic series for large x\n'
                             '    This may take some time (>4h).\n'
                             '4 : Fit optimal eps for integral representation.'
                        )
    args = parser.parse_args()

    switch = {1: lambda: print(series_small_a()),
              2: lambda: print(series_small_a_small_b()),
              3: lambda: print(asymptotic_series()),
              4: lambda: print(optimal_epsilon_integral())
              }
    switch.get(args.action, lambda: print("Invalid input."))()
    print(f"\n{(time() - t0)/60:.1f} minutes elapsed.\n")


if __name__ == '__main__':
    main()
<|endoftext|>"""Precompute series coefficients for log-Gamma."""

try:
    import mpmath
except ImportError:
    pass


def stirling_series(N):
    with mpmath.workdps(100):
        coeffs = [mpmath.bernoulli(2*n)/(2*n*(2*n - 1))
                  for n in range(1, N + 1)]
    return coeffs


def taylor_series_at_1(N):
    coeffs = []
    with mpmath.workdps(100):
        coeffs.append(-mpmath.euler)
        for n in range(2, N + 1):
            coeffs.append((-1)**n*mpmath.zeta(n)/n)
    return coeffs


def main():
    print(__doc__)
    print()
    stirling_coeffs = [mpmath.nstr(x, 20, min_fixed=0, max_fixed=0)
                       for x in stirling_series(8)[::-1]]
    taylor_coeffs = [mpmath.nstr(x, 20, min_fixed=0, max_fixed=0)
                     for x in taylor_series_at_1(23)[::-1]]
    print("Stirling series coefficients")
    print("----------------------------")
    print("\n".join(stirling_coeffs))
    print()
    print("Taylor series coefficients")
    print("--------------------------")
    print("\n".join(taylor_coeffs))
    print()


if __name__ == '__main__':
    main()
<|endoftext|>"""Compute a Pade approximation for the principal branch of the
Lambert W function around 0 and compare it to various other
approximations.

"""
import numpy as np

try:
    import mpmath
    import matplotlib.pyplot as plt
except ImportError:
    pass


def lambertw_pade():
    derivs = [mpmath.diff(mpmath.lambertw, 0, n=n) for n in range(6)]
    p, q = mpmath.pade(derivs, 3, 2)
    return p, q


def main():
    print(__doc__)
    with mpmath.workdps(50):
        p, q = lambertw_pade()
        p, q = p[::-1], q[::-1]
        print(f"p = {p}")
        print(f"q = {q}")

    x, y = np.linspace(-1.5, 1.5, 75), np.linspace(-1.5, 1.5, 75)
    x, y = np.meshgrid(x, y)
    z = x + 1j*y
    lambertw_std = []
    for z0 in z.flatten():
        lambertw_std.append(complex(mpmath.lambertw(z0)))
    lambertw_std = np.array(lambertw_std).reshape(x.shape)

    fig, axes = plt.subplots(nrows=3, ncols=1)
    # Compare Pade approximation to true result
    p = np.array([float(p0) for p0 in p])
    q = np.array([float(q0) for q0 in q])
    pade_approx = np.polyval(p, z)/np.polyval(q, z)
    pade_err = abs(pade_approx - lambertw_std)
    axes[0].pcolormesh(x, y, pade_err)
    # Compare two terms of asymptotic series to true result
    asy_approx = np.log(z) - np.log(np.log(z))
    asy_err = abs(asy_approx - lambertw_std)
    axes[1].pcolormesh(x, y, asy_err)
    # Compare two terms of the series around the branch point to the
    # true result
    p = np.sqrt(2*(np.exp(1)*z + 1))
    series_approx = -1 + p - p**2/3
    series_err = abs(series_approx - lambertw_std)
    im = axes[2].pcolormesh(x, y, series_err)

    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    pade_better = pade_err < asy_err
    im = ax.pcolormesh(x, y, pade_better)
    t = np.linspace(-0.3, 0.3)
    ax.plot(-2.5*abs(t) - 0.2, t, 'r')
    fig.colorbar(im, ax=ax)
    plt.show()


if __name__ == '__main__':
    main()
<|endoftext|>"""This script evaluates scipy's implementation of hyp2f1 against mpmath's.

Author: Albert Steppi

This script is long running and generates a large output file. With default
arguments, the generated file is roughly 700MB in size and it takes around
40 minutes using an Intel(R) Core(TM) i5-8250U CPU with n_jobs set to 8
(full utilization). There are optional arguments which can be used to restrict
(or enlarge) the computations performed. These are described below.
The output of this script can be analyzed to identify suitable test cases and
to find parameter and argument regions where hyp2f1 needs to be improved.

The script has one mandatory positional argument for specifying the path to
the location where the output file is to be placed, and 4 optional arguments
--n_jobs, --grid_size, --regions, and --parameter_groups. --n_jobs specifies
the number of processes to use if running in parallel. The default value is 1.
The other optional arguments are explained below.

Produces a tab separated values file with 11 columns. The first four columns
contain the parameters a, b, c and the argument z. The next two contain |z| and
a region code for which region of the complex plane belongs to. The regions are

    0) z == 1
    1) |z| < 0.9 and real(z) >= 0
    2) |z| <= 1 and real(z) < 0
    3) 0.9 <= |z| <= 1 and |1 - z| < 0.9:
    4) 0.9 <= |z| <= 1 and |1 - z| >= 0.9 and real(z) >= 0:
    5) 1 < |z| < 1.1 and |1 - z| >= 0.9 and real(z) >= 0
    6) |z| > 1 and not in 5)

The --regions optional argument allows the user to specify a list of regions
to which computation will be restricted.

Parameters a, b, c are taken from a 10 * 10 * 10 grid with values at

    -16, -8, -4, -2, -1, 1, 2, 4, 8, 16

with random perturbations applied.

There are 9 parameter groups handling the following cases.

    1) A, B, C, B - A, C - A, C - B, C - A - B all non-integral.
    2) B - A integral
    3) C - A integral
    4) C - B integral
    5) C - A - B integral
    6) A integral
    7) B integral
    8) C integral
    9) Wider range with c - a - b > 0.

The seventh column of the output file is an integer between 1 and 8 specifying
the parameter group as above.

The --parameter_groups optional argument allows the user to specify a list of
parameter groups to which computation will be restricted.

The argument z is taken from a grid in the box
    -box_size <= real(z) <= box_size, -box_size <= imag(z) <= box_size.
with grid size specified using the optional command line argument --grid_size,
and box_size specified with the command line argument --box_size.
The default value of grid_size is 20 and the default value of box_size is 2.0,
yielding a 20 * 20 grid in the box with corners -2-2j, -2+2j, 2-2j, 2+2j.

The final four columns have the expected value of hyp2f1 for the given
parameters and argument as calculated with mpmath, the observed value
calculated with scipy's hyp2f1, the relative error, and the absolute error.

As special cases of hyp2f1 are moved from the original Fortran implementation
into Cython, this script can be used to ensure that no regressions occur and
to point out where improvements are needed.
"""


import os
import csv
import argparse
import numpy as np
from itertools import product
from multiprocessing import Pool


from scipy.special import hyp2f1
from scipy.special.tests.test_hyp2f1 import mp_hyp2f1


def get_region(z):
    """Assign numbers for regions where hyp2f1 must be handled differently."""
    if z == 1 + 0j:
        return 0
    elif abs(z) < 0.9 and z.real >= 0:
        return 1
    elif abs(z) <= 1 and z.real < 0:
        return 2
    elif 0.9 <= abs(z) <= 1 and abs(1 - z) < 0.9:
        return 3
    elif 0.9 <= abs(z) <= 1 and abs(1 - z) >= 0.9:
        return 4
    elif 1 < abs(z) < 1.1 and abs(1 - z) >= 0.9 and z.real >= 0:
        return 5
    else:
        return 6


def get_result(a, b, c, z, group):
    """Get results for given parameter and value combination."""
    expected, observed = mp_hyp2f1(a, b, c, z), hyp2f1(a, b, c, z)
    if (
            np.isnan(observed) and np.isnan(expected) or
            expected == observed
    ):
        relative_error = 0.0
        absolute_error = 0.0
    elif np.isnan(observed):
        # Set error to infinity if result is nan when not expected to be.
        # Makes results easier to interpret.
        relative_error = float("inf")
        absolute_error = float("inf")
    else:
        absolute_error = abs(expected - observed)
        relative_error = absolute_error / abs(expected)

    return (
        a,
        b,
        c,
        z,
        abs(z),
        get_region(z),
        group,
        expected,
        observed,
        relative_error,
        absolute_error,
    )


def get_result_no_mp(a, b, c, z, group):
    """Get results for given parameter and value combination."""
    expected, observed = complex('nan'), hyp2f1(a, b, c, z)
    relative_error, absolute_error = float('nan'), float('nan')
    return (
        a,
        b,
        c,
        z,
        abs(z),
        get_region(z),
        group,
        expected,
        observed,
        relative_error,
        absolute_error,
    )


def get_results(params, Z, n_jobs=1, compute_mp=True):
    """Batch compute results for multiple parameter and argument values.

    Parameters
    ----------
    params : iterable
        iterable of tuples of floats (a, b, c) specifying parameter values
        a, b, c for hyp2f1
    Z : iterable of complex
        Arguments at which to evaluate hyp2f1
    n_jobs : Optional[int]
        Number of jobs for parallel execution.

    Returns
    -------
    list
        List of tuples of results values. See return value in source code
        of `get_result`.
    """
    input_ = (
        (a, b, c, z, group) for (a, b, c, group), z in product(params, Z)
    )

    with Pool(n_jobs) as pool:
        rows = pool.starmap(
            get_result if compute_mp else get_result_no_mp,
            input_
        )
    return rows


def _make_hyp2f1_test_case(a, b, c, z, rtol):
    """Generate string for single test case as used in test_hyp2f1.py."""
    expected = mp_hyp2f1(a, b, c, z)
    return (
        "    pytest.param(\n"
        "        Hyp2f1TestCase(\n"
        f"            a={a},\n"
        f"            b={b},\n"
        f"            c={c},\n"
        f"            z={z},\n"
        f"            expected={expected},\n"
        f"            rtol={rtol},\n"
        "        ),\n"
        "    ),"
    )


def make_hyp2f1_test_cases(rows):
    """Generate string for a list of test cases for test_hyp2f1.py.

    Parameters
    ----------
    rows : list
        List of lists of the form [a, b, c, z, rtol] where a, b, c, z are
        parameters and the argument for hyp2f1 and rtol is an expected
        relative error for the associated test case.

    Returns
    -------
    str
        String for a list of test cases. The output string can be printed
        or saved to a file and then copied into an argument for
        `pytest.mark.parameterize` within `scipy.special.tests.test_hyp2f1.py`.
    """
    result = "[\n"
    result += '\n'.join(
        _make_hyp2f1_test_case(a, b, c, z, rtol)
        for a, b, c, z, rtol in rows
    )
    result += "\n]"
    return result


def main(
        outpath,
        n_jobs=1,
        box_size=2.0,
        grid_size=20,
        regions=None,
        parameter_groups=None,
        compute_mp=True,
):
    outpath = os.path.realpath(os.path.expanduser(outpath))

    random_state = np.random.RandomState(1234)
    # Parameters a, b, c selected near these values.
    root_params = np.array(
        [-16, -8, -4, -2, -1, 1, 2, 4, 8, 16]
    )
    # Perturbations to apply to root values.
    perturbations = 0.1 * random_state.random_sample(
        size=(3, len(root_params))
    )

    params = []
    # Parameter group 1
    # -----------------
    # No integer differences. This has been confirmed for the above seed.
    A = root_params + perturbations[0, :]
    B = root_params + perturbations[1, :]
    C = root_params + perturbations[2, :]
    params.extend(
        sorted(
            ((a, b, c, 1) for a, b, c in product(A, B, C)),
            key=lambda x: max(abs(x[0]), abs(x[1])),
        )
    )

    # Parameter group 2
    # -----------------
    # B - A an integer
    A = root_params + 0.5
    B = root_params + 0.5
    C = root_params + perturbations[1, :]
    params.extend(
        sorted(
            ((a, b, c, 2) for a, b, c in product(A, B, C)),
            key=lambda x: max(abs(x[0]), abs(x[1])),
        )
    )

    # Parameter group 3
    # -----------------
    # C - A an integer
    A = root_params + 0.5
    B = root_params + perturbations[1, :]
    C = root_params + 0.5
    params.extend(
        sorted(
            ((a, b, c, 3) for a, b, c in product(A, B, C)),
            key=lambda x: max(abs(x[0]), abs(x[1])),
        )
    )

    # Parameter group 4
    # -----------------
    # C - B an integer
    A = root_params + perturbations[0, :]
    B = root_params + 0.5
    C = root_params + 0.5
    params.extend(
        sorted(
            ((a, b, c, 4) for a, b, c in product(A, B, C)),
            key=lambda x: max(abs(x[0]), abs(x[1])),
        )
    )

    # Parameter group 5
    # -----------------
    # C - A - B an integer
    A = root_params + 0.25
    B = root_params + 0.25
    C = root_params + 0.5
    params.extend(
        sorted(
            ((a, b, c, 5) for a, b, c in product(A, B, C)),
            key=lambda x: max(abs(x[0]), abs(x[1])),
        )
    )

    # Parameter group 6
    # -----------------
    # A an integer
    A = root_params
    B = root_params + perturbations[0, :]
    C = root_params + perturbations[1, :]
    params.extend(
        sorted(
            ((a, b, c, 6) for a, b, c in product(A, B, C)),
            key=lambda x: max(abs(x[0]), abs(x[1])),
        )
    )

    # Parameter group 7
    # -----------------
    # B an integer
    A = root_params + perturbations[0, :]
    B = root_params
    C = root_params + perturbations[1, :]
    params.extend(
        sorted(
            ((a, b, c, 7) for a, b, c in product(A, B, C)),
            key=lambda x: max(abs(x[0]), abs(x[1])),
        )
    )

    # Parameter group 8
    # -----------------
    # C an integer
    A = root_params + perturbations[0, :]
    B = root_params + perturbations[1, :]
    C = root_params
    params.extend(
        sorted(
            ((a, b, c, 8) for a, b, c in product(A, B, C)),
            key=lambda x: max(abs(x[0]), abs(x[1])),
        )
    )

    # Parameter group 9
    # -----------------
    # Wide range of magnitudes, c - a - b > 0.
    phi = (1 + np.sqrt(5))/2
    P = phi**np.arange(16)
    P = np.hstack([-P, P])
    group_9_params = sorted(
        (
            (a, b, c, 9) for a, b, c in product(P, P, P) if c - a - b > 0
        ),
        key=lambda x: max(abs(x[0]), abs(x[1])),
    )

    if parameter_groups is not None:
        # Group 9 params only used if specified in arguments.
        params.extend(group_9_params)
        params = [
            (a, b, c, group) for a, b, c, group in params
            if group in parameter_groups
        ]

    # grid_size * grid_size grid in box with corners
    # -2 - 2j, -2 + 2j, 2 - 2j, 2 + 2j
    X, Y = np.meshgrid(
        np.linspace(-box_size, box_size, grid_size),
        np.linspace(-box_size, box_size, grid_size)
    )
    Z = X + Y * 1j
    Z = Z.flatten().tolist()
    # Add z = 1 + 0j (region 0).
    Z.append(1 + 0j)
    if regions is not None:
        Z = [z for z in Z if get_region(z) in regions]

    # Evaluate scipy and mpmath's hyp2f1 for all parameter combinations
    # above against all arguments in the grid Z
    rows = get_results(params, Z, n_jobs=n_jobs, compute_mp=compute_mp)

    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "a",
                "b",
                "c",
                "z",
                "|z|",
                "region",
                "parameter_group",
                "expected",  # mpmath's hyp2f1
                "observed",  # scipy's hyp2f1
                "relative_error",
                "absolute_error",
            ]
        )
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test scipy's hyp2f1 against mpmath's on a grid in the"
        " complex plane over a grid of parameter values. Saves output to file"
        " specified in positional argument \"outpath\"."
        " Caution: With default arguments, the generated output file is"
        " roughly 700MB in size. Script may take several hours to finish if"
        " \"--n_jobs\" is set to 1."
    )
    parser.add_argument(
        "outpath", type=str, help="Path to output tsv file."
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of jobs for multiprocessing.",
    )
    parser.add_argument(
        "--box_size",
        type=float,
        default=2.0,
        help="hyp2f1 is evaluated in box of side_length 2*box_size centered"
        " at the origin."
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=20,
        help="hyp2f1 is evaluated on grid_size * grid_size grid in box of side"
        " length 2*box_size centered at the origin."
    )
    parser.add_argument(
        "--parameter_groups",
        type=int,
        nargs='+',
        default=None,
        help="Restrict to supplied parameter groups. See the Docstring for"
        " this module for more info on parameter groups. Calculate for all"
        " parameter groups by default."
    )
    parser.add_argument(
        "--regions",
        type=int,
        nargs='+',
        default=None,
        help="Restrict to argument z only within the supplied regions. See"
        " the Docstring for this module for more info on regions. Calculate"
        " for all regions by default."
    )
    parser.add_argument(
        "--no_mp",
        action='store_true',
        help="If this flag is set, do not compute results with mpmath. Saves"
        " time if results have already been computed elsewhere. Fills in"
        " \"expected\" column with None values."
    )
    args = parser.parse_args()
    compute_mp = not args.no_mp
    print(args.parameter_groups)
    main(
        args.outpath,
        n_jobs=args.n_jobs,
        box_size=args.box_size,
        grid_size=args.grid_size,
        parameter_groups=args.parameter_groups,
        regions=args.regions,
        compute_mp=compute_mp,
    )
<|endoftext|>"""Precompute the polynomials for the asymptotic expansion of the
generalized exponential integral.

Sources
-------
[1] NIST, Digital Library of Mathematical Functions,
    https://dlmf.nist.gov/8.20#ii

"""
import os

try:
    import sympy
    from sympy import Poly
    x = sympy.symbols('x')
except ImportError:
    pass


def generate_A(K):
    A = [Poly(1, x)]
    for k in range(K):
        A.append(Poly(1 - 2*k*x, x)*A[k] + Poly(x*(x + 1))*A[k].diff())
    return A


WARNING = """\
/* This file was automatically generated by _precompute/expn_asy.py.
 * Do not edit it manually!
 */
"""


def main():
    print(__doc__)
    fn = os.path.join('..', 'cephes', 'expn.h')

    K = 12
    A = generate_A(K)
    with open(fn + '.new', 'w') as f:
        f.write(WARNING)
        f.write(f"#define nA {len(A)}\n")
        for k, Ak in enumerate(A):
            ', '.join([str(x.evalf(18)) for x in Ak.coeffs()])
            f.write(f"static const double A{k}[] = {{tmp}};\n")
        ", ".join([f"A{k}" for k in range(K + 1)])
        f.write("static const double *A[] = {{tmp}};\n")
        ", ".join([str(Ak.degree()) for Ak in A])
        f.write("static const int Adegs[] = {{tmp}};\n")
    os.rename(fn + '.new', fn)


if __name__ == "__main__":
    main()
<|endoftext|>"""
Precompute coefficients of Temme's asymptotic expansion for gammainc.

This takes about 8 hours to run on a 2.3 GHz Macbook Pro with 4GB ram.

Sources:
[1] NIST, "Digital Library of Mathematical Functions",
    https://dlmf.nist.gov/

"""
import os
from scipy.special._precompute.utils import lagrange_inversion

try:
    import mpmath as mp
except ImportError:
    pass


def compute_a(n):
    """a_k from DLMF 5.11.6"""
    a = [mp.sqrt(2)/2]
    for k in range(1, n):
        ak = a[-1]/k
        for j in range(1, len(a)):
            ak -= a[j]*a[-j]/(j + 1)
        ak /= a[0]*(1 + mp.mpf(1)/(k + 1))
        a.append(ak)
    return a


def compute_g(n):
    """g_k from DLMF 5.11.3/5.11.5"""
    a = compute_a(2*n)
    g = [mp.sqrt(2)*mp.rf(0.5, k)*a[2*k] for k in range(n)]
    return g


def eta(lam):
    """Function from DLMF 8.12.1 shifted to be centered at 0."""
    if lam > 0:
        return mp.sqrt(2*(lam - mp.log(lam + 1)))
    elif lam < 0:
        return -mp.sqrt(2*(lam - mp.log(lam + 1)))
    else:
        return 0


def compute_alpha(n):
    """alpha_n from DLMF 8.12.13"""
    coeffs = mp.taylor(eta, 0, n - 1)
    return lagrange_inversion(coeffs)


def compute_d(K, N):
    """d_{k, n} from DLMF 8.12.12"""
    M = N + 2*K
    d0 = [-mp.mpf(1)/3]
    alpha = compute_alpha(M + 2)
    for n in range(1, M):
        d0.append((n + 2)*alpha[n+2])
    d = [d0]
    g = compute_g(K)
    for k in range(1, K):
        dk = []
        for n in range(M - 2*k):
            dk.append((-1)**k*g[k]*d[0][n] + (n + 2)*d[k-1][n+2])
        d.append(dk)
    for k in range(K):
        d[k] = d[k][:N]
    return d


header = \
r"""/* This file was automatically generated by _precomp/gammainc.py.
 * Do not edit it manually!
 */

#ifndef IGAM_H
#define IGAM_H

#define K {}
#define N {}

static const double d[K][N] =
{{"""

footer = \
r"""
#endif
"""


def main():
    print(__doc__)
    K = 25
    N = 25
    with mp.workdps(50):
        d = compute_d(K, N)
    fn = os.path.join(os.path.dirname(__file__), '..', 'cephes', 'igam.h')
    with open(fn + '.new', 'w') as f:
        f.write(header.format(K, N))
        for k, row in enumerate(d):
            row = [mp.nstr(x, 17, min_fixed=0, max_fixed=0) for x in row]
            f.write('{')
            f.write(", ".join(row))
            if k < K - 1:
                f.write('},\n')
            else:
                f.write('}};\n')
        f.write(footer)
    os.rename(fn + '.new', fn)


if __name__ == "__main__":
    main()
<|endoftext|>import mpmath


def f(x):
    return (mpmath.pi + x + mpmath.sin(x)) / (2*mpmath.pi)


# Note: 40 digits might be overkill; a few more digits than the default
# might be sufficient.
mpmath.mp.dps = 40
ts = mpmath.taylor(f, -mpmath.pi, 20)
p, q = mpmath.pade(ts, 9, 10)

p = [float(c) for c in p]
q = [float(c) for c in q]
print('p =', p)
print('q =', q)
<|endoftext|>try:
    import mpmath as mp
except ImportError:
    pass

try:
    from sympy.abc import x
except ImportError:
    pass


def lagrange_inversion(a):
    """Given a series

    f(x) = a[1]*x + a[2]*x**2 + ... + a[n-1]*x**(n - 1),

    use the Lagrange inversion formula to compute a series

    g(x) = b[1]*x + b[2]*x**2 + ... + b[n-1]*x**(n - 1)

    so that f(g(x)) = g(f(x)) = x mod x**n. We must have a[0] = 0, so
    necessarily b[0] = 0 too.

    The algorithm is naive and could be improved, but speed isn't an
    issue here and it's easy to read.

    """
    n = len(a)
    f = sum(a[i]*x**i for i in range(n))
    h = (x/f).series(x, 0, n).removeO()
    hpower = [h**0]
    for k in range(n):
        hpower.append((hpower[-1]*h).expand())
    b = [mp.mpf(0)]
    for k in range(1, n):
        b.append(hpower[k].coeff(x, k - 1)/k)
    b = [mp.mpf(x) for x in b]
    return b
<|endoftext|>"""
Convergence regions of the expansions used in ``struve.c``

Note that for v >> z both functions tend rapidly to 0,
and for v << -z, they tend to infinity.

The floating-point functions over/underflow in the lower left and right
corners of the figure.


Figure legend
=============

Red region
    Power series is close (1e-12) to the mpmath result

Blue region
    Asymptotic series is close to the mpmath result

Green region
    Bessel series is close to the mpmath result

Dotted colored lines
    Boundaries of the regions

Solid colored lines
    Boundaries estimated by the routine itself. These will be used
    for determining which of the results to use.

Black dashed line
    The line z = 0.7*|v| + 12

"""
import numpy as np
import matplotlib.pyplot as plt

import mpmath


def err_metric(a, b, atol=1e-290):
    m = abs(a - b) / (atol + abs(b))
    m[np.isinf(b) & (a == b)] = 0
    return m


def do_plot(is_h=True):
    from scipy.special._ufuncs import (_struve_power_series,
                                       _struve_asymp_large_z,
                                       _struve_bessel_series)

    vs = np.linspace(-1000, 1000, 91)
    zs = np.sort(np.r_[1e-5, 1.0, np.linspace(0, 700, 91)[1:]])

    rp = _struve_power_series(vs[:,None], zs[None,:], is_h)
    ra = _struve_asymp_large_z(vs[:,None], zs[None,:], is_h)
    rb = _struve_bessel_series(vs[:,None], zs[None,:], is_h)

    mpmath.mp.dps = 50
    if is_h:
        def sh(v, z):
            return float(mpmath.struveh(mpmath.mpf(v), mpmath.mpf(z)))
    else:
        def sh(v, z):
            return float(mpmath.struvel(mpmath.mpf(v), mpmath.mpf(z)))
    ex = np.vectorize(sh, otypes='d')(vs[:,None], zs[None,:])

    err_a = err_metric(ra[0], ex) + 1e-300
    err_p = err_metric(rp[0], ex) + 1e-300
    err_b = err_metric(rb[0], ex) + 1e-300

    err_est_a = abs(ra[1]/ra[0])
    err_est_p = abs(rp[1]/rp[0])
    err_est_b = abs(rb[1]/rb[0])

    z_cutoff = 0.7*abs(vs) + 12

    levels = [-1000, -12]

    plt.cla()

    plt.hold(1)
    plt.contourf(vs, zs, np.log10(err_p).T,
                 levels=levels, colors=['r', 'r'], alpha=0.1)
    plt.contourf(vs, zs, np.log10(err_a).T,
                 levels=levels, colors=['b', 'b'], alpha=0.1)
    plt.contourf(vs, zs, np.log10(err_b).T,
                 levels=levels, colors=['g', 'g'], alpha=0.1)

    plt.contour(vs, zs, np.log10(err_p).T,
                levels=levels, colors=['r', 'r'], linestyles=[':', ':'])
    plt.contour(vs, zs, np.log10(err_a).T,
                levels=levels, colors=['b', 'b'], linestyles=[':', ':'])
    plt.contour(vs, zs, np.log10(err_b).T,
                levels=levels, colors=['g', 'g'], linestyles=[':', ':'])

    lp = plt.contour(vs, zs, np.log10(err_est_p).T,
                     levels=levels, colors=['r', 'r'], linestyles=['-', '-'])
    la = plt.contour(vs, zs, np.log10(err_est_a).T,
                     levels=levels, colors=['b', 'b'], linestyles=['-', '-'])
    lb = plt.contour(vs, zs, np.log10(err_est_b).T,
                     levels=levels, colors=['g', 'g'], linestyles=['-', '-'])

    plt.clabel(lp, fmt={-1000: 'P', -12: 'P'})
    plt.clabel(la, fmt={-1000: 'A', -12: 'A'})
    plt.clabel(lb, fmt={-1000: 'B', -12: 'B'})

    plt.plot(vs, z_cutoff, 'k--')

    plt.xlim(vs.min(), vs.max())
    plt.ylim(zs.min(), zs.max())

    plt.xlabel('v')
    plt.ylabel('z')


def main():
    plt.clf()
    plt.subplot(121)
    do_plot(True)
    plt.title('Struve H')

    plt.subplot(122)
    do_plot(False)
    plt.title('Struve L')

    plt.savefig('struve_convergence.png')
    plt.show()


if __name__ == "__main__":
    main()
<|endoftext|>"""Compute gammainc and gammaincc for large arguments and parameters
and save the values to data files for use in tests. We can't just
compare to mpmath's gammainc in test_mpmath.TestSystematic because it
would take too long.

Note that mpmath's gammainc is computed using hypercomb, but since it
doesn't allow the user to increase the maximum number of terms used in
the series it doesn't converge for many arguments. To get around this
we copy the mpmath implementation but use more terms.

This takes about 17 minutes to run on a 2.3 GHz Macbook Pro with 4GB
ram.

Sources:
[1] Fredrik Johansson and others. mpmath: a Python library for
    arbitrary-precision floating-point arithmetic (version 0.19),
    December 2013. http://mpmath.org/.

"""
import os
from time import time
import numpy as np
from numpy import pi

from scipy.special._mptestutils import mpf2float

try:
    import mpmath as mp
except ImportError:
    pass


def gammainc(a, x, dps=50, maxterms=10**8):
    """Compute gammainc exactly like mpmath does but allow for more
    summands in hypercomb. See

    mpmath/functions/expintegrals.py#L134

    in the mpmath GitHub repository.

    """
    with mp.workdps(dps):
        z, a, b = mp.mpf(a), mp.mpf(x), mp.mpf(x)
        G = [z]
        negb = mp.fneg(b, exact=True)

        def h(z):
            T1 = [mp.exp(negb), b, z], [1, z, -1], [], G, [1], [1+z], b
            return (T1,)

        res = mp.hypercomb(h, [z], maxterms=maxterms)
        return mpf2float(res)


def gammaincc(a, x, dps=50, maxterms=10**8):
    """Compute gammaincc exactly like mpmath does but allow for more
    terms in hypercomb. See

    mpmath/functions/expintegrals.py#L187

    in the mpmath GitHub repository.

    """
    with mp.workdps(dps):
        z, a = a, x

        if mp.isint(z):
            try:
                # mpmath has a fast integer path
                return mpf2float(mp.gammainc(z, a=a, regularized=True))
            except mp.libmp.NoConvergence:
                pass
        nega = mp.fneg(a, exact=True)
        G = [z]
        # Use 2F0 series when possible; fall back to lower gamma representation
        try:
            def h(z):
                r = z-1
                return [([mp.exp(nega), a], [1, r], [], G, [1, -r], [], 1/nega)]
            return mpf2float(mp.hypercomb(h, [z], force_series=True))
        except mp.libmp.NoConvergence:
            def h(z):
                T1 = [], [1, z-1], [z], G, [], [], 0
                T2 = [-mp.exp(nega), a, z], [1, z, -1], [], G, [1], [1+z], a
                return T1, T2
            return mpf2float(mp.hypercomb(h, [z], maxterms=maxterms))


def main():
    t0 = time()
    # It would be nice to have data for larger values, but either this
    # requires prohibitively large precision (dps > 800) or mpmath has
    # a bug. For example, gammainc(1e20, 1e20, dps=800) returns a
    # value around 0.03, while the true value should be close to 0.5
    # (DLMF 8.12.15).
    print(__doc__)
    pwd = os.path.dirname(__file__)
    r = np.logspace(4, 14, 30)
    ltheta = np.logspace(np.log10(pi/4), np.log10(np.arctan(0.6)), 30)
    utheta = np.logspace(np.log10(pi/4), np.log10(np.arctan(1.4)), 30)

    regimes = [(gammainc, ltheta), (gammaincc, utheta)]
    for func, theta in regimes:
        rg, thetag = np.meshgrid(r, theta)
        a, x = rg*np.cos(thetag), rg*np.sin(thetag)
        a, x = a.flatten(), x.flatten()
        dataset = []
        for i, (a0, x0) in enumerate(zip(a, x)):
            if func == gammaincc:
                # Exploit the fast integer path in gammaincc whenever
                # possible so that the computation doesn't take too
                # long
                a0, x0 = np.floor(a0), np.floor(x0)
            dataset.append((a0, x0, func(a0, x0)))
        dataset = np.array(dataset)
        filename = os.path.join(pwd, '..', 'tests', 'data', 'local',
                                f'{func.__name__}.txt')
        np.savetxt(filename, dataset)

    print(f"{(time() - t0)/60} minutes elapsed")


if __name__ == "__main__":
    main()
<|endoftext|>"""Compute the Taylor series for zeta(x) - 1 around x = 0."""
try:
    import mpmath
except ImportError:
    pass


def zetac_series(N):
    coeffs = []
    with mpmath.workdps(100):
        coeffs.append(-1.5)
        for n in range(1, N):
            coeff = mpmath.diff(mpmath.zeta, 0, n)/mpmath.factorial(n)
            coeffs.append(coeff)
    return coeffs


def main():
    print(__doc__)
    coeffs = zetac_series(10)
    coeffs = [mpmath.nstr(x, 20, min_fixed=0, max_fixed=0)
              for x in coeffs]
    print("\n".join(coeffs[::-1]))


if __name__ == '__main__':
    main()
<|endoftext|>"""Compute a grid of values for Wright's generalized Bessel function
and save the values to data files for use in tests. Using mpmath directly in
tests would take too long.

This takes about 10 minutes to run on a 2.7 GHz i7 Macbook Pro.
"""
from functools import lru_cache
import os
from time import time

import numpy as np
from scipy.special._mptestutils import mpf2float

try:
    import mpmath as mp
except ImportError:
    pass

# exp_inf: smallest value x for which exp(x) == inf
exp_inf = 709.78271289338403


# 64 Byte per value
@lru_cache(maxsize=100_000)
def rgamma_cached(x, dps):
    with mp.workdps(dps):
        return mp.rgamma(x)


def mp_wright_bessel(a, b, x, dps=50, maxterms=2000):
    """Compute Wright's generalized Bessel function as Series with mpmath.
    """
    with mp.workdps(dps):
        a, b, x = mp.mpf(a), mp.mpf(b), mp.mpf(x)
        res = mp.nsum(lambda k: x**k / mp.fac(k)
                      * rgamma_cached(a * k + b, dps=dps),
                      [0, mp.inf],
                      tol=dps, method='s', steps=[maxterms]
                      )
        return mpf2float(res)


def main():
    t0 = time()
    print(__doc__)
    pwd = os.path.dirname(__file__)
    eps = np.finfo(float).eps * 100

    a_range = np.array([eps,
                        1e-4 * (1 - eps), 1e-4, 1e-4 * (1 + eps),
                        1e-3 * (1 - eps), 1e-3, 1e-3 * (1 + eps),
                        0.1, 0.5,
                        1 * (1 - eps), 1, 1 * (1 + eps),
                        1.5, 2, 4.999, 5, 10])
    b_range = np.array([0, eps, 1e-10, 1e-5, 0.1, 1, 2, 10, 20, 100])
    x_range = np.array([0, eps, 1 - eps, 1, 1 + eps,
                        1.5,
                        2 - eps, 2, 2 + eps,
                        9 - eps, 9, 9 + eps,
                        10 * (1 - eps), 10, 10 * (1 + eps),
                        100 * (1 - eps), 100, 100 * (1 + eps),
                        500, exp_inf, 1e3, 1e5, 1e10, 1e20])

    a_range, b_range, x_range = np.meshgrid(a_range, b_range, x_range,
                                            indexing='ij')
    a_range = a_range.flatten()
    b_range = b_range.flatten()
    x_range = x_range.flatten()

    # filter out some values, especially too large x
    bool_filter = ~((a_range < 5e-3) & (x_range >= exp_inf))
    bool_filter = bool_filter & ~((a_range < 0.2) & (x_range > exp_inf))
    bool_filter = bool_filter & ~((a_range < 0.5) & (x_range > 1e3))
    bool_filter = bool_filter & ~((a_range < 0.56) & (x_range > 5e3))
    bool_filter = bool_filter & ~((a_range < 1) & (x_range > 1e4))
    bool_filter = bool_filter & ~((a_range < 1.4) & (x_range > 1e5))
    bool_filter = bool_filter & ~((a_range < 1.8) & (x_range > 1e6))
    bool_filter = bool_filter & ~((a_range < 2.2) & (x_range > 1e7))
    bool_filter = bool_filter & ~((a_range < 2.5) & (x_range > 1e8))
    bool_filter = bool_filter & ~((a_range < 2.9) & (x_range > 1e9))
    bool_filter = bool_filter & ~((a_range < 3.3) & (x_range > 1e10))
    bool_filter = bool_filter & ~((a_range < 3.7) & (x_range > 1e11))
    bool_filter = bool_filter & ~((a_range < 4) & (x_range > 1e12))
    bool_filter = bool_filter & ~((a_range < 4.4) & (x_range > 1e13))
    bool_filter = bool_filter & ~((a_range < 4.7) & (x_range > 1e14))
    bool_filter = bool_filter & ~((a_range < 5.1) & (x_range > 1e15))
    bool_filter = bool_filter & ~((a_range < 5.4) & (x_range > 1e16))
    bool_filter = bool_filter & ~((a_range < 5.8) & (x_range > 1e17))
    bool_filter = bool_filter & ~((a_range < 6.2) & (x_range > 1e18))
    bool_filter = bool_filter & ~((a_range < 6.2) & (x_range > 1e18))
    bool_filter = bool_filter & ~((a_range < 6.5) & (x_range > 1e19))
    bool_filter = bool_filter & ~((a_range < 6.9) & (x_range > 1e20))

    # filter out known values that do not meet the required numerical accuracy
    # see test test_wright_data_grid_failures
    failing = np.array([
        [0.1, 100, 709.7827128933841],
        [0.5, 10, 709.7827128933841],
        [0.5, 10, 1000],
        [0.5, 100, 1000],
        [1, 20, 100000],
        [1, 100, 100000],
        [1.0000000000000222, 20, 100000],
        [1.0000000000000222, 100, 100000],
        [1.5, 0, 500],
        [1.5, 2.220446049250313e-14, 500],
        [1.5, 1.e-10, 500],
        [1.5, 1.e-05, 500],
        [1.5, 0.1, 500],
        [1.5, 20, 100000],
        [1.5, 100, 100000],
        ]).tolist()

    does_fail = np.full_like(a_range, False, dtype=bool)
    for i in range(x_range.size):
        if [a_range[i], b_range[i], x_range[i]] in failing:
            does_fail[i] = True

    # filter and flatten
    a_range = a_range[bool_filter]
    b_range = b_range[bool_filter]
    x_range = x_range[bool_filter]
    does_fail = does_fail[bool_filter]

    dataset = []
    print(f"Computing {x_range.size} single points.")
    print("Tests will fail for the following data points:")
    for i in range(x_range.size):
        a = a_range[i]
        b = b_range[i]
        x = x_range[i]
        # take care of difficult corner cases
        maxterms = 1000
        if a < 1e-6 and x >= exp_inf/10:
            maxterms = 2000
        f = mp_wright_bessel(a, b, x, maxterms=maxterms)
        if does_fail[i]:
            print("failing data point a, b, x, value = "
                  f"[{a}, {b}, {x}, {f}]")
        else:
            dataset.append((a, b, x, f))
    dataset = np.array(dataset)

    filename = os.path.join(pwd, '..', 'tests', 'data', 'local',
                            'wright_bessel.txt')
    np.savetxt(filename, dataset)

    print(f"{(time() - t0)/60:.1f} minutes elapsed")


if __name__ == "__main__":
    main()
<|endoftext|>import numpy as np

try:
    import mpmath
except ImportError:
    pass


def mpmath_wrightomega(x):
    return mpmath.lambertw(mpmath.exp(x), mpmath.mpf('-0.5'))


def wrightomega_series_error(x):
    series = x
    desired = mpmath_wrightomega(x)
    return abs(series - desired) / desired


def wrightomega_exp_error(x):
    exponential_approx = mpmath.exp(x)
    desired = mpmath_wrightomega(x)
    return abs(exponential_approx - desired) / desired


def main():
    desired_error = 2 * np.finfo(float).eps
    print('Series Error')
    for x in [1e5, 1e10, 1e15, 1e20]:
        with mpmath.workdps(100):
            error = wrightomega_series_error(x)
        print(x, error, error < desired_error)

    print('Exp error')
    for x in [-10, -25, -50, -100, -200, -400, -700, -740]:
        with mpmath.workdps(100):
            error = wrightomega_exp_error(x)
        print(x, error, error < desired_error)


if __name__ == '__main__':
    main()
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.

from scipy._lib.deprecation import _sub_module_deprecation

__all__: list[str] = []


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="integrate", module="dop",
                                   private_modules=["_dop"], all=__all__,
                                   attribute=name)
<|endoftext|>"""Boundary value problem solver."""
from warnings import warn

import numpy as np
from numpy.linalg import pinv

from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import OptimizeResult


EPS = np.finfo(float).eps


def estimate_fun_jac(fun, x, y, p, f0=None):
    """Estimate derivatives of an ODE system rhs with forward differences.

    Returns
    -------
    df_dy : ndarray, shape (n, n, m)
        Derivatives with respect to y. An element (i, j, q) corresponds to
        d f_i(x_q, y_q) / d (y_q)_j.
    df_dp : ndarray with shape (n, k, m) or None
        Derivatives with respect to p. An element (i, j, q) corresponds to
        d f_i(x_q, y_q, p) / d p_j. If `p` is empty, None is returned.
    """
    n, m = y.shape
    if f0 is None:
        f0 = fun(x, y, p)

    dtype = y.dtype

    df_dy = np.empty((n, n, m), dtype=dtype)
    h = EPS**0.5 * (1 + np.abs(y))
    for i in range(n):
        y_new = y.copy()
        y_new[i] += h[i]
        hi = y_new[i] - y[i]
        f_new = fun(x, y_new, p)
        df_dy[:, i, :] = (f_new - f0) / hi

    k = p.shape[0]
    if k == 0:
        df_dp = None
    else:
        df_dp = np.empty((n, k, m), dtype=dtype)
        h = EPS**0.5 * (1 + np.abs(p))
        for i in range(k):
            p_new = p.copy()
            p_new[i] += h[i]
            hi = p_new[i] - p[i]
            f_new = fun(x, y, p_new)
            df_dp[:, i, :] = (f_new - f0) / hi

    return df_dy, df_dp


def estimate_bc_jac(bc, ya, yb, p, bc0=None):
    """Estimate derivatives of boundary conditions with forward differences.

    Returns
    -------
    dbc_dya : ndarray, shape (n + k, n)
        Derivatives with respect to ya. An element (i, j) corresponds to
        d bc_i / d ya_j.
    dbc_dyb : ndarray, shape (n + k, n)
        Derivatives with respect to yb. An element (i, j) corresponds to
        d bc_i / d ya_j.
    dbc_dp : ndarray with shape (n + k, k) or None
        Derivatives with respect to p. An element (i, j) corresponds to
        d bc_i / d p_j. If `p` is empty, None is returned.
    """
    n = ya.shape[0]
    k = p.shape[0]

    if bc0 is None:
        bc0 = bc(ya, yb, p)

    dtype = ya.dtype

    dbc_dya = np.empty((n, n + k), dtype=dtype)
    h = EPS**0.5 * (1 + np.abs(ya))
    for i in range(n):
        ya_new = ya.copy()
        ya_new[i] += h[i]
        hi = ya_new[i] - ya[i]
        bc_new = bc(ya_new, yb, p)
        dbc_dya[i] = (bc_new - bc0) / hi
    dbc_dya = dbc_dya.T

    h = EPS**0.5 * (1 + np.abs(yb))
    dbc_dyb = np.empty((n, n + k), dtype=dtype)
    for i in range(n):
        yb_new = yb.copy()
        yb_new[i] += h[i]
        hi = yb_new[i] - yb[i]
        bc_new = bc(ya, yb_new, p)
        dbc_dyb[i] = (bc_new - bc0) / hi
    dbc_dyb = dbc_dyb.T

    if k == 0:
        dbc_dp = None
    else:
        h = EPS**0.5 * (1 + np.abs(p))
        dbc_dp = np.empty((k, n + k), dtype=dtype)
        for i in range(k):
            p_new = p.copy()
            p_new[i] += h[i]
            hi = p_new[i] - p[i]
            bc_new = bc(ya, yb, p_new)
            dbc_dp[i] = (bc_new - bc0) / hi
        dbc_dp = dbc_dp.T

    return dbc_dya, dbc_dyb, dbc_dp


def compute_jac_indices(n, m, k):
    """Compute indices for the collocation system Jacobian construction.

    See `construct_global_jac` for the explanation.
    """
    i_col = np.repeat(np.arange((m - 1) * n), n)
    j_col = (np.tile(np.arange(n), n * (m - 1)) +
             np.repeat(np.arange(m - 1) * n, n**2))

    i_bc = np.repeat(np.arange((m - 1) * n, m * n + k), n)
    j_bc = np.tile(np.arange(n), n + k)

    i_p_col = np.repeat(np.arange((m - 1) * n), k)
    j_p_col = np.tile(np.arange(m * n, m * n + k), (m - 1) * n)

    i_p_bc = np.repeat(np.arange((m - 1) * n, m * n + k), k)
    j_p_bc = np.tile(np.arange(m * n, m * n + k), n + k)

    i = np.hstack((i_col, i_col, i_bc, i_bc, i_p_col, i_p_bc))
    j = np.hstack((j_col, j_col + n,
                   j_bc, j_bc + (m - 1) * n,
                   j_p_col, j_p_bc))

    return i, j


def stacked_matmul(a, b):
    """Stacked matrix multiply: out[i,:,:] = np.dot(a[i,:,:], b[i,:,:]).

    Empirical optimization. Use outer Python loop and BLAS for large
    matrices, otherwise use a single einsum call.
    """
    if a.shape[1] > 50:
        out = np.empty((a.shape[0], a.shape[1], b.shape[2]))
        for i in range(a.shape[0]):
            out[i] = np.dot(a[i], b[i])
        return out
    else:
        return np.einsum('...ij,...jk->...ik', a, b)


def construct_global_jac(n, m, k, i_jac, j_jac, h, df_dy, df_dy_middle, df_dp,
                         df_dp_middle, dbc_dya, dbc_dyb, dbc_dp):
    """Construct the Jacobian of the collocation system.

    There are n * m + k functions: m - 1 collocations residuals, each
    containing n components, followed by n + k boundary condition residuals.

    There are n * m + k variables: m vectors of y, each containing n
    components, followed by k values of vector p.

    For example, let m = 4, n = 2 and k = 1, then the Jacobian will have
    the following sparsity structure:

        1 1 2 2 0 0 0 0  5
        1 1 2 2 0 0 0 0  5
        0 0 1 1 2 2 0 0  5
        0 0 1 1 2 2 0 0  5
        0 0 0 0 1 1 2 2  5
        0 0 0 0 1 1 2 2  5

        3 3 0 0 0 0 4 4  6
        3 3 0 0 0 0 4 4  6
        3 3 0 0 0 0 4 4  6

    Zeros denote identically zero values, other values denote different kinds
    of blocks in the matrix (see below). The blank row indicates the separation
    of collocation residuals from boundary conditions. And the blank column
    indicates the separation of y values from p values.

    Refer to [1]_  (p. 306) for the formula of n x n blocks for derivatives
    of collocation residuals with respect to y.

    Parameters
    ----------
    n : int
        Number of equations in the ODE system.
    m : int
        Number of nodes in the mesh.
    k : int
        Number of the unknown parameters.
    i_jac, j_jac : ndarray
        Row and column indices returned by `compute_jac_indices`. They
        represent different blocks in the Jacobian matrix in the following
        order (see the scheme above):

            * 1: m - 1 diagonal n x n blocks for the collocation residuals.
            * 2: m - 1 off-diagonal n x n blocks for the collocation residuals.
            * 3 : (n + k) x n block for the dependency of the boundary
              conditions on ya.
            * 4: (n + k) x n block for the dependency of the boundary
              conditions on yb.
            * 5: (m - 1) * n x k block for the dependency of the collocation
              residuals on p.
            * 6: (n + k) x k block for the dependency of the boundary
              conditions on p.

    df_dy : ndarray, shape (n, n, m)
        Jacobian of f with respect to y computed at the mesh nodes.
    df_dy_middle : ndarray, shape (n, n, m - 1)
        Jacobian of f with respect to y computed at the middle between the
        mesh nodes.
    df_dp : ndarray with shape (n, k, m) or None
        Jacobian of f with respect to p computed at the mesh nodes.
    df_dp_middle : ndarray with shape (n, k, m - 1) or None
        Jacobian of f with respect to p computed at the middle between the
        mesh nodes.
    dbc_dya, dbc_dyb : ndarray, shape (n, n)
        Jacobian of bc with respect to ya and yb.
    dbc_dp : ndarray with shape (n, k) or None
        Jacobian of bc with respect to p.

    Returns
    -------
    J : csc_matrix, shape (n * m + k, n * m + k)
        Jacobian of the collocation system in a sparse form.

    References
    ----------
    .. [1] J. Kierzenka, L. F. Shampine, "A BVP Solver Based on Residual
       Control and the Maltab PSE", ACM Trans. Math. Softw., Vol. 27,
       Number 3, pp. 299-316, 2001.
    """
    df_dy = np.transpose(df_dy, (2, 0, 1))
    df_dy_middle = np.transpose(df_dy_middle, (2, 0, 1))

    h = h[:, np.newaxis, np.newaxis]

    dtype = df_dy.dtype

    # Computing diagonal n x n blocks.
    dPhi_dy_0 = np.empty((m - 1, n, n), dtype=dtype)
    dPhi_dy_0[:] = -np.identity(n)
    dPhi_dy_0 -= h / 6 * (df_dy[:-1] + 2 * df_dy_middle)
    T = stacked_matmul(df_dy_middle, df_dy[:-1])
    dPhi_dy_0 -= h**2 / 12 * T

    # Computing off-diagonal n x n blocks.
    dPhi_dy_1 = np.empty((m - 1, n, n), dtype=dtype)
    dPhi_dy_1[:] = np.identity(n)
    dPhi_dy_1 -= h / 6 * (df_dy[1:] + 2 * df_dy_middle)
    T = stacked_matmul(df_dy_middle, df_dy[1:])
    dPhi_dy_1 += h**2 / 12 * T

    values = np.hstack((dPhi_dy_0.ravel(), dPhi_dy_1.ravel(), dbc_dya.ravel(),
                        dbc_dyb.ravel()))

    if k > 0:
        df_dp = np.transpose(df_dp, (2, 0, 1))
        df_dp_middle = np.transpose(df_dp_middle, (2, 0, 1))
        T = stacked_matmul(df_dy_middle, df_dp[:-1] - df_dp[1:])
        df_dp_middle += 0.125 * h * T
        dPhi_dp = -h/6 * (df_dp[:-1] + df_dp[1:] + 4 * df_dp_middle)
        values = np.hstack((values, dPhi_dp.ravel(), dbc_dp.ravel()))

    J = coo_matrix((values, (i_jac, j_jac)))
    return csc_matrix(J)


def collocation_fun(fun, y, p, x, h):
    """Evaluate collocation residuals.

    This function lies in the core of the method. The solution is sought
    as a cubic C1 continuous spline with derivatives matching the ODE rhs
    at given nodes `x`. Collocation conditions are formed from the equality
    of the spline derivatives and rhs of the ODE system in the middle points
    between nodes.

    Such method is classified to Lobbato IIIA family in ODE literature.
    Refer to [1]_ for the formula and some discussion.

    Returns
    -------
    col_res : ndarray, shape (n, m - 1)
        Collocation residuals at the middle points of the mesh intervals.
    y_middle : ndarray, shape (n, m - 1)
        Values of the cubic spline evaluated at the middle points of the mesh
        intervals.
    f : ndarray, shape (n, m)
        RHS of the ODE system evaluated at the mesh nodes.
    f_middle : ndarray, shape (n, m - 1)
        RHS of the ODE system evaluated at the middle points of the mesh
        intervals (and using `y_middle`).

    References
    ----------
    .. [1] J. Kierzenka, L. F. Shampine, "A BVP Solver Based on Residual
           Control and the Maltab PSE", ACM Trans. Math. Softw., Vol. 27,
           Number 3, pp. 299-316, 2001.
    """
    f = fun(x, y, p)
    y_middle = (0.5 * (y[:, 1:] + y[:, :-1]) -
                0.125 * h * (f[:, 1:] - f[:, :-1]))
    f_middle = fun(x[:-1] + 0.5 * h, y_middle, p)
    col_res = y[:, 1:] - y[:, :-1] - h / 6 * (f[:, :-1] + f[:, 1:] +
                                              4 * f_middle)

    return col_res, y_middle, f, f_middle


def prepare_sys(n, m, k, fun, bc, fun_jac, bc_jac, x, h):
    """Create the function and the Jacobian for the collocation system."""
    x_middle = x[:-1] + 0.5 * h
    i_jac, j_jac = compute_jac_indices(n, m, k)

    def col_fun(y, p):
        return collocation_fun(fun, y, p, x, h)

    def sys_jac(y, p, y_middle, f, f_middle, bc0):
        if fun_jac is None:
            df_dy, df_dp = estimate_fun_jac(fun, x, y, p, f)
            df_dy_middle, df_dp_middle = estimate_fun_jac(
                fun, x_middle, y_middle, p, f_middle)
        else:
            df_dy, df_dp = fun_jac(x, y, p)
            df_dy_middle, df_dp_middle = fun_jac(x_middle, y_middle, p)

        if bc_jac is None:
            dbc_dya, dbc_dyb, dbc_dp = estimate_bc_jac(bc, y[:, 0], y[:, -1],
                                                       p, bc0)
        else:
            dbc_dya, dbc_dyb, dbc_dp = bc_jac(y[:, 0], y[:, -1], p)

        return construct_global_jac(n, m, k, i_jac, j_jac, h, df_dy,
                                    df_dy_middle, df_dp, df_dp_middle, dbc_dya,
                                    dbc_dyb, dbc_dp)

    return col_fun, sys_jac


def solve_newton(n, m, h, col_fun, bc, jac, y, p, B, bvp_tol, bc_tol):
    """Solve the nonlinear collocation system by a Newton method.

    This is a simple Newton method with a backtracking line search. As
    advised in [1]_, an affine-invariant criterion function F = ||J^-1 r||^2
    is used, where J is the Jacobian matrix at the current iteration and r is
    the vector or collocation residuals (values of the system lhs).

    The method alters between full Newton iterations and the fixed-Jacobian
    iterations based

    There are other tricks proposed in [1]_, but they are not used as they
    don't seem to improve anything significantly, and even break the
    convergence on some test problems I tried.

    All important parameters of the algorithm are defined inside the function.

    Parameters
    ----------
    n : int
        Number of equations in the ODE system.
    m : int
        Number of nodes in the mesh.
    h : ndarray, shape (m-1,)
        Mesh intervals.
    col_fun : callable
        Function computing collocation residuals.
    bc : callable
        Function computing boundary condition residuals.
    jac : callable
        Function computing the Jacobian of the whole system (including
        collocation and boundary condition residuals). It is supposed to
        return csc_matrix.
    y : ndarray, shape (n, m)
        Initial guess for the function values at the mesh nodes.
    p : ndarray, shape (k,)
        Initial guess for the unknown parameters.
    B : ndarray with shape (n, n) or None
        Matrix to force the S y(a) = 0 condition for a problems with the
        singular term. If None, the singular term is assumed to be absent.
    bvp_tol : float
        Tolerance to which we want to solve a BVP.
    bc_tol : float
        Tolerance to which we want to satisfy the boundary conditions.

    Returns
    -------
    y : ndarray, shape (n, m)
        Final iterate for the function values at the mesh nodes.
    p : ndarray, shape (k,)
        Final iterate for the unknown parameters.
    singular : bool
        True, if the LU decomposition failed because Jacobian turned out
        to be singular.

    References
    ----------
    .. [1]  U. Ascher, R. Mattheij and R. Russell "Numerical Solution of
       Boundary Value Problems for Ordinary Differential Equations"
    """
    # We know that the solution residuals at the middle points of the mesh
    # are connected with collocation residuals  r_middle = 1.5 * col_res / h.
    # As our BVP solver tries to decrease relative residuals below a certain
    # tolerance, it seems reasonable to terminated Newton iterations by
    # comparison of r_middle / (1 + np.abs(f_middle)) with a certain threshold,
    # which we choose to be 1.5 orders lower than the BVP tolerance. We rewrite
    # the condition as col_res < tol_r * (1 + np.abs(f_middle)), then tol_r
    # should be computed as follows:
    tol_r = 2/3 * h * 5e-2 * bvp_tol

    # Maximum allowed number of Jacobian evaluation and factorization, in
    # other words, the maximum number of full Newton iterations. A small value
    # is recommended in the literature.
    max_njev = 4

    # Maximum number of iterations, considering that some of them can be
    # performed with the fixed Jacobian. In theory, such iterations are cheap,
    # but it's not that simple in Python.
    max_iter = 8

    # Minimum relative improvement of the criterion function to accept the
    # step (Armijo constant).
    sigma = 0.2

    # Step size decrease factor for backtracking.
    tau = 0.5

    # Maximum number of backtracking steps, the minimum step is then
    # tau ** n_trial.
    n_trial = 4

    col_res, y_middle, f, f_middle = col_fun(y, p)
    bc_res = bc(y[:, 0], y[:, -1], p)
    res = np.hstack((col_res.ravel(order='F'), bc_res))

    njev = 0
    singular = False
    recompute_jac = True
    for iteration in range(max_iter):
        if recompute_jac:
            J = jac(y, p, y_middle, f, f_middle, bc_res)
            njev += 1
            try:
                LU = splu(J)
            except RuntimeError:
                singular = True
                break

            step = LU.solve(res)
            cost = np.dot(step, step)

        y_step = step[:m * n].reshape((n, m), order='F')
        p_step = step[m * n:]

        alpha = 1
        for trial in range(n_trial + 1):
            y_new = y - alpha * y_step
            if B is not None:
                y_new[:, 0] = np.dot(B, y_new[:, 0])
            p_new = p - alpha * p_step

            col_res, y_middle, f, f_middle = col_fun(y_new, p_new)
            bc_res = bc(y_new[:, 0], y_new[:, -1], p_new)
            res = np.hstack((col_res.ravel(order='F'), bc_res))

            step_new = LU.solve(res)
            cost_new = np.dot(step_new, step_new)
            if cost_new < (1 - 2 * alpha * sigma) * cost:
                break

            if trial < n_trial:
                alpha *= tau

        y = y_new
        p = p_new

        if njev == max_njev:
            break

        if (np.all(np.abs(col_res) < tol_r * (1 + np.abs(f_middle))) and
                np.all(np.abs(bc_res) < bc_tol)):
            break

        # If the full step was taken, then we are going to continue with
        # the same Jacobian. This is the approach of BVP_SOLVER.
        if alpha == 1:
            step = step_new
            cost = cost_new
            recompute_jac = False
        else:
            recompute_jac = True

    return y, p, singular


def print_iteration_header():
    print(f"{'Iteration':^15}{'Max residual':^15}{'Max BC residual':^15}"
          f"{'Total nodes':^15}{'Nodes added':^15}")


def print_iteration_progress(iteration, residual, bc_residual, total_nodes,
                             nodes_added):
    print(f"{iteration:^15}{residual:^15.2e}{bc_residual:^15.2e}"
          f"{total_nodes:^15}{nodes_added:^15}")


class BVPResult(OptimizeResult):
    pass


TERMINATION_MESSAGES = {
    0: "The algorithm converged to the desired accuracy.",
    1: "The maximum number of mesh nodes is exceeded.",
    2: "A singular Jacobian encountered when solving the collocation system.",
    3: "The solver was unable to satisfy boundary conditions tolerance on iteration 10."
}


def estimate_rms_residuals(fun, sol, x, h, p, r_middle, f_middle):
    """Estimate rms values of collocation residuals using Lobatto quadrature.

    The residuals are defined as the difference between the derivatives of
    our solution and rhs of the ODE system. We use relative residuals, i.e.,
    normalized by 1 + np.abs(f). RMS values are computed as sqrt from the
    normalized integrals of the squared relative residuals over each interval.
    Integrals are estimated using 5-point Lobatto quadrature [1]_, we use the
    fact that residuals at the mesh nodes are identically zero.

    In [2] they don't normalize integrals by interval lengths, which gives
    a higher rate of convergence of the residuals by the factor of h**0.5.
    I chose to do such normalization for an ease of interpretation of return
    values as RMS estimates.

    Returns
    -------
    rms_res : ndarray, shape (m - 1,)
        Estimated rms values of the relative residuals over each interval.

    References
    ----------
    .. [1] http://mathworld.wolfram.com/LobattoQuadrature.html
    .. [2] J. Kierzenka, L. F. Shampine, "A BVP Solver Based on Residual
       Control and the Maltab PSE", ACM Trans. Math. Softw., Vol. 27,
       Number 3, pp. 299-316, 2001.
    """
    x_middle = x[:-1] + 0.5 * h
    s = 0.5 * h * (3/7)**0.5
    x1 = x_middle + s
    x2 = x_middle - s
    y1 = sol(x1)
    y2 = sol(x2)
    y1_prime = sol(x1, 1)
    y2_prime = sol(x2, 1)
    f1 = fun(x1, y1, p)
    f2 = fun(x2, y2, p)
    r1 = y1_prime - f1
    r2 = y2_prime - f2

    r_middle /= 1 + np.abs(f_middle)
    r1 /= 1 + np.abs(f1)
    r2 /= 1 + np.abs(f2)

    r1 = np.sum(np.real(r1 * np.conj(r1)), axis=0)
    r2 = np.sum(np.real(r2 * np.conj(r2)), axis=0)
    r_middle = np.sum(np.real(r_middle * np.conj(r_middle)), axis=0)

    return (0.5 * (32 / 45 * r_middle + 49 / 90 * (r1 + r2))) ** 0.5


def create_spline(y, yp, x, h):
    """Create a cubic spline given values and derivatives.

    Formulas for the coefficients are taken from interpolate.CubicSpline.

    Returns
    -------
    sol : PPoly
        Constructed spline as a PPoly instance.
    """
    from scipy.interpolate import PPoly

    n, m = y.shape
    c = np.empty((4, n, m - 1), dtype=y.dtype)
    slope = (y[:, 1:] - y[:, :-1]) / h
    t = (yp[:, :-1] + yp[:, 1:] - 2 * slope) / h
    c[0] = t / h
    c[1] = (slope - yp[:, :-1]) / h - t
    c[2] = yp[:, :-1]
    c[3] = y[:, :-1]
    c = np.moveaxis(c, 1, 0)

    return PPoly(c, x, extrapolate=True, axis=1)


def modify_mesh(x, insert_1, insert_2):
    """Insert nodes into a mesh.

    Nodes removal logic is not established, its impact on the solver is
    presumably negligible. So, only insertion is done in this function.

    Parameters
    ----------
    x : ndarray, shape (m,)
        Mesh nodes.
    insert_1 : ndarray
        Intervals to each insert 1 new node in the middle.
    insert_2 : ndarray
        Intervals to each insert 2 new nodes, such that divide an interval
        into 3 equal parts.

    Returns
    -------
    x_new : ndarray
        New mesh nodes.

    Notes
    -----
    `insert_1` and `insert_2` should not have common values.
    """
    # Because np.insert implementation apparently varies with a version of
    # NumPy, we use a simple and reliable approach with sorting.
    return np.sort(np.hstack((
        x,
        0.5 * (x[insert_1] + x[insert_1 + 1]),
        (2 * x[insert_2] + x[insert_2 + 1]) / 3,
        (x[insert_2] + 2 * x[insert_2 + 1]) / 3
    )))


def wrap_functions(fun, bc, fun_jac, bc_jac, k, a, S, D, dtype):
    """Wrap functions for unified usage in the solver."""
    if fun_jac is None:
        fun_jac_wrapped = None

    if bc_jac is None:
        bc_jac_wrapped = None

    if k == 0:
        def fun_p(x, y, _):
            return np.asarray(fun(x, y), dtype)

        def bc_wrapped(ya, yb, _):
            return np.asarray(bc(ya, yb), dtype)

        if fun_jac is not None:
            def fun_jac_p(x, y, _):
                return np.asarray(fun_jac(x, y), dtype), None

        if bc_jac is not None:
            def bc_jac_wrapped(ya, yb, _):
                dbc_dya, dbc_dyb = bc_jac(ya, yb)
                return (np.asarray(dbc_dya, dtype),
                        np.asarray(dbc_dyb, dtype), None)
    else:
        def fun_p(x, y, p):
            return np.asarray(fun(x, y, p), dtype)

        def bc_wrapped(x, y, p):
            return np.asarray(bc(x, y, p), dtype)

        if fun_jac is not None:
            def fun_jac_p(x, y, p):
                df_dy, df_dp = fun_jac(x, y, p)
                return np.asarray(df_dy, dtype), np.asarray(df_dp, dtype)

        if bc_jac is not None:
            def bc_jac_wrapped(ya, yb, p):
                dbc_dya, dbc_dyb, dbc_dp = bc_jac(ya, yb, p)
                return (np.asarray(dbc_dya, dtype), np.asarray(dbc_dyb, dtype),
                        np.asarray(dbc_dp, dtype))

    if S is None:
        fun_wrapped = fun_p
    else:
        def fun_wrapped(x, y, p):
            f = fun_p(x, y, p)
            if x[0] == a:
                f[:, 0] = np.dot(D, f[:, 0])
                f[:, 1:] += np.dot(S, y[:, 1:]) / (x[1:] - a)
            else:
                f += np.dot(S, y) / (x - a)
            return f

    if fun_jac is not None:
        if S is None:
            fun_jac_wrapped = fun_jac_p
        else:
            Sr = S[:, :, np.newaxis]

            def fun_jac_wrapped(x, y, p):
                df_dy, df_dp = fun_jac_p(x, y, p)
                if x[0] == a:
                    df_dy[:, :, 0] = np.dot(D, df_dy[:, :, 0])
                    df_dy[:, :, 1:] += Sr / (x[1:] - a)
                else:
                    df_dy += Sr / (x - a)

                return df_dy, df_dp

    return fun_wrapped, bc_wrapped, fun_jac_wrapped, bc_jac_wrapped


def solve_bvp(fun, bc, x, y, p=None, S=None, fun_jac=None, bc_jac=None,
              tol=1e-3, max_nodes=1000, verbose=0, bc_tol=None):
    """Solve a boundary value problem for a system of ODEs.

    This function numerically solves a first order system of ODEs subject to
    two-point boundary conditions::

        dy / dx = f(x, y, p) + S * y / (x - a), a <= x <= b
        bc(y(a), y(b), p) = 0

    Here x is a 1-D independent variable, y(x) is an n-D
    vector-valued function and p is a k-D vector of unknown
    parameters which is to be found along with y(x). For the problem to be
    determined, there must be n + k boundary conditions, i.e., bc must be an
    (n + k)-D function.

    The last singular term on the right-hand side of the system is optional.
    It is defined by an n-by-n matrix S, such that the solution must satisfy
    S y(a) = 0. This condition will be forced during iterations, so it must not
    contradict boundary conditions. See [2]_ for the explanation how this term
    is handled when solving BVPs numerically.

    Problems in a complex domain can be solved as well. In this case, y and p
    are considered to be complex, and f and bc are assumed to be complex-valued
    functions, but x stays real. Note that f and bc must be complex
    differentiable (satisfy Cauchy-Riemann equations [4]_), otherwise you
    should rewrite your problem for real and imaginary parts separately. To
    solve a problem in a complex domain, pass an initial guess for y with a
    complex data type (see below).

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(x, y)``,
        or ``fun(x, y, p)`` if parameters are present. All arguments are
        ndarray: ``x`` with shape (m,), ``y`` with shape (n, m), meaning that
        ``y[:, i]`` corresponds to ``x[i]``, and ``p`` with shape (k,). The
        return value must be an array with shape (n, m) and with the same
        layout as ``y``.
    bc : callable
        Function evaluating residuals of the boundary conditions. The calling
        signature is ``bc(ya, yb)``, or ``bc(ya, yb, p)`` if parameters are
        present. All arguments are ndarray: ``ya`` and ``yb`` with shape (n,),
        and ``p`` with shape (k,). The return value must be an array with
        shape (n + k,).
    x : array_like, shape (m,)
        Initial mesh. Must be a strictly increasing sequence of real numbers
        with ``x[0]=a`` and ``x[-1]=b``.
    y : array_like, shape (n, m)
        Initial guess for the function values at the mesh nodes, ith column
        corresponds to ``x[i]``. For problems in a complex domain pass `y`
        with a complex data type (even if the initial guess is purely real).
    p : array_like with shape (k,) or None, optional
        Initial guess for the unknown parameters. If None (default), it is
        assumed that the problem doesn't depend on any parameters.
    S : array_like with shape (n, n) or None
        Matrix defining the singular term. If None (default), the problem is
        solved without the singular term.
    fun_jac : callable or None, optional
        Function computing derivatives of f with respect to y and p. The
        calling signature is ``fun_jac(x, y)``, or ``fun_jac(x, y, p)`` if
        parameters are present. The return must contain 1 or 2 elements in the
        following order:

            * df_dy : array_like with shape (n, n, m), where an element
              (i, j, q) equals to d f_i(x_q, y_q, p) / d (y_q)_j.
            * df_dp : array_like with shape (n, k, m), where an element
              (i, j, q) equals to d f_i(x_q, y_q, p) / d p_j.

        Here q numbers nodes at which x and y are defined, whereas i and j
        number vector components. If the problem is solved without unknown
        parameters, df_dp should not be returned.

        If `fun_jac` is None (default), the derivatives will be estimated
        by the forward finite differences.
    bc_jac : callable or None, optional
        Function computing derivatives of bc with respect to ya, yb, and p.
        The calling signature is ``bc_jac(ya, yb)``, or ``bc_jac(ya, yb, p)``
        if parameters are present. The return must contain 2 or 3 elements in
        the following order:

            * dbc_dya : array_like with shape (n, n), where an element (i, j)
              equals to d bc_i(ya, yb, p) / d ya_j.
            * dbc_dyb : array_like with shape (n, n), where an element (i, j)
              equals to d bc_i(ya, yb, p) / d yb_j.
            * dbc_dp : array_like with shape (n, k), where an element (i, j)
              equals to d bc_i(ya, yb, p) / d p_j.

        If the problem is solved without unknown parameters, dbc_dp should not
        be returned.

        If `bc_jac` is None (default), the derivatives will be estimated by
        the forward finite differences.
    tol : float, optional
        Desired tolerance of the solution. If we define ``r = y' - f(x, y)``,
        where y is the found solution, then the solver tries to achieve on each
        mesh interval ``norm(r / (1 + abs(f)) < tol``, where ``norm`` is
        estimated in a root mean squared sense (using a numerical quadrature
        formula). Default is 1e-3.
    max_nodes : int, optional
        Maximum allowed number of the mesh nodes. If exceeded, the algorithm
        terminates. Default is 1000.
    verbose : {0, 1, 2}, optional
        Level of algorithm's verbosity:

            * 0 (default) : work silently.
            * 1 : display a termination report.
            * 2 : display progress during iterations.
    bc_tol : float, optional
        Desired absolute tolerance for the boundary condition residuals: `bc`
        value should satisfy ``abs(bc) < bc_tol`` component-wise.
        Equals to `tol` by default. Up to 10 iterations are allowed to achieve this
        tolerance.

    Returns
    -------
    Bunch object with the following fields defined:
    sol : PPoly
        Found solution for y as `scipy.interpolate.PPoly` instance, a C1
        continuous cubic spline.
    p : ndarray or None, shape (k,)
        Found parameters. None, if the parameters were not present in the
        problem.
    x : ndarray, shape (m,)
        Nodes of the final mesh.
    y : ndarray, shape (n, m)
        Solution values at the mesh nodes.
    yp : ndarray, shape (n, m)
        Solution derivatives at the mesh nodes.
    rms_residuals : ndarray, shape (m - 1,)
        RMS values of the relative residuals over each mesh interval (see the
        description of `tol` parameter).
    niter : int
        Number of completed iterations.
    status : int
        Reason for algorithm termination:

            * 0: The algorithm converged to the desired accuracy.
            * 1: The maximum number of mesh nodes is exceeded.
            * 2: A singular Jacobian encountered when solving the collocation
              system.

    message : string
        Verbal description of the termination reason.
    success : bool
        True if the algorithm converged to the desired accuracy (``status=0``).

    Notes
    -----
    This function implements a 4th order collocation algorithm with the
    control of residuals similar to [1]_. A collocation system is solved
    by a damped Newton method with an affine-invariant criterion function as
    described in [3]_.

    Note that in [1]_  integral residuals are defined without normalization
    by interval lengths. So, their definition is different by a multiplier of
    h**0.5 (h is an interval length) from the definition used here.

    .. versionadded:: 0.18.0

    References
    ----------
    .. [1] J. Kierzenka, L. F. Shampine, "A BVP Solver Based on Residual
           Control and the Maltab PSE", ACM Trans. Math. Softw., Vol. 27,
           Number 3, pp. 299-316, 2001.
    .. [2] L.F. Shampine, P. H. Muir and H. Xu, "A User-Friendly Fortran BVP
           Solver".
    .. [3] U. Ascher, R. Mattheij and R. Russell "Numerical Solution of
           Boundary Value Problems for Ordinary Differential Equations".
    .. [4] `Cauchy-Riemann equations
            <https://en.wikipedia.org/wiki/Cauchy-Riemann_equations>`_ on
            Wikipedia.

    Examples
    --------
    In the first example, we solve Bratu's problem::

        y'' + k * exp(y) = 0
        y(0) = y(1) = 0

    for k = 1.

    We rewrite the equation as a first-order system and implement its
    right-hand side evaluation::

        y1' = y2
        y2' = -exp(y1)

    >>> import numpy as np
    >>> def fun(x, y):
    ...     return np.vstack((y[1], -np.exp(y[0])))

    Implement evaluation of the boundary condition residuals:

    >>> def bc(ya, yb):
    ...     return np.array([ya[0], yb[0]])

    Define the initial mesh with 5 nodes:

    >>> x = np.linspace(0, 1, 5)

    This problem is known to have two solutions. To obtain both of them, we
    use two different initial guesses for y. We denote them by subscripts
    a and b.

    >>> y_a = np.zeros((2, x.size))
    >>> y_b = np.zeros((2, x.size))
    >>> y_b[0] = 3

    Now we are ready to run the solver.

    >>> from scipy.integrate import solve_bvp
    >>> res_a = solve_bvp(fun, bc, x, y_a)
    >>> res_b = solve_bvp(fun, bc, x, y_b)

    Let's plot the two found solutions. We take an advantage of having the
    solution in a spline form to produce a smooth plot.

    >>> x_plot = np.linspace(0, 1, 100)
    >>> y_plot_a = res_a.sol(x_plot)[0]
    >>> y_plot_b = res_b.sol(x_plot)[0]
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x_plot, y_plot_a, label='y_a')
    >>> plt.plot(x_plot, y_plot_b, label='y_b')
    >>> plt.legend()
    >>> plt.xlabel("x")
    >>> plt.ylabel("y")
    >>> plt.show()

    We see that the two solutions have similar shape, but differ in scale
    significantly.

    In the second example, we solve a simple Sturm-Liouville problem::

        y'' + k**2 * y = 0
        y(0) = y(1) = 0

    It is known that a non-trivial solution y = A * sin(k * x) is possible for
    k = pi * n, where n is an integer. To establish the normalization constant
    A = 1 we add a boundary condition::

        y'(0) = k

    Again, we rewrite our equation as a first-order system and implement its
    right-hand side evaluation::

        y1' = y2
        y2' = -k**2 * y1

    >>> def fun(x, y, p):
    ...     k = p[0]
    ...     return np.vstack((y[1], -k**2 * y[0]))

    Note that parameters p are passed as a vector (with one element in our
    case).

    Implement the boundary conditions:

    >>> def bc(ya, yb, p):
    ...     k = p[0]
    ...     return np.array([ya[0], yb[0], ya[1] - k])

    Set up the initial mesh and guess for y. We aim to find the solution for
    k = 2 * pi, to achieve that we set values of y to approximately follow
    sin(2 * pi * x):

    >>> x = np.linspace(0, 1, 5)
    >>> y = np.zeros((2, x.size))
    >>> y[0, 1] = 1
    >>> y[0, 3] = -1

    Run the solver with 6 as an initial guess for k.

    >>> sol = solve_bvp(fun, bc, x, y, p=[6])

    We see that the found k is approximately correct:

    >>> sol.p[0]
    6.28329460046

    And, finally, plot the solution to see the anticipated sinusoid:

    >>> x_plot = np.linspace(0, 1, 100)
    >>> y_plot = sol.sol(x_plot)[0]
    >>> plt.plot(x_plot, y_plot)
    >>> plt.xlabel("x")
    >>> plt.ylabel("y")
    >>> plt.show()
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("`x` must be 1 dimensional.")
    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("`x` must be strictly increasing.")
    a = x[0]

    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.complexfloating):
        dtype = complex
    else:
        dtype = float
    y = y.astype(dtype, copy=False)

    if y.ndim != 2:
        raise ValueError("`y` must be 2 dimensional.")
    if y.shape[1] != x.shape[0]:
        raise ValueError(f"`y` is expected to have {x.shape[0]} columns, but actually "
                         f"has {y.shape[1]}.")

    if p is None:
        p = np.array([])
    else:
        p = np.asarray(p, dtype=dtype)
    if p.ndim != 1:
        raise ValueError("`p` must be 1 dimensional.")

    if tol < 100 * EPS:
        warn(f"`tol` is too low, setting to {100 * EPS:.2e}", stacklevel=2)
        tol = 100 * EPS

    if verbose not in [0, 1, 2]:
        raise ValueError("`verbose` must be in [0, 1, 2].")

    n = y.shape[0]
    k = p.shape[0]

    if S is not None:
        S = np.asarray(S, dtype=dtype)
        if S.shape != (n, n):
            raise ValueError(f"`S` is expected to have shape {(n, n)}, "
                             f"but actually has {S.shape}")

        # Compute I - S^+ S to impose necessary boundary conditions.
        B = np.identity(n) - np.dot(pinv(S), S)

        y[:, 0] = np.dot(B, y[:, 0])

        # Compute (I - S)^+ to correct derivatives at x=a.
        D = pinv(np.identity(n) - S)
    else:
        B = None
        D = None

    if bc_tol is None:
        bc_tol = tol

    # Maximum number of iterations
    max_iteration = 10

    fun_wrapped, bc_wrapped, fun_jac_wrapped, bc_jac_wrapped = wrap_functions(
        fun, bc, fun_jac, bc_jac, k, a, S, D, dtype)

    f = fun_wrapped(x, y, p)
    if f.shape != y.shape:
        raise ValueError(f"`fun` return is expected to have shape {y.shape}, "
                         f"but actually has {f.shape}.")

    bc_res = bc_wrapped(y[:, 0], y[:, -1], p)
    if bc_res.shape != (n + k,):
        raise ValueError(f"`bc` return is expected to have shape {(n + k,)}, "
                         f"but actually has {bc_res.shape}.")

    status = 0
    iteration = 0
    if verbose == 2:
        print_iteration_header()

    while True:
        m = x.shape[0]

        col_fun, jac_sys = prepare_sys(n, m, k, fun_wrapped, bc_wrapped,
                                       fun_jac_wrapped, bc_jac_wrapped, x, h)
        y, p, singular = solve_newton(n, m, h, col_fun, bc_wrapped, jac_sys,
                                      y, p, B, tol, bc_tol)
        iteration += 1

        col_res, y_middle, f, f_middle = collocation_fun(fun_wrapped, y,
                                                         p, x, h)
        bc_res = bc_wrapped(y[:, 0], y[:, -1], p)
        max_bc_res = np.max(abs(bc_res))

        # This relation is not trivial, but can be verified.
        r_middle = 1.5 * col_res / h
        sol = create_spline(y, f, x, h)
        rms_res = estimate_rms_residuals(fun_wrapped, sol, x, h, p,
                                         r_middle, f_middle)
        max_rms_res = np.max(rms_res)

        if singular:
            status = 2
            break

        insert_1, = np.nonzero((rms_res > tol) & (rms_res < 100 * tol))
        insert_2, = np.nonzero(rms_res >= 100 * tol)
        nodes_added = insert_1.shape[0] + 2 * insert_2.shape[0]

        if m + nodes_added > max_nodes:
            status = 1
            if verbose == 2:
                nodes_added = f"({nodes_added})"
                print_iteration_progress(iteration, max_rms_res, max_bc_res,
                                         m, nodes_added)
            break

        if verbose == 2:
            print_iteration_progress(iteration, max_rms_res, max_bc_res, m,
                                     nodes_added)

        if nodes_added > 0:
            x = modify_mesh(x, insert_1, insert_2)
            h = np.diff(x)
            y = sol(x)
        elif max_bc_res <= bc_tol:
            status = 0
            break
        elif iteration >= max_iteration:
            status = 3
            break

    if verbose > 0:
        if status == 0:
            print(f"Solved in {iteration} iterations, number of nodes {x.shape[0]}. \n"
                  f"Maximum relative residual: {max_rms_res:.2e} \n"
                  f"Maximum boundary residual: {max_bc_res:.2e}")
        elif status == 1:
            print(f"Number of nodes is exceeded after iteration {iteration}. \n"
                  f"Maximum relative residual: {max_rms_res:.2e} \n"
                  f"Maximum boundary residual: {max_bc_res:.2e}")
        elif status == 2:
            print("Singular Jacobian encountered when solving the collocation "
                  f"system on iteration {iteration}. \n"
                  f"Maximum relative residual: {max_rms_res:.2e} \n"
                  f"Maximum boundary residual: {max_bc_res:.2e}")
        elif status == 3:
            print("The solver was unable to satisfy boundary conditions "
                  f"tolerance on iteration {iteration}. \n"
                  f"Maximum relative residual: {max_rms_res:.2e} \n"
                  f"Maximum boundary residual: {max_bc_res:.2e}")

    if p.size == 0:
        p = None

    return BVPResult(sol=sol, p=p, x=x, y=y, yp=f, rms_residuals=rms_res,
                     niter=iteration, status=status,
                     message=TERMINATION_MESSAGES[status], success=status == 0)
<|endoftext|>import numpy as np
import numpy.typing as npt
import math
import warnings
from collections import namedtuple
from collections.abc import Callable

from scipy.special import roots_legendre
from scipy.special import gammaln, logsumexp
from scipy._lib._util import _rng_spawn
from scipy._lib._array_api import _asarray, array_namespace, xp_broadcast_promote


__all__ = ['fixed_quad', 'romb',
           'trapezoid', 'simpson',
           'cumulative_trapezoid', 'newton_cotes',
           'qmc_quad', 'cumulative_simpson']


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
        The axis along which to integrate. The default is the last axis.

    Returns
    -------
    trapezoid : float or ndarray
        Definite integral of `y` = n-dimensional array as approximated along
        a single axis by the trapezoidal rule. If `y` is a 1-dimensional array,
        then the result is a float. If `n` is greater than 1, then the result
        is an `n`-1 dimensional array.

    See Also
    --------
    cumulative_trapezoid, simpson, romb

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
    Use the trapezoidal rule on evenly spaced points:

    >>> import numpy as np
    >>> from scipy import integrate
    >>> integrate.trapezoid([1, 2, 3])
    4.0

    The spacing between sample points can be selected by either the
    ``x`` or ``dx`` arguments:

    >>> integrate.trapezoid([1, 2, 3], x=[4, 6, 8])
    8.0
    >>> integrate.trapezoid([1, 2, 3], dx=2)
    8.0

    Using a decreasing ``x`` corresponds to integrating in reverse:

    >>> integrate.trapezoid([1, 2, 3], x=[8, 6, 4])
    -8.0

    More generally ``x`` is used to integrate along a parametric curve. We can
    estimate the integral :math:`\int_0^1 x^2 = 1/3` using:

    >>> x = np.linspace(0, 1, num=50)
    >>> y = x**2
    >>> integrate.trapezoid(y, x)
    0.33340274885464394

    Or estimate the area of a circle, noting we repeat the sample which closes
    the curve:

    >>> theta = np.linspace(0, 2 * np.pi, num=1000, endpoint=True)
    >>> integrate.trapezoid(np.cos(theta), x=np.sin(theta))
    3.141571941375841

    ``trapezoid`` can be applied along a specified axis to do multiple
    computations in one call:

    >>> a = np.arange(6).reshape(2, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> integrate.trapezoid(a, axis=0)
    array([1.5, 2.5, 3.5])
    >>> integrate.trapezoid(a, axis=1)
    array([2.,  8.])
    """
    xp = array_namespace(y)
    y = _asarray(y, xp=xp, subok=True)
    # Cannot just use the broadcasted arrays that are returned
    # because trapezoid does not follow normal broadcasting rules
    # cf. https://github.com/scipy/scipy/pull/21524#issuecomment-2354105942
    result_dtype = xp_broadcast_promote(y, force_floating=True, xp=xp)[0].dtype
    nd = y.ndim
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    if x is None:
        d = dx
    else:
        x = _asarray(x, xp=xp, subok=True)
        if x.ndim == 1:
            d = x[1:] - x[:-1]
            # make d broadcastable to y
            slice3 = [None] * nd
            slice3[axis] = slice(None)
            d = d[tuple(slice3)]
        else:
            # if x is n-D it should be broadcastable to y
            x = xp.broadcast_to(x, y.shape)
            d = x[tuple(slice1)] - x[tuple(slice2)]
    try:
        ret = xp.sum(
            d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0,
            axis=axis, dtype=result_dtype
        )
    except ValueError:
        # Operations didn't work, cast to ndarray
        d = xp.asarray(d)
        y = xp.asarray(y)
        ret = xp.sum(
            d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0,
            axis=axis, dtype=result_dtype
        )
    return ret


def _cached_roots_legendre(n):
    """
    Cache roots_legendre results to speed up calls of the fixed_quad
    function.
    """
    if n in _cached_roots_legendre.cache:
        return _cached_roots_legendre.cache[n]

    _cached_roots_legendre.cache[n] = roots_legendre(n)
    return _cached_roots_legendre.cache[n]


_cached_roots_legendre.cache = dict()


def fixed_quad(func, a, b, args=(), n=5):
    """
    Compute a definite integral using fixed-order Gaussian quadrature.

    Integrate `func` from `a` to `b` using Gaussian quadrature of
    order `n`.

    Parameters
    ----------
    func : callable
        A Python function or method to integrate (must accept vector inputs).
        If integrating a vector-valued function, the returned array must have
        shape ``(..., len(x))``.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    args : tuple, optional
        Extra arguments to pass to function, if any.
    n : int, optional
        Order of quadrature integration. Default is 5.

    Returns
    -------
    val : float
        Gaussian quadrature approximation to the integral
    none : None
        Statically returned value of None

    See Also
    --------
    quad : adaptive quadrature using QUADPACK
    dblquad : double integrals
    tplquad : triple integrals
    romb : integrators for sampled data
    simpson : integrators for sampled data
    cumulative_trapezoid : cumulative integration for sampled data

    Examples
    --------
    >>> from scipy import integrate
    >>> import numpy as np
    >>> f = lambda x: x**8
    >>> integrate.fixed_quad(f, 0.0, 1.0, n=4)
    (0.1110884353741496, None)
    >>> integrate.fixed_quad(f, 0.0, 1.0, n=5)
    (0.11111111111111102, None)
    >>> print(1/9.0)  # analytical result
    0.1111111111111111

    >>> integrate.fixed_quad(np.cos, 0.0, np.pi/2, n=4)
    (0.9999999771971152, None)
    >>> integrate.fixed_quad(np.cos, 0.0, np.pi/2, n=5)
    (1.000000000039565, None)
    >>> np.sin(np.pi/2)-np.sin(0)  # analytical result
    1.0

    """
    x, w = _cached_roots_legendre(n)
    x = np.real(x)
    if np.isinf(a) or np.isinf(b):
        raise ValueError("Gaussian quadrature is only available for "
                         "finite limits.")
    y = (b-a)*(x+1)/2.0 + a
    return (b-a)/2.0 * np.sum(w*func(y, *args), axis=-1), None


def tupleset(t, i, value):
    l = list(t)
    l[i] = value
    return tuple(l)


def cumulative_trapezoid(y, x=None, dx=1.0, axis=-1, initial=None):
    """
    Cumulatively integrate y(x) using the composite trapezoidal rule.

    Parameters
    ----------
    y : array_like
        Values to integrate.
    x : array_like, optional
        The coordinate to integrate along. If None (default), use spacing `dx`
        between consecutive elements in `y`.
    dx : float, optional
        Spacing between elements of `y`. Only used if `x` is None.
    axis : int, optional
        Specifies the axis to cumulate. Default is -1 (last axis).
    initial : scalar, optional
        If given, insert this value at the beginning of the returned result.
        0 or None are the only values accepted. Default is None, which means
        `res` has one element less than `y` along the axis of integration.

    Returns
    -------
    res : ndarray
        The result of cumulative integration of `y` along `axis`.
        If `initial` is None, the shape is such that the axis of integration
        has one less value than `y`. If `initial` is given, the shape is equal
        to that of `y`.

    See Also
    --------
    numpy.cumsum, numpy.cumprod
    cumulative_simpson : cumulative integration using Simpson's 1/3 rule
    quad : adaptive quadrature using QUADPACK
    fixed_quad : fixed-order Gaussian quadrature
    dblquad : double integrals
    tplquad : triple integrals
    romb : integrators for sampled data

    Examples
    --------
    >>> from scipy import integrate
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> x = np.linspace(-2, 2, num=20)
    >>> y = x
    >>> y_int = integrate.cumulative_trapezoid(y, x, initial=0)
    >>> plt.plot(x, y_int, 'ro', x, y[0] + 0.5 * x**2, 'b-')
    >>> plt.show()

    """
    y = np.asarray(y)
    if y.shape[axis] == 0:
        raise ValueError("At least one point is required along `axis`.")
    if x is None:
        d = dx
    else:
        x = np.asarray(x)
        if x.ndim == 1:
            d = np.diff(x)
            # reshape to correct shape
            shape = [1] * y.ndim
            shape[axis] = -1
            d = d.reshape(shape)
        elif len(x.shape) != len(y.shape):
            raise ValueError("If given, shape of x must be 1-D or the "
                             "same as y.")
        else:
            d = np.diff(x, axis=axis)

        if d.shape[axis] != y.shape[axis] - 1:
            raise ValueError("If given, length of x along axis must be the "
                             "same as y.")

    nd = len(y.shape)
    slice1 = tupleset((slice(None),)*nd, axis, slice(1, None))
    slice2 = tupleset((slice(None),)*nd, axis, slice(None, -1))
    res = np.cumsum(d * (y[slice1] + y[slice2]) / 2.0, axis=axis)

    if initial is not None:
        if initial != 0:
            raise ValueError("`initial` must be `None` or `0`.")
        if not np.isscalar(initial):
            raise ValueError("`initial` parameter should be a scalar.")

        shape = list(res.shape)
        shape[axis] = 1
        res = np.concatenate([np.full(shape, initial, dtype=res.dtype), res],
                             axis=axis)

    return res


def _basic_simpson(y, start, stop, x, dx, axis):
    nd = len(y.shape)
    if start is None:
        start = 0
    step = 2
    slice_all = (slice(None),)*nd
    slice0 = tupleset(slice_all, axis, slice(start, stop, step))
    slice1 = tupleset(slice_all, axis, slice(start+1, stop+1, step))
    slice2 = tupleset(slice_all, axis, slice(start+2, stop+2, step))

    if x is None:  # Even-spaced Simpson's rule.
        result = np.sum(y[slice0] + 4.0*y[slice1] + y[slice2], axis=axis)
        result *= dx / 3.0
    else:
        # Account for possibly different spacings.
        #    Simpson's rule changes a bit.
        h = np.diff(x, axis=axis)
        sl0 = tupleset(slice_all, axis, slice(start, stop, step))
        sl1 = tupleset(slice_all, axis, slice(start+1, stop+1, step))
        h0 = h[sl0].astype(float, copy=False)
        h1 = h[sl1].astype(float, copy=False)
        hsum = h0 + h1
        hprod = h0 * h1
        h0divh1 = np.true_divide(h0, h1, out=np.zeros_like(h0), where=h1 != 0)
        tmp = hsum/6.0 * (y[slice0] *
                          (2.0 - np.true_divide(1.0, h0divh1,
                                                out=np.zeros_like(h0divh1),
                                                where=h0divh1 != 0)) +
                          y[slice1] * (hsum *
                                       np.true_divide(hsum, hprod,
                                                      out=np.zeros_like(hsum),
                                                      where=hprod != 0)) +
                          y[slice2] * (2.0 - h0divh1))
        result = np.sum(tmp, axis=axis)
    return result


def simpson(y, x=None, *, dx=1.0, axis=-1):
    """
    Integrate y(x) using samples along the given axis and the composite
    Simpson's rule. If x is None, spacing of dx is assumed.

    Parameters
    ----------
    y : array_like
        Array to be integrated.
    x : array_like, optional
        If given, the points at which `y` is sampled.
    dx : float, optional
        Spacing of integration points along axis of `x`. Only used when
        `x` is None. Default is 1.
    axis : int, optional
        Axis along which to integrate. Default is the last axis.

    Returns
    -------
    float
        The estimated integral computed with the composite Simpson's rule.

    See Also
    --------
    quad : adaptive quadrature using QUADPACK
    fixed_quad : fixed-order Gaussian quadrature
    dblquad : double integrals
    tplquad : triple integrals
    romb : integrators for sampled data
    cumulative_trapezoid : cumulative integration for sampled data
    cumulative_simpson : cumulative integration using Simpson's 1/3 rule

    Notes
    -----
    For an odd number of samples that are equally spaced the result is
    exact if the function is a polynomial of order 3 or less. If
    the samples are not equally spaced, then the result is exact only
    if the function is a polynomial of order 2 or less.

    References
    ----------
    .. [1] Cartwright, Kenneth V. Simpson's Rule Cumulative Integration with
           MS Excel and Irregularly-spaced Data. Journal of Mathematical
           Sciences and Mathematics Education. 12 (2): 1-9

    Examples
    --------
    >>> from scipy import integrate
    >>> import numpy as np
    >>> x = np.arange(0, 10)
    >>> y = np.arange(0, 10)

    >>> integrate.simpson(y, x=x)
    40.5

    >>> y = np.power(x, 3)
    >>> integrate.simpson(y, x=x)
    1640.5
    >>> integrate.quad(lambda x: x**3, 0, 9)[0]
    1640.25

    """
    y = np.asarray(y)
    nd = len(y.shape)
    N = y.shape[axis]
    last_dx = dx
    returnshape = 0
    if x is not None:
        x = np.asarray(x)
        if len(x.shape) == 1:
            shapex = [1] * nd
            shapex[axis] = x.shape[0]
            saveshape = x.shape
            returnshape = 1
            x = x.reshape(tuple(shapex))
        elif len(x.shape) != len(y.shape):
            raise ValueError("If given, shape of x must be 1-D or the "
                             "same as y.")
        if x.shape[axis] != N:
            raise ValueError("If given, length of x along axis must be the "
                             "same as y.")

    if N % 2 == 0:
        val = 0.0
        result = 0.0
        slice_all = (slice(None),) * nd

        if N == 2:
            # need at least 3 points in integration axis to form parabolic
            # segment. If there are two points then any of 'avg', 'first',
            # 'last' should give the same result.
            slice1 = tupleset(slice_all, axis, -1)
            slice2 = tupleset(slice_all, axis, -2)
            if x is not None:
                last_dx = x[slice1] - x[slice2]
            val += 0.5 * last_dx * (y[slice1] + y[slice2])
        else:
            # use Simpson's rule on first intervals
            result = _basic_simpson(y, 0, N-3, x, dx, axis)

            slice1 = tupleset(slice_all, axis, -1)
            slice2 = tupleset(slice_all, axis, -2)
            slice3 = tupleset(slice_all, axis, -3)

            h = np.asarray([dx, dx], dtype=np.float64)
            if x is not None:
                # grab the last two spacings from the appropriate axis
                hm2 = tupleset(slice_all, axis, slice(-2, -1, 1))
                hm1 = tupleset(slice_all, axis, slice(-1, None, 1))

                diffs = np.float64(np.diff(x, axis=axis))
                h = [np.squeeze(diffs[hm2], axis=axis),
                     np.squeeze(diffs[hm1], axis=axis)]

            # This is the correction for the last interval according to
            # Cartwright.
            # However, I used the equations given at
            # https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_rule_for_irregularly_spaced_data
            # A footnote on Wikipedia says:
            # Cartwright 2017, Equation 8. The equation in Cartwright is
            # calculating the first interval whereas the equations in the
            # Wikipedia article are adjusting for the last integral. If the
            # proper algebraic substitutions are made, the equation results in
            # the values shown.
            num = 2 * h[1] ** 2 + 3 * h[0] * h[1]
            den = 6 * (h[1] + h[0])
            alpha = np.true_divide(
                num,
                den,
                out=np.zeros_like(den),
                where=den != 0
            )

            num = h[1] ** 2 + 3.0 * h[0] * h[1]
            den = 6 * h[0]
            beta = np.true_divide(
                num,
                den,
                out=np.zeros_like(den),
                where=den != 0
            )

            num = 1 * h[1] ** 3
            den = 6 * h[0] * (h[0] + h[1])
            eta = np.true_divide(
                num,
                den,
                out=np.zeros_like(den),
                where=den != 0
            )

            result += alpha*y[slice1] + beta*y[slice2] - eta*y[slice3]

        result += val
    else:
        result = _basic_simpson(y, 0, N-2, x, dx, axis)
    if returnshape:
        x = x.reshape(saveshape)
    return result


def _cumulatively_sum_simpson_integrals(
    y: np.ndarray, 
    dx: np.ndarray, 
    integration_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Calculate cumulative sum of Simpson integrals.
    Takes as input the integration function to be used. 
    The integration_func is assumed to return the cumulative sum using
    composite Simpson's rule. Assumes the axis of summation is -1.
    """
    sub_integrals_h1 = integration_func(y, dx)
    sub_integrals_h2 = integration_func(y[..., ::-1], dx[..., ::-1])[..., ::-1]
    
    shape = list(sub_integrals_h1.shape)
    shape[-1] += 1
    sub_integrals = np.empty(shape)
    sub_integrals[..., :-1:2] = sub_integrals_h1[..., ::2]
    sub_integrals[..., 1::2] = sub_integrals_h2[..., ::2]
    # Integral over last subinterval can only be calculated from 
    # formula for h2
    sub_integrals[..., -1] = sub_integrals_h2[..., -1]
    res = np.cumsum(sub_integrals, axis=-1)
    return res


def _cumulative_simpson_equal_intervals(y: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Calculate the Simpson integrals for all h1 intervals assuming equal interval
    widths. The function can also be used to calculate the integral for all
    h2 intervals by reversing the inputs, `y` and `dx`.
    """
    d = dx[..., :-1]
    f1 = y[..., :-2]
    f2 = y[..., 1:-1]
    f3 = y[..., 2:]

    # Calculate integral over the subintervals (eqn (10) of Reference [2])
    return d / 3 * (5 * f1 / 4 + 2 * f2 - f3 / 4)


def _cumulative_simpson_unequal_intervals(y: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Calculate the Simpson integrals for all h1 intervals assuming unequal interval
    widths. The function can also be used to calculate the integral for all
    h2 intervals by reversing the inputs, `y` and `dx`.
    """
    x21 = dx[..., :-1]
    x32 = dx[..., 1:]
    f1 = y[..., :-2]
    f2 = y[..., 1:-1]
    f3 = y[..., 2:]

    x31 = x21 + x32
    x21_x31 = x21/x31
    x21_x32 = x21/x32
    x21x21_x31x32 = x21_x31 * x21_x32

    # Calculate integral over the subintervals (eqn (8) of Reference [2])
    coeff1 = 3 - x21_x31
    coeff2 = 3 + x21x21_x31x32 + x21_x31
    coeff3 = -x21x21_x31x32

    return x21/6 * (coeff1*f1 + coeff2*f2 + coeff3*f3)


def _ensure_float_array(arr: npt.ArrayLike) -> np.ndarray:
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(float, copy=False)
    return arr


def cumulative_simpson(y, *, x=None, dx=1.0, axis=-1, initial=None):
    r"""
    Cumulatively integrate y(x) using the composite Simpson's 1/3 rule.
    The integral of the samples at every point is calculated by assuming a 
    quadratic relationship between each point and the two adjacent points.

    Parameters
    ----------
    y : array_like
        Values to integrate. Requires at least one point along `axis`. If two or fewer
        points are provided along `axis`, Simpson's integration is not possible and the
        result is calculated with `cumulative_trapezoid`.
    x : array_like, optional
        The coordinate to integrate along. Must have the same shape as `y` or
        must be 1D with the same length as `y` along `axis`. `x` must also be
        strictly increasing along `axis`.
        If `x` is None (default), integration is performed using spacing `dx`
        between consecutive elements in `y`.
    dx : scalar or array_like, optional
        Spacing between elements of `y`. Only used if `x` is None. Can either 
        be a float, or an array with the same shape as `y`, but of length one along
        `axis`. Default is 1.0.
    axis : int, optional
        Specifies the axis to integrate along. Default is -1 (last axis).
    initial : scalar or array_like, optional
        If given, insert this value at the beginning of the returned result,
        and add it to the rest of the result. Default is None, which means no
        value at ``x[0]`` is returned and `res` has one element less than `y`
        along the axis of integration. Can either be a float, or an array with
        the same shape as `y`, but of length one along `axis`.

    Returns
    -------
    res : ndarray
        The result of cumulative integration of `y` along `axis`.
        If `initial` is None, the shape is such that the axis of integration
        has one less value than `y`. If `initial` is given, the shape is equal
        to that of `y`.

    See Also
    --------
    numpy.cumsum
    cumulative_trapezoid : cumulative integration using the composite 
        trapezoidal rule
    simpson : integrator for sampled data using the Composite Simpson's Rule

    Notes
    -----

    .. versionadded:: 1.12.0

    The composite Simpson's 1/3 method can be used to approximate the definite 
    integral of a sampled input function :math:`y(x)` [1]_. The method assumes 
    a quadratic relationship over the interval containing any three consecutive
    sampled points.

    Consider three consecutive points: 
    :math:`(x_1, y_1), (x_2, y_2), (x_3, y_3)`.

    Assuming a quadratic relationship over the three points, the integral over
    the subinterval between :math:`x_1` and :math:`x_2` is given by formula
    (8) of [2]_:
    
    .. math::
        \int_{x_1}^{x_2} y(x) dx\ &= \frac{x_2-x_1}{6}\left[\
        \left\{3-\frac{x_2-x_1}{x_3-x_1}\right\} y_1 + \
        \left\{3 + \frac{(x_2-x_1)^2}{(x_3-x_2)(x_3-x_1)} + \
        \frac{x_2-x_1}{x_3-x_1}\right\} y_2\\
        - \frac{(x_2-x_1)^2}{(x_3-x_2)(x_3-x_1)} y_3\right]

    The integral between :math:`x_2` and :math:`x_3` is given by swapping
    appearances of :math:`x_1` and :math:`x_3`. The integral is estimated
    separately for each subinterval and then cumulatively summed to obtain
    the final result.
    
    For samples that are equally spaced, the result is exact if the function
    is a polynomial of order three or less [1]_ and the number of subintervals
    is even. Otherwise, the integral is exact for polynomials of order two or
    less. 

    References
    ----------
    .. [1] Wikipedia page: https://en.wikipedia.org/wiki/Simpson's_rule
    .. [2] Cartwright, Kenneth V. Simpson's Rule Cumulative Integration with
            MS Excel and Irregularly-spaced Data. Journal of Mathematical
            Sciences and Mathematics Education. 12 (2): 1-9

    Examples
    --------
    >>> from scipy import integrate
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-2, 2, num=20)
    >>> y = x**2
    >>> y_int = integrate.cumulative_simpson(y, x=x, initial=0)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, y_int, 'ro', x, x**3/3 - (x[0])**3/3, 'b-')
    >>> ax.grid()
    >>> plt.show()

    The output of `cumulative_simpson` is similar to that of iteratively
    calling `simpson` with successively higher upper limits of integration, but
    not identical.

    >>> def cumulative_simpson_reference(y, x):
    ...     return np.asarray([integrate.simpson(y[:i], x=x[:i])
    ...                        for i in range(2, len(y) + 1)])
    >>>
    >>> rng = np.random.default_rng(354673834679465)
    >>> x, y = rng.random(size=(2, 10))
    >>> x.sort()
    >>>
    >>> res = integrate.cumulative_simpson(y, x=x)
    >>> ref = cumulative_simpson_reference(y, x)
    >>> equal = np.abs(res - ref) < 1e-15
    >>> equal  # not equal when `simpson` has even number of subintervals
    array([False,  True, False,  True, False,  True, False,  True,  True])

    This is expected: because `cumulative_simpson` has access to more
    information than `simpson`, it can typically produce more accurate
    estimates of the underlying integral over subintervals.

    """
    y = _ensure_float_array(y)

    # validate `axis` and standardize to work along the last axis
    original_y = y
    original_shape = y.shape
    try:
        y = np.swapaxes(y, axis, -1)
    except IndexError as e:
        message = f"`axis={axis}` is not valid for `y` with `y.ndim={y.ndim}`."
        raise ValueError(message) from e
    if y.shape[-1] < 3:
        res = cumulative_trapezoid(original_y, x, dx=dx, axis=axis, initial=None)
        res = np.swapaxes(res, axis, -1)

    elif x is not None:
        x = _ensure_float_array(x)
        message = ("If given, shape of `x` must be the same as `y` or 1-D with "
                   "the same length as `y` along `axis`.")
        if not (x.shape == original_shape
                or (x.ndim == 1 and len(x) == original_shape[axis])):
            raise ValueError(message)

        x = np.broadcast_to(x, y.shape) if x.ndim == 1 else np.swapaxes(x, axis, -1)
        dx = np.diff(x, axis=-1)
        if np.any(dx <= 0):
            raise ValueError("Input x must be strictly increasing.")
        res = _cumulatively_sum_simpson_integrals(
            y, dx, _cumulative_simpson_unequal_intervals
        )

    else:
        dx = _ensure_float_array(dx)
        final_dx_shape = tupleset(original_shape, axis, original_shape[axis] - 1)
        alt_input_dx_shape = tupleset(original_shape, axis, 1)
        message = ("If provided, `dx` must either be a scalar or have the same "
                   "shape as `y` but with only 1 point along `axis`.")
        if not (dx.ndim == 0 or dx.shape == alt_input_dx_shape):
            raise ValueError(message)
        dx = np.broadcast_to(dx, final_dx_shape)
        dx = np.swapaxes(dx, axis, -1)
        res = _cumulatively_sum_simpson_integrals(
            y, dx, _cumulative_simpson_equal_intervals
        )

    if initial is not None:
        initial = _ensure_float_array(initial)
        alt_initial_input_shape = tupleset(original_shape, axis, 1)
        message = ("If provided, `initial` must either be a scalar or have the "
                   "same shape as `y` but with only 1 point along `axis`.")
        if not (initial.ndim == 0 or initial.shape == alt_initial_input_shape):
            raise ValueError(message)
        initial = np.broadcast_to(initial, alt_initial_input_shape)
        initial = np.swapaxes(initial, axis, -1)

        res += initial
        res = np.concatenate((initial, res), axis=-1)

    res = np.swapaxes(res, -1, axis)
    return res


def romb(y, dx=1.0, axis=-1, show=False):
    """
    Romberg integration using samples of a function.

    Parameters
    ----------
    y : array_like
        A vector of ``2**k + 1`` equally-spaced samples of a function.
    dx : float, optional
        The sample spacing. Default is 1.
    axis : int, optional
        The axis along which to integrate. Default is -1 (last axis).
    show : bool, optional
        When `y` is a single 1-D array, then if this argument is True
        print the table showing Richardson extrapolation from the
        samples. Default is False.

    Returns
    -------
    romb : ndarray
        The integrated result for `axis`.

    See Also
    --------
    quad : adaptive quadrature using QUADPACK
    fixed_quad : fixed-order Gaussian quadrature
    dblquad : double integrals
    tplquad : triple integrals
    simpson : integrators for sampled data
    cumulative_trapezoid : cumulative integration for sampled data

    Examples
    --------
    >>> from scipy import integrate
    >>> import numpy as np
    >>> x = np.arange(10, 14.25, 0.25)
    >>> y = np.arange(3, 12)

    >>> integrate.romb(y)
    56.0

    >>> y = np.sin(np.power(x, 2.5))
    >>> integrate.romb(y)
    -0.742561336672229

    >>> integrate.romb(y, show=True)
    Richardson Extrapolation Table for Romberg Integration
    ======================================================
    -0.81576
     4.63862  6.45674
    -1.10581 -3.02062 -3.65245
    -2.57379 -3.06311 -3.06595 -3.05664
    -1.34093 -0.92997 -0.78776 -0.75160 -0.74256
    ======================================================
    -0.742561336672229  # may vary

    """
    y = np.asarray(y)
    nd = len(y.shape)
    Nsamps = y.shape[axis]
    Ninterv = Nsamps-1
    n = 1
    k = 0
    while n < Ninterv:
        n <<= 1
        k += 1
    if n != Ninterv:
        raise ValueError("Number of samples must be one plus a "
                         "non-negative power of 2.")

    R = {}
    slice_all = (slice(None),) * nd
    slice0 = tupleset(slice_all, axis, 0)
    slicem1 = tupleset(slice_all, axis, -1)
    h = Ninterv * np.asarray(dx, dtype=float)
    R[(0, 0)] = (y[slice0] + y[slicem1])/2.0*h
    slice_R = slice_all
    start = stop = step = Ninterv
    for i in range(1, k+1):
        start >>= 1
        slice_R = tupleset(slice_R, axis, slice(start, stop, step))
        step >>= 1
        R[(i, 0)] = 0.5*(R[(i-1, 0)] + h*y[slice_R].sum(axis=axis))
        for j in range(1, i+1):
            prev = R[(i, j-1)]
            R[(i, j)] = prev + (prev-R[(i-1, j-1)]) / ((1 << (2*j))-1)
        h /= 2.0

    if show:
        if not np.isscalar(R[(0, 0)]):
            print("*** Printing table only supported for integrals" +
                  " of a single data set.")
        else:
            try:
                precis = show[0]
            except (TypeError, IndexError):
                precis = 5
            try:
                width = show[1]
            except (TypeError, IndexError):
                width = 8
            formstr = "%%%d.%df" % (width, precis)

            title = "Richardson Extrapolation Table for Romberg Integration"
            print(title, "=" * len(title), sep="\n", end="\n")
            for i in range(k+1):
                for j in range(i+1):
                    print(formstr % R[(i, j)], end=" ")
                print()
            print("=" * len(title))

    return R[(k, k)]


# Coefficients for Newton-Cotes quadrature
#
# These are the points being used
#  to construct the local interpolating polynomial
#  a are the weights for Newton-Cotes integration
#  B is the error coefficient.
#  error in these coefficients grows as N gets larger.
#  or as samples are closer and closer together

# You can use maxima to find these rational coefficients
#  for equally spaced data using the commands
#  a(i,N) := (integrate(product(r-j,j,0,i-1) * product(r-j,j,i+1,N),r,0,N)
#             / ((N-i)! * i!) * (-1)^(N-i));
#  Be(N) := N^(N+2)/(N+2)! * (N/(N+3) - sum((i/N)^(N+2)*a(i,N),i,0,N));
#  Bo(N) := N^(N+1)/(N+1)! * (N/(N+2) - sum((i/N)^(N+1)*a(i,N),i,0,N));
#  B(N) := (if (mod(N,2)=0) then Be(N) else Bo(N));
#
# pre-computed for equally-spaced weights
#
# num_a, den_a, int_a, num_B, den_B = _builtincoeffs[N]
#
#  a = num_a*array(int_a)/den_a
#  B = num_B*1.0 / den_B
#
#  integrate(f(x),x,x_0,x_N) = dx*sum(a*f(x_i)) + B*(dx)^(2k+3) f^(2k+2)(x*)
#    where k = N // 2
#
_builtincoeffs = {
    1: (1,2,[1,1],-1,12),
    2: (1,3,[1,4,1],-1,90),
    3: (3,8,[1,3,3,1],-3,80),
    4: (2,45,[7,32,12,32,7],-8,945),
    5: (5,288,[19,75,50,50,75,19],-275,12096),
    6: (1,140,[41,216,27,272,27,216,41],-9,1400),
    7: (7,17280,[751,3577,1323,2989,2989,1323,3577,751],-8183,518400),
    8: (4,14175,[989,5888,-928,10496,-4540,10496,-928,5888,989],
        -2368,467775),
    9: (9,89600,[2857,15741,1080,19344,5778,5778,19344,1080,
                 15741,2857], -4671, 394240),
    10: (5,299376,[16067,106300,-48525,272400,-260550,427368,
                   -260550,272400,-48525,106300,16067],
         -673175, 163459296),
    11: (11,87091200,[2171465,13486539,-3237113, 25226685,-9595542,
                      15493566,15493566,-9595542,25226685,-3237113,
                      13486539,2171465], -2224234463, 237758976000),
    12: (1, 5255250, [1364651,9903168,-7587864,35725120,-51491295,
                      87516288,-87797136,87516288,-51491295,35725120,
                      -7587864,9903168,1364651], -3012, 875875),
    13: (13, 402361344000,[8181904909, 56280729661, -31268252574,
                           156074417954,-151659573325,206683437987,
                           -43111992612,-43111992612,206683437987,
                           -151659573325,156074417954,-31268252574,
                           56280729661,8181904909], -2639651053,
         344881152000),
    14: (7, 2501928000, [90241897,710986864,-770720657,3501442784,
                         -6625093363,12630121616,-16802270373,19534438464,
                         -16802270373,12630121616,-6625093363,3501442784,
                         -770720657,710986864,90241897], -3740727473,
         1275983280000)
    }


def newton_cotes(rn, equal=0):
    r"""
    Return weights and error coefficient for Newton-Cotes integration.

    Suppose we have (N+1) samples of f at the positions
    x_0, x_1, ..., x_N. Then an N-point Newton-Cotes formula for the
    integral between x_0 and x_N is:

    :math:`\int_{x_0}^{x_N} f(x)dx = \Delta x \sum_{i=0}^{N} a_i f(x_i)
    + B_N (\Delta x)^{N+2} f^{N+1} (\xi)`

    where :math:`\xi \in [x_0,x_N]`
    and :math:`\Delta x = \frac{x_N-x_0}{N}` is the average samples spacing.

    If the samples are equally-spaced and N is even, then the error
    term is :math:`B_N (\Delta x)^{N+3} f^{N+2}(\xi)`.

    Parameters
    ----------
    rn : int
        The integer order for equally-spaced data or the relative positions of
        the samples with the first sample at 0 and the last at N, where N+1 is
        the length of `rn`. N is the order of the Newton-Cotes integration.
    equal : int, optional
        Set to 1 to enforce equally spaced data.

    Returns
    -------
    an : ndarray
        1-D array of weights to apply to the function at the provided sample
        positions.
    B : float
        Error coefficient.

    Notes
    -----
    Normally, the Newton-Cotes rules are used on smaller integration
    regions and a composite rule is used to return the total integral.

    Examples
    --------
    Compute the integral of sin(x) in [0, :math:`\pi`]:

    >>> from scipy.integrate import newton_cotes
    >>> import numpy as np
    >>> def f(x):
    ...     return np.sin(x)
    >>> a = 0
    >>> b = np.pi
    >>> exact = 2
    >>> for N in [2, 4, 6, 8, 10]:
    ...     x = np.linspace(a, b, N + 1)
    ...     an, B = newton_cotes(N, 1)
    ...     dx = (b - a) / N
    ...     quad = dx * np.sum(an * f(x))
    ...     error = abs(quad - exact)
    ...     print('{:2d}  {:10.9f}  {:.5e}'.format(N, quad, error))
    ...
     2   2.094395102   9.43951e-02
     4   1.998570732   1.42927e-03
     6   2.000017814   1.78136e-05
     8   1.999999835   1.64725e-07
    10   2.000000001   1.14677e-09

    """
    try:
        N = len(rn)-1
        if equal:
            rn = np.arange(N+1)
        elif np.all(np.diff(rn) == 1):
            equal = 1
    except Exception:
        N = rn
        rn = np.arange(N+1)
        equal = 1

    if equal and N in _builtincoeffs:
        na, da, vi, nb, db = _builtincoeffs[N]
        an = na * np.array(vi, dtype=float) / da
        return an, float(nb)/db

    if (rn[0] != 0) or (rn[-1] != N):
        raise ValueError("The sample positions must start at 0"
                         " and end at N")
    yi = rn / float(N)
    ti = 2 * yi - 1
    nvec = np.arange(N+1)
    C = ti ** nvec[:, np.newaxis]
    Cinv = np.linalg.inv(C)
    # improve precision of result
    for i in range(2):
        Cinv = 2*Cinv - Cinv.dot(C).dot(Cinv)
    vec = 2.0 / (nvec[::2]+1)
    ai = Cinv[:, ::2].dot(vec) * (N / 2.)

    if (N % 2 == 0) and equal:
        BN = N/(N+3.)
        power = N+2
    else:
        BN = N/(N+2.)
        power = N+1

    BN = BN - np.dot(yi**power, ai)
    p1 = power+1
    fac = power*math.log(N) - gammaln(p1)
    fac = math.exp(fac)
    return ai, BN*fac


def _qmc_quad_iv(func, a, b, n_points, n_estimates, qrng, log):

    # lazy import to avoid issues with partially-initialized submodule
    if not hasattr(qmc_quad, 'qmc'):
        from scipy import stats
        qmc_quad.stats = stats
    else:
        stats = qmc_quad.stats

    if not callable(func):
        message = "`func` must be callable."
        raise TypeError(message)

    # a, b will be modified, so copy. Oh well if it's copied twice.
    a = np.atleast_1d(a).copy()
    b = np.atleast_1d(b).copy()
    a, b = np.broadcast_arrays(a, b)
    dim = a.shape[0]

    try:
        func((a + b) / 2)
    except Exception as e:
        message = ("`func` must evaluate the integrand at points within "
                   "the integration range; e.g. `func( (a + b) / 2)` "
                   "must return the integrand at the centroid of the "
                   "integration volume.")
        raise ValueError(message) from e

    try:
        func(np.array([a, b]).T)
        vfunc = func
    except Exception as e:
        message = ("Exception encountered when attempting vectorized call to "
                   f"`func`: {e}. For better performance, `func` should "
                   "accept two-dimensional array `x` with shape `(len(a), "
                   "n_points)` and return an array of the integrand value at "
                   "each of the `n_points.")
        warnings.warn(message, stacklevel=3)

        def vfunc(x):
            return np.apply_along_axis(func, axis=-1, arr=x)

    n_points_int = np.int64(n_points)
    if n_points != n_points_int:
        message = "`n_points` must be an integer."
        raise TypeError(message)

    n_estimates_int = np.int64(n_estimates)
    if n_estimates != n_estimates_int:
        message = "`n_estimates` must be an integer."
        raise TypeError(message)

    if qrng is None:
        qrng = stats.qmc.Halton(dim)
    elif not isinstance(qrng, stats.qmc.QMCEngine):
        message = "`qrng` must be an instance of scipy.stats.qmc.QMCEngine."
        raise TypeError(message)

    if qrng.d != a.shape[0]:
        message = ("`qrng` must be initialized with dimensionality equal to "
                   "the number of variables in `a`, i.e., "
                   "`qrng.random().shape[-1]` must equal `a.shape[0]`.")
        raise ValueError(message)

    rng_seed = getattr(qrng, 'rng_seed', None)
    rng = stats._qmc.check_random_state(rng_seed)

    if log not in {True, False}:
        message = "`log` must be boolean (`True` or `False`)."
        raise TypeError(message)

    return (vfunc, a, b, n_points_int, n_estimates_int, qrng, rng, log, stats)


QMCQuadResult = namedtuple('QMCQuadResult', ['integral', 'standard_error'])


def qmc_quad(func, a, b, *, n_estimates=8, n_points=1024, qrng=None,
             log=False):
