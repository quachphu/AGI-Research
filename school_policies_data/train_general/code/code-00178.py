    .. versionadded:: 0.18.0

    References
    ----------
    .. [1] https://dlmf.nist.gov/10.47.E4
    .. [2] https://dlmf.nist.gov/10.51.E1
    .. [3] https://dlmf.nist.gov/10.51.E2
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    The spherical Bessel functions of the second kind :math:`y_n` accept
    both real and complex second argument. They can return a complex type:

    >>> from scipy.special import spherical_yn
    >>> spherical_yn(0, 3+5j)
    (8.022343088587197-9.880052589376795j)
    >>> type(spherical_yn(0, 3+5j))
    <class 'numpy.complex128'>

    We can verify the relation for the derivative from the Notes
    for :math:`n=3` in the interval :math:`[1, 2]`:

    >>> import numpy as np
    >>> x = np.arange(1.0, 2.0, 0.01)
    >>> np.allclose(spherical_yn(3, x, True),
    ...             spherical_yn(2, x) - 4/x * spherical_yn(3, x))
    True

    The first few :math:`y_n` with real argument:

    >>> import matplotlib.pyplot as plt
    >>> x = np.arange(0.0, 10.0, 0.01)
    >>> fig, ax = plt.subplots()
    >>> ax.set_ylim(-2.0, 1.0)
    >>> ax.set_title(r'Spherical Bessel functions $y_n$')
    >>> for n in np.arange(0, 4):
    ...     ax.plot(x, spherical_yn(n, x), label=rf'$y_{n}$')
    >>> plt.legend(loc='best')
    >>> plt.show()

    """
    n = np.asarray(n, dtype=np.dtype("long"))
    if derivative:
        return _spherical_yn_d(n, z)
    else:
        return _spherical_yn(n, z)


@use_reflection(+1)  # See DLMF 10.47(v) https://dlmf.nist.gov/10.47
def spherical_in(n, z, derivative=False):
    r"""Modified spherical Bessel function of the first kind or its derivative.

    Defined as [1]_,

    .. math:: i_n(z) = \sqrt{\frac{\pi}{2z}} I_{n + 1/2}(z),

    where :math:`I_n` is the modified Bessel function of the first kind.

    Parameters
    ----------
    n : int, array_like
        Order of the Bessel function (n >= 0).
    z : complex or float, array_like
        Argument of the Bessel function.
    derivative : bool, optional
        If True, the value of the derivative (rather than the function
        itself) is returned.

    Returns
    -------
    in : ndarray

    Notes
    -----
    The function is computed using its definitional relation to the
    modified cylindrical Bessel function of the first kind.

    The derivative is computed using the relations [2]_,

    .. math::
        i_n' = i_{n-1} - \frac{n + 1}{z} i_n.

        i_1' = i_0


    .. versionadded:: 0.18.0

    References
    ----------
    .. [1] https://dlmf.nist.gov/10.47.E7
    .. [2] https://dlmf.nist.gov/10.51.E5
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    The modified spherical Bessel functions of the first kind :math:`i_n`
    accept both real and complex second argument.
    They can return a complex type:

    >>> from scipy.special import spherical_in
    >>> spherical_in(0, 3+5j)
    (-1.1689867793369182-1.2697305267234222j)
    >>> type(spherical_in(0, 3+5j))
    <class 'numpy.complex128'>

    We can verify the relation for the derivative from the Notes
    for :math:`n=3` in the interval :math:`[1, 2]`:

    >>> import numpy as np
    >>> x = np.arange(1.0, 2.0, 0.01)
    >>> np.allclose(spherical_in(3, x, True),
    ...             spherical_in(2, x) - 4/x * spherical_in(3, x))
    True

    The first few :math:`i_n` with real argument:

    >>> import matplotlib.pyplot as plt
    >>> x = np.arange(0.0, 6.0, 0.01)
    >>> fig, ax = plt.subplots()
    >>> ax.set_ylim(-0.5, 5.0)
    >>> ax.set_title(r'Modified spherical Bessel functions $i_n$')
    >>> for n in np.arange(0, 4):
    ...     ax.plot(x, spherical_in(n, x), label=rf'$i_{n}$')
    >>> plt.legend(loc='best')
    >>> plt.show()

    """
    n = np.asarray(n, dtype=np.dtype("long"))
    if derivative:
        return _spherical_in_d(n, z)
    else:
        return _spherical_in(n, z)


def spherical_kn_reflection(n, z, derivative=False):
    # More complex than the other cases, and this will likely be re-implemented
    # in C++ anyway. Would require multiple function evaluations. Probably about
    # as fast to just resort to complex math, and much simpler.
    return spherical_kn(n, z + 0j, derivative=derivative).real


@use_reflection(reflection_fun=spherical_kn_reflection)
def spherical_kn(n, z, derivative=False):
    r"""Modified spherical Bessel function of the second kind or its derivative.

    Defined as [1]_,

    .. math:: k_n(z) = \sqrt{\frac{\pi}{2z}} K_{n + 1/2}(z),

    where :math:`K_n` is the modified Bessel function of the second kind.

    Parameters
    ----------
    n : int, array_like
        Order of the Bessel function (n >= 0).
    z : complex or float, array_like
        Argument of the Bessel function.
    derivative : bool, optional
        If True, the value of the derivative (rather than the function
        itself) is returned.

    Returns
    -------
    kn : ndarray

    Notes
    -----
    The function is computed using its definitional relation to the
    modified cylindrical Bessel function of the second kind.

    The derivative is computed using the relations [2]_,

    .. math::
        k_n' = -k_{n-1} - \frac{n + 1}{z} k_n.

        k_0' = -k_1


    .. versionadded:: 0.18.0

    References
    ----------
    .. [1] https://dlmf.nist.gov/10.47.E9
    .. [2] https://dlmf.nist.gov/10.51.E5
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    The modified spherical Bessel functions of the second kind :math:`k_n`
    accept both real and complex second argument.
    They can return a complex type:

    >>> from scipy.special import spherical_kn
    >>> spherical_kn(0, 3+5j)
    (0.012985785614001561+0.003354691603137546j)
    >>> type(spherical_kn(0, 3+5j))
    <class 'numpy.complex128'>

    We can verify the relation for the derivative from the Notes
    for :math:`n=3` in the interval :math:`[1, 2]`:

    >>> import numpy as np
    >>> x = np.arange(1.0, 2.0, 0.01)
    >>> np.allclose(spherical_kn(3, x, True),
    ...             - 4/x * spherical_kn(3, x) - spherical_kn(2, x))
    True

    The first few :math:`k_n` with real argument:

    >>> import matplotlib.pyplot as plt
    >>> x = np.arange(0.0, 4.0, 0.01)
    >>> fig, ax = plt.subplots()
    >>> ax.set_ylim(0.0, 5.0)
    >>> ax.set_title(r'Modified spherical Bessel functions $k_n$')
    >>> for n in np.arange(0, 4):
    ...     ax.plot(x, spherical_kn(n, x), label=rf'$k_{n}$')
    >>> plt.legend(loc='best')
    >>> plt.show()

    """
    n = np.asarray(n, dtype=np.dtype("long"))
    if derivative:
        return _spherical_kn_d(n, z)
    else:
        return _spherical_kn(n, z)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.special` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'ai_zeros',
    'assoc_laguerre',
    'bei_zeros',
    'beip_zeros',
    'ber_zeros',
    'bernoulli',
    'berp_zeros',
    'bi_zeros',
    'clpmn',
    'comb',
    'digamma',
    'diric',
    'erf_zeros',
    'euler',
    'factorial',
    'factorial2',
    'factorialk',
    'fresnel_zeros',
    'fresnelc_zeros',
    'fresnels_zeros',
    'gamma',
    'h1vp',
    'h2vp',
    'hankel1',
    'hankel2',
    'iv',
    'ivp',
    'jn_zeros',
    'jnjnp_zeros',
    'jnp_zeros',
    'jnyn_zeros',
    'jv',
    'jvp',
    'kei_zeros',
    'keip_zeros',
    'kelvin_zeros',
    'ker_zeros',
    'kerp_zeros',
    'kv',
    'kvp',
    'lmbda',
    'lpmn',
    'lpn',
    'lqmn',
    'lqn',
    'mathieu_a',
    'mathieu_b',
    'mathieu_even_coef',
    'mathieu_odd_coef',
    'obl_cv_seq',
    'pbdn_seq',
    'pbdv_seq',
    'pbvv_seq',
    'perm',
    'polygamma',
    'pro_cv_seq',
    'psi',
    'riccati_jn',
    'riccati_yn',
    'sinc',
    'y0_zeros',
    'y1_zeros',
    'y1p_zeros',
    'yn_zeros',
    'ynp_zeros',
    'yv',
    'yvp',
    'zeta'
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="special", module="basic",
                                   private_modules=["_basic", "_ufuncs"], all=__all__,
                                   attribute=name)
<|endoftext|>"""
========================================
Special functions (:mod:`scipy.special`)
========================================

.. currentmodule:: scipy.special

Almost all of the functions below accept NumPy arrays as input
arguments as well as single numbers. This means they follow
broadcasting and automatic array-looping rules. Technically,
they are `NumPy universal functions
<https://numpy.org/doc/stable/user/basics.ufuncs.html#ufuncs-basics>`_.
Functions which do not accept NumPy arrays are marked by a warning
in the section description.

.. seealso::

   `scipy.special.cython_special` -- Typed Cython versions of special functions


Error handling
==============

Errors are handled by returning NaNs or other appropriate values.
Some of the special function routines can emit warnings or raise
exceptions when an error occurs. By default this is disabled, except
for memory allocation errors, which result in an exception being raised.
To query and control the current error handling state the following
functions are provided.

.. autosummary::
   :toctree: generated/

   geterr                 -- Get the current way of handling special-function errors.
   seterr                 -- Set how special-function errors are handled.
   errstate               -- Context manager for special-function error handling.
   SpecialFunctionWarning -- Warning that can be emitted by special functions.
   SpecialFunctionError   -- Exception that can be raised by special functions.

Available functions
===================

Airy functions
--------------

.. autosummary::
   :toctree: generated/

   airy     -- Airy functions and their derivatives.
   airye    -- Exponentially scaled Airy functions and their derivatives.
   ai_zeros -- Compute `nt` zeros and values of the Airy function Ai and its derivative.
   bi_zeros -- Compute `nt` zeros and values of the Airy function Bi and its derivative.
   itairy   -- Integrals of Airy functions


Elliptic functions and integrals
--------------------------------

.. autosummary::
   :toctree: generated/

   ellipj    -- Jacobian elliptic functions.
   ellipk    -- Complete elliptic integral of the first kind.
   ellipkm1  -- Complete elliptic integral of the first kind around `m` = 1.
   ellipkinc -- Incomplete elliptic integral of the first kind.
   ellipe    -- Complete elliptic integral of the second kind.
   ellipeinc -- Incomplete elliptic integral of the second kind.
   elliprc   -- Degenerate symmetric integral RC.
   elliprd   -- Symmetric elliptic integral of the second kind.
   elliprf   -- Completely-symmetric elliptic integral of the first kind.
   elliprg   -- Completely-symmetric elliptic integral of the second kind.
   elliprj   -- Symmetric elliptic integral of the third kind.

Bessel functions
----------------

.. autosummary::
   :toctree: generated/

   jv                -- Bessel function of the first kind of real order and \
                        complex argument.
   jve               -- Exponentially scaled Bessel function of order `v`.
   yn                -- Bessel function of the second kind of integer order and \
                        real argument.
   yv                -- Bessel function of the second kind of real order and \
                        complex argument.
   yve               -- Exponentially scaled Bessel function of the second kind \
                        of real order.
   kn                -- Modified Bessel function of the second kind of integer \
                        order `n`
   kv                -- Modified Bessel function of the second kind of real order \
                        `v`
   kve               -- Exponentially scaled modified Bessel function of the \
                        second kind.
   iv                -- Modified Bessel function of the first kind of real order.
   ive               -- Exponentially scaled modified Bessel function of the \
                        first kind.
   hankel1           -- Hankel function of the first kind.
   hankel1e          -- Exponentially scaled Hankel function of the first kind.
   hankel2           -- Hankel function of the second kind.
   hankel2e          -- Exponentially scaled Hankel function of the second kind.
   wright_bessel     -- Wright's generalized Bessel function.
   log_wright_bessel -- Logarithm of Wright's generalized Bessel function.

The following function does not accept NumPy arrays (it is not a
universal function):

.. autosummary::
   :toctree: generated/

   lmbda -- Jahnke-Emden Lambda function, Lambdav(x).

Zeros of Bessel functions
^^^^^^^^^^^^^^^^^^^^^^^^^

The following functions do not accept NumPy arrays (they are not
universal functions):

.. autosummary::
   :toctree: generated/

   jnjnp_zeros -- Compute zeros of integer-order Bessel functions Jn and Jn'.
   jnyn_zeros  -- Compute nt zeros of Bessel functions Jn(x), Jn'(x), Yn(x), and Yn'(x).
   jn_zeros    -- Compute zeros of integer-order Bessel function Jn(x).
   jnp_zeros   -- Compute zeros of integer-order Bessel function derivative Jn'(x).
   yn_zeros    -- Compute zeros of integer-order Bessel function Yn(x).
   ynp_zeros   -- Compute zeros of integer-order Bessel function derivative Yn'(x).
   y0_zeros    -- Compute nt zeros of Bessel function Y0(z), and derivative at each zero.
   y1_zeros    -- Compute nt zeros of Bessel function Y1(z), and derivative at each zero.
   y1p_zeros   -- Compute nt zeros of Bessel derivative Y1'(z), and value at each zero.

Faster versions of common Bessel functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   j0  -- Bessel function of the first kind of order 0.
   j1  -- Bessel function of the first kind of order 1.
   y0  -- Bessel function of the second kind of order 0.
   y1  -- Bessel function of the second kind of order 1.
   i0  -- Modified Bessel function of order 0.
   i0e -- Exponentially scaled modified Bessel function of order 0.
   i1  -- Modified Bessel function of order 1.
   i1e -- Exponentially scaled modified Bessel function of order 1.
   k0  -- Modified Bessel function of the second kind of order 0, :math:`K_0`.
   k0e -- Exponentially scaled modified Bessel function K of order 0
   k1  -- Modified Bessel function of the second kind of order 1, :math:`K_1(x)`.
   k1e -- Exponentially scaled modified Bessel function K of order 1.

Integrals of Bessel functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   itj0y0     -- Integrals of Bessel functions of order 0.
   it2j0y0    -- Integrals related to Bessel functions of order 0.
   iti0k0     -- Integrals of modified Bessel functions of order 0.
   it2i0k0    -- Integrals related to modified Bessel functions of order 0.
   besselpoly -- Weighted integral of a Bessel function.

Derivatives of Bessel functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   jvp  -- Compute nth derivative of Bessel function Jv(z) with respect to `z`.
   yvp  -- Compute nth derivative of Bessel function Yv(z) with respect to `z`.
   kvp  -- Compute nth derivative of real-order modified Bessel function Kv(z)
   ivp  -- Compute nth derivative of modified Bessel function Iv(z) with respect to `z`.
   h1vp -- Compute nth derivative of Hankel function H1v(z) with respect to `z`.
   h2vp -- Compute nth derivative of Hankel function H2v(z) with respect to `z`.

Spherical Bessel functions
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   spherical_jn -- Spherical Bessel function of the first kind or its derivative.
   spherical_yn -- Spherical Bessel function of the second kind or its derivative.
   spherical_in -- Modified spherical Bessel function of the first kind or its derivative.
   spherical_kn -- Modified spherical Bessel function of the second kind or its derivative.

Riccati-Bessel functions
^^^^^^^^^^^^^^^^^^^^^^^^

The following functions do not accept NumPy arrays (they are not
universal functions):

.. autosummary::
   :toctree: generated/

   riccati_jn -- Compute Ricatti-Bessel function of the first kind and its derivative.
   riccati_yn -- Compute Ricatti-Bessel function of the second kind and its derivative.

Struve functions
----------------

.. autosummary::
   :toctree: generated/

   struve       -- Struve function.
   modstruve    -- Modified Struve function.
   itstruve0    -- Integral of the Struve function of order 0.
   it2struve0   -- Integral related to the Struve function of order 0.
   itmodstruve0 -- Integral of the modified Struve function of order 0.


Raw statistical functions
-------------------------

.. seealso:: :mod:`scipy.stats`: Friendly versions of these functions.

Binomial distribution
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   bdtr         -- Binomial distribution cumulative distribution function.
   bdtrc        -- Binomial distribution survival function.
   bdtri        -- Inverse function to `bdtr` with respect to `p`.
   bdtrik       -- Inverse function to `bdtr` with respect to `k`.
   bdtrin       -- Inverse function to `bdtr` with respect to `n`.

Beta distribution
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   btdtria      -- Inverse of `betainc` with respect to `a`.
   btdtrib      -- Inverse of `betainc` with respect to `b`.

F distribution
^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   fdtr         -- F cumulative distribution function.
   fdtrc        -- F survival function.
   fdtri        -- The `p`-th quantile of the F-distribution.
   fdtridfd     -- Inverse to `fdtr` vs dfd.

Gamma distribution
^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   gdtr         -- Gamma distribution cumulative distribution function.
   gdtrc        -- Gamma distribution survival function.
   gdtria       -- Inverse of `gdtr` vs a.
   gdtrib       -- Inverse of `gdtr` vs b.
   gdtrix       -- Inverse of `gdtr` vs x.

Negative binomial distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   nbdtr        -- Negative binomial cumulative distribution function.
   nbdtrc       -- Negative binomial survival function.
   nbdtri       -- Inverse of `nbdtr` vs `p`.
   nbdtrik      -- Inverse of `nbdtr` vs `k`.
   nbdtrin      -- Inverse of `nbdtr` vs `n`.

Noncentral F distribution
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   ncfdtr       -- Cumulative distribution function of the non-central F distribution.
   ncfdtridfd   -- Calculate degrees of freedom (denominator) for the noncentral F-distribution.
   ncfdtridfn   -- Calculate degrees of freedom (numerator) for the noncentral F-distribution.
   ncfdtri      -- Inverse cumulative distribution function of the non-central F distribution.
   ncfdtrinc    -- Calculate non-centrality parameter for non-central F distribution.

Noncentral t distribution
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   nctdtr       -- Cumulative distribution function of the non-central `t` distribution.
   nctdtridf    -- Calculate degrees of freedom for non-central t distribution.
   nctdtrit     -- Inverse cumulative distribution function of the non-central t distribution.
   nctdtrinc    -- Calculate non-centrality parameter for non-central t distribution.

Normal distribution
^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   nrdtrimn     -- Calculate mean of normal distribution given other params.
   nrdtrisd     -- Calculate standard deviation of normal distribution given other params.
   ndtr         -- Normal cumulative distribution function.
   log_ndtr     -- Logarithm of normal cumulative distribution function.
   ndtri        -- Inverse of `ndtr` vs x.
   ndtri_exp    -- Inverse of `log_ndtr` vs x.

Poisson distribution
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   pdtr         -- Poisson cumulative distribution function.
   pdtrc        -- Poisson survival function.
   pdtri        -- Inverse to `pdtr` vs m.
   pdtrik       -- Inverse to `pdtr` vs k.

Student t distribution
^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   stdtr        -- Student t distribution cumulative distribution function.
   stdtridf     -- Inverse of `stdtr` vs df.
   stdtrit      -- Inverse of `stdtr` vs `t`.

Chi square distribution
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   chdtr        -- Chi square cumulative distribution function.
   chdtrc       -- Chi square survival function.
   chdtri       -- Inverse to `chdtrc`.
   chdtriv      -- Inverse to `chdtr` vs `v`.

Non-central chi square distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   chndtr       -- Non-central chi square cumulative distribution function.
   chndtridf    -- Inverse to `chndtr` vs `df`.
   chndtrinc    -- Inverse to `chndtr` vs `nc`.
   chndtrix     -- Inverse to `chndtr` vs `x`.

Kolmogorov distribution
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   smirnov      -- Kolmogorov-Smirnov complementary cumulative distribution function.
   smirnovi     -- Inverse to `smirnov`.
   kolmogorov   -- Complementary cumulative distribution function of Kolmogorov distribution.
   kolmogi      -- Inverse function to `kolmogorov`.

Box-Cox transformation
^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   boxcox       -- Compute the Box-Cox transformation.
   boxcox1p     -- Compute the Box-Cox transformation of 1 + `x`.
   inv_boxcox   -- Compute the inverse of the Box-Cox transformation.
   inv_boxcox1p -- Compute the inverse of the Box-Cox transformation.


Sigmoidal functions
^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   logit        -- Logit ufunc for ndarrays.
   expit        -- Logistic sigmoid function.
   log_expit    -- Logarithm of the logistic sigmoid function.

Miscellaneous
^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   tklmbda      -- Tukey-Lambda cumulative distribution function.
   owens_t      -- Owen's T Function.


Information Theory functions
----------------------------

.. autosummary::
   :toctree: generated/

   entr         -- Elementwise function for computing entropy.
   rel_entr     -- Elementwise function for computing relative entropy.
   kl_div       -- Elementwise function for computing Kullback-Leibler divergence.
   huber        -- Huber loss function.
   pseudo_huber -- Pseudo-Huber loss function.


Gamma and related functions
---------------------------

.. autosummary::
   :toctree: generated/

   gamma        -- Gamma function.
   gammaln      -- Logarithm of the absolute value of the Gamma function for real inputs.
   loggamma     -- Principal branch of the logarithm of the Gamma function.
   gammasgn     -- Sign of the gamma function.
   gammainc     -- Regularized lower incomplete gamma function.
   gammaincinv  -- Inverse to `gammainc`.
   gammaincc    -- Regularized upper incomplete gamma function.
   gammainccinv -- Inverse to `gammaincc`.
   beta         -- Beta function.
   betaln       -- Natural logarithm of absolute value of beta function.
   betainc      -- Incomplete beta integral.
   betaincc     -- Complemented incomplete beta integral.
   betaincinv   -- Inverse function to beta integral.
   betainccinv  -- Inverse of the complemented incomplete beta integral.
   psi          -- The digamma function.
   rgamma       -- Gamma function inverted.
   polygamma    -- Polygamma function n.
   multigammaln -- Returns the log of multivariate gamma, also sometimes called the generalized gamma.
   digamma      -- psi(x[, out]).
   poch         -- Rising factorial (z)_m.

Error function and Fresnel integrals
------------------------------------

.. autosummary::
   :toctree: generated/

   erf           -- Returns the error function of complex argument.
   erfc          -- Complementary error function, ``1 - erf(x)``.
   erfcx         -- Scaled complementary error function, ``exp(x**2) * erfc(x)``.
   erfi          -- Imaginary error function, ``-i erf(i z)``.
   erfinv        -- Inverse function for erf.
   erfcinv       -- Inverse function for erfc.
   wofz          -- Faddeeva function.
   dawsn         -- Dawson's integral.
   fresnel       -- Fresnel sin and cos integrals.
   fresnel_zeros -- Compute nt complex zeros of sine and cosine Fresnel integrals S(z) and C(z).
   modfresnelp   -- Modified Fresnel positive integrals.
   modfresnelm   -- Modified Fresnel negative integrals.
   voigt_profile -- Voigt profile.

The following functions do not accept NumPy arrays (they are not
universal functions):

.. autosummary::
   :toctree: generated/

   erf_zeros      -- Compute nt complex zeros of error function erf(z).
   fresnelc_zeros -- Compute nt complex zeros of cosine Fresnel integral C(z).
   fresnels_zeros -- Compute nt complex zeros of sine Fresnel integral S(z).

Legendre functions
------------------

.. autosummary::
   :toctree: generated/

   legendre_p                 -- Legendre polynomials of the first kind.
   legendre_p_all             -- All Legendre polynomials of the first kind up to a specified order.
   assoc_legendre_p           -- Associated Legendre polynomials of the first kind.
   assoc_legendre_p_all       -- All associated Legendre polynomials of the first kind up to a specified order and degree.
   sph_legendre_p             -- Spherical Legendre polynomials of the first kind.
   sph_legendre_p_all         -- All spherical Legendre polynomials of the first kind up to a specified order and degree.
   sph_harm_y                 -- Spherical harmonics.
   sph_harm_y_all             -- All spherical harmonics up to a specified order and degree.

The following functions are in the process of being deprecated in favor of the above,
which provide a more flexible and consistent interface.

.. autosummary::
   :toctree: generated/

   lpmv                       -- Associated Legendre function of integer order and real degree.
   sph_harm                   -- Compute spherical harmonics.
   clpmn                      -- Associated Legendre function of the first kind for complex arguments.
   lpn                        -- Legendre function of the first kind.
   lqn                        -- Legendre function of the second kind.
   lpmn                       -- Sequence of associated Legendre functions of the first kind.
   lqmn                       -- Sequence of associated Legendre functions of the second kind.

Ellipsoidal harmonics
---------------------

.. autosummary::
   :toctree: generated/

   ellip_harm   -- Ellipsoidal harmonic functions E^p_n(l).
   ellip_harm_2 -- Ellipsoidal harmonic functions F^p_n(l).
   ellip_normal -- Ellipsoidal harmonic normalization constants gamma^p_n.

Orthogonal polynomials
----------------------

The following functions evaluate values of orthogonal polynomials:

.. autosummary::
   :toctree: generated/

   assoc_laguerre   -- Compute the generalized (associated) Laguerre polynomial of degree n and order k.
   eval_legendre    -- Evaluate Legendre polynomial at a point.
   eval_chebyt      -- Evaluate Chebyshev polynomial of the first kind at a point.
   eval_chebyu      -- Evaluate Chebyshev polynomial of the second kind at a point.
   eval_chebyc      -- Evaluate Chebyshev polynomial of the first kind on [-2, 2] at a point.
   eval_chebys      -- Evaluate Chebyshev polynomial of the second kind on [-2, 2] at a point.
   eval_jacobi      -- Evaluate Jacobi polynomial at a point.
   eval_laguerre    -- Evaluate Laguerre polynomial at a point.
   eval_genlaguerre -- Evaluate generalized Laguerre polynomial at a point.
   eval_hermite     -- Evaluate physicist's Hermite polynomial at a point.
   eval_hermitenorm -- Evaluate probabilist's (normalized) Hermite polynomial at a point.
   eval_gegenbauer  -- Evaluate Gegenbauer polynomial at a point.
   eval_sh_legendre -- Evaluate shifted Legendre polynomial at a point.
   eval_sh_chebyt   -- Evaluate shifted Chebyshev polynomial of the first kind at a point.
   eval_sh_chebyu   -- Evaluate shifted Chebyshev polynomial of the second kind at a point.
   eval_sh_jacobi   -- Evaluate shifted Jacobi polynomial at a point.

The following functions compute roots and quadrature weights for
orthogonal polynomials:

.. autosummary::
   :toctree: generated/

   roots_legendre    -- Gauss-Legendre quadrature.
   roots_chebyt      -- Gauss-Chebyshev (first kind) quadrature.
   roots_chebyu      -- Gauss-Chebyshev (second kind) quadrature.
   roots_chebyc      -- Gauss-Chebyshev (first kind) quadrature.
   roots_chebys      -- Gauss-Chebyshev (second kind) quadrature.
   roots_jacobi      -- Gauss-Jacobi quadrature.
   roots_laguerre    -- Gauss-Laguerre quadrature.
   roots_genlaguerre -- Gauss-generalized Laguerre quadrature.
   roots_hermite     -- Gauss-Hermite (physicist's) quadrature.
   roots_hermitenorm -- Gauss-Hermite (statistician's) quadrature.
   roots_gegenbauer  -- Gauss-Gegenbauer quadrature.
   roots_sh_legendre -- Gauss-Legendre (shifted) quadrature.
   roots_sh_chebyt   -- Gauss-Chebyshev (first kind, shifted) quadrature.
   roots_sh_chebyu   -- Gauss-Chebyshev (second kind, shifted) quadrature.
   roots_sh_jacobi   -- Gauss-Jacobi (shifted) quadrature.

The functions below, in turn, return the polynomial coefficients in
``orthopoly1d`` objects, which function similarly as `numpy.poly1d`.
The ``orthopoly1d`` class also has an attribute ``weights``, which returns
the roots, weights, and total weights for the appropriate form of Gaussian
quadrature. These are returned in an ``n x 3`` array with roots in the first
column, weights in the second column, and total weights in the final column.
Note that ``orthopoly1d`` objects are converted to `~numpy.poly1d` when doing
arithmetic, and lose information of the original orthogonal polynomial.

.. autosummary::
   :toctree: generated/

   legendre    -- Legendre polynomial.
   chebyt      -- Chebyshev polynomial of the first kind.
   chebyu      -- Chebyshev polynomial of the second kind.
   chebyc      -- Chebyshev polynomial of the first kind on :math:`[-2, 2]`.
   chebys      -- Chebyshev polynomial of the second kind on :math:`[-2, 2]`.
   jacobi      -- Jacobi polynomial.
   laguerre    -- Laguerre polynomial.
   genlaguerre -- Generalized (associated) Laguerre polynomial.
   hermite     -- Physicist's Hermite polynomial.
   hermitenorm -- Normalized (probabilist's) Hermite polynomial.
   gegenbauer  -- Gegenbauer (ultraspherical) polynomial.
   sh_legendre -- Shifted Legendre polynomial.
   sh_chebyt   -- Shifted Chebyshev polynomial of the first kind.
   sh_chebyu   -- Shifted Chebyshev polynomial of the second kind.
   sh_jacobi   -- Shifted Jacobi polynomial.

.. warning::

   Computing values of high-order polynomials (around ``order > 20``) using
   polynomial coefficients is numerically unstable. To evaluate polynomial
   values, the ``eval_*`` functions should be used instead.


Hypergeometric functions
------------------------

.. autosummary::
   :toctree: generated/

   hyp2f1 -- Gauss hypergeometric function 2F1(a, b; c; z).
   hyp1f1 -- Confluent hypergeometric function 1F1(a, b; x).
   hyperu -- Confluent hypergeometric function U(a, b, x) of the second kind.
   hyp0f1 -- Confluent hypergeometric limit function 0F1.


Parabolic cylinder functions
----------------------------

.. autosummary::
   :toctree: generated/

   pbdv -- Parabolic cylinder function D.
   pbvv -- Parabolic cylinder function V.
   pbwa -- Parabolic cylinder function W.

The following functions do not accept NumPy arrays (they are not
universal functions):

.. autosummary::
   :toctree: generated/

   pbdv_seq -- Parabolic cylinder functions Dv(x) and derivatives.
   pbvv_seq -- Parabolic cylinder functions Vv(x) and derivatives.
   pbdn_seq -- Parabolic cylinder functions Dn(z) and derivatives.

Mathieu and related functions
-----------------------------

.. autosummary::
   :toctree: generated/

   mathieu_a -- Characteristic value of even Mathieu functions.
   mathieu_b -- Characteristic value of odd Mathieu functions.

The following functions do not accept NumPy arrays (they are not
universal functions):

.. autosummary::
   :toctree: generated/

   mathieu_even_coef -- Fourier coefficients for even Mathieu and modified Mathieu functions.
   mathieu_odd_coef  -- Fourier coefficients for even Mathieu and modified Mathieu functions.

The following return both function and first derivative:

.. autosummary::
   :toctree: generated/

   mathieu_cem     -- Even Mathieu function and its derivative.
   mathieu_sem     -- Odd Mathieu function and its derivative.
   mathieu_modcem1 -- Even modified Mathieu function of the first kind and its derivative.
   mathieu_modcem2 -- Even modified Mathieu function of the second kind and its derivative.
   mathieu_modsem1 -- Odd modified Mathieu function of the first kind and its derivative.
   mathieu_modsem2 -- Odd modified Mathieu function of the second kind and its derivative.

Spheroidal wave functions
-------------------------

.. autosummary::
   :toctree: generated/

   pro_ang1   -- Prolate spheroidal angular function of the first kind and its derivative.
   pro_rad1   -- Prolate spheroidal radial function of the first kind and its derivative.
   pro_rad2   -- Prolate spheroidal radial function of the second kind and its derivative.
   obl_ang1   -- Oblate spheroidal angular function of the first kind and its derivative.
   obl_rad1   -- Oblate spheroidal radial function of the first kind and its derivative.
   obl_rad2   -- Oblate spheroidal radial function of the second kind and its derivative.
   pro_cv     -- Characteristic value of prolate spheroidal function.
   obl_cv     -- Characteristic value of oblate spheroidal function.
   pro_cv_seq -- Characteristic values for prolate spheroidal wave functions.
   obl_cv_seq -- Characteristic values for oblate spheroidal wave functions.

The following functions require pre-computed characteristic value:

.. autosummary::
   :toctree: generated/

   pro_ang1_cv -- Prolate spheroidal angular function pro_ang1 for precomputed characteristic value.
   pro_rad1_cv -- Prolate spheroidal radial function pro_rad1 for precomputed characteristic value.
   pro_rad2_cv -- Prolate spheroidal radial function pro_rad2 for precomputed characteristic value.
   obl_ang1_cv -- Oblate spheroidal angular function obl_ang1 for precomputed characteristic value.
   obl_rad1_cv -- Oblate spheroidal radial function obl_rad1 for precomputed characteristic value.
   obl_rad2_cv -- Oblate spheroidal radial function obl_rad2 for precomputed characteristic value.

Kelvin functions
----------------

.. autosummary::
   :toctree: generated/

   kelvin       -- Kelvin functions as complex numbers.
   kelvin_zeros -- Compute nt zeros of all Kelvin functions.
   ber          -- Kelvin function ber.
   bei          -- Kelvin function bei
   berp         -- Derivative of the Kelvin function `ber`.
   beip         -- Derivative of the Kelvin function `bei`.
   ker          -- Kelvin function ker.
   kei          -- Kelvin function ker.
   kerp         -- Derivative of the Kelvin function ker.
   keip         -- Derivative of the Kelvin function kei.

The following functions do not accept NumPy arrays (they are not
universal functions):

.. autosummary::
   :toctree: generated/

   ber_zeros  -- Compute nt zeros of the Kelvin function ber(x).
   bei_zeros  -- Compute nt zeros of the Kelvin function bei(x).
   berp_zeros -- Compute nt zeros of the Kelvin function ber'(x).
   beip_zeros -- Compute nt zeros of the Kelvin function bei'(x).
   ker_zeros  -- Compute nt zeros of the Kelvin function ker(x).
   kei_zeros  -- Compute nt zeros of the Kelvin function kei(x).
   kerp_zeros -- Compute nt zeros of the Kelvin function ker'(x).
   keip_zeros -- Compute nt zeros of the Kelvin function kei'(x).

Combinatorics
-------------

.. autosummary::
   :toctree: generated/

   comb -- The number of combinations of N things taken k at a time.
   perm -- Permutations of N things taken k at a time, i.e., k-permutations of N.
   stirling2 -- Stirling numbers of the second kind.

Lambert W and related functions
-------------------------------

.. autosummary::
   :toctree: generated/

   lambertw    -- Lambert W function.
   wrightomega -- Wright Omega function.

Other special functions
-----------------------

.. autosummary::
   :toctree: generated/

   agm         -- Arithmetic, Geometric Mean.
   bernoulli   -- Bernoulli numbers B0..Bn (inclusive).
   binom       -- Binomial coefficient
   diric       -- Periodic sinc function, also called the Dirichlet function.
   euler       -- Euler numbers E0..En (inclusive).
   expn        -- Exponential integral E_n.
   exp1        -- Exponential integral E_1 of complex argument z.
   expi        -- Exponential integral Ei.
   factorial   -- The factorial of a number or array of numbers.
   factorial2  -- Double factorial.
   factorialk  -- Multifactorial of n of order k, n(!!...!).
   shichi      -- Hyperbolic sine and cosine integrals.
   sici        -- Sine and cosine integrals.
   softmax     -- Softmax function.
   log_softmax -- Logarithm of softmax function.
   spence      -- Spence's function, also known as the dilogarithm.
   zeta        -- Riemann zeta function.
   zetac       -- Riemann zeta function minus 1.
   softplus    -- Softplus function.

Convenience functions
---------------------

.. autosummary::
   :toctree: generated/

   cbrt      -- Cube root of `x`.
   exp10     -- 10**x.
   exp2      -- 2**x.
   radian    -- Convert from degrees to radians.
   cosdg     -- Cosine of the angle `x` given in degrees.
   sindg     -- Sine of angle given in degrees.
   tandg     -- Tangent of angle x given in degrees.
   cotdg     -- Cotangent of the angle `x` given in degrees.
   log1p     -- Calculates log(1+x) for use when `x` is near zero.
   expm1     -- ``exp(x) - 1`` for use when `x` is near zero.
   cosm1     -- ``cos(x) - 1`` for use when `x` is near zero.
   powm1     -- ``x**y - 1`` for use when `y` is near zero or `x` is near 1.
   round     -- Round to nearest integer.
   xlogy     -- Compute ``x*log(y)`` so that the result is 0 if ``x = 0``.
   xlog1py   -- Compute ``x*log1p(y)`` so that the result is 0 if ``x = 0``.
   logsumexp -- Compute the log of the sum of exponentials of input elements.
   exprel    -- Relative error exponential, (exp(x)-1)/x, for use when `x` is near zero.
   sinc      -- Return the sinc function.

"""  # noqa: E501

import os
import warnings


def _load_libsf_error_state():
    """Load libsf_error_state.dll shared library on Windows

    libsf_error_state manages shared state used by
    ``scipy.special.seterr`` and ``scipy.special.geterr`` so that these
    can work consistently between special functions provided by different
    extension modules. This shared library is installed in scipy/special
    alongside this __init__.py file. Due to lack of rpath support, Windows
    cannot find shared libraries installed within wheels. To circumvent this,
    we pre-load ``lib_sf_error_state.dll`` when on Windows.

    The logic for this function was borrowed from the function ``make_init``
    in `scipy/tools/openblas_support.py`:
    https://github.com/scipy/scipy/blob/bb92c8014e21052e7dde67a76b28214dd1dcb94a/tools/openblas_support.py#L239-L274
    """  # noqa: E501
    if os.name == "nt":
        try:
            from ctypes import WinDLL
            basedir = os.path.dirname(__file__)
        except:  # noqa: E722
            pass
        else:
            dll_path = os.path.join(basedir, "libsf_error_state.dll")
            if os.path.exists(dll_path):
                WinDLL(dll_path)


_load_libsf_error_state()


from ._sf_error import SpecialFunctionWarning, SpecialFunctionError

from . import _ufuncs
from ._ufuncs import *

# Replace some function definitions from _ufuncs to add Array API support
from ._support_alternative_backends import (
    log_ndtr, ndtr, ndtri, erf, erfc, i0, i0e, i1, i1e, gammaln,
    gammainc, gammaincc, logit, expit, entr, rel_entr, xlogy,
    chdtr, chdtrc, betainc, betaincc, stdtr)

from . import _basic
from ._basic import *

from ._logsumexp import logsumexp, softmax, log_softmax

from . import _multiufuncs
from ._multiufuncs import *

from . import _orthogonal
from ._orthogonal import *

from ._spfun_stats import multigammaln
from ._ellip_harm import (
    ellip_harm,
    ellip_harm_2,
    ellip_normal
)
from ._lambertw import lambertw
from ._spherical_bessel import (
    spherical_jn,
    spherical_yn,
    spherical_in,
    spherical_kn
)

# Deprecated namespaces, to be removed in v2.0.0
from . import add_newdocs, basic, orthogonal, specfun, sf_error, spfun_stats

# We replace some function definitions from _ufuncs with those from
# _support_alternative_backends above, but those are all listed in _ufuncs.__all__,
# so there is no need to consider _support_alternative_backends.__all__ here.
__all__ = _ufuncs.__all__ + _basic.__all__ + _orthogonal.__all__ + _multiufuncs.__all__
__all__ += [
    'SpecialFunctionWarning',
    'SpecialFunctionError',
    'logsumexp',
    'softmax',
    'log_softmax',
    'multigammaln',
    'ellip_harm',
    'ellip_harm_2',
    'ellip_normal',
    'lambertw',
    'spherical_jn',
    'spherical_yn',
    'spherical_in',
    'spherical_kn',
]

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester


def _get_include():
    """This function is for development purposes only.

    This function could disappear or its behavior could change at any time.
    """
    import os
    return os.path.dirname(__file__)

<|endoftext|>import os
import sys
import time
from itertools import zip_longest

import numpy as np
from numpy.testing import assert_
import pytest

from scipy.special._testutils import assert_func_equal

try:
    import mpmath
except ImportError:
    pass


# ------------------------------------------------------------------------------
# Machinery for systematic tests with mpmath
# ------------------------------------------------------------------------------

class Arg:
    """Generate a set of numbers on the real axis, concentrating on
    'interesting' regions and covering all orders of magnitude.

    """

    def __init__(self, a=-np.inf, b=np.inf, inclusive_a=True, inclusive_b=True):
        if a > b:
            raise ValueError("a should be less than or equal to b")
        if a == -np.inf:
            a = -0.5*np.finfo(float).max
        if b == np.inf:
            b = 0.5*np.finfo(float).max
        self.a, self.b = a, b

        self.inclusive_a, self.inclusive_b = inclusive_a, inclusive_b

    def _positive_values(self, a, b, n):
        if a < 0:
            raise ValueError("a should be positive")

        # Try to put half of the points into a linspace between a and
        # 10 the other half in a logspace.
        if n % 2 == 0:
            nlogpts = n//2
            nlinpts = nlogpts
        else:
            nlogpts = n//2
            nlinpts = nlogpts + 1

        if a >= 10:
            # Outside of linspace range; just return a logspace.
            pts = np.logspace(np.log10(a), np.log10(b), n)
        elif a > 0 and b < 10:
            # Outside of logspace range; just return a linspace
            pts = np.linspace(a, b, n)
        elif a > 0:
            # Linspace between a and 10 and a logspace between 10 and
            # b.
            linpts = np.linspace(a, 10, nlinpts, endpoint=False)
            logpts = np.logspace(1, np.log10(b), nlogpts)
            pts = np.hstack((linpts, logpts))
        elif a == 0 and b <= 10:
            # Linspace between 0 and b and a logspace between 0 and
            # the smallest positive point of the linspace
            linpts = np.linspace(0, b, nlinpts)
            if linpts.size > 1:
                right = np.log10(linpts[1])
            else:
                right = -30
            logpts = np.logspace(-30, right, nlogpts, endpoint=False)
            pts = np.hstack((logpts, linpts))
        else:
            # Linspace between 0 and 10, logspace between 0 and the
            # smallest positive point of the linspace, and a logspace
            # between 10 and b.
            if nlogpts % 2 == 0:
                nlogpts1 = nlogpts//2
                nlogpts2 = nlogpts1
            else:
                nlogpts1 = nlogpts//2
                nlogpts2 = nlogpts1 + 1
            linpts = np.linspace(0, 10, nlinpts, endpoint=False)
            if linpts.size > 1:
                right = np.log10(linpts[1])
            else:
                right = -30
            logpts1 = np.logspace(-30, right, nlogpts1, endpoint=False)
            logpts2 = np.logspace(1, np.log10(b), nlogpts2)
            pts = np.hstack((logpts1, linpts, logpts2))

        return np.sort(pts)

    def values(self, n):
        """Return an array containing n numbers."""
        a, b = self.a, self.b
        if a == b:
            return np.zeros(n)

        if not self.inclusive_a:
            n += 1
        if not self.inclusive_b:
            n += 1

        if n % 2 == 0:
            n1 = n//2
            n2 = n1
        else:
            n1 = n//2
            n2 = n1 + 1

        if a >= 0:
            pospts = self._positive_values(a, b, n)
            negpts = []
        elif b <= 0:
            pospts = []
            negpts = -self._positive_values(-b, -a, n)
        else:
            pospts = self._positive_values(0, b, n1)
            negpts = -self._positive_values(0, -a, n2 + 1)
            # Don't want to get zero twice
            negpts = negpts[1:]
        pts = np.hstack((negpts[::-1], pospts))

        if not self.inclusive_a:
            pts = pts[1:]
        if not self.inclusive_b:
            pts = pts[:-1]
        return pts


class FixedArg:
    def __init__(self, values):
        self._values = np.asarray(values)

    def values(self, n):
        return self._values


class ComplexArg:
    def __init__(self, a=complex(-np.inf, -np.inf), b=complex(np.inf, np.inf)):
        self.real = Arg(a.real, b.real)
        self.imag = Arg(a.imag, b.imag)

    def values(self, n):
        m = int(np.floor(np.sqrt(n)))
        x = self.real.values(m)
        y = self.imag.values(m + 1)
        return (x[:,None] + 1j*y[None,:]).ravel()


class IntArg:
    def __init__(self, a=-1000, b=1000):
        self.a = a
        self.b = b

    def values(self, n):
        v1 = Arg(self.a, self.b).values(max(1 + n//2, n-5)).astype(int)
        v2 = np.arange(-5, 5)
        v = np.unique(np.r_[v1, v2])
        v = v[(v >= self.a) & (v < self.b)]
        return v


def get_args(argspec, n):
    if isinstance(argspec, np.ndarray):
        args = argspec.copy()
    else:
        nargs = len(argspec)
        ms = np.asarray(
            [1.5 if isinstance(spec, ComplexArg) else 1.0 for spec in argspec]
        )
        ms = (n**(ms/sum(ms))).astype(int) + 1

        args = [spec.values(m) for spec, m in zip(argspec, ms)]
        args = np.array(np.broadcast_arrays(*np.ix_(*args))).reshape(nargs, -1).T

    return args


class MpmathData:
    def __init__(self, scipy_func, mpmath_func, arg_spec, name=None,
                 dps=None, prec=None, n=None, rtol=1e-7, atol=1e-300,
                 ignore_inf_sign=False, distinguish_nan_and_inf=True,
                 nan_ok=True, param_filter=None):

        # mpmath tests are really slow (see gh-6989).  Use a small number of
        # points by default, increase back to 5000 (old default) if XSLOW is
        # set
        if n is None:
            try:
                is_xslow = int(os.environ.get('SCIPY_XSLOW', '0'))
            except ValueError:
                is_xslow = False

            n = 5000 if is_xslow else 500

        self.scipy_func = scipy_func
        self.mpmath_func = mpmath_func
        self.arg_spec = arg_spec
        self.dps = dps
        self.prec = prec
        self.n = n
        self.rtol = rtol
        self.atol = atol
        self.ignore_inf_sign = ignore_inf_sign
        self.nan_ok = nan_ok
        if isinstance(self.arg_spec, np.ndarray):
            self.is_complex = np.issubdtype(self.arg_spec.dtype, np.complexfloating)
        else:
            self.is_complex = any(
                [isinstance(arg, ComplexArg) for arg in self.arg_spec]
            )
        self.ignore_inf_sign = ignore_inf_sign
        self.distinguish_nan_and_inf = distinguish_nan_and_inf
        if not name or name == '<lambda>':
            name = getattr(scipy_func, '__name__', None)
        if not name or name == '<lambda>':
            name = getattr(mpmath_func, '__name__', None)
        self.name = name
        self.param_filter = param_filter

    def check(self):
        np.random.seed(1234)

        # Generate values for the arguments
        argarr = get_args(self.arg_spec, self.n)

        # Check
        old_dps, old_prec = mpmath.mp.dps, mpmath.mp.prec
        try:
            if self.dps is not None:
                dps_list = [self.dps]
            else:
                dps_list = [20]
            if self.prec is not None:
                mpmath.mp.prec = self.prec

            # Proper casting of mpmath input and output types. Using
            # native mpmath types as inputs gives improved precision
            # in some cases.
            if np.issubdtype(argarr.dtype, np.complexfloating):
                pytype = mpc2complex

                def mptype(x):
                    return mpmath.mpc(complex(x))
            else:
                def mptype(x):
                    return mpmath.mpf(float(x))

                def pytype(x):
                    if abs(x.imag) > 1e-16*(1 + abs(x.real)):
                        return np.nan
                    else:
                        return mpf2float(x.real)

            # Try out different dps until one (or none) works
            for j, dps in enumerate(dps_list):
                mpmath.mp.dps = dps

                try:
                    assert_func_equal(
                        self.scipy_func,
                        lambda *a: pytype(self.mpmath_func(*map(mptype, a))),
                        argarr,
                        vectorized=False,
                        rtol=self.rtol,
                        atol=self.atol,
                        ignore_inf_sign=self.ignore_inf_sign,
                        distinguish_nan_and_inf=self.distinguish_nan_and_inf,
                        nan_ok=self.nan_ok,
                        param_filter=self.param_filter
                    )
                    break
                except AssertionError:
                    if j >= len(dps_list)-1:
                        # reraise the Exception
                        tp, value, tb = sys.exc_info()
                        if value.__traceback__ is not tb:
                            raise value.with_traceback(tb)
                        raise value
        finally:
            mpmath.mp.dps, mpmath.mp.prec = old_dps, old_prec

    def __repr__(self):
        if self.is_complex:
            return f"<MpmathData: {self.name} (complex)>"
        else:
            return f"<MpmathData: {self.name}>"


def assert_mpmath_equal(*a, **kw):
    d = MpmathData(*a, **kw)
    d.check()


def nonfunctional_tooslow(func):
    return pytest.mark.skip(
        reason="    Test not yet functional (too slow), needs more work."
    )(func)


# ------------------------------------------------------------------------------
# Tools for dealing with mpmath quirks
# ------------------------------------------------------------------------------

def mpf2float(x):
    """
    Convert an mpf to the nearest floating point number. Just using
    float directly doesn't work because of results like this:

    with mp.workdps(50):
        float(mpf("0.99999999999999999")) = 0.9999999999999999

    """
    return float(mpmath.nstr(x, 17, min_fixed=0, max_fixed=0))


def mpc2complex(x):
    return complex(mpf2float(x.real), mpf2float(x.imag))


def trace_args(func):
    def tofloat(x):
        if isinstance(x, mpmath.mpc):
            return complex(x)
        else:
            return float(x)

    def wrap(*a, **kw):
        sys.stderr.write(f"{tuple(map(tofloat, a))!r}: ")
        sys.stderr.flush()
        try:
            r = func(*a, **kw)
            sys.stderr.write(f"-> {r!r}")
        finally:
            sys.stderr.write("\n")
            sys.stderr.flush()
        return r
    return wrap


try:
    import signal
    POSIX = ('setitimer' in dir(signal))
except ImportError:
    POSIX = False


class TimeoutError(Exception):
    pass


def time_limited(timeout=0.5, return_val=np.nan, use_sigalrm=True):
    """
    Decorator for setting a timeout for pure-Python functions.

    If the function does not return within `timeout` seconds, the
    value `return_val` is returned instead.

    On POSIX this uses SIGALRM by default. On non-POSIX, settrace is
    used. Do not use this with threads: the SIGALRM implementation
    does probably not work well. The settrace implementation only
    traces the current thread.

    The settrace implementation slows down execution speed. Slowdown
    by a factor around 10 is probably typical.
    """
    if POSIX and use_sigalrm:
        def sigalrm_handler(signum, frame):
            raise TimeoutError()

        def deco(func):
            def wrap(*a, **kw):
                old_handler = signal.signal(signal.SIGALRM, sigalrm_handler)
                signal.setitimer(signal.ITIMER_REAL, timeout)
                try:
                    return func(*a, **kw)
                except TimeoutError:
                    return return_val
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old_handler)
            return wrap
    else:
        def deco(func):
            def wrap(*a, **kw):
                start_time = time.time()

                def trace(frame, event, arg):
                    if time.time() - start_time > timeout:
                        raise TimeoutError()
                    return trace
                sys.settrace(trace)
                try:
                    return func(*a, **kw)
                except TimeoutError:
                    sys.settrace(None)
                    return return_val
                finally:
                    sys.settrace(None)
            return wrap
    return deco


def exception_to_nan(func):
    """Decorate function to return nan if it raises an exception"""
    def wrap(*a, **kw):
        try:
            return func(*a, **kw)
        except Exception:
            return np.nan
    return wrap


def inf_to_nan(func):
    """Decorate function to return nan if it returns inf"""
    def wrap(*a, **kw):
        v = func(*a, **kw)
        if not np.isfinite(v):
            return np.nan
        return v
    return wrap


def mp_assert_allclose(res, std, atol=0, rtol=1e-17):
    """
    Compare lists of mpmath.mpf's or mpmath.mpc's directly so that it
    can be done to higher precision than double.
    """
    failures = []
    for k, (resval, stdval) in enumerate(zip_longest(res, std)):
        if resval is None or stdval is None:
            raise ValueError('Lengths of inputs res and std are not equal.')
        if mpmath.fabs(resval - stdval) > atol + rtol*mpmath.fabs(stdval):
            failures.append((k, resval, stdval))

    nfail = len(failures)
    if nfail > 0:
        ndigits = int(abs(np.log10(rtol)))
        msg = [""]
        msg.append(f"Bad results ({nfail} out of {k + 1}) for the following points:")
        for k, resval, stdval in failures:
            resrep = mpmath.nstr(resval, ndigits, min_fixed=0, max_fixed=0)
            stdrep = mpmath.nstr(stdval, ndigits, min_fixed=0, max_fixed=0)
            if stdval == 0:
                rdiff = "inf"
            else:
                rdiff = mpmath.fabs((resval - stdval)/stdval)
                rdiff = mpmath.nstr(rdiff, 3)
            msg.append(f"{k}: {resrep} != {stdrep} (rdiff {rdiff})")
        assert_(False, "\n".join(msg))
<|endoftext|>import math
import numpy as np
from scipy._lib._util import _asarray_validated
from scipy._lib._array_api import (
    array_namespace,
    xp_size,
    xp_broadcast_promote,
    xp_copy,
    xp_float_to_complex,
    is_complex,
)
from scipy._lib import array_api_extra as xpx

__all__ = ["logsumexp", "softmax", "log_softmax"]


def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    """Compute the log of the sum of exponentials of input elements.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes over which the sum is taken. By default `axis` is None,
        and all elements are summed.

        .. versionadded:: 0.11.0
    b : array-like, optional
        Scaling factor for exp(`a`) must be of the same shape as `a` or
        broadcastable to `a`. These values may be negative in order to
        implement subtraction.

        .. versionadded:: 0.12.0
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the original array.

        .. versionadded:: 0.15.0
    return_sign : bool, optional
        If this is set to True, the result will be a pair containing sign
        information; if False, results that are negative will be returned
        as NaN. Default is False (no sign information).

        .. versionadded:: 0.16.0

    Returns
    -------
    res : ndarray
        The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
        more stable way. If `b` is given then ``np.log(np.sum(b*np.exp(a)))``
        is returned. If ``return_sign`` is True, ``res`` contains the log of
        the absolute value of the argument.
    sgn : ndarray
        If ``return_sign`` is True, this will be an array of floating-point
        numbers matching res containing +1, 0, -1 (for real-valued inputs)
        or a complex phase (for complex inputs). This gives the sign of the
        argument of the logarithm in ``res``.
        If ``return_sign`` is False, only one result is returned.

    See Also
    --------
    numpy.logaddexp, numpy.logaddexp2

    Notes
    -----
    NumPy has a logaddexp function which is very similar to `logsumexp`, but
    only handles two arguments. `logaddexp.reduce` is similar to this
    function, but may be less stable.

    The logarithm is a multivalued function: for each :math:`x` there is an
    infinite number of :math:`z` such that :math:`exp(z) = x`. The convention
    is to return the :math:`z` whose imaginary part lies in :math:`(-pi, pi]`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import logsumexp
    >>> a = np.arange(10)
    >>> logsumexp(a)
    9.4586297444267107
    >>> np.log(np.sum(np.exp(a)))
    9.4586297444267107

    With weights

    >>> a = np.arange(10)
    >>> b = np.arange(10, 0, -1)
    >>> logsumexp(a, b=b)
    9.9170178533034665
    >>> np.log(np.sum(b*np.exp(a)))
    9.9170178533034647

    Returning a sign flag

    >>> logsumexp([1,2],b=[1,-1],return_sign=True)
    (1.5413248546129181, -1.0)

    Notice that `logsumexp` does not directly support masked arrays. To use it
    on a masked array, convert the mask into zero weights:

    >>> a = np.ma.array([np.log(2), 2, np.log(3)],
    ...                  mask=[False, True, False])
    >>> b = (~a.mask).astype(int)
    >>> logsumexp(a.data, b=b), np.log(5)
    1.6094379124341005, 1.6094379124341005

    """
    xp = array_namespace(a, b)
    a, b = xp_broadcast_promote(a, b, ensure_writeable=True, force_floating=True, xp=xp)
    a = xpx.atleast_nd(a, ndim=1, xp=xp)
    b = xpx.atleast_nd(b, ndim=1, xp=xp) if b is not None else b
    axis = tuple(range(a.ndim)) if axis is None else axis

    if xp_size(a) != 0:
        with np.errstate(divide='ignore', invalid='ignore'):  # log of zero is OK
            out, sgn = _logsumexp(a, b, axis=axis, return_sign=return_sign, xp=xp)
    else:
        shape = np.asarray(a.shape)  # NumPy is convenient for shape manipulation
        shape[axis] = 1
        out = xp.full(tuple(shape), -xp.inf, dtype=a.dtype)
        sgn = xp.sign(out)

    if xp.isdtype(out.dtype, 'complex floating'):
        if return_sign:
            real = xp.real(sgn)
            imag = xp_float_to_complex(_wrap_radians(xp.imag(sgn), xp))
            sgn = real + imag*1j
        else:
            real = xp.real(out)
            imag = xp_float_to_complex(_wrap_radians(xp.imag(out), xp))
            out = real + imag*1j

    # Deal with shape details - reducing dimensions and convert 0-D to scalar for NumPy
    out = xp.squeeze(out, axis=axis) if not keepdims else out
    sgn = xp.squeeze(sgn, axis=axis) if (sgn is not None and not keepdims) else sgn
    out = out[()] if out.ndim == 0 else out
    sgn = sgn[()] if (sgn is not None and sgn.ndim == 0) else sgn

    return (out, sgn) if return_sign else out


def _wrap_radians(x, xp=None):
    xp = array_namespace(x) if xp is None else xp
    # Wrap radians to (-pi, pi] interval
    out = -((-x + math.pi) % (2 * math.pi) - math.pi)
    # preserve relative precision
    no_wrap = xp.abs(x) < xp.pi
    out[no_wrap] = x[no_wrap]
    return out


def _elements_and_indices_with_max_real(a, axis=-1, xp=None):
    # This is an array-API compatible `max` function that works something
    # like `np.max` for complex input. The important part is that it finds
    # the element with maximum real part. When there are multiple complex values
    # with this real part, it doesn't matter which we choose.
    # We could use `argmax` on real component, but array API doesn't yet have
    # `take_along_axis`, and even if it did, we would have problems with axis tuples.
    # Feel free to rewrite! It's ugly, but it's not the purpose of the PR, and
    # it gets the job done.
    xp = array_namespace(a) if xp is None else xp

    if xp.isdtype(a.dtype, "complex floating"):
        # select all elements with max real part.
        real_a = xp.real(a)
        max = xp.max(real_a, axis=axis, keepdims=True)
        mask = real_a == max

        # Of those, choose one arbitrarily. This is a reasonably
        # simple, array-API compatible way of doing so that doesn't
        # have a problem with `axis` being a tuple or None.
        i = xp.reshape(xp.arange(xp_size(a)), a.shape)
        i[~mask] = -1
        max_i = xp.max(i, axis=axis, keepdims=True)
        mask = i == max_i
        a = xp_copy(a)
        a[~mask] = 0
        max = xp.sum(a, axis=axis, dtype=a.dtype, keepdims=True)
    else:
        max = xp.max(a, axis=axis, keepdims=True)
        mask = a == max

    return xp.asarray(max), xp.asarray(mask)


def _sign(x, xp):
    return x / xp.where(x == 0, xp.asarray(1, dtype=x.dtype), xp.abs(x))


def _logsumexp(a, b, axis, return_sign, xp):

    # This has been around for about a decade, so let's consider it a feature:
    # Even if element of `a` is infinite or NaN, it adds nothing to the sum if
    # the corresponding weight is zero.
    if b is not None:
        a[b == 0] = -xp.inf

    # Find element with maximum real part, since this is what affects the magnitude
    # of the exponential. Possible enhancement: include log of `b` magnitude in `a`.
    a_max, i_max = _elements_and_indices_with_max_real(a, axis=axis, xp=xp)

    # for precision, these terms are separated out of the main sum.
    a[i_max] = -xp.inf
    i_max_dt = xp.astype(i_max, a.dtype)
    # This is an inefficient way of getting `m` because it is the sum of a sparse
    # array; however, this is the simplest way I can think of to get the right shape.
    m = (xp.sum(i_max_dt, axis=axis, keepdims=True, dtype=a.dtype) if b is None
         else xp.sum(b * i_max_dt, axis=axis, keepdims=True, dtype=a.dtype))

    # Arithmetic between infinities will introduce NaNs.
    # `+ a_max` at the end naturally corrects for removing them here.
    shift = xp.where(xp.isfinite(a_max), a_max, xp.asarray(0, dtype=a_max.dtype))

    # Shift, exponentiate, scale, and sum
    exp = b * xp.exp(a - shift) if b is not None else xp.exp(a - shift)
    s = xp.sum(exp, axis=axis, keepdims=True, dtype=exp.dtype)
    s = xp.where(s == 0, s, s/m)

    # Separate sign/magnitude information
    # Originally, this was only performed if `return_sign=True`.
    # However, this is also needed if any elements of `m < 0` or `s < -1`.
    # An improvement would be to perform the calculations only on these entries.

    # Use the numpy>=2.0 convention for sign.
    # When all array libraries agree, this can become sng = xp.sign(s).
    sgn = _sign(s + 1, xp=xp) * _sign(m, xp=xp)

    if xp.isdtype(s.dtype, "real floating"):
        # The log functions need positive arguments
        s = xp.where(s < -1, -s - 2, s)
        m = xp.abs(m)
    else:
        # `a_max` can have a sign component for complex input
        sgn = sgn * xp.exp(xp.imag(a_max) * xp.asarray(1.0j, dtype=a_max.dtype))

    # Take log and undo shift
    out = xp.log1p(s) + xp.log(m) + a_max

    if return_sign:
        if is_complex(out, xp):
            out = xp.real(out)
    elif xp.isdtype(out.dtype, 'real floating'):
        out[sgn < 0] = xp.nan

    return out, sgn


def softmax(x, axis=None):
    r"""Compute the softmax function.

    The softmax function transforms each element of a collection by
    computing the exponential of each element divided by the sum of the
    exponentials of all the elements. That is, if `x` is a one-dimensional
    numpy array::

        softmax(x) = np.exp(x)/sum(np.exp(x))

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int or tuple of ints, optional
        Axis to compute values along. Default is None and softmax will be
        computed over the entire array `x`.

    Returns
    -------
    s : ndarray
        An array the same shape as `x`. The result will sum to 1 along the
        specified axis.

    Notes
    -----
    The formula for the softmax function :math:`\sigma(x)` for a vector
    :math:`x = \{x_0, x_1, ..., x_{n-1}\}` is

    .. math:: \sigma(x)_j = \frac{e^{x_j}}{\sum_k e^{x_k}}

    The `softmax` function is the gradient of `logsumexp`.

    The implementation uses shifting to avoid overflow. See [1]_ for more
    details.

    .. versionadded:: 1.2.0

    References
    ----------
    .. [1] P. Blanchard, D.J. Higham, N.J. Higham, "Accurately computing the
       log-sum-exp and softmax functions", IMA Journal of Numerical Analysis,
       Vol.41(4), :doi:`10.1093/imanum/draa038`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import softmax
    >>> np.set_printoptions(precision=5)

    >>> x = np.array([[1, 0.5, 0.2, 3],
    ...               [1,  -1,   7, 3],
    ...               [2,  12,  13, 3]])
    ...

    Compute the softmax transformation over the entire array.

    >>> m = softmax(x)
    >>> m
    array([[  4.48309e-06,   2.71913e-06,   2.01438e-06,   3.31258e-05],
           [  4.48309e-06,   6.06720e-07,   1.80861e-03,   3.31258e-05],
           [  1.21863e-05,   2.68421e-01,   7.29644e-01,   3.31258e-05]])

    >>> m.sum()
    1.0

    Compute the softmax transformation along the first axis (i.e., the
    columns).

    >>> m = softmax(x, axis=0)

    >>> m
    array([[  2.11942e-01,   1.01300e-05,   2.75394e-06,   3.33333e-01],
           [  2.11942e-01,   2.26030e-06,   2.47262e-03,   3.33333e-01],
           [  5.76117e-01,   9.99988e-01,   9.97525e-01,   3.33333e-01]])

    >>> m.sum(axis=0)
    array([ 1.,  1.,  1.,  1.])

    Compute the softmax transformation along the second axis (i.e., the rows).

    >>> m = softmax(x, axis=1)
    >>> m
    array([[  1.05877e-01,   6.42177e-02,   4.75736e-02,   7.82332e-01],
           [  2.42746e-03,   3.28521e-04,   9.79307e-01,   1.79366e-02],
           [  1.22094e-05,   2.68929e-01,   7.31025e-01,   3.31885e-05]])

    >>> m.sum(axis=1)
    array([ 1.,  1.,  1.])

    """
    x = _asarray_validated(x, check_finite=False)
    x_max = np.amax(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)


def log_softmax(x, axis=None):
    r"""Compute the logarithm of the softmax function.

    In principle::

        log_softmax(x) = log(softmax(x))

    but using a more accurate implementation.

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int or tuple of ints, optional
        Axis to compute values along. Default is None and softmax will be
        computed over the entire array `x`.

    Returns
    -------
    s : ndarray or scalar
        An array with the same shape as `x`. Exponential of the result will
        sum to 1 along the specified axis. If `x` is a scalar, a scalar is
        returned.

    Notes
    -----
    `log_softmax` is more accurate than ``np.log(softmax(x))`` with inputs that
    make `softmax` saturate (see examples below).

    .. versionadded:: 1.5.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import log_softmax
    >>> from scipy.special import softmax
    >>> np.set_printoptions(precision=5)

    >>> x = np.array([1000.0, 1.0])

    >>> y = log_softmax(x)
    >>> y
    array([   0., -999.])

    >>> with np.errstate(divide='ignore'):
    ...   y = np.log(softmax(x))
    ...
    >>> y
    array([  0., -inf])

    """

    x = _asarray_validated(x, check_finite=False)

    x_max = np.amax(x, axis=axis, keepdims=True)

    if x_max.ndim > 0:
        x_max[~np.isfinite(x_max)] = 0
    elif not np.isfinite(x_max):
        x_max = 0

    tmp = x - x_max
    exp_tmp = np.exp(tmp)

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        s = np.sum(exp_tmp, axis=axis, keepdims=True)
        out = np.log(s)

    out = tmp - out
    return out
<|endoftext|>"""
A collection of functions to find the weights and abscissas for
Gaussian Quadrature.

These calculations are done by finding the eigenvalues of a
tridiagonal matrix whose entries are dependent on the coefficients
in the recursion formula for the orthogonal polynomials with the
corresponding weighting function over the interval.

Many recursion relations for orthogonal polynomials are given:

.. math::

    a1n f_{n+1} (x) = (a2n + a3n x ) f_n (x) - a4n f_{n-1} (x)

The recursion relation of interest is

.. math::

    P_{n+1} (x) = (x - A_n) P_n (x) - B_n P_{n-1} (x)

where :math:`P` has a different normalization than :math:`f`.

The coefficients can be found as:

.. math::

    A_n = -a2n / a3n
    \\qquad
    B_n = ( a4n / a3n \\sqrt{h_n-1 / h_n})^2

where

.. math::

    h_n = \\int_a^b w(x) f_n(x)^2

assume:

.. math::

    P_0 (x) = 1
    \\qquad
    P_{-1} (x) == 0

For the mathematical background, see [golub.welsch-1969-mathcomp]_ and
[abramowitz.stegun-1965]_.

References
----------
.. [golub.welsch-1969-mathcomp]
   Golub, Gene H, and John H Welsch. 1969. Calculation of Gauss
   Quadrature Rules. *Mathematics of Computation* 23, 221-230+s1--s10.

.. [abramowitz.stegun-1965]
   Abramowitz, Milton, and Irene A Stegun. (1965) *Handbook of
   Mathematical Functions: with Formulas, Graphs, and Mathematical
   Tables*. Gaithersburg, MD: National Bureau of Standards.
   http://www.math.sfu.ca/~cbm/aands/

.. [townsend.trogdon.olver-2014]
   Townsend, A. and Trogdon, T. and Olver, S. (2014)
   *Fast computation of Gauss quadrature nodes and
   weights on the whole real line*. :arXiv:`1410.5286`.

.. [townsend.trogdon.olver-2015]
   Townsend, A. and Trogdon, T. and Olver, S. (2015)
   *Fast computation of Gauss quadrature nodes and
   weights on the whole real line*.
   IMA Journal of Numerical Analysis
   :doi:`10.1093/imanum/drv002`.
"""
#
# Author:  Travis Oliphant 2000
# Updated Sep. 2003 (fixed bugs --- tested to be accurate)

# SciPy imports.
import numpy as np
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
                   hstack, arccos, arange)
from scipy import linalg
from scipy.special import airy

# Local imports.
# There is no .pyi file for _specfun
from . import _specfun  # type: ignore
from . import _ufuncs
_gam = _ufuncs.gamma

_polyfuns = ['legendre', 'chebyt', 'chebyu', 'chebyc', 'chebys',
             'jacobi', 'laguerre', 'genlaguerre', 'hermite',
             'hermitenorm', 'gegenbauer', 'sh_legendre', 'sh_chebyt',
             'sh_chebyu', 'sh_jacobi']

# Correspondence between new and old names of root functions
_rootfuns_map = {'roots_legendre': 'p_roots',
                 'roots_chebyt': 't_roots',
                 'roots_chebyu': 'u_roots',
                 'roots_chebyc': 'c_roots',
                 'roots_chebys': 's_roots',
                 'roots_jacobi': 'j_roots',
                 'roots_laguerre': 'l_roots',
                 'roots_genlaguerre': 'la_roots',
                 'roots_hermite': 'h_roots',
                 'roots_hermitenorm': 'he_roots',
                 'roots_gegenbauer': 'cg_roots',
                 'roots_sh_legendre': 'ps_roots',
                 'roots_sh_chebyt': 'ts_roots',
                 'roots_sh_chebyu': 'us_roots',
                 'roots_sh_jacobi': 'js_roots'}

__all__ = _polyfuns + list(_rootfuns_map.keys())


class orthopoly1d(np.poly1d):

    def __init__(self, roots, weights=None, hn=1.0, kn=1.0, wfunc=None,
                 limits=None, monic=False, eval_func=None):
        equiv_weights = [weights[k] / wfunc(roots[k]) for
                         k in range(len(roots))]
        mu = sqrt(hn)
        if monic:
            evf = eval_func
            if evf:
                knn = kn
                def eval_func(x):
                    return evf(x) / knn
            mu = mu / abs(kn)
            kn = 1.0

        # compute coefficients from roots, then scale
        poly = np.poly1d(roots, r=True)
        np.poly1d.__init__(self, poly.coeffs * float(kn))

        self.weights = np.array(list(zip(roots, weights, equiv_weights)))
        self.weight_func = wfunc
        self.limits = limits
        self.normcoef = mu

        # Note: eval_func will be discarded on arithmetic
        self._eval_func = eval_func

    def __call__(self, v):
        if self._eval_func and not isinstance(v, np.poly1d):
            return self._eval_func(v)
        else:
            return np.poly1d.__call__(self, v)

    def _scale(self, p):
        if p == 1.0:
            return
        self._coeffs *= p

        evf = self._eval_func
        if evf:
            self._eval_func = lambda x: evf(x) * p
        self.normcoef *= p


def _gen_roots_and_weights(n, mu0, an_func, bn_func, f, df, symmetrize, mu):
    """[x,w] = gen_roots_and_weights(n,an_func,sqrt_bn_func,mu)

    Returns the roots (x) of an nth order orthogonal polynomial,
    and weights (w) to use in appropriate Gaussian quadrature with that
    orthogonal polynomial.

    The polynomials have the recurrence relation
          P_n+1(x) = (x - A_n) P_n(x) - B_n P_n-1(x)

    an_func(n)          should return A_n
    sqrt_bn_func(n)     should return sqrt(B_n)
    mu ( = h_0 )        is the integral of the weight over the orthogonal
                        interval
    """
    k = np.arange(n, dtype='d')
    c = np.zeros((2, n))
    c[0,1:] = bn_func(k[1:])
    c[1,:] = an_func(k)
    x = linalg.eigvals_banded(c, overwrite_a_band=True)

    # improve roots by one application of Newton's method
    y = f(n, x)
    dy = df(n, x)
    x -= y/dy

    # fm and dy may contain very large/small values, so we
    # log-normalize them to maintain precision in the product fm*dy
    fm = f(n-1, x)
    log_fm = np.log(np.abs(fm))
    log_dy = np.log(np.abs(dy))
    fm /= np.exp((log_fm.max() + log_fm.min()) / 2.)
    dy /= np.exp((log_dy.max() + log_dy.min()) / 2.)
    w = 1.0 / (fm * dy)

    if symmetrize:
        w = (w + w[::-1]) / 2
        x = (x - x[::-1]) / 2

    w *= mu0 / w.sum()

    if mu:
        return x, w, mu0
    else:
        return x, w

# Jacobi Polynomials 1               P^(alpha,beta)_n(x)


def roots_jacobi(n, alpha, beta, mu=False):
    r"""Gauss-Jacobi quadrature.

    Compute the sample points and weights for Gauss-Jacobi
    quadrature. The sample points are the roots of the nth degree
    Jacobi polynomial, :math:`P^{\alpha, \beta}_n(x)`. These sample
    points and weights correctly integrate polynomials of degree
    :math:`2n - 1` or less over the interval :math:`[-1, 1]` with
    weight function :math:`w(x) = (1 - x)^{\alpha} (1 +
    x)^{\beta}`. See 22.2.1 in [AS]_ for details.

    Parameters
    ----------
    n : int
        quadrature order
    alpha : float
        alpha must be > -1
    beta : float
        beta must be > -1
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.fixed_quad

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    m = int(n)
    if n < 1 or n != m:
        raise ValueError("n must be a positive integer.")
    if alpha <= -1 or beta <= -1:
        raise ValueError("alpha and beta must be greater than -1.")

    if alpha == 0.0 and beta == 0.0:
        return roots_legendre(m, mu)
    if alpha == beta:
        return roots_gegenbauer(m, alpha+0.5, mu)

    if (alpha + beta) <= 1000:
        mu0 = 2.0**(alpha+beta+1) * _ufuncs.beta(alpha+1, beta+1)
    else:
        # Avoid overflows in pow and beta for very large parameters
        mu0 = np.exp((alpha + beta + 1) * np.log(2.0)
                     + _ufuncs.betaln(alpha+1, beta+1))
    a = alpha
    b = beta
    if a + b == 0.0:
        def an_func(k):
            return np.where(k == 0, (b - a) / (2 + a + b), 0.0)
    else:
        def an_func(k):
            return np.where(
                k == 0,
                (b - a) / (2 + a + b),
                (b * b - a * a) / ((2.0 * k + a + b) * (2.0 * k + a + b + 2))
            )

    def bn_func(k):
        return (
            2.0 / (2.0 * k + a + b)
            * np.sqrt((k + a) * (k + b) / (2 * k + a + b + 1))
            * np.where(k == 1, 1.0, np.sqrt(k * (k + a + b) / (2.0 * k + a + b - 1)))
        )

    def f(n, x):
        return _ufuncs.eval_jacobi(n, a, b, x)
    def df(n, x):
        return 0.5 * (n + a + b + 1) * _ufuncs.eval_jacobi(n - 1, a + 1, b + 1, x)
    return _gen_roots_and_weights(m, mu0, an_func, bn_func, f, df, False, mu)


def jacobi(n, alpha, beta, monic=False):
    r"""Jacobi polynomial.

    Defined to be the solution of

    .. math::
        (1 - x^2)\frac{d^2}{dx^2}P_n^{(\alpha, \beta)}
          + (\beta - \alpha - (\alpha + \beta + 2)x)
            \frac{d}{dx}P_n^{(\alpha, \beta)}
          + n(n + \alpha + \beta + 1)P_n^{(\alpha, \beta)} = 0

    for :math:`\alpha, \beta > -1`; :math:`P_n^{(\alpha, \beta)}` is a
    polynomial of degree :math:`n`.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    alpha : float
        Parameter, must be greater than -1.
    beta : float
        Parameter, must be greater than -1.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    P : orthopoly1d
        Jacobi polynomial.

    Notes
    -----
    For fixed :math:`\alpha, \beta`, the polynomials
    :math:`P_n^{(\alpha, \beta)}` are orthogonal over :math:`[-1, 1]`
    with weight function :math:`(1 - x)^\alpha(1 + x)^\beta`.

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    The Jacobi polynomials satisfy the recurrence relation:

    .. math::
        P_n^{(\alpha, \beta-1)}(x) - P_n^{(\alpha-1, \beta)}(x)
          = P_{n-1}^{(\alpha, \beta)}(x)

    This can be verified, for example, for :math:`\alpha = \beta = 2`
    and :math:`n = 1` over the interval :math:`[-1, 1]`:

    >>> import numpy as np
    >>> from scipy.special import jacobi
    >>> x = np.arange(-1.0, 1.0, 0.01)
    >>> np.allclose(jacobi(0, 2, 2)(x),
    ...             jacobi(1, 2, 1)(x) - jacobi(1, 1, 2)(x))
    True

    Plot of the Jacobi polynomial :math:`P_5^{(\alpha, -0.5)}` for
    different values of :math:`\alpha`:

    >>> import matplotlib.pyplot as plt
    >>> x = np.arange(-1.0, 1.0, 0.01)
    >>> fig, ax = plt.subplots()
    >>> ax.set_ylim(-2.0, 2.0)
    >>> ax.set_title(r'Jacobi polynomials $P_5^{(\alpha, -0.5)}$')
    >>> for alpha in np.arange(0, 4, 1):
    ...     ax.plot(x, jacobi(5, alpha, -0.5)(x), label=rf'$\alpha={alpha}$')
    >>> plt.legend(loc='best')
    >>> plt.show()

    """
    if n < 0:
        raise ValueError("n must be nonnegative.")

    def wfunc(x):
        return (1 - x) ** alpha * (1 + x) ** beta
    if n == 0:
        return orthopoly1d([], [], 1.0, 1.0, wfunc, (-1, 1), monic,
                           eval_func=np.ones_like)
    x, w, mu = roots_jacobi(n, alpha, beta, mu=True)
    ab1 = alpha + beta + 1.0
    hn = 2**ab1 / (2 * n + ab1) * _gam(n + alpha + 1)
    hn *= _gam(n + beta + 1.0) / _gam(n + 1) / _gam(n + ab1)
    kn = _gam(2 * n + ab1) / 2.0**n / _gam(n + 1) / _gam(n + ab1)
    # here kn = coefficient on x^n term
    p = orthopoly1d(x, w, hn, kn, wfunc, (-1, 1), monic,
                    lambda x: _ufuncs.eval_jacobi(n, alpha, beta, x))
    return p

# Jacobi Polynomials shifted         G_n(p,q,x)


def roots_sh_jacobi(n, p1, q1, mu=False):
    """Gauss-Jacobi (shifted) quadrature.

    Compute the sample points and weights for Gauss-Jacobi (shifted)
    quadrature. The sample points are the roots of the nth degree
    shifted Jacobi polynomial, :math:`G^{p,q}_n(x)`. These sample
    points and weights correctly integrate polynomials of degree
    :math:`2n - 1` or less over the interval :math:`[0, 1]` with
    weight function :math:`w(x) = (1 - x)^{p-q} x^{q-1}`. See 22.2.2
    in [AS]_ for details.

    Parameters
    ----------
    n : int
        quadrature order
    p1 : float
        (p1 - q1) must be > -1
    q1 : float
        q1 must be > 0
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.fixed_quad

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    if (p1-q1) <= -1 or q1 <= 0:
        message = "(p - q) must be greater than -1, and q must be greater than 0."
        raise ValueError(message)
    x, w, m = roots_jacobi(n, p1-q1, q1-1, True)
    x = (x + 1) / 2
    scale = 2.0**p1
    w /= scale
    m /= scale
    if mu:
        return x, w, m
    else:
        return x, w


def sh_jacobi(n, p, q, monic=False):
    r"""Shifted Jacobi polynomial.

    Defined by

    .. math::

        G_n^{(p, q)}(x)
          = \binom{2n + p - 1}{n}^{-1}P_n^{(p - q, q - 1)}(2x - 1),

    where :math:`P_n^{(\cdot, \cdot)}` is the nth Jacobi polynomial.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    p : float
        Parameter, must have :math:`p > q - 1`.
    q : float
        Parameter, must be greater than 0.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    G : orthopoly1d
        Shifted Jacobi polynomial.

    Notes
    -----
    For fixed :math:`p, q`, the polynomials :math:`G_n^{(p, q)}` are
    orthogonal over :math:`[0, 1]` with weight function :math:`(1 -
    x)^{p - q}x^{q - 1}`.

    """
    if n < 0:
        raise ValueError("n must be nonnegative.")

    def wfunc(x):
        return (1.0 - x) ** (p - q) * x ** (q - 1.0)
    if n == 0:
        return orthopoly1d([], [], 1.0, 1.0, wfunc, (-1, 1), monic,
                           eval_func=np.ones_like)
    n1 = n
    x, w = roots_sh_jacobi(n1, p, q)
    hn = _gam(n + 1) * _gam(n + q) * _gam(n + p) * _gam(n + p - q + 1)
    hn /= (2 * n + p) * (_gam(2 * n + p)**2)
    # kn = 1.0 in standard form so monic is redundant. Kept for compatibility.
    kn = 1.0
    pp = orthopoly1d(x, w, hn, kn, wfunc=wfunc, limits=(0, 1), monic=monic,
                     eval_func=lambda x: _ufuncs.eval_sh_jacobi(n, p, q, x))
    return pp

# Generalized Laguerre               L^(alpha)_n(x)


def roots_genlaguerre(n, alpha, mu=False):
    r"""Gauss-generalized Laguerre quadrature.

    Compute the sample points and weights for Gauss-generalized
    Laguerre quadrature. The sample points are the roots of the nth
    degree generalized Laguerre polynomial, :math:`L^{\alpha}_n(x)`.
    These sample points and weights correctly integrate polynomials of
    degree :math:`2n - 1` or less over the interval :math:`[0,
    \infty]` with weight function :math:`w(x) = x^{\alpha}
    e^{-x}`. See 22.3.9 in [AS]_ for details.

    Parameters
    ----------
    n : int
        quadrature order
    alpha : float
        alpha must be > -1
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.fixed_quad

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    m = int(n)
    if n < 1 or n != m:
        raise ValueError("n must be a positive integer.")
    if alpha < -1:
        raise ValueError("alpha must be greater than -1.")

    mu0 = _ufuncs.gamma(alpha + 1)

    if m == 1:
        x = np.array([alpha+1.0], 'd')
        w = np.array([mu0], 'd')
        if mu:
            return x, w, mu0
        else:
            return x, w

    def an_func(k):
        return 2 * k + alpha + 1
    def bn_func(k):
        return -np.sqrt(k * (k + alpha))
    def f(n, x):
        return _ufuncs.eval_genlaguerre(n, alpha, x)
    def df(n, x):
        return (n * _ufuncs.eval_genlaguerre(n, alpha, x)
                - (n + alpha) * _ufuncs.eval_genlaguerre(n - 1, alpha, x)) / x
    return _gen_roots_and_weights(m, mu0, an_func, bn_func, f, df, False, mu)


def genlaguerre(n, alpha, monic=False):
    r"""Generalized (associated) Laguerre polynomial.

    Defined to be the solution of

    .. math::
        x\frac{d^2}{dx^2}L_n^{(\alpha)}
          + (\alpha + 1 - x)\frac{d}{dx}L_n^{(\alpha)}
          + nL_n^{(\alpha)} = 0,

    where :math:`\alpha > -1`; :math:`L_n^{(\alpha)}` is a polynomial
    of degree :math:`n`.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    alpha : float
        Parameter, must be greater than -1.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    L : orthopoly1d
        Generalized Laguerre polynomial.

    See Also
    --------
    laguerre : Laguerre polynomial.
    hyp1f1 : confluent hypergeometric function

    Notes
    -----
    For fixed :math:`\alpha`, the polynomials :math:`L_n^{(\alpha)}`
    are orthogonal over :math:`[0, \infty)` with weight function
    :math:`e^{-x}x^\alpha`.

    The Laguerre polynomials are the special case where :math:`\alpha
    = 0`.

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    The generalized Laguerre polynomials are closely related to the confluent
    hypergeometric function :math:`{}_1F_1`:

        .. math::
            L_n^{(\alpha)} = \binom{n + \alpha}{n} {}_1F_1(-n, \alpha +1, x)

    This can be verified, for example,  for :math:`n = \alpha = 3` over the
    interval :math:`[-1, 1]`:

    >>> import numpy as np
    >>> from scipy.special import binom
    >>> from scipy.special import genlaguerre
    >>> from scipy.special import hyp1f1
    >>> x = np.arange(-1.0, 1.0, 0.01)
    >>> np.allclose(genlaguerre(3, 3)(x), binom(6, 3) * hyp1f1(-3, 4, x))
    True

    This is the plot of the generalized Laguerre polynomials
    :math:`L_3^{(\alpha)}` for some values of :math:`\alpha`:

    >>> import matplotlib.pyplot as plt
    >>> x = np.arange(-4.0, 12.0, 0.01)
    >>> fig, ax = plt.subplots()
    >>> ax.set_ylim(-5.0, 10.0)
    >>> ax.set_title(r'Generalized Laguerre polynomials $L_3^{\alpha}$')
    >>> for alpha in np.arange(0, 5):
    ...     ax.plot(x, genlaguerre(3, alpha)(x), label=rf'$L_3^{(alpha)}$')
    >>> plt.legend(loc='best')
    >>> plt.show()

    """
    if alpha <= -1:
        raise ValueError("alpha must be > -1")
    if n < 0:
        raise ValueError("n must be nonnegative.")

    if n == 0:
        n1 = n + 1
    else:
        n1 = n
    x, w = roots_genlaguerre(n1, alpha)
    def wfunc(x):
        return exp(-x) * x ** alpha
    if n == 0:
        x, w = [], []
    hn = _gam(n + alpha + 1) / _gam(n + 1)
    kn = (-1)**n / _gam(n + 1)
    p = orthopoly1d(x, w, hn, kn, wfunc, (0, inf), monic,
                    lambda x: _ufuncs.eval_genlaguerre(n, alpha, x))
    return p

# Laguerre                      L_n(x)


def roots_laguerre(n, mu=False):
    r"""Gauss-Laguerre quadrature.

    Compute the sample points and weights for Gauss-Laguerre
    quadrature. The sample points are the roots of the nth degree
    Laguerre polynomial, :math:`L_n(x)`. These sample points and
    weights correctly integrate polynomials of degree :math:`2n - 1`
    or less over the interval :math:`[0, \infty]` with weight function
    :math:`w(x) = e^{-x}`. See 22.2.13 in [AS]_ for details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.fixed_quad
    numpy.polynomial.laguerre.laggauss

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    return roots_genlaguerre(n, 0.0, mu=mu)


def laguerre(n, monic=False):
    r"""Laguerre polynomial.

    Defined to be the solution of

    .. math::
        x\frac{d^2}{dx^2}L_n + (1 - x)\frac{d}{dx}L_n + nL_n = 0;

    :math:`L_n` is a polynomial of degree :math:`n`.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    L : orthopoly1d
        Laguerre Polynomial.

    See Also
    --------
    genlaguerre : Generalized (associated) Laguerre polynomial.

    Notes
    -----
    The polynomials :math:`L_n` are orthogonal over :math:`[0,
    \infty)` with weight function :math:`e^{-x}`.

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    The Laguerre polynomials :math:`L_n` are the special case
    :math:`\alpha = 0` of the generalized Laguerre polynomials
    :math:`L_n^{(\alpha)}`.
    Let's verify it on the interval :math:`[-1, 1]`:

    >>> import numpy as np
    >>> from scipy.special import genlaguerre
    >>> from scipy.special import laguerre
    >>> x = np.arange(-1.0, 1.0, 0.01)
    >>> np.allclose(genlaguerre(3, 0)(x), laguerre(3)(x))
    True

    The polynomials :math:`L_n` also satisfy the recurrence relation:

    .. math::
        (n + 1)L_{n+1}(x) = (2n +1 -x)L_n(x) - nL_{n-1}(x)

    This can be easily checked on :math:`[0, 1]` for :math:`n = 3`:

    >>> x = np.arange(0.0, 1.0, 0.01)
    >>> np.allclose(4 * laguerre(4)(x),
    ...             (7 - x) * laguerre(3)(x) - 3 * laguerre(2)(x))
    True

    This is the plot of the first few Laguerre polynomials :math:`L_n`:

    >>> import matplotlib.pyplot as plt
    >>> x = np.arange(-1.0, 5.0, 0.01)
    >>> fig, ax = plt.subplots()
    >>> ax.set_ylim(-5.0, 5.0)
    >>> ax.set_title(r'Laguerre polynomials $L_n$')
    >>> for n in np.arange(0, 5):
    ...     ax.plot(x, laguerre(n)(x), label=rf'$L_{n}$')
    >>> plt.legend(loc='best')
    >>> plt.show()

    """
    if n < 0:
        raise ValueError("n must be nonnegative.")

    if n == 0:
        n1 = n + 1
    else:
        n1 = n
    x, w = roots_laguerre(n1)
    if n == 0:
        x, w = [], []
    hn = 1.0
    kn = (-1)**n / _gam(n + 1)
    p = orthopoly1d(x, w, hn, kn, lambda x: exp(-x), (0, inf), monic,
                    lambda x: _ufuncs.eval_laguerre(n, x))
    return p

# Hermite  1                         H_n(x)


def roots_hermite(n, mu=False):
    r"""Gauss-Hermite (physicist's) quadrature.

    Compute the sample points and weights for Gauss-Hermite
    quadrature. The sample points are the roots of the nth degree
    Hermite polynomial, :math:`H_n(x)`. These sample points and
    weights correctly integrate polynomials of degree :math:`2n - 1`
    or less over the interval :math:`[-\infty, \infty]` with weight
    function :math:`w(x) = e^{-x^2}`. See 22.2.14 in [AS]_ for
    details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.fixed_quad
    numpy.polynomial.hermite.hermgauss
    roots_hermitenorm

    Notes
    -----
    For small n up to 150 a modified version of the Golub-Welsch
    algorithm is used. Nodes are computed from the eigenvalue
    problem and improved by one step of a Newton iteration.
    The weights are computed from the well-known analytical formula.

    For n larger than 150 an optimal asymptotic algorithm is applied
    which computes nodes and weights in a numerically stable manner.
    The algorithm has linear runtime making computation for very
    large n (several thousand or more) feasible.

    References
    ----------
    .. [townsend.trogdon.olver-2014]
        Townsend, A. and Trogdon, T. and Olver, S. (2014)
        *Fast computation of Gauss quadrature nodes and
        weights on the whole real line*. :arXiv:`1410.5286`.
    .. [townsend.trogdon.olver-2015]
        Townsend, A. and Trogdon, T. and Olver, S. (2015)
        *Fast computation of Gauss quadrature nodes and
        weights on the whole real line*.
        IMA Journal of Numerical Analysis
        :doi:`10.1093/imanum/drv002`.
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    m = int(n)
    if n < 1 or n != m:
        raise ValueError("n must be a positive integer.")

    mu0 = np.sqrt(np.pi)
    if n <= 150:
        def an_func(k):
            return 0.0 * k
        def bn_func(k):
            return np.sqrt(k / 2.0)
        f = _ufuncs.eval_hermite
        def df(n, x):
            return 2.0 * n * _ufuncs.eval_hermite(n - 1, x)
        return _gen_roots_and_weights(m, mu0, an_func, bn_func, f, df, True, mu)
    else:
        nodes, weights = _roots_hermite_asy(m)
        if mu:
            return nodes, weights, mu0
        else:
            return nodes, weights


def _compute_tauk(n, k, maxit=5):
    """Helper function for Tricomi initial guesses

    For details, see formula 3.1 in lemma 3.1 in the
    original paper.

    Parameters
    ----------
    n : int
        Quadrature order
    k : ndarray of type int
        Index of roots :math:`\tau_k` to compute
    maxit : int
        Number of Newton maxit performed, the default
        value of 5 is sufficient.

    Returns
    -------
    tauk : ndarray
        Roots of equation 3.1

    See Also
    --------
    initial_nodes_a
    roots_hermite_asy
    """
    a = n % 2 - 0.5
    c = (4.0*floor(n/2.0) - 4.0*k + 3.0)*pi / (4.0*floor(n/2.0) + 2.0*a + 2.0)
    def f(x):
        return x - sin(x) - c
    def df(x):
        return 1.0 - cos(x)
    xi = 0.5*pi
    for i in range(maxit):
        xi = xi - f(xi)/df(xi)
    return xi


def _initial_nodes_a(n, k):
    r"""Tricomi initial guesses

    Computes an initial approximation to the square of the `k`-th
    (positive) root :math:`x_k` of the Hermite polynomial :math:`H_n`
    of order :math:`n`. The formula is the one from lemma 3.1 in the
    original paper. The guesses are accurate except in the region
    near :math:`\sqrt{2n + 1}`.

    Parameters
    ----------
    n : int
        Quadrature order
    k : ndarray of type int
        Index of roots to compute

    Returns
    -------
    xksq : ndarray
        Square of the approximate roots

    See Also
    --------
    initial_nodes
    roots_hermite_asy
    """
    tauk = _compute_tauk(n, k)
    sigk = cos(0.5*tauk)**2
    a = n % 2 - 0.5
    nu = 4.0*floor(n/2.0) + 2.0*a + 2.0
    # Initial approximation of Hermite roots (square)
    xksq = nu*sigk - 1.0/(3.0*nu) * (5.0/(4.0*(1.0-sigk)**2) - 1.0/(1.0-sigk) - 0.25)
    return xksq


def _initial_nodes_b(n, k):
    r"""Gatteschi initial guesses

    Computes an initial approximation to the square of the kth
    (positive) root :math:`x_k` of the Hermite polynomial :math:`H_n`
    of order :math:`n`. The formula is the one from lemma 3.2 in the
    original paper. The guesses are accurate in the region just
    below :math:`\sqrt{2n + 1}`.

    Parameters
    ----------
    n : int
        Quadrature order
    k : ndarray of type int
        Index of roots to compute

    Returns
    -------
    xksq : ndarray
        Square of the approximate root

    See Also
    --------
    initial_nodes
    roots_hermite_asy
    """
    a = n % 2 - 0.5
    nu = 4.0*floor(n/2.0) + 2.0*a + 2.0
    # Airy roots by approximation
    ak = _specfun.airyzo(k.max(), 1)[0][::-1]
    # Initial approximation of Hermite roots (square)
    xksq = (nu
            + 2.0**(2.0/3.0) * ak * nu**(1.0/3.0)
            + 1.0/5.0 * 2.0**(4.0/3.0) * ak**2 * nu**(-1.0/3.0)
            + (9.0/140.0 - 12.0/175.0 * ak**3) * nu**(-1.0)
            + (16.0/1575.0 * ak + 92.0/7875.0 * ak**4) * 2.0**(2.0/3.0) * nu**(-5.0/3.0)
            - (15152.0/3031875.0 * ak**5 + 1088.0/121275.0 * ak**2)
              * 2.0**(1.0/3.0) * nu**(-7.0/3.0))
    return xksq


def _initial_nodes(n):
    """Initial guesses for the Hermite roots

    Computes an initial approximation to the non-negative
    roots :math:`x_k` of the Hermite polynomial :math:`H_n`
    of order :math:`n`. The Tricomi and Gatteschi initial
    guesses are used in the region where they are accurate.

    Parameters
    ----------
    n : int
        Quadrature order

    Returns
    -------
    xk : ndarray
        Approximate roots

    See Also
    --------
    roots_hermite_asy
    """
    # Turnover point
    # linear polynomial fit to error of 10, 25, 40, ..., 1000 point rules
    fit = 0.49082003*n - 4.37859653
    turnover = around(fit).astype(int)
    # Compute all approximations
    ia = arange(1, int(floor(n*0.5)+1))
    ib = ia[::-1]
    xasq = _initial_nodes_a(n, ia[:turnover+1])
    xbsq = _initial_nodes_b(n, ib[turnover+1:])
    # Combine
    iv = sqrt(hstack([xasq, xbsq]))
    # Central node is always zero
    if n % 2 == 1:
        iv = hstack([0.0, iv])
    return iv


def _pbcf(n, theta):
    r"""Asymptotic series expansion of parabolic cylinder function

    The implementation is based on sections 3.2 and 3.3 from the
    original paper. Compared to the published version this code
    adds one more term to the asymptotic series. The detailed
    formulas can be found at [parabolic-asymptotics]_. The evaluation
    is done in a transformed variable :math:`\theta := \arccos(t)`
    where :math:`t := x / \mu` and :math:`\mu := \sqrt{2n + 1}`.

    Parameters
    ----------
    n : int
        Quadrature order
    theta : ndarray
        Transformed position variable

    Returns
    -------
    U : ndarray
        Value of the parabolic cylinder function :math:`U(a, \theta)`.
    Ud : ndarray
        Value of the derivative :math:`U^{\prime}(a, \theta)` of
        the parabolic cylinder function.

    See Also
    --------
    roots_hermite_asy

    References
    ----------
    .. [parabolic-asymptotics]
       https://dlmf.nist.gov/12.10#vii
    """
    st = sin(theta)
    ct = cos(theta)
    # https://dlmf.nist.gov/12.10#vii
    mu = 2.0*n + 1.0
    # https://dlmf.nist.gov/12.10#E23
    eta = 0.5*theta - 0.5*st*ct
    # https://dlmf.nist.gov/12.10#E39
    zeta = -(3.0*eta/2.0) ** (2.0/3.0)
    # https://dlmf.nist.gov/12.10#E40
    phi = (-zeta / st**2) ** (0.25)
    # Coefficients
    # https://dlmf.nist.gov/12.10#E43
    a0 = 1.0
    a1 = 0.10416666666666666667
    a2 = 0.08355034722222222222
    a3 = 0.12822657455632716049
    a4 = 0.29184902646414046425
    a5 = 0.88162726744375765242
    b0 = 1.0
    b1 = -0.14583333333333333333
    b2 = -0.09874131944444444444
    b3 = -0.14331205391589506173
    b4 = -0.31722720267841354810
    b5 = -0.94242914795712024914
    # Polynomials
    # https://dlmf.nist.gov/12.10#E9
    # https://dlmf.nist.gov/12.10#E10
    ctp = ct ** arange(16).reshape((-1,1))
    u0 = 1.0
    u1 = (1.0*ctp[3,:] - 6.0*ct) / 24.0
    u2 = (-9.0*ctp[4,:] + 249.0*ctp[2,:] + 145.0) / 1152.0
    u3 = (-4042.0*ctp[9,:] + 18189.0*ctp[7,:] - 28287.0*ctp[5,:]
          - 151995.0*ctp[3,:] - 259290.0*ct) / 414720.0
    u4 = (72756.0*ctp[10,:] - 321339.0*ctp[8,:] - 154982.0*ctp[6,:]
          + 50938215.0*ctp[4,:] + 122602962.0*ctp[2,:] + 12773113.0) / 39813120.0
    u5 = (82393456.0*ctp[15,:] - 617950920.0*ctp[13,:] + 1994971575.0*ctp[11,:]
          - 3630137104.0*ctp[9,:] + 4433574213.0*ctp[7,:] - 37370295816.0*ctp[5,:]
          - 119582875013.0*ctp[3,:] - 34009066266.0*ct) / 6688604160.0
    v0 = 1.0
    v1 = (1.0*ctp[3,:] + 6.0*ct) / 24.0
    v2 = (15.0*ctp[4,:] - 327.0*ctp[2,:] - 143.0) / 1152.0
    v3 = (-4042.0*ctp[9,:] + 18189.0*ctp[7,:] - 36387.0*ctp[5,:] 
          + 238425.0*ctp[3,:] + 259290.0*ct) / 414720.0
    v4 = (-121260.0*ctp[10,:] + 551733.0*ctp[8,:] - 151958.0*ctp[6,:]
          - 57484425.0*ctp[4,:] - 132752238.0*ctp[2,:] - 12118727) / 39813120.0
    v5 = (82393456.0*ctp[15,:] - 617950920.0*ctp[13,:] + 2025529095.0*ctp[11,:]
          - 3750839308.0*ctp[9,:] + 3832454253.0*ctp[7,:] + 35213253348.0*ctp[5,:]
          + 130919230435.0*ctp[3,:] + 34009066266*ct) / 6688604160.0
    # Airy Evaluation (Bi and Bip unused)
    Ai, Aip, Bi, Bip = airy(mu**(4.0/6.0) * zeta)
    # Prefactor for U
    P = 2.0*sqrt(pi) * mu**(1.0/6.0) * phi
    # Terms for U
    # https://dlmf.nist.gov/12.10#E42
    phip = phi ** arange(6, 31, 6).reshape((-1,1))
    A0 = b0*u0
    A1 = (b2*u0 + phip[0,:]*b1*u1 + phip[1,:]*b0*u2) / zeta**3
    A2 = (b4*u0 + phip[0,:]*b3*u1 + phip[1,:]*b2*u2 + phip[2,:]*b1*u3
          + phip[3,:]*b0*u4) / zeta**6
    B0 = -(a1*u0 + phip[0,:]*a0*u1) / zeta**2
    B1 = -(a3*u0 + phip[0,:]*a2*u1 + phip[1,:]*a1*u2 + phip[2,:]*a0*u3) / zeta**5
    B2 = -(a5*u0 + phip[0,:]*a4*u1 + phip[1,:]*a3*u2 + phip[2,:]*a2*u3
           + phip[3,:]*a1*u4 + phip[4,:]*a0*u5) / zeta**8
    # U
    # https://dlmf.nist.gov/12.10#E35
    U = P * (Ai * (A0 + A1/mu**2.0 + A2/mu**4.0) +
             Aip * (B0 + B1/mu**2.0 + B2/mu**4.0) / mu**(8.0/6.0))
    # Prefactor for derivative of U
    Pd = sqrt(2.0*pi) * mu**(2.0/6.0) / phi
    # Terms for derivative of U
    # https://dlmf.nist.gov/12.10#E46
    C0 = -(b1*v0 + phip[0,:]*b0*v1) / zeta
    C1 = -(b3*v0 + phip[0,:]*b2*v1 + phip[1,:]*b1*v2 + phip[2,:]*b0*v3) / zeta**4
    C2 = -(b5*v0 + phip[0,:]*b4*v1 + phip[1,:]*b3*v2 + phip[2,:]*b2*v3
           + phip[3,:]*b1*v4 + phip[4,:]*b0*v5) / zeta**7
    D0 = a0*v0
    D1 = (a2*v0 + phip[0,:]*a1*v1 + phip[1,:]*a0*v2) / zeta**3
    D2 = (a4*v0 + phip[0,:]*a3*v1 + phip[1,:]*a2*v2 + phip[2,:]*a1*v3
          + phip[3,:]*a0*v4) / zeta**6
    # Derivative of U
    # https://dlmf.nist.gov/12.10#E36
    Ud = Pd * (Ai * (C0 + C1/mu**2.0 + C2/mu**4.0) / mu**(4.0/6.0) +
               Aip * (D0 + D1/mu**2.0 + D2/mu**4.0))
    return U, Ud


def _newton(n, x_initial, maxit=5):
    """Newton iteration for polishing the asymptotic approximation
    to the zeros of the Hermite polynomials.

    Parameters
    ----------
    n : int
        Quadrature order
    x_initial : ndarray
        Initial guesses for the roots
    maxit : int
        Maximal number of Newton iterations.
        The default 5 is sufficient, usually
        only one or two steps are needed.

    Returns
    -------
    nodes : ndarray
        Quadrature nodes
    weights : ndarray
        Quadrature weights

    See Also
    --------
    roots_hermite_asy
    """
    # Variable transformation
    mu = sqrt(2.0*n + 1.0)
    t = x_initial / mu
    theta = arccos(t)
    # Newton iteration
    for i in range(maxit):
        u, ud = _pbcf(n, theta)
        dtheta = u / (sqrt(2.0) * mu * sin(theta) * ud)
        theta = theta + dtheta
        if max(abs(dtheta)) < 1e-14:
            break
    # Undo variable transformation
    x = mu * cos(theta)
    # Central node is always zero
    if n % 2 == 1:
        x[0] = 0.0
    # Compute weights
    w = exp(-x**2) / (2.0*ud**2)
    return x, w


def _roots_hermite_asy(n):
    r"""Gauss-Hermite (physicist's) quadrature for large n.

    Computes the sample points and weights for Gauss-Hermite quadrature.
    The sample points are the roots of the nth degree Hermite polynomial,
    :math:`H_n(x)`. These sample points and weights correctly integrate
    polynomials of degree :math:`2n - 1` or less over the interval
    :math:`[-\infty, \infty]` with weight function :math:`f(x) = e^{-x^2}`.

    This method relies on asymptotic expansions which work best for n > 150.
    The algorithm has linear runtime making computation for very large n
    feasible.

    Parameters
    ----------
    n : int
        quadrature order

    Returns
    -------
    nodes : ndarray
        Quadrature nodes
    weights : ndarray
        Quadrature weights

    See Also
    --------
    roots_hermite

    References
    ----------
    .. [townsend.trogdon.olver-2014]
       Townsend, A. and Trogdon, T. and Olver, S. (2014)
       *Fast computation of Gauss quadrature nodes and
       weights on the whole real line*. :arXiv:`1410.5286`.

    .. [townsend.trogdon.olver-2015]
       Townsend, A. and Trogdon, T. and Olver, S. (2015)
       *Fast computation of Gauss quadrature nodes and
       weights on the whole real line*.
       IMA Journal of Numerical Analysis
       :doi:`10.1093/imanum/drv002`.
    """
    iv = _initial_nodes(n)
    nodes, weights = _newton(n, iv)
    # Combine with negative parts
    if n % 2 == 0:
        nodes = hstack([-nodes[::-1], nodes])
        weights = hstack([weights[::-1], weights])
    else:
        nodes = hstack([-nodes[-1:0:-1], nodes])
        weights = hstack([weights[-1:0:-1], weights])
    # Scale weights
    weights *= sqrt(pi) / sum(weights)
    return nodes, weights


def hermite(n, monic=False):
    r"""Physicist's Hermite polynomial.

    Defined by

    .. math::

        H_n(x) = (-1)^ne^{x^2}\frac{d^n}{dx^n}e^{-x^2};

    :math:`H_n` is a polynomial of degree :math:`n`.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    H : orthopoly1d
        Hermite polynomial.

    Notes
    -----
    The polynomials :math:`H_n` are orthogonal over :math:`(-\infty,
    \infty)` with weight function :math:`e^{-x^2}`.

    Examples
    --------
    >>> from scipy import special
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> p_monic = special.hermite(3, monic=True)
    >>> p_monic
    poly1d([ 1. ,  0. , -1.5,  0. ])
    >>> p_monic(1)
    -0.49999999999999983
    >>> x = np.linspace(-3, 3, 400)
    >>> y = p_monic(x)
    >>> plt.plot(x, y)
    >>> plt.title("Monic Hermite polynomial of degree 3")
    >>> plt.xlabel("x")
    >>> plt.ylabel("H_3(x)")
    >>> plt.show()

    """
    if n < 0:
        raise ValueError("n must be nonnegative.")

    if n == 0:
        n1 = n + 1
    else:
        n1 = n
    x, w = roots_hermite(n1)
    def wfunc(x):
        return exp(-x * x)
    if n == 0:
        x, w = [], []
    hn = 2**n * _gam(n + 1) * sqrt(pi)
    kn = 2**n
    p = orthopoly1d(x, w, hn, kn, wfunc, (-inf, inf), monic,
                    lambda x: _ufuncs.eval_hermite(n, x))
    return p

# Hermite  2                         He_n(x)


def roots_hermitenorm(n, mu=False):
    r"""Gauss-Hermite (statistician's) quadrature.

    Compute the sample points and weights for Gauss-Hermite
    quadrature. The sample points are the roots of the nth degree
    Hermite polynomial, :math:`He_n(x)`. These sample points and
    weights correctly integrate polynomials of degree :math:`2n - 1`
    or less over the interval :math:`[-\infty, \infty]` with weight
    function :math:`w(x) = e^{-x^2/2}`. See 22.2.15 in [AS]_ for more
    details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.fixed_quad
    numpy.polynomial.hermite_e.hermegauss

    Notes
    -----
    For small n up to 150 a modified version of the Golub-Welsch
    algorithm is used. Nodes are computed from the eigenvalue
    problem and improved by one step of a Newton iteration.
    The weights are computed from the well-known analytical formula.

    For n larger than 150 an optimal asymptotic algorithm is used
    which computes nodes and weights in a numerical stable manner.
    The algorithm has linear runtime making computation for very
    large n (several thousand or more) feasible.

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    m = int(n)
    if n < 1 or n != m:
        raise ValueError("n must be a positive integer.")

    mu0 = np.sqrt(2.0*np.pi)
    if n <= 150:
        def an_func(k):
            return 0.0 * k
        def bn_func(k):
            return np.sqrt(k)
        f = _ufuncs.eval_hermitenorm
        def df(n, x):
            return n * _ufuncs.eval_hermitenorm(n - 1, x)
        return _gen_roots_and_weights(m, mu0, an_func, bn_func, f, df, True, mu)
    else:
        nodes, weights = _roots_hermite_asy(m)
        # Transform
        nodes *= sqrt(2)
        weights *= sqrt(2)
        if mu:
            return nodes, weights, mu0
        else:
            return nodes, weights


def hermitenorm(n, monic=False):
    r"""Normalized (probabilist's) Hermite polynomial.

    Defined by

    .. math::

        He_n(x) = (-1)^ne^{x^2/2}\frac{d^n}{dx^n}e^{-x^2/2};

    :math:`He_n` is a polynomial of degree :math:`n`.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    He : orthopoly1d
        Hermite polynomial.

    Notes
    -----

    The polynomials :math:`He_n` are orthogonal over :math:`(-\infty,
    \infty)` with weight function :math:`e^{-x^2/2}`.

    """
    if n < 0:
        raise ValueError("n must be nonnegative.")

    if n == 0:
        n1 = n + 1
    else:
        n1 = n
    x, w = roots_hermitenorm(n1)
    def wfunc(x):
        return exp(-x * x / 2.0)
    if n == 0:
        x, w = [], []
    hn = sqrt(2 * pi) * _gam(n + 1)
    kn = 1.0
    p = orthopoly1d(x, w, hn, kn, wfunc=wfunc, limits=(-inf, inf), monic=monic,
                    eval_func=lambda x: _ufuncs.eval_hermitenorm(n, x))
    return p

# The remainder of the polynomials can be derived from the ones above.

# Ultraspherical (Gegenbauer)        C^(alpha)_n(x)


def roots_gegenbauer(n, alpha, mu=False):
    r"""Gauss-Gegenbauer quadrature.

    Compute the sample points and weights for Gauss-Gegenbauer
    quadrature. The sample points are the roots of the nth degree
    Gegenbauer polynomial, :math:`C^{\alpha}_n(x)`. These sample
    points and weights correctly integrate polynomials of degree
    :math:`2n - 1` or less over the interval :math:`[-1, 1]` with
    weight function :math:`w(x) = (1 - x^2)^{\alpha - 1/2}`. See
    22.2.3 in [AS]_ for more details.

    Parameters
    ----------
    n : int
        quadrature order
    alpha : float
        alpha must be > -0.5
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.fixed_quad

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    m = int(n)
    if n < 1 or n != m:
        raise ValueError("n must be a positive integer.")
    if alpha < -0.5:
        raise ValueError("alpha must be greater than -0.5.")
    elif alpha == 0.0:
        # C(n,0,x) == 0 uniformly, however, as alpha->0, C(n,alpha,x)->T(n,x)
        # strictly, we should just error out here, since the roots are not
        # really defined, but we used to return something useful, so let's
        # keep doing so.
        return roots_chebyt(n, mu)

    if alpha <= 170:
        mu0 = (np.sqrt(np.pi) * _ufuncs.gamma(alpha + 0.5)) \
              / _ufuncs.gamma(alpha + 1)
    else:
        # For large alpha we use a Taylor series expansion around inf,
        # expressed as a 6th order polynomial of a^-1 and using Horner's
        # method to minimize computation and maximize precision
        inv_alpha = 1. / alpha
        coeffs = np.array([0.000207186, -0.00152206, -0.000640869,
                           0.00488281, 0.0078125, -0.125, 1.])
        mu0 = coeffs[0]
        for term in range(1, len(coeffs)):
            mu0 = mu0 * inv_alpha + coeffs[term]
        mu0 = mu0 * np.sqrt(np.pi / alpha)
    def an_func(k):
        return 0.0 * k
    def bn_func(k):
        return np.sqrt(k * (k + 2 * alpha - 1) / (4 * (k + alpha) * (k + alpha - 1)))
    def f(n, x):
        return _ufuncs.eval_gegenbauer(n, alpha, x)
    def df(n, x):
        return (
            -n * x * _ufuncs.eval_gegenbauer(n, alpha, x)
            + (n + 2 * alpha - 1) * _ufuncs.eval_gegenbauer(n - 1, alpha, x)
        ) / (1 - x ** 2)
    return _gen_roots_and_weights(m, mu0, an_func, bn_func, f, df, True, mu)


def gegenbauer(n, alpha, monic=False):
    r"""Gegenbauer (ultraspherical) polynomial.

    Defined to be the solution of

    .. math::
        (1 - x^2)\frac{d^2}{dx^2}C_n^{(\alpha)}
          - (2\alpha + 1)x\frac{d}{dx}C_n^{(\alpha)}
          + n(n + 2\alpha)C_n^{(\alpha)} = 0

    for :math:`\alpha > -1/2`; :math:`C_n^{(\alpha)}` is a polynomial
    of degree :math:`n`.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    alpha : float
        Parameter, must be greater than -0.5.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    C : orthopoly1d
        Gegenbauer polynomial.

    Notes
    -----
    The polynomials :math:`C_n^{(\alpha)}` are orthogonal over
    :math:`[-1,1]` with weight function :math:`(1 - x^2)^{(\alpha -
    1/2)}`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import special
    >>> import matplotlib.pyplot as plt

    We can initialize a variable ``p`` as a Gegenbauer polynomial using the
    `gegenbauer` function and evaluate at a point ``x = 1``.

    >>> p = special.gegenbauer(3, 0.5, monic=False)
    >>> p
    poly1d([ 2.5,  0. , -1.5,  0. ])
    >>> p(1)
    1.0

    To evaluate ``p`` at various points ``x`` in the interval ``(-3, 3)``,
    simply pass an array ``x`` to ``p`` as follows:

    >>> x = np.linspace(-3, 3, 400)
    >>> y = p(x)

    We can then visualize ``x, y`` using `matplotlib.pyplot`.

    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, y)
    >>> ax.set_title("Gegenbauer (ultraspherical) polynomial of degree 3")
    >>> ax.set_xlabel("x")
    >>> ax.set_ylabel("G_3(x)")
    >>> plt.show()

    """
    if not np.isfinite(alpha) or alpha <= -0.5 :
        raise ValueError("`alpha` must be a finite number greater than -1/2")
    base = jacobi(n, alpha - 0.5, alpha - 0.5, monic=monic)
    if monic or n == 0:
        return base
    #  Abrahmowitz and Stegan 22.5.20
    factor = (_gam(2*alpha + n) * _gam(alpha + 0.5) /
              _gam(2*alpha) / _gam(alpha + 0.5 + n))
    base._scale(factor)
    base.__dict__['_eval_func'] = lambda x: _ufuncs.eval_gegenbauer(float(n),
                                                                    alpha, x)
    return base

# Chebyshev of the first kind: T_n(x) =
#     n! sqrt(pi) / _gam(n+1./2)* P^(-1/2,-1/2)_n(x)
# Computed anew.


def roots_chebyt(n, mu=False):
    r"""Gauss-Chebyshev (first kind) quadrature.

    Computes the sample points and weights for Gauss-Chebyshev
    quadrature. The sample points are the roots of the nth degree
    Chebyshev polynomial of the first kind, :math:`T_n(x)`. These
    sample points and weights correctly integrate polynomials of
    degree :math:`2n - 1` or less over the interval :math:`[-1, 1]`
    with weight function :math:`w(x) = 1/\sqrt{1 - x^2}`. See 22.2.4
    in [AS]_ for more details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.fixed_quad
    numpy.polynomial.chebyshev.chebgauss

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    m = int(n)
    if n < 1 or n != m:
        raise ValueError('n must be a positive integer.')
    x = _ufuncs._sinpi(np.arange(-m + 1, m, 2) / (2*m))
    w = np.full_like(x, pi/m)
    if mu:
        return x, w, pi
    else:
        return x, w


def chebyt(n, monic=False):
    r"""Chebyshev polynomial of the first kind.

    Defined to be the solution of

    .. math::
        (1 - x^2)\frac{d^2}{dx^2}T_n - x\frac{d}{dx}T_n + n^2T_n = 0;

    :math:`T_n` is a polynomial of degree :math:`n`.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    T : orthopoly1d
        Chebyshev polynomial of the first kind.

    See Also
    --------
    chebyu : Chebyshev polynomial of the second kind.

    Notes
    -----
    The polynomials :math:`T_n` are orthogonal over :math:`[-1, 1]`
    with weight function :math:`(1 - x^2)^{-1/2}`.

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    Chebyshev polynomials of the first kind of order :math:`n` can
    be obtained as the determinant of specific :math:`n \times n`
    matrices. As an example we can check how the points obtained from
    the determinant of the following :math:`3 \times 3` matrix
    lay exactly on :math:`T_3`:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.linalg import det
    >>> from scipy.special import chebyt
    >>> x = np.arange(-1.0, 1.0, 0.01)
    >>> fig, ax = plt.subplots()
    >>> ax.set_ylim(-2.0, 2.0)
    >>> ax.set_title(r'Chebyshev polynomial $T_3$')
    >>> ax.plot(x, chebyt(3)(x), label=rf'$T_3$')
    >>> for p in np.arange(-1.0, 1.0, 0.1):
    ...     ax.plot(p,
    ...             det(np.array([[p, 1, 0], [1, 2*p, 1], [0, 1, 2*p]])),
    ...             'rx')
    >>> plt.legend(loc='best')
    >>> plt.show()

    They are also related to the Jacobi Polynomials
    :math:`P_n^{(-0.5, -0.5)}` through the relation:

    .. math::
        P_n^{(-0.5, -0.5)}(x) = \frac{1}{4^n} \binom{2n}{n} T_n(x)

    Let's verify it for :math:`n = 3`:

    >>> from scipy.special import binom
    >>> from scipy.special import jacobi
    >>> x = np.arange(-1.0, 1.0, 0.01)
    >>> np.allclose(jacobi(3, -0.5, -0.5)(x),
    ...             1/64 * binom(6, 3) * chebyt(3)(x))
    True

    We can plot the Chebyshev polynomials :math:`T_n` for some values
    of :math:`n`:

    >>> x = np.arange(-1.5, 1.5, 0.01)
    >>> fig, ax = plt.subplots()
    >>> ax.set_ylim(-4.0, 4.0)
    >>> ax.set_title(r'Chebyshev polynomials $T_n$')
    >>> for n in np.arange(2,5):
    ...     ax.plot(x, chebyt(n)(x), label=rf'$T_n={n}$')
    >>> plt.legend(loc='best')
    >>> plt.show()

    """
    if n < 0:
        raise ValueError("n must be nonnegative.")

    def wfunc(x):
        return 1.0 / sqrt(1 - x * x)
    if n == 0:
        return orthopoly1d([], [], pi, 1.0, wfunc, (-1, 1), monic,
                           lambda x: _ufuncs.eval_chebyt(n, x))
    n1 = n
    x, w, mu = roots_chebyt(n1, mu=True)
    hn = pi / 2
    kn = 2**(n - 1)
    p = orthopoly1d(x, w, hn, kn, wfunc, (-1, 1), monic,
                    lambda x: _ufuncs.eval_chebyt(n, x))
    return p

# Chebyshev of the second kind
#    U_n(x) = (n+1)! sqrt(pi) / (2*_gam(n+3./2)) * P^(1/2,1/2)_n(x)


def roots_chebyu(n, mu=False):
    r"""Gauss-Chebyshev (second kind) quadrature.

    Computes the sample points and weights for Gauss-Chebyshev
    quadrature. The sample points are the roots of the nth degree
    Chebyshev polynomial of the second kind, :math:`U_n(x)`. These
    sample points and weights correctly integrate polynomials of
    degree :math:`2n - 1` or less over the interval :math:`[-1, 1]`
    with weight function :math:`w(x) = \sqrt{1 - x^2}`. See 22.2.5 in
    [AS]_ for details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.fixed_quad

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    m = int(n)
    if n < 1 or n != m:
        raise ValueError('n must be a positive integer.')
    t = np.arange(m, 0, -1) * pi / (m + 1)
    x = np.cos(t)
    w = pi * np.sin(t)**2 / (m + 1)
    if mu:
        return x, w, pi / 2
    else:
        return x, w


def chebyu(n, monic=False):
    r"""Chebyshev polynomial of the second kind.

    Defined to be the solution of

    .. math::
        (1 - x^2)\frac{d^2}{dx^2}U_n - 3x\frac{d}{dx}U_n
          + n(n + 2)U_n = 0;

    :math:`U_n` is a polynomial of degree :math:`n`.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    U : orthopoly1d
        Chebyshev polynomial of the second kind.

    See Also
    --------
    chebyt : Chebyshev polynomial of the first kind.

    Notes
    -----
    The polynomials :math:`U_n` are orthogonal over :math:`[-1, 1]`
    with weight function :math:`(1 - x^2)^{1/2}`.

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    Chebyshev polynomials of the second kind of order :math:`n` can
    be obtained as the determinant of specific :math:`n \times n`
    matrices. As an example we can check how the points obtained from
    the determinant of the following :math:`3 \times 3` matrix
    lay exactly on :math:`U_3`:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.linalg import det
    >>> from scipy.special import chebyu
    >>> x = np.arange(-1.0, 1.0, 0.01)
    >>> fig, ax = plt.subplots()
    >>> ax.set_ylim(-2.0, 2.0)
    >>> ax.set_title(r'Chebyshev polynomial $U_3$')
    >>> ax.plot(x, chebyu(3)(x), label=rf'$U_3$')
    >>> for p in np.arange(-1.0, 1.0, 0.1):
    ...     ax.plot(p,
    ...             det(np.array([[2*p, 1, 0], [1, 2*p, 1], [0, 1, 2*p]])),
    ...             'rx')
    >>> plt.legend(loc='best')
    >>> plt.show()

    They satisfy the recurrence relation:

    .. math::
        U_{2n-1}(x) = 2 T_n(x)U_{n-1}(x)

    where the :math:`T_n` are the Chebyshev polynomial of the first kind.
    Let's verify it for :math:`n = 2`:

    >>> from scipy.special import chebyt
    >>> x = np.arange(-1.0, 1.0, 0.01)
    >>> np.allclose(chebyu(3)(x), 2 * chebyt(2)(x) * chebyu(1)(x))
    True

    We can plot the Chebyshev polynomials :math:`U_n` for some values
    of :math:`n`:

    >>> x = np.arange(-1.0, 1.0, 0.01)
    >>> fig, ax = plt.subplots()
    >>> ax.set_ylim(-1.5, 1.5)
    >>> ax.set_title(r'Chebyshev polynomials $U_n$')
    >>> for n in np.arange(1,5):
    ...     ax.plot(x, chebyu(n)(x), label=rf'$U_n={n}$')
    >>> plt.legend(loc='best')
    >>> plt.show()

    """
    base = jacobi(n, 0.5, 0.5, monic=monic)
    if monic:
        return base
    factor = sqrt(pi) / 2.0 * _gam(n + 2) / _gam(n + 1.5)
    base._scale(factor)
    return base

# Chebyshev of the first kind        C_n(x)


def roots_chebyc(n, mu=False):
    r"""Gauss-Chebyshev (first kind) quadrature.

    Compute the sample points and weights for Gauss-Chebyshev
    quadrature. The sample points are the roots of the nth degree
    Chebyshev polynomial of the first kind, :math:`C_n(x)`. These
    sample points and weights correctly integrate polynomials of
    degree :math:`2n - 1` or less over the interval :math:`[-2, 2]`
    with weight function :math:`w(x) = 1 / \sqrt{1 - (x/2)^2}`. See
    22.2.6 in [AS]_ for more details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.fixed_quad

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    x, w, m = roots_chebyt(n, True)
    x *= 2
    w *= 2
    m *= 2
    if mu:
        return x, w, m
    else:
        return x, w


def chebyc(n, monic=False):
    r"""Chebyshev polynomial of the first kind on :math:`[-2, 2]`.

    Defined as :math:`C_n(x) = 2T_n(x/2)`, where :math:`T_n` is the
    nth Chebychev polynomial of the first kind.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    C : orthopoly1d
        Chebyshev polynomial of the first kind on :math:`[-2, 2]`.

    See Also
    --------
    chebyt : Chebyshev polynomial of the first kind.

    Notes
    -----
    The polynomials :math:`C_n(x)` are orthogonal over :math:`[-2, 2]`
    with weight function :math:`1/\sqrt{1 - (x/2)^2}`.

    References
    ----------
    .. [1] Abramowitz and Stegun, "Handbook of Mathematical Functions"
           Section 22. National Bureau of Standards, 1972.

    """
    if n < 0:
        raise ValueError("n must be nonnegative.")

    if n == 0:
        n1 = n + 1
    else:
        n1 = n
    x, w = roots_chebyc(n1)
    if n == 0:
        x, w = [], []
    hn = 4 * pi * ((n == 0) + 1)
    kn = 1.0
    p = orthopoly1d(x, w, hn, kn,
                    wfunc=lambda x: 1.0 / sqrt(1 - x * x / 4.0),
                    limits=(-2, 2), monic=monic)
    if not monic:
        p._scale(2.0 / p(2))
        p.__dict__['_eval_func'] = lambda x: _ufuncs.eval_chebyc(n, x)
    return p

# Chebyshev of the second kind       S_n(x)


def roots_chebys(n, mu=False):
    r"""Gauss-Chebyshev (second kind) quadrature.

    Compute the sample points and weights for Gauss-Chebyshev
    quadrature. The sample points are the roots of the nth degree
    Chebyshev polynomial of the second kind, :math:`S_n(x)`. These
    sample points and weights correctly integrate polynomials of
    degree :math:`2n - 1` or less over the interval :math:`[-2, 2]`
    with weight function :math:`w(x) = \sqrt{1 - (x/2)^2}`. See 22.2.7
    in [AS]_ for more details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.fixed_quad

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    x, w, m = roots_chebyu(n, True)
    x *= 2
    w *= 2
    m *= 2
    if mu:
        return x, w, m
    else:
        return x, w


def chebys(n, monic=False):
    r"""Chebyshev polynomial of the second kind on :math:`[-2, 2]`.

    Defined as :math:`S_n(x) = U_n(x/2)` where :math:`U_n` is the
    nth Chebychev polynomial of the second kind.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    S : orthopoly1d
        Chebyshev polynomial of the second kind on :math:`[-2, 2]`.

    See Also
    --------
    chebyu : Chebyshev polynomial of the second kind

    Notes
    -----
    The polynomials :math:`S_n(x)` are orthogonal over :math:`[-2, 2]`
    with weight function :math:`\sqrt{1 - (x/2)}^2`.

    References
    ----------
    .. [1] Abramowitz and Stegun, "Handbook of Mathematical Functions"
           Section 22. National Bureau of Standards, 1972.

    """
    if n < 0:
        raise ValueError("n must be nonnegative.")

    if n == 0:
        n1 = n + 1
    else:
        n1 = n
    x, w = roots_chebys(n1)
    if n == 0:
        x, w = [], []
    hn = pi
    kn = 1.0
    p = orthopoly1d(x, w, hn, kn,
                    wfunc=lambda x: sqrt(1 - x * x / 4.0),
                    limits=(-2, 2), monic=monic)
    if not monic:
        factor = (n + 1.0) / p(2)
        p._scale(factor)
        p.__dict__['_eval_func'] = lambda x: _ufuncs.eval_chebys(n, x)
    return p

# Shifted Chebyshev of the first kind     T^*_n(x)


def roots_sh_chebyt(n, mu=False):
    r"""Gauss-Chebyshev (first kind, shifted) quadrature.

    Compute the sample points and weights for Gauss-Chebyshev
    quadrature. The sample points are the roots of the nth degree
    shifted Chebyshev polynomial of the first kind, :math:`T_n(x)`.
    These sample points and weights correctly integrate polynomials of
    degree :math:`2n - 1` or less over the interval :math:`[0, 1]`
    with weight function :math:`w(x) = 1/\sqrt{x - x^2}`. See 22.2.8
    in [AS]_ for more details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.fixed_quad

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    xw = roots_chebyt(n, mu)
    return ((xw[0] + 1) / 2,) + xw[1:]


def sh_chebyt(n, monic=False):
    r"""Shifted Chebyshev polynomial of the first kind.

    Defined as :math:`T^*_n(x) = T_n(2x - 1)` for :math:`T_n` the nth
    Chebyshev polynomial of the first kind.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    T : orthopoly1d
        Shifted Chebyshev polynomial of the first kind.

    Notes
    -----
    The polynomials :math:`T^*_n` are orthogonal over :math:`[0, 1]`
    with weight function :math:`(x - x^2)^{-1/2}`.

    """
    base = sh_jacobi(n, 0.0, 0.5, monic=monic)
    if monic:
        return base
    if n > 0:
        factor = 4**n / 2.0
    else:
        factor = 1.0
    base._scale(factor)
    return base


# Shifted Chebyshev of the second kind    U^*_n(x)
def roots_sh_chebyu(n, mu=False):
    r"""Gauss-Chebyshev (second kind, shifted) quadrature.

    Computes the sample points and weights for Gauss-Chebyshev
    quadrature. The sample points are the roots of the nth degree
    shifted Chebyshev polynomial of the second kind, :math:`U_n(x)`.
    These sample points and weights correctly integrate polynomials of
    degree :math:`2n - 1` or less over the interval :math:`[0, 1]`
    with weight function :math:`w(x) = \sqrt{x - x^2}`. See 22.2.9 in
    [AS]_ for more details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.fixed_quad

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    x, w, m = roots_chebyu(n, True)
    x = (x + 1) / 2
    m_us = _ufuncs.beta(1.5, 1.5)
    w *= m_us / m
    if mu:
        return x, w, m_us
    else:
        return x, w


def sh_chebyu(n, monic=False):
    r"""Shifted Chebyshev polynomial of the second kind.

    Defined as :math:`U^*_n(x) = U_n(2x - 1)` for :math:`U_n` the nth
    Chebyshev polynomial of the second kind.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    U : orthopoly1d
        Shifted Chebyshev polynomial of the second kind.

    Notes
    -----
    The polynomials :math:`U^*_n` are orthogonal over :math:`[0, 1]`
    with weight function :math:`(x - x^2)^{1/2}`.

    """
    base = sh_jacobi(n, 2.0, 1.5, monic=monic)
    if monic:
        return base
    factor = 4**n
    base._scale(factor)
    return base

# Legendre


def roots_legendre(n, mu=False):
    r"""Gauss-Legendre quadrature.

    Compute the sample points and weights for Gauss-Legendre
    quadrature [GL]_. The sample points are the roots of the nth degree
    Legendre polynomial :math:`P_n(x)`. These sample points and
    weights correctly integrate polynomials of degree :math:`2n - 1`
    or less over the interval :math:`[-1, 1]` with weight function
    :math:`w(x) = 1`. See 2.2.10 in [AS]_ for more details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.fixed_quad
    numpy.polynomial.legendre.leggauss

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.
    .. [GL] Gauss-Legendre quadrature, Wikipedia,
        https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import roots_legendre, eval_legendre
    >>> roots, weights = roots_legendre(9)

    ``roots`` holds the roots, and ``weights`` holds the weights for
    Gauss-Legendre quadrature.

    >>> roots
    array([-0.96816024, -0.83603111, -0.61337143, -0.32425342,  0.        ,
            0.32425342,  0.61337143,  0.83603111,  0.96816024])
    >>> weights
    array([0.08127439, 0.18064816, 0.2606107 , 0.31234708, 0.33023936,
           0.31234708, 0.2606107 , 0.18064816, 0.08127439])

    Verify that we have the roots by evaluating the degree 9 Legendre
    polynomial at ``roots``.  All the values are approximately zero:

    >>> eval_legendre(9, roots)
    array([-8.88178420e-16, -2.22044605e-16,  1.11022302e-16,  1.11022302e-16,
            0.00000000e+00, -5.55111512e-17, -1.94289029e-16,  1.38777878e-16,
           -8.32667268e-17])

    Here we'll show how the above values can be used to estimate the
    integral from 1 to 2 of f(t) = t + 1/t with Gauss-Legendre
    quadrature [GL]_.  First define the function and the integration
    limits.

    >>> def f(t):
    ...    return t + 1/t
    ...
    >>> a = 1
    >>> b = 2

    We'll use ``integral(f(t), t=a, t=b)`` to denote the definite integral
    of f from t=a to t=b.  The sample points in ``roots`` are from the
    interval [-1, 1], so we'll rewrite the integral with the simple change
    of variable::

        x = 2/(b - a) * t - (a + b)/(b - a)

    with inverse::

        t = (b - a)/2 * x + (a + b)/2

    Then::

        integral(f(t), a, b) =
            (b - a)/2 * integral(f((b-a)/2*x + (a+b)/2), x=-1, x=1)

    We can approximate the latter integral with the values returned
    by `roots_legendre`.

    Map the roots computed above from [-1, 1] to [a, b].

    >>> t = (b - a)/2 * roots + (a + b)/2

    Approximate the integral as the weighted sum of the function values.

    >>> (b - a)/2 * f(t).dot(weights)
    2.1931471805599276

    Compare that to the exact result, which is 3/2 + log(2):

    >>> 1.5 + np.log(2)
    2.1931471805599454

    """
    m = int(n)
    if n < 1 or n != m:
        raise ValueError("n must be a positive integer.")

    mu0 = 2.0
    def an_func(k):
        return 0.0 * k
    def bn_func(k):
        return k * np.sqrt(1.0 / (4 * k * k - 1))
    f = _ufuncs.eval_legendre
    def df(n, x):
        return (-n * x * _ufuncs.eval_legendre(n, x)
                + n * _ufuncs.eval_legendre(n - 1, x)) / (1 - x ** 2)
    return _gen_roots_and_weights(m, mu0, an_func, bn_func, f, df, True, mu)


def legendre(n, monic=False):
    r"""Legendre polynomial.

    Defined to be the solution of

    .. math::
        \frac{d}{dx}\left[(1 - x^2)\frac{d}{dx}P_n(x)\right]
          + n(n + 1)P_n(x) = 0;

    :math:`P_n(x)` is a polynomial of degree :math:`n`.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    P : orthopoly1d
        Legendre polynomial.

    Notes
    -----
    The polynomials :math:`P_n` are orthogonal over :math:`[-1, 1]`
    with weight function 1.

    Examples
    --------
    Generate the 3rd-order Legendre polynomial 1/2*(5x^3 + 0x^2 - 3x + 0):

    >>> from scipy.special import legendre
    >>> legendre(3)
    poly1d([ 2.5,  0. , -1.5,  0. ])

    """
    if n < 0:
        raise ValueError("n must be nonnegative.")

    if n == 0:
        n1 = n + 1
    else:
        n1 = n
    x, w = roots_legendre(n1)
    if n == 0:
        x, w = [], []
    hn = 2.0 / (2 * n + 1)
    kn = _gam(2 * n + 1) / _gam(n + 1)**2 / 2.0**n
    p = orthopoly1d(x, w, hn, kn, wfunc=lambda x: 1.0, limits=(-1, 1),
                    monic=monic,
                    eval_func=lambda x: _ufuncs.eval_legendre(n, x))
    return p

# Shifted Legendre              P^*_n(x)


def roots_sh_legendre(n, mu=False):
    r"""Gauss-Legendre (shifted) quadrature.

    Compute the sample points and weights for Gauss-Legendre
    quadrature. The sample points are the roots of the nth degree
    shifted Legendre polynomial :math:`P^*_n(x)`. These sample points
    and weights correctly integrate polynomials of degree :math:`2n -
    1` or less over the interval :math:`[0, 1]` with weight function
    :math:`w(x) = 1.0`. See 2.2.11 in [AS]_ for details.

    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.

    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights

    See Also
    --------
    scipy.integrate.fixed_quad

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    """
    x, w = roots_legendre(n)
    x = (x + 1) / 2
    w /= 2
    if mu:
        return x, w, 1.0
    else:
        return x, w


def sh_legendre(n, monic=False):
    r"""Shifted Legendre polynomial.

    Defined as :math:`P^*_n(x) = P_n(2x - 1)` for :math:`P_n` the nth
    Legendre polynomial.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    P : orthopoly1d
        Shifted Legendre polynomial.

    Notes
    -----
    The polynomials :math:`P^*_n` are orthogonal over :math:`[0, 1]`
    with weight function 1.

    """
    if n < 0:
        raise ValueError("n must be nonnegative.")

    def wfunc(x):
        return 0.0 * x + 1.0
    if n == 0:
        return orthopoly1d([], [], 1.0, 1.0, wfunc, (0, 1), monic,
                           lambda x: _ufuncs.eval_sh_legendre(n, x))
    x, w = roots_sh_legendre(n)
    hn = 1.0 / (2 * n + 1.0)
    kn = _gam(2 * n + 1) / _gam(n + 1)**2
    p = orthopoly1d(x, w, hn, kn, wfunc, limits=(0, 1), monic=monic,
                    eval_func=lambda x: _ufuncs.eval_sh_legendre(n, x))
    return p


# Make the old root function names an alias for the new ones
_modattrs = globals()
for newfun, oldfun in _rootfuns_map.items():
    _modattrs[oldfun] = _modattrs[newfun]
    __all__.append(oldfun)
<|endoftext|>"""Warnings and Exceptions that can be raised by special functions."""
import warnings


class SpecialFunctionWarning(Warning):
    """Warning that can be emitted by special functions."""
    pass


warnings.simplefilter("always", category=SpecialFunctionWarning)


class SpecialFunctionError(Exception):
    """Exception that can be raised by special functions."""
    pass
<|endoftext|>import os
import sys
import functools

import numpy as np
from scipy._lib._array_api import (
    array_namespace, scipy_namespace_for, is_numpy
)
from . import _ufuncs
# These don't really need to be imported, but otherwise IDEs might not realize
# that these are defined in this file / report an error in __init__.py
from ._ufuncs import (
    log_ndtr, ndtr, ndtri, erf, erfc, i0, i0e, i1, i1e, gammaln,  # noqa: F401
    gammainc, gammaincc, logit, expit, entr, rel_entr, xlogy,  # noqa: F401
    chdtr, chdtrc, betainc, betaincc, stdtr  # noqa: F401
)

_SCIPY_ARRAY_API = os.environ.get("SCIPY_ARRAY_API", False)
array_api_compat_prefix = "scipy._lib.array_api_compat"


def get_array_special_func(f_name, xp, n_array_args):
    spx = scipy_namespace_for(xp)
    f = None
    if is_numpy(xp):
        f = getattr(_ufuncs, f_name, None)
    elif spx is not None:
        f = getattr(spx.special, f_name, None)

    if f is not None:
        return f

    # if generic array-API implementation is available, use that;
    # otherwise, fall back to NumPy/SciPy
    if f_name in _generic_implementations:
        _f = _generic_implementations[f_name](xp=xp, spx=spx)
        if _f is not None:
            return _f

    _f = getattr(_ufuncs, f_name, None)
    def __f(*args, _f=_f, _xp=xp, **kwargs):
        array_args = args[:n_array_args]
        other_args = args[n_array_args:]
        array_args = [np.asarray(arg) for arg in array_args]
        out = _f(*array_args, *other_args, **kwargs)
        return _xp.asarray(out)

    return __f


def _get_shape_dtype(*args, xp):
    args = xp.broadcast_arrays(*args)
    shape = args[0].shape
    dtype = xp.result_type(*args)
    if xp.isdtype(dtype, 'integral'):
        dtype = xp.float64
        args = [xp.asarray(arg, dtype=dtype) for arg in args]
    return args, shape, dtype


def _rel_entr(xp, spx):
    def __rel_entr(x, y, *, xp=xp):
        args, shape, dtype = _get_shape_dtype(x, y, xp=xp)
        x, y = args
        res = xp.full(x.shape, xp.inf, dtype=dtype)
        res[(x == 0) & (y >= 0)] = xp.asarray(0, dtype=dtype)
        i = (x > 0) & (y > 0)
        res[i] = x[i] * (xp.log(x[i]) - xp.log(y[i]))
        return res
    return __rel_entr


def _xlogy(xp, spx):
    def __xlogy(x, y, *, xp=xp):
        with np.errstate(divide='ignore', invalid='ignore'):
            temp = x * xp.log(y)
        return xp.where(x == 0., xp.asarray(0., dtype=temp.dtype), temp)
    return __xlogy


def _chdtr(xp, spx):
    # The difference between this and just using `gammainc`
    # defined by `get_array_special_func` is that if `gammainc`
    # isn't found, we don't want to use the SciPy version; we'll
    # return None here and use the SciPy version of `chdtr`.
    gammainc = getattr(spx.special, 'gammainc', None) if spx else None  # noqa: F811
    if gammainc is None and hasattr(xp, 'special'):
        gammainc = getattr(xp.special, 'gammainc', None)
    if gammainc is None:
        return None

    def __chdtr(v, x):
        res = gammainc(v / 2, x / 2)  # this is almost all we need
        # The rest can be removed when google/jax#20507 is resolved
        mask = (v == 0) & (x > 0)  # JAX returns NaN
        res = xp.where(mask, 1., res)
        mask = xp.isinf(v) & xp.isinf(x)  # JAX returns 1.0
        return xp.where(mask, xp.nan, res)
    return __chdtr


def _chdtrc(xp, spx):
    # The difference between this and just using `gammaincc`
    # defined by `get_array_special_func` is that if `gammaincc`
    # isn't found, we don't want to use the SciPy version; we'll
    # return None here and use the SciPy version of `chdtrc`.
    gammaincc = getattr(spx.special, 'gammaincc', None) if spx else None  # noqa: F811
    if gammaincc is None and hasattr(xp, 'special'):
        gammaincc = getattr(xp.special, 'gammaincc', None)
    if gammaincc is None:
        return None

    def __chdtrc(v, x):
        res = xp.where(x >= 0, gammaincc(v/2, x/2), 1)
        i_nan = ((x == 0) & (v == 0)) | xp.isnan(x) | xp.isnan(v) | (v <= 0)
        res = xp.where(i_nan, xp.nan, res)
        return res
    return __chdtrc


def _betaincc(xp, spx):
    betainc = getattr(spx.special, 'betainc', None) if spx else None  # noqa: F811
    if betainc is None and hasattr(xp, 'special'):
        betainc = getattr(xp.special, 'betainc', None)
    if betainc is None:
        return None

    def __betaincc(a, b, x):
        # not perfect; might want to just rely on SciPy
        return betainc(b, a, 1-x)
    return __betaincc


def _stdtr(xp, spx):
    betainc = getattr(spx.special, 'betainc', None) if spx else None  # noqa: F811
    if betainc is None and hasattr(xp, 'special'):
        betainc = getattr(xp.special, 'betainc', None)
    if betainc is None:
        return None

    def __stdtr(df, t):
        x = df / (t ** 2 + df)
        tail = betainc(df / 2, 0.5, x) / 2
        return xp.where(t < 0, tail, 1 - tail)

    return __stdtr


_generic_implementations = {'rel_entr': _rel_entr,
                            'xlogy': _xlogy,
                            'chdtr': _chdtr,
                            'chdtrc': _chdtrc,
                            'betaincc': _betaincc,
                            'stdtr': _stdtr,
                            }


# functools.wraps doesn't work because:
# 'numpy.ufunc' object has no attribute '__module__'
def support_alternative_backends(f_name, n_array_args):
    func = getattr(_ufuncs, f_name)

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        xp = array_namespace(*args[:n_array_args])
        f = get_array_special_func(f_name, xp, n_array_args)
        return f(*args, **kwargs)

    return wrapped


array_special_func_map = {
    'log_ndtr': 1,
    'ndtr': 1,
    'ndtri': 1,
    'erf': 1,
    'erfc': 1,
    'i0': 1,
    'i0e': 1,
    'i1': 1,
    'i1e': 1,
    'gammaln': 1,
    'gammainc': 2,
    'gammaincc': 2,
    'logit': 1,
    'expit': 1,
    'entr': 1,
    'rel_entr': 2,
    'xlogy': 2,
    'chdtr': 2,
    'chdtrc': 2,
    'betainc': 3,
    'betaincc': 3,
    'stdtr': 2,
}

for f_name, n_array_args in array_special_func_map.items():
    f = (support_alternative_backends(f_name, n_array_args) if _SCIPY_ARRAY_API
         else getattr(_ufuncs, f_name))
    sys.modules[__name__].__dict__[f_name] = f

__all__ = list(array_special_func_map)
<|endoftext|># Last Change: Sat Mar 21 02:00 PM 2009 J

# Copyright (c) 2001, 2002 Enthought, Inc.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   a. Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#   b. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#   c. Neither the name of the Enthought nor the names of its contributors
#      may be used to endorse or promote products derived from this software
#      without specific prior written permission.
#
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

"""Some more special functions which may be useful for multivariate statistical
analysis."""

import numpy as np
from scipy.special import gammaln as loggam


__all__ = ['multigammaln']


def multigammaln(a, d):
    r"""Returns the log of multivariate gamma, also sometimes called the
    generalized gamma.

    Parameters
    ----------
    a : ndarray
        The multivariate gamma is computed for each item of `a`.
    d : int
        The dimension of the space of integration.

    Returns
    -------
    res : ndarray
        The values of the log multivariate gamma at the given points `a`.

    Notes
    -----
    The formal definition of the multivariate gamma of dimension d for a real
    `a` is

    .. math::

        \Gamma_d(a) = \int_{A>0} e^{-tr(A)} |A|^{a - (d+1)/2} dA

    with the condition :math:`a > (d-1)/2`, and :math:`A > 0` being the set of
    all the positive definite matrices of dimension `d`.  Note that `a` is a
    scalar: the integrand only is multivariate, the argument is not (the
    function is defined over a subset of the real set).

    This can be proven to be equal to the much friendlier equation

    .. math::

        \Gamma_d(a) = \pi^{d(d-1)/4} \prod_{i=1}^{d} \Gamma(a - (i-1)/2).

    References
    ----------
    R. J. Muirhead, Aspects of multivariate statistical theory (Wiley Series in
    probability and mathematical statistics).

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import multigammaln, gammaln
    >>> a = 23.5
    >>> d = 10
    >>> multigammaln(a, d)
    454.1488605074416

    Verify that the result agrees with the logarithm of the equation
    shown above:

    >>> d*(d-1)/4*np.log(np.pi) + gammaln(a - 0.5*np.arange(0, d)).sum()
    454.1488605074416
    """
    a = np.asarray(a)
    if not np.isscalar(d) or (np.floor(d) != d):
        raise ValueError("d should be a positive integer (dimension)")
    if np.any(a <= 0.5 * (d - 1)):
        raise ValueError(f"condition a ({a:f}) > 0.5 * (d-1) ({0.5 * (d-1):f}) not met")

    res = (d * (d-1) * 0.25) * np.log(np.pi)
    res += np.sum(loggam([(a - (j - 1.)/2) for j in range(1, d+1)]), axis=0)
    return res
<|endoftext|>#
# Author:  Travis Oliphant, 2002
#

import numpy as np
import math
import warnings
from collections import defaultdict
from heapq import heapify, heappop
from numpy import (pi, asarray, floor, isscalar, sqrt, where,
                   sin, place, issubdtype, extract, inexact, nan, zeros, sinc)

from . import _ufuncs
from ._ufuncs import (mathieu_a, mathieu_b, iv, jv, gamma, rgamma,
                      psi, hankel1, hankel2, yv, kv, poch, binom,
                      _stirling2_inexact)

from ._gufuncs import _lqn, _lqmn, _rctj, _rcty
from ._input_validation import _nonneg_int_or_fail
from . import _specfun
from ._comb import _comb_int
from ._multiufuncs import (assoc_legendre_p_all,
                           legendre_p_all)
from scipy._lib.deprecation import _deprecated


__all__ = [
    'ai_zeros',
    'assoc_laguerre',
    'bei_zeros',
    'beip_zeros',
    'ber_zeros',
    'bernoulli',
    'berp_zeros',
    'bi_zeros',
    'clpmn',
    'comb',
    'digamma',
    'diric',
    'erf_zeros',
    'euler',
    'factorial',
    'factorial2',
    'factorialk',
    'fresnel_zeros',
    'fresnelc_zeros',
    'fresnels_zeros',
    'h1vp',
    'h2vp',
    'ivp',
    'jn_zeros',
    'jnjnp_zeros',
    'jnp_zeros',
    'jnyn_zeros',
    'jvp',
    'kei_zeros',
    'keip_zeros',
    'kelvin_zeros',
    'ker_zeros',
    'kerp_zeros',
    'kvp',
    'lmbda',
    'lpmn',
    'lpn',
    'lqmn',
    'lqn',
    'mathieu_even_coef',
    'mathieu_odd_coef',
    'obl_cv_seq',
    'pbdn_seq',
    'pbdv_seq',
    'pbvv_seq',
    'perm',
    'polygamma',
    'pro_cv_seq',
    'riccati_jn',
    'riccati_yn',
    'sinc',
    'softplus',
    'stirling2',
    'y0_zeros',
    'y1_zeros',
    'y1p_zeros',
    'yn_zeros',
    'ynp_zeros',
    'yvp',
    'zeta'
]


__DEPRECATION_MSG_1_15 = (
    "`scipy.special.{}` is deprecated as of SciPy 1.15.0 and will be "
    "removed in SciPy 1.17.0. Please use `scipy.special.{}` instead."
)

# mapping k to last n such that factorialk(n, k) < np.iinfo(np.int64).max
_FACTORIALK_LIMITS_64BITS = {1: 20, 2: 33, 3: 44, 4: 54, 5: 65,
                             6: 74, 7: 84, 8: 93, 9: 101}
# mapping k to last n such that factorialk(n, k) < np.iinfo(np.int32).max
_FACTORIALK_LIMITS_32BITS = {1: 12, 2: 19, 3: 25, 4: 31, 5: 37,
                             6: 43, 7: 47, 8: 51, 9: 56}


def diric(x, n):
    """Periodic sinc function, also called the Dirichlet function.

    The Dirichlet function is defined as::

        diric(x, n) = sin(x * n/2) / (n * sin(x / 2)),

    where `n` is a positive integer.

    Parameters
    ----------
    x : array_like
        Input data
    n : int
        Integer defining the periodicity.

    Returns
    -------
    diric : ndarray

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import special
    >>> import matplotlib.pyplot as plt

    >>> x = np.linspace(-8*np.pi, 8*np.pi, num=201)
    >>> plt.figure(figsize=(8, 8));
    >>> for idx, n in enumerate([2, 3, 4, 9]):
    ...     plt.subplot(2, 2, idx+1)
    ...     plt.plot(x, special.diric(x, n))
    ...     plt.title('diric, n={}'.format(n))
    >>> plt.show()

    The following example demonstrates that `diric` gives the magnitudes
    (modulo the sign and scaling) of the Fourier coefficients of a
    rectangular pulse.

    Suppress output of values that are effectively 0:

    >>> np.set_printoptions(suppress=True)

    Create a signal `x` of length `m` with `k` ones:

    >>> m = 8
    >>> k = 3
    >>> x = np.zeros(m)
    >>> x[:k] = 1

    Use the FFT to compute the Fourier transform of `x`, and
    inspect the magnitudes of the coefficients:

    >>> np.abs(np.fft.fft(x))
    array([ 3.        ,  2.41421356,  1.        ,  0.41421356,  1.        ,
            0.41421356,  1.        ,  2.41421356])

    Now find the same values (up to sign) using `diric`. We multiply
    by `k` to account for the different scaling conventions of
    `numpy.fft.fft` and `diric`:

    >>> theta = np.linspace(0, 2*np.pi, m, endpoint=False)
    >>> k * special.diric(theta, k)
    array([ 3.        ,  2.41421356,  1.        , -0.41421356, -1.        ,
           -0.41421356,  1.        ,  2.41421356])
    """
    x, n = asarray(x), asarray(n)
    n = asarray(n + (x-x))
    x = asarray(x + (n-n))
    if issubdtype(x.dtype, inexact):
        ytype = x.dtype
    else:
        ytype = float
    y = zeros(x.shape, ytype)

    # empirical minval for 32, 64 or 128 bit float computations
    # where sin(x/2) < minval, result is fixed at +1 or -1
    if np.finfo(ytype).eps < 1e-18:
        minval = 1e-11
    elif np.finfo(ytype).eps < 1e-15:
        minval = 1e-7
    else:
        minval = 1e-3

    mask1 = (n <= 0) | (n != floor(n))
    place(y, mask1, nan)

    x = x / 2
    denom = sin(x)
    mask2 = (1-mask1) & (abs(denom) < minval)
    xsub = extract(mask2, x)
    nsub = extract(mask2, n)
    zsub = xsub / pi
    place(y, mask2, pow(-1, np.round(zsub)*(nsub-1)))

    mask = (1-mask1) & (1-mask2)
    xsub = extract(mask, x)
    nsub = extract(mask, n)
    dsub = extract(mask, denom)
    place(y, mask, sin(nsub*xsub)/(nsub*dsub))
    return y


def jnjnp_zeros(nt):
    """Compute zeros of integer-order Bessel functions Jn and Jn'.

    Results are arranged in order of the magnitudes of the zeros.

    Parameters
    ----------
    nt : int
        Number (<=1200) of zeros to compute

    Returns
    -------
    zo[l-1] : ndarray
        Value of the lth zero of Jn(x) and Jn'(x). Of length `nt`.
    n[l-1] : ndarray
        Order of the Jn(x) or Jn'(x) associated with lth zero. Of length `nt`.
    m[l-1] : ndarray
        Serial number of the zeros of Jn(x) or Jn'(x) associated
        with lth zero. Of length `nt`.
    t[l-1] : ndarray
        0 if lth zero in zo is zero of Jn(x), 1 if it is a zero of Jn'(x). Of
        length `nt`.

    See Also
    --------
    jn_zeros, jnp_zeros : to get separated arrays of zeros.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt > 1200):
        raise ValueError("Number must be integer <= 1200.")
    nt = int(nt)
    n, m, t, zo = _specfun.jdzo(nt)
    return zo[1:nt+1], n[:nt], m[:nt], t[:nt]


def jnyn_zeros(n, nt):
    """Compute nt zeros of Bessel functions Jn(x), Jn'(x), Yn(x), and Yn'(x).

    Returns 4 arrays of length `nt`, corresponding to the first `nt`
    zeros of Jn(x), Jn'(x), Yn(x), and Yn'(x), respectively. The zeros
    are returned in ascending order.

    Parameters
    ----------
    n : int
        Order of the Bessel functions
    nt : int
        Number (<=1200) of zeros to compute

    Returns
    -------
    Jn : ndarray
        First `nt` zeros of Jn
    Jnp : ndarray
        First `nt` zeros of Jn'
    Yn : ndarray
        First `nt` zeros of Yn
    Ynp : ndarray
        First `nt` zeros of Yn'

    See Also
    --------
    jn_zeros, jnp_zeros, yn_zeros, ynp_zeros

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    Examples
    --------
    Compute the first three roots of :math:`J_1`, :math:`J_1'`,
    :math:`Y_1` and :math:`Y_1'`.

    >>> from scipy.special import jnyn_zeros
    >>> jn_roots, jnp_roots, yn_roots, ynp_roots = jnyn_zeros(1, 3)
    >>> jn_roots, yn_roots
    (array([ 3.83170597,  7.01558667, 10.17346814]),
     array([2.19714133, 5.42968104, 8.59600587]))

    Plot :math:`J_1`, :math:`J_1'`, :math:`Y_1`, :math:`Y_1'` and their roots.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.special import jnyn_zeros, jvp, jn, yvp, yn
    >>> jn_roots, jnp_roots, yn_roots, ynp_roots = jnyn_zeros(1, 3)
    >>> fig, ax = plt.subplots()
    >>> xmax= 11
    >>> x = np.linspace(0, xmax)
    >>> x[0] += 1e-15
    >>> ax.plot(x, jn(1, x), label=r"$J_1$", c='r')
    >>> ax.plot(x, jvp(1, x, 1), label=r"$J_1'$", c='b')
    >>> ax.plot(x, yn(1, x), label=r"$Y_1$", c='y')
    >>> ax.plot(x, yvp(1, x, 1), label=r"$Y_1'$", c='c')
    >>> zeros = np.zeros((3, ))
    >>> ax.scatter(jn_roots, zeros, s=30, c='r', zorder=5,
    ...            label=r"$J_1$ roots")
    >>> ax.scatter(jnp_roots, zeros, s=30, c='b', zorder=5,
    ...            label=r"$J_1'$ roots")
    >>> ax.scatter(yn_roots, zeros, s=30, c='y', zorder=5,
    ...            label=r"$Y_1$ roots")
    >>> ax.scatter(ynp_roots, zeros, s=30, c='c', zorder=5,
    ...            label=r"$Y_1'$ roots")
    >>> ax.hlines(0, 0, xmax, color='k')
    >>> ax.set_ylim(-0.6, 0.6)
    >>> ax.set_xlim(0, xmax)
    >>> ax.legend(ncol=2, bbox_to_anchor=(1., 0.75))
    >>> plt.tight_layout()
    >>> plt.show()
    """
    if not (isscalar(nt) and isscalar(n)):
        raise ValueError("Arguments must be scalars.")
    if (floor(n) != n) or (floor(nt) != nt):
        raise ValueError("Arguments must be integers.")
    if (nt <= 0):
        raise ValueError("nt > 0")
    return _specfun.jyzo(abs(n), nt)


def jn_zeros(n, nt):
    r"""Compute zeros of integer-order Bessel functions Jn.

    Compute `nt` zeros of the Bessel functions :math:`J_n(x)` on the
    interval :math:`(0, \infty)`. The zeros are returned in ascending
    order. Note that this interval excludes the zero at :math:`x = 0`
    that exists for :math:`n > 0`.

    Parameters
    ----------
    n : int
        Order of Bessel function
    nt : int
        Number of zeros to return

    Returns
    -------
    ndarray
        First `nt` zeros of the Bessel function.

    See Also
    --------
    jv: Real-order Bessel functions of the first kind
    jnp_zeros: Zeros of :math:`Jn'`

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    Examples
    --------
    Compute the first four positive roots of :math:`J_3`.

    >>> from scipy.special import jn_zeros
    >>> jn_zeros(3, 4)
    array([ 6.3801619 ,  9.76102313, 13.01520072, 16.22346616])

    Plot :math:`J_3` and its first four positive roots. Note
    that the root located at 0 is not returned by `jn_zeros`.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.special import jn, jn_zeros
    >>> j3_roots = jn_zeros(3, 4)
    >>> xmax = 18
    >>> xmin = -1
    >>> x = np.linspace(xmin, xmax, 500)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, jn(3, x), label=r'$J_3$')
    >>> ax.scatter(j3_roots, np.zeros((4, )), s=30, c='r',
    ...            label=r"$J_3$_Zeros", zorder=5)
    >>> ax.scatter(0, 0, s=30, c='k',
    ...            label=r"Root at 0", zorder=5)
    >>> ax.hlines(0, 0, xmax, color='k')
    >>> ax.set_xlim(xmin, xmax)
    >>> plt.legend()
    >>> plt.show()
    """
    return jnyn_zeros(n, nt)[0]


def jnp_zeros(n, nt):
    r"""Compute zeros of integer-order Bessel function derivatives Jn'.

    Compute `nt` zeros of the functions :math:`J_n'(x)` on the
    interval :math:`(0, \infty)`. The zeros are returned in ascending
    order. Note that this interval excludes the zero at :math:`x = 0`
    that exists for :math:`n > 1`.

    Parameters
    ----------
    n : int
        Order of Bessel function
    nt : int
        Number of zeros to return

    Returns
    -------
    ndarray
        First `nt` zeros of the Bessel function.

    See Also
    --------
    jvp: Derivatives of integer-order Bessel functions of the first kind
    jv: Float-order Bessel functions of the first kind

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    Examples
    --------
    Compute the first four roots of :math:`J_2'`.

    >>> from scipy.special import jnp_zeros
    >>> jnp_zeros(2, 4)
    array([ 3.05423693,  6.70613319,  9.96946782, 13.17037086])

    As `jnp_zeros` yields the roots of :math:`J_n'`, it can be used to
    compute the locations of the peaks of :math:`J_n`. Plot
    :math:`J_2`, :math:`J_2'` and the locations of the roots of :math:`J_2'`.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.special import jn, jnp_zeros, jvp
    >>> j2_roots = jnp_zeros(2, 4)
    >>> xmax = 15
    >>> x = np.linspace(0, xmax, 500)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, jn(2, x), label=r'$J_2$')
    >>> ax.plot(x, jvp(2, x, 1), label=r"$J_2'$")
    >>> ax.hlines(0, 0, xmax, color='k')
    >>> ax.scatter(j2_roots, np.zeros((4, )), s=30, c='r',
    ...            label=r"Roots of $J_2'$", zorder=5)
    >>> ax.set_ylim(-0.4, 0.8)
    >>> ax.set_xlim(0, xmax)
    >>> plt.legend()
    >>> plt.show()
    """
    return jnyn_zeros(n, nt)[1]


def yn_zeros(n, nt):
    r"""Compute zeros of integer-order Bessel function Yn(x).

    Compute `nt` zeros of the functions :math:`Y_n(x)` on the interval
    :math:`(0, \infty)`. The zeros are returned in ascending order.

    Parameters
    ----------
    n : int
        Order of Bessel function
    nt : int
        Number of zeros to return

    Returns
    -------
    ndarray
        First `nt` zeros of the Bessel function.

    See Also
    --------
    yn: Bessel function of the second kind for integer order
    yv: Bessel function of the second kind for real order

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    Examples
    --------
    Compute the first four roots of :math:`Y_2`.

    >>> from scipy.special import yn_zeros
    >>> yn_zeros(2, 4)
    array([ 3.38424177,  6.79380751, 10.02347798, 13.20998671])

    Plot :math:`Y_2` and its first four roots.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.special import yn, yn_zeros
    >>> xmin = 2
    >>> xmax = 15
    >>> x = np.linspace(xmin, xmax, 500)
    >>> fig, ax = plt.subplots()
    >>> ax.hlines(0, xmin, xmax, color='k')
    >>> ax.plot(x, yn(2, x), label=r'$Y_2$')
    >>> ax.scatter(yn_zeros(2, 4), np.zeros((4, )), s=30, c='r',
    ...            label='Roots', zorder=5)
    >>> ax.set_ylim(-0.4, 0.4)
    >>> ax.set_xlim(xmin, xmax)
    >>> plt.legend()
    >>> plt.show()
    """
    return jnyn_zeros(n, nt)[2]


def ynp_zeros(n, nt):
    r"""Compute zeros of integer-order Bessel function derivatives Yn'(x).

    Compute `nt` zeros of the functions :math:`Y_n'(x)` on the
    interval :math:`(0, \infty)`. The zeros are returned in ascending
    order.

    Parameters
    ----------
    n : int
        Order of Bessel function
    nt : int
        Number of zeros to return

    Returns
    -------
    ndarray
        First `nt` zeros of the Bessel derivative function.


    See Also
    --------
    yvp

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    Examples
    --------
    Compute the first four roots of the first derivative of the
    Bessel function of second kind for order 0 :math:`Y_0'`.

    >>> from scipy.special import ynp_zeros
    >>> ynp_zeros(0, 4)
    array([ 2.19714133,  5.42968104,  8.59600587, 11.74915483])

    Plot :math:`Y_0`, :math:`Y_0'` and confirm visually that the roots of
    :math:`Y_0'` are located at local extrema of :math:`Y_0`.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.special import yn, ynp_zeros, yvp
    >>> zeros = ynp_zeros(0, 4)
    >>> xmax = 13
    >>> x = np.linspace(0, xmax, 500)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, yn(0, x), label=r'$Y_0$')
    >>> ax.plot(x, yvp(0, x, 1), label=r"$Y_0'$")
    >>> ax.scatter(zeros, np.zeros((4, )), s=30, c='r',
    ...            label=r"Roots of $Y_0'$", zorder=5)
    >>> for root in zeros:
    ...     y0_extremum =  yn(0, root)
    ...     lower = min(0, y0_extremum)
    ...     upper = max(0, y0_extremum)
    ...     ax.vlines(root, lower, upper, color='r')
    >>> ax.hlines(0, 0, xmax, color='k')
    >>> ax.set_ylim(-0.6, 0.6)
    >>> ax.set_xlim(0, xmax)
    >>> plt.legend()
    >>> plt.show()
    """
    return jnyn_zeros(n, nt)[3]


def y0_zeros(nt, complex=False):
    """Compute nt zeros of Bessel function Y0(z), and derivative at each zero.

    The derivatives are given by Y0'(z0) = -Y1(z0) at each zero z0.

    Parameters
    ----------
    nt : int
        Number of zeros to return
    complex : bool, default False
        Set to False to return only the real zeros; set to True to return only
        the complex zeros with negative real part and positive imaginary part.
        Note that the complex conjugates of the latter are also zeros of the
        function, but are not returned by this routine.

    Returns
    -------
    z0n : ndarray
        Location of nth zero of Y0(z)
    y0pz0n : ndarray
        Value of derivative Y0'(z0) for nth zero

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    Examples
    --------
    Compute the first 4 real roots and the derivatives at the roots of
    :math:`Y_0`:

    >>> import numpy as np
    >>> from scipy.special import y0_zeros
    >>> zeros, grads = y0_zeros(4)
    >>> with np.printoptions(precision=5):
    ...     print(f"Roots: {zeros}")
    ...     print(f"Gradients: {grads}")
    Roots: [ 0.89358+0.j  3.95768+0.j  7.08605+0.j 10.22235+0.j]
    Gradients: [-0.87942+0.j  0.40254+0.j -0.3001 +0.j  0.2497 +0.j]

    Plot the real part of :math:`Y_0` and the first four computed roots.

    >>> import matplotlib.pyplot as plt
    >>> from scipy.special import y0
    >>> xmin = 0
    >>> xmax = 11
    >>> x = np.linspace(xmin, xmax, 500)
    >>> fig, ax = plt.subplots()
    >>> ax.hlines(0, xmin, xmax, color='k')
    >>> ax.plot(x, y0(x), label=r'$Y_0$')
    >>> zeros, grads = y0_zeros(4)
    >>> ax.scatter(zeros.real, np.zeros((4, )), s=30, c='r',
    ...            label=r'$Y_0$_zeros', zorder=5)
    >>> ax.set_ylim(-0.5, 0.6)
    >>> ax.set_xlim(xmin, xmax)
    >>> plt.legend(ncol=2)
    >>> plt.show()

    Compute the first 4 complex roots and the derivatives at the roots of
    :math:`Y_0` by setting ``complex=True``:

    >>> y0_zeros(4, True)
    (array([ -2.40301663+0.53988231j,  -5.5198767 +0.54718001j,
             -8.6536724 +0.54841207j, -11.79151203+0.54881912j]),
     array([ 0.10074769-0.88196771j, -0.02924642+0.5871695j ,
             0.01490806-0.46945875j, -0.00937368+0.40230454j]))
    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("Arguments must be scalar positive integer.")
    kf = 0
    kc = not complex
    return _specfun.cyzo(nt, kf, kc)


def y1_zeros(nt, complex=False):
    """Compute nt zeros of Bessel function Y1(z), and derivative at each zero.

    The derivatives are given by Y1'(z1) = Y0(z1) at each zero z1.

    Parameters
    ----------
    nt : int
        Number of zeros to return
    complex : bool, default False
        Set to False to return only the real zeros; set to True to return only
        the complex zeros with negative real part and positive imaginary part.
        Note that the complex conjugates of the latter are also zeros of the
        function, but are not returned by this routine.

    Returns
    -------
    z1n : ndarray
        Location of nth zero of Y1(z)
    y1pz1n : ndarray
        Value of derivative Y1'(z1) for nth zero

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    Examples
    --------
    Compute the first 4 real roots and the derivatives at the roots of
    :math:`Y_1`:

    >>> import numpy as np
    >>> from scipy.special import y1_zeros
    >>> zeros, grads = y1_zeros(4)
    >>> with np.printoptions(precision=5):
    ...     print(f"Roots: {zeros}")
    ...     print(f"Gradients: {grads}")
    Roots: [ 2.19714+0.j  5.42968+0.j  8.59601+0.j 11.74915+0.j]
    Gradients: [ 0.52079+0.j -0.34032+0.j  0.27146+0.j -0.23246+0.j]

    Extract the real parts:

    >>> realzeros = zeros.real
    >>> realzeros
    array([ 2.19714133,  5.42968104,  8.59600587, 11.74915483])

    Plot :math:`Y_1` and the first four computed roots.

    >>> import matplotlib.pyplot as plt
    >>> from scipy.special import y1
    >>> xmin = 0
    >>> xmax = 13
    >>> x = np.linspace(xmin, xmax, 500)
    >>> zeros, grads = y1_zeros(4)
    >>> fig, ax = plt.subplots()
    >>> ax.hlines(0, xmin, xmax, color='k')
    >>> ax.plot(x, y1(x), label=r'$Y_1$')
    >>> ax.scatter(zeros.real, np.zeros((4, )), s=30, c='r',
    ...            label=r'$Y_1$_zeros', zorder=5)
    >>> ax.set_ylim(-0.5, 0.5)
    >>> ax.set_xlim(xmin, xmax)
    >>> plt.legend()
    >>> plt.show()

    Compute the first 4 complex roots and the derivatives at the roots of
    :math:`Y_1` by setting ``complex=True``:

    >>> y1_zeros(4, True)
    (array([ -0.50274327+0.78624371j,  -3.83353519+0.56235654j,
             -7.01590368+0.55339305j, -10.17357383+0.55127339j]),
     array([-0.45952768+1.31710194j,  0.04830191-0.69251288j,
            -0.02012695+0.51864253j,  0.011614  -0.43203296j]))
    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("Arguments must be scalar positive integer.")
    kf = 1
    kc = not complex
    return _specfun.cyzo(nt, kf, kc)


def y1p_zeros(nt, complex=False):
    """Compute nt zeros of Bessel derivative Y1'(z), and value at each zero.

    The values are given by Y1(z1) at each z1 where Y1'(z1)=0.

    Parameters
    ----------
    nt : int
        Number of zeros to return
    complex : bool, default False
        Set to False to return only the real zeros; set to True to return only
        the complex zeros with negative real part and positive imaginary part.
        Note that the complex conjugates of the latter are also zeros of the
        function, but are not returned by this routine.

    Returns
    -------
    z1pn : ndarray
        Location of nth zero of Y1'(z)
    y1z1pn : ndarray
        Value of derivative Y1(z1) for nth zero

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    Examples
    --------
    Compute the first four roots of :math:`Y_1'` and the values of
    :math:`Y_1` at these roots.

    >>> import numpy as np
    >>> from scipy.special import y1p_zeros
    >>> y1grad_roots, y1_values = y1p_zeros(4)
    >>> with np.printoptions(precision=5):
    ...     print(f"Y1' Roots: {y1grad_roots.real}")
    ...     print(f"Y1 values: {y1_values.real}")
    Y1' Roots: [ 3.68302  6.9415  10.1234  13.28576]
    Y1 values: [ 0.41673 -0.30317  0.25091 -0.21897]

    `y1p_zeros` can be used to calculate the extremal points of :math:`Y_1`
    directly. Here we plot :math:`Y_1` and the first four extrema.

    >>> import matplotlib.pyplot as plt
    >>> from scipy.special import y1, yvp
    >>> y1_roots, y1_values_at_roots = y1p_zeros(4)
    >>> real_roots = y1_roots.real
    >>> xmax = 15
    >>> x = np.linspace(0, xmax, 500)
    >>> x[0] += 1e-15
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, y1(x), label=r'$Y_1$')
    >>> ax.plot(x, yvp(1, x, 1), label=r"$Y_1'$")
    >>> ax.scatter(real_roots, np.zeros((4, )), s=30, c='r',
    ...            label=r"Roots of $Y_1'$", zorder=5)
    >>> ax.scatter(real_roots, y1_values_at_roots.real, s=30, c='k',
    ...            label=r"Extrema of $Y_1$", zorder=5)
    >>> ax.hlines(0, 0, xmax, color='k')
    >>> ax.set_ylim(-0.5, 0.5)
    >>> ax.set_xlim(0, xmax)
    >>> ax.legend(ncol=2, bbox_to_anchor=(1., 0.75))
    >>> plt.tight_layout()
    >>> plt.show()
    """
    if not isscalar(nt) or (floor(nt) != nt) or (nt <= 0):
        raise ValueError("Arguments must be scalar positive integer.")
    kf = 2
    kc = not complex
    return _specfun.cyzo(nt, kf, kc)


def _bessel_diff_formula(v, z, n, L, phase):
    # from AMS55.
    # L(v, z) = J(v, z), Y(v, z), H1(v, z), H2(v, z), phase = -1
    # L(v, z) = I(v, z) or exp(v*pi*i)K(v, z), phase = 1
    # For K, you can pull out the exp((v-k)*pi*i) into the caller
    v = asarray(v)
    p = 1.0
    s = L(v-n, z)
    for i in range(1, n+1):
        p = phase * (p * (n-i+1)) / i   # = choose(k, i)
        s += p*L(v-n + i*2, z)
    return s / (2.**n)


def jvp(v, z, n=1):
    """Compute derivatives of Bessel functions of the first kind.

    Compute the nth derivative of the Bessel function `Jv` with
    respect to `z`.

    Parameters
    ----------
    v : array_like or float
        Order of Bessel function
    z : complex
        Argument at which to evaluate the derivative; can be real or
        complex.
    n : int, default 1
        Order of derivative. For 0 returns the Bessel function `jv` itself.

    Returns
    -------
    scalar or ndarray
        Values of the derivative of the Bessel function.

    Notes
    -----
    The derivative is computed using the relation DLFM 10.6.7 [2]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    .. [2] NIST Digital Library of Mathematical Functions.
           https://dlmf.nist.gov/10.6.E7

    Examples
    --------

    Compute the Bessel function of the first kind of order 0 and
    its first two derivatives at 1.

    >>> from scipy.special import jvp
    >>> jvp(0, 1, 0), jvp(0, 1, 1), jvp(0, 1, 2)
    (0.7651976865579666, -0.44005058574493355, -0.3251471008130331)

    Compute the first derivative of the Bessel function of the first
    kind for several orders at 1 by providing an array for `v`.

    >>> jvp([0, 1, 2], 1, 1)
    array([-0.44005059,  0.3251471 ,  0.21024362])

    Compute the first derivative of the Bessel function of the first
    kind of order 0 at several points by providing an array for `z`.

    >>> import numpy as np
    >>> points = np.array([0., 1.5, 3.])
    >>> jvp(0, points, 1)
    array([-0.        , -0.55793651, -0.33905896])

    Plot the Bessel function of the first kind of order 1 and its
    first three derivatives.

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-10, 10, 1000)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, jvp(1, x, 0), label=r"$J_1$")
    >>> ax.plot(x, jvp(1, x, 1), label=r"$J_1'$")
    >>> ax.plot(x, jvp(1, x, 2), label=r"$J_1''$")
    >>> ax.plot(x, jvp(1, x, 3), label=r"$J_1'''$")
    >>> plt.legend()
    >>> plt.show()
    """
    n = _nonneg_int_or_fail(n, 'n')
    if n == 0:
        return jv(v, z)
    else:
        return _bessel_diff_formula(v, z, n, jv, -1)


def yvp(v, z, n=1):
    """Compute derivatives of Bessel functions of the second kind.

    Compute the nth derivative of the Bessel function `Yv` with
    respect to `z`.

    Parameters
    ----------
    v : array_like of float
        Order of Bessel function
    z : complex
        Argument at which to evaluate the derivative
    n : int, default 1
        Order of derivative. For 0 returns the BEssel function `yv`

    Returns
    -------
    scalar or ndarray
        nth derivative of the Bessel function.

    See Also
    --------
    yv : Bessel functions of the second kind

    Notes
    -----
    The derivative is computed using the relation DLFM 10.6.7 [2]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    .. [2] NIST Digital Library of Mathematical Functions.
           https://dlmf.nist.gov/10.6.E7

    Examples
    --------
    Compute the Bessel function of the second kind of order 0 and
    its first two derivatives at 1.

    >>> from scipy.special import yvp
    >>> yvp(0, 1, 0), yvp(0, 1, 1), yvp(0, 1, 2)
    (0.088256964215677, 0.7812128213002889, -0.8694697855159659)

    Compute the first derivative of the Bessel function of the second
    kind for several orders at 1 by providing an array for `v`.

    >>> yvp([0, 1, 2], 1, 1)
    array([0.78121282, 0.86946979, 2.52015239])

    Compute the first derivative of the Bessel function of the
    second kind of order 0 at several points by providing an array for `z`.

    >>> import numpy as np
    >>> points = np.array([0.5, 1.5, 3.])
    >>> yvp(0, points, 1)
    array([ 1.47147239,  0.41230863, -0.32467442])

    Plot the Bessel function of the second kind of order 1 and its
    first three derivatives.

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0, 5, 1000)
    >>> x[0] += 1e-15
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, yvp(1, x, 0), label=r"$Y_1$")
    >>> ax.plot(x, yvp(1, x, 1), label=r"$Y_1'$")
    >>> ax.plot(x, yvp(1, x, 2), label=r"$Y_1''$")
    >>> ax.plot(x, yvp(1, x, 3), label=r"$Y_1'''$")
    >>> ax.set_ylim(-10, 10)
    >>> plt.legend()
    >>> plt.show()
    """
    n = _nonneg_int_or_fail(n, 'n')
    if n == 0:
        return yv(v, z)
    else:
        return _bessel_diff_formula(v, z, n, yv, -1)


def kvp(v, z, n=1):
    """Compute derivatives of real-order modified Bessel function Kv(z)

    Kv(z) is the modified Bessel function of the second kind.
    Derivative is calculated with respect to `z`.

    Parameters
    ----------
    v : array_like of float
        Order of Bessel function
    z : array_like of complex
        Argument at which to evaluate the derivative
    n : int, default 1
        Order of derivative. For 0 returns the Bessel function `kv` itself.

    Returns
    -------
    out : ndarray
        The results

    See Also
    --------
    kv

    Notes
    -----
    The derivative is computed using the relation DLFM 10.29.5 [2]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 6.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    .. [2] NIST Digital Library of Mathematical Functions.
           https://dlmf.nist.gov/10.29.E5

    Examples
    --------
    Compute the modified bessel function of the second kind of order 0 and
    its first two derivatives at 1.

    >>> from scipy.special import kvp
    >>> kvp(0, 1, 0), kvp(0, 1, 1), kvp(0, 1, 2)
    (0.42102443824070834, -0.6019072301972346, 1.0229316684379428)

    Compute the first derivative of the modified Bessel function of the second
    kind for several orders at 1 by providing an array for `v`.

    >>> kvp([0, 1, 2], 1, 1)
    array([-0.60190723, -1.02293167, -3.85158503])

    Compute the first derivative of the modified Bessel function of the
    second kind of order 0 at several points by providing an array for `z`.

    >>> import numpy as np
    >>> points = np.array([0.5, 1.5, 3.])
    >>> kvp(0, points, 1)
    array([-1.65644112, -0.2773878 , -0.04015643])

    Plot the modified bessel function of the second kind and its
    first three derivatives.

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0, 5, 1000)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, kvp(1, x, 0), label=r"$K_1$")
    >>> ax.plot(x, kvp(1, x, 1), label=r"$K_1'$")
    >>> ax.plot(x, kvp(1, x, 2), label=r"$K_1''$")
    >>> ax.plot(x, kvp(1, x, 3), label=r"$K_1'''$")
    >>> ax.set_ylim(-2.5, 2.5)
    >>> plt.legend()
    >>> plt.show()
    """
    n = _nonneg_int_or_fail(n, 'n')
    if n == 0:
        return kv(v, z)
    else:
        return (-1)**n * _bessel_diff_formula(v, z, n, kv, 1)


def ivp(v, z, n=1):
    """Compute derivatives of modified Bessel functions of the first kind.

    Compute the nth derivative of the modified Bessel function `Iv`
    with respect to `z`.

    Parameters
    ----------
    v : array_like or float
        Order of Bessel function
    z : array_like
        Argument at which to evaluate the derivative; can be real or
        complex.
    n : int, default 1
        Order of derivative. For 0, returns the Bessel function `iv` itself.

    Returns
    -------
    scalar or ndarray
        nth derivative of the modified Bessel function.

    See Also
    --------
    iv

    Notes
    -----
    The derivative is computed using the relation DLFM 10.29.5 [2]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 6.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    .. [2] NIST Digital Library of Mathematical Functions.
           https://dlmf.nist.gov/10.29.E5

    Examples
    --------
    Compute the modified Bessel function of the first kind of order 0 and
    its first two derivatives at 1.

    >>> from scipy.special import ivp
    >>> ivp(0, 1, 0), ivp(0, 1, 1), ivp(0, 1, 2)
    (1.2660658777520084, 0.565159103992485, 0.7009067737595233)

    Compute the first derivative of the modified Bessel function of the first
    kind for several orders at 1 by providing an array for `v`.

    >>> ivp([0, 1, 2], 1, 1)
    array([0.5651591 , 0.70090677, 0.29366376])

    Compute the first derivative of the modified Bessel function of the
    first kind of order 0 at several points by providing an array for `z`.

    >>> import numpy as np
    >>> points = np.array([0., 1.5, 3.])
    >>> ivp(0, points, 1)
    array([0.        , 0.98166643, 3.95337022])

    Plot the modified Bessel function of the first kind of order 1 and its
    first three derivatives.

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(-5, 5, 1000)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, ivp(1, x, 0), label=r"$I_1$")
    >>> ax.plot(x, ivp(1, x, 1), label=r"$I_1'$")
    >>> ax.plot(x, ivp(1, x, 2), label=r"$I_1''$")
    >>> ax.plot(x, ivp(1, x, 3), label=r"$I_1'''$")
    >>> plt.legend()
    >>> plt.show()
    """
    n = _nonneg_int_or_fail(n, 'n')
    if n == 0:
        return iv(v, z)
    else:
        return _bessel_diff_formula(v, z, n, iv, 1)


def h1vp(v, z, n=1):
    """Compute derivatives of Hankel function H1v(z) with respect to `z`.

    Parameters
    ----------
    v : array_like
        Order of Hankel function
    z : array_like
        Argument at which to evaluate the derivative. Can be real or
        complex.
    n : int, default 1
        Order of derivative. For 0 returns the Hankel function `h1v` itself.

    Returns
    -------
    scalar or ndarray
        Values of the derivative of the Hankel function.

    See Also
    --------
    hankel1

    Notes
    -----
    The derivative is computed using the relation DLFM 10.6.7 [2]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    .. [2] NIST Digital Library of Mathematical Functions.
           https://dlmf.nist.gov/10.6.E7

    Examples
    --------
    Compute the Hankel function of the first kind of order 0 and
    its first two derivatives at 1.

    >>> from scipy.special import h1vp
    >>> h1vp(0, 1, 0), h1vp(0, 1, 1), h1vp(0, 1, 2)
    ((0.7651976865579664+0.088256964215677j),
     (-0.44005058574493355+0.7812128213002889j),
     (-0.3251471008130329-0.8694697855159659j))

    Compute the first derivative of the Hankel function of the first kind
    for several orders at 1 by providing an array for `v`.

    >>> h1vp([0, 1, 2], 1, 1)
    array([-0.44005059+0.78121282j,  0.3251471 +0.86946979j,
           0.21024362+2.52015239j])

    Compute the first derivative of the Hankel function of the first kind
    of order 0 at several points by providing an array for `z`.

    >>> import numpy as np
    >>> points = np.array([0.5, 1.5, 3.])
    >>> h1vp(0, points, 1)
    array([-0.24226846+1.47147239j, -0.55793651+0.41230863j,
           -0.33905896-0.32467442j])
    """
    n = _nonneg_int_or_fail(n, 'n')
    if n == 0:
        return hankel1(v, z)
    else:
        return _bessel_diff_formula(v, z, n, hankel1, -1)


def h2vp(v, z, n=1):
    """Compute derivatives of Hankel function H2v(z) with respect to `z`.

    Parameters
    ----------
    v : array_like
        Order of Hankel function
    z : array_like
        Argument at which to evaluate the derivative. Can be real or
        complex.
    n : int, default 1
        Order of derivative. For 0 returns the Hankel function `h2v` itself.

    Returns
    -------
    scalar or ndarray
        Values of the derivative of the Hankel function.

    See Also
    --------
    hankel2

    Notes
    -----
    The derivative is computed using the relation DLFM 10.6.7 [2]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    .. [2] NIST Digital Library of Mathematical Functions.
           https://dlmf.nist.gov/10.6.E7

    Examples
    --------
    Compute the Hankel function of the second kind of order 0 and
    its first two derivatives at 1.

    >>> from scipy.special import h2vp
    >>> h2vp(0, 1, 0), h2vp(0, 1, 1), h2vp(0, 1, 2)
    ((0.7651976865579664-0.088256964215677j),
     (-0.44005058574493355-0.7812128213002889j),
     (-0.3251471008130329+0.8694697855159659j))

    Compute the first derivative of the Hankel function of the second kind
    for several orders at 1 by providing an array for `v`.

    >>> h2vp([0, 1, 2], 1, 1)
    array([-0.44005059-0.78121282j,  0.3251471 -0.86946979j,
           0.21024362-2.52015239j])

    Compute the first derivative of the Hankel function of the second kind
    of order 0 at several points by providing an array for `z`.

    >>> import numpy as np
    >>> points = np.array([0.5, 1.5, 3.])
    >>> h2vp(0, points, 1)
    array([-0.24226846-1.47147239j, -0.55793651-0.41230863j,
           -0.33905896+0.32467442j])
    """
    n = _nonneg_int_or_fail(n, 'n')
    if n == 0:
        return hankel2(v, z)
    else:
        return _bessel_diff_formula(v, z, n, hankel2, -1)


def riccati_jn(n, x):
    r"""Compute Ricatti-Bessel function of the first kind and its derivative.

    The Ricatti-Bessel function of the first kind is defined as :math:`x
    j_n(x)`, where :math:`j_n` is the spherical Bessel function of the first
    kind of order :math:`n`.

    This function computes the value and first derivative of the
    Ricatti-Bessel function for all orders up to and including `n`.

    Parameters
    ----------
    n : int
        Maximum order of function to compute
    x : float
        Argument at which to evaluate

    Returns
    -------
    jn : ndarray
        Value of j0(x), ..., jn(x)
    jnp : ndarray
        First derivative j0'(x), ..., jn'(x)

    Notes
    -----
    The computation is carried out via backward recurrence, using the
    relation DLMF 10.51.1 [2]_.

    Wrapper for a Fortran routine created by Shanjie Zhang and Jianming
    Jin [1]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html
    .. [2] NIST Digital Library of Mathematical Functions.
           https://dlmf.nist.gov/10.51.E1

    """
    if not (isscalar(n) and isscalar(x)):
        raise ValueError("arguments must be scalars.")
    n = _nonneg_int_or_fail(n, 'n', strict=False)
    if (n == 0):
        n1 = 1
    else:
        n1 = n

    jn = np.empty((n1 + 1,), dtype=np.float64)
    jnp = np.empty_like(jn)

    _rctj(x, out=(jn, jnp))
    return jn[:(n+1)], jnp[:(n+1)]


def riccati_yn(n, x):
    """Compute Ricatti-Bessel function of the second kind and its derivative.

    The Ricatti-Bessel function of the second kind is defined here as :math:`+x
    y_n(x)`, where :math:`y_n` is the spherical Bessel function of the second
    kind of order :math:`n`. *Note that this is in contrast to a common convention
    that includes a minus sign in the definition.*

    This function computes the value and first derivative of the function for
    all orders up to and including `n`.

    Parameters
    ----------
    n : int
        Maximum order of function to compute
    x : float
        Argument at which to evaluate

    Returns
    -------
    yn : ndarray
        Value of y0(x), ..., yn(x)
    ynp : ndarray
        First derivative y0'(x), ..., yn'(x)

    Notes
    -----
    The computation is carried out via ascending recurrence, using the
    relation DLMF 10.51.1 [2]_.

    Wrapper for a Fortran routine created by Shanjie Zhang and Jianming
    Jin [1]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html
    .. [2] NIST Digital Library of Mathematical Functions.
           https://dlmf.nist.gov/10.51.E1

    """
    if not (isscalar(n) and isscalar(x)):
        raise ValueError("arguments must be scalars.")
    n = _nonneg_int_or_fail(n, 'n', strict=False)
    if (n == 0):
        n1 = 1
    else:
        n1 = n

    yn = np.empty((n1 + 1,), dtype=np.float64)
    ynp = np.empty_like(yn)
    _rcty(x, out=(yn, ynp))

    return yn[:(n+1)], ynp[:(n+1)]


def erf_zeros(nt):
    """Compute the first nt zero in the first quadrant, ordered by absolute value.

    Zeros in the other quadrants can be obtained by using the symmetries
    erf(-z) = erf(z) and erf(conj(z)) = conj(erf(z)).


    Parameters
    ----------
    nt : int
        The number of zeros to compute

    Returns
    -------
    The locations of the zeros of erf : ndarray (complex)
        Complex values at which zeros of erf(z)

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    Examples
    --------
    >>> from scipy import special
    >>> special.erf_zeros(1)
    array([1.45061616+1.880943j])

    Check that erf is (close to) zero for the value returned by erf_zeros

    >>> special.erf(special.erf_zeros(1))
    array([4.95159469e-14-1.16407394e-16j])

    """
    if (floor(nt) != nt) or (nt <= 0) or not isscalar(nt):
        raise ValueError("Argument must be positive scalar integer.")
    return _specfun.cerzo(nt)


def fresnelc_zeros(nt):
    """Compute nt complex zeros of cosine Fresnel integral C(z).

    Parameters
    ----------
    nt : int
        Number of zeros to compute

    Returns
    -------
    fresnelc_zeros: ndarray
        Zeros of the cosine Fresnel integral

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
