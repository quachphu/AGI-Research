    if x >= 0.5:  # Exact: 2 * smirnov
        prob = 2 * scipy.special.smirnov(n, x)
        return _select_and_clip_prob(1.0 - prob, prob, cdf=cdf)

    nxsquared = t * x
    if n <= 140:
        if nxsquared <= 0.754693:
            prob = _kolmogn_DMTW(n, x, cdf=True)
            return _select_and_clip_prob(prob, 1.0 - prob, cdf=cdf)
        if nxsquared <= 4:
            prob = _kolmogn_Pomeranz(n, x, cdf=True)
            return _select_and_clip_prob(prob, 1.0 - prob, cdf=cdf)
        # Now use Miller approximation of 2*smirnov
        prob = 2 * scipy.special.smirnov(n, x)
        return _select_and_clip_prob(1.0 - prob, prob, cdf=cdf)

    # Split CDF and SF as they have different cutoffs on nxsquared.
    if not cdf:
        if nxsquared >= 370.0:
            return 0.0
        if nxsquared >= 2.2:
            prob = 2 * scipy.special.smirnov(n, x)
            return _clip_prob(prob)
        # Fall through and compute the SF as 1.0-CDF
    if nxsquared >= 18.0:
        cdfprob = 1.0
    elif n <= 100000 and n * x**1.5 <= 1.4:
        cdfprob = _kolmogn_DMTW(n, x, cdf=True)
    else:
        cdfprob = _kolmogn_PelzGood(n, x, cdf=True)
    return _select_and_clip_prob(cdfprob, 1.0 - cdfprob, cdf=cdf)


def _kolmogn_p(n, x):
    """Computes the PDF for the two-sided Kolmogorov-Smirnov statistic.

    x must be of type float, n of type integer.
    """
    if np.isnan(n):
        return n  # Keep the same type of nan
    if int(n) != n or n <= 0:
        return np.nan
    if x >= 1.0 or x <= 0:
        return 0
    t = n * x
    if t <= 1.0:
        # Ruben-Gambino: n!/n^n * (2t-1)^n -> 2 n!/n^n * n^2 * (2t-1)^(n-1)
        if t <= 0.5:
            return 0.0
        if n <= 140:
            prd = np.prod(np.arange(1, n) * (1.0 / n) * (2 * t - 1))
        else:
            prd = np.exp(_log_nfactorial_div_n_pow_n(n) + (n-1) * np.log(2 * t - 1))
        return prd * 2 * n**2
    if t >= n - 1:
        # Ruben-Gambino : 1-2(1-x)**n -> 2n*(1-x)**(n-1)
        return 2 * (1.0 - x) ** (n-1) * n
    if x >= 0.5:
        return 2 * scipy.stats.ksone.pdf(x, n)

    # Just take a small delta.
    # Ideally x +/- delta would stay within [i/n, (i+1)/n] for some integer a.
    # as the CDF is a piecewise degree n polynomial.
    # It has knots at 1/n, 2/n, ... (n-1)/n
    # and is not a C-infinity function at the knots
    delta = x / 2.0**16
    delta = min(delta, x - 1.0/n)
    delta = min(delta, 0.5 - x)

    def _kk(_x):
        return kolmogn(n, _x)

    return _derivative(_kk, x, dx=delta, order=5)


def _kolmogni(n, p, q):
    """Computes the PPF/ISF of kolmogn.

    n of type integer, n>= 1
    p is the CDF, q the SF, p+q=1
    """
    if np.isnan(n):
        return n  # Keep the same type of nan
    if int(n) != n or n <= 0:
        return np.nan
    if p <= 0:
        return 1.0/n
    if q <= 0:
        return 1.0
    delta = np.exp((np.log(p) - scipy.special.loggamma(n+1))/n)
    if delta <= 1.0/n:
        return (delta + 1.0 / n) / 2
    x = -np.expm1(np.log(q/2.0)/n)
    if x >= 1 - 1.0/n:
        return x
    x1 = scu._kolmogci(p)/np.sqrt(n)
    x1 = min(x1, 1.0 - 1.0/n)

    def _f(x):
        return _kolmogn(n, x) - p

    return scipy.optimize.brentq(_f, 1.0/n, x1, xtol=1e-14)


def kolmogn(n, x, cdf=True):
    """Computes the CDF for the two-sided Kolmogorov-Smirnov distribution.

    The two-sided Kolmogorov-Smirnov distribution has as its CDF Pr(D_n <= x),
    for a sample of size n drawn from a distribution with CDF F(t), where
    :math:`D_n &= sup_t |F_n(t) - F(t)|`, and
    :math:`F_n(t)` is the Empirical Cumulative Distribution Function of the sample.

    Parameters
    ----------
    n : integer, array_like
        the number of samples
    x : float, array_like
        The K-S statistic, float between 0 and 1
    cdf : bool, optional
        whether to compute the CDF(default=true) or the SF.

    Returns
    -------
    cdf : ndarray
        CDF (or SF it cdf is False) at the specified locations.

    The return value has shape the result of numpy broadcasting n and x.
    """
    it = np.nditer([n, x, cdf, None], flags=['zerosize_ok'],
                   op_dtypes=[None, np.float64, np.bool_, np.float64])
    for _n, _x, _cdf, z in it:
        if np.isnan(_n):
            z[...] = _n
            continue
        if int(_n) != _n:
            raise ValueError(f'n is not integral: {_n}')
        z[...] = _kolmogn(int(_n), _x, cdf=_cdf)
    result = it.operands[-1]
    return result


def kolmognp(n, x):
    """Computes the PDF for the two-sided Kolmogorov-Smirnov distribution.

    Parameters
    ----------
    n : integer, array_like
        the number of samples
    x : float, array_like
        The K-S statistic, float between 0 and 1

    Returns
    -------
    pdf : ndarray
        The PDF at the specified locations

    The return value has shape the result of numpy broadcasting n and x.
    """
    it = np.nditer([n, x, None])
    for _n, _x, z in it:
        if np.isnan(_n):
            z[...] = _n
            continue
        if int(_n) != _n:
            raise ValueError(f'n is not integral: {_n}')
        z[...] = _kolmogn_p(int(_n), _x)
    result = it.operands[-1]
    return result


def kolmogni(n, q, cdf=True):
    """Computes the PPF(or ISF) for the two-sided Kolmogorov-Smirnov distribution.

    Parameters
    ----------
    n : integer, array_like
        the number of samples
    q : float, array_like
        Probabilities, float between 0 and 1
    cdf : bool, optional
        whether to compute the PPF(default=true) or the ISF.

    Returns
    -------
    ppf : ndarray
        PPF (or ISF if cdf is False) at the specified locations

    The return value has shape the result of numpy broadcasting n and x.
    """
    it = np.nditer([n, q, cdf, None])
    for _n, _q, _cdf, z in it:
        if np.isnan(_n):
            z[...] = _n
            continue
        if int(_n) != _n:
            raise ValueError(f'n is not integral: {_n}')
        _pcdf, _psf = (_q, 1-_q) if _cdf else (1-_q, _q)
        z[...] = _kolmogni(int(_n), _pcdf, _psf)
    result = it.operands[-1]
    return result
<|endoftext|>import warnings
import numpy as np
from . import distributions
from .._lib._bunch import _make_tuple_bunch
from ._stats_pythran import siegelslopes as siegelslopes_pythran

__all__ = ['_find_repeats', 'theilslopes', 'siegelslopes']

# This is not a namedtuple for backwards compatibility. See PR #12983
TheilslopesResult = _make_tuple_bunch('TheilslopesResult',
                                      ['slope', 'intercept',
                                       'low_slope', 'high_slope'])
SiegelslopesResult = _make_tuple_bunch('SiegelslopesResult',
                                       ['slope', 'intercept'])


def theilslopes(y, x=None, alpha=0.95, method='separate'):
    r"""
    Computes the Theil-Sen estimator for a set of points (x, y).

    `theilslopes` implements a method for robust linear regression.  It
    computes the slope as the median of all slopes between paired values.

    Parameters
    ----------
    y : array_like
        Dependent variable.
    x : array_like or None, optional
        Independent variable. If None, use ``arange(len(y))`` instead.
    alpha : float, optional
        Confidence degree between 0 and 1. Default is 95% confidence.
        Note that `alpha` is symmetric around 0.5, i.e. both 0.1 and 0.9 are
        interpreted as "find the 90% confidence interval".
    method : {'joint', 'separate'}, optional
        Method to be used for computing estimate for intercept.
        Following methods are supported,

            * 'joint': Uses np.median(y - slope * x) as intercept.
            * 'separate': Uses np.median(y) - slope * np.median(x)
                          as intercept.

        The default is 'separate'.

        .. versionadded:: 1.8.0

    Returns
    -------
    result : ``TheilslopesResult`` instance
        The return value is an object with the following attributes:

        slope : float
            Theil slope.
        intercept : float
            Intercept of the Theil line.
        low_slope : float
            Lower bound of the confidence interval on `slope`.
        high_slope : float
            Upper bound of the confidence interval on `slope`.

    See Also
    --------
    siegelslopes : a similar technique using repeated medians

    Notes
    -----
    The implementation of `theilslopes` follows [1]_. The intercept is
    not defined in [1]_, and here it is defined as ``median(y) -
    slope*median(x)``, which is given in [3]_. Other definitions of
    the intercept exist in the literature such as  ``median(y - slope*x)``
    in [4]_. The approach to compute the intercept can be determined by the
    parameter ``method``. A confidence interval for the intercept is not
    given as this question is not addressed in [1]_.

    For compatibility with older versions of SciPy, the return value acts
    like a ``namedtuple`` of length 4, with fields ``slope``, ``intercept``,
    ``low_slope``, and ``high_slope``, so one can continue to write::

        slope, intercept, low_slope, high_slope = theilslopes(y, x)

    References
    ----------
    .. [1] P.K. Sen, "Estimates of the regression coefficient based on
           Kendall's tau", J. Am. Stat. Assoc., Vol. 63, pp. 1379-1389, 1968.
    .. [2] H. Theil, "A rank-invariant method of linear and polynomial
           regression analysis I, II and III",  Nederl. Akad. Wetensch., Proc.
           53:, pp. 386-392, pp. 521-525, pp. 1397-1412, 1950.
    .. [3] W.L. Conover, "Practical nonparametric statistics", 2nd ed.,
           John Wiley and Sons, New York, pp. 493.
    .. [4] https://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt

    >>> x = np.linspace(-5, 5, num=150)
    >>> y = x + np.random.normal(size=x.size)
    >>> y[11:15] += 10  # add outliers
    >>> y[-5:] -= 7

    Compute the slope, intercept and 90% confidence interval.  For comparison,
    also compute the least-squares fit with `linregress`:

    >>> res = stats.theilslopes(y, x, 0.90, method='separate')
    >>> lsq_res = stats.linregress(x, y)

    Plot the results. The Theil-Sen regression line is shown in red, with the
    dashed red lines illustrating the confidence interval of the slope (note
    that the dashed red lines are not the confidence interval of the regression
    as the confidence interval of the intercept is not included). The green
    line shows the least-squares fit for comparison.

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot(x, y, 'b.')
    >>> ax.plot(x, res[1] + res[0] * x, 'r-')
    >>> ax.plot(x, res[1] + res[2] * x, 'r--')
    >>> ax.plot(x, res[1] + res[3] * x, 'r--')
    >>> ax.plot(x, lsq_res[1] + lsq_res[0] * x, 'g-')
    >>> plt.show()

    """
    if method not in ['joint', 'separate']:
        raise ValueError("method must be either 'joint' or 'separate'."
                         f"'{method}' is invalid.")
    # We copy both x and y so we can use _find_repeats.
    y = np.array(y, dtype=float, copy=True).ravel()
    if x is None:
        x = np.arange(len(y), dtype=float)
    else:
        x = np.array(x, dtype=float, copy=True).ravel()
        if len(x) != len(y):
            raise ValueError(f"Incompatible lengths ! ({len(y)}<>{len(x)})")

    # Compute sorted slopes only when deltax > 0
    deltax = x[:, np.newaxis] - x
    deltay = y[:, np.newaxis] - y
    slopes = deltay[deltax > 0] / deltax[deltax > 0]
    if not slopes.size:
        msg = "All `x` coordinates are identical."
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    slopes.sort()
    medslope = np.median(slopes)
    if method == 'joint':
        medinter = np.median(y - medslope * x)
    else:
        medinter = np.median(y) - medslope * np.median(x)
    # Now compute confidence intervals
    if alpha > 0.5:
        alpha = 1. - alpha

    z = distributions.norm.ppf(alpha / 2.)
    # This implements (2.6) from Sen (1968)
    _, nxreps = _find_repeats(x)
    _, nyreps = _find_repeats(y)
    nt = len(slopes)       # N in Sen (1968)
    ny = len(y)            # n in Sen (1968)
    # Equation 2.6 in Sen (1968):
    sigsq = 1/18. * (ny * (ny-1) * (2*ny+5) -
                     sum(k * (k-1) * (2*k + 5) for k in nxreps) -
                     sum(k * (k-1) * (2*k + 5) for k in nyreps))
    # Find the confidence interval indices in `slopes`
    try:
        sigma = np.sqrt(sigsq)
        Ru = min(int(np.round((nt - z*sigma)/2.)), len(slopes)-1)
        Rl = max(int(np.round((nt + z*sigma)/2.)) - 1, 0)
        delta = slopes[[Rl, Ru]]
    except (ValueError, IndexError):
        delta = (np.nan, np.nan)

    return TheilslopesResult(slope=medslope, intercept=medinter,
                             low_slope=delta[0], high_slope=delta[1])


def _find_repeats(arr):
    # This function assumes it may clobber its input.
    if len(arr) == 0:
        return np.array(0, np.float64), np.array(0, np.intp)

    # XXX This cast was previously needed for the Fortran implementation,
    # should we ditch it?
    arr = np.asarray(arr, np.float64).ravel()
    arr.sort()

    # Taken from NumPy 1.9's np.unique.
    change = np.concatenate(([True], arr[1:] != arr[:-1]))
    unique = arr[change]
    change_idx = np.concatenate(np.nonzero(change) + ([arr.size],))
    freq = np.diff(change_idx)
    atleast2 = freq > 1
    return unique[atleast2], freq[atleast2]


def siegelslopes(y, x=None, method="hierarchical"):
    r"""
    Computes the Siegel estimator for a set of points (x, y).

    `siegelslopes` implements a method for robust linear regression
    using repeated medians (see [1]_) to fit a line to the points (x, y).
    The method is robust to outliers with an asymptotic breakdown point
    of 50%.

    Parameters
    ----------
    y : array_like
        Dependent variable.
    x : array_like or None, optional
        Independent variable. If None, use ``arange(len(y))`` instead.
    method : {'hierarchical', 'separate'}
        If 'hierarchical', estimate the intercept using the estimated
        slope ``slope`` (default option).
        If 'separate', estimate the intercept independent of the estimated
        slope. See Notes for details.

    Returns
    -------
    result : ``SiegelslopesResult`` instance
        The return value is an object with the following attributes:

        slope : float
            Estimate of the slope of the regression line.
        intercept : float
            Estimate of the intercept of the regression line.

    See Also
    --------
    theilslopes : a similar technique without repeated medians

    Notes
    -----
    With ``n = len(y)``, compute ``m_j`` as the median of
    the slopes from the point ``(x[j], y[j])`` to all other `n-1` points.
    ``slope`` is then the median of all slopes ``m_j``.
    Two ways are given to estimate the intercept in [1]_ which can be chosen
    via the parameter ``method``.
    The hierarchical approach uses the estimated slope ``slope``
    and computes ``intercept`` as the median of ``y - slope*x``.
    The other approach estimates the intercept separately as follows: for
    each point ``(x[j], y[j])``, compute the intercepts of all the `n-1`
    lines through the remaining points and take the median ``i_j``.
    ``intercept`` is the median of the ``i_j``.

    The implementation computes `n` times the median of a vector of size `n`
    which can be slow for large vectors. There are more efficient algorithms
    (see [2]_) which are not implemented here.

    For compatibility with older versions of SciPy, the return value acts
    like a ``namedtuple`` of length 2, with fields ``slope`` and
    ``intercept``, so one can continue to write::

        slope, intercept = siegelslopes(y, x)

    References
    ----------
    .. [1] A. Siegel, "Robust Regression Using Repeated Medians",
           Biometrika, Vol. 69, pp. 242-244, 1982.

    .. [2] A. Stein and M. Werman, "Finding the repeated median regression
           line", Proceedings of the Third Annual ACM-SIAM Symposium on
           Discrete Algorithms, pp. 409-413, 1992.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt

    >>> x = np.linspace(-5, 5, num=150)
    >>> y = x + np.random.normal(size=x.size)
    >>> y[11:15] += 10  # add outliers
    >>> y[-5:] -= 7

    Compute the slope and intercept.  For comparison, also compute the
    least-squares fit with `linregress`:

    >>> res = stats.siegelslopes(y, x)
    >>> lsq_res = stats.linregress(x, y)

    Plot the results. The Siegel regression line is shown in red. The green
    line shows the least-squares fit for comparison.

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot(x, y, 'b.')
    >>> ax.plot(x, res[1] + res[0] * x, 'r-')
    >>> ax.plot(x, lsq_res[1] + lsq_res[0] * x, 'g-')
    >>> plt.show()

    """
    if method not in ['hierarchical', 'separate']:
        raise ValueError("method can only be 'hierarchical' or 'separate'")
    y = np.asarray(y).ravel()
    if x is None:
        x = np.arange(len(y), dtype=float)
    else:
        x = np.asarray(x, dtype=float).ravel()
        if len(x) != len(y):
            raise ValueError(f"Incompatible lengths ! ({len(y)}<>{len(x)})")
    dtype = np.result_type(x, y, np.float32)  # use at least float32
    y, x = y.astype(dtype), x.astype(dtype)
    medslope, medinter = siegelslopes_pythran(y, x, method)
    return SiegelslopesResult(slope=medslope, intercept=medinter)
<|endoftext|>import numpy as np

from scipy._lib._util import _get_nan
from scipy._lib._array_api import array_namespace, xp_copysign

from ._axis_nan_policy import _axis_nan_policy_factory


@_axis_nan_policy_factory(
    lambda x: x, n_outputs=1, result_to_tuple=lambda x: (x,)
)
def variation(a, axis=0, nan_policy='propagate', ddof=0, *, keepdims=False):
    """
    Compute the coefficient of variation.

    The coefficient of variation is the standard deviation divided by the
    mean.  This function is equivalent to::

        np.std(x, axis=axis, ddof=ddof) / np.mean(x)

    The default for ``ddof`` is 0, but many definitions of the coefficient
    of variation use the square root of the unbiased sample variance
    for the sample standard deviation, which corresponds to ``ddof=1``.

    The function does not take the absolute value of the mean of the data,
    so the return value is negative if the mean is negative.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or None, optional
        Axis along which to calculate the coefficient of variation.
        Default is 0. If None, compute over the whole array `a`.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains ``nan``.
        The following options are available:

          * 'propagate': return ``nan``
          * 'raise': raise an exception
          * 'omit': perform the calculation with ``nan`` values omitted

        The default is 'propagate'.
    ddof : int, optional
        Gives the "Delta Degrees Of Freedom" used when computing the
        standard deviation.  The divisor used in the calculation of the
        standard deviation is ``N - ddof``, where ``N`` is the number of
        elements.  `ddof` must be less than ``N``; if it isn't, the result
        will be ``nan`` or ``inf``, depending on ``N`` and the values in
        the array.  By default `ddof` is zero for backwards compatibility,
        but it is recommended to use ``ddof=1`` to ensure that the sample
        standard deviation is computed as the square root of the unbiased
        sample variance.

    Returns
    -------
    variation : ndarray
        The calculated variation along the requested axis.

    Notes
    -----
    There are several edge cases that are handled without generating a
    warning:

    * If both the mean and the standard deviation are zero, ``nan``
      is returned.
    * If the mean is zero and the standard deviation is nonzero, ``inf``
      is returned.
    * If the input has length zero (either because the array has zero
      length, or all the input values are ``nan`` and ``nan_policy`` is
      ``'omit'``), ``nan`` is returned.
    * If the input contains ``inf``, ``nan`` is returned.

    References
    ----------
    .. [1] Zwillinger, D. and Kokoska, S. (2000). CRC Standard
       Probability and Statistics Tables and Formulae. Chapman & Hall: New
       York. 2000.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import variation
    >>> variation([1, 2, 3, 4, 5], ddof=1)
    0.5270462766947299

    Compute the variation along a given dimension of an array that contains
    a few ``nan`` values:

    >>> x = np.array([[  10.0, np.nan, 11.0, 19.0, 23.0, 29.0, 98.0],
    ...               [  29.0,   30.0, 32.0, 33.0, 35.0, 56.0, 57.0],
    ...               [np.nan, np.nan, 12.0, 13.0, 16.0, 16.0, 17.0]])
    >>> variation(x, axis=1, ddof=1, nan_policy='omit')
    array([1.05109361, 0.31428986, 0.146483  ])

    """
    xp = array_namespace(a)
    a = xp.asarray(a)
    # `nan_policy` and `keepdims` are handled by `_axis_nan_policy`
    # `axis=None` is only handled for NumPy backend
    if axis is None:
        a = xp.reshape(a, (-1,))
        axis = 0

    n = a.shape[axis]
    NaN = _get_nan(a)

    if a.size == 0 or ddof > n:
        # Handle as a special case to avoid spurious warnings.
        # The return values, if any, are all nan.
        shp = list(a.shape)
        shp.pop(axis)
        result = xp.full(shp, fill_value=NaN)
        return result[()] if result.ndim == 0 else result

    mean_a = xp.mean(a, axis=axis)

    if ddof == n:
        # Another special case.  Result is either inf or nan.
        std_a = xp.std(a, axis=axis, correction=0)
        result = xp.where(std_a > 0, xp_copysign(xp.asarray(xp.inf), mean_a), NaN)
        return result[()] if result.ndim == 0 else result

    with np.errstate(divide='ignore', invalid='ignore'):
        std_a = xp.std(a, axis=axis, correction=ddof)
        result = std_a / mean_a

    return result[()] if result.ndim == 0 else result
<|endoftext|>import numpy as np

from scipy.special import ndtri
from scipy.optimize import brentq
from ._discrete_distns import nchypergeom_fisher
from ._common import ConfidenceInterval


def _sample_odds_ratio(table):
    """
    Given a table [[a, b], [c, d]], compute a*d/(b*c).

    Return nan if the numerator and denominator are 0.
    Return inf if just the denominator is 0.
    """
    # table must be a 2x2 numpy array.
    if table[1, 0] > 0 and table[0, 1] > 0:
        oddsratio = table[0, 0] * table[1, 1] / (table[1, 0] * table[0, 1])
    elif table[0, 0] == 0 or table[1, 1] == 0:
        oddsratio = np.nan
    else:
        oddsratio = np.inf
    return oddsratio


def _solve(func):
    """
    Solve func(nc) = 0.  func must be an increasing function.
    """
    # We could just as well call the variable `x` instead of `nc`, but we
    # always call this function with functions for which nc (the noncentrality
    # parameter) is the variable for which we are solving.
    nc = 1.0
    value = func(nc)
    if value == 0:
        return nc

    # Multiplicative factor by which to increase or decrease nc when
    # searching for a bracketing interval.
    factor = 2.0
    # Find a bracketing interval.
    if value > 0:
        nc /= factor
        while func(nc) > 0:
            nc /= factor
        lo = nc
        hi = factor*nc
    else:
        nc *= factor
        while func(nc) < 0:
            nc *= factor
        lo = nc/factor
        hi = nc

    # lo and hi bracket the solution for nc.
    nc = brentq(func, lo, hi, xtol=1e-13)
    return nc


def _nc_hypergeom_mean_inverse(x, M, n, N):
    """
    For the given noncentral hypergeometric parameters x, M, n,and N
    (table[0,0], total, row 0 sum and column 0 sum, resp., of a 2x2
    contingency table), find the noncentrality parameter of Fisher's
    noncentral hypergeometric distribution whose mean is x.
    """
    nc = _solve(lambda nc: nchypergeom_fisher.mean(M, n, N, nc) - x)
    return nc


def _hypergeom_params_from_table(table):
    # The notation M, n and N is consistent with stats.hypergeom and
    # stats.nchypergeom_fisher.
    x = table[0, 0]
    M = table.sum()
    n = table[0].sum()
    N = table[:, 0].sum()
    return x, M, n, N


def _ci_upper(table, alpha):
    """
    Compute the upper end of the confidence interval.
    """
    if _sample_odds_ratio(table) == np.inf:
        return np.inf

    x, M, n, N = _hypergeom_params_from_table(table)

    # nchypergeom_fisher.cdf is a decreasing function of nc, so we negate
    # it in the lambda expression.
    nc = _solve(lambda nc: -nchypergeom_fisher.cdf(x, M, n, N, nc) + alpha)
    return nc


def _ci_lower(table, alpha):
    """
    Compute the lower end of the confidence interval.
    """
    if _sample_odds_ratio(table) == 0:
        return 0

    x, M, n, N = _hypergeom_params_from_table(table)

    nc = _solve(lambda nc: nchypergeom_fisher.sf(x - 1, M, n, N, nc) - alpha)
    return nc


def _conditional_oddsratio(table):
    """
    Conditional MLE of the odds ratio for the 2x2 contingency table.
    """
    x, M, n, N = _hypergeom_params_from_table(table)
    # Get the bounds of the support.  The support of the noncentral
    # hypergeometric distribution with parameters M, n, and N is the same
    # for all values of the noncentrality parameter, so we can use 1 here.
    lo, hi = nchypergeom_fisher.support(M, n, N, 1)

    # Check if x is at one of the extremes of the support.  If so, we know
    # the odds ratio is either 0 or inf.
    if x == lo:
        # x is at the low end of the support.
        return 0
    if x == hi:
        # x is at the high end of the support.
        return np.inf

    nc = _nc_hypergeom_mean_inverse(x, M, n, N)
    return nc


def _conditional_oddsratio_ci(table, confidence_level=0.95,
                              alternative='two-sided'):
    """
    Conditional exact confidence interval for the odds ratio.
    """
    if alternative == 'two-sided':
        alpha = 0.5*(1 - confidence_level)
        lower = _ci_lower(table, alpha)
        upper = _ci_upper(table, alpha)
    elif alternative == 'less':
        lower = 0.0
        upper = _ci_upper(table, 1 - confidence_level)
    else:
        # alternative == 'greater'
        lower = _ci_lower(table, 1 - confidence_level)
        upper = np.inf

    return lower, upper


def _sample_odds_ratio_ci(table, confidence_level=0.95,
                          alternative='two-sided'):
    oddsratio = _sample_odds_ratio(table)
    log_or = np.log(oddsratio)
    se = np.sqrt((1/table).sum())
    if alternative == 'less':
        z = ndtri(confidence_level)
        loglow = -np.inf
        loghigh = log_or + z*se
    elif alternative == 'greater':
        z = ndtri(confidence_level)
        loglow = log_or - z*se
        loghigh = np.inf
    else:
        # alternative is 'two-sided'
        z = ndtri(0.5*confidence_level + 0.5)
        loglow = log_or - z*se
        loghigh = log_or + z*se

    return np.exp(loglow), np.exp(loghigh)


class OddsRatioResult:
    """
    Result of `scipy.stats.contingency.odds_ratio`.  See the
    docstring for `odds_ratio` for more details.

    Attributes
    ----------
    statistic : float
        The computed odds ratio.

        * If `kind` is ``'sample'``, this is sample (or unconditional)
          estimate, given by
          ``table[0, 0]*table[1, 1]/(table[0, 1]*table[1, 0])``.
        * If `kind` is ``'conditional'``, this is the conditional
          maximum likelihood estimate for the odds ratio. It is
          the noncentrality parameter of Fisher's noncentral
          hypergeometric distribution with the same hypergeometric
          parameters as `table` and whose mean is ``table[0, 0]``.

    Methods
    -------
    confidence_interval :
        Confidence interval for the odds ratio.
    """

    def __init__(self, _table, _kind, statistic):
        # for now, no need to make _table and _kind public, since this sort of
        # information is returned in very few `scipy.stats` results
        self._table = _table
        self._kind = _kind
        self.statistic = statistic

    def __repr__(self):
        return f"OddsRatioResult(statistic={self.statistic})"

    def confidence_interval(self, confidence_level=0.95,
                            alternative='two-sided'):
        """
        Confidence interval for the odds ratio.

        Parameters
        ----------
        confidence_level: float
            Desired confidence level for the confidence interval.
            The value must be given as a fraction between 0 and 1.
            Default is 0.95 (meaning 95%).

        alternative : {'two-sided', 'less', 'greater'}, optional
            The alternative hypothesis of the hypothesis test to which the
            confidence interval corresponds. That is, suppose the null
            hypothesis is that the true odds ratio equals ``OR`` and the
            confidence interval is ``(low, high)``. Then the following options
            for `alternative` are available (default is 'two-sided'):

            * 'two-sided': the true odds ratio is not equal to ``OR``. There
              is evidence against the null hypothesis at the chosen
              `confidence_level` if ``high < OR`` or ``low > OR``.
            * 'less': the true odds ratio is less than ``OR``. The ``low`` end
              of the confidence interval is 0, and there is evidence against
              the null hypothesis at  the chosen `confidence_level` if
              ``high < OR``.
            * 'greater': the true odds ratio is greater than ``OR``.  The
              ``high`` end of the confidence interval is ``np.inf``, and there
              is evidence against the null hypothesis at the chosen
              `confidence_level` if ``low > OR``.

        Returns
        -------
        ci : ``ConfidenceInterval`` instance
            The confidence interval, represented as an object with
            attributes ``low`` and ``high``.

        Notes
        -----
        When `kind` is ``'conditional'``, the limits of the confidence
        interval are the conditional "exact confidence limits" as described
        by Fisher [1]_. The conditional odds ratio and confidence interval are
        also discussed in Section 4.1.2 of the text by Sahai and Khurshid [2]_.

        When `kind` is ``'sample'``, the confidence interval is computed
        under the assumption that the logarithm of the odds ratio is normally
        distributed with standard error given by::

            se = sqrt(1/a + 1/b + 1/c + 1/d)

        where ``a``, ``b``, ``c`` and ``d`` are the elements of the
        contingency table.  (See, for example, [2]_, section 3.1.3.2,
        or [3]_, section 2.3.3).

        References
        ----------
        .. [1] R. A. Fisher (1935), The logic of inductive inference,
               Journal of the Royal Statistical Society, Vol. 98, No. 1,
               pp. 39-82.
        .. [2] H. Sahai and A. Khurshid (1996), Statistics in Epidemiology:
               Methods, Techniques, and Applications, CRC Press LLC, Boca
               Raton, Florida.
        .. [3] Alan Agresti, An Introduction to Categorical Data Analysis
               (second edition), Wiley, Hoboken, NJ, USA (2007).
        """
        if alternative not in ['two-sided', 'less', 'greater']:
            raise ValueError("`alternative` must be 'two-sided', 'less' or "
                             "'greater'.")

        if confidence_level < 0 or confidence_level > 1:
            raise ValueError('confidence_level must be between 0 and 1')

        if self._kind == 'conditional':
            ci = self._conditional_odds_ratio_ci(confidence_level, alternative)
        else:
            ci = self._sample_odds_ratio_ci(confidence_level, alternative)
        return ci

    def _conditional_odds_ratio_ci(self, confidence_level=0.95,
                                   alternative='two-sided'):
        """
        Confidence interval for the conditional odds ratio.
        """

        table = self._table
        if 0 in table.sum(axis=0) or 0 in table.sum(axis=1):
            # If both values in a row or column are zero, the p-value is 1,
            # the odds ratio is NaN and the confidence interval is (0, inf).
            ci = (0, np.inf)
        else:
            ci = _conditional_oddsratio_ci(table,
                                           confidence_level=confidence_level,
                                           alternative=alternative)
        return ConfidenceInterval(low=ci[0], high=ci[1])

    def _sample_odds_ratio_ci(self, confidence_level=0.95,
                              alternative='two-sided'):
        """
        Confidence interval for the sample odds ratio.
        """
        if confidence_level < 0 or confidence_level > 1:
            raise ValueError('confidence_level must be between 0 and 1')

        table = self._table
        if 0 in table.sum(axis=0) or 0 in table.sum(axis=1):
            # If both values in a row or column are zero, the p-value is 1,
            # the odds ratio is NaN and the confidence interval is (0, inf).
            ci = (0, np.inf)
        else:
            ci = _sample_odds_ratio_ci(table,
                                       confidence_level=confidence_level,
                                       alternative=alternative)
        return ConfidenceInterval(low=ci[0], high=ci[1])


def odds_ratio(table, *, kind='conditional'):
    r"""
    Compute the odds ratio for a 2x2 contingency table.

    Parameters
    ----------
    table : array_like of ints
        A 2x2 contingency table.  Elements must be non-negative integers.
    kind : str, optional
        Which kind of odds ratio to compute, either the sample
        odds ratio (``kind='sample'``) or the conditional odds ratio
        (``kind='conditional'``).  Default is ``'conditional'``.

    Returns
    -------
    result : `~scipy.stats._result_classes.OddsRatioResult` instance
        The returned object has two computed attributes:

        statistic : float
            * If `kind` is ``'sample'``, this is sample (or unconditional)
              estimate, given by
              ``table[0, 0]*table[1, 1]/(table[0, 1]*table[1, 0])``.
            * If `kind` is ``'conditional'``, this is the conditional
              maximum likelihood estimate for the odds ratio. It is
              the noncentrality parameter of Fisher's noncentral
              hypergeometric distribution with the same hypergeometric
              parameters as `table` and whose mean is ``table[0, 0]``.

        The object has the method `confidence_interval` that computes
        the confidence interval of the odds ratio.

    See Also
    --------
    scipy.stats.fisher_exact
    relative_risk
    :ref:`hypothesis_odds_ratio` : Extended example

    Notes
    -----
    The conditional odds ratio was discussed by Fisher (see "Example 1"
    of [1]_).  Texts that cover the odds ratio include [2]_ and [3]_.

    .. versionadded:: 1.10.0

    References
    ----------
    .. [1] R. A. Fisher (1935), The logic of inductive inference,
           Journal of the Royal Statistical Society, Vol. 98, No. 1,
           pp. 39-82.
    .. [2] Breslow NE, Day NE (1980). Statistical methods in cancer research.
           Volume I - The analysis of case-control studies. IARC Sci Publ.
           (32):5-338. PMID: 7216345. (See section 4.2.)
    .. [3] H. Sahai and A. Khurshid (1996), Statistics in Epidemiology:
           Methods, Techniques, and Applications, CRC Press LLC, Boca
           Raton, Florida.

    Examples
    --------
    In epidemiology, individuals are classified as "exposed" or
    "unexposed" to some factor or treatment. If the occurrence of some
    illness is under study, those who have the illness are often
    classified as "cases", and those without it are "noncases".  The
    counts of the occurrences of these classes gives a contingency
    table::

                    exposed    unexposed
        cases          a           b
        noncases       c           d

    The sample odds ratio may be written ``(a/c) / (b/d)``.  ``a/c`` can
    be interpreted as the odds of a case occurring in the exposed group,
    and ``b/d`` as the odds of a case occurring in the unexposed group.
    The sample odds ratio is the ratio of these odds.  If the odds ratio
    is greater than 1, it suggests that there is a positive association
    between being exposed and being a case.

    Interchanging the rows or columns of the contingency table inverts
    the odds ratio, so it is important to understand the meaning of labels
    given to the rows and columns of the table when interpreting the
    odds ratio.

    Consider a hypothetical example where it is hypothesized that exposure to a
    certain chemical is associated with increased occurrence of a certain
    disease. Suppose we have the following table for a collection of 410 people::

                exposed unexposed
        cases        7       15
        noncases    58      472

    The question we ask is "Is exposure to the chemical associated with
    increased risk of the disease?"

    Compute the odds ratio:

    >>> from scipy.stats.contingency import odds_ratio
    >>> res = odds_ratio([[7, 15], [58, 472]])
    >>> res.statistic
    3.7836687705553493

    For this sample, the odds of getting the disease for those who have been
    exposed to the chemical are almost 3.8 times that of those who have not been
    exposed.

    We can compute the 95% confidence interval for the odds ratio:

    >>> res.confidence_interval(confidence_level=0.95)
    ConfidenceInterval(low=1.2514829132266785, high=10.363493716701269)

    The 95% confidence interval for the conditional odds ratio is approximately
    (1.25, 10.4).

    For a more detailed example, see :ref:`hypothesis_odds_ratio`.
    """
    if kind not in ['conditional', 'sample']:
        raise ValueError("`kind` must be 'conditional' or 'sample'.")

    c = np.asarray(table)

    if c.shape != (2, 2):
        raise ValueError(f"Invalid shape {c.shape}. The input `table` must be "
                         "of shape (2, 2).")

    if not np.issubdtype(c.dtype, np.integer):
        raise ValueError("`table` must be an array of integers, but got "
                         f"type {c.dtype}")
    c = c.astype(np.int64)

    if np.any(c < 0):
        raise ValueError("All values in `table` must be nonnegative.")

    if 0 in c.sum(axis=0) or 0 in c.sum(axis=1):
        # If both values in a row or column are zero, the p-value is NaN and
        # the odds ratio is NaN.
        result = OddsRatioResult(_table=c, _kind=kind, statistic=np.nan)
        return result

    if kind == 'sample':
        oddsratio = _sample_odds_ratio(c)
    else:  # kind is 'conditional'
        oddsratio = _conditional_oddsratio(c)

    result = OddsRatioResult(_table=c, _kind=kind, statistic=oddsratio)
    return result
<|endoftext|>from collections import namedtuple
from dataclasses import dataclass
from math import comb
import numpy as np
import warnings
from itertools import combinations
import scipy.stats
from scipy.optimize import shgo
from . import distributions
from ._common import ConfidenceInterval
from ._continuous_distns import norm
from scipy.special import gamma, kv, gammaln
from scipy.fft import ifft
from ._stats_pythran import _a_ij_Aij_Dij2
from ._stats_pythran import (
    _concordant_pairs as _P, _discordant_pairs as _Q
)
from ._axis_nan_policy import _axis_nan_policy_factory
from scipy.stats import _stats_py

__all__ = ['epps_singleton_2samp', 'cramervonmises', 'somersd',
           'barnard_exact', 'boschloo_exact', 'cramervonmises_2samp',
           'tukey_hsd', 'poisson_means_test']

Epps_Singleton_2sampResult = namedtuple('Epps_Singleton_2sampResult',
                                        ('statistic', 'pvalue'))


@_axis_nan_policy_factory(Epps_Singleton_2sampResult, n_samples=2, too_small=4)
def epps_singleton_2samp(x, y, t=(0.4, 0.8)):
    """Compute the Epps-Singleton (ES) test statistic.

    Test the null hypothesis that two samples have the same underlying
    probability distribution.

    Parameters
    ----------
    x, y : array-like
        The two samples of observations to be tested. Input must not have more
        than one dimension. Samples can have different lengths, but both
        must have at least five observations.
    t : array-like, optional
        The points (t1, ..., tn) where the empirical characteristic function is
        to be evaluated. It should be positive distinct numbers. The default
        value (0.4, 0.8) is proposed in [1]_. Input must not have more than
        one dimension.

    Returns
    -------
    statistic : float
        The test statistic.
    pvalue : float
        The associated p-value based on the asymptotic chi2-distribution.

    See Also
    --------
    ks_2samp, anderson_ksamp

    Notes
    -----
    Testing whether two samples are generated by the same underlying
    distribution is a classical question in statistics. A widely used test is
    the Kolmogorov-Smirnov (KS) test which relies on the empirical
    distribution function. Epps and Singleton introduce a test based on the
    empirical characteristic function in [1]_.

    One advantage of the ES test compared to the KS test is that is does
    not assume a continuous distribution. In [1]_, the authors conclude
    that the test also has a higher power than the KS test in many
    examples. They recommend the use of the ES test for discrete samples as
    well as continuous samples with at least 25 observations each, whereas
    `anderson_ksamp` is recommended for smaller sample sizes in the
    continuous case.

    The p-value is computed from the asymptotic distribution of the test
    statistic which follows a `chi2` distribution. If the sample size of both
    `x` and `y` is below 25, the small sample correction proposed in [1]_ is
    applied to the test statistic.

    The default values of `t` are determined in [1]_ by considering
    various distributions and finding good values that lead to a high power
    of the test in general. Table III in [1]_ gives the optimal values for
    the distributions tested in that study. The values of `t` are scaled by
    the semi-interquartile range in the implementation, see [1]_.

    References
    ----------
    .. [1] T. W. Epps and K. J. Singleton, "An omnibus test for the two-sample
       problem using the empirical characteristic function", Journal of
       Statistical Computation and Simulation 26, p. 177--203, 1986.

    .. [2] S. J. Goerg and J. Kaiser, "Nonparametric testing of distributions
       - the Epps-Singleton two-sample test using the empirical characteristic
       function", The Stata Journal 9(3), p. 454--465, 2009.

    """
    # x and y are converted to arrays by the decorator
    t = np.asarray(t)
    # check if x and y are valid inputs
    nx, ny = len(x), len(y)
    if (nx < 5) or (ny < 5):
        raise ValueError('x and y should have at least 5 elements, but len(x) '
                         f'= {nx} and len(y) = {ny}.')
    if not np.isfinite(x).all():
        raise ValueError('x must not contain nonfinite values.')
    if not np.isfinite(y).all():
        raise ValueError('y must not contain nonfinite values.')
    n = nx + ny

    # check if t is valid
    if t.ndim > 1:
        raise ValueError(f't must be 1d, but t.ndim equals {t.ndim}.')
    if np.less_equal(t, 0).any():
        raise ValueError('t must contain positive elements only.')

    # rescale t with semi-iqr as proposed in [1]; import iqr here to avoid
    # circular import
    from scipy.stats import iqr
    sigma = iqr(np.hstack((x, y))) / 2
    ts = np.reshape(t, (-1, 1)) / sigma

    # covariance estimation of ES test
    gx = np.vstack((np.cos(ts*x), np.sin(ts*x))).T  # shape = (nx, 2*len(t))
    gy = np.vstack((np.cos(ts*y), np.sin(ts*y))).T
    cov_x = np.cov(gx.T, bias=True)  # the test uses biased cov-estimate
    cov_y = np.cov(gy.T, bias=True)
    est_cov = (n/nx)*cov_x + (n/ny)*cov_y
    est_cov_inv = np.linalg.pinv(est_cov)
    r = np.linalg.matrix_rank(est_cov_inv)
    if r < 2*len(t):
        warnings.warn('Estimated covariance matrix does not have full rank. '
                      'This indicates a bad choice of the input t and the '
                      'test might not be consistent.', # see p. 183 in [1]_
                      stacklevel=2)

    # compute test statistic w distributed asympt. as chisquare with df=r
    g_diff = np.mean(gx, axis=0) - np.mean(gy, axis=0)
    w = n*np.dot(g_diff.T, np.dot(est_cov_inv, g_diff))

    # apply small-sample correction
    if (max(nx, ny) < 25):
        corr = 1.0/(1.0 + n**(-0.45) + 10.1*(nx**(-1.7) + ny**(-1.7)))
        w = corr * w

    chi2 = _stats_py._SimpleChi2(r)
    p = _stats_py._get_pvalue(w, chi2, alternative='greater', symmetric=False, xp=np)

    return Epps_Singleton_2sampResult(w, p)


def poisson_means_test(k1, n1, k2, n2, *, diff=0, alternative='two-sided'):
    r"""
    Performs the Poisson means test, AKA the "E-test".

    This is a test of the null hypothesis that the difference between means of
    two Poisson distributions is `diff`. The samples are provided as the
    number of events `k1` and `k2` observed within measurement intervals
    (e.g. of time, space, number of observations) of sizes `n1` and `n2`.

    Parameters
    ----------
    k1 : int
        Number of events observed from distribution 1.
    n1: float
        Size of sample from distribution 1.
    k2 : int
        Number of events observed from distribution 2.
    n2 : float
        Size of sample from distribution 2.
    diff : float, default=0
        The hypothesized difference in means between the distributions
        underlying the samples.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

          * 'two-sided': the difference between distribution means is not
            equal to `diff`
          * 'less': the difference between distribution means is less than
            `diff`
          * 'greater': the difference between distribution means is greater
            than `diff`

    Returns
    -------
    statistic : float
        The test statistic (see [1]_ equation 3.3).
    pvalue : float
        The probability of achieving such an extreme value of the test
        statistic under the null hypothesis.

    Notes
    -----

    Let:

    .. math:: X_1 \sim \mbox{Poisson}(\mathtt{n1}\lambda_1)

    be a random variable independent of

    .. math:: X_2  \sim \mbox{Poisson}(\mathtt{n2}\lambda_2)

    and let ``k1`` and ``k2`` be the observed values of :math:`X_1`
    and :math:`X_2`, respectively. Then `poisson_means_test` uses the number
    of observed events ``k1`` and ``k2`` from samples of size ``n1`` and
    ``n2``, respectively, to test the null hypothesis that

    .. math::
       H_0: \lambda_1 - \lambda_2 = \mathtt{diff}

    A benefit of the E-test is that it has good power for small sample sizes,
    which can reduce sampling costs [1]_. It has been evaluated and determined
    to be more powerful than the comparable C-test, sometimes referred to as
    the Poisson exact test.

    References
    ----------
    .. [1]  Krishnamoorthy, K., & Thomson, J. (2004). A more powerful test for
       comparing two Poisson means. Journal of Statistical Planning and
       Inference, 119(1), 23-35.

    .. [2]  Przyborowski, J., & Wilenski, H. (1940). Homogeneity of results in
       testing samples from Poisson series: With an application to testing
       clover seed for dodder. Biometrika, 31(3/4), 313-323.

    Examples
    --------

    Suppose that a gardener wishes to test the number of dodder (weed) seeds
    in a sack of clover seeds that they buy from a seed company. It has
    previously been established that the number of dodder seeds in clover
    follows the Poisson distribution.

    A 100 gram sample is drawn from the sack before being shipped to the
    gardener. The sample is analyzed, and it is found to contain no dodder
    seeds; that is, `k1` is 0. However, upon arrival, the gardener draws
    another 100 gram sample from the sack. This time, three dodder seeds are
    found in the sample; that is, `k2` is 3. The gardener would like to
    know if the difference is significant and not due to chance. The
    null hypothesis is that the difference between the two samples is merely
    due to chance, or that :math:`\lambda_1 - \lambda_2 = \mathtt{diff}`
    where :math:`\mathtt{diff} = 0`. The alternative hypothesis is that the
    difference is not due to chance, or :math:`\lambda_1 - \lambda_2 \ne 0`.
    The gardener selects a significance level of 5% to reject the null
    hypothesis in favor of the alternative [2]_.

    >>> import scipy.stats as stats
    >>> res = stats.poisson_means_test(0, 100, 3, 100)
    >>> res.statistic, res.pvalue
    (-1.7320508075688772, 0.08837900929018157)

    The p-value is .088, indicating a near 9% chance of observing a value of
    the test statistic under the null hypothesis. This exceeds 5%, so the
    gardener does not reject the null hypothesis as the difference cannot be
    regarded as significant at this level.
    """

    _poisson_means_test_iv(k1, n1, k2, n2, diff, alternative)

    # "for a given k_1 and k_2, an estimate of \lambda_2 is given by" [1] (3.4)
    lmbd_hat2 = ((k1 + k2) / (n1 + n2) - diff * n1 / (n1 + n2))

    # "\hat{\lambda_{2k}} may be less than or equal to zero ... and in this
    # case the null hypothesis cannot be rejected ... [and] it is not necessary
    # to compute the p-value". [1] page 26 below eq. (3.6).
    if lmbd_hat2 <= 0:
        return _stats_py.SignificanceResult(0, 1)

    # The unbiased variance estimate [1] (3.2)
    var = k1 / (n1 ** 2) + k2 / (n2 ** 2)

    # The _observed_ pivot statistic from the input. It follows the
    # unnumbered equation following equation (3.3) This is used later in
    # comparison with the computed pivot statistics in an indicator function.
    t_k1k2 = (k1 / n1 - k2 / n2 - diff) / np.sqrt(var)

    # Equation (3.5) of [1] is lengthy, so it is broken into several parts,
    # beginning here. Note that the probability mass function of poisson is
    # exp^(-\mu)*\mu^k/k!, so and this is called with shape \mu, here noted
    # here as nlmbd_hat*. The strategy for evaluating the double summation in
    # (3.5) is to create two arrays of the values of the two products inside
    # the summation and then broadcast them together into a matrix, and then
    # sum across the entire matrix.

    # Compute constants (as seen in the first and second separated products in
    # (3.5).). (This is the shape (\mu) parameter of the poisson distribution.)
    nlmbd_hat1 = n1 * (lmbd_hat2 + diff)
    nlmbd_hat2 = n2 * lmbd_hat2

    # Determine summation bounds for tail ends of distribution rather than
    # summing to infinity. `x1*` is for the outer sum and `x2*` is the inner
    # sum.
    x1_lb, x1_ub = distributions.poisson.ppf([1e-10, 1 - 1e-16], nlmbd_hat1)
    x2_lb, x2_ub = distributions.poisson.ppf([1e-10, 1 - 1e-16], nlmbd_hat2)

    # Construct arrays to function as the x_1 and x_2 counters on the summation
    # in (3.5). `x1` is in columns and `x2` is in rows to allow for
    # broadcasting.
    x1 = np.arange(x1_lb, x1_ub + 1)
    x2 = np.arange(x2_lb, x2_ub + 1)[:, None]

    # These are the two products in equation (3.5) with `prob_x1` being the
    # first (left side) and `prob_x2` being the second (right side). (To
    # make as clear as possible: the 1st contains a "+ d" term, the 2nd does
    # not.)
    prob_x1 = distributions.poisson.pmf(x1, nlmbd_hat1)
    prob_x2 = distributions.poisson.pmf(x2, nlmbd_hat2)

    # compute constants for use in the "pivot statistic" per the
    # unnumbered equation following (3.3).
    lmbd_x1 = x1 / n1
    lmbd_x2 = x2 / n2
    lmbds_diff = lmbd_x1 - lmbd_x2 - diff
    var_x1x2 = lmbd_x1 / n1 + lmbd_x2 / n2

    # This is the 'pivot statistic' for use in the indicator of the summation
    # (left side of "I[.]").
    with np.errstate(invalid='ignore', divide='ignore'):
        t_x1x2 = lmbds_diff / np.sqrt(var_x1x2)

    # `[indicator]` implements the "I[.] ... the indicator function" per
    # the paragraph following equation (3.5).
    if alternative == 'two-sided':
        indicator = np.abs(t_x1x2) >= np.abs(t_k1k2)
    elif alternative == 'less':
        indicator = t_x1x2 <= t_k1k2
    else:
        indicator = t_x1x2 >= t_k1k2

    # Multiply all combinations of the products together, exclude terms
    # based on the `indicator` and then sum. (3.5)
    pvalue = np.sum((prob_x1 * prob_x2)[indicator])
    return _stats_py.SignificanceResult(t_k1k2, pvalue)


def _poisson_means_test_iv(k1, n1, k2, n2, diff, alternative):
    # """check for valid types and values of input to `poisson_mean_test`."""
    if k1 != int(k1) or k2 != int(k2):
        raise TypeError('`k1` and `k2` must be integers.')

    count_err = '`k1` and `k2` must be greater than or equal to 0.'
    if k1 < 0 or k2 < 0:
        raise ValueError(count_err)

    if n1 <= 0 or n2 <= 0:
        raise ValueError('`n1` and `n2` must be greater than 0.')

    if diff < 0:
        raise ValueError('diff must be greater than or equal to 0.')

    alternatives = {'two-sided', 'less', 'greater'}
    if alternative.lower() not in alternatives:
        raise ValueError(f"Alternative must be one of '{alternatives}'.")


class CramerVonMisesResult:
    def __init__(self, statistic, pvalue):
        self.statistic = statistic
        self.pvalue = pvalue

    def __repr__(self):
        return (f"{self.__class__.__name__}(statistic={self.statistic}, "
                f"pvalue={self.pvalue})")


def _psi1_mod(x):
    """
    psi1 is defined in equation 1.10 in Csörgő, S. and Faraway, J. (1996).
    This implements a modified version by excluding the term V(x) / 12
    (here: _cdf_cvm_inf(x) / 12) to avoid evaluating _cdf_cvm_inf(x)
    twice in _cdf_cvm.

    Implementation based on MAPLE code of Julian Faraway and R code of the
    function pCvM in the package goftest (v1.1.1), permission granted
    by Adrian Baddeley. Main difference in the implementation: the code
    here keeps adding terms of the series until the terms are small enough.
    """

    def _ed2(y):
        z = y**2 / 4
        b = kv(1/4, z) + kv(3/4, z)
        return np.exp(-z) * (y/2)**(3/2) * b / np.sqrt(np.pi)

    def _ed3(y):
        z = y**2 / 4
        c = np.exp(-z) / np.sqrt(np.pi)
        return c * (y/2)**(5/2) * (2*kv(1/4, z) + 3*kv(3/4, z) - kv(5/4, z))

    def _Ak(k, x):
        m = 2*k + 1
        sx = 2 * np.sqrt(x)
        y1 = x**(3/4)
        y2 = x**(5/4)

        e1 = m * gamma(k + 1/2) * _ed2((4 * k + 3)/sx) / (9 * y1)
        e2 = gamma(k + 1/2) * _ed3((4 * k + 1) / sx) / (72 * y2)
        e3 = 2 * (m + 2) * gamma(k + 3/2) * _ed3((4 * k + 5) / sx) / (12 * y2)
        e4 = 7 * m * gamma(k + 1/2) * _ed2((4 * k + 1) / sx) / (144 * y1)
        e5 = 7 * m * gamma(k + 1/2) * _ed2((4 * k + 5) / sx) / (144 * y1)

        return e1 + e2 + e3 + e4 + e5

    x = np.asarray(x)
    tot = np.zeros_like(x, dtype='float')
    cond = np.ones_like(x, dtype='bool')
    k = 0
    while np.any(cond):
        z = -_Ak(k, x[cond]) / (np.pi * gamma(k + 1))
        tot[cond] = tot[cond] + z
        cond[cond] = np.abs(z) >= 1e-7
        k += 1

    return tot


def _cdf_cvm_inf(x):
    """
    Calculate the cdf of the Cramér-von Mises statistic (infinite sample size).

    See equation 1.2 in Csörgő, S. and Faraway, J. (1996).

    Implementation based on MAPLE code of Julian Faraway and R code of the
    function pCvM in the package goftest (v1.1.1), permission granted
    by Adrian Baddeley. Main difference in the implementation: the code
    here keeps adding terms of the series until the terms are small enough.

    The function is not expected to be accurate for large values of x, say
    x > 4, when the cdf is very close to 1.
    """
    x = np.asarray(x)

    def term(x, k):
        # this expression can be found in [2], second line of (1.3)
        u = np.exp(gammaln(k + 0.5) - gammaln(k+1)) / (np.pi**1.5 * np.sqrt(x))
        y = 4*k + 1
        q = y**2 / (16*x)
        b = kv(0.25, q)
        return u * np.sqrt(y) * np.exp(-q) * b

    tot = np.zeros_like(x, dtype='float')
    cond = np.ones_like(x, dtype='bool')
    k = 0
    while np.any(cond):
        z = term(x[cond], k)
        tot[cond] = tot[cond] + z
        cond[cond] = np.abs(z) >= 1e-7
        k += 1

    return tot


def _cdf_cvm(x, n=None):
    """
    Calculate the cdf of the Cramér-von Mises statistic for a finite sample
    size n. If N is None, use the asymptotic cdf (n=inf).

    See equation 1.8 in Csörgő, S. and Faraway, J. (1996) for finite samples,
    1.2 for the asymptotic cdf.

    The function is not expected to be accurate for large values of x, say
    x > 2, when the cdf is very close to 1 and it might return values > 1
    in that case, e.g. _cdf_cvm(2.0, 12) = 1.0000027556716846. Moreover, it
    is not accurate for small values of n, especially close to the bounds of
    the distribution's domain, [1/(12*n), n/3], where the value jumps to 0
    and 1, respectively. These are limitations of the approximation by Csörgő
    and Faraway (1996) implemented in this function.
    """
    x = np.asarray(x)
    if n is None:
        y = _cdf_cvm_inf(x)
    else:
        # support of the test statistic is [12/n, n/3], see 1.1 in [2]
        y = np.zeros_like(x, dtype='float')
        sup = (1./(12*n) < x) & (x < n/3.)
        # note: _psi1_mod does not include the term _cdf_cvm_inf(x) / 12
        # therefore, we need to add it here
        y[sup] = _cdf_cvm_inf(x[sup]) * (1 + 1./(12*n)) + _psi1_mod(x[sup]) / n
        y[x >= n/3] = 1

    if y.ndim == 0:
        return y[()]
    return y


def _cvm_result_to_tuple(res):
    return res.statistic, res.pvalue


@_axis_nan_policy_factory(CramerVonMisesResult, n_samples=1, too_small=1,
                          result_to_tuple=_cvm_result_to_tuple)
def cramervonmises(rvs, cdf, args=()):
    """Perform the one-sample Cramér-von Mises test for goodness of fit.

    This performs a test of the goodness of fit of a cumulative distribution
    function (cdf) :math:`F` compared to the empirical distribution function
    :math:`F_n` of observed random variates :math:`X_1, ..., X_n` that are
    assumed to be independent and identically distributed ([1]_).
    The null hypothesis is that the :math:`X_i` have cumulative distribution
    :math:`F`.

    Parameters
    ----------
    rvs : array_like
        A 1-D array of observed values of the random variables :math:`X_i`.
        The sample must contain at least two observations.
    cdf : str or callable
        The cumulative distribution function :math:`F` to test the
        observations against. If a string, it should be the name of a
        distribution in `scipy.stats`. If a callable, that callable is used
        to calculate the cdf: ``cdf(x, *args) -> float``.
    args : tuple, optional
        Distribution parameters. These are assumed to be known; see Notes.

    Returns
    -------
    res : object with attributes
        statistic : float
            Cramér-von Mises statistic.
        pvalue : float
            The p-value.

    See Also
    --------
    kstest, cramervonmises_2samp

    Notes
    -----
    .. versionadded:: 1.6.0

    The p-value relies on the approximation given by equation 1.8 in [2]_.
    It is important to keep in mind that the p-value is only accurate if
    one tests a simple hypothesis, i.e. the parameters of the reference
    distribution are known. If the parameters are estimated from the data
    (composite hypothesis), the computed p-value is not reliable.

    References
    ----------
    .. [1] Cramér-von Mises criterion, Wikipedia,
           https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93von_Mises_criterion
    .. [2] Csörgő, S. and Faraway, J. (1996). The Exact and Asymptotic
           Distribution of Cramér-von Mises Statistics. Journal of the
           Royal Statistical Society, pp. 221-234.

    Examples
    --------

    Suppose we wish to test whether data generated by ``scipy.stats.norm.rvs``
    were, in fact, drawn from the standard normal distribution. We choose a
    significance level of ``alpha=0.05``.

    >>> import numpy as np
    >>> from scipy import stats
    >>> rng = np.random.default_rng(165417232101553420507139617764912913465)
    >>> x = stats.norm.rvs(size=500, random_state=rng)
    >>> res = stats.cramervonmises(x, 'norm')
    >>> res.statistic, res.pvalue
    (0.1072085112565724, 0.5508482238203407)

    The p-value exceeds our chosen significance level, so we do not
    reject the null hypothesis that the observed sample is drawn from the
    standard normal distribution.

    Now suppose we wish to check whether the same samples shifted by 2.1 is
    consistent with being drawn from a normal distribution with a mean of 2.

    >>> y = x + 2.1
    >>> res = stats.cramervonmises(y, 'norm', args=(2,))
    >>> res.statistic, res.pvalue
    (0.8364446265294695, 0.00596286797008283)

    Here we have used the `args` keyword to specify the mean (``loc``)
    of the normal distribution to test the data against. This is equivalent
    to the following, in which we create a frozen normal distribution with
    mean 2.1, then pass its ``cdf`` method as an argument.

    >>> frozen_dist = stats.norm(loc=2)
    >>> res = stats.cramervonmises(y, frozen_dist.cdf)
    >>> res.statistic, res.pvalue
    (0.8364446265294695, 0.00596286797008283)

    In either case, we would reject the null hypothesis that the observed
    sample is drawn from a normal distribution with a mean of 2 (and default
    variance of 1) because the p-value is less than our chosen
    significance level.

    """
    if isinstance(cdf, str):
        cdf = getattr(distributions, cdf).cdf

    vals = np.sort(np.asarray(rvs))

    if vals.size <= 1:
        raise ValueError('The sample must contain at least two observations.')

    n = len(vals)
    cdfvals = cdf(vals, *args)

    u = (2*np.arange(1, n+1) - 1)/(2*n)
    w = 1/(12*n) + np.sum((u - cdfvals)**2)

    # avoid small negative values that can occur due to the approximation
    p = np.clip(1. - _cdf_cvm(w, n), 0., None)

    return CramerVonMisesResult(statistic=w, pvalue=p)


def _get_wilcoxon_distr(n):
    """
    Distribution of probability of the Wilcoxon ranksum statistic r_plus (sum
    of ranks of positive differences).
    Returns an array with the probabilities of all the possible ranks
    r = 0, ..., n*(n+1)/2
    """
    c = np.ones(1, dtype=np.float64)
    for k in range(1, n + 1):
        prev_c = c
        c = np.zeros(k * (k + 1) // 2 + 1, dtype=np.float64)
        m = len(prev_c)
        c[:m] = prev_c * 0.5
        c[-m:] += prev_c * 0.5
    return c


def _get_wilcoxon_distr2(n):
    """
    Distribution of probability of the Wilcoxon ranksum statistic r_plus (sum
    of ranks of positive differences).
    Returns an array with the probabilities of all the possible ranks
    r = 0, ..., n*(n+1)/2
    This is a slower reference function
    References
    ----------
    .. [1] 1. Harris T, Hardin JW. Exact Wilcoxon Signed-Rank and Wilcoxon
        Mann-Whitney Ranksum Tests. The Stata Journal. 2013;13(2):337-343.
    """
    ai = np.arange(1, n+1)[:, None]
    t = n*(n+1)/2
    q = 2*t
    j = np.arange(q)
    theta = 2*np.pi/q*j
    phi_sp = np.prod(np.cos(theta*ai), axis=0)
    phi_s = np.exp(1j*theta*t) * phi_sp
    p = np.real(ifft(phi_s))
    res = np.zeros(int(t)+1)
    res[:-1:] = p[::2]
    res[0] /= 2
    res[-1] = res[0]
    return res


def _tau_b(A):
    """Calculate Kendall's tau-b and p-value from contingency table."""
    # See [2] 2.2 and 4.2

    # contingency table must be truly 2D
    if A.shape[0] == 1 or A.shape[1] == 1:
        return np.nan, np.nan

    NA = A.sum()
    PA = _P(A)
    QA = _Q(A)
    Sri2 = (A.sum(axis=1)**2).sum()
    Scj2 = (A.sum(axis=0)**2).sum()
    denominator = (NA**2 - Sri2)*(NA**2 - Scj2)

    tau = (PA-QA)/(denominator)**0.5

    numerator = 4*(_a_ij_Aij_Dij2(A) - (PA - QA)**2 / NA)
    s02_tau_b = numerator/denominator
    if s02_tau_b == 0:  # Avoid divide by zero
        return tau, 0
    Z = tau/s02_tau_b**0.5
    p = 2*norm.sf(abs(Z))  # 2-sided p-value

    return tau, p


def _somers_d(A, alternative='two-sided'):
    """Calculate Somers' D and p-value from contingency table."""
    # See [3] page 1740

    # contingency table must be truly 2D
    if A.shape[0] <= 1 or A.shape[1] <= 1:
        return np.nan, np.nan

    NA = A.sum()
    NA2 = NA**2
    PA = _P(A)
    QA = _Q(A)
    Sri2 = (A.sum(axis=1)**2).sum()

    d = (PA - QA)/(NA2 - Sri2)

    S = _a_ij_Aij_Dij2(A) - (PA-QA)**2/NA

    with np.errstate(divide='ignore'):
        Z = (PA - QA)/(4*(S))**0.5

    norm = _stats_py._SimpleNormal()
    p = _stats_py._get_pvalue(Z, norm, alternative, xp=np)

    return d, p


@dataclass
class SomersDResult:
    statistic: float
    pvalue: float
    table: np.ndarray


def somersd(x, y=None, alternative='two-sided'):
    r"""Calculates Somers' D, an asymmetric measure of ordinal association.

    Like Kendall's :math:`\tau`, Somers' :math:`D` is a measure of the
    correspondence between two rankings. Both statistics consider the
    difference between the number of concordant and discordant pairs in two
    rankings :math:`X` and :math:`Y`, and both are normalized such that values
    close  to 1 indicate strong agreement and values close to -1 indicate
    strong disagreement. They differ in how they are normalized. To show the
    relationship, Somers' :math:`D` can be defined in terms of Kendall's
    :math:`\tau_a`:

    .. math::
        D(Y|X) = \frac{\tau_a(X, Y)}{\tau_a(X, X)}

    Suppose the first ranking :math:`X` has :math:`r` distinct ranks and the
    second ranking :math:`Y` has :math:`s` distinct ranks. These two lists of
    :math:`n` rankings can also be viewed as an :math:`r \times s` contingency
    table in which element :math:`i, j` is the number of rank pairs with rank
    :math:`i` in ranking :math:`X` and rank :math:`j` in ranking :math:`Y`.
    Accordingly, `somersd` also allows the input data to be supplied as a
    single, 2D contingency table instead of as two separate, 1D rankings.

    Note that the definition of Somers' :math:`D` is asymmetric: in general,
    :math:`D(Y|X) \neq D(X|Y)`. ``somersd(x, y)`` calculates Somers'
    :math:`D(Y|X)`: the "row" variable :math:`X` is treated as an independent
    variable, and the "column" variable :math:`Y` is dependent. For Somers'
    :math:`D(X|Y)`, swap the input lists or transpose the input table.

    Parameters
    ----------
    x : array_like
        1D array of rankings, treated as the (row) independent variable.
        Alternatively, a 2D contingency table.
    y : array_like, optional
        If `x` is a 1D array of rankings, `y` is a 1D array of rankings of the
        same length, treated as the (column) dependent variable.
        If `x` is 2D, `y` is ignored.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        The following options are available:
        * 'two-sided': the rank correlation is nonzero
        * 'less': the rank correlation is negative (less than zero)
        * 'greater':  the rank correlation is positive (greater than zero)

    Returns
    -------
    res : SomersDResult
        A `SomersDResult` object with the following fields:

            statistic : float
               The Somers' :math:`D` statistic.
            pvalue : float
               The p-value for a hypothesis test whose null
               hypothesis is an absence of association, :math:`D=0`.
               See notes for more information.
            table : 2D array
               The contingency table formed from rankings `x` and `y` (or the
               provided contingency table, if `x` is a 2D array)

    See Also
    --------
    kendalltau : Calculates Kendall's tau, another correlation measure.
    weightedtau : Computes a weighted version of Kendall's tau.
    spearmanr : Calculates a Spearman rank-order correlation coefficient.
    pearsonr : Calculates a Pearson correlation coefficient.

    Notes
    -----
    This function follows the contingency table approach of [2]_ and
    [3]_. *p*-values are computed based on an asymptotic approximation of
    the test statistic distribution under the null hypothesis :math:`D=0`.

    Theoretically, hypothesis tests based on Kendall's :math:`tau` and Somers'
    :math:`D` should be identical.
    However, the *p*-values returned by `kendalltau` are based
    on the null hypothesis of *independence* between :math:`X` and :math:`Y`
    (i.e. the population from which pairs in :math:`X` and :math:`Y` are
    sampled contains equal numbers of all possible pairs), which is more
    specific than the null hypothesis :math:`D=0` used here. If the null
    hypothesis of independence is desired, it is acceptable to use the
    *p*-value returned by `kendalltau` with the statistic returned by
    `somersd` and vice versa. For more information, see [2]_.

    Contingency tables are formatted according to the convention used by
    SAS and R: the first ranking supplied (``x``) is the "row" variable, and
    the second ranking supplied (``y``) is the "column" variable. This is
    opposite the convention of Somers' original paper [1]_.

    References
    ----------
    .. [1] Robert H. Somers, "A New Asymmetric Measure of Association for
           Ordinal Variables", *American Sociological Review*, Vol. 27, No. 6,
           pp. 799--811, 1962.

    .. [2] Morton B. Brown and Jacqueline K. Benedetti, "Sampling Behavior of
           Tests for Correlation in Two-Way Contingency Tables", *Journal of
           the American Statistical Association* Vol. 72, No. 358, pp.
           309--315, 1977.

    .. [3] SAS Institute, Inc., "The FREQ Procedure (Book Excerpt)",
           *SAS/STAT 9.2 User's Guide, Second Edition*, SAS Publishing, 2009.

    .. [4] Laerd Statistics, "Somers' d using SPSS Statistics", *SPSS
           Statistics Tutorials and Statistical Guides*,
           https://statistics.laerd.com/spss-tutorials/somers-d-using-spss-statistics.php,
           Accessed July 31, 2020.

    Examples
    --------
    We calculate Somers' D for the example given in [4]_, in which a hotel
    chain owner seeks to determine the association between hotel room
    cleanliness and customer satisfaction. The independent variable, hotel
    room cleanliness, is ranked on an ordinal scale: "below average (1)",
    "average (2)", or "above average (3)". The dependent variable, customer
    satisfaction, is ranked on a second scale: "very dissatisfied (1)",
    "moderately dissatisfied (2)", "neither dissatisfied nor satisfied (3)",
    "moderately satisfied (4)", or "very satisfied (5)". 189 customers
    respond to the survey, and the results are cast into a contingency table
    with the hotel room cleanliness as the "row" variable and customer
    satisfaction as the "column" variable.

    +-----+-----+-----+-----+-----+-----+
    |     | (1) | (2) | (3) | (4) | (5) |
    +=====+=====+=====+=====+=====+=====+
    | (1) | 27  | 25  | 14  | 7   | 0   |
    +-----+-----+-----+-----+-----+-----+
    | (2) | 7   | 14  | 18  | 35  | 12  |
    +-----+-----+-----+-----+-----+-----+
    | (3) | 1   | 3   | 2   | 7   | 17  |
    +-----+-----+-----+-----+-----+-----+

    For example, 27 customers assigned their room a cleanliness ranking of
    "below average (1)" and a corresponding satisfaction of "very
    dissatisfied (1)". We perform the analysis as follows.

    >>> from scipy.stats import somersd
    >>> table = [[27, 25, 14, 7, 0], [7, 14, 18, 35, 12], [1, 3, 2, 7, 17]]
    >>> res = somersd(table)
    >>> res.statistic
    0.6032766111513396
    >>> res.pvalue
    1.0007091191074533e-27

    The value of the Somers' D statistic is approximately 0.6, indicating
    a positive correlation between room cleanliness and customer satisfaction
    in the sample.
    The *p*-value is very small, indicating a very small probability of
    observing such an extreme value of the statistic under the null
    hypothesis that the statistic of the entire population (from which
    our sample of 189 customers is drawn) is zero. This supports the
    alternative hypothesis that the true value of Somers' D for the population
    is nonzero.

    """
    x, y = np.array(x), np.array(y)
    if x.ndim == 1:
        if x.size != y.size:
            raise ValueError("Rankings must be of equal length.")
        table = scipy.stats.contingency.crosstab(x, y)[1]
    elif x.ndim == 2:
        if np.any(x < 0):
            raise ValueError("All elements of the contingency table must be "
                             "non-negative.")
        if np.any(x != x.astype(int)):
            raise ValueError("All elements of the contingency table must be "
                             "integer.")
        if x.nonzero()[0].size < 2:
            raise ValueError("At least two elements of the contingency table "
                             "must be nonzero.")
        table = x
    else:
        raise ValueError("x must be either a 1D or 2D array")
    # The table type is converted to a float to avoid an integer overflow
    d, p = _somers_d(table.astype(float), alternative)

    # add alias for consistency with other correlation functions
    res = SomersDResult(d, p, table)
    res.correlation = d
    return res


# This could be combined with `_all_partitions` in `_resampling.py`
def _all_partitions(nx, ny):
    """
    Partition a set of indices into two fixed-length sets in all possible ways

    Partition a set of indices 0 ... nx + ny - 1 into two sets of length nx and
    ny in all possible ways (ignoring order of elements).
    """
    z = np.arange(nx+ny)
    for c in combinations(z, nx):
        x = np.array(c)
        mask = np.ones(nx+ny, bool)
        mask[x] = False
        y = z[mask]
        yield x, y


def _compute_log_combinations(n):
    """Compute all log combination of C(n, k)."""
    gammaln_arr = gammaln(np.arange(n + 1) + 1)
    return gammaln(n + 1) - gammaln_arr - gammaln_arr[::-1]


@dataclass
class BarnardExactResult:
    statistic: float
    pvalue: float


def barnard_exact(table, alternative="two-sided", pooled=True, n=32):
    r"""Perform a Barnard exact test on a 2x2 contingency table.

    Parameters
    ----------
    table : array_like of ints
        A 2x2 contingency table.  Elements should be non-negative integers.

    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the null and alternative hypotheses. Default is 'two-sided'.
        Please see explanations in the Notes section below.

    pooled : bool, optional
        Whether to compute score statistic with pooled variance (as in
        Student's t-test, for example) or unpooled variance (as in Welch's
        t-test). Default is ``True``.

    n : int, optional
        Number of sampling points used in the construction of the sampling
        method. Note that this argument will automatically be converted to
        the next higher power of 2 since `scipy.stats.qmc.Sobol` is used to
        select sample points. Default is 32. Must be positive. In most cases,
        32 points is enough to reach good precision. More points comes at
        performance cost.

    Returns
    -------
    ber : BarnardExactResult
        A result object with the following attributes.

        statistic : float
            The Wald statistic with pooled or unpooled variance, depending
            on the user choice of `pooled`.

        pvalue : float
            P-value, the probability of obtaining a distribution at least as
            extreme as the one that was actually observed, assuming that the
            null hypothesis is true.

    See Also
    --------
    chi2_contingency : Chi-square test of independence of variables in a
        contingency table.
    fisher_exact : Fisher exact test on a 2x2 contingency table.
    boschloo_exact : Boschloo's exact test on a 2x2 contingency table,
        which is an uniformly more powerful alternative to Fisher's exact test.

    Notes
    -----
    Barnard's test is an exact test used in the analysis of contingency
    tables. It examines the association of two categorical variables, and
    is a more powerful alternative than Fisher's exact test
    for 2x2 contingency tables.

    Let's define :math:`X_0` a 2x2 matrix representing the observed sample,
    where each column stores the binomial experiment, as in the example
    below. Let's also define :math:`p_1, p_2` the theoretical binomial
    probabilities for  :math:`x_{11}` and :math:`x_{12}`. When using
    Barnard exact test, we can assert three different null hypotheses :

    - :math:`H_0 : p_1 \geq p_2` versus :math:`H_1 : p_1 < p_2`,
      with `alternative` = "less"

    - :math:`H_0 : p_1 \leq p_2` versus :math:`H_1 : p_1 > p_2`,
      with `alternative` = "greater"

    - :math:`H_0 : p_1 = p_2` versus :math:`H_1 : p_1 \neq p_2`,
      with `alternative` = "two-sided" (default one)

    In order to compute Barnard's exact test, we are using the Wald
    statistic [3]_ with pooled or unpooled variance.
    Under the default assumption that both variances are equal
    (``pooled = True``), the statistic is computed as:

    .. math::

        T(X) = \frac{
            \hat{p}_1 - \hat{p}_2
        }{
            \sqrt{
                \hat{p}(1 - \hat{p})
                (\frac{1}{c_1} +
                \frac{1}{c_2})
            }
        }

    with :math:`\hat{p}_1, \hat{p}_2` and :math:`\hat{p}` the estimator of
    :math:`p_1, p_2` and :math:`p`, the latter being the combined probability,
    given the assumption that :math:`p_1 = p_2`.

    If this assumption is invalid (``pooled = False``), the statistic is:

    .. math::

        T(X) = \frac{
            \hat{p}_1 - \hat{p}_2
        }{
            \sqrt{
                \frac{\hat{p}_1 (1 - \hat{p}_1)}{c_1} +
                \frac{\hat{p}_2 (1 - \hat{p}_2)}{c_2}
            }
        }

    The p-value is then computed as:

    .. math::

        \sum
            \binom{c_1}{x_{11}}
            \binom{c_2}{x_{12}}
            \pi^{x_{11} + x_{12}}
            (1 - \pi)^{t - x_{11} - x_{12}}

    where the sum is over all  2x2 contingency tables :math:`X` such that:
    * :math:`T(X) \leq T(X_0)` when `alternative` = "less",
    * :math:`T(X) \geq T(X_0)` when `alternative` = "greater", or
    * :math:`T(X) \geq |T(X_0)|` when `alternative` = "two-sided".
    Above, :math:`c_1, c_2` are the sum of the columns 1 and 2,
    and :math:`t` the total (sum of the 4 sample's element).

    The returned p-value is the maximum p-value taken over the nuisance
    parameter :math:`\pi`, where :math:`0 \leq \pi \leq 1`.

    This function's complexity is :math:`O(n c_1 c_2)`, where `n` is the
    number of sample points.

    References
    ----------
    .. [1] Barnard, G. A. "Significance Tests for 2x2 Tables". *Biometrika*.
           34.1/2 (1947): 123-138. :doi:`dpgkg3`

    .. [2] Mehta, Cyrus R., and Pralay Senchaudhuri. "Conditional versus
           unconditional exact tests for comparing two binomials."
           *Cytel Software Corporation* 675 (2003): 1-5.

    .. [3] "Wald Test". *Wikipedia*. https://en.wikipedia.org/wiki/Wald_test

    Examples
    --------
    An example use of Barnard's test is presented in [2]_.

        Consider the following example of a vaccine efficacy study
        (Chan, 1998). In a randomized clinical trial of 30 subjects, 15 were
        inoculated with a recombinant DNA influenza vaccine and the 15 were
        inoculated with a placebo. Twelve of the 15 subjects in the placebo
        group (80%) eventually became infected with influenza whereas for the
        vaccine group, only 7 of the 15 subjects (47%) became infected. The
        data are tabulated as a 2 x 2 table::

                Vaccine  Placebo
            Yes     7        12
            No      8        3

    When working with statistical hypothesis testing, we usually use a
    threshold probability or significance level upon which we decide
    to reject the null hypothesis :math:`H_0`. Suppose we choose the common
    significance level of 5%.

    Our alternative hypothesis is that the vaccine will lower the chance of
    becoming infected with the virus; that is, the probability :math:`p_1` of
    catching the virus with the vaccine will be *less than* the probability
    :math:`p_2` of catching the virus without the vaccine.  Therefore, we call
    `barnard_exact` with the ``alternative="less"`` option:

    >>> import scipy.stats as stats
    >>> res = stats.barnard_exact([[7, 12], [8, 3]], alternative="less")
    >>> res.statistic
    -1.894
    >>> res.pvalue
    0.03407

    Under the null hypothesis that the vaccine will not lower the chance of
    becoming infected, the probability of obtaining test results at least as
    extreme as the observed data is approximately 3.4%. Since this p-value is
    less than our chosen significance level, we have evidence to reject
    :math:`H_0` in favor of the alternative.

    Suppose we had used Fisher's exact test instead:

    >>> _, pvalue = stats.fisher_exact([[7, 12], [8, 3]], alternative="less")
    >>> pvalue
    0.0640

    With the same threshold significance of 5%, we would not have been able
    to reject the null hypothesis in favor of the alternative. As stated in
    [2]_, Barnard's test is uniformly more powerful than Fisher's exact test
    because Barnard's test does not condition on any margin. Fisher's test
    should only be used when both sets of marginals are fixed.

    """
    if n <= 0:
        raise ValueError(
            "Number of points `n` must be strictly positive, "
            f"found {n!r}"
        )

    table = np.asarray(table, dtype=np.int64)

    if not table.shape == (2, 2):
        raise ValueError("The input `table` must be of shape (2, 2).")

    if np.any(table < 0):
        raise ValueError("All values in `table` must be nonnegative.")

    if 0 in table.sum(axis=0):
        # If both values in column are zero, the p-value is 1 and
        # the score's statistic is NaN.
        return BarnardExactResult(np.nan, 1.0)

    total_col_1, total_col_2 = table.sum(axis=0)

    x1 = np.arange(total_col_1 + 1, dtype=np.int64).reshape(-1, 1)
    x2 = np.arange(total_col_2 + 1, dtype=np.int64).reshape(1, -1)

    # We need to calculate the wald statistics for each combination of x1 and
    # x2.
    p1, p2 = x1 / total_col_1, x2 / total_col_2

    if pooled:
        p = (x1 + x2) / (total_col_1 + total_col_2)
        variances = p * (1 - p) * (1 / total_col_1 + 1 / total_col_2)
    else:
        variances = p1 * (1 - p1) / total_col_1 + p2 * (1 - p2) / total_col_2

    # To avoid warning when dividing by 0
    with np.errstate(divide="ignore", invalid="ignore"):
        wald_statistic = np.divide((p1 - p2), np.sqrt(variances))

    wald_statistic[p1 == p2] = 0  # Removing NaN values

    wald_stat_obs = wald_statistic[table[0, 0], table[0, 1]]

    if alternative == "two-sided":
        index_arr = np.abs(wald_statistic) >= abs(wald_stat_obs)
    elif alternative == "less":
        index_arr = wald_statistic <= wald_stat_obs
    elif alternative == "greater":
        index_arr = wald_statistic >= wald_stat_obs
    else:
        msg = (
            "`alternative` should be one of {'two-sided', 'less', 'greater'},"
            f" found {alternative!r}"
        )
        raise ValueError(msg)

    x1_sum_x2 = x1 + x2

    x1_log_comb = _compute_log_combinations(total_col_1)
    x2_log_comb = _compute_log_combinations(total_col_2)
    x1_sum_x2_log_comb = x1_log_comb[x1] + x2_log_comb[x2]

    result = shgo(
        _get_binomial_log_p_value_with_nuisance_param,
        args=(x1_sum_x2, x1_sum_x2_log_comb, index_arr),
        bounds=((0, 1),),
        n=n,
        sampling_method="sobol",
    )

    # result.fun is the negative log pvalue and therefore needs to be
    # changed before return
    p_value = np.clip(np.exp(-result.fun), a_min=0, a_max=1)
    return BarnardExactResult(wald_stat_obs, p_value)


@dataclass
class BoschlooExactResult:
    statistic: float
    pvalue: float


def boschloo_exact(table, alternative="two-sided", n=32):
    r"""Perform Boschloo's exact test on a 2x2 contingency table.

    Parameters
    ----------
    table : array_like of ints
        A 2x2 contingency table.  Elements should be non-negative integers.

    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the null and alternative hypotheses. Default is 'two-sided'.
        Please see explanations in the Notes section below.

    n : int, optional
        Number of sampling points used in the construction of the sampling
        method. Note that this argument will automatically be converted to
        the next higher power of 2 since `scipy.stats.qmc.Sobol` is used to
        select sample points. Default is 32. Must be positive. In most cases,
        32 points is enough to reach good precision. More points comes at
        performance cost.

    Returns
    -------
    ber : BoschlooExactResult
        A result object with the following attributes.

        statistic : float
            The statistic used in Boschloo's test; that is, the p-value
            from Fisher's exact test.

        pvalue : float
            P-value, the probability of obtaining a distribution at least as
            extreme as the one that was actually observed, assuming that the
            null hypothesis is true.

    See Also
    --------
    chi2_contingency : Chi-square test of independence of variables in a
        contingency table.
    fisher_exact : Fisher exact test on a 2x2 contingency table.
    barnard_exact : Barnard's exact test, which is a more powerful alternative
        than Fisher's exact test for 2x2 contingency tables.

    Notes
    -----
    Boschloo's test is an exact test used in the analysis of contingency
    tables. It examines the association of two categorical variables, and
    is a uniformly more powerful alternative to Fisher's exact test
    for 2x2 contingency tables.

    Boschloo's exact test uses the p-value of Fisher's exact test as a
    statistic, and Boschloo's p-value is the probability under the null
    hypothesis of observing such an extreme value of this statistic.

    Let's define :math:`X_0` a 2x2 matrix representing the observed sample,
    where each column stores the binomial experiment, as in the example
    below. Let's also define :math:`p_1, p_2` the theoretical binomial
    probabilities for  :math:`x_{11}` and :math:`x_{12}`. When using
    Boschloo exact test, we can assert three different alternative hypotheses:

    - :math:`H_0 : p_1=p_2` versus :math:`H_1 : p_1 < p_2`,
      with `alternative` = "less"

    - :math:`H_0 : p_1=p_2` versus :math:`H_1 : p_1 > p_2`,
      with `alternative` = "greater"

    - :math:`H_0 : p_1=p_2` versus :math:`H_1 : p_1 \neq p_2`,
      with `alternative` = "two-sided" (default)

    There are multiple conventions for computing a two-sided p-value when the
    null distribution is asymmetric. Here, we apply the convention that the
    p-value of a two-sided test is twice the minimum of the p-values of the
    one-sided tests (clipped to 1.0). Note that `fisher_exact` follows a
    different convention, so for a given `table`, the statistic reported by
    `boschloo_exact` may differ from the p-value reported by `fisher_exact`
    when ``alternative='two-sided'``.

    .. versionadded:: 1.7.0

    References
    ----------
    .. [1] R.D. Boschloo. "Raised conditional level of significance for the
       2 x 2-table when testing the equality of two probabilities",
       Statistica Neerlandica, 24(1), 1970

    .. [2] "Boschloo's test", Wikipedia,
       https://en.wikipedia.org/wiki/Boschloo%27s_test

    .. [3] Lise M. Saari et al. "Employee attitudes and job satisfaction",
       Human Resource Management, 43(4), 395-407, 2004,
       :doi:`10.1002/hrm.20032`.

    Examples
    --------
    In the following example, we consider the article "Employee
    attitudes and job satisfaction" [3]_
    which reports the results of a survey from 63 scientists and 117 college
    professors. Of the 63 scientists, 31 said they were very satisfied with
    their jobs, whereas 74 of the college professors were very satisfied
    with their work. Is this significant evidence that college
    professors are happier with their work than scientists?
    The following table summarizes the data mentioned above::

                         college professors   scientists
        Very Satisfied   74                     31
        Dissatisfied     43                     32

    When working with statistical hypothesis testing, we usually use a
    threshold probability or significance level upon which we decide
    to reject the null hypothesis :math:`H_0`. Suppose we choose the common
    significance level of 5%.

    Our alternative hypothesis is that college professors are truly more
    satisfied with their work than scientists. Therefore, we expect
    :math:`p_1` the proportion of very satisfied college professors to be
    greater than :math:`p_2`, the proportion of very satisfied scientists.
    We thus call `boschloo_exact` with the ``alternative="greater"`` option:

    >>> import scipy.stats as stats
    >>> res = stats.boschloo_exact([[74, 31], [43, 32]], alternative="greater")
    >>> res.statistic
    0.0483
    >>> res.pvalue
    0.0355

    Under the null hypothesis that scientists are happier in their work than
    college professors, the probability of obtaining test
    results at least as extreme as the observed data is approximately 3.55%.
    Since this p-value is less than our chosen significance level, we have
    evidence to reject :math:`H_0` in favor of the alternative hypothesis.

    """
    hypergeom = distributions.hypergeom

    if n <= 0:
        raise ValueError(
            "Number of points `n` must be strictly positive,"
            f" found {n!r}"
        )

    table = np.asarray(table, dtype=np.int64)

    if not table.shape == (2, 2):
        raise ValueError("The input `table` must be of shape (2, 2).")

    if np.any(table < 0):
        raise ValueError("All values in `table` must be nonnegative.")

    if 0 in table.sum(axis=0):
        # If both values in column are zero, the p-value is 1 and
        # the score's statistic is NaN.
        return BoschlooExactResult(np.nan, np.nan)

    total_col_1, total_col_2 = table.sum(axis=0)
    total = total_col_1 + total_col_2
    x1 = np.arange(total_col_1 + 1, dtype=np.int64).reshape(1, -1)
    x2 = np.arange(total_col_2 + 1, dtype=np.int64).reshape(-1, 1)
    x1_sum_x2 = x1 + x2

    if alternative == 'less':
        pvalues = hypergeom.cdf(x1, total, x1_sum_x2, total_col_1).T
    elif alternative == 'greater':
        # Same formula as the 'less' case, but with the second column.
        pvalues = hypergeom.cdf(x2, total, x1_sum_x2, total_col_2).T
    elif alternative == 'two-sided':
        boschloo_less = boschloo_exact(table, alternative="less", n=n)
        boschloo_greater = boschloo_exact(table, alternative="greater", n=n)

        res = (
            boschloo_less if boschloo_less.pvalue < boschloo_greater.pvalue
            else boschloo_greater
        )

        # Two-sided p-value is defined as twice the minimum of the one-sided
        # p-values
        pvalue = np.clip(2 * res.pvalue, a_min=0, a_max=1)
        return BoschlooExactResult(res.statistic, pvalue)
    else:
        msg = (
            f"`alternative` should be one of {'two-sided', 'less', 'greater'},"
            f" found {alternative!r}"
        )
        raise ValueError(msg)

    fisher_stat = pvalues[table[0, 0], table[0, 1]]

    # fisher_stat * (1+1e-13) guards us from small numerical error. It is
    # equivalent to np.isclose with relative tol of 1e-13 and absolute tol of 0
    # For more throughout explanations, see gh-14178
    index_arr = pvalues <= fisher_stat * (1+1e-13)

    x1, x2, x1_sum_x2 = x1.T, x2.T, x1_sum_x2.T
    x1_log_comb = _compute_log_combinations(total_col_1)
    x2_log_comb = _compute_log_combinations(total_col_2)
    x1_sum_x2_log_comb = x1_log_comb[x1] + x2_log_comb[x2]

    result = shgo(
        _get_binomial_log_p_value_with_nuisance_param,
        args=(x1_sum_x2, x1_sum_x2_log_comb, index_arr),
        bounds=((0, 1),),
        n=n,
        sampling_method="sobol",
    )

    # result.fun is the negative log pvalue and therefore needs to be
    # changed before return
    p_value = np.clip(np.exp(-result.fun), a_min=0, a_max=1)
    return BoschlooExactResult(fisher_stat, p_value)


def _get_binomial_log_p_value_with_nuisance_param(
    nuisance_param, x1_sum_x2, x1_sum_x2_log_comb, index_arr
):
    r"""
    Compute the log pvalue in respect of a nuisance parameter considering
    a 2x2 sample space.

    Parameters
    ----------
    nuisance_param : float
        nuisance parameter used in the computation of the maximisation of
        the p-value. Must be between 0 and 1

    x1_sum_x2 : ndarray
        Sum of x1 and x2 inside barnard_exact

    x1_sum_x2_log_comb : ndarray
        sum of the log combination of x1 and x2

    index_arr : ndarray of boolean

    Returns
    -------
    p_value : float
        Return the maximum p-value considering every nuisance parameter
        between 0 and 1

    Notes
    -----

    Both Barnard's test and Boschloo's test iterate over a nuisance parameter
    :math:`\pi \in [0, 1]` to find the maximum p-value. To search this
    maxima, this function return the negative log pvalue with respect to the
    nuisance parameter passed in params. This negative log p-value is then
    used in `shgo` to find the minimum negative pvalue which is our maximum
    pvalue.

    Also, to compute the different combination used in the
    p-values' computation formula, this function uses `gammaln` which is
    more tolerant for large value than `scipy.special.comb`. `gammaln` gives
    a log combination. For the little precision loss, performances are
    improved a lot.
    """
    t1, t2 = x1_sum_x2.shape
    n = t1 + t2 - 2
    with np.errstate(divide="ignore", invalid="ignore"):
        log_nuisance = np.log(
            nuisance_param,
            out=np.zeros_like(nuisance_param),
            where=nuisance_param >= 0,
        )
        log_1_minus_nuisance = np.log(
            1 - nuisance_param,
            out=np.zeros_like(nuisance_param),
            where=1 - nuisance_param >= 0,
        )

        nuisance_power_x1_x2 = log_nuisance * x1_sum_x2
        nuisance_power_x1_x2[(x1_sum_x2 == 0)[:, :]] = 0

        nuisance_power_n_minus_x1_x2 = log_1_minus_nuisance * (n - x1_sum_x2)
        nuisance_power_n_minus_x1_x2[(x1_sum_x2 == n)[:, :]] = 0

        tmp_log_values_arr = (
            x1_sum_x2_log_comb
            + nuisance_power_x1_x2
            + nuisance_power_n_minus_x1_x2
        )

    tmp_values_from_index = tmp_log_values_arr[index_arr]

    # To avoid dividing by zero in log function and getting inf value,
    # values are centered according to the max
    max_value = tmp_values_from_index.max()

    # To have better result's precision, the log pvalue is taken here.
    # Indeed, pvalue is included inside [0, 1] interval. Passing the
    # pvalue to log makes the interval a lot bigger ([-inf, 0]), and thus
    # help us to achieve better precision
    with np.errstate(divide="ignore", invalid="ignore"):
        log_probs = np.exp(tmp_values_from_index - max_value).sum()
        log_pvalue = max_value + np.log(
            log_probs,
            out=np.full_like(log_probs, -np.inf),
            where=log_probs > 0,
        )

    # Since shgo find the minima, minus log pvalue is returned
    return -log_pvalue


def _pval_cvm_2samp_exact(s, m, n):
    """
    Compute the exact p-value of the Cramer-von Mises two-sample test
    for a given value s of the test statistic.
    m and n are the sizes of the samples.

    [1] Y. Xiao, A. Gordon, and A. Yakovlev, "A C++ Program for
        the Cramér-Von Mises Two-Sample Test", J. Stat. Soft.,
        vol. 17, no. 8, pp. 1-15, Dec. 2006.
    [2] T. W. Anderson "On the Distribution of the Two-Sample Cramer-von Mises
        Criterion," The Annals of Mathematical Statistics, Ann. Math. Statist.
        33(3), 1148-1159, (September, 1962)
    """

    # [1, p. 3]
    lcm = np.lcm(m, n)
    # [1, p. 4], below eq. 3
    a = lcm // m
    b = lcm // n
    # Combine Eq. 9 in [2] with Eq. 2 in [1] and solve for $\zeta$
    # Hint: `s` is $U$ in [2], and $T_2$ in [1] is $T$ in [2]
    mn = m * n
    zeta = lcm ** 2 * (m + n) * (6 * s - mn * (4 * mn - 1)) // (6 * mn ** 2)

    # bound maximum value that may appear in `gs` (remember both rows!)
    zeta_bound = lcm**2 * (m + n)  # bound elements in row 1
    combinations = comb(m + n, m)  # sum of row 2
    max_gs = max(zeta_bound, combinations)
    dtype = np.min_scalar_type(max_gs)

    # the frequency table of $g_{u, v}^+$ defined in [1, p. 6]
    gs = ([np.array([[0], [1]], dtype=dtype)]
          + [np.empty((2, 0), dtype=dtype) for _ in range(m)])
    for u in range(n + 1):
        next_gs = []
        tmp = np.empty((2, 0), dtype=dtype)
        for v, g in enumerate(gs):
            # Calculate g recursively with eq. 11 in [1]. Even though it
            # doesn't look like it, this also does 12/13 (all of Algorithm 1).
            vi, i0, i1 = np.intersect1d(tmp[0], g[0], return_indices=True)
            tmp = np.concatenate([
                np.stack([vi, tmp[1, i0] + g[1, i1]]),
                np.delete(tmp, i0, 1),
                np.delete(g, i1, 1)
            ], 1)
            res = (a * v - b * u) ** 2
            tmp[0] += res.astype(dtype)
            next_gs.append(tmp)
        gs = next_gs
    value, freq = gs[m]
    return np.float64(np.sum(freq[value >= zeta]) / combinations)


@_axis_nan_policy_factory(CramerVonMisesResult, n_samples=2, too_small=1,
                          result_to_tuple=_cvm_result_to_tuple)
def cramervonmises_2samp(x, y, method='auto'):
    """Perform the two-sample Cramér-von Mises test for goodness of fit.

    This is the two-sample version of the Cramér-von Mises test ([1]_):
    for two independent samples :math:`X_1, ..., X_n` and
    :math:`Y_1, ..., Y_m`, the null hypothesis is that the samples
    come from the same (unspecified) continuous distribution.

    Parameters
    ----------
    x : array_like
        A 1-D array of observed values of the random variables :math:`X_i`.
        Must contain at least two observations.
    y : array_like
        A 1-D array of observed values of the random variables :math:`Y_i`.
        Must contain at least two observations.
    method : {'auto', 'asymptotic', 'exact'}, optional
        The method used to compute the p-value, see Notes for details.
        The default is 'auto'.

    Returns
    -------
    res : object with attributes
        statistic : float
            Cramér-von Mises statistic.
        pvalue : float
            The p-value.

    See Also
    --------
    cramervonmises, anderson_ksamp, epps_singleton_2samp, ks_2samp

    Notes
    -----
    .. versionadded:: 1.7.0

    The statistic is computed according to equation 9 in [2]_. The
    calculation of the p-value depends on the keyword `method`:

    - ``asymptotic``: The p-value is approximated by using the limiting
      distribution of the test statistic.
    - ``exact``: The exact p-value is computed by enumerating all
      possible combinations of the test statistic, see [2]_.

    If ``method='auto'``, the exact approach is used
    if both samples contain equal to or less than 20 observations,
    otherwise the asymptotic distribution is used.

    If the underlying distribution is not continuous, the p-value is likely to
    be conservative (Section 6.2 in [3]_). When ranking the data to compute
    the test statistic, midranks are used if there are ties.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Cramer-von_Mises_criterion
    .. [2] Anderson, T.W. (1962). On the distribution of the two-sample
           Cramer-von-Mises criterion. The Annals of Mathematical
           Statistics, pp. 1148-1159.
    .. [3] Conover, W.J., Practical Nonparametric Statistics, 1971.

    Examples
    --------

    Suppose we wish to test whether two samples generated by
    ``scipy.stats.norm.rvs`` have the same distribution. We choose a
    significance level of alpha=0.05.

    >>> import numpy as np
    >>> from scipy import stats
    >>> rng = np.random.default_rng()
    >>> x = stats.norm.rvs(size=100, random_state=rng)
    >>> y = stats.norm.rvs(size=70, random_state=rng)
    >>> res = stats.cramervonmises_2samp(x, y)
    >>> res.statistic, res.pvalue
    (0.29376470588235293, 0.1412873014573014)

    The p-value exceeds our chosen significance level, so we do not
    reject the null hypothesis that the observed samples are drawn from the
    same distribution.

    For small sample sizes, one can compute the exact p-values:

    >>> x = stats.norm.rvs(size=7, random_state=rng)
    >>> y = stats.t.rvs(df=2, size=6, random_state=rng)
    >>> res = stats.cramervonmises_2samp(x, y, method='exact')
    >>> res.statistic, res.pvalue
    (0.197802197802198, 0.31643356643356646)

    The p-value based on the asymptotic distribution is a good approximation
    even though the sample size is small.

    >>> res = stats.cramervonmises_2samp(x, y, method='asymptotic')
    >>> res.statistic, res.pvalue
    (0.197802197802198, 0.2966041181527128)

    Independent of the method, one would not reject the null hypothesis at the
    chosen significance level in this example.

    """
    xa = np.sort(np.asarray(x))
    ya = np.sort(np.asarray(y))

    if xa.size <= 1 or ya.size <= 1:
        raise ValueError('x and y must contain at least two observations.')
    if method not in ['auto', 'exact', 'asymptotic']:
        raise ValueError('method must be either auto, exact or asymptotic.')

    nx = len(xa)
    ny = len(ya)

    if method == 'auto':
        if max(nx, ny) > 20:
            method = 'asymptotic'
        else:
            method = 'exact'

    # get ranks of x and y in the pooled sample
    z = np.concatenate([xa, ya])
    # in case of ties, use midrank (see [1])
    r = scipy.stats.rankdata(z, method='average')
    rx = r[:nx]
    ry = r[nx:]

    # compute U (eq. 10 in [2])
    u = nx * np.sum((rx - np.arange(1, nx+1))**2)
    u += ny * np.sum((ry - np.arange(1, ny+1))**2)

    # compute T (eq. 9 in [2])
    k, N = nx*ny, nx + ny
    t = u / (k*N) - (4*k - 1)/(6*N)

    if method == 'exact':
        p = _pval_cvm_2samp_exact(u, nx, ny)
    else:
        # compute expected value and variance of T (eq. 11 and 14 in [2])
        et = (1 + 1/N)/6
        vt = (N+1) * (4*k*N - 3*(nx**2 + ny**2) - 2*k)
        vt = vt / (45 * N**2 * 4 * k)

        # computed the normalized statistic (eq. 15 in [2])
        tn = 1/6 + (t - et) / np.sqrt(45 * vt)

        # approximate distribution of tn with limiting distribution
        # of the one-sample test statistic
        # if tn < 0.003, the _cdf_cvm_inf(tn) < 1.28*1e-18, return 1.0 directly
        if tn < 0.003:
            p = 1.0
        else:
            p = max(0, 1. - _cdf_cvm_inf(tn))

    return CramerVonMisesResult(statistic=t, pvalue=p)


class TukeyHSDResult:
    """Result of `scipy.stats.tukey_hsd`.

    Attributes
    ----------
    statistic : float ndarray
        The computed statistic of the test for each comparison. The element
        at index ``(i, j)`` is the statistic for the comparison between groups
        ``i`` and ``j``.
    pvalue : float ndarray
        The associated p-value from the studentized range distribution. The
        element at index ``(i, j)`` is the p-value for the comparison
        between groups ``i`` and ``j``.

    Notes
    -----
    The string representation of this object displays the most recently
    calculated confidence interval, and if none have been previously
    calculated, it will evaluate ``confidence_interval()``.

    References
    ----------
    .. [1] NIST/SEMATECH e-Handbook of Statistical Methods, "7.4.7.1. Tukey's
           Method."
           https://www.itl.nist.gov/div898/handbook/prc/section4/prc471.htm,
           28 November 2020.
    """

    def __init__(self, statistic, pvalue, _nobs, _ntreatments, _stand_err):
        self.statistic = statistic
        self.pvalue = pvalue
        self._ntreatments = _ntreatments
        self._nobs = _nobs
        self._stand_err = _stand_err
        self._ci = None
        self._ci_cl = None

    def __str__(self):
        # Note: `__str__` prints the confidence intervals from the most
        # recent call to `confidence_interval`. If it has not been called,
        # it will be called with the default CL of .95.
        if self._ci is None:
            self.confidence_interval(confidence_level=.95)
        s = ("Tukey's HSD Pairwise Group Comparisons"
             f" ({self._ci_cl*100:.1f}% Confidence Interval)\n")
        s += "Comparison  Statistic  p-value  Lower CI  Upper CI\n"
        for i in range(self.pvalue.shape[0]):
            for j in range(self.pvalue.shape[0]):
                if i != j:
                    s += (f" ({i} - {j}) {self.statistic[i, j]:>10.3f}"
                          f"{self.pvalue[i, j]:>10.3f}"
                          f"{self._ci.low[i, j]:>10.3f}"
                          f"{self._ci.high[i, j]:>10.3f}\n")
        return s

    def confidence_interval(self, confidence_level=.95):
        """Compute the confidence interval for the specified confidence level.

        Parameters
        ----------
        confidence_level : float, optional
            Confidence level for the computed confidence interval
            of the estimated proportion. Default is .95.

        Returns
        -------
        ci : ``ConfidenceInterval`` object
            The object has attributes ``low`` and ``high`` that hold the
            lower and upper bounds of the confidence intervals for each
            comparison. The high and low values are accessible for each
            comparison at index ``(i, j)`` between groups ``i`` and ``j``.

        References
        ----------
        .. [1] NIST/SEMATECH e-Handbook of Statistical Methods, "7.4.7.1.
               Tukey's Method."
               https://www.itl.nist.gov/div898/handbook/prc/section4/prc471.htm,
               28 November 2020.

        Examples
        --------
        >>> from scipy.stats import tukey_hsd
        >>> group0 = [24.5, 23.5, 26.4, 27.1, 29.9]
        >>> group1 = [28.4, 34.2, 29.5, 32.2, 30.1]
        >>> group2 = [26.1, 28.3, 24.3, 26.2, 27.8]
        >>> result = tukey_hsd(group0, group1, group2)
        >>> ci = result.confidence_interval()
        >>> ci.low
        array([[-3.649159, -8.249159, -3.909159],
               [ 0.950841, -3.649159,  0.690841],
               [-3.389159, -7.989159, -3.649159]])
        >>> ci.high
        array([[ 3.649159, -0.950841,  3.389159],
               [ 8.249159,  3.649159,  7.989159],
               [ 3.909159, -0.690841,  3.649159]])
        """
        # check to see if the supplied confidence level matches that of the
        # previously computed CI.
        if (self._ci is not None and self._ci_cl is not None and
                confidence_level == self._ci_cl):
            return self._ci

        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1.")
        # determine the critical value of the studentized range using the
        # appropriate confidence level, number of treatments, and degrees
        # of freedom as determined by the number of data less the number of
        # treatments. ("Confidence limits for Tukey's method")[1]. Note that
        # in the cases of unequal sample sizes there will be a criterion for
        # each group comparison.
        params = (confidence_level, self._nobs, self._ntreatments - self._nobs)
        srd = distributions.studentized_range.ppf(*params)
        # also called maximum critical value, the Tukey criterion is the
        # studentized range critical value * the square root of mean square
        # error over the sample size.
        tukey_criterion = srd * self._stand_err
        # the confidence levels are determined by the
        # `mean_differences` +- `tukey_criterion`
        upper_conf = self.statistic + tukey_criterion
        lower_conf = self.statistic - tukey_criterion
        self._ci = ConfidenceInterval(low=lower_conf, high=upper_conf)
        self._ci_cl = confidence_level
        return self._ci


def _tukey_hsd_iv(args):
    if (len(args)) < 2:
        raise ValueError("There must be more than 1 treatment.")
    args = [np.asarray(arg) for arg in args]
    for arg in args:
        if arg.ndim != 1:
            raise ValueError("Input samples must be one-dimensional.")
        if arg.size <= 1:
            raise ValueError("Input sample size must be greater than one.")
        if np.isinf(arg).any():
            raise ValueError("Input samples must be finite.")
    return args


def tukey_hsd(*args):
    """Perform Tukey's HSD test for equality of means over multiple treatments.

    Tukey's honestly significant difference (HSD) test performs pairwise
    comparison of means for a set of samples. Whereas ANOVA (e.g. `f_oneway`)
    assesses whether the true means underlying each sample are identical,
    Tukey's HSD is a post hoc test used to compare the mean of each sample
    to the mean of each other sample.

    The null hypothesis is that the distributions underlying the samples all
    have the same mean. The test statistic, which is computed for every
    possible pairing of samples, is simply the difference between the sample
    means. For each pair, the p-value is the probability under the null
    hypothesis (and other assumptions; see notes) of observing such an extreme
    value of the statistic, considering that many pairwise comparisons are
    being performed. Confidence intervals for the difference between each pair
    of means are also available.

    Parameters
    ----------
    sample1, sample2, ... : array_like
        The sample measurements for each group. There must be at least
        two arguments.

    Returns
    -------
    result : `~scipy.stats._result_classes.TukeyHSDResult` instance
        The return value is an object with the following attributes:

        statistic : float ndarray
            The computed statistic of the test for each comparison. The element
            at index ``(i, j)`` is the statistic for the comparison between
            groups ``i`` and ``j``.
        pvalue : float ndarray
            The computed p-value of the test for each comparison. The element
            at index ``(i, j)`` is the p-value for the comparison between
            groups ``i`` and ``j``.

        The object has the following methods:

        confidence_interval(confidence_level=0.95):
            Compute the confidence interval for the specified confidence level.

    See Also
    --------
    dunnett : performs comparison of means against a control group.

    Notes
    -----
    The use of this test relies on several assumptions.

    1. The observations are independent within and among groups.
    2. The observations within each group are normally distributed.
    3. The distributions from which the samples are drawn have the same finite
       variance.

    The original formulation of the test was for samples of equal size [6]_.
    In case of unequal sample sizes, the test uses the Tukey-Kramer method
    [4]_.

    References
    ----------
    .. [1] NIST/SEMATECH e-Handbook of Statistical Methods, "7.4.7.1. Tukey's
           Method."
           https://www.itl.nist.gov/div898/handbook/prc/section4/prc471.htm,
           28 November 2020.
    .. [2] Abdi, Herve & Williams, Lynne. (2021). "Tukey's Honestly Significant
           Difference (HSD) Test."
           https://personal.utdallas.edu/~herve/abdi-HSD2010-pretty.pdf
    .. [3] "One-Way ANOVA Using SAS PROC ANOVA & PROC GLM." SAS
           Tutorials, 2007, www.stattutorials.com/SAS/TUTORIAL-PROC-GLM.htm.
    .. [4] Kramer, Clyde Young. "Extension of Multiple Range Tests to Group
           Means with Unequal Numbers of Replications." Biometrics, vol. 12,
           no. 3, 1956, pp. 307-310. JSTOR, www.jstor.org/stable/3001469.
           Accessed 25 May 2021.
    .. [5] NIST/SEMATECH e-Handbook of Statistical Methods, "7.4.3.3.
           The ANOVA table and tests of hypotheses about means"
           https://www.itl.nist.gov/div898/handbook/prc/section4/prc433.htm,
           2 June 2021.
    .. [6] Tukey, John W. "Comparing Individual Means in the Analysis of
           Variance." Biometrics, vol. 5, no. 2, 1949, pp. 99-114. JSTOR,
           www.jstor.org/stable/3001913. Accessed 14 June 2021.


    Examples
    --------
    Here are some data comparing the time to relief of three brands of
    headache medicine, reported in minutes. Data adapted from [3]_.

    >>> import numpy as np
    >>> from scipy.stats import tukey_hsd
    >>> group0 = [24.5, 23.5, 26.4, 27.1, 29.9]
    >>> group1 = [28.4, 34.2, 29.5, 32.2, 30.1]
    >>> group2 = [26.1, 28.3, 24.3, 26.2, 27.8]

    We would like to see if the means between any of the groups are
    significantly different. First, visually examine a box and whisker plot.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(1, 1)
    >>> ax.boxplot([group0, group1, group2])
    >>> ax.set_xticklabels(["group0", "group1", "group2"]) # doctest: +SKIP
    >>> ax.set_ylabel("mean") # doctest: +SKIP
    >>> plt.show()

    From the box and whisker plot, we can see overlap in the interquartile
    ranges group 1 to group 2 and group 3, but we can apply the ``tukey_hsd``
    test to determine if the difference between means is significant. We
    set a significance level of .05 to reject the null hypothesis.

    >>> res = tukey_hsd(group0, group1, group2)
    >>> print(res)
    Tukey's HSD Pairwise Group Comparisons (95.0% Confidence Interval)
    Comparison  Statistic  p-value   Lower CI   Upper CI
    (0 - 1)     -4.600      0.014     -8.249     -0.951
    (0 - 2)     -0.260      0.980     -3.909      3.389
    (1 - 0)      4.600      0.014      0.951      8.249
    (1 - 2)      4.340      0.020      0.691      7.989
    (2 - 0)      0.260      0.980     -3.389      3.909
    (2 - 1)     -4.340      0.020     -7.989     -0.691

    The null hypothesis is that each group has the same mean. The p-value for
    comparisons between ``group0`` and ``group1`` as well as ``group1`` and
    ``group2`` do not exceed .05, so we reject the null hypothesis that they
    have the same means. The p-value of the comparison between ``group0``
    and ``group2`` exceeds .05, so we accept the null hypothesis that there
    is not a significant difference between their means.

    We can also compute the confidence interval associated with our chosen
    confidence level.

    >>> group0 = [24.5, 23.5, 26.4, 27.1, 29.9]
    >>> group1 = [28.4, 34.2, 29.5, 32.2, 30.1]
    >>> group2 = [26.1, 28.3, 24.3, 26.2, 27.8]
    >>> result = tukey_hsd(group0, group1, group2)
    >>> conf = res.confidence_interval(confidence_level=.99)
    >>> for ((i, j), l) in np.ndenumerate(conf.low):
    ...     # filter out self comparisons
    ...     if i != j:
    ...         h = conf.high[i,j]
    ...         print(f"({i} - {j}) {l:>6.3f} {h:>6.3f}")
    (0 - 1) -9.480  0.280
    (0 - 2) -5.140  4.620
    (1 - 0) -0.280  9.480
    (1 - 2) -0.540  9.220
    (2 - 0) -4.620  5.140
    (2 - 1) -9.220  0.540
    """
    args = _tukey_hsd_iv(args)
    ntreatments = len(args)
    means = np.asarray([np.mean(arg) for arg in args])
    nsamples_treatments = np.asarray([a.size for a in args])
    nobs = np.sum(nsamples_treatments)

    # determine mean square error [5]. Note that this is sometimes called
    # mean square error within.
    mse = (np.sum([np.var(arg, ddof=1) for arg in args] *
                  (nsamples_treatments - 1)) / (nobs - ntreatments))

    # The calculation of the standard error differs when treatments differ in
    # size. See ("Unequal sample sizes")[1].
    if np.unique(nsamples_treatments).size == 1:
        # all input groups are the same length, so only one value needs to be
        # calculated [1].
        normalize = 2 / nsamples_treatments[0]
    else:
        # to compare groups of differing sizes, we must compute a variance
        # value for each individual comparison. Use broadcasting to get the
        # resulting matrix. [3], verified against [4] (page 308).
        normalize = 1 / nsamples_treatments + 1 / nsamples_treatments[None].T

    # the standard error is used in the computation of the tukey criterion and
    # finding the p-values.
    stand_err = np.sqrt(normalize * mse / 2)

    # the mean difference is the test statistic.
    mean_differences = means[None].T - means

    # Calculate the t-statistic to use within the survival function of the
    # studentized range to get the p-value.
    t_stat = np.abs(mean_differences) / stand_err

    params = t_stat, ntreatments, nobs - ntreatments
    pvalues = distributions.studentized_range.sf(*params)

    return TukeyHSDResult(mean_differences, pvalues, ntreatments,
                          nobs, stand_err)
<|endoftext|>r"""
====================================================
Quasi-Monte Carlo submodule (:mod:`scipy.stats.qmc`)
====================================================

.. currentmodule:: scipy.stats.qmc

This module provides Quasi-Monte Carlo generators and associated helper
functions.


Quasi-Monte Carlo
=================

Engines
-------

.. autosummary::
   :toctree: generated/

   QMCEngine
   Sobol
   Halton
   LatinHypercube
   PoissonDisk
   MultinomialQMC
   MultivariateNormalQMC

Helpers
-------

.. autosummary::
   :toctree: generated/

   discrepancy
   geometric_discrepancy
   update_discrepancy
   scale


Introduction to Quasi-Monte Carlo
=================================

Quasi-Monte Carlo (QMC) methods [1]_, [2]_, [3]_ provide an
:math:`n \times d` array of numbers in :math:`[0,1]`. They can be used in
place of :math:`n` points from the :math:`U[0,1]^{d}` distribution. Compared to
random points, QMC points are designed to have fewer gaps and clumps. This is
quantified by discrepancy measures [4]_. From the Koksma-Hlawka
inequality [5]_ we know that low discrepancy reduces a bound on
integration error. Averaging a function :math:`f` over :math:`n` QMC points
can achieve an integration error close to :math:`O(n^{-1})` for well
behaved functions [2]_.

Most QMC constructions are designed for special values of :math:`n`
such as powers of 2 or large primes. Changing the sample
size by even one can degrade their performance, even their
rate of convergence [6]_. For instance :math:`n=100` points may give less
accuracy than :math:`n=64` if the method was designed for :math:`n=2^m`.

Some QMC constructions are extensible in :math:`n`: we can find
another special sample size :math:`n' > n` and often an infinite
sequence of increasing special sample sizes. Some QMC
constructions are extensible in :math:`d`: we can increase the dimension,
possibly to some upper bound, and typically without requiring
special values of :math:`d`. Some QMC methods are extensible in
both :math:`n` and :math:`d`.

QMC points are deterministic. That makes it hard to estimate the accuracy of
integrals estimated by averages over QMC points. Randomized QMC (RQMC) [7]_
points are constructed so that each point is individually :math:`U[0,1]^{d}`
while collectively the :math:`n` points retain their low discrepancy.
One can make :math:`R` independent replications of RQMC points to
see how stable a computation is. From :math:`R` independent values,
a t-test (or bootstrap t-test [8]_) then gives approximate confidence
intervals on the mean value. Some RQMC methods produce a
root mean squared error that is actually :math:`o(1/n)` and smaller than
the rate seen in unrandomized QMC. An intuitive explanation is
that the error is a sum of many small ones and random errors
cancel in a way that deterministic ones do not. RQMC also
has advantages on integrands that are singular or, for other
reasons, fail to be Riemann integrable.

(R)QMC cannot beat Bahkvalov's curse of dimension (see [9]_). For
any random or deterministic method, there are worst case functions
that will give it poor performance in high dimensions. A worst
case function for QMC might be 0 at all n points but very
large elsewhere. Worst case analyses get very pessimistic
in high dimensions. (R)QMC can bring a great improvement over
MC when the functions on which it is used are not worst case.
For instance (R)QMC can be especially effective on integrands
that are well approximated by sums of functions of
some small number of their input variables at a time [10]_, [11]_.
That property is often a surprising finding about those functions.

Also, to see an improvement over IID MC, (R)QMC requires a bit of smoothness of
the integrand, roughly the mixed first order derivative in each direction,
:math:`\partial^d f/\partial x_1 \cdots \partial x_d`, must be integral.
For instance, a function that is 1 inside the hypersphere and 0 outside of it
has infinite variation in the sense of Hardy and Krause for any dimension
:math:`d = 2`.

Scrambled nets are a kind of RQMC that have some valuable robustness
properties [12]_. If the integrand is square integrable, they give variance
:math:`var_{SNET} = o(1/n)`. There is a finite upper bound on
:math:`var_{SNET} / var_{MC}` that holds simultaneously for every square
integrable integrand. Scrambled nets satisfy a strong law of large numbers
for :math:`f` in :math:`L^p` when :math:`p>1`. In some
special cases there is a central limit theorem [13]_. For smooth enough
integrands they can achieve RMSE nearly :math:`O(n^{-3})`. See [12]_
for references about these properties.

The main kinds of QMC methods are lattice rules [14]_ and digital
nets and sequences [2]_, [15]_. The theories meet up in polynomial
lattice rules [16]_ which can produce digital nets. Lattice rules
require some form of search for good constructions. For digital
nets there are widely used default constructions.

The most widely used QMC methods are Sobol' sequences [17]_.
These are digital nets. They are extensible in both :math:`n` and :math:`d`.
They can be scrambled. The special sample sizes are powers
of 2. Another popular method are Halton sequences [18]_.
The constructions resemble those of digital nets. The earlier
dimensions have much better equidistribution properties than
later ones. There are essentially no special sample sizes.
They are not thought to be as accurate as Sobol' sequences.
They can be scrambled. The nets of Faure [19]_ are also widely
used. All dimensions are equally good, but the special sample
sizes grow rapidly with dimension :math:`d`. They can be scrambled.
The nets of Niederreiter and Xing [20]_ have the best asymptotic
properties but have not shown good empirical performance [21]_.

Higher order digital nets are formed by a digit interleaving process
in the digits of the constructed points. They can achieve higher
levels of asymptotic accuracy given higher smoothness conditions on :math:`f`
and they can be scrambled [22]_. There is little or no empirical work
showing the improved rate to be attained.

Using QMC is like using the entire period of a small random
number generator. The constructions are similar and so
therefore are the computational costs [23]_.

(R)QMC is sometimes improved by passing the points through
a baker's transformation (tent function) prior to using them.
That function has the form :math:`1-2|x-1/2|`. As :math:`x` goes from 0 to
1, this function goes from 0 to 1 and then back. It is very
useful to produce a periodic function for lattice rules [14]_,
and sometimes it improves the convergence rate [24]_.

It is not straightforward to apply QMC methods to Markov
chain Monte Carlo (MCMC).  We can think of MCMC as using
:math:`n=1` point in :math:`[0,1]^{d}` for very large :math:`d`, with
ergodic results corresponding to :math:`d \to \infty`. One proposal is
in [25]_ and under strong conditions an improved rate of convergence
has been shown [26]_.

Returning to Sobol' points: there are many versions depending
on what are called direction numbers. Those are the result of
searches and are tabulated. A very widely used set of direction
numbers come from [27]_. It is extensible in dimension up to
:math:`d=21201`.

References
----------
.. [1] Owen, Art B. "Monte Carlo Book: the Quasi-Monte Carlo parts." 2019.
.. [2] Niederreiter, Harald. "Random number generation and quasi-Monte Carlo
   methods." Society for Industrial and Applied Mathematics, 1992.
.. [3] Dick, Josef, Frances Y. Kuo, and Ian H. Sloan. "High-dimensional
   integration: the quasi-Monte Carlo way." Acta Numerica no. 22: 133, 2013.
.. [4] Aho, A. V., C. Aistleitner, T. Anderson, K. Appel, V. Arnol'd, N.
   Aronszajn, D. Asotsky et al. "W. Chen et al.(eds.), "A Panorama of
   Discrepancy Theory", Sringer International Publishing,
   Switzerland: 679, 2014.
.. [5] Hickernell, Fred J. "Koksma-Hlawka Inequality." Wiley StatsRef:
   Statistics Reference Online, 2014.
.. [6] Owen, Art B. "On dropping the first Sobol' point." :arxiv:`2008.08051`,
   2020.
.. [7] L'Ecuyer, Pierre, and Christiane Lemieux. "Recent advances in randomized
   quasi-Monte Carlo methods." In Modeling uncertainty, pp. 419-474. Springer,
   New York, NY, 2002.
.. [8] DiCiccio, Thomas J., and Bradley Efron. "Bootstrap confidence
   intervals." Statistical science: 189-212, 1996.
.. [9] Dimov, Ivan T. "Monte Carlo methods for applied scientists." World
   Scientific, 2008.
.. [10] Caflisch, Russel E., William J. Morokoff, and Art B. Owen. "Valuation
   of mortgage backed securities using Brownian bridges to reduce effective
   dimension." Journal of Computational Finance: no. 1 27-46, 1997.
.. [11] Sloan, Ian H., and Henryk Wozniakowski. "When are quasi-Monte Carlo
   algorithms efficient for high dimensional integrals?." Journal of Complexity
   14, no. 1 (1998): 1-33.
.. [12] Owen, Art B., and Daniel Rudolf, "A strong law of large numbers for
   scrambled net integration." SIAM Review, to appear.
.. [13] Loh, Wei-Liem. "On the asymptotic distribution of scrambled net
   quadrature." The Annals of Statistics 31, no. 4: 1282-1324, 2003.
.. [14] Sloan, Ian H. and S. Joe. "Lattice methods for multiple integration."
   Oxford University Press, 1994.
.. [15] Dick, Josef, and Friedrich Pillichshammer. "Digital nets and sequences:
   discrepancy theory and quasi-Monte Carlo integration." Cambridge University
   Press, 2010.
.. [16] Dick, Josef, F. Kuo, Friedrich Pillichshammer, and I. Sloan.
   "Construction algorithms for polynomial lattice rules for multivariate
   integration." Mathematics of computation 74, no. 252: 1895-1921, 2005.
.. [17] Sobol', Il'ya Meerovich. "On the distribution of points in a cube and
   the approximate evaluation of integrals." Zhurnal Vychislitel'noi Matematiki
   i Matematicheskoi Fiziki 7, no. 4: 784-802, 1967.
.. [18] Halton, John H. "On the efficiency of certain quasi-random sequences of
   points in evaluating multi-dimensional integrals." Numerische Mathematik 2,
   no. 1: 84-90, 1960.
.. [19] Faure, Henri. "Discrepance de suites associees a un systeme de
   numeration (en dimension s)." Acta arithmetica 41, no. 4: 337-351, 1982.
.. [20] Niederreiter, Harold, and Chaoping Xing. "Low-discrepancy sequences and
   global function fields with many rational places." Finite Fields and their
   applications 2, no. 3: 241-273, 1996.
.. [21] Hong, Hee Sun, and Fred J. Hickernell. "Algorithm 823: Implementing
   scrambled digital sequences." ACM Transactions on Mathematical Software
   (TOMS) 29, no. 2: 95-109, 2003.
.. [22] Dick, Josef. "Higher order scrambled digital nets achieve the optimal
   rate of the root mean square error for smooth integrands." The Annals of
   Statistics 39, no. 3: 1372-1398, 2011.
.. [23] Niederreiter, Harald. "Multidimensional numerical integration using
   pseudorandom numbers." In Stochastic Programming 84 Part I, pp. 17-38.
   Springer, Berlin, Heidelberg, 1986.
.. [24] Hickernell, Fred J. "Obtaining O (N-2+e) Convergence for Lattice
   Quadrature Rules." In Monte Carlo and Quasi-Monte Carlo Methods 2000,
   pp. 274-289. Springer, Berlin, Heidelberg, 2002.
.. [25] Owen, Art B., and Seth D. Tribble. "A quasi-Monte Carlo Metropolis
   algorithm." Proceedings of the National Academy of Sciences 102,
   no. 25: 8844-8849, 2005.
.. [26] Chen, Su. "Consistency and convergence rate of Markov chain quasi Monte
   Carlo with examples." PhD diss., Stanford University, 2011.
.. [27] Joe, Stephen, and Frances Y. Kuo. "Constructing Sobol sequences with
   better two-dimensional projections." SIAM Journal on Scientific Computing
   30, no. 5: 2635-2654, 2008.

"""
from ._qmc import *  # noqa: F403
from ._qmc import __all__  # noqa: F401
<|endoftext|>"""Quasi-Monte Carlo engines and helpers."""
import copy
import math
import numbers
import os
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import (
    ClassVar,
    Literal,
    overload,
    TYPE_CHECKING,
)
from collections.abc import Callable

import numpy as np

from scipy._lib._util import DecimalNumber, GeneratorType, IntNumber, SeedType

if TYPE_CHECKING:
    import numpy.typing as npt
    
import scipy.stats as stats
from scipy._lib._util import rng_integers, _rng_spawn, _transition_to_rng
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import distance, Voronoi
from scipy.special import gammainc
from ._sobol import (
    _initialize_v, _cscramble, _fill_p_cumulative, _draw, _fast_forward,
    _categorize, _MAXDIM
)
from ._qmc_cy import (
    _cy_wrapper_centered_discrepancy,
    _cy_wrapper_wrap_around_discrepancy,
    _cy_wrapper_mixture_discrepancy,
    _cy_wrapper_l2_star_discrepancy,
    _cy_wrapper_update_discrepancy,
    _cy_van_der_corput_scrambled,
    _cy_van_der_corput,
)


__all__ = ['scale', 'discrepancy', 'geometric_discrepancy', 'update_discrepancy',
           'QMCEngine', 'Sobol', 'Halton', 'LatinHypercube', 'PoissonDisk',
           'MultinomialQMC', 'MultivariateNormalQMC']


@overload
def check_random_state(seed: IntNumber | None = ...) -> np.random.Generator:
    ...


@overload
def check_random_state(seed: GeneratorType) -> GeneratorType:
    ...


# Based on scipy._lib._util.check_random_state
# This is going to be removed at the end of the SPEC 7 transition,
# so I'll just leave the argument name `seed` alone
def check_random_state(seed=None):
    """Turn `seed` into a `numpy.random.Generator` instance.

    Parameters
    ----------
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is an int or None, a new `numpy.random.Generator` is
        created using ``np.random.default_rng(seed)``.
        If `seed` is already a ``Generator`` or ``RandomState`` instance, then
        the provided instance is used.

    Returns
    -------
    seed : {`numpy.random.Generator`, `numpy.random.RandomState`}
        Random number generator.

    """
    if seed is None or isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.default_rng(seed)
    elif isinstance(seed, (np.random.RandomState, np.random.Generator)):
        return seed
    else:
        raise ValueError(f'{seed!r} cannot be used to seed a'
                         ' numpy.random.Generator instance')


def scale(
    sample: "npt.ArrayLike",
    l_bounds: "npt.ArrayLike",
    u_bounds: "npt.ArrayLike",
    *,
    reverse: bool = False
) -> np.ndarray:
    r"""Sample scaling from unit hypercube to different bounds.

    To convert a sample from :math:`[0, 1)` to :math:`[a, b), b>a`,
    with :math:`a` the lower bounds and :math:`b` the upper bounds.
    The following transformation is used:

    .. math::

        (b - a) \cdot \text{sample} + a

    Parameters
    ----------
    sample : array_like (n, d)
        Sample to scale.
    l_bounds, u_bounds : array_like (d,)
        Lower and upper bounds (resp. :math:`a`, :math:`b`) of transformed
        data. If `reverse` is True, range of the original data to transform
        to the unit hypercube.
    reverse : bool, optional
        Reverse the transformation from different bounds to the unit hypercube.
        Default is False.

    Returns
    -------
    sample : array_like (n, d)
        Scaled sample.

    Examples
    --------
    Transform 3 samples in the unit hypercube to bounds:

    >>> from scipy.stats import qmc
    >>> l_bounds = [-2, 0]
    >>> u_bounds = [6, 5]
    >>> sample = [[0.5 , 0.75],
    ...           [0.5 , 0.5],
    ...           [0.75, 0.25]]
    >>> sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
    >>> sample_scaled
    array([[2.  , 3.75],
           [2.  , 2.5 ],
           [4.  , 1.25]])

    And convert back to the unit hypercube:

    >>> sample_ = qmc.scale(sample_scaled, l_bounds, u_bounds, reverse=True)
    >>> sample_
    array([[0.5 , 0.75],
           [0.5 , 0.5 ],
           [0.75, 0.25]])

    """
    sample = np.asarray(sample)

    # Checking bounds and sample
    if not sample.ndim == 2:
        raise ValueError('Sample is not a 2D array')

    lower, upper = _validate_bounds(
        l_bounds=l_bounds, u_bounds=u_bounds, d=sample.shape[1]
    )

    if not reverse:
        # Checking that sample is within the hypercube
        if (sample.max() > 1.) or (sample.min() < 0.):
            raise ValueError('Sample is not in unit hypercube')

        return sample * (upper - lower) + lower
    else:
        # Checking that sample is within the bounds
        if not (np.all(sample >= lower) and np.all(sample <= upper)):
            raise ValueError('Sample is out of bounds')

        return (sample - lower) / (upper - lower)


def _ensure_in_unit_hypercube(sample: "npt.ArrayLike") -> np.ndarray:
    """Ensure that sample is a 2D array and is within a unit hypercube

    Parameters
    ----------
    sample : array_like (n, d)
        A 2D array of points.

    Returns
    -------
    np.ndarray
        The array interpretation of the input sample

    Raises
    ------
    ValueError
        If the input is not a 2D array or contains points outside of
        a unit hypercube.
    """
    sample = np.asarray(sample, dtype=np.float64, order="C")

    if not sample.ndim == 2:
        raise ValueError("Sample is not a 2D array")

    if (sample.max() > 1.) or (sample.min() < 0.):
        raise ValueError("Sample is not in unit hypercube")

    return sample


def discrepancy(
        sample: "npt.ArrayLike",
        *,
        iterative: bool = False,
        method: Literal["CD", "WD", "MD", "L2-star"] = "CD",
        workers: IntNumber = 1) -> float:
    """Discrepancy of a given sample.

    Parameters
    ----------
    sample : array_like (n, d)
        The sample to compute the discrepancy from.
    iterative : bool, optional
        Must be False if not using it for updating the discrepancy.
        Default is False. Refer to the notes for more details.
    method : str, optional
        Type of discrepancy, can be ``CD``, ``WD``, ``MD`` or ``L2-star``.
        Refer to the notes for more details. Default is ``CD``.
    workers : int, optional
        Number of workers to use for parallel processing. If -1 is given all
        CPU threads are used. Default is 1.

    Returns
    -------
    discrepancy : float
        Discrepancy.

    See Also
    --------
    geometric_discrepancy

    Notes
    -----
    The discrepancy is a uniformity criterion used to assess the space filling
    of a number of samples in a hypercube. A discrepancy quantifies the
    distance between the continuous uniform distribution on a hypercube and the
    discrete uniform distribution on :math:`n` distinct sample points.

    The lower the value is, the better the coverage of the parameter space is.

    For a collection of subsets of the hypercube, the discrepancy is the
    difference between the fraction of sample points in one of those
    subsets and the volume of that subset. There are different definitions of
    discrepancy corresponding to different collections of subsets. Some
    versions take a root mean square difference over subsets instead of
    a maximum.

    A measure of uniformity is reasonable if it satisfies the following
    criteria [1]_:

    1. It is invariant under permuting factors and/or runs.
    2. It is invariant under rotation of the coordinates.
    3. It can measure not only uniformity of the sample over the hypercube,
       but also the projection uniformity of the sample over non-empty
       subset of lower dimension hypercubes.
    4. There is some reasonable geometric meaning.
    5. It is easy to compute.
    6. It satisfies the Koksma-Hlawka-like inequality.
    7. It is consistent with other criteria in experimental design.

    Four methods are available:

    * ``CD``: Centered Discrepancy - subspace involves a corner of the
      hypercube
    * ``WD``: Wrap-around Discrepancy - subspace can wrap around bounds
    * ``MD``: Mixture Discrepancy - mix between CD/WD covering more criteria
    * ``L2-star``: L2-star discrepancy - like CD BUT variant to rotation

    See [2]_ for precise definitions of each method.

    Lastly, using ``iterative=True``, it is possible to compute the
    discrepancy as if we had :math:`n+1` samples. This is useful if we want
    to add a point to a sampling and check the candidate which would give the
    lowest discrepancy. Then you could just update the discrepancy with
    each candidate using `update_discrepancy`. This method is faster than
    computing the discrepancy for a large number of candidates.

    References
    ----------
    .. [1] Fang et al. "Design and modeling for computer experiments".
       Computer Science and Data Analysis Series, 2006.
    .. [2] Zhou Y.-D. et al. "Mixture discrepancy for quasi-random point sets."
       Journal of Complexity, 29 (3-4) , pp. 283-301, 2013.
    .. [3] T. T. Warnock. "Computational investigations of low discrepancy
       point sets." Applications of Number Theory to Numerical
       Analysis, Academic Press, pp. 319-343, 1972.

    Examples
    --------
    Calculate the quality of the sample using the discrepancy:

    >>> import numpy as np
    >>> from scipy.stats import qmc
    >>> space = np.array([[1, 3], [2, 6], [3, 2], [4, 5], [5, 1], [6, 4]])
    >>> l_bounds = [0.5, 0.5]
    >>> u_bounds = [6.5, 6.5]
    >>> space = qmc.scale(space, l_bounds, u_bounds, reverse=True)
    >>> space
    array([[0.08333333, 0.41666667],
           [0.25      , 0.91666667],
           [0.41666667, 0.25      ],
           [0.58333333, 0.75      ],
           [0.75      , 0.08333333],
           [0.91666667, 0.58333333]])
    >>> qmc.discrepancy(space)
    0.008142039609053464

    We can also compute iteratively the ``CD`` discrepancy by using
    ``iterative=True``.

    >>> disc_init = qmc.discrepancy(space[:-1], iterative=True)
    >>> disc_init
    0.04769081147119336
    >>> qmc.update_discrepancy(space[-1], space[:-1], disc_init)
    0.008142039609053513

    """
    sample = _ensure_in_unit_hypercube(sample)

    workers = _validate_workers(workers)

    methods = {
        "CD": _cy_wrapper_centered_discrepancy,
        "WD": _cy_wrapper_wrap_around_discrepancy,
        "MD": _cy_wrapper_mixture_discrepancy,
        "L2-star": _cy_wrapper_l2_star_discrepancy,
    }

    if method in methods:
        return methods[method](sample, iterative, workers=workers)
    else:
        raise ValueError(f"{method!r} is not a valid method. It must be one of"
                         f" {set(methods)!r}")


def geometric_discrepancy(
        sample: "npt.ArrayLike",
        method: Literal["mindist", "mst"] = "mindist",
        metric: str = "euclidean") -> float:
    """Discrepancy of a given sample based on its geometric properties.

    Parameters
    ----------
    sample : array_like (n, d)
        The sample to compute the discrepancy from.
    method : {"mindist", "mst"}, optional
        The method to use. One of ``mindist`` for minimum distance (default)
        or ``mst`` for minimum spanning tree.
    metric : str or callable, optional
        The distance metric to use. See the documentation
        for `scipy.spatial.distance.pdist` for the available metrics and
        the default.

    Returns
    -------
    discrepancy : float
        Discrepancy (higher values correspond to greater sample uniformity).

    See Also
    --------
    discrepancy

    Notes
    -----
    The discrepancy can serve as a simple measure of quality of a random sample.
    This measure is based on the geometric properties of the distribution of points
    in the sample, such as the minimum distance between any pair of points, or
    the mean edge length in a minimum spanning tree.

    The higher the value is, the better the coverage of the parameter space is.
    Note that this is different from `scipy.stats.qmc.discrepancy`, where lower
    values correspond to higher quality of the sample.

    Also note that when comparing different sampling strategies using this function,
    the sample size must be kept constant.

    It is possible to calculate two metrics from the minimum spanning tree:
    the mean edge length and the standard deviation of edges lengths. Using
    both metrics offers a better picture of uniformity than either metric alone,
    with higher mean and lower standard deviation being preferable (see [1]_
    for a brief discussion). This function currently only calculates the mean
    edge length.

    References
    ----------
    .. [1] Franco J. et al. "Minimum Spanning Tree: A new approach to assess the quality
       of the design of computer experiments." Chemometrics and Intelligent Laboratory
       Systems, 97 (2), pp. 164-169, 2009.

    Examples
    --------
    Calculate the quality of the sample using the minimum euclidean distance
    (the defaults):

    >>> import numpy as np
    >>> from scipy.stats import qmc
    >>> rng = np.random.default_rng(191468432622931918890291693003068437394)
    >>> sample = qmc.LatinHypercube(d=2, rng=rng).random(50)
    >>> qmc.geometric_discrepancy(sample)
    0.03708161435687876

    Calculate the quality using the mean edge length in the minimum
    spanning tree:

    >>> qmc.geometric_discrepancy(sample, method='mst')
    0.1105149978798376

    Display the minimum spanning tree and the points with
    the smallest distance:

    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.lines import Line2D
    >>> from scipy.sparse.csgraph import minimum_spanning_tree
    >>> from scipy.spatial.distance import pdist, squareform
    >>> dist = pdist(sample)
    >>> mst = minimum_spanning_tree(squareform(dist))
    >>> edges = np.where(mst.toarray() > 0)
    >>> edges = np.asarray(edges).T
    >>> min_dist = np.min(dist)
    >>> min_idx = np.argwhere(squareform(dist) == min_dist)[0]
    >>> fig, ax = plt.subplots(figsize=(10, 5))
    >>> _ = ax.set(aspect='equal', xlabel=r'$x_1$', ylabel=r'$x_2$',
    ...            xlim=[0, 1], ylim=[0, 1])
    >>> for edge in edges:
    ...     ax.plot(sample[edge, 0], sample[edge, 1], c='k')
    >>> ax.scatter(sample[:, 0], sample[:, 1])
    >>> ax.add_patch(plt.Circle(sample[min_idx[0]], min_dist, color='red', fill=False))
    >>> markers = [
    ...     Line2D([0], [0], marker='o', lw=0, label='Sample points'),
    ...     Line2D([0], [0], color='k', label='Minimum spanning tree'),
    ...     Line2D([0], [0], marker='o', lw=0, markerfacecolor='w', markeredgecolor='r',
    ...            label='Minimum point-to-point distance'),
    ... ]
    >>> ax.legend(handles=markers, loc='center left', bbox_to_anchor=(1, 0.5));
    >>> plt.show()

    """
    sample = _ensure_in_unit_hypercube(sample)
    if sample.shape[0] < 2:
        raise ValueError("Sample must contain at least two points")

    distances = distance.pdist(sample, metric=metric)  # type: ignore[call-overload]

    if np.any(distances == 0.0):
        warnings.warn("Sample contains duplicate points.", stacklevel=2)

    if method == "mindist":
        return np.min(distances[distances.nonzero()])
    elif method == "mst":
        fully_connected_graph = distance.squareform(distances)
        mst = minimum_spanning_tree(fully_connected_graph)
        distances = mst[mst.nonzero()]
        # TODO consider returning both the mean and the standard deviation
        # see [1] for a discussion
        return np.mean(distances)
    else:
        raise ValueError(f"{method!r} is not a valid method. "
                         f"It must be one of {{'mindist', 'mst'}}")


def update_discrepancy(
        x_new: "npt.ArrayLike",
        sample: "npt.ArrayLike",
        initial_disc: DecimalNumber) -> float:
    """Update the centered discrepancy with a new sample.

    Parameters
    ----------
    x_new : array_like (1, d)
        The new sample to add in `sample`.
    sample : array_like (n, d)
        The initial sample.
    initial_disc : float
        Centered discrepancy of the `sample`.

    Returns
    -------
    discrepancy : float
        Centered discrepancy of the sample composed of `x_new` and `sample`.

    Examples
    --------
    We can also compute iteratively the discrepancy by using
    ``iterative=True``.

    >>> import numpy as np
    >>> from scipy.stats import qmc
    >>> space = np.array([[1, 3], [2, 6], [3, 2], [4, 5], [5, 1], [6, 4]])
    >>> l_bounds = [0.5, 0.5]
    >>> u_bounds = [6.5, 6.5]
    >>> space = qmc.scale(space, l_bounds, u_bounds, reverse=True)
    >>> disc_init = qmc.discrepancy(space[:-1], iterative=True)
    >>> disc_init
    0.04769081147119336
    >>> qmc.update_discrepancy(space[-1], space[:-1], disc_init)
    0.008142039609053513

    """
    sample = np.asarray(sample, dtype=np.float64, order="C")
    x_new = np.asarray(x_new, dtype=np.float64, order="C")

    # Checking that sample is within the hypercube and 2D
    if not sample.ndim == 2:
        raise ValueError('Sample is not a 2D array')

    if (sample.max() > 1.) or (sample.min() < 0.):
        raise ValueError('Sample is not in unit hypercube')

    # Checking that x_new is within the hypercube and 1D
    if not x_new.ndim == 1:
        raise ValueError('x_new is not a 1D array')

    if not (np.all(x_new >= 0) and np.all(x_new <= 1)):
        raise ValueError('x_new is not in unit hypercube')

    if x_new.shape[0] != sample.shape[1]:
        raise ValueError("x_new and sample must be broadcastable")

    return _cy_wrapper_update_discrepancy(x_new, sample, initial_disc)


def _perturb_discrepancy(sample: np.ndarray, i1: int, i2: int, k: int,
                         disc: float):
    """Centered discrepancy after an elementary perturbation of a LHS.

    An elementary perturbation consists of an exchange of coordinates between
    two points: ``sample[i1, k] <-> sample[i2, k]``. By construction,
    this operation conserves the LHS properties.

    Parameters
    ----------
    sample : array_like (n, d)
        The sample (before permutation) to compute the discrepancy from.
    i1 : int
        The first line of the elementary permutation.
    i2 : int
        The second line of the elementary permutation.
    k : int
        The column of the elementary permutation.
    disc : float
        Centered discrepancy of the design before permutation.

    Returns
    -------
    discrepancy : float
        Centered discrepancy of the design after permutation.

    References
    ----------
    .. [1] Jin et al. "An efficient algorithm for constructing optimal design
       of computer experiments", Journal of Statistical Planning and
       Inference, 2005.

    """
    n = sample.shape[0]

    z_ij = sample - 0.5

    # Eq (19)
    c_i1j = (1. / n ** 2.
             * np.prod(0.5 * (2. + abs(z_ij[i1, :])
                              + abs(z_ij) - abs(z_ij[i1, :] - z_ij)), axis=1))
    c_i2j = (1. / n ** 2.
             * np.prod(0.5 * (2. + abs(z_ij[i2, :])
                              + abs(z_ij) - abs(z_ij[i2, :] - z_ij)), axis=1))

    # Eq (20)
    c_i1i1 = (1. / n ** 2 * np.prod(1 + abs(z_ij[i1, :]))
              - 2. / n * np.prod(1. + 0.5 * abs(z_ij[i1, :])
                                 - 0.5 * z_ij[i1, :] ** 2))
    c_i2i2 = (1. / n ** 2 * np.prod(1 + abs(z_ij[i2, :]))
              - 2. / n * np.prod(1. + 0.5 * abs(z_ij[i2, :])
                                 - 0.5 * z_ij[i2, :] ** 2))

    # Eq (22), typo in the article in the denominator i2 -> i1
    num = (2 + abs(z_ij[i2, k]) + abs(z_ij[:, k])
           - abs(z_ij[i2, k] - z_ij[:, k]))
    denum = (2 + abs(z_ij[i1, k]) + abs(z_ij[:, k])
             - abs(z_ij[i1, k] - z_ij[:, k]))
    gamma = num / denum

    # Eq (23)
    c_p_i1j = gamma * c_i1j
    # Eq (24)
    c_p_i2j = c_i2j / gamma

    alpha = (1 + abs(z_ij[i2, k])) / (1 + abs(z_ij[i1, k]))
    beta = (2 - abs(z_ij[i2, k])) / (2 - abs(z_ij[i1, k]))

    g_i1 = np.prod(1. + abs(z_ij[i1, :]))
    g_i2 = np.prod(1. + abs(z_ij[i2, :]))
    h_i1 = np.prod(1. + 0.5 * abs(z_ij[i1, :]) - 0.5 * (z_ij[i1, :] ** 2))
    h_i2 = np.prod(1. + 0.5 * abs(z_ij[i2, :]) - 0.5 * (z_ij[i2, :] ** 2))

    # Eq (25), typo in the article g is missing
    c_p_i1i1 = ((g_i1 * alpha) / (n ** 2) - 2. * alpha * beta * h_i1 / n)
    # Eq (26), typo in the article n ** 2
    c_p_i2i2 = ((g_i2 / ((n ** 2) * alpha)) - (2. * h_i2 / (n * alpha * beta)))

    # Eq (26)
    sum_ = c_p_i1j - c_i1j + c_p_i2j - c_i2j

    mask = np.ones(n, dtype=bool)
    mask[[i1, i2]] = False
    sum_ = sum(sum_[mask])

    disc_ep = (disc + c_p_i1i1 - c_i1i1 + c_p_i2i2 - c_i2i2 + 2 * sum_)

    return disc_ep


def primes_from_2_to(n: int) -> np.ndarray:
    """Prime numbers from 2 to *n*.

    Parameters
    ----------
    n : int
        Sup bound with ``n >= 6``.

    Returns
    -------
    primes : list(int)
        Primes in ``2 <= p < n``.

    Notes
    -----
    Taken from [1]_ by P.T. Roy, written consent given on 23.04.2021
    by the original author, Bruno Astrolino, for free use in SciPy under
    the 3-clause BSD.

    References
    ----------
    .. [1] `StackOverflow <https://stackoverflow.com/questions/2068372>`_.

    """
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=bool)
    for i in range(1, int(n ** 0.5) // 3 + 1):
        k = 3 * i + 1 | 1
        sieve[k * k // 3::2 * k] = False
        sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]


def n_primes(n: IntNumber) -> list[int]:
    """List of the n-first prime numbers.

    Parameters
    ----------
    n : int
        Number of prime numbers wanted.

    Returns
    -------
    primes : list(int)
        List of primes.

    """
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
              61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127,
              131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
              197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269,
              271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
              353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431,
              433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
              509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599,
              601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673,
              677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761,
              769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857,
              859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947,
              953, 967, 971, 977, 983, 991, 997][:n]

    if len(primes) < n:
        big_number = 2000
        while 'Not enough primes':
            primes = primes_from_2_to(big_number)[:n]  # type: ignore
            if len(primes) == n:
                break
            big_number += 1000

    return primes


def _van_der_corput_permutations(
    base: IntNumber, *, rng: SeedType = None
) -> np.ndarray:
    """Permutations for scrambling a Van der Corput sequence.

    Parameters
    ----------
    base : int
        Base of the sequence.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.

        .. versionchanged:: 1.15.0

            As part of the `SPEC-007 <https://scientific-python.org/specs/spec-0007/>`_
            transition from use of `numpy.random.RandomState` to
            `numpy.random.Generator`, this keyword was changed from `seed` to
            `rng`. During the transition, the behavior documented above is not
            accurate; see `check_random_state` for actual behavior. After the
            transition, this admonition can be removed.

    Returns
    -------
    permutations : array_like
        Permutation indices.

    Notes
    -----
    In Algorithm 1 of Owen 2017, a permutation of `np.arange(base)` is
    created for each positive integer `k` such that ``1 - base**-k < 1``
    using floating-point arithmetic. For double precision floats, the
    condition ``1 - base**-k < 1`` can also be written as ``base**-k >
    2**-54``, which makes it more apparent how many permutations we need
    to create.
    """
    rng = check_random_state(rng)
    count = math.ceil(54 / math.log2(base)) - 1
    permutations = np.repeat(np.arange(base)[None], count, axis=0)
    for perm in permutations:
        rng.shuffle(perm)

    return permutations


def van_der_corput(
        n: IntNumber,
        base: IntNumber = 2,
        *,
        start_index: IntNumber = 0,
        scramble: bool = False,
        permutations: "npt.ArrayLike | None" = None,
        rng: SeedType = None,
        workers: IntNumber = 1) -> np.ndarray:
    """Van der Corput sequence.

    Pseudo-random number generator based on a b-adic expansion.

    Scrambling uses permutations of the remainders (see [1]_). Multiple
    permutations are applied to construct a point. The sequence of
    permutations has to be the same for all points of the sequence.

    Parameters
    ----------
    n : int
        Number of element of the sequence.
    base : int, optional
        Base of the sequence. Default is 2.
    start_index : int, optional
        Index to start the sequence from. Default is 0.
    scramble : bool, optional
        If True, use Owen scrambling. Otherwise no scrambling is done.
        Default is True.
    permutations : array_like, optional
        Permutations used for scrambling.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.
    workers : int, optional
        Number of workers to use for parallel processing. If -1 is
        given all CPU threads are used. Default is 1.

    Returns
    -------
    sequence : list (n,)
        Sequence of Van der Corput.

    References
    ----------
    .. [1] A. B. Owen. "A randomized Halton algorithm in R",
       :arxiv:`1706.02808`, 2017.

    """
    if base < 2:
        raise ValueError("'base' must be at least 2")

    if scramble:
        if permutations is None:
            permutations = _van_der_corput_permutations(
                base=base, rng=rng
            )
        else:
            permutations = np.asarray(permutations)

        permutations = permutations.astype(np.int64)
        return _cy_van_der_corput_scrambled(n, base, start_index,
                                            permutations, workers)

    else:
        return _cy_van_der_corput(n, base, start_index, workers)


class QMCEngine(ABC):
    """A generic Quasi-Monte Carlo sampler class meant for subclassing.

    QMCEngine is a base class to construct a specific Quasi-Monte Carlo
    sampler. It cannot be used directly as a sampler.

    Parameters
    ----------
    d : int
        Dimension of the parameter space.
    optimization : {None, "random-cd", "lloyd"}, optional
        Whether to use an optimization scheme to improve the quality after
        sampling. Note that this is a post-processing step that does not
        guarantee that all properties of the sample will be conserved.
        Default is None.

        * ``random-cd``: random permutations of coordinates to lower the
          centered discrepancy. The best sample based on the centered
          discrepancy is constantly updated. Centered discrepancy-based
          sampling shows better space-filling robustness toward 2D and 3D
          subprojections compared to using other discrepancy measures.
        * ``lloyd``: Perturb samples using a modified Lloyd-Max algorithm.
          The process converges to equally spaced samples.

        .. versionadded:: 1.10.0

    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.

        .. versionchanged:: 1.15.0

            As part of the `SPEC-007 <https://scientific-python.org/specs/spec-0007/>`_
            transition from use of `numpy.random.RandomState` to
            `numpy.random.Generator`, this keyword was changed from `seed` to
            `rng`. For an interim period, both keywords will continue to work, although
            only one may be specified at a time. After the interim period, function
            calls using the `seed` keyword will emit warnings. Following a
            deprecation period, the `seed` keyword will be removed.

    Notes
    -----
    By convention samples are distributed over the half-open interval
    ``[0, 1)``. Instances of the class can access the attributes: ``d`` for
    the dimension; and ``rng`` for the random number generator.

    **Subclassing**

    When subclassing `QMCEngine` to create a new sampler,  ``__init__`` and
    ``random`` must be redefined.

    * ``__init__(d, rng=None)``: at least fix the dimension. If the sampler
      does not take advantage of a ``rng`` (deterministic methods like
      Halton), this parameter can be omitted.
    * ``_random(n, *, workers=1)``: draw ``n`` from the engine. ``workers``
      is used for parallelism. See `Halton` for example.

    Optionally, two other methods can be overwritten by subclasses:

    * ``reset``: Reset the engine to its original state.
    * ``fast_forward``: If the sequence is deterministic (like Halton
      sequence), then ``fast_forward(n)`` is skipping the ``n`` first draw.

    Examples
    --------
    To create a random sampler based on ``np.random.random``, we would do the
    following:

    >>> from scipy.stats import qmc
    >>> class RandomEngine(qmc.QMCEngine):
    ...     def __init__(self, d, rng=None):
    ...         super().__init__(d=d, rng=rng)
    ...
    ...
    ...     def _random(self, n=1, *, workers=1):
    ...         return self.rng.random((n, self.d))
    ...
    ...
    ...     def reset(self):
    ...         super().__init__(d=self.d, rng=self.rng_seed)
    ...         return self
    ...
    ...
    ...     def fast_forward(self, n):
    ...         self.random(n)
    ...         return self

    After subclassing `QMCEngine` to define the sampling strategy we want to
    use, we can create an instance to sample from.

    >>> engine = RandomEngine(2)
    >>> engine.random(5)
    array([[0.22733602, 0.31675834],  # random
           [0.79736546, 0.67625467],
           [0.39110955, 0.33281393],
           [0.59830875, 0.18673419],
           [0.67275604, 0.94180287]])

    We can also reset the state of the generator and resample again.

    >>> _ = engine.reset()
    >>> engine.random(5)
    array([[0.22733602, 0.31675834],  # random
           [0.79736546, 0.67625467],
           [0.39110955, 0.33281393],
           [0.59830875, 0.18673419],
           [0.67275604, 0.94180287]])

    """

    @abstractmethod
    @_transition_to_rng('seed', replace_doc=False)
    def __init__(
        self,
        d: IntNumber,
        *,
        optimization: Literal["random-cd", "lloyd"] | None = None,
        rng: SeedType = None
    ) -> None:
        self._initialize(d, optimization=optimization, rng=rng)

    # During SPEC 7 transition:
    # `__init__` has to be wrapped with @_transition_to_rng decorator
    # because it is public. Subclasses previously called `__init__`
    # directly, but this was problematic because arguments passed to
    # subclass `__init__` as `seed` would get passed to superclass
    # `__init__` as `rng`, rejecting `RandomState` arguments.
    def _initialize(
        self,
        d: IntNumber,
        *,
        optimization: Literal["random-cd", "lloyd"] | None = None,
        rng: SeedType = None
    ) -> None:
        if not np.issubdtype(type(d), np.integer) or d < 0:
            raise ValueError('d must be a non-negative integer value')

        self.d = d

        if isinstance(rng, np.random.Generator):
            # Spawn a Generator that we can own and reset.
            self.rng = _rng_spawn(rng, 1)[0]
        else:
            # Create our instance of Generator, does not need spawning
            # Also catch RandomState which cannot be spawned
            self.rng = check_random_state(rng)
        self.rng_seed = copy.deepcopy(self.rng)

        self.num_generated = 0

        config = {
            # random-cd
            "n_nochange": 100,
            "n_iters": 10_000,
            "rng": self.rng,

            # lloyd
            "tol": 1e-5,
            "maxiter": 10,
            "qhull_options": None,
        }
        self._optimization = optimization
        self.optimization_method = _select_optimizer(optimization, config)

    @abstractmethod
    def _random(
        self, n: IntNumber = 1, *, workers: IntNumber = 1
    ) -> np.ndarray:
        ...

    def random(
        self, n: IntNumber = 1, *, workers: IntNumber = 1
    ) -> np.ndarray:
        """Draw `n` in the half-open interval ``[0, 1)``.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space.
            Default is 1.
        workers : int, optional
            Only supported with `Halton`.
            Number of workers to use for parallel processing. If -1 is
            given all CPU threads are used. Default is 1. It becomes faster
            than one worker for `n` greater than :math:`10^3`.

        Returns
        -------
        sample : array_like (n, d)
            QMC sample.

        """
        sample = self._random(n, workers=workers)
        if self.optimization_method is not None:
            sample = self.optimization_method(sample)

        self.num_generated += n
        return sample

    def integers(
        self,
        l_bounds: "npt.ArrayLike",
        *,
        u_bounds: "npt.ArrayLike | None" = None,
        n: IntNumber = 1,
        endpoint: bool = False,
        workers: IntNumber = 1
    ) -> np.ndarray:
        r"""
        Draw `n` integers from `l_bounds` (inclusive) to `u_bounds`
        (exclusive), or if endpoint=True, `l_bounds` (inclusive) to
        `u_bounds` (inclusive).

        Parameters
        ----------
        l_bounds : int or array-like of ints
            Lowest (signed) integers to be drawn (unless ``u_bounds=None``,
            in which case this parameter is 0 and this value is used for
            `u_bounds`).
        u_bounds : int or array-like of ints, optional
            If provided, one above the largest (signed) integer to be drawn
            (see above for behavior if ``u_bounds=None``).
            If array-like, must contain integer values.
        n : int, optional
            Number of samples to generate in the parameter space.
            Default is 1.
        endpoint : bool, optional
            If true, sample from the interval ``[l_bounds, u_bounds]`` instead
            of the default ``[l_bounds, u_bounds)``. Defaults is False.
        workers : int, optional
            Number of workers to use for parallel processing. If -1 is
            given all CPU threads are used. Only supported when using `Halton`
            Default is 1.

        Returns
        -------
        sample : array_like (n, d)
            QMC sample.

        Notes
        -----
        It is safe to just use the same ``[0, 1)`` to integer mapping
        with QMC that you would use with MC. You still get unbiasedness,
        a strong law of large numbers, an asymptotically infinite variance
        reduction and a finite sample variance bound.

        To convert a sample from :math:`[0, 1)` to :math:`[a, b), b>a`,
        with :math:`a` the lower bounds and :math:`b` the upper bounds,
        the following transformation is used:

        .. math::

            \text{floor}((b - a) \cdot \text{sample} + a)

        """
        if u_bounds is None:
            u_bounds = l_bounds
            l_bounds = 0

        u_bounds = np.atleast_1d(u_bounds)
        l_bounds = np.atleast_1d(l_bounds)

        if endpoint:
            u_bounds = u_bounds + 1

        if (not np.issubdtype(l_bounds.dtype, np.integer) or
                not np.issubdtype(u_bounds.dtype, np.integer)):
            message = ("'u_bounds' and 'l_bounds' must be integers or"
                       " array-like of integers")
            raise ValueError(message)

        if isinstance(self, Halton):
            sample = self.random(n=n, workers=workers)
        else:
            sample = self.random(n=n)

        sample = scale(sample, l_bounds=l_bounds, u_bounds=u_bounds)
        sample = np.floor(sample).astype(np.int64)

        return sample

    def reset(self) -> "QMCEngine":
        """Reset the engine to base state.

        Returns
        -------
        engine : QMCEngine
            Engine reset to its base state.

        """
        rng = copy.deepcopy(self.rng_seed)
        self.rng = check_random_state(rng)
        self.num_generated = 0
        return self

    def fast_forward(self, n: IntNumber) -> "QMCEngine":
        """Fast-forward the sequence by `n` positions.

        Parameters
        ----------
        n : int
            Number of points to skip in the sequence.

        Returns
        -------
        engine : QMCEngine
            Engine reset to its base state.

        """
        self.random(n=n)
        return self


class Halton(QMCEngine):
    """Halton sequence.

    Pseudo-random number generator that generalize the Van der Corput sequence
    for multiple dimensions. The Halton sequence uses the base-two Van der
    Corput sequence for the first dimension, base-three for its second and
    base-:math:`n` for its n-dimension.

    Parameters
    ----------
    d : int
        Dimension of the parameter space.
    scramble : bool, optional
        If True, use Owen scrambling. Otherwise no scrambling is done.
        Default is True.
    optimization : {None, "random-cd", "lloyd"}, optional
        Whether to use an optimization scheme to improve the quality after
        sampling. Note that this is a post-processing step that does not
        guarantee that all properties of the sample will be conserved.
        Default is None.

        * ``random-cd``: random permutations of coordinates to lower the
          centered discrepancy. The best sample based on the centered
          discrepancy is constantly updated. Centered discrepancy-based
          sampling shows better space-filling robustness toward 2D and 3D
          subprojections compared to using other discrepancy measures.
        * ``lloyd``: Perturb samples using a modified Lloyd-Max algorithm.
          The process converges to equally spaced samples.

        .. versionadded:: 1.10.0

    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.

        .. versionchanged:: 1.15.0

            As part of the `SPEC-007 <https://scientific-python.org/specs/spec-0007/>`_
            transition from use of `numpy.random.RandomState` to
            `numpy.random.Generator`, this keyword was changed from `seed` to
            `rng`. For an interim period, both keywords will continue to work, although
            only one may be specified at a time. After the interim period, function
            calls using the `seed` keyword will emit warnings. Following a
            deprecation period, the `seed` keyword will be removed.

    Notes
    -----
    The Halton sequence has severe striping artifacts for even modestly
    large dimensions. These can be ameliorated by scrambling. Scrambling
    also supports replication-based error estimates and extends
    applicability to unbounded integrands.

    References
    ----------
    .. [1] Halton, "On the efficiency of certain quasi-random sequences of
       points in evaluating multi-dimensional integrals", Numerische
       Mathematik, 1960.
    .. [2] A. B. Owen. "A randomized Halton algorithm in R",
       :arxiv:`1706.02808`, 2017.

    Examples
    --------
    Generate samples from a low discrepancy sequence of Halton.

    >>> from scipy.stats import qmc
    >>> sampler = qmc.Halton(d=2, scramble=False)
    >>> sample = sampler.random(n=5)
    >>> sample
    array([[0.        , 0.        ],
           [0.5       , 0.33333333],
           [0.25      , 0.66666667],
           [0.75      , 0.11111111],
           [0.125     , 0.44444444]])

    Compute the quality of the sample using the discrepancy criterion.

    >>> qmc.discrepancy(sample)
    0.088893711419753

    If some wants to continue an existing design, extra points can be obtained
    by calling again `random`. Alternatively, you can skip some points like:

    >>> _ = sampler.fast_forward(5)
    >>> sample_continued = sampler.random(n=5)
    >>> sample_continued
    array([[0.3125    , 0.37037037],
           [0.8125    , 0.7037037 ],
           [0.1875    , 0.14814815],
           [0.6875    , 0.48148148],
           [0.4375    , 0.81481481]])

    Finally, samples can be scaled to bounds.

    >>> l_bounds = [0, 2]
    >>> u_bounds = [10, 5]
    >>> qmc.scale(sample_continued, l_bounds, u_bounds)
    array([[3.125     , 3.11111111],
           [8.125     , 4.11111111],
           [1.875     , 2.44444444],
           [6.875     , 3.44444444],
           [4.375     , 4.44444444]])

    """
    @_transition_to_rng('seed', replace_doc=False)
    def __init__(
        self, d: IntNumber, *, scramble: bool = True,
        optimization: Literal["random-cd", "lloyd"] | None = None,
        rng: SeedType = None
    ) -> None:
        # Used in `scipy.integrate.qmc_quad`
        self._init_quad = {'d': d, 'scramble': True,
                           'optimization': optimization}
        super()._initialize(d=d, optimization=optimization, rng=rng)

        # important to have ``type(bdim) == int`` for performance reason
        self.base = [int(bdim) for bdim in n_primes(d)]
        self.scramble = scramble

        self._initialize_permutations()

    def _initialize_permutations(self) -> None:
        """Initialize permutations for all Van der Corput sequences.

        Permutations are only needed for scrambling.
        """
        self._permutations: list = [None] * len(self.base)
        if self.scramble:
            for i, bdim in enumerate(self.base):
                permutations = _van_der_corput_permutations(
                    base=bdim, rng=self.rng
                )

                self._permutations[i] = permutations

    def _random(
        self, n: IntNumber = 1, *, workers: IntNumber = 1
    ) -> np.ndarray:
        """Draw `n` in the half-open interval ``[0, 1)``.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.
        workers : int, optional
            Number of workers to use for parallel processing. If -1 is
            given all CPU threads are used. Default is 1. It becomes faster
            than one worker for `n` greater than :math:`10^3`.

        Returns
        -------
        sample : array_like (n, d)
            QMC sample.

        """
        workers = _validate_workers(workers)
        # Generate a sample using a Van der Corput sequence per dimension.
        sample = [van_der_corput(n, bdim, start_index=self.num_generated,
                                 scramble=self.scramble,
                                 permutations=self._permutations[i],
                                 workers=workers)
                  for i, bdim in enumerate(self.base)]

        return np.array(sample).T.reshape(n, self.d)


class LatinHypercube(QMCEngine):
    r"""Latin hypercube sampling (LHS).

    A Latin hypercube sample [1]_ generates :math:`n` points in
    :math:`[0,1)^{d}`. Each univariate marginal distribution is stratified,
    placing exactly one point in :math:`[j/n, (j+1)/n)` for
    :math:`j=0,1,...,n-1`. They are still applicable when :math:`n << d`.

    Parameters
    ----------
    d : int
        Dimension of the parameter space.
    scramble : bool, optional
        When False, center samples within cells of a multi-dimensional grid.
        Otherwise, samples are randomly placed within cells of the grid.

        .. note::
            Setting ``scramble=False`` does not ensure deterministic output.
            For that, use the `rng` parameter.

        Default is True.

        .. versionadded:: 1.10.0

    optimization : {None, "random-cd", "lloyd"}, optional
        Whether to use an optimization scheme to improve the quality after
        sampling. Note that this is a post-processing step that does not
        guarantee that all properties of the sample will be conserved.
        Default is None.

        * ``random-cd``: random permutations of coordinates to lower the
          centered discrepancy. The best sample based on the centered
          discrepancy is constantly updated. Centered discrepancy-based
          sampling shows better space-filling robustness toward 2D and 3D
          subprojections compared to using other discrepancy measures.
        * ``lloyd``: Perturb samples using a modified Lloyd-Max algorithm.
          The process converges to equally spaced samples.

        .. versionadded:: 1.8.0
        .. versionchanged:: 1.10.0
            Add ``lloyd``.

    strength : {1, 2}, optional
        Strength of the LHS. ``strength=1`` produces a plain LHS while
        ``strength=2`` produces an orthogonal array based LHS of strength 2
        [7]_, [8]_. In that case, only ``n=p**2`` points can be sampled,
        with ``p`` a prime number. It also constrains ``d <= p + 1``.
        Default is 1.

        .. versionadded:: 1.8.0

    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.

        .. versionchanged:: 1.15.0

            As part of the `SPEC-007 <https://scientific-python.org/specs/spec-0007/>`_
            transition from use of `numpy.random.RandomState` to
            `numpy.random.Generator`, this keyword was changed from `seed` to
            `rng`. For an interim period, both keywords will continue to work, although
            only one may be specified at a time. After the interim period, function
            calls using the `seed` keyword will emit warnings. Following a
            deprecation period, the `seed` keyword will be removed.

    See Also
    --------
    :ref:`quasi-monte-carlo`

    Notes
    -----

    When LHS is used for integrating a function :math:`f` over :math:`n`,
    LHS is extremely effective on integrands that are nearly additive [2]_.
    With a LHS of :math:`n` points, the variance of the integral is always
    lower than plain MC on :math:`n-1` points [3]_. There is a central limit
    theorem for LHS on the mean and variance of the integral [4]_, but not
    necessarily for optimized LHS due to the randomization.

    :math:`A` is called an orthogonal array of strength :math:`t` if in each
    n-row-by-t-column submatrix of :math:`A`: all :math:`p^t` possible
    distinct rows occur the same number of times. The elements of :math:`A`
    are in the set :math:`\{0, 1, ..., p-1\}`, also called symbols.
    The constraint that :math:`p` must be a prime number is to allow modular
    arithmetic. Increasing strength adds some symmetry to the sub-projections
    of a sample. With strength 2, samples are symmetric along the diagonals of
    2D sub-projections. This may be undesirable, but on the other hand, the
    sample dispersion is improved.

    Strength 1 (plain LHS) brings an advantage over strength 0 (MC) and
    strength 2 is a useful increment over strength 1. Going to strength 3 is
    a smaller increment and scrambled QMC like Sobol', Halton are more
    performant [7]_.

    To create a LHS of strength 2, the orthogonal array :math:`A` is
    randomized by applying a random, bijective map of the set of symbols onto
    itself. For example, in column 0, all 0s might become 2; in column 1,
    all 0s might become 1, etc.
    Then, for each column :math:`i` and symbol :math:`j`, we add a plain,
    one-dimensional LHS of size :math:`p` to the subarray where
    :math:`A^i = j`. The resulting matrix is finally divided by :math:`p`.

    References
    ----------
    .. [1] Mckay et al., "A Comparison of Three Methods for Selecting Values
       of Input Variables in the Analysis of Output from a Computer Code."
       Technometrics, 1979.
    .. [2] M. Stein, "Large sample properties of simulations using Latin
       hypercube sampling." Technometrics 29, no. 2: 143-151, 1987.
    .. [3] A. B. Owen, "Monte Carlo variance of scrambled net quadrature."
       SIAM Journal on Numerical Analysis 34, no. 5: 1884-1910, 1997
    .. [4]  Loh, W.-L. "On Latin hypercube sampling." The annals of statistics
       24, no. 5: 2058-2080, 1996.
    .. [5] Fang et al. "Design and modeling for computer experiments".
       Computer Science and Data Analysis Series, 2006.
    .. [6] Damblin et al., "Numerical studies of space filling designs:
       optimization of Latin Hypercube Samples and subprojection properties."
       Journal of Simulation, 2013.
    .. [7] A. B. Owen , "Orthogonal arrays for computer experiments,
       integration and visualization." Statistica Sinica, 1992.
    .. [8] B. Tang, "Orthogonal Array-Based Latin Hypercubes."
       Journal of the American Statistical Association, 1993.
    .. [9] Seaholm, Susan K. et al. (1988). Latin hypercube sampling and the
       sensitivity analysis of a Monte Carlo epidemic model. Int J Biomed
       Comput, 23(1-2), 97-112. :doi:`10.1016/0020-7101(88)90067-0`

    Examples
    --------
    Generate samples from a Latin hypercube generator.

    >>> from scipy.stats import qmc
    >>> sampler = qmc.LatinHypercube(d=2)
    >>> sample = sampler.random(n=5)
    >>> sample
    array([[0.1545328 , 0.53664833], # random
            [0.84052691, 0.06474907],
            [0.52177809, 0.93343721],
            [0.68033825, 0.36265316],
            [0.26544879, 0.61163943]])

    Compute the quality of the sample using the discrepancy criterion.

    >>> qmc.discrepancy(sample)
    0.0196... # random

    Samples can be scaled to bounds.

    >>> l_bounds = [0, 2]
    >>> u_bounds = [10, 5]
    >>> qmc.scale(sample, l_bounds, u_bounds)
    array([[1.54532796, 3.609945 ], # random
            [8.40526909, 2.1942472 ],
            [5.2177809 , 4.80031164],
            [6.80338249, 3.08795949],
            [2.65448791, 3.83491828]])

    Below are other examples showing alternative ways to construct LHS with
    even better coverage of the space.

    Using a base LHS as a baseline.

    >>> sampler = qmc.LatinHypercube(d=2)
    >>> sample = sampler.random(n=5)
    >>> qmc.discrepancy(sample)
    0.0196...  # random

    Use the `optimization` keyword argument to produce a LHS with
    lower discrepancy at higher computational cost.

    >>> sampler = qmc.LatinHypercube(d=2, optimization="random-cd")
    >>> sample = sampler.random(n=5)
    >>> qmc.discrepancy(sample)
    0.0176...  # random

    Use the `strength` keyword argument to produce an orthogonal array based
    LHS of strength 2. In this case, the number of sample points must be the
    square of a prime number.

    >>> sampler = qmc.LatinHypercube(d=2, strength=2)
    >>> sample = sampler.random(n=9)
    >>> qmc.discrepancy(sample)
    0.00526...  # random

    Options could be combined to produce an optimized centered
    orthogonal array based LHS. After optimization, the result would not
    be guaranteed to be of strength 2.

    **Real-world example**

    In [9]_, a Latin Hypercube sampling (LHS) strategy was used to sample a
    parameter space to study the importance of each parameter of an epidemic
    model. Such analysis is also called a sensitivity analysis.

    Since the dimensionality of the problem is high (6), it is computationally
    expensive to cover the space. When numerical experiments are costly, QMC
    enables analysis that may not be possible if using a grid.

    The six parameters of the model represented the probability of illness,
    the probability of withdrawal, and four contact probabilities. The
    authors assumed uniform distributions for all parameters and generated
    50 samples.

    Using `scipy.stats.qmc.LatinHypercube` to replicate the protocol,
    the first step is to create a sample in the unit hypercube:

    >>> from scipy.stats import qmc
    >>> sampler = qmc.LatinHypercube(d=6)
    >>> sample = sampler.random(n=50)

    Then the sample can be scaled to the appropriate bounds:

    >>> l_bounds = [0.000125, 0.01, 0.0025, 0.05, 0.47, 0.7]
    >>> u_bounds = [0.000375, 0.03, 0.0075, 0.15, 0.87, 0.9]
    >>> sample_scaled = qmc.scale(sample, l_bounds, u_bounds)

    Such a sample was used to run the model 50 times, and a polynomial
    response surface was constructed. This allowed the authors to study the
    relative importance of each parameter across the range of possibilities
    of every other parameter.

    In this computer experiment, they showed a 14-fold reduction in the
    number of samples required to maintain an error below 2% on their
    response surface when compared to a grid sampling.

    """

    @_transition_to_rng('seed', replace_doc=False)
    def __init__(
        self, d: IntNumber, *,
        scramble: bool = True,
        strength: int = 1,
        optimization: Literal["random-cd", "lloyd"] | None = None,
        rng: SeedType = None
    ) -> None:
        # Used in `scipy.integrate.qmc_quad`
        self._init_quad = {'d': d, 'scramble': True, 'strength': strength,
                           'optimization': optimization}
        super()._initialize(d=d, rng=rng, optimization=optimization)
        self.scramble = scramble

        lhs_method_strength = {
            1: self._random_lhs,
            2: self._random_oa_lhs
        }

        try:
            self.lhs_method: Callable = lhs_method_strength[strength]
        except KeyError as exc:
            message = (f"{strength!r} is not a valid strength. It must be one"
                       f" of {set(lhs_method_strength)!r}")
            raise ValueError(message) from exc

    def _random(
        self, n: IntNumber = 1, *, workers: IntNumber = 1
    ) -> np.ndarray:
        lhs = self.lhs_method(n)
        return lhs

    def _random_lhs(self, n: IntNumber = 1) -> np.ndarray:
        """Base LHS algorithm."""
        if not self.scramble:
            samples: np.ndarray | float = 0.5
        else:
            samples = self.rng.uniform(size=(n, self.d))

        perms = np.tile(np.arange(1, n + 1),
                        (self.d, 1))  # type: ignore[arg-type]
        for i in range(self.d):
            self.rng.shuffle(perms[i, :])
        perms = perms.T

        samples = (perms - samples) / n
        return samples

    def _random_oa_lhs(self, n: IntNumber = 4) -> np.ndarray:
        """Orthogonal array based LHS of strength 2."""
        p = np.sqrt(n).astype(int)
        n_row = p**2
        n_col = p + 1

        primes = primes_from_2_to(p + 1)
        if p not in primes or n != n_row:
            raise ValueError(
                "n is not the square of a prime number. Close"
                f" values are {primes[-2:]**2}"
            )
        if self.d > p + 1:
            raise ValueError("n is too small for d. Must be n > (d-1)**2")

        oa_sample = np.zeros(shape=(n_row, n_col), dtype=int)

        # OA of strength 2
        arrays = np.tile(np.arange(p), (2, 1))
        oa_sample[:, :2] = np.stack(np.meshgrid(*arrays),
                                    axis=-1).reshape(-1, 2)
        for p_ in range(1, p):
            oa_sample[:, 2+p_-1] = np.mod(oa_sample[:, 0]
                                          + p_*oa_sample[:, 1], p)

        # scramble the OA
        oa_sample_ = np.empty(shape=(n_row, n_col), dtype=int)
        for j in range(n_col):
            perms = self.rng.permutation(p)
            oa_sample_[:, j] = perms[oa_sample[:, j]]
        
        oa_sample = oa_sample_
        # following is making a scrambled OA into an OA-LHS
        oa_lhs_sample = np.zeros(shape=(n_row, n_col))
        lhs_engine = LatinHypercube(d=1, scramble=self.scramble, strength=1,
                                    rng=self.rng)  # type: QMCEngine
        for j in range(n_col):
            for k in range(p):
                idx = oa_sample[:, j] == k
                lhs = lhs_engine.random(p).flatten()
                oa_lhs_sample[:, j][idx] = lhs + oa_sample[:, j][idx]

        oa_lhs_sample /= p

        return oa_lhs_sample[:, :self.d]


class Sobol(QMCEngine):
    """Engine for generating (scrambled) Sobol' sequences.

    Sobol' sequences are low-discrepancy, quasi-random numbers. Points
    can be drawn using two methods:

    * `random_base2`: safely draw :math:`n=2^m` points. This method
      guarantees the balance properties of the sequence.
    * `random`: draw an arbitrary number of points from the
      sequence. See warning below.

    Parameters
    ----------
    d : int
        Dimensionality of the sequence. Max dimensionality is 21201.
    scramble : bool, optional
        If True, use LMS+shift scrambling. Otherwise, no scrambling is done.
        Default is True.
    bits : int, optional
        Number of bits of the generator. Control the maximum number of points
        that can be generated, which is ``2**bits``. Maximal value is 64.
        It does not correspond to the return type, which is always
        ``np.float64`` to prevent points from repeating themselves.
        Default is None, which for backward compatibility, corresponds to 30.

        .. versionadded:: 1.9.0
    optimization : {None, "random-cd", "lloyd"}, optional
        Whether to use an optimization scheme to improve the quality after
        sampling. Note that this is a post-processing step that does not
        guarantee that all properties of the sample will be conserved.
        Default is None.

        * ``random-cd``: random permutations of coordinates to lower the
          centered discrepancy. The best sample based on the centered
          discrepancy is constantly updated. Centered discrepancy-based
          sampling shows better space-filling robustness toward 2D and 3D
          subprojections compared to using other discrepancy measures.
        * ``lloyd``: Perturb samples using a modified Lloyd-Max algorithm.
          The process converges to equally spaced samples.

        .. versionadded:: 1.10.0

    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.

        .. versionchanged:: 1.15.0

            As part of the `SPEC-007 <https://scientific-python.org/specs/spec-0007/>`_
            transition from use of `numpy.random.RandomState` to
            `numpy.random.Generator`, this keyword was changed from `seed` to
            `rng`. For an interim period, both keywords will continue to work, although
            only one may be specified at a time. After the interim period, function
            calls using the `seed` keyword will emit warnings. Following a
            deprecation period, the `seed` keyword will be removed.

    Notes
    -----
    Sobol' sequences [1]_ provide :math:`n=2^m` low discrepancy points in
    :math:`[0,1)^{d}`. Scrambling them [3]_ makes them suitable for singular
    integrands, provides a means of error estimation, and can improve their
    rate of convergence. The scrambling strategy which is implemented is a
    (left) linear matrix scramble (LMS) followed by a digital random shift
    (LMS+shift) [2]_.

    There are many versions of Sobol' sequences depending on their
    'direction numbers'. This code uses direction numbers from [4]_. Hence,
    the maximum number of dimension is 21201. The direction numbers have been
    precomputed with search criterion 6 and can be retrieved at
    https://web.maths.unsw.edu.au/~fkuo/sobol/.

    .. warning::

       Sobol' sequences are a quadrature rule and they lose their balance
       properties if one uses a sample size that is not a power of 2, or skips
       the first point, or thins the sequence [5]_.

       If :math:`n=2^m` points are not enough then one should take :math:`2^M`
       points for :math:`M>m`. When scrambling, the number R of independent
       replicates does not have to be a power of 2.

       Sobol' sequences are generated to some number :math:`B` of bits.
       After :math:`2^B` points have been generated, the sequence would
       repeat. Hence, an error is raised.
       The number of bits can be controlled with the parameter `bits`.

    References
    ----------
    .. [1] I. M. Sobol', "The distribution of points in a cube and the accurate
       evaluation of integrals." Zh. Vychisl. Mat. i Mat. Phys., 7:784-802,
       1967.
    .. [2] J. Matousek, "On the L2-discrepancy for anchored boxes."
       J. of Complexity 14, 527-556, 1998.
    .. [3] Art B. Owen, "Scrambling Sobol and Niederreiter-Xing points."
       Journal of Complexity, 14(4):466-489, December 1998.
    .. [4] S. Joe and F. Y. Kuo, "Constructing sobol sequences with better
       two-dimensional projections." SIAM Journal on Scientific Computing,
       30(5):2635-2654, 2008.
    .. [5] Art B. Owen, "On dropping the first Sobol' point."
       :arxiv:`2008.08051`, 2020.

    Examples
    --------
    Generate samples from a low discrepancy sequence of Sobol'.

    >>> from scipy.stats import qmc
    >>> sampler = qmc.Sobol(d=2, scramble=False)
    >>> sample = sampler.random_base2(m=3)
    >>> sample
    array([[0.   , 0.   ],
           [0.5  , 0.5  ],
           [0.75 , 0.25 ],
           [0.25 , 0.75 ],
           [0.375, 0.375],
           [0.875, 0.875],
           [0.625, 0.125],
           [0.125, 0.625]])

    Compute the quality of the sample using the discrepancy criterion.

    >>> qmc.discrepancy(sample)
    0.013882107204860938

    To continue an existing design, extra points can be obtained
    by calling again `random_base2`. Alternatively, you can skip some
    points like:

    >>> _ = sampler.reset()
    >>> _ = sampler.fast_forward(4)
    >>> sample_continued = sampler.random_base2(m=2)
    >>> sample_continued
    array([[0.375, 0.375],
           [0.875, 0.875],
           [0.625, 0.125],
           [0.125, 0.625]])

    Finally, samples can be scaled to bounds.

    >>> l_bounds = [0, 2]
    >>> u_bounds = [10, 5]
    >>> qmc.scale(sample_continued, l_bounds, u_bounds)
    array([[3.75 , 3.125],
           [8.75 , 4.625],
           [6.25 , 2.375],
           [1.25 , 3.875]])

    """

    MAXDIM: ClassVar[int] = _MAXDIM

    @_transition_to_rng('seed', replace_doc=False)
    def __init__(
        self, d: IntNumber, *, scramble: bool = True,
        bits: IntNumber | None = None, rng: SeedType = None,
        optimization: Literal["random-cd", "lloyd"] | None = None
    ) -> None:
        # Used in `scipy.integrate.qmc_quad`
        self._init_quad = {'d': d, 'scramble': True, 'bits': bits,
                           'optimization': optimization}

        super()._initialize(d=d, optimization=optimization, rng=rng)
        if d > self.MAXDIM:
            raise ValueError(
                f"Maximum supported dimensionality is {self.MAXDIM}."
            )

        self.bits = bits
        self.dtype_i: type
        self.scramble = scramble

        if self.bits is None:
            self.bits = 30

        if self.bits <= 32:
            self.dtype_i = np.uint32
        elif 32 < self.bits <= 64:
            self.dtype_i = np.uint64
        else:
            raise ValueError("Maximum supported 'bits' is 64")

        self.maxn = 2**self.bits

        # v is d x maxbit matrix
        self._sv: np.ndarray = np.zeros((d, self.bits), dtype=self.dtype_i)
        _initialize_v(self._sv, dim=d, bits=self.bits)

        if not scramble:
            self._shift: np.ndarray = np.zeros(d, dtype=self.dtype_i)
        else:
            # scramble self._shift and self._sv
            self._scramble()

        self._quasi = self._shift.copy()

        # normalization constant with the largest possible number
        # calculate in Python to not overflow int with 2**64
        self._scale = 1.0 / 2 ** self.bits

        self._first_point = (self._quasi * self._scale).reshape(1, -1)
        # explicit casting to float64
        self._first_point = self._first_point.astype(np.float64)

    def _scramble(self) -> None:
        """Scramble the sequence using LMS+shift."""
        # Generate shift vector
        self._shift = np.dot(
            rng_integers(self.rng, 2, size=(self.d, self.bits),
                         dtype=self.dtype_i),
            2 ** np.arange(self.bits, dtype=self.dtype_i),
        )
        # Generate lower triangular matrices (stacked across dimensions)
        ltm = np.tril(rng_integers(self.rng, 2,
                                   size=(self.d, self.bits, self.bits),
                                   dtype=self.dtype_i))
        _cscramble(
            dim=self.d, bits=self.bits,  # type: ignore[arg-type]
            ltm=ltm, sv=self._sv
        )

    def _random(
        self, n: IntNumber = 1, *, workers: IntNumber = 1
    ) -> np.ndarray:
        """Draw next point(s) in the Sobol' sequence.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.

        Returns
        -------
        sample : array_like (n, d)
            Sobol' sample.

        """
        sample: np.ndarray = np.empty((n, self.d), dtype=np.float64)

        if n == 0:
            return sample

        total_n = self.num_generated + n
        if total_n > self.maxn:
            msg = (
                f"At most 2**{self.bits}={self.maxn} distinct points can be "
                f"generated. {self.num_generated} points have been previously "
                f"generated, then: n={self.num_generated}+{n}={total_n}. "
            )
            if self.bits != 64:
                msg += "Consider increasing `bits`."
            raise ValueError(msg)

        if self.num_generated == 0:
            # verify n is 2**n
            if not (n & (n - 1) == 0):
                warnings.warn("The balance properties of Sobol' points require"
                              " n to be a power of 2.", stacklevel=2)

            if n == 1:
                sample = self._first_point
            else:
                _draw(
                    n=n - 1, num_gen=self.num_generated, dim=self.d,
                    scale=self._scale, sv=self._sv, quasi=self._quasi,
                    sample=sample
                )
                sample = np.concatenate(
                    [self._first_point, sample]
                )[:n]
        else:
            _draw(
                n=n, num_gen=self.num_generated - 1, dim=self.d,
                scale=self._scale, sv=self._sv, quasi=self._quasi,
                sample=sample
            )

        return sample

    def random_base2(self, m: IntNumber) -> np.ndarray:
        """Draw point(s) from the Sobol' sequence.

        This function draws :math:`n=2^m` points in the parameter space
        ensuring the balance properties of the sequence.

        Parameters
        ----------
        m : int
            Logarithm in base 2 of the number of samples; i.e., n = 2^m.

        Returns
        -------
        sample : array_like (n, d)
            Sobol' sample.

        """
        n = 2 ** m

        total_n = self.num_generated + n
        if not (total_n & (total_n - 1) == 0):
            raise ValueError('The balance properties of Sobol\' points require '
                             f'n to be a power of 2. {self.num_generated} points '
                             'have been previously generated, then: '
                             f'n={self.num_generated}+2**{m}={total_n}. '
                             'If you still want to do this, the function '
                             '\'Sobol.random()\' can be used.'
                             )

        return self.random(n)

    def reset(self) -> "Sobol":
