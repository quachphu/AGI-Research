    family `dist` are fit to *each resample*, and the `statistic` is computed
    between each sample and its corresponding fitted distribution. These
    values of the statistic form the Monte Carlo null distribution (not to be
    confused with the "null-hypothesized distribution" above).

    The p-value of the test is the proportion of statistic values in the Monte
    Carlo null distribution that are at least as extreme as the statistic value
    of the provided `data`. More precisely, the p-value is given by

    .. math::

        p = \frac{b + 1}
                 {m + 1}

    where :math:`b` is the number of statistic values in the Monte Carlo null
    distribution that are greater than or equal to the statistic value
    calculated for `data`, and :math:`m` is the number of elements in the
    Monte Carlo null distribution (`n_mc_samples`). The addition of :math:`1`
    to the numerator and denominator can be thought of as including the
    value of the statistic corresponding with `data` in the null distribution,
    but a more formal explanation is given in [5]_.

    *Limitations*

    The test can be very slow for some distribution families because unknown
    parameters of the distribution family must be fit to each of the Monte
    Carlo samples, and for most distributions in SciPy, distribution fitting
    performed via numerical optimization.

    *Anti-Pattern*

    For this reason, it may be tempting
    to treat parameters of the distribution pre-fit to `data` (by the user)
    as though they were `known_params`, as specification of all parameters of
    the distribution precludes the need to fit the distribution to each Monte
    Carlo sample. (This is essentially how the original Kilmogorov-Smirnov
    test is performed.) Although such a test can provide evidence against the
    null hypothesis, the test is conservative in the sense that small p-values
    will tend to (greatly) *overestimate* the probability of making a type I
    error (that is, rejecting the null hypothesis although it is true), and the
    power of the test is low (that is, it is less likely to reject the null
    hypothesis even when the null hypothesis is false).
    This is because the Monte Carlo samples are less likely to agree with the
    null-hypothesized distribution as well as `data`. This tends to increase
    the values of the statistic recorded in the null distribution, so that a
    larger number of them exceed the value of statistic for `data`, thereby
    inflating the p-value.

    References
    ----------
    .. [1] M. A. Stephens (1974). "EDF Statistics for Goodness of Fit and
           Some Comparisons." Journal of the American Statistical Association,
           Vol. 69, pp. 730-737.
    .. [2] W. Stute, W. G. Manteiga, and M. P. Quindimil (1993).
           "Bootstrap based goodness-of-fit-tests." Metrika 40.1: 243-256.
    .. [3] C. Genest, & B Rémillard. (2008). "Validity of the parametric
           bootstrap for goodness-of-fit testing in semiparametric models."
           Annales de l'IHP Probabilités et statistiques. Vol. 44. No. 6.
    .. [4] I. Kojadinovic and J. Yan (2012). "Goodness-of-fit testing based on
           a weighted bootstrap: A fast large-sample alternative to the
           parametric bootstrap." Canadian Journal of Statistics 40.3: 480-500.
    .. [5] B. Phipson and G. K. Smyth (2010). "Permutation P-values Should
           Never Be Zero: Calculating Exact P-values When Permutations Are
           Randomly Drawn." Statistical Applications in Genetics and Molecular
           Biology 9.1.
    .. [6] H. W. Lilliefors (1967). "On the Kolmogorov-Smirnov test for
           normality with mean and variance unknown." Journal of the American
           statistical Association 62.318: 399-402.
    .. [7] Filliben, James J. "The probability plot correlation coefficient
           test for normality." Technometrics 17.1 (1975): 111-117.

    Examples
    --------
    A well-known test of the null hypothesis that data were drawn from a
    given distribution is the Kolmogorov-Smirnov (KS) test, available in SciPy
    as `scipy.stats.ks_1samp`. Suppose we wish to test whether the following
    data:

    >>> import numpy as np
    >>> from scipy import stats
    >>> rng = np.random.default_rng()
    >>> x = stats.uniform.rvs(size=75, random_state=rng)

    were sampled from a normal distribution. To perform a KS test, the
    empirical distribution function of the observed data will be compared
    against the (theoretical) cumulative distribution function of a normal
    distribution. Of course, to do this, the normal distribution under the null
    hypothesis must be fully specified. This is commonly done by first fitting
    the ``loc`` and ``scale`` parameters of the distribution to the observed
    data, then performing the test.

    >>> loc, scale = np.mean(x), np.std(x, ddof=1)
    >>> cdf = stats.norm(loc, scale).cdf
    >>> stats.ks_1samp(x, cdf)
    KstestResult(statistic=0.1119257570456813,
                 pvalue=0.2827756409939257,
                 statistic_location=0.7751845155861765,
                 statistic_sign=-1)

    An advantage of the KS-test is that the p-value - the probability of
    obtaining a value of the test statistic under the null hypothesis as
    extreme as the value obtained from the observed data - can be calculated
    exactly and efficiently. `goodness_of_fit` can only approximate these
    results.

    >>> known_params = {'loc': loc, 'scale': scale}
    >>> res = stats.goodness_of_fit(stats.norm, x, known_params=known_params,
    ...                             statistic='ks', rng=rng)
    >>> res.statistic, res.pvalue
    (0.1119257570456813, 0.2788)

    The statistic matches exactly, but the p-value is estimated by forming
    a "Monte Carlo null distribution", that is, by explicitly drawing random
    samples from `scipy.stats.norm` with the provided parameters and
    calculating the stastic for each. The fraction of these statistic values
    at least as extreme as ``res.statistic`` approximates the exact p-value
    calculated by `scipy.stats.ks_1samp`.

    However, in many cases, we would prefer to test only that the data were
    sampled from one of *any* member of the normal distribution family, not
    specifically from the normal distribution with the location and scale
    fitted to the observed sample. In this case, Lilliefors [6]_ argued that
    the KS test is far too conservative (that is, the p-value overstates
    the actual probability of rejecting a true null hypothesis) and thus lacks
    power - the ability to reject the null hypothesis when the null hypothesis
    is actually false.
    Indeed, our p-value above is approximately 0.28, which is far too large
    to reject the null hypothesis at any common significance level.

    Consider why this might be. Note that in the KS test above, the statistic
    always compares data against the CDF of a normal distribution fitted to the
    *observed data*. This tends to reduce the value of the statistic for the
    observed data, but it is "unfair" when computing the statistic for other
    samples, such as those we randomly draw to form the Monte Carlo null
    distribution. It is easy to correct for this: whenever we compute the KS
    statistic of a sample, we use the CDF of a normal distribution fitted
    to *that sample*. The null distribution in this case has not been
    calculated exactly and is tyically approximated using Monte Carlo methods
    as described above. This is where `goodness_of_fit` excels.

    >>> res = stats.goodness_of_fit(stats.norm, x, statistic='ks',
    ...                             rng=rng)
    >>> res.statistic, res.pvalue
    (0.1119257570456813, 0.0196)

    Indeed, this p-value is much smaller, and small enough to (correctly)
    reject the null hypothesis at common significance levels, including 5% and
    2.5%.

    However, the KS statistic is not very sensitive to all deviations from
    normality. The original advantage of the KS statistic was the ability
    to compute the null distribution theoretically, but a more sensitive
    statistic - resulting in a higher test power - can be used now that we can
    approximate the null distribution
    computationally. The Anderson-Darling statistic [1]_ tends to be more
    sensitive, and critical values of the this statistic have been tabulated
    for various significance levels and sample sizes using Monte Carlo methods.

    >>> res = stats.anderson(x, 'norm')
    >>> print(res.statistic)
    1.2139573337497467
    >>> print(res.critical_values)
    [0.549 0.625 0.75  0.875 1.041]
    >>> print(res.significance_level)
    [15.  10.   5.   2.5  1. ]

    Here, the observed value of the statistic exceeds the critical value
    corresponding with a 1% significance level. This tells us that the p-value
    of the observed data is less than 1%, but what is it? We could interpolate
    from these (already-interpolated) values, but `goodness_of_fit` can
    estimate it directly.

    >>> res = stats.goodness_of_fit(stats.norm, x, statistic='ad',
    ...                             rng=rng)
    >>> res.statistic, res.pvalue
    (1.2139573337497467, 0.0034)

    A further advantage is that use of `goodness_of_fit` is not limited to
    a particular set of distributions or conditions on which parameters
    are known versus which must be estimated from data. Instead,
    `goodness_of_fit` can estimate p-values relatively quickly for any
    distribution with a sufficiently fast and reliable ``fit`` method. For
    instance, here we perform a goodness of fit test using the Cramer-von Mises
    statistic against the Rayleigh distribution with known location and unknown
    scale.

    >>> rng = np.random.default_rng()
    >>> x = stats.chi(df=2.2, loc=0, scale=2).rvs(size=1000, random_state=rng)
    >>> res = stats.goodness_of_fit(stats.rayleigh, x, statistic='cvm',
    ...                             known_params={'loc': 0}, rng=rng)

    This executes fairly quickly, but to check the reliability of the ``fit``
    method, we should inspect the fit result.

    >>> res.fit_result  # location is as specified, and scale is reasonable
      params: FitParams(loc=0.0, scale=2.1026719844231243)
     success: True
     message: 'The fit was performed successfully.'
    >>> import matplotlib.pyplot as plt  # matplotlib must be installed to plot
    >>> res.fit_result.plot()
    >>> plt.show()

    If the distribution is not fit to the observed data as well as possible,
    the test may not control the type I error rate, that is, the chance of
    rejecting the null hypothesis even when it is true.

    We should also look for extreme outliers in the null distribution that
    may be caused by unreliable fitting. These do not necessarily invalidate
    the result, but they tend to reduce the test's power.

    >>> _, ax = plt.subplots()
    >>> ax.hist(np.log10(res.null_distribution))
    >>> ax.set_xlabel("log10 of CVM statistic under the null hypothesis")
    >>> ax.set_ylabel("Frequency")
    >>> ax.set_title("Histogram of the Monte Carlo null distribution")
    >>> plt.show()

    This plot seems reassuring.

    If ``fit`` method is working reliably, and if the distribution of the test
    statistic is not particularly sensitive to the values of the fitted
    parameters, then the p-value provided by `goodness_of_fit` is expected to
    be a good approximation.

    >>> res.statistic, res.pvalue
    (0.2231991510248692, 0.0525)

    """
    args = _gof_iv(dist, data, known_params, fit_params, guessed_params,
                   statistic, n_mc_samples, rng)
    (dist, data, fixed_nhd_params, fixed_rfd_params, guessed_nhd_params,
     guessed_rfd_params, statistic, n_mc_samples_int, rng) = args

    # Fit null hypothesis distribution to data
    nhd_fit_fun = _get_fit_fun(dist, data, guessed_nhd_params,
                               fixed_nhd_params)
    nhd_vals = nhd_fit_fun(data)
    nhd_dist = dist(*nhd_vals)

    def rvs(size):
        return nhd_dist.rvs(size=size, random_state=rng)

    # Define statistic
    fit_fun = _get_fit_fun(dist, data, guessed_rfd_params, fixed_rfd_params)
    if callable(statistic):
        compare_fun = statistic
    else:
        compare_fun = _compare_dict[statistic]
    alternative = getattr(compare_fun, 'alternative', 'greater')

    def statistic_fun(data, axis):
        # Make things simple by always working along the last axis.
        data = np.moveaxis(data, axis, -1)
        rfd_vals = fit_fun(data)
        rfd_dist = dist(*rfd_vals)
        return compare_fun(rfd_dist, data, axis=-1)

    res = stats.monte_carlo_test(data, rvs, statistic_fun, vectorized=True,
                                 n_resamples=n_mc_samples, axis=-1,
                                 alternative=alternative)
    opt_res = optimize.OptimizeResult()
    opt_res.success = True
    opt_res.message = "The fit was performed successfully."
    opt_res.x = nhd_vals
    # Only continuous distributions for now, hence discrete=False
    # There's no fundamental limitation; it's just that we're not using
    # stats.fit, discrete distributions don't have `fit` method, and
    # we haven't written any vectorized fit functions for a discrete
    # distribution yet.
    return GoodnessOfFitResult(FitResult(dist, data, False, opt_res),
                               res.statistic, res.pvalue,
                               res.null_distribution)


def _get_fit_fun(dist, data, guessed_params, fixed_params):

    shape_names = [] if dist.shapes is None else dist.shapes.split(", ")
    param_names = shape_names + ['loc', 'scale']
    fparam_names = ['f'+name for name in param_names]
    all_fixed = not set(fparam_names).difference(fixed_params)
    guessed_shapes = [guessed_params.pop(x, None)
                      for x in shape_names if x in guessed_params]

    if all_fixed:
        def fit_fun(data):
            return [fixed_params[name] for name in fparam_names]
    # Define statistic, including fitting distribution to data
    elif dist in _fit_funs:
        def fit_fun(data):
            params = _fit_funs[dist](data, **fixed_params)
            params = np.asarray(np.broadcast_arrays(*params))
            if params.ndim > 1:
                params = params[..., np.newaxis]
            return params
    else:
        def fit_fun_1d(data):
            return dist.fit(data, *guessed_shapes, **guessed_params,
                            **fixed_params)

        def fit_fun(data):
            params = np.apply_along_axis(fit_fun_1d, axis=-1, arr=data)
            if params.ndim > 1:
                params = params.T[..., np.newaxis]
            return params

    return fit_fun


# Vectorized fitting functions. These are to accept ND `data` in which each
# row (slice along last axis) is a sample to fit and scalar fixed parameters.
# They return a tuple of shape parameter arrays, each of shape data.shape[:-1].
def _fit_norm(data, floc=None, fscale=None):
    loc = floc
    scale = fscale
    if loc is None and scale is None:
        loc = np.mean(data, axis=-1)
        scale = np.std(data, ddof=1, axis=-1)
    elif loc is None:
        loc = np.mean(data, axis=-1)
    elif scale is None:
        scale = np.sqrt(((data - loc)**2).mean(axis=-1))
    return loc, scale


_fit_funs = {stats.norm: _fit_norm}  # type: ignore[attr-defined]


# Vectorized goodness of fit statistic functions. These accept a frozen
# distribution object and `data` in which each row (slice along last axis) is
# a sample.


def _anderson_darling(dist, data, axis):
    x = np.sort(data, axis=-1)
    n = data.shape[-1]
    i = np.arange(1, n+1)
    Si = (2*i - 1)/n * (dist.logcdf(x) + dist.logsf(x[..., ::-1]))
    S = np.sum(Si, axis=-1)
    return -n - S


def _compute_dplus(cdfvals):  # adapted from _stats_py before gh-17062
    n = cdfvals.shape[-1]
    return (np.arange(1.0, n + 1) / n - cdfvals).max(axis=-1)


def _compute_dminus(cdfvals):
    n = cdfvals.shape[-1]
    return (cdfvals - np.arange(0.0, n)/n).max(axis=-1)


def _kolmogorov_smirnov(dist, data, axis=-1):
    x = np.sort(data, axis=axis)
    cdfvals = dist.cdf(x)
    cdfvals = np.moveaxis(cdfvals, axis, -1)
    Dplus = _compute_dplus(cdfvals)  # always works along last axis
    Dminus = _compute_dminus(cdfvals)
    return np.maximum(Dplus, Dminus)


def _corr(X, M):
    # Correlation coefficient r, simplified and vectorized as we need it.
    # See [7] Equation (2). Lemma 1/2 are only for distributions symmetric
    # about 0.
    Xm = X.mean(axis=-1, keepdims=True)
    Mm = M.mean(axis=-1, keepdims=True)
    num = np.sum((X - Xm) * (M - Mm), axis=-1)
    den = np.sqrt(np.sum((X - Xm)**2, axis=-1) * np.sum((M - Mm)**2, axis=-1))
    return num/den


def _filliben(dist, data, axis):
    # [7] Section 8 # 1
    X = np.sort(data, axis=-1)

    # [7] Section 8 # 2
    n = data.shape[-1]
    k = np.arange(1, n+1)
    # Filliben used an approximation for the uniform distribution order
    # statistic medians.
    # m = (k - .3175)/(n + 0.365)
    # m[-1] = 0.5**(1/n)
    # m[0] = 1 - m[-1]
    # We can just as easily use the (theoretically) exact values. See e.g.
    # https://en.wikipedia.org/wiki/Order_statistic
    # "Order statistics sampled from a uniform distribution"
    m = stats.beta(k, n + 1 - k).median()

    # [7] Section 8 # 3
    M = dist.ppf(m)

    # [7] Section 8 # 4
    return _corr(X, M)
_filliben.alternative = 'less'  # type: ignore[attr-defined]


def _cramer_von_mises(dist, data, axis):
    x = np.sort(data, axis=-1)
    n = data.shape[-1]
    cdfvals = dist.cdf(x)
    u = (2*np.arange(1, n+1) - 1)/(2*n)
    w = 1 / (12*n) + np.sum((u - cdfvals)**2, axis=-1)
    return w


_compare_dict = {"ad": _anderson_darling, "ks": _kolmogorov_smirnov,
                 "cvm": _cramer_von_mises, "filliben": _filliben}


def _gof_iv(dist, data, known_params, fit_params, guessed_params, statistic,
            n_mc_samples, rng):

    if not isinstance(dist, stats.rv_continuous):
        message = ("`dist` must be a (non-frozen) instance of "
                   "`stats.rv_continuous`.")
        raise TypeError(message)

    data = np.asarray(data, dtype=float)
    if not data.ndim == 1:
        message = "`data` must be a one-dimensional array of numbers."
        raise ValueError(message)

    # Leave validation of these key/value pairs to the `fit` method,
    # but collect these into dictionaries that will be used
    known_params = known_params or dict()
    fit_params = fit_params or dict()
    guessed_params = guessed_params or dict()

    known_params_f = {("f"+key): val for key, val in known_params.items()}
    fit_params_f = {("f"+key): val for key, val in fit_params.items()}

    # These are the values of parameters of the null distribution family
    # with which resamples are drawn
    fixed_nhd_params = known_params_f.copy()
    fixed_nhd_params.update(fit_params_f)

    # These are fixed when fitting the distribution family to resamples
    fixed_rfd_params = known_params_f.copy()

    # These are used as guesses when fitting the distribution family to
    # the original data
    guessed_nhd_params = guessed_params.copy()

    # These are used as guesses when fitting the distribution family to
    # resamples
    guessed_rfd_params = fit_params.copy()
    guessed_rfd_params.update(guessed_params)

    if not callable(statistic):
        statistic = statistic.lower()
        statistics = {'ad', 'ks', 'cvm', 'filliben'}
        if statistic not in statistics:
            message = f"`statistic` must be one of {statistics}."
            raise ValueError(message)

    n_mc_samples_int = int(n_mc_samples)
    if n_mc_samples_int != n_mc_samples:
        message = "`n_mc_samples` must be an integer."
        raise TypeError(message)

    rng = check_random_state(rng)

    return (dist, data, fixed_nhd_params, fixed_rfd_params, guessed_nhd_params,
            guessed_rfd_params, statistic, n_mc_samples_int, rng)
<|endoftext|>import sys

import numpy as np
from numpy import inf

from scipy import special
from scipy.stats._distribution_infrastructure import (
    ContinuousDistribution, _RealDomain, _RealParameter, _Parameterization,
    _combine_docs)

__all__ = ['Normal', 'Uniform']


class Normal(ContinuousDistribution):
    r"""Normal distribution with prescribed mean and standard deviation.

    The probability density function of the normal distribution is:

    .. math::

        f(x) = \frac{1}{\sigma \sqrt{2 \pi}} \exp {
            \left( -\frac{1}{2}\left( \frac{x - \mu}{\sigma} \right)^2 \right)}

    """
    # `ShiftedScaledDistribution` allows this to be generated automatically from
    # an instance of `StandardNormal`, but the normal distribution is so frequently
    # used that it's worth a bit of code duplication to get better performance.
    _mu_domain = _RealDomain(endpoints=(-inf, inf))
    _sigma_domain = _RealDomain(endpoints=(0, inf))
    _x_support = _RealDomain(endpoints=(-inf, inf))

    _mu_param = _RealParameter('mu',  symbol=r'\mu', domain=_mu_domain,
                               typical=(-1, 1))
    _sigma_param = _RealParameter('sigma', symbol=r'\sigma', domain=_sigma_domain,
                                  typical=(0.5, 1.5))
    _x_param = _RealParameter('x', domain=_x_support, typical=(-1, 1))

    _parameterizations = [_Parameterization(_mu_param, _sigma_param)]

    _variable = _x_param
    _normalization = 1/np.sqrt(2*np.pi)
    _log_normalization = np.log(2*np.pi)/2

    def __new__(cls, mu=None, sigma=None, **kwargs):
        if mu is None and sigma is None:
            return super().__new__(StandardNormal)
        return super().__new__(cls)

    def __init__(self, *, mu=0., sigma=1., **kwargs):
        super().__init__(mu=mu, sigma=sigma, **kwargs)

    def _logpdf_formula(self, x, *, mu, sigma, **kwargs):
        return StandardNormal._logpdf_formula(self, (x - mu)/sigma) - np.log(sigma)

    def _pdf_formula(self, x, *, mu, sigma, **kwargs):
        return StandardNormal._pdf_formula(self, (x - mu)/sigma) / sigma

    def _logcdf_formula(self, x, *, mu, sigma, **kwargs):
        return StandardNormal._logcdf_formula(self, (x - mu)/sigma)

    def _cdf_formula(self, x, *, mu, sigma, **kwargs):
        return StandardNormal._cdf_formula(self, (x - mu)/sigma)

    def _logccdf_formula(self, x, *, mu, sigma, **kwargs):
        return StandardNormal._logccdf_formula(self, (x - mu)/sigma)

    def _ccdf_formula(self, x, *, mu, sigma, **kwargs):
        return StandardNormal._ccdf_formula(self, (x - mu)/sigma)

    def _icdf_formula(self, x, *, mu, sigma, **kwargs):
        return StandardNormal._icdf_formula(self, x) * sigma + mu

    def _ilogcdf_formula(self, x, *, mu, sigma, **kwargs):
        return StandardNormal._ilogcdf_formula(self, x) * sigma + mu

    def _iccdf_formula(self, x, *, mu, sigma, **kwargs):
        return StandardNormal._iccdf_formula(self, x) * sigma + mu

    def _ilogccdf_formula(self, x, *, mu, sigma, **kwargs):
        return StandardNormal._ilogccdf_formula(self, x) * sigma + mu

    def _entropy_formula(self, *, mu, sigma, **kwargs):
        return StandardNormal._entropy_formula(self) + np.log(abs(sigma))

    def _logentropy_formula(self, *, mu, sigma, **kwargs):
        lH0 = StandardNormal._logentropy_formula(self)
        with np.errstate(divide='ignore'):
            # sigma = 1 -> log(sigma) = 0 -> log(log(sigma)) = -inf
            # Silence the unnecessary runtime warning
            lls = np.log(np.log(abs(sigma))+0j)
        return special.logsumexp(np.broadcast_arrays(lH0, lls), axis=0)

    def _median_formula(self, *, mu, sigma, **kwargs):
        return mu

    def _mode_formula(self, *, mu, sigma, **kwargs):
        return mu

    def _moment_raw_formula(self, order, *, mu, sigma, **kwargs):
        if order == 0:
            return np.ones_like(mu)
        elif order == 1:
            return mu
        else:
            return None
    _moment_raw_formula.orders = [0, 1]  # type: ignore[attr-defined]

    def _moment_central_formula(self, order, *, mu, sigma, **kwargs):
        if order == 0:
            return np.ones_like(mu)
        elif order % 2:
            return np.zeros_like(mu)
        else:
            # exact is faster (and obviously more accurate) for reasonable orders
            return sigma**order * special.factorial2(int(order) - 1, exact=True)

    def _sample_formula(self, sample_shape, full_shape, rng, *, mu, sigma, **kwargs):
        return rng.normal(loc=mu, scale=sigma, size=full_shape)[()]


def _log_diff(log_p, log_q):
    return special.logsumexp([log_p, log_q+np.pi*1j], axis=0)


class StandardNormal(Normal):
    r"""Standard normal distribution.

    The probability density function of the standard normal distribution is:

    .. math::

        f(x) = \frac{1}{\sqrt{2 \pi}} \exp \left( -\frac{1}{2} x^2 \right)

    """
    _x_support = _RealDomain(endpoints=(-inf, inf))
    _x_param = _RealParameter('x', domain=_x_support, typical=(-5, 5))
    _variable = _x_param
    _parameterizations = []
    _normalization = 1/np.sqrt(2*np.pi)
    _log_normalization = np.log(2*np.pi)/2
    mu = np.float64(0.)
    sigma = np.float64(1.)

    def __init__(self, **kwargs):
        ContinuousDistribution.__init__(self, **kwargs)

    def _logpdf_formula(self, x, **kwargs):
        return -(self._log_normalization + x**2/2)

    def _pdf_formula(self, x, **kwargs):
        return self._normalization * np.exp(-x**2/2)

    def _logcdf_formula(self, x, **kwargs):
        return special.log_ndtr(x)

    def _cdf_formula(self, x, **kwargs):
        return special.ndtr(x)

    def _logccdf_formula(self, x, **kwargs):
        return special.log_ndtr(-x)

    def _ccdf_formula(self, x, **kwargs):
        return special.ndtr(-x)

    def _icdf_formula(self, x, **kwargs):
        return special.ndtri(x)

    def _ilogcdf_formula(self, x, **kwargs):
        return special.ndtri_exp(x)

    def _iccdf_formula(self, x, **kwargs):
        return -special.ndtri(x)

    def _ilogccdf_formula(self, x, **kwargs):
        return -special.ndtri_exp(x)

    def _entropy_formula(self, **kwargs):
        return (1 + np.log(2*np.pi))/2

    def _logentropy_formula(self, **kwargs):
        return np.log1p(np.log(2*np.pi)) - np.log(2)

    def _median_formula(self, **kwargs):
        return 0

    def _mode_formula(self, **kwargs):
        return 0

    def _moment_raw_formula(self, order, **kwargs):
        raw_moments = {0: 1, 1: 0, 2: 1, 3: 0, 4: 3, 5: 0}
        return raw_moments.get(order, None)

    def _moment_central_formula(self, order, **kwargs):
        return self._moment_raw_formula(order, **kwargs)

    def _moment_standardized_formula(self, order, **kwargs):
        return self._moment_raw_formula(order, **kwargs)

    def _sample_formula(self, sample_shape, full_shape, rng, **kwargs):
        return rng.normal(size=full_shape)[()]


# currently for testing only
class _LogUniform(ContinuousDistribution):
    r"""Log-uniform distribution.

    The probability density function of the log-uniform distribution is:

    .. math::

        f(x; a, b) = \frac{1}
                          {x (\log(b) - \log(a))}

    If :math:`\log(X)` is a random variable that follows a uniform distribution
    between :math:`\log(a)` and :math:`\log(b)`, then :math:`X` is log-uniformly
    distributed with shape parameters :math:`a` and :math:`b`.

    """

    _a_domain = _RealDomain(endpoints=(0, inf))
    _b_domain = _RealDomain(endpoints=('a', inf))
    _log_a_domain = _RealDomain(endpoints=(-inf, inf))
    _log_b_domain = _RealDomain(endpoints=('log_a', inf))
    _x_support = _RealDomain(endpoints=('a', 'b'), inclusive=(True, True))

    _a_param = _RealParameter('a', domain=_a_domain, typical=(1e-3, 0.9))
    _b_param = _RealParameter('b', domain=_b_domain, typical=(1.1, 1e3))
    _log_a_param = _RealParameter('log_a', symbol=r'\log(a)',
                                  domain=_log_a_domain, typical=(-3, -0.1))
    _log_b_param = _RealParameter('log_b', symbol=r'\log(b)',
                                  domain=_log_b_domain, typical=(0.1, 3))
    _x_param = _RealParameter('x', domain=_x_support, typical=('a', 'b'))

    _b_domain.define_parameters(_a_param)
    _log_b_domain.define_parameters(_log_a_param)
    _x_support.define_parameters(_a_param, _b_param)

    _parameterizations = [_Parameterization(_log_a_param, _log_b_param),
                          _Parameterization(_a_param, _b_param)]
    _variable = _x_param

    def __init__(self, *, a=None, b=None, log_a=None, log_b=None, **kwargs):
        super().__init__(a=a, b=b, log_a=log_a, log_b=log_b, **kwargs)

    def _process_parameters(self, a=None, b=None, log_a=None, log_b=None, **kwargs):
        a = np.exp(log_a) if a is None else a
        b = np.exp(log_b) if b is None else b
        log_a = np.log(a) if log_a is None else log_a
        log_b = np.log(b) if log_b is None else log_b
        kwargs.update(dict(a=a, b=b, log_a=log_a, log_b=log_b))
        return kwargs

    # def _logpdf_formula(self, x, *, log_a, log_b, **kwargs):
    #     return -np.log(x) - np.log(log_b - log_a)

    def _pdf_formula(self, x, *, log_a, log_b, **kwargs):
        return ((log_b - log_a)*x)**-1

    # def _cdf_formula(self, x, *, log_a, log_b, **kwargs):
    #     return (np.log(x) - log_a)/(log_b - log_a)

    def _moment_raw_formula(self, order, log_a, log_b, **kwargs):
        if order == 0:
            return self._one
        t1 = self._one / (log_b - log_a) / order
        t2 = np.real(np.exp(_log_diff(order * log_b, order * log_a)))
        return t1 * t2


class Uniform(ContinuousDistribution):
    r"""Uniform distribution.

    The probability density function of the uniform distribution is:

    .. math::

        f(x; a, b) = \frac{1}
                          {b - a}

    """

    _a_domain = _RealDomain(endpoints=(-inf, inf))
    _b_domain = _RealDomain(endpoints=('a', inf))
    _x_support = _RealDomain(endpoints=('a', 'b'), inclusive=(True, True))

    _a_param = _RealParameter('a', domain=_a_domain, typical=(1e-3, 0.9))
    _b_param = _RealParameter('b', domain=_b_domain, typical=(1.1, 1e3))
    _x_param = _RealParameter('x', domain=_x_support, typical=('a', 'b'))

    _b_domain.define_parameters(_a_param)
    _x_support.define_parameters(_a_param, _b_param)

    _parameterizations = [_Parameterization(_a_param, _b_param)]
    _variable = _x_param

    def __init__(self, *, a=None, b=None, **kwargs):
        super().__init__(a=a, b=b, **kwargs)

    def _process_parameters(self, a=None, b=None, ab=None, **kwargs):
        ab = b - a
        kwargs.update(dict(a=a, b=b, ab=ab))
        return kwargs

    def _logpdf_formula(self, x, *, ab, **kwargs):
        return np.where(np.isnan(x), np.nan, -np.log(ab))

    def _pdf_formula(self, x, *, ab, **kwargs):
        return np.where(np.isnan(x), np.nan, 1/ab)

    def _logcdf_formula(self, x, *, a, ab, **kwargs):
        with np.errstate(divide='ignore'):
            return np.log(x - a) - np.log(ab)

    def _cdf_formula(self, x, *, a, ab, **kwargs):
        return (x - a) / ab

    def _logccdf_formula(self, x, *, b, ab, **kwargs):
        with np.errstate(divide='ignore'):
            return np.log(b - x) - np.log(ab)

    def _ccdf_formula(self, x, *, b, ab, **kwargs):
        return (b - x) / ab

    def _icdf_formula(self, p, *, a, ab, **kwargs):
        return a + ab*p

    def _iccdf_formula(self, p, *, b, ab, **kwargs):
        return b - ab*p

    def _entropy_formula(self, *, ab, **kwargs):
        return np.log(ab)

    def _mode_formula(self, *, a, b, ab, **kwargs):
        return a + 0.5*ab

    def _median_formula(self, *, a, b, ab, **kwargs):
        return a + 0.5*ab

    def _moment_raw_formula(self, order, a, b, ab, **kwargs):
        np1 = order + 1
        return (b**np1 - a**np1) / (np1 * ab)

    def _moment_central_formula(self, order, ab, **kwargs):
        return ab**2/12 if order == 2 else None

    _moment_central_formula.orders = [2]  # type: ignore[attr-defined]

    def _sample_formula(self, sample_shape, full_shape, rng, a, b, ab, **kwargs):
        try:
            return rng.uniform(a, b, size=full_shape)[()]
        except OverflowError:  # happens when there are NaNs
            return rng.uniform(0, 1, size=full_shape)*ab + a


class _Gamma(ContinuousDistribution):
    # Gamma distribution for testing only
    _a_domain = _RealDomain(endpoints=(0, inf))
    _x_support = _RealDomain(endpoints=(0, inf), inclusive=(False, False))

    _a_param = _RealParameter('a', domain=_a_domain, typical=(0.1, 10))
    _x_param = _RealParameter('x', domain=_x_support, typical=(0.1, 10))

    _parameterizations = [_Parameterization(_a_param)]
    _variable = _x_param

    def _pdf_formula(self, x, *, a, **kwargs):
        return x ** (a - 1) * np.exp(-x) / special.gamma(a)


# Distribution classes need only define the summary and beginning of the extended
# summary portion of the class documentation. All other documentation, including
# examples, is generated automatically.
_module = sys.modules[__name__].__dict__
for dist_name in __all__:
    _module[dist_name].__doc__ = _combine_docs(_module[dist_name])
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.stats` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = ["gaussian_kde"]  # noqa: F822


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="stats", module="kde",
                                   private_modules=["_kde"], all=__all__,
                                   attribute=name)
<|endoftext|>"""
An extension of scipy.stats._stats_py to support masked arrays

"""
# Original author (2007): Pierre GF Gerard-Marchant


__all__ = ['argstoarray',
           'count_tied_groups',
           'describe',
           'f_oneway', 'find_repeats','friedmanchisquare',
           'kendalltau','kendalltau_seasonal','kruskal','kruskalwallis',
           'ks_twosamp', 'ks_2samp', 'kurtosis', 'kurtosistest',
           'ks_1samp', 'kstest',
           'linregress',
           'mannwhitneyu', 'meppf','mode','moment','mquantiles','msign',
           'normaltest',
           'obrientransform',
           'pearsonr','plotting_positions','pointbiserialr',
           'rankdata',
           'scoreatpercentile','sem',
           'sen_seasonal_slopes','skew','skewtest','spearmanr',
           'siegelslopes', 'theilslopes',
           'tmax','tmean','tmin','trim','trimboth',
           'trimtail','trima','trimr','trimmed_mean','trimmed_std',
           'trimmed_stde','trimmed_var','tsem','ttest_1samp','ttest_onesamp',
           'ttest_ind','ttest_rel','tvar',
           'variation',
           'winsorize',
           'brunnermunzel',
           ]

import numpy as np
from numpy import ndarray
import numpy.ma as ma
from numpy.ma import masked, nomask
import math

import itertools
import warnings
from collections import namedtuple

from . import distributions
from scipy._lib._util import _rename_parameter, _contains_nan
from scipy._lib._bunch import _make_tuple_bunch
import scipy.special as special
import scipy.stats._stats_py
import scipy.stats._stats_py as _stats_py

from ._stats_mstats_common import (
        _find_repeats,
        theilslopes as stats_theilslopes,
        siegelslopes as stats_siegelslopes
        )


def _chk_asarray(a, axis):
    # Always returns a masked array, raveled for axis=None
    a = ma.asanyarray(a)
    if axis is None:
        a = ma.ravel(a)
        outaxis = 0
    else:
        outaxis = axis
    return a, outaxis


def _chk2_asarray(a, b, axis):
    a = ma.asanyarray(a)
    b = ma.asanyarray(b)
    if axis is None:
        a = ma.ravel(a)
        b = ma.ravel(b)
        outaxis = 0
    else:
        outaxis = axis
    return a, b, outaxis


def _chk_size(a, b):
    a = ma.asanyarray(a)
    b = ma.asanyarray(b)
    (na, nb) = (a.size, b.size)
    if na != nb:
        raise ValueError("The size of the input array should match!"
                         f" ({na} <> {nb})")
    return (a, b, na)


def _ttest_finish(df, t, alternative):
    """Common code between all 3 t-test functions."""
    # We use ``stdtr`` directly here to preserve masked arrays

    if alternative == 'less':
        pval = special._ufuncs.stdtr(df, t)
    elif alternative == 'greater':
        pval = special._ufuncs.stdtr(df, -t)
    elif alternative == 'two-sided':
        pval = special._ufuncs.stdtr(df, -np.abs(t))*2
    else:
        raise ValueError("alternative must be "
                         "'less', 'greater' or 'two-sided'")

    if t.ndim == 0:
        t = t[()]
    if pval.ndim == 0:
        pval = pval[()]

    return t, pval


def argstoarray(*args):
    """
    Constructs a 2D array from a group of sequences.

    Sequences are filled with missing values to match the length of the longest
    sequence.

    Parameters
    ----------
    *args : sequences
        Group of sequences.

    Returns
    -------
    argstoarray : MaskedArray
        A ( `m` x `n` ) masked array, where `m` is the number of arguments and
        `n` the length of the longest argument.

    Notes
    -----
    `numpy.ma.vstack` has identical behavior, but is called with a sequence
    of sequences.

    Examples
    --------
    A 2D masked array constructed from a group of sequences is returned.

    >>> from scipy.stats.mstats import argstoarray
    >>> argstoarray([1, 2, 3], [4, 5, 6])
    masked_array(
     data=[[1.0, 2.0, 3.0],
           [4.0, 5.0, 6.0]],
     mask=[[False, False, False],
           [False, False, False]],
     fill_value=1e+20)

    The returned masked array filled with missing values when the lengths of
    sequences are different.

    >>> argstoarray([1, 3], [4, 5, 6])
    masked_array(
     data=[[1.0, 3.0, --],
           [4.0, 5.0, 6.0]],
     mask=[[False, False,  True],
           [False, False, False]],
     fill_value=1e+20)

    """
    if len(args) == 1 and not isinstance(args[0], ndarray):
        output = ma.asarray(args[0])
        if output.ndim != 2:
            raise ValueError("The input should be 2D")
    else:
        n = len(args)
        m = max([len(k) for k in args])
        output = ma.array(np.empty((n,m), dtype=float), mask=True)
        for (k,v) in enumerate(args):
            output[k,:len(v)] = v

    output[np.logical_not(np.isfinite(output._data))] = masked
    return output


def find_repeats(arr):
    """Find repeats in arr and return a tuple (repeats, repeat_count).

    The input is cast to float64. Masked values are discarded.

    Parameters
    ----------
    arr : sequence
        Input array. The array is flattened if it is not 1D.

    Returns
    -------
    repeats : ndarray
        Array of repeated values.
    counts : ndarray
        Array of counts.

    Examples
    --------
    >>> from scipy.stats import mstats
    >>> mstats.find_repeats([2, 1, 2, 3, 2, 2, 5])
    (array([2.]), array([4]))

    In the above example, 2 repeats 4 times.

    >>> mstats.find_repeats([[10, 20, 1, 2], [5, 5, 4, 4]])
    (array([4., 5.]), array([2, 2]))

    In the above example, both 4 and 5 repeat 2 times.

    """
    # Make sure we get a copy. ma.compressed promises a "new array", but can
    # actually return a reference.
    compr = np.asarray(ma.compressed(arr), dtype=np.float64)
    try:
        need_copy = np.may_share_memory(compr, arr)
    except AttributeError:
        # numpy < 1.8.2 bug: np.may_share_memory([], []) raises,
        # while in numpy 1.8.2 and above it just (correctly) returns False.
        need_copy = False
    if need_copy:
        compr = compr.copy()
    return _find_repeats(compr)


def count_tied_groups(x, use_missing=False):
    """
    Counts the number of tied values.

    Parameters
    ----------
    x : sequence
        Sequence of data on which to counts the ties
    use_missing : bool, optional
        Whether to consider missing values as tied.

    Returns
    -------
    count_tied_groups : dict
        Returns a dictionary (nb of ties: nb of groups).

    Examples
    --------
    >>> from scipy.stats import mstats
    >>> import numpy as np
    >>> z = [0, 0, 0, 2, 2, 2, 3, 3, 4, 5, 6]
    >>> mstats.count_tied_groups(z)
    {2: 1, 3: 2}

    In the above example, the ties were 0 (3x), 2 (3x) and 3 (2x).

    >>> z = np.ma.array([0, 0, 1, 2, 2, 2, 3, 3, 4, 5, 6])
    >>> mstats.count_tied_groups(z)
    {2: 2, 3: 1}
    >>> z[[1,-1]] = np.ma.masked
    >>> mstats.count_tied_groups(z, use_missing=True)
    {2: 2, 3: 1}

    """
    nmasked = ma.getmask(x).sum()
    # We need the copy as find_repeats will overwrite the initial data
    data = ma.compressed(x).copy()
    (ties, counts) = find_repeats(data)
    nties = {}
    if len(ties):
        nties = dict(zip(np.unique(counts), itertools.repeat(1)))
        nties.update(dict(zip(*find_repeats(counts))))

    if nmasked and use_missing:
        try:
            nties[nmasked] += 1
        except KeyError:
            nties[nmasked] = 1

    return nties


def rankdata(data, axis=None, use_missing=False):
    """Returns the rank (also known as order statistics) of each data point
    along the given axis.

    If some values are tied, their rank is averaged.
    If some values are masked, their rank is set to 0 if use_missing is False,
    or set to the average rank of the unmasked values if use_missing is True.

    Parameters
    ----------
    data : sequence
        Input data. The data is transformed to a masked array
    axis : {None,int}, optional
        Axis along which to perform the ranking.
        If None, the array is first flattened. An exception is raised if
        the axis is specified for arrays with a dimension larger than 2
    use_missing : bool, optional
        Whether the masked values have a rank of 0 (False) or equal to the
        average rank of the unmasked values (True).

    """
    def _rank1d(data, use_missing=False):
        n = data.count()
        rk = np.empty(data.size, dtype=float)
        idx = data.argsort()
        rk[idx[:n]] = np.arange(1,n+1)

        if use_missing:
            rk[idx[n:]] = (n+1)/2.
        else:
            rk[idx[n:]] = 0

        repeats = find_repeats(data.copy())
        for r in repeats[0]:
            condition = (data == r).filled(False)
            rk[condition] = rk[condition].mean()
        return rk

    data = ma.array(data, copy=False)
    if axis is None:
        if data.ndim > 1:
            return _rank1d(data.ravel(), use_missing).reshape(data.shape)
        else:
            return _rank1d(data, use_missing)
    else:
        return ma.apply_along_axis(_rank1d,axis,data,use_missing).view(ndarray)


ModeResult = namedtuple('ModeResult', ('mode', 'count'))


def mode(a, axis=0):
    """
    Returns an array of the modal (most common) value in the passed array.

    Parameters
    ----------
    a : array_like
        n-dimensional array of which to find mode(s).
    axis : int or None, optional
        Axis along which to operate. Default is 0. If None, compute over
        the whole array `a`.

    Returns
    -------
    mode : ndarray
        Array of modal values.
    count : ndarray
        Array of counts for each mode.

    Notes
    -----
    For more details, see `scipy.stats.mode`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> from scipy.stats import mstats
    >>> m_arr = np.ma.array([1, 1, 0, 0, 0, 0], mask=[0, 0, 1, 1, 1, 0])
    >>> mstats.mode(m_arr)  # note that most zeros are masked
    ModeResult(mode=array([1.]), count=array([2.]))

    """
    return _mode(a, axis=axis, keepdims=True)


def _mode(a, axis=0, keepdims=True):
    # Don't want to expose `keepdims` from the public `mstats.mode`
    a, axis = _chk_asarray(a, axis)

    def _mode1D(a):
        (rep,cnt) = find_repeats(a)
        if not cnt.ndim:
            return (0, 0)
        elif cnt.size:
            return (rep[cnt.argmax()], cnt.max())
        else:
            return (a.min(), 1)

    if axis is None:
        output = _mode1D(ma.ravel(a))
        output = (ma.array(output[0]), ma.array(output[1]))
    else:
        output = ma.apply_along_axis(_mode1D, axis, a)
        if keepdims is None or keepdims:
            newshape = list(a.shape)
            newshape[axis] = 1
            slices = [slice(None)] * output.ndim
            slices[axis] = 0
            modes = output[tuple(slices)].reshape(newshape)
            slices[axis] = 1
            counts = output[tuple(slices)].reshape(newshape)
            output = (modes, counts)
        else:
            output = np.moveaxis(output, axis, 0)

    return ModeResult(*output)


def _betai(a, b, x):
    x = np.asanyarray(x)
    x = ma.where(x < 1.0, x, 1.0)  # if x > 1 then return 1.0
    return special.betainc(a, b, x)


def msign(x):
    """Returns the sign of x, or 0 if x is masked."""
    return ma.filled(np.sign(x), 0)


def pearsonr(x, y):
    r"""
    Pearson correlation coefficient and p-value for testing non-correlation.

    The Pearson correlation coefficient [1]_ measures the linear relationship
    between two datasets.  The calculation of the p-value relies on the
    assumption that each dataset is normally distributed.  (See Kowalski [3]_
    for a discussion of the effects of non-normality of the input on the
    distribution of the correlation coefficient.)  Like other correlation
    coefficients, this one varies between -1 and +1 with 0 implying no
    correlation. Correlations of -1 or +1 imply an exact linear relationship.

    Parameters
    ----------
    x : (N,) array_like
        Input array.
    y : (N,) array_like
        Input array.

    Returns
    -------
    r : float
        Pearson's correlation coefficient.
    p-value : float
        Two-tailed p-value.

    Warns
    -----
    `~scipy.stats.ConstantInputWarning`
        Raised if an input is a constant array.  The correlation coefficient
        is not defined in this case, so ``np.nan`` is returned.

    `~scipy.stats.NearConstantInputWarning`
        Raised if an input is "nearly" constant.  The array ``x`` is considered
        nearly constant if ``norm(x - mean(x)) < 1e-13 * abs(mean(x))``.
        Numerical errors in the calculation ``x - mean(x)`` in this case might
        result in an inaccurate calculation of r.

    See Also
    --------
    spearmanr : Spearman rank-order correlation coefficient.
    kendalltau : Kendall's tau, a correlation measure for ordinal data.

    Notes
    -----
    The correlation coefficient is calculated as follows:

    .. math::

        r = \frac{\sum (x - m_x) (y - m_y)}
                 {\sqrt{\sum (x - m_x)^2 \sum (y - m_y)^2}}

    where :math:`m_x` is the mean of the vector x and :math:`m_y` is
    the mean of the vector y.

    Under the assumption that x and y are drawn from
    independent normal distributions (so the population correlation coefficient
    is 0), the probability density function of the sample correlation
    coefficient r is ([1]_, [2]_):

    .. math::

        f(r) = \frac{{(1-r^2)}^{n/2-2}}{\mathrm{B}(\frac{1}{2},\frac{n}{2}-1)}

    where n is the number of samples, and B is the beta function.  This
    is sometimes referred to as the exact distribution of r.  This is
    the distribution that is used in `pearsonr` to compute the p-value.
    The distribution is a beta distribution on the interval [-1, 1],
    with equal shape parameters a = b = n/2 - 1.  In terms of SciPy's
    implementation of the beta distribution, the distribution of r is::

        dist = scipy.stats.beta(n/2 - 1, n/2 - 1, loc=-1, scale=2)

    The p-value returned by `pearsonr` is a two-sided p-value. The p-value
    roughly indicates the probability of an uncorrelated system
    producing datasets that have a Pearson correlation at least as extreme
    as the one computed from these datasets. More precisely, for a
    given sample with correlation coefficient r, the p-value is
    the probability that abs(r') of a random sample x' and y' drawn from
    the population with zero correlation would be greater than or equal
    to abs(r). In terms of the object ``dist`` shown above, the p-value
    for a given r and length n can be computed as::

        p = 2*dist.cdf(-abs(r))

    When n is 2, the above continuous distribution is not well-defined.
    One can interpret the limit of the beta distribution as the shape
    parameters a and b approach a = b = 0 as a discrete distribution with
    equal probability masses at r = 1 and r = -1.  More directly, one
    can observe that, given the data x = [x1, x2] and y = [y1, y2], and
    assuming x1 != x2 and y1 != y2, the only possible values for r are 1
    and -1.  Because abs(r') for any sample x' and y' with length 2 will
    be 1, the two-sided p-value for a sample of length 2 is always 1.

    References
    ----------
    .. [1] "Pearson correlation coefficient", Wikipedia,
           https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    .. [2] Student, "Probable error of a correlation coefficient",
           Biometrika, Volume 6, Issue 2-3, 1 September 1908, pp. 302-310.
    .. [3] C. J. Kowalski, "On the Effects of Non-Normality on the Distribution
           of the Sample Product-Moment Correlation Coefficient"
           Journal of the Royal Statistical Society. Series C (Applied
           Statistics), Vol. 21, No. 1 (1972), pp. 1-12.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> from scipy.stats import mstats
    >>> mstats.pearsonr([1, 2, 3, 4, 5], [10, 9, 2.5, 6, 4])
    (-0.7426106572325057, 0.1505558088534455)

    There is a linear dependence between x and y if y = a + b*x + e, where
    a,b are constants and e is a random error term, assumed to be independent
    of x. For simplicity, assume that x is standard normal, a=0, b=1 and let
    e follow a normal distribution with mean zero and standard deviation s>0.

    >>> s = 0.5
    >>> x = stats.norm.rvs(size=500)
    >>> e = stats.norm.rvs(scale=s, size=500)
    >>> y = x + e
    >>> mstats.pearsonr(x, y)
    (0.9029601878969703, 8.428978827629898e-185) # may vary

    This should be close to the exact value given by

    >>> 1/np.sqrt(1 + s**2)
    0.8944271909999159

    For s=0.5, we observe a high level of correlation. In general, a large
    variance of the noise reduces the correlation, while the correlation
    approaches one as the variance of the error goes to zero.

    It is important to keep in mind that no correlation does not imply
    independence unless (x, y) is jointly normal. Correlation can even be zero
    when there is a very simple dependence structure: if X follows a
    standard normal distribution, let y = abs(x). Note that the correlation
    between x and y is zero. Indeed, since the expectation of x is zero,
    cov(x, y) = E[x*y]. By definition, this equals E[x*abs(x)] which is zero
    by symmetry. The following lines of code illustrate this observation:

    >>> y = np.abs(x)
    >>> mstats.pearsonr(x, y)
    (-0.016172891856853524, 0.7182823678751942) # may vary

    A non-zero correlation coefficient can be misleading. For example, if X has
    a standard normal distribution, define y = x if x < 0 and y = 0 otherwise.
    A simple calculation shows that corr(x, y) = sqrt(2/Pi) = 0.797...,
    implying a high level of correlation:

    >>> y = np.where(x < 0, x, 0)
    >>> mstats.pearsonr(x, y)
    (0.8537091583771509, 3.183461621422181e-143) # may vary

    This is unintuitive since there is no dependence of x and y if x is larger
    than zero which happens in about half of the cases if we sample x and y.
    """
    (x, y, n) = _chk_size(x, y)
    (x, y) = (x.ravel(), y.ravel())
    # Get the common mask and the total nb of unmasked elements
    m = ma.mask_or(ma.getmask(x), ma.getmask(y))
    n -= m.sum()
    df = n-2
    if df < 0:
        return (masked, masked)

    return scipy.stats._stats_py.pearsonr(
                ma.masked_array(x, mask=m).compressed(),
                ma.masked_array(y, mask=m).compressed())


def spearmanr(x, y=None, use_ties=True, axis=None, nan_policy='propagate',
              alternative='two-sided'):
    """
    Calculates a Spearman rank-order correlation coefficient and the p-value
    to test for non-correlation.

    The Spearman correlation is a nonparametric measure of the linear
    relationship between two datasets. Unlike the Pearson correlation, the
    Spearman correlation does not assume that both datasets are normally
    distributed. Like other correlation coefficients, this one varies
    between -1 and +1 with 0 implying no correlation. Correlations of -1 or
    +1 imply a monotonic relationship. Positive correlations imply that
    as `x` increases, so does `y`. Negative correlations imply that as `x`
    increases, `y` decreases.

    Missing values are discarded pair-wise: if a value is missing in `x`, the
    corresponding value in `y` is masked.

    The p-value roughly indicates the probability of an uncorrelated system
    producing datasets that have a Spearman correlation at least as extreme
    as the one computed from these datasets. The p-values are not entirely
    reliable but are probably reasonable for datasets larger than 500 or so.

    Parameters
    ----------
    x, y : 1D or 2D array_like, y is optional
        One or two 1-D or 2-D arrays containing multiple variables and
        observations. When these are 1-D, each represents a vector of
        observations of a single variable. For the behavior in the 2-D case,
        see under ``axis``, below.
    use_ties : bool, optional
        DO NOT USE.  Does not do anything, keyword is only left in place for
        backwards compatibility reasons.
    axis : int or None, optional
        If axis=0 (default), then each column represents a variable, with
        observations in the rows. If axis=1, the relationship is transposed:
        each row represents a variable, while the columns contain observations.
        If axis=None, then both arrays will be raveled.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan. 'propagate' returns nan,
        'raise' throws an error, 'omit' performs the calculations ignoring nan
        values. Default is 'propagate'.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        The following options are available:

        * 'two-sided': the correlation is nonzero
        * 'less': the correlation is negative (less than zero)
        * 'greater':  the correlation is positive (greater than zero)

        .. versionadded:: 1.7.0

    Returns
    -------
    res : SignificanceResult
        An object containing attributes:

        statistic : float or ndarray (2-D square)
            Spearman correlation matrix or correlation coefficient (if only 2
            variables are given as parameters). Correlation matrix is square
            with length equal to total number of variables (columns or rows) in
            ``a`` and ``b`` combined.
        pvalue : float
            The p-value for a hypothesis test whose null hypothesis
            is that two sets of data are linearly uncorrelated. See
            `alternative` above for alternative hypotheses. `pvalue` has the
            same shape as `statistic`.

    References
    ----------
    [CRCProbStat2000] section 14.7

    """
    if not use_ties:
        raise ValueError("`use_ties=False` is not supported in SciPy >= 1.2.0")

    # Always returns a masked array, raveled if axis=None
    x, axisout = _chk_asarray(x, axis)
    if y is not None:
        # Deal only with 2-D `x` case.
        y, _ = _chk_asarray(y, axis)
        if axisout == 0:
            x = ma.column_stack((x, y))
        else:
            x = ma.vstack((x, y))

    if axisout == 1:
        # To simplify the code that follow (always use `n_obs, n_vars` shape)
        x = x.T

    if nan_policy == 'omit':
        x = ma.masked_invalid(x)

    def _spearmanr_2cols(x):
        # Mask the same observations for all variables, and then drop those
        # observations (can't leave them masked, rankdata is weird).
        x = ma.mask_rowcols(x, axis=0)
        x = x[~x.mask.any(axis=1), :]

        # If either column is entirely NaN or Inf
        if not np.any(x.data):
            res = scipy.stats._stats_py.SignificanceResult(np.nan, np.nan)
            res.correlation = np.nan
            return res

        m = ma.getmask(x)
        n_obs = x.shape[0]
        dof = n_obs - 2 - int(m.sum(axis=0)[0])
        if dof < 0:
            raise ValueError("The input must have at least 3 entries!")

        # Gets the ranks and rank differences
        x_ranked = rankdata(x, axis=0)
        rs = ma.corrcoef(x_ranked, rowvar=False).data

        # rs can have elements equal to 1, so avoid zero division warnings
        with np.errstate(divide='ignore'):
            # clip the small negative values possibly caused by rounding
            # errors before taking the square root
            t = rs * np.sqrt((dof / ((rs+1.0) * (1.0-rs))).clip(0))

        t, prob = _ttest_finish(dof, t, alternative)

        # For backwards compatibility, return scalars when comparing 2 columns
        if rs.shape == (2, 2):
            res = scipy.stats._stats_py.SignificanceResult(rs[1, 0],
                                                           prob[1, 0])
            res.correlation = rs[1, 0]
            return res
        else:
            res = scipy.stats._stats_py.SignificanceResult(rs, prob)
            res.correlation = rs
            return res

    # Need to do this per pair of variables, otherwise the dropped observations
    # in a third column mess up the result for a pair.
    n_vars = x.shape[1]
    if n_vars == 2:
        return _spearmanr_2cols(x)
    else:
        rs = np.ones((n_vars, n_vars), dtype=float)
        prob = np.zeros((n_vars, n_vars), dtype=float)
        for var1 in range(n_vars - 1):
            for var2 in range(var1+1, n_vars):
                result = _spearmanr_2cols(x[:, [var1, var2]])
                rs[var1, var2] = result.correlation
                rs[var2, var1] = result.correlation
                prob[var1, var2] = result.pvalue
                prob[var2, var1] = result.pvalue

        res = scipy.stats._stats_py.SignificanceResult(rs, prob)
        res.correlation = rs
        return res


def _kendall_p_exact(n, c, alternative='two-sided'):

    # Use the fact that distribution is symmetric: always calculate a CDF in
    # the left tail.
    # This will be the one-sided p-value if `c` is on the side of
    # the null distribution predicted by the alternative hypothesis.
    # The two-sided p-value will be twice this value.
    # If `c` is on the other side of the null distribution, we'll need to
    # take the complement and add back the probability mass at `c`.
    in_right_tail = (c >= (n*(n-1))//2 - c)
    alternative_greater = (alternative == 'greater')
    c = int(min(c, (n*(n-1))//2 - c))

    # Exact p-value, see Maurice G. Kendall, "Rank Correlation Methods"
    # (4th Edition), Charles Griffin & Co., 1970.
    if n <= 0:
        raise ValueError(f'n ({n}) must be positive')
    elif c < 0 or 4*c > n*(n-1):
        raise ValueError(f'c ({c}) must satisfy 0 <= 4c <= n(n-1) = {n*(n-1)}.')
    elif n == 1:
        prob = 1.0
        p_mass_at_c = 1
    elif n == 2:
        prob = 1.0
        p_mass_at_c = 0.5
    elif c == 0:
        prob = 2.0/math.factorial(n) if n < 171 else 0.0
        p_mass_at_c = prob/2
    elif c == 1:
        prob = 2.0/math.factorial(n-1) if n < 172 else 0.0
        p_mass_at_c = (n-1)/math.factorial(n)
    elif 4*c == n*(n-1) and alternative == 'two-sided':
        # I'm sure there's a simple formula for p_mass_at_c in this
        # case, but I don't know it. Use generic formula for one-sided p-value.
        prob = 1.0
    elif n < 171:
        new = np.zeros(c+1)
        new[0:2] = 1.0
        for j in range(3,n+1):
            new = np.cumsum(new)
            if j <= c:
                new[j:] -= new[:c+1-j]
        prob = 2.0*np.sum(new)/math.factorial(n)
        p_mass_at_c = new[-1]/math.factorial(n)
    else:
        new = np.zeros(c+1)
        new[0:2] = 1.0
        for j in range(3, n+1):
            new = np.cumsum(new)/j
            if j <= c:
                new[j:] -= new[:c+1-j]
        prob = np.sum(new)
        p_mass_at_c = new[-1]/2

    if alternative != 'two-sided':
        # if the alternative hypothesis and alternative agree,
        # one-sided p-value is half the two-sided p-value
        if in_right_tail == alternative_greater:
            prob /= 2
        else:
            prob = 1 - prob/2 + p_mass_at_c

    prob = np.clip(prob, 0, 1)

    return prob


def kendalltau(x, y, use_ties=True, use_missing=False, method='auto',
               alternative='two-sided'):
    """
    Computes Kendall's rank correlation tau on two variables *x* and *y*.

    Parameters
    ----------
    x : sequence
        First data list (for example, time).
    y : sequence
        Second data list.
    use_ties : {True, False}, optional
        Whether ties correction should be performed.
    use_missing : {False, True}, optional
        Whether missing data should be allocated a rank of 0 (False) or the
        average rank (True)
    method : {'auto', 'asymptotic', 'exact'}, optional
        Defines which method is used to calculate the p-value [1]_.
        'asymptotic' uses a normal approximation valid for large samples.
        'exact' computes the exact p-value, but can only be used if no ties
        are present. As the sample size increases, the 'exact' computation
        time may grow and the result may lose some precision.
        'auto' is the default and selects the appropriate
        method based on a trade-off between speed and accuracy.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        The following options are available:

        * 'two-sided': the rank correlation is nonzero
        * 'less': the rank correlation is negative (less than zero)
        * 'greater':  the rank correlation is positive (greater than zero)

    Returns
    -------
    res : SignificanceResult
        An object containing attributes:

        statistic : float
           The tau statistic.
        pvalue : float
           The p-value for a hypothesis test whose null hypothesis is
           an absence of association, tau = 0.

    References
    ----------
    .. [1] Maurice G. Kendall, "Rank Correlation Methods" (4th Edition),
           Charles Griffin & Co., 1970.

    """
    (x, y, n) = _chk_size(x, y)
    (x, y) = (x.flatten(), y.flatten())
    m = ma.mask_or(ma.getmask(x), ma.getmask(y))
    if m is not nomask:
        x = ma.array(x, mask=m, copy=True)
        y = ma.array(y, mask=m, copy=True)
        # need int() here, otherwise numpy defaults to 32 bit
        # integer on all Windows architectures, causing overflow.
        # int() will keep it infinite precision.
        n -= int(m.sum())

    if n < 2:
        res = scipy.stats._stats_py.SignificanceResult(np.nan, np.nan)
        res.correlation = np.nan
        return res

    rx = ma.masked_equal(rankdata(x, use_missing=use_missing), 0)
    ry = ma.masked_equal(rankdata(y, use_missing=use_missing), 0)
    idx = rx.argsort()
    (rx, ry) = (rx[idx], ry[idx])
    C = np.sum([((ry[i+1:] > ry[i]) * (rx[i+1:] > rx[i])).filled(0).sum()
                for i in range(len(ry)-1)], dtype=float)
    D = np.sum([((ry[i+1:] < ry[i])*(rx[i+1:] > rx[i])).filled(0).sum()
                for i in range(len(ry)-1)], dtype=float)
    xties = count_tied_groups(x)
    yties = count_tied_groups(y)
    if use_ties:
        corr_x = np.sum([v*k*(k-1) for (k,v) in xties.items()], dtype=float)
        corr_y = np.sum([v*k*(k-1) for (k,v) in yties.items()], dtype=float)
        denom = ma.sqrt((n*(n-1)-corr_x)/2. * (n*(n-1)-corr_y)/2.)
    else:
        denom = n*(n-1)/2.
    tau = (C-D) / denom

    if method == 'exact' and (xties or yties):
        raise ValueError("Ties found, exact method cannot be used.")

    if method == 'auto':
        if (not xties and not yties) and (n <= 33 or min(C, n*(n-1)/2.0-C) <= 1):
            method = 'exact'
        else:
            method = 'asymptotic'

    if not xties and not yties and method == 'exact':
        prob = _kendall_p_exact(n, C, alternative)

    elif method == 'asymptotic':
        var_s = n*(n-1)*(2*n+5)
        if use_ties:
            var_s -= np.sum([v*k*(k-1)*(2*k+5)*1. for (k,v) in xties.items()])
            var_s -= np.sum([v*k*(k-1)*(2*k+5)*1. for (k,v) in yties.items()])
            v1 = (np.sum([v*k*(k-1) for (k, v) in xties.items()], dtype=float) *
                  np.sum([v*k*(k-1) for (k, v) in yties.items()], dtype=float))
            v1 /= 2.*n*(n-1)
            if n > 2:
                v2 = np.sum([v*k*(k-1)*(k-2) for (k,v) in xties.items()],
                            dtype=float) * \
                     np.sum([v*k*(k-1)*(k-2) for (k,v) in yties.items()],
                            dtype=float)
                v2 /= 9.*n*(n-1)*(n-2)
            else:
                v2 = 0
        else:
            v1 = v2 = 0

        var_s /= 18.
        var_s += (v1 + v2)
        z = (C-D)/np.sqrt(var_s)
        prob = scipy.stats._stats_py._get_pvalue(z, distributions.norm, alternative)
    else:
        raise ValueError("Unknown method "+str(method)+" specified, please "
                         "use auto, exact or asymptotic.")

    res = scipy.stats._stats_py.SignificanceResult(tau[()], prob[()])
    res.correlation = tau
    return res


def kendalltau_seasonal(x):
    """
    Computes a multivariate Kendall's rank correlation tau, for seasonal data.

    Parameters
    ----------
    x : 2-D ndarray
        Array of seasonal data, with seasons in columns.

    """
    x = ma.array(x, subok=True, copy=False, ndmin=2)
    (n,m) = x.shape
    n_p = x.count(0)

    S_szn = sum(msign(x[i:]-x[i]).sum(0) for i in range(n))
    S_tot = S_szn.sum()

    n_tot = x.count()
    ties = count_tied_groups(x.compressed())
    corr_ties = sum(v*k*(k-1) for (k,v) in ties.items())
    denom_tot = ma.sqrt(1.*n_tot*(n_tot-1)*(n_tot*(n_tot-1)-corr_ties))/2.

    R = rankdata(x, axis=0, use_missing=True)
    K = ma.empty((m,m), dtype=int)
    covmat = ma.empty((m,m), dtype=float)
    denom_szn = ma.empty(m, dtype=float)
    for j in range(m):
        ties_j = count_tied_groups(x[:,j].compressed())
        corr_j = sum(v*k*(k-1) for (k,v) in ties_j.items())
        cmb = n_p[j]*(n_p[j]-1)
        for k in range(j,m,1):
            K[j,k] = sum(msign((x[i:,j]-x[i,j])*(x[i:,k]-x[i,k])).sum()
                         for i in range(n))
            covmat[j,k] = (K[j,k] + 4*(R[:,j]*R[:,k]).sum() -
                           n*(n_p[j]+1)*(n_p[k]+1))/3.
            K[k,j] = K[j,k]
            covmat[k,j] = covmat[j,k]

        denom_szn[j] = ma.sqrt(cmb*(cmb-corr_j)) / 2.

    var_szn = covmat.diagonal()

    z_szn = msign(S_szn) * (abs(S_szn)-1) / ma.sqrt(var_szn)
    z_tot_ind = msign(S_tot) * (abs(S_tot)-1) / ma.sqrt(var_szn.sum())
    z_tot_dep = msign(S_tot) * (abs(S_tot)-1) / ma.sqrt(covmat.sum())

    prob_szn = special.erfc(abs(z_szn.data)/np.sqrt(2))
    prob_tot_ind = special.erfc(abs(z_tot_ind)/np.sqrt(2))
    prob_tot_dep = special.erfc(abs(z_tot_dep)/np.sqrt(2))

    chi2_tot = (z_szn*z_szn).sum()
    chi2_trd = m * z_szn.mean()**2
    output = {'seasonal tau': S_szn/denom_szn,
              'global tau': S_tot/denom_tot,
              'global tau (alt)': S_tot/denom_szn.sum(),
              'seasonal p-value': prob_szn,
              'global p-value (indep)': prob_tot_ind,
              'global p-value (dep)': prob_tot_dep,
              'chi2 total': chi2_tot,
              'chi2 trend': chi2_trd,
              }
    return output


PointbiserialrResult = namedtuple('PointbiserialrResult', ('correlation',
                                                           'pvalue'))


def pointbiserialr(x, y):
    """Calculates a point biserial correlation coefficient and its p-value.

    Parameters
    ----------
    x : array_like of bools
        Input array.
    y : array_like
        Input array.

    Returns
    -------
    correlation : float
        R value
    pvalue : float
        2-tailed p-value

    Notes
    -----
    Missing values are considered pair-wise: if a value is missing in x,
    the corresponding value in y is masked.

    For more details on `pointbiserialr`, see `scipy.stats.pointbiserialr`.

    """
    x = ma.fix_invalid(x, copy=True).astype(bool)
    y = ma.fix_invalid(y, copy=True).astype(float)
    # Get rid of the missing data
    m = ma.mask_or(ma.getmask(x), ma.getmask(y))
    if m is not nomask:
        unmask = np.logical_not(m)
        x = x[unmask]
        y = y[unmask]

    n = len(x)
    # phat is the fraction of x values that are True
    phat = x.sum() / float(n)
    y0 = y[~x]  # y-values where x is False
    y1 = y[x]  # y-values where x is True
    y0m = y0.mean()
    y1m = y1.mean()

    rpb = (y1m - y0m)*np.sqrt(phat * (1-phat)) / y.std()

    df = n-2
    t = rpb*ma.sqrt(df/(1.0-rpb**2))
    prob = _betai(0.5*df, 0.5, df/(df+t*t))

    return PointbiserialrResult(rpb, prob)


def linregress(x, y=None):
    r"""
    Calculate a linear least-squares regression for two sets of measurements.

    Parameters
    ----------
    x, y : array_like
        Two sets of measurements.  Both arrays should have the same length N.  If
        only `x` is given (and ``y=None``), then it must be a two-dimensional
        array where one dimension has length 2.  The two sets of measurements
        are then found by splitting the array along the length-2 dimension. In
        the case where ``y=None`` and `x` is a 2xN array, ``linregress(x)`` is
        equivalent to ``linregress(x[0], x[1])``.

    Returns
    -------
    result : ``LinregressResult`` instance
        The return value is an object with the following attributes:

        slope : float
            Slope of the regression line.
        intercept : float
            Intercept of the regression line.
        rvalue : float
            The Pearson correlation coefficient. The square of ``rvalue``
            is equal to the coefficient of determination.
        pvalue : float
            The p-value for a hypothesis test whose null hypothesis is
            that the slope is zero, using Wald Test with t-distribution of
            the test statistic. See `alternative` above for alternative
            hypotheses.
        stderr : float
            Standard error of the estimated slope (gradient), under the
            assumption of residual normality.
        intercept_stderr : float
            Standard error of the estimated intercept, under the assumption
            of residual normality.

    See Also
    --------
    scipy.optimize.curve_fit :
        Use non-linear least squares to fit a function to data.
    scipy.optimize.leastsq :
        Minimize the sum of squares of a set of equations.

    Notes
    -----
    Missing values are considered pair-wise: if a value is missing in `x`,
    the corresponding value in `y` is masked.

    For compatibility with older versions of SciPy, the return value acts
    like a ``namedtuple`` of length 5, with fields ``slope``, ``intercept``,
    ``rvalue``, ``pvalue`` and ``stderr``, so one can continue to write::

        slope, intercept, r, p, se = linregress(x, y)

    With that style, however, the standard error of the intercept is not
    available.  To have access to all the computed values, including the
    standard error of the intercept, use the return value as an object
    with attributes, e.g.::

        result = linregress(x, y)
        print(result.intercept, result.intercept_stderr)

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy import stats
    >>> rng = np.random.default_rng()

    Generate some data:

    >>> x = rng.random(10)
    >>> y = 1.6*x + rng.random(10)

    Perform the linear regression:

    >>> res = stats.mstats.linregress(x, y)

    Coefficient of determination (R-squared):

    >>> print(f"R-squared: {res.rvalue**2:.6f}")
    R-squared: 0.717533

    Plot the data along with the fitted line:

    >>> plt.plot(x, y, 'o', label='original data')
    >>> plt.plot(x, res.intercept + res.slope*x, 'r', label='fitted line')
    >>> plt.legend()
    >>> plt.show()

    Calculate 95% confidence interval on slope and intercept:

    >>> # Two-sided inverse Students t-distribution
    >>> # p - probability, df - degrees of freedom
    >>> from scipy.stats import t
    >>> tinv = lambda p, df: abs(t.ppf(p/2, df))

    >>> ts = tinv(0.05, len(x)-2)
    >>> print(f"slope (95%): {res.slope:.6f} +/- {ts*res.stderr:.6f}")
    slope (95%): 1.453392 +/- 0.743465
    >>> print(f"intercept (95%): {res.intercept:.6f}"
    ...       f" +/- {ts*res.intercept_stderr:.6f}")
    intercept (95%): 0.616950 +/- 0.544475

    """
    if y is None:
        x = ma.array(x)
        if x.shape[0] == 2:
            x, y = x
        elif x.shape[1] == 2:
            x, y = x.T
        else:
            raise ValueError("If only `x` is given as input, "
                             "it has to be of shape (2, N) or (N, 2), "
                             f"provided shape was {x.shape}")
    else:
        x = ma.array(x)
        y = ma.array(y)

    x = x.flatten()
    y = y.flatten()

    if np.amax(x) == np.amin(x) and len(x) > 1:
        raise ValueError("Cannot calculate a linear regression "
                         "if all x values are identical")

    m = ma.mask_or(ma.getmask(x), ma.getmask(y), shrink=False)
    if m is not nomask:
        x = ma.array(x, mask=m)
        y = ma.array(y, mask=m)
        if np.any(~m):
            result = _stats_py.linregress(x.data[~m], y.data[~m])
        else:
            # All data is masked
            result = _stats_py.LinregressResult(slope=None, intercept=None,
                                                rvalue=None, pvalue=None,
                                                stderr=None,
                                                intercept_stderr=None)
    else:
        result = _stats_py.linregress(x.data, y.data)

    return result


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
    For more details on `theilslopes`, see `scipy.stats.theilslopes`.

    """
    y = ma.asarray(y).flatten()
    if x is None:
        x = ma.arange(len(y), dtype=float)
    else:
        x = ma.asarray(x).flatten()
        if len(x) != len(y):
            raise ValueError(f"Incompatible lengths ! ({len(y)}<>{len(x)})")

    m = ma.mask_or(ma.getmask(x), ma.getmask(y))
    y._mask = x._mask = m
    # Disregard any masked elements of x or y
    y = y.compressed()
    x = x.compressed().astype(float)
    # We now have unmasked arrays so can use `scipy.stats.theilslopes`
    return stats_theilslopes(y, x, alpha=alpha, method=method)


def siegelslopes(y, x=None, method="hierarchical"):
    r"""
    Computes the Siegel estimator for a set of points (x, y).

    `siegelslopes` implements a method for robust linear regression
    using repeated medians to fit a line to the points (x, y).
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
    For more details on `siegelslopes`, see `scipy.stats.siegelslopes`.

    """
    y = ma.asarray(y).ravel()
    if x is None:
        x = ma.arange(len(y), dtype=float)
    else:
        x = ma.asarray(x).ravel()
        if len(x) != len(y):
            raise ValueError(f"Incompatible lengths ! ({len(y)}<>{len(x)})")

    m = ma.mask_or(ma.getmask(x), ma.getmask(y))
    y._mask = x._mask = m
    # Disregard any masked elements of x or y
    y = y.compressed()
    x = x.compressed().astype(float)
    # We now have unmasked arrays so can use `scipy.stats.siegelslopes`
    return stats_siegelslopes(y, x, method=method)


SenSeasonalSlopesResult = _make_tuple_bunch('SenSeasonalSlopesResult',
                                            ['intra_slope', 'inter_slope'])


def sen_seasonal_slopes(x):
    r"""
    Computes seasonal Theil-Sen and Kendall slope estimators.

    The seasonal generalization of Sen's slope computes the slopes between all
    pairs of values within a "season" (column) of a 2D array. It returns an
    array containing the median of these "within-season" slopes for each
    season (the Theil-Sen slope estimator of each season), and it returns the
    median of the within-season slopes across all seasons (the seasonal Kendall
    slope estimator).

    Parameters
    ----------
    x : 2D array_like
        Each column of `x` contains measurements of the dependent variable
        within a season. The independent variable (usually time) of each season
        is assumed to be ``np.arange(x.shape[0])``.

    Returns
    -------
    result : ``SenSeasonalSlopesResult`` instance
        The return value is an object with the following attributes:

        intra_slope : ndarray
            For each season, the Theil-Sen slope estimator: the median of
            within-season slopes.
        inter_slope : float
            The seasonal Kendall slope estimator: the median of within-season
            slopes *across all* seasons.

    See Also
    --------
    theilslopes : the analogous function for non-seasonal data
    scipy.stats.theilslopes : non-seasonal slopes for non-masked arrays

    Notes
    -----
    The slopes :math:`d_{ijk}` within season :math:`i` are:

    .. math::

        d_{ijk} = \frac{x_{ij} - x_{ik}}
                            {j - k}

    for pairs of distinct integer indices :math:`j, k` of :math:`x`.

    Element :math:`i` of the returned `intra_slope` array is the median of the
    :math:`d_{ijk}` over all :math:`j < k`; this is the Theil-Sen slope
    estimator of season :math:`i`. The returned `inter_slope` value, better
    known as the seasonal Kendall slope estimator, is the median of the
    :math:`d_{ijk}` over all :math:`i, j, k`.

    References
    ----------
    .. [1] Hirsch, Robert M., James R. Slack, and Richard A. Smith.
           "Techniques of trend analysis for monthly water quality data."
           *Water Resources Research* 18.1 (1982): 107-121.

    Examples
    --------
    Suppose we have 100 observations of a dependent variable for each of four
    seasons:

    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> x = rng.random(size=(100, 4))

    We compute the seasonal slopes as:

    >>> from scipy import stats
    >>> intra_slope, inter_slope = stats.mstats.sen_seasonal_slopes(x)

    If we define a function to compute all slopes between observations within
    a season:

    >>> def dijk(yi):
    ...     n = len(yi)
    ...     x = np.arange(n)
    ...     dy = yi - yi[:, np.newaxis]
    ...     dx = x - x[:, np.newaxis]
    ...     # we only want unique pairs of distinct indices
    ...     mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    ...     return dy[mask]/dx[mask]

    then element ``i`` of ``intra_slope`` is the median of ``dijk[x[:, i]]``:

    >>> i = 2
    >>> np.allclose(np.median(dijk(x[:, i])), intra_slope[i])
    True

    and ``inter_slope`` is the median of the values returned by ``dijk`` for
    all seasons:

    >>> all_slopes = np.concatenate([dijk(x[:, i]) for i in range(x.shape[1])])
    >>> np.allclose(np.median(all_slopes), inter_slope)
    True

    Because the data are randomly generated, we would expect the median slopes
    to be nearly zero both within and across all seasons, and indeed they are:

    >>> intra_slope.data
    array([ 0.00124504, -0.00277761, -0.00221245, -0.00036338])
    >>> inter_slope
    -0.0010511779872922058

    """
    x = ma.array(x, subok=True, copy=False, ndmin=2)
    (n,_) = x.shape
    # Get list of slopes per season
    szn_slopes = ma.vstack([(x[i+1:]-x[i])/np.arange(1,n-i)[:,None]
                            for i in range(n)])
    szn_medslopes = ma.median(szn_slopes, axis=0)
    medslope = ma.median(szn_slopes, axis=None)
    return SenSeasonalSlopesResult(szn_medslopes, medslope)


Ttest_1sampResult = namedtuple('Ttest_1sampResult', ('statistic', 'pvalue'))


def ttest_1samp(a, popmean, axis=0, alternative='two-sided'):
    """
    Calculates the T-test for the mean of ONE group of scores.

    Parameters
    ----------
    a : array_like
        sample observation
    popmean : float or array_like
        expected value in null hypothesis, if array_like than it must have the
        same shape as `a` excluding the axis dimension
    axis : int or None, optional
        Axis along which to compute test. If None, compute over the whole
        array `a`.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

        * 'two-sided': the mean of the underlying distribution of the sample
          is different than the given population mean (`popmean`)
        * 'less': the mean of the underlying distribution of the sample is
          less than the given population mean (`popmean`)
        * 'greater': the mean of the underlying distribution of the sample is
          greater than the given population mean (`popmean`)

        .. versionadded:: 1.7.0

    Returns
    -------
    statistic : float or array
        t-statistic
    pvalue : float or array
        The p-value

    Notes
    -----
    For more details on `ttest_1samp`, see `scipy.stats.ttest_1samp`.

    """
    a, axis = _chk_asarray(a, axis)
    if a.size == 0:
        return (np.nan, np.nan)

    x = a.mean(axis=axis)
    v = a.var(axis=axis, ddof=1)
    n = a.count(axis=axis)
    # force df to be an array for masked division not to throw a warning
    df = ma.asanyarray(n - 1.0)
    svar = ((n - 1.0) * v) / df
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (x - popmean) / ma.sqrt(svar / n)

    t, prob = _ttest_finish(df, t, alternative)
    return Ttest_1sampResult(t, prob)


ttest_onesamp = ttest_1samp


Ttest_indResult = namedtuple('Ttest_indResult', ('statistic', 'pvalue'))


def ttest_ind(a, b, axis=0, equal_var=True, alternative='two-sided'):
    """
    Calculates the T-test for the means of TWO INDEPENDENT samples of scores.

    Parameters
    ----------
    a, b : array_like
        The arrays must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default).
    axis : int or None, optional
        Axis along which to compute test. If None, compute over the whole
        arrays, `a`, and `b`.
    equal_var : bool, optional
        If True, perform a standard independent 2 sample test that assumes equal
        population variances.
        If False, perform Welch's t-test, which does not assume equal population
        variance.

        .. versionadded:: 0.17.0
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

        * 'two-sided': the means of the distributions underlying the samples
          are unequal.
        * 'less': the mean of the distribution underlying the first sample
          is less than the mean of the distribution underlying the second
          sample.
        * 'greater': the mean of the distribution underlying the first
          sample is greater than the mean of the distribution underlying
          the second sample.

        .. versionadded:: 1.7.0

    Returns
    -------
    statistic : float or array
        The calculated t-statistic.
    pvalue : float or array
        The p-value.

    Notes
    -----
    For more details on `ttest_ind`, see `scipy.stats.ttest_ind`.

    """
    a, b, axis = _chk2_asarray(a, b, axis)

    if a.size == 0 or b.size == 0:
        return Ttest_indResult(np.nan, np.nan)

    (x1, x2) = (a.mean(axis), b.mean(axis))
    (v1, v2) = (a.var(axis=axis, ddof=1), b.var(axis=axis, ddof=1))
    (n1, n2) = (a.count(axis), b.count(axis))

    if equal_var:
        # force df to be an array for masked division not to throw a warning
        df = ma.asanyarray(n1 + n2 - 2.0)
        svar = ((n1-1)*v1+(n2-1)*v2) / df
        denom = ma.sqrt(svar*(1.0/n1 + 1.0/n2))  # n-D computation here!
    else:
        vn1 = v1/n1
        vn2 = v2/n2
        with np.errstate(divide='ignore', invalid='ignore'):
            df = (vn1 + vn2)**2 / (vn1**2 / (n1 - 1) + vn2**2 / (n2 - 1))

        # If df is undefined, variances are zero.
        # It doesn't matter what df is as long as it is not NaN.
        df = np.where(np.isnan(df), 1, df)
        denom = ma.sqrt(vn1 + vn2)

    with np.errstate(divide='ignore', invalid='ignore'):
        t = (x1-x2) / denom

    t, prob = _ttest_finish(df, t, alternative)
    return Ttest_indResult(t, prob)


Ttest_relResult = namedtuple('Ttest_relResult', ('statistic', 'pvalue'))


def ttest_rel(a, b, axis=0, alternative='two-sided'):
    """
    Calculates the T-test on TWO RELATED samples of scores, a and b.

    Parameters
    ----------
    a, b : array_like
        The arrays must have the same shape.
    axis : int or None, optional
        Axis along which to compute test. If None, compute over the whole
        arrays, `a`, and `b`.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

        * 'two-sided': the means of the distributions underlying the samples
          are unequal.
        * 'less': the mean of the distribution underlying the first sample
          is less than the mean of the distribution underlying the second
          sample.
        * 'greater': the mean of the distribution underlying the first
          sample is greater than the mean of the distribution underlying
          the second sample.

        .. versionadded:: 1.7.0

    Returns
    -------
    statistic : float or array
        t-statistic
    pvalue : float or array
        two-tailed p-value

    Notes
    -----
    For more details on `ttest_rel`, see `scipy.stats.ttest_rel`.

    """
    a, b, axis = _chk2_asarray(a, b, axis)
    if len(a) != len(b):
        raise ValueError('unequal length arrays')

    if a.size == 0 or b.size == 0:
        return Ttest_relResult(np.nan, np.nan)

    n = a.count(axis)
    df = ma.asanyarray(n-1.0)
    d = (a-b).astype('d')
    dm = d.mean(axis)
    v = d.var(axis=axis, ddof=1)
    denom = ma.sqrt(v / n)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = dm / denom

    t, prob = _ttest_finish(df, t, alternative)
    return Ttest_relResult(t, prob)


MannwhitneyuResult = namedtuple('MannwhitneyuResult', ('statistic',
                                                       'pvalue'))


def mannwhitneyu(x,y, use_continuity=True):
    """
    Computes the Mann-Whitney statistic

    Missing values in `x` and/or `y` are discarded.

    Parameters
    ----------
    x : sequence
        Input
    y : sequence
        Input
    use_continuity : {True, False}, optional
        Whether a continuity correction (1/2.) should be taken into account.

    Returns
    -------
    statistic : float
        The minimum of the Mann-Whitney statistics
    pvalue : float
        Approximate two-sided p-value assuming a normal distribution.

    """
    x = ma.asarray(x).compressed().view(ndarray)
    y = ma.asarray(y).compressed().view(ndarray)
    ranks = rankdata(np.concatenate([x,y]))
    (nx, ny) = (len(x), len(y))
    nt = nx + ny
    U = ranks[:nx].sum() - nx*(nx+1)/2.
    U = max(U, nx*ny - U)
    u = nx*ny - U

    mu = (nx*ny)/2.
    sigsq = (nt**3 - nt)/12.
    ties = count_tied_groups(ranks)
    sigsq -= sum(v*(k**3-k) for (k,v) in ties.items())/12.
    sigsq *= nx*ny/float(nt*(nt-1))

    if use_continuity:
        z = (U - 1/2. - mu) / ma.sqrt(sigsq)
    else:
        z = (U - mu) / ma.sqrt(sigsq)

    prob = special.erfc(abs(z)/np.sqrt(2))
    return MannwhitneyuResult(u, prob)


KruskalResult = namedtuple('KruskalResult', ('statistic', 'pvalue'))


def kruskal(*args):
    """
    Compute the Kruskal-Wallis H-test for independent samples

    Parameters
    ----------
    sample1, sample2, ... : array_like
       Two or more arrays with the sample measurements can be given as
       arguments.

    Returns
    -------
    statistic : float
       The Kruskal-Wallis H statistic, corrected for ties
    pvalue : float
       The p-value for the test using the assumption that H has a chi
       square distribution

    Notes
    -----
    For more details on `kruskal`, see `scipy.stats.kruskal`.

    Examples
    --------
    >>> from scipy.stats.mstats import kruskal

    Random samples from three different brands of batteries were tested
    to see how long the charge lasted. Results were as follows:

    >>> a = [6.3, 5.4, 5.7, 5.2, 5.0]
    >>> b = [6.9, 7.0, 6.1, 7.9]
    >>> c = [7.2, 6.9, 6.1, 6.5]

    Test the hypothesis that the distribution functions for all of the brands'
    durations are identical. Use 5% level of significance.

    >>> kruskal(a, b, c)
    KruskalResult(statistic=7.113812154696133, pvalue=0.028526948491942164)

    The null hypothesis is rejected at the 5% level of significance
    because the returned p-value is less than the critical value of 5%.

    """
    output = argstoarray(*args)
    ranks = ma.masked_equal(rankdata(output, use_missing=False), 0)
    sumrk = ranks.sum(-1)
    ngrp = ranks.count(-1)
    ntot = ranks.count()
    H = 12./(ntot*(ntot+1)) * (sumrk**2/ngrp).sum() - 3*(ntot+1)
    # Tie correction
    ties = count_tied_groups(ranks)
    T = 1. - sum(v*(k**3-k) for (k,v) in ties.items())/float(ntot**3-ntot)
    if T == 0:
        raise ValueError('All numbers are identical in kruskal')

    H /= T
    df = len(output) - 1
    prob = distributions.chi2.sf(H, df)
    return KruskalResult(H, prob)


kruskalwallis = kruskal


@_rename_parameter("mode", "method")
def ks_1samp(x, cdf, args=(), alternative="two-sided", method='auto'):
    """
    Computes the Kolmogorov-Smirnov test on one sample of masked values.

    Missing values in `x` are discarded.

    Parameters
    ----------
    x : array_like
        a 1-D array of observations of random variables.
    cdf : str or callable
        If a string, it should be the name of a distribution in `scipy.stats`.
        If a callable, that callable is used to calculate the cdf.
    args : tuple, sequence, optional
        Distribution parameters, used if `cdf` is a string.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Indicates the alternative hypothesis.  Default is 'two-sided'.
    method : {'auto', 'exact', 'asymp'}, optional
        Defines the method used for calculating the p-value.
        The following options are available (default is 'auto'):

          * 'auto' : use 'exact' for small size arrays, 'asymp' for large
          * 'exact' : use approximation to exact distribution of test statistic
          * 'asymp' : use asymptotic distribution of test statistic

    Returns
    -------
    d : float
        Value of the Kolmogorov Smirnov test
    p : float
        Corresponding p-value.

    """
    alternative = {'t': 'two-sided', 'g': 'greater', 'l': 'less'}.get(
       alternative.lower()[0], alternative)
    return scipy.stats._stats_py.ks_1samp(
        x, cdf, args=args, alternative=alternative, method=method)


@_rename_parameter("mode", "method")
def ks_2samp(data1, data2, alternative="two-sided", method='auto'):
    """
    Computes the Kolmogorov-Smirnov test on two samples.

    Missing values in `x` and/or `y` are discarded.

    Parameters
    ----------
    data1 : array_like
        First data set
    data2 : array_like
        Second data set
    alternative : {'two-sided', 'less', 'greater'}, optional
        Indicates the alternative hypothesis.  Default is 'two-sided'.
    method : {'auto', 'exact', 'asymp'}, optional
        Defines the method used for calculating the p-value.
        The following options are available (default is 'auto'):

          * 'auto' : use 'exact' for small size arrays, 'asymp' for large
          * 'exact' : use approximation to exact distribution of test statistic
          * 'asymp' : use asymptotic distribution of test statistic

    Returns
    -------
    d : float
        Value of the Kolmogorov Smirnov test
    p : float
        Corresponding p-value.

    """
    # Ideally this would be accomplished by
    # ks_2samp = scipy.stats._stats_py.ks_2samp
    # but the circular dependencies between _mstats_basic and stats prevent that.
    alternative = {'t': 'two-sided', 'g': 'greater', 'l': 'less'}.get(
       alternative.lower()[0], alternative)
    return scipy.stats._stats_py.ks_2samp(data1, data2,
                                          alternative=alternative,
                                          method=method)


ks_twosamp = ks_2samp


@_rename_parameter("mode", "method")
def kstest(data1, data2, args=(), alternative='two-sided', method='auto'):
    """

    Parameters
    ----------
    data1 : array_like
    data2 : str, callable or array_like
    args : tuple, sequence, optional
        Distribution parameters, used if `data1` or `data2` are strings.
    alternative : str, as documented in stats.kstest
    method : str, as documented in stats.kstest

    Returns
    -------
    tuple of (K-S statistic, probability)

    """
    return scipy.stats._stats_py.kstest(data1, data2, args,
                                        alternative=alternative, method=method)


def trima(a, limits=None, inclusive=(True,True)):
    """
    Trims an array by masking the data outside some given limits.

    Returns a masked version of the input array.

    Parameters
    ----------
    a : array_like
        Input array.
    limits : {None, tuple}, optional
        Tuple of (lower limit, upper limit) in absolute values.
        Values of the input array lower (greater) than the lower (upper) limit
        will be masked.  A limit is None indicates an open interval.
    inclusive : (bool, bool) tuple, optional
        Tuple of (lower flag, upper flag), indicating whether values exactly
        equal to the lower (upper) limit are allowed.

    Examples
    --------
    >>> from scipy.stats.mstats import trima
    >>> import numpy as np

    >>> a = np.arange(10)

    The interval is left-closed and right-open, i.e., `[2, 8)`.
    Trim the array by keeping only values in the interval.

    >>> trima(a, limits=(2, 8), inclusive=(True, False))
    masked_array(data=[--, --, 2, 3, 4, 5, 6, 7, --, --],
                 mask=[ True,  True, False, False, False, False, False, False,
                        True,  True],
           fill_value=999999)

    """
    a = ma.asarray(a)
    a.unshare_mask()
    if (limits is None) or (limits == (None, None)):
        return a

    (lower_lim, upper_lim) = limits
    (lower_in, upper_in) = inclusive
    condition = False
    if lower_lim is not None:
        if lower_in:
            condition |= (a < lower_lim)
        else:
            condition |= (a <= lower_lim)

    if upper_lim is not None:
        if upper_in:
            condition |= (a > upper_lim)
        else:
            condition |= (a >= upper_lim)

    a[condition.filled(True)] = masked
    return a


def trimr(a, limits=None, inclusive=(True, True), axis=None):
    """
    Trims an array by masking some proportion of the data on each end.
    Returns a masked version of the input array.

    Parameters
    ----------
    a : sequence
        Input array.
    limits : {None, tuple}, optional
        Tuple of the percentages to cut on each side of the array, with respect
        to the number of unmasked data, as floats between 0. and 1.
        Noting n the number of unmasked data before trimming, the
        (n*limits[0])th smallest data and the (n*limits[1])th largest data are
        masked, and the total number of unmasked data after trimming is
        n*(1.-sum(limits)).  The value of one limit can be set to None to
        indicate an open interval.
    inclusive : {(True,True) tuple}, optional
        Tuple of flags indicating whether the number of data being masked on
        the left (right) end should be truncated (True) or rounded (False) to
        integers.
    axis : {None,int}, optional
        Axis along which to trim. If None, the whole array is trimmed, but its
        shape is maintained.

    """
    def _trimr1D(a, low_limit, up_limit, low_inclusive, up_inclusive):
        n = a.count()
        idx = a.argsort()
        if low_limit:
            if low_inclusive:
                lowidx = int(low_limit*n)
            else:
                lowidx = int(np.round(low_limit*n))
            a[idx[:lowidx]] = masked
        if up_limit is not None:
            if up_inclusive:
                upidx = n - int(n*up_limit)
            else:
                upidx = n - int(np.round(n*up_limit))
            a[idx[upidx:]] = masked
        return a

    a = ma.asarray(a)
    a.unshare_mask()
    if limits is None:
        return a

    # Check the limits
    (lolim, uplim) = limits
    errmsg = "The proportion to cut from the %s should be between 0. and 1."
    if lolim is not None:
        if lolim > 1. or lolim < 0:
            raise ValueError(errmsg % 'beginning' + f"(got {lolim})")
    if uplim is not None:
        if uplim > 1. or uplim < 0:
            raise ValueError(errmsg % 'end' + f"(got {uplim})")

    (loinc, upinc) = inclusive

    if axis is None:
        shp = a.shape
        return _trimr1D(a.ravel(),lolim,uplim,loinc,upinc).reshape(shp)
    else:
        return ma.apply_along_axis(_trimr1D, axis, a, lolim,uplim,loinc,upinc)


trimdoc = """
    Parameters
    ----------
    a : sequence
        Input array
    limits : {None, tuple}, optional
        If `relative` is False, tuple (lower limit, upper limit) in absolute values.
        Values of the input array lower (greater) than the lower (upper) limit are
        masked.

        If `relative` is True, tuple (lower percentage, upper percentage) to cut
        on each side of the  array, with respect to the number of unmasked data.

        Noting n the number of unmasked data before trimming, the (n*limits[0])th
        smallest data and the (n*limits[1])th largest data are masked, and the
        total number of unmasked data after trimming is n*(1.-sum(limits))
        In each case, the value of one limit can be set to None to indicate an
        open interval.

        If limits is None, no trimming is performed
    inclusive : {(bool, bool) tuple}, optional
        If `relative` is False, tuple indicating whether values exactly equal
        to the absolute limits are allowed.
        If `relative` is True, tuple indicating whether the number of data
        being masked on each side should be rounded (True) or truncated
        (False).
    relative : bool, optional
        Whether to consider the limits as absolute values (False) or proportions
        to cut (True).
    axis : int, optional
        Axis along which to trim.
"""


def trim(a, limits=None, inclusive=(True,True), relative=False, axis=None):
    """
    Trims an array by masking the data outside some given limits.

    Returns a masked version of the input array.

    %s

    Examples
    --------
    >>> from scipy.stats.mstats import trim
    >>> z = [ 1, 2, 3, 4, 5, 6, 7, 8, 9,10]
    >>> print(trim(z,(3,8)))
    [-- -- 3 4 5 6 7 8 -- --]
    >>> print(trim(z,(0.1,0.2),relative=True))
    [-- 2 3 4 5 6 7 8 -- --]

    """
    if relative:
        return trimr(a, limits=limits, inclusive=inclusive, axis=axis)
    else:
        return trima(a, limits=limits, inclusive=inclusive)


if trim.__doc__:
    trim.__doc__ = trim.__doc__ % trimdoc


def trimboth(data, proportiontocut=0.2, inclusive=(True,True), axis=None):
    """
    Trims the smallest and largest data values.

    Trims the `data` by masking the ``int(proportiontocut * n)`` smallest and
    ``int(proportiontocut * n)`` largest values of data along the given axis,
    where n is the number of unmasked values before trimming.

    Parameters
    ----------
    data : ndarray
        Data to trim.
    proportiontocut : float, optional
        Percentage of trimming (as a float between 0 and 1).
        If n is the number of unmasked values before trimming, the number of
        values after trimming is ``(1 - 2*proportiontocut) * n``.
        Default is 0.2.
    inclusive : {(bool, bool) tuple}, optional
        Tuple indicating whether the number of data being masked on each side
        should be rounded (True) or truncated (False).
    axis : int, optional
        Axis along which to perform the trimming.
        If None, the input array is first flattened.

    """
    return trimr(data, limits=(proportiontocut,proportiontocut),
                 inclusive=inclusive, axis=axis)


def trimtail(data, proportiontocut=0.2, tail='left', inclusive=(True,True),
             axis=None):
    """
    Trims the data by masking values from one tail.

    Parameters
    ----------
    data : array_like
        Data to trim.
    proportiontocut : float, optional
        Percentage of trimming. If n is the number of unmasked values
        before trimming, the number of values after trimming is
        ``(1 - proportiontocut) * n``.  Default is 0.2.
    tail : {'left','right'}, optional
        If 'left' the `proportiontocut` lowest values will be masked.
        If 'right' the `proportiontocut` highest values will be masked.
        Default is 'left'.
    inclusive : {(bool, bool) tuple}, optional
        Tuple indicating whether the number of data being masked on each side
        should be rounded (True) or truncated (False).  Default is
        (True, True).
    axis : int, optional
        Axis along which to perform the trimming.
        If None, the input array is first flattened.  Default is None.

    Returns
    -------
    trimtail : ndarray
        Returned array of same shape as `data` with masked tail values.

    """
    tail = str(tail).lower()[0]
    if tail == 'l':
        limits = (proportiontocut,None)
    elif tail == 'r':
        limits = (None, proportiontocut)
    else:
        raise TypeError("The tail argument should be in ('left','right')")

    return trimr(data, limits=limits, axis=axis, inclusive=inclusive)


trim1 = trimtail


def trimmed_mean(a, limits=(0.1,0.1), inclusive=(1,1), relative=True,
                 axis=None):
    """Returns the trimmed mean of the data along the given axis.

    %s

    """
    if (not isinstance(limits,tuple)) and isinstance(limits,float):
        limits = (limits, limits)
    if relative:
        return trimr(a,limits=limits,inclusive=inclusive,axis=axis).mean(axis=axis)
    else:
        return trima(a,limits=limits,inclusive=inclusive).mean(axis=axis)


if trimmed_mean.__doc__:
    trimmed_mean.__doc__ = trimmed_mean.__doc__ % trimdoc


def trimmed_var(a, limits=(0.1,0.1), inclusive=(1,1), relative=True,
                axis=None, ddof=0):
    """Returns the trimmed variance of the data along the given axis.

    %s
    ddof : {0,integer}, optional
        Means Delta Degrees of Freedom. The denominator used during computations
        is (n-ddof). DDOF=0 corresponds to a biased estimate, DDOF=1 to an un-
        biased estimate of the variance.

    """
    if (not isinstance(limits,tuple)) and isinstance(limits,float):
        limits = (limits, limits)
    if relative:
        out = trimr(a,limits=limits, inclusive=inclusive,axis=axis)
    else:
        out = trima(a,limits=limits,inclusive=inclusive)

    return out.var(axis=axis, ddof=ddof)


if trimmed_var.__doc__:
    trimmed_var.__doc__ = trimmed_var.__doc__ % trimdoc


def trimmed_std(a, limits=(0.1,0.1), inclusive=(1,1), relative=True,
                axis=None, ddof=0):
    """Returns the trimmed standard deviation of the data along the given axis.

    %s
    ddof : {0,integer}, optional
        Means Delta Degrees of Freedom. The denominator used during computations
        is (n-ddof). DDOF=0 corresponds to a biased estimate, DDOF=1 to an un-
        biased estimate of the variance.

    """
    if (not isinstance(limits,tuple)) and isinstance(limits,float):
        limits = (limits, limits)
    if relative:
        out = trimr(a,limits=limits,inclusive=inclusive,axis=axis)
    else:
        out = trima(a,limits=limits,inclusive=inclusive)
    return out.std(axis=axis,ddof=ddof)


if trimmed_std.__doc__:
    trimmed_std.__doc__ = trimmed_std.__doc__ % trimdoc


def trimmed_stde(a, limits=(0.1,0.1), inclusive=(1,1), axis=None):
    """
    Returns the standard error of the trimmed mean along the given axis.

    Parameters
    ----------
    a : sequence
        Input array
    limits : {(0.1,0.1), tuple of float}, optional
        tuple (lower percentage, upper percentage) to cut  on each side of the
        array, with respect to the number of unmasked data.

        If n is the number of unmasked data before trimming, the values
        smaller than ``n * limits[0]`` and the values larger than
        ``n * `limits[1]`` are masked, and the total number of unmasked
        data after trimming is ``n * (1.-sum(limits))``.  In each case,
        the value of one limit can be set to None to indicate an open interval.
        If `limits` is None, no trimming is performed.
    inclusive : {(bool, bool) tuple} optional
        Tuple indicating whether the number of data being masked on each side
        should be rounded (True) or truncated (False).
    axis : int, optional
        Axis along which to trim.

    Returns
    -------
    trimmed_stde : scalar or ndarray

    """
    def _trimmed_stde_1D(a, low_limit, up_limit, low_inclusive, up_inclusive):
        "Returns the standard error of the trimmed mean for a 1D input data."
        n = a.count()
        idx = a.argsort()
        if low_limit:
            if low_inclusive:
                lowidx = int(low_limit*n)
            else:
                lowidx = np.round(low_limit*n)
            a[idx[:lowidx]] = masked
        if up_limit is not None:
            if up_inclusive:
                upidx = n - int(n*up_limit)
            else:
                upidx = n - np.round(n*up_limit)
            a[idx[upidx:]] = masked
        a[idx[:lowidx]] = a[idx[lowidx]]
        a[idx[upidx:]] = a[idx[upidx-1]]
        winstd = a.std(ddof=1)
        return winstd / ((1-low_limit-up_limit)*np.sqrt(len(a)))

    a = ma.array(a, copy=True, subok=True)
    a.unshare_mask()
    if limits is None:
        return a.std(axis=axis,ddof=1)/ma.sqrt(a.count(axis))
    if (not isinstance(limits,tuple)) and isinstance(limits,float):
        limits = (limits, limits)

    # Check the limits
    (lolim, uplim) = limits
    errmsg = "The proportion to cut from the %s should be between 0. and 1."
    if lolim is not None:
        if lolim > 1. or lolim < 0:
            raise ValueError(errmsg % 'beginning' + f"(got {lolim})")
    if uplim is not None:
        if uplim > 1. or uplim < 0:
            raise ValueError(errmsg % 'end' + f"(got {uplim})")

    (loinc, upinc) = inclusive
    if (axis is None):
        return _trimmed_stde_1D(a.ravel(),lolim,uplim,loinc,upinc)
    else:
        if a.ndim > 2:
            raise ValueError("Array 'a' must be at most two dimensional, "
                             "but got a.ndim = %d" % a.ndim)
        return ma.apply_along_axis(_trimmed_stde_1D, axis, a,
                                   lolim,uplim,loinc,upinc)


def _mask_to_limits(a, limits, inclusive):
    """Mask an array for values outside of given limits.

    This is primarily a utility function.

    Parameters
    ----------
    a : array
    limits : (float or None, float or None)
    A tuple consisting of the (lower limit, upper limit).  Values in the
    input array less than the lower limit or greater than the upper limit
    will be masked out. None implies no limit.
    inclusive : (bool, bool)
    A tuple consisting of the (lower flag, upper flag).  These flags
    determine whether values exactly equal to lower or upper are allowed.

    Returns
    -------
    A MaskedArray.

    Raises
    ------
    A ValueError if there are no values within the given limits.
    """
    lower_limit, upper_limit = limits
    lower_include, upper_include = inclusive
    am = ma.MaskedArray(a)
    if lower_limit is not None:
        if lower_include:
            am = ma.masked_less(am, lower_limit)
        else:
            am = ma.masked_less_equal(am, lower_limit)

    if upper_limit is not None:
        if upper_include:
            am = ma.masked_greater(am, upper_limit)
        else:
            am = ma.masked_greater_equal(am, upper_limit)

    if am.count() == 0:
        raise ValueError("No array values within given limits")

    return am


def tmean(a, limits=None, inclusive=(True, True), axis=None):
    """
    Compute the trimmed mean.

    Parameters
    ----------
    a : array_like
        Array of values.
    limits : None or (lower limit, upper limit), optional
        Values in the input array less than the lower limit or greater than the
        upper limit will be ignored.  When limits is None (default), then all
        values are used.  Either of the limit values in the tuple can also be
        None representing a half-open interval.
    inclusive : (bool, bool), optional
        A tuple consisting of the (lower flag, upper flag).  These flags
        determine whether values exactly equal to the lower or upper limits
        are included.  The default value is (True, True).
    axis : int or None, optional
        Axis along which to operate. If None, compute over the
        whole array. Default is None.

    Returns
    -------
    tmean : float

    Notes
    -----
    For more details on `tmean`, see `scipy.stats.tmean`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import mstats
    >>> a = np.array([[6, 8, 3, 0],
    ...               [3, 9, 1, 2],
    ...               [8, 7, 8, 2],
    ...               [5, 6, 0, 2],
    ...               [4, 5, 5, 2]])
    ...
    ...
    >>> mstats.tmean(a, (2,5))
    3.3
    >>> mstats.tmean(a, (2,5), axis=0)
    masked_array(data=[4.0, 5.0, 4.0, 2.0],
                 mask=[False, False, False, False],
           fill_value=1e+20)

    """
    return trima(a, limits=limits, inclusive=inclusive).mean(axis=axis)


def tvar(a, limits=None, inclusive=(True, True), axis=0, ddof=1):
    """
    Compute the trimmed variance

    This function computes the sample variance of an array of values,
    while ignoring values which are outside of given `limits`.

    Parameters
    ----------
    a : array_like
        Array of values.
    limits : None or (lower limit, upper limit), optional
        Values in the input array less than the lower limit or greater than the
        upper limit will be ignored. When limits is None, then all values are
        used. Either of the limit values in the tuple can also be None
        representing a half-open interval.  The default value is None.
    inclusive : (bool, bool), optional
        A tuple consisting of the (lower flag, upper flag).  These flags
        determine whether values exactly equal to the lower or upper limits
        are included.  The default value is (True, True).
    axis : int or None, optional
        Axis along which to operate. If None, compute over the
        whole array. Default is zero.
    ddof : int, optional
        Delta degrees of freedom. Default is 1.

    Returns
    -------
    tvar : float
        Trimmed variance.

    Notes
    -----
    For more details on `tvar`, see `scipy.stats.tvar`.

    """
    a = a.astype(float).ravel()
    if limits is None:
        n = (~a.mask).sum()  # todo: better way to do that?
        return np.ma.var(a) * n/(n-1.)
    am = _mask_to_limits(a, limits=limits, inclusive=inclusive)

    return np.ma.var(am, axis=axis, ddof=ddof)


def tmin(a, lowerlimit=None, axis=0, inclusive=True):
    """
    Compute the trimmed minimum

    Parameters
    ----------
    a : array_like
        array of values
    lowerlimit : None or float, optional
        Values in the input array less than the given limit will be ignored.
        When lowerlimit is None, then all values are used. The default value
        is None.
    axis : int or None, optional
        Axis along which to operate. Default is 0. If None, compute over the
        whole array `a`.
    inclusive : {True, False}, optional
        This flag determines whether values exactly equal to the lower limit
        are included.  The default value is True.

    Returns
    -------
    tmin : float, int or ndarray

    Notes
    -----
    For more details on `tmin`, see `scipy.stats.tmin`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import mstats
    >>> a = np.array([[6, 8, 3, 0],
    ...               [3, 2, 1, 2],
    ...               [8, 1, 8, 2],
    ...               [5, 3, 0, 2],
    ...               [4, 7, 5, 2]])
    ...
    >>> mstats.tmin(a, 5)
    masked_array(data=[5, 7, 5, --],
                 mask=[False, False, False,  True],
           fill_value=999999)

    """
    a, axis = _chk_asarray(a, axis)
    am = trima(a, (lowerlimit, None), (inclusive, False))
    return ma.minimum.reduce(am, axis)


def tmax(a, upperlimit=None, axis=0, inclusive=True):
    """
    Compute the trimmed maximum

    This function computes the maximum value of an array along a given axis,
    while ignoring values larger than a specified upper limit.

    Parameters
    ----------
    a : array_like
        array of values
    upperlimit : None or float, optional
        Values in the input array greater than the given limit will be ignored.
        When upperlimit is None, then all values are used. The default value
        is None.
    axis : int or None, optional
        Axis along which to operate. Default is 0. If None, compute over the
        whole array `a`.
    inclusive : {True, False}, optional
        This flag determines whether values exactly equal to the upper limit
        are included.  The default value is True.

    Returns
    -------
    tmax : float, int or ndarray

    Notes
    -----
    For more details on `tmax`, see `scipy.stats.tmax`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import mstats
    >>> a = np.array([[6, 8, 3, 0],
    ...               [3, 9, 1, 2],
    ...               [8, 7, 8, 2],
    ...               [5, 6, 0, 2],
    ...               [4, 5, 5, 2]])
    ...
    ...
    >>> mstats.tmax(a, 4)
    masked_array(data=[4, --, 3, 2],
                 mask=[False,  True, False, False],
           fill_value=999999)

    """
    a, axis = _chk_asarray(a, axis)
    am = trima(a, (None, upperlimit), (False, inclusive))
    return ma.maximum.reduce(am, axis)


def tsem(a, limits=None, inclusive=(True, True), axis=0, ddof=1):
    """
    Compute the trimmed standard error of the mean.

    This function finds the standard error of the mean for given
    values, ignoring values outside the given `limits`.

    Parameters
    ----------
    a : array_like
        array of values
    limits : None or (lower limit, upper limit), optional
        Values in the input array less than the lower limit or greater than the
        upper limit will be ignored. When limits is None, then all values are
        used. Either of the limit values in the tuple can also be None
        representing a half-open interval.  The default value is None.
    inclusive : (bool, bool), optional
        A tuple consisting of the (lower flag, upper flag).  These flags
        determine whether values exactly equal to the lower or upper limits
        are included.  The default value is (True, True).
    axis : int or None, optional
        Axis along which to operate. If None, compute over the
        whole array. Default is zero.
    ddof : int, optional
        Delta degrees of freedom. Default is 1.

    Returns
    -------
    tsem : float

    Notes
    -----
    For more details on `tsem`, see `scipy.stats.tsem`.

    """
    a = ma.asarray(a).ravel()
    if limits is None:
        n = float(a.count())
        return a.std(axis=axis, ddof=ddof)/ma.sqrt(n)

    am = trima(a.ravel(), limits, inclusive)
    sd = np.sqrt(am.var(axis=axis, ddof=ddof))
    return sd / np.sqrt(am.count())


def winsorize(a, limits=None, inclusive=(True, True), inplace=False,
              axis=None, nan_policy='propagate'):
    """Returns a Winsorized version of the input array.

    The (limits[0])th lowest values are set to the (limits[0])th percentile,
    and the (limits[1])th highest values are set to the (1 - limits[1])th
    percentile.
    Masked values are skipped.


    Parameters
    ----------
    a : sequence
        Input array.
    limits : {None, tuple of float}, optional
        Tuple of the percentages to cut on each side of the array, with respect
        to the number of unmasked data, as floats between 0. and 1.
        Noting n the number of unmasked data before trimming, the
        (n*limits[0])th smallest data and the (n*limits[1])th largest data are
        masked, and the total number of unmasked data after trimming
        is n*(1.-sum(limits)) The value of one limit can be set to None to
        indicate an open interval.
    inclusive : {(True, True) tuple}, optional
        Tuple indicating whether the number of data being masked on each side
        should be truncated (True) or rounded (False).
    inplace : {False, True}, optional
        Whether to winsorize in place (True) or to use a copy (False)
    axis : {None, int}, optional
        Axis along which to trim. If None, the whole array is trimmed, but its
        shape is maintained.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):

          * 'propagate': allows nan values and may overwrite or propagate them
          * 'raise': throws an error
          * 'omit': performs the calculations ignoring nan values

    Notes
    -----
    This function is applied to reduce the effect of possibly spurious outliers
    by limiting the extreme values.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats.mstats import winsorize

    A shuffled array contains integers from 1 to 10.

    >>> a = np.array([10, 4, 9, 8, 5, 3, 7, 2, 1, 6])

    The 10% of the lowest value (i.e., ``1``) and the 20% of the highest
    values (i.e., ``9`` and ``10``) are replaced.

    >>> winsorize(a, limits=[0.1, 0.2])
    masked_array(data=[8, 4, 8, 8, 5, 3, 7, 2, 2, 6],
                 mask=False,
           fill_value=999999)

    """
    def _winsorize1D(a, low_limit, up_limit, low_include, up_include,
                     contains_nan, nan_policy):
        n = a.count()
        idx = a.argsort()
        if contains_nan:
            nan_count = np.count_nonzero(np.isnan(a))
        if low_limit:
            if low_include:
                lowidx = int(low_limit * n)
            else:
                lowidx = np.round(low_limit * n).astype(int)
            if contains_nan and nan_policy == 'omit':
                lowidx = min(lowidx, n-nan_count-1)
            a[idx[:lowidx]] = a[idx[lowidx]]
        if up_limit is not None:
            if up_include:
                upidx = n - int(n * up_limit)
            else:
                upidx = n - np.round(n * up_limit).astype(int)
            if contains_nan and nan_policy == 'omit':
                a[idx[upidx:-nan_count]] = a[idx[upidx - 1]]
            else:
                a[idx[upidx:]] = a[idx[upidx - 1]]
        return a

    contains_nan, nan_policy = _contains_nan(a, nan_policy)
    # We are going to modify a: better make a copy
    a = ma.array(a, copy=np.logical_not(inplace))

    if limits is None:
        return a
    if (not isinstance(limits, tuple)) and isinstance(limits, float):
        limits = (limits, limits)

    # Check the limits
    (lolim, uplim) = limits
    errmsg = "The proportion to cut from the %s should be between 0. and 1."
    if lolim is not None:
        if lolim > 1. or lolim < 0:
            raise ValueError(errmsg % 'beginning' + f"(got {lolim})")
    if uplim is not None:
        if uplim > 1. or uplim < 0:
            raise ValueError(errmsg % 'end' + f"(got {uplim})")

    (loinc, upinc) = inclusive

    if axis is None:
        shp = a.shape
        return _winsorize1D(a.ravel(), lolim, uplim, loinc, upinc,
                            contains_nan, nan_policy).reshape(shp)
    else:
        return ma.apply_along_axis(_winsorize1D, axis, a, lolim, uplim, loinc,
                                   upinc, contains_nan, nan_policy)


def moment(a, moment=1, axis=0):
    """
    Calculates the nth moment about the mean for a sample.

    Parameters
    ----------
    a : array_like
       data
    moment : int, optional
       order of central moment that is returned
    axis : int or None, optional
       Axis along which the central moment is computed. Default is 0.
       If None, compute over the whole array `a`.

    Returns
    -------
    n-th central moment : ndarray or float
       The appropriate moment along the given axis or over all values if axis
       is None. The denominator for the moment calculation is the number of
       observations, no degrees of freedom correction is done.

    Notes
    -----
    For more details about `moment`, see `scipy.stats.moment`.

    """
    a, axis = _chk_asarray(a, axis)
    if a.size == 0:
        moment_shape = list(a.shape)
        del moment_shape[axis]
        dtype = a.dtype.type if a.dtype.kind in 'fc' else np.float64
        # empty array, return nan(s) with shape matching `moment`
        out_shape = (moment_shape if np.isscalar(moment)
                     else [len(moment)] + moment_shape)
        if len(out_shape) == 0:
            return dtype(np.nan)
        else:
            return ma.array(np.full(out_shape, np.nan, dtype=dtype))

    # for array_like moment input, return a value for each.
    if not np.isscalar(moment):
        mean = a.mean(axis, keepdims=True)
        mmnt = [_moment(a, i, axis, mean=mean) for i in moment]
        return ma.array(mmnt)
    else:
        return _moment(a, moment, axis)


# Moment with optional pre-computed mean, equal to a.mean(axis, keepdims=True)
def _moment(a, moment, axis, *, mean=None):
    if np.abs(moment - np.round(moment)) > 0:
        raise ValueError("All moment parameters must be integers")

    if moment == 0 or moment == 1:
        # By definition the zeroth moment about the mean is 1, and the first
        # moment is 0.
        shape = list(a.shape)
        del shape[axis]
        dtype = a.dtype.type if a.dtype.kind in 'fc' else np.float64

        if len(shape) == 0:
            return dtype(1.0 if moment == 0 else 0.0)
        else:
            return (ma.ones(shape, dtype=dtype) if moment == 0
                    else ma.zeros(shape, dtype=dtype))
    else:
        # Exponentiation by squares: form exponent sequence
        n_list = [moment]
        current_n = moment
        while current_n > 2:
            if current_n % 2:
                current_n = (current_n-1)/2
            else:
                current_n /= 2
            n_list.append(current_n)

        # Starting point for exponentiation by squares
        mean = a.mean(axis, keepdims=True) if mean is None else mean
        a_zero_mean = a - mean
        if n_list[-1] == 1:
            s = a_zero_mean.copy()
        else:
            s = a_zero_mean**2

        # Perform multiplications
        for n in n_list[-2::-1]:
            s = s**2
            if n % 2:
                s *= a_zero_mean
        return s.mean(axis)


def variation(a, axis=0, ddof=0):
    """
    Compute the coefficient of variation.

    The coefficient of variation is the standard deviation divided by the
    mean.  This function is equivalent to::

        np.std(x, axis=axis, ddof=ddof) / np.mean(x)

    The default for ``ddof`` is 0, but many definitions of the coefficient
    of variation use the square root of the unbiased sample variance
    for the sample standard deviation, which corresponds to ``ddof=1``.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or None, optional
        Axis along which to calculate the coefficient of variation. Default
        is 0. If None, compute over the whole array `a`.
    ddof : int, optional
        Delta degrees of freedom.  Default is 0.

    Returns
    -------
    variation : ndarray
        The calculated variation along the requested axis.

    Notes
    -----
    For more details about `variation`, see `scipy.stats.variation`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats.mstats import variation
    >>> a = np.array([2,8,4])
    >>> variation(a)
    0.5345224838248487
    >>> b = np.array([2,8,3,4])
    >>> c = np.ma.masked_array(b, mask=[0,0,1,0])
    >>> variation(c)
    0.5345224838248487

    In the example above, it can be seen that this works the same as
    `scipy.stats.variation` except 'stats.mstats.variation' ignores masked
    array elements.

    """
    a, axis = _chk_asarray(a, axis)
    return a.std(axis, ddof=ddof)/a.mean(axis)


def skew(a, axis=0, bias=True):
    """
    Computes the skewness of a data set.

    Parameters
    ----------
    a : ndarray
        data
    axis : int or None, optional
        Axis along which skewness is calculated. Default is 0.
        If None, compute over the whole array `a`.
    bias : bool, optional
        If False, then the calculations are corrected for statistical bias.

    Returns
    -------
    skewness : ndarray
        The skewness of values along an axis, returning 0 where all values are
        equal.

    Notes
    -----
    For more details about `skew`, see `scipy.stats.skew`.

    """
    a, axis = _chk_asarray(a,axis)
    mean = a.mean(axis, keepdims=True)
    m2 = _moment(a, 2, axis, mean=mean)
    m3 = _moment(a, 3, axis, mean=mean)
    zero = (m2 <= (np.finfo(m2.dtype).resolution * mean.squeeze(axis))**2)
    with np.errstate(all='ignore'):
        vals = ma.where(zero, 0, m3 / m2**1.5)

    if not bias and zero is not ma.masked and m2 is not ma.masked:
        n = a.count(axis)
        can_correct = ~zero & (n > 2)
        if can_correct.any():
            n = np.extract(can_correct, n)
            m2 = np.extract(can_correct, m2)
            m3 = np.extract(can_correct, m3)
            nval = ma.sqrt((n-1.0)*n)/(n-2.0)*m3/m2**1.5
            np.place(vals, can_correct, nval)
    return vals


def kurtosis(a, axis=0, fisher=True, bias=True):
    """
    Computes the kurtosis (Fisher or Pearson) of a dataset.

    Kurtosis is the fourth central moment divided by the square of the
    variance. If Fisher's definition is used, then 3.0 is subtracted from
    the result to give 0.0 for a normal distribution.

    If bias is False then the kurtosis is calculated using k statistics to
    eliminate bias coming from biased moment estimators

    Use `kurtosistest` to see if result is close enough to normal.

    Parameters
    ----------
    a : array
        data for which the kurtosis is calculated
    axis : int or None, optional
        Axis along which the kurtosis is calculated. Default is 0.
        If None, compute over the whole array `a`.
    fisher : bool, optional
        If True, Fisher's definition is used (normal ==> 0.0). If False,
        Pearson's definition is used (normal ==> 3.0).
    bias : bool, optional
        If False, then the calculations are corrected for statistical bias.

    Returns
    -------
    kurtosis : array
        The kurtosis of values along an axis. If all values are equal,
        return -3 for Fisher's definition and 0 for Pearson's definition.

    Notes
    -----
    For more details about `kurtosis`, see `scipy.stats.kurtosis`.

    """
    a, axis = _chk_asarray(a, axis)
    mean = a.mean(axis, keepdims=True)
    m2 = _moment(a, 2, axis, mean=mean)
    m4 = _moment(a, 4, axis, mean=mean)
    zero = (m2 <= (np.finfo(m2.dtype).resolution * mean.squeeze(axis))**2)
    with np.errstate(all='ignore'):
        vals = ma.where(zero, 0, m4 / m2**2.0)

    if not bias and zero is not ma.masked and m2 is not ma.masked:
        n = a.count(axis)
        can_correct = ~zero & (n > 3)
        if can_correct.any():
            n = np.extract(can_correct, n)
            m2 = np.extract(can_correct, m2)
            m4 = np.extract(can_correct, m4)
            nval = 1.0/(n-2)/(n-3)*((n*n-1.0)*m4/m2**2.0-3*(n-1)**2.0)
            np.place(vals, can_correct, nval+3.0)
    if fisher:
        return vals - 3
    else:
        return vals


DescribeResult = namedtuple('DescribeResult', ('nobs', 'minmax', 'mean',
                                               'variance', 'skewness',
                                               'kurtosis'))


def describe(a, axis=0, ddof=0, bias=True):
    """
    Computes several descriptive statistics of the passed array.

    Parameters
    ----------
    a : array_like
        Data array
    axis : int or None, optional
        Axis along which to calculate statistics. Default 0. If None,
        compute over the whole array `a`.
    ddof : int, optional
        degree of freedom (default 0); note that default ddof is different
        from the same routine in stats.describe
    bias : bool, optional
        If False, then the skewness and kurtosis calculations are corrected for
        statistical bias.

    Returns
    -------
    nobs : int
        (size of the data (discarding missing values)

    minmax : (int, int)
        min, max

    mean : float
        arithmetic mean

    variance : float
        unbiased variance

    skewness : float
        biased skewness

    kurtosis : float
        biased kurtosis

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats.mstats import describe
    >>> ma = np.ma.array(range(6), mask=[0, 0, 0, 1, 1, 1])
    >>> describe(ma)
    DescribeResult(nobs=np.int64(3), minmax=(masked_array(data=0,
                 mask=False,
           fill_value=999999), masked_array(data=2,
                 mask=False,
           fill_value=999999)), mean=np.float64(1.0),
           variance=np.float64(0.6666666666666666),
           skewness=masked_array(data=0., mask=False, fill_value=1e+20),
            kurtosis=np.float64(-1.5))

    """
    a, axis = _chk_asarray(a, axis)
    n = a.count(axis)
    mm = (ma.minimum.reduce(a, axis=axis), ma.maximum.reduce(a, axis=axis))
    m = a.mean(axis)
    v = a.var(axis, ddof=ddof)
    sk = skew(a, axis, bias=bias)
    kurt = kurtosis(a, axis, bias=bias)

    return DescribeResult(n, mm, m, v, sk, kurt)


def stde_median(data, axis=None):
    """Returns the McKean-Schrader estimate of the standard error of the sample
    median along the given axis. masked values are discarded.

    Parameters
    ----------
    data : ndarray
        Data to trim.
    axis : {None,int}, optional
        Axis along which to perform the trimming.
        If None, the input array is first flattened.

    """
    def _stdemed_1D(data):
        data = np.sort(data.compressed())
        n = len(data)
        z = 2.5758293035489004
        k = int(np.round((n+1)/2. - z * np.sqrt(n/4.),0))
        return ((data[n-k] - data[k-1])/(2.*z))

    data = ma.array(data, copy=False, subok=True)
    if (axis is None):
        return _stdemed_1D(data)
    else:
        if data.ndim > 2:
            raise ValueError("Array 'data' must be at most two dimensional, "
                             "but got data.ndim = %d" % data.ndim)
        return ma.apply_along_axis(_stdemed_1D, axis, data)


SkewtestResult = namedtuple('SkewtestResult', ('statistic', 'pvalue'))


def skewtest(a, axis=0, alternative='two-sided'):
    """
    Tests whether the skew is different from the normal distribution.

    Parameters
    ----------
    a : array_like
        The data to be tested
    axis : int or None, optional
       Axis along which statistics are calculated. Default is 0.
       If None, compute over the whole array `a`.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        The following options are available:

        * 'two-sided': the skewness of the distribution underlying the sample
          is different from that of the normal distribution (i.e. 0)
        * 'less': the skewness of the distribution underlying the sample
          is less than that of the normal distribution
        * 'greater': the skewness of the distribution underlying the sample
          is greater than that of the normal distribution

        .. versionadded:: 1.7.0

    Returns
    -------
    statistic : array_like
        The computed z-score for this test.
    pvalue : array_like
        A p-value for the hypothesis test

    Notes
    -----
    For more details about `skewtest`, see `scipy.stats.skewtest`.

    """
    a, axis = _chk_asarray(a, axis)
    if axis is None:
        a = a.ravel()
        axis = 0
    b2 = skew(a,axis)
    n = a.count(axis)
    if np.min(n) < 8:
        raise ValueError(
            "skewtest is not valid with less than 8 samples; %i samples"
            " were given." % np.min(n))

    y = b2 * ma.sqrt(((n+1)*(n+3)) / (6.0*(n-2)))
    beta2 = (3.0*(n*n+27*n-70)*(n+1)*(n+3)) / ((n-2.0)*(n+5)*(n+7)*(n+9))
    W2 = -1 + ma.sqrt(2*(beta2-1))
    delta = 1/ma.sqrt(0.5*ma.log(W2))
    alpha = ma.sqrt(2.0/(W2-1))
    y = ma.where(y == 0, 1, y)
    Z = delta*ma.log(y/alpha + ma.sqrt((y/alpha)**2+1))
    pvalue = scipy.stats._stats_py._get_pvalue(Z, distributions.norm, alternative)

    return SkewtestResult(Z[()], pvalue[()])


KurtosistestResult = namedtuple('KurtosistestResult', ('statistic', 'pvalue'))


def kurtosistest(a, axis=0, alternative='two-sided'):
    """
    Tests whether a dataset has normal kurtosis

    Parameters
    ----------
    a : array_like
        array of the sample data
    axis : int or None, optional
       Axis along which to compute test. Default is 0. If None,
       compute over the whole array `a`.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

        * 'two-sided': the kurtosis of the distribution underlying the sample
          is different from that of the normal distribution
        * 'less': the kurtosis of the distribution underlying the sample
          is less than that of the normal distribution
        * 'greater': the kurtosis of the distribution underlying the sample
          is greater than that of the normal distribution

        .. versionadded:: 1.7.0

    Returns
    -------
    statistic : array_like
        The computed z-score for this test.
    pvalue : array_like
        The p-value for the hypothesis test

    Notes
    -----
    For more details about `kurtosistest`, see `scipy.stats.kurtosistest`.

    """
    a, axis = _chk_asarray(a, axis)
    n = a.count(axis=axis)
    if np.min(n) < 5:
        raise ValueError(
            "kurtosistest requires at least 5 observations; %i observations"
            " were given." % np.min(n))
    if np.min(n) < 20:
        warnings.warn(
            "kurtosistest only valid for n>=20 ... continuing anyway, n=%i" % np.min(n),
            stacklevel=2,
        )

    b2 = kurtosis(a, axis, fisher=False)
    E = 3.0*(n-1) / (n+1)
    varb2 = 24.0*n*(n-2.)*(n-3) / ((n+1)*(n+1.)*(n+3)*(n+5))
    x = (b2-E)/ma.sqrt(varb2)
    sqrtbeta1 = 6.0*(n*n-5*n+2)/((n+7)*(n+9)) * np.sqrt((6.0*(n+3)*(n+5)) /
                                                        (n*(n-2)*(n-3)))
    A = 6.0 + 8.0/sqrtbeta1 * (2.0/sqrtbeta1 + np.sqrt(1+4.0/(sqrtbeta1**2)))
    term1 = 1 - 2./(9.0*A)
    denom = 1 + x*ma.sqrt(2/(A-4.0))
    if np.ma.isMaskedArray(denom):
        # For multi-dimensional array input
        denom[denom == 0.0] = masked
    elif denom == 0.0:
        denom = masked

    term2 = np.ma.where(denom > 0, ma.power((1-2.0/A)/denom, 1/3.0),
                        -ma.power(-(1-2.0/A)/denom, 1/3.0))
    Z = (term1 - term2) / np.sqrt(2/(9.0*A))
    pvalue = scipy.stats._stats_py._get_pvalue(Z, distributions.norm, alternative)

    return KurtosistestResult(Z[()], pvalue[()])


NormaltestResult = namedtuple('NormaltestResult', ('statistic', 'pvalue'))


def normaltest(a, axis=0):
    """
    Tests whether a sample differs from a normal distribution.

    Parameters
    ----------
    a : array_like
        The array containing the data to be tested.
    axis : int or None, optional
        Axis along which to compute test. Default is 0. If None,
        compute over the whole array `a`.

    Returns
    -------
    statistic : float or array
        ``s^2 + k^2``, where ``s`` is the z-score returned by `skewtest` and
        ``k`` is the z-score returned by `kurtosistest`.
    pvalue : float or array
       A 2-sided chi squared probability for the hypothesis test.

    Notes
    -----
    For more details about `normaltest`, see `scipy.stats.normaltest`.

    """
    a, axis = _chk_asarray(a, axis)
    s, _ = skewtest(a, axis)
    k, _ = kurtosistest(a, axis)
    k2 = s*s + k*k

    return NormaltestResult(k2, distributions.chi2.sf(k2, 2))


def mquantiles(a, prob=(.25, .5, .75), alphap=.4, betap=.4, axis=None,
               limit=()):
    """
    Computes empirical quantiles for a data array.

    Samples quantile are defined by ``Q(p) = (1-gamma)*x[j] + gamma*x[j+1]``,
    where ``x[j]`` is the j-th order statistic, and gamma is a function of
    ``j = floor(n*p + m)``, ``m = alphap + p*(1 - alphap - betap)`` and
    ``g = n*p + m - j``.

    Reinterpreting the above equations to compare to **R** lead to the
    equation: ``p(k) = (k - alphap)/(n + 1 - alphap - betap)``

    Typical values of (alphap,betap) are:
        - (0,1)    : ``p(k) = k/n`` : linear interpolation of cdf
          (**R** type 4)
        - (.5,.5)  : ``p(k) = (k - 1/2.)/n`` : piecewise linear function
          (**R** type 5)
        - (0,0)    : ``p(k) = k/(n+1)`` :
          (**R** type 6)
        - (1,1)    : ``p(k) = (k-1)/(n-1)``: p(k) = mode[F(x[k])].
          (**R** type 7, **R** default)
        - (1/3,1/3): ``p(k) = (k-1/3)/(n+1/3)``: Then p(k) ~ median[F(x[k])].
          The resulting quantile estimates are approximately median-unbiased
          regardless of the distribution of x.
          (**R** type 8)
        - (3/8,3/8): ``p(k) = (k-3/8)/(n+1/4)``: Blom.
          The resulting quantile estimates are approximately unbiased
          if x is normally distributed
          (**R** type 9)
        - (.4,.4)  : approximately quantile unbiased (Cunnane)
        - (.35,.35): APL, used with PWM

    Parameters
    ----------
    a : array_like
        Input data, as a sequence or array of dimension at most 2.
    prob : array_like, optional
        List of quantiles to compute.
    alphap : float, optional
        Plotting positions parameter, default is 0.4.
    betap : float, optional
        Plotting positions parameter, default is 0.4.
    axis : int, optional
        Axis along which to perform the trimming.
        If None (default), the input array is first flattened.
    limit : tuple, optional
        Tuple of (lower, upper) values.
        Values of `a` outside this open interval are ignored.

    Returns
    -------
    mquantiles : MaskedArray
        An array containing the calculated quantiles.

    Notes
    -----
    This formulation is very similar to **R** except the calculation of
    ``m`` from ``alphap`` and ``betap``, where in **R** ``m`` is defined
    with each type.

    References
    ----------
    .. [1] *R* statistical software: https://www.r-project.org/
    .. [2] *R* ``quantile`` function:
            http://stat.ethz.ch/R-manual/R-devel/library/stats/html/quantile.html

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats.mstats import mquantiles
    >>> a = np.array([6., 47., 49., 15., 42., 41., 7., 39., 43., 40., 36.])
    >>> mquantiles(a)
    array([ 19.2,  40. ,  42.8])

    Using a 2D array, specifying axis and limit.

    >>> data = np.array([[   6.,    7.,    1.],
    ...                  [  47.,   15.,    2.],
    ...                  [  49.,   36.,    3.],
    ...                  [  15.,   39.,    4.],
    ...                  [  42.,   40., -999.],
    ...                  [  41.,   41., -999.],
    ...                  [   7., -999., -999.],
    ...                  [  39., -999., -999.],
    ...                  [  43., -999., -999.],
    ...                  [  40., -999., -999.],
    ...                  [  36., -999., -999.]])
    >>> print(mquantiles(data, axis=0, limit=(0, 50)))
    [[19.2  14.6   1.45]
     [40.   37.5   2.5 ]
     [42.8  40.05  3.55]]

    >>> data[:, 2] = -999.
    >>> print(mquantiles(data, axis=0, limit=(0, 50)))
    [[19.200000000000003 14.6 --]
     [40.0 37.5 --]
     [42.800000000000004 40.05 --]]

    """
    def _quantiles1D(data,m,p):
        x = np.sort(data.compressed())
        n = len(x)
        if n == 0:
            return ma.array(np.empty(len(p), dtype=float), mask=True)
        elif n == 1:
            return ma.array(np.resize(x, p.shape), mask=nomask)
        aleph = (n*p + m)
        k = np.floor(aleph.clip(1, n-1)).astype(int)
        gamma = (aleph-k).clip(0,1)
        return (1.-gamma)*x[(k-1).tolist()] + gamma*x[k.tolist()]

    data = ma.array(a, copy=False)
    if data.ndim > 2:
        raise TypeError("Array should be 2D at most !")

    if limit:
        condition = (limit[0] < data) & (data < limit[1])
        data[~condition.filled(True)] = masked

    p = np.atleast_1d(np.asarray(prob))
    m = alphap + p*(1.-alphap-betap)
    # Computes quantiles along axis (or globally)
    if (axis is None):
        return _quantiles1D(data, m, p)

    return ma.apply_along_axis(_quantiles1D, axis, data, m, p)


def scoreatpercentile(data, per, limit=(), alphap=.4, betap=.4):
    """Calculate the score at the given 'per' percentile of the
    sequence a.  For example, the score at per=50 is the median.

    This function is a shortcut to mquantile

    """
    if (per < 0) or (per > 100.):
        raise ValueError(f"The percentile should be between 0. and 100. ! (got {per})")

    return mquantiles(data, prob=[per/100.], alphap=alphap, betap=betap,
                      limit=limit, axis=0).squeeze()


def plotting_positions(data, alpha=0.4, beta=0.4):
    """
    Returns plotting positions (or empirical percentile points) for the data.

    Plotting positions are defined as ``(i-alpha)/(n+1-alpha-beta)``, where:
        - i is the rank order statistics
        - n is the number of unmasked values along the given axis
        - `alpha` and `beta` are two parameters.

    Typical values for `alpha` and `beta` are:
        - (0,1)    : ``p(k) = k/n``, linear interpolation of cdf (R, type 4)
        - (.5,.5)  : ``p(k) = (k-1/2.)/n``, piecewise linear function
          (R, type 5)
        - (0,0)    : ``p(k) = k/(n+1)``, Weibull (R type 6)
        - (1,1)    : ``p(k) = (k-1)/(n-1)``, in this case,
          ``p(k) = mode[F(x[k])]``. That's R default (R type 7)
        - (1/3,1/3): ``p(k) = (k-1/3)/(n+1/3)``, then
          ``p(k) ~ median[F(x[k])]``.
          The resulting quantile estimates are approximately median-unbiased
          regardless of the distribution of x. (R type 8)
        - (3/8,3/8): ``p(k) = (k-3/8)/(n+1/4)``, Blom.
          The resulting quantile estimates are approximately unbiased
          if x is normally distributed (R type 9)
        - (.4,.4)  : approximately quantile unbiased (Cunnane)
        - (.35,.35): APL, used with PWM
        - (.3175, .3175): used in scipy.stats.probplot

    Parameters
    ----------
    data : array_like
        Input data, as a sequence or array of dimension at most 2.
    alpha : float, optional
        Plotting positions parameter. Default is 0.4.
    beta : float, optional
        Plotting positions parameter. Default is 0.4.

    Returns
    -------
    positions : MaskedArray
        The calculated plotting positions.

    """
    data = ma.array(data, copy=False).reshape(1,-1)
    n = data.count()
    plpos = np.empty(data.size, dtype=float)
    plpos[n:] = 0
    plpos[data.argsort(axis=None)[:n]] = ((np.arange(1, n+1) - alpha) /
                                          (n + 1.0 - alpha - beta))
    return ma.array(plpos, mask=data._mask)


meppf = plotting_positions


def obrientransform(*args):
    """
    Computes a transform on input data (any number of columns).  Used to
    test for homogeneity of variance prior to running one-way stats.  Each
    array in ``*args`` is one level of a factor.  If an `f_oneway()` run on
    the transformed data and found significant, variances are unequal.   From
    Maxwell and Delaney, p.112.

    Returns: transformed data for use in an ANOVA
    """
    data = argstoarray(*args).T
    v = data.var(axis=0,ddof=1)
    m = data.mean(0)
    n = data.count(0).astype(float)
    # result = ((N-1.5)*N*(a-m)**2 - 0.5*v*(n-1))/((n-1)*(n-2))
    data -= m
    data **= 2
    data *= (n-1.5)*n
    data -= 0.5*v*(n-1)
    data /= (n-1.)*(n-2.)
    if not ma.allclose(v,data.mean(0)):
        raise ValueError("Lack of convergence in obrientransform.")

    return data


def sem(a, axis=0, ddof=1):
    """
    Calculates the standard error of the mean of the input array.

    Also sometimes called standard error of measurement.

    Parameters
    ----------
    a : array_like
        An array containing the values for which the standard error is
        returned.
    axis : int or None, optional
        If axis is None, ravel `a` first. If axis is an integer, this will be
        the axis over which to operate. Defaults to 0.
    ddof : int, optional
        Delta degrees-of-freedom. How many degrees of freedom to adjust
        for bias in limited samples relative to the population estimate
        of variance. Defaults to 1.

    Returns
    -------
    s : ndarray or float
        The standard error of the mean in the sample(s), along the input axis.

    Notes
    -----
    The default value for `ddof` changed in scipy 0.15.0 to be consistent with
    `scipy.stats.sem` as well as with the most common definition used (like in
    the R documentation).

    Examples
    --------
    Find standard error along the first axis:

    >>> import numpy as np
    >>> from scipy import stats
    >>> a = np.arange(20).reshape(5,4)
    >>> print(stats.mstats.sem(a))
    [2.8284271247461903 2.8284271247461903 2.8284271247461903
     2.8284271247461903]

    Find standard error across the whole array, using n degrees of freedom:

    >>> print(stats.mstats.sem(a, axis=None, ddof=0))
    1.2893796958227628

    """
    a, axis = _chk_asarray(a, axis)
    n = a.count(axis=axis)
    s = a.std(axis=axis, ddof=ddof) / ma.sqrt(n)
    return s


F_onewayResult = namedtuple('F_onewayResult', ('statistic', 'pvalue'))


def f_oneway(*args):
    """
    Performs a 1-way ANOVA, returning an F-value and probability given
    any number of groups.  From Heiman, pp.394-7.

    Usage: ``f_oneway(*args)``, where ``*args`` is 2 or more arrays,
    one per treatment group.

    Returns
    -------
    statistic : float
        The computed F-value of the test.
    pvalue : float
        The associated p-value from the F-distribution.

    """
    # Construct a single array of arguments: each row is a group
    data = argstoarray(*args)
    ngroups = len(data)
    ntot = data.count()
    sstot = (data**2).sum() - (data.sum())**2/float(ntot)
    ssbg = (data.count(-1) * (data.mean(-1)-data.mean())**2).sum()
    sswg = sstot-ssbg
    dfbg = ngroups-1
    dfwg = ntot - ngroups
    msb = ssbg/float(dfbg)
    msw = sswg/float(dfwg)
    f = msb/msw
    prob = special.fdtrc(dfbg, dfwg, f)  # equivalent to stats.f.sf

    return F_onewayResult(f, prob)


FriedmanchisquareResult = namedtuple('FriedmanchisquareResult',
                                     ('statistic', 'pvalue'))


def friedmanchisquare(*args):
    """Friedman Chi-Square is a non-parametric, one-way within-subjects ANOVA.
    This function calculates the Friedman Chi-square test for repeated measures
    and returns the result, along with the associated probability value.

    Each input is considered a given group. Ideally, the number of treatments
    among each group should be equal. If this is not the case, only the first
    n treatments are taken into account, where n is the number of treatments
    of the smallest group.
    If a group has some missing values, the corresponding treatments are masked
    in the other groups.
    The test statistic is corrected for ties.

    Masked values in one group are propagated to the other groups.

    Returns
    -------
    statistic : float
        the test statistic.
    pvalue : float
        the associated p-value.

    """
    data = argstoarray(*args).astype(float)
    k = len(data)
    if k < 3:
        raise ValueError("Less than 3 groups (%i): " % k +
                         "the Friedman test is NOT appropriate.")

    ranked = ma.masked_values(rankdata(data, axis=0), 0)
    if ranked._mask is not nomask:
        ranked = ma.mask_cols(ranked)
        ranked = ranked.compressed().reshape(k,-1).view(ndarray)
    else:
        ranked = ranked._data
    (k,n) = ranked.shape
    # Ties correction
    repeats = [find_repeats(row) for row in ranked.T]
    ties = np.array([y for x, y in repeats if x.size > 0])
    tie_correction = 1 - (ties**3-ties).sum()/float(n*(k**3-k))

    ssbg = np.sum((ranked.sum(-1) - n*(k+1)/2.)**2)
    chisq = ssbg * 12./(n*k*(k+1)) * 1./tie_correction

    return FriedmanchisquareResult(chisq,
                                   distributions.chi2.sf(chisq, k-1))


BrunnerMunzelResult = namedtuple('BrunnerMunzelResult', ('statistic', 'pvalue'))


def brunnermunzel(x, y, alternative="two-sided", distribution="t"):
    """
    Compute the Brunner-Munzel test on samples x and y.

    Any missing values in `x` and/or `y` are discarded.

    The Brunner-Munzel test is a nonparametric test of the null hypothesis that
    when values are taken one by one from each group, the probabilities of
    getting large values in both groups are equal.
    Unlike the Wilcoxon-Mann-Whitney's U test, this does not require the
    assumption of equivariance of two groups. Note that this does not assume
    the distributions are same. This test works on two independent samples,
    which may have different sizes.

    Parameters
    ----------
    x, y : array_like
        Array of samples, should be one-dimensional.
    alternative : 'less', 'two-sided', or 'greater', optional
        Whether to get the p-value for the one-sided hypothesis ('less'
        or 'greater') or for the two-sided hypothesis ('two-sided').
        Defaults value is 'two-sided' .
    distribution : 't' or 'normal', optional
        Whether to get the p-value by t-distribution or by standard normal
        distribution.
        Defaults value is 't' .

    Returns
    -------
    statistic : float
        The Brunner-Munzer W statistic.
    pvalue : float
        p-value assuming an t distribution. One-sided or
        two-sided, depending on the choice of `alternative` and `distribution`.

    See Also
    --------
    mannwhitneyu : Mann-Whitney rank test on two samples.

    Notes
    -----
    For more details on `brunnermunzel`, see `scipy.stats.brunnermunzel`.

    Examples
    --------
    >>> from scipy.stats.mstats import brunnermunzel
    >>> import numpy as np
    >>> x1 = [1, 2, np.nan, np.nan, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1]
    >>> x2 = [3, 3, 4, 3, 1, 2, 3, 1, 1, 5, 4]
    >>> brunnermunzel(x1, x2)
    BrunnerMunzelResult(statistic=1.4723186918922935, pvalue=0.15479415300426624)  # may vary

    """  # noqa: E501
    x = ma.asarray(x).compressed().view(ndarray)
    y = ma.asarray(y).compressed().view(ndarray)
    nx = len(x)
    ny = len(y)
    if nx == 0 or ny == 0:
        return BrunnerMunzelResult(np.nan, np.nan)
    rankc = rankdata(np.concatenate((x,y)))
    rankcx = rankc[0:nx]
    rankcy = rankc[nx:nx+ny]
    rankcx_mean = np.mean(rankcx)
    rankcy_mean = np.mean(rankcy)
    rankx = rankdata(x)
    ranky = rankdata(y)
    rankx_mean = np.mean(rankx)
    ranky_mean = np.mean(ranky)

    Sx = np.sum(np.power(rankcx - rankx - rankcx_mean + rankx_mean, 2.0))
    Sx /= nx - 1
    Sy = np.sum(np.power(rankcy - ranky - rankcy_mean + ranky_mean, 2.0))
    Sy /= ny - 1

    wbfn = nx * ny * (rankcy_mean - rankcx_mean)
    wbfn /= (nx + ny) * np.sqrt(nx * Sx + ny * Sy)

    if distribution == "t":
        df_numer = np.power(nx * Sx + ny * Sy, 2.0)
        df_denom = np.power(nx * Sx, 2.0) / (nx - 1)
        df_denom += np.power(ny * Sy, 2.0) / (ny - 1)
        df = df_numer / df_denom
        p = distributions.t.cdf(wbfn, df)
    elif distribution == "normal":
        p = distributions.norm.cdf(wbfn)
    else:
        raise ValueError(
            "distribution should be 't' or 'normal'")

    if alternative == "greater":
        pass
    elif alternative == "less":
        p = 1 - p
    elif alternative == "two-sided":
        p = 2 * np.min([p, 1-p])
    else:
        raise ValueError(
            "alternative should be 'less', 'greater' or 'two-sided'")

    return BrunnerMunzelResult(wbfn, p)
<|endoftext|>"""
Additional statistics functions with support for masked arrays.

"""

# Original author (2007): Pierre GF Gerard-Marchant


__all__ = ['compare_medians_ms',
           'hdquantiles', 'hdmedian', 'hdquantiles_sd',
           'idealfourths',
           'median_cihs','mjci','mquantiles_cimj',
           'rsh',
           'trimmed_mean_ci',]


import numpy as np
from numpy import float64, ndarray

import numpy.ma as ma
from numpy.ma import MaskedArray

from . import _mstats_basic as mstats

from scipy.stats.distributions import norm, beta, t, binom


def hdquantiles(data, prob=(.25, .5, .75), axis=None, var=False,):
    """
    Computes quantile estimates with the Harrell-Davis method.

    The quantile estimates are calculated as a weighted linear combination
    of order statistics.

    Parameters
    ----------
    data : array_like
        Data array.
    prob : sequence, optional
        Sequence of probabilities at which to compute the quantiles.
    axis : int or None, optional
        Axis along which to compute the quantiles. If None, use a flattened
        array.
    var : bool, optional
        Whether to return the variance of the estimate.

    Returns
    -------
    hdquantiles : MaskedArray
        A (p,) array of quantiles (if `var` is False), or a (2,p) array of
        quantiles and variances (if `var` is True), where ``p`` is the
        number of quantiles.

    See Also
    --------
    hdquantiles_sd

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats.mstats import hdquantiles
    >>>
    >>> # Sample data
    >>> data = np.array([1.2, 2.5, 3.7, 4.0, 5.1, 6.3, 7.0, 8.2, 9.4])
    >>>
    >>> # Probabilities at which to compute quantiles
    >>> probabilities = [0.25, 0.5, 0.75]
    >>>
    >>> # Compute Harrell-Davis quantile estimates
    >>> quantile_estimates = hdquantiles(data, prob=probabilities)
    >>>
    >>> # Display the quantile estimates
    >>> for i, quantile in enumerate(probabilities):
    ...     print(f"{int(quantile * 100)}th percentile: {quantile_estimates[i]}")
    25th percentile: 3.1505820231763066 # may vary
    50th percentile: 5.194344084883956
    75th percentile: 7.430626414674935

    """
    def _hd_1D(data,prob,var):
        "Computes the HD quantiles for a 1D array. Returns nan for invalid data."
        xsorted = np.squeeze(np.sort(data.compressed().view(ndarray)))
        # Don't use length here, in case we have a numpy scalar
        n = xsorted.size

        hd = np.empty((2,len(prob)), float64)
        if n < 2:
            hd.flat = np.nan
            if var:
                return hd
            return hd[0]

        v = np.arange(n+1) / float(n)
        betacdf = beta.cdf
        for (i,p) in enumerate(prob):
            _w = betacdf(v, (n+1)*p, (n+1)*(1-p))
            w = _w[1:] - _w[:-1]
            hd_mean = np.dot(w, xsorted)
            hd[0,i] = hd_mean
            #
            hd[1,i] = np.dot(w, (xsorted-hd_mean)**2)
            #
        hd[0, prob == 0] = xsorted[0]
        hd[0, prob == 1] = xsorted[-1]
        if var:
            hd[1, prob == 0] = hd[1, prob == 1] = np.nan
            return hd
        return hd[0]
    # Initialization & checks
    data = ma.array(data, copy=False, dtype=float64)
    p = np.atleast_1d(np.asarray(prob))
    # Computes quantiles along axis (or globally)
    if (axis is None) or (data.ndim == 1):
        result = _hd_1D(data, p, var)
    else:
        if data.ndim > 2:
            raise ValueError("Array 'data' must be at most two dimensional, "
                             "but got data.ndim = %d" % data.ndim)
        result = ma.apply_along_axis(_hd_1D, axis, data, p, var)

    return ma.fix_invalid(result, copy=False)


def hdmedian(data, axis=-1, var=False):
    """
    Returns the Harrell-Davis estimate of the median along the given axis.

    Parameters
    ----------
    data : ndarray
        Data array.
    axis : int, optional
        Axis along which to compute the quantiles. If None, use a flattened
        array.
    var : bool, optional
        Whether to return the variance of the estimate.

    Returns
    -------
    hdmedian : MaskedArray
        The median values.  If ``var=True``, the variance is returned inside
        the masked array.  E.g. for a 1-D array the shape change from (1,) to
        (2,).

    """
    result = hdquantiles(data,[0.5], axis=axis, var=var)
    return result.squeeze()


def hdquantiles_sd(data, prob=(.25, .5, .75), axis=None):
    """
    The standard error of the Harrell-Davis quantile estimates by jackknife.

    Parameters
    ----------
    data : array_like
        Data array.
    prob : sequence, optional
        Sequence of quantiles to compute.
    axis : int, optional
        Axis along which to compute the quantiles. If None, use a flattened
        array.

    Returns
    -------
    hdquantiles_sd : MaskedArray
        Standard error of the Harrell-Davis quantile estimates.

    See Also
    --------
    hdquantiles

    """
    def _hdsd_1D(data, prob):
        "Computes the std error for 1D arrays."
        xsorted = np.sort(data.compressed())
        n = len(xsorted)

        hdsd = np.empty(len(prob), float64)
        if n < 2:
            hdsd.flat = np.nan

        vv = np.arange(n) / float(n-1)
        betacdf = beta.cdf

        for (i,p) in enumerate(prob):
            _w = betacdf(vv, n*p, n*(1-p))
            w = _w[1:] - _w[:-1]
            # cumulative sum of weights and data points if
            # ith point is left out for jackknife
            mx_ = np.zeros_like(xsorted)
            mx_[1:] = np.cumsum(w * xsorted[:-1])
            # similar but from the right
            mx_[:-1] += np.cumsum(w[::-1] * xsorted[:0:-1])[::-1]
            hdsd[i] = np.sqrt(mx_.var() * (n - 1))
        return hdsd

    # Initialization & checks
    data = ma.array(data, copy=False, dtype=float64)
    p = np.atleast_1d(np.asarray(prob))
    # Computes quantiles along axis (or globally)
    if (axis is None):
        result = _hdsd_1D(data, p)
    else:
        if data.ndim > 2:
            raise ValueError("Array 'data' must be at most two dimensional, "
                             "but got data.ndim = %d" % data.ndim)
        result = ma.apply_along_axis(_hdsd_1D, axis, data, p)

    return ma.fix_invalid(result, copy=False).ravel()


def trimmed_mean_ci(data, limits=(0.2,0.2), inclusive=(True,True),
                    alpha=0.05, axis=None):
    """
    Selected confidence interval of the trimmed mean along the given axis.

    Parameters
    ----------
    data : array_like
        Input data.
    limits : {None, tuple}, optional
        None or a two item tuple.
        Tuple of the percentages to cut on each side of the array, with respect
        to the number of unmasked data, as floats between 0. and 1. If ``n``
        is the number of unmasked data before trimming, then
        (``n * limits[0]``)th smallest data and (``n * limits[1]``)th
        largest data are masked.  The total number of unmasked data after
        trimming is ``n * (1. - sum(limits))``.
        The value of one limit can be set to None to indicate an open interval.

        Defaults to (0.2, 0.2).
    inclusive : (2,) tuple of boolean, optional
        If relative==False, tuple indicating whether values exactly equal to
        the absolute limits are allowed.
        If relative==True, tuple indicating whether the number of data being
        masked on each side should be rounded (True) or truncated (False).

        Defaults to (True, True).
    alpha : float, optional
        Confidence level of the intervals.

        Defaults to 0.05.
    axis : int, optional
        Axis along which to cut. If None, uses a flattened version of `data`.

        Defaults to None.

    Returns
    -------
    trimmed_mean_ci : (2,) ndarray
        The lower and upper confidence intervals of the trimmed data.

    """
    data = ma.array(data, copy=False)
    trimmed = mstats.trimr(data, limits=limits, inclusive=inclusive, axis=axis)
    tmean = trimmed.mean(axis)
    tstde = mstats.trimmed_stde(data,limits=limits,inclusive=inclusive,axis=axis)
    df = trimmed.count(axis) - 1
    tppf = t.ppf(1-alpha/2.,df)
    return np.array((tmean - tppf*tstde, tmean+tppf*tstde))


def mjci(data, prob=(0.25, 0.5, 0.75), axis=None):
    """
    Returns the Maritz-Jarrett estimators of the standard error of selected
    experimental quantiles of the data.

    Parameters
    ----------
    data : ndarray
        Data array.
    prob : sequence, optional
        Sequence of quantiles to compute.
    axis : int or None, optional
        Axis along which to compute the quantiles. If None, use a flattened
        array.

    """
    def _mjci_1D(data, p):
        data = np.sort(data.compressed())
        n = data.size
        prob = (np.array(p) * n + 0.5).astype(int)
        betacdf = beta.cdf

        mj = np.empty(len(prob), float64)
        x = np.arange(1,n+1, dtype=float64) / n
        y = x - 1./n
        for (i,m) in enumerate(prob):
            W = betacdf(x,m-1,n-m) - betacdf(y,m-1,n-m)
            C1 = np.dot(W,data)
            C2 = np.dot(W,data**2)
            mj[i] = np.sqrt(C2 - C1**2)
        return mj

    data = ma.array(data, copy=False)
    if data.ndim > 2:
        raise ValueError("Array 'data' must be at most two dimensional, "
                         "but got data.ndim = %d" % data.ndim)

    p = np.atleast_1d(np.asarray(prob))
    # Computes quantiles along axis (or globally)
    if (axis is None):
        return _mjci_1D(data, p)
    else:
        return ma.apply_along_axis(_mjci_1D, axis, data, p)


def mquantiles_cimj(data, prob=(0.25, 0.50, 0.75), alpha=0.05, axis=None):
    """
    Computes the alpha confidence interval for the selected quantiles of the
    data, with Maritz-Jarrett estimators.

    Parameters
    ----------
    data : ndarray
        Data array.
    prob : sequence, optional
        Sequence of quantiles to compute.
    alpha : float, optional
        Confidence level of the intervals.
    axis : int or None, optional
        Axis along which to compute the quantiles.
        If None, use a flattened array.

    Returns
    -------
    ci_lower : ndarray
        The lower boundaries of the confidence interval.  Of the same length as
        `prob`.
    ci_upper : ndarray
        The upper boundaries of the confidence interval.  Of the same length as
        `prob`.

    """
    alpha = min(alpha, 1 - alpha)
    z = norm.ppf(1 - alpha/2.)
    xq = mstats.mquantiles(data, prob, alphap=0, betap=0, axis=axis)
    smj = mjci(data, prob, axis=axis)
    return (xq - z * smj, xq + z * smj)


def median_cihs(data, alpha=0.05, axis=None):
    """
    Computes the alpha-level confidence interval for the median of the data.

    Uses the Hettmasperger-Sheather method.

    Parameters
    ----------
    data : array_like
        Input data. Masked values are discarded. The input should be 1D only,
        or `axis` should be set to None.
    alpha : float, optional
        Confidence level of the intervals.
    axis : int or None, optional
        Axis along which to compute the quantiles. If None, use a flattened
        array.

    Returns
    -------
    median_cihs
        Alpha level confidence interval.

    """
    def _cihs_1D(data, alpha):
        data = np.sort(data.compressed())
        n = len(data)
        alpha = min(alpha, 1-alpha)
        k = int(binom._ppf(alpha/2., n, 0.5))
        gk = binom.cdf(n-k,n,0.5) - binom.cdf(k-1,n,0.5)
        if gk < 1-alpha:
            k -= 1
            gk = binom.cdf(n-k,n,0.5) - binom.cdf(k-1,n,0.5)
        gkk = binom.cdf(n-k-1,n,0.5) - binom.cdf(k,n,0.5)
        I = (gk - 1 + alpha)/(gk - gkk)
        lambd = (n-k) * I / float(k + (n-2*k)*I)
        lims = (lambd*data[k] + (1-lambd)*data[k-1],
                lambd*data[n-k-1] + (1-lambd)*data[n-k])
        return lims
    data = ma.array(data, copy=False)
    # Computes quantiles along axis (or globally)
    if (axis is None):
        result = _cihs_1D(data, alpha)
    else:
        if data.ndim > 2:
            raise ValueError("Array 'data' must be at most two dimensional, "
                             "but got data.ndim = %d" % data.ndim)
        result = ma.apply_along_axis(_cihs_1D, axis, data, alpha)

    return result


def compare_medians_ms(group_1, group_2, axis=None):
    """
    Compares the medians from two independent groups along the given axis.

    The comparison is performed using the McKean-Schrader estimate of the
    standard error of the medians.

    Parameters
    ----------
    group_1 : array_like
        First dataset.  Has to be of size >=7.
    group_2 : array_like
        Second dataset.  Has to be of size >=7.
    axis : int, optional
        Axis along which the medians are estimated. If None, the arrays are
        flattened.  If `axis` is not None, then `group_1` and `group_2`
        should have the same shape.

    Returns
    -------
    compare_medians_ms : {float, ndarray}
        If `axis` is None, then returns a float, otherwise returns a 1-D
        ndarray of floats with a length equal to the length of `group_1`
        along `axis`.

    Examples
    --------

    >>> from scipy import stats
    >>> a = [1, 2, 3, 4, 5, 6, 7]
    >>> b = [8, 9, 10, 11, 12, 13, 14]
    >>> stats.mstats.compare_medians_ms(a, b, axis=None)
    1.0693225866553746e-05

    The function is vectorized to compute along a given axis.

    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> x = rng.random(size=(3, 7))
    >>> y = rng.random(size=(3, 8))
    >>> stats.mstats.compare_medians_ms(x, y, axis=1)
    array([0.36908985, 0.36092538, 0.2765313 ])

    References
    ----------
    .. [1] McKean, Joseph W., and Ronald M. Schrader. "A comparison of methods
       for studentizing the sample median." Communications in
       Statistics-Simulation and Computation 13.6 (1984): 751-773.

    """
    (med_1, med_2) = (ma.median(group_1,axis=axis), ma.median(group_2,axis=axis))
    (std_1, std_2) = (mstats.stde_median(group_1, axis=axis),
                      mstats.stde_median(group_2, axis=axis))
    W = np.abs(med_1 - med_2) / ma.sqrt(std_1**2 + std_2**2)
    return 1 - norm.cdf(W)


def idealfourths(data, axis=None):
    """
    Returns an estimate of the lower and upper quartiles.

    Uses the ideal fourths algorithm.

    Parameters
    ----------
    data : array_like
        Input array.
    axis : int, optional
        Axis along which the quartiles are estimated. If None, the arrays are
        flattened.

    Returns
    -------
    idealfourths : {list of floats, masked array}
        Returns the two internal values that divide `data` into four parts
        using the ideal fourths algorithm either along the flattened array
        (if `axis` is None) or along `axis` of `data`.

    """
    def _idf(data):
        x = data.compressed()
        n = len(x)
        if n < 3:
            return [np.nan,np.nan]
        (j,h) = divmod(n/4. + 5/12.,1)
        j = int(j)
        qlo = (1-h)*x[j-1] + h*x[j]
        k = n - j
        qup = (1-h)*x[k] + h*x[k-1]
        return [qlo, qup]
    data = ma.sort(data, axis=axis).view(MaskedArray)
    if (axis is None):
        return _idf(data)
    else:
        return ma.apply_along_axis(_idf, axis, data)


def rsh(data, points=None):
    """
    Evaluates Rosenblatt's shifted histogram estimators for each data point.

    Rosenblatt's estimator is a centered finite-difference approximation to the
    derivative of the empirical cumulative distribution function.

    Parameters
    ----------
    data : sequence
        Input data, should be 1-D. Masked values are ignored.
    points : sequence or None, optional
        Sequence of points where to evaluate Rosenblatt shifted histogram.
        If None, use the data.

    """
    data = ma.array(data, copy=False)
    if points is None:
        points = data
    else:
        points = np.atleast_1d(np.asarray(points))

    if data.ndim != 1:
        raise AttributeError("The input array should be 1D only !")

    n = data.count()
    r = idealfourths(data, axis=None)
    h = 1.2 * (r[-1]-r[0]) / n**(1./5)
    nhi = (data[:,None] <= points[None,:] + h).sum(0)
    nlo = (data[:,None] < points[None,:] - h).sum(0)
    return (nhi-nlo) / (2.*n*h)
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.
# Use the `scipy.stats` namespace for importing the functions
# included below.

from scipy._lib.deprecation import _sub_module_deprecation


__all__ = [  # noqa: F822
    'mvsdist',
    'bayes_mvs', 'kstat', 'kstatvar', 'probplot', 'ppcc_max', 'ppcc_plot',
    'boxcox_llf', 'boxcox', 'boxcox_normmax', 'boxcox_normplot',
    'shapiro', 'anderson', 'ansari', 'bartlett', 'levene',
    'fligner', 'mood', 'wilcoxon', 'median_test',
    'circmean', 'circvar', 'circstd', 'anderson_ksamp',
    'yeojohnson_llf', 'yeojohnson', 'yeojohnson_normmax',
    'yeojohnson_normplot', 'find_repeats', 'chi2_contingency', 'distributions',
]


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="stats", module="morestats",
                                   private_modules=["_morestats"], all=__all__,
                                   attribute=name)
<|endoftext|>import numpy as np
from scipy import stats
from scipy.stats._stats_py import _SimpleNormal, SignificanceResult, _get_pvalue
from scipy.stats._axis_nan_policy import _axis_nan_policy_factory


__all__ = ['chatterjeexi']


# TODO:
# - Adjust to respect dtype


def _xi_statistic(x, y, y_continuous):
    # Compute xi correlation statistic

    # `axis=-1` is guaranteed by _axis_nan_policy decorator
    n = x.shape[-1]

    # "Rearrange the data as (X(1), Y(1)), . . . ,(X(n), Y(n))
    # such that X(1) ≤ ··· ≤ X(n)"
    j = np.argsort(x, axis=-1)
    j, y = np.broadcast_arrays(j, y)
    y = np.take_along_axis(y, j, axis=-1)

    # "Let ri be the rank of Y(i), that is, the number of j such that Y(j) ≤ Y(i)"
    r = stats.rankdata(y, method='max', axis=-1)
    # " additionally define li to be the number of j such that Y(j) ≥ Y(i)"
    # Could probably compute this from r, but that can be an enhancement
    l = stats.rankdata(-y, method='max', axis=-1)

    num = np.sum(np.abs(np.diff(r, axis=-1)), axis=-1)
    if y_continuous:  # [1] Eq. 1.1
        statistic = 1 - 3 * num / (n ** 2 - 1)
    else:  # [1] Eq. 1.2
        den = 2 * np.sum((n - l) * l, axis=-1)
        statistic = 1 - n * num / den

    return statistic, r, l


def _xi_std(r, l, y_continuous):
    # Compute asymptotic standard deviation of xi under null hypothesis of independence

    # `axis=-1` is guaranteed by _axis_nan_policy decorator
    n = np.float64(r.shape[-1])

    # "Suppose that X and Y are independent and Y is continuous. Then
    # √n·ξn(X, Y) → N(0, 2/5) in distribution as n → ∞"
    if y_continuous:  # [1] Theorem 2.1
        return np.sqrt(2 / 5) / np.sqrt(n)

    # "Suppose that X and Y are independent. Then √n·ξn(X, Y)
    # converges to N(0, τ²) in distribution as n → ∞
    # [1] Eq. 2.2 and surrounding math
    i = np.arange(1, n + 1)
    u = np.sort(r, axis=-1)
    v = np.cumsum(u, axis=-1)
    an = 1 / n**4 * np.sum((2*n - 2*i + 1) * u**2, axis=-1)
    bn = 1 / n**5 * np.sum((v + (n - i)*u)**2, axis=-1)
    cn = 1 / n**3 * np.sum((2*n - 2*i + 1) * u, axis=-1)
    dn = 1 / n**3 * np.sum((l * (n - l)), axis=-1)
    tau2 = (an - 2*bn + cn**2) / dn**2

    return np.sqrt(tau2) / np.sqrt(n)


def _chatterjeexi_iv(y_continuous, method):
    # Input validation for `chatterjeexi`
    # x, y, `axis` input validation taken care of by decorator

    if y_continuous not in {True, False}:
        raise ValueError('`y_continuous` must be boolean.')

    if not isinstance(method, stats.PermutationMethod):
        method = method.lower()
        message = "`method` must be 'asymptotic' or a `PermutationMethod` instance."
        if method != 'asymptotic':
            raise ValueError(message)

    return y_continuous, method


def _unpack(res):
    return res.statistic, res.pvalue


@_axis_nan_policy_factory(SignificanceResult, paired=True, n_samples=2,
                          result_to_tuple=_unpack, n_outputs=2, too_small=1)
def chatterjeexi(x, y, *, axis=0, y_continuous=False, method='asymptotic'):
    r"""Compute the xi correlation and perform a test of independence

    The xi correlation coefficient is a measure of association between two
    variables; the value tends to be close to zero when the variables are
    independent and close to 1 when there is a strong association. Unlike
    other correlation coefficients, the xi correlation is effective even
    when the association is not monotonic.

    Parameters
    ----------
    x, y : array-like
        The samples: corresponding observations of the independent and
        dependent variable. The (N-d) arrays must be broadcastable.
    axis : int, default: 0
        Axis along which to perform the test.
    method : 'asymptotic' or `PermutationMethod` instance, optional
        Selects the method used to calculate the *p*-value.
        Default is 'asymptotic'. The following options are available.

        * ``'asymptotic'``: compares the standardized test statistic
          against the normal distribution.
        * `PermutationMethod` instance. In this case, the p-value
          is computed using `permutation_test` with the provided
          configuration options and other appropriate settings.

    y_continuous : bool, default: False
        Whether `y` is assumed to be drawn from a continuous distribution.
        If `y` is drawn from a continuous distribution, results are valid
        whether this is assumed or not, but enabling this assumption will
        result in faster computation and typically produce similar results.

    Returns
    -------
    res : SignificanceResult
        An object containing attributes:

        statistic : float
            The xi correlation statistic.
        pvalue : float
            The associated *p*-value: the probability of a statistic at least as
            high as the observed value under the null hypothesis of independence.

    See Also
    --------
    scipy.stats.pearsonr, scipy.stats.spearmanr, scipy.stats.kendalltau

    Notes
    -----
    There is currently no special handling of ties in `x`; they are broken arbitrarily
    by the implementation.

    [1]_ notes that the statistic is not symmetric in `x` and `y` *by design*:
    "...we may want to understand if :math:`Y` is a function :math:`X`, and not just
    if one of the variables is a function of the other." See [1]_ Remark 1.

    References
    ----------
    .. [1] Chatterjee, Sourav. "A new coefficient of correlation." Journal of
           the American Statistical Association 116.536 (2021): 2009-2022.
           :doi:`10.1080/01621459.2020.1758115`.

    Examples
    --------
    Generate perfectly correlated data, and observe that the xi correlation is
    nearly 1.0.

    >>> import numpy as np
    >>> from scipy import stats
    >>> rng = np.random.default_rng(348932549825235)
    >>> x = rng.uniform(0, 10, size=100)
    >>> y = np.sin(x)
    >>> res = stats.chatterjeexi(x, y)
    >>> res.statistic
    np.float64(0.9012901290129013)

    The probability of observing such a high value of the statistic under the
    null hypothesis of independence is very low.

    >>> res.pvalue
    np.float64(2.2206974648177804e-46)

    As noise is introduced, the correlation coefficient decreases.

    >>> noise = rng.normal(scale=[[0.1], [0.5], [1]], size=(3, 100))
    >>> res = stats.chatterjeexi(x, y + noise, axis=-1)
    >>> res.statistic
    array([0.79507951, 0.41824182, 0.16651665])

    Because the distribution of `y` is continuous, it is valid to pass
    ``y_continuous=True``. The statistic is identical, and the p-value
    (not shown) is only slightly different.

    >>> stats.chatterjeexi(x, y + noise, y_continuous=True, axis=-1).statistic
    array([0.79507951, 0.41824182, 0.16651665])

    """
    # x, y, `axis` input validation taken care of by decorator
    # In fact, `axis` is guaranteed to be -1
    y_continuous, method = _chatterjeexi_iv(y_continuous, method)

    # A highly negative statistic is possible, e.g.
    # x = np.arange(100.), y = (x % 2 == 0)
    # Unclear whether we should expose `alternative`, though.
    alternative = 'greater'

    if method == 'asymptotic':
        xi, r, l = _xi_statistic(x, y, y_continuous)
        std = _xi_std(r, l, y_continuous)
        norm = _SimpleNormal()
        pvalue = _get_pvalue(xi / std, norm, alternative=alternative)
    elif isinstance(method, stats.PermutationMethod):
        res = stats.permutation_test(
            # Could be faster if we just permuted the ranks; for now, keep it simple.
            data=(y,), statistic=lambda y, axis: _xi_statistic(x, y, y_continuous)[0],
            alternative=alternative, permutation_type='pairings', **method._asdict(),
            axis=-1)  # `axis=-1` is guaranteed by _axis_nan_policy decorator

        xi, pvalue = res.statistic, res.pvalue

    return SignificanceResult(xi, pvalue)
<|endoftext|>import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

import numpy as np

from scipy.stats._common import ConfidenceInterval
from scipy.stats._qmc import check_random_state
from scipy.stats._resampling import BootstrapResult
from scipy.stats import qmc, bootstrap
from scipy._lib._util import _transition_to_rng


if TYPE_CHECKING:
    import numpy.typing as npt
    from scipy._lib._util import DecimalNumber, IntNumber


__all__ = [
    'sobol_indices'
]


def f_ishigami(x: "npt.ArrayLike") -> "npt.NDArray[np.inexact[Any]]":
    r"""Ishigami function.

    .. math::

        Y(\mathbf{x}) = \sin x_1 + 7 \sin^2 x_2 + 0.1 x_3^4 \sin x_1

    with :math:`\mathbf{x} \in [-\pi, \pi]^3`.

    Parameters
    ----------
    x : array_like ([x1, x2, x3], n)

    Returns
    -------
    f : array_like (n,)
        Function evaluation.

    References
    ----------
    .. [1] Ishigami, T. and T. Homma. "An importance quantification technique
       in uncertainty analysis for computer models." IEEE,
       :doi:`10.1109/ISUMA.1990.151285`, 1990.
    """
    x = np.atleast_2d(x)
    f_eval = (
        np.sin(x[0])
        + 7 * np.sin(x[1])**2
        + 0.1 * (x[2]**4) * np.sin(x[0])
    )
    return f_eval


def sample_A_B(
    n,
    dists,
    rng=None
):
    """Sample two matrices A and B.

    Uses a Sobol' sequence with 2`d` columns to have 2 uncorrelated matrices.
    This is more efficient than using 2 random draw of Sobol'.
    See sec. 5 from [1]_.

    Output shape is (d, n).

    References
    ----------
    .. [1] Saltelli, A., P. Annoni, I. Azzini, F. Campolongo, M. Ratto, and
       S. Tarantola. "Variance based sensitivity analysis of model
       output. Design and estimator for the total sensitivity index."
       Computer Physics Communications, 181(2):259-270,
       :doi:`10.1016/j.cpc.2009.09.018`, 2010.
    """
    d = len(dists)
    A_B = qmc.Sobol(d=2*d, seed=rng, bits=64).random(n).T
    A_B = A_B.reshape(2, d, -1)
    try:
        for d_, dist in enumerate(dists):
            A_B[:, d_] = dist.ppf(A_B[:, d_])
    except AttributeError as exc:
        message = "Each distribution in `dists` must have method `ppf`."
        raise ValueError(message) from exc
    return A_B


def sample_AB(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """AB matrix.

    AB: rows of B into A. Shape (d, d, n).
    - Copy A into d "pages"
    - In the first page, replace 1st rows of A with 1st row of B.
    ...
    - In the dth page, replace dth row of A with dth row of B.
    - return the stack of pages
    """
    d, n = A.shape
    AB = np.tile(A, (d, 1, 1))
    i = np.arange(d)
    AB[i, i] = B[i]
    return AB


def saltelli_2010(
    f_A: np.ndarray, f_B: np.ndarray, f_AB: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""Saltelli2010 formulation.

    .. math::

        S_i = \frac{1}{N} \sum_{j=1}^N
        f(\mathbf{B})_j (f(\mathbf{AB}^{(i)})_j - f(\mathbf{A})_j)

    .. math::

        S_{T_i} = \frac{1}{N} \sum_{j=1}^N
        (f(\mathbf{A})_j - f(\mathbf{AB}^{(i)})_j)^2

    Parameters
    ----------
    f_A, f_B : array_like (s, n)
        Function values at A and B, respectively
    f_AB : array_like (d, s, n)
        Function values at each of the AB pages

    Returns
    -------
    s, st : array_like (s, d)
        First order and total order Sobol' indices.

    References
    ----------
    .. [1] Saltelli, A., P. Annoni, I. Azzini, F. Campolongo, M. Ratto, and
       S. Tarantola. "Variance based sensitivity analysis of model
       output. Design and estimator for the total sensitivity index."
       Computer Physics Communications, 181(2):259-270,
       :doi:`10.1016/j.cpc.2009.09.018`, 2010.
    """
    # Empirical variance calculated using output from A and B which are
    # independent. Output of AB is not independent and cannot be used
    var = np.var([f_A, f_B], axis=(0, -1))

    # We divide by the variance to have a ratio of variance
    # this leads to eq. 2
    s = np.mean(f_B * (f_AB - f_A), axis=-1) / var  # Table 2 (b)
    st = 0.5 * np.mean((f_A - f_AB) ** 2, axis=-1) / var  # Table 2 (f)

    return s.T, st.T


@dataclass
class BootstrapSobolResult:
    first_order: BootstrapResult
    total_order: BootstrapResult


@dataclass
class SobolResult:
    first_order: np.ndarray
    total_order: np.ndarray
    _indices_method: Callable
    _f_A: np.ndarray
    _f_B: np.ndarray
    _f_AB: np.ndarray
    _A: np.ndarray | None = None
    _B: np.ndarray | None = None
    _AB: np.ndarray | None = None
    _bootstrap_result: BootstrapResult | None = None

    def bootstrap(
        self,
        confidence_level: "DecimalNumber" = 0.95,
        n_resamples: "IntNumber" = 999
    ) -> BootstrapSobolResult:
        """Bootstrap Sobol' indices to provide confidence intervals.

        Parameters
        ----------
        confidence_level : float, default: ``0.95``
            The confidence level of the confidence intervals.
        n_resamples : int, default: ``999``
            The number of resamples performed to form the bootstrap
            distribution of the indices.

        Returns
        -------
        res : BootstrapSobolResult
            Bootstrap result containing the confidence intervals and the
            bootstrap distribution of the indices.

            An object with attributes:

            first_order : BootstrapResult
                Bootstrap result of the first order indices.
            total_order : BootstrapResult
                Bootstrap result of the total order indices.
            See `BootstrapResult` for more details.

        """
        def statistic(idx):
            f_A_ = self._f_A[:, idx]
            f_B_ = self._f_B[:, idx]
            f_AB_ = self._f_AB[..., idx]
            return self._indices_method(f_A_, f_B_, f_AB_)

        n = self._f_A.shape[1]

        res = bootstrap(
            [np.arange(n)], statistic=statistic, method="BCa",
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            bootstrap_result=self._bootstrap_result
        )
        self._bootstrap_result = res

        first_order = BootstrapResult(
            confidence_interval=ConfidenceInterval(
                res.confidence_interval.low[0], res.confidence_interval.high[0]
            ),
            bootstrap_distribution=res.bootstrap_distribution[0],
            standard_error=res.standard_error[0],
        )
        total_order = BootstrapResult(
            confidence_interval=ConfidenceInterval(
                res.confidence_interval.low[1], res.confidence_interval.high[1]
            ),
            bootstrap_distribution=res.bootstrap_distribution[1],
            standard_error=res.standard_error[1],
        )

        return BootstrapSobolResult(
            first_order=first_order, total_order=total_order
        )

@_transition_to_rng('random_state', replace_doc=False)
def sobol_indices(
    *,
    func,
    n,
    dists=None,
    method='saltelli_2010',
    rng=None
):
    r"""Global sensitivity indices of Sobol'.

    Parameters
    ----------
    func : callable or dict(str, array_like)
        If `func` is a callable, function to compute the Sobol' indices from.
        Its signature must be::

            func(x: ArrayLike) -> ArrayLike

        with ``x`` of shape ``(d, n)`` and output of shape ``(s, n)`` where:

        - ``d`` is the input dimensionality of `func`
          (number of input variables),
        - ``s`` is the output dimensionality of `func`
          (number of output variables), and
        - ``n`` is the number of samples (see `n` below).

        Function evaluation values must be finite.

        If `func` is a dictionary, contains the function evaluations from three
        different arrays. Keys must be: ``f_A``, ``f_B`` and ``f_AB``.
        ``f_A`` and ``f_B`` should have a shape ``(s, n)`` and ``f_AB``
        should have a shape ``(d, s, n)``.
        This is an advanced feature and misuse can lead to wrong analysis.
    n : int
        Number of samples used to generate the matrices ``A`` and ``B``.
        Must be a power of 2. The total number of points at which `func` is
        evaluated will be ``n*(d+2)``.
    dists : list(distributions), optional
        List of each parameter's distribution. The distribution of parameters
        depends on the application and should be carefully chosen.
        Parameters are assumed to be independently distributed, meaning there
        is no constraint nor relationship between their values.

        Distributions must be an instance of a class with a ``ppf``
        method.

        Must be specified if `func` is a callable, and ignored otherwise.
    method : Callable or str, default: 'saltelli_2010'
        Method used to compute the first and total Sobol' indices.

        If a callable, its signature must be::

            func(f_A: np.ndarray, f_B: np.ndarray, f_AB: np.ndarray)
            -> Tuple[np.ndarray, np.ndarray]

        with ``f_A, f_B`` of shape ``(s, n)`` and ``f_AB`` of shape
        ``(d, s, n)``.
        These arrays contain the function evaluations from three different sets
        of samples.
        The output is a tuple of the first and total indices with
        shape ``(s, d)``.
        This is an advanced feature and misuse can lead to wrong analysis.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.

        .. versionchanged:: 1.15.0

            As part of the `SPEC-007 <https://scientific-python.org/specs/spec-0007/>`_
            transition from use of `numpy.random.RandomState` to
            `numpy.random.Generator`, this keyword was changed from `random_state` to
            `rng`. For an interim period, both keywords will continue to work, although
            only one may be specified at a time. After the interim period, function
            calls using the `random_state` keyword will emit warnings. Following a
            deprecation period, the `random_state` keyword will be removed.

    Returns
    -------
    res : SobolResult
        An object with attributes:

        first_order : ndarray of shape (s, d)
            First order Sobol' indices.
        total_order : ndarray of shape (s, d)
            Total order Sobol' indices.

        And method:

        bootstrap(confidence_level: float, n_resamples: int)
        -> BootstrapSobolResult

            A method providing confidence intervals on the indices.
            See `scipy.stats.bootstrap` for more details.

            The bootstrapping is done on both first and total order indices,
            and they are available in `BootstrapSobolResult` as attributes
            ``first_order`` and ``total_order``.

    Notes
    -----
    The Sobol' method [1]_, [2]_ is a variance-based Sensitivity Analysis which
    obtains the contribution of each parameter to the variance of the
    quantities of interest (QoIs; i.e., the outputs of `func`).
    Respective contributions can be used to rank the parameters and
    also gauge the complexity of the model by computing the
    model's effective (or mean) dimension.

    .. note::

        Parameters are assumed to be independently distributed. Each
        parameter can still follow any distribution. In fact, the distribution
        is very important and should match the real distribution of the
        parameters.

    It uses a functional decomposition of the variance of the function to
    explore

    .. math::

        \mathbb{V}(Y) = \sum_{i}^{d} \mathbb{V}_i (Y) + \sum_{i<j}^{d}
        \mathbb{V}_{ij}(Y) + ... + \mathbb{V}_{1,2,...,d}(Y),

    introducing conditional variances:

    .. math::

        \mathbb{V}_i(Y) = \mathbb{\mathbb{V}}[\mathbb{E}(Y|x_i)]
        \qquad
        \mathbb{V}_{ij}(Y) = \mathbb{\mathbb{V}}[\mathbb{E}(Y|x_i x_j)]
        - \mathbb{V}_i(Y) - \mathbb{V}_j(Y),

    Sobol' indices are expressed as

    .. math::

        S_i = \frac{\mathbb{V}_i(Y)}{\mathbb{V}[Y]}
        \qquad
        S_{ij} =\frac{\mathbb{V}_{ij}(Y)}{\mathbb{V}[Y]}.

    :math:`S_{i}` corresponds to the first-order term which apprises the
    contribution of the i-th parameter, while :math:`S_{ij}` corresponds to the
    second-order term which informs about the contribution of interactions
    between the i-th and the j-th parameters. These equations can be
    generalized to compute higher order terms; however, they are expensive to
    compute and their interpretation is complex.
    This is why only first order indices are provided.

    Total order indices represent the global contribution of the parameters
    to the variance of the QoI and are defined as:

    .. math::

        S_{T_i} = S_i + \sum_j S_{ij} + \sum_{j,k} S_{ijk} + ...
        = 1 - \frac{\mathbb{V}[\mathbb{E}(Y|x_{\sim i})]}{\mathbb{V}[Y]}.

    First order indices sum to at most 1, while total order indices sum to at
    least 1. If there are no interactions, then first and total order indices
    are equal, and both first and total order indices sum to 1.

    .. warning::

        Negative Sobol' values are due to numerical errors. Increasing the
        number of points `n` should help.

        The number of sample required to have a good analysis increases with
        the dimensionality of the problem. e.g. for a 3 dimension problem,
        consider at minima ``n >= 2**12``. The more complex the model is,
        the more samples will be needed.

        Even for a purely additive model, the indices may not sum to 1 due
        to numerical noise.

    References
    ----------
    .. [1] Sobol, I. M.. "Sensitivity analysis for nonlinear mathematical
       models." Mathematical Modeling and Computational Experiment, 1:407-414,
       1993.
    .. [2] Sobol, I. M. (2001). "Global sensitivity indices for nonlinear
       mathematical models and their Monte Carlo estimates." Mathematics
       and Computers in Simulation, 55(1-3):271-280,
       :doi:`10.1016/S0378-4754(00)00270-6`, 2001.
    .. [3] Saltelli, A. "Making best use of model evaluations to
       compute sensitivity indices."  Computer Physics Communications,
       145(2):280-297, :doi:`10.1016/S0010-4655(02)00280-1`, 2002.
    .. [4] Saltelli, A., M. Ratto, T. Andres, F. Campolongo, J. Cariboni,
       D. Gatelli, M. Saisana, and S. Tarantola. "Global Sensitivity Analysis.
       The Primer." 2007.
    .. [5] Saltelli, A., P. Annoni, I. Azzini, F. Campolongo, M. Ratto, and
       S. Tarantola. "Variance based sensitivity analysis of model
       output. Design and estimator for the total sensitivity index."
       Computer Physics Communications, 181(2):259-270,
       :doi:`10.1016/j.cpc.2009.09.018`, 2010.
    .. [6] Ishigami, T. and T. Homma. "An importance quantification technique
       in uncertainty analysis for computer models." IEEE,
       :doi:`10.1109/ISUMA.1990.151285`, 1990.

    Examples
    --------
    The following is an example with the Ishigami function [6]_

    .. math::

        Y(\mathbf{x}) = \sin x_1 + 7 \sin^2 x_2 + 0.1 x_3^4 \sin x_1,

    with :math:`\mathbf{x} \in [-\pi, \pi]^3`. This function exhibits strong
    non-linearity and non-monotonicity.

    Remember, Sobol' indices assumes that samples are independently
    distributed. In this case we use a uniform distribution on each marginals.

    >>> import numpy as np
    >>> from scipy.stats import sobol_indices, uniform
    >>> rng = np.random.default_rng()
    >>> def f_ishigami(x):
    ...     f_eval = (
    ...         np.sin(x[0])
    ...         + 7 * np.sin(x[1])**2
    ...         + 0.1 * (x[2]**4) * np.sin(x[0])
    ...     )
    ...     return f_eval
    >>> indices = sobol_indices(
    ...     func=f_ishigami, n=1024,
    ...     dists=[
    ...         uniform(loc=-np.pi, scale=2*np.pi),
    ...         uniform(loc=-np.pi, scale=2*np.pi),
    ...         uniform(loc=-np.pi, scale=2*np.pi)
    ...     ],
    ...     rng=rng
    ... )
    >>> indices.first_order
    array([0.31637954, 0.43781162, 0.00318825])
    >>> indices.total_order
    array([0.56122127, 0.44287857, 0.24229595])

    Confidence interval can be obtained using bootstrapping.

    >>> boot = indices.bootstrap()

    Then, this information can be easily visualized.

    >>> import matplotlib.pyplot as plt
    >>> fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    >>> _ = axs[0].errorbar(
    ...     [1, 2, 3], indices.first_order, fmt='o',
    ...     yerr=[
    ...         indices.first_order - boot.first_order.confidence_interval.low,
    ...         boot.first_order.confidence_interval.high - indices.first_order
    ...     ],
    ... )
    >>> axs[0].set_ylabel("First order Sobol' indices")
    >>> axs[0].set_xlabel('Input parameters')
    >>> axs[0].set_xticks([1, 2, 3])
    >>> _ = axs[1].errorbar(
    ...     [1, 2, 3], indices.total_order, fmt='o',
    ...     yerr=[
    ...         indices.total_order - boot.total_order.confidence_interval.low,
    ...         boot.total_order.confidence_interval.high - indices.total_order
    ...     ],
    ... )
    >>> axs[1].set_ylabel("Total order Sobol' indices")
    >>> axs[1].set_xlabel('Input parameters')
    >>> axs[1].set_xticks([1, 2, 3])
    >>> plt.tight_layout()
    >>> plt.show()

    .. note::

        By default, `scipy.stats.uniform` has support ``[0, 1]``.
        Using the parameters ``loc`` and ``scale``, one obtains the uniform
        distribution on ``[loc, loc + scale]``.

    This result is particularly interesting because the first order index
