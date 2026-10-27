            differing set of significance levels depending on the
            distribution that is being tested against.
        fit_result : `~scipy.stats._result_classes.FitResult`
            An object containing the results of fitting the distribution to
            the data.

    See Also
    --------
    kstest : The Kolmogorov-Smirnov test for goodness-of-fit.

    Notes
    -----
    Critical values provided are for the following significance levels:

    normal/exponential
        15%, 10%, 5%, 2.5%, 1%
    logistic
        25%, 10%, 5%, 2.5%, 1%, 0.5%
    gumbel_l / gumbel_r
        25%, 10%, 5%, 2.5%, 1%
    weibull_min
        50%, 25%, 15%, 10%, 5%, 2.5%, 1%, 0.5%

    If the returned statistic is larger than these critical values then
    for the corresponding significance level, the null hypothesis that
    the data come from the chosen distribution can be rejected.
    The returned statistic is referred to as 'A2' in the references.

    For `weibull_min`, maximum likelihood estimation is known to be
    challenging. If the test returns successfully, then the first order
    conditions for a maximum likelihood estimate have been verified and
    the critical values correspond relatively well to the significance levels,
    provided that the sample is sufficiently large (>10 observations [7]).
    However, for some data - especially data with no left tail - `anderson`
    is likely to result in an error message. In this case, consider
    performing a custom goodness of fit test using
    `scipy.stats.monte_carlo_test`.

    References
    ----------
    .. [1] https://www.itl.nist.gov/div898/handbook/prc/section2/prc213.htm
    .. [2] Stephens, M. A. (1974). EDF Statistics for Goodness of Fit and
           Some Comparisons, Journal of the American Statistical Association,
           Vol. 69, pp. 730-737.
    .. [3] Stephens, M. A. (1976). Asymptotic Results for Goodness-of-Fit
           Statistics with Unknown Parameters, Annals of Statistics, Vol. 4,
           pp. 357-369.
    .. [4] Stephens, M. A. (1977). Goodness of Fit for the Extreme Value
           Distribution, Biometrika, Vol. 64, pp. 583-588.
    .. [5] Stephens, M. A. (1977). Goodness of Fit with Special Reference
           to Tests for Exponentiality , Technical Report No. 262,
           Department of Statistics, Stanford University, Stanford, CA.
    .. [6] Stephens, M. A. (1979). Tests of Fit for the Logistic Distribution
           Based on the Empirical Distribution Function, Biometrika, Vol. 66,
           pp. 591-595.
    .. [7] Richard A. Lockhart and Michael A. Stephens "Estimation and Tests of
           Fit for the Three-Parameter Weibull Distribution"
           Journal of the Royal Statistical Society.Series B(Methodological)
           Vol. 56, No. 3 (1994), pp. 491-500, Table 0.

    Examples
    --------
    Test the null hypothesis that a random sample was drawn from a normal
    distribution (with unspecified mean and standard deviation).

    >>> import numpy as np
    >>> from scipy.stats import anderson
    >>> rng = np.random.default_rng()
    >>> data = rng.random(size=35)
    >>> res = anderson(data)
    >>> res.statistic
    0.8398018749744764
    >>> res.critical_values
    array([0.527, 0.6  , 0.719, 0.839, 0.998])
    >>> res.significance_level
    array([15. , 10. ,  5. ,  2.5,  1. ])

    The value of the statistic (barely) exceeds the critical value associated
    with a significance level of 2.5%, so the null hypothesis may be rejected
    at a significance level of 2.5%, but not at a significance level of 1%.

    """ # numpy/numpydoc#87  # noqa: E501
    dist = dist.lower()
    if dist in {'extreme1', 'gumbel'}:
        dist = 'gumbel_l'
    dists = {'norm', 'expon', 'gumbel_l',
             'gumbel_r', 'logistic', 'weibull_min'}

    if dist not in dists:
        raise ValueError(f"Invalid distribution; dist must be in {dists}.")
    y = sort(x)
    xbar = np.mean(x, axis=0)
    N = len(y)
    if dist == 'norm':
        s = np.std(x, ddof=1, axis=0)
        w = (y - xbar) / s
        fit_params = xbar, s
        logcdf = distributions.norm.logcdf(w)
        logsf = distributions.norm.logsf(w)
        sig = array([15, 10, 5, 2.5, 1])
        critical = around(_Avals_norm / (1.0 + 4.0/N - 25.0/N/N), 3)
    elif dist == 'expon':
        w = y / xbar
        fit_params = 0, xbar
        logcdf = distributions.expon.logcdf(w)
        logsf = distributions.expon.logsf(w)
        sig = array([15, 10, 5, 2.5, 1])
        critical = around(_Avals_expon / (1.0 + 0.6/N), 3)
    elif dist == 'logistic':
        def rootfunc(ab, xj, N):
            a, b = ab
            tmp = (xj - a) / b
            tmp2 = exp(tmp)
            val = [np.sum(1.0/(1+tmp2), axis=0) - 0.5*N,
                   np.sum(tmp*(1.0-tmp2)/(1+tmp2), axis=0) + N]
            return array(val)

        sol0 = array([xbar, np.std(x, ddof=1, axis=0)])
        sol = optimize.fsolve(rootfunc, sol0, args=(x, N), xtol=1e-5)
        w = (y - sol[0]) / sol[1]
        fit_params = sol
        logcdf = distributions.logistic.logcdf(w)
        logsf = distributions.logistic.logsf(w)
        sig = array([25, 10, 5, 2.5, 1, 0.5])
        critical = around(_Avals_logistic / (1.0 + 0.25/N), 3)
    elif dist == 'gumbel_r':
        xbar, s = distributions.gumbel_r.fit(x)
        w = (y - xbar) / s
        fit_params = xbar, s
        logcdf = distributions.gumbel_r.logcdf(w)
        logsf = distributions.gumbel_r.logsf(w)
        sig = array([25, 10, 5, 2.5, 1])
        critical = around(_Avals_gumbel / (1.0 + 0.2/sqrt(N)), 3)
    elif dist == 'gumbel_l':
        xbar, s = distributions.gumbel_l.fit(x)
        w = (y - xbar) / s
        fit_params = xbar, s
        logcdf = distributions.gumbel_l.logcdf(w)
        logsf = distributions.gumbel_l.logsf(w)
        sig = array([25, 10, 5, 2.5, 1])
        critical = around(_Avals_gumbel / (1.0 + 0.2/sqrt(N)), 3)
    elif dist == 'weibull_min':
        message = ("Critical values of the test statistic are given for the "
                   "asymptotic distribution. These may not be accurate for "
                   "samples with fewer than 10 observations. Consider using "
                   "`scipy.stats.monte_carlo_test`.")
        if N < 10:
            warnings.warn(message, stacklevel=2)
        # [7] writes our 'c' as 'm', and they write `c = 1/m`. Use their names.
        m, loc, scale = distributions.weibull_min.fit(y)
        m, loc, scale = _weibull_fit_check((m, loc, scale), y)
        fit_params = m, loc, scale
        logcdf = stats.weibull_min(*fit_params).logcdf(y)
        logsf = stats.weibull_min(*fit_params).logsf(y)
        c = 1 / m  # m and c are as used in [7]
        sig = array([0.5, 0.75, 0.85, 0.9, 0.95, 0.975, 0.99, 0.995])
        critical = _get_As_weibull(c)
        # Goodness-of-fit tests should only be used to provide evidence
        # _against_ the null hypothesis. Be conservative and round up.
        critical = np.round(critical + 0.0005, decimals=3)

    i = arange(1, N + 1)
    A2 = -N - np.sum((2*i - 1.0) / N * (logcdf + logsf[::-1]), axis=0)

    # FitResult initializer expects an optimize result, so let's work with it
    message = '`anderson` successfully fit the distribution to the data.'
    res = optimize.OptimizeResult(success=True, message=message)
    res.x = np.array(fit_params)
    fit_result = FitResult(getattr(distributions, dist), y,
                           discrete=False, res=res)

    return AndersonResult(A2, critical, sig, fit_result=fit_result)


def _anderson_ksamp_midrank(samples, Z, Zstar, k, n, N):
    """Compute A2akN equation 7 of Scholz and Stephens.

    Parameters
    ----------
    samples : sequence of 1-D array_like
        Array of sample arrays.
    Z : array_like
        Sorted array of all observations.
    Zstar : array_like
        Sorted array of unique observations.
    k : int
        Number of samples.
    n : array_like
        Number of observations in each sample.
    N : int
        Total number of observations.

    Returns
    -------
    A2aKN : float
        The A2aKN statistics of Scholz and Stephens 1987.

    """
    A2akN = 0.
    Z_ssorted_left = Z.searchsorted(Zstar, 'left')
    if N == Zstar.size:
        lj = 1.
    else:
        lj = Z.searchsorted(Zstar, 'right') - Z_ssorted_left
    Bj = Z_ssorted_left + lj / 2.
    for i in arange(0, k):
        s = np.sort(samples[i])
        s_ssorted_right = s.searchsorted(Zstar, side='right')
        Mij = s_ssorted_right.astype(float)
        fij = s_ssorted_right - s.searchsorted(Zstar, 'left')
        Mij -= fij / 2.
        inner = lj / float(N) * (N*Mij - Bj*n[i])**2 / (Bj*(N - Bj) - N*lj/4.)
        A2akN += inner.sum() / n[i]
    A2akN *= (N - 1.) / N
    return A2akN


def _anderson_ksamp_right(samples, Z, Zstar, k, n, N):
    """Compute A2akN equation 6 of Scholz & Stephens.

    Parameters
    ----------
    samples : sequence of 1-D array_like
        Array of sample arrays.
    Z : array_like
        Sorted array of all observations.
    Zstar : array_like
        Sorted array of unique observations.
    k : int
        Number of samples.
    n : array_like
        Number of observations in each sample.
    N : int
        Total number of observations.

    Returns
    -------
    A2KN : float
        The A2KN statistics of Scholz and Stephens 1987.

    """
    A2kN = 0.
    lj = Z.searchsorted(Zstar[:-1], 'right') - Z.searchsorted(Zstar[:-1],
                                                              'left')
    Bj = lj.cumsum()
    for i in arange(0, k):
        s = np.sort(samples[i])
        Mij = s.searchsorted(Zstar[:-1], side='right')
        inner = lj / float(N) * (N * Mij - Bj * n[i])**2 / (Bj * (N - Bj))
        A2kN += inner.sum() / n[i]
    return A2kN


Anderson_ksampResult = _make_tuple_bunch(
    'Anderson_ksampResult',
    ['statistic', 'critical_values', 'pvalue'], []
)


def anderson_ksamp(samples, midrank=True, *, method=None):
    """The Anderson-Darling test for k-samples.

    The k-sample Anderson-Darling test is a modification of the
    one-sample Anderson-Darling test. It tests the null hypothesis
    that k-samples are drawn from the same population without having
    to specify the distribution function of that population. The
    critical values depend on the number of samples.

    Parameters
    ----------
    samples : sequence of 1-D array_like
        Array of sample data in arrays.
    midrank : bool, optional
        Type of Anderson-Darling test which is computed. Default
        (True) is the midrank test applicable to continuous and
        discrete populations. If False, the right side empirical
        distribution is used.
    method : PermutationMethod, optional
        Defines the method used to compute the p-value. If `method` is an
        instance of `PermutationMethod`, the p-value is computed using
        `scipy.stats.permutation_test` with the provided configuration options
        and other appropriate settings. Otherwise, the p-value is interpolated
        from tabulated values.

    Returns
    -------
    res : Anderson_ksampResult
        An object containing attributes:

        statistic : float
            Normalized k-sample Anderson-Darling test statistic.
        critical_values : array
            The critical values for significance levels 25%, 10%, 5%, 2.5%, 1%,
            0.5%, 0.1%.
        pvalue : float
            The approximate p-value of the test. If `method` is not
            provided, the value is floored / capped at 0.1% / 25%.

    Raises
    ------
    ValueError
        If fewer than 2 samples are provided, a sample is empty, or no
        distinct observations are in the samples.

    See Also
    --------
    ks_2samp : 2 sample Kolmogorov-Smirnov test
    anderson : 1 sample Anderson-Darling test

    Notes
    -----
    [1]_ defines three versions of the k-sample Anderson-Darling test:
    one for continuous distributions and two for discrete
    distributions, in which ties between samples may occur. The
    default of this routine is to compute the version based on the
    midrank empirical distribution function. This test is applicable
    to continuous and discrete data. If midrank is set to False, the
    right side empirical distribution is used for a test for discrete
    data. According to [1]_, the two discrete test statistics differ
    only slightly if a few collisions due to round-off errors occur in
    the test not adjusted for ties between samples.

    The critical values corresponding to the significance levels from 0.01
    to 0.25 are taken from [1]_. p-values are floored / capped
    at 0.1% / 25%. Since the range of critical values might be extended in
    future releases, it is recommended not to test ``p == 0.25``, but rather
    ``p >= 0.25`` (analogously for the lower bound).

    .. versionadded:: 0.14.0

    References
    ----------
    .. [1] Scholz, F. W and Stephens, M. A. (1987), K-Sample
           Anderson-Darling Tests, Journal of the American Statistical
           Association, Vol. 82, pp. 918-924.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> rng = np.random.default_rng()
    >>> res = stats.anderson_ksamp([rng.normal(size=50),
    ... rng.normal(loc=0.5, size=30)])
    >>> res.statistic, res.pvalue
    (1.974403288713695, 0.04991293614572478)
    >>> res.critical_values
    array([0.325, 1.226, 1.961, 2.718, 3.752, 4.592, 6.546])

    The null hypothesis that the two random samples come from the same
    distribution can be rejected at the 5% level because the returned
    test value is greater than the critical value for 5% (1.961) but
    not at the 2.5% level. The interpolation gives an approximate
    p-value of 4.99%.

    >>> samples = [rng.normal(size=50), rng.normal(size=30),
    ...            rng.normal(size=20)]
    >>> res = stats.anderson_ksamp(samples)
    >>> res.statistic, res.pvalue
    (-0.29103725200789504, 0.25)
    >>> res.critical_values
    array([ 0.44925884,  1.3052767 ,  1.9434184 ,  2.57696569,  3.41634856,
      4.07210043, 5.56419101])

    The null hypothesis cannot be rejected for three samples from an
    identical distribution. The reported p-value (25%) has been capped and
    may not be very accurate (since it corresponds to the value 0.449
    whereas the statistic is -0.291).

    In such cases where the p-value is capped or when sample sizes are
    small, a permutation test may be more accurate.

    >>> method = stats.PermutationMethod(n_resamples=9999, random_state=rng)
    >>> res = stats.anderson_ksamp(samples, method=method)
    >>> res.pvalue
    0.5254

    """
    k = len(samples)
    if (k < 2):
        raise ValueError("anderson_ksamp needs at least two samples")

    samples = list(map(np.asarray, samples))
    Z = np.sort(np.hstack(samples))
    N = Z.size
    Zstar = np.unique(Z)
    if Zstar.size < 2:
        raise ValueError("anderson_ksamp needs more than one distinct "
                         "observation")

    n = np.array([sample.size for sample in samples])
    if np.any(n == 0):
        raise ValueError("anderson_ksamp encountered sample without "
                         "observations")

    if midrank:
        A2kN_fun = _anderson_ksamp_midrank
    else:
        A2kN_fun = _anderson_ksamp_right
    A2kN = A2kN_fun(samples, Z, Zstar, k, n, N)

    def statistic(*samples):
        return A2kN_fun(samples, Z, Zstar, k, n, N)

    if method is not None:
        res = stats.permutation_test(samples, statistic, **method._asdict(),
                                     alternative='greater')

    H = (1. / n).sum()
    hs_cs = (1. / arange(N - 1, 1, -1)).cumsum()
    h = hs_cs[-1] + 1
    g = (hs_cs / arange(2, N)).sum()

    a = (4*g - 6) * (k - 1) + (10 - 6*g)*H
    b = (2*g - 4)*k**2 + 8*h*k + (2*g - 14*h - 4)*H - 8*h + 4*g - 6
    c = (6*h + 2*g - 2)*k**2 + (4*h - 4*g + 6)*k + (2*h - 6)*H + 4*h
    d = (2*h + 6)*k**2 - 4*h*k
    sigmasq = (a*N**3 + b*N**2 + c*N + d) / ((N - 1.) * (N - 2.) * (N - 3.))
    m = k - 1
    A2 = (A2kN - m) / math.sqrt(sigmasq)

    # The b_i values are the interpolation coefficients from Table 2
    # of Scholz and Stephens 1987
    b0 = np.array([0.675, 1.281, 1.645, 1.96, 2.326, 2.573, 3.085])
    b1 = np.array([-0.245, 0.25, 0.678, 1.149, 1.822, 2.364, 3.615])
    b2 = np.array([-0.105, -0.305, -0.362, -0.391, -0.396, -0.345, -0.154])
    critical = b0 + b1 / math.sqrt(m) + b2 / m

    sig = np.array([0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001])

    if A2 < critical.min() and method is None:
        p = sig.max()
        msg = (f"p-value capped: true value larger than {p}. Consider "
               "specifying `method` "
               "(e.g. `method=stats.PermutationMethod()`.)")
        warnings.warn(msg, stacklevel=2)
    elif A2 > critical.max() and method is None:
        p = sig.min()
        msg = (f"p-value floored: true value smaller than {p}. Consider "
               "specifying `method` "
               "(e.g. `method=stats.PermutationMethod()`.)")
        warnings.warn(msg, stacklevel=2)
    elif method is None:
        # interpolation of probit of significance level
        pf = np.polyfit(critical, log(sig), 2)
        p = math.exp(np.polyval(pf, A2))
    else:
        p = res.pvalue if method is not None else p

    # create result object with alias for backward compatibility
    res = Anderson_ksampResult(A2, critical, p)
    res.significance_level = p
    return res


AnsariResult = namedtuple('AnsariResult', ('statistic', 'pvalue'))


class _ABW:
    """Distribution of Ansari-Bradley W-statistic under the null hypothesis."""
    # TODO: calculate exact distribution considering ties
    # We could avoid summing over more than half the frequencies,
    # but initially it doesn't seem worth the extra complexity

    def __init__(self):
        """Minimal initializer."""
        self.m = None
        self.n = None
        self.astart = None
        self.total = None
        self.freqs = None

    def _recalc(self, n, m):
        """When necessary, recalculate exact distribution."""
        if n != self.n or m != self.m:
            self.n, self.m = n, m
            # distribution is NOT symmetric when m + n is odd
            # n is len(x), m is len(y), and ratio of scales is defined x/y
            astart, a1, _ = gscale(n, m)
            self.astart = astart  # minimum value of statistic
            # Exact distribution of test statistic under null hypothesis
            # expressed as frequencies/counts/integers to maintain precision.
            # Stored as floats to avoid overflow of sums.
            self.freqs = a1.astype(np.float64)
            self.total = self.freqs.sum()  # could calculate from m and n
            # probability mass is self.freqs / self.total;

    def pmf(self, k, n, m):
        """Probability mass function."""
        self._recalc(n, m)
        # The convention here is that PMF at k = 12.5 is the same as at k = 12,
        # -> use `floor` in case of ties.
        ind = np.floor(k - self.astart).astype(int)
        return self.freqs[ind] / self.total

    def cdf(self, k, n, m):
        """Cumulative distribution function."""
        self._recalc(n, m)
        # Null distribution derived without considering ties is
        # approximate. Round down to avoid Type I error.
        ind = np.ceil(k - self.astart).astype(int)
        return self.freqs[:ind+1].sum() / self.total

    def sf(self, k, n, m):
        """Survival function."""
        self._recalc(n, m)
        # Null distribution derived without considering ties is
        # approximate. Round down to avoid Type I error.
        ind = np.floor(k - self.astart).astype(int)
        return self.freqs[ind:].sum() / self.total


# Maintain state for faster repeat calls to ansari w/ method='exact'
# _ABW() is calculated once per thread and stored as an attribute on
# this thread-local variable inside ansari().
_abw_state = threading.local()


@_axis_nan_policy_factory(AnsariResult, n_samples=2)
def ansari(x, y, alternative='two-sided'):
    """Perform the Ansari-Bradley test for equal scale parameters.

    The Ansari-Bradley test ([1]_, [2]_) is a non-parametric test
    for the equality of the scale parameter of the distributions
    from which two samples were drawn. The null hypothesis states that
    the ratio of the scale of the distribution underlying `x` to the scale
    of the distribution underlying `y` is 1.

    Parameters
    ----------
    x, y : array_like
        Arrays of sample data.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        The following options are available:

        * 'two-sided': the ratio of scales is not equal to 1.
        * 'less': the ratio of scales is less than 1.
        * 'greater': the ratio of scales is greater than 1.

        .. versionadded:: 1.7.0

    Returns
    -------
    statistic : float
        The Ansari-Bradley test statistic.
    pvalue : float
        The p-value of the hypothesis test.

    See Also
    --------
    fligner : A non-parametric test for the equality of k variances
    mood : A non-parametric test for the equality of two scale parameters

    Notes
    -----
    The p-value given is exact when the sample sizes are both less than
    55 and there are no ties, otherwise a normal approximation for the
    p-value is used.

    References
    ----------
    .. [1] Ansari, A. R. and Bradley, R. A. (1960) Rank-sum tests for
           dispersions, Annals of Mathematical Statistics, 31, 1174-1189.
    .. [2] Sprent, Peter and N.C. Smeeton.  Applied nonparametric
           statistical methods.  3rd ed. Chapman and Hall/CRC. 2001.
           Section 5.8.2.
    .. [3] Nathaniel E. Helwig "Nonparametric Dispersion and Equality
           Tests" at http://users.stat.umn.edu/~helwig/notes/npde-Notes.pdf

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import ansari
    >>> rng = np.random.default_rng()

    For these examples, we'll create three random data sets.  The first
    two, with sizes 35 and 25, are drawn from a normal distribution with
    mean 0 and standard deviation 2.  The third data set has size 25 and
    is drawn from a normal distribution with standard deviation 1.25.

    >>> x1 = rng.normal(loc=0, scale=2, size=35)
    >>> x2 = rng.normal(loc=0, scale=2, size=25)
    >>> x3 = rng.normal(loc=0, scale=1.25, size=25)

    First we apply `ansari` to `x1` and `x2`.  These samples are drawn
    from the same distribution, so we expect the Ansari-Bradley test
    should not lead us to conclude that the scales of the distributions
    are different.

    >>> ansari(x1, x2)
    AnsariResult(statistic=541.0, pvalue=0.9762532927399098)

    With a p-value close to 1, we cannot conclude that there is a
    significant difference in the scales (as expected).

    Now apply the test to `x1` and `x3`:

    >>> ansari(x1, x3)
    AnsariResult(statistic=425.0, pvalue=0.0003087020407974518)

    The probability of observing such an extreme value of the statistic
    under the null hypothesis of equal scales is only 0.03087%. We take this
    as evidence against the null hypothesis in favor of the alternative:
    the scales of the distributions from which the samples were drawn
    are not equal.

    We can use the `alternative` parameter to perform a one-tailed test.
    In the above example, the scale of `x1` is greater than `x3` and so
    the ratio of scales of `x1` and `x3` is greater than 1. This means
    that the p-value when ``alternative='greater'`` should be near 0 and
    hence we should be able to reject the null hypothesis:

    >>> ansari(x1, x3, alternative='greater')
    AnsariResult(statistic=425.0, pvalue=0.0001543510203987259)

    As we can see, the p-value is indeed quite low. Use of
    ``alternative='less'`` should thus yield a large p-value:

    >>> ansari(x1, x3, alternative='less')
    AnsariResult(statistic=425.0, pvalue=0.9998643258449039)

    """
    if alternative not in {'two-sided', 'greater', 'less'}:
        raise ValueError("'alternative' must be 'two-sided',"
                         " 'greater', or 'less'.")

    if not hasattr(_abw_state, 'a'):
        _abw_state.a = _ABW()

    x, y = asarray(x), asarray(y)
    n = len(x)
    m = len(y)
    if m < 1:
        raise ValueError("Not enough other observations.")
    if n < 1:
        raise ValueError("Not enough test observations.")

    N = m + n
    xy = r_[x, y]  # combine
    rank = _stats_py.rankdata(xy)
    symrank = amin(array((rank, N - rank + 1)), 0)
    AB = np.sum(symrank[:n], axis=0)
    uxy = unique(xy)
    repeats = (len(uxy) != len(xy))
    exact = ((m < 55) and (n < 55) and not repeats)
    if repeats and (m < 55 or n < 55):
        warnings.warn("Ties preclude use of exact statistic.", stacklevel=2)
    if exact:
        if alternative == 'two-sided':
            pval = 2.0 * np.minimum(_abw_state.a.cdf(AB, n, m),
                                    _abw_state.a.sf(AB, n, m))
        elif alternative == 'greater':
            # AB statistic is _smaller_ when ratio of scales is larger,
            # so this is the opposite of the usual calculation
            pval = _abw_state.a.cdf(AB, n, m)
        else:
            pval = _abw_state.a.sf(AB, n, m)
        return AnsariResult(AB, min(1.0, pval))

    # otherwise compute normal approximation
    if N % 2:  # N odd
        mnAB = n * (N+1.0)**2 / 4.0 / N
        varAB = n * m * (N+1.0) * (3+N**2) / (48.0 * N**2)
    else:
        mnAB = n * (N+2.0) / 4.0
        varAB = m * n * (N+2) * (N-2.0) / 48 / (N-1.0)
    if repeats:   # adjust variance estimates
        # compute np.sum(tj * rj**2,axis=0)
        fac = np.sum(symrank**2, axis=0)
        if N % 2:  # N odd
            varAB = m * n * (16*N*fac - (N+1)**4) / (16.0 * N**2 * (N-1))
        else:  # N even
            varAB = m * n * (16*fac - N*(N+2)**2) / (16.0 * N * (N-1))

    # Small values of AB indicate larger dispersion for the x sample.
    # Large values of AB indicate larger dispersion for the y sample.
    # This is opposite to the way we define the ratio of scales. see [1]_.
    z = (mnAB - AB) / sqrt(varAB)
    pvalue = _get_pvalue(z, _SimpleNormal(), alternative, xp=np)
    return AnsariResult(AB[()], pvalue[()])


BartlettResult = namedtuple('BartlettResult', ('statistic', 'pvalue'))


@_axis_nan_policy_factory(BartlettResult, n_samples=None)
def bartlett(*samples, axis=0):
    r"""Perform Bartlett's test for equal variances.

    Bartlett's test tests the null hypothesis that all input samples
    are from populations with equal variances.  For samples
    from significantly non-normal populations, Levene's test
    `levene` is more robust.

    Parameters
    ----------
    sample1, sample2, ... : array_like
        arrays of sample data.  Only 1d arrays are accepted, they may have
        different lengths.

    Returns
    -------
    statistic : float
        The test statistic.
    pvalue : float
        The p-value of the test.

    See Also
    --------
    fligner : A non-parametric test for the equality of k variances
    levene : A robust parametric test for equality of k variances
    :ref:`hypothesis_bartlett` : Extended example

    Notes
    -----
    Conover et al. (1981) examine many of the existing parametric and
    nonparametric tests by extensive simulations and they conclude that the
    tests proposed by Fligner and Killeen (1976) and Levene (1960) appear to be
    superior in terms of robustness of departures from normality and power
    ([3]_).

    References
    ----------
    .. [1]  https://www.itl.nist.gov/div898/handbook/eda/section3/eda357.htm
    .. [2]  Snedecor, George W. and Cochran, William G. (1989), Statistical
              Methods, Eighth Edition, Iowa State University Press.
    .. [3] Park, C. and Lindsay, B. G. (1999). Robust Scale Estimation and
           Hypothesis Testing based on Quadratic Inference Function. Technical
           Report #99-03, Center for Likelihood Studies, Pennsylvania State
           University.
    .. [4] Bartlett, M. S. (1937). Properties of Sufficiency and Statistical
           Tests. Proceedings of the Royal Society of London. Series A,
           Mathematical and Physical Sciences, Vol. 160, No.901, pp. 268-282.

    Examples
    --------

    Test whether the lists `a`, `b` and `c` come from populations
    with equal variances.

    >>> import numpy as np
    >>> from scipy import stats
    >>> a = [8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99]
    >>> b = [8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05]
    >>> c = [8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98]
    >>> stat, p = stats.bartlett(a, b, c)
    >>> p
    1.1254782518834628e-05

    The very small p-value suggests that the populations do not have equal
    variances.

    This is not surprising, given that the sample variance of `b` is much
    larger than that of `a` and `c`:

    >>> [np.var(x, ddof=1) for x in [a, b, c]]
    [0.007054444444444413, 0.13073888888888888, 0.008890000000000002]

    For a more detailed example, see :ref:`hypothesis_bartlett`.
    """
    xp = array_namespace(*samples)

    k = len(samples)
    if k < 2:
        raise ValueError("Must enter at least two input sample vectors.")

    samples = _broadcast_arrays(samples, axis=axis, xp=xp)
    samples = [xp_moveaxis_to_end(sample, axis, xp=xp) for sample in samples]

    Ni = [xp.asarray(sample.shape[-1], dtype=sample.dtype) for sample in samples]
    Ni = [xp.broadcast_to(N, samples[0].shape[:-1]) for N in Ni]
    ssq = [xp.var(sample, correction=1, axis=-1) for sample in samples]
    Ni = [arr[xp.newaxis, ...] for arr in Ni]
    ssq = [arr[xp.newaxis, ...] for arr in ssq]
    Ni = xp.concat(Ni, axis=0)
    ssq = xp.concat(ssq, axis=0)
    # sum dtype can be removed when 2023.12 rules kick in
    dtype = Ni.dtype
    Ntot = xp.sum(Ni, axis=0, dtype=dtype)
    spsq = xp.sum((Ni - 1)*ssq, axis=0, dtype=dtype) / (Ntot - k)
    numer = ((Ntot - k) * xp.log(spsq)
             - xp.sum((Ni - 1)*xp.log(ssq), axis=0, dtype=dtype))
    denom = (1 + 1/(3*(k - 1))
             * ((xp.sum(1/(Ni - 1), axis=0, dtype=dtype)) - 1/(Ntot - k)))
    T = numer / denom

    chi2 = _SimpleChi2(xp.asarray(k-1))
    pvalue = _get_pvalue(T, chi2, alternative='greater', symmetric=False, xp=xp)

    T = xp.clip(T, min=0., max=xp.inf)
    T = T[()] if T.ndim == 0 else T
    pvalue = pvalue[()] if pvalue.ndim == 0 else pvalue

    return BartlettResult(T, pvalue)


LeveneResult = namedtuple('LeveneResult', ('statistic', 'pvalue'))


@_axis_nan_policy_factory(LeveneResult, n_samples=None)
def levene(*samples, center='median', proportiontocut=0.05):
    r"""Perform Levene test for equal variances.

    The Levene test tests the null hypothesis that all input samples
    are from populations with equal variances.  Levene's test is an
    alternative to Bartlett's test `bartlett` in the case where
    there are significant deviations from normality.

    Parameters
    ----------
    sample1, sample2, ... : array_like
        The sample data, possibly with different lengths. Only one-dimensional
        samples are accepted.
    center : {'mean', 'median', 'trimmed'}, optional
        Which function of the data to use in the test.  The default
        is 'median'.
    proportiontocut : float, optional
        When `center` is 'trimmed', this gives the proportion of data points
        to cut from each end. (See `scipy.stats.trim_mean`.)
        Default is 0.05.

    Returns
    -------
    statistic : float
        The test statistic.
    pvalue : float
        The p-value for the test.

    See Also
    --------
    fligner : A non-parametric test for the equality of k variances
    bartlett : A parametric test for equality of k variances in normal samples
    :ref:`hypothesis_levene` : Extended example

    Notes
    -----
    Three variations of Levene's test are possible.  The possibilities
    and their recommended usages are:

    * 'median' : Recommended for skewed (non-normal) distributions>
    * 'mean' : Recommended for symmetric, moderate-tailed distributions.
    * 'trimmed' : Recommended for heavy-tailed distributions.

    The test version using the mean was proposed in the original article
    of Levene ([2]_) while the median and trimmed mean have been studied by
    Brown and Forsythe ([3]_), sometimes also referred to as Brown-Forsythe
    test.

    References
    ----------
    .. [1] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35a.htm
    .. [2] Levene, H. (1960). In Contributions to Probability and Statistics:
           Essays in Honor of Harold Hotelling, I. Olkin et al. eds.,
           Stanford University Press, pp. 278-292.
    .. [3] Brown, M. B. and Forsythe, A. B. (1974), Journal of the American
           Statistical Association, 69, 364-367

    Examples
    --------

    Test whether the lists `a`, `b` and `c` come from populations
    with equal variances.

    >>> import numpy as np
    >>> from scipy import stats
    >>> a = [8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99]
    >>> b = [8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05]
    >>> c = [8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98]
    >>> stat, p = stats.levene(a, b, c)
    >>> p
    0.002431505967249681

    The small p-value suggests that the populations do not have equal
    variances.

    This is not surprising, given that the sample variance of `b` is much
    larger than that of `a` and `c`:

    >>> [np.var(x, ddof=1) for x in [a, b, c]]
    [0.007054444444444413, 0.13073888888888888, 0.008890000000000002]

    For a more detailed example, see :ref:`hypothesis_levene`.
    """
    if center not in ['mean', 'median', 'trimmed']:
        raise ValueError("center must be 'mean', 'median' or 'trimmed'.")

    k = len(samples)
    if k < 2:
        raise ValueError("Must enter at least two input sample vectors.")

    Ni = np.empty(k)
    Yci = np.empty(k, 'd')

    if center == 'median':

        def func(x):
            return np.median(x, axis=0)

    elif center == 'mean':

        def func(x):
            return np.mean(x, axis=0)

    else:  # center == 'trimmed'
        samples = tuple(_stats_py.trimboth(np.sort(sample), proportiontocut)
                        for sample in samples)

        def func(x):
            return np.mean(x, axis=0)

    for j in range(k):
        Ni[j] = len(samples[j])
        Yci[j] = func(samples[j])
    Ntot = np.sum(Ni, axis=0)

    # compute Zij's
    Zij = [None] * k
    for i in range(k):
        Zij[i] = abs(asarray(samples[i]) - Yci[i])

    # compute Zbari
    Zbari = np.empty(k, 'd')
    Zbar = 0.0
    for i in range(k):
        Zbari[i] = np.mean(Zij[i], axis=0)
        Zbar += Zbari[i] * Ni[i]

    Zbar /= Ntot
    numer = (Ntot - k) * np.sum(Ni * (Zbari - Zbar)**2, axis=0)

    # compute denom_variance
    dvar = 0.0
    for i in range(k):
        dvar += np.sum((Zij[i] - Zbari[i])**2, axis=0)

    denom = (k - 1.0) * dvar

    W = numer / denom
    pval = distributions.f.sf(W, k-1, Ntot-k)  # 1 - cdf
    return LeveneResult(W, pval)


def _apply_func(x, g, func):
    # g is list of indices into x
    #  separating x into different groups
    #  func should be applied over the groups
    g = unique(r_[0, g, len(x)])
    output = [func(x[g[k]:g[k+1]]) for k in range(len(g) - 1)]

    return asarray(output)


FlignerResult = namedtuple('FlignerResult', ('statistic', 'pvalue'))


@_axis_nan_policy_factory(FlignerResult, n_samples=None)
def fligner(*samples, center='median', proportiontocut=0.05):
    r"""Perform Fligner-Killeen test for equality of variance.

    Fligner's test tests the null hypothesis that all input samples
    are from populations with equal variances.  Fligner-Killeen's test is
    distribution free when populations are identical [2]_.

    Parameters
    ----------
    sample1, sample2, ... : array_like
        Arrays of sample data.  Need not be the same length.
    center : {'mean', 'median', 'trimmed'}, optional
        Keyword argument controlling which function of the data is used in
        computing the test statistic.  The default is 'median'.
    proportiontocut : float, optional
        When `center` is 'trimmed', this gives the proportion of data points
        to cut from each end. (See `scipy.stats.trim_mean`.)
        Default is 0.05.

    Returns
    -------
    statistic : float
        The test statistic.
    pvalue : float
        The p-value for the hypothesis test.

    See Also
    --------
    bartlett : A parametric test for equality of k variances in normal samples
    levene : A robust parametric test for equality of k variances
    :ref:`hypothesis_fligner` : Extended example

    Notes
    -----
    As with Levene's test there are three variants of Fligner's test that
    differ by the measure of central tendency used in the test.  See `levene`
    for more information.

    Conover et al. (1981) examine many of the existing parametric and
    nonparametric tests by extensive simulations and they conclude that the
    tests proposed by Fligner and Killeen (1976) and Levene (1960) appear to be
    superior in terms of robustness of departures from normality and power
    [3]_.

    References
    ----------
    .. [1] Park, C. and Lindsay, B. G. (1999). Robust Scale Estimation and
           Hypothesis Testing based on Quadratic Inference Function. Technical
           Report #99-03, Center for Likelihood Studies, Pennsylvania State
           University.
           https://cecas.clemson.edu/~cspark/cv/paper/qif/draftqif2.pdf
    .. [2] Fligner, M.A. and Killeen, T.J. (1976). Distribution-free two-sample
           tests for scale. Journal of the American Statistical Association.
           71(353), 210-213.
    .. [3] Park, C. and Lindsay, B. G. (1999). Robust Scale Estimation and
           Hypothesis Testing based on Quadratic Inference Function. Technical
           Report #99-03, Center for Likelihood Studies, Pennsylvania State
           University.
    .. [4] Conover, W. J., Johnson, M. E. and Johnson M. M. (1981). A
           comparative study of tests for homogeneity of variances, with
           applications to the outer continental shelf bidding data.
           Technometrics, 23(4), 351-361.

    Examples
    --------

    >>> import numpy as np
    >>> from scipy import stats

    Test whether the lists `a`, `b` and `c` come from populations
    with equal variances.

    >>> a = [8.88, 9.12, 9.04, 8.98, 9.00, 9.08, 9.01, 8.85, 9.06, 8.99]
    >>> b = [8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05]
    >>> c = [8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98]
    >>> stat, p = stats.fligner(a, b, c)
    >>> p
    0.00450826080004775

    The small p-value suggests that the populations do not have equal
    variances.

    This is not surprising, given that the sample variance of `b` is much
    larger than that of `a` and `c`:

    >>> [np.var(x, ddof=1) for x in [a, b, c]]
    [0.007054444444444413, 0.13073888888888888, 0.008890000000000002]

    For a more detailed example, see :ref:`hypothesis_fligner`.
    """
    if center not in ['mean', 'median', 'trimmed']:
        raise ValueError("center must be 'mean', 'median' or 'trimmed'.")

    k = len(samples)
    if k < 2:
        raise ValueError("Must enter at least two input sample vectors.")

    # Handle empty input
    for sample in samples:
        if sample.size == 0:
            NaN = _get_nan(*samples)
            return FlignerResult(NaN, NaN)

    if center == 'median':

        def func(x):
            return np.median(x, axis=0)

    elif center == 'mean':

        def func(x):
            return np.mean(x, axis=0)

    else:  # center == 'trimmed'
        samples = tuple(_stats_py.trimboth(sample, proportiontocut)
                        for sample in samples)

        def func(x):
            return np.mean(x, axis=0)

    Ni = asarray([len(samples[j]) for j in range(k)])
    Yci = asarray([func(samples[j]) for j in range(k)])
    Ntot = np.sum(Ni, axis=0)
    # compute Zij's
    Zij = [abs(asarray(samples[i]) - Yci[i]) for i in range(k)]
    allZij = []
    g = [0]
    for i in range(k):
        allZij.extend(list(Zij[i]))
        g.append(len(allZij))

    ranks = _stats_py.rankdata(allZij)
    sample = distributions.norm.ppf(ranks / (2*(Ntot + 1.0)) + 0.5)

    # compute Aibar
    Aibar = _apply_func(sample, g, np.sum) / Ni
    anbar = np.mean(sample, axis=0)
    varsq = np.var(sample, axis=0, ddof=1)
    statistic = np.sum(Ni * (asarray(Aibar) - anbar)**2.0, axis=0) / varsq
    chi2 = _SimpleChi2(k-1)
    pval = _get_pvalue(statistic, chi2, alternative='greater', symmetric=False, xp=np)
    return FlignerResult(statistic, pval)


@_axis_nan_policy_factory(lambda x1: (x1,), n_samples=4, n_outputs=1)
def _mood_inner_lc(xy, x, diffs, sorted_xy, n, m, N) -> float:
    # Obtain the unique values and their frequencies from the pooled samples.
    # "a_j, + b_j, = t_j, for j = 1, ... k" where `k` is the number of unique
    # classes, and "[t]he number of values associated with the x's and y's in
    # the jth class will be denoted by a_j, and b_j respectively."
    # (Mielke, 312)
    # Reuse previously computed sorted array and `diff` arrays to obtain the
    # unique values and counts. Prepend `diffs` with a non-zero to indicate
    # that the first element should be marked as not matching what preceded it.
    diffs_prep = np.concatenate(([1], diffs))
    # Unique elements are where the was a difference between elements in the
    # sorted array
    uniques = sorted_xy[diffs_prep != 0]
    # The count of each element is the bin size for each set of consecutive
    # differences where the difference is zero. Replace nonzero differences
    # with 1 and then use the cumulative sum to count the indices.
    t = np.bincount(np.cumsum(np.asarray(diffs_prep != 0, dtype=int)))[1:]
    k = len(uniques)
    js = np.arange(1, k + 1, dtype=int)
    # the `b` array mentioned in the paper is not used, outside of the
    # calculation of `t`, so we do not need to calculate it separately. Here
    # we calculate `a`. In plain language, `a[j]` is the number of values in
    # `x` that equal `uniques[j]`.
    sorted_xyx = np.sort(np.concatenate((xy, x)))
    diffs = np.diff(sorted_xyx)
    diffs_prep = np.concatenate(([1], diffs))
    diff_is_zero = np.asarray(diffs_prep != 0, dtype=int)
    xyx_counts = np.bincount(np.cumsum(diff_is_zero))[1:]
    a = xyx_counts - t
    # "Define .. a_0 = b_0 = t_0 = S_0 = 0" (Mielke 312) so we shift  `a`
    # and `t` arrays over 1 to allow a first element of 0 to accommodate this
    # indexing.
    t = np.concatenate(([0], t))
    a = np.concatenate(([0], a))
    # S is built from `t`, so it does not need a preceding zero added on.
    S = np.cumsum(t)
    # define a copy of `S` with a prepending zero for later use to avoid
    # the need for indexing.
    S_i_m1 = np.concatenate(([0], S[:-1]))

    # Psi, as defined by the 6th unnumbered equation on page 313 (Mielke).
    # Note that in the paper there is an error where the denominator `2` is
    # squared when it should be the entire equation.
    def psi(indicator):
        return (indicator - (N + 1)/2)**2

    # define summation range for use in calculation of phi, as seen in sum
    # in the unnumbered equation on the bottom of page 312 (Mielke).
    s_lower = S[js - 1] + 1
    s_upper = S[js] + 1
    phi_J = [np.arange(s_lower[idx], s_upper[idx]) for idx in range(k)]

    # for every range in the above array, determine the sum of psi(I) for
    # every element in the range. Divide all the sums by `t`. Following the
    # last unnumbered equation on page 312.
    phis = [np.sum(psi(I_j)) for I_j in phi_J] / t[js]

    # `T` is equal to a[j] * phi[j], per the first unnumbered equation on
    # page 312. `phis` is already in the order based on `js`, so we index
    # into `a` with `js` as well.
    T = sum(phis * a[js])

    # The approximate statistic
    E_0_T = n * (N * N - 1) / 12

    varM = (m * n * (N + 1.0) * (N ** 2 - 4) / 180 -
            m * n / (180 * N * (N - 1)) * np.sum(
                t * (t**2 - 1) * (t**2 - 4 + (15 * (N - S - S_i_m1) ** 2))
            ))

    return ((T - E_0_T) / np.sqrt(varM),)


def _mood_too_small(samples, kwargs, axis=-1):
    x, y = samples
    n = x.shape[axis]
    m = y.shape[axis]
    N = m + n
    return N < 3


@_axis_nan_policy_factory(SignificanceResult, n_samples=2, too_small=_mood_too_small)
def mood(x, y, axis=0, alternative="two-sided"):
    """Perform Mood's test for equal scale parameters.

    Mood's two-sample test for scale parameters is a non-parametric
    test for the null hypothesis that two samples are drawn from the
    same distribution with the same scale parameter.

    Parameters
    ----------
    x, y : array_like
        Arrays of sample data. There must be at least three observations
        total.
    axis : int, optional
        The axis along which the samples are tested.  `x` and `y` can be of
        different length along `axis`.
        If `axis` is None, `x` and `y` are flattened and the test is done on
        all values in the flattened arrays.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        The following options are available:

        * 'two-sided': the scales of the distributions underlying `x` and `y`
          are different.
        * 'less': the scale of the distribution underlying `x` is less than
          the scale of the distribution underlying `y`.
        * 'greater': the scale of the distribution underlying `x` is greater
          than the scale of the distribution underlying `y`.

        .. versionadded:: 1.7.0

    Returns
    -------
    res : SignificanceResult
        An object containing attributes:

        statistic : scalar or ndarray
            The z-score for the hypothesis test.  For 1-D inputs a scalar is
            returned.
        pvalue : scalar ndarray
            The p-value for the hypothesis test.

    See Also
    --------
    fligner : A non-parametric test for the equality of k variances
    ansari : A non-parametric test for the equality of 2 variances
    bartlett : A parametric test for equality of k variances in normal samples
    levene : A parametric test for equality of k variances

    Notes
    -----
    The data are assumed to be drawn from probability distributions ``f(x)``
    and ``f(x/s) / s`` respectively, for some probability density function f.
    The null hypothesis is that ``s == 1``.

    For multi-dimensional arrays, if the inputs are of shapes
    ``(n0, n1, n2, n3)``  and ``(n0, m1, n2, n3)``, then if ``axis=1``, the
    resulting z and p values will have shape ``(n0, n2, n3)``.  Note that
    ``n1`` and ``m1`` don't have to be equal, but the other dimensions do.

    References
    ----------
    [1] Mielke, Paul W. "Note on Some Squared Rank Tests with Existing Ties."
        Technometrics, vol. 9, no. 2, 1967, pp. 312-14. JSTOR,
        https://doi.org/10.2307/1266427. Accessed 18 May 2022.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> rng = np.random.default_rng()
    >>> x2 = rng.standard_normal((2, 45, 6, 7))
    >>> x1 = rng.standard_normal((2, 30, 6, 7))
    >>> res = stats.mood(x1, x2, axis=1)
    >>> res.pvalue.shape
    (2, 6, 7)

    Find the number of points where the difference in scale is not significant:

    >>> (res.pvalue > 0.1).sum()
    78

    Perform the test with different scales:

    >>> x1 = rng.standard_normal((2, 30))
    >>> x2 = rng.standard_normal((2, 35)) * 10.0
    >>> stats.mood(x1, x2, axis=1)
    SignificanceResult(statistic=array([-5.76174136, -6.12650783]),
                       pvalue=array([8.32505043e-09, 8.98287869e-10]))

    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if axis < 0:
        axis = x.ndim + axis

    # Determine shape of the result arrays
    res_shape = tuple([x.shape[ax] for ax in range(len(x.shape)) if ax != axis])
    if not (res_shape == tuple([y.shape[ax] for ax in range(len(y.shape)) if
                                ax != axis])):
        raise ValueError("Dimensions of x and y on all axes except `axis` "
                         "should match")

    n = x.shape[axis]
    m = y.shape[axis]
    N = m + n
    if N < 3:
        raise ValueError("Not enough observations.")

    xy = np.concatenate((x, y), axis=axis)
    # determine if any of the samples contain ties
    sorted_xy = np.sort(xy, axis=axis)
    diffs = np.diff(sorted_xy, axis=axis)
    if 0 in diffs:
        z = np.asarray(_mood_inner_lc(xy, x, diffs, sorted_xy, n, m, N,
                                      axis=axis))
    else:
        if axis != 0:
            xy = np.moveaxis(xy, axis, 0)

        xy = xy.reshape(xy.shape[0], -1)
        # Generalized to the n-dimensional case by adding the axis argument,
        # and using for loops, since rankdata is not vectorized.  For improving
        # performance consider vectorizing rankdata function.
        all_ranks = np.empty_like(xy)
        for j in range(xy.shape[1]):
            all_ranks[:, j] = _stats_py.rankdata(xy[:, j])

        Ri = all_ranks[:n]
        M = np.sum((Ri - (N + 1.0) / 2) ** 2, axis=0)
        # Approx stat.
        mnM = n * (N * N - 1.0) / 12
        varM = m * n * (N + 1.0) * (N + 2) * (N - 2) / 180
        z = (M - mnM) / sqrt(varM)
    pval = _get_pvalue(z, _SimpleNormal(), alternative, xp=np)

    if res_shape == ():
        # Return scalars, not 0-D arrays
        z = z[0]
        pval = pval[0]
    else:
        z.shape = res_shape
        pval.shape = res_shape
    return SignificanceResult(z[()], pval[()])


WilcoxonResult = _make_tuple_bunch('WilcoxonResult', ['statistic', 'pvalue'])


def wilcoxon_result_unpacker(res):
    if hasattr(res, 'zstatistic'):
        return res.statistic, res.pvalue, res.zstatistic
    else:
        return res.statistic, res.pvalue


def wilcoxon_result_object(statistic, pvalue, zstatistic=None):
    res = WilcoxonResult(statistic, pvalue)
    if zstatistic is not None:
        res.zstatistic = zstatistic
    return res


def wilcoxon_outputs(kwds):
    method = kwds.get('method', 'auto')
    if method == 'asymptotic':
        return 3
    return 2


@_rename_parameter("mode", "method")
@_axis_nan_policy_factory(
    wilcoxon_result_object, paired=True,
    n_samples=lambda kwds: 2 if kwds.get('y', None) is not None else 1,
    result_to_tuple=wilcoxon_result_unpacker, n_outputs=wilcoxon_outputs,
)
def wilcoxon(x, y=None, zero_method="wilcox", correction=False,
             alternative="two-sided", method='auto', *, axis=0):
    """Calculate the Wilcoxon signed-rank test.

    The Wilcoxon signed-rank test tests the null hypothesis that two
    related paired samples come from the same distribution. In particular,
    it tests whether the distribution of the differences ``x - y`` is symmetric
    about zero. It is a non-parametric version of the paired T-test.

    Parameters
    ----------
    x : array_like
        Either the first set of measurements (in which case ``y`` is the second
        set of measurements), or the differences between two sets of
        measurements (in which case ``y`` is not to be specified.)  Must be
        one-dimensional.
    y : array_like, optional
        Either the second set of measurements (if ``x`` is the first set of
        measurements), or not specified (if ``x`` is the differences between
        two sets of measurements.)  Must be one-dimensional.

        .. warning::
            When `y` is provided, `wilcoxon` calculates the test statistic
            based on the ranks of the absolute values of ``d = x - y``.
            Roundoff error in the subtraction can result in elements of ``d``
            being assigned different ranks even when they would be tied with
            exact arithmetic. Rather than passing `x` and `y` separately,
            consider computing the difference ``x - y``, rounding as needed to
            ensure that only truly unique elements are numerically distinct,
            and passing the result as `x`, leaving `y` at the default (None).

    zero_method : {"wilcox", "pratt", "zsplit"}, optional
        There are different conventions for handling pairs of observations
        with equal values ("zero-differences", or "zeros").

        * "wilcox": Discards all zero-differences (default); see [4]_.
        * "pratt": Includes zero-differences in the ranking process,
          but drops the ranks of the zeros (more conservative); see [3]_.
          In this case, the normal approximation is adjusted as in [5]_.
        * "zsplit": Includes zero-differences in the ranking process and
          splits the zero rank between positive and negative ones.

    correction : bool, optional
        If True, apply continuity correction by adjusting the Wilcoxon rank
        statistic by 0.5 towards the mean value when computing the
        z-statistic if a normal approximation is used.  Default is False.
    alternative : {"two-sided", "greater", "less"}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        In the following, let ``d`` represent the difference between the paired
        samples: ``d = x - y`` if both ``x`` and ``y`` are provided, or
        ``d = x`` otherwise.

        * 'two-sided': the distribution underlying ``d`` is not symmetric
          about zero.
        * 'less': the distribution underlying ``d`` is stochastically less
          than a distribution symmetric about zero.
        * 'greater': the distribution underlying ``d`` is stochastically
          greater than a distribution symmetric about zero.

    method : {"auto", "exact", "asymptotic"} or `PermutationMethod` instance, optional
        Method to calculate the p-value, see Notes. Default is "auto".

    axis : int or None, default: 0
        If an int, the axis of the input along which to compute the statistic.
        The statistic of each axis-slice (e.g. row) of the input will appear
        in a corresponding element of the output. If ``None``, the input will
        be raveled before computing the statistic.

    Returns
    -------
    An object with the following attributes.

    statistic : array_like
        If `alternative` is "two-sided", the sum of the ranks of the
        differences above or below zero, whichever is smaller.
        Otherwise the sum of the ranks of the differences above zero.
    pvalue : array_like
        The p-value for the test depending on `alternative` and `method`.
    zstatistic : array_like
        When ``method = 'asymptotic'``, this is the normalized z-statistic::

            z = (T - mn - d) / se

        where ``T`` is `statistic` as defined above, ``mn`` is the mean of the
        distribution under the null hypothesis, ``d`` is a continuity
        correction, and ``se`` is the standard error.
        When ``method != 'asymptotic'``, this attribute is not available.

    See Also
    --------
    kruskal, mannwhitneyu

    Notes
    -----
    In the following, let ``d`` represent the difference between the paired
    samples: ``d = x - y`` if both ``x`` and ``y`` are provided, or ``d = x``
    otherwise. Assume that all elements of ``d`` are independent and
    identically distributed observations, and all are distinct and nonzero.

    - When ``len(d)`` is sufficiently large, the null distribution of the
      normalized test statistic (`zstatistic` above) is approximately normal,
      and ``method = 'asymptotic'`` can be used to compute the p-value.

    - When ``len(d)`` is small, the normal approximation may not be accurate,
      and ``method='exact'`` is preferred (at the cost of additional
      execution time).

    - The default, ``method='auto'``, selects between the two:
      ``method='exact'`` is used when ``len(d) <= 50``, and
      ``method='asymptotic'`` is used otherwise.

    The presence of "ties" (i.e. not all elements of ``d`` are unique) or
    "zeros" (i.e. elements of ``d`` are zero) changes the null distribution
    of the test statistic, and ``method='exact'`` no longer calculates
    the exact p-value. If ``method='asymptotic'``, the z-statistic is adjusted
    for more accurate comparison against the standard normal, but still,
    for finite sample sizes, the standard normal is only an approximation of
    the true null distribution of the z-statistic. For such situations, the
    `method` parameter also accepts instances of `PermutationMethod`. In this
    case, the p-value is computed using `permutation_test` with the provided
    configuration options and other appropriate settings.

    The presence of ties and zeros affects the resolution of ``method='auto'``
    accordingly: exhasutive permutations are performed when ``len(d) <= 13``,
    and the asymptotic method is used otherwise. Note that they asymptotic
    method may not be very accurate even for ``len(d) > 14``; the threshold
    was chosen as a compromise between execution time and accuracy under the
    constraint that the results must be deterministic. Consider providing an
    instance of `PermutationMethod` method manually, choosing the
    ``n_resamples`` parameter to balance time constraints and accuracy
    requirements.

    Please also note that in the edge case that all elements of ``d`` are zero,
    the p-value relying on the normal approximaton cannot be computed (NaN)
    if ``zero_method='wilcox'`` or ``zero_method='pratt'``.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
    .. [2] Conover, W.J., Practical Nonparametric Statistics, 1971.
    .. [3] Pratt, J.W., Remarks on Zeros and Ties in the Wilcoxon Signed
       Rank Procedures, Journal of the American Statistical Association,
       Vol. 54, 1959, pp. 655-667. :doi:`10.1080/01621459.1959.10501526`
    .. [4] Wilcoxon, F., Individual Comparisons by Ranking Methods,
       Biometrics Bulletin, Vol. 1, 1945, pp. 80-83. :doi:`10.2307/3001968`
    .. [5] Cureton, E.E., The Normal Approximation to the Signed-Rank
       Sampling Distribution When Zero Differences are Present,
       Journal of the American Statistical Association, Vol. 62, 1967,
       pp. 1068-1069. :doi:`10.1080/01621459.1967.10500917`

    Examples
    --------
    In [4]_, the differences in height between cross- and self-fertilized
    corn plants is given as follows:

    >>> d = [6, 8, 14, 16, 23, 24, 28, 29, 41, -48, 49, 56, 60, -67, 75]

    Cross-fertilized plants appear to be higher. To test the null
    hypothesis that there is no height difference, we can apply the
    two-sided test:

    >>> from scipy.stats import wilcoxon
    >>> res = wilcoxon(d)
    >>> res.statistic, res.pvalue
    (24.0, 0.041259765625)

    Hence, we would reject the null hypothesis at a confidence level of 5%,
    concluding that there is a difference in height between the groups.
    To confirm that the median of the differences can be assumed to be
    positive, we use:

    >>> res = wilcoxon(d, alternative='greater')
    >>> res.statistic, res.pvalue
    (96.0, 0.0206298828125)

    This shows that the null hypothesis that the median is negative can be
    rejected at a confidence level of 5% in favor of the alternative that
    the median is greater than zero. The p-values above are exact. Using the
    normal approximation gives very similar values:

    >>> res = wilcoxon(d, method='asymptotic')
    >>> res.statistic, res.pvalue
    (24.0, 0.04088813291185591)

    Note that the statistic changed to 96 in the one-sided case (the sum
    of ranks of positive differences) whereas it is 24 in the two-sided
    case (the minimum of sum of ranks above and below zero).

    In the example above, the differences in height between paired plants are
    provided to `wilcoxon` directly. Alternatively, `wilcoxon` accepts two
    samples of equal length, calculates the differences between paired
    elements, then performs the test. Consider the samples ``x`` and ``y``:

    >>> import numpy as np
    >>> x = np.array([0.5, 0.825, 0.375, 0.5])
    >>> y = np.array([0.525, 0.775, 0.325, 0.55])
    >>> res = wilcoxon(x, y, alternative='greater')
    >>> res
    WilcoxonResult(statistic=5.0, pvalue=0.5625)

    Note that had we calculated the differences by hand, the test would have
    produced different results:

    >>> d = [-0.025, 0.05, 0.05, -0.05]
    >>> ref = wilcoxon(d, alternative='greater')
    >>> ref
    WilcoxonResult(statistic=6.0, pvalue=0.5)

    The substantial difference is due to roundoff error in the results of
    ``x-y``:

    >>> d - (x-y)
    array([2.08166817e-17, 6.93889390e-17, 1.38777878e-17, 4.16333634e-17])

    Even though we expected all the elements of ``(x-y)[1:]`` to have the same
    magnitude ``0.05``, they have slightly different magnitudes in practice,
    and therefore are assigned different ranks in the test. Before performing
    the test, consider calculating ``d`` and adjusting it as necessary to
    ensure that theoretically identically values are not numerically distinct.
    For example:

    >>> d2 = np.around(x - y, decimals=3)
    >>> wilcoxon(d2, alternative='greater')
    WilcoxonResult(statistic=6.0, pvalue=0.5)

    """
    # replace approx by asymptotic to ensure backwards compatability
    if method == "approx":
        method = "asymptotic"
    return _wilcoxon._wilcoxon_nd(x, y, zero_method, correction, alternative,
                                  method, axis)


MedianTestResult = _make_tuple_bunch(
    'MedianTestResult',
    ['statistic', 'pvalue', 'median', 'table'], []
)


def median_test(*samples, ties='below', correction=True, lambda_=1,
                nan_policy='propagate'):
    """Perform a Mood's median test.

    Test that two or more samples come from populations with the same median.

    Let ``n = len(samples)`` be the number of samples.  The "grand median" of
    all the data is computed, and a contingency table is formed by
    classifying the values in each sample as being above or below the grand
    median.  The contingency table, along with `correction` and `lambda_`,
    are passed to `scipy.stats.chi2_contingency` to compute the test statistic
    and p-value.

    Parameters
    ----------
    sample1, sample2, ... : array_like
        The set of samples.  There must be at least two samples.
        Each sample must be a one-dimensional sequence containing at least
        one value.  The samples are not required to have the same length.
    ties : str, optional
        Determines how values equal to the grand median are classified in
        the contingency table.  The string must be one of::

            "below":
                Values equal to the grand median are counted as "below".
            "above":
                Values equal to the grand median are counted as "above".
            "ignore":
                Values equal to the grand median are not counted.

        The default is "below".
    correction : bool, optional
        If True, *and* there are just two samples, apply Yates' correction
        for continuity when computing the test statistic associated with
        the contingency table.  Default is True.
    lambda_ : float or str, optional
        By default, the statistic computed in this test is Pearson's
        chi-squared statistic.  `lambda_` allows a statistic from the
        Cressie-Read power divergence family to be used instead.  See
        `power_divergence` for details.
        Default is 1 (Pearson's chi-squared statistic).
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan. 'propagate' returns nan,
        'raise' throws an error, 'omit' performs the calculations ignoring nan
        values. Default is 'propagate'.

    Returns
    -------
    res : MedianTestResult
        An object containing attributes:

        statistic : float
            The test statistic.  The statistic that is returned is determined
            by `lambda_`.  The default is Pearson's chi-squared statistic.
        pvalue : float
            The p-value of the test.
        median : float
            The grand median.
        table : ndarray
            The contingency table.  The shape of the table is (2, n), where
            n is the number of samples.  The first row holds the counts of the
            values above the grand median, and the second row holds the counts
            of the values below the grand median.  The table allows further
            analysis with, for example, `scipy.stats.chi2_contingency`, or with
            `scipy.stats.fisher_exact` if there are two samples, without having
            to recompute the table.  If ``nan_policy`` is "propagate" and there
            are nans in the input, the return value for ``table`` is ``None``.

    See Also
    --------
    kruskal : Compute the Kruskal-Wallis H-test for independent samples.
    mannwhitneyu : Computes the Mann-Whitney rank test on samples x and y.

    Notes
    -----
    .. versionadded:: 0.15.0

    References
    ----------
    .. [1] Mood, A. M., Introduction to the Theory of Statistics. McGraw-Hill
        (1950), pp. 394-399.
    .. [2] Zar, J. H., Biostatistical Analysis, 5th ed. Prentice Hall (2010).
        See Sections 8.12 and 10.15.

    Examples
    --------
    A biologist runs an experiment in which there are three groups of plants.
    Group 1 has 16 plants, group 2 has 15 plants, and group 3 has 17 plants.
    Each plant produces a number of seeds.  The seed counts for each group
    are::

        Group 1: 10 14 14 18 20 22 24 25 31 31 32 39 43 43 48 49
        Group 2: 28 30 31 33 34 35 36 40 44 55 57 61 91 92 99
        Group 3:  0  3  9 22 23 25 25 33 34 34 40 45 46 48 62 67 84

    The following code applies Mood's median test to these samples.

    >>> g1 = [10, 14, 14, 18, 20, 22, 24, 25, 31, 31, 32, 39, 43, 43, 48, 49]
    >>> g2 = [28, 30, 31, 33, 34, 35, 36, 40, 44, 55, 57, 61, 91, 92, 99]
    >>> g3 = [0, 3, 9, 22, 23, 25, 25, 33, 34, 34, 40, 45, 46, 48, 62, 67, 84]
    >>> from scipy.stats import median_test
    >>> res = median_test(g1, g2, g3)

    The median is

    >>> res.median
    34.0

    and the contingency table is

    >>> res.table
    array([[ 5, 10,  7],
           [11,  5, 10]])

    `p` is too large to conclude that the medians are not the same:

    >>> res.pvalue
    0.12609082774093244

    The "G-test" can be performed by passing ``lambda_="log-likelihood"`` to
    `median_test`.

    >>> res = median_test(g1, g2, g3, lambda_="log-likelihood")
    >>> res.pvalue
    0.12224779737117837

    The median occurs several times in the data, so we'll get a different
    result if, for example, ``ties="above"`` is used:

    >>> res = median_test(g1, g2, g3, ties="above")
    >>> res.pvalue
    0.063873276069553273

    >>> res.table
    array([[ 5, 11,  9],
           [11,  4,  8]])

    This example demonstrates that if the data set is not large and there
    are values equal to the median, the p-value can be sensitive to the
    choice of `ties`.

    """
    if len(samples) < 2:
        raise ValueError('median_test requires two or more samples.')

    ties_options = ['below', 'above', 'ignore']
    if ties not in ties_options:
        raise ValueError(f"invalid 'ties' option '{ties}'; 'ties' must be one "
                         f"of: {str(ties_options)[1:-1]}")

    data = [np.asarray(sample) for sample in samples]

    # Validate the sizes and shapes of the arguments.
    for k, d in enumerate(data):
        if d.size == 0:
            raise ValueError("Sample %d is empty. All samples must "
                             "contain at least one value." % (k + 1))
        if d.ndim != 1:
            raise ValueError("Sample %d has %d dimensions.  All "
                             "samples must be one-dimensional sequences." %
                             (k + 1, d.ndim))

    cdata = np.concatenate(data)
    contains_nan, nan_policy = _contains_nan(cdata, nan_policy)
    if contains_nan and nan_policy == 'propagate':
        return MedianTestResult(np.nan, np.nan, np.nan, None)

    if contains_nan:
        grand_median = np.median(cdata[~np.isnan(cdata)])
    else:
        grand_median = np.median(cdata)
    # When the minimum version of numpy supported by scipy is 1.9.0,
    # the above if/else statement can be replaced by the single line:
    #     grand_median = np.nanmedian(cdata)

    # Create the contingency table.
    table = np.zeros((2, len(data)), dtype=np.int64)
    for k, sample in enumerate(data):
        sample = sample[~np.isnan(sample)]

        nabove = count_nonzero(sample > grand_median)
        nbelow = count_nonzero(sample < grand_median)
        nequal = sample.size - (nabove + nbelow)
        table[0, k] += nabove
        table[1, k] += nbelow
        if ties == "below":
            table[1, k] += nequal
        elif ties == "above":
            table[0, k] += nequal

    # Check that no row or column of the table is all zero.
    # Such a table can not be given to chi2_contingency, because it would have
    # a zero in the table of expected frequencies.
    rowsums = table.sum(axis=1)
    if rowsums[0] == 0:
        raise ValueError(f"All values are below the grand median ({grand_median}).")
    if rowsums[1] == 0:
        raise ValueError(f"All values are above the grand median ({grand_median}).")
    if ties == "ignore":
        # We already checked that each sample has at least one value, but it
        # is possible that all those values equal the grand median.  If `ties`
        # is "ignore", that would result in a column of zeros in `table`.  We
        # check for that case here.
        zero_cols = np.nonzero((table == 0).all(axis=0))[0]
        if len(zero_cols) > 0:
            msg = ("All values in sample %d are equal to the grand "
                   "median (%r), so they are ignored, resulting in an "
                   "empty sample." % (zero_cols[0] + 1, grand_median))
            raise ValueError(msg)

    stat, p, dof, expected = chi2_contingency(table, lambda_=lambda_,
                                              correction=correction)
    return MedianTestResult(stat, p, grand_median, table)


def _circfuncs_common(samples, period, xp=None):
    xp = array_namespace(samples) if xp is None else xp

    if xp.isdtype(samples.dtype, 'integral'):
        dtype = xp.asarray(1.).dtype  # get default float type
        samples = xp.asarray(samples, dtype=dtype)

    # Recast samples as radians that range between 0 and 2 pi and calculate
    # the sine and cosine
    scaled_samples = samples * ((2.0 * pi) / period)
    sin_samp = xp.sin(scaled_samples)
    cos_samp = xp.cos(scaled_samples)

    return samples, sin_samp, cos_samp


@_axis_nan_policy_factory(
    lambda x: x, n_outputs=1, default_axis=None,
    result_to_tuple=lambda x: (x,)
)
def circmean(samples, high=2*pi, low=0, axis=None, nan_policy='propagate'):
    r"""Compute the circular mean of a sample of angle observations.

    Given :math:`n` angle observations :math:`x_1, \cdots, x_n` measured in
    radians, their *circular mean* is defined by ([1]_, Eq. 2.2.4)

    .. math::

       \mathrm{Arg} \left( \frac{1}{n} \sum_{k=1}^n e^{i x_k} \right)

    where :math:`i` is the imaginary unit and :math:`\mathop{\mathrm{Arg}} z`
    gives the principal value of the argument of complex number :math:`z`,
    restricted to the range :math:`[0,2\pi]` by default.  :math:`z` in the
    above expression is known as the `mean resultant vector`.

    Parameters
    ----------
    samples : array_like
        Input array of angle observations.  The value of a full angle is
        equal to ``(high - low)``.
    high : float, optional
        Upper boundary of the principal value of an angle.  Default is ``2*pi``.
    low : float, optional
        Lower boundary of the principal value of an angle.  Default is ``0``.

    Returns
    -------
    circmean : float
        Circular mean, restricted to the range ``[low, high]``.

        If the mean resultant vector is zero, an input-dependent,
        implementation-defined number between ``[low, high]`` is returned.
        If the input array is empty, ``np.nan`` is returned.

    See Also
    --------
    circstd : Circular standard deviation.
    circvar : Circular variance.

    References
    ----------
    .. [1] Mardia, K. V. and Jupp, P. E. *Directional Statistics*.
           John Wiley & Sons, 1999.

    Examples
    --------
    For readability, all angles are printed out in degrees.

    >>> import numpy as np
    >>> from scipy.stats import circmean
    >>> import matplotlib.pyplot as plt
    >>> angles = np.deg2rad(np.array([20, 30, 330]))
    >>> circmean = circmean(angles)
    >>> np.rad2deg(circmean)
    7.294976657784009

    >>> mean = angles.mean()
    >>> np.rad2deg(mean)
    126.66666666666666

    Plot and compare the circular mean against the arithmetic mean.

    >>> plt.plot(np.cos(np.linspace(0, 2*np.pi, 500)),
    ...          np.sin(np.linspace(0, 2*np.pi, 500)),
    ...          c='k')
    >>> plt.scatter(np.cos(angles), np.sin(angles), c='k')
    >>> plt.scatter(np.cos(circmean), np.sin(circmean), c='b',
    ...             label='circmean')
    >>> plt.scatter(np.cos(mean), np.sin(mean), c='r', label='mean')
    >>> plt.legend()
    >>> plt.axis('equal')
    >>> plt.show()

    """
    xp = array_namespace(samples)
    # Needed for non-NumPy arrays to get appropriate NaN result
    # Apparently atan2(0, 0) is 0, even though it is mathematically undefined
    if xp_size(samples) == 0:
        return xp.mean(samples, axis=axis)
    period = high - low
    samples, sin_samp, cos_samp = _circfuncs_common(samples, period, xp=xp)
    sin_sum = xp.sum(sin_samp, axis=axis)
    cos_sum = xp.sum(cos_samp, axis=axis)
    res = xp.atan2(sin_sum, cos_sum)

    res = res[()] if res.ndim == 0 else res
    return (res * (period / (2.0 * pi)) - low) % period + low


@_axis_nan_policy_factory(
    lambda x: x, n_outputs=1, default_axis=None,
    result_to_tuple=lambda x: (x,)
)
def circvar(samples, high=2*pi, low=0, axis=None, nan_policy='propagate'):
    r"""Compute the circular variance of a sample of angle observations.

    Given :math:`n` angle observations :math:`x_1, \cdots, x_n` measured in
    radians, their *circular variance* is defined by ([2]_, Eq. 2.3.3)

    .. math::

       1 - \left| \frac{1}{n} \sum_{k=1}^n e^{i x_k} \right|

    where :math:`i` is the imaginary unit and :math:`|z|` gives the length
    of the complex number :math:`z`.  :math:`|z|` in the above expression
    is known as the `mean resultant length`.

    Parameters
    ----------
    samples : array_like
        Input array of angle observations.  The value of a full angle is
        equal to ``(high - low)``.
    high : float, optional
        Upper boundary of the principal value of an angle.  Default is ``2*pi``.
    low : float, optional
        Lower boundary of the principal value of an angle.  Default is ``0``.

    Returns
    -------
    circvar : float
        Circular variance.  The returned value is in the range ``[0, 1]``,
        where ``0`` indicates no variance and ``1`` indicates large variance.

        If the input array is empty, ``np.nan`` is returned.

    See Also
    --------
    circmean : Circular mean.
    circstd : Circular standard deviation.

    Notes
    -----
    In the limit of small angles, the circular variance is close to
    half the 'linear' variance if measured in radians.

    References
    ----------
    .. [1] Fisher, N.I. *Statistical analysis of circular data*. Cambridge
           University Press, 1993.
    .. [2] Mardia, K. V. and Jupp, P. E. *Directional Statistics*.
           John Wiley & Sons, 1999.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import circvar
    >>> import matplotlib.pyplot as plt
    >>> samples_1 = np.array([0.072, -0.158, 0.077, 0.108, 0.286,
    ...                       0.133, -0.473, -0.001, -0.348, 0.131])
    >>> samples_2 = np.array([0.111, -0.879, 0.078, 0.733, 0.421,
    ...                       0.104, -0.136, -0.867,  0.012,  0.105])
    >>> circvar_1 = circvar(samples_1)
    >>> circvar_2 = circvar(samples_2)

    Plot the samples.

    >>> fig, (left, right) = plt.subplots(ncols=2)
    >>> for image in (left, right):
    ...     image.plot(np.cos(np.linspace(0, 2*np.pi, 500)),
    ...                np.sin(np.linspace(0, 2*np.pi, 500)),
    ...                c='k')
    ...     image.axis('equal')
    ...     image.axis('off')
    >>> left.scatter(np.cos(samples_1), np.sin(samples_1), c='k', s=15)
    >>> left.set_title(f"circular variance: {np.round(circvar_1, 2)!r}")
    >>> right.scatter(np.cos(samples_2), np.sin(samples_2), c='k', s=15)
    >>> right.set_title(f"circular variance: {np.round(circvar_2, 2)!r}")
    >>> plt.show()

    """
    xp = array_namespace(samples)
    period = high - low
    samples, sin_samp, cos_samp = _circfuncs_common(samples, period, xp=xp)
    sin_mean = xp.mean(sin_samp, axis=axis)
    cos_mean = xp.mean(cos_samp, axis=axis)
    hypotenuse = (sin_mean**2. + cos_mean**2.)**0.5
    # hypotenuse can go slightly above 1 due to rounding errors
    R = xp.clip(hypotenuse, max=1.)

    res = 1. - R
    return res


@_axis_nan_policy_factory(
    lambda x: x, n_outputs=1, default_axis=None,
    result_to_tuple=lambda x: (x,)
)
def circstd(samples, high=2*pi, low=0, axis=None, nan_policy='propagate', *,
            normalize=False):
    r"""
    Compute the circular standard deviation of a sample of angle observations.

    Given :math:`n` angle observations :math:`x_1, \cdots, x_n` measured in
    radians, their `circular standard deviation` is defined by
    ([2]_, Eq. 2.3.11)

    .. math::

       \sqrt{ -2 \log \left| \frac{1}{n} \sum_{k=1}^n e^{i x_k} \right| }

    where :math:`i` is the imaginary unit and :math:`|z|` gives the length
    of the complex number :math:`z`.  :math:`|z|` in the above expression
    is known as the `mean resultant length`.

    Parameters
    ----------
    samples : array_like
        Input array of angle observations.  The value of a full angle is
        equal to ``(high - low)``.
    high : float, optional
        Upper boundary of the principal value of an angle.  Default is ``2*pi``.
    low : float, optional
        Lower boundary of the principal value of an angle.  Default is ``0``.
    normalize : boolean, optional
        If ``False`` (the default), the return value is computed from the
        above formula with the input scaled by ``(2*pi)/(high-low)`` and
        the output scaled (back) by ``(high-low)/(2*pi)``.  If ``True``,
        the output is not scaled and is returned directly.

    Returns
    -------
    circstd : float
        Circular standard deviation, optionally normalized.

        If the input array is empty, ``np.nan`` is returned.

    See Also
    --------
    circmean : Circular mean.
    circvar : Circular variance.

    Notes
    -----
    In the limit of small angles, the circular standard deviation is close
    to the 'linear' standard deviation if ``normalize`` is ``False``.

    References
    ----------
    .. [1] Mardia, K. V. (1972). 2. In *Statistics of Directional Data*
       (pp. 18-24). Academic Press. :doi:`10.1016/C2013-0-07425-7`.
    .. [2] Mardia, K. V. and Jupp, P. E. *Directional Statistics*.
           John Wiley & Sons, 1999.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import circstd
    >>> import matplotlib.pyplot as plt
    >>> samples_1 = np.array([0.072, -0.158, 0.077, 0.108, 0.286,
    ...                       0.133, -0.473, -0.001, -0.348, 0.131])
    >>> samples_2 = np.array([0.111, -0.879, 0.078, 0.733, 0.421,
    ...                       0.104, -0.136, -0.867,  0.012,  0.105])
    >>> circstd_1 = circstd(samples_1)
    >>> circstd_2 = circstd(samples_2)

    Plot the samples.

    >>> fig, (left, right) = plt.subplots(ncols=2)
    >>> for image in (left, right):
    ...     image.plot(np.cos(np.linspace(0, 2*np.pi, 500)),
    ...                np.sin(np.linspace(0, 2*np.pi, 500)),
    ...                c='k')
    ...     image.axis('equal')
    ...     image.axis('off')
    >>> left.scatter(np.cos(samples_1), np.sin(samples_1), c='k', s=15)
    >>> left.set_title(f"circular std: {np.round(circstd_1, 2)!r}")
    >>> right.plot(np.cos(np.linspace(0, 2*np.pi, 500)),
    ...            np.sin(np.linspace(0, 2*np.pi, 500)),
    ...            c='k')
    >>> right.scatter(np.cos(samples_2), np.sin(samples_2), c='k', s=15)
    >>> right.set_title(f"circular std: {np.round(circstd_2, 2)!r}")
    >>> plt.show()

    """
    xp = array_namespace(samples)
    period = high - low
    samples, sin_samp, cos_samp = _circfuncs_common(samples, period, xp=xp)
    sin_mean = xp.mean(sin_samp, axis=axis)  # [1] (2.2.3)
    cos_mean = xp.mean(cos_samp, axis=axis)  # [1] (2.2.3)
    hypotenuse = (sin_mean**2. + cos_mean**2.)**0.5
    # hypotenuse can go slightly above 1 due to rounding errors
    R = xp.clip(hypotenuse, max=1.)  # [1] (2.2.4)

    res = (-2*xp.log(R))**0.5+0.0  # torch.pow returns -0.0 if R==1
    if not normalize:
        res *= (high-low)/(2.*pi)  # [1] (2.3.14) w/ (2.3.7)
    return res


class DirectionalStats:
    def __init__(self, mean_direction, mean_resultant_length):
        self.mean_direction = mean_direction
        self.mean_resultant_length = mean_resultant_length

    def __repr__(self):
        return (f"DirectionalStats(mean_direction={self.mean_direction},"
                f" mean_resultant_length={self.mean_resultant_length})")


def directional_stats(samples, *, axis=0, normalize=True):
    """
    Computes sample statistics for directional data.

    Computes the directional mean (also called the mean direction vector) and
    mean resultant length of a sample of vectors.

    The directional mean is a measure of "preferred direction" of vector data.
    It is analogous to the sample mean, but it is for use when the length of
    the data is irrelevant (e.g. unit vectors).

    The mean resultant length is a value between 0 and 1 used to quantify the
    dispersion of directional data: the smaller the mean resultant length, the
    greater the dispersion. Several definitions of directional variance
    involving the mean resultant length are given in [1]_ and [2]_.

    Parameters
    ----------
    samples : array_like
        Input array. Must be at least two-dimensional, and the last axis of the
        input must correspond with the dimensionality of the vector space.
        When the input is exactly two dimensional, this means that each row
        of the data is a vector observation.
    axis : int, default: 0
        Axis along which the directional mean is computed.
    normalize: boolean, default: True
        If True, normalize the input to ensure that each observation is a
        unit vector. It the observations are already unit vectors, consider
        setting this to False to avoid unnecessary computation.

    Returns
    -------
    res : DirectionalStats
        An object containing attributes:

        mean_direction : ndarray
            Directional mean.
        mean_resultant_length : ndarray
            The mean resultant length [1]_.

    See Also
    --------
    circmean: circular mean; i.e. directional mean for 2D *angles*
    circvar: circular variance; i.e. directional variance for 2D *angles*

    Notes
    -----
    This uses a definition of directional mean from [1]_.
    Assuming the observations are unit vectors, the calculation is as follows.

    .. code-block:: python

        mean = samples.mean(axis=0)
        mean_resultant_length = np.linalg.norm(mean)
        mean_direction = mean / mean_resultant_length

    This definition is appropriate for *directional* data (i.e. vector data
    for which the magnitude of each observation is irrelevant) but not
    for *axial* data (i.e. vector data for which the magnitude and *sign* of
    each observation is irrelevant).

    Several definitions of directional variance involving the mean resultant
    length ``R`` have been proposed, including ``1 - R`` [1]_, ``1 - R**2``
    [2]_, and ``2 * (1 - R)`` [2]_. Rather than choosing one, this function
    returns ``R`` as attribute `mean_resultant_length` so the user can compute
    their preferred measure of dispersion.

    References
    ----------
    .. [1] Mardia, Jupp. (2000). *Directional Statistics*
       (p. 163). Wiley.

    .. [2] https://en.wikipedia.org/wiki/Directional_statistics

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import directional_stats
    >>> data = np.array([[3, 4],    # first observation, 2D vector space
    ...                  [6, -8]])  # second observation
    >>> dirstats = directional_stats(data)
    >>> dirstats.mean_direction
    array([1., 0.])

    In contrast, the regular sample mean of the vectors would be influenced
    by the magnitude of each observation. Furthermore, the result would not be
    a unit vector.

    >>> data.mean(axis=0)
    array([4.5, -2.])

    An exemplary use case for `directional_stats` is to find a *meaningful*
    center for a set of observations on a sphere, e.g. geographical locations.

    >>> data = np.array([[0.8660254, 0.5, 0.],
    ...                  [0.8660254, -0.5, 0.]])
    >>> dirstats = directional_stats(data)
    >>> dirstats.mean_direction
    array([1., 0., 0.])

    The regular sample mean on the other hand yields a result which does not
    lie on the surface of the sphere.

    >>> data.mean(axis=0)
    array([0.8660254, 0., 0.])

    The function also returns the mean resultant length, which
    can be used to calculate a directional variance. For example, using the
    definition ``Var(z) = 1 - R`` from [2]_ where ``R`` is the
    mean resultant length, we can calculate the directional variance of the
    vectors in the above example as:

    >>> 1 - dirstats.mean_resultant_length
    0.13397459716167093
    """
    xp = array_namespace(samples)
    samples = xp.asarray(samples)

    if samples.ndim < 2:
        raise ValueError("samples must at least be two-dimensional. "
                         f"Instead samples has shape: {tuple(samples.shape)}")
    samples = xp.moveaxis(samples, axis, 0)
    if normalize:
        vectornorms = xp_vector_norm(samples, axis=-1, keepdims=True, xp=xp)
        samples = samples/vectornorms
    mean = xp.mean(samples, axis=0)
    mean_resultant_length = xp_vector_norm(mean, axis=-1, keepdims=True, xp=xp)
    mean_direction = mean / mean_resultant_length
    mrl = xp.squeeze(mean_resultant_length, axis=-1)
    mean_resultant_length = mrl[()] if mrl.ndim == 0 else mrl
    return DirectionalStats(mean_direction, mean_resultant_length)


def false_discovery_control(ps, *, axis=0, method='bh'):
    """Adjust p-values to control the false discovery rate.

    The false discovery rate (FDR) is the expected proportion of rejected null
    hypotheses that are actually true.
    If the null hypothesis is rejected when the *adjusted* p-value falls below
    a specified level, the false discovery rate is controlled at that level.

    Parameters
    ----------
    ps : 1D array_like
        The p-values to adjust. Elements must be real numbers between 0 and 1.
    axis : int
        The axis along which to perform the adjustment. The adjustment is
        performed independently along each axis-slice. If `axis` is None, `ps`
        is raveled before performing the adjustment.
    method : {'bh', 'by'}
        The false discovery rate control procedure to apply: ``'bh'`` is for
        Benjamini-Hochberg [1]_ (Eq. 1), ``'by'`` is for Benjaminini-Yekutieli
        [2]_ (Theorem 1.3). The latter is more conservative, but it is
        guaranteed to control the FDR even when the p-values are not from
        independent tests.

    Returns
    -------
    ps_adusted : array_like
        The adjusted p-values. If the null hypothesis is rejected where these
        fall below a specified level, the false discovery rate is controlled
        at that level.

    See Also
    --------
    combine_pvalues
    statsmodels.stats.multitest.multipletests

    Notes
    -----
    In multiple hypothesis testing, false discovery control procedures tend to
    offer higher power than familywise error rate control procedures (e.g.
    Bonferroni correction [1]_).

    If the p-values correspond with independent tests (or tests with
    "positive regression dependencies" [2]_), rejecting null hypotheses
    corresponding with Benjamini-Hochberg-adjusted p-values below :math:`q`
    controls the false discovery rate at a level less than or equal to
    :math:`q m_0 / m`, where :math:`m_0` is the number of true null hypotheses
    and :math:`m` is the total number of null hypotheses tested. The same is
    true even for dependent tests when the p-values are adjusted accorded to
    the more conservative Benjaminini-Yekutieli procedure.

    The adjusted p-values produced by this function are comparable to those
    produced by the R function ``p.adjust`` and the statsmodels function
    `statsmodels.stats.multitest.multipletests`. Please consider the latter
    for more advanced methods of multiple comparison correction.

    References
    ----------
    .. [1] Benjamini, Yoav, and Yosef Hochberg. "Controlling the false
           discovery rate: a practical and powerful approach to multiple
           testing." Journal of the Royal statistical society: series B
           (Methodological) 57.1 (1995): 289-300.

    .. [2] Benjamini, Yoav, and Daniel Yekutieli. "The control of the false
           discovery rate in multiple testing under dependency." Annals of
           statistics (2001): 1165-1188.

    .. [3] TileStats. FDR - Benjamini-Hochberg explained - Youtube.
           https://www.youtube.com/watch?v=rZKa4tW2NKs.

    .. [4] Neuhaus, Karl-Ludwig, et al. "Improved thrombolysis in acute
           myocardial infarction with front-loaded administration of alteplase:
           results of the rt-PA-APSAC patency study (TAPS)." Journal of the
           American College of Cardiology 19.5 (1992): 885-891.

    Examples
    --------
    We follow the example from [1]_.

        Thrombolysis with recombinant tissue-type plasminogen activator (rt-PA)
        and anisoylated plasminogen streptokinase activator (APSAC) in
        myocardial infarction has been proved to reduce mortality. [4]_
        investigated the effects of a new front-loaded administration of rt-PA
        versus those obtained with a standard regimen of APSAC, in a randomized
        multicentre trial in 421 patients with acute myocardial infarction.

    There were four families of hypotheses tested in the study, the last of
    which was "cardiac and other events after the start of thrombolitic
    treatment". FDR control may be desired in this family of hypotheses
    because it would not be appropriate to conclude that the front-loaded
    treatment is better if it is merely equivalent to the previous treatment.

    The p-values corresponding with the 15 hypotheses in this family were

    >>> ps = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
    ...       0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.000]

    If the chosen significance level is 0.05, we may be tempted to reject the
    null hypotheses for the tests corresponding with the first nine p-values,
    as the first nine p-values fall below the chosen significance level.
    However, this would ignore the problem of "multiplicity": if we fail to
    correct for the fact that multiple comparisons are being performed, we
    are more likely to incorrectly reject true null hypotheses.

    One approach to the multiplicity problem is to control the family-wise
    error rate (FWER), that is, the rate at which the null hypothesis is
    rejected when it is actually true. A common procedure of this kind is the
    Bonferroni correction [1]_.  We begin by multiplying the p-values by the
    number of hypotheses tested.

    >>> import numpy as np
    >>> np.array(ps) * len(ps)
    array([1.5000e-03, 6.0000e-03, 2.8500e-02, 1.4250e-01, 3.0150e-01,
           4.1700e-01, 4.4700e-01, 5.1600e-01, 6.8850e-01, 4.8600e+00,
           6.3930e+00, 8.5785e+00, 9.7920e+00, 1.1385e+01, 1.5000e+01])

    To control the FWER at 5%, we reject only the hypotheses corresponding
    with adjusted p-values less than 0.05. In this case, only the hypotheses
    corresponding with the first three p-values can be rejected. According to
    [1]_, these three hypotheses concerned "allergic reaction" and "two
    different aspects of bleeding."

    An alternative approach is to control the false discovery rate: the
    expected fraction of rejected null hypotheses that are actually true. The
    advantage of this approach is that it typically affords greater power: an
    increased rate of rejecting the null hypothesis when it is indeed false. To
    control the false discovery rate at 5%, we apply the Benjamini-Hochberg
    p-value adjustment.

    >>> from scipy import stats
    >>> stats.false_discovery_control(ps)
    array([0.0015    , 0.003     , 0.0095    , 0.035625  , 0.0603    ,
           0.06385714, 0.06385714, 0.0645    , 0.0765    , 0.486     ,
           0.58118182, 0.714875  , 0.75323077, 0.81321429, 1.        ])

    Now, the first *four* adjusted p-values fall below 0.05, so we would reject
    the null hypotheses corresponding with these *four* p-values. Rejection
    of the fourth null hypothesis was particularly important to the original
    study as it led to the conclusion that the new treatment had a
    "substantially lower in-hospital mortality rate."

    """
    # Input Validation and Special Cases
    ps = np.asarray(ps)

    ps_in_range = (np.issubdtype(ps.dtype, np.number)
                   and np.all(ps == np.clip(ps, 0, 1)))
    if not ps_in_range:
        raise ValueError("`ps` must include only numbers between 0 and 1.")

    methods = {'bh', 'by'}
    if method.lower() not in methods:
        raise ValueError(f"Unrecognized `method` '{method}'."
                         f"Method must be one of {methods}.")
    method = method.lower()

    if axis is None:
        axis = 0
        ps = ps.ravel()

    axis = np.asarray(axis)[()]
    if not np.issubdtype(axis.dtype, np.integer) or axis.size != 1:
        raise ValueError("`axis` must be an integer or `None`")

    if ps.size <= 1 or ps.shape[axis] <= 1:
        return ps[()]

    ps = np.moveaxis(ps, axis, -1)
    m = ps.shape[-1]

    # Main Algorithm
    # Equivalent to the ideas of [1] and [2], except that this adjusts the
    # p-values as described in [3]. The results are similar to those produced
    # by R's p.adjust.

    # "Let [ps] be the ordered observed p-values..."
    order = np.argsort(ps, axis=-1)
    ps = np.take_along_axis(ps, order, axis=-1)  # this copies ps

    # Equation 1 of [1] rearranged to reject when p is less than specified q
    i = np.arange(1, m+1)
    ps *= m / i

    # Theorem 1.3 of [2]
    if method == 'by':
        ps *= np.sum(1 / i)

    # accounts for rejecting all null hypotheses i for i < k, where k is
    # defined in Eq. 1 of either [1] or [2]. See [3]. Starting with the index j
    # of the second to last element, we replace element j with element j+1 if
    # the latter is smaller.
    np.minimum.accumulate(ps[..., ::-1], out=ps[..., ::-1], axis=-1)

    # Restore original order of axes and data
    np.put_along_axis(ps, order, values=ps.copy(), axis=-1)
    ps = np.moveaxis(ps, -1, axis)

    return np.clip(ps, 0, 1)
<|endoftext|>import numpy as np
from scipy.sparse import coo_matrix
from scipy._lib._bunch import _make_tuple_bunch


CrosstabResult = _make_tuple_bunch(
    "CrosstabResult", ["elements", "count"]
)


def crosstab(*args, levels=None, sparse=False):
    """
    Return table of counts for each possible unique combination in ``*args``.

    When ``len(args) > 1``, the array computed by this function is
    often referred to as a *contingency table* [1]_.

    The arguments must be sequences with the same length.  The second return
    value, `count`, is an integer array with ``len(args)`` dimensions.  If
    `levels` is None, the shape of `count` is ``(n0, n1, ...)``, where ``nk``
    is the number of unique elements in ``args[k]``.

    Parameters
    ----------
    *args : sequences
        A sequence of sequences whose unique aligned elements are to be
        counted.  The sequences in args must all be the same length.
    levels : sequence, optional
        If `levels` is given, it must be a sequence that is the same length as
        `args`.  Each element in `levels` is either a sequence or None.  If it
        is a sequence, it gives the values in the corresponding sequence in
        `args` that are to be counted.  If any value in the sequences in `args`
        does not occur in the corresponding sequence in `levels`, that value
        is ignored and not counted in the returned array `count`.  The default
        value of `levels` for ``args[i]`` is ``np.unique(args[i])``
    sparse : bool, optional
        If True, return a sparse matrix.  The matrix will be an instance of
        the `scipy.sparse.coo_matrix` class.  Because SciPy's sparse matrices
        must be 2-d, only two input sequences are allowed when `sparse` is
        True.  Default is False.

    Returns
    -------
    res : CrosstabResult
        An object containing the following attributes:

        elements : tuple of numpy.ndarrays.
            Tuple of length ``len(args)`` containing the arrays of elements
            that are counted in `count`.  These can be interpreted as the
            labels of the corresponding dimensions of `count`. If `levels` was
            given, then if ``levels[i]`` is not None, ``elements[i]`` will
            hold the values given in ``levels[i]``.
        count : numpy.ndarray or scipy.sparse.coo_matrix
            Counts of the unique elements in ``zip(*args)``, stored in an
            array. Also known as a *contingency table* when ``len(args) > 1``.

    See Also
    --------
    numpy.unique

    Notes
    -----
    .. versionadded:: 1.7.0

    References
    ----------
    .. [1] "Contingency table", http://en.wikipedia.org/wiki/Contingency_table

    Examples
    --------
    >>> from scipy.stats.contingency import crosstab

    Given the lists `a` and `x`, create a contingency table that counts the
    frequencies of the corresponding pairs.

    >>> a = ['A', 'B', 'A', 'A', 'B', 'B', 'A', 'A', 'B', 'B']
    >>> x = ['X', 'X', 'X', 'Y', 'Z', 'Z', 'Y', 'Y', 'Z', 'Z']
    >>> res = crosstab(a, x)
    >>> avals, xvals = res.elements
    >>> avals
    array(['A', 'B'], dtype='<U1')
    >>> xvals
    array(['X', 'Y', 'Z'], dtype='<U1')
    >>> res.count
    array([[2, 3, 0],
           [1, 0, 4]])

    So ``('A', 'X')`` occurs twice, ``('A', 'Y')`` occurs three times, etc.

    Higher dimensional contingency tables can be created.

    >>> p = [0, 0, 0, 0, 1, 1, 1, 0, 0, 1]
    >>> res = crosstab(a, x, p)
    >>> res.count
    array([[[2, 0],
            [2, 1],
            [0, 0]],
           [[1, 0],
            [0, 0],
            [1, 3]]])
    >>> res.count.shape
    (2, 3, 2)

    The values to be counted can be set by using the `levels` argument.
    It allows the elements of interest in each input sequence to be
    given explicitly instead finding the unique elements of the sequence.

    For example, suppose one of the arguments is an array containing the
    answers to a survey question, with integer values 1 to 4.  Even if the
    value 1 does not occur in the data, we want an entry for it in the table.

    >>> q1 = [2, 3, 3, 2, 4, 4, 2, 3, 4, 4, 4, 3, 3, 3, 4]  # 1 does not occur.
    >>> q2 = [4, 4, 2, 2, 2, 4, 1, 1, 2, 2, 4, 2, 2, 2, 4]  # 3 does not occur.
    >>> options = [1, 2, 3, 4]
    >>> res = crosstab(q1, q2, levels=(options, options))
    >>> res.count
    array([[0, 0, 0, 0],
           [1, 1, 0, 1],
           [1, 4, 0, 1],
           [0, 3, 0, 3]])

    If `levels` is given, but an element of `levels` is None, the unique values
    of the corresponding argument are used. For example,

    >>> res = crosstab(q1, q2, levels=(None, options))
    >>> res.elements
    [array([2, 3, 4]), [1, 2, 3, 4]]
    >>> res.count
    array([[1, 1, 0, 1],
           [1, 4, 0, 1],
           [0, 3, 0, 3]])

    If we want to ignore the pairs where 4 occurs in ``q2``, we can
    give just the values [1, 2] to `levels`, and the 4 will be ignored:

    >>> res = crosstab(q1, q2, levels=(None, [1, 2]))
    >>> res.elements
    [array([2, 3, 4]), [1, 2]]
    >>> res.count
    array([[1, 1],
           [1, 4],
           [0, 3]])

    Finally, let's repeat the first example, but return a sparse matrix:

    >>> res = crosstab(a, x, sparse=True)
    >>> res.count
    <COOrdinate sparse matrix of dtype 'int64'
        with 4 stored elements and shape (2, 3)>
    >>> res.count.toarray()
    array([[2, 3, 0],
           [1, 0, 4]])

    """
    nargs = len(args)
    if nargs == 0:
        raise TypeError("At least one input sequence is required.")

    len0 = len(args[0])
    if not all(len(a) == len0 for a in args[1:]):
        raise ValueError("All input sequences must have the same length.")

    if sparse and nargs != 2:
        raise ValueError("When `sparse` is True, only two input sequences "
                         "are allowed.")

    if levels is None:
        # Call np.unique with return_inverse=True on each argument.
        actual_levels, indices = zip(*[np.unique(a, return_inverse=True)
                                       for a in args])
    else:
        # `levels` is not None...
        if len(levels) != nargs:
            raise ValueError('len(levels) must equal the number of input '
                             'sequences')

        args = [np.asarray(arg) for arg in args]
        mask = np.zeros((nargs, len0), dtype=np.bool_)
        inv = np.zeros((nargs, len0), dtype=np.intp)
        actual_levels = []
        for k, (levels_list, arg) in enumerate(zip(levels, args)):
            if levels_list is None:
                levels_list, inv[k, :] = np.unique(arg, return_inverse=True)
                mask[k, :] = True
            else:
                q = arg == np.asarray(levels_list).reshape(-1, 1)
                mask[k, :] = np.any(q, axis=0)
                qnz = q.T.nonzero()
                inv[k, qnz[0]] = qnz[1]
            actual_levels.append(levels_list)

        mask_all = mask.all(axis=0)
        indices = tuple(inv[:, mask_all])

    if sparse:
        count = coo_matrix((np.ones(len(indices[0]), dtype=int),
                            (indices[0], indices[1])))
        count.sum_duplicates()
    else:
        shape = [len(u) for u in actual_levels]
        count = np.zeros(shape, dtype=int)
        np.add.at(count, indices, 1)

    return CrosstabResult(actual_levels, count)
<|endoftext|>import numpy as np

from scipy import stats
from ._stats_py import _get_pvalue, _rankdata, _SimpleNormal
from . import _morestats
from ._axis_nan_policy import _broadcast_arrays
from ._hypotests import _get_wilcoxon_distr
from scipy._lib._util import _lazywhere, _get_nan


class WilcoxonDistribution:

    def __init__(self, n):
        n = np.asarray(n).astype(int, copy=False)
        self.n = n
        self._dists = {ni: _get_wilcoxon_distr(ni) for ni in np.unique(n)}

    def _cdf1(self, k, n):
        pmfs = self._dists[n]
        return pmfs[:k + 1].sum()

    def _cdf(self, k, n):
        return np.vectorize(self._cdf1, otypes=[float])(k, n)

    def _sf1(self, k, n):
        pmfs = self._dists[n]
        return pmfs[k:].sum()

    def _sf(self, k, n):
        return np.vectorize(self._sf1, otypes=[float])(k, n)

    def mean(self):
        return self.n * (self.n + 1) / 4

    def _prep(self, k):
        k = np.asarray(k).astype(int, copy=False)
        mn = self.mean()
        out = np.empty(k.shape, dtype=np.float64)
        return k, mn, out

    def cdf(self, k):
        k, mn, out = self._prep(k)
        return _lazywhere(k <= mn, (k, self.n), self._cdf,
                          f2=lambda k, n: 1 - self._sf(k+1, n))[()]

    def sf(self, k):
        k, mn, out = self._prep(k)
        return _lazywhere(k <= mn, (k, self.n), self._sf,
                          f2=lambda k, n: 1 - self._cdf(k-1, n))[()]


def _wilcoxon_iv(x, y, zero_method, correction, alternative, method, axis):

    axis = np.asarray(axis)[()]
    message = "`axis` must be an integer."
    if not np.issubdtype(axis.dtype, np.integer) or axis.ndim != 0:
        raise ValueError(message)

    message = '`axis` must be compatible with the shape(s) of `x` (and `y`)'
    try:
        if y is None:
            x = np.asarray(x)
            d = x
        else:
            x, y = _broadcast_arrays((x, y), axis=axis)
            d = x - y
        d = np.moveaxis(d, axis, -1)
    except np.AxisError as e:
        raise ValueError(message) from e

    message = "`x` and `y` must have the same length along `axis`."
    if y is not None and x.shape[axis] != y.shape[axis]:
        raise ValueError(message)

    message = "`x` (and `y`, if provided) must be an array of real numbers."
    if np.issubdtype(d.dtype, np.integer):
        d = d.astype(np.float64)
    if not np.issubdtype(d.dtype, np.floating):
        raise ValueError(message)

    zero_method = str(zero_method).lower()
    zero_methods = {"wilcox", "pratt", "zsplit"}
    message = f"`zero_method` must be one of {zero_methods}."
    if zero_method not in zero_methods:
        raise ValueError(message)

    corrections = {True, False}
    message = f"`correction` must be one of {corrections}."
    if correction not in corrections:
        raise ValueError(message)

    alternative = str(alternative).lower()
    alternatives = {"two-sided", "less", "greater"}
    message = f"`alternative` must be one of {alternatives}."
    if alternative not in alternatives:
        raise ValueError(message)

    if not isinstance(method, stats.PermutationMethod):
        methods = {"auto", "asymptotic", "exact"}
        message = (f"`method` must be one of {methods} or "
                   "an instance of `stats.PermutationMethod`.")
        if method not in methods:
            raise ValueError(message)
    output_z = True if method == 'asymptotic' else False

    # For small samples, we decide later whether to perform an exact test or a
    # permutation test. The reason is that the presence of ties is not
    # known at the input validation stage.
    n_zero = np.sum(d == 0)
    if method == "auto" and d.shape[-1] > 50:
        method = "asymptotic"

    return d, zero_method, correction, alternative, method, axis, output_z, n_zero


def _wilcoxon_statistic(d, method, zero_method='wilcox'):

    i_zeros = (d == 0)

    if zero_method == 'wilcox':
        # Wilcoxon's method for treating zeros was to remove them from
        # the calculation. We do this by replacing 0s with NaNs, which
        # are ignored anyway.
        if not d.flags['WRITEABLE']:
            d = d.copy()
        d[i_zeros] = np.nan

    i_nan = np.isnan(d)
    n_nan = np.sum(i_nan, axis=-1)
    count = d.shape[-1] - n_nan

    r, t = _rankdata(abs(d), 'average', return_ties=True)

    r_plus = np.sum((d > 0) * r, axis=-1)
    r_minus = np.sum((d < 0) * r, axis=-1)

    has_ties = (t == 0).any()

    if zero_method == "zsplit":
        # The "zero-split" method for treating zeros is to add half their contribution
        # to r_plus and half to r_minus.
        # See gh-2263 for the origin of this method.
        r_zero_2 = np.sum(i_zeros * r, axis=-1) / 2
        r_plus += r_zero_2
        r_minus += r_zero_2

    mn = count * (count + 1.) * 0.25
    se = count * (count + 1.) * (2. * count + 1.)

    if zero_method == "pratt":
        # Pratt's method for treating zeros was just to modify the z-statistic.

        # normal approximation needs to be adjusted, see Cureton (1967)
        n_zero = i_zeros.sum(axis=-1)
        mn -= n_zero * (n_zero + 1.) * 0.25
        se -= n_zero * (n_zero + 1.) * (2. * n_zero + 1.)

        # zeros are not to be included in tie-correction.
        # any tie counts corresponding with zeros are in the 0th column
        t[i_zeros.any(axis=-1), 0] = 0

    tie_correct = (t**3 - t).sum(axis=-1)
    se -= tie_correct/2
    se = np.sqrt(se / 24)

    # se = 0 means that no non-zero values are left in d. we only need z
    # if method is asymptotic. however, if method="auto", the switch to
    # asymptotic might only happen after the statistic is calculated, so z
    # needs to be computed. in all other cases, avoid division by zero warning
    # (z is not needed anyways)
    if method in ["asymptotic", "auto"]:
        z = (r_plus - mn) / se
    else:
        z = np.nan

    return r_plus, r_minus, se, z, count, has_ties


def _correction_sign(z, alternative):
    if alternative == 'greater':
        return 1
    elif alternative == 'less':
        return -1
    else:
        return np.sign(z)


def _wilcoxon_nd(x, y=None, zero_method='wilcox', correction=True,
                 alternative='two-sided', method='auto', axis=0):

    temp = _wilcoxon_iv(x, y, zero_method, correction, alternative, method, axis)
    d, zero_method, correction, alternative, method, axis, output_z, n_zero = temp

    if d.size == 0:
        NaN = _get_nan(d)
        res = _morestats.WilcoxonResult(statistic=NaN, pvalue=NaN)
        if method == 'asymptotic':
            res.zstatistic = NaN
        return res

    r_plus, r_minus, se, z, count, has_ties = _wilcoxon_statistic(
        d, method, zero_method
    )

    # we only know if there are ties after computing the statistic and not
    # at the input validation stage. if the original method was auto and
    # the decision was to use an exact test, we override this to
    # a permutation test now (since method='exact' is not exact in the
    # presence of ties)
    if method == "auto":
        if not (has_ties or n_zero > 0):
            method = "exact"
        elif d.shape[-1] <= 13:
            # the possible outcomes to be simulated by the permutation test
            # are 2**n, where n is the sample size.
            # if n <= 13, the p-value is deterministic since 2**13 is less
            # than 9999, the default number of n_resamples
            method = stats.PermutationMethod()
        else:
            # if there are ties and the sample size is too large to
            # run a deterministic permutation test, fall back to asymptotic
            method = "asymptotic"

    if method == 'asymptotic':
        if correction:
            sign = _correction_sign(z, alternative)
            z -= sign * 0.5 / se
        p = _get_pvalue(z, _SimpleNormal(), alternative, xp=np)
    elif method == 'exact':
        dist = WilcoxonDistribution(count)
        # The null distribution in `dist` is exact only if there are no ties
        # or zeros. If there are ties or zeros, the statistic can be non-
        # integral, but the null distribution is only defined for integral
        # values of the statistic. Therefore, we're conservative: round
        # non-integral statistic up before computing CDF and down before
        # computing SF. This preserves symmetry w.r.t. alternatives and
        # order of the input arguments. See gh-19872.
        if alternative == 'less':
            p = dist.cdf(np.ceil(r_plus))
        elif alternative == 'greater':
            p = dist.sf(np.floor(r_plus))
        else:
            p = 2 * np.minimum(dist.sf(np.floor(r_plus)),
                               dist.cdf(np.ceil(r_plus)))
            p = np.clip(p, 0, 1)
    else:  # `PermutationMethod` instance (already validated)
        p = stats.permutation_test(
            (d,), lambda d: _wilcoxon_statistic(d, method, zero_method)[0],
            permutation_type='samples', **method._asdict(),
            alternative=alternative, axis=-1).pvalue

    # for backward compatibility...
    statistic = np.minimum(r_plus, r_minus) if alternative=='two-sided' else r_plus
    z = -np.abs(z) if (alternative == 'two-sided' and method == 'asymptotic') else z

    res = _morestats.WilcoxonResult(statistic=statistic, pvalue=p[()])
    if output_z:
        res.zstatistic = z[()]
    return res
<|endoftext|>import warnings
import numpy as np
from itertools import combinations, permutations, product
from collections.abc import Sequence
from dataclasses import dataclass, field
import inspect

from scipy._lib._util import (check_random_state, _rename_parameter, rng_integers,
                              _transition_to_rng)
from scipy._lib._array_api import array_namespace, is_numpy, xp_moveaxis_to_end
from scipy.special import ndtr, ndtri, comb, factorial

from ._common import ConfidenceInterval
from ._axis_nan_policy import _broadcast_concatenate, _broadcast_arrays
from ._warnings_errors import DegenerateDataWarning

__all__ = ['bootstrap', 'monte_carlo_test', 'permutation_test']


def _vectorize_statistic(statistic):
    """Vectorize an n-sample statistic"""
    # This is a little cleaner than np.nditer at the expense of some data
    # copying: concatenate samples together, then use np.apply_along_axis
    def stat_nd(*data, axis=0):
        lengths = [sample.shape[axis] for sample in data]
        split_indices = np.cumsum(lengths)[:-1]
        z = _broadcast_concatenate(data, axis)

        # move working axis to position 0 so that new dimensions in the output
        # of `statistic` are _prepended_. ("This axis is removed, and replaced
        # with new dimensions...")
        z = np.moveaxis(z, axis, 0)

        def stat_1d(z):
            data = np.split(z, split_indices)
            return statistic(*data)

        return np.apply_along_axis(stat_1d, 0, z)[()]
    return stat_nd


def _jackknife_resample(sample, batch=None):
    """Jackknife resample the sample. Only one-sample stats for now."""
    n = sample.shape[-1]
    batch_nominal = batch or n

    for k in range(0, n, batch_nominal):
        # col_start:col_end are the observations to remove
        batch_actual = min(batch_nominal, n-k)

        # jackknife - each row leaves out one observation
        j = np.ones((batch_actual, n), dtype=bool)
        np.fill_diagonal(j[:, k:k+batch_actual], False)
        i = np.arange(n)
        i = np.broadcast_to(i, (batch_actual, n))
        i = i[j].reshape((batch_actual, n-1))

        resamples = sample[..., i]
        yield resamples


def _bootstrap_resample(sample, n_resamples=None, rng=None):
    """Bootstrap resample the sample."""
    n = sample.shape[-1]

    # bootstrap - each row is a random resample of original observations
    i = rng_integers(rng, 0, n, (n_resamples, n))

    resamples = sample[..., i]
    return resamples


def _percentile_of_score(a, score, axis):
    """Vectorized, simplified `scipy.stats.percentileofscore`.
    Uses logic of the 'mean' value of percentileofscore's kind parameter.

    Unlike `stats.percentileofscore`, the percentile returned is a fraction
    in [0, 1].
    """
    B = a.shape[axis]
    return ((a < score).sum(axis=axis) + (a <= score).sum(axis=axis)) / (2 * B)


def _percentile_along_axis(theta_hat_b, alpha):
    """`np.percentile` with different percentile for each slice."""
    # the difference between _percentile_along_axis and np.percentile is that
    # np.percentile gets _all_ the qs for each axis slice, whereas
    # _percentile_along_axis gets the q corresponding with each axis slice
    shape = theta_hat_b.shape[:-1]
    alpha = np.broadcast_to(alpha, shape)
    percentiles = np.zeros_like(alpha, dtype=np.float64)
    for indices, alpha_i in np.ndenumerate(alpha):
        if np.isnan(alpha_i):
            # e.g. when bootstrap distribution has only one unique element
            msg = (
                "The BCa confidence interval cannot be calculated."
                " This problem is known to occur when the distribution"
                " is degenerate or the statistic is np.min."
            )
            warnings.warn(DegenerateDataWarning(msg), stacklevel=3)
            percentiles[indices] = np.nan
        else:
            theta_hat_b_i = theta_hat_b[indices]
            percentiles[indices] = np.percentile(theta_hat_b_i, alpha_i)
    return percentiles[()]  # return scalar instead of 0d array


def _bca_interval(data, statistic, axis, alpha, theta_hat_b, batch):
    """Bias-corrected and accelerated interval."""
    # closely follows [1] 14.3 and 15.4 (Eq. 15.36)

    # calculate z0_hat
    theta_hat = np.asarray(statistic(*data, axis=axis))[..., None]
    percentile = _percentile_of_score(theta_hat_b, theta_hat, axis=-1)
    z0_hat = ndtri(percentile)

    # calculate a_hat
    theta_hat_ji = []  # j is for sample of data, i is for jackknife resample
    for j, sample in enumerate(data):
        # _jackknife_resample will add an axis prior to the last axis that
        # corresponds with the different jackknife resamples. Do the same for
        # each sample of the data to ensure broadcastability. We need to
        # create a copy of the list containing the samples anyway, so do this
        # in the loop to simplify the code. This is not the bottleneck...
        samples = [np.expand_dims(sample, -2) for sample in data]
        theta_hat_i = []
        for jackknife_sample in _jackknife_resample(sample, batch):
            samples[j] = jackknife_sample
            broadcasted = _broadcast_arrays(samples, axis=-1)
            theta_hat_i.append(statistic(*broadcasted, axis=-1))
        theta_hat_ji.append(theta_hat_i)

    theta_hat_ji = [np.concatenate(theta_hat_i, axis=-1)
                    for theta_hat_i in theta_hat_ji]

    n_j = [theta_hat_i.shape[-1] for theta_hat_i in theta_hat_ji]

    theta_hat_j_dot = [theta_hat_i.mean(axis=-1, keepdims=True)
                       for theta_hat_i in theta_hat_ji]

    U_ji = [(n - 1) * (theta_hat_dot - theta_hat_i)
            for theta_hat_dot, theta_hat_i, n
            in zip(theta_hat_j_dot, theta_hat_ji, n_j)]

    nums = [(U_i**3).sum(axis=-1)/n**3 for U_i, n in zip(U_ji, n_j)]
    dens = [(U_i**2).sum(axis=-1)/n**2 for U_i, n in zip(U_ji, n_j)]
    a_hat = 1/6 * sum(nums) / sum(dens)**(3/2)

    # calculate alpha_1, alpha_2
    z_alpha = ndtri(alpha)
    z_1alpha = -z_alpha
    num1 = z0_hat + z_alpha
    alpha_1 = ndtr(z0_hat + num1/(1 - a_hat*num1))
    num2 = z0_hat + z_1alpha
    alpha_2 = ndtr(z0_hat + num2/(1 - a_hat*num2))
    return alpha_1, alpha_2, a_hat  # return a_hat for testing


def _bootstrap_iv(data, statistic, vectorized, paired, axis, confidence_level,
                  alternative, n_resamples, batch, method, bootstrap_result,
                  rng):
    """Input validation and standardization for `bootstrap`."""

    if vectorized not in {True, False, None}:
        raise ValueError("`vectorized` must be `True`, `False`, or `None`.")

    if vectorized is None:
        vectorized = 'axis' in inspect.signature(statistic).parameters

    if not vectorized:
        statistic = _vectorize_statistic(statistic)

    axis_int = int(axis)
    if axis != axis_int:
        raise ValueError("`axis` must be an integer.")

    n_samples = 0
    try:
        n_samples = len(data)
    except TypeError:
        raise ValueError("`data` must be a sequence of samples.")

    if n_samples == 0:
        raise ValueError("`data` must contain at least one sample.")

    message = ("Ignoring the dimension specified by `axis`, arrays in `data` do not "
               "have the same shape. Beginning in SciPy 1.16.0, `bootstrap` will "
               "explicitly broadcast elements of `data` to the same shape (ignoring "
               "`axis`) before performing the calculation. To avoid this warning in "
               "the meantime, ensure that all samples have the same shape (except "
               "potentially along `axis`).")
    data = [np.atleast_1d(sample) for sample in data]
    reduced_shapes = set()
    for sample in data:
        reduced_shape = list(sample.shape)
        reduced_shape.pop(axis)
        reduced_shapes.add(tuple(reduced_shape))
    if len(reduced_shapes) != 1:
        warnings.warn(message, FutureWarning, stacklevel=3)

    data_iv = []
    for sample in data:
        if sample.shape[axis_int] <= 1:
            raise ValueError("each sample in `data` must contain two or more "
                             "observations along `axis`.")
        sample = np.moveaxis(sample, axis_int, -1)
        data_iv.append(sample)

    if paired not in {True, False}:
        raise ValueError("`paired` must be `True` or `False`.")

    if paired:
        n = data_iv[0].shape[-1]
        for sample in data_iv[1:]:
            if sample.shape[-1] != n:
                message = ("When `paired is True`, all samples must have the "
                           "same length along `axis`")
                raise ValueError(message)

        # to generate the bootstrap distribution for paired-sample statistics,
        # resample the indices of the observations
        def statistic(i, axis=-1, data=data_iv, unpaired_statistic=statistic):
            data = [sample[..., i] for sample in data]
            return unpaired_statistic(*data, axis=axis)

        data_iv = [np.arange(n)]

    confidence_level_float = float(confidence_level)

    alternative = alternative.lower()
    alternatives = {'two-sided', 'less', 'greater'}
    if alternative not in alternatives:
        raise ValueError(f"`alternative` must be one of {alternatives}")

    n_resamples_int = int(n_resamples)
    if n_resamples != n_resamples_int or n_resamples_int < 0:
        raise ValueError("`n_resamples` must be a non-negative integer.")

    if batch is None:
        batch_iv = batch
    else:
        batch_iv = int(batch)
        if batch != batch_iv or batch_iv <= 0:
            raise ValueError("`batch` must be a positive integer or None.")

    methods = {'percentile', 'basic', 'bca'}
    method = method.lower()
    if method not in methods:
        raise ValueError(f"`method` must be in {methods}")

    message = "`bootstrap_result` must have attribute `bootstrap_distribution'"
    if (bootstrap_result is not None
            and not hasattr(bootstrap_result, "bootstrap_distribution")):
        raise ValueError(message)

    message = ("Either `bootstrap_result.bootstrap_distribution.size` or "
               "`n_resamples` must be positive.")
    if ((not bootstrap_result or
         not bootstrap_result.bootstrap_distribution.size)
            and n_resamples_int == 0):
        raise ValueError(message)

    rng = check_random_state(rng)

    return (data_iv, statistic, vectorized, paired, axis_int,
            confidence_level_float, alternative, n_resamples_int, batch_iv,
            method, bootstrap_result, rng)


@dataclass
class BootstrapResult:
    """Result object returned by `scipy.stats.bootstrap`.

    Attributes
    ----------
    confidence_interval : ConfidenceInterval
        The bootstrap confidence interval as an instance of
        `collections.namedtuple` with attributes `low` and `high`.
    bootstrap_distribution : ndarray
        The bootstrap distribution, that is, the value of `statistic` for
        each resample. The last dimension corresponds with the resamples
        (e.g. ``res.bootstrap_distribution.shape[-1] == n_resamples``).
    standard_error : float or ndarray
        The bootstrap standard error, that is, the sample standard
        deviation of the bootstrap distribution.

    """
    confidence_interval: ConfidenceInterval
    bootstrap_distribution: np.ndarray
    standard_error: float | np.ndarray


@_transition_to_rng('random_state')
def bootstrap(data, statistic, *, n_resamples=9999, batch=None,
              vectorized=None, paired=False, axis=0, confidence_level=0.95,
              alternative='two-sided', method='BCa', bootstrap_result=None,
              rng=None):
    r"""
    Compute a two-sided bootstrap confidence interval of a statistic.

    When `method` is ``'percentile'`` and `alternative` is ``'two-sided'``,
    a bootstrap confidence interval is computed according to the following
    procedure.

    1. Resample the data: for each sample in `data` and for each of
       `n_resamples`, take a random sample of the original sample
       (with replacement) of the same size as the original sample.

    2. Compute the bootstrap distribution of the statistic: for each set of
       resamples, compute the test statistic.

    3. Determine the confidence interval: find the interval of the bootstrap
       distribution that is

       - symmetric about the median and
       - contains `confidence_level` of the resampled statistic values.

    While the ``'percentile'`` method is the most intuitive, it is rarely
    used in practice. Two more common methods are available, ``'basic'``
    ('reverse percentile') and ``'BCa'`` ('bias-corrected and accelerated');
    they differ in how step 3 is performed.

    If the samples in `data` are  taken at random from their respective
    distributions :math:`n` times, the confidence interval returned by
    `bootstrap` will contain the true value of the statistic for those
    distributions approximately `confidence_level`:math:`\, \times \, n` times.

    Parameters
    ----------
    data : sequence of array-like
         Each element of `data` is a sample containing scalar observations from an
         underlying distribution. Elements of `data` must be broadcastable to the
         same shape (with the possible exception of the dimension specified by `axis`).

         .. versionchanged:: 1.14.0
             `bootstrap` will now emit a ``FutureWarning`` if the shapes of the
             elements of `data` are not the same (with the exception of the dimension
             specified by `axis`).
             Beginning in SciPy 1.16.0, `bootstrap` will explicitly broadcast the
             elements to the same shape (except along `axis`) before performing
             the calculation.

    statistic : callable
        Statistic for which the confidence interval is to be calculated.
        `statistic` must be a callable that accepts ``len(data)`` samples
        as separate arguments and returns the resulting statistic.
        If `vectorized` is set ``True``,
        `statistic` must also accept a keyword argument `axis` and be
        vectorized to compute the statistic along the provided `axis`.
    n_resamples : int, default: ``9999``
        The number of resamples performed to form the bootstrap distribution
        of the statistic.
    batch : int, optional
        The number of resamples to process in each vectorized call to
        `statistic`. Memory usage is O( `batch` * ``n`` ), where ``n`` is the
        sample size. Default is ``None``, in which case ``batch = n_resamples``
        (or ``batch = max(n_resamples, n)`` for ``method='BCa'``).
    vectorized : bool, optional
        If `vectorized` is set ``False``, `statistic` will not be passed
        keyword argument `axis` and is expected to calculate the statistic
        only for 1D samples. If ``True``, `statistic` will be passed keyword
        argument `axis` and is expected to calculate the statistic along `axis`
        when passed an ND sample array. If ``None`` (default), `vectorized`
        will be set ``True`` if ``axis`` is a parameter of `statistic`. Use of
        a vectorized statistic typically reduces computation time.
    paired : bool, default: ``False``
        Whether the statistic treats corresponding elements of the samples
        in `data` as paired. If True, `bootstrap` resamples an array of
        *indices* and uses the same indices for all arrays in `data`; otherwise,
        `bootstrap` independently resamples the elements of each array.
    axis : int, default: ``0``
        The axis of the samples in `data` along which the `statistic` is
        calculated.
    confidence_level : float, default: ``0.95``
        The confidence level of the confidence interval.
    alternative : {'two-sided', 'less', 'greater'}, default: ``'two-sided'``
        Choose ``'two-sided'`` (default) for a two-sided confidence interval,
        ``'less'`` for a one-sided confidence interval with the lower bound
        at ``-np.inf``, and ``'greater'`` for a one-sided confidence interval
        with the upper bound at ``np.inf``. The other bound of the one-sided
        confidence intervals is the same as that of a two-sided confidence
        interval with `confidence_level` twice as far from 1.0; e.g. the upper
        bound of a 95% ``'less'``  confidence interval is the same as the upper
        bound of a 90% ``'two-sided'`` confidence interval.
    method : {'percentile', 'basic', 'bca'}, default: ``'BCa'``
        Whether to return the 'percentile' bootstrap confidence interval
        (``'percentile'``), the 'basic' (AKA 'reverse') bootstrap confidence
        interval (``'basic'``), or the bias-corrected and accelerated bootstrap
        confidence interval (``'BCa'``).
    bootstrap_result : BootstrapResult, optional
        Provide the result object returned by a previous call to `bootstrap`
        to include the previous bootstrap distribution in the new bootstrap
        distribution. This can be used, for example, to change
        `confidence_level`, change `method`, or see the effect of performing
        additional resampling without repeating computations.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.

    Returns
    -------
    res : BootstrapResult
        An object with attributes:

        confidence_interval : ConfidenceInterval
            The bootstrap confidence interval as an instance of
            `collections.namedtuple` with attributes `low` and `high`.
        bootstrap_distribution : ndarray
            The bootstrap distribution, that is, the value of `statistic` for
            each resample. The last dimension corresponds with the resamples
            (e.g. ``res.bootstrap_distribution.shape[-1] == n_resamples``).
        standard_error : float or ndarray
            The bootstrap standard error, that is, the sample standard
            deviation of the bootstrap distribution.

    Warns
    -----
    `~scipy.stats.DegenerateDataWarning`
        Generated when ``method='BCa'`` and the bootstrap distribution is
        degenerate (e.g. all elements are identical).

    Notes
    -----
    Elements of the confidence interval may be NaN for ``method='BCa'`` if
    the bootstrap distribution is degenerate (e.g. all elements are identical).
    In this case, consider using another `method` or inspecting `data` for
    indications that other analysis may be more appropriate (e.g. all
    observations are identical).

    References
    ----------
    .. [1] B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap,
       Chapman & Hall/CRC, Boca Raton, FL, USA (1993)
    .. [2] Nathaniel E. Helwig, "Bootstrap Confidence Intervals",
       http://users.stat.umn.edu/~helwig/notes/bootci-Notes.pdf
    .. [3] Bootstrapping (statistics), Wikipedia,
       https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29

    Examples
    --------
    Suppose we have sampled data from an unknown distribution.

    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> from scipy.stats import norm
    >>> dist = norm(loc=2, scale=4)  # our "unknown" distribution
    >>> data = dist.rvs(size=100, random_state=rng)

    We are interested in the standard deviation of the distribution.

    >>> std_true = dist.std()      # the true value of the statistic
    >>> print(std_true)
    4.0
    >>> std_sample = np.std(data)  # the sample statistic
    >>> print(std_sample)
    3.9460644295563863

    The bootstrap is used to approximate the variability we would expect if we
    were to repeatedly sample from the unknown distribution and calculate the
    statistic of the sample each time. It does this by repeatedly resampling
    values *from the original sample* with replacement and calculating the
    statistic of each resample. This results in a "bootstrap distribution" of
    the statistic.

    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import bootstrap
    >>> data = (data,)  # samples must be in a sequence
    >>> res = bootstrap(data, np.std, confidence_level=0.9, rng=rng)
    >>> fig, ax = plt.subplots()
    >>> ax.hist(res.bootstrap_distribution, bins=25)
    >>> ax.set_title('Bootstrap Distribution')
    >>> ax.set_xlabel('statistic value')
    >>> ax.set_ylabel('frequency')
    >>> plt.show()

    The standard error quantifies this variability. It is calculated as the
    standard deviation of the bootstrap distribution.

    >>> res.standard_error
    0.24427002125829136
    >>> res.standard_error == np.std(res.bootstrap_distribution, ddof=1)
    True

    The bootstrap distribution of the statistic is often approximately normal
    with scale equal to the standard error.

    >>> x = np.linspace(3, 5)
    >>> pdf = norm.pdf(x, loc=std_sample, scale=res.standard_error)
    >>> fig, ax = plt.subplots()
    >>> ax.hist(res.bootstrap_distribution, bins=25, density=True)
    >>> ax.plot(x, pdf)
    >>> ax.set_title('Normal Approximation of the Bootstrap Distribution')
    >>> ax.set_xlabel('statistic value')
    >>> ax.set_ylabel('pdf')
    >>> plt.show()

    This suggests that we could construct a 90% confidence interval on the
    statistic based on quantiles of this normal distribution.

    >>> norm.interval(0.9, loc=std_sample, scale=res.standard_error)
    (3.5442759991341726, 4.3478528599786)

    Due to central limit theorem, this normal approximation is accurate for a
    variety of statistics and distributions underlying the samples; however,
    the approximation is not reliable in all cases. Because `bootstrap` is
    designed to work with arbitrary underlying distributions and statistics,
    it uses more advanced techniques to generate an accurate confidence
    interval.

    >>> print(res.confidence_interval)
    ConfidenceInterval(low=3.57655333533867, high=4.382043696342881)

    If we sample from the original distribution 100 times and form a bootstrap
    confidence interval for each sample, the confidence interval
    contains the true value of the statistic approximately 90% of the time.

    >>> n_trials = 100
    >>> ci_contains_true_std = 0
    >>> for i in range(n_trials):
    ...    data = (dist.rvs(size=100, random_state=rng),)
    ...    res = bootstrap(data, np.std, confidence_level=0.9,
    ...                    n_resamples=999, rng=rng)
    ...    ci = res.confidence_interval
    ...    if ci[0] < std_true < ci[1]:
    ...        ci_contains_true_std += 1
    >>> print(ci_contains_true_std)
    88

    Rather than writing a loop, we can also determine the confidence intervals
    for all 100 samples at once.

    >>> data = (dist.rvs(size=(n_trials, 100), random_state=rng),)
    >>> res = bootstrap(data, np.std, axis=-1, confidence_level=0.9,
    ...                 n_resamples=999, rng=rng)
    >>> ci_l, ci_u = res.confidence_interval

    Here, `ci_l` and `ci_u` contain the confidence interval for each of the
    ``n_trials = 100`` samples.

    >>> print(ci_l[:5])
    [3.86401283 3.33304394 3.52474647 3.54160981 3.80569252]
    >>> print(ci_u[:5])
    [4.80217409 4.18143252 4.39734707 4.37549713 4.72843584]

    And again, approximately 90% contain the true value, ``std_true = 4``.

    >>> print(np.sum((ci_l < std_true) & (std_true < ci_u)))
    93

    `bootstrap` can also be used to estimate confidence intervals of
    multi-sample statistics. For example, to get a confidence interval
    for the difference between means, we write a function that accepts
    two sample arguments and returns only the statistic. The use of the
    ``axis`` argument ensures that all mean calculations are perform in
    a single vectorized call, which is faster than looping over pairs
    of resamples in Python.

    >>> def my_statistic(sample1, sample2, axis=-1):
    ...     mean1 = np.mean(sample1, axis=axis)
    ...     mean2 = np.mean(sample2, axis=axis)
    ...     return mean1 - mean2

    Here, we use the 'percentile' method with the default 95% confidence level.

    >>> sample1 = norm.rvs(scale=1, size=100, random_state=rng)
    >>> sample2 = norm.rvs(scale=2, size=100, random_state=rng)
    >>> data = (sample1, sample2)
    >>> res = bootstrap(data, my_statistic, method='basic', rng=rng)
    >>> print(my_statistic(sample1, sample2))
    0.16661030792089523
    >>> print(res.confidence_interval)
    ConfidenceInterval(low=-0.29087973240818693, high=0.6371338699912273)

    The bootstrap estimate of the standard error is also available.

    >>> print(res.standard_error)
    0.238323948262459

    Paired-sample statistics work, too. For example, consider the Pearson
    correlation coefficient.

    >>> from scipy.stats import pearsonr
    >>> n = 100
    >>> x = np.linspace(0, 10, n)
    >>> y = x + rng.uniform(size=n)
    >>> print(pearsonr(x, y)[0])  # element 0 is the statistic
    0.9954306665125647

    We wrap `pearsonr` so that it returns only the statistic, ensuring
    that we use the `axis` argument because it is available.

    >>> def my_statistic(x, y, axis=-1):
    ...     return pearsonr(x, y, axis=axis)[0]

    We call `bootstrap` using ``paired=True``.

    >>> res = bootstrap((x, y), my_statistic, paired=True, rng=rng)
    >>> print(res.confidence_interval)
    ConfidenceInterval(low=0.9941504301315878, high=0.996377412215445)

    The result object can be passed back into `bootstrap` to perform additional
    resampling:

    >>> len(res.bootstrap_distribution)
    9999
    >>> res = bootstrap((x, y), my_statistic, paired=True,
    ...                 n_resamples=1000, rng=rng,
    ...                 bootstrap_result=res)
    >>> len(res.bootstrap_distribution)
    10999

    or to change the confidence interval options:

    >>> res2 = bootstrap((x, y), my_statistic, paired=True,
    ...                  n_resamples=0, rng=rng, bootstrap_result=res,
    ...                  method='percentile', confidence_level=0.9)
    >>> np.testing.assert_equal(res2.bootstrap_distribution,
    ...                         res.bootstrap_distribution)
    >>> res.confidence_interval
    ConfidenceInterval(low=0.9941574828235082, high=0.9963781698210212)

    without repeating computation of the original bootstrap distribution.

    """
    # Input validation
    args = _bootstrap_iv(data, statistic, vectorized, paired, axis,
                         confidence_level, alternative, n_resamples, batch,
                         method, bootstrap_result, rng)
    (data, statistic, vectorized, paired, axis, confidence_level,
     alternative, n_resamples, batch, method, bootstrap_result,
     rng) = args

    theta_hat_b = ([] if bootstrap_result is None
                   else [bootstrap_result.bootstrap_distribution])

    batch_nominal = batch or n_resamples or 1

    for k in range(0, n_resamples, batch_nominal):
        batch_actual = min(batch_nominal, n_resamples-k)
        # Generate resamples
        resampled_data = []
        for sample in data:
            resample = _bootstrap_resample(sample, n_resamples=batch_actual,
                                           rng=rng)
            resampled_data.append(resample)

        # Compute bootstrap distribution of statistic
        theta_hat_b.append(statistic(*resampled_data, axis=-1))
    theta_hat_b = np.concatenate(theta_hat_b, axis=-1)

    # Calculate percentile interval
    alpha = ((1 - confidence_level)/2 if alternative == 'two-sided'
             else (1 - confidence_level))
    if method == 'bca':
        interval = _bca_interval(data, statistic, axis=-1, alpha=alpha,
                                 theta_hat_b=theta_hat_b, batch=batch)[:2]
        percentile_fun = _percentile_along_axis
    else:
        interval = alpha, 1-alpha

        def percentile_fun(a, q):
            return np.percentile(a=a, q=q, axis=-1)

    # Calculate confidence interval of statistic
    ci_l = percentile_fun(theta_hat_b, interval[0]*100)
    ci_u = percentile_fun(theta_hat_b, interval[1]*100)
    if method == 'basic':  # see [3]
        theta_hat = statistic(*data, axis=-1)
        ci_l, ci_u = 2*theta_hat - ci_u, 2*theta_hat - ci_l

    if alternative == 'less':
        ci_l = np.full_like(ci_l, -np.inf)
    elif alternative == 'greater':
        ci_u = np.full_like(ci_u, np.inf)

    return BootstrapResult(confidence_interval=ConfidenceInterval(ci_l, ci_u),
                           bootstrap_distribution=theta_hat_b,
                           standard_error=np.std(theta_hat_b, ddof=1, axis=-1))


def _monte_carlo_test_iv(data, rvs, statistic, vectorized, n_resamples,
                         batch, alternative, axis):
    """Input validation for `monte_carlo_test`."""
    axis_int = int(axis)
    if axis != axis_int:
        raise ValueError("`axis` must be an integer.")

    if vectorized not in {True, False, None}:
        raise ValueError("`vectorized` must be `True`, `False`, or `None`.")

    if not isinstance(rvs, Sequence):
        rvs = (rvs,)
        data = (data,)
    for rvs_i in rvs:
        if not callable(rvs_i):
            raise TypeError("`rvs` must be callable or sequence of callables.")

    # At this point, `data` should be a sequence
    # If it isn't, the user passed a sequence for `rvs` but not `data`
    message = "If `rvs` is a sequence, `len(rvs)` must equal `len(data)`."
    try:
        len(data)
    except TypeError as e:
        raise ValueError(message) from e
    if not len(rvs) == len(data):
        raise ValueError(message)

    if not callable(statistic):
        raise TypeError("`statistic` must be callable.")

    if vectorized is None:
        try:
            signature = inspect.signature(statistic).parameters
        except ValueError as e:
            message = (f"Signature inspection of {statistic=} failed; "
                       "pass `vectorize` explicitly.")
            raise ValueError(message) from e
        vectorized = 'axis' in signature

    xp = array_namespace(*data)

    if not vectorized:
        if is_numpy(xp):
            statistic_vectorized = _vectorize_statistic(statistic)
        else:
            message = ("`statistic` must be vectorized (i.e. support an `axis` "
                       f"argument) when `data` contains {xp.__name__} arrays.")
            raise ValueError(message)
    else:
        statistic_vectorized = statistic

    data = _broadcast_arrays(data, axis, xp=xp)
    data_iv = []
    for sample in data:
        sample = xp.broadcast_to(sample, (1,)) if sample.ndim == 0 else sample
        sample = xp_moveaxis_to_end(sample, axis_int, xp=xp)
        data_iv.append(sample)

    n_resamples_int = int(n_resamples)
    if n_resamples != n_resamples_int or n_resamples_int <= 0:
        raise ValueError("`n_resamples` must be a positive integer.")

    if batch is None:
        batch_iv = batch
    else:
        batch_iv = int(batch)
        if batch != batch_iv or batch_iv <= 0:
            raise ValueError("`batch` must be a positive integer or None.")

    alternatives = {'two-sided', 'greater', 'less'}
    alternative = alternative.lower()
    if alternative not in alternatives:
        raise ValueError(f"`alternative` must be in {alternatives}")

    # Infer the desired p-value dtype based on the input types
    min_float = getattr(xp, 'float16', xp.float32)
    dtype = xp.result_type(*data_iv, min_float)

    return (data_iv, rvs, statistic_vectorized, vectorized, n_resamples_int,
            batch_iv, alternative, axis_int, dtype, xp)


@dataclass
class MonteCarloTestResult:
    """Result object returned by `scipy.stats.monte_carlo_test`.

    Attributes
    ----------
    statistic : float or ndarray
        The observed test statistic of the sample.
    pvalue : float or ndarray
        The p-value for the given alternative.
    null_distribution : ndarray
        The values of the test statistic generated under the null
        hypothesis.
    """
    statistic: float | np.ndarray
    pvalue: float | np.ndarray
    null_distribution: np.ndarray


@_rename_parameter('sample', 'data')
def monte_carlo_test(data, rvs, statistic, *, vectorized=None,
                     n_resamples=9999, batch=None, alternative="two-sided",
                     axis=0):
    r"""Perform a Monte Carlo hypothesis test.

    `data` contains a sample or a sequence of one or more samples. `rvs`
    specifies the distribution(s) of the sample(s) in `data` under the null
    hypothesis. The value of `statistic` for the given `data` is compared
    against a Monte Carlo null distribution: the value of the statistic for
    each of `n_resamples` sets of samples generated using `rvs`. This gives
    the p-value, the probability of observing such an extreme value of the
    test statistic under the null hypothesis.

    Parameters
    ----------
    data : array-like or sequence of array-like
        An array or sequence of arrays of observations.
    rvs : callable or tuple of callables
        A callable or sequence of callables that generates random variates
        under the null hypothesis. Each element of `rvs` must be a callable
        that accepts keyword argument ``size`` (e.g. ``rvs(size=(m, n))``) and
        returns an N-d array sample of that shape. If `rvs` is a sequence, the
        number of callables in `rvs` must match the number of samples in
        `data`, i.e. ``len(rvs) == len(data)``. If `rvs` is a single callable,
        `data` is treated as a single sample.
    statistic : callable
        Statistic for which the p-value of the hypothesis test is to be
        calculated. `statistic` must be a callable that accepts a sample
        (e.g. ``statistic(sample)``) or ``len(rvs)`` separate samples (e.g.
        ``statistic(samples1, sample2)`` if `rvs` contains two callables and
        `data` contains two samples) and returns the resulting statistic.
        If `vectorized` is set ``True``, `statistic` must also accept a keyword
        argument `axis` and be vectorized to compute the statistic along the
        provided `axis` of the samples in `data`.
    vectorized : bool, optional
        If `vectorized` is set ``False``, `statistic` will not be passed
        keyword argument `axis` and is expected to calculate the statistic
        only for 1D samples. If ``True``, `statistic` will be passed keyword
        argument `axis` and is expected to calculate the statistic along `axis`
        when passed ND sample arrays. If ``None`` (default), `vectorized`
        will be set ``True`` if ``axis`` is a parameter of `statistic`. Use of
        a vectorized statistic typically reduces computation time.
    n_resamples : int, default: 9999
        Number of samples drawn from each of the callables of `rvs`.
        Equivalently, the number statistic values under the null hypothesis
        used as the Monte Carlo null distribution.
    batch : int, optional
        The number of Monte Carlo samples to process in each call to
        `statistic`. Memory usage is O( `batch` * ``sample.size[axis]`` ). Default
        is ``None``, in which case `batch` equals `n_resamples`.
    alternative : {'two-sided', 'less', 'greater'}
        The alternative hypothesis for which the p-value is calculated.
        For each alternative, the p-value is defined as follows.

        - ``'greater'`` : the percentage of the null distribution that is
          greater than or equal to the observed value of the test statistic.
        - ``'less'`` : the percentage of the null distribution that is
          less than or equal to the observed value of the test statistic.
        - ``'two-sided'`` : twice the smaller of the p-values above.

    axis : int, default: 0
        The axis of `data` (or each sample within `data`) over which to
        calculate the statistic.

    Returns
    -------
    res : MonteCarloTestResult
        An object with attributes:

        statistic : float or ndarray
            The test statistic of the observed `data`.
        pvalue : float or ndarray
            The p-value for the given alternative.
        null_distribution : ndarray
            The values of the test statistic generated under the null
            hypothesis.

    .. warning::
        The p-value is calculated by counting the elements of the null
        distribution that are as extreme or more extreme than the observed
        value of the statistic. Due to the use of finite precision arithmetic,
        some statistic functions return numerically distinct values when the
        theoretical values would be exactly equal. In some cases, this could
        lead to a large error in the calculated p-value. `monte_carlo_test`
        guards against this by considering elements in the null distribution
        that are "close" (within a relative tolerance of 100 times the
        floating point epsilon of inexact dtypes) to the observed
        value of the test statistic as equal to the observed value of the
        test statistic. However, the user is advised to inspect the null
        distribution to assess whether this method of comparison is
        appropriate, and if not, calculate the p-value manually.

    References
    ----------

    .. [1] B. Phipson and G. K. Smyth. "Permutation P-values Should Never Be
       Zero: Calculating Exact P-values When Permutations Are Randomly Drawn."
       Statistical Applications in Genetics and Molecular Biology 9.1 (2010).

    Examples
    --------

    Suppose we wish to test whether a small sample has been drawn from a normal
    distribution. We decide that we will use the skew of the sample as a
    test statistic, and we will consider a p-value of 0.05 to be statistically
    significant.

    >>> import numpy as np
    >>> from scipy import stats
    >>> def statistic(x, axis):
    ...     return stats.skew(x, axis)

    After collecting our data, we calculate the observed value of the test
    statistic.

    >>> rng = np.random.default_rng()
    >>> x = stats.skewnorm.rvs(a=1, size=50, random_state=rng)
    >>> statistic(x, axis=0)
    0.12457412450240658

    To determine the probability of observing such an extreme value of the
    skewness by chance if the sample were drawn from the normal distribution,
    we can perform a Monte Carlo hypothesis test. The test will draw many
    samples at random from their normal distribution, calculate the skewness
    of each sample, and compare our original skewness against this
    distribution to determine an approximate p-value.

    >>> from scipy.stats import monte_carlo_test
    >>> # because our statistic is vectorized, we pass `vectorized=True`
    >>> rvs = lambda size: stats.norm.rvs(size=size, random_state=rng)
    >>> res = monte_carlo_test(x, rvs, statistic, vectorized=True)
    >>> print(res.statistic)
    0.12457412450240658
    >>> print(res.pvalue)
    0.7012

    The probability of obtaining a test statistic less than or equal to the
    observed value under the null hypothesis is ~70%. This is greater than
    our chosen threshold of 5%, so we cannot consider this to be significant
    evidence against the null hypothesis.

    Note that this p-value essentially matches that of
    `scipy.stats.skewtest`, which relies on an asymptotic distribution of a
    test statistic based on the sample skewness.

    >>> stats.skewtest(x).pvalue
    0.6892046027110614

    This asymptotic approximation is not valid for small sample sizes, but
    `monte_carlo_test` can be used with samples of any size.

    >>> x = stats.skewnorm.rvs(a=1, size=7, random_state=rng)
    >>> # stats.skewtest(x) would produce an error due to small sample
    >>> res = monte_carlo_test(x, rvs, statistic, vectorized=True)

    The Monte Carlo distribution of the test statistic is provided for
    further investigation.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.hist(res.null_distribution, bins=50)
    >>> ax.set_title("Monte Carlo distribution of test statistic")
    >>> ax.set_xlabel("Value of Statistic")
    >>> ax.set_ylabel("Frequency")
    >>> plt.show()

    """
    args = _monte_carlo_test_iv(data, rvs, statistic, vectorized,
                                n_resamples, batch, alternative, axis)
    (data, rvs, statistic, vectorized, n_resamples,
     batch, alternative, axis, dtype, xp) = args

    # Some statistics return plain floats; ensure they're at least a NumPy float
    observed = xp.asarray(statistic(*data, axis=-1))
    observed = observed[()] if observed.ndim == 0 else observed

    n_observations = [sample.shape[-1] for sample in data]
    batch_nominal = batch or n_resamples
    null_distribution = []
    for k in range(0, n_resamples, batch_nominal):
        batch_actual = min(batch_nominal, n_resamples - k)
        resamples = [rvs_i(size=(batch_actual, n_observations_i))
                     for rvs_i, n_observations_i in zip(rvs, n_observations)]
        null_distribution.append(statistic(*resamples, axis=-1))
    null_distribution = xp.concat(null_distribution)
    null_distribution = xp.reshape(null_distribution, [-1] + [1]*observed.ndim)

    # relative tolerance for detecting numerically distinct but
    # theoretically equal values in the null distribution
    eps =  (0 if not xp.isdtype(observed.dtype, ('real floating'))
            else xp.finfo(observed.dtype).eps*100)
    gamma = xp.abs(eps * observed)

    def less(null_distribution, observed):
        cmps = null_distribution <= observed + gamma
        cmps = xp.asarray(cmps, dtype=dtype)
        pvalues = (xp.sum(cmps, axis=0, dtype=dtype) + 1.) / (n_resamples + 1.)
        return pvalues

    def greater(null_distribution, observed):
        cmps = null_distribution >= observed - gamma
        cmps = xp.asarray(cmps, dtype=dtype)
        pvalues = (xp.sum(cmps, axis=0, dtype=dtype) + 1.) / (n_resamples + 1.)
        return pvalues

    def two_sided(null_distribution, observed):
        pvalues_less = less(null_distribution, observed)
        pvalues_greater = greater(null_distribution, observed)
        pvalues = xp.minimum(pvalues_less, pvalues_greater) * 2
        return pvalues

    compare = {"less": less,
               "greater": greater,
               "two-sided": two_sided}

    pvalues = compare[alternative](null_distribution, observed)
    pvalues = xp.clip(pvalues, 0., 1.)

    return MonteCarloTestResult(observed, pvalues, null_distribution)


@dataclass
class PowerResult:
    """Result object returned by `scipy.stats.power`.

    Attributes
    ----------
    power : float or ndarray
        The estimated power.
    pvalues : float or ndarray
        The simulated p-values.
    """
    power: float | np.ndarray
    pvalues: float | np.ndarray


def _wrap_kwargs(fun):
    """Wrap callable to accept arbitrary kwargs and ignore unused ones"""

    try:
        keys = set(inspect.signature(fun).parameters.keys())
    except ValueError:
        # NumPy Generator methods can't be inspected
        keys = {'size'}

    # Set keys=keys/fun=fun to avoid late binding gotcha
    def wrapped_rvs_i(*args, keys=keys, fun=fun, **all_kwargs):
        kwargs = {key: val for key, val in all_kwargs.items()
                  if key in keys}
        return fun(*args, **kwargs)
    return wrapped_rvs_i


def _power_iv(rvs, test, n_observations, significance, vectorized,
              n_resamples, batch, kwargs):
    """Input validation for `monte_carlo_test`."""

    if vectorized not in {True, False, None}:
        raise ValueError("`vectorized` must be `True`, `False`, or `None`.")

    if not isinstance(rvs, Sequence):
        rvs = (rvs,)
        n_observations = (n_observations,)
    for rvs_i in rvs:
        if not callable(rvs_i):
            raise TypeError("`rvs` must be callable or sequence of callables.")

    if not len(rvs) == len(n_observations):
        message = ("If `rvs` is a sequence, `len(rvs)` "
                   "must equal `len(n_observations)`.")
        raise ValueError(message)

    significance = np.asarray(significance)[()]
    if (not np.issubdtype(significance.dtype, np.floating)
            or np.min(significance) < 0 or np.max(significance) > 1):
        raise ValueError("`significance` must contain floats between 0 and 1.")

    kwargs = dict() if kwargs is None else kwargs
    if not isinstance(kwargs, dict):
        raise TypeError("`kwargs` must be a dictionary that maps keywords to arrays.")

    vals = kwargs.values()
    keys = kwargs.keys()

    # Wrap callables to ignore unused keyword arguments
    wrapped_rvs = [_wrap_kwargs(rvs_i) for rvs_i in rvs]

    # Broadcast, then ravel nobs/kwarg combinations. In the end,
    # `nobs` and `vals` have shape (# of combinations, number of variables)
    tmp = np.asarray(np.broadcast_arrays(*n_observations, *vals))
    shape = tmp.shape
    if tmp.ndim == 1:
        tmp = tmp[np.newaxis, :]
    else:
        tmp = tmp.reshape((shape[0], -1)).T
    nobs, vals = tmp[:, :len(rvs)], tmp[:, len(rvs):]
    nobs = nobs.astype(int)

    if not callable(test):
        raise TypeError("`test` must be callable.")

    if vectorized is None:
        vectorized = 'axis' in inspect.signature(test).parameters

    if not vectorized:
        test_vectorized = _vectorize_statistic(test)
    else:
        test_vectorized = test
    # Wrap `test` function to ignore unused kwargs
    test_vectorized = _wrap_kwargs(test_vectorized)

    n_resamples_int = int(n_resamples)
    if n_resamples != n_resamples_int or n_resamples_int <= 0:
        raise ValueError("`n_resamples` must be a positive integer.")

    if batch is None:
        batch_iv = batch
    else:
        batch_iv = int(batch)
        if batch != batch_iv or batch_iv <= 0:
            raise ValueError("`batch` must be a positive integer or None.")

    return (wrapped_rvs, test_vectorized, nobs, significance, vectorized,
            n_resamples_int, batch_iv, vals, keys, shape[1:])


def power(test, rvs, n_observations, *, significance=0.01, vectorized=None,
          n_resamples=10000, batch=None, kwargs=None):
    r"""Simulate the power of a hypothesis test under an alternative hypothesis.

    Parameters
    ----------
    test : callable
        Hypothesis test for which the power is to be simulated.
        `test` must be a callable that accepts a sample (e.g. ``test(sample)``)
        or ``len(rvs)`` separate samples (e.g. ``test(samples1, sample2)`` if
        `rvs` contains two callables and `n_observations` contains two values)
        and returns the p-value of the test.
        If `vectorized` is set to ``True``, `test` must also accept a keyword
        argument `axis` and be vectorized to perform the test along the
        provided `axis` of the samples.
        Any callable from `scipy.stats` with an `axis` argument that returns an
        object with a `pvalue` attribute is also acceptable.
    rvs : callable or tuple of callables
        A callable or sequence of callables that generate(s) random variates
        under the alternative hypothesis. Each element of `rvs` must accept
        keyword argument ``size`` (e.g. ``rvs(size=(m, n))``) and return an
        N-d array of that shape. If `rvs` is a sequence, the number of callables
        in `rvs` must match the number of elements of `n_observations`, i.e.
        ``len(rvs) == len(n_observations)``. If `rvs` is a single callable,
        `n_observations` is treated as a single element.
    n_observations : tuple of ints or tuple of integer arrays
        If a sequence of ints, each is the sizes of a sample to be passed to `test`.
        If a sequence of integer arrays, the power is simulated for each
        set of corresponding sample sizes. See Examples.
    significance : float or array_like of floats, default: 0.01
        The threshold for significance; i.e., the p-value below which the
        hypothesis test results will be considered as evidence against the null
        hypothesis. Equivalently, the acceptable rate of Type I error under
        the null hypothesis. If an array, the power is simulated for each
        significance threshold.
    kwargs : dict, optional
        Keyword arguments to be passed to `rvs` and/or `test` callables.
        Introspection is used to determine which keyword arguments may be
        passed to each callable.
        The value corresponding with each keyword must be an array.
        Arrays must be broadcastable with one another and with each array in
        `n_observations`. The power is simulated for each set of corresponding
        sample sizes and arguments. See Examples.
    vectorized : bool, optional
        If `vectorized` is set to ``False``, `test` will not be passed keyword
        argument `axis` and is expected to perform the test only for 1D samples.
        If ``True``, `test` will be passed keyword argument `axis` and is
        expected to perform the test along `axis` when passed N-D sample arrays.
        If ``None`` (default), `vectorized` will be set ``True`` if ``axis`` is
        a parameter of `test`. Use of a vectorized test typically reduces
        computation time.
    n_resamples : int, default: 10000
        Number of samples drawn from each of the callables of `rvs`.
        Equivalently, the number tests performed under the alternative
        hypothesis to approximate the power.
    batch : int, optional
        The number of samples to process in each call to `test`. Memory usage is
        proportional to the product of `batch` and the largest sample size. Default
        is ``None``, in which case `batch` equals `n_resamples`.

    Returns
    -------
    res : PowerResult
        An object with attributes:

        power : float or ndarray
            The estimated power against the alternative.
        pvalues : ndarray
            The p-values observed under the alternative hypothesis.

    Notes
    -----
    The power is simulated as follows:

    - Draw many random samples (or sets of samples), each of the size(s)
      specified by `n_observations`, under the alternative specified by
      `rvs`.
    - For each sample (or set of samples), compute the p-value according to
      `test`. These p-values are recorded in the ``pvalues`` attribute of
      the result object.
    - Compute the proportion of p-values that are less than the `significance`
      level. This is the power recorded in the ``power`` attribute of the
      result object.

    Suppose that `significance` is an array with shape ``shape1``, the elements
    of `kwargs` and `n_observations` are mutually broadcastable to shape ``shape2``,
    and `test` returns an array of p-values of shape ``shape3``. Then the result
    object ``power`` attribute will be of shape ``shape1 + shape2 + shape3``, and
    the ``pvalues`` attribute will be of shape ``shape2 + shape3 + (n_resamples,)``.

    Examples
    --------
    Suppose we wish to simulate the power of the independent sample t-test
    under the following conditions:

    - The first sample has 10 observations drawn from a normal distribution
      with mean 0.
    - The second sample has 12 observations drawn from a normal distribution
      with mean 1.0.
    - The threshold on p-values for significance is 0.05.

    >>> import numpy as np
    >>> from scipy import stats
    >>> rng = np.random.default_rng(2549598345528)
    >>>
    >>> test = stats.ttest_ind
    >>> n_observations = (10, 12)
    >>> rvs1 = rng.normal
    >>> rvs2 = lambda size: rng.normal(loc=1, size=size)
    >>> rvs = (rvs1, rvs2)
    >>> res = stats.power(test, rvs, n_observations, significance=0.05)
    >>> res.power
    0.6116

    With samples of size 10 and 12, respectively, the power of the t-test
    with a significance threshold of 0.05 is approximately 60% under the chosen
    alternative. We can investigate the effect of sample size on the power
    by passing sample size arrays.

    >>> import matplotlib.pyplot as plt
    >>> nobs_x = np.arange(5, 21)
    >>> nobs_y = nobs_x
    >>> n_observations = (nobs_x, nobs_y)
    >>> res = stats.power(test, rvs, n_observations, significance=0.05)
    >>> ax = plt.subplot()
    >>> ax.plot(nobs_x, res.power)
    >>> ax.set_xlabel('Sample Size')
    >>> ax.set_ylabel('Simulated Power')
    >>> ax.set_title('Simulated Power of `ttest_ind` with Equal Sample Sizes')
    >>> plt.show()

    Alternatively, we can investigate the impact that effect size has on the power.
    In this case, the effect size is the location of the distribution underlying
    the second sample.

    >>> n_observations = (10, 12)
    >>> loc = np.linspace(0, 1, 20)
    >>> rvs2 = lambda size, loc: rng.normal(loc=loc, size=size)
    >>> rvs = (rvs1, rvs2)
    >>> res = stats.power(test, rvs, n_observations, significance=0.05,
    ...                   kwargs={'loc': loc})
    >>> ax = plt.subplot()
    >>> ax.plot(loc, res.power)
    >>> ax.set_xlabel('Effect Size')
    >>> ax.set_ylabel('Simulated Power')
    >>> ax.set_title('Simulated Power of `ttest_ind`, Varying Effect Size')
    >>> plt.show()

    We can also use `power` to estimate the Type I error rate (also referred to by the
    ambiguous term "size") of a test and assess whether it matches the nominal level.
    For example, the null hypothesis of `jarque_bera` is that the sample was drawn from
    a distribution with the same skewness and kurtosis as the normal distribution. To
    estimate the Type I error rate, we can consider the null hypothesis to be a true
    *alternative* hypothesis and calculate the power.

    >>> test = stats.jarque_bera
    >>> n_observations = 10
    >>> rvs = rng.normal
    >>> significance = np.linspace(0.0001, 0.1, 1000)
    >>> res = stats.power(test, rvs, n_observations, significance=significance)
    >>> size = res.power

    As shown below, the Type I error rate of the test is far below the nominal level
    for such a small sample, as mentioned in its documentation.

    >>> ax = plt.subplot()
    >>> ax.plot(significance, size)
    >>> ax.plot([0, 0.1], [0, 0.1], '--')
    >>> ax.set_xlabel('nominal significance level')
    >>> ax.set_ylabel('estimated test size (Type I error rate)')
    >>> ax.set_title('Estimated test size vs nominal significance level')
    >>> ax.set_aspect('equal', 'box')
    >>> ax.legend(('`ttest_1samp`', 'ideal test'))
    >>> plt.show()

    As one might expect from such a conservative test, the power is quite low with
    respect to some alternatives. For example, the power of the test under the
    alternative that the sample was drawn from the Laplace distribution may not
    be much greater than the Type I error rate.

    >>> rvs = rng.laplace
    >>> significance = np.linspace(0.0001, 0.1, 1000)
    >>> res = stats.power(test, rvs, n_observations, significance=0.05)
    >>> print(res.power)
    0.0587

    This is not a mistake in SciPy's implementation; it is simply due to the fact
    that the null distribution of the test statistic is derived under the assumption
    that the sample size is large (i.e. approaches infinity), and this asymptotic
    approximation is not accurate for small samples. In such cases, resampling
    and Monte Carlo methods (e.g. `permutation_test`, `goodness_of_fit`,
    `monte_carlo_test`) may be more appropriate.

    """
    tmp = _power_iv(rvs, test, n_observations, significance,
                    vectorized, n_resamples, batch, kwargs)
    (rvs, test, nobs, significance,
     vectorized, n_resamples, batch, args, kwds, shape)= tmp

    batch_nominal = batch or n_resamples
    pvalues = []  # results of various nobs/kwargs combinations
    for nobs_i, args_i in zip(nobs, args):
        kwargs_i = dict(zip(kwds, args_i))
        pvalues_i = []  # results of batches; fixed nobs/kwargs combination
        for k in range(0, n_resamples, batch_nominal):
            batch_actual = min(batch_nominal, n_resamples - k)
            resamples = [rvs_j(size=(batch_actual, nobs_ij), **kwargs_i)
                         for rvs_j, nobs_ij in zip(rvs, nobs_i)]
            res = test(*resamples, **kwargs_i, axis=-1)
            p = getattr(res, 'pvalue', res)
            pvalues_i.append(p)
        # Concatenate results from batches
        pvalues_i = np.concatenate(pvalues_i, axis=-1)
        pvalues.append(pvalues_i)
    # `test` can return result with array of p-values
    shape += pvalues_i.shape[:-1]
    # Concatenate results from various nobs/kwargs combinations
    pvalues = np.concatenate(pvalues, axis=0)
    # nobs/kwargs arrays were raveled to single axis; unravel
    pvalues = pvalues.reshape(shape + (-1,))
    if significance.ndim > 0:
        newdims = tuple(range(significance.ndim, pvalues.ndim + significance.ndim))
        significance = np.expand_dims(significance, newdims)
    powers = np.mean(pvalues < significance, axis=-1)

    return PowerResult(power=powers, pvalues=pvalues)


@dataclass
class PermutationTestResult:
    """Result object returned by `scipy.stats.permutation_test`.

    Attributes
    ----------
    statistic : float or ndarray
        The observed test statistic of the data.
    pvalue : float or ndarray
        The p-value for the given alternative.
    null_distribution : ndarray
        The values of the test statistic generated under the null
        hypothesis.
    """
    statistic: float | np.ndarray
    pvalue: float | np.ndarray
    null_distribution: np.ndarray


def _all_partitions_concatenated(ns):
    """
    Generate all partitions of indices of groups of given sizes, concatenated

    `ns` is an iterable of ints.
    """
    def all_partitions(z, n):
        for c in combinations(z, n):
            x0 = set(c)
            x1 = z - x0
            yield [x0, x1]

    def all_partitions_n(z, ns):
        if len(ns) == 0:
            yield [z]
            return
        for c in all_partitions(z, ns[0]):
            for d in all_partitions_n(c[1], ns[1:]):
                yield c[0:1] + d

    z = set(range(np.sum(ns)))
    for partitioning in all_partitions_n(z, ns[:]):
        x = np.concatenate([list(partition)
                            for partition in partitioning]).astype(int)
        yield x


def _batch_generator(iterable, batch):
    """A generator that yields batches of elements from an iterable"""
    iterator = iter(iterable)
    if batch <= 0:
        raise ValueError("`batch` must be positive.")
    z = [item for i, item in zip(range(batch), iterator)]
    while z:  # we don't want StopIteration without yielding an empty list
        yield z
        z = [item for i, item in zip(range(batch), iterator)]


def _pairings_permutations_gen(n_permutations, n_samples, n_obs_sample, batch,
                               rng):
    # Returns a generator that yields arrays of size
    # `(batch, n_samples, n_obs_sample)`.
    # Each row is an independent permutation of indices 0 to `n_obs_sample`.
    batch = min(batch, n_permutations)

    if hasattr(rng, 'permuted'):
        def batched_perm_generator():
            indices = np.arange(n_obs_sample)
            indices = np.tile(indices, (batch, n_samples, 1))
            for k in range(0, n_permutations, batch):
                batch_actual = min(batch, n_permutations-k)
                # Don't permute in place, otherwise results depend on `batch`
                permuted_indices = rng.permuted(indices, axis=-1)
                yield permuted_indices[:batch_actual]
    else:  # RandomState and early Generators don't have `permuted`
        def batched_perm_generator():
            for k in range(0, n_permutations, batch):
                batch_actual = min(batch, n_permutations-k)
                size = (batch_actual, n_samples, n_obs_sample)
                x = rng.random(size=size)
                yield np.argsort(x, axis=-1)[:batch_actual]

    return batched_perm_generator()


def _calculate_null_both(data, statistic, n_permutations, batch,
                         rng=None):
    """
    Calculate null distribution for independent sample tests.
    """
    n_samples = len(data)

    # compute number of permutations
    # (distinct partitions of data into samples of these sizes)
    n_obs_i = [sample.shape[-1] for sample in data]  # observations per sample
    n_obs_ic = np.cumsum(n_obs_i)
    n_obs = n_obs_ic[-1]  # total number of observations
    n_max = np.prod([comb(n_obs_ic[i], n_obs_ic[i-1])
                     for i in range(n_samples-1, 0, -1)])

    # perm_generator is an iterator that produces permutations of indices
    # from 0 to n_obs. We'll concatenate the samples, use these indices to
    # permute the data, then split the samples apart again.
    if n_permutations >= n_max:
        exact_test = True
        n_permutations = n_max
        perm_generator = _all_partitions_concatenated(n_obs_i)
    else:
        exact_test = False
        # Neither RandomState.permutation nor Generator.permutation
        # can permute axis-slices independently. If this feature is
        # added in the future, batches of the desired size should be
        # generated in a single call.
        perm_generator = (rng.permutation(n_obs)
                          for i in range(n_permutations))

    batch = batch or int(n_permutations)
    null_distribution = []

    # First, concatenate all the samples. In batches, permute samples with
    # indices produced by the `perm_generator`, split them into new samples of
    # the original sizes, compute the statistic for each batch, and add these
    # statistic values to the null distribution.
    data = np.concatenate(data, axis=-1)
    for indices in _batch_generator(perm_generator, batch=batch):
        indices = np.array(indices)

        # `indices` is 2D: each row is a permutation of the indices.
        # We use it to index `data` along its last axis, which corresponds
        # with observations.
        # After indexing, the second to last axis of `data_batch` corresponds
        # with permutations, and the last axis corresponds with observations.
        data_batch = data[..., indices]

        # Move the permutation axis to the front: we'll concatenate a list
        # of batched statistic values along this zeroth axis to form the
        # null distribution.
        data_batch = np.moveaxis(data_batch, -2, 0)
        data_batch = np.split(data_batch, n_obs_ic[:-1], axis=-1)
        null_distribution.append(statistic(*data_batch, axis=-1))
    null_distribution = np.concatenate(null_distribution, axis=0)

    return null_distribution, n_permutations, exact_test


def _calculate_null_pairings(data, statistic, n_permutations, batch,
                             rng=None):
    """
    Calculate null distribution for association tests.
    """
    n_samples = len(data)

    # compute number of permutations (factorial(n) permutations of each sample)
    n_obs_sample = data[0].shape[-1]  # observations per sample; same for each
    n_max = factorial(n_obs_sample)**n_samples

    # `perm_generator` is an iterator that produces a list of permutations of
    # indices from 0 to n_obs_sample, one for each sample.
    if n_permutations >= n_max:
        exact_test = True
        n_permutations = n_max
        batch = batch or int(n_permutations)
        # Cartesian product of the sets of all permutations of indices
        perm_generator = product(*(permutations(range(n_obs_sample))
                                   for i in range(n_samples)))
        batched_perm_generator = _batch_generator(perm_generator, batch=batch)
    else:
        exact_test = False
        batch = batch or int(n_permutations)
        # Separate random permutations of indices for each sample.
        # Again, it would be nice if RandomState/Generator.permutation
        # could permute each axis-slice separately.
        args = n_permutations, n_samples, n_obs_sample, batch, rng
        batched_perm_generator = _pairings_permutations_gen(*args)

    null_distribution = []

    for indices in batched_perm_generator:
        indices = np.array(indices)

        # `indices` is 3D: the zeroth axis is for permutations, the next is
        # for samples, and the last is for observations. Swap the first two
        # to make the zeroth axis correspond with samples, as it does for
        # `data`.
        indices = np.swapaxes(indices, 0, 1)

        # When we're done, `data_batch` will be a list of length `n_samples`.
        # Each element will be a batch of random permutations of one sample.
        # The zeroth axis of each batch will correspond with permutations,
        # and the last will correspond with observations. (This makes it
        # easy to pass into `statistic`.)
        data_batch = [None]*n_samples
        for i in range(n_samples):
            data_batch[i] = data[i][..., indices[i]]
            data_batch[i] = np.moveaxis(data_batch[i], -2, 0)

        null_distribution.append(statistic(*data_batch, axis=-1))
    null_distribution = np.concatenate(null_distribution, axis=0)

    return null_distribution, n_permutations, exact_test


def _calculate_null_samples(data, statistic, n_permutations, batch,
                            rng=None):
    """
    Calculate null distribution for paired-sample tests.
    """
    n_samples = len(data)

    # By convention, the meaning of the "samples" permutations type for
    # data with only one sample is to flip the sign of the observations.
    # Achieve this by adding a second sample - the negative of the original.
    if n_samples == 1:
        data = [data[0], -data[0]]

    # The "samples" permutation strategy is the same as the "pairings"
    # strategy except the roles of samples and observations are flipped.
    # So swap these axes, then we'll use the function for the "pairings"
    # strategy to do all the work!
    data = np.swapaxes(data, 0, -1)

    # (Of course, the user's statistic doesn't know what we've done here,
    # so we need to pass it what it's expecting.)
    def statistic_wrapped(*data, axis):
        data = np.swapaxes(data, 0, -1)
        if n_samples == 1:
            data = data[0:1]
        return statistic(*data, axis=axis)

    return _calculate_null_pairings(data, statistic_wrapped, n_permutations,
                                    batch, rng)


def _permutation_test_iv(data, statistic, permutation_type, vectorized,
                         n_resamples, batch, alternative, axis, rng):
    """Input validation for `permutation_test`."""

    axis_int = int(axis)
    if axis != axis_int:
        raise ValueError("`axis` must be an integer.")

    permutation_types = {'samples', 'pairings', 'independent'}
    permutation_type = permutation_type.lower()
    if permutation_type not in permutation_types:
        raise ValueError(f"`permutation_type` must be in {permutation_types}.")

    if vectorized not in {True, False, None}:
        raise ValueError("`vectorized` must be `True`, `False`, or `None`.")

    if vectorized is None:
        vectorized = 'axis' in inspect.signature(statistic).parameters

    if not vectorized:
        statistic = _vectorize_statistic(statistic)

    message = "`data` must be a tuple containing at least two samples"
    try:
        if len(data) < 2 and permutation_type == 'independent':
            raise ValueError(message)
    except TypeError:
        raise TypeError(message)

    data = _broadcast_arrays(data, axis)
    data_iv = []
    for sample in data:
        sample = np.atleast_1d(sample)
        if sample.shape[axis] <= 1:
            raise ValueError("each sample in `data` must contain two or more "
                             "observations along `axis`.")
        sample = np.moveaxis(sample, axis_int, -1)
        data_iv.append(sample)

    n_resamples_int = (int(n_resamples) if not np.isinf(n_resamples)
                       else np.inf)
    if n_resamples != n_resamples_int or n_resamples_int <= 0:
        raise ValueError("`n_resamples` must be a positive integer.")

    if batch is None:
        batch_iv = batch
    else:
        batch_iv = int(batch)
        if batch != batch_iv or batch_iv <= 0:
            raise ValueError("`batch` must be a positive integer or None.")

    alternatives = {'two-sided', 'greater', 'less'}
    alternative = alternative.lower()
    if alternative not in alternatives:
        raise ValueError(f"`alternative` must be in {alternatives}")

    rng = check_random_state(rng)

    return (data_iv, statistic, permutation_type, vectorized, n_resamples_int,
            batch_iv, alternative, axis_int, rng)


@_transition_to_rng('random_state')
def permutation_test(data, statistic, *, permutation_type='independent',
                     vectorized=None, n_resamples=9999, batch=None,
                     alternative="two-sided", axis=0, rng=None):
    r"""
    Performs a permutation test of a given statistic on provided data.

    For independent sample statistics, the null hypothesis is that the data are
    randomly sampled from the same distribution.
    For paired sample statistics, two null hypothesis can be tested:
    that the data are paired at random or that the data are assigned to samples
    at random.

    Parameters
    ----------
    data : iterable of array-like
        Contains the samples, each of which is an array of observations.
        Dimensions of sample arrays must be compatible for broadcasting except
        along `axis`.
    statistic : callable
        Statistic for which the p-value of the hypothesis test is to be
        calculated. `statistic` must be a callable that accepts samples
        as separate arguments (e.g. ``statistic(*data)``) and returns the
        resulting statistic.
        If `vectorized` is set ``True``, `statistic` must also accept a keyword
        argument `axis` and be vectorized to compute the statistic along the
        provided `axis` of the sample arrays.
    permutation_type : {'independent', 'samples', 'pairings'}, optional
        The type of permutations to be performed, in accordance with the
        null hypothesis. The first two permutation types are for paired sample
        statistics, in which all samples contain the same number of
        observations and observations with corresponding indices along `axis`
        are considered to be paired; the third is for independent sample
        statistics.

        - ``'samples'`` : observations are assigned to different samples
          but remain paired with the same observations from other samples.
          This permutation type is appropriate for paired sample hypothesis
          tests such as the Wilcoxon signed-rank test and the paired t-test.
        - ``'pairings'`` : observations are paired with different observations,
          but they remain within the same sample. This permutation type is
          appropriate for association/correlation tests with statistics such
          as Spearman's :math:`\rho`, Kendall's :math:`\tau`, and Pearson's
          :math:`r`.
        - ``'independent'`` (default) : observations are assigned to different
          samples. Samples may contain different numbers of observations. This
          permutation type is appropriate for independent sample hypothesis
          tests such as the Mann-Whitney :math:`U` test and the independent
          sample t-test.

          Please see the Notes section below for more detailed descriptions
          of the permutation types.

    vectorized : bool, optional
        If `vectorized` is set ``False``, `statistic` will not be passed
        keyword argument `axis` and is expected to calculate the statistic
        only for 1D samples. If ``True``, `statistic` will be passed keyword
        argument `axis` and is expected to calculate the statistic along `axis`
        when passed an ND sample array. If ``None`` (default), `vectorized`
        will be set ``True`` if ``axis`` is a parameter of `statistic`. Use
        of a vectorized statistic typically reduces computation time.
    n_resamples : int or np.inf, default: 9999
        Number of random permutations (resamples) used to approximate the null
        distribution. If greater than or equal to the number of distinct
        permutations, the exact null distribution will be computed.
        Note that the number of distinct permutations grows very rapidly with
        the sizes of samples, so exact tests are feasible only for very small
        data sets.
    batch : int, optional
        The number of permutations to process in each call to `statistic`.
        Memory usage is O( `batch` * ``n`` ), where ``n`` is the total size
        of all samples, regardless of the value of `vectorized`. Default is
        ``None``, in which case ``batch`` is the number of permutations.
    alternative : {'two-sided', 'less', 'greater'}, optional
        The alternative hypothesis for which the p-value is calculated.
        For each alternative, the p-value is defined for exact tests as
        follows.

        - ``'greater'`` : the percentage of the null distribution that is
          greater than or equal to the observed value of the test statistic.
        - ``'less'`` : the percentage of the null distribution that is
          less than or equal to the observed value of the test statistic.
        - ``'two-sided'`` (default) : twice the smaller of the p-values above.

        Note that p-values for randomized tests are calculated according to the
        conservative (over-estimated) approximation suggested in [2]_ and [3]_
        rather than the unbiased estimator suggested in [4]_. That is, when
        calculating the proportion of the randomized null distribution that is
        as extreme as the observed value of the test statistic, the values in
        the numerator and denominator are both increased by one. An
        interpretation of this adjustment is that the observed value of the
        test statistic is always included as an element of the randomized
        null distribution.
        The convention used for two-sided p-values is not universal;
        the observed test statistic and null distribution are returned in
        case a different definition is preferred.

    axis : int, default: 0
        The axis of the (broadcasted) samples over which to calculate the
        statistic. If samples have a different number of dimensions,
        singleton dimensions are prepended to samples with fewer dimensions
        before `axis` is considered.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.

    Returns
    -------
    res : PermutationTestResult
        An object with attributes:

        statistic : float or ndarray
            The observed test statistic of the data.
        pvalue : float or ndarray
            The p-value for the given alternative.
        null_distribution : ndarray
            The values of the test statistic generated under the null
            hypothesis.

    Notes
    -----

    The three types of permutation tests supported by this function are
    described below.

    **Unpaired statistics** (``permutation_type='independent'``):

    The null hypothesis associated with this permutation type is that all
    observations are sampled from the same underlying distribution and that
    they have been assigned to one of the samples at random.

    Suppose ``data`` contains two samples; e.g. ``a, b = data``.
    When ``1 < n_resamples < binom(n, k)``, where

    * ``k`` is the number of observations in ``a``,
    * ``n`` is the total number of observations in ``a`` and ``b``, and
    * ``binom(n, k)`` is the binomial coefficient (``n`` choose ``k``),

    the data are pooled (concatenated), randomly assigned to either the first
    or second sample, and the statistic is calculated. This process is
    performed repeatedly, `permutation` times, generating a distribution of the
    statistic under the null hypothesis. The statistic of the original
    data is compared to this distribution to determine the p-value.

    When ``n_resamples >= binom(n, k)``, an exact test is performed: the data
    are *partitioned* between the samples in each distinct way exactly once,
    and the exact null distribution is formed.
    Note that for a given partitioning of the data between the samples,
    only one ordering/permutation of the data *within* each sample is
    considered. For statistics that do not depend on the order of the data
    within samples, this dramatically reduces computational cost without
    affecting the shape of the null distribution (because the frequency/count
    of each value is affected by the same factor).

    For ``a = [a1, a2, a3, a4]`` and ``b = [b1, b2, b3]``, an example of this
    permutation type is ``x = [b3, a1, a2, b2]`` and ``y = [a4, b1, a3]``.
    Because only one ordering/permutation of the data *within* each sample
    is considered in an exact test, a resampling like ``x = [b3, a1, b2, a2]``
    and ``y = [a4, a3, b1]`` would *not* be considered distinct from the
    example above.

    ``permutation_type='independent'`` does not support one-sample statistics,
    but it can be applied to statistics with more than two samples. In this
    case, if ``n`` is an array of the number of observations within each
    sample, the number of distinct partitions is::

        np.prod([binom(sum(n[i:]), sum(n[i+1:])) for i in range(len(n)-1)])

    **Paired statistics, permute pairings** (``permutation_type='pairings'``):

    The null hypothesis associated with this permutation type is that
    observations within each sample are drawn from the same underlying
    distribution and that pairings with elements of other samples are
    assigned at random.

    Suppose ``data`` contains only one sample; e.g. ``a, = data``, and we
    wish to consider all possible pairings of elements of ``a`` with elements
    of a second sample, ``b``. Let ``n`` be the number of observations in
    ``a``, which must also equal the number of observations in ``b``.

    When ``1 < n_resamples < factorial(n)``, the elements of ``a`` are
    randomly permuted. The user-supplied statistic accepts one data argument,
    say ``a_perm``, and calculates the statistic considering ``a_perm`` and
    ``b``. This process is performed repeatedly, `permutation` times,
    generating a distribution of the statistic under the null hypothesis.
    The statistic of the original data is compared to this distribution to
    determine the p-value.

    When ``n_resamples >= factorial(n)``, an exact test is performed:
    ``a`` is permuted in each distinct way exactly once. Therefore, the
    `statistic` is computed for each unique pairing of samples between ``a``
    and ``b`` exactly once.

    For ``a = [a1, a2, a3]`` and ``b = [b1, b2, b3]``, an example of this
    permutation type is ``a_perm = [a3, a1, a2]`` while ``b`` is left
    in its original order.

    ``permutation_type='pairings'`` supports ``data`` containing any number
    of samples, each of which must contain the same number of observations.
    All samples provided in ``data`` are permuted *independently*. Therefore,
    if ``m`` is the number of samples and ``n`` is the number of observations
    within each sample, then the number of permutations in an exact test is::

        factorial(n)**m

    Note that if a two-sample statistic, for example, does not inherently
    depend on the order in which observations are provided - only on the
    *pairings* of observations - then only one of the two samples should be
    provided in ``data``. This dramatically reduces computational cost without
    affecting the shape of the null distribution (because the frequency/count
    of each value is affected by the same factor).

    **Paired statistics, permute samples** (``permutation_type='samples'``):

    The null hypothesis associated with this permutation type is that
    observations within each pair are drawn from the same underlying
    distribution and that the sample to which they are assigned is random.

    Suppose ``data`` contains two samples; e.g. ``a, b = data``.
    Let ``n`` be the number of observations in ``a``, which must also equal
    the number of observations in ``b``.

    When ``1 < n_resamples < 2**n``, the elements of ``a`` are ``b`` are
    randomly swapped between samples (maintaining their pairings) and the
    statistic is calculated. This process is performed repeatedly,
    `permutation` times,  generating a distribution of the statistic under the
    null hypothesis. The statistic of the original data is compared to this
    distribution to determine the p-value.

    When ``n_resamples >= 2**n``, an exact test is performed: the observations
    are assigned to the two samples in each distinct way (while maintaining
    pairings) exactly once.

    For ``a = [a1, a2, a3]`` and ``b = [b1, b2, b3]``, an example of this
    permutation type is ``x = [b1, a2, b3]`` and ``y = [a1, b2, a3]``.

    ``permutation_type='samples'`` supports ``data`` containing any number
    of samples, each of which must contain the same number of observations.
    If ``data`` contains more than one sample, paired observations within
    ``data`` are exchanged between samples *independently*. Therefore, if ``m``
    is the number of samples and ``n`` is the number of observations within
    each sample, then the number of permutations in an exact test is::

        factorial(m)**n

    Several paired-sample statistical tests, such as the Wilcoxon signed rank
    test and paired-sample t-test, can be performed considering only the
    *difference* between two paired elements. Accordingly, if ``data`` contains
    only one sample, then the null distribution is formed by independently
    changing the *sign* of each observation.

    .. warning::
        The p-value is calculated by counting the elements of the null
        distribution that are as extreme or more extreme than the observed
        value of the statistic. Due to the use of finite precision arithmetic,
        some statistic functions return numerically distinct values when the
        theoretical values would be exactly equal. In some cases, this could
        lead to a large error in the calculated p-value. `permutation_test`
        guards against this by considering elements in the null distribution
        that are "close" (within a relative tolerance of 100 times the
        floating point epsilon of inexact dtypes) to the observed
        value of the test statistic as equal to the observed value of the
        test statistic. However, the user is advised to inspect the null
        distribution to assess whether this method of comparison is
        appropriate, and if not, calculate the p-value manually. See example
        below.

    References
    ----------

    .. [1] R. A. Fisher. The Design of Experiments, 6th Ed (1951).
    .. [2] B. Phipson and G. K. Smyth. "Permutation P-values Should Never Be
       Zero: Calculating Exact P-values When Permutations Are Randomly Drawn."
       Statistical Applications in Genetics and Molecular Biology 9.1 (2010).
    .. [3] M. D. Ernst. "Permutation Methods: A Basis for Exact Inference".
       Statistical Science (2004).
    .. [4] B. Efron and R. J. Tibshirani. An Introduction to the Bootstrap
       (1993).

    Examples
    --------

    Suppose we wish to test whether two samples are drawn from the same
    distribution. Assume that the underlying distributions are unknown to us,
    and that before observing the data, we hypothesized that the mean of the
    first sample would be less than that of the second sample. We decide that
    we will use the difference between the sample means as a test statistic,
    and we will consider a p-value of 0.05 to be statistically significant.

    For efficiency, we write the function defining the test statistic in a
    vectorized fashion: the samples ``x`` and ``y`` can be ND arrays, and the
    statistic will be calculated for each axis-slice along `axis`.

    >>> import numpy as np
    >>> def statistic(x, y, axis):
    ...     return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    After collecting our data, we calculate the observed value of the test
    statistic.

    >>> from scipy.stats import norm
    >>> rng = np.random.default_rng()
    >>> x = norm.rvs(size=5, random_state=rng)
    >>> y = norm.rvs(size=6, loc = 3, random_state=rng)
    >>> statistic(x, y, 0)
    -3.5411688580987266

    Indeed, the test statistic is negative, suggesting that the true mean of
    the distribution underlying ``x`` is less than that of the distribution
    underlying ``y``. To determine the probability of this occurring by chance
    if the two samples were drawn from the same distribution, we perform
    a permutation test.

    >>> from scipy.stats import permutation_test
    >>> # because our statistic is vectorized, we pass `vectorized=True`
    >>> # `n_resamples=np.inf` indicates that an exact test is to be performed
    >>> res = permutation_test((x, y), statistic, vectorized=True,
    ...                        n_resamples=np.inf, alternative='less')
    >>> print(res.statistic)
    -3.5411688580987266
    >>> print(res.pvalue)
    0.004329004329004329

    The probability of obtaining a test statistic less than or equal to the
    observed value under the null hypothesis is 0.4329%. This is less than our
    chosen threshold of 5%, so we consider this to be significant evidence
    against the null hypothesis in favor of the alternative.

    Because the size of the samples above was small, `permutation_test` could
    perform an exact test. For larger samples, we resort to a randomized
    permutation test.

    >>> x = norm.rvs(size=100, random_state=rng)
    >>> y = norm.rvs(size=120, loc=0.2, random_state=rng)
    >>> res = permutation_test((x, y), statistic, n_resamples=9999,
    ...                        vectorized=True, alternative='less',
    ...                        rng=rng)
    >>> print(res.statistic)
    -0.4230459671240913
    >>> print(res.pvalue)
    0.0015

    The approximate probability of obtaining a test statistic less than or
    equal to the observed value under the null hypothesis is 0.0225%. This is
    again less than our chosen threshold of 5%, so again we have significant
    evidence to reject the null hypothesis in favor of the alternative.

    For large samples and number of permutations, the result is comparable to
    that of the corresponding asymptotic test, the independent sample t-test.

    >>> from scipy.stats import ttest_ind
