    :math:`S_{x_3} = 0` whereas its total order is :math:`S_{T_{x_3}} = 0.244`.
    This means that higher order interactions with :math:`x_3` are responsible
    for the difference. Almost 25% of the observed variance
    on the QoI is due to the correlations between :math:`x_3` and :math:`x_1`,
    although :math:`x_3` by itself has no impact on the QoI.

    The following gives a visual explanation of Sobol' indices on this
    function. Let's generate 1024 samples in :math:`[-\pi, \pi]^3` and
    calculate the value of the output.

    >>> from scipy.stats import qmc
    >>> n_dim = 3
    >>> p_labels = ['$x_1$', '$x_2$', '$x_3$']
    >>> sample = qmc.Sobol(d=n_dim, seed=rng).random(1024)
    >>> sample = qmc.scale(
    ...     sample=sample,
    ...     l_bounds=[-np.pi, -np.pi, -np.pi],
    ...     u_bounds=[np.pi, np.pi, np.pi]
    ... )
    >>> output = f_ishigami(sample.T)

    Now we can do scatter plots of the output with respect to each parameter.
    This gives a visual way to understand how each parameter impacts the
    output of the function.

    >>> fig, ax = plt.subplots(1, n_dim, figsize=(12, 4))
    >>> for i in range(n_dim):
    ...     xi = sample[:, i]
    ...     ax[i].scatter(xi, output, marker='+')
    ...     ax[i].set_xlabel(p_labels[i])
    >>> ax[0].set_ylabel('Y')
    >>> plt.tight_layout()
    >>> plt.show()

    Now Sobol' goes a step further:
    by conditioning the output value by given values of the parameter
    (black lines), the conditional output mean is computed. It corresponds to
    the term :math:`\mathbb{E}(Y|x_i)`. Taking the variance of this term gives
    the numerator of the Sobol' indices.

    >>> mini = np.min(output)
    >>> maxi = np.max(output)
    >>> n_bins = 10
    >>> bins = np.linspace(-np.pi, np.pi, num=n_bins, endpoint=False)
    >>> dx = bins[1] - bins[0]
    >>> fig, ax = plt.subplots(1, n_dim, figsize=(12, 4))
    >>> for i in range(n_dim):
    ...     xi = sample[:, i]
    ...     ax[i].scatter(xi, output, marker='+')
    ...     ax[i].set_xlabel(p_labels[i])
    ...     for bin_ in bins:
    ...         idx = np.where((bin_ <= xi) & (xi <= bin_ + dx))
    ...         xi_ = xi[idx]
    ...         y_ = output[idx]
    ...         ave_y_ = np.mean(y_)
    ...         ax[i].plot([bin_ + dx/2] * 2, [mini, maxi], c='k')
    ...         ax[i].scatter(bin_ + dx/2, ave_y_, c='r')
    >>> ax[0].set_ylabel('Y')
    >>> plt.tight_layout()
    >>> plt.show()

    Looking at :math:`x_3`, the variance
    of the mean is zero leading to :math:`S_{x_3} = 0`. But we can further
    observe that the variance of the output is not constant along the parameter
    values of :math:`x_3`. This heteroscedasticity is explained by higher order
    interactions. Moreover, an heteroscedasticity is also noticeable on
    :math:`x_1` leading to an interaction between :math:`x_3` and :math:`x_1`.
    On :math:`x_2`, the variance seems to be constant and thus null interaction
    with this parameter can be supposed.

    This case is fairly simple to analyse visually---although it is only a
    qualitative analysis. Nevertheless, when the number of input parameters
    increases such analysis becomes unrealistic as it would be difficult to
    conclude on high-order terms. Hence the benefit of using Sobol' indices.

    """
    rng = check_random_state(rng)

    n_ = int(n)
    if not (n_ & (n_ - 1) == 0) or n != n_:
        raise ValueError(
            "The balance properties of Sobol' points require 'n' "
            "to be a power of 2."
        )
    n = n_

    if not callable(method):
        indices_methods = {
            "saltelli_2010": saltelli_2010,
        }
        try:
            method = method.lower()  # type: ignore[assignment]
            indices_method_ = indices_methods[method]
        except KeyError as exc:
            message = (
                f"{method!r} is not a valid 'method'. It must be one of"
                f" {set(indices_methods)!r} or a callable."
            )
            raise ValueError(message) from exc
    else:
        indices_method_ = method
        sig = inspect.signature(indices_method_)

        if set(sig.parameters) != {'f_A', 'f_B', 'f_AB'}:
            message = (
                "If 'method' is a callable, it must have the following"
                f" signature: {inspect.signature(saltelli_2010)}"
            )
            raise ValueError(message)

    def indices_method(f_A, f_B, f_AB):
        """Wrap indices method to ensure proper output dimension.

        1D when single output, 2D otherwise.
        """
        return np.squeeze(indices_method_(f_A=f_A, f_B=f_B, f_AB=f_AB))

    if callable(func):
        if dists is None:
            raise ValueError(
                "'dists' must be defined when 'func' is a callable."
            )

        def wrapped_func(x):
            return np.atleast_2d(func(x))

        A, B = sample_A_B(n=n, dists=dists, rng=rng)
        AB = sample_AB(A=A, B=B)

        f_A = wrapped_func(A)

        if f_A.shape[1] != n:
            raise ValueError(
                "'func' output should have a shape ``(s, -1)`` with ``s`` "
                "the number of output."
            )

        def funcAB(AB):
            d, d, n = AB.shape
            AB = np.moveaxis(AB, 0, -1).reshape(d, n*d)
            f_AB = wrapped_func(AB)
            return np.moveaxis(f_AB.reshape((-1, n, d)), -1, 0)

        f_B = wrapped_func(B)
        f_AB = funcAB(AB)
    else:
        message = (
            "When 'func' is a dictionary, it must contain the following "
            "keys: 'f_A', 'f_B' and 'f_AB'."
            "'f_A' and 'f_B' should have a shape ``(s, n)`` and 'f_AB' "
            "should have a shape ``(d, s, n)``."
        )
        try:
            f_A, f_B, f_AB = map(lambda arr: arr.copy(), np.atleast_2d(
                func['f_A'], func['f_B'], func['f_AB']
            ))
        except KeyError as exc:
            raise ValueError(message) from exc

        if f_A.shape[1] != n or f_A.shape != f_B.shape or \
                f_AB.shape == f_A.shape or f_AB.shape[-1] % n != 0:
            raise ValueError(message)

    # Normalization by mean
    # Sobol', I. and Levitan, Y. L. (1999). On the use of variance reducing
    # multipliers in monte carlo computations of a global sensitivity index.
    # Computer Physics Communications, 117(1) :52-61.
    mean = np.mean([f_A, f_B], axis=(0, -1)).reshape(-1, 1)
    f_A -= mean
    f_B -= mean
    f_AB -= mean

    # Compute indices
    # Filter warnings for constant output as var = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        first_order, total_order = indices_method(f_A=f_A, f_B=f_B, f_AB=f_AB)

    # null variance means null indices
    first_order[~np.isfinite(first_order)] = 0
    total_order[~np.isfinite(total_order)] = 0

    res = dict(
        first_order=first_order,
        total_order=total_order,
        _indices_method=indices_method,
        _f_A=f_A,
        _f_B=f_B,
        _f_AB=f_AB
    )

    if callable(func):
        res.update(
            dict(
                _A=A,
                _B=B,
                _AB=AB,
            )
        )

    return SobolResult(**res)
<|endoftext|>import threading
import numpy as np
from collections import namedtuple
from scipy import special
from scipy import stats
from scipy.stats._stats_py import _rankdata
from ._axis_nan_policy import _axis_nan_policy_factory


def _broadcast_concatenate(x, y, axis):
    '''Broadcast then concatenate arrays, leaving concatenation axis last'''
    x = np.moveaxis(x, axis, -1)
    y = np.moveaxis(y, axis, -1)
    z = np.broadcast(x[..., 0], y[..., 0])
    x = np.broadcast_to(x, z.shape + (x.shape[-1],))
    y = np.broadcast_to(y, z.shape + (y.shape[-1],))
    z = np.concatenate((x, y), axis=-1)
    return x, y, z


class _MWU:
    '''Distribution of MWU statistic under the null hypothesis'''

    def __init__(self, n1, n2):
        self._reset(n1, n2)

    def set_shapes(self, n1, n2):
        n1, n2 = min(n1, n2), max(n1, n2)
        if (n1, n2) == (self.n1, self.n2):
            return

        self.n1 = n1
        self.n2 = n2
        self.s_array = np.zeros(0, dtype=int)
        self.configurations = np.zeros(0, dtype=np.uint64)

    def reset(self):
        self._reset(self.n1, self.n2)

    def _reset(self, n1, n2):
        self.n1 = None
        self.n2 = None
        self.set_shapes(n1, n2)

    def pmf(self, k):

        # In practice, `pmf` is never called with k > m*n/2.
        # If it were, we'd exploit symmetry here:
        # k = np.array(k, copy=True)
        # k2 = m*n - k
        # i = k2 < k
        # k[i] = k2[i]

        pmfs = self.build_u_freqs_array(np.max(k))
        return pmfs[k]

    def cdf(self, k):
        '''Cumulative distribution function'''

        # In practice, `cdf` is never called with k > m*n/2.
        # If it were, we'd exploit symmetry here rather than in `sf`
        pmfs = self.build_u_freqs_array(np.max(k))
        cdfs = np.cumsum(pmfs)
        return cdfs[k]

    def sf(self, k):
        '''Survival function'''
        # Note that both CDF and SF include the PMF at k. The p-value is
        # calculated from the SF and should include the mass at k, so this
        # is desirable

        # Use the fact that the distribution is symmetric and sum from the left
        kc = np.asarray(self.n1*self.n2 - k)  # complement of k
        i = k < kc
        if np.any(i):
            kc[i] = k[i]
            cdfs = np.asarray(self.cdf(kc))
            cdfs[i] = 1. - cdfs[i] + self.pmf(kc[i])
        else:
            cdfs = np.asarray(self.cdf(kc))
        return cdfs[()]

    # build_sigma_array and build_u_freqs_array adapted from code
    # by @toobaz with permission. Thanks to @andreasloe for the suggestion.
    # See https://github.com/scipy/scipy/pull/4933#issuecomment-1898082691
    def build_sigma_array(self, a):
        n1, n2 = self.n1, self.n2
        if a + 1 <= self.s_array.size:
            return self.s_array[1:a+1]

        s_array = np.zeros(a + 1, dtype=int)

        for d in np.arange(1, n1 + 1):
            # All multiples of d, except 0:
            indices = np.arange(d, a + 1, d)
            # \epsilon_d = 1:
            s_array[indices] += d

        for d in np.arange(n2 + 1, n2 + n1 + 1):
            # All multiples of d, except 0:
            indices = np.arange(d, a + 1, d)
            # \epsilon_d = -1:
            s_array[indices] -= d

        # We don't need 0:
        self.s_array = s_array
        return s_array[1:]

    def build_u_freqs_array(self, maxu):
        """
        Build all the array of frequencies for u from 0 to maxu.
        Assumptions:
          n1 <= n2
          maxu <= n1 * n2 / 2
        """
        n1, n2 = self.n1, self.n2
        total = special.binom(n1 + n2, n1)

        if maxu + 1 <= self.configurations.size:
            return self.configurations[:maxu + 1] / total

        s_array = self.build_sigma_array(maxu)

        # Start working with ints, for maximum precision and efficiency:
        configurations = np.zeros(maxu + 1, dtype=np.uint64)
        configurations_is_uint = True
        uint_max = np.iinfo(np.uint64).max
        # How many ways to have U=0? 1
        configurations[0] = 1

        for u in np.arange(1, maxu + 1):
            coeffs = s_array[u - 1::-1]
            new_val = np.dot(configurations[:u], coeffs) / u
            if new_val > uint_max and configurations_is_uint:
                # OK, we got into numbers too big for uint64.
                # So now we start working with floats.
                # By doing this since the beginning, we would have lost precision.
                # (And working on python long ints would be unbearably slow)
                configurations = configurations.astype(float)
                configurations_is_uint = False
            configurations[u] = new_val

        self.configurations = configurations
        return configurations / total


# Maintain state for faster repeat calls to `mannwhitneyu`.
# _MWU() is calculated once per thread and stored as an attribute on
# this thread-local variable inside mannwhitneyu().
_mwu_state = threading.local()


def _get_mwu_z(U, n1, n2, t, axis=0, continuity=True):
    '''Standardized MWU statistic'''
    # Follows mannwhitneyu [2]
    mu = n1 * n2 / 2
    n = n1 + n2

    # Tie correction according to [2], "Normal approximation and tie correction"
    # "A more computationally-efficient form..."
    tie_term = (t**3 - t).sum(axis=-1)
    s = np.sqrt(n1*n2/12 * ((n + 1) - tie_term/(n*(n-1))))

    numerator = U - mu

    # Continuity correction.
    # Because SF is always used to calculate the p-value, we can always
    # _subtract_ 0.5 for the continuity correction. This always increases the
    # p-value to account for the rest of the probability mass _at_ q = U.
    if continuity:
        numerator -= 0.5

    # no problem evaluating the norm SF at an infinity
    with np.errstate(divide='ignore', invalid='ignore'):
        z = numerator / s
    return z


def _mwu_input_validation(x, y, use_continuity, alternative, axis, method):
    ''' Input validation and standardization for mannwhitneyu '''
    # Would use np.asarray_chkfinite, but infs are OK
    x, y = np.atleast_1d(x), np.atleast_1d(y)
    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError('`x` and `y` must not contain NaNs.')
    if np.size(x) == 0 or np.size(y) == 0:
        raise ValueError('`x` and `y` must be of nonzero size.')

    bools = {True, False}
    if use_continuity not in bools:
        raise ValueError(f'`use_continuity` must be one of {bools}.')

    alternatives = {"two-sided", "less", "greater"}
    alternative = alternative.lower()
    if alternative not in alternatives:
        raise ValueError(f'`alternative` must be one of {alternatives}.')

    axis_int = int(axis)
    if axis != axis_int:
        raise ValueError('`axis` must be an integer.')

    if not isinstance(method, stats.PermutationMethod):
        methods = {"asymptotic", "exact", "auto"}
        method = method.lower()
        if method not in methods:
            raise ValueError(f'`method` must be one of {methods}.')

    return x, y, use_continuity, alternative, axis_int, method


def _mwu_choose_method(n1, n2, ties):
    """Choose method 'asymptotic' or 'exact' depending on input size, ties"""

    # if both inputs are large, asymptotic is OK
    if n1 > 8 and n2 > 8:
        return "asymptotic"

    # if there are any ties, asymptotic is preferred
    if ties:
        return "asymptotic"

    return "exact"


MannwhitneyuResult = namedtuple('MannwhitneyuResult', ('statistic', 'pvalue'))


@_axis_nan_policy_factory(MannwhitneyuResult, n_samples=2)
def mannwhitneyu(x, y, use_continuity=True, alternative="two-sided",
                 axis=0, method="auto"):
    r'''Perform the Mann-Whitney U rank test on two independent samples.

    The Mann-Whitney U test is a nonparametric test of the null hypothesis
    that the distribution underlying sample `x` is the same as the
    distribution underlying sample `y`. It is often used as a test of
    difference in location between distributions.

    Parameters
    ----------
    x, y : array-like
        N-d arrays of samples. The arrays must be broadcastable except along
        the dimension given by `axis`.
    use_continuity : bool, optional
            Whether a continuity correction (1/2) should be applied.
            Default is True when `method` is ``'asymptotic'``; has no effect
            otherwise.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        Let *SX(u)* and *SY(u)* be the survival functions of the
        distributions underlying `x` and `y`, respectively. Then the following
        alternative hypotheses are available:

        * 'two-sided': the distributions are not equal, i.e. *SX(u) ≠ SY(u)* for
          at least one *u*.
        * 'less': the distribution underlying `x` is stochastically less
          than the distribution underlying `y`, i.e. *SX(u) < SY(u)* for all *u*.
        * 'greater': the distribution underlying `x` is stochastically greater
          than the distribution underlying `y`, i.e. *SX(u) > SY(u)* for all *u*.

        Under a more restrictive set of assumptions, the alternative hypotheses
        can be expressed in terms of the locations of the distributions;
        see [5]_ section 5.1.
    axis : int, optional
        Axis along which to perform the test. Default is 0.
    method : {'auto', 'asymptotic', 'exact'} or `PermutationMethod` instance, optional
        Selects the method used to calculate the *p*-value.
        Default is 'auto'. The following options are available.

        * ``'asymptotic'``: compares the standardized test statistic
          against the normal distribution, correcting for ties.
        * ``'exact'``: computes the exact *p*-value by comparing the observed
          :math:`U` statistic against the exact distribution of the :math:`U`
          statistic under the null hypothesis. No correction is made for ties.
        * ``'auto'``: chooses ``'exact'`` when the size of one of the samples
          is less than or equal to 8 and there are no ties;
          chooses ``'asymptotic'`` otherwise.
        * `PermutationMethod` instance. In this case, the p-value
          is computed using `permutation_test` with the provided
          configuration options and other appropriate settings.

    Returns
    -------
    res : MannwhitneyuResult
        An object containing attributes:

        statistic : float
            The Mann-Whitney U statistic corresponding with sample `x`. See
            Notes for the test statistic corresponding with sample `y`.
        pvalue : float
            The associated *p*-value for the chosen `alternative`.

    Notes
    -----
    If ``U1`` is the statistic corresponding with sample `x`, then the
    statistic corresponding with sample `y` is
    ``U2 = x.shape[axis] * y.shape[axis] - U1``.

    `mannwhitneyu` is for independent samples. For related / paired samples,
    consider `scipy.stats.wilcoxon`.

    `method` ``'exact'`` is recommended when there are no ties and when either
    sample size is less than 8 [1]_. The implementation follows the algorithm
    reported in [3]_.
    Note that the exact method is *not* corrected for ties, but
    `mannwhitneyu` will not raise errors or warnings if there are ties in the
    data. If there are ties and either samples is small (fewer than ~10
    observations), consider passing an instance of `PermutationMethod`
    as the `method` to perform a permutation test.

    The Mann-Whitney U test is a non-parametric version of the t-test for
    independent samples. When the means of samples from the populations
    are normally distributed, consider `scipy.stats.ttest_ind`.

    See Also
    --------
    scipy.stats.wilcoxon, scipy.stats.ranksums, scipy.stats.ttest_ind

    References
    ----------
    .. [1] H.B. Mann and D.R. Whitney, "On a test of whether one of two random
           variables is stochastically larger than the other", The Annals of
           Mathematical Statistics, Vol. 18, pp. 50-60, 1947.
    .. [2] Mann-Whitney U Test, Wikipedia,
           http://en.wikipedia.org/wiki/Mann-Whitney_U_test
    .. [3] Andreas Löffler,
           "Über eine Partition der nat. Zahlen und ihr Anwendung beim U-Test",
           Wiss. Z. Univ. Halle, XXXII'83 pp. 87-89.
    .. [4] Rosie Shier, "Statistics: 2.3 The Mann-Whitney U Test", Mathematics
           Learning Support Centre, 2004.
    .. [5] Michael P. Fay and Michael A. Proschan. "Wilcoxon-Mann-Whitney
           or t-test? On assumptions for hypothesis tests and multiple \
           interpretations of decision rules." Statistics surveys, Vol. 4, pp.
           1-39, 2010. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2857732/

    Examples
    --------
    We follow the example from [4]_: nine randomly sampled young adults were
    diagnosed with type II diabetes at the ages below.

    >>> males = [19, 22, 16, 29, 24]
    >>> females = [20, 11, 17, 12]

    We use the Mann-Whitney U test to assess whether there is a statistically
    significant difference in the diagnosis age of males and females.
    The null hypothesis is that the distribution of male diagnosis ages is
    the same as the distribution of female diagnosis ages. We decide
    that a confidence level of 95% is required to reject the null hypothesis
    in favor of the alternative that the distributions are different.
    Since the number of samples is very small and there are no ties in the
    data, we can compare the observed test statistic against the *exact*
    distribution of the test statistic under the null hypothesis.

    >>> from scipy.stats import mannwhitneyu
    >>> U1, p = mannwhitneyu(males, females, method="exact")
    >>> print(U1)
    17.0

    `mannwhitneyu` always reports the statistic associated with the first
    sample, which, in this case, is males. This agrees with :math:`U_M = 17`
    reported in [4]_. The statistic associated with the second statistic
    can be calculated:

    >>> nx, ny = len(males), len(females)
    >>> U2 = nx*ny - U1
    >>> print(U2)
    3.0

    This agrees with :math:`U_F = 3` reported in [4]_. The two-sided
    *p*-value can be calculated from either statistic, and the value produced
    by `mannwhitneyu` agrees with :math:`p = 0.11` reported in [4]_.

    >>> print(p)
    0.1111111111111111

    The exact distribution of the test statistic is asymptotically normal, so
    the example continues by comparing the exact *p*-value against the
    *p*-value produced using the normal approximation.

    >>> _, pnorm = mannwhitneyu(males, females, method="asymptotic")
    >>> print(pnorm)
    0.11134688653314041

    Here `mannwhitneyu`'s reported *p*-value appears to conflict with the
    value :math:`p = 0.09` given in [4]_. The reason is that [4]_
    does not apply the continuity correction performed by `mannwhitneyu`;
    `mannwhitneyu` reduces the distance between the test statistic and the
    mean :math:`\mu = n_x n_y / 2` by 0.5 to correct for the fact that the
    discrete statistic is being compared against a continuous distribution.
    Here, the :math:`U` statistic used is less than the mean, so we reduce
    the distance by adding 0.5 in the numerator.

    >>> import numpy as np
    >>> from scipy.stats import norm
    >>> U = min(U1, U2)
    >>> N = nx + ny
    >>> z = (U - nx*ny/2 + 0.5) / np.sqrt(nx*ny * (N + 1)/ 12)
    >>> p = 2 * norm.cdf(z)  # use CDF to get p-value from smaller statistic
    >>> print(p)
    0.11134688653314041

    If desired, we can disable the continuity correction to get a result
    that agrees with that reported in [4]_.

    >>> _, pnorm = mannwhitneyu(males, females, use_continuity=False,
    ...                         method="asymptotic")
    >>> print(pnorm)
    0.0864107329737

    Regardless of whether we perform an exact or asymptotic test, the
    probability of the test statistic being as extreme or more extreme by
    chance exceeds 5%, so we do not consider the results statistically
    significant.

    Suppose that, before seeing the data, we had hypothesized that females
    would tend to be diagnosed at a younger age than males.
    In that case, it would be natural to provide the female ages as the
    first input, and we would have performed a one-sided test using
    ``alternative = 'less'``: females are diagnosed at an age that is
    stochastically less than that of males.

    >>> res = mannwhitneyu(females, males, alternative="less", method="exact")
    >>> print(res)
    MannwhitneyuResult(statistic=3.0, pvalue=0.05555555555555555)

    Again, the probability of getting a sufficiently low value of the
    test statistic by chance under the null hypothesis is greater than 5%,
    so we do not reject the null hypothesis in favor of our alternative.

    If it is reasonable to assume that the means of samples from the
    populations are normally distributed, we could have used a t-test to
    perform the analysis.

    >>> from scipy.stats import ttest_ind
    >>> res = ttest_ind(females, males, alternative="less")
    >>> print(res)
    TtestResult(statistic=-2.239334696520584,
                pvalue=0.030068441095757924,
                df=7.0)

    Under this assumption, the *p*-value would be low enough to reject the
    null hypothesis in favor of the alternative.

    '''

    x, y, use_continuity, alternative, axis_int, method = (
        _mwu_input_validation(x, y, use_continuity, alternative, axis, method))

    x, y, xy = _broadcast_concatenate(x, y, axis)

    n1, n2 = x.shape[-1], y.shape[-1]

    # Follows [2]
    ranks, t = _rankdata(xy, 'average', return_ties=True)  # method 2, step 1
    R1 = ranks[..., :n1].sum(axis=-1)                      # method 2, step 2
    U1 = R1 - n1*(n1+1)/2                                  # method 2, step 3
    U2 = n1 * n2 - U1                                      # as U1 + U2 = n1 * n2

    if alternative == "greater":
        U, f = U1, 1  # U is the statistic to use for p-value, f is a factor
    elif alternative == "less":
        U, f = U2, 1  # Due to symmetry, use SF of U2 rather than CDF of U1
    else:
        U, f = np.maximum(U1, U2), 2  # multiply SF by two for two-sided test

    if method == "auto":
        method = _mwu_choose_method(n1, n2, np.any(t > 1))

    if method == "exact":
        if not hasattr(_mwu_state, 's'):
            _mwu_state.s = _MWU(0, 0)
        _mwu_state.s.set_shapes(n1, n2)
        p = _mwu_state.s.sf(U.astype(int))
    elif method == "asymptotic":
        z = _get_mwu_z(U, n1, n2, t, continuity=use_continuity)
        p = stats.norm.sf(z)
    else:  # `PermutationMethod` instance (already validated)
        def statistic(x, y, axis):
            return mannwhitneyu(x, y, use_continuity=use_continuity,
                                alternative=alternative, axis=axis,
                                method="asymptotic").statistic

        res = stats.permutation_test((x, y), statistic, axis=axis,
                                     **method._asdict(), alternative=alternative)
        p = res.pvalue
        f = 1

    p *= f

    # Ensure that test statistic is not greater than 1
    # This could happen for exact test when U = m*n/2
    p = np.clip(p, 0, 1)

    return MannwhitneyuResult(U1, p)
<|endoftext|>from itertools import permutations
import numpy as np
import math
from ._continuous_distns import norm
import scipy.stats
from dataclasses import dataclass


@dataclass
class PageTrendTestResult:
    statistic: float
    pvalue: float
    method: str


def page_trend_test(data, ranked=False, predicted_ranks=None, method='auto'):
    r"""
    Perform Page's Test, a measure of trend in observations between treatments.

    Page's Test (also known as Page's :math:`L` test) is useful when:

    * there are :math:`n \geq 3` treatments,
    * :math:`m \geq 2` subjects are observed for each treatment, and
    * the observations are hypothesized to have a particular order.

    Specifically, the test considers the null hypothesis that

    .. math::

        m_1 = m_2 = m_3 \cdots = m_n,

    where :math:`m_j` is the mean of the observed quantity under treatment
    :math:`j`, against the alternative hypothesis that

    .. math::

        m_1 \leq m_2 \leq m_3 \leq \cdots \leq m_n,

    where at least one inequality is strict.

    As noted by [4]_, Page's :math:`L` test has greater statistical power than
    the Friedman test against the alternative that there is a difference in
    trend, as Friedman's test only considers a difference in the means of the
    observations without considering their order. Whereas Spearman :math:`\rho`
    considers the correlation between the ranked observations of two variables
    (e.g. the airspeed velocity of a swallow vs. the weight of the coconut it
    carries), Page's :math:`L` is concerned with a trend in an observation
    (e.g. the airspeed velocity of a swallow) across several distinct
    treatments (e.g. carrying each of five coconuts of different weight) even
    as the observation is repeated with multiple subjects (e.g. one European
    swallow and one African swallow).

    Parameters
    ----------
    data : array-like
        A :math:`m \times n` array; the element in row :math:`i` and
        column :math:`j` is the observation corresponding with subject
        :math:`i` and treatment :math:`j`. By default, the columns are
        assumed to be arranged in order of increasing predicted mean.

    ranked : boolean, optional
        By default, `data` is assumed to be observations rather than ranks;
        it will be ranked with `scipy.stats.rankdata` along ``axis=1``. If
        `data` is provided in the form of ranks, pass argument ``True``.

    predicted_ranks : array-like, optional
        The predicted ranks of the column means. If not specified,
        the columns are assumed to be arranged in order of increasing
        predicted mean, so the default `predicted_ranks` are
        :math:`[1, 2, \dots, n-1, n]`.

    method : {'auto', 'asymptotic', 'exact'}, optional
        Selects the method used to calculate the *p*-value. The following
        options are available.

        * 'auto': selects between 'exact' and 'asymptotic' to
          achieve reasonably accurate results in reasonable time (default)
        * 'asymptotic': compares the standardized test statistic against
          the normal distribution
        * 'exact': computes the exact *p*-value by comparing the observed
          :math:`L` statistic against those realized by all possible
          permutations of ranks (under the null hypothesis that each
          permutation is equally likely)

    Returns
    -------
    res : PageTrendTestResult
        An object containing attributes:

        statistic : float
            Page's :math:`L` test statistic.
        pvalue : float
            The associated *p*-value
        method : {'asymptotic', 'exact'}
            The method used to compute the *p*-value

    See Also
    --------
    rankdata, friedmanchisquare, spearmanr

    Notes
    -----
    As noted in [1]_, "the :math:`n` 'treatments' could just as well represent
    :math:`n` objects or events or performances or persons or trials ranked."
    Similarly, the :math:`m` 'subjects' could equally stand for :math:`m`
    "groupings by ability or some other control variable, or judges doing
    the ranking, or random replications of some other sort."

    The procedure for calculating the :math:`L` statistic, adapted from
    [1]_, is:

    1. "Predetermine with careful logic the appropriate hypotheses
       concerning the predicted ordering of the experimental results.
       If no reasonable basis for ordering any treatments is known, the
       :math:`L` test is not appropriate."
    2. "As in other experiments, determine at what level of confidence
       you will reject the null hypothesis that there is no agreement of
       experimental results with the monotonic hypothesis."
    3. "Cast the experimental material into a two-way table of :math:`n`
       columns (treatments, objects ranked, conditions) and :math:`m`
       rows (subjects, replication groups, levels of control variables)."
    4. "When experimental observations are recorded, rank them across each
       row", e.g. ``ranks = scipy.stats.rankdata(data, axis=1)``.
    5. "Add the ranks in each column", e.g.
       ``colsums = np.sum(ranks, axis=0)``.
    6. "Multiply each sum of ranks by the predicted rank for that same
       column", e.g. ``products = predicted_ranks * colsums``.
    7. "Sum all such products", e.g. ``L = products.sum()``.

    [1]_ continues by suggesting use of the standardized statistic

    .. math::

        \chi_L^2 = \frac{\left[12L-3mn(n+1)^2\right]^2}{mn^2(n^2-1)(n+1)}

    "which is distributed approximately as chi-square with 1 degree of
    freedom. The ordinary use of :math:`\chi^2` tables would be
    equivalent to a two-sided test of agreement. If a one-sided test
    is desired, *as will almost always be the case*, the probability
    discovered in the chi-square table should be *halved*."

    However, this standardized statistic does not distinguish between the
    observed values being well correlated with the predicted ranks and being
    _anti_-correlated with the predicted ranks. Instead, we follow [2]_
    and calculate the standardized statistic

    .. math::

        \Lambda = \frac{L - E_0}{\sqrt{V_0}},

    where :math:`E_0 = \frac{1}{4} mn(n+1)^2` and
    :math:`V_0 = \frac{1}{144} mn^2(n+1)(n^2-1)`, "which is asymptotically
    normal under the null hypothesis".

    The *p*-value for ``method='exact'`` is generated by comparing the observed
    value of :math:`L` against the :math:`L` values generated for all
    :math:`(n!)^m` possible permutations of ranks. The calculation is performed
    using the recursive method of [5].

    The *p*-values are not adjusted for the possibility of ties. When
    ties are present, the reported  ``'exact'`` *p*-values may be somewhat
    larger (i.e. more conservative) than the true *p*-value [2]_. The
    ``'asymptotic'``` *p*-values, however, tend to be smaller (i.e. less
    conservative) than the ``'exact'`` *p*-values.

    References
    ----------
    .. [1] Ellis Batten Page, "Ordered hypotheses for multiple treatments:
       a significant test for linear ranks", *Journal of the American
       Statistical Association* 58(301), p. 216--230, 1963.

    .. [2] Markus Neuhauser, *Nonparametric Statistical Test: A computational
       approach*, CRC Press, p. 150--152, 2012.

    .. [3] Statext LLC, "Page's L Trend Test - Easy Statistics", *Statext -
       Statistics Study*, https://www.statext.com/practice/PageTrendTest03.php,
       Accessed July 12, 2020.

    .. [4] "Page's Trend Test", *Wikipedia*, WikimediaFoundation,
       https://en.wikipedia.org/wiki/Page%27s_trend_test,
       Accessed July 12, 2020.

    .. [5] Robert E. Odeh, "The exact distribution of Page's L-statistic in
       the two-way layout", *Communications in Statistics - Simulation and
       Computation*,  6(1), p. 49--61, 1977.

    Examples
    --------
    We use the example from [3]_: 10 students are asked to rate three
    teaching methods - tutorial, lecture, and seminar - on a scale of 1-5,
    with 1 being the lowest and 5 being the highest. We have decided that
    a confidence level of 99% is required to reject the null hypothesis in
    favor of our alternative: that the seminar will have the highest ratings
    and the tutorial will have the lowest. Initially, the data have been
    tabulated with each row representing an individual student's ratings of
    the three methods in the following order: tutorial, lecture, seminar.

    >>> table = [[3, 4, 3],
    ...          [2, 2, 4],
    ...          [3, 3, 5],
    ...          [1, 3, 2],
    ...          [2, 3, 2],
    ...          [2, 4, 5],
    ...          [1, 2, 4],
    ...          [3, 4, 4],
    ...          [2, 4, 5],
    ...          [1, 3, 4]]

    Because the tutorial is hypothesized to have the lowest ratings, the
    column corresponding with tutorial rankings should be first; the seminar
    is hypothesized to have the highest ratings, so its column should be last.
    Since the columns are already arranged in this order of increasing
    predicted mean, we can pass the table directly into `page_trend_test`.

    >>> from scipy.stats import page_trend_test
    >>> res = page_trend_test(table)
    >>> res
    PageTrendTestResult(statistic=133.5, pvalue=0.0018191161948127822,
                        method='exact')

    This *p*-value indicates that there is a 0.1819% chance that
    the :math:`L` statistic would reach such an extreme value under the null
    hypothesis. Because 0.1819% is less than 1%, we have evidence to reject
    the null hypothesis in favor of our alternative at a 99% confidence level.

    The value of the :math:`L` statistic is 133.5. To check this manually,
    we rank the data such that high scores correspond with high ranks, settling
    ties with an average rank:

    >>> from scipy.stats import rankdata
    >>> ranks = rankdata(table, axis=1)
    >>> ranks
    array([[1.5, 3. , 1.5],
           [1.5, 1.5, 3. ],
           [1.5, 1.5, 3. ],
           [1. , 3. , 2. ],
           [1.5, 3. , 1.5],
           [1. , 2. , 3. ],
           [1. , 2. , 3. ],
           [1. , 2.5, 2.5],
           [1. , 2. , 3. ],
           [1. , 2. , 3. ]])

    We add the ranks within each column, multiply the sums by the
    predicted ranks, and sum the products.

    >>> import numpy as np
    >>> m, n = ranks.shape
    >>> predicted_ranks = np.arange(1, n+1)
    >>> L = (predicted_ranks * np.sum(ranks, axis=0)).sum()
    >>> res.statistic == L
    True

    As presented in [3]_, the asymptotic approximation of the *p*-value is the
    survival function of the normal distribution evaluated at the standardized
    test statistic:

    >>> from scipy.stats import norm
    >>> E0 = (m*n*(n+1)**2)/4
    >>> V0 = (m*n**2*(n+1)*(n**2-1))/144
    >>> Lambda = (L-E0)/np.sqrt(V0)
    >>> p = norm.sf(Lambda)
    >>> p
    0.0012693433690751756

    This does not precisely match the *p*-value reported by `page_trend_test`
    above. The asymptotic distribution is not very accurate, nor conservative,
    for :math:`m \leq 12` and :math:`n \leq 8`, so `page_trend_test` chose to
    use ``method='exact'`` based on the dimensions of the table and the
    recommendations in Page's original paper [1]_. To override
    `page_trend_test`'s choice, provide the `method` argument.

    >>> res = page_trend_test(table, method="asymptotic")
    >>> res
    PageTrendTestResult(statistic=133.5, pvalue=0.0012693433690751756,
                        method='asymptotic')

    If the data are already ranked, we can pass in the ``ranks`` instead of
    the ``table`` to save computation time.

    >>> res = page_trend_test(ranks,             # ranks of data
    ...                       ranked=True,       # data is already ranked
    ...                       )
    >>> res
    PageTrendTestResult(statistic=133.5, pvalue=0.0018191161948127822,
                        method='exact')

    Suppose the raw data had been tabulated in an order different from the
    order of predicted means, say lecture, seminar, tutorial.

    >>> table = np.asarray(table)[:, [1, 2, 0]]

    Since the arrangement of this table is not consistent with the assumed
    ordering, we can either rearrange the table or provide the
    `predicted_ranks`. Remembering that the lecture is predicted
    to have the middle rank, the seminar the highest, and tutorial the lowest,
    we pass:

    >>> res = page_trend_test(table,             # data as originally tabulated
    ...                       predicted_ranks=[2, 3, 1],  # our predicted order
    ...                       )
    >>> res
    PageTrendTestResult(statistic=133.5, pvalue=0.0018191161948127822,
                        method='exact')

    """

    # Possible values of the method parameter and the corresponding function
    # used to evaluate the p value
    methods = {"asymptotic": _l_p_asymptotic,
               "exact": _l_p_exact,
               "auto": None}
    if method not in methods:
        raise ValueError(f"`method` must be in {set(methods)}")

    ranks = np.asarray(data)
    if ranks.ndim != 2:  # TODO: relax this to accept 3d arrays?
        raise ValueError("`data` must be a 2d array.")

    m, n = ranks.shape
    if m < 2 or n < 3:
        raise ValueError("Page's L is only appropriate for data with two "
                         "or more rows and three or more columns.")

    if np.any(np.isnan(data)):
        raise ValueError("`data` contains NaNs, which cannot be ranked "
                         "meaningfully")

    # ensure NumPy array and rank the data if it's not already ranked
    if ranked:
        # Only a basic check on whether data is ranked. Checking that the data
        # is properly ranked could take as much time as ranking it.
        if not (ranks.min() >= 1 and ranks.max() <= ranks.shape[1]):
            raise ValueError("`data` is not properly ranked. Rank the data or "
                             "pass `ranked=False`.")
    else:
        ranks = scipy.stats.rankdata(data, axis=-1)

    # generate predicted ranks if not provided, ensure valid NumPy array
    if predicted_ranks is None:
        predicted_ranks = np.arange(1, n+1)
    else:
        predicted_ranks = np.asarray(predicted_ranks)
        if (predicted_ranks.ndim < 1 or
                (set(predicted_ranks) != set(range(1, n+1)) or
                 len(predicted_ranks) != n)):
            raise ValueError(f"`predicted_ranks` must include each integer "
                             f"from 1 to {n} (the number of columns in "
                             f"`data`) exactly once.")

    if not isinstance(ranked, bool):
        raise TypeError("`ranked` must be boolean.")

    # Calculate the L statistic
    L = _l_vectorized(ranks, predicted_ranks)

    # Calculate the p-value
    if method == "auto":
        method = _choose_method(ranks)
    p_fun = methods[method]  # get the function corresponding with the method
    p = p_fun(L, m, n)

    page_result = PageTrendTestResult(statistic=L, pvalue=p, method=method)
    return page_result


def _choose_method(ranks):
    '''Choose method for computing p-value automatically'''
    m, n = ranks.shape
    if n > 8 or (m > 12 and n > 3) or m > 20:  # as in [1], [4]
        method = "asymptotic"
    else:
        method = "exact"
    return method


def _l_vectorized(ranks, predicted_ranks):
    '''Calculate's Page's L statistic for each page of a 3d array'''
    colsums = ranks.sum(axis=-2, keepdims=True)
    products = predicted_ranks * colsums
    Ls = products.sum(axis=-1)
    Ls = Ls[0] if Ls.size == 1 else Ls.ravel()
    return Ls


def _l_p_asymptotic(L, m, n):
    '''Calculate the p-value of Page's L from the asymptotic distribution'''
    # Using [1] as a reference, the asymptotic p-value would be calculated as:
    # chi_L = (12*L - 3*m*n*(n+1)**2)**2/(m*n**2*(n**2-1)*(n+1))
    # p = chi2.sf(chi_L, df=1, loc=0, scale=1)/2
    # but this is insensitive to the direction of the hypothesized ranking

    # See [2] page 151
    E0 = (m*n*(n+1)**2)/4
    V0 = (m*n**2*(n+1)*(n**2-1))/144
    Lambda = (L-E0)/np.sqrt(V0)
    # This is a one-sided "greater" test - calculate the probability that the
    # L statistic under H0 would be greater than the observed L statistic
    p = norm.sf(Lambda)
    return p


def _l_p_exact(L, m, n):
    '''Calculate the p-value of Page's L exactly'''
    # [1] uses m, n; [5] uses n, k.
    # Switch convention here because exact calculation code references [5].
    L, n, k = int(L), int(m), int(n)
    _pagel_state.set_k(k)
    return _pagel_state.sf(L, n)


class _PageL:
    '''Maintains state between `page_trend_test` executions'''

    def __init__(self):
        '''Lightweight initialization'''
        self.all_pmfs = {}

    def set_k(self, k):
        '''Calculate lower and upper limits of L for single row'''
        self.k = k
        # See [5] top of page 52
        self.a, self.b = (k*(k+1)*(k+2))//6, (k*(k+1)*(2*k+1))//6

    def sf(self, l, n):
        '''Survival function of Page's L statistic'''
        ps = [self.pmf(l, n) for l in range(l, n*self.b + 1)]
        return np.sum(ps)

    def p_l_k_1(self):
        '''Relative frequency of each L value over all possible single rows'''

        # See [5] Equation (6)
        ranks = range(1, self.k+1)
        # generate all possible rows of length k
        rank_perms = np.array(list(permutations(ranks)))
        # compute Page's L for all possible rows
        Ls = (ranks*rank_perms).sum(axis=1)
        # count occurrences of each L value
        counts = np.histogram(Ls, np.arange(self.a-0.5, self.b+1.5))[0]
        # factorial(k) is number of possible permutations
        return counts/math.factorial(self.k)

    def pmf(self, l, n):
        '''Recursive function to evaluate p(l, k, n); see [5] Equation 1'''

        if n not in self.all_pmfs:
            self.all_pmfs[n] = {}
        if self.k not in self.all_pmfs[n]:
            self.all_pmfs[n][self.k] = {}

        # Cache results to avoid repeating calculation. Initially this was
        # written with lru_cache, but this seems faster? Also, we could add
        # an option to save this for future lookup.
        if l in self.all_pmfs[n][self.k]:
            return self.all_pmfs[n][self.k][l]

        if n == 1:
            ps = self.p_l_k_1()  # [5] Equation 6
            ls = range(self.a, self.b+1)
            # not fast, but we'll only be here once
            self.all_pmfs[n][self.k] = {l: p for l, p in zip(ls, ps)}
            return self.all_pmfs[n][self.k][l]

        p = 0
        low = max(l-(n-1)*self.b, self.a)  # [5] Equation 2
        high = min(l-(n-1)*self.a, self.b)

        # [5] Equation 1
        for t in range(low, high+1):
            p1 = self.pmf(l-t, n-1)
            p2 = self.pmf(t, 1)
            p += p1*p2
        self.all_pmfs[n][self.k][l] = p
        return p


# Maintain state for faster repeat calls to page_trend_test w/ method='exact'
_pagel_state = _PageL()
<|endoftext|>"""
.. _statsrefmanual:

==========================================
Statistical functions (:mod:`scipy.stats`)
==========================================

.. currentmodule:: scipy.stats

This module contains a large number of probability distributions,
summary and frequency statistics, correlation functions and statistical
tests, masked statistics, kernel density estimation, quasi-Monte Carlo
functionality, and more.

Statistics is a very large area, and there are topics that are out of scope
for SciPy and are covered by other packages. Some of the most important ones
are:

- `statsmodels <https://www.statsmodels.org/stable/index.html>`__:
  regression, linear models, time series analysis, extensions to topics
  also covered by ``scipy.stats``.
- `Pandas <https://pandas.pydata.org/>`__: tabular data, time series
  functionality, interfaces to other statistical languages.
- `PyMC <https://docs.pymc.io/>`__: Bayesian statistical
  modeling, probabilistic machine learning.
- `scikit-learn <https://scikit-learn.org/>`__: classification, regression,
  model selection.
- `Seaborn <https://seaborn.pydata.org/>`__: statistical data visualization.
- `rpy2 <https://rpy2.github.io/>`__: Python to R bridge.


Probability distributions
=========================

Each univariate distribution is an instance of a subclass of `rv_continuous`
(`rv_discrete` for discrete distributions):

.. autosummary::
   :toctree: generated/

   rv_continuous
   rv_discrete
   rv_histogram

Continuous distributions
------------------------

.. autosummary::
   :toctree: generated/

   alpha             -- Alpha
   anglit            -- Anglit
   arcsine           -- Arcsine
   argus             -- Argus
   beta              -- Beta
   betaprime         -- Beta Prime
   bradford          -- Bradford
   burr              -- Burr (Type III)
   burr12            -- Burr (Type XII)
   cauchy            -- Cauchy
   chi               -- Chi
   chi2              -- Chi-squared
   cosine            -- Cosine
   crystalball       -- Crystalball
   dgamma            -- Double Gamma
   dpareto_lognorm   -- Double Pareto Lognormal
   dweibull          -- Double Weibull
   erlang            -- Erlang
   expon             -- Exponential
   exponnorm         -- Exponentially Modified Normal
   exponweib         -- Exponentiated Weibull
   exponpow          -- Exponential Power
   f                 -- F (Snecdor F)
   fatiguelife       -- Fatigue Life (Birnbaum-Saunders)
   fisk              -- Fisk
   foldcauchy        -- Folded Cauchy
   foldnorm          -- Folded Normal
   genlogistic       -- Generalized Logistic
   gennorm           -- Generalized normal
   genpareto         -- Generalized Pareto
   genexpon          -- Generalized Exponential
   genextreme        -- Generalized Extreme Value
   gausshyper        -- Gauss Hypergeometric
   gamma             -- Gamma
   gengamma          -- Generalized gamma
   genhalflogistic   -- Generalized Half Logistic
   genhyperbolic     -- Generalized Hyperbolic
   geninvgauss       -- Generalized Inverse Gaussian
   gibrat            -- Gibrat
   gompertz          -- Gompertz (Truncated Gumbel)
   gumbel_r          -- Right Sided Gumbel, Log-Weibull, Fisher-Tippett, Extreme Value Type I
   gumbel_l          -- Left Sided Gumbel, etc.
   halfcauchy        -- Half Cauchy
   halflogistic      -- Half Logistic
   halfnorm          -- Half Normal
   halfgennorm       -- Generalized Half Normal
   hypsecant         -- Hyperbolic Secant
   invgamma          -- Inverse Gamma
   invgauss          -- Inverse Gaussian
   invweibull        -- Inverse Weibull
   irwinhall         -- Irwin-Hall
   jf_skew_t         -- Jones and Faddy Skew-T
   johnsonsb         -- Johnson SB
   johnsonsu         -- Johnson SU
   kappa4            -- Kappa 4 parameter
   kappa3            -- Kappa 3 parameter
   ksone             -- Distribution of Kolmogorov-Smirnov one-sided test statistic
   kstwo             -- Distribution of Kolmogorov-Smirnov two-sided test statistic
   kstwobign         -- Limiting Distribution of scaled Kolmogorov-Smirnov two-sided test statistic.
   landau            -- Landau
   laplace           -- Laplace
   laplace_asymmetric    -- Asymmetric Laplace
   levy              -- Levy
   levy_l
   levy_stable
   logistic          -- Logistic
   loggamma          -- Log-Gamma
   loglaplace        -- Log-Laplace (Log Double Exponential)
   lognorm           -- Log-Normal
   loguniform        -- Log-Uniform
   lomax             -- Lomax (Pareto of the second kind)
   maxwell           -- Maxwell
   mielke            -- Mielke's Beta-Kappa
   moyal             -- Moyal
   nakagami          -- Nakagami
   ncx2              -- Non-central chi-squared
   ncf               -- Non-central F
   nct               -- Non-central Student's T
   norm              -- Normal (Gaussian)
   norminvgauss      -- Normal Inverse Gaussian
   pareto            -- Pareto
   pearson3          -- Pearson type III
   powerlaw          -- Power-function
   powerlognorm      -- Power log normal
   powernorm         -- Power normal
   rdist             -- R-distribution
   rayleigh          -- Rayleigh
   rel_breitwigner   -- Relativistic Breit-Wigner
   rice              -- Rice
   recipinvgauss     -- Reciprocal Inverse Gaussian
   semicircular      -- Semicircular
   skewcauchy        -- Skew Cauchy
   skewnorm          -- Skew normal
   studentized_range    -- Studentized Range
   t                 -- Student's T
   trapezoid         -- Trapezoidal
   triang            -- Triangular
   truncexpon        -- Truncated Exponential
   truncnorm         -- Truncated Normal
   truncpareto       -- Truncated Pareto
   truncweibull_min  -- Truncated minimum Weibull distribution
   tukeylambda       -- Tukey-Lambda
   uniform           -- Uniform
   vonmises          -- Von-Mises (Circular)
   vonmises_line     -- Von-Mises (Line)
   wald              -- Wald
   weibull_min       -- Minimum Weibull (see Frechet)
   weibull_max       -- Maximum Weibull (see Frechet)
   wrapcauchy        -- Wrapped Cauchy

The ``fit`` method of the univariate continuous distributions uses
maximum likelihood estimation to fit the distribution to a data set.
The ``fit`` method can accept regular data or *censored data*.
Censored data is represented with instances of the `CensoredData`
class.

.. autosummary::
   :toctree: generated/

   CensoredData


Multivariate distributions
--------------------------

.. autosummary::
   :toctree: generated/

   multivariate_normal    -- Multivariate normal distribution
   matrix_normal          -- Matrix normal distribution
   dirichlet              -- Dirichlet
   dirichlet_multinomial  -- Dirichlet multinomial distribution
   wishart                -- Wishart
   invwishart             -- Inverse Wishart
   multinomial            -- Multinomial distribution
   special_ortho_group    -- SO(N) group
   ortho_group            -- O(N) group
   unitary_group          -- U(N) group
   random_correlation     -- random correlation matrices
   multivariate_t         -- Multivariate t-distribution
   multivariate_hypergeom -- Multivariate hypergeometric distribution
   normal_inverse_gamma   -- Normal-inverse-gamma distribution
   random_table           -- Distribution of random tables with given marginals
   uniform_direction      -- Uniform distribution on S(N-1)
   vonmises_fisher        -- Von Mises-Fisher distribution

`scipy.stats.multivariate_normal` methods accept instances
of the following class to represent the covariance.

.. autosummary::
   :toctree: generated/

   Covariance             -- Representation of a covariance matrix


Discrete distributions
----------------------

.. autosummary::
   :toctree: generated/

   bernoulli                -- Bernoulli
   betabinom                -- Beta-Binomial
   betanbinom               -- Beta-Negative Binomial
   binom                    -- Binomial
   boltzmann                -- Boltzmann (Truncated Discrete Exponential)
   dlaplace                 -- Discrete Laplacian
   geom                     -- Geometric
   hypergeom                -- Hypergeometric
   logser                   -- Logarithmic (Log-Series, Series)
   nbinom                   -- Negative Binomial
   nchypergeom_fisher       -- Fisher's Noncentral Hypergeometric
   nchypergeom_wallenius    -- Wallenius's Noncentral Hypergeometric
   nhypergeom               -- Negative Hypergeometric
   planck                   -- Planck (Discrete Exponential)
   poisson                  -- Poisson
   poisson_binom            -- Poisson Binomial
   randint                  -- Discrete Uniform
   skellam                  -- Skellam
   yulesimon                -- Yule-Simon
   zipf                     -- Zipf (Zeta)
   zipfian                  -- Zipfian


An overview of statistical functions is given below.  Many of these functions
have a similar version in `scipy.stats.mstats` which work for masked arrays.

Summary statistics
==================

.. autosummary::
   :toctree: generated/

   describe          -- Descriptive statistics
   gmean             -- Geometric mean
   hmean             -- Harmonic mean
   pmean             -- Power mean
   kurtosis          -- Fisher or Pearson kurtosis
   mode              -- Modal value
   moment            -- Central moment
   lmoment
   expectile         -- Expectile
   skew              -- Skewness
   kstat             --
   kstatvar          --
   tmean             -- Truncated arithmetic mean
   tvar              -- Truncated variance
   tmin              --
   tmax              --
   tstd              --
   tsem              --
   variation         -- Coefficient of variation
   find_repeats
   rankdata
   tiecorrect
   trim_mean
   gstd              -- Geometric Standard Deviation
   iqr
   sem
   bayes_mvs
   mvsdist
   entropy
   differential_entropy
   median_abs_deviation

Frequency statistics
====================

.. autosummary::
   :toctree: generated/

   cumfreq
   percentileofscore
   scoreatpercentile
   relfreq

.. autosummary::
   :toctree: generated/

   binned_statistic     -- Compute a binned statistic for a set of data.
   binned_statistic_2d  -- Compute a 2-D binned statistic for a set of data.
   binned_statistic_dd  -- Compute a d-D binned statistic for a set of data.

.. _hypotests:

Hypothesis Tests and related functions
======================================
SciPy has many functions for performing hypothesis tests that return a
test statistic and a p-value, and several of them return confidence intervals
and/or other related information.

The headings below are based on common uses of the functions within, but due to
the wide variety of statistical procedures, any attempt at coarse-grained
categorization will be imperfect. Also, note that tests within the same heading
are not interchangeable in general (e.g. many have different distributional
assumptions).

One Sample Tests / Paired Sample Tests
--------------------------------------
One sample tests are typically used to assess whether a single sample was
drawn from a specified distribution or a distribution with specified properties
(e.g. zero mean).

.. autosummary::
   :toctree: generated/

   ttest_1samp
   binomtest
   quantile_test
   skewtest
   kurtosistest
   normaltest
   jarque_bera
   shapiro
   anderson
   cramervonmises
   ks_1samp
   goodness_of_fit
   chisquare
   power_divergence

Paired sample tests are often used to assess whether two samples were drawn
from the same distribution; they differ from the independent sample tests below
in that each observation in one sample is treated as paired with a
closely-related observation in the other sample (e.g. when environmental
factors are controlled between observations within a pair but not among pairs).
They can also be interpreted or used as one-sample tests (e.g. tests on the
mean or median of *differences* between paired observations).

.. autosummary::
   :toctree: generated/

   ttest_rel
   wilcoxon

Association/Correlation Tests
-----------------------------

These tests are often used to assess whether there is a relationship (e.g.
linear) between paired observations in multiple samples or among the
coordinates of multivariate observations.

.. autosummary::
   :toctree: generated/

   linregress
   pearsonr
   spearmanr
   pointbiserialr
   kendalltau
   chatterjeexi
   weightedtau
   somersd
   siegelslopes
   theilslopes
   page_trend_test
   multiscale_graphcorr

These association tests and are to work with samples in the form of contingency
tables. Supporting functions are available in `scipy.stats.contingency`.

.. autosummary::
   :toctree: generated/

   chi2_contingency
   fisher_exact
   barnard_exact
   boschloo_exact

Independent Sample Tests
------------------------
Independent sample tests are typically used to assess whether multiple samples
were independently drawn from the same distribution or different distributions
with a shared property (e.g. equal means).

Some tests are specifically for comparing two samples.

.. autosummary::
   :toctree: generated/

   ttest_ind_from_stats
   poisson_means_test
   ttest_ind
   mannwhitneyu
   bws_test
   ranksums
   brunnermunzel
   mood
   ansari
   cramervonmises_2samp
   epps_singleton_2samp
   ks_2samp
   kstest

Others are generalized to multiple samples.

.. autosummary::
   :toctree: generated/

   f_oneway
   tukey_hsd
   dunnett
   kruskal
   alexandergovern
   fligner
   levene
   bartlett
   median_test
   friedmanchisquare
   anderson_ksamp

Resampling and Monte Carlo Methods
----------------------------------
The following functions can reproduce the p-value and confidence interval
results of most of the functions above, and often produce accurate results in a
wider variety of conditions. They can also be used to perform hypothesis tests
and generate confidence intervals for custom statistics. This flexibility comes
at the cost of greater computational requirements and stochastic results.

.. autosummary::
   :toctree: generated/

   monte_carlo_test
   permutation_test
   bootstrap
   power

Instances of the following object can be passed into some hypothesis test
functions to perform a resampling or Monte Carlo version of the hypothesis
test.

.. autosummary::
   :toctree: generated/

   MonteCarloMethod
   PermutationMethod
   BootstrapMethod

Multiple Hypothesis Testing and Meta-Analysis
---------------------------------------------
These functions are for assessing the results of individual tests as a whole.
Functions for performing specific multiple hypothesis tests (e.g. post hoc
tests) are listed above.

.. autosummary::
   :toctree: generated/

   combine_pvalues
   false_discovery_control


The following functions are related to the tests above but do not belong in the
above categories.

Random Variables
================

.. autosummary::
   :toctree: generated/

   make_distribution
   Normal
   Uniform
   Mixture
   order_statistic
   truncate
   abs
   exp
   log

Quasi-Monte Carlo
=================

.. toctree::
   :maxdepth: 4

   stats.qmc

Contingency Tables
==================

.. toctree::
   :maxdepth: 4

   stats.contingency

Masked statistics functions
===========================

.. toctree::

   stats.mstats


Other statistical functionality
===============================

Transformations
---------------

.. autosummary::
   :toctree: generated/

   boxcox
   boxcox_normmax
   boxcox_llf
   yeojohnson
   yeojohnson_normmax
   yeojohnson_llf
   obrientransform
   sigmaclip
   trimboth
   trim1
   zmap
   zscore
   gzscore

Statistical distances
---------------------

.. autosummary::
   :toctree: generated/

   wasserstein_distance
   wasserstein_distance_nd
   energy_distance

Sampling
--------

.. toctree::
   :maxdepth: 4

   stats.sampling

Fitting / Survival Analysis
---------------------------

.. autosummary::
   :toctree: generated/

   fit
   ecdf
   logrank

Directional statistical functions
---------------------------------

.. autosummary::
   :toctree: generated/

   directional_stats
   circmean
   circvar
   circstd

Sensitivity Analysis
--------------------

.. autosummary::
   :toctree: generated/

   sobol_indices

Plot-tests
----------

.. autosummary::
   :toctree: generated/

   ppcc_max
   ppcc_plot
   probplot
   boxcox_normplot
   yeojohnson_normplot

Univariate and multivariate kernel density estimation
-----------------------------------------------------

.. autosummary::
   :toctree: generated/

   gaussian_kde

Warnings / Errors used in :mod:`scipy.stats`
--------------------------------------------

.. autosummary::
   :toctree: generated/

   DegenerateDataWarning
   ConstantInputWarning
   NearConstantInputWarning
   FitError

Result classes used in :mod:`scipy.stats`
-----------------------------------------

.. warning::

    These classes are private, but they are included here because instances
    of them are returned by other statistical functions. User import and
    instantiation is not supported.

.. toctree::
   :maxdepth: 2

   stats._result_classes

"""  # noqa: E501

from ._warnings_errors import (ConstantInputWarning, NearConstantInputWarning,
                               DegenerateDataWarning, FitError)
from ._stats_py import *
from ._variation import variation
from .distributions import *
from ._morestats import *
from ._multicomp import *
from ._binomtest import binomtest
from ._binned_statistic import *
from ._kde import gaussian_kde
from . import mstats
from . import qmc
from ._multivariate import *
from . import contingency
from .contingency import chi2_contingency
from ._censored_data import CensoredData
from ._resampling import (bootstrap, monte_carlo_test, permutation_test, power,
                          MonteCarloMethod, PermutationMethod, BootstrapMethod)
from ._entropy import *
from ._hypotests import *
from ._page_trend_test import page_trend_test
from ._mannwhitneyu import mannwhitneyu
from ._bws_test import bws_test
from ._fit import fit, goodness_of_fit
from ._covariance import Covariance
from ._sensitivity_analysis import *
from ._survival import *
from ._distribution_infrastructure import (
    make_distribution, Mixture, order_statistic, truncate, exp, log, abs
)
from ._new_distributions import Normal, Uniform
from ._mgc import multiscale_graphcorr
from ._correlation import chatterjeexi


# Deprecated namespaces, to be removed in v2.0.0
from . import (
    biasedurn, kde, morestats, mstats_basic, mstats_extras, mvn, stats
)


__all__ = [s for s in dir() if not s.startswith("_")]  # Remove dunders.

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
<|endoftext|># Temporary file separated from _distribution_infrastructure.py
# to simplify the diff during PR review.
from abc import ABC, abstractmethod

class _ProbabilityDistribution(ABC):
    @abstractmethod
    def support(self):
        r"""Support of the random variable

        The support of a random variable is set of all possible outcomes;
        i.e., the subset of the domain of argument :math:`x` for which
        the probability density function :math:`f(x)` is nonzero.

        This function returns lower and upper bounds of the support.

        Returns
        -------
        out : tuple of Array
            The lower and upper bounds of the support.

        See Also
        --------
        pdf

        References
        ----------
        .. [1] Support (mathematics), *Wikipedia*,
               https://en.wikipedia.org/wiki/Support_(mathematics)

        Notes
        -----
        Suppose a continuous probability distribution has support ``(l, r)``.
        The following table summarizes the value returned by methods
        of ``ContinuousDistribution`` for arguments outside the support.

        +----------------+---------------------+---------------------+
        | Method         | Value for ``x < l`` | Value for ``x > r`` |
        +================+=====================+=====================+
        | ``pdf(x)``     | 0                   | 0                   |
        +----------------+---------------------+---------------------+
        | ``logpdf(x)``  | -inf                | -inf                |
        +----------------+---------------------+---------------------+
        | ``cdf(x)``     | 0                   | 1                   |
        +----------------+---------------------+---------------------+
        | ``logcdf(x)``  | -inf                | 0                   |
        +----------------+---------------------+---------------------+
        | ``ccdf(x)``    | 1                   | 0                   |
        +----------------+---------------------+---------------------+
        | ``logccdf(x)`` | 0                   | -inf                |
        +----------------+---------------------+---------------------+

        For the ``cdf`` and related methods, the inequality need not be
        strict; i.e. the tabulated value is returned when the method is
        evaluated *at* the corresponding boundary.

        The following table summarizes the value returned by the inverse
        methods of ``ContinuousDistribution`` for arguments at the boundaries
        of the domain ``0`` to ``1``.

        +-------------+-----------+-----------+
        | Method      | ``x = 0`` | ``x = 1`` |
        +=============+===========+===========+
        | ``icdf(x)`` | ``l``     | ``r``     |
        +-------------+-----------+-----------+
        | ``icdf(x)`` | ``r``     | ``l``     |
        +-------------+-----------+-----------+

        For the inverse log-functions, the same values are returned for
        for ``x = log(0)`` and ``x = log(1)``. All inverse functions return
        ``nan`` when evaluated at an argument outside the domain ``0`` to ``1``.

        Examples
        --------
        Instantiate a distribution with the desired parameters:

        >>> from scipy import stats
        >>> X = stats.Uniform(a=-0.5, b=0.5)

        Retrieve the support of the distribution:

        >>> X.support()
        (-0.5, 0.5)

        For a distribution with infinite support,

        >>> X = stats.Normal()
        >>> X.support()
        (-inf, inf)

        Due to underflow, the numerical value returned by the PDF may be zero
        even for arguments within the support, even if the true value is
        nonzero. In such cases, the log-PDF may be useful.

        >>> X.pdf([-100., 100.])
        array([0., 0.])
        >>> X.logpdf([-100., 100.])
        array([-5000.91893853, -5000.91893853])

        Use cases for the log-CDF and related methods are analogous.

        """
        raise NotImplementedError()

    @abstractmethod
    def sample(self, shape, *, method, rng):
        r"""Random sample from the distribution.

        Parameters
        ----------
        shape : tuple of ints, default: ()
            The shape of the sample to draw. If the parameters of the distribution
            underlying the random variable are arrays of shape ``param_shape``,
            the output array will be of shape ``shape + param_shape``.
        method : {None, 'formula', 'inverse_transform'}
            The strategy used to produce the sample. By default (``None``),
            the infrastructure chooses between the following options,
            listed in order of precedence.

            - ``'formula'``: an implementation specific to the distribution
            - ``'inverse_transform'``: generate a uniformly distributed sample and
              return the inverse CDF at these arguments.

            Not all `method` options are available for all distributions.
            If the selected `method` is not available, a `NotImplementedError``
            will be raised.
        rng : `numpy.random.Generator` or `scipy.stats.QMCEngine`, optional
            Pseudo- or quasi-random number generator state. When `rng` is None,
            a new `numpy.random.Generator` is created using entropy from the
            operating system. Types other than `numpy.random.Generator` and
            `scipy.stats.QMCEngine` are passed to `numpy.random.default_rng`
            to instantiate a ``Generator``.

            If `rng` is an instance of `scipy.stats.QMCEngine` configured to use
            scrambling and `shape` is not empty, then each slice along the zeroth
            axis of the result is a "quasi-independent", low-discrepancy sequence;
            that is, they are distinct sequences that can be treated as statistically
            independent for most practical purposes. Separate calls to `sample`
            produce new quasi-independent, low-discrepancy sequences.

        References
        ----------
        .. [1] Sampling (statistics), *Wikipedia*,
               https://en.wikipedia.org/wiki/Sampling_(statistics)

        Examples
        --------
        Instantiate a distribution with the desired parameters:

        >>> import numpy as np
        >>> from scipy import stats
        >>> X = stats.Uniform(a=0., b=1.)

        Generate a pseudorandom sample:

        >>> x = X.sample((1000, 1))
        >>> octiles = (np.arange(8) + 1) / 8
        >>> np.count_nonzero(x <= octiles, axis=0)
        array([ 148,  263,  387,  516,  636,  751,  865, 1000])  # may vary

        >>> X = stats.Uniform(a=np.zeros((3, 1)), b=np.ones(2))
        >>> X.a.shape,
        (3, 2)
        >>> x = X.sample(shape=(5, 4))
        >>> x.shape
        (5, 4, 3, 2)

        """
        raise NotImplementedError()

    @abstractmethod
    def moment(self, order, kind, *, method):
        r"""Raw, central, or standard moment of positive integer order.

        In terms of probability density function :math:`f(x)` and support
        :math:`\chi`, the "raw" moment (about the origin) of order :math:`n` of
        a random variable :math:`X` is:

        .. math::

            \mu'_n(X) = \int_{\chi} x^n f(x) dx

        The "central" moment is the raw moment taken about the mean,
        :math:`\mu = \mu'_1`:

        .. math::

            \mu_n(X) = \int_{\chi} (x - \mu) ^n f(x) dx

        The "standardized" moment is the central moment normalized by the
        :math:`n^\text{th}` power of the standard deviation
        :math:`\sigma = \sqrt{\mu_2}` to produce a scale invariant quantity:

        .. math::

            \tilde{\mu}_n(X) = \frac{\mu_n(X)}
                                    {\sigma^n}

        Parameters
        ----------
        order : int
            The integer order of the moment; i.e. :math:`n` in the formulae above.
        kind : {'raw', 'central', 'standardized'}
            Whether to return the raw (default), central, or standardized moment
            defined above.
        method : {None, 'formula', 'general', 'transform', 'normalize', 'quadrature', 'cache'}
            The strategy used to evaluate the moment. By default (``None``),
            the infrastructure chooses between the following options,
            listed in order of precedence.

            - ``'cache'``: use the value of the moment most recently calculated
              via another method
            - ``'formula'``: use a formula for the moment itself
            - ``'general'``: use a general result that is true for all distributions
              with finite moments; for instance, the zeroth raw moment is
              identically 1
            - ``'transform'``: transform a raw moment to a central moment or
              vice versa (see Notes)
            - ``'normalize'``: normalize a central moment to get a standardized
              or vice versa
            - ``'quadrature'``: numerically integrate according to the definition

            Not all `method` options are available for all orders, kinds, and
            distributions. If the selected `method` is not available, a
            ``NotImplementedError`` will be raised.

        Returns
        -------
        out : array
            The moment of the random variable of the specified order and kind.

        See Also
        --------
        pdf
        mean
        variance
        standard_deviation
        skewness
        kurtosis

        Notes
        -----
        Not all distributions have finite moments of all orders; moments of some
        orders may be undefined or infinite. If a formula for the moment is not
        specifically implemented for the chosen distribution, SciPy will attempt
        to compute the moment via a generic method, which may yield a finite
        result where none exists. This is not a critical bug, but an opportunity
        for an enhancement.

        The definition of a raw moment in the summary is specific to the raw moment
        about the origin. The raw moment about any point :math:`a` is:

        .. math::

            E[(X-a)^n] = \int_{\chi} (x-a)^n f(x) dx

        In this notation, a raw moment about the origin is :math:`\mu'_n = E[x^n]`,
        and a central moment is :math:`\mu_n = E[(x-\mu)^n]`, where :math:`\mu`
        is the first raw moment; i.e. the mean.

        The ``'transform'`` method takes advantage of the following relationships
        between moments taken about different points :math:`a` and :math:`b`.

        .. math::

            E[(X-b)^n] =  \sum_{i=0}^n E[(X-a)^i] {n \choose i} (a - b)^{n-i}

        For instance, to transform the raw moment to the central moment, we let
        :math:`b = \mu` and :math:`a = 0`.

        The distribution infrastructure provides flexibility for distribution
        authors to implement separate formulas for raw moments, central moments,
        and standardized moments of any order. By default, the moment of the
        desired order and kind is evaluated from the formula if such a formula
        is available; if not, the infrastructure uses any formulas that are
        available rather than resorting directly to numerical integration.
        For instance, if formulas for the first three raw moments are
        available and the third standardized moments is desired, the
        infrastructure will evaluate the raw moments and perform the transforms
        and standardization required. The decision tree is somewhat complex,
        but the strategy for obtaining a moment of a given order and kind
        (possibly as an intermediate step due to the recursive nature of the
        transform formula above) roughly follows this order of priority:

        #. Use cache (if order of same moment and kind has been calculated)
        #. Use formula (if available)
        #. Transform between raw and central moment and/or normalize to convert
           between central and standardized moments (if efficient)
        #. Use a generic result true for most distributions (if available)
        #. Use quadrature

        References
        ----------
        .. [1] Moment, *Wikipedia*,
               https://en.wikipedia.org/wiki/Moment_(mathematics)

        Examples
        --------
        Instantiate a distribution with the desired parameters:

        >>> from scipy import stats
        >>> X = stats.Normal(mu=1., sigma=2.)

        Evaluate the first raw moment:

        >>> X.moment(order=1, kind='raw')
        1.0
        >>> X.moment(order=1, kind='raw') == X.mean() == X.mu
        True

        Evaluate the second central moment:

        >>> X.moment(order=2, kind='central')
        4.0
        >>> X.moment(order=2, kind='central') == X.variance() == X.sigma**2
        True

        Evaluate the fourth standardized moment:

        >>> X.moment(order=4, kind='standardized')
        3.0
        >>> X.moment(order=4, kind='standardized') == X.kurtosis(convention='non-excess')
        True

        """  # noqa:E501
        raise NotImplementedError()

    @abstractmethod
    def mean(self, *, method):
        r"""Mean (raw first moment about the origin)

        Parameters
        ----------
        method : {None, 'formula', 'transform', 'quadrature', 'cache'}
            Method used to calculate the raw first moment. Not
            all methods are available for all distributions. See
            `moment` for details.

        See Also
        --------
        moment
        median
        mode

        References
        ----------
        .. [1] Mean, *Wikipedia*,
               https://en.wikipedia.org/wiki/Mean#Mean_of_a_probability_distribution

        Examples
        --------
        Instantiate a distribution with the desired parameters:

        >>> from scipy import stats
        >>> X = stats.Normal(mu=1., sigma=2.)

        Evaluate the variance:

        >>> X.mean()
        1.0
        >>> X.mean() == X.moment(order=1, kind='raw') == X.mu
        True

        """
        raise NotImplementedError()

    @abstractmethod
    def median(self, *, method):
        r"""Median (50th percentil)

        If a continuous random variable :math:`X` has probability :math:`0.5` of
        taking on a value less than :math:`m`, then :math:`m` is the median.
        That is, the median is the value :math:`m` for which:

        .. math::

            P(X ≤ m) = 0.5 = P(X ≥ m)

        Parameters
        ----------
        method : {None, 'formula', 'icdf'}
            The strategy used to evaluate the median.
            By default (``None``), the infrastructure chooses between the
            following options, listed in order of precedence.

            - ``'formula'``: use a formula for the median
            - ``'icdf'``: evaluate the inverse CDF of 0.5

            Not all `method` options are available for all distributions.
            If the selected `method` is not available, a ``NotImplementedError``
            will be raised.

        Returns
        -------
        out : array
            The median

        See Also
        --------
        mean
        mode
        icdf

        References
        ----------
        .. [1] Median, *Wikipedia*,
               https://en.wikipedia.org/wiki/Median#Probability_distributions

        Examples
        --------
        Instantiate a distribution with the desired parameters:

        >>> from scipy import stats
        >>> X = stats.Uniform(a=0., b=10.)

        Compute the median:

        >>> X.median()
        np.float64(5.0)
        >>> X.median() == X.icdf(0.5) == X.iccdf(0.5)
        True

        """
        raise NotImplementedError()

    @abstractmethod
    def mode(self, *, method):
        r"""Mode (most likely value)

        Informally, the mode is a value that a random variable has the highest
        probability (density) of assuming. That is, the mode is the element of
        the support :math:`\chi` that maximizes the probability density
        function :math:`f(x)`:

        .. math::

            \text{mode} = \arg\max_{x \in \chi} f(x)

        Parameters
        ----------
        method : {None, 'formula', 'optimization'}
            The strategy used to evaluate the mode.
            By default (``None``), the infrastructure chooses between the
            following options, listed in order of precedence.

            - ``'formula'``: use a formula for the median
            - ``'optimization'``: numerically maximize the PDF

            Not all `method` options are available for all distributions.
            If the selected `method` is not available, a ``NotImplementedError``
            will be raised.

        Returns
        -------
        out : array
            The mode

        See Also
        --------
        mean
        median
        pdf

        Notes
        -----
        For some distributions

        #. the mode is not unique (e.g. the uniform distribution);
        #. the PDF has one or more singularities, and it is debateable whether
           a singularity is considered to be in the domain and called the mode
           (e.g. the gamma distribution with shape parameter less than 1); and/or
        #. the probability density function may have one or more local maxima
           that are not a global maximum (e.g. mixture distributions).

        In such cases, `mode` will

        #. return a single value,
        #. consider the mode to occur at a singularity, and/or
        #. return a local maximum which may or may not be a global maximum.

        If a formula for the mode is not specifically implemented for the
        chosen distribution, SciPy will attempt to compute the mode
        numerically, which may not meet the user's preferred definition of a
        mode. In such cases, the user is encouraged to subclass the
        distribution and override ``mode``.

        References
        ----------
        .. [1] Mode (statistics), *Wikipedia*,
               https://en.wikipedia.org/wiki/Mode_(statistics)

        Examples
        --------
        Instantiate a distribution with the desired parameters:

        >>> from scipy import stats
        >>> X = stats.Normal(mu=1., sigma=2.)

        Evaluate the mode:

        >>> X.mode()
        1.0

        If the mode is not uniquely defined, ``mode`` nonetheless returns a
        single value.

        >>> X = stats.Uniform(a=0., b=1.)
        >>> X.mode()
        0.5

        If this choice does not satisfy your requirements, subclass the
        distribution and override ``mode``:

        >>> class BetterUniform(stats.Uniform):
        ...     def mode(self):
        ...         return self.b
        >>> X = BetterUniform(a=0., b=1.)
        >>> X.mode()
        1.0

        """
        raise NotImplementedError()

    @abstractmethod
    def variance(self, *, method):
        r"""Variance (central second moment)

        Parameters
        ----------
        method : {None, 'formula', 'transform', 'normalize', 'quadrature', 'cache'}
            Method used to calculate the central second moment. Not
            all methods are available for all distributions. See
            `moment` for details.

        See Also
        --------
        moment
        standard_deviation
        mean

        References
        ----------
        .. [1] Variance, *Wikipedia*,
               https://en.wikipedia.org/wiki/Variance#Absolutely_continuous_random_variable

        Examples
        --------
        Instantiate a distribution with the desired parameters:

        >>> from scipy import stats
        >>> X = stats.Normal(mu=1., sigma=2.)

        Evaluate the variance:

        >>> X.variance()
        4.0
        >>> X.variance() == X.moment(order=2, kind='central') == X.sigma**2
        True

        """
        raise NotImplementedError()

    @abstractmethod
    def standard_deviation(self, *, method):
        r"""Standard deviation (square root of the second central moment)

        Parameters
        ----------
        method : {None, 'formula', 'transform', 'normalize', 'quadrature', 'cache'}
            Method used to calculate the central second moment. Not
            all methods are available for all distributions. See
            `moment` for details.

        See Also
        --------
        variance
        mean
        moment

        References
        ----------
        .. [1] Standard deviation, *Wikipedia*,
               https://en.wikipedia.org/wiki/Standard_deviation#Definition_of_population_values

        Examples
        --------
        Instantiate a distribution with the desired parameters:

        >>> from scipy import stats
        >>> X = stats.Normal(mu=1., sigma=2.)

        Evaluate the standard deviation:

        >>> X.standard_deviation()
        2.0
        >>> X.standard_deviation() == X.moment(order=2, kind='central')**0.5 == X.sigma
        True

        """
        raise NotImplementedError()

    @abstractmethod
    def skewness(self, *, method):
        r"""Skewness (standardized third moment)

        Parameters
        ----------
        method : {None, 'formula', 'general', 'transform', 'normalize', 'cache'}
            Method used to calculate the standardized third moment. Not
            all methods are available for all distributions. See
            `moment` for details.

        See Also
        --------
        moment
        mean
        variance

        References
        ----------
        .. [1] Skewness, *Wikipedia*,
               https://en.wikipedia.org/wiki/Skewness

        Examples
        --------
        Instantiate a distribution with the desired parameters:

        >>> from scipy import stats
        >>> X = stats.Normal(mu=1., sigma=2.)

        Evaluate the skewness:

        >>> X.skewness()
        0.0
        >>> X.skewness() == X.moment(order=3, kind='standardized')
        True

        """
        raise NotImplementedError()

    @abstractmethod
    def kurtosis(self, *, method):
        r"""Kurtosis (standardized fourth moment)

        By default, this is the standardized fourth moment, also known as the
        "non-excess" or "Pearson" kurtosis (e.g. the kurtosis of the normal
        distribution is 3). The "excess" or "Fisher" kurtosis (the standardized
        fourth moment minus 3) is available via the `convention` parameter.

        Parameters
        ----------
        method : {None, 'formula', 'general', 'transform', 'normalize', 'cache'}
            Method used to calculate the standardized fourth moment. Not
            all methods are available for all distributions. See
            `moment` for details.
        convention : {'non-excess', 'excess'}
            Two distinct conventions are available:

            - ``'non-excess'``: the standardized fourth moment (Pearson's kurtosis)
            - ``'excess'``: the standardized fourth moment minus 3 (Fisher's kurtosis)

            The default is ``'non-excess'``.

        See Also
        --------
        moment
        mean
        variance

        References
        ----------
        .. [1] Kurtosis, *Wikipedia*,
               https://en.wikipedia.org/wiki/Kurtosis

        Examples
        --------
        Instantiate a distribution with the desired parameters:

        >>> from scipy import stats
        >>> X = stats.Normal(mu=1., sigma=2.)

        Evaluate the kurtosis:

        >>> X.kurtosis()
        3.0
        >>> (X.kurtosis()
        ...  == X.kurtosis(convention='excess') + 3.
        ...  == X.moment(order=4, kind='standardized'))
        True

        """
        raise NotImplementedError()

    @abstractmethod
    def pdf(self, x, /, *, method):
        r"""Probability density function

        The probability density function ("PDF"), denoted :math:`f(x)`, is the
        probability *per unit length* that the random variable will assume the
        value :math:`x`. Mathematically, it can be defined as the derivative
        of the cumulative distribution function :math:`F(x)`:

        .. math::

            f(x) = \frac{d}{dx} F(x)

        `pdf` accepts `x` for :math:`x`.

        Parameters
        ----------
        x : array_like
            The argument of the PDF.
        method : {None, 'formula', 'logexp'}
            The strategy used to evaluate the PDF. By default (``None``), the
            infrastructure chooses between the following options, listed in
            order of precedence.

            - ``'formula'``: use a formula for the PDF itself
            - ``'logexp'``: evaluate the log-PDF and exponentiate

            Not all `method` options are available for all distributions.
            If the selected `method` is not available, a ``NotImplementedError``
            will be raised.

        Returns
        -------
        out : array
            The PDF evaluated at the argument `x`.

        See Also
        --------
        cdf
        logpdf

        Notes
        -----
        Suppose a continuous probability distribution has support :math:`[l, r]`.
        By definition of the support, the PDF evaluates to its minimum value
        of :math:`0` outside the support; i.e. for :math:`x < l` or
        :math:`x > r`. The maximum of the PDF may be less than or greater than
        :math:`1`; since the valus is a probability *density*, only its integral
        over the support must equal :math:`1`.

        References
        ----------
        .. [1] Probability density function, *Wikipedia*,
               https://en.wikipedia.org/wiki/Probability_density_function

        Examples
        --------
        Instantiate a distribution with the desired parameters:

        >>> from scipy import stats
        >>> X = stats.Uniform(a=-1., b=1.)

        Evaluate the PDF at the desired argument:

        >>> X.pdf(0.25)
        0.5

        """
        raise NotImplementedError()

    @abstractmethod
    def logpdf(self, x, /, *, method):
        r"""Log of the probability density function

        The probability density function ("PDF"), denoted :math:`f(x)`, is the
        probability *per unit length* that the random variable will assume the
        value :math:`x`. Mathematically, it can be defined as the derivative
        of the cumulative distribution function :math:`F(x)`:

        .. math::

            f(x) = \frac{d}{dx} F(x)

        `logpdf` computes the logarithm of the probability density function
        ("log-PDF"), :math:`\log(f(x))`, but it may be numerically favorable
        compared to the naive implementation (computing :math:`f(x)` and
        taking the logarithm).

        `logpdf` accepts `x` for :math:`x`.

        Parameters
        ----------
        x : array_like
            The argument of the log-PDF.
        method : {None, 'formula', 'logexp'}
            The strategy used to evaluate the log-PDF. By default (``None``), the
            infrastructure chooses between the following options, listed in order
            of precedence.

            - ``'formula'``: use a formula for the log-PDF itself
            - ``'logexp'``: evaluate the PDF and takes its logarithm

            Not all `method` options are available for all distributions.
            If the selected `method` is not available, a ``NotImplementedError``
            will be raised.

        Returns
        -------
        out : array
            The log-PDF evaluated at the argument `x`.

        See Also
        --------
        pdf
        logcdf

        Notes
        -----
        Suppose a continuous probability distribution has support :math:`[l, r]`.
        By definition of the support, the log-PDF evaluates to its minimum value
        of :math:`-\infty` (i.e. :math:`\log(0)`) outside the support; i.e. for
        :math:`x < l` or :math:`x > r`. The maximum of the log-PDF may be less
        than or greater than :math:`\log(1) = 0` because the maximum of the PDF
        can be any positive real.

        For distributions with infinite support, it is common for `pdf` to return
        a value of ``0`` when the argument is theoretically within the support;
        this can occur because the true value of the PDF is too small to be
        represented by the chosen dtype. The log-PDF, however, will often be finite
        (not ``-inf``) over a much larger domain. Consequently, it may be preferred
        to work with the logarithms of probabilities and probability densities to
        avoid underflow.

        References
        ----------
        .. [1] Probability density function, *Wikipedia*,
               https://en.wikipedia.org/wiki/Probability_density_function

        Examples
        --------
        Instantiate a distribution with the desired parameters:

        >>> import numpy as np
        >>> from scipy import stats
        >>> X = stats.Uniform(a=-1.0, b=1.0)

        Evaluate the log-PDF at the desired argument:

        >>> X.logpdf(0.5)
        -0.6931471805599453
        >>> np.allclose(X.logpdf(0.5), np.log(X.pdf(0.5)))
        True

        """
        raise NotImplementedError()

    @abstractmethod
    def cdf(self, x, y, /, *, method):
        r"""Cumulative distribution function

        The cumulative distribution function ("CDF"), denoted :math:`F(x)`, is
        the probability the random variable :math:`X` will assume a value
        less than or equal to :math:`x`:

        .. math::

            F(x) = P(X ≤ x)

        A two-argument variant of this function is also defined as the
        probability the random variable :math:`X` will assume a value between
        :math:`x` and :math:`y`.

        .. math::

            F(x, y) = P(x ≤ X ≤ y)

        `cdf` accepts `x` for :math:`x` and `y` for :math:`y`.

        Parameters
        ----------
        x, y : array_like
            The arguments of the CDF. `x` is required; `y` is optional.
        method : {None, 'formula', 'logexp', 'complement', 'quadrature', 'subtraction'}
            The strategy used to evaluate the CDF.
            By default (``None``), the one-argument form of the function
            chooses between the following options, listed in order of precedence.

            - ``'formula'``: use a formula for the CDF itself
            - ``'logexp'``: evaluate the log-CDF and exponentiate
            - ``'complement'``: evaluate the CCDF and take the complement
            - ``'quadrature'``: numerically integrate the PDF

            In place of ``'complement'``, the two-argument form accepts:

            - ``'subtraction'``: compute the CDF at each argument and take
              the difference.

            Not all `method` options are available for all distributions.
            If the selected `method` is not available, a ``NotImplementedError``
            will be raised.

        Returns
        -------
        out : array
            The CDF evaluated at the provided argument(s).

        See Also
        --------
        logcdf
        ccdf

        Notes
        -----
        Suppose a continuous probability distribution has support :math:`[l, r]`.
        The CDF :math:`F(x)` is related to the probability density function
        :math:`f(x)` by:

        .. math::

            F(x) = \int_l^x f(u) du

        The two argument version is:

        .. math::

            F(x, y) = \int_x^y f(u) du = F(y) - F(x)

        The CDF evaluates to its minimum value of :math:`0` for :math:`x ≤ l`
        and its maximum value of :math:`1` for :math:`x ≥ r`.

        The CDF is also known simply as the "distribution function".

        References
        ----------
        .. [1] Cumulative distribution function, *Wikipedia*,
               https://en.wikipedia.org/wiki/Cumulative_distribution_function

        Examples
        --------
        Instantiate a distribution with the desired parameters:

        >>> from scipy import stats
        >>> X = stats.Uniform(a=-0.5, b=0.5)

        Evaluate the CDF at the desired argument:

        >>> X.cdf(0.25)
        0.75

        Evaluate the cumulative probability between two arguments:

        >>> X.cdf(-0.25, 0.25) == X.cdf(0.25) - X.cdf(-0.25)
        True

        """  # noqa: E501
        raise NotImplementedError()

    @abstractmethod
    def icdf(self, p, /, *, method):
        r"""Inverse of the cumulative distribution function.

        The inverse of the cumulative distribution function ("inverse CDF"),
        denoted :math:`F^{-1}(p)`, is the argument :math:`x` for which the
        cumulative distribution function :math:`F(x)` evaluates to :math:`p`.

        .. math::

            F^{-1}(p) = x \quad \text{s.t.} \quad F(x) = p

        `icdf` accepts `p` for :math:`p \in [0, 1]`.

        Parameters
        ----------
        p : array_like
            The argument of the inverse CDF.
        method : {None, 'formula', 'complement', 'inversion'}
            The strategy used to evaluate the inverse CDF.
            By default (``None``), the infrastructure chooses between the
            following options, listed in order of precedence.

            - ``'formula'``: use a formula for the inverse CDF itself
            - ``'complement'``: evaluate the inverse CCDF at the
              complement of `p`
            - ``'inversion'``: solve numerically for the argument at which the
              CDF is equal to `p`

            Not all `method` options are available for all distributions.
            If the selected `method` is not available, a ``NotImplementedError``
            will be raised.

        Returns
        -------
        out : array
            The inverse CDF evaluated at the provided argument.

        See Also
        --------
        cdf
        ilogcdf

        Notes
        -----
        Suppose a continuous probability distribution has support :math:`[l, r]`. The
        inverse CDF returns its minimum value of :math:`l` at :math:`p = 0`
        and its maximum value of :math:`r` at :math:`p = 1`. Because the CDF
        has range :math:`[0, 1]`, the inverse CDF is only defined on the
        domain :math:`[0, 1]`; for :math:`p < 0` and :math:`p > 1`, `icdf`
        returns ``nan``.

        The inverse CDF is also known as the quantile function, percentile function,
        and percent-point function.

        References
        ----------
        .. [1] Quantile function, *Wikipedia*,
               https://en.wikipedia.org/wiki/Quantile_function

        Examples
        --------
        Instantiate a distribution with the desired parameters:

        >>> import numpy as np
        >>> from scipy import stats
        >>> X = stats.Uniform(a=-0.5, b=0.5)

        Evaluate the inverse CDF at the desired argument:

        >>> X.icdf(0.25)
        -0.25
        >>> np.allclose(X.cdf(X.icdf(0.25)), 0.25)
        True

        This function returns NaN when the argument is outside the domain.

        >>> X.icdf([-0.1, 0, 1, 1.1])
        array([ nan, -0.5,  0.5,  nan])

        """
        raise NotImplementedError()

    @abstractmethod
    def ccdf(self, x, y, /, *, method):
        r"""Complementary cumulative distribution function

        The complementary cumulative distribution function ("CCDF"), denoted
        :math:`G(x)`, is the complement of the cumulative distribution function
        :math:`F(x)`; i.e., probability the random variable :math:`X` will
        assume a value greater than :math:`x`:

        .. math::

            G(x) = 1 - F(x) = P(X > x)

        A two-argument variant of this function is:

        .. math::

            G(x, y) = 1 - F(x, y) = P(X < x \text{ or } X > y)

        `ccdf` accepts `x` for :math:`x` and `y` for :math:`y`.

        Parameters
        ----------
        x, y : array_like
            The arguments of the CCDF. `x` is required; `y` is optional.
        method : {None, 'formula', 'logexp', 'complement', 'quadrature', 'addition'}
            The strategy used to evaluate the CCDF.
            By default (``None``), the infrastructure chooses between the
            following options, listed in order of precedence.

            - ``'formula'``: use a formula for the CCDF itself
            - ``'logexp'``: evaluate the log-CCDF and exponentiate
            - ``'complement'``: evaluate the CDF and take the complement
            - ``'quadrature'``: numerically integrate the PDF

            The two-argument form chooses between:

            - ``'formula'``: use a formula for the CCDF itself
            - ``'addition'``: compute the CDF at `x` and the CCDF at `y`, then add

            Not all `method` options are available for all distributions.
            If the selected `method` is not available, a ``NotImplementedError``
            will be raised.

        Returns
        -------
        out : array
            The CCDF evaluated at the provided argument(s).

        See Also
        --------
        cdf
        logccdf

        Notes
        -----
        Suppose a continuous probability distribution has support :math:`[l, r]`.
        The CCDF :math:`G(x)` is related to the probability density function
        :math:`f(x)` by:

        .. math::

            G(x) = \int_x^r f(u) du

        The two argument version is:

        .. math::

            G(x, y) = \int_l^x f(u) du + \int_y^r f(u) du

        The CCDF returns its minimum value of :math:`0` for :math:`x ≥ r`
        and its maximum value of :math:`1` for :math:`x ≤ l`.

        The CCDF is also known as the "survival function".

        References
        ----------
        .. [1] Cumulative distribution function, *Wikipedia*,
               https://en.wikipedia.org/wiki/Cumulative_distribution_function#Derived_functions

        Examples
        --------
        Instantiate a distribution with the desired parameters:

        >>> import numpy as np
        >>> from scipy import stats
        >>> X = stats.Uniform(a=-0.5, b=0.5)

        Evaluate the CCDF at the desired argument:

        >>> X.ccdf(0.25)
        0.25
        >>> np.allclose(X.ccdf(0.25), 1-X.cdf(0.25))
        True

        Evaluate the complement of the cumulative probability between two arguments:

        >>> X.ccdf(-0.25, 0.25) == X.cdf(-0.25) + X.ccdf(0.25)
        True

        """  # noqa: E501
        raise NotImplementedError()

    @abstractmethod
    def iccdf(self, p, /, *, method):
        r"""Inverse complementary cumulative distribution function.

        The inverse complementary cumulative distribution function ("inverse CCDF"),
        denoted :math:`G^{-1}(p)`, is the argument :math:`x` for which the
        complementary cumulative distribution function :math:`G(x)` evaluates to
        :math:`p`.

        .. math::

            G^{-1}(p) = x \quad \text{s.t.} \quad G(x) = p

        `iccdf` accepts `p` for :math:`p \in [0, 1]`.

        Parameters
        ----------
        p : array_like
            The argument of the inverse CCDF.
        method : {None, 'formula', 'complement', 'inversion'}
            The strategy used to evaluate the inverse CCDF.
            By default (``None``), the infrastructure chooses between the
            following options, listed in order of precedence.

            - ``'formula'``: use a formula for the inverse CCDF itself
            - ``'complement'``: evaluate the inverse CDF at the
              complement of `p`
            - ``'inversion'``: solve numerically for the argument at which the
              CCDF is equal to `p`

            Not all `method` options are available for all distributions.
            If the selected `method` is not available, a ``NotImplementedError``
            will be raised.

        Returns
        -------
        out : array
            The inverse CCDF evaluated at the provided argument.

        Notes
        -----
        Suppose a continuous probability distribution has support :math:`[l, r]`. The
        inverse CCDF returns its minimum value of :math:`l` at :math:`p = 1`
        and its maximum value of :math:`r` at :math:`p = 0`. Because the CCDF
        has range :math:`[0, 1]`, the inverse CCDF is only defined on the
        domain :math:`[0, 1]`; for :math:`p < 0` and :math:`p > 1`, ``iccdf``
        returns ``nan``.

        See Also
        --------
        icdf
        ilogccdf

        Examples
        --------
        Instantiate a distribution with the desired parameters:

        >>> import numpy as np
        >>> from scipy import stats
        >>> X = stats.Uniform(a=-0.5, b=0.5)

        Evaluate the inverse CCDF at the desired argument:

        >>> X.iccdf(0.25)
        0.25
        >>> np.allclose(X.iccdf(0.25), X.icdf(1-0.25))
        True

        This function returns NaN when the argument is outside the domain.

        >>> X.iccdf([-0.1, 0, 1, 1.1])
        array([ nan,  0.5, -0.5,  nan])

        """
        raise NotImplementedError()

    @abstractmethod
    def logcdf(self, x, y, /, *, method):
        r"""Log of the cumulative distribution function

        The cumulative distribution function ("CDF"), denoted :math:`F(x)`, is
        the probability the random variable :math:`X` will assume a value
        less than or equal to :math:`x`:

        .. math::

            F(x) = P(X ≤ x)

        A two-argument variant of this function is also defined as the
        probability the random variable :math:`X` will assume a value between
        :math:`x` and :math:`y`.

        .. math::

            F(x, y) = P(x ≤ X ≤ y)

        `logcdf` computes the logarithm of the cumulative distribution function
        ("log-CDF"), :math:`\log(F(x))`/:math:`\log(F(x, y))`, but it may be
        numerically favorable compared to the naive implementation (computing
        the CDF and taking the logarithm).

        `logcdf` accepts `x` for :math:`x` and `y` for :math:`y`.

        Parameters
        ----------
        x, y : array_like
            The arguments of the log-CDF. `x` is required; `y` is optional.
        method : {None, 'formula', 'logexp', 'complement', 'quadrature', 'subtraction'}
            The strategy used to evaluate the log-CDF.
            By default (``None``), the one-argument form of the function
            chooses between the following options, listed in order of precedence.

            - ``'formula'``: use a formula for the log-CDF itself
            - ``'logexp'``: evaluate the CDF and take the logarithm
            - ``'complement'``: evaluate the log-CCDF and take the
              logarithmic complement (see Notes)
            - ``'quadrature'``: numerically log-integrate the log-PDF

            In place of ``'complement'``, the two-argument form accepts:

            - ``'subtraction'``: compute the log-CDF at each argument and take
              the logarithmic difference (see Notes)

            Not all `method` options are available for all distributions.
            If the selected `method` is not available, a ``NotImplementedError``
            will be raised.

        Returns
        -------
        out : array
            The log-CDF evaluated at the provided argument(s).

        See Also
        --------
        cdf
        logccdf

        Notes
        -----
        Suppose a continuous probability distribution has support :math:`[l, r]`.
        The log-CDF evaluates to its minimum value of :math:`\log(0) = -\infty`
        for :math:`x ≤ l` and its maximum value of :math:`\log(1) = 0` for
        :math:`x ≥ r`.

        For distributions with infinite support, it is common for
        `cdf` to return a value of ``0`` when the argument
        is theoretically within the support; this can occur because the true value
        of the CDF is too small to be represented by the chosen dtype. `logcdf`,
        however, will often return a finite (not ``-inf``) result over a much larger
        domain. Similarly, `logcdf` may provided a strictly negative result with
        arguments for which `cdf` would return ``1.0``. Consequently, it may be
        preferred to work with the logarithms of probabilities to avoid underflow
        and related limitations of floating point numbers.

        The "logarithmic complement" of a number :math:`z` is mathematically
        equivalent to :math:`\log(1-\exp(z))`, but it is computed to avoid loss
        of precision when :math:`\exp(z)` is nearly :math:`0` or :math:`1`.
        Similarly, the term "logarithmic difference" of :math:`w` and :math:`z`
        is used here to mean :math:`\log(\exp(w)-\exp(z))`.

        If ``y < x``, the CDF is negative, and therefore the log-CCDF
        is complex with imaginary part :math:`\pi`. For
        consistency, the result of this function always has complex dtype
        when `y` is provided, regardless of the value of the imaginary part.

        References
        ----------
        .. [1] Cumulative distribution function, *Wikipedia*,
               https://en.wikipedia.org/wiki/Cumulative_distribution_function

        Examples
        --------
        Instantiate a distribution with the desired parameters:

        >>> import numpy as np
        >>> from scipy import stats
        >>> X = stats.Uniform(a=-0.5, b=0.5)

        Evaluate the log-CDF at the desired argument:

        >>> X.logcdf(0.25)
        -0.287682072451781
        >>> np.allclose(X.logcdf(0.), np.log(X.cdf(0.)))
        True

        """  # noqa: E501
        raise NotImplementedError()

    @abstractmethod
    def ilogcdf(self, logp, /, *, method):
        r"""Inverse of the logarithm of the cumulative distribution function.

        The inverse of the logarithm of the cumulative distribution function
        ("inverse log-CDF") is the argument :math:`x` for which the logarithm
        of the cumulative distribution function :math:`\log(F(x))` evaluates
        to :math:`\log(p)`.

        Mathematically, it is equivalent to :math:`F^{-1}(\exp(y))`, where
        :math:`y = \log(p)`, but it may be numerically favorable compared to
        the naive implementation (computing :math:`p = \exp(y)`, then
        :math:`F^{-1}(p)`).

        `ilogcdf` accepts `logp` for :math:`\log(p) ≤ 0`.

        Parameters
        ----------
        logp : array_like
            The argument of the inverse log-CDF.
        method : {None, 'formula', 'complement', 'inversion'}
            The strategy used to evaluate the inverse log-CDF.
            By default (``None``), the infrastructure chooses between the
            following options, listed in order of precedence.

            - ``'formula'``: use a formula for the inverse log-CDF itself
            - ``'complement'``: evaluate the inverse log-CCDF at the
              logarithmic complement of `logp` (see Notes)
            - ``'inversion'``: solve numerically for the argument at which the
              log-CDF is equal to `logp`

            Not all `method` options are available for all distributions.
            If the selected `method` is not available, a ``NotImplementedError``
            will be raised.

        Returns
        -------
        out : array
            The inverse log-CDF evaluated at the provided argument.

        See Also
        --------
        icdf
        logcdf

        Notes
        -----
        Suppose a continuous probability distribution has support :math:`[l, r]`.
        The inverse log-CDF returns its minimum value of :math:`l` at
        :math:`\log(p) = \log(0) = -\infty` and its maximum value of :math:`r` at
        :math:`\log(p) = \log(1) = 0`. Because the log-CDF has range
        :math:`[-\infty, 0]`, the inverse log-CDF is only defined on the
        negative reals; for :math:`\log(p) > 0`, `ilogcdf` returns ``nan``.

        Occasionally, it is needed to find the argument of the CDF for which
        the resulting probability is very close to ``0`` or ``1`` - too close to
        represent accurately with floating point arithmetic. In many cases,
        however, the *logarithm* of this resulting probability may be
        represented in floating point arithmetic, in which case this function
        may be used to find the argument of the CDF for which the *logarithm*
        of the resulting probability is :math:`y = \log(p)`.

        The "logarithmic complement" of a number :math:`z` is mathematically
        equivalent to :math:`\log(1-\exp(z))`, but it is computed to avoid loss
        of precision when :math:`\exp(z)` is nearly :math:`0` or :math:`1`.

        Examples
        --------
        Instantiate a distribution with the desired parameters:

        >>> import numpy as np
        >>> from scipy import stats
        >>> X = stats.Uniform(a=-0.5, b=0.5)

        Evaluate the inverse log-CDF at the desired argument:

        >>> X.ilogcdf(-0.25)
        0.2788007830714034
        >>> np.allclose(X.ilogcdf(-0.25), X.icdf(np.exp(-0.25)))
        True

        """
        raise NotImplementedError()

    @abstractmethod
    def logccdf(self, x, y, /, *, method):
        r"""Log of the complementary cumulative distribution function

        The complementary cumulative distribution function ("CCDF"), denoted
        :math:`G(x)` is the complement of the cumulative distribution function
        :math:`F(x)`; i.e., probability the random variable :math:`X` will
        assume a value greater than :math:`x`:

        .. math::

            G(x) = 1 - F(x) = P(X > x)

        A two-argument variant of this function is:

        .. math::

            G(x, y) = 1 - F(x, y) = P(X < x \quad \text{or} \quad X > y)

        `logccdf` computes the logarithm of the complementary cumulative
        distribution function ("log-CCDF"), :math:`\log(G(x))`/:math:`\log(G(x, y))`,
        but it may be numerically favorable compared to the naive implementation
        (computing the CDF and taking the logarithm).

        `logccdf` accepts `x` for :math:`x` and `y` for :math:`y`.

        Parameters
        ----------
        x, y : array_like
            The arguments of the log-CCDF. `x` is required; `y` is optional.
        method : {None, 'formula', 'logexp', 'complement', 'quadrature', 'addition'}
            The strategy used to evaluate the log-CCDF.
            By default (``None``), the one-argument form of the function
            chooses between the following options, listed in order of precedence.

            - ``'formula'``: use a formula for the log CCDF itself
            - ``'logexp'``: evaluate the CCDF and take the logarithm
            - ``'complement'``: evaluate the log-CDF and take the
              logarithmic complement (see Notes)
            - ``'quadrature'``: numerically log-integrate the log-PDF

            The two-argument form chooses between:

            - ``'formula'``: use a formula for the log CCDF itself
            - ``'addition'``: compute the log-CDF at `x` and the log-CCDF at `y`,
              then take the logarithmic sum (see Notes)

            Not all `method` options are available for all distributions.
            If the selected `method` is not available, a ``NotImplementedError``
            will be raised.

        Returns
        -------
        out : array
            The log-CCDF evaluated at the provided argument(s).

        See Also
        --------
        ccdf
        logcdf

        Notes
        -----
        Suppose a continuous probability distribution has support :math:`[l, r]`.
        The log-CCDF returns its minimum value of :math:`\log(0)=-\infty` for
        :math:`x ≥ r` and its maximum value of :math:`\log(1) = 0` for
        :math:`x ≤ l`.

        For distributions with infinite support, it is common for
        `ccdf` to return a value of ``0`` when the argument
        is theoretically within the support; this can occur because the true value
        of the CCDF is too small to be represented by the chosen dtype. The log
        of the CCDF, however, will often be finite (not ``-inf``) over a much larger
        domain. Similarly, `logccdf` may provided a strictly negative result with
        arguments for which `ccdf` would return ``1.0``. Consequently, it may be
        preferred to work with the logarithms of probabilities to avoid underflow
        and related limitations of floating point numbers.

        The "logarithmic complement" of a number :math:`z` is mathematically
        equivalent to :math:`\log(1-\exp(z))`, but it is computed to avoid loss
        of precision when :math:`\exp(z)` is nearly :math:`0` or :math:`1`.
        Similarly, the term "logarithmic sum" of :math:`w` and :math:`z`
        is used here to mean the :math:`\log(\exp(w)+\exp(z))`, AKA
        :math:`\text{LogSumExp}(w, z)`.

        References
        ----------
        .. [1] Cumulative distribution function, *Wikipedia*,
               https://en.wikipedia.org/wiki/Cumulative_distribution_function#Derived_functions

        Examples
        --------
        Instantiate a distribution with the desired parameters:

        >>> import numpy as np
        >>> from scipy import stats
        >>> X = stats.Uniform(a=-0.5, b=0.5)

        Evaluate the log-CCDF at the desired argument:

        >>> X.logccdf(0.25)
        -1.3862943611198906
        >>> np.allclose(X.logccdf(0.), np.log(X.ccdf(0.)))
        True

        """  # noqa: E501
        raise NotImplementedError()

    @abstractmethod
    def ilogccdf(self, logp, /, *, method):
        r"""Inverse of the log of the complementary cumulative distribution function.

        The inverse of the logarithm of the complementary cumulative distribution
        function ("inverse log-CCDF") is the argument :math:`x` for which the logarithm
        of the complementary cumulative distribution function :math:`\log(G(x))`
        evaluates to :math:`\log(p)`.

        Mathematically, it is equivalent to :math:`G^{-1}(\exp(y))`, where
        :math:`y = \log(p)`, but it may be numerically favorable compared to the naive
        implementation (computing :math:`p = \exp(y)`, then :math:`G^{-1}(p)`).

        `ilogccdf` accepts `logp` for :math:`\log(p) ≤ 0`.

        Parameters
        ----------
        x : array_like
            The argument of the inverse log-CCDF.
        method : {None, 'formula', 'complement', 'inversion'}
            The strategy used to evaluate the inverse log-CCDF.
            By default (``None``), the infrastructure chooses between the
            following options, listed in order of precedence.

            - ``'formula'``: use a formula for the inverse log-CCDF itself
            - ``'complement'``: evaluate the inverse log-CDF at the
              logarithmic complement of `x` (see Notes)
            - ``'inversion'``: solve numerically for the argument at which the
              log-CCDF is equal to `x`

            Not all `method` options are available for all distributions.
            If the selected `method` is not available, a ``NotImplementedError``
            will be raised.

        Returns
        -------
        out : array
            The inverse log-CCDF evaluated at the provided argument.

        Notes
        -----
        Suppose a continuous probability distribution has support :math:`[l, r]`. The
        inverse log-CCDF returns its minimum value of :math:`l` at
        :math:`\log(p) = \log(1) = 0` and its maximum value of :math:`r` at
        :math:`\log(p) = \log(0) = -\infty`. Because the log-CCDF has range
        :math:`[-\infty, 0]`, the inverse log-CDF is only defined on the
        negative reals; for :math:`\log(p) > 0`, `ilogccdf` returns ``nan``.

        Occasionally, it is needed to find the argument of the CCDF for which
        the resulting probability is very close to ``0`` or ``1`` - too close to
        represent accurately with floating point arithmetic. In many cases,
        however, the *logarithm* of this resulting probability may be
        represented in floating point arithmetic, in which case this function
        may be used to find the argument of the CCDF for which the *logarithm*
        of the resulting probability is `y = \log(p)`.

        The "logarithmic complement" of a number :math:`z` is mathematically
        equivalent to :math:`\log(1-\exp(z))`, but it is computed to avoid loss
        of precision when :math:`\exp(z)` is nearly :math:`0` or :math:`1`.

        See Also
        --------
        iccdf
        ilogccdf

        Examples
        --------
        Instantiate a distribution with the desired parameters:

        >>> import numpy as np
        >>> from scipy import stats
        >>> X = stats.Uniform(a=-0.5, b=0.5)

        Evaluate the inverse log-CCDF at the desired argument:

        >>> X.ilogccdf(-0.25)
        -0.2788007830714034
        >>> np.allclose(X.ilogccdf(-0.25), X.iccdf(np.exp(-0.25)))
        True

        """
        raise NotImplementedError()

    @abstractmethod
    def logentropy(self, *, method):
        r"""Logarithm of the differential entropy

        In terms of probability density function :math:`f(x)` and support
        :math:`\chi`, the differential entropy (or simply "entropy") of a random
        variable :math:`X` is:

        .. math::

            h(X) = - \int_{\chi} f(x) \log f(x) dx

        `logentropy` computes the logarithm of the differential entropy
        ("log-entropy"), :math:`log(h(X))`, but it may be numerically favorable
        compared to the naive implementation (computing :math:`h(X)` then
        taking the logarithm).

        Parameters
        ----------
        method : {None, 'formula', 'logexp', 'quadrature}
            The strategy used to evaluate the log-entropy. By default
            (``None``), the infrastructure chooses between the following options,
            listed in order of precedence.

            - ``'formula'``: use a formula for the log-entropy itself
            - ``'logexp'``: evaluate the entropy and take the logarithm
            - ``'quadrature'``: numerically log-integrate the logarithm of the
              entropy integrand

            Not all `method` options are available for all distributions.
            If the selected `method` is not available, a ``NotImplementedError``
            will be raised.

        Returns
        -------
        out : array
            The log-entropy.

        See Also
        --------
        entropy
        logpdf

        Notes
        -----
        If the entropy of a distribution is negative, then the log-entropy
        is complex with imaginary part :math:`\pi`. For
        consistency, the result of this function always has complex dtype,
        regardless of the value of the imaginary part.

        References
        ----------
        .. [1] Differential entropy, *Wikipedia*,
               https://en.wikipedia.org/wiki/Differential_entropy

        Examples
        --------
        Instantiate a distribution with the desired parameters:

        >>> import numpy as np
        >>> from scipy import stats
        >>> X = stats.Uniform(a=-1., b=1.)

        Evaluate the log-entropy:

        >>> X.logentropy()
        (-0.3665129205816642+0j)
        >>> np.allclose(np.exp(X.logentropy()), X.entropy())
        True

        For a random variable with negative entropy, the log-entropy has an
        imaginary part equal to `np.pi`.

        >>> X = stats.Uniform(a=-.1, b=.1)
        >>> X.entropy(), X.logentropy()
        (-1.6094379124341007, (0.4758849953271105+3.141592653589793j))

        """
        raise NotImplementedError()
        
    @abstractmethod
    def entropy(self, *, method):
        r"""Differential entropy

        In terms of probability density function :math:`f(x)` and support
        :math:`\chi`, the differential entropy (or simply "entropy") of a
        continuous random variable :math:`X` is:

        .. math::

            h(X) = - \int_{\chi} f(x) \log f(x) dx

        Parameters
        ----------
        method : {None, 'formula', 'logexp', 'quadrature'}
            The strategy used to evaluate the entropy. By default (``None``),
            the infrastructure chooses between the following options, listed
            in order of precedence.

            - ``'formula'``: use a formula for the entropy itself
            - ``'logexp'``: evaluate the log-entropy and exponentiate
            - ``'quadrature'``: use numerical integration

            Not all `method` options are available for all distributions.
            If the selected `method` is not available, a ``NotImplementedError``
            will be raised.

        Returns
        -------
        out : array
            The entropy of the random variable.

        See Also
        --------
        logentropy
        pdf

        Notes
        -----
        This function calculates the entropy using the natural logarithm; i.e.
        the logarithm with base :math:`e`. Consequently, the value is expressed
        in (dimensionless) "units" of nats. To convert the entropy to different
        units (i.e. corresponding with a different base), divide the result by
        the natural logarithm of the desired base.

        References
        ----------
        .. [1] Differential entropy, *Wikipedia*,
               https://en.wikipedia.org/wiki/Differential_entropy

        Examples
        --------
        Instantiate a distribution with the desired parameters:

        >>> from scipy import stats
        >>> X = stats.Uniform(a=-1., b=1.)

        Evaluate the entropy:

        >>> X.entropy()
        0.6931471805599454

        """
        raise NotImplementedError()
<|endoftext|>import math
import warnings
import threading
from collections import namedtuple

import numpy as np
from numpy import (isscalar, r_, log, around, unique, asarray, zeros,
                   arange, sort, amin, amax, sqrt, array,
                   pi, exp, ravel, count_nonzero)

from scipy import optimize, special, interpolate, stats
from scipy._lib._bunch import _make_tuple_bunch
from scipy._lib._util import _rename_parameter, _contains_nan, _get_nan

from scipy._lib._array_api import (
    array_namespace,
    xp_size,
    xp_moveaxis_to_end,
    xp_vector_norm,
)

from ._ansari_swilk_statistics import gscale, swilk
from . import _stats_py, _wilcoxon
from ._fit import FitResult
from ._stats_py import (find_repeats, _get_pvalue, SignificanceResult,  # noqa:F401
                        _SimpleNormal, _SimpleChi2)
from .contingency import chi2_contingency
from . import distributions
from ._distn_infrastructure import rv_generic
from ._axis_nan_policy import _axis_nan_policy_factory, _broadcast_arrays


__all__ = ['mvsdist',
           'bayes_mvs', 'kstat', 'kstatvar', 'probplot', 'ppcc_max', 'ppcc_plot',
           'boxcox_llf', 'boxcox', 'boxcox_normmax', 'boxcox_normplot',
           'shapiro', 'anderson', 'ansari', 'bartlett', 'levene',
           'fligner', 'mood', 'wilcoxon', 'median_test',
           'circmean', 'circvar', 'circstd', 'anderson_ksamp',
           'yeojohnson_llf', 'yeojohnson', 'yeojohnson_normmax',
           'yeojohnson_normplot', 'directional_stats',
           'false_discovery_control'
           ]


Mean = namedtuple('Mean', ('statistic', 'minmax'))
Variance = namedtuple('Variance', ('statistic', 'minmax'))
Std_dev = namedtuple('Std_dev', ('statistic', 'minmax'))


def bayes_mvs(data, alpha=0.90):
    r"""
    Bayesian confidence intervals for the mean, var, and std.

    Parameters
    ----------
    data : array_like
        Input data, if multi-dimensional it is flattened to 1-D by `bayes_mvs`.
        Requires 2 or more data points.
    alpha : float, optional
        Probability that the returned confidence interval contains
        the true parameter.

    Returns
    -------
    mean_cntr, var_cntr, std_cntr : tuple
        The three results are for the mean, variance and standard deviation,
        respectively.  Each result is a tuple of the form::

            (center, (lower, upper))

        with ``center`` the mean of the conditional pdf of the value given the
        data, and ``(lower, upper)`` a confidence interval, centered on the
        median, containing the estimate to a probability ``alpha``.

    See Also
    --------
    mvsdist

    Notes
    -----
    Each tuple of mean, variance, and standard deviation estimates represent
    the (center, (lower, upper)) with center the mean of the conditional pdf
    of the value given the data and (lower, upper) is a confidence interval
    centered on the median, containing the estimate to a probability
    ``alpha``.

    Converts data to 1-D and assumes all data has the same mean and variance.
    Uses Jeffrey's prior for variance and std.

    Equivalent to ``tuple((x.mean(), x.interval(alpha)) for x in mvsdist(dat))``

    References
    ----------
    T.E. Oliphant, "A Bayesian perspective on estimating mean, variance, and
    standard-deviation from data", https://scholarsarchive.byu.edu/facpub/278,
    2006.

    Examples
    --------
    First a basic example to demonstrate the outputs:

    >>> from scipy import stats
    >>> data = [6, 9, 12, 7, 8, 8, 13]
    >>> mean, var, std = stats.bayes_mvs(data)
    >>> mean
    Mean(statistic=9.0, minmax=(7.103650222612533, 10.896349777387467))
    >>> var
    Variance(statistic=10.0, minmax=(3.176724206, 24.45910382))
    >>> std
    Std_dev(statistic=2.9724954732045084,
            minmax=(1.7823367265645143, 4.945614605014631))

    Now we generate some normally distributed random data, and get estimates of
    mean and standard deviation with 95% confidence intervals for those
    estimates:

    >>> n_samples = 100000
    >>> data = stats.norm.rvs(size=n_samples)
    >>> res_mean, res_var, res_std = stats.bayes_mvs(data, alpha=0.95)

    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.hist(data, bins=100, density=True, label='Histogram of data')
    >>> ax.vlines(res_mean.statistic, 0, 0.5, colors='r', label='Estimated mean')
    >>> ax.axvspan(res_mean.minmax[0],res_mean.minmax[1], facecolor='r',
    ...            alpha=0.2, label=r'Estimated mean (95% limits)')
    >>> ax.vlines(res_std.statistic, 0, 0.5, colors='g', label='Estimated scale')
    >>> ax.axvspan(res_std.minmax[0],res_std.minmax[1], facecolor='g', alpha=0.2,
    ...            label=r'Estimated scale (95% limits)')

    >>> ax.legend(fontsize=10)
    >>> ax.set_xlim([-4, 4])
    >>> ax.set_ylim([0, 0.5])
    >>> plt.show()

    """
    m, v, s = mvsdist(data)
    if alpha >= 1 or alpha <= 0:
        raise ValueError(f"0 < alpha < 1 is required, but {alpha=} was given.")

    m_res = Mean(m.mean(), m.interval(alpha))
    v_res = Variance(v.mean(), v.interval(alpha))
    s_res = Std_dev(s.mean(), s.interval(alpha))

    return m_res, v_res, s_res


def mvsdist(data):
    """
    'Frozen' distributions for mean, variance, and standard deviation of data.

    Parameters
    ----------
    data : array_like
        Input array. Converted to 1-D using ravel.
        Requires 2 or more data-points.

    Returns
    -------
    mdist : "frozen" distribution object
        Distribution object representing the mean of the data.
    vdist : "frozen" distribution object
        Distribution object representing the variance of the data.
    sdist : "frozen" distribution object
        Distribution object representing the standard deviation of the data.

    See Also
    --------
    bayes_mvs

    Notes
    -----
    The return values from ``bayes_mvs(data)`` is equivalent to
    ``tuple((x.mean(), x.interval(0.90)) for x in mvsdist(data))``.

    In other words, calling ``<dist>.mean()`` and ``<dist>.interval(0.90)``
    on the three distribution objects returned from this function will give
    the same results that are returned from `bayes_mvs`.

    References
    ----------
    T.E. Oliphant, "A Bayesian perspective on estimating mean, variance, and
    standard-deviation from data", https://scholarsarchive.byu.edu/facpub/278,
    2006.

    Examples
    --------
    >>> from scipy import stats
    >>> data = [6, 9, 12, 7, 8, 8, 13]
    >>> mean, var, std = stats.mvsdist(data)

    We now have frozen distribution objects "mean", "var" and "std" that we can
    examine:

    >>> mean.mean()
    9.0
    >>> mean.interval(0.95)
    (6.6120585482655692, 11.387941451734431)
    >>> mean.std()
    1.1952286093343936

    """
    x = ravel(data)
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 data-points.")
    xbar = x.mean()
    C = x.var()
    if n > 1000:  # gaussian approximations for large n
        mdist = distributions.norm(loc=xbar, scale=math.sqrt(C / n))
        sdist = distributions.norm(loc=math.sqrt(C), scale=math.sqrt(C / (2. * n)))
        vdist = distributions.norm(loc=C, scale=math.sqrt(2.0 / n) * C)
    else:
        nm1 = n - 1
        fac = n * C / 2.
        val = nm1 / 2.
        mdist = distributions.t(nm1, loc=xbar, scale=math.sqrt(C / nm1))
        sdist = distributions.gengamma(val, -2, scale=math.sqrt(fac))
        vdist = distributions.invgamma(val, scale=fac)
    return mdist, vdist, sdist


@_axis_nan_policy_factory(
    lambda x: x, result_to_tuple=lambda x: (x,), n_outputs=1, default_axis=None
)
def kstat(data, n=2, *, axis=None):
    r"""
    Return the `n` th k-statistic ( ``1<=n<=4`` so far).

    The `n` th k-statistic ``k_n`` is the unique symmetric unbiased estimator of the
    `n` th cumulant :math:`\kappa_n` [1]_ [2]_.

    Parameters
    ----------
    data : array_like
        Input array.
    n : int, {1, 2, 3, 4}, optional
        Default is equal to 2.
    axis : int or None, default: None
        If an int, the axis of the input along which to compute the statistic.
        The statistic of each axis-slice (e.g. row) of the input will appear
        in a corresponding element of the output. If ``None``, the input will
        be raveled before computing the statistic.

    Returns
    -------
    kstat : float
        The `n` th k-statistic.

    See Also
    --------
    kstatvar : Returns an unbiased estimator of the variance of the k-statistic
    moment : Returns the n-th central moment about the mean for a sample.

    Notes
    -----
    For a sample size :math:`n`, the first few k-statistics are given by

    .. math::

        k_1 &= \frac{S_1}{n}, \\
        k_2 &= \frac{nS_2 - S_1^2}{n(n-1)}, \\
        k_3 &= \frac{2S_1^3 - 3nS_1S_2 + n^2S_3}{n(n-1)(n-2)}, \\
        k_4 &= \frac{-6S_1^4 + 12nS_1^2S_2 - 3n(n-1)S_2^2 - 4n(n+1)S_1S_3
        + n^2(n+1)S_4}{n (n-1)(n-2)(n-3)},

    where

    .. math::

        S_r \equiv \sum_{i=1}^n X_i^r,

    and :math:`X_i` is the :math:`i` th data point.

    References
    ----------
    .. [1] http://mathworld.wolfram.com/k-Statistic.html

    .. [2] http://mathworld.wolfram.com/Cumulant.html

    Examples
    --------
    >>> from scipy import stats
    >>> from numpy.random import default_rng
    >>> rng = default_rng()

    As sample size increases, `n`-th moment and `n`-th k-statistic converge to the
    same number (although they aren't identical). In the case of the normal
    distribution, they converge to zero.

    >>> for i in range(2,8):
    ...     x = rng.normal(size=10**i)
    ...     m, k = stats.moment(x, 3), stats.kstat(x, 3)
    ...     print(f"{i=}: {m=:.3g}, {k=:.3g}, {(m-k)=:.3g}")
    i=2: m=-0.631, k=-0.651, (m-k)=0.0194  # random
    i=3: m=0.0282, k=0.0283, (m-k)=-8.49e-05
    i=4: m=-0.0454, k=-0.0454, (m-k)=1.36e-05
    i=6: m=7.53e-05, k=7.53e-05, (m-k)=-2.26e-09
    i=7: m=0.00166, k=0.00166, (m-k)=-4.99e-09
    i=8: m=-2.88e-06 k=-2.88e-06, (m-k)=8.63e-13
    """
    xp = array_namespace(data)
    data = xp.asarray(data)
    if n > 4 or n < 1:
        raise ValueError("k-statistics only supported for 1<=n<=4")
    n = int(n)
    if axis is None:
        data = xp.reshape(data, (-1,))
        axis = 0

    N = data.shape[axis]

    S = [None] + [xp.sum(data**k, axis=axis) for k in range(1, n + 1)]
    if n == 1:
        return S[1] * 1.0/N
    elif n == 2:
        return (N*S[2] - S[1]**2.0) / (N*(N - 1.0))
    elif n == 3:
        return (2*S[1]**3 - 3*N*S[1]*S[2] + N*N*S[3]) / (N*(N - 1.0)*(N - 2.0))
    elif n == 4:
        return ((-6*S[1]**4 + 12*N*S[1]**2 * S[2] - 3*N*(N-1.0)*S[2]**2 -
                 4*N*(N+1)*S[1]*S[3] + N*N*(N+1)*S[4]) /
                (N*(N-1.0)*(N-2.0)*(N-3.0)))
    else:
        raise ValueError("Should not be here.")


@_axis_nan_policy_factory(
    lambda x: x, result_to_tuple=lambda x: (x,), n_outputs=1, default_axis=None
)
def kstatvar(data, n=2, *, axis=None):
    r"""Return an unbiased estimator of the variance of the k-statistic.

    See `kstat` and [1]_ for more details about the k-statistic.

    Parameters
    ----------
    data : array_like
        Input array.
    n : int, {1, 2}, optional
        Default is equal to 2.
    axis : int or None, default: None
        If an int, the axis of the input along which to compute the statistic.
        The statistic of each axis-slice (e.g. row) of the input will appear
        in a corresponding element of the output. If ``None``, the input will
        be raveled before computing the statistic.

    Returns
    -------
    kstatvar : float
        The `n` th k-statistic variance.

    See Also
    --------
    kstat : Returns the n-th k-statistic.
    moment : Returns the n-th central moment about the mean for a sample.

    Notes
    -----
    Unbiased estimators of the variances of the first two k-statistics are given by

    .. math::

        \mathrm{var}(k_1) &= \frac{k_2}{n}, \\
        \mathrm{var}(k_2) &= \frac{2k_2^2n + (n-1)k_4}{n(n - 1)}.

    References
    ----------
    .. [1] http://mathworld.wolfram.com/k-Statistic.html

    """  # noqa: E501
    xp = array_namespace(data)
    data = xp.asarray(data)
    if axis is None:
        data = xp.reshape(data, (-1,))
        axis = 0
    N = data.shape[axis]

    if n == 1:
        return kstat(data, n=2, axis=axis, _no_deco=True) * 1.0/N
    elif n == 2:
        k2 = kstat(data, n=2, axis=axis, _no_deco=True)
        k4 = kstat(data, n=4, axis=axis, _no_deco=True)
        return (2*N*k2**2 + (N-1)*k4) / (N*(N+1))
    else:
        raise ValueError("Only n=1 or n=2 supported.")


def _calc_uniform_order_statistic_medians(n):
    """Approximations of uniform order statistic medians.

    Parameters
    ----------
    n : int
        Sample size.

    Returns
    -------
    v : 1d float array
        Approximations of the order statistic medians.

    References
    ----------
    .. [1] James J. Filliben, "The Probability Plot Correlation Coefficient
           Test for Normality", Technometrics, Vol. 17, pp. 111-117, 1975.

    Examples
    --------
    Order statistics of the uniform distribution on the unit interval
    are marginally distributed according to beta distributions.
    The expectations of these order statistic are evenly spaced across
    the interval, but the distributions are skewed in a way that
    pushes the medians slightly towards the endpoints of the unit interval:

    >>> import numpy as np
    >>> n = 4
    >>> k = np.arange(1, n+1)
    >>> from scipy.stats import beta
    >>> a = k
    >>> b = n-k+1
    >>> beta.mean(a, b)
    array([0.2, 0.4, 0.6, 0.8])
    >>> beta.median(a, b)
    array([0.15910358, 0.38572757, 0.61427243, 0.84089642])

    The Filliben approximation uses the exact medians of the smallest
    and greatest order statistics, and the remaining medians are approximated
    by points spread evenly across a sub-interval of the unit interval:

    >>> from scipy.stats._morestats import _calc_uniform_order_statistic_medians
    >>> _calc_uniform_order_statistic_medians(n)
    array([0.15910358, 0.38545246, 0.61454754, 0.84089642])

    This plot shows the skewed distributions of the order statistics
    of a sample of size four from a uniform distribution on the unit interval:

    >>> import matplotlib.pyplot as plt
    >>> x = np.linspace(0.0, 1.0, num=50, endpoint=True)
    >>> pdfs = [beta.pdf(x, a[i], b[i]) for i in range(n)]
    >>> plt.figure()
    >>> plt.plot(x, pdfs[0], x, pdfs[1], x, pdfs[2], x, pdfs[3])

    """
    v = np.empty(n, dtype=np.float64)
    v[-1] = 0.5**(1.0 / n)
    v[0] = 1 - v[-1]
    i = np.arange(2, n)
    v[1:-1] = (i - 0.3175) / (n + 0.365)
    return v


def _parse_dist_kw(dist, enforce_subclass=True):
    """Parse `dist` keyword.

    Parameters
    ----------
    dist : str or stats.distributions instance.
        Several functions take `dist` as a keyword, hence this utility
        function.
    enforce_subclass : bool, optional
        If True (default), `dist` needs to be a
        `_distn_infrastructure.rv_generic` instance.
        It can sometimes be useful to set this keyword to False, if a function
        wants to accept objects that just look somewhat like such an instance
        (for example, they have a ``ppf`` method).

    """
    if isinstance(dist, rv_generic):
        pass
    elif isinstance(dist, str):
        try:
            dist = getattr(distributions, dist)
        except AttributeError as e:
            raise ValueError(f"{dist} is not a valid distribution name") from e
    elif enforce_subclass:
        msg = ("`dist` should be a stats.distributions instance or a string "
               "with the name of such a distribution.")
        raise ValueError(msg)

    return dist


def _add_axis_labels_title(plot, xlabel, ylabel, title):
    """Helper function to add axes labels and a title to stats plots."""
    try:
        if hasattr(plot, 'set_title'):
            # Matplotlib Axes instance or something that looks like it
            plot.set_title(title)
            plot.set_xlabel(xlabel)
            plot.set_ylabel(ylabel)
        else:
            # matplotlib.pyplot module
            plot.title(title)
            plot.xlabel(xlabel)
            plot.ylabel(ylabel)
    except Exception:
        # Not an MPL object or something that looks (enough) like it.
        # Don't crash on adding labels or title
        pass


def probplot(x, sparams=(), dist='norm', fit=True, plot=None, rvalue=False):
    """
    Calculate quantiles for a probability plot, and optionally show the plot.

    Generates a probability plot of sample data against the quantiles of a
    specified theoretical distribution (the normal distribution by default).
    `probplot` optionally calculates a best-fit line for the data and plots the
    results using Matplotlib or a given plot function.

    Parameters
    ----------
    x : array_like
        Sample/response data from which `probplot` creates the plot.
    sparams : tuple, optional
        Distribution-specific shape parameters (shape parameters plus location
        and scale).
    dist : str or stats.distributions instance, optional
        Distribution or distribution function name. The default is 'norm' for a
        normal probability plot.  Objects that look enough like a
        stats.distributions instance (i.e. they have a ``ppf`` method) are also
        accepted.
    fit : bool, optional
        Fit a least-squares regression (best-fit) line to the sample data if
        True (default).
    plot : object, optional
        If given, plots the quantiles.
        If given and `fit` is True, also plots the least squares fit.
        `plot` is an object that has to have methods "plot" and "text".
        The `matplotlib.pyplot` module or a Matplotlib Axes object can be used,
        or a custom object with the same methods.
        Default is None, which means that no plot is created.
    rvalue : bool, optional
        If `plot` is provided and `fit` is True, setting `rvalue` to True
        includes the coefficient of determination on the plot.
        Default is False.

    Returns
    -------
    (osm, osr) : tuple of ndarrays
        Tuple of theoretical quantiles (osm, or order statistic medians) and
        ordered responses (osr).  `osr` is simply sorted input `x`.
        For details on how `osm` is calculated see the Notes section.
    (slope, intercept, r) : tuple of floats, optional
        Tuple  containing the result of the least-squares fit, if that is
        performed by `probplot`. `r` is the square root of the coefficient of
        determination.  If ``fit=False`` and ``plot=None``, this tuple is not
        returned.

    Notes
    -----
    Even if `plot` is given, the figure is not shown or saved by `probplot`;
    ``plt.show()`` or ``plt.savefig('figname.png')`` should be used after
    calling `probplot`.

    `probplot` generates a probability plot, which should not be confused with
    a Q-Q or a P-P plot.  Statsmodels has more extensive functionality of this
    type, see ``statsmodels.api.ProbPlot``.

    The formula used for the theoretical quantiles (horizontal axis of the
    probability plot) is Filliben's estimate::

        quantiles = dist.ppf(val), for

                0.5**(1/n),                  for i = n
          val = (i - 0.3175) / (n + 0.365),  for i = 2, ..., n-1
                1 - 0.5**(1/n),              for i = 1

    where ``i`` indicates the i-th ordered value and ``n`` is the total number
    of values.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt
    >>> nsample = 100
    >>> rng = np.random.default_rng()

    A t distribution with small degrees of freedom:

    >>> ax1 = plt.subplot(221)
    >>> x = stats.t.rvs(3, size=nsample, random_state=rng)
    >>> res = stats.probplot(x, plot=plt)

    A t distribution with larger degrees of freedom:

    >>> ax2 = plt.subplot(222)
    >>> x = stats.t.rvs(25, size=nsample, random_state=rng)
    >>> res = stats.probplot(x, plot=plt)

    A mixture of two normal distributions with broadcasting:

    >>> ax3 = plt.subplot(223)
    >>> x = stats.norm.rvs(loc=[0,5], scale=[1,1.5],
    ...                    size=(nsample//2,2), random_state=rng).ravel()
    >>> res = stats.probplot(x, plot=plt)

    A standard normal distribution:

    >>> ax4 = plt.subplot(224)
    >>> x = stats.norm.rvs(loc=0, scale=1, size=nsample, random_state=rng)
    >>> res = stats.probplot(x, plot=plt)

    Produce a new figure with a loggamma distribution, using the ``dist`` and
    ``sparams`` keywords:

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> x = stats.loggamma.rvs(c=2.5, size=500, random_state=rng)
    >>> res = stats.probplot(x, dist=stats.loggamma, sparams=(2.5,), plot=ax)
    >>> ax.set_title("Probplot for loggamma dist with shape parameter 2.5")

    Show the results with Matplotlib:

    >>> plt.show()

    """
    x = np.asarray(x)
    if x.size == 0:
        if fit:
            return (x, x), (np.nan, np.nan, 0.0)
        else:
            return x, x

    osm_uniform = _calc_uniform_order_statistic_medians(len(x))
    dist = _parse_dist_kw(dist, enforce_subclass=False)
    if sparams is None:
        sparams = ()
    if isscalar(sparams):
        sparams = (sparams,)
    if not isinstance(sparams, tuple):
        sparams = tuple(sparams)

    osm = dist.ppf(osm_uniform, *sparams)
    osr = sort(x)
    if fit:
        # perform a linear least squares fit.
        slope, intercept, r, prob, _ = _stats_py.linregress(osm, osr)

    if plot is not None:
        plot.plot(osm, osr, 'bo')
        if fit:
            plot.plot(osm, slope*osm + intercept, 'r-')
        _add_axis_labels_title(plot, xlabel='Theoretical quantiles',
                               ylabel='Ordered Values',
                               title='Probability Plot')

        # Add R^2 value to the plot as text
        if fit and rvalue:
            xmin = amin(osm)
            xmax = amax(osm)
            ymin = amin(x)
            ymax = amax(x)
            posx = xmin + 0.70 * (xmax - xmin)
            posy = ymin + 0.01 * (ymax - ymin)
            plot.text(posx, posy, f"$R^2={r ** 2:1.4f}$")

    if fit:
        return (osm, osr), (slope, intercept, r)
    else:
        return osm, osr


def ppcc_max(x, brack=(0.0, 1.0), dist='tukeylambda'):
    """Calculate the shape parameter that maximizes the PPCC.

    The probability plot correlation coefficient (PPCC) plot can be used
    to determine the optimal shape parameter for a one-parameter family
    of distributions. ``ppcc_max`` returns the shape parameter that would
    maximize the probability plot correlation coefficient for the given
    data to a one-parameter family of distributions.

    Parameters
    ----------
    x : array_like
        Input array.
    brack : tuple, optional
        Triple (a,b,c) where (a<b<c). If bracket consists of two numbers (a, c)
        then they are assumed to be a starting interval for a downhill bracket
        search (see `scipy.optimize.brent`).
    dist : str or stats.distributions instance, optional
        Distribution or distribution function name.  Objects that look enough
        like a stats.distributions instance (i.e. they have a ``ppf`` method)
        are also accepted.  The default is ``'tukeylambda'``.

    Returns
    -------
    shape_value : float
        The shape parameter at which the probability plot correlation
        coefficient reaches its max value.

    See Also
    --------
    ppcc_plot, probplot, boxcox

    Notes
    -----
    The brack keyword serves as a starting point which is useful in corner
    cases. One can use a plot to obtain a rough visual estimate of the location
    for the maximum to start the search near it.

    References
    ----------
    .. [1] J.J. Filliben, "The Probability Plot Correlation Coefficient Test
           for Normality", Technometrics, Vol. 17, pp. 111-117, 1975.
    .. [2] Engineering Statistics Handbook, NIST/SEMATEC,
           https://www.itl.nist.gov/div898/handbook/eda/section3/ppccplot.htm

    Examples
    --------
    First we generate some random data from a Weibull distribution
    with shape parameter 2.5:

    >>> import numpy as np
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> c = 2.5
    >>> x = stats.weibull_min.rvs(c, scale=4, size=2000, random_state=rng)

    Generate the PPCC plot for this data with the Weibull distribution.

    >>> fig, ax = plt.subplots(figsize=(8, 6))
    >>> res = stats.ppcc_plot(x, c/2, 2*c, dist='weibull_min', plot=ax)

    We calculate the value where the shape should reach its maximum and a
    red line is drawn there. The line should coincide with the highest
    point in the PPCC graph.

    >>> cmax = stats.ppcc_max(x, brack=(c/2, 2*c), dist='weibull_min')
    >>> ax.axvline(cmax, color='r')
    >>> plt.show()

    """
    dist = _parse_dist_kw(dist)
    osm_uniform = _calc_uniform_order_statistic_medians(len(x))
    osr = sort(x)

    # this function computes the x-axis values of the probability plot
    #  and computes a linear regression (including the correlation)
    #  and returns 1-r so that a minimization function maximizes the
    #  correlation
    def tempfunc(shape, mi, yvals, func):
        xvals = func(mi, shape)
        r, prob = _stats_py.pearsonr(xvals, yvals)
        return 1 - r

    return optimize.brent(tempfunc, brack=brack,
                          args=(osm_uniform, osr, dist.ppf))


def ppcc_plot(x, a, b, dist='tukeylambda', plot=None, N=80):
    """Calculate and optionally plot probability plot correlation coefficient.

    The probability plot correlation coefficient (PPCC) plot can be used to
    determine the optimal shape parameter for a one-parameter family of
    distributions.  It cannot be used for distributions without shape
    parameters
    (like the normal distribution) or with multiple shape parameters.

    By default a Tukey-Lambda distribution (`stats.tukeylambda`) is used. A
    Tukey-Lambda PPCC plot interpolates from long-tailed to short-tailed
    distributions via an approximately normal one, and is therefore
    particularly useful in practice.

    Parameters
    ----------
    x : array_like
        Input array.
    a, b : scalar
        Lower and upper bounds of the shape parameter to use.
    dist : str or stats.distributions instance, optional
        Distribution or distribution function name.  Objects that look enough
        like a stats.distributions instance (i.e. they have a ``ppf`` method)
        are also accepted.  The default is ``'tukeylambda'``.
    plot : object, optional
        If given, plots PPCC against the shape parameter.
        `plot` is an object that has to have methods "plot" and "text".
        The `matplotlib.pyplot` module or a Matplotlib Axes object can be used,
        or a custom object with the same methods.
        Default is None, which means that no plot is created.
    N : int, optional
        Number of points on the horizontal axis (equally distributed from
        `a` to `b`).

    Returns
    -------
    svals : ndarray
        The shape values for which `ppcc` was calculated.
    ppcc : ndarray
        The calculated probability plot correlation coefficient values.

    See Also
    --------
    ppcc_max, probplot, boxcox_normplot, tukeylambda

    References
    ----------
    J.J. Filliben, "The Probability Plot Correlation Coefficient Test for
    Normality", Technometrics, Vol. 17, pp. 111-117, 1975.

    Examples
    --------
    First we generate some random data from a Weibull distribution
    with shape parameter 2.5, and plot the histogram of the data:

    >>> import numpy as np
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> c = 2.5
    >>> x = stats.weibull_min.rvs(c, scale=4, size=2000, random_state=rng)

    Take a look at the histogram of the data.

    >>> fig1, ax = plt.subplots(figsize=(9, 4))
    >>> ax.hist(x, bins=50)
    >>> ax.set_title('Histogram of x')
    >>> plt.show()

    Now we explore this data with a PPCC plot as well as the related
    probability plot and Box-Cox normplot.  A red line is drawn where we
    expect the PPCC value to be maximal (at the shape parameter ``c``
    used above):

    >>> fig2 = plt.figure(figsize=(12, 4))
    >>> ax1 = fig2.add_subplot(1, 3, 1)
    >>> ax2 = fig2.add_subplot(1, 3, 2)
    >>> ax3 = fig2.add_subplot(1, 3, 3)
    >>> res = stats.probplot(x, plot=ax1)
    >>> res = stats.boxcox_normplot(x, -4, 4, plot=ax2)
    >>> res = stats.ppcc_plot(x, c/2, 2*c, dist='weibull_min', plot=ax3)
    >>> ax3.axvline(c, color='r')
    >>> plt.show()

    """
    if b <= a:
        raise ValueError("`b` has to be larger than `a`.")

    svals = np.linspace(a, b, num=N)
    ppcc = np.empty_like(svals)
    for k, sval in enumerate(svals):
        _, r2 = probplot(x, sval, dist=dist, fit=True)
        ppcc[k] = r2[-1]

    if plot is not None:
        plot.plot(svals, ppcc, 'x')
        _add_axis_labels_title(plot, xlabel='Shape Values',
                               ylabel='Prob Plot Corr. Coef.',
                               title=f'({dist}) PPCC Plot')

    return svals, ppcc


def _log_mean(logx):
    # compute log of mean of x from log(x)
    res = special.logsumexp(logx, axis=0) - math.log(logx.shape[0])
    return res


def _log_var(logx, xp):
    # compute log of variance of x from log(x)
    logmean = _log_mean(logx)
    # get complex dtype with component dtypes same as `logx` dtype;
    # see data-apis/array-api#841
    dtype = xp.result_type(logx.dtype, xp.complex64)
    pij = xp.full(logx.shape, pi * 1j, dtype=dtype)
    logxmu = special.logsumexp(xp.stack((logx, logmean + pij)), axis=0)
    res = (xp.real(xp.asarray(special.logsumexp(2 * logxmu, axis=0)))
           - math.log(logx.shape[0]))
    return res


def boxcox_llf(lmb, data):
    r"""The boxcox log-likelihood function.

    Parameters
    ----------
    lmb : scalar
        Parameter for Box-Cox transformation.  See `boxcox` for details.
    data : array_like
        Data to calculate Box-Cox log-likelihood for.  If `data` is
        multi-dimensional, the log-likelihood is calculated along the first
        axis.

    Returns
    -------
    llf : float or ndarray
        Box-Cox log-likelihood of `data` given `lmb`.  A float for 1-D `data`,
        an array otherwise.

    See Also
    --------
    boxcox, probplot, boxcox_normplot, boxcox_normmax

    Notes
    -----
    The Box-Cox log-likelihood function is defined here as

    .. math::

        llf = (\lambda - 1) \sum_i(\log(x_i)) -
              N/2 \log(\sum_i (y_i - \bar{y})^2 / N),

    where ``y`` is the Box-Cox transformed input data ``x``.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    Generate some random variates and calculate Box-Cox log-likelihood values
    for them for a range of ``lmbda`` values:

    >>> rng = np.random.default_rng()
    >>> x = stats.loggamma.rvs(5, loc=10, size=1000, random_state=rng)
    >>> lmbdas = np.linspace(-2, 10)
    >>> llf = np.zeros(lmbdas.shape, dtype=float)
    >>> for ii, lmbda in enumerate(lmbdas):
    ...     llf[ii] = stats.boxcox_llf(lmbda, x)

    Also find the optimal lmbda value with `boxcox`:

    >>> x_most_normal, lmbda_optimal = stats.boxcox(x)

    Plot the log-likelihood as function of lmbda.  Add the optimal lmbda as a
    horizontal line to check that that's really the optimum:

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot(lmbdas, llf, 'b.-')
    >>> ax.axhline(stats.boxcox_llf(lmbda_optimal, x), color='r')
    >>> ax.set_xlabel('lmbda parameter')
    >>> ax.set_ylabel('Box-Cox log-likelihood')

    Now add some probability plots to show that where the log-likelihood is
    maximized the data transformed with `boxcox` looks closest to normal:

    >>> locs = [3, 10, 4]  # 'lower left', 'center', 'lower right'
    >>> for lmbda, loc in zip([-1, lmbda_optimal, 9], locs):
    ...     xt = stats.boxcox(x, lmbda=lmbda)
    ...     (osm, osr), (slope, intercept, r_sq) = stats.probplot(xt)
    ...     ax_inset = inset_axes(ax, width="20%", height="20%", loc=loc)
    ...     ax_inset.plot(osm, osr, 'c.', osm, slope*osm + intercept, 'k-')
    ...     ax_inset.set_xticklabels([])
    ...     ax_inset.set_yticklabels([])
    ...     ax_inset.set_title(r'$\lambda=%1.2f$' % lmbda)

    >>> plt.show()

    """
    xp = array_namespace(data)
    data = xp.asarray(data)
    N = data.shape[0]
    if N == 0:
        return xp.nan

    dt = data.dtype
    if xp.isdtype(dt, 'integral'):
        data = xp.asarray(data, dtype=xp.float64)
        dt = xp.float64

    logdata = xp.log(data)

    # Compute the variance of the transformed data.
    if lmb == 0:
        logvar = xp.log(xp.var(logdata, axis=0))
    else:
        # Transform without the constant offset 1/lmb.  The offset does
        # not affect the variance, and the subtraction of the offset can
        # lead to loss of precision.
        # Division by lmb can be factored out to enhance numerical stability.
        logx = lmb * logdata
        logvar = _log_var(logx, xp) - 2 * math.log(abs(lmb))

    res = (lmb - 1) * xp.sum(logdata, axis=0) - N/2 * logvar
    res = xp.astype(res, dt)
    res = res[()] if res.ndim == 0 else res
    return res


def _boxcox_conf_interval(x, lmax, alpha):
    # Need to find the lambda for which
    #  f(x,lmbda) >= f(x,lmax) - 0.5*chi^2_alpha;1
    fac = 0.5 * distributions.chi2.ppf(1 - alpha, 1)
    target = boxcox_llf(lmax, x) - fac

    def rootfunc(lmbda, data, target):
        return boxcox_llf(lmbda, data) - target

    # Find positive endpoint of interval in which answer is to be found
    newlm = lmax + 0.5
    N = 0
    while (rootfunc(newlm, x, target) > 0.0) and (N < 500):
        newlm += 0.1
        N += 1

    if N == 500:
        raise RuntimeError("Could not find endpoint.")

    lmplus = optimize.brentq(rootfunc, lmax, newlm, args=(x, target))

    # Now find negative interval in the same way
    newlm = lmax - 0.5
    N = 0
    while (rootfunc(newlm, x, target) > 0.0) and (N < 500):
        newlm -= 0.1
        N += 1

    if N == 500:
        raise RuntimeError("Could not find endpoint.")

    lmminus = optimize.brentq(rootfunc, newlm, lmax, args=(x, target))
    return lmminus, lmplus


def boxcox(x, lmbda=None, alpha=None, optimizer=None):
    r"""Return a dataset transformed by a Box-Cox power transformation.

    Parameters
    ----------
    x : ndarray
        Input array to be transformed.

        If `lmbda` is not None, this is an alias of
        `scipy.special.boxcox`.
        Returns nan if ``x < 0``; returns -inf if ``x == 0 and lmbda < 0``.

        If `lmbda` is None, array must be positive, 1-dimensional, and
        non-constant.

    lmbda : scalar, optional
        If `lmbda` is None (default), find the value of `lmbda` that maximizes
        the log-likelihood function and return it as the second output
        argument.

        If `lmbda` is not None, do the transformation for that value.

    alpha : float, optional
        If `lmbda` is None and `alpha` is not None (default), return the
        ``100 * (1-alpha)%`` confidence  interval for `lmbda` as the third
        output argument. Must be between 0.0 and 1.0.

        If `lmbda` is not None, `alpha` is ignored.
    optimizer : callable, optional
        If `lmbda` is None, `optimizer` is the scalar optimizer used to find
        the value of `lmbda` that minimizes the negative log-likelihood
        function. `optimizer` is a callable that accepts one argument:

        fun : callable
            The objective function, which evaluates the negative
            log-likelihood function at a provided value of `lmbda`

        and returns an object, such as an instance of
        `scipy.optimize.OptimizeResult`, which holds the optimal value of
        `lmbda` in an attribute `x`.

        See the example in `boxcox_normmax` or the documentation of
        `scipy.optimize.minimize_scalar` for more information.

        If `lmbda` is not None, `optimizer` is ignored.

    Returns
    -------
    boxcox : ndarray
        Box-Cox power transformed array.
    maxlog : float, optional
        If the `lmbda` parameter is None, the second returned argument is
        the `lmbda` that maximizes the log-likelihood function.
    (min_ci, max_ci) : tuple of float, optional
        If `lmbda` parameter is None and `alpha` is not None, this returned
        tuple of floats represents the minimum and maximum confidence limits
        given `alpha`.

    See Also
    --------
    probplot, boxcox_normplot, boxcox_normmax, boxcox_llf

    Notes
    -----
    The Box-Cox transform is given by::

        y = (x**lmbda - 1) / lmbda,  for lmbda != 0
            log(x),                  for lmbda = 0

    `boxcox` requires the input data to be positive.  Sometimes a Box-Cox
    transformation provides a shift parameter to achieve this; `boxcox` does
    not.  Such a shift parameter is equivalent to adding a positive constant to
    `x` before calling `boxcox`.

    The confidence limits returned when `alpha` is provided give the interval
    where:

    .. math::

        llf(\hat{\lambda}) - llf(\lambda) < \frac{1}{2}\chi^2(1 - \alpha, 1),

    with ``llf`` the log-likelihood function and :math:`\chi^2` the chi-squared
    function.

    References
    ----------
    G.E.P. Box and D.R. Cox, "An Analysis of Transformations", Journal of the
    Royal Statistical Society B, 26, 211-252 (1964).

    Examples
    --------
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt

    We generate some random variates from a non-normal distribution and make a
    probability plot for it, to show it is non-normal in the tails:

    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(211)
    >>> x = stats.loggamma.rvs(5, size=500) + 5
    >>> prob = stats.probplot(x, dist=stats.norm, plot=ax1)
    >>> ax1.set_xlabel('')
    >>> ax1.set_title('Probplot against normal distribution')

    We now use `boxcox` to transform the data so it's closest to normal:

    >>> ax2 = fig.add_subplot(212)
    >>> xt, _ = stats.boxcox(x)
    >>> prob = stats.probplot(xt, dist=stats.norm, plot=ax2)
    >>> ax2.set_title('Probplot after Box-Cox transformation')

    >>> plt.show()

    """
    x = np.asarray(x)

    if lmbda is not None:  # single transformation
        return special.boxcox(x, lmbda)

    if x.ndim != 1:
        raise ValueError("Data must be 1-dimensional.")

    if x.size == 0:
        return x

    if np.all(x == x[0]):
        raise ValueError("Data must not be constant.")

    if np.any(x <= 0):
        raise ValueError("Data must be positive.")

    # If lmbda=None, find the lmbda that maximizes the log-likelihood function.
    lmax = boxcox_normmax(x, method='mle', optimizer=optimizer)
    y = boxcox(x, lmax)

    if alpha is None:
        return y, lmax
    else:
        # Find confidence interval
        interval = _boxcox_conf_interval(x, lmax, alpha)
        return y, lmax, interval


def _boxcox_inv_lmbda(x, y):
    # compute lmbda given x and y for Box-Cox transformation
    num = special.lambertw(-(x ** (-1 / y)) * np.log(x) / y, k=-1)
    return np.real(-num / np.log(x) - 1 / y)


class _BigFloat:
    def __repr__(self):
        return "BIG_FLOAT"


_BigFloat_singleton = _BigFloat()


def boxcox_normmax(
    x, brack=None, method='pearsonr', optimizer=None, *, ymax=_BigFloat_singleton
):
    """Compute optimal Box-Cox transform parameter for input data.

    Parameters
    ----------
    x : array_like
        Input array. All entries must be positive, finite, real numbers.
    brack : 2-tuple, optional, default (-2.0, 2.0)
         The starting interval for a downhill bracket search for the default
         `optimize.brent` solver. Note that this is in most cases not
         critical; the final result is allowed to be outside this bracket.
         If `optimizer` is passed, `brack` must be None.
    method : str, optional
        The method to determine the optimal transform parameter (`boxcox`
        ``lmbda`` parameter). Options are:

        'pearsonr'  (default)
            Maximizes the Pearson correlation coefficient between
            ``y = boxcox(x)`` and the expected values for ``y`` if `x` would be
            normally-distributed.

        'mle'
            Maximizes the log-likelihood `boxcox_llf`.  This is the method used
            in `boxcox`.

        'all'
            Use all optimization methods available, and return all results.
            Useful to compare different methods.
    optimizer : callable, optional
        `optimizer` is a callable that accepts one argument:

        fun : callable
            The objective function to be minimized. `fun` accepts one argument,
            the Box-Cox transform parameter `lmbda`, and returns the value of
            the function (e.g., the negative log-likelihood) at the provided
            argument. The job of `optimizer` is to find the value of `lmbda`
            that *minimizes* `fun`.

        and returns an object, such as an instance of
        `scipy.optimize.OptimizeResult`, which holds the optimal value of
        `lmbda` in an attribute `x`.

        See the example below or the documentation of
        `scipy.optimize.minimize_scalar` for more information.
    ymax : float, optional
        The unconstrained optimal transform parameter may cause Box-Cox
        transformed data to have extreme magnitude or even overflow.
        This parameter constrains MLE optimization such that the magnitude
        of the transformed `x` does not exceed `ymax`. The default is
        the maximum value of the input dtype. If set to infinity,
        `boxcox_normmax` returns the unconstrained optimal lambda.
        Ignored when ``method='pearsonr'``.

    Returns
    -------
    maxlog : float or ndarray
        The optimal transform parameter found.  An array instead of a scalar
        for ``method='all'``.

    See Also
    --------
    boxcox, boxcox_llf, boxcox_normplot, scipy.optimize.minimize_scalar

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt

    We can generate some data and determine the optimal ``lmbda`` in various
    ways:

    >>> rng = np.random.default_rng()
    >>> x = stats.loggamma.rvs(5, size=30, random_state=rng) + 5
    >>> y, lmax_mle = stats.boxcox(x)
    >>> lmax_pearsonr = stats.boxcox_normmax(x)

    >>> lmax_mle
    2.217563431465757
    >>> lmax_pearsonr
    2.238318660200961
    >>> stats.boxcox_normmax(x, method='all')
    array([2.23831866, 2.21756343])

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> prob = stats.boxcox_normplot(x, -10, 10, plot=ax)
    >>> ax.axvline(lmax_mle, color='r')
    >>> ax.axvline(lmax_pearsonr, color='g', ls='--')

    >>> plt.show()

    Alternatively, we can define our own `optimizer` function. Suppose we
    are only interested in values of `lmbda` on the interval [6, 7], we
    want to use `scipy.optimize.minimize_scalar` with ``method='bounded'``,
    and we want to use tighter tolerances when optimizing the log-likelihood
    function. To do this, we define a function that accepts positional argument
    `fun` and uses `scipy.optimize.minimize_scalar` to minimize `fun` subject
    to the provided bounds and tolerances:

    >>> from scipy import optimize
    >>> options = {'xatol': 1e-12}  # absolute tolerance on `x`
    >>> def optimizer(fun):
    ...     return optimize.minimize_scalar(fun, bounds=(6, 7),
    ...                                     method="bounded", options=options)
    >>> stats.boxcox_normmax(x, optimizer=optimizer)
    6.000000000
    """
    x = np.asarray(x)

    if not np.all(np.isfinite(x) & (x >= 0)):
        message = ("The `x` argument of `boxcox_normmax` must contain "
                   "only positive, finite, real numbers.")
        raise ValueError(message)

    end_msg = "exceed specified `ymax`."
    if ymax is _BigFloat_singleton:
        dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64
        # 10000 is a safety factor because `special.boxcox` overflows prematurely.
        ymax = np.finfo(dtype).max / 10000
        end_msg = f"overflow in {dtype}."
    elif ymax <= 0:
        raise ValueError("`ymax` must be strictly positive")

    # If optimizer is not given, define default 'brent' optimizer.
    if optimizer is None:

        # Set default value for `brack`.
        if brack is None:
            brack = (-2.0, 2.0)

        def _optimizer(func, args):
            return optimize.brent(func, args=args, brack=brack)

    # Otherwise check optimizer.
    else:
        if not callable(optimizer):
            raise ValueError("`optimizer` must be a callable")

        if brack is not None:
            raise ValueError("`brack` must be None if `optimizer` is given")

        # `optimizer` is expected to return a `OptimizeResult` object, we here
        # get the solution to the optimization problem.
        def _optimizer(func, args):
            def func_wrapped(x):
                return func(x, *args)
            return getattr(optimizer(func_wrapped), 'x', None)

    def _pearsonr(x):
        osm_uniform = _calc_uniform_order_statistic_medians(len(x))
        xvals = distributions.norm.ppf(osm_uniform)

        def _eval_pearsonr(lmbda, xvals, samps):
            # This function computes the x-axis values of the probability plot
            # and computes a linear regression (including the correlation) and
            # returns ``1 - r`` so that a minimization function maximizes the
            # correlation.
            y = boxcox(samps, lmbda)
            yvals = np.sort(y)
            r, prob = _stats_py.pearsonr(xvals, yvals)
            return 1 - r

        return _optimizer(_eval_pearsonr, args=(xvals, x))

    def _mle(x):
        def _eval_mle(lmb, data):
            # function to minimize
            return -boxcox_llf(lmb, data)

        return _optimizer(_eval_mle, args=(x,))

    def _all(x):
        maxlog = np.empty(2, dtype=float)
        maxlog[0] = _pearsonr(x)
        maxlog[1] = _mle(x)
        return maxlog

    methods = {'pearsonr': _pearsonr,
               'mle': _mle,
               'all': _all}
    if method not in methods.keys():
        raise ValueError(f"Method {method} not recognized.")

    optimfunc = methods[method]

    res = optimfunc(x)

    if res is None:
        message = ("The `optimizer` argument of `boxcox_normmax` must return "
                   "an object containing the optimal `lmbda` in attribute `x`.")
        raise ValueError(message)
    elif not np.isinf(ymax):  # adjust the final lambda
        # x > 1, boxcox(x) > 0; x < 1, boxcox(x) < 0
        xmax, xmin = np.max(x), np.min(x)
        if xmin >= 1:
            x_treme = xmax
        elif xmax <= 1:
            x_treme = xmin
        else:  # xmin < 1 < xmax
            indicator = special.boxcox(xmax, res) > abs(special.boxcox(xmin, res))
            if isinstance(res, np.ndarray):
                indicator = indicator[1]  # select corresponds with 'mle'
            x_treme = xmax if indicator else xmin

        mask = abs(special.boxcox(x_treme, res)) > ymax
        if np.any(mask):
            message = (
                f"The optimal lambda is {res}, but the returned lambda is the "
                f"constrained optimum to ensure that the maximum or the minimum "
                f"of the transformed data does not " + end_msg
            )
            warnings.warn(message, stacklevel=2)

            # Return the constrained lambda to ensure the transformation
            # does not cause overflow or exceed specified `ymax`
            constrained_res = _boxcox_inv_lmbda(x_treme, ymax * np.sign(x_treme - 1))

            if isinstance(res, np.ndarray):
                res[mask] = constrained_res
            else:
                res = constrained_res
    return res


def _normplot(method, x, la, lb, plot=None, N=80):
    """Compute parameters for a Box-Cox or Yeo-Johnson normality plot,
    optionally show it.

    See `boxcox_normplot` or `yeojohnson_normplot` for details.
    """

    if method == 'boxcox':
        title = 'Box-Cox Normality Plot'
        transform_func = boxcox
    else:
        title = 'Yeo-Johnson Normality Plot'
        transform_func = yeojohnson

    x = np.asarray(x)
    if x.size == 0:
        return x

    if lb <= la:
        raise ValueError("`lb` has to be larger than `la`.")

    if method == 'boxcox' and np.any(x <= 0):
        raise ValueError("Data must be positive.")

    lmbdas = np.linspace(la, lb, num=N)
    ppcc = lmbdas * 0.0
    for i, val in enumerate(lmbdas):
        # Determine for each lmbda the square root of correlation coefficient
        # of transformed x
        z = transform_func(x, lmbda=val)
        _, (_, _, r) = probplot(z, dist='norm', fit=True)
        ppcc[i] = r

    if plot is not None:
        plot.plot(lmbdas, ppcc, 'x')
        _add_axis_labels_title(plot, xlabel='$\\lambda$',
                               ylabel='Prob Plot Corr. Coef.',
                               title=title)

    return lmbdas, ppcc


def boxcox_normplot(x, la, lb, plot=None, N=80):
    """Compute parameters for a Box-Cox normality plot, optionally show it.

    A Box-Cox normality plot shows graphically what the best transformation
    parameter is to use in `boxcox` to obtain a distribution that is close
    to normal.

    Parameters
    ----------
    x : array_like
        Input array.
    la, lb : scalar
        The lower and upper bounds for the ``lmbda`` values to pass to `boxcox`
        for Box-Cox transformations.  These are also the limits of the
        horizontal axis of the plot if that is generated.
    plot : object, optional
        If given, plots the quantiles and least squares fit.
        `plot` is an object that has to have methods "plot" and "text".
        The `matplotlib.pyplot` module or a Matplotlib Axes object can be used,
        or a custom object with the same methods.
        Default is None, which means that no plot is created.
    N : int, optional
        Number of points on the horizontal axis (equally distributed from
        `la` to `lb`).

    Returns
    -------
    lmbdas : ndarray
        The ``lmbda`` values for which a Box-Cox transform was done.
    ppcc : ndarray
        Probability Plot Correlation Coefficient, as obtained from `probplot`
        when fitting the Box-Cox transformed input `x` against a normal
        distribution.

    See Also
    --------
    probplot, boxcox, boxcox_normmax, boxcox_llf, ppcc_max

    Notes
    -----
    Even if `plot` is given, the figure is not shown or saved by
    `boxcox_normplot`; ``plt.show()`` or ``plt.savefig('figname.png')``
    should be used after calling `probplot`.

    Examples
    --------
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt

    Generate some non-normally distributed data, and create a Box-Cox plot:

    >>> x = stats.loggamma.rvs(5, size=500) + 5
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> prob = stats.boxcox_normplot(x, -20, 20, plot=ax)

    Determine and plot the optimal ``lmbda`` to transform ``x`` and plot it in
    the same plot:

    >>> _, maxlog = stats.boxcox(x)
    >>> ax.axvline(maxlog, color='r')

    >>> plt.show()

    """
    return _normplot('boxcox', x, la, lb, plot, N)


def yeojohnson(x, lmbda=None):
    r"""Return a dataset transformed by a Yeo-Johnson power transformation.

    Parameters
    ----------
    x : ndarray
        Input array.  Should be 1-dimensional.
    lmbda : float, optional
        If ``lmbda`` is ``None``, find the lambda that maximizes the
        log-likelihood function and return it as the second output argument.
        Otherwise the transformation is done for the given value.

    Returns
    -------
    yeojohnson: ndarray
        Yeo-Johnson power transformed array.
    maxlog : float, optional
        If the `lmbda` parameter is None, the second returned argument is
        the lambda that maximizes the log-likelihood function.

    See Also
    --------
    probplot, yeojohnson_normplot, yeojohnson_normmax, yeojohnson_llf, boxcox

    Notes
    -----
    The Yeo-Johnson transform is given by::

        y = ((x + 1)**lmbda - 1) / lmbda,                for x >= 0, lmbda != 0
            log(x + 1),                                  for x >= 0, lmbda = 0
            -((-x + 1)**(2 - lmbda) - 1) / (2 - lmbda),  for x < 0, lmbda != 2
            -log(-x + 1),                                for x < 0, lmbda = 2

    Unlike `boxcox`, `yeojohnson` does not require the input data to be
    positive.

    .. versionadded:: 1.2.0


    References
    ----------
    I. Yeo and R.A. Johnson, "A New Family of Power Transformations to
    Improve Normality or Symmetry", Biometrika 87.4 (2000):


    Examples
    --------
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt

    We generate some random variates from a non-normal distribution and make a
    probability plot for it, to show it is non-normal in the tails:

    >>> fig = plt.figure()
    >>> ax1 = fig.add_subplot(211)
    >>> x = stats.loggamma.rvs(5, size=500) + 5
    >>> prob = stats.probplot(x, dist=stats.norm, plot=ax1)
    >>> ax1.set_xlabel('')
    >>> ax1.set_title('Probplot against normal distribution')

    We now use `yeojohnson` to transform the data so it's closest to normal:

    >>> ax2 = fig.add_subplot(212)
    >>> xt, lmbda = stats.yeojohnson(x)
    >>> prob = stats.probplot(xt, dist=stats.norm, plot=ax2)
    >>> ax2.set_title('Probplot after Yeo-Johnson transformation')

    >>> plt.show()

    """
    x = np.asarray(x)
    if x.size == 0:
        return x

    if np.issubdtype(x.dtype, np.complexfloating):
        raise ValueError('Yeo-Johnson transformation is not defined for '
                         'complex numbers.')

    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64, copy=False)

    if lmbda is not None:
        return _yeojohnson_transform(x, lmbda)

    # if lmbda=None, find the lmbda that maximizes the log-likelihood function.
    lmax = yeojohnson_normmax(x)
    y = _yeojohnson_transform(x, lmax)

    return y, lmax


def _yeojohnson_transform(x, lmbda):
    """Returns `x` transformed by the Yeo-Johnson power transform with given
    parameter `lmbda`.
    """
    dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64
    out = np.zeros_like(x, dtype=dtype)
    pos = x >= 0  # binary mask

    # when x >= 0
    if abs(lmbda) < np.spacing(1.):
        out[pos] = np.log1p(x[pos])
    else:  # lmbda != 0
        # more stable version of: ((x + 1) ** lmbda - 1) / lmbda
        out[pos] = np.expm1(lmbda * np.log1p(x[pos])) / lmbda

    # when x < 0
    if abs(lmbda - 2) > np.spacing(1.):
        out[~pos] = -np.expm1((2 - lmbda) * np.log1p(-x[~pos])) / (2 - lmbda)
    else:  # lmbda == 2
        out[~pos] = -np.log1p(-x[~pos])

    return out


def yeojohnson_llf(lmb, data):
    r"""The yeojohnson log-likelihood function.

    Parameters
    ----------
    lmb : scalar
        Parameter for Yeo-Johnson transformation. See `yeojohnson` for
        details.
    data : array_like
        Data to calculate Yeo-Johnson log-likelihood for. If `data` is
        multi-dimensional, the log-likelihood is calculated along the first
        axis.

    Returns
    -------
    llf : float
        Yeo-Johnson log-likelihood of `data` given `lmb`.

    See Also
    --------
    yeojohnson, probplot, yeojohnson_normplot, yeojohnson_normmax

    Notes
    -----
    The Yeo-Johnson log-likelihood function is defined here as

    .. math::

        llf = -N/2 \log(\hat{\sigma}^2) + (\lambda - 1)
              \sum_i \text{ sign }(x_i)\log(|x_i| + 1)

    where :math:`\hat{\sigma}^2` is estimated variance of the Yeo-Johnson
    transformed input data ``x``.

    .. versionadded:: 1.2.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    Generate some random variates and calculate Yeo-Johnson log-likelihood
    values for them for a range of ``lmbda`` values:

    >>> x = stats.loggamma.rvs(5, loc=10, size=1000)
    >>> lmbdas = np.linspace(-2, 10)
    >>> llf = np.zeros(lmbdas.shape, dtype=float)
    >>> for ii, lmbda in enumerate(lmbdas):
    ...     llf[ii] = stats.yeojohnson_llf(lmbda, x)

    Also find the optimal lmbda value with `yeojohnson`:

    >>> x_most_normal, lmbda_optimal = stats.yeojohnson(x)

    Plot the log-likelihood as function of lmbda.  Add the optimal lmbda as a
    horizontal line to check that that's really the optimum:

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot(lmbdas, llf, 'b.-')
    >>> ax.axhline(stats.yeojohnson_llf(lmbda_optimal, x), color='r')
    >>> ax.set_xlabel('lmbda parameter')
    >>> ax.set_ylabel('Yeo-Johnson log-likelihood')

    Now add some probability plots to show that where the log-likelihood is
    maximized the data transformed with `yeojohnson` looks closest to normal:

    >>> locs = [3, 10, 4]  # 'lower left', 'center', 'lower right'
    >>> for lmbda, loc in zip([-1, lmbda_optimal, 9], locs):
    ...     xt = stats.yeojohnson(x, lmbda=lmbda)
    ...     (osm, osr), (slope, intercept, r_sq) = stats.probplot(xt)
    ...     ax_inset = inset_axes(ax, width="20%", height="20%", loc=loc)
    ...     ax_inset.plot(osm, osr, 'c.', osm, slope*osm + intercept, 'k-')
    ...     ax_inset.set_xticklabels([])
    ...     ax_inset.set_yticklabels([])
    ...     ax_inset.set_title(r'$\lambda=%1.2f$' % lmbda)

    >>> plt.show()

    """
    data = np.asarray(data)
    n_samples = data.shape[0]

    if n_samples == 0:
        return np.nan

    trans = _yeojohnson_transform(data, lmb)
    trans_var = trans.var(axis=0)
    loglike = np.empty_like(trans_var)

    # Avoid RuntimeWarning raised by np.log when the variance is too low
    tiny_variance = trans_var < np.finfo(trans_var.dtype).tiny
    loglike[tiny_variance] = np.inf

    loglike[~tiny_variance] = (
        -n_samples / 2 * np.log(trans_var[~tiny_variance]))
    loglike[~tiny_variance] += (
        (lmb - 1) * (np.sign(data) * np.log1p(np.abs(data))).sum(axis=0))
    return loglike


def yeojohnson_normmax(x, brack=None):
    """Compute optimal Yeo-Johnson transform parameter.

    Compute optimal Yeo-Johnson transform parameter for input data, using
    maximum likelihood estimation.

    Parameters
    ----------
    x : array_like
        Input array.
    brack : 2-tuple, optional
        The starting interval for a downhill bracket search with
        `optimize.brent`. Note that this is in most cases not critical; the
        final result is allowed to be outside this bracket. If None,
        `optimize.fminbound` is used with bounds that avoid overflow.

    Returns
    -------
    maxlog : float
        The optimal transform parameter found.

    See Also
    --------
    yeojohnson, yeojohnson_llf, yeojohnson_normplot

    Notes
    -----
    .. versionadded:: 1.2.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt

    Generate some data and determine optimal ``lmbda``

    >>> rng = np.random.default_rng()
    >>> x = stats.loggamma.rvs(5, size=30, random_state=rng) + 5
    >>> lmax = stats.yeojohnson_normmax(x)

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> prob = stats.yeojohnson_normplot(x, -10, 10, plot=ax)
    >>> ax.axvline(lmax, color='r')

    >>> plt.show()

    """
    def _neg_llf(lmbda, data):
        llf = yeojohnson_llf(lmbda, data)
        # reject likelihoods that are inf which are likely due to small
        # variance in the transformed space
        llf[np.isinf(llf)] = -np.inf
        return -llf

    with np.errstate(invalid='ignore'):
        if not np.all(np.isfinite(x)):
            raise ValueError('Yeo-Johnson input must be finite.')
        if np.all(x == 0):
            return 1.0
        if brack is not None:
            return optimize.brent(_neg_llf, brack=brack, args=(x,))
        x = np.asarray(x)
        dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64
        # Allow values up to 20 times the maximum observed value to be safely
        # transformed without over- or underflow.
        log1p_max_x = np.log1p(20 * np.max(np.abs(x)))
        # Use half of floating point's exponent range to allow safe computation
        # of the variance of the transformed data.
        log_eps = np.log(np.finfo(dtype).eps)
        log_tiny_float = (np.log(np.finfo(dtype).tiny) - log_eps) / 2
        log_max_float = (np.log(np.finfo(dtype).max) + log_eps) / 2
        # Compute the bounds by approximating the inverse of the Yeo-Johnson
        # transform on the smallest and largest floating point exponents, given
        # the largest data we expect to observe. See [1] for further details.
        # [1] https://github.com/scipy/scipy/pull/18852#issuecomment-1630286174
        lb = log_tiny_float / log1p_max_x
        ub = log_max_float / log1p_max_x
        # Convert the bounds if all or some of the data is negative.
        if np.all(x < 0):
            lb, ub = 2 - ub, 2 - lb
        elif np.any(x < 0):
            lb, ub = max(2 - ub, lb), min(2 - lb, ub)
        # Match `optimize.brent`'s tolerance.
        tol_brent = 1.48e-08
        return optimize.fminbound(_neg_llf, lb, ub, args=(x,), xtol=tol_brent)


def yeojohnson_normplot(x, la, lb, plot=None, N=80):
    """Compute parameters for a Yeo-Johnson normality plot, optionally show it.

    A Yeo-Johnson normality plot shows graphically what the best
    transformation parameter is to use in `yeojohnson` to obtain a
    distribution that is close to normal.

    Parameters
    ----------
    x : array_like
        Input array.
    la, lb : scalar
        The lower and upper bounds for the ``lmbda`` values to pass to
        `yeojohnson` for Yeo-Johnson transformations. These are also the
        limits of the horizontal axis of the plot if that is generated.
    plot : object, optional
        If given, plots the quantiles and least squares fit.
        `plot` is an object that has to have methods "plot" and "text".
        The `matplotlib.pyplot` module or a Matplotlib Axes object can be used,
        or a custom object with the same methods.
        Default is None, which means that no plot is created.
    N : int, optional
        Number of points on the horizontal axis (equally distributed from
        `la` to `lb`).

    Returns
    -------
    lmbdas : ndarray
        The ``lmbda`` values for which a Yeo-Johnson transform was done.
    ppcc : ndarray
        Probability Plot Correlation Coefficient, as obtained from `probplot`
        when fitting the Box-Cox transformed input `x` against a normal
        distribution.

    See Also
    --------
    probplot, yeojohnson, yeojohnson_normmax, yeojohnson_llf, ppcc_max

    Notes
    -----
    Even if `plot` is given, the figure is not shown or saved by
    `boxcox_normplot`; ``plt.show()`` or ``plt.savefig('figname.png')``
    should be used after calling `probplot`.

    .. versionadded:: 1.2.0

    Examples
    --------
    >>> from scipy import stats
    >>> import matplotlib.pyplot as plt

    Generate some non-normally distributed data, and create a Yeo-Johnson plot:

    >>> x = stats.loggamma.rvs(5, size=500) + 5
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> prob = stats.yeojohnson_normplot(x, -20, 20, plot=ax)

    Determine and plot the optimal ``lmbda`` to transform ``x`` and plot it in
    the same plot:

    >>> _, maxlog = stats.yeojohnson(x)
    >>> ax.axvline(maxlog, color='r')

    >>> plt.show()

    """
    return _normplot('yeojohnson', x, la, lb, plot, N)


ShapiroResult = namedtuple('ShapiroResult', ('statistic', 'pvalue'))


@_axis_nan_policy_factory(ShapiroResult, n_samples=1, too_small=2, default_axis=None)
def shapiro(x):
    r"""Perform the Shapiro-Wilk test for normality.

    The Shapiro-Wilk test tests the null hypothesis that the
    data was drawn from a normal distribution.

    Parameters
    ----------
    x : array_like
        Array of sample data. Must contain at least three observations.

    Returns
    -------
    statistic : float
        The test statistic.
    p-value : float
        The p-value for the hypothesis test.

    See Also
    --------
    anderson : The Anderson-Darling test for normality
    kstest : The Kolmogorov-Smirnov test for goodness of fit.
    :ref:`hypothesis_shapiro` : Extended example

    Notes
    -----
    The algorithm used is described in [4]_ but censoring parameters as
    described are not implemented. For N > 5000 the W test statistic is
    accurate, but the p-value may not be.

    References
    ----------
    .. [1] https://www.itl.nist.gov/div898/handbook/prc/section2/prc213.htm
           :doi:`10.18434/M32189`
    .. [2] Shapiro, S. S. & Wilk, M.B, "An analysis of variance test for
           normality (complete samples)", Biometrika, 1965, Vol. 52,
           pp. 591-611, :doi:`10.2307/2333709`
    .. [3] Razali, N. M. & Wah, Y. B., "Power comparisons of Shapiro-Wilk,
           Kolmogorov-Smirnov, Lilliefors and Anderson-Darling tests", Journal
           of Statistical Modeling and Analytics, 2011, Vol. 2, pp. 21-33.
    .. [4] Royston P., "Remark AS R94: A Remark on Algorithm AS 181: The
           W-test for Normality", 1995, Applied Statistics, Vol. 44,
           :doi:`10.2307/2986146`

    Examples
    --------

    >>> import numpy as np
    >>> from scipy import stats
    >>> rng = np.random.default_rng()
    >>> x = stats.norm.rvs(loc=5, scale=3, size=100, random_state=rng)
    >>> shapiro_test = stats.shapiro(x)
    >>> shapiro_test
    ShapiroResult(statistic=0.9813305735588074, pvalue=0.16855233907699585)
    >>> shapiro_test.statistic
    0.9813305735588074
    >>> shapiro_test.pvalue
    0.16855233907699585

    For a more detailed example, see :ref:`hypothesis_shapiro`.
    """
    x = np.ravel(x).astype(np.float64)

    N = len(x)
    if N < 3:
        raise ValueError("Data must be at least length 3.")

    a = zeros(N//2, dtype=np.float64)
    init = 0

    y = sort(x)
    y -= x[N//2]  # subtract the median (or a nearby value); see gh-15777

    w, pw, ifault = swilk(y, a, init)
    if ifault not in [0, 2]:
        warnings.warn("scipy.stats.shapiro: Input data has range zero. The"
                      " results may not be accurate.", stacklevel=2)
    if N > 5000:
        warnings.warn("scipy.stats.shapiro: For N > 5000, computed p-value "
                      f"may not be accurate. Current N is {N}.",
                      stacklevel=2)

    # `w` and `pw` are always Python floats, which are double precision.
    # We want to ensure that they are NumPy floats, so until dtypes are
    # respected, we can explicitly convert each to float64 (faster than
    # `np.array([w, pw])`).
    return ShapiroResult(np.float64(w), np.float64(pw))


# Values from Stephens, M A, "EDF Statistics for Goodness of Fit and
#             Some Comparisons", Journal of the American Statistical
#             Association, Vol. 69, Issue 347, Sept. 1974, pp 730-737
_Avals_norm = array([0.576, 0.656, 0.787, 0.918, 1.092])
_Avals_expon = array([0.922, 1.078, 1.341, 1.606, 1.957])
# From Stephens, M A, "Goodness of Fit for the Extreme Value Distribution",
#             Biometrika, Vol. 64, Issue 3, Dec. 1977, pp 583-588.
_Avals_gumbel = array([0.474, 0.637, 0.757, 0.877, 1.038])
# From Stephens, M A, "Tests of Fit for the Logistic Distribution Based
#             on the Empirical Distribution Function.", Biometrika,
#             Vol. 66, Issue 3, Dec. 1979, pp 591-595.
_Avals_logistic = array([0.426, 0.563, 0.660, 0.769, 0.906, 1.010])
# From Richard A. Lockhart and Michael A. Stephens "Estimation and Tests of
#             Fit for the Three-Parameter Weibull Distribution"
#             Journal of the Royal Statistical Society.Series B(Methodological)
#             Vol. 56, No. 3 (1994), pp. 491-500, table 1. Keys are c*100
_Avals_weibull = [[0.292, 0.395, 0.467, 0.522, 0.617, 0.711, 0.836, 0.931],
                  [0.295, 0.399, 0.471, 0.527, 0.623, 0.719, 0.845, 0.941],
                  [0.298, 0.403, 0.476, 0.534, 0.631, 0.728, 0.856, 0.954],
                  [0.301, 0.408, 0.483, 0.541, 0.640, 0.738, 0.869, 0.969],
                  [0.305, 0.414, 0.490, 0.549, 0.650, 0.751, 0.885, 0.986],
                  [0.309, 0.421, 0.498, 0.559, 0.662, 0.765, 0.902, 1.007],
                  [0.314, 0.429, 0.508, 0.570, 0.676, 0.782, 0.923, 1.030],
                  [0.320, 0.438, 0.519, 0.583, 0.692, 0.802, 0.947, 1.057],
                  [0.327, 0.448, 0.532, 0.598, 0.711, 0.824, 0.974, 1.089],
                  [0.334, 0.469, 0.547, 0.615, 0.732, 0.850, 1.006, 1.125],
                  [0.342, 0.472, 0.563, 0.636, 0.757, 0.879, 1.043, 1.167]]
_Avals_weibull = np.array(_Avals_weibull)
_cvals_weibull = np.linspace(0, 0.5, 11)
_get_As_weibull = interpolate.interp1d(_cvals_weibull, _Avals_weibull.T,
                                       kind='linear', bounds_error=False,
                                       fill_value=_Avals_weibull[-1])


def _weibull_fit_check(params, x):
    # Refine the fit returned by `weibull_min.fit` to ensure that the first
    # order necessary conditions are satisfied. If not, raise an error.
    # Here, use `m` for the shape parameter to be consistent with [7]
    # and avoid confusion with `c` as defined in [7].
    n = len(x)
    m, u, s = params

    def dnllf_dm(m, u):
        # Partial w.r.t. shape w/ optimal scale. See [7] Equation 5.
        xu = x-u
        return (1/m - (xu**m*np.log(xu)).sum()/(xu**m).sum()
                + np.log(xu).sum()/n)

    def dnllf_du(m, u):
        # Partial w.r.t. loc w/ optimal scale. See [7] Equation 6.
        xu = x-u
        return (m-1)/m*(xu**-1).sum() - n*(xu**(m-1)).sum()/(xu**m).sum()

    def get_scale(m, u):
        # Partial w.r.t. scale solved in terms of shape and location.
        # See [7] Equation 7.
        return ((x-u)**m/n).sum()**(1/m)

    def dnllf(params):
        # Partial derivatives of the NLLF w.r.t. parameters, i.e.
        # first order necessary conditions for MLE fit.
        return [dnllf_dm(*params), dnllf_du(*params)]

    suggestion = ("Maximum likelihood estimation is known to be challenging "
                  "for the three-parameter Weibull distribution. Consider "
                  "performing a custom goodness-of-fit test using "
                  "`scipy.stats.monte_carlo_test`.")

    if np.allclose(u, np.min(x)) or m < 1:
        # The critical values provided by [7] don't seem to control the
        # Type I error rate in this case. Error out.
        message = ("Maximum likelihood estimation has converged to "
                   "a solution in which the location is equal to the minimum "
                   "of the data, the shape parameter is less than 2, or both. "
                   "The table of critical values in [7] does not "
                   "include this case. " + suggestion)
        raise ValueError(message)

    try:
        # Refine the MLE / verify that first-order necessary conditions are
        # satisfied. If so, the critical values provided in [7] seem reliable.
        with np.errstate(over='raise', invalid='raise'):
            res = optimize.root(dnllf, params[:-1])

        message = ("Solution of MLE first-order conditions failed: "
                   f"{res.message}. `anderson` cannot continue. " + suggestion)
        if not res.success:
            raise ValueError(message)

    except (FloatingPointError, ValueError) as e:
        message = ("An error occurred while fitting the Weibull distribution "
                   "to the data, so `anderson` cannot continue. " + suggestion)
        raise ValueError(message) from e

    m, u = res.x
    s = get_scale(m, u)
    return m, u, s


AndersonResult = _make_tuple_bunch('AndersonResult',
                                   ['statistic', 'critical_values',
                                    'significance_level'], ['fit_result'])


def anderson(x, dist='norm'):
    """Anderson-Darling test for data coming from a particular distribution.

    The Anderson-Darling test tests the null hypothesis that a sample is
    drawn from a population that follows a particular distribution.
    For the Anderson-Darling test, the critical values depend on
    which distribution is being tested against.  This function works
    for normal, exponential, logistic, weibull_min, or Gumbel (Extreme Value
    Type I) distributions.

    Parameters
    ----------
    x : array_like
        Array of sample data.
    dist : {'norm', 'expon', 'logistic', 'gumbel', 'gumbel_l', 'gumbel_r', 'extreme1', 'weibull_min'}, optional
        The type of distribution to test against.  The default is 'norm'.
        The names 'extreme1', 'gumbel_l' and 'gumbel' are synonyms for the
        same distribution.

    Returns
    -------
    result : AndersonResult
        An object with the following attributes:

        statistic : float
            The Anderson-Darling test statistic.
        critical_values : list
            The critical values for this distribution.
        significance_level : list
            The significance levels for the corresponding critical values
            in percents.  The function returns critical values for a
