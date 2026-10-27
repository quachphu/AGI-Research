    >>> res_asymptotic = ttest_ind(x, y, alternative='less')
    >>> print(res_asymptotic.pvalue)
    0.0014669545224902675

    The permutation distribution of the test statistic is provided for
    further investigation.

    >>> import matplotlib.pyplot as plt
    >>> plt.hist(res.null_distribution, bins=50)
    >>> plt.title("Permutation distribution of test statistic")
    >>> plt.xlabel("Value of Statistic")
    >>> plt.ylabel("Frequency")
    >>> plt.show()

    Inspection of the null distribution is essential if the statistic suffers
    from inaccuracy due to limited machine precision. Consider the following
    case:

    >>> from scipy.stats import pearsonr
    >>> x = [1, 2, 4, 3]
    >>> y = [2, 4, 6, 8]
    >>> def statistic(x, y, axis=-1):
    ...     return pearsonr(x, y, axis=axis).statistic
    >>> res = permutation_test((x, y), statistic, vectorized=True,
    ...                        permutation_type='pairings',
    ...                        alternative='greater')
    >>> r, pvalue, null = res.statistic, res.pvalue, res.null_distribution

    In this case, some elements of the null distribution differ from the
    observed value of the correlation coefficient ``r`` due to numerical noise.
    We manually inspect the elements of the null distribution that are nearly
    the same as the observed value of the test statistic.

    >>> r
    0.7999999999999999
    >>> unique = np.unique(null)
    >>> unique
    array([-1. , -1. , -0.8, -0.8, -0.8, -0.6, -0.4, -0.4, -0.2, -0.2, -0.2,
        0. ,  0.2,  0.2,  0.2,  0.4,  0.4,  0.6,  0.8,  0.8,  0.8,  1. ,
        1. ])  # may vary
    >>> unique[np.isclose(r, unique)].tolist()
    [0.7999999999999998, 0.7999999999999999, 0.8]  # may vary

    If `permutation_test` were to perform the comparison naively, the
    elements of the null distribution with value ``0.7999999999999998`` would
    not be considered as extreme or more extreme as the observed value of the
    statistic, so the calculated p-value would be too small.

    >>> incorrect_pvalue = np.count_nonzero(null >= r) / len(null)
    >>> incorrect_pvalue
    0.14583333333333334  # may vary

    Instead, `permutation_test` treats elements of the null distribution that
    are within ``max(1e-14, abs(r)*1e-14)`` of the observed value of the
    statistic ``r`` to be equal to ``r``.

    >>> correct_pvalue = np.count_nonzero(null >= r - 1e-14) / len(null)
    >>> correct_pvalue
    0.16666666666666666
    >>> res.pvalue == correct_pvalue
    True

    This method of comparison is expected to be accurate in most practical
    situations, but the user is advised to assess this by inspecting the
    elements of the null distribution that are close to the observed value
    of the statistic. Also, consider the use of statistics that can be
    calculated using exact arithmetic (e.g. integer statistics).

    """
    args = _permutation_test_iv(data, statistic, permutation_type, vectorized,
                                n_resamples, batch, alternative, axis,
                                rng)
    (data, statistic, permutation_type, vectorized, n_resamples, batch,
     alternative, axis, rng) = args

    observed = statistic(*data, axis=-1)

    null_calculators = {"pairings": _calculate_null_pairings,
                        "samples": _calculate_null_samples,
                        "independent": _calculate_null_both}
    null_calculator_args = (data, statistic, n_resamples,
                            batch, rng)
    calculate_null = null_calculators[permutation_type]
    null_distribution, n_resamples, exact_test = (
        calculate_null(*null_calculator_args))

    # See References [2] and [3]
    adjustment = 0 if exact_test else 1

    # relative tolerance for detecting numerically distinct but
    # theoretically equal values in the null distribution
    eps =  (0 if not np.issubdtype(observed.dtype, np.inexact)
            else np.finfo(observed.dtype).eps*100)
    gamma = np.abs(eps * observed)

    def less(null_distribution, observed):
        cmps = null_distribution <= observed + gamma
        pvalues = (cmps.sum(axis=0) + adjustment) / (n_resamples + adjustment)
        return pvalues

    def greater(null_distribution, observed):
        cmps = null_distribution >= observed - gamma
        pvalues = (cmps.sum(axis=0) + adjustment) / (n_resamples + adjustment)
        return pvalues

    def two_sided(null_distribution, observed):
        pvalues_less = less(null_distribution, observed)
        pvalues_greater = greater(null_distribution, observed)
        pvalues = np.minimum(pvalues_less, pvalues_greater) * 2
        return pvalues

    compare = {"less": less,
               "greater": greater,
               "two-sided": two_sided}

    pvalues = compare[alternative](null_distribution, observed)
    pvalues = np.clip(pvalues, 0, 1)

    return PermutationTestResult(observed, pvalues, null_distribution)


@dataclass
class ResamplingMethod:
    """Configuration information for a statistical resampling method.

    Instances of this class can be passed into the `method` parameter of some
    hypothesis test functions to perform a resampling or Monte Carlo version
    of the hypothesis test.

    Attributes
    ----------
    n_resamples : int
        The number of resamples to perform or Monte Carlo samples to draw.
    batch : int, optional
        The number of resamples to process in each vectorized call to
        the statistic. Batch sizes >>1 tend to be faster when the statistic
        is vectorized, but memory usage scales linearly with the batch size.
        Default is ``None``, which processes all resamples in a single batch.

    """
    n_resamples: int = 9999
    batch: int = None  # type: ignore[assignment]


@dataclass
class MonteCarloMethod(ResamplingMethod):
    """Configuration information for a Monte Carlo hypothesis test.

    Instances of this class can be passed into the `method` parameter of some
    hypothesis test functions to perform a Monte Carlo version of the
    hypothesis tests.

    Attributes
    ----------
    n_resamples : int, optional
        The number of Monte Carlo samples to draw. Default is 9999.
    batch : int, optional
        The number of Monte Carlo samples to process in each vectorized call to
        the statistic. Batch sizes >>1 tend to be faster when the statistic
        is vectorized, but memory usage scales linearly with the batch size.
        Default is ``None``, which processes all samples in a single batch.
    rvs : callable or tuple of callables, optional
        A callable or sequence of callables that generates random variates
        under the null hypothesis. Each element of `rvs` must be a callable
        that accepts keyword argument ``size`` (e.g. ``rvs(size=(m, n))``) and
        returns an N-d array sample of that shape. If `rvs` is a sequence, the
        number of callables in `rvs` must match the number of samples passed
        to the hypothesis test in which the `MonteCarloMethod` is used. Default
        is ``None``, in which case the hypothesis test function chooses values
        to match the standard version of the hypothesis test. For example,
        the null hypothesis of `scipy.stats.pearsonr` is typically that the
        samples are drawn from the standard normal distribution, so
        ``rvs = (rng.normal, rng.normal)`` where
        ``rng = np.random.default_rng()``.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator state. When `rng` is None, a new
        `numpy.random.Generator` is created using entropy from the
        operating system. Types other than `numpy.random.Generator` are
        passed to `numpy.random.default_rng` to instantiate a ``Generator``.

    """
    rvs: object = None
    rng: object = None

    def __init__(self, n_resamples=9999, batch=None, rvs=None, rng=None):
        if (rvs is not None) and (rng is not None):
            message = 'Use of `rvs` and `rng` are mutually exclusive.'
            raise ValueError(message)

        self.n_resamples = n_resamples
        self.batch = batch
        self.rvs = rvs
        self.rng = rng

    def _asdict(self):
        # `dataclasses.asdict` deepcopies; we don't want that.
        return dict(n_resamples=self.n_resamples, batch=self.batch,
                    rvs=self.rvs, rng=self.rng)


_rs_deprecation = ("Use of attribute `random_state` is deprecated and replaced by "
                   "`rng`. Support for `random_state` will be removed in SciPy 1.19.0. "
                   "To silence this warning and ensure consistent behavior in SciPy "
                   "1.19.0, control the RNG using attribute `rng`. Values set using "
                   "attribute `rng` will be validated by `np.random.default_rng`, so "
                   "the behavior corresponding with a given value may change compared "
                   "to use of `random_state`. For example, 1) `None` will result in "
                   "unpredictable random numbers, 2) an integer will result in a "
                   "different stream of random numbers, (with the same distribution), "
                   "and 3) `np.random` or `RandomState` instances will result in an "
                   "error. See the documentation of `default_rng` for more "
                   "information.")


@dataclass
class PermutationMethod(ResamplingMethod):
    """Configuration information for a permutation hypothesis test.

    Instances of this class can be passed into the `method` parameter of some
    hypothesis test functions to perform a permutation version of the
    hypothesis tests.

    Attributes
    ----------
    n_resamples : int, optional
        The number of resamples to perform. Default is 9999.
    batch : int, optional
        The number of resamples to process in each vectorized call to
        the statistic. Batch sizes >>1 tend to be faster when the statistic
        is vectorized, but memory usage scales linearly with the batch size.
        Default is ``None``, which processes all resamples in a single batch.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator used to perform resampling.

        If `rng` is passed by keyword to the initializer or the `rng` attribute is used
        directly, types other than `numpy.random.Generator` are passed to
        `numpy.random.default_rng` to instantiate a ``Generator`` before use.
        If `rng` is already a ``Generator`` instance, then the provided instance is
        used. Specify `rng` for repeatable behavior.

        If this argument is passed by position, if `random_state` is passed by keyword
        into the initializer, or if the `random_state` attribute is used directly,
        legacy behavior for `random_state` applies:

        - If `random_state` is None (or `numpy.random`), the `numpy.random.RandomState`
          singleton is used.
        - If `random_state` is an int, a new ``RandomState`` instance is used,
          seeded with `random_state`.
        - If `random_state` is already a ``Generator`` or ``RandomState`` instance then
          that instance is used.

        .. versionchanged:: 1.15.0

            As part of the `SPEC-007 <https://scientific-python.org/specs/spec-0007/>`_
            transition from use of `numpy.random.RandomState` to
            `numpy.random.Generator`, this attribute name was changed from
            `random_state` to `rng`. For an interim period, both names will continue to
            work, although only one may be specified at a time. After the interim
            period, uses of `random_state` will emit warnings. The behavior of both
            `random_state` and `rng` are outlined above, but only `rng` should be used
            in new code.

    """
    rng: object  # type: ignore[misc]
    _rng: object = field(init=False, repr=False, default=None)  # type: ignore[assignment]

    @property
    def random_state(self):
        # Uncomment in SciPy 1.17.0
        # warnings.warn(_rs_deprecation, DeprecationWarning, stacklevel=2)
        return self._random_state

    @random_state.setter
    def random_state(self, val):
        # Uncomment in SciPy 1.17.0
        # warnings.warn(_rs_deprecation, DeprecationWarning, stacklevel=2)
        self._random_state = val

    @property  # type: ignore[no-redef]
    def rng(self):  # noqa: F811
        return self._rng

    def __init__(self, n_resamples=9999, batch=None, random_state=None, *, rng=None):
        # Uncomment in SciPy 1.17.0
        # warnings.warn(_rs_deprecation.replace('attribute', 'argument'),
        #               DeprecationWarning, stacklevel=2)
        self._rng = rng
        self._random_state = random_state
        super().__init__(n_resamples=n_resamples, batch=batch)

    def _asdict(self):
        # `dataclasses.asdict` deepcopies; we don't want that.
        d = dict(n_resamples=self.n_resamples, batch=self.batch)
        if self.rng is not None:
            d['rng'] = self.rng
        if self.random_state is not None:
            d['random_state'] = self.random_state
        return d


@dataclass
class BootstrapMethod(ResamplingMethod):
    """Configuration information for a bootstrap confidence interval.

    Instances of this class can be passed into the `method` parameter of some
    confidence interval methods to generate a bootstrap confidence interval.

    Attributes
    ----------
    n_resamples : int, optional
        The number of resamples to perform. Default is 9999.
    batch : int, optional
        The number of resamples to process in each vectorized call to
        the statistic. Batch sizes >>1 tend to be faster when the statistic
        is vectorized, but memory usage scales linearly with the batch size.
        Default is ``None``, which processes all resamples in a single batch.
    rng : `numpy.random.Generator`, optional
        Pseudorandom number generator used to perform resampling.

        If `rng` is passed by keyword to the initializer or the `rng` attribute is used
        directly, types other than `numpy.random.Generator` are passed to
        `numpy.random.default_rng` to instantiate a ``Generator``  before use.
        If `rng` is already a ``Generator`` instance, then the provided instance is
        used. Specify `rng` for repeatable behavior.

        If this argument is passed by position, if `random_state` is passed by keyword
        into the initializer, or if the `random_state` attribute is used directly,
        legacy behavior for `random_state` applies:

        - If `random_state` is None (or `numpy.random`), the `numpy.random.RandomState`
          singleton is used.
        - If `random_state` is an int, a new ``RandomState`` instance is used,
          seeded with `random_state`.
        - If `random_state` is already a ``Generator`` or ``RandomState`` instance then
          that instance is used.

        .. versionchanged:: 1.15.0

            As part of the `SPEC-007 <https://scientific-python.org/specs/spec-0007/>`_
            transition from use of `numpy.random.RandomState` to
            `numpy.random.Generator`, this attribute name was changed from
            `random_state` to `rng`. For an interim period, both names will continue to
            work, although only one may be specified at a time. After the interim
            period, uses of `random_state` will emit warnings. The behavior of both
            `random_state` and `rng` are outlined above, but only `rng` should be used
            in new code.

    method : {'BCa', 'percentile', 'basic'}
        Whether to use the 'percentile' bootstrap ('percentile'), the 'basic'
        (AKA 'reverse') bootstrap ('basic'), or the bias-corrected and
        accelerated bootstrap ('BCa', default).

    """
    rng: object  # type: ignore[misc]
    _rng: object = field(init=False, repr=False, default=None)  # type: ignore[assignment]
    method: str = 'BCa'

    @property
    def random_state(self):
        # Uncomment in SciPy 1.17.0
        # warnings.warn(_rs_deprecation, DeprecationWarning, stacklevel=2)
        return self._random_state

    @random_state.setter
    def random_state(self, val):
        # Uncomment in SciPy 1.17.0
        # warnings.warn(_rs_deprecation, DeprecationWarning, stacklevel=2)
        self._random_state = val

    @property  # type: ignore[no-redef]
    def rng(self):  # noqa: F811
        return self._rng

    def __init__(self, n_resamples=9999, batch=None, random_state=None,
                 method='BCa', *, rng=None):
        # Uncomment in SciPy 1.17.0
        # warnings.warn(_rs_deprecation.replace('attribute', 'argument'),
        #               DeprecationWarning, stacklevel=2)
        self._rng = rng  # don't validate with `default_rng`
        self._random_state = random_state
        self.method = method
        super().__init__(n_resamples=n_resamples, batch=batch)

    def _asdict(self):
        # `dataclasses.asdict` deepcopies; we don't want that.
        d = dict(n_resamples=self.n_resamples, batch=self.batch,
                 method=self.method)
        if self.rng is not None:
            d['rng'] = self.rng
        if self.random_state is not None:
            d['random_state'] = self.random_state
        return d
<|endoftext|># This file is not meant for public use and will be removed in SciPy v2.0.0.

from scipy._lib.deprecation import _sub_module_deprecation


__all__: list[str] = []


def __dir__():
    return __all__


def __getattr__(name):
    return _sub_module_deprecation(sub_package="stats", module="biasedurn",
                                   private_modules=["_biasedurn"], all=__all__,
                                   attribute=name)
<|endoftext|>#
# Author:  Travis Oliphant  2002-2011 with contributions from
#          SciPy Developers 2004-2011
#
from functools import partial

from scipy import special
from scipy.special import entr, logsumexp, betaln, gammaln as gamln, zeta
from scipy._lib._util import _lazywhere, rng_integers
from scipy.interpolate import interp1d

from numpy import floor, ceil, log, exp, sqrt, log1p, expm1, tanh, cosh, sinh

import numpy as np

from ._distn_infrastructure import (rv_discrete, get_distribution_names,
                                    _vectorize_rvs_over_shapes,
                                    _ShapeInfo, _isintegral,
                                    rv_discrete_frozen)
from ._biasedurn import (_PyFishersNCHypergeometric,
                         _PyWalleniusNCHypergeometric,
                         _PyStochasticLib3)
from ._stats_pythran import _poisson_binom

import scipy.special._ufuncs as scu



class binom_gen(rv_discrete):
    r"""A binomial discrete random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `binom` is:

    .. math::

       f(k) = \binom{n}{k} p^k (1-p)^{n-k}

    for :math:`k \in \{0, 1, \dots, n\}`, :math:`0 \leq p \leq 1`

    `binom` takes :math:`n` and :math:`p` as shape parameters,
    where :math:`p` is the probability of a single success
    and :math:`1-p` is the probability of a single failure.

    This distribution uses routines from the Boost Math C++ library for
    the computation of the ``pmf``, ``cdf``, ``sf``, ``ppf`` and ``isf``
    methods. [1]_

    %(after_notes)s

    References
    ----------
    .. [1] The Boost Developers. "Boost C++ Libraries". https://www.boost.org/.

    %(example)s

    See Also
    --------
    hypergeom, nbinom, nhypergeom

    """
    def _shape_info(self):
        return [_ShapeInfo("n", True, (0, np.inf), (True, False)),
                _ShapeInfo("p", False, (0, 1), (True, True))]

    def _rvs(self, n, p, size=None, random_state=None):
        return random_state.binomial(n, p, size)

    def _argcheck(self, n, p):
        return (n >= 0) & _isintegral(n) & (p >= 0) & (p <= 1)

    def _get_support(self, n, p):
        return self.a, n

    def _logpmf(self, x, n, p):
        k = floor(x)
        combiln = (gamln(n+1) - (gamln(k+1) + gamln(n-k+1)))
        return combiln + special.xlogy(k, p) + special.xlog1py(n-k, -p)

    def _pmf(self, x, n, p):
        # binom.pmf(k) = choose(n, k) * p**k * (1-p)**(n-k)
        return scu._binom_pmf(x, n, p)

    def _cdf(self, x, n, p):
        k = floor(x)
        return scu._binom_cdf(k, n, p)

    def _sf(self, x, n, p):
        k = floor(x)
        return scu._binom_sf(k, n, p)

    def _isf(self, x, n, p):
        return scu._binom_isf(x, n, p)

    def _ppf(self, q, n, p):
        return scu._binom_ppf(q, n, p)

    def _stats(self, n, p, moments='mv'):
        mu = n * p
        var = mu - n * np.square(p)
        g1, g2 = None, None
        if 's' in moments:
            pq = p - np.square(p)
            npq_sqrt = np.sqrt(n * pq)
            t1 = np.reciprocal(npq_sqrt)
            t2 = (2.0 * p) / npq_sqrt
            g1 = t1 - t2
        if 'k' in moments:
            pq = p - np.square(p)
            npq = n * pq
            t1 = np.reciprocal(npq)
            t2 = 6.0/n
            g2 = t1 - t2
        return mu, var, g1, g2

    def _entropy(self, n, p):
        k = np.r_[0:n + 1]
        vals = self._pmf(k, n, p)
        return np.sum(entr(vals), axis=0)


binom = binom_gen(name='binom')


class bernoulli_gen(binom_gen):
    r"""A Bernoulli discrete random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `bernoulli` is:

    .. math::

       f(k) = \begin{cases}1-p  &\text{if } k = 0\\
                           p    &\text{if } k = 1\end{cases}

    for :math:`k` in :math:`\{0, 1\}`, :math:`0 \leq p \leq 1`

    `bernoulli` takes :math:`p` as shape parameter,
    where :math:`p` is the probability of a single success
    and :math:`1-p` is the probability of a single failure.

    %(after_notes)s

    %(example)s

    """
    def _shape_info(self):
        return [_ShapeInfo("p", False, (0, 1), (True, True))]

    def _rvs(self, p, size=None, random_state=None):
        return binom_gen._rvs(self, 1, p, size=size, random_state=random_state)

    def _argcheck(self, p):
        return (p >= 0) & (p <= 1)

    def _get_support(self, p):
        # Overrides binom_gen._get_support!x
        return self.a, self.b

    def _logpmf(self, x, p):
        return binom._logpmf(x, 1, p)

    def _pmf(self, x, p):
        # bernoulli.pmf(k) = 1-p  if k = 0
        #                  = p    if k = 1
        return binom._pmf(x, 1, p)

    def _cdf(self, x, p):
        return binom._cdf(x, 1, p)

    def _sf(self, x, p):
        return binom._sf(x, 1, p)

    def _isf(self, x, p):
        return binom._isf(x, 1, p)

    def _ppf(self, q, p):
        return binom._ppf(q, 1, p)

    def _stats(self, p):
        return binom._stats(1, p)

    def _entropy(self, p):
        return entr(p) + entr(1-p)


bernoulli = bernoulli_gen(b=1, name='bernoulli')


class betabinom_gen(rv_discrete):
    r"""A beta-binomial discrete random variable.

    %(before_notes)s

    Notes
    -----
    The beta-binomial distribution is a binomial distribution with a
    probability of success `p` that follows a beta distribution.

    The probability mass function for `betabinom` is:

    .. math::

       f(k) = \binom{n}{k} \frac{B(k + a, n - k + b)}{B(a, b)}

    for :math:`k \in \{0, 1, \dots, n\}`, :math:`n \geq 0`, :math:`a > 0`,
    :math:`b > 0`, where :math:`B(a, b)` is the beta function.

    `betabinom` takes :math:`n`, :math:`a`, and :math:`b` as shape parameters.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Beta-binomial_distribution

    %(after_notes)s

    .. versionadded:: 1.4.0

    See Also
    --------
    beta, binom

    %(example)s

    """
    def _shape_info(self):
        return [_ShapeInfo("n", True, (0, np.inf), (True, False)),
                _ShapeInfo("a", False, (0, np.inf), (False, False)),
                _ShapeInfo("b", False, (0, np.inf), (False, False))]

    def _rvs(self, n, a, b, size=None, random_state=None):
        p = random_state.beta(a, b, size)
        return random_state.binomial(n, p, size)

    def _get_support(self, n, a, b):
        return 0, n

    def _argcheck(self, n, a, b):
        return (n >= 0) & _isintegral(n) & (a > 0) & (b > 0)

    def _logpmf(self, x, n, a, b):
        k = floor(x)
        combiln = -log(n + 1) - betaln(n - k + 1, k + 1)
        return combiln + betaln(k + a, n - k + b) - betaln(a, b)

    def _pmf(self, x, n, a, b):
        return exp(self._logpmf(x, n, a, b))

    def _stats(self, n, a, b, moments='mv'):
        e_p = a / (a + b)
        e_q = 1 - e_p
        mu = n * e_p
        var = n * (a + b + n) * e_p * e_q / (a + b + 1)
        g1, g2 = None, None
        if 's' in moments:
            g1 = 1.0 / sqrt(var)
            g1 *= (a + b + 2 * n) * (b - a)
            g1 /= (a + b + 2) * (a + b)
        if 'k' in moments:
            g2 = (a + b).astype(e_p.dtype)
            g2 *= (a + b - 1 + 6 * n)
            g2 += 3 * a * b * (n - 2)
            g2 += 6 * n ** 2
            g2 -= 3 * e_p * b * n * (6 - n)
            g2 -= 18 * e_p * e_q * n ** 2
            g2 *= (a + b) ** 2 * (1 + a + b)
            g2 /= (n * a * b * (a + b + 2) * (a + b + 3) * (a + b + n))
            g2 -= 3
        return mu, var, g1, g2


betabinom = betabinom_gen(name='betabinom')


class nbinom_gen(rv_discrete):
    r"""A negative binomial discrete random variable.

    %(before_notes)s

    Notes
    -----
    Negative binomial distribution describes a sequence of i.i.d. Bernoulli
    trials, repeated until a predefined, non-random number of successes occurs.

    The probability mass function of the number of failures for `nbinom` is:

    .. math::

       f(k) = \binom{k+n-1}{n-1} p^n (1-p)^k

    for :math:`k \ge 0`, :math:`0 < p \leq 1`

    `nbinom` takes :math:`n` and :math:`p` as shape parameters where :math:`n`
    is the number of successes, :math:`p` is the probability of a single
    success, and :math:`1-p` is the probability of a single failure.

    Another common parameterization of the negative binomial distribution is
    in terms of the mean number of failures :math:`\mu` to achieve :math:`n`
    successes. The mean :math:`\mu` is related to the probability of success
    as

    .. math::

       p = \frac{n}{n + \mu}

    The number of successes :math:`n` may also be specified in terms of a
    "dispersion", "heterogeneity", or "aggregation" parameter :math:`\alpha`,
    which relates the mean :math:`\mu` to the variance :math:`\sigma^2`,
    e.g. :math:`\sigma^2 = \mu + \alpha \mu^2`. Regardless of the convention
    used for :math:`\alpha`,

    .. math::

       p &= \frac{\mu}{\sigma^2} \\
       n &= \frac{\mu^2}{\sigma^2 - \mu}

    This distribution uses routines from the Boost Math C++ library for
    the computation of the ``pmf``, ``cdf``, ``sf``, ``ppf``, ``isf``
    and ``stats`` methods. [1]_

    %(after_notes)s

    References
    ----------
    .. [1] The Boost Developers. "Boost C++ Libraries". https://www.boost.org/.

    %(example)s

    See Also
    --------
    hypergeom, binom, nhypergeom

    """
    def _shape_info(self):
        return [_ShapeInfo("n", True, (0, np.inf), (True, False)),
                _ShapeInfo("p", False, (0, 1), (True, True))]

    def _rvs(self, n, p, size=None, random_state=None):
        return random_state.negative_binomial(n, p, size)

    def _argcheck(self, n, p):
        return (n > 0) & (p > 0) & (p <= 1)

    def _pmf(self, x, n, p):
        # nbinom.pmf(k) = choose(k+n-1, n-1) * p**n * (1-p)**k
        return scu._nbinom_pmf(x, n, p)

    def _logpmf(self, x, n, p):
        coeff = gamln(n+x) - gamln(x+1) - gamln(n)
        return coeff + n*log(p) + special.xlog1py(x, -p)

    def _cdf(self, x, n, p):
        k = floor(x)
        return scu._nbinom_cdf(k, n, p)

    def _logcdf(self, x, n, p):
        k = floor(x)
        k, n, p = np.broadcast_arrays(k, n, p)
        cdf = self._cdf(k, n, p)
        cond = cdf > 0.5
        def f1(k, n, p):
            return np.log1p(-special.betainc(k + 1, n, 1 - p))

        # do calc in place
        logcdf = cdf
        with np.errstate(divide='ignore'):
            logcdf[cond] = f1(k[cond], n[cond], p[cond])
            logcdf[~cond] = np.log(cdf[~cond])
        return logcdf

    def _sf(self, x, n, p):
        k = floor(x)
        return scu._nbinom_sf(k, n, p)

    def _isf(self, x, n, p):
        with np.errstate(over='ignore'):  # see gh-17432
            return scu._nbinom_isf(x, n, p)

    def _ppf(self, q, n, p):
        with np.errstate(over='ignore'):  # see gh-17432
            return scu._nbinom_ppf(q, n, p)

    def _stats(self, n, p):
        return (
            scu._nbinom_mean(n, p),
            scu._nbinom_variance(n, p),
            scu._nbinom_skewness(n, p),
            scu._nbinom_kurtosis_excess(n, p),
        )


nbinom = nbinom_gen(name='nbinom')


class betanbinom_gen(rv_discrete):
    r"""A beta-negative-binomial discrete random variable.

    %(before_notes)s

    Notes
    -----
    The beta-negative-binomial distribution is a negative binomial
    distribution with a probability of success `p` that follows a
    beta distribution.

    The probability mass function for `betanbinom` is:

    .. math::

       f(k) = \binom{n + k - 1}{k} \frac{B(a + n, b + k)}{B(a, b)}

    for :math:`k \ge 0`, :math:`n \geq 0`, :math:`a > 0`,
    :math:`b > 0`, where :math:`B(a, b)` is the beta function.

    `betanbinom` takes :math:`n`, :math:`a`, and :math:`b` as shape parameters.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Beta_negative_binomial_distribution

    %(after_notes)s

    .. versionadded:: 1.12.0

    See Also
    --------
    betabinom : Beta binomial distribution

    %(example)s

    """
    def _shape_info(self):
        return [_ShapeInfo("n", True, (0, np.inf), (True, False)),
                _ShapeInfo("a", False, (0, np.inf), (False, False)),
                _ShapeInfo("b", False, (0, np.inf), (False, False))]

    def _rvs(self, n, a, b, size=None, random_state=None):
        p = random_state.beta(a, b, size)
        return random_state.negative_binomial(n, p, size)

    def _argcheck(self, n, a, b):
        return (n >= 0) & _isintegral(n) & (a > 0) & (b > 0)

    def _logpmf(self, x, n, a, b):
        k = floor(x)
        combiln = -np.log(n + k) - betaln(n, k + 1)
        return combiln + betaln(a + n, b + k) - betaln(a, b)

    def _pmf(self, x, n, a, b):
        return exp(self._logpmf(x, n, a, b))

    def _stats(self, n, a, b, moments='mv'):
        # reference: Wolfram Alpha input
        # BetaNegativeBinomialDistribution[a, b, n]
        def mean(n, a, b):
            return n * b / (a - 1.)
        mu = _lazywhere(a > 1, (n, a, b), f=mean, fillvalue=np.inf)
        def var(n, a, b):
            return (n * b * (n + a - 1.) * (a + b - 1.)
                    / ((a - 2.) * (a - 1.)**2.))
        var = _lazywhere(a > 2, (n, a, b), f=var, fillvalue=np.inf)
        g1, g2 = None, None
        def skew(n, a, b):
            return ((2 * n + a - 1.) * (2 * b + a - 1.)
                    / (a - 3.) / sqrt(n * b * (n + a - 1.) * (b + a - 1.)
                    / (a - 2.)))
        if 's' in moments:
            g1 = _lazywhere(a > 3, (n, a, b), f=skew, fillvalue=np.inf)
        def kurtosis(n, a, b):
            term = (a - 2.)
            term_2 = ((a - 1.)**2. * (a**2. + a * (6 * b - 1.)
                      + 6. * (b - 1.) * b)
                      + 3. * n**2. * ((a + 5.) * b**2. + (a + 5.)
                      * (a - 1.) * b + 2. * (a - 1.)**2)
                      + 3 * (a - 1.) * n
                      * ((a + 5.) * b**2. + (a + 5.) * (a - 1.) * b
                      + 2. * (a - 1.)**2.))
            denominator = ((a - 4.) * (a - 3.) * b * n
                           * (a + b - 1.) * (a + n - 1.))
            # Wolfram Alpha uses Pearson kurtosis, so we subtract 3 to get
            # scipy's Fisher kurtosis
            return term * term_2 / denominator - 3.
        if 'k' in moments:
            g2 = _lazywhere(a > 4, (n, a, b), f=kurtosis, fillvalue=np.inf)
        return mu, var, g1, g2


betanbinom = betanbinom_gen(name='betanbinom')


class geom_gen(rv_discrete):
    r"""A geometric discrete random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `geom` is:

    .. math::

        f(k) = (1-p)^{k-1} p

    for :math:`k \ge 1`, :math:`0 < p \leq 1`

    `geom` takes :math:`p` as shape parameter,
    where :math:`p` is the probability of a single success
    and :math:`1-p` is the probability of a single failure.

    Note that when drawing random samples, the probability of observations that exceed
    ``np.iinfo(np.int64).max`` increases rapidly as $p$ decreases below $10^{-17}$. For
    $p < 10^{-20}$, almost all observations would exceed the maximum ``int64``; however,
    the output dtype is always ``int64``, so these values are clipped to the maximum.

    %(after_notes)s

    See Also
    --------
    planck

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo("p", False, (0, 1), (True, True))]

    def _rvs(self, p, size=None, random_state=None):
        res = random_state.geometric(p, size=size)
        # RandomState.geometric can wrap around to negative values; make behavior
        # consistent with Generator.geometric by replacing with maximum integer.
        max_int = np.iinfo(res.dtype).max
        return np.where(res < 0, max_int, res)

    def _argcheck(self, p):
        return (p <= 1) & (p > 0)

    def _pmf(self, k, p):
        return np.power(1-p, k-1) * p

    def _logpmf(self, k, p):
        return special.xlog1py(k - 1, -p) + log(p)

    def _cdf(self, x, p):
        k = floor(x)
        return -expm1(log1p(-p)*k)

    def _sf(self, x, p):
        return np.exp(self._logsf(x, p))

    def _logsf(self, x, p):
        k = floor(x)
        return k*log1p(-p)

    def _ppf(self, q, p):
        vals = ceil(log1p(-q) / log1p(-p))
        temp = self._cdf(vals-1, p)
        return np.where((temp >= q) & (vals > 0), vals-1, vals)

    def _stats(self, p):
        mu = 1.0/p
        qr = 1.0-p
        var = qr / p / p
        g1 = (2.0-p) / sqrt(qr)
        g2 = np.polyval([1, -6, 6], p)/(1.0-p)
        return mu, var, g1, g2

    def _entropy(self, p):
        return -np.log(p) - np.log1p(-p) * (1.0-p) / p


geom = geom_gen(a=1, name='geom', longname="A geometric")


class hypergeom_gen(rv_discrete):
    r"""A hypergeometric discrete random variable.

    The hypergeometric distribution models drawing objects from a bin.
    `M` is the total number of objects, `n` is total number of Type I objects.
    The random variate represents the number of Type I objects in `N` drawn
    without replacement from the total population.

    %(before_notes)s

    Notes
    -----
    The symbols used to denote the shape parameters (`M`, `n`, and `N`) are not
    universally accepted.  See the Examples for a clarification of the
    definitions used here.

    The probability mass function is defined as,

    .. math:: p(k, M, n, N) = \frac{\binom{n}{k} \binom{M - n}{N - k}}
                                   {\binom{M}{N}}

    for :math:`k \in [\max(0, N - M + n), \min(n, N)]`, where the binomial
    coefficients are defined as,

    .. math:: \binom{n}{k} \equiv \frac{n!}{k! (n - k)!}.

    This distribution uses routines from the Boost Math C++ library for
    the computation of the ``pmf``, ``cdf``, ``sf`` and ``stats`` methods. [1]_

    %(after_notes)s

    References
    ----------
    .. [1] The Boost Developers. "Boost C++ Libraries". https://www.boost.org/.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import hypergeom
    >>> import matplotlib.pyplot as plt

    Suppose we have a collection of 20 animals, of which 7 are dogs.  Then if
    we want to know the probability of finding a given number of dogs if we
    choose at random 12 of the 20 animals, we can initialize a frozen
    distribution and plot the probability mass function:

    >>> [M, n, N] = [20, 7, 12]
    >>> rv = hypergeom(M, n, N)
    >>> x = np.arange(0, n+1)
    >>> pmf_dogs = rv.pmf(x)

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot(x, pmf_dogs, 'bo')
    >>> ax.vlines(x, 0, pmf_dogs, lw=2)
    >>> ax.set_xlabel('# of dogs in our group of chosen animals')
    >>> ax.set_ylabel('hypergeom PMF')
    >>> plt.show()

    Instead of using a frozen distribution we can also use `hypergeom`
    methods directly.  To for example obtain the cumulative distribution
    function, use:

    >>> prb = hypergeom.cdf(x, M, n, N)

    And to generate random numbers:

    >>> R = hypergeom.rvs(M, n, N, size=10)

    See Also
    --------
    nhypergeom, binom, nbinom

    """
    def _shape_info(self):
        return [_ShapeInfo("M", True, (0, np.inf), (True, False)),
                _ShapeInfo("n", True, (0, np.inf), (True, False)),
                _ShapeInfo("N", True, (0, np.inf), (True, False))]

    def _rvs(self, M, n, N, size=None, random_state=None):
        return random_state.hypergeometric(n, M-n, N, size=size)

    def _get_support(self, M, n, N):
        return np.maximum(N-(M-n), 0), np.minimum(n, N)

    def _argcheck(self, M, n, N):
        cond = (M > 0) & (n >= 0) & (N >= 0)
        cond &= (n <= M) & (N <= M)
        cond &= _isintegral(M) & _isintegral(n) & _isintegral(N)
        return cond

    def _logpmf(self, k, M, n, N):
        tot, good = M, n
        bad = tot - good
        result = (betaln(good+1, 1) + betaln(bad+1, 1) + betaln(tot-N+1, N+1) -
                  betaln(k+1, good-k+1) - betaln(N-k+1, bad-N+k+1) -
                  betaln(tot+1, 1))
        return result

    def _pmf(self, k, M, n, N):
        return scu._hypergeom_pmf(k, n, N, M)

    def _cdf(self, k, M, n, N):
        return scu._hypergeom_cdf(k, n, N, M)

    def _stats(self, M, n, N):
        M, n, N = 1. * M, 1. * n, 1. * N
        m = M - n

        # Boost kurtosis_excess doesn't return the same as the value
        # computed here.
        g2 = M * (M + 1) - 6. * N * (M - N) - 6. * n * m
        g2 *= (M - 1) * M * M
        g2 += 6. * n * N * (M - N) * m * (5. * M - 6)
        g2 /= n * N * (M - N) * m * (M - 2.) * (M - 3.)
        return (
            scu._hypergeom_mean(n, N, M),
            scu._hypergeom_variance(n, N, M),
            scu._hypergeom_skewness(n, N, M),
            g2,
        )

    def _entropy(self, M, n, N):
        k = np.r_[N - (M - n):min(n, N) + 1]
        vals = self.pmf(k, M, n, N)
        return np.sum(entr(vals), axis=0)

    def _sf(self, k, M, n, N):
        return scu._hypergeom_sf(k, n, N, M)

    def _logsf(self, k, M, n, N):
        res = []
        for quant, tot, good, draw in zip(*np.broadcast_arrays(k, M, n, N)):
            if (quant + 0.5) * (tot + 0.5) < (good - 0.5) * (draw - 0.5):
                # Less terms to sum if we calculate log(1-cdf)
                res.append(log1p(-exp(self.logcdf(quant, tot, good, draw))))
            else:
                # Integration over probability mass function using logsumexp
                k2 = np.arange(quant + 1, draw + 1)
                res.append(logsumexp(self._logpmf(k2, tot, good, draw)))
        return np.asarray(res)

    def _logcdf(self, k, M, n, N):
        res = []
        for quant, tot, good, draw in zip(*np.broadcast_arrays(k, M, n, N)):
            if (quant + 0.5) * (tot + 0.5) > (good - 0.5) * (draw - 0.5):
                # Less terms to sum if we calculate log(1-sf)
                res.append(log1p(-exp(self.logsf(quant, tot, good, draw))))
            else:
                # Integration over probability mass function using logsumexp
                k2 = np.arange(0, quant + 1)
                res.append(logsumexp(self._logpmf(k2, tot, good, draw)))
        return np.asarray(res)


hypergeom = hypergeom_gen(name='hypergeom')


class nhypergeom_gen(rv_discrete):
    r"""A negative hypergeometric discrete random variable.

    Consider a box containing :math:`M` balls:, :math:`n` red and
    :math:`M-n` blue. We randomly sample balls from the box, one
    at a time and *without* replacement, until we have picked :math:`r`
    blue balls. `nhypergeom` is the distribution of the number of
    red balls :math:`k` we have picked.

    %(before_notes)s

    Notes
    -----
    The symbols used to denote the shape parameters (`M`, `n`, and `r`) are not
    universally accepted. See the Examples for a clarification of the
    definitions used here.

    The probability mass function is defined as,

    .. math:: f(k; M, n, r) = \frac{{{k+r-1}\choose{k}}{{M-r-k}\choose{n-k}}}
                                   {{M \choose n}}

    for :math:`k \in [0, n]`, :math:`n \in [0, M]`, :math:`r \in [0, M-n]`,
    and the binomial coefficient is:

    .. math:: \binom{n}{k} \equiv \frac{n!}{k! (n - k)!}.

    It is equivalent to observing :math:`k` successes in :math:`k+r-1`
    samples with :math:`k+r`'th sample being a failure. The former
    can be modelled as a hypergeometric distribution. The probability
    of the latter is simply the number of failures remaining
    :math:`M-n-(r-1)` divided by the size of the remaining population
    :math:`M-(k+r-1)`. This relationship can be shown as:

    .. math:: NHG(k;M,n,r) = HG(k;M,n,k+r-1)\frac{(M-n-(r-1))}{(M-(k+r-1))}

    where :math:`NHG` is probability mass function (PMF) of the
    negative hypergeometric distribution and :math:`HG` is the
    PMF of the hypergeometric distribution.

    %(after_notes)s

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import nhypergeom
    >>> import matplotlib.pyplot as plt

    Suppose we have a collection of 20 animals, of which 7 are dogs.
    Then if we want to know the probability of finding a given number
    of dogs (successes) in a sample with exactly 12 animals that
    aren't dogs (failures), we can initialize a frozen distribution
    and plot the probability mass function:

    >>> M, n, r = [20, 7, 12]
    >>> rv = nhypergeom(M, n, r)
    >>> x = np.arange(0, n+2)
    >>> pmf_dogs = rv.pmf(x)

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot(x, pmf_dogs, 'bo')
    >>> ax.vlines(x, 0, pmf_dogs, lw=2)
    >>> ax.set_xlabel('# of dogs in our group with given 12 failures')
    >>> ax.set_ylabel('nhypergeom PMF')
    >>> plt.show()

    Instead of using a frozen distribution we can also use `nhypergeom`
    methods directly.  To for example obtain the probability mass
    function, use:

    >>> prb = nhypergeom.pmf(x, M, n, r)

    And to generate random numbers:

    >>> R = nhypergeom.rvs(M, n, r, size=10)

    To verify the relationship between `hypergeom` and `nhypergeom`, use:

    >>> from scipy.stats import hypergeom, nhypergeom
    >>> M, n, r = 45, 13, 8
    >>> k = 6
    >>> nhypergeom.pmf(k, M, n, r)
    0.06180776620271643
    >>> hypergeom.pmf(k, M, n, k+r-1) * (M - n - (r-1)) / (M - (k+r-1))
    0.06180776620271644

    See Also
    --------
    hypergeom, binom, nbinom

    References
    ----------
    .. [1] Negative Hypergeometric Distribution on Wikipedia
           https://en.wikipedia.org/wiki/Negative_hypergeometric_distribution

    .. [2] Negative Hypergeometric Distribution from
           http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Negativehypergeometric.pdf

    """

    def _shape_info(self):
        return [_ShapeInfo("M", True, (0, np.inf), (True, False)),
                _ShapeInfo("n", True, (0, np.inf), (True, False)),
                _ShapeInfo("r", True, (0, np.inf), (True, False))]

    def _get_support(self, M, n, r):
        return 0, n

    def _argcheck(self, M, n, r):
        cond = (n >= 0) & (n <= M) & (r >= 0) & (r <= M-n)
        cond &= _isintegral(M) & _isintegral(n) & _isintegral(r)
        return cond

    def _rvs(self, M, n, r, size=None, random_state=None):

        @_vectorize_rvs_over_shapes
        def _rvs1(M, n, r, size, random_state):
            # invert cdf by calculating all values in support, scalar M, n, r
            a, b = self.support(M, n, r)
            ks = np.arange(a, b+1)
            cdf = self.cdf(ks, M, n, r)
            ppf = interp1d(cdf, ks, kind='next', fill_value='extrapolate')
            rvs = ppf(random_state.uniform(size=size)).astype(int)
            if size is None:
                return rvs.item()
            return rvs

        return _rvs1(M, n, r, size=size, random_state=random_state)

    def _logpmf(self, k, M, n, r):
        cond = ((r == 0) & (k == 0))
        result = _lazywhere(~cond, (k, M, n, r),
                            lambda k, M, n, r:
                                (-betaln(k+1, r) + betaln(k+r, 1) -
                                 betaln(n-k+1, M-r-n+1) + betaln(M-r-k+1, 1) +
                                 betaln(n+1, M-n+1) - betaln(M+1, 1)),
                            fillvalue=0.0)
        return result

    def _pmf(self, k, M, n, r):
        # same as the following but numerically more precise
        # return comb(k+r-1, k) * comb(M-r-k, n-k) / comb(M, n)
        return exp(self._logpmf(k, M, n, r))

    def _stats(self, M, n, r):
        # Promote the datatype to at least float
        # mu = rn / (M-n+1)
        M, n, r = 1.*M, 1.*n, 1.*r
        mu = r*n / (M-n+1)

        var = r*(M+1)*n / ((M-n+1)*(M-n+2)) * (1 - r / (M-n+1))

        # The skew and kurtosis are mathematically
        # intractable so return `None`. See [2]_.
        g1, g2 = None, None
        return mu, var, g1, g2


nhypergeom = nhypergeom_gen(name='nhypergeom')


# FIXME: Fails _cdfvec
class logser_gen(rv_discrete):
    r"""A Logarithmic (Log-Series, Series) discrete random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `logser` is:

    .. math::

        f(k) = - \frac{p^k}{k \log(1-p)}

    for :math:`k \ge 1`, :math:`0 < p < 1`

    `logser` takes :math:`p` as shape parameter,
    where :math:`p` is the probability of a single success
    and :math:`1-p` is the probability of a single failure.

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo("p", False, (0, 1), (True, True))]

    def _rvs(self, p, size=None, random_state=None):
        # looks wrong for p>0.5, too few k=1
        # trying to use generic is worse, no k=1 at all
        return random_state.logseries(p, size=size)

    def _argcheck(self, p):
        return (p > 0) & (p < 1)

    def _pmf(self, k, p):
        # logser.pmf(k) = - p**k / (k*log(1-p))
        return -np.power(p, k) * 1.0 / k / special.log1p(-p)

    def _stats(self, p):
        r = special.log1p(-p)
        mu = p / (p - 1.0) / r
        mu2p = -p / r / (p - 1.0)**2
        var = mu2p - mu*mu
        mu3p = -p / r * (1.0+p) / (1.0 - p)**3
        mu3 = mu3p - 3*mu*mu2p + 2*mu**3
        g1 = mu3 / np.power(var, 1.5)

        mu4p = -p / r * (
            1.0 / (p-1)**2 - 6*p / (p - 1)**3 + 6*p*p / (p-1)**4)
        mu4 = mu4p - 4*mu3p*mu + 6*mu2p*mu*mu - 3*mu**4
        g2 = mu4 / var**2 - 3.0
        return mu, var, g1, g2


logser = logser_gen(a=1, name='logser', longname='A logarithmic')


class poisson_gen(rv_discrete):
    r"""A Poisson discrete random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `poisson` is:

    .. math::

        f(k) = \exp(-\mu) \frac{\mu^k}{k!}

    for :math:`k \ge 0`.

    `poisson` takes :math:`\mu \geq 0` as shape parameter.
    When :math:`\mu = 0`, the ``pmf`` method
    returns ``1.0`` at quantile :math:`k = 0`.

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo("mu", False, (0, np.inf), (True, False))]

    # Override rv_discrete._argcheck to allow mu=0.
    def _argcheck(self, mu):
        return mu >= 0

    def _rvs(self, mu, size=None, random_state=None):
        return random_state.poisson(mu, size)

    def _logpmf(self, k, mu):
        Pk = special.xlogy(k, mu) - gamln(k + 1) - mu
        return Pk

    def _pmf(self, k, mu):
        # poisson.pmf(k) = exp(-mu) * mu**k / k!
        return exp(self._logpmf(k, mu))

    def _cdf(self, x, mu):
        k = floor(x)
        return special.pdtr(k, mu)

    def _sf(self, x, mu):
        k = floor(x)
        return special.pdtrc(k, mu)

    def _ppf(self, q, mu):
        vals = ceil(special.pdtrik(q, mu))
        vals1 = np.maximum(vals - 1, 0)
        temp = special.pdtr(vals1, mu)
        return np.where(temp >= q, vals1, vals)

    def _stats(self, mu):
        var = mu
        tmp = np.asarray(mu)
        mu_nonzero = tmp > 0
        g1 = _lazywhere(mu_nonzero, (tmp,), lambda x: sqrt(1.0/x), np.inf)
        g2 = _lazywhere(mu_nonzero, (tmp,), lambda x: 1.0/x, np.inf)
        return mu, var, g1, g2


poisson = poisson_gen(name="poisson", longname='A Poisson')


class planck_gen(rv_discrete):
    r"""A Planck discrete exponential random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `planck` is:

    .. math::

        f(k) = (1-\exp(-\lambda)) \exp(-\lambda k)

    for :math:`k \ge 0` and :math:`\lambda > 0`.

    `planck` takes :math:`\lambda` as shape parameter. The Planck distribution
    can be written as a geometric distribution (`geom`) with
    :math:`p = 1 - \exp(-\lambda)` shifted by ``loc = -1``.

    %(after_notes)s

    See Also
    --------
    geom

    %(example)s

    """
    def _shape_info(self):
        return [_ShapeInfo("lambda", False, (0, np.inf), (False, False))]

    def _argcheck(self, lambda_):
        return lambda_ > 0

    def _pmf(self, k, lambda_):
        return -expm1(-lambda_)*exp(-lambda_*k)

    def _cdf(self, x, lambda_):
        k = floor(x)
        return -expm1(-lambda_*(k+1))

    def _sf(self, x, lambda_):
        return exp(self._logsf(x, lambda_))

    def _logsf(self, x, lambda_):
        k = floor(x)
        return -lambda_*(k+1)

    def _ppf(self, q, lambda_):
        vals = ceil(-1.0/lambda_ * log1p(-q)-1)
        vals1 = (vals-1).clip(*(self._get_support(lambda_)))
        temp = self._cdf(vals1, lambda_)
        return np.where(temp >= q, vals1, vals)

    def _rvs(self, lambda_, size=None, random_state=None):
        # use relation to geometric distribution for sampling
        p = -expm1(-lambda_)
        return random_state.geometric(p, size=size) - 1.0

    def _stats(self, lambda_):
        mu = 1/expm1(lambda_)
        var = exp(-lambda_)/(expm1(-lambda_))**2
        g1 = 2*cosh(lambda_/2.0)
        g2 = 4+2*cosh(lambda_)
        return mu, var, g1, g2

    def _entropy(self, lambda_):
        C = -expm1(-lambda_)
        return lambda_*exp(-lambda_)/C - log(C)


planck = planck_gen(a=0, name='planck', longname='A discrete exponential ')


class boltzmann_gen(rv_discrete):
    r"""A Boltzmann (Truncated Discrete Exponential) random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `boltzmann` is:

    .. math::

        f(k) = (1-\exp(-\lambda)) \exp(-\lambda k) / (1-\exp(-\lambda N))

    for :math:`k = 0,..., N-1`.

    `boltzmann` takes :math:`\lambda > 0` and :math:`N > 0` as shape parameters.

    %(after_notes)s

    %(example)s

    """
    def _shape_info(self):
        return [_ShapeInfo("lambda_", False, (0, np.inf), (False, False)),
                _ShapeInfo("N", True, (0, np.inf), (False, False))]

    def _argcheck(self, lambda_, N):
        return (lambda_ > 0) & (N > 0) & _isintegral(N)

    def _get_support(self, lambda_, N):
        return self.a, N - 1

    def _pmf(self, k, lambda_, N):
        # boltzmann.pmf(k) =
        #               (1-exp(-lambda_)*exp(-lambda_*k)/(1-exp(-lambda_*N))
        fact = (1-exp(-lambda_))/(1-exp(-lambda_*N))
        return fact*exp(-lambda_*k)

    def _cdf(self, x, lambda_, N):
        k = floor(x)
        return (1-exp(-lambda_*(k+1)))/(1-exp(-lambda_*N))

    def _ppf(self, q, lambda_, N):
        qnew = q*(1-exp(-lambda_*N))
        vals = ceil(-1.0/lambda_ * log(1-qnew)-1)
        vals1 = (vals-1).clip(0.0, np.inf)
        temp = self._cdf(vals1, lambda_, N)
        return np.where(temp >= q, vals1, vals)

    def _stats(self, lambda_, N):
        z = exp(-lambda_)
        zN = exp(-lambda_*N)
        mu = z/(1.0-z)-N*zN/(1-zN)
        var = z/(1.0-z)**2 - N*N*zN/(1-zN)**2
        trm = (1-zN)/(1-z)
        trm2 = (z*trm**2 - N*N*zN)
        g1 = z*(1+z)*trm**3 - N**3*zN*(1+zN)
        g1 = g1 / trm2**(1.5)
        g2 = z*(1+4*z+z*z)*trm**4 - N**4 * zN*(1+4*zN+zN*zN)
        g2 = g2 / trm2 / trm2
        return mu, var, g1, g2


boltzmann = boltzmann_gen(name='boltzmann', a=0,
                          longname='A truncated discrete exponential ')


class randint_gen(rv_discrete):
    r"""A uniform discrete random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `randint` is:

    .. math::

        f(k) = \frac{1}{\texttt{high} - \texttt{low}}

    for :math:`k \in \{\texttt{low}, \dots, \texttt{high} - 1\}`.

    `randint` takes :math:`\texttt{low}` and :math:`\texttt{high}` as shape
    parameters.

    %(after_notes)s

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import randint
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(1, 1)

    Calculate the first four moments:

    >>> low, high = 7, 31
    >>> mean, var, skew, kurt = randint.stats(low, high, moments='mvsk')

    Display the probability mass function (``pmf``):

    >>> x = np.arange(low - 5, high + 5)
    >>> ax.plot(x, randint.pmf(x, low, high), 'bo', ms=8, label='randint pmf')
    >>> ax.vlines(x, 0, randint.pmf(x, low, high), colors='b', lw=5, alpha=0.5)

    Alternatively, the distribution object can be called (as a function) to
    fix the shape and location. This returns a "frozen" RV object holding the
    given parameters fixed.

    Freeze the distribution and display the frozen ``pmf``:

    >>> rv = randint(low, high)
    >>> ax.vlines(x, 0, rv.pmf(x), colors='k', linestyles='-',
    ...           lw=1, label='frozen pmf')
    >>> ax.legend(loc='lower center')
    >>> plt.show()

    Check the relationship between the cumulative distribution function
    (``cdf``) and its inverse, the percent point function (``ppf``):

    >>> q = np.arange(low, high)
    >>> p = randint.cdf(q, low, high)
    >>> np.allclose(q, randint.ppf(p, low, high))
    True

    Generate random numbers:

    >>> r = randint.rvs(low, high, size=1000)

    """

    def _shape_info(self):
        return [_ShapeInfo("low", True, (-np.inf, np.inf), (False, False)),
                _ShapeInfo("high", True, (-np.inf, np.inf), (False, False))]

    def _argcheck(self, low, high):
        return (high > low) & _isintegral(low) & _isintegral(high)

    def _get_support(self, low, high):
        return low, high-1

    def _pmf(self, k, low, high):
        # randint.pmf(k) = 1./(high - low)
        p = np.ones_like(k) / (np.asarray(high, dtype=np.int64) - low)
        return np.where((k >= low) & (k < high), p, 0.)

    def _cdf(self, x, low, high):
        k = floor(x)
        return (k - low + 1.) / (high - low)

    def _ppf(self, q, low, high):
        vals = ceil(q * (high - low) + low) - 1
        vals1 = (vals - 1).clip(low, high)
        temp = self._cdf(vals1, low, high)
        return np.where(temp >= q, vals1, vals)

    def _stats(self, low, high):
        m2, m1 = np.asarray(high), np.asarray(low)
        mu = (m2 + m1 - 1.0) / 2
        d = m2 - m1
        var = (d*d - 1) / 12.0
        g1 = 0.0
        g2 = -6.0/5.0 * (d*d + 1.0) / (d*d - 1.0)
        return mu, var, g1, g2

    def _rvs(self, low, high, size=None, random_state=None):
        """An array of *size* random integers >= ``low`` and < ``high``."""
        if np.asarray(low).size == 1 and np.asarray(high).size == 1:
            # no need to vectorize in that case
            return rng_integers(random_state, low, high, size=size)

        if size is not None:
            # NumPy's RandomState.randint() doesn't broadcast its arguments.
            # Use `broadcast_to()` to extend the shapes of low and high
            # up to size.  Then we can use the numpy.vectorize'd
            # randint without needing to pass it a `size` argument.
            low = np.broadcast_to(low, size)
            high = np.broadcast_to(high, size)
        randint = np.vectorize(partial(rng_integers, random_state),
                               otypes=[np.dtype(int)])
        return randint(low, high)

    def _entropy(self, low, high):
        return log(high - low)


randint = randint_gen(name='randint', longname='A discrete uniform '
                      '(random integer)')


# FIXME: problems sampling.
class zipf_gen(rv_discrete):
    r"""A Zipf (Zeta) discrete random variable.

    %(before_notes)s

    See Also
    --------
    zipfian

    Notes
    -----
    The probability mass function for `zipf` is:

    .. math::

        f(k, a) = \frac{1}{\zeta(a) k^a}

    for :math:`k \ge 1`, :math:`a > 1`.

    `zipf` takes :math:`a > 1` as shape parameter. :math:`\zeta` is the
    Riemann zeta function (`scipy.special.zeta`)

    The Zipf distribution is also known as the zeta distribution, which is
    a special case of the Zipfian distribution (`zipfian`).

    %(after_notes)s

    References
    ----------
    .. [1] "Zeta Distribution", Wikipedia,
           https://en.wikipedia.org/wiki/Zeta_distribution

    %(example)s

    Confirm that `zipf` is the large `n` limit of `zipfian`.

    >>> import numpy as np
    >>> from scipy.stats import zipf, zipfian
    >>> k = np.arange(11)
    >>> np.allclose(zipf.pmf(k, a), zipfian.pmf(k, a, n=10000000))
    True

    """

    def _shape_info(self):
        return [_ShapeInfo("a", False, (1, np.inf), (False, False))]

    def _rvs(self, a, size=None, random_state=None):
        return random_state.zipf(a, size=size)

    def _argcheck(self, a):
        return a > 1

    def _pmf(self, k, a):
        k = k.astype(np.float64)
        # zipf.pmf(k, a) = 1/(zeta(a) * k**a)
        Pk = 1.0 / special.zeta(a, 1) * k**-a
        return Pk

    def _munp(self, n, a):
        return _lazywhere(
            a > n + 1, (a, n),
            lambda a, n: special.zeta(a - n, 1) / special.zeta(a, 1),
            np.inf)


zipf = zipf_gen(a=1, name='zipf', longname='A Zipf')


def _gen_harmonic_gt1(n, a):
    """Generalized harmonic number, a > 1"""
    # See https://en.wikipedia.org/wiki/Harmonic_number; search for "hurwitz"
    return zeta(a, 1) - zeta(a, n+1)


def _gen_harmonic_leq1(n, a):
    """Generalized harmonic number, a <= 1"""
    if not np.size(n):
        return n
    n_max = np.max(n)  # loop starts at maximum of all n
    out = np.zeros_like(a, dtype=float)
    # add terms of harmonic series; starting from smallest to avoid roundoff
    for i in np.arange(n_max, 0, -1, dtype=float):
        mask = i <= n  # don't add terms after nth
        out[mask] += 1/i**a[mask]
    return out


def _gen_harmonic(n, a):
    """Generalized harmonic number"""
    n, a = np.broadcast_arrays(n, a)
    return _lazywhere(a > 1, (n, a),
                      f=_gen_harmonic_gt1, f2=_gen_harmonic_leq1)


class zipfian_gen(rv_discrete):
    r"""A Zipfian discrete random variable.

    %(before_notes)s

    See Also
    --------
    zipf

    Notes
    -----
    The probability mass function for `zipfian` is:

    .. math::

        f(k, a, n) = \frac{1}{H_{n,a} k^a}

    for :math:`k \in \{1, 2, \dots, n-1, n\}`, :math:`a \ge 0`,
    :math:`n \in \{1, 2, 3, \dots\}`.

    `zipfian` takes :math:`a` and :math:`n` as shape parameters.
    :math:`H_{n,a}` is the :math:`n`:sup:`th` generalized harmonic
    number of order :math:`a`.

    The Zipfian distribution reduces to the Zipf (zeta) distribution as
    :math:`n \rightarrow \infty`.

    %(after_notes)s

    References
    ----------
    .. [1] "Zipf's Law", Wikipedia, https://en.wikipedia.org/wiki/Zipf's_law
    .. [2] Larry Leemis, "Zipf Distribution", Univariate Distribution
           Relationships. http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Zipf.pdf

    %(example)s

    Confirm that `zipfian` reduces to `zipf` for large `n`, ``a > 1``.

    >>> import numpy as np
    >>> from scipy.stats import zipf, zipfian
    >>> k = np.arange(11)
    >>> np.allclose(zipfian.pmf(k, a=3.5, n=10000000), zipf.pmf(k, a=3.5))
    True

    """

    def _shape_info(self):
        return [_ShapeInfo("a", False, (0, np.inf), (True, False)),
                _ShapeInfo("n", True, (0, np.inf), (False, False))]

    def _argcheck(self, a, n):
        # we need np.asarray here because moment (maybe others) don't convert
        return (a >= 0) & (n > 0) & (n == np.asarray(n, dtype=int))

    def _get_support(self, a, n):
        return 1, n

    def _pmf(self, k, a, n):
        k = k.astype(np.float64)
        return 1.0 / _gen_harmonic(n, a) * k**-a

    def _cdf(self, k, a, n):
        return _gen_harmonic(k, a) / _gen_harmonic(n, a)

    def _sf(self, k, a, n):
        k = k + 1  # # to match SciPy convention
        # see http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Zipf.pdf
        return ((k**a*(_gen_harmonic(n, a) - _gen_harmonic(k, a)) + 1)
                / (k**a*_gen_harmonic(n, a)))

    def _stats(self, a, n):
        # see # see http://www.math.wm.edu/~leemis/chart/UDR/PDFs/Zipf.pdf
        Hna = _gen_harmonic(n, a)
        Hna1 = _gen_harmonic(n, a-1)
        Hna2 = _gen_harmonic(n, a-2)
        Hna3 = _gen_harmonic(n, a-3)
        Hna4 = _gen_harmonic(n, a-4)
        mu1 = Hna1/Hna
        mu2n = (Hna2*Hna - Hna1**2)
        mu2d = Hna**2
        mu2 = mu2n / mu2d
        g1 = (Hna3/Hna - 3*Hna1*Hna2/Hna**2 + 2*Hna1**3/Hna**3)/mu2**(3/2)
        g2 = (Hna**3*Hna4 - 4*Hna**2*Hna1*Hna3 + 6*Hna*Hna1**2*Hna2
              - 3*Hna1**4) / mu2n**2
        g2 -= 3
        return mu1, mu2, g1, g2


zipfian = zipfian_gen(a=1, name='zipfian', longname='A Zipfian')


class dlaplace_gen(rv_discrete):
    r"""A  Laplacian discrete random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `dlaplace` is:

    .. math::

        f(k) = \tanh(a/2) \exp(-a |k|)

    for integers :math:`k` and :math:`a > 0`.

    `dlaplace` takes :math:`a` as shape parameter.

    %(after_notes)s

    %(example)s

    """

    def _shape_info(self):
        return [_ShapeInfo("a", False, (0, np.inf), (False, False))]

    def _pmf(self, k, a):
        # dlaplace.pmf(k) = tanh(a/2) * exp(-a*abs(k))
        return tanh(a/2.0) * exp(-a * abs(k))

    def _cdf(self, x, a):
        k = floor(x)

        def f(k, a):
            return 1.0 - exp(-a * k) / (exp(a) + 1)

        def f2(k, a):
            return exp(a * (k + 1)) / (exp(a) + 1)

        return _lazywhere(k >= 0, (k, a), f=f, f2=f2)

    def _ppf(self, q, a):
        const = 1 + exp(a)
        vals = ceil(np.where(q < 1.0 / (1 + exp(-a)),
                             log(q*const) / a - 1,
                             -log((1-q) * const) / a))
        vals1 = vals - 1
        return np.where(self._cdf(vals1, a) >= q, vals1, vals)

    def _stats(self, a):
        ea = exp(a)
        mu2 = 2.*ea/(ea-1.)**2
        mu4 = 2.*ea*(ea**2+10.*ea+1.) / (ea-1.)**4
        return 0., mu2, 0., mu4/mu2**2 - 3.

    def _entropy(self, a):
        return a / sinh(a) - log(tanh(a/2.0))

    def _rvs(self, a, size=None, random_state=None):
        # The discrete Laplace is equivalent to the two-sided geometric
        # distribution with PMF:
        #   f(k) = (1 - alpha)/(1 + alpha) * alpha^abs(k)
        #   Reference:
        #     https://www.sciencedirect.com/science/
        #     article/abs/pii/S0378375804003519
        # Furthermore, the two-sided geometric distribution is
        # equivalent to the difference between two iid geometric
        # distributions.
        #   Reference (page 179):
        #     https://pdfs.semanticscholar.org/61b3/
        #     b99f466815808fd0d03f5d2791eea8b541a1.pdf
        # Thus, we can leverage the following:
        #   1) alpha = e^-a
        #   2) probability_of_success = 1 - alpha (Bernoulli trial)
        probOfSuccess = -np.expm1(-np.asarray(a))
        x = random_state.geometric(probOfSuccess, size=size)
        y = random_state.geometric(probOfSuccess, size=size)
        return x - y


dlaplace = dlaplace_gen(a=-np.inf,
                        name='dlaplace', longname='A discrete Laplacian')


class poisson_binom_gen(rv_discrete):
    r"""A Poisson Binomial discrete random variable.

    %(before_notes)s

    See Also
    --------
    binom

    Notes
    -----
    The probability mass function for `poisson_binom` is:

    .. math::

     f(k; p_1, p_2, ..., p_n) = \sum_{A \in F_k} \prod_{i \in A} p_i \prod_{j \in A^C} 1 - p_j

    where :math:`k \in \{0, 1, \dots, n-1, n\}`, :math:`F_k` is the set of all
    subsets of :math:`k` integers that can be selected :math:`\{0, 1, \dots, n-1, n\}`,
    and :math:`A^C` is the complement of a set :math:`A`.

    `poisson_binom` accepts a single array argument ``p`` for shape parameters
    :math:`0 ≤ p_i ≤ 1`, where the last axis corresponds with the index :math:`i` and
    any others are for batch dimensions. Broadcasting behaves according to the usual
    rules except that the last axis of ``p`` is ignored. Instances of this class do
    not support serialization/unserialization.

    %(after_notes)s

    References
    ----------
    .. [1] "Poisson binomial distribution", Wikipedia,
           https://en.wikipedia.org/wiki/Poisson_binomial_distribution
    .. [2] Biscarri, William, Sihai Dave Zhao, and Robert J. Brunner. "A simple and
           fast method for computing the Poisson binomial distribution function".
           Computational Statistics & Data Analysis 122 (2018) 92-100.
           :doi:`10.1016/j.csda.2018.01.007`

    %(example)s

    """  # noqa: E501
    def _shape_info(self):
        # message = 'Fitting is not implemented for this distribution."
        # raise NotImplementedError(message)
        return []

    def _argcheck(self, *args):
        p = np.stack(args, axis=0)
        conds = (0 <= p) & (p <= 1)
        return np.all(conds, axis=0)

    def _rvs(self, *args, size=None, random_state=None):
        # convenient to work along the last axis here to avoid interference with `size`
        p = np.stack(args, axis=-1)
        # Size passed by the user is the *shape of the returned array*, so it won't
        # contain the length of the last axis of p.
        size = (p.shape if size is None else
                (size, 1) if np.isscalar(size) else tuple(size) + (1,))
        size = np.broadcast_shapes(p.shape, size)
        return bernoulli._rvs(p, size=size, random_state=random_state).sum(axis=-1)

    def _get_support(self, *args):
        return 0, len(args)

    def _pmf(self, k, *args):
        k = np.atleast_1d(k).astype(np.int64)
        k, *args = np.broadcast_arrays(k, *args)
        args = np.asarray(args, dtype=np.float64)
        return _poisson_binom(k, args, 'pmf')

    def _cdf(self, k, *args):
        k = np.atleast_1d(k).astype(np.int64)
        k, *args = np.broadcast_arrays(k, *args)
        args = np.asarray(args, dtype=np.float64)
        return _poisson_binom(k, args, 'cdf')

    def _stats(self, *args, **kwds):
        p = np.stack(args, axis=0)
        mean = np.sum(p, axis=0)
        var = np.sum(p * (1-p), axis=0)
        return (mean, var, None, None)

    def __call__(self, *args, **kwds):
        return poisson_binomial_frozen(self, *args, **kwds)


poisson_binom = poisson_binom_gen(name='poisson_binom', longname='A Poisson binomial',
                                  shapes='p')

# The _parse_args methods don't work with vector-valued shape parameters, so we rewrite
# them. Note that `p` is accepted as an array with the index `i` of `p_i` corresponding
# with the last axis; we return it as a tuple (p_1, p_2, ..., p_n) so that it looks
# like `n` scalar (or arrays of scalar-valued) shape parameters to the infrastructure.

def _parse_args_rvs(self, p, loc=0, size=None):
    return tuple(np.moveaxis(p, -1, 0)), loc, 1.0, size

def _parse_args_stats(self, p, loc=0, moments='mv'):
    return tuple(np.moveaxis(p, -1, 0)), loc, 1.0, moments

def _parse_args(self, p, loc=0):
    return tuple(np.moveaxis(p, -1, 0)), loc, 1.0

# The infrastructure manually binds these methods to the instance, so
# we can only override them by manually binding them, too.
_pb_obj, _pb_cls = poisson_binom, poisson_binom_gen  # shorter names (for PEP8)
poisson_binom._parse_args_rvs = _parse_args_rvs.__get__(_pb_obj, _pb_cls)
poisson_binom._parse_args_stats = _parse_args_stats.__get__(_pb_obj, _pb_cls)
poisson_binom._parse_args = _parse_args.__get__(_pb_obj, _pb_cls)

class poisson_binomial_frozen(rv_discrete_frozen):
    # copied from rv_frozen; we just need to bind the `_parse_args` methods
    def __init__(self, dist, *args, **kwds):                        # verbatim
        self.args = args                                            # verbatim
        self.kwds = kwds                                            # verbatim

        # create a new instance                                     # verbatim
        self.dist = dist.__class__(**dist._updated_ctor_param())    # verbatim

        # Here is the only modification
        self.dist._parse_args_rvs = _parse_args_rvs.__get__(_pb_obj, _pb_cls)
        self.dist._parse_args_stats = _parse_args_stats.__get__(_pb_obj, _pb_cls)
        self.dist._parse_args = _parse_args.__get__(_pb_obj, _pb_cls)

        shapes, _, _ = self.dist._parse_args(*args, **kwds)         # verbatim
        self.a, self.b = self.dist._get_support(*shapes)            # verbatim

    def expect(self, func=None, lb=None, ub=None, conditional=False, **kwds):
        a, loc, scale = self.dist._parse_args(*self.args, **self.kwds)
        # Here's the modification: we pass all args (including `loc`) into the `args`
        # parameter of `expect` so the shape only goes through `_parse_args` once.
        return self.dist.expect(func, self.args, loc, lb, ub, conditional, **kwds)


class skellam_gen(rv_discrete):
    r"""A  Skellam discrete random variable.

    %(before_notes)s

    Notes
    -----
    Probability distribution of the difference of two correlated or
    uncorrelated Poisson random variables.

    Let :math:`k_1` and :math:`k_2` be two Poisson-distributed r.v. with
    expected values :math:`\lambda_1` and :math:`\lambda_2`. Then,
    :math:`k_1 - k_2` follows a Skellam distribution with parameters
    :math:`\mu_1 = \lambda_1 - \rho \sqrt{\lambda_1 \lambda_2}` and
    :math:`\mu_2 = \lambda_2 - \rho \sqrt{\lambda_1 \lambda_2}`, where
    :math:`\rho` is the correlation coefficient between :math:`k_1` and
    :math:`k_2`. If the two Poisson-distributed r.v. are independent then
    :math:`\rho = 0`.

    Parameters :math:`\mu_1` and :math:`\mu_2` must be strictly positive.

    For details see: https://en.wikipedia.org/wiki/Skellam_distribution

    `skellam` takes :math:`\mu_1` and :math:`\mu_2` as shape parameters.

    %(after_notes)s

    %(example)s

    """
    def _shape_info(self):
        return [_ShapeInfo("mu1", False, (0, np.inf), (False, False)),
                _ShapeInfo("mu2", False, (0, np.inf), (False, False))]

    def _rvs(self, mu1, mu2, size=None, random_state=None):
        n = size
        return (random_state.poisson(mu1, n) -
                random_state.poisson(mu2, n))

    def _pmf(self, x, mu1, mu2):
        with np.errstate(over='ignore'):  # see gh-17432
            px = np.where(x < 0,
                          scu._ncx2_pdf(2*mu2, 2*(1-x), 2*mu1)*2,
                          scu._ncx2_pdf(2*mu1, 2*(1+x), 2*mu2)*2)
            # ncx2.pdf() returns nan's for extremely low probabilities
        return px

    def _cdf(self, x, mu1, mu2):
        x = floor(x)
        with np.errstate(over='ignore'):  # see gh-17432
            px = np.where(x < 0,
                          scu._ncx2_cdf(2*mu2, -2*x, 2*mu1),
                          1 - scu._ncx2_cdf(2*mu1, 2*(x+1), 2*mu2))
        return px

    def _stats(self, mu1, mu2):
        mean = mu1 - mu2
        var = mu1 + mu2
        g1 = mean / sqrt((var)**3)
        g2 = 1 / var
        return mean, var, g1, g2


skellam = skellam_gen(a=-np.inf, name="skellam", longname='A Skellam')


class yulesimon_gen(rv_discrete):
    r"""A Yule-Simon discrete random variable.

    %(before_notes)s

    Notes
    -----

    The probability mass function for the `yulesimon` is:

    .. math::

        f(k) =  \alpha B(k, \alpha+1)

    for :math:`k=1,2,3,...`, where :math:`\alpha>0`.
    Here :math:`B` refers to the `scipy.special.beta` function.

    The sampling of random variates is based on pg 553, Section 6.3 of [1]_.
    Our notation maps to the referenced logic via :math:`\alpha=a-1`.

    For details see the wikipedia entry [2]_.

    References
    ----------
    .. [1] Devroye, Luc. "Non-uniform Random Variate Generation",
         (1986) Springer, New York.

    .. [2] https://en.wikipedia.org/wiki/Yule-Simon_distribution

    %(after_notes)s

    %(example)s

    """
    def _shape_info(self):
        return [_ShapeInfo("alpha", False, (0, np.inf), (False, False))]

    def _rvs(self, alpha, size=None, random_state=None):
        E1 = random_state.standard_exponential(size)
        E2 = random_state.standard_exponential(size)
        ans = ceil(-E1 / log1p(-exp(-E2 / alpha)))
        return ans

    def _pmf(self, x, alpha):
        return alpha * special.beta(x, alpha + 1)

    def _argcheck(self, alpha):
        return (alpha > 0)

    def _logpmf(self, x, alpha):
        return log(alpha) + special.betaln(x, alpha + 1)

    def _cdf(self, x, alpha):
        return 1 - x * special.beta(x, alpha + 1)

    def _sf(self, x, alpha):
        return x * special.beta(x, alpha + 1)

    def _logsf(self, x, alpha):
        return log(x) + special.betaln(x, alpha + 1)

    def _stats(self, alpha):
        mu = np.where(alpha <= 1, np.inf, alpha / (alpha - 1))
        mu2 = np.where(alpha > 2,
                       alpha**2 / ((alpha - 2.0) * (alpha - 1)**2),
                       np.inf)
        mu2 = np.where(alpha <= 1, np.nan, mu2)
        g1 = np.where(alpha > 3,
                      sqrt(alpha - 2) * (alpha + 1)**2 / (alpha * (alpha - 3)),
                      np.inf)
        g1 = np.where(alpha <= 2, np.nan, g1)
        g2 = np.where(alpha > 4,
                      alpha + 3 + ((11 * alpha**3 - 49 * alpha - 22) /
                                   (alpha * (alpha - 4) * (alpha - 3))),
                      np.inf)
        g2 = np.where(alpha <= 2, np.nan, g2)
        return mu, mu2, g1, g2


yulesimon = yulesimon_gen(name='yulesimon', a=1)


class _nchypergeom_gen(rv_discrete):
    r"""A noncentral hypergeometric discrete random variable.

    For subclassing by nchypergeom_fisher_gen and nchypergeom_wallenius_gen.

    """

    rvs_name = None
    dist = None

    def _shape_info(self):
        return [_ShapeInfo("M", True, (0, np.inf), (True, False)),
                _ShapeInfo("n", True, (0, np.inf), (True, False)),
                _ShapeInfo("N", True, (0, np.inf), (True, False)),
                _ShapeInfo("odds", False, (0, np.inf), (False, False))]

    def _get_support(self, M, n, N, odds):
        N, m1, n = M, n, N  # follow Wikipedia notation
        m2 = N - m1
        x_min = np.maximum(0, n - m2)
        x_max = np.minimum(n, m1)
        return x_min, x_max

    def _argcheck(self, M, n, N, odds):
        M, n = np.asarray(M), np.asarray(n),
        N, odds = np.asarray(N), np.asarray(odds)
        cond1 = (M.astype(int) == M) & (M >= 0)
        cond2 = (n.astype(int) == n) & (n >= 0)
        cond3 = (N.astype(int) == N) & (N >= 0)
        cond4 = odds > 0
        cond5 = N <= M
        cond6 = n <= M
        return cond1 & cond2 & cond3 & cond4 & cond5 & cond6

    def _rvs(self, M, n, N, odds, size=None, random_state=None):

        @_vectorize_rvs_over_shapes
        def _rvs1(M, n, N, odds, size, random_state):
            length = np.prod(size)
            urn = _PyStochasticLib3()
            rv_gen = getattr(urn, self.rvs_name)
            rvs = rv_gen(N, n, M, odds, length, random_state)
            rvs = rvs.reshape(size)
            return rvs

        return _rvs1(M, n, N, odds, size=size, random_state=random_state)

    def _pmf(self, x, M, n, N, odds):

        x, M, n, N, odds = np.broadcast_arrays(x, M, n, N, odds)
        if x.size == 0:  # np.vectorize doesn't work with zero size input
            return np.empty_like(x)

        @np.vectorize
        def _pmf1(x, M, n, N, odds):
            urn = self.dist(N, n, M, odds, 1e-12)
            return urn.probability(x)

        return _pmf1(x, M, n, N, odds)

    def _stats(self, M, n, N, odds, moments):

        @np.vectorize
        def _moments1(M, n, N, odds):
            urn = self.dist(N, n, M, odds, 1e-12)
            return urn.moments()

        m, v = (_moments1(M, n, N, odds) if ("m" in moments or "v" in moments)
                else (None, None))
        s, k = None, None
        return m, v, s, k


class nchypergeom_fisher_gen(_nchypergeom_gen):
    r"""A Fisher's noncentral hypergeometric discrete random variable.

    Fisher's noncentral hypergeometric distribution models drawing objects of
    two types from a bin. `M` is the total number of objects, `n` is the
    number of Type I objects, and `odds` is the odds ratio: the odds of
    selecting a Type I object rather than a Type II object when there is only
    one object of each type.
    The random variate represents the number of Type I objects drawn if we
    take a handful of objects from the bin at once and find out afterwards
    that we took `N` objects.

    %(before_notes)s

    See Also
    --------
    nchypergeom_wallenius, hypergeom, nhypergeom

    Notes
    -----
    Let mathematical symbols :math:`N`, :math:`n`, and :math:`M` correspond
    with parameters `N`, `n`, and `M` (respectively) as defined above.

    The probability mass function is defined as

    .. math::

        p(x; M, n, N, \omega) =
        \frac{\binom{n}{x}\binom{M - n}{N-x}\omega^x}{P_0},

    for
    :math:`x \in [x_l, x_u]`,
    :math:`M \in {\mathbb N}`,
    :math:`n \in [0, M]`,
    :math:`N \in [0, M]`,
    :math:`\omega > 0`,
    where
    :math:`x_l = \max(0, N - (M - n))`,
    :math:`x_u = \min(N, n)`,

    .. math::

        P_0 = \sum_{y=x_l}^{x_u} \binom{n}{y}\binom{M - n}{N-y}\omega^y,

    and the binomial coefficients are defined as

    .. math:: \binom{n}{k} \equiv \frac{n!}{k! (n - k)!}.

    `nchypergeom_fisher` uses the BiasedUrn package by Agner Fog with
    permission for it to be distributed under SciPy's license.

    The symbols used to denote the shape parameters (`N`, `n`, and `M`) are not
    universally accepted; they are chosen for consistency with `hypergeom`.

    Note that Fisher's noncentral hypergeometric distribution is distinct
    from Wallenius' noncentral hypergeometric distribution, which models
    drawing a pre-determined `N` objects from a bin one by one.
    When the odds ratio is unity, however, both distributions reduce to the
    ordinary hypergeometric distribution.

    %(after_notes)s

    References
    ----------
    .. [1] Agner Fog, "Biased Urn Theory".
           https://cran.r-project.org/web/packages/BiasedUrn/vignettes/UrnTheory.pdf

    .. [2] "Fisher's noncentral hypergeometric distribution", Wikipedia,
           https://en.wikipedia.org/wiki/Fisher's_noncentral_hypergeometric_distribution

    %(example)s

    """

    rvs_name = "rvs_fisher"
    dist = _PyFishersNCHypergeometric


nchypergeom_fisher = nchypergeom_fisher_gen(
    name='nchypergeom_fisher',
    longname="A Fisher's noncentral hypergeometric")


class nchypergeom_wallenius_gen(_nchypergeom_gen):
    r"""A Wallenius' noncentral hypergeometric discrete random variable.

    Wallenius' noncentral hypergeometric distribution models drawing objects of
    two types from a bin. `M` is the total number of objects, `n` is the
    number of Type I objects, and `odds` is the odds ratio: the odds of
    selecting a Type I object rather than a Type II object when there is only
    one object of each type.
    The random variate represents the number of Type I objects drawn if we
    draw a pre-determined `N` objects from a bin one by one.

    %(before_notes)s

    See Also
    --------
    nchypergeom_fisher, hypergeom, nhypergeom

    Notes
    -----
    Let mathematical symbols :math:`N`, :math:`n`, and :math:`M` correspond
    with parameters `N`, `n`, and `M` (respectively) as defined above.

    The probability mass function is defined as

    .. math::

        p(x; N, n, M) = \binom{n}{x} \binom{M - n}{N-x}
        \int_0^1 \left(1-t^{\omega/D}\right)^x\left(1-t^{1/D}\right)^{N-x} dt

    for
    :math:`x \in [x_l, x_u]`,
    :math:`M \in {\mathbb N}`,
    :math:`n \in [0, M]`,
    :math:`N \in [0, M]`,
    :math:`\omega > 0`,
    where
    :math:`x_l = \max(0, N - (M - n))`,
    :math:`x_u = \min(N, n)`,

    .. math::

        D = \omega(n - x) + ((M - n)-(N-x)),

    and the binomial coefficients are defined as

    .. math:: \binom{n}{k} \equiv \frac{n!}{k! (n - k)!}.

    `nchypergeom_wallenius` uses the BiasedUrn package by Agner Fog with
    permission for it to be distributed under SciPy's license.

    The symbols used to denote the shape parameters (`N`, `n`, and `M`) are not
    universally accepted; they are chosen for consistency with `hypergeom`.

    Note that Wallenius' noncentral hypergeometric distribution is distinct
    from Fisher's noncentral hypergeometric distribution, which models
    take a handful of objects from the bin at once, finding out afterwards
    that `N` objects were taken.
    When the odds ratio is unity, however, both distributions reduce to the
    ordinary hypergeometric distribution.

    %(after_notes)s

    References
    ----------
    .. [1] Agner Fog, "Biased Urn Theory".
           https://cran.r-project.org/web/packages/BiasedUrn/vignettes/UrnTheory.pdf

    .. [2] "Wallenius' noncentral hypergeometric distribution", Wikipedia,
           https://en.wikipedia.org/wiki/Wallenius'_noncentral_hypergeometric_distribution

    %(example)s

    """

    rvs_name = "rvs_wallenius"
    dist = _PyWalleniusNCHypergeometric


nchypergeom_wallenius = nchypergeom_wallenius_gen(
    name='nchypergeom_wallenius',
    longname="A Wallenius' noncentral hypergeometric")


# Collect names of classes and objects in this module.
pairs = list(globals().copy().items())
_distn_names, _distn_gen_names = get_distribution_names(pairs, rv_discrete)

__all__ = _distn_names + _distn_gen_names
<|endoftext|># This module exists only to allow Sphinx to generate docs
# for the result objects returned by some functions in stats
# _without_ adding them to the main stats documentation page.

"""
Result classes
--------------

.. currentmodule:: scipy.stats._result_classes

.. autosummary::
   :toctree: generated/

   RelativeRiskResult
   BinomTestResult
   TukeyHSDResult
   DunnettResult
   PearsonRResult
   FitResult
   OddsRatioResult
   TtestResult
   ECDFResult
   EmpiricalDistributionFunction

"""

__all__ = ['BinomTestResult', 'RelativeRiskResult', 'TukeyHSDResult',
           'PearsonRResult', 'FitResult', 'OddsRatioResult',
           'TtestResult', 'DunnettResult', 'ECDFResult',
           'EmpiricalDistributionFunction']


from ._binomtest import BinomTestResult
from ._odds_ratio import OddsRatioResult
from ._relative_risk import RelativeRiskResult
from ._hypotests import TukeyHSDResult
from ._multicomp import DunnettResult
from ._stats_py import PearsonRResult, TtestResult
from ._fit import FitResult
from ._survival import ECDFResult, EmpiricalDistributionFunction
<|endoftext|>import math
import numbers
import numpy as np
from scipy import stats
from scipy import special as sc
from ._qmc import (check_random_state as check_random_state_qmc,
                   Halton, QMCEngine)
from ._unuran.unuran_wrapper import NumericalInversePolynomial
from scipy._lib._util import check_random_state


__all__ = ['FastGeneratorInversion', 'RatioUniforms']


# define pdfs and other helper functions to create the generators

def argus_pdf(x, chi):
    # approach follows Baumgarten/Hoermann: Generating ARGUS random variates
    # for chi > 5, use relationship of the ARGUS distribution to Gamma(1.5)
    if chi <= 5:
        y = 1 - x * x
        return x * math.sqrt(y) * math.exp(-0.5 * chi**2 * y)
    return math.sqrt(x) * math.exp(-x)


def argus_gamma_trf(x, chi):
    if chi <= 5:
        return x
    return np.sqrt(1.0 - 2 * x / chi**2)


def argus_gamma_inv_trf(x, chi):
    if chi <= 5:
        return x
    return 0.5 * chi**2 * (1 - x**2)


def betaprime_pdf(x, a, b):
    if x > 0:
        logf = (a - 1) * math.log(x) - (a + b) * math.log1p(x) - sc.betaln(a, b)
        return math.exp(logf)
    else:
        # return pdf at x == 0 separately to avoid runtime warnings
        if a > 1:
            return 0
        elif a < 1:
            return np.inf
        else:
            return 1 / sc.beta(a, b)


def beta_valid_params(a, b):
    return (min(a, b) >= 0.1) and (max(a, b) <= 700)


def gamma_pdf(x, a):
    if x > 0:
        return math.exp(-math.lgamma(a) + (a - 1.0) * math.log(x) - x)
    else:
        return 0 if a >= 1 else np.inf


def invgamma_pdf(x, a):
    if x > 0:
        return math.exp(-(a + 1.0) * math.log(x) - math.lgamma(a) - 1 / x)
    else:
        return 0 if a >= 1 else np.inf


def burr_pdf(x, cc, dd):
    # note: we use np.exp instead of math.exp, otherwise an overflow
    # error can occur in the setup, e.g., for parameters
    # 1.89128135, 0.30195177, see test test_burr_overflow
    if x > 0:
        lx = math.log(x)
        return np.exp(-(cc + 1) * lx - (dd + 1) * math.log1p(np.exp(-cc * lx)))
    else:
        return 0


def burr12_pdf(x, cc, dd):
    if x > 0:
        lx = math.log(x)
        logterm = math.log1p(math.exp(cc * lx))
        return math.exp((cc - 1) * lx - (dd + 1) * logterm + math.log(cc * dd))
    else:
        return 0


def chi_pdf(x, a):
    if x > 0:
        return math.exp(
            (a - 1) * math.log(x)
            - 0.5 * (x * x)
            - (a / 2 - 1) * math.log(2)
            - math.lgamma(0.5 * a)
        )
    else:
        return 0 if a >= 1 else np.inf


def chi2_pdf(x, df):
    if x > 0:
        return math.exp(
            (df / 2 - 1) * math.log(x)
            - 0.5 * x
            - (df / 2) * math.log(2)
            - math.lgamma(0.5 * df)
        )
    else:
        return 0 if df >= 1 else np.inf


def alpha_pdf(x, a):
    if x > 0:
        return math.exp(-2.0 * math.log(x) - 0.5 * (a - 1.0 / x) ** 2)
    return 0.0


def bradford_pdf(x, c):
    if 0 <= x <= 1:
        return 1.0 / (1.0 + c * x)
    return 0.0


def crystalball_pdf(x, b, m):
    if x > -b:
        return math.exp(-0.5 * x * x)
    return math.exp(m * math.log(m / b) - 0.5 * b * b - m * math.log(m / b - b - x))


def weibull_min_pdf(x, c):
    if x > 0:
        return c * math.exp((c - 1) * math.log(x) - x**c)
    return 0.0


def weibull_max_pdf(x, c):
    if x < 0:
        return c * math.exp((c - 1) * math.log(-x) - ((-x) ** c))
    return 0.0


def invweibull_pdf(x, c):
    if x > 0:
        return c * math.exp(-(c + 1) * math.log(x) - x ** (-c))
    return 0.0


def wald_pdf(x):
    if x > 0:
        return math.exp(-((x - 1) ** 2) / (2 * x)) / math.sqrt(x**3)
    return 0.0


def geninvgauss_mode(p, b):
    if p > 1:  # equivalent mode formulas numerical more stable versions
        return (math.sqrt((1 - p) ** 2 + b**2) - (1 - p)) / b
    return b / (math.sqrt((1 - p) ** 2 + b**2) + (1 - p))


def geninvgauss_pdf(x, p, b):
    m = geninvgauss_mode(p, b)
    lfm = (p - 1) * math.log(m) - 0.5 * b * (m + 1 / m)
    if x > 0:
        return math.exp((p - 1) * math.log(x) - 0.5 * b * (x + 1 / x) - lfm)
    return 0.0


def invgauss_mode(mu):
    return 1.0 / (math.sqrt(1.5 * 1.5 + 1 / (mu * mu)) + 1.5)


def invgauss_pdf(x, mu):
    m = invgauss_mode(mu)
    lfm = -1.5 * math.log(m) - (m - mu) ** 2 / (2 * m * mu**2)
    if x > 0:
        return math.exp(-1.5 * math.log(x) - (x - mu) ** 2 / (2 * x * mu**2) - lfm)
    return 0.0


def powerlaw_pdf(x, a):
    if x > 0:
        return x ** (a - 1)
    return 0.0


# Define a dictionary: for a given distribution (keys), another dictionary
# (values) specifies the parameters for NumericalInversePolynomial (PINV).
# The keys of the latter dictionary are:
# - pdf: the pdf of the distribution (callable). The signature of the pdf
#   is float -> float (i.e., the function does not have to be vectorized).
#   If possible, functions like log or exp from the module math should be
#   preferred over functions from numpy since the PINV setup will be faster
#   in that case.
# - check_pinv_params: callable f that returns true if the shape parameters
#   (args) are recommended parameters for PINV (i.e., the u-error does
#   not exceed the default tolerance)
# - center: scalar if the center does not depend on args, otherwise
#   callable that returns the center as a function of the shape parameters
# - rvs_transform: a callable that can be used to transform the rvs that
#   are distributed according to the pdf to the target distribution
#   (as an example, see the entry for the beta distribution)
# - rvs_transform_inv: the inverse of rvs_transform (it is required
#   for the transformed ppf)
# - mirror_uniform: boolean or a callable that returns true or false
#   depending on the shape parameters. If True, the ppf is applied
#   to 1-u instead of u to generate rvs, where u is a uniform rv.
#   While both u and 1-u are uniform, it can be required to use 1-u
#   to compute the u-error correctly. This is only relevant for the argus
#   distribution.
# The only required keys are "pdf" and "check_pinv_params".
# All other keys are optional.

PINV_CONFIG = {
    "alpha": {
        "pdf": alpha_pdf,
        "check_pinv_params": lambda a: 1.0e-11 <= a < 2.1e5,
        "center": lambda a: 0.25 * (math.sqrt(a * a + 8.0) - a),
    },
    "anglit": {
        "pdf": lambda x: math.cos(2 * x) + 1.0e-13,
        # +1.e-13 is necessary, otherwise PINV has strange problems as
        # f(upper border) is very close to 0
        "center": 0,
    },
    "argus": {
        "pdf": argus_pdf,
        "center": lambda chi: 0.7 if chi <= 5 else 0.5,
        "check_pinv_params": lambda chi: 1e-20 < chi < 901,
        "rvs_transform": argus_gamma_trf,
        "rvs_transform_inv": argus_gamma_inv_trf,
        "mirror_uniform": lambda chi: chi > 5,
    },
    "beta": {
        "pdf": betaprime_pdf,
        "center": lambda a, b: max(0.1, (a - 1) / (b + 1)),
        "check_pinv_params": beta_valid_params,
        "rvs_transform": lambda x, *args: x / (1 + x),
        "rvs_transform_inv": lambda x, *args: x / (1 - x) if x < 1 else np.inf,
    },
    "betaprime": {
        "pdf": betaprime_pdf,
        "center": lambda a, b: max(0.1, (a - 1) / (b + 1)),
        "check_pinv_params": beta_valid_params,
    },
    "bradford": {
        "pdf": bradford_pdf,
        "check_pinv_params": lambda a: 1.0e-6 <= a <= 1e9,
        "center": 0.5,
    },
    "burr": {
        "pdf": burr_pdf,
        "center": lambda a, b: (2 ** (1 / b) - 1) ** (-1 / a),
        "check_pinv_params": lambda a, b: (min(a, b) >= 0.3) and (max(a, b) <= 50),
    },
    "burr12": {
        "pdf": burr12_pdf,
        "center": lambda a, b: (2 ** (1 / b) - 1) ** (1 / a),
        "check_pinv_params": lambda a, b: (min(a, b) >= 0.2) and (max(a, b) <= 50),
    },
    "cauchy": {
        "pdf": lambda x: 1 / (1 + (x * x)),
        "center": 0,
    },
    "chi": {
        "pdf": chi_pdf,
        "check_pinv_params": lambda df: 0.05 <= df <= 1.0e6,
        "center": lambda a: math.sqrt(a),
    },
    "chi2": {
        "pdf": chi2_pdf,
        "check_pinv_params": lambda df: 0.07 <= df <= 1e6,
        "center": lambda a: a,
    },
    "cosine": {
        "pdf": lambda x: 1 + math.cos(x),
        "center": 0,
    },
    "crystalball": {
        "pdf": crystalball_pdf,
        "check_pinv_params": lambda b, m: (0.01 <= b <= 5.5)
        and (1.1 <= m <= 75.1),
        "center": 0.0,
    },
    "expon": {
        "pdf": lambda x: math.exp(-x),
        "center": 1.0,
    },
    "gamma": {
        "pdf": gamma_pdf,
        "check_pinv_params": lambda a: 0.04 <= a <= 1e6,
        "center": lambda a: a,
    },
    "gennorm": {
        "pdf": lambda x, b: math.exp(-abs(x) ** b),
        "check_pinv_params": lambda b: 0.081 <= b <= 45.0,
        "center": 0.0,
    },
    "geninvgauss": {
        "pdf": geninvgauss_pdf,
        "check_pinv_params": lambda p, b: (abs(p) <= 1200.0)
        and (1.0e-10 <= b <= 1200.0),
        "center": geninvgauss_mode,
    },
    "gumbel_l": {
        "pdf": lambda x: math.exp(x - math.exp(x)),
        "center": -0.6,
    },
    "gumbel_r": {
        "pdf": lambda x: math.exp(-x - math.exp(-x)),
        "center": 0.6,
    },
    "hypsecant": {
        "pdf": lambda x: 1.0 / (math.exp(x) + math.exp(-x)),
        "center": 0.0,
    },
    "invgamma": {
        "pdf": invgamma_pdf,
        "check_pinv_params": lambda a: 0.04 <= a <= 1e6,
        "center": lambda a: 1 / a,
    },
    "invgauss": {
        "pdf": invgauss_pdf,
        "check_pinv_params": lambda mu: 1.0e-10 <= mu <= 1.0e9,
        "center": invgauss_mode,
    },
    "invweibull": {
        "pdf": invweibull_pdf,
        "check_pinv_params": lambda a: 0.12 <= a <= 512,
        "center": 1.0,
    },
    "laplace": {
        "pdf": lambda x: math.exp(-abs(x)),
        "center": 0.0,
    },
    "logistic": {
        "pdf": lambda x: math.exp(-x) / (1 + math.exp(-x)) ** 2,
        "center": 0.0,
    },
    "maxwell": {
        "pdf": lambda x: x * x * math.exp(-0.5 * x * x),
        "center": 1.41421,
    },
    "moyal": {
        "pdf": lambda x: math.exp(-(x + math.exp(-x)) / 2),
        "center": 1.2,
    },
    "norm": {
        "pdf": lambda x: math.exp(-x * x / 2),
        "center": 0.0,
    },
    "pareto": {
        "pdf": lambda x, b: x ** -(b + 1),
        "center": lambda b: b / (b - 1) if b > 2 else 1.5,
        "check_pinv_params": lambda b: 0.08 <= b <= 400000,
    },
    "powerlaw": {
        "pdf": powerlaw_pdf,
        "center": 1.0,
        "check_pinv_params": lambda a: 0.06 <= a <= 1.0e5,
    },
    "t": {
        "pdf": lambda x, df: (1 + x * x / df) ** (-0.5 * (df + 1)),
        "check_pinv_params": lambda a: 0.07 <= a <= 1e6,
        "center": 0.0,
    },
    "rayleigh": {
        "pdf": lambda x: x * math.exp(-0.5 * (x * x)),
        "center": 1.0,
    },
    "semicircular": {
        "pdf": lambda x: math.sqrt(1.0 - (x * x)),
        "center": 0,
    },
    "wald": {
        "pdf": wald_pdf,
        "center": 1.0,
    },
    "weibull_max": {
        "pdf": weibull_max_pdf,
        "check_pinv_params": lambda a: 0.25 <= a <= 512,
        "center": -1.0,
    },
    "weibull_min": {
        "pdf": weibull_min_pdf,
        "check_pinv_params": lambda a: 0.25 <= a <= 512,
        "center": 1.0,
    },
}


def _validate_qmc_input(qmc_engine, d, seed):
    # Input validation for `qmc_engine` and `d`
    # Error messages for invalid `d` are raised by QMCEngine
    # we could probably use a stats.qmc.check_qrandom_state
    if isinstance(qmc_engine, QMCEngine):
        if d is not None and qmc_engine.d != d:
            message = "`d` must be consistent with dimension of `qmc_engine`."
            raise ValueError(message)
        d = qmc_engine.d if d is None else d
    elif qmc_engine is None:
        d = 1 if d is None else d
        qmc_engine = Halton(d, seed=seed)
    else:
        message = (
            "`qmc_engine` must be an instance of "
            "`scipy.stats.qmc.QMCEngine` or `None`."
        )
        raise ValueError(message)

    return qmc_engine, d


class CustomDistPINV:
    def __init__(self, pdf, args):
        self._pdf = lambda x: pdf(x, *args)

    def pdf(self, x):
        return self._pdf(x)


class FastGeneratorInversion:
    """
    Fast sampling by numerical inversion of the CDF for a large class of
    continuous distributions in `scipy.stats`.

    Parameters
    ----------
    dist : rv_frozen object
        Frozen distribution object from `scipy.stats`. The list of supported
        distributions can be found in the Notes section. The shape parameters,
        `loc` and `scale` used to create the distributions must be scalars.
        For example, for the Gamma distribution with shape parameter `p`,
        `p` has to be a float, and for the beta distribution with shape
        parameters (a, b), both a and b have to be floats.
    domain : tuple of floats, optional
        If one wishes to sample from a truncated/conditional distribution,
        the domain has to be specified.
        The default is None. In that case, the random variates are not
        truncated, and the domain is inferred from the support of the
        distribution.
    ignore_shape_range : boolean, optional.
        If False, shape parameters that are outside of the valid range
        of values to ensure that the numerical accuracy (see Notes) is
        high, raise a ValueError. If True, any shape parameters that are valid
        for the distribution are accepted. This can be useful for testing.
        The default is False.
    random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional

            A NumPy random number generator or seed for the underlying NumPy
            random number generator used to generate the stream of uniform
            random numbers.
            If `random_state` is None, it uses ``self.random_state``.
            If `random_state` is an int,
            ``np.random.default_rng(random_state)`` is used.
            If `random_state` is already a ``Generator`` or ``RandomState``
            instance then that instance is used.

    Attributes
    ----------
    loc : float
        The location parameter.
    random_state : {`numpy.random.Generator`, `numpy.random.RandomState`}
        The random state used in relevant methods like `rvs` (unless
        another `random_state` is passed as an argument to these methods).
    scale : float
        The scale parameter.

    Methods
    -------
    cdf
    evaluate_error
    ppf
    qrvs
    rvs
    support

    Notes
    -----
    The class creates an object for continuous distributions specified
    by `dist`. The method `rvs` uses a generator from
    `scipy.stats.sampling` that is created when the object is instantiated.
    In addition, the methods `qrvs` and `ppf` are added.
    `qrvs` generate samples based on quasi-random numbers from
    `scipy.stats.qmc`. `ppf` is the PPF based on the
    numerical inversion method in [1]_ (`NumericalInversePolynomial`) that is
    used to generate random variates.

    Supported distributions (`distname`) are:
    ``alpha``, ``anglit``, ``argus``, ``beta``, ``betaprime``, ``bradford``,
    ``burr``, ``burr12``, ``cauchy``, ``chi``, ``chi2``, ``cosine``,
    ``crystalball``, ``expon``, ``gamma``, ``gennorm``, ``geninvgauss``,
    ``gumbel_l``, ``gumbel_r``, ``hypsecant``, ``invgamma``, ``invgauss``,
    ``invweibull``, ``laplace``, ``logistic``, ``maxwell``, ``moyal``,
    ``norm``, ``pareto``, ``powerlaw``, ``t``, ``rayleigh``, ``semicircular``,
    ``wald``, ``weibull_max``, ``weibull_min``.

    `rvs` relies on the accuracy of the numerical inversion. If very extreme
    shape parameters are used, the numerical inversion might not work. However,
    for all implemented distributions, the admissible shape parameters have
    been tested, and an error will be raised if the user supplies values
    outside of the allowed range. The u-error should not exceed 1e-10 for all
    valid parameters. Note that warnings might be raised even if parameters
    are within the valid range when the object is instantiated.
    To check numerical accuracy, the method `evaluate_error` can be used.

    Note that all implemented distributions are also part of `scipy.stats`, and
    the object created by `FastGeneratorInversion` relies on methods like
    `ppf`, `cdf` and `pdf` from `rv_frozen`. The main benefit of using this
    class can be summarized as follows: Once the generator to sample random
    variates is created in the setup step, sampling and evaluation of
    the PPF using `ppf` are very fast,
    and performance is essentially independent of the distribution. Therefore,
    a substantial speed-up can be achieved for many distributions if large
    numbers of random variates are required. It is important to know that this
    fast sampling is achieved by inversion of the CDF. Thus, one uniform
    random variate is transformed into a non-uniform variate, which is an
    advantage for several simulation methods, e.g., when
    the variance reduction methods of common random variates or
    antithetic variates are be used ([2]_).

    In addition, inversion makes it possible to
    - to use a QMC generator from `scipy.stats.qmc` (method `qrvs`),
    - to generate random variates truncated to an interval. For example, if
    one aims to sample standard normal random variates from
    the interval (2, 4), this can be easily achieved by using the parameter
    `domain`.

    The location and scale that are initially defined by `dist`
    can be reset without having to rerun the setup
    step to create the generator that is used for sampling. The relation
    of the distribution `Y` with `loc` and `scale` to the standard
    distribution `X` (i.e., ``loc=0`` and ``scale=1``) is given by
    ``Y = loc + scale * X``.

    References
    ----------
    .. [1] Derflinger, Gerhard, Wolfgang Hörmann, and Josef Leydold.
           "Random variate  generation by numerical inversion when only the
           density is known." ACM Transactions on Modeling and Computer
           Simulation (TOMACS) 20.4 (2010): 1-25.
    .. [2] Hörmann, Wolfgang, Josef Leydold and Gerhard Derflinger.
           "Automatic nonuniform random number generation."
           Springer, 2004.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> from scipy.stats.sampling import FastGeneratorInversion

    Let's start with a simple example to illustrate the main features:

    >>> gamma_frozen = stats.gamma(1.5)
    >>> gamma_dist = FastGeneratorInversion(gamma_frozen)
    >>> r = gamma_dist.rvs(size=1000)

    The mean should be approximately equal to the shape parameter 1.5:

    >>> r.mean()
    1.52423591130436  # may vary

    Similarly, we can draw a sample based on quasi-random numbers:

    >>> r = gamma_dist.qrvs(size=1000)
    >>> r.mean()
    1.4996639255942914  # may vary

    Compare the PPF against approximation `ppf`.

    >>> q = [0.001, 0.2, 0.5, 0.8, 0.999]
    >>> np.max(np.abs(gamma_frozen.ppf(q) - gamma_dist.ppf(q)))
    4.313394796895409e-08

    To confirm that the numerical inversion is accurate, we evaluate the
    approximation error (u-error), which should be below 1e-10 (for more
    details, refer to the documentation of `evaluate_error`):

    >>> gamma_dist.evaluate_error()
    (7.446320551265581e-11, nan)  # may vary

    Note that the location and scale can be changed without instantiating a
    new generator:

    >>> gamma_dist.loc = 2
    >>> gamma_dist.scale = 3
    >>> r = gamma_dist.rvs(size=1000)

    The mean should be approximately 2 + 3*1.5 = 6.5.

    >>> r.mean()
    6.399549295242894  # may vary

    Let us also illustrate how truncation can be applied:

    >>> trunc_norm = FastGeneratorInversion(stats.norm(), domain=(3, 4))
    >>> r = trunc_norm.rvs(size=1000)
    >>> 3 < r.min() < r.max() < 4
    True

    Check the mean:

    >>> r.mean()
    3.250433367078603  # may vary

    >>> stats.norm.expect(lb=3, ub=4, conditional=True)
    3.260454285589997

    In this particular, case, `scipy.stats.truncnorm` could also be used to
    generate truncated normal random variates.

    """

    def __init__(
        self,
        dist,
        *,
        domain=None,
        ignore_shape_range=False,
        random_state=None,
    ):

        if isinstance(dist, stats.distributions.rv_frozen):
            distname = dist.dist.name
            if distname not in PINV_CONFIG.keys():
                raise ValueError(
                    f"Distribution '{distname}' is not supported."
                    f"It must be one of {list(PINV_CONFIG.keys())}"
                    )
        else:
            raise ValueError("`dist` must be a frozen distribution object")

        loc = dist.kwds.get("loc", 0)
        scale = dist.kwds.get("scale", 1)
        args = dist.args
        if not np.isscalar(loc):
            raise ValueError("loc must be scalar.")
        if not np.isscalar(scale):
            raise ValueError("scale must be scalar.")

        self._frozendist = getattr(stats, distname)(
            *args,
            loc=loc,
            scale=scale,
        )
        self._distname = distname

        nargs = np.broadcast_arrays(args)[0].size
        nargs_expected = self._frozendist.dist.numargs
        if nargs != nargs_expected:
            raise ValueError(
                f"Each of the {nargs_expected} shape parameters must be a "
                f"scalar, but {nargs} values are provided."
            )

        self.random_state = random_state

        if domain is None:
            self._domain = self._frozendist.support()
            self._p_lower = 0.0
            self._p_domain = 1.0
        else:
            self._domain = domain
            self._p_lower = self._frozendist.cdf(self._domain[0])
            _p_domain = self._frozendist.cdf(self._domain[1]) - self._p_lower
            self._p_domain = _p_domain
        self._set_domain_adj()
        self._ignore_shape_range = ignore_shape_range

        # the domain to be passed to NumericalInversePolynomial
        # define a separate variable since in case of a transformation,
        # domain_pinv will not be the same as self._domain
        self._domain_pinv = self._domain

        # get information about the distribution from the config to set up
        # the generator
        dist = self._process_config(distname, args)

        if self._rvs_transform_inv is not None:
            d0 = self._rvs_transform_inv(self._domain[0], *args)
            d1 = self._rvs_transform_inv(self._domain[1], *args)
            if d0 > d1:
                # swap values if transformation if decreasing
                d0, d1 = d1, d0
            # only update _domain_pinv and not _domain
            # _domain refers to the original distribution, _domain_pinv
            # to the transformed distribution
            self._domain_pinv = d0, d1

        # self._center has been set by the call self._process_config
        # check if self._center is inside the transformed domain
        # _domain_pinv, otherwise move it to the endpoint that is closer
        if self._center is not None:
            if self._center < self._domain_pinv[0]:
                self._center = self._domain_pinv[0]
            elif self._center > self._domain_pinv[1]:
                self._center = self._domain_pinv[1]

        self._rng = NumericalInversePolynomial(
            dist,
            random_state=self.random_state,
            domain=self._domain_pinv,
            center=self._center,
            )

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, random_state):
        self._random_state = check_random_state_qmc(random_state)

    @property
    def loc(self):
        return self._frozendist.kwds.get("loc", 0)

    @loc.setter
    def loc(self, loc):
        if not np.isscalar(loc):
            raise ValueError("loc must be scalar.")
        self._frozendist.kwds["loc"] = loc
        # update the adjusted domain that depends on loc and scale
        self._set_domain_adj()

    @property
    def scale(self):
        return self._frozendist.kwds.get("scale", 0)

    @scale.setter
    def scale(self, scale):
        if not np.isscalar(scale):
            raise ValueError("scale must be scalar.")
        self._frozendist.kwds["scale"] = scale
        # update the adjusted domain that depends on loc and scale
        self._set_domain_adj()

    def _set_domain_adj(self):
        """ Adjust the domain based on loc and scale. """
        loc = self.loc
        scale = self.scale
        lb = self._domain[0] * scale + loc
        ub = self._domain[1] * scale + loc
        self._domain_adj = (lb, ub)

    def _process_config(self, distname, args):
        cfg = PINV_CONFIG[distname]
        if "check_pinv_params" in cfg:
            if not self._ignore_shape_range:
                if not cfg["check_pinv_params"](*args):
                    msg = ("No generator is defined for the shape parameters "
                           f"{args}. Use ignore_shape_range to proceed "
                           "with the selected values.")
                    raise ValueError(msg)

        if "center" in cfg.keys():
            if not np.isscalar(cfg["center"]):
                self._center = cfg["center"](*args)
            else:
                self._center = cfg["center"]
        else:
            self._center = None
        self._rvs_transform = cfg.get("rvs_transform", None)
        self._rvs_transform_inv = cfg.get("rvs_transform_inv", None)
        _mirror_uniform = cfg.get("mirror_uniform", None)
        if _mirror_uniform is None:
            self._mirror_uniform = False
        else:
            self._mirror_uniform = _mirror_uniform(*args)

        return CustomDistPINV(cfg["pdf"], args)

    def rvs(self, size=None):
        """
        Sample from the distribution by inversion.

        Parameters
        ----------
        size : int or tuple, optional
            The shape of samples. Default is ``None`` in which case a scalar
            sample is returned.

        Returns
        -------
        rvs : array_like
            A NumPy array of random variates.

        Notes
        -----
        Random variates are generated by numerical inversion of the CDF, i.e.,
        `ppf` computed by `NumericalInversePolynomial` when the class
        is instantiated. Note that the
        default ``rvs`` method of the rv_continuous class is
        overwritten. Hence, a different stream of random numbers is generated
        even if the same seed is used.
        """
        # note: we cannot use self._rng.rvs directly in case
        # self._mirror_uniform is true
        u = self.random_state.uniform(size=size)
        if self._mirror_uniform:
            u = 1 - u
        r = self._rng.ppf(u)
        if self._rvs_transform is not None:
            r = self._rvs_transform(r, *self._frozendist.args)
        return self.loc + self.scale * r

    def ppf(self, q):
        """
        Very fast PPF (inverse CDF) of the distribution which
        is a very close approximation of the exact PPF values.

        Parameters
        ----------
        u : array_like
            Array with probabilities.

        Returns
        -------
        ppf : array_like
            Quantiles corresponding to the values in `u`.

        Notes
        -----
        The evaluation of the PPF is very fast but it may have a large
        relative error in the far tails. The numerical precision of the PPF
        is controlled by the u-error, that is,
        ``max |u - CDF(PPF(u))|`` where the max is taken over points in
        the interval [0,1], see `evaluate_error`.

        Note that this PPF is designed to generate random samples.
        """
        q = np.asarray(q)
        if self._mirror_uniform:
            x = self._rng.ppf(1 - q)
        else:
            x = self._rng.ppf(q)
        if self._rvs_transform is not None:
            x = self._rvs_transform(x, *self._frozendist.args)
        return self.scale * x + self.loc

    def qrvs(self, size=None, d=None, qmc_engine=None):
        """
        Quasi-random variates of the given distribution.

        The `qmc_engine` is used to draw uniform quasi-random variates, and
        these are converted to quasi-random variates of the given distribution
        using inverse transform sampling.

        Parameters
        ----------
        size : int, tuple of ints, or None; optional
            Defines shape of random variates array. Default is ``None``.
        d : int or None, optional
            Defines dimension of uniform quasi-random variates to be
            transformed. Default is ``None``.
        qmc_engine : scipy.stats.qmc.QMCEngine(d=1), optional
            Defines the object to use for drawing
            quasi-random variates. Default is ``None``, which uses
            `scipy.stats.qmc.Halton(1)`.

        Returns
        -------
        rvs : ndarray or scalar
            Quasi-random variates. See Notes for shape information.

        Notes
        -----
        The shape of the output array depends on `size`, `d`, and `qmc_engine`.
        The intent is for the interface to be natural, but the detailed rules
        to achieve this are complicated.

        - If `qmc_engine` is ``None``, a `scipy.stats.qmc.Halton` instance is
          created with dimension `d`. If `d` is not provided, ``d=1``.
        - If `qmc_engine` is not ``None`` and `d` is ``None``, `d` is
          determined from the dimension of the `qmc_engine`.
        - If `qmc_engine` is not ``None`` and `d` is not ``None`` but the
          dimensions are inconsistent, a ``ValueError`` is raised.
        - After `d` is determined according to the rules above, the output
          shape is ``tuple_shape + d_shape``, where:

              - ``tuple_shape = tuple()`` if `size` is ``None``,
              - ``tuple_shape = (size,)`` if `size` is an ``int``,
              - ``tuple_shape = size`` if `size` is a sequence,
              - ``d_shape = tuple()`` if `d` is ``None`` or `d` is 1, and
              - ``d_shape = (d,)`` if `d` is greater than 1.

        The elements of the returned array are part of a low-discrepancy
        sequence. If `d` is 1, this means that none of the samples are truly
        independent. If `d` > 1, each slice ``rvs[..., i]`` will be of a
        quasi-independent sequence; see `scipy.stats.qmc.QMCEngine` for
        details. Note that when `d` > 1, the samples returned are still those
        of the provided univariate distribution, not a multivariate
        generalization of that distribution.

        """
        qmc_engine, d = _validate_qmc_input(qmc_engine, d, self.random_state)
        # mainly copied from unuran_wrapper.pyx.templ
        # `rvs` is flexible about whether `size` is an int or tuple, so this
        # should be, too.
        try:
            if size is None:
                tuple_size = (1,)
            else:
                tuple_size = tuple(size)
        except TypeError:
            tuple_size = (size,)
        # we do not use rng.qrvs directly since we need to be
        # able to apply the ppf to 1 - u
        N = 1 if size is None else np.prod(size)
        u = qmc_engine.random(N)
        if self._mirror_uniform:
            u = 1 - u
        qrvs = self._ppf(u)
        if self._rvs_transform is not None:
            qrvs = self._rvs_transform(qrvs, *self._frozendist.args)
        if size is None:
            qrvs = qrvs.squeeze()[()]
        else:
            if d == 1:
                qrvs = qrvs.reshape(tuple_size)
            else:
                qrvs = qrvs.reshape(tuple_size + (d,))
        return self.loc + self.scale * qrvs

    def evaluate_error(self, size=100000, random_state=None, x_error=False):
        """
        Evaluate the numerical accuracy of the inversion (u- and x-error).

        Parameters
        ----------
        size : int, optional
            The number of random points over which the error is estimated.
            Default is ``100000``.
        random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional

            A NumPy random number generator or seed for the underlying NumPy
            random number generator used to generate the stream of uniform
            random numbers.
            If `random_state` is None, use ``self.random_state``.
            If `random_state` is an int,
            ``np.random.default_rng(random_state)`` is used.
            If `random_state` is already a ``Generator`` or ``RandomState``
            instance then that instance is used.

        Returns
        -------
        u_error, x_error : tuple of floats
            A NumPy array of random variates.

        Notes
        -----
        The numerical precision of the inverse CDF `ppf` is controlled by
        the u-error. It is computed as follows:
        ``max |u - CDF(PPF(u))|`` where the max is taken `size` random
        points in the interval [0,1]. `random_state` determines the random
        sample. Note that if `ppf` was exact, the u-error would be zero.

        The x-error measures the direct distance between the exact PPF
        and `ppf`. If ``x_error`` is set to ``True`, it is
        computed as the maximum of the minimum of the relative and absolute
        x-error:
        ``max(min(x_error_abs[i], x_error_rel[i]))`` where
        ``x_error_abs[i] = |PPF(u[i]) - PPF_fast(u[i])|``,
        ``x_error_rel[i] = max |(PPF(u[i]) - PPF_fast(u[i])) / PPF(u[i])|``.
        Note that it is important to consider the relative x-error in the case
        that ``PPF(u)`` is close to zero or very large.

        By default, only the u-error is evaluated and the x-error is set to
        ``np.nan``. Note that the evaluation of the x-error will be very slow
        if the implementation of the PPF is slow.

        Further information about these error measures can be found in [1]_.

        References
        ----------
        .. [1] Derflinger, Gerhard, Wolfgang Hörmann, and Josef Leydold.
               "Random variate  generation by numerical inversion when only the
               density is known." ACM Transactions on Modeling and Computer
               Simulation (TOMACS) 20.4 (2010): 1-25.

        Examples
        --------

        >>> import numpy as np
        >>> from scipy import stats
        >>> from scipy.stats.sampling import FastGeneratorInversion

        Create an object for the normal distribution:

        >>> d_norm_frozen = stats.norm()
        >>> d_norm = FastGeneratorInversion(d_norm_frozen)

        To confirm that the numerical inversion is accurate, we evaluate the
        approximation error (u-error and x-error).

        >>> u_error, x_error = d_norm.evaluate_error(x_error=True)

        The u-error should be below 1e-10:

        >>> u_error
        8.785783212061915e-11  # may vary

        Compare the PPF against approximation `ppf`:

        >>> q = [0.001, 0.2, 0.4, 0.6, 0.8, 0.999]
        >>> diff = np.abs(d_norm_frozen.ppf(q) - d_norm.ppf(q))
        >>> x_error_abs = np.max(diff)
        >>> x_error_abs
        1.2937954707581412e-08

        This is the absolute x-error evaluated at the points q. The relative
        error is given by

        >>> x_error_rel = np.max(diff / np.abs(d_norm_frozen.ppf(q)))
        >>> x_error_rel
        4.186725600453555e-09

        The x_error computed above is derived in a very similar way over a
        much larger set of random values q. At each value q[i], the minimum
        of the relative and absolute error is taken. The final value is then
        derived as the maximum of these values. In our example, we get the
        following value:

        >>> x_error
        4.507068014335139e-07  # may vary

        """
        if not isinstance(size, (numbers.Integral, np.integer)):
            raise ValueError("size must be an integer.")
        # urng will be used to draw the samples for testing the error
        # it must not interfere with self.random_state. therefore, do not
        # call self.rvs, but draw uniform random numbers and apply
        # self.ppf (note: like in rvs, consider self._mirror_uniform)
        urng = check_random_state_qmc(random_state)
        u = urng.uniform(size=size)
        if self._mirror_uniform:
            u = 1 - u
        x = self.ppf(u)
        uerr = np.max(np.abs(self._cdf(x) - u))
        if not x_error:
            return uerr, np.nan
        ppf_u = self._ppf(u)
        x_error_abs = np.abs(self.ppf(u)-ppf_u)
        x_error_rel = x_error_abs / np.abs(ppf_u)
        x_error_combined = np.array([x_error_abs, x_error_rel]).min(axis=0)
        return uerr, np.max(x_error_combined)

    def support(self):
        """Support of the distribution.

        Returns
        -------
        a, b : float
            end-points of the distribution's support.

        Notes
        -----

        Note that the support of the distribution depends on `loc`,
        `scale` and `domain`.

        Examples
        --------

        >>> from scipy import stats
        >>> from scipy.stats.sampling import FastGeneratorInversion

        Define a truncated normal distribution:

        >>> d_norm = FastGeneratorInversion(stats.norm(), domain=(0, 1))
        >>> d_norm.support()
        (0, 1)

        Shift the distribution:

        >>> d_norm.loc = 2.5
        >>> d_norm.support()
        (2.5, 3.5)

        """
        return self._domain_adj

    def _cdf(self, x):
        """Cumulative distribution function (CDF)

        Parameters
        ----------
        x : array_like
            The values where the CDF is evaluated

        Returns
        -------
        y : ndarray
            CDF evaluated at x

        """
        y = self._frozendist.cdf(x)
        if self._p_domain == 1.0:
            return y
        return np.clip((y - self._p_lower) / self._p_domain, 0, 1)

    def _ppf(self, q):
        """Percent point function (inverse of `cdf`)

        Parameters
        ----------
        q : array_like
            lower tail probability

        Returns
        -------
        x : array_like
            quantile corresponding to the lower tail probability q.

        """
        if self._p_domain == 1.0:
            return self._frozendist.ppf(q)
        x = self._frozendist.ppf(self._p_domain * np.array(q) + self._p_lower)
        return np.clip(x, self._domain_adj[0], self._domain_adj[1])


class RatioUniforms:
    """
    Generate random samples from a probability density function using the
    ratio-of-uniforms method.

    Parameters
    ----------
    pdf : callable
        A function with signature `pdf(x)` that is proportional to the
        probability density function of the distribution.
    umax : float
        The upper bound of the bounding rectangle in the u-direction.
    vmin : float
        The lower bound of the bounding rectangle in the v-direction.
    vmax : float
        The upper bound of the bounding rectangle in the v-direction.
    c : float, optional.
        Shift parameter of ratio-of-uniforms method, see Notes. Default is 0.
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Methods
    -------
    rvs

    Notes
    -----
    Given a univariate probability density function `pdf` and a constant `c`,
    define the set ``A = {(u, v) : 0 < u <= sqrt(pdf(v/u + c))}``.
    If ``(U, V)`` is a random vector uniformly distributed over ``A``,
    then ``V/U + c`` follows a distribution according to `pdf`.

    The above result (see [1]_, [2]_) can be used to sample random variables
    using only the PDF, i.e. no inversion of the CDF is required. Typical
    choices of `c` are zero or the mode of `pdf`. The set ``A`` is a subset of
    the rectangle ``R = [0, umax] x [vmin, vmax]`` where

    - ``umax = sup sqrt(pdf(x))``
    - ``vmin = inf (x - c) sqrt(pdf(x))``
    - ``vmax = sup (x - c) sqrt(pdf(x))``

    In particular, these values are finite if `pdf` is bounded and
    ``x**2 * pdf(x)`` is bounded (i.e. subquadratic tails).
    One can generate ``(U, V)`` uniformly on ``R`` and return
    ``V/U + c`` if ``(U, V)`` are also in ``A`` which can be directly
    verified.

    The algorithm is not changed if one replaces `pdf` by k * `pdf` for any
    constant k > 0. Thus, it is often convenient to work with a function
    that is proportional to the probability density function by dropping
    unnecessary normalization factors.

    Intuitively, the method works well if ``A`` fills up most of the
    enclosing rectangle such that the probability is high that ``(U, V)``
    lies in ``A`` whenever it lies in ``R`` as the number of required
    iterations becomes too large otherwise. To be more precise, note that
    the expected number of iterations to draw ``(U, V)`` uniformly
    distributed on ``R`` such that ``(U, V)`` is also in ``A`` is given by
    the ratio ``area(R) / area(A) = 2 * umax * (vmax - vmin) / area(pdf)``,
    where `area(pdf)` is the integral of `pdf` (which is equal to one if the
    probability density function is used but can take on other values if a
    function proportional to the density is used). The equality holds since
    the area of ``A`` is equal to ``0.5 * area(pdf)`` (Theorem 7.1 in [1]_).
    If the sampling fails to generate a single random variate after 50000
    iterations (i.e. not a single draw is in ``A``), an exception is raised.

    If the bounding rectangle is not correctly specified (i.e. if it does not
    contain ``A``), the algorithm samples from a distribution different from
    the one given by `pdf`. It is therefore recommended to perform a
    test such as `~scipy.stats.kstest` as a check.

    References
    ----------
    .. [1] L. Devroye, "Non-Uniform Random Variate Generation",
       Springer-Verlag, 1986.

    .. [2] W. Hoermann and J. Leydold, "Generating generalized inverse Gaussian
       random variates", Statistics and Computing, 24(4), p. 547--557, 2014.

    .. [3] A.J. Kinderman and J.F. Monahan, "Computer Generation of Random
       Variables Using the Ratio of Uniform Deviates",
       ACM Transactions on Mathematical Software, 3(3), p. 257--260, 1977.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats

    >>> from scipy.stats.sampling import RatioUniforms
    >>> rng = np.random.default_rng()

    Simulate normally distributed random variables. It is easy to compute the
    bounding rectangle explicitly in that case. For simplicity, we drop the
    normalization factor of the density.

    >>> f = lambda x: np.exp(-x**2 / 2)
    >>> v = np.sqrt(f(np.sqrt(2))) * np.sqrt(2)
    >>> umax = np.sqrt(f(0))
    >>> gen = RatioUniforms(f, umax=umax, vmin=-v, vmax=v, random_state=rng)
    >>> r = gen.rvs(size=2500)

    The K-S test confirms that the random variates are indeed normally
    distributed (normality is not rejected at 5% significance level):

    >>> stats.kstest(r, 'norm')[1]
    0.250634764150542

    The exponential distribution provides another example where the bounding
    rectangle can be determined explicitly.

    >>> gen = RatioUniforms(lambda x: np.exp(-x), umax=1, vmin=0,
    ...                     vmax=2*np.exp(-1), random_state=rng)
    >>> r = gen.rvs(1000)
    >>> stats.kstest(r, 'expon')[1]
    0.21121052054580314

    """
    
    def __init__(self, pdf, *, umax, vmin, vmax, c=0, random_state=None):
        if vmin >= vmax:
            raise ValueError("vmin must be smaller than vmax.")

        if umax <= 0:
            raise ValueError("umax must be positive.")
        
        self._pdf = pdf
        self._umax = umax
        self._vmin = vmin
        self._vmax = vmax
        self._c = c
        self._rng = check_random_state(random_state)

    def rvs(self, size=1):
        """Sampling of random variates

        Parameters
        ----------
        size : int or tuple of ints, optional
            Number of random variates to be generated (default is 1).

        Returns
        -------
        rvs : ndarray
            The random variates distributed according to the probability
            distribution defined by the pdf.

        """
        size1d = tuple(np.atleast_1d(size))
        N = np.prod(size1d)  # number of rvs needed, reshape upon return

        # start sampling using ratio of uniforms method
        x = np.zeros(N)
        simulated, i = 0, 1

        # loop until N rvs have been generated: expected runtime is finite.
        # to avoid infinite loop, raise exception if not a single rv has been
        # generated after 50000 tries. even if the expected number of iterations
        # is 1000, the probability of this event is (1-1/1000)**50000
        # which is of order 10e-22
        while simulated < N:
            k = N - simulated
            # simulate uniform rvs on [0, umax] and [vmin, vmax]
            u1 = self._umax * self._rng.uniform(size=k)
            v1 = self._rng.uniform(self._vmin, self._vmax, size=k)
            # apply rejection method
            rvs = v1 / u1 + self._c
            accept = (u1**2 <= self._pdf(rvs))
            num_accept = np.sum(accept)
            if num_accept > 0:
                x[simulated:(simulated + num_accept)] = rvs[accept]
                simulated += num_accept

            if (simulated == 0) and (i*N >= 50000):
                msg = (
                    f"Not a single random variate could be generated in {i*N} "
                    "attempts. The ratio of uniforms method does not appear "
                    "to work for the provided parameters. Please check the "
                    "pdf and the bounds."
                )
                raise RuntimeError(msg)
            i += 1

        return np.reshape(x, size1d)
<|endoftext|>import operator
from dataclasses import dataclass
import numpy as np
from scipy.special import ndtri
from ._common import ConfidenceInterval


def _validate_int(n, bound, name):
    msg = f'{name} must be an integer not less than {bound}, but got {n!r}'
    try:
        n = operator.index(n)
    except TypeError:
        raise TypeError(msg) from None
    if n < bound:
        raise ValueError(msg)
    return n


@dataclass
class RelativeRiskResult:
    """
    Result of `scipy.stats.contingency.relative_risk`.

    Attributes
    ----------
    relative_risk : float
        This is::

            (exposed_cases/exposed_total) / (control_cases/control_total)

    exposed_cases : int
        The number of "cases" (i.e. occurrence of disease or other event
        of interest) among the sample of "exposed" individuals.
    exposed_total : int
        The total number of "exposed" individuals in the sample.
    control_cases : int
        The number of "cases" among the sample of "control" or non-exposed
        individuals.
    control_total : int
        The total number of "control" individuals in the sample.

    Methods
    -------
    confidence_interval :
        Compute the confidence interval for the relative risk estimate.
    """

    relative_risk: float
    exposed_cases: int
    exposed_total: int
    control_cases: int
    control_total: int

    def confidence_interval(self, confidence_level=0.95):
        """
        Compute the confidence interval for the relative risk.

        The confidence interval is computed using the Katz method
        (i.e. "Method C" of [1]_; see also [2]_, section 3.1.2).

        Parameters
        ----------
        confidence_level : float, optional
            The confidence level to use for the confidence interval.
            Default is 0.95.

        Returns
        -------
        ci : ConfidenceInterval instance
            The return value is an object with attributes ``low`` and
            ``high`` that hold the confidence interval.

        References
        ----------
        .. [1] D. Katz, J. Baptista, S. P. Azen and M. C. Pike, "Obtaining
               confidence intervals for the risk ratio in cohort studies",
               Biometrics, 34, 469-474 (1978).
        .. [2] Hardeo Sahai and Anwer Khurshid, Statistics in Epidemiology,
               CRC Press LLC, Boca Raton, FL, USA (1996).


        Examples
        --------
        >>> from scipy.stats.contingency import relative_risk
        >>> result = relative_risk(exposed_cases=10, exposed_total=75,
        ...                        control_cases=12, control_total=225)
        >>> result.relative_risk
        2.5
        >>> result.confidence_interval()
        ConfidenceInterval(low=1.1261564003469628, high=5.549850800541033)
        """
        if not 0 <= confidence_level <= 1:
            raise ValueError('confidence_level must be in the interval '
                             '[0, 1].')

        # Handle edge cases where either exposed_cases or control_cases
        # is zero.  We follow the convention of the R function riskratio
        # from the epitools library.
        if self.exposed_cases == 0 and self.control_cases == 0:
            # relative risk is nan.
            return ConfidenceInterval(low=np.nan, high=np.nan)
        elif self.exposed_cases == 0:
            # relative risk is 0.
            return ConfidenceInterval(low=0.0, high=np.nan)
        elif self.control_cases == 0:
            # relative risk is inf
            return ConfidenceInterval(low=np.nan, high=np.inf)

        alpha = 1 - confidence_level
        z = ndtri(1 - alpha/2)
        rr = self.relative_risk

        # Estimate of the variance of log(rr) is
        # var(log(rr)) = 1/exposed_cases - 1/exposed_total +
        #                1/control_cases - 1/control_total
        # and the standard error is the square root of that.
        se = np.sqrt(1/self.exposed_cases - 1/self.exposed_total +
                     1/self.control_cases - 1/self.control_total)
        delta = z*se
        katz_lo = rr*np.exp(-delta)
        katz_hi = rr*np.exp(delta)
        return ConfidenceInterval(low=katz_lo, high=katz_hi)


def relative_risk(exposed_cases, exposed_total, control_cases, control_total):
    """
    Compute the relative risk (also known as the risk ratio).

    This function computes the relative risk associated with a 2x2
    contingency table ([1]_, section 2.2.3; [2]_, section 3.1.2). Instead
    of accepting a table as an argument, the individual numbers that are
    used to compute the relative risk are given as separate parameters.
    This is to avoid the ambiguity of which row or column of the contingency
    table corresponds to the "exposed" cases and which corresponds to the
    "control" cases.  Unlike, say, the odds ratio, the relative risk is not
    invariant under an interchange of the rows or columns.

    Parameters
    ----------
    exposed_cases : nonnegative int
        The number of "cases" (i.e. occurrence of disease or other event
        of interest) among the sample of "exposed" individuals.
    exposed_total : positive int
        The total number of "exposed" individuals in the sample.
    control_cases : nonnegative int
        The number of "cases" among the sample of "control" or non-exposed
        individuals.
    control_total : positive int
        The total number of "control" individuals in the sample.

    Returns
    -------
    result : instance of `~scipy.stats._result_classes.RelativeRiskResult`
        The object has the float attribute ``relative_risk``, which is::

            rr = (exposed_cases/exposed_total) / (control_cases/control_total)

        The object also has the method ``confidence_interval`` to compute
        the confidence interval of the relative risk for a given confidence
        level.

    See Also
    --------
    odds_ratio

    Notes
    -----
    The R package epitools has the function `riskratio`, which accepts
    a table with the following layout::

                        disease=0   disease=1
        exposed=0 (ref)    n00         n01
        exposed=1          n10         n11

    With a 2x2 table in the above format, the estimate of the CI is
    computed by `riskratio` when the argument method="wald" is given,
    or with the function `riskratio.wald`.

    For example, in a test of the incidence of lung cancer among a
    sample of smokers and nonsmokers, the "exposed" category would
    correspond to "is a smoker" and the "disease" category would
    correspond to "has or had lung cancer".

    To pass the same data to ``relative_risk``, use::

        relative_risk(n11, n10 + n11, n01, n00 + n01)

    .. versionadded:: 1.7.0

    References
    ----------
    .. [1] Alan Agresti, An Introduction to Categorical Data Analysis
           (second edition), Wiley, Hoboken, NJ, USA (2007).
    .. [2] Hardeo Sahai and Anwer Khurshid, Statistics in Epidemiology,
           CRC Press LLC, Boca Raton, FL, USA (1996).

    Examples
    --------
    >>> from scipy.stats.contingency import relative_risk

    This example is from Example 3.1 of [2]_.  The results of a heart
    disease study are summarized in the following table::

                 High CAT   Low CAT    Total
                 --------   -------    -----
        CHD         27         44        71
        No CHD      95        443       538

        Total      122        487       609

    CHD is coronary heart disease, and CAT refers to the level of
    circulating catecholamine.  CAT is the "exposure" variable, and
    high CAT is the "exposed" category. So the data from the table
    to be passed to ``relative_risk`` is::

        exposed_cases = 27
        exposed_total = 122
        control_cases = 44
        control_total = 487

    >>> result = relative_risk(27, 122, 44, 487)
    >>> result.relative_risk
    2.4495156482861398

    Find the confidence interval for the relative risk.

    >>> result.confidence_interval(confidence_level=0.95)
    ConfidenceInterval(low=1.5836990926700116, high=3.7886786315466354)

    The interval does not contain 1, so the data supports the statement
    that high CAT is associated with greater risk of CHD.
    """
    # Relative risk is a trivial calculation.  The nontrivial part is in the
    # `confidence_interval` method of the RelativeRiskResult class.

    exposed_cases = _validate_int(exposed_cases, 0, "exposed_cases")
    exposed_total = _validate_int(exposed_total, 1, "exposed_total")
    control_cases = _validate_int(control_cases, 0, "control_cases")
    control_total = _validate_int(control_total, 1, "control_total")

    if exposed_cases > exposed_total:
        raise ValueError('exposed_cases must not exceed exposed_total.')
    if control_cases > control_total:
        raise ValueError('control_cases must not exceed control_total.')

    if exposed_cases == 0 and control_cases == 0:
        # relative risk is 0/0.
        rr = np.nan
    elif exposed_cases == 0:
        # relative risk is 0/nonzero
        rr = 0.0
    elif control_cases == 0:
        # relative risk is nonzero/0.
        rr = np.inf
    else:
        p1 = exposed_cases / exposed_total
        p2 = control_cases / control_total
        rr = p1 / p2
    return RelativeRiskResult(relative_risk=rr,
                              exposed_cases=exposed_cases,
                              exposed_total=exposed_total,
                              control_cases=control_cases,
                              control_total=control_total)
<|endoftext|>#
# Author:  Travis Oliphant  2002-2011 with contributions from
#          SciPy Developers 2004-2011
#
from scipy._lib._util import getfullargspec_no_self as _getfullargspec

import sys
import keyword
import re
import types
import warnings
from itertools import zip_longest

from scipy._lib import doccer
from ._distr_params import distcont, distdiscrete
from scipy._lib._util import check_random_state, _lazywhere

from scipy.special import comb, entr


# for root finding for continuous distribution ppf, and maximum likelihood
# estimation
from scipy import optimize

# for functions of continuous distributions (e.g. moments, entropy, cdf)
from scipy import integrate

# to approximate the pdf of a continuous distribution given its cdf
from scipy._lib._finite_differences import _derivative

# for scipy.stats.entropy. Attempts to import just that function or file
# have cause import problems
from scipy import stats

from numpy import (arange, putmask, ones, shape, ndarray, zeros, floor,
                   logical_and, log, sqrt, place, argmax, vectorize, asarray,
                   nan, inf, isinf, empty)

import numpy as np
from ._constants import _XMAX, _LOGXMAX
from ._censored_data import CensoredData
from scipy.stats._warnings_errors import FitError

# These are the docstring parts used for substitution in specific
# distribution docstrings

docheaders = {'methods': """\nMethods\n-------\n""",
              'notes': """\nNotes\n-----\n""",
              'examples': """\nExamples\n--------\n"""}

_doc_rvs = """\
rvs(%(shapes)s, loc=0, scale=1, size=1, random_state=None)
    Random variates.
"""
_doc_pdf = """\
pdf(x, %(shapes)s, loc=0, scale=1)
    Probability density function.
"""
_doc_logpdf = """\
logpdf(x, %(shapes)s, loc=0, scale=1)
    Log of the probability density function.
"""
_doc_pmf = """\
pmf(k, %(shapes)s, loc=0, scale=1)
    Probability mass function.
"""
_doc_logpmf = """\
logpmf(k, %(shapes)s, loc=0, scale=1)
    Log of the probability mass function.
"""
_doc_cdf = """\
cdf(x, %(shapes)s, loc=0, scale=1)
    Cumulative distribution function.
"""
_doc_logcdf = """\
logcdf(x, %(shapes)s, loc=0, scale=1)
    Log of the cumulative distribution function.
"""
_doc_sf = """\
sf(x, %(shapes)s, loc=0, scale=1)
    Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).
"""  # noqa: E501
_doc_logsf = """\
logsf(x, %(shapes)s, loc=0, scale=1)
    Log of the survival function.
"""
_doc_ppf = """\
ppf(q, %(shapes)s, loc=0, scale=1)
    Percent point function (inverse of ``cdf`` --- percentiles).
"""
_doc_isf = """\
isf(q, %(shapes)s, loc=0, scale=1)
    Inverse survival function (inverse of ``sf``).
"""
_doc_moment = """\
moment(order, %(shapes)s, loc=0, scale=1)
    Non-central moment of the specified order.
"""
_doc_stats = """\
stats(%(shapes)s, loc=0, scale=1, moments='mv')
    Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').
"""
_doc_entropy = """\
entropy(%(shapes)s, loc=0, scale=1)
    (Differential) entropy of the RV.
"""
_doc_fit = """\
fit(data)
    Parameter estimates for generic data.
    See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the
    keyword arguments.
"""  # noqa: E501
_doc_expect = """\
expect(func, args=(%(shapes_)s), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)
    Expected value of a function (of one argument) with respect to the distribution.
"""  # noqa: E501
_doc_expect_discrete = """\
expect(func, args=(%(shapes_)s), loc=0, lb=None, ub=None, conditional=False)
    Expected value of a function (of one argument) with respect to the distribution.
"""
_doc_median = """\
median(%(shapes)s, loc=0, scale=1)
    Median of the distribution.
"""
_doc_mean = """\
mean(%(shapes)s, loc=0, scale=1)
    Mean of the distribution.
"""
_doc_var = """\
var(%(shapes)s, loc=0, scale=1)
    Variance of the distribution.
"""
_doc_std = """\
std(%(shapes)s, loc=0, scale=1)
    Standard deviation of the distribution.
"""
_doc_interval = """\
interval(confidence, %(shapes)s, loc=0, scale=1)
    Confidence interval with equal areas around the median.
"""
_doc_allmethods = ''.join([docheaders['methods'], _doc_rvs, _doc_pdf,
                           _doc_logpdf, _doc_cdf, _doc_logcdf, _doc_sf,
                           _doc_logsf, _doc_ppf, _doc_isf, _doc_moment,
                           _doc_stats, _doc_entropy, _doc_fit,
                           _doc_expect, _doc_median,
                           _doc_mean, _doc_var, _doc_std, _doc_interval])

_doc_default_longsummary = """\
As an instance of the `rv_continuous` class, `%(name)s` object inherits from it
a collection of generic methods (see below for the full list),
and completes them with details specific for this particular distribution.
"""

_doc_default_frozen_note = """
Alternatively, the object may be called (as a function) to fix the shape,
location, and scale parameters returning a "frozen" continuous RV object:

rv = %(name)s(%(shapes)s, loc=0, scale=1)
    - Frozen RV object with the same methods but holding the given shape,
      location, and scale fixed.
"""
_doc_default_example = """\
Examples
--------
>>> import numpy as np
>>> from scipy.stats import %(name)s
>>> import matplotlib.pyplot as plt
>>> fig, ax = plt.subplots(1, 1)

Calculate the first four moments:

%(set_vals_stmt)s
>>> mean, var, skew, kurt = %(name)s.stats(%(shapes)s, moments='mvsk')

Display the probability density function (``pdf``):

>>> x = np.linspace(%(name)s.ppf(0.01, %(shapes)s),
...                 %(name)s.ppf(0.99, %(shapes)s), 100)
>>> ax.plot(x, %(name)s.pdf(x, %(shapes)s),
...        'r-', lw=5, alpha=0.6, label='%(name)s pdf')

Alternatively, the distribution object can be called (as a function)
to fix the shape, location and scale parameters. This returns a "frozen"
RV object holding the given parameters fixed.

Freeze the distribution and display the frozen ``pdf``:

>>> rv = %(name)s(%(shapes)s)
>>> ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

Check accuracy of ``cdf`` and ``ppf``:

>>> vals = %(name)s.ppf([0.001, 0.5, 0.999], %(shapes)s)
>>> np.allclose([0.001, 0.5, 0.999], %(name)s.cdf(vals, %(shapes)s))
True

Generate random numbers:

>>> r = %(name)s.rvs(%(shapes)s, size=1000)

And compare the histogram:

>>> ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
>>> ax.set_xlim([x[0], x[-1]])
>>> ax.legend(loc='best', frameon=False)
>>> plt.show()

"""

_doc_default_locscale = """\
The probability density above is defined in the "standardized" form. To shift
and/or scale the distribution use the ``loc`` and ``scale`` parameters.
Specifically, ``%(name)s.pdf(x, %(shapes)s, loc, scale)`` is identically
equivalent to ``%(name)s.pdf(y, %(shapes)s) / scale`` with
``y = (x - loc) / scale``. Note that shifting the location of a distribution
does not make it a "noncentral" distribution; noncentral generalizations of
some distributions are available in separate classes.
"""

_doc_default = ''.join([_doc_default_longsummary,
                        _doc_allmethods,
                        '\n',
                        _doc_default_example])

_doc_default_before_notes = ''.join([_doc_default_longsummary,
                                     _doc_allmethods])

docdict = {
    'rvs': _doc_rvs,
    'pdf': _doc_pdf,
    'logpdf': _doc_logpdf,
    'cdf': _doc_cdf,
    'logcdf': _doc_logcdf,
    'sf': _doc_sf,
    'logsf': _doc_logsf,
    'ppf': _doc_ppf,
    'isf': _doc_isf,
    'stats': _doc_stats,
    'entropy': _doc_entropy,
    'fit': _doc_fit,
    'moment': _doc_moment,
    'expect': _doc_expect,
    'interval': _doc_interval,
    'mean': _doc_mean,
    'std': _doc_std,
    'var': _doc_var,
    'median': _doc_median,
    'allmethods': _doc_allmethods,
    'longsummary': _doc_default_longsummary,
    'frozennote': _doc_default_frozen_note,
    'example': _doc_default_example,
    'default': _doc_default,
    'before_notes': _doc_default_before_notes,
    'after_notes': _doc_default_locscale
}

# Reuse common content between continuous and discrete docs, change some
# minor bits.
docdict_discrete = docdict.copy()

docdict_discrete['pmf'] = _doc_pmf
docdict_discrete['logpmf'] = _doc_logpmf
docdict_discrete['expect'] = _doc_expect_discrete
_doc_disc_methods = ['rvs', 'pmf', 'logpmf', 'cdf', 'logcdf', 'sf', 'logsf',
                     'ppf', 'isf', 'stats', 'entropy', 'expect', 'median',
                     'mean', 'var', 'std', 'interval']
for obj in _doc_disc_methods:
    docdict_discrete[obj] = docdict_discrete[obj].replace(', scale=1', '')

_doc_disc_methods_err_varname = ['cdf', 'logcdf', 'sf', 'logsf']
for obj in _doc_disc_methods_err_varname:
    docdict_discrete[obj] = docdict_discrete[obj].replace('(x, ', '(k, ')

docdict_discrete.pop('pdf')
docdict_discrete.pop('logpdf')

_doc_allmethods = ''.join([docdict_discrete[obj] for obj in _doc_disc_methods])
docdict_discrete['allmethods'] = docheaders['methods'] + _doc_allmethods

docdict_discrete['longsummary'] = _doc_default_longsummary.replace(
    'rv_continuous', 'rv_discrete')

_doc_default_frozen_note = """
Alternatively, the object may be called (as a function) to fix the shape and
location parameters returning a "frozen" discrete RV object:

rv = %(name)s(%(shapes)s, loc=0)
    - Frozen RV object with the same methods but holding the given shape and
      location fixed.
"""
docdict_discrete['frozennote'] = _doc_default_frozen_note

_doc_default_discrete_example = """\
Examples
--------
>>> import numpy as np
>>> from scipy.stats import %(name)s
>>> import matplotlib.pyplot as plt
>>> fig, ax = plt.subplots(1, 1)

Calculate the first four moments:

%(set_vals_stmt)s
>>> mean, var, skew, kurt = %(name)s.stats(%(shapes)s, moments='mvsk')

Display the probability mass function (``pmf``):

>>> x = np.arange(%(name)s.ppf(0.01, %(shapes)s),
...               %(name)s.ppf(0.99, %(shapes)s))
>>> ax.plot(x, %(name)s.pmf(x, %(shapes)s), 'bo', ms=8, label='%(name)s pmf')
>>> ax.vlines(x, 0, %(name)s.pmf(x, %(shapes)s), colors='b', lw=5, alpha=0.5)

Alternatively, the distribution object can be called (as a function)
to fix the shape and location. This returns a "frozen" RV object holding
the given parameters fixed.

Freeze the distribution and display the frozen ``pmf``:

>>> rv = %(name)s(%(shapes)s)
>>> ax.vlines(x, 0, rv.pmf(x), colors='k', linestyles='-', lw=1,
...         label='frozen pmf')
>>> ax.legend(loc='best', frameon=False)
>>> plt.show()

Check accuracy of ``cdf`` and ``ppf``:

>>> prob = %(name)s.cdf(x, %(shapes)s)
>>> np.allclose(x, %(name)s.ppf(prob, %(shapes)s))
True

Generate random numbers:

>>> r = %(name)s.rvs(%(shapes)s, size=1000)
"""


_doc_default_discrete_locscale = """\
The probability mass function above is defined in the "standardized" form.
To shift distribution use the ``loc`` parameter.
Specifically, ``%(name)s.pmf(k, %(shapes)s, loc)`` is identically
equivalent to ``%(name)s.pmf(k - loc, %(shapes)s)``.
"""

docdict_discrete['example'] = _doc_default_discrete_example
docdict_discrete['after_notes'] = _doc_default_discrete_locscale

_doc_default_before_notes = ''.join([docdict_discrete['longsummary'],
                                     docdict_discrete['allmethods']])
docdict_discrete['before_notes'] = _doc_default_before_notes

_doc_default_disc = ''.join([docdict_discrete['longsummary'],
                             docdict_discrete['allmethods'],
                             docdict_discrete['frozennote'],
                             docdict_discrete['example']])
docdict_discrete['default'] = _doc_default_disc

# clean up all the separate docstring elements, we do not need them anymore
for obj in [s for s in dir() if s.startswith('_doc_')]:
    exec('del ' + obj)
del obj


def _moment(data, n, mu=None):
    if mu is None:
        mu = data.mean()
    return ((data - mu)**n).mean()


def _moment_from_stats(n, mu, mu2, g1, g2, moment_func, args):
    if (n == 0):
        return 1.0
    elif (n == 1):
        if mu is None:
            val = moment_func(1, *args)
        else:
            val = mu
    elif (n == 2):
        if mu2 is None or mu is None:
            val = moment_func(2, *args)
        else:
            val = mu2 + mu*mu
    elif (n == 3):
        if g1 is None or mu2 is None or mu is None:
            val = moment_func(3, *args)
        else:
            mu3 = g1 * np.power(mu2, 1.5)  # 3rd central moment
            val = mu3+3*mu*mu2+mu*mu*mu  # 3rd non-central moment
    elif (n == 4):
        if g1 is None or g2 is None or mu2 is None or mu is None:
            val = moment_func(4, *args)
        else:
            mu4 = (g2+3.0)*(mu2**2.0)  # 4th central moment
            mu3 = g1*np.power(mu2, 1.5)  # 3rd central moment
            val = mu4+4*mu*mu3+6*mu*mu*mu2+mu*mu*mu*mu
    else:
        val = moment_func(n, *args)

    return val


def _skew(data):
    """
    skew is third central moment / variance**(1.5)
    """
    data = np.ravel(data)
    mu = data.mean()
    m2 = ((data - mu)**2).mean()
    m3 = ((data - mu)**3).mean()
    return m3 / np.power(m2, 1.5)


def _kurtosis(data):
    """Fisher's excess kurtosis is fourth central moment / variance**2 - 3."""
    data = np.ravel(data)
    mu = data.mean()
    m2 = ((data - mu)**2).mean()
    m4 = ((data - mu)**4).mean()
    return m4 / m2**2 - 3

def _vectorize_rvs_over_shapes(_rvs1):
    """Decorator that vectorizes _rvs method to work on ndarray shapes"""
    # _rvs1 must be a _function_ that accepts _scalar_ args as positional
    # arguments, `size` and `random_state` as keyword arguments.
    # _rvs1 must return a random variate array with shape `size`. If `size` is
    # None, _rvs1 must return a scalar.
    # When applied to _rvs1, this decorator broadcasts ndarray args
    # and loops over them, calling _rvs1 for each set of scalar args.
    # For usage example, see _nchypergeom_gen
    def _rvs(*args, size, random_state):
        _rvs1_size, _rvs1_indices = _check_shape(args[0].shape, size)

        size = np.array(size)
        _rvs1_size = np.array(_rvs1_size)
        _rvs1_indices = np.array(_rvs1_indices)

        if np.all(_rvs1_indices):  # all args are scalars
            return _rvs1(*args, size, random_state)

        out = np.empty(size)

        # out.shape can mix dimensions associated with arg_shape and _rvs1_size
        # Sort them to arg_shape + _rvs1_size for easy indexing of dimensions
        # corresponding with the different sets of scalar args
        j0 = np.arange(out.ndim)
        j1 = np.hstack((j0[~_rvs1_indices], j0[_rvs1_indices]))
        out = np.moveaxis(out, j1, j0)

        for i in np.ndindex(*size[~_rvs1_indices]):
            # arg can be squeezed because singleton dimensions will be
            # associated with _rvs1_size, not arg_shape per _check_shape
            out[i] = _rvs1(*[np.squeeze(arg)[i] for arg in args],
                           _rvs1_size, random_state)

        return np.moveaxis(out, j0, j1)  # move axes back before returning
    return _rvs


def _fit_determine_optimizer(optimizer):
    if not callable(optimizer) and isinstance(optimizer, str):
        if not optimizer.startswith('fmin_'):
            optimizer = "fmin_"+optimizer
        if optimizer == 'fmin_':
            optimizer = 'fmin'
        try:
            optimizer = getattr(optimize, optimizer)
        except AttributeError as e:
            raise ValueError(f"{optimizer} is not a valid optimizer") from e
    return optimizer

def _isintegral(x):
    return x == np.round(x)

def _sum_finite(x):
    """
    For a 1D array x, return a tuple containing the sum of the
    finite values of x and the number of nonfinite values.

    This is a utility function used when evaluating the negative
    loglikelihood for a distribution and an array of samples.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats._distn_infrastructure import _sum_finite
    >>> tot, nbad = _sum_finite(np.array([-2, -np.inf, 5, 1]))
    >>> tot
    4.0
    >>> nbad
    1
    """
    finite_x = np.isfinite(x)
    bad_count = finite_x.size - np.count_nonzero(finite_x)
    return np.sum(x[finite_x]), bad_count


# Frozen RV class
class rv_frozen:

    def __init__(self, dist, *args, **kwds):
        self.args = args
        self.kwds = kwds

        # create a new instance
        self.dist = dist.__class__(**dist._updated_ctor_param())

        shapes, _, _ = self.dist._parse_args(*args, **kwds)
        self.a, self.b = self.dist._get_support(*shapes)

    @property
    def random_state(self):
        return self.dist._random_state

    @random_state.setter
    def random_state(self, seed):
        self.dist._random_state = check_random_state(seed)

    def cdf(self, x):
        return self.dist.cdf(x, *self.args, **self.kwds)

    def logcdf(self, x):
        return self.dist.logcdf(x, *self.args, **self.kwds)

    def ppf(self, q):
        return self.dist.ppf(q, *self.args, **self.kwds)

    def isf(self, q):
        return self.dist.isf(q, *self.args, **self.kwds)

    def rvs(self, size=None, random_state=None):
        kwds = self.kwds.copy()
        kwds.update({'size': size, 'random_state': random_state})
        return self.dist.rvs(*self.args, **kwds)

    def sf(self, x):
        return self.dist.sf(x, *self.args, **self.kwds)

    def logsf(self, x):
        return self.dist.logsf(x, *self.args, **self.kwds)

    def stats(self, moments='mv'):
        kwds = self.kwds.copy()
        kwds.update({'moments': moments})
        return self.dist.stats(*self.args, **kwds)

    def median(self):
        return self.dist.median(*self.args, **self.kwds)

    def mean(self):
        return self.dist.mean(*self.args, **self.kwds)

    def var(self):
        return self.dist.var(*self.args, **self.kwds)

    def std(self):
        return self.dist.std(*self.args, **self.kwds)

    def moment(self, order=None):
        return self.dist.moment(order, *self.args, **self.kwds)

    def entropy(self):
        return self.dist.entropy(*self.args, **self.kwds)

    def interval(self, confidence=None):
        return self.dist.interval(confidence, *self.args, **self.kwds)

    def expect(self, func=None, lb=None, ub=None, conditional=False, **kwds):
        # expect method only accepts shape parameters as positional args
        # hence convert self.args, self.kwds, also loc/scale
        # See the .expect method docstrings for the meaning of
        # other parameters.
        a, loc, scale = self.dist._parse_args(*self.args, **self.kwds)
        if isinstance(self.dist, rv_discrete):
            return self.dist.expect(func, a, loc, lb, ub, conditional, **kwds)
        else:
            return self.dist.expect(func, a, loc, scale, lb, ub,
                                    conditional, **kwds)

    def support(self):
        return self.dist.support(*self.args, **self.kwds)


class rv_discrete_frozen(rv_frozen):

    def pmf(self, k):
        return self.dist.pmf(k, *self.args, **self.kwds)

    def logpmf(self, k):  # No error
        return self.dist.logpmf(k, *self.args, **self.kwds)


class rv_continuous_frozen(rv_frozen):

    def pdf(self, x):
        return self.dist.pdf(x, *self.args, **self.kwds)

    def logpdf(self, x):
        return self.dist.logpdf(x, *self.args, **self.kwds)


def argsreduce(cond, *args):
    """Clean arguments to:

    1. Ensure all arguments are iterable (arrays of dimension at least one
    2. If cond != True and size > 1, ravel(args[i]) where ravel(condition) is
       True, in 1D.

    Return list of processed arguments.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats._distn_infrastructure import argsreduce
    >>> rng = np.random.default_rng()
    >>> A = rng.random((4, 5))
    >>> B = 2
    >>> C = rng.random((1, 5))
    >>> cond = np.ones(A.shape)
    >>> [A1, B1, C1] = argsreduce(cond, A, B, C)
    >>> A1.shape
    (4, 5)
    >>> B1.shape
    (1,)
    >>> C1.shape
    (1, 5)
    >>> cond[2,:] = 0
    >>> [A1, B1, C1] = argsreduce(cond, A, B, C)
    >>> A1.shape
    (15,)
    >>> B1.shape
    (1,)
    >>> C1.shape
    (15,)

    """
    # some distributions assume arguments are iterable.
    newargs = np.atleast_1d(*args)

    # np.atleast_1d returns an array if only one argument, or a list of arrays
    # if more than one argument.
    if not isinstance(newargs, (list | tuple)):
        newargs = (newargs,)

    if np.all(cond):
        # broadcast arrays with cond
        *newargs, cond = np.broadcast_arrays(*newargs, cond)
        return [arg.ravel() for arg in newargs]

    s = cond.shape
    # np.extract returns flattened arrays, which are not broadcastable together
    # unless they are either the same size or size == 1.
    return [(arg if np.size(arg) == 1
            else np.extract(cond, np.broadcast_to(arg, s)))
            for arg in newargs]


parse_arg_template = """
def _parse_args(self, %(shape_arg_str)s %(locscale_in)s):
    return (%(shape_arg_str)s), %(locscale_out)s

def _parse_args_rvs(self, %(shape_arg_str)s %(locscale_in)s, size=None):
    return self._argcheck_rvs(%(shape_arg_str)s %(locscale_out)s, size=size)

def _parse_args_stats(self, %(shape_arg_str)s %(locscale_in)s, moments='mv'):
    return (%(shape_arg_str)s), %(locscale_out)s, moments
"""


class rv_generic:
    """Class which encapsulates common functionality between rv_discrete
    and rv_continuous.

    """

    def __init__(self, seed=None):
        super().__init__()

        # figure out if _stats signature has 'moments' keyword
        sig = _getfullargspec(self._stats)
        self._stats_has_moments = ((sig.varkw is not None) or
                                   ('moments' in sig.args) or
                                   ('moments' in sig.kwonlyargs))
        self._random_state = check_random_state(seed)

    @property
    def random_state(self):
        """Get or set the generator object for generating random variates.

        If `random_state` is None (or `np.random`), the
        `numpy.random.RandomState` singleton is used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``Generator`` or ``RandomState``
        instance, that instance is used.

        """
        return self._random_state

    @random_state.setter
    def random_state(self, seed):
        self._random_state = check_random_state(seed)

    def __setstate__(self, state):
        try:
            self.__dict__.update(state)
            # attaches the dynamically created methods on each instance.
            # if a subclass overrides rv_generic.__setstate__, or implements
            # it's own _attach_methods, then it must make sure that
            # _attach_argparser_methods is called.
            self._attach_methods()
        except ValueError:
            # reconstitute an old pickle scipy<1.6, that contains
            # (_ctor_param, random_state) as state
            self._ctor_param = state[0]
            self._random_state = state[1]
            self.__init__()

    def _attach_methods(self):
        """Attaches dynamically created methods to the rv_* instance.

        This method must be overridden by subclasses, and must itself call
         _attach_argparser_methods. This method is called in __init__ in
         subclasses, and in __setstate__
        """
        raise NotImplementedError

    def _attach_argparser_methods(self):
        """
        Generates the argument-parsing functions dynamically and attaches
        them to the instance.

        Should be called from `_attach_methods`, typically in __init__ and
        during unpickling (__setstate__)
        """
        ns = {}
        exec(self._parse_arg_template, ns)
        # NB: attach to the instance, not class
        for name in ['_parse_args', '_parse_args_stats', '_parse_args_rvs']:
            setattr(self, name, types.MethodType(ns[name], self))

    def _construct_argparser(
            self, meths_to_inspect, locscale_in, locscale_out):
        """Construct the parser string for the shape arguments.

        This method should be called in __init__ of a class for each
        distribution. It creates the `_parse_arg_template` attribute that is
        then used by `_attach_argparser_methods` to dynamically create and
        attach the `_parse_args`, `_parse_args_stats`, `_parse_args_rvs`
        methods to the instance.

        If self.shapes is a non-empty string, interprets it as a
        comma-separated list of shape parameters.

        Otherwise inspects the call signatures of `meths_to_inspect`
        and constructs the argument-parsing functions from these.
        In this case also sets `shapes` and `numargs`.
        """

        if self.shapes:
            # sanitize the user-supplied shapes
            if not isinstance(self.shapes, str):
                raise TypeError('shapes must be a string.')

            shapes = self.shapes.replace(',', ' ').split()

            for field in shapes:
                if keyword.iskeyword(field):
                    raise SyntaxError('keywords cannot be used as shapes.')
                if not re.match('^[_a-zA-Z][_a-zA-Z0-9]*$', field):
                    raise SyntaxError(
                        'shapes must be valid python identifiers')
        else:
            # find out the call signatures (_pdf, _cdf etc), deduce shape
            # arguments. Generic methods only have 'self, x', any further args
            # are shapes.
            shapes_list = []
            for meth in meths_to_inspect:
                shapes_args = _getfullargspec(meth)  # NB does not contain self
                args = shapes_args.args[1:]       # peel off 'x', too

                if args:
                    shapes_list.append(args)

                    # *args or **kwargs are not allowed w/automatic shapes
                    if shapes_args.varargs is not None:
                        raise TypeError(
                            '*args are not allowed w/out explicit shapes')
                    if shapes_args.varkw is not None:
                        raise TypeError(
                            '**kwds are not allowed w/out explicit shapes')
                    if shapes_args.kwonlyargs:
                        raise TypeError(
                            'kwonly args are not allowed w/out explicit shapes')
                    if shapes_args.defaults is not None:
                        raise TypeError('defaults are not allowed for shapes')

            if shapes_list:
                shapes = shapes_list[0]

                # make sure the signatures are consistent
                for item in shapes_list:
                    if item != shapes:
                        raise TypeError('Shape arguments are inconsistent.')
            else:
                shapes = []

        # have the arguments, construct the method from template
        shapes_str = ', '.join(shapes) + ', ' if shapes else ''  # NB: not None
        dct = dict(shape_arg_str=shapes_str,
                   locscale_in=locscale_in,
                   locscale_out=locscale_out,
                   )

        # this string is used by _attach_argparser_methods
        self._parse_arg_template = parse_arg_template % dct

        self.shapes = ', '.join(shapes) if shapes else None
        if not hasattr(self, 'numargs'):
            # allows more general subclassing with *args
            self.numargs = len(shapes)

    def _construct_doc(self, docdict, shapes_vals=None):
        """Construct the instance docstring with string substitutions."""
        tempdict = docdict.copy()
        tempdict['name'] = self.name or 'distname'
        tempdict['shapes'] = self.shapes or ''

        if shapes_vals is None:
            shapes_vals = ()
        try:
            vals = ', '.join(f'{val:.3g}' for val in shapes_vals)
        except TypeError:
            vals = ', '.join(f'{val}' for val in shapes_vals)
        tempdict['vals'] = vals

        tempdict['shapes_'] = self.shapes or ''
        if self.shapes and self.numargs == 1:
            tempdict['shapes_'] += ','

        if self.shapes:
            tempdict['set_vals_stmt'] = f'>>> {self.shapes} = {vals}'
        else:
            tempdict['set_vals_stmt'] = ''

        if self.shapes is None:
            # remove shapes from call parameters if there are none
            for item in ['default', 'before_notes']:
                tempdict[item] = tempdict[item].replace(
                    "\n%(shapes)s : array_like\n    shape parameters", "")
        for i in range(2):
            if self.shapes is None:
                # necessary because we use %(shapes)s in two forms (w w/o ", ")
                self.__doc__ = self.__doc__.replace("%(shapes)s, ", "")
            try:
                self.__doc__ = doccer.docformat(self.__doc__, tempdict)
            except TypeError as e:
                raise Exception("Unable to construct docstring for "
                                f"distribution \"{self.name}\": {repr(e)}") from e

        # correct for empty shapes
        self.__doc__ = self.__doc__.replace('(, ', '(').replace(', )', ')')

    def _construct_default_doc(self, longname=None,
                               docdict=None, discrete='continuous'):
        """Construct instance docstring from the default template."""
        if longname is None:
            longname = 'A'
        self.__doc__ = ''.join([f'{longname} {discrete} random variable.',
                                '\n\n%(before_notes)s\n', docheaders['notes'],
                                '\n%(example)s'])
        self._construct_doc(docdict)

    def freeze(self, *args, **kwds):
        """Freeze the distribution for the given arguments.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution.  Should include all
            the non-optional arguments, may include ``loc`` and ``scale``.

        Returns
        -------
        rv_frozen : rv_frozen instance
            The frozen distribution.

        """
        if isinstance(self, rv_continuous):
            return rv_continuous_frozen(self, *args, **kwds)
        else:
            return rv_discrete_frozen(self, *args, **kwds)

    def __call__(self, *args, **kwds):
        return self.freeze(*args, **kwds)
    __call__.__doc__ = freeze.__doc__

    # The actual calculation functions (no basic checking need be done)
    # If these are defined, the others won't be looked at.
    # Otherwise, the other set can be defined.
    def _stats(self, *args, **kwds):
        return None, None, None, None

    # Noncentral moments (also known as the moment about the origin).
    # Expressed in LaTeX, munp would be $\mu'_{n}$, i.e. "mu-sub-n-prime".
    # The primed mu is a widely used notation for the noncentral moment.
    def _munp(self, n, *args):
        # Silence floating point warnings from integration.
        with np.errstate(all='ignore'):
            vals = self.generic_moment(n, *args)
        return vals

    def _argcheck_rvs(self, *args, **kwargs):
        # Handle broadcasting and size validation of the rvs method.
        # Subclasses should not have to override this method.
        # The rule is that if `size` is not None, then `size` gives the
        # shape of the result (integer values of `size` are treated as
        # tuples with length 1; i.e. `size=3` is the same as `size=(3,)`.)
        #
        # `args` is expected to contain the shape parameters (if any), the
        # location and the scale in a flat tuple (e.g. if there are two
        # shape parameters `a` and `b`, `args` will be `(a, b, loc, scale)`).
        # The only keyword argument expected is 'size'.
        size = kwargs.get('size', None)
        all_bcast = np.broadcast_arrays(*args)

        def squeeze_left(a):
            while a.ndim > 0 and a.shape[0] == 1:
                a = a[0]
            return a

        # Eliminate trivial leading dimensions.  In the convention
        # used by numpy's random variate generators, trivial leading
        # dimensions are effectively ignored.  In other words, when `size`
        # is given, trivial leading dimensions of the broadcast parameters
        # in excess of the number of dimensions  in size are ignored, e.g.
        #   >>> np.random.normal([[1, 3, 5]], [[[[0.01]]]], size=3)
        #   array([ 1.00104267,  3.00422496,  4.99799278])
        # If `size` is not given, the exact broadcast shape is preserved:
        #   >>> np.random.normal([[1, 3, 5]], [[[[0.01]]]])
        #   array([[[[ 1.00862899,  3.00061431,  4.99867122]]]])
        #
        all_bcast = [squeeze_left(a) for a in all_bcast]
        bcast_shape = all_bcast[0].shape
        bcast_ndim = all_bcast[0].ndim

        if size is None:
            size_ = bcast_shape
        else:
            size_ = tuple(np.atleast_1d(size))

        # Check compatibility of size_ with the broadcast shape of all
        # the parameters.  This check is intended to be consistent with
        # how the numpy random variate generators (e.g. np.random.normal,
        # np.random.beta) handle their arguments.   The rule is that, if size
        # is given, it determines the shape of the output.  Broadcasting
        # can't change the output size.

        # This is the standard broadcasting convention of extending the
        # shape with fewer dimensions with enough dimensions of length 1
        # so that the two shapes have the same number of dimensions.
        ndiff = bcast_ndim - len(size_)
        if ndiff < 0:
            bcast_shape = (1,)*(-ndiff) + bcast_shape
        elif ndiff > 0:
            size_ = (1,)*ndiff + size_

        # This compatibility test is not standard.  In "regular" broadcasting,
        # two shapes are compatible if for each dimension, the lengths are the
        # same or one of the lengths is 1.  Here, the length of a dimension in
        # size_ must not be less than the corresponding length in bcast_shape.
        ok = all([bcdim == 1 or bcdim == szdim
                  for (bcdim, szdim) in zip(bcast_shape, size_)])
        if not ok:
            raise ValueError("size does not match the broadcast shape of "
                             f"the parameters. {size}, {size_}, {bcast_shape}")

        param_bcast = all_bcast[:-2]
        loc_bcast = all_bcast[-2]
        scale_bcast = all_bcast[-1]

        return param_bcast, loc_bcast, scale_bcast, size_

    # These are the methods you must define (standard form functions)
    # NB: generic _pdf, _logpdf, _cdf are different for
    # rv_continuous and rv_discrete hence are defined in there
    def _argcheck(self, *args):
        """Default check for correct values on args and keywords.

        Returns condition array of 1's where arguments are correct and
         0's where they are not.

        """
        cond = 1
        for arg in args:
            cond = logical_and(cond, (asarray(arg) > 0))
        return cond

    def _get_support(self, *args, **kwargs):
        """Return the support of the (unscaled, unshifted) distribution.

        *Must* be overridden by distributions which have support dependent
        upon the shape parameters of the distribution.  Any such override
        *must not* set or change any of the class members, as these members
        are shared amongst all instances of the distribution.

        Parameters
        ----------
        arg1, arg2, ... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).

        Returns
        -------
        a, b : numeric (float, or int or +/-np.inf)
            end-points of the distribution's support for the specified
            shape parameters.
        """
        return self.a, self.b

    def _support_mask(self, x, *args):
        a, b = self._get_support(*args)
        with np.errstate(invalid='ignore'):
            return (a <= x) & (x <= b)

    def _open_support_mask(self, x, *args):
        a, b = self._get_support(*args)
        with np.errstate(invalid='ignore'):
            return (a < x) & (x < b)

    def _rvs(self, *args, size=None, random_state=None):
        # This method must handle size being a tuple, and it must
        # properly broadcast *args and size.  size might be
        # an empty tuple, which means a scalar random variate is to be
        # generated.

        # Use basic inverse cdf algorithm for RV generation as default.
        U = random_state.uniform(size=size)
        Y = self._ppf(U, *args)
        return Y

    def _logcdf(self, x, *args):
        with np.errstate(divide='ignore'):
            return log(self._cdf(x, *args))

    def _sf(self, x, *args):
        return 1.0-self._cdf(x, *args)

    def _logsf(self, x, *args):
        with np.errstate(divide='ignore'):
            return log(self._sf(x, *args))

    def _ppf(self, q, *args):
        return self._ppfvec(q, *args)

    def _isf(self, q, *args):
        return self._ppf(1.0-q, *args)  # use correct _ppf for subclasses

    # These are actually called, and should not be overwritten if you
    # want to keep error checking.
    def rvs(self, *args, **kwds):
        """Random variates of given type.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).
        scale : array_like, optional
            Scale parameter (default=1).
        size : int or tuple of ints, optional
            Defining number of random variates (default is 1).
        random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional

            If `random_state` is None (or `np.random`), the
            `numpy.random.RandomState` singleton is used.
            If `random_state` is an int, a new ``RandomState`` instance is
            used, seeded with `random_state`.
            If `random_state` is already a ``Generator`` or ``RandomState``
            instance, that instance is used.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of given `size`.

        """
        discrete = kwds.pop('discrete', None)
        rndm = kwds.pop('random_state', None)
        args, loc, scale, size = self._parse_args_rvs(*args, **kwds)
        cond = logical_and(self._argcheck(*args), (scale >= 0))
        if not np.all(cond):
            message = ("Domain error in arguments. The `scale` parameter must "
                       "be positive for all distributions, and many "
                       "distributions have restrictions on shape parameters. "
                       f"Please see the `scipy.stats.{self.name}` "
                       "documentation for details.")
            raise ValueError(message)

        if np.all(scale == 0):
            return loc*ones(size, 'd')

        # extra gymnastics needed for a custom random_state
        if rndm is not None:
            random_state_saved = self._random_state
            random_state = check_random_state(rndm)
        else:
            random_state = self._random_state

        vals = self._rvs(*args, size=size, random_state=random_state)

        vals = vals * scale + loc

        # do not forget to restore the _random_state
        if rndm is not None:
            self._random_state = random_state_saved

        # Cast to int if discrete
        if discrete and not isinstance(self, rv_sample):
            if size == ():
                vals = int(vals)
            else:
                vals = vals.astype(np.int64)

        return vals

    def stats(self, *args, **kwds):
        """Some statistics of the given RV.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional (continuous RVs only)
            scale parameter (default=1)
        moments : str, optional
            composed of letters ['mvsk'] defining which moments to compute:
            'm' = mean,
            'v' = variance,
            's' = (Fisher's) skew,
            'k' = (Fisher's) kurtosis.
            (default is 'mv')

        Returns
        -------
        stats : sequence
            of requested moments.

        """
        args, loc, scale, moments = self._parse_args_stats(*args, **kwds)
        # scale = 1 by construction for discrete RVs
        loc, scale = map(asarray, (loc, scale))
        args = tuple(map(asarray, args))
        cond = self._argcheck(*args) & (scale > 0) & (loc == loc)
        output = []
        default = np.full(shape(cond), fill_value=self.badvalue)

        # Use only entries that are valid in calculation
        if np.any(cond):
            goodargs = argsreduce(cond, *(args+(scale, loc)))
            scale, loc, goodargs = goodargs[-2], goodargs[-1], goodargs[:-2]

            if self._stats_has_moments:
                mu, mu2, g1, g2 = self._stats(*goodargs,
                                              **{'moments': moments})
            else:
                mu, mu2, g1, g2 = self._stats(*goodargs)

            if 'm' in moments:
                if mu is None:
                    mu = self._munp(1, *goodargs)
                out0 = default.copy()
                place(out0, cond, mu * scale + loc)
                output.append(out0)

            if 'v' in moments:
                if mu2 is None:
                    mu2p = self._munp(2, *goodargs)
                    if mu is None:
                        mu = self._munp(1, *goodargs)
                    # if mean is inf then var is also inf
                    with np.errstate(invalid='ignore'):
                        mu2 = np.where(~np.isinf(mu), mu2p - mu**2, np.inf)
                out0 = default.copy()
                place(out0, cond, mu2 * scale * scale)
                output.append(out0)

            if 's' in moments:
                if g1 is None:
                    mu3p = self._munp(3, *goodargs)
                    if mu is None:
                        mu = self._munp(1, *goodargs)
                    if mu2 is None:
                        mu2p = self._munp(2, *goodargs)
                        with np.errstate(invalid='ignore'):
                            mu2 = mu2p - mu * mu
                    with np.errstate(invalid='ignore'):
                        mu3 = (-mu*mu - 3*mu2)*mu + mu3p
                        g1 = mu3 / np.power(mu2, 1.5)
                out0 = default.copy()
                place(out0, cond, g1)
                output.append(out0)

            if 'k' in moments:
                if g2 is None:
                    mu4p = self._munp(4, *goodargs)
                    if mu is None:
                        mu = self._munp(1, *goodargs)
                    if mu2 is None:
                        mu2p = self._munp(2, *goodargs)
                        with np.errstate(invalid='ignore'):
                            mu2 = mu2p - mu * mu
                    if g1 is None:
                        mu3 = None
                    else:
                        # (mu2**1.5) breaks down for nan and inf
                        mu3 = g1 * np.power(mu2, 1.5)
                    if mu3 is None:
                        mu3p = self._munp(3, *goodargs)
                        with np.errstate(invalid='ignore'):
                            mu3 = (-mu * mu - 3 * mu2) * mu + mu3p
                    with np.errstate(invalid='ignore'):
                        mu4 = ((-mu**2 - 6*mu2) * mu - 4*mu3)*mu + mu4p
                        g2 = mu4 / mu2**2.0 - 3.0
                out0 = default.copy()
                place(out0, cond, g2)
                output.append(out0)
        else:  # no valid args
            output = [default.copy() for _ in moments]

        output = [out[()] for out in output]
        if len(output) == 1:
            return output[0]
        else:
            return tuple(output)

    def entropy(self, *args, **kwds):
        """Differential entropy of the RV.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            Location parameter (default=0).
        scale : array_like, optional  (continuous distributions only).
            Scale parameter (default=1).

        Notes
        -----
        Entropy is defined base `e`:

        >>> import numpy as np
        >>> from scipy.stats._distn_infrastructure import rv_discrete
        >>> drv = rv_discrete(values=((0, 1), (0.5, 0.5)))
        >>> np.allclose(drv.entropy(), np.log(2.0))
        True

        """
        args, loc, scale = self._parse_args(*args, **kwds)
        # NB: for discrete distributions scale=1 by construction in _parse_args
        loc, scale = map(asarray, (loc, scale))
        args = tuple(map(asarray, args))
        cond0 = self._argcheck(*args) & (scale > 0) & (loc == loc)
        output = zeros(shape(cond0), 'd')
        place(output, (1-cond0), self.badvalue)
        goodargs = argsreduce(cond0, scale, *args)
        goodscale = goodargs[0]
        goodargs = goodargs[1:]
        place(output, cond0, self.vecentropy(*goodargs) + log(goodscale))
        return output[()]

    def moment(self, order, *args, **kwds):
        """non-central moment of distribution of specified order.

        Parameters
        ----------
        order : int, order >= 1
            Order of moment.
        arg1, arg2, arg3,... : float
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        """
        n = order
        shapes, loc, scale = self._parse_args(*args, **kwds)
        args = np.broadcast_arrays(*(*shapes, loc, scale))
        *shapes, loc, scale = args

        i0 = np.logical_and(self._argcheck(*shapes), scale > 0)
        i1 = np.logical_and(i0, loc == 0)
        i2 = np.logical_and(i0, loc != 0)

        args = argsreduce(i0, *shapes, loc, scale)
        *shapes, loc, scale = args

        if (floor(n) != n):
            raise ValueError("Moment must be an integer.")
        if (n < 0):
            raise ValueError("Moment must be positive.")
        mu, mu2, g1, g2 = None, None, None, None
        if (n > 0) and (n < 5):
            if self._stats_has_moments:
                mdict = {'moments': {1: 'm', 2: 'v', 3: 'vs', 4: 'mvsk'}[n]}
            else:
                mdict = {}
            mu, mu2, g1, g2 = self._stats(*shapes, **mdict)
        val = np.empty(loc.shape)  # val needs to be indexed by loc
        val[...] = _moment_from_stats(n, mu, mu2, g1, g2, self._munp, shapes)

        # Convert to transformed  X = L + S*Y
        # E[X^n] = E[(L+S*Y)^n] = L^n sum(comb(n, k)*(S/L)^k E[Y^k], k=0...n)
        result = zeros(i0.shape)
        place(result, ~i0, self.badvalue)

        if i1.any():
            res1 = scale[loc == 0]**n * val[loc == 0]
            place(result, i1, res1)

        if i2.any():
            mom = [mu, mu2, g1, g2]
            arrs = [i for i in mom if i is not None]
            idx = [i for i in range(4) if mom[i] is not None]
            if any(idx):
                arrs = argsreduce(loc != 0, *arrs)
                j = 0
                for i in idx:
                    mom[i] = arrs[j]
                    j += 1
            mu, mu2, g1, g2 = mom
            args = argsreduce(loc != 0, *shapes, loc, scale, val)
            *shapes, loc, scale, val = args

            res2 = zeros(loc.shape, dtype='d')
            fac = scale / loc
            for k in range(n):
                valk = _moment_from_stats(k, mu, mu2, g1, g2, self._munp,
                                          shapes)
                res2 += comb(n, k, exact=True)*fac**k * valk
            res2 += fac**n * val
            res2 *= loc**n
            place(result, i2, res2)

        return result[()]

    def median(self, *args, **kwds):
        """Median of the distribution.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            Location parameter, Default is 0.
        scale : array_like, optional
            Scale parameter, Default is 1.

        Returns
        -------
        median : float
            The median of the distribution.

        See Also
        --------
        rv_discrete.ppf
            Inverse of the CDF

        """
        return self.ppf(0.5, *args, **kwds)

    def mean(self, *args, **kwds):
        """Mean of the distribution.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        mean : float
            the mean of the distribution

        """
        kwds['moments'] = 'm'
        res = self.stats(*args, **kwds)
        if isinstance(res, ndarray) and res.ndim == 0:
            return res[()]
        return res

    def var(self, *args, **kwds):
        """Variance of the distribution.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        var : float
            the variance of the distribution

        """
        kwds['moments'] = 'v'
        res = self.stats(*args, **kwds)
        if isinstance(res, ndarray) and res.ndim == 0:
            return res[()]
        return res

    def std(self, *args, **kwds):
        """Standard deviation of the distribution.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : array_like, optional
            location parameter (default=0)
        scale : array_like, optional
            scale parameter (default=1)

        Returns
        -------
        std : float
            standard deviation of the distribution

        """
        kwds['moments'] = 'v'
        res = sqrt(self.stats(*args, **kwds))
        return res

    def interval(self, confidence, *args, **kwds):
        """Confidence interval with equal areas around the median.

        Parameters
        ----------
        confidence : array_like of float
            Probability that an rv will be drawn from the returned range.
            Each value should be in the range [0, 1].
        arg1, arg2, ... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            location parameter, Default is 0.
        scale : array_like, optional
            scale parameter, Default is 1.

        Returns
        -------
        a, b : ndarray of float
            end-points of range that contain ``100 * alpha %`` of the rv's
            possible values.

        Notes
        -----
        This is implemented as ``ppf([p_tail, 1-p_tail])``, where
        ``ppf`` is the inverse cumulative distribution function and
        ``p_tail = (1-confidence)/2``. Suppose ``[c, d]`` is the support of a
        discrete distribution; then ``ppf([0, 1]) == (c-1, d)``. Therefore,
        when ``confidence=1`` and the distribution is discrete, the left end
        of the interval will be beyond the support of the distribution.
        For discrete distributions, the interval will limit the probability
        in each tail to be less than or equal to ``p_tail`` (usually
        strictly less).

        """
        alpha = confidence

        alpha = asarray(alpha)
        if np.any((alpha > 1) | (alpha < 0)):
            raise ValueError("alpha must be between 0 and 1 inclusive")
        q1 = (1.0-alpha)/2
        q2 = (1.0+alpha)/2
        a = self.ppf(q1, *args, **kwds)
        b = self.ppf(q2, *args, **kwds)
        return a, b

    def support(self, *args, **kwargs):
        """Support of the distribution.

        Parameters
        ----------
        arg1, arg2, ... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        loc : array_like, optional
            location parameter, Default is 0.
        scale : array_like, optional
            scale parameter, Default is 1.

        Returns
        -------
        a, b : array_like
            end-points of the distribution's support.

        """
        args, loc, scale = self._parse_args(*args, **kwargs)
        arrs = np.broadcast_arrays(*args, loc, scale)
        args, loc, scale = arrs[:-2], arrs[-2], arrs[-1]
        cond = self._argcheck(*args) & (scale > 0)
        _a, _b = self._get_support(*args)
        if cond.all():
            return _a * scale + loc, _b * scale + loc
        elif cond.ndim == 0:
            return self.badvalue, self.badvalue
        # promote bounds to at least float to fill in the badvalue
        _a, _b = np.asarray(_a).astype('d'), np.asarray(_b).astype('d')
        out_a, out_b = _a * scale + loc, _b * scale + loc
        place(out_a, 1-cond, self.badvalue)
        place(out_b, 1-cond, self.badvalue)
        return out_a, out_b

    def nnlf(self, theta, x):
        """Negative loglikelihood function.
        Notes
        -----
        This is ``-sum(log pdf(x, theta), axis=0)`` where `theta` are the
        parameters (including loc and scale).
        """
        loc, scale, args = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return inf
        x = (asarray(x)-loc) / scale
        n_log_scale = len(x) * log(scale)
        if np.any(~self._support_mask(x, *args)):
            return inf
        return self._nnlf(x, *args) + n_log_scale

    def _nnlf(self, x, *args):
        return -np.sum(self._logpxf(x, *args), axis=0)

    def _nlff_and_penalty(self, x, args, log_fitfun):
        # negative log fit function
        cond0 = ~self._support_mask(x, *args)
        n_bad = np.count_nonzero(cond0, axis=0)
        if n_bad > 0:
            x = argsreduce(~cond0, x)[0]
        logff = log_fitfun(x, *args)
        finite_logff = np.isfinite(logff)
        n_bad += np.sum(~finite_logff, axis=0)
        if n_bad > 0:
            penalty = n_bad * log(_XMAX) * 100
            return -np.sum(logff[finite_logff], axis=0) + penalty
        return -np.sum(logff, axis=0)

    def _penalized_nnlf(self, theta, x):
        """Penalized negative loglikelihood function.
        i.e., - sum (log pdf(x, theta), axis=0) + penalty
        where theta are the parameters (including loc and scale)
        """
        loc, scale, args = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return inf
        x = asarray((x-loc) / scale)
        n_log_scale = len(x) * log(scale)
        return self._nlff_and_penalty(x, args, self._logpxf) + n_log_scale

    def _penalized_nlpsf(self, theta, x):
        """Penalized negative log product spacing function.
        i.e., - sum (log (diff (cdf (x, theta))), axis=0) + penalty
        where theta are the parameters (including loc and scale)
        Follows reference [1] of scipy.stats.fit
        """
        loc, scale, args = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return inf
        x = (np.sort(x) - loc)/scale

        def log_psf(x, *args):
            x, lj = np.unique(x, return_counts=True)  # fast for sorted x
            cdf_data = self._cdf(x, *args) if x.size else []
            if not (x.size and 1 - cdf_data[-1] <= 0):
                cdf = np.concatenate(([0], cdf_data, [1]))
                lj = np.concatenate((lj, [1]))
            else:
                cdf = np.concatenate(([0], cdf_data))
            # here we could use logcdf w/ logsumexp trick to take differences,
            # but in the context of the method, it seems unlikely to matter
            return lj * np.log(np.diff(cdf) / lj)

        return self._nlff_and_penalty(x, args, log_psf)


class _ShapeInfo:
    def __init__(self, name, integrality=False, domain=(-np.inf, np.inf),
                 inclusive=(True, True)):
        self.name = name
        self.integrality = integrality
        self.endpoints = domain
        self.inclusive = inclusive

        domain = list(domain)
        if np.isfinite(domain[0]) and not inclusive[0]:
            domain[0] = np.nextafter(domain[0], np.inf)
        if np.isfinite(domain[1]) and not inclusive[1]:
            domain[1] = np.nextafter(domain[1], -np.inf)
        self.domain = domain


def _get_fixed_fit_value(kwds, names):
    """
    Given names such as ``['f0', 'fa', 'fix_a']``, check that there is
    at most one non-None value in `kwds` associated with those names.
    Return that value, or None if none of the names occur in `kwds`.
    As a side effect, all occurrences of those names in `kwds` are
    removed.
    """
    vals = [(name, kwds.pop(name)) for name in names if name in kwds]
    if len(vals) > 1:
        repeated = [name for name, val in vals]
        raise ValueError("fit method got multiple keyword arguments to "
                         "specify the same fixed parameter: " +
                         ', '.join(repeated))
    return vals[0][1] if vals else None


#  continuous random variables: implement maybe later
#
#  hf  --- Hazard Function (PDF / SF)
#  chf  --- Cumulative hazard function (-log(SF))
#  psf --- Probability sparsity function (reciprocal of the pdf) in
#                units of percent-point-function (as a function of q).
#                Also, the derivative of the percent-point function.


class rv_continuous(rv_generic):
    """A generic continuous random variable class meant for subclassing.

    `rv_continuous` is a base class to construct specific distribution classes
    and instances for continuous random variables. It cannot be used
    directly as a distribution.

    Parameters
    ----------
    momtype : int, optional
        The type of generic moment calculation to use: 0 for pdf, 1 (default)
        for ppf.
    a : float, optional
        Lower bound of the support of the distribution, default is minus
        infinity.
    b : float, optional
        Upper bound of the support of the distribution, default is plus
        infinity.
    xtol : float, optional
        The tolerance for fixed point calculation for generic ppf.
    badvalue : float, optional
        The value in a result arrays that indicates a value that for which
        some argument restriction is violated, default is np.nan.
    name : str, optional
