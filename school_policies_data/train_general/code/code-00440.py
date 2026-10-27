    >>> from sympy import Symbol, S

    >>> a = S.One / 5
    >>> z = Symbol("z")

    >>> X = FlorySchulz("x", a)

    >>> density(X)(z)
    (4/5)**(z - 1)*z/25

    >>> E(X)
    9

    >>> variance(X)
    40

    References
    ==========

    https://en.wikipedia.org/wiki/Flory%E2%80%93Schulz_distribution
    """
    return rv(name, FlorySchulzDistribution, a)


#-------------------------------------------------------------------------------
# Geometric distribution ------------------------------------------------------------

class GeometricDistribution(SingleDiscreteDistribution):
    _argnames = ('p',)
    set = S.Naturals

    @staticmethod
    def check(p):
        _value_check((0 < p, p <= 1), "p must be between 0 and 1")

    def pdf(self, k):
        return (1 - self.p)**(k - 1) * self.p

    def _characteristic_function(self, t):
        p = self.p
        return p * exp(I*t) / (1 - (1 - p)*exp(I*t))

    def _moment_generating_function(self, t):
        p = self.p
        return p * exp(t) / (1 - (1 - p) * exp(t))


def Geometric(name, p):
    r"""
    Create a discrete random variable with a Geometric distribution.

    Explanation
    ===========

    The density of the Geometric distribution is given by

    .. math::
        f(k) := p (1 - p)^{k - 1}

    Parameters
    ==========

    p : A probability between 0 and 1

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Geometric, density, E, variance
    >>> from sympy import Symbol, S

    >>> p = S.One / 5
    >>> z = Symbol("z")

    >>> X = Geometric("x", p)

    >>> density(X)(z)
    (4/5)**(z - 1)/5

    >>> E(X)
    5

    >>> variance(X)
    20

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Geometric_distribution
    .. [2] https://mathworld.wolfram.com/GeometricDistribution.html

    """
    return rv(name, GeometricDistribution, p)


#-------------------------------------------------------------------------------
# Hermite distribution ---------------------------------------------------------


class HermiteDistribution(SingleDiscreteDistribution):
    _argnames = ('a1', 'a2')
    set = S.Naturals0

    @staticmethod
    def check(a1, a2):
        _value_check(a1.is_nonnegative, 'Parameter a1 must be >= 0.')
        _value_check(a2.is_nonnegative, 'Parameter a2 must be >= 0.')

    def pdf(self, k):
        a1, a2 = self.a1, self.a2
        term1 = exp(-(a1 + a2))
        j = Dummy("j", integer=True)
        num = a1**(k - 2*j) * a2**j
        den = factorial(k - 2*j) * factorial(j)
        return term1 * Sum(num/den, (j, 0, k//2)).doit()

    def _moment_generating_function(self, t):
        a1, a2 = self.a1, self.a2
        term1 = a1 * (exp(t) - 1)
        term2 = a2 * (exp(2*t) - 1)
        return exp(term1 + term2)

    def _characteristic_function(self, t):
        a1, a2 = self.a1, self.a2
        term1 = a1 * (exp(I*t) - 1)
        term2 = a2 * (exp(2*I*t) - 1)
        return exp(term1 + term2)

def Hermite(name, a1, a2):
    r"""
    Create a discrete random variable with a Hermite distribution.

    Explanation
    ===========

    The density of the Hermite distribution is given by

    .. math::
        f(x):= e^{-a_1 -a_2}\sum_{j=0}^{\left \lfloor x/2 \right \rfloor}
                    \frac{a_{1}^{x-2j}a_{2}^{j}}{(x-2j)!j!}

    Parameters
    ==========

    a1 : A Positive number greater than equal to 0.
    a2 : A Positive number greater than equal to 0.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Hermite, density, E, variance
    >>> from sympy import Symbol

    >>> a1 = Symbol("a1", positive=True)
    >>> a2 = Symbol("a2", positive=True)
    >>> x = Symbol("x")

    >>> H = Hermite("H", a1=5, a2=4)

    >>> density(H)(2)
    33*exp(-9)/2

    >>> E(H)
    13

    >>> variance(H)
    21

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hermite_distribution

    """

    return rv(name, HermiteDistribution, a1, a2)


#-------------------------------------------------------------------------------
# Logarithmic distribution ------------------------------------------------------------

class LogarithmicDistribution(SingleDiscreteDistribution):
    _argnames = ('p',)

    set = S.Naturals

    @staticmethod
    def check(p):
        _value_check((p > 0, p < 1), "p should be between 0 and 1")

    def pdf(self, k):
        p = self.p
        return (-1) * p**k / (k * log(1 - p))

    def _characteristic_function(self, t):
        p = self.p
        return log(1 - p * exp(I*t)) / log(1 - p)

    def _moment_generating_function(self, t):
        p = self.p
        return log(1 - p * exp(t)) / log(1 - p)


def Logarithmic(name, p):
    r"""
    Create a discrete random variable with a Logarithmic distribution.

    Explanation
    ===========

    The density of the Logarithmic distribution is given by

    .. math::
        f(k) := \frac{-p^k}{k \ln{(1 - p)}}

    Parameters
    ==========

    p : A value between 0 and 1

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Logarithmic, density, E, variance
    >>> from sympy import Symbol, S

    >>> p = S.One / 5
    >>> z = Symbol("z")

    >>> X = Logarithmic("x", p)

    >>> density(X)(z)
    -1/(5**z*z*log(4/5))

    >>> E(X)
    -1/(-4*log(5) + 8*log(2))

    >>> variance(X)
    -1/((-4*log(5) + 8*log(2))*(-2*log(5) + 4*log(2))) + 1/(-64*log(2)*log(5) + 64*log(2)**2 + 16*log(5)**2) - 10/(-32*log(5) + 64*log(2))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Logarithmic_distribution
    .. [2] https://mathworld.wolfram.com/LogarithmicDistribution.html

    """
    return rv(name, LogarithmicDistribution, p)


#-------------------------------------------------------------------------------
# Negative binomial distribution ------------------------------------------------------------

class NegativeBinomialDistribution(SingleDiscreteDistribution):
    _argnames = ('r', 'p')
    set = S.Naturals0

    @staticmethod
    def check(r, p):
        _value_check(r > 0, 'r should be positive')
        _value_check((p > 0, p < 1), 'p should be between 0 and 1')

    def pdf(self, k):
        r = self.r
        p = self.p

        return binomial(k + r - 1, k) * (1 - p)**k * p**r

    def _characteristic_function(self, t):
        r = self.r
        p = self.p

        return (p / (1 - (1 - p) * exp(I*t)))**r

    def _moment_generating_function(self, t):
        r = self.r
        p = self.p

        return (p / (1 - (1 - p) * exp(t)))**r

def NegativeBinomial(name, r, p):
    r"""
    Create a discrete random variable with a Negative Binomial distribution.

    Explanation
    ===========

    The density of the Negative Binomial distribution is given by

    .. math::
        f(k) := \binom{k + r - 1}{k} (1 - p)^k p^r

    Parameters
    ==========

    r : A positive value
        Number of successes until the experiment is stopped.
    p : A value between 0 and 1
        Probability of success.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import NegativeBinomial, density, E, variance
    >>> from sympy import Symbol, S

    >>> r = 5
    >>> p = S.One / 3
    >>> z = Symbol("z")

    >>> X = NegativeBinomial("x", r, p)

    >>> density(X)(z)
    (2/3)**z*binomial(z + 4, z)/243

    >>> E(X)
    10

    >>> variance(X)
    30

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Negative_binomial_distribution
    .. [2] https://mathworld.wolfram.com/NegativeBinomialDistribution.html

    """
    return rv(name, NegativeBinomialDistribution, r, p)


#-------------------------------------------------------------------------------
# Poisson distribution ------------------------------------------------------------

class PoissonDistribution(SingleDiscreteDistribution):
    _argnames = ('lamda',)

    set = S.Naturals0

    @staticmethod
    def check(lamda):
        _value_check(lamda > 0, "Lambda must be positive")

    def pdf(self, k):
        return self.lamda**k / factorial(k) * exp(-self.lamda)

    def _characteristic_function(self, t):
        return exp(self.lamda * (exp(I*t) - 1))

    def _moment_generating_function(self, t):
        return exp(self.lamda * (exp(t) - 1))

    def expectation(self, expr, var, evaluate=True, **kwargs):
        if evaluate:
            if expr == var:
                return self.lamda
            if (
                isinstance(expr, FallingFactorial)
                and expr.args[1].is_integer
                and expr.args[1].is_positive
                and expr.args[0] == var
            ):
                return self.lamda ** expr.args[1]
        return super().expectation(expr, var, evaluate, **kwargs)

def Poisson(name, lamda):
    r"""
    Create a discrete random variable with a Poisson distribution.

    Explanation
    ===========

    The density of the Poisson distribution is given by

    .. math::
        f(k) := \frac{\lambda^{k} e^{- \lambda}}{k!}

    Parameters
    ==========

    lamda : Positive number, a rate

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Poisson, density, E, variance
    >>> from sympy import Symbol, simplify

    >>> rate = Symbol("lambda", positive=True)
    >>> z = Symbol("z")

    >>> X = Poisson("x", rate)

    >>> density(X)(z)
    lambda**z*exp(-lambda)/factorial(z)

    >>> E(X)
    lambda

    >>> simplify(variance(X))
    lambda

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Poisson_distribution
    .. [2] https://mathworld.wolfram.com/PoissonDistribution.html

    """
    return rv(name, PoissonDistribution, lamda)


# -----------------------------------------------------------------------------
# Skellam distribution --------------------------------------------------------


class SkellamDistribution(SingleDiscreteDistribution):
    _argnames = ('mu1', 'mu2')
    set = S.Integers

    @staticmethod
    def check(mu1, mu2):
        _value_check(mu1 >= 0, 'Parameter mu1 must be >= 0')
        _value_check(mu2 >= 0, 'Parameter mu2 must be >= 0')

    def pdf(self, k):
        (mu1, mu2) = (self.mu1, self.mu2)
        term1 = exp(-(mu1 + mu2)) * (mu1 / mu2) ** (k / 2)
        term2 = besseli(k, 2 * sqrt(mu1 * mu2))
        return term1 * term2

    def _cdf(self, x):
        raise NotImplementedError(
            "Skellam doesn't have closed form for the CDF.")

    def _characteristic_function(self, t):
        (mu1, mu2) = (self.mu1, self.mu2)
        return exp(-(mu1 + mu2) + mu1 * exp(I * t) + mu2 * exp(-I * t))

    def _moment_generating_function(self, t):
        (mu1, mu2) = (self.mu1, self.mu2)
        return exp(-(mu1 + mu2) + mu1 * exp(t) + mu2 * exp(-t))


def Skellam(name, mu1, mu2):
    r"""
    Create a discrete random variable with a Skellam distribution.

    Explanation
    ===========

    The Skellam is the distribution of the difference N1 - N2
    of two statistically independent random variables N1 and N2
    each Poisson-distributed with respective expected values mu1 and mu2.

    The density of the Skellam distribution is given by

    .. math::
        f(k) := e^{-(\mu_1+\mu_2)}(\frac{\mu_1}{\mu_2})^{k/2}I_k(2\sqrt{\mu_1\mu_2})

    Parameters
    ==========

    mu1 : A non-negative value
    mu2 : A non-negative value

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Skellam, density, E, variance
    >>> from sympy import Symbol, pprint

    >>> z = Symbol("z", integer=True)
    >>> mu1 = Symbol("mu1", positive=True)
    >>> mu2 = Symbol("mu2", positive=True)
    >>> X = Skellam("x", mu1, mu2)

    >>> pprint(density(X)(z), use_unicode=False)
         z
         -
         2
    /mu1\   -mu1 - mu2        /       _____   _____\
    |---| *e          *besseli\z, 2*\/ mu1 *\/ mu2 /
    \mu2/
    >>> E(X)
    mu1 - mu2
    >>> variance(X).expand()
    mu1 + mu2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Skellam_distribution

    """
    return rv(name, SkellamDistribution, mu1, mu2)


#-------------------------------------------------------------------------------
# Yule-Simon distribution ------------------------------------------------------------

class YuleSimonDistribution(SingleDiscreteDistribution):
    _argnames = ('rho',)
    set = S.Naturals

    @staticmethod
    def check(rho):
        _value_check(rho > 0, 'rho should be positive')

    def pdf(self, k):
        rho = self.rho
        return rho * beta(k, rho + 1)

    def _cdf(self, x):
        return Piecewise((1 - floor(x) * beta(floor(x), self.rho + 1), x >= 1), (0, True))

    def _characteristic_function(self, t):
        rho = self.rho
        return rho * hyper((1, 1), (rho + 2,), exp(I*t)) * exp(I*t) / (rho + 1)

    def _moment_generating_function(self, t):
        rho = self.rho
        return rho * hyper((1, 1), (rho + 2,), exp(t)) * exp(t) / (rho + 1)


def YuleSimon(name, rho):
    r"""
    Create a discrete random variable with a Yule-Simon distribution.

    Explanation
    ===========

    The density of the Yule-Simon distribution is given by

    .. math::
        f(k) := \rho B(k, \rho + 1)

    Parameters
    ==========

    rho : A positive value

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import YuleSimon, density, E, variance
    >>> from sympy import Symbol, simplify

    >>> p = 5
    >>> z = Symbol("z")

    >>> X = YuleSimon("x", p)

    >>> density(X)(z)
    5*beta(z, 6)

    >>> simplify(E(X))
    5/4

    >>> simplify(variance(X))
    25/48

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Yule%E2%80%93Simon_distribution

    """
    return rv(name, YuleSimonDistribution, rho)


#-------------------------------------------------------------------------------
# Zeta distribution ------------------------------------------------------------

class ZetaDistribution(SingleDiscreteDistribution):
    _argnames = ('s',)
    set = S.Naturals

    @staticmethod
    def check(s):
        _value_check(s > 1, 's should be greater than 1')

    def pdf(self, k):
        s = self.s
        return 1 / (k**s * zeta(s))

    def _characteristic_function(self, t):
        return polylog(self.s, exp(I*t)) / zeta(self.s)

    def _moment_generating_function(self, t):
        return polylog(self.s, exp(t)) / zeta(self.s)


def Zeta(name, s):
    r"""
    Create a discrete random variable with a Zeta distribution.

    Explanation
    ===========

    The density of the Zeta distribution is given by

    .. math::
        f(k) := \frac{1}{k^s \zeta{(s)}}

    Parameters
    ==========

    s : A value greater than 1

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Zeta, density, E, variance
    >>> from sympy import Symbol

    >>> s = 5
    >>> z = Symbol("z")

    >>> X = Zeta("x", s)

    >>> density(X)(z)
    1/(z**5*zeta(5))

    >>> E(X)
    pi**4/(90*zeta(5))

    >>> variance(X)
    -pi**8/(8100*zeta(5)**2) + zeta(3)/zeta(5)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Zeta_distribution

    """
    return rv(name, ZetaDistribution, s)
<|endoftext|>from sympy.core.basic import Basic
from sympy.stats.rv import PSpace, _symbol_converter, RandomMatrixSymbol

class RandomMatrixPSpace(PSpace):
    """
    Represents probability space for
    random matrices. It contains the mechanics
    for handling the API calls for random matrices.
    """
    def __new__(cls, sym, model=None):
        sym = _symbol_converter(sym)
        if model:
            return Basic.__new__(cls, sym, model)
        else:
            return Basic.__new__(cls, sym)

    @property
    def model(self):
        try:
            return self.args[1]
        except IndexError:
            return None

    def compute_density(self, expr, *args):
        rms = expr.atoms(RandomMatrixSymbol)
        if len(rms) > 2 or (not isinstance(expr, RandomMatrixSymbol)):
            raise NotImplementedError("Currently, no algorithm has been "
                    "implemented to handle general expressions containing "
                    "multiple random matrices.")
        return self.model.density(expr)
<|endoftext|>from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.numbers import (I, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.trace import Trace
from sympy.tensor.indexed import IndexedBase
from sympy.core.sympify import _sympify
from sympy.stats.rv import _symbol_converter, Density, RandomMatrixSymbol, is_random
from sympy.stats.joint_rv_types import JointDistributionHandmade
from sympy.stats.random_matrix import RandomMatrixPSpace
from sympy.tensor.array import ArrayComprehension

__all__ = [
    'CircularEnsemble',
    'CircularUnitaryEnsemble',
    'CircularOrthogonalEnsemble',
    'CircularSymplecticEnsemble',
    'GaussianEnsemble',
    'GaussianUnitaryEnsemble',
    'GaussianOrthogonalEnsemble',
    'GaussianSymplecticEnsemble',
    'joint_eigen_distribution',
    'JointEigenDistribution',
    'level_spacing_distribution'
]

@is_random.register(RandomMatrixSymbol)
def _(x):
    return True


class RandomMatrixEnsembleModel(Basic):
    """
    Base class for random matrix ensembles.
    It acts as an umbrella and contains
    the methods common to all the ensembles
    defined in sympy.stats.random_matrix_models.
    """
    def __new__(cls, sym, dim=None):
        sym, dim = _symbol_converter(sym), _sympify(dim)
        if dim.is_integer == False:
            raise ValueError("Dimension of the random matrices must be "
                                "integers, received %s instead."%(dim))
        return Basic.__new__(cls, sym, dim)

    symbol = property(lambda self: self.args[0])
    dimension = property(lambda self: self.args[1])

    def density(self, expr):
        return Density(expr)

    def __call__(self, expr):
        return self.density(expr)

class GaussianEnsembleModel(RandomMatrixEnsembleModel):
    """
    Abstract class for Gaussian ensembles.
    Contains the properties common to all the
    gaussian ensembles.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Random_matrix#Gaussian_ensembles
    .. [2] https://arxiv.org/pdf/1712.07903.pdf
    """
    def _compute_normalization_constant(self, beta, n):
        """
        Helper function for computing normalization
        constant for joint probability density of eigen
        values of Gaussian ensembles.

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Selberg_integral#Mehta's_integral
        """
        n = S(n)
        prod_term = lambda j: gamma(1 + beta*S(j)/2)/gamma(S.One + beta/S(2))
        j = Dummy('j', integer=True, positive=True)
        term1 = Product(prod_term(j), (j, 1, n)).doit()
        term2 = (2/(beta*n))**(beta*n*(n - 1)/4 + n/2)
        term3 = (2*pi)**(n/2)
        return term1 * term2 * term3

    def _compute_joint_eigen_distribution(self, beta):
        """
        Helper function for computing the joint
        probability distribution of eigen values
        of the random matrix.
        """
        n = self.dimension
        Zbn = self._compute_normalization_constant(beta, n)
        l = IndexedBase('l')
        i = Dummy('i', integer=True, positive=True)
        j = Dummy('j', integer=True, positive=True)
        k = Dummy('k', integer=True, positive=True)
        term1 = exp((-S(n)/2) * Sum(l[k]**2, (k, 1, n)).doit())
        sub_term = Lambda(i, Product(Abs(l[j] - l[i])**beta, (j, i + 1, n)))
        term2 = Product(sub_term(i).doit(), (i, 1, n - 1)).doit()
        syms = ArrayComprehension(l[k], (k, 1, n)).doit()
        return Lambda(tuple(syms), (term1 * term2)/Zbn)

class GaussianUnitaryEnsembleModel(GaussianEnsembleModel):
    @property
    def normalization_constant(self):
        n = self.dimension
        return 2**(S(n)/2) * pi**(S(n**2)/2)

    def density(self, expr):
        n, ZGUE = self.dimension, self.normalization_constant
        h_pspace = RandomMatrixPSpace('P', model=self)
        H = RandomMatrixSymbol('H', n, n, pspace=h_pspace)
        return Lambda(H, exp(-S(n)/2 * Trace(H**2))/ZGUE)(expr)

    def joint_eigen_distribution(self):
        return self._compute_joint_eigen_distribution(S(2))

    def level_spacing_distribution(self):
        s = Dummy('s')
        f = (32/pi**2)*(s**2)*exp((-4/pi)*s**2)
        return Lambda(s, f)

class GaussianOrthogonalEnsembleModel(GaussianEnsembleModel):
    @property
    def normalization_constant(self):
        n = self.dimension
        _H = MatrixSymbol('_H', n, n)
        return Integral(exp(-S(n)/4 * Trace(_H**2)))

    def density(self, expr):
        n, ZGOE = self.dimension, self.normalization_constant
        h_pspace = RandomMatrixPSpace('P', model=self)
        H = RandomMatrixSymbol('H', n, n, pspace=h_pspace)
        return Lambda(H, exp(-S(n)/4 * Trace(H**2))/ZGOE)(expr)

    def joint_eigen_distribution(self):
        return self._compute_joint_eigen_distribution(S.One)

    def level_spacing_distribution(self):
        s = Dummy('s')
        f = (pi/2)*s*exp((-pi/4)*s**2)
        return Lambda(s, f)

class GaussianSymplecticEnsembleModel(GaussianEnsembleModel):
    @property
    def normalization_constant(self):
        n = self.dimension
        _H = MatrixSymbol('_H', n, n)
        return Integral(exp(-S(n) * Trace(_H**2)))

    def density(self, expr):
        n, ZGSE = self.dimension, self.normalization_constant
        h_pspace = RandomMatrixPSpace('P', model=self)
        H = RandomMatrixSymbol('H', n, n, pspace=h_pspace)
        return Lambda(H, exp(-S(n) * Trace(H**2))/ZGSE)(expr)

    def joint_eigen_distribution(self):
        return self._compute_joint_eigen_distribution(S(4))

    def level_spacing_distribution(self):
        s = Dummy('s')
        f = ((S(2)**18)/((S(3)**6)*(pi**3)))*(s**4)*exp((-64/(9*pi))*s**2)
        return Lambda(s, f)

def GaussianEnsemble(sym, dim):
    sym, dim = _symbol_converter(sym), _sympify(dim)
    model = GaussianEnsembleModel(sym, dim)
    rmp = RandomMatrixPSpace(sym, model=model)
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)

def GaussianUnitaryEnsemble(sym, dim):
    """
    Represents Gaussian Unitary Ensembles.

    Examples
    ========

    >>> from sympy.stats import GaussianUnitaryEnsemble as GUE, density
    >>> from sympy import MatrixSymbol
    >>> G = GUE('U', 2)
    >>> X = MatrixSymbol('X', 2, 2)
    >>> density(G)(X)
    exp(-Trace(X**2))/(2*pi**2)
    """
    sym, dim = _symbol_converter(sym), _sympify(dim)
    model = GaussianUnitaryEnsembleModel(sym, dim)
    rmp = RandomMatrixPSpace(sym, model=model)
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)

def GaussianOrthogonalEnsemble(sym, dim):
    """
    Represents Gaussian Orthogonal Ensembles.

    Examples
    ========

    >>> from sympy.stats import GaussianOrthogonalEnsemble as GOE, density
    >>> from sympy import MatrixSymbol
    >>> G = GOE('U', 2)
    >>> X = MatrixSymbol('X', 2, 2)
    >>> density(G)(X)
    exp(-Trace(X**2)/2)/Integral(exp(-Trace(_H**2)/2), _H)
    """
    sym, dim = _symbol_converter(sym), _sympify(dim)
    model = GaussianOrthogonalEnsembleModel(sym, dim)
    rmp = RandomMatrixPSpace(sym, model=model)
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)

def GaussianSymplecticEnsemble(sym, dim):
    """
    Represents Gaussian Symplectic Ensembles.

    Examples
    ========

    >>> from sympy.stats import GaussianSymplecticEnsemble as GSE, density
    >>> from sympy import MatrixSymbol
    >>> G = GSE('U', 2)
    >>> X = MatrixSymbol('X', 2, 2)
    >>> density(G)(X)
    exp(-2*Trace(X**2))/Integral(exp(-2*Trace(_H**2)), _H)
    """
    sym, dim = _symbol_converter(sym), _sympify(dim)
    model = GaussianSymplecticEnsembleModel(sym, dim)
    rmp = RandomMatrixPSpace(sym, model=model)
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)

class CircularEnsembleModel(RandomMatrixEnsembleModel):
    """
    Abstract class for Circular ensembles.
    Contains the properties and methods
    common to all the circular ensembles.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Circular_ensemble
    """
    def density(self, expr):
        # TODO : Add support for Lie groups(as extensions of sympy.diffgeom)
        #        and define measures on them
        raise NotImplementedError("Support for Haar measure hasn't been "
                                  "implemented yet, therefore the density of "
                                  "%s cannot be computed."%(self))

    def _compute_joint_eigen_distribution(self, beta):
        """
        Helper function to compute the joint distribution of phases
        of the complex eigen values of matrices belonging to any
        circular ensembles.
        """
        n = self.dimension
        Zbn = ((2*pi)**n)*(gamma(beta*n/2 + 1)/S(gamma(beta/2 + 1))**n)
        t = IndexedBase('t')
        i, j, k = (Dummy('i', integer=True), Dummy('j', integer=True),
                   Dummy('k', integer=True))
        syms = ArrayComprehension(t[i], (i, 1, n)).doit()
        f = Product(Product(Abs(exp(I*t[k]) - exp(I*t[j]))**beta, (j, k + 1, n)).doit(),
                    (k, 1, n - 1)).doit()
        return Lambda(tuple(syms), f/Zbn)

class CircularUnitaryEnsembleModel(CircularEnsembleModel):
    def joint_eigen_distribution(self):
        return self._compute_joint_eigen_distribution(S(2))

class CircularOrthogonalEnsembleModel(CircularEnsembleModel):
    def joint_eigen_distribution(self):
        return self._compute_joint_eigen_distribution(S.One)

class CircularSymplecticEnsembleModel(CircularEnsembleModel):
    def joint_eigen_distribution(self):
        return self._compute_joint_eigen_distribution(S(4))

def CircularEnsemble(sym, dim):
    sym, dim = _symbol_converter(sym), _sympify(dim)
    model = CircularEnsembleModel(sym, dim)
    rmp = RandomMatrixPSpace(sym, model=model)
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)

def CircularUnitaryEnsemble(sym, dim):
    """
    Represents Circular Unitary Ensembles.

    Examples
    ========

    >>> from sympy.stats import CircularUnitaryEnsemble as CUE
    >>> from sympy.stats import joint_eigen_distribution
    >>> C = CUE('U', 1)
    >>> joint_eigen_distribution(C)
    Lambda(t[1], Product(Abs(exp(I*t[_j]) - exp(I*t[_k]))**2, (_j, _k + 1, 1), (_k, 1, 0))/(2*pi))

    Note
    ====

    As can be seen above in the example, density of CiruclarUnitaryEnsemble
    is not evaluated because the exact definition is based on haar measure of
    unitary group which is not unique.
    """
    sym, dim = _symbol_converter(sym), _sympify(dim)
    model = CircularUnitaryEnsembleModel(sym, dim)
    rmp = RandomMatrixPSpace(sym, model=model)
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)

def CircularOrthogonalEnsemble(sym, dim):
    """
    Represents Circular Orthogonal Ensembles.

    Examples
    ========

    >>> from sympy.stats import CircularOrthogonalEnsemble as COE
    >>> from sympy.stats import joint_eigen_distribution
    >>> C = COE('O', 1)
    >>> joint_eigen_distribution(C)
    Lambda(t[1], Product(Abs(exp(I*t[_j]) - exp(I*t[_k])), (_j, _k + 1, 1), (_k, 1, 0))/(2*pi))

    Note
    ====

    As can be seen above in the example, density of CiruclarOrthogonalEnsemble
    is not evaluated because the exact definition is based on haar measure of
    unitary group which is not unique.
    """
    sym, dim = _symbol_converter(sym), _sympify(dim)
    model = CircularOrthogonalEnsembleModel(sym, dim)
    rmp = RandomMatrixPSpace(sym, model=model)
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)

def CircularSymplecticEnsemble(sym, dim):
    """
    Represents Circular Symplectic Ensembles.

    Examples
    ========

    >>> from sympy.stats import CircularSymplecticEnsemble as CSE
    >>> from sympy.stats import joint_eigen_distribution
    >>> C = CSE('S', 1)
    >>> joint_eigen_distribution(C)
    Lambda(t[1], Product(Abs(exp(I*t[_j]) - exp(I*t[_k]))**4, (_j, _k + 1, 1), (_k, 1, 0))/(2*pi))

    Note
    ====

    As can be seen above in the example, density of CiruclarSymplecticEnsemble
    is not evaluated because the exact definition is based on haar measure of
    unitary group which is not unique.
    """
    sym, dim = _symbol_converter(sym), _sympify(dim)
    model = CircularSymplecticEnsembleModel(sym, dim)
    rmp = RandomMatrixPSpace(sym, model=model)
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)

def joint_eigen_distribution(mat):
    """
    For obtaining joint probability distribution
    of eigen values of random matrix.

    Parameters
    ==========

    mat: RandomMatrixSymbol
        The matrix symbol whose eigen values are to be considered.

    Returns
    =======

    Lambda

    Examples
    ========

    >>> from sympy.stats import GaussianUnitaryEnsemble as GUE
    >>> from sympy.stats import joint_eigen_distribution
    >>> U = GUE('U', 2)
    >>> joint_eigen_distribution(U)
    Lambda((l[1], l[2]), exp(-l[1]**2 - l[2]**2)*Product(Abs(l[_i] - l[_j])**2, (_j, _i + 1, 2), (_i, 1, 1))/pi)
    """
    if not isinstance(mat, RandomMatrixSymbol):
        raise ValueError("%s is not of type, RandomMatrixSymbol."%(mat))
    return mat.pspace.model.joint_eigen_distribution()

def JointEigenDistribution(mat):
    """
    Creates joint distribution of eigen values of matrices with random
    expressions.

    Parameters
    ==========

    mat: Matrix
        The matrix under consideration.

    Returns
    =======

    JointDistributionHandmade

    Examples
    ========

    >>> from sympy.stats import Normal, JointEigenDistribution
    >>> from sympy import Matrix
    >>> A = [[Normal('A00', 0, 1), Normal('A01', 0, 1)],
    ... [Normal('A10', 0, 1), Normal('A11', 0, 1)]]
    >>> JointEigenDistribution(Matrix(A))
    JointDistributionHandmade(-sqrt(A00**2 - 2*A00*A11 + 4*A01*A10 + A11**2)/2
    + A00/2 + A11/2, sqrt(A00**2 - 2*A00*A11 + 4*A01*A10 + A11**2)/2 + A00/2 + A11/2)

    """
    eigenvals = mat.eigenvals(multiple=True)
    if not all(is_random(eigenval) for eigenval in set(eigenvals)):
        raise ValueError("Eigen values do not have any random expression, "
                         "joint distribution cannot be generated.")
    return JointDistributionHandmade(*eigenvals)

def level_spacing_distribution(mat):
    """
    For obtaining distribution of level spacings.

    Parameters
    ==========

    mat: RandomMatrixSymbol
        The random matrix symbol whose eigen values are
        to be considered for finding the level spacings.

    Returns
    =======

    Lambda

    Examples
    ========

    >>> from sympy.stats import GaussianUnitaryEnsemble as GUE
    >>> from sympy.stats import level_spacing_distribution
    >>> U = GUE('U', 2)
    >>> level_spacing_distribution(U)
    Lambda(_s, 32*_s**2*exp(-4*_s**2/pi)/pi**2)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Random_matrix#Distribution_of_level_spacings
    """
    return mat.pspace.model.level_spacing_distribution()
<|endoftext|>"""
Main Random Variables Module

Defines abstract random variable type.
Contains interfaces for probability space object (PSpace) as well as standard
operators, P, E, sample, density, where, quantile

See Also
========

sympy.stats.crv
sympy.stats.frv
sympy.stats.rv_interface
"""

from __future__ import annotations
from functools import singledispatch
from math import prod

from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import (Function, Lambda)
from sympy.core.logic import fuzzy_and
from sympy.core.mul import Mul
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.functions.special.delta_functions import DiracDelta
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.logic.boolalg import (And, Or)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.tensor.indexed import Indexed
from sympy.utilities.lambdify import lambdify
from sympy.core.relational import Relational
from sympy.core.sympify import _sympify
from sympy.sets.sets import FiniteSet, ProductSet, Intersection
from sympy.solvers.solveset import solveset
from sympy.external import import_module
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable


__doctest_requires__ = {('sample',): ['scipy']}


x = Symbol('x')


@singledispatch
def is_random(x):
    return False


@is_random.register(Basic)
def _(x):
    atoms = x.free_symbols
    return any(is_random(i) for i in atoms)


class RandomDomain(Basic):
    """
    Represents a set of variables and the values which they can take.

    See Also
    ========

    sympy.stats.crv.ContinuousDomain
    sympy.stats.frv.FiniteDomain
    """

    is_ProductDomain = False
    is_Finite = False
    is_Continuous = False
    is_Discrete = False

    def __new__(cls, symbols, *args):
        symbols = FiniteSet(*symbols)
        return Basic.__new__(cls, symbols, *args)

    @property
    def symbols(self):
        return self.args[0]

    @property
    def set(self):
        return self.args[1]

    def __contains__(self, other):
        raise NotImplementedError()

    def compute_expectation(self, expr):
        raise NotImplementedError()


class SingleDomain(RandomDomain):
    """
    A single variable and its domain.

    See Also
    ========

    sympy.stats.crv.SingleContinuousDomain
    sympy.stats.frv.SingleFiniteDomain
    """
    def __new__(cls, symbol, set):
        assert symbol.is_Symbol
        return Basic.__new__(cls, symbol, set)

    @property
    def symbol(self):
        return self.args[0]

    @property
    def symbols(self):
        return FiniteSet(self.symbol)

    def __contains__(self, other):
        if len(other) != 1:
            return False
        sym, val = tuple(other)[0]
        return self.symbol == sym and val in self.set


class MatrixDomain(RandomDomain):
    """
    A Random Matrix variable and its domain.

    """
    def __new__(cls, symbol, set):
        symbol, set = _symbol_converter(symbol), _sympify(set)
        return Basic.__new__(cls, symbol, set)

    @property
    def symbol(self):
        return self.args[0]

    @property
    def symbols(self):
        return FiniteSet(self.symbol)


class ConditionalDomain(RandomDomain):
    """
    A RandomDomain with an attached condition.

    See Also
    ========

    sympy.stats.crv.ConditionalContinuousDomain
    sympy.stats.frv.ConditionalFiniteDomain
    """
    def __new__(cls, fulldomain, condition):
        condition = condition.xreplace({rs: rs.symbol
            for rs in random_symbols(condition)})
        return Basic.__new__(cls, fulldomain, condition)

    @property
    def symbols(self):
        return self.fulldomain.symbols

    @property
    def fulldomain(self):
        return self.args[0]

    @property
    def condition(self):
        return self.args[1]

    @property
    def set(self):
        raise NotImplementedError("Set of Conditional Domain not Implemented")

    def as_boolean(self):
        return And(self.fulldomain.as_boolean(), self.condition)


class PSpace(Basic):
    """
    A Probability Space.

    Explanation
    ===========

    Probability Spaces encode processes that equal different values
    probabilistically. These underly Random Symbols which occur in SymPy
    expressions and contain the mechanics to evaluate statistical statements.

    See Also
    ========

    sympy.stats.crv.ContinuousPSpace
    sympy.stats.frv.FinitePSpace
    """

    is_Finite: bool | None = None  # Fails test if not set to None
    is_Continuous: bool | None = None  # Fails test if not set to None
    is_Discrete: bool | None = None  # Fails test if not set to None
    is_real: bool | None

    @property
    def domain(self):
        return self.args[0]

    @property
    def density(self):
        return self.args[1]

    @property
    def values(self):
        return frozenset(RandomSymbol(sym, self) for sym in self.symbols)

    @property
    def symbols(self):
        return self.domain.symbols

    def where(self, condition):
        raise NotImplementedError()

    def compute_density(self, expr):
        raise NotImplementedError()

    def sample(self, size=(), library='scipy', seed=None):
        raise NotImplementedError()

    def probability(self, condition):
        raise NotImplementedError()

    def compute_expectation(self, expr):
        raise NotImplementedError()


class SinglePSpace(PSpace):
    """
    Represents the probabilities of a set of random events that can be
    attributed to a single variable/symbol.
    """
    def __new__(cls, s, distribution):
        s = _symbol_converter(s)
        return Basic.__new__(cls, s, distribution)

    @property
    def value(self):
        return RandomSymbol(self.symbol, self)

    @property
    def symbol(self):
        return self.args[0]

    @property
    def distribution(self):
        return self.args[1]

    @property
    def pdf(self):
        return self.distribution.pdf(self.symbol)


class RandomSymbol(Expr):
    """
    Random Symbols represent ProbabilitySpaces in SymPy Expressions.
    In principle they can take on any value that their symbol can take on
    within the associated PSpace with probability determined by the PSpace
    Density.

    Explanation
    ===========

    Random Symbols contain pspace and symbol properties.
    The pspace property points to the represented Probability Space
    The symbol is a standard SymPy Symbol that is used in that probability space
    for example in defining a density.

    You can form normal SymPy expressions using RandomSymbols and operate on
    those expressions with the Functions

    E - Expectation of a random expression
    P - Probability of a condition
    density - Probability Density of an expression
    given - A new random expression (with new random symbols) given a condition

    An object of the RandomSymbol type should almost never be created by the
    user. They tend to be created instead by the PSpace class's value method.
    Traditionally a user does not even do this but instead calls one of the
    convenience functions Normal, Exponential, Coin, Die, FiniteRV, etc....
    """

    def __new__(cls, symbol, pspace=None):
        from sympy.stats.joint_rv import JointRandomSymbol
        if pspace is None:
            # Allow single arg, representing pspace == PSpace()
            pspace = PSpace()
        symbol = _symbol_converter(symbol)
        if not isinstance(pspace, PSpace):
            raise TypeError("pspace variable should be of type PSpace")
        if cls == JointRandomSymbol and isinstance(pspace, SinglePSpace):
            cls = RandomSymbol
        return Basic.__new__(cls, symbol, pspace)

    is_finite = True
    is_symbol = True
    is_Atom = True

    _diff_wrt = True

    pspace = property(lambda self: self.args[1])
    symbol = property(lambda self: self.args[0])
    name   = property(lambda self: self.symbol.name)

    def _eval_is_positive(self):
        return self.symbol.is_positive

    def _eval_is_integer(self):
        return self.symbol.is_integer

    def _eval_is_real(self):
        return self.symbol.is_real or self.pspace.is_real

    @property
    def is_commutative(self):
        return self.symbol.is_commutative

    @property
    def free_symbols(self):
        return {self}

class RandomIndexedSymbol(RandomSymbol):

    def __new__(cls, idx_obj, pspace=None):
        if pspace is None:
            # Allow single arg, representing pspace == PSpace()
            pspace = PSpace()
        if not isinstance(idx_obj, (Indexed, Function)):
            raise TypeError("An Function or Indexed object is expected not %s"%(idx_obj))
        return Basic.__new__(cls, idx_obj, pspace)

    symbol = property(lambda self: self.args[0])
    name = property(lambda self: str(self.args[0]))

    @property
    def key(self):
        if isinstance(self.symbol, Indexed):
            return self.symbol.args[1]
        elif isinstance(self.symbol, Function):
            return self.symbol.args[0]

    @property
    def free_symbols(self):
        if self.key.free_symbols:
            free_syms = self.key.free_symbols
            free_syms.add(self)
            return free_syms
        return {self}

    @property
    def pspace(self):
        return self.args[1]

class RandomMatrixSymbol(RandomSymbol, MatrixSymbol): # type: ignore
    def __new__(cls, symbol, n, m, pspace=None):
        n, m = _sympify(n), _sympify(m)
        symbol = _symbol_converter(symbol)
        if pspace is None:
            # Allow single arg, representing pspace == PSpace()
            pspace = PSpace()
        return Basic.__new__(cls, symbol, n, m, pspace)

    symbol = property(lambda self: self.args[0])
    pspace = property(lambda self: self.args[3])


class ProductPSpace(PSpace):
    """
    Abstract class for representing probability spaces with multiple random
    variables.

    See Also
    ========

    sympy.stats.rv.IndependentProductPSpace
    sympy.stats.joint_rv.JointPSpace
    """
    pass

class IndependentProductPSpace(ProductPSpace):
    """
    A probability space resulting from the merger of two independent probability
    spaces.

    Often created using the function, pspace.
    """

    def __new__(cls, *spaces):
        rs_space_dict = {}
        for space in spaces:
            for value in space.values:
                rs_space_dict[value] = space

        symbols = FiniteSet(*[val.symbol for val in rs_space_dict.keys()])

        # Overlapping symbols
        from sympy.stats.joint_rv import MarginalDistribution
        from sympy.stats.compound_rv import CompoundDistribution
        if len(symbols) < sum(len(space.symbols) for space in spaces if not
         isinstance(space.distribution, (
            CompoundDistribution, MarginalDistribution))):
            raise ValueError("Overlapping Random Variables")

        if all(space.is_Finite for space in spaces):
            from sympy.stats.frv import ProductFinitePSpace
            cls = ProductFinitePSpace

        obj = Basic.__new__(cls, *FiniteSet(*spaces))

        return obj

    @property
    def pdf(self):
        p = Mul(*[space.pdf for space in self.spaces])
        return p.subs({rv: rv.symbol for rv in self.values})

    @property
    def rs_space_dict(self):
        d = {}
        for space in self.spaces:
            for value in space.values:
                d[value] = space
        return d

    @property
    def symbols(self):
        return FiniteSet(*[val.symbol for val in self.rs_space_dict.keys()])

    @property
    def spaces(self):
        return FiniteSet(*self.args)

    @property
    def values(self):
        return sumsets(space.values for space in self.spaces)

    def compute_expectation(self, expr, rvs=None, evaluate=False, **kwargs):
        rvs = rvs or self.values
        rvs = frozenset(rvs)
        for space in self.spaces:
            expr = space.compute_expectation(expr, rvs & space.values, evaluate=False, **kwargs)
        if evaluate and hasattr(expr, 'doit'):
            return expr.doit(**kwargs)
        return expr

    @property
    def domain(self):
        return ProductDomain(*[space.domain for space in self.spaces])

    @property
    def density(self):
        raise NotImplementedError("Density not available for ProductSpaces")

    def sample(self, size=(), library='scipy', seed=None):
        return {k: v for space in self.spaces
            for k, v in space.sample(size=size, library=library, seed=seed).items()}


    def probability(self, condition, **kwargs):
        cond_inv = False
        if isinstance(condition, Ne):
            condition = Eq(condition.args[0], condition.args[1])
            cond_inv = True
        elif isinstance(condition, And): # they are independent
            return Mul(*[self.probability(arg) for arg in condition.args])
        elif isinstance(condition, Or): # they are independent
            return Add(*[self.probability(arg) for arg in condition.args])
        expr = condition.lhs - condition.rhs
        rvs = random_symbols(expr)
        dens = self.compute_density(expr)
        if any(pspace(rv).is_Continuous for rv in rvs):
            from sympy.stats.crv import SingleContinuousPSpace
            from sympy.stats.crv_types import ContinuousDistributionHandmade
            if expr in self.values:
                # Marginalize all other random symbols out of the density
                randomsymbols = tuple(set(self.values) - frozenset([expr]))
                symbols = tuple(rs.symbol for rs in randomsymbols)
                pdf = self.domain.integrate(self.pdf, symbols, **kwargs)
                return Lambda(expr.symbol, pdf)
            dens = ContinuousDistributionHandmade(dens)
            z = Dummy('z', real=True)
            space = SingleContinuousPSpace(z, dens)
            result = space.probability(condition.__class__(space.value, 0))
        else:
            from sympy.stats.drv import SingleDiscretePSpace
            from sympy.stats.drv_types import DiscreteDistributionHandmade
            dens = DiscreteDistributionHandmade(dens)
            z = Dummy('z', integer=True)
            space = SingleDiscretePSpace(z, dens)
            result = space.probability(condition.__class__(space.value, 0))
        return result if not cond_inv else S.One - result

    def compute_density(self, expr, **kwargs):
        rvs = random_symbols(expr)
        if any(pspace(rv).is_Continuous for rv in rvs):
            z = Dummy('z', real=True)
            expr = self.compute_expectation(DiracDelta(expr - z),
             **kwargs)
        else:
            z = Dummy('z', integer=True)
            expr = self.compute_expectation(KroneckerDelta(expr, z),
             **kwargs)
        return Lambda(z, expr)

    def compute_cdf(self, expr, **kwargs):
        raise ValueError("CDF not well defined on multivariate expressions")

    def conditional_space(self, condition, normalize=True, **kwargs):
        rvs = random_symbols(condition)
        condition = condition.xreplace({rv: rv.symbol for rv in self.values})
        pspaces = [pspace(rv) for rv in rvs]
        if any(ps.is_Continuous for ps in pspaces):
            from sympy.stats.crv import (ConditionalContinuousDomain,
                ContinuousPSpace)
            space = ContinuousPSpace
            domain = ConditionalContinuousDomain(self.domain, condition)
        elif any(ps.is_Discrete for ps in pspaces):
            from sympy.stats.drv import (ConditionalDiscreteDomain,
                DiscretePSpace)
            space = DiscretePSpace
            domain = ConditionalDiscreteDomain(self.domain, condition)
        elif all(ps.is_Finite for ps in pspaces):
            from sympy.stats.frv import FinitePSpace
            return FinitePSpace.conditional_space(self, condition)
        if normalize:
            replacement  = {rv: Dummy(str(rv)) for rv in self.symbols}
            norm = domain.compute_expectation(self.pdf, **kwargs)
            pdf = self.pdf / norm.xreplace(replacement)
            # XXX: Converting symbols from set to tuple. The order matters to
            # Lambda though so we shouldn't be starting with a set here...
            density = Lambda(tuple(domain.symbols), pdf)

        return space(domain, density)

class ProductDomain(RandomDomain):
    """
    A domain resulting from the merger of two independent domains.

    See Also
    ========
    sympy.stats.crv.ProductContinuousDomain
    sympy.stats.frv.ProductFiniteDomain
    """
    is_ProductDomain = True

    def __new__(cls, *domains):
        # Flatten any product of products
        domains2 = []
        for domain in domains:
            if not domain.is_ProductDomain:
                domains2.append(domain)
            else:
                domains2.extend(domain.domains)
        domains2 = FiniteSet(*domains2)

        if all(domain.is_Finite for domain in domains2):
            from sympy.stats.frv import ProductFiniteDomain
            cls = ProductFiniteDomain
        if all(domain.is_Continuous for domain in domains2):
            from sympy.stats.crv import ProductContinuousDomain
            cls = ProductContinuousDomain
        if all(domain.is_Discrete for domain in domains2):
            from sympy.stats.drv import ProductDiscreteDomain
            cls = ProductDiscreteDomain

        return Basic.__new__(cls, *domains2)

    @property
    def sym_domain_dict(self):
        return {symbol: domain for domain in self.domains
                                     for symbol in domain.symbols}

    @property
    def symbols(self):
        return FiniteSet(*[sym for domain in self.domains
                               for sym    in domain.symbols])

    @property
    def domains(self):
        return self.args

    @property
    def set(self):
        return ProductSet(*(domain.set for domain in self.domains))

    def __contains__(self, other):
        # Split event into each subdomain
        for domain in self.domains:
            # Collect the parts of this event which associate to this domain
            elem = frozenset([item for item in other
                              if sympify(domain.symbols.contains(item[0]))
                              is S.true])
            # Test this sub-event
            if elem not in domain:
                return False
        # All subevents passed
        return True

    def as_boolean(self):
        return And(*[domain.as_boolean() for domain in self.domains])


def random_symbols(expr):
    """
    Returns all RandomSymbols within a SymPy Expression.
    """
    atoms = getattr(expr, 'atoms', None)
    if atoms is not None:
        comp = lambda rv: rv.symbol.name
        l = list(atoms(RandomSymbol))
        return sorted(l, key=comp)
    else:
        return []


def pspace(expr):
    """
    Returns the underlying Probability Space of a random expression.

    For internal use.

    Examples
    ========

    >>> from sympy.stats import pspace, Normal
    >>> X = Normal('X', 0, 1)
    >>> pspace(2*X + 1) == X.pspace
    True
    """
    expr = sympify(expr)
    if isinstance(expr, RandomSymbol) and expr.pspace is not None:
        return expr.pspace
    if expr.has(RandomMatrixSymbol):
        rm = list(expr.atoms(RandomMatrixSymbol))[0]
        return rm.pspace

    rvs = random_symbols(expr)
    if not rvs:
        raise ValueError("Expression containing Random Variable expected, not %s" % (expr))
    # If only one space present
    if all(rv.pspace == rvs[0].pspace for rv in rvs):
        return rvs[0].pspace
    from sympy.stats.compound_rv import CompoundPSpace
    from sympy.stats.stochastic_process import StochasticPSpace
    for rv in rvs:
        if isinstance(rv.pspace, (CompoundPSpace, StochasticPSpace)):
            return rv.pspace
    # Otherwise make a product space
    return IndependentProductPSpace(*[rv.pspace for rv in rvs])


def sumsets(sets):
    """
    Union of sets
    """
    return frozenset().union(*sets)


def rs_swap(a, b):
    """
    Build a dictionary to swap RandomSymbols based on their underlying symbol.

    i.e.
    if    ``X = ('x', pspace1)``
    and   ``Y = ('x', pspace2)``
    then ``X`` and ``Y`` match and the key, value pair
    ``{X:Y}`` will appear in the result

    Inputs: collections a and b of random variables which share common symbols
    Output: dict mapping RVs in a to RVs in b
    """
    d = {}
    for rsa in a:
        d[rsa] = [rsb for rsb in b if rsa.symbol == rsb.symbol][0]
    return d


def given(expr, condition=None, **kwargs):
    r""" Conditional Random Expression.

    Explanation
    ===========

    From a random expression and a condition on that expression creates a new
    probability space from the condition and returns the same expression on that
    conditional probability space.

    Examples
    ========

    >>> from sympy.stats import given, density, Die
    >>> X = Die('X', 6)
    >>> Y = given(X, X > 3)
    >>> density(Y).dict
    {4: 1/3, 5: 1/3, 6: 1/3}

    Following convention, if the condition is a random symbol then that symbol
    is considered fixed.

    >>> from sympy.stats import Normal
    >>> from sympy import pprint
    >>> from sympy.abc import z

    >>> X = Normal('X', 0, 1)
    >>> Y = Normal('Y', 0, 1)
    >>> pprint(density(X + Y, Y)(z), use_unicode=False)
                    2
           -(-Y + z)
           -----------
      ___       2
    \/ 2 *e
    ------------------
             ____
         2*\/ pi
    """

    if not is_random(condition) or pspace_independent(expr, condition):
        return expr

    if isinstance(condition, RandomSymbol):
        condition = Eq(condition, condition.symbol)

    condsymbols = random_symbols(condition)
    if (isinstance(condition, Eq) and len(condsymbols) == 1 and
        not isinstance(pspace(expr).domain, ConditionalDomain)):
        rv = tuple(condsymbols)[0]

        results = solveset(condition, rv)
        if isinstance(results, Intersection) and S.Reals in results.args:
            results = list(results.args[1])

        sums = 0
        for res in results:
            temp = expr.subs(rv, res)
            if temp == True:
                return True
            if temp != False:
                # XXX: This seems nonsensical but preserves existing behaviour
                # after the change that Relational is no longer a subclass of
                # Expr. Here expr is sometimes Relational and sometimes Expr
                # but we are trying to add them with +=. This needs to be
                # fixed somehow.
                if sums == 0 and isinstance(expr, Relational):
                    sums = expr.subs(rv, res)
                else:
                    sums += expr.subs(rv, res)
        if sums == 0:
            return False
        return sums

    # Get full probability space of both the expression and the condition
    fullspace = pspace(Tuple(expr, condition))
    # Build new space given the condition
    space = fullspace.conditional_space(condition, **kwargs)
    # Dictionary to swap out RandomSymbols in expr with new RandomSymbols
    # That point to the new conditional space
    swapdict = rs_swap(fullspace.values, space.values)
    # Swap random variables in the expression
    expr = expr.xreplace(swapdict)
    return expr


def expectation(expr, condition=None, numsamples=None, evaluate=True, **kwargs):
    """
    Returns the expected value of a random expression.

    Parameters
    ==========

    expr : Expr containing RandomSymbols
        The expression of which you want to compute the expectation value
    given : Expr containing RandomSymbols
        A conditional expression. E(X, X>0) is expectation of X given X > 0
    numsamples : int
        Enables sampling and approximates the expectation with this many samples
    evalf : Bool (defaults to True)
        If sampling return a number rather than a complex expression
    evaluate : Bool (defaults to True)
        In case of continuous systems return unevaluated integral

    Examples
    ========

    >>> from sympy.stats import E, Die
    >>> X = Die('X', 6)
    >>> E(X)
    7/2
    >>> E(2*X + 1)
    8

    >>> E(X, X > 3) # Expectation of X given that it is above 3
    5
    """

    if not is_random(expr):  # expr isn't random?
        return expr
    kwargs['numsamples'] = numsamples
    from sympy.stats.symbolic_probability import Expectation
    if evaluate:
        return Expectation(expr, condition).doit(**kwargs)
    return Expectation(expr, condition)


def probability(condition, given_condition=None, numsamples=None,
                evaluate=True, **kwargs):
    """
    Probability that a condition is true, optionally given a second condition.

    Parameters
    ==========

    condition : Combination of Relationals containing RandomSymbols
        The condition of which you want to compute the probability
    given_condition : Combination of Relationals containing RandomSymbols
        A conditional expression. P(X > 1, X > 0) is expectation of X > 1
        given X > 0
    numsamples : int
        Enables sampling and approximates the probability with this many samples
    evaluate : Bool (defaults to True)
        In case of continuous systems return unevaluated integral

    Examples
    ========

    >>> from sympy.stats import P, Die
    >>> from sympy import Eq
    >>> X, Y = Die('X', 6), Die('Y', 6)
    >>> P(X > 3)
    1/2
    >>> P(Eq(X, 5), X > 2) # Probability that X == 5 given that X > 2
    1/4
    >>> P(X > Y)
    5/12
    """

    kwargs['numsamples'] = numsamples
    from sympy.stats.symbolic_probability import Probability
    if evaluate:
        return Probability(condition, given_condition).doit(**kwargs)
    return Probability(condition, given_condition)


class Density(Basic):
    expr = property(lambda self: self.args[0])

    def __new__(cls, expr, condition = None):
        expr = _sympify(expr)
        if condition is None:
            obj = Basic.__new__(cls, expr)
        else:
            condition = _sympify(condition)
            obj = Basic.__new__(cls, expr, condition)
        return obj

    @property
    def condition(self):
        if len(self.args) > 1:
            return self.args[1]
        else:
            return None

    def doit(self, evaluate=True, **kwargs):
        from sympy.stats.random_matrix import RandomMatrixPSpace
        from sympy.stats.joint_rv import JointPSpace
        from sympy.stats.matrix_distributions import MatrixPSpace
        from sympy.stats.compound_rv import CompoundPSpace
        from sympy.stats.frv import SingleFiniteDistribution
        expr, condition = self.expr, self.condition

        if isinstance(expr, SingleFiniteDistribution):
            return expr.dict
        if condition is not None:
            # Recompute on new conditional expr
            expr = given(expr, condition, **kwargs)
        if not random_symbols(expr):
            return Lambda(x, DiracDelta(x - expr))
        if isinstance(expr, RandomSymbol):
            if isinstance(expr.pspace, (SinglePSpace, JointPSpace, MatrixPSpace)) and \
                hasattr(expr.pspace, 'distribution'):
                return expr.pspace.distribution
            elif isinstance(expr.pspace, RandomMatrixPSpace):
                return expr.pspace.model
        if isinstance(pspace(expr), CompoundPSpace):
            kwargs['compound_evaluate'] = evaluate
        result = pspace(expr).compute_density(expr, **kwargs)

        if evaluate and hasattr(result, 'doit'):
            return result.doit()
        else:
            return result


def density(expr, condition=None, evaluate=True, numsamples=None, **kwargs):
    """
    Probability density of a random expression, optionally given a second
    condition.

    Explanation
    ===========

    This density will take on different forms for different types of
    probability spaces. Discrete variables produce Dicts. Continuous
    variables produce Lambdas.

    Parameters
    ==========

    expr : Expr containing RandomSymbols
        The expression of which you want to compute the density value
    condition : Relational containing RandomSymbols
        A conditional expression. density(X > 1, X > 0) is density of X > 1
        given X > 0
    numsamples : int
        Enables sampling and approximates the density with this many samples

    Examples
    ========

    >>> from sympy.stats import density, Die, Normal
    >>> from sympy import Symbol

    >>> x = Symbol('x')
    >>> D = Die('D', 6)
    >>> X = Normal(x, 0, 1)

    >>> density(D).dict
    {1: 1/6, 2: 1/6, 3: 1/6, 4: 1/6, 5: 1/6, 6: 1/6}
    >>> density(2*D).dict
    {2: 1/6, 4: 1/6, 6: 1/6, 8: 1/6, 10: 1/6, 12: 1/6}
    >>> density(X)(x)
    sqrt(2)*exp(-x**2/2)/(2*sqrt(pi))
    """

    if numsamples:
        return sampling_density(expr, condition, numsamples=numsamples,
                **kwargs)

    return Density(expr, condition).doit(evaluate=evaluate, **kwargs)


def cdf(expr, condition=None, evaluate=True, **kwargs):
    """
    Cumulative Distribution Function of a random expression.

    optionally given a second condition.

    Explanation
    ===========

    This density will take on different forms for different types of
    probability spaces.
    Discrete variables produce Dicts.
    Continuous variables produce Lambdas.

    Examples
    ========

    >>> from sympy.stats import density, Die, Normal, cdf

    >>> D = Die('D', 6)
    >>> X = Normal('X', 0, 1)

    >>> density(D).dict
    {1: 1/6, 2: 1/6, 3: 1/6, 4: 1/6, 5: 1/6, 6: 1/6}
    >>> cdf(D)
    {1: 1/6, 2: 1/3, 3: 1/2, 4: 2/3, 5: 5/6, 6: 1}
    >>> cdf(3*D, D > 2)
    {9: 1/4, 12: 1/2, 15: 3/4, 18: 1}

    >>> cdf(X)
    Lambda(_z, erf(sqrt(2)*_z/2)/2 + 1/2)
    """
    if condition is not None:  # If there is a condition
        # Recompute on new conditional expr
        return cdf(given(expr, condition, **kwargs), **kwargs)

    # Otherwise pass work off to the ProbabilitySpace
    result = pspace(expr).compute_cdf(expr, **kwargs)

    if evaluate and hasattr(result, 'doit'):
        return result.doit()
    else:
        return result


def characteristic_function(expr, condition=None, evaluate=True, **kwargs):
    """
    Characteristic function of a random expression, optionally given a second condition.

    Returns a Lambda.

    Examples
    ========

    >>> from sympy.stats import Normal, DiscreteUniform, Poisson, characteristic_function

    >>> X = Normal('X', 0, 1)
    >>> characteristic_function(X)
    Lambda(_t, exp(-_t**2/2))

    >>> Y = DiscreteUniform('Y', [1, 2, 7])
    >>> characteristic_function(Y)
    Lambda(_t, exp(7*_t*I)/3 + exp(2*_t*I)/3 + exp(_t*I)/3)

    >>> Z = Poisson('Z', 2)
    >>> characteristic_function(Z)
    Lambda(_t, exp(2*exp(_t*I) - 2))
    """
    if condition is not None:
        return characteristic_function(given(expr, condition, **kwargs), **kwargs)

    result = pspace(expr).compute_characteristic_function(expr, **kwargs)

    if evaluate and hasattr(result, 'doit'):
        return result.doit()
    else:
        return result

def moment_generating_function(expr, condition=None, evaluate=True, **kwargs):
    if condition is not None:
        return moment_generating_function(given(expr, condition, **kwargs), **kwargs)

    result = pspace(expr).compute_moment_generating_function(expr, **kwargs)

    if evaluate and hasattr(result, 'doit'):
        return result.doit()
    else:
        return result

def where(condition, given_condition=None, **kwargs):
    """
    Returns the domain where a condition is True.

    Examples
    ========

    >>> from sympy.stats import where, Die, Normal
    >>> from sympy import And

    >>> D1, D2 = Die('a', 6), Die('b', 6)
    >>> a, b = D1.symbol, D2.symbol
    >>> X = Normal('x', 0, 1)

    >>> where(X**2<1)
    Domain: (-1 < x) & (x < 1)

    >>> where(X**2<1).set
    Interval.open(-1, 1)

    >>> where(And(D1<=D2, D2<3))
    Domain: (Eq(a, 1) & Eq(b, 1)) | (Eq(a, 1) & Eq(b, 2)) | (Eq(a, 2) & Eq(b, 2))
    """
    if given_condition is not None:  # If there is a condition
        # Recompute on new conditional expr
        return where(given(condition, given_condition, **kwargs), **kwargs)

    # Otherwise pass work off to the ProbabilitySpace
    return pspace(condition).where(condition, **kwargs)


@doctest_depends_on(modules=('scipy',))
def sample(expr, condition=None, size=(), library='scipy',
           numsamples=1, seed=None, **kwargs):
    """
    A realization of the random expression.

    Parameters
    ==========

    expr : Expression of random variables
        Expression from which sample is extracted
    condition : Expr containing RandomSymbols
        A conditional expression
    size : int, tuple
        Represents size of each sample in numsamples
    library : str
        - 'scipy' : Sample using scipy
        - 'numpy' : Sample using numpy
        - 'pymc'  : Sample using PyMC

        Choose any of the available options to sample from as string,
        by default is 'scipy'
    numsamples : int
        Number of samples, each with size as ``size``.

        .. deprecated:: 1.9

        The ``numsamples`` parameter is deprecated and is only provided for
        compatibility with v1.8. Use a list comprehension or an additional
        dimension in ``size`` instead. See
        :ref:`deprecated-sympy-stats-numsamples` for details.

    seed :
        An object to be used as seed by the given external library for sampling `expr`.
        Following is the list of possible types of object for the supported libraries,

        - 'scipy': int, numpy.random.RandomState, numpy.random.Generator
        - 'numpy': int, numpy.random.RandomState, numpy.random.Generator
        - 'pymc': int

        Optional, by default None, in which case seed settings
        related to the given library will be used.
        No modifications to environment's global seed settings
        are done by this argument.

    Returns
    =======

    sample: float/list/numpy.ndarray
        one sample or a collection of samples of the random expression.

        - sample(X) returns float/numpy.float64/numpy.int64 object.
        - sample(X, size=int/tuple) returns numpy.ndarray object.

    Examples
    ========

    >>> from sympy.stats import Die, sample, Normal, Geometric
    >>> X, Y, Z = Die('X', 6), Die('Y', 6), Die('Z', 6) # Finite Random Variable
    >>> die_roll = sample(X + Y + Z)
    >>> die_roll # doctest: +SKIP
    3
    >>> N = Normal('N', 3, 4) # Continuous Random Variable
    >>> samp = sample(N)
    >>> samp in N.pspace.domain.set
    True
    >>> samp = sample(N, N>0)
    >>> samp > 0
    True
    >>> samp_list = sample(N, size=4)
    >>> [sam in N.pspace.domain.set for sam in samp_list]
    [True, True, True, True]
    >>> sample(N, size = (2,3)) # doctest: +SKIP
    array([[5.42519758, 6.40207856, 4.94991743],
       [1.85819627, 6.83403519, 1.9412172 ]])
    >>> G = Geometric('G', 0.5) # Discrete Random Variable
    >>> samp_list = sample(G, size=3)
    >>> samp_list # doctest: +SKIP
    [1, 3, 2]
    >>> [sam in G.pspace.domain.set for sam in samp_list]
    [True, True, True]
    >>> MN = Normal("MN", [3, 4], [[2, 1], [1, 2]]) # Joint Random Variable
    >>> samp_list = sample(MN, size=4)
    >>> samp_list # doctest: +SKIP
    [array([2.85768055, 3.38954165]),
     array([4.11163337, 4.3176591 ]),
     array([0.79115232, 1.63232916]),
     array([4.01747268, 3.96716083])]
    >>> [tuple(sam) in MN.pspace.domain.set for sam in samp_list]
    [True, True, True, True]

    .. versionchanged:: 1.7.0
        sample used to return an iterator containing the samples instead of value.

    .. versionchanged:: 1.9.0
        sample returns values or array of values instead of an iterator and numsamples is deprecated.

    """

    iterator = sample_iter(expr, condition, size=size, library=library,
                                                        numsamples=numsamples, seed=seed)

    if numsamples != 1:
        sympy_deprecation_warning(
            f"""
            The numsamples parameter to sympy.stats.sample() is deprecated.
            Either use a list comprehension, like

            [sample(...) for i in range({numsamples})]

            or add a dimension to size, like

            sample(..., size={(numsamples,) + size})
            """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-sympy-stats-numsamples",
        )
        return [next(iterator) for i in range(numsamples)]

    return next(iterator)


def quantile(expr, evaluate=True, **kwargs):
    r"""
    Return the :math:`p^{th}` order quantile of a probability distribution.

    Explanation
    ===========

    Quantile is defined as the value at which the probability of the random
    variable is less than or equal to the given probability.

    .. math::
        Q(p) = \inf\{x \in (-\infty, \infty) : p \le F(x)\}

    Examples
    ========

    >>> from sympy.stats import quantile, Die, Exponential
    >>> from sympy import Symbol, pprint
    >>> p = Symbol("p")

    >>> l = Symbol("lambda", positive=True)
    >>> X = Exponential("x", l)
    >>> quantile(X)(p)
    -log(1 - p)/lambda

    >>> D = Die("d", 6)
    >>> pprint(quantile(D)(p), use_unicode=False)
    /nan  for Or(p > 1, p < 0)
    |
    | 1       for p <= 1/6
    |
    | 2       for p <= 1/3
    |
    < 3       for p <= 1/2
    |
    | 4       for p <= 2/3
    |
    | 5       for p <= 5/6
    |
    \ 6        for p <= 1

    """
    result = pspace(expr).compute_quantile(expr, **kwargs)

    if evaluate and hasattr(result, 'doit'):
        return result.doit()
    else:
        return result

def sample_iter(expr, condition=None, size=(), library='scipy',
                    numsamples=S.Infinity, seed=None, **kwargs):

    """
    Returns an iterator of realizations from the expression given a condition.

    Parameters
    ==========

    expr: Expr
        Random expression to be realized
    condition: Expr, optional
        A conditional expression
    size : int, tuple
        Represents size of each sample in numsamples
    numsamples: integer, optional
        Length of the iterator (defaults to infinity)
    seed :
        An object to be used as seed by the given external library for sampling `expr`.
        Following is the list of possible types of object for the supported libraries,

        - 'scipy': int, numpy.random.RandomState, numpy.random.Generator
        - 'numpy': int, numpy.random.RandomState, numpy.random.Generator
        - 'pymc': int

        Optional, by default None, in which case seed settings
        related to the given library will be used.
        No modifications to environment's global seed settings
        are done by this argument.

    Examples
    ========

    >>> from sympy.stats import Normal, sample_iter
    >>> X = Normal('X', 0, 1)
    >>> expr = X*X + 3
    >>> iterator = sample_iter(expr, numsamples=3) # doctest: +SKIP
    >>> list(iterator) # doctest: +SKIP
    [12, 4, 7]

    Returns
    =======

    sample_iter: iterator object
        iterator object containing the sample/samples of given expr

    See Also
    ========

    sample
    sampling_P
    sampling_E

    """
    from sympy.stats.joint_rv import JointRandomSymbol
    if not import_module(library):
        raise ValueError("Failed to import %s" % library)

    if condition is not None:
        ps = pspace(Tuple(expr, condition))
    else:
        ps = pspace(expr)

    rvs = list(ps.values)
    if isinstance(expr, JointRandomSymbol):
        expr = expr.subs({expr: RandomSymbol(expr.symbol, expr.pspace)})
    else:
        sub = {}
        for arg in expr.args:
            if isinstance(arg, JointRandomSymbol):
                sub[arg] = RandomSymbol(arg.symbol, arg.pspace)
        expr = expr.subs(sub)

    def fn_subs(*args):
        return expr.subs(dict(zip(rvs, args)))

    def given_fn_subs(*args):
        if condition is not None:
            return condition.subs(dict(zip(rvs, args)))
        return False

    if library in ('pymc', 'pymc3'):
        # Currently unable to lambdify in pymc
        # TODO : Remove when lambdify accepts 'pymc' as module
        fn = lambdify(rvs, expr, **kwargs)
    else:
        fn = lambdify(rvs, expr, modules=library, **kwargs)


    if condition is not None:
        given_fn = lambdify(rvs, condition, **kwargs)

    def return_generator_infinite():
        count = 0
        _size = (1,)+((size,) if isinstance(size, int) else size)
        while count < numsamples:
            d = ps.sample(size=_size, library=library, seed=seed)  # a dictionary that maps RVs to values
            args = [d[rv][0] for rv in rvs]

            if condition is not None:  # Check that these values satisfy the condition
                # TODO: Replace the try-except block with only given_fn(*args)
                # once lambdify works with unevaluated SymPy objects.
                try:
                    gd = given_fn(*args)
                except (NameError, TypeError):
                    gd = given_fn_subs(*args)
                if gd != True and gd != False:
                    raise ValueError(
                        "Conditions must not contain free symbols")
                if not gd:  # If the values don't satisfy then try again
                    continue

            yield fn(*args)
            count += 1

    def return_generator_finite():
        faulty = True
        while faulty:
            d = ps.sample(size=(numsamples,) + ((size,) if isinstance(size, int) else size),
                          library=library, seed=seed) # a dictionary that maps RVs to values

            faulty = False
            count = 0
            while count < numsamples and not faulty:
                args = [d[rv][count] for rv in rvs]
                if condition is not None:  # Check that these values satisfy the condition
                    # TODO: Replace the try-except block with only given_fn(*args)
                    # once lambdify works with unevaluated SymPy objects.
                    try:
                        gd = given_fn(*args)
                    except (NameError, TypeError):
                        gd = given_fn_subs(*args)
                    if gd != True and gd != False:
                        raise ValueError(
                            "Conditions must not contain free symbols")
                    if not gd:  # If the values don't satisfy then try again
                        faulty = True

                count += 1

        count = 0
        while count < numsamples:
            args = [d[rv][count] for rv in rvs]
            # TODO: Replace the try-except block with only fn(*args)
            # once lambdify works with unevaluated SymPy objects.
            try:
                yield fn(*args)
            except (NameError, TypeError):
                yield fn_subs(*args)
            count += 1

    if numsamples is S.Infinity:
        return return_generator_infinite()

    return return_generator_finite()

def sample_iter_lambdify(expr, condition=None, size=(),
                         numsamples=S.Infinity, seed=None, **kwargs):

    return sample_iter(expr, condition=condition, size=size,
                       numsamples=numsamples, seed=seed, **kwargs)

def sample_iter_subs(expr, condition=None, size=(),
                     numsamples=S.Infinity, seed=None, **kwargs):

    return sample_iter(expr, condition=condition, size=size,
                       numsamples=numsamples, seed=seed, **kwargs)


def sampling_P(condition, given_condition=None, library='scipy', numsamples=1,
               evalf=True, seed=None, **kwargs):
    """
    Sampling version of P.

    See Also
    ========

    P
    sampling_E
    sampling_density

    """

    count_true = 0
    count_false = 0
    samples = sample_iter(condition, given_condition, library=library,
                          numsamples=numsamples, seed=seed, **kwargs)

    for sample in samples:
        if sample:
            count_true += 1
        else:
            count_false += 1

    result = S(count_true) / numsamples
    if evalf:
        return result.evalf()
    else:
        return result


def sampling_E(expr, given_condition=None, library='scipy', numsamples=1,
               evalf=True, seed=None, **kwargs):
    """
    Sampling version of E.

    See Also
    ========

    P
    sampling_P
    sampling_density
    """
    samples = list(sample_iter(expr, given_condition, library=library,
                          numsamples=numsamples, seed=seed, **kwargs))
    result = Add(*samples) / numsamples

    if evalf:
        return result.evalf()
    else:
        return result

def sampling_density(expr, given_condition=None, library='scipy',
                    numsamples=1, seed=None, **kwargs):
    """
    Sampling version of density.

    See Also
    ========
    density
    sampling_P
    sampling_E
    """

    results = {}
    for result in sample_iter(expr, given_condition, library=library,
                              numsamples=numsamples, seed=seed, **kwargs):
        results[result] = results.get(result, 0) + 1

    return results


def dependent(a, b):
    """
    Dependence of two random expressions.

    Two expressions are independent if knowledge of one does not change
    computations on the other.

    Examples
    ========

    >>> from sympy.stats import Normal, dependent, given
    >>> from sympy import Tuple, Eq

    >>> X, Y = Normal('X', 0, 1), Normal('Y', 0, 1)
    >>> dependent(X, Y)
    False
    >>> dependent(2*X + Y, -Y)
    True
    >>> X, Y = given(Tuple(X, Y), Eq(X + Y, 3))
    >>> dependent(X, Y)
    True

    See Also
    ========

    independent
    """
    if pspace_independent(a, b):
        return False

    z = Symbol('z', real=True)
    # Dependent if density is unchanged when one is given information about
    # the other
    return (density(a, Eq(b, z)) != density(a) or
            density(b, Eq(a, z)) != density(b))


def independent(a, b):
    """
    Independence of two random expressions.

    Two expressions are independent if knowledge of one does not change
    computations on the other.

    Examples
    ========

    >>> from sympy.stats import Normal, independent, given
    >>> from sympy import Tuple, Eq

    >>> X, Y = Normal('X', 0, 1), Normal('Y', 0, 1)
    >>> independent(X, Y)
    True
    >>> independent(2*X + Y, -Y)
    False
    >>> X, Y = given(Tuple(X, Y), Eq(X + Y, 3))
    >>> independent(X, Y)
    False

    See Also
    ========

    dependent
    """
    return not dependent(a, b)


def pspace_independent(a, b):
    """
    Tests for independence between a and b by checking if their PSpaces have
    overlapping symbols. This is a sufficient but not necessary condition for
    independence and is intended to be used internally.

    Notes
    =====

    pspace_independent(a, b) implies independent(a, b)
    independent(a, b) does not imply pspace_independent(a, b)
    """
    a_symbols = set(pspace(b).symbols)
    b_symbols = set(pspace(a).symbols)

    if len(set(random_symbols(a)).intersection(random_symbols(b))) != 0:
        return False

    if len(a_symbols.intersection(b_symbols)) == 0:
        return True
    return None


def rv_subs(expr, symbols=None):
    """
    Given a random expression replace all random variables with their symbols.

    If symbols keyword is given restrict the swap to only the symbols listed.
    """
    if symbols is None:
        symbols = random_symbols(expr)
    if not symbols:
        return expr
    swapdict = {rv: rv.symbol for rv in symbols}
    return expr.subs(swapdict)


class NamedArgsMixin:
    _argnames: tuple[str, ...] = ()

    def __getattr__(self, attr):
        try:
            return self.args[self._argnames.index(attr)]
        except ValueError:
            raise AttributeError("'%s' object has no attribute '%s'" % (
                type(self).__name__, attr))


class Distribution(Basic):

    def sample(self, size=(), library='scipy', seed=None):
        """ A random realization from the distribution """

        module = import_module(library)
        if library in {'scipy', 'numpy', 'pymc3', 'pymc'} and module is None:
            raise ValueError("Failed to import %s" % library)

        if library == 'scipy':
            # scipy does not require map as it can handle using custom distributions.
            # However, we will still use a map where we can.

            # TODO: do this for drv.py and frv.py if necessary.
            # TODO: add more distributions here if there are more
            # See links below referring to sections beginning with "A common parametrization..."
            # I will remove all these comments if everything is ok.

            from sympy.stats.sampling.sample_scipy import do_sample_scipy
            import numpy
            if seed is None or isinstance(seed, int):
                rand_state = numpy.random.default_rng(seed=seed)
            else:
                rand_state = seed
            samps = do_sample_scipy(self, size, rand_state)

        elif library == 'numpy':
            from sympy.stats.sampling.sample_numpy import do_sample_numpy
            import numpy
            if seed is None or isinstance(seed, int):
                rand_state = numpy.random.default_rng(seed=seed)
            else:
                rand_state = seed
            _size = None if size == () else size
            samps = do_sample_numpy(self, _size, rand_state)
        elif library in ('pymc', 'pymc3'):
            from sympy.stats.sampling.sample_pymc import do_sample_pymc
            import logging
            logging.getLogger("pymc").setLevel(logging.ERROR)
            try:
                import pymc
            except ImportError:
                import pymc3 as pymc

            with pymc.Model():
                if do_sample_pymc(self) is not None:
                    samps = pymc.sample(draws=prod(size), chains=1, compute_convergence_checks=False,
                            progressbar=False, random_seed=seed, return_inferencedata=False)[:]['X']
                    samps = samps.reshape(size)
                else:
                    samps = None

        else:
            raise NotImplementedError("Sampling from %s is not supported yet."
                                      % str(library))

        if samps is not None:
            return samps
        raise NotImplementedError(
            "Sampling for %s is not currently implemented from %s"
            % (self, library))


def _value_check(condition, message):
    """
    Raise a ValueError with message if condition is False, else
    return True if all conditions were True, else False.

    Examples
    ========

    >>> from sympy.stats.rv import _value_check
    >>> from sympy.abc import a, b, c
    >>> from sympy import And, Dummy

    >>> _value_check(2 < 3, '')
    True

    Here, the condition is not False, but it does not evaluate to True
    so False is returned (but no error is raised). So checking if the
    return value is True or False will tell you if all conditions were
    evaluated.

    >>> _value_check(a < b, '')
    False

    In this case the condition is False so an error is raised:

    >>> r = Dummy(real=True)
    >>> _value_check(r < r - 1, 'condition is not true')
    Traceback (most recent call last):
    ...
    ValueError: condition is not true

    If no condition of many conditions must be False, they can be
    checked by passing them as an iterable:

    >>> _value_check((a < 0, b < 0, c < 0), '')
    False

    The iterable can be a generator, too:

    >>> _value_check((i < 0 for i in (a, b, c)), '')
    False

    The following are equivalent to the above but do not pass
    an iterable:

    >>> all(_value_check(i < 0, '') for i in (a, b, c))
    False
    >>> _value_check(And(a < 0, b < 0, c < 0), '')
    False
    """
    if not iterable(condition):
        condition = [condition]
    truth = fuzzy_and(condition)
    if truth == False:
        raise ValueError(message)
    return truth == True

def _symbol_converter(sym):
    """
    Casts the parameter to Symbol if it is 'str'
    otherwise no operation is performed on it.

    Parameters
    ==========

    sym
        The parameter to be converted.

    Returns
    =======

    Symbol
        the parameter converted to Symbol.

    Raises
    ======

    TypeError
        If the parameter is not an instance of both str and
        Symbol.

    Examples
    ========

    >>> from sympy import Symbol
    >>> from sympy.stats.rv import _symbol_converter
    >>> s = _symbol_converter('s')
    >>> isinstance(s, Symbol)
    True
    >>> _symbol_converter(1)
    Traceback (most recent call last):
    ...
    TypeError: 1 is neither a Symbol nor a string
    >>> r = Symbol('r')
    >>> isinstance(r, Symbol)
    True
    """
    if isinstance(sym, str):
        sym = Symbol(sym)
    if not isinstance(sym, Symbol):
        raise TypeError("%s is neither a Symbol nor a string"%(sym))
    return sym

def sample_stochastic_process(process):
    """
    This function is used to sample from stochastic process.

    Parameters
    ==========

    process: StochasticProcess
        Process used to extract the samples. It must be an instance of
        StochasticProcess

    Examples
    ========

    >>> from sympy.stats import sample_stochastic_process, DiscreteMarkovChain
    >>> from sympy import Matrix
    >>> T = Matrix([[0.5, 0.2, 0.3],[0.2, 0.5, 0.3],[0.2, 0.3, 0.5]])
    >>> Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
    >>> next(sample_stochastic_process(Y)) in Y.state_space
    True
    >>> next(sample_stochastic_process(Y))  # doctest: +SKIP
    0
    >>> next(sample_stochastic_process(Y)) # doctest: +SKIP
    2

    Returns
    =======

    sample: iterator object
        iterator object containing the sample of given process

    """
    from sympy.stats.stochastic_process_types import StochasticProcess
    if not isinstance(process, StochasticProcess):
        raise ValueError("Process must be an instance of Stochastic Process")
    return process.sample()
<|endoftext|>from sympy.core.basic import Basic
from sympy.stats.joint_rv import ProductPSpace
from sympy.stats.rv import ProductDomain, _symbol_converter, Distribution


class StochasticPSpace(ProductPSpace):
    """
    Represents probability space of stochastic processes
    and their random variables. Contains mechanics to do
    computations for queries of stochastic processes.

    Explanation
    ===========

    Initialized by symbol, the specific process and
    distribution(optional) if the random indexed symbols
    of the process follows any specific distribution, like,
    in Bernoulli Process, each random indexed symbol follows
    Bernoulli distribution. For processes with memory, this
    parameter should not be passed.
    """

    def __new__(cls, sym, process, distribution=None):
        sym = _symbol_converter(sym)
        from sympy.stats.stochastic_process_types import StochasticProcess
        if not isinstance(process, StochasticProcess):
            raise TypeError("`process` must be an instance of StochasticProcess.")
        if distribution is None:
            distribution = Distribution()
        return Basic.__new__(cls, sym, process, distribution)

    @property
    def process(self):
        """
        The associated stochastic process.
        """
        return self.args[1]

    @property
    def domain(self):
        return ProductDomain(self.process.index_set,
                             self.process.state_space)

    @property
    def symbol(self):
        return self.args[0]

    @property
    def distribution(self):
        return self.args[2]

    def probability(self, condition, given_condition=None, evaluate=True, **kwargs):
        """
        Transfers the task of handling queries to the specific stochastic
        process because every process has their own logic of handling such
        queries.
        """
        return self.process.probability(condition, given_condition, evaluate, **kwargs)

    def compute_expectation(self, expr, condition=None, evaluate=True, **kwargs):
        """
        Transfers the task of handling queries to the specific stochastic
        process because every process has their own logic of handling such
        queries.
        """
        return self.process.expectation(expr, condition, evaluate, **kwargs)
<|endoftext|>"""
Joint Random Variables Module

See Also
========
sympy.stats.rv
sympy.stats.frv
sympy.stats.crv
sympy.stats.drv
"""
from math import prod

from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.sets.sets import ProductSet
from sympy.tensor.indexed import Indexed
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum, summation
from sympy.core.containers import Tuple
from sympy.integrals.integrals import Integral, integrate
from sympy.matrices import ImmutableMatrix, matrix2numpy, list2numpy
from sympy.stats.crv import SingleContinuousDistribution, SingleContinuousPSpace
from sympy.stats.drv import SingleDiscreteDistribution, SingleDiscretePSpace
from sympy.stats.rv import (ProductPSpace, NamedArgsMixin, Distribution,
                            ProductDomain, RandomSymbol, random_symbols,
                            SingleDomain, _symbol_converter)
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import filldedent
from sympy.external import import_module

# __all__ = ['marginal_distribution']

class JointPSpace(ProductPSpace):
    """
    Represents a joint probability space. Represented using symbols for
    each component and a distribution.
    """
    def __new__(cls, sym, dist):
        if isinstance(dist, SingleContinuousDistribution):
            return SingleContinuousPSpace(sym, dist)
        if isinstance(dist, SingleDiscreteDistribution):
            return SingleDiscretePSpace(sym, dist)
        sym = _symbol_converter(sym)
        return Basic.__new__(cls, sym, dist)

    @property
    def set(self):
        return self.domain.set

    @property
    def symbol(self):
        return self.args[0]

    @property
    def distribution(self):
        return self.args[1]

    @property
    def value(self):
        return JointRandomSymbol(self.symbol, self)

    @property
    def component_count(self):
        _set = self.distribution.set
        if isinstance(_set, ProductSet):
            return S(len(_set.args))
        elif isinstance(_set, Product):
            return _set.limits[0][-1]
        return S.One

    @property
    def pdf(self):
        sym = [Indexed(self.symbol, i) for i in range(self.component_count)]
        return self.distribution(*sym)

    @property
    def domain(self):
        rvs = random_symbols(self.distribution)
        if not rvs:
            return SingleDomain(self.symbol, self.distribution.set)
        return ProductDomain(*[rv.pspace.domain for rv in rvs])

    def component_domain(self, index):
        return self.set.args[index]

    def marginal_distribution(self, *indices):
        count = self.component_count
        if count.atoms(Symbol):
            raise ValueError("Marginal distributions cannot be computed "
                                "for symbolic dimensions. It is a work under progress.")
        orig = [Indexed(self.symbol, i) for i in range(count)]
        all_syms = [Symbol(str(i)) for i in orig]
        replace_dict = dict(zip(all_syms, orig))
        sym = tuple(Symbol(str(Indexed(self.symbol, i))) for i in indices)
        limits = [[i,] for i in all_syms if i not in sym]
        index = 0
        for i in range(count):
            if i not in indices:
                limits[index].append(self.distribution.set.args[i])
                limits[index] = tuple(limits[index])
                index += 1
        if self.distribution.is_Continuous:
            f = Lambda(sym, integrate(self.distribution(*all_syms), *limits))
        elif self.distribution.is_Discrete:
            f = Lambda(sym, summation(self.distribution(*all_syms), *limits))
        return f.xreplace(replace_dict)

    def compute_expectation(self, expr, rvs=None, evaluate=False, **kwargs):
        syms = tuple(self.value[i] for i in range(self.component_count))
        rvs = rvs or syms
        if not any(i in rvs for i in syms):
            return expr
        expr = expr*self.pdf
        for rv in rvs:
            if isinstance(rv, Indexed):
                expr = expr.xreplace({rv: Indexed(str(rv.base), rv.args[1])})
            elif isinstance(rv, RandomSymbol):
                expr = expr.xreplace({rv: rv.symbol})
        if self.value in random_symbols(expr):
            raise NotImplementedError(filldedent('''
            Expectations of expression with unindexed joint random symbols
            cannot be calculated yet.'''))
        limits = tuple((Indexed(str(rv.base),rv.args[1]),
            self.distribution.set.args[rv.args[1]]) for rv in syms)
        return Integral(expr, *limits)

    def where(self, condition):
        raise NotImplementedError()

    def compute_density(self, expr):
        raise NotImplementedError()

    def sample(self, size=(), library='scipy', seed=None):
        """
        Internal sample method

        Returns dictionary mapping RandomSymbol to realization value.
        """
        return {RandomSymbol(self.symbol, self): self.distribution.sample(size,
                    library=library, seed=seed)}

    def probability(self, condition):
        raise NotImplementedError()


class SampleJointScipy:
    """Returns the sample from scipy of the given distribution"""
    def __new__(cls, dist, size, seed=None):
        return cls._sample_scipy(dist, size, seed)

    @classmethod
    def _sample_scipy(cls, dist, size, seed):
        """Sample from SciPy."""

        import numpy
        if seed is None or isinstance(seed, int):
            rand_state = numpy.random.default_rng(seed=seed)
        else:
            rand_state = seed
        from scipy import stats as scipy_stats
        scipy_rv_map = {
            'MultivariateNormalDistribution': lambda dist, size: scipy_stats.multivariate_normal.rvs(
                mean=matrix2numpy(dist.mu).flatten(),
                cov=matrix2numpy(dist.sigma), size=size, random_state=rand_state),
            'MultivariateBetaDistribution': lambda dist, size: scipy_stats.dirichlet.rvs(
                alpha=list2numpy(dist.alpha, float).flatten(), size=size, random_state=rand_state),
            'MultinomialDistribution': lambda dist, size: scipy_stats.multinomial.rvs(
                n=int(dist.n), p=list2numpy(dist.p, float).flatten(), size=size, random_state=rand_state)
        }

        sample_shape = {
            'MultivariateNormalDistribution': lambda dist: matrix2numpy(dist.mu).flatten().shape,
            'MultivariateBetaDistribution': lambda dist: list2numpy(dist.alpha).flatten().shape,
            'MultinomialDistribution': lambda dist: list2numpy(dist.p).flatten().shape
        }

        dist_list = scipy_rv_map.keys()

        if dist.__class__.__name__ not in dist_list:
            return None

        samples = scipy_rv_map[dist.__class__.__name__](dist, size)
        return samples.reshape(size + sample_shape[dist.__class__.__name__](dist))

class SampleJointNumpy:
    """Returns the sample from numpy of the given distribution"""

    def __new__(cls, dist, size, seed=None):
        return cls._sample_numpy(dist, size, seed)

    @classmethod
    def _sample_numpy(cls, dist, size, seed):
        """Sample from NumPy."""

        import numpy
        if seed is None or isinstance(seed, int):
            rand_state = numpy.random.default_rng(seed=seed)
        else:
            rand_state = seed
        numpy_rv_map = {
            'MultivariateNormalDistribution': lambda dist, size: rand_state.multivariate_normal(
                mean=matrix2numpy(dist.mu, float).flatten(),
                cov=matrix2numpy(dist.sigma, float), size=size),
            'MultivariateBetaDistribution': lambda dist, size: rand_state.dirichlet(
                alpha=list2numpy(dist.alpha, float).flatten(), size=size),
            'MultinomialDistribution': lambda dist, size: rand_state.multinomial(
                n=int(dist.n), pvals=list2numpy(dist.p, float).flatten(), size=size)
        }

        sample_shape = {
            'MultivariateNormalDistribution': lambda dist: matrix2numpy(dist.mu).flatten().shape,
            'MultivariateBetaDistribution': lambda dist: list2numpy(dist.alpha).flatten().shape,
            'MultinomialDistribution': lambda dist: list2numpy(dist.p).flatten().shape
        }

        dist_list = numpy_rv_map.keys()

        if dist.__class__.__name__ not in dist_list:
            return None

        samples = numpy_rv_map[dist.__class__.__name__](dist, prod(size))
        return samples.reshape(size + sample_shape[dist.__class__.__name__](dist))

class SampleJointPymc:
    """Returns the sample from pymc of the given distribution"""

    def __new__(cls, dist, size, seed=None):
        return cls._sample_pymc(dist, size, seed)

    @classmethod
    def _sample_pymc(cls, dist, size, seed):
        """Sample from PyMC."""

        try:
            import pymc
        except ImportError:
            import pymc3 as pymc
        pymc_rv_map = {
            'MultivariateNormalDistribution': lambda dist:
                pymc.MvNormal('X', mu=matrix2numpy(dist.mu, float).flatten(),
                cov=matrix2numpy(dist.sigma, float), shape=(1, dist.mu.shape[0])),
            'MultivariateBetaDistribution': lambda dist:
                pymc.Dirichlet('X', a=list2numpy(dist.alpha, float).flatten()),
            'MultinomialDistribution': lambda dist:
                pymc.Multinomial('X', n=int(dist.n),
                p=list2numpy(dist.p, float).flatten(), shape=(1, len(dist.p)))
        }

        sample_shape = {
            'MultivariateNormalDistribution': lambda dist: matrix2numpy(dist.mu).flatten().shape,
            'MultivariateBetaDistribution': lambda dist: list2numpy(dist.alpha).flatten().shape,
            'MultinomialDistribution': lambda dist: list2numpy(dist.p).flatten().shape
        }

        dist_list = pymc_rv_map.keys()

        if dist.__class__.__name__ not in dist_list:
            return None

        import logging
        logging.getLogger("pymc3").setLevel(logging.ERROR)
        with pymc.Model():
            pymc_rv_map[dist.__class__.__name__](dist)
            samples = pymc.sample(draws=prod(size), chains=1, progressbar=False, random_seed=seed, return_inferencedata=False, compute_convergence_checks=False)[:]['X']
        return samples.reshape(size + sample_shape[dist.__class__.__name__](dist))


_get_sample_class_jrv = {
    'scipy': SampleJointScipy,
    'pymc3': SampleJointPymc,
    'pymc': SampleJointPymc,
    'numpy': SampleJointNumpy
}

class JointDistribution(Distribution, NamedArgsMixin):
    """
    Represented by the random variables part of the joint distribution.
    Contains methods for PDF, CDF, sampling, marginal densities, etc.
    """

    _argnames = ('pdf', )

    def __new__(cls, *args):
        args = list(map(sympify, args))
        for i in range(len(args)):
            if isinstance(args[i], list):
                args[i] = ImmutableMatrix(args[i])
        return Basic.__new__(cls, *args)

    @property
    def domain(self):
        return ProductDomain(self.symbols)

    @property
    def pdf(self):
        return self.density.args[1]

    def cdf(self, other):
        if not isinstance(other, dict):
            raise ValueError("%s should be of type dict, got %s"%(other, type(other)))
        rvs = other.keys()
        _set = self.domain.set.sets
        expr = self.pdf(tuple(i.args[0] for i in self.symbols))
        for i in range(len(other)):
            if rvs[i].is_Continuous:
                density = Integral(expr, (rvs[i], _set[i].inf,
                    other[rvs[i]]))
            elif rvs[i].is_Discrete:
                density = Sum(expr, (rvs[i], _set[i].inf,
                    other[rvs[i]]))
        return density

    def sample(self, size=(), library='scipy', seed=None):
        """ A random realization from the distribution """

        libraries = ('scipy', 'numpy', 'pymc3', 'pymc')
        if library not in libraries:
            raise NotImplementedError("Sampling from %s is not supported yet."
                                        % str(library))
        if not import_module(library):
            raise ValueError("Failed to import %s" % library)

        samps = _get_sample_class_jrv[library](self, size, seed=seed)

        if samps is not None:
            return samps
        raise NotImplementedError(
                "Sampling for %s is not currently implemented from %s"
                % (self.__class__.__name__, library)
                )

    def __call__(self, *args):
        return self.pdf(*args)

class JointRandomSymbol(RandomSymbol):
    """
    Representation of random symbols with joint probability distributions
    to allow indexing."
    """
    def __getitem__(self, key):
        if isinstance(self.pspace, JointPSpace):
            if (self.pspace.component_count <= key) == True:
                raise ValueError("Index keys for %s can only up to %s." %
                    (self.name, self.pspace.component_count - 1))
            return Indexed(self, key)



class MarginalDistribution(Distribution):
    """
    Represents the marginal distribution of a joint probability space.

    Initialised using a probability distribution and random variables(or
    their indexed components) which should be a part of the resultant
    distribution.
    """

    def __new__(cls, dist, *rvs):
        if len(rvs) == 1 and iterable(rvs[0]):
            rvs = tuple(rvs[0])
        if not all(isinstance(rv, (Indexed, RandomSymbol)) for rv in rvs):
            raise ValueError(filldedent('''Marginal distribution can be
             intitialised only in terms of random variables or indexed random
             variables'''))
        rvs = Tuple.fromiter(rv for rv in rvs)
        if not isinstance(dist, JointDistribution) and len(random_symbols(dist)) == 0:
            return dist
        return Basic.__new__(cls, dist, rvs)

    def check(self):
        pass

    @property
    def set(self):
        rvs = [i for i in self.args[1] if isinstance(i, RandomSymbol)]
        return ProductSet(*[rv.pspace.set for rv in rvs])

    @property
    def symbols(self):
        rvs = self.args[1]
        return {rv.pspace.symbol for rv in rvs}

    def pdf(self, *x):
        expr, rvs = self.args[0], self.args[1]
        marginalise_out = [i for i in random_symbols(expr) if i not in rvs]
        if isinstance(expr, JointDistribution):
            count = len(expr.domain.args)
            x = Dummy('x', real=True)
            syms = tuple(Indexed(x, i) for i in count)
            expr = expr.pdf(syms)
        else:
            syms = tuple(rv.pspace.symbol if isinstance(rv, RandomSymbol) else rv.args[0] for rv in rvs)
        return Lambda(syms, self.compute_pdf(expr, marginalise_out))(*x)

    def compute_pdf(self, expr, rvs):
        for rv in rvs:
            lpdf = 1
            if isinstance(rv, RandomSymbol):
                lpdf = rv.pspace.pdf
            expr = self.marginalise_out(expr*lpdf, rv)
        return expr

    def marginalise_out(self, expr, rv):
        from sympy.concrete.summations import Sum
        if isinstance(rv, RandomSymbol):
            dom = rv.pspace.set
        elif isinstance(rv, Indexed):
            dom = rv.base.component_domain(
                rv.pspace.component_domain(rv.args[1]))
        expr = expr.xreplace({rv: rv.pspace.symbol})
        if rv.pspace.is_Continuous:
            #TODO: Modify to support integration
            #for all kinds of sets.
            expr = Integral(expr, (rv.pspace.symbol, dom))
        elif rv.pspace.is_Discrete:
            #incorporate this into `Sum`/`summation`
            if dom in (S.Integers, S.Naturals, S.Naturals0):
                dom = (dom.inf, dom.sup)
            expr = Sum(expr, (rv.pspace.symbol, dom))
        return expr

    def __call__(self, *args):
        return self.pdf(*args)
<|endoftext|>"""
Finite Discrete Random Variables - Prebuilt variable types

Contains
========
FiniteRV
DiscreteUniform
Die
Bernoulli
Coin
Binomial
BetaBinomial
Hypergeometric
Rademacher
IdealSoliton
RobustSoliton
"""


from sympy.core.cache import cacheit
from sympy.core.function import Lambda
from sympy.core.numbers import (Integer, Rational)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import Or
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Range
from sympy.sets.sets import (Intersection, Interval)
from sympy.functions.special.beta_functions import beta as beta_fn
from sympy.stats.frv import (SingleFiniteDistribution,
                             SingleFinitePSpace)
from sympy.stats.rv import _value_check, Density, is_random
from sympy.utilities.iterables import multiset
from sympy.utilities.misc import filldedent


__all__ = ['FiniteRV',
'DiscreteUniform',
'Die',
'Bernoulli',
'Coin',
'Binomial',
'BetaBinomial',
'Hypergeometric',
'Rademacher',
'IdealSoliton',
'RobustSoliton',
]

def rv(name, cls, *args, **kwargs):
    args = list(map(sympify, args))
    dist = cls(*args)
    if kwargs.pop('check', True):
        dist.check(*args)
    pspace = SingleFinitePSpace(name, dist)
    if any(is_random(arg) for arg in args):
        from sympy.stats.compound_rv import CompoundPSpace, CompoundDistribution
        pspace = CompoundPSpace(name, CompoundDistribution(dist))
    return pspace.value

class FiniteDistributionHandmade(SingleFiniteDistribution):

    @property
    def dict(self):
        return self.args[0]

    def pmf(self, x):
        x = Symbol('x')
        return Lambda(x, Piecewise(*(
            [(v, Eq(k, x)) for k, v in self.dict.items()] + [(S.Zero, True)])))

    @property
    def set(self):
        return set(self.dict.keys())

    @staticmethod
    def check(density):
        for p in density.values():
            _value_check((p >= 0, p <= 1),
                        "Probability at a point must be between 0 and 1.")
        val = sum(density.values())
        _value_check(Eq(val, 1) != S.false, "Total Probability must be 1.")

def FiniteRV(name, density, **kwargs):
    r"""
    Create a Finite Random Variable given a dict representing the density.

    Parameters
    ==========

    name : Symbol
        Represents name of the random variable.
    density : dict
        Dictionary containing the pdf of finite distribution
    check : bool
        If True, it will check whether the given density
        integrates to 1 over the given set. If False, it
        will not perform this check. Default is False.

    Examples
    ========

    >>> from sympy.stats import FiniteRV, P, E

    >>> density = {0: .1, 1: .2, 2: .3, 3: .4}
    >>> X = FiniteRV('X', density)

    >>> E(X)
    2.00000000000000
    >>> P(X >= 2)
    0.700000000000000

    Returns
    =======

    RandomSymbol

    """
    # have a default of False while `rv` should have a default of True
    kwargs['check'] = kwargs.pop('check', False)
    return rv(name, FiniteDistributionHandmade, density, **kwargs)

class DiscreteUniformDistribution(SingleFiniteDistribution):

    @staticmethod
    def check(*args):
        # not using _value_check since there is a
        # suggestion for the user
        if len(set(args)) != len(args):
            weights = multiset(args)
            n = Integer(len(args))
            for k in weights:
                weights[k] /= n
            raise ValueError(filldedent("""
                Repeated args detected but set expected. For a
                distribution having different weights for each
                item use the following:""") + (
                '\nS("FiniteRV(%s, %s)")' % ("'X'", weights)))

    @property
    def p(self):
        return Rational(1, len(self.args))

    @property  # type: ignore
    @cacheit
    def dict(self):
        return dict.fromkeys(self.set, self.p)

    @property
    def set(self):
        return set(self.args)

    def pmf(self, x):
        if x in self.args:
            return self.p
        else:
            return S.Zero


def DiscreteUniform(name, items):
    r"""
    Create a Finite Random Variable representing a uniform distribution over
    the input set.

    Parameters
    ==========

    items : list/tuple
        Items over which Uniform distribution is to be made

    Examples
    ========

    >>> from sympy.stats import DiscreteUniform, density
    >>> from sympy import symbols

    >>> X = DiscreteUniform('X', symbols('a b c')) # equally likely over a, b, c
    >>> density(X).dict
    {a: 1/3, b: 1/3, c: 1/3}

    >>> Y = DiscreteUniform('Y', list(range(5))) # distribution over a range
    >>> density(Y).dict
    {0: 1/5, 1: 1/5, 2: 1/5, 3: 1/5, 4: 1/5}

    Returns
    =======

    RandomSymbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Discrete_uniform_distribution
    .. [2] https://mathworld.wolfram.com/DiscreteUniformDistribution.html

    """
    return rv(name, DiscreteUniformDistribution, *items)


class DieDistribution(SingleFiniteDistribution):
    _argnames = ('sides',)

    @staticmethod
    def check(sides):
        _value_check((sides.is_positive, sides.is_integer),
                    "number of sides must be a positive integer.")

    @property
    def is_symbolic(self):
        return not self.sides.is_number

    @property
    def high(self):
        return self.sides

    @property
    def low(self):
        return S.One

    @property
    def set(self):
        if self.is_symbolic:
            return Intersection(S.Naturals0, Interval(0, self.sides))
        return set(map(Integer, range(1, self.sides + 1)))

    def pmf(self, x):
        x = sympify(x)
        if not (x.is_number or x.is_Symbol or is_random(x)):
            raise ValueError("'x' expected as an argument of type 'number', 'Symbol', or "
                        "'RandomSymbol' not %s" % (type(x)))
        cond = Ge(x, 1) & Le(x, self.sides) & Contains(x, S.Integers)
        return Piecewise((S.One/self.sides, cond), (S.Zero, True))

def Die(name, sides=6):
    r"""
    Create a Finite Random Variable representing a fair die.

    Parameters
    ==========

    sides : Integer
        Represents the number of sides of the Die, by default is 6

    Examples
    ========

    >>> from sympy.stats import Die, density
    >>> from sympy import Symbol

    >>> D6 = Die('D6', 6) # Six sided Die
    >>> density(D6).dict
    {1: 1/6, 2: 1/6, 3: 1/6, 4: 1/6, 5: 1/6, 6: 1/6}

    >>> D4 = Die('D4', 4) # Four sided Die
    >>> density(D4).dict
    {1: 1/4, 2: 1/4, 3: 1/4, 4: 1/4}

    >>> n = Symbol('n', positive=True, integer=True)
    >>> Dn = Die('Dn', n) # n sided Die
    >>> density(Dn).dict
    Density(DieDistribution(n))
    >>> density(Dn).dict.subs(n, 4).doit()
    {1: 1/4, 2: 1/4, 3: 1/4, 4: 1/4}

    Returns
    =======

    RandomSymbol
    """

    return rv(name, DieDistribution, sides)


class BernoulliDistribution(SingleFiniteDistribution):
    _argnames = ('p', 'succ', 'fail')

    @staticmethod
    def check(p, succ, fail):
        _value_check((p >= 0, p <= 1),
                    "p should be in range [0, 1].")

    @property
    def set(self):
        return {self.succ, self.fail}

    def pmf(self, x):
        if isinstance(self.succ, Symbol) and isinstance(self.fail, Symbol):
            return Piecewise((self.p, x == self.succ),
                             (1 - self.p, x == self.fail),
                             (S.Zero, True))
        return Piecewise((self.p, Eq(x, self.succ)),
                         (1 - self.p, Eq(x, self.fail)),
                         (S.Zero, True))


def Bernoulli(name, p, succ=1, fail=0):
    r"""
    Create a Finite Random Variable representing a Bernoulli process.

    Parameters
    ==========

    p : Rational number between 0 and 1
       Represents probability of success
    succ : Integer/symbol/string
       Represents event of success
    fail : Integer/symbol/string
       Represents event of failure

    Examples
    ========

    >>> from sympy.stats import Bernoulli, density
    >>> from sympy import S

    >>> X = Bernoulli('X', S(3)/4) # 1-0 Bernoulli variable, probability = 3/4
    >>> density(X).dict
    {0: 1/4, 1: 3/4}

    >>> X = Bernoulli('X', S.Half, 'Heads', 'Tails') # A fair coin toss
    >>> density(X).dict
    {Heads: 1/2, Tails: 1/2}

    Returns
    =======

    RandomSymbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bernoulli_distribution
    .. [2] https://mathworld.wolfram.com/BernoulliDistribution.html

    """

    return rv(name, BernoulliDistribution, p, succ, fail)


def Coin(name, p=S.Half):
    r"""
    Create a Finite Random Variable representing a Coin toss.

    This is an equivalent of a Bernoulli random variable with
    "H" and "T" as success and failure events respectively.

    Parameters
    ==========

    p : Rational Number between 0 and 1
      Represents probability of getting "Heads", by default is Half

    Examples
    ========

    >>> from sympy.stats import Coin, density
    >>> from sympy import Rational

    >>> C = Coin('C') # A fair coin toss
    >>> density(C).dict
    {H: 1/2, T: 1/2}

    >>> C2 = Coin('C2', Rational(3, 5)) # An unfair coin
    >>> density(C2).dict
    {H: 3/5, T: 2/5}

    Returns
    =======

    RandomSymbol

    See Also
    ========

    sympy.stats.Binomial

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Coin_flipping

    """
    return rv(name, BernoulliDistribution, p, 'H', 'T')


class BinomialDistribution(SingleFiniteDistribution):
    _argnames = ('n', 'p', 'succ', 'fail')

    @staticmethod
    def check(n, p, succ, fail):
        _value_check((n.is_integer, n.is_nonnegative),
                    "'n' must be nonnegative integer.")
        _value_check((p <= 1, p >= 0),
                    "p should be in range [0, 1].")

    @property
    def high(self):
        return self.n

    @property
    def low(self):
        return S.Zero

    @property
    def is_symbolic(self):
        return not self.n.is_number

    @property
    def set(self):
        if self.is_symbolic:
            return Intersection(S.Naturals0, Interval(0, self.n))
        return set(self.dict.keys())

    def pmf(self, x):
        n, p = self.n, self.p
        x = sympify(x)
        if not (x.is_number or x.is_Symbol or is_random(x)):
            raise ValueError("'x' expected as an argument of type 'number', 'Symbol', or "
                        "'RandomSymbol' not %s" % (type(x)))
        cond = Ge(x, 0) & Le(x, n) & Contains(x, S.Integers)
        return Piecewise((binomial(n, x) * p**x * (1 - p)**(n - x), cond), (S.Zero, True))

    @property  # type: ignore
    @cacheit
    def dict(self):
        if self.is_symbolic:
            return Density(self)
        return {k*self.succ + (self.n-k)*self.fail: self.pmf(k)
                    for k in range(0, self.n + 1)}


def Binomial(name, n, p, succ=1, fail=0):
    r"""
    Create a Finite Random Variable representing a binomial distribution.

    Parameters
    ==========

    n : Positive Integer
      Represents number of trials
    p : Rational Number between 0 and 1
      Represents probability of success
    succ : Integer/symbol/string
      Represents event of success, by default is 1
    fail : Integer/symbol/string
      Represents event of failure, by default is 0

    Examples
    ========

    >>> from sympy.stats import Binomial, density
    >>> from sympy import S, Symbol

    >>> X = Binomial('X', 4, S.Half) # Four "coin flips"
    >>> density(X).dict
    {0: 1/16, 1: 1/4, 2: 3/8, 3: 1/4, 4: 1/16}

    >>> n = Symbol('n', positive=True, integer=True)
    >>> p = Symbol('p', positive=True)
    >>> X = Binomial('X', n, S.Half) # n "coin flips"
    >>> density(X).dict
    Density(BinomialDistribution(n, 1/2, 1, 0))
    >>> density(X).dict.subs(n, 4).doit()
    {0: 1/16, 1: 1/4, 2: 3/8, 3: 1/4, 4: 1/16}

    Returns
    =======

    RandomSymbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Binomial_distribution
    .. [2] https://mathworld.wolfram.com/BinomialDistribution.html

    """

    return rv(name, BinomialDistribution, n, p, succ, fail)

#-------------------------------------------------------------------------------
# Beta-binomial distribution ----------------------------------------------------------

class BetaBinomialDistribution(SingleFiniteDistribution):
    _argnames = ('n', 'alpha', 'beta')

    @staticmethod
    def check(n, alpha, beta):
        _value_check((n.is_integer, n.is_nonnegative),
        "'n' must be nonnegative integer. n = %s." % str(n))
        _value_check((alpha > 0),
        "'alpha' must be: alpha > 0 . alpha = %s" % str(alpha))
        _value_check((beta > 0),
        "'beta' must be: beta > 0 . beta = %s" % str(beta))

    @property
    def high(self):
        return self.n

    @property
    def low(self):
        return S.Zero

    @property
    def is_symbolic(self):
        return not self.n.is_number

    @property
    def set(self):
        if self.is_symbolic:
            return Intersection(S.Naturals0, Interval(0, self.n))
        return set(map(Integer, range(self.n + 1)))

    def pmf(self, k):
        n, a, b = self.n, self.alpha, self.beta
        return binomial(n, k) * beta_fn(k + a, n - k + b) / beta_fn(a, b)


def BetaBinomial(name, n, alpha, beta):
    r"""
    Create a Finite Random Variable representing a Beta-binomial distribution.

    Parameters
    ==========

    n : Positive Integer
      Represents number of trials
    alpha : Real positive number
    beta : Real positive number

    Examples
    ========

    >>> from sympy.stats import BetaBinomial, density

    >>> X = BetaBinomial('X', 2, 1, 1)
    >>> density(X).dict
    {0: 1/3, 1: 2*beta(2, 2), 2: 1/3}

    Returns
    =======

    RandomSymbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Beta-binomial_distribution
    .. [2] https://mathworld.wolfram.com/BetaBinomialDistribution.html

    """

    return rv(name, BetaBinomialDistribution, n, alpha, beta)


class HypergeometricDistribution(SingleFiniteDistribution):
    _argnames = ('N', 'm', 'n')

    @staticmethod
    def check(n, N, m):
        _value_check((N.is_integer, N.is_nonnegative),
                     "'N' must be nonnegative integer. N = %s." % str(N))
        _value_check((n.is_integer, n.is_nonnegative),
                     "'n' must be nonnegative integer. n = %s." % str(n))
        _value_check((m.is_integer, m.is_nonnegative),
                     "'m' must be nonnegative integer. m = %s." % str(m))

    @property
    def is_symbolic(self):
        return not all(x.is_number for x in (self.N, self.m, self.n))

    @property
    def high(self):
        return Piecewise((self.n, Lt(self.n, self.m) != False), (self.m, True))

    @property
    def low(self):
        return Piecewise((0, Gt(0, self.n + self.m - self.N) != False), (self.n + self.m - self.N, True))

    @property
    def set(self):
        N, m, n = self.N, self.m, self.n
        if self.is_symbolic:
            return Intersection(S.Naturals0, Interval(self.low, self.high))
        return set(range(max(0, n + m - N), min(n, m) + 1))

    def pmf(self, k):
        N, m, n = self.N, self.m, self.n
        return S(binomial(m, k) * binomial(N - m, n - k))/binomial(N, n)


def Hypergeometric(name, N, m, n):
    r"""
    Create a Finite Random Variable representing a hypergeometric distribution.

    Parameters
    ==========

    N : Positive Integer
      Represents finite population of size N.
    m : Positive Integer
      Represents number of trials with required feature.
    n : Positive Integer
      Represents numbers of draws.


    Examples
    ========

    >>> from sympy.stats import Hypergeometric, density

    >>> X = Hypergeometric('X', 10, 5, 3) # 10 marbles, 5 white (success), 3 draws
    >>> density(X).dict
    {0: 1/12, 1: 5/12, 2: 5/12, 3: 1/12}

    Returns
    =======

    RandomSymbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hypergeometric_distribution
    .. [2] https://mathworld.wolfram.com/HypergeometricDistribution.html

    """
    return rv(name, HypergeometricDistribution, N, m, n)


class RademacherDistribution(SingleFiniteDistribution):

    @property
    def set(self):
        return {-1, 1}

    @property
    def pmf(self):
        k = Dummy('k')
        return Lambda(k, Piecewise((S.Half, Or(Eq(k, -1), Eq(k, 1))), (S.Zero, True)))

def Rademacher(name):
    r"""
    Create a Finite Random Variable representing a Rademacher distribution.

    Examples
    ========

    >>> from sympy.stats import Rademacher, density

    >>> X = Rademacher('X')
    >>> density(X).dict
    {-1: 1/2, 1: 1/2}

    Returns
    =======

    RandomSymbol

    See Also
    ========

    sympy.stats.Bernoulli

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Rademacher_distribution

    """
    return rv(name, RademacherDistribution)

class IdealSolitonDistribution(SingleFiniteDistribution):
    _argnames = ('k',)

    @staticmethod
    def check(k):
        _value_check(k.is_integer and k.is_positive,
                    "'k' must be a positive integer.")

    @property
    def low(self):
        return S.One

    @property
    def high(self):
        return self.k

    @property
    def set(self):
        return set(map(Integer, range(1, self.k + 1)))

    @property # type: ignore
    @cacheit
    def dict(self):
        if self.k.is_Symbol:
            return Density(self)
        d = {1: Rational(1, self.k)}
        d.update({i: Rational(1, i*(i - 1)) for i in range(2, self.k + 1)})
        return d

    def pmf(self, x):
        x = sympify(x)
        if not (x.is_number or x.is_Symbol or is_random(x)):
            raise ValueError("'x' expected as an argument of type 'number', 'Symbol', or "
                        "'RandomSymbol' not %s" % (type(x)))
        cond1 = Eq(x, 1) & x.is_integer
        cond2 = Ge(x, 1) & Le(x, self.k) & x.is_integer
        return Piecewise((1/self.k, cond1), (1/(x*(x - 1)), cond2), (S.Zero, True))

def IdealSoliton(name, k):
    r"""
    Create a Finite Random Variable of Ideal Soliton Distribution

    Parameters
    ==========

    k : Positive Integer
        Represents the number of input symbols in an LT (Luby Transform) code.

    Examples
    ========

    >>> from sympy.stats import IdealSoliton, density, P, E
    >>> sol = IdealSoliton('sol', 5)
    >>> density(sol).dict
    {1: 1/5, 2: 1/2, 3: 1/6, 4: 1/12, 5: 1/20}
    >>> density(sol).set
    {1, 2, 3, 4, 5}

    >>> from sympy import Symbol
    >>> k = Symbol('k', positive=True, integer=True)
    >>> sol = IdealSoliton('sol', k)
    >>> density(sol).dict
    Density(IdealSolitonDistribution(k))
    >>> density(sol).dict.subs(k, 10).doit()
    {1: 1/10, 2: 1/2, 3: 1/6, 4: 1/12, 5: 1/20, 6: 1/30, 7: 1/42, 8: 1/56, 9: 1/72, 10: 1/90}

    >>> E(sol.subs(k, 10))
    7381/2520

    >>> P(sol.subs(k, 4) > 2)
    1/4

    Returns
    =======

    RandomSymbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Soliton_distribution#Ideal_distribution
    .. [2] https://pages.cs.wisc.edu/~suman/courses/740/papers/luby02lt.pdf

    """
    return rv(name, IdealSolitonDistribution, k)

class RobustSolitonDistribution(SingleFiniteDistribution):
    _argnames= ('k', 'delta', 'c')

    @staticmethod
    def check(k, delta, c):
        _value_check(k.is_integer and k.is_positive,
                    "'k' must be a positive integer")
        _value_check(Gt(delta, 0) and Le(delta, 1),
                    "'delta' must be a real number in the interval (0,1)")
        _value_check(c.is_positive,
                    "'c' must be a positive real number.")

    @property
    def R(self):
        return self.c * log(self.k/self.delta) * self.k**0.5

    @property
    def Z(self):
        z = 0
        for i in Range(1, round(self.k/self.R)):
            z += (1/i)
        z += log(self.R/self.delta)
        return 1 + z * self.R/self.k

    @property
    def low(self):
        return S.One

    @property
    def high(self):
        return self.k

    @property
    def set(self):
        return set(map(Integer, range(1, self.k + 1)))

    @property
    def is_symbolic(self):
        return not (self.k.is_number and self.c.is_number and self.delta.is_number)

    def pmf(self, x):
        x = sympify(x)
        if not (x.is_number or x.is_Symbol or is_random(x)):
            raise ValueError("'x' expected as an argument of type 'number', 'Symbol', or "
                        "'RandomSymbol' not %s" % (type(x)))

        cond1 = Eq(x, 1) & x.is_integer
        cond2 = Ge(x, 1) & Le(x, self.k) & x.is_integer
        rho = Piecewise((Rational(1, self.k), cond1), (Rational(1, x*(x-1)), cond2), (S.Zero, True))

        cond1 = Ge(x, 1) & Le(x, round(self.k/self.R)-1)
        cond2 = Eq(x, round(self.k/self.R))
        tau = Piecewise((self.R/(self.k * x), cond1), (self.R * log(self.R/self.delta)/self.k, cond2), (S.Zero, True))

        return (rho + tau)/self.Z

def RobustSoliton(name, k, delta, c):
    r'''
    Create a Finite Random Variable of Robust Soliton Distribution

    Parameters
    ==========

    k : Positive Integer
        Represents the number of input symbols in an LT (Luby Transform) code.
    delta : Positive Rational Number
            Represents the failure probability. Must be in the interval (0,1).
    c : Positive Rational Number
        Constant of proportionality. Values close to 1 are recommended

    Examples
    ========

    >>> from sympy.stats import RobustSoliton, density, P, E
    >>> robSol = RobustSoliton('robSol', 5, 0.5, 0.01)
    >>> density(robSol).dict
    {1: 0.204253668152708, 2: 0.490631107897393, 3: 0.165210624506162, 4: 0.0834387731899302, 5: 0.0505633404760675}
    >>> density(robSol).set
    {1, 2, 3, 4, 5}

    >>> from sympy import Symbol
    >>> k = Symbol('k', positive=True, integer=True)
    >>> c = Symbol('c', positive=True)
    >>> robSol = RobustSoliton('robSol', k, 0.5, c)
    >>> density(robSol).dict
    Density(RobustSolitonDistribution(k, 0.5, c))
    >>> density(robSol).dict.subs(k, 10).subs(c, 0.03).doit()
    {1: 0.116641095387194, 2: 0.467045731687165, 3: 0.159984123349381, 4: 0.0821431680681869, 5: 0.0505765646770100,
    6: 0.0345781523420719, 7: 0.0253132820710503, 8: 0.0194459129233227, 9: 0.0154831166726115, 10: 0.0126733075238887}

    >>> E(robSol.subs(k, 10).subs(c, 0.05))
    2.91358846104106

    >>> P(robSol.subs(k, 4).subs(c, 0.1) > 2)
    0.243650614389834

    Returns
    =======

    RandomSymbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Soliton_distribution#Robust_distribution
    .. [2] https://www.inference.org.uk/mackay/itprnn/ps/588.596.pdf
    .. [3] https://pages.cs.wisc.edu/~suman/courses/740/papers/luby02lt.pdf

    '''
    return rv(name, RobustSolitonDistribution, k, delta, c)
<|endoftext|>"""
Finite Discrete Random Variables Module

See Also
========
sympy.stats.frv_types
sympy.stats.rv
sympy.stats.crv
"""
from itertools import product

from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.function import Lambda
from sympy.core.mul import Mul
from sympy.core.numbers import (I, nan)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.piecewise import Piecewise
from sympy.logic.boolalg import (And, Or)
from sympy.sets.sets import Intersection
from sympy.core.containers import Dict
from sympy.core.logic import Logic
from sympy.core.relational import Relational
from sympy.core.sympify import _sympify
from sympy.sets.sets import FiniteSet
from sympy.stats.rv import (RandomDomain, ProductDomain, ConditionalDomain,
                            PSpace, IndependentProductPSpace, SinglePSpace, random_symbols,
                            sumsets, rv_subs, NamedArgsMixin, Density, Distribution)


class FiniteDensity(dict):
    """
    A domain with Finite Density.
    """
    def __call__(self, item):
        """
        Make instance of a class callable.

        If item belongs to current instance of a class, return it.

        Otherwise, return 0.
        """
        item = sympify(item)
        if item in self:
            return self[item]
        else:
            return 0

    @property
    def dict(self):
        """
        Return item as dictionary.
        """
        return dict(self)

class FiniteDomain(RandomDomain):
    """
    A domain with discrete finite support

    Represented using a FiniteSet.
    """
    is_Finite = True

    @property
    def symbols(self):
        return FiniteSet(sym for sym, val in self.elements)

    @property
    def elements(self):
        return self.args[0]

    @property
    def dict(self):
        return FiniteSet(*[Dict(dict(el)) for el in self.elements])

    def __contains__(self, other):
        return other in self.elements

    def __iter__(self):
        return self.elements.__iter__()

    def as_boolean(self):
        return Or(*[And(*[Eq(sym, val) for sym, val in item]) for item in self])


class SingleFiniteDomain(FiniteDomain):
    """
    A FiniteDomain over a single symbol/set

    Example: The possibilities of a *single* die roll.
    """

    def __new__(cls, symbol, set):
        if not isinstance(set, FiniteSet) and \
            not isinstance(set, Intersection):
            set = FiniteSet(*set)
        return Basic.__new__(cls, symbol, set)

    @property
    def symbol(self):
        return self.args[0]

    @property
    def symbols(self):
        return FiniteSet(self.symbol)

    @property
    def set(self):
        return self.args[1]

    @property
    def elements(self):
        return FiniteSet(*[frozenset(((self.symbol, elem), )) for elem in self.set])

    def __iter__(self):
        return (frozenset(((self.symbol, elem),)) for elem in self.set)

    def __contains__(self, other):
        sym, val = tuple(other)[0]
        return sym == self.symbol and val in self.set


class ProductFiniteDomain(ProductDomain, FiniteDomain):
    """
    A Finite domain consisting of several other FiniteDomains

    Example: The possibilities of the rolls of three independent dice
    """

    def __iter__(self):
        proditer = product(*self.domains)
        return (sumsets(items) for items in proditer)

    @property
    def elements(self):
        return FiniteSet(*self)


class ConditionalFiniteDomain(ConditionalDomain, ProductFiniteDomain):
    """
    A FiniteDomain that has been restricted by a condition

    Example: The possibilities of a die roll under the condition that the
    roll is even.
    """

    def __new__(cls, domain, condition):
        """
        Create a new instance of ConditionalFiniteDomain class
        """
        if condition is True:
            return domain
        cond = rv_subs(condition)
        return Basic.__new__(cls, domain, cond)

    def _test(self, elem):
        """
        Test the value. If value is boolean, return it. If value is equality
        relational (two objects are equal), return it with left-hand side
        being equal to right-hand side. Otherwise, raise ValueError exception.
        """
        val = self.condition.xreplace(dict(elem))
        if val in [True, False]:
            return val
        elif val.is_Equality:
            return val.lhs == val.rhs
        raise ValueError("Undecidable if %s" % str(val))

    def __contains__(self, other):
        return other in self.fulldomain and self._test(other)

    def __iter__(self):
        return (elem for elem in self.fulldomain if self._test(elem))

    @property
    def set(self):
        if isinstance(self.fulldomain, SingleFiniteDomain):
            return FiniteSet(*[elem for elem in self.fulldomain.set
                               if frozenset(((self.fulldomain.symbol, elem),)) in self])
        else:
            raise NotImplementedError(
                "Not implemented on multi-dimensional conditional domain")

    def as_boolean(self):
        return FiniteDomain.as_boolean(self)


class SingleFiniteDistribution(Distribution, NamedArgsMixin):
    def __new__(cls, *args):
        args = list(map(sympify, args))
        return Basic.__new__(cls, *args)

    @staticmethod
    def check(*args):
        pass

    @property # type: ignore
    @cacheit
    def dict(self):
        if self.is_symbolic:
            return Density(self)
        return {k: self.pmf(k) for k in self.set}

    def pmf(self, *args): # to be overridden by specific distribution
        raise NotImplementedError()

    @property
    def set(self): # to be overridden by specific distribution
        raise NotImplementedError()

    values = property(lambda self: self.dict.values)
    items = property(lambda self: self.dict.items)
    is_symbolic = property(lambda self: False)
    __iter__ = property(lambda self: self.dict.__iter__)
    __getitem__ = property(lambda self: self.dict.__getitem__)

    def __call__(self, *args):
        return self.pmf(*args)

    def __contains__(self, other):
        return other in self.set


#=============================================
#=========  Probability Space  ===============
#=============================================


class FinitePSpace(PSpace):
    """
    A Finite Probability Space

    Represents the probabilities of a finite number of events.
    """
    is_Finite = True

    def __new__(cls, domain, density):
        density = {sympify(key): sympify(val)
                for key, val in density.items()}
        public_density = Dict(density)

        obj = PSpace.__new__(cls, domain, public_density)
        obj._density = density
        return obj

    def prob_of(self, elem):
        elem = sympify(elem)
        density = self._density
        if isinstance(list(density.keys())[0], FiniteSet):
            return density.get(elem, S.Zero)
        return density.get(tuple(elem)[0][1], S.Zero)

    def where(self, condition):
        assert all(r.symbol in self.symbols for r in random_symbols(condition))
        return ConditionalFiniteDomain(self.domain, condition)

    def compute_density(self, expr):
        expr = rv_subs(expr, self.values)
        d = FiniteDensity()
        for elem in self.domain:
            val = expr.xreplace(dict(elem))
            prob = self.prob_of(elem)
            d[val] = d.get(val, S.Zero) + prob
        return d

    @cacheit
    def compute_cdf(self, expr):
        d = self.compute_density(expr)
        cum_prob = S.Zero
        cdf = []
        for key in sorted(d):
            prob = d[key]
            cum_prob += prob
            cdf.append((key, cum_prob))

        return dict(cdf)

    @cacheit
    def sorted_cdf(self, expr, python_float=False):
        cdf = self.compute_cdf(expr)
        items = list(cdf.items())
        sorted_items = sorted(items, key=lambda val_cumprob: val_cumprob[1])
        if python_float:
            sorted_items = [(v, float(cum_prob))
                    for v, cum_prob in sorted_items]
        return sorted_items

    @cacheit
    def compute_characteristic_function(self, expr):
        d = self.compute_density(expr)
        t = Dummy('t', real=True)

        return Lambda(t, sum(exp(I*k*t)*v for k,v in d.items()))

    @cacheit
    def compute_moment_generating_function(self, expr):
        d = self.compute_density(expr)
        t = Dummy('t', real=True)

        return Lambda(t, sum(exp(k*t)*v for k,v in d.items()))

    def compute_expectation(self, expr, rvs=None, **kwargs):
        rvs = rvs or self.values
        expr = rv_subs(expr, rvs)
        probs = [self.prob_of(elem) for elem in self.domain]
        if isinstance(expr, (Logic, Relational)):
            parse_domain = [tuple(elem)[0][1] for elem in self.domain]
            bools = [expr.xreplace(dict(elem)) for elem in self.domain]
        else:
            parse_domain = [expr.xreplace(dict(elem)) for elem in self.domain]
            bools = [True for elem in self.domain]
        return sum(Piecewise((prob * elem, blv), (S.Zero, True))
                for prob, elem, blv in zip(probs, parse_domain, bools))

    def compute_quantile(self, expr):
        cdf = self.compute_cdf(expr)
        p = Dummy('p', real=True)
        set = ((nan, (p < 0) | (p > 1)),)
        for key, value in cdf.items():
            set = set + ((key, p <= value), )
        return Lambda(p, Piecewise(*set))

    def probability(self, condition):
        cond_symbols = frozenset(rs.symbol for rs in random_symbols(condition))
        cond = rv_subs(condition)
        if not cond_symbols.issubset(self.symbols):
            raise ValueError("Cannot compare foreign random symbols, %s"
                             %(str(cond_symbols - self.symbols)))
        if isinstance(condition, Relational) and \
            (not cond.free_symbols.issubset(self.domain.free_symbols)):
            rv = condition.lhs if isinstance(condition.rhs, Symbol) else condition.rhs
            return sum(Piecewise(
                       (self.prob_of(elem), condition.subs(rv, list(elem)[0][1])),
                       (S.Zero, True)) for elem in self.domain)
        return sympify(sum(self.prob_of(elem) for elem in self.where(condition)))

    def conditional_space(self, condition):
        domain = self.where(condition)
        prob = self.probability(condition)
        density = {key: val / prob
                for key, val in self._density.items() if domain._test(key)}
        return FinitePSpace(domain, density)

    def sample(self, size=(), library='scipy', seed=None):
        """
        Internal sample method

        Returns dictionary mapping RandomSymbol to realization value.
        """
        return {self.value: self.distribution.sample(size, library, seed)}


class SingleFinitePSpace(SinglePSpace, FinitePSpace):
    """
    A single finite probability space

    Represents the probabilities of a set of random events that can be
    attributed to a single variable/symbol.

    This class is implemented by many of the standard FiniteRV types such as
    Die, Bernoulli, Coin, etc....
    """
    @property
    def domain(self):
        return SingleFiniteDomain(self.symbol, self.distribution.set)

    @property
    def _is_symbolic(self):
        """
        Helper property to check if the distribution
        of the random variable is having symbolic
        dimension.
        """
        return self.distribution.is_symbolic

    @property
    def distribution(self):
        return self.args[1]

    def pmf(self, expr):
        return self.distribution.pmf(expr)

    @property # type: ignore
    @cacheit
    def _density(self):
        return {FiniteSet((self.symbol, val)): prob
                    for val, prob in self.distribution.dict.items()}

    @cacheit
    def compute_characteristic_function(self, expr):
        if self._is_symbolic:
            d = self.compute_density(expr)
            t = Dummy('t', real=True)
            ki = Dummy('ki')
            return Lambda(t, Sum(d(ki)*exp(I*ki*t), (ki, self.args[1].low, self.args[1].high)))
        expr = rv_subs(expr, self.values)
        return FinitePSpace(self.domain, self.distribution).compute_characteristic_function(expr)

    @cacheit
    def compute_moment_generating_function(self, expr):
        if self._is_symbolic:
            d = self.compute_density(expr)
            t = Dummy('t', real=True)
            ki = Dummy('ki')
            return Lambda(t, Sum(d(ki)*exp(ki*t), (ki, self.args[1].low, self.args[1].high)))
        expr = rv_subs(expr, self.values)
        return FinitePSpace(self.domain, self.distribution).compute_moment_generating_function(expr)

    def compute_quantile(self, expr):
        if self._is_symbolic:
            raise NotImplementedError("Computing quantile for random variables "
            "with symbolic dimension because the bounds of searching the required "
            "value is undetermined.")
        expr = rv_subs(expr, self.values)
        return FinitePSpace(self.domain, self.distribution).compute_quantile(expr)

    def compute_density(self, expr):
        if self._is_symbolic:
            rv = list(random_symbols(expr))[0]
            k = Dummy('k', integer=True)
            cond = True if not isinstance(expr, (Relational, Logic)) \
                     else expr.subs(rv, k)
            return Lambda(k,
            Piecewise((self.pmf(k), And(k >= self.args[1].low,
            k <= self.args[1].high, cond)), (S.Zero, True)))
        expr = rv_subs(expr, self.values)
        return FinitePSpace(self.domain, self.distribution).compute_density(expr)

    def compute_cdf(self, expr):
        if self._is_symbolic:
            d = self.compute_density(expr)
            k = Dummy('k')
            ki = Dummy('ki')
            return Lambda(k, Sum(d(ki), (ki, self.args[1].low, k)))
        expr = rv_subs(expr, self.values)
        return FinitePSpace(self.domain, self.distribution).compute_cdf(expr)

    def compute_expectation(self, expr, rvs=None, **kwargs):
        if self._is_symbolic:
            rv = random_symbols(expr)[0]
            k = Dummy('k', integer=True)
            expr = expr.subs(rv, k)
            cond = True if not isinstance(expr, (Relational, Logic)) \
                    else expr
            func = self.pmf(k) * k if cond != True else self.pmf(k) * expr
            return Sum(Piecewise((func, cond), (S.Zero, True)),
                (k, self.distribution.low, self.distribution.high)).doit()

        expr = _sympify(expr)
        expr = rv_subs(expr, rvs)
        return FinitePSpace(self.domain, self.distribution).compute_expectation(expr, rvs, **kwargs)

    def probability(self, condition):
        if self._is_symbolic:
            #TODO: Implement the mechanism for handling queries for symbolic sized distributions.
            raise NotImplementedError("Currently, probability queries are not "
            "supported for random variables with symbolic sized distributions.")
        condition = rv_subs(condition)
        return FinitePSpace(self.domain, self.distribution).probability(condition)

    def conditional_space(self, condition):
        """
        This method is used for transferring the
        computation to probability method because
        conditional space of random variables with
        symbolic dimensions is currently not possible.
        """
        if self._is_symbolic:
            self
        domain = self.where(condition)
        prob = self.probability(condition)
        density = {key: val / prob
                for key, val in self._density.items() if domain._test(key)}
        return FinitePSpace(domain, density)


class ProductFinitePSpace(IndependentProductPSpace, FinitePSpace):
    """
    A collection of several independent finite probability spaces
    """
    @property
    def domain(self):
        return ProductFiniteDomain(*[space.domain for space in self.spaces])

    @property  # type: ignore
    @cacheit
    def _density(self):
        proditer = product(*[iter(space._density.items())
            for space in self.spaces])
        d = {}
        for items in proditer:
            elems, probs = list(zip(*items))
            elem = sumsets(elems)
            prob = Mul(*probs)
            d[elem] = d.get(elem, S.Zero) + prob
        return Dict(d)

    @property  # type: ignore
    @cacheit
    def density(self):
        return Dict(self._density)

    def probability(self, condition):
        return FinitePSpace.probability(self, condition)

    def compute_density(self, expr):
        return FinitePSpace.compute_density(self, expr)
<|endoftext|>"""
SymPy statistics module

Introduces a random variable type into the SymPy language.

Random variables may be declared using prebuilt functions such as
Normal, Exponential, Coin, Die, etc...  or built with functions like FiniteRV.

Queries on random expressions can be made using the functions

========================= =============================
    Expression                    Meaning
------------------------- -----------------------------
 ``P(condition)``          Probability
 ``E(expression)``         Expected value
 ``H(expression)``         Entropy
 ``variance(expression)``  Variance
 ``density(expression)``   Probability Density Function
 ``sample(expression)``    Produce a realization
 ``where(condition)``      Where the condition is true
========================= =============================

Examples
========

>>> from sympy.stats import P, E, variance, Die, Normal
>>> from sympy import simplify
>>> X, Y = Die('X', 6), Die('Y', 6) # Define two six sided dice
>>> Z = Normal('Z', 0, 1) # Declare a Normal random variable with mean 0, std 1
>>> P(X>3) # Probability X is greater than 3
1/2
>>> E(X+Y) # Expectation of the sum of two dice
7
>>> variance(X+Y) # Variance of the sum of two dice
35/6
>>> simplify(P(Z>1)) # Probability of Z being greater than 1
1/2 - erf(sqrt(2)/2)/2


One could also create custom distribution and define custom random variables
as follows:

1. If you want to create a Continuous Random Variable:

>>> from sympy.stats import ContinuousRV, P, E
>>> from sympy import exp, Symbol, Interval, oo
>>> x = Symbol('x')
>>> pdf = exp(-x) # pdf of the Continuous Distribution
>>> Z = ContinuousRV(x, pdf, set=Interval(0, oo))
>>> E(Z)
1
>>> P(Z > 5)
exp(-5)

1.1 To create an instance of Continuous Distribution:

>>> from sympy.stats import ContinuousDistributionHandmade
>>> from sympy import Lambda
>>> dist = ContinuousDistributionHandmade(Lambda(x, pdf), set=Interval(0, oo))
>>> dist.pdf(x)
exp(-x)

2. If you want to create a Discrete Random Variable:

>>> from sympy.stats import DiscreteRV, P, E
>>> from sympy import Symbol, S
>>> p = S(1)/2
>>> x = Symbol('x', integer=True, positive=True)
>>> pdf = p*(1 - p)**(x - 1)
>>> D = DiscreteRV(x, pdf, set=S.Naturals)
>>> E(D)
2
>>> P(D > 3)
1/8

2.1 To create an instance of Discrete Distribution:

>>> from sympy.stats import DiscreteDistributionHandmade
>>> from sympy import Lambda
>>> dist = DiscreteDistributionHandmade(Lambda(x, pdf), set=S.Naturals)
>>> dist.pdf(x)
2**(1 - x)/2

3. If you want to create a Finite Random Variable:

>>> from sympy.stats import FiniteRV, P, E
>>> from sympy import Rational, Eq
>>> pmf = {1: Rational(1, 3), 2: Rational(1, 6), 3: Rational(1, 4), 4: Rational(1, 4)}
>>> X = FiniteRV('X', pmf)
>>> E(X)
29/12
>>> P(X > 3)
1/4

3.1 To create an instance of Finite Distribution:

>>> from sympy.stats import FiniteDistributionHandmade
>>> dist = FiniteDistributionHandmade(pmf)
>>> dist.pmf(x)
Lambda(x, Piecewise((1/3, Eq(x, 1)), (1/6, Eq(x, 2)), (1/4, Eq(x, 3) | Eq(x, 4)), (0, True)))
"""

__all__ = [
    'P', 'E', 'H', 'density', 'where', 'given', 'sample', 'cdf','median',
    'characteristic_function', 'pspace', 'sample_iter', 'variance', 'std',
    'skewness', 'kurtosis', 'covariance', 'dependent', 'entropy', 'independent',
    'random_symbols', 'correlation', 'factorial_moment', 'moment', 'cmoment',
    'sampling_density', 'moment_generating_function', 'smoment', 'quantile',
    'coskewness', 'sample_stochastic_process',

    'FiniteRV', 'DiscreteUniform', 'Die', 'Bernoulli', 'Coin', 'Binomial',
    'BetaBinomial', 'Hypergeometric', 'Rademacher', 'IdealSoliton', 'RobustSoliton',
    'FiniteDistributionHandmade',

    'ContinuousRV', 'Arcsin', 'Benini', 'Beta', 'BetaNoncentral', 'BetaPrime',
    'BoundedPareto', 'Cauchy', 'Chi', 'ChiNoncentral', 'ChiSquared', 'Dagum', 'Davis', 'Erlang',
    'ExGaussian', 'Exponential', 'ExponentialPower', 'FDistribution',
    'FisherZ', 'Frechet', 'Gamma', 'GammaInverse', 'Gompertz', 'Gumbel',
    'Kumaraswamy', 'Laplace', 'Levy', 'Logistic','LogCauchy', 'LogLogistic', 'LogitNormal', 'LogNormal', 'Lomax',
    'Moyal', 'Maxwell', 'Nakagami', 'Normal', 'GaussianInverse', 'Pareto', 'PowerFunction',
    'QuadraticU', 'RaisedCosine', 'Rayleigh','Reciprocal', 'StudentT', 'ShiftedGompertz',
    'Trapezoidal', 'Triangular', 'Uniform', 'UniformSum', 'VonMises', 'Wald',
    'Weibull', 'WignerSemicircle', 'ContinuousDistributionHandmade',

    'FlorySchulz', 'Geometric','Hermite', 'Logarithmic', 'NegativeBinomial', 'Poisson', 'Skellam',
    'YuleSimon', 'Zeta', 'DiscreteRV', 'DiscreteDistributionHandmade',

    'JointRV', 'Dirichlet', 'GeneralizedMultivariateLogGamma',
    'GeneralizedMultivariateLogGammaOmega', 'Multinomial', 'MultivariateBeta',
    'MultivariateEwens', 'MultivariateT', 'NegativeMultinomial',
    'NormalGamma', 'MultivariateNormal', 'MultivariateLaplace', 'marginal_distribution',

    'StochasticProcess', 'DiscreteTimeStochasticProcess',
    'DiscreteMarkovChain', 'TransitionMatrixOf', 'StochasticStateSpaceOf',
    'GeneratorMatrixOf', 'ContinuousMarkovChain', 'BernoulliProcess',
    'PoissonProcess', 'WienerProcess', 'GammaProcess',

    'CircularEnsemble', 'CircularUnitaryEnsemble',
    'CircularOrthogonalEnsemble', 'CircularSymplecticEnsemble',
    'GaussianEnsemble', 'GaussianUnitaryEnsemble',
    'GaussianOrthogonalEnsemble', 'GaussianSymplecticEnsemble',
    'joint_eigen_distribution', 'JointEigenDistribution',
    'level_spacing_distribution',

    'MatrixGamma', 'Wishart', 'MatrixNormal', 'MatrixStudentT',

    'Probability', 'Expectation', 'Variance', 'Covariance', 'Moment',
    'CentralMoment',

    'ExpectationMatrix', 'VarianceMatrix', 'CrossCovarianceMatrix'

]
from .rv_interface import (P, E, H, density, where, given, sample, cdf, median,
        characteristic_function, pspace, sample_iter, variance, std, skewness,
        kurtosis, covariance, dependent, entropy, independent, random_symbols,
        correlation, factorial_moment, moment, cmoment, sampling_density,
        moment_generating_function, smoment, quantile, coskewness,
        sample_stochastic_process)

from .frv_types import (FiniteRV, DiscreteUniform, Die, Bernoulli, Coin,
        Binomial, BetaBinomial, Hypergeometric, Rademacher,
        FiniteDistributionHandmade, IdealSoliton, RobustSoliton)

from .crv_types import (ContinuousRV, Arcsin, Benini, Beta, BetaNoncentral,
        BetaPrime, BoundedPareto, Cauchy, Chi, ChiNoncentral, ChiSquared,
        Dagum, Davis, Erlang, ExGaussian, Exponential, ExponentialPower,
        FDistribution, FisherZ, Frechet, Gamma, GammaInverse, GaussianInverse,
        Gompertz, Gumbel, Kumaraswamy, Laplace, Levy, Logistic, LogCauchy,
        LogLogistic, LogitNormal, LogNormal, Lomax, Maxwell, Moyal, Nakagami,
        Normal, Pareto, QuadraticU, RaisedCosine, Rayleigh, Reciprocal,
        StudentT, PowerFunction, ShiftedGompertz, Trapezoidal, Triangular,
        Uniform, UniformSum, VonMises, Wald, Weibull, WignerSemicircle,
        ContinuousDistributionHandmade)

from .drv_types import (FlorySchulz, Geometric, Hermite, Logarithmic, NegativeBinomial, Poisson,
        Skellam, YuleSimon, Zeta, DiscreteRV, DiscreteDistributionHandmade)

from .joint_rv_types import (JointRV, Dirichlet,
        GeneralizedMultivariateLogGamma, GeneralizedMultivariateLogGammaOmega,
        Multinomial, MultivariateBeta, MultivariateEwens, MultivariateT,
        NegativeMultinomial, NormalGamma, MultivariateNormal, MultivariateLaplace,
        marginal_distribution)

from .stochastic_process_types import (StochasticProcess,
        DiscreteTimeStochasticProcess, DiscreteMarkovChain,
        TransitionMatrixOf, StochasticStateSpaceOf, GeneratorMatrixOf,
        ContinuousMarkovChain, BernoulliProcess, PoissonProcess, WienerProcess,
        GammaProcess)

from .random_matrix_models import (CircularEnsemble, CircularUnitaryEnsemble,
        CircularOrthogonalEnsemble, CircularSymplecticEnsemble,
        GaussianEnsemble, GaussianUnitaryEnsemble, GaussianOrthogonalEnsemble,
        GaussianSymplecticEnsemble, joint_eigen_distribution,
        JointEigenDistribution, level_spacing_distribution)

from .matrix_distributions import MatrixGamma, Wishart, MatrixNormal, MatrixStudentT

from .symbolic_probability import (Probability, Expectation, Variance,
        Covariance, Moment, CentralMoment)

from .symbolic_multivariate_probability import (ExpectationMatrix, VarianceMatrix,
        CrossCovarianceMatrix)
<|endoftext|>"""
Continuous Random Variables - Prebuilt variables

Contains
========
Arcsin
Benini
Beta
BetaNoncentral
BetaPrime
BoundedPareto
Cauchy
Chi
ChiNoncentral
ChiSquared
Dagum
Davis
Erlang
ExGaussian
Exponential
ExponentialPower
FDistribution
FisherZ
Frechet
Gamma
GammaInverse
Gumbel
Gompertz
Kumaraswamy
Laplace
Levy
LogCauchy
Logistic
LogLogistic
LogitNormal
LogNormal
Lomax
Maxwell
Moyal
Nakagami
Normal
Pareto
PowerFunction
QuadraticU
RaisedCosine
Rayleigh
Reciprocal
ShiftedGompertz
StudentT
Trapezoidal
Triangular
Uniform
UniformSum
VonMises
Wald
Weibull
WignerSemicircle
"""



from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import (atan, cos, sin, tan)
from sympy.functions.special.bessel import (besseli, besselj, besselk)
from sympy.functions.special.beta_functions import beta as beta_fn
from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.complexes import (Abs, sign)
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.hyperbolic import sinh
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt, Max, Min
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import asin
from sympy.functions.special.error_functions import (erf, erfc, erfi, erfinv, expint)
from sympy.functions.special.gamma_functions import (gamma, lowergamma, uppergamma)
from sympy.functions.special.zeta_functions import zeta
from sympy.functions.special.hyper import hyper
from sympy.integrals.integrals import integrate
from sympy.logic.boolalg import And
from sympy.sets.sets import Interval
from sympy.matrices import MatrixBase
from sympy.stats.crv import SingleContinuousPSpace, SingleContinuousDistribution
from sympy.stats.rv import _value_check, is_random

oo = S.Infinity

__all__ = ['ContinuousRV',
'Arcsin',
'Benini',
'Beta',
'BetaNoncentral',
'BetaPrime',
'BoundedPareto',
'Cauchy',
'Chi',
'ChiNoncentral',
'ChiSquared',
'Dagum',
'Davis',
'Erlang',
'ExGaussian',
'Exponential',
'ExponentialPower',
'FDistribution',
'FisherZ',
'Frechet',
'Gamma',
'GammaInverse',
'Gompertz',
'Gumbel',
'Kumaraswamy',
'Laplace',
'Levy',
'LogCauchy',
'Logistic',
'LogLogistic',
'LogitNormal',
'LogNormal',
'Lomax',
'Maxwell',
'Moyal',
'Nakagami',
'Normal',
'GaussianInverse',
'Pareto',
'PowerFunction',
'QuadraticU',
'RaisedCosine',
'Rayleigh',
'Reciprocal',
'StudentT',
'ShiftedGompertz',
'Trapezoidal',
'Triangular',
'Uniform',
'UniformSum',
'VonMises',
'Wald',
'Weibull',
'WignerSemicircle',
]


@is_random.register(MatrixBase)
def _(x):
    return any(is_random(i) for i in x)

def rv(symbol, cls, args, **kwargs):
    args = list(map(sympify, args))
    dist = cls(*args)
    if kwargs.pop('check', True):
        dist.check(*args)
    pspace = SingleContinuousPSpace(symbol, dist)
    if any(is_random(arg) for arg in args):
        from sympy.stats.compound_rv import CompoundPSpace, CompoundDistribution
        pspace = CompoundPSpace(symbol, CompoundDistribution(dist))
    return pspace.value


class ContinuousDistributionHandmade(SingleContinuousDistribution):
    _argnames = ('pdf',)

    def __new__(cls, pdf, set=Interval(-oo, oo)):
        return Basic.__new__(cls, pdf, set)

    @property
    def set(self):
        return self.args[1]

    @staticmethod
    def check(pdf, set):
        x = Dummy('x')
        val = integrate(pdf(x), (x, set))
        _value_check(Eq(val, 1) != S.false, "The pdf on the given set is incorrect.")


def ContinuousRV(symbol, density, set=Interval(-oo, oo), **kwargs):
    """
    Create a Continuous Random Variable given the following:

    Parameters
    ==========

    symbol : Symbol
        Represents name of the random variable.
    density : Expression containing symbol
        Represents probability density function.
    set : set/Interval
        Represents the region where the pdf is valid, by default is real line.
    check : bool
        If True, it will check whether the given density
        integrates to 1 over the given set. If False, it
        will not perform this check. Default is False.


    Returns
    =======

    RandomSymbol

    Many common continuous random variable types are already implemented.
    This function should be necessary only very rarely.


    Examples
    ========

    >>> from sympy import Symbol, sqrt, exp, pi
    >>> from sympy.stats import ContinuousRV, P, E

    >>> x = Symbol("x")

    >>> pdf = sqrt(2)*exp(-x**2/2)/(2*sqrt(pi)) # Normal distribution
    >>> X = ContinuousRV(x, pdf)

    >>> E(X)
    0
    >>> P(X>0)
    1/2
    """
    pdf = Piecewise((density, set.as_relational(symbol)), (0, True))
    pdf = Lambda(symbol, pdf)
    # have a default of False while `rv` should have a default of True
    kwargs['check'] = kwargs.pop('check', False)
    return rv(symbol.name, ContinuousDistributionHandmade, (pdf, set), **kwargs)

########################################
# Continuous Probability Distributions #
########################################

#-------------------------------------------------------------------------------
# Arcsin distribution ----------------------------------------------------------


class ArcsinDistribution(SingleContinuousDistribution):
    _argnames = ('a', 'b')

    @property
    def set(self):
        return Interval(self.a, self.b)

    def pdf(self, x):
        a, b = self.a, self.b
        return 1/(pi*sqrt((x - a)*(b - x)))

    def _cdf(self, x):
        a, b = self.a, self.b
        return Piecewise(
            (S.Zero, x < a),
            (2*asin(sqrt((x - a)/(b - a)))/pi, x <= b),
            (S.One, True))


def Arcsin(name, a=0, b=1):
    r"""
    Create a Continuous Random Variable with an arcsin distribution.

    The density of the arcsin distribution is given by

    .. math::
        f(x) := \frac{1}{\pi\sqrt{(x-a)(b-x)}}

    with :math:`x \in (a,b)`. It must hold that :math:`-\infty < a < b < \infty`.

    Parameters
    ==========

    a : Real number, the left interval boundary
    b : Real number, the right interval boundary

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Arcsin, density, cdf
    >>> from sympy import Symbol

    >>> a = Symbol("a", real=True)
    >>> b = Symbol("b", real=True)
    >>> z = Symbol("z")

    >>> X = Arcsin("x", a, b)

    >>> density(X)(z)
    1/(pi*sqrt((-a + z)*(b - z)))

    >>> cdf(X)(z)
    Piecewise((0, a > z),
            (2*asin(sqrt((-a + z)/(-a + b)))/pi, b >= z),
            (1, True))


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Arcsine_distribution

    """

    return rv(name, ArcsinDistribution, (a, b))

#-------------------------------------------------------------------------------
# Benini distribution ----------------------------------------------------------


class BeniniDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'beta', 'sigma')

    @staticmethod
    def check(alpha, beta, sigma):
        _value_check(alpha > 0, "Shape parameter Alpha must be positive.")
        _value_check(beta > 0, "Shape parameter Beta must be positive.")
        _value_check(sigma > 0, "Scale parameter Sigma must be positive.")

    @property
    def set(self):
        return Interval(self.sigma, oo)

    def pdf(self, x):
        alpha, beta, sigma = self.alpha, self.beta, self.sigma
        return (exp(-alpha*log(x/sigma) - beta*log(x/sigma)**2)
               *(alpha/x + 2*beta*log(x/sigma)/x))

    def _moment_generating_function(self, t):
        raise NotImplementedError('The moment generating function of the '
                                  'Benini distribution does not exist.')

def Benini(name, alpha, beta, sigma):
    r"""
    Create a Continuous Random Variable with a Benini distribution.

    The density of the Benini distribution is given by

    .. math::
        f(x) := e^{-\alpha\log{\frac{x}{\sigma}}
                -\beta\log^2\left[{\frac{x}{\sigma}}\right]}
                \left(\frac{\alpha}{x}+\frac{2\beta\log{\frac{x}{\sigma}}}{x}\right)

    This is a heavy-tailed distribution and is also known as the log-Rayleigh
    distribution.

    Parameters
    ==========

    alpha : Real number, `\alpha > 0`, a shape
    beta : Real number, `\beta > 0`, a shape
    sigma : Real number, `\sigma > 0`, a scale

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Benini, density, cdf
    >>> from sympy import Symbol, pprint

    >>> alpha = Symbol("alpha", positive=True)
    >>> beta = Symbol("beta", positive=True)
    >>> sigma = Symbol("sigma", positive=True)
    >>> z = Symbol("z")

    >>> X = Benini("x", alpha, beta, sigma)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
    /                  /  z  \\             /  z  \            2/  z  \
    |        2*beta*log|-----||  - alpha*log|-----| - beta*log  |-----|
    |alpha             \sigma/|             \sigma/             \sigma/
    |----- + -----------------|*e
    \  z             z        /

    >>> cdf(X)(z)
    Piecewise((1 - exp(-alpha*log(z/sigma) - beta*log(z/sigma)**2), sigma <= z),
            (0, True))


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Benini_distribution
    .. [2] https://reference.wolfram.com/legacy/v8/ref/BeniniDistribution.html

    """

    return rv(name, BeniniDistribution, (alpha, beta, sigma))

#-------------------------------------------------------------------------------
# Beta distribution ------------------------------------------------------------


class BetaDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'beta')

    set = Interval(0, 1)

    @staticmethod
    def check(alpha, beta):
        _value_check(alpha > 0, "Shape parameter Alpha must be positive.")
        _value_check(beta > 0, "Shape parameter Beta must be positive.")

    def pdf(self, x):
        alpha, beta = self.alpha, self.beta
        return x**(alpha - 1) * (1 - x)**(beta - 1) / beta_fn(alpha, beta)

    def _characteristic_function(self, t):
        return hyper((self.alpha,), (self.alpha + self.beta,), I*t)

    def _moment_generating_function(self, t):
        return hyper((self.alpha,), (self.alpha + self.beta,), t)


def Beta(name, alpha, beta):
    r"""
    Create a Continuous Random Variable with a Beta distribution.

    The density of the Beta distribution is given by

    .. math::
        f(x) := \frac{x^{\alpha-1}(1-x)^{\beta-1}} {\mathrm{B}(\alpha,\beta)}

    with :math:`x \in [0,1]`.

    Parameters
    ==========

    alpha : Real number, `\alpha > 0`, a shape
    beta : Real number, `\beta > 0`, a shape

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Beta, density, E, variance
    >>> from sympy import Symbol, simplify, pprint, factor

    >>> alpha = Symbol("alpha", positive=True)
    >>> beta = Symbol("beta", positive=True)
    >>> z = Symbol("z")

    >>> X = Beta("x", alpha, beta)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
     alpha - 1        beta - 1
    z         *(1 - z)
    --------------------------
          B(alpha, beta)

    >>> simplify(E(X))
    alpha/(alpha + beta)

    >>> factor(simplify(variance(X)))
    alpha*beta/((alpha + beta)**2*(alpha + beta + 1))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Beta_distribution
    .. [2] https://mathworld.wolfram.com/BetaDistribution.html

    """

    return rv(name, BetaDistribution, (alpha, beta))

#-------------------------------------------------------------------------------
# Noncentral Beta distribution ------------------------------------------------------------


class BetaNoncentralDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'beta', 'lamda')

    set = Interval(0, 1)

    @staticmethod
    def check(alpha, beta, lamda):
        _value_check(alpha > 0, "Shape parameter Alpha must be positive.")
        _value_check(beta > 0, "Shape parameter Beta must be positive.")
        _value_check(lamda >= 0, "Noncentrality parameter Lambda must be positive")

    def pdf(self, x):
        alpha, beta, lamda = self.alpha, self.beta, self.lamda
        k = Dummy("k")
        return Sum(exp(-lamda / 2) * (lamda / 2)**k * x**(alpha + k - 1) *(
            1 - x)**(beta - 1) / (factorial(k) * beta_fn(alpha + k, beta)), (k, 0, oo))

def BetaNoncentral(name, alpha, beta, lamda):
    r"""
    Create a Continuous Random Variable with a Type I Noncentral Beta distribution.

    The density of the Noncentral Beta distribution is given by

    .. math::
        f(x) := \sum_{k=0}^\infty e^{-\lambda/2}\frac{(\lambda/2)^k}{k!}
                \frac{x^{\alpha+k-1}(1-x)^{\beta-1}}{\mathrm{B}(\alpha+k,\beta)}

    with :math:`x \in [0,1]`.

    Parameters
    ==========

    alpha : Real number, `\alpha > 0`, a shape
    beta : Real number, `\beta > 0`, a shape
    lamda : Real number, `\lambda \geq 0`, noncentrality parameter

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import BetaNoncentral, density, cdf
    >>> from sympy import Symbol, pprint

    >>> alpha = Symbol("alpha", positive=True)
    >>> beta = Symbol("beta", positive=True)
    >>> lamda = Symbol("lamda", nonnegative=True)
    >>> z = Symbol("z")

    >>> X = BetaNoncentral("x", alpha, beta, lamda)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
      oo
    _____
    \    `
     \                                              -lamda
      \                          k                  -------
       \    k + alpha - 1 /lamda\         beta - 1     2
        )  z             *|-----| *(1 - z)        *e
       /                  \  2  /
      /    ------------------------------------------------
     /                  B(k + alpha, beta)*k!
    /____,
    k = 0

    Compute cdf with specific 'x', 'alpha', 'beta' and 'lamda' values as follows:

    >>> cdf(BetaNoncentral("x", 1, 1, 1), evaluate=False)(2).doit()
    2*exp(1/2)

    The argument evaluate=False prevents an attempt at evaluation
    of the sum for general x, before the argument 2 is passed.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Noncentral_beta_distribution
    .. [2] https://reference.wolfram.com/language/ref/NoncentralBetaDistribution.html

    """

    return rv(name, BetaNoncentralDistribution, (alpha, beta, lamda))


#-------------------------------------------------------------------------------
# Beta prime distribution ------------------------------------------------------


class BetaPrimeDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'beta')

    @staticmethod
    def check(alpha, beta):
        _value_check(alpha > 0, "Shape parameter Alpha must be positive.")
        _value_check(beta > 0, "Shape parameter Beta must be positive.")

    set = Interval(0, oo)

    def pdf(self, x):
        alpha, beta = self.alpha, self.beta
        return x**(alpha - 1)*(1 + x)**(-alpha - beta)/beta_fn(alpha, beta)

def BetaPrime(name, alpha, beta):
    r"""
    Create a continuous random variable with a Beta prime distribution.

    The density of the Beta prime distribution is given by

    .. math::
        f(x) := \frac{x^{\alpha-1} (1+x)^{-\alpha -\beta}}{B(\alpha,\beta)}

    with :math:`x > 0`.

    Parameters
    ==========

    alpha : Real number, `\alpha > 0`, a shape
    beta : Real number, `\beta > 0`, a shape

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import BetaPrime, density
    >>> from sympy import Symbol, pprint

    >>> alpha = Symbol("alpha", positive=True)
    >>> beta = Symbol("beta", positive=True)
    >>> z = Symbol("z")

    >>> X = BetaPrime("x", alpha, beta)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
     alpha - 1        -alpha - beta
    z         *(z + 1)
    -------------------------------
             B(alpha, beta)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Beta_prime_distribution
    .. [2] https://mathworld.wolfram.com/BetaPrimeDistribution.html

    """

    return rv(name, BetaPrimeDistribution, (alpha, beta))

#-------------------------------------------------------------------------------
# Bounded Pareto Distribution --------------------------------------------------
class BoundedParetoDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'left', 'right')

    @property
    def set(self):
        return Interval(self.left, self.right)

    @staticmethod
    def check(alpha, left, right):
        _value_check (alpha.is_positive, "Shape must be positive.")
        _value_check (left.is_positive, "Left value should be positive.")
        _value_check (right > left, "Right should be greater than left.")

    def pdf(self, x):
        alpha, left, right = self.alpha, self.left, self.right
        num = alpha * (left**alpha) * x**(- alpha -1)
        den = 1 - (left/right)**alpha
        return num/den

def BoundedPareto(name, alpha, left, right):
    r"""
    Create a continuous random variable with a Bounded Pareto distribution.

    The density of the Bounded Pareto distribution is given by

    .. math::
        f(x) := \frac{\alpha L^{\alpha}x^{-\alpha-1}}{1-(\frac{L}{H})^{\alpha}}

    Parameters
    ==========

    alpha : Real Number, `\alpha > 0`
        Shape parameter
    left : Real Number, `left > 0`
        Location parameter
    right : Real Number, `right > left`
        Location parameter

    Examples
    ========

    >>> from sympy.stats import BoundedPareto, density, cdf, E
    >>> from sympy import symbols
    >>> L, H = symbols('L, H', positive=True)
    >>> X = BoundedPareto('X', 2, L, H)
    >>> x = symbols('x')
    >>> density(X)(x)
    2*L**2/(x**3*(1 - L**2/H**2))
    >>> cdf(X)(x)
    Piecewise((-H**2*L**2/(x**2*(H**2 - L**2)) + H**2/(H**2 - L**2), L <= x), (0, True))
    >>> E(X).simplify()
    2*H*L/(H + L)

    Returns
    =======

    RandomSymbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Pareto_distribution#Bounded_Pareto_distribution

    """
    return rv (name, BoundedParetoDistribution, (alpha, left, right))

# ------------------------------------------------------------------------------
# Cauchy distribution ----------------------------------------------------------


class CauchyDistribution(SingleContinuousDistribution):
    _argnames = ('x0', 'gamma')

    @staticmethod
    def check(x0, gamma):
        _value_check(gamma > 0, "Scale parameter Gamma must be positive.")
        _value_check(x0.is_real, "Location parameter must be real.")

    def pdf(self, x):
        return 1/(pi*self.gamma*(1 + ((x - self.x0)/self.gamma)**2))

    def _cdf(self, x):
        x0, gamma = self.x0, self.gamma
        return (1/pi)*atan((x - x0)/gamma) + S.Half

    def _characteristic_function(self, t):
        return exp(self.x0 * I * t -  self.gamma * Abs(t))

    def _moment_generating_function(self, t):
        raise NotImplementedError("The moment generating function for the "
                                  "Cauchy distribution does not exist.")

    def _quantile(self, p):
        return self.x0 + self.gamma*tan(pi*(p - S.Half))


def Cauchy(name, x0, gamma):
    r"""
    Create a continuous random variable with a Cauchy distribution.

    The density of the Cauchy distribution is given by

    .. math::
        f(x) := \frac{1}{\pi \gamma [1 + {(\frac{x-x_0}{\gamma})}^2]}

    Parameters
    ==========

    x0 : Real number, the location
    gamma : Real number, `\gamma > 0`, a scale

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Cauchy, density
    >>> from sympy import Symbol

    >>> x0 = Symbol("x0")
    >>> gamma = Symbol("gamma", positive=True)
    >>> z = Symbol("z")

    >>> X = Cauchy("x", x0, gamma)

    >>> density(X)(z)
    1/(pi*gamma*(1 + (-x0 + z)**2/gamma**2))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Cauchy_distribution
    .. [2] https://mathworld.wolfram.com/CauchyDistribution.html

    """

    return rv(name, CauchyDistribution, (x0, gamma))

#-------------------------------------------------------------------------------
# Chi distribution -------------------------------------------------------------


class ChiDistribution(SingleContinuousDistribution):
    _argnames = ('k',)

    @staticmethod
    def check(k):
        _value_check(k > 0, "Number of degrees of freedom (k) must be positive.")
        _value_check(k.is_integer, "Number of degrees of freedom (k) must be an integer.")

    set = Interval(0, oo)

    def pdf(self, x):
        return 2**(1 - self.k/2)*x**(self.k - 1)*exp(-x**2/2)/gamma(self.k/2)

    def _characteristic_function(self, t):
        k = self.k

        part_1 = hyper((k/2,), (S.Half,), -t**2/2)
        part_2 = I*t*sqrt(2)*gamma((k+1)/2)/gamma(k/2)
        part_3 = hyper(((k+1)/2,), (Rational(3, 2),), -t**2/2)
        return part_1 + part_2*part_3

    def _moment_generating_function(self, t):
        k = self.k

        part_1 = hyper((k / 2,), (S.Half,), t ** 2 / 2)
        part_2 = t * sqrt(2) * gamma((k + 1) / 2) / gamma(k / 2)
        part_3 = hyper(((k + 1) / 2,), (S(3) / 2,), t ** 2 / 2)
        return part_1 + part_2 * part_3

def Chi(name, k):
    r"""
    Create a continuous random variable with a Chi distribution.

    The density of the Chi distribution is given by

    .. math::
        f(x) := \frac{2^{1-k/2}x^{k-1}e^{-x^2/2}}{\Gamma(k/2)}

    with :math:`x \geq 0`.

    Parameters
    ==========

    k : Positive integer, The number of degrees of freedom

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Chi, density, E
    >>> from sympy import Symbol, simplify

    >>> k = Symbol("k", integer=True)
    >>> z = Symbol("z")

    >>> X = Chi("x", k)

    >>> density(X)(z)
    2**(1 - k/2)*z**(k - 1)*exp(-z**2/2)/gamma(k/2)

    >>> simplify(E(X))
    sqrt(2)*gamma(k/2 + 1/2)/gamma(k/2)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Chi_distribution
    .. [2] https://mathworld.wolfram.com/ChiDistribution.html

    """

    return rv(name, ChiDistribution, (k,))

#-------------------------------------------------------------------------------
# Non-central Chi distribution -------------------------------------------------


class ChiNoncentralDistribution(SingleContinuousDistribution):
    _argnames = ('k', 'l')

    @staticmethod
    def check(k, l):
        _value_check(k > 0, "Number of degrees of freedom (k) must be positive.")
        _value_check(k.is_integer, "Number of degrees of freedom (k) must be an integer.")
        _value_check(l > 0, "Shift parameter Lambda must be positive.")

    set = Interval(0, oo)

    def pdf(self, x):
        k, l = self.k, self.l
        return exp(-(x**2+l**2)/2)*x**k*l / (l*x)**(k/2) * besseli(k/2-1, l*x)

def ChiNoncentral(name, k, l):
    r"""
    Create a continuous random variable with a non-central Chi distribution.

    Explanation
    ===========

    The density of the non-central Chi distribution is given by

    .. math::
        f(x) := \frac{e^{-(x^2+\lambda^2)/2} x^k\lambda}
                {(\lambda x)^{k/2}} I_{k/2-1}(\lambda x)

    with `x \geq 0`. Here, `I_\nu (x)` is the
    :ref:`modified Bessel function of the first kind <besseli>`.

    Parameters
    ==========

    k : A positive Integer, $k > 0$
        The number of degrees of freedom.
    lambda : Real number, `\lambda > 0`
        Shift parameter.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import ChiNoncentral, density
    >>> from sympy import Symbol

    >>> k = Symbol("k", integer=True)
    >>> l = Symbol("l")
    >>> z = Symbol("z")

    >>> X = ChiNoncentral("x", k, l)

    >>> density(X)(z)
    l*z**k*exp(-l**2/2 - z**2/2)*besseli(k/2 - 1, l*z)/(l*z)**(k/2)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Noncentral_chi_distribution
    """

    return rv(name, ChiNoncentralDistribution, (k, l))

#-------------------------------------------------------------------------------
# Chi squared distribution -----------------------------------------------------


class ChiSquaredDistribution(SingleContinuousDistribution):
    _argnames = ('k',)

    @staticmethod
    def check(k):
        _value_check(k > 0, "Number of degrees of freedom (k) must be positive.")
        _value_check(k.is_integer, "Number of degrees of freedom (k) must be an integer.")

    set = Interval(0, oo)

    def pdf(self, x):
        k = self.k
        return 1/(2**(k/2)*gamma(k/2))*x**(k/2 - 1)*exp(-x/2)

    def _cdf(self, x):
        k = self.k
        return Piecewise(
                (S.One/gamma(k/2)*lowergamma(k/2, x/2), x >= 0),
                (0, True)
        )

    def _characteristic_function(self, t):
        return (1 - 2*I*t)**(-self.k/2)

    def  _moment_generating_function(self, t):
        return (1 - 2*t)**(-self.k/2)


def ChiSquared(name, k):
    r"""
    Create a continuous random variable with a Chi-squared distribution.

    Explanation
    ===========

    The density of the Chi-squared distribution is given by

    .. math::
        f(x) := \frac{1}{2^{\frac{k}{2}}\Gamma\left(\frac{k}{2}\right)}
                x^{\frac{k}{2}-1} e^{-\frac{x}{2}}

    with :math:`x \geq 0`.

    Parameters
    ==========

    k : Positive integer
        The number of degrees of freedom.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import ChiSquared, density, E, variance, moment
    >>> from sympy import Symbol

    >>> k = Symbol("k", integer=True, positive=True)
    >>> z = Symbol("z")

    >>> X = ChiSquared("x", k)

    >>> density(X)(z)
    z**(k/2 - 1)*exp(-z/2)/(2**(k/2)*gamma(k/2))

    >>> E(X)
    k

    >>> variance(X)
    2*k

    >>> moment(X, 3)
    k**3 + 6*k**2 + 8*k

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Chi_squared_distribution
    .. [2] https://mathworld.wolfram.com/Chi-SquaredDistribution.html
    """

    return rv(name, ChiSquaredDistribution, (k, ))

#-------------------------------------------------------------------------------
# Dagum distribution -----------------------------------------------------------


class DagumDistribution(SingleContinuousDistribution):
    _argnames = ('p', 'a', 'b')

    set = Interval(0, oo)

    @staticmethod
    def check(p, a, b):
        _value_check(p > 0, "Shape parameter p must be positive.")
        _value_check(a > 0, "Shape parameter a must be positive.")
        _value_check(b > 0, "Scale parameter b must be positive.")

    def pdf(self, x):
        p, a, b = self.p, self.a, self.b
        return a*p/x*((x/b)**(a*p)/(((x/b)**a + 1)**(p + 1)))

    def _cdf(self, x):
        p, a, b = self.p, self.a, self.b
        return Piecewise(((S.One + (S(x)/b)**-a)**-p, x>=0),
                    (S.Zero, True))

def Dagum(name, p, a, b):
    r"""
    Create a continuous random variable with a Dagum distribution.

    Explanation
    ===========

    The density of the Dagum distribution is given by

    .. math::
        f(x) := \frac{a p}{x} \left( \frac{\left(\tfrac{x}{b}\right)^{a p}}
                {\left(\left(\tfrac{x}{b}\right)^a + 1 \right)^{p+1}} \right)

    with :math:`x > 0`.

    Parameters
    ==========

    p : Real number
        `p > 0`, a shape.
    a : Real number
        `a > 0`, a shape.
    b : Real number
        `b > 0`, a scale.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Dagum, density, cdf
    >>> from sympy import Symbol

    >>> p = Symbol("p", positive=True)
    >>> a = Symbol("a", positive=True)
    >>> b = Symbol("b", positive=True)
    >>> z = Symbol("z")

    >>> X = Dagum("x", p, a, b)

    >>> density(X)(z)
    a*p*(z/b)**(a*p)*((z/b)**a + 1)**(-p - 1)/z

    >>> cdf(X)(z)
    Piecewise(((1 + (z/b)**(-a))**(-p), z >= 0), (0, True))


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Dagum_distribution

    """

    return rv(name, DagumDistribution, (p, a, b))

#-------------------------------------------------------------------------------
# Davis distribution -----------------------------------------------------------

class DavisDistribution(SingleContinuousDistribution):
    _argnames = ('b', 'n', 'mu')

    set = Interval(0, oo)

    @staticmethod
    def check(b, n, mu):
        _value_check(b > 0, "Scale parameter b must be positive.")
        _value_check(n > 1, "Shape parameter n must be above 1.")
        _value_check(mu > 0, "Location parameter mu must be positive.")

    def pdf(self, x):
        b, n, mu = self.b, self.n, self.mu
        dividend = b**n*(x - mu)**(-1-n)
        divisor = (exp(b/(x-mu))-1)*(gamma(n)*zeta(n))
        return dividend/divisor


def Davis(name, b, n, mu):
    r""" Create a continuous random variable with Davis distribution.

    Explanation
    ===========

    The density of Davis distribution is given by

    .. math::
        f(x; \mu; b, n) := \frac{b^{n}(x - \mu)^{1-n}}{ \left( e^{\frac{b}{x-\mu}} - 1 \right) \Gamma(n)\zeta(n)}

    with :math:`x \in [0,\infty]`.

    Davis distribution is a generalization of the Planck's law of radiation from statistical physics. It is used for modeling income distribution.

    Parameters
    ==========
    b : Real number
        `p > 0`, a scale.
    n : Real number
        `n > 1`, a shape.
    mu : Real number
        `mu > 0`, a location.

    Returns
    =======

    RandomSymbol

    Examples
    ========
    >>> from sympy.stats import Davis, density
    >>> from sympy import Symbol
    >>> b = Symbol("b", positive=True)
    >>> n = Symbol("n", positive=True)
    >>> mu = Symbol("mu", positive=True)
    >>> z = Symbol("z")
    >>> X = Davis("x", b, n, mu)
    >>> density(X)(z)
    b**n*(-mu + z)**(-n - 1)/((exp(b/(-mu + z)) - 1)*gamma(n)*zeta(n))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Davis_distribution
    .. [2] https://reference.wolfram.com/language/ref/DavisDistribution.html

    """
    return rv(name, DavisDistribution, (b, n, mu))


#-------------------------------------------------------------------------------
# Erlang distribution ----------------------------------------------------------


def Erlang(name, k, l):
    r"""
    Create a continuous random variable with an Erlang distribution.

    Explanation
    ===========

    The density of the Erlang distribution is given by

    .. math::
        f(x) := \frac{\lambda^k x^{k-1} e^{-\lambda x}}{(k-1)!}

    with :math:`x \in [0,\infty]`.

    Parameters
    ==========

    k : Positive integer
    l : Real number, `\lambda > 0`, the rate

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Erlang, density, cdf, E, variance
    >>> from sympy import Symbol, simplify, pprint

    >>> k = Symbol("k", integer=True, positive=True)
    >>> l = Symbol("l", positive=True)
    >>> z = Symbol("z")

    >>> X = Erlang("x", k, l)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
     k  k - 1  -l*z
    l *z     *e
    ---------------
        Gamma(k)

    >>> C = cdf(X)(z)
    >>> pprint(C, use_unicode=False)
    /lowergamma(k, l*z)
    |------------------  for z > 0
    <     Gamma(k)
    |
    \        0           otherwise


    >>> E(X)
    k/l

    >>> simplify(variance(X))
    k/l**2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Erlang_distribution
    .. [2] https://mathworld.wolfram.com/ErlangDistribution.html

    """

    return rv(name, GammaDistribution, (k, S.One/l))

# -------------------------------------------------------------------------------
# ExGaussian distribution -----------------------------------------------------


class ExGaussianDistribution(SingleContinuousDistribution):
    _argnames = ('mean', 'std', 'rate')

    set = Interval(-oo, oo)

    @staticmethod
    def check(mean, std, rate):
        _value_check(
            std > 0, "Standard deviation of ExGaussian must be positive.")
        _value_check(rate > 0, "Rate of ExGaussian must be positive.")

    def pdf(self, x):
        mean, std, rate = self.mean, self.std, self.rate
        term1 = rate/2
        term2 = exp(rate * (2 * mean + rate * std**2 - 2*x)/2)
        term3 = erfc((mean + rate*std**2 - x)/(sqrt(2)*std))
        return term1*term2*term3

    def _cdf(self, x):
        from sympy.stats import cdf
        mean, std, rate = self.mean, self.std, self.rate
        u = rate*(x - mean)
        v = rate*std
        GaussianCDF1 = cdf(Normal('x', 0, v))(u)
        GaussianCDF2 = cdf(Normal('x', v**2, v))(u)

        return GaussianCDF1 - exp(-u + (v**2/2) + log(GaussianCDF2))

    def _characteristic_function(self, t):
        mean, std, rate = self.mean, self.std, self.rate
        term1 = (1 - I*t/rate)**(-1)
        term2 = exp(I*mean*t - std**2*t**2/2)
        return term1 * term2

    def _moment_generating_function(self, t):
        mean, std, rate = self.mean, self.std, self.rate
        term1 = (1 - t/rate)**(-1)
        term2 = exp(mean*t + std**2*t**2/2)
        return term1*term2


def ExGaussian(name, mean, std, rate):
    r"""
    Create a continuous random variable with an Exponentially modified
    Gaussian (EMG) distribution.

    Explanation
    ===========

    The density of the exponentially modified Gaussian distribution is given by

    .. math::
        f(x) := \frac{\lambda}{2}e^{\frac{\lambda}{2}(2\mu+\lambda\sigma^2-2x)}
            \text{erfc}(\frac{\mu + \lambda\sigma^2 - x}{\sqrt{2}\sigma})

    with $x > 0$. Note that the expected value is `1/\lambda`.

    Parameters
    ==========

    name : A string giving a name for this distribution
    mean : A Real number, the mean of Gaussian component
    std : A positive Real number,
        :math: `\sigma^2 > 0` the variance of Gaussian component
    rate : A positive Real number,
        :math: `\lambda > 0` the rate of Exponential component

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import ExGaussian, density, cdf, E
    >>> from sympy.stats import variance, skewness
    >>> from sympy import Symbol, pprint, simplify

    >>> mean = Symbol("mu")
    >>> std = Symbol("sigma", positive=True)
    >>> rate = Symbol("lamda", positive=True)
    >>> z = Symbol("z")
    >>> X = ExGaussian("x", mean, std, rate)

    >>> pprint(density(X)(z), use_unicode=False)
                 /           2             \
           lamda*\lamda*sigma  + 2*mu - 2*z/
           ---------------------------------     /  ___ /           2         \\
                           2                     |\/ 2 *\lamda*sigma  + mu - z/|
    lamda*e                                 *erfc|-----------------------------|
                                                 \           2*sigma           /
    ----------------------------------------------------------------------------
                                         2

    >>> cdf(X)(z)
    -(erf(sqrt(2)*(-lamda**2*sigma**2 + lamda*(-mu + z))/(2*lamda*sigma))/2 + 1/2)*exp(lamda**2*sigma**2/2 - lamda*(-mu + z)) + erf(sqrt(2)*(-mu + z)/(2*sigma))/2 + 1/2

    >>> E(X)
    (lamda*mu + 1)/lamda

    >>> simplify(variance(X))
    sigma**2 + lamda**(-2)

    >>> simplify(skewness(X))
    2/(lamda**2*sigma**2 + 1)**(3/2)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution
    """
    return rv(name, ExGaussianDistribution, (mean, std, rate))

#-------------------------------------------------------------------------------
# Exponential distribution -----------------------------------------------------


class ExponentialDistribution(SingleContinuousDistribution):
    _argnames = ('rate',)

    set  = Interval(0, oo)

    @staticmethod
    def check(rate):
        _value_check(rate > 0, "Rate must be positive.")

    def pdf(self, x):
        return self.rate * exp(-self.rate*x)

    def _cdf(self, x):
        return Piecewise(
                (S.One - exp(-self.rate*x), x >= 0),
                (0, True),
        )

    def _characteristic_function(self, t):
        rate = self.rate
        return rate / (rate - I*t)

    def _moment_generating_function(self, t):
        rate = self.rate
        return rate / (rate - t)

    def _quantile(self, p):
        return -log(1-p)/self.rate


def Exponential(name, rate):
    r"""
    Create a continuous random variable with an Exponential distribution.

    Explanation
    ===========

    The density of the exponential distribution is given by

    .. math::
        f(x) := \lambda \exp(-\lambda x)

    with $x > 0$. Note that the expected value is `1/\lambda`.

    Parameters
    ==========

    rate : A positive Real number, `\lambda > 0`, the rate (or inverse scale/inverse mean)

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Exponential, density, cdf, E
    >>> from sympy.stats import variance, std, skewness, quantile
    >>> from sympy import Symbol

    >>> l = Symbol("lambda", positive=True)
    >>> z = Symbol("z")
    >>> p = Symbol("p")
    >>> X = Exponential("x", l)

    >>> density(X)(z)
    lambda*exp(-lambda*z)

    >>> cdf(X)(z)
    Piecewise((1 - exp(-lambda*z), z >= 0), (0, True))

    >>> quantile(X)(p)
    -log(1 - p)/lambda

    >>> E(X)
    1/lambda

    >>> variance(X)
    lambda**(-2)

    >>> skewness(X)
    2

    >>> X = Exponential('x', 10)

    >>> density(X)(z)
    10*exp(-10*z)

    >>> E(X)
    1/10

    >>> std(X)
    1/10

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Exponential_distribution
    .. [2] https://mathworld.wolfram.com/ExponentialDistribution.html

    """

    return rv(name, ExponentialDistribution, (rate, ))


# -------------------------------------------------------------------------------
# Exponential Power distribution -----------------------------------------------------

class ExponentialPowerDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 'alpha', 'beta')

    set = Interval(-oo, oo)

    @staticmethod
    def check(mu, alpha, beta):
        _value_check(alpha > 0, "Scale parameter alpha must be positive.")
        _value_check(beta > 0, "Shape parameter beta must be positive.")

    def pdf(self, x):
        mu, alpha, beta = self.mu, self.alpha, self.beta
        num = beta*exp(-(Abs(x - mu)/alpha)**beta)
        den = 2*alpha*gamma(1/beta)
        return num/den

    def _cdf(self, x):
        mu, alpha, beta = self.mu, self.alpha, self.beta
        num = lowergamma(1/beta, (Abs(x - mu) / alpha)**beta)
        den = 2*gamma(1/beta)
        return sign(x - mu)*num/den + S.Half


def ExponentialPower(name, mu, alpha, beta):
    r"""
    Create a Continuous Random Variable with Exponential Power distribution.
    This distribution is known also as Generalized Normal
    distribution version 1.

    Explanation
    ===========

    The density of the Exponential Power distribution is given by

    .. math::
        f(x) := \frac{\beta}{2\alpha\Gamma(\frac{1}{\beta})}
            e^{{-(\frac{|x - \mu|}{\alpha})^{\beta}}}

    with :math:`x \in [ - \infty, \infty ]`.

    Parameters
    ==========

    mu : Real number
        A location.
    alpha : Real number,`\alpha > 0`
        A  scale.
    beta : Real number, `\beta > 0`
        A shape.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import ExponentialPower, density, cdf
    >>> from sympy import Symbol, pprint
    >>> z = Symbol("z")
    >>> mu = Symbol("mu")
    >>> alpha = Symbol("alpha", positive=True)
    >>> beta = Symbol("beta", positive=True)
    >>> X = ExponentialPower("x", mu, alpha, beta)
    >>> pprint(density(X)(z), use_unicode=False)
                     beta
           /|mu - z|\
          -|--------|
           \ alpha  /
    beta*e
    ---------------------
                  / 1  \
     2*alpha*Gamma|----|
                  \beta/
    >>> cdf(X)(z)
    1/2 + lowergamma(1/beta, (Abs(mu - z)/alpha)**beta)*sign(-mu + z)/(2*gamma(1/beta))

    References
    ==========

    .. [1] https://reference.wolfram.com/language/ref/ExponentialPowerDistribution.html
    .. [2] https://en.wikipedia.org/wiki/Generalized_normal_distribution#Version_1

    """
    return rv(name, ExponentialPowerDistribution, (mu, alpha, beta))


#-------------------------------------------------------------------------------
# F distribution ---------------------------------------------------------------


class FDistributionDistribution(SingleContinuousDistribution):
    _argnames = ('d1', 'd2')

    set = Interval(0, oo)

    @staticmethod
    def check(d1, d2):
        _value_check((d1 > 0, d1.is_integer),
            "Degrees of freedom d1 must be positive integer.")
        _value_check((d2 > 0, d2.is_integer),
            "Degrees of freedom d2 must be positive integer.")

    def pdf(self, x):
        d1, d2 = self.d1, self.d2
        return (sqrt((d1*x)**d1*d2**d2 / (d1*x+d2)**(d1+d2))
               / (x * beta_fn(d1/2, d2/2)))

    def _moment_generating_function(self, t):
        raise NotImplementedError('The moment generating function for the '
                                  'F-distribution does not exist.')

def FDistribution(name, d1, d2):
    r"""
    Create a continuous random variable with a F distribution.

    Explanation
    ===========

    The density of the F distribution is given by

    .. math::
        f(x) := \frac{\sqrt{\frac{(d_1 x)^{d_1} d_2^{d_2}}
                {(d_1 x + d_2)^{d_1 + d_2}}}}
                {x \mathrm{B} \left(\frac{d_1}{2}, \frac{d_2}{2}\right)}

    with :math:`x > 0`.

    Parameters
    ==========

    d1 : `d_1 > 0`, where `d_1` is the degrees of freedom (`n_1 - 1`)
    d2 : `d_2 > 0`, where `d_2` is the degrees of freedom (`n_2 - 1`)

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import FDistribution, density
    >>> from sympy import Symbol, pprint

    >>> d1 = Symbol("d1", positive=True)
    >>> d2 = Symbol("d2", positive=True)
    >>> z = Symbol("z")

    >>> X = FDistribution("x", d1, d2)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
      d2
      --    ______________________________
      2    /       d1            -d1 - d2
    d2  *\/  (d1*z)  *(d1*z + d2)
    --------------------------------------
                    /d1  d2\
                 z*B|--, --|
                    \2   2 /

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/F-distribution
    .. [2] https://mathworld.wolfram.com/F-Distribution.html

    """

    return rv(name, FDistributionDistribution, (d1, d2))

#-------------------------------------------------------------------------------
# Fisher Z distribution --------------------------------------------------------

class FisherZDistribution(SingleContinuousDistribution):
    _argnames = ('d1', 'd2')

    set = Interval(-oo, oo)

    @staticmethod
    def check(d1, d2):
        _value_check(d1 > 0, "Degree of freedom d1 must be positive.")
        _value_check(d2 > 0, "Degree of freedom d2 must be positive.")

    def pdf(self, x):
        d1, d2 = self.d1, self.d2
        return (2*d1**(d1/2)*d2**(d2/2) / beta_fn(d1/2, d2/2) *
               exp(d1*x) / (d1*exp(2*x)+d2)**((d1+d2)/2))

def FisherZ(name, d1, d2):
    r"""
    Create a Continuous Random Variable with an Fisher's Z distribution.

    Explanation
    ===========

    The density of the Fisher's Z distribution is given by

    .. math::
        f(x) := \frac{2d_1^{d_1/2} d_2^{d_2/2}} {\mathrm{B}(d_1/2, d_2/2)}
                \frac{e^{d_1z}}{\left(d_1e^{2z}+d_2\right)^{\left(d_1+d_2\right)/2}}


    .. TODO - What is the difference between these degrees of freedom?

    Parameters
    ==========

    d1 : `d_1 > 0`
        Degree of freedom.
    d2 : `d_2 > 0`
        Degree of freedom.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import FisherZ, density
    >>> from sympy import Symbol, pprint

    >>> d1 = Symbol("d1", positive=True)
    >>> d2 = Symbol("d2", positive=True)
    >>> z = Symbol("z")

    >>> X = FisherZ("x", d1, d2)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
                                d1   d2
        d1   d2               - -- - --
        --   --                 2    2
        2    2  /    2*z     \           d1*z
    2*d1  *d2  *\d1*e    + d2/         *e
    -----------------------------------------
                     /d1  d2\
                    B|--, --|
                     \2   2 /

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fisher%27s_z-distribution
    .. [2] https://mathworld.wolfram.com/Fishersz-Distribution.html

    """

    return rv(name, FisherZDistribution, (d1, d2))

#-------------------------------------------------------------------------------
# Frechet distribution ---------------------------------------------------------

class FrechetDistribution(SingleContinuousDistribution):
    _argnames = ('a', 's', 'm')

    set = Interval(0, oo)

    @staticmethod
    def check(a, s, m):
        _value_check(a > 0, "Shape parameter alpha must be positive.")
        _value_check(s > 0, "Scale parameter s must be positive.")

    def __new__(cls, a, s=1, m=0):
        a, s, m = list(map(sympify, (a, s, m)))
        return Basic.__new__(cls, a, s, m)

    def pdf(self, x):
        a, s, m = self.a, self.s, self.m
        return a/s * ((x-m)/s)**(-1-a) * exp(-((x-m)/s)**(-a))

    def _cdf(self, x):
        a, s, m = self.a, self.s, self.m
        return Piecewise((exp(-((x-m)/s)**(-a)), x >= m),
                        (S.Zero, True))

def Frechet(name, a, s=1, m=0):
    r"""
    Create a continuous random variable with a Frechet distribution.

    Explanation
    ===========

