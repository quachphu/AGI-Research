    The density of the Frechet distribution is given by

    .. math::
        f(x) := \frac{\alpha}{s} \left(\frac{x-m}{s}\right)^{-1-\alpha}
                 e^{-(\frac{x-m}{s})^{-\alpha}}

    with :math:`x \geq m`.

    Parameters
    ==========

    a : Real number, :math:`a \in \left(0, \infty\right)` the shape
    s : Real number, :math:`s \in \left(0, \infty\right)` the scale
    m : Real number, :math:`m \in \left(-\infty, \infty\right)` the minimum

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Frechet, density, cdf
    >>> from sympy import Symbol

    >>> a = Symbol("a", positive=True)
    >>> s = Symbol("s", positive=True)
    >>> m = Symbol("m", real=True)
    >>> z = Symbol("z")

    >>> X = Frechet("x", a, s, m)

    >>> density(X)(z)
    a*((-m + z)/s)**(-a - 1)*exp(-1/((-m + z)/s)**a)/s

    >>> cdf(X)(z)
    Piecewise((exp(-1/((-m + z)/s)**a), m <= z), (0, True))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fr%C3%A9chet_distribution

    """

    return rv(name, FrechetDistribution, (a, s, m))

#-------------------------------------------------------------------------------
# Gamma distribution -----------------------------------------------------------


class GammaDistribution(SingleContinuousDistribution):
    _argnames = ('k', 'theta')

    set = Interval(0, oo)

    @staticmethod
    def check(k, theta):
        _value_check(k > 0, "k must be positive")
        _value_check(theta > 0, "Theta must be positive")

    def pdf(self, x):
        k, theta = self.k, self.theta
        return x**(k - 1) * exp(-x/theta) / (gamma(k)*theta**k)

    def _cdf(self, x):
        k, theta = self.k, self.theta
        return Piecewise(
                    (lowergamma(k, S(x)/theta)/gamma(k), x > 0),
                    (S.Zero, True))

    def _characteristic_function(self, t):
        return (1 - self.theta*I*t)**(-self.k)

    def _moment_generating_function(self, t):
        return (1- self.theta*t)**(-self.k)


def Gamma(name, k, theta):
    r"""
    Create a continuous random variable with a Gamma distribution.

    Explanation
    ===========

    The density of the Gamma distribution is given by

    .. math::
        f(x) := \frac{1}{\Gamma(k) \theta^k} x^{k - 1} e^{-\frac{x}{\theta}}

    with :math:`x \in [0,1]`.

    Parameters
    ==========

    k : Real number, `k > 0`, a shape
    theta : Real number, `\theta > 0`, a scale

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Gamma, density, cdf, E, variance
    >>> from sympy import Symbol, pprint, simplify

    >>> k = Symbol("k", positive=True)
    >>> theta = Symbol("theta", positive=True)
    >>> z = Symbol("z")

    >>> X = Gamma("x", k, theta)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
                      -z
                    -----
         -k  k - 1  theta
    theta  *z     *e
    ---------------------
           Gamma(k)

    >>> C = cdf(X, meijerg=True)(z)
    >>> pprint(C, use_unicode=False)
    /            /     z  \
    |k*lowergamma|k, -----|
    |            \   theta/
    <----------------------  for z >= 0
    |     Gamma(k + 1)
    |
    \          0             otherwise

    >>> E(X)
    k*theta

    >>> V = simplify(variance(X))
    >>> pprint(V, use_unicode=False)
           2
    k*theta


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gamma_distribution
    .. [2] https://mathworld.wolfram.com/GammaDistribution.html

    """

    return rv(name, GammaDistribution, (k, theta))

#-------------------------------------------------------------------------------
# Inverse Gamma distribution ---------------------------------------------------


class GammaInverseDistribution(SingleContinuousDistribution):
    _argnames = ('a', 'b')

    set = Interval(0, oo)

    @staticmethod
    def check(a, b):
        _value_check(a > 0, "alpha must be positive")
        _value_check(b > 0, "beta must be positive")

    def pdf(self, x):
        a, b = self.a, self.b
        return b**a/gamma(a) * x**(-a-1) * exp(-b/x)

    def _cdf(self, x):
        a, b = self.a, self.b
        return Piecewise((uppergamma(a,b/x)/gamma(a), x > 0),
                        (S.Zero, True))

    def _characteristic_function(self, t):
        a, b = self.a, self.b
        return 2 * (-I*b*t)**(a/2) * besselk(a, sqrt(-4*I*b*t)) / gamma(a)

    def _moment_generating_function(self, t):
        raise NotImplementedError('The moment generating function for the '
                                  'gamma inverse distribution does not exist.')

def GammaInverse(name, a, b):
    r"""
    Create a continuous random variable with an inverse Gamma distribution.

    Explanation
    ===========

    The density of the inverse Gamma distribution is given by

    .. math::
        f(x) := \frac{\beta^\alpha}{\Gamma(\alpha)} x^{-\alpha - 1}
                \exp\left(\frac{-\beta}{x}\right)

    with :math:`x > 0`.

    Parameters
    ==========

    a : Real number, `a > 0`, a shape
    b : Real number, `b > 0`, a scale

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import GammaInverse, density, cdf
    >>> from sympy import Symbol, pprint

    >>> a = Symbol("a", positive=True)
    >>> b = Symbol("b", positive=True)
    >>> z = Symbol("z")

    >>> X = GammaInverse("x", a, b)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
                -b
                ---
     a  -a - 1   z
    b *z      *e
    ---------------
       Gamma(a)

    >>> cdf(X)(z)
    Piecewise((uppergamma(a, b/z)/gamma(a), z > 0), (0, True))


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse-gamma_distribution

    """

    return rv(name, GammaInverseDistribution, (a, b))


#-------------------------------------------------------------------------------
# Gumbel distribution (Maximum and Minimum) --------------------------------------------------------


class GumbelDistribution(SingleContinuousDistribution):
    _argnames = ('beta', 'mu', 'minimum')

    set = Interval(-oo, oo)

    @staticmethod
    def check(beta, mu, minimum):
        _value_check(beta > 0, "Scale parameter beta must be positive.")

    def pdf(self, x):
        beta, mu = self.beta, self.mu
        z = (x - mu)/beta
        f_max = (1/beta)*exp(-z - exp(-z))
        f_min = (1/beta)*exp(z - exp(z))
        return Piecewise((f_min, self.minimum), (f_max, not self.minimum))

    def _cdf(self, x):
        beta, mu = self.beta, self.mu
        z = (x - mu)/beta
        F_max = exp(-exp(-z))
        F_min = 1 - exp(-exp(z))
        return Piecewise((F_min, self.minimum), (F_max, not self.minimum))

    def _characteristic_function(self, t):
        cf_max = gamma(1 - I*self.beta*t) * exp(I*self.mu*t)
        cf_min = gamma(1 + I*self.beta*t) * exp(I*self.mu*t)
        return Piecewise((cf_min, self.minimum), (cf_max, not self.minimum))

    def _moment_generating_function(self, t):
        mgf_max = gamma(1 - self.beta*t) * exp(self.mu*t)
        mgf_min = gamma(1 + self.beta*t) * exp(self.mu*t)
        return Piecewise((mgf_min, self.minimum), (mgf_max, not self.minimum))

def Gumbel(name, beta, mu, minimum=False):
    r"""
    Create a Continuous Random Variable with Gumbel distribution.

    Explanation
    ===========

    The density of the Gumbel distribution is given by

    For Maximum

    .. math::
        f(x) := \dfrac{1}{\beta} \exp \left( -\dfrac{x-\mu}{\beta}
                - \exp \left( -\dfrac{x - \mu}{\beta} \right) \right)

    with :math:`x \in [ - \infty, \infty ]`.

    For Minimum

    .. math::
        f(x) := \frac{e^{- e^{\frac{- \mu + x}{\beta}} + \frac{- \mu + x}{\beta}}}{\beta}

    with :math:`x \in [ - \infty, \infty ]`.

    Parameters
    ==========

    mu : Real number, `\mu`, a location
    beta : Real number, `\beta > 0`, a scale
    minimum : Boolean, by default ``False``, set to ``True`` for enabling minimum distribution

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Gumbel, density, cdf
    >>> from sympy import Symbol
    >>> x = Symbol("x")
    >>> mu = Symbol("mu")
    >>> beta = Symbol("beta", positive=True)
    >>> X = Gumbel("x", beta, mu)
    >>> density(X)(x)
    exp(-exp(-(-mu + x)/beta) - (-mu + x)/beta)/beta
    >>> cdf(X)(x)
    exp(-exp(-(-mu + x)/beta))

    References
    ==========

    .. [1] https://mathworld.wolfram.com/GumbelDistribution.html
    .. [2] https://en.wikipedia.org/wiki/Gumbel_distribution
    .. [3] https://web.archive.org/web/20200628222206/http://www.mathwave.com/help/easyfit/html/analyses/distributions/gumbel_max.html
    .. [4] https://web.archive.org/web/20200628222212/http://www.mathwave.com/help/easyfit/html/analyses/distributions/gumbel_min.html

    """
    return rv(name, GumbelDistribution, (beta, mu, minimum))

#-------------------------------------------------------------------------------
# Gompertz distribution --------------------------------------------------------

class GompertzDistribution(SingleContinuousDistribution):
    _argnames = ('b', 'eta')

    set = Interval(0, oo)

    @staticmethod
    def check(b, eta):
        _value_check(b > 0, "b must be positive")
        _value_check(eta > 0, "eta must be positive")

    def pdf(self, x):
        eta, b = self.eta, self.b
        return b*eta*exp(b*x)*exp(eta)*exp(-eta*exp(b*x))

    def _cdf(self, x):
        eta, b = self.eta, self.b
        return 1 - exp(eta)*exp(-eta*exp(b*x))

    def _moment_generating_function(self, t):
        eta, b = self.eta, self.b
        return eta * exp(eta) * expint(t/b, eta)

def Gompertz(name, b, eta):
    r"""
    Create a Continuous Random Variable with Gompertz distribution.

    Explanation
    ===========

    The density of the Gompertz distribution is given by

    .. math::
        f(x) := b \eta e^{b x} e^{\eta} \exp \left(-\eta e^{bx} \right)

    with :math:`x \in [0, \infty)`.

    Parameters
    ==========

    b : Real number, `b > 0`, a scale
    eta : Real number, `\eta > 0`, a shape

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Gompertz, density
    >>> from sympy import Symbol

    >>> b = Symbol("b", positive=True)
    >>> eta = Symbol("eta", positive=True)
    >>> z = Symbol("z")

    >>> X = Gompertz("x", b, eta)

    >>> density(X)(z)
    b*eta*exp(eta)*exp(b*z)*exp(-eta*exp(b*z))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gompertz_distribution

    """
    return rv(name, GompertzDistribution, (b, eta))

#-------------------------------------------------------------------------------
# Kumaraswamy distribution -----------------------------------------------------


class KumaraswamyDistribution(SingleContinuousDistribution):
    _argnames = ('a', 'b')

    set = Interval(0, oo)

    @staticmethod
    def check(a, b):
        _value_check(a > 0, "a must be positive")
        _value_check(b > 0, "b must be positive")

    def pdf(self, x):
        a, b = self.a, self.b
        return a * b * x**(a-1) * (1-x**a)**(b-1)

    def _cdf(self, x):
        a, b = self.a, self.b
        return Piecewise(
            (S.Zero, x < S.Zero),
            (1 - (1 - x**a)**b, x <= S.One),
            (S.One, True))

def Kumaraswamy(name, a, b):
    r"""
    Create a Continuous Random Variable with a Kumaraswamy distribution.

    Explanation
    ===========

    The density of the Kumaraswamy distribution is given by

    .. math::
        f(x) := a b x^{a-1} (1-x^a)^{b-1}

    with :math:`x \in [0,1]`.

    Parameters
    ==========

    a : Real number, `a > 0`, a shape
    b : Real number, `b > 0`, a shape

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Kumaraswamy, density, cdf
    >>> from sympy import Symbol, pprint

    >>> a = Symbol("a", positive=True)
    >>> b = Symbol("b", positive=True)
    >>> z = Symbol("z")

    >>> X = Kumaraswamy("x", a, b)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
                       b - 1
         a - 1 /     a\
    a*b*z     *\1 - z /

    >>> cdf(X)(z)
    Piecewise((0, z < 0), (1 - (1 - z**a)**b, z <= 1), (1, True))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Kumaraswamy_distribution

    """

    return rv(name, KumaraswamyDistribution, (a, b))

#-------------------------------------------------------------------------------
# Laplace distribution ---------------------------------------------------------


class LaplaceDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 'b')

    set = Interval(-oo, oo)

    @staticmethod
    def check(mu, b):
        _value_check(b > 0, "Scale parameter b must be positive.")
        _value_check(mu.is_real, "Location parameter mu should be real")

    def pdf(self, x):
        mu, b = self.mu, self.b
        return 1/(2*b)*exp(-Abs(x - mu)/b)

    def _cdf(self, x):
        mu, b = self.mu, self.b
        return Piecewise(
                    (S.Half*exp((x - mu)/b), x < mu),
                    (S.One - S.Half*exp(-(x - mu)/b), x >= mu)
                        )

    def _characteristic_function(self, t):
        return exp(self.mu*I*t) / (1 + self.b**2*t**2)

    def _moment_generating_function(self, t):
        return exp(self.mu*t) / (1 - self.b**2*t**2)

def Laplace(name, mu, b):
    r"""
    Create a continuous random variable with a Laplace distribution.

    Explanation
    ===========

    The density of the Laplace distribution is given by

    .. math::
        f(x) := \frac{1}{2 b} \exp \left(-\frac{|x-\mu|}b \right)

    Parameters
    ==========

    mu : Real number or a list/matrix, the location (mean) or the
        location vector
    b : Real number or a positive definite matrix, representing a scale
        or the covariance matrix.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Laplace, density, cdf
    >>> from sympy import Symbol, pprint

    >>> mu = Symbol("mu")
    >>> b = Symbol("b", positive=True)
    >>> z = Symbol("z")

    >>> X = Laplace("x", mu, b)

    >>> density(X)(z)
    exp(-Abs(mu - z)/b)/(2*b)

    >>> cdf(X)(z)
    Piecewise((exp((-mu + z)/b)/2, mu > z), (1 - exp((mu - z)/b)/2, True))

    >>> L = Laplace('L', [1, 2], [[1, 0], [0, 1]])
    >>> pprint(density(L)(1, 2), use_unicode=False)
     5        /     ____\
    e *besselk\0, \/ 35 /
    ---------------------
              pi

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Laplace_distribution
    .. [2] https://mathworld.wolfram.com/LaplaceDistribution.html

    """

    if isinstance(mu, (list, MatrixBase)) and\
        isinstance(b, (list, MatrixBase)):
        from sympy.stats.joint_rv_types import MultivariateLaplace
        return MultivariateLaplace(name, mu, b)

    return rv(name, LaplaceDistribution, (mu, b))

#-------------------------------------------------------------------------------
# Levy distribution ---------------------------------------------------------


class LevyDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 'c')

    @property
    def set(self):
        return Interval(self.mu, oo)

    @staticmethod
    def check(mu, c):
        _value_check(c > 0, "c (scale parameter) must be positive")
        _value_check(mu.is_real, "mu (location parameter) must be real")

    def pdf(self, x):
        mu, c = self.mu, self.c
        return sqrt(c/(2*pi))*exp(-c/(2*(x - mu)))/((x - mu)**(S.One + S.Half))

    def _cdf(self, x):
        mu, c = self.mu, self.c
        return erfc(sqrt(c/(2*(x - mu))))

    def _characteristic_function(self, t):
        mu, c = self.mu, self.c
        return exp(I * mu * t - sqrt(-2 * I * c * t))

    def _moment_generating_function(self, t):
        raise NotImplementedError('The moment generating function of Levy distribution does not exist.')

def Levy(name, mu, c):
    r"""
    Create a continuous random variable with a Levy distribution.

    The density of the Levy distribution is given by

    .. math::
        f(x) := \sqrt(\frac{c}{2 \pi}) \frac{\exp -\frac{c}{2 (x - \mu)}}{(x - \mu)^{3/2}}

    Parameters
    ==========

    mu : Real number
        The location parameter.
    c : Real number, `c > 0`
        A scale parameter.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Levy, density, cdf
    >>> from sympy import Symbol

    >>> mu = Symbol("mu", real=True)
    >>> c = Symbol("c", positive=True)
    >>> z = Symbol("z")

    >>> X = Levy("x", mu, c)

    >>> density(X)(z)
    sqrt(2)*sqrt(c)*exp(-c/(-2*mu + 2*z))/(2*sqrt(pi)*(-mu + z)**(3/2))

    >>> cdf(X)(z)
    erfc(sqrt(c)*sqrt(1/(-2*mu + 2*z)))

    References
    ==========
    .. [1] https://en.wikipedia.org/wiki/L%C3%A9vy_distribution
    .. [2] https://mathworld.wolfram.com/LevyDistribution.html
    """

    return rv(name, LevyDistribution, (mu, c))

#-------------------------------------------------------------------------------
# Log-Cauchy distribution --------------------------------------------------------


class LogCauchyDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 'sigma')

    set = Interval.open(0, oo)

    @staticmethod
    def check(mu, sigma):
        _value_check((sigma > 0) != False, "Scale parameter Gamma must be positive.")
        _value_check(mu.is_real != False, "Location parameter must be real.")

    def pdf(self, x):
        mu, sigma = self.mu, self.sigma
        return 1/(x*pi)*(sigma/((log(x) - mu)**2 + sigma**2))

    def _cdf(self, x):
        mu, sigma = self.mu, self.sigma
        return (1/pi)*atan((log(x) - mu)/sigma) + S.Half

    def _characteristic_function(self, t):
        raise NotImplementedError("The characteristic function for the "
                                  "Log-Cauchy distribution does not exist.")

    def _moment_generating_function(self, t):
        raise NotImplementedError("The moment generating function for the "
                                  "Log-Cauchy distribution does not exist.")

def LogCauchy(name, mu, sigma):
    r"""
    Create a continuous random variable with a Log-Cauchy distribution.
    The density of the Log-Cauchy distribution is given by

    .. math::
        f(x) := \frac{1}{\pi x} \frac{\sigma}{(log(x)-\mu^2) + \sigma^2}

    Parameters
    ==========

    mu : Real number, the location

    sigma : Real number, `\sigma > 0`, a scale

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import LogCauchy, density, cdf
    >>> from sympy import Symbol, S

    >>> mu = 2
    >>> sigma = S.One / 5
    >>> z = Symbol("z")

    >>> X = LogCauchy("x", mu, sigma)

    >>> density(X)(z)
    1/(5*pi*z*((log(z) - 2)**2 + 1/25))

    >>> cdf(X)(z)
    atan(5*log(z) - 10)/pi + 1/2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Log-Cauchy_distribution
    """

    return rv(name, LogCauchyDistribution, (mu, sigma))


#-------------------------------------------------------------------------------
# Logistic distribution --------------------------------------------------------


class LogisticDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 's')

    set = Interval(-oo, oo)

    @staticmethod
    def check(mu, s):
        _value_check(s > 0, "Scale parameter s must be positive.")

    def pdf(self, x):
        mu, s = self.mu, self.s
        return exp(-(x - mu)/s)/(s*(1 + exp(-(x - mu)/s))**2)

    def _cdf(self, x):
        mu, s = self.mu, self.s
        return S.One/(1 + exp(-(x - mu)/s))

    def _characteristic_function(self, t):
        return Piecewise((exp(I*t*self.mu) * pi*self.s*t / sinh(pi*self.s*t), Ne(t, 0)), (S.One, True))

    def _moment_generating_function(self, t):
        return exp(self.mu*t) * beta_fn(1 - self.s*t, 1 + self.s*t)

    def _quantile(self, p):
        return self.mu - self.s*log(-S.One + S.One/p)

def Logistic(name, mu, s):
    r"""
    Create a continuous random variable with a logistic distribution.

    Explanation
    ===========

    The density of the logistic distribution is given by

    .. math::
        f(x) := \frac{e^{-(x-\mu)/s}} {s\left(1+e^{-(x-\mu)/s}\right)^2}

    Parameters
    ==========

    mu : Real number, the location (mean)
    s : Real number, `s > 0`, a scale

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Logistic, density, cdf
    >>> from sympy import Symbol

    >>> mu = Symbol("mu", real=True)
    >>> s = Symbol("s", positive=True)
    >>> z = Symbol("z")

    >>> X = Logistic("x", mu, s)

    >>> density(X)(z)
    exp((mu - z)/s)/(s*(exp((mu - z)/s) + 1)**2)

    >>> cdf(X)(z)
    1/(exp((mu - z)/s) + 1)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Logistic_distribution
    .. [2] https://mathworld.wolfram.com/LogisticDistribution.html

    """

    return rv(name, LogisticDistribution, (mu, s))

#-------------------------------------------------------------------------------
# Log-logistic distribution --------------------------------------------------------


class LogLogisticDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'beta')

    set = Interval(0, oo)

    @staticmethod
    def check(alpha, beta):
        _value_check(alpha > 0, "Scale parameter Alpha must be positive.")
        _value_check(beta > 0, "Shape parameter Beta must be positive.")

    def pdf(self, x):
        a, b = self.alpha, self.beta
        return ((b/a)*(x/a)**(b - 1))/(1 + (x/a)**b)**2

    def _cdf(self, x):
        a, b = self.alpha, self.beta
        return 1/(1 + (x/a)**(-b))

    def _quantile(self, p):
        a, b = self.alpha, self.beta
        return a*((p/(1 - p))**(1/b))

    def expectation(self, expr, var, **kwargs):
        a, b = self.args
        return Piecewise((S.NaN, b <= 1), (pi*a/(b*sin(pi/b)), True))

def LogLogistic(name, alpha, beta):
    r"""
    Create a continuous random variable with a log-logistic distribution.
    The distribution is unimodal when ``beta > 1``.

    Explanation
    ===========

    The density of the log-logistic distribution is given by

    .. math::
        f(x) := \frac{(\frac{\beta}{\alpha})(\frac{x}{\alpha})^{\beta - 1}}
                {(1 + (\frac{x}{\alpha})^{\beta})^2}

    Parameters
    ==========

    alpha : Real number, `\alpha > 0`, scale parameter and median of distribution
    beta : Real number, `\beta > 0`, a shape parameter

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import LogLogistic, density, cdf, quantile
    >>> from sympy import Symbol, pprint

    >>> alpha = Symbol("alpha", positive=True)
    >>> beta = Symbol("beta", positive=True)
    >>> p = Symbol("p")
    >>> z = Symbol("z", positive=True)

    >>> X = LogLogistic("x", alpha, beta)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
                  beta - 1
           /  z  \
      beta*|-----|
           \alpha/
    ------------------------
                           2
          /       beta    \
          |/  z  \        |
    alpha*||-----|     + 1|
          \\alpha/        /

    >>> cdf(X)(z)
    1/(1 + (z/alpha)**(-beta))

    >>> quantile(X)(p)
    alpha*(p/(1 - p))**(1/beta)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Log-logistic_distribution

    """

    return rv(name, LogLogisticDistribution, (alpha, beta))

#-------------------------------------------------------------------------------
#Logit-Normal distribution------------------------------------------------------

class LogitNormalDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 's')
    set = Interval.open(0, 1)

    @staticmethod
    def check(mu, s):
        _value_check((s ** 2).is_real is not False and s ** 2 > 0, "Squared scale parameter s must be positive.")
        _value_check(mu.is_real is not False, "Location parameter must be real")

    def _logit(self, x):
        return log(x / (1 - x))

    def pdf(self, x):
        mu, s = self.mu, self.s
        return exp(-(self._logit(x) - mu)**2/(2*s**2))*(S.One/sqrt(2*pi*(s**2)))*(1/(x*(1 - x)))

    def _cdf(self, x):
        mu, s = self.mu, self.s
        return (S.One/2)*(1 + erf((self._logit(x) - mu)/(sqrt(2*s**2))))


def LogitNormal(name, mu, s):
    r"""
    Create a continuous random variable with a Logit-Normal distribution.

    The density of the logistic distribution is given by

    .. math::
        f(x) := \frac{1}{s \sqrt{2 \pi}} \frac{1}{x(1 - x)} e^{- \frac{(logit(x)  - \mu)^2}{s^2}}
        where logit(x) = \log(\frac{x}{1 - x})
    Parameters
    ==========

    mu : Real number, the location (mean)
    s : Real number, `s > 0`, a scale

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import LogitNormal, density, cdf
    >>> from sympy import Symbol,pprint

    >>> mu = Symbol("mu", real=True)
    >>> s = Symbol("s", positive=True)
    >>> z = Symbol("z")
    >>> X = LogitNormal("x",mu,s)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
                              2
            /         /  z  \\
           -|-mu + log|-----||
            \         \1 - z//
           ---------------------
                       2
      ___           2*s
    \/ 2 *e
    ----------------------------
            ____
        2*\/ pi *s*z*(1 - z)

    >>> density(X)(z)
    sqrt(2)*exp(-(-mu + log(z/(1 - z)))**2/(2*s**2))/(2*sqrt(pi)*s*z*(1 - z))

    >>> cdf(X)(z)
    erf(sqrt(2)*(-mu + log(z/(1 - z)))/(2*s))/2 + 1/2


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Logit-normal_distribution

    """

    return rv(name, LogitNormalDistribution, (mu, s))

#-------------------------------------------------------------------------------
# Log Normal distribution ------------------------------------------------------


class LogNormalDistribution(SingleContinuousDistribution):
    _argnames = ('mean', 'std')

    set = Interval(0, oo)

    @staticmethod
    def check(mean, std):
        _value_check(std > 0, "Parameter std must be positive.")

    def pdf(self, x):
        mean, std = self.mean, self.std
        return exp(-(log(x) - mean)**2 / (2*std**2)) / (x*sqrt(2*pi)*std)

    def _cdf(self, x):
        mean, std = self.mean, self.std
        return Piecewise(
                (S.Half + S.Half*erf((log(x) - mean)/sqrt(2)/std), x > 0),
                (S.Zero, True)
        )

    def _moment_generating_function(self, t):
        raise NotImplementedError('Moment generating function of the log-normal distribution is not defined.')


def LogNormal(name, mean, std):
    r"""
    Create a continuous random variable with a log-normal distribution.

    Explanation
    ===========

    The density of the log-normal distribution is given by

    .. math::
        f(x) := \frac{1}{x\sqrt{2\pi\sigma^2}}
                e^{-\frac{\left(\ln x-\mu\right)^2}{2\sigma^2}}

    with :math:`x \geq 0`.

    Parameters
    ==========

    mu : Real number
        The log-scale.
    sigma : Real number
        A shape. ($\sigma^2 > 0$)

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import LogNormal, density
    >>> from sympy import Symbol, pprint

    >>> mu = Symbol("mu", real=True)
    >>> sigma = Symbol("sigma", positive=True)
    >>> z = Symbol("z")

    >>> X = LogNormal("x", mu, sigma)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
                          2
           -(-mu + log(z))
           -----------------
                      2
      ___      2*sigma
    \/ 2 *e
    ------------------------
            ____
        2*\/ pi *sigma*z


    >>> X = LogNormal('x', 0, 1) # Mean 0, standard deviation 1

    >>> density(X)(z)
    sqrt(2)*exp(-log(z)**2/2)/(2*sqrt(pi)*z)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lognormal
    .. [2] https://mathworld.wolfram.com/LogNormalDistribution.html

    """

    return rv(name, LogNormalDistribution, (mean, std))

#-------------------------------------------------------------------------------
# Lomax Distribution -----------------------------------------------------------

class LomaxDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'lamda',)
    set = Interval(0, oo)

    @staticmethod
    def check(alpha, lamda):
        _value_check(alpha.is_real, "Shape parameter should be real.")
        _value_check(lamda.is_real, "Scale parameter should be real.")
        _value_check(alpha.is_positive, "Shape parameter should be positive.")
        _value_check(lamda.is_positive, "Scale parameter should be positive.")

    def pdf(self, x):
        lamba, alpha = self.lamda, self.alpha
        return (alpha/lamba) * (S.One + x/lamba)**(-alpha-1)

def Lomax(name, alpha, lamda):
    r"""
    Create a continuous random variable with a Lomax distribution.

    Explanation
    ===========

    The density of the Lomax distribution is given by

    .. math::
        f(x) := \frac{\alpha}{\lambda}\left[1+\frac{x}{\lambda}\right]^{-(\alpha+1)}

    Parameters
    ==========

    alpha : Real Number, `\alpha > 0`
        Shape parameter
    lamda : Real Number, `\lambda > 0`
        Scale parameter

    Examples
    ========

    >>> from sympy.stats import Lomax, density, cdf, E
    >>> from sympy import symbols
    >>> a, l = symbols('a, l', positive=True)
    >>> X = Lomax('X', a, l)
    >>> x = symbols('x')
    >>> density(X)(x)
    a*(1 + x/l)**(-a - 1)/l
    >>> cdf(X)(x)
    Piecewise((1 - 1/(1 + x/l)**a, x >= 0), (0, True))
    >>> a = 2
    >>> X = Lomax('X', a, l)
    >>> E(X)
    l

    Returns
    =======

    RandomSymbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lomax_distribution

    """
    return rv(name, LomaxDistribution, (alpha, lamda))

#-------------------------------------------------------------------------------
# Maxwell distribution ---------------------------------------------------------


class MaxwellDistribution(SingleContinuousDistribution):
    _argnames = ('a',)

    set = Interval(0, oo)

    @staticmethod
    def check(a):
        _value_check(a > 0, "Parameter a must be positive.")

    def pdf(self, x):
        a = self.a
        return sqrt(2/pi)*x**2*exp(-x**2/(2*a**2))/a**3

    def _cdf(self, x):
        a = self.a
        return erf(sqrt(2)*x/(2*a)) - sqrt(2)*x*exp(-x**2/(2*a**2))/(sqrt(pi)*a)

def Maxwell(name, a):
    r"""
    Create a continuous random variable with a Maxwell distribution.

    Explanation
    ===========

    The density of the Maxwell distribution is given by

    .. math::
        f(x) := \sqrt{\frac{2}{\pi}} \frac{x^2 e^{-x^2/(2a^2)}}{a^3}

    with :math:`x \geq 0`.

    .. TODO - what does the parameter mean?

    Parameters
    ==========

    a : Real number, `a > 0`

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Maxwell, density, E, variance
    >>> from sympy import Symbol, simplify

    >>> a = Symbol("a", positive=True)
    >>> z = Symbol("z")

    >>> X = Maxwell("x", a)

    >>> density(X)(z)
    sqrt(2)*z**2*exp(-z**2/(2*a**2))/(sqrt(pi)*a**3)

    >>> E(X)
    2*sqrt(2)*a/sqrt(pi)

    >>> simplify(variance(X))
    a**2*(-8 + 3*pi)/pi

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Maxwell_distribution
    .. [2] https://mathworld.wolfram.com/MaxwellDistribution.html

    """

    return rv(name, MaxwellDistribution, (a, ))

#-------------------------------------------------------------------------------
# Moyal Distribution -----------------------------------------------------------
class MoyalDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 'sigma')

    @staticmethod
    def check(mu, sigma):
        _value_check(mu.is_real, "Location parameter must be real.")
        _value_check(sigma.is_real and sigma > 0, "Scale parameter must be real\
        and positive.")

    def pdf(self, x):
        mu, sigma = self.mu, self.sigma
        num = exp(-(exp(-(x - mu)/sigma) + (x - mu)/(sigma))/2)
        den = (sqrt(2*pi) * sigma)
        return num/den

    def _characteristic_function(self, t):
        mu, sigma = self.mu, self.sigma
        term1 = exp(I*t*mu)
        term2 = (2**(-I*sigma*t) * gamma(Rational(1, 2) - I*t*sigma))
        return (term1 * term2)/sqrt(pi)

    def _moment_generating_function(self, t):
        mu, sigma = self.mu, self.sigma
        term1 = exp(t*mu)
        term2 = (2**(-1*sigma*t) * gamma(Rational(1, 2) - t*sigma))
        return (term1 * term2)/sqrt(pi)

def Moyal(name, mu, sigma):
    r"""
    Create a continuous random variable with a Moyal distribution.

    Explanation
    ===========

    The density of the Moyal distribution is given by

    .. math::
        f(x) := \frac{\exp-\frac{1}{2}\exp-\frac{x-\mu}{\sigma}-\frac{x-\mu}{2\sigma}}{\sqrt{2\pi}\sigma}

    with :math:`x \in \mathbb{R}`.

    Parameters
    ==========

    mu : Real number
        Location parameter
    sigma : Real positive number
        Scale parameter

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Moyal, density, cdf
    >>> from sympy import Symbol, simplify
    >>> mu = Symbol("mu", real=True)
    >>> sigma = Symbol("sigma", positive=True, real=True)
    >>> z = Symbol("z")
    >>> X = Moyal("x", mu, sigma)
    >>> density(X)(z)
    sqrt(2)*exp(-exp((mu - z)/sigma)/2 - (-mu + z)/(2*sigma))/(2*sqrt(pi)*sigma)
    >>> simplify(cdf(X)(z))
    1 - erf(sqrt(2)*exp((mu - z)/(2*sigma))/2)

    References
    ==========

    .. [1] https://reference.wolfram.com/language/ref/MoyalDistribution.html
    .. [2] https://www.stat.rice.edu/~dobelman/textfiles/DistributionsHandbook.pdf

    """

    return rv(name, MoyalDistribution, (mu, sigma))

#-------------------------------------------------------------------------------
# Nakagami distribution --------------------------------------------------------


class NakagamiDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 'omega')

    set = Interval(0, oo)

    @staticmethod
    def check(mu, omega):
        _value_check(mu >= S.Half, "Shape parameter mu must be greater than equal to 1/2.")
        _value_check(omega > 0, "Spread parameter omega must be positive.")

    def pdf(self, x):
        mu, omega = self.mu, self.omega
        return 2*mu**mu/(gamma(mu)*omega**mu)*x**(2*mu - 1)*exp(-mu/omega*x**2)

    def _cdf(self, x):
        mu, omega = self.mu, self.omega
        return Piecewise(
                    (lowergamma(mu, (mu/omega)*x**2)/gamma(mu), x > 0),
                    (S.Zero, True))

def Nakagami(name, mu, omega):
    r"""
    Create a continuous random variable with a Nakagami distribution.

    Explanation
    ===========

    The density of the Nakagami distribution is given by

    .. math::
        f(x) := \frac{2\mu^\mu}{\Gamma(\mu)\omega^\mu} x^{2\mu-1}
                \exp\left(-\frac{\mu}{\omega}x^2 \right)

    with :math:`x > 0`.

    Parameters
    ==========

    mu : Real number, `\mu \geq \frac{1}{2}`, a shape
    omega : Real number, `\omega > 0`, the spread

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Nakagami, density, E, variance, cdf
    >>> from sympy import Symbol, simplify, pprint

    >>> mu = Symbol("mu", positive=True)
    >>> omega = Symbol("omega", positive=True)
    >>> z = Symbol("z")

    >>> X = Nakagami("x", mu, omega)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
                                    2
                               -mu*z
                               -------
        mu      -mu  2*mu - 1  omega
    2*mu  *omega   *z        *e
    ----------------------------------
                Gamma(mu)

    >>> simplify(E(X))
    sqrt(mu)*sqrt(omega)*gamma(mu + 1/2)/gamma(mu + 1)

    >>> V = simplify(variance(X))
    >>> pprint(V, use_unicode=False)
                        2
             omega*Gamma (mu + 1/2)
    omega - -----------------------
            Gamma(mu)*Gamma(mu + 1)

    >>> cdf(X)(z)
    Piecewise((lowergamma(mu, mu*z**2/omega)/gamma(mu), z > 0),
            (0, True))


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Nakagami_distribution

    """

    return rv(name, NakagamiDistribution, (mu, omega))

#-------------------------------------------------------------------------------
# Normal distribution ----------------------------------------------------------


class NormalDistribution(SingleContinuousDistribution):
    _argnames = ('mean', 'std')

    @staticmethod
    def check(mean, std):
        _value_check(std > 0, "Standard deviation must be positive")

    def pdf(self, x):
        return exp(-(x - self.mean)**2 / (2*self.std**2)) / (sqrt(2*pi)*self.std)

    def _cdf(self, x):
        mean, std = self.mean, self.std
        return erf(sqrt(2)*(-mean + x)/(2*std))/2 + S.Half

    def _characteristic_function(self, t):
        mean, std = self.mean, self.std
        return exp(I*mean*t - std**2*t**2/2)

    def _moment_generating_function(self, t):
        mean, std = self.mean, self.std
        return exp(mean*t + std**2*t**2/2)

    def _quantile(self, p):
        mean, std = self.mean, self.std
        return mean + std*sqrt(2)*erfinv(2*p - 1)


def Normal(name, mean, std):
    r"""
    Create a continuous random variable with a Normal distribution.

    Explanation
    ===========

    The density of the Normal distribution is given by

    .. math::
        f(x) := \frac{1}{\sigma\sqrt{2\pi}} e^{ -\frac{(x-\mu)^2}{2\sigma^2} }

    Parameters
    ==========

    mu : Real number or a list representing the mean or the mean vector
    sigma : Real number or a positive definite square matrix,
         :math:`\sigma^2 > 0`, the variance

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Normal, density, E, std, cdf, skewness, quantile, marginal_distribution
    >>> from sympy import Symbol, simplify, pprint

    >>> mu = Symbol("mu")
    >>> sigma = Symbol("sigma", positive=True)
    >>> z = Symbol("z")
    >>> y = Symbol("y")
    >>> p = Symbol("p")
    >>> X = Normal("x", mu, sigma)

    >>> density(X)(z)
    sqrt(2)*exp(-(-mu + z)**2/(2*sigma**2))/(2*sqrt(pi)*sigma)

    >>> C = simplify(cdf(X))(z) # it needs a little more help...
    >>> pprint(C, use_unicode=False)
       /  ___          \
       |\/ 2 *(-mu + z)|
    erf|---------------|
       \    2*sigma    /   1
    -------------------- + -
             2             2

    >>> quantile(X)(p)
    mu + sqrt(2)*sigma*erfinv(2*p - 1)

    >>> simplify(skewness(X))
    0

    >>> X = Normal("x", 0, 1) # Mean 0, standard deviation 1
    >>> density(X)(z)
    sqrt(2)*exp(-z**2/2)/(2*sqrt(pi))

    >>> E(2*X + 1)
    1

    >>> simplify(std(2*X + 1))
    2

    >>> m = Normal('X', [1, 2], [[2, 1], [1, 2]])
    >>> pprint(density(m)(y, z), use_unicode=False)
              2          2
             y    y*z   z
           - -- + --- - -- + z - 1
      ___    3     3    3
    \/ 3 *e
    ------------------------------
                 6*pi

    >>> marginal_distribution(m, m[0])(1)
     1/(2*sqrt(pi))


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Normal_distribution
    .. [2] https://mathworld.wolfram.com/NormalDistributionFunction.html

    """

    if isinstance(mean, list) or getattr(mean, 'is_Matrix', False) and\
        isinstance(std, list) or getattr(std, 'is_Matrix', False):
        from sympy.stats.joint_rv_types import MultivariateNormal
        return MultivariateNormal(name, mean, std)
    return rv(name, NormalDistribution, (mean, std))


#-------------------------------------------------------------------------------
# Inverse Gaussian distribution ----------------------------------------------------------


class GaussianInverseDistribution(SingleContinuousDistribution):
    _argnames = ('mean', 'shape')

    @property
    def set(self):
        return Interval(0, oo)

    @staticmethod
    def check(mean, shape):
        _value_check(shape > 0, "Shape parameter must be positive")
        _value_check(mean > 0, "Mean must be positive")

    def pdf(self, x):
        mu, s = self.mean, self.shape
        return exp(-s*(x - mu)**2 / (2*x*mu**2)) * sqrt(s/(2*pi*x**3))

    def _cdf(self, x):
        from sympy.stats import cdf
        mu, s = self.mean, self.shape
        stdNormalcdf = cdf(Normal('x', 0, 1))

        first_term = stdNormalcdf(sqrt(s/x) * ((x/mu) - S.One))
        second_term = exp(2*s/mu) * stdNormalcdf(-sqrt(s/x)*(x/mu + S.One))

        return  first_term + second_term

    def _characteristic_function(self, t):
        mu, s = self.mean, self.shape
        return exp((s/mu)*(1 - sqrt(1 - (2*mu**2*I*t)/s)))

    def _moment_generating_function(self, t):
        mu, s = self.mean, self.shape
        return exp((s/mu)*(1 - sqrt(1 - (2*mu**2*t)/s)))


def GaussianInverse(name, mean, shape):
    r"""
    Create a continuous random variable with an Inverse Gaussian distribution.
    Inverse Gaussian distribution is also known as Wald distribution.

    Explanation
    ===========

    The density of the Inverse Gaussian distribution is given by

    .. math::
        f(x) := \sqrt{\frac{\lambda}{2\pi x^3}} e^{-\frac{\lambda(x-\mu)^2}{2x\mu^2}}

    Parameters
    ==========

    mu :
        Positive number representing the mean.
    lambda :
        Positive number representing the shape parameter.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import GaussianInverse, density, E, std, skewness
    >>> from sympy import Symbol, pprint

    >>> mu = Symbol("mu", positive=True)
    >>> lamda = Symbol("lambda", positive=True)
    >>> z = Symbol("z", positive=True)
    >>> X = GaussianInverse("x", mu, lamda)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
                                       2
                      -lambda*(-mu + z)
                      -------------------
                                2
      ___   ________        2*mu *z
    \/ 2 *\/ lambda *e
    -------------------------------------
                    ____  3/2
                2*\/ pi *z

    >>> E(X)
    mu

    >>> std(X).expand()
    mu**(3/2)/sqrt(lambda)

    >>> skewness(X).expand()
    3*sqrt(mu)/sqrt(lambda)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution
    .. [2] https://mathworld.wolfram.com/InverseGaussianDistribution.html

    """

    return rv(name, GaussianInverseDistribution, (mean, shape))

Wald = GaussianInverse

#-------------------------------------------------------------------------------
# Pareto distribution ----------------------------------------------------------


class ParetoDistribution(SingleContinuousDistribution):
    _argnames = ('xm', 'alpha')

    @property
    def set(self):
        return Interval(self.xm, oo)

    @staticmethod
    def check(xm, alpha):
        _value_check(xm > 0, "Xm must be positive")
        _value_check(alpha > 0, "Alpha must be positive")

    def pdf(self, x):
        xm, alpha = self.xm, self.alpha
        return alpha * xm**alpha / x**(alpha + 1)

    def _cdf(self, x):
        xm, alpha = self.xm, self.alpha
        return Piecewise(
                (S.One - xm**alpha/x**alpha, x>=xm),
                (0, True),
        )

    def _moment_generating_function(self, t):
        xm, alpha = self.xm, self.alpha
        return alpha * (-xm*t)**alpha * uppergamma(-alpha, -xm*t)

    def _characteristic_function(self, t):
        xm, alpha = self.xm, self.alpha
        return alpha * (-I * xm * t) ** alpha * uppergamma(-alpha, -I * xm * t)


def Pareto(name, xm, alpha):
    r"""
    Create a continuous random variable with the Pareto distribution.

    Explanation
    ===========

    The density of the Pareto distribution is given by

    .. math::
        f(x) := \frac{\alpha\,x_m^\alpha}{x^{\alpha+1}}

    with :math:`x \in [x_m,\infty]`.

    Parameters
    ==========

    xm : Real number, `x_m > 0`, a scale
    alpha : Real number, `\alpha > 0`, a shape

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Pareto, density
    >>> from sympy import Symbol

    >>> xm = Symbol("xm", positive=True)
    >>> beta = Symbol("beta", positive=True)
    >>> z = Symbol("z")

    >>> X = Pareto("x", xm, beta)

    >>> density(X)(z)
    beta*xm**beta*z**(-beta - 1)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Pareto_distribution
    .. [2] https://mathworld.wolfram.com/ParetoDistribution.html

    """

    return rv(name, ParetoDistribution, (xm, alpha))

#-------------------------------------------------------------------------------
# PowerFunction distribution ---------------------------------------------------


class PowerFunctionDistribution(SingleContinuousDistribution):
    _argnames=('alpha','a','b')

    @property
    def set(self):
        return Interval(self.a, self.b)

    @staticmethod
    def check(alpha, a, b):
        _value_check(a.is_real, "Continuous Boundary parameter should be real.")
        _value_check(b.is_real, "Continuous Boundary parameter should be real.")
        _value_check(a < b, " 'a' the left Boundary must be smaller than 'b' the right Boundary." )
        _value_check(alpha.is_positive, "Continuous Shape parameter should be positive.")

    def pdf(self, x):
        alpha, a, b = self.alpha, self.a, self.b
        num = alpha*(x - a)**(alpha - 1)
        den = (b - a)**alpha
        return num/den

def PowerFunction(name, alpha, a, b):
    r"""
    Creates a continuous random variable with a Power Function Distribution.

    Explanation
    ===========

    The density of PowerFunction distribution is given by

    .. math::
        f(x) := \frac{{\alpha}(x - a)^{\alpha - 1}}{(b - a)^{\alpha}}

    with :math:`x \in [a,b]`.

    Parameters
    ==========

    alpha : Positive number, `0 < \alpha`, the shape parameter
    a : Real number, :math:`-\infty < a`, the left boundary
    b : Real number, :math:`a < b < \infty`, the right boundary

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import PowerFunction, density, cdf, E, variance
    >>> from sympy import Symbol
    >>> alpha = Symbol("alpha", positive=True)
    >>> a = Symbol("a", real=True)
    >>> b = Symbol("b", real=True)
    >>> z = Symbol("z")

    >>> X = PowerFunction("X", 2, a, b)

    >>> density(X)(z)
    (-2*a + 2*z)/(-a + b)**2

    >>> cdf(X)(z)
    Piecewise((a**2/(a**2 - 2*a*b + b**2) - 2*a*z/(a**2 - 2*a*b + b**2) +
    z**2/(a**2 - 2*a*b + b**2), a <= z), (0, True))

    >>> alpha = 2
    >>> a = 0
    >>> b = 1
    >>> Y = PowerFunction("Y", alpha, a, b)

    >>> E(Y)
    2/3

    >>> variance(Y)
    1/18

    References
    ==========

    .. [1] https://web.archive.org/web/20200204081320/http://www.mathwave.com/help/easyfit/html/analyses/distributions/power_func.html

    """
    return rv(name, PowerFunctionDistribution, (alpha, a, b))

#-------------------------------------------------------------------------------
# QuadraticU distribution ------------------------------------------------------


class QuadraticUDistribution(SingleContinuousDistribution):
    _argnames = ('a', 'b')

    @property
    def set(self):
        return Interval(self.a, self.b)

    @staticmethod
    def check(a, b):
        _value_check(b > a, "Parameter b must be in range (%s, oo)."%(a))

    def pdf(self, x):
        a, b = self.a, self.b
        alpha = 12 / (b-a)**3
        beta = (a+b) / 2
        return Piecewise(
                  (alpha * (x-beta)**2, And(a<=x, x<=b)),
                  (S.Zero, True))

    def _moment_generating_function(self, t):
        a, b = self.a, self.b
        return -3 * (exp(a*t) * (4  + (a**2 + 2*a*(-2 + b) + b**2) * t) \
        - exp(b*t) * (4 + (-4*b + (a + b)**2) * t)) / ((a-b)**3 * t**2)

    def _characteristic_function(self, t):
        a, b = self.a, self.b
        return -3*I*(exp(I*a*t*exp(I*b*t)) * (4*I - (-4*b + (a+b)**2)*t)) \
                / ((a-b)**3 * t**2)


def QuadraticU(name, a, b):
    r"""
    Create a Continuous Random Variable with a U-quadratic distribution.

    Explanation
    ===========

    The density of the U-quadratic distribution is given by

    .. math::
        f(x) := \alpha (x-\beta)^2

    with :math:`x \in [a,b]`.

    Parameters
    ==========

    a : Real number
    b : Real number, :math:`a < b`

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import QuadraticU, density
    >>> from sympy import Symbol, pprint

    >>> a = Symbol("a", real=True)
    >>> b = Symbol("b", real=True)
    >>> z = Symbol("z")

    >>> X = QuadraticU("x", a, b)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
    /                2
    |   /  a   b    \
    |12*|- - - - + z|
    |   \  2   2    /
    <-----------------  for And(b >= z, a <= z)
    |            3
    |    (-a + b)
    |
    \        0                 otherwise

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/U-quadratic_distribution

    """

    return rv(name, QuadraticUDistribution, (a, b))

#-------------------------------------------------------------------------------
# RaisedCosine distribution ----------------------------------------------------


class RaisedCosineDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 's')

    @property
    def set(self):
        return Interval(self.mu - self.s, self.mu + self.s)

    @staticmethod
    def check(mu, s):
        _value_check(s > 0, "s must be positive")

    def pdf(self, x):
        mu, s = self.mu, self.s
        return Piecewise(
                ((1+cos(pi*(x-mu)/s)) / (2*s), And(mu-s<=x, x<=mu+s)),
                (S.Zero, True))

    def _characteristic_function(self, t):
        mu, s = self.mu, self.s
        return Piecewise((exp(-I*pi*mu/s)/2, Eq(t, -pi/s)),
                         (exp(I*pi*mu/s)/2, Eq(t, pi/s)),
                         (pi**2*sin(s*t)*exp(I*mu*t) / (s*t*(pi**2 - s**2*t**2)), True))

    def _moment_generating_function(self, t):
        mu, s = self.mu, self.s
        return pi**2 * sinh(s*t) * exp(mu*t) /  (s*t*(pi**2 + s**2*t**2))

def RaisedCosine(name, mu, s):
    r"""
    Create a Continuous Random Variable with a raised cosine distribution.

    Explanation
    ===========

    The density of the raised cosine distribution is given by

    .. math::
        f(x) := \frac{1}{2s}\left(1+\cos\left(\frac{x-\mu}{s}\pi\right)\right)

    with :math:`x \in [\mu-s,\mu+s]`.

    Parameters
    ==========

    mu : Real number
    s : Real number, `s > 0`

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import RaisedCosine, density
    >>> from sympy import Symbol, pprint

    >>> mu = Symbol("mu", real=True)
    >>> s = Symbol("s", positive=True)
    >>> z = Symbol("z")

    >>> X = RaisedCosine("x", mu, s)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
    /   /pi*(-mu + z)\
    |cos|------------| + 1
    |   \     s      /
    <---------------------  for And(z >= mu - s, z <= mu + s)
    |         2*s
    |
    \          0                        otherwise

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Raised_cosine_distribution

    """

    return rv(name, RaisedCosineDistribution, (mu, s))

#-------------------------------------------------------------------------------
# Rayleigh distribution --------------------------------------------------------


class RayleighDistribution(SingleContinuousDistribution):
    _argnames = ('sigma',)

    set = Interval(0, oo)

    @staticmethod
    def check(sigma):
        _value_check(sigma > 0, "Scale parameter sigma must be positive.")

    def pdf(self, x):
        sigma = self.sigma
        return x/sigma**2*exp(-x**2/(2*sigma**2))

    def _cdf(self, x):
        sigma = self.sigma
        return 1 - exp(-(x**2/(2*sigma**2)))

    def _characteristic_function(self, t):
        sigma = self.sigma
        return 1 - sigma*t*exp(-sigma**2*t**2/2) * sqrt(pi/2) * (erfi(sigma*t/sqrt(2)) - I)

    def _moment_generating_function(self, t):
        sigma = self.sigma
        return 1 + sigma*t*exp(sigma**2*t**2/2) * sqrt(pi/2) * (erf(sigma*t/sqrt(2)) + 1)


def Rayleigh(name, sigma):
    r"""
    Create a continuous random variable with a Rayleigh distribution.

    Explanation
    ===========

    The density of the Rayleigh distribution is given by

    .. math ::
        f(x) := \frac{x}{\sigma^2} e^{-x^2/2\sigma^2}

    with :math:`x > 0`.

    Parameters
    ==========

    sigma : Real number, `\sigma > 0`

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Rayleigh, density, E, variance
    >>> from sympy import Symbol

    >>> sigma = Symbol("sigma", positive=True)
    >>> z = Symbol("z")

    >>> X = Rayleigh("x", sigma)

    >>> density(X)(z)
    z*exp(-z**2/(2*sigma**2))/sigma**2

    >>> E(X)
    sqrt(2)*sqrt(pi)*sigma/2

    >>> variance(X)
    -pi*sigma**2/2 + 2*sigma**2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Rayleigh_distribution
    .. [2] https://mathworld.wolfram.com/RayleighDistribution.html

    """

    return rv(name, RayleighDistribution, (sigma, ))

#-------------------------------------------------------------------------------
# Reciprocal distribution --------------------------------------------------------

class ReciprocalDistribution(SingleContinuousDistribution):
    _argnames = ('a', 'b')

    @property
    def set(self):
        return Interval(self.a, self.b)

    @staticmethod
    def check(a, b):
        _value_check(a > 0, "Parameter > 0. a = %s"%a)
        _value_check((a < b),
        "Parameter b must be in range (%s, +oo]. b = %s"%(a, b))

    def pdf(self, x):
        a, b = self.a, self.b
        return 1/(x*(log(b) - log(a)))


def Reciprocal(name, a, b):
    r"""Creates a continuous random variable with a reciprocal distribution.


    Parameters
    ==========

    a : Real number, :math:`0 < a`
    b : Real number, :math:`a < b`

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Reciprocal, density, cdf
    >>> from sympy import symbols
    >>> a, b, x = symbols('a, b, x', positive=True)
    >>> R = Reciprocal('R', a, b)

    >>> density(R)(x)
    1/(x*(-log(a) + log(b)))
    >>> cdf(R)(x)
    Piecewise((log(a)/(log(a) - log(b)) - log(x)/(log(a) - log(b)), a <= x), (0, True))

    Reference
    =========

    .. [1] https://en.wikipedia.org/wiki/Reciprocal_distribution

    """
    return rv(name, ReciprocalDistribution, (a, b))


#-------------------------------------------------------------------------------
# Shifted Gompertz distribution ------------------------------------------------


class ShiftedGompertzDistribution(SingleContinuousDistribution):
    _argnames = ('b', 'eta')

    set = Interval(0, oo)

    @staticmethod
    def check(b, eta):
        _value_check(b > 0, "b must be positive")
        _value_check(eta > 0, "eta must be positive")

    def pdf(self, x):
        b, eta = self.b, self.eta
        return b*exp(-b*x)*exp(-eta*exp(-b*x))*(1+eta*(1-exp(-b*x)))

def ShiftedGompertz(name, b, eta):
    r"""
    Create a continuous random variable with a Shifted Gompertz distribution.

    Explanation
    ===========

    The density of the Shifted Gompertz distribution is given by

    .. math::
        f(x) := b e^{-b x} e^{-\eta \exp(-b x)} \left[1 + \eta(1 - e^(-bx)) \right]

    with :math:`x \in [0, \infty)`.

    Parameters
    ==========

    b : Real number, `b > 0`, a scale
    eta : Real number, `\eta > 0`, a shape

    Returns
    =======

    RandomSymbol

    Examples
    ========
    >>> from sympy.stats import ShiftedGompertz, density
    >>> from sympy import Symbol

    >>> b = Symbol("b", positive=True)
    >>> eta = Symbol("eta", positive=True)
    >>> x = Symbol("x")

    >>> X = ShiftedGompertz("x", b, eta)

    >>> density(X)(x)
    b*(eta*(1 - exp(-b*x)) + 1)*exp(-b*x)*exp(-eta*exp(-b*x))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Shifted_Gompertz_distribution

    """
    return rv(name, ShiftedGompertzDistribution, (b, eta))

#-------------------------------------------------------------------------------
# StudentT distribution --------------------------------------------------------


class StudentTDistribution(SingleContinuousDistribution):
    _argnames = ('nu',)

    set = Interval(-oo, oo)

    @staticmethod
    def check(nu):
        _value_check(nu > 0, "Degrees of freedom nu must be positive.")

    def pdf(self, x):
        nu = self.nu
        return 1/(sqrt(nu)*beta_fn(S.Half, nu/2))*(1 + x**2/nu)**(-(nu + 1)/2)

    def _cdf(self, x):
        nu = self.nu
        return S.Half + x*gamma((nu+1)/2)*hyper((S.Half, (nu+1)/2),
                                (Rational(3, 2),), -x**2/nu)/(sqrt(pi*nu)*gamma(nu/2))

    def _moment_generating_function(self, t):
        raise NotImplementedError('The moment generating function for the Student-T distribution is undefined.')


def StudentT(name, nu):
    r"""
    Create a continuous random variable with a student's t distribution.

    Explanation
    ===========

    The density of the student's t distribution is given by

    .. math::
        f(x) := \frac{\Gamma \left(\frac{\nu+1}{2} \right)}
                {\sqrt{\nu\pi}\Gamma \left(\frac{\nu}{2} \right)}
                \left(1+\frac{x^2}{\nu} \right)^{-\frac{\nu+1}{2}}

    Parameters
    ==========

    nu : Real number, `\nu > 0`, the degrees of freedom

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import StudentT, density, cdf
    >>> from sympy import Symbol, pprint

    >>> nu = Symbol("nu", positive=True)
    >>> z = Symbol("z")

    >>> X = StudentT("x", nu)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
               nu   1
             - -- - -
               2    2
     /     2\
     |    z |
     |1 + --|
     \    nu/
    -----------------
      ____  /     nu\
    \/ nu *B|1/2, --|
            \     2 /

    >>> cdf(X)(z)
    1/2 + z*gamma(nu/2 + 1/2)*hyper((1/2, nu/2 + 1/2), (3/2,),
                                -z**2/nu)/(sqrt(pi)*sqrt(nu)*gamma(nu/2))


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Student_t-distribution
    .. [2] https://mathworld.wolfram.com/Studentst-Distribution.html

    """

    return rv(name, StudentTDistribution, (nu, ))

#-------------------------------------------------------------------------------
# Trapezoidal distribution ------------------------------------------------------


class TrapezoidalDistribution(SingleContinuousDistribution):
    _argnames = ('a', 'b', 'c', 'd')

    @property
    def set(self):
        return Interval(self.a, self.d)

    @staticmethod
    def check(a, b, c, d):
        _value_check(a < d, "Lower bound parameter a < %s. a = %s"%(d, a))
        _value_check((a <= b, b < c),
        "Level start parameter b must be in range [%s, %s). b = %s"%(a, c, b))
        _value_check((b < c, c <= d),
        "Level end parameter c must be in range (%s, %s]. c = %s"%(b, d, c))
        _value_check(d >= c, "Upper bound parameter d > %s. d = %s"%(c, d))

    def pdf(self, x):
        a, b, c, d = self.a, self.b, self.c, self.d
        return Piecewise(
            (2*(x-a) / ((b-a)*(d+c-a-b)), And(a <= x, x < b)),
            (2 / (d+c-a-b), And(b <= x, x < c)),
            (2*(d-x) / ((d-c)*(d+c-a-b)), And(c <= x, x <= d)),
            (S.Zero, True))

def Trapezoidal(name, a, b, c, d):
    r"""
    Create a continuous random variable with a trapezoidal distribution.

    Explanation
    ===========

    The density of the trapezoidal distribution is given by

    .. math::
        f(x) := \begin{cases}
                  0 & \mathrm{for\ } x < a, \\
                  \frac{2(x-a)}{(b-a)(d+c-a-b)} & \mathrm{for\ } a \le x < b, \\
                  \frac{2}{d+c-a-b} & \mathrm{for\ } b \le x < c, \\
                  \frac{2(d-x)}{(d-c)(d+c-a-b)} & \mathrm{for\ } c \le x < d, \\
                  0 & \mathrm{for\ } d < x.
                \end{cases}

    Parameters
    ==========

    a : Real number, :math:`a < d`
    b : Real number, :math:`a \le b < c`
    c : Real number, :math:`b < c \le d`
    d : Real number

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Trapezoidal, density
    >>> from sympy import Symbol, pprint

    >>> a = Symbol("a")
    >>> b = Symbol("b")
    >>> c = Symbol("c")
    >>> d = Symbol("d")
    >>> z = Symbol("z")

    >>> X = Trapezoidal("x", a,b,c,d)

    >>> pprint(density(X)(z), use_unicode=False)
    /        -2*a + 2*z
    |-------------------------  for And(a <= z, b > z)
    |(-a + b)*(-a - b + c + d)
    |
    |           2
    |     --------------        for And(b <= z, c > z)
    <     -a - b + c + d
    |
    |        2*d - 2*z
    |-------------------------  for And(d >= z, c <= z)
    |(-c + d)*(-a - b + c + d)
    |
    \            0                     otherwise

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trapezoidal_distribution

    """
    return rv(name, TrapezoidalDistribution, (a, b, c, d))

#-------------------------------------------------------------------------------
# Triangular distribution ------------------------------------------------------


class TriangularDistribution(SingleContinuousDistribution):
    _argnames = ('a', 'b', 'c')

    @property
    def set(self):
        return Interval(self.a, self.b)

    @staticmethod
    def check(a, b, c):
        _value_check(b > a, "Parameter b > %s. b = %s"%(a, b))
        _value_check((a <= c, c <= b),
        "Parameter c must be in range [%s, %s]. c = %s"%(a, b, c))

    def pdf(self, x):
        a, b, c = self.a, self.b, self.c
        return Piecewise(
            (2*(x - a)/((b - a)*(c - a)), And(a <= x, x < c)),
            (2/(b - a), Eq(x, c)),
            (2*(b - x)/((b - a)*(b - c)), And(c < x, x <= b)),
            (S.Zero, True))

    def _characteristic_function(self, t):
        a, b, c = self.a, self.b, self.c
        return -2 *((b-c) * exp(I*a*t) - (b-a) * exp(I*c*t) + (c-a) * exp(I*b*t)) / ((b-a)*(c-a)*(b-c)*t**2)

    def _moment_generating_function(self, t):
        a, b, c = self.a, self.b, self.c
        return 2 * ((b - c) * exp(a * t) - (b - a) * exp(c * t) + (c - a) * exp(b * t)) / (
        (b - a) * (c - a) * (b - c) * t ** 2)


def Triangular(name, a, b, c):
    r"""
    Create a continuous random variable with a triangular distribution.

    Explanation
    ===========

    The density of the triangular distribution is given by

    .. math::
        f(x) := \begin{cases}
                  0 & \mathrm{for\ } x < a, \\
                  \frac{2(x-a)}{(b-a)(c-a)} & \mathrm{for\ } a \le x < c, \\
                  \frac{2}{b-a} & \mathrm{for\ } x = c, \\
                  \frac{2(b-x)}{(b-a)(b-c)} & \mathrm{for\ } c < x \le b, \\
                  0 & \mathrm{for\ } b < x.
                \end{cases}

    Parameters
    ==========

    a : Real number, :math:`a \in \left(-\infty, \infty\right)`
    b : Real number, :math:`a < b`
    c : Real number, :math:`a \leq c \leq b`

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Triangular, density
    >>> from sympy import Symbol, pprint

    >>> a = Symbol("a")
    >>> b = Symbol("b")
    >>> c = Symbol("c")
    >>> z = Symbol("z")

    >>> X = Triangular("x", a,b,c)

    >>> pprint(density(X)(z), use_unicode=False)
    /    -2*a + 2*z
    |-----------------  for And(a <= z, c > z)
    |(-a + b)*(-a + c)
    |
    |       2
    |     ------              for c = z
    <     -a + b
    |
    |   2*b - 2*z
    |----------------   for And(b >= z, c < z)
    |(-a + b)*(b - c)
    |
    \        0                otherwise

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Triangular_distribution
    .. [2] https://mathworld.wolfram.com/TriangularDistribution.html

    """

    return rv(name, TriangularDistribution, (a, b, c))

#-------------------------------------------------------------------------------
# Uniform distribution ---------------------------------------------------------


class UniformDistribution(SingleContinuousDistribution):
    _argnames = ('left', 'right')

    @property
    def set(self):
        return Interval(self.left, self.right)

    @staticmethod
    def check(left, right):
        _value_check(left < right, "Lower limit should be less than Upper limit.")

    def pdf(self, x):
        left, right = self.left, self.right
        return Piecewise(
            (S.One/(right - left), And(left <= x, x <= right)),
            (S.Zero, True)
        )

    def _cdf(self, x):
        left, right = self.left, self.right
        return Piecewise(
            (S.Zero, x < left),
            ((x - left)/(right - left), x <= right),
            (S.One, True)
        )

    def _characteristic_function(self, t):
        left, right = self.left, self.right
        return Piecewise(((exp(I*t*right) - exp(I*t*left)) / (I*t*(right - left)), Ne(t, 0)),
                         (S.One, True))

    def _moment_generating_function(self, t):
        left, right = self.left, self.right
        return Piecewise(((exp(t*right) - exp(t*left)) / (t * (right - left)), Ne(t, 0)),
                         (S.One, True))

    def expectation(self, expr, var, **kwargs):
        kwargs['evaluate'] = True
        result = SingleContinuousDistribution.expectation(self, expr, var, **kwargs)
        result = result.subs({Max(self.left, self.right): self.right,
                              Min(self.left, self.right): self.left})
        return result


def Uniform(name, left, right):
    r"""
    Create a continuous random variable with a uniform distribution.

    Explanation
    ===========

    The density of the uniform distribution is given by

    .. math::
        f(x) := \begin{cases}
                  \frac{1}{b - a} & \text{for } x \in [a,b]  \\
                  0               & \text{otherwise}
                \end{cases}

    with :math:`x \in [a,b]`.

    Parameters
    ==========

    a : Real number, :math:`-\infty < a`, the left boundary
    b : Real number, :math:`a < b < \infty`, the right boundary

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Uniform, density, cdf, E, variance
    >>> from sympy import Symbol, simplify

    >>> a = Symbol("a", negative=True)
    >>> b = Symbol("b", positive=True)
    >>> z = Symbol("z")

    >>> X = Uniform("x", a, b)

    >>> density(X)(z)
    Piecewise((1/(-a + b), (b >= z) & (a <= z)), (0, True))

    >>> cdf(X)(z)
    Piecewise((0, a > z), ((-a + z)/(-a + b), b >= z), (1, True))

    >>> E(X)
    a/2 + b/2

    >>> simplify(variance(X))
    a**2/12 - a*b/6 + b**2/12

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Uniform_distribution_%28continuous%29
    .. [2] https://mathworld.wolfram.com/UniformDistribution.html

    """

    return rv(name, UniformDistribution, (left, right))

#-------------------------------------------------------------------------------
# UniformSum distribution ------------------------------------------------------


class UniformSumDistribution(SingleContinuousDistribution):
    _argnames = ('n',)

    @property
    def set(self):
        return Interval(0, self.n)

    @staticmethod
    def check(n):
        _value_check((n > 0, n.is_integer),
        "Parameter n must be positive integer.")

    def pdf(self, x):
        n = self.n
        k = Dummy("k")
        return 1/factorial(
            n - 1)*Sum((-1)**k*binomial(n, k)*(x - k)**(n - 1), (k, 0, floor(x)))

    def _cdf(self, x):
        n = self.n
        k = Dummy("k")
        return Piecewise((S.Zero, x < 0),
                        (1/factorial(n)*Sum((-1)**k*binomial(n, k)*(x - k)**(n),
                        (k, 0, floor(x))), x <= n),
                        (S.One, True))

    def _characteristic_function(self, t):
        return ((exp(I*t) - 1) / (I*t))**self.n

    def _moment_generating_function(self, t):
        return ((exp(t) - 1) / t)**self.n

def UniformSum(name, n):
    r"""
    Create a continuous random variable with an Irwin-Hall distribution.

    Explanation
    ===========

    The probability distribution function depends on a single parameter
    $n$ which is an integer.

    The density of the Irwin-Hall distribution is given by

    .. math ::
        f(x) := \frac{1}{(n-1)!}\sum_{k=0}^{\left\lfloor x\right\rfloor}(-1)^k
                \binom{n}{k}(x-k)^{n-1}

    Parameters
    ==========

    n : A positive integer, `n > 0`

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import UniformSum, density, cdf
    >>> from sympy import Symbol, pprint

    >>> n = Symbol("n", integer=True)
    >>> z = Symbol("z")

    >>> X = UniformSum("x", n)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
    floor(z)
      ___
      \  `
       \         k         n - 1 /n\
        )    (-1) *(-k + z)     *| |
       /                         \k/
      /__,
     k = 0
    --------------------------------
                (n - 1)!

    >>> cdf(X)(z)
    Piecewise((0, z < 0), (Sum((-1)**_k*(-_k + z)**n*binomial(n, _k),
                    (_k, 0, floor(z)))/factorial(n), n >= z), (1, True))


    Compute cdf with specific 'x' and 'n' values as follows :
    >>> cdf(UniformSum("x", 5), evaluate=False)(2).doit()
    9/40

    The argument evaluate=False prevents an attempt at evaluation
    of the sum for general n, before the argument 2 is passed.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Uniform_sum_distribution
    .. [2] https://mathworld.wolfram.com/UniformSumDistribution.html

    """

    return rv(name, UniformSumDistribution, (n, ))

#-------------------------------------------------------------------------------
# VonMises distribution --------------------------------------------------------


class VonMisesDistribution(SingleContinuousDistribution):
    _argnames = ('mu', 'k')

    set = Interval(0, 2*pi)

    @staticmethod
    def check(mu, k):
        _value_check(k > 0, "k must be positive")

    def pdf(self, x):
        mu, k = self.mu, self.k
        return exp(k*cos(x-mu)) / (2*pi*besseli(0, k))

def VonMises(name, mu, k):
    r"""
    Create a Continuous Random Variable with a von Mises distribution.

    Explanation
    ===========

    The density of the von Mises distribution is given by

    .. math::
        f(x) := \frac{e^{\kappa\cos(x-\mu)}}{2\pi I_0(\kappa)}

    with :math:`x \in [0,2\pi]`.

    Parameters
    ==========

    mu : Real number
        Measure of location.
    k : Real number
        Measure of concentration.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import VonMises, density
    >>> from sympy import Symbol, pprint

    >>> mu = Symbol("mu")
    >>> k = Symbol("k", positive=True)
    >>> z = Symbol("z")

    >>> X = VonMises("x", mu, k)

    >>> D = density(X)(z)
    >>> pprint(D, use_unicode=False)
         k*cos(mu - z)
        e
    ------------------
    2*pi*besseli(0, k)


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Von_Mises_distribution
    .. [2] https://mathworld.wolfram.com/vonMisesDistribution.html

    """

    return rv(name, VonMisesDistribution, (mu, k))

#-------------------------------------------------------------------------------
# Weibull distribution ---------------------------------------------------------


class WeibullDistribution(SingleContinuousDistribution):
    _argnames = ('alpha', 'beta')

    set = Interval(0, oo)

    @staticmethod
    def check(alpha, beta):
        _value_check(alpha > 0, "Alpha must be positive")
        _value_check(beta > 0, "Beta must be positive")

    def pdf(self, x):
        alpha, beta = self.alpha, self.beta
        return beta * (x/alpha)**(beta - 1) * exp(-(x/alpha)**beta) / alpha


def Weibull(name, alpha, beta):
    r"""
    Create a continuous random variable with a Weibull distribution.

    Explanation
    ===========

    The density of the Weibull distribution is given by

    .. math::
        f(x) := \begin{cases}
                  \frac{k}{\lambda}\left(\frac{x}{\lambda}\right)^{k-1}
                  e^{-(x/\lambda)^{k}} & x\geq0\\
                  0 & x<0
                \end{cases}

    Parameters
    ==========

    lambda : Real number, $\lambda > 0$, a scale
    k : Real number, $k > 0$, a shape

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Weibull, density, E, variance
    >>> from sympy import Symbol, simplify

    >>> l = Symbol("lambda", positive=True)
    >>> k = Symbol("k", positive=True)
    >>> z = Symbol("z")

    >>> X = Weibull("x", l, k)

    >>> density(X)(z)
    k*(z/lambda)**(k - 1)*exp(-(z/lambda)**k)/lambda

    >>> simplify(E(X))
    lambda*gamma(1 + 1/k)

    >>> simplify(variance(X))
    lambda**2*(-gamma(1 + 1/k)**2 + gamma(1 + 2/k))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Weibull_distribution
    .. [2] https://mathworld.wolfram.com/WeibullDistribution.html

    """

    return rv(name, WeibullDistribution, (alpha, beta))

#-------------------------------------------------------------------------------
# Wigner semicircle distribution -----------------------------------------------


class WignerSemicircleDistribution(SingleContinuousDistribution):
    _argnames = ('R',)

    @property
    def set(self):
        return Interval(-self.R, self.R)

    @staticmethod
    def check(R):
        _value_check(R > 0, "Radius R must be positive.")

    def pdf(self, x):
        R = self.R
        return 2/(pi*R**2)*sqrt(R**2 - x**2)

    def _characteristic_function(self, t):
        return Piecewise((2 * besselj(1, self.R*t) / (self.R*t), Ne(t, 0)),
                         (S.One, True))

    def _moment_generating_function(self, t):
        return Piecewise((2 * besseli(1, self.R*t) / (self.R*t), Ne(t, 0)),
                         (S.One, True))

def WignerSemicircle(name, R):
    r"""
    Create a continuous random variable with a Wigner semicircle distribution.

    Explanation
    ===========

    The density of the Wigner semicircle distribution is given by

    .. math::
        f(x) := \frac2{\pi R^2}\,\sqrt{R^2-x^2}

    with :math:`x \in [-R,R]`.

    Parameters
    ==========

    R : Real number, `R > 0`, the radius

    Returns
    =======

    A RandomSymbol.

    Examples
    ========

    >>> from sympy.stats import WignerSemicircle, density, E
    >>> from sympy import Symbol

    >>> R = Symbol("R", positive=True)
    >>> z = Symbol("z")

    >>> X = WignerSemicircle("x", R)

    >>> density(X)(z)
    2*sqrt(R**2 - z**2)/(pi*R**2)

    >>> E(X)
    0

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Wigner_semicircle_distribution
    .. [2] https://mathworld.wolfram.com/WignersSemicircleLaw.html

    """

    return rv(name, WignerSemicircleDistribution, (R,))
<|endoftext|>from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.symbol import Dummy
from sympy.integrals.integrals import Integral
from sympy.stats.rv import (NamedArgsMixin, random_symbols, _symbol_converter,
                        PSpace, RandomSymbol, is_random, Distribution)
from sympy.stats.crv import ContinuousDistribution, SingleContinuousPSpace
from sympy.stats.drv import DiscreteDistribution, SingleDiscretePSpace
from sympy.stats.frv import SingleFiniteDistribution, SingleFinitePSpace
from sympy.stats.crv_types import ContinuousDistributionHandmade
from sympy.stats.drv_types import DiscreteDistributionHandmade
from sympy.stats.frv_types import FiniteDistributionHandmade


class CompoundPSpace(PSpace):
    """
    A temporary Probability Space for the Compound Distribution. After
    Marginalization, this returns the corresponding Probability Space of the
    parent distribution.
    """

    def __new__(cls, s, distribution):
        s = _symbol_converter(s)
        if isinstance(distribution, ContinuousDistribution):
            return SingleContinuousPSpace(s, distribution)
        if isinstance(distribution, DiscreteDistribution):
            return SingleDiscretePSpace(s, distribution)
        if isinstance(distribution, SingleFiniteDistribution):
            return SingleFinitePSpace(s, distribution)
        if not isinstance(distribution, CompoundDistribution):
            raise ValueError("%s should be an isinstance of "
                        "CompoundDistribution"%(distribution))
        return Basic.__new__(cls, s, distribution)

    @property
    def value(self):
        return RandomSymbol(self.symbol, self)

    @property
    def symbol(self):
        return self.args[0]

    @property
    def is_Continuous(self):
        return self.distribution.is_Continuous

    @property
    def is_Finite(self):
        return self.distribution.is_Finite

    @property
    def is_Discrete(self):
        return self.distribution.is_Discrete

    @property
    def distribution(self):
        return self.args[1]

    @property
    def pdf(self):
        return self.distribution.pdf(self.symbol)

    @property
    def set(self):
        return self.distribution.set

    @property
    def domain(self):
        return self._get_newpspace().domain

    def _get_newpspace(self, evaluate=False):
        x = Dummy('x')
        parent_dist = self.distribution.args[0]
        func = Lambda(x, self.distribution.pdf(x, evaluate))
        new_pspace = self._transform_pspace(self.symbol, parent_dist, func)
        if new_pspace is not None:
            return new_pspace
        message = ("Compound Distribution for %s is not implemented yet" % str(parent_dist))
        raise NotImplementedError(message)

    def _transform_pspace(self, sym, dist, pdf):
        """
        This function returns the new pspace of the distribution using handmade
        Distributions and their corresponding pspace.
        """
        pdf = Lambda(sym, pdf(sym))
        _set = dist.set
        if isinstance(dist, ContinuousDistribution):
            return SingleContinuousPSpace(sym, ContinuousDistributionHandmade(pdf, _set))
        elif isinstance(dist, DiscreteDistribution):
            return SingleDiscretePSpace(sym, DiscreteDistributionHandmade(pdf, _set))
        elif isinstance(dist, SingleFiniteDistribution):
            dens = {k: pdf(k) for k in _set}
            return SingleFinitePSpace(sym, FiniteDistributionHandmade(dens))

    def compute_density(self, expr, *, compound_evaluate=True, **kwargs):
        new_pspace = self._get_newpspace(compound_evaluate)
        expr = expr.subs({self.value: new_pspace.value})
        return new_pspace.compute_density(expr, **kwargs)

    def compute_cdf(self, expr, *, compound_evaluate=True, **kwargs):
        new_pspace = self._get_newpspace(compound_evaluate)
        expr = expr.subs({self.value: new_pspace.value})
        return new_pspace.compute_cdf(expr, **kwargs)

    def compute_expectation(self, expr, rvs=None, evaluate=False, **kwargs):
        new_pspace = self._get_newpspace(evaluate)
        expr = expr.subs({self.value: new_pspace.value})
        if rvs:
            rvs = rvs.subs({self.value: new_pspace.value})
        if isinstance(new_pspace, SingleFinitePSpace):
            return new_pspace.compute_expectation(expr, rvs, **kwargs)
        return new_pspace.compute_expectation(expr, rvs, evaluate, **kwargs)

    def probability(self, condition, *, compound_evaluate=True, **kwargs):
        new_pspace = self._get_newpspace(compound_evaluate)
        condition = condition.subs({self.value: new_pspace.value})
        return new_pspace.probability(condition)

    def conditional_space(self, condition, *, compound_evaluate=True, **kwargs):
        new_pspace = self._get_newpspace(compound_evaluate)
        condition = condition.subs({self.value: new_pspace.value})
        return new_pspace.conditional_space(condition)


class CompoundDistribution(Distribution, NamedArgsMixin):
    """
    Class for Compound Distributions.

    Parameters
    ==========

    dist : Distribution
        Distribution must contain a random parameter

    Examples
    ========

    >>> from sympy.stats.compound_rv import CompoundDistribution
    >>> from sympy.stats.crv_types import NormalDistribution
    >>> from sympy.stats import Normal
    >>> from sympy.abc import x
    >>> X = Normal('X', 2, 4)
    >>> N = NormalDistribution(X, 4)
    >>> C = CompoundDistribution(N)
    >>> C.set
    Interval(-oo, oo)
    >>> C.pdf(x, evaluate=True).simplify()
    exp(-x**2/64 + x/16 - 1/16)/(8*sqrt(pi))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Compound_probability_distribution

    """

    def __new__(cls, dist):
        if not isinstance(dist, (ContinuousDistribution,
                SingleFiniteDistribution, DiscreteDistribution)):
            message = "Compound Distribution for %s is not implemented yet" % str(dist)
            raise NotImplementedError(message)
        if not cls._compound_check(dist):
            return dist
        return Basic.__new__(cls, dist)

    @property
    def set(self):
        return self.args[0].set

    @property
    def is_Continuous(self):
        return isinstance(self.args[0], ContinuousDistribution)

    @property
    def is_Finite(self):
        return isinstance(self.args[0], SingleFiniteDistribution)

    @property
    def is_Discrete(self):
        return isinstance(self.args[0], DiscreteDistribution)

    def pdf(self, x, evaluate=False):
        dist = self.args[0]
        randoms = [rv for rv in dist.args if is_random(rv)]
        if isinstance(dist, SingleFiniteDistribution):
            y = Dummy('y', integer=True, negative=False)
            expr = dist.pmf(y)
        else:
            y = Dummy('y')
            expr = dist.pdf(y)
        for rv in randoms:
            expr = self._marginalise(expr, rv, evaluate)
        return Lambda(y, expr)(x)

    def _marginalise(self, expr, rv, evaluate):
        if isinstance(rv.pspace.distribution, SingleFiniteDistribution):
            rv_dens = rv.pspace.distribution.pmf(rv)
        else:
            rv_dens = rv.pspace.distribution.pdf(rv)
        rv_dom = rv.pspace.domain.set
        if rv.pspace.is_Discrete or rv.pspace.is_Finite:
            expr = Sum(expr*rv_dens, (rv, rv_dom._inf,
                    rv_dom._sup))
        else:
            expr = Integral(expr*rv_dens, (rv, rv_dom._inf,
                    rv_dom._sup))
        if evaluate:
            return expr.doit()
        return expr

    @classmethod
    def _compound_check(self, dist):
        """
        Checks if the given distribution contains random parameters.
        """
        randoms = []
        for arg in dist.args:
            randoms.extend(random_symbols(arg))
        if len(randoms) == 0:
            return False
        return True
<|endoftext|>from functools import singledispatch

from sympy.core.symbol import Dummy
from sympy.functions.elementary.exponential import exp
from sympy.utilities.lambdify import lambdify
from sympy.external import import_module
from sympy.stats import DiscreteDistributionHandmade
from sympy.stats.crv import SingleContinuousDistribution
from sympy.stats.crv_types import ChiSquaredDistribution, ExponentialDistribution, GammaDistribution, \
    LogNormalDistribution, NormalDistribution, ParetoDistribution, UniformDistribution, BetaDistribution, \
    StudentTDistribution, CauchyDistribution
from sympy.stats.drv_types import GeometricDistribution, LogarithmicDistribution, NegativeBinomialDistribution, \
    PoissonDistribution, SkellamDistribution, YuleSimonDistribution, ZetaDistribution
from sympy.stats.frv import SingleFiniteDistribution


scipy = import_module("scipy", import_kwargs={'fromlist':['stats']})


@singledispatch
def do_sample_scipy(dist, size, seed):
    return None


# CRV

@do_sample_scipy.register(SingleContinuousDistribution)
def _(dist: SingleContinuousDistribution, size, seed):
    # if we don't need to make a handmade pdf, we won't
    import scipy.stats

    z = Dummy('z')
    handmade_pdf = lambdify(z, dist.pdf(z), ['numpy', 'scipy'])

    class scipy_pdf(scipy.stats.rv_continuous):
        def _pdf(dist, x):
            return handmade_pdf(x)

    scipy_rv = scipy_pdf(a=float(dist.set._inf),
                         b=float(dist.set._sup), name='scipy_pdf')
    return scipy_rv.rvs(size=size, random_state=seed)


@do_sample_scipy.register(ChiSquaredDistribution)
def _(dist: ChiSquaredDistribution, size, seed):
    # same parametrisation
    return scipy.stats.chi2.rvs(df=float(dist.k), size=size, random_state=seed)


@do_sample_scipy.register(ExponentialDistribution)
def _(dist: ExponentialDistribution, size, seed):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html#scipy.stats.expon
    return scipy.stats.expon.rvs(scale=1 / float(dist.rate), size=size, random_state=seed)


@do_sample_scipy.register(GammaDistribution)
def _(dist: GammaDistribution, size, seed):
    # https://stackoverflow.com/questions/42150965/how-to-plot-gamma-distribution-with-alpha-and-beta-parameters-in-python
    return scipy.stats.gamma.rvs(a=float(dist.k), scale=float(dist.theta), size=size, random_state=seed)


@do_sample_scipy.register(LogNormalDistribution)
def _(dist: LogNormalDistribution, size, seed):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html
    return scipy.stats.lognorm.rvs(scale=float(exp(dist.mean)), s=float(dist.std), size=size, random_state=seed)


@do_sample_scipy.register(NormalDistribution)
def _(dist: NormalDistribution, size, seed):
    return scipy.stats.norm.rvs(loc=float(dist.mean), scale=float(dist.std), size=size, random_state=seed)


@do_sample_scipy.register(ParetoDistribution)
def _(dist: ParetoDistribution, size, seed):
    # https://stackoverflow.com/questions/42260519/defining-pareto-distribution-in-python-scipy
    return scipy.stats.pareto.rvs(b=float(dist.alpha), scale=float(dist.xm), size=size, random_state=seed)


@do_sample_scipy.register(StudentTDistribution)
def _(dist: StudentTDistribution, size, seed):
    return scipy.stats.t.rvs(df=float(dist.nu), size=size, random_state=seed)


@do_sample_scipy.register(UniformDistribution)
def _(dist: UniformDistribution, size, seed):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html
    return scipy.stats.uniform.rvs(loc=float(dist.left), scale=float(dist.right - dist.left), size=size, random_state=seed)


@do_sample_scipy.register(BetaDistribution)
def _(dist: BetaDistribution, size, seed):
    # same parametrisation
    return scipy.stats.beta.rvs(a=float(dist.alpha), b=float(dist.beta), size=size, random_state=seed)


@do_sample_scipy.register(CauchyDistribution)
def _(dist: CauchyDistribution, size, seed):
    return scipy.stats.cauchy.rvs(loc=float(dist.x0), scale=float(dist.gamma), size=size, random_state=seed)


# DRV:

@do_sample_scipy.register(DiscreteDistributionHandmade)
def _(dist: DiscreteDistributionHandmade, size, seed):
    from scipy.stats import rv_discrete

    z = Dummy('z')
    handmade_pmf = lambdify(z, dist.pdf(z), ['numpy', 'scipy'])

    class scipy_pmf(rv_discrete):
        def _pmf(dist, x):
            return handmade_pmf(x)

    scipy_rv = scipy_pmf(a=float(dist.set._inf), b=float(dist.set._sup),
                         name='scipy_pmf')
    return scipy_rv.rvs(size=size, random_state=seed)


@do_sample_scipy.register(GeometricDistribution)
def _(dist: GeometricDistribution, size, seed):
    return scipy.stats.geom.rvs(p=float(dist.p), size=size, random_state=seed)


@do_sample_scipy.register(LogarithmicDistribution)
def _(dist: LogarithmicDistribution, size, seed):
    return scipy.stats.logser.rvs(p=float(dist.p), size=size, random_state=seed)


@do_sample_scipy.register(NegativeBinomialDistribution)
def _(dist: NegativeBinomialDistribution, size, seed):
    return scipy.stats.nbinom.rvs(n=float(dist.r), p=float(dist.p), size=size, random_state=seed)


@do_sample_scipy.register(PoissonDistribution)
def _(dist: PoissonDistribution, size, seed):
    return scipy.stats.poisson.rvs(mu=float(dist.lamda), size=size, random_state=seed)


@do_sample_scipy.register(SkellamDistribution)
def _(dist: SkellamDistribution, size, seed):
    return scipy.stats.skellam.rvs(mu1=float(dist.mu1), mu2=float(dist.mu2), size=size, random_state=seed)


@do_sample_scipy.register(YuleSimonDistribution)
def _(dist: YuleSimonDistribution, size, seed):
    return scipy.stats.yulesimon.rvs(alpha=float(dist.rho), size=size, random_state=seed)


@do_sample_scipy.register(ZetaDistribution)
def _(dist: ZetaDistribution, size, seed):
    return scipy.stats.zipf.rvs(a=float(dist.s), size=size, random_state=seed)


# FRV:

@do_sample_scipy.register(SingleFiniteDistribution)
def _(dist: SingleFiniteDistribution, size, seed):
    # scipy can handle with custom distributions

    from scipy.stats import rv_discrete
    density_ = dist.dict
    x, y = [], []
    for k, v in density_.items():
        x.append(int(k))
        y.append(float(v))
    scipy_rv = rv_discrete(name='scipy_rv', values=(x, y))
    return scipy_rv.rvs(size=size, random_state=seed)
<|endoftext|>from functools import singledispatch
from sympy.external import import_module
from sympy.stats.crv_types import BetaDistribution, CauchyDistribution, ChiSquaredDistribution, ExponentialDistribution, \
    GammaDistribution, LogNormalDistribution, NormalDistribution, ParetoDistribution, UniformDistribution, \
    GaussianInverseDistribution
from sympy.stats.drv_types import PoissonDistribution, GeometricDistribution, NegativeBinomialDistribution
from sympy.stats.frv_types import BinomialDistribution, BernoulliDistribution


try:
    import pymc
except ImportError:
    pymc = import_module('pymc3')

@singledispatch
def do_sample_pymc(dist):
    return None


# CRV:

@do_sample_pymc.register(BetaDistribution)
def _(dist: BetaDistribution):
    return pymc.Beta('X', alpha=float(dist.alpha), beta=float(dist.beta))


@do_sample_pymc.register(CauchyDistribution)
def _(dist: CauchyDistribution):
    return pymc.Cauchy('X', alpha=float(dist.x0), beta=float(dist.gamma))


@do_sample_pymc.register(ChiSquaredDistribution)
def _(dist: ChiSquaredDistribution):
    return pymc.ChiSquared('X', nu=float(dist.k))


@do_sample_pymc.register(ExponentialDistribution)
def _(dist: ExponentialDistribution):
    return pymc.Exponential('X', lam=float(dist.rate))


@do_sample_pymc.register(GammaDistribution)
def _(dist: GammaDistribution):
    return pymc.Gamma('X', alpha=float(dist.k), beta=1 / float(dist.theta))


@do_sample_pymc.register(LogNormalDistribution)
def _(dist: LogNormalDistribution):
    return pymc.Lognormal('X', mu=float(dist.mean), sigma=float(dist.std))


@do_sample_pymc.register(NormalDistribution)
def _(dist: NormalDistribution):
    return pymc.Normal('X', float(dist.mean), float(dist.std))


@do_sample_pymc.register(GaussianInverseDistribution)
def _(dist: GaussianInverseDistribution):
    return pymc.Wald('X', mu=float(dist.mean), lam=float(dist.shape))


@do_sample_pymc.register(ParetoDistribution)
def _(dist: ParetoDistribution):
    return pymc.Pareto('X', alpha=float(dist.alpha), m=float(dist.xm))


@do_sample_pymc.register(UniformDistribution)
def _(dist: UniformDistribution):
    return pymc.Uniform('X', lower=float(dist.left), upper=float(dist.right))


# DRV:

@do_sample_pymc.register(GeometricDistribution)
def _(dist: GeometricDistribution):
    return pymc.Geometric('X', p=float(dist.p))


@do_sample_pymc.register(NegativeBinomialDistribution)
def _(dist: NegativeBinomialDistribution):
    return pymc.NegativeBinomial('X', mu=float((dist.p * dist.r) / (1 - dist.p)),
                                  alpha=float(dist.r))


@do_sample_pymc.register(PoissonDistribution)
def _(dist: PoissonDistribution):
    return pymc.Poisson('X', mu=float(dist.lamda))


# FRV:

@do_sample_pymc.register(BernoulliDistribution)
def _(dist: BernoulliDistribution):
    return pymc.Bernoulli('X', p=float(dist.p))


@do_sample_pymc.register(BinomialDistribution)
def _(dist: BinomialDistribution):
    return pymc.Binomial('X', n=int(dist.n), p=float(dist.p))
<|endoftext|>from functools import singledispatch

from sympy.external import import_module
from sympy.stats.crv_types import BetaDistribution, ChiSquaredDistribution, ExponentialDistribution, GammaDistribution, \
    LogNormalDistribution, NormalDistribution, ParetoDistribution, UniformDistribution, FDistributionDistribution, GumbelDistribution, LaplaceDistribution, \
    LogisticDistribution, RayleighDistribution, TriangularDistribution
from sympy.stats.drv_types import GeometricDistribution, PoissonDistribution, ZetaDistribution
from sympy.stats.frv_types import BinomialDistribution, HypergeometricDistribution


numpy = import_module('numpy')


@singledispatch
def do_sample_numpy(dist, size, rand_state):
    return None


# CRV:

@do_sample_numpy.register(BetaDistribution)
def _(dist: BetaDistribution, size, rand_state):
    return rand_state.beta(a=float(dist.alpha), b=float(dist.beta), size=size)


@do_sample_numpy.register(ChiSquaredDistribution)
def _(dist: ChiSquaredDistribution, size, rand_state):
    return rand_state.chisquare(df=float(dist.k), size=size)


@do_sample_numpy.register(ExponentialDistribution)
def _(dist: ExponentialDistribution, size, rand_state):
    return rand_state.exponential(1 / float(dist.rate), size=size)

@do_sample_numpy.register(FDistributionDistribution)
def _(dist: FDistributionDistribution, size, rand_state):
    return rand_state.f(dfnum = float(dist.d1), dfden = float(dist.d2), size=size)

@do_sample_numpy.register(GammaDistribution)
def _(dist: GammaDistribution, size, rand_state):
    return rand_state.gamma(shape = float(dist.k), scale = float(dist.theta), size=size)

@do_sample_numpy.register(GumbelDistribution)
def _(dist: GumbelDistribution, size, rand_state):
    return rand_state.gumbel(loc = float(dist.mu), scale = float(dist.beta), size=size)

@do_sample_numpy.register(LaplaceDistribution)
def _(dist: LaplaceDistribution, size, rand_state):
    return rand_state.laplace(loc = float(dist.mu), scale = float(dist.b), size=size)

@do_sample_numpy.register(LogisticDistribution)
def _(dist: LogisticDistribution, size, rand_state):
    return rand_state.logistic(loc = float(dist.mu), scale = float(dist.s), size=size)

@do_sample_numpy.register(LogNormalDistribution)
def _(dist: LogNormalDistribution, size, rand_state):
    return rand_state.lognormal(mean = float(dist.mean), sigma = float(dist.std), size=size)

@do_sample_numpy.register(NormalDistribution)
def _(dist: NormalDistribution, size, rand_state):
    return rand_state.normal(loc = float(dist.mean), scale = float(dist.std), size=size)

@do_sample_numpy.register(RayleighDistribution)
def _(dist: RayleighDistribution, size, rand_state):
    return rand_state.rayleigh(scale = float(dist.sigma), size=size)

@do_sample_numpy.register(ParetoDistribution)
def _(dist: ParetoDistribution, size, rand_state):
    return (numpy.random.pareto(a=float(dist.alpha), size=size) + 1) * float(dist.xm)

@do_sample_numpy.register(TriangularDistribution)
def _(dist: TriangularDistribution, size, rand_state):
    return rand_state.triangular(left = float(dist.a), mode = float(dist.b), right = float(dist.c), size=size)

@do_sample_numpy.register(UniformDistribution)
def _(dist: UniformDistribution, size, rand_state):
    return rand_state.uniform(low=float(dist.left), high=float(dist.right), size=size)


# DRV:

@do_sample_numpy.register(GeometricDistribution)
def _(dist: GeometricDistribution, size, rand_state):
    return rand_state.geometric(p=float(dist.p), size=size)


@do_sample_numpy.register(PoissonDistribution)
def _(dist: PoissonDistribution, size, rand_state):
    return rand_state.poisson(lam=float(dist.lamda), size=size)


@do_sample_numpy.register(ZetaDistribution)
def _(dist: ZetaDistribution, size, rand_state):
    return rand_state.zipf(a=float(dist.s), size=size)


# FRV:

@do_sample_numpy.register(BinomialDistribution)
def _(dist: BinomialDistribution, size, rand_state):
    return rand_state.binomial(n=int(dist.n), p=float(dist.p), size=size)

@do_sample_numpy.register(HypergeometricDistribution)
def _(dist: HypergeometricDistribution, size, rand_state):
    return rand_state.hypergeometric(ngood = int(dist.N), nbad = int(dist.m), nsample = int(dist.n), size=size)
<|endoftext|>""" rewrite of lambdify - This stuff is not stable at all.

It is for internal use in the new plotting module.
It may (will! see the Q'n'A in the source) be rewritten.

It's completely self contained. Especially it does not use lambdarepr.

It does not aim to replace the current lambdify. Most importantly it will never
ever support anything else than SymPy expressions (no Matrices, dictionaries
and so on).
"""


import re
from sympy.core.numbers import (I, NumberSymbol, oo, zoo)
from sympy.core.symbol import Symbol
from sympy.utilities.iterables import numbered_symbols

#  We parse the expression string into a tree that identifies functions. Then
# we translate the names of the functions and we translate also some strings
# that are not names of functions (all this according to translation
# dictionaries).
#  If the translation goes to another module (like numpy) the
# module is imported and 'func' is translated to 'module.func'.
#  If a function can not be translated, the inner nodes of that part of the
# tree are not translated. So if we have Integral(sqrt(x)), sqrt is not
# translated to np.sqrt and the Integral does not crash.
#  A namespace for all this is generated by crawling the (func, args) tree of
# the expression. The creation of this namespace involves many ugly
# workarounds.
#  The namespace consists of all the names needed for the SymPy expression and
# all the name of modules used for translation. Those modules are imported only
# as a name (import numpy as np) in order to keep the namespace small and
# manageable.

#  Please, if there is a bug, do not try to fix it here! Rewrite this by using
# the method proposed in the last Q'n'A below. That way the new function will
# work just as well, be just as simple, but it wont need any new workarounds.
#  If you insist on fixing it here, look at the workarounds in the function
# sympy_expression_namespace and in lambdify.

# Q: Why are you not using Python abstract syntax tree?
# A: Because it is more complicated and not much more powerful in this case.

# Q: What if I have Symbol('sin') or g=Function('f')?
# A: You will break the algorithm. We should use srepr to defend against this?
#  The problem with Symbol('sin') is that it will be printed as 'sin'. The
# parser will distinguish it from the function 'sin' because functions are
# detected thanks to the opening parenthesis, but the lambda expression won't
# understand the difference if we have also the sin function.
# The solution (complicated) is to use srepr and maybe ast.
#  The problem with the g=Function('f') is that it will be printed as 'f' but in
# the global namespace we have only 'g'. But as the same printer is used in the
# constructor of the namespace there will be no problem.

# Q: What if some of the printers are not printing as expected?
# A: The algorithm wont work. You must use srepr for those cases. But even
# srepr may not print well. All problems with printers should be considered
# bugs.

# Q: What about _imp_ functions?
# A: Those are taken care for by evalf. A special case treatment will work
# faster but it's not worth the code complexity.

# Q: Will ast fix all possible problems?
# A: No. You will always have to use some printer. Even srepr may not work in
# some cases. But if the printer does not work, that should be considered a
# bug.

# Q: Is there same way to fix all possible problems?
# A: Probably by constructing our strings ourself by traversing the (func,
# args) tree and creating the namespace at the same time. That actually sounds
# good.

from sympy.external import import_module
import warnings

#TODO debugging output


class vectorized_lambdify:
    """ Return a sufficiently smart, vectorized and lambdified function.

    Returns only reals.

    Explanation
    ===========

    This function uses experimental_lambdify to created a lambdified
    expression ready to be used with numpy. Many of the functions in SymPy
    are not implemented in numpy so in some cases we resort to Python cmath or
    even to evalf.

    The following translations are tried:
      only numpy complex
      - on errors raised by SymPy trying to work with ndarray:
          only Python cmath and then vectorize complex128

    When using Python cmath there is no need for evalf or float/complex
    because Python cmath calls those.

    This function never tries to mix numpy directly with evalf because numpy
    does not understand SymPy Float. If this is needed one can use the
    float_wrap_evalf/complex_wrap_evalf options of experimental_lambdify or
    better one can be explicit about the dtypes that numpy works with.
    Check numpy bug http://projects.scipy.org/numpy/ticket/1013 to know what
    types of errors to expect.
    """
    def __init__(self, args, expr):
        self.args = args
        self.expr = expr
        self.np = import_module('numpy')

        self.lambda_func_1 = experimental_lambdify(
            args, expr, use_np=True)
        self.vector_func_1 = self.lambda_func_1

        self.lambda_func_2 = experimental_lambdify(
            args, expr, use_python_cmath=True)
        self.vector_func_2 = self.np.vectorize(
            self.lambda_func_2, otypes=[complex])

        self.vector_func = self.vector_func_1
        self.failure = False

    def __call__(self, *args):
        np = self.np

        try:
            temp_args = (np.array(a, dtype=complex) for a in args)
            results = self.vector_func(*temp_args)
            results = np.ma.masked_where(
                np.abs(results.imag) > 1e-7 * np.abs(results),
                results.real, copy=False)
            return results
        except ValueError:
            if self.failure:
                raise

            self.failure = True
            self.vector_func = self.vector_func_2
            warnings.warn(
                'The evaluation of the expression is problematic. '
                'We are trying a failback method that may still work. '
                'Please report this as a bug.')
            return self.__call__(*args)


class lambdify:
    """Returns the lambdified function.

    Explanation
    ===========

    This function uses experimental_lambdify to create a lambdified
    expression. It uses cmath to lambdify the expression. If the function
    is not implemented in Python cmath, Python cmath calls evalf on those
    functions.
    """

    def __init__(self, args, expr):
        self.args = args
        self.expr = expr
        self.lambda_func_1 = experimental_lambdify(
            args, expr, use_python_cmath=True, use_evalf=True)
        self.lambda_func_2 = experimental_lambdify(
            args, expr, use_python_math=True, use_evalf=True)
        self.lambda_func_3 = experimental_lambdify(
            args, expr, use_evalf=True, complex_wrap_evalf=True)
        self.lambda_func = self.lambda_func_1
        self.failure = False

    def __call__(self, args):
        try:
            #The result can be sympy.Float. Hence wrap it with complex type.
            result = complex(self.lambda_func(args))
            if abs(result.imag) > 1e-7 * abs(result):
                return None
            return result.real
        except (ZeroDivisionError, OverflowError):
            return None
        except TypeError as e:
            if self.failure:
                raise e

            if self.lambda_func == self.lambda_func_1:
                self.lambda_func = self.lambda_func_2
                return self.__call__(args)

            self.failure = True
            self.lambda_func = self.lambda_func_3
            warnings.warn(
                'The evaluation of the expression is problematic. '
                'We are trying a failback method that may still work. '
                'Please report this as a bug.', stacklevel=2)
            return self.__call__(args)


def experimental_lambdify(*args, **kwargs):
    l = Lambdifier(*args, **kwargs)
    return l


class Lambdifier:
    def __init__(self, args, expr, print_lambda=False, use_evalf=False,
                 float_wrap_evalf=False, complex_wrap_evalf=False,
                 use_np=False, use_python_math=False, use_python_cmath=False,
                 use_interval=False):

        self.print_lambda = print_lambda
        self.use_evalf = use_evalf
        self.float_wrap_evalf = float_wrap_evalf
        self.complex_wrap_evalf = complex_wrap_evalf
        self.use_np = use_np
        self.use_python_math = use_python_math
        self.use_python_cmath = use_python_cmath
        self.use_interval = use_interval

        # Constructing the argument string
        # - check
        if not all(isinstance(a, Symbol) for a in args):
            raise ValueError('The arguments must be Symbols.')
        # - use numbered symbols
        syms = numbered_symbols(exclude=expr.free_symbols)
        newargs = [next(syms) for _ in args]
        expr = expr.xreplace(dict(zip(args, newargs)))
        argstr = ', '.join([str(a) for a in newargs])
        del syms, newargs, args

        # Constructing the translation dictionaries and making the translation
        self.dict_str = self.get_dict_str()
        self.dict_fun = self.get_dict_fun()
        exprstr = str(expr)
        newexpr = self.tree2str_translate(self.str2tree(exprstr))

        # Constructing the namespaces
        namespace = {}
        namespace.update(self.sympy_atoms_namespace(expr))
        namespace.update(self.sympy_expression_namespace(expr))
        # XXX Workaround
        # Ugly workaround because Pow(a,Half) prints as sqrt(a)
        # and sympy_expression_namespace can not catch it.
        from sympy.functions.elementary.miscellaneous import sqrt
        namespace.update({'sqrt': sqrt})
        namespace.update({'Eq': lambda x, y: x == y})
        namespace.update({'Ne': lambda x, y: x != y})
        # End workaround.
        if use_python_math:
            namespace.update({'math': __import__('math')})
        if use_python_cmath:
            namespace.update({'cmath': __import__('cmath')})
        if use_np:
            try:
                namespace.update({'np': __import__('numpy')})
            except ImportError:
                raise ImportError(
                    'experimental_lambdify failed to import numpy.')
        if use_interval:
            namespace.update({'imath': __import__(
                'sympy.plotting.intervalmath', fromlist=['intervalmath'])})
            namespace.update({'math': __import__('math')})

        # Construct the lambda
        if self.print_lambda:
            print(newexpr)
        eval_str = 'lambda %s : ( %s )' % (argstr, newexpr)
        self.eval_str = eval_str
        exec("MYNEWLAMBDA = %s" % eval_str, namespace)
        self.lambda_func = namespace['MYNEWLAMBDA']

    def __call__(self, *args, **kwargs):
        return self.lambda_func(*args, **kwargs)


    ##############################################################################
    # Dicts for translating from SymPy to other modules
    ##############################################################################
    ###
    # builtins
    ###
    # Functions with different names in builtins
    builtin_functions_different = {
        'Min': 'min',
        'Max': 'max',
        'Abs': 'abs',
    }

    # Strings that should be translated
    builtin_not_functions = {
        'I': '1j',
#        'oo': '1e400',
    }

    ###
    # numpy
    ###

    # Functions that are the same in numpy
    numpy_functions_same = [
        'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'exp', 'log',
        'sqrt', 'floor', 'conjugate', 'sign',
    ]

    # Functions with different names in numpy
    numpy_functions_different = {
        "acos": "arccos",
        "acosh": "arccosh",
        "arg": "angle",
        "asin": "arcsin",
        "asinh": "arcsinh",
        "atan": "arctan",
        "atan2": "arctan2",
        "atanh": "arctanh",
        "ceiling": "ceil",
        "im": "imag",
        "ln": "log",
        "Max": "amax",
        "Min": "amin",
        "re": "real",
        "Abs": "abs",
    }

    # Strings that should be translated
    numpy_not_functions = {
        'pi': 'np.pi',
        'oo': 'np.inf',
        'E': 'np.e',
    }

    ###
    # Python math
    ###

    # Functions that are the same in math
    math_functions_same = [
        'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
        'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
        'exp', 'log', 'erf', 'sqrt', 'floor', 'factorial', 'gamma',
    ]

    # Functions with different names in math
    math_functions_different = {
        'ceiling': 'ceil',
        'ln': 'log',
        'loggamma': 'lgamma'
    }

    # Strings that should be translated
    math_not_functions = {
        'pi': 'math.pi',
        'E': 'math.e',
    }

    ###
    # Python cmath
    ###

    # Functions that are the same in cmath
    cmath_functions_same = [
        'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
        'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
        'exp', 'log', 'sqrt',
    ]

    # Functions with different names in cmath
    cmath_functions_different = {
        'ln': 'log',
        'arg': 'phase',
    }

    # Strings that should be translated
    cmath_not_functions = {
        'pi': 'cmath.pi',
        'E': 'cmath.e',
    }

    ###
    # intervalmath
    ###

    interval_not_functions = {
        'pi': 'math.pi',
        'E': 'math.e'
    }

    interval_functions_same = [
        'sin', 'cos', 'exp', 'tan', 'atan', 'log',
        'sqrt', 'cosh', 'sinh', 'tanh', 'floor',
        'acos', 'asin', 'acosh', 'asinh', 'atanh',
        'Abs', 'And', 'Or'
    ]

    interval_functions_different = {
        'Min': 'imin',
        'Max': 'imax',
        'ceiling': 'ceil',

    }

    ###
    # mpmath, etc
    ###
    #TODO

    ###
    # Create the final ordered tuples of dictionaries
    ###

    # For strings
    def get_dict_str(self):
        dict_str = dict(self.builtin_not_functions)
        if self.use_np:
            dict_str.update(self.numpy_not_functions)
        if self.use_python_math:
            dict_str.update(self.math_not_functions)
        if self.use_python_cmath:
            dict_str.update(self.cmath_not_functions)
        if self.use_interval:
            dict_str.update(self.interval_not_functions)
        return dict_str

    # For functions
    def get_dict_fun(self):
        dict_fun = dict(self.builtin_functions_different)
        if self.use_np:
            for s in self.numpy_functions_same:
                dict_fun[s] = 'np.' + s
            for k, v in self.numpy_functions_different.items():
                dict_fun[k] = 'np.' + v
        if self.use_python_math:
            for s in self.math_functions_same:
                dict_fun[s] = 'math.' + s
            for k, v in self.math_functions_different.items():
                dict_fun[k] = 'math.' + v
        if self.use_python_cmath:
            for s in self.cmath_functions_same:
                dict_fun[s] = 'cmath.' + s
            for k, v in self.cmath_functions_different.items():
                dict_fun[k] = 'cmath.' + v
        if self.use_interval:
            for s in self.interval_functions_same:
                dict_fun[s] = 'imath.' + s
            for k, v in self.interval_functions_different.items():
                dict_fun[k] = 'imath.' + v
        return dict_fun

    ##############################################################################
    # The translator functions, tree parsers, etc.
    ##############################################################################

    def str2tree(self, exprstr):
        """Converts an expression string to a tree.

        Explanation
        ===========

        Functions are represented by ('func_name(', tree_of_arguments).
        Other expressions are (head_string, mid_tree, tail_str).
        Expressions that do not contain functions are directly returned.

        Examples
        ========

        >>> from sympy.abc import x, y, z
        >>> from sympy import Integral, sin
        >>> from sympy.plotting.experimental_lambdify import Lambdifier
        >>> str2tree = Lambdifier([x], x).str2tree

        >>> str2tree(str(Integral(x, (x, 1, y))))
        ('', ('Integral(', 'x, (x, 1, y)'), ')')
        >>> str2tree(str(x+y))
        'x + y'
        >>> str2tree(str(x+y*sin(z)+1))
        ('x + y*', ('sin(', 'z'), ') + 1')
        >>> str2tree('sin(y*(y + 1.1) + (sin(y)))')
        ('', ('sin(', ('y*(y + 1.1) + (', ('sin(', 'y'), '))')), ')')
        """
        #matches the first 'function_name('
        first_par = re.search(r'(\w+\()', exprstr)
        if first_par is None:
            return exprstr
        else:
            start = first_par.start()
            end = first_par.end()
            head = exprstr[:start]
            func = exprstr[start:end]
            tail = exprstr[end:]
            count = 0
            for i, c in enumerate(tail):
                if c == '(':
                    count += 1
                elif c == ')':
                    count -= 1
                if count == -1:
                    break
            func_tail = self.str2tree(tail[:i])
            tail = self.str2tree(tail[i:])
            return (head, (func, func_tail), tail)

    @classmethod
    def tree2str(cls, tree):
        """Converts a tree to string without translations.

        Examples
        ========

        >>> from sympy.abc import x, y, z
        >>> from sympy import sin
        >>> from sympy.plotting.experimental_lambdify import Lambdifier
        >>> str2tree = Lambdifier([x], x).str2tree
        >>> tree2str = Lambdifier([x], x).tree2str

        >>> tree2str(str2tree(str(x+y*sin(z)+1)))
        'x + y*sin(z) + 1'
        """
        if isinstance(tree, str):
            return tree
        else:
            return ''.join(map(cls.tree2str, tree))

    def tree2str_translate(self, tree):
        """Converts a tree to string with translations.

        Explanation
        ===========

        Function names are translated by translate_func.
        Other strings are translated by translate_str.
        """
        if isinstance(tree, str):
            return self.translate_str(tree)
        elif isinstance(tree, tuple) and len(tree) == 2:
            return self.translate_func(tree[0][:-1], tree[1])
        else:
            return ''.join([self.tree2str_translate(t) for t in tree])

    def translate_str(self, estr):
        """Translate substrings of estr using in order the dictionaries in
        dict_tuple_str."""
        for pattern, repl in self.dict_str.items():
                estr = re.sub(pattern, repl, estr)
        return estr

    def translate_func(self, func_name, argtree):
        """Translate function names and the tree of arguments.

        Explanation
        ===========

        If the function name is not in the dictionaries of dict_tuple_fun then the
        function is surrounded by a float((...).evalf()).

        The use of float is necessary as np.<function>(sympy.Float(..)) raises an
        error."""
        if func_name in self.dict_fun:
            new_name = self.dict_fun[func_name]
            argstr = self.tree2str_translate(argtree)
            return new_name + '(' + argstr
        elif func_name in ['Eq', 'Ne']:
            op = {'Eq': '==', 'Ne': '!='}
            return "(lambda x, y: x {} y)({}".format(op[func_name], self.tree2str_translate(argtree))
        else:
            template = '(%s(%s)).evalf(' if self.use_evalf else '%s(%s'
            if self.float_wrap_evalf:
                template = 'float(%s)' % template
            elif self.complex_wrap_evalf:
                template = 'complex(%s)' % template

            # Wrapping should only happen on the outermost expression, which
            # is the only thing we know will be a number.
            float_wrap_evalf = self.float_wrap_evalf
            complex_wrap_evalf = self.complex_wrap_evalf
            self.float_wrap_evalf = False
            self.complex_wrap_evalf = False
            ret =  template % (func_name, self.tree2str_translate(argtree))
            self.float_wrap_evalf = float_wrap_evalf
            self.complex_wrap_evalf = complex_wrap_evalf
            return ret

    ##############################################################################
    # The namespace constructors
    ##############################################################################

    @classmethod
    def sympy_expression_namespace(cls, expr):
        """Traverses the (func, args) tree of an expression and creates a SymPy
        namespace. All other modules are imported only as a module name. That way
        the namespace is not polluted and rests quite small. It probably causes much
        more variable lookups and so it takes more time, but there are no tests on
        that for the moment."""
        if expr is None:
            return {}
        else:
            funcname = str(expr.func)
            # XXX Workaround
            # Here we add an ugly workaround because str(func(x))
            # is not always the same as str(func). Eg
            # >>> str(Integral(x))
            # "Integral(x)"
            # >>> str(Integral)
            # "<class 'sympy.integrals.integrals.Integral'>"
            # >>> str(sqrt(x))
            # "sqrt(x)"
            # >>> str(sqrt)
            # "<function sqrt at 0x3d92de8>"
            # >>> str(sin(x))
            # "sin(x)"
            # >>> str(sin)
            # "sin"
            # Either one of those can be used but not all at the same time.
            # The code considers the sin example as the right one.
            regexlist = [
                r'<class \'sympy[\w.]*?.([\w]*)\'>$',
                # the example Integral
                r'<function ([\w]*) at 0x[\w]*>$',    # the example sqrt
            ]
            for r in regexlist:
                m = re.match(r, funcname)
                if m is not None:
                    funcname = m.groups()[0]
            # End of the workaround
            # XXX debug: print funcname
            args_dict = {}
            for a in expr.args:
                if (isinstance(a, (Symbol, NumberSymbol)) or a in [I, zoo, oo]):
                    continue
                else:
                    args_dict.update(cls.sympy_expression_namespace(a))
            args_dict.update({funcname: expr.func})
            return args_dict

    @staticmethod
    def sympy_atoms_namespace(expr):
        """For no real reason this function is separated from
        sympy_expression_namespace. It can be moved to it."""
        atoms = expr.atoms(Symbol, NumberSymbol, I, zoo, oo)
        d = {}
        for a in atoms:
            # XXX debug: print 'atom:' + str(a)
            d[str(a)] = a
        return d
<|endoftext|>
from sympy.external import import_module
import sympy.plotting.backends.base_backend as base_backend


# N.B.
# When changing the minimum module version for matplotlib, please change
# the same in the `SymPyDocTestFinder`` in `sympy/testing/runtests.py`


__doctest_requires__ = {
    ("PlotGrid",): ["matplotlib"],
}


class PlotGrid:
    """This class helps to plot subplots from already created SymPy plots
    in a single figure.

    Examples
    ========

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

        >>> from sympy import symbols
        >>> from sympy.plotting import plot, plot3d, PlotGrid
        >>> x, y = symbols('x, y')
        >>> p1 = plot(x, x**2, x**3, (x, -5, 5))
        >>> p2 = plot((x**2, (x, -6, 6)), (x, (x, -5, 5)))
        >>> p3 = plot(x**3, (x, -5, 5))
        >>> p4 = plot3d(x*y, (x, -5, 5), (y, -5, 5))

    Plotting vertically in a single line:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

        >>> PlotGrid(2, 1, p1, p2)
        PlotGrid object containing:
        Plot[0]:Plot object containing:
        [0]: cartesian line: x for x over (-5.0, 5.0)
        [1]: cartesian line: x**2 for x over (-5.0, 5.0)
        [2]: cartesian line: x**3 for x over (-5.0, 5.0)
        Plot[1]:Plot object containing:
        [0]: cartesian line: x**2 for x over (-6.0, 6.0)
        [1]: cartesian line: x for x over (-5.0, 5.0)

    Plotting horizontally in a single line:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

        >>> PlotGrid(1, 3, p2, p3, p4)
        PlotGrid object containing:
        Plot[0]:Plot object containing:
        [0]: cartesian line: x**2 for x over (-6.0, 6.0)
        [1]: cartesian line: x for x over (-5.0, 5.0)
        Plot[1]:Plot object containing:
        [0]: cartesian line: x**3 for x over (-5.0, 5.0)
        Plot[2]:Plot object containing:
        [0]: cartesian surface: x*y for x over (-5.0, 5.0) and y over (-5.0, 5.0)

    Plotting in a grid form:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

        >>> PlotGrid(2, 2, p1, p2, p3, p4)
        PlotGrid object containing:
        Plot[0]:Plot object containing:
        [0]: cartesian line: x for x over (-5.0, 5.0)
        [1]: cartesian line: x**2 for x over (-5.0, 5.0)
        [2]: cartesian line: x**3 for x over (-5.0, 5.0)
        Plot[1]:Plot object containing:
        [0]: cartesian line: x**2 for x over (-6.0, 6.0)
        [1]: cartesian line: x for x over (-5.0, 5.0)
        Plot[2]:Plot object containing:
        [0]: cartesian line: x**3 for x over (-5.0, 5.0)
        Plot[3]:Plot object containing:
        [0]: cartesian surface: x*y for x over (-5.0, 5.0) and y over (-5.0, 5.0)

    """
    def __init__(self, nrows, ncolumns, *args, show=True, size=None, **kwargs):
        """
        Parameters
        ==========

        nrows :
            The number of rows that should be in the grid of the
            required subplot.
        ncolumns :
            The number of columns that should be in the grid
            of the required subplot.

        nrows and ncolumns together define the required grid.

        Arguments
        =========

        A list of predefined plot objects entered in a row-wise sequence
        i.e. plot objects which are to be in the top row of the required
        grid are written first, then the second row objects and so on

        Keyword arguments
        =================

        show : Boolean
            The default value is set to ``True``. Set show to ``False`` and
            the function will not display the subplot. The returned instance
            of the ``PlotGrid`` class can then be used to save or display the
            plot by calling the ``save()`` and ``show()`` methods
            respectively.
        size : (float, float), optional
            A tuple in the form (width, height) in inches to specify the size of
            the overall figure. The default value is set to ``None``, meaning
            the size will be set by the default backend.
        """
        self.matplotlib = import_module('matplotlib',
            import_kwargs={'fromlist': ['pyplot', 'cm', 'collections']},
            min_module_version='1.1.0', catch=(RuntimeError,))
        self.nrows = nrows
        self.ncolumns = ncolumns
        self._series = []
        self._fig = None
        self.args = args
        for arg in args:
            self._series.append(arg._series)
        self.size = size
        if show and self.matplotlib:
            self.show()

    def _create_figure(self):
        gs = self.matplotlib.gridspec.GridSpec(self.nrows, self.ncolumns)
        mapping = {}
        c = 0
        for i in range(self.nrows):
            for j in range(self.ncolumns):
                if c < len(self.args):
                    mapping[gs[i, j]] = self.args[c]
                c += 1

        kw = {} if not self.size else {"figsize": self.size}
        self._fig = self.matplotlib.pyplot.figure(**kw)
        for spec, p in mapping.items():
            kw = ({"projection": "3d"} if (len(p._series) > 0 and
                p._series[0].is_3D) else {})
            cur_ax = self._fig.add_subplot(spec, **kw)
            p._plotgrid_fig = self._fig
            p._plotgrid_ax = cur_ax
            p.process_series()

    @property
    def fig(self):
        if not self._fig:
            self._create_figure()
        return self._fig

    @property
    def _backend(self):
        return self

    def close(self):
        self.matplotlib.pyplot.close(self.fig)

    def show(self):
        if base_backend._show:
            self.fig.tight_layout()
            self.matplotlib.pyplot.show()
        else:
            self.close()

    def save(self, path):
        self.fig.savefig(path)

    def __str__(self):
        plot_strs = [('Plot[%d]:' % i) + str(plot)
                      for i, plot in enumerate(self.args)]

        return 'PlotGrid object containing:\n' + '\n'.join(plot_strs)
<|endoftext|>"""Implicit plotting module for SymPy.

Explanation
===========

The module implements a data series called ImplicitSeries which is used by
``Plot`` class to plot implicit plots for different backends. The module,
by default, implements plotting using interval arithmetic. It switches to a
fall back algorithm if the expression cannot be plotted using interval arithmetic.
It is also possible to specify to use the fall back algorithm for all plots.

Boolean combinations of expressions cannot be plotted by the fall back
algorithm.

See Also
========

sympy.plotting.plot

References
==========

.. [1] Jeffrey Allen Tupper. Reliable Two-Dimensional Graphing Methods for
Mathematical Formulae with Two Free Variables.

.. [2] Jeffrey Allen Tupper. Graphing Equations with Generalized Interval
Arithmetic. Master's thesis. University of Toronto, 1996

"""


from sympy.core.containers import Tuple
from sympy.core.symbol import (Dummy, Symbol)
from sympy.polys.polyutils import _sort_gens
from sympy.plotting.series import ImplicitSeries, _set_discretization_points
from sympy.plotting.plot import plot_factory
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import flatten


__doctest_requires__ = {'plot_implicit': ['matplotlib']}


@doctest_depends_on(modules=('matplotlib',))
def plot_implicit(expr, x_var=None, y_var=None, adaptive=True, depth=0,
                  n=300, line_color="blue", show=True, **kwargs):
    """A plot function to plot implicit equations / inequalities.

    Arguments
    =========

    - expr : The equation / inequality that is to be plotted.
    - x_var (optional) : symbol to plot on x-axis or tuple giving symbol
      and range as ``(symbol, xmin, xmax)``
    - y_var (optional) : symbol to plot on y-axis or tuple giving symbol
      and range as ``(symbol, ymin, ymax)``

    If neither ``x_var`` nor ``y_var`` are given then the free symbols in the
    expression will be assigned in the order they are sorted.

    The following keyword arguments can also be used:

    - ``adaptive`` Boolean. The default value is set to True. It has to be
        set to False if you want to use a mesh grid.

    - ``depth`` integer. The depth of recursion for adaptive mesh grid.
        Default value is 0. Takes value in the range (0, 4).

    - ``n`` integer. The number of points if adaptive mesh grid is not
        used. Default value is 300. This keyword argument replaces ``points``,
        which should be considered deprecated.

    - ``show`` Boolean. Default value is True. If set to False, the plot will
        not be shown. See ``Plot`` for further information.

    - ``title`` string. The title for the plot.

    - ``xlabel`` string. The label for the x-axis

    - ``ylabel`` string. The label for the y-axis

    Aesthetics options:

    - ``line_color``: float or string. Specifies the color for the plot.
        See ``Plot`` to see how to set color for the plots.
        Default value is "Blue"

    plot_implicit, by default, uses interval arithmetic to plot functions. If
    the expression cannot be plotted using interval arithmetic, it defaults to
    a generating a contour using a mesh grid of fixed number of points. By
    setting adaptive to False, you can force plot_implicit to use the mesh
    grid. The mesh grid method can be effective when adaptive plotting using
    interval arithmetic, fails to plot with small line width.

    Examples
    ========

    Plot expressions:

    .. plot::
        :context: reset
        :format: doctest
        :include-source: True

        >>> from sympy import plot_implicit, symbols, Eq, And
        >>> x, y = symbols('x y')

    Without any ranges for the symbols in the expression:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> p1 = plot_implicit(Eq(x**2 + y**2, 5))

    With the range for the symbols:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> p2 = plot_implicit(
        ...     Eq(x**2 + y**2, 3), (x, -3, 3), (y, -3, 3))

    With depth of recursion as argument:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> p3 = plot_implicit(
        ...     Eq(x**2 + y**2, 5), (x, -4, 4), (y, -4, 4), depth = 2)

    Using mesh grid and not using adaptive meshing:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> p4 = plot_implicit(
        ...     Eq(x**2 + y**2, 5), (x, -5, 5), (y, -2, 2),
        ...     adaptive=False)

    Using mesh grid without using adaptive meshing with number of points
    specified:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> p5 = plot_implicit(
        ...     Eq(x**2 + y**2, 5), (x, -5, 5), (y, -2, 2),
        ...     adaptive=False, n=400)

    Plotting regions:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> p6 = plot_implicit(y > x**2)

    Plotting Using boolean conjunctions:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> p7 = plot_implicit(And(y > x, y > -x))

    When plotting an expression with a single variable (y - 1, for example),
    specify the x or the y variable explicitly:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> p8 = plot_implicit(y - 1, y_var=y)
        >>> p9 = plot_implicit(x - 1, x_var=x)
    """

    xyvar = [i for i in (x_var, y_var) if i is not None]
    free_symbols = expr.free_symbols
    range_symbols = Tuple(*flatten(xyvar)).free_symbols
    undeclared = free_symbols - range_symbols
    if len(free_symbols & range_symbols) > 2:
        raise NotImplementedError("Implicit plotting is not implemented for "
                                  "more than 2 variables")

    #Create default ranges if the range is not provided.
    default_range = Tuple(-5, 5)
    def _range_tuple(s):
        if isinstance(s, Symbol):
            return Tuple(s) + default_range
        if len(s) == 3:
            return Tuple(*s)
        raise ValueError('symbol or `(symbol, min, max)` expected but got %s' % s)

    if len(xyvar) == 0:
        xyvar = list(_sort_gens(free_symbols))
    var_start_end_x = _range_tuple(xyvar[0])
    x = var_start_end_x[0]
    if len(xyvar) != 2:
        if x in undeclared or not undeclared:
            xyvar.append(Dummy('f(%s)' % x.name))
        else:
            xyvar.append(undeclared.pop())
    var_start_end_y = _range_tuple(xyvar[1])

    kwargs = _set_discretization_points(kwargs, ImplicitSeries)
    series_argument = ImplicitSeries(
        expr, var_start_end_x, var_start_end_y,
        adaptive=adaptive, depth=depth,
        n=n, line_color=line_color)

    #set the x and y limits
    kwargs['xlim'] = tuple(float(x) for x in var_start_end_x[1:])
    kwargs['ylim'] = tuple(float(y) for y in var_start_end_y[1:])
    # set the x and y labels
    kwargs.setdefault('xlabel', var_start_end_x[0])
    kwargs.setdefault('ylabel', var_start_end_y[0])
    p = plot_factory(series_argument, **kwargs)
    if show:
        p.show()
    return p
<|endoftext|>"""Plotting module for SymPy.

A plot is represented by the ``Plot`` class that contains a reference to the
backend and a list of the data series to be plotted. The data series are
instances of classes meant to simplify getting points and meshes from SymPy
expressions. ``plot_backends`` is a dictionary with all the backends.

This module gives only the essential. For all the fancy stuff use directly
the backend. You can get the backend wrapper for every plot from the
``_backend`` attribute. Moreover the data series classes have various useful
methods like ``get_points``, ``get_meshes``, etc, that may
be useful if you wish to use another plotting library.

Especially if you need publication ready graphs and this module is not enough
for you - just get the ``_backend`` attribute and add whatever you want
directly to it. In the case of matplotlib (the common way to graph data in
python) just copy ``_backend.fig`` which is the figure and ``_backend.ax``
which is the axis and work on them as you would on any other matplotlib object.

Simplicity of code takes much greater importance than performance. Do not use it
if you care at all about performance. A new backend instance is initialized
every time you call ``show()`` and the old one is left to the garbage collector.
"""

from sympy.concrete.summations import Sum
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function, AppliedUndef
from sympy.core.symbol import (Dummy, Symbol, Wild)
from sympy.external import import_module
from sympy.functions import sign
from sympy.plotting.backends.base_backend import Plot
from sympy.plotting.backends.matplotlibbackend import MatplotlibBackend
from sympy.plotting.backends.textbackend import TextBackend
from sympy.plotting.series import (
    LineOver1DRangeSeries, Parametric2DLineSeries, Parametric3DLineSeries,
    ParametricSurfaceSeries, SurfaceOver2DRangeSeries, ContourSeries)
from sympy.plotting.utils import _check_arguments, _plot_sympify
from sympy.tensor.indexed import Indexed
# to maintain back-compatibility
from sympy.plotting.plotgrid import PlotGrid # noqa: F401
from sympy.plotting.series import BaseSeries # noqa: F401
from sympy.plotting.series import Line2DBaseSeries # noqa: F401
from sympy.plotting.series import Line3DBaseSeries # noqa: F401
from sympy.plotting.series import SurfaceBaseSeries # noqa: F401
from sympy.plotting.series import List2DSeries # noqa: F401
from sympy.plotting.series import GenericDataSeries # noqa: F401
from sympy.plotting.series import centers_of_faces # noqa: F401
from sympy.plotting.series import centers_of_segments # noqa: F401
from sympy.plotting.series import flat # noqa: F401
from sympy.plotting.backends.base_backend import unset_show # noqa: F401
from sympy.plotting.backends.matplotlibbackend import _matplotlib_list # noqa: F401
from sympy.plotting.textplot import textplot # noqa: F401


__doctest_requires__ = {
    ('plot3d',
     'plot3d_parametric_line',
     'plot3d_parametric_surface',
     'plot_parametric'): ['matplotlib'],
    # XXX: The plot doctest possibly should not require matplotlib. It fails at
    # plot(x**2, (x, -5, 5)) which should be fine for text backend.
    ('plot',): ['matplotlib'],
}


def _process_summations(sum_bound, *args):
    """Substitute oo (infinity) in the lower/upper bounds of a summation with
    some integer number.

    Parameters
    ==========

    sum_bound : int
        oo will be substituted with this integer number.
    *args : list/tuple
        pre-processed arguments of the form (expr, range, ...)

    Notes
    =====
    Let's consider the following summation: ``Sum(1 / x**2, (x, 1, oo))``.
    The current implementation of lambdify (SymPy 1.12 at the time of
    writing this) will create something of this form:
    ``sum(1 / x**2 for x in range(1, INF))``
    The problem is that ``type(INF)`` is float, while ``range`` requires
    integers: the evaluation fails.
    Instead of modifying ``lambdify`` (which requires a deep knowledge), just
    replace it with some integer number.
    """
    def new_bound(t, bound):
        if (not t.is_number) or t.is_finite:
            return t
        if sign(t) >= 0:
            return bound
        return -bound

    args = list(args)
    expr = args[0]

    # select summations whose lower/upper bound is infinity
    w = Wild("w", properties=[
        lambda t: isinstance(t, Sum),
        lambda t: any((not a[1].is_finite) or (not a[2].is_finite) for i, a in enumerate(t.args) if i > 0)
    ])

    for t in list(expr.find(w)):
        sums_args = list(t.args)
        for i, a in enumerate(sums_args):
            if i > 0:
                sums_args[i] = (a[0], new_bound(a[1], sum_bound),
                    new_bound(a[2], sum_bound))
        s = Sum(*sums_args)
        expr = expr.subs(t, s)
    args[0] = expr
    return args


def _build_line_series(*args, **kwargs):
    """Loop over the provided arguments and create the necessary line series.
    """
    series = []
    sum_bound = int(kwargs.get("sum_bound", 1000))
    for arg in args:
        expr, r, label, rendering_kw = arg
        kw = kwargs.copy()
        if rendering_kw is not None:
            kw["rendering_kw"] = rendering_kw
        # TODO: _process_piecewise check goes here
        if not callable(expr):
            arg = _process_summations(sum_bound, *arg)
        series.append(LineOver1DRangeSeries(*arg[:-1], **kw))
    return series


def _create_series(series_type, plot_expr, **kwargs):
    """Extract the rendering_kw dictionary from the provided arguments and
    create an appropriate data series.
    """
    series = []
    for args in plot_expr:
        kw = kwargs.copy()
        if args[-1] is not None:
            kw["rendering_kw"] = args[-1]
        series.append(series_type(*args[:-1], **kw))
    return series


def _set_labels(series, labels, rendering_kw):
    """Apply the `label` and `rendering_kw` keyword arguments to the series.
    """
    if not isinstance(labels, (list, tuple)):
        labels = [labels]
    if len(labels) > 0:
        if len(labels) == 1 and len(series) > 1:
            # if one label is provided and multiple series are being plotted,
            # set the same label to all data series. It maintains
            # back-compatibility
            labels *= len(series)
        if len(series) != len(labels):
            raise ValueError("The number of labels must be equal to the "
                "number of expressions being plotted.\nReceived "
                f"{len(series)} expressions and {len(labels)} labels")

        for s, l in zip(series, labels):
            s.label = l

    if rendering_kw:
        if isinstance(rendering_kw, dict):
            rendering_kw = [rendering_kw]
        if len(rendering_kw) == 1:
            rendering_kw *= len(series)
        elif len(series) != len(rendering_kw):
            raise ValueError("The number of rendering dictionaries must be "
                "equal to the number of expressions being plotted.\nReceived "
                f"{len(series)} expressions and {len(labels)} labels")
        for s, r in zip(series, rendering_kw):
            s.rendering_kw = r


def plot_factory(*args, **kwargs):
    backend = kwargs.pop("backend", "default")
    if isinstance(backend, str):
        if backend == "default":
            matplotlib = import_module('matplotlib',
                min_module_version='1.1.0', catch=(RuntimeError,))
            if matplotlib:
                return MatplotlibBackend(*args, **kwargs)
            return TextBackend(*args, **kwargs)
        return plot_backends[backend](*args, **kwargs)
    elif (type(backend) == type) and issubclass(backend, Plot):
        return backend(*args, **kwargs)
    else:
        raise TypeError("backend must be either a string or a subclass of ``Plot``.")


plot_backends = {
    'matplotlib': MatplotlibBackend,
    'text': TextBackend,
}


####New API for plotting module ####

# TODO: Add color arrays for plots.
# TODO: Add more plotting options for 3d plots.
# TODO: Adaptive sampling for 3D plots.

def plot(*args, show=True, **kwargs):
    """Plots a function of a single variable as a curve.

    Parameters
    ==========

    args :
        The first argument is the expression representing the function
        of single variable to be plotted.

        The last argument is a 3-tuple denoting the range of the free
        variable. e.g. ``(x, 0, 5)``

        Typical usage examples are in the following:

        - Plotting a single expression with a single range.
            ``plot(expr, range, **kwargs)``
        - Plotting a single expression with the default range (-10, 10).
            ``plot(expr, **kwargs)``
        - Plotting multiple expressions with a single range.
            ``plot(expr1, expr2, ..., range, **kwargs)``
        - Plotting multiple expressions with multiple ranges.
            ``plot((expr1, range1), (expr2, range2), ..., **kwargs)``

        It is best practice to specify range explicitly because default
        range may change in the future if a more advanced default range
        detection algorithm is implemented.

    show : bool, optional
        The default value is set to ``True``. Set show to ``False`` and
        the function will not display the plot. The returned instance of
        the ``Plot`` class can then be used to save or display the plot
        by calling the ``save()`` and ``show()`` methods respectively.

    line_color : string, or float, or function, optional
        Specifies the color for the plot.
        See ``Plot`` to see how to set color for the plots.
        Note that by setting ``line_color``, it would be applied simultaneously
        to all the series.

    title : str, optional
        Title of the plot. It is set to the latex representation of
        the expression, if the plot has only one expression.

    label : str, optional
        The label of the expression in the plot. It will be used when
        called with ``legend``. Default is the name of the expression.
        e.g. ``sin(x)``

    xlabel : str or expression, optional
        Label for the x-axis.

    ylabel : str or expression, optional
        Label for the y-axis.

    xscale : 'linear' or 'log', optional
        Sets the scaling of the x-axis.

    yscale : 'linear' or 'log', optional
        Sets the scaling of the y-axis.

    axis_center : (float, float), optional
        Tuple of two floats denoting the coordinates of the center or
        {'center', 'auto'}

    xlim : (float, float), optional
        Denotes the x-axis limits, ``(min, max)```.

    ylim : (float, float), optional
        Denotes the y-axis limits, ``(min, max)```.

    annotations : list, optional
        A list of dictionaries specifying the type of annotation
        required. The keys in the dictionary should be equivalent
        to the arguments of the :external:mod:`matplotlib`'s
        :external:meth:`~matplotlib.axes.Axes.annotate` method.

    markers : list, optional
        A list of dictionaries specifying the type the markers required.
        The keys in the dictionary should be equivalent to the arguments
        of the :external:mod:`matplotlib`'s :external:func:`~matplotlib.pyplot.plot()` function
        along with the marker related keyworded arguments.

    rectangles : list, optional
        A list of dictionaries specifying the dimensions of the
        rectangles to be plotted. The keys in the dictionary should be
        equivalent to the arguments of the :external:mod:`matplotlib`'s
        :external:class:`~matplotlib.patches.Rectangle` class.

    fill : dict, optional
        A dictionary specifying the type of color filling required in
        the plot. The keys in the dictionary should be equivalent to the
        arguments of the :external:mod:`matplotlib`'s
        :external:meth:`~matplotlib.axes.Axes.fill_between` method.

    adaptive : bool, optional
        The default value for the ``adaptive`` parameter is now ``False``.
        To enable adaptive sampling, set ``adaptive=True`` and specify ``n`` if uniform sampling is required.

        The plotting uses an adaptive algorithm which samples
        recursively to accurately plot. The adaptive algorithm uses a
        random point near the midpoint of two points that has to be
        further sampled. Hence the same plots can appear slightly
        different.

    depth : int, optional
        Recursion depth of the adaptive algorithm. A depth of value
        `n` samples a maximum of `2^{n}` points.

        If the ``adaptive`` flag is set to ``False``, this will be
        ignored.

    n : int, optional
        Used when the ``adaptive`` is set to ``False``. The function
        is uniformly sampled at ``n`` number of points. If the ``adaptive``
        flag is set to ``True``, this will be ignored.
        This keyword argument replaces ``nb_of_points``, which should be
        considered deprecated.

    size : (float, float), optional
        A tuple in the form (width, height) in inches to specify the size of
        the overall figure. The default value is set to ``None``, meaning
        the size will be set by the default backend.

    Examples
    ========

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> from sympy import symbols
       >>> from sympy.plotting import plot
       >>> x = symbols('x')

    Single Plot

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot(x**2, (x, -5, 5))
       Plot object containing:
       [0]: cartesian line: x**2 for x over (-5.0, 5.0)

    Multiple plots with single range.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot(x, x**2, x**3, (x, -5, 5))
       Plot object containing:
       [0]: cartesian line: x for x over (-5.0, 5.0)
       [1]: cartesian line: x**2 for x over (-5.0, 5.0)
       [2]: cartesian line: x**3 for x over (-5.0, 5.0)

    Multiple plots with different ranges.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot((x**2, (x, -6, 6)), (x, (x, -5, 5)))
       Plot object containing:
       [0]: cartesian line: x**2 for x over (-6.0, 6.0)
       [1]: cartesian line: x for x over (-5.0, 5.0)

    No adaptive sampling by default. If adaptive sampling is required, set ``adaptive=True``.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot(x**2, adaptive=True, n=400)
       Plot object containing:
       [0]: cartesian line: x**2 for x over (-10.0, 10.0)

    See Also
    ========

    Plot, LineOver1DRangeSeries

    """
    args = _plot_sympify(args)
    plot_expr = _check_arguments(args, 1, 1, **kwargs)
    params = kwargs.get("params", None)
    free = set()
    for p in plot_expr:
        if not isinstance(p[1][0], str):
            free |= {p[1][0]}
        else:
            free |= {Symbol(p[1][0])}
    if params:
        free = free.difference(params.keys())
    x = free.pop() if free else Symbol("x")
    kwargs.setdefault('xlabel', x)
    kwargs.setdefault('ylabel', Function('f')(x))

    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
    series = _build_line_series(*plot_expr, **kwargs)
    _set_labels(series, labels, rendering_kw)

    plots = plot_factory(*series, **kwargs)
    if show:
        plots.show()
    return plots


def plot_parametric(*args, show=True, **kwargs):
    """
    Plots a 2D parametric curve.

    Parameters
    ==========

    args
        Common specifications are:

        - Plotting a single parametric curve with a range
            ``plot_parametric((expr_x, expr_y), range)``
        - Plotting multiple parametric curves with the same range
            ``plot_parametric((expr_x, expr_y), ..., range)``
        - Plotting multiple parametric curves with different ranges
            ``plot_parametric((expr_x, expr_y, range), ...)``

        ``expr_x`` is the expression representing $x$ component of the
        parametric function.

        ``expr_y`` is the expression representing $y$ component of the
        parametric function.

        ``range`` is a 3-tuple denoting the parameter symbol, start and
        stop. For example, ``(u, 0, 5)``.

        If the range is not specified, then a default range of (-10, 10)
        is used.

        However, if the arguments are specified as
        ``(expr_x, expr_y, range), ...``, you must specify the ranges
        for each expressions manually.

        Default range may change in the future if a more advanced
        algorithm is implemented.

    adaptive : bool, optional
        Specifies whether to use the adaptive sampling or not.

        The default value is set to ``True``. Set adaptive to ``False``
        and specify ``n`` if uniform sampling is required.

    depth :  int, optional
        The recursion depth of the adaptive algorithm. A depth of
        value $n$ samples a maximum of $2^n$ points.

    n : int, optional
        Used when the ``adaptive`` flag is set to ``False``. Specifies the
        number of the points used for the uniform sampling.
        This keyword argument replaces ``nb_of_points``, which should be
        considered deprecated.

    line_color : string, or float, or function, optional
        Specifies the color for the plot.
        See ``Plot`` to see how to set color for the plots.
        Note that by setting ``line_color``, it would be applied simultaneously
        to all the series.

    label : str, optional
        The label of the expression in the plot. It will be used when
        called with ``legend``. Default is the name of the expression.
        e.g. ``sin(x)``

    xlabel : str, optional
        Label for the x-axis.

    ylabel : str, optional
        Label for the y-axis.

    xscale : 'linear' or 'log', optional
        Sets the scaling of the x-axis.

    yscale : 'linear' or 'log', optional
        Sets the scaling of the y-axis.

    axis_center : (float, float), optional
        Tuple of two floats denoting the coordinates of the center or
        {'center', 'auto'}

    xlim : (float, float), optional
        Denotes the x-axis limits, ``(min, max)```.

    ylim : (float, float), optional
        Denotes the y-axis limits, ``(min, max)```.

    size : (float, float), optional
        A tuple in the form (width, height) in inches to specify the size of
        the overall figure. The default value is set to ``None``, meaning
        the size will be set by the default backend.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import plot_parametric, symbols, cos, sin
       >>> u = symbols('u')

    A parametric plot with a single expression:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_parametric((cos(u), sin(u)), (u, -5, 5))
       Plot object containing:
       [0]: parametric cartesian line: (cos(u), sin(u)) for u over (-5.0, 5.0)

    A parametric plot with multiple expressions with the same range:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_parametric((cos(u), sin(u)), (u, cos(u)), (u, -10, 10))
       Plot object containing:
       [0]: parametric cartesian line: (cos(u), sin(u)) for u over (-10.0, 10.0)
       [1]: parametric cartesian line: (u, cos(u)) for u over (-10.0, 10.0)

    A parametric plot with multiple expressions with different ranges
    for each curve:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot_parametric((cos(u), sin(u), (u, -5, 5)),
       ...     (cos(u), u, (u, -5, 5)))
       Plot object containing:
       [0]: parametric cartesian line: (cos(u), sin(u)) for u over (-5.0, 5.0)
       [1]: parametric cartesian line: (cos(u), u) for u over (-5.0, 5.0)

    Notes
    =====

    The plotting uses an adaptive algorithm which samples recursively to
    accurately plot the curve. The adaptive algorithm uses a random point
    near the midpoint of two points that has to be further sampled.
    Hence, repeating the same plot command can give slightly different
    results because of the random sampling.

    If there are multiple plots, then the same optional arguments are
    applied to all the plots drawn in the same canvas. If you want to
    set these options separately, you can index the returned ``Plot``
    object and set it.

    For example, when you specify ``line_color`` once, it would be
    applied simultaneously to both series.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

        >>> from sympy import pi
        >>> expr1 = (u, cos(2*pi*u)/2 + 1/2)
        >>> expr2 = (u, sin(2*pi*u)/2 + 1/2)
        >>> p = plot_parametric(expr1, expr2, (u, 0, 1), line_color='blue')

    If you want to specify the line color for the specific series, you
    should index each item and apply the property manually.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

        >>> p[0].line_color = 'red'
        >>> p.show()

    See Also
    ========

    Plot, Parametric2DLineSeries
    """
    args = _plot_sympify(args)
    plot_expr = _check_arguments(args, 2, 1, **kwargs)

    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
    series = _create_series(Parametric2DLineSeries, plot_expr, **kwargs)
    _set_labels(series, labels, rendering_kw)

    plots = plot_factory(*series, **kwargs)
    if show:
        plots.show()
    return plots


def plot3d_parametric_line(*args, show=True, **kwargs):
    """
    Plots a 3D parametric line plot.

    Usage
    =====

    Single plot:

    ``plot3d_parametric_line(expr_x, expr_y, expr_z, range, **kwargs)``

    If the range is not specified, then a default range of (-10, 10) is used.

    Multiple plots.

    ``plot3d_parametric_line((expr_x, expr_y, expr_z, range), ..., **kwargs)``

    Ranges have to be specified for every expression.

    Default range may change in the future if a more advanced default range
    detection algorithm is implemented.

    Arguments
    =========

    expr_x : Expression representing the function along x.

    expr_y : Expression representing the function along y.

    expr_z : Expression representing the function along z.

    range : (:class:`~.Symbol`, float, float)
        A 3-tuple denoting the range of the parameter variable, e.g., (u, 0, 5).

    Keyword Arguments
    =================

    Arguments for ``Parametric3DLineSeries`` class.

    n : int
        The range is uniformly sampled at ``n`` number of points.
        This keyword argument replaces ``nb_of_points``, which should be
        considered deprecated.

    Aesthetics:

    line_color : string, or float, or function, optional
        Specifies the color for the plot.
        See ``Plot`` to see how to set color for the plots.
        Note that by setting ``line_color``, it would be applied simultaneously
        to all the series.

    label : str
        The label to the plot. It will be used when called with ``legend=True``
        to denote the function with the given label in the plot.

    If there are multiple plots, then the same series arguments are applied to
    all the plots. If you want to set these options separately, you can index
    the returned ``Plot`` object and set it.

    Arguments for ``Plot`` class.

    title : str
        Title of the plot.

    size : (float, float), optional
        A tuple in the form (width, height) in inches to specify the size of
        the overall figure. The default value is set to ``None``, meaning
        the size will be set by the default backend.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, sin
       >>> from sympy.plotting import plot3d_parametric_line
       >>> u = symbols('u')

    Single plot.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d_parametric_line(cos(u), sin(u), u, (u, -5, 5))
       Plot object containing:
       [0]: 3D parametric cartesian line: (cos(u), sin(u), u) for u over (-5.0, 5.0)


    Multiple plots.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d_parametric_line((cos(u), sin(u), u, (u, -5, 5)),
       ...     (sin(u), u**2, u, (u, -5, 5)))
       Plot object containing:
       [0]: 3D parametric cartesian line: (cos(u), sin(u), u) for u over (-5.0, 5.0)
       [1]: 3D parametric cartesian line: (sin(u), u**2, u) for u over (-5.0, 5.0)


    See Also
    ========

    Plot, Parametric3DLineSeries

    """
    args = _plot_sympify(args)
    plot_expr = _check_arguments(args, 3, 1, **kwargs)
    kwargs.setdefault("xlabel", "x")
    kwargs.setdefault("ylabel", "y")
    kwargs.setdefault("zlabel", "z")

    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
    series = _create_series(Parametric3DLineSeries, plot_expr, **kwargs)
    _set_labels(series, labels, rendering_kw)

    plots = plot_factory(*series, **kwargs)
    if show:
        plots.show()
    return plots


def _plot3d_plot_contour_helper(Series, *args, **kwargs):
    """plot3d and plot_contour are structurally identical. Let's reduce
    code repetition.
    """
    # NOTE: if this import would be at the top-module level, it would trigger
    # SymPy's optional-dependencies tests to fail.
    from sympy.vector import BaseScalar

    args = _plot_sympify(args)
    plot_expr = _check_arguments(args, 1, 2, **kwargs)

    free_x = set()
    free_y = set()
    _types = (Symbol, BaseScalar, Indexed, AppliedUndef)
    for p in plot_expr:
        free_x |= {p[1][0]} if isinstance(p[1][0], _types) else {Symbol(p[1][0])}
        free_y |= {p[2][0]} if isinstance(p[2][0], _types) else {Symbol(p[2][0])}
    x = free_x.pop() if free_x else Symbol("x")
    y = free_y.pop() if free_y else Symbol("y")
    kwargs.setdefault("xlabel", x)
    kwargs.setdefault("ylabel", y)
    kwargs.setdefault("zlabel", Function('f')(x, y))

    # if a polar discretization is requested and automatic labelling has ben
    # applied, hide the labels on the x-y axis.
    if kwargs.get("is_polar", False):
        if callable(kwargs["xlabel"]):
            kwargs["xlabel"] = ""
        if callable(kwargs["ylabel"]):
            kwargs["ylabel"] = ""

    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
    series = _create_series(Series, plot_expr, **kwargs)
    _set_labels(series, labels, rendering_kw)
    plots = plot_factory(*series, **kwargs)
    if kwargs.get("show", True):
        plots.show()
    return plots


def plot3d(*args, show=True, **kwargs):
    """
    Plots a 3D surface plot.

    Usage
    =====

    Single plot

    ``plot3d(expr, range_x, range_y, **kwargs)``

    If the ranges are not specified, then a default range of (-10, 10) is used.

    Multiple plot with the same range.

    ``plot3d(expr1, expr2, range_x, range_y, **kwargs)``

    If the ranges are not specified, then a default range of (-10, 10) is used.

    Multiple plots with different ranges.

    ``plot3d((expr1, range_x, range_y), (expr2, range_x, range_y), ..., **kwargs)``

    Ranges have to be specified for every expression.

    Default range may change in the future if a more advanced default range
    detection algorithm is implemented.

    Arguments
    =========

    expr : Expression representing the function along x.

    range_x : (:class:`~.Symbol`, float, float)
        A 3-tuple denoting the range of the x variable, e.g. (x, 0, 5).

    range_y : (:class:`~.Symbol`, float, float)
        A 3-tuple denoting the range of the y variable, e.g. (y, 0, 5).

    Keyword Arguments
    =================

    Arguments for ``SurfaceOver2DRangeSeries`` class:

    n1 : int
        The x range is sampled uniformly at ``n1`` of points.
        This keyword argument replaces ``nb_of_points_x``, which should be
        considered deprecated.

    n2 : int
        The y range is sampled uniformly at ``n2`` of points.
        This keyword argument replaces ``nb_of_points_y``, which should be
        considered deprecated.

    Aesthetics:

    surface_color : Function which returns a float
        Specifies the color for the surface of the plot.
        See :class:`~.Plot` for more details.

    If there are multiple plots, then the same series arguments are applied to
    all the plots. If you want to set these options separately, you can index
    the returned ``Plot`` object and set it.

    Arguments for ``Plot`` class:

    title : str
        Title of the plot.

    size : (float, float), optional
        A tuple in the form (width, height) in inches to specify the size of the
        overall figure. The default value is set to ``None``, meaning the size will
        be set by the default backend.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols
       >>> from sympy.plotting import plot3d
       >>> x, y = symbols('x y')

    Single plot

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d(x*y, (x, -5, 5), (y, -5, 5))
       Plot object containing:
       [0]: cartesian surface: x*y for x over (-5.0, 5.0) and y over (-5.0, 5.0)


    Multiple plots with same range

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d(x*y, -x*y, (x, -5, 5), (y, -5, 5))
       Plot object containing:
       [0]: cartesian surface: x*y for x over (-5.0, 5.0) and y over (-5.0, 5.0)
       [1]: cartesian surface: -x*y for x over (-5.0, 5.0) and y over (-5.0, 5.0)


    Multiple plots with different ranges.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d((x**2 + y**2, (x, -5, 5), (y, -5, 5)),
       ...     (x*y, (x, -3, 3), (y, -3, 3)))
       Plot object containing:
       [0]: cartesian surface: x**2 + y**2 for x over (-5.0, 5.0) and y over (-5.0, 5.0)
       [1]: cartesian surface: x*y for x over (-3.0, 3.0) and y over (-3.0, 3.0)


    See Also
    ========

    Plot, SurfaceOver2DRangeSeries

    """
    kwargs.setdefault("show", show)
    return _plot3d_plot_contour_helper(
        SurfaceOver2DRangeSeries, *args, **kwargs)


def plot3d_parametric_surface(*args, show=True, **kwargs):
    """
    Plots a 3D parametric surface plot.

    Explanation
    ===========

    Single plot.

    ``plot3d_parametric_surface(expr_x, expr_y, expr_z, range_u, range_v, **kwargs)``

    If the ranges is not specified, then a default range of (-10, 10) is used.

    Multiple plots.

    ``plot3d_parametric_surface((expr_x, expr_y, expr_z, range_u, range_v), ..., **kwargs)``

    Ranges have to be specified for every expression.

    Default range may change in the future if a more advanced default range
    detection algorithm is implemented.

    Arguments
    =========

    expr_x : Expression representing the function along ``x``.

    expr_y : Expression representing the function along ``y``.

    expr_z : Expression representing the function along ``z``.

    range_u : (:class:`~.Symbol`, float, float)
        A 3-tuple denoting the range of the u variable, e.g. (u, 0, 5).

    range_v : (:class:`~.Symbol`, float, float)
        A 3-tuple denoting the range of the v variable, e.g. (v, 0, 5).

    Keyword Arguments
    =================

    Arguments for ``ParametricSurfaceSeries`` class:

    n1 : int
        The ``u`` range is sampled uniformly at ``n1`` of points.
        This keyword argument replaces ``nb_of_points_u``, which should be
        considered deprecated.

    n2 : int
        The ``v`` range is sampled uniformly at ``n2`` of points.
        This keyword argument replaces ``nb_of_points_v``, which should be
        considered deprecated.

    Aesthetics:

    surface_color : Function which returns a float
        Specifies the color for the surface of the plot. See
        :class:`~Plot` for more details.

    If there are multiple plots, then the same series arguments are applied for
    all the plots. If you want to set these options separately, you can index
    the returned ``Plot`` object and set it.


    Arguments for ``Plot`` class:

    title : str
        Title of the plot.

    size : (float, float), optional
        A tuple in the form (width, height) in inches to specify the size of the
        overall figure. The default value is set to ``None``, meaning the size will
        be set by the default backend.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, sin
       >>> from sympy.plotting import plot3d_parametric_surface
       >>> u, v = symbols('u v')

    Single plot.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d_parametric_surface(cos(u + v), sin(u - v), u - v,
       ...     (u, -5, 5), (v, -5, 5))
       Plot object containing:
       [0]: parametric cartesian surface: (cos(u + v), sin(u - v), u - v) for u over (-5.0, 5.0) and v over (-5.0, 5.0)


    See Also
    ========

    Plot, ParametricSurfaceSeries

    """

    args = _plot_sympify(args)
    plot_expr = _check_arguments(args, 3, 2, **kwargs)
    kwargs.setdefault("xlabel", "x")
    kwargs.setdefault("ylabel", "y")
    kwargs.setdefault("zlabel", "z")

    labels = kwargs.pop("label", [])
    rendering_kw = kwargs.pop("rendering_kw", None)
    series = _create_series(ParametricSurfaceSeries, plot_expr, **kwargs)
    _set_labels(series, labels, rendering_kw)

    plots = plot_factory(*series, **kwargs)
    if show:
        plots.show()
    return plots

def plot_contour(*args, show=True, **kwargs):
    """
    Draws contour plot of a function

    Usage
    =====

    Single plot

    ``plot_contour(expr, range_x, range_y, **kwargs)``

    If the ranges are not specified, then a default range of (-10, 10) is used.

    Multiple plot with the same range.

    ``plot_contour(expr1, expr2, range_x, range_y, **kwargs)``

    If the ranges are not specified, then a default range of (-10, 10) is used.

    Multiple plots with different ranges.

    ``plot_contour((expr1, range_x, range_y), (expr2, range_x, range_y), ..., **kwargs)``

    Ranges have to be specified for every expression.

    Default range may change in the future if a more advanced default range
    detection algorithm is implemented.

    Arguments
    =========

    expr : Expression representing the function along x.

    range_x : (:class:`Symbol`, float, float)
        A 3-tuple denoting the range of the x variable, e.g. (x, 0, 5).

    range_y : (:class:`Symbol`, float, float)
        A 3-tuple denoting the range of the y variable, e.g. (y, 0, 5).

    Keyword Arguments
    =================

    Arguments for ``ContourSeries`` class:

    n1 : int
        The x range is sampled uniformly at ``n1`` of points.
        This keyword argument replaces ``nb_of_points_x``, which should be
        considered deprecated.

    n2 : int
        The y range is sampled uniformly at ``n2`` of points.
        This keyword argument replaces ``nb_of_points_y``, which should be
        considered deprecated.

    Aesthetics:

    surface_color : Function which returns a float
        Specifies the color for the surface of the plot. See
        :class:`sympy.plotting.Plot` for more details.

    If there are multiple plots, then the same series arguments are applied to
    all the plots. If you want to set these options separately, you can index
    the returned ``Plot`` object and set it.

    Arguments for ``Plot`` class:

    title : str
        Title of the plot.

    size : (float, float), optional
        A tuple in the form (width, height) in inches to specify the size of
        the overall figure. The default value is set to ``None``, meaning
        the size will be set by the default backend.

    See Also
    ========

    Plot, ContourSeries

    """
    kwargs.setdefault("show", show)
    return _plot3d_plot_contour_helper(ContourSeries, *args, **kwargs)


def check_arguments(args, expr_len, nb_of_free_symbols):
    """
    Checks the arguments and converts into tuples of the
    form (exprs, ranges).

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import cos, sin, symbols
       >>> from sympy.plotting.plot import check_arguments
       >>> x = symbols('x')
       >>> check_arguments([cos(x), sin(x)], 2, 1)
           [(cos(x), sin(x), (x, -10, 10))]

       >>> check_arguments([x, x**2], 1, 1)
           [(x, (x, -10, 10)), (x**2, (x, -10, 10))]
    """
    if not args:
        return []
    if expr_len > 1 and isinstance(args[0], Expr):
        # Multiple expressions same range.
        # The arguments are tuples when the expression length is
        # greater than 1.
        if len(args) < expr_len:
            raise ValueError("len(args) should not be less than expr_len")
        for i in range(len(args)):
            if isinstance(args[i], Tuple):
                break
        else:
            i = len(args) + 1

        exprs = Tuple(*args[:i])
        free_symbols = list(set().union(*[e.free_symbols for e in exprs]))
        if len(args) == expr_len + nb_of_free_symbols:
            #Ranges given
            plots = [exprs + Tuple(*args[expr_len:])]
        else:
            default_range = Tuple(-10, 10)
            ranges = []
            for symbol in free_symbols:
                ranges.append(Tuple(symbol) + default_range)

            for i in range(len(free_symbols) - nb_of_free_symbols):
                ranges.append(Tuple(Dummy()) + default_range)
            plots = [exprs + Tuple(*ranges)]
        return plots

    if isinstance(args[0], Expr) or (isinstance(args[0], Tuple) and
                                     len(args[0]) == expr_len and
                                     expr_len != 3):
        # Cannot handle expressions with number of expression = 3. It is
        # not possible to differentiate between expressions and ranges.
        #Series of plots with same range
        for i in range(len(args)):
            if isinstance(args[i], Tuple) and len(args[i]) != expr_len:
                break
            if not isinstance(args[i], Tuple):
                args[i] = Tuple(args[i])
        else:
            i = len(args) + 1

        exprs = args[:i]
        assert all(isinstance(e, Expr) for expr in exprs for e in expr)
        free_symbols = list(set().union(*[e.free_symbols for expr in exprs
                                        for e in expr]))

        if len(free_symbols) > nb_of_free_symbols:
            raise ValueError("The number of free_symbols in the expression "
                             "is greater than %d" % nb_of_free_symbols)
        if len(args) == i + nb_of_free_symbols and isinstance(args[i], Tuple):
            ranges = Tuple(*list(args[
                           i:i + nb_of_free_symbols]))
            plots = [expr + ranges for expr in exprs]
            return plots
        else:
            # Use default ranges.
            default_range = Tuple(-10, 10)
            ranges = []
            for symbol in free_symbols:
                ranges.append(Tuple(symbol) + default_range)

            for i in range(nb_of_free_symbols - len(free_symbols)):
                ranges.append(Tuple(Dummy()) + default_range)
            ranges = Tuple(*ranges)
            plots = [expr + ranges for expr in exprs]
            return plots

    elif isinstance(args[0], Tuple) and len(args[0]) == expr_len + nb_of_free_symbols:
        # Multiple plots with different ranges.
        for arg in args:
            for i in range(expr_len):
                if not isinstance(arg[i], Expr):
                    raise ValueError("Expected an expression, given %s" %
                                     str(arg[i]))
            for i in range(nb_of_free_symbols):
                if not len(arg[i + expr_len]) == 3:
                    raise ValueError("The ranges should be a tuple of "
                                     "length 3, got %s" % str(arg[i + expr_len]))
        return args
<|endoftext|>from sympy.core.containers import Tuple
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef
from sympy.core.relational import Relational
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.logic.boolalg import BooleanFunction
from sympy.sets.fancysets import ImageSet
from sympy.sets.sets import FiniteSet
from sympy.tensor.indexed import Indexed


def _get_free_symbols(exprs):
    """Returns the free symbols of a symbolic expression.

    If the expression contains any of these elements, assume that they are
    the "free symbols" of the expression:

    * indexed objects
    * applied undefined function (useful for sympy.physics.mechanics module)
    """
    if not isinstance(exprs, (list, tuple, set)):
        exprs = [exprs]
    if all(callable(e) for e in exprs):
        return set()

    free = set().union(*[e.atoms(Indexed) for e in exprs])
    free = free.union(*[e.atoms(AppliedUndef) for e in exprs])
    return free or set().union(*[e.free_symbols for e in exprs])


def extract_solution(set_sol, n=10):
    """Extract numerical solutions from a set solution (computed by solveset,
    linsolve, nonlinsolve). Often, it is not trivial do get something useful
    out of them.

    Parameters
    ==========

    n : int, optional
        In order to replace ImageSet with FiniteSet, an iterator is created
        for each ImageSet contained in `set_sol`, starting from 0 up to `n`.
        Default value: 10.
    """
    images = set_sol.find(ImageSet)
    for im in images:
        it = iter(im)
        s = FiniteSet(*[next(it) for n in range(0, n)])
        set_sol = set_sol.subs(im, s)
    return set_sol


def _plot_sympify(args):
    """This function recursively loop over the arguments passed to the plot
    functions: the sympify function will be applied to all arguments except
    those of type string/dict.

    Generally, users can provide the following arguments to a plot function:

    expr, range1 [tuple, opt], ..., label [str, opt], rendering_kw [dict, opt]

    `expr, range1, ...` can be sympified, whereas `label, rendering_kw` can't.
    In particular, whenever a special character like $, {, }, ... is used in
    the `label`, sympify will raise an error.
    """
    if isinstance(args, Expr):
        return args

    args = list(args)
    for i, a in enumerate(args):
        if isinstance(a, (list, tuple)):
            args[i] = Tuple(*_plot_sympify(a), sympify=False)
        elif not (isinstance(a, (str, dict)) or callable(a)
            # NOTE: check if it is a vector from sympy.physics.vector module
            # without importing the module (because it slows down SymPy's
            # import process and triggers SymPy's optional-dependencies
            # tests to fail).
            or ((a.__class__.__name__ == "Vector") and not isinstance(a, Basic))
        ):
            args[i] = sympify(a)
    return args


def _create_ranges(exprs, ranges, npar, label="", params=None):
    """This function does two things:

    1. Check if the number of free symbols is in agreement with the type of
       plot chosen. For example, plot() requires 1 free symbol;
       plot3d() requires 2 free symbols.
    2. Sometime users create plots without providing ranges for the variables.
       Here we create the necessary ranges.

    Parameters
    ==========

    exprs : iterable
        The expressions from which to extract the free symbols
    ranges : iterable
        The limiting ranges provided by the user
    npar : int
        The number of free symbols required by the plot functions.
        For example,
        npar=1 for plot, npar=2 for plot3d, ...
    params : dict
        A dictionary mapping symbols to parameters for interactive plot.
    """
    get_default_range = lambda symbol: Tuple(symbol, -10, 10)

    free_symbols = _get_free_symbols(exprs)
    if params is not None:
        free_symbols = free_symbols.difference(params.keys())

    if len(free_symbols) > npar:
        raise ValueError(
            "Too many free symbols.\n"
            + "Expected {} free symbols.\n".format(npar)
            + "Received {}: {}".format(len(free_symbols), free_symbols)
        )

    if len(ranges) > npar:
        raise ValueError(
            "Too many ranges. Received %s, expected %s" % (len(ranges), npar))

    # free symbols in the ranges provided by the user
    rfs = set().union([r[0] for r in ranges])
    if len(rfs) != len(ranges):
        raise ValueError("Multiple ranges with the same symbol")

    if len(ranges) < npar:
        symbols = free_symbols.difference(rfs)
        if symbols != set():
            # add a range for each missing free symbols
            for s in symbols:
                ranges.append(get_default_range(s))
        # if there is still room, fill them with dummys
        for i in range(npar - len(ranges)):
            ranges.append(get_default_range(Dummy()))

    if len(free_symbols) == npar:
        # there could be times when this condition is not met, for example
        # plotting the function f(x, y) = x (which is a plane); in this case,
        # free_symbols = {x} whereas rfs = {x, y} (or x and Dummy)
        rfs = set().union([r[0] for r in ranges])
        if len(free_symbols.difference(rfs)) > 0:
            raise ValueError(
                "Incompatible free symbols of the expressions with "
                "the ranges.\n"
                + "Free symbols in the expressions: {}\n".format(free_symbols)
                + "Free symbols in the ranges: {}".format(rfs)
            )
    return ranges


def _is_range(r):
    """A range is defined as (symbol, start, end). start and end should
    be numbers.
    """
    # TODO: prange check goes here
    return (
        isinstance(r, Tuple)
        and (len(r) == 3)
        and (not isinstance(r.args[1], str)) and r.args[1].is_number
        and (not isinstance(r.args[2], str)) and r.args[2].is_number
    )


def _unpack_args(*args):
    """Given a list/tuple of arguments previously processed by _plot_sympify()
    and/or _check_arguments(), separates and returns its components:
    expressions, ranges, label and rendering keywords.

    Examples
    ========

    >>> from sympy import cos, sin, symbols
    >>> from sympy.plotting.utils import _plot_sympify, _unpack_args
    >>> x, y = symbols('x, y')
    >>> args = (sin(x), (x, -10, 10), "f1")
    >>> args = _plot_sympify(args)
    >>> _unpack_args(*args)
    ([sin(x)], [(x, -10, 10)], 'f1', None)

    >>> args = (sin(x**2 + y**2), (x, -2, 2), (y, -3, 3), "f2")
    >>> args = _plot_sympify(args)
    >>> _unpack_args(*args)
    ([sin(x**2 + y**2)], [(x, -2, 2), (y, -3, 3)], 'f2', None)

    >>> args = (sin(x + y), cos(x - y), x + y, (x, -2, 2), (y, -3, 3), "f3")
    >>> args = _plot_sympify(args)
    >>> _unpack_args(*args)
    ([sin(x + y), cos(x - y), x + y], [(x, -2, 2), (y, -3, 3)], 'f3', None)
    """
    ranges = [t for t in args if _is_range(t)]
    labels = [t for t in args if isinstance(t, str)]
    label = None if not labels else labels[0]
    rendering_kw = [t for t in args if isinstance(t, dict)]
    rendering_kw = None if not rendering_kw else rendering_kw[0]
    # NOTE: why None? because args might have been preprocessed by
    # _check_arguments, so None might represent the rendering_kw
    results = [not (_is_range(a) or isinstance(a, (str, dict)) or (a is None)) for a in args]
    exprs = [a for a, b in zip(args, results) if b]
    return exprs, ranges, label, rendering_kw


def _check_arguments(args, nexpr, npar, **kwargs):
    """Checks the arguments and converts into tuples of the
    form (exprs, ranges, label, rendering_kw).

    Parameters
    ==========

    args
        The arguments provided to the plot functions
    nexpr
        The number of sub-expression forming an expression to be plotted.
        For example:
        nexpr=1 for plot.
        nexpr=2 for plot_parametric: a curve is represented by a tuple of two
            elements.
        nexpr=1 for plot3d.
        nexpr=3 for plot3d_parametric_line: a curve is represented by a tuple
            of three elements.
    npar
        The number of free symbols required by the plot functions. For example,
        npar=1 for plot, npar=2 for plot3d, ...
    **kwargs :
        keyword arguments passed to the plotting function. It will be used to
        verify if ``params`` has ben provided.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import cos, sin, symbols
       >>> from sympy.plotting.plot import _check_arguments
       >>> x = symbols('x')
       >>> _check_arguments([cos(x), sin(x)], 2, 1)
       [(cos(x), sin(x), (x, -10, 10), None, None)]

       >>> _check_arguments([cos(x), sin(x), "test"], 2, 1)
       [(cos(x), sin(x), (x, -10, 10), 'test', None)]

       >>> _check_arguments([cos(x), sin(x), "test", {"a": 0, "b": 1}], 2, 1)
       [(cos(x), sin(x), (x, -10, 10), 'test', {'a': 0, 'b': 1})]

       >>> _check_arguments([x, x**2], 1, 1)
       [(x, (x, -10, 10), None, None), (x**2, (x, -10, 10), None, None)]
    """
    if not args:
        return []
    output = []
    params = kwargs.get("params", None)

    if all(isinstance(a, (Expr, Relational, BooleanFunction)) for a in args[:nexpr]):
        # In this case, with a single plot command, we are plotting either:
        #   1. one expression
        #   2. multiple expressions over the same range

        exprs, ranges, label, rendering_kw = _unpack_args(*args)
        free_symbols = set().union(*[e.free_symbols for e in exprs])
        ranges = _create_ranges(exprs, ranges, npar, label, params)

        if nexpr > 1:
            # in case of plot_parametric or plot3d_parametric_line, there will
            # be 2 or 3 expressions defining a curve. Group them together.
            if len(exprs) == nexpr:
                exprs = (tuple(exprs),)
        for expr in exprs:
            # need this if-else to deal with both plot/plot3d and
            # plot_parametric/plot3d_parametric_line
            is_expr = isinstance(expr, (Expr, Relational, BooleanFunction))
            e = (expr,) if is_expr else expr
            output.append((*e, *ranges, label, rendering_kw))

    else:
        # In this case, we are plotting multiple expressions, each one with its
        # range. Each "expression" to be plotted has the following form:
        # (expr, range, label) where label is optional

        _, ranges, labels, rendering_kw = _unpack_args(*args)
        labels = [labels] if labels else []

        # number of expressions
        n = (len(ranges) + len(labels) +
            (len(rendering_kw) if rendering_kw is not None else 0))
        new_args = args[:-n] if n > 0 else args

        # at this point, new_args might just be [expr]. But I need it to be
        # [[expr]] in order to be able to loop over
        # [expr, range [opt], label [opt]]
        if not isinstance(new_args[0], (list, tuple, Tuple)):
            new_args = [new_args]

        # Each arg has the form (expr1, expr2, ..., range1 [optional], ...,
        #   label [optional], rendering_kw [optional])
        for arg in new_args:
            # look for "local" range and label. If there is not, use "global".
            l = [a for a in arg if isinstance(a, str)]
            if not l:
                l = labels
            r = [a for a in arg if _is_range(a)]
            if not r:
                r = ranges.copy()
            rend_kw = [a for a in arg if isinstance(a, dict)]
            rend_kw = rendering_kw if len(rend_kw) == 0 else rend_kw[0]

            # NOTE: arg = arg[:nexpr] may raise an exception if lambda
            # functions are used. Execute the following instead:
            arg = [arg[i] for i in range(nexpr)]
            free_symbols = set()
            if all(not callable(a) for a in arg):
                free_symbols = free_symbols.union(*[a.free_symbols for a in arg])
            if len(r) != npar:
                r = _create_ranges(arg, r, npar, "", params)

            label = None if not l else l[0]
            output.append((*arg, *r, label, rend_kw))
    return output
<|endoftext|>### The base class for all series
from collections.abc import Callable
from sympy.calculus.util import continuous_domain
from sympy.concrete import Sum, Product
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import arity
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol
from sympy.functions import atan2, zeta, frac, ceiling, floor, im
from sympy.core.relational import (Equality, GreaterThan,
    LessThan, Relational, Ne)
from sympy.core.sympify import sympify
from sympy.external import import_module
from sympy.logic.boolalg import BooleanFunction
from sympy.plotting.utils import _get_free_symbols, extract_solution
from sympy.printing.latex import latex
from sympy.printing.pycode import PythonCodePrinter
from sympy.printing.precedence import precedence
from sympy.sets.sets import Set, Interval, Union
from sympy.simplify.simplify import nsimplify
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.lambdify import lambdify
from .intervalmath import interval
import warnings


class IntervalMathPrinter(PythonCodePrinter):
    """A printer to be used inside `plot_implicit` when `adaptive=True`,
    in which case the interval arithmetic module is going to be used, which
    requires the following edits.
    """
    def _print_And(self, expr):
        PREC = precedence(expr)
        return " & ".join(self.parenthesize(a, PREC)
                for a in sorted(expr.args, key=default_sort_key))

    def _print_Or(self, expr):
        PREC = precedence(expr)
        return " | ".join(self.parenthesize(a, PREC)
                for a in sorted(expr.args, key=default_sort_key))


def _uniform_eval(f1, f2, *args, modules=None,
    force_real_eval=False, has_sum=False):
    """
    Note: this is an experimental function, as such it is prone to changes.
    Please, do not use it in your code.
    """
    np = import_module('numpy')

    def wrapper_func(func, *args):
        try:
            return complex(func(*args))
        except (ZeroDivisionError, OverflowError):
            return complex(np.nan, np.nan)

    # NOTE: np.vectorize is much slower than numpy vectorized operations.
    # However, this modules must be able to evaluate functions also with
    # mpmath or sympy.
    wrapper_func = np.vectorize(wrapper_func, otypes=[complex])

    def _eval_with_sympy(err=None):
        if f2 is None:
            msg = "Impossible to evaluate the provided numerical function"
            if err is None:
                msg += "."
            else:
                msg += "because the following exception was raised:\n"
                "{}: {}".format(type(err).__name__, err)
            raise RuntimeError(msg)
        if err:
            warnings.warn(
                "The evaluation with %s failed.\n" % (
                    "NumPy/SciPy" if not modules else modules) +
                "{}: {}\n".format(type(err).__name__, err) +
                "Trying to evaluate the expression with Sympy, but it might "
                "be a slow operation."
            )
        return wrapper_func(f2, *args)

    if modules == "sympy":
        return _eval_with_sympy()

    try:
        return wrapper_func(f1, *args)
    except Exception as err:
        return _eval_with_sympy(err)


def _adaptive_eval(f, x):
    """Evaluate f(x) with an adaptive algorithm. Post-process the result.
    If a symbolic expression is evaluated with SymPy, it might returns
    another symbolic expression, containing additions, ...
    Force evaluation to a float.

    Parameters
    ==========
    f : callable
    x : float
    """
    np = import_module('numpy')

    y = f(x)
    if isinstance(y, Expr) and (not y.is_Number):
        y = y.evalf()
    y = complex(y)
    if y.imag > 1e-08:
        return np.nan
    return y.real


def _get_wrapper_for_expr(ret):
    wrapper = "%s"
    if ret == "real":
        wrapper = "re(%s)"
    elif ret == "imag":
        wrapper = "im(%s)"
    elif ret == "abs":
        wrapper = "abs(%s)"
    elif ret == "arg":
        wrapper = "arg(%s)"
    return wrapper


class BaseSeries:
    """Base class for the data objects containing stuff to be plotted.

    Notes
    =====

    The backend should check if it supports the data series that is given.
    (e.g. TextBackend supports only LineOver1DRangeSeries).
    It is the backend responsibility to know how to use the class of
    data series that is given.

    Some data series classes are grouped (using a class attribute like is_2Dline)
    according to the api they present (based only on convention). The backend is
    not obliged to use that api (e.g. LineOver1DRangeSeries belongs to the
    is_2Dline group and presents the get_points method, but the
    TextBackend does not use the get_points method).

    BaseSeries
    """

    # Some flags follow. The rationale for using flags instead of checking base
    # classes is that setting multiple flags is simpler than multiple
    # inheritance.

    is_2Dline = False
    # Some of the backends expect:
    #  - get_points returning 1D np.arrays list_x, list_y
    #  - get_color_array returning 1D np.array (done in Line2DBaseSeries)
    # with the colors calculated at the points from get_points

    is_3Dline = False
    # Some of the backends expect:
    #  - get_points returning 1D np.arrays list_x, list_y, list_y
    #  - get_color_array returning 1D np.array (done in Line2DBaseSeries)
    # with the colors calculated at the points from get_points

    is_3Dsurface = False
    # Some of the backends expect:
    #   - get_meshes returning mesh_x, mesh_y, mesh_z (2D np.arrays)
    #   - get_points an alias for get_meshes

    is_contour = False
    # Some of the backends expect:
    #   - get_meshes returning mesh_x, mesh_y, mesh_z (2D np.arrays)
    #   - get_points an alias for get_meshes

    is_implicit = False
    # Some of the backends expect:
    #   - get_meshes returning mesh_x (1D array), mesh_y(1D array,
    #     mesh_z (2D np.arrays)
    #   - get_points an alias for get_meshes
    # Different from is_contour as the colormap in backend will be
    # different

    is_interactive = False
    # An interactive series can update its data.

    is_parametric = False
    # The calculation of aesthetics expects:
    #   - get_parameter_points returning one or two np.arrays (1D or 2D)
    # used for calculation aesthetics

    is_generic = False
    # Represent generic user-provided numerical data

    is_vector = False
    is_2Dvector = False
    is_3Dvector = False
    # Represents a 2D or 3D vector data series

    _N = 100
    # default number of discretization points for uniform sampling. Each
    # subclass can set its number.

    def __init__(self, *args, **kwargs):
        kwargs = _set_discretization_points(kwargs.copy(), type(self))
        # discretize the domain using only integer numbers
        self.only_integers = kwargs.get("only_integers", False)
        # represents the evaluation modules to be used by lambdify
        self.modules = kwargs.get("modules", None)
        # plot functions might create data series that might not be useful to
        # be shown on the legend, for example wireframe lines on 3D plots.
        self.show_in_legend = kwargs.get("show_in_legend", True)
        # line and surface series can show data with a colormap, hence a
        # colorbar is essential to understand the data. However, sometime it
        # is useful to hide it on series-by-series base. The following keyword
        # controls whether the series should show a colorbar or not.
        self.colorbar = kwargs.get("colorbar", True)
        # Some series might use a colormap as default coloring. Setting this
        # attribute to False will inform the backends to use solid color.
        self.use_cm = kwargs.get("use_cm", False)
        # If True, the backend will attempt to render it on a polar-projection
        # axis, or using a polar discretization if a 3D plot is requested
        self.is_polar = kwargs.get("is_polar", kwargs.get("polar", False))
        # If True, the rendering will use points, not lines.
        self.is_point = kwargs.get("is_point", kwargs.get("point", False))
        # some backend is able to render latex, other needs standard text
        self._label = self._latex_label = ""

        self._ranges = []
        self._n = [
            int(kwargs.get("n1", self._N)),
            int(kwargs.get("n2", self._N)),
            int(kwargs.get("n3", self._N))
        ]
        self._scales = [
            kwargs.get("xscale", "linear"),
            kwargs.get("yscale", "linear"),
            kwargs.get("zscale", "linear")
        ]

        # enable interactive widget plots
        self._params = kwargs.get("params", {})
        if not isinstance(self._params, dict):
            raise TypeError("`params` must be a dictionary mapping symbols "
                "to numeric values.")
        if len(self._params) > 0:
            self.is_interactive = True

        # contains keyword arguments that will be passed to the rendering
        # function of the chosen plotting library
        self.rendering_kw = kwargs.get("rendering_kw", {})

        # numerical transformation functions to be applied to the output data:
        # x, y, z (coordinates), p (parameter on parametric plots)
        self._tx = kwargs.get("tx", None)
        self._ty = kwargs.get("ty", None)
        self._tz = kwargs.get("tz", None)
        self._tp = kwargs.get("tp", None)
        if not all(callable(t) or (t is None) for t in
            [self._tx, self._ty, self._tz, self._tp]):
            raise TypeError("`tx`, `ty`, `tz`, `tp` must be functions.")

        # list of numerical functions representing the expressions to evaluate
        self._functions = []
        # signature for the numerical functions
        self._signature = []
        # some expressions don't like to be evaluated over complex data.
        # if that's the case, set this to True
        self._force_real_eval = kwargs.get("force_real_eval", None)
        # this attribute will eventually contain a dictionary with the
        # discretized ranges
        self._discretized_domain = None
        # whether the series contains any interactive range, which is a range
        # where the minimum and maximum values can be changed with an
        # interactive widget
        self._interactive_ranges = False
        # NOTE: consider a generic summation, for example:
        #   s = Sum(cos(pi * x), (x, 1, y))
        # This gets lambdified to something:
        #   sum(cos(pi*x) for x in range(1, y+1))
        # Hence, y needs to be an integer, otherwise it raises:
        #   TypeError: 'complex' object cannot be interpreted as an integer
        # This list will contains symbols that are upper bound to summations
        # or products
        self._needs_to_be_int = []
        # a color function will be responsible to set the line/surface color
        # according to some logic. Each data series will et an appropriate
        # default value.
        self.color_func = None
        # NOTE: color_func usually receives numerical functions that are going
        # to be evaluated over the coordinates of the computed points (or the
        # discretized meshes).
        # However, if an expression is given to color_func, then it will be
        # lambdified with symbols in self._signature, and it will be evaluated
        # with the same data used to evaluate the plotted expression.
        self._eval_color_func_with_signature = False

    def _block_lambda_functions(self, *exprs):
        """Some data series can be used to plot numerical functions, others
        cannot. Execute this method inside the `__init__` to prevent the
        processing of numerical functions.
        """
        if any(callable(e) for e in exprs):
            raise TypeError(type(self).__name__ + " requires a symbolic "
                "expression.")

    def _check_fs(self):
        """ Checks if there are enough parameters and free symbols.
        """
        exprs, ranges = self.expr, self.ranges
        params, label = self.params, self.label
        exprs = exprs if hasattr(exprs, "__iter__") else [exprs]
        if any(callable(e) for e in exprs):
            return

        # from the expression's free symbols, remove the ones used in
        # the parameters and the ranges
        fs = _get_free_symbols(exprs)
        fs = fs.difference(params.keys())
        if ranges is not None:
            fs = fs.difference([r[0] for r in ranges])

        if len(fs) > 0:
            raise ValueError(
                "Incompatible expression and parameters.\n"
                + "Expression: {}\n".format(
                    (exprs, ranges, label) if ranges is not None else (exprs, label))
                + "params: {}\n".format(params)
                + "Specify what these symbols represent: {}\n".format(fs)
                + "Are they ranges or parameters?"
            )

