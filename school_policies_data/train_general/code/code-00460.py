#----------------------------------------------------------------------------
# Hermite polynomials
#


class hermite(OrthogonalPolynomial):
    r"""
    ``hermite(n, x)`` gives the $n$th Hermite polynomial in $x$, $H_n(x)$.

    Explanation
    ===========

    The Hermite polynomials are orthogonal on $(-\infty, \infty)$
    with respect to the weight $\exp\left(-x^2\right)$.

    Examples
    ========

    >>> from sympy import hermite, diff
    >>> from sympy.abc import x, n
    >>> hermite(0, x)
    1
    >>> hermite(1, x)
    2*x
    >>> hermite(2, x)
    4*x**2 - 2
    >>> hermite(n, x)
    hermite(n, x)
    >>> diff(hermite(n,x), x)
    2*n*hermite(n - 1, x)
    >>> hermite(n, -x)
    (-1)**n*hermite(n, x)

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite_prob,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hermite_polynomial
    .. [2] https://mathworld.wolfram.com/HermitePolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/HermiteH/

    """

    _ortho_poly = staticmethod(hermite_poly)

    @classmethod
    def eval(cls, n, x):
        if not n.is_Number:
            # Symbolic result H_n(x)
            # H_n(-x)  --->  (-1)**n * H_n(x)
            if x.could_extract_minus_sign():
                return S.NegativeOne**n * hermite(n, -x)
            # We can evaluate for some special values of x
            if x.is_zero:
                return 2**n * sqrt(S.Pi) / gamma((S.One - n)/2)
            elif x is S.Infinity:
                return S.Infinity
        else:
            # n is a given fixed integer, evaluate into polynomial
            if n.is_negative:
                raise ValueError(
                    "The index n must be nonnegative integer (got %r)" % n)
            else:
                return cls._eval_at_order(n, x)

    def fdiff(self, argindex=2):
        if argindex == 1:
            # Diff wrt n
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # Diff wrt x
            n, x = self.args
            return 2*n*hermite(n - 1, x)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, x, **kwargs):
        from sympy.concrete.summations import Sum
        k = Dummy("k")
        kern = S.NegativeOne**k / (factorial(k)*factorial(n - 2*k)) * (2*x)**(n - 2*k)
        return factorial(n)*Sum(kern, (k, 0, floor(n/2)))

    def _eval_rewrite_as_polynomial(self, n, x, **kwargs):
        # This function is just kept for backwards compatibility
        # but should not be used
        return self._eval_rewrite_as_Sum(n, x, **kwargs)

    def _eval_rewrite_as_hermite_prob(self, n, x, **kwargs):
        return sqrt(2)**n * hermite_prob(n, x*sqrt(2))


class hermite_prob(OrthogonalPolynomial):
    r"""
    ``hermite_prob(n, x)`` gives the $n$th probabilist's Hermite polynomial
    in $x$, $He_n(x)$.

    Explanation
    ===========

    The probabilist's Hermite polynomials are orthogonal on $(-\infty, \infty)$
    with respect to the weight $\exp\left(-\frac{x^2}{2}\right)$. They are monic
    polynomials, related to the plain Hermite polynomials (:py:class:`~.hermite`) by

    .. math :: He_n(x) = 2^{-n/2} H_n(x/\sqrt{2})

    Examples
    ========

    >>> from sympy import hermite_prob, diff, I
    >>> from sympy.abc import x, n
    >>> hermite_prob(1, x)
    x
    >>> hermite_prob(5, x)
    x**5 - 10*x**3 + 15*x
    >>> diff(hermite_prob(n,x), x)
    n*hermite_prob(n - 1, x)
    >>> hermite_prob(n, -x)
    (-1)**n*hermite_prob(n, x)

    The sum of absolute values of coefficients of $He_n(x)$ is the number of
    matchings in the complete graph $K_n$ or telephone number, A000085 in the OEIS:

    >>> [hermite_prob(n,I) / I**n for n in range(11)]
    [1, 1, 2, 4, 10, 26, 76, 232, 764, 2620, 9496]

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite,
    laguerre, assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hermite_polynomial
    .. [2] https://mathworld.wolfram.com/HermitePolynomial.html
    """

    _ortho_poly = staticmethod(hermite_prob_poly)

    @classmethod
    def eval(cls, n, x):
        if not n.is_Number:
            if x.could_extract_minus_sign():
                return S.NegativeOne**n * hermite_prob(n, -x)
            if x.is_zero:
                return sqrt(S.Pi) / gamma((S.One-n) / 2)
            elif x is S.Infinity:
                return S.Infinity
        else:
            if n.is_negative:
                ValueError("n must be a nonnegative integer, not %r" % n)
            else:
                return cls._eval_at_order(n, x)

    def fdiff(self, argindex=2):
        if argindex == 2:
            n, x = self.args
            return n*hermite_prob(n-1, x)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, x, **kwargs):
        from sympy.concrete.summations import Sum
        k = Dummy("k")
        kern = (-S.Half)**k * x**(n-2*k) / (factorial(k) * factorial(n-2*k))
        return factorial(n)*Sum(kern, (k, 0, floor(n/2)))

    def _eval_rewrite_as_polynomial(self, n, x, **kwargs):
        # This function is just kept for backwards compatibility
        # but should not be used
        return self._eval_rewrite_as_Sum(n, x, **kwargs)

    def _eval_rewrite_as_hermite(self, n, x, **kwargs):
        return sqrt(2)**(-n) * hermite(n, x/sqrt(2))


#----------------------------------------------------------------------------
# Laguerre polynomials
#


class laguerre(OrthogonalPolynomial):
    r"""
    Returns the $n$th Laguerre polynomial in $x$, $L_n(x)$.

    Examples
    ========

    >>> from sympy import laguerre, diff
    >>> from sympy.abc import x, n
    >>> laguerre(0, x)
    1
    >>> laguerre(1, x)
    1 - x
    >>> laguerre(2, x)
    x**2/2 - 2*x + 1
    >>> laguerre(3, x)
    -x**3/6 + 3*x**2/2 - 3*x + 1

    >>> laguerre(n, x)
    laguerre(n, x)

    >>> diff(laguerre(n, x), x)
    -assoc_laguerre(n - 1, 1, x)

    Parameters
    ==========

    n : int
        Degree of Laguerre polynomial. Must be `n \ge 0`.

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    assoc_laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Laguerre_polynomial
    .. [2] https://mathworld.wolfram.com/LaguerrePolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/LaguerreL/
    .. [4] https://functions.wolfram.com/Polynomials/LaguerreL3/

    """

    _ortho_poly = staticmethod(laguerre_poly)

    @classmethod
    def eval(cls, n, x):
        if n.is_integer is False:
            raise ValueError("Error: n should be an integer.")
        if not n.is_Number:
            # Symbolic result L_n(x)
            # L_{n}(-x)  --->  exp(-x) * L_{-n-1}(x)
            # L_{-n}(x)  --->  exp(x) * L_{n-1}(-x)
            if n.could_extract_minus_sign() and not(-n - 1).could_extract_minus_sign():
                return exp(x)*laguerre(-n - 1, -x)
            # We can evaluate for some special values of x
            if x.is_zero:
                return S.One
            elif x is S.NegativeInfinity:
                return S.Infinity
            elif x is S.Infinity:
                return S.NegativeOne**n * S.Infinity
        else:
            if n.is_negative:
                return exp(x)*laguerre(-n - 1, -x)
            else:
                return cls._eval_at_order(n, x)

    def fdiff(self, argindex=2):
        if argindex == 1:
            # Diff wrt n
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # Diff wrt x
            n, x = self.args
            return -assoc_laguerre(n - 1, 1, x)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, x, **kwargs):
        from sympy.concrete.summations import Sum
        # Make sure n \in N_0
        if n.is_negative:
            return exp(x) * self._eval_rewrite_as_Sum(-n - 1, -x, **kwargs)
        if n.is_integer is False:
            raise ValueError("Error: n should be an integer.")
        k = Dummy("k")
        kern = RisingFactorial(-n, k) / factorial(k)**2 * x**k
        return Sum(kern, (k, 0, n))

    def _eval_rewrite_as_polynomial(self, n, x, **kwargs):
        # This function is just kept for backwards compatibility
        # but should not be used
        return self._eval_rewrite_as_Sum(n, x, **kwargs)


class assoc_laguerre(OrthogonalPolynomial):
    r"""
    Returns the $n$th generalized Laguerre polynomial in $x$, $L_n(x)$.

    Examples
    ========

    >>> from sympy import assoc_laguerre, diff
    >>> from sympy.abc import x, n, a
    >>> assoc_laguerre(0, a, x)
    1
    >>> assoc_laguerre(1, a, x)
    a - x + 1
    >>> assoc_laguerre(2, a, x)
    a**2/2 + 3*a/2 + x**2/2 + x*(-a - 2) + 1
    >>> assoc_laguerre(3, a, x)
    a**3/6 + a**2 + 11*a/6 - x**3/6 + x**2*(a/2 + 3/2) +
        x*(-a**2/2 - 5*a/2 - 3) + 1

    >>> assoc_laguerre(n, a, 0)
    binomial(a + n, a)

    >>> assoc_laguerre(n, a, x)
    assoc_laguerre(n, a, x)

    >>> assoc_laguerre(n, 0, x)
    laguerre(n, x)

    >>> diff(assoc_laguerre(n, a, x), x)
    -assoc_laguerre(n - 1, a + 1, x)

    >>> diff(assoc_laguerre(n, a, x), a)
    Sum(assoc_laguerre(_k, a, x)/(-a + n), (_k, 0, n - 1))

    Parameters
    ==========

    n : int
        Degree of Laguerre polynomial. Must be `n \ge 0`.

    alpha : Expr
        Arbitrary expression. For ``alpha=0`` regular Laguerre
        polynomials will be generated.

    See Also
    ========

    jacobi, gegenbauer,
    chebyshevt, chebyshevt_root, chebyshevu, chebyshevu_root,
    legendre, assoc_legendre,
    hermite, hermite_prob,
    laguerre,
    sympy.polys.orthopolys.jacobi_poly
    sympy.polys.orthopolys.gegenbauer_poly
    sympy.polys.orthopolys.chebyshevt_poly
    sympy.polys.orthopolys.chebyshevu_poly
    sympy.polys.orthopolys.hermite_poly
    sympy.polys.orthopolys.hermite_prob_poly
    sympy.polys.orthopolys.legendre_poly
    sympy.polys.orthopolys.laguerre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Laguerre_polynomial#Generalized_Laguerre_polynomials
    .. [2] https://mathworld.wolfram.com/AssociatedLaguerrePolynomial.html
    .. [3] https://functions.wolfram.com/Polynomials/LaguerreL/
    .. [4] https://functions.wolfram.com/Polynomials/LaguerreL3/

    """

    @classmethod
    def eval(cls, n, alpha, x):
        # L_{n}^{0}(x)  --->  L_{n}(x)
        if alpha.is_zero:
            return laguerre(n, x)

        if not n.is_Number:
            # We can evaluate for some special values of x
            if x.is_zero:
                return binomial(n + alpha, alpha)
            elif x is S.Infinity and n > 0:
                return S.NegativeOne**n * S.Infinity
            elif x is S.NegativeInfinity and n > 0:
                return S.Infinity
        else:
            # n is a given fixed integer, evaluate into polynomial
            if n.is_negative:
                raise ValueError(
                    "The index n must be nonnegative integer (got %r)" % n)
            else:
                return laguerre_poly(n, x, alpha)

    def fdiff(self, argindex=3):
        from sympy.concrete.summations import Sum
        if argindex == 1:
            # Diff wrt n
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # Diff wrt alpha
            n, alpha, x = self.args
            k = Dummy("k")
            return Sum(assoc_laguerre(k, alpha, x) / (n - alpha), (k, 0, n - 1))
        elif argindex == 3:
            # Diff wrt x
            n, alpha, x = self.args
            return -assoc_laguerre(n - 1, alpha + 1, x)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Sum(self, n, alpha, x, **kwargs):
        from sympy.concrete.summations import Sum
        # Make sure n \in N_0
        if n.is_negative or n.is_integer is False:
            raise ValueError("Error: n should be a non-negative integer.")
        k = Dummy("k")
        kern = RisingFactorial(
            -n, k) / (gamma(k + alpha + 1) * factorial(k)) * x**k
        return gamma(n + alpha + 1) / factorial(n) * Sum(kern, (k, 0, n))

    def _eval_rewrite_as_polynomial(self, n, alpha, x, **kwargs):
        # This function is just kept for backwards compatibility
        # but should not be used
        return self._eval_rewrite_as_Sum(n, alpha, x, **kwargs)

    def _eval_conjugate(self):
        n, alpha, x = self.args
        return self.func(n, alpha.conjugate(), x.conjugate())
<|endoftext|>""" This module contains the Mathieu functions.
"""

from sympy.core.function import DefinedFunction, ArgumentIndexError
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin, cos


class MathieuBase(DefinedFunction):
    """
    Abstract base class for Mathieu functions.

    This class is meant to reduce code duplication.

    """

    unbranched = True

    def _eval_conjugate(self):
        a, q, z = self.args
        return self.func(a.conjugate(), q.conjugate(), z.conjugate())


class mathieus(MathieuBase):
    r"""
    The Mathieu Sine function $S(a,q,z)$.

    Explanation
    ===========

    This function is one solution of the Mathieu differential equation:

    .. math ::
        y(x)^{\prime\prime} + (a - 2 q \cos(2 x)) y(x) = 0

    The other solution is the Mathieu Cosine function.

    Examples
    ========

    >>> from sympy import diff, mathieus
    >>> from sympy.abc import a, q, z

    >>> mathieus(a, q, z)
    mathieus(a, q, z)

    >>> mathieus(a, 0, z)
    sin(sqrt(a)*z)

    >>> diff(mathieus(a, q, z), z)
    mathieusprime(a, q, z)

    See Also
    ========

    mathieuc: Mathieu cosine function.
    mathieusprime: Derivative of Mathieu sine function.
    mathieucprime: Derivative of Mathieu cosine function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Mathieu_function
    .. [2] https://dlmf.nist.gov/28
    .. [3] https://mathworld.wolfram.com/MathieuFunction.html
    .. [4] https://functions.wolfram.com/MathieuandSpheroidalFunctions/MathieuS/

    """

    def fdiff(self, argindex=1):
        if argindex == 3:
            a, q, z = self.args
            return mathieusprime(a, q, z)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, a, q, z):
        if q.is_Number and q.is_zero:
            return sin(sqrt(a)*z)
        # Try to pull out factors of -1
        if z.could_extract_minus_sign():
            return -cls(a, q, -z)


class mathieuc(MathieuBase):
    r"""
    The Mathieu Cosine function $C(a,q,z)$.

    Explanation
    ===========

    This function is one solution of the Mathieu differential equation:

    .. math ::
        y(x)^{\prime\prime} + (a - 2 q \cos(2 x)) y(x) = 0

    The other solution is the Mathieu Sine function.

    Examples
    ========

    >>> from sympy import diff, mathieuc
    >>> from sympy.abc import a, q, z

    >>> mathieuc(a, q, z)
    mathieuc(a, q, z)

    >>> mathieuc(a, 0, z)
    cos(sqrt(a)*z)

    >>> diff(mathieuc(a, q, z), z)
    mathieucprime(a, q, z)

    See Also
    ========

    mathieus: Mathieu sine function
    mathieusprime: Derivative of Mathieu sine function
    mathieucprime: Derivative of Mathieu cosine function

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Mathieu_function
    .. [2] https://dlmf.nist.gov/28
    .. [3] https://mathworld.wolfram.com/MathieuFunction.html
    .. [4] https://functions.wolfram.com/MathieuandSpheroidalFunctions/MathieuC/

    """

    def fdiff(self, argindex=1):
        if argindex == 3:
            a, q, z = self.args
            return mathieucprime(a, q, z)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, a, q, z):
        if q.is_Number and q.is_zero:
            return cos(sqrt(a)*z)
        # Try to pull out factors of -1
        if z.could_extract_minus_sign():
            return cls(a, q, -z)


class mathieusprime(MathieuBase):
    r"""
    The derivative $S^{\prime}(a,q,z)$ of the Mathieu Sine function.

    Explanation
    ===========

    This function is one solution of the Mathieu differential equation:

    .. math ::
        y(x)^{\prime\prime} + (a - 2 q \cos(2 x)) y(x) = 0

    The other solution is the Mathieu Cosine function.

    Examples
    ========

    >>> from sympy import diff, mathieusprime
    >>> from sympy.abc import a, q, z

    >>> mathieusprime(a, q, z)
    mathieusprime(a, q, z)

    >>> mathieusprime(a, 0, z)
    sqrt(a)*cos(sqrt(a)*z)

    >>> diff(mathieusprime(a, q, z), z)
    (-a + 2*q*cos(2*z))*mathieus(a, q, z)

    See Also
    ========

    mathieus: Mathieu sine function
    mathieuc: Mathieu cosine function
    mathieucprime: Derivative of Mathieu cosine function

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Mathieu_function
    .. [2] https://dlmf.nist.gov/28
    .. [3] https://mathworld.wolfram.com/MathieuFunction.html
    .. [4] https://functions.wolfram.com/MathieuandSpheroidalFunctions/MathieuSPrime/

    """

    def fdiff(self, argindex=1):
        if argindex == 3:
            a, q, z = self.args
            return (2*q*cos(2*z) - a)*mathieus(a, q, z)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, a, q, z):
        if q.is_Number and q.is_zero:
            return sqrt(a)*cos(sqrt(a)*z)
        # Try to pull out factors of -1
        if z.could_extract_minus_sign():
            return cls(a, q, -z)


class mathieucprime(MathieuBase):
    r"""
    The derivative $C^{\prime}(a,q,z)$ of the Mathieu Cosine function.

    Explanation
    ===========

    This function is one solution of the Mathieu differential equation:

    .. math ::
        y(x)^{\prime\prime} + (a - 2 q \cos(2 x)) y(x) = 0

    The other solution is the Mathieu Sine function.

    Examples
    ========

    >>> from sympy import diff, mathieucprime
    >>> from sympy.abc import a, q, z

    >>> mathieucprime(a, q, z)
    mathieucprime(a, q, z)

    >>> mathieucprime(a, 0, z)
    -sqrt(a)*sin(sqrt(a)*z)

    >>> diff(mathieucprime(a, q, z), z)
    (-a + 2*q*cos(2*z))*mathieuc(a, q, z)

    See Also
    ========

    mathieus: Mathieu sine function
    mathieuc: Mathieu cosine function
    mathieusprime: Derivative of Mathieu sine function

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Mathieu_function
    .. [2] https://dlmf.nist.gov/28
    .. [3] https://mathworld.wolfram.com/MathieuFunction.html
    .. [4] https://functions.wolfram.com/MathieuandSpheroidalFunctions/MathieuCPrime/

    """

    def fdiff(self, argindex=1):
        if argindex == 3:
            a, q, z = self.args
            return (2*q*cos(2*z) - a)*mathieuc(a, q, z)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, a, q, z):
        if q.is_Number and q.is_zero:
            return -sqrt(a)*sin(sqrt(a)*z)
        # Try to pull out factors of -1
        if z.could_extract_minus_sign():
            return -cls(a, q, -z)
<|endoftext|>from functools import wraps

from sympy.core import S
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import DefinedFunction, ArgumentIndexError, _mexpand
from sympy.core.logic import fuzzy_or, fuzzy_not
from sympy.core.numbers import Rational, pi, I
from sympy.core.power import Pow
from sympy.core.symbol import Dummy, uniquely_named_symbol, Wild
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial, RisingFactorial
from sympy.functions.elementary.trigonometric import sin, cos, csc, cot
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import cbrt, sqrt, root
from sympy.functions.elementary.complexes import (Abs, re, im, polar_lift, unpolarify)
from sympy.functions.special.gamma_functions import gamma, digamma, uppergamma
from sympy.functions.special.hyper import hyper
from sympy.polys.orthopolys import spherical_bessel_fn

from mpmath import mp, workprec

# TODO
# o Scorer functions G1 and G2
# o Asymptotic expansions
#   These are possible, e.g. for fixed order, but since the bessel type
#   functions are oscillatory they are not actually tractable at
#   infinity, so this is not particularly useful right now.
# o Nicer series expansions.
# o More rewriting.
# o Add solvers to ode.py (or rather add solvers for the hypergeometric equation).


class BesselBase(DefinedFunction):
    """
    Abstract base class for Bessel-type functions.

    This class is meant to reduce code duplication.
    All Bessel-type functions can 1) be differentiated, with the derivatives
    expressed in terms of similar functions, and 2) be rewritten in terms
    of other Bessel-type functions.

    Here, Bessel-type functions are assumed to have one complex parameter.

    To use this base class, define class attributes ``_a`` and ``_b`` such that
    ``2*F_n' = -_a*F_{n+1} + b*F_{n-1}``.

    """

    @property
    def order(self):
        """ The order of the Bessel-type function. """
        return self.args[0]

    @property
    def argument(self):
        """ The argument of the Bessel-type function. """
        return self.args[1]

    @classmethod
    def eval(cls, nu, z):
        return

    def fdiff(self, argindex=2):
        if argindex != 2:
            raise ArgumentIndexError(self, argindex)
        return (self._b/2 * self.__class__(self.order - 1, self.argument) -
                self._a/2 * self.__class__(self.order + 1, self.argument))

    def _eval_conjugate(self):
        z = self.argument
        if z.is_extended_negative is False:
            return self.__class__(self.order.conjugate(), z.conjugate())

    def _eval_is_meromorphic(self, x, a):
        nu, z = self.order, self.argument

        if nu.has(x):
            return False
        if not z._eval_is_meromorphic(x, a):
            return None
        z0 = z.subs(x, a)
        if nu.is_integer:
            if isinstance(self, (besselj, besseli, hn1, hn2, jn, yn)) or not nu.is_zero:
                return fuzzy_not(z0.is_infinite)
        return fuzzy_not(fuzzy_or([z0.is_zero, z0.is_infinite]))

    def _eval_expand_func(self, **hints):
        nu, z, f = self.order, self.argument, self.__class__
        if nu.is_real:
            if (nu - 1).is_positive:
                return (-self._a*self._b*f(nu - 2, z)._eval_expand_func() +
                        2*self._a*(nu - 1)*f(nu - 1, z)._eval_expand_func()/z)
            elif (nu + 1).is_negative:
                return (2*self._b*(nu + 1)*f(nu + 1, z)._eval_expand_func()/z -
                        self._a*self._b*f(nu + 2, z)._eval_expand_func())
        return self

    def _eval_simplify(self, **kwargs):
        from sympy.simplify.simplify import besselsimp
        return besselsimp(self)


class besselj(BesselBase):
    r"""
    Bessel function of the first kind.

    Explanation
    ===========

    The Bessel $J$ function of order $\nu$ is defined to be the function
    satisfying Bessel's differential equation

    .. math ::
        z^2 \frac{\mathrm{d}^2 w}{\mathrm{d}z^2}
        + z \frac{\mathrm{d}w}{\mathrm{d}z} + (z^2 - \nu^2) w = 0,

    with Laurent expansion

    .. math ::
        J_\nu(z) = z^\nu \left(\frac{1}{\Gamma(\nu + 1) 2^\nu} + O(z^2) \right),

    if $\nu$ is not a negative integer. If $\nu=-n \in \mathbb{Z}_{<0}$
    *is* a negative integer, then the definition is

    .. math ::
        J_{-n}(z) = (-1)^n J_n(z).

    Examples
    ========

    Create a Bessel function object:

    >>> from sympy import besselj, jn
    >>> from sympy.abc import z, n
    >>> b = besselj(n, z)

    Differentiate it:

    >>> b.diff(z)
    besselj(n - 1, z)/2 - besselj(n + 1, z)/2

    Rewrite in terms of spherical Bessel functions:

    >>> b.rewrite(jn)
    sqrt(2)*sqrt(z)*jn(n - 1/2, z)/sqrt(pi)

    Access the parameter and argument:

    >>> b.order
    n
    >>> b.argument
    z

    See Also
    ========

    bessely, besseli, besselk

    References
    ==========

    .. [1] Abramowitz, Milton; Stegun, Irene A., eds. (1965), "Chapter 9",
           Handbook of Mathematical Functions with Formulas, Graphs, and
           Mathematical Tables
    .. [2] Luke, Y. L. (1969), The Special Functions and Their
           Approximations, Volume 1
    .. [3] https://en.wikipedia.org/wiki/Bessel_function
    .. [4] https://functions.wolfram.com/Bessel-TypeFunctions/BesselJ/

    """

    _a = S.One
    _b = S.One

    @classmethod
    def eval(cls, nu, z):
        if z.is_zero:
            if nu.is_zero:
                return S.One
            elif (nu.is_integer and nu.is_zero is False) or re(nu).is_positive:
                return S.Zero
            elif re(nu).is_negative and not (nu.is_integer is True):
                return S.ComplexInfinity
            elif nu.is_imaginary:
                return S.NaN
        if z in (S.Infinity, S.NegativeInfinity):
            return S.Zero

        if z.could_extract_minus_sign():
            return (z)**nu*(-z)**(-nu)*besselj(nu, -z)
        if nu.is_integer:
            if nu.could_extract_minus_sign():
                return S.NegativeOne**(-nu)*besselj(-nu, z)
            newz = z.extract_multiplicatively(I)
            if newz:  # NOTE we don't want to change the function if z==0
                return I**(nu)*besseli(nu, newz)

        # branch handling:
        if nu.is_integer:
            newz = unpolarify(z)
            if newz != z:
                return besselj(nu, newz)
        else:
            newz, n = z.extract_branch_factor()
            if n != 0:
                return exp(2*n*pi*nu*I)*besselj(nu, newz)
        nnu = unpolarify(nu)
        if nu != nnu:
            return besselj(nnu, z)

    def _eval_rewrite_as_besseli(self, nu, z, **kwargs):
        return exp(I*pi*nu/2)*besseli(nu, polar_lift(-I)*z)

    def _eval_rewrite_as_bessely(self, nu, z, **kwargs):
        if nu.is_integer is False:
            return csc(pi*nu)*bessely(-nu, z) - cot(pi*nu)*bessely(nu, z)

    def _eval_rewrite_as_jn(self, nu, z, **kwargs):
        return sqrt(2*z/pi)*jn(nu - S.Half, self.argument)

    def _eval_as_leading_term(self, x, logx, cdir):
        nu, z = self.args
        try:
            arg = z.as_leading_term(x)
        except NotImplementedError:
            return self
        c, e = arg.as_coeff_exponent(x)

        if e.is_positive:
            return arg**nu/(2**nu*gamma(nu + 1))
        elif e.is_negative:
            cdir = 1 if cdir == 0 else cdir
            sign = c*cdir**e
            if not sign.is_negative:
                # Refer Abramowitz and Stegun 1965, p. 364 for more information on
                # asymptotic approximation of besselj function.
                return sqrt(2)*cos(z - pi*(2*nu + 1)/4)/sqrt(pi*z)
            return self

        return super(besselj, self)._eval_as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_is_extended_real(self):
        nu, z = self.args
        if nu.is_integer and z.is_extended_real:
            return True

    def _eval_nseries(self, x, n, logx, cdir=0):
        # Refer https://functions.wolfram.com/Bessel-TypeFunctions/BesselJ/06/01/04/01/01/0003/
        # for more information on nseries expansion of besselj function.
        from sympy.series.order import Order
        nu, z = self.args

        # In case of powers less than 1, number of terms need to be computed
        # separately to avoid repeated callings of _eval_nseries with wrong n
        try:
            _, exp = z.leadterm(x)
        except (ValueError, NotImplementedError):
            return self

        if exp.is_positive:
            newn = ceiling(n/exp)
            o = Order(x**n, x)
            r = (z/2)._eval_nseries(x, n, logx, cdir).removeO()
            if r is S.Zero:
                return o
            t = (_mexpand(r**2) + o).removeO()

            term = r**nu/gamma(nu + 1)
            s = [term]
            for k in range(1, (newn + 1)//2):
                term *= -t/(k*(nu + k))
                term = (_mexpand(term) + o).removeO()
                s.append(term)
            return Add(*s) + o

        return super(besselj, self)._eval_nseries(x, n, logx, cdir)


class bessely(BesselBase):
    r"""
    Bessel function of the second kind.

    Explanation
    ===========

    The Bessel $Y$ function of order $\nu$ is defined as

    .. math ::
        Y_\nu(z) = \lim_{\mu \to \nu} \frac{J_\mu(z) \cos(\pi \mu)
                                            - J_{-\mu}(z)}{\sin(\pi \mu)},

    where $J_\mu(z)$ is the Bessel function of the first kind.

    It is a solution to Bessel's equation, and linearly independent from
    $J_\nu$.

    Examples
    ========

    >>> from sympy import bessely, yn
    >>> from sympy.abc import z, n
    >>> b = bessely(n, z)
    >>> b.diff(z)
    bessely(n - 1, z)/2 - bessely(n + 1, z)/2
    >>> b.rewrite(yn)
    sqrt(2)*sqrt(z)*yn(n - 1/2, z)/sqrt(pi)

    See Also
    ========

    besselj, besseli, besselk

    References
    ==========

    .. [1] https://functions.wolfram.com/Bessel-TypeFunctions/BesselY/

    """

    _a = S.One
    _b = S.One

    @classmethod
    def eval(cls, nu, z):
        if z.is_zero:
            if nu.is_zero:
                return S.NegativeInfinity
            elif re(nu).is_zero is False:
                return S.ComplexInfinity
            elif re(nu).is_zero:
                return S.NaN
        if z in (S.Infinity, S.NegativeInfinity):
            return S.Zero
        if z == I*S.Infinity:
            return exp(I*pi*(nu + 1)/2) * S.Infinity
        if z == I*S.NegativeInfinity:
            return exp(-I*pi*(nu + 1)/2) * S.Infinity

        if nu.is_integer:
            if nu.could_extract_minus_sign():
                return S.NegativeOne**(-nu)*bessely(-nu, z)

    def _eval_rewrite_as_besselj(self, nu, z, **kwargs):
        if nu.is_integer is False:
            return csc(pi*nu)*(cos(pi*nu)*besselj(nu, z) - besselj(-nu, z))

    def _eval_rewrite_as_besseli(self, nu, z, **kwargs):
        aj = self._eval_rewrite_as_besselj(*self.args)
        if aj:
            return aj.rewrite(besseli)

    def _eval_rewrite_as_yn(self, nu, z, **kwargs):
        return sqrt(2*z/pi) * yn(nu - S.Half, self.argument)

    def _eval_as_leading_term(self, x, logx, cdir):
        nu, z = self.args
        try:
            arg = z.as_leading_term(x)
        except NotImplementedError:
            return self
        c, e = arg.as_coeff_exponent(x)

        if e.is_positive:
            term_one = ((2/pi)*log(z/2)*besselj(nu, z))
            term_two = -(z/2)**(-nu)*factorial(nu - 1)/pi if (nu).is_positive else S.Zero
            term_three = -(z/2)**nu/(pi*factorial(nu))*(digamma(nu + 1) - S.EulerGamma)
            arg = Add(*[term_one, term_two, term_three]).as_leading_term(x, logx=logx)
            return arg
        elif e.is_negative:
            cdir = 1 if cdir == 0 else cdir
            sign = c*cdir**e
            if not sign.is_negative:
                # Refer Abramowitz and Stegun 1965, p. 364 for more information on
                # asymptotic approximation of bessely function.
                return sqrt(2)*(-sin(pi*nu/2 - z + pi/4) + 3*cos(pi*nu/2 - z + pi/4)/(8*z))*sqrt(1/z)/sqrt(pi)
            return self

        return super(bessely, self)._eval_as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_is_extended_real(self):
        nu, z = self.args
        if nu.is_integer and z.is_positive:
            return True

    def _eval_nseries(self, x, n, logx, cdir=0):
        # Refer https://functions.wolfram.com/Bessel-TypeFunctions/BesselY/06/01/04/01/02/0008/
        # for more information on nseries expansion of bessely function.
        from sympy.series.order import Order
        nu, z = self.args

        # In case of powers less than 1, number of terms need to be computed
        # separately to avoid repeated callings of _eval_nseries with wrong n
        try:
            _, exp = z.leadterm(x)
        except (ValueError, NotImplementedError):
            return self

        if exp.is_positive and nu.is_integer:
            newn = ceiling(n/exp)
            bn = besselj(nu, z)
            a = ((2/pi)*log(z/2)*bn)._eval_nseries(x, n, logx, cdir)

            b, c = [], []
            o = Order(x**n, x)
            r = (z/2)._eval_nseries(x, n, logx, cdir).removeO()
            if r is S.Zero:
                return o
            t = (_mexpand(r**2) + o).removeO()

            if nu > S.Zero:
                term = r**(-nu)*factorial(nu - 1)/pi
                b.append(term)
                for k in range(1, nu):
                    denom = (nu - k)*k
                    if denom == S.Zero:
                        term *= t/k
                    else:
                        term *= t/denom
                    term = (_mexpand(term) + o).removeO()
                    b.append(term)

            p = r**nu/(pi*factorial(nu))
            term = p*(digamma(nu + 1) - S.EulerGamma)
            c.append(term)
            for k in range(1, (newn + 1)//2):
                p *= -t/(k*(k + nu))
                p = (_mexpand(p) + o).removeO()
                term = p*(digamma(k + nu + 1) + digamma(k + 1))
                c.append(term)
            return a - Add(*b) - Add(*c) # Order term comes from a

        return super(bessely, self)._eval_nseries(x, n, logx, cdir)


class besseli(BesselBase):
    r"""
    Modified Bessel function of the first kind.

    Explanation
    ===========

    The Bessel $I$ function is a solution to the modified Bessel equation

    .. math ::
        z^2 \frac{\mathrm{d}^2 w}{\mathrm{d}z^2}
        + z \frac{\mathrm{d}w}{\mathrm{d}z} + (z^2 + \nu^2)^2 w = 0.

    It can be defined as

    .. math ::
        I_\nu(z) = i^{-\nu} J_\nu(iz),

    where $J_\nu(z)$ is the Bessel function of the first kind.

    Examples
    ========

    >>> from sympy import besseli
    >>> from sympy.abc import z, n
    >>> besseli(n, z).diff(z)
    besseli(n - 1, z)/2 + besseli(n + 1, z)/2

    See Also
    ========

    besselj, bessely, besselk

    References
    ==========

    .. [1] https://functions.wolfram.com/Bessel-TypeFunctions/BesselI/

    """

    _a = -S.One
    _b = S.One

    @classmethod
    def eval(cls, nu, z):
        if z.is_zero:
            if nu.is_zero:
                return S.One
            elif (nu.is_integer and nu.is_zero is False) or re(nu).is_positive:
                return S.Zero
            elif re(nu).is_negative and not (nu.is_integer is True):
                return S.ComplexInfinity
            elif nu.is_imaginary:
                return S.NaN
        if im(z) in (S.Infinity, S.NegativeInfinity):
            return S.Zero
        if z is S.Infinity:
            return S.Infinity
        if z is S.NegativeInfinity:
            return (-1)**nu*S.Infinity

        if z.could_extract_minus_sign():
            return (z)**nu*(-z)**(-nu)*besseli(nu, -z)
        if nu.is_integer:
            if nu.could_extract_minus_sign():
                return besseli(-nu, z)
            newz = z.extract_multiplicatively(I)
            if newz:  # NOTE we don't want to change the function if z==0
                return I**(-nu)*besselj(nu, -newz)

        # branch handling:
        if nu.is_integer:
            newz = unpolarify(z)
            if newz != z:
                return besseli(nu, newz)
        else:
            newz, n = z.extract_branch_factor()
            if n != 0:
                return exp(2*n*pi*nu*I)*besseli(nu, newz)
        nnu = unpolarify(nu)
        if nu != nnu:
            return besseli(nnu, z)

    def _eval_rewrite_as_tractable(self, nu, z, limitvar=None, **kwargs):
        if z.is_extended_real:
            return exp(z)*_besseli(nu, z)

    def _eval_rewrite_as_besselj(self, nu, z, **kwargs):
        return exp(-I*pi*nu/2)*besselj(nu, polar_lift(I)*z)

    def _eval_rewrite_as_bessely(self, nu, z, **kwargs):
        aj = self._eval_rewrite_as_besselj(*self.args)
        if aj:
            return aj.rewrite(bessely)

    def _eval_rewrite_as_jn(self, nu, z, **kwargs):
        return self._eval_rewrite_as_besselj(*self.args).rewrite(jn)

    def _eval_is_extended_real(self):
        nu, z = self.args
        if nu.is_integer and z.is_extended_real:
            return True

    def _eval_as_leading_term(self, x, logx, cdir):
        nu, z = self.args
        try:
            arg = z.as_leading_term(x)
        except NotImplementedError:
            return self
        c, e = arg.as_coeff_exponent(x)

        if e.is_positive:
            return arg**nu/(2**nu*gamma(nu + 1))
        elif e.is_negative:
            cdir = 1 if cdir == 0 else cdir
            sign = c*cdir**e
            if not sign.is_negative:
                # Refer Abramowitz and Stegun 1965, p. 377 for more information on
                # asymptotic approximation of besseli function.
                return exp(z)/sqrt(2*pi*z)
            return self

        return super(besseli, self)._eval_as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):
        # Refer https://functions.wolfram.com/Bessel-TypeFunctions/BesselI/06/01/04/01/01/0003/
        # for more information on nseries expansion of besseli function.
        from sympy.series.order import Order
        nu, z = self.args

        # In case of powers less than 1, number of terms need to be computed
        # separately to avoid repeated callings of _eval_nseries with wrong n
        try:
            _, exp = z.leadterm(x)
        except (ValueError, NotImplementedError):
            return self

        if exp.is_positive:
            newn = ceiling(n/exp)
            o = Order(x**n, x)
            r = (z/2)._eval_nseries(x, n, logx, cdir).removeO()
            if r is S.Zero:
                return o
            t = (_mexpand(r**2) + o).removeO()

            term = r**nu/gamma(nu + 1)
            s = [term]
            for k in range(1, (newn + 1)//2):
                term *= t/(k*(nu + k))
                term = (_mexpand(term) + o).removeO()
                s.append(term)
            return Add(*s) + o

        return super(besseli, self)._eval_nseries(x, n, logx, cdir)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.functions.combinatorial.factorials import RisingFactorial
        from sympy.series.order import Order
        point = args0[1]

        if point in [S.Infinity, S.NegativeInfinity]:
            nu, z = self.args
            s = [(RisingFactorial(Rational(2*nu - 1, 2), k)*RisingFactorial(Rational(2*nu + 1, 2), k))/\
            ((2)**(k)*z**(Rational(2*k + 1, 2))*factorial(k)) for k in range(n)] + [Order(1/z**(Rational(2*n + 1, 2)), x)]
            return exp(z)/sqrt(2*pi) * (Add(*s))

        return super()._eval_aseries(n, args0, x, logx)


class besselk(BesselBase):
    r"""
    Modified Bessel function of the second kind.

    Explanation
    ===========

    The Bessel $K$ function of order $\nu$ is defined as

    .. math ::
        K_\nu(z) = \lim_{\mu \to \nu} \frac{\pi}{2}
                   \frac{I_{-\mu}(z) -I_\mu(z)}{\sin(\pi \mu)},

    where $I_\mu(z)$ is the modified Bessel function of the first kind.

    It is a solution of the modified Bessel equation, and linearly independent
    from $Y_\nu$.

    Examples
    ========

    >>> from sympy import besselk
    >>> from sympy.abc import z, n
    >>> besselk(n, z).diff(z)
    -besselk(n - 1, z)/2 - besselk(n + 1, z)/2

    See Also
    ========

    besselj, besseli, bessely

    References
    ==========

    .. [1] https://functions.wolfram.com/Bessel-TypeFunctions/BesselK/

    """

    _a = S.One
    _b = -S.One

    @classmethod
    def eval(cls, nu, z):
        if z.is_zero:
            if nu.is_zero:
                return S.Infinity
            elif re(nu).is_zero is False:
                return S.ComplexInfinity
            elif re(nu).is_zero:
                return S.NaN
        if z in (S.Infinity, I*S.Infinity, I*S.NegativeInfinity):
            return S.Zero

        if nu.is_integer:
            if nu.could_extract_minus_sign():
                return besselk(-nu, z)

    def _eval_rewrite_as_besseli(self, nu, z, **kwargs):
        if nu.is_integer is False:
            return pi*csc(pi*nu)*(besseli(-nu, z) - besseli(nu, z))/2

    def _eval_rewrite_as_besselj(self, nu, z, **kwargs):
        ai = self._eval_rewrite_as_besseli(*self.args)
        if ai:
            return ai.rewrite(besselj)

    def _eval_rewrite_as_bessely(self, nu, z, **kwargs):
        aj = self._eval_rewrite_as_besselj(*self.args)
        if aj:
            return aj.rewrite(bessely)

    def _eval_rewrite_as_yn(self, nu, z, **kwargs):
        ay = self._eval_rewrite_as_bessely(*self.args)
        if ay:
            return ay.rewrite(yn)

    def _eval_is_extended_real(self):
        nu, z = self.args
        if nu.is_integer and z.is_positive:
            return True

    def _eval_rewrite_as_tractable(self, nu, z, limitvar=None, **kwargs):
        if z.is_extended_real:
            return exp(-z)*_besselk(nu, z)

    def _eval_as_leading_term(self, x, logx, cdir):
        nu, z = self.args
        try:
            arg = z.as_leading_term(x)
        except NotImplementedError:
            return self
        _, e = arg.as_coeff_exponent(x)

        if e.is_positive:
            if nu.is_zero:
                # Equation 9.6.8 of Abramowitz and Stegun (10th ed, 1972).
                term = -log(z) - S.EulerGamma + log(2)
            elif nu.is_nonzero:
                # Equation 9.6.9 of Abramowitz and Stegun (10th ed, 1972).
                term = gamma(Abs(nu))*(z/2)**(-Abs(nu))/2
            else:
                raise NotImplementedError(f"Cannot proceed without knowing if {nu} is zero or not.")

            return term.as_leading_term(x, logx=logx)
        elif e.is_negative:
            # Equation 9.7.2 of Abramowitz and Stegun (10th ed, 1972).
            return sqrt(pi)*exp(-arg)/sqrt(2*arg)
        else:
            return self.func(nu, arg)

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.series.order import Order
        nu, z = self.args

        try:
            _, exp = z.leadterm(x)
        except (ValueError, NotImplementedError):
            return self

        # In case of powers less than 1, number of terms need to be computed
        # separately to avoid repeated callings of _eval_nseries with wrong n
        if exp.is_positive:
            r = (z/2)._eval_nseries(x, n, logx, cdir).removeO()
            if r is S.Zero:
                return Order(z**(-nu) + z**nu, x)

            o = Order(x**n, x)
            if nu.is_integer:
                # Reference: https://functions.wolfram.com/Bessel-TypeFunctions/BesselK/06/01/04/01/02/0008/ (only for integer order)
                newn = ceiling(n/exp)
                bn = besseli(nu, z)
                a = ((-1)**(nu - 1)*log(z/2)*bn)._eval_nseries(x, n, logx, cdir)

                b, c = [], []
                t = _mexpand(r**2)

                if nu > S.Zero:
                    term = r**(-nu)*factorial(nu - 1)/2
                    b.append(term)
                    for k in range(1, nu):
                        term *= t/((k - nu)*k)
                        term = (_mexpand(term) + o).removeO()
                        b.append(term)

                p = r**nu*(-1)**nu/(2*factorial(nu))
                term = p*(digamma(nu + 1) - S.EulerGamma)
                c.append(term)
                for k in range(1, (newn + 1)//2):
                    p *= t/(k*(k + nu))
                    p = (_mexpand(p) + o).removeO()
                    term = p*(digamma(k + nu + 1) + digamma(k + 1))
                    c.append(term)
                return a + Add(*b) + Add(*c) + o
            elif nu.is_noninteger:
                # Reference: https://functions.wolfram.com/Bessel-TypeFunctions/BesselK/06/01/04/01/01/0003/
                # (only for non-integer order).
                # While the expression in the reference above seems correct
                # for non-real order as well, it would need some manipulation
                # (not implemented) to be written as a power series in x with
                # real exponents [e.g. Dunster 1990. "Bessel functions
                # of purely imaginary order, with an application to second-order
                # linear differential equations having a large parameter".
                # SIAM J. Math. Anal. Vol 21, No. 4, pp 995-1018.].
                newn_a = ceiling((n+nu)/exp)
                newn_b = ceiling((n-nu)/exp)

                a, b = [], []
                for k in range((newn_a+1)//2):
                    term = gamma(nu)*r**(2*k-nu)/(2*RisingFactorial(1-nu, k)*factorial(k))
                    a.append(_mexpand(term))
                for k in range((newn_b+1)//2):
                    term = gamma(-nu)*r**(2*k+nu)/(2*RisingFactorial(nu+1, k)*factorial(k))
                    b.append(_mexpand(term))
                return Add(*a) + Add(*b) + o
            else:
                raise NotImplementedError("besselk expansion is only implemented for real order")

        return super(besselk, self)._eval_nseries(x, n, logx, cdir)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.functions.combinatorial.factorials import RisingFactorial
        from sympy.series.order import Order
        point = args0[1]

        if point in [S.Infinity, S.NegativeInfinity]:
            nu, z = self.args
            s = [(RisingFactorial(Rational(2*nu - 1, 2), k)*RisingFactorial(Rational(2*nu + 1, 2), k))/\
            ((-2)**(k)*z**(Rational(2*k + 1, 2))*factorial(k)) for k in range(n)] +[Order(1/z**(Rational(2*n + 1, 2)), x)]
            return (exp(-z)*sqrt(pi/2))*Add(*s)

        return super()._eval_aseries(n, args0, x, logx)


class hankel1(BesselBase):
    r"""
    Hankel function of the first kind.

    Explanation
    ===========

    This function is defined as

    .. math ::
        H_\nu^{(1)} = J_\nu(z) + iY_\nu(z),

    where $J_\nu(z)$ is the Bessel function of the first kind, and
    $Y_\nu(z)$ is the Bessel function of the second kind.

    It is a solution to Bessel's equation.

    Examples
    ========

    >>> from sympy import hankel1
    >>> from sympy.abc import z, n
    >>> hankel1(n, z).diff(z)
    hankel1(n - 1, z)/2 - hankel1(n + 1, z)/2

    See Also
    ========

    hankel2, besselj, bessely

    References
    ==========

    .. [1] https://functions.wolfram.com/Bessel-TypeFunctions/HankelH1/

    """

    _a = S.One
    _b = S.One

    def _eval_conjugate(self):
        z = self.argument
        if z.is_extended_negative is False:
            return hankel2(self.order.conjugate(), z.conjugate())


class hankel2(BesselBase):
    r"""
    Hankel function of the second kind.

    Explanation
    ===========

    This function is defined as

    .. math ::
        H_\nu^{(2)} = J_\nu(z) - iY_\nu(z),

    where $J_\nu(z)$ is the Bessel function of the first kind, and
    $Y_\nu(z)$ is the Bessel function of the second kind.

    It is a solution to Bessel's equation, and linearly independent from
    $H_\nu^{(1)}$.

    Examples
    ========

    >>> from sympy import hankel2
    >>> from sympy.abc import z, n
    >>> hankel2(n, z).diff(z)
    hankel2(n - 1, z)/2 - hankel2(n + 1, z)/2

    See Also
    ========

    hankel1, besselj, bessely

    References
    ==========

    .. [1] https://functions.wolfram.com/Bessel-TypeFunctions/HankelH2/

    """

    _a = S.One
    _b = S.One

    def _eval_conjugate(self):
        z = self.argument
        if z.is_extended_negative is False:
            return hankel1(self.order.conjugate(), z.conjugate())


def assume_integer_order(fn):
    @wraps(fn)
    def g(self, nu, z):
        if nu.is_integer:
            return fn(self, nu, z)
    return g


class SphericalBesselBase(BesselBase):
    """
    Base class for spherical Bessel functions.

    These are thin wrappers around ordinary Bessel functions,
    since spherical Bessel functions differ from the ordinary
    ones just by a slight change in order.

    To use this class, define the ``_eval_evalf()`` and ``_expand()`` methods.

    """

    def _expand(self, **hints):
        """ Expand self into a polynomial. Nu is guaranteed to be Integer. """
        raise NotImplementedError('expansion')

    def _eval_expand_func(self, **hints):
        if self.order.is_Integer:
            return self._expand(**hints)
        return self

    def fdiff(self, argindex=2):
        if argindex != 2:
            raise ArgumentIndexError(self, argindex)
        return self.__class__(self.order - 1, self.argument) - \
            self * (self.order + 1)/self.argument


def _jn(n, z):
    return (spherical_bessel_fn(n, z)*sin(z) +
            S.NegativeOne**(n + 1)*spherical_bessel_fn(-n - 1, z)*cos(z))


def _yn(n, z):
    # (-1)**(n + 1) * _jn(-n - 1, z)
    return (S.NegativeOne**(n + 1) * spherical_bessel_fn(-n - 1, z)*sin(z) -
            spherical_bessel_fn(n, z)*cos(z))


class jn(SphericalBesselBase):
    r"""
    Spherical Bessel function of the first kind.

    Explanation
    ===========

    This function is a solution to the spherical Bessel equation

    .. math ::
        z^2 \frac{\mathrm{d}^2 w}{\mathrm{d}z^2}
          + 2z \frac{\mathrm{d}w}{\mathrm{d}z} + (z^2 - \nu(\nu + 1)) w = 0.

    It can be defined as

    .. math ::
        j_\nu(z) = \sqrt{\frac{\pi}{2z}} J_{\nu + \frac{1}{2}}(z),

    where $J_\nu(z)$ is the Bessel function of the first kind.

    The spherical Bessel functions of integral order are
    calculated using the formula:

    .. math:: j_n(z) = f_n(z) \sin{z} + (-1)^{n+1} f_{-n-1}(z) \cos{z},

    where the coefficients $f_n(z)$ are available as
    :func:`sympy.polys.orthopolys.spherical_bessel_fn`.

    Examples
    ========

    >>> from sympy import Symbol, jn, sin, cos, expand_func, besselj, bessely
    >>> z = Symbol("z")
    >>> nu = Symbol("nu", integer=True)
    >>> print(expand_func(jn(0, z)))
    sin(z)/z
    >>> expand_func(jn(1, z)) == sin(z)/z**2 - cos(z)/z
    True
    >>> expand_func(jn(3, z))
    (-6/z**2 + 15/z**4)*sin(z) + (1/z - 15/z**3)*cos(z)
    >>> jn(nu, z).rewrite(besselj)
    sqrt(2)*sqrt(pi)*sqrt(1/z)*besselj(nu + 1/2, z)/2
    >>> jn(nu, z).rewrite(bessely)
    (-1)**nu*sqrt(2)*sqrt(pi)*sqrt(1/z)*bessely(-nu - 1/2, z)/2
    >>> jn(2, 5.2+0.3j).evalf(20)
    0.099419756723640344491 - 0.054525080242173562897*I

    See Also
    ========

    besselj, bessely, besselk, yn

    References
    ==========

    .. [1] https://dlmf.nist.gov/10.47

    """
    @classmethod
    def eval(cls, nu, z):
        if z.is_zero:
            if nu.is_zero:
                return S.One
            elif nu.is_integer:
                if nu.is_positive:
                    return S.Zero
                else:
                    return S.ComplexInfinity
        if z in (S.NegativeInfinity, S.Infinity):
            return S.Zero

    def _eval_rewrite_as_besselj(self, nu, z, **kwargs):
        return sqrt(pi/(2*z)) * besselj(nu + S.Half, z)

    def _eval_rewrite_as_bessely(self, nu, z, **kwargs):
        return S.NegativeOne**nu * sqrt(pi/(2*z)) * bessely(-nu - S.Half, z)

    def _eval_rewrite_as_yn(self, nu, z, **kwargs):
        return S.NegativeOne**(nu) * yn(-nu - 1, z)

    def _expand(self, **hints):
        return _jn(self.order, self.argument)

    def _eval_evalf(self, prec):
        if self.order.is_Integer:
            return self.rewrite(besselj)._eval_evalf(prec)


class yn(SphericalBesselBase):
    r"""
    Spherical Bessel function of the second kind.

    Explanation
    ===========

    This function is another solution to the spherical Bessel equation, and
    linearly independent from $j_n$. It can be defined as

    .. math ::
        y_\nu(z) = \sqrt{\frac{\pi}{2z}} Y_{\nu + \frac{1}{2}}(z),

    where $Y_\nu(z)$ is the Bessel function of the second kind.

    For integral orders $n$, $y_n$ is calculated using the formula:

    .. math:: y_n(z) = (-1)^{n+1} j_{-n-1}(z)

    Examples
    ========

    >>> from sympy import Symbol, yn, sin, cos, expand_func, besselj, bessely
    >>> z = Symbol("z")
    >>> nu = Symbol("nu", integer=True)
    >>> print(expand_func(yn(0, z)))
    -cos(z)/z
    >>> expand_func(yn(1, z)) == -cos(z)/z**2-sin(z)/z
    True
    >>> yn(nu, z).rewrite(besselj)
    (-1)**(nu + 1)*sqrt(2)*sqrt(pi)*sqrt(1/z)*besselj(-nu - 1/2, z)/2
    >>> yn(nu, z).rewrite(bessely)
    sqrt(2)*sqrt(pi)*sqrt(1/z)*bessely(nu + 1/2, z)/2
    >>> yn(2, 5.2+0.3j).evalf(20)
    0.18525034196069722536 + 0.014895573969924817587*I

    See Also
    ========

    besselj, bessely, besselk, jn

    References
    ==========

    .. [1] https://dlmf.nist.gov/10.47

    """
    @assume_integer_order
    def _eval_rewrite_as_besselj(self, nu, z, **kwargs):
        return S.NegativeOne**(nu+1) * sqrt(pi/(2*z)) * besselj(-nu - S.Half, z)

    @assume_integer_order
    def _eval_rewrite_as_bessely(self, nu, z, **kwargs):
        return sqrt(pi/(2*z)) * bessely(nu + S.Half, z)

    def _eval_rewrite_as_jn(self, nu, z, **kwargs):
        return S.NegativeOne**(nu + 1) * jn(-nu - 1, z)

    def _expand(self, **hints):
        return _yn(self.order, self.argument)

    def _eval_evalf(self, prec):
        if self.order.is_Integer:
            return self.rewrite(bessely)._eval_evalf(prec)


class SphericalHankelBase(SphericalBesselBase):

    @assume_integer_order
    def _eval_rewrite_as_besselj(self, nu, z, **kwargs):
        # jn +- I*yn
        # jn as beeselj: sqrt(pi/(2*z)) * besselj(nu + S.Half, z)
        # yn as besselj: (-1)**(nu+1) * sqrt(pi/(2*z)) * besselj(-nu - S.Half, z)
        hks = self._hankel_kind_sign
        return sqrt(pi/(2*z))*(besselj(nu + S.Half, z) +
                               hks*I*S.NegativeOne**(nu+1)*besselj(-nu - S.Half, z))

    @assume_integer_order
    def _eval_rewrite_as_bessely(self, nu, z, **kwargs):
        # jn +- I*yn
        # jn as bessely: (-1)**nu * sqrt(pi/(2*z)) * bessely(-nu - S.Half, z)
        # yn as bessely: sqrt(pi/(2*z)) * bessely(nu + S.Half, z)
        hks = self._hankel_kind_sign
        return sqrt(pi/(2*z))*(S.NegativeOne**nu*bessely(-nu - S.Half, z) +
                               hks*I*bessely(nu + S.Half, z))

    def _eval_rewrite_as_yn(self, nu, z, **kwargs):
        hks = self._hankel_kind_sign
        return jn(nu, z).rewrite(yn) + hks*I*yn(nu, z)

    def _eval_rewrite_as_jn(self, nu, z, **kwargs):
        hks = self._hankel_kind_sign
        return jn(nu, z) + hks*I*yn(nu, z).rewrite(jn)

    def _eval_expand_func(self, **hints):
        if self.order.is_Integer:
            return self._expand(**hints)
        else:
            nu = self.order
            z = self.argument
            hks = self._hankel_kind_sign
            return jn(nu, z) + hks*I*yn(nu, z)

    def _expand(self, **hints):
        n = self.order
        z = self.argument
        hks = self._hankel_kind_sign

        # fully expanded version
        # return ((fn(n, z) * sin(z) +
        #          (-1)**(n + 1) * fn(-n - 1, z) * cos(z)) +  # jn
        #         (hks * I * (-1)**(n + 1) *
        #          (fn(-n - 1, z) * hk * I * sin(z) +
        #           (-1)**(-n) * fn(n, z) * I * cos(z)))  # +-I*yn
        #         )

        return (_jn(n, z) + hks*I*_yn(n, z)).expand()

    def _eval_evalf(self, prec):
        if self.order.is_Integer:
            return self.rewrite(besselj)._eval_evalf(prec)


class hn1(SphericalHankelBase):
    r"""
    Spherical Hankel function of the first kind.

    Explanation
    ===========

    This function is defined as

    .. math:: h_\nu^(1)(z) = j_\nu(z) + i y_\nu(z),

    where $j_\nu(z)$ and $y_\nu(z)$ are the spherical
    Bessel function of the first and second kinds.

    For integral orders $n$, $h_n^(1)$ is calculated using the formula:

    .. math:: h_n^(1)(z) = j_{n}(z) + i (-1)^{n+1} j_{-n-1}(z)

    Examples
    ========

    >>> from sympy import Symbol, hn1, hankel1, expand_func, yn, jn
    >>> z = Symbol("z")
    >>> nu = Symbol("nu", integer=True)
    >>> print(expand_func(hn1(nu, z)))
    jn(nu, z) + I*yn(nu, z)
    >>> print(expand_func(hn1(0, z)))
    sin(z)/z - I*cos(z)/z
    >>> print(expand_func(hn1(1, z)))
    -I*sin(z)/z - cos(z)/z + sin(z)/z**2 - I*cos(z)/z**2
    >>> hn1(nu, z).rewrite(jn)
    (-1)**(nu + 1)*I*jn(-nu - 1, z) + jn(nu, z)
    >>> hn1(nu, z).rewrite(yn)
    (-1)**nu*yn(-nu - 1, z) + I*yn(nu, z)
    >>> hn1(nu, z).rewrite(hankel1)
    sqrt(2)*sqrt(pi)*sqrt(1/z)*hankel1(nu, z)/2

    See Also
    ========

    hn2, jn, yn, hankel1, hankel2

    References
    ==========

    .. [1] https://dlmf.nist.gov/10.47

    """

    _hankel_kind_sign = S.One

    @assume_integer_order
    def _eval_rewrite_as_hankel1(self, nu, z, **kwargs):
        return sqrt(pi/(2*z))*hankel1(nu, z)


class hn2(SphericalHankelBase):
    r"""
    Spherical Hankel function of the second kind.

    Explanation
    ===========

    This function is defined as

    .. math:: h_\nu^(2)(z) = j_\nu(z) - i y_\nu(z),

    where $j_\nu(z)$ and $y_\nu(z)$ are the spherical
    Bessel function of the first and second kinds.

    For integral orders $n$, $h_n^(2)$ is calculated using the formula:

    .. math:: h_n^(2)(z) = j_{n} - i (-1)^{n+1} j_{-n-1}(z)

    Examples
    ========

    >>> from sympy import Symbol, hn2, hankel2, expand_func, jn, yn
    >>> z = Symbol("z")
    >>> nu = Symbol("nu", integer=True)
    >>> print(expand_func(hn2(nu, z)))
    jn(nu, z) - I*yn(nu, z)
    >>> print(expand_func(hn2(0, z)))
    sin(z)/z + I*cos(z)/z
    >>> print(expand_func(hn2(1, z)))
    I*sin(z)/z - cos(z)/z + sin(z)/z**2 + I*cos(z)/z**2
    >>> hn2(nu, z).rewrite(hankel2)
    sqrt(2)*sqrt(pi)*sqrt(1/z)*hankel2(nu, z)/2
    >>> hn2(nu, z).rewrite(jn)
    -(-1)**(nu + 1)*I*jn(-nu - 1, z) + jn(nu, z)
    >>> hn2(nu, z).rewrite(yn)
    (-1)**nu*yn(-nu - 1, z) - I*yn(nu, z)

    See Also
    ========

    hn1, jn, yn, hankel1, hankel2

    References
    ==========

    .. [1] https://dlmf.nist.gov/10.47

    """

    _hankel_kind_sign = -S.One

    @assume_integer_order
    def _eval_rewrite_as_hankel2(self, nu, z, **kwargs):
        return sqrt(pi/(2*z))*hankel2(nu, z)


def jn_zeros(n, k, method="sympy", dps=15):
    """
    Zeros of the spherical Bessel function of the first kind.

    Explanation
    ===========

    This returns an array of zeros of $jn$ up to the $k$-th zero.

    * method = "sympy": uses `mpmath.besseljzero
      <https://mpmath.org/doc/current/functions/bessel.html#mpmath.besseljzero>`_
    * method = "scipy": uses the
      `SciPy's sph_jn <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jn_zeros.html>`_
      and
      `newton <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html>`_
      to find all
      roots, which is faster than computing the zeros using a general
      numerical solver, but it requires SciPy and only works with low
      precision floating point numbers. (The function used with
      method="sympy" is a recent addition to mpmath; before that a general
      solver was used.)

    Examples
    ========

    >>> from sympy import jn_zeros
    >>> jn_zeros(2, 4, dps=5)
    [5.7635, 9.095, 12.323, 15.515]

    See Also
    ========

    jn, yn, besselj, besselk, bessely

    Parameters
    ==========

    n : integer
        order of Bessel function

    k : integer
        number of zeros to return


    """
    from math import pi as math_pi

    if method == "sympy":
        from mpmath import besseljzero
        from mpmath.libmp.libmpf import dps_to_prec
        prec = dps_to_prec(dps)
        return [Expr._from_mpmath(besseljzero(S(n + 0.5)._to_mpmath(prec),
                                              int(l)), prec)
                for l in range(1, k + 1)]
    elif method == "scipy":
        from scipy.optimize import newton
        try:
            from scipy.special import spherical_jn
            f = lambda x: spherical_jn(n, x)
        except ImportError:
            from scipy.special import sph_jn
            f = lambda x: sph_jn(n, x)[0][-1]
    else:
        raise NotImplementedError("Unknown method.")

    def solver(f, x):
        if method == "scipy":
            root = newton(f, x)
        else:
            raise NotImplementedError("Unknown method.")
        return root

    # we need to approximate the position of the first root:
    root = n + math_pi
    # determine the first root exactly:
    root = solver(f, root)
    roots = [root]
    for i in range(k - 1):
        # estimate the position of the next root using the last root + pi:
        root = solver(f, root + math_pi)
        roots.append(root)
    return roots


class AiryBase(DefinedFunction):
    """
    Abstract base class for Airy functions.

    This class is meant to reduce code duplication.

    """

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real

    def as_real_imag(self, deep=True, **hints):
        z = self.args[0]
        zc = z.conjugate()
        f = self.func
        u = (f(z)+f(zc))/2
        v = I*(f(zc)-f(z))/2
        return u, v

    def _eval_expand_complex(self, deep=True, **hints):
        re_part, im_part = self.as_real_imag(deep=deep, **hints)
        return re_part + im_part*I


class airyai(AiryBase):
    r"""
    The Airy function $\operatorname{Ai}$ of the first kind.

    Explanation
    ===========

    The Airy function $\operatorname{Ai}(z)$ is defined to be the function
    satisfying Airy's differential equation

    .. math::
        \frac{\mathrm{d}^2 w(z)}{\mathrm{d}z^2} - z w(z) = 0.

    Equivalently, for real $z$

    .. math::
        \operatorname{Ai}(z) := \frac{1}{\pi}
        \int_0^\infty \cos\left(\frac{t^3}{3} + z t\right) \mathrm{d}t.

    Examples
    ========

    Create an Airy function object:

    >>> from sympy import airyai
    >>> from sympy.abc import z

    >>> airyai(z)
    airyai(z)

    Several special values are known:

    >>> airyai(0)
    3**(1/3)/(3*gamma(2/3))
    >>> from sympy import oo
    >>> airyai(oo)
    0
    >>> airyai(-oo)
    0

    The Airy function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(airyai(z))
    airyai(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(airyai(z), z)
    airyaiprime(z)
    >>> diff(airyai(z), z, 2)
    z*airyai(z)

    Series expansion is also supported:

    >>> from sympy import series
    >>> series(airyai(z), z, 0, 3)
    3**(5/6)*gamma(1/3)/(6*pi) - 3**(1/6)*z*gamma(2/3)/(2*pi) + O(z**3)

    We can numerically evaluate the Airy function to arbitrary precision
    on the whole complex plane:

    >>> airyai(-2).evalf(50)
    0.22740742820168557599192443603787379946077222541710

    Rewrite $\operatorname{Ai}(z)$ in terms of hypergeometric functions:

    >>> from sympy import hyper
    >>> airyai(z).rewrite(hyper)
    -3**(2/3)*z*hyper((), (4/3,), z**3/9)/(3*gamma(1/3)) + 3**(1/3)*hyper((), (2/3,), z**3/9)/(3*gamma(2/3))

    See Also
    ========

    airybi: Airy function of the second kind.
    airyaiprime: Derivative of the Airy function of the first kind.
    airybiprime: Derivative of the Airy function of the second kind.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Airy_function
    .. [2] https://dlmf.nist.gov/9
    .. [3] https://encyclopediaofmath.org/wiki/Airy_functions
    .. [4] https://mathworld.wolfram.com/AiryFunctions.html

    """

    nargs = 1
    unbranched = True

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Zero
            elif arg is S.NegativeInfinity:
                return S.Zero
            elif arg.is_zero:
                return S.One / (3**Rational(2, 3) * gamma(Rational(2, 3)))
        if arg.is_zero:
            return S.One / (3**Rational(2, 3) * gamma(Rational(2, 3)))

    def fdiff(self, argindex=1):
        if argindex == 1:
            return airyaiprime(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 1:
                p = previous_terms[-1]
                return ((cbrt(3)*x)**(-n)*(cbrt(3)*x)**(n + 1)*sin(pi*(n*Rational(2, 3) + Rational(4, 3)))*factorial(n) *
                        gamma(n/3 + Rational(2, 3))/(sin(pi*(n*Rational(2, 3) + Rational(2, 3)))*factorial(n + 1)*gamma(n/3 + Rational(1, 3))) * p)
            else:
                return (S.One/(3**Rational(2, 3)*pi) * gamma((n+S.One)/S(3)) * sin(Rational(2, 3)*pi*(n+S.One)) /
                        factorial(n) * (cbrt(3)*x)**n)

    def _eval_rewrite_as_besselj(self, z, **kwargs):
        ot = Rational(1, 3)
        tt = Rational(2, 3)
        a = Pow(-z, Rational(3, 2))
        if re(z).is_negative:
            return ot*sqrt(-z) * (besselj(-ot, tt*a) + besselj(ot, tt*a))

    def _eval_rewrite_as_besseli(self, z, **kwargs):
        ot = Rational(1, 3)
        tt = Rational(2, 3)
        a = Pow(z, Rational(3, 2))
        if re(z).is_positive:
            return ot*sqrt(z) * (besseli(-ot, tt*a) - besseli(ot, tt*a))
        else:
            return ot*(Pow(a, ot)*besseli(-ot, tt*a) - z*Pow(a, -ot)*besseli(ot, tt*a))

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        pf1 = S.One / (3**Rational(2, 3)*gamma(Rational(2, 3)))
        pf2 = z / (root(3, 3)*gamma(Rational(1, 3)))
        return pf1 * hyper([], [Rational(2, 3)], z**3/9) - pf2 * hyper([], [Rational(4, 3)], z**3/9)

    def _eval_expand_func(self, **hints):
        arg = self.args[0]
        symbs = arg.free_symbols

        if len(symbs) == 1:
            z = symbs.pop()
            c = Wild("c", exclude=[z])
            d = Wild("d", exclude=[z])
            m = Wild("m", exclude=[z])
            n = Wild("n", exclude=[z])
            M = arg.match(c*(d*z**n)**m)
            if M is not None:
                m = M[m]
                # The transformation is given by 03.05.16.0001.01
                # https://functions.wolfram.com/Bessel-TypeFunctions/AiryAi/16/01/01/0001/
                if (3*m).is_integer:
                    c = M[c]
                    d = M[d]
                    n = M[n]
                    pf = (d * z**n)**m / (d**m * z**(m*n))
                    newarg = c * d**m * z**(m*n)
                    return S.Half * ((pf + S.One)*airyai(newarg) - (pf - S.One)/sqrt(3)*airybi(newarg))


class airybi(AiryBase):
    r"""
    The Airy function $\operatorname{Bi}$ of the second kind.

    Explanation
    ===========

    The Airy function $\operatorname{Bi}(z)$ is defined to be the function
    satisfying Airy's differential equation

    .. math::
        \frac{\mathrm{d}^2 w(z)}{\mathrm{d}z^2} - z w(z) = 0.

    Equivalently, for real $z$

    .. math::
        \operatorname{Bi}(z) := \frac{1}{\pi}
                 \int_0^\infty
                   \exp\left(-\frac{t^3}{3} + z t\right)
                   + \sin\left(\frac{t^3}{3} + z t\right) \mathrm{d}t.

    Examples
    ========

    Create an Airy function object:

    >>> from sympy import airybi
    >>> from sympy.abc import z

    >>> airybi(z)
    airybi(z)

    Several special values are known:

    >>> airybi(0)
    3**(5/6)/(3*gamma(2/3))
    >>> from sympy import oo
    >>> airybi(oo)
    oo
    >>> airybi(-oo)
    0

    The Airy function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(airybi(z))
    airybi(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(airybi(z), z)
    airybiprime(z)
    >>> diff(airybi(z), z, 2)
    z*airybi(z)

    Series expansion is also supported:

    >>> from sympy import series
    >>> series(airybi(z), z, 0, 3)
    3**(1/3)*gamma(1/3)/(2*pi) + 3**(2/3)*z*gamma(2/3)/(2*pi) + O(z**3)

    We can numerically evaluate the Airy function to arbitrary precision
    on the whole complex plane:

    >>> airybi(-2).evalf(50)
    -0.41230258795639848808323405461146104203453483447240

    Rewrite $\operatorname{Bi}(z)$ in terms of hypergeometric functions:

    >>> from sympy import hyper
    >>> airybi(z).rewrite(hyper)
    3**(1/6)*z*hyper((), (4/3,), z**3/9)/gamma(1/3) + 3**(5/6)*hyper((), (2/3,), z**3/9)/(3*gamma(2/3))

    See Also
    ========

    airyai: Airy function of the first kind.
    airyaiprime: Derivative of the Airy function of the first kind.
    airybiprime: Derivative of the Airy function of the second kind.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Airy_function
    .. [2] https://dlmf.nist.gov/9
    .. [3] https://encyclopediaofmath.org/wiki/Airy_functions
    .. [4] https://mathworld.wolfram.com/AiryFunctions.html

    """

    nargs = 1
    unbranched = True

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.Zero
            elif arg.is_zero:
                return S.One / (3**Rational(1, 6) * gamma(Rational(2, 3)))

        if arg.is_zero:
            return S.One / (3**Rational(1, 6) * gamma(Rational(2, 3)))

    def fdiff(self, argindex=1):
        if argindex == 1:
            return airybiprime(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 1:
                p = previous_terms[-1]
                return (cbrt(3)*x * Abs(sin(Rational(2, 3)*pi*(n + S.One))) * factorial((n - S.One)/S(3)) /
                        ((n + S.One) * Abs(cos(Rational(2, 3)*pi*(n + S.Half))) * factorial((n - 2)/S(3))) * p)
            else:
                return (S.One/(root(3, 6)*pi) * gamma((n + S.One)/S(3)) * Abs(sin(Rational(2, 3)*pi*(n + S.One))) /
                        factorial(n) * (cbrt(3)*x)**n)

    def _eval_rewrite_as_besselj(self, z, **kwargs):
        ot = Rational(1, 3)
        tt = Rational(2, 3)
        a = Pow(-z, Rational(3, 2))
        if re(z).is_negative:
            return sqrt(-z/3) * (besselj(-ot, tt*a) - besselj(ot, tt*a))

    def _eval_rewrite_as_besseli(self, z, **kwargs):
        ot = Rational(1, 3)
        tt = Rational(2, 3)
        a = Pow(z, Rational(3, 2))
        if re(z).is_positive:
            return sqrt(z)/sqrt(3) * (besseli(-ot, tt*a) + besseli(ot, tt*a))
        else:
            b = Pow(a, ot)
            c = Pow(a, -ot)
            return sqrt(ot)*(b*besseli(-ot, tt*a) + z*c*besseli(ot, tt*a))

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        pf1 = S.One / (root(3, 6)*gamma(Rational(2, 3)))
        pf2 = z*root(3, 6) / gamma(Rational(1, 3))
        return pf1 * hyper([], [Rational(2, 3)], z**3/9) + pf2 * hyper([], [Rational(4, 3)], z**3/9)

    def _eval_expand_func(self, **hints):
        arg = self.args[0]
        symbs = arg.free_symbols

        if len(symbs) == 1:
            z = symbs.pop()
            c = Wild("c", exclude=[z])
            d = Wild("d", exclude=[z])
            m = Wild("m", exclude=[z])
            n = Wild("n", exclude=[z])
            M = arg.match(c*(d*z**n)**m)
            if M is not None:
                m = M[m]
                # The transformation is given by 03.06.16.0001.01
                # https://functions.wolfram.com/Bessel-TypeFunctions/AiryBi/16/01/01/0001/
                if (3*m).is_integer:
                    c = M[c]
                    d = M[d]
                    n = M[n]
                    pf = (d * z**n)**m / (d**m * z**(m*n))
                    newarg = c * d**m * z**(m*n)
                    return S.Half * (sqrt(3)*(S.One - pf)*airyai(newarg) + (S.One + pf)*airybi(newarg))


class airyaiprime(AiryBase):
    r"""
    The derivative $\operatorname{Ai}^\prime$ of the Airy function of the first
    kind.

    Explanation
    ===========

    The Airy function $\operatorname{Ai}^\prime(z)$ is defined to be the
    function

    .. math::
        \operatorname{Ai}^\prime(z) := \frac{\mathrm{d} \operatorname{Ai}(z)}{\mathrm{d} z}.

    Examples
    ========

    Create an Airy function object:

    >>> from sympy import airyaiprime
    >>> from sympy.abc import z

    >>> airyaiprime(z)
    airyaiprime(z)

    Several special values are known:

    >>> airyaiprime(0)
    -3**(2/3)/(3*gamma(1/3))
    >>> from sympy import oo
    >>> airyaiprime(oo)
    0

    The Airy function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(airyaiprime(z))
    airyaiprime(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(airyaiprime(z), z)
    z*airyai(z)
    >>> diff(airyaiprime(z), z, 2)
    z*airyaiprime(z) + airyai(z)

    Series expansion is also supported:

    >>> from sympy import series
    >>> series(airyaiprime(z), z, 0, 3)
    -3**(2/3)/(3*gamma(1/3)) + 3**(1/3)*z**2/(6*gamma(2/3)) + O(z**3)

    We can numerically evaluate the Airy function to arbitrary precision
    on the whole complex plane:

    >>> airyaiprime(-2).evalf(50)
    0.61825902074169104140626429133247528291577794512415

    Rewrite $\operatorname{Ai}^\prime(z)$ in terms of hypergeometric functions:

    >>> from sympy import hyper
    >>> airyaiprime(z).rewrite(hyper)
    3**(1/3)*z**2*hyper((), (5/3,), z**3/9)/(6*gamma(2/3)) - 3**(2/3)*hyper((), (1/3,), z**3/9)/(3*gamma(1/3))

    See Also
    ========

    airyai: Airy function of the first kind.
    airybi: Airy function of the second kind.
    airybiprime: Derivative of the Airy function of the second kind.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Airy_function
    .. [2] https://dlmf.nist.gov/9
    .. [3] https://encyclopediaofmath.org/wiki/Airy_functions
    .. [4] https://mathworld.wolfram.com/AiryFunctions.html

    """

    nargs = 1
    unbranched = True

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Zero

        if arg.is_zero:
            return S.NegativeOne / (3**Rational(1, 3) * gamma(Rational(1, 3)))

    def fdiff(self, argindex=1):
        if argindex == 1:
            return self.args[0]*airyai(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_evalf(self, prec):
        z = self.args[0]._to_mpmath(prec)
        with workprec(prec):
            res = mp.airyai(z, derivative=1)
        return Expr._from_mpmath(res, prec)

    def _eval_rewrite_as_besselj(self, z, **kwargs):
        tt = Rational(2, 3)
        a = Pow(-z, Rational(3, 2))
        if re(z).is_negative:
            return z/3 * (besselj(-tt, tt*a) - besselj(tt, tt*a))

    def _eval_rewrite_as_besseli(self, z, **kwargs):
        ot = Rational(1, 3)
        tt = Rational(2, 3)
        a = tt * Pow(z, Rational(3, 2))
        if re(z).is_positive:
            return z/3 * (besseli(tt, a) - besseli(-tt, a))
        else:
            a = Pow(z, Rational(3, 2))
            b = Pow(a, tt)
            c = Pow(a, -tt)
            return ot * (z**2*c*besseli(tt, tt*a) - b*besseli(-ot, tt*a))

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        pf1 = z**2 / (2*3**Rational(2, 3)*gamma(Rational(2, 3)))
        pf2 = 1 / (root(3, 3)*gamma(Rational(1, 3)))
        return pf1 * hyper([], [Rational(5, 3)], z**3/9) - pf2 * hyper([], [Rational(1, 3)], z**3/9)

    def _eval_expand_func(self, **hints):
        arg = self.args[0]
        symbs = arg.free_symbols

        if len(symbs) == 1:
            z = symbs.pop()
            c = Wild("c", exclude=[z])
            d = Wild("d", exclude=[z])
            m = Wild("m", exclude=[z])
            n = Wild("n", exclude=[z])
            M = arg.match(c*(d*z**n)**m)
            if M is not None:
                m = M[m]
                # The transformation is in principle
                # given by 03.07.16.0001.01 but note
                # that there is an error in this formula.
                # https://functions.wolfram.com/Bessel-TypeFunctions/AiryAiPrime/16/01/01/0001/
                if (3*m).is_integer:
                    c = M[c]
                    d = M[d]
                    n = M[n]
                    pf = (d**m * z**(n*m)) / (d * z**n)**m
                    newarg = c * d**m * z**(n*m)
                    return S.Half * ((pf + S.One)*airyaiprime(newarg) + (pf - S.One)/sqrt(3)*airybiprime(newarg))


class airybiprime(AiryBase):
    r"""
    The derivative $\operatorname{Bi}^\prime$ of the Airy function of the first
    kind.

    Explanation
    ===========

    The Airy function $\operatorname{Bi}^\prime(z)$ is defined to be the
    function

    .. math::
        \operatorname{Bi}^\prime(z) := \frac{\mathrm{d} \operatorname{Bi}(z)}{\mathrm{d} z}.

    Examples
    ========

    Create an Airy function object:

    >>> from sympy import airybiprime
    >>> from sympy.abc import z

    >>> airybiprime(z)
    airybiprime(z)

    Several special values are known:

    >>> airybiprime(0)
    3**(1/6)/gamma(1/3)
    >>> from sympy import oo
    >>> airybiprime(oo)
    oo
    >>> airybiprime(-oo)
    0

    The Airy function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(airybiprime(z))
    airybiprime(conjugate(z))

    Differentiation with respect to $z$ is supported:

    >>> from sympy import diff
    >>> diff(airybiprime(z), z)
    z*airybi(z)
    >>> diff(airybiprime(z), z, 2)
    z*airybiprime(z) + airybi(z)

    Series expansion is also supported:

    >>> from sympy import series
    >>> series(airybiprime(z), z, 0, 3)
    3**(1/6)/gamma(1/3) + 3**(5/6)*z**2/(6*gamma(2/3)) + O(z**3)

    We can numerically evaluate the Airy function to arbitrary precision
    on the whole complex plane:

    >>> airybiprime(-2).evalf(50)
    0.27879516692116952268509756941098324140300059345163

    Rewrite $\operatorname{Bi}^\prime(z)$ in terms of hypergeometric functions:

    >>> from sympy import hyper
    >>> airybiprime(z).rewrite(hyper)
    3**(5/6)*z**2*hyper((), (5/3,), z**3/9)/(6*gamma(2/3)) + 3**(1/6)*hyper((), (1/3,), z**3/9)/gamma(1/3)

    See Also
    ========

    airyai: Airy function of the first kind.
    airybi: Airy function of the second kind.
    airyaiprime: Derivative of the Airy function of the first kind.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Airy_function
    .. [2] https://dlmf.nist.gov/9
    .. [3] https://encyclopediaofmath.org/wiki/Airy_functions
    .. [4] https://mathworld.wolfram.com/AiryFunctions.html

    """

    nargs = 1
    unbranched = True

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.Zero
            elif arg.is_zero:
                return 3**Rational(1, 6) / gamma(Rational(1, 3))

        if arg.is_zero:
            return 3**Rational(1, 6) / gamma(Rational(1, 3))


    def fdiff(self, argindex=1):
        if argindex == 1:
            return self.args[0]*airybi(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_evalf(self, prec):
        z = self.args[0]._to_mpmath(prec)
        with workprec(prec):
            res = mp.airybi(z, derivative=1)
        return Expr._from_mpmath(res, prec)

    def _eval_rewrite_as_besselj(self, z, **kwargs):
        tt = Rational(2, 3)
        a = tt * Pow(-z, Rational(3, 2))
        if re(z).is_negative:
            return -z/sqrt(3) * (besselj(-tt, a) + besselj(tt, a))

    def _eval_rewrite_as_besseli(self, z, **kwargs):
        ot = Rational(1, 3)
        tt = Rational(2, 3)
        a = tt * Pow(z, Rational(3, 2))
        if re(z).is_positive:
            return z/sqrt(3) * (besseli(-tt, a) + besseli(tt, a))
        else:
            a = Pow(z, Rational(3, 2))
            b = Pow(a, tt)
            c = Pow(a, -tt)
            return sqrt(ot) * (b*besseli(-tt, tt*a) + z**2*c*besseli(tt, tt*a))

    def _eval_rewrite_as_hyper(self, z, **kwargs):
        pf1 = z**2 / (2*root(3, 6)*gamma(Rational(2, 3)))
        pf2 = root(3, 6) / gamma(Rational(1, 3))
        return pf1 * hyper([], [Rational(5, 3)], z**3/9) + pf2 * hyper([], [Rational(1, 3)], z**3/9)

    def _eval_expand_func(self, **hints):
        arg = self.args[0]
        symbs = arg.free_symbols

        if len(symbs) == 1:
            z = symbs.pop()
            c = Wild("c", exclude=[z])
            d = Wild("d", exclude=[z])
            m = Wild("m", exclude=[z])
            n = Wild("n", exclude=[z])
            M = arg.match(c*(d*z**n)**m)
            if M is not None:
                m = M[m]
                # The transformation is in principle
                # given by 03.08.16.0001.01 but note
                # that there is an error in this formula.
                # https://functions.wolfram.com/Bessel-TypeFunctions/AiryBiPrime/16/01/01/0001/
                if (3*m).is_integer:
                    c = M[c]
                    d = M[d]
                    n = M[n]
                    pf = (d**m * z**(n*m)) / (d * z**n)**m
                    newarg = c * d**m * z**(n*m)
                    return S.Half * (sqrt(3)*(pf - S.One)*airyaiprime(newarg) + (pf + S.One)*airybiprime(newarg))


class marcumq(DefinedFunction):
    r"""
    The Marcum Q-function.

    Explanation
    ===========

    The Marcum Q-function is defined by the meromorphic continuation of

    .. math::
        Q_m(a, b) = a^{- m + 1} \int_{b}^{\infty} x^{m} e^{- \frac{a^{2}}{2} - \frac{x^{2}}{2}} I_{m - 1}\left(a x\right)\, dx

    Examples
    ========

    >>> from sympy import marcumq
    >>> from sympy.abc import m, a, b
    >>> marcumq(m, a, b)
    marcumq(m, a, b)

    Special values:

    >>> marcumq(m, 0, b)
    uppergamma(m, b**2/2)/gamma(m)
    >>> marcumq(0, 0, 0)
    0
    >>> marcumq(0, a, 0)
    1 - exp(-a**2/2)
    >>> marcumq(1, a, a)
    1/2 + exp(-a**2)*besseli(0, a**2)/2
    >>> marcumq(2, a, a)
    1/2 + exp(-a**2)*besseli(0, a**2)/2 + exp(-a**2)*besseli(1, a**2)

    Differentiation with respect to $a$ and $b$ is supported:

    >>> from sympy import diff
    >>> diff(marcumq(m, a, b), a)
    a*(-marcumq(m, a, b) + marcumq(m + 1, a, b))
    >>> diff(marcumq(m, a, b), b)
    -a**(1 - m)*b**m*exp(-a**2/2 - b**2/2)*besseli(m - 1, a*b)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Marcum_Q-function
    .. [2] https://mathworld.wolfram.com/MarcumQ-Function.html

    """

    @classmethod
    def eval(cls, m, a, b):
        if a is S.Zero:
            if m is S.Zero and b is S.Zero:
                return S.Zero
            return uppergamma(m, b**2 * S.Half) / gamma(m)

        if m is S.Zero and b is S.Zero:
            return 1 - 1 / exp(a**2 * S.Half)

        if a == b:
            if m is S.One:
                return (1 + exp(-a**2) * besseli(0, a**2))*S.Half
            if m == 2:
                return S.Half + S.Half * exp(-a**2) * besseli(0, a**2) + exp(-a**2) * besseli(1, a**2)

        if a.is_zero:
            if m.is_zero and b.is_zero:
                return S.Zero
            return uppergamma(m, b**2*S.Half) / gamma(m)

        if m.is_zero and b.is_zero:
            return 1 - 1 / exp(a**2*S.Half)

    def fdiff(self, argindex=2):
        m, a, b = self.args
        if argindex == 2:
            return a * (-marcumq(m, a, b) + marcumq(1+m, a, b))
        elif argindex == 3:
            return (-b**m / a**(m-1)) * exp(-(a**2 + b**2)/2) * besseli(m-1, a*b)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Integral(self, m, a, b, **kwargs):
        from sympy.integrals.integrals import Integral
        x = kwargs.get('x', Dummy(uniquely_named_symbol('x').name))
        return a ** (1 - m) * \
               Integral(x**m * exp(-(x**2 + a**2)/2) * besseli(m-1, a*x), [x, b, S.Infinity])

    def _eval_rewrite_as_Sum(self, m, a, b, **kwargs):
        from sympy.concrete.summations import Sum
        k = kwargs.get('k', Dummy('k'))
        return exp(-(a**2 + b**2) / 2) * Sum((a/b)**k * besseli(k, a*b), [k, 1-m, S.Infinity])

    def _eval_rewrite_as_besseli(self, m, a, b, **kwargs):
        if a == b:
            if m == 1:
                return (1 + exp(-a**2) * besseli(0, a**2)) / 2
            if m.is_Integer and m >= 2:
                s = sum(besseli(i, a**2) for i in range(1, m))
                return S.Half + exp(-a**2) * besseli(0, a**2) / 2 + exp(-a**2) * s

    def _eval_is_zero(self):
        if all(arg.is_zero for arg in self.args):
            return True

class _besseli(DefinedFunction):
    """
    Helper function to make the $\\mathrm{besseli}(nu, z)$
    function tractable for the Gruntz algorithm.

    """

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.functions.combinatorial.factorials import RisingFactorial
        from sympy.series.order import Order
        point = args0[1]

        if point in [S.Infinity, S.NegativeInfinity]:
            nu, z = self.args
            l = [((RisingFactorial(Rational(2*nu - 1, 2), k)*RisingFactorial(
                    Rational(2*nu + 1, 2), k))/((2)**(k)*z**(Rational(2*k + 1, 2))*factorial(k))) for k in range(n)]
            return sqrt(pi/(2))*(Add(*l)) + Order(1/z**(Rational(2*n + 1, 2)), x)

        return super()._eval_aseries(n, args0, x, logx)

    def _eval_rewrite_as_intractable(self, nu, z, **kwargs):
        return exp(-z)*besseli(nu, z)

    def _eval_nseries(self, x, n, logx, cdir=0):
        x0 = self.args[0].limit(x, 0)
        if x0.is_zero:
            f = self._eval_rewrite_as_intractable(*self.args)
            return f._eval_nseries(x, n, logx)
        return super()._eval_nseries(x, n, logx)


class _besselk(DefinedFunction):
    """
    Helper function to make the $\\mathrm{besselk}(nu, z)$
    function tractable for the Gruntz algorithm.

    """

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.functions.combinatorial.factorials import RisingFactorial
        from sympy.series.order import Order
        point = args0[1]

        if point in [S.Infinity, S.NegativeInfinity]:
            nu, z = self.args
            l = [((RisingFactorial(Rational(2*nu - 1, 2), k)*RisingFactorial(
                    Rational(2*nu + 1, 2), k))/((-2)**(k)*z**(Rational(2*k + 1, 2))*factorial(k))) for k in range(n)]
            return sqrt(pi/(2))*(Add(*l)) + Order(1/z**(Rational(2*n + 1, 2)), x)

        return super()._eval_aseries(n, args0, x, logx)

    def _eval_rewrite_as_intractable(self,nu, z, **kwargs):
        return exp(z)*besselk(nu, z)

    def _eval_nseries(self, x, n, logx, cdir=0):
        x0 = self.args[0].limit(x, 0)
        if x0.is_zero:
            f = self._eval_rewrite_as_intractable(*self.args)
            return f._eval_nseries(x, n, logx)
        return super()._eval_nseries(x, n, logx)
<|endoftext|>from math import prod

from sympy.core import Add, S, Dummy, expand_func
from sympy.core.expr import Expr
from sympy.core.function import DefinedFunction, ArgumentIndexError, PoleError
from sympy.core.logic import fuzzy_and, fuzzy_not
from sympy.core.numbers import Rational, pi, oo, I
from sympy.core.power import Pow
from sympy.functions.special.zeta_functions import zeta
from sympy.functions.special.error_functions import erf, erfc, Ei
from sympy.functions.elementary.complexes import re, unpolarify
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin, cos, cot
from sympy.functions.combinatorial.numbers import bernoulli, harmonic
from sympy.functions.combinatorial.factorials import factorial, rf, RisingFactorial
from sympy.utilities.misc import as_int

from mpmath import mp, workprec
from mpmath.libmp.libmpf import prec_to_dps

def intlike(n):
    try:
        as_int(n, strict=False)
        return True
    except ValueError:
        return False

###############################################################################
############################ COMPLETE GAMMA FUNCTION ##########################
###############################################################################

class gamma(DefinedFunction):
    r"""
    The gamma function

    .. math::
        \Gamma(x) := \int^{\infty}_{0} t^{x-1} e^{-t} \mathrm{d}t.

    Explanation
    ===========

    The ``gamma`` function implements the function which passes through the
    values of the factorial function (i.e., $\Gamma(n) = (n - 1)!$ when n is
    an integer). More generally, $\Gamma(z)$ is defined in the whole complex
    plane except at the negative integers where there are simple poles.

    Examples
    ========

    >>> from sympy import S, I, pi, gamma
    >>> from sympy.abc import x

    Several special values are known:

    >>> gamma(1)
    1
    >>> gamma(4)
    6
    >>> gamma(S(3)/2)
    sqrt(pi)/2

    The ``gamma`` function obeys the mirror symmetry:

    >>> from sympy import conjugate
    >>> conjugate(gamma(x))
    gamma(conjugate(x))

    Differentiation with respect to $x$ is supported:

    >>> from sympy import diff
    >>> diff(gamma(x), x)
    gamma(x)*polygamma(0, x)

    Series expansion is also supported:

    >>> from sympy import series
    >>> series(gamma(x), x, 0, 3)
    1/x - EulerGamma + x*(EulerGamma**2/2 + pi**2/12) + x**2*(-EulerGamma*pi**2/12 - zeta(3)/3 - EulerGamma**3/6) + O(x**3)

    We can numerically evaluate the ``gamma`` function to arbitrary precision
    on the whole complex plane:

    >>> gamma(pi).evalf(40)
    2.288037795340032417959588909060233922890
    >>> gamma(1+I).evalf(20)
    0.49801566811835604271 - 0.15494982830181068512*I

    See Also
    ========

    lowergamma: Lower incomplete gamma function.
    uppergamma: Upper incomplete gamma function.
    polygamma: Polygamma function.
    loggamma: Log Gamma function.
    digamma: Digamma function.
    trigamma: Trigamma function.
    sympy.functions.special.beta_functions.beta: Euler Beta function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gamma_function
    .. [2] https://dlmf.nist.gov/5
    .. [3] https://mathworld.wolfram.com/GammaFunction.html
    .. [4] https://functions.wolfram.com/GammaBetaErf/Gamma/

    """

    unbranched = True
    _singularities = (S.ComplexInfinity,)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return self.func(self.args[0])*polygamma(0, self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is oo:
                return oo
            elif intlike(arg):
                if arg.is_positive:
                    return factorial(arg - 1)
                else:
                    return S.ComplexInfinity
            elif arg.is_Rational:
                if arg.q == 2:
                    n = abs(arg.p) // arg.q

                    if arg.is_positive:
                        k, coeff = n, S.One
                    else:
                        n = k = n + 1

                        if n & 1 == 0:
                            coeff = S.One
                        else:
                            coeff = S.NegativeOne

                    coeff *= prod(range(3, 2*k, 2))

                    if arg.is_positive:
                        return coeff*sqrt(pi) / 2**n
                    else:
                        return 2**n*sqrt(pi) / coeff

    def _eval_expand_func(self, **hints):
        arg = self.args[0]
        if arg.is_Rational:
            if abs(arg.p) > arg.q:
                x = Dummy('x')
                n = arg.p // arg.q
                p = arg.p - n*arg.q
                return self.func(x + n)._eval_expand_func().subs(x, Rational(p, arg.q))

        if arg.is_Add:
            coeff, tail = arg.as_coeff_add()
            if coeff and coeff.q != 1:
                intpart = floor(coeff)
                tail = (coeff - intpart,) + tail
                coeff = intpart
            tail = arg._new_rawargs(*tail, reeval=False)
            return self.func(tail)*RisingFactorial(tail, coeff)

        return self.func(*self.args)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def _eval_is_real(self):
        x = self.args[0]
        if x.is_nonpositive and x.is_integer:
            return False
        if intlike(x) and x <= 0:
            return False
        if x.is_positive or x.is_noninteger:
            return True

    def _eval_is_positive(self):
        x = self.args[0]
        if x.is_positive:
            return True
        elif x.is_noninteger:
            return floor(x).is_even

    def _eval_rewrite_as_tractable(self, z, limitvar=None, **kwargs):
        return exp(loggamma(z))

    def _eval_rewrite_as_factorial(self, z, **kwargs):
        return factorial(z - 1)

    def _eval_nseries(self, x, n, logx, cdir=0):
        x0 = self.args[0].limit(x, 0)
        if not (x0.is_Integer and x0 <= 0):
            return super()._eval_nseries(x, n, logx)
        t = self.args[0] - x0
        return (self.func(t + 1)/rf(self.args[0], -x0 + 1))._eval_nseries(x, n, logx)

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0]
        x0 = arg.subs(x, 0)

        if x0.is_integer and x0.is_nonpositive:
            n = -x0
            res = S.NegativeOne**n/self.func(n + 1)
            return res/(arg + n).as_leading_term(x)
        elif not x0.is_infinite:
            return self.func(x0)
        raise PoleError()


###############################################################################
################## LOWER and UPPER INCOMPLETE GAMMA FUNCTIONS #################
###############################################################################

class lowergamma(DefinedFunction):
    r"""
    The lower incomplete gamma function.

    Explanation
    ===========

    It can be defined as the meromorphic continuation of

    .. math::
        \gamma(s, x) := \int_0^x t^{s-1} e^{-t} \mathrm{d}t = \Gamma(s) - \Gamma(s, x).

    This can be shown to be the same as

    .. math::
        \gamma(s, x) = \frac{x^s}{s} {}_1F_1\left({s \atop s+1} \middle| -x\right),

    where ${}_1F_1$ is the (confluent) hypergeometric function.

    Examples
    ========

    >>> from sympy import lowergamma, S
    >>> from sympy.abc import s, x
    >>> lowergamma(s, x)
    lowergamma(s, x)
    >>> lowergamma(3, x)
    -2*(x**2/2 + x + 1)*exp(-x) + 2
    >>> lowergamma(-S(1)/2, x)
    -2*sqrt(pi)*erf(sqrt(x)) - 2*exp(-x)/sqrt(x)

    See Also
    ========

    gamma: Gamma function.
    uppergamma: Upper incomplete gamma function.
    polygamma: Polygamma function.
    loggamma: Log Gamma function.
    digamma: Digamma function.
    trigamma: Trigamma function.
    sympy.functions.special.beta_functions.beta: Euler Beta function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Incomplete_gamma_function#Lower_incomplete_gamma_function
    .. [2] Abramowitz, Milton; Stegun, Irene A., eds. (1965), Chapter 6,
           Section 5, Handbook of Mathematical Functions with Formulas, Graphs,
           and Mathematical Tables
    .. [3] https://dlmf.nist.gov/8
    .. [4] https://functions.wolfram.com/GammaBetaErf/Gamma2/
    .. [5] https://functions.wolfram.com/GammaBetaErf/Gamma3/

    """


    def fdiff(self, argindex=2):
        from sympy.functions.special.hyper import meijerg
        if argindex == 2:
            a, z = self.args
            return exp(-unpolarify(z))*z**(a - 1)
        elif argindex == 1:
            a, z = self.args
            return gamma(a)*digamma(a) - log(z)*uppergamma(a, z) \
                - meijerg([], [1, 1], [0, 0, a], [], z)

        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, a, x):
        # For lack of a better place, we use this one to extract branching
        # information. The following can be
        # found in the literature (c/f references given above), albeit scattered:
        # 1) For fixed x != 0, lowergamma(s, x) is an entire function of s
        # 2) For fixed positive integers s, lowergamma(s, x) is an entire
        #    function of x.
        # 3) For fixed non-positive integers s,
        #    lowergamma(s, exp(I*2*pi*n)*x) =
        #              2*pi*I*n*(-1)**(-s)/factorial(-s) + lowergamma(s, x)
        #    (this follows from lowergamma(s, x).diff(x) = x**(s-1)*exp(-x)).
        # 4) For fixed non-integral s,
        #    lowergamma(s, x) = x**s*gamma(s)*lowergamma_unbranched(s, x),
        #    where lowergamma_unbranched(s, x) is an entire function (in fact
        #    of both s and x), i.e.
        #    lowergamma(s, exp(2*I*pi*n)*x) = exp(2*pi*I*n*a)*lowergamma(a, x)
        if x is S.Zero:
            return S.Zero
        nx, n = x.extract_branch_factor()
        if a.is_integer and a.is_positive:
            nx = unpolarify(x)
            if nx != x:
                return lowergamma(a, nx)
        elif a.is_integer and a.is_nonpositive:
            if n != 0:
                return 2*pi*I*n*S.NegativeOne**(-a)/factorial(-a) + lowergamma(a, nx)
        elif n != 0:
            return exp(2*pi*I*n*a)*lowergamma(a, nx)

        # Special values.
        if a.is_Number:
            if a is S.One:
                return S.One - exp(-x)
            elif a is S.Half:
                return sqrt(pi)*erf(sqrt(x))
            elif a.is_Integer or (2*a).is_Integer:
                b = a - 1
                if b.is_positive:
                    if a.is_integer:
                        return factorial(b) - exp(-x) * factorial(b) * Add(*[x ** k / factorial(k) for k in range(a)])
                    else:
                        return gamma(a)*(lowergamma(S.Half, x)/sqrt(pi) - exp(-x)*Add(*[x**(k - S.Half)/gamma(S.Half + k) for k in range(1, a + S.Half)]))

                if not a.is_Integer:
                    return S.NegativeOne**(S.Half - a)*pi*erf(sqrt(x))/gamma(1 - a) + exp(-x)*Add(*[x**(k + a - 1)*gamma(a)/gamma(a + k) for k in range(1, Rational(3, 2) - a)])

        if x.is_zero:
            return S.Zero

    def _eval_evalf(self, prec):
        if all(x.is_number for x in self.args):
            a = self.args[0]._to_mpmath(prec)
            z = self.args[1]._to_mpmath(prec)
            with workprec(prec):
                res = mp.gammainc(a, 0, z)
            return Expr._from_mpmath(res, prec)
        else:
            return self

    def _eval_conjugate(self):
        x = self.args[1]
        if x not in (S.Zero, S.NegativeInfinity):
            return self.func(self.args[0].conjugate(), x.conjugate())

    def _eval_is_meromorphic(self, x, a):
        # By https://en.wikipedia.org/wiki/Incomplete_gamma_function#Holomorphic_extension,
        #    lowergamma(s, z) = z**s*gamma(s)*gammastar(s, z),
        # where gammastar(s, z) is holomorphic for all s and z.
        # Hence the singularities of lowergamma are z = 0  (branch
        # point) and nonpositive integer values of s (poles of gamma(s)).
        s, z = self.args
        args_merom = fuzzy_and([z._eval_is_meromorphic(x, a),
            s._eval_is_meromorphic(x, a)])
        if not args_merom:
            return args_merom
        z0 = z.subs(x, a)
        if s.is_integer:
            return fuzzy_and([s.is_positive, z0.is_finite])
        s0 = s.subs(x, a)
        return fuzzy_and([s0.is_finite, z0.is_finite, fuzzy_not(z0.is_zero)])

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import O
        s, z = self.args
        if args0[0] is oo and not z.has(x):
            coeff = z**s*exp(-z)
            sum_expr = sum(z**k/rf(s, k + 1) for k in range(n - 1))
            o = O(z**s*s**(-n))
            return coeff*sum_expr + o
        return super()._eval_aseries(n, args0, x, logx)

    def _eval_rewrite_as_uppergamma(self, s, x, **kwargs):
        return gamma(s) - uppergamma(s, x)

    def _eval_rewrite_as_expint(self, s, x, **kwargs):
        from sympy.functions.special.error_functions import expint
        if s.is_integer and s.is_nonpositive:
            return self
        return self.rewrite(uppergamma).rewrite(expint)

    def _eval_is_zero(self):
        x = self.args[1]
        if x.is_zero:
            return True


class uppergamma(DefinedFunction):
    r"""
    The upper incomplete gamma function.

    Explanation
    ===========

    It can be defined as the meromorphic continuation of

    .. math::
        \Gamma(s, x) := \int_x^\infty t^{s-1} e^{-t} \mathrm{d}t = \Gamma(s) - \gamma(s, x).

    where $\gamma(s, x)$ is the lower incomplete gamma function,
    :class:`lowergamma`. This can be shown to be the same as

    .. math::
        \Gamma(s, x) = \Gamma(s) - \frac{x^s}{s} {}_1F_1\left({s \atop s+1} \middle| -x\right),

    where ${}_1F_1$ is the (confluent) hypergeometric function.

    The upper incomplete gamma function is also essentially equivalent to the
    generalized exponential integral:

    .. math::
        \operatorname{E}_{n}(x) = \int_{1}^{\infty}{\frac{e^{-xt}}{t^n} \, dt} = x^{n-1}\Gamma(1-n,x).

    Examples
    ========

    >>> from sympy import uppergamma, S
    >>> from sympy.abc import s, x
    >>> uppergamma(s, x)
    uppergamma(s, x)
    >>> uppergamma(3, x)
    2*(x**2/2 + x + 1)*exp(-x)
    >>> uppergamma(-S(1)/2, x)
    -2*sqrt(pi)*erfc(sqrt(x)) + 2*exp(-x)/sqrt(x)
    >>> uppergamma(-2, x)
    expint(3, x)/x**2

    See Also
    ========

    gamma: Gamma function.
    lowergamma: Lower incomplete gamma function.
    polygamma: Polygamma function.
    loggamma: Log Gamma function.
    digamma: Digamma function.
    trigamma: Trigamma function.
    sympy.functions.special.beta_functions.beta: Euler Beta function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Incomplete_gamma_function#Upper_incomplete_gamma_function
    .. [2] Abramowitz, Milton; Stegun, Irene A., eds. (1965), Chapter 6,
           Section 5, Handbook of Mathematical Functions with Formulas, Graphs,
           and Mathematical Tables
    .. [3] https://dlmf.nist.gov/8
    .. [4] https://functions.wolfram.com/GammaBetaErf/Gamma2/
    .. [5] https://functions.wolfram.com/GammaBetaErf/Gamma3/
    .. [6] https://en.wikipedia.org/wiki/Exponential_integral#Relation_with_other_functions

    """


    def fdiff(self, argindex=2):
        from sympy.functions.special.hyper import meijerg
        if argindex == 2:
            a, z = self.args
            return -exp(-unpolarify(z))*z**(a - 1)
        elif argindex == 1:
            a, z = self.args
            return uppergamma(a, z)*log(z) + meijerg([], [1, 1], [0, 0, a], [], z)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_evalf(self, prec):
        if all(x.is_number for x in self.args):
            a = self.args[0]._to_mpmath(prec)
            z = self.args[1]._to_mpmath(prec)
            with workprec(prec):
                res = mp.gammainc(a, z, mp.inf)
            return Expr._from_mpmath(res, prec)
        return self

    @classmethod
    def eval(cls, a, z):
        from sympy.functions.special.error_functions import expint
        if z.is_Number:
            if z is S.NaN:
                return S.NaN
            elif z is oo:
                return S.Zero
            elif z.is_zero:
                if re(a).is_positive:
                    return gamma(a)

        # We extract branching information here. C/f lowergamma.
        nx, n = z.extract_branch_factor()
        if a.is_integer and a.is_positive:
            nx = unpolarify(z)
            if z != nx:
                return uppergamma(a, nx)
        elif a.is_integer and a.is_nonpositive:
            if n != 0:
                return -2*pi*I*n*S.NegativeOne**(-a)/factorial(-a) + uppergamma(a, nx)
        elif n != 0:
            return gamma(a)*(1 - exp(2*pi*I*n*a)) + exp(2*pi*I*n*a)*uppergamma(a, nx)

        # Special values.
        if a.is_Number:
            if a is S.Zero and z.is_positive:
                return -Ei(-z)
            elif a is S.One:
                return exp(-z)
            elif a is S.Half:
                return sqrt(pi)*erfc(sqrt(z))
            elif a.is_Integer or (2*a).is_Integer:
                b = a - 1
                if b.is_positive:
                    if a.is_integer:
                        return exp(-z) * factorial(b) * Add(*[z**k / factorial(k)
                                                              for k in range(a)])
                    else:
                        return (gamma(a) * erfc(sqrt(z)) +
                                S.NegativeOne**(a - S(3)/2) * exp(-z) * sqrt(z)
                                * Add(*[gamma(-S.Half - k) * (-z)**k / gamma(1-a)
                                        for k in range(a - S.Half)]))
                elif b.is_Integer:
                    return expint(-b, z)*unpolarify(z)**(b + 1)

                if not a.is_Integer:
                    return (S.NegativeOne**(S.Half - a) * pi*erfc(sqrt(z))/gamma(1-a)
                            - z**a * exp(-z) * Add(*[z**k * gamma(a) / gamma(a+k+1)
                                                     for k in range(S.Half - a)]))

        if a.is_zero and z.is_positive:
            return -Ei(-z)

        if z.is_zero and re(a).is_positive:
            return gamma(a)

    def _eval_conjugate(self):
        z = self.args[1]
        if z not in (S.Zero, S.NegativeInfinity):
            return self.func(self.args[0].conjugate(), z.conjugate())

    def _eval_is_meromorphic(self, x, a):
        return lowergamma._eval_is_meromorphic(self, x, a)

    def _eval_rewrite_as_lowergamma(self, s, x, **kwargs):
        return gamma(s) - lowergamma(s, x)

    def _eval_rewrite_as_tractable(self, s, x, **kwargs):
        return exp(loggamma(s)) - lowergamma(s, x)

    def _eval_rewrite_as_expint(self, s, x, **kwargs):
        from sympy.functions.special.error_functions import expint
        return expint(1 - s, x)*x**s


###############################################################################
###################### POLYGAMMA and LOGGAMMA FUNCTIONS #######################
###############################################################################

class polygamma(DefinedFunction):
    r"""
    The function ``polygamma(n, z)`` returns ``log(gamma(z)).diff(n + 1)``.

    Explanation
    ===========

    It is a meromorphic function on $\mathbb{C}$ and defined as the $(n+1)$-th
    derivative of the logarithm of the gamma function:

    .. math::
        \psi^{(n)} (z) := \frac{\mathrm{d}^{n+1}}{\mathrm{d} z^{n+1}} \log\Gamma(z).

    For `n` not a nonnegative integer the generalization by Espinosa and Moll [5]_
    is used:

    .. math:: \psi(s,z) = \frac{\zeta'(s+1, z) + (\gamma + \psi(-s)) \zeta(s+1, z)}
        {\Gamma(-s)}

    Examples
    ========

    Several special values are known:

    >>> from sympy import S, polygamma
    >>> polygamma(0, 1)
    -EulerGamma
    >>> polygamma(0, 1/S(2))
    -2*log(2) - EulerGamma
    >>> polygamma(0, 1/S(3))
    -log(3) - sqrt(3)*pi/6 - EulerGamma - log(sqrt(3))
    >>> polygamma(0, 1/S(4))
    -pi/2 - log(4) - log(2) - EulerGamma
    >>> polygamma(0, 2)
    1 - EulerGamma
    >>> polygamma(0, 23)
    19093197/5173168 - EulerGamma

    >>> from sympy import oo, I
    >>> polygamma(0, oo)
    oo
    >>> polygamma(0, -oo)
    oo
    >>> polygamma(0, I*oo)
    oo
    >>> polygamma(0, -I*oo)
    oo

    Differentiation with respect to $x$ is supported:

    >>> from sympy import Symbol, diff
    >>> x = Symbol("x")
    >>> diff(polygamma(0, x), x)
    polygamma(1, x)
    >>> diff(polygamma(0, x), x, 2)
    polygamma(2, x)
    >>> diff(polygamma(0, x), x, 3)
    polygamma(3, x)
    >>> diff(polygamma(1, x), x)
    polygamma(2, x)
    >>> diff(polygamma(1, x), x, 2)
    polygamma(3, x)
    >>> diff(polygamma(2, x), x)
    polygamma(3, x)
    >>> diff(polygamma(2, x), x, 2)
    polygamma(4, x)

    >>> n = Symbol("n")
    >>> diff(polygamma(n, x), x)
    polygamma(n + 1, x)
    >>> diff(polygamma(n, x), x, 2)
    polygamma(n + 2, x)

    We can rewrite ``polygamma`` functions in terms of harmonic numbers:

    >>> from sympy import harmonic
    >>> polygamma(0, x).rewrite(harmonic)
    harmonic(x - 1) - EulerGamma
    >>> polygamma(2, x).rewrite(harmonic)
    2*harmonic(x - 1, 3) - 2*zeta(3)
    >>> ni = Symbol("n", integer=True)
    >>> polygamma(ni, x).rewrite(harmonic)
    (-1)**(n + 1)*(-harmonic(x - 1, n + 1) + zeta(n + 1))*factorial(n)

    See Also
    ========

    gamma: Gamma function.
    lowergamma: Lower incomplete gamma function.
    uppergamma: Upper incomplete gamma function.
    loggamma: Log Gamma function.
    digamma: Digamma function.
    trigamma: Trigamma function.
    sympy.functions.special.beta_functions.beta: Euler Beta function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Polygamma_function
    .. [2] https://mathworld.wolfram.com/PolygammaFunction.html
    .. [3] https://functions.wolfram.com/GammaBetaErf/PolyGamma/
    .. [4] https://functions.wolfram.com/GammaBetaErf/PolyGamma2/
    .. [5] O. Espinosa and V. Moll, "A generalized polygamma function",
           *Integral Transforms and Special Functions* (2004), 101-115.

    """

    @classmethod
    def eval(cls, n, z):
        if n is S.NaN or z is S.NaN:
            return S.NaN
        elif z is oo:
            return oo if n.is_zero else S.Zero
        elif z.is_Integer and z.is_nonpositive:
            return S.ComplexInfinity
        elif n is S.NegativeOne:
            return loggamma(z) - log(2*pi) / 2
        elif n.is_zero:
            if z is -oo or z.extract_multiplicatively(I) in (oo, -oo):
                return oo
            elif z.is_Integer:
                return harmonic(z-1) - S.EulerGamma
            elif z.is_Rational:
                # TODO n == 1 also can do some rational z
                p, q = z.as_numer_denom()
                # only expand for small denominators to avoid creating long expressions
                if q <= 6:
                    return expand_func(polygamma(S.Zero, z, evaluate=False))
        elif n.is_integer and n.is_nonnegative:
            nz = unpolarify(z)
            if z != nz:
                return polygamma(n, nz)
            if z.is_Integer:
                return S.NegativeOne**(n+1) * factorial(n) * zeta(n+1, z)
            elif z is S.Half:
                return S.NegativeOne**(n+1) * factorial(n) * (2**(n+1)-1) * zeta(n+1)

    def _eval_is_real(self):
        if self.args[0].is_positive and self.args[1].is_positive:
            return True

    def _eval_is_complex(self):
        z = self.args[1]
        is_negative_integer = fuzzy_and([z.is_negative, z.is_integer])
        return fuzzy_and([z.is_complex, fuzzy_not(is_negative_integer)])

    def _eval_is_positive(self):
        n, z = self.args
        if n.is_positive:
            if n.is_odd and z.is_real:
                return True
            if n.is_even and z.is_positive:
                return False

    def _eval_is_negative(self):
        n, z = self.args
        if n.is_positive:
            if n.is_even and z.is_positive:
                return True
            if n.is_odd and z.is_real:
                return False

    def _eval_expand_func(self, **hints):
        n, z = self.args

        if n.is_Integer and n.is_nonnegative:
            if z.is_Add:
                coeff = z.args[0]
                if coeff.is_Integer:
                    e = -(n + 1)
                    if coeff > 0:
                        tail = Add(*[Pow(
                            z - i, e) for i in range(1, int(coeff) + 1)])
                    else:
                        tail = -Add(*[Pow(
                            z + i, e) for i in range(int(-coeff))])
                    return polygamma(n, z - coeff) + S.NegativeOne**n*factorial(n)*tail

            elif z.is_Mul:
                coeff, z = z.as_two_terms()
                if coeff.is_Integer and coeff.is_positive:
                    tail = [polygamma(n, z + Rational(
                        i, coeff)) for i in range(int(coeff))]
                    if n == 0:
                        return Add(*tail)/coeff + log(coeff)
                    else:
                        return Add(*tail)/coeff**(n + 1)
                z *= coeff

        if n == 0 and z.is_Rational:
            p, q = z.as_numer_denom()

            # Reference:
            #   Values of the polygamma functions at rational arguments, J. Choi, 2007
            part_1 = -S.EulerGamma - pi * cot(p * pi / q) / 2 - log(q) + Add(
                *[cos(2 * k * pi * p / q) * log(2 * sin(k * pi / q)) for k in range(1, q)])

            if z > 0:
                n = floor(z)
                z0 = z - n
                return part_1 + Add(*[1 / (z0 + k) for k in range(n)])
            elif z < 0:
                n = floor(1 - z)
                z0 = z + n
                return part_1 - Add(*[1 / (z0 - 1 - k) for k in range(n)])

        if n == -1:
            return loggamma(z) - log(2*pi) / 2
        if n.is_integer is False or n.is_nonnegative is False:
            s = Dummy("s")
            dzt = zeta(s, z).diff(s).subs(s, n+1)
            return (dzt + (S.EulerGamma + digamma(-n)) * zeta(n+1, z)) / gamma(-n)

        return polygamma(n, z)

    def _eval_rewrite_as_zeta(self, n, z, **kwargs):
        if n.is_integer and n.is_positive:
            return S.NegativeOne**(n + 1)*factorial(n)*zeta(n + 1, z)

    def _eval_rewrite_as_harmonic(self, n, z, **kwargs):
        if n.is_integer:
            if n.is_zero:
                return harmonic(z - 1) - S.EulerGamma
            else:
                return S.NegativeOne**(n+1) * factorial(n) * (zeta(n+1) - harmonic(z-1, n+1))

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.series.order import Order
        n, z = [a.as_leading_term(x) for a in self.args]
        o = Order(z, x)
        if n == 0 and o.contains(1/x):
            logx = log(x) if logx is None else logx
            return o.getn() * logx
        else:
            return self.func(n, z)

    def fdiff(self, argindex=2):
        if argindex == 2:
            n, z = self.args[:2]
            return polygamma(n + 1, z)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        if args0[1] != oo or not \
                (self.args[0].is_Integer and self.args[0].is_nonnegative):
            return super()._eval_aseries(n, args0, x, logx)
        z = self.args[1]
        N = self.args[0]

        if N == 0:
            # digamma function series
            # Abramowitz & Stegun, p. 259, 6.3.18
            r = log(z) - 1/(2*z)
            o = None
            if n < 2:
                o = Order(1/z, x)
            else:
                m = ceiling((n + 1)//2)
                l = [bernoulli(2*k) / (2*k*z**(2*k)) for k in range(1, m)]
                r -= Add(*l)
                o = Order(1/z**n, x)
            return r._eval_nseries(x, n, logx) + o
        else:
            # proper polygamma function
            # Abramowitz & Stegun, p. 260, 6.4.10
            # We return terms to order higher than O(x**n) on purpose
            # -- otherwise we would not be able to return any terms for
            #    quite a long time!
            fac = gamma(N)
            e0 = fac + N*fac/(2*z)
            m = ceiling((n + 1)//2)
            for k in range(1, m):
                fac = fac*(2*k + N - 1)*(2*k + N - 2) / ((2*k)*(2*k - 1))
                e0 += bernoulli(2*k)*fac/z**(2*k)
            o = Order(1/z**(2*m), x)
            if n == 0:
                o = Order(1/z, x)
            elif n == 1:
                o = Order(1/z**2, x)
            r = e0._eval_nseries(z, n, logx) + o
            return (-1 * (-1/z)**N * r)._eval_nseries(x, n, logx)

    def _eval_evalf(self, prec):
        if not all(i.is_number for i in self.args):
            return
        s = self.args[0]._to_mpmath(prec+12)
        z = self.args[1]._to_mpmath(prec+12)
        if mp.isint(z) and z <= 0:
            return S.ComplexInfinity
        with workprec(prec+12):
            if mp.isint(s) and s >= 0:
                res = mp.polygamma(s, z)
            else:
                zt = mp.zeta(s+1, z)
                dzt = mp.zeta(s+1, z, 1)
                res = (dzt + (mp.euler + mp.digamma(-s)) * zt) * mp.rgamma(-s)
        return Expr._from_mpmath(res, prec)


class loggamma(DefinedFunction):
    r"""
    The ``loggamma`` function implements the logarithm of the
    gamma function (i.e., $\log\Gamma(x)$).

    Examples
    ========

    Several special values are known. For numerical integral
    arguments we have:

    >>> from sympy import loggamma
    >>> loggamma(-2)
    oo
    >>> loggamma(0)
    oo
    >>> loggamma(1)
    0
    >>> loggamma(2)
    0
    >>> loggamma(3)
    log(2)

    And for symbolic values:

    >>> from sympy import Symbol
    >>> n = Symbol("n", integer=True, positive=True)
    >>> loggamma(n)
    log(gamma(n))
    >>> loggamma(-n)
    oo

    For half-integral values:

    >>> from sympy import S
    >>> loggamma(S(5)/2)
    log(3*sqrt(pi)/4)
    >>> loggamma(n/2)
    log(2**(1 - n)*sqrt(pi)*gamma(n)/gamma(n/2 + 1/2))

    And general rational arguments:

    >>> from sympy import expand_func
    >>> L = loggamma(S(16)/3)
    >>> expand_func(L).doit()
    -5*log(3) + loggamma(1/3) + log(4) + log(7) + log(10) + log(13)
    >>> L = loggamma(S(19)/4)
    >>> expand_func(L).doit()
    -4*log(4) + loggamma(3/4) + log(3) + log(7) + log(11) + log(15)
    >>> L = loggamma(S(23)/7)
    >>> expand_func(L).doit()
    -3*log(7) + log(2) + loggamma(2/7) + log(9) + log(16)

    The ``loggamma`` function has the following limits towards infinity:

    >>> from sympy import oo
    >>> loggamma(oo)
    oo
    >>> loggamma(-oo)
    zoo

    The ``loggamma`` function obeys the mirror symmetry
    if $x \in \mathbb{C} \setminus \{-\infty, 0\}$:

    >>> from sympy.abc import x
    >>> from sympy import conjugate
    >>> conjugate(loggamma(x))
    loggamma(conjugate(x))

    Differentiation with respect to $x$ is supported:

    >>> from sympy import diff
    >>> diff(loggamma(x), x)
    polygamma(0, x)

    Series expansion is also supported:

    >>> from sympy import series
    >>> series(loggamma(x), x, 0, 4).cancel()
    -log(x) - EulerGamma*x + pi**2*x**2/12 - x**3*zeta(3)/3 + O(x**4)

    We can numerically evaluate the ``loggamma`` function
    to arbitrary precision on the whole complex plane:

    >>> from sympy import I
    >>> loggamma(5).evalf(30)
    3.17805383034794561964694160130
    >>> loggamma(I).evalf(20)
    -0.65092319930185633889 - 1.8724366472624298171*I

    See Also
    ========

    gamma: Gamma function.
    lowergamma: Lower incomplete gamma function.
    uppergamma: Upper incomplete gamma function.
    polygamma: Polygamma function.
    digamma: Digamma function.
    trigamma: Trigamma function.
    sympy.functions.special.beta_functions.beta: Euler Beta function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gamma_function
    .. [2] https://dlmf.nist.gov/5
    .. [3] https://mathworld.wolfram.com/LogGammaFunction.html
    .. [4] https://functions.wolfram.com/GammaBetaErf/LogGamma/

    """
    @classmethod
    def eval(cls, z):
        if z.is_integer:
            if z.is_nonpositive:
                return oo
            elif z.is_positive:
                return log(gamma(z))
        elif z.is_rational:
            p, q = z.as_numer_denom()
            # Half-integral values:
            if p.is_positive and q == 2:
                return log(sqrt(pi) * 2**(1 - p) * gamma(p) / gamma((p + 1)*S.Half))

        if z is oo:
            return oo
        elif abs(z) is oo:
            return S.ComplexInfinity
        if z is S.NaN:
            return S.NaN

    def _eval_expand_func(self, **hints):
        from sympy.concrete.summations import Sum
        z = self.args[0]

        if z.is_Rational:
            p, q = z.as_numer_denom()
            # General rational arguments (u + p/q)
            # Split z as n + p/q with p < q
            n = p // q
            p = p - n*q
            if p.is_positive and q.is_positive and p < q:
                k = Dummy("k")
                if n.is_positive:
                    return loggamma(p / q) - n*log(q) + Sum(log((k - 1)*q + p), (k, 1, n))
                elif n.is_negative:
                    return loggamma(p / q) - n*log(q) + pi*I*n - Sum(log(k*q - p), (k, 1, -n))
                elif n.is_zero:
                    return loggamma(p / q)

        return self

    def _eval_nseries(self, x, n, logx=None, cdir=0):
        x0 = self.args[0].limit(x, 0)
        if x0.is_zero:
            f = self._eval_rewrite_as_intractable(*self.args)
            return f._eval_nseries(x, n, logx)
        return super()._eval_nseries(x, n, logx)

    def _eval_aseries(self, n, args0, x, logx):
        from sympy.series.order import Order
        if args0[0] != oo:
            return super()._eval_aseries(n, args0, x, logx)
        z = self.args[0]
        r = log(z)*(z - S.Half) - z + log(2*pi)/2
        l = [bernoulli(2*k) / (2*k*(2*k - 1)*z**(2*k - 1)) for k in range(1, n)]
        o = None
        if n == 0:
            o = Order(1, x)
        else:
            o = Order(1/z**n, x)
        # It is very inefficient to first add the order and then do the nseries
        return (r + Add(*l))._eval_nseries(x, n, logx) + o

    def _eval_rewrite_as_intractable(self, z, **kwargs):
        return log(gamma(z))

    def _eval_is_real(self):
        z = self.args[0]
        if z.is_positive:
            return True
        elif z.is_nonpositive:
            return False

    def _eval_conjugate(self):
        z = self.args[0]
        if z not in (S.Zero, S.NegativeInfinity):
            return self.func(z.conjugate())

    def fdiff(self, argindex=1):
        if argindex == 1:
            return polygamma(0, self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)


class digamma(DefinedFunction):
    r"""
    The ``digamma`` function is the first derivative of the ``loggamma``
    function

    .. math::
        \psi(x) := \frac{\mathrm{d}}{\mathrm{d} z} \log\Gamma(z)
                = \frac{\Gamma'(z)}{\Gamma(z) }.

    In this case, ``digamma(z) = polygamma(0, z)``.

    Examples
    ========

    >>> from sympy import digamma
    >>> digamma(0)
    zoo
    >>> from sympy import Symbol
    >>> z = Symbol('z')
    >>> digamma(z)
    polygamma(0, z)

    To retain ``digamma`` as it is:

    >>> digamma(0, evaluate=False)
    digamma(0)
    >>> digamma(z, evaluate=False)
    digamma(z)

    See Also
    ========

    gamma: Gamma function.
    lowergamma: Lower incomplete gamma function.
    uppergamma: Upper incomplete gamma function.
    polygamma: Polygamma function.
    loggamma: Log Gamma function.
    trigamma: Trigamma function.
    sympy.functions.special.beta_functions.beta: Euler Beta function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Digamma_function
    .. [2] https://mathworld.wolfram.com/DigammaFunction.html
    .. [3] https://functions.wolfram.com/GammaBetaErf/PolyGamma2/

    """
    def _eval_evalf(self, prec):
        z = self.args[0]
        nprec = prec_to_dps(prec)
        return polygamma(0, z).evalf(n=nprec)

    def fdiff(self, argindex=1):
        z = self.args[0]
        return polygamma(0, z).fdiff()

    def _eval_is_real(self):
        z = self.args[0]
        return polygamma(0, z).is_real

    def _eval_is_positive(self):
        z = self.args[0]
        return polygamma(0, z).is_positive

    def _eval_is_negative(self):
        z = self.args[0]
        return polygamma(0, z).is_negative

    def _eval_aseries(self, n, args0, x, logx):
        as_polygamma = self.rewrite(polygamma)
        args0 = [S.Zero,] + args0
        return as_polygamma._eval_aseries(n, args0, x, logx)

    @classmethod
    def eval(cls, z):
        return polygamma(0, z)

    def _eval_expand_func(self, **hints):
        z = self.args[0]
        return polygamma(0, z).expand(func=True)

    def _eval_rewrite_as_harmonic(self, z, **kwargs):
        return harmonic(z - 1) - S.EulerGamma

    def _eval_rewrite_as_polygamma(self, z, **kwargs):
        return polygamma(0, z)

    def _eval_as_leading_term(self, x, logx, cdir):
        z = self.args[0]
        return polygamma(0, z).as_leading_term(x)



class trigamma(DefinedFunction):
    r"""
    The ``trigamma`` function is the second derivative of the ``loggamma``
    function

    .. math::
        \psi^{(1)}(z) := \frac{\mathrm{d}^{2}}{\mathrm{d} z^{2}} \log\Gamma(z).

    In this case, ``trigamma(z) = polygamma(1, z)``.

    Examples
    ========

    >>> from sympy import trigamma
    >>> trigamma(0)
    zoo
    >>> from sympy import Symbol
    >>> z = Symbol('z')
    >>> trigamma(z)
    polygamma(1, z)

    To retain ``trigamma`` as it is:

    >>> trigamma(0, evaluate=False)
    trigamma(0)
    >>> trigamma(z, evaluate=False)
    trigamma(z)


    See Also
    ========

    gamma: Gamma function.
    lowergamma: Lower incomplete gamma function.
    uppergamma: Upper incomplete gamma function.
    polygamma: Polygamma function.
    loggamma: Log Gamma function.
    digamma: Digamma function.
    sympy.functions.special.beta_functions.beta: Euler Beta function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigamma_function
    .. [2] https://mathworld.wolfram.com/TrigammaFunction.html
    .. [3] https://functions.wolfram.com/GammaBetaErf/PolyGamma2/

    """
    def _eval_evalf(self, prec):
        z = self.args[0]
        nprec = prec_to_dps(prec)
        return polygamma(1, z).evalf(n=nprec)

    def fdiff(self, argindex=1):
        z = self.args[0]
        return polygamma(1, z).fdiff()

    def _eval_is_real(self):
        z = self.args[0]
        return polygamma(1, z).is_real

    def _eval_is_positive(self):
        z = self.args[0]
        return polygamma(1, z).is_positive

    def _eval_is_negative(self):
        z = self.args[0]
        return polygamma(1, z).is_negative

    def _eval_aseries(self, n, args0, x, logx):
        as_polygamma = self.rewrite(polygamma)
        args0 = [S.One,] + args0
        return as_polygamma._eval_aseries(n, args0, x, logx)

    @classmethod
    def eval(cls, z):
        return polygamma(1, z)

    def _eval_expand_func(self, **hints):
        z = self.args[0]
        return polygamma(1, z).expand(func=True)

    def _eval_rewrite_as_zeta(self, z, **kwargs):
        return zeta(2, z)

    def _eval_rewrite_as_polygamma(self, z, **kwargs):
        return polygamma(1, z)

    def _eval_rewrite_as_harmonic(self, z, **kwargs):
        return -harmonic(z - 1, 2) + pi**2 / 6

    def _eval_as_leading_term(self, x, logx, cdir):
        z = self.args[0]
        return polygamma(1, z).as_leading_term(x)


###############################################################################
##################### COMPLETE MULTIVARIATE GAMMA FUNCTION ####################
###############################################################################


class multigamma(DefinedFunction):
    r"""
    The multivariate gamma function is a generalization of the gamma function

    .. math::
        \Gamma_p(z) = \pi^{p(p-1)/4}\prod_{k=1}^p \Gamma[z + (1 - k)/2].

    In a special case, ``multigamma(x, 1) = gamma(x)``.

    Examples
    ========

    >>> from sympy import S, multigamma
    >>> from sympy import Symbol
    >>> x = Symbol('x')
    >>> p = Symbol('p', positive=True, integer=True)

    >>> multigamma(x, p)
    pi**(p*(p - 1)/4)*Product(gamma(-_k/2 + x + 1/2), (_k, 1, p))

    Several special values are known:

    >>> multigamma(1, 1)
    1
    >>> multigamma(4, 1)
    6
    >>> multigamma(S(3)/2, 1)
    sqrt(pi)/2

    Writing ``multigamma`` in terms of the ``gamma`` function:

    >>> multigamma(x, 1)
    gamma(x)

    >>> multigamma(x, 2)
    sqrt(pi)*gamma(x)*gamma(x - 1/2)

    >>> multigamma(x, 3)
    pi**(3/2)*gamma(x)*gamma(x - 1)*gamma(x - 1/2)

    Parameters
    ==========

    p : order or dimension of the multivariate gamma function

    See Also
    ========

    gamma, lowergamma, uppergamma, polygamma, loggamma, digamma, trigamma,
    sympy.functions.special.beta_functions.beta

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Multivariate_gamma_function

    """
    unbranched = True

    def fdiff(self, argindex=2):
        from sympy.concrete.summations import Sum
        if argindex == 2:
            x, p = self.args
            k = Dummy("k")
            return self.func(x, p)*Sum(polygamma(0, x + (1 - k)/2), (k, 1, p))
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, x, p):
        from sympy.concrete.products import Product
        if p.is_positive is False or p.is_integer is False:
            raise ValueError('Order parameter p must be positive integer.')
        k = Dummy("k")
        return (pi**(p*(p - 1)/4)*Product(gamma(x + (1 - k)/2),
                                          (k, 1, p))).doit()

    def _eval_conjugate(self):
        x, p = self.args
        return self.func(x.conjugate(), p)

    def _eval_is_real(self):
        x, p = self.args
        y = 2*x
        if y.is_integer and (y <= (p - 1)) is True:
            return False
        if intlike(y) and (y <= (p - 1)):
            return False
        if y > (p - 1) or y.is_noninteger:
            return True
<|endoftext|>from sympy.core import S
from sympy.core.function import DefinedFunction, ArgumentIndexError
from sympy.core.symbol import Dummy, uniquely_named_symbol
from sympy.functions.special.gamma_functions import gamma, digamma
from sympy.functions.combinatorial.numbers import catalan
from sympy.functions.elementary.complexes import conjugate

# See mpmath #569 and SymPy #20569
def betainc_mpmath_fix(a, b, x1, x2, reg=0):
    from mpmath import betainc, mpf
    if x1 == x2:
        return mpf(0)
    else:
        return betainc(a, b, x1, x2, reg)

###############################################################################
############################ COMPLETE BETA  FUNCTION ##########################
###############################################################################

class beta(DefinedFunction):
    r"""
    The beta integral is called the Eulerian integral of the first kind by
    Legendre:

    .. math::
        \mathrm{B}(x,y)  \int^{1}_{0} t^{x-1} (1-t)^{y-1} \mathrm{d}t.

    Explanation
    ===========

    The Beta function or Euler's first integral is closely associated
    with the gamma function. The Beta function is often used in probability
    theory and mathematical statistics. It satisfies properties like:

    .. math::
        \mathrm{B}(a,1) = \frac{1}{a} \\
        \mathrm{B}(a,b) = \mathrm{B}(b,a)  \\
        \mathrm{B}(a,b) = \frac{\Gamma(a) \Gamma(b)}{\Gamma(a+b)}

    Therefore for integral values of $a$ and $b$:

    .. math::
        \mathrm{B} = \frac{(a-1)! (b-1)!}{(a+b-1)!}

    A special case of the Beta function when `x = y` is the
    Central Beta function. It satisfies properties like:

    .. math::
        \mathrm{B}(x) = 2^{1 - 2x}\mathrm{B}(x, \frac{1}{2})
        \mathrm{B}(x) = 2^{1 - 2x} cos(\pi x) \mathrm{B}(\frac{1}{2} - x, x)
        \mathrm{B}(x) = \int_{0}^{1} \frac{t^x}{(1 + t)^{2x}} dt
        \mathrm{B}(x) = \frac{2}{x} \prod_{n = 1}^{\infty} \frac{n(n + 2x)}{(n + x)^2}

    Examples
    ========

    >>> from sympy import I, pi
    >>> from sympy.abc import x, y

    The Beta function obeys the mirror symmetry:

    >>> from sympy import beta, conjugate
    >>> conjugate(beta(x, y))
    beta(conjugate(x), conjugate(y))

    Differentiation with respect to both $x$ and $y$ is supported:

    >>> from sympy import beta, diff
    >>> diff(beta(x, y), x)
    (polygamma(0, x) - polygamma(0, x + y))*beta(x, y)

    >>> diff(beta(x, y), y)
    (polygamma(0, y) - polygamma(0, x + y))*beta(x, y)

    >>> diff(beta(x), x)
    2*(polygamma(0, x) - polygamma(0, 2*x))*beta(x, x)

    We can numerically evaluate the Beta function to
    arbitrary precision for any complex numbers x and y:

    >>> from sympy import beta
    >>> beta(pi).evalf(40)
    0.02671848900111377452242355235388489324562

    >>> beta(1 + I).evalf(20)
    -0.2112723729365330143 - 0.7655283165378005676*I

    See Also
    ========

    gamma: Gamma function.
    uppergamma: Upper incomplete gamma function.
    lowergamma: Lower incomplete gamma function.
    polygamma: Polygamma function.
    loggamma: Log Gamma function.
    digamma: Digamma function.
    trigamma: Trigamma function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Beta_function
    .. [2] https://mathworld.wolfram.com/BetaFunction.html
    .. [3] https://dlmf.nist.gov/5.12

    """
    unbranched = True

    def fdiff(self, argindex):
        x, y = self.args
        if argindex == 1:
            # Diff wrt x
            return beta(x, y)*(digamma(x) - digamma(x + y))
        elif argindex == 2:
            # Diff wrt y
            return beta(x, y)*(digamma(y) - digamma(x + y))
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, x, y=None):
        if y is None:
            return beta(x, x)
        if x.is_Number and y.is_Number:
            return beta(x, y, evaluate=False).doit()

    def doit(self, **hints):
        x = xold = self.args[0]
        # Deal with unevaluated single argument beta
        single_argument = len(self.args) == 1
        y = yold = self.args[0] if single_argument else self.args[1]
        if hints.get('deep', True):
            x = x.doit(**hints)
            y = y.doit(**hints)
        if y.is_zero or x.is_zero:
            return S.ComplexInfinity
        if y is S.One:
            return 1/x
        if x is S.One:
            return 1/y
        if y == x + 1:
            return 1/(x*y*catalan(x))
        s = x + y
        if (s.is_integer and s.is_negative and x.is_integer is False and
            y.is_integer is False):
            return S.Zero
        if x == xold and y == yold and not single_argument:
            return self
        return beta(x, y)

    def _eval_expand_func(self, **hints):
        x, y = self.args
        return gamma(x)*gamma(y) / gamma(x + y)

    def _eval_is_real(self):
        return self.args[0].is_real and self.args[1].is_real

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate(), self.args[1].conjugate())

    def _eval_rewrite_as_gamma(self, x, y, piecewise=True, **kwargs):
        return self._eval_expand_func(**kwargs)

    def _eval_rewrite_as_Integral(self, x, y, **kwargs):
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', [x, y]).name)
        return Integral(t**(x - 1)*(1 - t)**(y - 1), (t, 0, 1))

###############################################################################
########################## INCOMPLETE BETA FUNCTION ###########################
###############################################################################

class betainc(DefinedFunction):
    r"""
    The Generalized Incomplete Beta function is defined as

    .. math::
        \mathrm{B}_{(x_1, x_2)}(a, b) = \int_{x_1}^{x_2} t^{a - 1} (1 - t)^{b - 1} dt

    The Incomplete Beta function is a special case
    of the Generalized Incomplete Beta function :

    .. math:: \mathrm{B}_z (a, b) = \mathrm{B}_{(0, z)}(a, b)

    The Incomplete Beta function satisfies :

    .. math:: \mathrm{B}_z (a, b) = (-1)^a \mathrm{B}_{\frac{z}{z - 1}} (a, 1 - a - b)

    The Beta function is a special case of the Incomplete Beta function :

    .. math:: \mathrm{B}(a, b) = \mathrm{B}_{1}(a, b)

    Examples
    ========

    >>> from sympy import betainc, symbols, conjugate
    >>> a, b, x, x1, x2 = symbols('a b x x1 x2')

    The Generalized Incomplete Beta function is given by:

    >>> betainc(a, b, x1, x2)
    betainc(a, b, x1, x2)

    The Incomplete Beta function can be obtained as follows:

    >>> betainc(a, b, 0, x)
    betainc(a, b, 0, x)

    The Incomplete Beta function obeys the mirror symmetry:

    >>> conjugate(betainc(a, b, x1, x2))
    betainc(conjugate(a), conjugate(b), conjugate(x1), conjugate(x2))

    We can numerically evaluate the Incomplete Beta function to
    arbitrary precision for any complex numbers a, b, x1 and x2:

    >>> from sympy import betainc, I
    >>> betainc(2, 3, 4, 5).evalf(10)
    56.08333333
    >>> betainc(0.75, 1 - 4*I, 0, 2 + 3*I).evalf(25)
    0.2241657956955709603655887 + 0.3619619242700451992411724*I

    The Generalized Incomplete Beta function can be expressed
    in terms of the Generalized Hypergeometric function.

    >>> from sympy import hyper
    >>> betainc(a, b, x1, x2).rewrite(hyper)
    (-x1**a*hyper((a, 1 - b), (a + 1,), x1) + x2**a*hyper((a, 1 - b), (a + 1,), x2))/a

    See Also
    ========

    beta: Beta function
    hyper: Generalized Hypergeometric function

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function
    .. [2] https://dlmf.nist.gov/8.17
    .. [3] https://functions.wolfram.com/GammaBetaErf/Beta4/
    .. [4] https://functions.wolfram.com/GammaBetaErf/BetaRegularized4/02/

    """
    nargs = 4
    unbranched = True

    def fdiff(self, argindex):
        a, b, x1, x2 = self.args
        if argindex == 3:
            # Diff wrt x1
            return -(1 - x1)**(b - 1)*x1**(a - 1)
        elif argindex == 4:
            # Diff wrt x2
            return (1 - x2)**(b - 1)*x2**(a - 1)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_mpmath(self):
        return betainc_mpmath_fix, self.args

    def _eval_is_real(self):
        if all(arg.is_real for arg in self.args):
            return True

    def _eval_conjugate(self):
        return self.func(*map(conjugate, self.args))

    def _eval_rewrite_as_Integral(self, a, b, x1, x2, **kwargs):
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', [a, b, x1, x2]).name)
        return Integral(t**(a - 1)*(1 - t)**(b - 1), (t, x1, x2))

    def _eval_rewrite_as_hyper(self, a, b, x1, x2, **kwargs):
        from sympy.functions.special.hyper import hyper
        return (x2**a * hyper((a, 1 - b), (a + 1,), x2) - x1**a * hyper((a, 1 - b), (a + 1,), x1)) / a

###############################################################################
#################### REGULARIZED INCOMPLETE BETA FUNCTION #####################
###############################################################################

class betainc_regularized(DefinedFunction):
    r"""
    The Generalized Regularized Incomplete Beta function is given by

    .. math::
        \mathrm{I}_{(x_1, x_2)}(a, b) = \frac{\mathrm{B}_{(x_1, x_2)}(a, b)}{\mathrm{B}(a, b)}

    The Regularized Incomplete Beta function is a special case
    of the Generalized Regularized Incomplete Beta function :

    .. math:: \mathrm{I}_z (a, b) = \mathrm{I}_{(0, z)}(a, b)

    The Regularized Incomplete Beta function is the cumulative distribution
    function of the beta distribution.

    Examples
    ========

    >>> from sympy import betainc_regularized, symbols, conjugate
    >>> a, b, x, x1, x2 = symbols('a b x x1 x2')

    The Generalized Regularized Incomplete Beta
    function is given by:

    >>> betainc_regularized(a, b, x1, x2)
    betainc_regularized(a, b, x1, x2)

    The Regularized Incomplete Beta function
    can be obtained as follows:

    >>> betainc_regularized(a, b, 0, x)
    betainc_regularized(a, b, 0, x)

    The Regularized Incomplete Beta function
    obeys the mirror symmetry:

    >>> conjugate(betainc_regularized(a, b, x1, x2))
    betainc_regularized(conjugate(a), conjugate(b), conjugate(x1), conjugate(x2))

    We can numerically evaluate the Regularized Incomplete Beta function
    to arbitrary precision for any complex numbers a, b, x1 and x2:

    >>> from sympy import betainc_regularized, pi, E
    >>> betainc_regularized(1, 2, 0, 0.25).evalf(10)
    0.4375000000
    >>> betainc_regularized(pi, E, 0, 1).evalf(5)
    1.00000

    The Generalized Regularized Incomplete Beta function can be
    expressed in terms of the Generalized Hypergeometric function.

    >>> from sympy import hyper
    >>> betainc_regularized(a, b, x1, x2).rewrite(hyper)
    (-x1**a*hyper((a, 1 - b), (a + 1,), x1) + x2**a*hyper((a, 1 - b), (a + 1,), x2))/(a*beta(a, b))

    See Also
    ========

    beta: Beta function
    hyper: Generalized Hypergeometric function

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function
    .. [2] https://dlmf.nist.gov/8.17
    .. [3] https://functions.wolfram.com/GammaBetaErf/Beta4/
    .. [4] https://functions.wolfram.com/GammaBetaErf/BetaRegularized4/02/

    """
    nargs = 4
    unbranched = True

    def __new__(cls, a, b, x1, x2):
        return super().__new__(cls, a, b, x1, x2)

    def _eval_mpmath(self):
        return betainc_mpmath_fix, (*self.args, S(1))

    def fdiff(self, argindex):
        a, b, x1, x2 = self.args
        if argindex == 3:
            # Diff wrt x1
            return -(1 - x1)**(b - 1)*x1**(a - 1) / beta(a, b)
        elif argindex == 4:
            # Diff wrt x2
            return (1 - x2)**(b - 1)*x2**(a - 1) / beta(a, b)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_is_real(self):
        if all(arg.is_real for arg in self.args):
            return True

    def _eval_conjugate(self):
        return self.func(*map(conjugate, self.args))

    def _eval_rewrite_as_Integral(self, a, b, x1, x2, **kwargs):
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', [a, b, x1, x2]).name)
        integrand = t**(a - 1)*(1 - t)**(b - 1)
        expr = Integral(integrand, (t, x1, x2))
        return expr / Integral(integrand, (t, 0, 1))

    def _eval_rewrite_as_hyper(self, a, b, x1, x2, **kwargs):
        from sympy.functions.special.hyper import hyper
        expr = (x2**a * hyper((a, 1 - b), (a + 1,), x2) - x1**a * hyper((a, 1 - b), (a + 1,), x1)) / a
        return expr / beta(a, b)
<|endoftext|>from sympy.core import S, sympify, NumberKind
from sympy.utilities.iterables import sift
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.operations import LatticeOp, ShortCircuit
from sympy.core.function import (Application, Lambda,
    ArgumentIndexError, DefinedFunction)
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.power import Pow
from sympy.core.relational import Eq, Relational
from sympy.core.singleton import Singleton
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy
from sympy.core.rules import Transform
from sympy.core.logic import fuzzy_and, fuzzy_or, _torf
from sympy.core.traversal import walk
from sympy.core.numbers import Integer
from sympy.logic.boolalg import And, Or


def _minmax_as_Piecewise(op, *args):
    # helper for Min/Max rewrite as Piecewise
    from sympy.functions.elementary.piecewise import Piecewise
    ec = []
    for i, a in enumerate(args):
        c = [Relational(a, args[j], op) for j in range(i + 1, len(args))]
        ec.append((a, And(*c)))
    return Piecewise(*ec)


class IdentityFunction(Lambda, metaclass=Singleton):
    """
    The identity function

    Examples
    ========

    >>> from sympy import Id, Symbol
    >>> x = Symbol('x')
    >>> Id(x)
    x

    """

    _symbol = Dummy('x')

    @property
    def signature(self):
        return Tuple(self._symbol)

    @property
    def expr(self):
        return self._symbol


Id = S.IdentityFunction

###############################################################################
############################# ROOT and SQUARE ROOT FUNCTION ###################
###############################################################################


def sqrt(arg, evaluate=None):
    """Returns the principal square root.

    Parameters
    ==========

    evaluate : bool, optional
        The parameter determines if the expression should be evaluated.
        If ``None``, its value is taken from
        ``global_parameters.evaluate``.

    Examples
    ========

    >>> from sympy import sqrt, Symbol, S
    >>> x = Symbol('x')

    >>> sqrt(x)
    sqrt(x)

    >>> sqrt(x)**2
    x

    Note that sqrt(x**2) does not simplify to x.

    >>> sqrt(x**2)
    sqrt(x**2)

    This is because the two are not equal to each other in general.
    For example, consider x == -1:

    >>> from sympy import Eq
    >>> Eq(sqrt(x**2), x).subs(x, -1)
    False

    This is because sqrt computes the principal square root, so the square may
    put the argument in a different branch.  This identity does hold if x is
    positive:

    >>> y = Symbol('y', positive=True)
    >>> sqrt(y**2)
    y

    You can force this simplification by using the powdenest() function with
    the force option set to True:

    >>> from sympy import powdenest
    >>> sqrt(x**2)
    sqrt(x**2)
    >>> powdenest(sqrt(x**2), force=True)
    x

    To get both branches of the square root you can use the rootof function:

    >>> from sympy import rootof

    >>> [rootof(x**2-3,i) for i in (0,1)]
    [-sqrt(3), sqrt(3)]

    Although ``sqrt`` is printed, there is no ``sqrt`` function so looking for
    ``sqrt`` in an expression will fail:

    >>> from sympy.utilities.misc import func_name
    >>> func_name(sqrt(x))
    'Pow'
    >>> sqrt(x).has(sqrt)
    False

    To find ``sqrt`` look for ``Pow`` with an exponent of ``1/2``:

    >>> (x + 1/sqrt(x)).find(lambda i: i.is_Pow and abs(i.exp) is S.Half)
    {1/sqrt(x)}

    See Also
    ========

    sympy.polys.rootoftools.rootof, root, real_root

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Square_root
    .. [2] https://en.wikipedia.org/wiki/Principal_value
    """
    # arg = sympify(arg) is handled by Pow
    return Pow(arg, S.Half, evaluate=evaluate)


def cbrt(arg, evaluate=None):
    """Returns the principal cube root.

    Parameters
    ==========

    evaluate : bool, optional
        The parameter determines if the expression should be evaluated.
        If ``None``, its value is taken from
        ``global_parameters.evaluate``.

    Examples
    ========

    >>> from sympy import cbrt, Symbol
    >>> x = Symbol('x')

    >>> cbrt(x)
    x**(1/3)

    >>> cbrt(x)**3
    x

    Note that cbrt(x**3) does not simplify to x.

    >>> cbrt(x**3)
    (x**3)**(1/3)

    This is because the two are not equal to each other in general.
    For example, consider `x == -1`:

    >>> from sympy import Eq
    >>> Eq(cbrt(x**3), x).subs(x, -1)
    False

    This is because cbrt computes the principal cube root, this
    identity does hold if `x` is positive:

    >>> y = Symbol('y', positive=True)
    >>> cbrt(y**3)
    y

    See Also
    ========

    sympy.polys.rootoftools.rootof, root, real_root

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Cube_root
    .. [2] https://en.wikipedia.org/wiki/Principal_value

    """
    return Pow(arg, Rational(1, 3), evaluate=evaluate)


def root(arg, n, k=0, evaluate=None):
    r"""Returns the *k*-th *n*-th root of ``arg``.

    Parameters
    ==========

    k : int, optional
        Should be an integer in $\{0, 1, ..., n-1\}$.
        Defaults to the principal root if $0$.

    evaluate : bool, optional
        The parameter determines if the expression should be evaluated.
        If ``None``, its value is taken from
        ``global_parameters.evaluate``.

    Examples
    ========

    >>> from sympy import root, Rational
    >>> from sympy.abc import x, n

    >>> root(x, 2)
    sqrt(x)

    >>> root(x, 3)
    x**(1/3)

    >>> root(x, n)
    x**(1/n)

    >>> root(x, -Rational(2, 3))
    x**(-3/2)

    To get the k-th n-th root, specify k:

    >>> root(-2, 3, 2)
    -(-1)**(2/3)*2**(1/3)

    To get all n n-th roots you can use the rootof function.
    The following examples show the roots of unity for n
    equal 2, 3 and 4:

    >>> from sympy import rootof

    >>> [rootof(x**2 - 1, i) for i in range(2)]
    [-1, 1]

    >>> [rootof(x**3 - 1,i) for i in range(3)]
    [1, -1/2 - sqrt(3)*I/2, -1/2 + sqrt(3)*I/2]

    >>> [rootof(x**4 - 1,i) for i in range(4)]
    [-1, 1, -I, I]

    SymPy, like other symbolic algebra systems, returns the
    complex root of negative numbers. This is the principal
    root and differs from the text-book result that one might
    be expecting. For example, the cube root of -8 does not
    come back as -2:

    >>> root(-8, 3)
    2*(-1)**(1/3)

    The real_root function can be used to either make the principal
    result real (or simply to return the real root directly):

    >>> from sympy import real_root
    >>> real_root(_)
    -2
    >>> real_root(-32, 5)
    -2

    Alternatively, the n//2-th n-th root of a negative number can be
    computed with root:

    >>> root(-32, 5, 5//2)
    -2

    See Also
    ========

    sympy.polys.rootoftools.rootof
    sympy.core.intfunc.integer_nthroot
    sqrt, real_root

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Square_root
    .. [2] https://en.wikipedia.org/wiki/Real_root
    .. [3] https://en.wikipedia.org/wiki/Root_of_unity
    .. [4] https://en.wikipedia.org/wiki/Principal_value
    .. [5] https://mathworld.wolfram.com/CubeRoot.html

    """
    n = sympify(n)
    if k:
        return Mul(Pow(arg, S.One/n, evaluate=evaluate), S.NegativeOne**(2*k/n), evaluate=evaluate)
    return Pow(arg, 1/n, evaluate=evaluate)


def real_root(arg, n=None, evaluate=None):
    r"""Return the real *n*'th-root of *arg* if possible.

    Parameters
    ==========

    n : int or None, optional
        If *n* is ``None``, then all instances of
        $(-n)^{1/\text{odd}}$ will be changed to $-n^{1/\text{odd}}$.
        This will only create a real root of a principal root.
        The presence of other factors may cause the result to not be
        real.

    evaluate : bool, optional
        The parameter determines if the expression should be evaluated.
        If ``None``, its value is taken from
        ``global_parameters.evaluate``.

    Examples
    ========

    >>> from sympy import root, real_root

    >>> real_root(-8, 3)
    -2
    >>> root(-8, 3)
    2*(-1)**(1/3)
    >>> real_root(_)
    -2

    If one creates a non-principal root and applies real_root, the
    result will not be real (so use with caution):

    >>> root(-8, 3, 2)
    -2*(-1)**(2/3)
    >>> real_root(_)
    -2*(-1)**(2/3)

    See Also
    ========

    sympy.polys.rootoftools.rootof
    sympy.core.intfunc.integer_nthroot
    root, sqrt
    """
    from sympy.functions.elementary.complexes import Abs, im, sign
    from sympy.functions.elementary.piecewise import Piecewise
    if n is not None:
        return Piecewise(
            (root(arg, n, evaluate=evaluate), Or(Eq(n, S.One), Eq(n, S.NegativeOne))),
            (Mul(sign(arg), root(Abs(arg), n, evaluate=evaluate), evaluate=evaluate),
            And(Eq(im(arg), S.Zero), Eq(Mod(n, 2), S.One))),
            (root(arg, n, evaluate=evaluate), True))
    rv = sympify(arg)
    n1pow = Transform(lambda x: -(-x.base)**x.exp,
                      lambda x:
                      x.is_Pow and
                      x.base.is_negative and
                      x.exp.is_Rational and
                      x.exp.p == 1 and x.exp.q % 2)
    return rv.xreplace(n1pow)

###############################################################################
############################# MINIMUM and MAXIMUM #############################
###############################################################################


class MinMaxBase(Expr, LatticeOp):
    def __new__(cls, *args, **assumptions):
        from sympy.core.parameters import global_parameters
        evaluate = assumptions.pop('evaluate', global_parameters.evaluate)
        args = (sympify(arg) for arg in args)

        # first standard filter, for cls.zero and cls.identity
        # also reshape Max(a, Max(b, c)) to Max(a, b, c)

        if evaluate:
            try:
                args = frozenset(cls._new_args_filter(args))
            except ShortCircuit:
                return cls.zero
            # remove redundant args that are easily identified
            args = cls._collapse_arguments(args, **assumptions)
            # find local zeros
            args = cls._find_localzeros(args, **assumptions)
        args = frozenset(args)

        if not args:
            return cls.identity

        if len(args) == 1:
            return list(args).pop()

        # base creation
        obj = Expr.__new__(cls, *ordered(args), **assumptions)
        obj._argset = args
        return obj

    @classmethod
    def _collapse_arguments(cls, args, **assumptions):
        """Remove redundant args.

        Examples
        ========

        >>> from sympy import Min, Max
        >>> from sympy.abc import a, b, c, d, e

        Any arg in parent that appears in any
        parent-like function in any of the flat args
        of parent can be removed from that sub-arg:

        >>> Min(a, Max(b, Min(a, c, d)))
        Min(a, Max(b, Min(c, d)))

        If the arg of parent appears in an opposite-than parent
        function in any of the flat args of parent that function
        can be replaced with the arg:

        >>> Min(a, Max(b, Min(c, d, Max(a, e))))
        Min(a, Max(b, Min(a, c, d)))
        """
        if not args:
            return args
        args = list(ordered(args))
        if cls == Min:
            other = Max
        else:
            other = Min

        # find global comparable max of Max and min of Min if a new
        # value is being introduced in these args at position 0 of
        # the ordered args
        if args[0].is_number:
            sifted = mins, maxs = [], []
            for i in args:
                for v in walk(i, Min, Max):
                    if v.args[0].is_comparable:
                        sifted[isinstance(v, Max)].append(v)
            small = Min.identity
            for i in mins:
                v = i.args[0]
                if v.is_number and (v < small) == True:
                    small = v
            big = Max.identity
            for i in maxs:
                v = i.args[0]
                if v.is_number and (v > big) == True:
                    big = v
            # at the point when this function is called from __new__,
            # there may be more than one numeric arg present since
            # local zeros have not been handled yet, so look through
            # more than the first arg
            if cls == Min:
                for arg in args:
                    if not arg.is_number:
                        break
                    if (arg < small) == True:
                        small = arg
            elif cls == Max:
                for arg in args:
                    if not arg.is_number:
                        break
                    if (arg > big) == True:
                        big = arg
            T = None
            if cls == Min:
                if small != Min.identity:
                    other = Max
                    T = small
            elif big != Max.identity:
                other = Min
                T = big
            if T is not None:
                # remove numerical redundancy
                for i in range(len(args)):
                    a = args[i]
                    if isinstance(a, other):
                        a0 = a.args[0]
                        if ((a0 > T) if other == Max else (a0 < T)) == True:
                            args[i] = cls.identity

        # remove redundant symbolic args
        def do(ai, a):
            if not isinstance(ai, (Min, Max)):
                return ai
            cond = a in ai.args
            if not cond:
                return ai.func(*[do(i, a) for i in ai.args],
                    evaluate=False)
            if isinstance(ai, cls):
                return ai.func(*[do(i, a) for i in ai.args if i != a],
                    evaluate=False)
            return a
        for i, a in enumerate(args):
            args[i + 1:] = [do(ai, a) for ai in args[i + 1:]]

        # factor out common elements as for
        # Min(Max(x, y), Max(x, z)) -> Max(x, Min(y, z))
        # and vice versa when swapping Min/Max -- do this only for the
        # easy case where all functions contain something in common;
        # trying to find some optimal subset of args to modify takes
        # too long

        def factor_minmax(args):
            is_other = lambda arg: isinstance(arg, other)
            other_args, remaining_args = sift(args, is_other, binary=True)
            if not other_args:
                return args

            # Min(Max(x, y, z), Max(x, y, u, v)) -> {x,y}, ({z}, {u,v})
            arg_sets = [set(arg.args) for arg in other_args]
            common = set.intersection(*arg_sets)
            if not common:
                return args

            new_other_args = list(common)
            arg_sets_diff = [arg_set - common for arg_set in arg_sets]

            # If any set is empty after removing common then all can be
            # discarded e.g. Min(Max(a, b, c), Max(a, b)) -> Max(a, b)
            if all(arg_sets_diff):
                other_args_diff = [other(*s, evaluate=False) for s in arg_sets_diff]
                new_other_args.append(cls(*other_args_diff, evaluate=False))

            other_args_factored = other(*new_other_args, evaluate=False)
            return remaining_args + [other_args_factored]

        if len(args) > 1:
            args = factor_minmax(args)

        return args

    @classmethod
    def _new_args_filter(cls, arg_sequence):
        """
        Generator filtering args.

        first standard filter, for cls.zero and cls.identity.
        Also reshape ``Max(a, Max(b, c))`` to ``Max(a, b, c)``,
        and check arguments for comparability
        """
        for arg in arg_sequence:
            # pre-filter, checking comparability of arguments
            if not isinstance(arg, Expr) or arg.is_extended_real is False or (
                    arg.is_number and
                    not arg.is_comparable):
                raise ValueError("The argument '%s' is not comparable." % arg)

            if arg == cls.zero:
                raise ShortCircuit(arg)
            elif arg == cls.identity:
                continue
            elif arg.func == cls:
                yield from arg.args
            else:
                yield arg

    @classmethod
    def _find_localzeros(cls, values, **options):
        """
        Sequentially allocate values to localzeros.

        When a value is identified as being more extreme than another member it
        replaces that member; if this is never true, then the value is simply
        appended to the localzeros.
        """
        localzeros = set()
        for v in values:
            is_newzero = True
            localzeros_ = list(localzeros)
            for z in localzeros_:
                if id(v) == id(z):
                    is_newzero = False
                else:
                    con = cls._is_connected(v, z)
                    if con:
                        is_newzero = False
                        if con is True or con == cls:
                            localzeros.remove(z)
                            localzeros.update([v])
            if is_newzero:
                localzeros.update([v])
        return localzeros

    @classmethod
    def _is_connected(cls, x, y):
        """
        Check if x and y are connected somehow.
        """
        for i in range(2):
            if x == y:
                return True
            t, f = Max, Min
            for op in "><":
                for j in range(2):
                    try:
                        if op == ">":
                            v = x >= y
                        else:
                            v = x <= y
                    except TypeError:
                        return False  # non-real arg
                    if not v.is_Relational:
                        return t if v else f
                    t, f = f, t
                    x, y = y, x
                x, y = y, x  # run next pass with reversed order relative to start
            # simplification can be expensive, so be conservative
            # in what is attempted
            x = factor_terms(x - y)
            y = S.Zero

        return False

    def _eval_derivative(self, s):
        # f(x).diff(s) -> x.diff(s) * f.fdiff(1)(s)
        i = 0
        l = []
        for a in self.args:
            i += 1
            da = a.diff(s)
            if da.is_zero:
                continue
            try:
                df = self.fdiff(i)
            except ArgumentIndexError:
                df = super().fdiff(i)
            l.append(df * da)
        return Add(*l)

    def _eval_rewrite_as_Abs(self, *args, **kwargs):
        from sympy.functions.elementary.complexes import Abs
        s = (args[0] + self.func(*args[1:]))/2
        d = abs(args[0] - self.func(*args[1:]))/2
        return (s + d if isinstance(self, Max) else s - d).rewrite(Abs)

    def evalf(self, n=15, **options):
        return self.func(*[a.evalf(n, **options) for a in self.args])

    def n(self, *args, **kwargs):
        return self.evalf(*args, **kwargs)

    _eval_is_algebraic = lambda s: _torf(i.is_algebraic for i in s.args)
    _eval_is_antihermitian = lambda s: _torf(i.is_antihermitian for i in s.args)
    _eval_is_commutative = lambda s: _torf(i.is_commutative for i in s.args)
    _eval_is_complex = lambda s: _torf(i.is_complex for i in s.args)
    _eval_is_composite = lambda s: _torf(i.is_composite for i in s.args)
    _eval_is_even = lambda s: _torf(i.is_even for i in s.args)
    _eval_is_finite = lambda s: _torf(i.is_finite for i in s.args)
    _eval_is_hermitian = lambda s: _torf(i.is_hermitian for i in s.args)
    _eval_is_imaginary = lambda s: _torf(i.is_imaginary for i in s.args)
    _eval_is_infinite = lambda s: _torf(i.is_infinite for i in s.args)
    _eval_is_integer = lambda s: _torf(i.is_integer for i in s.args)
    _eval_is_irrational = lambda s: _torf(i.is_irrational for i in s.args)
    _eval_is_negative = lambda s: _torf(i.is_negative for i in s.args)
    _eval_is_noninteger = lambda s: _torf(i.is_noninteger for i in s.args)
    _eval_is_nonnegative = lambda s: _torf(i.is_nonnegative for i in s.args)
    _eval_is_nonpositive = lambda s: _torf(i.is_nonpositive for i in s.args)
    _eval_is_nonzero = lambda s: _torf(i.is_nonzero for i in s.args)
    _eval_is_odd = lambda s: _torf(i.is_odd for i in s.args)
    _eval_is_polar = lambda s: _torf(i.is_polar for i in s.args)
    _eval_is_positive = lambda s: _torf(i.is_positive for i in s.args)
    _eval_is_prime = lambda s: _torf(i.is_prime for i in s.args)
    _eval_is_rational = lambda s: _torf(i.is_rational for i in s.args)
    _eval_is_real = lambda s: _torf(i.is_real for i in s.args)
    _eval_is_extended_real = lambda s: _torf(i.is_extended_real for i in s.args)
    _eval_is_transcendental = lambda s: _torf(i.is_transcendental for i in s.args)
    _eval_is_zero = lambda s: _torf(i.is_zero for i in s.args)


class Max(MinMaxBase, Application):
    r"""
    Return, if possible, the maximum value of the list.

    When number of arguments is equal one, then
    return this argument.

    When number of arguments is equal two, then
    return, if possible, the value from (a, b) that is $\ge$ the other.

    In common case, when the length of list greater than 2, the task
    is more complicated. Return only the arguments, which are greater
    than others, if it is possible to determine directional relation.

    If is not possible to determine such a relation, return a partially
    evaluated result.

    Assumptions are used to make the decision too.

    Also, only comparable arguments are permitted.

    It is named ``Max`` and not ``max`` to avoid conflicts
    with the built-in function ``max``.


    Examples
    ========

    >>> from sympy import Max, Symbol, oo
    >>> from sympy.abc import x, y, z
    >>> p = Symbol('p', positive=True)
    >>> n = Symbol('n', negative=True)

    >>> Max(x, -2)
    Max(-2, x)
    >>> Max(x, -2).subs(x, 3)
    3
    >>> Max(p, -2)
    p
    >>> Max(x, y)
    Max(x, y)
    >>> Max(x, y) == Max(y, x)
    True
    >>> Max(x, Max(y, z))
    Max(x, y, z)
    >>> Max(n, 8, p, 7, -oo)
    Max(8, p)
    >>> Max (1, x, oo)
    oo

    * Algorithm

    The task can be considered as searching of supremums in the
    directed complete partial orders [1]_.

    The source values are sequentially allocated by the isolated subsets
    in which supremums are searched and result as Max arguments.

    If the resulted supremum is single, then it is returned.

    The isolated subsets are the sets of values which are only the comparable
    with each other in the current set. E.g. natural numbers are comparable with
    each other, but not comparable with the `x` symbol. Another example: the
    symbol `x` with negative assumption is comparable with a natural number.

    Also there are "least" elements, which are comparable with all others,
    and have a zero property (maximum or minimum for all elements).
    For example, in case of $\infty$, the allocation operation is terminated
    and only this value is returned.

    Assumption:
       - if $A > B > C$ then $A > C$
       - if $A = B$ then $B$ can be removed

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Directed_complete_partial_order
    .. [2] https://en.wikipedia.org/wiki/Lattice_%28order%29

    See Also
    ========

    Min : find minimum values
    """
    zero = S.Infinity
    identity = S.NegativeInfinity

    def fdiff( self, argindex ):
        from sympy.functions.special.delta_functions import Heaviside
        n = len(self.args)
        if 0 < argindex and argindex <= n:
            argindex -= 1
            if n == 2:
                return Heaviside(self.args[argindex] - self.args[1 - argindex])
            newargs = tuple([self.args[i] for i in range(n) if i != argindex])
            return Heaviside(self.args[argindex] - Max(*newargs))
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Heaviside(self, *args, **kwargs):
        from sympy.functions.special.delta_functions import Heaviside
        return Add(*[j*Mul(*[Heaviside(j - i) for i in args if i!=j]) \
                for j in args])

    def _eval_rewrite_as_Piecewise(self, *args, **kwargs):
        return _minmax_as_Piecewise('>=', *args)

    def _eval_is_positive(self):
        return fuzzy_or(a.is_positive for a in self.args)

    def _eval_is_nonnegative(self):
        return fuzzy_or(a.is_nonnegative for a in self.args)

    def _eval_is_negative(self):
        return fuzzy_and(a.is_negative for a in self.args)


class Min(MinMaxBase, Application):
    """
    Return, if possible, the minimum value of the list.
    It is named ``Min`` and not ``min`` to avoid conflicts
    with the built-in function ``min``.

    Examples
    ========

    >>> from sympy import Min, Symbol, oo
    >>> from sympy.abc import x, y
    >>> p = Symbol('p', positive=True)
    >>> n = Symbol('n', negative=True)

    >>> Min(x, -2)
    Min(-2, x)
    >>> Min(x, -2).subs(x, 3)
    -2
    >>> Min(p, -3)
    -3
    >>> Min(x, y)
    Min(x, y)
    >>> Min(n, 8, p, -7, p, oo)
    Min(-7, n)

    See Also
    ========

    Max : find maximum values
    """
    zero = S.NegativeInfinity
    identity = S.Infinity

    def fdiff( self, argindex ):
        from sympy.functions.special.delta_functions import Heaviside
        n = len(self.args)
        if 0 < argindex and argindex <= n:
            argindex -= 1
            if n == 2:
                return Heaviside( self.args[1-argindex] - self.args[argindex] )
            newargs = tuple([ self.args[i] for i in range(n) if i != argindex])
            return Heaviside( Min(*newargs) - self.args[argindex] )
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Heaviside(self, *args, **kwargs):
        from sympy.functions.special.delta_functions import Heaviside
        return Add(*[j*Mul(*[Heaviside(i-j) for i in args if i!=j]) \
                for j in args])

    def _eval_rewrite_as_Piecewise(self, *args, **kwargs):
        return _minmax_as_Piecewise('<=', *args)

    def _eval_is_positive(self):
        return fuzzy_and(a.is_positive for a in self.args)

    def _eval_is_nonnegative(self):
        return fuzzy_and(a.is_nonnegative for a in self.args)

    def _eval_is_negative(self):
        return fuzzy_or(a.is_negative for a in self.args)


class Rem(DefinedFunction):
    """Returns the remainder when ``p`` is divided by ``q`` where ``p`` is finite
    and ``q`` is not equal to zero. The result, ``p - int(p/q)*q``, has the same sign
    as the divisor.

    Parameters
    ==========

    p : Expr
        Dividend.

    q : Expr
        Divisor.

    Notes
    =====

    ``Rem`` corresponds to the ``%`` operator in C.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy import Rem
    >>> Rem(x**3, y)
    Rem(x**3, y)
    >>> Rem(x**3, y).subs({x: -5, y: 3})
    -2

    See Also
    ========

    Mod
    """
    kind = NumberKind

    @classmethod
    def eval(cls, p, q):
        """Return the function remainder if both p, q are numbers and q is not
        zero.
        """

        if q.is_zero:
            raise ZeroDivisionError("Division by zero")
        if p is S.NaN or q is S.NaN or p.is_finite is False or q.is_finite is False:
            return S.NaN
        if p is S.Zero or p in (q, -q) or (p.is_integer and q == 1):
            return S.Zero

        if q.is_Number:
            if p.is_Number:
                return p - Integer(p/q)*q
<|endoftext|>from __future__ import annotations
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import DefinedFunction, ArgumentIndexError, PoleError, expand_mul
from sympy.core.logic import fuzzy_not, fuzzy_or, FuzzyBool, fuzzy_and
from sympy.core.mod import Mod
from sympy.core.numbers import Rational, pi, Integer, Float, equal_valued
from sympy.core.relational import Ne, Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial, RisingFactorial
from sympy.functions.combinatorial.numbers import bernoulli, euler
from sympy.functions.elementary.complexes import arg as arg_f, im, re
from sympy.functions.elementary.exponential import log, exp
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt, Min, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary._trigonometric_special import (
    cos_table, ipartfrac, fermat_coords)
from sympy.logic.boolalg import And
from sympy.ntheory import factorint
from sympy.polys.specialpolys import symmetric_poly
from sympy.utilities.iterables import numbered_symbols


###############################################################################
########################## UTILITIES ##########################################
###############################################################################


def _imaginary_unit_as_coefficient(arg):
    """ Helper to extract symbolic coefficient for imaginary unit """
    if isinstance(arg, Float):
        return None
    else:
        return arg.as_coefficient(S.ImaginaryUnit)

###############################################################################
########################## TRIGONOMETRIC FUNCTIONS ############################
###############################################################################


class TrigonometricFunction(DefinedFunction):
    """Base class for trigonometric functions. """

    unbranched = True
    _singularities = (S.ComplexInfinity,)

    def _eval_is_rational(self):
        s = self.func(*self.args)
        if s.func == self.func:
            if s.args[0].is_rational and fuzzy_not(s.args[0].is_zero):
                return False
        else:
            return s.is_rational

    def _eval_is_algebraic(self):
        s = self.func(*self.args)
        if s.func == self.func:
            if fuzzy_not(self.args[0].is_zero) and self.args[0].is_algebraic:
                return False
            pi_coeff = _pi_coeff(self.args[0])
            if pi_coeff is not None and pi_coeff.is_rational:
                return True
        else:
            return s.is_algebraic

    def _eval_expand_complex(self, deep=True, **hints):
        re_part, im_part = self.as_real_imag(deep=deep, **hints)
        return re_part + im_part*S.ImaginaryUnit

    def _as_real_imag(self, deep=True, **hints):
        if self.args[0].is_extended_real:
            if deep:
                hints['complex'] = False
                return (self.args[0].expand(deep, **hints), S.Zero)
            else:
                return (self.args[0], S.Zero)
        if deep:
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            re, im = self.args[0].as_real_imag()
        return (re, im)

    def _period(self, general_period, symbol=None):
        f = expand_mul(self.args[0])
        if symbol is None:
            symbol = tuple(f.free_symbols)[0]

        if not f.has(symbol):
            return S.Zero

        if f == symbol:
            return general_period

        if symbol in f.free_symbols:
            if f.is_Mul:
                g, h = f.as_independent(symbol)
                if h == symbol:
                    return general_period/abs(g)

            if f.is_Add:
                a, h = f.as_independent(symbol)
                g, h = h.as_independent(symbol, as_Add=False)
                if h == symbol:
                    return general_period/abs(g)

        raise NotImplementedError("Use the periodicity function instead.")


@cacheit
def _table2():
    # If nested sqrt's are worse than un-evaluation
    # you can require q to be in (1, 2, 3, 4, 6, 12)
    # q <= 12, q=15, q=20, q=24, q=30, q=40, q=60, q=120 return
    # expressions with 2 or fewer sqrt nestings.
    return {
        12: (3, 4),
        20: (4, 5),
        30: (5, 6),
        15: (6, 10),
        24: (6, 8),
        40: (8, 10),
        60: (20, 30),
        120: (40, 60)
    }


def _peeloff_pi(arg):
    r"""
    Split ARG into two parts, a "rest" and a multiple of $\pi$.
    This assumes ARG to be an Add.
    The multiple of $\pi$ returned in the second position is always a Rational.

    Examples
    ========

    >>> from sympy.functions.elementary.trigonometric import _peeloff_pi
    >>> from sympy import pi
    >>> from sympy.abc import x, y
    >>> _peeloff_pi(x + pi/2)
    (x, 1/2)
    >>> _peeloff_pi(x + 2*pi/3 + pi*y)
    (x + pi*y + pi/6, 1/2)

    """
    pi_coeff = S.Zero
    rest_terms = []
    for a in Add.make_args(arg):
        K = a.coeff(pi)
        if K and K.is_rational:
            pi_coeff += K
        else:
            rest_terms.append(a)

    if pi_coeff is S.Zero:
        return arg, S.Zero

    m1 = (pi_coeff % S.Half)
    m2 = pi_coeff - m1
    if m2.is_integer or ((2*m2).is_integer and m2.is_even is False):
        return Add(*(rest_terms + [m1*pi])), m2
    return arg, S.Zero


def _pi_coeff(arg: Expr, cycles: int = 1) -> Expr | None:
    r"""
    When arg is a Number times $\pi$ (e.g. $3\pi/2$) then return the Number
    normalized to be in the range $[0, 2]$, else `None`.

    When an even multiple of $\pi$ is encountered, if it is multiplying
    something with known parity then the multiple is returned as 0 otherwise
    as 2.

    Examples
    ========

    >>> from sympy.functions.elementary.trigonometric import _pi_coeff
    >>> from sympy import pi, Dummy
    >>> from sympy.abc import x
    >>> _pi_coeff(3*x*pi)
    3*x
    >>> _pi_coeff(11*pi/7)
    11/7
    >>> _pi_coeff(-11*pi/7)
    3/7
    >>> _pi_coeff(4*pi)
    0
    >>> _pi_coeff(5*pi)
    1
    >>> _pi_coeff(5.0*pi)
    1
    >>> _pi_coeff(5.5*pi)
    3/2
    >>> _pi_coeff(2 + pi)

    >>> _pi_coeff(2*Dummy(integer=True)*pi)
    2
    >>> _pi_coeff(2*Dummy(even=True)*pi)
    0

    """
    if arg is pi:
        return S.One
    elif not arg:
        return S.Zero
    elif arg.is_Mul:
        cx = arg.coeff(pi)
        if cx:
            c, x = cx.as_coeff_Mul()  # pi is not included as coeff
            if c.is_Float:
                # recast exact binary fractions to Rationals
                f = abs(c) % 1
                if f != 0:
                    p = -int(round(log(f, 2).evalf()))
                    m = 2**p
                    cm = c*m
                    i = int(cm)
                    if equal_valued(i, cm):
                        c = Rational(i, m)
                        cx = c*x
                else:
                    c = Rational(int(c))
                    cx = c*x
            if x.is_integer:
                c2 = c % 2
                if c2 == 1:
                    return x
                elif not c2:
                    if x.is_even is not None:  # known parity
                        return S.Zero
                    return Integer(2)
                else:
                    return c2*x
            return cx
    elif arg.is_zero:
        return S.Zero
    return None


class sin(TrigonometricFunction):
    r"""
    The sine function.

    Returns the sine of x (measured in radians).

    Explanation
    ===========

    This function will evaluate automatically in the
    case $x/\pi$ is some rational number [4]_.  For example,
    if $x$ is a multiple of $\pi$, $\pi/2$, $\pi/3$, $\pi/4$, and $\pi/6$.

    Examples
    ========

    >>> from sympy import sin, pi
    >>> from sympy.abc import x
    >>> sin(x**2).diff(x)
    2*x*cos(x**2)
    >>> sin(1).diff(x)
    0
    >>> sin(pi)
    0
    >>> sin(pi/2)
    1
    >>> sin(pi/6)
    1/2
    >>> sin(pi/12)
    -sqrt(2)/4 + sqrt(6)/4


    See Also
    ========

    csc, cos, sec, tan, cot
    asin, acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.14
    .. [3] https://functions.wolfram.com/ElementaryFunctions/Sin
    .. [4] https://mathworld.wolfram.com/TrigonometryAngles.html

    """

    def period(self, symbol=None):
        return self._period(2*pi, symbol)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return cos(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        from sympy.calculus.accumulationbounds import AccumBounds
        from sympy.sets.setexpr import SetExpr
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg.is_zero:
                return S.Zero
            elif arg in (S.Infinity, S.NegativeInfinity):
                return AccumBounds(-1, 1)

        if arg is S.ComplexInfinity:
            return S.NaN

        if isinstance(arg, AccumBounds):
            from sympy.sets.sets import FiniteSet
            min, max = arg.min, arg.max
            d = floor(min/(2*pi))
            if min is not S.NegativeInfinity:
                min = min - d*2*pi
            if max is not S.Infinity:
                max = max - d*2*pi
            if AccumBounds(min, max).intersection(FiniteSet(pi/2, pi*Rational(5, 2))) \
                    is not S.EmptySet and \
                    AccumBounds(min, max).intersection(FiniteSet(pi*Rational(3, 2),
                        pi*Rational(7, 2))) is not S.EmptySet:
                return AccumBounds(-1, 1)
            elif AccumBounds(min, max).intersection(FiniteSet(pi/2, pi*Rational(5, 2))) \
                    is not S.EmptySet:
                return AccumBounds(Min(sin(min), sin(max)), 1)
            elif AccumBounds(min, max).intersection(FiniteSet(pi*Rational(3, 2), pi*Rational(8, 2))) \
                        is not S.EmptySet:
                return AccumBounds(-1, Max(sin(min), sin(max)))
            else:
                return AccumBounds(Min(sin(min), sin(max)),
                                Max(sin(min), sin(max)))
        elif isinstance(arg, SetExpr):
            return arg._eval_func(cls)

        if arg.could_extract_minus_sign():
            return -cls(-arg)

        i_coeff = _imaginary_unit_as_coefficient(arg)
        if i_coeff is not None:
            from sympy.functions.elementary.hyperbolic import sinh
            return S.ImaginaryUnit*sinh(i_coeff)

        pi_coeff = _pi_coeff(arg)
        if pi_coeff is not None:
            if pi_coeff.is_integer:
                return S.Zero

            if (2*pi_coeff).is_integer:
                # is_even-case handled above as then pi_coeff.is_integer,
                # so check if known to be not even
                if pi_coeff.is_even is False:
                    return S.NegativeOne**(pi_coeff - S.Half)

            if not pi_coeff.is_Rational:
                narg = pi_coeff*pi
                if narg != arg:
                    return cls(narg)
                return None

            # https://github.com/sympy/sympy/issues/6048
            # transform a sine to a cosine, to avoid redundant code
            if pi_coeff.is_Rational:
                x = pi_coeff % 2
                if x > 1:
                    return -cls((x % 1)*pi)
                if 2*x > 1:
                    return cls((1 - x)*pi)
                narg = ((pi_coeff + Rational(3, 2)) % 2)*pi
                result = cos(narg)
                if not isinstance(result, cos):
                    return result
                if pi_coeff*pi != arg:
                    return cls(pi_coeff*pi)
                return None

        if arg.is_Add:
            x, m = _peeloff_pi(arg)
            if m:
                m = m*pi
                return sin(m)*cos(x) + cos(m)*sin(x)

        if arg.is_zero:
            return S.Zero

        if isinstance(arg, asin):
            return arg.args[0]

        if isinstance(arg, atan):
            x = arg.args[0]
            return x/sqrt(1 + x**2)

        if isinstance(arg, atan2):
            y, x = arg.args
            return y/sqrt(x**2 + y**2)

        if isinstance(arg, acos):
            x = arg.args[0]
            return sqrt(1 - x**2)

        if isinstance(arg, acot):
            x = arg.args[0]
            return 1/(sqrt(1 + 1/x**2)*x)

        if isinstance(arg, acsc):
            x = arg.args[0]
            return 1/x

        if isinstance(arg, asec):
            x = arg.args[0]
            return sqrt(1 - 1/x**2)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)

            if len(previous_terms) > 2:
                p = previous_terms[-2]
                return -p*x**2/(n*(n - 1))
            else:
                return S.NegativeOne**(n//2)*x**n/factorial(n)

    def _eval_nseries(self, x, n, logx, cdir=0):
        arg = self.args[0]
        if logx is not None:
            arg = arg.subs(log(x), logx)
        if arg.subs(x, 0).has(S.NaN, S.ComplexInfinity):
            raise PoleError("Cannot expand %s around 0" % (self))
        return super()._eval_nseries(x, n=n, logx=logx, cdir=cdir)

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        from sympy.functions.elementary.hyperbolic import HyperbolicFunction
        I = S.ImaginaryUnit
        if isinstance(arg, (TrigonometricFunction, HyperbolicFunction)):
            arg = arg.func(arg.args[0]).rewrite(exp)
        return (exp(arg*I) - exp(-arg*I))/(2*I)

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        if isinstance(arg, log):
            I = S.ImaginaryUnit
            x = arg.args[0]
            return I*x**-I/2 - I*x**I /2

    def _eval_rewrite_as_cos(self, arg, **kwargs):
        return cos(arg - pi/2, evaluate=False)

    def _eval_rewrite_as_tan(self, arg, **kwargs):
        tan_half = tan(S.Half*arg)
        return 2*tan_half/(1 + tan_half**2)

    def _eval_rewrite_as_sincos(self, arg, **kwargs):
        return sin(arg)*cos(arg)/cos(arg)

    def _eval_rewrite_as_cot(self, arg, **kwargs):
        cot_half = cot(S.Half*arg)
        return Piecewise((0, And(Eq(im(arg), 0), Eq(Mod(arg, pi), 0))),
                         (2*cot_half/(1 + cot_half**2), True))

    def _eval_rewrite_as_pow(self, arg, **kwargs):
        return self.rewrite(cos, **kwargs).rewrite(pow, **kwargs)

    def _eval_rewrite_as_sqrt(self, arg, **kwargs):
        return self.rewrite(cos, **kwargs).rewrite(sqrt, **kwargs)

    def _eval_rewrite_as_csc(self, arg, **kwargs):
        return 1/csc(arg)

    def _eval_rewrite_as_sec(self, arg, **kwargs):
        return 1/sec(arg - pi/2, evaluate=False)

    def _eval_rewrite_as_sinc(self, arg, **kwargs):
        return arg*sinc(arg)

    def _eval_rewrite_as_besselj(self, arg, **kwargs):
        from sympy.functions.special.bessel import besselj
        return sqrt(pi*arg/2)*besselj(S.Half, arg)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
        from sympy.functions.elementary.hyperbolic import cosh, sinh
        re, im = self._as_real_imag(deep=deep, **hints)
        return (sin(re)*cosh(im), cos(re)*sinh(im))

    def _eval_expand_trig(self, **hints):
        from sympy.functions.special.polynomials import chebyshevt, chebyshevu
        arg = self.args[0]
        x = None
        if arg.is_Add:  # TODO, implement more if deep stuff here
            # TODO: Do this more efficiently for more than two terms
            x, y = arg.as_two_terms()
            sx = sin(x, evaluate=False)._eval_expand_trig()
            sy = sin(y, evaluate=False)._eval_expand_trig()
            cx = cos(x, evaluate=False)._eval_expand_trig()
            cy = cos(y, evaluate=False)._eval_expand_trig()
            return sx*cy + sy*cx
        elif arg.is_Mul:
            n, x = arg.as_coeff_Mul(rational=True)
            if n.is_Integer:  # n will be positive because of .eval
                # canonicalization

                # See https://mathworld.wolfram.com/Multiple-AngleFormulas.html
                if n.is_odd:
                    return S.NegativeOne**((n - 1)/2)*chebyshevt(n, sin(x))
                else:
                    return expand_mul(S.NegativeOne**(n/2 - 1)*cos(x)*
                                      chebyshevu(n - 1, sin(x)), deep=False)
        return sin(arg)

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.calculus.accumulationbounds import AccumBounds
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        n = x0/pi
        if n.is_integer:
            lt = (arg - n*pi).as_leading_term(x)
            return (S.NegativeOne**n)*lt
        if x0 is S.ComplexInfinity:
            x0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if x0 in [S.Infinity, S.NegativeInfinity]:
            return AccumBounds(-1, 1)
        return self.func(x0) if x0.is_finite else self

    def _eval_is_extended_real(self):
        if self.args[0].is_extended_real:
            return True

    def _eval_is_finite(self):
        arg = self.args[0]
        if arg.is_extended_real:
            return True

    def _eval_is_zero(self):
        rest, pi_mult = _peeloff_pi(self.args[0])
        if rest.is_zero:
            return pi_mult.is_integer

    def _eval_is_complex(self):
        if self.args[0].is_extended_real \
                or self.args[0].is_complex:
            return True


class cos(TrigonometricFunction):
    """
    The cosine function.

    Returns the cosine of x (measured in radians).

    Explanation
    ===========

    See :func:`sin` for notes about automatic evaluation.

    Examples
    ========

    >>> from sympy import cos, pi
    >>> from sympy.abc import x
    >>> cos(x**2).diff(x)
    -2*x*sin(x**2)
    >>> cos(1).diff(x)
    0
    >>> cos(pi)
    -1
    >>> cos(pi/2)
    0
    >>> cos(2*pi/3)
    -1/2
    >>> cos(pi/12)
    sqrt(2)/4 + sqrt(6)/4

    See Also
    ========

    sin, csc, sec, tan, cot
    asin, acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.14
    .. [3] https://functions.wolfram.com/ElementaryFunctions/Cos

    """

    def period(self, symbol=None):
        return self._period(2*pi, symbol)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return -sin(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        from sympy.functions.special.polynomials import chebyshevt
        from sympy.calculus.accumulationbounds import AccumBounds
        from sympy.sets.setexpr import SetExpr
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg.is_zero:
                return S.One
            elif arg in (S.Infinity, S.NegativeInfinity):
                # In this case it is better to return AccumBounds(-1, 1)
                # rather than returning S.NaN, since AccumBounds(-1, 1)
                # preserves the information that sin(oo) is between
                # -1 and 1, where S.NaN does not do that.
                return AccumBounds(-1, 1)

        if arg is S.ComplexInfinity:
            return S.NaN

        if isinstance(arg, AccumBounds):
            return sin(arg + pi/2)
        elif isinstance(arg, SetExpr):
            return arg._eval_func(cls)

        if arg.is_extended_real and arg.is_finite is False:
            return AccumBounds(-1, 1)

        if arg.could_extract_minus_sign():
            return cls(-arg)

        i_coeff = _imaginary_unit_as_coefficient(arg)
        if i_coeff is not None:
            from sympy.functions.elementary.hyperbolic import cosh
            return cosh(i_coeff)

        pi_coeff = _pi_coeff(arg)
        if pi_coeff is not None:
            if pi_coeff.is_integer:
                return (S.NegativeOne)**pi_coeff

            if (2*pi_coeff).is_integer:
                # is_even-case handled above as then pi_coeff.is_integer,
                # so check if known to be not even
                if pi_coeff.is_even is False:
                    return S.Zero

            if not pi_coeff.is_Rational:
                narg = pi_coeff*pi
                if narg != arg:
                    return cls(narg)
                return None

            # cosine formula #####################
            # https://github.com/sympy/sympy/issues/6048
            # explicit calculations are performed for
            # cos(k pi/n) for n = 8,10,12,15,20,24,30,40,60,120
            # Some other exact values like cos(k pi/240) can be
            # calculated using a partial-fraction decomposition
            # by calling cos( X ).rewrite(sqrt)
            if pi_coeff.is_Rational:
                q = pi_coeff.q
                p = pi_coeff.p % (2*q)
                if p > q:
                    narg = (pi_coeff - 1)*pi
                    return -cls(narg)
                if 2*p > q:
                    narg = (1 - pi_coeff)*pi
                    return -cls(narg)

                # If nested sqrt's are worse than un-evaluation
                # you can require q to be in (1, 2, 3, 4, 6, 12)
                # q <= 12, q=15, q=20, q=24, q=30, q=40, q=60, q=120 return
                # expressions with 2 or fewer sqrt nestings.
                table2 = _table2()
                if q in table2:
                    a, b = table2[q]
                    a, b = p*pi/a, p*pi/b
                    nvala, nvalb = cls(a), cls(b)
                    if None in (nvala, nvalb):
                        return None
                    return nvala*nvalb + cls(pi/2 - a)*cls(pi/2 - b)

                if q > 12:
                    return None

                cst_table_some = {
                    3: S.Half,
                    5: (sqrt(5) + 1) / 4,
                }
                if q in cst_table_some:
                    cts = cst_table_some[pi_coeff.q]
                    return chebyshevt(pi_coeff.p, cts).expand()

                if 0 == q % 2:
                    narg = (pi_coeff*2)*pi
                    nval = cls(narg)
                    if None == nval:
                        return None
                    x = (2*pi_coeff + 1)/2
                    sign_cos = (-1)**((-1 if x < 0 else 1)*int(abs(x)))
                    return sign_cos*sqrt( (1 + nval)/2 )
            return None

        if arg.is_Add:
            x, m = _peeloff_pi(arg)
            if m:
                m = m*pi
                return cos(m)*cos(x) - sin(m)*sin(x)

        if arg.is_zero:
            return S.One

        if isinstance(arg, acos):
            return arg.args[0]

        if isinstance(arg, atan):
            x = arg.args[0]
            return 1/sqrt(1 + x**2)

        if isinstance(arg, atan2):
            y, x = arg.args
            return x/sqrt(x**2 + y**2)

        if isinstance(arg, asin):
            x = arg.args[0]
            return sqrt(1 - x ** 2)

        if isinstance(arg, acot):
            x = arg.args[0]
            return 1/sqrt(1 + 1/x**2)

        if isinstance(arg, acsc):
            x = arg.args[0]
            return sqrt(1 - 1/x**2)

        if isinstance(arg, asec):
            x = arg.args[0]
            return 1/x

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)

            if len(previous_terms) > 2:
                p = previous_terms[-2]
                return -p*x**2/(n*(n - 1))
            else:
                return S.NegativeOne**(n//2)*x**n/factorial(n)

    def _eval_nseries(self, x, n, logx, cdir=0):
        arg = self.args[0]
        if logx is not None:
            arg = arg.subs(log(x), logx)
        if arg.subs(x, 0).has(S.NaN, S.ComplexInfinity):
            raise PoleError("Cannot expand %s around 0" % (self))
        return super()._eval_nseries(x, n=n, logx=logx, cdir=cdir)

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        I = S.ImaginaryUnit
        from sympy.functions.elementary.hyperbolic import HyperbolicFunction
        if isinstance(arg, (TrigonometricFunction, HyperbolicFunction)):
            arg = arg.func(arg.args[0]).rewrite(exp, **kwargs)
        return (exp(arg*I) + exp(-arg*I))/2

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        if isinstance(arg, log):
            I = S.ImaginaryUnit
            x = arg.args[0]
            return x**I/2 + x**-I/2

    def _eval_rewrite_as_sin(self, arg, **kwargs):
        return sin(arg + pi/2, evaluate=False)

    def _eval_rewrite_as_tan(self, arg, **kwargs):
        tan_half = tan(S.Half*arg)**2
        return (1 - tan_half)/(1 + tan_half)

    def _eval_rewrite_as_sincos(self, arg, **kwargs):
        return sin(arg)*cos(arg)/sin(arg)

    def _eval_rewrite_as_cot(self, arg, **kwargs):
        cot_half = cot(S.Half*arg)**2
        return Piecewise((1, And(Eq(im(arg), 0), Eq(Mod(arg, 2*pi), 0))),
                         ((cot_half - 1)/(cot_half + 1), True))

    def _eval_rewrite_as_pow(self, arg, **kwargs):
        return self._eval_rewrite_as_sqrt(arg, **kwargs)

    def _eval_rewrite_as_sqrt(self, arg: Expr, **kwargs):
        from sympy.functions.special.polynomials import chebyshevt

        pi_coeff = _pi_coeff(arg)
        if pi_coeff is None:
            return None

        if isinstance(pi_coeff, Integer):
            return None

        if not isinstance(pi_coeff, Rational):
            return None

        cst_table_some = cos_table()

        if pi_coeff.q in cst_table_some:
            rv = chebyshevt(pi_coeff.p, cst_table_some[pi_coeff.q]())
            if pi_coeff.q < 257:
                rv = rv.expand()
            return rv

        if not pi_coeff.q % 2:  # recursively remove factors of 2
            pico2 = pi_coeff * 2
            nval = cos(pico2 * pi).rewrite(sqrt, **kwargs)
            x = (pico2 + 1) / 2
            sign_cos = -1 if int(x) % 2 else 1
            return sign_cos * sqrt((1 + nval) / 2)

        FC = fermat_coords(pi_coeff.q)
        if FC:
            denoms = FC
        else:
            denoms = [b**e for b, e in factorint(pi_coeff.q).items()]

        apart = ipartfrac(*denoms)
        decomp = (pi_coeff.p * Rational(n, d) for n, d in zip(apart, denoms))
        X = [(x[1], x[0]*pi) for x in zip(decomp, numbered_symbols('z'))]
        pcls = cos(sum(x[0] for x in X))._eval_expand_trig().subs(X)

        if not FC or len(FC) == 1:
