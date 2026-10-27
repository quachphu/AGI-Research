        args = _flatten(args)

        # Multiplication of no sequences is EmptySequence
        if not args:
            return S.EmptySequence

        if Intersection(*(a.interval for a in args)) is S.EmptySet:
            return S.EmptySequence

        # reduce using known rules
        if evaluate:
            return SeqMul.reduce(args)

        args = list(ordered(args, SeqBase._start_key))

        return Basic.__new__(cls, *args)

    @staticmethod
    def reduce(args):
        """Simplify a :class:`SeqMul` using known rules.

        Explanation
        ===========

        Iterates through all pairs and ask the constituent
        sequences if they can simplify themselves with any other constituent.

        Notes
        =====

        adapted from ``Union.reduce``

        """
        new_args = True
        while new_args:
            for id1, s in enumerate(args):
                new_args = False
                for id2, t in enumerate(args):
                    if id1 == id2:
                        continue
                    new_seq = s._mul(t)
                    # This returns None if s does not know how to multiply
                    # with t. Returns the newly multiplied sequence otherwise
                    if new_seq is not None:
                        new_args = [a for a in args if a not in (s, t)]
                        new_args.append(new_seq)
                        break
                if new_args:
                    args = new_args
                    break

        if len(args) == 1:
            return args.pop()
        else:
            return SeqMul(args, evaluate=False)

    def _eval_coeff(self, pt):
        """multiplies the coefficients of all the sequences at point pt"""
        val = 1
        for a in self.args:
            val *= a.coeff(pt)
        return val
<|endoftext|>from sympy.core.sympify import sympify


def aseries(expr, x=None, n=6, bound=0, hir=False):
    """
    See the docstring of Expr.aseries() for complete details of this wrapper.

    """
    expr = sympify(expr)
    return expr.aseries(x, n, bound, hir)
<|endoftext|>"""A functions module, includes all the standard functions.

Combinatorial - factorial, fibonacci, harmonic, bernoulli...
Elementary - hyperbolic, trigonometric, exponential, floor and ceiling, sqrt...
Special - gamma, zeta,spherical harmonics...
"""

from sympy.functions.combinatorial.factorials import (factorial, factorial2,
        rf, ff, binomial, RisingFactorial, FallingFactorial, subfactorial)
from sympy.functions.combinatorial.numbers import (carmichael, fibonacci, lucas, tribonacci,
        harmonic, bernoulli, bell, euler, catalan, genocchi, andre, partition, divisor_sigma,
        udivisor_sigma, legendre_symbol, jacobi_symbol, kronecker_symbol, mobius,
        primenu, primeomega, totient, reduced_totient, primepi, motzkin)
from sympy.functions.elementary.miscellaneous import (sqrt, root, Min, Max,
        Id, real_root, cbrt, Rem)
from sympy.functions.elementary.complexes import (re, im, sign, Abs,
        conjugate, arg, polar_lift, periodic_argument, unbranched_argument,
        principal_branch, transpose, adjoint, polarify, unpolarify)
from sympy.functions.elementary.trigonometric import (sin, cos, tan,
        sec, csc, cot, sinc, asin, acos, atan, asec, acsc, acot, atan2)
from sympy.functions.elementary.exponential import (exp_polar, exp, log,
        LambertW)
from sympy.functions.elementary.hyperbolic import (sinh, cosh, tanh, coth,
        sech, csch, asinh, acosh, atanh, acoth, asech, acsch)
from sympy.functions.elementary.integers import floor, ceiling, frac
from sympy.functions.elementary.piecewise import (Piecewise, piecewise_fold,
                                                  piecewise_exclusive)
from sympy.functions.special.error_functions import (erf, erfc, erfi, erf2,
        erfinv, erfcinv, erf2inv, Ei, expint, E1, li, Li, Si, Ci, Shi, Chi,
        fresnels, fresnelc)
from sympy.functions.special.gamma_functions import (gamma, lowergamma,
        uppergamma, polygamma, loggamma, digamma, trigamma, multigamma)
from sympy.functions.special.zeta_functions import (dirichlet_eta, zeta,
        lerchphi, polylog, stieltjes, riemann_xi)
from sympy.functions.special.tensor_functions import (Eijk, LeviCivita,
        KroneckerDelta)
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.functions.special.delta_functions import DiracDelta, Heaviside
from sympy.functions.special.bsplines import bspline_basis, bspline_basis_set, interpolating_spline
from sympy.functions.special.bessel import (besselj, bessely, besseli, besselk,
        hankel1, hankel2, jn, yn, jn_zeros, hn1, hn2, airyai, airybi, airyaiprime, airybiprime, marcumq)
from sympy.functions.special.hyper import hyper, meijerg, appellf1
from sympy.functions.special.polynomials import (legendre, assoc_legendre,
        hermite, hermite_prob, chebyshevt, chebyshevu, chebyshevu_root,
        chebyshevt_root, laguerre, assoc_laguerre, gegenbauer, jacobi, jacobi_normalized)
from sympy.functions.special.spherical_harmonics import Ynm, Ynm_c, Znm
from sympy.functions.special.elliptic_integrals import (elliptic_k,
        elliptic_f, elliptic_e, elliptic_pi)
from sympy.functions.special.beta_functions import beta, betainc, betainc_regularized
from sympy.functions.special.mathieu_functions import (mathieus, mathieuc,
        mathieusprime, mathieucprime)
ln = log

__all__ = [
    'factorial', 'factorial2', 'rf', 'ff', 'binomial', 'RisingFactorial',
    'FallingFactorial', 'subfactorial',

    'carmichael', 'fibonacci', 'lucas', 'motzkin', 'tribonacci', 'harmonic',
    'bernoulli', 'bell', 'euler', 'catalan', 'genocchi', 'andre', 'partition',
    'divisor_sigma', 'udivisor_sigma', 'legendre_symbol', 'jacobi_symbol', 'kronecker_symbol',
    'mobius', 'primenu', 'primeomega', 'totient', 'reduced_totient', 'primepi',

    'sqrt', 'root', 'Min', 'Max', 'Id', 'real_root', 'cbrt', 'Rem',

    're', 'im', 'sign', 'Abs', 'conjugate', 'arg', 'polar_lift',
    'periodic_argument', 'unbranched_argument', 'principal_branch',
    'transpose', 'adjoint', 'polarify', 'unpolarify',

    'sin', 'cos', 'tan', 'sec', 'csc', 'cot', 'sinc', 'asin', 'acos', 'atan',
    'asec', 'acsc', 'acot', 'atan2',

    'exp_polar', 'exp', 'ln', 'log', 'LambertW',

    'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch', 'asinh', 'acosh', 'atanh',
    'acoth', 'asech', 'acsch',

    'floor', 'ceiling', 'frac',

    'Piecewise', 'piecewise_fold', 'piecewise_exclusive',

    'erf', 'erfc', 'erfi', 'erf2', 'erfinv', 'erfcinv', 'erf2inv', 'Ei',
    'expint', 'E1', 'li', 'Li', 'Si', 'Ci', 'Shi', 'Chi', 'fresnels',
    'fresnelc',

    'gamma', 'lowergamma', 'uppergamma', 'polygamma', 'loggamma', 'digamma',
    'trigamma', 'multigamma',

    'dirichlet_eta', 'zeta', 'lerchphi', 'polylog', 'stieltjes', 'riemann_xi',

    'Eijk', 'LeviCivita', 'KroneckerDelta',

    'SingularityFunction',

    'DiracDelta', 'Heaviside',

    'bspline_basis', 'bspline_basis_set', 'interpolating_spline',

    'besselj', 'bessely', 'besseli', 'besselk', 'hankel1', 'hankel2', 'jn',
    'yn', 'jn_zeros', 'hn1', 'hn2', 'airyai', 'airybi', 'airyaiprime',
    'airybiprime', 'marcumq',

    'hyper', 'meijerg', 'appellf1',

    'legendre', 'assoc_legendre', 'hermite', 'hermite_prob', 'chebyshevt',
    'chebyshevu', 'chebyshevu_root', 'chebyshevt_root', 'laguerre',
    'assoc_laguerre', 'gegenbauer', 'jacobi', 'jacobi_normalized',

    'Ynm', 'Ynm_c', 'Znm',

    'elliptic_k', 'elliptic_f', 'elliptic_e', 'elliptic_pi',

    'beta', 'betainc', 'betainc_regularized',

    'mathieus', 'mathieuc', 'mathieusprime', 'mathieucprime',
]
<|endoftext|>from __future__ import annotations
from functools import reduce

from sympy.core import S, sympify, Dummy, Mod
from sympy.core.cache import cacheit
from sympy.core.function import DefinedFunction, ArgumentIndexError, PoleError
from sympy.core.logic import fuzzy_and
from sympy.core.numbers import Integer, pi, I
from sympy.core.relational import Eq
from sympy.external.gmpy import gmpy as _gmpy
from sympy.ntheory import sieve
from sympy.ntheory.residue_ntheory import binomial_mod
from sympy.polys.polytools import Poly

from math import factorial as _factorial, prod, sqrt as _sqrt

class CombinatorialFunction(DefinedFunction):
    """Base class for combinatorial functions. """

    def _eval_simplify(self, **kwargs):
        from sympy.simplify.combsimp import combsimp
        # combinatorial function with non-integer arguments is
        # automatically passed to gammasimp
        expr = combsimp(self)
        measure = kwargs['measure']
        if measure(expr) <= kwargs['ratio']*measure(self):
            return expr
        return self


###############################################################################
######################## FACTORIAL and MULTI-FACTORIAL ########################
###############################################################################


class factorial(CombinatorialFunction):
    r"""Implementation of factorial function over nonnegative integers.
       By convention (consistent with the gamma function and the binomial
       coefficients), factorial of a negative integer is complex infinity.

       The factorial is very important in combinatorics where it gives
       the number of ways in which `n` objects can be permuted. It also
       arises in calculus, probability, number theory, etc.

       There is strict relation of factorial with gamma function. In
       fact `n! = gamma(n+1)` for nonnegative integers. Rewrite of this
       kind is very useful in case of combinatorial simplification.

       Computation of the factorial is done using two algorithms. For
       small arguments a precomputed look up table is used. However for bigger
       input algorithm Prime-Swing is used. It is the fastest algorithm
       known and computes `n!` via prime factorization of special class
       of numbers, called here the 'Swing Numbers'.

       Examples
       ========

       >>> from sympy import Symbol, factorial, S
       >>> n = Symbol('n', integer=True)

       >>> factorial(0)
       1

       >>> factorial(7)
       5040

       >>> factorial(-2)
       zoo

       >>> factorial(n)
       factorial(n)

       >>> factorial(2*n)
       factorial(2*n)

       >>> factorial(S(1)/2)
       factorial(1/2)

       See Also
       ========

       factorial2, RisingFactorial, FallingFactorial
    """

    def fdiff(self, argindex=1):
        from sympy.functions.special.gamma_functions import (gamma, polygamma)
        if argindex == 1:
            return gamma(self.args[0] + 1)*polygamma(0, self.args[0] + 1)
        else:
            raise ArgumentIndexError(self, argindex)

    _small_swing = [
        1, 1, 1, 3, 3, 15, 5, 35, 35, 315, 63, 693, 231, 3003, 429, 6435, 6435, 109395,
        12155, 230945, 46189, 969969, 88179, 2028117, 676039, 16900975, 1300075,
        35102025, 5014575, 145422675, 9694845, 300540195, 300540195
    ]

    _small_factorials: list[int] = []

    @classmethod
    def _swing(cls, n):
        if n < 33:
            return cls._small_swing[n]
        else:
            N, primes = int(_sqrt(n)), []

            for prime in sieve.primerange(3, N + 1):
                p, q = 1, n

                while True:
                    q //= prime

                    if q > 0:
                        if q & 1 == 1:
                            p *= prime
                    else:
                        break

                if p > 1:
                    primes.append(p)

            for prime in sieve.primerange(N + 1, n//3 + 1):
                if (n // prime) & 1 == 1:
                    primes.append(prime)

            L_product = prod(sieve.primerange(n//2 + 1, n + 1))
            R_product = prod(primes)

            return L_product*R_product

    @classmethod
    def _recursive(cls, n):
        if n < 2:
            return 1
        else:
            return (cls._recursive(n//2)**2)*cls._swing(n)

    @classmethod
    def eval(cls, n):
        n = sympify(n)

        if n.is_Number:
            if n.is_zero:
                return S.One
            elif n is S.Infinity:
                return S.Infinity
            elif n.is_Integer:
                if n.is_negative:
                    return S.ComplexInfinity
                else:
                    n = n.p

                    if n < 20:
                        if not cls._small_factorials:
                            result = 1
                            for i in range(1, 20):
                                result *= i
                                cls._small_factorials.append(result)
                        result = cls._small_factorials[n-1]

                    # GMPY factorial is faster, use it when available
                    #
                    # XXX: There is a sympy.external.gmpy.factorial function
                    # which provides gmpy.fac if available or the flint version
                    # if flint is used. It could be used here to avoid the
                    # conditional logic but it needs to be checked whether the
                    # pure Python fallback used there is as fast as the
                    # fallback used here (perhaps the fallback here should be
                    # moved to sympy.external.ntheory).
                    elif _gmpy is not None:
                        result = _gmpy.fac(n)

                    else:
                        bits = bin(n).count('1')
                        result = cls._recursive(n)*2**(n - bits)

                    return Integer(result)

    def _facmod(self, n, q):
        res, N = 1, int(_sqrt(n))

        # Exponent of prime p in n! is e_p(n) = [n/p] + [n/p**2] + ...
        # for p > sqrt(n), e_p(n) < sqrt(n), the primes with [n/p] = m,
        # occur consecutively and are grouped together in pw[m] for
        # simultaneous exponentiation at a later stage
        pw = [1]*N

        m = 2 # to initialize the if condition below
        for prime in sieve.primerange(2, n + 1):
            if m > 1:
                m, y = 0, n // prime
                while y:
                    m += y
                    y //= prime
            if m < N:
                pw[m] = pw[m]*prime % q
            else:
                res = res*pow(prime, m, q) % q

        for ex, bs in enumerate(pw):
            if ex == 0 or bs == 1:
                continue
            if bs == 0:
                return 0
            res = res*pow(bs, ex, q) % q

        return res

    def _eval_Mod(self, q):
        n = self.args[0]
        if n.is_integer and n.is_nonnegative and q.is_integer:
            aq = abs(q)
            d = aq - n
            if d.is_nonpositive:
                return S.Zero
            else:
                isprime = aq.is_prime
                if d == 1:
                    # Apply Wilson's theorem (if a natural number n > 1
                    # is a prime number, then (n-1)! = -1 mod n) and
                    # its inverse (if n > 4 is a composite number, then
                    # (n-1)! = 0 mod n)
                    if isprime:
                        return -1 % q
                    elif isprime is False and (aq - 6).is_nonnegative:
                        return S.Zero
                elif n.is_Integer and q.is_Integer:
                    n, d, aq = map(int, (n, d, aq))
                    if isprime and (d - 1 < n):
                        fc = self._facmod(d - 1, aq)
                        fc = pow(fc, aq - 2, aq)
                        if d%2:
                            fc = -fc
                    else:
                        fc = self._facmod(n, aq)

                    return fc % q

    def _eval_rewrite_as_gamma(self, n, piecewise=True, **kwargs):
        from sympy.functions.special.gamma_functions import gamma
        return gamma(n + 1)

    def _eval_rewrite_as_Product(self, n, **kwargs):
        from sympy.concrete.products import Product
        if n.is_nonnegative and n.is_integer:
            i = Dummy('i', integer=True)
            return Product(i, (i, 1, n))

    def _eval_is_integer(self):
        if self.args[0].is_integer and self.args[0].is_nonnegative:
            return True

    def _eval_is_positive(self):
        if self.args[0].is_integer and self.args[0].is_nonnegative:
            return True

    def _eval_is_even(self):
        x = self.args[0]
        if x.is_integer and x.is_nonnegative:
            return (x - 2).is_nonnegative

    def _eval_is_composite(self):
        x = self.args[0]
        if x.is_integer and x.is_nonnegative:
            return (x - 3).is_nonnegative

    def _eval_is_real(self):
        x = self.args[0]
        if x.is_nonnegative or x.is_noninteger:
            return True

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0].as_leading_term(x)
        arg0 = arg.subs(x, 0)
        if arg0.is_zero:
            return S.One
        elif not arg0.is_infinite:
            return self.func(arg)
        raise PoleError("Cannot expand %s around 0" % (self))

class MultiFactorial(CombinatorialFunction):
    pass


class subfactorial(CombinatorialFunction):
    r"""The subfactorial counts the derangements of $n$ items and is
    defined for non-negative integers as:

    .. math:: !n = \begin{cases} 1 & n = 0 \\ 0 & n = 1 \\
                    (n-1)(!(n-1) + !(n-2)) & n > 1 \end{cases}

    It can also be written as ``int(round(n!/exp(1)))`` but the
    recursive definition with caching is implemented for this function.

    An interesting analytic expression is the following [2]_

    .. math:: !x = \Gamma(x + 1, -1)/e

    which is valid for non-negative integers `x`. The above formula
    is not very useful in case of non-integers. `\Gamma(x + 1, -1)` is
    single-valued only for integral arguments `x`, elsewhere on the positive
    real axis it has an infinite number of branches none of which are real.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Subfactorial
    .. [2] https://mathworld.wolfram.com/Subfactorial.html

    Examples
    ========

    >>> from sympy import subfactorial
    >>> from sympy.abc import n
    >>> subfactorial(n + 1)
    subfactorial(n + 1)
    >>> subfactorial(5)
    44

    See Also
    ========

    factorial, uppergamma,
    sympy.utilities.iterables.generate_derangements
    """

    @classmethod
    @cacheit
    def _eval(self, n):
        if not n:
            return S.One
        elif n == 1:
            return S.Zero
        else:
            z1, z2 = 1, 0
            for i in range(2, n + 1):
                z1, z2 = z2, (i - 1)*(z2 + z1)
            return z2

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg.is_Integer and arg.is_nonnegative:
                return cls._eval(arg)
            elif arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity

    def _eval_is_even(self):
        if self.args[0].is_odd and self.args[0].is_nonnegative:
            return True

    def _eval_is_integer(self):
        if self.args[0].is_integer and self.args[0].is_nonnegative:
            return True

    def _eval_rewrite_as_factorial(self, arg, **kwargs):
        from sympy.concrete.summations import summation
        i = Dummy('i')
        f = S.NegativeOne**i / factorial(i)
        return factorial(arg) * summation(f, (i, 0, arg))

    def _eval_rewrite_as_gamma(self, arg, piecewise=True, **kwargs):
        from sympy.functions.elementary.exponential import exp
        from sympy.functions.special.gamma_functions import (gamma, lowergamma)
        return (S.NegativeOne**(arg + 1)*exp(-I*pi*arg)*lowergamma(arg + 1, -1)
                + gamma(arg + 1))*exp(-1)

    def _eval_rewrite_as_uppergamma(self, arg, **kwargs):
        from sympy.functions.special.gamma_functions import uppergamma
        return uppergamma(arg + 1, -1)/S.Exp1

    def _eval_is_nonnegative(self):
        if self.args[0].is_integer and self.args[0].is_nonnegative:
            return True

    def _eval_is_odd(self):
        if self.args[0].is_even and self.args[0].is_nonnegative:
            return True


class factorial2(CombinatorialFunction):
    r"""The double factorial `n!!`, not to be confused with `(n!)!`

    The double factorial is defined for nonnegative integers and for odd
    negative integers as:

    .. math:: n!! = \begin{cases} 1 & n = 0 \\
                    n(n-2)(n-4) \cdots 1 & n\ \text{positive odd} \\
                    n(n-2)(n-4) \cdots 2 & n\ \text{positive even} \\
                    (n+2)!!/(n+2) & n\ \text{negative odd} \end{cases}

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Double_factorial

    Examples
    ========

    >>> from sympy import factorial2, var
    >>> n = var('n')
    >>> n
    n
    >>> factorial2(n + 1)
    factorial2(n + 1)
    >>> factorial2(5)
    15
    >>> factorial2(-1)
    1
    >>> factorial2(-5)
    1/3

    See Also
    ========

    factorial, RisingFactorial, FallingFactorial
    """

    @classmethod
    def eval(cls, arg):
        # TODO: extend this to complex numbers?

        if arg.is_Number:
            if not arg.is_Integer:
                raise ValueError("argument must be nonnegative integer "
                                    "or negative odd integer")

            # This implementation is faster than the recursive one
            # It also avoids "maximum recursion depth exceeded" runtime error
            if arg.is_nonnegative:
                if arg.is_even:
                    k = arg / 2
                    return 2**k * factorial(k)
                return factorial(arg) / factorial2(arg - 1)


            if arg.is_odd:
                return arg*(S.NegativeOne)**((1 - arg)/2) / factorial2(-arg)
            raise ValueError("argument must be nonnegative integer "
                                "or negative odd integer")


    def _eval_is_even(self):
        # Double factorial is even for every positive even input
        n = self.args[0]
        if n.is_integer:
            if n.is_odd:
                return False
            if n.is_even:
                if n.is_positive:
                    return True
                if n.is_zero:
                    return False

    def _eval_is_integer(self):
        # Double factorial is an integer for every nonnegative input, and for
        # -1 and -3
        n = self.args[0]
        if n.is_integer:
            if (n + 1).is_nonnegative:
                return True
            if n.is_odd:
                return (n + 3).is_nonnegative

    def _eval_is_odd(self):
        # Double factorial is odd for every odd input not smaller than -3, and
        # for 0
        n = self.args[0]
        if n.is_odd:
            return (n + 3).is_nonnegative
        if n.is_even:
            if n.is_positive:
                return False
            if n.is_zero:
                return True

    def _eval_is_positive(self):
        # Double factorial is positive for every nonnegative input, and for
        # every odd negative input which is of the form -1-4k for an
        # nonnegative integer k
        n = self.args[0]
        if n.is_integer:
            if (n + 1).is_nonnegative:
                return True
            if n.is_odd:
                return ((n + 1) / 2).is_even

    def _eval_rewrite_as_gamma(self, n, piecewise=True, **kwargs):
        from sympy.functions.elementary.miscellaneous import sqrt
        from sympy.functions.elementary.piecewise import Piecewise
        from sympy.functions.special.gamma_functions import gamma
        return 2**(n/2)*gamma(n/2 + 1) * Piecewise((1, Eq(Mod(n, 2), 0)),
                (sqrt(2/pi), Eq(Mod(n, 2), 1)))


###############################################################################
######################## RISING and FALLING FACTORIALS ########################
###############################################################################


class RisingFactorial(CombinatorialFunction):
    r"""
    Rising factorial (also called Pochhammer symbol [1]_) is a double valued
    function arising in concrete mathematics, hypergeometric functions
    and series expansions. It is defined by:

    .. math:: \texttt{rf(y, k)} = (x)^k = x \cdot (x+1) \cdots (x+k-1)

    where `x` can be arbitrary expression and `k` is an integer. For
    more information check "Concrete mathematics" by Graham, pp. 66
    or visit https://mathworld.wolfram.com/RisingFactorial.html page.

    When `x` is a `~.Poly` instance of degree $\ge 1$ with a single variable,
    `(x)^k = x(y) \cdot x(y+1) \cdots x(y+k-1)`, where `y` is the
    variable of `x`. This is as described in [2]_.

    Examples
    ========

    >>> from sympy import rf, Poly
    >>> from sympy.abc import x
    >>> rf(x, 0)
    1
    >>> rf(1, 5)
    120
    >>> rf(x, 5) == x*(1 + x)*(2 + x)*(3 + x)*(4 + x)
    True
    >>> rf(Poly(x**3, x), 2)
    Poly(x**6 + 3*x**5 + 3*x**4 + x**3, x, domain='ZZ')

    Rewriting is complicated unless the relationship between
    the arguments is known, but rising factorial can
    be rewritten in terms of gamma, factorial, binomial,
    and falling factorial.

    >>> from sympy import Symbol, factorial, ff, binomial, gamma
    >>> n = Symbol('n', integer=True, positive=True)
    >>> R = rf(n, n + 2)
    >>> for i in (rf, ff, factorial, binomial, gamma):
    ...  R.rewrite(i)
    ...
    RisingFactorial(n, n + 2)
    FallingFactorial(2*n + 1, n + 2)
    factorial(2*n + 1)/factorial(n - 1)
    binomial(2*n + 1, n + 2)*factorial(n + 2)
    gamma(2*n + 2)/gamma(n)

    See Also
    ========

    factorial, factorial2, FallingFactorial

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Pochhammer_symbol
    .. [2] Peter Paule, "Greatest Factorial Factorization and Symbolic
           Summation", Journal of Symbolic Computation, vol. 20, pp. 235-268,
           1995.

    """

    @classmethod
    def eval(cls, x, k):
        x = sympify(x)
        k = sympify(k)

        if x is S.NaN or k is S.NaN:
            return S.NaN
        elif x is S.One:
            return factorial(k)
        elif k.is_Integer:
            if k.is_zero:
                return S.One
            else:
                if k.is_positive:
                    if x is S.Infinity:
                        return S.Infinity
                    elif x is S.NegativeInfinity:
                        if k.is_odd:
                            return S.NegativeInfinity
                        else:
                            return S.Infinity
                    else:
                        if isinstance(x, Poly):
                            gens = x.gens
                            if len(gens)!= 1:
                                raise ValueError("rf only defined for "
                                            "polynomials on one generator")
                            else:
                                return reduce(lambda r, i:
                                              r*(x.shift(i)),
                                              range(int(k)), 1)
                        else:
                            return reduce(lambda r, i: r*(x + i),
                                          range(int(k)), 1)

                else:
                    if x is S.Infinity:
                        return S.Infinity
                    elif x is S.NegativeInfinity:
                        return S.Infinity
                    else:
                        if isinstance(x, Poly):
                            gens = x.gens
                            if len(gens)!= 1:
                                raise ValueError("rf only defined for "
                                            "polynomials on one generator")
                            else:
                                return 1/reduce(lambda r, i:
                                                r*(x.shift(-i)),
                                                range(1, abs(int(k)) + 1), 1)
                        else:
                            return 1/reduce(lambda r, i:
                                            r*(x - i),
                                            range(1, abs(int(k)) + 1), 1)

        if k.is_integer == False:
            if x.is_integer and x.is_negative:
                return S.Zero

    def _eval_rewrite_as_gamma(self, x, k, piecewise=True, **kwargs):
        from sympy.functions.elementary.piecewise import Piecewise
        from sympy.functions.special.gamma_functions import gamma
        if not piecewise:
            if (x <= 0) == True:
                return S.NegativeOne**k*gamma(1 - x) / gamma(-k - x + 1)
            return gamma(x + k) / gamma(x)
        return Piecewise(
            (gamma(x + k) / gamma(x), x > 0),
            (S.NegativeOne**k*gamma(1 - x) / gamma(-k - x + 1), True))

    def _eval_rewrite_as_FallingFactorial(self, x, k, **kwargs):
        return FallingFactorial(x + k - 1, k)

    def _eval_rewrite_as_factorial(self, x, k, **kwargs):
        from sympy.functions.elementary.piecewise import Piecewise
        if x.is_integer and k.is_integer:
            return Piecewise(
                (factorial(k + x - 1)/factorial(x - 1), x > 0),
                (S.NegativeOne**k*factorial(-x)/factorial(-k - x), True))

    def _eval_rewrite_as_binomial(self, x, k, **kwargs):
        if k.is_integer:
            return factorial(k) * binomial(x + k - 1, k)

    def _eval_rewrite_as_tractable(self, x, k, limitvar=None, **kwargs):
        from sympy.functions.special.gamma_functions import gamma
        if limitvar:
            k_lim = k.subs(limitvar, S.Infinity)
            if k_lim is S.Infinity:
                return (gamma(x + k).rewrite('tractable', deep=True) / gamma(x))
            elif k_lim is S.NegativeInfinity:
                return (S.NegativeOne**k*gamma(1 - x) / gamma(-k - x + 1).rewrite('tractable', deep=True))
        return self.rewrite(gamma).rewrite('tractable', deep=True)

    def _eval_is_integer(self):
        return fuzzy_and((self.args[0].is_integer, self.args[1].is_integer,
                          self.args[1].is_nonnegative))


class FallingFactorial(CombinatorialFunction):
    r"""
    Falling factorial (related to rising factorial) is a double valued
    function arising in concrete mathematics, hypergeometric functions
    and series expansions. It is defined by

    .. math:: \texttt{ff(x, k)} = (x)_k = x \cdot (x-1) \cdots (x-k+1)

    where `x` can be arbitrary expression and `k` is an integer. For
    more information check "Concrete mathematics" by Graham, pp. 66
    or [1]_.

    When `x` is a `~.Poly` instance of degree $\ge 1$ with single variable,
    `(x)_k = x(y) \cdot x(y-1) \cdots x(y-k+1)`, where `y` is the
    variable of `x`. This is as described in

    >>> from sympy import ff, Poly, Symbol
    >>> from sympy.abc import x
    >>> n = Symbol('n', integer=True)

    >>> ff(x, 0)
    1
    >>> ff(5, 5)
    120
    >>> ff(x, 5) == x*(x - 1)*(x - 2)*(x - 3)*(x - 4)
    True
    >>> ff(Poly(x**2, x), 2)
    Poly(x**4 - 2*x**3 + x**2, x, domain='ZZ')
    >>> ff(n, n)
    factorial(n)

    Rewriting is complicated unless the relationship between
    the arguments is known, but falling factorial can
    be rewritten in terms of gamma, factorial and binomial
    and rising factorial.

    >>> from sympy import factorial, rf, gamma, binomial, Symbol
    >>> n = Symbol('n', integer=True, positive=True)
    >>> F = ff(n, n - 2)
    >>> for i in (rf, ff, factorial, binomial, gamma):
    ...  F.rewrite(i)
    ...
    RisingFactorial(3, n - 2)
    FallingFactorial(n, n - 2)
    factorial(n)/2
    binomial(n, n - 2)*factorial(n - 2)
    gamma(n + 1)/2

    See Also
    ========

    factorial, factorial2, RisingFactorial

    References
    ==========

    .. [1] https://mathworld.wolfram.com/FallingFactorial.html
    .. [2] Peter Paule, "Greatest Factorial Factorization and Symbolic
           Summation", Journal of Symbolic Computation, vol. 20, pp. 235-268,
           1995.

    """

    @classmethod
    def eval(cls, x, k):
        x = sympify(x)
        k = sympify(k)

        if x is S.NaN or k is S.NaN:
            return S.NaN
        elif k.is_integer and x == k:
            return factorial(x)
        elif k.is_Integer:
            if k.is_zero:
                return S.One
            else:
                if k.is_positive:
                    if x is S.Infinity:
                        return S.Infinity
                    elif x is S.NegativeInfinity:
                        if k.is_odd:
                            return S.NegativeInfinity
                        else:
                            return S.Infinity
                    else:
                        if isinstance(x, Poly):
                            gens = x.gens
                            if len(gens)!= 1:
                                raise ValueError("ff only defined for "
                                            "polynomials on one generator")
                            else:
                                return reduce(lambda r, i:
                                              r*(x.shift(-i)),
                                              range(int(k)), 1)
                        else:
                            return reduce(lambda r, i: r*(x - i),
                                          range(int(k)), 1)
                else:
                    if x is S.Infinity:
                        return S.Infinity
                    elif x is S.NegativeInfinity:
                        return S.Infinity
                    else:
                        if isinstance(x, Poly):
                            gens = x.gens
                            if len(gens)!= 1:
                                raise ValueError("rf only defined for "
                                            "polynomials on one generator")
                            else:
                                return 1/reduce(lambda r, i:
                                                r*(x.shift(i)),
                                                range(1, abs(int(k)) + 1), 1)
                        else:
                            return 1/reduce(lambda r, i: r*(x + i),
                                            range(1, abs(int(k)) + 1), 1)

    def _eval_rewrite_as_gamma(self, x, k, piecewise=True, **kwargs):
        from sympy.functions.elementary.piecewise import Piecewise
        from sympy.functions.special.gamma_functions import gamma
        if not piecewise:
            if (x < 0) == True:
                return S.NegativeOne**k*gamma(k - x) / gamma(-x)
            return gamma(x + 1) / gamma(x - k + 1)
        return Piecewise(
            (gamma(x + 1) / gamma(x - k + 1), x >= 0),
            (S.NegativeOne**k*gamma(k - x) / gamma(-x), True))

    def _eval_rewrite_as_RisingFactorial(self, x, k, **kwargs):
        return rf(x - k + 1, k)

    def _eval_rewrite_as_binomial(self, x, k, **kwargs):
        if k.is_integer:
            return factorial(k) * binomial(x, k)

    def _eval_rewrite_as_factorial(self, x, k, **kwargs):
        from sympy.functions.elementary.piecewise import Piecewise
        if x.is_integer and k.is_integer:
            return Piecewise(
                (factorial(x)/factorial(-k + x), x >= 0),
                (S.NegativeOne**k*factorial(k - x - 1)/factorial(-x - 1), True))

    def _eval_rewrite_as_tractable(self, x, k, limitvar=None, **kwargs):
        from sympy.functions.special.gamma_functions import gamma
        if limitvar:
            k_lim = k.subs(limitvar, S.Infinity)
            if k_lim is S.Infinity:
                return (S.NegativeOne**k*gamma(k - x).rewrite('tractable', deep=True) / gamma(-x))
            elif k_lim is S.NegativeInfinity:
                return (gamma(x + 1) / gamma(x - k + 1).rewrite('tractable', deep=True))
        return self.rewrite(gamma).rewrite('tractable', deep=True)

    def _eval_is_integer(self):
        return fuzzy_and((self.args[0].is_integer, self.args[1].is_integer,
                          self.args[1].is_nonnegative))


rf = RisingFactorial
ff = FallingFactorial

###############################################################################
########################### BINOMIAL COEFFICIENTS #############################
###############################################################################


class binomial(CombinatorialFunction):
    r"""Implementation of the binomial coefficient. It can be defined
    in two ways depending on its desired interpretation:

    .. math:: \binom{n}{k} = \frac{n!}{k!(n-k)!}\ \text{or}\
                \binom{n}{k} = \frac{(n)_k}{k!}

    First, in a strict combinatorial sense it defines the
    number of ways we can choose `k` elements from a set of
    `n` elements. In this case both arguments are nonnegative
    integers and binomial is computed using an efficient
    algorithm based on prime factorization.

    The other definition is generalization for arbitrary `n`,
    however `k` must also be nonnegative. This case is very
    useful when evaluating summations.

    For the sake of convenience, for negative integer `k` this function
    will return zero no matter the other argument.

    To expand the binomial when `n` is a symbol, use either
    ``expand_func()`` or ``expand(func=True)``. The former will keep
    the polynomial in factored form while the latter will expand the
    polynomial itself. See examples for details.

    Examples
    ========

    >>> from sympy import Symbol, Rational, binomial, expand_func
    >>> n = Symbol('n', integer=True, positive=True)

    >>> binomial(15, 8)
    6435

    >>> binomial(n, -1)
    0

    Rows of Pascal's triangle can be generated with the binomial function:

    >>> for N in range(8):
    ...     print([binomial(N, i) for i in range(N + 1)])
    ...
    [1]
    [1, 1]
    [1, 2, 1]
    [1, 3, 3, 1]
    [1, 4, 6, 4, 1]
    [1, 5, 10, 10, 5, 1]
    [1, 6, 15, 20, 15, 6, 1]
    [1, 7, 21, 35, 35, 21, 7, 1]

    As can a given diagonal, e.g. the 4th diagonal:

    >>> N = -4
    >>> [binomial(N, i) for i in range(1 - N)]
    [1, -4, 10, -20, 35]

    >>> binomial(Rational(5, 4), 3)
    -5/128
    >>> binomial(Rational(-5, 4), 3)
    -195/128

    >>> binomial(n, 3)
    binomial(n, 3)

    >>> binomial(n, 3).expand(func=True)
    n**3/6 - n**2/2 + n/3

    >>> expand_func(binomial(n, 3))
    n*(n - 2)*(n - 1)/6

    In many cases, we can also compute binomial coefficients modulo a
    prime p quickly using Lucas' Theorem [2]_, though we need to include
    `evaluate=False` to postpone evaluation:

    >>> from sympy import Mod
    >>> Mod(binomial(156675, 4433, evaluate=False), 10**5 + 3)
    28625

    Using a generalisation of Lucas's Theorem given by Granville [3]_,
    we can extend this to arbitrary n:

    >>> Mod(binomial(10**18, 10**12, evaluate=False), (10**5 + 3)**2)
    3744312326

    References
    ==========

    .. [1] https://www.johndcook.com/blog/binomial_coefficients/
    .. [2] https://en.wikipedia.org/wiki/Lucas%27s_theorem
    .. [3] Binomial coefficients modulo prime powers, Andrew Granville,
        Available: https://web.archive.org/web/20170202003812/http://www.dms.umontreal.ca/~andrew/PDF/BinCoeff.pdf
    """

    def fdiff(self, argindex=1):
        from sympy.functions.special.gamma_functions import polygamma
        if argindex == 1:
            # https://functions.wolfram.com/GammaBetaErf/Binomial/20/01/01/
            n, k = self.args
            return binomial(n, k)*(polygamma(0, n + 1) - \
                polygamma(0, n - k + 1))
        elif argindex == 2:
            # https://functions.wolfram.com/GammaBetaErf/Binomial/20/01/02/
            n, k = self.args
            return binomial(n, k)*(polygamma(0, n - k + 1) - \
                polygamma(0, k + 1))
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def _eval(self, n, k):
        # n.is_Number and k.is_Integer and k != 1 and n != k

        if k.is_Integer:
            if n.is_Integer and n >= 0:
                n, k = int(n), int(k)

                if k > n:
                    return S.Zero
                elif k > n // 2:
                    k = n - k

                # XXX: This conditional logic should be moved to
                # sympy.external.gmpy and the pure Python version of bincoef
                # should be moved to sympy.external.ntheory.
                if _gmpy is not None:
                    return Integer(_gmpy.bincoef(n, k))

                d, result = n - k, 1
                for i in range(1, k + 1):
                    d += 1
                    result = result * d // i
                return Integer(result)
            else:
                d, result = n - k, 1
                for i in range(1, k + 1):
                    d += 1
                    result *= d
                return result / _factorial(k)

    @classmethod
    def eval(cls, n, k):
        n, k = map(sympify, (n, k))
        d = n - k
        n_nonneg, n_isint = n.is_nonnegative, n.is_integer
        if k.is_zero or ((n_nonneg or n_isint is False)
                and d.is_zero):
            return S.One
        if (k - 1).is_zero or ((n_nonneg or n_isint is False)
                and (d - 1).is_zero):
            return n
        if k.is_integer:
            if k.is_negative or (n_nonneg and n_isint and d.is_negative):
                return S.Zero
            elif n.is_number:
                res = cls._eval(n, k)
                return res.expand(basic=True) if res else res
        elif n_nonneg is False and n_isint:
            # a special case when binomial evaluates to complex infinity
            return S.ComplexInfinity
        elif k.is_number:
            from sympy.functions.special.gamma_functions import gamma
            return gamma(n + 1)/(gamma(k + 1)*gamma(n - k + 1))

    def _eval_Mod(self, q):
        n, k = self.args

        if any(x.is_integer is False for x in (n, k, q)):
            raise ValueError("Integers expected for binomial Mod")

        if all(x.is_Integer for x in (n, k, q)):
            n, k = map(int, (n, k))
            aq, res = abs(q), 1

            # handle negative integers k or n
            if k < 0:
                return S.Zero
            if n < 0:
                n = -n + k - 1
                res = -1 if k%2 else 1

            # non negative integers k and n
            if k > n:
                return S.Zero

            isprime = aq.is_prime
            aq = int(aq)
            if isprime:
                if aq < n:
                    # use Lucas Theorem
                    N, K = n, k
                    while N or K:
                        res = res*binomial(N % aq, K % aq) % aq
                        N, K = N // aq, K // aq

                else:
                    # use Factorial Modulo
                    d = n - k
                    if k > d:
                        k, d = d, k
                    kf = 1
                    for i in range(2, k + 1):
                        kf = kf*i % aq
                    df = kf
                    for i in range(k + 1, d + 1):
                        df = df*i % aq
                    res *= df
                    for i in range(d + 1, n + 1):
                        res = res*i % aq

                    res *= pow(kf*df % aq, aq - 2, aq)
                    res %= aq

            elif _sqrt(q) < k and q != 1:
                res = binomial_mod(n, k, q)

            else:
                # Binomial Factorization is performed by calculating the
                # exponents of primes <= n in `n! /(k! (n - k)!)`,
                # for non-negative integers n and k. As the exponent of
                # prime in n! is e_p(n) = [n/p] + [n/p**2] + ...
                # the exponent of prime in binomial(n, k) would be
                # e_p(n) - e_p(k) - e_p(n - k)
                M = int(_sqrt(n))
                for prime in sieve.primerange(2, n + 1):
                    if prime > n - k:
                        res = res*prime % aq
                    elif prime > n // 2:
                        continue
                    elif prime > M:
                        if n % prime < k % prime:
                            res = res*prime % aq
                    else:
                        N, K = n, k
                        exp = a = 0

                        while N > 0:
                            a = int((N % prime) < (K % prime + a))
                            N, K = N // prime, K // prime
                            exp += a

                        if exp > 0:
                            res *= pow(prime, exp, aq)
                            res %= aq

            return S(res % q)

    def _eval_expand_func(self, **hints):
        """
        Function to expand binomial(n, k) when m is positive integer
        Also,
        n is self.args[0] and k is self.args[1] while using binomial(n, k)
        """
        n = self.args[0]
        if n.is_Number:
            return binomial(*self.args)

        k = self.args[1]
        if (n-k).is_Integer:
            k = n - k

        if k.is_Integer:
            if k.is_zero:
                return S.One
            elif k.is_negative:
                return S.Zero
            else:
                n, result = self.args[0], 1
                for i in range(1, k + 1):
                    result *= n - k + i
                return result / _factorial(k)
        else:
            return binomial(*self.args)

    def _eval_rewrite_as_factorial(self, n, k, **kwargs):
        return factorial(n)/(factorial(k)*factorial(n - k))

    def _eval_rewrite_as_gamma(self, n, k, piecewise=True, **kwargs):
        from sympy.functions.special.gamma_functions import gamma
        return gamma(n + 1)/(gamma(k + 1)*gamma(n - k + 1))

    def _eval_rewrite_as_tractable(self, n, k, limitvar=None, **kwargs):
        return self._eval_rewrite_as_gamma(n, k).rewrite('tractable')

    def _eval_rewrite_as_FallingFactorial(self, n, k, **kwargs):
        if k.is_integer:
            return ff(n, k) / factorial(k)

    def _eval_is_integer(self):
        n, k = self.args
        if n.is_integer and k.is_integer:
            return True
        elif k.is_integer is False:
            return False

    def _eval_is_nonnegative(self):
        n, k = self.args
        if n.is_integer and k.is_integer:
            if n.is_nonnegative or k.is_negative or k.is_even:
                return True
            elif k.is_even is False:
                return  False

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.functions.special.gamma_functions import gamma
        return self.rewrite(gamma)._eval_as_leading_term(x, logx=logx, cdir=cdir)
<|endoftext|>"""
This module implements some special functions that commonly appear in
combinatorial contexts (e.g. in power series); in particular,
sequences of rational numbers such as Bernoulli and Fibonacci numbers.

Factorials, binomial coefficients and related functions are located in
the separate 'factorials' module.
"""
from __future__ import annotations
from math import prod
from collections import defaultdict
from typing import Callable

from sympy.core import S, Symbol, Add, Dummy
from sympy.core.cache import cacheit
from sympy.core.containers import Dict
from sympy.core.expr import Expr
from sympy.core.function import ArgumentIndexError, DefinedFunction, expand_mul
from sympy.core.logic import fuzzy_not
from sympy.core.mul import Mul
from sympy.core.numbers import E, I, pi, oo, Rational, Integer
from sympy.core.relational import Eq, is_le, is_gt, is_lt
from sympy.external.gmpy import SYMPY_INTS, remove, lcm, legendre, jacobi, kronecker
from sympy.functions.combinatorial.factorials import (binomial,
    factorial, subfactorial)
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.piecewise import Piecewise
from sympy.ntheory.factor_ import (factorint, _divisor_sigma, is_carmichael,
                                   find_carmichael_numbers_in_range, find_first_n_carmichaels)
from sympy.ntheory.generate import _primepi
from sympy.ntheory.partitions_ import _partition, _partition_rec
from sympy.ntheory.primetest import isprime, is_square
from sympy.polys.appellseqs import bernoulli_poly, euler_poly, genocchi_poly
from sympy.polys.polytools import cancel
from sympy.utilities.enumerative import MultisetPartitionTraverser
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import multiset, multiset_derangements, iterable
from sympy.utilities.memoization import recurrence_memo
from sympy.utilities.misc import as_int

from mpmath import mp, workprec
from mpmath.libmp import ifib as _ifib


def _product(a, b):
    return prod(range(a, b + 1))


# Dummy symbol used for computing polynomial sequences
_sym = Symbol('x')


#----------------------------------------------------------------------------#
#                                                                            #
#                           Carmichael numbers                               #
#                                                                            #
#----------------------------------------------------------------------------#

class carmichael(DefinedFunction):
    r"""
    Carmichael Numbers:

    Certain cryptographic algorithms make use of big prime numbers.
    However, checking whether a big number is prime is not so easy.
    Randomized prime number checking tests exist that offer a high degree of
    confidence of accurate determination at low cost, such as the Fermat test.

    Let 'a' be a random number between $2$ and $n - 1$, where $n$ is the
    number whose primality we are testing. Then, $n$ is probably prime if it
    satisfies the modular arithmetic congruence relation:

    .. math :: a^{n-1} = 1 \pmod{n}

    (where mod refers to the modulo operation)

    If a number passes the Fermat test several times, then it is prime with a
    high probability.

    Unfortunately, certain composite numbers (non-primes) still pass the Fermat
    test with every number smaller than themselves.
    These numbers are called Carmichael numbers.

    A Carmichael number will pass a Fermat primality test to every base $b$
    relatively prime to the number, even though it is not actually prime.
    This makes tests based on Fermat's Little Theorem less effective than
    strong probable prime tests such as the Baillie-PSW primality test and
    the Miller-Rabin primality test.

    Examples
    ========

    >>> from sympy.ntheory.factor_ import find_first_n_carmichaels, find_carmichael_numbers_in_range
    >>> find_first_n_carmichaels(5)
    [561, 1105, 1729, 2465, 2821]
    >>> find_carmichael_numbers_in_range(0, 562)
    [561]
    >>> find_carmichael_numbers_in_range(0,1000)
    [561]
    >>> find_carmichael_numbers_in_range(0,2000)
    [561, 1105, 1729]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Carmichael_number
    .. [2] https://en.wikipedia.org/wiki/Fermat_primality_test
    .. [3] https://www.jstor.org/stable/23248683?seq=1#metadata_info_tab_contents
    """

    @staticmethod
    def is_perfect_square(n):
        sympy_deprecation_warning(
        """
is_perfect_square is just a wrapper around sympy.ntheory.primetest.is_square
so use that directly instead.
        """,
        deprecated_since_version="1.11",
        active_deprecations_target='deprecated-carmichael-static-methods',
        )
        return is_square(n)

    @staticmethod
    def divides(p, n):
        sympy_deprecation_warning(
        """
        divides can be replaced by directly testing n % p == 0.
        """,
        deprecated_since_version="1.11",
        active_deprecations_target='deprecated-carmichael-static-methods',
        )
        return n % p == 0

    @staticmethod
    def is_prime(n):
        sympy_deprecation_warning(
        """
is_prime is just a wrapper around sympy.ntheory.primetest.isprime so use that
directly instead.
        """,
        deprecated_since_version="1.11",
        active_deprecations_target='deprecated-carmichael-static-methods',
        )
        return isprime(n)

    @staticmethod
    def is_carmichael(n):
        sympy_deprecation_warning(
        """
is_carmichael is just a wrapper around sympy.ntheory.factor_.is_carmichael so use that
directly instead.
        """,
        deprecated_since_version="1.13",
        active_deprecations_target='deprecated-ntheory-symbolic-functions',
        )
        return is_carmichael(n)

    @staticmethod
    def find_carmichael_numbers_in_range(x, y):
        sympy_deprecation_warning(
        """
find_carmichael_numbers_in_range is just a wrapper around sympy.ntheory.factor_.find_carmichael_numbers_in_range so use that
directly instead.
        """,
        deprecated_since_version="1.13",
        active_deprecations_target='deprecated-ntheory-symbolic-functions',
        )
        return find_carmichael_numbers_in_range(x, y)

    @staticmethod
    def find_first_n_carmichaels(n):
        sympy_deprecation_warning(
        """
find_first_n_carmichaels is just a wrapper around sympy.ntheory.factor_.find_first_n_carmichaels so use that
directly instead.
        """,
        deprecated_since_version="1.13",
        active_deprecations_target='deprecated-ntheory-symbolic-functions',
        )
        return find_first_n_carmichaels(n)


#----------------------------------------------------------------------------#
#                                                                            #
#                           Fibonacci numbers                                #
#                                                                            #
#----------------------------------------------------------------------------#


class fibonacci(DefinedFunction):
    r"""
    Fibonacci numbers / Fibonacci polynomials

    The Fibonacci numbers are the integer sequence defined by the
    initial terms `F_0 = 0`, `F_1 = 1` and the two-term recurrence
    relation `F_n = F_{n-1} + F_{n-2}`.  This definition
    extended to arbitrary real and complex arguments using
    the formula

    .. math :: F_z = \frac{\phi^z - \cos(\pi z) \phi^{-z}}{\sqrt 5}

    The Fibonacci polynomials are defined by `F_1(x) = 1`,
    `F_2(x) = x`, and `F_n(x) = x*F_{n-1}(x) + F_{n-2}(x)` for `n > 2`.
    For all positive integers `n`, `F_n(1) = F_n`.

    * ``fibonacci(n)`` gives the `n^{th}` Fibonacci number, `F_n`
    * ``fibonacci(n, x)`` gives the `n^{th}` Fibonacci polynomial in `x`, `F_n(x)`

    Examples
    ========

    >>> from sympy import fibonacci, Symbol

    >>> [fibonacci(x) for x in range(11)]
    [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    >>> fibonacci(5, Symbol('t'))
    t**4 + 3*t**2 + 1

    See Also
    ========

    bell, bernoulli, catalan, euler, harmonic, lucas, genocchi, partition, tribonacci

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fibonacci_number
    .. [2] https://mathworld.wolfram.com/FibonacciNumber.html

    """

    @staticmethod
    def _fib(n):
        return _ifib(n)

    @staticmethod
    @recurrence_memo([None, S.One, _sym])
    def _fibpoly(n, prev):
        return (prev[-2] + _sym*prev[-1]).expand()

    @classmethod
    def eval(cls, n, sym=None):
        if n is S.Infinity:
            return S.Infinity

        if n.is_Integer:
            if sym is None:
                n = int(n)
                if n < 0:
                    return S.NegativeOne**(n + 1) * fibonacci(-n)
                else:
                    return Integer(cls._fib(n))
            else:
                if n < 1:
                    raise ValueError("Fibonacci polynomials are defined "
                       "only for positive integer indices.")
                return cls._fibpoly(n).subs(_sym, sym)

    def _eval_rewrite_as_tractable(self, n, **kwargs):
        from sympy.functions import sqrt, cos
        return (S.GoldenRatio**n - cos(S.Pi*n)/S.GoldenRatio**n)/sqrt(5)

    def _eval_rewrite_as_sqrt(self, n, **kwargs):
        from sympy.functions.elementary.miscellaneous import sqrt
        return 2**(-n)*sqrt(5)*((1 + sqrt(5))**n - (-sqrt(5) + 1)**n) / 5

    def _eval_rewrite_as_GoldenRatio(self,n, **kwargs):
        return (S.GoldenRatio**n - 1/(-S.GoldenRatio)**n)/(2*S.GoldenRatio-1)


#----------------------------------------------------------------------------#
#                                                                            #
#                               Lucas numbers                                #
#                                                                            #
#----------------------------------------------------------------------------#


class lucas(DefinedFunction):
    """
    Lucas numbers

    Lucas numbers satisfy a recurrence relation similar to that of
    the Fibonacci sequence, in which each term is the sum of the
    preceding two. They are generated by choosing the initial
    values `L_0 = 2` and `L_1 = 1`.

    * ``lucas(n)`` gives the `n^{th}` Lucas number

    Examples
    ========

    >>> from sympy import lucas

    >>> [lucas(x) for x in range(11)]
    [2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123]

    See Also
    ========

    bell, bernoulli, catalan, euler, fibonacci, harmonic, genocchi, partition, tribonacci

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lucas_number
    .. [2] https://mathworld.wolfram.com/LucasNumber.html

    """

    @classmethod
    def eval(cls, n):
        if n is S.Infinity:
            return S.Infinity

        if n.is_Integer:
            return fibonacci(n + 1) + fibonacci(n - 1)

    def _eval_rewrite_as_sqrt(self, n, **kwargs):
        from sympy.functions.elementary.miscellaneous import sqrt
        return 2**(-n)*((1 + sqrt(5))**n + (-sqrt(5) + 1)**n)


#----------------------------------------------------------------------------#
#                                                                            #
#                             Tribonacci numbers                             #
#                                                                            #
#----------------------------------------------------------------------------#


class tribonacci(DefinedFunction):
    r"""
    Tribonacci numbers / Tribonacci polynomials

    The Tribonacci numbers are the integer sequence defined by the
    initial terms `T_0 = 0`, `T_1 = 1`, `T_2 = 1` and the three-term
    recurrence relation `T_n = T_{n-1} + T_{n-2} + T_{n-3}`.

    The Tribonacci polynomials are defined by `T_0(x) = 0`, `T_1(x) = 1`,
    `T_2(x) = x^2`, and `T_n(x) = x^2 T_{n-1}(x) + x T_{n-2}(x) + T_{n-3}(x)`
    for `n > 2`.  For all positive integers `n`, `T_n(1) = T_n`.

    * ``tribonacci(n)`` gives the `n^{th}` Tribonacci number, `T_n`
    * ``tribonacci(n, x)`` gives the `n^{th}` Tribonacci polynomial in `x`, `T_n(x)`

    Examples
    ========

    >>> from sympy import tribonacci, Symbol

    >>> [tribonacci(x) for x in range(11)]
    [0, 1, 1, 2, 4, 7, 13, 24, 44, 81, 149]
    >>> tribonacci(5, Symbol('t'))
    t**8 + 3*t**5 + 3*t**2

    See Also
    ========

    bell, bernoulli, catalan, euler, fibonacci, harmonic, lucas, genocchi, partition

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Generalizations_of_Fibonacci_numbers#Tribonacci_numbers
    .. [2] https://mathworld.wolfram.com/TribonacciNumber.html
    .. [3] https://oeis.org/A000073

    """

    @staticmethod
    @recurrence_memo([S.Zero, S.One, S.One])
    def _trib(n, prev):
        return (prev[-3] + prev[-2] + prev[-1])

    @staticmethod
    @recurrence_memo([S.Zero, S.One, _sym**2])
    def _tribpoly(n, prev):
        return (prev[-3] + _sym*prev[-2] + _sym**2*prev[-1]).expand()

    @classmethod
    def eval(cls, n, sym=None):
        if n is S.Infinity:
            return S.Infinity

        if n.is_Integer:
            n = int(n)
            if n < 0:
                raise ValueError("Tribonacci polynomials are defined "
                       "only for non-negative integer indices.")
            if sym is None:
                return Integer(cls._trib(n))
            else:
                return cls._tribpoly(n).subs(_sym, sym)

    def _eval_rewrite_as_sqrt(self, n, **kwargs):
        from sympy.functions.elementary.miscellaneous import cbrt, sqrt
        w = (-1 + S.ImaginaryUnit * sqrt(3)) / 2
        a = (1 + cbrt(19 + 3*sqrt(33)) + cbrt(19 - 3*sqrt(33))) / 3
        b = (1 + w*cbrt(19 + 3*sqrt(33)) + w**2*cbrt(19 - 3*sqrt(33))) / 3
        c = (1 + w**2*cbrt(19 + 3*sqrt(33)) + w*cbrt(19 - 3*sqrt(33))) / 3
        Tn = (a**(n + 1)/((a - b)*(a - c))
            + b**(n + 1)/((b - a)*(b - c))
            + c**(n + 1)/((c - a)*(c - b)))
        return Tn

    def _eval_rewrite_as_TribonacciConstant(self, n, **kwargs):
        from sympy.functions.elementary.integers import floor
        from sympy.functions.elementary.miscellaneous import cbrt, sqrt
        b = cbrt(586 + 102*sqrt(33))
        Tn = 3 * b * S.TribonacciConstant**n / (b**2 - 2*b + 4)
        return floor(Tn + S.Half)


#----------------------------------------------------------------------------#
#                                                                            #
#                           Bernoulli numbers                                #
#                                                                            #
#----------------------------------------------------------------------------#


class bernoulli(DefinedFunction):
    r"""
    Bernoulli numbers / Bernoulli polynomials / Bernoulli function

    The Bernoulli numbers are a sequence of rational numbers
    defined by `B_0 = 1` and the recursive relation (`n > 0`):

    .. math :: n+1 = \sum_{k=0}^n \binom{n+1}{k} B_k

    They are also commonly defined by their exponential generating
    function, which is `\frac{x}{1 - e^{-x}}`. For odd indices > 1,
    the Bernoulli numbers are zero.

    The Bernoulli polynomials satisfy the analogous formula:

    .. math :: B_n(x) = \sum_{k=0}^n (-1)^k \binom{n}{k} B_k x^{n-k}

    Bernoulli numbers and Bernoulli polynomials are related as
    `B_n(1) = B_n`.

    The generalized Bernoulli function `\operatorname{B}(s, a)`
    is defined for any complex `s` and `a`, except where `a` is a
    nonpositive integer and `s` is not a nonnegative integer. It is
    an entire function of `s` for fixed `a`, related to the Hurwitz
    zeta function by

    .. math:: \operatorname{B}(s, a) = \begin{cases}
              -s \zeta(1-s, a) & s \ne 0 \\ 1 & s = 0 \end{cases}

    When `s` is a nonnegative integer this function reduces to the
    Bernoulli polynomials: `\operatorname{B}(n, x) = B_n(x)`. When
    `a` is omitted it is assumed to be 1, yielding the (ordinary)
    Bernoulli function which interpolates the Bernoulli numbers and is
    related to the Riemann zeta function.

    We compute Bernoulli numbers using Ramanujan's formula:

    .. math :: B_n = \frac{A(n) - S(n)}{\binom{n+3}{n}}

    where:

    .. math :: A(n) = \begin{cases} \frac{n+3}{3} &
        n \equiv 0\ \text{or}\ 2 \pmod{6} \\
        -\frac{n+3}{6} & n \equiv 4 \pmod{6} \end{cases}

    and:

    .. math :: S(n) = \sum_{k=1}^{[n/6]} \binom{n+3}{n-6k} B_{n-6k}

    This formula is similar to the sum given in the definition, but
    cuts `\frac{2}{3}` of the terms. For Bernoulli polynomials, we use
    Appell sequences.

    For `n` a nonnegative integer and `s`, `a`, `x` arbitrary complex numbers,

    * ``bernoulli(n)`` gives the nth Bernoulli number, `B_n`
    * ``bernoulli(s)`` gives the Bernoulli function `\operatorname{B}(s)`
    * ``bernoulli(n, x)`` gives the nth Bernoulli polynomial in `x`, `B_n(x)`
    * ``bernoulli(s, a)`` gives the generalized Bernoulli function
      `\operatorname{B}(s, a)`

    .. versionchanged:: 1.12
        ``bernoulli(1)`` gives `+\frac{1}{2}` instead of `-\frac{1}{2}`.
        This choice of value confers several theoretical advantages [5]_,
        including the extension to complex parameters described above
        which this function now implements. The previous behavior, defined
        only for nonnegative integers `n`, can be obtained with
        ``(-1)**n*bernoulli(n)``.

    Examples
    ========

    >>> from sympy import bernoulli
    >>> from sympy.abc import x
    >>> [bernoulli(n) for n in range(11)]
    [1, 1/2, 1/6, 0, -1/30, 0, 1/42, 0, -1/30, 0, 5/66]
    >>> bernoulli(1000001)
    0
    >>> bernoulli(3, x)
    x**3 - 3*x**2/2 + x/2

    See Also
    ========

    andre, bell, catalan, euler, fibonacci, harmonic, lucas, genocchi,
    partition, tribonacci, sympy.polys.appellseqs.bernoulli_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bernoulli_number
    .. [2] https://en.wikipedia.org/wiki/Bernoulli_polynomial
    .. [3] https://mathworld.wolfram.com/BernoulliNumber.html
    .. [4] https://mathworld.wolfram.com/BernoulliPolynomial.html
    .. [5] Peter Luschny, "The Bernoulli Manifesto",
           https://luschny.de/math/zeta/The-Bernoulli-Manifesto.html
    .. [6] Peter Luschny, "An introduction to the Bernoulli function",
           https://arxiv.org/abs/2009.06743

    """

    args: tuple[Integer]

    # Calculates B_n for positive even n
    @staticmethod
    def _calc_bernoulli(n):
        s = 0
        a = int(binomial(n + 3, n - 6))
        for j in range(1, n//6 + 1):
            s += a * bernoulli(n - 6*j)
            # Avoid computing each binomial coefficient from scratch
            a *= _product(n - 6 - 6*j + 1, n - 6*j)
            a //= _product(6*j + 4, 6*j + 9)
        if n % 6 == 4:
            s = -Rational(n + 3, 6) - s
        else:
            s = Rational(n + 3, 3) - s
        return s / binomial(n + 3, n)

    # We implement a specialized memoization scheme to handle each
    # case modulo 6 separately
    _cache = {0: S.One, 1: Rational(1, 2), 2: Rational(1, 6), 4: Rational(-1, 30)}
    _highest = {0: 0, 1: 1, 2: 2, 4: 4}

    @classmethod
    def eval(cls, n, x=None):
        if x is S.One:
            return cls(n)
        elif n.is_zero:
            return S.One
        elif n.is_integer is False or n.is_nonnegative is False:
            if x is not None and x.is_Integer and x.is_nonpositive:
                return S.NaN
            return
        # Bernoulli numbers
        elif x is None:
            if n is S.One:
                return S.Half
            elif n.is_odd and (n-1).is_positive:
                return S.Zero
            elif n.is_Number:
                n = int(n)
                # Use mpmath for enormous Bernoulli numbers
                if n > 500:
                    p, q = mp.bernfrac(n)
                    return Rational(int(p), int(q))
                case = n % 6
                highest_cached = cls._highest[case]
                if n <= highest_cached:
                    return cls._cache[n]
                # To avoid excessive recursion when, say, bernoulli(1000) is
                # requested, calculate and cache the entire sequence ... B_988,
                # B_994, B_1000 in increasing order
                for i in range(highest_cached + 6, n + 6, 6):
                    b = cls._calc_bernoulli(i)
                    cls._cache[i] = b
                    cls._highest[case] = i
                return b
        # Bernoulli polynomials
        elif n.is_Number:
            return bernoulli_poly(n, x)

    def _eval_rewrite_as_zeta(self, n, x=1, **kwargs):
        from sympy.functions.special.zeta_functions import zeta
        return Piecewise((1, Eq(n, 0)), (-n * zeta(1-n, x), True))

    def _eval_evalf(self, prec):
        if not all(x.is_number for x in self.args):
            return
        n = self.args[0]._to_mpmath(prec)
        x = (self.args[1] if len(self.args) > 1 else S.One)._to_mpmath(prec)
        with workprec(prec):
            if n == 0:
                res = mp.mpf(1)
            elif n == 1:
                res = x - mp.mpf(0.5)
            elif mp.isint(n) and n >= 0:
                res = mp.bernoulli(n) if x == 1 else mp.bernpoly(n, x)
            else:
                res = -n * mp.zeta(1-n, x)
        return Expr._from_mpmath(res, prec)


#----------------------------------------------------------------------------#
#                                                                            #
#                                Bell numbers                                #
#                                                                            #
#----------------------------------------------------------------------------#


class bell(DefinedFunction):
    r"""
    Bell numbers / Bell polynomials

    The Bell numbers satisfy `B_0 = 1` and

    .. math:: B_n = \sum_{k=0}^{n-1} \binom{n-1}{k} B_k.

    They are also given by:

    .. math:: B_n = \frac{1}{e} \sum_{k=0}^{\infty} \frac{k^n}{k!}.

    The Bell polynomials are given by `B_0(x) = 1` and

    .. math:: B_n(x) = x \sum_{k=1}^{n-1} \binom{n-1}{k-1} B_{k-1}(x).

    The second kind of Bell polynomials (are sometimes called "partial" Bell
    polynomials or incomplete Bell polynomials) are defined as

    .. math:: B_{n,k}(x_1, x_2,\dotsc x_{n-k+1}) =
            \sum_{j_1+j_2+j_2+\dotsb=k \atop j_1+2j_2+3j_2+\dotsb=n}
                \frac{n!}{j_1!j_2!\dotsb j_{n-k+1}!}
                \left(\frac{x_1}{1!} \right)^{j_1}
                \left(\frac{x_2}{2!} \right)^{j_2} \dotsb
                \left(\frac{x_{n-k+1}}{(n-k+1)!} \right) ^{j_{n-k+1}}.

    * ``bell(n)`` gives the `n^{th}` Bell number, `B_n`.
    * ``bell(n, x)`` gives the `n^{th}` Bell polynomial, `B_n(x)`.
    * ``bell(n, k, (x1, x2, ...))`` gives Bell polynomials of the second kind,
      `B_{n,k}(x_1, x_2, \dotsc, x_{n-k+1})`.

    Notes
    =====

    Not to be confused with Bernoulli numbers and Bernoulli polynomials,
    which use the same notation.

    Examples
    ========

    >>> from sympy import bell, Symbol, symbols

    >>> [bell(n) for n in range(11)]
    [1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975]
    >>> bell(30)
    846749014511809332450147
    >>> bell(4, Symbol('t'))
    t**4 + 6*t**3 + 7*t**2 + t
    >>> bell(6, 2, symbols('x:6')[1:])
    6*x1*x5 + 15*x2*x4 + 10*x3**2

    See Also
    ========

    bernoulli, catalan, euler, fibonacci, harmonic, lucas, genocchi, partition, tribonacci

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bell_number
    .. [2] https://mathworld.wolfram.com/BellNumber.html
    .. [3] https://mathworld.wolfram.com/BellPolynomial.html

    """

    @staticmethod
    @recurrence_memo([1, 1])
    def _bell(n, prev):
        s = 1
        a = 1
        for k in range(1, n):
            a = a * (n - k) // k
            s += a * prev[k]
        return s

    @staticmethod
    @recurrence_memo([S.One, _sym])
    def _bell_poly(n, prev):
        s = 1
        a = 1
        for k in range(2, n + 1):
            a = a * (n - k + 1) // (k - 1)
            s += a * prev[k - 1]
        return expand_mul(_sym * s)

    @staticmethod
    def _bell_incomplete_poly(n, k, symbols):
        r"""
        The second kind of Bell polynomials (incomplete Bell polynomials).

        Calculated by recurrence formula:

        .. math:: B_{n,k}(x_1, x_2, \dotsc, x_{n-k+1}) =
                \sum_{m=1}^{n-k+1}
                \x_m \binom{n-1}{m-1} B_{n-m,k-1}(x_1, x_2, \dotsc, x_{n-m-k})

        where
            `B_{0,0} = 1;`
            `B_{n,0} = 0; for n \ge 1`
            `B_{0,k} = 0; for k \ge 1`

        """
        if (n == 0) and (k == 0):
            return S.One
        elif (n == 0) or (k == 0):
            return S.Zero
        s = S.Zero
        a = S.One
        for m in range(1, n - k + 2):
            s += a * bell._bell_incomplete_poly(
                n - m, k - 1, symbols) * symbols[m - 1]
            a = a * (n - m) / m
        return expand_mul(s)

    @classmethod
    def eval(cls, n, k_sym=None, symbols=None):
        if n is S.Infinity:
            if k_sym is None:
                return S.Infinity
            else:
                raise ValueError("Bell polynomial is not defined")

        if n.is_negative or n.is_integer is False:
            raise ValueError("a non-negative integer expected")

        if n.is_Integer and n.is_nonnegative:
            if k_sym is None:
                return Integer(cls._bell(int(n)))
            elif symbols is None:
                return cls._bell_poly(int(n)).subs(_sym, k_sym)
            else:
                r = cls._bell_incomplete_poly(int(n), int(k_sym), symbols)
                return r

    def _eval_rewrite_as_Sum(self, n, k_sym=None, symbols=None, **kwargs):
        from sympy.concrete.summations import Sum
        if (k_sym is not None) or (symbols is not None):
            return self

        # Dobinski's formula
        if not n.is_nonnegative:
            return self
        k = Dummy('k', integer=True, nonnegative=True)
        return 1 / E * Sum(k**n / factorial(k), (k, 0, S.Infinity))


#----------------------------------------------------------------------------#
#                                                                            #
#                              Harmonic numbers                              #
#                                                                            #
#----------------------------------------------------------------------------#


class harmonic(DefinedFunction):
    r"""
    Harmonic numbers

    The nth harmonic number is given by `\operatorname{H}_{n} =
    1 + \frac{1}{2} + \frac{1}{3} + \ldots + \frac{1}{n}`.

    More generally:

    .. math:: \operatorname{H}_{n,m} = \sum_{k=1}^{n} \frac{1}{k^m}

    As `n \rightarrow \infty`, `\operatorname{H}_{n,m} \rightarrow \zeta(m)`,
    the Riemann zeta function.

    * ``harmonic(n)`` gives the nth harmonic number, `\operatorname{H}_n`

    * ``harmonic(n, m)`` gives the nth generalized harmonic number
      of order `m`, `\operatorname{H}_{n,m}`, where
      ``harmonic(n) == harmonic(n, 1)``

    This function can be extended to complex `n` and `m` where `n` is not a
    negative integer or `m` is a nonpositive integer as

    .. math:: \operatorname{H}_{n,m} = \begin{cases} \zeta(m) - \zeta(m, n+1)
            & m \ne 1 \\ \psi(n+1) + \gamma & m = 1 \end{cases}

    Examples
    ========

    >>> from sympy import harmonic, oo

    >>> [harmonic(n) for n in range(6)]
    [0, 1, 3/2, 11/6, 25/12, 137/60]
    >>> [harmonic(n, 2) for n in range(6)]
    [0, 1, 5/4, 49/36, 205/144, 5269/3600]
    >>> harmonic(oo, 2)
    pi**2/6

    >>> from sympy import Symbol, Sum
    >>> n = Symbol("n")

    >>> harmonic(n).rewrite(Sum)
    Sum(1/_k, (_k, 1, n))

    We can evaluate harmonic numbers for all integral and positive
    rational arguments:

    >>> from sympy import S, expand_func, simplify
    >>> harmonic(8)
    761/280
    >>> harmonic(11)
    83711/27720

    >>> H = harmonic(1/S(3))
    >>> H
    harmonic(1/3)
    >>> He = expand_func(H)
    >>> He
    -log(6) - sqrt(3)*pi/6 + 2*Sum(log(sin(_k*pi/3))*cos(2*_k*pi/3), (_k, 1, 1))
                           + 3*Sum(1/(3*_k + 1), (_k, 0, 0))
    >>> He.doit()
    -log(6) - sqrt(3)*pi/6 - log(sqrt(3)/2) + 3
    >>> H = harmonic(25/S(7))
    >>> He = simplify(expand_func(H).doit())
    >>> He
    log(sin(2*pi/7)**(2*cos(16*pi/7))/(14*sin(pi/7)**(2*cos(pi/7))*cos(pi/14)**(2*sin(pi/14)))) + pi*tan(pi/14)/2 + 30247/9900
    >>> He.n(40)
    1.983697455232980674869851942390639915940
    >>> harmonic(25/S(7)).n(40)
    1.983697455232980674869851942390639915940

    We can rewrite harmonic numbers in terms of polygamma functions:

    >>> from sympy import digamma, polygamma
    >>> m = Symbol("m", integer=True, positive=True)

    >>> harmonic(n).rewrite(digamma)
    polygamma(0, n + 1) + EulerGamma

    >>> harmonic(n).rewrite(polygamma)
    polygamma(0, n + 1) + EulerGamma

    >>> harmonic(n,3).rewrite(polygamma)
    polygamma(2, n + 1)/2 + zeta(3)

    >>> simplify(harmonic(n,m).rewrite(polygamma))
    Piecewise((polygamma(0, n + 1) + EulerGamma, Eq(m, 1)),
    (-(-1)**m*polygamma(m - 1, n + 1)/factorial(m - 1) + zeta(m), True))

    Integer offsets in the argument can be pulled out:

    >>> from sympy import expand_func

    >>> expand_func(harmonic(n+4))
    harmonic(n) + 1/(n + 4) + 1/(n + 3) + 1/(n + 2) + 1/(n + 1)

    >>> expand_func(harmonic(n-4))
    harmonic(n) - 1/(n - 1) - 1/(n - 2) - 1/(n - 3) - 1/n

    Some limits can be computed as well:

    >>> from sympy import limit, oo

    >>> limit(harmonic(n), n, oo)
    oo

    >>> limit(harmonic(n, 2), n, oo)
    pi**2/6

    >>> limit(harmonic(n, 3), n, oo)
    zeta(3)

    For `m > 1`, `H_{n,m}` tends to `\zeta(m)` in the limit of infinite `n`:

    >>> m = Symbol("m", positive=True)
    >>> limit(harmonic(n, m+1), n, oo)
    zeta(m + 1)

    See Also
    ========

    bell, bernoulli, catalan, euler, fibonacci, lucas, genocchi, partition, tribonacci

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Harmonic_number
    .. [2] https://functions.wolfram.com/GammaBetaErf/HarmonicNumber/
    .. [3] https://functions.wolfram.com/GammaBetaErf/HarmonicNumber2/

    """

    # This prevents redundant recalculations and speeds up harmonic number computations.
    harmonic_cache: dict[Integer, Callable[[int], Rational]] = {}

    @classmethod
    def eval(cls, n, m=None):
        from sympy.functions.special.zeta_functions import zeta
        if m is S.One:
            return cls(n)
        if m is None:
            m = S.One
        if n.is_zero:
            return S.Zero
        elif m.is_zero:
            return n
        elif n is S.Infinity:
            if m.is_negative:
                return S.NaN
            elif is_le(m, S.One):
                return S.Infinity
            elif is_gt(m, S.One):
                return zeta(m)
        elif m.is_Integer and m.is_nonpositive:
            return (bernoulli(1-m, n+1) - bernoulli(1-m)) / (1-m)
        elif n.is_Integer:
            if n.is_negative and (m.is_integer is False or m.is_nonpositive is False):
                return S.ComplexInfinity if m is S.One else S.NaN
            if n.is_nonnegative:
                if m.is_Integer:
                    if m not in cls.harmonic_cache:
                        @recurrence_memo([0])
                        def f(n, prev):
                            return prev[-1] + S.One / n**m
                        cls.harmonic_cache[m] = f
                    return cls.harmonic_cache[m](int(n))
                return Add(*(k**(-m) for k in range(1, int(n) + 1)))

    def _eval_rewrite_as_polygamma(self, n, m=S.One, **kwargs):
        from sympy.functions.special.gamma_functions import gamma, polygamma
        if m.is_integer and m.is_positive:
            return Piecewise((polygamma(0, n+1) + S.EulerGamma, Eq(m, 1)),
                    (S.NegativeOne**m * (polygamma(m-1, 1) - polygamma(m-1, n+1)) /
                    gamma(m), True))

    def _eval_rewrite_as_digamma(self, n, m=1, **kwargs):
        from sympy.functions.special.gamma_functions import polygamma
        return self.rewrite(polygamma)

    def _eval_rewrite_as_trigamma(self, n, m=1, **kwargs):
        from sympy.functions.special.gamma_functions import polygamma
        return self.rewrite(polygamma)

    def _eval_rewrite_as_Sum(self, n, m=None, **kwargs):
        from sympy.concrete.summations import Sum
        k = Dummy("k", integer=True)
        if m is None:
            m = S.One
        return Sum(k**(-m), (k, 1, n))

    def _eval_rewrite_as_zeta(self, n, m=S.One, **kwargs):
        from sympy.functions.special.zeta_functions import zeta
        from sympy.functions.special.gamma_functions import digamma
        return Piecewise((digamma(n + 1) + S.EulerGamma, Eq(m, 1)),
                         (zeta(m) - zeta(m, n+1), True))

    def _eval_expand_func(self, **hints):
        from sympy.concrete.summations import Sum
        n = self.args[0]
        m = self.args[1] if len(self.args) == 2 else 1

        if m == S.One:
            if n.is_Add:
                off = n.args[0]
                nnew = n - off
                if off.is_Integer and off.is_positive:
                    result = [S.One/(nnew + i) for i in range(off, 0, -1)] + [harmonic(nnew)]
                    return Add(*result)
                elif off.is_Integer and off.is_negative:
                    result = [-S.One/(nnew + i) for i in range(0, off, -1)] + [harmonic(nnew)]
                    return Add(*result)

            if n.is_Rational:
                # Expansions for harmonic numbers at general rational arguments (u + p/q)
                # Split n as u + p/q with p < q
                p, q = n.as_numer_denom()
                u = p // q
                p = p - u * q
                if u.is_nonnegative and p.is_positive and q.is_positive and p < q:
                    from sympy.functions.elementary.exponential import log
                    from sympy.functions.elementary.integers import floor
                    from sympy.functions.elementary.trigonometric import sin, cos, cot
                    k = Dummy("k")
                    t1 = q * Sum(1 / (q * k + p), (k, 0, u))
                    t2 = 2 * Sum(cos((2 * pi * p * k) / S(q)) *
                                   log(sin((pi * k) / S(q))),
                                   (k, 1, floor((q - 1) / S(2))))
                    t3 = (pi / 2) * cot((pi * p) / q) + log(2 * q)
                    return t1 + t2 - t3

        return self

    def _eval_rewrite_as_tractable(self, n, m=1, limitvar=None, **kwargs):
        from sympy.functions.special.zeta_functions import zeta
        from sympy.functions.special.gamma_functions import polygamma
        pg = self.rewrite(polygamma)
        if not isinstance(pg, harmonic):
            return pg.rewrite("tractable", deep=True)
        arg = m - S.One
        if arg.is_nonzero:
            return (zeta(m) - zeta(m, n+1)).rewrite("tractable", deep=True)

    def _eval_evalf(self, prec):
        if not all(x.is_number for x in self.args):
            return
        n = self.args[0]._to_mpmath(prec)
        m = (self.args[1] if len(self.args) > 1 else S.One)._to_mpmath(prec)
        if mp.isint(n) and n < 0:
            return S.NaN
        with workprec(prec):
            if m == 1:
                res = mp.harmonic(n)
            else:
                res = mp.zeta(m) - mp.zeta(m, n+1)
        return Expr._from_mpmath(res, prec)

    def fdiff(self, argindex=1):
        from sympy.functions.special.zeta_functions import zeta
        if len(self.args) == 2:
            n, m = self.args
        else:
            n, m = self.args + (1,)
        if argindex == 1:
            return m * zeta(m+1, n+1)
        else:
            raise ArgumentIndexError


#----------------------------------------------------------------------------#
#                                                                            #
#                           Euler numbers                                    #
#                                                                            #
#----------------------------------------------------------------------------#


class euler(DefinedFunction):
    r"""
    Euler numbers / Euler polynomials / Euler function

    The Euler numbers are given by:

    .. math:: E_{2n} = I \sum_{k=1}^{2n+1} \sum_{j=0}^k \binom{k}{j}
        \frac{(-1)^j (k-2j)^{2n+1}}{2^k I^k k}

    .. math:: E_{2n+1} = 0

    Euler numbers and Euler polynomials are related by

    .. math:: E_n = 2^n E_n\left(\frac{1}{2}\right).

    We compute symbolic Euler polynomials using Appell sequences,
    but numerical evaluation of the Euler polynomial is computed
    more efficiently (and more accurately) using the mpmath library.

    The Euler polynomials are special cases of the generalized Euler function,
    related to the Genocchi function as

    .. math:: \operatorname{E}(s, a) = -\frac{\operatorname{G}(s+1, a)}{s+1}

    with the limit of `\psi\left(\frac{a+1}{2}\right) - \psi\left(\frac{a}{2}\right)`
    being taken when `s = -1`. The (ordinary) Euler function interpolating
    the Euler numbers is then obtained as
    `\operatorname{E}(s) = 2^s \operatorname{E}\left(s, \frac{1}{2}\right)`.

    * ``euler(n)`` gives the nth Euler number `E_n`.
    * ``euler(s)`` gives the Euler function `\operatorname{E}(s)`.
    * ``euler(n, x)`` gives the nth Euler polynomial `E_n(x)`.
    * ``euler(s, a)`` gives the generalized Euler function `\operatorname{E}(s, a)`.

    Examples
    ========

    >>> from sympy import euler, Symbol, S
    >>> [euler(n) for n in range(10)]
    [1, 0, -1, 0, 5, 0, -61, 0, 1385, 0]
    >>> [2**n*euler(n,1) for n in range(10)]
    [1, 1, 0, -2, 0, 16, 0, -272, 0, 7936]
    >>> n = Symbol("n")
    >>> euler(n + 2*n)
    euler(3*n)

    >>> x = Symbol("x")
    >>> euler(n, x)
    euler(n, x)

    >>> euler(0, x)
    1
    >>> euler(1, x)
    x - 1/2
    >>> euler(2, x)
    x**2 - x
    >>> euler(3, x)
    x**3 - 3*x**2/2 + 1/4
    >>> euler(4, x)
    x**4 - 2*x**3 + x

    >>> euler(12, S.Half)
    2702765/4096
    >>> euler(12)
    2702765

    See Also
    ========

    andre, bell, bernoulli, catalan, fibonacci, harmonic, lucas, genocchi,
    partition, tribonacci, sympy.polys.appellseqs.euler_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euler_numbers
    .. [2] https://mathworld.wolfram.com/EulerNumber.html
    .. [3] https://en.wikipedia.org/wiki/Alternating_permutation
    .. [4] https://mathworld.wolfram.com/AlternatingPermutation.html

    """

    @classmethod
    def eval(cls, n, x=None):
        if n.is_zero:
            return S.One
        elif n is S.NegativeOne:
            if x is None:
                return S.Pi/2
            from sympy.functions.special.gamma_functions import digamma
            return digamma((x+1)/2) - digamma(x/2)
        elif n.is_integer is False or n.is_nonnegative is False:
            return
        # Euler numbers
        elif x is None:
            if n.is_odd and n.is_positive:
                return S.Zero
            elif n.is_Number:
                from mpmath import mp
                n = n._to_mpmath(mp.prec)
                res = mp.eulernum(n, exact=True)
                return Integer(res)
        # Euler polynomials
        elif n.is_Number:
            from sympy.core.evalf import pure_complex
            n = int(n)
            reim = pure_complex(x, or_real=True)
            if reim and all(a.is_Float or a.is_Integer for a in reim) \
                    and any(a.is_Float for a in reim):
                from mpmath import mp
                prec = min([a._prec for a in reim if a.is_Float])
                with workprec(prec):
                    res = mp.eulerpoly(n, x)
                return Expr._from_mpmath(res, prec)
            return euler_poly(n, x)

    def _eval_rewrite_as_Sum(self, n, x=None, **kwargs):
        from sympy.concrete.summations import Sum
        if x is None and n.is_even:
            k = Dummy("k", integer=True)
            j = Dummy("j", integer=True)
            n = n / 2
            Em = (S.ImaginaryUnit * Sum(Sum(binomial(k, j) * (S.NegativeOne**j *
                                                              (k - 2*j)**(2*n + 1)) /
                  (2**k*S.ImaginaryUnit**k * k), (j, 0, k)), (k, 1, 2*n + 1)))
            return Em
        if x:
            k = Dummy("k", integer=True)
            return Sum(binomial(n, k)*euler(k)/2**k*(x - S.Half)**(n - k), (k, 0, n))

    def _eval_rewrite_as_genocchi(self, n, x=None, **kwargs):
        if x is None:
            return Piecewise((S.Pi/2, Eq(n, -1)),
                             (-2**n * genocchi(n+1, S.Half) / (n+1), True))
        from sympy.functions.special.gamma_functions import digamma
        return Piecewise((digamma((x+1)/2) - digamma(x/2), Eq(n, -1)),
                         (-genocchi(n+1, x) / (n+1), True))

    def _eval_evalf(self, prec):
        if not all(i.is_number for i in self.args):
            return
        from mpmath import mp
        m, x = (self.args[0], None) if len(self.args) == 1 else self.args
        m = m._to_mpmath(prec)
        if x is not None:
            x = x._to_mpmath(prec)
        with workprec(prec):
            if mp.isint(m) and m >= 0:
                res = mp.eulernum(m) if x is None else mp.eulerpoly(m, x)
            else:
                if m == -1:
                    res = mp.pi if x is None else mp.digamma((x+1)/2) - mp.digamma(x/2)
                else:
                    y = 0.5 if x is None else x
                    res = 2 * (mp.zeta(-m, y) - 2**(m+1) * mp.zeta(-m, (y+1)/2))
                if x is None:
                    res *= 2**m
        return Expr._from_mpmath(res, prec)


#----------------------------------------------------------------------------#
#                                                                            #
#                              Catalan numbers                               #
#                                                                            #
#----------------------------------------------------------------------------#


class catalan(DefinedFunction):
    r"""
    Catalan numbers

    The `n^{th}` catalan number is given by:

    .. math :: C_n = \frac{1}{n+1} \binom{2n}{n}

    * ``catalan(n)`` gives the `n^{th}` Catalan number, `C_n`

    Examples
    ========

    >>> from sympy import (Symbol, binomial, gamma, hyper,
    ...     catalan, diff, combsimp, Rational, I)

    >>> [catalan(i) for i in range(1,10)]
    [1, 2, 5, 14, 42, 132, 429, 1430, 4862]

    >>> n = Symbol("n", integer=True)

    >>> catalan(n)
    catalan(n)

    Catalan numbers can be transformed into several other, identical
    expressions involving other mathematical functions

    >>> catalan(n).rewrite(binomial)
    binomial(2*n, n)/(n + 1)

    >>> catalan(n).rewrite(gamma)
    4**n*gamma(n + 1/2)/(sqrt(pi)*gamma(n + 2))

    >>> catalan(n).rewrite(hyper)
    hyper((-n, 1 - n), (2,), 1)

    For some non-integer values of n we can get closed form
    expressions by rewriting in terms of gamma functions:

    >>> catalan(Rational(1, 2)).rewrite(gamma)
    8/(3*pi)

    We can differentiate the Catalan numbers C(n) interpreted as a
    continuous real function in n:

    >>> diff(catalan(n), n)
    (polygamma(0, n + 1/2) - polygamma(0, n + 2) + log(4))*catalan(n)

    As a more advanced example consider the following ratio
    between consecutive numbers:

    >>> combsimp((catalan(n + 1)/catalan(n)).rewrite(binomial))
    2*(2*n + 1)/(n + 2)

    The Catalan numbers can be generalized to complex numbers:

    >>> catalan(I).rewrite(gamma)
    4**I*gamma(1/2 + I)/(sqrt(pi)*gamma(2 + I))

    and evaluated with arbitrary precision:

    >>> catalan(I).evalf(20)
    0.39764993382373624267 - 0.020884341620842555705*I

    See Also
    ========

    andre, bell, bernoulli, euler, fibonacci, harmonic, lucas, genocchi,
    partition, tribonacci, sympy.functions.combinatorial.factorials.binomial

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Catalan_number
    .. [2] https://mathworld.wolfram.com/CatalanNumber.html
    .. [3] https://functions.wolfram.com/GammaBetaErf/CatalanNumber/
    .. [4] http://geometer.org/mathcircles/catalan.pdf

    """

    @classmethod
    def eval(cls, n):
        from sympy.functions.special.gamma_functions import gamma
        if (n.is_Integer and n.is_nonnegative) or \
           (n.is_noninteger and n.is_negative):
            return 4**n*gamma(n + S.Half)/(gamma(S.Half)*gamma(n + 2))

        if (n.is_integer and n.is_negative):
            if (n + 1).is_negative:
                return S.Zero
            if (n + 1).is_zero:
                return Rational(-1, 2)

    def fdiff(self, argindex=1):
        from sympy.functions.elementary.exponential import log
        from sympy.functions.special.gamma_functions import polygamma
        n = self.args[0]
        return catalan(n)*(polygamma(0, n + S.Half) - polygamma(0, n + 2) + log(4))

    def _eval_rewrite_as_binomial(self, n, **kwargs):
        return binomial(2*n, n)/(n + 1)

    def _eval_rewrite_as_factorial(self, n, **kwargs):
        return factorial(2*n) / (factorial(n+1) * factorial(n))

    def _eval_rewrite_as_gamma(self, n, piecewise=True, **kwargs):
        from sympy.functions.special.gamma_functions import gamma
        # The gamma function allows to generalize Catalan numbers to complex n
        return 4**n*gamma(n + S.Half)/(gamma(S.Half)*gamma(n + 2))

    def _eval_rewrite_as_hyper(self, n, **kwargs):
        from sympy.functions.special.hyper import hyper
        return hyper([1 - n, -n], [2], 1)

    def _eval_rewrite_as_Product(self, n, **kwargs):
        from sympy.concrete.products import Product
        if not (n.is_integer and n.is_nonnegative):
            return self
        k = Dummy('k', integer=True, positive=True)
        return Product((n + k) / k, (k, 2, n))

    def _eval_is_integer(self):
        if self.args[0].is_integer and self.args[0].is_nonnegative:
            return True

    def _eval_is_positive(self):
        if self.args[0].is_nonnegative:
            return True

    def _eval_is_composite(self):
        if self.args[0].is_integer and (self.args[0] - 3).is_positive:
            return True

    def _eval_evalf(self, prec):
        from sympy.functions.special.gamma_functions import gamma
        if self.args[0].is_number:
            return self.rewrite(gamma)._eval_evalf(prec)


#----------------------------------------------------------------------------#
#                                                                            #
#                           Genocchi numbers                                 #
#                                                                            #
#----------------------------------------------------------------------------#


class genocchi(DefinedFunction):
    r"""
    Genocchi numbers / Genocchi polynomials / Genocchi function

    The Genocchi numbers are a sequence of integers `G_n` that satisfy the
    relation:

    .. math:: \frac{-2t}{1 + e^{-t}} = \sum_{n=0}^\infty \frac{G_n t^n}{n!}

    They are related to the Bernoulli numbers by

    .. math:: G_n = 2 (1 - 2^n) B_n

    and generalize like the Bernoulli numbers to the Genocchi polynomials and
    function as

    .. math:: \operatorname{G}(s, a) = 2 \left(\operatorname{B}(s, a) -
              2^s \operatorname{B}\left(s, \frac{a+1}{2}\right)\right)

    .. versionchanged:: 1.12
        ``genocchi(1)`` gives `-1` instead of `1`.

    Examples
    ========

    >>> from sympy import genocchi, Symbol
    >>> [genocchi(n) for n in range(9)]
    [0, -1, -1, 0, 1, 0, -3, 0, 17]
    >>> n = Symbol('n', integer=True, positive=True)
    >>> genocchi(2*n + 1)
    0
    >>> x = Symbol('x')
    >>> genocchi(4, x)
    -4*x**3 + 6*x**2 - 1

    See Also
    ========

    bell, bernoulli, catalan, euler, fibonacci, harmonic, lucas, partition, tribonacci
    sympy.polys.appellseqs.genocchi_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Genocchi_number
    .. [2] https://mathworld.wolfram.com/GenocchiNumber.html
    .. [3] Peter Luschny, "An introduction to the Bernoulli function",
           https://arxiv.org/abs/2009.06743

    """

    @classmethod
    def eval(cls, n, x=None):
        if x is S.One:
            return cls(n)
        elif n.is_integer is False or n.is_nonnegative is False:
            return
        # Genocchi numbers
        elif x is None:
            if n.is_odd and (n-1).is_positive:
                return S.Zero
            elif n.is_Number:
                return 2 * (1-S(2)**n) * bernoulli(n)
        # Genocchi polynomials
        elif n.is_Number:
            return genocchi_poly(n, x)

    def _eval_rewrite_as_bernoulli(self, n, x=1, **kwargs):
        if x == 1 and n.is_integer and n.is_nonnegative:
            return 2 * (1-S(2)**n) * bernoulli(n)
        return 2 * (bernoulli(n, x) - 2**n * bernoulli(n, (x+1) / 2))

    def _eval_rewrite_as_dirichlet_eta(self, n, x=1, **kwargs):
        from sympy.functions.special.zeta_functions import dirichlet_eta
        return -2*n * dirichlet_eta(1-n, x)

    def _eval_is_integer(self):
        if len(self.args) > 1 and self.args[1] != 1:
            return
        n = self.args[0]
        if n.is_integer and n.is_nonnegative:
            return True

    def _eval_is_negative(self):
        if len(self.args) > 1 and self.args[1] != 1:
            return
        n = self.args[0]
        if n.is_integer and n.is_nonnegative:
            if n.is_odd:
                return fuzzy_not((n-1).is_positive)
            return (n/2).is_odd

    def _eval_is_positive(self):
        if len(self.args) > 1 and self.args[1] != 1:
            return
        n = self.args[0]
        if n.is_integer and n.is_nonnegative:
            if n.is_zero or n.is_odd:
                return False
            return (n/2).is_even

    def _eval_is_even(self):
        if len(self.args) > 1 and self.args[1] != 1:
            return
        n = self.args[0]
        if n.is_integer and n.is_nonnegative:
            if n.is_even:
                return n.is_zero
            return (n-1).is_positive

    def _eval_is_odd(self):
        if len(self.args) > 1 and self.args[1] != 1:
            return
        n = self.args[0]
        if n.is_integer and n.is_nonnegative:
            if n.is_even:
                return fuzzy_not(n.is_zero)
            return fuzzy_not((n-1).is_positive)

    def _eval_is_prime(self):
        if len(self.args) > 1 and self.args[1] != 1:
            return
        n = self.args[0]
        # only G_6 = -3 and G_8 = 17 are prime,
        # but SymPy does not consider negatives as prime
        # so only n=8 is tested
        return (n-8).is_zero

    def _eval_evalf(self, prec):
        if all(i.is_number for i in self.args):
            return self.rewrite(bernoulli)._eval_evalf(prec)


#----------------------------------------------------------------------------#
#                                                                            #
#                              Andre numbers                                 #
#                                                                            #
#----------------------------------------------------------------------------#


class andre(DefinedFunction):
    r"""
    Andre numbers / Andre function

    The Andre number `\mathcal{A}_n` is Luschny's name for half the number of
    *alternating permutations* on `n` elements, where a permutation is alternating
    if adjacent elements alternately compare "greater" and "smaller" going from
    left to right. For example, `2 < 3 > 1 < 4` is an alternating permutation.

    This sequence is A000111 in the OEIS, which assigns the names *up/down numbers*
    and *Euler zigzag numbers*. It satisfies a recurrence relation similar to that
    for the Catalan numbers, with `\mathcal{A}_0 = 1` and

    .. math:: 2 \mathcal{A}_{n+1} = \sum_{k=0}^n \binom{n}{k} \mathcal{A}_k \mathcal{A}_{n-k}

    The Bernoulli and Euler numbers are signed transformations of the odd- and
    even-indexed elements of this sequence respectively:

    .. math :: \operatorname{B}_{2k} = \frac{2k \mathcal{A}_{2k-1}}{(-4)^k - (-16)^k}

    .. math :: \operatorname{E}_{2k} = (-1)^k \mathcal{A}_{2k}

    Like the Bernoulli and Euler numbers, the Andre numbers are interpolated by the
    entire Andre function:

    .. math :: \mathcal{A}(s) = (-i)^{s+1} \operatorname{Li}_{-s}(i) +
            i^{s+1} \operatorname{Li}_{-s}(-i) = \\ \frac{2 \Gamma(s+1)}{(2\pi)^{s+1}}
            (\zeta(s+1, 1/4) - \zeta(s+1, 3/4) \cos{\pi s})

    Examples
    ========

    >>> from sympy import andre, euler, bernoulli
    >>> [andre(n) for n in range(11)]
    [1, 1, 1, 2, 5, 16, 61, 272, 1385, 7936, 50521]
    >>> [(-1)**k * andre(2*k) for k in range(7)]
    [1, -1, 5, -61, 1385, -50521, 2702765]
    >>> [euler(2*k) for k in range(7)]
    [1, -1, 5, -61, 1385, -50521, 2702765]
    >>> [andre(2*k-1) * (2*k) / ((-4)**k - (-16)**k) for k in range(1, 8)]
    [1/6, -1/30, 1/42, -1/30, 5/66, -691/2730, 7/6]
    >>> [bernoulli(2*k) for k in range(1, 8)]
    [1/6, -1/30, 1/42, -1/30, 5/66, -691/2730, 7/6]

    See Also
    ========

    bernoulli, catalan, euler, sympy.polys.appellseqs.andre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Alternating_permutation
    .. [2] https://mathworld.wolfram.com/EulerZigzagNumber.html
    .. [3] Peter Luschny, "An introduction to the Bernoulli function",
           https://arxiv.org/abs/2009.06743
    """

    @classmethod
    def eval(cls, n):
        if n is S.NaN:
            return S.NaN
        elif n is S.Infinity:
            return S.Infinity
        if n.is_zero:
            return S.One
        elif n == -1:
            return -log(2)
        elif n == -2:
            return -2*S.Catalan
        elif n.is_Integer:
            if n.is_nonnegative and n.is_even:
                return abs(euler(n))
            elif n.is_odd:
                from sympy.functions.special.zeta_functions import zeta
                m = -n-1
                return I**m * Rational(1-2**m, 4**m) * zeta(-n)

    def _eval_rewrite_as_zeta(self, s, **kwargs):
        from sympy.functions.elementary.trigonometric import cos
        from sympy.functions.special.gamma_functions import gamma
        from sympy.functions.special.zeta_functions import zeta
        return 2 * gamma(s+1) / (2*pi)**(s+1) * \
                (zeta(s+1, S.One/4) - cos(pi*s) * zeta(s+1, S(3)/4))

    def _eval_rewrite_as_polylog(self, s, **kwargs):
        from sympy.functions.special.zeta_functions import polylog
        return (-I)**(s+1) * polylog(-s, I) + I**(s+1) * polylog(-s, -I)

    def _eval_is_integer(self):
        n = self.args[0]
        if n.is_integer and n.is_nonnegative:
            return True

    def _eval_is_positive(self):
        if self.args[0].is_nonnegative:
            return True

    def _eval_evalf(self, prec):
        if not self.args[0].is_number:
            return
        s = self.args[0]._to_mpmath(prec+12)
        with workprec(prec+12):
            sp, cp = mp.sinpi(s/2), mp.cospi(s/2)
            res = 2*mp.dirichlet(-s, (-sp, cp, sp, -cp))
        return Expr._from_mpmath(res, prec)


#----------------------------------------------------------------------------#
#                                                                            #
#                           Partition numbers                                #
#                                                                            #
#----------------------------------------------------------------------------#

class partition(DefinedFunction):
    r"""
    Partition numbers

    The Partition numbers are a sequence of integers `p_n` that represent the
    number of distinct ways of representing `n` as a sum of natural numbers
    (with order irrelevant). The generating function for `p_n` is given by:

    .. math:: \sum_{n=0}^\infty p_n x^n = \prod_{k=1}^\infty (1 - x^k)^{-1}

    Examples
    ========

    >>> from sympy import partition, Symbol
    >>> [partition(n) for n in range(9)]
    [1, 1, 2, 3, 5, 7, 11, 15, 22]
    >>> n = Symbol('n', integer=True, negative=True)
    >>> partition(n)
    0

    See Also
    ========

    bell, bernoulli, catalan, euler, fibonacci, harmonic, lucas, genocchi, tribonacci

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Partition_(number_theory%29
    .. [2] https://en.wikipedia.org/wiki/Pentagonal_number_theorem

    """
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, n):
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if n.is_negative is True:
            return S.Zero
        if n.is_zero is True or n is S.One:
            return S.One
        if n.is_Integer is True:
            return S(_partition(as_int(n)))

    def _eval_is_positive(self):
        if self.args[0].is_nonnegative is True:
            return True

    def _eval_Mod(self, q):
        # Ramanujan's congruences
        n = self.args[0]
        for p, rem in [(5, 4), (7, 5), (11, 6)]:
            if q == p and n % q == rem:
                return S.Zero


class divisor_sigma(DefinedFunction):
    r"""
    Calculate the divisor function `\sigma_k(n)` for positive integer n

    ``divisor_sigma(n, k)`` is equal to ``sum([x**k for x in divisors(n)])``

    If n's prime factorization is:

    .. math ::
        n = \prod_{i=1}^\omega p_i^{m_i},

    then

    .. math ::
        \sigma_k(n) = \prod_{i=1}^\omega (1+p_i^k+p_i^{2k}+\cdots
        + p_i^{m_ik}).

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import divisor_sigma
    >>> divisor_sigma(18, 0)
    6
    >>> divisor_sigma(39, 1)
    56
    >>> divisor_sigma(12, 2)
    210
    >>> divisor_sigma(37)
    38

    See Also
    ========

    sympy.ntheory.factor_.divisor_count, totient, sympy.ntheory.factor_.divisors, sympy.ntheory.factor_.factorint

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Divisor_function

    """
    is_integer = True
    is_positive = True

    @classmethod
    def eval(cls, n, k=S.One):
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if n.is_positive is False:
            raise ValueError("n should be a positive integer")
        if k.is_integer is False:
            raise TypeError("k should be an integer")
        if k.is_nonnegative is False:
            raise ValueError("k should be a nonnegative integer")
        if n.is_prime is True:
            return 1 + n**k
        if n is S.One:
            return S.One
        if n.is_Integer is True:
            if k.is_zero is True:
                return Mul(*[e + 1 for e in factorint(n).values()])
            if k.is_Integer is True:
                return S(_divisor_sigma(as_int(n), as_int(k)))
            if k.is_zero is False:
                return Mul(*[cancel((p**(k*(e + 1)) - 1) / (p**k - 1)) for p, e in factorint(n).items()])


class udivisor_sigma(DefinedFunction):
    r"""
    Calculate the unitary divisor function `\sigma_k^*(n)` for positive integer n

    ``udivisor_sigma(n, k)`` is equal to ``sum([x**k for x in udivisors(n)])``

    If n's prime factorization is:

    .. math ::
        n = \prod_{i=1}^\omega p_i^{m_i},

    then

    .. math ::
        \sigma_k^*(n) = \prod_{i=1}^\omega (1+ p_i^{m_ik}).

    Parameters
    ==========

    k : power of divisors in the sum

        for k = 0, 1:
        ``udivisor_sigma(n, 0)`` is equal to ``udivisor_count(n)``
        ``udivisor_sigma(n, 1)`` is equal to ``sum(udivisors(n))``

        Default for k is 1.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import udivisor_sigma
    >>> udivisor_sigma(18, 0)
    4
    >>> udivisor_sigma(74, 1)
    114
    >>> udivisor_sigma(36, 3)
    47450
    >>> udivisor_sigma(111)
    152

    See Also
    ========

    sympy.ntheory.factor_.divisor_count, totient, sympy.ntheory.factor_.divisors,
    sympy.ntheory.factor_.udivisors, sympy.ntheory.factor_.udivisor_count, divisor_sigma,
    sympy.ntheory.factor_.factorint

    References
    ==========

    .. [1] https://mathworld.wolfram.com/UnitaryDivisorFunction.html

    """
    is_integer = True
    is_positive = True

    @classmethod
    def eval(cls, n, k=S.One):
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if n.is_positive is False:
            raise ValueError("n should be a positive integer")
        if k.is_integer is False:
            raise TypeError("k should be an integer")
        if k.is_nonnegative is False:
            raise ValueError("k should be a nonnegative integer")
        if n.is_prime is True:
            return 1 + n**k
        if n.is_Integer:
            return Mul(*[1+p**(k*e) for p, e in factorint(n).items()])


class legendre_symbol(DefinedFunction):
    r"""
    Returns the Legendre symbol `(a / p)`.

    For an integer ``a`` and an odd prime ``p``, the Legendre symbol is
    defined as

    .. math ::
        \genfrac(){}{}{a}{p} = \begin{cases}
             0 & \text{if } p \text{ divides } a\\
             1 & \text{if } a \text{ is a quadratic residue modulo } p\\
            -1 & \text{if } a \text{ is a quadratic nonresidue modulo } p
        \end{cases}

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import legendre_symbol
    >>> [legendre_symbol(i, 7) for i in range(7)]
    [0, 1, 1, -1, 1, -1, -1]
    >>> sorted(set([i**2 % 7 for i in range(7)]))
    [0, 1, 2, 4]

    See Also
    ========

    sympy.ntheory.residue_ntheory.is_quad_residue, jacobi_symbol

    """
    is_integer = True
    is_prime = False

    @classmethod
    def eval(cls, a, p):
        if a.is_integer is False:
            raise TypeError("a should be an integer")
        if p.is_integer is False:
            raise TypeError("p should be an integer")
        if p.is_prime is False or p.is_odd is False:
            raise ValueError("p should be an odd prime integer")
        if (a % p).is_zero is True:
            return S.Zero
        if a is S.One:
            return S.One
        if a.is_Integer is True and p.is_Integer is True:
            return S(legendre(as_int(a), as_int(p)))


class jacobi_symbol(DefinedFunction):
    r"""
    Returns the Jacobi symbol `(m / n)`.

    For any integer ``m`` and any positive odd integer ``n`` the Jacobi symbol
    is defined as the product of the Legendre symbols corresponding to the
    prime factors of ``n``:

    .. math ::
        \genfrac(){}{}{m}{n} =
            \genfrac(){}{}{m}{p^{1}}^{\alpha_1}
            \genfrac(){}{}{m}{p^{2}}^{\alpha_2}
            ...
            \genfrac(){}{}{m}{p^{k}}^{\alpha_k}
            \text{ where } n =
                p_1^{\alpha_1}
                p_2^{\alpha_2}
                ...
                p_k^{\alpha_k}

    Like the Legendre symbol, if the Jacobi symbol `\genfrac(){}{}{m}{n} = -1`
    then ``m`` is a quadratic nonresidue modulo ``n``.

    But, unlike the Legendre symbol, if the Jacobi symbol
    `\genfrac(){}{}{m}{n} = 1` then ``m`` may or may not be a quadratic residue
    modulo ``n``.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import jacobi_symbol, legendre_symbol
    >>> from sympy import S
    >>> jacobi_symbol(45, 77)
    -1
    >>> jacobi_symbol(60, 121)
    1

    The relationship between the ``jacobi_symbol`` and ``legendre_symbol`` can
    be demonstrated as follows:

    >>> L = legendre_symbol
    >>> S(45).factors()
    {3: 2, 5: 1}
    >>> jacobi_symbol(7, 45) == L(7, 3)**2 * L(7, 5)**1
    True

    See Also
    ========

    sympy.ntheory.residue_ntheory.is_quad_residue, legendre_symbol

    """
    is_integer = True
    is_prime = False

    @classmethod
    def eval(cls, m, n):
        if m.is_integer is False:
            raise TypeError("m should be an integer")
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if n.is_positive is False or n.is_odd is False:
            raise ValueError("n should be an odd positive integer")
        if m is S.One or n is S.One:
            return S.One
        if (m % n).is_zero is True:
            return S.Zero
        if m.is_Integer is True and n.is_Integer is True:
            return S(jacobi(as_int(m), as_int(n)))


class kronecker_symbol(DefinedFunction):
    r"""
    Returns the Kronecker symbol `(a / n)`.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import kronecker_symbol
    >>> kronecker_symbol(45, 77)
    -1
    >>> kronecker_symbol(13, -120)
    1

    See Also
    ========

    jacobi_symbol, legendre_symbol

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Kronecker_symbol

    """
    is_integer = True
    is_prime = False

    @classmethod
    def eval(cls, a, n):
        if a.is_integer is False:
            raise TypeError("a should be an integer")
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if a is S.One or n is S.One:
            return S.One
        if a.is_Integer is True and n.is_Integer is True:
            return S(kronecker(as_int(a), as_int(n)))


class mobius(DefinedFunction):
    """
    Mobius function maps natural number to {-1, 0, 1}

    It is defined as follows:
        1) `1` if `n = 1`.
        2) `0` if `n` has a squared prime factor.
        3) `(-1)^k` if `n` is a square-free positive integer with `k`
           number of prime factors.

    It is an important multiplicative function in number theory
    and combinatorics.  It has applications in mathematical series,
    algebraic number theory and also physics (Fermion operator has very
    concrete realization with Mobius Function model).

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import mobius
    >>> mobius(13*7)
    1
    >>> mobius(1)
    1
    >>> mobius(13*7*5)
    -1
    >>> mobius(13**2)
    0

    Even in the case of a symbol, if it clearly contains a squared prime factor, it will be zero.

    >>> from sympy import Symbol
    >>> n = Symbol("n", integer=True, positive=True)
    >>> mobius(4*n)
    0
    >>> mobius(n**2)
    0

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/M%C3%B6bius_function
    .. [2] Thomas Koshy "Elementary Number Theory with Applications"
    .. [3] https://oeis.org/A008683

    """
    is_integer = True
    is_prime = False

    @classmethod
    def eval(cls, n):
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if n.is_positive is False:
            raise ValueError("n should be a positive integer")
        if n.is_prime is True:
            return S.NegativeOne
        if n is S.One:
            return S.One
        result = None
        for m, e in (_.as_base_exp() for _ in Mul.make_args(n)):
            if m.is_integer is True and m.is_positive is True and \
               e.is_integer is True and e.is_positive is True:
                lt = is_lt(S.One, e) # 1 < e
                if lt is True:
                    result = S.Zero
                elif m.is_Integer is True:
                    factors = factorint(m)
                    if any(v > 1 for v in factors.values()):
                        result = S.Zero
                    elif lt is False:
                        s = S.NegativeOne if len(factors) % 2 else S.One
                        if result is None:
                            result = s
                        else:
                            result *= s
            else:
                return
        return result


class primenu(DefinedFunction):
    r"""
    Calculate the number of distinct prime factors for a positive integer n.

    If n's prime factorization is:

    .. math ::
        n = \prod_{i=1}^k p_i^{m_i},

    then ``primenu(n)`` or `\nu(n)` is:

    .. math ::
        \nu(n) = k.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import primenu
    >>> primenu(1)
    0
    >>> primenu(30)
    3

    See Also
    ========

    sympy.ntheory.factor_.factorint

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PrimeFactor.html
    .. [2] https://oeis.org/A001221

    """
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, n):
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if n.is_positive is False:
            raise ValueError("n should be a positive integer")
        if n.is_prime is True:
            return S.One
        if n is S.One:
            return S.Zero
        if n.is_Integer is True:
            return S(len(factorint(n)))


class primeomega(DefinedFunction):
    r"""
    Calculate the number of prime factors counting multiplicities for a
    positive integer n.

    If n's prime factorization is:

    .. math ::
        n = \prod_{i=1}^k p_i^{m_i},

    then ``primeomega(n)``  or `\Omega(n)` is:

    .. math ::
        \Omega(n) = \sum_{i=1}^k m_i.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import primeomega
    >>> primeomega(1)
    0
    >>> primeomega(20)
    3

    See Also
    ========

    sympy.ntheory.factor_.factorint

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PrimeFactor.html
    .. [2] https://oeis.org/A001222

    """
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, n):
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if n.is_positive is False:
            raise ValueError("n should be a positive integer")
        if n.is_prime is True:
            return S.One
        if n is S.One:
            return S.Zero
        if n.is_Integer is True:
            return S(sum(factorint(n).values()))


class totient(DefinedFunction):
    r"""
    Calculate the Euler totient function phi(n)

    ``totient(n)`` or `\phi(n)` is the number of positive integers `\leq` n
    that are relatively prime to n.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import totient
    >>> totient(1)
    1
    >>> totient(25)
    20
    >>> totient(45) == totient(5)*totient(9)
    True

    See Also
    ========

    sympy.ntheory.factor_.divisor_count

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euler%27s_totient_function
    .. [2] https://mathworld.wolfram.com/TotientFunction.html
    .. [3] https://oeis.org/A000010

    """
    is_integer = True
    is_positive = True

    @classmethod
    def eval(cls, n):
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if n.is_positive is False:
            raise ValueError("n should be a positive integer")
        if n is S.One:
            return S.One
        if n.is_prime is True:
            return n - 1
        if isinstance(n, Dict):
            return S(prod(p**(k-1)*(p-1) for p, k in n.items()))
        if n.is_Integer is True:
            return S(prod(p**(k-1)*(p-1) for p, k in factorint(n).items()))


class reduced_totient(DefinedFunction):
    r"""
    Calculate the Carmichael reduced totient function lambda(n)

    ``reduced_totient(n)`` or `\lambda(n)` is the smallest m > 0 such that
    `k^m \equiv 1 \mod n` for all k relatively prime to n.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import reduced_totient
    >>> reduced_totient(1)
    1
    >>> reduced_totient(8)
    2
    >>> reduced_totient(30)
    4

    See Also
    ========

    totient

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Carmichael_function
    .. [2] https://mathworld.wolfram.com/CarmichaelFunction.html
    .. [3] https://oeis.org/A002322

    """
    is_integer = True
    is_positive = True

    @classmethod
    def eval(cls, n):
        if n.is_integer is False:
            raise TypeError("n should be an integer")
        if n.is_positive is False:
            raise ValueError("n should be a positive integer")
        if n is S.One:
            return S.One
        if n.is_prime is True:
            return n - 1
        if isinstance(n, Dict):
            t = 1
            if 2 in n:
                t = (1 << (n[2] - 2)) if 2 < n[2] else n[2]
            return S(lcm(int(t), *(int(p-1)*int(p)**int(k-1) for p, k in n.items() if p != 2)))
        if n.is_Integer is True:
            n, t = remove(int(n), 2)
            if not t:
                t = 1
            elif 2 < t:
                t = 1 << (t - 2)
            return S(lcm(t, *((p-1)*p**(k-1) for p, k in factorint(n).items())))


class primepi(DefinedFunction):
    r""" Represents the prime counting function pi(n) = the number
    of prime numbers less than or equal to n.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import primepi
    >>> from sympy import prime, prevprime, isprime
    >>> primepi(25)
    9

    So there are 9 primes less than or equal to 25. Is 25 prime?

    >>> isprime(25)
    False

    It is not. So the first prime less than 25 must be the
    9th prime:

    >>> prevprime(25) == prime(9)
    True

    See Also
    ========

    sympy.ntheory.primetest.isprime : Test if n is prime
    sympy.ntheory.generate.primerange : Generate all primes in a given range
    sympy.ntheory.generate.prime : Return the nth prime

    References
    ==========

    .. [1] https://oeis.org/A000720

    """
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, n):
        if n is S.Infinity:
            return S.Infinity
        if n is S.NegativeInfinity:
            return S.Zero
        if n.is_real is False:
            raise TypeError("n should be a real")
        if is_lt(n, S(2)) is True:
            return S.Zero
        try:
            n = int(n)
        except TypeError:
            return
        return S(_primepi(n))


#######################################################################
###
### Functions for enumerating partitions, permutations and combinations
###
#######################################################################


class _MultisetHistogram(tuple):
    __slots__ = ()


_N = -1
_ITEMS = -2
_M = slice(None, _ITEMS)


def _multiset_histogram(n):
    """Return tuple used in permutation and combination counting. Input
    is a dictionary giving items with counts as values or a sequence of
    items (which need not be sorted).

    The data is stored in a class deriving from tuple so it is easily
    recognized and so it can be converted easily to a list.
    """
    if isinstance(n, dict):  # item: count
        if not all(isinstance(v, int) and v >= 0 for v in n.values()):
            raise ValueError
        tot = sum(n.values())
        items = sum(1 for k in n if n[k] > 0)
        return _MultisetHistogram([n[k] for k in n if n[k] > 0] + [items, tot])
    else:
        n = list(n)
        s = set(n)
        lens = len(s)
        lenn = len(n)
        if lens == lenn:
            n = [1]*lenn + [lenn, lenn]
            return _MultisetHistogram(n)
        m = dict(zip(s, range(lens)))
        d = dict(zip(range(lens), (0,)*lens))
        for i in n:
            d[m[i]] += 1
        return _multiset_histogram(d)


def nP(n, k=None, replacement=False):
    """Return the number of permutations of ``n`` items taken ``k`` at a time.

    Possible values for ``n``:

        integer - set of length ``n``

        sequence - converted to a multiset internally

        multiset - {element: multiplicity}

    If ``k`` is None then the total of all permutations of length 0
    through the number of items represented by ``n`` will be returned.

    If ``replacement`` is True then a given item can appear more than once
    in the ``k`` items. (For example, for 'ab' permutations of 2 would
    include 'aa', 'ab', 'ba' and 'bb'.) The multiplicity of elements in
    ``n`` is ignored when ``replacement`` is True but the total number
    of elements is considered since no element can appear more times than
    the number of elements in ``n``.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import nP
    >>> from sympy.utilities.iterables import multiset_permutations, multiset
    >>> nP(3, 2)
    6
    >>> nP('abc', 2) == nP(multiset('abc'), 2) == 6
    True
    >>> nP('aab', 2)
    3
    >>> nP([1, 2, 2], 2)
    3
    >>> [nP(3, i) for i in range(4)]
    [1, 3, 6, 6]
    >>> nP(3) == sum(_)
    True

    When ``replacement`` is True, each item can have multiplicity
    equal to the length represented by ``n``:

    >>> nP('aabc', replacement=True)
    121
    >>> [len(list(multiset_permutations('aaaabbbbcccc', i))) for i in range(5)]
    [1, 3, 9, 27, 81]
    >>> sum(_)
    121

    See Also
    ========
    sympy.utilities.iterables.multiset_permutations

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Permutation

    """
    try:
        n = as_int(n)
    except ValueError:
        return Integer(_nP(_multiset_histogram(n), k, replacement))
    return Integer(_nP(n, k, replacement))


@cacheit
def _nP(n, k=None, replacement=False):

    if k == 0:
        return 1
    if isinstance(n, SYMPY_INTS):  # n different items
        # assert n >= 0
        if k is None:
            return sum(_nP(n, i, replacement) for i in range(n + 1))
        elif replacement:
            return n**k
        elif k > n:
            return 0
        elif k == n:
            return factorial(k)
        elif k == 1:
            return n
        else:
            # assert k >= 0
            return _product(n - k + 1, n)
    elif isinstance(n, _MultisetHistogram):
        if k is None:
            return sum(_nP(n, i, replacement) for i in range(n[_N] + 1))
        elif replacement:
            return n[_ITEMS]**k
        elif k == n[_N]:
            return factorial(k)/prod([factorial(i) for i in n[_M] if i > 1])
        elif k > n[_N]:
            return 0
        elif k == 1:
            return n[_ITEMS]
        else:
            # assert k >= 0
            tot = 0
            n = list(n)
            for i in range(len(n[_M])):
                if not n[i]:
                    continue
                n[_N] -= 1
                if n[i] == 1:
                    n[i] = 0
                    n[_ITEMS] -= 1
                    tot += _nP(_MultisetHistogram(n), k - 1)
                    n[_ITEMS] += 1
                    n[i] = 1
                else:
                    n[i] -= 1
                    tot += _nP(_MultisetHistogram(n), k - 1)
                    n[i] += 1
                n[_N] += 1
            return tot


@cacheit
def _AOP_product(n):
    """for n = (m1, m2, .., mk) return the coefficients of the polynomial,
    prod(sum(x**i for i in range(nj + 1)) for nj in n); i.e. the coefficients
    of the product of AOPs (all-one polynomials) or order given in n.  The
    resulting coefficient corresponding to x**r is the number of r-length
    combinations of sum(n) elements with multiplicities given in n.
    The coefficients are given as a default dictionary (so if a query is made
    for a key that is not present, 0 will be returned).

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import _AOP_product
    >>> from sympy.abc import x
    >>> n = (2, 2, 3)  # e.g. aabbccc
    >>> prod = ((x**2 + x + 1)*(x**2 + x + 1)*(x**3 + x**2 + x + 1)).expand()
    >>> c = _AOP_product(n); dict(c)
    {0: 1, 1: 3, 2: 6, 3: 8, 4: 8, 5: 6, 6: 3, 7: 1}
    >>> [c[i] for i in range(8)] == [prod.coeff(x, i) for i in range(8)]
    True

    The generating poly used here is the same as that listed in
    https://tinyurl.com/cep849r, but in a refactored form.

    """

    n = list(n)
    ord = sum(n)
    need = (ord + 2)//2
    rv = [1]*(n.pop() + 1)
    rv.extend((0,) * (need - len(rv)))
    rv = rv[:need]
    while n:
        ni = n.pop()
        N = ni + 1
        was = rv[:]
        for i in range(1, min(N, len(rv))):
            rv[i] += rv[i - 1]
        for i in range(N, need):
            rv[i] += rv[i - 1] - was[i - N]
    rev = list(reversed(rv))
    if ord % 2:
        rv = rv + rev
    else:
        rv[-1:] = rev
    d = defaultdict(int)
    for i, r in enumerate(rv):
        d[i] = r
    return d


def nC(n, k=None, replacement=False):
    """Return the number of combinations of ``n`` items taken ``k`` at a time.

    Possible values for ``n``:

        integer - set of length ``n``

        sequence - converted to a multiset internally

        multiset - {element: multiplicity}

    If ``k`` is None then the total of all combinations of length 0
    through the number of items represented in ``n`` will be returned.

    If ``replacement`` is True then a given item can appear more than once
    in the ``k`` items. (For example, for 'ab' sets of 2 would include 'aa',
    'ab', and 'bb'.) The multiplicity of elements in ``n`` is ignored when
    ``replacement`` is True but the total number of elements is considered
    since no element can appear more times than the number of elements in
    ``n``.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import nC
    >>> from sympy.utilities.iterables import multiset_combinations
    >>> nC(3, 2)
    3
    >>> nC('abc', 2)
    3
    >>> nC('aab', 2)
    2

    When ``replacement`` is True, each item can have multiplicity
    equal to the length represented by ``n``:

    >>> nC('aabc', replacement=True)
    35
    >>> [len(list(multiset_combinations('aaaabbbbcccc', i))) for i in range(5)]
    [1, 3, 6, 10, 15]
    >>> sum(_)
    35

    If there are ``k`` items with multiplicities ``m_1, m_2, ..., m_k``
    then the total of all combinations of length 0 through ``k`` is the
    product, ``(m_1 + 1)*(m_2 + 1)*...*(m_k + 1)``. When the multiplicity
    of each item is 1 (i.e., k unique items) then there are 2**k
    combinations. For example, if there are 4 unique items, the total number
    of combinations is 16:

    >>> sum(nC(4, i) for i in range(5))
    16

    See Also
    ========

    sympy.utilities.iterables.multiset_combinations

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Combination
    .. [2] https://tinyurl.com/cep849r

    """

    if isinstance(n, SYMPY_INTS):
        if k is None:
            if not replacement:
                return 2**n
            return sum(nC(n, i, replacement) for i in range(n + 1))
        if k < 0:
            raise ValueError("k cannot be negative")
        if replacement:
            return binomial(n + k - 1, k)
        return binomial(n, k)
    if isinstance(n, _MultisetHistogram):
        N = n[_N]
        if k is None:
            if not replacement:
                return prod(m + 1 for m in n[_M])
            return sum(nC(n, i, replacement) for i in range(N + 1))
        elif replacement:
            return nC(n[_ITEMS], k, replacement)
        # assert k >= 0
        elif k in (1, N - 1):
            return n[_ITEMS]
        elif k in (0, N):
            return 1
        return _AOP_product(tuple(n[_M]))[k]
    else:
        return nC(_multiset_histogram(n), k, replacement)


def _eval_stirling1(n, k):
    if n == k == 0:
        return S.One
    if 0 in (n, k):
        return S.Zero

    # some special values
    if n == k:
        return S.One
    elif k == n - 1:
        return binomial(n, 2)
    elif k == n - 2:
        return (3*n - 1)*binomial(n, 3)/4
    elif k == n - 3:
        return binomial(n, 2)*binomial(n, 4)

    return _stirling1(n, k)


@cacheit
def _stirling1(n, k):
    row = [0, 1]+[0]*(k-1) # for n = 1
    for i in range(2, n+1):
        for j in range(min(k,i), 0, -1):
            row[j] = (i-1) * row[j] + row[j-1]
    return Integer(row[k])


def _eval_stirling2(n, k):
    if n == k == 0:
        return S.One
    if 0 in (n, k):
        return S.Zero

    # some special values
    if n == k:
        return S.One
    elif k == n - 1:
        return binomial(n, 2)
    elif k == 1:
        return S.One
    elif k == 2:
        return Integer(2**(n - 1) - 1)

    return _stirling2(n, k)


@cacheit
def _stirling2(n, k):
    row = [0, 1]+[0]*(k-1) # for n = 1
    for i in range(2, n+1):
        for j in range(min(k,i), 0, -1):
            row[j] = j * row[j] + row[j-1]
    return Integer(row[k])


def stirling(n, k, d=None, kind=2, signed=False):
    r"""Return Stirling number $S(n, k)$ of the first or second (default) kind.

    The sum of all Stirling numbers of the second kind for $k = 1$
    through $n$ is ``bell(n)``. The recurrence relationship for these numbers
    is:

    .. math :: {0 \brace 0} = 1; {n \brace 0} = {0 \brace k} = 0;

    .. math :: {{n+1} \brace k} = j {n \brace k} + {n \brace {k-1}}

    where $j$ is:
        $n$ for Stirling numbers of the first kind,
        $-n$ for signed Stirling numbers of the first kind,
        $k$ for Stirling numbers of the second kind.

    The first kind of Stirling number counts the number of permutations of
    ``n`` distinct items that have ``k`` cycles; the second kind counts the
    ways in which ``n`` distinct items can be partitioned into ``k`` parts.
    If ``d`` is given, the "reduced Stirling number of the second kind" is
    returned: $S^{d}(n, k) = S(n - d + 1, k - d + 1)$ with $n \ge k \ge d$.
    (This counts the ways to partition $n$ consecutive integers into $k$
    groups with no pairwise difference less than $d$. See example below.)

    To obtain the signed Stirling numbers of the first kind, use keyword
    ``signed=True``. Using this keyword automatically sets ``kind`` to 1.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import stirling, bell
    >>> from sympy.combinatorics import Permutation
    >>> from sympy.utilities.iterables import multiset_partitions, permutations

    First kind (unsigned by default):

    >>> [stirling(6, i, kind=1) for i in range(7)]
    [0, 120, 274, 225, 85, 15, 1]
    >>> perms = list(permutations(range(4)))
    >>> [sum(Permutation(p).cycles == i for p in perms) for i in range(5)]
    [0, 6, 11, 6, 1]
    >>> [stirling(4, i, kind=1) for i in range(5)]
    [0, 6, 11, 6, 1]

    First kind (signed):

    >>> [stirling(4, i, signed=True) for i in range(5)]
    [0, -6, 11, -6, 1]

    Second kind:

    >>> [stirling(10, i) for i in range(12)]
    [0, 1, 511, 9330, 34105, 42525, 22827, 5880, 750, 45, 1, 0]
    >>> sum(_) == bell(10)
    True
    >>> len(list(multiset_partitions(range(4), 2))) == stirling(4, 2)
    True

    Reduced second kind:

    >>> from sympy import subsets, oo
    >>> def delta(p):
    ...    if len(p) == 1:
    ...        return oo
    ...    return min(abs(i[0] - i[1]) for i in subsets(p, 2))
    >>> parts = multiset_partitions(range(5), 3)
    >>> d = 2
    >>> sum(1 for p in parts if all(delta(i) >= d for i in p))
    7
    >>> stirling(5, 3, 2)
    7

    See Also
    ========
    sympy.utilities.iterables.multiset_partitions


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Stirling_numbers_of_the_first_kind
    .. [2] https://en.wikipedia.org/wiki/Stirling_numbers_of_the_second_kind

    """
    # TODO: make this a class like bell()

    n = as_int(n)
    k = as_int(k)
    if n < 0:
        raise ValueError('n must be nonnegative')
    if k > n:
        return S.Zero
    if d:
        # assert k >= d
        # kind is ignored -- only kind=2 is supported
        return _eval_stirling2(n - d + 1, k - d + 1)
    elif signed:
        # kind is ignored -- only kind=1 is supported
        return S.NegativeOne**(n - k)*_eval_stirling1(n, k)

    if kind == 1:
        return _eval_stirling1(n, k)
    elif kind == 2:
        return _eval_stirling2(n, k)
    else:
        raise ValueError('kind must be 1 or 2, not %s' % k)


@cacheit
def _nT(n, k):
    """Return the partitions of ``n`` items into ``k`` parts. This
    is used by ``nT`` for the case when ``n`` is an integer."""
    # really quick exits
    if k > n or k < 0:
        return 0
    if k in (1, n):
        return 1
    if k == 0:
        return 0
    # exits that could be done below but this is quicker
    if k == 2:
        return n//2
    d = n - k
    if d <= 3:
        return d
    # quick exit
    if 3*k >= n:  # or, equivalently, 2*k >= d
        # all the information needed in this case
        # will be in the cache needed to calculate
        # partition(d), so...
        # update cache
        tot = _partition_rec(d)
        # and correct for values not needed
        if d - k > 0:
            tot -= sum(_partition_rec.fetch_item(slice(d - k)))
        return tot
    # regular exit
    # nT(n, k) = Sum(nT(n - k, m), (m, 1, k));
    # calculate needed nT(i, j) values
    p = [1]*d
    for i in range(2, k + 1):
        for m  in range(i + 1, d):
            p[m] += p[m - i]
        d -= 1
    # if p[0] were appended to the end of p then the last
    # k values of p are the nT(n, j) values for 0 < j < k in reverse
    # order p[-1] = nT(n, 1), p[-2] = nT(n, 2), etc.... Instead of
    # putting the 1 from p[0] there, however, it is simply added to
    # the sum below which is valid for 1 < k <= n//2
    return (1 + sum(p[1 - k:]))


def nT(n, k=None):
    """Return the number of ``k``-sized partitions of ``n`` items.

    Possible values for ``n``:

        integer - ``n`` identical items

        sequence - converted to a multiset internally

        multiset - {element: multiplicity}

    Note: the convention for ``nT`` is different than that of ``nC`` and
    ``nP`` in that
    here an integer indicates ``n`` *identical* items instead of a set of
    length ``n``; this is in keeping with the ``partitions`` function which
    treats its integer-``n`` input like a list of ``n`` 1s. One can use
    ``range(n)`` for ``n`` to indicate ``n`` distinct items.

    If ``k`` is None then the total number of ways to partition the elements
    represented in ``n`` will be returned.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import nT

    Partitions of the given multiset:

    >>> [nT('aabbc', i) for i in range(1, 7)]
    [1, 8, 11, 5, 1, 0]
    >>> nT('aabbc') == sum(_)
    True

    >>> [nT("mississippi", i) for i in range(1, 12)]
    [1, 74, 609, 1521, 1768, 1224, 579, 197, 50, 9, 1]

    Partitions when all items are identical:

    >>> [nT(5, i) for i in range(1, 6)]
    [1, 2, 2, 1, 1]
    >>> nT('1'*5) == sum(_)
    True

    When all items are different:

    >>> [nT(range(5), i) for i in range(1, 6)]
    [1, 15, 25, 10, 1]
    >>> nT(range(5)) == sum(_)
    True

    Partitions of an integer expressed as a sum of positive integers:

    >>> from sympy import partition
    >>> partition(4)
    5
    >>> nT(4, 1) + nT(4, 2) + nT(4, 3) + nT(4, 4)
    5
    >>> nT('1'*4)
    5

    See Also
    ========
    sympy.utilities.iterables.partitions
    sympy.utilities.iterables.multiset_partitions
    sympy.functions.combinatorial.numbers.partition

    References
    ==========

    .. [1] https://web.archive.org/web/20210507012732/https://teaching.csse.uwa.edu.au/units/CITS7209/partition.pdf

    """

    if isinstance(n, SYMPY_INTS):
        # n identical items
        if k is None:
            return partition(n)
        if isinstance(k, SYMPY_INTS):
            n = as_int(n)
            k = as_int(k)
            return Integer(_nT(n, k))
    if not isinstance(n, _MultisetHistogram):
        try:
            # if n contains hashable items there is some
            # quick handling that can be done
            u = len(set(n))
            if u <= 1:
                return nT(len(n), k)
            elif u == len(n):
                n = range(u)
            raise TypeError
        except TypeError:
            n = _multiset_histogram(n)
    N = n[_N]
    if k is None and N == 1:
        return 1
    if k in (1, N):
        return 1
    if k == 2 or N == 2 and k is None:
        m, r = divmod(N, 2)
        rv = sum(nC(n, i) for i in range(1, m + 1))
        if not r:
            rv -= nC(n, m)//2
        if k is None:
            rv += 1  # for k == 1
        return rv
    if N == n[_ITEMS]:
        # all distinct
        if k is None:
            return bell(N)
        return stirling(N, k)
    m = MultisetPartitionTraverser()
    if k is None:
        return m.count_partitions(n[_M])
    # MultisetPartitionTraverser does not have a range-limited count
    # method, so need to enumerate and count
    tot = 0
    for discard in m.enum_range(n[_M], k-1, k):
        tot += 1
    return tot


#-----------------------------------------------------------------------------#
#                                                                             #
#                          Motzkin numbers                                    #
#                                                                             #
#-----------------------------------------------------------------------------#


class motzkin(DefinedFunction):
    """
    The nth Motzkin number is the number
    of ways of drawing non-intersecting chords
    between n points on a circle (not necessarily touching
    every point by a chord). The Motzkin numbers are named
    after Theodore Motzkin and have diverse applications
    in geometry, combinatorics and number theory.

    Motzkin numbers are the integer sequence defined by the
    initial terms `M_0 = 1`, `M_1 = 1` and the two-term recurrence relation
    `M_n = \frac{2*n + 1}{n + 2} * M_{n-1} + \frac{3n - 3}{n + 2} * M_{n-2}`.


    Examples
    ========

    >>> from sympy import motzkin

    >>> motzkin.is_motzkin(5)
    False
    >>> motzkin.find_motzkin_numbers_in_range(2,300)
    [2, 4, 9, 21, 51, 127]
    >>> motzkin.find_motzkin_numbers_in_range(2,900)
    [2, 4, 9, 21, 51, 127, 323, 835]
    >>> motzkin.find_first_n_motzkins(10)
    [1, 1, 2, 4, 9, 21, 51, 127, 323, 835]


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Motzkin_number
    .. [2] https://mathworld.wolfram.com/MotzkinNumber.html

    """

    @staticmethod
    def is_motzkin(n):
        try:
            n = as_int(n)
        except ValueError:
            return False
        if n > 0:
            if n in (1, 2):
                return True

            tn1 = 1
            tn = 2
            i = 3
            while tn < n:
                a = ((2*i + 1)*tn + (3*i - 3)*tn1)/(i + 2)
                i += 1
                tn1 = tn
                tn = a

            if tn == n:
                return True
            else:
                return False

        else:
            return False

    @staticmethod
    def find_motzkin_numbers_in_range(x, y):
        if 0 <= x <= y:
            motzkins = []
            if x <= 1 <= y:
                motzkins.append(1)
            tn1 = 1
            tn = 2
            i = 3
            while tn <= y:
                if tn >= x:
                    motzkins.append(tn)
                a = ((2*i + 1)*tn + (3*i - 3)*tn1)/(i + 2)
                i += 1
                tn1 = tn
                tn = int(a)

            return motzkins

        else:
            raise ValueError('The provided range is not valid. This condition should satisfy x <= y')

    @staticmethod
    def find_first_n_motzkins(n):
        try:
            n = as_int(n)
        except ValueError:
            raise ValueError('The provided number must be a positive integer')
        if n < 0:
            raise ValueError('The provided number must be a positive integer')
        motzkins = [1]
        if n >= 1:
            motzkins.append(1)
        tn1 = 1
        tn = 2
        i = 3
        while i <= n:
            motzkins.append(tn)
            a = ((2*i + 1)*tn + (3*i - 3)*tn1)/(i + 2)
            i += 1
            tn1 = tn
            tn = int(a)

        return motzkins

    @staticmethod
    @recurrence_memo([S.One, S.One])
    def _motzkin(n, prev):
        return ((2*n + 1)*prev[-1] + (3*n - 3)*prev[-2]) // (n + 2)

    @classmethod
    def eval(cls, n):
        try:
            n = as_int(n)
        except ValueError:
            raise ValueError('The provided number must be a positive integer')
        if n < 0:
            raise ValueError('The provided number must be a positive integer')
        return Integer(cls._motzkin(n - 1))


def nD(i=None, brute=None, *, n=None, m=None):
    """return the number of derangements for: ``n`` unique items, ``i``
    items (as a sequence or multiset), or multiplicities, ``m`` given
    as a sequence or multiset.

    Examples
    ========

    >>> from sympy.utilities.iterables import generate_derangements as enum
    >>> from sympy.functions.combinatorial.numbers import nD

    A derangement ``d`` of sequence ``s`` has all ``d[i] != s[i]``:

    >>> set([''.join(i) for i in enum('abc')])
    {'bca', 'cab'}
    >>> nD('abc')
    2

    Input as iterable or dictionary (multiset form) is accepted:

    >>> assert nD([1, 2, 2, 3, 3, 3]) == nD({1: 1, 2: 2, 3: 3})

    By default, a brute-force enumeration and count of multiset permutations
    is only done if there are fewer than 9 elements. There may be cases when
    there is high multiplicity with few unique elements that will benefit
    from a brute-force enumeration, too. For this reason, the `brute`
    keyword (default None) is provided. When False, the brute-force
    enumeration will never be used. When True, it will always be used.

    >>> nD('1111222233', brute=True)
    44

    For convenience, one may specify ``n`` distinct items using the
    ``n`` keyword:

    >>> assert nD(n=3) == nD('abc') == 2

    Since the number of derangments depends on the multiplicity of the
    elements and not the elements themselves, it may be more convenient
    to give a list or multiset of multiplicities using keyword ``m``:

    >>> assert nD('abc') == nD(m=(1,1,1)) == nD(m={1:3}) == 2

    """
    from sympy.integrals.integrals import integrate
    from sympy.functions.special.polynomials import laguerre
    from sympy.abc import x
    def ok(x):
        if not isinstance(x, SYMPY_INTS):
            raise TypeError('expecting integer values')
        if x < 0:
            raise ValueError('value must not be negative')
        return True

    if (i, n, m).count(None) != 2:
        raise ValueError('enter only 1 of i, n, or m')
    if i is not None:
        if isinstance(i, SYMPY_INTS):
            raise TypeError('items must be a list or dictionary')
        if not i:
            return S.Zero
        if type(i) is not dict:
            s = list(i)
            ms = multiset(s)
        elif type(i) is dict:
            all(ok(_) for _ in i.values())
            ms = {k: v for k, v in i.items() if v}
            s = None
        if not ms:
            return S.Zero
        N = sum(ms.values())
        counts = multiset(ms.values())
        nkey = len(ms)
    elif n is not None:
        ok(n)
        if not n:
            return S.Zero
        return subfactorial(n)
    elif m is not None:
        if isinstance(m, dict):
            all(ok(i) and ok(j) for i, j in m.items())
            counts = {k: v for k, v in m.items() if k*v}
        elif iterable(m) or isinstance(m, str):
            m = list(m)
            all(ok(i) for i in m)
            counts = multiset([i for i in m if i])
        else:
            raise TypeError('expecting iterable')
        if not counts:
            return S.Zero
        N = sum(k*v for k, v in counts.items())
        nkey = sum(counts.values())
        s = None
    big = int(max(counts))
    if big == 1:  # no repetition
        return subfactorial(nkey)
    nval = len(counts)
    if big*2 > N:
        return S.Zero
    if big*2 == N:
        if nkey == 2 and nval == 1:
            return S.One  # aaabbb
        if nkey - 1 == big:  # one element repeated
            return factorial(big)  # e.g. abc part of abcddd
    if N < 9 and brute is None or brute:
        # for all possibilities, this was found to be faster
        if s is None:
            s = []
            i = 0
            for m, v in counts.items():
                for j in range(v):
                    s.extend([i]*m)
                    i += 1
        return Integer(sum(1 for i in multiset_derangements(s)))
    from sympy.functions.elementary.exponential import exp
    return Integer(abs(integrate(exp(-x)*Mul(*[
        laguerre(i, x)**m for i, m in counts.items()]), (x, 0, oo))))
<|endoftext|>""" Elliptic Integrals. """

from sympy.core import S, pi, I, Rational
from sympy.core.function import DefinedFunction, ArgumentIndexError
from sympy.core.symbol import Dummy,uniquely_named_symbol
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.hyperbolic import atanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin, tan
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper, meijerg

class elliptic_k(DefinedFunction):
    r"""
    The complete elliptic integral of the first kind, defined by

    .. math:: K(m) = F\left(\tfrac{\pi}{2}\middle| m\right)

    where $F\left(z\middle| m\right)$ is the Legendre incomplete
    elliptic integral of the first kind.

    Explanation
    ===========

    The function $K(m)$ is a single-valued function on the complex
    plane with branch cut along the interval $(1, \infty)$.

    Note that our notation defines the incomplete elliptic integral
    in terms of the parameter $m$ instead of the elliptic modulus
    (eccentricity) $k$.
    In this case, the parameter $m$ is defined as $m=k^2$.

    Examples
    ========

    >>> from sympy import elliptic_k, I
    >>> from sympy.abc import m
    >>> elliptic_k(0)
    pi/2
    >>> elliptic_k(1.0 + I)
    1.50923695405127 + 0.625146415202697*I
    >>> elliptic_k(m).series(n=3)
    pi/2 + pi*m/8 + 9*pi*m**2/128 + O(m**3)

    See Also
    ========

    elliptic_f

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Elliptic_integrals
    .. [2] https://functions.wolfram.com/EllipticIntegrals/EllipticK

    """

    @classmethod
    def eval(cls, m):
        if m.is_zero:
            return pi*S.Half
        elif m is S.Half:
            return 8*pi**Rational(3, 2)/gamma(Rational(-1, 4))**2
        elif m is S.One:
            return S.ComplexInfinity
        elif m is S.NegativeOne:
            return gamma(Rational(1, 4))**2/(4*sqrt(2*pi))
        elif m in (S.Infinity, S.NegativeInfinity, I*S.Infinity,
                   I*S.NegativeInfinity, S.ComplexInfinity):
            return S.Zero

    def fdiff(self, argindex=1):
        m = self.args[0]
        return (elliptic_e(m) - (1 - m)*elliptic_k(m))/(2*m*(1 - m))

    def _eval_conjugate(self):
        m = self.args[0]
        if (m.is_real and (m - 1).is_positive) is False:
            return self.func(m.conjugate())

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.simplify import hyperexpand
        return hyperexpand(self.rewrite(hyper)._eval_nseries(x, n=n, logx=logx))

    def _eval_rewrite_as_hyper(self, m, **kwargs):
        return pi*S.Half*hyper((S.Half, S.Half), (S.One,), m)

    def _eval_rewrite_as_meijerg(self, m, **kwargs):
        return meijerg(((S.Half, S.Half), []), ((S.Zero,), (S.Zero,)), -m)/2

    def _eval_is_zero(self):
        m = self.args[0]
        if m.is_infinite:
            return True

    def _eval_rewrite_as_Integral(self, *args, **kwargs):
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', args).name)
        m = self.args[0]
        return Integral(1/sqrt(1 - m*sin(t)**2), (t, 0, pi/2))


class elliptic_f(DefinedFunction):
    r"""
    The Legendre incomplete elliptic integral of the first
    kind, defined by

    .. math:: F\left(z\middle| m\right) =
              \int_0^z \frac{dt}{\sqrt{1 - m \sin^2 t}}

    Explanation
    ===========

    This function reduces to a complete elliptic integral of
    the first kind, $K(m)$, when $z = \pi/2$.

    Note that our notation defines the incomplete elliptic integral
    in terms of the parameter $m$ instead of the elliptic modulus
    (eccentricity) $k$.
    In this case, the parameter $m$ is defined as $m=k^2$.

    Examples
    ========

    >>> from sympy import elliptic_f, I
    >>> from sympy.abc import z, m
    >>> elliptic_f(z, m).series(z)
    z + z**5*(3*m**2/40 - m/30) + m*z**3/6 + O(z**6)
    >>> elliptic_f(3.0 + I/2, 1.0 + I)
    2.909449841483 + 1.74720545502474*I

    See Also
    ========

    elliptic_k

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Elliptic_integrals
    .. [2] https://functions.wolfram.com/EllipticIntegrals/EllipticF

    """

    @classmethod
    def eval(cls, z, m):
        if z.is_zero:
            return S.Zero
        if m.is_zero:
            return z
        k = 2*z/pi
        if k.is_integer:
            return k*elliptic_k(m)
        elif m in (S.Infinity, S.NegativeInfinity):
            return S.Zero
        elif z.could_extract_minus_sign():
            return -elliptic_f(-z, m)

    def fdiff(self, argindex=1):
        z, m = self.args
        fm = sqrt(1 - m*sin(z)**2)
        if argindex == 1:
            return 1/fm
        elif argindex == 2:
            return (elliptic_e(z, m)/(2*m*(1 - m)) - elliptic_f(z, m)/(2*m) -
                    sin(2*z)/(4*(1 - m)*fm))
        raise ArgumentIndexError(self, argindex)

    def _eval_conjugate(self):
        z, m = self.args
        if (m.is_real and (m - 1).is_positive) is False:
            return self.func(z.conjugate(), m.conjugate())

    def _eval_rewrite_as_Integral(self, *args, **kwargs):
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', args).name)
        z, m = self.args[0], self.args[1]
        return Integral(1/(sqrt(1 - m*sin(t)**2)), (t, 0, z))

    def _eval_is_zero(self):
        z, m = self.args
        if z.is_zero:
            return True
        if m.is_extended_real and m.is_infinite:
            return True


class elliptic_e(DefinedFunction):
    r"""
    Called with two arguments $z$ and $m$, evaluates the
    incomplete elliptic integral of the second kind, defined by

    .. math:: E\left(z\middle| m\right) = \int_0^z \sqrt{1 - m \sin^2 t} dt

    Called with a single argument $m$, evaluates the Legendre complete
    elliptic integral of the second kind

    .. math:: E(m) = E\left(\tfrac{\pi}{2}\middle| m\right)

    Explanation
    ===========

    The function $E(m)$ is a single-valued function on the complex
    plane with branch cut along the interval $(1, \infty)$.

    Note that our notation defines the incomplete elliptic integral
    in terms of the parameter $m$ instead of the elliptic modulus
    (eccentricity) $k$.
    In this case, the parameter $m$ is defined as $m=k^2$.

    Examples
    ========

    >>> from sympy import elliptic_e, I
    >>> from sympy.abc import z, m
    >>> elliptic_e(z, m).series(z)
    z + z**5*(-m**2/40 + m/30) - m*z**3/6 + O(z**6)
    >>> elliptic_e(m).series(n=4)
    pi/2 - pi*m/8 - 3*pi*m**2/128 - 5*pi*m**3/512 + O(m**4)
    >>> elliptic_e(1 + I, 2 - I/2).n()
    1.55203744279187 + 0.290764986058437*I
    >>> elliptic_e(0)
    pi/2
    >>> elliptic_e(2.0 - I)
    0.991052601328069 + 0.81879421395609*I

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Elliptic_integrals
    .. [2] https://functions.wolfram.com/EllipticIntegrals/EllipticE2
    .. [3] https://functions.wolfram.com/EllipticIntegrals/EllipticE

    """

    @classmethod
    def eval(cls, m, z=None):
        if z is not None:
            z, m = m, z
            k = 2*z/pi
            if m.is_zero:
                return z
            if z.is_zero:
                return S.Zero
            elif k.is_integer:
                return k*elliptic_e(m)
            elif m in (S.Infinity, S.NegativeInfinity):
                return S.ComplexInfinity
            elif z.could_extract_minus_sign():
                return -elliptic_e(-z, m)
        else:
            if m.is_zero:
                return pi/2
            elif m is S.One:
                return S.One
            elif m is S.Infinity:
                return I*S.Infinity
            elif m is S.NegativeInfinity:
                return S.Infinity
            elif m is S.ComplexInfinity:
                return S.ComplexInfinity

    def fdiff(self, argindex=1):
        if len(self.args) == 2:
            z, m = self.args
            if argindex == 1:
                return sqrt(1 - m*sin(z)**2)
            elif argindex == 2:
                return (elliptic_e(z, m) - elliptic_f(z, m))/(2*m)
        else:
            m = self.args[0]
            if argindex == 1:
                return (elliptic_e(m) - elliptic_k(m))/(2*m)
        raise ArgumentIndexError(self, argindex)

    def _eval_conjugate(self):
        if len(self.args) == 2:
            z, m = self.args
            if (m.is_real and (m - 1).is_positive) is False:
                return self.func(z.conjugate(), m.conjugate())
        else:
            m = self.args[0]
            if (m.is_real and (m - 1).is_positive) is False:
                return self.func(m.conjugate())

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.simplify import hyperexpand
        if len(self.args) == 1:
            return hyperexpand(self.rewrite(hyper)._eval_nseries(x, n=n, logx=logx))
        return super()._eval_nseries(x, n=n, logx=logx)

    def _eval_rewrite_as_hyper(self, *args, **kwargs):
        if len(args) == 1:
            m = args[0]
            return (pi/2)*hyper((Rational(-1, 2), S.Half), (S.One,), m)

    def _eval_rewrite_as_meijerg(self, *args, **kwargs):
        if len(args) == 1:
            m = args[0]
            return -meijerg(((S.Half, Rational(3, 2)), []), \
                            ((S.Zero,), (S.Zero,)), -m)/4

    def _eval_rewrite_as_Integral(self, *args, **kwargs):
        from sympy.integrals.integrals import Integral
        z, m = (pi/2, self.args[0]) if len(self.args) == 1 else self.args
        t = Dummy(uniquely_named_symbol('t', args).name)
        return Integral(sqrt(1 - m*sin(t)**2), (t, 0, z))


class elliptic_pi(DefinedFunction):
    r"""
    Called with three arguments $n$, $z$ and $m$, evaluates the
    Legendre incomplete elliptic integral of the third kind, defined by

    .. math:: \Pi\left(n; z\middle| m\right) = \int_0^z \frac{dt}
              {\left(1 - n \sin^2 t\right) \sqrt{1 - m \sin^2 t}}

    Called with two arguments $n$ and $m$, evaluates the complete
    elliptic integral of the third kind:

    .. math:: \Pi\left(n\middle| m\right) =
              \Pi\left(n; \tfrac{\pi}{2}\middle| m\right)

    Explanation
    ===========

    Note that our notation defines the incomplete elliptic integral
    in terms of the parameter $m$ instead of the elliptic modulus
    (eccentricity) $k$.
    In this case, the parameter $m$ is defined as $m=k^2$.

    Examples
    ========

    >>> from sympy import elliptic_pi, I
    >>> from sympy.abc import z, n, m
    >>> elliptic_pi(n, z, m).series(z, n=4)
    z + z**3*(m/6 + n/3) + O(z**4)
    >>> elliptic_pi(0.5 + I, 1.0 - I, 1.2)
    2.50232379629182 - 0.760939574180767*I
    >>> elliptic_pi(0, 0)
    pi/2
    >>> elliptic_pi(1.0 - I/3, 2.0 + I)
    3.29136443417283 + 0.32555634906645*I

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Elliptic_integrals
    .. [2] https://functions.wolfram.com/EllipticIntegrals/EllipticPi3
    .. [3] https://functions.wolfram.com/EllipticIntegrals/EllipticPi

    """

    @classmethod
    def eval(cls, n, m, z=None):
        if z is not None:
            z, m = m, z
            if n.is_zero:
                return elliptic_f(z, m)
            elif n is S.One:
                return (elliptic_f(z, m) +
                        (sqrt(1 - m*sin(z)**2)*tan(z) -
                         elliptic_e(z, m))/(1 - m))
            k = 2*z/pi
            if k.is_integer:
                return k*elliptic_pi(n, m)
            elif m.is_zero:
                return atanh(sqrt(n - 1)*tan(z))/sqrt(n - 1)
            elif n == m:
                return (elliptic_f(z, n) - elliptic_pi(1, z, n) +
                        tan(z)/sqrt(1 - n*sin(z)**2))
            elif n in (S.Infinity, S.NegativeInfinity):
                return S.Zero
            elif m in (S.Infinity, S.NegativeInfinity):
                return S.Zero
            elif z.could_extract_minus_sign():
                return -elliptic_pi(n, -z, m)
            if n.is_zero:
                return elliptic_f(z, m)
            if m.is_extended_real and m.is_infinite or \
                    n.is_extended_real and n.is_infinite:
                return S.Zero
        else:
            if n.is_zero:
                return elliptic_k(m)
            elif n is S.One:
                return S.ComplexInfinity
            elif m.is_zero:
                return pi/(2*sqrt(1 - n))
            elif m == S.One:
                return S.NegativeInfinity/sign(n - 1)
            elif n == m:
                return elliptic_e(n)/(1 - n)
            elif n in (S.Infinity, S.NegativeInfinity):
                return S.Zero
            elif m in (S.Infinity, S.NegativeInfinity):
                return S.Zero
            if n.is_zero:
                return elliptic_k(m)
            if m.is_extended_real and m.is_infinite or \
                    n.is_extended_real and n.is_infinite:
                return S.Zero

    def _eval_conjugate(self):
        if len(self.args) == 3:
            n, z, m = self.args
            if (n.is_real and (n - 1).is_positive) is False and \
               (m.is_real and (m - 1).is_positive) is False:
                return self.func(n.conjugate(), z.conjugate(), m.conjugate())
        else:
            n, m = self.args
            return self.func(n.conjugate(), m.conjugate())

    def fdiff(self, argindex=1):
        if len(self.args) == 3:
            n, z, m = self.args
            fm, fn = sqrt(1 - m*sin(z)**2), 1 - n*sin(z)**2
            if argindex == 1:
                return (elliptic_e(z, m) + (m - n)*elliptic_f(z, m)/n +
                        (n**2 - m)*elliptic_pi(n, z, m)/n -
                        n*fm*sin(2*z)/(2*fn))/(2*(m - n)*(n - 1))
            elif argindex == 2:
                return 1/(fm*fn)
            elif argindex == 3:
                return (elliptic_e(z, m)/(m - 1) +
                        elliptic_pi(n, z, m) -
                        m*sin(2*z)/(2*(m - 1)*fm))/(2*(n - m))
        else:
            n, m = self.args
            if argindex == 1:
                return (elliptic_e(m) + (m - n)*elliptic_k(m)/n +
                        (n**2 - m)*elliptic_pi(n, m)/n)/(2*(m - n)*(n - 1))
            elif argindex == 2:
                return (elliptic_e(m)/(m - 1) + elliptic_pi(n, m))/(2*(n - m))
        raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Integral(self, *args, **kwargs):
        from sympy.integrals.integrals import Integral
        if len(self.args) == 2:
            n, m, z = self.args[0], self.args[1], pi/2
        else:
            n, z, m = self.args
        t = Dummy(uniquely_named_symbol('t', args).name)
        return Integral(1/((1 - n*sin(t)**2)*sqrt(1 - m*sin(t)**2)), (t, 0, z))
<|endoftext|>from sympy.core.expr import Expr
from sympy.core.function import DefinedFunction, ArgumentIndexError
from sympy.core.numbers import I, pi
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.functions import assoc_legendre
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import Abs, conjugate
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin, cos, cot

_x = Dummy("x")

class Ynm(DefinedFunction):
    r"""
    Spherical harmonics defined as

    .. math::
        Y_n^m(\theta, \varphi) := \sqrt{\frac{(2n+1)(n-m)!}{4\pi(n+m)!}}
                                  \exp(i m \varphi)
                                  \mathrm{P}_n^m\left(\cos(\theta)\right)

    Explanation
    ===========

    ``Ynm()`` gives the spherical harmonic function of order $n$ and $m$
    in $\theta$ and $\varphi$, $Y_n^m(\theta, \varphi)$. The four
    parameters are as follows: $n \geq 0$ an integer and $m$ an integer
    such that $-n \leq m \leq n$ holds. The two angles are real-valued
    with $\theta \in [0, \pi]$ and $\varphi \in [0, 2\pi]$.

    Examples
    ========

    >>> from sympy import Ynm, Symbol, simplify
    >>> from sympy.abc import n,m
    >>> theta = Symbol("theta")
    >>> phi = Symbol("phi")

    >>> Ynm(n, m, theta, phi)
    Ynm(n, m, theta, phi)

    Several symmetries are known, for the order:

    >>> Ynm(n, -m, theta, phi)
    (-1)**m*exp(-2*I*m*phi)*Ynm(n, m, theta, phi)

    As well as for the angles:

    >>> Ynm(n, m, -theta, phi)
    Ynm(n, m, theta, phi)

    >>> Ynm(n, m, theta, -phi)
    exp(-2*I*m*phi)*Ynm(n, m, theta, phi)

    For specific integers $n$ and $m$ we can evaluate the harmonics
    to more useful expressions:

    >>> simplify(Ynm(0, 0, theta, phi).expand(func=True))
    1/(2*sqrt(pi))

    >>> simplify(Ynm(1, -1, theta, phi).expand(func=True))
    sqrt(6)*exp(-I*phi)*sin(theta)/(4*sqrt(pi))

    >>> simplify(Ynm(1, 0, theta, phi).expand(func=True))
    sqrt(3)*cos(theta)/(2*sqrt(pi))

    >>> simplify(Ynm(1, 1, theta, phi).expand(func=True))
    -sqrt(6)*exp(I*phi)*sin(theta)/(4*sqrt(pi))

    >>> simplify(Ynm(2, -2, theta, phi).expand(func=True))
    sqrt(30)*exp(-2*I*phi)*sin(theta)**2/(8*sqrt(pi))

    >>> simplify(Ynm(2, -1, theta, phi).expand(func=True))
    sqrt(30)*exp(-I*phi)*sin(2*theta)/(8*sqrt(pi))

    >>> simplify(Ynm(2, 0, theta, phi).expand(func=True))
    sqrt(5)*(3*cos(theta)**2 - 1)/(4*sqrt(pi))

    >>> simplify(Ynm(2, 1, theta, phi).expand(func=True))
    -sqrt(30)*exp(I*phi)*sin(2*theta)/(8*sqrt(pi))

    >>> simplify(Ynm(2, 2, theta, phi).expand(func=True))
    sqrt(30)*exp(2*I*phi)*sin(theta)**2/(8*sqrt(pi))

    We can differentiate the functions with respect
    to both angles:

    >>> from sympy import Ynm, Symbol, diff
    >>> from sympy.abc import n,m
    >>> theta = Symbol("theta")
    >>> phi = Symbol("phi")

    >>> diff(Ynm(n, m, theta, phi), theta)
    m*cot(theta)*Ynm(n, m, theta, phi) + sqrt((-m + n)*(m + n + 1))*exp(-I*phi)*Ynm(n, m + 1, theta, phi)

    >>> diff(Ynm(n, m, theta, phi), phi)
    I*m*Ynm(n, m, theta, phi)

    Further we can compute the complex conjugation:

    >>> from sympy import Ynm, Symbol, conjugate
    >>> from sympy.abc import n,m
    >>> theta = Symbol("theta")
    >>> phi = Symbol("phi")

    >>> conjugate(Ynm(n, m, theta, phi))
    (-1)**(2*m)*exp(-2*I*m*phi)*Ynm(n, m, theta, phi)

    To get back the well known expressions in spherical
    coordinates, we use full expansion:

    >>> from sympy import Ynm, Symbol, expand_func
    >>> from sympy.abc import n,m
    >>> theta = Symbol("theta")
    >>> phi = Symbol("phi")

    >>> expand_func(Ynm(n, m, theta, phi))
    sqrt((2*n + 1)*factorial(-m + n)/factorial(m + n))*exp(I*m*phi)*assoc_legendre(n, m, cos(theta))/(2*sqrt(pi))

    See Also
    ========

    Ynm_c, Znm

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Spherical_harmonics
    .. [2] https://mathworld.wolfram.com/SphericalHarmonic.html
    .. [3] https://functions.wolfram.com/Polynomials/SphericalHarmonicY/
    .. [4] https://dlmf.nist.gov/14.30

    """

    @classmethod
    def eval(cls, n, m, theta, phi):
        # Handle negative index m and arguments theta, phi
        if m.could_extract_minus_sign():
            m = -m
            return S.NegativeOne**m * exp(-2*I*m*phi) * Ynm(n, m, theta, phi)
        if theta.could_extract_minus_sign():
            theta = -theta
            return Ynm(n, m, theta, phi)
        if phi.could_extract_minus_sign():
            phi = -phi
            return exp(-2*I*m*phi) * Ynm(n, m, theta, phi)

        # TODO Add more simplififcation here

    def _eval_expand_func(self, **hints):
        n, m, theta, phi = self.args
        rv = (sqrt((2*n + 1)/(4*pi) * factorial(n - m)/factorial(n + m)) *
                exp(I*m*phi) * assoc_legendre(n, m, cos(theta)))
        # We can do this because of the range of theta
        return rv.subs(sqrt(-cos(theta)**2 + 1), sin(theta))

    def fdiff(self, argindex=4):
        if argindex == 1:
            # Diff wrt n
            raise ArgumentIndexError(self, argindex)
        elif argindex == 2:
            # Diff wrt m
            raise ArgumentIndexError(self, argindex)
        elif argindex == 3:
            # Diff wrt theta
            n, m, theta, phi = self.args
            return (m * cot(theta) * Ynm(n, m, theta, phi) +
                    sqrt((n - m)*(n + m + 1)) * exp(-I*phi) * Ynm(n, m + 1, theta, phi))
        elif argindex == 4:
            # Diff wrt phi
            n, m, theta, phi = self.args
            return I * m * Ynm(n, m, theta, phi)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_polynomial(self, n, m, theta, phi, **kwargs):
        # TODO: Make sure n \in N
        # TODO: Assert |m| <= n ortherwise we should return 0
        return self.expand(func=True)

    def _eval_rewrite_as_sin(self, n, m, theta, phi, **kwargs):
        return self.rewrite(cos)

    def _eval_rewrite_as_cos(self, n, m, theta, phi, **kwargs):
        # This method can be expensive due to extensive use of simplification!
        from sympy.simplify import simplify, trigsimp
        # TODO: Make sure n \in N
        # TODO: Assert |m| <= n ortherwise we should return 0
        term = simplify(self.expand(func=True))
        # We can do this because of the range of theta
        term = term.xreplace({Abs(sin(theta)):sin(theta)})
        return simplify(trigsimp(term))

    def _eval_conjugate(self):
        # TODO: Make sure theta \in R and phi \in R
        n, m, theta, phi = self.args
        return S.NegativeOne**m * self.func(n, -m, theta, phi)

    def as_real_imag(self, deep=True, **hints):
        # TODO: Handle deep and hints
        n, m, theta, phi = self.args
        re = (sqrt((2*n + 1)/(4*pi) * factorial(n - m)/factorial(n + m)) *
              cos(m*phi) * assoc_legendre(n, m, cos(theta)))
        im = (sqrt((2*n + 1)/(4*pi) * factorial(n - m)/factorial(n + m)) *
              sin(m*phi) * assoc_legendre(n, m, cos(theta)))
        return (re, im)

    def _eval_evalf(self, prec):
        # Note: works without this function by just calling
        #       mpmath for Legendre polynomials. But using
        #       the dedicated function directly is cleaner.
        from mpmath import mp, workprec
        n = self.args[0]._to_mpmath(prec)
        m = self.args[1]._to_mpmath(prec)
        theta = self.args[2]._to_mpmath(prec)
        phi = self.args[3]._to_mpmath(prec)
        with workprec(prec):
            res = mp.spherharm(n, m, theta, phi)
        return Expr._from_mpmath(res, prec)


def Ynm_c(n, m, theta, phi):
    r"""
    Conjugate spherical harmonics defined as

    .. math::
        \overline{Y_n^m(\theta, \varphi)} := (-1)^m Y_n^{-m}(\theta, \varphi).

    Examples
    ========

    >>> from sympy import Ynm_c, Symbol, simplify
    >>> from sympy.abc import n,m
    >>> theta = Symbol("theta")
    >>> phi = Symbol("phi")
    >>> Ynm_c(n, m, theta, phi)
    (-1)**(2*m)*exp(-2*I*m*phi)*Ynm(n, m, theta, phi)
    >>> Ynm_c(n, m, -theta, phi)
    (-1)**(2*m)*exp(-2*I*m*phi)*Ynm(n, m, theta, phi)

    For specific integers $n$ and $m$ we can evaluate the harmonics
    to more useful expressions:

    >>> simplify(Ynm_c(0, 0, theta, phi).expand(func=True))
    1/(2*sqrt(pi))
    >>> simplify(Ynm_c(1, -1, theta, phi).expand(func=True))
    sqrt(6)*exp(I*(-phi + 2*conjugate(phi)))*sin(theta)/(4*sqrt(pi))

    See Also
    ========

    Ynm, Znm

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Spherical_harmonics
    .. [2] https://mathworld.wolfram.com/SphericalHarmonic.html
    .. [3] https://functions.wolfram.com/Polynomials/SphericalHarmonicY/

    """
    return conjugate(Ynm(n, m, theta, phi))


class Znm(DefinedFunction):
    r"""
    Real spherical harmonics defined as

    .. math::

        Z_n^m(\theta, \varphi) :=
        \begin{cases}
          \frac{Y_n^m(\theta, \varphi) + \overline{Y_n^m(\theta, \varphi)}}{\sqrt{2}} &\quad m > 0 \\
          Y_n^m(\theta, \varphi) &\quad m = 0 \\
          \frac{Y_n^m(\theta, \varphi) - \overline{Y_n^m(\theta, \varphi)}}{i \sqrt{2}} &\quad m < 0 \\
        \end{cases}

    which gives in simplified form

    .. math::

        Z_n^m(\theta, \varphi) =
        \begin{cases}
          \frac{Y_n^m(\theta, \varphi) + (-1)^m Y_n^{-m}(\theta, \varphi)}{\sqrt{2}} &\quad m > 0 \\
          Y_n^m(\theta, \varphi) &\quad m = 0 \\
          \frac{Y_n^m(\theta, \varphi) - (-1)^m Y_n^{-m}(\theta, \varphi)}{i \sqrt{2}} &\quad m < 0 \\
        \end{cases}

    Examples
    ========

    >>> from sympy import Znm, Symbol, simplify
    >>> from sympy.abc import n, m
    >>> theta = Symbol("theta")
    >>> phi = Symbol("phi")
    >>> Znm(n, m, theta, phi)
    Znm(n, m, theta, phi)

    For specific integers n and m we can evaluate the harmonics
    to more useful expressions:

    >>> simplify(Znm(0, 0, theta, phi).expand(func=True))
    1/(2*sqrt(pi))
    >>> simplify(Znm(1, 1, theta, phi).expand(func=True))
    -sqrt(3)*sin(theta)*cos(phi)/(2*sqrt(pi))
    >>> simplify(Znm(2, 1, theta, phi).expand(func=True))
    -sqrt(15)*sin(2*theta)*cos(phi)/(4*sqrt(pi))

    See Also
    ========

    Ynm, Ynm_c

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Spherical_harmonics
    .. [2] https://mathworld.wolfram.com/SphericalHarmonic.html
    .. [3] https://functions.wolfram.com/Polynomials/SphericalHarmonicY/

    """

    @classmethod
    def eval(cls, n, m, theta, phi):
        if m.is_positive:
            zz = (Ynm(n, m, theta, phi) + Ynm_c(n, m, theta, phi)) / sqrt(2)
            return zz
        elif m.is_zero:
            return Ynm(n, m, theta, phi)
        elif m.is_negative:
            zz = (Ynm(n, m, theta, phi) - Ynm_c(n, m, theta, phi)) / (sqrt(2)*I)
            return zz
<|endoftext|>from sympy.core import S, oo, diff
from sympy.core.function import DefinedFunction, ArgumentIndexError
from sympy.core.logic import fuzzy_not
from sympy.core.relational import Eq
from sympy.functions.elementary.complexes import im
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.delta_functions import Heaviside

###############################################################################
############################# SINGULARITY FUNCTION ############################
###############################################################################


class SingularityFunction(DefinedFunction):
    r"""
    Singularity functions are a class of discontinuous functions.

    Explanation
    ===========

    Singularity functions take a variable, an offset, and an exponent as
    arguments. These functions are represented using Macaulay brackets as:

    SingularityFunction(x, a, n) := <x - a>^n

    The singularity function will automatically evaluate to
    ``Derivative(DiracDelta(x - a), x, -n - 1)`` if ``n < 0``
    and ``(x - a)**n*Heaviside(x - a, 1)`` if ``n >= 0``.

    Examples
    ========

    >>> from sympy import SingularityFunction, diff, Piecewise, DiracDelta, Heaviside, Symbol
    >>> from sympy.abc import x, a, n
    >>> SingularityFunction(x, a, n)
    SingularityFunction(x, a, n)
    >>> y = Symbol('y', positive=True)
    >>> n = Symbol('n', nonnegative=True)
    >>> SingularityFunction(y, -10, n)
    (y + 10)**n
    >>> y = Symbol('y', negative=True)
    >>> SingularityFunction(y, 10, n)
    0
    >>> SingularityFunction(x, 4, -1).subs(x, 4)
    oo
    >>> SingularityFunction(x, 10, -2).subs(x, 10)
    oo
    >>> SingularityFunction(4, 1, 5)
    243
    >>> diff(SingularityFunction(x, 1, 5) + SingularityFunction(x, 1, 4), x)
    4*SingularityFunction(x, 1, 3) + 5*SingularityFunction(x, 1, 4)
    >>> diff(SingularityFunction(x, 4, 0), x, 2)
    SingularityFunction(x, 4, -2)
    >>> SingularityFunction(x, 4, 5).rewrite(Piecewise)
    Piecewise(((x - 4)**5, x >= 4), (0, True))
    >>> expr = SingularityFunction(x, a, n)
    >>> y = Symbol('y', positive=True)
    >>> n = Symbol('n', nonnegative=True)
    >>> expr.subs({x: y, a: -10, n: n})
    (y + 10)**n

    The methods ``rewrite(DiracDelta)``, ``rewrite(Heaviside)``, and
    ``rewrite('HeavisideDiracDelta')`` returns the same output. One can use any
    of these methods according to their choice.

    >>> expr = SingularityFunction(x, 4, 5) + SingularityFunction(x, -3, -1) - SingularityFunction(x, 0, -2)
    >>> expr.rewrite(Heaviside)
    (x - 4)**5*Heaviside(x - 4, 1) + DiracDelta(x + 3) - DiracDelta(x, 1)
    >>> expr.rewrite(DiracDelta)
    (x - 4)**5*Heaviside(x - 4, 1) + DiracDelta(x + 3) - DiracDelta(x, 1)
    >>> expr.rewrite('HeavisideDiracDelta')
    (x - 4)**5*Heaviside(x - 4, 1) + DiracDelta(x + 3) - DiracDelta(x, 1)

    See Also
    ========

    DiracDelta, Heaviside

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Singularity_function

    """

    is_real = True

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of a DiracDelta Function.

        Explanation
        ===========

        The difference between ``diff()`` and ``fdiff()`` is: ``diff()`` is the
        user-level function and ``fdiff()`` is an object method. ``fdiff()`` is
        a convenience method available in the ``Function`` class. It returns
        the derivative of the function without considering the chain rule.
        ``diff(function, x)`` calls ``Function._eval_derivative`` which in turn
        calls ``fdiff()`` internally to compute the derivative of the function.

        """

        if argindex == 1:
            x, a, n = self.args
            if n in (S.Zero, S.NegativeOne, S(-2), S(-3)):
                return self.func(x, a, n-1)
            elif n.is_positive:
                return n*self.func(x, a, n-1)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, variable, offset, exponent):
        """
        Returns a simplified form or a value of Singularity Function depending
        on the argument passed by the object.

        Explanation
        ===========

        The ``eval()`` method is automatically called when the
        ``SingularityFunction`` class is about to be instantiated and it
        returns either some simplified instance or the unevaluated instance
        depending on the argument passed. In other words, ``eval()`` method is
        not needed to be called explicitly, it is being called and evaluated
        once the object is called.

        Examples
        ========

        >>> from sympy import SingularityFunction, Symbol, nan
        >>> from sympy.abc import x, a, n
        >>> SingularityFunction(x, a, n)
        SingularityFunction(x, a, n)
        >>> SingularityFunction(5, 3, 2)
        4
        >>> SingularityFunction(x, a, nan)
        nan
        >>> SingularityFunction(x, 3, 0).subs(x, 3)
        1
        >>> SingularityFunction(4, 1, 5)
        243
        >>> x = Symbol('x', positive = True)
        >>> a = Symbol('a', negative = True)
        >>> n = Symbol('n', nonnegative = True)
        >>> SingularityFunction(x, a, n)
        (-a + x)**n
        >>> x = Symbol('x', negative = True)
        >>> a = Symbol('a', positive = True)
        >>> SingularityFunction(x, a, n)
        0

        """

        x = variable
        a = offset
        n = exponent
        shift = (x - a)

        if fuzzy_not(im(shift).is_zero):
            raise ValueError("Singularity Functions are defined only for Real Numbers.")
        if fuzzy_not(im(n).is_zero):
            raise ValueError("Singularity Functions are not defined for imaginary exponents.")
        if shift is S.NaN or n is S.NaN:
            return S.NaN
        if (n + 4).is_negative:
            raise ValueError("Singularity Functions are not defined for exponents less than -4.")
        if shift.is_extended_negative:
            return S.Zero
        if n.is_nonnegative:
            if shift.is_zero:  # use literal 0 in case of Symbol('z', zero=True)
                return S.Zero**n
            if shift.is_extended_nonnegative:
                return shift**n
        if n in (S.NegativeOne, -2, -3, -4):
            if shift.is_negative or shift.is_extended_positive:
                return S.Zero
            if shift.is_zero:
                return oo

    def _eval_rewrite_as_Piecewise(self, *args, **kwargs):
        '''
        Converts a Singularity Function expression into its Piecewise form.

        '''
        x, a, n = self.args

        if n in (S.NegativeOne, S(-2), S(-3), S(-4)):
            return Piecewise((oo, Eq(x - a, 0)), (0, True))
        elif n.is_nonnegative:
            return Piecewise(((x - a)**n, x - a >= 0), (0, True))

    def _eval_rewrite_as_Heaviside(self, *args, **kwargs):
        '''
        Rewrites a Singularity Function expression using Heavisides and DiracDeltas.

        '''
        x, a, n = self.args

        if n == -4:
            return diff(Heaviside(x - a), x.free_symbols.pop(), 4)
        if n == -3:
            return diff(Heaviside(x - a), x.free_symbols.pop(), 3)
        if n == -2:
            return diff(Heaviside(x - a), x.free_symbols.pop(), 2)
        if n == -1:
            return diff(Heaviside(x - a), x.free_symbols.pop(), 1)
        if n.is_nonnegative:
            return (x - a)**n*Heaviside(x - a, 1)

    def _eval_as_leading_term(self, x, logx, cdir):
        z, a, n = self.args
        shift = (z - a).subs(x, 0)
        if n < 0:
            return S.Zero
        elif n.is_zero and shift.is_zero:
            return S.Zero if cdir == -1 else S.One
        elif shift.is_positive:
            return shift**n
        return S.Zero

    def _eval_nseries(self, x, n, logx=None, cdir=0):
        z, a, n = self.args
        shift = (z - a).subs(x, 0)
        if n < 0:
            return S.Zero
        elif n.is_zero and shift.is_zero:
            return S.Zero if cdir == -1 else S.One
        elif shift.is_positive:
            return ((z - a)**n)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return S.Zero

    _eval_rewrite_as_DiracDelta = _eval_rewrite_as_Heaviside
    _eval_rewrite_as_HeavisideDiracDelta = _eval_rewrite_as_Heaviside
<|endoftext|>from sympy.core import S, diff
from sympy.core.function import DefinedFunction, ArgumentIndexError
from sympy.core.logic import fuzzy_not
from sympy.core.relational import Eq, Ne
from sympy.functions.elementary.complexes import im, sign
from sympy.functions.elementary.piecewise import Piecewise
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polyroots import roots
from sympy.utilities.misc import filldedent


###############################################################################
################################ DELTA FUNCTION ###############################
###############################################################################


class DiracDelta(DefinedFunction):
    r"""
    The DiracDelta function and its derivatives.

    Explanation
    ===========

    DiracDelta is not an ordinary function. It can be rigorously defined either
    as a distribution or as a measure.

    DiracDelta only makes sense in definite integrals, and in particular,
    integrals of the form ``Integral(f(x)*DiracDelta(x - x0), (x, a, b))``,
    where it equals ``f(x0)`` if ``a <= x0 <= b`` and ``0`` otherwise. Formally,
    DiracDelta acts in some ways like a function that is ``0`` everywhere except
    at ``0``, but in many ways it also does not. It can often be useful to treat
    DiracDelta in formal ways, building up and manipulating expressions with
    delta functions (which may eventually be integrated), but care must be taken
    to not treat it as a real function. SymPy's ``oo`` is similar. It only
    truly makes sense formally in certain contexts (such as integration limits),
    but SymPy allows its use everywhere, and it tries to be consistent with
    operations on it (like ``1/oo``), but it is easy to get into trouble and get
    wrong results if ``oo`` is treated too much like a number. Similarly, if
    DiracDelta is treated too much like a function, it is easy to get wrong or
    nonsensical results.

    DiracDelta function has the following properties:

    1) $\frac{d}{d x} \theta(x) = \delta(x)$
    2) $\int_{-\infty}^\infty \delta(x - a)f(x)\, dx = f(a)$ and $\int_{a-
       \epsilon}^{a+\epsilon} \delta(x - a)f(x)\, dx = f(a)$
    3) $\delta(x) = 0$ for all $x \neq 0$
    4) $\delta(g(x)) = \sum_i \frac{\delta(x - x_i)}{\|g'(x_i)\|}$ where $x_i$
       are the roots of $g$
    5) $\delta(-x) = \delta(x)$

    Derivatives of ``k``-th order of DiracDelta have the following properties:

    6) $\delta(x, k) = 0$ for all $x \neq 0$
    7) $\delta(-x, k) = -\delta(x, k)$ for odd $k$
    8) $\delta(-x, k) = \delta(x, k)$ for even $k$

    Examples
    ========

    >>> from sympy import DiracDelta, diff, pi
    >>> from sympy.abc import x, y

    >>> DiracDelta(x)
    DiracDelta(x)
    >>> DiracDelta(1)
    0
    >>> DiracDelta(-1)
    0
    >>> DiracDelta(pi)
    0
    >>> DiracDelta(x - 4).subs(x, 4)
    DiracDelta(0)
    >>> diff(DiracDelta(x))
    DiracDelta(x, 1)
    >>> diff(DiracDelta(x - 1), x, 2)
    DiracDelta(x - 1, 2)
    >>> diff(DiracDelta(x**2 - 1), x, 2)
    2*(2*x**2*DiracDelta(x**2 - 1, 2) + DiracDelta(x**2 - 1, 1))
    >>> DiracDelta(3*x).is_simple(x)
    True
    >>> DiracDelta(x**2).is_simple(x)
    False
    >>> DiracDelta((x**2 - 1)*y).expand(diracdelta=True, wrt=x)
    DiracDelta(x - 1)/(2*Abs(y)) + DiracDelta(x + 1)/(2*Abs(y))

    See Also
    ========

    Heaviside
    sympy.simplify.simplify.simplify, is_simple
    sympy.functions.special.tensor_functions.KroneckerDelta

    References
    ==========

    .. [1] https://mathworld.wolfram.com/DeltaFunction.html

    """

    is_real = True

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of a DiracDelta Function.

        Explanation
        ===========

        The difference between ``diff()`` and ``fdiff()`` is: ``diff()`` is the
        user-level function and ``fdiff()`` is an object method. ``fdiff()`` is
        a convenience method available in the ``Function`` class. It returns
        the derivative of the function without considering the chain rule.
        ``diff(function, x)`` calls ``Function._eval_derivative`` which in turn
        calls ``fdiff()`` internally to compute the derivative of the function.

        Examples
        ========

        >>> from sympy import DiracDelta, diff
        >>> from sympy.abc import x

        >>> DiracDelta(x).fdiff()
        DiracDelta(x, 1)

        >>> DiracDelta(x, 1).fdiff()
        DiracDelta(x, 2)

        >>> DiracDelta(x**2 - 1).fdiff()
        DiracDelta(x**2 - 1, 1)

        >>> diff(DiracDelta(x, 1)).fdiff()
        DiracDelta(x, 3)

        Parameters
        ==========

        argindex : integer
            degree of derivative

        """
        if argindex == 1:
            #I didn't know if there is a better way to handle default arguments
            k = 0
            if len(self.args) > 1:
                k = self.args[1]
            return self.func(self.args[0], k + 1)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg, k=S.Zero):
        """
        Returns a simplified form or a value of DiracDelta depending on the
        argument passed by the DiracDelta object.

        Explanation
        ===========

        The ``eval()`` method is automatically called when the ``DiracDelta``
        class is about to be instantiated and it returns either some simplified
        instance or the unevaluated instance depending on the argument passed.
        In other words, ``eval()`` method is not needed to be called explicitly,
        it is being called and evaluated once the object is called.

        Examples
        ========

        >>> from sympy import DiracDelta, S
        >>> from sympy.abc import x

        >>> DiracDelta(x)
        DiracDelta(x)

        >>> DiracDelta(-x, 1)
        -DiracDelta(x, 1)

        >>> DiracDelta(1)
        0

        >>> DiracDelta(5, 1)
        0

        >>> DiracDelta(0)
        DiracDelta(0)

        >>> DiracDelta(-1)
        0

        >>> DiracDelta(S.NaN)
        nan

        >>> DiracDelta(x - 100).subs(x, 5)
        0

        >>> DiracDelta(x - 100).subs(x, 100)
        DiracDelta(0)

        Parameters
        ==========

        k : integer
            order of derivative

        arg : argument passed to DiracDelta

        """
        if not k.is_Integer or k.is_negative:
            raise ValueError("Error: the second argument of DiracDelta must be \
            a non-negative integer, %s given instead." % (k,))
        if arg is S.NaN:
            return S.NaN
        if arg.is_nonzero:
            return S.Zero
        if fuzzy_not(im(arg).is_zero):
            raise ValueError(filldedent('''
                Function defined only for Real Values.
                Complex part: %s  found in %s .''' % (
                repr(im(arg)), repr(arg))))
        c, nc = arg.args_cnc()
        if c and c[0] is S.NegativeOne:
            # keep this fast and simple instead of using
            # could_extract_minus_sign
            if k.is_odd:
                return -cls(-arg, k)
            elif k.is_even:
                return cls(-arg, k) if k else cls(-arg)
        elif k.is_zero:
            return cls(arg, evaluate=False)

    def _eval_expand_diracdelta(self, **hints):
        """
        Compute a simplified representation of the function using
        property number 4. Pass ``wrt`` as a hint to expand the expression
        with respect to a particular variable.

        Explanation
        ===========

        ``wrt`` is:

        - a variable with respect to which a DiracDelta expression will
        get expanded.

        Examples
        ========

        >>> from sympy import DiracDelta
        >>> from sympy.abc import x, y

        >>> DiracDelta(x*y).expand(diracdelta=True, wrt=x)
        DiracDelta(x)/Abs(y)
        >>> DiracDelta(x*y).expand(diracdelta=True, wrt=y)
        DiracDelta(y)/Abs(x)

        >>> DiracDelta(x**2 + x - 2).expand(diracdelta=True, wrt=x)
        DiracDelta(x - 1)/3 + DiracDelta(x + 2)/3

        See Also
        ========

        is_simple, Diracdelta

        """
        wrt = hints.get('wrt', None)
        if wrt is None:
            free = self.free_symbols
            if len(free) == 1:
                wrt = free.pop()
            else:
                raise TypeError(filldedent('''
            When there is more than 1 free symbol or variable in the expression,
            the 'wrt' keyword is required as a hint to expand when using the
            DiracDelta hint.'''))

        if not self.args[0].has(wrt) or (len(self.args) > 1 and self.args[1] != 0 ):
            return self
        try:
            argroots = roots(self.args[0], wrt)
            result = 0
            valid = True
            darg = abs(diff(self.args[0], wrt))
            for r, m in argroots.items():
                if r.is_real is not False and m == 1:
                    result += self.func(wrt - r)/darg.subs(wrt, r)
                else:
                    # don't handle non-real and if m != 1 then
                    # a polynomial will have a zero in the derivative (darg)
                    # at r
                    valid = False
                    break
            if valid:
                return result
        except PolynomialError:
            pass
        return self

    def is_simple(self, x):
        """
        Tells whether the argument(args[0]) of DiracDelta is a linear
        expression in *x*.

        Examples
        ========

        >>> from sympy import DiracDelta, cos
        >>> from sympy.abc import x, y

        >>> DiracDelta(x*y).is_simple(x)
        True
        >>> DiracDelta(x*y).is_simple(y)
        True

        >>> DiracDelta(x**2 + x - 2).is_simple(x)
        False

        >>> DiracDelta(cos(x)).is_simple(x)
        False

        Parameters
        ==========

        x : can be a symbol

        See Also
        ========

        sympy.simplify.simplify.simplify, DiracDelta

        """
        p = self.args[0].as_poly(x)
        if p:
            return p.degree() == 1
        return False

    def _eval_rewrite_as_Piecewise(self, *args, **kwargs):
        """
        Represents DiracDelta in a piecewise form.

        Examples
        ========

        >>> from sympy import DiracDelta, Piecewise, Symbol
        >>> x = Symbol('x')

        >>> DiracDelta(x).rewrite(Piecewise)
        Piecewise((DiracDelta(0), Eq(x, 0)), (0, True))

        >>> DiracDelta(x - 5).rewrite(Piecewise)
        Piecewise((DiracDelta(0), Eq(x, 5)), (0, True))

        >>> DiracDelta(x**2 - 5).rewrite(Piecewise)
           Piecewise((DiracDelta(0), Eq(x**2, 5)), (0, True))

        >>> DiracDelta(x - 5, 4).rewrite(Piecewise)
        DiracDelta(x - 5, 4)

        """
        if len(args) == 1:
            return Piecewise((DiracDelta(0), Eq(args[0], 0)), (0, True))

    def _eval_rewrite_as_SingularityFunction(self, *args, **kwargs):
        """
        Returns the DiracDelta expression written in the form of Singularity
        Functions.

        """
        from sympy.solvers import solve
        from sympy.functions.special.singularity_functions import SingularityFunction
        if self == DiracDelta(0):
            return SingularityFunction(0, 0, -1)
        if self == DiracDelta(0, 1):
            return SingularityFunction(0, 0, -2)
        free = self.free_symbols
        if len(free) == 1:
            x = (free.pop())
            if len(args) == 1:
                return SingularityFunction(x, solve(args[0], x)[0], -1)
            return SingularityFunction(x, solve(args[0], x)[0], -args[1] - 1)
        else:
            # I don't know how to handle the case for DiracDelta expressions
            # having arguments with more than one variable.
            raise TypeError(filldedent('''
                rewrite(SingularityFunction) does not support
                arguments with more that one variable.'''))


###############################################################################
############################## HEAVISIDE FUNCTION #############################
###############################################################################


class Heaviside(DefinedFunction):
    r"""
    Heaviside step function.

    Explanation
    ===========

    The Heaviside step function has the following properties:

    1) $\frac{d}{d x} \theta(x) = \delta(x)$
    2) $\theta(x) = \begin{cases} 0 & \text{for}\: x < 0 \\ \frac{1}{2} &
       \text{for}\: x = 0 \\1 & \text{for}\: x > 0 \end{cases}$
    3) $\frac{d}{d x} \max(x, 0) = \theta(x)$

    Heaviside(x) is printed as $\theta(x)$ with the SymPy LaTeX printer.

    The value at 0 is set differently in different fields. SymPy uses 1/2,
    which is a convention from electronics and signal processing, and is
    consistent with solving improper integrals by Fourier transform and
    convolution.

    To specify a different value of Heaviside at ``x=0``, a second argument
    can be given. Using ``Heaviside(x, nan)`` gives an expression that will
    evaluate to nan for x=0.

    .. versionchanged:: 1.9 ``Heaviside(0)`` now returns 1/2 (before: undefined)

    Examples
    ========

    >>> from sympy import Heaviside, nan
    >>> from sympy.abc import x
    >>> Heaviside(9)
    1
    >>> Heaviside(-9)
    0
    >>> Heaviside(0)
    1/2
    >>> Heaviside(0, nan)
    nan
    >>> (Heaviside(x) + 1).replace(Heaviside(x), Heaviside(x, 1))
    Heaviside(x, 1) + 1

    See Also
    ========

    DiracDelta

    References
    ==========

    .. [1] https://mathworld.wolfram.com/HeavisideStepFunction.html
    .. [2] https://dlmf.nist.gov/1.16#iv

    """

    is_real = True

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of a Heaviside Function.

        Examples
        ========

        >>> from sympy import Heaviside, diff
        >>> from sympy.abc import x

        >>> Heaviside(x).fdiff()
        DiracDelta(x)

        >>> Heaviside(x**2 - 1).fdiff()
        DiracDelta(x**2 - 1)

        >>> diff(Heaviside(x)).fdiff()
        DiracDelta(x, 1)

        Parameters
        ==========

        argindex : integer
            order of derivative

        """
        if argindex == 1:
            return DiracDelta(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    def __new__(cls, arg, H0=S.Half, **options):
        if isinstance(H0, Heaviside) and len(H0.args) == 1:
            H0 = S.Half
        return super(cls, cls).__new__(cls, arg, H0, **options)

    @property
    def pargs(self):
        """Args without default S.Half"""
        args = self.args
        if args[1] is S.Half:
            args = args[:1]
        return args

    @classmethod
    def eval(cls, arg, H0=S.Half):
        """
        Returns a simplified form or a value of Heaviside depending on the
        argument passed by the Heaviside object.

        Explanation
        ===========

        The ``eval()`` method is automatically called when the ``Heaviside``
        class is about to be instantiated and it returns either some simplified
        instance or the unevaluated instance depending on the argument passed.
        In other words, ``eval()`` method is not needed to be called explicitly,
        it is being called and evaluated once the object is called.

        Examples
        ========

        >>> from sympy import Heaviside, S
        >>> from sympy.abc import x

        >>> Heaviside(x)
        Heaviside(x)

        >>> Heaviside(19)
        1

        >>> Heaviside(0)
        1/2

        >>> Heaviside(0, 1)
        1

        >>> Heaviside(-5)
        0

        >>> Heaviside(S.NaN)
        nan

        >>> Heaviside(x - 100).subs(x, 5)
        0

        >>> Heaviside(x - 100).subs(x, 105)
        1

        Parameters
        ==========

        arg : argument passed by Heaviside object

        H0 : value of Heaviside(0)

        """
        if arg.is_extended_negative:
            return S.Zero
        elif arg.is_extended_positive:
            return S.One
        elif arg.is_zero:
            return H0
        elif arg is S.NaN:
            return S.NaN
        elif fuzzy_not(im(arg).is_zero):
            raise ValueError("Function defined only for Real Values. Complex part: %s  found in %s ." % (repr(im(arg)), repr(arg)) )

    def _eval_rewrite_as_Piecewise(self, arg, H0=None, **kwargs):
        """
        Represents Heaviside in a Piecewise form.

        Examples
        ========

        >>> from sympy import Heaviside, Piecewise, Symbol, nan
        >>> x = Symbol('x')

        >>> Heaviside(x).rewrite(Piecewise)
        Piecewise((0, x < 0), (1/2, Eq(x, 0)), (1, True))

        >>> Heaviside(x,nan).rewrite(Piecewise)
        Piecewise((0, x < 0), (nan, Eq(x, 0)), (1, True))

        >>> Heaviside(x - 5).rewrite(Piecewise)
        Piecewise((0, x < 5), (1/2, Eq(x, 5)), (1, True))

        >>> Heaviside(x**2 - 1).rewrite(Piecewise)
        Piecewise((0, x**2 < 1), (1/2, Eq(x**2, 1)), (1, True))

        """
        if H0 == 0:
            return Piecewise((0, arg <= 0), (1, True))
        if H0 == 1:
            return Piecewise((0, arg < 0), (1, True))
        return Piecewise((0, arg < 0), (H0, Eq(arg, 0)), (1, True))

    def _eval_rewrite_as_sign(self, arg, H0=S.Half, **kwargs):
        """
        Represents the Heaviside function in the form of sign function.

        Explanation
        ===========

        The value of Heaviside(0) must be 1/2 for rewriting as sign to be
        strictly equivalent. For easier usage, we also allow this rewriting
