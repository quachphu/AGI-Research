        for l in expr.limits:
            if x in l[1:]:
                symb = l[0]
                break
        for term in expr.function.lseries(symb, logx):
            yield integrate(term, *expr.limits)

    def _eval_nseries(self, x, n, logx=None, cdir=0):
        symb = x
        for l in self.limits:
            if x in l[1:]:
                symb = l[0]
                break
        terms, order = self.function.nseries(
            x=symb, n=n, logx=logx).as_coeff_add(Order)
        order = [o.subs(symb, x) for o in order]
        return integrate(terms, *self.limits) + Add(*order)*x

    def _eval_as_leading_term(self, x, logx, cdir):
        series_gen = self.args[0].lseries(x)
        for leading_term in series_gen:
            if leading_term != 0:
                break
        return integrate(leading_term, *self.args[1:])

    def _eval_simplify(self, **kwargs):
        expr = factor_terms(self)
        if isinstance(expr, Integral):
            from sympy.simplify.simplify import simplify
            return expr.func(*[simplify(i, **kwargs) for i in expr.args])
        return expr.simplify(**kwargs)

    def as_sum(self, n=None, method="midpoint", evaluate=True):
        """
        Approximates a definite integral by a sum.

        Parameters
        ==========

        n :
            The number of subintervals to use, optional.
        method :
            One of: 'left', 'right', 'midpoint', 'trapezoid'.
        evaluate : bool
            If False, returns an unevaluated Sum expression. The default
            is True, evaluate the sum.

        Notes
        =====

        These methods of approximate integration are described in [1].

        Examples
        ========

        >>> from sympy import Integral, sin, sqrt
        >>> from sympy.abc import x, n
        >>> e = Integral(sin(x), (x, 3, 7))
        >>> e
        Integral(sin(x), (x, 3, 7))

        For demonstration purposes, this interval will only be split into 2
        regions, bounded by [3, 5] and [5, 7].

        The left-hand rule uses function evaluations at the left of each
        interval:

        >>> e.as_sum(2, 'left')
        2*sin(5) + 2*sin(3)

        The midpoint rule uses evaluations at the center of each interval:

        >>> e.as_sum(2, 'midpoint')
        2*sin(4) + 2*sin(6)

        The right-hand rule uses function evaluations at the right of each
        interval:

        >>> e.as_sum(2, 'right')
        2*sin(5) + 2*sin(7)

        The trapezoid rule uses function evaluations on both sides of the
        intervals. This is equivalent to taking the average of the left and
        right hand rule results:

        >>> s = e.as_sum(2, 'trapezoid')
        >>> s
        2*sin(5) + sin(3) + sin(7)
        >>> (e.as_sum(2, 'left') + e.as_sum(2, 'right'))/2 == s
        True

        Here, the discontinuity at x = 0 can be avoided by using the
        midpoint or right-hand method:

        >>> e = Integral(1/sqrt(x), (x, 0, 1))
        >>> e.as_sum(5).n(4)
        1.730
        >>> e.as_sum(10).n(4)
        1.809
        >>> e.doit().n(4)  # the actual value is 2
        2.000

        The left- or trapezoid method will encounter the discontinuity and
        return infinity:

        >>> e.as_sum(5, 'left')
        zoo

        The number of intervals can be symbolic. If omitted, a dummy symbol
        will be used for it.

        >>> e = Integral(x**2, (x, 0, 2))
        >>> e.as_sum(n, 'right').expand()
        8/3 + 4/n + 4/(3*n**2)

        This shows that the midpoint rule is more accurate, as its error
        term decays as the square of n:

        >>> e.as_sum(method='midpoint').expand()
        8/3 - 2/(3*_n**2)

        A symbolic sum is returned with evaluate=False:

        >>> e.as_sum(n, 'midpoint', evaluate=False)
        2*Sum((2*_k/n - 1/n)**2, (_k, 1, n))/n

        See Also
        ========

        Integral.doit : Perform the integration using any hints

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Riemann_sum#Riemann_summation_methods
        """

        from sympy.concrete.summations import Sum
        limits = self.limits
        if len(limits) > 1:
            raise NotImplementedError(
                "Multidimensional midpoint rule not implemented yet")
        else:
            limit = limits[0]
            if (len(limit) != 3 or limit[1].is_finite is False or
                limit[2].is_finite is False):
                raise ValueError("Expecting a definite integral over "
                                  "a finite interval.")
        if n is None:
            n = Dummy('n', integer=True, positive=True)
        else:
            n = sympify(n)
        if (n.is_positive is False or n.is_integer is False or
            n.is_finite is False):
            raise ValueError("n must be a positive integer, got %s" % n)
        x, a, b = limit
        dx = (b - a)/n
        k = Dummy('k', integer=True, positive=True)
        f = self.function

        if method == "left":
            result = dx*Sum(f.subs(x, a + (k-1)*dx), (k, 1, n))
        elif method == "right":
            result = dx*Sum(f.subs(x, a + k*dx), (k, 1, n))
        elif method == "midpoint":
            result = dx*Sum(f.subs(x, a + k*dx - dx/2), (k, 1, n))
        elif method == "trapezoid":
            result = dx*((f.subs(x, a) + f.subs(x, b))/2 +
                Sum(f.subs(x, a + k*dx), (k, 1, n - 1)))
        else:
            raise ValueError("Unknown method %s" % method)
        return result.doit() if evaluate else result

    def principal_value(self, **kwargs):
        """
        Compute the Cauchy Principal Value of the definite integral of a real function in the given interval
        on the real axis.

        Explanation
        ===========

        In mathematics, the Cauchy principal value, is a method for assigning values to certain improper
        integrals which would otherwise be undefined.

        Examples
        ========

        >>> from sympy import Integral, oo
        >>> from sympy.abc import x
        >>> Integral(x+1, (x, -oo, oo)).principal_value()
        oo
        >>> f = 1 / (x**3)
        >>> Integral(f, (x, -oo, oo)).principal_value()
        0
        >>> Integral(f, (x, -10, 10)).principal_value()
        0
        >>> Integral(f, (x, -10, oo)).principal_value() + Integral(f, (x, -oo, 10)).principal_value()
        0

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Cauchy_principal_value
        .. [2] https://mathworld.wolfram.com/CauchyPrincipalValue.html
        """
        if len(self.limits) != 1 or len(list(self.limits[0])) != 3:
            raise ValueError("You need to insert a variable, lower_limit, and upper_limit correctly to calculate "
                             "cauchy's principal value")
        x, a, b = self.limits[0]
        if not (a.is_comparable and b.is_comparable and a <= b):
            raise ValueError("The lower_limit must be smaller than or equal to the upper_limit to calculate "
                             "cauchy's principal value. Also, a and b need to be comparable.")
        if a == b:
            return S.Zero

        from sympy.calculus.singularities import singularities

        r = Dummy('r')
        f = self.function
        singularities_list = [s for s in singularities(f, x) if s.is_comparable and a <= s <= b]
        for i in singularities_list:
            if i in (a, b):
                raise ValueError(
                    'The principal value is not defined in the given interval due to singularity at %d.' % (i))
        F = integrate(f, x, **kwargs)
        if F.has(Integral):
            return self
        if a is -oo and b is oo:
            I = limit(F - F.subs(x, -x), x, oo)
        else:
            I = limit(F, x, b, '-') - limit(F, x, a, '+')
        for s in singularities_list:
            I += limit(((F.subs(x, s - r)) - F.subs(x, s + r)), r, 0, '+')
        return I



def integrate(*args, meijerg=None, conds='piecewise', risch=None, heurisch=None, manual=None, **kwargs):
    """integrate(f, var, ...)

    .. deprecated:: 1.6

       Using ``integrate()`` with :class:`~.Poly` is deprecated. Use
       :meth:`.Poly.integrate` instead. See :ref:`deprecated-integrate-poly`.

    Explanation
    ===========

    Compute definite or indefinite integral of one or more variables
    using Risch-Norman algorithm and table lookup. This procedure is
    able to handle elementary algebraic and transcendental functions
    and also a huge class of special functions, including Airy,
    Bessel, Whittaker and Lambert.

    var can be:

    - a symbol                   -- indefinite integration
    - a tuple (symbol, a)        -- indefinite integration with result
                                    given with ``a`` replacing ``symbol``
    - a tuple (symbol, a, b)     -- definite integration

    Several variables can be specified, in which case the result is
    multiple integration. (If var is omitted and the integrand is
    univariate, the indefinite integral in that variable will be performed.)

    Indefinite integrals are returned without terms that are independent
    of the integration variables. (see examples)

    Definite improper integrals often entail delicate convergence
    conditions. Pass conds='piecewise', 'separate' or 'none' to have
    these returned, respectively, as a Piecewise function, as a separate
    result (i.e. result will be a tuple), or not at all (default is
    'piecewise').

    **Strategy**

    SymPy uses various approaches to definite integration. One method is to
    find an antiderivative for the integrand, and then use the fundamental
    theorem of calculus. Various functions are implemented to integrate
    polynomial, rational and trigonometric functions, and integrands
    containing DiracDelta terms.

    SymPy also implements the part of the Risch algorithm, which is a decision
    procedure for integrating elementary functions, i.e., the algorithm can
    either find an elementary antiderivative, or prove that one does not
    exist.  There is also a (very successful, albeit somewhat slow) general
    implementation of the heuristic Risch algorithm.  This algorithm will
    eventually be phased out as more of the full Risch algorithm is
    implemented. See the docstring of Integral._eval_integral() for more
    details on computing the antiderivative using algebraic methods.

    The option risch=True can be used to use only the (full) Risch algorithm.
    This is useful if you want to know if an elementary function has an
    elementary antiderivative.  If the indefinite Integral returned by this
    function is an instance of NonElementaryIntegral, that means that the
    Risch algorithm has proven that integral to be non-elementary.  Note that
    by default, additional methods (such as the Meijer G method outlined
    below) are tried on these integrals, as they may be expressible in terms
    of special functions, so if you only care about elementary answers, use
    risch=True.  Also note that an unevaluated Integral returned by this
    function is not necessarily a NonElementaryIntegral, even with risch=True,
    as it may just be an indication that the particular part of the Risch
    algorithm needed to integrate that function is not yet implemented.

    Another family of strategies comes from re-writing the integrand in
    terms of so-called Meijer G-functions. Indefinite integrals of a
    single G-function can always be computed, and the definite integral
    of a product of two G-functions can be computed from zero to
    infinity. Various strategies are implemented to rewrite integrands
    as G-functions, and use this information to compute integrals (see
    the ``meijerint`` module).

    The option manual=True can be used to use only an algorithm that tries
    to mimic integration by hand. This algorithm does not handle as many
    integrands as the other algorithms implemented but may return results in
    a more familiar form. The ``manualintegrate`` module has functions that
    return the steps used (see the module docstring for more information).

    In general, the algebraic methods work best for computing
    antiderivatives of (possibly complicated) combinations of elementary
    functions. The G-function methods work best for computing definite
    integrals from zero to infinity of moderately complicated
    combinations of special functions, or indefinite integrals of very
    simple combinations of special functions.

    The strategy employed by the integration code is as follows:

    - If computing a definite integral, and both limits are real,
      and at least one limit is +- oo, try the G-function method of
      definite integration first.

    - Try to find an antiderivative, using all available methods, ordered
      by performance (that is try fastest method first, slowest last; in
      particular polynomial integration is tried first, Meijer
      G-functions second to last, and heuristic Risch last).

    - If still not successful, try G-functions irrespective of the
      limits.

    The option meijerg=True, False, None can be used to, respectively:
    always use G-function methods and no others, never use G-function
    methods, or use all available methods (in order as described above).
    It defaults to None.

    Examples
    ========

    >>> from sympy import integrate, log, exp, oo
    >>> from sympy.abc import a, x, y

    >>> integrate(x*y, x)
    x**2*y/2

    >>> integrate(log(x), x)
    x*log(x) - x

    >>> integrate(log(x), (x, 1, a))
    a*log(a) - a + 1

    >>> integrate(x)
    x**2/2

    Terms that are independent of x are dropped by indefinite integration:

    >>> from sympy import sqrt
    >>> integrate(sqrt(1 + x), (x, 0, x))
    2*(x + 1)**(3/2)/3 - 2/3
    >>> integrate(sqrt(1 + x), x)
    2*(x + 1)**(3/2)/3

    >>> integrate(x*y)
    Traceback (most recent call last):
    ...
    ValueError: specify integration variables to integrate x*y

    Note that ``integrate(x)`` syntax is meant only for convenience
    in interactive sessions and should be avoided in library code.

    >>> integrate(x**a*exp(-x), (x, 0, oo)) # same as conds='piecewise'
    Piecewise((gamma(a + 1), re(a) > -1),
        (Integral(x**a*exp(-x), (x, 0, oo)), True))

    >>> integrate(x**a*exp(-x), (x, 0, oo), conds='none')
    gamma(a + 1)

    >>> integrate(x**a*exp(-x), (x, 0, oo), conds='separate')
    (gamma(a + 1), re(a) > -1)

    See Also
    ========

    Integral, Integral.doit

    """
    doit_flags = {
        'deep': False,
        'meijerg': meijerg,
        'conds': conds,
        'risch': risch,
        'heurisch': heurisch,
        'manual': manual
        }

    integral = Integral(*args, **kwargs)

    if isinstance(integral, Integral):
        return integral.doit(**doit_flags)
    else:
        new_args = [a.doit(**doit_flags) if isinstance(a, Integral) else a
            for a in integral.args]
        return integral.func(*new_args)

def line_integrate(field, curve, vars):
    """line_integrate(field, Curve, variables)

    Compute the line integral.

    Examples
    ========

    >>> from sympy import Curve, line_integrate, E, ln
    >>> from sympy.abc import x, y, t
    >>> C = Curve([E**t + 1, E**t - 1], (t, 0, ln(2)))
    >>> line_integrate(x + y, C, [x, y])
    3*sqrt(2)

    See Also
    ========

    sympy.integrals.integrals.integrate, Integral
    """
    from sympy.geometry import Curve
    F = sympify(field)
    if not F:
        raise ValueError(
            "Expecting function specifying field as first argument.")
    if not isinstance(curve, Curve):
        raise ValueError("Expecting Curve entity as second argument.")
    if not is_sequence(vars):
        raise ValueError("Expecting ordered iterable for variables.")
    if len(curve.functions) != len(vars):
        raise ValueError("Field variable size does not match curve dimension.")

    if curve.parameter in vars:
        raise ValueError("Curve parameter clashes with field parameters.")

    # Calculate derivatives for line parameter functions
    # F(r) -> F(r(t)) and finally F(r(t)*r'(t))
    Ft = F
    dldt = 0
    for i, var in enumerate(vars):
        _f = curve.functions[i]
        _dn = diff(_f, curve.parameter)
        # ...arc length
        dldt = dldt + (_dn * _dn)
        Ft = Ft.subs(var, _f)
    Ft = Ft * sqrt(dldt)

    integral = Integral(Ft, curve.limits).doit(deep=False)
    return integral


### Property function dispatching ###

@shape.register(Integral)
def _(expr):
    return shape(expr.function)

# Delayed imports
from .deltafunctions import deltaintegrate
from .meijerint import meijerint_definite, meijerint_indefinite, _debug
from .trigonometry import trigintegrate
<|endoftext|>"""This module implements tools for integrating rational functions. """

from sympy.core.function import Lambda
from sympy.core.numbers import I
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import atan
from sympy.polys.polyerrors import DomainError
from sympy.polys.polyroots import roots
from sympy.polys.polytools import cancel
from sympy.polys.rootoftools import RootSum
from sympy.polys import Poly, resultant, ZZ


def ratint(f, x, **flags):
    """
    Performs indefinite integration of rational functions.

    Explanation
    ===========

    Given a field :math:`K` and a rational function :math:`f = p/q`,
    where :math:`p` and :math:`q` are polynomials in :math:`K[x]`,
    returns a function :math:`g` such that :math:`f = g'`.

    Examples
    ========

    >>> from sympy.integrals.rationaltools import ratint
    >>> from sympy.abc import x

    >>> ratint(36/(x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2), x)
    (12*x + 6)/(x**2 - 1) + 4*log(x - 2) - 4*log(x + 1)

    References
    ==========

    .. [1] M. Bronstein, Symbolic Integration I: Transcendental
       Functions, Second Edition, Springer-Verlag, 2005, pp. 35-70

    See Also
    ========

    sympy.integrals.integrals.Integral.doit
    sympy.integrals.rationaltools.ratint_logpart
    sympy.integrals.rationaltools.ratint_ratpart

    """
    if isinstance(f, tuple):
        p, q = f
    else:
        p, q = f.as_numer_denom()

    p, q = Poly(p, x, composite=False, field=True), Poly(q, x, composite=False, field=True)

    coeff, p, q = p.cancel(q)
    poly, p = p.div(q)

    result = poly.integrate(x).as_expr()

    if p.is_zero:
        return coeff*result

    g, h = ratint_ratpart(p, q, x)

    P, Q = h.as_numer_denom()

    P = Poly(P, x)
    Q = Poly(Q, x)

    q, r = P.div(Q)

    result += g + q.integrate(x).as_expr()

    if not r.is_zero:
        symbol = flags.get('symbol', 't')

        if not isinstance(symbol, Symbol):
            t = Dummy(symbol)
        else:
            t = symbol.as_dummy()

        L = ratint_logpart(r, Q, x, t)

        real = flags.get('real')

        if real is None:
            if isinstance(f, tuple):
                p, q = f
                atoms = p.atoms() | q.atoms()
            else:
                atoms = f.atoms()

            for elt in atoms - {x}:
                if not elt.is_extended_real:
                    real = False
                    break
            else:
                real = True

        eps = S.Zero

        if not real:
            for h, q in L:
                _, h = h.primitive()
                eps += RootSum(
                    q, Lambda(t, t*log(h.as_expr())), quadratic=True)
        else:
            for h, q in L:
                _, h = h.primitive()
                R = log_to_real(h, q, x, t)

                if R is not None:
                    eps += R
                else:
                    eps += RootSum(
                        q, Lambda(t, t*log(h.as_expr())), quadratic=True)

        result += eps

    return coeff*result


def ratint_ratpart(f, g, x):
    """
    Horowitz-Ostrogradsky algorithm.

    Explanation
    ===========

    Given a field K and polynomials f and g in K[x], such that f and g
    are coprime and deg(f) < deg(g), returns fractions A and B in K(x),
    such that f/g = A' + B and B has square-free denominator.

    Examples
    ========

        >>> from sympy.integrals.rationaltools import ratint_ratpart
        >>> from sympy.abc import x, y
        >>> from sympy import Poly
        >>> ratint_ratpart(Poly(1, x, domain='ZZ'),
        ... Poly(x + 1, x, domain='ZZ'), x)
        (0, 1/(x + 1))
        >>> ratint_ratpart(Poly(1, x, domain='EX'),
        ... Poly(x**2 + y**2, x, domain='EX'), x)
        (0, 1/(x**2 + y**2))
        >>> ratint_ratpart(Poly(36, x, domain='ZZ'),
        ... Poly(x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2, x, domain='ZZ'), x)
        ((12*x + 6)/(x**2 - 1), 12/(x**2 - x - 2))

    See Also
    ========

    ratint, ratint_logpart
    """
    from sympy.solvers.solvers import solve

    f = Poly(f, x)
    g = Poly(g, x)

    u, v, _ = g.cofactors(g.diff())

    n = u.degree()
    m = v.degree()

    A_coeffs = [ Dummy('a' + str(n - i)) for i in range(0, n) ]
    B_coeffs = [ Dummy('b' + str(m - i)) for i in range(0, m) ]

    C_coeffs = A_coeffs + B_coeffs

    A = Poly(A_coeffs, x, domain=ZZ[C_coeffs])
    B = Poly(B_coeffs, x, domain=ZZ[C_coeffs])

    H = f - A.diff()*v + A*(u.diff()*v).quo(u) - B*u

    result = solve(H.coeffs(), C_coeffs)

    A = A.as_expr().subs(result)
    B = B.as_expr().subs(result)

    rat_part = cancel(A/u.as_expr(), x)
    log_part = cancel(B/v.as_expr(), x)

    return rat_part, log_part


def ratint_logpart(f, g, x, t=None):
    r"""
    Lazard-Rioboo-Trager algorithm.

    Explanation
    ===========

    Given a field K and polynomials f and g in K[x], such that f and g
    are coprime, deg(f) < deg(g) and g is square-free, returns a list
    of tuples (s_i, q_i) of polynomials, for i = 1..n, such that s_i
    in K[t, x] and q_i in K[t], and::

                           ___    ___
                 d  f   d  \  `   \  `
                 -- - = --  )      )   a log(s_i(a, x))
                 dx g   dx /__,   /__,
                          i=1..n a | q_i(a) = 0

    Examples
    ========

    >>> from sympy.integrals.rationaltools import ratint_logpart
    >>> from sympy.abc import x
    >>> from sympy import Poly
    >>> ratint_logpart(Poly(1, x, domain='ZZ'),
    ... Poly(x**2 + x + 1, x, domain='ZZ'), x)
    [(Poly(x + 3*_t/2 + 1/2, x, domain='QQ[_t]'),
    ...Poly(3*_t**2 + 1, _t, domain='ZZ'))]
    >>> ratint_logpart(Poly(12, x, domain='ZZ'),
    ... Poly(x**2 - x - 2, x, domain='ZZ'), x)
    [(Poly(x - 3*_t/8 - 1/2, x, domain='QQ[_t]'),
    ...Poly(-_t**2 + 16, _t, domain='ZZ'))]

    See Also
    ========

    ratint, ratint_ratpart
    """
    f, g = Poly(f, x), Poly(g, x)

    t = t or Dummy('t')
    a, b = g, f - g.diff()*Poly(t, x)

    res, R = resultant(a, b, includePRS=True)
    res = Poly(res, t, composite=False)

    assert res, "BUG: resultant(%s, %s) cannot be zero" % (a, b)

    R_map, H = {}, []

    for r in R:
        R_map[r.degree()] = r

    def _include_sign(c, sqf):
        if c.is_extended_real and (c < 0) == True:
            h, k = sqf[0]
            c_poly = c.as_poly(h.gens)
            sqf[0] = h*c_poly, k

    C, res_sqf = res.sqf_list()
    _include_sign(C, res_sqf)

    for q, i in res_sqf:
        _, q = q.primitive()

        if g.degree() == i:
            H.append((g, q))
        else:
            h = R_map[i]
            h_lc = Poly(h.LC(), t, field=True)

            c, h_lc_sqf = h_lc.sqf_list(all=True)
            _include_sign(c, h_lc_sqf)

            for a, j in h_lc_sqf:
                h = h.quo(Poly(a.gcd(q)**j, x))

            inv, coeffs = h_lc.invert(q), [S.One]

            for coeff in h.coeffs()[1:]:
                coeff = coeff.as_poly(inv.gens)
                T = (inv*coeff).rem(q)
                coeffs.append(T.as_expr())

            h = Poly(dict(list(zip(h.monoms(), coeffs))), x)

            H.append((h, q))

    return H


def log_to_atan(f, g):
    """
    Convert complex logarithms to real arctangents.

    Explanation
    ===========

    Given a real field K and polynomials f and g in K[x], with g != 0,
    returns a sum h of arctangents of polynomials in K[x], such that:

                   dh   d         f + I g
                   -- = -- I log( ------- )
                   dx   dx        f - I g

    Examples
    ========

        >>> from sympy.integrals.rationaltools import log_to_atan
        >>> from sympy.abc import x
        >>> from sympy import Poly, sqrt, S
        >>> log_to_atan(Poly(x, x, domain='ZZ'), Poly(1, x, domain='ZZ'))
        2*atan(x)
        >>> log_to_atan(Poly(x + S(1)/2, x, domain='QQ'),
        ... Poly(sqrt(3)/2, x, domain='EX'))
        2*atan(2*sqrt(3)*x/3 + sqrt(3)/3)

    See Also
    ========

    log_to_real
    """
    if f.degree() < g.degree():
        f, g = -g, f

    f = f.to_field()
    g = g.to_field()

    p, q = f.div(g)

    if q.is_zero:
        return 2*atan(p.as_expr())
    else:
        s, t, h = g.gcdex(-f)
        u = (f*s + g*t).quo(h)
        A = 2*atan(u.as_expr())

        return A + log_to_atan(s, t)


def _get_real_roots(f, x):
    """get real roots of f if possible"""
    rs = roots(f, filter='R')

    try:
        num_roots = f.count_roots()
    except DomainError:
        return rs
    else:
        if len(rs) == num_roots:
            return rs
        else:
            return None


def log_to_real(h, q, x, t):
    r"""
    Convert complex logarithms to real functions.

    Explanation
    ===========

    Given real field K and polynomials h in K[t,x] and q in K[t],
    returns real function f such that:
                          ___
                  df   d  \  `
                  -- = --  )  a log(h(a, x))
                  dx   dx /__,
                         a | q(a) = 0

    Examples
    ========

        >>> from sympy.integrals.rationaltools import log_to_real
        >>> from sympy.abc import x, y
        >>> from sympy import Poly, S
        >>> log_to_real(Poly(x + 3*y/2 + S(1)/2, x, domain='QQ[y]'),
        ... Poly(3*y**2 + 1, y, domain='ZZ'), x, y)
        2*sqrt(3)*atan(2*sqrt(3)*x/3 + sqrt(3)/3)/3
        >>> log_to_real(Poly(x**2 - 1, x, domain='ZZ'),
        ... Poly(-2*y + 1, y, domain='ZZ'), x, y)
        log(x**2 - 1)/2

    See Also
    ========

    log_to_atan
    """
    from sympy.simplify.radsimp import collect
    u, v = symbols('u,v', cls=Dummy)

    H = h.as_expr().xreplace({t: u + I*v}).expand()
    Q = q.as_expr().xreplace({t: u + I*v}).expand()

    H_map = collect(H, I, evaluate=False)
    Q_map = collect(Q, I, evaluate=False)

    a, b = H_map.get(S.One, S.Zero), H_map.get(I, S.Zero)
    c, d = Q_map.get(S.One, S.Zero), Q_map.get(I, S.Zero)

    R = Poly(resultant(c, d, v), u)

    R_u = _get_real_roots(R, u)

    if R_u is None:
        return None

    result = S.Zero

    for r_u in R_u.keys():
        C = Poly(c.xreplace({u: r_u}), v)
        if not C:
            # t was split into real and imaginary parts
            # and denom Q(u, v) = c + I*d. We just found
            # that c(r_u) is 0 so the roots are in d
            C = Poly(d.xreplace({u: r_u}), v)
            # we were going to reject roots from C that
            # did not set d to zero, but since we are now
            # using C = d and c is already 0, there is
            # nothing to check
            d = S.Zero

        R_v = _get_real_roots(C, v)

        if R_v is None:
            return None

        R_v_paired = [] # take one from each pair of conjugate roots
        for r_v in R_v:
            if r_v not in R_v_paired and -r_v not in R_v_paired:
                if r_v.is_negative or r_v.could_extract_minus_sign():
                    R_v_paired.append(-r_v)
                elif not r_v.is_zero:
                    R_v_paired.append(r_v)

        for r_v in R_v_paired:

            D = d.xreplace({u: r_u, v: r_v})

            if D.evalf(chop=True) != 0:
                continue

            A = Poly(a.xreplace({u: r_u, v: r_v}), x)
            B = Poly(b.xreplace({u: r_u, v: r_v}), x)

            AB = (A**2 + B**2).as_expr()

            result += r_u*log(AB) + r_v*log_to_atan(A, B)

    R_q = _get_real_roots(q, t)

    if R_q is None:
        return None

    for r in R_q.keys():
        result += r*log(h.as_expr().subs(t, r))

    return result
<|endoftext|>"""
The Risch Algorithm for transcendental function integration.

The core algorithms for the Risch algorithm are here.  The subproblem
algorithms are in the rde.py and prde.py files for the Risch
Differential Equation solver and the parametric problems solvers,
respectively.  All important information concerning the differential extension
for an integrand is stored in a DifferentialExtension object, which in the code
is usually called DE.  Throughout the code and Inside the DifferentialExtension
object, the conventions/attribute names are that the base domain is QQ and each
differential extension is x, t0, t1, ..., tn-1 = DE.t. DE.x is the variable of
integration (Dx == 1), DE.D is a list of the derivatives of
x, t1, t2, ..., tn-1 = t, DE.T is the list [x, t1, t2, ..., tn-1], DE.t is the
outer-most variable of the differential extension at the given level (the level
can be adjusted using DE.increment_level() and DE.decrement_level()),
k is the field C(x, t0, ..., tn-2), where C is the constant field.  The
numerator of a fraction is denoted by a and the denominator by
d.  If the fraction is named f, fa == numer(f) and fd == denom(f).
Fractions are returned as tuples (fa, fd).  DE.d and DE.t are used to
represent the topmost derivation and extension variable, respectively.
The docstring of a function signifies whether an argument is in k[t], in
which case it will just return a Poly in t, or in k(t), in which case it
will return the fraction (fa, fd). Other variable names probably come
from the names used in Bronstein's book.
"""
from types import GeneratorType
from functools import reduce

from sympy.core.function import Lambda
from sympy.core.mul import Mul
from sympy.core.intfunc import ilcm
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.exponential import log, exp
from sympy.functions.elementary.hyperbolic import (cosh, coth, sinh,
    tanh)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (atan, sin, cos,
    tan, acot, cot, asin, acos)
from .integrals import integrate, Integral
from .heurisch import _symbols
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import (real_roots, cancel, Poly, gcd,
    reduced)
from sympy.polys.rootoftools import RootSum
from sympy.utilities.iterables import numbered_symbols


def integer_powers(exprs):
    """
    Rewrites a list of expressions as integer multiples of each other.

    Explanation
    ===========

    For example, if you have [x, x/2, x**2 + 1, 2*x/3], then you can rewrite
    this as [(x/6) * 6, (x/6) * 3, (x**2 + 1) * 1, (x/6) * 4]. This is useful
    in the Risch integration algorithm, where we must write exp(x) + exp(x/2)
    as (exp(x/2))**2 + exp(x/2), but not as exp(x) + sqrt(exp(x)) (this is
    because only the transcendental case is implemented and we therefore cannot
    integrate algebraic extensions). The integer multiples returned by this
    function for each term are the smallest possible (their content equals 1).

    Returns a list of tuples where the first element is the base term and the
    second element is a list of `(item, factor)` terms, where `factor` is the
    integer multiplicative factor that must multiply the base term to obtain
    the original item.

    The easiest way to understand this is to look at an example:

    >>> from sympy.abc import x
    >>> from sympy.integrals.risch import integer_powers
    >>> integer_powers([x, x/2, x**2 + 1, 2*x/3])
    [(x/6, [(x, 6), (x/2, 3), (2*x/3, 4)]), (x**2 + 1, [(x**2 + 1, 1)])]

    We can see how this relates to the example at the beginning of the
    docstring.  It chose x/6 as the first base term.  Then, x can be written as
    (x/2) * 2, so we get (0, 2), and so on. Now only element (x**2 + 1)
    remains, and there are no other terms that can be written as a rational
    multiple of that, so we get that it can be written as (x**2 + 1) * 1.

    """
    # Here is the strategy:

    # First, go through each term and determine if it can be rewritten as a
    # rational multiple of any of the terms gathered so far.
    # cancel(a/b).is_Rational is sufficient for this.  If it is a multiple, we
    # add its multiple to the dictionary.

    terms = {}
    for term in exprs:
        for trm, trm_list in terms.items():
            a = cancel(term/trm)
            if a.is_Rational:
                trm_list.append((term, a))
                break
        else:
            terms[term] = [(term, S.One)]

    # After we have done this, we have all the like terms together, so we just
    # need to find a common denominator so that we can get the base term and
    # integer multiples such that each term can be written as an integer
    # multiple of the base term, and the content of the integers is 1.

    newterms = {}
    for term, term_list in terms.items():
        common_denom = reduce(ilcm, [i.as_numer_denom()[1] for _, i in
            term_list])
        newterm = term/common_denom
        newmults = [(i, j*common_denom) for i, j in term_list]
        newterms[newterm] = newmults

    return sorted(iter(newterms.items()), key=lambda item: item[0].sort_key())


class DifferentialExtension:
    """
    A container for all the information relating to a differential extension.

    Explanation
    ===========

    The attributes of this object are (see also the docstring of __init__):

    - f: The original (Expr) integrand.
    - x: The variable of integration.
    - T: List of variables in the extension.
    - D: List of derivations in the extension; corresponds to the elements of T.
    - fa: Poly of the numerator of the integrand.
    - fd: Poly of the denominator of the integrand.
    - Tfuncs: Lambda() representations of each element of T (except for x).
      For back-substitution after integration.
    - backsubs: A (possibly empty) list of further substitutions to be made on
      the final integral to make it look more like the integrand.
    - exts:
    - extargs:
    - cases: List of string representations of the cases of T.
    - t: The top level extension variable, as defined by the current level
      (see level below).
    - d: The top level extension derivation, as defined by the current
      derivation (see level below).
    - case: The string representation of the case of self.d.
    (Note that self.T and self.D will always contain the complete extension,
    regardless of the level.  Therefore, you should ALWAYS use DE.t and DE.d
    instead of DE.T[-1] and DE.D[-1].  If you want to have a list of the
    derivations or variables only up to the current level, use
    DE.D[:len(DE.D) + DE.level + 1] and DE.T[:len(DE.T) + DE.level + 1].  Note
    that, in particular, the derivation() function does this.)

    The following are also attributes, but will probably not be useful other
    than in internal use:
    - newf: Expr form of fa/fd.
    - level: The number (between -1 and -len(self.T)) such that
      self.T[self.level] == self.t and self.D[self.level] == self.d.
      Use the methods self.increment_level() and self.decrement_level() to change
      the current level.
    """
    # __slots__ is defined mainly so we can iterate over all the attributes
    # of the class easily (the memory use doesn't matter too much, since we
    # only create one DifferentialExtension per integration).  Also, it's nice
    # to have a safeguard when debugging.
    __slots__ = ('f', 'x', 'T', 'D', 'fa', 'fd', 'Tfuncs', 'backsubs',
        'exts', 'extargs', 'cases', 'case', 't', 'd', 'newf', 'level',
        'ts', 'dummy')

    def __init__(self, f=None, x=None, handle_first='log', dummy=False, extension=None, rewrite_complex=None):
        """
        Tries to build a transcendental extension tower from ``f`` with respect to ``x``.

        Explanation
        ===========

        If it is successful, creates a DifferentialExtension object with, among
        others, the attributes fa, fd, D, T, Tfuncs, and backsubs such that
        fa and fd are Polys in T[-1] with rational coefficients in T[:-1],
        fa/fd == f, and D[i] is a Poly in T[i] with rational coefficients in
        T[:i] representing the derivative of T[i] for each i from 1 to len(T).
        Tfuncs is a list of Lambda objects for back replacing the functions
        after integrating.  Lambda() is only used (instead of lambda) to make
        them easier to test and debug. Note that Tfuncs corresponds to the
        elements of T, except for T[0] == x, but they should be back-substituted
        in reverse order.  backsubs is a (possibly empty) back-substitution list
        that should be applied on the completed integral to make it look more
        like the original integrand.

        If it is unsuccessful, it raises NotImplementedError.

        You can also create an object by manually setting the attributes as a
        dictionary to the extension keyword argument.  You must include at least
        D.  Warning, any attribute that is not given will be set to None. The
        attributes T, t, d, cases, case, x, and level are set automatically and
        do not need to be given.  The functions in the Risch Algorithm will NOT
        check to see if an attribute is None before using it.  This also does not
        check to see if the extension is valid (non-algebraic) or even if it is
        self-consistent.  Therefore, this should only be used for
        testing/debugging purposes.
        """
        # XXX: If you need to debug this function, set the break point here

        if extension:
            if 'D' not in extension:
                raise ValueError("At least the key D must be included with "
                    "the extension flag to DifferentialExtension.")
            for attr in extension:
                setattr(self, attr, extension[attr])

            self._auto_attrs()

            return
        elif f is None or x is None:
            raise ValueError("Either both f and x or a manual extension must "
            "be given.")

        if handle_first not in ('log', 'exp'):
            raise ValueError("handle_first must be 'log' or 'exp', not %s." %
                str(handle_first))

        # f will be the original function, self.f might change if we reset
        # (e.g., we pull out a constant from an exponential)
        self.f = f
        self.x = x
        # setting the default value 'dummy'
        self.dummy = dummy
        self.reset()
        exp_new_extension, log_new_extension = True, True

        # case of 'automatic' choosing
        if rewrite_complex is None:
            rewrite_complex = I in self.f.atoms()

        if rewrite_complex:
            rewritables = {
                (sin, cos, cot, tan, sinh, cosh, coth, tanh): exp,
                (asin, acos, acot, atan): log,
            }
            # rewrite the trigonometric components
            for candidates, rule in rewritables.items():
                self.newf = self.newf.rewrite(candidates, rule)
            self.newf = cancel(self.newf)
        else:
            if any(i.has(x) for i in self.f.atoms(sin, cos, tan, atan, asin, acos)):
                raise NotImplementedError("Trigonometric extensions are not "
                "supported (yet!)")

        exps = set()
        pows = set()
        numpows = set()
        sympows = set()
        logs = set()
        symlogs = set()

        while True:
            if self.newf.is_rational_function(*self.T):
                break

            if not exp_new_extension and not log_new_extension:
                # We couldn't find a new extension on the last pass, so I guess
                # we can't do it.
                raise NotImplementedError("Couldn't find an elementary "
                    "transcendental extension for %s.  Try using a " % str(f) +
                    "manual extension with the extension flag.")

            exps, pows, numpows, sympows, log_new_extension = \
                    self._rewrite_exps_pows(exps, pows, numpows, sympows, log_new_extension)

            logs, symlogs = self._rewrite_logs(logs, symlogs)

            if handle_first == 'exp' or not log_new_extension:
                exp_new_extension = self._exp_part(exps)
                if exp_new_extension is None:
                    # reset and restart
                    self.f = self.newf
                    self.reset()
                    exp_new_extension = True
                    continue

            if handle_first == 'log' or not exp_new_extension:
                log_new_extension = self._log_part(logs)

        self.fa, self.fd = frac_in(self.newf, self.t)
        self._auto_attrs()

        return

    def __getattr__(self, attr):
        # Avoid AttributeErrors when debugging
        if attr not in self.__slots__:
            raise AttributeError("%s has no attribute %s" % (repr(self), repr(attr)))
        return None

    def _rewrite_exps_pows(self, exps, pows, numpows,
            sympows, log_new_extension):
        """
        Rewrite exps/pows for better processing.
        """
        from .prde import is_deriv_k

        # Pre-preparsing.
        #################
        # Get all exp arguments, so we can avoid ahead of time doing
        # something like t1 = exp(x), t2 = exp(x/2) == sqrt(t1).

        # Things like sqrt(exp(x)) do not automatically simplify to
        # exp(x/2), so they will be viewed as algebraic.  The easiest way
        # to handle this is to convert all instances of exp(a)**Rational
        # to exp(Rational*a) before doing anything else.  Note that the
        # _exp_part code can generate terms of this form, so we do need to
        # do this at each pass (or else modify it to not do that).

        ratpows = [i for i in self.newf.atoms(Pow)
                   if (isinstance(i.base, exp) and i.exp.is_Rational)]

        ratpows_repl = [
            (i, i.base.base**(i.exp*i.base.exp)) for i in ratpows]
        self.backsubs += [(j, i) for i, j in ratpows_repl]
        self.newf = self.newf.xreplace(dict(ratpows_repl))

        # To make the process deterministic, the args are sorted
        # so that functions with smaller op-counts are processed first.
        # Ties are broken with the default_sort_key.

        # XXX Although the method is deterministic no additional work
        # has been done to guarantee that the simplest solution is
        # returned and that it would be affected be using different
        # variables. Though it is possible that this is the case
        # one should know that it has not been done intentionally, so
        # further improvements may be possible.

        # TODO: This probably doesn't need to be completely recomputed at
        # each pass.
        exps = update_sets(exps, self.newf.atoms(exp),
            lambda i: i.exp.is_rational_function(*self.T) and
            i.exp.has(*self.T))
        pows = update_sets(pows, self.newf.atoms(Pow),
            lambda i: i.exp.is_rational_function(*self.T) and
            i.exp.has(*self.T))
        numpows = update_sets(numpows, set(pows),
            lambda i: not i.base.has(*self.T))
        sympows = update_sets(sympows, set(pows) - set(numpows),
            lambda i: i.base.is_rational_function(*self.T) and
            not i.exp.is_Integer)

        # The easiest way to deal with non-base E powers is to convert them
        # into base E, integrate, and then convert back.
        for i in ordered(pows):
            old = i
            new = exp(i.exp*log(i.base))
            # If exp is ever changed to automatically reduce exp(x*log(2))
            # to 2**x, then this will break.  The solution is to not change
            # exp to do that :)
            if i in sympows:
                if i.exp.is_Rational:
                    raise NotImplementedError("Algebraic extensions are "
                        "not supported (%s)." % str(i))
                # We can add a**b only if log(a) in the extension, because
                # a**b == exp(b*log(a)).
                basea, based = frac_in(i.base, self.t)
                A = is_deriv_k(basea, based, self)
                if A is None:
                    # Nonelementary monomial (so far)

                    # TODO: Would there ever be any benefit from just
                    # adding log(base) as a new monomial?
                    # ANSWER: Yes, otherwise we can't integrate x**x (or
                    # rather prove that it has no elementary integral)
                    # without first manually rewriting it as exp(x*log(x))
                    self.newf = self.newf.xreplace({old: new})
                    self.backsubs += [(new, old)]
                    log_new_extension = self._log_part([log(i.base)])
                    exps = update_sets(exps, self.newf.atoms(exp), lambda i:
                        i.exp.is_rational_function(*self.T) and i.exp.has(*self.T))
                    continue
                ans, u, const = A
                newterm = exp(i.exp*(log(const) + u))
                # Under the current implementation, exp kills terms
                # only if they are of the form a*log(x), where a is a
                # Number.  This case should have already been killed by the
                # above tests.  Again, if this changes to kill more than
                # that, this will break, which maybe is a sign that you
                # shouldn't be changing that.  Actually, if anything, this
                # auto-simplification should be removed.  See
                # https://groups.google.com/group/sympy/browse_thread/thread/a61d48235f16867f

                self.newf = self.newf.xreplace({i: newterm})

            elif i not in numpows:
                continue
            else:
                # i in numpows
                newterm = new
            # TODO: Just put it in self.Tfuncs
            self.backsubs.append((new, old))
            self.newf = self.newf.xreplace({old: newterm})
            exps.append(newterm)

        return exps, pows, numpows, sympows, log_new_extension

    def _rewrite_logs(self, logs, symlogs):
        """
        Rewrite logs for better processing.
        """
        atoms = self.newf.atoms(log)
        logs = update_sets(logs, atoms,
            lambda i: i.args[0].is_rational_function(*self.T) and
            i.args[0].has(*self.T))
        symlogs = update_sets(symlogs, atoms,
            lambda i: i.has(*self.T) and i.args[0].is_Pow and
            i.args[0].base.is_rational_function(*self.T) and
            not i.args[0].exp.is_Integer)

        # We can handle things like log(x**y) by converting it to y*log(x)
        # This will fix not only symbolic exponents of the argument, but any
        # non-Integer exponent, like log(sqrt(x)).  The exponent can also
        # depend on x, like log(x**x).
        for i in ordered(symlogs):
            # Unlike in the exponential case above, we do not ever
            # potentially add new monomials (above we had to add log(a)).
            # Therefore, there is no need to run any is_deriv functions
            # here.  Just convert log(a**b) to b*log(a) and let
            # log_new_extension() handle it from there.
            lbase = log(i.args[0].base)
            logs.append(lbase)
            new = i.args[0].exp*lbase
            self.newf = self.newf.xreplace({i: new})
            self.backsubs.append((new, i))

        # remove any duplicates
        logs = sorted(set(logs), key=default_sort_key)

        return logs, symlogs

    def _auto_attrs(self):
        """
        Set attributes that are generated automatically.
        """
        if not self.T:
            # i.e., when using the extension flag and T isn't given
            self.T = [i.gen for i in self.D]
        if not self.x:
            self.x = self.T[0]
        self.cases = [get_case(d, t) for d, t in zip(self.D, self.T)]
        self.level = -1
        self.t = self.T[self.level]
        self.d = self.D[self.level]
        self.case = self.cases[self.level]

    def _exp_part(self, exps):
        """
        Try to build an exponential extension.

        Returns
        =======

        Returns True if there was a new extension, False if there was no new
        extension but it was able to rewrite the given exponentials in terms
        of the existing extension, and None if the entire extension building
        process should be restarted.  If the process fails because there is no
        way around an algebraic extension (e.g., exp(log(x)/2)), it will raise
        NotImplementedError.
        """
        from .prde import is_log_deriv_k_t_radical
        new_extension = False
        restart = False
        expargs = [i.exp for i in exps]
        ip = integer_powers(expargs)
        for arg, others in ip:
            # Minimize potential problems with algebraic substitution
            others.sort(key=lambda i: i[1])

            arga, argd = frac_in(arg, self.t)
            A = is_log_deriv_k_t_radical(arga, argd, self)

            if A is not None:
                ans, u, n, const = A
                # if n is 1 or -1, it's algebraic, but we can handle it
                if n == -1:
                    # This probably will never happen, because
                    # Rational.as_numer_denom() returns the negative term in
                    # the numerator.  But in case that changes, reduce it to
                    # n == 1.
                    n = 1
                    u **= -1
                    const *= -1
                    ans = [(i, -j) for i, j in ans]

                if n == 1:
                    # Example: exp(x + x**2) over QQ(x, exp(x), exp(x**2))
                    self.newf = self.newf.xreplace({exp(arg): exp(const)*Mul(*[
                        u**power for u, power in ans])})
                    self.newf = self.newf.xreplace({exp(p*exparg):
                        exp(const*p) * Mul(*[u**power for u, power in ans])
                        for exparg, p in others})
                    # TODO: Add something to backsubs to put exp(const*p)
                    # back together.

                    continue

                else:
                    # Bad news: we have an algebraic radical.  But maybe we
                    # could still avoid it by choosing a different extension.
                    # For example, integer_powers() won't handle exp(x/2 + 1)
                    # over QQ(x, exp(x)), but if we pull out the exp(1), it
                    # will.  Or maybe we have exp(x + x**2/2), over
                    # QQ(x, exp(x), exp(x**2)), which is exp(x)*sqrt(exp(x**2)),
                    # but if we use QQ(x, exp(x), exp(x**2/2)), then they will
                    # all work.
                    #
                    # So here is what we do: If there is a non-zero const, pull
                    # it out and retry.  Also, if len(ans) > 1, then rewrite
                    # exp(arg) as the product of exponentials from ans, and
                    # retry that.  If const == 0 and len(ans) == 1, then we
                    # assume that it would have been handled by either
                    # integer_powers() or n == 1 above if it could be handled,
                    # so we give up at that point.  For example, you can never
                    # handle exp(log(x)/2) because it equals sqrt(x).

                    if const or len(ans) > 1:
                        rad = Mul(*[term**(power/n) for term, power in ans])
                        self.newf = self.newf.xreplace({exp(p*exparg):
                            exp(const*p)*rad for exparg, p in others})
                        self.newf = self.newf.xreplace(dict(list(zip(reversed(self.T),
                            reversed([f(self.x) for f in self.Tfuncs])))))
                        restart = True
                        break
                    else:
                        # TODO: give algebraic dependence in error string
                        raise NotImplementedError("Cannot integrate over "
                            "algebraic extensions.")

            else:
                arga, argd = frac_in(arg, self.t)
                darga = (argd*derivation(Poly(arga, self.t), self) -
                    arga*derivation(Poly(argd, self.t), self))
                dargd = argd**2
                darga, dargd = darga.cancel(dargd, include=True)
                darg = darga.as_expr()/dargd.as_expr()
                self.t = next(self.ts)
                self.T.append(self.t)
                self.extargs.append(arg)
                self.exts.append('exp')
                self.D.append(darg.as_poly(self.t, expand=False)*Poly(self.t,
                    self.t, expand=False))
                if self.dummy:
                    i = Dummy("i")
                else:
                    i = Symbol('i')
                self.Tfuncs += [Lambda(i, exp(arg.subs(self.x, i)))]
                self.newf = self.newf.xreplace(
                        {exp(exparg): self.t**p for exparg, p in others})
                new_extension = True

        if restart:
            return None
        return new_extension

    def _log_part(self, logs):
        """
        Try to build a logarithmic extension.

        Returns
        =======

        Returns True if there was a new extension and False if there was no new
        extension but it was able to rewrite the given logarithms in terms
        of the existing extension.  Unlike with exponential extensions, there
        is no way that a logarithm is not transcendental over and cannot be
        rewritten in terms of an already existing extension in a non-algebraic
        way, so this function does not ever return None or raise
        NotImplementedError.
        """
        from .prde import is_deriv_k
        new_extension = False
        logargs = [i.args[0] for i in logs]
        for arg in ordered(logargs):
            # The log case is easier, because whenever a logarithm is algebraic
            # over the base field, it is of the form a1*t1 + ... an*tn + c,
            # which is a polynomial, so we can just replace it with that.
            # In other words, we don't have to worry about radicals.
            arga, argd = frac_in(arg, self.t)
            A = is_deriv_k(arga, argd, self)
            if A is not None:
                ans, u, const = A
                newterm = log(const) + u
                self.newf = self.newf.xreplace({log(arg): newterm})
                continue

            else:
                arga, argd = frac_in(arg, self.t)
                darga = (argd*derivation(Poly(arga, self.t), self) -
                    arga*derivation(Poly(argd, self.t), self))
                dargd = argd**2
                darg = darga.as_expr()/dargd.as_expr()
                self.t = next(self.ts)
                self.T.append(self.t)
                self.extargs.append(arg)
                self.exts.append('log')
                self.D.append(cancel(darg.as_expr()/arg).as_poly(self.t,
                    expand=False))
                if self.dummy:
                    i = Dummy("i")
                else:
                    i = Symbol('i')
                self.Tfuncs += [Lambda(i, log(arg.subs(self.x, i)))]
                self.newf = self.newf.xreplace({log(arg): self.t})
                new_extension = True

        return new_extension

    @property
    def _important_attrs(self):
        """
        Returns some of the more important attributes of self.

        Explanation
        ===========

        Used for testing and debugging purposes.

        The attributes are (fa, fd, D, T, Tfuncs, backsubs,
        exts, extargs).
        """
        return (self.fa, self.fd, self.D, self.T, self.Tfuncs,
            self.backsubs, self.exts, self.extargs)

    # NOTE: this printing doesn't follow the Python's standard
    # eval(repr(DE)) == DE, where DE is the DifferentialExtension object,
    # also this printing is supposed to contain all the important
    # attributes of a DifferentialExtension object
    def __repr__(self):
        # no need to have GeneratorType object printed in it
        r = [(attr, getattr(self, attr)) for attr in self.__slots__
                if not isinstance(getattr(self, attr), GeneratorType)]
        return self.__class__.__name__ + '(dict(%r))' % (r)

    # fancy printing of DifferentialExtension object
    def __str__(self):
        return (self.__class__.__name__ + '({fa=%s, fd=%s, D=%s})' %
                (self.fa, self.fd, self.D))

    # should only be used for debugging purposes, internally
    # f1 = f2 = log(x) at different places in code execution
    # may return D1 != D2 as True, since 'level' or other attribute
    # may differ
    def __eq__(self, other):
        for attr in self.__class__.__slots__:
            d1, d2 = getattr(self, attr), getattr(other, attr)
            if not (isinstance(d1, GeneratorType) or d1 == d2):
                return False
        return True

    def reset(self):
        """
        Reset self to an initial state.  Used by __init__.
        """
        self.t = self.x
        self.T = [self.x]
        self.D = [Poly(1, self.x)]
        self.level = -1
        self.exts = [None]
        self.extargs = [None]
        if self.dummy:
            self.ts = numbered_symbols('t', cls=Dummy)
        else:
            # For testing
            self.ts = numbered_symbols('t')
        # For various things that we change to make things work that we need to
        # change back when we are done.
        self.backsubs = []
        self.Tfuncs = []
        self.newf = self.f

    def indices(self, extension):
        """
        Parameters
        ==========

        extension : str
            Represents a valid extension type.

        Returns
        =======

        list: A list of indices of 'exts' where extension of
            type 'extension' is present.

        Examples
        ========

        >>> from sympy.integrals.risch import DifferentialExtension
        >>> from sympy import log, exp
        >>> from sympy.abc import x
        >>> DE = DifferentialExtension(log(x) + exp(x), x, handle_first='exp')
        >>> DE.indices('log')
        [2]
        >>> DE.indices('exp')
        [1]

        """
        return [i for i, ext in enumerate(self.exts) if ext == extension]

    def increment_level(self):
        """
        Increment the level of self.

        Explanation
        ===========

        This makes the working differential extension larger.  self.level is
        given relative to the end of the list (-1, -2, etc.), so we do not need
        do worry about it when building the extension.
        """
        if self.level >= -1:
            raise ValueError("The level of the differential extension cannot "
                "be incremented any further.")

        self.level += 1
        self.t = self.T[self.level]
        self.d = self.D[self.level]
        self.case = self.cases[self.level]
        return None

    def decrement_level(self):
        """
        Decrease the level of self.

        Explanation
        ===========

        This makes the working differential extension smaller.  self.level is
        given relative to the end of the list (-1, -2, etc.), so we do not need
        do worry about it when building the extension.
        """
        if self.level <= -len(self.T):
            raise ValueError("The level of the differential extension cannot "
                "be decremented any further.")

        self.level -= 1
        self.t = self.T[self.level]
        self.d = self.D[self.level]
        self.case = self.cases[self.level]
        return None


def update_sets(seq, atoms, func):
    s = set(seq)
    s = atoms.intersection(s)
    new = atoms - s
    s.update(list(filter(func, new)))
    return list(s)


class DecrementLevel:
    """
    A context manager for decrementing the level of a DifferentialExtension.
    """
    __slots__ = ('DE',)

    def __init__(self, DE):
        self.DE = DE
        return

    def __enter__(self):
        self.DE.decrement_level()

    def __exit__(self, exc_type, exc_value, traceback):
        self.DE.increment_level()


class NonElementaryIntegralException(Exception):
    """
    Exception used by subroutines within the Risch algorithm to indicate to one
    another that the function being integrated does not have an elementary
    integral in the given differential field.
    """
    # TODO: Rewrite algorithms below to use this (?)

    # TODO: Pass through information about why the integral was nonelementary,
    # and store that in the resulting NonElementaryIntegral somehow.
    pass


def gcdex_diophantine(a, b, c):
    """
    Extended Euclidean Algorithm, Diophantine version.

    Explanation
    ===========

    Given ``a``, ``b`` in K[x] and ``c`` in (a, b), the ideal generated by ``a`` and
    ``b``, return (s, t) such that s*a + t*b == c and either s == 0 or s.degree()
    < b.degree().
    """
    # Extended Euclidean Algorithm (Diophantine Version) pg. 13
    # TODO: This should go in densetools.py.
    # XXX: Better name?

    s, g = a.half_gcdex(b)
    s *= c.exquo(g)  # Inexact division means c is not in (a, b)
    if s and s.degree() >= b.degree():
        _, s = s.div(b)
    t = (c - s*a).exquo(b)
    return (s, t)


def frac_in(f, t, *, cancel=False, **kwargs):
    """
    Returns the tuple (fa, fd), where fa and fd are Polys in t.

    Explanation
    ===========

    This is a common idiom in the Risch Algorithm functions, so we abstract
    it out here. ``f`` should be a basic expression, a Poly, or a tuple (fa, fd),
    where fa and fd are either basic expressions or Polys, and f == fa/fd.
    **kwargs are applied to Poly.
    """
    if isinstance(f, tuple):
        fa, fd = f
        f = fa.as_expr()/fd.as_expr()
    fa, fd = f.as_expr().as_numer_denom()
    fa, fd = fa.as_poly(t, **kwargs), fd.as_poly(t, **kwargs)
    if cancel:
        fa, fd = fa.cancel(fd, include=True)
    if fa is None or fd is None:
        raise ValueError("Could not turn %s into a fraction in %s." % (f, t))
    return (fa, fd)


def as_poly_1t(p, t, z):
    """
    (Hackish) way to convert an element ``p`` of K[t, 1/t] to K[t, z].

    In other words, ``z == 1/t`` will be a dummy variable that Poly can handle
    better.

    See issue 5131.

    Examples
    ========

    >>> from sympy import random_poly
    >>> from sympy.integrals.risch import as_poly_1t
    >>> from sympy.abc import x, z

    >>> p1 = random_poly(x, 10, -10, 10)
    >>> p2 = random_poly(x, 10, -10, 10)
    >>> p = p1 + p2.subs(x, 1/x)
    >>> as_poly_1t(p, x, z).as_expr().subs(z, 1/x) == p
    True
    """
    # TODO: Use this on the final result.  That way, we can avoid answers like
    # (...)*exp(-x).
    pa, pd = frac_in(p, t, cancel=True)
    if not pd.is_monomial:
        # XXX: Is there a better Poly exception that we could raise here?
        # Either way, if you see this (from the Risch Algorithm) it indicates
        # a bug.
        raise PolynomialError("%s is not an element of K[%s, 1/%s]." % (p, t, t))

    t_part, remainder = pa.div(pd)

    ans = t_part.as_poly(t, z, expand=False)

    if remainder:
        one = remainder.one
        tp = t*one
        r = pd.degree() - remainder.degree()
        z_part = remainder.transform(one, tp) * tp**r
        z_part = z_part.replace(t, z).to_field().quo_ground(pd.LC())
        ans += z_part.as_poly(t, z, expand=False)

    return ans


def derivation(p, DE, coefficientD=False, basic=False):
    """
    Computes Dp.

    Explanation
    ===========

    Given the derivation D with D = d/dx and p is a polynomial in t over
    K(x), return Dp.

    If coefficientD is True, it computes the derivation kD
    (kappaD), which is defined as kD(sum(ai*Xi**i, (i, 0, n))) ==
    sum(Dai*Xi**i, (i, 1, n)) (Definition 3.2.2, page 80).  X in this case is
    T[-1], so coefficientD computes the derivative just with respect to T[:-1],
    with T[-1] treated as a constant.

    If ``basic=True``, the returns a Basic expression.  Elements of D can still be
    instances of Poly.
    """
    if basic:
        r = 0
    else:
        r = Poly(0, DE.t)

    t = DE.t
    if coefficientD:
        if DE.level <= -len(DE.T):
            # 'base' case, the answer is 0.
            return r
        DE.decrement_level()

    D = DE.D[:len(DE.D) + DE.level + 1]
    T = DE.T[:len(DE.T) + DE.level + 1]

    for d, v in zip(D, T):
        pv = p.as_poly(v)
        if pv is None or basic:
            pv = p.as_expr()

        if basic:
            r += d.as_expr()*pv.diff(v)
        else:
            r += (d.as_expr()*pv.diff(v).as_expr()).as_poly(t)

    if basic:
        r = cancel(r)
    if coefficientD:
        DE.increment_level()

    return r


def get_case(d, t):
    """
    Returns the type of the derivation d.

    Returns one of {'exp', 'tan', 'base', 'primitive', 'other_linear',
    'other_nonlinear'}.
    """
    if not d.expr.has(t):
        if d.is_one:
            return 'base'
        return 'primitive'
    if d.rem(Poly(t, t)).is_zero:
        return 'exp'
    if d.rem(Poly(1 + t**2, t)).is_zero:
        return 'tan'
    if d.degree(t) > 1:
        return 'other_nonlinear'
    return 'other_linear'


def splitfactor(p, DE, coefficientD=False, z=None):
    """
    Splitting factorization.

    Explanation
    ===========

    Given a derivation D on k[t] and ``p`` in k[t], return (p_n, p_s) in
    k[t] x k[t] such that p = p_n*p_s, p_s is special, and each square
    factor of p_n is normal.

    Page. 100
    """
    kinv = [1/x for x in DE.T[:DE.level]]
    if z:
        kinv.append(z)

    One = Poly(1, DE.t, domain=p.get_domain())
    Dp = derivation(p, DE, coefficientD=coefficientD)
    # XXX: Is this right?
    if p.is_zero:
        return (p, One)

    if not p.expr.has(DE.t):
        s = p.as_poly(*kinv).gcd(Dp.as_poly(*kinv)).as_poly(DE.t)
        n = p.exquo(s)
        return (n, s)

    if not Dp.is_zero:
        h = p.gcd(Dp).to_field()
        g = p.gcd(p.diff(DE.t)).to_field()
        s = h.exquo(g)

        if s.degree(DE.t) == 0:
            return (p, One)

        q_split = splitfactor(p.exquo(s), DE, coefficientD=coefficientD)

        return (q_split[0], q_split[1]*s)
    else:
        return (p, One)


def splitfactor_sqf(p, DE, coefficientD=False, z=None, basic=False):
    """
    Splitting Square-free Factorization.

    Explanation
    ===========

    Given a derivation D on k[t] and ``p`` in k[t], returns (N1, ..., Nm)
    and (S1, ..., Sm) in k[t]^m such that p =
    (N1*N2**2*...*Nm**m)*(S1*S2**2*...*Sm**m) is a splitting
    factorization of ``p`` and the Ni and Si are square-free and coprime.
    """
    # TODO: This algorithm appears to be faster in every case
    # TODO: Verify this and splitfactor() for multiple extensions
    kkinv = [1/x for x in DE.T[:DE.level]] + DE.T[:DE.level]
    if z:
        kkinv = [z]

    S = []
    N = []
    p_sqf = p.sqf_list_include()
    if p.is_zero:
        return (((p, 1),), ())

    for pi, i in p_sqf:
        Si = pi.as_poly(*kkinv).gcd(derivation(pi, DE,
            coefficientD=coefficientD,basic=basic).as_poly(*kkinv)).as_poly(DE.t)
        pi = Poly(pi, DE.t)
        Si = Poly(Si, DE.t)
        Ni = pi.exquo(Si)
        if not Si.is_one:
            S.append((Si, i))
        if not Ni.is_one:
            N.append((Ni, i))

    return (tuple(N), tuple(S))


def canonical_representation(a, d, DE):
    """
    Canonical Representation.

    Explanation
    ===========

    Given a derivation D on k[t] and f = a/d in k(t), return (f_p, f_s,
    f_n) in k[t] x k(t) x k(t) such that f = f_p + f_s + f_n is the
    canonical representation of f (f_p is a polynomial, f_s is reduced
    (has a special denominator), and f_n is simple (has a normal
    denominator).
    """
    # Make d monic
    l = Poly(1/d.LC(), DE.t)
    a, d = a.mul(l), d.mul(l)

    q, r = a.div(d)
    dn, ds = splitfactor(d, DE)

    b, c = gcdex_diophantine(dn.as_poly(DE.t), ds.as_poly(DE.t), r.as_poly(DE.t))
    b, c = b.as_poly(DE.t), c.as_poly(DE.t)

    return (q, (b, ds), (c, dn))


def hermite_reduce(a, d, DE):
    """
    Hermite Reduction - Mack's Linear Version.

    Given a derivation D on k(t) and f = a/d in k(t), returns g, h, r in
    k(t) such that f = Dg + h + r, h is simple, and r is reduced.

    """
    # Make d monic
    l = Poly(1/d.LC(), DE.t)
    a, d = a.mul(l), d.mul(l)

    fp, fs, fn = canonical_representation(a, d, DE)
    a, d = fn
    l = Poly(1/d.LC(), DE.t)
    a, d = a.mul(l), d.mul(l)

    ga = Poly(0, DE.t)
    gd = Poly(1, DE.t)

    dd = derivation(d, DE)
    dm = gcd(d.to_field(), dd.to_field()).as_poly(DE.t)
    ds, _ = d.div(dm)

    while dm.degree(DE.t) > 0:

        ddm = derivation(dm, DE)
        dm2 = gcd(dm.to_field(), ddm.to_field())
        dms, _ = dm.div(dm2)
        ds_ddm = ds.mul(ddm)
        ds_ddm_dm, _ = ds_ddm.div(dm)

        b, c = gcdex_diophantine(-ds_ddm_dm.as_poly(DE.t),
            dms.as_poly(DE.t), a.as_poly(DE.t))
        b, c = b.as_poly(DE.t), c.as_poly(DE.t)

        db = derivation(b, DE).as_poly(DE.t)
        ds_dms, _ = ds.div(dms)
        a = c.as_poly(DE.t) - db.mul(ds_dms).as_poly(DE.t)

        ga = ga*dm + b*gd
        gd = gd*dm
        ga, gd = ga.cancel(gd, include=True)
        dm = dm2

    q, r = a.div(ds)
    ga, gd = ga.cancel(gd, include=True)

    r, d = r.cancel(ds, include=True)
    rra = q*fs[1] + fp*fs[1] + fs[0]
    rrd = fs[1]
    rra, rrd = rra.cancel(rrd, include=True)

    return ((ga, gd), (r, d), (rra, rrd))


def polynomial_reduce(p, DE):
    """
    Polynomial Reduction.

    Explanation
    ===========

    Given a derivation D on k(t) and p in k[t] where t is a nonlinear
    monomial over k, return q, r in k[t] such that p = Dq  + r, and
    deg(r) < deg_t(Dt).
    """
    q = Poly(0, DE.t)
    while p.degree(DE.t) >= DE.d.degree(DE.t):
        m = p.degree(DE.t) - DE.d.degree(DE.t) + 1
        q0 = Poly(DE.t**m, DE.t).mul(Poly(p.as_poly(DE.t).LC()/
            (m*DE.d.LC()), DE.t))
        q += q0
        p = p - derivation(q0, DE)

    return (q, p)


def laurent_series(a, d, F, n, DE):
    """
    Contribution of ``F`` to the full partial fraction decomposition of A/D.

    Explanation
    ===========

    Given a field K of characteristic 0 and ``A``,``D``,``F`` in K[x] with D monic,
    nonzero, coprime with A, and ``F`` the factor of multiplicity n in the square-
    free factorization of D, return the principal parts of the Laurent series of
    A/D at all the zeros of ``F``.
    """
    if F.degree()==0:
        return 0
    Z = _symbols('z', n)
    z = Symbol('z')
    Z.insert(0, z)
    delta_a = Poly(0, DE.t)
    delta_d = Poly(1, DE.t)

    E = d.quo(F**n)
    ha, hd = (a, E*Poly(z**n, DE.t))
    dF = derivation(F,DE)
    B, _ = gcdex_diophantine(E, F, Poly(1,DE.t))
    C, _ = gcdex_diophantine(dF, F, Poly(1,DE.t))

    # initialization
    F_store = F
    V, DE_D_list, H_list= [], [], []

    for j in range(0, n):
    # jth derivative of z would be substituted with dfnth/(j+1) where dfnth =(d^n)f/(dx)^n
        F_store = derivation(F_store, DE)
        v = (F_store.as_expr())/(j + 1)
        V.append(v)
        DE_D_list.append(Poly(Z[j + 1],Z[j]))

    DE_new = DifferentialExtension(extension = {'D': DE_D_list}) #a differential indeterminate
    for j in range(0, n):
        zEha = Poly(z**(n + j), DE.t)*E**(j + 1)*ha
        zEhd = hd
        Pa, Pd = cancel((zEha, zEhd))[1], cancel((zEha, zEhd))[2]
        Q = Pa.quo(Pd)
        for i in range(0, j + 1):
            Q = Q.subs(Z[i], V[i])
        Dha = (hd*derivation(ha, DE, basic=True).as_poly(DE.t)
             + ha*derivation(hd, DE, basic=True).as_poly(DE.t)
             + hd*derivation(ha, DE_new, basic=True).as_poly(DE.t)
             + ha*derivation(hd, DE_new, basic=True).as_poly(DE.t))
        Dhd = Poly(j + 1, DE.t)*hd**2
        ha, hd = Dha, Dhd

        Ff, _ = F.div(gcd(F, Q))
        F_stara, F_stard = frac_in(Ff, DE.t)
        if F_stara.degree(DE.t) - F_stard.degree(DE.t) > 0:
            QBC = Poly(Q, DE.t)*B**(1 + j)*C**(n + j)
            H = QBC
            H_list.append(H)
            H = (QBC*F_stard).rem(F_stara)
            alphas = real_roots(F_stara)
            for alpha in list(alphas):
                delta_a = delta_a*Poly((DE.t - alpha)**(n - j), DE.t) + Poly(H.eval(alpha), DE.t)
                delta_d = delta_d*Poly((DE.t - alpha)**(n - j), DE.t)
    return (delta_a, delta_d, H_list)


def recognize_derivative(a, d, DE, z=None):
    """
    Compute the squarefree factorization of the denominator of f
    and for each Di the polynomial H in K[x] (see Theorem 2.7.1), using the
    LaurentSeries algorithm. Write Di = GiEi where Gj = gcd(Hn, Di) and
    gcd(Ei,Hn) = 1. Since the residues of f at the roots of Gj are all 0, and
    the residue of f at a root alpha of Ei is Hi(a) != 0, f is the derivative of a
    rational function if and only if Ei = 1 for each i, which is equivalent to
    Di | H[-1] for each i.
    """
    flag =True
    a, d = a.cancel(d, include=True)
    _, r = a.div(d)
    Np, Sp = splitfactor_sqf(d, DE, coefficientD=True, z=z)

    j = 1
    for s, _ in Sp:
        delta_a, delta_d, H = laurent_series(r, d, s, j, DE)
        g = gcd(d, H[-1]).as_poly()
        if g is not d:
            flag = False
            break
        j = j + 1
    return flag


def recognize_log_derivative(a, d, DE, z=None):
    """
    There exists a v in K(x)* such that f = dv/v
    where f a rational function if and only if f can be written as f = A/D
    where D is squarefree,deg(A) < deg(D), gcd(A, D) = 1,
    and all the roots of the Rothstein-Trager resultant are integers. In that case,
    any of the Rothstein-Trager, Lazard-Rioboo-Trager or Czichowski algorithm
    produces u in K(x) such that du/dx = uf.
    """

    z = z or Dummy('z')
    a, d = a.cancel(d, include=True)
    _, a = a.div(d)

    pz = Poly(z, DE.t)
    Dd = derivation(d, DE)
    q = a - pz*Dd
    r, _ = d.resultant(q, includePRS=True)
    r = Poly(r, z)
    Np, Sp = splitfactor_sqf(r, DE, coefficientD=True, z=z)

    for s, _ in Sp:
        # TODO also consider the complex roots which should
        # turn the flag false
        a = real_roots(s.as_poly(z))

        if not all(j.is_Integer for j in a):
            return False
    return True

def residue_reduce(a, d, DE, z=None, invert=True):
    """
    Lazard-Rioboo-Rothstein-Trager resultant reduction.

    Explanation
    ===========

    Given a derivation ``D`` on k(t) and f in k(t) simple, return g
    elementary over k(t) and a Boolean b in {True, False} such that f -
    Dg in k[t] if b == True or f + h and f + h - Dg do not have an
    elementary integral over k(t) for any h in k<t> (reduced) if b ==
    False.

    Returns (G, b), where G is a tuple of tuples of the form (s_i, S_i),
    such that g = Add(*[RootSum(s_i, lambda z: z*log(S_i(z, t))) for
    S_i, s_i in G]). f - Dg is the remaining integral, which is elementary
    only if b == True, and hence the integral of f is elementary only if
    b == True.

    f - Dg is not calculated in this function because that would require
    explicitly calculating the RootSum.  Use residue_reduce_derivation().
    """
    # TODO: Use log_to_atan() from rationaltools.py
    # If r = residue_reduce(...), then the logarithmic part is given by:
    # sum([RootSum(a[0].as_poly(z), lambda i: i*log(a[1].as_expr()).subs(z,
    # i)).subs(t, log(x)) for a in r[0]])

    z = z or Dummy('z')
    a, d = a.cancel(d, include=True)
    a, d = a.to_field().mul_ground(1/d.LC()), d.to_field().mul_ground(1/d.LC())
    kkinv = [1/x for x in DE.T[:DE.level]] + DE.T[:DE.level]

    if a.is_zero:
        return ([], True)
    _, a = a.div(d)

    pz = Poly(z, DE.t)

    Dd = derivation(d, DE)
    q = a - pz*Dd

    if Dd.degree(DE.t) <= d.degree(DE.t):
        r, R = d.resultant(q, includePRS=True)
    else:
        r, R = q.resultant(d, includePRS=True)

    R_map, H = {}, []
    for i in R:
        R_map[i.degree()] = i

    r = Poly(r, z)
    Np, Sp = splitfactor_sqf(r, DE, coefficientD=True, z=z)

    for s, i in Sp:
        if i == d.degree(DE.t):
            s = Poly(s, z).monic()
            H.append((s, d))
        else:
            h = R_map.get(i)
            if h is None:
                continue
            h_lc = Poly(h.as_poly(DE.t).LC(), DE.t, field=True)

            h_lc_sqf = h_lc.sqf_list_include(all=True)

            for a, j in h_lc_sqf:
                h = Poly(h, DE.t, field=True).exquo(Poly(gcd(a, s**j, *kkinv),
                    DE.t))

            s = Poly(s, z).monic()

            if invert:
                h_lc = Poly(h.as_poly(DE.t).LC(), DE.t, field=True, expand=False)
                inv, coeffs = h_lc.as_poly(z, field=True).invert(s), [S.One]

                for coeff in h.coeffs()[1:]:
                    L = reduced(inv*coeff.as_poly(inv.gens), [s])[1]
                    coeffs.append(L.as_expr())

                h = Poly(dict(list(zip(h.monoms(), coeffs))), DE.t)

            H.append((s, h))

    b = not any(cancel(i.as_expr()).has(DE.t, z) for i, _ in Np)

    return (H, b)


def residue_reduce_to_basic(H, DE, z):
    """
    Converts the tuple returned by residue_reduce() into a Basic expression.
    """
    # TODO: check what Lambda does with RootOf
    i = Dummy('i')
    s = list(zip(reversed(DE.T), reversed([f(DE.x) for f in DE.Tfuncs])))

    return sum(RootSum(a[0].as_poly(z), Lambda(i, i*log(a[1].as_expr()).subs(
        {z: i}).subs(s))) for a in H)


def residue_reduce_derivation(H, DE, z):
    """
    Computes the derivation of an expression returned by residue_reduce().

    In general, this is a rational function in t, so this returns an
    as_expr() result.
    """
    # TODO: verify that this is correct for multiple extensions
    i = Dummy('i')
    return S(sum(RootSum(a[0].as_poly(z), Lambda(i, i*derivation(a[1],
        DE).as_expr().subs(z, i)/a[1].as_expr().subs(z, i))) for a in H))


def integrate_primitive_polynomial(p, DE):
    """
    Integration of primitive polynomials.

    Explanation
    ===========

    Given a primitive monomial t over k, and ``p`` in k[t], return q in k[t],
    r in k, and a bool b in {True, False} such that r = p - Dq is in k if b is
    True, or r = p - Dq does not have an elementary integral over k(t) if b is
    False.
    """
    Zero = Poly(0, DE.t)
    q = Poly(0, DE.t)

    if not p.expr.has(DE.t):
        return (Zero, p, True)

    from .prde import limited_integrate
    while True:
        if not p.expr.has(DE.t):
            return (q, p, True)

        Dta, Dtb = frac_in(DE.d, DE.T[DE.level - 1])

        with DecrementLevel(DE):  # We had better be integrating the lowest extension (x)
                                  # with ratint().
            a = p.LC()
            aa, ad = frac_in(a, DE.t)

            try:
                rv = limited_integrate(aa, ad, [(Dta, Dtb)], DE)
                if rv is None:
                    raise NonElementaryIntegralException
                (ba, bd), c = rv
            except NonElementaryIntegralException:
                return (q, p, False)

        m = p.degree(DE.t)
        q0 = c[0].as_poly(DE.t)*Poly(DE.t**(m + 1)/(m + 1), DE.t) + \
            (ba.as_expr()/bd.as_expr()).as_poly(DE.t)*Poly(DE.t**m, DE.t)

        p = p - derivation(q0, DE)
        q = q + q0


def integrate_primitive(a, d, DE, z=None):
    """
    Integration of primitive functions.

    Explanation
    ===========

    Given a primitive monomial t over k and f in k(t), return g elementary over
    k(t), i in k(t), and b in {True, False} such that i = f - Dg is in k if b
    is True or i = f - Dg does not have an elementary integral over k(t) if b
    is False.

    This function returns a Basic expression for the first argument.  If b is
    True, the second argument is Basic expression in k to recursively integrate.
    If b is False, the second argument is an unevaluated Integral, which has
    been proven to be nonelementary.
    """
    # XXX: a and d must be canceled, or this might return incorrect results
    z = z or Dummy("z")
    s = list(zip(reversed(DE.T), reversed([f(DE.x) for f in DE.Tfuncs])))

    g1, h, r = hermite_reduce(a, d, DE)
    g2, b = residue_reduce(h[0], h[1], DE, z=z)
    if not b:
        i = cancel(a.as_expr()/d.as_expr() - (g1[1]*derivation(g1[0], DE) -
            g1[0]*derivation(g1[1], DE)).as_expr()/(g1[1]**2).as_expr() -
            residue_reduce_derivation(g2, DE, z))
        i = NonElementaryIntegral(cancel(i).subs(s), DE.x)
        return ((g1[0].as_expr()/g1[1].as_expr()).subs(s) +
            residue_reduce_to_basic(g2, DE, z), i, b)

    # h - Dg2 + r
    p = cancel(h[0].as_expr()/h[1].as_expr() - residue_reduce_derivation(g2,
        DE, z) + r[0].as_expr()/r[1].as_expr())
    p = p.as_poly(DE.t)

    q, i, b = integrate_primitive_polynomial(p, DE)

    ret = ((g1[0].as_expr()/g1[1].as_expr() + q.as_expr()).subs(s) +
        residue_reduce_to_basic(g2, DE, z))
    if not b:
        # TODO: This does not do the right thing when b is False
        i = NonElementaryIntegral(cancel(i.as_expr()).subs(s), DE.x)
    else:
        i = cancel(i.as_expr())

    return (ret, i, b)


def integrate_hyperexponential_polynomial(p, DE, z):
    """
    Integration of hyperexponential polynomials.

    Explanation
    ===========

    Given a hyperexponential monomial t over k and ``p`` in k[t, 1/t], return q in
    k[t, 1/t] and a bool b in {True, False} such that p - Dq in k if b is True,
    or p - Dq does not have an elementary integral over k(t) if b is False.
    """
    t1 = DE.t
    dtt = DE.d.exquo(Poly(DE.t, DE.t))
    qa = Poly(0, DE.t)
    qd = Poly(1, DE.t)
    b = True

    if p.is_zero:
        return(qa, qd, b)

    from sympy.integrals.rde import rischDE

    with DecrementLevel(DE):
        for i in range(-p.degree(z), p.degree(t1) + 1):
            if not i:
                continue
            elif i < 0:
                # If you get AttributeError: 'NoneType' object has no attribute 'nth'
                # then this should really not have expand=False
                # But it shouldn't happen because p is already a Poly in t and z
                a = p.as_poly(z, expand=False).nth(-i)
            else:
                # If you get AttributeError: 'NoneType' object has no attribute 'nth'
                # then this should really not have expand=False
                a = p.as_poly(t1, expand=False).nth(i)

            aa, ad = frac_in(a, DE.t, field=True)
            aa, ad = aa.cancel(ad, include=True)
            iDt = Poly(i, t1)*dtt
            iDta, iDtd = frac_in(iDt, DE.t, field=True)
            try:
                va, vd = rischDE(iDta, iDtd, Poly(aa, DE.t), Poly(ad, DE.t), DE)
                va, vd = frac_in((va, vd), t1, cancel=True)
            except NonElementaryIntegralException:
                b = False
            else:
                qa = qa*vd + va*Poly(t1**i)*qd
                qd *= vd

    return (qa, qd, b)


def integrate_hyperexponential(a, d, DE, z=None, conds='piecewise'):
    """
    Integration of hyperexponential functions.

    Explanation
    ===========

    Given a hyperexponential monomial t over k and f in k(t), return g
    elementary over k(t), i in k(t), and a bool b in {True, False} such that
    i = f - Dg is in k if b is True or i = f - Dg does not have an elementary
    integral over k(t) if b is False.

    This function returns a Basic expression for the first argument.  If b is
    True, the second argument is Basic expression in k to recursively integrate.
    If b is False, the second argument is an unevaluated Integral, which has
    been proven to be nonelementary.
    """
    # XXX: a and d must be canceled, or this might return incorrect results
    z = z or Dummy("z")
    s = list(zip(reversed(DE.T), reversed([f(DE.x) for f in DE.Tfuncs])))

    g1, h, r = hermite_reduce(a, d, DE)
    g2, b = residue_reduce(h[0], h[1], DE, z=z)
    if not b:
        i = cancel(a.as_expr()/d.as_expr() - (g1[1]*derivation(g1[0], DE) -
            g1[0]*derivation(g1[1], DE)).as_expr()/(g1[1]**2).as_expr() -
            residue_reduce_derivation(g2, DE, z))
        i = NonElementaryIntegral(cancel(i.subs(s)), DE.x)
        return ((g1[0].as_expr()/g1[1].as_expr()).subs(s) +
            residue_reduce_to_basic(g2, DE, z), i, b)

    # p should be a polynomial in t and 1/t, because Sirr == k[t, 1/t]
    # h - Dg2 + r
    p = cancel(h[0].as_expr()/h[1].as_expr() - residue_reduce_derivation(g2,
        DE, z) + r[0].as_expr()/r[1].as_expr())
    pp = as_poly_1t(p, DE.t, z)

    qa, qd, b = integrate_hyperexponential_polynomial(pp, DE, z)

    i = pp.nth(0, 0)

    ret = ((g1[0].as_expr()/g1[1].as_expr()).subs(s) \
        + residue_reduce_to_basic(g2, DE, z))

    qas = qa.as_expr().subs(s)
    qds = qd.as_expr().subs(s)
    if conds == 'piecewise' and DE.x not in qds.free_symbols:
        # We have to be careful if the exponent is S.Zero!

        # XXX: Does qd = 0 always necessarily correspond to the exponential
        # equaling 1?
        ret += Piecewise(
                (qas/qds, Ne(qds, 0)),
                (integrate((p - i).subs(DE.t, 1).subs(s), DE.x), True)
            )
    else:
        ret += qas/qds

    if not b:
        i = p - (qd*derivation(qa, DE) - qa*derivation(qd, DE)).as_expr()/\
            (qd**2).as_expr()
        i = NonElementaryIntegral(cancel(i).subs(s), DE.x)
    return (ret, i, b)


def integrate_hypertangent_polynomial(p, DE):
    """
    Integration of hypertangent polynomials.

    Explanation
    ===========

    Given a differential field k such that sqrt(-1) is not in k, a
    hypertangent monomial t over k, and p in k[t], return q in k[t] and
    c in k such that p - Dq - c*D(t**2 + 1)/(t**1 + 1) is in k and p -
    Dq does not have an elementary integral over k(t) if Dc != 0.
    """
    # XXX: Make sure that sqrt(-1) is not in k.
    q, r = polynomial_reduce(p, DE)
    a = DE.d.exquo(Poly(DE.t**2 + 1, DE.t))
    c = Poly(r.nth(1)/(2*a.as_expr()), DE.t)
    return (q, c)


def integrate_nonlinear_no_specials(a, d, DE, z=None):
    """
    Integration of nonlinear monomials with no specials.

    Explanation
    ===========

    Given a nonlinear monomial t over k such that Sirr ({p in k[t] | p is
    special, monic, and irreducible}) is empty, and f in k(t), returns g
    elementary over k(t) and a Boolean b in {True, False} such that f - Dg is
    in k if b == True, or f - Dg does not have an elementary integral over k(t)
    if b == False.

    This function is applicable to all nonlinear extensions, but in the case
    where it returns b == False, it will only have proven that the integral of
    f - Dg is nonelementary if Sirr is empty.

    This function returns a Basic expression.
    """
    # TODO: Integral from k?
    # TODO: split out nonelementary integral
    # XXX: a and d must be canceled, or this might not return correct results
    z = z or Dummy("z")
    s = list(zip(reversed(DE.T), reversed([f(DE.x) for f in DE.Tfuncs])))

    g1, h, r = hermite_reduce(a, d, DE)
    g2, b = residue_reduce(h[0], h[1], DE, z=z)
    if not b:
        return ((g1[0].as_expr()/g1[1].as_expr()).subs(s) +
            residue_reduce_to_basic(g2, DE, z), b)

    # Because f has no specials, this should be a polynomial in t, or else
    # there is a bug.
    p = cancel(h[0].as_expr()/h[1].as_expr() - residue_reduce_derivation(g2,
        DE, z).as_expr() + r[0].as_expr()/r[1].as_expr()).as_poly(DE.t)
    q1, q2 = polynomial_reduce(p, DE)

    if q2.expr.has(DE.t):
        b = False
    else:
        b = True

    ret = (cancel(g1[0].as_expr()/g1[1].as_expr() + q1.as_expr()).subs(s) +
        residue_reduce_to_basic(g2, DE, z))
    return (ret, b)


class NonElementaryIntegral(Integral):
    """
    Represents a nonelementary Integral.

    Explanation
    ===========

    If the result of integrate() is an instance of this class, it is
    guaranteed to be nonelementary.  Note that integrate() by default will try
    to find any closed-form solution, even in terms of special functions which
    may themselves not be elementary.  To make integrate() only give
    elementary solutions, or, in the cases where it can prove the integral to
    be nonelementary, instances of this class, use integrate(risch=True).
    In this case, integrate() may raise NotImplementedError if it cannot make
    such a determination.

    integrate() uses the deterministic Risch algorithm to integrate elementary
    functions or prove that they have no elementary integral.  In some cases,
    this algorithm can split an integral into an elementary and nonelementary
    part, so that the result of integrate will be the sum of an elementary
    expression and a NonElementaryIntegral.

    Examples
    ========

    >>> from sympy import integrate, exp, log, Integral
    >>> from sympy.abc import x

    >>> a = integrate(exp(-x**2), x, risch=True)
    >>> print(a)
    Integral(exp(-x**2), x)
    >>> type(a)
    <class 'sympy.integrals.risch.NonElementaryIntegral'>

    >>> expr = (2*log(x)**2 - log(x) - x**2)/(log(x)**3 - x**2*log(x))
    >>> b = integrate(expr, x, risch=True)
    >>> print(b)
    -log(-x + log(x))/2 + log(x + log(x))/2 + Integral(1/log(x), x)
    >>> type(b.atoms(Integral).pop())
    <class 'sympy.integrals.risch.NonElementaryIntegral'>

    """
    # TODO: This is useful in and of itself, because isinstance(result,
    # NonElementaryIntegral) will tell if the integral has been proven to be
    # elementary. But should we do more?  Perhaps a no-op .doit() if
    # elementary=True?  Or maybe some information on why the integral is
    # nonelementary.
    pass


def risch_integrate(f, x, extension=None, handle_first='log',
                    separate_integral=False, rewrite_complex=None,
                    conds='piecewise'):
    r"""
    The Risch Integration Algorithm.

    Explanation
    ===========

    Only transcendental functions are supported.  Currently, only exponentials
    and logarithms are supported, but support for trigonometric functions is
    forthcoming.

    If this function returns an unevaluated Integral in the result, it means
    that it has proven that integral to be nonelementary.  Any errors will
    result in raising NotImplementedError.  The unevaluated Integral will be
    an instance of NonElementaryIntegral, a subclass of Integral.

    handle_first may be either 'exp' or 'log'.  This changes the order in
    which the extension is built, and may result in a different (but
    equivalent) solution (for an example of this, see issue 5109).  It is also
    possible that the integral may be computed with one but not the other,
    because not all cases have been implemented yet.  It defaults to 'log' so
    that the outer extension is exponential when possible, because more of the
    exponential case has been implemented.

    If ``separate_integral`` is ``True``, the result is returned as a tuple (ans, i),
    where the integral is ans + i, ans is elementary, and i is either a
    NonElementaryIntegral or 0.  This useful if you want to try further
    integrating the NonElementaryIntegral part using other algorithms to
    possibly get a solution in terms of special functions.  It is False by
    default.

    Examples
    ========

    >>> from sympy.integrals.risch import risch_integrate
    >>> from sympy import exp, log, pprint
    >>> from sympy.abc import x

    First, we try integrating exp(-x**2). Except for a constant factor of
    2/sqrt(pi), this is the famous error function.

    >>> pprint(risch_integrate(exp(-x**2), x))
      /
     |
     |    2
     |  -x
     | e    dx
     |
    /

    The unevaluated Integral in the result means that risch_integrate() has
    proven that exp(-x**2) does not have an elementary anti-derivative.

    In many cases, risch_integrate() can split out the elementary
    anti-derivative part from the nonelementary anti-derivative part.
    For example,

    >>> pprint(risch_integrate((2*log(x)**2 - log(x) - x**2)/(log(x)**3 -
    ... x**2*log(x)), x))
                                             /
                                            |
      log(-x + log(x))   log(x + log(x))    |   1
    - ---------------- + --------------- +  | ------ dx
             2                  2           | log(x)
                                            |
                                           /

    This means that it has proven that the integral of 1/log(x) is
    nonelementary.  This function is also known as the logarithmic integral,
    and is often denoted as Li(x).

    risch_integrate() currently only accepts purely transcendental functions
    with exponentials and logarithms, though note that this can include
    nested exponentials and logarithms, as well as exponentials with bases
    other than E.

    >>> pprint(risch_integrate(exp(x)*exp(exp(x)), x))
     / x\
     \e /
    e
    >>> pprint(risch_integrate(exp(exp(x)), x))
      /
     |
     |  / x\
     |  \e /
     | e     dx
     |
    /

    >>> pprint(risch_integrate(x*x**x*log(x) + x**x + x*x**x, x))
       x
    x*x
    >>> pprint(risch_integrate(x**x, x))
      /
     |
     |  x
     | x  dx
     |
    /

    >>> pprint(risch_integrate(-1/(x*log(x)*log(log(x))**2), x))
         1
    -----------
    log(log(x))

    """
    f = S(f)

    DE = extension or DifferentialExtension(f, x, handle_first=handle_first,
            dummy=True, rewrite_complex=rewrite_complex)
    fa, fd = DE.fa, DE.fd

    result = S.Zero
    for case in reversed(DE.cases):
        if not fa.expr.has(DE.t) and not fd.expr.has(DE.t) and not case == 'base':
            DE.decrement_level()
            fa, fd = frac_in((fa, fd), DE.t)
            continue

        fa, fd = fa.cancel(fd, include=True)
        if case == 'exp':
            ans, i, b = integrate_hyperexponential(fa, fd, DE, conds=conds)
        elif case == 'primitive':
            ans, i, b = integrate_primitive(fa, fd, DE)
        elif case == 'base':
            # XXX: We can't call ratint() directly here because it doesn't
            # handle polynomials correctly.
            ans = integrate(fa.as_expr()/fd.as_expr(), DE.x, risch=False)
            b = False
            i = S.Zero
        else:
            raise NotImplementedError("Only exponential and logarithmic "
            "extensions are currently supported.")

        result += ans
        if b:
            DE.decrement_level()
            fa, fd = frac_in(i, DE.t)
        else:
            result = result.subs(DE.backsubs)
            if not i.is_zero:
                i = NonElementaryIntegral(i.function.subs(DE.backsubs),i.limits)
            if not separate_integral:
                result += i
                return result
            else:

                if isinstance(i, NonElementaryIntegral):
                    return (result, i)
                else:
                    return (result, 0)
<|endoftext|>r"""Module that defines indexed objects.

The classes ``IndexedBase``, ``Indexed``, and ``Idx`` represent a
matrix element ``M[i, j]`` as in the following diagram::

       1) The Indexed class represents the entire indexed object.
                  |
               ___|___
              '       '
               M[i, j]
              /   \__\______
              |             |
              |             |
              |     2) The Idx class represents indices; each Idx can
              |        optionally contain information about its range.
              |
        3) IndexedBase represents the 'stem' of an indexed object, here `M`.
           The stem used by itself is usually taken to represent the entire
           array.

There can be any number of indices on an Indexed object.  No
transformation properties are implemented in these Base objects, but
implicit contraction of repeated indices is supported.

Note that the support for complicated (i.e. non-atomic) integer
expressions as indices is limited.  (This should be improved in
future releases.)

Examples
========

To express the above matrix element example you would write:

>>> from sympy import symbols, IndexedBase, Idx
>>> M = IndexedBase('M')
>>> i, j = symbols('i j', cls=Idx)
>>> M[i, j]
M[i, j]

Repeated indices in a product implies a summation, so to express a
matrix-vector product in terms of Indexed objects:

>>> x = IndexedBase('x')
>>> M[i, j]*x[j]
M[i, j]*x[j]

If the indexed objects will be converted to component based arrays, e.g.
with the code printers or the autowrap framework, you also need to provide
(symbolic or numerical) dimensions.  This can be done by passing an
optional shape parameter to IndexedBase upon construction:

>>> dim1, dim2 = symbols('dim1 dim2', integer=True)
>>> A = IndexedBase('A', shape=(dim1, 2*dim1, dim2))
>>> A.shape
(dim1, 2*dim1, dim2)
>>> A[i, j, 3].shape
(dim1, 2*dim1, dim2)

If an IndexedBase object has no shape information, it is assumed that the
array is as large as the ranges of its indices:

>>> n, m = symbols('n m', integer=True)
>>> i = Idx('i', m)
>>> j = Idx('j', n)
>>> M[i, j].shape
(m, n)
>>> M[i, j].ranges
[(0, m - 1), (0, n - 1)]

The above can be compared with the following:

>>> A[i, 2, j].shape
(dim1, 2*dim1, dim2)
>>> A[i, 2, j].ranges
[(0, m - 1), None, (0, n - 1)]

To analyze the structure of indexed expressions, you can use the methods
get_indices() and get_contraction_structure():

>>> from sympy.tensor import get_indices, get_contraction_structure
>>> get_indices(A[i, j, j])
({i}, {})
>>> get_contraction_structure(A[i, j, j])
{(j,): {A[i, j, j]}}

See the appropriate docstrings for a detailed explanation of the output.
"""

#   TODO:  (some ideas for improvement)
#
#   o test and guarantee numpy compatibility
#      - implement full support for broadcasting
#      - strided arrays
#
#   o more functions to analyze indexed expressions
#      - identify standard constructs, e.g matrix-vector product in a subexpression
#
#   o functions to generate component based arrays (numpy and sympy.Matrix)
#      - generate a single array directly from Indexed
#      - convert simple sub-expressions
#
#   o sophisticated indexing (possibly in subclasses to preserve simplicity)
#      - Idx with range smaller than dimension of Indexed
#      - Idx with stepsize != 1
#      - Idx with step determined by function call
from collections.abc import Iterable

from sympy.core.numbers import Number
from sympy.core.assumptions import StdFactKB
from sympy.core import Expr, Tuple, sympify, S
from sympy.core.symbol import _filter_assumptions, Symbol
from sympy.core.logic import fuzzy_bool, fuzzy_not
from sympy.core.sympify import _sympify
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.multipledispatch import dispatch
from sympy.utilities.iterables import is_sequence, NotIterable
from sympy.utilities.misc import filldedent


class IndexException(Exception):
    pass


class Indexed(Expr):
    """Represents a mathematical object with indices.

    >>> from sympy import Indexed, IndexedBase, Idx, symbols
    >>> i, j = symbols('i j', cls=Idx)
    >>> Indexed('A', i, j)
    A[i, j]

    It is recommended that ``Indexed`` objects be created by indexing ``IndexedBase``:
    ``IndexedBase('A')[i, j]`` instead of ``Indexed(IndexedBase('A'), i, j)``.

    >>> A = IndexedBase('A')
    >>> a_ij = A[i, j]           # Prefer this,
    >>> b_ij = Indexed(A, i, j)  # over this.
    >>> a_ij == b_ij
    True

    """
    is_Indexed = True
    is_symbol = True
    is_Atom = True

    def __new__(cls, base, *args, **kw_args):
        from sympy.tensor.array.ndim_array import NDimArray
        from sympy.matrices.matrixbase import MatrixBase

        if not args:
            raise IndexException("Indexed needs at least one index.")
        if isinstance(base, (str, Symbol)):
            base = IndexedBase(base)
        elif not hasattr(base, '__getitem__') and not isinstance(base, IndexedBase):
            raise TypeError(filldedent("""
                The base can only be replaced with a string, Symbol,
                IndexedBase or an object with a method for getting
                items (i.e. an object with a `__getitem__` method).
                """))
        args = list(map(sympify, args))
        if isinstance(base, (NDimArray, Iterable, Tuple, MatrixBase)) and all(i.is_number for i in args):
            if len(args) == 1:
                return base[args[0]]
            else:
                return base[args]

        base = _sympify(base)

        obj = Expr.__new__(cls, base, *args, **kw_args)

        IndexedBase._set_assumptions(obj, base.assumptions0)

        return obj

    def _hashable_content(self):
        return super()._hashable_content() + tuple(sorted(self.assumptions0.items()))

    @property
    def name(self):
        return str(self)

    @property
    def _diff_wrt(self):
        """Allow derivatives with respect to an ``Indexed`` object."""
        return True

    def _eval_derivative(self, wrt):
        from sympy.tensor.array.ndim_array import NDimArray

        if isinstance(wrt, Indexed) and wrt.base == self.base:
            if len(self.indices) != len(wrt.indices):
                msg = "Different # of indices: d({!s})/d({!s})".format(self,
                                                                       wrt)
                raise IndexException(msg)
            result = S.One
            for index1, index2 in zip(self.indices, wrt.indices):
                result *= KroneckerDelta(index1, index2)
            return result
        elif isinstance(self.base, NDimArray):
            from sympy.tensor.array import derive_by_array
            return Indexed(derive_by_array(self.base, wrt), *self.args[1:])
        else:
            if Tuple(self.indices).has(wrt):
                return S.NaN
            return S.Zero

    @property
    def assumptions0(self):
        return {k: v for k, v in self._assumptions.items() if v is not None}

    @property
    def base(self):
        """Returns the ``IndexedBase`` of the ``Indexed`` object.

        Examples
        ========

        >>> from sympy import Indexed, IndexedBase, Idx, symbols
        >>> i, j = symbols('i j', cls=Idx)
        >>> Indexed('A', i, j).base
        A
        >>> B = IndexedBase('B')
        >>> B == B[i, j].base
        True

        """
        return self.args[0]

    @property
    def indices(self):
        """
        Returns the indices of the ``Indexed`` object.

        Examples
        ========

        >>> from sympy import Indexed, Idx, symbols
        >>> i, j = symbols('i j', cls=Idx)
        >>> Indexed('A', i, j).indices
        (i, j)

        """
        return self.args[1:]

    @property
    def rank(self):
        """
        Returns the rank of the ``Indexed`` object.

        Examples
        ========

        >>> from sympy import Indexed, Idx, symbols
        >>> i, j, k, l, m = symbols('i:m', cls=Idx)
        >>> Indexed('A', i, j).rank
        2
        >>> q = Indexed('A', i, j, k, l, m)
        >>> q.rank
        5
        >>> q.rank == len(q.indices)
        True

        """
        return len(self.args) - 1

    @property
    def shape(self):
        """Returns a list with dimensions of each index.

        Dimensions is a property of the array, not of the indices.  Still, if
        the ``IndexedBase`` does not define a shape attribute, it is assumed
        that the ranges of the indices correspond to the shape of the array.

        >>> from sympy import IndexedBase, Idx, symbols
        >>> n, m = symbols('n m', integer=True)
        >>> i = Idx('i', m)
        >>> j = Idx('j', m)
        >>> A = IndexedBase('A', shape=(n, n))
        >>> B = IndexedBase('B')
        >>> A[i, j].shape
        (n, n)
        >>> B[i, j].shape
        (m, m)
        """

        if self.base.shape:
            return self.base.shape
        sizes = []
        for i in self.indices:
            upper = getattr(i, 'upper', None)
            lower = getattr(i, 'lower', None)
            if None in (upper, lower):
                raise IndexException(filldedent("""
                    Range is not defined for all indices in: %s""" % self))
            try:
                size = upper - lower + 1
            except TypeError:
                raise IndexException(filldedent("""
                    Shape cannot be inferred from Idx with
                    undefined range: %s""" % self))
            sizes.append(size)
        return Tuple(*sizes)

    @property
    def ranges(self):
        """Returns a list of tuples with lower and upper range of each index.

        If an index does not define the data members upper and lower, the
        corresponding slot in the list contains ``None`` instead of a tuple.

        Examples
        ========

        >>> from sympy import Indexed,Idx, symbols
        >>> Indexed('A', Idx('i', 2), Idx('j', 4), Idx('k', 8)).ranges
        [(0, 1), (0, 3), (0, 7)]
        >>> Indexed('A', Idx('i', 3), Idx('j', 3), Idx('k', 3)).ranges
        [(0, 2), (0, 2), (0, 2)]
        >>> x, y, z = symbols('x y z', integer=True)
        >>> Indexed('A', x, y, z).ranges
        [None, None, None]

        """
        ranges = []
        sentinel = object()
        for i in self.indices:
            upper = getattr(i, 'upper', sentinel)
            lower = getattr(i, 'lower', sentinel)
            if sentinel not in (upper, lower):
                ranges.append((lower, upper))
            else:
                ranges.append(None)
        return ranges

    def _sympystr(self, p):
        indices = list(map(p.doprint, self.indices))
        return "%s[%s]" % (p.doprint(self.base), ", ".join(indices))

    @property
    def free_symbols(self):
        base_free_symbols = self.base.free_symbols
        indices_free_symbols = {
            fs for i in self.indices for fs in i.free_symbols}
        if base_free_symbols:
            return {self} | base_free_symbols | indices_free_symbols
        else:
            return indices_free_symbols

    @property
    def expr_free_symbols(self):
        from sympy.utilities.exceptions import sympy_deprecation_warning
        sympy_deprecation_warning("""
        The expr_free_symbols property is deprecated. Use free_symbols to get
        the free symbols of an expression.
        """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-expr-free-symbols")

        return {self}


class IndexedBase(Expr, NotIterable):
    """Represent the base or stem of an indexed object

    The IndexedBase class represent an array that contains elements. The main purpose
    of this class is to allow the convenient creation of objects of the Indexed
    class.  The __getitem__ method of IndexedBase returns an instance of
    Indexed.  Alone, without indices, the IndexedBase class can be used as a
    notation for e.g. matrix equations, resembling what you could do with the
    Symbol class.  But, the IndexedBase class adds functionality that is not
    available for Symbol instances:

      -  An IndexedBase object can optionally store shape information.  This can
         be used in to check array conformance and conditions for numpy
         broadcasting.  (TODO)
      -  An IndexedBase object implements syntactic sugar that allows easy symbolic
         representation of array operations, using implicit summation of
         repeated indices.
      -  The IndexedBase object symbolizes a mathematical structure equivalent
         to arrays, and is recognized as such for code generation and automatic
         compilation and wrapping.

    >>> from sympy.tensor import IndexedBase, Idx
    >>> from sympy import symbols
    >>> A = IndexedBase('A'); A
    A
    >>> type(A)
    <class 'sympy.tensor.indexed.IndexedBase'>

    When an IndexedBase object receives indices, it returns an array with named
    axes, represented by an Indexed object:

    >>> i, j = symbols('i j', integer=True)
    >>> A[i, j, 2]
    A[i, j, 2]
    >>> type(A[i, j, 2])
    <class 'sympy.tensor.indexed.Indexed'>

    The IndexedBase constructor takes an optional shape argument.  If given,
    it overrides any shape information in the indices. (But not the index
    ranges!)

    >>> m, n, o, p = symbols('m n o p', integer=True)
    >>> i = Idx('i', m)
    >>> j = Idx('j', n)
    >>> A[i, j].shape
    (m, n)
    >>> B = IndexedBase('B', shape=(o, p))
    >>> B[i, j].shape
    (o, p)

    Assumptions can be specified with keyword arguments the same way as for Symbol:

    >>> A_real = IndexedBase('A', real=True)
    >>> A_real.is_real
    True
    >>> A != A_real
    True

    Assumptions can also be inherited if a Symbol is used to initialize the IndexedBase:

    >>> I = symbols('I', integer=True)
    >>> C_inherit = IndexedBase(I)
    >>> C_explicit = IndexedBase('I', integer=True)
    >>> C_inherit == C_explicit
    True
    """
    is_symbol = True
    is_Atom = True

    @staticmethod
    def _set_assumptions(obj, assumptions):
        """Set assumptions on obj, making sure to apply consistent values."""
        tmp_asm_copy = assumptions.copy()
        is_commutative = fuzzy_bool(assumptions.get('commutative', True))
        assumptions['commutative'] = is_commutative
        obj._assumptions = StdFactKB(assumptions)
        obj._assumptions._generator = tmp_asm_copy  # Issue #8873

    def __new__(cls, label, shape=None, *, offset=S.Zero, strides=None, **kw_args):
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.tensor.array.ndim_array import NDimArray

        assumptions, kw_args = _filter_assumptions(kw_args)
        if isinstance(label, str):
            label = Symbol(label, **assumptions)
        elif isinstance(label, Symbol):
            assumptions = label._merge(assumptions)
        elif isinstance(label, (MatrixBase, NDimArray)):
            return label
        elif isinstance(label, Iterable):
            return _sympify(label)
        else:
            label = _sympify(label)

        if is_sequence(shape):
            shape = Tuple(*shape)
        elif shape is not None:
            shape = Tuple(shape)

        if shape is not None:
            obj = Expr.__new__(cls, label, shape)
        else:
            obj = Expr.__new__(cls, label)
        obj._shape = shape
        obj._offset = offset
        obj._strides = strides
        obj._name = str(label)

        IndexedBase._set_assumptions(obj, assumptions)
        return obj

    @property
    def name(self):
        return self._name

    def _hashable_content(self):
        return super()._hashable_content() + tuple(sorted(self.assumptions0.items()))

    @property
    def assumptions0(self):
        return {k: v for k, v in self._assumptions.items() if v is not None}

    def __getitem__(self, indices, **kw_args):
        if is_sequence(indices):
            # Special case needed because M[*my_tuple] is a syntax error.
            if self.shape and len(self.shape) != len(indices):
                raise IndexException("Rank mismatch.")
            return Indexed(self, *indices, **kw_args)
        else:
            if self.shape and len(self.shape) != 1:
                raise IndexException("Rank mismatch.")
            return Indexed(self, indices, **kw_args)

    @property
    def shape(self):
        """Returns the shape of the ``IndexedBase`` object.

        Examples
        ========

        >>> from sympy import IndexedBase, Idx
        >>> from sympy.abc import x, y
        >>> IndexedBase('A', shape=(x, y)).shape
        (x, y)

        Note: If the shape of the ``IndexedBase`` is specified, it will override
        any shape information given by the indices.

        >>> A = IndexedBase('A', shape=(x, y))
        >>> B = IndexedBase('B')
        >>> i = Idx('i', 2)
        >>> j = Idx('j', 1)
        >>> A[i, j].shape
        (x, y)
        >>> B[i, j].shape
        (2, 1)

        """
        return self._shape

    @property
    def strides(self):
        """Returns the strided scheme for the ``IndexedBase`` object.

        Normally this is a tuple denoting the number of
        steps to take in the respective dimension when traversing
        an array. For code generation purposes strides='C' and
        strides='F' can also be used.

        strides='C' would mean that code printer would unroll
        in row-major order and 'F' means unroll in column major
        order.

        """

        return self._strides

    @property
    def offset(self):
        """Returns the offset for the ``IndexedBase`` object.

        This is the value added to the resulting index when the
        2D Indexed object is unrolled to a 1D form. Used in code
        generation.

        Examples
        ==========
        >>> from sympy.printing import ccode
        >>> from sympy.tensor import IndexedBase, Idx
        >>> from sympy import symbols
        >>> l, m, n, o = symbols('l m n o', integer=True)
        >>> A = IndexedBase('A', strides=(l, m, n), offset=o)
        >>> i, j, k = map(Idx, 'ijk')
        >>> ccode(A[i, j, k])
        'A[l*i + m*j + n*k + o]'

        """
        return self._offset

    @property
    def label(self):
        """Returns the label of the ``IndexedBase`` object.

        Examples
        ========

        >>> from sympy import IndexedBase
        >>> from sympy.abc import x, y
        >>> IndexedBase('A', shape=(x, y)).label
        A

        """
        return self.args[0]

    def _sympystr(self, p):
        return p.doprint(self.label)


class Idx(Expr):
    """Represents an integer index as an ``Integer`` or integer expression.

    There are a number of ways to create an ``Idx`` object.  The constructor
    takes two arguments:

    ``label``
        An integer or a symbol that labels the index.
    ``range``
        Optionally you can specify a range as either

        * ``Symbol`` or integer: This is interpreted as a dimension. Lower and
          upper bounds are set to ``0`` and ``range - 1``, respectively.
        * ``tuple``: The two elements are interpreted as the lower and upper
          bounds of the range, respectively.

    Note: bounds of the range are assumed to be either integer or infinite (oo
    and -oo are allowed to specify an unbounded range). If ``n`` is given as a
    bound, then ``n.is_integer`` must not return false.

    For convenience, if the label is given as a string it is automatically
    converted to an integer symbol.  (Note: this conversion is not done for
    range or dimension arguments.)

    Examples
    ========

    >>> from sympy import Idx, symbols, oo
    >>> n, i, L, U = symbols('n i L U', integer=True)

    If a string is given for the label an integer ``Symbol`` is created and the
    bounds are both ``None``:

    >>> idx = Idx('qwerty'); idx
    qwerty
    >>> idx.lower, idx.upper
    (None, None)

    Both upper and lower bounds can be specified:

    >>> idx = Idx(i, (L, U)); idx
    i
    >>> idx.lower, idx.upper
    (L, U)

    When only a single bound is given it is interpreted as the dimension
    and the lower bound defaults to 0:

    >>> idx = Idx(i, n); idx.lower, idx.upper
    (0, n - 1)
    >>> idx = Idx(i, 4); idx.lower, idx.upper
    (0, 3)
    >>> idx = Idx(i, oo); idx.lower, idx.upper
    (0, oo)

    """

    is_integer = True
    is_finite = True
    is_real = True
    is_symbol = True
    is_Atom = True
    _diff_wrt = True

    def __new__(cls, label, range=None, **kw_args):

        if isinstance(label, str):
            label = Symbol(label, integer=True)
        label, range = list(map(sympify, (label, range)))

        if label.is_Number:
            if not label.is_integer:
                raise TypeError("Index is not an integer number.")
            return label

        if not label.is_integer:
            raise TypeError("Idx object requires an integer label.")

        elif is_sequence(range):
            if len(range) != 2:
                raise ValueError(filldedent("""
                    Idx range tuple must have length 2, but got %s""" % len(range)))
            for bound in range:
                if (bound.is_integer is False and bound is not S.Infinity
                        and bound is not S.NegativeInfinity):
                    raise TypeError("Idx object requires integer bounds.")
            args = label, Tuple(*range)
        elif isinstance(range, Expr):
            if range is not S.Infinity and fuzzy_not(range.is_integer):
                raise TypeError("Idx object requires an integer dimension.")
            args = label, Tuple(0, range - 1)
        elif range:
            raise TypeError(filldedent("""
                The range must be an ordered iterable or
                integer SymPy expression."""))
        else:
            args = label,

        obj = Expr.__new__(cls, *args, **kw_args)
        obj._assumptions["finite"] = True
        obj._assumptions["real"] = True
        return obj

    @property
    def label(self):
        """Returns the label (Integer or integer expression) of the Idx object.

        Examples
        ========

        >>> from sympy import Idx, Symbol
        >>> x = Symbol('x', integer=True)
        >>> Idx(x).label
        x
        >>> j = Symbol('j', integer=True)
        >>> Idx(j).label
        j
        >>> Idx(j + 1).label
        j + 1

        """
        return self.args[0]

    @property
    def lower(self):
        """Returns the lower bound of the ``Idx``.

        Examples
        ========

        >>> from sympy import Idx
        >>> Idx('j', 2).lower
        0
        >>> Idx('j', 5).lower
        0
        >>> Idx('j').lower is None
        True

        """
        try:
            return self.args[1][0]
        except IndexError:
            return

    @property
    def upper(self):
        """Returns the upper bound of the ``Idx``.

        Examples
        ========

        >>> from sympy import Idx
        >>> Idx('j', 2).upper
        1
        >>> Idx('j', 5).upper
        4
        >>> Idx('j').upper is None
        True

        """
        try:
            return self.args[1][1]
        except IndexError:
            return

    def _sympystr(self, p):
        return p.doprint(self.label)

    @property
    def name(self):
        return self.label.name if self.label.is_Symbol else str(self.label)

    @property
    def free_symbols(self):
        return {self}


@dispatch(Idx, Idx)
def _eval_is_ge(lhs, rhs): # noqa:F811

    other_upper = rhs if rhs.upper is None else rhs.upper
    other_lower = rhs if rhs.lower is None else rhs.lower

    if lhs.lower is not None and (lhs.lower >= other_upper) == True:
        return True
    if lhs.upper is not None and (lhs.upper < other_lower) == True:
        return False
    return None


@dispatch(Idx, Number)  # type:ignore
def _eval_is_ge(lhs, rhs): # noqa:F811

    other_upper = rhs
    other_lower = rhs

    if lhs.lower is not None and (lhs.lower >= other_upper) == True:
        return True
    if lhs.upper is not None and (lhs.upper < other_lower) == True:
        return False
    return None


@dispatch(Number, Idx)  # type:ignore
def _eval_is_ge(lhs, rhs): # noqa:F811

    other_upper = lhs
    other_lower = lhs

    if rhs.upper is not None and (rhs.upper <= other_lower) == True:
        return True
    if rhs.lower is not None and (rhs.lower > other_upper) == True:
        return False
    return None
<|endoftext|>"""Module with functions operating on IndexedBase, Indexed and Idx objects

    - Check shape conformance
    - Determine indices in resulting expression

    etc.

    Methods in this module could be implemented by calling methods on Expr
    objects instead.  When things stabilize this could be a useful
    refactoring.
"""

from functools import reduce

from sympy.core.function import Function
from sympy.functions import exp, Piecewise
from sympy.tensor.indexed import Idx, Indexed
from sympy.utilities import sift

from collections import OrderedDict

class IndexConformanceException(Exception):
    pass

def _unique_and_repeated(inds):
    """
    Returns the unique and repeated indices. Also note, from the examples given below
    that the order of indices is maintained as given in the input.

    Examples
    ========

    >>> from sympy.tensor.index_methods import _unique_and_repeated
    >>> _unique_and_repeated([2, 3, 1, 3, 0, 4, 0])
    ([2, 1, 4], [3, 0])
    """
    uniq = OrderedDict()
    for i in inds:
        if i in uniq:
            uniq[i] = 0
        else:
            uniq[i] = 1
    return sift(uniq, lambda x: uniq[x], binary=True)

def _remove_repeated(inds):
    """
    Removes repeated objects from sequences

    Returns a set of the unique objects and a tuple of all that have been
    removed.

    Examples
    ========

    >>> from sympy.tensor.index_methods import _remove_repeated
    >>> l1 = [1, 2, 3, 2]
    >>> _remove_repeated(l1)
    ({1, 3}, (2,))

    """
    u, r = _unique_and_repeated(inds)
    return set(u), tuple(r)


def _get_indices_Mul(expr, return_dummies=False):
    """Determine the outer indices of a Mul object.

    Examples
    ========

    >>> from sympy.tensor.index_methods import _get_indices_Mul
    >>> from sympy.tensor.indexed import IndexedBase, Idx
    >>> i, j, k = map(Idx, ['i', 'j', 'k'])
    >>> x = IndexedBase('x')
    >>> y = IndexedBase('y')
    >>> _get_indices_Mul(x[i, k]*y[j, k])
    ({i, j}, {})
    >>> _get_indices_Mul(x[i, k]*y[j, k], return_dummies=True)
    ({i, j}, {}, (k,))

    """

    inds = list(map(get_indices, expr.args))
    inds, syms = list(zip(*inds))

    inds = list(map(list, inds))
    inds = list(reduce(lambda x, y: x + y, inds))
    inds, dummies = _remove_repeated(inds)

    symmetry = {}
    for s in syms:
        for pair in s:
            if pair in symmetry:
                symmetry[pair] *= s[pair]
            else:
                symmetry[pair] = s[pair]

    if return_dummies:
        return inds, symmetry, dummies
    else:
        return inds, symmetry


def _get_indices_Pow(expr):
    """Determine outer indices of a power or an exponential.

    A power is considered a universal function, so that the indices of a Pow is
    just the collection of indices present in the expression.  This may be
    viewed as a bit inconsistent in the special case:

        x[i]**2 = x[i]*x[i]                                                      (1)

    The above expression could have been interpreted as the contraction of x[i]
    with itself, but we choose instead to interpret it as a function

        lambda y: y**2

    applied to each element of x (a universal function in numpy terms).  In
    order to allow an interpretation of (1) as a contraction, we need
    contravariant and covariant Idx subclasses.  (FIXME: this is not yet
    implemented)

    Expressions in the base or exponent are subject to contraction as usual,
    but an index that is present in the exponent, will not be considered
    contractable with its own base.  Note however, that indices in the same
    exponent can be contracted with each other.

    Examples
    ========

    >>> from sympy.tensor.index_methods import _get_indices_Pow
    >>> from sympy import Pow, exp, IndexedBase, Idx
    >>> A = IndexedBase('A')
    >>> x = IndexedBase('x')
    >>> i, j, k = map(Idx, ['i', 'j', 'k'])
    >>> _get_indices_Pow(exp(A[i, j]*x[j]))
    ({i}, {})
    >>> _get_indices_Pow(Pow(x[i], x[i]))
    ({i}, {})
    >>> _get_indices_Pow(Pow(A[i, j]*x[j], x[i]))
    ({i}, {})

    """
    base, exp = expr.as_base_exp()
    binds, bsyms = get_indices(base)
    einds, esyms = get_indices(exp)

    inds = binds | einds

    # FIXME: symmetries from power needs to check special cases, else nothing
    symmetries = {}

    return inds, symmetries


def _get_indices_Add(expr):
    """Determine outer indices of an Add object.

    In a sum, each term must have the same set of outer indices.  A valid
    expression could be

        x(i)*y(j) - x(j)*y(i)

    But we do not allow expressions like:

        x(i)*y(j) - z(j)*z(j)

    FIXME: Add support for Numpy broadcasting

    Examples
    ========

    >>> from sympy.tensor.index_methods import _get_indices_Add
    >>> from sympy.tensor.indexed import IndexedBase, Idx
    >>> i, j, k = map(Idx, ['i', 'j', 'k'])
    >>> x = IndexedBase('x')
    >>> y = IndexedBase('y')
    >>> _get_indices_Add(x[i] + x[k]*y[i, k])
    ({i}, {})

    """

    inds = list(map(get_indices, expr.args))
    inds, syms = list(zip(*inds))

    # allow broadcast of scalars
    non_scalars = [x for x in inds if x != set()]
    if not non_scalars:
        return set(), {}

    if not all(x == non_scalars[0] for x in non_scalars[1:]):
        raise IndexConformanceException("Indices are not consistent: %s" % expr)
    if not reduce(lambda x, y: x != y or y, syms):
        symmetries = syms[0]
    else:
        # FIXME: search for symmetries
        symmetries = {}

    return non_scalars[0], symmetries


def get_indices(expr):
    """Determine the outer indices of expression ``expr``

    By *outer* we mean indices that are not summation indices.  Returns a set
    and a dict.  The set contains outer indices and the dict contains
    information about index symmetries.

    Examples
    ========

    >>> from sympy.tensor.index_methods import get_indices
    >>> from sympy import symbols
    >>> from sympy.tensor import IndexedBase
    >>> x, y, A = map(IndexedBase, ['x', 'y', 'A'])
    >>> i, j, a, z = symbols('i j a z', integer=True)

    The indices of the total expression is determined, Repeated indices imply a
    summation, for instance the trace of a matrix A:

    >>> get_indices(A[i, i])
    (set(), {})

    In the case of many terms, the terms are required to have identical
    outer indices.  Else an IndexConformanceException is raised.

    >>> get_indices(x[i] + A[i, j]*y[j])
    ({i}, {})

    :Exceptions:

    An IndexConformanceException means that the terms ar not compatible, e.g.

    >>> get_indices(x[i] + y[j])                #doctest: +SKIP
            (...)
    IndexConformanceException: Indices are not consistent: x(i) + y(j)

    .. warning::
       The concept of *outer* indices applies recursively, starting on the deepest
       level.  This implies that dummies inside parenthesis are assumed to be
       summed first, so that the following expression is handled gracefully:

       >>> get_indices((x[i] + A[i, j]*y[j])*x[j])
       ({i, j}, {})

       This is correct and may appear convenient, but you need to be careful
       with this as SymPy will happily .expand() the product, if requested.  The
       resulting expression would mix the outer ``j`` with the dummies inside
       the parenthesis, which makes it a different expression.  To be on the
       safe side, it is best to avoid such ambiguities by using unique indices
       for all contractions that should be held separate.

    """
    # We call ourself recursively to determine indices of sub expressions.

    # break recursion
    if isinstance(expr, Indexed):
        c = expr.indices
        inds, dummies = _remove_repeated(c)
        return inds, {}
    elif expr is None:
        return set(), {}
    elif isinstance(expr, Idx):
        return {expr}, {}
    elif expr.is_Atom:
        return set(), {}


    # recurse via specialized functions
    else:
        if expr.is_Mul:
            return _get_indices_Mul(expr)
        elif expr.is_Add:
            return _get_indices_Add(expr)
        elif expr.is_Pow or isinstance(expr, exp):
            return _get_indices_Pow(expr)

        elif isinstance(expr, Piecewise):
            # FIXME:  No support for Piecewise yet
            return set(), {}
        elif isinstance(expr, Function):
            # Support ufunc like behaviour by returning indices from arguments.
            # Functions do not interpret repeated indices across arguments
            # as summation
            ind0 = set()
            for arg in expr.args:
                ind, sym = get_indices(arg)
                ind0 |= ind
            return ind0, sym

        # this test is expensive, so it should be at the end
        elif not expr.has(Indexed):
            return set(), {}
        raise NotImplementedError(
            "FIXME: No specialized handling of type %s" % type(expr))


def get_contraction_structure(expr):
    """Determine dummy indices of ``expr`` and describe its structure

    By *dummy* we mean indices that are summation indices.

    The structure of the expression is determined and described as follows:

    1) A conforming summation of Indexed objects is described with a dict where
       the keys are summation indices and the corresponding values are sets
       containing all terms for which the summation applies.  All Add objects
       in the SymPy expression tree are described like this.

    2) For all nodes in the SymPy expression tree that are *not* of type Add, the
       following applies:

       If a node discovers contractions in one of its arguments, the node
       itself will be stored as a key in the dict.  For that key, the
       corresponding value is a list of dicts, each of which is the result of a
       recursive call to get_contraction_structure().  The list contains only
       dicts for the non-trivial deeper contractions, omitting dicts with None
       as the one and only key.

    .. Note:: The presence of expressions among the dictionary keys indicates
       multiple levels of index contractions.  A nested dict displays nested
       contractions and may itself contain dicts from a deeper level.  In
       practical calculations the summation in the deepest nested level must be
       calculated first so that the outer expression can access the resulting
       indexed object.

    Examples
    ========

    >>> from sympy.tensor.index_methods import get_contraction_structure
    >>> from sympy import default_sort_key
    >>> from sympy.tensor import IndexedBase, Idx
    >>> x, y, A = map(IndexedBase, ['x', 'y', 'A'])
    >>> i, j, k, l = map(Idx, ['i', 'j', 'k', 'l'])
    >>> get_contraction_structure(x[i]*y[i] + A[j, j])
    {(i,): {x[i]*y[i]}, (j,): {A[j, j]}}
    >>> get_contraction_structure(x[i]*y[j])
    {None: {x[i]*y[j]}}

    A multiplication of contracted factors results in nested dicts representing
    the internal contractions.

    >>> d = get_contraction_structure(x[i, i]*y[j, j])
    >>> sorted(d.keys(), key=default_sort_key)
    [None, x[i, i]*y[j, j]]

    In this case, the product has no contractions:

    >>> d[None]
    {x[i, i]*y[j, j]}

    Factors are contracted "first":

    >>> sorted(d[x[i, i]*y[j, j]], key=default_sort_key)
    [{(i,): {x[i, i]}}, {(j,): {y[j, j]}}]

    A parenthesized Add object is also returned as a nested dictionary.  The
    term containing the parenthesis is a Mul with a contraction among the
    arguments, so it will be found as a key in the result.  It stores the
    dictionary resulting from a recursive call on the Add expression.

    >>> d = get_contraction_structure(x[i]*(y[i] + A[i, j]*x[j]))
    >>> sorted(d.keys(), key=default_sort_key)
    [(A[i, j]*x[j] + y[i])*x[i], (i,)]
    >>> d[(i,)]
    {(A[i, j]*x[j] + y[i])*x[i]}
    >>> d[x[i]*(A[i, j]*x[j] + y[i])]
    [{None: {y[i]}, (j,): {A[i, j]*x[j]}}]

    Powers with contractions in either base or exponent will also be found as
    keys in the dictionary, mapping to a list of results from recursive calls:

    >>> d = get_contraction_structure(A[j, j]**A[i, i])
    >>> d[None]
    {A[j, j]**A[i, i]}
    >>> nested_contractions = d[A[j, j]**A[i, i]]
    >>> nested_contractions[0]
    {(j,): {A[j, j]}}
    >>> nested_contractions[1]
    {(i,): {A[i, i]}}

    The description of the contraction structure may appear complicated when
    represented with a string in the above examples, but it is easy to iterate
    over:

    >>> from sympy import Expr
    >>> for key in d:
    ...     if isinstance(key, Expr):
    ...         continue
    ...     for term in d[key]:
    ...         if term in d:
    ...             # treat deepest contraction first
    ...             pass
    ...     # treat outermost contactions here

    """

    # We call ourself recursively to inspect sub expressions.

    if isinstance(expr, Indexed):
        junk, key = _remove_repeated(expr.indices)
        return {key or None: {expr}}
    elif expr.is_Atom:
        return {None: {expr}}
    elif expr.is_Mul:
        junk, junk, key = _get_indices_Mul(expr, return_dummies=True)
        result = {key or None: {expr}}
        # recurse on every factor
        nested = []
        for fac in expr.args:
            facd = get_contraction_structure(fac)
            if not (None in facd and len(facd) == 1):
                nested.append(facd)
        if nested:
            result[expr] = nested
        return result
    elif expr.is_Pow or isinstance(expr, exp):
        # recurse in base and exp separately.  If either has internal
        # contractions we must include ourselves as a key in the returned dict
        b, e = expr.as_base_exp()
        dbase = get_contraction_structure(b)
        dexp = get_contraction_structure(e)

        dicts = []
        for d in dbase, dexp:
            if not (None in d and len(d) == 1):
                dicts.append(d)
        result = {None: {expr}}
        if dicts:
            result[expr] = dicts
        return result
    elif expr.is_Add:
        # Note: we just collect all terms with identical summation indices, We
        # do nothing to identify equivalent terms here, as this would require
        # substitutions or pattern matching in expressions of unknown
        # complexity.
        result = {}
        for term in expr.args:
            # recurse on every term
            d = get_contraction_structure(term)
            for key in d:
                if key in result:
                    result[key] |= d[key]
                else:
                    result[key] = d[key]
        return result

    elif isinstance(expr, Piecewise):
        # FIXME:  No support for Piecewise yet
        return {None: expr}
    elif isinstance(expr, Function):
        # Collect non-trivial contraction structures in each argument
        # We do not report repeated indices in separate arguments as a
        # contraction
        deeplist = []
        for arg in expr.args:
            deep = get_contraction_structure(arg)
            if not (None in deep and len(deep) == 1):
                deeplist.append(deep)
        d = {None: {expr}}
        if deeplist:
            d[expr] = deeplist
        return d

    # this test is expensive, so it should be at the end
    elif not expr.has(Indexed):
        return {None: {expr}}
    raise NotImplementedError(
        "FIXME: No specialized handling of type %s" % type(expr))
<|endoftext|>from sympy import permutedims
from sympy.core.numbers import Number
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.tensor.tensor import Tensor, TensExpr, TensAdd, TensMul


class PartialDerivative(TensExpr):
    """
    Partial derivative for tensor expressions.

    Examples
    ========

    >>> from sympy.tensor.tensor import TensorIndexType, TensorHead
    >>> from sympy.tensor.toperators import PartialDerivative
    >>> from sympy import symbols
    >>> L = TensorIndexType("L")
    >>> A = TensorHead("A", [L])
    >>> B = TensorHead("B", [L])
    >>> i, j, k = symbols("i j k")

    >>> expr = PartialDerivative(A(i), A(j))
    >>> expr
    PartialDerivative(A(i), A(j))

    The ``PartialDerivative`` object behaves like a tensorial expression:

    >>> expr.get_indices()
    [i, -j]

    Notice that the deriving variables have opposite valence than the
    printed one: ``A(j)`` is printed as covariant, but the index of the
    derivative is actually contravariant, i.e. ``-j``.

    Indices can be contracted:

    >>> expr = PartialDerivative(A(i), A(i))
    >>> expr
    PartialDerivative(A(L_0), A(L_0))
    >>> expr.get_indices()
    [L_0, -L_0]

    The method ``.get_indices()`` always returns all indices (even the
    contracted ones). If only uncontracted indices are needed, call
    ``.get_free_indices()``:

    >>> expr.get_free_indices()
    []

    Nested partial derivatives are flattened:

    >>> expr = PartialDerivative(PartialDerivative(A(i), A(j)), A(k))
    >>> expr
    PartialDerivative(A(i), A(j), A(k))
    >>> expr.get_indices()
    [i, -j, -k]

    Replace a derivative with array values:

    >>> from sympy.abc import x, y
    >>> from sympy import sin, log
    >>> compA = [sin(x), log(x)*y**3]
    >>> compB = [x, y]
    >>> expr = PartialDerivative(A(i), B(j))
    >>> expr.replace_with_arrays({A(i): compA, B(i): compB})
    [[cos(x), 0], [y**3/x, 3*y**2*log(x)]]

    The returned array is indexed by `(i, -j)`.

    Be careful that other SymPy modules put the indices of the deriving
    variables before the indices of the derivand in the derivative result.
    For example:

    >>> expr.get_free_indices()
    [i, -j]

    >>> from sympy import Matrix, Array
    >>> Matrix(compA).diff(Matrix(compB)).reshape(2, 2)
    [[cos(x), y**3/x], [0, 3*y**2*log(x)]]
    >>> Array(compA).diff(Array(compB))
    [[cos(x), y**3/x], [0, 3*y**2*log(x)]]

    These are the transpose of the result of ``PartialDerivative``,
    as the matrix and the array modules put the index `-j` before `i` in the
    derivative result. An array read with index order `(-j, i)` is indeed the
    transpose of the same array read with index order `(i, -j)`. By specifying
    the index order to ``.replace_with_arrays`` one can get a compatible
    expression:

    >>> expr.replace_with_arrays({A(i): compA, B(i): compB}, [-j, i])
    [[cos(x), y**3/x], [0, 3*y**2*log(x)]]
    """

    def __new__(cls, expr, *variables):

        # Flatten:
        if isinstance(expr, PartialDerivative):
            variables = expr.variables + variables
            expr = expr.expr

        args, indices, free, dum = cls._contract_indices_for_derivative(
            S(expr), variables)

        obj = TensExpr.__new__(cls, *args)

        obj._indices = indices
        obj._free = free
        obj._dum = dum
        return obj

    @property
    def coeff(self):
        return S.One

    @property
    def nocoeff(self):
        return self

    @classmethod
    def _contract_indices_for_derivative(cls, expr, variables):
        variables_opposite_valence = []

        for i in variables:
            if isinstance(i, Tensor):
                i_free_indices = i.get_free_indices()
                variables_opposite_valence.append(
                        i.xreplace({k: -k for k in i_free_indices}))
            elif isinstance(i, Symbol):
                variables_opposite_valence.append(i)

        args, indices, free, dum = TensMul._tensMul_contract_indices(
            [expr] + variables_opposite_valence, replace_indices=True)

        for i in range(1, len(args)):
            args_i = args[i]
            if isinstance(args_i, Tensor):
                i_indices = args[i].get_free_indices()
                args[i] = args[i].xreplace({k: -k for k in i_indices})

        return args, indices, free, dum

    def doit(self, **hints):
        args, indices, free, dum = self._contract_indices_for_derivative(self.expr, self.variables)

        obj = self.func(*args)
        obj._indices = indices
        obj._free = free
        obj._dum = dum

        return obj

    def _expand_partial_derivative(self):
        args, indices, free, dum = self._contract_indices_for_derivative(self.expr, self.variables)

        obj = self.func(*args)
        obj._indices = indices
        obj._free = free
        obj._dum = dum

        result = obj

        if not args[0].free_symbols:
            return S.Zero
        elif isinstance(obj.expr, TensAdd):
            # take care of sums of multi PDs
            result = obj.expr.func(*[
                    self.func(a, *obj.variables)._expand_partial_derivative()
                    for a in result.expr.args])
        elif isinstance(obj.expr, TensMul):
            # take care of products of multi PDs
            if len(obj.variables) == 1:
                # derivative with respect to single variable
                terms = []
                mulargs = list(obj.expr.args)
                for ind in range(len(mulargs)):
                    if not isinstance(sympify(mulargs[ind]), Number):
                        # a number coefficient is not considered for
                        # expansion of PartialDerivative
                        d = self.func(mulargs[ind], *obj.variables)._expand_partial_derivative()
                        terms.append(TensMul(*(mulargs[:ind]
                                               + [d]
                                               + mulargs[(ind + 1):])))
                result = TensAdd.fromiter(terms)
            else:
                # derivative with respect to multiple variables
                # decompose:
                # partial(expr, (u, v))
                # = partial(partial(expr, u).doit(), v).doit()
                result = obj.expr  # init with expr
                for v in obj.variables:
                    result = self.func(result, v)._expand_partial_derivative()
                    # then throw PD on it

        return result

    def _perform_derivative(self):
        result = self.expr
        for v in self.variables:
            if isinstance(result, TensExpr):
                result = result._eval_partial_derivative(v)
            else:
                if v._diff_wrt:
                    result = result._eval_derivative(v)
                else:
                    result = S.Zero
        return result

    def get_indices(self):
        return self._indices

    def get_free_indices(self):
        free = sorted(self._free, key=lambda x: x[1])
        return [i[0] for i in free]

    def _replace_indices(self, repl):
        expr = self.expr.xreplace(repl)
        mirrored = {-k: -v for k, v in repl.items()}
        variables = [i.xreplace(mirrored) for i in self.variables]
        return self.func(expr, *variables)

    @property
    def expr(self):
        return self.args[0]

    @property
    def variables(self):
        return self.args[1:]

    def _extract_data(self, replacement_dict):
        from .array import derive_by_array, tensorcontraction
        indices, array = self.expr._extract_data(replacement_dict)
        for variable in self.variables:
            var_indices, var_array = variable._extract_data(replacement_dict)
            var_indices = [-i for i in var_indices]
            coeff_array, var_array = zip(*[i.as_coeff_Mul() for i in var_array])
            dim_before = len(array.shape)
            array = derive_by_array(array, var_array)
            dim_after = len(array.shape)
            dim_increase = dim_after - dim_before
            array = permutedims(array, [i + dim_increase for i in range(dim_before)] + list(range(dim_increase)))
            array = array.as_mutable()
            varindex = var_indices[0]
            # Remove coefficients of base vector:
            coeff_index = [0] + [slice(None) for i in range(len(indices))]
            for i, coeff in enumerate(coeff_array):
                coeff_index[0] = i
                array[tuple(coeff_index)] /= coeff
            if -varindex in indices:
                pos = indices.index(-varindex)
                array = tensorcontraction(array, (0, pos+1))
                indices.pop(pos)
            else:
                indices.append(varindex)
        return indices, array
<|endoftext|>"""A module to manipulate symbolic objects with indices including tensors

"""
from .indexed import IndexedBase, Idx, Indexed
from .index_methods import get_contraction_structure, get_indices
from .functions import shape
from .array import (MutableDenseNDimArray, ImmutableDenseNDimArray,
    MutableSparseNDimArray, ImmutableSparseNDimArray, NDimArray, tensorproduct,
    tensorcontraction, tensordiagonal, derive_by_array, permutedims, Array,
    DenseNDimArray, SparseNDimArray,)

__all__ = [
    'IndexedBase', 'Idx', 'Indexed',

    'get_contraction_structure', 'get_indices',

    'shape',

    'MutableDenseNDimArray', 'ImmutableDenseNDimArray',
    'MutableSparseNDimArray', 'ImmutableSparseNDimArray', 'NDimArray',
    'tensorproduct', 'tensorcontraction', 'tensordiagonal', 'derive_by_array', 'permutedims',
    'Array', 'DenseNDimArray', 'SparseNDimArray',
]
<|endoftext|>"""
This module defines tensors with abstract index notation.

The abstract index notation has been first formalized by Penrose.

Tensor indices are formal objects, with a tensor type; there is no
notion of index range, it is only possible to assign the dimension,
used to trace the Kronecker delta; the dimension can be a Symbol.

The Einstein summation convention is used.
The covariant indices are indicated with a minus sign in front of the index.

For instance the tensor ``t = p(a)*A(b,c)*q(-c)`` has the index ``c``
contracted.

A tensor expression ``t`` can be called; called with its
indices in sorted order it is equal to itself:
in the above example ``t(a, b) == t``;
one can call ``t`` with different indices; ``t(c, d) == p(c)*A(d,a)*q(-a)``.

The contracted indices are dummy indices, internally they have no name,
the indices being represented by a graph-like structure.

Tensors are put in canonical form using ``canon_bp``, which uses
the Butler-Portugal algorithm for canonicalization using the monoterm
symmetries of the tensors.

If there is a (anti)symmetric metric, the indices can be raised and
lowered when the tensor is put in canonical form.
"""

from __future__ import annotations
from typing import Any
from functools import reduce
from math import prod

from abc import abstractmethod, ABC
from collections import defaultdict
import operator
import itertools

from sympy.core.numbers import (Integer, Rational)
from sympy.combinatorics import Permutation
from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, \
    bsgs_direct_product, canonicalize, riemann_bsgs
from sympy.core import Basic, Expr, sympify, Add, Mul, S
from sympy.core.cache import clear_cache
from sympy.core.containers import Tuple, Dict
from sympy.core.function import WildFunction
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol, symbols, Wild
from sympy.core.sympify import CantSympify, _sympify
from sympy.core.operations import AssocOp
from sympy.external.gmpy import SYMPY_INTS
from sympy.matrices import eye
from sympy.utilities.exceptions import (sympy_deprecation_warning,
                                        SymPyDeprecationWarning,
                                        ignore_warnings)
from sympy.utilities.decorator import memoize_property, deprecated
from sympy.utilities.iterables import sift


def deprecate_data():
    sympy_deprecation_warning(
        """
        The data attribute of TensorIndexType is deprecated. Use The
        replace_with_arrays() method instead.
        """,
        deprecated_since_version="1.4",
        active_deprecations_target="deprecated-tensorindextype-attrs",
        stacklevel=4,
    )

def deprecate_fun_eval():
    sympy_deprecation_warning(
        """
        The Tensor.fun_eval() method is deprecated. Use
        Tensor.substitute_indices() instead.
        """,
        deprecated_since_version="1.5",
        active_deprecations_target="deprecated-tensor-fun-eval",
        stacklevel=4,
    )


def deprecate_call():
    sympy_deprecation_warning(
        """
        Calling a tensor like Tensor(*indices) is deprecated. Use
        Tensor.substitute_indices() instead.
        """,
        deprecated_since_version="1.5",
        active_deprecations_target="deprecated-tensor-fun-eval",
        stacklevel=4,
    )


class _IndexStructure(CantSympify):
    """
    This class handles the indices (free and dummy ones). It contains the
    algorithms to manage the dummy indices replacements and contractions of
    free indices under multiplications of tensor expressions, as well as stuff
    related to canonicalization sorting, getting the permutation of the
    expression and so on. It also includes tools to get the ``TensorIndex``
    objects corresponding to the given index structure.
    """

    def __init__(self, free, dum, index_types, indices, canon_bp=False):
        self.free = free
        self.dum = dum
        self.index_types = index_types
        self.indices = indices
        self._ext_rank = len(self.free) + 2*len(self.dum)
        self.dum.sort(key=lambda x: x[0])

    @staticmethod
    def from_indices(*indices):
        """
        Create a new ``_IndexStructure`` object from a list of ``indices``.

        Explanation
        ===========

        ``indices``     ``TensorIndex`` objects, the indices. Contractions are
                        detected upon construction.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, _IndexStructure
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> m0, m1, m2, m3 = tensor_indices('m0,m1,m2,m3', Lorentz)
        >>> _IndexStructure.from_indices(m0, m1, -m1, m3)
        _IndexStructure([(m0, 0), (m3, 3)], [(1, 2)], [Lorentz, Lorentz, Lorentz, Lorentz])
        """

        free, dum = _IndexStructure._free_dum_from_indices(*indices)
        index_types = [i.tensor_index_type for i in indices]
        indices = _IndexStructure._replace_dummy_names(indices, free, dum)
        return _IndexStructure(free, dum, index_types, indices)

    @staticmethod
    def from_components_free_dum(components, free, dum):
        index_types = []
        for component in components:
            index_types.extend(component.index_types)
        indices = _IndexStructure.generate_indices_from_free_dum_index_types(free, dum, index_types)
        return _IndexStructure(free, dum, index_types, indices)

    @staticmethod
    def _free_dum_from_indices(*indices):
        """
        Convert ``indices`` into ``free``, ``dum`` for single component tensor.

        Explanation
        ===========

        ``free``     list of tuples ``(index, pos, 0)``,
                     where ``pos`` is the position of index in
                     the list of indices formed by the component tensors

        ``dum``      list of tuples ``(pos_contr, pos_cov, 0, 0)``

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, \
            _IndexStructure
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> m0, m1, m2, m3 = tensor_indices('m0,m1,m2,m3', Lorentz)
        >>> _IndexStructure._free_dum_from_indices(m0, m1, -m1, m3)
        ([(m0, 0), (m3, 3)], [(1, 2)])
        """
        n = len(indices)
        if n == 1:
            return [(indices[0], 0)], []

        # find the positions of the free indices and of the dummy indices
        free = [True]*len(indices)
        index_dict = {}
        dum = []
        for i, index in enumerate(indices):
            name = index.name
            typ = index.tensor_index_type
            contr = index.is_up
            if (name, typ) in index_dict:
                # found a pair of dummy indices
                is_contr, pos = index_dict[(name, typ)]
                # check consistency and update free
                if is_contr:
                    if contr:
                        raise ValueError('two equal contravariant indices in slots %d and %d' %(pos, i))
                    else:
                        free[pos] = False
                        free[i] = False
                else:
                    if contr:
                        free[pos] = False
                        free[i] = False
                    else:
                        raise ValueError('two equal covariant indices in slots %d and %d' %(pos, i))
                if contr:
                    dum.append((i, pos))
                else:
                    dum.append((pos, i))
            else:
                index_dict[(name, typ)] = index.is_up, i

        free = [(index, i) for i, index in enumerate(indices) if free[i]]
        free.sort()
        return free, dum

    def get_indices(self):
        """
        Get a list of indices, creating new tensor indices to complete dummy indices.
        """
        return self.indices[:]

    @staticmethod
    def generate_indices_from_free_dum_index_types(free, dum, index_types):
        indices = [None]*(len(free)+2*len(dum))
        for idx, pos in free:
            indices[pos] = idx

        generate_dummy_name = _IndexStructure._get_generator_for_dummy_indices(free)
        for pos1, pos2 in dum:
            typ1 = index_types[pos1]
            indname = generate_dummy_name(typ1)
            indices[pos1] = TensorIndex(indname, typ1, True)
            indices[pos2] = TensorIndex(indname, typ1, False)

        return _IndexStructure._replace_dummy_names(indices, free, dum)

    @staticmethod
    def _get_generator_for_dummy_indices(free):
        cdt = defaultdict(int)
        # if the free indices have names with dummy_name, start with an
        # index higher than those for the dummy indices
        # to avoid name collisions
        for indx, ipos in free:
            if indx.name.split('_')[0] == indx.tensor_index_type.dummy_name:
                cdt[indx.tensor_index_type] = max(cdt[indx.tensor_index_type], int(indx.name.split('_')[1]) + 1)

        def dummy_name_gen(tensor_index_type):
            nd = str(cdt[tensor_index_type])
            cdt[tensor_index_type] += 1
            return tensor_index_type.dummy_name + '_' + nd

        return dummy_name_gen

    @staticmethod
    def _replace_dummy_names(indices, free, dum):
        dum.sort(key=lambda x: x[0])
        new_indices = list(indices)
        assert len(indices) == len(free) + 2*len(dum)
        generate_dummy_name = _IndexStructure._get_generator_for_dummy_indices(free)
        for ipos1, ipos2 in dum:
            typ1 = new_indices[ipos1].tensor_index_type
            indname = generate_dummy_name(typ1)
            new_indices[ipos1] = TensorIndex(indname, typ1, True)
            new_indices[ipos2] = TensorIndex(indname, typ1, False)
        return new_indices

    def get_free_indices(self) -> list[TensorIndex]:
        """
        Get a list of free indices.
        """
        # get sorted indices according to their position:
        free = sorted(self.free, key=lambda x: x[1])
        return [i[0] for i in free]

    def __str__(self):
        return "_IndexStructure({}, {}, {})".format(self.free, self.dum, self.index_types)

    def __repr__(self):
        return self.__str__()

    def _get_sorted_free_indices_for_canon(self):
        sorted_free = self.free[:]
        sorted_free.sort(key=lambda x: x[0])
        return sorted_free

    def _get_sorted_dum_indices_for_canon(self):
        return sorted(self.dum, key=lambda x: x[0])

    def _get_lexicographically_sorted_index_types(self):
        permutation = self.indices_canon_args()[0]
        index_types = [None]*self._ext_rank
        for i, it in enumerate(self.index_types):
            index_types[permutation(i)] = it
        return index_types

    def _get_lexicographically_sorted_indices(self):
        permutation = self.indices_canon_args()[0]
        indices = [None]*self._ext_rank
        for i, it in enumerate(self.indices):
            indices[permutation(i)] = it
        return indices

    def perm2tensor(self, g, is_canon_bp=False):
        """
        Returns a ``_IndexStructure`` instance corresponding to the permutation ``g``.

        Explanation
        ===========

        ``g``  permutation corresponding to the tensor in the representation
        used in canonicalization

        ``is_canon_bp``   if True, then ``g`` is the permutation
        corresponding to the canonical form of the tensor
        """
        sorted_free = [i[0] for i in self._get_sorted_free_indices_for_canon()]
        lex_index_types = self._get_lexicographically_sorted_index_types()
        lex_indices = self._get_lexicographically_sorted_indices()
        nfree = len(sorted_free)
        rank = self._ext_rank
        dum = [[None]*2 for i in range((rank - nfree)//2)]
        free = []

        index_types = [None]*rank
        indices = [None]*rank
        for i in range(rank):
            gi = g[i]
            index_types[i] = lex_index_types[gi]
            indices[i] = lex_indices[gi]
            if gi < nfree:
                ind = sorted_free[gi]
                assert index_types[i] == sorted_free[gi].tensor_index_type
                free.append((ind, i))
            else:
                j = gi - nfree
                idum, cov = divmod(j, 2)
                if cov:
                    dum[idum][1] = i
                else:
                    dum[idum][0] = i
        dum = [tuple(x) for x in dum]

        return _IndexStructure(free, dum, index_types, indices)

    def indices_canon_args(self):
        """
        Returns ``(g, dummies, msym, v)``, the entries of ``canonicalize``

        See ``canonicalize`` in ``tensor_can.py`` in combinatorics module.
        """
        # to be called after sorted_components
        from sympy.combinatorics.permutations import _af_new
        n = self._ext_rank
        g = [None]*n + [n, n+1]

        # Converts the symmetry of the metric into msym from .canonicalize()
        # method in the combinatorics module
        def metric_symmetry_to_msym(metric):
            if metric is None:
                return None
            sym = metric.symmetry
            if sym == TensorSymmetry.fully_symmetric(2):
                return 0
            if sym == TensorSymmetry.fully_symmetric(-2):
                return 1
            return None

        # ordered indices: first the free indices, ordered by types
        # then the dummy indices, ordered by types and contravariant before
        # covariant
        # g[position in tensor] = position in ordered indices
        for i, (indx, ipos) in enumerate(self._get_sorted_free_indices_for_canon()):
            g[ipos] = i
        pos = len(self.free)
        j = len(self.free)
        dummies = []
        prev = None
        a = []
        msym = []
        for ipos1, ipos2 in self._get_sorted_dum_indices_for_canon():
            g[ipos1] = j
            g[ipos2] = j + 1
            j += 2
            typ = self.index_types[ipos1]
            if typ != prev:
                if a:
                    dummies.append(a)
                a = [pos, pos + 1]
                prev = typ
                msym.append(metric_symmetry_to_msym(typ.metric))
            else:
                a.extend([pos, pos + 1])
            pos += 2
        if a:
            dummies.append(a)

        return _af_new(g), dummies, msym


def components_canon_args(components):
    numtyp = []
    prev = None
    for t in components:
        if t == prev:
            numtyp[-1][1] += 1
        else:
            prev = t
            numtyp.append([prev, 1])
    v = []
    for h, n in numtyp:
        if h.comm in (0, 1):
            comm = h.comm
        else:
            comm = TensorManager.get_comm(h.comm, h.comm)
        v.append((h.symmetry.base, h.symmetry.generators, n, comm))
    return v


class _TensorDataLazyEvaluator(CantSympify):
    """
    EXPERIMENTAL: do not rely on this class, it may change without deprecation
    warnings in future versions of SymPy.

    Explanation
    ===========

    This object contains the logic to associate components data to a tensor
    expression. Components data are set via the ``.data`` property of tensor
    expressions, is stored inside this class as a mapping between the tensor
    expression and the ``ndarray``.

    Computations are executed lazily: whereas the tensor expressions can have
    contractions, tensor products, and additions, components data are not
    computed until they are accessed by reading the ``.data`` property
    associated to the tensor expression.
    """
    _substitutions_dict: dict[Any, Any] = {}
    _substitutions_dict_tensmul: dict[Any, Any] = {}

    def __getitem__(self, key):
        dat = self._get(key)
        if dat is None:
            return None

        from .array import NDimArray
        if not isinstance(dat, NDimArray):
            return dat

        if dat.rank() == 0:
            return dat[()]
        elif dat.rank() == 1 and len(dat) == 1:
            return dat[0]
        return dat

    def _get(self, key):
        """
        Retrieve ``data`` associated with ``key``.

        Explanation
        ===========

        This algorithm looks into ``self._substitutions_dict`` for all
        ``TensorHead`` in the ``TensExpr`` (or just ``TensorHead`` if key is a
        TensorHead instance). It reconstructs the components data that the
        tensor expression should have by performing on components data the
        operations that correspond to the abstract tensor operations applied.

        Metric tensor is handled in a different manner: it is pre-computed in
        ``self._substitutions_dict_tensmul``.
        """
        if key in self._substitutions_dict:
            return self._substitutions_dict[key]

        if isinstance(key, TensorHead):
            return None

        if isinstance(key, Tensor):
            # special case to handle metrics. Metric tensors cannot be
            # constructed through contraction by the metric, their
            # components show if they are a matrix or its inverse.
            signature = tuple([i.is_up for i in key.get_indices()])
            srch = (key.component,) + signature
            if srch in self._substitutions_dict_tensmul:
                return self._substitutions_dict_tensmul[srch]
            array_list = [self.data_from_tensor(key)]
            return self.data_contract_dum(array_list, key.dum, key.ext_rank)

        if isinstance(key, TensMul):
            tensmul_args = key.args
            if len(tensmul_args) == 1 and len(tensmul_args[0].components) == 1:
                # special case to handle metrics. Metric tensors cannot be
                # constructed through contraction by the metric, their
                # components show if they are a matrix or its inverse.
                signature = tuple([i.is_up for i in tensmul_args[0].get_indices()])
                srch = (tensmul_args[0].components[0],) + signature
                if srch in self._substitutions_dict_tensmul:
                    return self._substitutions_dict_tensmul[srch]
            #data_list = [self.data_from_tensor(i) for i in tensmul_args if isinstance(i, TensExpr)]
            data_list = [self.data_from_tensor(i) if isinstance(i, Tensor) else i.data for i in tensmul_args if isinstance(i, TensExpr)]
            coeff = prod([i for i in tensmul_args if not isinstance(i, TensExpr)])
            if all(i is None for i in data_list):
                return None
            if any(i is None for i in data_list):
                raise ValueError("Mixing tensors with associated components "\
                                 "data with tensors without components data")
            data_result = self.data_contract_dum(data_list, key.dum, key.ext_rank)
            return coeff*data_result

        if isinstance(key, TensAdd):
            data_list = []
            free_args_list = []
            for arg in key.args:
                if isinstance(arg, TensExpr):
                    data_list.append(arg.data)
                    free_args_list.append([x[0] for x in arg.free])
                else:
                    data_list.append(arg)
                    free_args_list.append([])
            if all(i is None for i in data_list):
                return None
            if any(i is None for i in data_list):
                raise ValueError("Mixing tensors with associated components "\
                                 "data with tensors without components data")

            sum_list = []
            from .array import permutedims
            for data, free_args in zip(data_list, free_args_list):
                if len(free_args) < 2:
                    sum_list.append(data)
                else:
                    free_args_pos = {y: x for x, y in enumerate(free_args)}
                    axes = [free_args_pos[arg] for arg in key.free_args]
                    sum_list.append(permutedims(data, axes))
            return reduce(lambda x, y: x+y, sum_list)

        return None

    @staticmethod
    def data_contract_dum(ndarray_list, dum, ext_rank):
        from .array import tensorproduct, tensorcontraction, MutableDenseNDimArray
        arrays = list(map(MutableDenseNDimArray, ndarray_list))
        prodarr = tensorproduct(*arrays)
        return tensorcontraction(prodarr, *dum)

    def data_tensorhead_from_tensmul(self, data, tensmul, tensorhead):
        """
        This method is used when assigning components data to a ``TensMul``
        object, it converts components data to a fully contravariant ndarray,
        which is then stored according to the ``TensorHead`` key.
        """
        if data is None:
            return None

        return self._correct_signature_from_indices(
            data,
            tensmul.get_indices(),
            tensmul.free,
            tensmul.dum,
            True)

    def data_from_tensor(self, tensor):
        """
        This method corrects the components data to the right signature
        (covariant/contravariant) using the metric associated with each
        ``TensorIndexType``.
        """
        tensorhead = tensor.component

        if tensorhead.data is None:
            return None

        return self._correct_signature_from_indices(
            tensorhead.data,
            tensor.get_indices(),
            tensor.free,
            tensor.dum)

    def _assign_data_to_tensor_expr(self, key, data):
        if isinstance(key, TensAdd):
            raise ValueError('cannot assign data to TensAdd')
        # here it is assumed that `key` is a `TensMul` instance.
        if len(key.components) != 1:
            raise ValueError('cannot assign data to TensMul with multiple components')
        tensorhead = key.components[0]
        newdata = self.data_tensorhead_from_tensmul(data, key, tensorhead)
        return tensorhead, newdata

    def _check_permutations_on_data(self, tens, data):
        from .array import permutedims
        from .array.arrayop import Flatten

        if isinstance(tens, TensorHead):
            rank = tens.rank
            generators = tens.symmetry.generators
        elif isinstance(tens, Tensor):
            rank = tens.rank
            generators = tens.components[0].symmetry.generators
        elif isinstance(tens, TensorIndexType):
            rank = tens.metric.rank
            generators = tens.metric.symmetry.generators

        # Every generator is a permutation, check that by permuting the array
        # by that permutation, the array will be the same, except for a
        # possible sign change if the permutation admits it.
        for gener in generators:
            sign_change = +1 if (gener(rank) == rank) else -1
            data_swapped = data
            last_data = data
            permute_axes = list(map(gener, range(rank)))
            # the order of a permutation is the number of times to get the
            # identity by applying that permutation.
            for i in range(gener.order()-1):
                data_swapped = permutedims(data_swapped, permute_axes)
                # if any value in the difference array is non-zero, raise an error:
                if any(Flatten(last_data - sign_change*data_swapped)):
                    raise ValueError("Component data symmetry structure error")
                last_data = data_swapped

    def __setitem__(self, key, value):
        """
        Set the components data of a tensor object/expression.

        Explanation
        ===========

        Components data are transformed to the all-contravariant form and stored
        with the corresponding ``TensorHead`` object. If a ``TensorHead`` object
        cannot be uniquely identified, it will raise an error.
        """
        data = _TensorDataLazyEvaluator.parse_data(value)
        self._check_permutations_on_data(key, data)

        # TensorHead and TensorIndexType can be assigned data directly, while
        # TensMul must first convert data to a fully contravariant form, and
        # assign it to its corresponding TensorHead single component.
        if not isinstance(key, (TensorHead, TensorIndexType)):
            key, data = self._assign_data_to_tensor_expr(key, data)

        if isinstance(key, TensorHead):
            for dim, indextype in zip(data.shape, key.index_types):
                if indextype.data is None:
                    raise ValueError("index type {} has no components data"\
                    " associated (needed to raise/lower index)".format(indextype))
                if not indextype.dim.is_number:
                    continue
                if dim != indextype.dim:
                    raise ValueError("wrong dimension of ndarray")
        self._substitutions_dict[key] = data

    def __delitem__(self, key):
        del self._substitutions_dict[key]

    def __contains__(self, key):
        return key in self._substitutions_dict

    def add_metric_data(self, metric, data):
        """
        Assign data to the ``metric`` tensor. The metric tensor behaves in an
        anomalous way when raising and lowering indices.

        Explanation
        ===========

        A fully covariant metric is the inverse transpose of the fully
        contravariant metric (it is meant matrix inverse). If the metric is
        symmetric, the transpose is not necessary and mixed
        covariant/contravariant metrics are Kronecker deltas.
        """
        # hard assignment, data should not be added to `TensorHead` for metric:
        # the problem with `TensorHead` is that the metric is anomalous, i.e.
        # raising and lowering the index means considering the metric or its
        # inverse, this is not the case for other tensors.
        self._substitutions_dict_tensmul[metric, True, True] = data
        inverse_transpose = self.inverse_transpose_matrix(data)
        # in symmetric spaces, the transpose is the same as the original matrix,
        # the full covariant metric tensor is the inverse transpose, so this
        # code will be able to handle non-symmetric metrics.
        self._substitutions_dict_tensmul[metric, False, False] = inverse_transpose
        # now mixed cases, these are identical to the unit matrix if the metric
        # is symmetric.
        m = data.tomatrix()
        invt = inverse_transpose.tomatrix()
        self._substitutions_dict_tensmul[metric, True, False] = m * invt
        self._substitutions_dict_tensmul[metric, False, True] = invt * m

    @staticmethod
    def _flip_index_by_metric(data, metric, pos):
        from .array import tensorproduct, tensorcontraction

        mdim = metric.rank()
        ddim = data.rank()

        if pos == 0:
            data = tensorcontraction(
                tensorproduct(
                    metric,
                    data
                ),
                (1, mdim+pos)
            )
        else:
            data = tensorcontraction(
                tensorproduct(
                    data,
                    metric
                ),
                (pos, ddim)
            )
        return data

    @staticmethod
    def inverse_matrix(ndarray):
        m = ndarray.tomatrix().inv()
        return _TensorDataLazyEvaluator.parse_data(m)

    @staticmethod
    def inverse_transpose_matrix(ndarray):
        m = ndarray.tomatrix().inv().T
        return _TensorDataLazyEvaluator.parse_data(m)

    @staticmethod
    def _correct_signature_from_indices(data, indices, free, dum, inverse=False):
        """
        Utility function to correct the values inside the components data
        ndarray according to whether indices are covariant or contravariant.

        It uses the metric matrix to lower values of covariant indices.
        """
        # change the ndarray values according covariantness/contravariantness of the indices
        # use the metric
        for i, indx in enumerate(indices):
            if not indx.is_up and not inverse:
                data = _TensorDataLazyEvaluator._flip_index_by_metric(data, indx.tensor_index_type.data, i)
            elif not indx.is_up and inverse:
                data = _TensorDataLazyEvaluator._flip_index_by_metric(
                    data,
                    _TensorDataLazyEvaluator.inverse_matrix(indx.tensor_index_type.data),
                    i
                )
        return data

    @staticmethod
    def _sort_data_axes(old, new):
        from .array import permutedims

        new_data = old.data.copy()

        old_free = [i[0] for i in old.free]
        new_free = [i[0] for i in new.free]

        for i in range(len(new_free)):
            for j in range(i, len(old_free)):
                if old_free[j] == new_free[i]:
                    old_free[i], old_free[j] = old_free[j], old_free[i]
                    new_data = permutedims(new_data, (i, j))
                    break
        return new_data

    @staticmethod
    def add_rearrange_tensmul_parts(new_tensmul, old_tensmul):
        def sorted_compo():
            return _TensorDataLazyEvaluator._sort_data_axes(old_tensmul, new_tensmul)

        _TensorDataLazyEvaluator._substitutions_dict[new_tensmul] = sorted_compo()

    @staticmethod
    def parse_data(data):
        """
        Transform ``data`` to array. The parameter ``data`` may
        contain data in various formats, e.g. nested lists, SymPy ``Matrix``,
        and so on.

        Examples
        ========

        >>> from sympy.tensor.tensor import _TensorDataLazyEvaluator
        >>> _TensorDataLazyEvaluator.parse_data([1, 3, -6, 12])
        [1, 3, -6, 12]

        >>> _TensorDataLazyEvaluator.parse_data([[1, 2], [4, 7]])
        [[1, 2], [4, 7]]
        """
        from .array import MutableDenseNDimArray

        if not isinstance(data, MutableDenseNDimArray):
            if len(data) == 2 and hasattr(data[0], '__call__'):
                data = MutableDenseNDimArray(data[0], data[1])
            else:
                data = MutableDenseNDimArray(data)
        return data

_tensor_data_substitution_dict = _TensorDataLazyEvaluator()


class _TensorManager:
    """
    Class to manage tensor properties.

    Notes
    =====

    Tensors belong to tensor commutation groups; each group has a label
    ``comm``; there are predefined labels:

    ``0``   tensors commuting with any other tensor

    ``1``   tensors anticommuting among themselves

    ``2``   tensors not commuting, apart with those with ``comm=0``

    Other groups can be defined using ``set_comm``; tensors in those
    groups commute with those with ``comm=0``; by default they
    do not commute with any other group.
    """
    def __init__(self):
        self._comm_init()

    def _comm_init(self):
        self._comm = [{} for i in range(3)]
        for i in range(3):
            self._comm[0][i] = 0
            self._comm[i][0] = 0
        self._comm[1][1] = 1
        self._comm[2][1] = None
        self._comm[1][2] = None
        self._comm_symbols2i = {0:0, 1:1, 2:2}
        self._comm_i2symbol = {0:0, 1:1, 2:2}

    @property
    def comm(self):
        return self._comm

    def comm_symbols2i(self, i):
        """
        Get the commutation group number corresponding to ``i``.

        ``i`` can be a symbol or a number or a string.

        If ``i`` is not already defined its commutation group number
        is set.
        """
        if i not in self._comm_symbols2i:
            n = len(self._comm)
            self._comm.append({})
            self._comm[n][0] = 0
            self._comm[0][n] = 0
            self._comm_symbols2i[i] = n
            self._comm_i2symbol[n] = i
            return n
        return self._comm_symbols2i[i]

    def comm_i2symbol(self, i):
        """
        Returns the symbol corresponding to the commutation group number.
        """
        return self._comm_i2symbol[i]

    def set_comm(self, i, j, c):
        """
        Set the commutation parameter ``c`` for commutation groups ``i, j``.

        Parameters
        ==========

        i, j : symbols representing commutation groups

        c  :  group commutation number

        Notes
        =====

        ``i, j`` can be symbols, strings or numbers,
        apart from ``0, 1`` and ``2`` which are reserved respectively
        for commuting, anticommuting tensors and tensors not commuting
        with any other group apart with the commuting tensors.
        For the remaining cases, use this method to set the commutation rules;
        by default ``c=None``.

        The group commutation number ``c`` is assigned in correspondence
        to the group commutation symbols; it can be

        0        commuting

        1        anticommuting

        None     no commutation property

        Examples
        ========

        ``G`` and ``GH`` do not commute with themselves and commute with
        each other; A is commuting.

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorHead, TensorManager, TensorSymmetry
        >>> Lorentz = TensorIndexType('Lorentz')
        >>> i0,i1,i2,i3,i4 = tensor_indices('i0:5', Lorentz)
        >>> A = TensorHead('A', [Lorentz])
        >>> G = TensorHead('G', [Lorentz], TensorSymmetry.no_symmetry(1), 'Gcomm')
        >>> GH = TensorHead('GH', [Lorentz], TensorSymmetry.no_symmetry(1), 'GHcomm')
        >>> TensorManager.set_comm('Gcomm', 'GHcomm', 0)
        >>> (GH(i1)*G(i0)).canon_bp()
        G(i0)*GH(i1)
        >>> (G(i1)*G(i0)).canon_bp()
        G(i1)*G(i0)
        >>> (G(i1)*A(i0)).canon_bp()
        A(i0)*G(i1)
        """
        if c not in (0, 1, None):
            raise ValueError('`c` can assume only the values 0, 1 or None')

        i = sympify(i)
        j = sympify(j)

        if i not in self._comm_symbols2i:
            n = len(self._comm)
            self._comm.append({})
            self._comm[n][0] = 0
            self._comm[0][n] = 0
            self._comm_symbols2i[i] = n
            self._comm_i2symbol[n] = i
        if j not in self._comm_symbols2i:
            n = len(self._comm)
            self._comm.append({})
            self._comm[0][n] = 0
            self._comm[n][0] = 0
            self._comm_symbols2i[j] = n
            self._comm_i2symbol[n] = j
        ni = self._comm_symbols2i[i]
        nj = self._comm_symbols2i[j]
        self._comm[ni][nj] = c
        self._comm[nj][ni] = c

        """
        Cached sympy functions (e.g. expand) may have cached the results of
        expressions involving tensors, but those results may not be valid after
        changing the commutation properties. To stay on the safe side, we clear
        the cache of all functions.
        """
        clear_cache()

    def set_comms(self, *args):
        """
        Set the commutation group numbers ``c`` for symbols ``i, j``.

        Parameters
        ==========

        args : sequence of ``(i, j, c)``
        """
        for i, j, c in args:
            self.set_comm(i, j, c)

    def get_comm(self, i, j):
        """
        Return the commutation parameter for commutation group numbers ``i, j``

        see ``_TensorManager.set_comm``
        """
        return self._comm[i].get(j, 0 if i == 0 or j == 0 else None)

    def clear(self):
        """
        Clear the TensorManager.
        """
        self._comm_init()


TensorManager = _TensorManager()


class TensorIndexType(Basic):
    """
    A TensorIndexType is characterized by its name and its metric.

    Parameters
    ==========

    name : name of the tensor type
    dummy_name : name of the head of dummy indices
    dim : dimension, it can be a symbol or an integer or ``None``
    eps_dim : dimension of the epsilon tensor
    metric_symmetry : integer that denotes metric symmetry or ``None`` for no metric
    metric_name : string with the name of the metric tensor

    Attributes
    ==========

    ``metric`` : the metric tensor
    ``delta`` : ``Kronecker delta``
    ``epsilon`` : the ``Levi-Civita epsilon`` tensor
    ``data`` : (deprecated) a property to add ``ndarray`` values, to work in a specified basis.

    Notes
    =====

    The possible values of the ``metric_symmetry`` parameter are:

        ``1``   :   metric tensor is fully symmetric
        ``0``   :   metric tensor possesses no index symmetry
        ``-1``  :   metric tensor is fully antisymmetric
        ``None``:   there is no metric tensor (metric equals to ``None``)

    The metric is assumed to be symmetric by default. It can also be set
    to a custom tensor by the ``.set_metric()`` method.

    If there is a metric the metric is used to raise and lower indices.

    In the case of non-symmetric metric, the following raising and
    lowering conventions will be adopted:

    ``psi(a) = g(a, b)*psi(-b); chi(-a) = chi(b)*g(-b, -a)``

    From these it is easy to find:

    ``g(-a, b) = delta(-a, b)``

    where ``delta(-a, b) = delta(b, -a)`` is the ``Kronecker delta``
    (see ``TensorIndex`` for the conventions on indices).
    For antisymmetric metrics there is also the following equality:

    ``g(a, -b) = -delta(a, -b)``

    If there is no metric it is not possible to raise or lower indices;
    e.g. the index of the defining representation of ``SU(N)``
    is 'covariant' and the conjugate representation is
    'contravariant'; for ``N > 2`` they are linearly independent.

    ``eps_dim`` is by default equal to ``dim``, if the latter is an integer;
    else it can be assigned (for use in naive dimensional regularization);
    if ``eps_dim`` is not an integer ``epsilon`` is ``None``.

    Examples
    ========

    >>> from sympy.tensor.tensor import TensorIndexType
    >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    >>> Lorentz.metric
    metric(Lorentz,Lorentz)
    """

    def __new__(cls, name, dummy_name=None, dim=None, eps_dim=None,
                metric_symmetry=1, metric_name='metric', **kwargs):
        if 'dummy_fmt' in kwargs:
            dummy_fmt = kwargs['dummy_fmt']
            sympy_deprecation_warning(
                f"""
                The dummy_fmt keyword to TensorIndexType is deprecated. Use
                dummy_name={dummy_fmt} instead.
                """,
                deprecated_since_version="1.5",
                active_deprecations_target="deprecated-tensorindextype-dummy-fmt",
            )
            dummy_name = dummy_fmt

        if isinstance(name, str):
            name = Symbol(name)

        if dummy_name is None:
            dummy_name = str(name)[0]
        if isinstance(dummy_name, str):
            dummy_name = Symbol(dummy_name)

        if dim is None:
            dim = Symbol("dim_" + dummy_name.name)
        else:
            dim = sympify(dim)

        if eps_dim is None:
            eps_dim = dim
        else:
            eps_dim = sympify(eps_dim)

        metric_symmetry = sympify(metric_symmetry)

        if isinstance(metric_name, str):
            metric_name = Symbol(metric_name)

        if 'metric' in kwargs:
            SymPyDeprecationWarning(
                """
                The 'metric' keyword argument to TensorIndexType is
                deprecated. Use the 'metric_symmetry' keyword argument or the
                TensorIndexType.set_metric() method instead.
                """,
                deprecated_since_version="1.5",
                active_deprecations_target="deprecated-tensorindextype-metric",
            )
            metric = kwargs.get('metric')
            if metric is not None:
                if metric in (True, False, 0, 1):
                    metric_name = 'metric'
                    #metric_antisym = metric
                else:
                    metric_name = metric.name
                    #metric_antisym = metric.antisym

                if metric:
                    metric_symmetry = -1
                else:
                    metric_symmetry = 1

        obj = Basic.__new__(cls, name, dummy_name, dim, eps_dim,
                            metric_symmetry, metric_name)

        obj._autogenerated = []
        return obj

    @property
    def name(self):
        return self.args[0].name

    @property
    def dummy_name(self):
        return self.args[1].name

    @property
    def dim(self):
        return self.args[2]

    @property
    def eps_dim(self):
        return self.args[3]

    @memoize_property
    def metric(self):
        metric_symmetry = self.args[4]
        metric_name = self.args[5]
        if metric_symmetry is None:
            return None

        if metric_symmetry == 0:
            symmetry = TensorSymmetry.no_symmetry(2)
        elif metric_symmetry == 1:
            symmetry = TensorSymmetry.fully_symmetric(2)
        elif metric_symmetry == -1:
            symmetry = TensorSymmetry.fully_symmetric(-2)

        return TensorHead(metric_name, [self]*2, symmetry)

    @memoize_property
    def delta(self):
        return TensorHead('KD', [self]*2, TensorSymmetry.fully_symmetric(2))

    @memoize_property
    def epsilon(self):
        if not isinstance(self.eps_dim, (SYMPY_INTS, Integer)):
            return None
        symmetry = TensorSymmetry.fully_symmetric(-self.eps_dim)
        return TensorHead('Eps', [self]*self.eps_dim, symmetry)

    def set_metric(self, tensor):
        self._metric = tensor

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return self.name

    __repr__ = __str__

    # Everything below this line is deprecated

    @property
    def data(self):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            return _tensor_data_substitution_dict[self]

    @data.setter
    def data(self, data):
        deprecate_data()
        # This assignment is a bit controversial, should metric components be assigned
        # to the metric only or also to the TensorIndexType object? The advantage here
        # is the ability to assign a 1D array and transform it to a 2D diagonal array.
        from .array import MutableDenseNDimArray

        data = _TensorDataLazyEvaluator.parse_data(data)
        if data.rank() > 2:
            raise ValueError("data have to be of rank 1 (diagonal metric) or 2.")
        if data.rank() == 1:
            if self.dim.is_number:
                nda_dim = data.shape[0]
                if nda_dim != self.dim:
                    raise ValueError("Dimension mismatch")

            dim = data.shape[0]
            newndarray = MutableDenseNDimArray.zeros(dim, dim)
            for i, val in enumerate(data):
                newndarray[i, i] = val
            data = newndarray
        dim1, dim2 = data.shape
        if dim1 != dim2:
            raise ValueError("Non-square matrix tensor.")
        if self.dim.is_number:
            if self.dim != dim1:
                raise ValueError("Dimension mismatch")
        _tensor_data_substitution_dict[self] = data
        _tensor_data_substitution_dict.add_metric_data(self.metric, data)
        with ignore_warnings(SymPyDeprecationWarning):
            delta = self.get_kronecker_delta()
        i1 = TensorIndex('i1', self)
        i2 = TensorIndex('i2', self)
        with ignore_warnings(SymPyDeprecationWarning):
            delta(i1, -i2).data = _TensorDataLazyEvaluator.parse_data(eye(dim1))

    @data.deleter
    def data(self):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            if self in _tensor_data_substitution_dict:
                del _tensor_data_substitution_dict[self]
            if self.metric in _tensor_data_substitution_dict:
                del _tensor_data_substitution_dict[self.metric]

    @deprecated(
        """
        The TensorIndexType.get_kronecker_delta() method is deprecated. Use
        the TensorIndexType.delta attribute instead.
        """,
        deprecated_since_version="1.5",
        active_deprecations_target="deprecated-tensorindextype-methods",
    )
    def get_kronecker_delta(self):
        sym2 = TensorSymmetry(get_symmetric_group_sgs(2))
        delta = TensorHead('KD', [self]*2, sym2)
        return delta

    @deprecated(
        """
        The TensorIndexType.get_epsilon() method is deprecated. Use
        the TensorIndexType.epsilon attribute instead.
        """,
        deprecated_since_version="1.5",
        active_deprecations_target="deprecated-tensorindextype-methods",
    )
    def get_epsilon(self):
        if not isinstance(self._eps_dim, (SYMPY_INTS, Integer)):
            return None
        sym = TensorSymmetry(get_symmetric_group_sgs(self._eps_dim, 1))
        epsilon = TensorHead('Eps', [self]*self._eps_dim, sym)
        return epsilon

    def _components_data_full_destroy(self):
        """
        EXPERIMENTAL: do not rely on this API method.

        This destroys components data associated to the ``TensorIndexType``, if
        any, specifically:

        * metric tensor data
        * Kronecker tensor data
        """
        if self in _tensor_data_substitution_dict:
            del _tensor_data_substitution_dict[self]

        def delete_tensmul_data(key):
            if key in _tensor_data_substitution_dict._substitutions_dict_tensmul:
                del _tensor_data_substitution_dict._substitutions_dict_tensmul[key]

        # delete metric data:
        delete_tensmul_data((self.metric, True, True))
        delete_tensmul_data((self.metric, True, False))
        delete_tensmul_data((self.metric, False, True))
        delete_tensmul_data((self.metric, False, False))

        # delete delta tensor data:
        delta = self.get_kronecker_delta()
        if delta in _tensor_data_substitution_dict:
            del _tensor_data_substitution_dict[delta]


class TensorIndex(Basic):
    """
    Represents a tensor index

    Parameters
    ==========

    name : name of the index, or ``True`` if you want it to be automatically assigned
    tensor_index_type : ``TensorIndexType`` of the index
    is_up :  flag for contravariant index (is_up=True by default)

    Attributes
    ==========

    ``name``
    ``tensor_index_type``
    ``is_up``

    Notes
    =====

    Tensor indices are contracted with the Einstein summation convention.

    An index can be in contravariant or in covariant form; in the latter
    case it is represented prepending a ``-`` to the index name. Adding
    ``-`` to a covariant (is_up=False) index makes it contravariant.

    Dummy indices have a name with head given by
    ``tensor_inde_type.dummy_name`` with underscore and a number.

    Similar to ``symbols`` multiple contravariant indices can be created
    at once using ``tensor_indices(s, typ)``, where ``s`` is a string
    of names.


    Examples
    ========

    >>> from sympy.tensor.tensor import TensorIndexType, TensorIndex, TensorHead, tensor_indices
    >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    >>> mu = TensorIndex('mu', Lorentz, is_up=False)
    >>> nu, rho = tensor_indices('nu, rho', Lorentz)
    >>> A = TensorHead('A', [Lorentz, Lorentz])
    >>> A(mu, nu)
    A(-mu, nu)
    >>> A(-mu, -rho)
    A(mu, -rho)
    >>> A(mu, -mu)
    A(-L_0, L_0)
    """
    def __new__(cls, name, tensor_index_type, is_up=True):
        if isinstance(name, str):
            name_symbol = Symbol(name)
        elif isinstance(name, Symbol):
            name_symbol = name
        elif name is True:
            name = "_i{}".format(len(tensor_index_type._autogenerated))
            name_symbol = Symbol(name)
            tensor_index_type._autogenerated.append(name_symbol)
        else:
            raise ValueError("invalid name")

        is_up = sympify(is_up)
        return Basic.__new__(cls, name_symbol, tensor_index_type, is_up)

    @property
    def name(self):
        return self.args[0].name

    @property
    def tensor_index_type(self):
        return self.args[1]

    @property
    def is_up(self):
        return self.args[2]

    def _print(self):
        s = self.name
        if not self.is_up:
            s = '-%s' % s
        return s

    def __lt__(self, other):
        return ((self.tensor_index_type, self.name) <
                (other.tensor_index_type, other.name))

    def __neg__(self):
        t1 = TensorIndex(self.name, self.tensor_index_type,
                (not self.is_up))
        return t1


def tensor_indices(s, typ):
    """
    Returns list of tensor indices given their names and their types.

    Parameters
    ==========

    s : string of comma separated names of indices

    typ : ``TensorIndexType`` of the indices

    Examples
    ========

    >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices
    >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    >>> a, b, c, d = tensor_indices('a,b,c,d', Lorentz)
    """
    if isinstance(s, str):
        a = [x.name for x in symbols(s, seq=True)]
    else:
        raise ValueError('expecting a string')

    tilist = [TensorIndex(i, typ) for i in a]
    if len(tilist) == 1:
        return tilist[0]
    return tilist


class TensorSymmetry(Basic):
    """
    Monoterm symmetry of a tensor (i.e. any symmetric or anti-symmetric
    index permutation). For the relevant terminology see ``tensor_can.py``
    section of the combinatorics module.

    Parameters
    ==========

    bsgs : tuple ``(base, sgs)`` BSGS of the symmetry of the tensor

    Attributes
    ==========

    ``base`` : base of the BSGS
    ``generators`` : generators of the BSGS
    ``rank`` : rank of the tensor

    Notes
    =====

    A tensor can have an arbitrary monoterm symmetry provided by its BSGS.
    Multiterm symmetries, like the cyclic symmetry of the Riemann tensor
    (i.e., Bianchi identity), are not covered. See combinatorics module for
    information on how to generate BSGS for a general index permutation group.
    Simple symmetries can be generated using built-in methods.

    See Also
    ========

    sympy.combinatorics.tensor_can.get_symmetric_group_sgs

    Examples
    ========

    Define a symmetric tensor of rank 2

    >>> from sympy.tensor.tensor import TensorIndexType, TensorSymmetry, get_symmetric_group_sgs, TensorHead
    >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    >>> sym = TensorSymmetry(get_symmetric_group_sgs(2))
    >>> T = TensorHead('T', [Lorentz]*2, sym)

    Note, that the same can also be done using built-in TensorSymmetry methods

    >>> sym2 = TensorSymmetry.fully_symmetric(2)
    >>> sym == sym2
    True
    """
    def __new__(cls, *args, **kw_args):
        if len(args) == 1:
            base, generators = args[0]
        elif len(args) == 2:
            base, generators = args
        else:
            raise TypeError("bsgs required, either two separate parameters or one tuple")

        if not isinstance(base, Tuple):
            base = Tuple(*base)
        if not isinstance(generators, Tuple):
            generators = Tuple(*generators)

        return Basic.__new__(cls, base, generators, **kw_args)

    @property
    def base(self):
        return self.args[0]

    @property
    def generators(self):
        return self.args[1]

    @property
    def rank(self):
        return self.generators[0].size - 2

    @classmethod
    def fully_symmetric(cls, rank):
        """
        Returns a fully symmetric (antisymmetric if ``rank``<0)
        TensorSymmetry object for ``abs(rank)`` indices.
        """
        if rank > 0:
            bsgs = get_symmetric_group_sgs(rank, False)
        elif rank < 0:
            bsgs = get_symmetric_group_sgs(-rank, True)
        elif rank == 0:
            bsgs = ([], [Permutation(1)])
        return TensorSymmetry(bsgs)

    @classmethod
    def direct_product(cls, *args):
        """
        Returns a TensorSymmetry object that is being a direct product of
        fully (anti-)symmetric index permutation groups.

        Notes
        =====

        Some examples for different values of ``(*args)``:
        ``(1)``         vector, equivalent to ``TensorSymmetry.fully_symmetric(1)``
        ``(2)``         tensor with 2 symmetric indices, equivalent to ``.fully_symmetric(2)``
        ``(-2)``        tensor with 2 antisymmetric indices, equivalent to ``.fully_symmetric(-2)``
        ``(2, -2)``     tensor with the first 2 indices commuting and the last 2 anticommuting
        ``(1, 1, 1)``   tensor with 3 indices without any symmetry
        """
        base, sgs = [], [Permutation(1)]
        for arg in args:
            if arg > 0:
                bsgs2 = get_symmetric_group_sgs(arg, False)
            elif arg < 0:
                bsgs2 = get_symmetric_group_sgs(-arg, True)
            else:
                continue
            base, sgs = bsgs_direct_product(base, sgs, *bsgs2)

        return TensorSymmetry(base, sgs)

    @classmethod
    def riemann(cls):
        """
        Returns a monotorem symmetry of the Riemann tensor
        """
        return TensorSymmetry(riemann_bsgs)

    @classmethod
    def no_symmetry(cls, rank):
        """
        TensorSymmetry object for ``rank`` indices with no symmetry
        """
        return TensorSymmetry([], [Permutation(rank+1)])


@deprecated(
    """
    The tensorsymmetry() function is deprecated. Use the TensorSymmetry
    constructor instead.
    """,
    deprecated_since_version="1.5",
    active_deprecations_target="deprecated-tensorsymmetry",
)
def tensorsymmetry(*args):
    """
    Returns a ``TensorSymmetry`` object. This method is deprecated, use
    ``TensorSymmetry.direct_product()`` or ``.riemann()`` instead.

    Explanation
    ===========

    One can represent a tensor with any monoterm slot symmetry group
    using a BSGS.

    ``args`` can be a BSGS
    ``args[0]``    base
    ``args[1]``    sgs

    Usually tensors are in (direct products of) representations
    of the symmetric group;
    ``args`` can be a list of lists representing the shapes of Young tableaux

    Notes
    =====

    For instance:
    ``[[1]]``       vector
    ``[[1]*n]``     symmetric tensor of rank ``n``
    ``[[n]]``       antisymmetric tensor of rank ``n``
    ``[[2, 2]]``    monoterm slot symmetry of the Riemann tensor
    ``[[1],[1]]``   vector*vector
    ``[[2],[1],[1]`` (antisymmetric tensor)*vector*vector

    Notice that with the shape ``[2, 2]`` we associate only the monoterm
