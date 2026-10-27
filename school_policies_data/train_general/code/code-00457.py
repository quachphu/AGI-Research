        rv = Sum( Product(f, (i, a, h - 1)) * Product(f, (i, h + 1, b)) * Derivative(f, x, evaluate=True).subs(i, h), (h, a, b))
        return rv

    def is_convergent(self):
        r"""
        See docs of :obj:`.Sum.is_convergent()` for explanation of convergence
        in SymPy.

        Explanation
        ===========

        The infinite product:

        .. math::

            \prod_{1 \leq i < \infty} f(i)

        is defined by the sequence of partial products:

        .. math::

            \prod_{i=1}^{n} f(i) = f(1) f(2) \cdots f(n)

        as n increases without bound. The product converges to a non-zero
        value if and only if the sum:

        .. math::

            \sum_{1 \leq i < \infty} \log{f(n)}

        converges.

        Examples
        ========

        >>> from sympy import Product, Symbol, cos, pi, exp, oo
        >>> n = Symbol('n', integer=True)
        >>> Product(n/(n + 1), (n, 1, oo)).is_convergent()
        False
        >>> Product(1/n**2, (n, 1, oo)).is_convergent()
        False
        >>> Product(cos(pi/n), (n, 1, oo)).is_convergent()
        True
        >>> Product(exp(-n**2), (n, 1, oo)).is_convergent()
        False

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Infinite_product
        """
        sequence_term = self.function
        log_sum = log(sequence_term)
        lim = self.limits
        try:
            is_conv = Sum(log_sum, *lim).is_convergent()
        except NotImplementedError:
            if Sum(sequence_term - 1, *lim).is_absolutely_convergent() is S.true:
                return S.true
            raise NotImplementedError("The algorithm to find the product convergence of %s "
                                        "is not yet implemented" % (sequence_term))
        return is_conv

    def reverse_order(expr, *indices):
        """
        Reverse the order of a limit in a Product.

        Explanation
        ===========

        ``reverse_order(expr, *indices)`` reverses some limits in the expression
        ``expr`` which can be either a ``Sum`` or a ``Product``. The selectors in
        the argument ``indices`` specify some indices whose limits get reversed.
        These selectors are either variable names or numerical indices counted
        starting from the inner-most limit tuple.

        Examples
        ========

        >>> from sympy import gamma, Product, simplify, Sum
        >>> from sympy.abc import x, y, a, b, c, d
        >>> P = Product(x, (x, a, b))
        >>> Pr = P.reverse_order(x)
        >>> Pr
        Product(1/x, (x, b + 1, a - 1))
        >>> Pr = Pr.doit()
        >>> Pr
        1/RisingFactorial(b + 1, a - b - 1)
        >>> simplify(Pr.rewrite(gamma))
        Piecewise((gamma(b + 1)/gamma(a), b > -1), ((-1)**(-a + b + 1)*gamma(1 - a)/gamma(-b), True))
        >>> P = P.doit()
        >>> P
        RisingFactorial(a, -a + b + 1)
        >>> simplify(P.rewrite(gamma))
        Piecewise((gamma(b + 1)/gamma(a), a > 0), ((-1)**(-a + b + 1)*gamma(1 - a)/gamma(-b), True))

        While one should prefer variable names when specifying which limits
        to reverse, the index counting notation comes in handy in case there
        are several symbols with the same name.

        >>> S = Sum(x*y, (x, a, b), (y, c, d))
        >>> S
        Sum(x*y, (x, a, b), (y, c, d))
        >>> S0 = S.reverse_order(0)
        >>> S0
        Sum(-x*y, (x, b + 1, a - 1), (y, c, d))
        >>> S1 = S0.reverse_order(1)
        >>> S1
        Sum(x*y, (x, b + 1, a - 1), (y, d + 1, c - 1))

        Of course we can mix both notations:

        >>> Sum(x*y, (x, a, b), (y, 2, 5)).reverse_order(x, 1)
        Sum(x*y, (x, b + 1, a - 1), (y, 6, 1))
        >>> Sum(x*y, (x, a, b), (y, 2, 5)).reverse_order(y, x)
        Sum(x*y, (x, b + 1, a - 1), (y, 6, 1))

        See Also
        ========

        sympy.concrete.expr_with_intlimits.ExprWithIntLimits.index,
        reorder_limit,
        sympy.concrete.expr_with_intlimits.ExprWithIntLimits.reorder

        References
        ==========

        .. [1] Michael Karr, "Summation in Finite Terms", Journal of the ACM,
               Volume 28 Issue 2, April 1981, Pages 305-350
               https://dl.acm.org/doi/10.1145/322248.322255

        """
        l_indices = list(indices)

        for i, indx in enumerate(l_indices):
            if not isinstance(indx, int):
                l_indices[i] = expr.index(indx)

        e = 1
        limits = []
        for i, limit in enumerate(expr.limits):
            l = limit
            if i in l_indices:
                e = -e
                l = (limit[0], limit[2] + 1, limit[1] - 1)
            limits.append(l)

        return Product(expr.function ** e, *limits)


def product(*args, **kwargs):
    r"""
    Compute the product.

    Explanation
    ===========

    The notation for symbols is similar to the notation used in Sum or
    Integral. product(f, (i, a, b)) computes the product of f with
    respect to i from a to b, i.e.,

    ::

                                     b
                                   _____
        product(f(n), (i, a, b)) = |   | f(n)
                                   |   |
                                   i = a

    If it cannot compute the product, it returns an unevaluated Product object.
    Repeated products can be computed by introducing additional symbols tuples::

    Examples
    ========

    >>> from sympy import product, symbols
    >>> i, n, m, k = symbols('i n m k', integer=True)

    >>> product(i, (i, 1, k))
    factorial(k)
    >>> product(m, (i, 1, k))
    m**k
    >>> product(i, (i, 1, k), (k, 1, n))
    Product(factorial(k), (k, 1, n))

    """

    prod = Product(*args, **kwargs)

    if isinstance(prod, Product):
        return prod.doit(deep=False)
    else:
        return prod
<|endoftext|>"""Gosper's algorithm for hypergeometric summation. """

from sympy.core import S, Dummy, symbols
from sympy.polys import Poly, parallel_poly_from_expr, factor
from sympy.utilities.iterables import is_sequence


def gosper_normal(f, g, n, polys=True):
    r"""
    Compute the Gosper's normal form of ``f`` and ``g``.

    Explanation
    ===========

    Given relatively prime univariate polynomials ``f`` and ``g``,
    rewrite their quotient to a normal form defined as follows:

    .. math::
        \frac{f(n)}{g(n)} = Z \cdot \frac{A(n) C(n+1)}{B(n) C(n)}

    where ``Z`` is an arbitrary constant and ``A``, ``B``, ``C`` are
    monic polynomials in ``n`` with the following properties:

    1. `\gcd(A(n), B(n+h)) = 1 \forall h \in \mathbb{N}`
    2. `\gcd(B(n), C(n+1)) = 1`
    3. `\gcd(A(n), C(n)) = 1`

    This normal form, or rational factorization in other words, is a
    crucial step in Gosper's algorithm and in solving of difference
    equations. It can be also used to decide if two hypergeometric
    terms are similar or not.

    This procedure will return a tuple containing elements of this
    factorization in the form ``(Z*A, B, C)``.

    Examples
    ========

    >>> from sympy.concrete.gosper import gosper_normal
    >>> from sympy.abc import n

    >>> gosper_normal(4*n+5, 2*(4*n+1)*(2*n+3), n, polys=False)
    (1/4, n + 3/2, n + 1/4)

    """
    (p, q), opt = parallel_poly_from_expr(
        (f, g), n, field=True, extension=True)

    a, A = p.LC(), p.monic()
    b, B = q.LC(), q.monic()

    C, Z = A.one, a/b
    h = Dummy('h')

    D = Poly(n + h, n, h, domain=opt.domain)

    R = A.resultant(B.compose(D))
    roots = {r for r in R.ground_roots().keys() if r.is_Integer and r >= 0}
    for i in sorted(roots):
        d = A.gcd(B.shift(+i))

        A = A.quo(d)
        B = B.quo(d.shift(-i))

        for j in range(1, i + 1):
            C *= d.shift(-j)

    A = A.mul_ground(Z)

    if not polys:
        A = A.as_expr()
        B = B.as_expr()
        C = C.as_expr()

    return A, B, C


def gosper_term(f, n):
    r"""
    Compute Gosper's hypergeometric term for ``f``.

    Explanation
    ===========

    Suppose ``f`` is a hypergeometric term such that:

    .. math::
        s_n = \sum_{k=0}^{n-1} f_k

    and `f_k` does not depend on `n`. Returns a hypergeometric
    term `g_n` such that `g_{n+1} - g_n = f_n`.

    Examples
    ========

    >>> from sympy.concrete.gosper import gosper_term
    >>> from sympy import factorial
    >>> from sympy.abc import n

    >>> gosper_term((4*n + 1)*factorial(n)/factorial(2*n + 1), n)
    (-n - 1/2)/(n + 1/4)

    """
    from sympy.simplify import hypersimp
    r = hypersimp(f, n)

    if r is None:
        return None    # 'f' is *not* a hypergeometric term

    p, q = r.as_numer_denom()

    A, B, C = gosper_normal(p, q, n)
    B = B.shift(-1)

    N = S(A.degree())
    M = S(B.degree())
    K = S(C.degree())

    if (N != M) or (A.LC() != B.LC()):
        D = {K - max(N, M)}
    elif not N:
        D = {K - N + 1, S.Zero}
    else:
        D = {K - N + 1, (B.nth(N - 1) - A.nth(N - 1))/A.LC()}

    for d in set(D):
        if not d.is_Integer or d < 0:
            D.remove(d)

    if not D:
        return None    # 'f(n)' is *not* Gosper-summable

    d = max(D)

    coeffs = symbols('c:%s' % (d + 1), cls=Dummy)
    domain = A.get_domain().inject(*coeffs)

    x = Poly(coeffs, n, domain=domain)
    H = A*x.shift(1) - B*x - C

    from sympy.solvers.solvers import solve
    solution = solve(H.coeffs(), coeffs)

    if solution is None:
        return None    # 'f(n)' is *not* Gosper-summable

    x = x.as_expr().subs(solution)

    for coeff in coeffs:
        if coeff not in solution:
            x = x.subs(coeff, 0)

    if x.is_zero:
        return None    # 'f(n)' is *not* Gosper-summable
    else:
        return B.as_expr()*x/C.as_expr()


def gosper_sum(f, k):
    r"""
    Gosper's hypergeometric summation algorithm.

    Explanation
    ===========

    Given a hypergeometric term ``f`` such that:

    .. math ::
        s_n = \sum_{k=0}^{n-1} f_k

    and `f(n)` does not depend on `n`, returns `g_{n} - g(0)` where
    `g_{n+1} - g_n = f_n`, or ``None`` if `s_n` cannot be expressed
    in closed form as a sum of hypergeometric terms.

    Examples
    ========

    >>> from sympy.concrete.gosper import gosper_sum
    >>> from sympy import factorial
    >>> from sympy.abc import n, k

    >>> f = (4*k + 1)*factorial(k)/factorial(2*k + 1)
    >>> gosper_sum(f, (k, 0, n))
    (-factorial(n) + 2*factorial(2*n + 1))/factorial(2*n + 1)
    >>> _.subs(n, 2) == sum(f.subs(k, i) for i in [0, 1, 2])
    True
    >>> gosper_sum(f, (k, 3, n))
    (-60*factorial(n) + factorial(2*n + 1))/(60*factorial(2*n + 1))
    >>> _.subs(n, 5) == sum(f.subs(k, i) for i in [3, 4, 5])
    True

    References
    ==========

    .. [1] Marko Petkovsek, Herbert S. Wilf, Doron Zeilberger, A = B,
           AK Peters, Ltd., Wellesley, MA, USA, 1997, pp. 73--100

    """
    indefinite = False

    if is_sequence(k):
        k, a, b = k
    else:
        indefinite = True

    g = gosper_term(f, k)

    if g is None:
        return None

    if indefinite:
        result = f*g
    else:
        result = (f*(g + 1)).subs(k, b) - (f*g).subs(k, a)

        if result is S.NaN:
            try:
                result = (f*(g + 1)).limit(k, b) - (f*g).limit(k, a)
            except NotImplementedError:
                result = None

    return factor(result)
<|endoftext|>from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.polys.polytools import lcm
from sympy.utilities import public

@public
def approximants(l, X=Symbol('x'), simplify=False):
    """
    Return a generator for consecutive Pade approximants for a series.
    It can also be used for computing the rational generating function of a
    series when possible, since the last approximant returned by the generator
    will be the generating function (if any).

    Explanation
    ===========

    The input list can contain more complex expressions than integer or rational
    numbers; symbols may also be involved in the computation. An example below
    show how to compute the generating function of the whole Pascal triangle.

    The generator can be asked to apply the sympy.simplify function on each
    generated term, which will make the computation slower; however it may be
    useful when symbols are involved in the expressions.

    Examples
    ========

    >>> from sympy.series import approximants
    >>> from sympy import lucas, fibonacci, symbols, binomial
    >>> g = [lucas(k) for k in range(16)]
    >>> [e for e in approximants(g)]
    [2, -4/(x - 2), (5*x - 2)/(3*x - 1), (x - 2)/(x**2 + x - 1)]

    >>> h = [fibonacci(k) for k in range(16)]
    >>> [e for e in approximants(h)]
    [x, -x/(x - 1), (x**2 - x)/(2*x - 1), -x/(x**2 + x - 1)]

    >>> x, t = symbols("x,t")
    >>> p=[sum(binomial(k,i)*x**i for i in range(k+1)) for k in range(16)]
    >>> y = approximants(p, t)
    >>> for k in range(3): print(next(y))
    1
    (x + 1)/((-x - 1)*(t*(x + 1) + (x + 1)/(-x - 1)))
    nan

    >>> y = approximants(p, t, simplify=True)
    >>> for k in range(3): print(next(y))
    1
    -1/(t*(x + 1) - 1)
    nan

    See Also
    ========

    sympy.concrete.guess.guess_generating_function_rational
    mpmath.pade
    """
    from sympy.simplify import simplify as simp
    from sympy.simplify.radsimp import denom
    p1, q1 = [S.One], [S.Zero]
    p2, q2 = [S.Zero], [S.One]
    while len(l):
        b = 0
        while l[b]==0:
            b += 1
            if b == len(l):
                return
        m = [S.One/l[b]]
        for k in range(b+1, len(l)):
            s = 0
            for j in range(b, k):
                s -= l[j+1] * m[b-j-1]
            m.append(s/l[b])
        l = m
        a, l[0] = l[0], 0
        p = [0] * max(len(p2), b+len(p1))
        q = [0] * max(len(q2), b+len(q1))
        for k in range(len(p2)):
            p[k] = a*p2[k]
        for k in range(b, b+len(p1)):
            p[k] += p1[k-b]
        for k in range(len(q2)):
            q[k] = a*q2[k]
        for k in range(b, b+len(q1)):
            q[k] += q1[k-b]
        while p[-1]==0: p.pop()
        while q[-1]==0: q.pop()
        p1, p2 = p2, p
        q1, q2 = q2, q

        # yield result
        c = 1
        for x in p:
            c = lcm(c, denom(x))
        for x in q:
            c = lcm(c, denom(x))
        out = ( sum(c*e*X**k for k, e in enumerate(p))
              / sum(c*e*X**k for k, e in enumerate(q)) )
        if simplify:
            yield(simp(out))
        else:
            yield out
    return
<|endoftext|>from sympy.core import S, sympify, Expr, Dummy, Add, Mul
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.function import Function, PoleError, expand_power_base, expand_log
from sympy.core.sorting import default_sort_key
from sympy.functions.elementary.exponential import exp, log
from sympy.sets.sets import Complement
from sympy.utilities.iterables import uniq, is_sequence


class Order(Expr):
    r""" Represents the limiting behavior of some function.

    Explanation
    ===========

    The order of a function characterizes the function based on the limiting
    behavior of the function as it goes to some limit. Only taking the limit
    point to be a number is currently supported. This is expressed in
    big O notation [1]_.

    The formal definition for the order of a function `g(x)` about a point `a`
    is such that `g(x) = O(f(x))` as `x \rightarrow a` if and only if there
    exists a `\delta > 0` and an `M > 0` such that `|g(x)| \leq M|f(x)|` for
    `|x-a| < \delta`.  This is equivalent to `\limsup_{x \rightarrow a}
    |g(x)/f(x)| < \infty`.

    Let's illustrate it on the following example by taking the expansion of
    `\sin(x)` about 0:

    .. math ::
        \sin(x) = x - x^3/3! + O(x^5)

    where in this case `O(x^5) = x^5/5! - x^7/7! + \cdots`. By the definition
    of `O`, there is a `\delta > 0` and an `M` such that:

    .. math ::
        |x^5/5! - x^7/7! + ....| <= M|x^5| \text{ for } |x| < \delta

    or by the alternate definition:

    .. math ::
        \lim_{x \rightarrow 0} | (x^5/5! - x^7/7! + ....) / x^5| < \infty

    which surely is true, because

    .. math ::
        \lim_{x \rightarrow 0} | (x^5/5! - x^7/7! + ....) / x^5| = 1/5!


    As it is usually used, the order of a function can be intuitively thought
    of representing all terms of powers greater than the one specified. For
    example, `O(x^3)` corresponds to any terms proportional to `x^3,
    x^4,\ldots` and any higher power. For a polynomial, this leaves terms
    proportional to `x^2`, `x` and constants.

    Examples
    ========

    >>> from sympy import O, oo, cos, pi
    >>> from sympy.abc import x, y

    >>> O(x + x**2)
    O(x)
    >>> O(x + x**2, (x, 0))
    O(x)
    >>> O(x + x**2, (x, oo))
    O(x**2, (x, oo))

    >>> O(1 + x*y)
    O(1, x, y)
    >>> O(1 + x*y, (x, 0), (y, 0))
    O(1, x, y)
    >>> O(1 + x*y, (x, oo), (y, oo))
    O(x*y, (x, oo), (y, oo))

    >>> O(1) in O(1, x)
    True
    >>> O(1, x) in O(1)
    False
    >>> O(x) in O(1, x)
    True
    >>> O(x**2) in O(x)
    True

    >>> O(x)*x
    O(x**2)
    >>> O(x) - O(x)
    O(x)
    >>> O(cos(x))
    O(1)
    >>> O(cos(x), (x, pi/2))
    O(x - pi/2, (x, pi/2))

    References
    ==========

    .. [1] `Big O notation <https://en.wikipedia.org/wiki/Big_O_notation>`_

    Notes
    =====

    In ``O(f(x), x)`` the expression ``f(x)`` is assumed to have a leading
    term.  ``O(f(x), x)`` is automatically transformed to
    ``O(f(x).as_leading_term(x),x)``.

        ``O(expr*f(x), x)`` is ``O(f(x), x)``

        ``O(expr, x)`` is ``O(1)``

        ``O(0, x)`` is 0.

    Multivariate O is also supported:

        ``O(f(x, y), x, y)`` is transformed to
        ``O(f(x, y).as_leading_term(x,y).as_leading_term(y), x, y)``

    In the multivariate case, it is assumed the limits w.r.t. the various
    symbols commute.

    If no symbols are passed then all symbols in the expression are used
    and the limit point is assumed to be zero.

    """

    is_Order = True

    __slots__ = ()

    @cacheit
    def __new__(cls, expr, *args, **kwargs):
        expr = sympify(expr)

        if not args:
            if expr.is_Order:
                variables = expr.variables
                point = expr.point
            else:
                variables = list(expr.free_symbols)
                point = [S.Zero]*len(variables)
        else:
            args = list(args if is_sequence(args) else [args])
            variables, point = [], []
            if is_sequence(args[0]):
                for a in args:
                    v, p = list(map(sympify, a))
                    variables.append(v)
                    point.append(p)
            else:
                variables = list(map(sympify, args))
                point = [S.Zero]*len(variables)

        if not all(v.is_symbol for v in variables):
            raise TypeError('Variables are not symbols, got %s' % variables)

        if len(list(uniq(variables))) != len(variables):
            raise ValueError('Variables are supposed to be unique symbols, got %s' % variables)

        if expr.is_Order:
            expr_vp = dict(expr.args[1:])
            new_vp = dict(expr_vp)
            vp = dict(zip(variables, point))
            for v, p in vp.items():
                if v in new_vp.keys():
                    if p != new_vp[v]:
                        raise NotImplementedError(
                            "Mixing Order at different points is not supported.")
                else:
                    new_vp[v] = p
            if set(expr_vp.keys()) == set(new_vp.keys()):
                return expr
            else:
                variables = list(new_vp.keys())
                point = [new_vp[v] for v in variables]

        if expr is S.NaN:
            return S.NaN

        if any(x in p.free_symbols for x in variables for p in point):
            raise ValueError('Got %s as a point.' % point)

        if variables:
            if any(p != point[0] for p in point):
                raise NotImplementedError(
                    "Multivariable orders at different points are not supported.")
            if point[0] in (S.Infinity, S.Infinity*S.ImaginaryUnit):
                s = {k: 1/Dummy() for k in variables}
                rs = {1/v: 1/k for k, v in s.items()}
                ps = [S.Zero for p in point]
            elif point[0] in (S.NegativeInfinity, S.NegativeInfinity*S.ImaginaryUnit):
                s = {k: -1/Dummy() for k in variables}
                rs = {-1/v: -1/k for k, v in s.items()}
                ps = [S.Zero for p in point]
            elif point[0] is not S.Zero:
                s = {k: Dummy() + point[0] for k in variables}
                rs = {(v - point[0]).together(): k - point[0] for k, v in s.items()}
                ps = [S.Zero for p in point]
            else:
                s = ()
                rs = ()
                ps = list(point)

            expr = expr.subs(s)

            if expr.is_Add:
                expr = expr.factor()

            if s:
                args = tuple([r[0] for r in rs.items()])
            else:
                args = tuple(variables)

            if len(variables) > 1:
                # XXX: better way?  We need this expand() to
                # workaround e.g: expr = x*(x + y).
                # (x*(x + y)).as_leading_term(x, y) currently returns
                # x*y (wrong order term!).  That's why we want to deal with
                # expand()'ed expr (handled in "if expr.is_Add" branch below).
                expr = expr.expand()

            old_expr = None
            while old_expr != expr:
                old_expr = expr
                if expr.is_Add:
                    lst = expr.extract_leading_order(args)
                    expr = Add(*[f.expr for (e, f) in lst])

                elif expr:
                    try:
                        expr = expr.as_leading_term(*args)
                    except PoleError:
                        if isinstance(expr, Function) or\
                                all(isinstance(arg, Function) for arg in expr.args):
                            # It is not possible to simplify an expression
                            # containing only functions (which raise error on
                            # call to leading term) further
                            pass
                        else:
                            orders = []
                            pts = tuple(zip(args, ps))
                            for arg in expr.args:
                                try:
                                    lt = arg.as_leading_term(*args)
                                except PoleError:
                                    lt = arg
                                if lt not in args:
                                    order = Order(lt)
                                else:
                                    order = Order(lt, *pts)
                                orders.append(order)
                            if expr.is_Add:
                                new_expr = Order(Add(*orders), *pts)
                                if new_expr.is_Add:
                                    new_expr = Order(Add(*[a.expr for a in new_expr.args]), *pts)
                                expr = new_expr.expr
                            elif expr.is_Mul:
                                expr = Mul(*[a.expr for a in orders])
                            elif expr.is_Pow:
                                e = expr.exp
                                b = expr.base
                                expr = exp(e * log(b))

                    # It would probably be better to handle this somewhere
                    # else. This is needed for a testcase in which there is a
                    # symbol with the assumptions zero=True.
                    if expr.is_zero:
                        expr = S.Zero
                    else:
                        expr = expr.as_independent(*args, as_Add=False)[1]

                    expr = expand_power_base(expr)
                    expr = expand_log(expr)

                    if len(args) == 1:
                        # The definition of O(f(x)) symbol explicitly stated that
                        # the argument of f(x) is irrelevant.  That's why we can
                        # combine some power exponents (only "on top" of the
                        # expression tree for f(x)), e.g.:
                        # x**p * (-x)**q -> x**(p+q) for real p, q.
                        x = args[0]
                        margs = list(Mul.make_args(
                            expr.as_independent(x, as_Add=False)[1]))

                        for i, t in enumerate(margs):
                            if t.is_Pow:
                                b, q = t.args
                                if b in (x, -x) and q.is_real and not q.has(x):
                                    margs[i] = x**q
                                elif b.is_Pow and not b.exp.has(x):
                                    b, r = b.args
                                    if b in (x, -x) and r.is_real:
                                        margs[i] = x**(r*q)
                                elif b.is_Mul and b.args[0] is S.NegativeOne:
                                    b = -b
                                    if b.is_Pow and not b.exp.has(x):
                                        b, r = b.args
                                        if b in (x, -x) and r.is_real:
                                            margs[i] = x**(r*q)

                        expr = Mul(*margs)

            expr = expr.subs(rs)

        if expr.is_Order:
            expr = expr.expr

        if not expr.has(*variables) and not expr.is_zero:
            expr = S.One

        # create Order instance:
        vp = dict(zip(variables, point))
        variables.sort(key=default_sort_key)
        point = [vp[v] for v in variables]
        args = (expr,) + Tuple(*zip(variables, point))
        obj = Expr.__new__(cls, *args)
        return obj

    def _eval_nseries(self, x, n, logx, cdir=0):
        return self

    @property
    def expr(self):
        return self.args[0]

    @property
    def variables(self):
        if self.args[1:]:
            return tuple(x[0] for x in self.args[1:])
        else:
            return ()

    @property
    def point(self):
        if self.args[1:]:
            return tuple(x[1] for x in self.args[1:])
        else:
            return ()

    @property
    def free_symbols(self):
        return self.expr.free_symbols | set(self.variables)

    def _eval_power(b, e):
        if e.is_Number and e.is_nonnegative:
            return b.func(b.expr ** e, *b.args[1:])
        if e == O(1):
            return b
        return

    def as_expr_variables(self, order_symbols):
        if order_symbols is None:
            order_symbols = self.args[1:]
        else:
            if (not all(o[1] == order_symbols[0][1] for o in order_symbols) and
                    not all(p == self.point[0] for p in self.point)):  # pragma: no cover
                raise NotImplementedError('Order at points other than 0 '
                    'or oo not supported, got %s as a point.' % self.point)
            if order_symbols and order_symbols[0][1] != self.point[0]:
                raise NotImplementedError(
                        "Multiplying Order at different points is not supported.")
            order_symbols = dict(order_symbols)
            for s, p in dict(self.args[1:]).items():
                if s not in order_symbols.keys():
                    order_symbols[s] = p
            order_symbols = sorted(order_symbols.items(), key=lambda x: default_sort_key(x[0]))
        return self.expr, tuple(order_symbols)

    def removeO(self):
        return S.Zero

    def getO(self):
        return self

    @cacheit
    def contains(self, expr):
        r"""
        Return True if expr belongs to Order(self.expr, \*self.variables).
        Return False if self belongs to expr.
        Return None if the inclusion relation cannot be determined
        (e.g. when self and expr have different symbols).
        """
        expr = sympify(expr)
        if expr.is_zero:
            return True
        if expr is S.NaN:
            return False
        point = self.point[0] if self.point else S.Zero
        if expr.is_Order:
            if (any(p != point for p in expr.point) or
                   any(p != point for p in self.point)):
                return None
            if expr.expr == self.expr:
                # O(1) + O(1), O(1) + O(1, x), etc.
                return all(x in self.args[1:] for x in expr.args[1:])
            if expr.expr.is_Add:
                return all(self.contains(x) for x in expr.expr.args)
            if self.expr.is_Add and point.is_zero:
                return any(self.func(x, *self.args[1:]).contains(expr)
                            for x in self.expr.args)
            if self.variables and expr.variables:
                common_symbols = tuple(
                    [s for s in self.variables if s in expr.variables])
            elif self.variables:
                common_symbols = self.variables
            else:
                common_symbols = expr.variables
            if not common_symbols:
                return None
            if (self.expr.is_Pow and len(self.variables) == 1
                and self.variables == expr.variables):
                    symbol = self.variables[0]
                    other = expr.expr.as_independent(symbol, as_Add=False)[1]
                    if (other.is_Pow and other.base == symbol and
                        self.expr.base == symbol):
                            if point.is_zero:
                                rv = (self.expr.exp - other.exp).is_nonpositive
                            if point.is_infinite:
                                rv = (self.expr.exp - other.exp).is_nonnegative
                            if rv is not None:
                                return rv

            from sympy.simplify.powsimp import powsimp
            r = None
            ratio = self.expr/expr.expr
            ratio = powsimp(ratio, deep=True, combine='exp')
            for s in common_symbols:
                from sympy.series.limits import Limit
                l = Limit(ratio, s, point).doit(heuristics=False)
                if not isinstance(l, Limit):
                    l = l != 0
                else:
                    l = None
                if r is None:
                    r = l
                else:
                    if r != l:
                        return
            return r

        if self.expr.is_Pow and len(self.variables) == 1:
            symbol = self.variables[0]
            other = expr.as_independent(symbol, as_Add=False)[1]
            if (other.is_Pow and other.base == symbol and
                self.expr.base == symbol):
                    if point.is_zero:
                        rv = (self.expr.exp - other.exp).is_nonpositive
                    if point.is_infinite:
                        rv = (self.expr.exp - other.exp).is_nonnegative
                    if rv is not None:
                        return rv

        obj = self.func(expr, *self.args[1:])
        return self.contains(obj)

    def __contains__(self, other):
        result = self.contains(other)
        if result is None:
            raise TypeError('contains did not evaluate to a bool')
        return result

    def _eval_subs(self, old, new):
        if old in self.variables:
            newexpr = self.expr.subs(old, new)
            i = self.variables.index(old)
            newvars = list(self.variables)
            newpt = list(self.point)
            if new.is_symbol:
                newvars[i] = new
            else:
                syms = new.free_symbols
                if len(syms) == 1 or old in syms:
                    if old in syms:
                        var = self.variables[i]
                    else:
                        var = syms.pop()
                    # First, try to substitute self.point in the "new"
                    # expr to see if this is a fixed point.
                    # E.g.  O(y).subs(y, sin(x))
                    from sympy import limit
                    if new.has(Order) and limit(new.getO().expr, var, new.getO().point[0]) == self.point[i]:
                        point = new.getO().point[0]
                        return Order(newexpr, *zip([var], [point]))
                    else:
                        point = new.subs(var, self.point[i])
                    if point != self.point[i]:
                        from sympy.solvers.solveset import solveset
                        d = Dummy()
                        sol = solveset(old - new.subs(var, d), d)
                        if isinstance(sol, Complement):
                            e1 = sol.args[0]
                            e2 = sol.args[1]
                            sol = set(e1) - set(e2)
                        res = [dict(zip((d, ), sol))]
                        point = d.subs(res[0]).limit(old, self.point[i])
                    newvars[i] = var
                    newpt[i] = point
                elif old not in syms:
                    del newvars[i], newpt[i]
                    if not syms and new == self.point[i]:
                        newvars.extend(syms)
                        newpt.extend([S.Zero]*len(syms))
                else:
                    return
            return Order(newexpr, *zip(newvars, newpt))

    def _eval_conjugate(self):
        expr = self.expr._eval_conjugate()
        if expr is not None:
            return self.func(expr, *self.args[1:])

    def _eval_derivative(self, x):
        return self.func(self.expr.diff(x), *self.args[1:]) or self

    def _eval_transpose(self):
        expr = self.expr._eval_transpose()
        if expr is not None:
            return self.func(expr, *self.args[1:])

    def __neg__(self):
        return self

O = Order
<|endoftext|>def finite_diff(expression, variable, increment=1):
    """
    Takes as input a polynomial expression and the variable used to construct
    it and returns the difference between function's value when the input is
    incremented to 1 and the original function value. If you want an increment
    other than one supply it as a third argument.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.series.kauers import finite_diff
    >>> finite_diff(x**2, x)
    2*x + 1
    >>> finite_diff(y**3 + 2*y**2 + 3*y + 4, y)
    3*y**2 + 7*y + 6
    >>> finite_diff(x**2 + 3*x + 8, x, 2)
    4*x + 10
    >>> finite_diff(z**3 + 8*z, z, 3)
    9*z**2 + 27*z + 51
    """
    expression = expression.expand()
    expression2 = expression.subs(variable, variable + increment)
    expression2 = expression2.expand()
    return expression2 - expression

def finite_diff_kauers(sum):
    """
    Takes as input a Sum instance and returns the difference between the sum
    with the upper index incremented by 1 and the original sum. For example,
    if S(n) is a sum, then finite_diff_kauers will return S(n + 1) - S(n).

    Examples
    ========

    >>> from sympy.series.kauers import finite_diff_kauers
    >>> from sympy import Sum
    >>> from sympy.abc import x, y, m, n, k
    >>> finite_diff_kauers(Sum(k, (k, 1, n)))
    n + 1
    >>> finite_diff_kauers(Sum(1/k, (k, 1, n)))
    1/(n + 1)
    >>> finite_diff_kauers(Sum((x*y**2), (x, 1, n), (y, 1, m)))
    (m + 1)**2*(n + 1)
    >>> finite_diff_kauers(Sum((x*y), (x, 1, m), (y, 1, n)))
    (m + 1)*(n + 1)
    """
    function = sum.function
    for l in sum.limits:
        function = function.subs(l[0], l[- 1] + 1)
    return function
<|endoftext|>from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core import S, Symbol, Add, sympify, Expr, PoleError, Mul
from sympy.core.exprtools import factor_terms
from sympy.core.numbers import Float, _illegal
from sympy.core.function import AppliedUndef
from sympy.core.symbol import Dummy
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (Abs, sign, arg, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.special.gamma_functions import gamma
from sympy.polys import PolynomialError, factor
from sympy.series.order import Order
from .gruntz import gruntz

def limit(e, z, z0, dir="+"):
    """Computes the limit of ``e(z)`` at the point ``z0``.

    Parameters
    ==========

    e : expression, the limit of which is to be taken

    z : symbol representing the variable in the limit.
        Other symbols are treated as constants. Multivariate limits
        are not supported.

    z0 : the value toward which ``z`` tends. Can be any expression,
        including ``oo`` and ``-oo``.

    dir : string, optional (default: "+")
        The limit is bi-directional if ``dir="+-"``, from the right
        (z->z0+) if ``dir="+"``, and from the left (z->z0-) if
        ``dir="-"``. For infinite ``z0`` (``oo`` or ``-oo``), the ``dir``
        argument is determined from the direction of the infinity
        (i.e., ``dir="-"`` for ``oo``).

    Examples
    ========

    >>> from sympy import limit, sin, oo
    >>> from sympy.abc import x
    >>> limit(sin(x)/x, x, 0)
    1
    >>> limit(1/x, x, 0) # default dir='+'
    oo
    >>> limit(1/x, x, 0, dir="-")
    -oo
    >>> limit(1/x, x, 0, dir='+-')
    zoo
    >>> limit(1/x, x, oo)
    0

    Notes
    =====

    First we try some heuristics for easy and frequent cases like "x", "1/x",
    "x**2" and similar, so that it's fast. For all other cases, we use the
    Gruntz algorithm (see the gruntz() function).

    See Also
    ========

     limit_seq : returns the limit of a sequence.
    """

    return Limit(e, z, z0, dir).doit(deep=False)


def heuristics(e, z, z0, dir):
    """Computes the limit of an expression term-wise.
    Parameters are the same as for the ``limit`` function.
    Works with the arguments of expression ``e`` one by one, computing
    the limit of each and then combining the results. This approach
    works only for simple limits, but it is fast.
    """

    rv = None
    if z0 is S.Infinity:
        rv = limit(e.subs(z, 1/z), z, S.Zero, "+")
        if isinstance(rv, Limit):
            return
    elif (e.is_Mul or e.is_Add or e.is_Pow or (e.is_Function and not isinstance(e, AppliedUndef))):
        r = []
        from sympy.simplify.simplify import together
        for a in e.args:
            l = limit(a, z, z0, dir)
            if l.has(S.Infinity) and l.is_finite is None:
                if isinstance(e, Add):
                    m = factor_terms(e)
                    if not isinstance(m, Mul): # try together
                        m = together(m)
                    if not isinstance(m, Mul): # try factor if the previous methods failed
                        m = factor(e)
                    if isinstance(m, Mul):
                        return heuristics(m, z, z0, dir)
                    return
                return
            elif isinstance(l, Limit):
                return
            elif l is S.NaN:
                return
            else:
                r.append(l)
        if r:
            rv = e.func(*r)
            if rv is S.NaN and e.is_Mul and any(isinstance(rr, AccumBounds) for rr in r):
                r2 = []
                e2 = []
                for ii, rval in enumerate(r):
                    if isinstance(rval, AccumBounds):
                        r2.append(rval)
                    else:
                        e2.append(e.args[ii])

                if len(e2) > 0:
                    e3 = Mul(*e2).simplify()
                    l = limit(e3, z, z0, dir)
                    rv = l * Mul(*r2)

            if rv is S.NaN:
                try:
                    from sympy.simplify.ratsimp import ratsimp
                    rat_e = ratsimp(e)
                except PolynomialError:
                    return
                if rat_e is S.NaN or rat_e == e:
                    return
                return limit(rat_e, z, z0, dir)
    return rv


class Limit(Expr):
    """Represents an unevaluated limit.

    Examples
    ========

    >>> from sympy import Limit, sin
    >>> from sympy.abc import x
    >>> Limit(sin(x)/x, x, 0)
    Limit(sin(x)/x, x, 0, dir='+')
    >>> Limit(1/x, x, 0, dir="-")
    Limit(1/x, x, 0, dir='-')

    """

    def __new__(cls, e, z, z0, dir="+"):
        e = sympify(e)
        z = sympify(z)
        z0 = sympify(z0)

        if z0 in (S.Infinity, S.ImaginaryUnit*S.Infinity):
            dir = "-"
        elif z0 in (S.NegativeInfinity, S.ImaginaryUnit*S.NegativeInfinity):
            dir = "+"

        if(z0.has(z)):
            raise NotImplementedError("Limits approaching a variable point are"
                    " not supported (%s -> %s)" % (z, z0))
        if isinstance(dir, str):
            dir = Symbol(dir)
        elif not isinstance(dir, Symbol):
            raise TypeError("direction must be of type basestring or "
                    "Symbol, not %s" % type(dir))
        if str(dir) not in ('+', '-', '+-'):
            raise ValueError("direction must be one of '+', '-' "
                    "or '+-', not %s" % dir)

        obj = Expr.__new__(cls)
        obj._args = (e, z, z0, dir)
        return obj


    @property
    def free_symbols(self):
        e = self.args[0]
        isyms = e.free_symbols
        isyms.difference_update(self.args[1].free_symbols)
        isyms.update(self.args[2].free_symbols)
        return isyms


    def pow_heuristics(self, e):
        _, z, z0, _ = self.args
        b1, e1 = e.base, e.exp
        if not b1.has(z):
            res = limit(e1*log(b1), z, z0)
            return exp(res)

        ex_lim = limit(e1, z, z0)
        base_lim = limit(b1, z, z0)

        if base_lim is S.One:
            if ex_lim in (S.Infinity, S.NegativeInfinity):
                res = limit(e1*(b1 - 1), z, z0)
                return exp(res)
        if base_lim is S.NegativeInfinity and ex_lim is S.Infinity:
            return S.ComplexInfinity


    def doit(self, **hints):
        """Evaluates the limit.

        Parameters
        ==========

        deep : bool, optional (default: True)
            Invoke the ``doit`` method of the expressions involved before
            taking the limit.

        hints : optional keyword arguments
            To be passed to ``doit`` methods; only used if deep is True.
        """

        e, z, z0, dir = self.args

        if str(dir) == '+-':
            r = limit(e, z, z0, dir='+')
            l = limit(e, z, z0, dir='-')
            if isinstance(r, Limit) and isinstance(l, Limit):
                if r.args[0] == l.args[0]:
                    return self
            if r == l:
                return l
            if r.is_infinite and l.is_infinite:
                return S.ComplexInfinity
            raise ValueError("The limit does not exist since "
                             "left hand limit = %s and right hand limit = %s"
                             % (l, r))

        if z0 is S.ComplexInfinity:
            raise NotImplementedError("Limits at complex "
                                    "infinity are not implemented")

        if z0.is_infinite:
            cdir = sign(z0)
            cdir = cdir/abs(cdir)
            e = e.subs(z, cdir*z)
            dir = "-"
            z0 = S.Infinity

        if hints.get('deep', True):
            e = e.doit(**hints)
            z = z.doit(**hints)
            z0 = z0.doit(**hints)

        if e == z:
            return z0

        if not e.has(z):
            return e

        if z0 is S.NaN:
            return S.NaN

        if e.has(*_illegal):
            return self

        if e.is_Order:
            return Order(limit(e.expr, z, z0), *e.args[1:])

        cdir = S.Zero
        if str(dir) == "+":
            cdir = S.One
        elif str(dir) == "-":
            cdir = S.NegativeOne

        def set_signs(expr):
            if not expr.args:
                return expr
            newargs = tuple(set_signs(arg) for arg in expr.args)
            if newargs != expr.args:
                expr = expr.func(*newargs)
            abs_flag = isinstance(expr, Abs)
            arg_flag = isinstance(expr, arg)
            sign_flag = isinstance(expr, sign)
            if abs_flag or sign_flag or arg_flag:
                try:
                    sig = limit(expr.args[0], z, z0, dir)
                    if sig.is_zero:
                        sig = limit(1/expr.args[0], z, z0, dir)
                except NotImplementedError:
                    return expr
                else:
                    if sig.is_extended_real:
                        if (sig < 0) == True:
                            return (-expr.args[0] if abs_flag else
                                    S.NegativeOne if sign_flag else S.Pi)
                        elif (sig > 0) == True:
                            return (expr.args[0] if abs_flag else
                                    S.One if sign_flag else S.Zero)
            return expr

        if e.has(Float):
            # Convert floats like 0.5 to exact SymPy numbers like S.Half, to
            # prevent rounding errors which can lead to unexpected execution
            # of conditional blocks that work on comparisons
            # Also see comments in https://github.com/sympy/sympy/issues/19453
            from sympy.simplify.simplify import nsimplify
            e = nsimplify(e)
        e = set_signs(e)


        if e.is_meromorphic(z, z0):
            if z0 is S.Infinity:
                newe = e.subs(z, 1/z)
                # cdir changes sign as oo- should become 0+
                cdir = -cdir
            else:
                newe = e.subs(z, z + z0)
            try:
                coeff, ex = newe.leadterm(z, cdir=cdir)
            except ValueError:
                pass
            else:
                if ex > 0:
                    return S.Zero
                elif ex == 0:
                    return coeff
                if cdir == 1 or not(int(ex) & 1):
                    return S.Infinity*sign(coeff)
                elif cdir == -1:
                    return S.NegativeInfinity*sign(coeff)
                else:
                    return S.ComplexInfinity

        if z0 is S.Infinity:
            if e.is_Mul:
                e = factor_terms(e)
            dummy = Dummy('z', positive=z.is_positive, negative=z.is_negative, real=z.is_real)
            newe = e.subs(z, 1/dummy)
            # cdir changes sign as oo- should become 0+
            cdir = -cdir
            newz = dummy
        else:
            newe = e.subs(z, z + z0)
            newz = z
        try:
            coeff, ex = newe.leadterm(newz, cdir=cdir)
        except (ValueError, NotImplementedError, PoleError):
            # The NotImplementedError catching is for custom functions
            from sympy.simplify.powsimp import powsimp
            e = powsimp(e)
            if e.is_Pow:
                r = self.pow_heuristics(e)
                if r is not None:
                    return r
            try:
                coeff = newe.as_leading_term(newz, cdir=cdir)
                if coeff != newe and (coeff.has(exp) or coeff.has(S.Exp1)):
                    return gruntz(coeff, newz, 0, "-" if re(cdir).is_negative else "+")
            except (ValueError, NotImplementedError, PoleError):
                pass
        else:
            if isinstance(coeff, AccumBounds) and ex == S.Zero:
                return coeff
            if coeff.has(S.Infinity, S.NegativeInfinity, S.ComplexInfinity, S.NaN):
                return self
            if not coeff.has(newz):
                if ex.is_positive:
                    return S.Zero
                elif ex == 0:
                    return coeff
                elif ex.is_negative:
                    if cdir == 1:
                        return S.Infinity*sign(coeff)
                    elif cdir == -1:
                        return S.NegativeInfinity*sign(coeff)*S.NegativeOne**(S.One + ex)
                    else:
                        return S.ComplexInfinity
                else:
                    raise NotImplementedError("Not sure of sign of %s" % ex)

        # gruntz fails on factorials but works with the gamma function
        # If no factorial term is present, e should remain unchanged.
        # factorial is defined to be zero for negative inputs (which
        # differs from gamma) so only rewrite for non-negative z0.
        if z0.is_extended_nonnegative:
            e = e.rewrite(factorial, gamma)

        l = None

        try:
            r = gruntz(e, z, z0, dir)
            if r is S.NaN or l is S.NaN:
                raise PoleError()
        except (PoleError, ValueError):
            if l is not None:
                raise
            r = heuristics(e, z, z0, dir)
            if r is None:
                return self

        return r
<|endoftext|>"""
Convergence acceleration / extrapolation methods for series and
sequences.

References:
Carl M. Bender & Steven A. Orszag, "Advanced Mathematical Methods for
Scientists and Engineers: Asymptotic Methods and Perturbation Theory",
Springer 1999. (Shanks transformation: pp. 368-375, Richardson
extrapolation: pp. 375-377.)
"""

from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.functions.combinatorial.factorials import factorial


def richardson(A, k, n, N):
    """
    Calculate an approximation for lim k->oo A(k) using Richardson
    extrapolation with the terms A(n), A(n+1), ..., A(n+N+1).
    Choosing N ~= 2*n often gives good results.

    Examples
    ========

    A simple example is to calculate exp(1) using the limit definition.
    This limit converges slowly; n = 100 only produces two accurate
    digits:

        >>> from sympy.abc import n
        >>> e = (1 + 1/n)**n
        >>> print(round(e.subs(n, 100).evalf(), 10))
        2.7048138294

    Richardson extrapolation with 11 appropriately chosen terms gives
    a value that is accurate to the indicated precision:

        >>> from sympy import E
        >>> from sympy.series.acceleration import richardson
        >>> print(round(richardson(e, n, 10, 20).evalf(), 10))
        2.7182818285
        >>> print(round(E.evalf(), 10))
        2.7182818285

    Another useful application is to speed up convergence of series.
    Computing 100 terms of the zeta(2) series 1/k**2 yields only
    two accurate digits:

        >>> from sympy.abc import k, n
        >>> from sympy import Sum
        >>> A = Sum(k**-2, (k, 1, n))
        >>> print(round(A.subs(n, 100).evalf(), 10))
        1.6349839002

    Richardson extrapolation performs much better:

        >>> from sympy import pi
        >>> print(round(richardson(A, n, 10, 20).evalf(), 10))
        1.6449340668
        >>> print(round(((pi**2)/6).evalf(), 10))     # Exact value
        1.6449340668

    """
    s = S.Zero
    for j in range(0, N + 1):
        s += (A.subs(k, Integer(n + j)).doit() * (n + j)**N *
              S.NegativeOne**(j + N) / (factorial(j) * factorial(N - j)))
    return s


def shanks(A, k, n, m=1):
    """
    Calculate an approximation for lim k->oo A(k) using the n-term Shanks
    transformation S(A)(n). With m > 1, calculate the m-fold recursive
    Shanks transformation S(S(...S(A)...))(n).

    The Shanks transformation is useful for summing Taylor series that
    converge slowly near a pole or singularity, e.g. for log(2):

        >>> from sympy.abc import k, n
        >>> from sympy import Sum, Integer
        >>> from sympy.series.acceleration import shanks
        >>> A = Sum(Integer(-1)**(k+1) / k, (k, 1, n))
        >>> print(round(A.subs(n, 100).doit().evalf(), 10))
        0.6881721793
        >>> print(round(shanks(A, n, 25).evalf(), 10))
        0.6931396564
        >>> print(round(shanks(A, n, 25, 5).evalf(), 10))
        0.6931471806

    The correct value is 0.6931471805599453094172321215.
    """
    table = [A.subs(k, Integer(j)).doit() for j in range(n + m + 2)]
    table2 = table.copy()

    for i in range(1, m + 1):
        for j in range(i, n + m + 1):
            x, y, z = table[j - 1], table[j], table[j + 1]
            table2[j] = (z*x - y**2) / (z + x - 2*y)
        table = table2.copy()
    return table[n]
<|endoftext|>"""Formal Power Series"""

from collections import defaultdict

from sympy.core.numbers import (nan, oo, zoo)
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import Derivative, Function, expand
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.relational import Eq
from sympy.sets.sets import Interval
from sympy.core.singleton import S
from sympy.core.symbol import Wild, Dummy, symbols, Symbol
from sympy.core.sympify import sympify
from sympy.discrete.convolutions import convolution
from sympy.functions.combinatorial.factorials import binomial, factorial, rf
from sympy.functions.combinatorial.numbers import bell
from sympy.functions.elementary.integers import floor, frac, ceiling
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.series.limits import Limit
from sympy.series.order import Order
from sympy.series.sequences import sequence
from sympy.series.series_class import SeriesBase
from sympy.utilities.iterables import iterable



def rational_algorithm(f, x, k, order=4, full=False):
    """
    Rational algorithm for computing
    formula of coefficients of Formal Power Series
    of a function.

    Explanation
    ===========

    Applicable when f(x) or some derivative of f(x)
    is a rational function in x.

    :func:`rational_algorithm` uses :func:`~.apart` function for partial fraction
    decomposition. :func:`~.apart` by default uses 'undetermined coefficients
    method'. By setting ``full=True``, 'Bronstein's algorithm' can be used
    instead.

    Looks for derivative of a function up to 4'th order (by default).
    This can be overridden using order option.

    Parameters
    ==========

    x : Symbol
    order : int, optional
        Order of the derivative of ``f``, Default is 4.
    full : bool

    Returns
    =======

    formula : Expr
    ind : Expr
        Independent terms.
    order : int
    full : bool

    Examples
    ========

    >>> from sympy import log, atan
    >>> from sympy.series.formal import rational_algorithm as ra
    >>> from sympy.abc import x, k

    >>> ra(1 / (1 - x), x, k)
    (1, 0, 0)
    >>> ra(log(1 + x), x, k)
    (-1/((-1)**k*k), 0, 1)

    >>> ra(atan(x), x, k, full=True)
    ((-I/(2*(-I)**k) + I/(2*I**k))/k, 0, 1)

    Notes
    =====

    By setting ``full=True``, range of admissible functions to be solved using
    ``rational_algorithm`` can be increased. This option should be used
    carefully as it can significantly slow down the computation as ``doit`` is
    performed on the :class:`~.RootSum` object returned by the :func:`~.apart`
    function. Use ``full=False`` whenever possible.

    See Also
    ========

    sympy.polys.partfrac.apart

    References
    ==========

    .. [1] Formal Power Series - Dominik Gruntz, Wolfram Koepf
    .. [2] Power Series in Computer Algebra - Wolfram Koepf

    """
    from sympy.polys import RootSum, apart
    from sympy.integrals import integrate

    diff = f
    ds = []  # list of diff

    for i in range(order + 1):
        if i:
            diff = diff.diff(x)

        if diff.is_rational_function(x):
            coeff, sep = S.Zero, S.Zero

            terms = apart(diff, x, full=full)
            if terms.has(RootSum):
                terms = terms.doit()

            for t in Add.make_args(terms):
                num, den = t.as_numer_denom()
                if not den.has(x):
                    sep += t
                else:
                    if isinstance(den, Mul):
                        # m*(n*x - a)**j -> (n*x - a)**j
                        ind = den.as_independent(x)
                        den = ind[1]
                        num /= ind[0]

                    # (n*x - a)**j -> (x - b)
                    den, j = den.as_base_exp()
                    a, xterm = den.as_coeff_add(x)

                    # term -> m/x**n
                    if not a:
                        sep += t
                        continue

                    xc = xterm[0].coeff(x)
                    a /= -xc
                    num /= xc**j

                    ak = ((-1)**j * num *
                          binomial(j + k - 1, k).rewrite(factorial) /
                          a**(j + k))
                    coeff += ak

            # Hacky, better way?
            if coeff.is_zero:
                return None
            if (coeff.has(x) or coeff.has(zoo) or coeff.has(oo) or
                    coeff.has(nan)):
                return None

            for j in range(i):
                coeff = (coeff / (k + j + 1))
                sep = integrate(sep, x)
                sep += (ds.pop() - sep).limit(x, 0)  # constant of integration
            return (coeff.subs(k, k - i), sep, i)

        else:
            ds.append(diff)

    return None


def rational_independent(terms, x):
    """
    Returns a list of all the rationally independent terms.

    Examples
    ========

    >>> from sympy import sin, cos
    >>> from sympy.series.formal import rational_independent
    >>> from sympy.abc import x

    >>> rational_independent([cos(x), sin(x)], x)
    [cos(x), sin(x)]
    >>> rational_independent([x**2, sin(x), x*sin(x), x**3], x)
    [x**3 + x**2, x*sin(x) + sin(x)]
    """
    if not terms:
        return []

    ind = terms[0:1]

    for t in terms[1:]:
        n = t.as_independent(x)[1]
        for i, term in enumerate(ind):
            d = term.as_independent(x)[1]
            q = (n / d).cancel()
            if q.is_rational_function(x):
                ind[i] += t
                break
        else:
            ind.append(t)
    return ind


def simpleDE(f, x, g, order=4):
    r"""
    Generates simple DE.

    Explanation
    ===========

    DE is of the form

    .. math::
        f^k(x) + \sum\limits_{j=0}^{k-1} A_j f^j(x) = 0

    where :math:`A_j` should be rational function in x.

    Generates DE's upto order 4 (default). DE's can also have free parameters.

    By increasing order, higher order DE's can be found.

    Yields a tuple of (DE, order).
    """
    from sympy.solvers.solveset import linsolve

    a = symbols('a:%d' % (order))

    def _makeDE(k):
        eq = f.diff(x, k) + Add(*[a[i]*f.diff(x, i) for i in range(0, k)])
        DE = g(x).diff(x, k) + Add(*[a[i]*g(x).diff(x, i) for i in range(0, k)])
        return eq, DE

    found = False
    for k in range(1, order + 1):
        eq, DE = _makeDE(k)
        eq = eq.expand()
        terms = eq.as_ordered_terms()
        ind = rational_independent(terms, x)
        if found or len(ind) == k:
            sol = dict(zip(a, (i for s in linsolve(ind, a[:k]) for i in s)))
            if sol:
                found = True
                DE = DE.subs(sol)
            DE = DE.as_numer_denom()[0]
            DE = DE.factor().as_coeff_mul(Derivative)[1][0]
            yield DE.collect(Derivative(g(x))), k


def exp_re(DE, r, k):
    """Converts a DE with constant coefficients (explike) into a RE.

    Explanation
    ===========

    Performs the substitution:

    .. math::
        f^j(x) \\to r(k + j)

    Normalises the terms so that lowest order of a term is always r(k).

    Examples
    ========

    >>> from sympy import Function, Derivative
    >>> from sympy.series.formal import exp_re
    >>> from sympy.abc import x, k
    >>> f, r = Function('f'), Function('r')

    >>> exp_re(-f(x) + Derivative(f(x)), r, k)
    -r(k) + r(k + 1)
    >>> exp_re(Derivative(f(x), x) + Derivative(f(x), (x, 2)), r, k)
    r(k) + r(k + 1)

    See Also
    ========

    sympy.series.formal.hyper_re
    """
    RE = S.Zero

    g = DE.atoms(Function).pop()

    mini = None
    for t in Add.make_args(DE):
        coeff, d = t.as_independent(g)
        if isinstance(d, Derivative):
            j = d.derivative_count
        else:
            j = 0
        if mini is None or j < mini:
            mini = j
        RE += coeff * r(k + j)
    if mini:
        RE = RE.subs(k, k - mini)
    return RE


def hyper_re(DE, r, k):
    """
    Converts a DE into a RE.

    Explanation
    ===========

    Performs the substitution:

    .. math::
        x^l f^j(x) \\to (k + 1 - l)_j . a_{k + j - l}

    Normalises the terms so that lowest order of a term is always r(k).

    Examples
    ========

    >>> from sympy import Function, Derivative
    >>> from sympy.series.formal import hyper_re
    >>> from sympy.abc import x, k
    >>> f, r = Function('f'), Function('r')

    >>> hyper_re(-f(x) + Derivative(f(x)), r, k)
    (k + 1)*r(k + 1) - r(k)
    >>> hyper_re(-x*f(x) + Derivative(f(x), (x, 2)), r, k)
    (k + 2)*(k + 3)*r(k + 3) - r(k)

    See Also
    ========

    sympy.series.formal.exp_re
    """
    RE = S.Zero

    g = DE.atoms(Function).pop()
    x = g.atoms(Symbol).pop()

    mini = None
    for t in Add.make_args(DE.expand()):
        coeff, d = t.as_independent(g)
        c, v = coeff.as_independent(x)
        l = v.as_coeff_exponent(x)[1]
        if isinstance(d, Derivative):
            j = d.derivative_count
        else:
            j = 0
        RE += c * rf(k + 1 - l, j) * r(k + j - l)
        if mini is None or j - l < mini:
            mini = j - l

    RE = RE.subs(k, k - mini)

    m = Wild('m')
    return RE.collect(r(k + m))


def _transformation_a(f, x, P, Q, k, m, shift):
    f *= x**(-shift)
    P = P.subs(k, k + shift)
    Q = Q.subs(k, k + shift)
    return f, P, Q, m


def _transformation_c(f, x, P, Q, k, m, scale):
    f = f.subs(x, x**scale)
    P = P.subs(k, k / scale)
    Q = Q.subs(k, k / scale)
    m *= scale
    return f, P, Q, m


def _transformation_e(f, x, P, Q, k, m):
    f = f.diff(x)
    P = P.subs(k, k + 1) * (k + m + 1)
    Q = Q.subs(k, k + 1) * (k + 1)
    return f, P, Q, m


def _apply_shift(sol, shift):
    return [(res, cond + shift) for res, cond in sol]


def _apply_scale(sol, scale):
    return [(res, cond / scale) for res, cond in sol]


def _apply_integrate(sol, x, k):
    return [(res / ((cond + 1)*(cond.as_coeff_Add()[1].coeff(k))), cond + 1)
            for res, cond in sol]


def _compute_formula(f, x, P, Q, k, m, k_max):
    """Computes the formula for f."""
    from sympy.polys import roots

    sol = []
    for i in range(k_max + 1, k_max + m + 1):
        if (i < 0) == True:
            continue
        r = f.diff(x, i).limit(x, 0) / factorial(i)
        if r.is_zero:
            continue

        kterm = m*k + i
        res = r

        p = P.subs(k, kterm)
        q = Q.subs(k, kterm)
        c1 = p.subs(k, 1/k).leadterm(k)[0]
        c2 = q.subs(k, 1/k).leadterm(k)[0]
        res *= (-c1 / c2)**k

        res *= Mul(*[rf(-r, k)**mul for r, mul in roots(p, k).items()])
        res /= Mul(*[rf(-r, k)**mul for r, mul in roots(q, k).items()])

        sol.append((res, kterm))

    return sol


def _rsolve_hypergeometric(f, x, P, Q, k, m):
    """
    Recursive wrapper to rsolve_hypergeometric.

    Explanation
    ===========

    Returns a Tuple of (formula, series independent terms,
    maximum power of x in independent terms) if successful
    otherwise ``None``.

    See :func:`rsolve_hypergeometric` for details.
    """
    from sympy.polys import lcm, roots
    from sympy.integrals import integrate

    # transformation - c
    proots, qroots = roots(P, k), roots(Q, k)
    all_roots = dict(proots)
    all_roots.update(qroots)
    scale = lcm([r.as_numer_denom()[1] for r, t in all_roots.items()
                 if r.is_rational])
    f, P, Q, m = _transformation_c(f, x, P, Q, k, m, scale)

    # transformation - a
    qroots = roots(Q, k)
    if qroots:
        k_min = Min(*qroots.keys())
    else:
        k_min = S.Zero
    shift = k_min + m
    f, P, Q, m = _transformation_a(f, x, P, Q, k, m, shift)

    l = (x*f).limit(x, 0)
    if not isinstance(l, Limit) and l != 0:  # Ideally should only be l != 0
        return None

    qroots = roots(Q, k)
    if qroots:
        k_max = Max(*qroots.keys())
    else:
        k_max = S.Zero

    ind, mp = S.Zero, -oo
    for i in range(k_max + m + 1):
        r = f.diff(x, i).limit(x, 0) / factorial(i)
        if r.is_finite is False:
            old_f = f
            f, P, Q, m = _transformation_a(f, x, P, Q, k, m, i)
            f, P, Q, m = _transformation_e(f, x, P, Q, k, m)
            sol, ind, mp = _rsolve_hypergeometric(f, x, P, Q, k, m)
            sol = _apply_integrate(sol, x, k)
            sol = _apply_shift(sol, i)
            ind = integrate(ind, x)
            ind += (old_f - ind).limit(x, 0)  # constant of integration
            mp += 1
            return sol, ind, mp
        elif r:
            ind += r*x**(i + shift)
            pow_x = Rational((i + shift), scale)
            if pow_x > mp:
                mp = pow_x  # maximum power of x
    ind = ind.subs(x, x**(1/scale))

    sol = _compute_formula(f, x, P, Q, k, m, k_max)
    sol = _apply_shift(sol, shift)
    sol = _apply_scale(sol, scale)

    return sol, ind, mp


def rsolve_hypergeometric(f, x, P, Q, k, m):
    """
    Solves RE of hypergeometric type.

    Explanation
    ===========

    Attempts to solve RE of the form

    Q(k)*a(k + m) - P(k)*a(k)

    Transformations that preserve Hypergeometric type:

        a. x**n*f(x): b(k + m) = R(k - n)*b(k)
        b. f(A*x): b(k + m) = A**m*R(k)*b(k)
        c. f(x**n): b(k + n*m) = R(k/n)*b(k)
        d. f(x**(1/m)): b(k + 1) = R(k*m)*b(k)
        e. f'(x): b(k + m) = ((k + m + 1)/(k + 1))*R(k + 1)*b(k)

    Some of these transformations have been used to solve the RE.

    Returns
    =======

    formula : Expr
    ind : Expr
        Independent terms.
    order : int

    Examples
    ========

    >>> from sympy import exp, ln, S
    >>> from sympy.series.formal import rsolve_hypergeometric as rh
    >>> from sympy.abc import x, k

    >>> rh(exp(x), x, -S.One, (k + 1), k, 1)
    (Piecewise((1/factorial(k), Eq(Mod(k, 1), 0)), (0, True)), 1, 1)

    >>> rh(ln(1 + x), x, k**2, k*(k + 1), k, 1)
    (Piecewise(((-1)**(k - 1)*factorial(k - 1)/RisingFactorial(2, k - 1),
     Eq(Mod(k, 1), 0)), (0, True)), x, 2)

    References
    ==========

    .. [1] Formal Power Series - Dominik Gruntz, Wolfram Koepf
    .. [2] Power Series in Computer Algebra - Wolfram Koepf
    """
    result = _rsolve_hypergeometric(f, x, P, Q, k, m)

    if result is None:
        return None

    sol_list, ind, mp = result

    sol_dict = defaultdict(lambda: S.Zero)
    for res, cond in sol_list:
        j, mk = cond.as_coeff_Add()
        c = mk.coeff(k)

        if j.is_integer is False:
            res *= x**frac(j)
            j = floor(j)

        res = res.subs(k, (k - j) / c)
        cond = Eq(k % c, j % c)
        sol_dict[cond] += res  # Group together formula for same conditions

    sol = [(res, cond) for cond, res in sol_dict.items()]
    sol.append((S.Zero, True))
    sol = Piecewise(*sol)

    if mp is -oo:
        s = S.Zero
    elif mp.is_integer is False:
        s = ceiling(mp)
    else:
        s = mp + 1

    #  save all the terms of
    #  form 1/x**k in ind
    if s < 0:
        ind += sum(sequence(sol * x**k, (k, s, -1)))
        s = S.Zero

    return (sol, ind, s)


def _solve_hyper_RE(f, x, RE, g, k):
    """See docstring of :func:`rsolve_hypergeometric` for details."""
    terms = Add.make_args(RE)

    if len(terms) == 2:
        gs = list(RE.atoms(Function))
        P, Q = map(RE.coeff, gs)
        m = gs[1].args[0] - gs[0].args[0]
        if m < 0:
            P, Q = Q, P
            m = abs(m)
        return rsolve_hypergeometric(f, x, P, Q, k, m)


def _solve_explike_DE(f, x, DE, g, k):
    """Solves DE with constant coefficients."""
    from sympy.solvers import rsolve

    for t in Add.make_args(DE):
        coeff, d = t.as_independent(g)
        if coeff.free_symbols:
            return

    RE = exp_re(DE, g, k)

    init = {}
    for i in range(len(Add.make_args(RE))):
        if i:
            f = f.diff(x)
        init[g(k).subs(k, i)] = f.limit(x, 0)

    sol = rsolve(RE, g(k), init)

    if sol:
        return (sol / factorial(k), S.Zero, S.Zero)


def _solve_simple(f, x, DE, g, k):
    """Converts DE into RE and solves using :func:`rsolve`."""
    from sympy.solvers import rsolve

    RE = hyper_re(DE, g, k)

    init = {}
    for i in range(len(Add.make_args(RE))):
        if i:
            f = f.diff(x)
        init[g(k).subs(k, i)] = f.limit(x, 0) / factorial(i)

    sol = rsolve(RE, g(k), init)

    if sol:
        return (sol, S.Zero, S.Zero)


def _transform_explike_DE(DE, g, x, order, syms):
    """Converts DE with free parameters into DE with constant coefficients."""
    from sympy.solvers.solveset import linsolve

    eq = []
    highest_coeff = DE.coeff(Derivative(g(x), x, order))
    for i in range(order):
        coeff = DE.coeff(Derivative(g(x), x, i))
        coeff = (coeff / highest_coeff).expand().collect(x)
        eq.extend(Add.make_args(coeff))
    temp = []
    for e in eq:
        if e.has(x):
            break
        elif e.has(Symbol):
            temp.append(e)
    else:
        eq = temp
    if eq:
        sol = dict(zip(syms, (i for s in linsolve(eq, list(syms)) for i in s)))
        if sol:
            DE = DE.subs(sol)
            DE = DE.factor().as_coeff_mul(Derivative)[1][0]
            DE = DE.collect(Derivative(g(x)))
    return DE


def _transform_DE_RE(DE, g, k, order, syms):
    """Converts DE with free parameters into RE of hypergeometric type."""
    from sympy.solvers.solveset import linsolve

    RE = hyper_re(DE, g, k)

    eq = [RE.coeff(g(k + i)) for i in range(1, order)]
    sol = dict(zip(syms, (i for s in linsolve(eq, list(syms)) for i in s)))
    if sol:
        m = Wild('m')
        RE = RE.subs(sol)
        RE = RE.factor().as_numer_denom()[0].collect(g(k + m))
        RE = RE.as_coeff_mul(g)[1][0]
        for i in range(order):  # smallest order should be g(k)
            if RE.coeff(g(k + i)) and i:
                RE = RE.subs(k, k - i)
                break
    return RE


def solve_de(f, x, DE, order, g, k):
    """
    Solves the DE.

    Explanation
    ===========

    Tries to solve DE by either converting into a RE containing two terms or
    converting into a DE having constant coefficients.

    Returns
    =======

    formula : Expr
    ind : Expr
        Independent terms.
    order : int

    Examples
    ========

    >>> from sympy import Derivative as D, Function
    >>> from sympy import exp, ln
    >>> from sympy.series.formal import solve_de
    >>> from sympy.abc import x, k
    >>> f = Function('f')

    >>> solve_de(exp(x), x, D(f(x), x) - f(x), 1, f, k)
    (Piecewise((1/factorial(k), Eq(Mod(k, 1), 0)), (0, True)), 1, 1)

    >>> solve_de(ln(1 + x), x, (x + 1)*D(f(x), x, 2) + D(f(x)), 2, f, k)
    (Piecewise(((-1)**(k - 1)*factorial(k - 1)/RisingFactorial(2, k - 1),
     Eq(Mod(k, 1), 0)), (0, True)), x, 2)
    """
    sol = None
    syms = DE.free_symbols.difference({g, x})

    if syms:
        RE = _transform_DE_RE(DE, g, k, order, syms)
    else:
        RE = hyper_re(DE, g, k)
    if not RE.free_symbols.difference({k}):
        sol = _solve_hyper_RE(f, x, RE, g, k)

    if sol:
        return sol

    if syms:
        DE = _transform_explike_DE(DE, g, x, order, syms)
    if not DE.free_symbols.difference({x}):
        sol = _solve_explike_DE(f, x, DE, g, k)

    if sol:
        return sol


def hyper_algorithm(f, x, k, order=4):
    """
    Hypergeometric algorithm for computing Formal Power Series.

    Explanation
    ===========

    Steps:
        * Generates DE
        * Convert the DE into RE
        * Solves the RE

    Examples
    ========

    >>> from sympy import exp, ln
    >>> from sympy.series.formal import hyper_algorithm

    >>> from sympy.abc import x, k

    >>> hyper_algorithm(exp(x), x, k)
    (Piecewise((1/factorial(k), Eq(Mod(k, 1), 0)), (0, True)), 1, 1)

    >>> hyper_algorithm(ln(1 + x), x, k)
    (Piecewise(((-1)**(k - 1)*factorial(k - 1)/RisingFactorial(2, k - 1),
     Eq(Mod(k, 1), 0)), (0, True)), x, 2)

    See Also
    ========

    sympy.series.formal.simpleDE
    sympy.series.formal.solve_de
    """
    g = Function('g')

    des = []  # list of DE's
    sol = None
    for DE, i in simpleDE(f, x, g, order):
        if DE is not None:
            sol = solve_de(f, x, DE, i, g, k)
        if sol:
            return sol
        if not DE.free_symbols.difference({x}):
            des.append(DE)

    # If nothing works
    # Try plain rsolve
    for DE in des:
        sol = _solve_simple(f, x, DE, g, k)
        if sol:
            return sol


def _compute_fps(f, x, x0, dir, hyper, order, rational, full):
    """Recursive wrapper to compute fps.

    See :func:`compute_fps` for details.
    """
    if x0 in [S.Infinity, S.NegativeInfinity]:
        dir = S.One if x0 is S.Infinity else -S.One
        temp = f.subs(x, 1/x)
        result = _compute_fps(temp, x, 0, dir, hyper, order, rational, full)
        if result is None:
            return None
        return (result[0], result[1].subs(x, 1/x), result[2].subs(x, 1/x))
    elif x0 or dir == -S.One:
        if dir == -S.One:
            rep = -x + x0
            rep2 = -x
            rep2b = x0
        else:
            rep = x + x0
            rep2 = x
            rep2b = -x0
        temp = f.subs(x, rep)
        result = _compute_fps(temp, x, 0, S.One, hyper, order, rational, full)
        if result is None:
            return None
        return (result[0], result[1].subs(x, rep2 + rep2b),
                result[2].subs(x, rep2 + rep2b))

    if f.is_polynomial(x):
        k = Dummy('k')
        ak = sequence(Coeff(f, x, k), (k, 1, oo))
        xk = sequence(x**k, (k, 0, oo))
        ind = f.coeff(x, 0)
        return ak, xk, ind

    #  Break instances of Add
    #  this allows application of different
    #  algorithms on different terms increasing the
    #  range of admissible functions.
    if isinstance(f, Add):
        result = False
        ak = sequence(S.Zero, (0, oo))
        ind, xk = S.Zero, None
        for t in Add.make_args(f):
            res = _compute_fps(t, x, 0, S.One, hyper, order, rational, full)
            if res:
                if not result:
                    result = True
                    xk = res[1]
                if res[0].start > ak.start:
                    seq = ak
                    s, f = ak.start, res[0].start
                else:
                    seq = res[0]
                    s, f = res[0].start, ak.start
                save = Add(*[z[0]*z[1] for z in zip(seq[0:(f - s)], xk[s:f])])
                ak += res[0]
                ind += res[2] + save
            else:
                ind += t
        if result:
            return ak, xk, ind
        return None

    # The symbolic term - symb, if present, is being separated from the function
    # Otherwise symb is being set to S.One
    syms = f.free_symbols.difference({x})
    (f, symb) = expand(f).as_independent(*syms)

    result = None

    # from here on it's x0=0 and dir=1 handling
    k = Dummy('k')
    if rational:
        result = rational_algorithm(f, x, k, order, full)

    if result is None and hyper:
        result = hyper_algorithm(f, x, k, order)

    if result is None:
        return None

    from sympy.simplify.powsimp import powsimp
    if symb.is_zero:
        symb = S.One
    else:
        symb = powsimp(symb)
    ak = sequence(result[0], (k, result[2], oo))
    xk_formula = powsimp(x**k * symb)
    xk = sequence(xk_formula, (k, 0, oo))
    ind = powsimp(result[1] * symb)

    return ak, xk, ind


def compute_fps(f, x, x0=0, dir=1, hyper=True, order=4, rational=True,
                full=False):
    """
    Computes the formula for Formal Power Series of a function.

    Explanation
    ===========

    Tries to compute the formula by applying the following techniques
    (in order):

    * rational_algorithm
    * Hypergeometric algorithm

    Parameters
    ==========

    x : Symbol
    x0 : number, optional
        Point to perform series expansion about. Default is 0.
    dir : {1, -1, '+', '-'}, optional
        If dir is 1 or '+' the series is calculated from the right and
        for -1 or '-' the series is calculated from the left. For smooth
        functions this flag will not alter the results. Default is 1.
    hyper : {True, False}, optional
        Set hyper to False to skip the hypergeometric algorithm.
        By default it is set to False.
    order : int, optional
        Order of the derivative of ``f``, Default is 4.
    rational : {True, False}, optional
        Set rational to False to skip rational algorithm. By default it is set
        to True.
    full : {True, False}, optional
        Set full to True to increase the range of rational algorithm.
        See :func:`rational_algorithm` for details. By default it is set to
        False.

    Returns
    =======

    ak : sequence
        Sequence of coefficients.
    xk : sequence
        Sequence of powers of x.
    ind : Expr
        Independent terms.
    mul : Pow
        Common terms.

    See Also
    ========

    sympy.series.formal.rational_algorithm
    sympy.series.formal.hyper_algorithm
    """
    f = sympify(f)
    x = sympify(x)

    if not f.has(x):
        return None

    x0 = sympify(x0)

    if dir == '+':
        dir = S.One
    elif dir == '-':
        dir = -S.One
    elif dir not in [S.One, -S.One]:
        raise ValueError("Dir must be '+' or '-'")
    else:
        dir = sympify(dir)

    return _compute_fps(f, x, x0, dir, hyper, order, rational, full)


class Coeff(Function):
    """
    Coeff(p, x, n) represents the nth coefficient of the polynomial p in x
    """
    @classmethod
    def eval(cls, p, x, n):
        if p.is_polynomial(x) and n.is_integer:
            return p.coeff(x, n)


class FormalPowerSeries(SeriesBase):
    """
    Represents Formal Power Series of a function.

    Explanation
    ===========

    No computation is performed. This class should only to be used to represent
    a series. No checks are performed.

    For computing a series use :func:`fps`.

    See Also
    ========

    sympy.series.formal.fps
    """
    def __new__(cls, *args):
        args = map(sympify, args)
        return Expr.__new__(cls, *args)

    def __init__(self, *args):
        ak = args[4][0]
        k = ak.variables[0]
        self.ak_seq = sequence(ak.formula, (k, 1, oo))
        self.fact_seq = sequence(factorial(k), (k, 1, oo))
        self.bell_coeff_seq = self.ak_seq * self.fact_seq
        self.sign_seq = sequence((-1, 1), (k, 1, oo))

    @property
    def function(self):
        return self.args[0]

    @property
    def x(self):
        return self.args[1]

    @property
    def x0(self):
        return self.args[2]

    @property
    def dir(self):
        return self.args[3]

    @property
    def ak(self):
        return self.args[4][0]

    @property
    def xk(self):
        return self.args[4][1]

    @property
    def ind(self):
        return self.args[4][2]

    @property
    def interval(self):
        return Interval(0, oo)

    @property
    def start(self):
        return self.interval.inf

    @property
    def stop(self):
        return self.interval.sup

    @property
    def length(self):
        return oo

    @property
    def infinite(self):
        """Returns an infinite representation of the series"""
        from sympy.concrete import Sum
        ak, xk = self.ak, self.xk
        k = ak.variables[0]
        inf_sum = Sum(ak.formula * xk.formula, (k, ak.start, ak.stop))

        return self.ind + inf_sum

    def _get_pow_x(self, term):
        """Returns the power of x in a term."""
        xterm, pow_x = term.as_independent(self.x)[1].as_base_exp()
        if not xterm.has(self.x):
            return S.Zero
        return pow_x

    def polynomial(self, n=6):
        """
        Truncated series as polynomial.

        Explanation
        ===========

        Returns series expansion of ``f`` upto order ``O(x**n)``
        as a polynomial(without ``O`` term).
        """
        terms = []
        sym = self.free_symbols
        for i, t in enumerate(self):
            xp = self._get_pow_x(t)
            if xp.has(*sym):
                xp = xp.as_coeff_add(*sym)[0]
            if xp >= n:
                break
            elif xp.is_integer is True and i == n + 1:
                break
            elif t is not S.Zero:
                terms.append(t)

        return Add(*terms)

    def truncate(self, n=6):
        """
        Truncated series.

        Explanation
        ===========

        Returns truncated series expansion of f upto
        order ``O(x**n)``.

        If n is ``None``, returns an infinite iterator.
        """
        if n is None:
            return iter(self)

        x, x0 = self.x, self.x0
        pt_xk = self.xk.coeff(n)
        if x0 is S.NegativeInfinity:
            x0 = S.Infinity

        return self.polynomial(n) + Order(pt_xk, (x, x0))

    def zero_coeff(self):
        return self._eval_term(0)

    def _eval_term(self, pt):
        try:
            pt_xk = self.xk.coeff(pt)
            pt_ak = self.ak.coeff(pt).simplify()  # Simplify the coefficients
        except IndexError:
            term = S.Zero
        else:
            term = (pt_ak * pt_xk)

        if self.ind:
            ind = S.Zero
            sym = self.free_symbols
            for t in Add.make_args(self.ind):
                pow_x = self._get_pow_x(t)
                if pow_x.has(*sym):
                    pow_x = pow_x.as_coeff_add(*sym)[0]
                if pt == 0 and pow_x < 1:
                    ind += t
                elif pow_x >= pt and pow_x < pt + 1:
                    ind += t
            term += ind

        return term.collect(self.x)

    def _eval_subs(self, old, new):
        x = self.x
        if old.has(x):
            return self

    def _eval_as_leading_term(self, x, logx, cdir):
        for t in self:
            if t is not S.Zero:
                return t

    def _eval_derivative(self, x):
        f = self.function.diff(x)
        ind = self.ind.diff(x)

        pow_xk = self._get_pow_x(self.xk.formula)
        ak = self.ak
        k = ak.variables[0]
        if ak.formula.has(x):
            form = []
            for e, c in ak.formula.args:
                temp = S.Zero
                for t in Add.make_args(e):
                    pow_x = self._get_pow_x(t)
                    temp += t * (pow_xk + pow_x)
                form.append((temp, c))
            form = Piecewise(*form)
            ak = sequence(form.subs(k, k + 1), (k, ak.start - 1, ak.stop))
        else:
            ak = sequence((ak.formula * pow_xk).subs(k, k + 1),
                          (k, ak.start - 1, ak.stop))

        return self.func(f, self.x, self.x0, self.dir, (ak, self.xk, ind))

    def integrate(self, x=None, **kwargs):
        """
        Integrate Formal Power Series.

        Examples
        ========

        >>> from sympy import fps, sin, integrate
        >>> from sympy.abc import x
        >>> f = fps(sin(x))
        >>> f.integrate(x).truncate()
        -1 + x**2/2 - x**4/24 + O(x**6)
        >>> integrate(f, (x, 0, 1))
        1 - cos(1)
        """
        from sympy.integrals import integrate

        if x is None:
            x = self.x
        elif iterable(x):
            return integrate(self.function, x)

        f = integrate(self.function, x)
        ind = integrate(self.ind, x)
        ind += (f - ind).limit(x, 0)  # constant of integration

        pow_xk = self._get_pow_x(self.xk.formula)
        ak = self.ak
        k = ak.variables[0]
        if ak.formula.has(x):
            form = []
            for e, c in ak.formula.args:
                temp = S.Zero
                for t in Add.make_args(e):
                    pow_x = self._get_pow_x(t)
                    temp += t / (pow_xk + pow_x + 1)
                form.append((temp, c))
            form = Piecewise(*form)
            ak = sequence(form.subs(k, k - 1), (k, ak.start + 1, ak.stop))
        else:
            ak = sequence((ak.formula / (pow_xk + 1)).subs(k, k - 1),
                          (k, ak.start + 1, ak.stop))

        return self.func(f, self.x, self.x0, self.dir, (ak, self.xk, ind))

    def product(self, other, x=None, n=6):
        """
        Multiplies two Formal Power Series, using discrete convolution and
        return the truncated terms upto specified order.

        Parameters
        ==========

        n : Number, optional
            Specifies the order of the term up to which the polynomial should
            be truncated.

        Examples
        ========

        >>> from sympy import fps, sin, exp
        >>> from sympy.abc import x
        >>> f1 = fps(sin(x))
        >>> f2 = fps(exp(x))

        >>> f1.product(f2, x).truncate(4)
        x + x**2 + x**3/3 + O(x**4)

        See Also
        ========

        sympy.discrete.convolutions
        sympy.series.formal.FormalPowerSeriesProduct

        """

        if n is None:
            return iter(self)

        other = sympify(other)

        if not isinstance(other, FormalPowerSeries):
            raise ValueError("Both series should be an instance of FormalPowerSeries"
                             " class.")

        if self.dir != other.dir:
            raise ValueError("Both series should be calculated from the"
                             " same direction.")
        elif self.x0 != other.x0:
            raise ValueError("Both series should be calculated about the"
                             " same point.")

        elif self.x != other.x:
            raise ValueError("Both series should have the same symbol.")

        return FormalPowerSeriesProduct(self, other)

    def coeff_bell(self, n):
        r"""
        self.coeff_bell(n) returns a sequence of Bell polynomials of the second kind.
        Note that ``n`` should be a integer.

        The second kind of Bell polynomials (are sometimes called "partial" Bell
        polynomials or incomplete Bell polynomials) are defined as

        .. math::
            B_{n,k}(x_1, x_2,\dotsc x_{n-k+1}) =
                \sum_{j_1+j_2+j_2+\dotsb=k \atop j_1+2j_2+3j_2+\dotsb=n}
                \frac{n!}{j_1!j_2!\dotsb j_{n-k+1}!}
                \left(\frac{x_1}{1!} \right)^{j_1}
                \left(\frac{x_2}{2!} \right)^{j_2} \dotsb
                \left(\frac{x_{n-k+1}}{(n-k+1)!} \right) ^{j_{n-k+1}}.

        * ``bell(n, k, (x1, x2, ...))`` gives Bell polynomials of the second kind,
          `B_{n,k}(x_1, x_2, \dotsc, x_{n-k+1})`.

        See Also
        ========

        sympy.functions.combinatorial.numbers.bell

        """

        inner_coeffs = [bell(n, j, tuple(self.bell_coeff_seq[:n-j+1])) for j in range(1, n+1)]

        k = Dummy('k')
        return sequence(tuple(inner_coeffs), (k, 1, oo))

    def compose(self, other, x=None, n=6):
        r"""
        Returns the truncated terms of the formal power series of the composed function,
        up to specified ``n``.

        Explanation
        ===========

        If ``f`` and ``g`` are two formal power series of two different functions,
        then the coefficient sequence ``ak`` of the composed formal power series `fp`
        will be as follows.

        .. math::
            \sum\limits_{k=0}^{n} b_k B_{n,k}(x_1, x_2, \dotsc, x_{n-k+1})

        Parameters
        ==========

        n : Number, optional
            Specifies the order of the term up to which the polynomial should
            be truncated.

        Examples
        ========

        >>> from sympy import fps, sin, exp
        >>> from sympy.abc import x
        >>> f1 = fps(exp(x))
        >>> f2 = fps(sin(x))

        >>> f1.compose(f2, x).truncate()
        1 + x + x**2/2 - x**4/8 - x**5/15 + O(x**6)

        >>> f1.compose(f2, x).truncate(8)
        1 + x + x**2/2 - x**4/8 - x**5/15 - x**6/240 + x**7/90 + O(x**8)

        See Also
        ========

        sympy.functions.combinatorial.numbers.bell
        sympy.series.formal.FormalPowerSeriesCompose

        References
        ==========

        .. [1] Comtet, Louis: Advanced combinatorics; the art of finite and infinite expansions. Reidel, 1974.

        """

        if n is None:
            return iter(self)

        other = sympify(other)

        if not isinstance(other, FormalPowerSeries):
            raise ValueError("Both series should be an instance of FormalPowerSeries"
                             " class.")

        if self.dir != other.dir:
            raise ValueError("Both series should be calculated from the"
                             " same direction.")
        elif self.x0 != other.x0:
            raise ValueError("Both series should be calculated about the"
                             " same point.")

        elif self.x != other.x:
            raise ValueError("Both series should have the same symbol.")

        if other._eval_term(0).as_coeff_mul(other.x)[0] is not S.Zero:
            raise ValueError("The formal power series of the inner function should not have any "
                "constant coefficient term.")

        return FormalPowerSeriesCompose(self, other)

    def inverse(self, x=None, n=6):
        r"""
        Returns the truncated terms of the inverse of the formal power series,
        up to specified ``n``.

        Explanation
        ===========

        If ``f`` and ``g`` are two formal power series of two different functions,
        then the coefficient sequence ``ak`` of the composed formal power series ``fp``
        will be as follows.

        .. math::
            \sum\limits_{k=0}^{n} (-1)^{k} x_0^{-k-1} B_{n,k}(x_1, x_2, \dotsc, x_{n-k+1})

        Parameters
        ==========

        n : Number, optional
            Specifies the order of the term up to which the polynomial should
            be truncated.

        Examples
        ========

        >>> from sympy import fps, exp, cos
        >>> from sympy.abc import x
        >>> f1 = fps(exp(x))
        >>> f2 = fps(cos(x))

        >>> f1.inverse(x).truncate()
        1 - x + x**2/2 - x**3/6 + x**4/24 - x**5/120 + O(x**6)

        >>> f2.inverse(x).truncate(8)
        1 + x**2/2 + 5*x**4/24 + 61*x**6/720 + O(x**8)

        See Also
        ========

        sympy.functions.combinatorial.numbers.bell
        sympy.series.formal.FormalPowerSeriesInverse

        References
        ==========

        .. [1] Comtet, Louis: Advanced combinatorics; the art of finite and infinite expansions. Reidel, 1974.

        """

        if n is None:
            return iter(self)

        if self._eval_term(0).is_zero:
            raise ValueError("Constant coefficient should exist for an inverse of a formal"
                " power series to exist.")

        return FormalPowerSeriesInverse(self)

    def __add__(self, other):
        other = sympify(other)

        if isinstance(other, FormalPowerSeries):
            if self.dir != other.dir:
                raise ValueError("Both series should be calculated from the"
                                 " same direction.")
            elif self.x0 != other.x0:
                raise ValueError("Both series should be calculated about the"
                                 " same point.")

            x, y = self.x, other.x
            f = self.function + other.function.subs(y, x)

            if self.x not in f.free_symbols:
                return f

            ak = self.ak + other.ak
            if self.ak.start > other.ak.start:
                seq = other.ak
                s, e = other.ak.start, self.ak.start
            else:
                seq = self.ak
                s, e = self.ak.start, other.ak.start
            save = Add(*[z[0]*z[1] for z in zip(seq[0:(e - s)], self.xk[s:e])])
            ind = self.ind + other.ind + save

            return self.func(f, x, self.x0, self.dir, (ak, self.xk, ind))

        elif not other.has(self.x):
            f = self.function + other
            ind = self.ind + other

            return self.func(f, self.x, self.x0, self.dir,
                             (self.ak, self.xk, ind))

        return Add(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return self.func(-self.function, self.x, self.x0, self.dir,
                         (-self.ak, self.xk, -self.ind))

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __mul__(self, other):
        other = sympify(other)

        if other.has(self.x):
            return Mul(self, other)

        f = self.function * other
        ak = self.ak.coeff_mul(other)
        ind = self.ind * other

        return self.func(f, self.x, self.x0, self.dir, (ak, self.xk, ind))

    def __rmul__(self, other):
        return self.__mul__(other)


class FiniteFormalPowerSeries(FormalPowerSeries):
    """Base Class for Product, Compose and Inverse classes"""

    def __init__(self, *args):
        pass

    @property
    def ffps(self):
        return self.args[0]

    @property
    def gfps(self):
        return self.args[1]

    @property
    def f(self):
        return self.ffps.function

    @property
    def g(self):
        return self.gfps.function

    @property
    def infinite(self):
        raise NotImplementedError("No infinite version for an object of"
                     " FiniteFormalPowerSeries class.")

    def _eval_terms(self, n):
        raise NotImplementedError("(%s)._eval_terms()" % self)

    def _eval_term(self, pt):
        raise NotImplementedError("By the current logic, one can get terms"
                                   "upto a certain order, instead of getting term by term.")

    def polynomial(self, n):
        return self._eval_terms(n)

    def truncate(self, n=6):
        ffps = self.ffps
        pt_xk = ffps.xk.coeff(n)
        x, x0 = ffps.x, ffps.x0

        return self.polynomial(n) + Order(pt_xk, (x, x0))

    def _eval_derivative(self, x):
        raise NotImplementedError

    def integrate(self, x):
        raise NotImplementedError


class FormalPowerSeriesProduct(FiniteFormalPowerSeries):
    """Represents the product of two formal power series of two functions.

    Explanation
    ===========

    No computation is performed. Terms are calculated using a term by term logic,
    instead of a point by point logic.

    There are two differences between a :obj:`FormalPowerSeries` object and a
    :obj:`FormalPowerSeriesProduct` object. The first argument contains the two
    functions involved in the product. Also, the coefficient sequence contains
    both the coefficient sequence of the formal power series of the involved functions.

    See Also
    ========

    sympy.series.formal.FormalPowerSeries
    sympy.series.formal.FiniteFormalPowerSeries

    """

    def __init__(self, *args):
        ffps, gfps = self.ffps, self.gfps

        k = ffps.ak.variables[0]
        self.coeff1 = sequence(ffps.ak.formula, (k, 0, oo))

        k = gfps.ak.variables[0]
        self.coeff2 = sequence(gfps.ak.formula, (k, 0, oo))

    @property
    def function(self):
        """Function of the product of two formal power series."""
        return self.f * self.g

    def _eval_terms(self, n):
        """
        Returns the first ``n`` terms of the product formal power series.
        Term by term logic is implemented here.

        Examples
        ========

        >>> from sympy import fps, sin, exp
        >>> from sympy.abc import x
        >>> f1 = fps(sin(x))
        >>> f2 = fps(exp(x))
        >>> fprod = f1.product(f2, x)

        >>> fprod._eval_terms(4)
        x**3/3 + x**2 + x

        See Also
        ========

        sympy.series.formal.FormalPowerSeries.product

        """
        coeff1, coeff2 = self.coeff1, self.coeff2

        aks = convolution(coeff1[:n], coeff2[:n])

        terms = []
        for i in range(0, n):
            terms.append(aks[i] * self.ffps.xk.coeff(i))

        return Add(*terms)


class FormalPowerSeriesCompose(FiniteFormalPowerSeries):
    """
    Represents the composed formal power series of two functions.

    Explanation
    ===========

    No computation is performed. Terms are calculated using a term by term logic,
    instead of a point by point logic.

    There are two differences between a :obj:`FormalPowerSeries` object and a
    :obj:`FormalPowerSeriesCompose` object. The first argument contains the outer
    function and the inner function involved in the omposition. Also, the
    coefficient sequence contains the generic sequence which is to be multiplied
    by a custom ``bell_seq`` finite sequence. The finite terms will then be added up to
    get the final terms.

    See Also
    ========

    sympy.series.formal.FormalPowerSeries
    sympy.series.formal.FiniteFormalPowerSeries

    """

    @property
    def function(self):
        """Function for the composed formal power series."""
        f, g, x = self.f, self.g, self.ffps.x
        return f.subs(x, g)

    def _eval_terms(self, n):
        """
        Returns the first `n` terms of the composed formal power series.
        Term by term logic is implemented here.

        Explanation
        ===========

        The coefficient sequence of the :obj:`FormalPowerSeriesCompose` object is the generic sequence.
        It is multiplied by ``bell_seq`` to get a sequence, whose terms are added up to get
        the final terms for the polynomial.

        Examples
        ========

        >>> from sympy import fps, sin, exp
        >>> from sympy.abc import x
        >>> f1 = fps(exp(x))
        >>> f2 = fps(sin(x))
        >>> fcomp = f1.compose(f2, x)

        >>> fcomp._eval_terms(6)
        -x**5/15 - x**4/8 + x**2/2 + x + 1

        >>> fcomp._eval_terms(8)
        x**7/90 - x**6/240 - x**5/15 - x**4/8 + x**2/2 + x + 1

        See Also
        ========

        sympy.series.formal.FormalPowerSeries.compose
        sympy.series.formal.FormalPowerSeries.coeff_bell

        """

        ffps, gfps = self.ffps, self.gfps
        terms = [ffps.zero_coeff()]

        for i in range(1, n):
            bell_seq = gfps.coeff_bell(i)
            seq = (ffps.bell_coeff_seq * bell_seq)
            terms.append(Add(*(seq[:i])) / ffps.fact_seq[i-1] * ffps.xk.coeff(i))

        return Add(*terms)


class FormalPowerSeriesInverse(FiniteFormalPowerSeries):
    """
    Represents the Inverse of a formal power series.

    Explanation
    ===========

    No computation is performed. Terms are calculated using a term by term logic,
    instead of a point by point logic.

    There is a single difference between a :obj:`FormalPowerSeries` object and a
    :obj:`FormalPowerSeriesInverse` object. The coefficient sequence contains the
    generic sequence which is to be multiplied by a custom ``bell_seq`` finite sequence.
    The finite terms will then be added up to get the final terms.

    See Also
    ========

    sympy.series.formal.FormalPowerSeries
    sympy.series.formal.FiniteFormalPowerSeries

    """
    def __init__(self, *args):
        ffps = self.ffps
        k = ffps.xk.variables[0]

        inv = ffps.zero_coeff()
        inv_seq = sequence(inv ** (-(k + 1)), (k, 1, oo))
        self.aux_seq = ffps.sign_seq * ffps.fact_seq * inv_seq

    @property
    def function(self):
        """Function for the inverse of a formal power series."""
        f = self.f
        return 1 / f

    @property
    def g(self):
        raise ValueError("Only one function is considered while performing"
                        "inverse of a formal power series.")

    @property
    def gfps(self):
        raise ValueError("Only one function is considered while performing"
                        "inverse of a formal power series.")

    def _eval_terms(self, n):
        """
        Returns the first ``n`` terms of the composed formal power series.
        Term by term logic is implemented here.

        Explanation
        ===========

        The coefficient sequence of the `FormalPowerSeriesInverse` object is the generic sequence.
        It is multiplied by ``bell_seq`` to get a sequence, whose terms are added up to get
        the final terms for the polynomial.

        Examples
        ========

        >>> from sympy import fps, exp, cos
        >>> from sympy.abc import x
        >>> f1 = fps(exp(x))
        >>> f2 = fps(cos(x))
        >>> finv1, finv2 = f1.inverse(), f2.inverse()

        >>> finv1._eval_terms(6)
        -x**5/120 + x**4/24 - x**3/6 + x**2/2 - x + 1

        >>> finv2._eval_terms(8)
        61*x**6/720 + 5*x**4/24 + x**2/2 + 1

        See Also
        ========

        sympy.series.formal.FormalPowerSeries.inverse
        sympy.series.formal.FormalPowerSeries.coeff_bell

        """
        ffps = self.ffps
        terms = [ffps.zero_coeff()]

        for i in range(1, n):
            bell_seq = ffps.coeff_bell(i)
            seq = (self.aux_seq * bell_seq)
            terms.append(Add(*(seq[:i])) / ffps.fact_seq[i-1] * ffps.xk.coeff(i))

        return Add(*terms)


def fps(f, x=None, x0=0, dir=1, hyper=True, order=4, rational=True, full=False):
    """
    Generates Formal Power Series of ``f``.

    Explanation
    ===========

    Returns the formal series expansion of ``f`` around ``x = x0``
    with respect to ``x`` in the form of a ``FormalPowerSeries`` object.

    Formal Power Series is represented using an explicit formula
    computed using different algorithms.

    See :func:`compute_fps` for the more details regarding the computation
    of formula.

    Parameters
    ==========

    x : Symbol, optional
        If x is None and ``f`` is univariate, the univariate symbols will be
        supplied, otherwise an error will be raised.
    x0 : number, optional
        Point to perform series expansion about. Default is 0.
    dir : {1, -1, '+', '-'}, optional
        If dir is 1 or '+' the series is calculated from the right and
        for -1 or '-' the series is calculated from the left. For smooth
        functions this flag will not alter the results. Default is 1.
    hyper : {True, False}, optional
        Set hyper to False to skip the hypergeometric algorithm.
        By default it is set to False.
    order : int, optional
        Order of the derivative of ``f``, Default is 4.
    rational : {True, False}, optional
        Set rational to False to skip rational algorithm. By default it is set
        to True.
    full : {True, False}, optional
        Set full to True to increase the range of rational algorithm.
        See :func:`rational_algorithm` for details. By default it is set to
        False.

    Examples
    ========

    >>> from sympy import fps, ln, atan, sin
    >>> from sympy.abc import x, n

    Rational Functions

    >>> fps(ln(1 + x)).truncate()
    x - x**2/2 + x**3/3 - x**4/4 + x**5/5 + O(x**6)

    >>> fps(atan(x), full=True).truncate()
    x - x**3/3 + x**5/5 + O(x**6)

    Symbolic Functions

    >>> fps(x**n*sin(x**2), x).truncate(8)
    -x**(n + 6)/6 + x**(n + 2) + O(x**(n + 8))

    See Also
    ========

    sympy.series.formal.FormalPowerSeries
    sympy.series.formal.compute_fps
    """
    f = sympify(f)

    if x is None:
        free = f.free_symbols
        if len(free) == 1:
            x = free.pop()
        elif not free:
            return f
        else:
            raise NotImplementedError("multivariate formal power series")

    result = compute_fps(f, x, x0, dir, hyper, order, rational, full)

    if result is None:
        return f

    return FormalPowerSeries(f, x, x0, dir, result)
<|endoftext|>"""Limits of sequences"""

from sympy.calculus.accumulationbounds import AccumulationBounds
from sympy.core.add import Add
from sympy.core.function import PoleError
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.numbers import fibonacci
from sympy.functions.combinatorial.factorials import factorial, subfactorial
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.series.limits import Limit


def difference_delta(expr, n=None, step=1):
    """Difference Operator.

    Explanation
    ===========

    Discrete analog of differential operator. Given a sequence x[n],
    returns the sequence x[n + step] - x[n].

    Examples
    ========

    >>> from sympy import difference_delta as dd
    >>> from sympy.abc import n
    >>> dd(n*(n + 1), n)
    2*n + 2
    >>> dd(n*(n + 1), n, 2)
    4*n + 6

    References
    ==========

    .. [1] https://reference.wolfram.com/language/ref/DifferenceDelta.html
    """
    expr = sympify(expr)

    if n is None:
        f = expr.free_symbols
        if len(f) == 1:
            n = f.pop()
        elif len(f) == 0:
            return S.Zero
        else:
            raise ValueError("Since there is more than one variable in the"
                             " expression, a variable must be supplied to"
                             " take the difference of %s" % expr)
    step = sympify(step)
    if step.is_number is False or step.is_finite is False:
        raise ValueError("Step should be a finite number.")

    if hasattr(expr, '_eval_difference_delta'):
        result = expr._eval_difference_delta(n, step)
        if result:
            return result

    return expr.subs(n, n + step) - expr


def dominant(expr, n):
    """Finds the dominant term in a sum, that is a term that dominates
    every other term.

    Explanation
    ===========

    If limit(a/b, n, oo) is oo then a dominates b.
    If limit(a/b, n, oo) is 0 then b dominates a.
    Otherwise, a and b are comparable.

    If there is no unique dominant term, then returns ``None``.

    Examples
    ========

    >>> from sympy import Sum
    >>> from sympy.series.limitseq import dominant
    >>> from sympy.abc import n, k
    >>> dominant(5*n**3 + 4*n**2 + n + 1, n)
    5*n**3
    >>> dominant(2**n + Sum(k, (k, 0, n)), n)
    2**n

    See Also
    ========

    sympy.series.limitseq.dominant
    """
    terms = Add.make_args(expr.expand(func=True))
    term0 = terms[-1]
    comp = [term0]  # comparable terms
    for t in terms[:-1]:
        r = term0/t
        e = r.gammasimp()
        if e == r:
            e = r.factor()
        l = limit_seq(e, n)
        if l is None:
            return None
        elif l.is_zero:
            term0 = t
            comp = [term0]
        elif l not in [S.Infinity, S.NegativeInfinity]:
            comp.append(t)
    if len(comp) > 1:
        return None
    return term0


def _limit_inf(expr, n):
    try:
        return Limit(expr, n, S.Infinity).doit(deep=False)
    except (NotImplementedError, PoleError):
        return None


def _limit_seq(expr, n, trials):
    from sympy.concrete.summations import Sum

    for i in range(trials):
        if not expr.has(Sum):
            result = _limit_inf(expr, n)
            if result is not None:
                return result

        num, den = expr.as_numer_denom()
        if not den.has(n) or not num.has(n):
            result = _limit_inf(expr.doit(), n)
            if result is not None:
                return result
            return None

        num, den = (difference_delta(t.expand(), n) for t in [num, den])
        expr = (num / den).gammasimp()

        if not expr.has(Sum):
            result = _limit_inf(expr, n)
            if result is not None:
                return result

        num, den = expr.as_numer_denom()

        num = dominant(num, n)
        if num is None:
            return None

        den = dominant(den, n)
        if den is None:
            return None

        expr = (num / den).gammasimp()


def limit_seq(expr, n=None, trials=5):
    """Finds the limit of a sequence as index ``n`` tends to infinity.

    Parameters
    ==========

    expr : Expr
        SymPy expression for the ``n-th`` term of the sequence
    n : Symbol, optional
        The index of the sequence, an integer that tends to positive
        infinity. If None, inferred from the expression unless it has
        multiple symbols.
    trials: int, optional
        The algorithm is highly recursive. ``trials`` is a safeguard from
        infinite recursion in case the limit is not easily computed by the
        algorithm. Try increasing ``trials`` if the algorithm returns ``None``.

    Admissible Terms
    ================

    The algorithm is designed for sequences built from rational functions,
    indefinite sums, and indefinite products over an indeterminate n. Terms of
    alternating sign are also allowed, but more complex oscillatory behavior is
    not supported.

    Examples
    ========

    >>> from sympy import limit_seq, Sum, binomial
    >>> from sympy.abc import n, k, m
    >>> limit_seq((5*n**3 + 3*n**2 + 4) / (3*n**3 + 4*n - 5), n)
    5/3
    >>> limit_seq(binomial(2*n, n) / Sum(binomial(2*k, k), (k, 1, n)), n)
    3/4
    >>> limit_seq(Sum(k**2 * Sum(2**m/m, (m, 1, k)), (k, 1, n)) / (2**n*n), n)
    4

    See Also
    ========

    sympy.series.limitseq.dominant

    References
    ==========

    .. [1] Computing Limits of Sequences - Manuel Kauers
    """

    from sympy.concrete.summations import Sum
    if n is None:
        free = expr.free_symbols
        if len(free) == 1:
            n = free.pop()
        elif not free:
            return expr
        else:
            raise ValueError("Expression has more than one variable. "
                             "Please specify a variable.")
    elif n not in expr.free_symbols:
        return expr

    expr = expr.rewrite(fibonacci, S.GoldenRatio)
    expr = expr.rewrite(factorial, subfactorial, gamma)
    n_ = Dummy("n", integer=True, positive=True)
    n1 = Dummy("n", odd=True, positive=True)
    n2 = Dummy("n", even=True, positive=True)

    # If there is a negative term raised to a power involving n, or a
    # trigonometric function, then consider even and odd n separately.
    powers = (p.as_base_exp() for p in expr.atoms(Pow))
    if (any(b.is_negative and e.has(n) for b, e in powers) or
            expr.has(cos, sin)):
        L1 = _limit_seq(expr.xreplace({n: n1}), n1, trials)
        if L1 is not None:
            L2 = _limit_seq(expr.xreplace({n: n2}), n2, trials)
            if L1 != L2:
                if L1.is_comparable and L2.is_comparable:
                    return AccumulationBounds(Min(L1, L2), Max(L1, L2))
                else:
                    return None
    else:
        L1 = _limit_seq(expr.xreplace({n: n_}), n_, trials)
    if L1 is not None:
        return L1
    else:
        if expr.is_Add:
            limits = [limit_seq(term, n, trials) for term in expr.args]
            if any(result is None for result in limits):
                return None
            else:
                return Add(*limits)
        # Maybe the absolute value is easier to deal with (though not if
        # it has a Sum). If it tends to 0, the limit is 0.
        elif not expr.has(Sum):
            lim = _limit_seq(Abs(expr.xreplace({n: n_})), n_, trials)
            if lim is not None and lim.is_zero:
                return S.Zero
<|endoftext|>from sympy.core.sympify import sympify


def series(expr, x=None, x0=0, n=6, dir="+"):
    """Series expansion of expr around point `x = x0`.

    Parameters
    ==========

    expr : Expression
           The expression whose series is to be expanded.

    x : Symbol
        It is the variable of the expression to be calculated.

    x0 : Value
         The value around which ``x`` is calculated. Can be any value
         from ``-oo`` to ``oo``.

    n : Value
        The number of terms upto which the series is to be expanded.

    dir : String, optional
          The series-expansion can be bi-directional. If ``dir="+"``,
          then (x->x0+). If ``dir="-"``, then (x->x0-). For infinite
          ``x0`` (``oo`` or ``-oo``), the ``dir`` argument is determined
          from the direction of the infinity (i.e., ``dir="-"`` for
          ``oo``).

    Examples
    ========

    >>> from sympy import series, tan, oo
    >>> from sympy.abc import x
    >>> f = tan(x)
    >>> series(f, x, 2, 6, "+")
    tan(2) + (1 + tan(2)**2)*(x - 2) + (x - 2)**2*(tan(2)**3 + tan(2)) +
    (x - 2)**3*(1/3 + 4*tan(2)**2/3 + tan(2)**4) + (x - 2)**4*(tan(2)**5 +
    5*tan(2)**3/3 + 2*tan(2)/3) + (x - 2)**5*(2/15 + 17*tan(2)**2/15 +
    2*tan(2)**4 + tan(2)**6) + O((x - 2)**6, (x, 2))

    >>> series(f, x, 2, 3, "-")
    tan(2) + (2 - x)*(-tan(2)**2 - 1) + (2 - x)**2*(tan(2)**3 + tan(2))
    + O((x - 2)**3, (x, 2))

    >>> series(f, x, 2, oo, "+")
    Traceback (most recent call last):
    ...
    TypeError: 'Infinity' object cannot be interpreted as an integer

    Returns
    =======

    Expr
        Series expansion of the expression about x0

    See Also
    ========

    sympy.core.expr.Expr.series: See the docstring of Expr.series() for complete details of this wrapper.
    """
    expr = sympify(expr)
    return expr.series(x, x0, n, dir)
<|endoftext|>"""A module that handles series: find a limit, order the series etc.
"""
from .order import Order
from .limits import limit, Limit
from .gruntz import gruntz
from .series import series
from .approximants import approximants
from .residues import residue
from .sequences import SeqPer, SeqFormula, sequence, SeqAdd, SeqMul
from .fourier import fourier_series
from .formal import fps
from .limitseq import difference_delta, limit_seq

from sympy.core.singleton import S
EmptySequence = S.EmptySequence

O = Order

__all__ = ['Order', 'O', 'limit', 'Limit', 'gruntz', 'series', 'approximants',
        'residue', 'EmptySequence', 'SeqPer', 'SeqFormula', 'sequence',
        'SeqAdd', 'SeqMul', 'fourier_series', 'fps', 'difference_delta',
        'limit_seq'
        ]
<|endoftext|>"""
Limits
======

Implemented according to the PhD thesis
https://www.cybertester.com/data/gruntz.pdf, which contains very thorough
descriptions of the algorithm including many examples.  We summarize here
the gist of it.

All functions are sorted according to how rapidly varying they are at
infinity using the following rules. Any two functions f and g can be
compared using the properties of L:

L=lim  log|f(x)| / log|g(x)|           (for x -> oo)

We define >, < ~ according to::

    1. f > g .... L=+-oo

        we say that:
        - f is greater than any power of g
        - f is more rapidly varying than g
        - f goes to infinity/zero faster than g

    2. f < g .... L=0

        we say that:
        - f is lower than any power of g

    3. f ~ g .... L!=0, +-oo

        we say that:
        - both f and g are bounded from above and below by suitable integral
          powers of the other

Examples
========
::
    2 < x < exp(x) < exp(x**2) < exp(exp(x))
    2 ~ 3 ~ -5
    x ~ x**2 ~ x**3 ~ 1/x ~ x**m ~ -x
    exp(x) ~ exp(-x) ~ exp(2x) ~ exp(x)**2 ~ exp(x+exp(-x))
    f ~ 1/f

So we can divide all the functions into comparability classes (x and x^2
belong to one class, exp(x) and exp(-x) belong to some other class). In
principle, we could compare any two functions, but in our algorithm, we
do not compare anything below the class 2~3~-5 (for example log(x) is
below this), so we set 2~3~-5 as the lowest comparability class.

Given the function f, we find the list of most rapidly varying (mrv set)
subexpressions of it. This list belongs to the same comparability class.
Let's say it is {exp(x), exp(2x)}. Using the rule f ~ 1/f we find an
element "w" (either from the list or a new one) from the same
comparability class which goes to zero at infinity. In our example we
set w=exp(-x) (but we could also set w=exp(-2x) or w=exp(-3x) ...). We
rewrite the mrv set using w, in our case {1/w, 1/w^2}, and substitute it
into f. Then we expand f into a series in w::

    f = c0*w^e0 + c1*w^e1 + ... + O(w^en),       where e0<e1<...<en, c0!=0

but for x->oo, lim f = lim c0*w^e0, because all the other terms go to zero,
because w goes to zero faster than the ci and ei. So::

    for e0>0, lim f = 0
    for e0<0, lim f = +-oo   (the sign depends on the sign of c0)
    for e0=0, lim f = lim c0

We need to recursively compute limits at several places of the algorithm, but
as is shown in the PhD thesis, it always finishes.

Important functions from the implementation:

compare(a, b, x) compares "a" and "b" by computing the limit L.
mrv(e, x) returns list of most rapidly varying (mrv) subexpressions of "e"
rewrite(e, Omega, x, wsym) rewrites "e" in terms of w
leadterm(f, x) returns the lowest power term in the series of f
mrv_leadterm(e, x) returns the lead term (c0, e0) for e
limitinf(e, x) computes lim e  (for x->oo)
limit(e, z, z0) computes any limit by converting it to the case x->oo

All the functions are really simple and straightforward except
rewrite(), which is the most difficult/complex part of the algorithm.
When the algorithm fails, the bugs are usually in the series expansion
(i.e. in SymPy) or in rewrite.

This code is almost exact rewrite of the Maple code inside the Gruntz
thesis.

Debugging
---------

Because the gruntz algorithm is highly recursive, it's difficult to
figure out what went wrong inside a debugger. Instead, turn on nice
debug prints by defining the environment variable SYMPY_DEBUG. For
example:

[user@localhost]: SYMPY_DEBUG=True ./bin/isympy

In [1]: limit(sin(x)/x, x, 0)
limitinf(_x*sin(1/_x), _x) = 1
+-mrv_leadterm(_x*sin(1/_x), _x) = (1, 0)
| +-mrv(_x*sin(1/_x), _x) = set([_x])
| | +-mrv(_x, _x) = set([_x])
| | +-mrv(sin(1/_x), _x) = set([_x])
| |   +-mrv(1/_x, _x) = set([_x])
| |     +-mrv(_x, _x) = set([_x])
| +-mrv_leadterm(exp(_x)*sin(exp(-_x)), _x, set([exp(_x)])) = (1, 0)
|   +-rewrite(exp(_x)*sin(exp(-_x)), set([exp(_x)]), _x, _w) = (1/_w*sin(_w), -_x)
|     +-sign(_x, _x) = 1
|     +-mrv_leadterm(1, _x) = (1, 0)
+-sign(0, _x) = 0
+-limitinf(1, _x) = 1

And check manually which line is wrong. Then go to the source code and
debug this function to figure out the exact problem.

"""
from functools import reduce

from sympy.core import Basic, S, Mul, PoleError
from sympy.core.cache import cacheit
from sympy.core.function import AppliedUndef
from sympy.core.intfunc import ilcm
from sympy.core.numbers import I, oo
from sympy.core.symbol import Dummy, Wild
from sympy.core.traversal import bottom_up

from sympy.functions import log, exp, sign as _sign
from sympy.series.order import Order
from sympy.utilities.misc import debug_decorator as debug
from sympy.utilities.timeutils import timethis

timeit = timethis('gruntz')


def compare(a, b, x):
    """Returns "<" if a<b, "=" for a == b, ">" for a>b"""
    # log(exp(...)) must always be simplified here for termination
    la, lb = log(a), log(b)
    if isinstance(a, Basic) and (isinstance(a, exp) or (a.is_Pow and a.base == S.Exp1)):
        la = a.exp
    if isinstance(b, Basic) and (isinstance(b, exp) or (b.is_Pow and b.base == S.Exp1)):
        lb = b.exp

    c = limitinf(la/lb, x)
    if c == 0:
        return "<"
    elif c.is_infinite:
        return ">"
    else:
        return "="


class SubsSet(dict):
    """
    Stores (expr, dummy) pairs, and how to rewrite expr-s.

    Explanation
    ===========

    The gruntz algorithm needs to rewrite certain expressions in term of a new
    variable w. We cannot use subs, because it is just too smart for us. For
    example::

        > Omega=[exp(exp(_p - exp(-_p))/(1 - 1/_p)), exp(exp(_p))]
        > O2=[exp(-exp(_p) + exp(-exp(-_p))*exp(_p)/(1 - 1/_p))/_w, 1/_w]
        > e = exp(exp(_p - exp(-_p))/(1 - 1/_p)) - exp(exp(_p))
        > e.subs(Omega[0],O2[0]).subs(Omega[1],O2[1])
        -1/w + exp(exp(p)*exp(-exp(-p))/(1 - 1/p))

    is really not what we want!

    So we do it the hard way and keep track of all the things we potentially
    want to substitute by dummy variables. Consider the expression::

        exp(x - exp(-x)) + exp(x) + x.

    The mrv set is {exp(x), exp(-x), exp(x - exp(-x))}.
    We introduce corresponding dummy variables d1, d2, d3 and rewrite::

        d3 + d1 + x.

    This class first of all keeps track of the mapping expr->variable, i.e.
    will at this stage be a dictionary::

        {exp(x): d1, exp(-x): d2, exp(x - exp(-x)): d3}.

    [It turns out to be more convenient this way round.]
    But sometimes expressions in the mrv set have other expressions from the
    mrv set as subexpressions, and we need to keep track of that as well. In
    this case, d3 is really exp(x - d2), so rewrites at this stage is::

        {d3: exp(x-d2)}.

    The function rewrite uses all this information to correctly rewrite our
    expression in terms of w. In this case w can be chosen to be exp(-x),
    i.e. d2. The correct rewriting then is::

        exp(-w)/w + 1/w + x.
    """
    def __init__(self):
        self.rewrites = {}

    def __repr__(self):
        return super().__repr__() + ', ' + self.rewrites.__repr__()

    def __getitem__(self, key):
        if key not in self:
            self[key] = Dummy()
        return dict.__getitem__(self, key)

    def do_subs(self, e):
        """Substitute the variables with expressions"""
        for expr, var in self.items():
            e = e.xreplace({var: expr})
        return e

    def meets(self, s2):
        """Tell whether or not self and s2 have non-empty intersection"""
        return set(self.keys()).intersection(list(s2.keys())) != set()

    def union(self, s2, exps=None):
        """Compute the union of self and s2, adjusting exps"""
        res = self.copy()
        tr = {}
        for expr, var in s2.items():
            if expr in self:
                if exps:
                    exps = exps.xreplace({var: res[expr]})
                tr[var] = res[expr]
            else:
                res[expr] = var
        for var, rewr in s2.rewrites.items():
            res.rewrites[var] = rewr.xreplace(tr)
        return res, exps

    def copy(self):
        """Create a shallow copy of SubsSet"""
        r = SubsSet()
        r.rewrites = self.rewrites.copy()
        for expr, var in self.items():
            r[expr] = var
        return r


@debug
def mrv(e, x):
    """Returns a SubsSet of most rapidly varying (mrv) subexpressions of 'e',
       and e rewritten in terms of these"""
    from sympy.simplify.powsimp import powsimp
    e = powsimp(e, deep=True, combine='exp')
    if not isinstance(e, Basic):
        raise TypeError("e should be an instance of Basic")
    if not e.has(x):
        return SubsSet(), e
    elif e == x:
        s = SubsSet()
        return s, s[x]
    elif e.is_Mul or e.is_Add:
        i, d = e.as_independent(x)  # throw away x-independent terms
        if d.func != e.func:
            s, expr = mrv(d, x)
            return s, e.func(i, expr)
        a, b = d.as_two_terms()
        s1, e1 = mrv(a, x)
        s2, e2 = mrv(b, x)
        return mrv_max1(s1, s2, e.func(i, e1, e2), x)
    elif e.is_Pow and e.base != S.Exp1:
        e1 = S.One
        while e.is_Pow:
            b1 = e.base
            e1 *= e.exp
            e = b1
        if b1 == 1:
            return SubsSet(), b1
        if e1.has(x):
            return mrv(exp(e1*log(b1)), x)
        else:
            s, expr = mrv(b1, x)
            return s, expr**e1
    elif isinstance(e, log):
        s, expr = mrv(e.args[0], x)
        return s, log(expr)
    elif isinstance(e, exp) or (e.is_Pow and e.base == S.Exp1):
        # We know from the theory of this algorithm that exp(log(...)) may always
        # be simplified here, and doing so is vital for termination.
        if isinstance(e.exp, log):
            return mrv(e.exp.args[0], x)
        # if a product has an infinite factor the result will be
        # infinite if there is no zero, otherwise NaN; here, we
        # consider the result infinite if any factor is infinite
        li = limitinf(e.exp, x)
        if any(_.is_infinite for _ in Mul.make_args(li)):
            s1 = SubsSet()
            e1 = s1[e]
            s2, e2 = mrv(e.exp, x)
            su = s1.union(s2)[0]
            su.rewrites[e1] = exp(e2)
            return mrv_max3(s1, e1, s2, exp(e2), su, e1, x)
        else:
            s, expr = mrv(e.exp, x)
            return s, exp(expr)
    elif isinstance(e, AppliedUndef):
        raise ValueError("MRV set computation for UndefinedFunction is not allowed")
    elif e.is_Function:
        l = [mrv(a, x) for a in e.args]
        l2 = [s for (s, _) in l if s != SubsSet()]
        if len(l2) != 1:
            # e.g. something like BesselJ(x, x)
            raise NotImplementedError("MRV set computation for functions in"
                                      " several variables not implemented.")
        s, ss = l2[0], SubsSet()
        args = [ss.do_subs(x[1]) for x in l]
        return s, e.func(*args)
    elif e.is_Derivative:
        raise NotImplementedError("MRV set computation for derivatives"
                                  " not implemented yet.")
    raise NotImplementedError(
        "Don't know how to calculate the mrv of '%s'" % e)


def mrv_max3(f, expsf, g, expsg, union, expsboth, x):
    """
    Computes the maximum of two sets of expressions f and g, which
    are in the same comparability class, i.e. max() compares (two elements of)
    f and g and returns either (f, expsf) [if f is larger], (g, expsg)
    [if g is larger] or (union, expsboth) [if f, g are of the same class].
    """
    if not isinstance(f, SubsSet):
        raise TypeError("f should be an instance of SubsSet")
    if not isinstance(g, SubsSet):
        raise TypeError("g should be an instance of SubsSet")
    if f == SubsSet():
        return g, expsg
    elif g == SubsSet():
        return f, expsf
    elif f.meets(g):
        return union, expsboth

    c = compare(list(f.keys())[0], list(g.keys())[0], x)
    if c == ">":
        return f, expsf
    elif c == "<":
        return g, expsg
    else:
        if c != "=":
            raise ValueError("c should be =")
        return union, expsboth


def mrv_max1(f, g, exps, x):
    """Computes the maximum of two sets of expressions f and g, which
    are in the same comparability class, i.e. mrv_max1() compares (two elements of)
    f and g and returns the set, which is in the higher comparability class
    of the union of both, if they have the same order of variation.
    Also returns exps, with the appropriate substitutions made.
    """
    u, b = f.union(g, exps)
    return mrv_max3(f, g.do_subs(exps), g, f.do_subs(exps),
                    u, b, x)


@debug
@cacheit
@timeit
def sign(e, x):
    """
    Returns a sign of an expression e(x) for x->oo.

    ::

        e >  0 for x sufficiently large ...  1
        e == 0 for x sufficiently large ...  0
        e <  0 for x sufficiently large ... -1

    The result of this function is currently undefined if e changes sign
    arbitrarily often for arbitrarily large x (e.g. sin(x)).

    Note that this returns zero only if e is *constantly* zero
    for x sufficiently large. [If e is constant, of course, this is just
    the same thing as the sign of e.]
    """
    if not isinstance(e, Basic):
        raise TypeError("e should be an instance of Basic")

    if e.is_positive:
        return 1
    elif e.is_negative:
        return -1
    elif e.is_zero:
        return 0

    elif not e.has(x):
        from sympy.simplify import logcombine
        e = logcombine(e)
        return _sign(e)
    elif e == x:
        return 1
    elif e.is_Mul:
        a, b = e.as_two_terms()
        sa = sign(a, x)
        if not sa:
            return 0
        return sa * sign(b, x)
    elif isinstance(e, exp):
        return 1
    elif e.is_Pow:
        if e.base == S.Exp1:
            return 1
        s = sign(e.base, x)
        if s == 1:
            return 1
        if e.exp.is_Integer:
            return s**e.exp
    elif isinstance(e, log) and e.args[0].is_positive:
        return sign(e.args[0] - 1, x)

    # if all else fails, do it the hard way
    c0, e0 = mrv_leadterm(e, x)
    return sign(c0, x)


@debug
@timeit
@cacheit
def limitinf(e, x):
    """Limit e(x) for x-> oo."""
    # rewrite e in terms of tractable functions only

    old = e
    if not e.has(x):
        return e  # e is a constant
    from sympy.simplify.powsimp import powdenest
    from sympy.calculus.util import AccumBounds
    if e.has(Order):
        e = e.expand().removeO()
    if not x.is_positive or x.is_integer:
        # We make sure that x.is_positive is True and x.is_integer is None
        # so we get all the correct mathematical behavior from the expression.
        # We need a fresh variable.
        p = Dummy('p', positive=True)
        e = e.subs(x, p)
        x = p
    e = e.rewrite('tractable', deep=True, limitvar=x)
    e = powdenest(e)
    if isinstance(e, AccumBounds):
        if mrv_leadterm(e.min, x) != mrv_leadterm(e.max, x):
            raise NotImplementedError
        c0, e0 = mrv_leadterm(e.min, x)
    else:
        c0, e0 = mrv_leadterm(e, x)
    sig = sign(e0, x)
    if sig == 1:
        return S.Zero  # e0>0: lim f = 0
    elif sig == -1:  # e0<0: lim f = +-oo (the sign depends on the sign of c0)
        if c0.match(I*Wild("a", exclude=[I])):
            return c0*oo
        s = sign(c0, x)
        # the leading term shouldn't be 0:
        if s == 0:
            raise ValueError("Leading term should not be 0")
        return s*oo
    elif sig == 0:
        if c0 == old:
            c0 = c0.cancel()
        return limitinf(c0, x)  # e0=0: lim f = lim c0
    else:
        raise ValueError("{} could not be evaluated".format(sig))


def moveup2(s, x):
    r = SubsSet()
    for expr, var in s.items():
        r[expr.xreplace({x: exp(x)})] = var
    for var, expr in s.rewrites.items():
        r.rewrites[var] = s.rewrites[var].xreplace({x: exp(x)})
    return r


def moveup(l, x):
    return [e.xreplace({x: exp(x)}) for e in l]


@debug
@timeit
@cacheit
def mrv_leadterm(e, x):
    """Returns (c0, e0) for e."""
    Omega = SubsSet()
    if not e.has(x):
        return (e, S.Zero)
    if Omega == SubsSet():
        Omega, exps = mrv(e, x)
    if not Omega:
        # e really does not depend on x after simplification
        return exps, S.Zero
    if x in Omega:
        # move the whole omega up (exponentiate each term):
        Omega_up = moveup2(Omega, x)
        exps_up = moveup([exps], x)[0]
        # NOTE: there is no need to move this down!
        Omega = Omega_up
        exps = exps_up
    #
    # The positive dummy, w, is used here so log(w*2) etc. will expand;
    # a unique dummy is needed in this algorithm
    #
    # For limits of complex functions, the algorithm would have to be
    # improved, or just find limits of Re and Im components separately.
    #
    w = Dummy("w", positive=True)
    f, logw = rewrite(exps, Omega, x, w)

    # Ensure expressions of the form exp(log(...)) don't get simplified automatically in the previous steps.
    # see: https://github.com/sympy/sympy/issues/15323#issuecomment-478639399
    f = f.replace(lambda f: f.is_Pow and f.has(x), lambda f: exp(log(f.base)*f.exp))

    try:
        lt = f.leadterm(w, logx=logw)
    except (NotImplementedError, PoleError, ValueError):
        n0 = 1
        _series = Order(1)
        incr = S.One
        while _series.is_Order:
            _series = f._eval_nseries(w, n=n0+incr, logx=logw)
            incr *= 2
        series = _series.expand().removeO()
        try:
            lt = series.leadterm(w, logx=logw)
        except (NotImplementedError, PoleError, ValueError):
            lt = f.as_coeff_exponent(w)
            if lt[0].has(w):
                base = f.as_base_exp()[0].as_coeff_exponent(w)
                ex = f.as_base_exp()[1]
                lt = (base[0]**ex, base[1]*ex)
    return (lt[0].subs(log(w), logw), lt[1])


def build_expression_tree(Omega, rewrites):
    r""" Helper function for rewrite.

    We need to sort Omega (mrv set) so that we replace an expression before
    we replace any expression in terms of which it has to be rewritten::

        e1 ---> e2 ---> e3
                 \
                  -> e4

    Here we can do e1, e2, e3, e4 or e1, e2, e4, e3.
    To do this we assemble the nodes into a tree, and sort them by height.

    This function builds the tree, rewrites then sorts the nodes.
    """
    class Node:
        def __init__(self):
            self.before = []
            self.expr = None
            self.var = None
        def ht(self):
            return reduce(lambda x, y: x + y,
                          [x.ht() for x in self.before], 1)
    nodes = {}
    for expr, v in Omega:
        n = Node()
        n.var = v
        n.expr = expr
        nodes[v] = n
    for _, v in Omega:
        if v in rewrites:
            n = nodes[v]
            r = rewrites[v]
            for _, v2 in Omega:
                if r.has(v2):
                    n.before.append(nodes[v2])

    return nodes


@debug
@timeit
def rewrite(e, Omega, x, wsym):
    """e(x) ... the function
    Omega ... the mrv set
    wsym ... the symbol which is going to be used for w

    Returns the rewritten e in terms of w and log(w). See test_rewrite1()
    for examples and correct results.
    """

    from sympy import AccumBounds
    if not isinstance(Omega, SubsSet):
        raise TypeError("Omega should be an instance of SubsSet")
    if len(Omega) == 0:
        raise ValueError("Length cannot be 0")
    # all items in Omega must be exponentials
    for t in Omega.keys():
        if not isinstance(t, exp):
            raise ValueError("Value should be exp")
    rewrites = Omega.rewrites
    Omega = list(Omega.items())

    nodes = build_expression_tree(Omega, rewrites)
    Omega.sort(key=lambda x: nodes[x[1]].ht(), reverse=True)

    # make sure we know the sign of each exp() term; after the loop,
    # g is going to be the "w" - the simplest one in the mrv set
    for g, _ in Omega:
        sig = sign(g.exp, x)
        if sig != 1 and sig != -1 and not sig.has(AccumBounds):
            raise NotImplementedError('Result depends on the sign of %s' % sig)
    if sig == 1:
        wsym = 1/wsym  # if g goes to oo, substitute 1/w
    # O2 is a list, which results by rewriting each item in Omega using "w"
    O2 = []
    denominators = []
    for f, var in Omega:
        c = limitinf(f.exp/g.exp, x)
        if c.is_Rational:
            denominators.append(c.q)
        arg = f.exp
        if var in rewrites:
            if not isinstance(rewrites[var], exp):
                raise ValueError("Value should be exp")
            arg = rewrites[var].args[0]
        O2.append((var, exp((arg - c*g.exp))*wsym**c))

    # Remember that Omega contains subexpressions of "e". So now we find
    # them in "e" and substitute them for our rewriting, stored in O2

    # the following powsimp is necessary to automatically combine exponentials,
    # so that the .xreplace() below succeeds:
    # TODO this should not be necessary
    from sympy.simplify.powsimp import powsimp
    f = powsimp(e, deep=True, combine='exp')
    for a, b in O2:
        f = f.xreplace({a: b})

    for _, var in Omega:
        assert not f.has(var)

    # finally compute the logarithm of w (logw).
    logw = g.exp
    if sig == 1:
        logw = -logw  # log(w)->log(1/w)=-log(w)

    # Some parts of SymPy have difficulty computing series expansions with
    # non-integral exponents. The following heuristic improves the situation:
    exponent = reduce(ilcm, denominators, 1)
    f = f.subs({wsym: wsym**exponent})
    logw /= exponent

    # bottom_up function is required for a specific case - when f is
    # -exp(p/(p + 1)) + exp(-p**2/(p + 1) + p). No current simplification
    # methods reduce this to 0 while not expanding polynomials.
    f = bottom_up(f, lambda w: getattr(w, 'normal', lambda: w)())

    return f, logw


def gruntz(e, z, z0, dir="+"):
    """
    Compute the limit of e(z) at the point z0 using the Gruntz algorithm.

    Explanation
    ===========

    ``z0`` can be any expression, including oo and -oo.

    For ``dir="+"`` (default) it calculates the limit from the right
    (z->z0+) and for ``dir="-"`` the limit from the left (z->z0-). For infinite z0
    (oo or -oo), the dir argument does not matter.

    This algorithm is fully described in the module docstring in the gruntz.py
    file. It relies heavily on the series expansion. Most frequently, gruntz()
    is only used if the faster limit() function (which uses heuristics) fails.
    """
    if not z.is_symbol:
        raise NotImplementedError("Second argument must be a Symbol")

    # convert all limits to the limit z->oo; sign of z is handled in limitinf
    r = None
    if z0 in (oo, I*oo):
        e0 = e
    elif z0 in (-oo, -I*oo):
        e0 = e.subs(z, -z)
    else:
        if str(dir) == "-":
            e0 = e.subs(z, z0 - 1/z)
        elif str(dir) == "+":
            e0 = e.subs(z, z0 + 1/z)
        else:
            raise NotImplementedError("dir must be '+' or '-'")

    r = limitinf(e0, z)

    # This is a bit of a heuristic for nice results... we always rewrite
    # tractable functions in terms of familiar intractable ones.
    # It might be nicer to rewrite the exactly to what they were initially,
    # but that would take some work to implement.
    return r.rewrite('intractable', deep=True)
<|endoftext|>"""Fourier Series"""

from sympy.core.numbers import (oo, pi)
from sympy.core.symbol import Wild
from sympy.core.expr import Expr
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import sin, cos, sinc
from sympy.series.series_class import SeriesBase
from sympy.series.sequences import SeqFormula
from sympy.sets.sets import Interval
from sympy.utilities.iterables import is_sequence


__doctest_requires__ = {('fourier_series',): ['matplotlib']}


def fourier_cos_seq(func, limits, n):
    """Returns the cos sequence in a Fourier series"""
    from sympy.integrals import integrate
    x, L = limits[0], limits[2] - limits[1]
    cos_term = cos(2*n*pi*x / L)
    formula = 2 * cos_term * integrate(func * cos_term, limits) / L
    a0 = formula.subs(n, S.Zero) / 2
    return a0, SeqFormula(2 * cos_term * integrate(func * cos_term, limits)
                          / L, (n, 1, oo))


def fourier_sin_seq(func, limits, n):
    """Returns the sin sequence in a Fourier series"""
    from sympy.integrals import integrate
    x, L = limits[0], limits[2] - limits[1]
    sin_term = sin(2*n*pi*x / L)
    return SeqFormula(2 * sin_term * integrate(func * sin_term, limits)
                      / L, (n, 1, oo))


def _process_limits(func, limits):
    """
    Limits should be of the form (x, start, stop).
    x should be a symbol. Both start and stop should be bounded.

    Explanation
    ===========

    * If x is not given, x is determined from func.
    * If limits is None. Limit of the form (x, -pi, pi) is returned.

    Examples
    ========

    >>> from sympy.series.fourier import _process_limits as pari
    >>> from sympy.abc import x
    >>> pari(x**2, (x, -2, 2))
    (x, -2, 2)
    >>> pari(x**2, (-2, 2))
    (x, -2, 2)
    >>> pari(x**2, None)
    (x, -pi, pi)
    """
    def _find_x(func):
        free = func.free_symbols
        if len(free) == 1:
            return free.pop()
        elif not free:
            return Dummy('k')
        else:
            raise ValueError(
                " specify dummy variables for %s. If the function contains"
                " more than one free symbol, a dummy variable should be"
                " supplied explicitly e.g. FourierSeries(m*n**2, (n, -pi, pi))"
                % func)

    x, start, stop = None, None, None
    if limits is None:
        x, start, stop = _find_x(func), -pi, pi
    if is_sequence(limits, Tuple):
        if len(limits) == 3:
            x, start, stop = limits
        elif len(limits) == 2:
            x = _find_x(func)
            start, stop = limits

    if not isinstance(x, Symbol) or start is None or stop is None:
        raise ValueError('Invalid limits given: %s' % str(limits))

    unbounded = [S.NegativeInfinity, S.Infinity]
    if start in unbounded or stop in unbounded:
        raise ValueError("Both the start and end value should be bounded")

    return sympify((x, start, stop))


def finite_check(f, x, L):

    def check_fx(exprs, x):
        return x not in exprs.free_symbols

    def check_sincos(_expr, x, L):
        if isinstance(_expr, (sin, cos)):
            sincos_args = _expr.args[0]

            if sincos_args.match(a*(pi/L)*x + b) is not None:
                return True
            else:
                return False

    from sympy.simplify.fu import TR2, TR1, sincos_to_sum
    _expr = sincos_to_sum(TR2(TR1(f)))
    add_coeff = _expr.as_coeff_add()

    a = Wild('a', properties=[lambda k: k.is_Integer, lambda k: k != S.Zero, ])
    b = Wild('b', properties=[lambda k: x not in k.free_symbols, ])

    for s in add_coeff[1]:
        mul_coeffs = s.as_coeff_mul()[1]
        for t in mul_coeffs:
            if not (check_fx(t, x) or check_sincos(t, x, L)):
                return False, f

    return True, _expr


class FourierSeries(SeriesBase):
    r"""Represents Fourier sine/cosine series.

    Explanation
    ===========

    This class only represents a fourier series.
    No computation is performed.

    For how to compute Fourier series, see the :func:`fourier_series`
    docstring.

    See Also
    ========

    sympy.series.fourier.fourier_series
    """
    def __new__(cls, *args):
        args = map(sympify, args)
        return Expr.__new__(cls, *args)

    @property
    def function(self):
        return self.args[0]

    @property
    def x(self):
        return self.args[1][0]

    @property
    def period(self):
        return (self.args[1][1], self.args[1][2])

    @property
    def a0(self):
        return self.args[2][0]

    @property
    def an(self):
        return self.args[2][1]

    @property
    def bn(self):
        return self.args[2][2]

    @property
    def interval(self):
        return Interval(0, oo)

    @property
    def start(self):
        return self.interval.inf

    @property
    def stop(self):
        return self.interval.sup

    @property
    def length(self):
        return oo

    @property
    def L(self):
        return abs(self.period[1] - self.period[0]) / 2

    def _eval_subs(self, old, new):
        x = self.x
        if old.has(x):
            return self

    def truncate(self, n=3):
        """
        Return the first n nonzero terms of the series.

        If ``n`` is None return an iterator.

        Parameters
        ==========

        n : int or None
            Amount of non-zero terms in approximation or None.

        Returns
        =======

        Expr or iterator :
            Approximation of function expanded into Fourier series.

        Examples
        ========

        >>> from sympy import fourier_series, pi
        >>> from sympy.abc import x
        >>> s = fourier_series(x, (x, -pi, pi))
        >>> s.truncate(4)
        2*sin(x) - sin(2*x) + 2*sin(3*x)/3 - sin(4*x)/2

        See Also
        ========

        sympy.series.fourier.FourierSeries.sigma_approximation
        """
        if n is None:
            return iter(self)

        terms = []
        for t in self:
            if len(terms) == n:
                break
            if t is not S.Zero:
                terms.append(t)

        return Add(*terms)

    def sigma_approximation(self, n=3):
        r"""
        Return :math:`\sigma`-approximation of Fourier series with respect
        to order n.

        Explanation
        ===========

        Sigma approximation adjusts a Fourier summation to eliminate the Gibbs
        phenomenon which would otherwise occur at discontinuities.
        A sigma-approximated summation for a Fourier series of a T-periodical
        function can be written as

        .. math::
            s(\theta) = \frac{1}{2} a_0 + \sum _{k=1}^{m-1}
            \operatorname{sinc} \Bigl( \frac{k}{m} \Bigr) \cdot
            \left[ a_k \cos \Bigl( \frac{2\pi k}{T} \theta \Bigr)
            + b_k \sin \Bigl( \frac{2\pi k}{T} \theta \Bigr) \right],

        where :math:`a_0, a_k, b_k, k=1,\ldots,{m-1}` are standard Fourier
        series coefficients and
        :math:`\operatorname{sinc} \Bigl( \frac{k}{m} \Bigr)` is a Lanczos
        :math:`\sigma` factor (expressed in terms of normalized
        :math:`\operatorname{sinc}` function).

        Parameters
        ==========

        n : int
            Highest order of the terms taken into account in approximation.

        Returns
        =======

        Expr :
            Sigma approximation of function expanded into Fourier series.

        Examples
        ========

        >>> from sympy import fourier_series, pi
        >>> from sympy.abc import x
        >>> s = fourier_series(x, (x, -pi, pi))
        >>> s.sigma_approximation(4)
        2*sin(x)*sinc(pi/4) - 2*sin(2*x)/pi + 2*sin(3*x)*sinc(3*pi/4)/3

        See Also
        ========

        sympy.series.fourier.FourierSeries.truncate

        Notes
        =====

        The behaviour of
        :meth:`~sympy.series.fourier.FourierSeries.sigma_approximation`
        is different from :meth:`~sympy.series.fourier.FourierSeries.truncate`
        - it takes all nonzero terms of degree smaller than n, rather than
        first n nonzero ones.

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Gibbs_phenomenon
        .. [2] https://en.wikipedia.org/wiki/Sigma_approximation
        """
        terms = [sinc(pi * i / n) * t for i, t in enumerate(self[:n])
                 if t is not S.Zero]
        return Add(*terms)

    def shift(self, s):
        """
        Shift the function by a term independent of x.

        Explanation
        ===========

        f(x) -> f(x) + s

        This is fast, if Fourier series of f(x) is already
        computed.

        Examples
        ========

        >>> from sympy import fourier_series, pi
        >>> from sympy.abc import x
        >>> s = fourier_series(x**2, (x, -pi, pi))
        >>> s.shift(1).truncate()
        -4*cos(x) + cos(2*x) + 1 + pi**2/3
        """
        s, x = sympify(s), self.x

        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))

        a0 = self.a0 + s
        sfunc = self.function + s

        return self.func(sfunc, self.args[1], (a0, self.an, self.bn))

    def shiftx(self, s):
        """
        Shift x by a term independent of x.

        Explanation
        ===========

        f(x) -> f(x + s)

        This is fast, if Fourier series of f(x) is already
        computed.

        Examples
        ========

        >>> from sympy import fourier_series, pi
        >>> from sympy.abc import x
        >>> s = fourier_series(x**2, (x, -pi, pi))
        >>> s.shiftx(1).truncate()
        -4*cos(x + 1) + cos(2*x + 2) + pi**2/3
        """
        s, x = sympify(s), self.x

        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))

        an = self.an.subs(x, x + s)
        bn = self.bn.subs(x, x + s)
        sfunc = self.function.subs(x, x + s)

        return self.func(sfunc, self.args[1], (self.a0, an, bn))

    def scale(self, s):
        """
        Scale the function by a term independent of x.

        Explanation
        ===========

        f(x) -> s * f(x)

        This is fast, if Fourier series of f(x) is already
        computed.

        Examples
        ========

        >>> from sympy import fourier_series, pi
        >>> from sympy.abc import x
        >>> s = fourier_series(x**2, (x, -pi, pi))
        >>> s.scale(2).truncate()
        -8*cos(x) + 2*cos(2*x) + 2*pi**2/3
        """
        s, x = sympify(s), self.x

        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))

        an = self.an.coeff_mul(s)
        bn = self.bn.coeff_mul(s)
        a0 = self.a0 * s
        sfunc = self.args[0] * s

        return self.func(sfunc, self.args[1], (a0, an, bn))

    def scalex(self, s):
        """
        Scale x by a term independent of x.

        Explanation
        ===========

        f(x) -> f(s*x)

        This is fast, if Fourier series of f(x) is already
        computed.

        Examples
        ========

        >>> from sympy import fourier_series, pi
        >>> from sympy.abc import x
        >>> s = fourier_series(x**2, (x, -pi, pi))
        >>> s.scalex(2).truncate()
        -4*cos(2*x) + cos(4*x) + pi**2/3
        """
        s, x = sympify(s), self.x

        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))

        an = self.an.subs(x, x * s)
        bn = self.bn.subs(x, x * s)
        sfunc = self.function.subs(x, x * s)

        return self.func(sfunc, self.args[1], (self.a0, an, bn))

    def _eval_as_leading_term(self, x, logx, cdir):
        for t in self:
            if t is not S.Zero:
                return t

    def _eval_term(self, pt):
        if pt == 0:
            return self.a0
        return self.an.coeff(pt) + self.bn.coeff(pt)

    def __neg__(self):
        return self.scale(-1)

    def __add__(self, other):
        if isinstance(other, FourierSeries):
            if self.period != other.period:
                raise ValueError("Both the series should have same periods")

            x, y = self.x, other.x
            function = self.function + other.function.subs(y, x)

            if self.x not in function.free_symbols:
                return function

            an = self.an + other.an
            bn = self.bn + other.bn
            a0 = self.a0 + other.a0

            return self.func(function, self.args[1], (a0, an, bn))

        return Add(self, other)

    def __sub__(self, other):
        return self.__add__(-other)


class FiniteFourierSeries(FourierSeries):
    r"""Represents Finite Fourier sine/cosine series.

    For how to compute Fourier series, see the :func:`fourier_series`
    docstring.

    Parameters
    ==========

    f : Expr
        Expression for finding fourier_series

    limits : ( x, start, stop)
        x is the independent variable for the expression f
        (start, stop) is the period of the fourier series

    exprs: (a0, an, bn) or Expr
        a0 is the constant term a0 of the fourier series
        an is a dictionary of coefficients of cos terms
         an[k] = coefficient of cos(pi*(k/L)*x)
        bn is a dictionary of coefficients of sin terms
         bn[k] = coefficient of sin(pi*(k/L)*x)

        or exprs can be an expression to be converted to fourier form

    Methods
    =======

    This class is an extension of FourierSeries class.
    Please refer to sympy.series.fourier.FourierSeries for
    further information.

    See Also
    ========

    sympy.series.fourier.FourierSeries
    sympy.series.fourier.fourier_series
    """

    def __new__(cls, f, limits, exprs):
        f = sympify(f)
        limits = sympify(limits)
        exprs = sympify(exprs)

        if not (isinstance(exprs, Tuple) and len(exprs) == 3):  # exprs is not of form (a0, an, bn)
            # Converts the expression to fourier form
            c, e = exprs.as_coeff_add()
            from sympy.simplify.fu import TR10
            rexpr = c + Add(*[TR10(i) for i in e])
            a0, exp_ls = rexpr.expand(trig=False, power_base=False, power_exp=False, log=False).as_coeff_add()

            x = limits[0]
            L = abs(limits[2] - limits[1]) / 2

            a = Wild('a', properties=[lambda k: k.is_Integer, lambda k: k is not S.Zero, ])
            b = Wild('b', properties=[lambda k: x not in k.free_symbols, ])

            an = {}
            bn = {}

            # separates the coefficients of sin and cos terms in dictionaries an, and bn
            for p in exp_ls:
                t = p.match(b * cos(a * (pi / L) * x))
                q = p.match(b * sin(a * (pi / L) * x))
                if t:
                    an[t[a]] = t[b] + an.get(t[a], S.Zero)
                elif q:
                    bn[q[a]] = q[b] + bn.get(q[a], S.Zero)
                else:
                    a0 += p

            exprs = Tuple(a0, an, bn)

        return Expr.__new__(cls, f, limits, exprs)

    @property
    def interval(self):
        _length = 1 if self.a0 else 0
        _length += max(set(self.an.keys()).union(set(self.bn.keys()))) + 1
        return Interval(0, _length)

    @property
    def length(self):
        return self.stop - self.start

    def shiftx(self, s):
        s, x = sympify(s), self.x

        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))

        _expr = self.truncate().subs(x, x + s)
        sfunc = self.function.subs(x, x + s)

        return self.func(sfunc, self.args[1], _expr)

    def scale(self, s):
        s, x = sympify(s), self.x

        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))

        _expr = self.truncate() * s
        sfunc = self.function * s

        return self.func(sfunc, self.args[1], _expr)

    def scalex(self, s):
        s, x = sympify(s), self.x

        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))

        _expr = self.truncate().subs(x, x * s)
        sfunc = self.function.subs(x, x * s)

        return self.func(sfunc, self.args[1], _expr)

    def _eval_term(self, pt):
        if pt == 0:
            return self.a0

        _term = self.an.get(pt, S.Zero) * cos(pt * (pi / self.L) * self.x) \
                + self.bn.get(pt, S.Zero) * sin(pt * (pi / self.L) * self.x)
        return _term

    def __add__(self, other):
        if isinstance(other, FourierSeries):
            return other.__add__(fourier_series(self.function, self.args[1],\
                                                finite=False))
        elif isinstance(other, FiniteFourierSeries):
            if self.period != other.period:
                raise ValueError("Both the series should have same periods")

            x, y = self.x, other.x
            function = self.function + other.function.subs(y, x)

            if self.x not in function.free_symbols:
                return function

            return fourier_series(function, limits=self.args[1])


def fourier_series(f, limits=None, finite=True):
    r"""Computes the Fourier trigonometric series expansion.

    Explanation
    ===========

    Fourier trigonometric series of $f(x)$ over the interval $(a, b)$
    is defined as:

    .. math::
        \frac{a_0}{2} + \sum_{n=1}^{\infty}
        (a_n \cos(\frac{2n \pi x}{L}) + b_n \sin(\frac{2n \pi x}{L}))

    where the coefficients are:

    .. math::
        L = b - a

    .. math::
        a_0 = \frac{2}{L} \int_{a}^{b}{f(x) dx}

    .. math::
        a_n = \frac{2}{L} \int_{a}^{b}{f(x) \cos(\frac{2n \pi x}{L}) dx}

    .. math::
        b_n = \frac{2}{L} \int_{a}^{b}{f(x) \sin(\frac{2n \pi x}{L}) dx}

    The condition whether the function $f(x)$ given should be periodic
    or not is more than necessary, because it is sufficient to consider
    the series to be converging to $f(x)$ only in the given interval,
    not throughout the whole real line.

    This also brings a lot of ease for the computation because
    you do not have to make $f(x)$ artificially periodic by
    wrapping it with piecewise, modulo operations,
    but you can shape the function to look like the desired periodic
    function only in the interval $(a, b)$, and the computed series will
    automatically become the series of the periodic version of $f(x)$.

    This property is illustrated in the examples section below.

    Parameters
    ==========

    limits : (sym, start, end), optional
        *sym* denotes the symbol the series is computed with respect to.

        *start* and *end* denotes the start and the end of the interval
        where the fourier series converges to the given function.

        Default range is specified as $-\pi$ and $\pi$.

    Returns
    =======

    FourierSeries
        A symbolic object representing the Fourier trigonometric series.

    Examples
    ========

    Computing the Fourier series of $f(x) = x^2$:

    >>> from sympy import fourier_series, pi
    >>> from sympy.abc import x
    >>> f = x**2
    >>> s = fourier_series(f, (x, -pi, pi))
    >>> s1 = s.truncate(n=3)
    >>> s1
    -4*cos(x) + cos(2*x) + pi**2/3

    Shifting of the Fourier series:

    >>> s.shift(1).truncate()
    -4*cos(x) + cos(2*x) + 1 + pi**2/3
    >>> s.shiftx(1).truncate()
    -4*cos(x + 1) + cos(2*x + 2) + pi**2/3

    Scaling of the Fourier series:

    >>> s.scale(2).truncate()
    -8*cos(x) + 2*cos(2*x) + 2*pi**2/3
    >>> s.scalex(2).truncate()
    -4*cos(2*x) + cos(4*x) + pi**2/3

    Computing the Fourier series of $f(x) = x$:

    This illustrates how truncating to the higher order gives better
    convergence.

    .. plot::
        :context: reset
        :format: doctest
        :include-source: True

        >>> from sympy import fourier_series, pi, plot
        >>> from sympy.abc import x
        >>> f = x
        >>> s = fourier_series(f, (x, -pi, pi))
        >>> s1 = s.truncate(n = 3)
        >>> s2 = s.truncate(n = 5)
        >>> s3 = s.truncate(n = 7)
        >>> p = plot(f, s1, s2, s3, (x, -pi, pi), show=False, legend=True)

        >>> p[0].line_color = (0, 0, 0)
        >>> p[0].label = 'x'
        >>> p[1].line_color = (0.7, 0.7, 0.7)
        >>> p[1].label = 'n=3'
        >>> p[2].line_color = (0.5, 0.5, 0.5)
        >>> p[2].label = 'n=5'
        >>> p[3].line_color = (0.3, 0.3, 0.3)
        >>> p[3].label = 'n=7'

        >>> p.show()

    This illustrates how the series converges to different sawtooth
    waves if the different ranges are specified.

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> s1 = fourier_series(x, (x, -1, 1)).truncate(10)
        >>> s2 = fourier_series(x, (x, -pi, pi)).truncate(10)
        >>> s3 = fourier_series(x, (x, 0, 1)).truncate(10)
        >>> p = plot(x, s1, s2, s3, (x, -5, 5), show=False, legend=True)

        >>> p[0].line_color = (0, 0, 0)
        >>> p[0].label = 'x'
        >>> p[1].line_color = (0.7, 0.7, 0.7)
        >>> p[1].label = '[-1, 1]'
        >>> p[2].line_color = (0.5, 0.5, 0.5)
        >>> p[2].label = '[-pi, pi]'
        >>> p[3].line_color = (0.3, 0.3, 0.3)
        >>> p[3].label = '[0, 1]'

        >>> p.show()

    Notes
    =====

    Computing Fourier series can be slow
    due to the integration required in computing
    an, bn.

    It is faster to compute Fourier series of a function
    by using shifting and scaling on an already
    computed Fourier series rather than computing
    again.

    e.g. If the Fourier series of ``x**2`` is known
    the Fourier series of ``x**2 - 1`` can be found by shifting by ``-1``.

    See Also
    ========

    sympy.series.fourier.FourierSeries

    References
    ==========

    .. [1] https://mathworld.wolfram.com/FourierSeries.html
    """
    f = sympify(f)

    limits = _process_limits(f, limits)
    x = limits[0]

    if x not in f.free_symbols:
        return f

    if finite:
        L = abs(limits[2] - limits[1]) / 2
        is_finite, res_f = finite_check(f, x, L)
        if is_finite:
            return FiniteFourierSeries(f, limits, res_f)

    n = Dummy('n')
    center = (limits[1] + limits[2]) / 2
    if center.is_zero:
        neg_f = f.subs(x, -x)
        if f == neg_f:
            a0, an = fourier_cos_seq(f, limits, n)
            bn = SeqFormula(0, (1, oo))
            return FourierSeries(f, limits, (a0, an, bn))
        elif f == -neg_f:
            a0 = S.Zero
            an = SeqFormula(0, (1, oo))
            bn = fourier_sin_seq(f, limits, n)
            return FourierSeries(f, limits, (a0, an, bn))
    a0, an = fourier_cos_seq(f, limits, n)
    bn = fourier_sin_seq(f, limits, n)
    return FourierSeries(f, limits, (a0, an, bn))
<|endoftext|>"""
This module implements the Residue function and related tools for working
with residues.
"""

from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.utilities.timeutils import timethis


@timethis('residue')
def residue(expr, x, x0):
    """
    Finds the residue of ``expr`` at the point x=x0.

    The residue is defined as the coefficient of ``1/(x-x0)`` in the power series
    expansion about ``x=x0``.

    Examples
    ========

    >>> from sympy import Symbol, residue, sin
    >>> x = Symbol("x")
    >>> residue(1/x, x, 0)
    1
    >>> residue(1/x**2, x, 0)
    0
    >>> residue(2/sin(x), x, 0)
    2

    This function is essential for the Residue Theorem [1].

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Residue_theorem
    """
    # The current implementation uses series expansion to
    # calculate it. A more general implementation is explained in
    # the section 5.6 of the Bronstein's book {M. Bronstein:
    # Symbolic Integration I, Springer Verlag (2005)}. For purely
    # rational functions, the algorithm is much easier. See
    # sections 2.4, 2.5, and 2.7 (this section actually gives an
    # algorithm for computing any Laurent series coefficient for
    # a rational function). The theory in section 2.4 will help to
    # understand why the resultant works in the general algorithm.
    # For the definition of a resultant, see section 1.4 (and any
    # previous sections for more review).

    from sympy.series.order import Order
    from sympy.simplify.radsimp import collect
    expr = sympify(expr)
    if x0 != 0:
        expr = expr.subs(x, x + x0)
    for n in (0, 1, 2, 4, 8, 16, 32):
        s = expr.nseries(x, n=n)
        if not s.has(Order) or s.getn() >= 0:
            break
    s = collect(s.removeO(), x)
    if s.is_Add:
        args = s.args
    else:
        args = [s]
    res = S.Zero
    for arg in args:
        c, m = arg.as_coeff_mul(x)
        m = Mul(*m)
        if not (m in (S.One, x) or (m.is_Pow and m.exp.is_Integer)):
            raise NotImplementedError('term of unexpected form: %s' % m)
        if m == 1/x:
            res += c
    return res
<|endoftext|>"""
Contains the base class for series
Made using sequences in mind
"""

from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.cache import cacheit


class SeriesBase(Expr):
    """Base Class for series"""

    @property
    def interval(self):
        """The interval on which the series is defined"""
        raise NotImplementedError("(%s).interval" % self)

    @property
    def start(self):
        """The starting point of the series. This point is included"""
        raise NotImplementedError("(%s).start" % self)

    @property
    def stop(self):
        """The ending point of the series. This point is included"""
        raise NotImplementedError("(%s).stop" % self)

    @property
    def length(self):
        """Length of the series expansion"""
        raise NotImplementedError("(%s).length" % self)

    @property
    def variables(self):
        """Returns a tuple of variables that are bounded"""
        return ()

    @property
    def free_symbols(self):
        """
        This method returns the symbols in the object, excluding those
        that take on a specific value (i.e. the dummy symbols).
        """
        return ({j for i in self.args for j in i.free_symbols}
                .difference(self.variables))

    @cacheit
    def term(self, pt):
        """Term at point pt of a series"""
        if pt < self.start or pt > self.stop:
            raise IndexError("Index %s out of bounds %s" % (pt, self.interval))
        return self._eval_term(pt)

    def _eval_term(self, pt):
        raise NotImplementedError("The _eval_term method should be added to"
                                  "%s to return series term so it is available"
                                  "when 'term' calls it."
                                  % self.func)

    def _ith_point(self, i):
        """
        Returns the i'th point of a series
        If start point is negative infinity, point is returned from the end.
        Assumes the first point to be indexed zero.

        Examples
        ========

        TODO
        """
        if self.start is S.NegativeInfinity:
            initial = self.stop
            step = -1
        else:
            initial = self.start
            step = 1

        return initial + i*step

    def __iter__(self):
        i = 0
        while i < self.length:
            pt = self._ith_point(i)
            yield self.term(pt)
            i += 1

    def __getitem__(self, index):
        if isinstance(index, int):
            index = self._ith_point(index)
            return self.term(index)
        elif isinstance(index, slice):
            start, stop = index.start, index.stop
            if start is None:
                start = 0
            if stop is None:
                stop = self.length
            return [self.term(self._ith_point(i)) for i in
                    range(start, stop, index.step or 1)]
<|endoftext|>from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.decorators import call_highest_priority
from sympy.core.parameters import global_parameters
from sympy.core.function import AppliedUndef, expand
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.relational import Eq
from sympy.core.singleton import S, Singleton
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol, Wild
from sympy.core.sympify import sympify
from sympy.matrices import Matrix
from sympy.polys import lcm, factor
from sympy.sets.sets import Interval, Intersection
from sympy.tensor.indexed import Idx
from sympy.utilities.iterables import flatten, is_sequence, iterable


###############################################################################
#                            SEQUENCES                                        #
###############################################################################


class SeqBase(Basic):
    """Base class for sequences"""

    is_commutative = True
    _op_priority = 15

    @staticmethod
    def _start_key(expr):
        """Return start (if possible) else S.Infinity.

        adapted from Set._infimum_key
        """
        try:
            start = expr.start
        except NotImplementedError:
            start = S.Infinity
        return start

    def _intersect_interval(self, other):
        """Returns start and stop.

        Takes intersection over the two intervals.
        """
        interval = Intersection(self.interval, other.interval)
        return interval.inf, interval.sup

    @property
    def gen(self):
        """Returns the generator for the sequence"""
        raise NotImplementedError("(%s).gen" % self)

    @property
    def interval(self):
        """The interval on which the sequence is defined"""
        raise NotImplementedError("(%s).interval" % self)

    @property
    def start(self):
        """The starting point of the sequence. This point is included"""
        raise NotImplementedError("(%s).start" % self)

    @property
    def stop(self):
        """The ending point of the sequence. This point is included"""
        raise NotImplementedError("(%s).stop" % self)

    @property
    def length(self):
        """Length of the sequence"""
        raise NotImplementedError("(%s).length" % self)

    @property
    def variables(self):
        """Returns a tuple of variables that are bounded"""
        return ()

    @property
    def free_symbols(self):
        """
        This method returns the symbols in the object, excluding those
        that take on a specific value (i.e. the dummy symbols).

        Examples
        ========

        >>> from sympy import SeqFormula
        >>> from sympy.abc import n, m
        >>> SeqFormula(m*n**2, (n, 0, 5)).free_symbols
        {m}
        """
        return ({j for i in self.args for j in i.free_symbols
                   .difference(self.variables)})

    @cacheit
    def coeff(self, pt):
        """Returns the coefficient at point pt"""
        if pt < self.start or pt > self.stop:
            raise IndexError("Index %s out of bounds %s" % (pt, self.interval))
        return self._eval_coeff(pt)

    def _eval_coeff(self, pt):
        raise NotImplementedError("The _eval_coeff method should be added to"
                                  "%s to return coefficient so it is available"
                                  "when coeff calls it."
                                  % self.func)

    def _ith_point(self, i):
        """Returns the i'th point of a sequence.

        Explanation
        ===========

        If start point is negative infinity, point is returned from the end.
        Assumes the first point to be indexed zero.

        Examples
        =========

        >>> from sympy import oo
        >>> from sympy.series.sequences import SeqPer

        bounded

        >>> SeqPer((1, 2, 3), (-10, 10))._ith_point(0)
        -10
        >>> SeqPer((1, 2, 3), (-10, 10))._ith_point(5)
        -5

        End is at infinity

        >>> SeqPer((1, 2, 3), (0, oo))._ith_point(5)
        5

        Starts at negative infinity

        >>> SeqPer((1, 2, 3), (-oo, 0))._ith_point(5)
        -5
        """
        if self.start is S.NegativeInfinity:
            initial = self.stop
        else:
            initial = self.start

        if self.start is S.NegativeInfinity:
            step = -1
        else:
            step = 1

        return initial + i*step

    def _add(self, other):
        """
        Should only be used internally.

        Explanation
        ===========

        self._add(other) returns a new, term-wise added sequence if self
        knows how to add with other, otherwise it returns ``None``.

        ``other`` should only be a sequence object.

        Used within :class:`SeqAdd` class.
        """
        return None

    def _mul(self, other):
        """
        Should only be used internally.

        Explanation
        ===========

        self._mul(other) returns a new, term-wise multiplied sequence if self
        knows how to multiply with other, otherwise it returns ``None``.

        ``other`` should only be a sequence object.

        Used within :class:`SeqMul` class.
        """
        return None

    def coeff_mul(self, other):
        """
        Should be used when ``other`` is not a sequence. Should be
        defined to define custom behaviour.

        Examples
        ========

        >>> from sympy import SeqFormula
        >>> from sympy.abc import n
        >>> SeqFormula(n**2).coeff_mul(2)
        SeqFormula(2*n**2, (n, 0, oo))

        Notes
        =====

        '*' defines multiplication of sequences with sequences only.
        """
        return Mul(self, other)

    def __add__(self, other):
        """Returns the term-wise addition of 'self' and 'other'.

        ``other`` should be a sequence.

        Examples
        ========

        >>> from sympy import SeqFormula
        >>> from sympy.abc import n
        >>> SeqFormula(n**2) + SeqFormula(n**3)
        SeqFormula(n**3 + n**2, (n, 0, oo))
        """
        if not isinstance(other, SeqBase):
            raise TypeError('cannot add sequence and %s' % type(other))
        return SeqAdd(self, other)

    @call_highest_priority('__add__')
    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        """Returns the term-wise subtraction of ``self`` and ``other``.

        ``other`` should be a sequence.

        Examples
        ========

        >>> from sympy import SeqFormula
        >>> from sympy.abc import n
        >>> SeqFormula(n**2) - (SeqFormula(n))
        SeqFormula(n**2 - n, (n, 0, oo))
        """
        if not isinstance(other, SeqBase):
            raise TypeError('cannot subtract sequence and %s' % type(other))
        return SeqAdd(self, -other)

    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        return (-self) + other

    def __neg__(self):
        """Negates the sequence.

        Examples
        ========

        >>> from sympy import SeqFormula
        >>> from sympy.abc import n
        >>> -SeqFormula(n**2)
        SeqFormula(-n**2, (n, 0, oo))
        """
        return self.coeff_mul(-1)

    def __mul__(self, other):
        """Returns the term-wise multiplication of 'self' and 'other'.

        ``other`` should be a sequence. For ``other`` not being a
        sequence see :func:`coeff_mul` method.

        Examples
        ========

        >>> from sympy import SeqFormula
        >>> from sympy.abc import n
        >>> SeqFormula(n**2) * (SeqFormula(n))
        SeqFormula(n**3, (n, 0, oo))
        """
        if not isinstance(other, SeqBase):
            raise TypeError('cannot multiply sequence and %s' % type(other))
        return SeqMul(self, other)

    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return self * other

    def __iter__(self):
        for i in range(self.length):
            pt = self._ith_point(i)
            yield self.coeff(pt)

    def __getitem__(self, index):
        if isinstance(index, int):
            index = self._ith_point(index)
            return self.coeff(index)
        elif isinstance(index, slice):
            start, stop = index.start, index.stop
            if start is None:
                start = 0
            if stop is None:
                stop = self.length
            return [self.coeff(self._ith_point(i)) for i in
                    range(start, stop, index.step or 1)]

    def find_linear_recurrence(self,n,d=None,gfvar=None):
        r"""
        Finds the shortest linear recurrence that satisfies the first n
        terms of sequence of order `\leq` ``n/2`` if possible.
        If ``d`` is specified, find shortest linear recurrence of order
        `\leq` min(d, n/2) if possible.
        Returns list of coefficients ``[b(1), b(2), ...]`` corresponding to the
        recurrence relation ``x(n) = b(1)*x(n-1) + b(2)*x(n-2) + ...``
        Returns ``[]`` if no recurrence is found.
        If gfvar is specified, also returns ordinary generating function as a
        function of gfvar.

        Examples
        ========

        >>> from sympy import sequence, sqrt, oo, lucas
        >>> from sympy.abc import n, x, y
        >>> sequence(n**2).find_linear_recurrence(10, 2)
        []
        >>> sequence(n**2).find_linear_recurrence(10)
        [3, -3, 1]
        >>> sequence(2**n).find_linear_recurrence(10)
        [2]
        >>> sequence(23*n**4+91*n**2).find_linear_recurrence(10)
        [5, -10, 10, -5, 1]
        >>> sequence(sqrt(5)*(((1 + sqrt(5))/2)**n - (-(1 + sqrt(5))/2)**(-n))/5).find_linear_recurrence(10)
        [1, 1]
        >>> sequence(x+y*(-2)**(-n), (n, 0, oo)).find_linear_recurrence(30)
        [1/2, 1/2]
        >>> sequence(3*5**n + 12).find_linear_recurrence(20,gfvar=x)
        ([6, -5], 3*(5 - 21*x)/((x - 1)*(5*x - 1)))
        >>> sequence(lucas(n)).find_linear_recurrence(15,gfvar=x)
        ([1, 1], (x - 2)/(x**2 + x - 1))
        """
        from sympy.simplify import simplify
        x = [simplify(expand(t)) for t in self[:n]]
        lx = len(x)
        if d is None:
            r = lx//2
        else:
            r = min(d,lx//2)
        coeffs = []
        for l in range(1, r+1):
            l2 = 2*l
            mlist = []
            for k in range(l):
                mlist.append(x[k:k+l])
            m = Matrix(mlist)
            if m.det() != 0:
                y = simplify(m.LUsolve(Matrix(x[l:l2])))
                if lx == l2:
                    coeffs = flatten(y[::-1])
                    break
                mlist = []
                for k in range(l,lx-l):
                    mlist.append(x[k:k+l])
                m = Matrix(mlist)
                if m*y == Matrix(x[l2:]):
                    coeffs = flatten(y[::-1])
                    break
        if gfvar is None:
            return coeffs
        else:
            l = len(coeffs)
            if l == 0:
                return [], None
            else:
                n, d = x[l-1]*gfvar**(l-1), 1 - coeffs[l-1]*gfvar**l
                for i in range(l-1):
                    n += x[i]*gfvar**i
                    for j in range(l-i-1):
                        n -= coeffs[i]*x[j]*gfvar**(i+j+1)
                    d -= coeffs[i]*gfvar**(i+1)
                return coeffs, simplify(factor(n)/factor(d))

class EmptySequence(SeqBase, metaclass=Singleton):
    """Represents an empty sequence.

    The empty sequence is also available as a singleton as
    ``S.EmptySequence``.

    Examples
    ========

    >>> from sympy import EmptySequence, SeqPer
    >>> from sympy.abc import x
    >>> EmptySequence
    EmptySequence
    >>> SeqPer((1, 2), (x, 0, 10)) + EmptySequence
    SeqPer((1, 2), (x, 0, 10))
    >>> SeqPer((1, 2)) * EmptySequence
    EmptySequence
    >>> EmptySequence.coeff_mul(-1)
    EmptySequence
    """

    @property
    def interval(self):
        return S.EmptySet

    @property
    def length(self):
        return S.Zero

    def coeff_mul(self, coeff):
        """See docstring of SeqBase.coeff_mul"""
        return self

    def __iter__(self):
        return iter([])


class SeqExpr(SeqBase):
    """Sequence expression class.

    Various sequences should inherit from this class.

    Examples
    ========

    >>> from sympy.series.sequences import SeqExpr
    >>> from sympy.abc import x
    >>> from sympy import Tuple
    >>> s = SeqExpr(Tuple(1, 2, 3), Tuple(x, 0, 10))
    >>> s.gen
    (1, 2, 3)
    >>> s.interval
    Interval(0, 10)
    >>> s.length
    11

    See Also
    ========

    sympy.series.sequences.SeqPer
    sympy.series.sequences.SeqFormula
    """

    @property
    def gen(self):
        return self.args[0]

    @property
    def interval(self):
        return Interval(self.args[1][1], self.args[1][2])

    @property
    def start(self):
        return self.interval.inf

    @property
    def stop(self):
        return self.interval.sup

    @property
    def length(self):
        return self.stop - self.start + 1

    @property
    def variables(self):
        return (self.args[1][0],)


class SeqPer(SeqExpr):
    """
    Represents a periodic sequence.

    The elements are repeated after a given period.

    Examples
    ========

    >>> from sympy import SeqPer, oo
    >>> from sympy.abc import k

    >>> s = SeqPer((1, 2, 3), (0, 5))
    >>> s.periodical
    (1, 2, 3)
    >>> s.period
    3

    For value at a particular point

    >>> s.coeff(3)
    1

    supports slicing

    >>> s[:]
    [1, 2, 3, 1, 2, 3]

    iterable

    >>> list(s)
    [1, 2, 3, 1, 2, 3]

    sequence starts from negative infinity

    >>> SeqPer((1, 2, 3), (-oo, 0))[0:6]
    [1, 2, 3, 1, 2, 3]

    Periodic formulas

    >>> SeqPer((k, k**2, k**3), (k, 0, oo))[0:6]
    [0, 1, 8, 3, 16, 125]

    See Also
    ========

    sympy.series.sequences.SeqFormula
    """

    def __new__(cls, periodical, limits=None):
        periodical = sympify(periodical)

        def _find_x(periodical):
            free = periodical.free_symbols
            if len(periodical.free_symbols) == 1:
                return free.pop()
            else:
                return Dummy('k')

        x, start, stop = None, None, None
        if limits is None:
            x, start, stop = _find_x(periodical), 0, S.Infinity
        if is_sequence(limits, Tuple):
            if len(limits) == 3:
                x, start, stop = limits
            elif len(limits) == 2:
                x = _find_x(periodical)
                start, stop = limits

        if not isinstance(x, (Symbol, Idx)) or start is None or stop is None:
            raise ValueError('Invalid limits given: %s' % str(limits))

        if start is S.NegativeInfinity and stop is S.Infinity:
                raise ValueError("Both the start and end value"
                                 "cannot be unbounded")

        limits = sympify((x, start, stop))

        if is_sequence(periodical, Tuple):
            periodical = sympify(tuple(flatten(periodical)))
        else:
            raise ValueError("invalid period %s should be something "
                             "like e.g (1, 2) " % periodical)

        if Interval(limits[1], limits[2]) is S.EmptySet:
            return S.EmptySequence

        return Basic.__new__(cls, periodical, limits)

    @property
    def period(self):
        return len(self.gen)

    @property
    def periodical(self):
        return self.gen

    def _eval_coeff(self, pt):
        if self.start is S.NegativeInfinity:
            idx = (self.stop - pt) % self.period
        else:
            idx = (pt - self.start) % self.period
        return self.periodical[idx].subs(self.variables[0], pt)

    def _add(self, other):
        """See docstring of SeqBase._add"""
        if isinstance(other, SeqPer):
            per1, lper1 = self.periodical, self.period
            per2, lper2 = other.periodical, other.period

            per_length = lcm(lper1, lper2)

            new_per = []
            for x in range(per_length):
                ele1 = per1[x % lper1]
                ele2 = per2[x % lper2]
                new_per.append(ele1 + ele2)

            start, stop = self._intersect_interval(other)
            return SeqPer(new_per, (self.variables[0], start, stop))

    def _mul(self, other):
        """See docstring of SeqBase._mul"""
        if isinstance(other, SeqPer):
            per1, lper1 = self.periodical, self.period
            per2, lper2 = other.periodical, other.period

            per_length = lcm(lper1, lper2)

            new_per = []
            for x in range(per_length):
                ele1 = per1[x % lper1]
                ele2 = per2[x % lper2]
                new_per.append(ele1 * ele2)

            start, stop = self._intersect_interval(other)
            return SeqPer(new_per, (self.variables[0], start, stop))

    def coeff_mul(self, coeff):
        """See docstring of SeqBase.coeff_mul"""
        coeff = sympify(coeff)
        per = [x * coeff for x in self.periodical]
        return SeqPer(per, self.args[1])


class SeqFormula(SeqExpr):
    """
    Represents sequence based on a formula.

    Elements are generated using a formula.

    Examples
    ========

    >>> from sympy import SeqFormula, oo, Symbol
    >>> n = Symbol('n')
    >>> s = SeqFormula(n**2, (n, 0, 5))
    >>> s.formula
    n**2

    For value at a particular point

    >>> s.coeff(3)
    9

    supports slicing

    >>> s[:]
    [0, 1, 4, 9, 16, 25]

    iterable

    >>> list(s)
    [0, 1, 4, 9, 16, 25]

    sequence starts from negative infinity

    >>> SeqFormula(n**2, (-oo, 0))[0:6]
    [0, 1, 4, 9, 16, 25]

    See Also
    ========

    sympy.series.sequences.SeqPer
    """

    def __new__(cls, formula, limits=None):
        formula = sympify(formula)

        def _find_x(formula):
            free = formula.free_symbols
            if len(free) == 1:
                return free.pop()
            elif not free:
                return Dummy('k')
            else:
                raise ValueError(
                    " specify dummy variables for %s. If the formula contains"
                    " more than one free symbol, a dummy variable should be"
                    " supplied explicitly e.g., SeqFormula(m*n**2, (n, 0, 5))"
                    % formula)

        x, start, stop = None, None, None
        if limits is None:
            x, start, stop = _find_x(formula), 0, S.Infinity
        if is_sequence(limits, Tuple):
            if len(limits) == 3:
                x, start, stop = limits
            elif len(limits) == 2:
                x = _find_x(formula)
                start, stop = limits

        if not isinstance(x, (Symbol, Idx)) or start is None or stop is None:
            raise ValueError('Invalid limits given: %s' % str(limits))

        if start is S.NegativeInfinity and stop is S.Infinity:
                raise ValueError("Both the start and end value "
                                 "cannot be unbounded")
        limits = sympify((x, start, stop))

        if Interval(limits[1], limits[2]) is S.EmptySet:
            return S.EmptySequence

        return Basic.__new__(cls, formula, limits)

    @property
    def formula(self):
        return self.gen

    def _eval_coeff(self, pt):
        d = self.variables[0]
        return self.formula.subs(d, pt)

    def _add(self, other):
        """See docstring of SeqBase._add"""
        if isinstance(other, SeqFormula):
            form1, v1 = self.formula, self.variables[0]
            form2, v2 = other.formula, other.variables[0]
            formula = form1 + form2.subs(v2, v1)
            start, stop = self._intersect_interval(other)
            return SeqFormula(formula, (v1, start, stop))

    def _mul(self, other):
        """See docstring of SeqBase._mul"""
        if isinstance(other, SeqFormula):
            form1, v1 = self.formula, self.variables[0]
            form2, v2 = other.formula, other.variables[0]
            formula = form1 * form2.subs(v2, v1)
            start, stop = self._intersect_interval(other)
            return SeqFormula(formula, (v1, start, stop))

    def coeff_mul(self, coeff):
        """See docstring of SeqBase.coeff_mul"""
        coeff = sympify(coeff)
        formula = self.formula * coeff
        return SeqFormula(formula, self.args[1])

    def expand(self, *args, **kwargs):
        return SeqFormula(expand(self.formula, *args, **kwargs), self.args[1])

class RecursiveSeq(SeqBase):
    """
    A finite degree recursive sequence.

    Explanation
    ===========

    That is, a sequence a(n) that depends on a fixed, finite number of its
    previous values. The general form is

        a(n) = f(a(n - 1), a(n - 2), ..., a(n - d))

    for some fixed, positive integer d, where f is some function defined by a
    SymPy expression.

    Parameters
    ==========

    recurrence : SymPy expression defining recurrence
        This is *not* an equality, only the expression that the nth term is
        equal to. For example, if :code:`a(n) = f(a(n - 1), ..., a(n - d))`,
        then the expression should be :code:`f(a(n - 1), ..., a(n - d))`.

    yn : applied undefined function
        Represents the nth term of the sequence as e.g. :code:`y(n)` where
        :code:`y` is an undefined function and `n` is the sequence index.

    n : symbolic argument
        The name of the variable that the recurrence is in, e.g., :code:`n` if
        the recurrence function is :code:`y(n)`.

    initial : iterable with length equal to the degree of the recurrence
        The initial values of the recurrence.

    start : start value of sequence (inclusive)

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> from sympy.series.sequences import RecursiveSeq
    >>> y = Function("y")
    >>> n = symbols("n")
    >>> fib = RecursiveSeq(y(n - 1) + y(n - 2), y(n), n, [0, 1])

    >>> fib.coeff(3) # Value at a particular point
    2

    >>> fib[:6] # supports slicing
    [0, 1, 1, 2, 3, 5]

    >>> fib.recurrence # inspect recurrence
    Eq(y(n), y(n - 2) + y(n - 1))

    >>> fib.degree # automatically determine degree
    2

    >>> for x in zip(range(10), fib): # supports iteration
    ...     print(x)
    (0, 0)
    (1, 1)
    (2, 1)
    (3, 2)
    (4, 3)
    (5, 5)
    (6, 8)
    (7, 13)
    (8, 21)
    (9, 34)

    See Also
    ========

    sympy.series.sequences.SeqFormula

    """

    def __new__(cls, recurrence, yn, n, initial=None, start=0):
        if not isinstance(yn, AppliedUndef):
            raise TypeError("recurrence sequence must be an applied undefined function"
                            ", found `{}`".format(yn))

        if not isinstance(n, Basic) or not n.is_symbol:
            raise TypeError("recurrence variable must be a symbol"
                            ", found `{}`".format(n))

        if yn.args != (n,):
            raise TypeError("recurrence sequence does not match symbol")

        y = yn.func

        k = Wild("k", exclude=(n,))
        degree = 0

        # Find all applications of y in the recurrence and check that:
        #   1. The function y is only being used with a single argument; and
        #   2. All arguments are n + k for constant negative integers k.

        prev_ys = recurrence.find(y)
        for prev_y in prev_ys:
            if len(prev_y.args) != 1:
                raise TypeError("Recurrence should be in a single variable")

            shift = prev_y.args[0].match(n + k)[k]
            if not (shift.is_constant() and shift.is_integer and shift < 0):
                raise TypeError("Recurrence should have constant,"
                                " negative, integer shifts"
                                " (found {})".format(prev_y))

            if -shift > degree:
                degree = -shift

        if not initial:
            initial = [Dummy("c_{}".format(k)) for k in range(degree)]

        if len(initial) != degree:
            raise ValueError("Number of initial terms must equal degree")

        degree = Integer(degree)
        start = sympify(start)

        initial = Tuple(*(sympify(x) for x in initial))

        seq = Basic.__new__(cls, recurrence, yn, n, initial, start)

        seq.cache = {y(start + k): init for k, init in enumerate(initial)}
        seq.degree = degree

        return seq

    @property
    def _recurrence(self):
        """Equation defining recurrence."""
        return self.args[0]

    @property
    def recurrence(self):
        """Equation defining recurrence."""
        return Eq(self.yn, self.args[0])

    @property
    def yn(self):
        """Applied function representing the nth term"""
        return self.args[1]

    @property
    def y(self):
        """Undefined function for the nth term of the sequence"""
        return self.yn.func

    @property
    def n(self):
        """Sequence index symbol"""
        return self.args[2]

    @property
    def initial(self):
        """The initial values of the sequence"""
        return self.args[3]

    @property
    def start(self):
        """The starting point of the sequence. This point is included"""
        return self.args[4]

    @property
    def stop(self):
        """The ending point of the sequence. (oo)"""
        return S.Infinity

    @property
    def interval(self):
        """Interval on which sequence is defined."""
        return (self.start, S.Infinity)

    def _eval_coeff(self, index):
        if index - self.start < len(self.cache):
            return self.cache[self.y(index)]

        for current in range(len(self.cache), index + 1):
            # Use xreplace over subs for performance.
            # See issue #10697.
            seq_index = self.start + current
            current_recurrence = self._recurrence.xreplace({self.n: seq_index})
            new_term = current_recurrence.xreplace(self.cache)

            self.cache[self.y(seq_index)] = new_term

        return self.cache[self.y(self.start + current)]

    def __iter__(self):
        index = self.start
        while True:
            yield self._eval_coeff(index)
            index += 1


def sequence(seq, limits=None):
    """
    Returns appropriate sequence object.

    Explanation
    ===========

    If ``seq`` is a SymPy sequence, returns :class:`SeqPer` object
    otherwise returns :class:`SeqFormula` object.

    Examples
    ========

    >>> from sympy import sequence
    >>> from sympy.abc import n
    >>> sequence(n**2, (n, 0, 5))
    SeqFormula(n**2, (n, 0, 5))
    >>> sequence((1, 2, 3), (n, 0, 5))
    SeqPer((1, 2, 3), (n, 0, 5))

    See Also
    ========

    sympy.series.sequences.SeqPer
    sympy.series.sequences.SeqFormula
    """
    seq = sympify(seq)

    if is_sequence(seq, Tuple):
        return SeqPer(seq, limits)
    else:
        return SeqFormula(seq, limits)


###############################################################################
#                            OPERATIONS                                       #
###############################################################################


class SeqExprOp(SeqBase):
    """
    Base class for operations on sequences.

    Examples
    ========

    >>> from sympy.series.sequences import SeqExprOp, sequence
    >>> from sympy.abc import n
    >>> s1 = sequence(n**2, (n, 0, 10))
    >>> s2 = sequence((1, 2, 3), (n, 5, 10))
    >>> s = SeqExprOp(s1, s2)
    >>> s.gen
    (n**2, (1, 2, 3))
    >>> s.interval
    Interval(5, 10)
    >>> s.length
    6

    See Also
    ========

    sympy.series.sequences.SeqAdd
    sympy.series.sequences.SeqMul
    """
    @property
    def gen(self):
        """Generator for the sequence.

        returns a tuple of generators of all the argument sequences.
        """
        return tuple(a.gen for a in self.args)

    @property
    def interval(self):
        """Sequence is defined on the intersection
        of all the intervals of respective sequences
        """
        return Intersection(*(a.interval for a in self.args))

    @property
    def start(self):
        return self.interval.inf

    @property
    def stop(self):
        return self.interval.sup

    @property
    def variables(self):
        """Cumulative of all the bound variables"""
        return tuple(flatten([a.variables for a in self.args]))

    @property
    def length(self):
        return self.stop - self.start + 1


class SeqAdd(SeqExprOp):
    """Represents term-wise addition of sequences.

    Rules:
        * The interval on which sequence is defined is the intersection
          of respective intervals of sequences.
        * Anything + :class:`EmptySequence` remains unchanged.
        * Other rules are defined in ``_add`` methods of sequence classes.

    Examples
    ========

    >>> from sympy import EmptySequence, oo, SeqAdd, SeqPer, SeqFormula
    >>> from sympy.abc import n
    >>> SeqAdd(SeqPer((1, 2), (n, 0, oo)), EmptySequence)
    SeqPer((1, 2), (n, 0, oo))
    >>> SeqAdd(SeqPer((1, 2), (n, 0, 5)), SeqPer((1, 2), (n, 6, 10)))
    EmptySequence
    >>> SeqAdd(SeqPer((1, 2), (n, 0, oo)), SeqFormula(n**2, (n, 0, oo)))
    SeqAdd(SeqFormula(n**2, (n, 0, oo)), SeqPer((1, 2), (n, 0, oo)))
    >>> SeqAdd(SeqFormula(n**3), SeqFormula(n**2))
    SeqFormula(n**3 + n**2, (n, 0, oo))

    See Also
    ========

    sympy.series.sequences.SeqMul
    """

    def __new__(cls, *args, **kwargs):
        evaluate = kwargs.get('evaluate', global_parameters.evaluate)

        # flatten inputs
        args = list(args)

        # adapted from sympy.sets.sets.Union
        def _flatten(arg):
            if isinstance(arg, SeqBase):
                if isinstance(arg, SeqAdd):
                    return sum(map(_flatten, arg.args), [])
                else:
                    return [arg]
            if iterable(arg):
                return sum(map(_flatten, arg), [])
            raise TypeError("Input must be Sequences or "
                            " iterables of Sequences")
        args = _flatten(args)

        args = [a for a in args if a is not S.EmptySequence]

        # Addition of no sequences is EmptySequence
        if not args:
            return S.EmptySequence

        if Intersection(*(a.interval for a in args)) is S.EmptySet:
            return S.EmptySequence

        # reduce using known rules
        if evaluate:
            return SeqAdd.reduce(args)

        args = list(ordered(args, SeqBase._start_key))

        return Basic.__new__(cls, *args)

    @staticmethod
    def reduce(args):
        """Simplify :class:`SeqAdd` using known rules.

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
                    new_seq = s._add(t)
                    # This returns None if s does not know how to add
                    # with t. Returns the newly added sequence otherwise
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
            return SeqAdd(args, evaluate=False)

    def _eval_coeff(self, pt):
        """adds up the coefficients of all the sequences at point pt"""
        return sum(a.coeff(pt) for a in self.args)


class SeqMul(SeqExprOp):
    r"""Represents term-wise multiplication of sequences.

    Explanation
    ===========

    Handles multiplication of sequences only. For multiplication
    with other objects see :func:`SeqBase.coeff_mul`.

    Rules:
        * The interval on which sequence is defined is the intersection
          of respective intervals of sequences.
        * Anything \* :class:`EmptySequence` returns :class:`EmptySequence`.
        * Other rules are defined in ``_mul`` methods of sequence classes.

    Examples
    ========

    >>> from sympy import EmptySequence, oo, SeqMul, SeqPer, SeqFormula
    >>> from sympy.abc import n
    >>> SeqMul(SeqPer((1, 2), (n, 0, oo)), EmptySequence)
    EmptySequence
    >>> SeqMul(SeqPer((1, 2), (n, 0, 5)), SeqPer((1, 2), (n, 6, 10)))
    EmptySequence
    >>> SeqMul(SeqPer((1, 2), (n, 0, oo)), SeqFormula(n**2))
    SeqMul(SeqFormula(n**2, (n, 0, oo)), SeqPer((1, 2), (n, 0, oo)))
    >>> SeqMul(SeqFormula(n**3), SeqFormula(n**2))
    SeqFormula(n**5, (n, 0, oo))

    See Also
    ========

    sympy.series.sequences.SeqAdd
    """

    def __new__(cls, *args, **kwargs):
        evaluate = kwargs.get('evaluate', global_parameters.evaluate)

        # flatten inputs
        args = list(args)

        # adapted from sympy.sets.sets.Union
        def _flatten(arg):
            if isinstance(arg, SeqBase):
                if isinstance(arg, SeqMul):
                    return sum(map(_flatten, arg.args), [])
                else:
                    return [arg]
            elif iterable(arg):
                return sum(map(_flatten, arg), [])
            raise TypeError("Input must be Sequences or "
                            " iterables of Sequences")
