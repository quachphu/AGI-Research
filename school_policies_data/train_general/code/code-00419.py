                expr.args [1].args [-1].exp is S.NegativeOne):

            expr2    = together (expr)
            expr2ops = count_ops_alg(expr2)[0]

            if expr2ops < exprops:
                expr       = expr2
                simplified = True

        else:
            simplified = True

    return (expr, simplified) if withsimp else expr


bottom_up = deprecated(
    """
    Using bottom_up from the sympy.simplify.simplify submodule is
    deprecated.

    Instead, use bottom_up from the top-level sympy namespace, like

        sympy.bottom_up
    """,
    deprecated_since_version="1.10",
    active_deprecations_target="deprecated-traversal-functions-moved",
)(_bottom_up)


# XXX: This function really should either be private API or exported in the
# top-level sympy/__init__.py
walk = deprecated(
    """
    Using walk from the sympy.simplify.simplify submodule is
    deprecated.

    Instead, use walk from sympy.core.traversal.walk
    """,
    deprecated_since_version="1.10",
    active_deprecations_target="deprecated-traversal-functions-moved",
)(_walk)
<|endoftext|>from sympy.core import Add, Expr, Mul, S, sympify
from sympy.core.function import _mexpand, count_ops, expand_mul
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy
from sympy.functions import root, sign, sqrt
from sympy.polys import Poly, PolynomialError


def is_sqrt(expr):
    """Return True if expr is a sqrt, otherwise False."""

    return expr.is_Pow and expr.exp.is_Rational and abs(expr.exp) is S.Half


def sqrt_depth(p) -> int:
    """Return the maximum depth of any square root argument of p.

    >>> from sympy.functions.elementary.miscellaneous import sqrt
    >>> from sympy.simplify.sqrtdenest import sqrt_depth

    Neither of these square roots contains any other square roots
    so the depth is 1:

    >>> sqrt_depth(1 + sqrt(2)*(1 + sqrt(3)))
    1

    The sqrt(3) is contained within a square root so the depth is
    2:

    >>> sqrt_depth(1 + sqrt(2)*sqrt(1 + sqrt(3)))
    2
    """
    if p is S.ImaginaryUnit:
        return 1
    if p.is_Atom:
        return 0
    if p.is_Add or p.is_Mul:
        return max(sqrt_depth(x) for x in p.args)
    if is_sqrt(p):
        return sqrt_depth(p.base) + 1
    return 0


def is_algebraic(p):
    """Return True if p is comprised of only Rationals or square roots
    of Rationals and algebraic operations.

    Examples
    ========

    >>> from sympy.functions.elementary.miscellaneous import sqrt
    >>> from sympy.simplify.sqrtdenest import is_algebraic
    >>> from sympy import cos
    >>> is_algebraic(sqrt(2)*(3/(sqrt(7) + sqrt(5)*sqrt(2))))
    True
    >>> is_algebraic(sqrt(2)*(3/(sqrt(7) + sqrt(5)*cos(2))))
    False
    """

    if p.is_Rational:
        return True
    elif p.is_Atom:
        return False
    elif is_sqrt(p) or p.is_Pow and p.exp.is_Integer:
        return is_algebraic(p.base)
    elif p.is_Add or p.is_Mul:
        return all(is_algebraic(x) for x in p.args)
    else:
        return False


def _subsets(n):
    """
    Returns all possible subsets of the set (0, 1, ..., n-1) except the
    empty set, listed in reversed lexicographical order according to binary
    representation, so that the case of the fourth root is treated last.

    Examples
    ========

    >>> from sympy.simplify.sqrtdenest import _subsets
    >>> _subsets(2)
    [[1, 0], [0, 1], [1, 1]]

    """
    if n == 1:
        a = [[1]]
    elif n == 2:
        a = [[1, 0], [0, 1], [1, 1]]
    elif n == 3:
        a = [[1, 0, 0], [0, 1, 0], [1, 1, 0],
             [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    else:
        b = _subsets(n - 1)
        a0 = [x + [0] for x in b]
        a1 = [x + [1] for x in b]
        a = a0 + [[0]*(n - 1) + [1]] + a1
    return a


def sqrtdenest(expr, max_iter=3):
    """Denests sqrts in an expression that contain other square roots
    if possible, otherwise returns the expr unchanged. This is based on the
    algorithms of [1].

    Examples
    ========

    >>> from sympy.simplify.sqrtdenest import sqrtdenest
    >>> from sympy import sqrt
    >>> sqrtdenest(sqrt(5 + 2 * sqrt(6)))
    sqrt(2) + sqrt(3)

    See Also
    ========

    sympy.solvers.solvers.unrad

    References
    ==========

    .. [1] https://web.archive.org/web/20210806201615/https://researcher.watson.ibm.com/researcher/files/us-fagin/symb85.pdf

    .. [2] D. J. Jeffrey and A. D. Rich, 'Symplifying Square Roots of Square Roots
           by Denesting' (available at https://www.cybertester.com/data/denest.pdf)

    """
    expr = expand_mul(expr)
    for i in range(max_iter):
        z = _sqrtdenest0(expr)
        if expr == z:
            return expr
        expr = z
    return expr


def _sqrt_match(p):
    """Return [a, b, r] for p.match(a + b*sqrt(r)) where, in addition to
    matching, sqrt(r) also has then maximal sqrt_depth among addends of p.

    Examples
    ========

    >>> from sympy.functions.elementary.miscellaneous import sqrt
    >>> from sympy.simplify.sqrtdenest import _sqrt_match
    >>> _sqrt_match(1 + sqrt(2) + sqrt(2)*sqrt(3) +  2*sqrt(1+sqrt(5)))
    [1 + sqrt(2) + sqrt(6), 2, 1 + sqrt(5)]
    """
    from sympy.simplify.radsimp import split_surds

    p = _mexpand(p)
    if p.is_Number:
        res = (p, S.Zero, S.Zero)
    elif p.is_Add:
        pargs = sorted(p.args, key=default_sort_key)
        sqargs = [x**2 for x in pargs]
        if all(sq.is_Rational and sq.is_positive for sq in sqargs):
            r, b, a = split_surds(p)
            res = a, b, r
            return list(res)
        # to make the process canonical, the argument is included in the tuple
        # so when the max is selected, it will be the largest arg having a
        # given depth
        v = [(sqrt_depth(x), x, i) for i, x in enumerate(pargs)]
        nmax = max(v, key=default_sort_key)
        if nmax[0] == 0:
            res = []
        else:
            # select r
            depth, _, i = nmax
            r = pargs.pop(i)
            v.pop(i)
            b = S.One
            if r.is_Mul:
                bv = []
                rv = []
                for x in r.args:
                    if sqrt_depth(x) < depth:
                        bv.append(x)
                    else:
                        rv.append(x)
                b = Mul._from_args(bv)
                r = Mul._from_args(rv)
            # collect terms containing r
            a1 = []
            b1 = [b]
            for x in v:
                if x[0] < depth:
                    a1.append(x[1])
                else:
                    x1 = x[1]
                    if x1 == r:
                        b1.append(1)
                    else:
                        if x1.is_Mul:
                            x1args = list(x1.args)
                            if r in x1args:
                                x1args.remove(r)
                                b1.append(Mul(*x1args))
                            else:
                                a1.append(x[1])
                        else:
                            a1.append(x[1])
            a = Add(*a1)
            b = Add(*b1)
            res = (a, b, r**2)
    else:
        b, r = p.as_coeff_Mul()
        if is_sqrt(r):
            res = (S.Zero, b, r**2)
        else:
            res = []
    return list(res)


class SqrtdenestStopIteration(StopIteration):
    pass


def _sqrtdenest0(expr):
    """Returns expr after denesting its arguments."""

    if is_sqrt(expr):
        n, d = expr.as_numer_denom()
        if d is S.One:  # n is a square root
            if n.base.is_Add:
                args = sorted(n.base.args, key=default_sort_key)
                if len(args) > 2 and all((x**2).is_Integer for x in args):
                    try:
                        return _sqrtdenest_rec(n)
                    except SqrtdenestStopIteration:
                        pass
                expr = sqrt(_mexpand(Add(*[_sqrtdenest0(x) for x in args])))
            return _sqrtdenest1(expr)
        else:
            n, d = [_sqrtdenest0(i) for i in (n, d)]
            return n/d

    if isinstance(expr, Add):
        cs = []
        args = []
        for arg in expr.args:
            c, a = arg.as_coeff_Mul()
            cs.append(c)
            args.append(a)

        if all(c.is_Rational for c in cs) and all(is_sqrt(arg) for arg in args):
            return _sqrt_ratcomb(cs, args)

    if isinstance(expr, Expr):
        args = expr.args
        if args:
            return expr.func(*[_sqrtdenest0(a) for a in args])
    return expr


def _sqrtdenest_rec(expr):
    """Helper that denests the square root of three or more surds.

    Explanation
    ===========

    It returns the denested expression; if it cannot be denested it
    throws SqrtdenestStopIteration

    Algorithm: expr.base is in the extension Q_m = Q(sqrt(r_1),..,sqrt(r_k));
    split expr.base = a + b*sqrt(r_k), where `a` and `b` are on
    Q_(m-1) = Q(sqrt(r_1),..,sqrt(r_(k-1))); then a**2 - b**2*r_k is
    on Q_(m-1); denest sqrt(a**2 - b**2*r_k) and so on.
    See [1], section 6.

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.simplify.sqrtdenest import _sqrtdenest_rec
    >>> _sqrtdenest_rec(sqrt(-72*sqrt(2) + 158*sqrt(5) + 498))
    -sqrt(10) + sqrt(2) + 9 + 9*sqrt(5)
    >>> w=-6*sqrt(55)-6*sqrt(35)-2*sqrt(22)-2*sqrt(14)+2*sqrt(77)+6*sqrt(10)+65
    >>> _sqrtdenest_rec(sqrt(w))
    -sqrt(11) - sqrt(7) + sqrt(2) + 3*sqrt(5)
    """
    from sympy.simplify.radsimp import radsimp, rad_rationalize, split_surds
    if not expr.is_Pow:
        return sqrtdenest(expr)
    if expr.base < 0:
        return sqrt(-1)*_sqrtdenest_rec(sqrt(-expr.base))
    g, a, b = split_surds(expr.base)
    a = a*sqrt(g)
    if a < b:
        a, b = b, a
    c2 = _mexpand(a**2 - b**2)
    if len(c2.args) > 2:
        g, a1, b1 = split_surds(c2)
        a1 = a1*sqrt(g)
        if a1 < b1:
            a1, b1 = b1, a1
        c2_1 = _mexpand(a1**2 - b1**2)
        c_1 = _sqrtdenest_rec(sqrt(c2_1))
        d_1 = _sqrtdenest_rec(sqrt(a1 + c_1))
        num, den = rad_rationalize(b1, d_1)
        c = _mexpand(d_1/sqrt(2) + num/(den*sqrt(2)))
    else:
        c = _sqrtdenest1(sqrt(c2))

    if sqrt_depth(c) > 1:
        raise SqrtdenestStopIteration
    ac = a + c
    if len(ac.args) >= len(expr.args):
        if count_ops(ac) >= count_ops(expr.base):
            raise SqrtdenestStopIteration
    d = sqrtdenest(sqrt(ac))
    if sqrt_depth(d) > 1:
        raise SqrtdenestStopIteration
    num, den = rad_rationalize(b, d)
    r = d/sqrt(2) + num/(den*sqrt(2))
    r = radsimp(r)
    return _mexpand(r)


def _sqrtdenest1(expr, denester=True):
    """Return denested expr after denesting with simpler methods or, that
    failing, using the denester."""

    from sympy.simplify.simplify import radsimp

    if not is_sqrt(expr):
        return expr

    a = expr.base
    if a.is_Atom:
        return expr
    val = _sqrt_match(a)
    if not val:
        return expr

    a, b, r = val
    # try a quick numeric denesting
    d2 = _mexpand(a**2 - b**2*r)
    if d2.is_Rational:
        if d2.is_positive:
            z = _sqrt_numeric_denest(a, b, r, d2)
            if z is not None:
                return z
        else:
            # fourth root case
            # sqrtdenest(sqrt(3 + 2*sqrt(3))) =
            # sqrt(2)*3**(1/4)/2 + sqrt(2)*3**(3/4)/2
            dr2 = _mexpand(-d2*r)
            dr = sqrt(dr2)
            if dr.is_Rational:
                z = _sqrt_numeric_denest(_mexpand(b*r), a, r, dr2)
                if z is not None:
                    return z/root(r, 4)

    else:
        z = _sqrt_symbolic_denest(a, b, r)
        if z is not None:
            return z

    if not denester or not is_algebraic(expr):
        return expr

    res = sqrt_biquadratic_denest(expr, a, b, r, d2)
    if res:
        return res

    # now call to the denester
    av0 = [a, b, r, d2]
    z = _denester([radsimp(expr**2)], av0, 0, sqrt_depth(expr))[0]
    if av0[1] is None:
        return expr
    if z is not None:
        if sqrt_depth(z) == sqrt_depth(expr) and count_ops(z) > count_ops(expr):
            return expr
        return z
    return expr


def _sqrt_symbolic_denest(a, b, r):
    """Given an expression, sqrt(a + b*sqrt(b)), return the denested
    expression or None.

    Explanation
    ===========

    If r = ra + rb*sqrt(rr), try replacing sqrt(rr) in ``a`` with
    (y**2 - ra)/rb, and if the result is a quadratic, ca*y**2 + cb*y + cc, and
    (cb + b)**2 - 4*ca*cc is 0, then sqrt(a + b*sqrt(r)) can be rewritten as
    sqrt(ca*(sqrt(r) + (cb + b)/(2*ca))**2).

    Examples
    ========

    >>> from sympy.simplify.sqrtdenest import _sqrt_symbolic_denest, sqrtdenest
    >>> from sympy import sqrt, Symbol
    >>> from sympy.abc import x

    >>> a, b, r = 16 - 2*sqrt(29), 2, -10*sqrt(29) + 55
    >>> _sqrt_symbolic_denest(a, b, r)
    sqrt(11 - 2*sqrt(29)) + sqrt(5)

    If the expression is numeric, it will be simplified:

    >>> w = sqrt(sqrt(sqrt(3) + 1) + 1) + 1 + sqrt(2)
    >>> sqrtdenest(sqrt((w**2).expand()))
    1 + sqrt(2) + sqrt(1 + sqrt(1 + sqrt(3)))

    Otherwise, it will only be simplified if assumptions allow:

    >>> w = w.subs(sqrt(3), sqrt(x + 3))
    >>> sqrtdenest(sqrt((w**2).expand()))
    sqrt((sqrt(sqrt(sqrt(x + 3) + 1) + 1) + 1 + sqrt(2))**2)

    Notice that the argument of the sqrt is a square. If x is made positive
    then the sqrt of the square is resolved:

    >>> _.subs(x, Symbol('x', positive=True))
    sqrt(sqrt(sqrt(x + 3) + 1) + 1) + 1 + sqrt(2)
    """

    a, b, r = map(sympify, (a, b, r))
    rval = _sqrt_match(r)
    if not rval:
        return None
    ra, rb, rr = rval
    if rb:
        y = Dummy('y', positive=True)
        try:
            newa = Poly(a.subs(sqrt(rr), (y**2 - ra)/rb), y)
        except PolynomialError:
            return None
        if newa.degree() == 2:
            ca, cb, cc = newa.all_coeffs()
            cb += b
            if _mexpand(cb**2 - 4*ca*cc).equals(0):
                z = sqrt(ca*(sqrt(r) + cb/(2*ca))**2)
                if z.is_number:
                    z = _mexpand(Mul._from_args(z.as_content_primitive()))
                return z


def _sqrt_numeric_denest(a, b, r, d2):
    r"""Helper that denest
    $\sqrt{a + b \sqrt{r}}, d^2 = a^2 - b^2 r > 0$

    If it cannot be denested, it returns ``None``.
    """
    d = sqrt(d2)
    s = a + d
    # sqrt_depth(res) <= sqrt_depth(s) + 1
    # sqrt_depth(expr) = sqrt_depth(r) + 2
    # there is denesting if sqrt_depth(s) + 1 < sqrt_depth(r) + 2
    # if s**2 is Number there is a fourth root
    if sqrt_depth(s) < sqrt_depth(r) + 1 or (s**2).is_Rational:
        s1, s2 = sign(s), sign(b)
        if s1 == s2 == -1:
            s1 = s2 = 1
        res = (s1 * sqrt(a + d) + s2 * sqrt(a - d)) * sqrt(2) / 2
        return res.expand()


def sqrt_biquadratic_denest(expr, a, b, r, d2):
    """denest expr = sqrt(a + b*sqrt(r))
    where a, b, r are linear combinations of square roots of
    positive rationals on the rationals (SQRR) and r > 0, b != 0,
    d2 = a**2 - b**2*r > 0

    If it cannot denest it returns None.

    Explanation
    ===========

    Search for a solution A of type SQRR of the biquadratic equation
    4*A**4 - 4*a*A**2 + b**2*r = 0                               (1)
    sqd = sqrt(a**2 - b**2*r)
    Choosing the sqrt to be positive, the possible solutions are
    A = sqrt(a/2 +/- sqd/2)
    Since a, b, r are SQRR, then a**2 - b**2*r is a SQRR,
    so if sqd can be denested, it is done by
    _sqrtdenest_rec, and the result is a SQRR.
    Similarly for A.
    Examples of solutions (in both cases a and sqd are positive):

      Example of expr with solution sqrt(a/2 + sqd/2) but not
      solution sqrt(a/2 - sqd/2):
      expr = sqrt(-sqrt(15) - sqrt(2)*sqrt(-sqrt(5) + 5) - sqrt(3) + 8)
      a = -sqrt(15) - sqrt(3) + 8; sqd = -2*sqrt(5) - 2 + 4*sqrt(3)

      Example of expr with solution sqrt(a/2 - sqd/2) but not
      solution sqrt(a/2 + sqd/2):
      w = 2 + r2 + r3 + (1 + r3)*sqrt(2 + r2 + 5*r3)
      expr = sqrt((w**2).expand())
      a = 4*sqrt(6) + 8*sqrt(2) + 47 + 28*sqrt(3)
      sqd = 29 + 20*sqrt(3)

    Define B = b/2*A; eq.(1) implies a = A**2 + B**2*r; then
    expr**2 = a + b*sqrt(r) = (A + B*sqrt(r))**2

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.simplify.sqrtdenest import _sqrt_match, sqrt_biquadratic_denest
    >>> z = sqrt((2*sqrt(2) + 4)*sqrt(2 + sqrt(2)) + 5*sqrt(2) + 8)
    >>> a, b, r = _sqrt_match(z**2)
    >>> d2 = a**2 - b**2*r
    >>> sqrt_biquadratic_denest(z, a, b, r, d2)
    sqrt(2) + sqrt(sqrt(2) + 2) + 2
    """
    from sympy.simplify.radsimp import radsimp, rad_rationalize
    if r <= 0 or d2 < 0 or not b or sqrt_depth(expr.base) < 2:
        return None
    for x in (a, b, r):
        for y in x.args:
            y2 = y**2
            if not y2.is_Integer or not y2.is_positive:
                return None
    sqd = _mexpand(sqrtdenest(sqrt(radsimp(d2))))
    if sqrt_depth(sqd) > 1:
        return None
    x1, x2 = [a/2 + sqd/2, a/2 - sqd/2]
    # look for a solution A with depth 1
    for x in (x1, x2):
        A = sqrtdenest(sqrt(x))
        if sqrt_depth(A) > 1:
            continue
        Bn, Bd = rad_rationalize(b, _mexpand(2*A))
        B = Bn/Bd
        z = A + B*sqrt(r)
        if z < 0:
            z = -z
        return _mexpand(z)
    return None


def _denester(nested, av0, h, max_depth_level):
    """Denests a list of expressions that contain nested square roots.

    Explanation
    ===========

    Algorithm based on <http://www.almaden.ibm.com/cs/people/fagin/symb85.pdf>.

    It is assumed that all of the elements of 'nested' share the same
    bottom-level radicand. (This is stated in the paper, on page 177, in
    the paragraph immediately preceding the algorithm.)

    When evaluating all of the arguments in parallel, the bottom-level
    radicand only needs to be denested once. This means that calling
    _denester with x arguments results in a recursive invocation with x+1
    arguments; hence _denester has polynomial complexity.

    However, if the arguments were evaluated separately, each call would
    result in two recursive invocations, and the algorithm would have
    exponential complexity.

    This is discussed in the paper in the middle paragraph of page 179.
    """
    from sympy.simplify.simplify import radsimp
    if h > max_depth_level:
        return None, None
    if av0[1] is None:
        return None, None
    if (av0[0] is None and
            all(n.is_Number for n in nested)):  # no arguments are nested
        for f in _subsets(len(nested)):  # test subset 'f' of nested
            p = _mexpand(Mul(*[nested[i] for i in range(len(f)) if f[i]]))
            if f.count(1) > 1 and f[-1]:
                p = -p
            sqp = sqrt(p)
            if sqp.is_Rational:
                return sqp, f  # got a perfect square so return its square root.
        # Otherwise, return the radicand from the previous invocation.
        return sqrt(nested[-1]), [0]*len(nested)
    else:
        R = None
        if av0[0] is not None:
            values = [av0[:2]]
            R = av0[2]
            nested2 = [av0[3], R]
            av0[0] = None
        else:
            values = list(filter(None, [_sqrt_match(expr) for expr in nested]))
            for v in values:
                if v[2]:  # Since if b=0, r is not defined
                    if R is not None:
                        if R != v[2]:
                            av0[1] = None
                            return None, None
                    else:
                        R = v[2]
            if R is None:
                # return the radicand from the previous invocation
                return sqrt(nested[-1]), [0]*len(nested)
            nested2 = [_mexpand(v[0]**2) -
                       _mexpand(R*v[1]**2) for v in values] + [R]
        d, f = _denester(nested2, av0, h + 1, max_depth_level)
        if not f:
            return None, None
        if not any(f[i] for i in range(len(nested))):
            v = values[-1]
            return sqrt(v[0] + _mexpand(v[1]*d)), f
        else:
            p = Mul(*[nested[i] for i in range(len(nested)) if f[i]])
            v = _sqrt_match(p)
            if 1 in f and f.index(1) < len(nested) - 1 and f[len(nested) - 1]:
                v[0] = -v[0]
                v[1] = -v[1]
            if not f[len(nested)]:  # Solution denests with square roots
                vad = _mexpand(v[0] + d)
                if vad <= 0:
                    # return the radicand from the previous invocation.
                    return sqrt(nested[-1]), [0]*len(nested)
                if not(sqrt_depth(vad) <= sqrt_depth(R) + 1 or
                       (vad**2).is_Number):
                    av0[1] = None
                    return None, None

                sqvad = _sqrtdenest1(sqrt(vad), denester=False)
                if not (sqrt_depth(sqvad) <= sqrt_depth(R) + 1):
                    av0[1] = None
                    return None, None
                sqvad1 = radsimp(1/sqvad)
                res = _mexpand(sqvad/sqrt(2) + (v[1]*sqrt(R)*sqvad1/sqrt(2)))
                return res, f

                      #          sign(v[1])*sqrt(_mexpand(v[1]**2*R*vad1/2))), f
            else:  # Solution requires a fourth root
                s2 = _mexpand(v[1]*R) + d
                if s2 <= 0:
                    return sqrt(nested[-1]), [0]*len(nested)
                FR, s = root(_mexpand(R), 4), sqrt(s2)
                return _mexpand(s/(sqrt(2)*FR) + v[0]*FR/(sqrt(2)*s)), f


def _sqrt_ratcomb(cs, args):
    """Denest rational combinations of radicals.

    Based on section 5 of [1].

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.simplify.sqrtdenest import sqrtdenest
    >>> z = sqrt(1+sqrt(3)) + sqrt(3+3*sqrt(3)) - sqrt(10+6*sqrt(3))
    >>> sqrtdenest(z)
    0
    """
    from sympy.simplify.radsimp import radsimp

    # check if there exists a pair of sqrt that can be denested
    def find(a):
        n = len(a)
        for i in range(n - 1):
            for j in range(i + 1, n):
                s1 = a[i].base
                s2 = a[j].base
                p = _mexpand(s1 * s2)
                s = sqrtdenest(sqrt(p))
                if s != sqrt(p):
                    return s, i, j

    indices = find(args)
    if indices is None:
        return Add(*[c * arg for c, arg in zip(cs, args)])

    s, i1, i2 = indices

    c2 = cs.pop(i2)
    args.pop(i2)
    a1 = args[i1]

    # replace a2 by s/a1
    cs[i1] += radsimp(c2 * s / a1.base)

    return _sqrt_ratcomb(cs, args)
<|endoftext|>from collections import defaultdict
from functools import reduce
from math import prod

from sympy.core.function import expand_log, count_ops, _coeff_isneg
from sympy.core import sympify, Basic, Dummy, S, Add, Mul, Pow, expand_mul, factor_terms
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.numbers import Integer, Rational, equal_valued
from sympy.core.mul import _keep_coeff
from sympy.core.rules import Transform
from sympy.functions import exp_polar, exp, log, root, polarify, unpolarify
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.polys import lcm, gcd
from sympy.ntheory.factor_ import multiplicity



def powsimp(expr, deep=False, combine='all', force=False, measure=count_ops):
    """
    Reduce expression by combining powers with similar bases and exponents.

    Explanation
    ===========

    If ``deep`` is ``True`` then powsimp() will also simplify arguments of
    functions. By default ``deep`` is set to ``False``.

    If ``force`` is ``True`` then bases will be combined without checking for
    assumptions, e.g. sqrt(x)*sqrt(y) -> sqrt(x*y) which is not true
    if x and y are both negative.

    You can make powsimp() only combine bases or only combine exponents by
    changing combine='base' or combine='exp'.  By default, combine='all',
    which does both.  combine='base' will only combine::

         a   a          a                          2x      x
        x * y  =>  (x*y)   as well as things like 2   =>  4

    and combine='exp' will only combine
    ::

         a   b      (a + b)
        x * x  =>  x

    combine='exp' will strictly only combine exponents in the way that used
    to be automatic.  Also use deep=True if you need the old behavior.

    When combine='all', 'exp' is evaluated first.  Consider the first
    example below for when there could be an ambiguity relating to this.
    This is done so things like the second example can be completely
    combined.  If you want 'base' combined first, do something like
    powsimp(powsimp(expr, combine='base'), combine='exp').

    Examples
    ========

    >>> from sympy import powsimp, exp, log, symbols
    >>> from sympy.abc import x, y, z, n
    >>> powsimp(x**y*x**z*y**z, combine='all')
    x**(y + z)*y**z
    >>> powsimp(x**y*x**z*y**z, combine='exp')
    x**(y + z)*y**z
    >>> powsimp(x**y*x**z*y**z, combine='base', force=True)
    x**y*(x*y)**z

    >>> powsimp(x**z*x**y*n**z*n**y, combine='all', force=True)
    (n*x)**(y + z)
    >>> powsimp(x**z*x**y*n**z*n**y, combine='exp')
    n**(y + z)*x**(y + z)
    >>> powsimp(x**z*x**y*n**z*n**y, combine='base', force=True)
    (n*x)**y*(n*x)**z

    >>> x, y = symbols('x y', positive=True)
    >>> powsimp(log(exp(x)*exp(y)))
    log(exp(x)*exp(y))
    >>> powsimp(log(exp(x)*exp(y)), deep=True)
    x + y

    Radicals with Mul bases will be combined if combine='exp'

    >>> from sympy import sqrt
    >>> x, y = symbols('x y')

    Two radicals are automatically joined through Mul:

    >>> a=sqrt(x*sqrt(y))
    >>> a*a**3 == a**4
    True

    But if an integer power of that radical has been
    autoexpanded then Mul does not join the resulting factors:

    >>> a**4 # auto expands to a Mul, no longer a Pow
    x**2*y
    >>> _*a # so Mul doesn't combine them
    x**2*y*sqrt(x*sqrt(y))
    >>> powsimp(_) # but powsimp will
    (x*sqrt(y))**(5/2)
    >>> powsimp(x*y*a) # but won't when doing so would violate assumptions
    x*y*sqrt(x*sqrt(y))

    """
    def recurse(arg, **kwargs):
        _deep = kwargs.get('deep', deep)
        _combine = kwargs.get('combine', combine)
        _force = kwargs.get('force', force)
        _measure = kwargs.get('measure', measure)
        return powsimp(arg, _deep, _combine, _force, _measure)

    expr = sympify(expr)

    if (not isinstance(expr, Basic) or isinstance(expr, MatrixSymbol) or (
            expr.is_Atom or expr in (exp_polar(0), exp_polar(1)))):
        return expr

    if deep or expr.is_Add or expr.is_Mul and _y not in expr.args:
        expr = expr.func(*[recurse(w) for w in expr.args])

    if expr.is_Pow:
        return recurse(expr*_y, deep=False)/_y

    if not expr.is_Mul:
        return expr

    # handle the Mul
    if combine in ('exp', 'all'):
        # Collect base/exp data, while maintaining order in the
        # non-commutative parts of the product
        c_powers = defaultdict(list)
        nc_part = []
        newexpr = []
        coeff = S.One
        for term in expr.args:
            if term.is_Rational:
                coeff *= term
                continue
            if term.is_Pow:
                term = _denest_pow(term)
            if term.is_commutative:
                b, e = term.as_base_exp()
                if deep:
                    b, e = [recurse(i) for i in [b, e]]
                if b.is_Pow or isinstance(b, exp):
                    # don't let smthg like sqrt(x**a) split into x**a, 1/2
                    # or else it will be joined as x**(a/2) later
                    b, e = b**e, S.One
                c_powers[b].append(e)
            else:
                # This is the logic that combines exponents for equal,
                # but non-commutative bases: A**x*A**y == A**(x+y).
                if nc_part:
                    b1, e1 = nc_part[-1].as_base_exp()
                    b2, e2 = term.as_base_exp()
                    if (b1 == b2 and
                            e1.is_commutative and e2.is_commutative):
                        nc_part[-1] = Pow(b1, Add(e1, e2))
                        continue
                nc_part.append(term)

        # add up exponents of common bases
        for b, e in ordered(iter(c_powers.items())):
            # allow 2**x/4 -> 2**(x - 2); don't do this when b and e are
            # Numbers since autoevaluation will undo it, e.g.
            # 2**(1/3)/4 -> 2**(1/3 - 2) -> 2**(1/3)/4
            if (b and b.is_Rational and not all(ei.is_Number for ei in e) and \
                    coeff is not S.One and
                    b not in (S.One, S.NegativeOne)):
                m = multiplicity(abs(b), abs(coeff))
                if m:
                    e.append(m)
                    coeff /= b**m
            c_powers[b] = Add(*e)
        if coeff is not S.One:
            if coeff in c_powers:
                c_powers[coeff] += S.One
            else:
                c_powers[coeff] = S.One

        # convert to plain dictionary
        c_powers = dict(c_powers)

        # check for base and inverted base pairs
        be = list(c_powers.items())
        skip = set()  # skip if we already saw them
        for b, e in be:
            if b in skip:
                continue
            bpos = b.is_positive or b.is_polar
            if bpos:
                binv = 1/b
                #Special case for float 1
                if b.is_Float and equal_valued(b, 1):
                    c_powers[b] = S.One
                    continue
                if b != binv and binv in c_powers:
                    if b.as_numer_denom()[0] is S.One:
                        c_powers.pop(b)
                        c_powers[binv] -= e
                    else:
                        skip.add(binv)
                        e = c_powers.pop(binv)
                        c_powers[b] -= e

        # check for base and negated base pairs
        be = list(c_powers.items())
        _n = S.NegativeOne
        for b, e in be:
            if (b.is_Symbol or b.is_Add) and -b in c_powers and b in c_powers:
                if (b.is_positive is not None or e.is_integer):
                    if e.is_integer or b.is_negative:
                        c_powers[-b] += c_powers.pop(b)
                    else:  # (-b).is_positive so use its e
                        e = c_powers.pop(-b)
                        c_powers[b] += e
                    if _n in c_powers:
                        c_powers[_n] += e
                    else:
                        c_powers[_n] = e

        # filter c_powers and convert to a list
        c_powers = [(b, e) for b, e in c_powers.items() if e]

        # ==============================================================
        # check for Mul bases of Rational powers that can be combined with
        # separated bases, e.g. x*sqrt(x*y)*sqrt(x*sqrt(x*y)) ->
        # (x*sqrt(x*y))**(3/2)
        # ---------------- helper functions

        def ratq(x):
            '''Return Rational part of x's exponent as it appears in the bkey.
            '''
            return bkey(x)[0][1]

        def bkey(b, e=None):
            '''Return (b**s, c.q), c.p where e -> c*s. If e is not given then
            it will be taken by using as_base_exp() on the input b.
            e.g.
                x**3/2 -> (x, 2), 3
                x**y -> (x**y, 1), 1
                x**(2*y/3) -> (x**y, 3), 2
                exp(x/2) -> (exp(a), 2), 1

            '''
            if e is not None:  # coming from c_powers or from below
                if e.is_Integer:
                    return (b, S.One), e
                elif e.is_Rational:
                    return (b, Integer(e.q)), Integer(e.p)
                else:
                    c, m = e.as_coeff_Mul(rational=True)
                    if c is not S.One:
                        if m.is_integer:
                            return (b, Integer(c.q)), m*Integer(c.p)
                        return (b**m, Integer(c.q)), Integer(c.p)
                    else:
                        return (b**e, S.One), S.One
            else:
                return bkey(*b.as_base_exp())

        def update(b):
            '''Decide what to do with base, b. If its exponent is now an
            integer multiple of the Rational denominator, then remove it
            and put the factors of its base in the common_b dictionary or
            update the existing bases if necessary. If it has been zeroed
            out, simply remove the base.
            '''
            newe, r = divmod(common_b[b], b[1])
            if not r:
                common_b.pop(b)
                if newe:
                    for m in Mul.make_args(b[0]**newe):
                        b, e = bkey(m)
                        if b not in common_b:
                            common_b[b] = 0
                        common_b[b] += e
                        if b[1] != 1:
                            bases.append(b)
        # ---------------- end of helper functions

        # assemble a dictionary of the factors having a Rational power
        common_b = {}
        done = []
        bases = []
        for b, e in c_powers:
            b, e = bkey(b, e)
            if b in common_b:
                common_b[b] = common_b[b] + e
            else:
                common_b[b] = e
            if b[1] != 1 and b[0].is_Mul:
                bases.append(b)
        bases.sort(key=default_sort_key)  # this makes tie-breaking canonical
        bases.sort(key=measure, reverse=True)  # handle longest first
        for base in bases:
            if base not in common_b:  # it may have been removed already
                continue
            b, exponent = base
            last = False  # True when no factor of base is a radical
            qlcm = 1  # the lcm of the radical denominators
            while True:
                bstart = b
                qstart = qlcm

                bb = []  # list of factors
                ee = []  # (factor's expo. and it's current value in common_b)
                for bi in Mul.make_args(b):
                    bib, bie = bkey(bi)
                    if bib not in common_b or common_b[bib] < bie:
                        ee = bb = []  # failed
                        break
                    ee.append([bie, common_b[bib]])
                    bb.append(bib)
                if ee:
                    # find the number of integral extractions possible
                    # e.g. [(1, 2), (2, 2)] -> min(2/1, 2/2) -> 1
                    min1 = ee[0][1]//ee[0][0]
                    for i in range(1, len(ee)):
                        rat = ee[i][1]//ee[i][0]
                        if rat < 1:
                            break
                        min1 = min(min1, rat)
                    else:
                        # update base factor counts
                        # e.g. if ee = [(2, 5), (3, 6)] then min1 = 2
                        # and the new base counts will be 5-2*2 and 6-2*3
                        for i in range(len(bb)):
                            common_b[bb[i]] -= min1*ee[i][0]
                            update(bb[i])
                        # update the count of the base
                        # e.g. x**2*y*sqrt(x*sqrt(y)) the count of x*sqrt(y)
                        # will increase by 4 to give bkey (x*sqrt(y), 2, 5)
                        common_b[base] += min1*qstart*exponent
                if (last  # no more radicals in base
                    or len(common_b) == 1  # nothing left to join with
                    or all(k[1] == 1 for k in common_b)  # no rad's in common_b
                        ):
                    break
                # see what we can exponentiate base by to remove any radicals
                # so we know what to search for
                # e.g. if base were x**(1/2)*y**(1/3) then we should
                # exponentiate by 6 and look for powers of x and y in the ratio
                # of 2 to 3
                qlcm = lcm([ratq(bi) for bi in Mul.make_args(bstart)])
                if qlcm == 1:
                    break  # we are done
                b = bstart**qlcm
                qlcm *= qstart
                if all(ratq(bi) == 1 for bi in Mul.make_args(b)):
                    last = True  # we are going to be done after this next pass
            # this base no longer can find anything to join with and
            # since it was longer than any other we are done with it
            b, q = base
            done.append((b, common_b.pop(base)*Rational(1, q)))

        # update c_powers and get ready to continue with powsimp
        c_powers = done
        # there may be terms still in common_b that were bases that were
        # identified as needing processing, so remove those, too
        for (b, q), e in common_b.items():
            if (b.is_Pow or isinstance(b, exp)) and \
                    q is not S.One and not b.exp.is_Rational:
                b, be = b.as_base_exp()
                b = b**(be/q)
            else:
                b = root(b, q)
            c_powers.append((b, e))
        check = len(c_powers)
        c_powers = dict(c_powers)
        assert len(c_powers) == check  # there should have been no duplicates
        # ==============================================================

        # rebuild the expression
        newexpr = expr.func(*(newexpr + [Pow(b, e) for b, e in c_powers.items()]))
        if combine == 'exp':
            return expr.func(newexpr, expr.func(*nc_part))
        else:
            return recurse(expr.func(*nc_part), combine='base') * \
                recurse(newexpr, combine='base')

    elif combine == 'base':

        # Build c_powers and nc_part.  These must both be lists not
        # dicts because exp's are not combined.
        c_powers = []
        nc_part = []
        for term in expr.args:
            if term.is_commutative:
                c_powers.append(list(term.as_base_exp()))
            else:
                nc_part.append(term)

        # Pull out numerical coefficients from exponent if assumptions allow
        # e.g., 2**(2*x) => 4**x
        for i in range(len(c_powers)):
            b, e = c_powers[i]
            if not (all(x.is_nonnegative for x in b.as_numer_denom()) or e.is_integer or force or b.is_polar):
                continue
            exp_c, exp_t = e.as_coeff_Mul(rational=True)
            if exp_c is not S.One and exp_t is not S.One:
                c_powers[i] = [Pow(b, exp_c), exp_t]

        # Combine bases whenever they have the same exponent and
        # assumptions allow
        # first gather the potential bases under the common exponent
        c_exp = defaultdict(list)
        for b, e in c_powers:
            if deep:
                e = recurse(e)
            if e.is_Add and (b.is_positive or e.is_integer):
                e = factor_terms(e)
                if _coeff_isneg(e):
                    e = -e
                    b = 1/b
            c_exp[e].append(b)
        del c_powers

        # Merge back in the results of the above to form a new product
        c_powers = defaultdict(list)
        for e in c_exp:
            bases = c_exp[e]

            # calculate the new base for e

            if len(bases) == 1:
                new_base = bases[0]
            elif e.is_integer or force:
                new_base = expr.func(*bases)
            else:
                # see which ones can be joined
                unk = []
                nonneg = []
                neg = []
                for bi in bases:
                    if bi.is_negative:
                        neg.append(bi)
                    elif bi.is_nonnegative:
                        nonneg.append(bi)
                    elif bi.is_polar:
                        nonneg.append(
                            bi)  # polar can be treated like non-negative
                    else:
                        unk.append(bi)
                if len(unk) == 1 and not neg or len(neg) == 1 and not unk:
                    # a single neg or a single unk can join the rest
                    nonneg.extend(unk + neg)
                    unk = neg = []
                elif neg:
                    # their negative signs cancel in groups of 2*q if we know
                    # that e = p/q else we have to treat them as unknown
                    israt = False
                    if e.is_Rational:
                        israt = True
                    else:
                        p, d = e.as_numer_denom()
                        if p.is_integer and d.is_integer:
                            israt = True
                    if israt:
                        neg = [-w for w in neg]
                        unk.extend([S.NegativeOne]*len(neg))
                    else:
                        unk.extend(neg)
                        neg = []
                    del israt

                # these shouldn't be joined
                for b in unk:
                    c_powers[b].append(e)
                # here is a new joined base
                new_base = expr.func(*(nonneg + neg))
                # if there are positive parts they will just get separated
                # again unless some change is made

                def _terms(e):
                    # return the number of terms of this expression
                    # when multiplied out -- assuming no joining of terms
                    if e.is_Add:
                        return sum(_terms(ai) for ai in e.args)
                    if e.is_Mul:
                        return prod([_terms(mi) for mi in e.args])
                    return 1
                xnew_base = expand_mul(new_base, deep=False)
                if len(Add.make_args(xnew_base)) < _terms(new_base):
                    new_base = factor_terms(xnew_base)

            c_powers[new_base].append(e)

        # break out the powers from c_powers now
        c_part = [Pow(b, ei) for b, e in c_powers.items() for ei in e]

        # we're done
        return expr.func(*(c_part + nc_part))

    else:
        raise ValueError("combine must be one of ('all', 'exp', 'base').")


def powdenest(eq, force=False, polar=False):
    r"""
    Collect exponents on powers as assumptions allow.

    Explanation
    ===========

    Given ``(bb**be)**e``, this can be simplified as follows:
        * if ``bb`` is positive, or
        * ``e`` is an integer, or
        * ``|be| < 1`` then this simplifies to ``bb**(be*e)``

    Given a product of powers raised to a power, ``(bb1**be1 *
    bb2**be2...)**e``, simplification can be done as follows:

    - if e is positive, the gcd of all bei can be joined with e;
    - all non-negative bb can be separated from those that are negative
      and their gcd can be joined with e; autosimplification already
      handles this separation.
    - integer factors from powers that have integers in the denominator
      of the exponent can be removed from any term and the gcd of such
      integers can be joined with e

    Setting ``force`` to ``True`` will make symbols that are not explicitly
    negative behave as though they are positive, resulting in more
    denesting.

    Setting ``polar`` to ``True`` will do simplifications on the Riemann surface of
    the logarithm, also resulting in more denestings.

    When there are sums of logs in exp() then a product of powers may be
    obtained e.g. ``exp(3*(log(a) + 2*log(b)))`` - > ``a**3*b**6``.

    Examples
    ========

    >>> from sympy.abc import a, b, x, y, z
    >>> from sympy import Symbol, exp, log, sqrt, symbols, powdenest

    >>> powdenest((x**(2*a/3))**(3*x))
    (x**(2*a/3))**(3*x)
    >>> powdenest(exp(3*x*log(2)))
    2**(3*x)

    Assumptions may prevent expansion:

    >>> powdenest(sqrt(x**2))
    sqrt(x**2)

    >>> p = symbols('p', positive=True)
    >>> powdenest(sqrt(p**2))
    p

    No other expansion is done.

    >>> i, j = symbols('i,j', integer=True)
    >>> powdenest((x**x)**(i + j)) # -X-> (x**x)**i*(x**x)**j
    x**(x*(i + j))

    But exp() will be denested by moving all non-log terms outside of
    the function; this may result in the collapsing of the exp to a power
    with a different base:

    >>> powdenest(exp(3*y*log(x)))
    x**(3*y)
    >>> powdenest(exp(y*(log(a) + log(b))))
    (a*b)**y
    >>> powdenest(exp(3*(log(a) + log(b))))
    a**3*b**3

    If assumptions allow, symbols can also be moved to the outermost exponent:

    >>> i = Symbol('i', integer=True)
    >>> powdenest(((x**(2*i))**(3*y))**x)
    ((x**(2*i))**(3*y))**x
    >>> powdenest(((x**(2*i))**(3*y))**x, force=True)
    x**(6*i*x*y)

    >>> powdenest(((x**(2*a/3))**(3*y/i))**x)
    ((x**(2*a/3))**(3*y/i))**x
    >>> powdenest((x**(2*i)*y**(4*i))**z, force=True)
    (x*y**2)**(2*i*z)

    >>> n = Symbol('n', negative=True)

    >>> powdenest((x**i)**y, force=True)
    x**(i*y)
    >>> powdenest((n**i)**x, force=True)
    (n**i)**x

    """
    from sympy.simplify.simplify import posify

    if force:
        def _denest(b, e):
            if not isinstance(b, (Pow, exp)):
                return b.is_positive, Pow(b, e, evaluate=False)
            return _denest(b.base, b.exp*e)
        reps = []
        for p in eq.atoms(Pow, exp):
            if isinstance(p.base, (Pow, exp)):
                ok, dp = _denest(*p.args)
                if ok is not False:
                    reps.append((p, dp))
        if reps:
            eq = eq.subs(reps)
        eq, reps = posify(eq)
        return powdenest(eq, force=False, polar=polar).xreplace(reps)

    if polar:
        eq, rep = polarify(eq)
        return unpolarify(powdenest(unpolarify(eq, exponents_only=True)), rep)

    new = powsimp(eq)
    return new.xreplace(Transform(
        _denest_pow, filter=lambda m: m.is_Pow or isinstance(m, exp)))

_y = Dummy('y')


def _denest_pow(eq):
    """
    Denest powers.

    This is a helper function for powdenest that performs the actual
    transformation.
    """
    from sympy.simplify.simplify import logcombine

    b, e = eq.as_base_exp()
    if b.is_Pow or isinstance(b, exp) and e != 1:
        new = b._eval_power(e)
        if new is not None:
            eq = new
            b, e = new.as_base_exp()

    # denest exp with log terms in exponent
    if b is S.Exp1 and e.is_Mul:
        logs = []
        other = []
        for ei in e.args:
            if any(isinstance(ai, log) for ai in Add.make_args(ei)):
                logs.append(ei)
            else:
                other.append(ei)
        logs = logcombine(Mul(*logs))
        return Pow(exp(logs), Mul(*other))

    _, be = b.as_base_exp()
    if be is S.One and not (b.is_Mul or
                            b.is_Rational and b.q != 1 or
                            b.is_positive):
        return eq

    # denest eq which is either pos**e or Pow**e or Mul**e or
    # Mul(b1**e1, b2**e2)

    # handle polar numbers specially
    polars, nonpolars = [], []
    for bb in Mul.make_args(b):
        if bb.is_polar:
            polars.append(bb.as_base_exp())
        else:
            nonpolars.append(bb)
    if len(polars) == 1 and not polars[0][0].is_Mul:
        return Pow(polars[0][0], polars[0][1]*e)*powdenest(Mul(*nonpolars)**e)
    elif polars:
        return Mul(*[powdenest(bb**(ee*e)) for (bb, ee) in polars]) \
            *powdenest(Mul(*nonpolars)**e)

    if b.is_Integer:
        # use log to see if there is a power here
        logb = expand_log(log(b))
        if logb.is_Mul:
            c, logb = logb.args
            e *= c
            base = logb.args[0]
            return Pow(base, e)

    # if b is not a Mul or any factor is an atom then there is nothing to do
    if not b.is_Mul or any(s.is_Atom for s in Mul.make_args(b)):
        return eq

    # let log handle the case of the base of the argument being a Mul, e.g.
    # sqrt(x**(2*i)*y**(6*i)) -> x**i*y**(3**i) if x and y are positive; we
    # will take the log, expand it, and then factor out the common powers that
    # now appear as coefficient. We do this manually since terms_gcd pulls out
    # fractions, terms_gcd(x+x*y/2) -> x*(y + 2)/2 and we don't want the 1/2;
    # gcd won't pull out numerators from a fraction: gcd(3*x, 9*x/2) -> x but
    # we want 3*x. Neither work with noncommutatives.

    def nc_gcd(aa, bb):
        a, b = [i.as_coeff_Mul() for i in [aa, bb]]
        c = gcd(a[0], b[0]).as_numer_denom()[0]
        g = Mul(*(a[1].args_cnc(cset=True)[0] & b[1].args_cnc(cset=True)[0]))
        return _keep_coeff(c, g)

    glogb = expand_log(log(b))
    if glogb.is_Add:
        args = glogb.args
        g = reduce(nc_gcd, args)
        if g != 1:
            cg, rg = g.as_coeff_Mul()
            glogb = _keep_coeff(cg, rg*Add(*[a/g for a in args]))

    # now put the log back together again
    if isinstance(glogb, log) or not glogb.is_Mul:
        if glogb.args[0].is_Pow or isinstance(glogb.args[0], exp):
            glogb = _denest_pow(glogb.args[0])
            if (abs(glogb.exp) < 1) == True:
                return Pow(glogb.base, glogb.exp*e)
        return eq

    # the log(b) was a Mul so join any adds with logcombine
    add = []
    other = []
    for a in glogb.args:
        if a.is_Add:
            add.append(a)
        else:
            other.append(a)
    return Pow(exp(logcombine(Mul(*add))), e*Mul(*other))
<|endoftext|>from collections import defaultdict
from functools import reduce

from sympy.core import (sympify, Basic, S, Expr, factor_terms,
                        Mul, Add, bottom_up)
from sympy.core.cache import cacheit
from sympy.core.function import (count_ops, _mexpand, FunctionClass, expand,
                                 expand_mul, _coeff_isneg, Derivative)
from sympy.core.numbers import I, Integer
from sympy.core.intfunc import igcd
from sympy.core.sorting import _nodes
from sympy.core.symbol import Dummy, symbols, Wild
from sympy.external.gmpy import SYMPY_INTS
from sympy.functions import sin, cos, exp, cosh, tanh, sinh, tan, cot, coth
from sympy.functions import atan2
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.polys import Poly, factor, cancel, parallel_poly_from_expr
from sympy.polys.domains import ZZ
from sympy.polys.polyerrors import PolificationFailed
from sympy.polys.polytools import groebner
from sympy.simplify.cse_main import cse
from sympy.strategies.core import identity
from sympy.strategies.tree import greedy
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import debug

def trigsimp_groebner(expr, hints=[], quick=False, order="grlex",
                      polynomial=False):
    """
    Simplify trigonometric expressions using a groebner basis algorithm.

    Explanation
    ===========

    This routine takes a fraction involving trigonometric or hyperbolic
    expressions, and tries to simplify it. The primary metric is the
    total degree. Some attempts are made to choose the simplest possible
    expression of the minimal degree, but this is non-rigorous, and also
    very slow (see the ``quick=True`` option).

    If ``polynomial`` is set to True, instead of simplifying numerator and
    denominator together, this function just brings numerator and denominator
    into a canonical form. This is much faster, but has potentially worse
    results. However, if the input is a polynomial, then the result is
    guaranteed to be an equivalent polynomial of minimal degree.

    The most important option is hints. Its entries can be any of the
    following:

    - a natural number
    - a function
    - an iterable of the form (func, var1, var2, ...)
    - anything else, interpreted as a generator

    A number is used to indicate that the search space should be increased.
    A function is used to indicate that said function is likely to occur in a
    simplified expression.
    An iterable is used indicate that func(var1 + var2 + ...) is likely to
    occur in a simplified .
    An additional generator also indicates that it is likely to occur.
    (See examples below).

    This routine carries out various computationally intensive algorithms.
    The option ``quick=True`` can be used to suppress one particularly slow
    step (at the expense of potentially more complicated results, but never at
    the expense of increased total degree).

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy import sin, tan, cos, sinh, cosh, tanh
    >>> from sympy.simplify.trigsimp import trigsimp_groebner

    Suppose you want to simplify ``sin(x)*cos(x)``. Naively, nothing happens:

    >>> ex = sin(x)*cos(x)
    >>> trigsimp_groebner(ex)
    sin(x)*cos(x)

    This is because ``trigsimp_groebner`` only looks for a simplification
    involving just ``sin(x)`` and ``cos(x)``. You can tell it to also try
    ``2*x`` by passing ``hints=[2]``:

    >>> trigsimp_groebner(ex, hints=[2])
    sin(2*x)/2
    >>> trigsimp_groebner(sin(x)**2 - cos(x)**2, hints=[2])
    -cos(2*x)

    Increasing the search space this way can quickly become expensive. A much
    faster way is to give a specific expression that is likely to occur:

    >>> trigsimp_groebner(ex, hints=[sin(2*x)])
    sin(2*x)/2

    Hyperbolic expressions are similarly supported:

    >>> trigsimp_groebner(sinh(2*x)/sinh(x))
    2*cosh(x)

    Note how no hints had to be passed, since the expression already involved
    ``2*x``.

    The tangent function is also supported. You can either pass ``tan`` in the
    hints, to indicate that tan should be tried whenever cosine or sine are,
    or you can pass a specific generator:

    >>> trigsimp_groebner(sin(x)/cos(x), hints=[tan])
    tan(x)
    >>> trigsimp_groebner(sinh(x)/cosh(x), hints=[tanh(x)])
    tanh(x)

    Finally, you can use the iterable form to suggest that angle sum formulae
    should be tried:

    >>> ex = (tan(x) + tan(y))/(1 - tan(x)*tan(y))
    >>> trigsimp_groebner(ex, hints=[(tan, x, y)])
    tan(x + y)
    """
    # TODO
    #  - preprocess by replacing everything by funcs we can handle
    # - optionally use cot instead of tan
    # - more intelligent hinting.
    #     For example, if the ideal is small, and we have sin(x), sin(y),
    #     add sin(x + y) automatically... ?
    # - algebraic numbers ...
    # - expressions of lowest degree are not distinguished properly
    #   e.g. 1 - sin(x)**2
    # - we could try to order the generators intelligently, so as to influence
    #   which monomials appear in the quotient basis

    # THEORY
    # ------
    # Ratsimpmodprime above can be used to "simplify" a rational function
    # modulo a prime ideal. "Simplify" mainly means finding an equivalent
    # expression of lower total degree.
    #
    # We intend to use this to simplify trigonometric functions. To do that,
    # we need to decide (a) which ring to use, and (b) modulo which ideal to
    # simplify. In practice, (a) means settling on a list of "generators"
    # a, b, c, ..., such that the fraction we want to simplify is a rational
    # function in a, b, c, ..., with coefficients in ZZ (integers).
    # (2) means that we have to decide what relations to impose on the
    # generators. There are two practical problems:
    #   (1) The ideal has to be *prime* (a technical term).
    #   (2) The relations have to be polynomials in the generators.
    #
    # We typically have two kinds of generators:
    # - trigonometric expressions, like sin(x), cos(5*x), etc
    # - "everything else", like gamma(x), pi, etc.
    #
    # Since this function is trigsimp, we will concentrate on what to do with
    # trigonometric expressions. We can also simplify hyperbolic expressions,
    # but the extensions should be clear.
    #
    # One crucial point is that all *other* generators really should behave
    # like indeterminates. In particular if (say) "I" is one of them, then
    # in fact I**2 + 1 = 0 and we may and will compute non-sensical
    # expressions. However, we can work with a dummy and add the relation
    # I**2 + 1 = 0 to our ideal, then substitute back in the end.
    #
    # Now regarding trigonometric generators. We split them into groups,
    # according to the argument of the trigonometric functions. We want to
    # organise this in such a way that most trigonometric identities apply in
    # the same group. For example, given sin(x), cos(2*x) and cos(y), we would
    # group as [sin(x), cos(2*x)] and [cos(y)].
    #
    # Our prime ideal will be built in three steps:
    # (1) For each group, compute a "geometrically prime" ideal of relations.
    #     Geometrically prime means that it generates a prime ideal in
    #     CC[gens], not just ZZ[gens].
    # (2) Take the union of all the generators of the ideals for all groups.
    #     By the geometric primality condition, this is still prime.
    # (3) Add further inter-group relations which preserve primality.
    #
    # Step (1) works as follows. We will isolate common factors in the
    # argument, so that all our generators are of the form sin(n*x), cos(n*x)
    # or tan(n*x), with n an integer. Suppose first there are no tan terms.
    # The ideal [sin(x)**2 + cos(x)**2 - 1] is geometrically prime, since
    # X**2 + Y**2 - 1 is irreducible over CC.
    # Now, if we have a generator sin(n*x), than we can, using trig identities,
    # express sin(n*x) as a polynomial in sin(x) and cos(x). We can add this
    # relation to the ideal, preserving geometric primality, since the quotient
    # ring is unchanged.
    # Thus we have treated all sin and cos terms.
    # For tan(n*x), we add a relation tan(n*x)*cos(n*x) - sin(n*x) = 0.
    # (This requires of course that we already have relations for cos(n*x) and
    # sin(n*x).) It is not obvious, but it seems that this preserves geometric
    # primality.
    # XXX A real proof would be nice. HELP!
    #     Sketch that <S**2 + C**2 - 1, C*T - S> is a prime ideal of
    #     CC[S, C, T]:
    #     - it suffices to show that the projective closure in CP**3 is
    #       irreducible
    #     - using the half-angle substitutions, we can express sin(x), tan(x),
    #       cos(x) as rational functions in tan(x/2)
    #     - from this, we get a rational map from CP**1 to our curve
    #     - this is a morphism, hence the curve is prime
    #
    # Step (2) is trivial.
    #
    # Step (3) works by adding selected relations of the form
    # sin(x + y) - sin(x)*cos(y) - sin(y)*cos(x), etc. Geometric primality is
    # preserved by the same argument as before.

    def parse_hints(hints):
        """Split hints into (n, funcs, iterables, gens)."""
        n = 1
        funcs, iterables, gens = [], [], []
        for e in hints:
            if isinstance(e, (SYMPY_INTS, Integer)):
                n = e
            elif isinstance(e, FunctionClass):
                funcs.append(e)
            elif iterable(e):
                iterables.append((e[0], e[1:]))
                # XXX sin(x+2y)?
                # Note: we go through polys so e.g.
                # sin(-x) -> -sin(x) -> sin(x)
                gens.extend(parallel_poly_from_expr(
                    [e[0](x) for x in e[1:]] + [e[0](Add(*e[1:]))])[1].gens)
            else:
                gens.append(e)
        return n, funcs, iterables, gens

    def build_ideal(x, terms):
        """
        Build generators for our ideal. ``Terms`` is an iterable with elements of
        the form (fn, coeff), indicating that we have a generator fn(coeff*x).

        If any of the terms is trigonometric, sin(x) and cos(x) are guaranteed
        to appear in terms. Similarly for hyperbolic functions. For tan(n*x),
        sin(n*x) and cos(n*x) are guaranteed.
        """
        I = []
        y = Dummy('y')
        for fn, coeff in terms:
            for c, s, t, rel in (
                    [cos, sin, tan, cos(x)**2 + sin(x)**2 - 1],
                    [cosh, sinh, tanh, cosh(x)**2 - sinh(x)**2 - 1]):
                if coeff == 1 and fn in [c, s]:
                    I.append(rel)
                elif fn == t:
                    I.append(t(coeff*x)*c(coeff*x) - s(coeff*x))
                elif fn in [c, s]:
                    cn = fn(coeff*y).expand(trig=True).subs(y, x)
                    I.append(fn(coeff*x) - cn)
        return list(set(I))

    def analyse_gens(gens, hints):
        """
        Analyse the generators ``gens``, using the hints ``hints``.

        The meaning of ``hints`` is described in the main docstring.
        Return a new list of generators, and also the ideal we should
        work with.
        """
        # First parse the hints
        n, funcs, iterables, extragens = parse_hints(hints)
        debug('n=%s   funcs: %s   iterables: %s    extragens: %s',
              (funcs, iterables, extragens))

        # We just add the extragens to gens and analyse them as before
        gens = list(gens)
        gens.extend(extragens)

        # remove duplicates
        funcs = list(set(funcs))
        iterables = list(set(iterables))
        gens = list(set(gens))

        # all the functions we can do anything with
        allfuncs = {sin, cos, tan, sinh, cosh, tanh}
        # sin(3*x) -> ((3, x), sin)
        trigterms = [(g.args[0].as_coeff_mul(), g.func) for g in gens
                     if g.func in allfuncs]
        # Our list of new generators - start with anything that we cannot
        # work with (i.e. is not a trigonometric term)
        freegens = [g for g in gens if g.func not in allfuncs]
        newgens = []
        trigdict = {}
        for (coeff, var), fn in trigterms:
            trigdict.setdefault(var, []).append((coeff, fn))
        res = [] # the ideal

        for key, val in trigdict.items():
            # We have now assembeled a dictionary. Its keys are common
            # arguments in trigonometric expressions, and values are lists of
            # pairs (fn, coeff). x0, (fn, coeff) in trigdict means that we
            # need to deal with fn(coeff*x0). We take the rational gcd of the
            # coeffs, call it ``gcd``. We then use x = x0/gcd as "base symbol",
            # all other arguments are integral multiples thereof.
            # We will build an ideal which works with sin(x), cos(x).
            # If hint tan is provided, also work with tan(x). Moreover, if
            # n > 1, also work with sin(k*x) for k <= n, and similarly for cos
            # (and tan if the hint is provided). Finally, any generators which
            # the ideal does not work with but we need to accommodate (either
            # because it was in expr or because it was provided as a hint)
            # we also build into the ideal.
            # This selection process is expressed in the list ``terms``.
            # build_ideal then generates the actual relations in our ideal,
            # from this list.
            fns = [x[1] for x in val]
            val = [x[0] for x in val]
            gcd = reduce(igcd, val)
            terms = [(fn, v/gcd) for (fn, v) in zip(fns, val)]
            fs = set(funcs + fns)
            for c, s, t in ([cos, sin, tan], [cosh, sinh, tanh]):
                if any(x in fs for x in (c, s, t)):
                    fs.add(c)
                    fs.add(s)
            for fn in fs:
                terms.extend((fn, k) for k in range(1, n + 1))
            extra = []
            for fn, v in terms:
                if fn == tan:
                    extra.append((sin, v))
                    extra.append((cos, v))
                if fn in [sin, cos] and tan in fs:
                    extra.append((tan, v))
                if fn == tanh:
                    extra.append((sinh, v))
                    extra.append((cosh, v))
                if fn in [sinh, cosh] and tanh in fs:
                    extra.append((tanh, v))
            terms.extend(extra)
            x = gcd*Mul(*key)
            r = build_ideal(x, terms)
            res.extend(r)
            newgens.extend({fn(v*x) for fn, v in terms})

        # Add generators for compound expressions from iterables
        for fn, args in iterables:
            if fn == tan:
                # Tan expressions are recovered from sin and cos.
                iterables.extend([(sin, args), (cos, args)])
            elif fn == tanh:
                # Tanh expressions are recovered from sihn and cosh.
                iterables.extend([(sinh, args), (cosh, args)])
            else:
                dummys = symbols('d:%i' % len(args), cls=Dummy)
                expr = fn( Add(*dummys)).expand(trig=True).subs(list(zip(dummys, args)))
                res.append(fn(Add(*args)) - expr)

        if myI in gens:
            res.append(myI**2 + 1)
            freegens.remove(myI)
            newgens.append(myI)

        return res, freegens, newgens

    myI = Dummy('I')
    expr = expr.subs(S.ImaginaryUnit, myI)
    subs = [(myI, S.ImaginaryUnit)]

    num, denom = cancel(expr).as_numer_denom()
    try:
        (pnum, pdenom), opt = parallel_poly_from_expr([num, denom])
    except PolificationFailed:
        return expr
    debug('initial gens:', opt.gens)
    ideal, freegens, gens = analyse_gens(opt.gens, hints)
    debug('ideal:', ideal)
    debug('new gens:', gens, " -- len", len(gens))
    debug('free gens:', freegens, " -- len", len(gens))
    # NOTE we force the domain to be ZZ to stop polys from injecting generators
    #      (which is usually a sign of a bug in the way we build the ideal)
    if not gens:
        return expr
    G = groebner(ideal, order=order, gens=gens, domain=ZZ)
    debug('groebner basis:', list(G), " -- len", len(G))

    # If our fraction is a polynomial in the free generators, simplify all
    # coefficients separately:

    from sympy.simplify.ratsimp import ratsimpmodprime

    if freegens and pdenom.has_only_gens(*set(gens).intersection(pdenom.gens)):
        num = Poly(num, gens=gens+freegens).eject(*gens)
        res = []
        for monom, coeff in num.terms():
            ourgens = set(parallel_poly_from_expr([coeff, denom])[1].gens)
            # We compute the transitive closure of all generators that can
            # be reached from our generators through relations in the ideal.
            changed = True
            while changed:
                changed = False
                for p in ideal:
                    p = Poly(p)
                    if not ourgens.issuperset(p.gens) and \
                       not p.has_only_gens(*set(p.gens).difference(ourgens)):
                        changed = True
                        ourgens.update(p.exclude().gens)
            # NOTE preserve order!
            realgens = [x for x in gens if x in ourgens]
            # The generators of the ideal have now been (implicitly) split
            # into two groups: those involving ourgens and those that don't.
            # Since we took the transitive closure above, these two groups
            # live in subgrings generated by a *disjoint* set of variables.
            # Any sensible groebner basis algorithm will preserve this disjoint
            # structure (i.e. the elements of the groebner basis can be split
            # similarly), and and the two subsets of the groebner basis then
            # form groebner bases by themselves. (For the smaller generating
            # sets, of course.)
            ourG = [g.as_expr() for g in G.polys if
                    g.has_only_gens(*ourgens.intersection(g.gens))]
            res.append(Mul(*[a**b for a, b in zip(freegens, monom)]) * \
                       ratsimpmodprime(coeff/denom, ourG, order=order,
                                       gens=realgens, quick=quick, domain=ZZ,
                                       polynomial=polynomial).subs(subs))
        return Add(*res)
        # NOTE The following is simpler and has less assumptions on the
        #      groebner basis algorithm. If the above turns out to be broken,
        #      use this.
        return Add(*[Mul(*[a**b for a, b in zip(freegens, monom)]) * \
                     ratsimpmodprime(coeff/denom, list(G), order=order,
                                     gens=gens, quick=quick, domain=ZZ)
                     for monom, coeff in num.terms()])
    else:
        return ratsimpmodprime(
            expr, list(G), order=order, gens=freegens+gens,
            quick=quick, domain=ZZ, polynomial=polynomial).subs(subs)


_trigs = (TrigonometricFunction, HyperbolicFunction)


def _trigsimp_inverse(rv):

    def check_args(x, y):
        try:
            return x.args[0] == y.args[0]
        except IndexError:
            return False

    def f(rv):
        # for simple functions
        g = getattr(rv, 'inverse', None)
        if (g is not None and isinstance(rv.args[0], g()) and
                isinstance(g()(1), TrigonometricFunction)):
            return rv.args[0].args[0]

        # for atan2 simplifications, harder because atan2 has 2 args
        if isinstance(rv, atan2):
            y, x = rv.args
            if _coeff_isneg(y):
                return -f(atan2(-y, x))
            elif _coeff_isneg(x):
                return S.Pi - f(atan2(y, -x))

            if check_args(x, y):
                if isinstance(y, sin) and isinstance(x, cos):
                    return x.args[0]
                if isinstance(y, cos) and isinstance(x, sin):
                    return S.Pi / 2 - x.args[0]

        return rv

    return bottom_up(rv, f)


def trigsimp(expr, inverse=False, **opts):
    """Returns a reduced expression by using known trig identities.

    Parameters
    ==========

    inverse : bool, optional
        If ``inverse=True``, it will be assumed that a composition of inverse
        functions, such as sin and asin, can be cancelled in any order.
        For example, ``asin(sin(x))`` will yield ``x`` without checking whether
        x belongs to the set where this relation is true. The default is False.
        Default : True

    method : string, optional
        Specifies the method to use. Valid choices are:

        - ``'matching'``, default
        - ``'groebner'``
        - ``'combined'``
        - ``'fu'``
        - ``'old'``

        If ``'matching'``, simplify the expression recursively by targeting
        common patterns. If ``'groebner'``, apply an experimental groebner
        basis algorithm. In this case further options are forwarded to
        ``trigsimp_groebner``, please refer to
        its docstring. If ``'combined'``, it first runs the groebner basis
        algorithm with small default parameters, then runs the ``'matching'``
        algorithm. If ``'fu'``, run the collection of trigonometric
        transformations described by Fu, et al. (see the
        :py:func:`~sympy.simplify.fu.fu` docstring). If ``'old'``, the original
        SymPy trig simplification function is run.
    opts :
        Optional keyword arguments passed to the method. See each method's
        function docstring for details.

    Examples
    ========

    >>> from sympy import trigsimp, sin, cos, log
    >>> from sympy.abc import x
    >>> e = 2*sin(x)**2 + 2*cos(x)**2
    >>> trigsimp(e)
    2

    Simplification occurs wherever trigonometric functions are located.

    >>> trigsimp(log(e))
    log(2)

    Using ``method='groebner'`` (or ``method='combined'``) might lead to
    greater simplification.

    The old trigsimp routine can be accessed as with method ``method='old'``.

    >>> from sympy import coth, tanh
    >>> t = 3*tanh(x)**7 - 2/coth(x)**7
    >>> trigsimp(t, method='old') == t
    True
    >>> trigsimp(t)
    tanh(x)**7

    """
    from sympy.simplify.fu import fu

    expr = sympify(expr)

    _eval_trigsimp = getattr(expr, '_eval_trigsimp', None)
    if _eval_trigsimp is not None:
        return _eval_trigsimp(**opts)

    old = opts.pop('old', False)
    if not old:
        opts.pop('deep', None)
        opts.pop('recursive', None)
        method = opts.pop('method', 'matching')
    else:
        method = 'old'

    def groebnersimp(ex, **opts):
        def traverse(e):
            if e.is_Atom:
                return e
            args = [traverse(x) for x in e.args]
            if e.is_Function or e.is_Pow:
                args = [trigsimp_groebner(x, **opts) for x in args]
            return e.func(*args)
        new = traverse(ex)
        if not isinstance(new, Expr):
            return new
        return trigsimp_groebner(new, **opts)

    trigsimpfunc = {
        'fu': (lambda x: fu(x, **opts)),
        'matching': (lambda x: futrig(x)),
        'groebner': (lambda x: groebnersimp(x, **opts)),
        'combined': (lambda x: futrig(groebnersimp(x,
                               polynomial=True, hints=[2, tan]))),
        'old': lambda x: trigsimp_old(x, **opts),
                   }[method]

    expr_simplified = trigsimpfunc(expr)
    if inverse:
        expr_simplified = _trigsimp_inverse(expr_simplified)

    return expr_simplified


def exptrigsimp(expr):
    """
    Simplifies exponential / trigonometric / hyperbolic functions.

    Examples
    ========

    >>> from sympy import exptrigsimp, exp, cosh, sinh
    >>> from sympy.abc import z

    >>> exptrigsimp(exp(z) + exp(-z))
    2*cosh(z)
    >>> exptrigsimp(cosh(z) - sinh(z))
    exp(-z)
    """
    from sympy.simplify.fu import hyper_as_trig, TR2i

    def exp_trig(e):
        # select the better of e, and e rewritten in terms of exp or trig
        # functions
        choices = [e]
        if e.has(*_trigs):
            choices.append(e.rewrite(exp))
        choices.append(e.rewrite(cos))
        return min(*choices, key=count_ops)
    newexpr = bottom_up(expr, exp_trig)

    def f(rv):
        if not rv.is_Mul:
            return rv
        commutative_part, noncommutative_part = rv.args_cnc()
        # Since as_powers_dict loses order information,
        # if there is more than one noncommutative factor,
        # it should only be used to simplify the commutative part.
        if (len(noncommutative_part) > 1):
            return f(Mul(*commutative_part))*Mul(*noncommutative_part)
        rvd = rv.as_powers_dict()
        newd = rvd.copy()

        def signlog(expr, sign=S.One):
            if expr is S.Exp1:
                return sign, S.One
            elif isinstance(expr, exp) or (expr.is_Pow and expr.base == S.Exp1):
                return sign, expr.exp
            elif sign is S.One:
                return signlog(-expr, sign=-S.One)
            else:
                return None, None

        ee = rvd[S.Exp1]
        for k in rvd:
            if k.is_Add and len(k.args) == 2:
                # k == c*(1 + sign*E**x)
                c = k.args[0]
                sign, x = signlog(k.args[1]/c)
                if not x:
                    continue
                m = rvd[k]
                newd[k] -= m
                if ee == -x*m/2:
                    # sinh and cosh
                    newd[S.Exp1] -= ee
                    ee = 0
                    if sign == 1:
                        newd[2*c*cosh(x/2)] += m
                    else:
                        newd[-2*c*sinh(x/2)] += m
                elif newd[1 - sign*S.Exp1**x] == -m:
                    # tanh
                    del newd[1 - sign*S.Exp1**x]
                    if sign == 1:
                        newd[-c/tanh(x/2)] += m
                    else:
                        newd[-c*tanh(x/2)] += m
                else:
                    newd[1 + sign*S.Exp1**x] += m
                    newd[c] += m

        return Mul(*[k**newd[k] for k in newd])
    newexpr = bottom_up(newexpr, f)

    # sin/cos and sinh/cosh ratios to tan and tanh, respectively
    if newexpr.has(HyperbolicFunction):
        e, f = hyper_as_trig(newexpr)
        newexpr = f(TR2i(e))
    if newexpr.has(TrigonometricFunction):
        newexpr = TR2i(newexpr)

    # can we ever generate an I where there was none previously?
    if not (newexpr.has(I) and not expr.has(I)):
        expr = newexpr
    return expr

#-------------------- the old trigsimp routines ---------------------

def trigsimp_old(expr, *, first=True, **opts):
    """
    Reduces expression by using known trig identities.

    Notes
    =====

    deep:
    - Apply trigsimp inside all objects with arguments

    recursive:
    - Use common subexpression elimination (cse()) and apply
    trigsimp recursively (this is quite expensive if the
    expression is large)

    method:
    - Determine the method to use. Valid choices are 'matching' (default),
    'groebner', 'combined', 'fu' and 'futrig'. If 'matching', simplify the
    expression recursively by pattern matching. If 'groebner', apply an
    experimental groebner basis algorithm. In this case further options
    are forwarded to ``trigsimp_groebner``, please refer to its docstring.
    If 'combined', first run the groebner basis algorithm with small
    default parameters, then run the 'matching' algorithm. 'fu' runs the
    collection of trigonometric transformations described by Fu, et al.
    (see the `fu` docstring) while `futrig` runs a subset of Fu-transforms
    that mimic the behavior of `trigsimp`.

    compare:
    - show input and output from `trigsimp` and `futrig` when different,
    but returns the `trigsimp` value.

    Examples
    ========

    >>> from sympy import trigsimp, sin, cos, log, cot
    >>> from sympy.abc import x
    >>> e = 2*sin(x)**2 + 2*cos(x)**2
    >>> trigsimp(e, old=True)
    2
    >>> trigsimp(log(e), old=True)
    log(2*sin(x)**2 + 2*cos(x)**2)
    >>> trigsimp(log(e), deep=True, old=True)
    log(2)

    Using `method="groebner"` (or `"combined"`) can sometimes lead to a lot
    more simplification:

    >>> e = (-sin(x) + 1)/cos(x) + cos(x)/(-sin(x) + 1)
    >>> trigsimp(e, old=True)
    (1 - sin(x))/cos(x) + cos(x)/(1 - sin(x))
    >>> trigsimp(e, method="groebner", old=True)
    2/cos(x)

    >>> trigsimp(1/cot(x)**2, compare=True, old=True)
          futrig: tan(x)**2
    cot(x)**(-2)

    """
    old = expr
    if first:
        if not expr.has(*_trigs):
            return expr

        trigsyms = set().union(*[t.free_symbols for t in expr.atoms(*_trigs)])
        if len(trigsyms) > 1:
            from sympy.simplify.simplify import separatevars

            d = separatevars(expr)
            if d.is_Mul:
                d = separatevars(d, dict=True) or d
            if isinstance(d, dict):
                expr = 1
                for v in d.values():
                    # remove hollow factoring
                    was = v
                    v = expand_mul(v)
                    opts['first'] = False
                    vnew = trigsimp(v, **opts)
                    if vnew == v:
                        vnew = was
                    expr *= vnew
                old = expr
            else:
                if d.is_Add:
                    for s in trigsyms:
                        r, e = expr.as_independent(s)
                        if r:
                            opts['first'] = False
                            expr = r + trigsimp(e, **opts)
                            if not expr.is_Add:
                                break
                    old = expr

    recursive = opts.pop('recursive', False)
    deep = opts.pop('deep', False)
    method = opts.pop('method', 'matching')

    def groebnersimp(ex, deep, **opts):
        def traverse(e):
            if e.is_Atom:
                return e
            args = [traverse(x) for x in e.args]
            if e.is_Function or e.is_Pow:
                args = [trigsimp_groebner(x, **opts) for x in args]
            return e.func(*args)
        if deep:
            ex = traverse(ex)
        return trigsimp_groebner(ex, **opts)

    trigsimpfunc = {
        'matching': (lambda x, d: _trigsimp(x, d)),
        'groebner': (lambda x, d: groebnersimp(x, d, **opts)),
        'combined': (lambda x, d: _trigsimp(groebnersimp(x,
                                       d, polynomial=True, hints=[2, tan]),
                                   d))
                   }[method]

    if recursive:
        w, g = cse(expr)
        g = trigsimpfunc(g[0], deep)

        for sub in reversed(w):
            g = g.subs(sub[0], sub[1])
            g = trigsimpfunc(g, deep)
        result = g
    else:
        result = trigsimpfunc(expr, deep)

    if opts.get('compare', False):
        f = futrig(old)
        if f != result:
            print('\tfutrig:', f)

    return result


def _dotrig(a, b):
    """Helper to tell whether ``a`` and ``b`` have the same sorts
    of symbols in them -- no need to test hyperbolic patterns against
    expressions that have no hyperbolics in them."""
    return a.func == b.func and (
        a.has(TrigonometricFunction) and b.has(TrigonometricFunction) or
        a.has(HyperbolicFunction) and b.has(HyperbolicFunction))


_trigpat = None
def _trigpats():
    global _trigpat
    a, b, c = symbols('a b c', cls=Wild)
    d = Wild('d', commutative=False)

    # for the simplifications like sinh/cosh -> tanh:
    # DO NOT REORDER THE FIRST 14 since these are assumed to be in this
    # order in _match_div_rewrite.
    matchers_division = (
        (a*sin(b)**c/cos(b)**c, a*tan(b)**c, sin(b), cos(b)),
        (a*tan(b)**c*cos(b)**c, a*sin(b)**c, sin(b), cos(b)),
        (a*cot(b)**c*sin(b)**c, a*cos(b)**c, sin(b), cos(b)),
        (a*tan(b)**c/sin(b)**c, a/cos(b)**c, sin(b), cos(b)),
        (a*cot(b)**c/cos(b)**c, a/sin(b)**c, sin(b), cos(b)),
        (a*cot(b)**c*tan(b)**c, a, sin(b), cos(b)),
        (a*(cos(b) + 1)**c*(cos(b) - 1)**c,
            a*(-sin(b)**2)**c, cos(b) + 1, cos(b) - 1),
        (a*(sin(b) + 1)**c*(sin(b) - 1)**c,
            a*(-cos(b)**2)**c, sin(b) + 1, sin(b) - 1),

        (a*sinh(b)**c/cosh(b)**c, a*tanh(b)**c, S.One, S.One),
        (a*tanh(b)**c*cosh(b)**c, a*sinh(b)**c, S.One, S.One),
        (a*coth(b)**c*sinh(b)**c, a*cosh(b)**c, S.One, S.One),
        (a*tanh(b)**c/sinh(b)**c, a/cosh(b)**c, S.One, S.One),
        (a*coth(b)**c/cosh(b)**c, a/sinh(b)**c, S.One, S.One),
        (a*coth(b)**c*tanh(b)**c, a, S.One, S.One),

        (c*(tanh(a) + tanh(b))/(1 + tanh(a)*tanh(b)),
            tanh(a + b)*c, S.One, S.One),
    )

    matchers_add = (
        (c*sin(a)*cos(b) + c*cos(a)*sin(b) + d, sin(a + b)*c + d),
        (c*cos(a)*cos(b) - c*sin(a)*sin(b) + d, cos(a + b)*c + d),
        (c*sin(a)*cos(b) - c*cos(a)*sin(b) + d, sin(a - b)*c + d),
        (c*cos(a)*cos(b) + c*sin(a)*sin(b) + d, cos(a - b)*c + d),
        (c*sinh(a)*cosh(b) + c*sinh(b)*cosh(a) + d, sinh(a + b)*c + d),
        (c*cosh(a)*cosh(b) + c*sinh(a)*sinh(b) + d, cosh(a + b)*c + d),
    )

    # for cos(x)**2 + sin(x)**2 -> 1
    matchers_identity = (
        (a*sin(b)**2, a - a*cos(b)**2),
        (a*tan(b)**2, a*(1/cos(b))**2 - a),
        (a*cot(b)**2, a*(1/sin(b))**2 - a),
        (a*sin(b + c), a*(sin(b)*cos(c) + sin(c)*cos(b))),
        (a*cos(b + c), a*(cos(b)*cos(c) - sin(b)*sin(c))),
        (a*tan(b + c), a*((tan(b) + tan(c))/(1 - tan(b)*tan(c)))),

        (a*sinh(b)**2, a*cosh(b)**2 - a),
        (a*tanh(b)**2, a - a*(1/cosh(b))**2),
        (a*coth(b)**2, a + a*(1/sinh(b))**2),
        (a*sinh(b + c), a*(sinh(b)*cosh(c) + sinh(c)*cosh(b))),
        (a*cosh(b + c), a*(cosh(b)*cosh(c) + sinh(b)*sinh(c))),
        (a*tanh(b + c), a*((tanh(b) + tanh(c))/(1 + tanh(b)*tanh(c)))),

    )

    # Reduce any lingering artifacts, such as sin(x)**2 changing
    # to 1-cos(x)**2 when sin(x)**2 was "simpler"
    artifacts = (
        (a - a*cos(b)**2 + c, a*sin(b)**2 + c, cos),
        (a - a*(1/cos(b))**2 + c, -a*tan(b)**2 + c, cos),
        (a - a*(1/sin(b))**2 + c, -a*cot(b)**2 + c, sin),

        (a - a*cosh(b)**2 + c, -a*sinh(b)**2 + c, cosh),
        (a - a*(1/cosh(b))**2 + c, a*tanh(b)**2 + c, cosh),
        (a + a*(1/sinh(b))**2 + c, a*coth(b)**2 + c, sinh),

        # same as above but with noncommutative prefactor
        (a*d - a*d*cos(b)**2 + c, a*d*sin(b)**2 + c, cos),
        (a*d - a*d*(1/cos(b))**2 + c, -a*d*tan(b)**2 + c, cos),
        (a*d - a*d*(1/sin(b))**2 + c, -a*d*cot(b)**2 + c, sin),

        (a*d - a*d*cosh(b)**2 + c, -a*d*sinh(b)**2 + c, cosh),
        (a*d - a*d*(1/cosh(b))**2 + c, a*d*tanh(b)**2 + c, cosh),
        (a*d + a*d*(1/sinh(b))**2 + c, a*d*coth(b)**2 + c, sinh),
    )

    _trigpat = (a, b, c, d, matchers_division, matchers_add,
        matchers_identity, artifacts)
    return _trigpat


def _replace_mul_fpowxgpow(expr, f, g, rexp, h, rexph):
    """Helper for _match_div_rewrite.

    Replace f(b_)**c_*g(b_)**(rexp(c_)) with h(b)**rexph(c) if f(b_)
    and g(b_) are both positive or if c_ is an integer.
    """
    # assert expr.is_Mul and expr.is_commutative and f != g
    fargs = defaultdict(int)
    gargs = defaultdict(int)
    args = []
    for x in expr.args:
        if x.is_Pow or x.func in (f, g):
            b, e = x.as_base_exp()
            if b.is_positive or e.is_integer:
                if b.func == f:
                    fargs[b.args[0]] += e
                    continue
                elif b.func == g:
                    gargs[b.args[0]] += e
                    continue
        args.append(x)
    common = set(fargs) & set(gargs)
    hit = False
    while common:
        key = common.pop()
        fe = fargs.pop(key)
        ge = gargs.pop(key)
        if fe == rexp(ge):
            args.append(h(key)**rexph(fe))
            hit = True
        else:
            fargs[key] = fe
            gargs[key] = ge
    if not hit:
        return expr
    while fargs:
        key, e = fargs.popitem()
        args.append(f(key)**e)
    while gargs:
        key, e = gargs.popitem()
        args.append(g(key)**e)
    return Mul(*args)


_idn = lambda x: x
_midn = lambda x: -x
_one = lambda x: S.One

def _match_div_rewrite(expr, i):
    """helper for __trigsimp"""
    if i == 0:
        expr = _replace_mul_fpowxgpow(expr, sin, cos,
            _midn, tan, _idn)
    elif i == 1:
        expr = _replace_mul_fpowxgpow(expr, tan, cos,
            _idn, sin, _idn)
    elif i == 2:
        expr = _replace_mul_fpowxgpow(expr, cot, sin,
            _idn, cos, _idn)
    elif i == 3:
        expr = _replace_mul_fpowxgpow(expr, tan, sin,
            _midn, cos, _midn)
    elif i == 4:
        expr = _replace_mul_fpowxgpow(expr, cot, cos,
            _midn, sin, _midn)
    elif i == 5:
        expr = _replace_mul_fpowxgpow(expr, cot, tan,
            _idn, _one, _idn)
    # i in (6, 7) is skipped
    elif i == 8:
        expr = _replace_mul_fpowxgpow(expr, sinh, cosh,
            _midn, tanh, _idn)
    elif i == 9:
        expr = _replace_mul_fpowxgpow(expr, tanh, cosh,
            _idn, sinh, _idn)
    elif i == 10:
        expr = _replace_mul_fpowxgpow(expr, coth, sinh,
            _idn, cosh, _idn)
    elif i == 11:
        expr = _replace_mul_fpowxgpow(expr, tanh, sinh,
            _midn, cosh, _midn)
    elif i == 12:
        expr = _replace_mul_fpowxgpow(expr, coth, cosh,
            _midn, sinh, _midn)
    elif i == 13:
        expr = _replace_mul_fpowxgpow(expr, coth, tanh,
            _idn, _one, _idn)
    else:
        return None
    return expr


def _trigsimp(expr, deep=False):
    # protect the cache from non-trig patterns; we only allow
    # trig patterns to enter the cache
    if expr.has(*_trigs):
        return __trigsimp(expr, deep)
    return expr


@cacheit
def __trigsimp(expr, deep=False):
    """recursive helper for trigsimp"""
    from sympy.simplify.fu import TR10i

    if _trigpat is None:
        _trigpats()
    a, b, c, d, matchers_division, matchers_add, \
    matchers_identity, artifacts = _trigpat

    if expr.is_Mul:
        # do some simplifications like sin/cos -> tan:
        if not expr.is_commutative:
            com, nc = expr.args_cnc()
            expr = _trigsimp(Mul._from_args(com), deep)*Mul._from_args(nc)
        else:
            for i, (pattern, simp, ok1, ok2) in enumerate(matchers_division):
                if not _dotrig(expr, pattern):
                    continue

                newexpr = _match_div_rewrite(expr, i)
                if newexpr is not None:
                    if newexpr != expr:
                        expr = newexpr
                        break
                    else:
                        continue

                # use SymPy matching instead
                res = expr.match(pattern)
                if res and res.get(c, 0):
                    if not res[c].is_integer:
                        ok = ok1.subs(res)
                        if not ok.is_positive:
                            continue
                        ok = ok2.subs(res)
                        if not ok.is_positive:
                            continue
                    # if "a" contains any of trig or hyperbolic funcs with
                    # argument "b" then skip the simplification
                    if any(w.args[0] == res[b] for w in res[a].atoms(
                            TrigonometricFunction, HyperbolicFunction)):
                        continue
                    # simplify and finish:
                    expr = simp.subs(res)
                    break  # process below

    if expr.is_Add:
        args = []
        for term in expr.args:
            if not term.is_commutative:
                com, nc = term.args_cnc()
                nc = Mul._from_args(nc)
                term = Mul._from_args(com)
            else:
                nc = S.One
            term = _trigsimp(term, deep)
            for pattern, result in matchers_identity:
                res = term.match(pattern)
                if res is not None:
                    term = result.subs(res)
                    break
            args.append(term*nc)
        if args != expr.args:
            expr = Add(*args)
            expr = min(expr, expand(expr), key=count_ops)
        if expr.is_Add:
            for pattern, result in matchers_add:
                if not _dotrig(expr, pattern):
                    continue
                expr = TR10i(expr)
                if expr.has(HyperbolicFunction):
                    res = expr.match(pattern)
                    # if "d" contains any trig or hyperbolic funcs with
                    # argument "a" or "b" then skip the simplification;
                    # this isn't perfect -- see tests
                    if res is None or not (a in res and b in res) or any(
                        w.args[0] in (res[a], res[b]) for w in res[d].atoms(
                            TrigonometricFunction, HyperbolicFunction)):
                        continue
                    expr = result.subs(res)
                    break

        # Reduce any lingering artifacts, such as sin(x)**2 changing
        # to 1 - cos(x)**2 when sin(x)**2 was "simpler"
        for pattern, result, ex in artifacts:
            if not _dotrig(expr, pattern):
                continue
            # Substitute a new wild that excludes some function(s)
            # to help influence a better match. This is because
            # sometimes, for example, 'a' would match sec(x)**2
            a_t = Wild('a', exclude=[ex])
            pattern = pattern.subs(a, a_t)
            result = result.subs(a, a_t)

            m = expr.match(pattern)
            was = None
            while m and was != expr:
                was = expr
                if m[a_t] == 0 or \
                        -m[a_t] in m[c].args or m[a_t] + m[c] == 0:
                    break
                if d in m and m[a_t]*m[d] + m[c] == 0:
                    break
                expr = result.subs(m)
                m = expr.match(pattern)
                m.setdefault(c, S.Zero)

    elif expr.is_Mul or expr.is_Pow or deep and expr.args:
        expr = expr.func(*[_trigsimp(a, deep) for a in expr.args])

    try:
        if not expr.has(*_trigs):
            raise TypeError
        e = expr.atoms(exp)
        new = expr.rewrite(exp, deep=deep)
        if new == e:
            raise TypeError
        fnew = factor(new)
        if fnew != new:
            new = min([new, factor(new)], key=count_ops)
        # if all exp that were introduced disappeared then accept it
        if not (new.atoms(exp) - e):
            expr = new
    except TypeError:
        pass

    return expr
#------------------- end of old trigsimp routines --------------------


def futrig(e, *, hyper=True, **kwargs):
    """Return simplified ``e`` using Fu-like transformations.
    This is not the "Fu" algorithm. This is called by default
    from ``trigsimp``. By default, hyperbolics subexpressions
    will be simplified, but this can be disabled by setting
    ``hyper=False``.

    Examples
    ========

    >>> from sympy import trigsimp, tan, sinh, tanh
    >>> from sympy.simplify.trigsimp import futrig
    >>> from sympy.abc import x
    >>> trigsimp(1/tan(x)**2)
    tan(x)**(-2)

    >>> futrig(sinh(x)/tanh(x))
    cosh(x)

    """
    from sympy.simplify.fu import hyper_as_trig

    e = sympify(e)

    if not isinstance(e, Basic):
        return e

    if not e.args:
        return e

    old = e
    e = bottom_up(e, _futrig)

    if hyper and e.has(HyperbolicFunction):
        e, f = hyper_as_trig(e)
        e = f(bottom_up(e, _futrig))

    if e != old and e.is_Mul and e.args[0].is_Rational:
        # redistribute leading coeff on 2-arg Add
        e = Mul(*e.as_coeff_Mul())
    return e


def _futrig(e):
    """Helper for futrig."""
    from sympy.simplify.fu import (
        TR1, TR2, TR3, TR2i, TR10, L, TR10i,
        TR8, TR6, TR15, TR16, TR111, TR5, TRmorrie, TR11, _TR11, TR14, TR22,
        TR12)

    if not e.has(TrigonometricFunction):
        return e

    if e.is_Mul:
        coeff, e = e.as_independent(TrigonometricFunction)
    else:
        coeff = None

    Lops = lambda x: (L(x), x.count_ops(), _nodes(x), len(x.args), x.is_Add)
    trigs = lambda x: x.has(TrigonometricFunction)

    tree = [identity,
        (
        TR3,  # canonical angles
        TR1,  # sec-csc -> cos-sin
        TR12,  # expand tan of sum
        lambda x: _eapply(factor, x, trigs),
        TR2,  # tan-cot -> sin-cos
        [identity, lambda x: _eapply(_mexpand, x, trigs)],
        TR2i,  # sin-cos ratio -> tan
        lambda x: _eapply(lambda i: factor(i.normal()), x, trigs),
        TR14,  # factored identities
        TR5,  # sin-pow -> cos_pow
        TR10,  # sin-cos of sums -> sin-cos prod
        TR11, _TR11, TR6, # reduce double angles and rewrite cos pows
        lambda x: _eapply(factor, x, trigs),
        TR14,  # factored powers of identities
        [identity, lambda x: _eapply(_mexpand, x, trigs)],
        TR10i,  # sin-cos products > sin-cos of sums
        TRmorrie,
        [identity, TR8],  # sin-cos products -> sin-cos of sums
        [identity, lambda x: TR2i(TR2(x))],  # tan -> sin-cos -> tan
        [
            lambda x: _eapply(expand_mul, TR5(x), trigs),
            lambda x: _eapply(
                expand_mul, TR15(x), trigs)], # pos/neg powers of sin
        [
            lambda x:  _eapply(expand_mul, TR6(x), trigs),
            lambda x:  _eapply(
                expand_mul, TR16(x), trigs)], # pos/neg powers of cos
        TR111,  # tan, sin, cos to neg power -> cot, csc, sec
        [identity, TR2i],  # sin-cos ratio to tan
        [identity, lambda x: _eapply(
            expand_mul, TR22(x), trigs)],  # tan-cot to sec-csc
        TR1, TR2, TR2i,
        [identity, lambda x: _eapply(
            factor_terms, TR12(x), trigs)],  # expand tan of sum
        )]
    e = greedy(tree, objective=Lops)(e)

    if coeff is not None:
        e = coeff * e

    return e


def _is_Expr(e):
    """_eapply helper to tell whether ``e`` and all its args
    are Exprs."""
    if isinstance(e, Derivative):
        return _is_Expr(e.expr)
    if not isinstance(e, Expr):
        return False
    return all(_is_Expr(i) for i in e.args)


def _eapply(func, e, cond=None):
    """Apply ``func`` to ``e`` if all args are Exprs else only
    apply it to those args that *are* Exprs."""
    if not isinstance(e, Expr):
        return e
    if _is_Expr(e) or not e.args:
        return func(e)
    return e.func(*[
        _eapply(func, ei) if (cond is None or cond(ei)) else ei
        for ei in e.args])
<|endoftext|>from itertools import combinations_with_replacement
from sympy.core import symbols, Add, Dummy
from sympy.core.numbers import Rational
from sympy.polys import cancel, ComputationFailed, parallel_poly_from_expr, reduced, Poly
from sympy.polys.monomials import Monomial, monomial_div
from sympy.polys.polyerrors import DomainError, PolificationFailed
from sympy.utilities.misc import debug, debugf

def ratsimp(expr):
    """
    Put an expression over a common denominator, cancel and reduce.

    Examples
    ========

    >>> from sympy import ratsimp
    >>> from sympy.abc import x, y
    >>> ratsimp(1/x + 1/y)
    (x + y)/(x*y)
    """

    f, g = cancel(expr).as_numer_denom()
    try:
        Q, r = reduced(f, [g], field=True, expand=False)
    except ComputationFailed:
        return f/g

    return Add(*Q) + cancel(r/g)


def ratsimpmodprime(expr, G, *gens, quick=True, polynomial=False, **args):
    """
    Simplifies a rational expression ``expr`` modulo the prime ideal
    generated by ``G``.  ``G`` should be a Groebner basis of the
    ideal.

    Examples
    ========

    >>> from sympy.simplify.ratsimp import ratsimpmodprime
    >>> from sympy.abc import x, y
    >>> eq = (x + y**5 + y)/(x - y)
    >>> ratsimpmodprime(eq, [x*y**5 - x - y], x, y, order='lex')
    (-x**2 - x*y - x - y)/(-x**2 + x*y)

    If ``polynomial`` is ``False``, the algorithm computes a rational
    simplification which minimizes the sum of the total degrees of
    the numerator and the denominator.

    If ``polynomial`` is ``True``, this function just brings numerator and
    denominator into a canonical form. This is much faster, but has
    potentially worse results.

    References
    ==========

    .. [1] M. Monagan, R. Pearce, Rational Simplification Modulo a Polynomial
        Ideal, https://dl.acm.org/doi/pdf/10.1145/1145768.1145809
        (specifically, the second algorithm)
    """
    from sympy.solvers.solvers import solve

    debug('ratsimpmodprime', expr)

    # usual preparation of polynomials:

    num, denom = cancel(expr).as_numer_denom()

    try:
        polys, opt = parallel_poly_from_expr([num, denom] + G, *gens, **args)
    except PolificationFailed:
        return expr

    domain = opt.domain

    if domain.has_assoc_Field:
        opt.domain = domain.get_field()
    else:
        raise DomainError(
            "Cannot compute rational simplification over %s" % domain)

    # compute only once
    leading_monomials = [g.LM(opt.order) for g in polys[2:]]
    tested = set()

    def staircase(n):
        """
        Compute all monomials with degree less than ``n`` that are
        not divisible by any element of ``leading_monomials``.
        """
        if n == 0:
            return [1]
        S = []
        for mi in combinations_with_replacement(range(len(opt.gens)), n):
            m = [0]*len(opt.gens)
            for i in mi:
                m[i] += 1
            if all(monomial_div(m, lmg) is None for lmg in
                   leading_monomials):
                S.append(m)

        return [Monomial(s).as_expr(*opt.gens) for s in S] + staircase(n - 1)

    def _ratsimpmodprime(a, b, allsol, N=0, D=0):
        r"""
        Computes a rational simplification of ``a/b`` which minimizes
        the sum of the total degrees of the numerator and the denominator.

        Explanation
        ===========

        The algorithm proceeds by looking at ``a * d - b * c`` modulo
        the ideal generated by ``G`` for some ``c`` and ``d`` with degree
        less than ``a`` and ``b`` respectively.
        The coefficients of ``c`` and ``d`` are indeterminates and thus
        the coefficients of the normalform of ``a * d - b * c`` are
        linear polynomials in these indeterminates.
        If these linear polynomials, considered as system of
        equations, have a nontrivial solution, then `\frac{a}{b}
        \equiv \frac{c}{d}` modulo the ideal generated by ``G``. So,
        by construction, the degree of ``c`` and ``d`` is less than
        the degree of ``a`` and ``b``, so a simpler representation
        has been found.
        After a simpler representation has been found, the algorithm
        tries to reduce the degree of the numerator and denominator
        and returns the result afterwards.

        As an extension, if quick=False, we look at all possible degrees such
        that the total degree is less than *or equal to* the best current
        solution. We retain a list of all solutions of minimal degree, and try
        to find the best one at the end.
        """
        c, d = a, b
        steps = 0

        maxdeg = a.total_degree() + b.total_degree()
        if quick:
            bound = maxdeg - 1
        else:
            bound = maxdeg
        while N + D <= bound:
            if (N, D) in tested:
                break
            tested.add((N, D))

            M1 = staircase(N)
            M2 = staircase(D)
            debugf('%s / %s: %s, %s', (N, D, M1, M2))

            Cs = symbols("c:%d" % len(M1), cls=Dummy)
            Ds = symbols("d:%d" % len(M2), cls=Dummy)
            ng = Cs + Ds

            c_hat = Poly(
                sum(Cs[i] * M1[i] for i in range(len(M1))), opt.gens + ng)
            d_hat = Poly(
                sum(Ds[i] * M2[i] for i in range(len(M2))), opt.gens + ng)

            r = reduced(a * d_hat - b * c_hat, G, opt.gens + ng,
                        order=opt.order, polys=True)[1]

            S = Poly(r, gens=opt.gens).coeffs()
            sol = solve(S, Cs + Ds, particular=True, quick=True)

            if sol and not all(s == 0 for s in sol.values()):
                c = c_hat.subs(sol)
                d = d_hat.subs(sol)

                # The "free" variables occurring before as parameters
                # might still be in the substituted c, d, so set them
                # to the value chosen before:
                c = c.subs(dict(list(zip(Cs + Ds, [1] * (len(Cs) + len(Ds))))))
                d = d.subs(dict(list(zip(Cs + Ds, [1] * (len(Cs) + len(Ds))))))

                c = Poly(c, opt.gens)
                d = Poly(d, opt.gens)
                if d == 0:
                    raise ValueError('Ideal not prime?')

                allsol.append((c_hat, d_hat, S, Cs + Ds))
                if N + D != maxdeg:
                    allsol = [allsol[-1]]

                break

            steps += 1
            N += 1
            D += 1

        if steps > 0:
            c, d, allsol = _ratsimpmodprime(c, d, allsol, N, D - steps)
            c, d, allsol = _ratsimpmodprime(c, d, allsol, N - steps, D)

        return c, d, allsol

    # preprocessing. this improves performance a bit when deg(num)
    # and deg(denom) are large:
    num = reduced(num, G, opt.gens, order=opt.order)[1]
    denom = reduced(denom, G, opt.gens, order=opt.order)[1]

    if polynomial:
        return (num/denom).cancel()

    c, d, allsol = _ratsimpmodprime(
        Poly(num, opt.gens, domain=opt.domain), Poly(denom, opt.gens, domain=opt.domain), [])
    if not quick and allsol:
        debugf('Looking for best minimal solution. Got: %s', len(allsol))
        newsol = []
        for c_hat, d_hat, S, ng in allsol:
            sol = solve(S, ng, particular=True, quick=False)
            # all values of sol should be numbers; if not, solve is broken
            newsol.append((c_hat.subs(sol), d_hat.subs(sol)))
        c, d = min(newsol, key=lambda x: len(x[0].terms()) + len(x[1].terms()))

    if not domain.is_Field:
        cn, c = c.clear_denoms(convert=True)
        dn, d = d.clear_denoms(convert=True)
        r = Rational(cn, dn)
    else:
        r = Rational(1)

    return (c*r.q)/(d*r.p)
<|endoftext|>""" Optimizations of the expression tree representation for better CSE
opportunities.
"""
from sympy.core import Add, Basic, Mul
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.traversal import preorder_traversal


def sub_pre(e):
    """ Replace y - x with -(x - y) if -1 can be extracted from y - x.
    """
    # replacing Add, A, from which -1 can be extracted with -1*-A
    adds = [a for a in e.atoms(Add) if a.could_extract_minus_sign()]
    reps = {}
    ignore = set()
    for a in adds:
        na = -a
        if na.is_Mul:  # e.g. MatExpr
            ignore.add(a)
            continue
        reps[a] = Mul._from_args([S.NegativeOne, na])

    e = e.xreplace(reps)

    # repeat again for persisting Adds but mark these with a leading 1, -1
    # e.g. y - x -> 1*-1*(x - y)
    if isinstance(e, Basic):
        negs = {}
        for a in sorted(e.atoms(Add), key=default_sort_key):
            if a in ignore:
                continue
            if a in reps:
                negs[a] = reps[a]
            elif a.could_extract_minus_sign():
                negs[a] = Mul._from_args([S.One, S.NegativeOne, -a])
        e = e.xreplace(negs)
    return e


def sub_post(e):
    """ Replace 1*-1*x with -x.
    """
    replacements = []
    for node in preorder_traversal(e):
        if isinstance(node, Mul) and \
            node.args[0] is S.One and node.args[1] is S.NegativeOne:
            replacements.append((node, -Mul._from_args(node.args[2:])))
    for node, replacement in replacements:
        e = e.xreplace({node: replacement})

    return e
<|endoftext|>""" Tools for doing common subexpression elimination.
"""
from collections import defaultdict

from sympy.core import Basic, Mul, Add, Pow, sympify
from sympy.core.containers import Tuple, OrderedSet
from sympy.core.exprtools import factor_terms
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import symbols, Symbol
from sympy.matrices import (MatrixBase, Matrix, ImmutableMatrix,
                            SparseMatrix, ImmutableSparseMatrix)
from sympy.matrices.expressions import (MatrixExpr, MatrixSymbol, MatMul,
                                        MatAdd, MatPow, Inverse)
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.polys.rootoftools import RootOf
from sympy.utilities.iterables import numbered_symbols, sift, \
        topological_sort, iterable

from . import cse_opts

# (preprocessor, postprocessor) pairs which are commonly useful. They should
# each take a SymPy expression and return a possibly transformed expression.
# When used in the function ``cse()``, the target expressions will be transformed
# by each of the preprocessor functions in order. After the common
# subexpressions are eliminated, each resulting expression will have the
# postprocessor functions transform them in *reverse* order in order to undo the
# transformation if necessary. This allows the algorithm to operate on
# a representation of the expressions that allows for more optimization
# opportunities.
# ``None`` can be used to specify no transformation for either the preprocessor or
# postprocessor.


basic_optimizations = [(cse_opts.sub_pre, cse_opts.sub_post),
                       (factor_terms, None)]

# sometimes we want the output in a different format; non-trivial
# transformations can be put here for users
# ===============================================================


def reps_toposort(r):
    """Sort replacements ``r`` so (k1, v1) appears before (k2, v2)
    if k2 is in v1's free symbols. This orders items in the
    way that cse returns its results (hence, in order to use the
    replacements in a substitution option it would make sense
    to reverse the order).

    Examples
    ========

    >>> from sympy.simplify.cse_main import reps_toposort
    >>> from sympy.abc import x, y
    >>> from sympy import Eq
    >>> for l, r in reps_toposort([(x, y + 1), (y, 2)]):
    ...     print(Eq(l, r))
    ...
    Eq(y, 2)
    Eq(x, y + 1)

    """
    r = sympify(r)
    E = []
    for c1, (k1, v1) in enumerate(r):
        for c2, (k2, v2) in enumerate(r):
            if k1 in v2.free_symbols:
                E.append((c1, c2))
    return [r[i] for i in topological_sort((range(len(r)), E))]


def cse_separate(r, e):
    """Move expressions that are in the form (symbol, expr) out of the
    expressions and sort them into the replacements using the reps_toposort.

    Examples
    ========

    >>> from sympy.simplify.cse_main import cse_separate
    >>> from sympy.abc import x, y, z
    >>> from sympy import cos, exp, cse, Eq, symbols
    >>> x0, x1 = symbols('x:2')
    >>> eq = (x + 1 + exp((x + 1)/(y + 1)) + cos(y + 1))
    >>> cse([eq, Eq(x, z + 1), z - 2], postprocess=cse_separate) in [
    ... [[(x0, y + 1), (x, z + 1), (x1, x + 1)],
    ...  [x1 + exp(x1/x0) + cos(x0), z - 2]],
    ... [[(x1, y + 1), (x, z + 1), (x0, x + 1)],
    ...  [x0 + exp(x0/x1) + cos(x1), z - 2]]]
    ...
    True
    """
    d = sift(e, lambda w: w.is_Equality and w.lhs.is_Symbol)
    r = r + [w.args for w in d[True]]
    e = d[False]
    return [reps_toposort(r), e]


def cse_release_variables(r, e):
    """
    Return tuples giving ``(a, b)`` where ``a`` is a symbol and ``b`` is
    either an expression or None. The value of None is used when a
    symbol is no longer needed for subsequent expressions.

    Use of such output can reduce the memory footprint of lambdified
    expressions that contain large, repeated subexpressions.

    Examples
    ========

    >>> from sympy import cse
    >>> from sympy.simplify.cse_main import cse_release_variables
    >>> from sympy.abc import x, y
    >>> eqs = [(x + y - 1)**2, x, x + y, (x + y)/(2*x + 1) + (x + y - 1)**2, (2*x + 1)**(x + y)]
    >>> defs, rvs = cse_release_variables(*cse(eqs))
    >>> for i in defs:
    ...   print(i)
    ...
    (x0, x + y)
    (x1, (x0 - 1)**2)
    (x2, 2*x + 1)
    (_3, x0/x2 + x1)
    (_4, x2**x0)
    (x2, None)
    (_0, x1)
    (x1, None)
    (_2, x0)
    (x0, None)
    (_1, x)
    >>> print(rvs)
    (_0, _1, _2, _3, _4)
    """
    if not r:
        return r, e

    s, p = zip(*r)
    esyms = symbols('_:%d' % len(e))
    syms = list(esyms)
    s = list(s)
    in_use = set(s)
    p = list(p)
    # sort e so those with most sub-expressions appear first
    e = [(e[i], syms[i]) for i in range(len(e))]
    e, syms = zip(*sorted(e,
        key=lambda x: -sum(p[s.index(i)].count_ops()
        for i in x[0].free_symbols & in_use)))
    syms = list(syms)
    p += e
    rv = []
    i = len(p) - 1
    while i >= 0:
        _p = p.pop()
        c = in_use & _p.free_symbols
        if c: # sorting for canonical results
            rv.extend([(s, None) for s in sorted(c, key=str)])
        if i >= len(r):
            rv.append((syms.pop(), _p))
        else:
            rv.append((s[i], _p))
        in_use -= c
        i -= 1
    rv.reverse()
    return rv, esyms


# ====end of cse postprocess idioms===========================


def preprocess_for_cse(expr, optimizations):
    """ Preprocess an expression to optimize for common subexpression
    elimination.

    Parameters
    ==========

    expr : SymPy expression
        The target expression to optimize.
    optimizations : list of (callable, callable) pairs
        The (preprocessor, postprocessor) pairs.

    Returns
    =======

    expr : SymPy expression
        The transformed expression.
    """
    for pre, post in optimizations:
        if pre is not None:
            expr = pre(expr)
    return expr


def postprocess_for_cse(expr, optimizations):
    """Postprocess an expression after common subexpression elimination to
    return the expression to canonical SymPy form.

    Parameters
    ==========

    expr : SymPy expression
        The target expression to transform.
    optimizations : list of (callable, callable) pairs, optional
        The (preprocessor, postprocessor) pairs.  The postprocessors will be
        applied in reversed order to undo the effects of the preprocessors
        correctly.

    Returns
    =======

    expr : SymPy expression
        The transformed expression.
    """
    for pre, post in reversed(optimizations):
        if post is not None:
            expr = post(expr)
    return expr


class FuncArgTracker:
    """
    A class which manages a mapping from functions to arguments and an inverse
    mapping from arguments to functions.
    """

    def __init__(self, funcs):
        # To minimize the number of symbolic comparisons, all function arguments
        # get assigned a value number.
        self.value_numbers = {}
        self.value_number_to_value = []

        # Both of these maps use integer indices for arguments / functions.
        self.arg_to_funcset = []
        self.func_to_argset = []

        for func_i, func in enumerate(funcs):
            func_argset = OrderedSet()

            for func_arg in func.args:
                arg_number = self.get_or_add_value_number(func_arg)
                func_argset.add(arg_number)
                self.arg_to_funcset[arg_number].add(func_i)

            self.func_to_argset.append(func_argset)

    def get_args_in_value_order(self, argset):
        """
        Return the list of arguments in sorted order according to their value
        numbers.
        """
        return [self.value_number_to_value[argn] for argn in sorted(argset)]

    def get_or_add_value_number(self, value):
        """
        Return the value number for the given argument.
        """
        nvalues = len(self.value_numbers)
        value_number = self.value_numbers.setdefault(value, nvalues)
        if value_number == nvalues:
            self.value_number_to_value.append(value)
            self.arg_to_funcset.append(OrderedSet())
        return value_number

    def stop_arg_tracking(self, func_i):
        """
        Remove the function func_i from the argument to function mapping.
        """
        for arg in self.func_to_argset[func_i]:
            self.arg_to_funcset[arg].remove(func_i)


    def get_common_arg_candidates(self, argset, min_func_i=0):
        """Return a dict whose keys are function numbers. The entries of the dict are
        the number of arguments said function has in common with
        ``argset``. Entries have at least 2 items in common.  All keys have
        value at least ``min_func_i``.
        """
        count_map = defaultdict(lambda: 0)
        if not argset:
            return count_map

        funcsets = [self.arg_to_funcset[arg] for arg in argset]
        # As an optimization below, we handle the largest funcset separately from
        # the others.
        largest_funcset = max(funcsets, key=len)

        for funcset in funcsets:
            if largest_funcset is funcset:
                continue
            for func_i in funcset:
                if func_i >= min_func_i:
                    count_map[func_i] += 1

        # We pick the smaller of the two containers (count_map, largest_funcset)
        # to iterate over to reduce the number of iterations needed.
        (smaller_funcs_container,
         larger_funcs_container) = sorted(
                 [largest_funcset, count_map],
                 key=len)

        for func_i in smaller_funcs_container:
            # Not already in count_map? It can't possibly be in the output, so
            # skip it.
            if count_map[func_i] < 1:
                continue

            if func_i in larger_funcs_container:
                count_map[func_i] += 1

        return {k: v for k, v in count_map.items() if v >= 2}

    def get_subset_candidates(self, argset, restrict_to_funcset=None):
        """
        Return a set of functions each of which whose argument list contains
        ``argset``, optionally filtered only to contain functions in
        ``restrict_to_funcset``.
        """
        iarg = iter(argset)

        indices = OrderedSet(
            fi for fi in self.arg_to_funcset[next(iarg)])

        if restrict_to_funcset is not None:
            indices &= restrict_to_funcset

        for arg in iarg:
            indices &= self.arg_to_funcset[arg]

        return indices

    def update_func_argset(self, func_i, new_argset):
        """
        Update a function with a new set of arguments.
        """
        new_args = OrderedSet(new_argset)
        old_args = self.func_to_argset[func_i]

        for deleted_arg in old_args - new_args:
            self.arg_to_funcset[deleted_arg].remove(func_i)
        for added_arg in new_args - old_args:
            self.arg_to_funcset[added_arg].add(func_i)

        self.func_to_argset[func_i].clear()
        self.func_to_argset[func_i].update(new_args)


class Unevaluated:

    def __init__(self, func, args):
        self.func = func
        self.args = args

    def __str__(self):
        return "Uneval<{}>({})".format(
                self.func, ", ".join(str(a) for a in self.args))

    def as_unevaluated_basic(self):
        return self.func(*self.args, evaluate=False)

    @property
    def free_symbols(self):
        return set().union(*[a.free_symbols for a in self.args])

    __repr__ = __str__


def match_common_args(func_class, funcs, opt_subs):
    """
    Recognize and extract common subexpressions of function arguments within a
    set of function calls. For instance, for the following function calls::

        x + z + y
        sin(x + y)

    this will extract a common subexpression of `x + y`::

        w = x + y
        w + z
        sin(w)

    The function we work with is assumed to be associative and commutative.

    Parameters
    ==========

    func_class: class
        The function class (e.g. Add, Mul)
    funcs: list of functions
        A list of function calls.
    opt_subs: dict
        A dictionary of substitutions which this function may update.
    """

    # Sort to ensure that whole-function subexpressions come before the items
    # that use them.
    funcs = sorted(funcs, key=lambda f: len(f.args))
    arg_tracker = FuncArgTracker(funcs)

    changed = OrderedSet()

    for i in range(len(funcs)):
        common_arg_candidates_counts = arg_tracker.get_common_arg_candidates(
                arg_tracker.func_to_argset[i], min_func_i=i + 1)

        # Sort the candidates in order of match size.
        # This makes us try combining smaller matches first.
        common_arg_candidates = OrderedSet(sorted(
                common_arg_candidates_counts.keys(),
                key=lambda k: (common_arg_candidates_counts[k], k)))

        while common_arg_candidates:
            j = common_arg_candidates.pop(last=False)

            com_args = arg_tracker.func_to_argset[i].intersection(
                    arg_tracker.func_to_argset[j])

            if len(com_args) <= 1:
                # This may happen if a set of common arguments was already
                # combined in a previous iteration.
                continue

            # For all sets, replace the common symbols by the function
            # over them, to allow recursive matches.

            diff_i = arg_tracker.func_to_argset[i].difference(com_args)
            if diff_i:
                # com_func needs to be unevaluated to allow for recursive matches.
                com_func = Unevaluated(
                        func_class, arg_tracker.get_args_in_value_order(com_args))
                com_func_number = arg_tracker.get_or_add_value_number(com_func)
                arg_tracker.update_func_argset(i, diff_i | OrderedSet([com_func_number]))
                changed.add(i)
            else:
                # Treat the whole expression as a CSE.
                #
                # The reason this needs to be done is somewhat subtle. Within
                # tree_cse(), to_eliminate only contains expressions that are
                # seen more than once. The problem is unevaluated expressions
                # do not compare equal to the evaluated equivalent. So
                # tree_cse() won't mark funcs[i] as a CSE if we use an
                # unevaluated version.
                com_func_number = arg_tracker.get_or_add_value_number(funcs[i])

            diff_j = arg_tracker.func_to_argset[j].difference(com_args)
            arg_tracker.update_func_argset(j, diff_j | OrderedSet([com_func_number]))
            changed.add(j)

            for k in arg_tracker.get_subset_candidates(
                    com_args, common_arg_candidates):
                diff_k = arg_tracker.func_to_argset[k].difference(com_args)
                arg_tracker.update_func_argset(k, diff_k | OrderedSet([com_func_number]))
                changed.add(k)

        if i in changed:
            opt_subs[funcs[i]] = Unevaluated(func_class,
                arg_tracker.get_args_in_value_order(arg_tracker.func_to_argset[i]))

        arg_tracker.stop_arg_tracking(i)


def opt_cse(exprs, order='canonical'):
    """Find optimization opportunities in Adds, Muls, Pows and negative
    coefficient Muls.

    Parameters
    ==========

    exprs : list of SymPy expressions
        The expressions to optimize.
    order : string, 'none' or 'canonical'
        The order by which Mul and Add arguments are processed. For large
        expressions where speed is a concern, use the setting order='none'.

    Returns
    =======

    opt_subs : dictionary of expression substitutions
        The expression substitutions which can be useful to optimize CSE.

    Examples
    ========

    >>> from sympy.simplify.cse_main import opt_cse
    >>> from sympy.abc import x
    >>> opt_subs = opt_cse([x**-2])
    >>> k, v = list(opt_subs.keys())[0], list(opt_subs.values())[0]
    >>> print((k, v.as_unevaluated_basic()))
    (x**(-2), 1/(x**2))
    """
    opt_subs = {}

    adds = OrderedSet()
    muls = OrderedSet()

    seen_subexp = set()
    collapsible_subexp = set()

    def _find_opts(expr):

        if not isinstance(expr, (Basic, Unevaluated)):
            return

        if expr.is_Atom or expr.is_Order:
            return

        if iterable(expr):
            list(map(_find_opts, expr))
            return

        if expr in seen_subexp:
            return expr
        seen_subexp.add(expr)

        list(map(_find_opts, expr.args))

        if not isinstance(expr, MatrixExpr) and expr.could_extract_minus_sign():
            # XXX -expr does not always work rigorously for some expressions
            # containing UnevaluatedExpr.
            # https://github.com/sympy/sympy/issues/24818
            if isinstance(expr, Add):
                neg_expr = Add(*(-i for i in expr.args))
            else:
                neg_expr = -expr

            if not neg_expr.is_Atom:
                opt_subs[expr] = Unevaluated(Mul, (S.NegativeOne, neg_expr))
                seen_subexp.add(neg_expr)
                expr = neg_expr

        if isinstance(expr, (Mul, MatMul)):
            if len(expr.args) == 1:
                collapsible_subexp.add(expr)
            else:
                muls.add(expr)

        elif isinstance(expr, (Add, MatAdd)):
            if len(expr.args) == 1:
                collapsible_subexp.add(expr)
            else:
                adds.add(expr)

        elif isinstance(expr, Inverse):
            # Do not want to treat `Inverse` as a `MatPow`
            pass

        elif isinstance(expr, (Pow, MatPow)):
            base, exp = expr.base, expr.exp
            if exp.could_extract_minus_sign():
                opt_subs[expr] = Unevaluated(Pow, (Pow(base, -exp), -1))

    for e in exprs:
        if isinstance(e, (Basic, Unevaluated)):
            _find_opts(e)

    # Handle collapsing of multinary operations with single arguments
    edges = [(s, s.args[0]) for s in collapsible_subexp
             if s.args[0] in collapsible_subexp]
    for e in reversed(topological_sort((collapsible_subexp, edges))):
        opt_subs[e] = opt_subs.get(e.args[0], e.args[0])

    # split muls into commutative
    commutative_muls = OrderedSet()
    for m in muls:
        c, nc = m.args_cnc(cset=False)
        if c:
            c_mul = m.func(*c)
            if nc:
                if c_mul == 1:
                    new_obj = m.func(*nc)
                else:
                    if isinstance(m, MatMul):
                        new_obj = m.func(c_mul, *nc, evaluate=False)
                    else:
                        new_obj = m.func(c_mul, m.func(*nc), evaluate=False)
                opt_subs[m] = new_obj
            if len(c) > 1:
                commutative_muls.add(c_mul)

    match_common_args(Add, adds, opt_subs)
    match_common_args(Mul, commutative_muls, opt_subs)

    return opt_subs


def tree_cse(exprs, symbols, opt_subs=None, order='canonical', ignore=()):
    """Perform raw CSE on expression tree, taking opt_subs into account.

    Parameters
    ==========

    exprs : list of SymPy expressions
        The expressions to reduce.
    symbols : infinite iterator yielding unique Symbols
        The symbols used to label the common subexpressions which are pulled
        out.
    opt_subs : dictionary of expression substitutions
        The expressions to be substituted before any CSE action is performed.
    order : string, 'none' or 'canonical'
        The order by which Mul and Add arguments are processed. For large
        expressions where speed is a concern, use the setting order='none'.
    ignore : iterable of Symbols
        Substitutions containing any Symbol from ``ignore`` will be ignored.
    """
    if opt_subs is None:
        opt_subs = {}

    ## Find repeated sub-expressions

    to_eliminate = set()

    seen_subexp = set()
    excluded_symbols = set()

    def _find_repeated(expr):
        if not isinstance(expr, (Basic, Unevaluated)):
            return

        if isinstance(expr, RootOf):
            return

        if isinstance(expr, Basic) and (
                expr.is_Atom or
                expr.is_Order or
                isinstance(expr, (MatrixSymbol, MatrixElement))):
            if expr.is_Symbol:
                excluded_symbols.add(expr.name)
            return

        if iterable(expr):
            args = expr

        else:
            if expr in seen_subexp:
                for ign in ignore:
                    if ign in expr.free_symbols:
                        break
                else:
                    to_eliminate.add(expr)
                    return

            seen_subexp.add(expr)

            if expr in opt_subs:
                expr = opt_subs[expr]

            args = expr.args

        list(map(_find_repeated, args))

    for e in exprs:
        if isinstance(e, Basic):
            _find_repeated(e)

    ## Rebuild tree

    # Remove symbols from the generator that conflict with names in the expressions.
    symbols = (_ for _ in symbols if _.name not in excluded_symbols)

    replacements = []

    subs = {}

    def _rebuild(expr):
        if not isinstance(expr, (Basic, Unevaluated)):
            return expr

        if not expr.args:
            return expr

        if iterable(expr):
            new_args = [_rebuild(arg) for arg in expr.args]
            return expr.func(*new_args)

        if expr in subs:
            return subs[expr]

        orig_expr = expr
        if expr in opt_subs:
            expr = opt_subs[expr]

        # If enabled, parse Muls and Adds arguments by order to ensure
        # replacement order independent from hashes
        if order != 'none':
            if isinstance(expr, (Mul, MatMul)):
                c, nc = expr.args_cnc()
                if c == [1]:
                    args = nc
                else:
                    args = list(ordered(c)) + nc
            elif isinstance(expr, (Add, MatAdd)):
                args = list(ordered(expr.args))
            else:
                args = expr.args
        else:
            args = expr.args

        new_args = list(map(_rebuild, args))
        if isinstance(expr, Unevaluated) or new_args != args:
            new_expr = expr.func(*new_args)
        else:
            new_expr = expr

        if orig_expr in to_eliminate:
            try:
                sym = next(symbols)
            except StopIteration:
                raise ValueError("Symbols iterator ran out of symbols.")

            if isinstance(orig_expr, MatrixExpr):
                sym = MatrixSymbol(sym.name, orig_expr.rows,
                    orig_expr.cols)

            subs[orig_expr] = sym
            replacements.append((sym, new_expr))
            return sym

        else:
            return new_expr

    reduced_exprs = []
    for e in exprs:
        if isinstance(e, Basic):
            reduced_e = _rebuild(e)
        else:
            reduced_e = e
        reduced_exprs.append(reduced_e)
    return replacements, reduced_exprs


def cse(exprs, symbols=None, optimizations=None, postprocess=None,
        order='canonical', ignore=(), list=True):
    """ Perform common subexpression elimination on an expression.

    Parameters
    ==========

    exprs : list of SymPy expressions, or a single SymPy expression
        The expressions to reduce.
    symbols : infinite iterator yielding unique Symbols
        The symbols used to label the common subexpressions which are pulled
        out. The ``numbered_symbols`` generator is useful. The default is a
        stream of symbols of the form "x0", "x1", etc. This must be an
        infinite iterator.
    optimizations : list of (callable, callable) pairs
        The (preprocessor, postprocessor) pairs of external optimization
        functions. Optionally 'basic' can be passed for a set of predefined
        basic optimizations. Such 'basic' optimizations were used by default
        in old implementation, however they can be really slow on larger
        expressions. Now, no pre or post optimizations are made by default.
    postprocess : a function which accepts the two return values of cse and
        returns the desired form of output from cse, e.g. if you want the
        replacements reversed the function might be the following lambda:
        lambda r, e: return reversed(r), e
    order : string, 'none' or 'canonical'
        The order by which Mul and Add arguments are processed. If set to
        'canonical', arguments will be canonically ordered. If set to 'none',
        ordering will be faster but dependent on expressions hashes, thus
        machine dependent and variable. For large expressions where speed is a
        concern, use the setting order='none'.
    ignore : iterable of Symbols
        Substitutions containing any Symbol from ``ignore`` will be ignored.
    list : bool, (default True)
        Returns expression in list or else with same type as input (when False).

    Returns
    =======

    replacements : list of (Symbol, expression) pairs
        All of the common subexpressions that were replaced. Subexpressions
        earlier in this list might show up in subexpressions later in this
        list.
    reduced_exprs : list of SymPy expressions
        The reduced expressions with all of the replacements above.

    Examples
    ========

    >>> from sympy import cse, SparseMatrix
    >>> from sympy.abc import x, y, z, w
    >>> cse(((w + x + y + z)*(w + y + z))/(w + x)**3)
    ([(x0, y + z), (x1, w + x)], [(w + x0)*(x0 + x1)/x1**3])


    List of expressions with recursive substitutions:

    >>> m = SparseMatrix([x + y, x + y + z])
    >>> cse([(x+y)**2, x + y + z, y + z, x + z + y, m])
    ([(x0, x + y), (x1, x0 + z)], [x0**2, x1, y + z, x1, Matrix([
    [x0],
    [x1]])])

    Note: the type and mutability of input matrices is retained.

    >>> isinstance(_[1][-1], SparseMatrix)
    True

    The user may disallow substitutions containing certain symbols:

    >>> cse([y**2*(x + 1), 3*y**2*(x + 1)], ignore=(y,))
    ([(x0, x + 1)], [x0*y**2, 3*x0*y**2])

    The default return value for the reduced expression(s) is a list, even if there is only
    one expression. The `list` flag preserves the type of the input in the output:

    >>> cse(x)
    ([], [x])
    >>> cse(x, list=False)
    ([], x)
    """
    if not list:
        return _cse_homogeneous(exprs,
            symbols=symbols, optimizations=optimizations,
            postprocess=postprocess, order=order, ignore=ignore)

    if isinstance(exprs, (int, float)):
        exprs = sympify(exprs)

    # Handle the case if just one expression was passed.
    if isinstance(exprs, (Basic, MatrixBase)):
        exprs = [exprs]

    copy = exprs
    temp = []
    for e in exprs:
        if isinstance(e, (Matrix, ImmutableMatrix)):
            temp.append(Tuple(*e.flat()))
        elif isinstance(e, (SparseMatrix, ImmutableSparseMatrix)):
            temp.append(Tuple(*e.todok().items()))
        else:
            temp.append(e)
    exprs = temp
    del temp

    if optimizations is None:
        optimizations = []
    elif optimizations == 'basic':
        optimizations = basic_optimizations

    # Preprocess the expressions to give us better optimization opportunities.
    reduced_exprs = [preprocess_for_cse(e, optimizations) for e in exprs]

    if symbols is None:
        symbols = numbered_symbols(cls=Symbol)
    else:
        # In case we get passed an iterable with an __iter__ method instead of
        # an actual iterator.
        symbols = iter(symbols)

    # Find other optimization opportunities.
    opt_subs = opt_cse(reduced_exprs, order)

    # Main CSE algorithm.
    replacements, reduced_exprs = tree_cse(reduced_exprs, symbols, opt_subs,
                                           order, ignore)

    # Postprocess the expressions to return the expressions to canonical form.
    exprs = copy
    replacements = [(sym, postprocess_for_cse(subtree, optimizations))
                    for sym, subtree in replacements]
    reduced_exprs = [postprocess_for_cse(e, optimizations)
                     for e in reduced_exprs]

    # Get the matrices back
    for i, e in enumerate(exprs):
        if isinstance(e, (Matrix, ImmutableMatrix)):
            reduced_exprs[i] = Matrix(e.rows, e.cols, reduced_exprs[i])
            if isinstance(e, ImmutableMatrix):
                reduced_exprs[i] = reduced_exprs[i].as_immutable()
        elif isinstance(e, (SparseMatrix, ImmutableSparseMatrix)):
            m = SparseMatrix(e.rows, e.cols, {})
            for k, v in reduced_exprs[i]:
                m[k] = v
            if isinstance(e, ImmutableSparseMatrix):
                m = m.as_immutable()
            reduced_exprs[i] = m

    if postprocess is None:
        return replacements, reduced_exprs

    return postprocess(replacements, reduced_exprs)


def _cse_homogeneous(exprs, **kwargs):
    """
    Same as ``cse`` but the ``reduced_exprs`` are returned
    with the same type as ``exprs`` or a sympified version of the same.

    Parameters
    ==========

    exprs : an Expr, iterable of Expr or dictionary with Expr values
        the expressions in which repeated subexpressions will be identified
    kwargs : additional arguments for the ``cse`` function

    Returns
    =======

    replacements : list of (Symbol, expression) pairs
        All of the common subexpressions that were replaced. Subexpressions
        earlier in this list might show up in subexpressions later in this
        list.
    reduced_exprs : list of SymPy expressions
        The reduced expressions with all of the replacements above.

    Examples
    ========

    >>> from sympy.simplify.cse_main import cse
    >>> from sympy import cos, Tuple, Matrix
    >>> from sympy.abc import x
    >>> output = lambda x: type(cse(x, list=False)[1])
    >>> output(1)
    <class 'sympy.core.numbers.One'>
    >>> output('cos(x)')
    <class 'str'>
    >>> output(cos(x))
    cos
    >>> output(Tuple(1, x))
    <class 'sympy.core.containers.Tuple'>
    >>> output(Matrix([[1,0], [0,1]]))
    <class 'sympy.matrices.dense.MutableDenseMatrix'>
    >>> output([1, x])
    <class 'list'>
    >>> output((1, x))
    <class 'tuple'>
    >>> output({1, x})
    <class 'set'>
    """
    if isinstance(exprs, str):
        replacements, reduced_exprs = _cse_homogeneous(
            sympify(exprs), **kwargs)
        return replacements, repr(reduced_exprs)
    if isinstance(exprs, (list, tuple, set)):
        replacements, reduced_exprs = cse(exprs, **kwargs)
        return replacements, type(exprs)(reduced_exprs)
    if isinstance(exprs, dict):
        keys = list(exprs.keys()) # In order to guarantee the order of the elements.
        replacements, values = cse([exprs[k] for k in keys], **kwargs)
        reduced_exprs = dict(zip(keys, values))
        return replacements, reduced_exprs

    try:
        replacements, (reduced_exprs,) = cse(exprs, **kwargs)
    except TypeError: # For example 'mpf' objects
        return [], exprs
    else:
        return replacements, reduced_exprs
<|endoftext|>"""The module helps converting SymPy expressions into shorter forms of them.

for example:
the expression E**(pi*I) will be converted into -1
the expression (x+x)**2 will be converted into 4*x**2
"""
from .simplify import (simplify, hypersimp, hypersimilar,
    logcombine, separatevars, posify, besselsimp, kroneckersimp,
    signsimp, nsimplify)

from .fu import FU, fu

from .sqrtdenest import sqrtdenest

from .cse_main import cse

from .epathtools import epath, EPath

from .hyperexpand import hyperexpand

from .radsimp import collect, rcollect, radsimp, collect_const, fraction, numer, denom

from .trigsimp import trigsimp, exptrigsimp

from .powsimp import powsimp, powdenest

from .combsimp import combsimp

from .gammasimp import gammasimp

from .ratsimp import ratsimp, ratsimpmodprime

__all__ = [
    'simplify', 'hypersimp', 'hypersimilar', 'logcombine', 'separatevars',
    'posify', 'besselsimp', 'kroneckersimp', 'signsimp',
    'nsimplify',

    'FU', 'fu',

    'sqrtdenest',

    'cse',

    'epath', 'EPath',

    'hyperexpand',

    'collect', 'rcollect', 'radsimp', 'collect_const', 'fraction', 'numer',
    'denom',

    'trigsimp', 'exptrigsimp',

    'powsimp', 'powdenest',

    'combsimp',

    'gammasimp',

    'ratsimp', 'ratsimpmodprime',
]
<|endoftext|>from sympy.core import Function, S, Mul, Pow, Add
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.function import expand_func
from sympy.core.symbol import Dummy
from sympy.functions import gamma, sqrt, sin
from sympy.polys import factor, cancel
from sympy.utilities.iterables import sift, uniq


def gammasimp(expr):
    r"""
    Simplify expressions with gamma functions.

    Explanation
    ===========

    This function takes as input an expression containing gamma
    functions or functions that can be rewritten in terms of gamma
    functions and tries to minimize the number of those functions and
    reduce the size of their arguments.

    The algorithm works by rewriting all gamma functions as expressions
    involving rising factorials (Pochhammer symbols) and applies
    recurrence relations and other transformations applicable to rising
    factorials, to reduce their arguments, possibly letting the resulting
    rising factorial to cancel. Rising factorials with the second argument
    being an integer are expanded into polynomial forms and finally all
    other rising factorial are rewritten in terms of gamma functions.

    Then the following two steps are performed.

    1. Reduce the number of gammas by applying the reflection theorem
       gamma(x)*gamma(1-x) == pi/sin(pi*x).
    2. Reduce the number of gammas by applying the multiplication theorem
       gamma(x)*gamma(x+1/n)*...*gamma(x+(n-1)/n) == C*gamma(n*x).

    It then reduces the number of prefactors by absorbing them into gammas
    where possible and expands gammas with rational argument.

    All transformation rules can be found (or were derived from) here:

    .. [1] https://functions.wolfram.com/GammaBetaErf/Pochhammer/17/01/02/
    .. [2] https://functions.wolfram.com/GammaBetaErf/Pochhammer/27/01/0005/

    Examples
    ========

    >>> from sympy.simplify import gammasimp
    >>> from sympy import gamma, Symbol
    >>> from sympy.abc import x
    >>> n = Symbol('n', integer = True)

    >>> gammasimp(gamma(x)/gamma(x - 3))
    (x - 3)*(x - 2)*(x - 1)
    >>> gammasimp(gamma(n + 3))
    gamma(n + 3)

    """

    expr = expr.rewrite(gamma)

    # compute_ST will be looking for Functions and we don't want
    # it looking for non-gamma functions: issue 22606
    # so we mask free, non-gamma functions
    f = expr.atoms(Function)
    # take out gammas
    gammas = {i for i in f if isinstance(i, gamma)}
    if not gammas:
        return expr  # avoid side effects like factoring
    f -= gammas
    # keep only those without bound symbols
    f = f & expr.as_dummy().atoms(Function)
    if f:
        dum, fun, simp = zip(*[
            (Dummy(), fi, fi.func(*[
                _gammasimp(a, as_comb=False) for a in fi.args]))
            for fi in ordered(f)])
        d = expr.xreplace(dict(zip(fun, dum)))
        return _gammasimp(d, as_comb=False).xreplace(dict(zip(dum, simp)))

    return _gammasimp(expr, as_comb=False)


def _gammasimp(expr, as_comb):
    """
    Helper function for gammasimp and combsimp.

    Explanation
    ===========

    Simplifies expressions written in terms of gamma function. If
    as_comb is True, it tries to preserve integer arguments. See
    docstring of gammasimp for more information. This was part of
    combsimp() in combsimp.py.
    """
    expr = expr.replace(gamma,
        lambda n: _rf(1, (n - 1).expand()))

    if as_comb:
        expr = expr.replace(_rf,
            lambda a, b: gamma(b + 1))
    else:
        expr = expr.replace(_rf,
            lambda a, b: gamma(a + b)/gamma(a))

    def rule_gamma(expr, level=0):
        """ Simplify products of gamma functions further. """

        if expr.is_Atom:
            return expr

        def gamma_rat(x):
            # helper to simplify ratios of gammas
            was = x.count(gamma)
            xx = x.replace(gamma, lambda n: _rf(1, (n - 1).expand()
                ).replace(_rf, lambda a, b: gamma(a + b)/gamma(a)))
            if xx.count(gamma) < was:
                x = xx
            return x

        def gamma_factor(x):
            # return True if there is a gamma factor in shallow args
            if isinstance(x, gamma):
                return True
            if x.is_Add or x.is_Mul:
                return any(gamma_factor(xi) for xi in x.args)
            if x.is_Pow and (x.exp.is_integer or x.base.is_positive):
                return gamma_factor(x.base)
            return False

        # recursion step
        if level == 0:
            expr = expr.func(*[rule_gamma(x, level + 1) for x in expr.args])
            level += 1

        if not expr.is_Mul:
            return expr

        # non-commutative step
        if level == 1:
            args, nc = expr.args_cnc()
            if not args:
                return expr
            if nc:
                return rule_gamma(Mul._from_args(args), level + 1)*Mul._from_args(nc)
            level += 1

        # pure gamma handling, not factor absorption
        if level == 2:
            T, F = sift(expr.args, gamma_factor, binary=True)
            gamma_ind = Mul(*F)
            d = Mul(*T)

            nd, dd = d.as_numer_denom()
            for ipass in range(2):
                args = list(ordered(Mul.make_args(nd)))
                for i, ni in enumerate(args):
                    if ni.is_Add:
                        ni, dd = Add(*[
                            rule_gamma(gamma_rat(a/dd), level + 1) for a in ni.args]
                            ).as_numer_denom()
                        args[i] = ni
                        if not dd.has(gamma):
                            break
                nd = Mul(*args)
                if ipass ==  0 and not gamma_factor(nd):
                    break
                nd, dd = dd, nd  # now process in reversed order
            expr = gamma_ind*nd/dd
            if not (expr.is_Mul and (gamma_factor(dd) or gamma_factor(nd))):
                return expr
            level += 1

        # iteration until constant
        if level == 3:
            while True:
                was = expr
                expr = rule_gamma(expr, 4)
                if expr == was:
                    return expr

        numer_gammas = []
        denom_gammas = []
        numer_others = []
        denom_others = []
        def explicate(p):
            if p is S.One:
                return None, []
            b, e = p.as_base_exp()
            if e.is_Integer:
                if isinstance(b, gamma):
                    return True, [b.args[0]]*e
                else:
                    return False, [b]*e
            else:
                return False, [p]

        newargs = list(ordered(expr.args))
        while newargs:
            n, d = newargs.pop().as_numer_denom()
            isg, l = explicate(n)
            if isg:
                numer_gammas.extend(l)
            elif isg is False:
                numer_others.extend(l)
            isg, l = explicate(d)
            if isg:
                denom_gammas.extend(l)
            elif isg is False:
                denom_others.extend(l)

        # =========== level 2 work: pure gamma manipulation =========

        if not as_comb:
            # Try to reduce the number of gamma factors by applying the
            # reflection formula gamma(x)*gamma(1-x) = pi/sin(pi*x)
            for gammas, numer, denom in [(
                numer_gammas, numer_others, denom_others),
                    (denom_gammas, denom_others, numer_others)]:
                new = []
                while gammas:
                    g1 = gammas.pop()
                    if g1.is_integer:
                        new.append(g1)
                        continue
                    for i, g2 in enumerate(gammas):
                        n = g1 + g2 - 1
                        if not n.is_Integer:
                            continue
                        numer.append(S.Pi)
                        denom.append(sin(S.Pi*g1))
                        gammas.pop(i)
                        if n > 0:
                            numer.extend(1 - g1 + k for k in range(n))
                        elif n < 0:
                            denom.extend(-g1 - k for k in range(-n))
                        break
                    else:
                        new.append(g1)
                # /!\ updating IN PLACE
                gammas[:] = new

            # Try to reduce the number of gammas by using the duplication
            # theorem to cancel an upper and lower: gamma(2*s)/gamma(s) =
            # 2**(2*s + 1)/(4*sqrt(pi))*gamma(s + 1/2). Although this could
            # be done with higher argument ratios like gamma(3*x)/gamma(x),
            # this would not reduce the number of gammas as in this case.
            for ng, dg, no, do in [(numer_gammas, denom_gammas, numer_others,
                                    denom_others),
                                   (denom_gammas, numer_gammas, denom_others,
                                    numer_others)]:

                while True:
                    for x in ng:
                        for y in dg:
                            n = x - 2*y
                            if n.is_Integer:
                                break
                        else:
                            continue
                        break
                    else:
                        break
                    ng.remove(x)
                    dg.remove(y)
                    if n > 0:
                        no.extend(2*y + k for k in range(n))
                    elif n < 0:
                        do.extend(2*y - 1 - k for k in range(-n))
                    ng.append(y + S.Half)
                    no.append(2**(2*y - 1))
                    do.append(sqrt(S.Pi))

            # Try to reduce the number of gamma factors by applying the
            # multiplication theorem (used when n gammas with args differing
            # by 1/n mod 1 are encountered).
            #
            # run of 2 with args differing by 1/2
            #
            # >>> gammasimp(gamma(x)*gamma(x+S.Half))
            # 2*sqrt(2)*2**(-2*x - 1/2)*sqrt(pi)*gamma(2*x)
            #
            # run of 3 args differing by 1/3 (mod 1)
            #
            # >>> gammasimp(gamma(x)*gamma(x+S(1)/3)*gamma(x+S(2)/3))
            # 6*3**(-3*x - 1/2)*pi*gamma(3*x)
            # >>> gammasimp(gamma(x)*gamma(x+S(1)/3)*gamma(x+S(5)/3))
            # 2*3**(-3*x - 1/2)*pi*(3*x + 2)*gamma(3*x)
            #
            def _run(coeffs):
                # find runs in coeffs such that the difference in terms (mod 1)
                # of t1, t2, ..., tn is 1/n
                u = list(uniq(coeffs))
                for i in range(len(u)):
                    dj = ([((u[j] - u[i]) % 1, j) for j in range(i + 1, len(u))])
                    for one, j in dj:
                        if one.p == 1 and one.q != 1:
                            n = one.q
                            got = [i]
                            get = list(range(1, n))
                            for d, j in dj:
                                m = n*d
                                if m.is_Integer and m in get:
                                    get.remove(m)
                                    got.append(j)
                                    if not get:
                                        break
                            else:
                                continue
                            for i, j in enumerate(got):
                                c = u[j]
                                coeffs.remove(c)
                                got[i] = c
                            return one.q, got[0], got[1:]

            def _mult_thm(gammas, numer, denom):
                # pull off and analyze the leading coefficient from each gamma arg
                # looking for runs in those Rationals

                # expr -> coeff + resid -> rats[resid] = coeff
                rats = {}
                for g in gammas:
                    c, resid = g.as_coeff_Add()
                    rats.setdefault(resid, []).append(c)

                # look for runs in Rationals for each resid
                keys = sorted(rats, key=default_sort_key)
                for resid in keys:
                    coeffs = sorted(rats[resid])
                    new = []
                    while True:
                        run = _run(coeffs)
                        if run is None:
                            break

                        # process the sequence that was found:
                        # 1) convert all the gamma functions to have the right
                        #    argument (could be off by an integer)
                        # 2) append the factors corresponding to the theorem
                        # 3) append the new gamma function

                        n, ui, other = run

                        # (1)
                        for u in other:
                            con = resid + u - 1
                            for k in range(int(u - ui)):
                                numer.append(con - k)

                        con = n*(resid + ui)  # for (2) and (3)

                        # (2)
                        numer.append((2*S.Pi)**(S(n - 1)/2)*
                                     n**(S.Half - con))
                        # (3)
                        new.append(con)

                    # restore resid to coeffs
                    rats[resid] = [resid + c for c in coeffs] + new

                # rebuild the gamma arguments
                g = []
                for resid in keys:
                    g += rats[resid]
                # /!\ updating IN PLACE
                gammas[:] = g

            for l, numer, denom in [(numer_gammas, numer_others, denom_others),
                                    (denom_gammas, denom_others, numer_others)]:
                _mult_thm(l, numer, denom)

        # =========== level >= 2 work: factor absorption =========

        if level >= 2:
            # Try to absorb factors into the gammas: x*gamma(x) -> gamma(x + 1)
            # and gamma(x)/(x - 1) -> gamma(x - 1)
            # This code (in particular repeated calls to find_fuzzy) can be very
            # slow.
            def find_fuzzy(l, x):
                if not l:
                    return
                S1, T1 = compute_ST(x)
                for y in l:
                    S2, T2 = inv[y]
                    if T1 != T2 or (not S1.intersection(S2) and
                                    (S1 != set() or S2 != set())):
                        continue
                    # XXX we want some simplification (e.g. cancel or
                    # simplify) but no matter what it's slow.
                    a = len(cancel(x/y).free_symbols)
                    b = len(x.free_symbols)
                    c = len(y.free_symbols)
                    # TODO is there a better heuristic?
                    if a == 0 and (b > 0 or c > 0):
                        return y

            # We thus try to avoid expensive calls by building the following
            # "invariants": For every factor or gamma function argument
            #   - the set of free symbols S
            #   - the set of functional components T
            # We will only try to absorb if T1==T2 and (S1 intersect S2 != emptyset
            # or S1 == S2 == emptyset)
            inv = {}

            def compute_ST(expr):
                if expr in inv:
                    return inv[expr]
                return (expr.free_symbols, expr.atoms(Function).union(
                        {e.exp for e in expr.atoms(Pow)}))

            def update_ST(expr):
                inv[expr] = compute_ST(expr)
            for expr in numer_gammas + denom_gammas + numer_others + denom_others:
                update_ST(expr)

            for gammas, numer, denom in [(
                numer_gammas, numer_others, denom_others),
                    (denom_gammas, denom_others, numer_others)]:
                new = []
                while gammas:
                    g = gammas.pop()
                    cont = True
                    while cont:
                        cont = False
                        y = find_fuzzy(numer, g)
                        if y is not None:
                            numer.remove(y)
                            if y != g:
                                numer.append(y/g)
                                update_ST(y/g)
                            g += 1
                            cont = True
                        y = find_fuzzy(denom, g - 1)
                        if y is not None:
                            denom.remove(y)
                            if y != g - 1:
                                numer.append((g - 1)/y)
                                update_ST((g - 1)/y)
                            g -= 1
                            cont = True
                    new.append(g)
                # /!\ updating IN PLACE
                gammas[:] = new

        # =========== rebuild expr ==================================

        return Mul(*[gamma(g) for g in numer_gammas]) \
            / Mul(*[gamma(g) for g in denom_gammas]) \
            * Mul(*numer_others) / Mul(*denom_others)

    was = factor(expr)
    # (for some reason we cannot use Basic.replace in this case)
    expr = rule_gamma(was)
    if expr != was:
        expr = factor(expr)

    expr = expr.replace(gamma,
        lambda n: expand_func(gamma(n)) if n.is_Rational else gamma(n))

    return expr


class _rf(Function):
    @classmethod
    def eval(cls, a, b):
        if b.is_Integer:
            if not b:
                return S.One

            n = int(b)

            if n > 0:
                return Mul(*[a + i for i in range(n)])
            elif n < 0:
                return 1/Mul(*[a - i for i in range(1, -n + 1)])
        else:
            if b.is_Add:
                c, _b = b.as_coeff_Add()

                if c.is_Integer:
                    if c > 0:
                        return _rf(a, _b)*_rf(a + _b, c)
                    elif c < 0:
                        return _rf(a, _b)/_rf(a + _b + c, -c)

            if a.is_Add:
                c, _a = a.as_coeff_Add()

                if c.is_Integer:
                    if c > 0:
                        return _rf(_a, b)*_rf(_a + b, c)/_rf(_a, c)
                    elif c < 0:
                        return _rf(_a, b)*_rf(_a + c, -c)/_rf(_a + b + c, -c)
<|endoftext|>""" This module cooks up a docstring when imported. Its only purpose is to
    be displayed in the sphinx documentation. """

from sympy.core.relational import Eq
from sympy.functions.special.hyper import hyper
from sympy.printing.latex import latex
from sympy.simplify.hyperexpand import FormulaCollection

c = FormulaCollection()

doc = ""

for f in c.formulae:
    obj = Eq(hyper(f.func.ap, f.func.bq, f.z),
             f.closed_form.rewrite('nonrepsmall'))
    doc += ".. math::\n  %s\n" % latex(obj)

__doc__ = doc
<|endoftext|>"""Tools for manipulation of expressions using paths. """

from sympy.core import Basic


class EPath:
    r"""
    Manipulate expressions using paths.

    EPath grammar in EBNF notation::

        literal   ::= /[A-Za-z_][A-Za-z_0-9]*/
        number    ::= /-?\d+/
        type      ::= literal
        attribute ::= literal "?"
        all       ::= "*"
        slice     ::= "[" number? (":" number? (":" number?)?)? "]"
        range     ::= all | slice
        query     ::= (type | attribute) ("|" (type | attribute))*
        selector  ::= range | query range?
        path      ::= "/" selector ("/" selector)*

    See the docstring of the epath() function.

    """

    __slots__ = ("_path", "_epath")

    def __new__(cls, path):
        """Construct new EPath. """
        if isinstance(path, EPath):
            return path

        if not path:
            raise ValueError("empty EPath")

        _path = path

        if path[0] == '/':
            path = path[1:]
        else:
            raise NotImplementedError("non-root EPath")

        epath = []

        for selector in path.split('/'):
            selector = selector.strip()

            if not selector:
                raise ValueError("empty selector")

            index = 0

            for c in selector:
                if c.isalnum() or c in ('_', '|', '?'):
                    index += 1
                else:
                    break

            attrs = []
            types = []

            if index:
                elements = selector[:index]
                selector = selector[index:]

                for element in elements.split('|'):
                    element = element.strip()

                    if not element:
                        raise ValueError("empty element")

                    if element.endswith('?'):
                        attrs.append(element[:-1])
                    else:
                        types.append(element)

            span = None

            if selector == '*':
                pass
            else:
                if selector.startswith('['):
                    try:
                        i = selector.index(']')
                    except ValueError:
                        raise ValueError("expected ']', got EOL")

                    _span, span = selector[1:i], []

                    if ':' not in _span:
                        span = int(_span)
                    else:
                        for elt in _span.split(':', 3):
                            if not elt:
                                span.append(None)
                            else:
                                span.append(int(elt))

                        span = slice(*span)

                    selector = selector[i + 1:]

                if selector:
                    raise ValueError("trailing characters in selector")

            epath.append((attrs, types, span))

        obj = object.__new__(cls)

        obj._path = _path
        obj._epath = epath

        return obj

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self._path)

    def _get_ordered_args(self, expr):
        """Sort ``expr.args`` using printing order. """
        if expr.is_Add:
            return expr.as_ordered_terms()
        elif expr.is_Mul:
            return expr.as_ordered_factors()
        else:
            return expr.args

    def _hasattrs(self, expr, attrs) -> bool:
        """Check if ``expr`` has any of ``attrs``. """
        return all(hasattr(expr, attr) for attr in attrs)

    def _hastypes(self, expr, types):
        """Check if ``expr`` is any of ``types``. """
        _types = [ cls.__name__ for cls in expr.__class__.mro() ]
        return bool(set(_types).intersection(types))

    def _has(self, expr, attrs, types):
        """Apply ``_hasattrs`` and ``_hastypes`` to ``expr``. """
        if not (attrs or types):
            return True

        if attrs and self._hasattrs(expr, attrs):
            return True

        if types and self._hastypes(expr, types):
            return True

        return False

    def apply(self, expr, func, args=None, kwargs=None):
        """
        Modify parts of an expression selected by a path.

        Examples
        ========

        >>> from sympy.simplify.epathtools import EPath
        >>> from sympy import sin, cos, E
        >>> from sympy.abc import x, y, z, t

        >>> path = EPath("/*/[0]/Symbol")
        >>> expr = [((x, 1), 2), ((3, y), z)]

        >>> path.apply(expr, lambda expr: expr**2)
        [((x**2, 1), 2), ((3, y**2), z)]

        >>> path = EPath("/*/*/Symbol")
        >>> expr = t + sin(x + 1) + cos(x + y + E)

        >>> path.apply(expr, lambda expr: 2*expr)
        t + sin(2*x + 1) + cos(2*x + 2*y + E)

        """
        def _apply(path, expr, func):
            if not path:
                return func(expr)
            else:
                selector, path = path[0], path[1:]
                attrs, types, span = selector

                if isinstance(expr, Basic):
                    if not expr.is_Atom:
                        args, basic = self._get_ordered_args(expr), True
                    else:
                        return expr
                elif hasattr(expr, '__iter__'):
                    args, basic = expr, False
                else:
                    return expr

                args = list(args)

                if span is not None:
                    if isinstance(span, slice):
                        indices = range(*span.indices(len(args)))
                    else:
                        indices = [span]
                else:
                    indices = range(len(args))

                for i in indices:
                    try:
                        arg = args[i]
                    except IndexError:
                        continue

                    if self._has(arg, attrs, types):
                        args[i] = _apply(path, arg, func)

                if basic:
                    return expr.func(*args)
                else:
                    return expr.__class__(args)

        _args, _kwargs = args or (), kwargs or {}
        _func = lambda expr: func(expr, *_args, **_kwargs)

        return _apply(self._epath, expr, _func)

    def select(self, expr):
        """
        Retrieve parts of an expression selected by a path.

        Examples
        ========

        >>> from sympy.simplify.epathtools import EPath
        >>> from sympy import sin, cos, E
        >>> from sympy.abc import x, y, z, t

        >>> path = EPath("/*/[0]/Symbol")
        >>> expr = [((x, 1), 2), ((3, y), z)]

        >>> path.select(expr)
        [x, y]

        >>> path = EPath("/*/*/Symbol")
        >>> expr = t + sin(x + 1) + cos(x + y + E)

        >>> path.select(expr)
        [x, x, y]

        """
        result = []

        def _select(path, expr):
            if not path:
                result.append(expr)
            else:
                selector, path = path[0], path[1:]
                attrs, types, span = selector

                if isinstance(expr, Basic):
                    args = self._get_ordered_args(expr)
                elif hasattr(expr, '__iter__'):
                    args = expr
                else:
                    return

                if span is not None:
                    if isinstance(span, slice):
                        args = args[span]
                    else:
                        try:
                            args = [args[span]]
                        except IndexError:
                            return

                for arg in args:
                    if self._has(arg, attrs, types):
                        _select(path, arg)

        _select(self._epath, expr)
        return result


def epath(path, expr=None, func=None, args=None, kwargs=None):
    r"""
    Manipulate parts of an expression selected by a path.

    Explanation
    ===========

    This function allows to manipulate large nested expressions in single
    line of code, utilizing techniques to those applied in XML processing
    standards (e.g. XPath).

    If ``func`` is ``None``, :func:`epath` retrieves elements selected by
    the ``path``. Otherwise it applies ``func`` to each matching element.

    Note that it is more efficient to create an EPath object and use the select
    and apply methods of that object, since this will compile the path string
    only once.  This function should only be used as a convenient shortcut for
    interactive use.

    This is the supported syntax:

    * select all: ``/*``
          Equivalent of ``for arg in args:``.
    * select slice: ``/[0]`` or ``/[1:5]`` or ``/[1:5:2]``
          Supports standard Python's slice syntax.
    * select by type: ``/list`` or ``/list|tuple``
          Emulates ``isinstance()``.
    * select by attribute: ``/__iter__?``
          Emulates ``hasattr()``.

    Parameters
    ==========

    path : str | EPath
        A path as a string or a compiled EPath.
    expr : Basic | iterable
        An expression or a container of expressions.
    func : callable (optional)
        A callable that will be applied to matching parts.
    args : tuple (optional)
        Additional positional arguments to ``func``.
    kwargs : dict (optional)
        Additional keyword arguments to ``func``.

    Examples
    ========

    >>> from sympy.simplify.epathtools import epath
    >>> from sympy import sin, cos, E
    >>> from sympy.abc import x, y, z, t

    >>> path = "/*/[0]/Symbol"
    >>> expr = [((x, 1), 2), ((3, y), z)]

    >>> epath(path, expr)
    [x, y]
    >>> epath(path, expr, lambda expr: expr**2)
    [((x**2, 1), 2), ((3, y**2), z)]

    >>> path = "/*/*/Symbol"
    >>> expr = t + sin(x + 1) + cos(x + y + E)

    >>> epath(path, expr)
    [x, x, y]
    >>> epath(path, expr, lambda expr: 2*expr)
    t + sin(2*x + 1) + cos(2*x + 2*y + E)

    """
    _epath = EPath(path)

    if expr is None:
        return _epath
    if func is None:
        return _epath.select(expr)
    else:
        return _epath.apply(expr, func, args, kwargs)
<|endoftext|>from collections import defaultdict

from sympy.core import sympify, S, Mul, Derivative, Pow
from sympy.core.add import _unevaluated_Add, Add
from sympy.core.assumptions import assumptions
from sympy.core.exprtools import Factors, gcd_terms
from sympy.core.function import _mexpand, expand_mul, expand_power_base
from sympy.core.mul import _keep_coeff, _unevaluated_Mul, _mulsort
from sympy.core.numbers import Rational, zoo, nan
from sympy.core.parameters import global_parameters
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.symbol import Dummy, Wild, symbols
from sympy.functions import exp, sqrt, log
from sympy.functions.elementary.complexes import Abs
from sympy.polys import gcd
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.utilities.iterables import iterable, sift




def collect(expr, syms, func=None, evaluate=None, exact=False, distribute_order_term=True):
    """
    Collect additive terms of an expression.

    Explanation
    ===========

    This function collects additive terms of an expression with respect
    to a list of expression up to powers with rational exponents. By the
    term symbol here are meant arbitrary expressions, which can contain
    powers, products, sums etc. In other words symbol is a pattern which
    will be searched for in the expression's terms.

    The input expression is not expanded by :func:`collect`, so user is
    expected to provide an expression in an appropriate form. This makes
    :func:`collect` more predictable as there is no magic happening behind the
    scenes. However, it is important to note, that powers of products are
    converted to products of powers using the :func:`~.expand_power_base`
    function.

    There are two possible types of output. First, if ``evaluate`` flag is
    set, this function will return an expression with collected terms or
    else it will return a dictionary with expressions up to rational powers
    as keys and collected coefficients as values.

    Examples
    ========

    >>> from sympy import S, collect, expand, factor, Wild
    >>> from sympy.abc import a, b, c, x, y

    This function can collect symbolic coefficients in polynomials or
    rational expressions. It will manage to find all integer or rational
    powers of collection variable::

        >>> collect(a*x**2 + b*x**2 + a*x - b*x + c, x)
        c + x**2*(a + b) + x*(a - b)

    The same result can be achieved in dictionary form::

        >>> d = collect(a*x**2 + b*x**2 + a*x - b*x + c, x, evaluate=False)
        >>> d[x**2]
        a + b
        >>> d[x]
        a - b
        >>> d[S.One]
        c

    You can also work with multivariate polynomials. However, remember that
    this function is greedy so it will care only about a single symbol at time,
    in specification order::

        >>> collect(x**2 + y*x**2 + x*y + y + a*y, [x, y])
        x**2*(y + 1) + x*y + y*(a + 1)

    Also more complicated expressions can be used as patterns::

        >>> from sympy import sin, log
        >>> collect(a*sin(2*x) + b*sin(2*x), sin(2*x))
        (a + b)*sin(2*x)

        >>> collect(a*x*log(x) + b*(x*log(x)), x*log(x))
        x*(a + b)*log(x)

    You can use wildcards in the pattern::

        >>> w = Wild('w1')
        >>> collect(a*x**y - b*x**y, w**y)
        x**y*(a - b)

    It is also possible to work with symbolic powers, although it has more
    complicated behavior, because in this case power's base and symbolic part
    of the exponent are treated as a single symbol::

        >>> collect(a*x**c + b*x**c, x)
        a*x**c + b*x**c
        >>> collect(a*x**c + b*x**c, x**c)
        x**c*(a + b)

    However if you incorporate rationals to the exponents, then you will get
    well known behavior::

        >>> collect(a*x**(2*c) + b*x**(2*c), x**c)
        x**(2*c)*(a + b)

    Note also that all previously stated facts about :func:`collect` function
    apply to the exponential function, so you can get::

        >>> from sympy import exp
        >>> collect(a*exp(2*x) + b*exp(2*x), exp(x))
        (a + b)*exp(2*x)

    If you are interested only in collecting specific powers of some symbols
    then set ``exact`` flag to True::

        >>> collect(a*x**7 + b*x**7, x, exact=True)
        a*x**7 + b*x**7
        >>> collect(a*x**7 + b*x**7, x**7, exact=True)
        x**7*(a + b)

    If you want to collect on any object containing symbols, set
    ``exact`` to None:

        >>> collect(x*exp(x) + sin(x)*y + sin(x)*2 + 3*x, x, exact=None)
        x*exp(x) + 3*x + (y + 2)*sin(x)
        >>> collect(a*x*y + x*y + b*x + x, [x, y], exact=None)
        x*y*(a + 1) + x*(b + 1)

    You can also apply this function to differential equations, where
    derivatives of arbitrary order can be collected. Note that if you
    collect with respect to a function or a derivative of a function, all
    derivatives of that function will also be collected. Use
    ``exact=True`` to prevent this from happening::

        >>> from sympy import Derivative as D, collect, Function
        >>> f = Function('f') (x)

        >>> collect(a*D(f,x) + b*D(f,x), D(f,x))
        (a + b)*Derivative(f(x), x)

        >>> collect(a*D(D(f,x),x) + b*D(D(f,x),x), f)
        (a + b)*Derivative(f(x), (x, 2))

        >>> collect(a*D(D(f,x),x) + b*D(D(f,x),x), D(f,x), exact=True)
        a*Derivative(f(x), (x, 2)) + b*Derivative(f(x), (x, 2))

        >>> collect(a*D(f,x) + b*D(f,x) + a*f + b*f, f)
        (a + b)*f(x) + (a + b)*Derivative(f(x), x)

    Or you can even match both derivative order and exponent at the same time::

        >>> collect(a*D(D(f,x),x)**2 + b*D(D(f,x),x)**2, D(f,x))
        (a + b)*Derivative(f(x), (x, 2))**2

    Finally, you can apply a function to each of the collected coefficients.
    For example you can factorize symbolic coefficients of polynomial::

        >>> f = expand((x + a + 1)**3)

        >>> collect(f, x, factor)
        x**3 + 3*x**2*(a + 1) + 3*x*(a + 1)**2 + (a + 1)**3

    .. note:: Arguments are expected to be in expanded form, so you might have
              to call :func:`~.expand` prior to calling this function.

    See Also
    ========

    collect_const, collect_sqrt, rcollect
    """
    expr = sympify(expr)
    syms = [sympify(i) for i in (syms if iterable(syms) else [syms])]

    # replace syms[i] if it is not x, -x or has Wild symbols
    cond = lambda x: x.is_Symbol or (-x).is_Symbol or bool(
        x.atoms(Wild))
    _, nonsyms = sift(syms, cond, binary=True)
    if nonsyms:
        reps = dict(zip(nonsyms, [Dummy(**assumptions(i)) for i in nonsyms]))
        syms = [reps.get(s, s) for s in syms]
        rv = collect(expr.subs(reps), syms,
            func=func, evaluate=evaluate, exact=exact,
            distribute_order_term=distribute_order_term)
        urep = {v: k for k, v in reps.items()}
        if not isinstance(rv, dict):
            return rv.xreplace(urep)
        else:
            return {urep.get(k, k).xreplace(urep): v.xreplace(urep)
                    for k, v in rv.items()}

    # see if other expressions should be considered
    if exact is None:
        _syms = set()
        for i in Add.make_args(expr):
            if not i.has_free(*syms) or i in syms:
                continue
            if not i.is_Mul and i not in syms:
                _syms.add(i)
            else:
                # identify compound generators
                g = i._new_rawargs(*i.as_coeff_mul(*syms)[1])
                if g not in syms:
                    _syms.add(g)
        simple = all(i.is_Pow and i.base in syms for i in _syms)
        syms = syms + list(ordered(_syms))
        if not simple:
            return collect(expr, syms,
            func=func, evaluate=evaluate, exact=False,
            distribute_order_term=distribute_order_term)

    if evaluate is None:
        evaluate = global_parameters.evaluate

    def make_expression(terms):
        product = []

        for term, rat, sym, deriv in terms:
            if deriv is not None:
                var, order = deriv
                for _ in range(order):
                    term = Derivative(term, var)

            if sym is None:
                if rat is S.One:
                    product.append(term)
                else:
                    product.append(Pow(term, rat))
            else:
                product.append(Pow(term, rat*sym))

        return Mul(*product)

    def parse_derivative(deriv):
        # scan derivatives tower in the input expression and return
        # underlying function and maximal differentiation order
        expr, sym, order = deriv.expr, deriv.variables[0], 1

        for s in deriv.variables[1:]:
            if s == sym:
                order += 1
            else:
                raise NotImplementedError(
                    'Improve MV Derivative support in collect')

        while isinstance(expr, Derivative):
            s0 = expr.variables[0]

            if any(s != s0 for s in expr.variables):
                raise NotImplementedError(
                    'Improve MV Derivative support in collect')

            if s0 == sym:
                expr, order = expr.expr, order + len(expr.variables)
            else:
                break

        return expr, (sym, Rational(order))

    def parse_term(expr):
        """Parses expression expr and outputs tuple (sexpr, rat_expo,
        sym_expo, deriv)
        where:
         - sexpr is the base expression
         - rat_expo is the rational exponent that sexpr is raised to
         - sym_expo is the symbolic exponent that sexpr is raised to
         - deriv contains the derivatives of the expression

         For example, the output of x would be (x, 1, None, None)
         the output of 2**x would be (2, 1, x, None).
        """
        rat_expo, sym_expo = S.One, None
        sexpr, deriv = expr, None

        if expr.is_Pow:
            if isinstance(expr.base, Derivative):
                sexpr, deriv = parse_derivative(expr.base)
            else:
                sexpr = expr.base

            if expr.base == S.Exp1:
                arg = expr.exp
                if arg.is_Rational:
                    sexpr, rat_expo = S.Exp1, arg
                elif arg.is_Mul:
                    coeff, tail = arg.as_coeff_Mul(rational=True)
                    sexpr, rat_expo = exp(tail), coeff

            elif expr.exp.is_Number:
                rat_expo = expr.exp
            else:
                coeff, tail = expr.exp.as_coeff_Mul()

                if coeff.is_Number:
                    rat_expo, sym_expo = coeff, tail
                else:
                    sym_expo = expr.exp
        elif isinstance(expr, exp):
            arg = expr.exp
            if arg.is_Rational:
                sexpr, rat_expo = S.Exp1, arg
            elif arg.is_Mul:
                coeff, tail = arg.as_coeff_Mul(rational=True)
                sexpr, rat_expo = exp(tail), coeff
        elif isinstance(expr, Derivative):
            sexpr, deriv = parse_derivative(expr)

        return sexpr, rat_expo, sym_expo, deriv

    def parse_expression(terms, pattern):
        """Parse terms searching for a pattern.
        Terms is a list of tuples as returned by parse_terms;
        Pattern is an expression treated as a product of factors.
        """
        pattern = Mul.make_args(pattern)

        if len(terms) < len(pattern):
            # pattern is longer than matched product
            # so no chance for positive parsing result
            return None
        else:
            pattern = [parse_term(elem) for elem in pattern]

            terms = terms[:]  # need a copy
            elems, common_expo, has_deriv = [], None, False

            for elem, e_rat, e_sym, e_ord in pattern:

                if elem.is_Number and e_rat == 1 and e_sym is None:
                    # a constant is a match for everything
                    continue

                for j in range(len(terms)):
                    if terms[j] is None:
                        continue

                    term, t_rat, t_sym, t_ord = terms[j]

                    # keeping track of whether one of the terms had
                    # a derivative or not as this will require rebuilding
                    # the expression later
                    if t_ord is not None:
                        has_deriv = True

                    if (term.match(elem) is not None and
                            (t_sym == e_sym or t_sym is not None and
                            e_sym is not None and
                            t_sym.match(e_sym) is not None)):
                        if exact is False:
                            # we don't have to be exact so find common exponent
                            # for both expression's term and pattern's element
                            expo = t_rat / e_rat

                            if common_expo is None:
                                # first time
                                common_expo = expo
                            else:
                                # common exponent was negotiated before so
                                # there is no chance for a pattern match unless
                                # common and current exponents are equal
                                if common_expo != expo:
                                    common_expo = 1
                        else:
                            # we ought to be exact so all fields of
                            # interest must match in every details
                            if e_rat != t_rat or e_ord != t_ord:
                                continue

                        # found common term so remove it from the expression
                        # and try to match next element in the pattern
                        elems.append(terms[j])
                        terms[j] = None

                        break

                else:
                    # pattern element not found
                    return None

            return [_f for _f in terms if _f], elems, common_expo, has_deriv

    if evaluate:
        if expr.is_Add:
            o = expr.getO() or 0
            expr = expr.func(*[
                    collect(a, syms, func, True, exact, distribute_order_term)
                    for a in expr.args if a != o]) + o
        elif expr.is_Mul:
            return expr.func(*[
                collect(term, syms, func, True, exact, distribute_order_term)
                for term in expr.args])
        elif expr.is_Pow:
            b = collect(
                expr.base, syms, func, True, exact, distribute_order_term)
            return Pow(b, expr.exp)

    syms = [expand_power_base(i, deep=False) for i in syms]

    order_term = None

    if distribute_order_term:
        order_term = expr.getO()

        if order_term is not None:
            if order_term.has(*syms):
                order_term = None
            else:
                expr = expr.removeO()

    summa = [expand_power_base(i, deep=False) for i in Add.make_args(expr)]

    collected, disliked = defaultdict(list), S.Zero
    for product in summa:
        c, nc = product.args_cnc(split_1=False)
        args = list(ordered(c)) + nc
        terms = [parse_term(i) for i in args]
        small_first = True

        for symbol in syms:
            if isinstance(symbol, Derivative) and small_first:
                terms = list(reversed(terms))
                small_first = not small_first
            result = parse_expression(terms, symbol)

            if result is not None:
                if not symbol.is_commutative:
                    raise AttributeError("Can not collect noncommutative symbol")

                terms, elems, common_expo, has_deriv = result

                # when there was derivative in current pattern we
                # will need to rebuild its expression from scratch
                if not has_deriv:
                    margs = []
                    for elem in elems:
                        if elem[2] is None:
                            e = elem[1]
                        else:
                            e = elem[1]*elem[2]
                        margs.append(Pow(elem[0], e))
                    index = Mul(*margs)
                else:
                    index = make_expression(elems)
                terms = expand_power_base(make_expression(terms), deep=False)
                index = expand_power_base(index, deep=False)
                collected[index].append(terms)
                break
        else:
            # none of the patterns matched
            disliked += product
    # add terms now for each key
    collected = {k: Add(*v) for k, v in collected.items()}

    if disliked is not S.Zero:
        collected[S.One] = disliked

    if order_term is not None:
        for key, val in collected.items():
            collected[key] = val + order_term

    if func is not None:
        collected = {
            key: func(val) for key, val in collected.items()}

    if evaluate:
        return Add(*[key*val for key, val in collected.items()])
    else:
        return collected


def rcollect(expr, *vars):
    """
    Recursively collect sums in an expression.

    Examples
    ========

    >>> from sympy.simplify import rcollect
    >>> from sympy.abc import x, y

    >>> expr = (x**2*y + x*y + x + y)/(x + y)

    >>> rcollect(expr, y)
    (x + y*(x**2 + x + 1))/(x + y)

    See Also
    ========

    collect, collect_const, collect_sqrt
    """
    if expr.is_Atom or not expr.has(*vars):
        return expr
    else:
        expr = expr.__class__(*[rcollect(arg, *vars) for arg in expr.args])

        if expr.is_Add:
            return collect(expr, vars)
        else:
            return expr


def collect_sqrt(expr, evaluate=None):
    """Return expr with terms having common square roots collected together.
    If ``evaluate`` is False a count indicating the number of sqrt-containing
    terms will be returned and, if non-zero, the terms of the Add will be
    returned, else the expression itself will be returned as a single term.
    If ``evaluate`` is True, the expression with any collected terms will be
    returned.

    Note: since I = sqrt(-1), it is collected, too.

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.simplify.radsimp import collect_sqrt
    >>> from sympy.abc import a, b

    >>> r2, r3, r5 = [sqrt(i) for i in [2, 3, 5]]
    >>> collect_sqrt(a*r2 + b*r2)
    sqrt(2)*(a + b)
    >>> collect_sqrt(a*r2 + b*r2 + a*r3 + b*r3)
    sqrt(2)*(a + b) + sqrt(3)*(a + b)
    >>> collect_sqrt(a*r2 + b*r2 + a*r3 + b*r5)
    sqrt(3)*a + sqrt(5)*b + sqrt(2)*(a + b)

    If evaluate is False then the arguments will be sorted and
    returned as a list and a count of the number of sqrt-containing
    terms will be returned:

    >>> collect_sqrt(a*r2 + b*r2 + a*r3 + b*r5, evaluate=False)
    ((sqrt(3)*a, sqrt(5)*b, sqrt(2)*(a + b)), 3)
    >>> collect_sqrt(a*sqrt(2) + b, evaluate=False)
    ((b, sqrt(2)*a), 1)
    >>> collect_sqrt(a + b, evaluate=False)
    ((a + b,), 0)

    See Also
    ========

    collect, collect_const, rcollect
    """
    if evaluate is None:
        evaluate = global_parameters.evaluate
    # this step will help to standardize any complex arguments
    # of sqrts
    coeff, expr = expr.as_content_primitive()
    vars = set()
    for a in Add.make_args(expr):
        for m in a.args_cnc()[0]:
            if m.is_number and (
                    m.is_Pow and m.exp.is_Rational and m.exp.q == 2 or
                    m is S.ImaginaryUnit):
                vars.add(m)

    # we only want radicals, so exclude Number handling; in this case
    # d will be evaluated
    d = collect_const(expr, *vars, Numbers=False)
    hit = expr != d

    if not evaluate:
        nrad = 0
        # make the evaluated args canonical
        args = list(ordered(Add.make_args(d)))
        for i, m in enumerate(args):
            c, nc = m.args_cnc()
            for ci in c:
                # XXX should this be restricted to ci.is_number as above?
                if ci.is_Pow and ci.exp.is_Rational and ci.exp.q == 2 or \
                        ci is S.ImaginaryUnit:
                    nrad += 1
                    break
            args[i] *= coeff
        if not (hit or nrad):
            args = [Add(*args)]
        return tuple(args), nrad

    return coeff*d


def collect_abs(expr):
    """Return ``expr`` with arguments of multiple Abs in a term collected
    under a single instance.

    Examples
    ========

    >>> from sympy.simplify.radsimp import collect_abs
    >>> from sympy.abc import x
    >>> collect_abs(abs(x + 1)/abs(x**2 - 1))
    Abs((x + 1)/(x**2 - 1))
    >>> collect_abs(abs(1/x))
    Abs(1/x)
    """
    def _abs(mul):
        c, nc = mul.args_cnc()
        a = []
        o = []
        for i in c:
            if isinstance(i, Abs):
                a.append(i.args[0])
            elif isinstance(i, Pow) and isinstance(i.base, Abs) and i.exp.is_real:
                a.append(i.base.args[0]**i.exp)
            else:
                o.append(i)
        if len(a) < 2 and not any(i.exp.is_negative for i in a if isinstance(i, Pow)):
            return mul
        absarg = Mul(*a)
        A = Abs(absarg)
        args = [A]
        args.extend(o)
        if not A.has(Abs):
            args.extend(nc)
            return Mul(*args)
        if not isinstance(A, Abs):
            # reevaluate and make it unevaluated
            A = Abs(absarg, evaluate=False)
        args[0] = A
        _mulsort(args)
        args.extend(nc)  # nc always go last
        return Mul._from_args(args, is_commutative=not nc)

    return expr.replace(
        lambda x: isinstance(x, Mul),
        lambda x: _abs(x)).replace(
            lambda x: isinstance(x, Pow),
            lambda x: _abs(x))


def collect_const(expr, *vars, Numbers=True):
    """A non-greedy collection of terms with similar number coefficients in
    an Add expr. If ``vars`` is given then only those constants will be
    targeted. Although any Number can also be targeted, if this is not
    desired set ``Numbers=False`` and no Float or Rational will be collected.

    Parameters
    ==========

    expr : SymPy expression
        This parameter defines the expression the expression from which
        terms with similar coefficients are to be collected. A non-Add
        expression is returned as it is.

    vars : variable length collection of Numbers, optional
        Specifies the constants to target for collection. Can be multiple in
        number.

    Numbers : bool
        Specifies to target all instance of
        :class:`sympy.core.numbers.Number` class. If ``Numbers=False``, then
        no Float or Rational will be collected.

    Returns
    =======

    expr : Expr
        Returns an expression with similar coefficient terms collected.

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.abc import s, x, y, z
    >>> from sympy.simplify.radsimp import collect_const
    >>> collect_const(sqrt(3) + sqrt(3)*(1 + sqrt(2)))
    sqrt(3)*(sqrt(2) + 2)
    >>> collect_const(sqrt(3)*s + sqrt(7)*s + sqrt(3) + sqrt(7))
    (sqrt(3) + sqrt(7))*(s + 1)
    >>> s = sqrt(2) + 2
    >>> collect_const(sqrt(3)*s + sqrt(3) + sqrt(7)*s + sqrt(7))
    (sqrt(2) + 3)*(sqrt(3) + sqrt(7))
    >>> collect_const(sqrt(3)*s + sqrt(3) + sqrt(7)*s + sqrt(7), sqrt(3))
    sqrt(7) + sqrt(3)*(sqrt(2) + 3) + sqrt(7)*(sqrt(2) + 2)

    The collection is sign-sensitive, giving higher precedence to the
    unsigned values:

    >>> collect_const(x - y - z)
    x - (y + z)
    >>> collect_const(-y - z)
    -(y + z)
    >>> collect_const(2*x - 2*y - 2*z, 2)
    2*(x - y - z)
    >>> collect_const(2*x - 2*y - 2*z, -2)
    2*x - 2*(y + z)

    See Also
    ========

    collect, collect_sqrt, rcollect
    """
    if not expr.is_Add:
        return expr

    recurse = False

    if not vars:
        recurse = True
        vars = set()
        for a in expr.args:
            for m in Mul.make_args(a):
                if m.is_number:
                    vars.add(m)
    else:
        vars = sympify(vars)
    if not Numbers:
        vars = [v for v in vars if not v.is_Number]

    vars = list(ordered(vars))
    for v in vars:
        terms = defaultdict(list)
        Fv = Factors(v)
        for m in Add.make_args(expr):
            f = Factors(m)
            q, r = f.div(Fv)
            if r.is_one:
                # only accept this as a true factor if
                # it didn't change an exponent from an Integer
                # to a non-Integer, e.g. 2/sqrt(2) -> sqrt(2)
                # -- we aren't looking for this sort of change
                fwas = f.factors.copy()
                fnow = q.factors
                if not any(k in fwas and fwas[k].is_Integer and not
                        fnow[k].is_Integer for k in fnow):
                    terms[v].append(q.as_expr())
                    continue
            terms[S.One].append(m)

        args = []
        hit = False
        uneval = False
        for k in ordered(terms):
            v = terms[k]
            if k is S.One:
                args.extend(v)
                continue

            if len(v) > 1:
                v = Add(*v)
                hit = True
                if recurse and v != expr:
                    vars.append(v)
            else:
                v = v[0]

            # be careful not to let uneval become True unless
            # it must be because it's going to be more expensive
            # to rebuild the expression as an unevaluated one
            if Numbers and k.is_Number and v.is_Add:
                args.append(_keep_coeff(k, v, sign=True))
                uneval = True
            else:
                args.append(k*v)

        if hit:
            if uneval:
                expr = _unevaluated_Add(*args)
            else:
                expr = Add(*args)
            if not expr.is_Add:
                break

    return expr


def radsimp(expr, symbolic=True, max_terms=4):
    r"""
    Rationalize the denominator by removing square roots.

    Explanation
    ===========

    The expression returned from radsimp must be used with caution
    since if the denominator contains symbols, it will be possible to make
    substitutions that violate the assumptions of the simplification process:
    that for a denominator matching a + b*sqrt(c), a != +/-b*sqrt(c). (If
    there are no symbols, this assumptions is made valid by collecting terms
    of sqrt(c) so the match variable ``a`` does not contain ``sqrt(c)``.) If
    you do not want the simplification to occur for symbolic denominators, set
    ``symbolic`` to False.

    If there are more than ``max_terms`` radical terms then the expression is
    returned unchanged.

    Examples
    ========

    >>> from sympy import radsimp, sqrt, Symbol, pprint
    >>> from sympy import factor_terms, fraction, signsimp
    >>> from sympy.simplify.radsimp import collect_sqrt
    >>> from sympy.abc import a, b, c

    >>> radsimp(1/(2 + sqrt(2)))
    (2 - sqrt(2))/2
    >>> x,y = map(Symbol, 'xy')
    >>> e = ((2 + 2*sqrt(2))*x + (2 + sqrt(8))*y)/(2 + sqrt(2))
    >>> radsimp(e)
    sqrt(2)*(x + y)

    No simplification beyond removal of the gcd is done. One might
    want to polish the result a little, however, by collecting
    square root terms:

    >>> r2 = sqrt(2)
    >>> r5 = sqrt(5)
    >>> ans = radsimp(1/(y*r2 + x*r2 + a*r5 + b*r5)); pprint(ans)
        ___       ___       ___       ___
      \/ 5 *a + \/ 5 *b - \/ 2 *x - \/ 2 *y
    ------------------------------------------
       2               2      2              2
    5*a  + 10*a*b + 5*b  - 2*x  - 4*x*y - 2*y

    >>> n, d = fraction(ans)
    >>> pprint(factor_terms(signsimp(collect_sqrt(n))/d, radical=True))
            ___             ___
          \/ 5 *(a + b) - \/ 2 *(x + y)
    ------------------------------------------
       2               2      2              2
    5*a  + 10*a*b + 5*b  - 2*x  - 4*x*y - 2*y

    If radicals in the denominator cannot be removed or there is no denominator,
    the original expression will be returned.

    >>> radsimp(sqrt(2)*x + sqrt(2))
    sqrt(2)*x + sqrt(2)

    Results with symbols will not always be valid for all substitutions:

    >>> eq = 1/(a + b*sqrt(c))
    >>> eq.subs(a, b*sqrt(c))
    1/(2*b*sqrt(c))
    >>> radsimp(eq).subs(a, b*sqrt(c))
    nan

    If ``symbolic=False``, symbolic denominators will not be transformed (but
    numeric denominators will still be processed):

    >>> radsimp(eq, symbolic=False)
    1/(a + b*sqrt(c))

    """
    from sympy.core.expr import Expr
    from sympy.simplify.simplify import signsimp

    syms = symbols("a:d A:D")
    def _num(rterms):
        # return the multiplier that will simplify the expression described
        # by rterms [(sqrt arg, coeff), ... ]
        a, b, c, d, A, B, C, D = syms
        if len(rterms) == 2:
            reps = dict(list(zip([A, a, B, b], [j for i in rterms for j in i])))
            return (
            sqrt(A)*a - sqrt(B)*b).xreplace(reps)
        if len(rterms) == 3:
            reps = dict(list(zip([A, a, B, b, C, c], [j for i in rterms for j in i])))
            return (
            (sqrt(A)*a + sqrt(B)*b - sqrt(C)*c)*(2*sqrt(A)*sqrt(B)*a*b - A*a**2 -
            B*b**2 + C*c**2)).xreplace(reps)
        elif len(rterms) == 4:
            reps = dict(list(zip([A, a, B, b, C, c, D, d], [j for i in rterms for j in i])))
            return ((sqrt(A)*a + sqrt(B)*b - sqrt(C)*c - sqrt(D)*d)*(2*sqrt(A)*sqrt(B)*a*b
                - A*a**2 - B*b**2 - 2*sqrt(C)*sqrt(D)*c*d + C*c**2 +
                D*d**2)*(-8*sqrt(A)*sqrt(B)*sqrt(C)*sqrt(D)*a*b*c*d + A**2*a**4 -
                2*A*B*a**2*b**2 - 2*A*C*a**2*c**2 - 2*A*D*a**2*d**2 + B**2*b**4 -
                2*B*C*b**2*c**2 - 2*B*D*b**2*d**2 + C**2*c**4 - 2*C*D*c**2*d**2 +
                D**2*d**4)).xreplace(reps)
        elif len(rterms) == 1:
            return sqrt(rterms[0][0])
        else:
            raise NotImplementedError

    def ispow2(d, log2=False):
        if not d.is_Pow:
            return False
        e = d.exp
        if e.is_Rational and e.q == 2 or symbolic and denom(e) == 2:
            return True
        if log2:
            q = 1
            if e.is_Rational:
                q = e.q
            elif symbolic:
                d = denom(e)
                if d.is_Integer:
                    q = d
            if q != 1 and log(q, 2).is_Integer:
                return True
        return False

    def handle(expr):
        # Handle first reduces to the case
        # expr = 1/d, where d is an add, or d is base**p/2.
        # We do this by recursively calling handle on each piece.
        from sympy.simplify.simplify import nsimplify

        if expr.is_Atom:
            return expr
        elif not isinstance(expr, Expr):
            return expr.func(*[handle(a) for a in expr.args])

        n, d = fraction(expr)

        if d.is_Atom and n.is_Atom:
            return expr
        elif not n.is_Atom:
            n = n.func(*[handle(a) for a in n.args])
            return _unevaluated_Mul(n, handle(1/d))
        elif n is not S.One:
            return _unevaluated_Mul(n, handle(1/d))
        elif d.is_Mul:
            return _unevaluated_Mul(*[handle(1/d) for d in d.args])

        # By this step, expr is 1/d, and d is not a mul.
        if not symbolic and d.free_symbols:
            return expr

        if ispow2(d):
            d2 = sqrtdenest(sqrt(d.base))**numer(d.exp)
            if d2 != d:
                return handle(1/d2)
        elif d.is_Pow and (d.exp.is_integer or d.base.is_positive):
            # (1/d**i) = (1/d)**i
            return handle(1/d.base)**d.exp

        if not (d.is_Add or ispow2(d)):
            return 1/d.func(*[handle(a) for a in d.args])

        # handle 1/d treating d as an Add (though it may not be)

        keep = True  # keep changes that are made

        # flatten it and collect radicals after checking for special
        # conditions
        d = _mexpand(d)

        # did it change?
        if d.is_Atom:
            return 1/d

        # is it a number that might be handled easily?
        if d.is_number:
            _d = nsimplify(d)
            if _d.is_Number and _d.equals(d):
                return 1/_d

        while True:
            # collect similar terms
            collected = defaultdict(list)
