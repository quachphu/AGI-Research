    q, r = dmp_pdiv(f, g, u, K)

    if dmp_zero_p(r, u):
        return q
    else:
        raise ExactQuotientFailed(f, g)


def dup_rr_div(f, g, K):
    """
    Univariate division with remainder over a ring.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_rr_div(x**2 + 1, 2*x - 4)
    (0, x**2 + 1)

    """
    df = dup_degree(f)
    dg = dup_degree(g)

    q, r, dr = [], f, df

    if not g:
        raise ZeroDivisionError("polynomial division")
    elif df < dg:
        return q, r

    lc_g = dup_LC(g, K)

    while True:
        lc_r = dup_LC(r, K)

        if lc_r % lc_g:
            break

        c = K.exquo(lc_r, lc_g)
        j = dr - dg

        q = dup_add_term(q, c, j, K)
        h = dup_mul_term(g, c, j, K)
        r = dup_sub(r, h, K)

        _dr, dr = dr, dup_degree(r)

        if dr < dg:
            break
        elif not (dr < _dr):
            raise PolynomialDivisionFailed(f, g, K)

    return q, r


def dmp_rr_div(f, g, u, K):
    """
    Multivariate division with remainder over a ring.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_rr_div(x**2 + x*y, 2*x + 2)
    (0, x**2 + x*y)

    """
    if not u:
        return dup_rr_div(f, g, K)

    df = dmp_degree(f, u)
    dg = dmp_degree(g, u)

    if dg < 0:
        raise ZeroDivisionError("polynomial division")

    q, r, dr = dmp_zero(u), f, df

    if df < dg:
        return q, r

    lc_g, v = dmp_LC(g, K), u - 1

    while True:
        lc_r = dmp_LC(r, K)
        c, R = dmp_rr_div(lc_r, lc_g, v, K)

        if not dmp_zero_p(R, v):
            break

        j = dr - dg

        q = dmp_add_term(q, c, j, u, K)
        h = dmp_mul_term(g, c, j, u, K)
        r = dmp_sub(r, h, u, K)

        _dr, dr = dr, dmp_degree(r, u)

        if dr < dg:
            break
        elif not (dr < _dr):
            raise PolynomialDivisionFailed(f, g, K)

    return q, r


def dup_ff_div(f, g, K):
    """
    Polynomial division with remainder over a field.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    >>> R.dup_ff_div(x**2 + 1, 2*x - 4)
    (1/2*x + 1, 5)

    """
    df = dup_degree(f)
    dg = dup_degree(g)

    q, r, dr = [], f, df

    if not g:
        raise ZeroDivisionError("polynomial division")
    elif df < dg:
        return q, r

    lc_g = dup_LC(g, K)

    while True:
        lc_r = dup_LC(r, K)

        c = K.exquo(lc_r, lc_g)
        j = dr - dg

        q = dup_add_term(q, c, j, K)
        h = dup_mul_term(g, c, j, K)
        r = dup_sub(r, h, K)

        _dr, dr = dr, dup_degree(r)

        if dr < dg:
            break
        elif dr == _dr and not K.is_Exact:
            # remove leading term created by rounding error
            r = dup_strip(r[1:])
            dr = dup_degree(r)
            if dr < dg:
                break
        elif not (dr < _dr):
            raise PolynomialDivisionFailed(f, g, K)

    return q, r


def dmp_ff_div(f, g, u, K):
    """
    Polynomial division with remainder over a field.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x,y = ring("x,y", QQ)

    >>> R.dmp_ff_div(x**2 + x*y, 2*x + 2)
    (1/2*x + 1/2*y - 1/2, -y + 1)

    """
    if not u:
        return dup_ff_div(f, g, K)

    df = dmp_degree(f, u)
    dg = dmp_degree(g, u)

    if dg < 0:
        raise ZeroDivisionError("polynomial division")

    q, r, dr = dmp_zero(u), f, df

    if df < dg:
        return q, r

    lc_g, v = dmp_LC(g, K), u - 1

    while True:
        lc_r = dmp_LC(r, K)
        c, R = dmp_ff_div(lc_r, lc_g, v, K)

        if not dmp_zero_p(R, v):
            break

        j = dr - dg

        q = dmp_add_term(q, c, j, u, K)
        h = dmp_mul_term(g, c, j, u, K)
        r = dmp_sub(r, h, u, K)

        _dr, dr = dr, dmp_degree(r, u)

        if dr < dg:
            break
        elif not (dr < _dr):
            raise PolynomialDivisionFailed(f, g, K)

    return q, r


def dup_div(f, g, K):
    """
    Polynomial division with remainder in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x = ring("x", ZZ)
    >>> R.dup_div(x**2 + 1, 2*x - 4)
    (0, x**2 + 1)

    >>> R, x = ring("x", QQ)
    >>> R.dup_div(x**2 + 1, 2*x - 4)
    (1/2*x + 1, 5)

    """
    if K.is_Field:
        return dup_ff_div(f, g, K)
    else:
        return dup_rr_div(f, g, K)


def dup_rem(f, g, K):
    """
    Returns polynomial remainder in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x = ring("x", ZZ)
    >>> R.dup_rem(x**2 + 1, 2*x - 4)
    x**2 + 1

    >>> R, x = ring("x", QQ)
    >>> R.dup_rem(x**2 + 1, 2*x - 4)
    5

    """
    return dup_div(f, g, K)[1]


def dup_quo(f, g, K):
    """
    Returns exact polynomial quotient in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x = ring("x", ZZ)
    >>> R.dup_quo(x**2 + 1, 2*x - 4)
    0

    >>> R, x = ring("x", QQ)
    >>> R.dup_quo(x**2 + 1, 2*x - 4)
    1/2*x + 1

    """
    return dup_div(f, g, K)[0]


def dup_exquo(f, g, K):
    """
    Returns polynomial quotient in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_exquo(x**2 - 1, x - 1)
    x + 1

    >>> R.dup_exquo(x**2 + 1, 2*x - 4)
    Traceback (most recent call last):
    ...
    ExactQuotientFailed: [2, -4] does not divide [1, 0, 1]

    """
    q, r = dup_div(f, g, K)

    if not r:
        return q
    else:
        raise ExactQuotientFailed(f, g)


def dmp_div(f, g, u, K):
    """
    Polynomial division with remainder in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x,y = ring("x,y", ZZ)
    >>> R.dmp_div(x**2 + x*y, 2*x + 2)
    (0, x**2 + x*y)

    >>> R, x,y = ring("x,y", QQ)
    >>> R.dmp_div(x**2 + x*y, 2*x + 2)
    (1/2*x + 1/2*y - 1/2, -y + 1)

    """
    if K.is_Field:
        return dmp_ff_div(f, g, u, K)
    else:
        return dmp_rr_div(f, g, u, K)


def dmp_rem(f, g, u, K):
    """
    Returns polynomial remainder in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x,y = ring("x,y", ZZ)
    >>> R.dmp_rem(x**2 + x*y, 2*x + 2)
    x**2 + x*y

    >>> R, x,y = ring("x,y", QQ)
    >>> R.dmp_rem(x**2 + x*y, 2*x + 2)
    -y + 1

    """
    return dmp_div(f, g, u, K)[1]


def dmp_quo(f, g, u, K):
    """
    Returns exact polynomial quotient in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x,y = ring("x,y", ZZ)
    >>> R.dmp_quo(x**2 + x*y, 2*x + 2)
    0

    >>> R, x,y = ring("x,y", QQ)
    >>> R.dmp_quo(x**2 + x*y, 2*x + 2)
    1/2*x + 1/2*y - 1/2

    """
    return dmp_div(f, g, u, K)[0]


def dmp_exquo(f, g, u, K):
    """
    Returns polynomial quotient in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = x**2 + x*y
    >>> g = x + y
    >>> h = 2*x + 2

    >>> R.dmp_exquo(f, g)
    x

    >>> R.dmp_exquo(f, h)
    Traceback (most recent call last):
    ...
    ExactQuotientFailed: [[2], [2]] does not divide [[1], [1, 0], []]

    """
    q, r = dmp_div(f, g, u, K)

    if dmp_zero_p(r, u):
        return q
    else:
        raise ExactQuotientFailed(f, g)


def dup_max_norm(f, K):
    """
    Returns maximum norm of a polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_max_norm(-x**2 + 2*x - 3)
    3

    """
    if not f:
        return K.zero
    else:
        return max(dup_abs(f, K))


def dmp_max_norm(f, u, K):
    """
    Returns maximum norm of a polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_max_norm(2*x*y - x - 3)
    3

    """
    if not u:
        return dup_max_norm(f, K)

    v = u - 1

    return max(dmp_max_norm(c, v, K) for c in f)


def dup_l1_norm(f, K):
    """
    Returns l1 norm of a polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_l1_norm(2*x**3 - 3*x**2 + 1)
    6

    """
    if not f:
        return K.zero
    else:
        return sum(dup_abs(f, K))


def dmp_l1_norm(f, u, K):
    """
    Returns l1 norm of a polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_l1_norm(2*x*y - x - 3)
    6

    """
    if not u:
        return dup_l1_norm(f, K)

    v = u - 1

    return sum(dmp_l1_norm(c, v, K) for c in f)


def dup_l2_norm_squared(f, K):
    """
    Returns squared l2 norm of a polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_l2_norm_squared(2*x**3 - 3*x**2 + 1)
    14

    """
    return sum([coeff**2 for coeff in f], K.zero)


def dmp_l2_norm_squared(f, u, K):
    """
    Returns squared l2 norm of a polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_l2_norm_squared(2*x*y - x - 3)
    14

    """
    if not u:
        return dup_l2_norm_squared(f, K)

    v = u - 1

    return sum(dmp_l2_norm_squared(c, v, K) for c in f)


def dup_expand(polys, K):
    """
    Multiply together several polynomials in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_expand([x**2 - 1, x, 2])
    2*x**3 - 2*x

    """
    if not polys:
        return [K.one]

    f = polys[0]

    for g in polys[1:]:
        f = dup_mul(f, g, K)

    return f


def dmp_expand(polys, u, K):
    """
    Multiply together several polynomials in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_expand([x**2 + y**2, x + 1])
    x**3 + x**2 + x*y**2 + y**2

    """
    if not polys:
        return dmp_one(u, K)

    f = polys[0]

    for g in polys[1:]:
        f = dmp_mul(f, g, u, K)

    return f
<|endoftext|>r"""
Sparse distributed elements of free modules over multivariate (generalized)
polynomial rings.

This code and its data structures are very much like the distributed
polynomials, except that the first "exponent" of the monomial is
a module generator index. That is, the multi-exponent ``(i, e_1, ..., e_n)``
represents the "monomial" `x_1^{e_1} \cdots x_n^{e_n} f_i` of the free module
`F` generated by `f_1, \ldots, f_r` over (a localization of) the ring
`K[x_1, \ldots, x_n]`. A module element is simply stored as a list of terms
ordered by the monomial order. Here a term is a pair of a multi-exponent and a
coefficient. In general, this coefficient should never be zero (since it can
then be omitted). The zero module element is stored as an empty list.

The main routines are ``sdm_nf_mora`` and ``sdm_groebner`` which can be used
to compute, respectively, weak normal forms and standard bases. They work with
arbitrary (not necessarily global) monomial orders.

In general, product orders have to be used to construct valid monomial orders
for modules. However, ``lex`` can be used as-is.

Note that the "level" (number of variables, i.e. parameter u+1 in
distributedpolys.py) is never needed in this code.

The main reference for this file is [SCA],
"A Singular Introduction to Commutative Algebra".
"""


from itertools import permutations

from sympy.polys.monomials import (
    monomial_mul, monomial_lcm, monomial_div, monomial_deg
)

from sympy.polys.polytools import Poly
from sympy.polys.polyutils import parallel_dict_from_expr
from sympy.core.singleton import S
from sympy.core.sympify import sympify

# Additional monomial tools.


def sdm_monomial_mul(M, X):
    """
    Multiply tuple ``X`` representing a monomial of `K[X]` into the tuple
    ``M`` representing a monomial of `F`.

    Examples
    ========

    Multiplying `xy^3` into `x f_1` yields `x^2 y^3 f_1`:

    >>> from sympy.polys.distributedmodules import sdm_monomial_mul
    >>> sdm_monomial_mul((1, 1, 0), (1, 3))
    (1, 2, 3)
    """
    return (M[0],) + monomial_mul(X, M[1:])


def sdm_monomial_deg(M):
    """
    Return the total degree of ``M``.

    Examples
    ========

    For example, the total degree of `x^2 y f_5` is 3:

    >>> from sympy.polys.distributedmodules import sdm_monomial_deg
    >>> sdm_monomial_deg((5, 2, 1))
    3
    """
    return monomial_deg(M[1:])


def sdm_monomial_lcm(A, B):
    r"""
    Return the "least common multiple" of ``A`` and ``B``.

    IF `A = M e_j` and `B = N e_j`, where `M` and `N` are polynomial monomials,
    this returns `\lcm(M, N) e_j`. Note that ``A`` and ``B`` involve distinct
    monomials.

    Otherwise the result is undefined.

    Examples
    ========

    >>> from sympy.polys.distributedmodules import sdm_monomial_lcm
    >>> sdm_monomial_lcm((1, 2, 3), (1, 0, 5))
    (1, 2, 5)
    """
    return (A[0],) + monomial_lcm(A[1:], B[1:])


def sdm_monomial_divides(A, B):
    """
    Does there exist a (polynomial) monomial X such that XA = B?

    Examples
    ========

    Positive examples:

    In the following examples, the monomial is given in terms of x, y and the
    generator(s), f_1, f_2 etc. The tuple form of that monomial is used in
    the call to sdm_monomial_divides.
    Note: the generator appears last in the expression but first in the tuple
    and other factors appear in the same order that they appear in the monomial
    expression.

    `A = f_1` divides `B = f_1`

    >>> from sympy.polys.distributedmodules import sdm_monomial_divides
    >>> sdm_monomial_divides((1, 0, 0), (1, 0, 0))
    True

    `A = f_1` divides `B = x^2 y f_1`

    >>> sdm_monomial_divides((1, 0, 0), (1, 2, 1))
    True

    `A = xy f_5` divides `B = x^2 y f_5`

    >>> sdm_monomial_divides((5, 1, 1), (5, 2, 1))
    True

    Negative examples:

    `A = f_1` does not divide `B = f_2`

    >>> sdm_monomial_divides((1, 0, 0), (2, 0, 0))
    False

    `A = x f_1` does not divide `B = f_1`

    >>> sdm_monomial_divides((1, 1, 0), (1, 0, 0))
    False

    `A = xy^2 f_5` does not divide `B = y f_5`

    >>> sdm_monomial_divides((5, 1, 2), (5, 0, 1))
    False
    """
    return A[0] == B[0] and all(a <= b for a, b in zip(A[1:], B[1:]))


# The actual distributed modules code.

def sdm_LC(f, K):
    """Returns the leading coefficient of ``f``. """
    if not f:
        return K.zero
    else:
        return f[0][1]


def sdm_to_dict(f):
    """Make a dictionary from a distributed polynomial. """
    return dict(f)


def sdm_from_dict(d, O):
    """
    Create an sdm from a dictionary.

    Here ``O`` is the monomial order to use.

    Examples
    ========

    >>> from sympy.polys.distributedmodules import sdm_from_dict
    >>> from sympy.polys import QQ, lex
    >>> dic = {(1, 1, 0): QQ(1), (1, 0, 0): QQ(2), (0, 1, 0): QQ(0)}
    >>> sdm_from_dict(dic, lex)
    [((1, 1, 0), 1), ((1, 0, 0), 2)]
    """
    return sdm_strip(sdm_sort(list(d.items()), O))


def sdm_sort(f, O):
    """Sort terms in ``f`` using the given monomial order ``O``. """
    return sorted(f, key=lambda term: O(term[0]), reverse=True)


def sdm_strip(f):
    """Remove terms with zero coefficients from ``f`` in ``K[X]``. """
    return [ (monom, coeff) for monom, coeff in f if coeff ]


def sdm_add(f, g, O, K):
    """
    Add two module elements ``f``, ``g``.

    Addition is done over the ground field ``K``, monomials are ordered
    according to ``O``.

    Examples
    ========

    All examples use lexicographic order.

    `(xy f_1) + (f_2) = f_2 + xy f_1`

    >>> from sympy.polys.distributedmodules import sdm_add
    >>> from sympy.polys import lex, QQ
    >>> sdm_add([((1, 1, 1), QQ(1))], [((2, 0, 0), QQ(1))], lex, QQ)
    [((2, 0, 0), 1), ((1, 1, 1), 1)]

    `(xy f_1) + (-xy f_1)` = 0`

    >>> sdm_add([((1, 1, 1), QQ(1))], [((1, 1, 1), QQ(-1))], lex, QQ)
    []

    `(f_1) + (2f_1) = 3f_1`

    >>> sdm_add([((1, 0, 0), QQ(1))], [((1, 0, 0), QQ(2))], lex, QQ)
    [((1, 0, 0), 3)]

    `(yf_1) + (xf_1) = xf_1 + yf_1`

    >>> sdm_add([((1, 0, 1), QQ(1))], [((1, 1, 0), QQ(1))], lex, QQ)
    [((1, 1, 0), 1), ((1, 0, 1), 1)]
    """
    h = dict(f)

    for monom, c in g:
        if monom in h:
            coeff = h[monom] + c

            if not coeff:
                del h[monom]
            else:
                h[monom] = coeff
        else:
            h[monom] = c

    return sdm_from_dict(h, O)


def sdm_LM(f):
    r"""
    Returns the leading monomial of ``f``.

    Only valid if `f \ne 0`.

    Examples
    ========

    >>> from sympy.polys.distributedmodules import sdm_LM, sdm_from_dict
    >>> from sympy.polys import QQ, lex
    >>> dic = {(1, 2, 3): QQ(1), (4, 0, 0): QQ(1), (4, 0, 1): QQ(1)}
    >>> sdm_LM(sdm_from_dict(dic, lex))
    (4, 0, 1)
    """
    return f[0][0]


def sdm_LT(f):
    r"""
    Returns the leading term of ``f``.

    Only valid if `f \ne 0`.

    Examples
    ========

    >>> from sympy.polys.distributedmodules import sdm_LT, sdm_from_dict
    >>> from sympy.polys import QQ, lex
    >>> dic = {(1, 2, 3): QQ(1), (4, 0, 0): QQ(2), (4, 0, 1): QQ(3)}
    >>> sdm_LT(sdm_from_dict(dic, lex))
    ((4, 0, 1), 3)
    """
    return f[0]


def sdm_mul_term(f, term, O, K):
    """
    Multiply a distributed module element ``f`` by a (polynomial) term ``term``.

    Multiplication of coefficients is done over the ground field ``K``, and
    monomials are ordered according to ``O``.

    Examples
    ========

    `0 f_1 = 0`

    >>> from sympy.polys.distributedmodules import sdm_mul_term
    >>> from sympy.polys import lex, QQ
    >>> sdm_mul_term([((1, 0, 0), QQ(1))], ((0, 0), QQ(0)), lex, QQ)
    []

    `x 0 = 0`

    >>> sdm_mul_term([], ((1, 0), QQ(1)), lex, QQ)
    []

    `(x) (f_1) = xf_1`

    >>> sdm_mul_term([((1, 0, 0), QQ(1))], ((1, 0), QQ(1)), lex, QQ)
    [((1, 1, 0), 1)]

    `(2xy) (3x f_1 + 4y f_2) = 8xy^2 f_2 + 6x^2y f_1`

    >>> f = [((2, 0, 1), QQ(4)), ((1, 1, 0), QQ(3))]
    >>> sdm_mul_term(f, ((1, 1), QQ(2)), lex, QQ)
    [((2, 1, 2), 8), ((1, 2, 1), 6)]
    """
    X, c = term

    if not f or not c:
        return []
    else:
        if K.is_one(c):
            return [ (sdm_monomial_mul(f_M, X), f_c) for f_M, f_c in f ]
        else:
            return [ (sdm_monomial_mul(f_M, X), f_c * c) for f_M, f_c in f ]


def sdm_zero():
    """Return the zero module element."""
    return []


def sdm_deg(f):
    """
    Degree of ``f``.

    This is the maximum of the degrees of all its monomials.
    Invalid if ``f`` is zero.

    Examples
    ========

    >>> from sympy.polys.distributedmodules import sdm_deg
    >>> sdm_deg([((1, 2, 3), 1), ((10, 0, 1), 1), ((2, 3, 4), 4)])
    7
    """
    return max(sdm_monomial_deg(M[0]) for M in f)


# Conversion

def sdm_from_vector(vec, O, K, **opts):
    """
    Create an sdm from an iterable of expressions.

    Coefficients are created in the ground field ``K``, and terms are ordered
    according to monomial order ``O``. Named arguments are passed on to the
    polys conversion code and can be used to specify for example generators.

    Examples
    ========

    >>> from sympy.polys.distributedmodules import sdm_from_vector
    >>> from sympy.abc import x, y, z
    >>> from sympy.polys import QQ, lex
    >>> sdm_from_vector([x**2+y**2, 2*z], lex, QQ)
    [((1, 0, 0, 1), 2), ((0, 2, 0, 0), 1), ((0, 0, 2, 0), 1)]
    """
    dics, gens = parallel_dict_from_expr(sympify(vec), **opts)
    dic = {}
    for i, d in enumerate(dics):
        for k, v in d.items():
            dic[(i,) + k] = K.convert(v)
    return sdm_from_dict(dic, O)


def sdm_to_vector(f, gens, K, n=None):
    """
    Convert sdm ``f`` into a list of polynomial expressions.

    The generators for the polynomial ring are specified via ``gens``. The rank
    of the module is guessed, or passed via ``n``. The ground field is assumed
    to be ``K``.

    Examples
    ========

    >>> from sympy.polys.distributedmodules import sdm_to_vector
    >>> from sympy.abc import x, y, z
    >>> from sympy.polys import QQ
    >>> f = [((1, 0, 0, 1), QQ(2)), ((0, 2, 0, 0), QQ(1)), ((0, 0, 2, 0), QQ(1))]
    >>> sdm_to_vector(f, [x, y, z], QQ)
    [x**2 + y**2, 2*z]
    """
    dic = sdm_to_dict(f)
    dics = {}
    for k, v in dic.items():
        dics.setdefault(k[0], []).append((k[1:], v))
    n = n or len(dics)
    res = []
    for k in range(n):
        if k in dics:
            res.append(Poly(dict(dics[k]), gens=gens, domain=K).as_expr())
        else:
            res.append(S.Zero)
    return res

# Algorithms.


def sdm_spoly(f, g, O, K, phantom=None):
    """
    Compute the generalized s-polynomial of ``f`` and ``g``.

    The ground field is assumed to be ``K``, and monomials ordered according to
    ``O``.

    This is invalid if either of ``f`` or ``g`` is zero.

    If the leading terms of `f` and `g` involve different basis elements of
    `F`, their s-poly is defined to be zero. Otherwise it is a certain linear
    combination of `f` and `g` in which the leading terms cancel.
    See [SCA, defn 2.3.6] for details.

    If ``phantom`` is not ``None``, it should be a pair of module elements on
    which to perform the same operation(s) as on ``f`` and ``g``. The in this
    case both results are returned.

    Examples
    ========

    >>> from sympy.polys.distributedmodules import sdm_spoly
    >>> from sympy.polys import QQ, lex
    >>> f = [((2, 1, 1), QQ(1)), ((1, 0, 1), QQ(1))]
    >>> g = [((2, 3, 0), QQ(1))]
    >>> h = [((1, 2, 3), QQ(1))]
    >>> sdm_spoly(f, h, lex, QQ)
    []
    >>> sdm_spoly(f, g, lex, QQ)
    [((1, 2, 1), 1)]
    """
    if not f or not g:
        return sdm_zero()
    LM1 = sdm_LM(f)
    LM2 = sdm_LM(g)
    if LM1[0] != LM2[0]:
        return sdm_zero()
    LM1 = LM1[1:]
    LM2 = LM2[1:]
    lcm = monomial_lcm(LM1, LM2)
    m1 = monomial_div(lcm, LM1)
    m2 = monomial_div(lcm, LM2)
    c = K.quo(-sdm_LC(f, K), sdm_LC(g, K))
    r1 = sdm_add(sdm_mul_term(f, (m1, K.one), O, K),
                 sdm_mul_term(g, (m2, c), O, K), O, K)
    if phantom is None:
        return r1
    r2 = sdm_add(sdm_mul_term(phantom[0], (m1, K.one), O, K),
                 sdm_mul_term(phantom[1], (m2, c), O, K), O, K)
    return r1, r2


def sdm_ecart(f):
    """
    Compute the ecart of ``f``.

    This is defined to be the difference of the total degree of `f` and the
    total degree of the leading monomial of `f` [SCA, defn 2.3.7].

    Invalid if f is zero.

    Examples
    ========

    >>> from sympy.polys.distributedmodules import sdm_ecart
    >>> sdm_ecart([((1, 2, 3), 1), ((1, 0, 1), 1)])
    0
    >>> sdm_ecart([((2, 2, 1), 1), ((1, 5, 1), 1)])
    3
    """
    return sdm_deg(f) - sdm_monomial_deg(sdm_LM(f))


def sdm_nf_mora(f, G, O, K, phantom=None):
    r"""
    Compute a weak normal form of ``f`` with respect to ``G`` and order ``O``.

    The ground field is assumed to be ``K``, and monomials ordered according to
    ``O``.

    Weak normal forms are defined in [SCA, defn 2.3.3]. They are not unique.
    This function deterministically computes a weak normal form, depending on
    the order of `G`.

    The most important property of a weak normal form is the following: if
    `R` is the ring associated with the monomial ordering (if the ordering is
    global, we just have `R = K[x_1, \ldots, x_n]`, otherwise it is a certain
    localization thereof), `I` any ideal of `R` and `G` a standard basis for
    `I`, then for any `f \in R`, we have `f \in I` if and only if
    `NF(f | G) = 0`.

    This is the generalized Mora algorithm for computing weak normal forms with
    respect to arbitrary monomial orders [SCA, algorithm 2.3.9].

    If ``phantom`` is not ``None``, it should be a pair of "phantom" arguments
    on which to perform the same computations as on ``f``, ``G``, both results
    are then returned.
    """
    from itertools import repeat
    h = f
    T = list(G)
    if phantom is not None:
        # "phantom" variables with suffix p
        hp = phantom[0]
        Tp = list(phantom[1])
        phantom = True
    else:
        Tp = repeat([])
        phantom = False
    while h:
        # TODO better data structure!!!
        Th = [(g, sdm_ecart(g), gp) for g, gp in zip(T, Tp)
              if sdm_monomial_divides(sdm_LM(g), sdm_LM(h))]
        if not Th:
            break
        g, _, gp = min(Th, key=lambda x: x[1])
        if sdm_ecart(g) > sdm_ecart(h):
            T.append(h)
            if phantom:
                Tp.append(hp)
        if phantom:
            h, hp = sdm_spoly(h, g, O, K, phantom=(hp, gp))
        else:
            h = sdm_spoly(h, g, O, K)
    if phantom:
        return h, hp
    return h


def sdm_nf_buchberger(f, G, O, K, phantom=None):
    r"""
    Compute a weak normal form of ``f`` with respect to ``G`` and order ``O``.

    The ground field is assumed to be ``K``, and monomials ordered according to
    ``O``.

    This is the standard Buchberger algorithm for computing weak normal forms with
    respect to *global* monomial orders [SCA, algorithm 1.6.10].

    If ``phantom`` is not ``None``, it should be a pair of "phantom" arguments
    on which to perform the same computations as on ``f``, ``G``, both results
    are then returned.
    """
    from itertools import repeat
    h = f
    T = list(G)
    if phantom is not None:
        # "phantom" variables with suffix p
        hp = phantom[0]
        Tp = list(phantom[1])
        phantom = True
    else:
        Tp = repeat([])
        phantom = False
    while h:
        try:
            g, gp = next((g, gp) for g, gp in zip(T, Tp)
                         if sdm_monomial_divides(sdm_LM(g), sdm_LM(h)))
        except StopIteration:
            break
        if phantom:
            h, hp = sdm_spoly(h, g, O, K, phantom=(hp, gp))
        else:
            h = sdm_spoly(h, g, O, K)
    if phantom:
        return h, hp
    return h


def sdm_nf_buchberger_reduced(f, G, O, K):
    r"""
    Compute a reduced normal form of ``f`` with respect to ``G`` and order ``O``.

    The ground field is assumed to be ``K``, and monomials ordered according to
    ``O``.

    In contrast to weak normal forms, reduced normal forms *are* unique, but
    their computation is more expensive.

    This is the standard Buchberger algorithm for computing reduced normal forms
    with respect to *global* monomial orders [SCA, algorithm 1.6.11].

    The ``pantom`` option is not supported, so this normal form cannot be used
    as a normal form for the "extended" groebner algorithm.
    """
    h = sdm_zero()
    g = f
    while g:
        g = sdm_nf_buchberger(g, G, O, K)
        if g:
            h = sdm_add(h, [sdm_LT(g)], O, K)
            g = g[1:]
    return h


def sdm_groebner(G, NF, O, K, extended=False):
    """
    Compute a minimal standard basis of ``G`` with respect to order ``O``.

    The algorithm uses a normal form ``NF``, for example ``sdm_nf_mora``.
    The ground field is assumed to be ``K``, and monomials ordered according
    to ``O``.

    Let `N` denote the submodule generated by elements of `G`. A standard
    basis for `N` is a subset `S` of `N`, such that `in(S) = in(N)`, where for
    any subset `X` of `F`, `in(X)` denotes the submodule generated by the
    initial forms of elements of `X`. [SCA, defn 2.3.2]

    A standard basis is called minimal if no subset of it is a standard basis.

    One may show that standard bases are always generating sets.

    Minimal standard bases are not unique. This algorithm computes a
    deterministic result, depending on the particular order of `G`.

    If ``extended=True``, also compute the transition matrix from the initial
    generators to the groebner basis. That is, return a list of coefficient
    vectors, expressing the elements of the groebner basis in terms of the
    elements of ``G``.

    This functions implements the "sugar" strategy, see

    Giovini et al: "One sugar cube, please" OR Selection strategies in
    Buchberger algorithm.
    """

    # The critical pair set.
    # A critical pair is stored as (i, j, s, t) where (i, j) defines the pair
    # (by indexing S), s is the sugar of the pair, and t is the lcm of their
    # leading monomials.
    P = []

    # The eventual standard basis.
    S = []
    Sugars = []

    def Ssugar(i, j):
        """Compute the sugar of the S-poly corresponding to (i, j)."""
        LMi = sdm_LM(S[i])
        LMj = sdm_LM(S[j])
        return max(Sugars[i] - sdm_monomial_deg(LMi),
                   Sugars[j] - sdm_monomial_deg(LMj)) \
            + sdm_monomial_deg(sdm_monomial_lcm(LMi, LMj))

    ourkey = lambda p: (p[2], O(p[3]), p[1])

    def update(f, sugar, P):
        """Add f with sugar ``sugar`` to S, update P."""
        if not f:
            return P
        k = len(S)
        S.append(f)
        Sugars.append(sugar)

        LMf = sdm_LM(f)

        def removethis(pair):
            i, j, s, t = pair
            if LMf[0] != t[0]:
                return False
            tik = sdm_monomial_lcm(LMf, sdm_LM(S[i]))
            tjk = sdm_monomial_lcm(LMf, sdm_LM(S[j]))
            return tik != t and tjk != t and sdm_monomial_divides(tik, t) and \
                sdm_monomial_divides(tjk, t)
        # apply the chain criterion
        P = [p for p in P if not removethis(p)]

        # new-pair set
        N = [(i, k, Ssugar(i, k), sdm_monomial_lcm(LMf, sdm_LM(S[i])))
             for i in range(k) if LMf[0] == sdm_LM(S[i])[0]]
        # TODO apply the product criterion?
        N.sort(key=ourkey)
        remove = set()
        for i, p in enumerate(N):
            for j in range(i + 1, len(N)):
                if sdm_monomial_divides(p[3], N[j][3]):
                    remove.add(j)

        # TODO mergesort?
        P.extend(reversed([p for i, p in enumerate(N) if i not in remove]))
        P.sort(key=ourkey, reverse=True)
        # NOTE reverse-sort, because we want to pop from the end
        return P

    # Figure out the number of generators in the ground ring.
    try:
        # NOTE: we look for the first non-zero vector, take its first monomial
        #       the number of generators in the ring is one less than the length
        #       (since the zeroth entry is for the module generators)
        numgens = len(next(x[0] for x in G if x)[0]) - 1
    except StopIteration:
        # No non-zero elements in G ...
        if extended:
            return [], []
        return []

    # This list will store expressions of the elements of S in terms of the
    # initial generators
    coefficients = []

    # First add all the elements of G to S
    for i, f in enumerate(G):
        P = update(f, sdm_deg(f), P)
        if extended and f:
            coefficients.append(sdm_from_dict({(i,) + (0,)*numgens: K(1)}, O))

    # Now carry out the buchberger algorithm.
    while P:
        i, j, s, t = P.pop()
        f, g = S[i], S[j]
        if extended:
            sp, coeff = sdm_spoly(f, g, O, K,
                                  phantom=(coefficients[i], coefficients[j]))
            h, hcoeff = NF(sp, S, O, K, phantom=(coeff, coefficients))
            if h:
                coefficients.append(hcoeff)
        else:
            h = NF(sdm_spoly(f, g, O, K), S, O, K)
        P = update(h, Ssugar(i, j), P)

    # Finally interreduce the standard basis.
    # (TODO again, better data structures)
    S = {(tuple(f), i) for i, f in enumerate(S)}
    for (a, ai), (b, bi) in permutations(S, 2):
        A = sdm_LM(a)
        B = sdm_LM(b)
        if sdm_monomial_divides(A, B) and (b, bi) in S and (a, ai) in S:
            S.remove((b, bi))

    L = sorted(((list(f), i) for f, i in S), key=lambda p: O(sdm_LM(p[0])),
               reverse=True)
    res = [x[0] for x in L]
    if extended:
        return res, [coefficients[i] for _, i in L]
    return res
<|endoftext|>"""Sparse polynomial rings. """

from __future__ import annotations

from operator import add, mul, lt, le, gt, ge
from functools import reduce
from types import GeneratorType

from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.intfunc import igcd
from sympy.core.symbol import Symbol, symbols as _symbols
from sympy.core.sympify import CantSympify, sympify
from sympy.ntheory.multinomial import multinomial_coefficients
from sympy.polys.compatibility import IPolys
from sympy.polys.constructor import construct_domain
from sympy.polys.densebasic import ninf, dmp_to_dict, dmp_from_dict
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.polynomialring import PolynomialRing
from sympy.polys.heuristicgcd import heugcd
from sympy.polys.monomials import MonomialOps
from sympy.polys.orderings import lex, MonomialOrder
from sympy.polys.polyerrors import (
    CoercionFailed, GeneratorsError,
    ExactQuotientFailed, MultivariatePolynomialError)
from sympy.polys.polyoptions import (Domain as DomainOpt,
                                     Order as OrderOpt, build_options)
from sympy.polys.polyutils import (expr_from_dict, _dict_reorder,
                                   _parallel_dict_from_expr)
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public, subsets
from sympy.utilities.iterables import is_sequence
from sympy.utilities.magic import pollute

@public
def ring(symbols, domain, order: MonomialOrder|str = lex):
    """Construct a polynomial ring returning ``(ring, x_1, ..., x_n)``.

    Parameters
    ==========

    symbols : str
        Symbol/Expr or sequence of str, Symbol/Expr (non-empty)
    domain : :class:`~.Domain` or coercible
    order : :class:`~.MonomialOrder` or coercible, optional, defaults to ``lex``

    Examples
    ========

    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.orderings import lex

    >>> R, x, y, z = ring("x,y,z", ZZ, lex)
    >>> R
    Polynomial ring in x, y, z over ZZ with lex order
    >>> x + y + z
    x + y + z
    >>> type(_)
    <class 'sympy.polys.rings.PolyElement'>

    """
    _ring = PolyRing(symbols, domain, order)
    return (_ring,) + _ring.gens

@public
def xring(symbols, domain, order=lex):
    """Construct a polynomial ring returning ``(ring, (x_1, ..., x_n))``.

    Parameters
    ==========

    symbols : str
        Symbol/Expr or sequence of str, Symbol/Expr (non-empty)
    domain : :class:`~.Domain` or coercible
    order : :class:`~.MonomialOrder` or coercible, optional, defaults to ``lex``

    Examples
    ========

    >>> from sympy.polys.rings import xring
    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.orderings import lex

    >>> R, (x, y, z) = xring("x,y,z", ZZ, lex)
    >>> R
    Polynomial ring in x, y, z over ZZ with lex order
    >>> x + y + z
    x + y + z
    >>> type(_)
    <class 'sympy.polys.rings.PolyElement'>

    """
    _ring = PolyRing(symbols, domain, order)
    return (_ring, _ring.gens)

@public
def vring(symbols, domain, order=lex):
    """Construct a polynomial ring and inject ``x_1, ..., x_n`` into the global namespace.

    Parameters
    ==========

    symbols : str
        Symbol/Expr or sequence of str, Symbol/Expr (non-empty)
    domain : :class:`~.Domain` or coercible
    order : :class:`~.MonomialOrder` or coercible, optional, defaults to ``lex``

    Examples
    ========

    >>> from sympy.polys.rings import vring
    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.orderings import lex

    >>> vring("x,y,z", ZZ, lex)
    Polynomial ring in x, y, z over ZZ with lex order
    >>> x + y + z # noqa:
    x + y + z
    >>> type(_)
    <class 'sympy.polys.rings.PolyElement'>

    """
    _ring = PolyRing(symbols, domain, order)
    pollute([ sym.name for sym in _ring.symbols ], _ring.gens)
    return _ring

@public
def sring(exprs, *symbols, **options):
    """Construct a ring deriving generators and domain from options and input expressions.

    Parameters
    ==========

    exprs : :class:`~.Expr` or sequence of :class:`~.Expr` (sympifiable)
    symbols : sequence of :class:`~.Symbol`/:class:`~.Expr`
    options : keyword arguments understood by :class:`~.Options`

    Examples
    ========

    >>> from sympy import sring, symbols

    >>> x, y, z = symbols("x,y,z")
    >>> R, f = sring(x + 2*y + 3*z)
    >>> R
    Polynomial ring in x, y, z over ZZ with lex order
    >>> f
    x + 2*y + 3*z
    >>> type(_)
    <class 'sympy.polys.rings.PolyElement'>

    """
    single = False

    if not is_sequence(exprs):
        exprs, single = [exprs], True

    exprs = list(map(sympify, exprs))
    opt = build_options(symbols, options)

    # TODO: rewrite this so that it doesn't use expand() (see poly()).
    reps, opt = _parallel_dict_from_expr(exprs, opt)

    if opt.domain is None:
        coeffs = sum([ list(rep.values()) for rep in reps ], [])

        opt.domain, coeffs_dom = construct_domain(coeffs, opt=opt)

        coeff_map = dict(zip(coeffs, coeffs_dom))
        reps = [{m: coeff_map[c] for m, c in rep.items()} for rep in reps]

    _ring = PolyRing(opt.gens, opt.domain, opt.order)
    polys = list(map(_ring.from_dict, reps))

    if single:
        return (_ring, polys[0])
    else:
        return (_ring, polys)

def _parse_symbols(symbols):
    if isinstance(symbols, str):
        return _symbols(symbols, seq=True) if symbols else ()
    elif isinstance(symbols, Expr):
        return (symbols,)
    elif is_sequence(symbols):
        if all(isinstance(s, str) for s in symbols):
            return _symbols(symbols)
        elif all(isinstance(s, Expr) for s in symbols):
            return symbols

    raise GeneratorsError("expected a string, Symbol or expression or a non-empty sequence of strings, Symbols or expressions")


class PolyRing(DefaultPrinting, IPolys):
    """Multivariate distributed polynomial ring. """

    gens: tuple[PolyElement, ...]
    symbols: tuple[Expr, ...]
    ngens: int
    domain: Domain
    order: MonomialOrder

    def __new__(cls, symbols, domain, order=lex):
        symbols = tuple(_parse_symbols(symbols))
        ngens = len(symbols)
        domain = DomainOpt.preprocess(domain)
        order = OrderOpt.preprocess(order)

        _hash_tuple = (cls.__name__, symbols, ngens, domain, order)

        if domain.is_Composite and set(symbols) & set(domain.symbols):
            raise GeneratorsError("polynomial ring and it's ground domain share generators")

        obj = object.__new__(cls)
        obj._hash_tuple = _hash_tuple
        obj._hash = hash(_hash_tuple)
        obj.symbols = symbols
        obj.ngens = ngens
        obj.domain = domain
        obj.order = order

        obj.dtype = PolyElement(obj, ()).new

        obj.zero_monom = (0,)*ngens
        obj.gens = obj._gens()
        obj._gens_set = set(obj.gens)

        obj._one = [(obj.zero_monom, domain.one)]

        if ngens:
            # These expect monomials in at least one variable
            codegen = MonomialOps(ngens)
            obj.monomial_mul = codegen.mul()
            obj.monomial_pow = codegen.pow()
            obj.monomial_mulpow = codegen.mulpow()
            obj.monomial_ldiv = codegen.ldiv()
            obj.monomial_div = codegen.div()
            obj.monomial_lcm = codegen.lcm()
            obj.monomial_gcd = codegen.gcd()
        else:
            monunit = lambda a, b: ()
            obj.monomial_mul = monunit
            obj.monomial_pow = monunit
            obj.monomial_mulpow = lambda a, b, c: ()
            obj.monomial_ldiv = monunit
            obj.monomial_div = monunit
            obj.monomial_lcm = monunit
            obj.monomial_gcd = monunit


        if order is lex:
            obj.leading_expv = max
        else:
            obj.leading_expv = lambda f: max(f, key=order)

        for symbol, generator in zip(obj.symbols, obj.gens):
            if isinstance(symbol, Symbol):
                name = symbol.name

                if not hasattr(obj, name):
                    setattr(obj, name, generator)

        return obj

    def _gens(self):
        """Return a list of polynomial generators. """
        one = self.domain.one
        _gens = []
        for i in range(self.ngens):
            expv = self.monomial_basis(i)
            poly = self.zero
            poly[expv] = one
            _gens.append(poly)
        return tuple(_gens)

    def __getnewargs__(self):
        return (self.symbols, self.domain, self.order)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["leading_expv"]

        for key in state:
            if key.startswith("monomial_"):
                del state[key]

        return state

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return isinstance(other, PolyRing) and \
            (self.symbols, self.domain, self.ngens, self.order) == \
            (other.symbols, other.domain, other.ngens, other.order)

    def __ne__(self, other):
        return not self == other

    def clone(self, symbols=None, domain=None, order=None):
        # Need a hashable tuple for cacheit to work
        if symbols is not None and isinstance(symbols, list):
            symbols = tuple(symbols)
        return self._clone(symbols, domain, order)

    @cacheit
    def _clone(self, symbols, domain, order):
        return self.__class__(symbols or self.symbols, domain or self.domain, order or self.order)

    def monomial_basis(self, i):
        """Return the ith-basis element. """
        basis = [0]*self.ngens
        basis[i] = 1
        return tuple(basis)

    @property
    def zero(self):
        return self.dtype([])

    @property
    def one(self):
        return self.dtype(self._one)

    def is_element(self, element):
        """True if ``element`` is an element of this ring. False otherwise. """
        return isinstance(element, PolyElement) and element.ring == self

    def domain_new(self, element, orig_domain=None):
        return self.domain.convert(element, orig_domain)

    def ground_new(self, coeff):
        return self.term_new(self.zero_monom, coeff)

    def term_new(self, monom, coeff):
        coeff = self.domain_new(coeff)
        poly = self.zero
        if coeff:
            poly[monom] = coeff
        return poly

    def ring_new(self, element):
        if isinstance(element, PolyElement):
            if self == element.ring:
                return element
            elif isinstance(self.domain, PolynomialRing) and self.domain.ring == element.ring:
                return self.ground_new(element)
            else:
                raise NotImplementedError("conversion")
        elif isinstance(element, str):
            raise NotImplementedError("parsing")
        elif isinstance(element, dict):
            return self.from_dict(element)
        elif isinstance(element, list):
            try:
                return self.from_terms(element)
            except ValueError:
                return self.from_list(element)
        elif isinstance(element, Expr):
            return self.from_expr(element)
        else:
            return self.ground_new(element)

    __call__ = ring_new

    def from_dict(self, element, orig_domain=None):
        domain_new = self.domain_new
        poly = self.zero

        for monom, coeff in element.items():
            coeff = domain_new(coeff, orig_domain)
            if coeff:
                poly[monom] = coeff

        return poly

    def from_terms(self, element, orig_domain=None):
        return self.from_dict(dict(element), orig_domain)

    def from_list(self, element):
        return self.from_dict(dmp_to_dict(element, self.ngens-1, self.domain))

    def _rebuild_expr(self, expr, mapping):
        domain = self.domain

        def _rebuild(expr):
            generator = mapping.get(expr)

            if generator is not None:
                return generator
            elif expr.is_Add:
                return reduce(add, list(map(_rebuild, expr.args)))
            elif expr.is_Mul:
                return reduce(mul, list(map(_rebuild, expr.args)))
            else:
                # XXX: Use as_base_exp() to handle Pow(x, n) and also exp(n)
                # XXX: E can be a generator e.g. sring([exp(2)]) -> ZZ[E]
                base, exp = expr.as_base_exp()
                if exp.is_Integer and exp > 1:
                    return _rebuild(base)**int(exp)
                else:
                    return self.ground_new(domain.convert(expr))

        return _rebuild(sympify(expr))

    def from_expr(self, expr):
        mapping = dict(list(zip(self.symbols, self.gens)))

        try:
            poly = self._rebuild_expr(expr, mapping)
        except CoercionFailed:
            raise ValueError("expected an expression convertible to a polynomial in %s, got %s" % (self, expr))
        else:
            return self.ring_new(poly)

    def index(self, gen):
        """Compute index of ``gen`` in ``self.gens``. """
        if gen is None:
            if self.ngens:
                i = 0
            else:
                i = -1  # indicate impossible choice
        elif isinstance(gen, int):
            i = gen

            if 0 <= i and i < self.ngens:
                pass
            elif -self.ngens <= i and i <= -1:
                i = -i - 1
            else:
                raise ValueError("invalid generator index: %s" % gen)
        elif self.is_element(gen):
            try:
                i = self.gens.index(gen)
            except ValueError:
                raise ValueError("invalid generator: %s" % gen)
        elif isinstance(gen, str):
            try:
                i = self.symbols.index(gen)
            except ValueError:
                raise ValueError("invalid generator: %s" % gen)
        else:
            raise ValueError("expected a polynomial generator, an integer, a string or None, got %s" % gen)

        return i

    def drop(self, *gens):
        """Remove specified generators from this ring. """
        indices = set(map(self.index, gens))
        symbols = [ s for i, s in enumerate(self.symbols) if i not in indices ]

        if not symbols:
            return self.domain
        else:
            return self.clone(symbols=symbols)

    def __getitem__(self, key):
        symbols = self.symbols[key]

        if not symbols:
            return self.domain
        else:
            return self.clone(symbols=symbols)

    def to_ground(self):
        # TODO: should AlgebraicField be a Composite domain?
        if self.domain.is_Composite or hasattr(self.domain, 'domain'):
            return self.clone(domain=self.domain.domain)
        else:
            raise ValueError("%s is not a composite domain" % self.domain)

    def to_domain(self):
        return PolynomialRing(self)

    def to_field(self):
        from sympy.polys.fields import FracField
        return FracField(self.symbols, self.domain, self.order)

    @property
    def is_univariate(self):
        return len(self.gens) == 1

    @property
    def is_multivariate(self):
        return len(self.gens) > 1

    def add(self, *objs):
        """
        Add a sequence of polynomials or containers of polynomials.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> R, x = ring("x", ZZ)
        >>> R.add([ x**2 + 2*i + 3 for i in range(4) ])
        4*x**2 + 24
        >>> _.factor_list()
        (4, [(x**2 + 6, 1)])

        """
        p = self.zero

        for obj in objs:
            if is_sequence(obj, include=GeneratorType):
                p += self.add(*obj)
            else:
                p += obj

        return p

    def mul(self, *objs):
        """
        Multiply a sequence of polynomials or containers of polynomials.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> R, x = ring("x", ZZ)
        >>> R.mul([ x**2 + 2*i + 3 for i in range(4) ])
        x**8 + 24*x**6 + 206*x**4 + 744*x**2 + 945
        >>> _.factor_list()
        (1, [(x**2 + 3, 1), (x**2 + 5, 1), (x**2 + 7, 1), (x**2 + 9, 1)])

        """
        p = self.one

        for obj in objs:
            if is_sequence(obj, include=GeneratorType):
                p *= self.mul(*obj)
            else:
                p *= obj

        return p

    def drop_to_ground(self, *gens):
        r"""
        Remove specified generators from the ring and inject them into
        its domain.
        """
        indices = set(map(self.index, gens))
        symbols = [s for i, s in enumerate(self.symbols) if i not in indices]
        gens = [gen for i, gen in enumerate(self.gens) if i not in indices]

        if not symbols:
            return self
        else:
            return self.clone(symbols=symbols, domain=self.drop(*gens))

    def compose(self, other):
        """Add the generators of ``other`` to ``self``"""
        if self != other:
            syms = set(self.symbols).union(set(other.symbols))
            return self.clone(symbols=list(syms))
        else:
            return self

    def add_gens(self, symbols):
        """Add the elements of ``symbols`` as generators to ``self``"""
        syms = set(self.symbols).union(set(symbols))
        return self.clone(symbols=list(syms))

    def symmetric_poly(self, n):
        """
        Return the elementary symmetric polynomial of degree *n* over
        this ring's generators.
        """
        if n < 0 or n > self.ngens:
            raise ValueError("Cannot generate symmetric polynomial of order %s for %s" % (n, self.gens))
        elif not n:
            return self.one
        else:
            poly = self.zero
            for s in subsets(range(self.ngens), int(n)):
                monom = tuple(int(i in s) for i in range(self.ngens))
                poly += self.term_new(monom, self.domain.one)
            return poly


class PolyElement(DomainElement, DefaultPrinting, CantSympify, dict):
    """Element of multivariate distributed polynomial ring. """

    def __init__(self, ring, init):
        super().__init__(init)
        self.ring = ring
        # This check would be too slow to run every time:
        # self._check()

    def _check(self):
        assert isinstance(self, PolyElement)
        assert isinstance(self.ring, PolyRing)
        dom = self.ring.domain
        assert isinstance(dom, Domain)
        for monom, coeff in self.items():
            assert dom.of_type(coeff)
            assert len(monom) == self.ring.ngens
            assert all(isinstance(exp, int) and exp >= 0 for exp in monom)

    def new(self, init):
        return self.__class__(self.ring, init)

    def parent(self):
        return self.ring.to_domain()

    def __getnewargs__(self):
        return (self.ring, list(self.iterterms()))

    _hash = None

    def __hash__(self):
        # XXX: This computes a hash of a dictionary, but currently we don't
        # protect dictionary from being changed so any use site modifications
        # will make hashing go wrong. Use this feature with caution until we
        # figure out how to make a safe API without compromising speed of this
        # low-level class.
        _hash = self._hash
        if _hash is None:
            self._hash = _hash = hash((self.ring, frozenset(self.items())))
        return _hash

    def copy(self):
        """Return a copy of polynomial self.

        Polynomials are mutable; if one is interested in preserving
        a polynomial, and one plans to use inplace operations, one
        can copy the polynomial. This method makes a shallow copy.

        Examples
        ========

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.rings import ring

        >>> R, x, y = ring('x, y', ZZ)
        >>> p = (x + y)**2
        >>> p1 = p.copy()
        >>> p2 = p
        >>> p[R.zero_monom] = 3
        >>> p
        x**2 + 2*x*y + y**2 + 3
        >>> p1
        x**2 + 2*x*y + y**2
        >>> p2
        x**2 + 2*x*y + y**2 + 3

        """
        return self.new(self)

    def set_ring(self, new_ring):
        if self.ring == new_ring:
            return self
        elif self.ring.symbols != new_ring.symbols:
            terms = list(zip(*_dict_reorder(self, self.ring.symbols, new_ring.symbols)))
            return new_ring.from_terms(terms, self.ring.domain)
        else:
            return new_ring.from_dict(self, self.ring.domain)

    def as_expr(self, *symbols):
        if not symbols:
            symbols = self.ring.symbols
        elif len(symbols) != self.ring.ngens:
            raise ValueError(
                "Wrong number of symbols, expected %s got %s" %
                (self.ring.ngens, len(symbols))
            )

        return expr_from_dict(self.as_expr_dict(), *symbols)

    def as_expr_dict(self):
        to_sympy = self.ring.domain.to_sympy
        return {monom: to_sympy(coeff) for monom, coeff in self.iterterms()}

    def clear_denoms(self):
        domain = self.ring.domain

        if not domain.is_Field or not domain.has_assoc_Ring:
            return domain.one, self

        ground_ring = domain.get_ring()
        common = ground_ring.one
        lcm = ground_ring.lcm
        denom = domain.denom

        for coeff in self.values():
            common = lcm(common, denom(coeff))

        poly = self.new([ (k, v*common) for k, v in self.items() ])
        return common, poly

    def strip_zero(self):
        """Eliminate monomials with zero coefficient. """
        for k, v in list(self.items()):
            if not v:
                del self[k]

    def __eq__(p1, p2):
        """Equality test for polynomials.

        Examples
        ========

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.rings import ring

        >>> _, x, y = ring('x, y', ZZ)
        >>> p1 = (x + y)**2 + (x - y)**2
        >>> p1 == 4*x*y
        False
        >>> p1 == 2*(x**2 + y**2)
        True

        """
        if not p2:
            return not p1
        elif p1.ring.is_element(p2):
            return dict.__eq__(p1, p2)
        elif len(p1) > 1:
            return False
        else:
            return p1.get(p1.ring.zero_monom) == p2

    def __ne__(p1, p2):
        return not p1 == p2

    def almosteq(p1, p2, tolerance=None):
        """Approximate equality test for polynomials. """
        ring = p1.ring

        if ring.is_element(p2):
            if set(p1.keys()) != set(p2.keys()):
                return False

            almosteq = ring.domain.almosteq

            for k in p1.keys():
                if not almosteq(p1[k], p2[k], tolerance):
                    return False
            return True
        elif len(p1) > 1:
            return False
        else:
            try:
                p2 = ring.domain.convert(p2)
            except CoercionFailed:
                return False
            else:
                return ring.domain.almosteq(p1.const(), p2, tolerance)

    def sort_key(self):
        return (len(self), self.terms())

    def _cmp(p1, p2, op):
        if p1.ring.is_element(p2):
            return op(p1.sort_key(), p2.sort_key())
        else:
            return NotImplemented

    def __lt__(p1, p2):
        return p1._cmp(p2, lt)
    def __le__(p1, p2):
        return p1._cmp(p2, le)
    def __gt__(p1, p2):
        return p1._cmp(p2, gt)
    def __ge__(p1, p2):
        return p1._cmp(p2, ge)

    def _drop(self, gen):
        ring = self.ring
        i = ring.index(gen)

        if ring.ngens == 1:
            return i, ring.domain
        else:
            symbols = list(ring.symbols)
            del symbols[i]
            return i, ring.clone(symbols=symbols)

    def drop(self, gen):
        i, ring = self._drop(gen)

        if self.ring.ngens == 1:
            if self.is_ground:
                return self.coeff(1)
            else:
                raise ValueError("Cannot drop %s" % gen)
        else:
            poly = ring.zero

            for k, v in self.items():
                if k[i] == 0:
                    K = list(k)
                    del K[i]
                    poly[tuple(K)] = v
                else:
                    raise ValueError("Cannot drop %s" % gen)

            return poly

    def _drop_to_ground(self, gen):
        ring = self.ring
        i = ring.index(gen)

        symbols = list(ring.symbols)
        del symbols[i]
        return i, ring.clone(symbols=symbols, domain=ring[i])

    def drop_to_ground(self, gen):
        if self.ring.ngens == 1:
            raise ValueError("Cannot drop only generator to ground")

        i, ring = self._drop_to_ground(gen)
        poly = ring.zero
        gen = ring.domain.gens[0]

        for monom, coeff in self.iterterms():
            mon = monom[:i] + monom[i+1:]
            if mon not in poly:
                poly[mon] = (gen**monom[i]).mul_ground(coeff)
            else:
                poly[mon] += (gen**monom[i]).mul_ground(coeff)

        return poly

    def to_dense(self):
        return dmp_from_dict(self, self.ring.ngens-1, self.ring.domain)

    def to_dict(self):
        return dict(self)

    def str(self, printer, precedence, exp_pattern, mul_symbol):
        if not self:
            return printer._print(self.ring.domain.zero)
        prec_mul = precedence["Mul"]
        prec_atom = precedence["Atom"]
        ring = self.ring
        symbols = ring.symbols
        ngens = ring.ngens
        zm = ring.zero_monom
        sexpvs = []
        for expv, coeff in self.terms():
            negative = ring.domain.is_negative(coeff)
            sign = " - " if negative else " + "
            sexpvs.append(sign)
            if expv == zm:
                scoeff = printer._print(coeff)
                if negative and scoeff.startswith("-"):
                    scoeff = scoeff[1:]
            else:
                if negative:
                    coeff = -coeff
                if coeff != self.ring.domain.one:
                    scoeff = printer.parenthesize(coeff, prec_mul, strict=True)
                else:
                    scoeff = ''
            sexpv = []
            for i in range(ngens):
                exp = expv[i]
                if not exp:
                    continue
                symbol = printer.parenthesize(symbols[i], prec_atom, strict=True)
                if exp != 1:
                    if exp != int(exp) or exp < 0:
                        sexp = printer.parenthesize(exp, prec_atom, strict=False)
                    else:
                        sexp = exp
                    sexpv.append(exp_pattern % (symbol, sexp))
                else:
                    sexpv.append('%s' % symbol)
            if scoeff:
                sexpv = [scoeff] + sexpv
            sexpvs.append(mul_symbol.join(sexpv))
        if sexpvs[0] in [" + ", " - "]:
            head = sexpvs.pop(0)
            if head == " - ":
                sexpvs.insert(0, "-")
        return "".join(sexpvs)

    @property
    def is_generator(self):
        return self in self.ring._gens_set

    @property
    def is_ground(self):
        return not self or (len(self) == 1 and self.ring.zero_monom in self)

    @property
    def is_monomial(self):
        return not self or (len(self) == 1 and self.LC == 1)

    @property
    def is_term(self):
        return len(self) <= 1

    @property
    def is_negative(self):
        return self.ring.domain.is_negative(self.LC)

    @property
    def is_positive(self):
        return self.ring.domain.is_positive(self.LC)

    @property
    def is_nonnegative(self):
        return self.ring.domain.is_nonnegative(self.LC)

    @property
    def is_nonpositive(self):
        return self.ring.domain.is_nonpositive(self.LC)

    @property
    def is_zero(f):
        return not f

    @property
    def is_one(f):
        return f == f.ring.one

    @property
    def is_monic(f):
        return f.ring.domain.is_one(f.LC)

    @property
    def is_primitive(f):
        return f.ring.domain.is_one(f.content())

    @property
    def is_linear(f):
        return all(sum(monom) <= 1 for monom in f.itermonoms())

    @property
    def is_quadratic(f):
        return all(sum(monom) <= 2 for monom in f.itermonoms())

    @property
    def is_squarefree(f):
        if not f.ring.ngens:
            return True
        return f.ring.dmp_sqf_p(f)

    @property
    def is_irreducible(f):
        if not f.ring.ngens:
            return True
        return f.ring.dmp_irreducible_p(f)

    @property
    def is_cyclotomic(f):
        if f.ring.is_univariate:
            return f.ring.dup_cyclotomic_p(f)
        else:
            raise MultivariatePolynomialError("cyclotomic polynomial")

    def __neg__(self):
        return self.new([ (monom, -coeff) for monom, coeff in self.iterterms() ])

    def __pos__(self):
        return self

    def __add__(p1, p2):
        """Add two polynomials.

        Examples
        ========

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.rings import ring

        >>> _, x, y = ring('x, y', ZZ)
        >>> (x + y)**2 + (x - y)**2
        2*x**2 + 2*y**2

        """
        if not p2:
            return p1.copy()
        ring = p1.ring
        if ring.is_element(p2):
            p = p1.copy()
            get = p.get
            zero = ring.domain.zero
            for k, v in p2.items():
                v = get(k, zero) + v
                if v:
                    p[k] = v
                else:
                    del p[k]
            return p
        elif isinstance(p2, PolyElement):
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__radd__(p1)
            else:
                return NotImplemented

        try:
            cp2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            p = p1.copy()
            if not cp2:
                return p
            zm = ring.zero_monom
            if zm not in p1.keys():
                p[zm] = cp2
            else:
                if p2 == -p[zm]:
                    del p[zm]
                else:
                    p[zm] += cp2
            return p

    def __radd__(p1, n):
        p = p1.copy()
        if not n:
            return p
        ring = p1.ring
        try:
            n = ring.domain_new(n)
        except CoercionFailed:
            return NotImplemented
        else:
            zm = ring.zero_monom
            if zm not in p1.keys():
                p[zm] = n
            else:
                if n == -p[zm]:
                    del p[zm]
                else:
                    p[zm] += n
            return p

    def __sub__(p1, p2):
        """Subtract polynomial p2 from p1.

        Examples
        ========

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.rings import ring

        >>> _, x, y = ring('x, y', ZZ)
        >>> p1 = x + y**2
        >>> p2 = x*y + y**2
        >>> p1 - p2
        -x*y + x

        """
        if not p2:
            return p1.copy()
        ring = p1.ring
        if ring.is_element(p2):
            p = p1.copy()
            get = p.get
            zero = ring.domain.zero
            for k, v in p2.items():
                v = get(k, zero) - v
                if v:
                    p[k] = v
                else:
                    del p[k]
            return p
        elif isinstance(p2, PolyElement):
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__rsub__(p1)
            else:
                return NotImplemented

        try:
            p2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            p = p1.copy()
            zm = ring.zero_monom
            if zm not in p1.keys():
                p[zm] = -p2
            else:
                if p2 == p[zm]:
                    del p[zm]
                else:
                    p[zm] -= p2
            return p

    def __rsub__(p1, n):
        """n - p1 with n convertible to the coefficient domain.

        Examples
        ========

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.rings import ring

        >>> _, x, y = ring('x, y', ZZ)
        >>> p = x + y
        >>> 4 - p
        -x - y + 4

        """
        ring = p1.ring
        try:
            n = ring.domain_new(n)
        except CoercionFailed:
            return NotImplemented
        else:
            p = ring.zero
            for expv in p1:
                p[expv] = -p1[expv]
            p += n
            # p._check()
            return p

    def __mul__(p1, p2):
        """Multiply two polynomials.

        Examples
        ========

        >>> from sympy.polys.domains import QQ
        >>> from sympy.polys.rings import ring

        >>> _, x, y = ring('x, y', QQ)
        >>> p1 = x + y
        >>> p2 = x - y
        >>> p1*p2
        x**2 - y**2

        """
        ring = p1.ring
        p = ring.zero
        if not p1 or not p2:
            return p
        elif ring.is_element(p2):
            get = p.get
            zero = ring.domain.zero
            monomial_mul = ring.monomial_mul
            p2it = list(p2.items())
            for exp1, v1 in p1.items():
                for exp2, v2 in p2it:
                    exp = monomial_mul(exp1, exp2)
                    p[exp] = get(exp, zero) + v1*v2
            p.strip_zero()
            # p._check()
            return p
        elif isinstance(p2, PolyElement):
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__rmul__(p1)
            else:
                return NotImplemented

        try:
            p2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            for exp1, v1 in p1.items():
                v = v1*p2
                if v:
                    p[exp1] = v
            # p._check()
            return p

    def __rmul__(p1, p2):
        """p2 * p1 with p2 in the coefficient domain of p1.

        Examples
        ========

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.rings import ring

        >>> _, x, y = ring('x, y', ZZ)
        >>> p = x + y
        >>> 4 * p
        4*x + 4*y

        """
        p = p1.ring.zero
        if not p2:
            return p
        try:
            p2 = p.ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            for exp1, v1 in p1.items():
                v = p2*v1
                if v:
                    p[exp1] = v
            return p

    def __pow__(self, n):
        """raise polynomial to power `n`

        Examples
        ========

        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.rings import ring

        >>> _, x, y = ring('x, y', ZZ)
        >>> p = x + y**2
        >>> p**3
        x**3 + 3*x**2*y**2 + 3*x*y**4 + y**6

        """
        if not isinstance(n, int):
            raise TypeError("exponent must be an integer, got %s" % n)
        elif n < 0:
            raise ValueError("exponent must be a non-negative integer, got %s" % n)

        ring = self.ring

        if not n:
            if self:
                return ring.one
            else:
                raise ValueError("0**0")
        elif len(self) == 1:
            monom, coeff = list(self.items())[0]
            p = ring.zero
            if coeff == ring.domain.one:
                p[ring.monomial_pow(monom, n)] = coeff
            else:
                p[ring.monomial_pow(monom, n)] = coeff**n
            # p._check()
            return p

        # For ring series, we need negative and rational exponent support only
        # with monomials.
        n = int(n)
        if n < 0:
            raise ValueError("Negative exponent")

        elif n == 1:
            return self.copy()
        elif n == 2:
            return self.square()
        elif n == 3:
            return self*self.square()
        elif len(self) <= 5: # TODO: use an actual density measure
            return self._pow_multinomial(n)
        else:
            return self._pow_generic(n)

    def _pow_generic(self, n):
        p = self.ring.one
        c = self

        while True:
            if n & 1:
                p = p*c
                n -= 1
                if not n:
                    break

            c = c.square()
            n = n // 2

        return p

    def _pow_multinomial(self, n):
        multinomials = multinomial_coefficients(len(self), n).items()
        monomial_mulpow = self.ring.monomial_mulpow
        zero_monom = self.ring.zero_monom
        terms = self.items()
        zero = self.ring.domain.zero
        poly = self.ring.zero

        for multinomial, multinomial_coeff in multinomials:
            product_monom = zero_monom
            product_coeff = multinomial_coeff

            for exp, (monom, coeff) in zip(multinomial, terms):
                if exp:
                    product_monom = monomial_mulpow(product_monom, monom, exp)
                    product_coeff *= coeff**exp

            monom = tuple(product_monom)
            coeff = product_coeff

            coeff = poly.get(monom, zero) + coeff

            if coeff:
                poly[monom] = coeff
            elif monom in poly:
                del poly[monom]

        return poly

    def square(self):
        """square of a polynomial

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y = ring('x, y', ZZ)
        >>> p = x + y**2
        >>> p.square()
        x**2 + 2*x*y**2 + y**4

        """
        ring = self.ring
        p = ring.zero
        get = p.get
        keys = list(self.keys())
        zero = ring.domain.zero
        monomial_mul = ring.monomial_mul
        for i in range(len(keys)):
            k1 = keys[i]
            pk = self[k1]
            for j in range(i):
                k2 = keys[j]
                exp = monomial_mul(k1, k2)
                p[exp] = get(exp, zero) + pk*self[k2]
        p = p.imul_num(2)
        get = p.get
        for k, v in self.items():
            k2 = monomial_mul(k, k)
            p[k2] = get(k2, zero) + v**2
        p.strip_zero()
        # p._check()
        return p

    def __divmod__(p1, p2):
        ring = p1.ring

        if not p2:
            raise ZeroDivisionError("polynomial division")
        elif ring.is_element(p2):
            return p1.div(p2)
        elif isinstance(p2, PolyElement):
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__rdivmod__(p1)
            else:
                return NotImplemented

        try:
            p2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            return (p1.quo_ground(p2), p1.rem_ground(p2))

    def __rdivmod__(p1, p2):
        ring = p1.ring
        try:
            p2 = ring.ground_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            return p2.div(p1)

    def __mod__(p1, p2):
        ring = p1.ring

        if not p2:
            raise ZeroDivisionError("polynomial division")
        elif ring.is_element(p2):
            return p1.rem(p2)
        elif isinstance(p2, PolyElement):
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__rmod__(p1)
            else:
                return NotImplemented

        try:
            p2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            return p1.rem_ground(p2)

    def __rmod__(p1, p2):
        ring = p1.ring
        try:
            p2 = ring.ground_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            return p2.rem(p1)

    def __floordiv__(p1, p2):
        ring = p1.ring

        if not p2:
            raise ZeroDivisionError("polynomial division")
        elif ring.is_element(p2):
            return p1.quo(p2)
        elif isinstance(p2, PolyElement):
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__rtruediv__(p1)
            else:
                return NotImplemented

        try:
            p2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            return p1.quo_ground(p2)

    def __rfloordiv__(p1, p2):
        ring = p1.ring
        try:
            p2 = ring.ground_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            return p2.quo(p1)

    def __truediv__(p1, p2):
        ring = p1.ring

        if not p2:
            raise ZeroDivisionError("polynomial division")
        elif ring.is_element(p2):
            return p1.exquo(p2)
        elif isinstance(p2, PolyElement):
            if isinstance(ring.domain, PolynomialRing) and ring.domain.ring == p2.ring:
                pass
            elif isinstance(p2.ring.domain, PolynomialRing) and p2.ring.domain.ring == ring:
                return p2.__rtruediv__(p1)
            else:
                return NotImplemented

        try:
            p2 = ring.domain_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            return p1.quo_ground(p2)

    def __rtruediv__(p1, p2):
        ring = p1.ring
        try:
            p2 = ring.ground_new(p2)
        except CoercionFailed:
            return NotImplemented
        else:
            return p2.exquo(p1)

    def _term_div(self):
        zm = self.ring.zero_monom
        domain = self.ring.domain
        domain_quo = domain.quo
        monomial_div = self.ring.monomial_div

        if domain.is_Field:
            def term_div(a_lm_a_lc, b_lm_b_lc):
                a_lm, a_lc = a_lm_a_lc
                b_lm, b_lc = b_lm_b_lc
                if b_lm == zm: # apparently this is a very common case
                    monom = a_lm
                else:
                    monom = monomial_div(a_lm, b_lm)
                if monom is not None:
                    return monom, domain_quo(a_lc, b_lc)
                else:
                    return None
        else:
            def term_div(a_lm_a_lc, b_lm_b_lc):
                a_lm, a_lc = a_lm_a_lc
                b_lm, b_lc = b_lm_b_lc
                if b_lm == zm: # apparently this is a very common case
                    monom = a_lm
                else:
                    monom = monomial_div(a_lm, b_lm)
                if not (monom is None or a_lc % b_lc):
                    return monom, domain_quo(a_lc, b_lc)
                else:
                    return None

        return term_div

    def div(self, fv):
        """Division algorithm, see [CLO] p64.

        fv array of polynomials
           return qv, r such that
           self = sum(fv[i]*qv[i]) + r

        All polynomials are required not to be Laurent polynomials.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y = ring('x, y', ZZ)
        >>> f = x**3
        >>> f0 = x - y**2
        >>> f1 = x - y
        >>> qv, r = f.div((f0, f1))
        >>> qv[0]
        x**2 + x*y**2 + y**4
        >>> qv[1]
        0
        >>> r
        y**6

        """
        ring = self.ring
        ret_single = False
        if isinstance(fv, PolyElement):
            ret_single = True
            fv = [fv]
        if not all(fv):
            raise ZeroDivisionError("polynomial division")
        if not self:
            if ret_single:
                return ring.zero, ring.zero
            else:
                return [], ring.zero
        for f in fv:
            if f.ring != ring:
                raise ValueError('self and f must have the same ring')
        s = len(fv)
        qv = [ring.zero for i in range(s)]
        p = self.copy()
        r = ring.zero
        term_div = self._term_div()
        expvs = [fx.leading_expv() for fx in fv]
        while p:
            i = 0
            divoccurred = 0
            while i < s and divoccurred == 0:
                expv = p.leading_expv()
                term = term_div((expv, p[expv]), (expvs[i], fv[i][expvs[i]]))
                if term is not None:
                    expv1, c = term
                    qv[i] = qv[i]._iadd_monom((expv1, c))
                    p = p._iadd_poly_monom(fv[i], (expv1, -c))
                    divoccurred = 1
                else:
                    i += 1
            if not divoccurred:
                expv =  p.leading_expv()
                r = r._iadd_monom((expv, p[expv]))
                del p[expv]
        if expv == ring.zero_monom:
            r += p
        if ret_single:
            if not qv:
                return ring.zero, r
            else:
                return qv[0], r
        else:
            return qv, r

    def rem(self, G):
        f = self
        if isinstance(G, PolyElement):
            G = [G]
        if not all(G):
            raise ZeroDivisionError("polynomial division")
        ring = f.ring
        domain = ring.domain
        zero = domain.zero
        monomial_mul = ring.monomial_mul
        r = ring.zero
        term_div = f._term_div()
        ltf = f.LT
        f = f.copy()
        get = f.get
        while f:
            for g in G:
                tq = term_div(ltf, g.LT)
                if tq is not None:
                    m, c = tq
                    for mg, cg in g.iterterms():
                        m1 = monomial_mul(mg, m)
                        c1 = get(m1, zero) - c*cg
                        if not c1:
                            del f[m1]
                        else:
                            f[m1] = c1
                    ltm = f.leading_expv()
                    if ltm is not None:
                        ltf = ltm, f[ltm]

                    break
            else:
                ltm, ltc = ltf
                if ltm in r:
                    r[ltm] += ltc
                else:
                    r[ltm] = ltc
                del f[ltm]
                ltm = f.leading_expv()
                if ltm is not None:
                    ltf = ltm, f[ltm]

        return r

    def quo(f, G):
        return f.div(G)[0]

    def exquo(f, G):
        q, r = f.div(G)

        if not r:
            return q
        else:
            raise ExactQuotientFailed(f, G)

    def _iadd_monom(self, mc):
        """add to self the monomial coeff*x0**i0*x1**i1*...
        unless self is a generator -- then just return the sum of the two.

        mc is a tuple, (monom, coeff), where monomial is (i0, i1, ...)

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y = ring('x, y', ZZ)
        >>> p = x**4 + 2*y
        >>> m = (1, 2)
        >>> p1 = p._iadd_monom((m, 5))
        >>> p1
        x**4 + 5*x*y**2 + 2*y
        >>> p1 is p
        True
        >>> p = x
        >>> p1 = p._iadd_monom((m, 5))
        >>> p1
        5*x*y**2 + x
        >>> p1 is p
        False

        """
        if self in self.ring._gens_set:
            cpself = self.copy()
        else:
            cpself = self
        expv, coeff = mc
        c = cpself.get(expv)
        if c is None:
            cpself[expv] = coeff
        else:
            c += coeff
            if c:
                cpself[expv] = c
            else:
                del cpself[expv]
        return cpself

    def _iadd_poly_monom(self, p2, mc):
        """add to self the product of (p)*(coeff*x0**i0*x1**i1*...)
        unless self is a generator -- then just return the sum of the two.

        mc is a tuple, (monom, coeff), where monomial is (i0, i1, ...)

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y, z = ring('x, y, z', ZZ)
        >>> p1 = x**4 + 2*y
        >>> p2 = y + z
        >>> m = (1, 2, 3)
        >>> p1 = p1._iadd_poly_monom(p2, (m, 3))
        >>> p1
        x**4 + 3*x*y**3*z**3 + 3*x*y**2*z**4 + 2*y

        """
        p1 = self
        if p1 in p1.ring._gens_set:
            p1 = p1.copy()
        (m, c) = mc
        get = p1.get
        zero = p1.ring.domain.zero
        monomial_mul = p1.ring.monomial_mul
        for k, v in p2.items():
            ka = monomial_mul(k, m)
            coeff = get(ka, zero) + v*c
            if coeff:
                p1[ka] = coeff
            else:
                del p1[ka]
        return p1

    def degree(f, x=None):
        """
        The leading degree in ``x`` or the main variable.

        Note that the degree of 0 is negative infinity (``float('-inf')``)

        """
        i = f.ring.index(x)

        if not f:
            return ninf
        elif i < 0:
            return 0
        else:
            return max(monom[i] for monom in f.itermonoms())

    def degrees(f):
        """
        A tuple containing leading degrees in all variables.

        Note that the degree of 0 is negative infinity (``float('-inf')``)

        """
        if not f:
            return (ninf,)*f.ring.ngens
        else:
            return tuple(map(max, list(zip(*f.itermonoms()))))

    def tail_degree(f, x=None):
        """
        The tail degree in ``x`` or the main variable.

        Note that the degree of 0 is negative infinity (``float('-inf')``)

        """
        i = f.ring.index(x)

        if not f:
            return ninf
        elif i < 0:
            return 0
        else:
            return min(monom[i] for monom in f.itermonoms())

    def tail_degrees(f):
        """
        A tuple containing tail degrees in all variables.

        Note that the degree of 0 is negative infinity (``float('-inf')``)

        """
        if not f:
            return (ninf,)*f.ring.ngens
        else:
            return tuple(map(min, list(zip(*f.itermonoms()))))

    def leading_expv(self):
        """Leading monomial tuple according to the monomial ordering.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y, z = ring('x, y, z', ZZ)
        >>> p = x**4 + x**3*y + x**2*z**2 + z**7
        >>> p.leading_expv()
        (4, 0, 0)

        """
        if self:
            return self.ring.leading_expv(self)
        else:
            return None

    def _get_coeff(self, expv):
        return self.get(expv, self.ring.domain.zero)

    def coeff(self, element):
        """
        Returns the coefficient that stands next to the given monomial.

        Parameters
        ==========

        element : PolyElement (with ``is_monomial = True``) or 1

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y, z = ring("x,y,z", ZZ)
        >>> f = 3*x**2*y - x*y*z + 7*z**3 + 23

        >>> f.coeff(x**2*y)
        3
        >>> f.coeff(x*y)
        0
        >>> f.coeff(1)
        23

        """
        if element == 1:
            return self._get_coeff(self.ring.zero_monom)
        elif self.ring.is_element(element):
            terms = list(element.iterterms())
            if len(terms) == 1:
                monom, coeff = terms[0]
                if coeff == self.ring.domain.one:
                    return self._get_coeff(monom)

        raise ValueError("expected a monomial, got %s" % element)

    def const(self):
        """Returns the constant coefficient. """
        return self._get_coeff(self.ring.zero_monom)

    @property
    def LC(self):
        return self._get_coeff(self.leading_expv())

    @property
    def LM(self):
        expv = self.leading_expv()
        if expv is None:
            return self.ring.zero_monom
        else:
            return expv

    def leading_monom(self):
        """
        Leading monomial as a polynomial element.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y = ring('x, y', ZZ)
        >>> (3*x*y + y**2).leading_monom()
        x*y

        """
        p = self.ring.zero
        expv = self.leading_expv()
        if expv:
            p[expv] = self.ring.domain.one
        return p

    @property
    def LT(self):
        expv = self.leading_expv()
        if expv is None:
            return (self.ring.zero_monom, self.ring.domain.zero)
        else:
            return (expv, self._get_coeff(expv))

    def leading_term(self):
        """Leading term as a polynomial element.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y = ring('x, y', ZZ)
        >>> (3*x*y + y**2).leading_term()
        3*x*y

        """
        p = self.ring.zero
        expv = self.leading_expv()
        if expv is not None:
            p[expv] = self[expv]
        return p

    def _sorted(self, seq, order):
        if order is None:
            order = self.ring.order
        else:
            order = OrderOpt.preprocess(order)

        if order is lex:
            return sorted(seq, key=lambda monom: monom[0], reverse=True)
        else:
            return sorted(seq, key=lambda monom: order(monom[0]), reverse=True)

    def coeffs(self, order=None):
        """Ordered list of polynomial coefficients.

        Parameters
        ==========

        order : :class:`~.MonomialOrder` or coercible, optional

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.orderings import lex, grlex

        >>> _, x, y = ring("x, y", ZZ, lex)
        >>> f = x*y**7 + 2*x**2*y**3

        >>> f.coeffs()
        [2, 1]
        >>> f.coeffs(grlex)
        [1, 2]

        """
        return [ coeff for _, coeff in self.terms(order) ]

    def monoms(self, order=None):
        """Ordered list of polynomial monomials.

        Parameters
        ==========

        order : :class:`~.MonomialOrder` or coercible, optional

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.orderings import lex, grlex

        >>> _, x, y = ring("x, y", ZZ, lex)
        >>> f = x*y**7 + 2*x**2*y**3

        >>> f.monoms()
        [(2, 3), (1, 7)]
        >>> f.monoms(grlex)
        [(1, 7), (2, 3)]

        """
        return [ monom for monom, _ in self.terms(order) ]

    def terms(self, order=None):
        """Ordered list of polynomial terms.

        Parameters
        ==========

        order : :class:`~.MonomialOrder` or coercible, optional

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ
        >>> from sympy.polys.orderings import lex, grlex

        >>> _, x, y = ring("x, y", ZZ, lex)
        >>> f = x*y**7 + 2*x**2*y**3

        >>> f.terms()
        [((2, 3), 2), ((1, 7), 1)]
        >>> f.terms(grlex)
        [((1, 7), 1), ((2, 3), 2)]

        """
        return self._sorted(list(self.items()), order)

    def itercoeffs(self):
        """Iterator over coefficients of a polynomial. """
        return iter(self.values())

    def itermonoms(self):
        """Iterator over monomials of a polynomial. """
        return iter(self.keys())

    def iterterms(self):
        """Iterator over terms of a polynomial. """
        return iter(self.items())

    def listcoeffs(self):
        """Unordered list of polynomial coefficients. """
        return list(self.values())

    def listmonoms(self):
        """Unordered list of polynomial monomials. """
        return list(self.keys())

    def listterms(self):
        """Unordered list of polynomial terms. """
        return list(self.items())

    def imul_num(p, c):
        """multiply inplace the polynomial p by an element in the
        coefficient ring, provided p is not one of the generators;
        else multiply not inplace

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y = ring('x, y', ZZ)
        >>> p = x + y**2
        >>> p1 = p.imul_num(3)
        >>> p1
        3*x + 3*y**2
        >>> p1 is p
        True
        >>> p = x
        >>> p1 = p.imul_num(3)
        >>> p1
        3*x
        >>> p1 is p
        False

        """
        if p in p.ring._gens_set:
            return p*c
        if not c:
            p.clear()
            return
        for exp in p:
            p[exp] *= c
        return p

    def content(f):
        """Returns GCD of polynomial's coefficients. """
        domain = f.ring.domain
        cont = domain.zero
        gcd = domain.gcd

        for coeff in f.itercoeffs():
            cont = gcd(cont, coeff)

        return cont

    def primitive(f):
        """Returns content and a primitive polynomial. """
        cont = f.content()
        if cont == f.ring.domain.zero:
            return (cont, f)
        return cont, f.quo_ground(cont)

    def monic(f):
        """Divides all coefficients by the leading coefficient. """
        if not f:
            return f
        else:
            return f.quo_ground(f.LC)

    def mul_ground(f, x):
        if not x:
            return f.ring.zero

        terms = [ (monom, coeff*x) for monom, coeff in f.iterterms() ]
        return f.new(terms)

    def mul_monom(f, monom):
        monomial_mul = f.ring.monomial_mul
        terms = [ (monomial_mul(f_monom, monom), f_coeff) for f_monom, f_coeff in f.items() ]
        return f.new(terms)

    def mul_term(f, term):
        monom, coeff = term

        if not f or not coeff:
            return f.ring.zero
        elif monom == f.ring.zero_monom:
            return f.mul_ground(coeff)

        monomial_mul = f.ring.monomial_mul
        terms = [ (monomial_mul(f_monom, monom), f_coeff*coeff) for f_monom, f_coeff in f.items() ]
        return f.new(terms)

    def quo_ground(f, x):
        domain = f.ring.domain

        if not x:
            raise ZeroDivisionError('polynomial division')
        if not f or x == domain.one:
            return f

        if domain.is_Field:
            quo = domain.quo
            terms = [ (monom, quo(coeff, x)) for monom, coeff in f.iterterms() ]
        else:
            terms = [ (monom, coeff // x) for monom, coeff in f.iterterms() if not (coeff % x) ]

        return f.new(terms)

    def quo_term(f, term):
        monom, coeff = term

        if not coeff:
            raise ZeroDivisionError("polynomial division")
        elif not f:
            return f.ring.zero
        elif monom == f.ring.zero_monom:
            return f.quo_ground(coeff)

        term_div = f._term_div()

        terms = [ term_div(t, term) for t in f.iterterms() ]
        return f.new([ t for t in terms if t is not None ])

    def trunc_ground(f, p):
        if f.ring.domain.is_ZZ:
            terms = []

            for monom, coeff in f.iterterms():
                coeff = coeff % p

                if coeff > p // 2:
                    coeff = coeff - p

                terms.append((monom, coeff))
        else:
            terms = [ (monom, coeff % p) for monom, coeff in f.iterterms() ]

        poly = f.new(terms)
        poly.strip_zero()
        return poly

    rem_ground = trunc_ground

    def extract_ground(self, g):
        f = self
        fc = f.content()
        gc = g.content()

        gcd = f.ring.domain.gcd(fc, gc)

        f = f.quo_ground(gcd)
        g = g.quo_ground(gcd)

        return gcd, f, g

    def _norm(f, norm_func):
        if not f:
            return f.ring.domain.zero
        else:
            ground_abs = f.ring.domain.abs
            return norm_func([ ground_abs(coeff) for coeff in f.itercoeffs() ])

    def max_norm(f):
        return f._norm(max)

    def l1_norm(f):
        return f._norm(sum)

    def deflate(f, *G):
        ring = f.ring
        polys = [f] + list(G)

        J = [0]*ring.ngens

        for p in polys:
            for monom in p.itermonoms():
                for i, m in enumerate(monom):
                    J[i] = igcd(J[i], m)

        for i, b in enumerate(J):
            if not b:
                J[i] = 1

        J = tuple(J)

        if all(b == 1 for b in J):
            return J, polys

        H = []

        for p in polys:
            h = ring.zero

            for I, coeff in p.iterterms():
                N = [ i // j for i, j in zip(I, J) ]
                h[tuple(N)] = coeff

            H.append(h)

        return J, H

    def inflate(f, J):
        poly = f.ring.zero

        for I, coeff in f.iterterms():
            N = [ i*j for i, j in zip(I, J) ]
            poly[tuple(N)] = coeff

        return poly

    def lcm(self, g):
        f = self
        domain = f.ring.domain

        if not domain.is_Field:
            fc, f = f.primitive()
            gc, g = g.primitive()
            c = domain.lcm(fc, gc)

        h = (f*g).quo(f.gcd(g))

        if not domain.is_Field:
            return h.mul_ground(c)
        else:
            return h.monic()

    def gcd(f, g):
        return f.cofactors(g)[0]

    def cofactors(f, g):
        if not f and not g:
            zero = f.ring.zero
            return zero, zero, zero
        elif not f:
            h, cff, cfg = f._gcd_zero(g)
            return h, cff, cfg
        elif not g:
            h, cfg, cff = g._gcd_zero(f)
            return h, cff, cfg
        elif len(f) == 1:
            h, cff, cfg = f._gcd_monom(g)
            return h, cff, cfg
        elif len(g) == 1:
            h, cfg, cff = g._gcd_monom(f)
            return h, cff, cfg

        J, (f, g) = f.deflate(g)
        h, cff, cfg = f._gcd(g)

        return (h.inflate(J), cff.inflate(J), cfg.inflate(J))

    def _gcd_zero(f, g):
        one, zero = f.ring.one, f.ring.zero
        if g.is_nonnegative:
            return g, zero, one
        else:
            return -g, zero, -one

    def _gcd_monom(f, g):
        ring = f.ring
        ground_gcd = ring.domain.gcd
        ground_quo = ring.domain.quo
        monomial_gcd = ring.monomial_gcd
        monomial_ldiv = ring.monomial_ldiv
        mf, cf = list(f.iterterms())[0]
        _mgcd, _cgcd = mf, cf
        for mg, cg in g.iterterms():
            _mgcd = monomial_gcd(_mgcd, mg)
            _cgcd = ground_gcd(_cgcd, cg)
        h = f.new([(_mgcd, _cgcd)])
        cff = f.new([(monomial_ldiv(mf, _mgcd), ground_quo(cf, _cgcd))])
        cfg = f.new([(monomial_ldiv(mg, _mgcd), ground_quo(cg, _cgcd)) for mg, cg in g.iterterms()])
        return h, cff, cfg

    def _gcd(f, g):
        ring = f.ring

        if ring.domain.is_QQ:
            return f._gcd_QQ(g)
        elif ring.domain.is_ZZ:
            return f._gcd_ZZ(g)
        else: # TODO: don't use dense representation (port PRS algorithms)
            return ring.dmp_inner_gcd(f, g)

    def _gcd_ZZ(f, g):
        return heugcd(f, g)

    def _gcd_QQ(self, g):
        f = self
        ring = f.ring
        new_ring = ring.clone(domain=ring.domain.get_ring())

        cf, f = f.clear_denoms()
        cg, g = g.clear_denoms()

        f = f.set_ring(new_ring)
        g = g.set_ring(new_ring)

        h, cff, cfg = f._gcd_ZZ(g)

        h = h.set_ring(ring)
        c, h = h.LC, h.monic()

        cff = cff.set_ring(ring).mul_ground(ring.domain.quo(c, cf))
        cfg = cfg.set_ring(ring).mul_ground(ring.domain.quo(c, cg))

        return h, cff, cfg

    def cancel(self, g):
        """
        Cancel common factors in a rational function ``f/g``.

        Examples
        ========

        >>> from sympy.polys import ring, ZZ
        >>> R, x,y = ring("x,y", ZZ)

        >>> (2*x**2 - 2).cancel(x**2 - 2*x + 1)
        (2*x + 2, x - 1)

        """
        f = self
        ring = f.ring

        if not f:
            return f, ring.one

        domain = ring.domain

        if not (domain.is_Field and domain.has_assoc_Ring):
            _, p, q = f.cofactors(g)
        else:
            new_ring = ring.clone(domain=domain.get_ring())

            cq, f = f.clear_denoms()
            cp, g = g.clear_denoms()

            f = f.set_ring(new_ring)
            g = g.set_ring(new_ring)

            _, p, q = f.cofactors(g)
            _, cp, cq = new_ring.domain.cofactors(cp, cq)

            p = p.set_ring(ring)
            q = q.set_ring(ring)

            p = p.mul_ground(cp)
            q = q.mul_ground(cq)

        # Make canonical with respect to sign or quadrant in the case of ZZ_I
        # or QQ_I. This ensures that the LC of the denominator is canonical by
        # multiplying top and bottom by a unit of the ring.
        u = q.canonical_unit()
        if u == domain.one:
            pass
        elif u == -domain.one:
            p, q = -p, -q
        else:
            p = p.mul_ground(u)
            q = q.mul_ground(u)

        return p, q

    def canonical_unit(f):
        domain = f.ring.domain
        return domain.canonical_unit(f.LC)

    def diff(f, x):
        """Computes partial derivative in ``x``.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y = ring("x,y", ZZ)
        >>> p = x + x**2*y**3
        >>> p.diff(x)
        2*x*y**3 + 1

        """
        ring = f.ring
        i = ring.index(x)
        m = ring.monomial_basis(i)
        g = ring.zero
        for expv, coeff in f.iterterms():
            if expv[i]:
                e = ring.monomial_ldiv(expv, m)
                g[e] = ring.domain_new(coeff*expv[i])
        return g

    def __call__(f, *values):
        if 0 < len(values) <= f.ring.ngens:
            return f.evaluate(list(zip(f.ring.gens, values)))
        else:
            raise ValueError("expected at least 1 and at most %s values, got %s" % (f.ring.ngens, len(values)))

    def evaluate(self, x, a=None):
        f = self

        if isinstance(x, list) and a is None:
            (X, a), x = x[0], x[1:]
            f = f.evaluate(X, a)

            if not x:
                return f
            else:
                x = [ (Y.drop(X), a) for (Y, a) in x ]
                return f.evaluate(x)

        ring = f.ring
        i = ring.index(x)
        a = ring.domain.convert(a)

        if ring.ngens == 1:
            result = ring.domain.zero

            for (n,), coeff in f.iterterms():
                result += coeff*a**n

            return result
        else:
            poly = ring.drop(x).zero

            for monom, coeff in f.iterterms():
                n, monom = monom[i], monom[:i] + monom[i+1:]
                coeff = coeff*a**n

                if monom in poly:
                    coeff = coeff + poly[monom]

                    if coeff:
                        poly[monom] = coeff
                    else:
                        del poly[monom]
                else:
                    if coeff:
                        poly[monom] = coeff

            return poly

    def subs(self, x, a=None):
        f = self

        if isinstance(x, list) and a is None:
            for X, a in x:
                f = f.subs(X, a)
            return f

        ring = f.ring
        i = ring.index(x)
        a = ring.domain.convert(a)

        if ring.ngens == 1:
            result = ring.domain.zero

            for (n,), coeff in f.iterterms():
                result += coeff*a**n

            return ring.ground_new(result)
        else:
            poly = ring.zero

            for monom, coeff in f.iterterms():
                n, monom = monom[i], monom[:i] + (0,) + monom[i+1:]
                coeff = coeff*a**n

                if monom in poly:
                    coeff = coeff + poly[monom]

                    if coeff:
                        poly[monom] = coeff
                    else:
                        del poly[monom]
                else:
                    if coeff:
                        poly[monom] = coeff

            return poly

    def symmetrize(self):
        r"""
        Rewrite *self* in terms of elementary symmetric polynomials.

        Explanation
        ===========

        If this :py:class:`~.PolyElement` belongs to a ring of $n$ variables,
        we can try to write it as a function of the elementary symmetric
        polynomials on $n$ variables. We compute a symmetric part, and a
        remainder for any part we were not able to symmetrize.

        Examples
        ========

        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.domains import ZZ
        >>> R, x, y = ring("x,y", ZZ)

        >>> f = x**2 + y**2
        >>> f.symmetrize()
        (x**2 - 2*y, 0, [(x, x + y), (y, x*y)])

        >>> f = x**2 - y**2
        >>> f.symmetrize()
        (x**2 - 2*y, -2*y**2, [(x, x + y), (y, x*y)])

        Returns
        =======

        Triple ``(p, r, m)``
            ``p`` is a :py:class:`~.PolyElement` that represents our attempt
            to express *self* as a function of elementary symmetric
            polynomials. Each variable in ``p`` stands for one of the
            elementary symmetric polynomials. The correspondence is given
            by ``m``.

            ``r`` is the remainder.

            ``m`` is a list of pairs, giving the mapping from variables in
            ``p`` to elementary symmetric polynomials.

            The triple satisfies the equation ``p.compose(m) + r == self``.
            If the remainder ``r`` is zero, *self* is symmetric. If it is
            nonzero, we were not able to represent *self* as symmetric.

        See Also
        ========

        sympy.polys.polyfuncs.symmetrize

        References
        ==========

        .. [1] Lauer, E. Algorithms for symmetrical polynomials, Proc. 1976
            ACM Symp. on Symbolic and Algebraic Computing, NY 242-247.
            https://dl.acm.org/doi/pdf/10.1145/800205.806342

        """
        f = self.copy()
        ring = f.ring
        n = ring.ngens

        if not n:
            return f, ring.zero, []

        polys = [ring.symmetric_poly(i+1) for i in range(n)]

        poly_powers = {}
        def get_poly_power(i, n):
            if (i, n) not in poly_powers:
                poly_powers[(i, n)] = polys[i]**n
            return poly_powers[(i, n)]

        indices = list(range(n - 1))
        weights = list(range(n, 0, -1))

        symmetric = ring.zero

        while f:
            _height, _monom, _coeff = -1, None, None

            for i, (monom, coeff) in enumerate(f.terms()):
                if all(monom[i] >= monom[i + 1] for i in indices):
                    height = max(n*m for n, m in zip(weights, monom))

                    if height > _height:
                        _height, _monom, _coeff = height, monom, coeff

            if _height != -1:
                monom, coeff = _monom, _coeff
            else:
                break

            exponents = []
            for m1, m2 in zip(monom, monom[1:] + (0,)):
                exponents.append(m1 - m2)

            symmetric += ring.term_new(tuple(exponents), coeff)

            product = coeff
            for i, n in enumerate(exponents):
                product *= get_poly_power(i, n)
            f -= product

        mapping = list(zip(ring.gens, polys))

        return symmetric, f, mapping

    def compose(f, x, a=None):
        ring = f.ring
        poly = ring.zero
        gens_map = dict(zip(ring.gens, range(ring.ngens)))

        if a is not None:
            replacements = [(x, a)]
        else:
            if isinstance(x, list):
                replacements = list(x)
            elif isinstance(x, dict):
                replacements = sorted(x.items(), key=lambda k: gens_map[k[0]])
            else:
                raise ValueError("expected a generator, value pair a sequence of such pairs")

        for k, (x, g) in enumerate(replacements):
            replacements[k] = (gens_map[x], ring.ring_new(g))

        for monom, coeff in f.iterterms():
            monom = list(monom)
            subpoly = ring.one

            for i, g in replacements:
                n, monom[i] = monom[i], 0
                if n:
                    subpoly *= g**n

            subpoly = subpoly.mul_term((tuple(monom), coeff))
            poly += subpoly

        return poly

    def coeff_wrt(self, x, deg):
        """
        Coefficient of ``self`` with respect to ``x**deg``.

        Treating ``self`` as a univariate polynomial in ``x`` this finds the
        coefficient of ``x**deg`` as a polynomial in the other generators.

        Parameters
        ==========

        x : generator or generator index
            The generator or generator index to compute the expression for.
        deg : int
            The degree of the monomial to compute the expression for.

        Returns
        =======

        :py:class:`~.PolyElement`
            The coefficient of ``x**deg`` as a polynomial in the same ring.

        Examples
        ========

        >>> from sympy.polys import ring, ZZ
        >>> R, x, y, z = ring("x, y, z", ZZ)

        >>> p = 2*x**4 + 3*y**4 + 10*z**2 + 10*x*z**2
        >>> deg = 2
        >>> p.coeff_wrt(2, deg) # Using the generator index
        10*x + 10
        >>> p.coeff_wrt(z, deg) # Using the generator
        10*x + 10
        >>> p.coeff(z**2) # shows the difference between coeff and coeff_wrt
        10

        See Also
        ========

        coeff, coeffs

        """
        p = self
        i = p.ring.index(x)
        terms = [(m, c) for m, c in p.iterterms() if m[i] == deg]

        if not terms:
            return p.ring.zero

        monoms, coeffs = zip(*terms)
        monoms = [m[:i] + (0,) + m[i + 1:] for m in monoms]
        return p.ring.from_dict(dict(zip(monoms, coeffs)))

    def prem(self, g, x=None):
        """
        Pseudo-remainder of the polynomial ``self`` with respect to ``g``.

        The pseudo-quotient ``q`` and pseudo-remainder ``r`` with respect to
        ``z`` when dividing ``f`` by ``g`` satisfy ``m*f = g*q + r``,
        where ``deg(r,z) < deg(g,z)`` and
        ``m = LC(g,z)**(deg(f,z) - deg(g,z)+1)``.

        See :meth:`pdiv` for explanation of pseudo-division.


        Parameters
        ==========

        g : :py:class:`~.PolyElement`
            The polynomial to divide ``self`` by.
        x : generator or generator index, optional
            The main variable of the polynomials and default is first generator.

        Returns
        =======

        :py:class:`~.PolyElement`
            The pseudo-remainder polynomial.

        Raises
        ======

        ZeroDivisionError : If ``g`` is the zero polynomial.

        Examples
        ========

        >>> from sympy.polys import ring, ZZ
        >>> R, x, y = ring("x, y", ZZ)

        >>> f = x**2 + x*y
        >>> g = 2*x + 2
        >>> f.prem(g) # first generator is chosen by default if it is not given
        -4*y + 4
        >>> f.rem(g) # shows the difference between prem and rem
        x**2 + x*y
        >>> f.prem(g, y) # generator is given
        0
        >>> f.prem(g, 1) # generator index is given
        0

        See Also
        ========

        pdiv, pquo, pexquo, sympy.polys.domains.ring.Ring.rem

        """
        f = self
        x = f.ring.index(x)
        df = f.degree(x)
        dg = g.degree(x)

        if dg < 0:
            raise ZeroDivisionError('polynomial division')

        r, dr = f, df

        if df < dg:
            return r

        N = df - dg + 1

        lc_g = g.coeff_wrt(x, dg)

        xp = f.ring.gens[x]

        while True:

            lc_r = r.coeff_wrt(x, dr)
            j, N = dr - dg, N - 1

            R = r * lc_g
            G = g * lc_r * xp**j
            r = R - G

            dr = r.degree(x)

            if dr < dg:
                break

        c = lc_g ** N

        return r * c

    def pdiv(self, g, x=None):
        """
        Computes the pseudo-division of the polynomial ``self`` with respect to ``g``.

        The pseudo-division algorithm is used to find the pseudo-quotient ``q``
        and pseudo-remainder ``r`` such that ``m*f = g*q + r``, where ``m``
        represents the multiplier and ``f`` is the dividend polynomial.

        The pseudo-quotient ``q`` and pseudo-remainder ``r`` are polynomials in
        the variable ``x``, with the degree of ``r`` with respect to ``x``
        being strictly less than the degree of ``g`` with respect to ``x``.

        The multiplier ``m`` is defined as
        ``LC(g, x) ^ (deg(f, x) - deg(g, x) + 1)``,
        where ``LC(g, x)`` represents the leading coefficient of ``g``.

        It is important to note that in the context of the ``prem`` method,
        multivariate polynomials in a ring, such as ``R[x,y,z]``, are treated
        as univariate polynomials with coefficients that are polynomials,
        such as ``R[x,y][z]``. When dividing ``f`` by ``g`` with respect to the
        variable ``z``, the pseudo-quotient ``q`` and pseudo-remainder ``r``
        satisfy ``m*f = g*q + r``, where ``deg(r, z) < deg(g, z)``
        and ``m = LC(g, z)^(deg(f, z) - deg(g, z) + 1)``.

        In this function, the pseudo-remainder ``r`` can be obtained using the
        ``prem`` method, the pseudo-quotient ``q`` can
        be obtained using the ``pquo`` method, and
        the function ``pdiv`` itself returns a tuple ``(q, r)``.


        Parameters
        ==========

        g : :py:class:`~.PolyElement`
            The polynomial to divide ``self`` by.
        x : generator or generator index, optional
            The main variable of the polynomials and default is first generator.

        Returns
        =======

        :py:class:`~.PolyElement`
            The pseudo-division polynomial (tuple of ``q`` and ``r``).

        Raises
        ======

        ZeroDivisionError : If ``g`` is the zero polynomial.

        Examples
        ========

        >>> from sympy.polys import ring, ZZ
        >>> R, x, y = ring("x, y", ZZ)

        >>> f = x**2 + x*y
        >>> g = 2*x + 2
        >>> f.pdiv(g) # first generator is chosen by default if it is not given
        (2*x + 2*y - 2, -4*y + 4)
        >>> f.div(g) # shows the difference between pdiv and div
        (0, x**2 + x*y)
        >>> f.pdiv(g, y) # generator is given
        (2*x**3 + 2*x**2*y + 6*x**2 + 2*x*y + 8*x + 4, 0)
        >>> f.pdiv(g, 1) # generator index is given
        (2*x**3 + 2*x**2*y + 6*x**2 + 2*x*y + 8*x + 4, 0)

        See Also
        ========

        prem
            Computes only the pseudo-remainder more efficiently than
            `f.pdiv(g)[1]`.
        pquo
            Returns only the pseudo-quotient.
        pexquo
            Returns only an exact pseudo-quotient having no remainder.
        div
            Returns quotient and remainder of f and g polynomials.

        """
        f = self
        x = f.ring.index(x)

        df = f.degree(x)
        dg = g.degree(x)

        if dg < 0:
            raise ZeroDivisionError("polynomial division")

        q, r, dr = x, f, df

        if df < dg:
            return q, r

        N = df - dg + 1
        lc_g = g.coeff_wrt(x, dg)

        xp = f.ring.gens[x]

        while True:

            lc_r = r.coeff_wrt(x, dr)
            j, N = dr - dg, N - 1

            Q = q * lc_g

            q = Q + (lc_r)*xp**j

            R = r * lc_g

            G = g * lc_r * xp**j

            r = R - G

            dr = r.degree(x)

            if dr < dg:
                break

        c = lc_g**N

        q = q * c
        r = r * c

        return q, r

    def pquo(self, g, x=None):
        """
        Polynomial pseudo-quotient in multivariate polynomial ring.

        Examples
        ========
        >>> from sympy.polys import ring, ZZ
        >>> R, x,y = ring("x,y", ZZ)

        >>> f = x**2 + x*y
        >>> g = 2*x + 2*y
        >>> h = 2*x + 2
        >>> f.pquo(g)
        2*x
        >>> f.quo(g) # shows the difference between pquo and quo
        0
        >>> f.pquo(h)
        2*x + 2*y - 2
        >>> f.quo(h) # shows the difference between pquo and quo
        0

        See Also
        ========

        prem, pdiv, pexquo, sympy.polys.domains.ring.Ring.quo

        """
        f = self
        return f.pdiv(g, x)[0]

    def pexquo(self, g, x=None):
        """
        Polynomial exact pseudo-quotient in multivariate polynomial ring.

        Examples
        ========
        >>> from sympy.polys import ring, ZZ
        >>> R, x,y = ring("x,y", ZZ)

        >>> f = x**2 + x*y
        >>> g = 2*x + 2*y
        >>> h = 2*x + 2
        >>> f.pexquo(g)
        2*x
        >>> f.exquo(g) # shows the difference between pexquo and exquo
        Traceback (most recent call last):
        ...
        ExactQuotientFailed: 2*x + 2*y does not divide x**2 + x*y
        >>> f.pexquo(h)
        Traceback (most recent call last):
        ...
        ExactQuotientFailed: 2*x + 2 does not divide x**2 + x*y

        See Also
        ========

        prem, pdiv, pquo, sympy.polys.domains.ring.Ring.exquo

        """
        f = self
        q, r = f.pdiv(g, x)

        if r.is_zero:
            return q
        else:
            raise ExactQuotientFailed(f, g)

    def subresultants(self, g, x=None):
        """
        Computes the subresultant PRS of two polynomials ``self`` and ``g``.

        Parameters
        ==========

        g : :py:class:`~.PolyElement`
            The second polynomial.
        x : generator or generator index
            The variable with respect to which the subresultant sequence is computed.

        Returns
        =======

        R : list
            Returns a list polynomials representing the subresultant PRS.

        Examples
        ========

        >>> from sympy.polys import ring, ZZ
        >>> R, x, y = ring("x, y", ZZ)

        >>> f = x**2*y + x*y
        >>> g = x + y
        >>> f.subresultants(g) # first generator is chosen by default if not given
        [x**2*y + x*y, x + y, y**3 - y**2]
        >>> f.subresultants(g, 0) # generator index is given
        [x**2*y + x*y, x + y, y**3 - y**2]
        >>> f.subresultants(g, y) # generator is given
        [x**2*y + x*y, x + y, x**3 + x**2]

        """
        f = self
        x = f.ring.index(x)
        n = f.degree(x)
        m = g.degree(x)

        if n < m:
            f, g = g, f
            n, m = m, n

        if f == 0:
            return [0, 0]

        if g == 0:
            return [f, 1]

        R = [f, g]

        d = n - m
        b = (-1) ** (d + 1)

        # Compute the pseudo-remainder for f and g
        h = f.prem(g, x)
        h = h * b

        # Compute the coefficient of g with respect to x**m
        lc = g.coeff_wrt(x, m)

        c = lc ** d

        S = [1, c]

        c = -c

        while h:
            k = h.degree(x)

            R.append(h)
            f, g, m, d = g, h, k, m - k

            b = -lc * c ** d
            h = f.prem(g, x)
            h = h.exquo(b)

            lc = g.coeff_wrt(x, k)

            if d > 1:
                p = (-lc) ** d
                q = c ** (d - 1)
                c = p.exquo(q)
            else:
                c = -lc

            S.append(-c)

        return R

    # TODO: following methods should point to polynomial
    # representation independent algorithm implementations.

    def half_gcdex(f, g):
        return f.ring.dmp_half_gcdex(f, g)

    def gcdex(f, g):
        return f.ring.dmp_gcdex(f, g)

    def resultant(f, g):
        return f.ring.dmp_resultant(f, g)

    def discriminant(f):
        return f.ring.dmp_discriminant(f)

    def decompose(f):
        if f.ring.is_univariate:
            return f.ring.dup_decompose(f)
        else:
            raise MultivariatePolynomialError("polynomial decomposition")

    def shift(f, a):
        if f.ring.is_univariate:
            return f.ring.dup_shift(f, a)
        else:
            raise MultivariatePolynomialError("shift: use shift_list instead")

    def shift_list(f, a):
        return f.ring.dmp_shift(f, a)

    def sturm(f):
        if f.ring.is_univariate:
            return f.ring.dup_sturm(f)
        else:
            raise MultivariatePolynomialError("sturm sequence")

    def gff_list(f):
        return f.ring.dmp_gff_list(f)

    def norm(f):
        return f.ring.dmp_norm(f)

    def sqf_norm(f):
        return f.ring.dmp_sqf_norm(f)

    def sqf_part(f):
        return f.ring.dmp_sqf_part(f)

    def sqf_list(f, all=False):
        return f.ring.dmp_sqf_list(f, all=all)

    def factor_list(f):
        return f.ring.dmp_factor_list(f)
<|endoftext|>r"""
Efficient functions for generating Appell sequences.

An Appell sequence is a zero-indexed sequence of polynomials `p_i(x)`
satisfying `p_{i+1}'(x)=(i+1)p_i(x)` for all `i`. This definition leads
to the following iterative algorithm:

.. math :: p_0(x) = c_0,\ p_i(x) = i \int_0^x p_{i-1}(t)\,dt + c_i

The constant coefficients `c_i` are usually determined from the
just-evaluated integral and `i`.

Appell sequences satisfy the following identity from umbral calculus:

.. math :: p_n(x+y) = \sum_{k=0}^n \binom{n}{k} p_k(x) y^{n-k}

References
==========

.. [1] https://en.wikipedia.org/wiki/Appell_sequence
.. [2] Peter Luschny, "An introduction to the Bernoulli function",
       https://arxiv.org/abs/2009.06743
"""
from sympy.polys.densearith import dup_mul_ground, dup_sub_ground, dup_quo_ground
from sympy.polys.densetools import dup_eval, dup_integrate
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polytools import named_poly
from sympy.utilities import public

def dup_bernoulli(n, K):
    """Low-level implementation of Bernoulli polynomials."""
    if n < 1:
        return [K.one]
    p = [K.one, K(-1,2)]
    for i in range(2, n+1):
        p = dup_integrate(dup_mul_ground(p, K(i), K), 1, K)
        if i % 2 == 0:
            p = dup_sub_ground(p, dup_eval(p, K(1,2), K) * K(1<<(i-1), (1<<i)-1), K)
    return p

@public
def bernoulli_poly(n, x=None, polys=False):
    r"""Generates the Bernoulli polynomial `\operatorname{B}_n(x)`.

    `\operatorname{B}_n(x)` is the unique polynomial satisfying

    .. math :: \int_{x}^{x+1} \operatorname{B}_n(t) \,dt = x^n.

    Based on this, we have for nonnegative integer `s` and integer
    `a` and `b`

    .. math :: \sum_{k=a}^{b} k^s = \frac{\operatorname{B}_{s+1}(b+1) -
            \operatorname{B}_{s+1}(a)}{s+1}

    which is related to Jakob Bernoulli's original motivation for introducing
    the Bernoulli numbers, the values of these polynomials at `x = 1`.

    Examples
    ========

    >>> from sympy import summation
    >>> from sympy.abc import x
    >>> from sympy.polys import bernoulli_poly
    >>> bernoulli_poly(5, x)
    x**5 - 5*x**4/2 + 5*x**3/3 - x/6

    >>> def psum(p, a, b):
    ...     return (bernoulli_poly(p+1,b+1) - bernoulli_poly(p+1,a)) / (p+1)
    >>> psum(4, -6, 27)
    3144337
    >>> summation(x**4, (x, -6, 27))
    3144337

    >>> psum(1, 1, x).factor()
    x*(x + 1)/2
    >>> psum(2, 1, x).factor()
    x*(x + 1)*(2*x + 1)/6
    >>> psum(3, 1, x).factor()
    x**2*(x + 1)**2/4

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.

    See Also
    ========

    sympy.functions.combinatorial.numbers.bernoulli

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bernoulli_polynomials
    """
    return named_poly(n, dup_bernoulli, QQ, "Bernoulli polynomial", (x,), polys)

def dup_bernoulli_c(n, K):
    """Low-level implementation of central Bernoulli polynomials."""
    p = [K.one]
    for i in range(1, n+1):
        p = dup_integrate(dup_mul_ground(p, K(i), K), 1, K)
        if i % 2 == 0:
            p = dup_sub_ground(p, dup_eval(p, K.one, K) * K((1<<(i-1))-1, (1<<i)-1), K)
    return p

@public
def bernoulli_c_poly(n, x=None, polys=False):
    r"""Generates the central Bernoulli polynomial `\operatorname{B}_n^c(x)`.

    These are scaled and shifted versions of the plain Bernoulli polynomials,
    done in such a way that `\operatorname{B}_n^c(x)` is an even or odd function
    for even or odd `n` respectively:

    .. math :: \operatorname{B}_n^c(x) = 2^n \operatorname{B}_n
            \left(\frac{x+1}{2}\right)

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
    return named_poly(n, dup_bernoulli_c, QQ, "central Bernoulli polynomial", (x,), polys)

def dup_genocchi(n, K):
    """Low-level implementation of Genocchi polynomials."""
    if n < 1:
        return [K.zero]
    p = [-K.one]
    for i in range(2, n+1):
        p = dup_integrate(dup_mul_ground(p, K(i), K), 1, K)
        if i % 2 == 0:
            p = dup_sub_ground(p, dup_eval(p, K.one, K) // K(2), K)
    return p

@public
def genocchi_poly(n, x=None, polys=False):
    r"""Generates the Genocchi polynomial `\operatorname{G}_n(x)`.

    `\operatorname{G}_n(x)` is twice the difference between the plain and
    central Bernoulli polynomials, so has degree `n-1`:

    .. math :: \operatorname{G}_n(x) = 2 (\operatorname{B}_n(x) -
            \operatorname{B}_n^c(x))

    The factor of 2 in the definition endows `\operatorname{G}_n(x)` with
    integer coefficients.

    Parameters
    ==========

    n : int
        Degree of the polynomial plus one.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.

    See Also
    ========

    sympy.functions.combinatorial.numbers.genocchi
    """
    return named_poly(n, dup_genocchi, ZZ, "Genocchi polynomial", (x,), polys)

def dup_euler(n, K):
    """Low-level implementation of Euler polynomials."""
    return dup_quo_ground(dup_genocchi(n+1, ZZ), K(-n-1), K)

@public
def euler_poly(n, x=None, polys=False):
    r"""Generates the Euler polynomial `\operatorname{E}_n(x)`.

    These are scaled and reindexed versions of the Genocchi polynomials:

    .. math :: \operatorname{E}_n(x) = -\frac{\operatorname{G}_{n+1}(x)}{n+1}

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.

    See Also
    ========

    sympy.functions.combinatorial.numbers.euler
    """
    return named_poly(n, dup_euler, QQ, "Euler polynomial", (x,), polys)

def dup_andre(n, K):
    """Low-level implementation of Andre polynomials."""
    p = [K.one]
    for i in range(1, n+1):
        p = dup_integrate(dup_mul_ground(p, K(i), K), 1, K)
        if i % 2 == 0:
            p = dup_sub_ground(p, dup_eval(p, K.one, K), K)
    return p

@public
def andre_poly(n, x=None, polys=False):
    r"""Generates the Andre polynomial `\mathcal{A}_n(x)`.

    This is the Appell sequence where the constant coefficients form the sequence
    of Euler numbers ``euler(n)``. As such they have integer coefficients
    and parities matching the parity of `n`.

    Luschny calls these the *Swiss-knife polynomials* because their values
    at 0 and 1 can be simply transformed into both the Bernoulli and Euler
    numbers. Here they are called the Andre polynomials because
    `|\mathcal{A}_n(n\bmod 2)|` for `n \ge 0` generates what Luschny calls
    the *Andre numbers*, A000111 in the OEIS.

    Examples
    ========

    >>> from sympy import bernoulli, euler, genocchi
    >>> from sympy.abc import x
    >>> from sympy.polys import andre_poly
    >>> andre_poly(9, x)
    x**9 - 36*x**7 + 630*x**5 - 5124*x**3 + 12465*x

    >>> [andre_poly(n, 0) for n in range(11)]
    [1, 0, -1, 0, 5, 0, -61, 0, 1385, 0, -50521]
    >>> [euler(n) for n in range(11)]
    [1, 0, -1, 0, 5, 0, -61, 0, 1385, 0, -50521]
    >>> [andre_poly(n-1, 1) * n / (4**n - 2**n) for n in range(1, 11)]
    [1/2, 1/6, 0, -1/30, 0, 1/42, 0, -1/30, 0, 5/66]
    >>> [bernoulli(n) for n in range(1, 11)]
    [1/2, 1/6, 0, -1/30, 0, 1/42, 0, -1/30, 0, 5/66]
    >>> [-andre_poly(n-1, -1) * n / (-2)**(n-1) for n in range(1, 11)]
    [-1, -1, 0, 1, 0, -3, 0, 17, 0, -155]
    >>> [genocchi(n) for n in range(1, 11)]
    [-1, -1, 0, 1, 0, -3, 0, 17, 0, -155]

    >>> [abs(andre_poly(n, n%2)) for n in range(11)]
    [1, 1, 1, 2, 5, 16, 61, 272, 1385, 7936, 50521]

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.

    See Also
    ========

    sympy.functions.combinatorial.numbers.andre

    References
    ==========

    .. [1] Peter Luschny, "An introduction to the Bernoulli function",
           https://arxiv.org/abs/2009.06743
    """
    return named_poly(n, dup_andre, ZZ, "Andre polynomial", (x,), polys)
<|endoftext|>from sympy.core import S
from sympy.polys import Poly


def dispersionset(p, q=None, *gens, **args):
    r"""Compute the *dispersion set* of two polynomials.

    For two polynomials `f(x)` and `g(x)` with `\deg f > 0`
    and `\deg g > 0` the dispersion set `\operatorname{J}(f, g)` is defined as:

    .. math::
        \operatorname{J}(f, g)
        & := \{a \in \mathbb{N}_0 | \gcd(f(x), g(x+a)) \neq 1\} \\
        &  = \{a \in \mathbb{N}_0 | \deg \gcd(f(x), g(x+a)) \geq 1\}

    For a single polynomial one defines `\operatorname{J}(f) := \operatorname{J}(f, f)`.

    Examples
    ========

    >>> from sympy import poly
    >>> from sympy.polys.dispersion import dispersion, dispersionset
    >>> from sympy.abc import x

    Dispersion set and dispersion of a simple polynomial:

    >>> fp = poly((x - 3)*(x + 3), x)
    >>> sorted(dispersionset(fp))
    [0, 6]
    >>> dispersion(fp)
    6

    Note that the definition of the dispersion is not symmetric:

    >>> fp = poly(x**4 - 3*x**2 + 1, x)
    >>> gp = fp.shift(-3)
    >>> sorted(dispersionset(fp, gp))
    [2, 3, 4]
    >>> dispersion(fp, gp)
    4
    >>> sorted(dispersionset(gp, fp))
    []
    >>> dispersion(gp, fp)
    -oo

    Computing the dispersion also works over field extensions:

    >>> from sympy import sqrt
    >>> fp = poly(x**2 + sqrt(5)*x - 1, x, domain='QQ<sqrt(5)>')
    >>> gp = poly(x**2 + (2 + sqrt(5))*x + sqrt(5), x, domain='QQ<sqrt(5)>')
    >>> sorted(dispersionset(fp, gp))
    [2]
    >>> sorted(dispersionset(gp, fp))
    [1, 4]

    We can even perform the computations for polynomials
    having symbolic coefficients:

    >>> from sympy.abc import a
    >>> fp = poly(4*x**4 + (4*a + 8)*x**3 + (a**2 + 6*a + 4)*x**2 + (a**2 + 2*a)*x, x)
    >>> sorted(dispersionset(fp))
    [0, 1]

    See Also
    ========

    dispersion

    References
    ==========

    .. [1] [ManWright94]_
    .. [2] [Koepf98]_
    .. [3] [Abramov71]_
    .. [4] [Man93]_
    """
    # Check for valid input
    same = False if q is not None else True
    if same:
        q = p

    p = Poly(p, *gens, **args)
    q = Poly(q, *gens, **args)

    if not p.is_univariate or not q.is_univariate:
        raise ValueError("Polynomials need to be univariate")

    # The generator
    if not p.gen == q.gen:
        raise ValueError("Polynomials must have the same generator")
    gen = p.gen

    # We define the dispersion of constant polynomials to be zero
    if p.degree() < 1 or q.degree() < 1:
        return {0}

    # Factor p and q over the rationals
    fp = p.factor_list()
    fq = q.factor_list() if not same else fp

    # Iterate over all pairs of factors
    J = set()
    for s, unused in fp[1]:
        for t, unused in fq[1]:
            m = s.degree()
            n = t.degree()
            if n != m:
                continue
            an = s.LC()
            bn = t.LC()
            if not (an - bn).is_zero:
                continue
            # Note that the roles of `s` and `t` below are switched
            # w.r.t. the original paper. This is for consistency
            # with the description in the book of W. Koepf.
            anm1 = s.coeff_monomial(gen**(m-1))
            bnm1 = t.coeff_monomial(gen**(n-1))
            alpha = (anm1 - bnm1) / S(n*bn)
            if not alpha.is_integer:
                continue
            if alpha < 0 or alpha in J:
                continue
            if n > 1 and not (s - t.shift(alpha)).is_zero:
                continue
            J.add(alpha)

    return J


def dispersion(p, q=None, *gens, **args):
    r"""Compute the *dispersion* of polynomials.

    For two polynomials `f(x)` and `g(x)` with `\deg f > 0`
    and `\deg g > 0` the dispersion `\operatorname{dis}(f, g)` is defined as:

    .. math::
        \operatorname{dis}(f, g)
        & := \max\{ J(f,g) \cup \{0\} \} \\
        &  = \max\{ \{a \in \mathbb{N} | \gcd(f(x), g(x+a)) \neq 1\} \cup \{0\} \}

    and for a single polynomial `\operatorname{dis}(f) := \operatorname{dis}(f, f)`.
    Note that we make the definition `\max\{\} := -\infty`.

    Examples
    ========

    >>> from sympy import poly
    >>> from sympy.polys.dispersion import dispersion, dispersionset
    >>> from sympy.abc import x

    Dispersion set and dispersion of a simple polynomial:

    >>> fp = poly((x - 3)*(x + 3), x)
    >>> sorted(dispersionset(fp))
    [0, 6]
    >>> dispersion(fp)
    6

    Note that the definition of the dispersion is not symmetric:

    >>> fp = poly(x**4 - 3*x**2 + 1, x)
    >>> gp = fp.shift(-3)
    >>> sorted(dispersionset(fp, gp))
    [2, 3, 4]
    >>> dispersion(fp, gp)
    4
    >>> sorted(dispersionset(gp, fp))
    []
    >>> dispersion(gp, fp)
    -oo

    The maximum of an empty set is defined to be `-\infty`
    as seen in this example.

    Computing the dispersion also works over field extensions:

    >>> from sympy import sqrt
    >>> fp = poly(x**2 + sqrt(5)*x - 1, x, domain='QQ<sqrt(5)>')
    >>> gp = poly(x**2 + (2 + sqrt(5))*x + sqrt(5), x, domain='QQ<sqrt(5)>')
    >>> sorted(dispersionset(fp, gp))
    [2]
    >>> sorted(dispersionset(gp, fp))
    [1, 4]

    We can even perform the computations for polynomials
    having symbolic coefficients:

    >>> from sympy.abc import a
    >>> fp = poly(4*x**4 + (4*a + 8)*x**3 + (a**2 + 6*a + 4)*x**2 + (a**2 + 2*a)*x, x)
    >>> sorted(dispersionset(fp))
    [0, 1]

    See Also
    ========

    dispersionset

    References
    ==========

    .. [1] [ManWright94]_
    .. [2] [Koepf98]_
    .. [3] [Abramov71]_
    .. [4] [Man93]_
    """
    J = dispersionset(p, q, *gens, **args)
    if not J:
        # Definition for maximum of empty set
        j = S.NegativeInfinity
    else:
        j = max(J)
    return j
<|endoftext|>"""Options manager for :class:`~.Poly` and public API functions. """

from __future__ import annotations

__all__ = ["Options"]

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.sympify import sympify
from sympy.polys.polyerrors import GeneratorsError, OptionError, FlagError
from sympy.utilities import numbered_symbols, topological_sort, public
from sympy.utilities.iterables import has_dups, is_sequence

import sympy.polys

import re

class Option:
    """Base class for all kinds of options. """

    option: str | None = None

    is_Flag = False

    requires: list[str] = []
    excludes: list[str] = []

    after: list[str] = []
    before: list[str] = []

    @classmethod
    def default(cls):
        return None

    @classmethod
    def preprocess(cls, option):
        return None

    @classmethod
    def postprocess(cls, options):
        pass


class Flag(Option):
    """Base class for all kinds of flags. """

    is_Flag = True


class BooleanOption(Option):
    """An option that must have a boolean value or equivalent assigned. """

    @classmethod
    def preprocess(cls, value):
        if value in [True, False]:
            return bool(value)
        else:
            raise OptionError("'%s' must have a boolean value assigned, got %s" % (cls.option, value))


class OptionType(type):
    """Base type for all options that does registers options. """

    def __init__(cls, *args, **kwargs):
        @property
        def getter(self):
            try:
                return self[cls.option]
            except KeyError:
                return cls.default()

        setattr(Options, cls.option, getter)
        Options.__options__[cls.option] = cls


@public
class Options(dict):
    """
    Options manager for polynomial manipulation module.

    Examples
    ========

    >>> from sympy.polys.polyoptions import Options
    >>> from sympy.polys.polyoptions import build_options

    >>> from sympy.abc import x, y, z

    >>> Options((x, y, z), {'domain': 'ZZ'})
    {'auto': False, 'domain': ZZ, 'gens': (x, y, z)}

    >>> build_options((x, y, z), {'domain': 'ZZ'})
    {'auto': False, 'domain': ZZ, 'gens': (x, y, z)}

    **Options**

    * Expand --- boolean option
    * Gens --- option
    * Wrt --- option
    * Sort --- option
    * Order --- option
    * Field --- boolean option
    * Greedy --- boolean option
    * Domain --- option
    * Split --- boolean option
    * Gaussian --- boolean option
    * Extension --- option
    * Modulus --- option
    * Symmetric --- boolean option
    * Strict --- boolean option

    **Flags**

    * Auto --- boolean flag
    * Frac --- boolean flag
    * Formal --- boolean flag
    * Polys --- boolean flag
    * Include --- boolean flag
    * All --- boolean flag
    * Gen --- flag
    * Series --- boolean flag

    """

    __order__ = None
    __options__: dict[str, type[Option]] = {}

    gens: tuple[Expr, ...]
    domain: sympy.polys.domains.Domain

    def __init__(self, gens, args, flags=None, strict=False):
        dict.__init__(self)

        if gens and args.get('gens', ()):
            raise OptionError(
                "both '*gens' and keyword argument 'gens' supplied")
        elif gens:
            args = dict(args)
            args['gens'] = gens

        defaults = args.pop('defaults', {})

        def preprocess_options(args):
            for option, value in args.items():
                try:
                    cls = self.__options__[option]
                except KeyError:
                    raise OptionError("'%s' is not a valid option" % option)

                if issubclass(cls, Flag):
                    if flags is None or option not in flags:
                        if strict:
                            raise OptionError("'%s' flag is not allowed in this context" % option)

                if value is not None:
                    self[option] = cls.preprocess(value)

        preprocess_options(args)

        for key in dict(defaults):
            if key in self:
                del defaults[key]
            else:
                for option in self.keys():
                    cls = self.__options__[option]

                    if key in cls.excludes:
                        del defaults[key]
                        break

        preprocess_options(defaults)

        for option in self.keys():
            cls = self.__options__[option]

            for require_option in cls.requires:
                if self.get(require_option) is None:
                    raise OptionError("'%s' option is only allowed together with '%s'" % (option, require_option))

            for exclude_option in cls.excludes:
                if self.get(exclude_option) is not None:
                    raise OptionError("'%s' option is not allowed together with '%s'" % (option, exclude_option))

        for option in self.__order__:
            self.__options__[option].postprocess(self)

    @classmethod
    def _init_dependencies_order(cls):
        """Resolve the order of options' processing. """
        if cls.__order__ is None:
            vertices, edges = [], set()

            for name, option in cls.__options__.items():
                vertices.append(name)

                edges.update((_name, name) for _name in option.after)

                edges.update((name, _name) for _name in option.before)

            try:
                cls.__order__ = topological_sort((vertices, list(edges)))
            except ValueError:
                raise RuntimeError(
                    "cycle detected in sympy.polys options framework")

    def clone(self, updates={}):
        """Clone ``self`` and update specified options. """
        obj = dict.__new__(self.__class__)

        for option, value in self.items():
            obj[option] = value

        for option, value in updates.items():
            obj[option] = value

        return obj

    def __setattr__(self, attr, value):
        if attr in self.__options__:
            self[attr] = value
        else:
            super().__setattr__(attr, value)

    @property
    def args(self):
        args = {}

        for option, value in self.items():
            if value is not None and option != 'gens':
                cls = self.__options__[option]

                if not issubclass(cls, Flag):
                    args[option] = value

        return args

    @property
    def options(self):
        options = {}

        for option, cls in self.__options__.items():
            if not issubclass(cls, Flag):
                options[option] = getattr(self, option)

        return options

    @property
    def flags(self):
        flags = {}

        for option, cls in self.__options__.items():
            if issubclass(cls, Flag):
                flags[option] = getattr(self, option)

        return flags


class Expand(BooleanOption, metaclass=OptionType):
    """``expand`` option to polynomial manipulation functions. """

    option = 'expand'

    requires: list[str] = []
    excludes: list[str] = []

    @classmethod
    def default(cls):
        return True


class Gens(Option, metaclass=OptionType):
    """``gens`` option to polynomial manipulation functions. """

    option = 'gens'

    requires: list[str] = []
    excludes: list[str] = []

    @classmethod
    def default(cls):
        return ()

    @classmethod
    def preprocess(cls, gens):
        if isinstance(gens, Basic):
            gens = (gens,)
        elif len(gens) == 1 and is_sequence(gens[0]):
            gens = gens[0]

        if gens == (None,):
            gens = ()
        elif has_dups(gens):
            raise GeneratorsError("duplicated generators: %s" % str(gens))
        elif any(gen.is_commutative is False for gen in gens):
            raise GeneratorsError("non-commutative generators: %s" % str(gens))

        return tuple(gens)


class Wrt(Option, metaclass=OptionType):
    """``wrt`` option to polynomial manipulation functions. """

    option = 'wrt'

    requires: list[str] = []
    excludes: list[str] = []

    _re_split = re.compile(r"\s*,\s*|\s+")

    @classmethod
    def preprocess(cls, wrt):
        if isinstance(wrt, Basic):
            return [str(wrt)]
        elif isinstance(wrt, str):
            wrt = wrt.strip()
            if wrt.endswith(','):
                raise OptionError('Bad input: missing parameter.')
            if not wrt:
                return []
            return list(cls._re_split.split(wrt))
        elif hasattr(wrt, '__getitem__'):
            return list(map(str, wrt))
        else:
            raise OptionError("invalid argument for 'wrt' option")


class Sort(Option, metaclass=OptionType):
    """``sort`` option to polynomial manipulation functions. """

    option = 'sort'

    requires: list[str] = []
    excludes: list[str] = []

    @classmethod
    def default(cls):
        return []

    @classmethod
    def preprocess(cls, sort):
        if isinstance(sort, str):
            return [ gen.strip() for gen in sort.split('>') ]
        elif hasattr(sort, '__getitem__'):
            return list(map(str, sort))
        else:
            raise OptionError("invalid argument for 'sort' option")


class Order(Option, metaclass=OptionType):
    """``order`` option to polynomial manipulation functions. """

    option = 'order'

    requires: list[str] = []
    excludes: list[str] = []

    @classmethod
    def default(cls):
        return sympy.polys.orderings.lex

    @classmethod
    def preprocess(cls, order):
        return sympy.polys.orderings.monomial_key(order)


class Field(BooleanOption, metaclass=OptionType):
    """``field`` option to polynomial manipulation functions. """

    option = 'field'

    requires: list[str] = []
    excludes = ['domain', 'split', 'gaussian']


class Greedy(BooleanOption, metaclass=OptionType):
    """``greedy`` option to polynomial manipulation functions. """

    option = 'greedy'

    requires: list[str] = []
    excludes = ['domain', 'split', 'gaussian', 'extension', 'modulus', 'symmetric']


class Composite(BooleanOption, metaclass=OptionType):
    """``composite`` option to polynomial manipulation functions. """

    option = 'composite'

    @classmethod
    def default(cls):
        return None

    requires: list[str] = []
    excludes = ['domain', 'split', 'gaussian', 'extension', 'modulus', 'symmetric']


class Domain(Option, metaclass=OptionType):
    """``domain`` option to polynomial manipulation functions. """

    option = 'domain'

    requires: list[str] = []
    excludes = ['field', 'greedy', 'split', 'gaussian', 'extension']

    after = ['gens']

    _re_realfield = re.compile(r"^(R|RR)(_(\d+))?$")
    _re_complexfield = re.compile(r"^(C|CC)(_(\d+))?$")
    _re_finitefield = re.compile(r"^(FF|GF)\((\d+)\)$")
    _re_polynomial = re.compile(r"^(Z|ZZ|Q|QQ|ZZ_I|QQ_I|R|RR|C|CC)\[(.+)\]$")
    _re_fraction = re.compile(r"^(Z|ZZ|Q|QQ)\((.+)\)$")
    _re_algebraic = re.compile(r"^(Q|QQ)\<(.+)\>$")

    @classmethod
    def preprocess(cls, domain):
        if isinstance(domain, sympy.polys.domains.Domain):
            return domain
        elif hasattr(domain, 'to_domain'):
            return domain.to_domain()
        elif isinstance(domain, str):
            if domain in ['Z', 'ZZ']:
                return sympy.polys.domains.ZZ

            if domain in ['Q', 'QQ']:
                return sympy.polys.domains.QQ

            if domain == 'ZZ_I':
                return sympy.polys.domains.ZZ_I

            if domain == 'QQ_I':
                return sympy.polys.domains.QQ_I

            if domain == 'EX':
                return sympy.polys.domains.EX

            r = cls._re_realfield.match(domain)

            if r is not None:
                _, _, prec = r.groups()

                if prec is None:
                    return sympy.polys.domains.RR
                else:
                    return sympy.polys.domains.RealField(int(prec))

            r = cls._re_complexfield.match(domain)

            if r is not None:
                _, _, prec = r.groups()

                if prec is None:
                    return sympy.polys.domains.CC
                else:
                    return sympy.polys.domains.ComplexField(int(prec))

            r = cls._re_finitefield.match(domain)

            if r is not None:
                return sympy.polys.domains.FF(int(r.groups()[1]))

            r = cls._re_polynomial.match(domain)

            if r is not None:
                ground, gens = r.groups()

                gens = list(map(sympify, gens.split(',')))

                if ground in ['Z', 'ZZ']:
                    return sympy.polys.domains.ZZ.poly_ring(*gens)
                elif ground in ['Q', 'QQ']:
                    return sympy.polys.domains.QQ.poly_ring(*gens)
                elif ground in ['R', 'RR']:
                    return sympy.polys.domains.RR.poly_ring(*gens)
                elif ground == 'ZZ_I':
                    return sympy.polys.domains.ZZ_I.poly_ring(*gens)
                elif ground == 'QQ_I':
                    return sympy.polys.domains.QQ_I.poly_ring(*gens)
                else:
                    return sympy.polys.domains.CC.poly_ring(*gens)

            r = cls._re_fraction.match(domain)

            if r is not None:
                ground, gens = r.groups()

                gens = list(map(sympify, gens.split(',')))

                if ground in ['Z', 'ZZ']:
                    return sympy.polys.domains.ZZ.frac_field(*gens)
                else:
                    return sympy.polys.domains.QQ.frac_field(*gens)

            r = cls._re_algebraic.match(domain)

            if r is not None:
                gens = list(map(sympify, r.groups()[1].split(',')))
                return sympy.polys.domains.QQ.algebraic_field(*gens)

        raise OptionError('expected a valid domain specification, got %s' % domain)

    @classmethod
    def postprocess(cls, options):
        if 'gens' in options and 'domain' in options and options['domain'].is_Composite and \
                (set(options['domain'].symbols) & set(options['gens'])):
            raise GeneratorsError(
                "ground domain and generators interfere together")
        elif ('gens' not in options or not options['gens']) and \
                'domain' in options and options['domain'] == sympy.polys.domains.EX:
            raise GeneratorsError("you have to provide generators because EX domain was requested")


class Split(BooleanOption, metaclass=OptionType):
    """``split`` option to polynomial manipulation functions. """

    option = 'split'

    requires: list[str] = []
    excludes = ['field', 'greedy', 'domain', 'gaussian', 'extension',
        'modulus', 'symmetric']

    @classmethod
    def postprocess(cls, options):
        if 'split' in options:
            raise NotImplementedError("'split' option is not implemented yet")


class Gaussian(BooleanOption, metaclass=OptionType):
    """``gaussian`` option to polynomial manipulation functions. """

    option = 'gaussian'

    requires: list[str] = []
    excludes = ['field', 'greedy', 'domain', 'split', 'extension',
        'modulus', 'symmetric']

    @classmethod
    def postprocess(cls, options):
        if 'gaussian' in options and options['gaussian'] is True:
            options['domain'] = sympy.polys.domains.QQ_I
            Extension.postprocess(options)


class Extension(Option, metaclass=OptionType):
    """``extension`` option to polynomial manipulation functions. """

    option = 'extension'

    requires: list[str] = []
    excludes = ['greedy', 'domain', 'split', 'gaussian', 'modulus',
        'symmetric']

    @classmethod
    def preprocess(cls, extension):
        if extension == 1:
            return bool(extension)
        elif extension == 0:
            raise OptionError("'False' is an invalid argument for 'extension'")
        else:
            if not hasattr(extension, '__iter__'):
                extension = {extension}
            else:
                if not extension:
                    extension = None
                else:
                    extension = set(extension)

            return extension

    @classmethod
    def postprocess(cls, options):
        if 'extension' in options and options['extension'] is not True:
            options['domain'] = sympy.polys.domains.QQ.algebraic_field(
                *options['extension'])


class Modulus(Option, metaclass=OptionType):
    """``modulus`` option to polynomial manipulation functions. """

    option = 'modulus'

    requires: list[str] = []
    excludes = ['greedy', 'split', 'domain', 'gaussian', 'extension']

    @classmethod
    def preprocess(cls, modulus):
        modulus = sympify(modulus)

        if modulus.is_Integer and modulus > 0:
            return int(modulus)
        else:
            raise OptionError(
                "'modulus' must a positive integer, got %s" % modulus)

    @classmethod
    def postprocess(cls, options):
        if 'modulus' in options:
            modulus = options['modulus']
            symmetric = options.get('symmetric', True)
            options['domain'] = sympy.polys.domains.FF(modulus, symmetric)


class Symmetric(BooleanOption, metaclass=OptionType):
    """``symmetric`` option to polynomial manipulation functions. """

    option = 'symmetric'

    requires = ['modulus']
    excludes = ['greedy', 'domain', 'split', 'gaussian', 'extension']


class Strict(BooleanOption, metaclass=OptionType):
    """``strict`` option to polynomial manipulation functions. """

    option = 'strict'

    @classmethod
    def default(cls):
        return True


class Auto(BooleanOption, Flag, metaclass=OptionType):
    """``auto`` flag to polynomial manipulation functions. """

    option = 'auto'

    after = ['field', 'domain', 'extension', 'gaussian']

    @classmethod
    def default(cls):
        return True

    @classmethod
    def postprocess(cls, options):
        if ('domain' in options or 'field' in options) and 'auto' not in options:
            options['auto'] = False


class Frac(BooleanOption, Flag, metaclass=OptionType):
    """``auto`` option to polynomial manipulation functions. """

    option = 'frac'

    @classmethod
    def default(cls):
        return False


class Formal(BooleanOption, Flag, metaclass=OptionType):
    """``formal`` flag to polynomial manipulation functions. """

    option = 'formal'

    @classmethod
    def default(cls):
        return False


class Polys(BooleanOption, Flag, metaclass=OptionType):
    """``polys`` flag to polynomial manipulation functions. """

    option = 'polys'


class Include(BooleanOption, Flag, metaclass=OptionType):
    """``include`` flag to polynomial manipulation functions. """

    option = 'include'

    @classmethod
    def default(cls):
        return False


class All(BooleanOption, Flag, metaclass=OptionType):
    """``all`` flag to polynomial manipulation functions. """

    option = 'all'

    @classmethod
    def default(cls):
        return False


class Gen(Flag, metaclass=OptionType):
    """``gen`` flag to polynomial manipulation functions. """

    option = 'gen'

    @classmethod
    def default(cls):
        return 0

    @classmethod
    def preprocess(cls, gen):
        if isinstance(gen, (Basic, int)):
            return gen
        else:
            raise OptionError("invalid argument for 'gen' option")


class Series(BooleanOption, Flag, metaclass=OptionType):
    """``series`` flag to polynomial manipulation functions. """

    option = 'series'

    @classmethod
    def default(cls):
        return False


class Symbols(Flag, metaclass=OptionType):
    """``symbols`` flag to polynomial manipulation functions. """

    option = 'symbols'

    @classmethod
    def default(cls):
        return numbered_symbols('s', start=1)

    @classmethod
    def preprocess(cls, symbols):
        if hasattr(symbols, '__iter__'):
            return iter(symbols)
        else:
            raise OptionError("expected an iterator or iterable container, got %s" % symbols)


class Method(Flag, metaclass=OptionType):
    """``method`` flag to polynomial manipulation functions. """

    option = 'method'

    @classmethod
    def preprocess(cls, method):
        if isinstance(method, str):
            return method.lower()
        else:
            raise OptionError("expected a string, got %s" % method)


def build_options(gens, args=None):
    """Construct options from keyword arguments or ... options. """
    if args is None:
        gens, args = (), gens

    if len(args) != 1 or 'opt' not in args or gens:
        return Options(gens, args)
    else:
        return args['opt']


def allowed_flags(args, flags):
    """
    Allow specified flags to be used in the given context.

    Examples
    ========

    >>> from sympy.polys.polyoptions import allowed_flags
    >>> from sympy.polys.domains import ZZ

    >>> allowed_flags({'domain': ZZ}, [])

    >>> allowed_flags({'domain': ZZ, 'frac': True}, [])
    Traceback (most recent call last):
    ...
    FlagError: 'frac' flag is not allowed in this context

    >>> allowed_flags({'domain': ZZ, 'frac': True}, ['frac'])

    """
    flags = set(flags)

    for arg in args.keys():
        try:
            if Options.__options__[arg].is_Flag and arg not in flags:
                raise FlagError(
                    "'%s' flag is not allowed in this context" % arg)
        except KeyError:
            raise OptionError("'%s' is not a valid option" % arg)


def set_defaults(options, **defaults):
    """Update options with default values. """
    if 'defaults' not in options:
        options = dict(options)
        options['defaults'] = defaults

    return options

Options._init_dependencies_order()
<|endoftext|>"""
Puiseux rings. These are used by the ring_series module to represented
truncated Puiseux series. Elements of a Puiseux ring are like polynomials
except that the exponents can be negative or rational rather than just
non-negative integers.
"""

# Previously the ring_series module used PolyElement to represent Puiseux
# series. This is problematic because it means that PolyElement has to support
# negative and non-integer exponents which most polynomial representations do
# not support. This module provides an implementation of a ring for Puiseux
# series that can be used by ring_series without breaking the basic invariants
# of polynomial rings.
#
# Ideally there would be more of a proper series type that can keep track of
# not just the leading terms of a truncated series but also the precision
# of the series. For now the rings here are just introduced to keep the
# interface that ring_series was using before.

from __future__ import annotations

from sympy.polys.domains import QQ
from sympy.polys.rings import PolyRing, PolyElement
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.external.gmpy import gcd, lcm


from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Any, Unpack
    from sympy.core.expr import Expr
    from sympy.polys.domains import Domain
    from collections.abc import Iterable, Iterator


def puiseux_ring(
    symbols: str | list[Expr], domain: Domain
) -> tuple[PuiseuxRing, Unpack[tuple[PuiseuxPoly, ...]]]:
    """Construct a Puiseux ring.

    This function constructs a Puiseux ring with the given symbols and domain.

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.puiseux import puiseux_ring
    >>> R, x, y = puiseux_ring('x y', QQ)
    >>> R
    PuiseuxRing((x, y), QQ)
    >>> p = 5*x**QQ(1,2) + 7/y
    >>> p
    7*y**(-1) + 5*x**(1/2)
    """
    ring = PuiseuxRing(symbols, domain)
    return (ring,) + ring.gens # type: ignore


class PuiseuxRing:
    """Ring of Puiseux polynomials.

    A Puiseux polynomial is a truncated Puiseux series. The exponents of the
    monomials can be negative or rational numbers. This ring is used by the
    ring_series module:

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.puiseux import puiseux_ring
    >>> from sympy.polys.ring_series import rs_exp, rs_nth_root
    >>> ring, x, y = puiseux_ring('x y', QQ)
    >>> f = x**2 + y**3
    >>> f
    y**3 + x**2
    >>> f.diff(x)
    2*x
    >>> rs_exp(x, x, 5)
    1 + x + 1/2*x**2 + 1/6*x**3 + 1/24*x**4

    Importantly the Puiseux ring can represent truncated series with negative
    and fractional exponents:

    >>> f = 1/x + 1/y**2
    >>> f
    x**(-1) + y**(-2)
    >>> f.diff(x)
    -1*x**(-2)

    >>> rs_nth_root(8*x + x**2 + x**3, 3, x, 5)
    2*x**(1/3) + 1/12*x**(4/3) + 23/288*x**(7/3) + -139/20736*x**(10/3)

    See Also
    ========

    sympy.polys.ring_series.rs_series
    PuiseuxPoly
    """
    def __init__(self, symbols: str | list[Expr], domain: Domain):

        poly_ring = PolyRing(symbols, domain)

        domain = poly_ring.domain
        ngens = poly_ring.ngens

        self.poly_ring = poly_ring
        self.domain = domain

        self.symbols = poly_ring.symbols
        self.gens = tuple([self.from_poly(g) for g in poly_ring.gens])
        self.ngens = ngens

        self.zero = self.from_poly(poly_ring.zero)
        self.one = self.from_poly(poly_ring.one)

        self.zero_monom = poly_ring.zero_monom # type: ignore
        self.monomial_mul = poly_ring.monomial_mul # type: ignore

    def __repr__(self) -> str:
        return f"PuiseuxRing({self.symbols}, {self.domain})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, PuiseuxRing):
            return NotImplemented
        return self.symbols == other.symbols and self.domain == other.domain

    def from_poly(self, poly: PolyElement) -> PuiseuxPoly:
        """Create a Puiseux polynomial from a polynomial.

        >>> from sympy.polys.domains import QQ
        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.puiseux import puiseux_ring
        >>> R1, x1 = ring('x', QQ)
        >>> R2, x2 = puiseux_ring('x', QQ)
        >>> R2.from_poly(x1**2)
        x**2
        """
        return PuiseuxPoly(poly, self)

    def from_dict(self, terms: dict[tuple[int, ...], Any]) -> PuiseuxPoly:
        """Create a Puiseux polynomial from a dictionary of terms.

        >>> from sympy.polys.domains import QQ
        >>> from sympy.polys.puiseux import puiseux_ring
        >>> R, x = puiseux_ring('x', QQ)
        >>> R.from_dict({(QQ(1,2),): QQ(3)})
        3*x**(1/2)
        """
        return PuiseuxPoly.from_dict(terms, self)

    def from_int(self, n: int) -> PuiseuxPoly:
        """Create a Puiseux polynomial from an integer.

        >>> from sympy.polys.domains import QQ
        >>> from sympy.polys.puiseux import puiseux_ring
        >>> R, x = puiseux_ring('x', QQ)
        >>> R.from_int(3)
        3
        """
        return self.from_poly(self.poly_ring(n))

    def domain_new(self, arg: Any) -> Any:
        """Create a new element of the domain.

        >>> from sympy.polys.domains import QQ
        >>> from sympy.polys.puiseux import puiseux_ring
        >>> R, x = puiseux_ring('x', QQ)
        >>> R.domain_new(3)
        3
        >>> QQ.of_type(_)
        True
        """
        return self.poly_ring.domain_new(arg)

    def ground_new(self, arg: Any) -> PuiseuxPoly:
        """Create a new element from a ground element.

        >>> from sympy.polys.domains import QQ
        >>> from sympy.polys.puiseux import puiseux_ring, PuiseuxPoly
        >>> R, x = puiseux_ring('x', QQ)
        >>> R.ground_new(3)
        3
        >>> isinstance(_, PuiseuxPoly)
        True
        """
        return self.from_poly(self.poly_ring.ground_new(arg))

    def __call__(self, arg: Any) -> PuiseuxPoly:
        """Coerce an element into the ring.

        >>> from sympy.polys.domains import QQ
        >>> from sympy.polys.puiseux import puiseux_ring
        >>> R, x = puiseux_ring('x', QQ)
        >>> R(3)
        3
        >>> R({(QQ(1,2),): QQ(3)})
        3*x**(1/2)
        """
        if isinstance(arg, dict):
            return self.from_dict(arg)
        else:
            return self.from_poly(self.poly_ring(arg))

    def index(self, x: PuiseuxPoly) -> int:
        """Return the index of a generator.

        >>> from sympy.polys.domains import QQ
        >>> from sympy.polys.puiseux import puiseux_ring
        >>> R, x, y = puiseux_ring('x y', QQ)
        >>> R.index(x)
        0
        >>> R.index(y)
        1
        """
        return self.gens.index(x)


def _div_poly_monom(poly: PolyElement, monom: Iterable[int]) -> PolyElement:
    ring = poly.ring
    div = ring.monomial_div
    return ring.from_dict({div(m, monom): c for m, c in poly.terms()})


def _mul_poly_monom(poly: PolyElement, monom: Iterable[int]) -> PolyElement:
    ring = poly.ring
    mul = ring.monomial_mul
    return ring.from_dict({mul(m, monom): c for m, c in poly.terms()})


def _div_monom(monom: Iterable[int], div: Iterable[int]) -> tuple[int, ...]:
    return tuple(mi - di for mi, di in zip(monom, div))


class PuiseuxPoly:
    """Puiseux polynomial. Represents a truncated Puiseux series.

    See the :class:`PuiseuxRing` class for more information.

    >>> from sympy import QQ
    >>> from sympy.polys.puiseux import puiseux_ring
    >>> R, x, y = puiseux_ring('x, y', QQ)
    >>> p = 5*x**2 + 7*y**3
    >>> p
    7*y**3 + 5*x**2

    The internal representation of a Puiseux polynomial wraps a normal
    polynomial. To support negative powers the polynomial is considered to be
    divided by a monomial.

    >>> p2 = 1/x + 1/y**2
    >>> p2.monom # x*y**2
    (1, 2)
    >>> p2.poly
    x + y**2
    >>> (y**2 + x) / (x*y**2) == p2
    True

    To support fractional powers the polynomial is considered to be a function
    of ``x**(1/nx), y**(1/ny), ...``. The representation keeps track of a
    monomial and a list of exponent denominators so that the polynomial can be
    used to represent both negative and fractional powers.

    >>> p3 = x**QQ(1,2) + y**QQ(2,3)
    >>> p3.ns
    (2, 3)
    >>> p3.poly
    x + y**2

    See Also
    ========

    sympy.polys.puiseux.PuiseuxRing
    sympy.polys.rings.PolyElement
    """

    ring: PuiseuxRing
    poly: PolyElement
    monom: tuple[int, ...] | None
    ns: tuple[int, ...] | None

    def __new__(cls, poly: PolyElement, ring: PuiseuxRing) -> PuiseuxPoly:
        return cls._new(ring, poly, None, None)

    @classmethod
    def _new(
        cls,
        ring: PuiseuxRing,
        poly: PolyElement,
        monom: tuple[int, ...] | None,
        ns: tuple[int, ...] | None,
    ) -> PuiseuxPoly:
        poly, monom, ns = cls._normalize(poly, monom, ns)
        return cls._new_raw(ring, poly, monom, ns)

    @classmethod
    def _new_raw(
        cls,
        ring: PuiseuxRing,
        poly: PolyElement,
        monom: tuple[int, ...] | None,
        ns: tuple[int, ...] | None,
    ) -> PuiseuxPoly:
        obj = object.__new__(cls)
        obj.ring = ring
        obj.poly = poly
        obj.monom = monom
        obj.ns = ns
        return obj

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, PuiseuxPoly):
            return (
                self.poly == other.poly
                and self.monom == other.monom
                and self.ns == other.ns
            )
        elif self.monom is None and self.ns is None:
            return self.poly.__eq__(other)
        else:
            return NotImplemented

    @classmethod
    def _normalize(
        cls,
        poly: PolyElement,
        monom: tuple[int, ...] | None,
        ns: tuple[int, ...] | None,
    ) -> tuple[PolyElement, tuple[int, ...] | None, tuple[int, ...] | None]:
        if monom is None and ns is None:
            return poly, None, None

        if monom is not None:
            degs = [max(d, 0) for d in poly.tail_degrees()]
            if all(di >= mi for di, mi in zip(degs, monom)):
                poly = _div_poly_monom(poly, monom)
                monom = None
            elif any(degs):
                poly = _div_poly_monom(poly, degs)
                monom = _div_monom(monom, degs)

        if ns is not None:
            factors_d, [poly_d] = poly.deflate()
            degrees = poly.degrees()
            monom_d = monom if monom is not None else [0] * len(degrees)
            ns_new = []
            monom_new = []
            inflations = []
            for fi, ni, di, mi in zip(factors_d, ns, degrees, monom_d):
                if di == 0:
                    g = gcd(ni, mi)
                else:
                    g = gcd(fi, ni, mi)
                ns_new.append(ni // g)
                monom_new.append(mi // g)
                inflations.append(fi // g)

            if any(infl > 1 for infl in inflations):
                poly_d = poly_d.inflate(inflations)

            poly = poly_d

            if monom is not None:
                monom = tuple(monom_new)

            if all(n == 1 for n in ns_new):
                ns = None
            else:
                ns = tuple(ns_new)

        return poly, monom, ns

    @classmethod
    def _monom_fromint(
        cls,
        monom: tuple[int, ...],
        dmonom: tuple[int, ...] | None,
        ns: tuple[int, ...] | None,
    ) -> tuple[Any, ...]:
        if dmonom is not None and ns is not None:
            return tuple(QQ(mi - di, ni) for mi, di, ni in zip(monom, dmonom, ns))
        elif dmonom is not None:
            return tuple(QQ(mi - di) for mi, di in zip(monom, dmonom))
        elif ns is not None:
            return tuple(QQ(mi, ni) for mi, ni in zip(monom, ns))
        else:
            return tuple(QQ(mi) for mi in monom)

    @classmethod
    def _monom_toint(
        cls,
        monom: tuple[Any, ...],
        dmonom: tuple[int, ...] | None,
        ns: tuple[int, ...] | None,
    ) -> tuple[int, ...]:
        if dmonom is not None and ns is not None:
            return tuple(
                int((mi * ni).numerator + di) for mi, di, ni in zip(monom, dmonom, ns)
            )
        elif dmonom is not None:
            return tuple(int(mi.numerator + di) for mi, di in zip(monom, dmonom))
        elif ns is not None:
            return tuple(int((mi * ni).numerator) for mi, ni in zip(monom, ns))
        else:
            return tuple(int(mi.numerator) for mi in monom)

    def itermonoms(self) -> Iterator[tuple[Any, ...]]:
        """Iterate over the monomials of a Puiseux polynomial.

        >>> from sympy import QQ
        >>> from sympy.polys.puiseux import puiseux_ring
        >>> R, x, y = puiseux_ring('x, y', QQ)
        >>> p = 5*x**2 + 7*y**3
        >>> list(p.itermonoms())
        [(2, 0), (0, 3)]
        >>> p[(2, 0)]
        5
        """
        monom, ns = self.monom, self.ns
        for m in self.poly.itermonoms():
            yield self._monom_fromint(m, monom, ns)

    def monoms(self) -> list[tuple[Any, ...]]:
        """Return a list of the monomials of a Puiseux polynomial."""
        return list(self.itermonoms())

    def __iter__(self) -> Iterator[tuple[tuple[Any, ...], Any]]:
        return self.itermonoms()

    def __getitem__(self, monom: tuple[int, ...]) -> Any:
        monom = self._monom_toint(monom, self.monom, self.ns)
        return self.poly[monom]

    def __len__(self) -> int:
        return len(self.poly)

    def iterterms(self) -> Iterator[tuple[tuple[Any, ...], Any]]:
        """Iterate over the terms of a Puiseux polynomial.

        >>> from sympy import QQ
        >>> from sympy.polys.puiseux import puiseux_ring
        >>> R, x, y = puiseux_ring('x, y', QQ)
        >>> p = 5*x**2 + 7*y**3
        >>> list(p.iterterms())
        [((2, 0), 5), ((0, 3), 7)]
        """
        monom, ns = self.monom, self.ns
        for m, coeff in self.poly.iterterms():
            mq = self._monom_fromint(m, monom, ns)
            yield mq, coeff

    def terms(self) -> list[tuple[tuple[Any, ...], Any]]:
        """Return a list of the terms of a Puiseux polynomial."""
        return list(self.iterterms())

    @property
    def is_term(self) -> bool:
        """Return True if the Puiseux polynomial is a single term."""
        return self.poly.is_term

    def to_dict(self) -> dict[tuple[int, ...], Any]:
        """Return a dictionary representation of a Puiseux polynomial."""
        return dict(self.iterterms())

    @classmethod
    def from_dict(
        cls, terms: dict[tuple[Any, ...], Any], ring: PuiseuxRing
    ) -> PuiseuxPoly:
        """Create a Puiseux polynomial from a dictionary of terms.

        >>> from sympy import QQ
        >>> from sympy.polys.puiseux import puiseux_ring, PuiseuxPoly
        >>> R, x = puiseux_ring('x', QQ)
        >>> PuiseuxPoly.from_dict({(QQ(1,2),): QQ(3)}, R)
        3*x**(1/2)
        >>> R.from_dict({(QQ(1,2),): QQ(3)})
        3*x**(1/2)
        """
        ns = [1] * ring.ngens
        mon = [0] * ring.ngens
        for mo in terms:
            ns = [lcm(n, m.denominator) for n, m in zip(ns, mo)]
            mon = [min(m, n) for m, n in zip(mo, mon)]

        if not any(mon):
            monom = None
        else:
            monom = tuple(-int((m * n).numerator) for m, n in zip(mon, ns))

        if all(n == 1 for n in ns):
            ns_final = None
        else:
            ns_final = tuple(ns)

        terms_p = {cls._monom_toint(m, monom, ns_final): coeff for m, coeff in terms.items()}

        poly = ring.poly_ring.from_dict(terms_p)

        return cls._new(ring, poly, monom, ns_final)

    def as_expr(self) -> Expr:
        """Convert a Puiseux polynomial to :class:`~sympy.core.expr.Expr`.

        >>> from sympy import QQ, Expr
        >>> from sympy.polys.puiseux import puiseux_ring
        >>> R, x = puiseux_ring('x', QQ)
        >>> p = 5*x**2 + 7*x**3
        >>> p.as_expr()
        7*x**3 + 5*x**2
        >>> isinstance(_, Expr)
        True
        """
        ring = self.ring
        dom = ring.domain
        symbols = ring.symbols
        terms = []
        for monom, coeff in self.iterterms():
            coeff_expr = dom.to_sympy(coeff)
            monoms_expr = []
            for i, m in enumerate(monom):
                monoms_expr.append(symbols[i] ** m)
            terms.append(Mul(coeff_expr, *monoms_expr))
        return Add(*terms)

    def __repr__(self) -> str:

        def format_power(base: str, exp: int) -> str:
            if exp == 1:
                return base
            elif exp >= 0 and int(exp) == exp:
                return f"{base}**{exp}"
            else:
                return f"{base}**({exp})"

        ring = self.ring
        dom = ring.domain

        syms = [str(s) for s in ring.symbols]
        terms_str = []
        for monom, coeff in sorted(self.terms()):
            monom_str = "*".join(format_power(s, e) for s, e in zip(syms, monom) if e)
            if coeff == dom.one:
                if monom_str:
                    terms_str.append(monom_str)
                else:
                    terms_str.append("1")
            elif not monom_str:
                terms_str.append(str(coeff))
            else:
                terms_str.append(f"{coeff}*{monom_str}")

        return " + ".join(terms_str)

    def _unify(
        self, other: PuiseuxPoly
    ) -> tuple[
        PolyElement, PolyElement, tuple[int, ...] | None, tuple[int, ...] | None
    ]:
        """Bring two Puiseux polynomials to a common monom and ns."""
        poly1, monom1, ns1 = self.poly, self.monom, self.ns
        poly2, monom2, ns2 = other.poly, other.monom, other.ns

        if monom1 == monom2 and ns1 == ns2:
            return poly1, poly2, monom1, ns1

        if ns1 == ns2:
            ns = ns1
        elif ns1 is not None and ns2 is not None:
            ns = tuple(lcm(n1, n2) for n1, n2 in zip(ns1, ns2))
            f1 = [n // n1 for n, n1 in zip(ns, ns1)]
            f2 = [n // n2 for n, n2 in zip(ns, ns2)]
            poly1 = poly1.inflate(f1)
            poly2 = poly2.inflate(f2)
            if monom1 is not None:
                monom1 = tuple(m * f for m, f in zip(monom1, f1))
            if monom2 is not None:
                monom2 = tuple(m * f for m, f in zip(monom2, f2))
        elif ns2 is not None:
            ns = ns2
            poly1 = poly1.inflate(ns)
            if monom1 is not None:
                monom1 = tuple(m * n for m, n in zip(monom1, ns))
        elif ns1 is not None:
            ns = ns1
            poly2 = poly2.inflate(ns)
            if monom2 is not None:
                monom2 = tuple(m * n for m, n in zip(monom2, ns))
        else:
            assert False

        if monom1 == monom2:
            monom = monom1
        elif monom1 is not None and monom2 is not None:
            monom = tuple(max(m1, m2) for m1, m2 in zip(monom1, monom2))
            poly1 = _mul_poly_monom(poly1, _div_monom(monom, monom1))
            poly2 = _mul_poly_monom(poly2, _div_monom(monom, monom2))
        elif monom2 is not None:
            monom = monom2
            poly1 = _mul_poly_monom(poly1, monom2)
        elif monom1 is not None:
            monom = monom1
            poly2 = _mul_poly_monom(poly2, monom1)
        else:
            assert False

        return poly1, poly2, monom, ns

    def __pos__(self) -> PuiseuxPoly:
        return self

    def __neg__(self) -> PuiseuxPoly:
        return self._new_raw(self.ring, -self.poly, self.monom, self.ns)

    def __add__(self, other: Any) -> PuiseuxPoly:
        if isinstance(other, PuiseuxPoly):
            if self.ring != other.ring:
                raise ValueError("Cannot add Puiseux polynomials from different rings")
            return self._add(other)
        domain = self.ring.domain
        if isinstance(other, int):
            return self._add_ground(domain.convert_from(QQ(other), QQ))
        elif domain.of_type(other):
            return self._add_ground(other)
        else:
            return NotImplemented

    def __radd__(self, other: Any) -> PuiseuxPoly:
        domain = self.ring.domain
        if isinstance(other, int):
            return self._add_ground(domain.convert_from(QQ(other), QQ))
        elif domain.of_type(other):
            return self._add_ground(other)
        else:
            return NotImplemented

    def __sub__(self, other: Any) -> PuiseuxPoly:
        if isinstance(other, PuiseuxPoly):
            if self.ring != other.ring:
                raise ValueError(
                    "Cannot subtract Puiseux polynomials from different rings"
                )
            return self._sub(other)
        domain = self.ring.domain
        if isinstance(other, int):
            return self._sub_ground(domain.convert_from(QQ(other), QQ))
        elif domain.of_type(other):
            return self._sub_ground(other)
        else:
            return NotImplemented

    def __rsub__(self, other: Any) -> PuiseuxPoly:
        domain = self.ring.domain
        if isinstance(other, int):
            return self._rsub_ground(domain.convert_from(QQ(other), QQ))
        elif domain.of_type(other):
            return self._rsub_ground(other)
        else:
            return NotImplemented

    def __mul__(self, other: Any) -> PuiseuxPoly:
        if isinstance(other, PuiseuxPoly):
            if self.ring != other.ring:
                raise ValueError(
                    "Cannot multiply Puiseux polynomials from different rings"
                )
            return self._mul(other)
        domain = self.ring.domain
        if isinstance(other, int):
            return self._mul_ground(domain.convert_from(QQ(other), QQ))
        elif domain.of_type(other):
            return self._mul_ground(other)
        else:
            return NotImplemented

    def __rmul__(self, other: Any) -> PuiseuxPoly:
        domain = self.ring.domain
        if isinstance(other, int):
            return self._mul_ground(domain.convert_from(QQ(other), QQ))
        elif domain.of_type(other):
            return self._mul_ground(other)
        else:
            return NotImplemented

    def __pow__(self, other: Any) -> PuiseuxPoly:
        if isinstance(other, int):
            if other >= 0:
                return self._pow_pint(other)
            else:
                return self._pow_nint(-other)
        elif QQ.of_type(other):
            return self._pow_rational(other)
        else:
            return NotImplemented

    def __truediv__(self, other: Any) -> PuiseuxPoly:
        if isinstance(other, PuiseuxPoly):
            if self.ring != other.ring:
                raise ValueError(
                    "Cannot divide Puiseux polynomials from different rings"
                )
            return self._mul(other._inv())
        domain = self.ring.domain
        if isinstance(other, int):
            return self._mul_ground(domain.convert_from(QQ(1, other), QQ))
        elif domain.of_type(other):
            return self._div_ground(other)
        else:
            return NotImplemented

    def __rtruediv__(self, other: Any) -> PuiseuxPoly:
        if isinstance(other, int):
            return self._inv()._mul_ground(self.ring.domain.convert_from(QQ(other), QQ))
        elif self.ring.domain.of_type(other):
            return self._inv()._mul_ground(other)
        else:
            return NotImplemented

    def _add(self, other: PuiseuxPoly) -> PuiseuxPoly:
        poly1, poly2, monom, ns = self._unify(other)
        return self._new(self.ring, poly1 + poly2, monom, ns)

    def _add_ground(self, ground: Any) -> PuiseuxPoly:
        return self._add(self.ring.ground_new(ground))

    def _sub(self, other: PuiseuxPoly) -> PuiseuxPoly:
        poly1, poly2, monom, ns = self._unify(other)
        return self._new(self.ring, poly1 - poly2, monom, ns)

    def _sub_ground(self, ground: Any) -> PuiseuxPoly:
        return self._sub(self.ring.ground_new(ground))

    def _rsub_ground(self, ground: Any) -> PuiseuxPoly:
        return self.ring.ground_new(ground)._sub(self)

    def _mul(self, other: PuiseuxPoly) -> PuiseuxPoly:
        poly1, poly2, monom, ns = self._unify(other)
        if monom is not None:
            monom = tuple(2 * e for e in monom)
        return self._new(self.ring, poly1 * poly2, monom, ns)

    def _mul_ground(self, ground: Any) -> PuiseuxPoly:
        return self._new_raw(self.ring, self.poly * ground, self.monom, self.ns)

    def _div_ground(self, ground: Any) -> PuiseuxPoly:
        return self._new_raw(self.ring, self.poly / ground, self.monom, self.ns)

    def _pow_pint(self, n: int) -> PuiseuxPoly:
        assert n >= 0
        monom = self.monom
        if monom is not None:
            monom = tuple(m * n for m in monom)
        return self._new(self.ring, self.poly**n, monom, self.ns)

    def _pow_nint(self, n: int) -> PuiseuxPoly:
        return self._inv()._pow_pint(n)

    def _pow_rational(self, n: Any) -> PuiseuxPoly:
        if not self.is_term:
            raise ValueError("Only monomials can be raised to a rational power")
        [(monom, coeff)] = self.terms()
        domain = self.ring.domain
        if not domain.is_one(coeff):
            raise ValueError("Only monomials can be raised to a rational power")
        monom = tuple(m * n for m in monom)
        return self.ring.from_dict({monom: domain.one})

    def _inv(self) -> PuiseuxPoly:
        if not self.is_term:
            raise ValueError("Only terms can be inverted")
        [(monom, coeff)] = self.terms()
        domain = self.ring.domain
        if not domain.is_Field and not domain.is_one(coeff):
            raise ValueError("Cannot invert non-unit coefficient")
        monom = tuple(-m for m in monom)
        coeff = 1 / coeff
        return self.ring.from_dict({monom: coeff})

    def diff(self, x: PuiseuxPoly) -> PuiseuxPoly:
        """Differentiate a Puiseux polynomial with respect to a variable.

        >>> from sympy import QQ
        >>> from sympy.polys.puiseux import puiseux_ring
        >>> R, x, y = puiseux_ring('x, y', QQ)
        >>> p = 5*x**2 + 7*y**3
        >>> p.diff(x)
        10*x
        >>> p.diff(y)
        21*y**2
        """
        ring = self.ring
        i = ring.index(x)
        g = {}
        for expv, coeff in self.iterterms():
            n = expv[i]
            if n:
                e = list(expv)
                e[i] -= 1
                g[tuple(e)] = coeff * n
        return ring(g)
<|endoftext|>"""Algorithms for partial fraction decomposition of rational functions. """


from sympy.core import S, Add, sympify, Function, Lambda, Dummy
from sympy.core.traversal import preorder_traversal
from sympy.polys import Poly, RootSum, cancel, factor
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polyoptions import allowed_flags, set_defaults
from sympy.polys.polytools import parallel_poly_from_expr
from sympy.utilities import numbered_symbols, take, xthreaded, public


@xthreaded
@public
def apart(f, x=None, full=False, **options):
    """
    Compute partial fraction decomposition of a rational function.

    Given a rational function ``f``, computes the partial fraction
    decomposition of ``f``. Two algorithms are available: One is based on the
    undetermined coefficients method, the other is Bronstein's full partial
    fraction decomposition algorithm.

    The undetermined coefficients method (selected by ``full=False``) uses
    polynomial factorization (and therefore accepts the same options as
    factor) for the denominator. Per default it works over the rational
    numbers, therefore decomposition of denominators with non-rational roots
    (e.g. irrational, complex roots) is not supported by default (see options
    of factor).

    Bronstein's algorithm can be selected by using ``full=True`` and allows a
    decomposition of denominators with non-rational roots. A human-readable
    result can be obtained via ``doit()`` (see examples below).

    Examples
    ========

    >>> from sympy.polys.partfrac import apart
    >>> from sympy.abc import x, y

    By default, using the undetermined coefficients method:

    >>> apart(y/(x + 2)/(x + 1), x)
    -y/(x + 2) + y/(x + 1)

    The undetermined coefficients method does not provide a result when the
    denominators roots are not rational:

    >>> apart(y/(x**2 + x + 1), x)
    y/(x**2 + x + 1)

    You can choose Bronstein's algorithm by setting ``full=True``:

    >>> apart(y/(x**2 + x + 1), x, full=True)
    RootSum(_w**2 + _w + 1, Lambda(_a, (-2*_a*y/3 - y/3)/(-_a + x)))

    Calling ``doit()`` yields a human-readable result:

    >>> apart(y/(x**2 + x + 1), x, full=True).doit()
    (-y/3 - 2*y*(-1/2 - sqrt(3)*I/2)/3)/(x + 1/2 + sqrt(3)*I/2) + (-y/3 -
        2*y*(-1/2 + sqrt(3)*I/2)/3)/(x + 1/2 - sqrt(3)*I/2)


    See Also
    ========

    apart_list, assemble_partfrac_list
    """
    allowed_flags(options, [])

    f = sympify(f)

    if f.is_Atom:
        return f
    else:
        P, Q = f.as_numer_denom()

    _options = options.copy()
    options = set_defaults(options, extension=True)
    try:
        (P, Q), opt = parallel_poly_from_expr((P, Q), x, **options)
    except PolynomialError as msg:
        if f.is_commutative:
            raise PolynomialError(msg)
        # non-commutative
        if f.is_Mul:
            c, nc = f.args_cnc(split_1=False)
            nc = f.func(*nc)
            if c:
                c = apart(f.func._from_args(c), x=x, full=full, **_options)
                return c*nc
            else:
                return nc
        elif f.is_Add:
            c = []
            nc = []
            for i in f.args:
                if i.is_commutative:
                    c.append(i)
                else:
                    try:
                        nc.append(apart(i, x=x, full=full, **_options))
                    except NotImplementedError:
                        nc.append(i)
            return apart(f.func(*c), x=x, full=full, **_options) + f.func(*nc)
        else:
            reps = []
            pot = preorder_traversal(f)
            next(pot)
            for e in pot:
                try:
                    reps.append((e, apart(e, x=x, full=full, **_options)))
                    pot.skip()  # this was handled successfully
                except NotImplementedError:
                    pass
            return f.xreplace(dict(reps))

    if P.is_multivariate:
        fc = f.cancel()
        if fc != f:
            return apart(fc, x=x, full=full, **_options)

        raise NotImplementedError(
            "multivariate partial fraction decomposition")

    common, P, Q = P.cancel(Q)

    poly, P = P.div(Q, auto=True)
    P, Q = P.rat_clear_denoms(Q)

    if Q.degree() <= 1:
        partial = P/Q
    else:
        if not full:
            partial = apart_undetermined_coeffs(P, Q)
        else:
            partial = apart_full_decomposition(P, Q)

    terms = S.Zero

    for term in Add.make_args(partial):
        if term.has(RootSum):
            terms += term
        else:
            terms += factor(term)

    return common*(poly.as_expr() + terms)


def apart_undetermined_coeffs(P, Q):
    """Partial fractions via method of undetermined coefficients. """
    X = numbered_symbols(cls=Dummy)
    partial, symbols = [], []

    _, factors = Q.factor_list()

    for f, k in factors:
        n, q = f.degree(), Q

        for i in range(1, k + 1):
            coeffs, q = take(X, n), q.quo(f)
            partial.append((coeffs, q, f, i))
            symbols.extend(coeffs)

    dom = Q.get_domain().inject(*symbols)
    F = Poly(0, Q.gen, domain=dom)

    for i, (coeffs, q, f, k) in enumerate(partial):
        h = Poly(coeffs, Q.gen, domain=dom)
        partial[i] = (h, f, k)
        q = q.set_domain(dom)
        F += h*q

    system, result = [], S.Zero

    for (k,), coeff in F.terms():
        system.append(coeff - P.nth(k))

    from sympy.solvers import solve
    solution = solve(system, symbols)

    for h, f, k in partial:
        h = h.as_expr().subs(solution)
        result += h/f.as_expr()**k

    return result


def apart_full_decomposition(P, Q):
    """
    Bronstein's full partial fraction decomposition algorithm.

    Given a univariate rational function ``f``, performing only GCD
    operations over the algebraic closure of the initial ground domain
    of definition, compute full partial fraction decomposition with
    fractions having linear denominators.

    Note that no factorization of the initial denominator of ``f`` is
    performed. The final decomposition is formed in terms of a sum of
    :class:`RootSum` instances.

    References
    ==========

    .. [1] [Bronstein93]_

    """
    return assemble_partfrac_list(apart_list(P/Q, P.gens[0]))


@public
def apart_list(f, x=None, dummies=None, **options):
    """
    Compute partial fraction decomposition of a rational function
    and return the result in structured form.

    Given a rational function ``f`` compute the partial fraction decomposition
    of ``f``. Only Bronstein's full partial fraction decomposition algorithm
    is supported by this method. The return value is highly structured and
    perfectly suited for further algorithmic treatment rather than being
    human-readable. The function returns a tuple holding three elements:

    * The first item is the common coefficient, free of the variable `x` used
      for decomposition. (It is an element of the base field `K`.)

    * The second item is the polynomial part of the decomposition. This can be
      the zero polynomial. (It is an element of `K[x]`.)

    * The third part itself is a list of quadruples. Each quadruple
      has the following elements in this order:

      - The (not necessarily irreducible) polynomial `D` whose roots `w_i` appear
        in the linear denominator of a bunch of related fraction terms. (This item
        can also be a list of explicit roots. However, at the moment ``apart_list``
        never returns a result this way, but the related ``assemble_partfrac_list``
        function accepts this format as input.)

      - The numerator of the fraction, written as a function of the root `w`

      - The linear denominator of the fraction *excluding its power exponent*,
        written as a function of the root `w`.

      - The power to which the denominator has to be raised.

    On can always rebuild a plain expression by using the function ``assemble_partfrac_list``.

    Examples
    ========

    A first example:

    >>> from sympy.polys.partfrac import apart_list, assemble_partfrac_list
    >>> from sympy.abc import x, t

    >>> f = (2*x**3 - 2*x) / (x**2 - 2*x + 1)
    >>> pfd = apart_list(f)
    >>> pfd
    (1,
    Poly(2*x + 4, x, domain='ZZ'),
    [(Poly(_w - 1, _w, domain='ZZ'), Lambda(_a, 4), Lambda(_a, -_a + x), 1)])

    >>> assemble_partfrac_list(pfd)
    2*x + 4 + 4/(x - 1)

    Second example:

    >>> f = (-2*x - 2*x**2) / (3*x**2 - 6*x)
    >>> pfd = apart_list(f)
    >>> pfd
    (-1,
    Poly(2/3, x, domain='QQ'),
    [(Poly(_w - 2, _w, domain='ZZ'), Lambda(_a, 2), Lambda(_a, -_a + x), 1)])

    >>> assemble_partfrac_list(pfd)
    -2/3 - 2/(x - 2)

    Another example, showing symbolic parameters:

    >>> pfd = apart_list(t/(x**2 + x + t), x)
    >>> pfd
    (1,
    Poly(0, x, domain='ZZ[t]'),
    [(Poly(_w**2 + _w + t, _w, domain='ZZ[t]'),
    Lambda(_a, -2*_a*t/(4*t - 1) - t/(4*t - 1)),
    Lambda(_a, -_a + x),
    1)])

    >>> assemble_partfrac_list(pfd)
    RootSum(_w**2 + _w + t, Lambda(_a, (-2*_a*t/(4*t - 1) - t/(4*t - 1))/(-_a + x)))

    This example is taken from Bronstein's original paper:

    >>> f = 36 / (x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2)
    >>> pfd = apart_list(f)
    >>> pfd
    (1,
    Poly(0, x, domain='ZZ'),
    [(Poly(_w - 2, _w, domain='ZZ'), Lambda(_a, 4), Lambda(_a, -_a + x), 1),
    (Poly(_w**2 - 1, _w, domain='ZZ'), Lambda(_a, -3*_a - 6), Lambda(_a, -_a + x), 2),
    (Poly(_w + 1, _w, domain='ZZ'), Lambda(_a, -4), Lambda(_a, -_a + x), 1)])

    >>> assemble_partfrac_list(pfd)
    -4/(x + 1) - 3/(x + 1)**2 - 9/(x - 1)**2 + 4/(x - 2)

    See also
    ========

    apart, assemble_partfrac_list

    References
    ==========

    .. [1] [Bronstein93]_

    """
    allowed_flags(options, [])

    f = sympify(f)

    if f.is_Atom:
        return f
    else:
        P, Q = f.as_numer_denom()

    options = set_defaults(options, extension=True)
    (P, Q), opt = parallel_poly_from_expr((P, Q), x, **options)

    if P.is_multivariate:
        raise NotImplementedError(
            "multivariate partial fraction decomposition")

    common, P, Q = P.cancel(Q)

    poly, P = P.div(Q, auto=True)
    P, Q = P.rat_clear_denoms(Q)

    polypart = poly

    if dummies is None:
        def dummies(name):
            d = Dummy(name)
            while True:
                yield d

        dummies = dummies("w")

    rationalpart = apart_list_full_decomposition(P, Q, dummies)

    return (common, polypart, rationalpart)


def apart_list_full_decomposition(P, Q, dummygen):
    """
    Bronstein's full partial fraction decomposition algorithm.

    Given a univariate rational function ``f``, performing only GCD
    operations over the algebraic closure of the initial ground domain
    of definition, compute full partial fraction decomposition with
    fractions having linear denominators.

    Note that no factorization of the initial denominator of ``f`` is
    performed. The final decomposition is formed in terms of a sum of
    :class:`RootSum` instances.

    References
    ==========

    .. [1] [Bronstein93]_

    """
    P_orig, Q_orig, x, U = P, Q, P.gen, []

    u = Function('u')(x)
    a = Dummy('a')

    partial = []

    for d, n in Q.sqf_list_include(all=True):
        b = d.as_expr()
        U += [ u.diff(x, n - 1) ]

        h = cancel(P_orig/Q_orig.quo(d**n)) / u**n

        H, subs = [h], []

        for j in range(1, n):
            H += [ H[-1].diff(x) / j ]

        for j in range(1, n + 1):
            subs += [ (U[j - 1], b.diff(x, j) / j) ]

        for j in range(0, n):
            P, Q = cancel(H[j]).as_numer_denom()

            for i in range(0, j + 1):
                P = P.subs(*subs[j - i])

            Q = Q.subs(*subs[0])

            P = Poly(P, x)
            Q = Poly(Q, x)

            G = P.gcd(d)
            D = d.quo(G)

            B, g = Q.half_gcdex(D)
            b = (P * B.quo(g)).rem(D)

            Dw = D.subs(x, next(dummygen))
            numer = Lambda(a, b.as_expr().subs(x, a))
            denom = Lambda(a, (x - a))
            exponent = n-j

            partial.append((Dw, numer, denom, exponent))

    return partial


@public
def assemble_partfrac_list(partial_list):
    r"""Reassemble a full partial fraction decomposition
    from a structured result obtained by the function ``apart_list``.

    Examples
    ========

    This example is taken from Bronstein's original paper:

    >>> from sympy.polys.partfrac import apart_list, assemble_partfrac_list
    >>> from sympy.abc import x

    >>> f = 36 / (x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2)
    >>> pfd = apart_list(f)
    >>> pfd
    (1,
    Poly(0, x, domain='ZZ'),
    [(Poly(_w - 2, _w, domain='ZZ'), Lambda(_a, 4), Lambda(_a, -_a + x), 1),
    (Poly(_w**2 - 1, _w, domain='ZZ'), Lambda(_a, -3*_a - 6), Lambda(_a, -_a + x), 2),
    (Poly(_w + 1, _w, domain='ZZ'), Lambda(_a, -4), Lambda(_a, -_a + x), 1)])

    >>> assemble_partfrac_list(pfd)
    -4/(x + 1) - 3/(x + 1)**2 - 9/(x - 1)**2 + 4/(x - 2)

    If we happen to know some roots we can provide them easily inside the structure:

    >>> pfd = apart_list(2/(x**2-2))
    >>> pfd
    (1,
    Poly(0, x, domain='ZZ'),
    [(Poly(_w**2 - 2, _w, domain='ZZ'),
    Lambda(_a, _a/2),
    Lambda(_a, -_a + x),
    1)])

    >>> pfda = assemble_partfrac_list(pfd)
    >>> pfda
    RootSum(_w**2 - 2, Lambda(_a, _a/(-_a + x)))/2

    >>> pfda.doit()
    -sqrt(2)/(2*(x + sqrt(2))) + sqrt(2)/(2*(x - sqrt(2)))

    >>> from sympy import Dummy, Poly, Lambda, sqrt
    >>> a = Dummy("a")
    >>> pfd = (1, Poly(0, x, domain='ZZ'), [([sqrt(2),-sqrt(2)], Lambda(a, a/2), Lambda(a, -a + x), 1)])

    >>> assemble_partfrac_list(pfd)
    -sqrt(2)/(2*(x + sqrt(2))) + sqrt(2)/(2*(x - sqrt(2)))

    See Also
    ========

    apart, apart_list
    """
    # Common factor
    common = partial_list[0]

    # Polynomial part
    polypart = partial_list[1]
    pfd = polypart.as_expr()

    # Rational parts
    for r, nf, df, ex in partial_list[2]:
        if isinstance(r, Poly):
            # Assemble in case the roots are given implicitly by a polynomials
            an, nu = nf.variables, nf.expr
            ad, de = df.variables, df.expr
            # Hack to make dummies equal because Lambda created new Dummies
            de = de.subs(ad[0], an[0])
            func = Lambda(tuple(an), nu/de**ex)
            pfd += RootSum(r, func, auto=False, quadratic=False)
        else:
            # Assemble in case the roots are given explicitly by a list of algebraic numbers
            for root in r:
                pfd += nf(root)/df(root)**ex

    return common*pfd
<|endoftext|>"""Heuristic polynomial GCD algorithm (HEUGCD). """

from .polyerrors import HeuristicGCDFailed

HEU_GCD_MAX = 6

def heugcd(f, g):
    """
    Heuristic polynomial GCD in ``Z[X]``.

    Given univariate polynomials ``f`` and ``g`` in ``Z[X]``, returns
    their GCD and cofactors, i.e. polynomials ``h``, ``cff`` and ``cfg``
    such that::

          h = gcd(f, g), cff = quo(f, h) and cfg = quo(g, h)

    The algorithm is purely heuristic which means it may fail to compute
    the GCD. This will be signaled by raising an exception. In this case
    you will need to switch to another GCD method.

    The algorithm computes the polynomial GCD by evaluating polynomials
    ``f`` and ``g`` at certain points and computing (fast) integer GCD
    of those evaluations. The polynomial GCD is recovered from the integer
    image by interpolation. The evaluation process reduces f and g variable
    by variable into a large integer. The final step is to verify if the
    interpolated polynomial is the correct GCD. This gives cofactors of
    the input polynomials as a side effect.

    Examples
    ========

    >>> from sympy.polys.heuristicgcd import heugcd
    >>> from sympy.polys import ring, ZZ

    >>> R, x,y, = ring("x,y", ZZ)

    >>> f = x**2 + 2*x*y + y**2
    >>> g = x**2 + x*y

    >>> h, cff, cfg = heugcd(f, g)
    >>> h, cff, cfg
    (x + y, x + y, x)

    >>> cff*h == f
    True
    >>> cfg*h == g
    True

    References
    ==========

    .. [1] [Liao95]_

    """
    assert f.ring == g.ring and f.ring.domain.is_ZZ

    ring = f.ring
    x0 = ring.gens[0]
    domain = ring.domain

    gcd, f, g = f.extract_ground(g)

    f_norm = f.max_norm()
    g_norm = g.max_norm()

    B = domain(2*min(f_norm, g_norm) + 29)

    x = max(min(B, 99*domain.sqrt(B)),
            2*min(f_norm // abs(f.LC),
                  g_norm // abs(g.LC)) + 4)

    for i in range(0, HEU_GCD_MAX):
        ff = f.evaluate(x0, x)
        gg = g.evaluate(x0, x)

        if ff and gg:
            if ring.ngens == 1:
                h, cff, cfg = domain.cofactors(ff, gg)
            else:
                h, cff, cfg = heugcd(ff, gg)

            h = _gcd_interpolate(h, x, ring)
            h = h.primitive()[1]

            cff_, r = f.div(h)

            if not r:
                cfg_, r = g.div(h)

                if not r:
                    h = h.mul_ground(gcd)
                    return h, cff_, cfg_

            cff = _gcd_interpolate(cff, x, ring)

            h, r = f.div(cff)

            if not r:
                cfg_, r = g.div(h)

                if not r:
                    h = h.mul_ground(gcd)
                    return h, cff, cfg_

            cfg = _gcd_interpolate(cfg, x, ring)

            h, r = g.div(cfg)

            if not r:
                cff_, r = f.div(h)

                if not r:
                    h = h.mul_ground(gcd)
                    return h, cff_, cfg

        x = 73794*x * domain.sqrt(domain.sqrt(x)) // 27011

    raise HeuristicGCDFailed('no luck')

def _gcd_interpolate(h, x, ring):
    """Interpolate polynomial GCD from integer GCD. """
    f, i = ring.zero, 0

    # TODO: don't expose poly repr implementation details
    if ring.ngens == 1:
        while h:
            g = h % x
            if g > x // 2: g -= x
            h = (h - g) // x

            # f += X**i*g
            if g:
                f[(i,)] = g
            i += 1
    else:
        while h:
            g = h.trunc_ground(x)
            h = (h - g).quo_ground(x)

            # f += X**i*g
            if g:
                for monom, coeff in g.iterterms():
                    f[(i,) + monom] = coeff
            i += 1

    if f.LC < 0:
        return -f
    else:
        return  f
<|endoftext|>from sympy.core.expr import Expr
from sympy.core.symbol import Dummy
from sympy.core.sympify import _sympify

from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polytools import Poly, parallel_poly_from_expr
from sympy.polys.domains import QQ

from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.domainscalar import DomainScalar


class MutablePolyDenseMatrix:
    """
    A mutable matrix of objects from poly module or to operate with them.

    Examples
    ========

    >>> from sympy.polys.polymatrix import PolyMatrix
    >>> from sympy import Symbol, Poly
    >>> x = Symbol('x')
    >>> pm1 = PolyMatrix([[Poly(x**2, x), Poly(-x, x)], [Poly(x**3, x), Poly(-1 + x, x)]])
    >>> v1 = PolyMatrix([[1, 0], [-1, 0]], x)
    >>> pm1*v1
    PolyMatrix([
    [    x**2 + x, 0],
    [x**3 - x + 1, 0]], ring=QQ[x])

