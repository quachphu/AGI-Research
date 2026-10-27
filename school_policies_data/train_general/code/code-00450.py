    if dmp_ground_TC(f, u, K) or dmp_zero_p(f, u):
        return (0,)*(u + 1), f

    F = dmp_to_dict(f, u)
    G = monomial_min(*list(F.keys()))

    if all(g == 0 for g in G):
        return G, f

    f = {}

    for monom, coeff in F.items():
        f[monomial_div(monom, G)] = coeff

    return G, dmp_from_dict(f, u, K)


def _rec_list_terms(g, v, monom):
    """Recursive helper for :func:`dmp_list_terms`."""
    d, terms = dmp_degree(g, v), []

    if not v:
        for i, c in enumerate(g):
            if not c:
                continue

            terms.append((monom + (d - i,), c))
    else:
        w = v - 1

        for i, c in enumerate(g):
            terms.extend(_rec_list_terms(c, w, monom + (d - i,)))

    return terms


def dmp_list_terms(f, u, K, order=None):
    """
    List all non-zero terms from ``f`` in the given order ``order``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_list_terms

    >>> f = ZZ.map([[1, 1], [2, 3]])

    >>> dmp_list_terms(f, 1, ZZ)
    [((1, 1), 1), ((1, 0), 1), ((0, 1), 2), ((0, 0), 3)]
    >>> dmp_list_terms(f, 1, ZZ, order='grevlex')
    [((1, 1), 1), ((1, 0), 1), ((0, 1), 2), ((0, 0), 3)]

    """
    def sort(terms, O):
        return sorted(terms, key=lambda term: O(term[0]), reverse=True)

    terms = _rec_list_terms(f, u, ())

    if not terms:
        return [((0,)*(u + 1), K.zero)]

    if order is None:
        return terms
    else:
        return sort(terms, monomial_key(order))


def dup_apply_pairs(f, g, h, args, K):
    """
    Apply ``h`` to pairs of coefficients of ``f`` and ``g``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_apply_pairs

    >>> h = lambda x, y, z: 2*x + y - z

    >>> dup_apply_pairs([1, 2, 3], [3, 2, 1], h, (1,), ZZ)
    [4, 5, 6]

    """
    n, m = len(f), len(g)

    if n != m:
        if n > m:
            g = [K.zero]*(n - m) + g
        else:
            f = [K.zero]*(m - n) + f

    result = []

    for a, b in zip(f, g):
        result.append(h(a, b, *args))

    return dup_strip(result)


def dmp_apply_pairs(f, g, h, args, u, K):
    """
    Apply ``h`` to pairs of coefficients of ``f`` and ``g``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_apply_pairs

    >>> h = lambda x, y, z: 2*x + y - z

    >>> dmp_apply_pairs([[1], [2, 3]], [[3], [2, 1]], h, (1,), 1, ZZ)
    [[4], [5, 6]]

    """
    if not u:
        return dup_apply_pairs(f, g, h, args, K)

    n, m, v = len(f), len(g), u - 1

    if n != m:
        if n > m:
            g = dmp_zeros(n - m, v, K) + g
        else:
            f = dmp_zeros(m - n, v, K) + f

    result = []

    for a, b in zip(f, g):
        result.append(dmp_apply_pairs(a, b, h, args, v, K))

    return dmp_strip(result, u)


def dup_slice(f, m, n, K):
    """Take a continuous subsequence of terms of ``f`` in ``K[x]``. """
    k = len(f)

    if k >= m:
        M = k - m
    else:
        M = 0
    if k >= n:
        N = k - n
    else:
        N = 0

    f = f[N:M]

    while f and f[0] == K.zero:
        f.pop(0)

    if not f:
        return []
    else:
        return f + [K.zero]*m


def dmp_slice(f, m, n, u, K):
    """Take a continuous subsequence of terms of ``f`` in ``K[X]``. """
    return dmp_slice_in(f, m, n, 0, u, K)


def dmp_slice_in(f, m, n, j, u, K):
    """Take a continuous subsequence of terms of ``f`` in ``x_j`` in ``K[X]``. """
    if j < 0 or j > u:
        raise IndexError("-%s <= j < %s expected, got %s" % (u, u, j))

    if not u:
        return dup_slice(f, m, n, K)

    f, g = dmp_to_dict(f, u), {}

    for monom, coeff in f.items():
        k = monom[j]

        if k < m or k >= n:
            monom = monom[:j] + (0,) + monom[j + 1:]

        if monom in g:
            g[monom] += coeff
        else:
            g[monom] = coeff

    return dmp_from_dict(g, u, K)


def dup_random(n, a, b, K):
    """
    Return a polynomial of degree ``n`` with coefficients in ``[a, b]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_random

    >>> dup_random(3, -10, 10, ZZ) #doctest: +SKIP
    [-2, -8, 9, -4]

    """
    f = [ K.convert(random.randint(a, b)) for _ in range(0, n + 1) ]

    while not f[0]:
        f[0] = K.convert(random.randint(a, b))

    return f
<|endoftext|>"""Dense univariate polynomials with coefficients in Galois fields. """

from math import ceil as _ceil, sqrt as _sqrt, prod

from sympy.core.random import uniform, _randint
from sympy.external.gmpy import SYMPY_INTS, MPZ, invert
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import _sort_factors


def gf_crt(U, M, K=None):
    """
    Chinese Remainder Theorem.

    Given a set of integer residues ``u_0,...,u_n`` and a set of
    co-prime integer moduli ``m_0,...,m_n``, returns an integer
    ``u``, such that ``u = u_i mod m_i`` for ``i = ``0,...,n``.

    Examples
    ========

    Consider a set of residues ``U = [49, 76, 65]``
    and a set of moduli ``M = [99, 97, 95]``. Then we have::

       >>> from sympy.polys.domains import ZZ
       >>> from sympy.polys.galoistools import gf_crt

       >>> gf_crt([49, 76, 65], [99, 97, 95], ZZ)
       639985

    This is the correct result because::

       >>> [639985 % m for m in [99, 97, 95]]
       [49, 76, 65]

    Note: this is a low-level routine with no error checking.

    See Also
    ========

    sympy.ntheory.modular.crt : a higher level crt routine
    sympy.ntheory.modular.solve_congruence

    """
    p = prod(M, start=K.one)
    v = K.zero

    for u, m in zip(U, M):
        e = p // m
        s, _, _ = K.gcdex(e, m)
        v += e*(u*s % m)

    return v % p


def gf_crt1(M, K):
    """
    First part of the Chinese Remainder Theorem.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_crt, gf_crt1, gf_crt2
    >>> U = [49, 76, 65]
    >>> M = [99, 97, 95]

    The following two codes have the same result.

    >>> gf_crt(U, M, ZZ)
    639985

    >>> p, E, S = gf_crt1(M, ZZ)
    >>> gf_crt2(U, M, p, E, S, ZZ)
    639985

    However, it is faster when we want to fix ``M`` and
    compute for multiple U, i.e. the following cases:

    >>> p, E, S = gf_crt1(M, ZZ)
    >>> Us = [[49, 76, 65], [23, 42, 67]]
    >>> for U in Us:
    ...     print(gf_crt2(U, M, p, E, S, ZZ))
    639985
    236237

    See Also
    ========

    sympy.ntheory.modular.crt1 : a higher level crt routine
    sympy.polys.galoistools.gf_crt
    sympy.polys.galoistools.gf_crt2

    """
    E, S = [], []
    p = prod(M, start=K.one)

    for m in M:
        E.append(p // m)
        S.append(K.gcdex(E[-1], m)[0] % m)

    return p, E, S


def gf_crt2(U, M, p, E, S, K):
    """
    Second part of the Chinese Remainder Theorem.

    See ``gf_crt1`` for usage.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_crt2

    >>> U = [49, 76, 65]
    >>> M = [99, 97, 95]
    >>> p = 912285
    >>> E = [9215, 9405, 9603]
    >>> S = [62, 24, 12]

    >>> gf_crt2(U, M, p, E, S, ZZ)
    639985

    See Also
    ========

    sympy.ntheory.modular.crt2 : a higher level crt routine
    sympy.polys.galoistools.gf_crt
    sympy.polys.galoistools.gf_crt1

    """
    v = K.zero

    for u, m, e, s in zip(U, M, E, S):
        v += e*(u*s % m)

    return v % p


def gf_int(a, p):
    """
    Coerce ``a mod p`` to an integer in the range ``[-p/2, p/2]``.

    Examples
    ========

    >>> from sympy.polys.galoistools import gf_int

    >>> gf_int(2, 7)
    2
    >>> gf_int(5, 7)
    -2

    """
    if a <= p // 2:
        return a
    else:
        return a - p


def gf_degree(f):
    """
    Return the leading degree of ``f``.

    Examples
    ========

    >>> from sympy.polys.galoistools import gf_degree

    >>> gf_degree([1, 1, 2, 0])
    3
    >>> gf_degree([])
    -1

    """
    return len(f) - 1


def gf_LC(f, K):
    """
    Return the leading coefficient of ``f``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_LC

    >>> gf_LC([3, 0, 1], ZZ)
    3

    """
    if not f:
        return K.zero
    else:
        return f[0]


def gf_TC(f, K):
    """
    Return the trailing coefficient of ``f``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_TC

    >>> gf_TC([3, 0, 1], ZZ)
    1

    """
    if not f:
        return K.zero
    else:
        return f[-1]


def gf_strip(f):
    """
    Remove leading zeros from ``f``.


    Examples
    ========

    >>> from sympy.polys.galoistools import gf_strip

    >>> gf_strip([0, 0, 0, 3, 0, 1])
    [3, 0, 1]

    """
    if not f or f[0]:
        return f

    k = 0

    for coeff in f:
        if coeff:
            break
        else:
            k += 1

    return f[k:]


def gf_trunc(f, p):
    """
    Reduce all coefficients modulo ``p``.

    Examples
    ========

    >>> from sympy.polys.galoistools import gf_trunc

    >>> gf_trunc([7, -2, 3], 5)
    [2, 3, 3]

    """
    return gf_strip([ a % p for a in f ])


def gf_normal(f, p, K):
    """
    Normalize all coefficients in ``K``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_normal

    >>> gf_normal([5, 10, 21, -3], 5, ZZ)
    [1, 2]

    """
    return gf_trunc(list(map(K, f)), p)


def gf_from_dict(f, p, K):
    """
    Create a ``GF(p)[x]`` polynomial from a dict.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_from_dict

    >>> gf_from_dict({10: ZZ(4), 4: ZZ(33), 0: ZZ(-1)}, 5, ZZ)
    [4, 0, 0, 0, 0, 0, 3, 0, 0, 0, 4]

    """
    n, h = max(f.keys()), []

    if isinstance(n, SYMPY_INTS):
        for k in range(n, -1, -1):
            h.append(f.get(k, K.zero) % p)
    else:
        (n,) = n

        for k in range(n, -1, -1):
            h.append(f.get((k,), K.zero) % p)

    return gf_trunc(h, p)


def gf_to_dict(f, p, symmetric=True):
    """
    Convert a ``GF(p)[x]`` polynomial to a dict.

    Examples
    ========

    >>> from sympy.polys.galoistools import gf_to_dict

    >>> gf_to_dict([4, 0, 0, 0, 0, 0, 3, 0, 0, 0, 4], 5)
    {0: -1, 4: -2, 10: -1}
    >>> gf_to_dict([4, 0, 0, 0, 0, 0, 3, 0, 0, 0, 4], 5, symmetric=False)
    {0: 4, 4: 3, 10: 4}

    """
    n, result = gf_degree(f), {}

    for k in range(0, n + 1):
        if symmetric:
            a = gf_int(f[n - k], p)
        else:
            a = f[n - k]

        if a:
            result[k] = a

    return result


def gf_from_int_poly(f, p):
    """
    Create a ``GF(p)[x]`` polynomial from ``Z[x]``.

    Examples
    ========

    >>> from sympy.polys.galoistools import gf_from_int_poly

    >>> gf_from_int_poly([7, -2, 3], 5)
    [2, 3, 3]

    """
    return gf_trunc(f, p)


def gf_to_int_poly(f, p, symmetric=True):
    """
    Convert a ``GF(p)[x]`` polynomial to ``Z[x]``.


    Examples
    ========

    >>> from sympy.polys.galoistools import gf_to_int_poly

    >>> gf_to_int_poly([2, 3, 3], 5)
    [2, -2, -2]
    >>> gf_to_int_poly([2, 3, 3], 5, symmetric=False)
    [2, 3, 3]

    """
    if symmetric:
        return [ gf_int(c, p) for c in f ]
    else:
        return f


def gf_neg(f, p, K):
    """
    Negate a polynomial in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_neg

    >>> gf_neg([3, 2, 1, 0], 5, ZZ)
    [2, 3, 4, 0]

    """
    return [ -coeff % p for coeff in f ]


def gf_add_ground(f, a, p, K):
    """
    Compute ``f + a`` where ``f`` in ``GF(p)[x]`` and ``a`` in ``GF(p)``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_add_ground

    >>> gf_add_ground([3, 2, 4], 2, 5, ZZ)
    [3, 2, 1]

    """
    if not f:
        a = a % p
    else:
        a = (f[-1] + a) % p

        if len(f) > 1:
            return f[:-1] + [a]

    if not a:
        return []
    else:
        return [a]


def gf_sub_ground(f, a, p, K):
    """
    Compute ``f - a`` where ``f`` in ``GF(p)[x]`` and ``a`` in ``GF(p)``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_sub_ground

    >>> gf_sub_ground([3, 2, 4], 2, 5, ZZ)
    [3, 2, 2]

    """
    if not f:
        a = -a % p
    else:
        a = (f[-1] - a) % p

        if len(f) > 1:
            return f[:-1] + [a]

    if not a:
        return []
    else:
        return [a]


def gf_mul_ground(f, a, p, K):
    """
    Compute ``f * a`` where ``f`` in ``GF(p)[x]`` and ``a`` in ``GF(p)``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_mul_ground

    >>> gf_mul_ground([3, 2, 4], 2, 5, ZZ)
    [1, 4, 3]

    """
    if not a:
        return []
    else:
        return [ (a*b) % p for b in f ]


def gf_quo_ground(f, a, p, K):
    """
    Compute ``f/a`` where ``f`` in ``GF(p)[x]`` and ``a`` in ``GF(p)``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_quo_ground

    >>> gf_quo_ground(ZZ.map([3, 2, 4]), ZZ(2), 5, ZZ)
    [4, 1, 2]

    """
    return gf_mul_ground(f, K.invert(a, p), p, K)


def gf_add(f, g, p, K):
    """
    Add polynomials in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_add

    >>> gf_add([3, 2, 4], [2, 2, 2], 5, ZZ)
    [4, 1]

    """
    if not f:
        return g
    if not g:
        return f

    df = gf_degree(f)
    dg = gf_degree(g)

    if df == dg:
        return gf_strip([ (a + b) % p for a, b in zip(f, g) ])
    else:
        k = abs(df - dg)

        if df > dg:
            h, f = f[:k], f[k:]
        else:
            h, g = g[:k], g[k:]

        return h + [ (a + b) % p for a, b in zip(f, g) ]


def gf_sub(f, g, p, K):
    """
    Subtract polynomials in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_sub

    >>> gf_sub([3, 2, 4], [2, 2, 2], 5, ZZ)
    [1, 0, 2]

    """
    if not g:
        return f
    if not f:
        return gf_neg(g, p, K)

    df = gf_degree(f)
    dg = gf_degree(g)

    if df == dg:
        return gf_strip([ (a - b) % p for a, b in zip(f, g) ])
    else:
        k = abs(df - dg)

        if df > dg:
            h, f = f[:k], f[k:]
        else:
            h, g = gf_neg(g[:k], p, K), g[k:]

        return h + [ (a - b) % p for a, b in zip(f, g) ]


def gf_mul(f, g, p, K):
    """
    Multiply polynomials in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_mul

    >>> gf_mul([3, 2, 4], [2, 2, 2], 5, ZZ)
    [1, 0, 3, 2, 3]

    """
    df = gf_degree(f)
    dg = gf_degree(g)

    dh = df + dg
    h = [0]*(dh + 1)

    for i in range(0, dh + 1):
        coeff = K.zero

        for j in range(max(0, i - dg), min(i, df) + 1):
            coeff += f[j]*g[i - j]

        h[i] = coeff % p

    return gf_strip(h)


def gf_sqr(f, p, K):
    """
    Square polynomials in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_sqr

    >>> gf_sqr([3, 2, 4], 5, ZZ)
    [4, 2, 3, 1, 1]

    """
    df = gf_degree(f)

    dh = 2*df
    h = [0]*(dh + 1)

    for i in range(0, dh + 1):
        coeff = K.zero

        jmin = max(0, i - df)
        jmax = min(i, df)

        n = jmax - jmin + 1

        jmax = jmin + n // 2 - 1

        for j in range(jmin, jmax + 1):
            coeff += f[j]*f[i - j]

        coeff += coeff

        if n & 1:
            elem = f[jmax + 1]
            coeff += elem**2

        h[i] = coeff % p

    return gf_strip(h)


def gf_add_mul(f, g, h, p, K):
    """
    Returns ``f + g*h`` where ``f``, ``g``, ``h`` in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_add_mul
    >>> gf_add_mul([3, 2, 4], [2, 2, 2], [1, 4], 5, ZZ)
    [2, 3, 2, 2]
    """
    return gf_add(f, gf_mul(g, h, p, K), p, K)


def gf_sub_mul(f, g, h, p, K):
    """
    Compute ``f - g*h`` where ``f``, ``g``, ``h`` in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_sub_mul

    >>> gf_sub_mul([3, 2, 4], [2, 2, 2], [1, 4], 5, ZZ)
    [3, 3, 2, 1]

    """
    return gf_sub(f, gf_mul(g, h, p, K), p, K)


def gf_expand(F, p, K):
    """
    Expand results of :func:`~.factor` in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_expand

    >>> gf_expand([([3, 2, 4], 1), ([2, 2], 2), ([3, 1], 3)], 5, ZZ)
    [4, 3, 0, 3, 0, 1, 4, 1]

    """
    if isinstance(F, tuple):
        lc, F = F
    else:
        lc = K.one

    g = [lc]

    for f, k in F:
        f = gf_pow(f, k, p, K)
        g = gf_mul(g, f, p, K)

    return g


def gf_div(f, g, p, K):
    """
    Division with remainder in ``GF(p)[x]``.

    Given univariate polynomials ``f`` and ``g`` with coefficients in a
    finite field with ``p`` elements, returns polynomials ``q`` and ``r``
    (quotient and remainder) such that ``f = q*g + r``.

    Consider polynomials ``x**3 + x + 1`` and ``x**2 + x`` in GF(2)::

       >>> from sympy.polys.domains import ZZ
       >>> from sympy.polys.galoistools import gf_div, gf_add_mul

       >>> gf_div(ZZ.map([1, 0, 1, 1]), ZZ.map([1, 1, 0]), 2, ZZ)
       ([1, 1], [1])

    As result we obtained quotient ``x + 1`` and remainder ``1``, thus::

       >>> gf_add_mul(ZZ.map([1]), ZZ.map([1, 1]), ZZ.map([1, 1, 0]), 2, ZZ)
       [1, 0, 1, 1]

    References
    ==========

    .. [1] [Monagan93]_
    .. [2] [Gathen99]_

    """
    df = gf_degree(f)
    dg = gf_degree(g)

    if not g:
        raise ZeroDivisionError("polynomial division")
    elif df < dg:
        return [], f

    inv = K.invert(g[0], p)

    h, dq, dr = list(f), df - dg, dg - 1

    for i in range(0, df + 1):
        coeff = h[i]

        for j in range(max(0, dg - i), min(df - i, dr) + 1):
            coeff -= h[i + j - dg] * g[dg - j]

        if i <= dq:
            coeff *= inv

        h[i] = coeff % p

    return h[:dq + 1], gf_strip(h[dq + 1:])


def gf_rem(f, g, p, K):
    """
    Compute polynomial remainder in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_rem

    >>> gf_rem(ZZ.map([1, 0, 1, 1]), ZZ.map([1, 1, 0]), 2, ZZ)
    [1]

    """
    return gf_div(f, g, p, K)[1]


def gf_quo(f, g, p, K):
    """
    Compute exact quotient in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_quo

    >>> gf_quo(ZZ.map([1, 0, 1, 1]), ZZ.map([1, 1, 0]), 2, ZZ)
    [1, 1]
    >>> gf_quo(ZZ.map([1, 0, 3, 2, 3]), ZZ.map([2, 2, 2]), 5, ZZ)
    [3, 2, 4]

    """
    df = gf_degree(f)
    dg = gf_degree(g)

    if not g:
        raise ZeroDivisionError("polynomial division")
    elif df < dg:
        return []

    inv = K.invert(g[0], p)

    h, dq, dr = f[:], df - dg, dg - 1

    for i in range(0, dq + 1):
        coeff = h[i]

        for j in range(max(0, dg - i), min(df - i, dr) + 1):
            coeff -= h[i + j - dg] * g[dg - j]

        h[i] = (coeff * inv) % p

    return h[:dq + 1]


def gf_exquo(f, g, p, K):
    """
    Compute polynomial quotient in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_exquo

    >>> gf_exquo(ZZ.map([1, 0, 3, 2, 3]), ZZ.map([2, 2, 2]), 5, ZZ)
    [3, 2, 4]

    >>> gf_exquo(ZZ.map([1, 0, 1, 1]), ZZ.map([1, 1, 0]), 2, ZZ)
    Traceback (most recent call last):
    ...
    ExactQuotientFailed: [1, 1, 0] does not divide [1, 0, 1, 1]

    """
    q, r = gf_div(f, g, p, K)

    if not r:
        return q
    else:
        raise ExactQuotientFailed(f, g)


def gf_lshift(f, n, K):
    """
    Efficiently multiply ``f`` by ``x**n``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_lshift

    >>> gf_lshift([3, 2, 4], 4, ZZ)
    [3, 2, 4, 0, 0, 0, 0]

    """
    if not f:
        return f
    else:
        return f + [K.zero]*n


def gf_rshift(f, n, K):
    """
    Efficiently divide ``f`` by ``x**n``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_rshift

    >>> gf_rshift([1, 2, 3, 4, 0], 3, ZZ)
    ([1, 2], [3, 4, 0])

    """
    if not n:
        return f, []
    else:
        return f[:-n], f[-n:]


def gf_pow(f, n, p, K):
    """
    Compute ``f**n`` in ``GF(p)[x]`` using repeated squaring.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_pow

    >>> gf_pow([3, 2, 4], 3, 5, ZZ)
    [2, 4, 4, 2, 2, 1, 4]

    """
    if not n:
        return [K.one]
    elif n == 1:
        return f
    elif n == 2:
        return gf_sqr(f, p, K)

    h = [K.one]

    while True:
        if n & 1:
            h = gf_mul(h, f, p, K)
            n -= 1

        n >>= 1

        if not n:
            break

        f = gf_sqr(f, p, K)

    return h

def gf_frobenius_monomial_base(g, p, K):
    """
    return the list of ``x**(i*p) mod g in Z_p`` for ``i = 0, .., n - 1``
    where ``n = gf_degree(g)``

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_frobenius_monomial_base
    >>> g = ZZ.map([1, 0, 2, 1])
    >>> gf_frobenius_monomial_base(g, 5, ZZ)
    [[1], [4, 4, 2], [1, 2]]

    """
    n = gf_degree(g)
    if n == 0:
        return []
    b = [0]*n
    b[0] = [1]
    if p < n:
        for i in range(1, n):
            mon = gf_lshift(b[i - 1], p, K)
            b[i] = gf_rem(mon, g, p, K)
    elif n > 1:
        b[1] = gf_pow_mod([K.one, K.zero], p, g, p, K)
        for i in range(2, n):
            b[i] = gf_mul(b[i - 1], b[1], p, K)
            b[i] = gf_rem(b[i], g, p, K)

    return b

def gf_frobenius_map(f, g, b, p, K):
    """
    compute gf_pow_mod(f, p, g, p, K) using the Frobenius map

    Parameters
    ==========

    f, g : polynomials in ``GF(p)[x]``
    b : frobenius monomial base
    p : prime number
    K : domain

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_frobenius_monomial_base, gf_frobenius_map
    >>> f = ZZ.map([2, 1, 0, 1])
    >>> g = ZZ.map([1, 0, 2, 1])
    >>> p = 5
    >>> b = gf_frobenius_monomial_base(g, p, ZZ)
    >>> r = gf_frobenius_map(f, g, b, p, ZZ)
    >>> gf_frobenius_map(f, g, b, p, ZZ)
    [4, 0, 3]
    """
    m = gf_degree(g)
    if gf_degree(f) >= m:
        f = gf_rem(f, g, p, K)
    if not f:
        return []
    n = gf_degree(f)
    sf = [f[-1]]
    for i in range(1, n + 1):
        v = gf_mul_ground(b[i], f[n - i], p, K)
        sf = gf_add(sf, v, p, K)
    return sf

def _gf_pow_pnm1d2(f, n, g, b, p, K):
    """
    utility function for ``gf_edf_zassenhaus``
    Compute ``f**((p**n - 1) // 2)`` in ``GF(p)[x]/(g)``
    ``f**((p**n - 1) // 2) = (f*f**p*...*f**(p**n - 1))**((p - 1) // 2)``
    """
    f = gf_rem(f, g, p, K)
    h = f
    r = f
    for i in range(1, n):
        h = gf_frobenius_map(h, g, b, p, K)
        r = gf_mul(r, h, p, K)
        r = gf_rem(r, g, p, K)

    res = gf_pow_mod(r, (p - 1)//2, g, p, K)
    return res

def gf_pow_mod(f, n, g, p, K):
    """
    Compute ``f**n`` in ``GF(p)[x]/(g)`` using repeated squaring.

    Given polynomials ``f`` and ``g`` in ``GF(p)[x]`` and a non-negative
    integer ``n``, efficiently computes ``f**n (mod g)`` i.e. the remainder
    of ``f**n`` from division by ``g``, using the repeated squaring algorithm.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_pow_mod

    >>> gf_pow_mod(ZZ.map([3, 2, 4]), 3, ZZ.map([1, 1]), 5, ZZ)
    []

    References
    ==========

    .. [1] [Gathen99]_

    """
    if not n:
        return [K.one]
    elif n == 1:
        return gf_rem(f, g, p, K)
    elif n == 2:
        return gf_rem(gf_sqr(f, p, K), g, p, K)

    h = [K.one]

    while True:
        if n & 1:
            h = gf_mul(h, f, p, K)
            h = gf_rem(h, g, p, K)
            n -= 1

        n >>= 1

        if not n:
            break

        f = gf_sqr(f, p, K)
        f = gf_rem(f, g, p, K)

    return h


def gf_gcd(f, g, p, K):
    """
    Euclidean Algorithm in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_gcd

    >>> gf_gcd(ZZ.map([3, 2, 4]), ZZ.map([2, 2, 3]), 5, ZZ)
    [1, 3]

    """
    while g:
        f, g = g, gf_rem(f, g, p, K)

    return gf_monic(f, p, K)[1]


def gf_lcm(f, g, p, K):
    """
    Compute polynomial LCM in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_lcm

    >>> gf_lcm(ZZ.map([3, 2, 4]), ZZ.map([2, 2, 3]), 5, ZZ)
    [1, 2, 0, 4]

    """
    if not f or not g:
        return []

    h = gf_quo(gf_mul(f, g, p, K),
               gf_gcd(f, g, p, K), p, K)

    return gf_monic(h, p, K)[1]


def gf_cofactors(f, g, p, K):
    """
    Compute polynomial GCD and cofactors in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_cofactors

    >>> gf_cofactors(ZZ.map([3, 2, 4]), ZZ.map([2, 2, 3]), 5, ZZ)
    ([1, 3], [3, 3], [2, 1])

    """
    if not f and not g:
        return ([], [], [])

    h = gf_gcd(f, g, p, K)

    return (h, gf_quo(f, h, p, K),
            gf_quo(g, h, p, K))


def gf_gcdex(f, g, p, K):
    """
    Extended Euclidean Algorithm in ``GF(p)[x]``.

    Given polynomials ``f`` and ``g`` in ``GF(p)[x]``, computes polynomials
    ``s``, ``t`` and ``h``, such that ``h = gcd(f, g)`` and ``s*f + t*g = h``.
    The typical application of EEA is solving polynomial diophantine equations.

    Consider polynomials ``f = (x + 7) (x + 1)``, ``g = (x + 7) (x**2 + 1)``
    in ``GF(11)[x]``. Application of Extended Euclidean Algorithm gives::

       >>> from sympy.polys.domains import ZZ
       >>> from sympy.polys.galoistools import gf_gcdex, gf_mul, gf_add

       >>> s, t, g = gf_gcdex(ZZ.map([1, 8, 7]), ZZ.map([1, 7, 1, 7]), 11, ZZ)
       >>> s, t, g
       ([5, 6], [6], [1, 7])

    As result we obtained polynomials ``s = 5*x + 6`` and ``t = 6``, and
    additionally ``gcd(f, g) = x + 7``. This is correct because::

       >>> S = gf_mul(s, ZZ.map([1, 8, 7]), 11, ZZ)
       >>> T = gf_mul(t, ZZ.map([1, 7, 1, 7]), 11, ZZ)

       >>> gf_add(S, T, 11, ZZ) == [1, 7]
       True

    References
    ==========

    .. [1] [Gathen99]_

    """
    if not (f or g):
        return [K.one], [], []

    p0, r0 = gf_monic(f, p, K)
    p1, r1 = gf_monic(g, p, K)

    if not f:
        return [], [K.invert(p1, p)], r1
    if not g:
        return [K.invert(p0, p)], [], r0

    s0, s1 = [K.invert(p0, p)], []
    t0, t1 = [], [K.invert(p1, p)]

    while True:
        Q, R = gf_div(r0, r1, p, K)

        if not R:
            break

        (lc, r1), r0 = gf_monic(R, p, K), r1

        inv = K.invert(lc, p)

        s = gf_sub_mul(s0, s1, Q, p, K)
        t = gf_sub_mul(t0, t1, Q, p, K)

        s1, s0 = gf_mul_ground(s, inv, p, K), s1
        t1, t0 = gf_mul_ground(t, inv, p, K), t1

    return s1, t1, r1


def gf_monic(f, p, K):
    """
    Compute LC and a monic polynomial in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_monic

    >>> gf_monic(ZZ.map([3, 2, 4]), 5, ZZ)
    (3, [1, 4, 3])

    """
    if not f:
        return K.zero, []
    else:
        lc = f[0]

        if K.is_one(lc):
            return lc, list(f)
        else:
            return lc, gf_quo_ground(f, lc, p, K)


def gf_diff(f, p, K):
    """
    Differentiate polynomial in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_diff

    >>> gf_diff([3, 2, 4], 5, ZZ)
    [1, 2]

    """
    df = gf_degree(f)

    h, n = [K.zero]*df, df

    for coeff in f[:-1]:
        coeff *= K(n)
        coeff %= p

        if coeff:
            h[df - n] = coeff

        n -= 1

    return gf_strip(h)


def gf_eval(f, a, p, K):
    """
    Evaluate ``f(a)`` in ``GF(p)`` using Horner scheme.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_eval

    >>> gf_eval([3, 2, 4], 2, 5, ZZ)
    0

    """
    result = K.zero

    for c in f:
        result *= a
        result += c
        result %= p

    return result


def gf_multi_eval(f, A, p, K):
    """
    Evaluate ``f(a)`` for ``a`` in ``[a_1, ..., a_n]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_multi_eval

    >>> gf_multi_eval([3, 2, 4], [0, 1, 2, 3, 4], 5, ZZ)
    [4, 4, 0, 2, 0]

    """
    return [ gf_eval(f, a, p, K) for a in A ]


def gf_compose(f, g, p, K):
    """
    Compute polynomial composition ``f(g)`` in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_compose

    >>> gf_compose([3, 2, 4], [2, 2, 2], 5, ZZ)
    [2, 4, 0, 3, 0]

    """
    if len(g) <= 1:
        return gf_strip([gf_eval(f, gf_LC(g, K), p, K)])

    if not f:
        return []

    h = [f[0]]

    for c in f[1:]:
        h = gf_mul(h, g, p, K)
        h = gf_add_ground(h, c, p, K)

    return h


def gf_compose_mod(g, h, f, p, K):
    """
    Compute polynomial composition ``g(h)`` in ``GF(p)[x]/(f)``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_compose_mod

    >>> gf_compose_mod(ZZ.map([3, 2, 4]), ZZ.map([2, 2, 2]), ZZ.map([4, 3]), 5, ZZ)
    [4]

    """
    if not g:
        return []

    comp = [g[0]]

    for a in g[1:]:
        comp = gf_mul(comp, h, p, K)
        comp = gf_add_ground(comp, a, p, K)
        comp = gf_rem(comp, f, p, K)

    return comp


def gf_trace_map(a, b, c, n, f, p, K):
    """
    Compute polynomial trace map in ``GF(p)[x]/(f)``.

    Given a polynomial ``f`` in ``GF(p)[x]``, polynomials ``a``, ``b``,
    ``c`` in the quotient ring ``GF(p)[x]/(f)`` such that ``b = c**t
    (mod f)`` for some positive power ``t`` of ``p``, and a positive
    integer ``n``, returns a mapping::

       a -> a**t**n, a + a**t + a**t**2 + ... + a**t**n (mod f)

    In factorization context, ``b = x**p mod f`` and ``c = x mod f``.
    This way we can efficiently compute trace polynomials in equal
    degree factorization routine, much faster than with other methods,
    like iterated Frobenius algorithm, for large degrees.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_trace_map

    >>> gf_trace_map([1, 2], [4, 4], [1, 1], 4, [3, 2, 4], 5, ZZ)
    ([1, 3], [1, 3])

    References
    ==========

    .. [1] [Gathen92]_

    """
    u = gf_compose_mod(a, b, f, p, K)
    v = b

    if n & 1:
        U = gf_add(a, u, p, K)
        V = b
    else:
        U = a
        V = c

    n >>= 1

    while n:
        u = gf_add(u, gf_compose_mod(u, v, f, p, K), p, K)
        v = gf_compose_mod(v, v, f, p, K)

        if n & 1:
            U = gf_add(U, gf_compose_mod(u, V, f, p, K), p, K)
            V = gf_compose_mod(v, V, f, p, K)

        n >>= 1

    return gf_compose_mod(a, V, f, p, K), U

def _gf_trace_map(f, n, g, b, p, K):
    """
    utility for ``gf_edf_shoup``
    """
    f = gf_rem(f, g, p, K)
    h = f
    r = f
    for i in range(1, n):
        h = gf_frobenius_map(h, g, b, p, K)
        r = gf_add(r, h, p, K)
        r = gf_rem(r, g, p, K)
    return r


def gf_random(n, p, K):
    """
    Generate a random polynomial in ``GF(p)[x]`` of degree ``n``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_random
    >>> gf_random(10, 5, ZZ) #doctest: +SKIP
    [1, 2, 3, 2, 1, 1, 1, 2, 0, 4, 2]

    """
    pi = int(p)
    return [K.one] + [ K(int(uniform(0, pi))) for i in range(0, n) ]


def gf_irreducible(n, p, K):
    """
    Generate random irreducible polynomial of degree ``n`` in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_irreducible
    >>> gf_irreducible(10, 5, ZZ) #doctest: +SKIP
    [1, 4, 2, 2, 3, 2, 4, 1, 4, 0, 4]

    """
    while True:
        f = gf_random(n, p, K)
        if gf_irreducible_p(f, p, K):
            return f


def gf_irred_p_ben_or(f, p, K):
    """
    Ben-Or's polynomial irreducibility test over finite fields.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_irred_p_ben_or

    >>> gf_irred_p_ben_or(ZZ.map([1, 4, 2, 2, 3, 2, 4, 1, 4, 0, 4]), 5, ZZ)
    True
    >>> gf_irred_p_ben_or(ZZ.map([3, 2, 4]), 5, ZZ)
    False

    """
    n = gf_degree(f)

    if n <= 1:
        return True

    _, f = gf_monic(f, p, K)
    if n < 5:
        H = h = gf_pow_mod([K.one, K.zero], p, f, p, K)

        for i in range(0, n//2):
            g = gf_sub(h, [K.one, K.zero], p, K)

            if gf_gcd(f, g, p, K) == [K.one]:
                h = gf_compose_mod(h, H, f, p, K)
            else:
                return False
    else:
        b = gf_frobenius_monomial_base(f, p, K)
        H = h = gf_frobenius_map([K.one, K.zero], f, b, p, K)
        for i in range(0, n//2):
            g = gf_sub(h, [K.one, K.zero], p, K)
            if gf_gcd(f, g, p, K) == [K.one]:
                h = gf_frobenius_map(h, f, b, p, K)
            else:
                return False

    return True


def gf_irred_p_rabin(f, p, K):
    """
    Rabin's polynomial irreducibility test over finite fields.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_irred_p_rabin

    >>> gf_irred_p_rabin(ZZ.map([1, 4, 2, 2, 3, 2, 4, 1, 4, 0, 4]), 5, ZZ)
    True
    >>> gf_irred_p_rabin(ZZ.map([3, 2, 4]), 5, ZZ)
    False

    """
    n = gf_degree(f)

    if n <= 1:
        return True

    _, f = gf_monic(f, p, K)

    x = [K.one, K.zero]

    from sympy.ntheory import factorint

    indices = { n//d for d in factorint(n) }

    b = gf_frobenius_monomial_base(f, p, K)
    h = b[1]

    for i in range(1, n):
        if i in indices:
            g = gf_sub(h, x, p, K)

            if gf_gcd(f, g, p, K) != [K.one]:
                return False

        h = gf_frobenius_map(h, f, b, p, K)

    return h == x

_irred_methods = {
    'ben-or': gf_irred_p_ben_or,
    'rabin': gf_irred_p_rabin,
}


def gf_irreducible_p(f, p, K):
    """
    Test irreducibility of a polynomial ``f`` in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_irreducible_p

    >>> gf_irreducible_p(ZZ.map([1, 4, 2, 2, 3, 2, 4, 1, 4, 0, 4]), 5, ZZ)
    True
    >>> gf_irreducible_p(ZZ.map([3, 2, 4]), 5, ZZ)
    False

    """
    method = query('GF_IRRED_METHOD')

    if method is not None:
        irred = _irred_methods[method](f, p, K)
    else:
        irred = gf_irred_p_rabin(f, p, K)

    return irred


def gf_sqf_p(f, p, K):
    """
    Return ``True`` if ``f`` is square-free in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_sqf_p

    >>> gf_sqf_p(ZZ.map([3, 2, 4]), 5, ZZ)
    True
    >>> gf_sqf_p(ZZ.map([2, 4, 4, 2, 2, 1, 4]), 5, ZZ)
    False

    """
    _, f = gf_monic(f, p, K)

    if not f:
        return True
    else:
        return gf_gcd(f, gf_diff(f, p, K), p, K) == [K.one]


def gf_sqf_part(f, p, K):
    """
    Return square-free part of a ``GF(p)[x]`` polynomial.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_sqf_part

    >>> gf_sqf_part(ZZ.map([1, 1, 3, 0, 1, 0, 2, 2, 1]), 5, ZZ)
    [1, 4, 3]

    """
    _, sqf = gf_sqf_list(f, p, K)

    g = [K.one]

    for f, _ in sqf:
        g = gf_mul(g, f, p, K)

    return g


def gf_sqf_list(f, p, K, all=False):
    """
    Return the square-free decomposition of a ``GF(p)[x]`` polynomial.

    Given a polynomial ``f`` in ``GF(p)[x]``, returns the leading coefficient
    of ``f`` and a square-free decomposition ``f_1**e_1 f_2**e_2 ... f_k**e_k``
    such that all ``f_i`` are monic polynomials and ``(f_i, f_j)`` for ``i != j``
    are co-prime and ``e_1 ... e_k`` are given in increasing order. All trivial
    terms (i.e. ``f_i = 1``) are not included in the output.

    Consider polynomial ``f = x**11 + 1`` over ``GF(11)[x]``::

       >>> from sympy.polys.domains import ZZ

       >>> from sympy.polys.galoistools import (
       ...     gf_from_dict, gf_diff, gf_sqf_list, gf_pow,
       ... )
       ... # doctest: +NORMALIZE_WHITESPACE

       >>> f = gf_from_dict({11: ZZ(1), 0: ZZ(1)}, 11, ZZ)

    Note that ``f'(x) = 0``::

       >>> gf_diff(f, 11, ZZ)
       []

    This phenomenon does not happen in characteristic zero. However we can
    still compute square-free decomposition of ``f`` using ``gf_sqf()``::

       >>> gf_sqf_list(f, 11, ZZ)
       (1, [([1, 1], 11)])

    We obtained factorization ``f = (x + 1)**11``. This is correct because::

       >>> gf_pow([1, 1], 11, 11, ZZ) == f
       True

    References
    ==========

    .. [1] [Geddes92]_

    """
    n, sqf, factors, r = 1, False, [], int(p)

    lc, f = gf_monic(f, p, K)

    if gf_degree(f) < 1:
        return lc, []

    while True:
        F = gf_diff(f, p, K)

        if F != []:
            g = gf_gcd(f, F, p, K)
            h = gf_quo(f, g, p, K)

            i = 1

            while h != [K.one]:
                G = gf_gcd(g, h, p, K)
                H = gf_quo(h, G, p, K)

                if gf_degree(H) > 0:
                    factors.append((H, i*n))

                g, h, i = gf_quo(g, G, p, K), G, i + 1

            if g == [K.one]:
                sqf = True
            else:
                f = g

        if not sqf:
            d = gf_degree(f) // r

            for i in range(0, d + 1):
                f[i] = f[i*r]

            f, n = f[:d + 1], n*r
        else:
            break

    if all:
        raise ValueError("'all=True' is not supported yet")

    return lc, factors


def gf_Qmatrix(f, p, K):
    """
    Calculate Berlekamp's ``Q`` matrix.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_Qmatrix

    >>> gf_Qmatrix([3, 2, 4], 5, ZZ)
    [[1, 0],
     [3, 4]]

    >>> gf_Qmatrix([1, 0, 0, 0, 1], 5, ZZ)
    [[1, 0, 0, 0],
     [0, 4, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 4]]

    """
    n, r = gf_degree(f), int(p)

    q = [K.one] + [K.zero]*(n - 1)
    Q = [list(q)] + [[]]*(n - 1)

    for i in range(1, (n - 1)*r + 1):
        qq, c = [(-q[-1]*f[-1]) % p], q[-1]

        for j in range(1, n):
            qq.append((q[j - 1] - c*f[-j - 1]) % p)

        if not (i % r):
            Q[i//r] = list(qq)

        q = qq

    return Q


def gf_Qbasis(Q, p, K):
    """
    Compute a basis of the kernel of ``Q``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_Qmatrix, gf_Qbasis

    >>> gf_Qbasis(gf_Qmatrix([1, 0, 0, 0, 1], 5, ZZ), 5, ZZ)
    [[1, 0, 0, 0], [0, 0, 1, 0]]

    >>> gf_Qbasis(gf_Qmatrix([3, 2, 4], 5, ZZ), 5, ZZ)
    [[1, 0]]

    """
    Q, n = [ list(q) for q in Q ], len(Q)

    for k in range(0, n):
        Q[k][k] = (Q[k][k] - K.one) % p

    for k in range(0, n):
        for i in range(k, n):
            if Q[k][i]:
                break
        else:
            continue

        inv = K.invert(Q[k][i], p)

        for j in range(0, n):
            Q[j][i] = (Q[j][i]*inv) % p

        for j in range(0, n):
            t = Q[j][k]
            Q[j][k] = Q[j][i]
            Q[j][i] = t

        for i in range(0, n):
            if i != k:
                q = Q[k][i]

                for j in range(0, n):
                    Q[j][i] = (Q[j][i] - Q[j][k]*q) % p

    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                Q[i][j] = (K.one - Q[i][j]) % p
            else:
                Q[i][j] = (-Q[i][j]) % p

    basis = []

    for q in Q:
        if any(q):
            basis.append(q)

    return basis


def gf_berlekamp(f, p, K):
    """
    Factor a square-free ``f`` in ``GF(p)[x]`` for small ``p``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_berlekamp

    >>> gf_berlekamp([1, 0, 0, 0, 1], 5, ZZ)
    [[1, 0, 2], [1, 0, 3]]

    """
    Q = gf_Qmatrix(f, p, K)
    V = gf_Qbasis(Q, p, K)

    for i, v in enumerate(V):
        V[i] = gf_strip(list(reversed(v)))

    factors = [f]

    for k in range(1, len(V)):
        for f in list(factors):
            s = K.zero

            while s < p:
                g = gf_sub_ground(V[k], s, p, K)
                h = gf_gcd(f, g, p, K)

                if h != [K.one] and h != f:
                    factors.remove(f)

                    f = gf_quo(f, h, p, K)
                    factors.extend([f, h])

                if len(factors) == len(V):
                    return _sort_factors(factors, multiple=False)

                s += K.one

    return _sort_factors(factors, multiple=False)


def gf_ddf_zassenhaus(f, p, K):
    """
    Cantor-Zassenhaus: Deterministic Distinct Degree Factorization

    Given a monic square-free polynomial ``f`` in ``GF(p)[x]``, computes
    partial distinct degree factorization ``f_1 ... f_d`` of ``f`` where
    ``deg(f_i) != deg(f_j)`` for ``i != j``. The result is returned as a
    list of pairs ``(f_i, e_i)`` where ``deg(f_i) > 0`` and ``e_i > 0``
    is an argument to the equal degree factorization routine.

    Consider the polynomial ``x**15 - 1`` in ``GF(11)[x]``::

       >>> from sympy.polys.domains import ZZ
       >>> from sympy.polys.galoistools import gf_from_dict

       >>> f = gf_from_dict({15: ZZ(1), 0: ZZ(-1)}, 11, ZZ)

    Distinct degree factorization gives::

       >>> from sympy.polys.galoistools import gf_ddf_zassenhaus

       >>> gf_ddf_zassenhaus(f, 11, ZZ)
       [([1, 0, 0, 0, 0, 10], 1), ([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], 2)]

    which means ``x**15 - 1 = (x**5 - 1) (x**10 + x**5 + 1)``. To obtain
    factorization into irreducibles, use equal degree factorization
    procedure (EDF) with each of the factors.

    References
    ==========

    .. [1] [Gathen99]_
    .. [2] [Geddes92]_

    """
    i, g, factors = 1, [K.one, K.zero], []

    b = gf_frobenius_monomial_base(f, p, K)
    while 2*i <= gf_degree(f):
        g = gf_frobenius_map(g, f, b, p, K)
        h = gf_gcd(f, gf_sub(g, [K.one, K.zero], p, K), p, K)

        if h != [K.one]:
            factors.append((h, i))

            f = gf_quo(f, h, p, K)
            g = gf_rem(g, f, p, K)
            b = gf_frobenius_monomial_base(f, p, K)

        i += 1

    if f != [K.one]:
        return factors + [(f, gf_degree(f))]
    else:
        return factors


def gf_edf_zassenhaus(f, n, p, K):
    """
    Cantor-Zassenhaus: Probabilistic Equal Degree Factorization

    Given a monic square-free polynomial ``f`` in ``GF(p)[x]`` and
    an integer ``n``, such that ``n`` divides ``deg(f)``, returns all
    irreducible factors ``f_1,...,f_d`` of ``f``, each of degree ``n``.
    EDF procedure gives complete factorization over Galois fields.

    Consider the square-free polynomial ``f = x**3 + x**2 + x + 1`` in
    ``GF(5)[x]``. Let's compute its irreducible factors of degree one::

       >>> from sympy.polys.domains import ZZ
       >>> from sympy.polys.galoistools import gf_edf_zassenhaus

       >>> gf_edf_zassenhaus([1,1,1,1], 1, 5, ZZ)
       [[1, 1], [1, 2], [1, 3]]

    Notes
    =====

    The case p == 2 is handled by Cohen's Algorithm 3.4.8. The case p odd is
    as in Geddes Algorithm 8.9 (or Cohen's Algorithm 3.4.6).

    References
    ==========

    .. [1] [Gathen99]_
    .. [2] [Geddes92]_ Algorithm 8.9
    .. [3] [Cohen93]_ Algorithm 3.4.8

    """
    factors = [f]

    if gf_degree(f) <= n:
        return factors

    N = gf_degree(f) // n
    if p != 2:
        b = gf_frobenius_monomial_base(f, p, K)

    t = [K.one, K.zero]
    while len(factors) < N:
        if p == 2:
            h = r = t

            for i in range(n - 1):
                r = gf_pow_mod(r, 2, f, p, K)
                h = gf_add(h, r, p, K)

            g = gf_gcd(f, h, p, K)
            t += [K.zero, K.zero]
        else:
            r = gf_random(2 * n - 1, p, K)
            h = _gf_pow_pnm1d2(r, n, f, b, p, K)
            g = gf_gcd(f, gf_sub_ground(h, K.one, p, K), p, K)

        if g != [K.one] and g != f:
            factors = gf_edf_zassenhaus(g, n, p, K) \
                + gf_edf_zassenhaus(gf_quo(f, g, p, K), n, p, K)

    return _sort_factors(factors, multiple=False)


def gf_ddf_shoup(f, p, K):
    """
    Kaltofen-Shoup: Deterministic Distinct Degree Factorization

    Given a monic square-free polynomial ``f`` in ``GF(p)[x]``, computes
    partial distinct degree factorization ``f_1,...,f_d`` of ``f`` where
    ``deg(f_i) != deg(f_j)`` for ``i != j``. The result is returned as a
    list of pairs ``(f_i, e_i)`` where ``deg(f_i) > 0`` and ``e_i > 0``
    is an argument to the equal degree factorization routine.

    This algorithm is an improved version of Zassenhaus algorithm for
    large ``deg(f)`` and modulus ``p`` (especially for ``deg(f) ~ lg(p)``).

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_ddf_shoup, gf_from_dict

    >>> f = gf_from_dict({6: ZZ(1), 5: ZZ(-1), 4: ZZ(1), 3: ZZ(1), 1: ZZ(-1)}, 3, ZZ)

    >>> gf_ddf_shoup(f, 3, ZZ)
    [([1, 1, 0], 1), ([1, 1, 0, 1, 2], 2)]

    References
    ==========

    .. [1] [Kaltofen98]_
    .. [2] [Shoup95]_
    .. [3] [Gathen92]_

    """
    n = gf_degree(f)
    k = int(_ceil(_sqrt(n//2)))
    b = gf_frobenius_monomial_base(f, p, K)
    h = gf_frobenius_map([K.one, K.zero], f, b, p, K)
    # U[i] = x**(p**i)
    U = [[K.one, K.zero], h] + [K.zero]*(k - 1)

    for i in range(2, k + 1):
        U[i] = gf_frobenius_map(U[i-1], f, b, p, K)

    h, U = U[k], U[:k]
    # V[i] = x**(p**(k*(i+1)))
    V = [h] + [K.zero]*(k - 1)

    for i in range(1, k):
        V[i] = gf_compose_mod(V[i - 1], h, f, p, K)

    factors = []

    for i, v in enumerate(V):
        h, j = [K.one], k - 1

        for u in U:
            g = gf_sub(v, u, p, K)
            h = gf_mul(h, g, p, K)
            h = gf_rem(h, f, p, K)

        g = gf_gcd(f, h, p, K)
        f = gf_quo(f, g, p, K)

        for u in reversed(U):
            h = gf_sub(v, u, p, K)
            F = gf_gcd(g, h, p, K)

            if F != [K.one]:
                factors.append((F, k*(i + 1) - j))

            g, j = gf_quo(g, F, p, K), j - 1

    if f != [K.one]:
        factors.append((f, gf_degree(f)))

    return factors

def gf_edf_shoup(f, n, p, K):
    """
    Gathen-Shoup: Probabilistic Equal Degree Factorization

    Given a monic square-free polynomial ``f`` in ``GF(p)[x]`` and integer
    ``n`` such that ``n`` divides ``deg(f)``, returns all irreducible factors
    ``f_1,...,f_d`` of ``f``, each of degree ``n``. This is a complete
    factorization over Galois fields.

    This algorithm is an improved version of Zassenhaus algorithm for
    large ``deg(f)`` and modulus ``p`` (especially for ``deg(f) ~ lg(p)``).

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_edf_shoup

    >>> gf_edf_shoup(ZZ.map([1, 2837, 2277]), 1, 2917, ZZ)
    [[1, 852], [1, 1985]]

    References
    ==========

    .. [1] [Shoup91]_
    .. [2] [Gathen92]_

    """
    N, q = gf_degree(f), int(p)

    if not N:
        return []
    if N <= n:
        return [f]

    factors, x = [f], [K.one, K.zero]

    r = gf_random(N - 1, p, K)

    if p == 2:
        h = gf_pow_mod(x, q, f, p, K)
        H = gf_trace_map(r, h, x, n - 1, f, p, K)[1]
        h1 = gf_gcd(f, H, p, K)
        h2 = gf_quo(f, h1, p, K)

        factors = gf_edf_shoup(h1, n, p, K) \
            + gf_edf_shoup(h2, n, p, K)
    else:
        b = gf_frobenius_monomial_base(f, p, K)
        H = _gf_trace_map(r, n, f, b, p, K)
        h = gf_pow_mod(H, (q - 1)//2, f, p, K)

        h1 = gf_gcd(f, h, p, K)
        h2 = gf_gcd(f, gf_sub_ground(h, K.one, p, K), p, K)
        h3 = gf_quo(f, gf_mul(h1, h2, p, K), p, K)

        factors = gf_edf_shoup(h1, n, p, K) \
            + gf_edf_shoup(h2, n, p, K) \
            + gf_edf_shoup(h3, n, p, K)

    return _sort_factors(factors, multiple=False)


def gf_zassenhaus(f, p, K):
    """
    Factor a square-free ``f`` in ``GF(p)[x]`` for medium ``p``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_zassenhaus

    >>> gf_zassenhaus(ZZ.map([1, 4, 3]), 5, ZZ)
    [[1, 1], [1, 3]]

    """
    factors = []

    for factor, n in gf_ddf_zassenhaus(f, p, K):
        factors += gf_edf_zassenhaus(factor, n, p, K)

    return _sort_factors(factors, multiple=False)


def gf_shoup(f, p, K):
    """
    Factor a square-free ``f`` in ``GF(p)[x]`` for large ``p``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_shoup

    >>> gf_shoup(ZZ.map([1, 4, 3]), 5, ZZ)
    [[1, 1], [1, 3]]

    """
    factors = []

    for factor, n in gf_ddf_shoup(f, p, K):
        factors += gf_edf_shoup(factor, n, p, K)

    return _sort_factors(factors, multiple=False)

_factor_methods = {
    'berlekamp': gf_berlekamp,  # ``p`` : small
    'zassenhaus': gf_zassenhaus,  # ``p`` : medium
    'shoup': gf_shoup,      # ``p`` : large
}


def gf_factor_sqf(f, p, K, method=None):
    """
    Factor a square-free polynomial ``f`` in ``GF(p)[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.galoistools import gf_factor_sqf

    >>> gf_factor_sqf(ZZ.map([3, 2, 4]), 5, ZZ)
    (3, [[1, 1], [1, 3]])

    """
    lc, f = gf_monic(f, p, K)

    if gf_degree(f) < 1:
        return lc, []

    method = method or query('GF_FACTOR_METHOD')

    if method is not None:
        factors = _factor_methods[method](f, p, K)
    else:
        factors = gf_zassenhaus(f, p, K)

    return lc, factors


def gf_factor(f, p, K):
    """
    Factor (non square-free) polynomials in ``GF(p)[x]``.

    Given a possibly non square-free polynomial ``f`` in ``GF(p)[x]``,
    returns its complete factorization into irreducibles::

                 f_1(x)**e_1 f_2(x)**e_2 ... f_d(x)**e_d

    where each ``f_i`` is a monic polynomial and ``gcd(f_i, f_j) == 1``,
    for ``i != j``.  The result is given as a tuple consisting of the
    leading coefficient of ``f`` and a list of factors of ``f`` with
    their multiplicities.

    The algorithm proceeds by first computing square-free decomposition
    of ``f`` and then iteratively factoring each of square-free factors.

    Consider a non square-free polynomial ``f = (7*x + 1) (x + 2)**2`` in
    ``GF(11)[x]``. We obtain its factorization into irreducibles as follows::

       >>> from sympy.polys.domains import ZZ
       >>> from sympy.polys.galoistools import gf_factor

       >>> gf_factor(ZZ.map([5, 2, 7, 2]), 11, ZZ)
       (5, [([1, 2], 1), ([1, 8], 2)])

    We arrived with factorization ``f = 5 (x + 2) (x + 8)**2``. We did not
    recover the exact form of the input polynomial because we requested to
    get monic factors of ``f`` and its leading coefficient separately.

    Square-free factors of ``f`` can be factored into irreducibles over
    ``GF(p)`` using three very different methods:

    Berlekamp
        efficient for very small values of ``p`` (usually ``p < 25``)
    Cantor-Zassenhaus
        efficient on average input and with "typical" ``p``
    Shoup-Kaltofen-Gathen
        efficient with very large inputs and modulus

    If you want to use a specific factorization method, instead of the default
    one, set ``GF_FACTOR_METHOD`` with one of ``berlekamp``, ``zassenhaus`` or
    ``shoup`` values.

    References
    ==========

    .. [1] [Gathen99]_

    """
    lc, f = gf_monic(f, p, K)

    if gf_degree(f) < 1:
        return lc, []

    factors = []

    for g, n in gf_sqf_list(f, p, K)[1]:
        for h in gf_factor_sqf(g, p, K)[1]:
            factors.append((h, n))

    return lc, _sort_factors(factors)


def gf_value(f, a):
    """
    Value of polynomial 'f' at 'a' in field R.

    Examples
    ========

    >>> from sympy.polys.galoistools import gf_value

    >>> gf_value([1, 7, 2, 4], 11)
    2204

    """
    result = 0
    for c in f:
        result *= a
        result += c
    return result


def linear_congruence(a, b, m):
    """
    Returns the values of x satisfying a*x congruent b mod(m)

    Here m is positive integer and a, b are natural numbers.
    This function returns only those values of x which are distinct mod(m).

    Examples
    ========

    >>> from sympy.polys.galoistools import linear_congruence

    >>> linear_congruence(3, 12, 15)
    [4, 9, 14]

    There are 3 solutions distinct mod(15) since gcd(a, m) = gcd(3, 15) = 3.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Linear_congruence_theorem

    """
    from sympy.polys.polytools import gcdex
    if a % m == 0:
        if b % m == 0:
            return list(range(m))
        else:
            return []
    r, _, g = gcdex(a, m)
    if b % g != 0:
        return []
    return [(r * b // g + t * m // g) % m for t in range(g)]


def _raise_mod_power(x, s, p, f):
    """
    Used in gf_csolve to generate solutions of f(x) cong 0 mod(p**(s + 1))
    from the solutions of f(x) cong 0 mod(p**s).

    Examples
    ========

    >>> from sympy.polys.galoistools import _raise_mod_power
    >>> from sympy.polys.galoistools import csolve_prime

    These is the solutions of f(x) = x**2 + x + 7 cong 0 mod(3)

    >>> f = [1, 1, 7]
    >>> csolve_prime(f, 3)
    [1]
    >>> [ i for i in range(3) if not (i**2 + i + 7) % 3]
    [1]

    The solutions of f(x) cong 0 mod(9) are constructed from the
    values returned from _raise_mod_power:

    >>> x, s, p = 1, 1, 3
    >>> V = _raise_mod_power(x, s, p, f)
    >>> [x + v * p**s for v in V]
    [1, 4, 7]

    And these are confirmed with the following:

    >>> [ i for i in range(3**2) if not (i**2 + i + 7) % 3**2]
    [1, 4, 7]

    """
    from sympy.polys.domains import ZZ
    f_f = gf_diff(f, p, ZZ)
    alpha = gf_value(f_f, x)
    beta = - gf_value(f, x) // p**s
    return linear_congruence(alpha, beta, p)


def _csolve_prime_las_vegas(f, p, seed=None):
    r""" Solutions of `f(x) \equiv 0 \pmod{p}`, `f(0) \not\equiv 0 \pmod{p}`.

    Explanation
    ===========

    This algorithm is classified as the Las Vegas method.
    That is, it always returns the correct answer and solves the problem
    fast in many cases, but if it is unlucky, it does not answer forever.

    Suppose the polynomial f is not a zero polynomial. Assume further
    that it is of degree at most p-1 and `f(0)\not\equiv 0 \pmod{p}`.
    These assumptions are not an essential part of the algorithm,
    only that it is more convenient for the function calling this
    function to resolve them.

    Note that `x^{p-1} - 1 \equiv \prod_{a=1}^{p-1}(x - a) \pmod{p}`.
    Thus, the greatest common divisor with f is `\prod_{s \in S}(x - s)`,
    with S being the set of solutions to f. Furthermore,
    when a is randomly determined, `(x+a)^{(p-1)/2}-1` is
    a polynomial with (p-1)/2 randomly chosen solutions.
    The greatest common divisor of f may be a nontrivial factor of f.

    When p is large and the degree of f is small,
    it is faster than naive solution methods.

    Parameters
    ==========

    f : polynomial
    p : prime number

    Returns
    =======

    list[int]
        a list of solutions, sorted in ascending order
        by integers in the range [1, p). The same value
        does not exist in the list even if there is
        a multiple solution. If no solution exists, returns [].

    Examples
    ========

    >>> from sympy.polys.galoistools import _csolve_prime_las_vegas
    >>> _csolve_prime_las_vegas([1, 4, 3], 7) # x^2 + 4x + 3 = 0 (mod 7)
    [4, 6]
    >>> _csolve_prime_las_vegas([5, 7, 1, 9], 11) # 5x^3 + 7x^2 + x + 9 = 0 (mod 11)
    [1, 5, 8]

    References
    ==========

    .. [1] R. Crandall and C. Pomerance "Prime Numbers", 2nd Ed., Algorithm 2.3.10

    """
    from sympy.polys.domains import ZZ
    from sympy.ntheory import sqrt_mod
    randint = _randint(seed)
    root = set()
    g = gf_pow_mod([1, 0], p - 1, f, p, ZZ)
    g = gf_sub_ground(g, 1, p, ZZ)
    # We want to calculate gcd(x**(p-1) - 1, f(x))
    factors = [gf_gcd(f, g, p, ZZ)]
    while factors:
        f = factors.pop()
        # If the degree is small, solve directly
        if len(f) <= 1:
            continue
        if len(f) == 2:
            root.add(-invert(f[0], p) * f[1] % p)
            continue
        if len(f) == 3:
            inv = invert(f[0], p)
            b = f[1] * inv % p
            b = (b + p * (b % 2)) // 2
            root.update((r - b) % p for r in
                        sqrt_mod(b**2 - f[2] * inv, p, all_roots=True))
            continue
        while True:
            # Determine `a` randomly and
            # compute gcd((x+a)**((p-1)//2)-1, f(x))
            a = randint(0, p - 1)
            g = gf_pow_mod([1, a], (p - 1) // 2, f, p, ZZ)
            g = gf_sub_ground(g, 1, p, ZZ)
            g = gf_gcd(f, g, p, ZZ)
            if 1 < len(g) < len(f):
                factors.append(g)
                factors.append(gf_div(f, g, p, ZZ)[0])
                break
    return sorted(root)


def csolve_prime(f, p, e=1):
    r""" Solutions of `f(x) \equiv 0 \pmod{p^e}`.

    Parameters
    ==========

    f : polynomial
    p : prime number
    e : positive integer

    Returns
    =======

    list[int]
        a list of solutions, sorted in ascending order
        by integers in the range [1, p**e). The same value
        does not exist in the list even if there is
        a multiple solution. If no solution exists, returns [].

    Examples
    ========

    >>> from sympy.polys.galoistools import csolve_prime
    >>> csolve_prime([1, 1, 7], 3, 1)
    [1]
    >>> csolve_prime([1, 1, 7], 3, 2)
    [1, 4, 7]

    Solutions [7, 4, 1] (mod 3**2) are generated by ``_raise_mod_power()``
    from solution [1] (mod 3).
    """
    from sympy.polys.domains import ZZ
    g = [MPZ(int(c)) for c in f]
    # Convert to polynomial of degree at most p-1
    for i in range(len(g) - p):
        g[i + p - 1] += g[i]
        g[i] = 0
    g = gf_trunc(g, p)
    # Checks whether g(x) is divisible by x
    k = 0
    while k < len(g) and g[len(g) - k - 1] == 0:
        k += 1
    if k:
        g = g[:-k]
        root_zero = [0]
    else:
        root_zero = []
    if g == []:
        X1 = list(range(p))
    elif len(g)**2 < p:
        # The conditions under which `_csolve_prime_las_vegas` is faster than
        # a naive solution are worth considering.
        X1 = root_zero + _csolve_prime_las_vegas(g, p)
    else:
        X1 = root_zero + [i for i in range(p) if gf_eval(g, i, p, ZZ) == 0]
    if e == 1:
        return X1
    X = []
    S = list(zip(X1, [1]*len(X1)))
    while S:
        x, s = S.pop()
        if s == e:
            X.append(x)
        else:
            s1 = s + 1
            ps = p**s
            S.extend([(x + v*ps, s1) for v in _raise_mod_power(x, s, p, f)])
    return sorted(X)


def gf_csolve(f, n):
    """
    To solve f(x) congruent 0 mod(n).

    n is divided into canonical factors and f(x) cong 0 mod(p**e) will be
    solved for each factor. Applying the Chinese Remainder Theorem to the
    results returns the final answers.

    Examples
    ========

    Solve [1, 1, 7] congruent 0 mod(189):

    >>> from sympy.polys.galoistools import gf_csolve
    >>> gf_csolve([1, 1, 7], 189)
    [13, 49, 76, 112, 139, 175]

    See Also
    ========

    sympy.ntheory.residue_ntheory.polynomial_congruence : a higher level solving routine

    References
    ==========

    .. [1] 'An introduction to the Theory of Numbers' 5th Edition by Ivan Niven,
           Zuckerman and Montgomery.

    """
    from sympy.polys.domains import ZZ
    from sympy.ntheory import factorint
    P = factorint(n)
    X = [csolve_prime(f, p, e) for p, e in P.items()]
    pools = list(map(tuple, X))
    perms = [[]]
    for pool in pools:
        perms = [x + [y] for x in perms for y in pool]
    dist_factors = [pow(p, e) for p, e in P.items()]
    return sorted([gf_crt(per, dist_factors, ZZ) for per in perms])
<|endoftext|>"""OO layer for several polynomial representations. """

from __future__ import annotations

from sympy.external.gmpy import GROUND_TYPES

from sympy.utilities.exceptions import sympy_deprecation_warning

from sympy.core.numbers import oo
from sympy.core.sympify import CantSympify
from sympy.polys.polyutils import PicklableWithSlots, _sort_factors
from sympy.polys.domains import Domain, ZZ, QQ

from sympy.polys.polyerrors import (
    CoercionFailed,
    ExactQuotientFailed,
    DomainError,
    NotInvertible,
)

from sympy.polys.densebasic import (
    ninf,
    dmp_validate,
    dup_normal, dmp_normal,
    dup_convert, dmp_convert,
    dmp_from_sympy,
    dup_strip,
    dmp_degree_in,
    dmp_degree_list,
    dmp_negative_p,
    dmp_ground_LC,
    dmp_ground_TC,
    dmp_ground_nth,
    dmp_one, dmp_ground,
    dmp_zero, dmp_zero_p, dmp_one_p, dmp_ground_p,
    dup_from_dict, dmp_from_dict,
    dmp_to_dict,
    dmp_deflate,
    dmp_inject, dmp_eject,
    dmp_terms_gcd,
    dmp_list_terms, dmp_exclude,
    dup_slice, dmp_slice_in, dmp_permute,
    dmp_to_tuple,)

from sympy.polys.densearith import (
    dmp_add_ground,
    dmp_sub_ground,
    dmp_mul_ground,
    dmp_quo_ground,
    dmp_exquo_ground,
    dmp_abs,
    dmp_neg,
    dmp_add,
    dmp_sub,
    dmp_mul,
    dmp_sqr,
    dmp_pow,
    dmp_pdiv,
    dmp_prem,
    dmp_pquo,
    dmp_pexquo,
    dmp_div,
    dmp_rem,
    dmp_quo,
    dmp_exquo,
    dmp_add_mul, dmp_sub_mul,
    dmp_max_norm,
    dmp_l1_norm,
    dmp_l2_norm_squared)

from sympy.polys.densetools import (
    dmp_clear_denoms,
    dmp_integrate_in,
    dmp_diff_in,
    dmp_eval_in,
    dup_revert,
    dmp_ground_trunc,
    dmp_ground_content,
    dmp_ground_primitive,
    dmp_ground_monic,
    dmp_compose,
    dup_decompose,
    dup_shift,
    dmp_shift,
    dup_transform,
    dmp_lift)

from sympy.polys.euclidtools import (
    dup_half_gcdex, dup_gcdex, dup_invert,
    dmp_subresultants,
    dmp_resultant,
    dmp_discriminant,
    dmp_inner_gcd,
    dmp_gcd,
    dmp_lcm,
    dmp_cancel)

from sympy.polys.sqfreetools import (
    dup_gff_list,
    dmp_norm,
    dmp_sqf_p,
    dmp_sqf_norm,
    dmp_sqf_part,
    dmp_sqf_list, dmp_sqf_list_include)

from sympy.polys.factortools import (
    dup_cyclotomic_p, dmp_irreducible_p,
    dmp_factor_list, dmp_factor_list_include)

from sympy.polys.rootisolation import (
    dup_isolate_real_roots_sqf,
    dup_isolate_real_roots,
    dup_isolate_all_roots_sqf,
    dup_isolate_all_roots,
    dup_refine_real_root,
    dup_count_real_roots,
    dup_count_complex_roots,
    dup_sturm,
    dup_cauchy_upper_bound,
    dup_cauchy_lower_bound,
    dup_mignotte_sep_bound_squared)

from sympy.polys.polyerrors import (
    UnificationFailed,
    PolynomialError)


if GROUND_TYPES == 'flint':
    import flint
    def _supported_flint_domain(D):
        return D.is_ZZ or D.is_QQ or D.is_FF and D._is_flint
else:
    flint = None
    def _supported_flint_domain(D):
        return False


class DMP(CantSympify):
    """Dense Multivariate Polynomials over `K`. """

    __slots__ = ()

    lev: int
    dom: Domain

    def __new__(cls, rep, dom, lev=None):

        if lev is None:
            rep, lev = dmp_validate(rep)
        elif not isinstance(rep, list):
            raise CoercionFailed("expected list, got %s" % type(rep))

        return cls.new(rep, dom, lev)

    @classmethod
    def new(cls, rep, dom, lev):
        # It would be too slow to call _validate_args always at runtime.
        # Ideally this checking would be handled by a static type checker.
        #
        #cls._validate_args(rep, dom, lev)
        if flint is not None:
            if lev == 0 and _supported_flint_domain(dom):
                return DUP_Flint._new(rep, dom, lev)

        return DMP_Python._new(rep, dom, lev)

    @property
    def rep(f):
        """Get the representation of ``f``. """

        sympy_deprecation_warning("""
        Accessing the ``DMP.rep`` attribute is deprecated. The internal
        representation of ``DMP`` instances can now be ``DUP_Flint`` when the
        ground types are ``flint``. In this case the ``DMP`` instance does not
        have a ``rep`` attribute. Use ``DMP.to_list()`` instead. Using
        ``DMP.to_list()`` also works in previous versions of SymPy.
        """,
            deprecated_since_version="1.13",
            active_deprecations_target="dmp-rep",
        )

        return f.to_list()

    def to_best(f):
        """Convert to DUP_Flint if possible.

        This method should be used when the domain or level is changed and it
        potentially becomes possible to convert from DMP_Python to DUP_Flint.
        """
        if flint is not None:
            if isinstance(f, DMP_Python) and f.lev == 0 and _supported_flint_domain(f.dom):
                return DUP_Flint.new(f._rep, f.dom, f.lev)

        return f

    @classmethod
    def _validate_args(cls, rep, dom, lev):
        assert isinstance(dom, Domain)
        assert isinstance(lev, int) and lev >= 0

        def validate_rep(rep, lev):
            assert isinstance(rep, list)
            if lev == 0:
                assert all(dom.of_type(c) for c in rep)
            else:
                for r in rep:
                    validate_rep(r, lev - 1)

        validate_rep(rep, lev)

    @classmethod
    def from_dict(cls, rep, lev, dom):
        rep = dmp_from_dict(rep, lev, dom)
        return cls.new(rep, dom, lev)

    @classmethod
    def from_list(cls, rep, lev, dom):
        """Create an instance of ``cls`` given a list of native coefficients. """
        return cls.new(dmp_convert(rep, lev, None, dom), dom, lev)

    @classmethod
    def from_sympy_list(cls, rep, lev, dom):
        """Create an instance of ``cls`` given a list of SymPy coefficients. """
        return cls.new(dmp_from_sympy(rep, lev, dom), dom, lev)

    @classmethod
    def from_monoms_coeffs(cls, monoms, coeffs, lev, dom):
        return cls(dict(list(zip(monoms, coeffs))), dom, lev)

    def convert(f, dom):
        """Convert ``f`` to a ``DMP`` over the new domain. """
        if f.dom == dom:
            return f
        elif f.lev or flint is None:
            return f._convert(dom)
        elif isinstance(f, DUP_Flint):
            if _supported_flint_domain(dom):
                return f._convert(dom)
            else:
                return f.to_DMP_Python()._convert(dom)
        elif isinstance(f, DMP_Python):
            if _supported_flint_domain(dom):
                return f._convert(dom).to_DUP_Flint()
            else:
                return f._convert(dom)
        else:
            raise RuntimeError("unreachable code")

    def _convert(f, dom):
        raise NotImplementedError

    @classmethod
    def zero(cls, lev, dom):
        return DMP(dmp_zero(lev), dom, lev)

    @classmethod
    def one(cls, lev, dom):
        return DMP(dmp_one(lev, dom), dom, lev)

    def _one(f):
        raise NotImplementedError

    def __repr__(f):
        return "%s(%s, %s)" % (f.__class__.__name__, f.to_list(), f.dom)

    def __hash__(f):
        return hash((f.__class__.__name__, f.to_tuple(), f.lev, f.dom))

    def __getnewargs__(self):
        return self.to_list(), self.dom, self.lev

    def ground_new(f, coeff):
        """Construct a new ground instance of ``f``. """
        raise NotImplementedError

    def unify_DMP(f, g):
        """Unify and return ``DMP`` instances of ``f`` and ``g``. """
        if not isinstance(g, DMP) or f.lev != g.lev:
            raise UnificationFailed("Cannot unify %s with %s" % (f, g))

        if f.dom != g.dom:
            dom = f.dom.unify(g.dom)
            f = f.convert(dom)
            g = g.convert(dom)

        return f, g

    def to_dict(f, zero=False):
        """Convert ``f`` to a dict representation with native coefficients. """
        return dmp_to_dict(f.to_list(), f.lev, f.dom, zero=zero)

    def to_sympy_dict(f, zero=False):
        """Convert ``f`` to a dict representation with SymPy coefficients. """
        rep = f.to_dict(zero=zero)

        for k, v in rep.items():
            rep[k] = f.dom.to_sympy(v)

        return rep

    def to_sympy_list(f):
        """Convert ``f`` to a list representation with SymPy coefficients. """
        def sympify_nested_list(rep):
            out = []
            for val in rep:
                if isinstance(val, list):
                    out.append(sympify_nested_list(val))
                else:
                    out.append(f.dom.to_sympy(val))
            return out

        return sympify_nested_list(f.to_list())

    def to_list(f):
        """Convert ``f`` to a list representation with native coefficients. """
        raise NotImplementedError

    def to_tuple(f):
        """
        Convert ``f`` to a tuple representation with native coefficients.

        This is needed for hashing.
        """
        raise NotImplementedError

    def to_ring(f):
        """Make the ground domain a ring. """
        return f.convert(f.dom.get_ring())

    def to_field(f):
        """Make the ground domain a field. """
        return f.convert(f.dom.get_field())

    def to_exact(f):
        """Make the ground domain exact. """
        return f.convert(f.dom.get_exact())

    def slice(f, m, n, j=0):
        """Take a continuous subsequence of terms of ``f``. """
        if not f.lev and not j:
            return f._slice(m, n)
        else:
            return f._slice_lev(m, n, j)

    def _slice(f, m, n):
        raise NotImplementedError

    def _slice_lev(f, m, n, j):
        raise NotImplementedError

    def coeffs(f, order=None):
        """Returns all non-zero coefficients from ``f`` in lex order. """
        return [ c for _, c in f.terms(order=order) ]

    def monoms(f, order=None):
        """Returns all non-zero monomials from ``f`` in lex order. """
        return [ m for m, _ in f.terms(order=order) ]

    def terms(f, order=None):
        """Returns all non-zero terms from ``f`` in lex order. """
        if f.is_zero:
            zero_monom = (0,)*(f.lev + 1)
            return [(zero_monom, f.dom.zero)]
        else:
            return f._terms(order=order)

    def _terms(f, order=None):
        raise NotImplementedError

    def all_coeffs(f):
        """Returns all coefficients from ``f``. """
        if f.lev:
            raise PolynomialError('multivariate polynomials not supported')

        if not f:
            return [f.dom.zero]
        else:
            return list(f.to_list())

    def all_monoms(f):
        """Returns all monomials from ``f``. """
        if f.lev:
            raise PolynomialError('multivariate polynomials not supported')

        n = f.degree()

        if n < 0:
            return [(0,)]
        else:
            return [ (n - i,) for i, c in enumerate(f.to_list()) ]

    def all_terms(f):
        """Returns all terms from a ``f``. """
        if f.lev:
            raise PolynomialError('multivariate polynomials not supported')

        n = f.degree()

        if n < 0:
            return [((0,), f.dom.zero)]
        else:
            return [ ((n - i,), c) for i, c in enumerate(f.to_list()) ]

    def lift(f):
        """Convert algebraic coefficients to rationals. """
        return f._lift().to_best()

    def _lift(f):
        raise NotImplementedError

    def deflate(f):
        """Reduce degree of `f` by mapping `x_i^m` to `y_i`. """
        raise NotImplementedError

    def inject(f, front=False):
        """Inject ground domain generators into ``f``. """
        raise NotImplementedError

    def eject(f, dom, front=False):
        """Eject selected generators into the ground domain. """
        raise NotImplementedError

    def exclude(f):
        r"""
        Remove useless generators from ``f``.

        Returns the removed generators and the new excluded ``f``.

        Examples
        ========

        >>> from sympy.polys.polyclasses import DMP
        >>> from sympy.polys.domains import ZZ

        >>> DMP([[[ZZ(1)]], [[ZZ(1)], [ZZ(2)]]], ZZ).exclude()
        ([2], DMP_Python([[1], [1, 2]], ZZ))

        """
        J, F = f._exclude()
        return J, F.to_best()

    def _exclude(f):
        raise NotImplementedError

    def permute(f, P):
        r"""
        Returns a polynomial in `K[x_{P(1)}, ..., x_{P(n)}]`.

        Examples
        ========

        >>> from sympy.polys.polyclasses import DMP
        >>> from sympy.polys.domains import ZZ

        >>> DMP([[[ZZ(2)], [ZZ(1), ZZ(0)]], [[]]], ZZ).permute([1, 0, 2])
        DMP_Python([[[2], []], [[1, 0], []]], ZZ)

        >>> DMP([[[ZZ(2)], [ZZ(1), ZZ(0)]], [[]]], ZZ).permute([1, 2, 0])
        DMP_Python([[[1], []], [[2, 0], []]], ZZ)

        """
        return f._permute(P)

    def _permute(f, P):
        raise NotImplementedError

    def terms_gcd(f):
        """Remove GCD of terms from the polynomial ``f``. """
        raise NotImplementedError

    def abs(f):
        """Make all coefficients in ``f`` positive. """
        raise NotImplementedError

    def neg(f):
        """Negate all coefficients in ``f``. """
        raise NotImplementedError

    def add_ground(f, c):
        """Add an element of the ground domain to ``f``. """
        return f._add_ground(f.dom.convert(c))

    def sub_ground(f, c):
        """Subtract an element of the ground domain from ``f``. """
        return f._sub_ground(f.dom.convert(c))

    def mul_ground(f, c):
        """Multiply ``f`` by a an element of the ground domain. """
        return f._mul_ground(f.dom.convert(c))

    def quo_ground(f, c):
        """Quotient of ``f`` by a an element of the ground domain. """
        return f._quo_ground(f.dom.convert(c))

    def exquo_ground(f, c):
        """Exact quotient of ``f`` by a an element of the ground domain. """
        return f._exquo_ground(f.dom.convert(c))

    def add(f, g):
        """Add two multivariate polynomials ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._add(G)

    def sub(f, g):
        """Subtract two multivariate polynomials ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._sub(G)

    def mul(f, g):
        """Multiply two multivariate polynomials ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._mul(G)

    def sqr(f):
        """Square a multivariate polynomial ``f``. """
        return f._sqr()

    def pow(f, n):
        """Raise ``f`` to a non-negative power ``n``. """
        if not isinstance(n, int):
            raise TypeError("``int`` expected, got %s" % type(n))
        return f._pow(n)

    def pdiv(f, g):
        """Polynomial pseudo-division of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._pdiv(G)

    def prem(f, g):
        """Polynomial pseudo-remainder of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._prem(G)

    def pquo(f, g):
        """Polynomial pseudo-quotient of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._pquo(G)

    def pexquo(f, g):
        """Polynomial exact pseudo-quotient of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._pexquo(G)

    def div(f, g):
        """Polynomial division with remainder of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._div(G)

    def rem(f, g):
        """Computes polynomial remainder of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._rem(G)

    def quo(f, g):
        """Computes polynomial quotient of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._quo(G)

    def exquo(f, g):
        """Computes polynomial exact quotient of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._exquo(G)

    def _add_ground(f, c):
        raise NotImplementedError

    def _sub_ground(f, c):
        raise NotImplementedError

    def _mul_ground(f, c):
        raise NotImplementedError

    def _quo_ground(f, c):
        raise NotImplementedError

    def _exquo_ground(f, c):
        raise NotImplementedError

    def _add(f, g):
        raise NotImplementedError

    def _sub(f, g):
        raise NotImplementedError

    def _mul(f, g):
        raise NotImplementedError

    def _sqr(f):
        raise NotImplementedError

    def _pow(f, n):
        raise NotImplementedError

    def _pdiv(f, g):
        raise NotImplementedError

    def _prem(f, g):
        raise NotImplementedError

    def _pquo(f, g):
        raise NotImplementedError

    def _pexquo(f, g):
        raise NotImplementedError

    def _div(f, g):
        raise NotImplementedError

    def _rem(f, g):
        raise NotImplementedError

    def _quo(f, g):
        raise NotImplementedError

    def _exquo(f, g):
        raise NotImplementedError

    def degree(f, j=0):
        """Returns the leading degree of ``f`` in ``x_j``. """
        if not isinstance(j, int):
            raise TypeError("``int`` expected, got %s" % type(j))

        return f._degree(j)

    def _degree(f, j):
        raise NotImplementedError

    def degree_list(f):
        """Returns a list of degrees of ``f``. """
        raise NotImplementedError

    def total_degree(f):
        """Returns the total degree of ``f``. """
        raise NotImplementedError

    def homogenize(f, s):
        """Return homogeneous polynomial of ``f``"""
        td = f.total_degree()
        result = {}
        new_symbol = (s == len(f.terms()[0][0]))
        for term in f.terms():
            d = sum(term[0])
            if d < td:
                i = td - d
            else:
                i = 0
            if new_symbol:
                result[term[0] + (i,)] = term[1]
            else:
                l = list(term[0])
                l[s] += i
                result[tuple(l)] = term[1]
        return DMP.from_dict(result, f.lev + int(new_symbol), f.dom)

    def homogeneous_order(f):
        """Returns the homogeneous order of ``f``. """
        if f.is_zero:
            return -oo

        monoms = f.monoms()
        tdeg = sum(monoms[0])

        for monom in monoms:
            _tdeg = sum(monom)

            if _tdeg != tdeg:
                return None

        return tdeg

    def LC(f):
        """Returns the leading coefficient of ``f``. """
        raise NotImplementedError

    def TC(f):
        """Returns the trailing coefficient of ``f``. """
        raise NotImplementedError

    def nth(f, *N):
        """Returns the ``n``-th coefficient of ``f``. """
        if all(isinstance(n, int) for n in N):
            return f._nth(N)
        else:
            raise TypeError("a sequence of integers expected")

    def _nth(f, N):
        raise NotImplementedError

    def max_norm(f):
        """Returns maximum norm of ``f``. """
        raise NotImplementedError

    def l1_norm(f):
        """Returns l1 norm of ``f``. """
        raise NotImplementedError

    def l2_norm_squared(f):
        """Return squared l2 norm of ``f``. """
        raise NotImplementedError

    def clear_denoms(f):
        """Clear denominators, but keep the ground domain. """
        raise NotImplementedError

    def integrate(f, m=1, j=0):
        """Computes the ``m``-th order indefinite integral of ``f`` in ``x_j``. """
        if not isinstance(m, int):
            raise TypeError("``int`` expected, got %s" % type(m))

        if not isinstance(j, int):
            raise TypeError("``int`` expected, got %s" % type(j))

        return f._integrate(m, j)

    def _integrate(f, m, j):
        raise NotImplementedError

    def diff(f, m=1, j=0):
        """Computes the ``m``-th order derivative of ``f`` in ``x_j``. """
        if not isinstance(m, int):
            raise TypeError("``int`` expected, got %s" % type(m))

        if not isinstance(j, int):
            raise TypeError("``int`` expected, got %s" % type(j))

        return f._diff(m, j)

    def _diff(f, m, j):
        raise NotImplementedError

    def eval(f, a, j=0):
        """Evaluates ``f`` at the given point ``a`` in ``x_j``. """
        if not isinstance(j, int):
            raise TypeError("``int`` expected, got %s" % type(j))
        elif not (0 <= j <= f.lev):
            raise ValueError("invalid variable index %s" % j)

        if f.lev:
            return f._eval_lev(a, j)
        else:
            return f._eval(a)

    def _eval(f, a):
        raise NotImplementedError

    def _eval_lev(f, a, j):
        raise NotImplementedError

    def half_gcdex(f, g):
        """Half extended Euclidean algorithm, if univariate. """
        F, G = f.unify_DMP(g)

        if F.lev:
            raise ValueError('univariate polynomial expected')

        return F._half_gcdex(G)

    def _half_gcdex(f, g):
        raise NotImplementedError

    def gcdex(f, g):
        """Extended Euclidean algorithm, if univariate. """
        F, G = f.unify_DMP(g)

        if F.lev:
            raise ValueError('univariate polynomial expected')

        if not F.dom.is_Field:
            raise DomainError('ground domain must be a field')

        return F._gcdex(G)

    def _gcdex(f, g):
        raise NotImplementedError

    def invert(f, g):
        """Invert ``f`` modulo ``g``, if possible. """
        F, G = f.unify_DMP(g)

        if F.lev:
            raise ValueError('univariate polynomial expected')

        return F._invert(G)

    def _invert(f, g):
        raise NotImplementedError

    def revert(f, n):
        """Compute ``f**(-1)`` mod ``x**n``. """
        if f.lev:
            raise ValueError('univariate polynomial expected')

        return f._revert(n)

    def _revert(f, n):
        raise NotImplementedError

    def subresultants(f, g):
        """Computes subresultant PRS sequence of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._subresultants(G)

    def _subresultants(f, g):
        raise NotImplementedError

    def resultant(f, g, includePRS=False):
        """Computes resultant of ``f`` and ``g`` via PRS. """
        F, G = f.unify_DMP(g)
        if includePRS:
            return F._resultant_includePRS(G)
        else:
            return F._resultant(G)

    def _resultant(f, g, includePRS=False):
        raise NotImplementedError

    def discriminant(f):
        """Computes discriminant of ``f``. """
        raise NotImplementedError

    def cofactors(f, g):
        """Returns GCD of ``f`` and ``g`` and their cofactors. """
        F, G = f.unify_DMP(g)
        return F._cofactors(G)

    def _cofactors(f, g):
        raise NotImplementedError

    def gcd(f, g):
        """Returns polynomial GCD of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._gcd(G)

    def _gcd(f, g):
        raise NotImplementedError

    def lcm(f, g):
        """Returns polynomial LCM of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._lcm(G)

    def _lcm(f, g):
        raise NotImplementedError

    def cancel(f, g, include=True):
        """Cancel common factors in a rational function ``f/g``. """
        F, G = f.unify_DMP(g)

        if include:
            return F._cancel_include(G)
        else:
            return F._cancel(G)

    def _cancel(f, g):
        raise NotImplementedError

    def _cancel_include(f, g):
        raise NotImplementedError

    def trunc(f, p):
        """Reduce ``f`` modulo a constant ``p``. """
        return f._trunc(f.dom.convert(p))

    def _trunc(f, p):
        raise NotImplementedError

    def monic(f):
        """Divides all coefficients by ``LC(f)``. """
        raise NotImplementedError

    def content(f):
        """Returns GCD of polynomial coefficients. """
        raise NotImplementedError

    def primitive(f):
        """Returns content and a primitive form of ``f``. """
        raise NotImplementedError

    def compose(f, g):
        """Computes functional composition of ``f`` and ``g``. """
        F, G = f.unify_DMP(g)
        return F._compose(G)

    def _compose(f, g):
        raise NotImplementedError

    def decompose(f):
        """Computes functional decomposition of ``f``. """
        if f.lev:
            raise ValueError('univariate polynomial expected')

        return f._decompose()

    def _decompose(f):
        raise NotImplementedError

    def shift(f, a):
        """Efficiently compute Taylor shift ``f(x + a)``. """
        if f.lev:
            raise ValueError('univariate polynomial expected')

        return f._shift(f.dom.convert(a))

    def shift_list(f, a):
        """Efficiently compute Taylor shift ``f(X + A)``. """
        a = [f.dom.convert(ai) for ai in a]
        return f._shift_list(a)

    def _shift(f, a):
        raise NotImplementedError

    def transform(f, p, q):
        """Evaluate functional transformation ``q**n * f(p/q)``."""
        if f.lev:
            raise ValueError('univariate polynomial expected')

        P, Q = p.unify_DMP(q)
        F, P = f.unify_DMP(P)
        F, Q = F.unify_DMP(Q)

        return F._transform(P, Q)

    def _transform(f, p, q):
        raise NotImplementedError

    def sturm(f):
        """Computes the Sturm sequence of ``f``. """
        if f.lev:
            raise ValueError('univariate polynomial expected')

        return f._sturm()

    def _sturm(f):
        raise NotImplementedError

    def cauchy_upper_bound(f):
        """Computes the Cauchy upper bound on the roots of ``f``. """
        if f.lev:
            raise ValueError('univariate polynomial expected')

        return f._cauchy_upper_bound()

    def _cauchy_upper_bound(f):
        raise NotImplementedError

    def cauchy_lower_bound(f):
        """Computes the Cauchy lower bound on the nonzero roots of ``f``. """
        if f.lev:
            raise ValueError('univariate polynomial expected')

        return f._cauchy_lower_bound()

    def _cauchy_lower_bound(f):
        raise NotImplementedError

    def mignotte_sep_bound_squared(f):
        """Computes the squared Mignotte bound on root separations of ``f``. """
        if f.lev:
            raise ValueError('univariate polynomial expected')

        return f._mignotte_sep_bound_squared()

    def _mignotte_sep_bound_squared(f):
        raise NotImplementedError

    def gff_list(f):
        """Computes greatest factorial factorization of ``f``. """
        if f.lev:
            raise ValueError('univariate polynomial expected')

        return f._gff_list()

    def _gff_list(f):
        raise NotImplementedError

    def norm(f):
        """Computes ``Norm(f)``."""
        raise NotImplementedError

    def sqf_norm(f):
        """Computes square-free norm of ``f``. """
        raise NotImplementedError

    def sqf_part(f):
        """Computes square-free part of ``f``. """
        raise NotImplementedError

    def sqf_list(f, all=False):
        """Returns a list of square-free factors of ``f``. """
        raise NotImplementedError

    def sqf_list_include(f, all=False):
        """Returns a list of square-free factors of ``f``. """
        raise NotImplementedError

    def factor_list(f):
        """Returns a list of irreducible factors of ``f``. """
        raise NotImplementedError

    def factor_list_include(f):
        """Returns a list of irreducible factors of ``f``. """
        raise NotImplementedError

    def intervals(f, all=False, eps=None, inf=None, sup=None, fast=False, sqf=False):
        """Compute isolating intervals for roots of ``f``. """
        if f.lev:
            raise PolynomialError("Cannot isolate roots of a multivariate polynomial")

        if all and sqf:
            return f._isolate_all_roots_sqf(eps=eps, inf=inf, sup=sup, fast=fast)
        elif all and not sqf:
            return f._isolate_all_roots(eps=eps, inf=inf, sup=sup, fast=fast)
        elif not all and sqf:
            return f._isolate_real_roots_sqf(eps=eps, inf=inf, sup=sup, fast=fast)
        else:
            return f._isolate_real_roots(eps=eps, inf=inf, sup=sup, fast=fast)

    def _isolate_all_roots(f, eps, inf, sup, fast):
        raise NotImplementedError

    def _isolate_all_roots_sqf(f, eps, inf, sup, fast):
        raise NotImplementedError

    def _isolate_real_roots(f, eps, inf, sup, fast):
        raise NotImplementedError

    def _isolate_real_roots_sqf(f, eps, inf, sup, fast):
        raise NotImplementedError

    def refine_root(f, s, t, eps=None, steps=None, fast=False):
        """
        Refine an isolating interval to the given precision.

        ``eps`` should be a rational number.

        """
        if f.lev:
            raise PolynomialError(
                "Cannot refine a root of a multivariate polynomial")

        return f._refine_real_root(s, t, eps=eps, steps=steps, fast=fast)

    def _refine_real_root(f, s, t, eps, steps, fast):
        raise NotImplementedError

    def count_real_roots(f, inf=None, sup=None):
        """Return the number of real roots of ``f`` in ``[inf, sup]``. """
        raise NotImplementedError

    def count_complex_roots(f, inf=None, sup=None):
        """Return the number of complex roots of ``f`` in ``[inf, sup]``. """
        raise NotImplementedError

    @property
    def is_zero(f):
        """Returns ``True`` if ``f`` is a zero polynomial. """
        raise NotImplementedError

    @property
    def is_one(f):
        """Returns ``True`` if ``f`` is a unit polynomial. """
        raise NotImplementedError

    @property
    def is_ground(f):
        """Returns ``True`` if ``f`` is an element of the ground domain. """
        raise NotImplementedError

    @property
    def is_sqf(f):
        """Returns ``True`` if ``f`` is a square-free polynomial. """
        raise NotImplementedError

    @property
    def is_monic(f):
        """Returns ``True`` if the leading coefficient of ``f`` is one. """
        raise NotImplementedError

    @property
    def is_primitive(f):
        """Returns ``True`` if the GCD of the coefficients of ``f`` is one. """
        raise NotImplementedError

    @property
    def is_linear(f):
        """Returns ``True`` if ``f`` is linear in all its variables. """
        raise NotImplementedError

    @property
    def is_quadratic(f):
        """Returns ``True`` if ``f`` is quadratic in all its variables. """
        raise NotImplementedError

    @property
    def is_monomial(f):
        """Returns ``True`` if ``f`` is zero or has only one term. """
        raise NotImplementedError

    @property
    def is_homogeneous(f):
        """Returns ``True`` if ``f`` is a homogeneous polynomial. """
        raise NotImplementedError

    @property
    def is_irreducible(f):
        """Returns ``True`` if ``f`` has no factors over its domain. """
        raise NotImplementedError

    @property
    def is_cyclotomic(f):
        """Returns ``True`` if ``f`` is a cyclotomic polynomial. """
        raise NotImplementedError

    def __abs__(f):
        return f.abs()

    def __neg__(f):
        return f.neg()

    def __add__(f, g):
        if isinstance(g, DMP):
            return f.add(g)
        else:
            try:
                return f.add_ground(g)
            except CoercionFailed:
                return NotImplemented

    def __radd__(f, g):
        return f.__add__(g)

    def __sub__(f, g):
        if isinstance(g, DMP):
            return f.sub(g)
        else:
            try:
                return f.sub_ground(g)
            except CoercionFailed:
                return NotImplemented

    def __rsub__(f, g):
        return (-f).__add__(g)

    def __mul__(f, g):
        if isinstance(g, DMP):
            return f.mul(g)
        else:
            try:
                return f.mul_ground(g)
            except CoercionFailed:
                return NotImplemented

    def __rmul__(f, g):
        return f.__mul__(g)

    def __truediv__(f, g):
        if isinstance(g, DMP):
            return f.exquo(g)
        else:
            try:
                return f.mul_ground(g)
            except CoercionFailed:
                return NotImplemented

    def __rtruediv__(f, g):
        if isinstance(g, DMP):
            return g.exquo(f)
        else:
            try:
                return f._one().mul_ground(g).exquo(f)
            except CoercionFailed:
                return NotImplemented

    def __pow__(f, n):
        return f.pow(n)

    def __divmod__(f, g):
        return f.div(g)

    def __mod__(f, g):
        return f.rem(g)

    def __floordiv__(f, g):
        if isinstance(g, DMP):
            return f.quo(g)
        else:
            try:
                return f.quo_ground(g)
            except TypeError:
                return NotImplemented

    def __eq__(f, g):
        if f is g:
            return True
        if not isinstance(g, DMP):
            return NotImplemented
        try:
            F, G = f.unify_DMP(g)
        except UnificationFailed:
            return False
        else:
            return F._strict_eq(G)

    def _strict_eq(f, g):
        raise NotImplementedError

    def eq(f, g, strict=False):
        if not strict:
            return f == g
        else:
            return f._strict_eq(g)

    def ne(f, g, strict=False):
        return not f.eq(g, strict=strict)

    def __lt__(f, g):
        F, G = f.unify_DMP(g)
        return F.to_list() < G.to_list()

    def __le__(f, g):
        F, G = f.unify_DMP(g)
        return F.to_list() <= G.to_list()

    def __gt__(f, g):
        F, G = f.unify_DMP(g)
        return F.to_list() > G.to_list()

    def __ge__(f, g):
        F, G = f.unify_DMP(g)
        return F.to_list() >= G.to_list()

    def __bool__(f):
        return not f.is_zero


class DMP_Python(DMP):
    """Dense Multivariate Polynomials over `K`. """

    __slots__ = ('_rep', 'dom', 'lev')

    @classmethod
    def _new(cls, rep, dom, lev):
        obj = object.__new__(cls)
        obj._rep = rep
        obj.lev = lev
        obj.dom = dom
        return obj

    def _strict_eq(f, g):
        if type(f) != type(g):
            return False
        return f.lev == g.lev and f.dom == g.dom and f._rep == g._rep

    def per(f, rep):
        """Create a DMP out of the given representation. """
        return f._new(rep, f.dom, f.lev)

    def ground_new(f, coeff):
        """Construct a new ground instance of ``f``. """
        return f._new(dmp_ground(coeff, f.lev), f.dom, f.lev)

    def _one(f):
        return f.one(f.lev, f.dom)

    def unify(f, g):
        """Unify representations of two multivariate polynomials. """
        # XXX: This function is not really used any more since there is
        # unify_DMP now.
        if not isinstance(g, DMP) or f.lev != g.lev:
            raise UnificationFailed("Cannot unify %s with %s" % (f, g))

        if f.dom == g.dom:
            return f.lev, f.dom, f.per, f._rep, g._rep
        else:
            lev, dom = f.lev, f.dom.unify(g.dom)

            F = dmp_convert(f._rep, lev, f.dom, dom)
            G = dmp_convert(g._rep, lev, g.dom, dom)

            def per(rep):
                return f._new(rep, dom, lev)

            return lev, dom, per, F, G

    def to_DUP_Flint(f):
        """Convert ``f`` to a Flint representation. """
        return DUP_Flint._new(f._rep, f.dom, f.lev)

    def to_list(f):
        """Convert ``f`` to a list representation with native coefficients. """
        return list(f._rep)

    def to_tuple(f):
        """Convert ``f`` to a tuple representation with native coefficients. """
        return dmp_to_tuple(f._rep, f.lev)

    def _convert(f, dom):
        """Convert the ground domain of ``f``. """
        return f._new(dmp_convert(f._rep, f.lev, f.dom, dom), dom, f.lev)

    def _slice(f, m, n):
        """Take a continuous subsequence of terms of ``f``. """
        rep = dup_slice(f._rep, m, n, f.dom)
        return f._new(rep, f.dom, f.lev)

    def _slice_lev(f, m, n, j):
        """Take a continuous subsequence of terms of ``f``. """
        rep = dmp_slice_in(f._rep, m, n, j, f.lev, f.dom)
        return f._new(rep, f.dom, f.lev)

    def _terms(f, order=None):
        """Returns all non-zero terms from ``f`` in lex order. """
        return dmp_list_terms(f._rep, f.lev, f.dom, order=order)

    def _lift(f):
        """Convert algebraic coefficients to rationals. """
        r = dmp_lift(f._rep, f.lev, f.dom)
        return f._new(r, f.dom.dom, f.lev)

    def deflate(f):
        """Reduce degree of `f` by mapping `x_i^m` to `y_i`. """
        J, F = dmp_deflate(f._rep, f.lev, f.dom)
        return J, f.per(F)

    def inject(f, front=False):
        """Inject ground domain generators into ``f``. """
        F, lev = dmp_inject(f._rep, f.lev, f.dom, front=front)
        # XXX: domain and level changed here
        return f._new(F, f.dom.dom, lev)

    def eject(f, dom, front=False):
        """Eject selected generators into the ground domain. """
        F = dmp_eject(f._rep, f.lev, dom, front=front)
        # XXX: domain and level changed here
        return f._new(F, dom, f.lev - len(dom.symbols))

    def _exclude(f):
        """Remove useless generators from ``f``. """
        J, F, u = dmp_exclude(f._rep, f.lev, f.dom)
        # XXX: level changed here
        return J, f._new(F, f.dom, u)

    def _permute(f, P):
        """Returns a polynomial in `K[x_{P(1)}, ..., x_{P(n)}]`. """
        return f.per(dmp_permute(f._rep, P, f.lev, f.dom))

    def terms_gcd(f):
        """Remove GCD of terms from the polynomial ``f``. """
        J, F = dmp_terms_gcd(f._rep, f.lev, f.dom)
        return J, f.per(F)

    def _add_ground(f, c):
        """Add an element of the ground domain to ``f``. """
        return f.per(dmp_add_ground(f._rep, c, f.lev, f.dom))

    def _sub_ground(f, c):
        """Subtract an element of the ground domain from ``f``. """
        return f.per(dmp_sub_ground(f._rep, c, f.lev, f.dom))

    def _mul_ground(f, c):
        """Multiply ``f`` by a an element of the ground domain. """
        return f.per(dmp_mul_ground(f._rep, c, f.lev, f.dom))

    def _quo_ground(f, c):
        """Quotient of ``f`` by a an element of the ground domain. """
        return f.per(dmp_quo_ground(f._rep, c, f.lev, f.dom))

    def _exquo_ground(f, c):
        """Exact quotient of ``f`` by a an element of the ground domain. """
        return f.per(dmp_exquo_ground(f._rep, c, f.lev, f.dom))

    def abs(f):
        """Make all coefficients in ``f`` positive. """
        return f.per(dmp_abs(f._rep, f.lev, f.dom))

    def neg(f):
        """Negate all coefficients in ``f``. """
        return f.per(dmp_neg(f._rep, f.lev, f.dom))

    def _add(f, g):
        """Add two multivariate polynomials ``f`` and ``g``. """
        return f.per(dmp_add(f._rep, g._rep, f.lev, f.dom))

    def _sub(f, g):
        """Subtract two multivariate polynomials ``f`` and ``g``. """
        return f.per(dmp_sub(f._rep, g._rep, f.lev, f.dom))

    def _mul(f, g):
        """Multiply two multivariate polynomials ``f`` and ``g``. """
        return f.per(dmp_mul(f._rep, g._rep, f.lev, f.dom))

    def sqr(f):
        """Square a multivariate polynomial ``f``. """
        return f.per(dmp_sqr(f._rep, f.lev, f.dom))

    def _pow(f, n):
        """Raise ``f`` to a non-negative power ``n``. """
        return f.per(dmp_pow(f._rep, n, f.lev, f.dom))

    def _pdiv(f, g):
        """Polynomial pseudo-division of ``f`` and ``g``. """
        q, r = dmp_pdiv(f._rep, g._rep, f.lev, f.dom)
        return f.per(q), f.per(r)

    def _prem(f, g):
        """Polynomial pseudo-remainder of ``f`` and ``g``. """
        return f.per(dmp_prem(f._rep, g._rep, f.lev, f.dom))

    def _pquo(f, g):
        """Polynomial pseudo-quotient of ``f`` and ``g``. """
        return f.per(dmp_pquo(f._rep, g._rep, f.lev, f.dom))

    def _pexquo(f, g):
        """Polynomial exact pseudo-quotient of ``f`` and ``g``. """
        return f.per(dmp_pexquo(f._rep, g._rep, f.lev, f.dom))

    def _div(f, g):
        """Polynomial division with remainder of ``f`` and ``g``. """
        q, r = dmp_div(f._rep, g._rep, f.lev, f.dom)
        return f.per(q), f.per(r)

    def _rem(f, g):
        """Computes polynomial remainder of ``f`` and ``g``. """
        return f.per(dmp_rem(f._rep, g._rep, f.lev, f.dom))

    def _quo(f, g):
        """Computes polynomial quotient of ``f`` and ``g``. """
        return f.per(dmp_quo(f._rep, g._rep, f.lev, f.dom))

    def _exquo(f, g):
        """Computes polynomial exact quotient of ``f`` and ``g``. """
        return f.per(dmp_exquo(f._rep, g._rep, f.lev, f.dom))

    def _degree(f, j=0):
        """Returns the leading degree of ``f`` in ``x_j``. """
        return dmp_degree_in(f._rep, j, f.lev)

    def degree_list(f):
        """Returns a list of degrees of ``f``. """
        return dmp_degree_list(f._rep, f.lev)

    def total_degree(f):
        """Returns the total degree of ``f``. """
        return max(sum(m) for m in f.monoms())

    def LC(f):
        """Returns the leading coefficient of ``f``. """
        return dmp_ground_LC(f._rep, f.lev, f.dom)

    def TC(f):
        """Returns the trailing coefficient of ``f``. """
        return dmp_ground_TC(f._rep, f.lev, f.dom)

    def _nth(f, N):
        """Returns the ``n``-th coefficient of ``f``. """
        return dmp_ground_nth(f._rep, N, f.lev, f.dom)

    def max_norm(f):
        """Returns maximum norm of ``f``. """
        return dmp_max_norm(f._rep, f.lev, f.dom)

    def l1_norm(f):
        """Returns l1 norm of ``f``. """
        return dmp_l1_norm(f._rep, f.lev, f.dom)

    def l2_norm_squared(f):
        """Return squared l2 norm of ``f``. """
        return dmp_l2_norm_squared(f._rep, f.lev, f.dom)

    def clear_denoms(f):
        """Clear denominators, but keep the ground domain. """
        coeff, F = dmp_clear_denoms(f._rep, f.lev, f.dom)
        return coeff, f.per(F)

    def _integrate(f, m=1, j=0):
        """Computes the ``m``-th order indefinite integral of ``f`` in ``x_j``. """
        return f.per(dmp_integrate_in(f._rep, m, j, f.lev, f.dom))

    def _diff(f, m=1, j=0):
        """Computes the ``m``-th order derivative of ``f`` in ``x_j``. """
        return f.per(dmp_diff_in(f._rep, m, j, f.lev, f.dom))

    def _eval(f, a):
        return dmp_eval_in(f._rep, f.dom.convert(a), 0, f.lev, f.dom)

    def _eval_lev(f, a, j):
        rep = dmp_eval_in(f._rep, f.dom.convert(a), j, f.lev, f.dom)
        return f.new(rep, f.dom, f.lev - 1)

    def _half_gcdex(f, g):
        """Half extended Euclidean algorithm, if univariate. """
        s, h = dup_half_gcdex(f._rep, g._rep, f.dom)
        return f.per(s), f.per(h)

    def _gcdex(f, g):
        """Extended Euclidean algorithm, if univariate. """
        s, t, h = dup_gcdex(f._rep, g._rep, f.dom)
        return f.per(s), f.per(t), f.per(h)

    def _invert(f, g):
        """Invert ``f`` modulo ``g``, if possible. """
        s = dup_invert(f._rep, g._rep, f.dom)
        return f.per(s)

    def _revert(f, n):
        """Compute ``f**(-1)`` mod ``x**n``. """
        return f.per(dup_revert(f._rep, n, f.dom))

    def _subresultants(f, g):
        """Computes subresultant PRS sequence of ``f`` and ``g``. """
        R = dmp_subresultants(f._rep, g._rep, f.lev, f.dom)
        return list(map(f.per, R))

    def _resultant_includePRS(f, g):
        """Computes resultant of ``f`` and ``g`` via PRS. """
        res, R = dmp_resultant(f._rep, g._rep, f.lev, f.dom, includePRS=True)
        if f.lev:
            res = f.new(res, f.dom, f.lev - 1)
        return res, list(map(f.per, R))

    def _resultant(f, g):
        res = dmp_resultant(f._rep, g._rep, f.lev, f.dom)
        if f.lev:
            res = f.new(res, f.dom, f.lev - 1)
        return res

    def discriminant(f):
        """Computes discriminant of ``f``. """
        res = dmp_discriminant(f._rep, f.lev, f.dom)
        if f.lev:
            res = f.new(res, f.dom, f.lev - 1)
        return res

    def _cofactors(f, g):
        """Returns GCD of ``f`` and ``g`` and their cofactors. """
        h, cff, cfg = dmp_inner_gcd(f._rep, g._rep, f.lev, f.dom)
        return f.per(h), f.per(cff), f.per(cfg)

    def _gcd(f, g):
        """Returns polynomial GCD of ``f`` and ``g``. """
        return f.per(dmp_gcd(f._rep, g._rep, f.lev, f.dom))

    def _lcm(f, g):
        """Returns polynomial LCM of ``f`` and ``g``. """
        return f.per(dmp_lcm(f._rep, g._rep, f.lev, f.dom))

    def _cancel(f, g):
        """Cancel common factors in a rational function ``f/g``. """
        cF, cG, F, G = dmp_cancel(f._rep, g._rep, f.lev, f.dom, include=False)
        return cF, cG, f.per(F), f.per(G)

    def _cancel_include(f, g):
        """Cancel common factors in a rational function ``f/g``. """
        F, G = dmp_cancel(f._rep, g._rep, f.lev, f.dom, include=True)
        return f.per(F), f.per(G)

    def _trunc(f, p):
        """Reduce ``f`` modulo a constant ``p``. """
        return f.per(dmp_ground_trunc(f._rep, p, f.lev, f.dom))

    def monic(f):
        """Divides all coefficients by ``LC(f)``. """
        return f.per(dmp_ground_monic(f._rep, f.lev, f.dom))

    def content(f):
        """Returns GCD of polynomial coefficients. """
        return dmp_ground_content(f._rep, f.lev, f.dom)

    def primitive(f):
        """Returns content and a primitive form of ``f``. """
        cont, F = dmp_ground_primitive(f._rep, f.lev, f.dom)
        return cont, f.per(F)

    def _compose(f, g):
        """Computes functional composition of ``f`` and ``g``. """
        return f.per(dmp_compose(f._rep, g._rep, f.lev, f.dom))

    def _decompose(f):
        """Computes functional decomposition of ``f``. """
        return list(map(f.per, dup_decompose(f._rep, f.dom)))

    def _shift(f, a):
        """Efficiently compute Taylor shift ``f(x + a)``. """
        return f.per(dup_shift(f._rep, a, f.dom))

    def _shift_list(f, a):
        """Efficiently compute Taylor shift ``f(X + A)``. """
        return f.per(dmp_shift(f._rep, a, f.lev, f.dom))

    def _transform(f, p, q):
        """Evaluate functional transformation ``q**n * f(p/q)``."""
        return f.per(dup_transform(f._rep, p._rep, q._rep, f.dom))

    def _sturm(f):
        """Computes the Sturm sequence of ``f``. """
        return list(map(f.per, dup_sturm(f._rep, f.dom)))

    def _cauchy_upper_bound(f):
        """Computes the Cauchy upper bound on the roots of ``f``. """
        return dup_cauchy_upper_bound(f._rep, f.dom)

    def _cauchy_lower_bound(f):
        """Computes the Cauchy lower bound on the nonzero roots of ``f``. """
        return dup_cauchy_lower_bound(f._rep, f.dom)

    def _mignotte_sep_bound_squared(f):
        """Computes the squared Mignotte bound on root separations of ``f``. """
        return dup_mignotte_sep_bound_squared(f._rep, f.dom)

    def _gff_list(f):
        """Computes greatest factorial factorization of ``f``. """
        return [ (f.per(g), k) for g, k in dup_gff_list(f._rep, f.dom) ]

    def norm(f):
        """Computes ``Norm(f)``."""
        r = dmp_norm(f._rep, f.lev, f.dom)
        return f.new(r, f.dom.dom, f.lev)

    def sqf_norm(f):
        """Computes square-free norm of ``f``. """
        s, g, r = dmp_sqf_norm(f._rep, f.lev, f.dom)
        return s, f.per(g), f.new(r, f.dom.dom, f.lev)

    def sqf_part(f):
        """Computes square-free part of ``f``. """
        return f.per(dmp_sqf_part(f._rep, f.lev, f.dom))

    def sqf_list(f, all=False):
        """Returns a list of square-free factors of ``f``. """
        coeff, factors = dmp_sqf_list(f._rep, f.lev, f.dom, all)
        return coeff, [ (f.per(g), k) for g, k in factors ]

    def sqf_list_include(f, all=False):
        """Returns a list of square-free factors of ``f``. """
        factors = dmp_sqf_list_include(f._rep, f.lev, f.dom, all)
        return [ (f.per(g), k) for g, k in factors ]

    def factor_list(f):
        """Returns a list of irreducible factors of ``f``. """
        coeff, factors = dmp_factor_list(f._rep, f.lev, f.dom)
        return coeff, [ (f.per(g), k) for g, k in factors ]

    def factor_list_include(f):
        """Returns a list of irreducible factors of ``f``. """
        factors = dmp_factor_list_include(f._rep, f.lev, f.dom)
        return [ (f.per(g), k) for g, k in factors ]

    def _isolate_real_roots(f, eps, inf, sup, fast):
        return dup_isolate_real_roots(f._rep, f.dom, eps=eps, inf=inf, sup=sup, fast=fast)

    def _isolate_real_roots_sqf(f, eps, inf, sup, fast):
        return dup_isolate_real_roots_sqf(f._rep, f.dom, eps=eps, inf=inf, sup=sup, fast=fast)

    def _isolate_all_roots(f, eps, inf, sup, fast):
        return dup_isolate_all_roots(f._rep, f.dom, eps=eps, inf=inf, sup=sup, fast=fast)

    def _isolate_all_roots_sqf(f, eps, inf, sup, fast):
        return dup_isolate_all_roots_sqf(f._rep, f.dom, eps=eps, inf=inf, sup=sup, fast=fast)

    def _refine_real_root(f, s, t, eps, steps, fast):
        return dup_refine_real_root(f._rep, s, t, f.dom, eps=eps, steps=steps, fast=fast)

    def count_real_roots(f, inf=None, sup=None):
        """Return the number of real roots of ``f`` in ``[inf, sup]``. """
        return dup_count_real_roots(f._rep, f.dom, inf=inf, sup=sup)

    def count_complex_roots(f, inf=None, sup=None):
        """Return the number of complex roots of ``f`` in ``[inf, sup]``. """
        return dup_count_complex_roots(f._rep, f.dom, inf=inf, sup=sup)

    @property
    def is_zero(f):
        """Returns ``True`` if ``f`` is a zero polynomial. """
        return dmp_zero_p(f._rep, f.lev)

    @property
    def is_one(f):
        """Returns ``True`` if ``f`` is a unit polynomial. """
        return dmp_one_p(f._rep, f.lev, f.dom)

    @property
    def is_ground(f):
        """Returns ``True`` if ``f`` is an element of the ground domain. """
        return dmp_ground_p(f._rep, None, f.lev)

    @property
    def is_sqf(f):
        """Returns ``True`` if ``f`` is a square-free polynomial. """
        return dmp_sqf_p(f._rep, f.lev, f.dom)

    @property
    def is_monic(f):
        """Returns ``True`` if the leading coefficient of ``f`` is one. """
        return f.dom.is_one(dmp_ground_LC(f._rep, f.lev, f.dom))

    @property
    def is_primitive(f):
        """Returns ``True`` if the GCD of the coefficients of ``f`` is one. """
        return f.dom.is_one(dmp_ground_content(f._rep, f.lev, f.dom))

    @property
    def is_linear(f):
        """Returns ``True`` if ``f`` is linear in all its variables. """
        return all(sum(monom) <= 1 for monom in dmp_to_dict(f._rep, f.lev, f.dom).keys())

    @property
    def is_quadratic(f):
        """Returns ``True`` if ``f`` is quadratic in all its variables. """
        return all(sum(monom) <= 2 for monom in dmp_to_dict(f._rep, f.lev, f.dom).keys())

    @property
    def is_monomial(f):
        """Returns ``True`` if ``f`` is zero or has only one term. """
        return len(f.to_dict()) <= 1

    @property
    def is_homogeneous(f):
        """Returns ``True`` if ``f`` is a homogeneous polynomial. """
        return f.homogeneous_order() is not None

    @property
    def is_irreducible(f):
        """Returns ``True`` if ``f`` has no factors over its domain. """
        return dmp_irreducible_p(f._rep, f.lev, f.dom)

    @property
    def is_cyclotomic(f):
        """Returns ``True`` if ``f`` is a cyclotomic polynomial. """
        if not f.lev:
            return dup_cyclotomic_p(f._rep, f.dom)
        else:
            return False


class DUP_Flint(DMP):
    """Dense Multivariate Polynomials over `K`. """

    lev = 0

    __slots__ = ('_rep', 'dom', '_cls')

    def __reduce__(self):
        return self.__class__, (self.to_list(), self.dom, self.lev)

    @classmethod
    def _new(cls, rep, dom, lev):
        rep = cls._flint_poly(rep[::-1], dom, lev)
        return cls.from_rep(rep, dom)

    def to_list(f):
        """Convert ``f`` to a list representation with native coefficients. """
        return f._rep.coeffs()[::-1]

    @classmethod
    def _flint_poly(cls, rep, dom, lev):
        assert _supported_flint_domain(dom)
        assert lev == 0
        flint_cls = cls._get_flint_poly_cls(dom)
        return flint_cls(rep)

    @classmethod
    def _get_flint_poly_cls(cls, dom):
        if dom.is_ZZ:
            return flint.fmpz_poly
        elif dom.is_QQ:
            return flint.fmpq_poly
        elif dom.is_FF:
            return dom._poly_ctx
        else:
            raise RuntimeError("Domain %s is not supported with flint" % dom)

    @classmethod
    def from_rep(cls, rep, dom):
        """Create a DMP from the given representation. """

        if dom.is_ZZ:
            assert isinstance(rep, flint.fmpz_poly)
            _cls = flint.fmpz_poly
        elif dom.is_QQ:
            assert isinstance(rep, flint.fmpq_poly)
            _cls = flint.fmpq_poly
        elif dom.is_FF:
            assert isinstance(rep, (flint.nmod_poly, flint.fmpz_mod_poly))
            c = dom.characteristic()
            __cls = type(rep)
            _cls = lambda e: __cls(e, c)
        else:
            raise RuntimeError("Domain %s is not supported with flint" % dom)

        obj = object.__new__(cls)
        obj.dom = dom
        obj._rep = rep
        obj._cls = _cls

        return obj

    def _strict_eq(f, g):
        if type(f) != type(g):
            return False
        return f.dom == g.dom and f._rep == g._rep

    def ground_new(f, coeff):
        """Construct a new ground instance of ``f``. """
        return f.from_rep(f._cls([coeff]), f.dom)

    def _one(f):
        return f.ground_new(f.dom.one)

    def unify(f, g):
        """Unify representations of two polynomials. """
        raise RuntimeError

    def to_DMP_Python(f):
        """Convert ``f`` to a Python native representation. """
        return DMP_Python._new(f.to_list(), f.dom, f.lev)

    def to_tuple(f):
        """Convert ``f`` to a tuple representation with native coefficients. """
        return tuple(f.to_list())

    def _convert(f, dom):
        """Convert the ground domain of ``f``. """
        if dom == QQ and f.dom == ZZ:
            return f.from_rep(flint.fmpq_poly(f._rep), dom)
        elif _supported_flint_domain(dom) and _supported_flint_domain(f.dom):
            # XXX: python-flint should provide a faster way to do this.
            return f.to_DMP_Python()._convert(dom).to_DUP_Flint()
        else:
            raise RuntimeError(f"DUP_Flint: Cannot convert {f.dom} to {dom}")

    def _slice(f, m, n):
        """Take a continuous subsequence of terms of ``f``. """
        coeffs = f._rep.coeffs()[m:n]
        return f.from_rep(f._cls(coeffs), f.dom)

    def _slice_lev(f, m, n, j):
        """Take a continuous subsequence of terms of ``f``. """
        # Only makes sense for multivariate polynomials
        raise NotImplementedError

    def _terms(f, order=None):
        """Returns all non-zero terms from ``f`` in lex order. """
        if order is None or order.alias == 'lex':
            terms = [ ((n,), c) for n, c in enumerate(f._rep.coeffs()) if c ]
            return terms[::-1]
        else:
            # XXX: InverseOrder (ilex) comes here. We could handle that case
            # efficiently by reversing the coefficients but it is not clear
            # how to test if the order is InverseOrder.
            #
            # Otherwise why would the order ever be different for univariate
            # polynomials?
            return f.to_DMP_Python()._terms(order=order)

    def _lift(f):
        """Convert algebraic coefficients to rationals. """
        # This is for algebraic number fields which DUP_Flint does not support
        raise NotImplementedError

    def deflate(f):
        """Reduce degree of `f` by mapping `x_i^m` to `y_i`. """
        # XXX: Check because otherwise this segfaults with python-flint:
        #
        #  >>> flint.fmpz_poly([]).deflation()
        #  Exception (fmpz_poly_deflate). Division by zero.
        #  Aborted (core dumped
        #
        if f.is_zero:
            return (1,), f
        g, n = f._rep.deflation()
        return (n,), f.from_rep(g, f.dom)

    def inject(f, front=False):
        """Inject ground domain generators into ``f``. """
        # Ground domain would need to be a poly ring
        raise NotImplementedError

    def eject(f, dom, front=False):
        """Eject selected generators into the ground domain. """
        # Only makes sense for multivariate polynomials
        raise NotImplementedError

    def _exclude(f):
        """Remove useless generators from ``f``. """
        # Only makes sense for multivariate polynomials
        raise NotImplementedError

    def _permute(f, P):
        """Returns a polynomial in `K[x_{P(1)}, ..., x_{P(n)}]`. """
        # Only makes sense for multivariate polynomials
        raise NotImplementedError

    def terms_gcd(f):
        """Remove GCD of terms from the polynomial ``f``. """
        # XXX: python-flint should have primitive, content, etc methods.
        J, F = f.to_DMP_Python().terms_gcd()
        return J, F.to_DUP_Flint()

    def _add_ground(f, c):
        """Add an element of the ground domain to ``f``. """
        return f.from_rep(f._rep + c, f.dom)

    def _sub_ground(f, c):
        """Subtract an element of the ground domain from ``f``. """
        return f.from_rep(f._rep - c, f.dom)

    def _mul_ground(f, c):
        """Multiply ``f`` by a an element of the ground domain. """
        return f.from_rep(f._rep * c, f.dom)

    def _quo_ground(f, c):
        """Quotient of ``f`` by a an element of the ground domain. """
        return f.from_rep(f._rep // c, f.dom)

    def _exquo_ground(f, c):
        """Exact quotient of ``f`` by an element of the ground domain. """
        q, r = divmod(f._rep, c)
        if r:
            raise ExactQuotientFailed(f, c)
        return f.from_rep(q, f.dom)

    def abs(f):
        """Make all coefficients in ``f`` positive. """
        return f.to_DMP_Python().abs().to_DUP_Flint()

    def neg(f):
        """Negate all coefficients in ``f``. """
        return f.from_rep(-f._rep, f.dom)

    def _add(f, g):
        """Add two multivariate polynomials ``f`` and ``g``. """
        return f.from_rep(f._rep + g._rep, f.dom)

    def _sub(f, g):
        """Subtract two multivariate polynomials ``f`` and ``g``. """
        return f.from_rep(f._rep - g._rep, f.dom)

    def _mul(f, g):
        """Multiply two multivariate polynomials ``f`` and ``g``. """
        return f.from_rep(f._rep * g._rep, f.dom)

    def sqr(f):
        """Square a multivariate polynomial ``f``. """
        return f.from_rep(f._rep ** 2, f.dom)

    def _pow(f, n):
        """Raise ``f`` to a non-negative power ``n``. """
        return f.from_rep(f._rep ** n, f.dom)

    def _pdiv(f, g):
        """Polynomial pseudo-division of ``f`` and ``g``. """
        d = f.degree() - g.degree() + 1
        q, r = divmod(g.LC()**d * f._rep, g._rep)
        return f.from_rep(q, f.dom), f.from_rep(r, f.dom)

    def _prem(f, g):
        """Polynomial pseudo-remainder of ``f`` and ``g``. """
        d = f.degree() - g.degree() + 1
        q = (g.LC()**d * f._rep) % g._rep
        return f.from_rep(q, f.dom)

    def _pquo(f, g):
        """Polynomial pseudo-quotient of ``f`` and ``g``. """
        d = f.degree() - g.degree() + 1
        r = (g.LC()**d * f._rep) // g._rep
        return f.from_rep(r, f.dom)

    def _pexquo(f, g):
        """Polynomial exact pseudo-quotient of ``f`` and ``g``. """
        d = f.degree() - g.degree() + 1
        q, r = divmod(g.LC()**d * f._rep, g._rep)
        if r:
            raise ExactQuotientFailed(f, g)
        return f.from_rep(q, f.dom)

    def _div(f, g):
        """Polynomial division with remainder of ``f`` and ``g``. """
        if f.dom.is_Field:
            q, r = divmod(f._rep, g._rep)
            return f.from_rep(q, f.dom), f.from_rep(r, f.dom)
        else:
            # XXX: python-flint defines division in ZZ[x] differently
            q, r = f.to_DMP_Python()._div(g.to_DMP_Python())
            return q.to_DUP_Flint(), r.to_DUP_Flint()

    def _rem(f, g):
        """Computes polynomial remainder of ``f`` and ``g``. """
        return f.from_rep(f._rep % g._rep, f.dom)

    def _quo(f, g):
        """Computes polynomial quotient of ``f`` and ``g``. """
        return f.from_rep(f._rep // g._rep, f.dom)

    def _exquo(f, g):
        """Computes polynomial exact quotient of ``f`` and ``g``. """
        q, r = f._div(g)
        if r:
            raise ExactQuotientFailed(f, g)
        return q

    def _degree(f, j=0):
        """Returns the leading degree of ``f`` in ``x_j``. """
        d = f._rep.degree()
        if d == -1:
            d = ninf
        return d

    def degree_list(f):
        """Returns a list of degrees of ``f``. """
        return ( f._degree() ,)

    def total_degree(f):
        """Returns the total degree of ``f``. """
        return f._degree()

    def LC(f):
        """Returns the leading coefficient of ``f``. """
        return f._rep[f._rep.degree()]

    def TC(f):
        """Returns the trailing coefficient of ``f``. """
        return f._rep[0]

    def _nth(f, N):
        """Returns the ``n``-th coefficient of ``f``. """
        [n] = N
        return f._rep[n]

    def max_norm(f):
        """Returns maximum norm of ``f``. """
        return f.to_DMP_Python().max_norm()

    def l1_norm(f):
        """Returns l1 norm of ``f``. """
        return f.to_DMP_Python().l1_norm()

    def l2_norm_squared(f):
        """Return squared l2 norm of ``f``. """
        return f.to_DMP_Python().l2_norm_squared()

    def clear_denoms(f):
        """Clear denominators, but keep the ground domain. """
        R = f.dom
        if R.is_QQ:
            denom = f._rep.denom()
            numer = f.from_rep(f._cls(f._rep.numer()), f.dom)
            return denom, numer
        elif R.is_ZZ or R.is_FiniteField:
            return R.one, f
        else:
            raise NotImplementedError

    def _integrate(f, m=1, j=0):
        """Computes the ``m``-th order indefinite integral of ``f`` in ``x_j``. """
        assert j == 0
        if f.dom.is_Field:
            rep = f._rep
            for i in range(m):
                rep = rep.integral()
            return f.from_rep(rep, f.dom)
        else:
            return f.to_DMP_Python()._integrate(m=m, j=j).to_DUP_Flint()

    def _diff(f, m=1, j=0):
        """Computes the ``m``-th order derivative of ``f``. """
        assert j == 0
        rep = f._rep
        for i in range(m):
            rep = rep.derivative()
        return f.from_rep(rep, f.dom)

    def _eval(f, a):
        # XXX: This method is called with many different input types. Ideally
        # we could use e.g. fmpz_poly.__call__ here but more thought needs to
        # go into which types this is supposed to be called with and what types
        # it should return.
        return f.to_DMP_Python()._eval(a)

    def _eval_lev(f, a, j):
        # Only makes sense for multivariate polynomials
        raise NotImplementedError

    def _half_gcdex(f, g):
        """Half extended Euclidean algorithm. """
        s, h = f.to_DMP_Python()._half_gcdex(g.to_DMP_Python())
        return s.to_DUP_Flint(), h.to_DUP_Flint()

    def _gcdex(f, g):
        """Extended Euclidean algorithm. """
        h, s, t = f._rep.xgcd(g._rep)
        return f.from_rep(s, f.dom), f.from_rep(t, f.dom), f.from_rep(h, f.dom)

    def _invert(f, g):
        """Invert ``f`` modulo ``g``, if possible. """
        R = f.dom
        if R.is_Field:
            gcd, F_inv, _ = f._rep.xgcd(g._rep)
            # XXX: Should be gcd != 1 but nmod_poly does not compare equal to
            # other types.
            if gcd != 0*gcd + 1:
                raise NotInvertible("zero divisor")
            return f.from_rep(F_inv, R)
        else:
            # fmpz_poly does not have xgcd or invert and this is not well
            # defined in general.
            return f.to_DMP_Python()._invert(g.to_DMP_Python()).to_DUP_Flint()

    def _revert(f, n):
        """Compute ``f**(-1)`` mod ``x**n``. """
        # XXX: Use fmpz_series etc for reversion?
        # Maybe python-flint should provide revert for fmpz_poly...
        return f.to_DMP_Python()._revert(n).to_DUP_Flint()

    def _subresultants(f, g):
        """Computes subresultant PRS sequence of ``f`` and ``g``. """
        # XXX: Maybe _fmpz_poly_pseudo_rem_cohen could be used...
        R = f.to_DMP_Python()._subresultants(g.to_DMP_Python())
        return [ g.to_DUP_Flint() for g in R ]

    def _resultant_includePRS(f, g):
        """Computes resultant of ``f`` and ``g`` via PRS. """
        # XXX: Maybe _fmpz_poly_pseudo_rem_cohen could be used...
        res, R = f.to_DMP_Python()._resultant_includePRS(g.to_DMP_Python())
        return res, [ g.to_DUP_Flint() for g in R ]

    def _resultant(f, g):
        """Computes resultant of ``f`` and ``g``. """
        # XXX: Use fmpz_mpoly etc when possible...
        return f.to_DMP_Python()._resultant(g.to_DMP_Python())

    def discriminant(f):
        """Computes discriminant of ``f``. """
        # XXX: Use fmpz_mpoly etc when possible...
        return f.to_DMP_Python().discriminant()

    def _cofactors(f, g):
        """Returns GCD of ``f`` and ``g`` and their cofactors. """
        h = f.gcd(g)
        return h, f.exquo(h), g.exquo(h)

    def _gcd(f, g):
        """Returns polynomial GCD of ``f`` and ``g``. """
        return f.from_rep(f._rep.gcd(g._rep), f.dom)

    def _lcm(f, g):
        """Returns polynomial LCM of ``f`` and ``g``. """
        # XXX: python-flint should have a lcm method
        if not (f and g):
            return f.ground_new(f.dom.zero)

        l = f._mul(g)._exquo(f._gcd(g))

        if l.dom.is_Field:
            l = l.monic()
        elif l.LC() < 0:
            l = l.neg()

        return l

    def _cancel(f, g):
        """Cancel common factors in a rational function ``f/g``. """
        assert f.dom == g.dom
        R = f.dom

        # Think carefully about how to handle denominators and coefficient
        # canonicalisation if more domains are permitted...
        assert R.is_ZZ or R.is_QQ or R.is_FiniteField

        if R.is_FiniteField:
            h = f._gcd(g)
            F, G = f.exquo(h), g.exquo(h)
            return R.one, R.one, F, G

        if R.is_QQ:
            cG, F = f.clear_denoms()
            cF, G = g.clear_denoms()
        else:
            cG, F = R.one, f
            cF, G = R.one, g

        cH = cF.gcd(cG)
        cF, cG = cF // cH, cG // cH

        H = F._gcd(G)
        F, G = F.exquo(H), G.exquo(H)

        f_neg = F.LC() < 0
        g_neg = G.LC() < 0

        if f_neg and g_neg:
            F, G = F.neg(), G.neg()
        elif f_neg:
            cF, F = -cF, F.neg()
        elif g_neg:
            cF, G = -cF, G.neg()

        return cF, cG, F, G

    def _cancel_include(f, g):
        """Cancel common factors in a rational function ``f/g``. """
        cF, cG, F, G = f._cancel(g)
        return F._mul_ground(cF), G._mul_ground(cG)

    def _trunc(f, p):
        """Reduce ``f`` modulo a constant ``p``. """
        return f.to_DMP_Python()._trunc(p).to_DUP_Flint()

    def monic(f):
        """Divides all coefficients by ``LC(f)``. """
        # XXX: python-flint should add monic
        return f._exquo_ground(f.LC())

    def content(f):
        """Returns GCD of polynomial coefficients. """
        # XXX: python-flint should have a content method
        return f.to_DMP_Python().content()

    def primitive(f):
        """Returns content and a primitive form of ``f``. """
        cont = f.content()
        if f.is_zero:
            return f.dom.zero, f
        prim = f._exquo_ground(cont)
        return cont, prim

    def _compose(f, g):
        """Computes functional composition of ``f`` and ``g``. """
        return f.from_rep(f._rep(g._rep), f.dom)

    def _decompose(f):
        """Computes functional decomposition of ``f``. """
        return [ g.to_DUP_Flint() for g in f.to_DMP_Python()._decompose() ]

    def _shift(f, a):
        """Efficiently compute Taylor shift ``f(x + a)``. """
        x_plus_a = f._cls([a, f.dom.one])
        return f.from_rep(f._rep(x_plus_a), f.dom)

    def _transform(f, p, q):
        """Evaluate functional transformation ``q**n * f(p/q)``."""
        F, P, Q = f.to_DMP_Python(), p.to_DMP_Python(), q.to_DMP_Python()
        return F.transform(P, Q).to_DUP_Flint()

    def _sturm(f):
        """Computes the Sturm sequence of ``f``. """
        return [ g.to_DUP_Flint() for g in f.to_DMP_Python()._sturm() ]

    def _cauchy_upper_bound(f):
        """Computes the Cauchy upper bound on the roots of ``f``. """
        return f.to_DMP_Python()._cauchy_upper_bound()

    def _cauchy_lower_bound(f):
        """Computes the Cauchy lower bound on the nonzero roots of ``f``. """
        return f.to_DMP_Python()._cauchy_lower_bound()

    def _mignotte_sep_bound_squared(f):
        """Computes the squared Mignotte bound on root separations of ``f``. """
        return f.to_DMP_Python()._mignotte_sep_bound_squared()

    def _gff_list(f):
        """Computes greatest factorial factorization of ``f``. """
        F = f.to_DMP_Python()
        return [ (g.to_DUP_Flint(), k) for g, k in F.gff_list() ]

    def norm(f):
        """Computes ``Norm(f)``."""
        # This is for algebraic number fields which DUP_Flint does not support
        raise NotImplementedError

    def sqf_norm(f):
        """Computes square-free norm of ``f``. """
        # This is for algebraic number fields which DUP_Flint does not support
        raise NotImplementedError

    def sqf_part(f):
        """Computes square-free part of ``f``. """
        return f._exquo(f._gcd(f._diff()))

    def sqf_list(f, all=False):
        """Returns a list of square-free factors of ``f``. """
        # XXX: python-flint should provide square free factorisation.
        coeff, factors = f.to_DMP_Python().sqf_list(all=all)
        return coeff, [ (g.to_DUP_Flint(), k) for g, k in factors ]

    def sqf_list_include(f, all=False):
        """Returns a list of square-free factors of ``f``. """
        factors = f.to_DMP_Python().sqf_list_include(all=all)
        return [ (g.to_DUP_Flint(), k) for g, k in factors ]

    def factor_list(f):
        """Returns a list of irreducible factors of ``f``. """

        if f.dom.is_ZZ or f.dom.is_FF:
            # python-flint matches polys here
            coeff, factors = f._rep.factor()
            factors = [ (f.from_rep(g, f.dom), k) for g, k in factors ]

        elif f.dom.is_QQ:
            # python-flint returns monic factors over QQ whereas polys returns
            # denominator free factors.
            coeff, factors = f._rep.factor()
            factors_monic = [ (f.from_rep(g, f.dom), k) for g, k in factors ]

            # Absorb the denominators into coeff
            factors = []
            for g, k in factors_monic:
                d, g = g.clear_denoms()
                coeff /= d**k
                factors.append((g, k))

        else:
            # Check carefully when adding more domains here...
            raise RuntimeError("Domain %s is not supported with flint" % f.dom)

        # We need to match the way that polys orders the factors
        factors = f._sort_factors(factors)

        return coeff, factors

    def factor_list_include(f):
        """Returns a list of irreducible factors of ``f``. """
        # XXX: factor_list_include seems to be broken in general:
        #
        #   >>> Poly(2*(x - 1)**3, x).factor_list_include()
        #   [(Poly(2*x - 2, x, domain='ZZ'), 3)]
        #
        # Let's not try to implement it here.
        factors = f.to_DMP_Python().factor_list_include()
        return [ (g.to_DUP_Flint(), k) for g, k in factors ]

    def _sort_factors(f, factors):
        """Sort a list of factors to canonical order. """
        # Convert the factors to lists and use _sort_factors from polys
        factors = [ (g.to_list(), k) for g, k in factors ]
        factors = _sort_factors(factors, multiple=True)
        to_dup_flint = lambda g: f.from_rep(f._cls(g[::-1]), f.dom)
        return [ (to_dup_flint(g), k) for g, k in factors ]

    def _isolate_real_roots(f, eps, inf, sup, fast):
        return f.to_DMP_Python()._isolate_real_roots(eps, inf, sup, fast)

    def _isolate_real_roots_sqf(f, eps, inf, sup, fast):
        return f.to_DMP_Python()._isolate_real_roots_sqf(eps, inf, sup, fast)

    def _isolate_all_roots(f, eps, inf, sup, fast):
        # fmpz_poly and fmpq_poly have a complex_roots method that could be
        # used here. It probably makes more sense to add analogous methods in
        # python-flint though.
        return f.to_DMP_Python()._isolate_all_roots(eps, inf, sup, fast)

    def _isolate_all_roots_sqf(f, eps, inf, sup, fast):
        return f.to_DMP_Python()._isolate_all_roots_sqf(eps, inf, sup, fast)

    def _refine_real_root(f, s, t, eps, steps, fast):
        return f.to_DMP_Python()._refine_real_root(s, t, eps, steps, fast)

    def count_real_roots(f, inf=None, sup=None):
        """Return the number of real roots of ``f`` in ``[inf, sup]``. """
        return f.to_DMP_Python().count_real_roots(inf=inf, sup=sup)

    def count_complex_roots(f, inf=None, sup=None):
        """Return the number of complex roots of ``f`` in ``[inf, sup]``. """
        return f.to_DMP_Python().count_complex_roots(inf=inf, sup=sup)

    @property
    def is_zero(f):
        """Returns ``True`` if ``f`` is a zero polynomial. """
        return not f._rep

    @property
    def is_one(f):
        """Returns ``True`` if ``f`` is a unit polynomial. """
        return f._rep == f.dom.one

    @property
    def is_ground(f):
        """Returns ``True`` if ``f`` is an element of the ground domain. """
        return f._rep.degree() <= 0

    @property
    def is_linear(f):
        """Returns ``True`` if ``f`` is linear in all its variables. """
        return f._rep.degree() <= 1

    @property
    def is_quadratic(f):
        """Returns ``True`` if ``f`` is quadratic in all its variables. """
        return f._rep.degree() <= 2

    @property
    def is_monomial(f):
        """Returns ``True`` if ``f`` is zero or has only one term. """
        fr = f._rep
        return fr.degree() < 0 or not any(fr[n] for n in range(fr.degree()))

    @property
    def is_monic(f):
        """Returns ``True`` if the leading coefficient of ``f`` is one. """
        return f.LC() == f.dom.one

    @property
    def is_primitive(f):
        """Returns ``True`` if the GCD of the coefficients of ``f`` is one. """
        return f.to_DMP_Python().is_primitive

    @property
    def is_homogeneous(f):
        """Returns ``True`` if ``f`` is a homogeneous polynomial. """
        return f.to_DMP_Python().is_homogeneous

    @property
    def is_sqf(f):
        """Returns ``True`` if ``f`` is a square-free polynomial. """
        g = f._rep.gcd(f._rep.derivative())
        return g.degree() <= 0

    @property
    def is_irreducible(f):
        """Returns ``True`` if ``f`` has no factors over its domain. """
        _, factors = f._rep.factor()
        if len(factors) == 0:
            return True
        elif len(factors) == 1:
            return factors[0][1] == 1
        else:
            return False

    @property
    def is_cyclotomic(f):
        """Returns ``True`` if ``f`` is a cyclotomic polynomial. """
        if f.dom.is_QQ:
            try:
                f = f.convert(ZZ)
            except CoercionFailed:
                return False
        if f.dom.is_ZZ:
            return bool(f._rep.is_cyclotomic())
        else:
            # This is what dup_cyclotomic_p does...
            return False


def init_normal_DMF(num, den, lev, dom):
    return DMF(dmp_normal(num, lev, dom),
               dmp_normal(den, lev, dom), dom, lev)


class DMF(PicklableWithSlots, CantSympify):
    """Dense Multivariate Fractions over `K`. """

    __slots__ = ('num', 'den', 'lev', 'dom')

    def __init__(self, rep, dom, lev=None):
        num, den, lev = self._parse(rep, dom, lev)
        num, den = dmp_cancel(num, den, lev, dom)

        self.num = num
        self.den = den
        self.lev = lev
        self.dom = dom

    @classmethod
    def new(cls, rep, dom, lev=None):
        num, den, lev = cls._parse(rep, dom, lev)

        obj = object.__new__(cls)

        obj.num = num
        obj.den = den
        obj.lev = lev
        obj.dom = dom

        return obj

    def ground_new(self, rep):
        return self.new(rep, self.dom, self.lev)

    @classmethod
    def _parse(cls, rep, dom, lev=None):
        if isinstance(rep, tuple):
            num, den = rep

            if lev is not None:
                if isinstance(num, dict):
                    num = dmp_from_dict(num, lev, dom)

                if isinstance(den, dict):
                    den = dmp_from_dict(den, lev, dom)
            else:
                num, num_lev = dmp_validate(num)
                den, den_lev = dmp_validate(den)

                if num_lev == den_lev:
                    lev = num_lev
                else:
                    raise ValueError('inconsistent number of levels')

            if dmp_zero_p(den, lev):
                raise ZeroDivisionError('fraction denominator')

            if dmp_zero_p(num, lev):
                den = dmp_one(lev, dom)
            else:
                if dmp_negative_p(den, lev, dom):
                    num = dmp_neg(num, lev, dom)
                    den = dmp_neg(den, lev, dom)
        else:
            num = rep

            if lev is not None:
                if isinstance(num, dict):
                    num = dmp_from_dict(num, lev, dom)
                elif not isinstance(num, list):
                    num = dmp_ground(dom.convert(num), lev)
            else:
                num, lev = dmp_validate(num)

            den = dmp_one(lev, dom)

        return num, den, lev

    def __repr__(f):
        return "%s((%s, %s), %s)" % (f.__class__.__name__, f.num, f.den, f.dom)

    def __hash__(f):
        return hash((f.__class__.__name__, dmp_to_tuple(f.num, f.lev),
            dmp_to_tuple(f.den, f.lev), f.lev, f.dom))

    def poly_unify(f, g):
        """Unify a multivariate fraction and a polynomial. """
        if not isinstance(g, DMP) or f.lev != g.lev:
            raise UnificationFailed("Cannot unify %s with %s" % (f, g))

        if f.dom == g.dom:
            return (f.lev, f.dom, f.per, (f.num, f.den), g._rep)
        else:
            lev, dom = f.lev, f.dom.unify(g.dom)

            F = (dmp_convert(f.num, lev, f.dom, dom),
                 dmp_convert(f.den, lev, f.dom, dom))

            G = dmp_convert(g._rep, lev, g.dom, dom)

            def per(num, den, cancel=True, kill=False, lev=lev):
                if kill:
                    if not lev:
                        return num/den
                    else:
                        lev = lev - 1

                if cancel:
                    num, den = dmp_cancel(num, den, lev, dom)

                return f.__class__.new((num, den), dom, lev)

            return lev, dom, per, F, G

    def frac_unify(f, g):
        """Unify representations of two multivariate fractions. """
        if not isinstance(g, DMF) or f.lev != g.lev:
            raise UnificationFailed("Cannot unify %s with %s" % (f, g))

        if f.dom == g.dom:
            return (f.lev, f.dom, f.per, (f.num, f.den),
                                         (g.num, g.den))
        else:
            lev, dom = f.lev, f.dom.unify(g.dom)

            F = (dmp_convert(f.num, lev, f.dom, dom),
                 dmp_convert(f.den, lev, f.dom, dom))

            G = (dmp_convert(g.num, lev, g.dom, dom),
                 dmp_convert(g.den, lev, g.dom, dom))

            def per(num, den, cancel=True, kill=False, lev=lev):
                if kill:
                    if not lev:
                        return num/den
                    else:
                        lev = lev - 1

                if cancel:
                    num, den = dmp_cancel(num, den, lev, dom)

                return f.__class__.new((num, den), dom, lev)

            return lev, dom, per, F, G

    def per(f, num, den, cancel=True, kill=False):
        """Create a DMF out of the given representation. """
        lev, dom = f.lev, f.dom

        if kill:
            if not lev:
                return num/den
            else:
                lev -= 1

        if cancel:
            num, den = dmp_cancel(num, den, lev, dom)

        return f.__class__.new((num, den), dom, lev)

    def half_per(f, rep, kill=False):
        """Create a DMP out of the given representation. """
        lev = f.lev

        if kill:
            if not lev:
                return rep
            else:
                lev -= 1

        return DMP(rep, f.dom, lev)

    @classmethod
    def zero(cls, lev, dom):
        return cls.new(0, dom, lev)

    @classmethod
    def one(cls, lev, dom):
        return cls.new(1, dom, lev)

    def numer(f):
        """Returns the numerator of ``f``. """
        return f.half_per(f.num)

    def denom(f):
        """Returns the denominator of ``f``. """
        return f.half_per(f.den)

    def cancel(f):
        """Remove common factors from ``f.num`` and ``f.den``. """
        return f.per(f.num, f.den)

    def neg(f):
        """Negate all coefficients in ``f``. """
        return f.per(dmp_neg(f.num, f.lev, f.dom), f.den, cancel=False)

    def add_ground(f, c):
        """Add an element of the ground domain to ``f``. """
        return f + f.ground_new(c)

    def add(f, g):
        """Add two multivariate fractions ``f`` and ``g``. """
        if isinstance(g, DMP):
            lev, dom, per, (F_num, F_den), G = f.poly_unify(g)
            num, den = dmp_add_mul(F_num, F_den, G, lev, dom), F_den
        else:
            lev, dom, per, F, G = f.frac_unify(g)
            (F_num, F_den), (G_num, G_den) = F, G

            num = dmp_add(dmp_mul(F_num, G_den, lev, dom),
                          dmp_mul(F_den, G_num, lev, dom), lev, dom)
            den = dmp_mul(F_den, G_den, lev, dom)

        return per(num, den)

    def sub(f, g):
        """Subtract two multivariate fractions ``f`` and ``g``. """
        if isinstance(g, DMP):
            lev, dom, per, (F_num, F_den), G = f.poly_unify(g)
            num, den = dmp_sub_mul(F_num, F_den, G, lev, dom), F_den
        else:
            lev, dom, per, F, G = f.frac_unify(g)
            (F_num, F_den), (G_num, G_den) = F, G

            num = dmp_sub(dmp_mul(F_num, G_den, lev, dom),
                          dmp_mul(F_den, G_num, lev, dom), lev, dom)
            den = dmp_mul(F_den, G_den, lev, dom)

        return per(num, den)

    def mul(f, g):
        """Multiply two multivariate fractions ``f`` and ``g``. """
        if isinstance(g, DMP):
            lev, dom, per, (F_num, F_den), G = f.poly_unify(g)
            num, den = dmp_mul(F_num, G, lev, dom), F_den
        else:
            lev, dom, per, F, G = f.frac_unify(g)
            (F_num, F_den), (G_num, G_den) = F, G

            num = dmp_mul(F_num, G_num, lev, dom)
            den = dmp_mul(F_den, G_den, lev, dom)

        return per(num, den)

    def pow(f, n):
        """Raise ``f`` to a non-negative power ``n``. """
        if isinstance(n, int):
            num, den = f.num, f.den
            if n < 0:
                num, den, n = den, num, -n
            return f.per(dmp_pow(num, n, f.lev, f.dom),
                         dmp_pow(den, n, f.lev, f.dom), cancel=False)
        else:
            raise TypeError("``int`` expected, got %s" % type(n))

    def quo(f, g):
        """Computes quotient of fractions ``f`` and ``g``. """
        if isinstance(g, DMP):
            lev, dom, per, (F_num, F_den), G = f.poly_unify(g)
            num, den = F_num, dmp_mul(F_den, G, lev, dom)
        else:
            lev, dom, per, F, G = f.frac_unify(g)
            (F_num, F_den), (G_num, G_den) = F, G

            num = dmp_mul(F_num, G_den, lev, dom)
            den = dmp_mul(F_den, G_num, lev, dom)

        return per(num, den)

    exquo = quo

    def invert(f, check=True):
        """Computes inverse of a fraction ``f``. """
        return f.per(f.den, f.num, cancel=False)

    @property
    def is_zero(f):
        """Returns ``True`` if ``f`` is a zero fraction. """
        return dmp_zero_p(f.num, f.lev)

    @property
    def is_one(f):
        """Returns ``True`` if ``f`` is a unit fraction. """
        return dmp_one_p(f.num, f.lev, f.dom) and \
            dmp_one_p(f.den, f.lev, f.dom)

    def __neg__(f):
        return f.neg()

    def __add__(f, g):
        if isinstance(g, (DMP, DMF)):
            return f.add(g)
        elif g in f.dom:
            return f.add_ground(f.dom.convert(g))

        try:
            return f.add(f.half_per(g))
        except (TypeError, CoercionFailed, NotImplementedError):
            return NotImplemented

    def __radd__(f, g):
        return f.__add__(g)

    def __sub__(f, g):
        if isinstance(g, (DMP, DMF)):
            return f.sub(g)

        try:
            return f.sub(f.half_per(g))
        except (TypeError, CoercionFailed, NotImplementedError):
            return NotImplemented

    def __rsub__(f, g):
        return (-f).__add__(g)

    def __mul__(f, g):
        if isinstance(g, (DMP, DMF)):
            return f.mul(g)

        try:
            return f.mul(f.half_per(g))
        except (TypeError, CoercionFailed, NotImplementedError):
            return NotImplemented

    def __rmul__(f, g):
        return f.__mul__(g)

    def __pow__(f, n):
        return f.pow(n)

    def __truediv__(f, g):
        if isinstance(g, (DMP, DMF)):
            return f.quo(g)

        try:
            return f.quo(f.half_per(g))
        except (TypeError, CoercionFailed, NotImplementedError):
            return NotImplemented

    def __rtruediv__(self, g):
        return self.invert(check=False)*g

    def __eq__(f, g):
        try:
            if isinstance(g, DMP):
                _, _, _, (F_num, F_den), G = f.poly_unify(g)

                if f.lev == g.lev:
                    return dmp_one_p(F_den, f.lev, f.dom) and F_num == G
            else:
                _, _, _, F, G = f.frac_unify(g)

                if f.lev == g.lev:
                    return F == G
        except UnificationFailed:
            pass

        return False

    def __ne__(f, g):
        try:
            if isinstance(g, DMP):
                _, _, _, (F_num, F_den), G = f.poly_unify(g)

                if f.lev == g.lev:
                    return not (dmp_one_p(F_den, f.lev, f.dom) and F_num == G)
            else:
                _, _, _, F, G = f.frac_unify(g)

                if f.lev == g.lev:
                    return F != G
        except UnificationFailed:
            pass

        return True

    def __lt__(f, g):
        _, _, _, F, G = f.frac_unify(g)
        return F < G

    def __le__(f, g):
        _, _, _, F, G = f.frac_unify(g)
        return F <= G

    def __gt__(f, g):
        _, _, _, F, G = f.frac_unify(g)
        return F > G

    def __ge__(f, g):
        _, _, _, F, G = f.frac_unify(g)
        return F >= G

    def __bool__(f):
        return not dmp_zero_p(f.num, f.lev)


def init_normal_ANP(rep, mod, dom):
    return ANP(dup_normal(rep, dom),
               dup_normal(mod, dom), dom)


class ANP(CantSympify):
    """Dense Algebraic Number Polynomials over a field. """

    __slots__ = ('_rep', '_mod', 'dom')

    def __new__(cls, rep, mod, dom):
        if isinstance(rep, DMP):
            pass
        elif type(rep) is dict: # don't use isinstance
            rep = DMP(dup_from_dict(rep, dom), dom, 0)
        else:
            if isinstance(rep, list):
                rep = [dom.convert(a) for a in rep]
            else:
                rep = [dom.convert(rep)]
            rep = DMP(dup_strip(rep), dom, 0)

        if isinstance(mod, DMP):
            pass
        elif isinstance(mod, dict):
            mod = DMP(dup_from_dict(mod, dom), dom, 0)
        else:
            mod = DMP(dup_strip(mod), dom, 0)

        return cls.new(rep, mod, dom)

    @classmethod
    def new(cls, rep, mod, dom):
        if not (rep.dom == mod.dom == dom):
            raise RuntimeError("Inconsistent domain")
        obj = super().__new__(cls)
        obj._rep = rep
        obj._mod = mod
        obj.dom = dom
        return obj

    # XXX: It should be possible to use __getnewargs__ rather than __reduce__
    # but it doesn't work for some reason. Probably this would be easier if
    # python-flint supported pickling for polynomial types.
    def __reduce__(self):
        return ANP, (self.rep, self.mod, self.dom)

    @property
    def rep(self):
        return self._rep.to_list()

    @property
    def mod(self):
        return self.mod_to_list()

    def to_DMP(self):
        return self._rep

    def mod_to_DMP(self):
        return self._mod

    def per(f, rep):
        return f.new(rep, f._mod, f.dom)

    def __repr__(f):
        return "%s(%s, %s, %s)" % (f.__class__.__name__, f._rep.to_list(), f._mod.to_list(), f.dom)

    def __hash__(f):
        return hash((f.__class__.__name__, f.to_tuple(), f._mod.to_tuple(), f.dom))

    def convert(f, dom):
        """Convert ``f`` to a ``ANP`` over a new domain. """
        if f.dom == dom:
            return f
        else:
            return f.new(f._rep.convert(dom), f._mod.convert(dom), dom)

    def unify(f, g):
        """Unify representations of two algebraic numbers. """

        # XXX: This unify method is not used any more because unify_ANP is used
        # instead.

        if not isinstance(g, ANP) or f.mod != g.mod:
            raise UnificationFailed("Cannot unify %s with %s" % (f, g))

        if f.dom == g.dom:
            return f.dom, f.per, f.rep, g.rep, f.mod
        else:
            dom = f.dom.unify(g.dom)

            F = dup_convert(f.rep, f.dom, dom)
            G = dup_convert(g.rep, g.dom, dom)

            if dom != f.dom and dom != g.dom:
                mod = dup_convert(f.mod, f.dom, dom)
            else:
                if dom == f.dom:
                    mod = f.mod
                else:
                    mod = g.mod

            per = lambda rep: ANP(rep, mod, dom)

        return dom, per, F, G, mod

    def unify_ANP(f, g):
        """Unify and return ``DMP`` instances of ``f`` and ``g``. """
        if not isinstance(g, ANP) or f._mod != g._mod:
            raise UnificationFailed("Cannot unify %s with %s" % (f, g))

        # The domain is almost always QQ but there are some tests involving ZZ
        if f.dom != g.dom:
            dom = f.dom.unify(g.dom)
            f = f.convert(dom)
            g = g.convert(dom)

        return f._rep, g._rep, f._mod, f.dom

    @classmethod
    def zero(cls, mod, dom):
        return ANP(0, mod, dom)

    @classmethod
    def one(cls, mod, dom):
        return ANP(1, mod, dom)

    def to_dict(f):
        """Convert ``f`` to a dict representation with native coefficients. """
        return f._rep.to_dict()

    def to_sympy_dict(f):
        """Convert ``f`` to a dict representation with SymPy coefficients. """
        rep = dmp_to_dict(f.rep, 0, f.dom)

        for k, v in rep.items():
            rep[k] = f.dom.to_sympy(v)

        return rep

    def to_list(f):
        """Convert ``f`` to a list representation with native coefficients. """
        return f._rep.to_list()

    def mod_to_list(f):
        """Return ``f.mod`` as a list with native coefficients. """
        return f._mod.to_list()

    def to_sympy_list(f):
        """Convert ``f`` to a list representation with SymPy coefficients. """
        return [ f.dom.to_sympy(c) for c in f.to_list() ]

    def to_tuple(f):
        """
        Convert ``f`` to a tuple representation with native coefficients.

        This is needed for hashing.
        """
        return f._rep.to_tuple()

    @classmethod
    def from_list(cls, rep, mod, dom):
        return ANP(dup_strip(list(map(dom.convert, rep))), mod, dom)

    def add_ground(f, c):
        """Add an element of the ground domain to ``f``. """
        return f.per(f._rep.add_ground(c))

    def sub_ground(f, c):
        """Subtract an element of the ground domain from ``f``. """
        return f.per(f._rep.sub_ground(c))

    def mul_ground(f, c):
        """Multiply ``f`` by an element of the ground domain. """
        return f.per(f._rep.mul_ground(c))

    def quo_ground(f, c):
        """Quotient of ``f`` by an element of the ground domain. """
        return f.per(f._rep.quo_ground(c))

    def neg(f):
        return f.per(f._rep.neg())

    def add(f, g):
        F, G, mod, dom = f.unify_ANP(g)
        return f.new(F.add(G), mod, dom)

    def sub(f, g):
        F, G, mod, dom = f.unify_ANP(g)
        return f.new(F.sub(G), mod, dom)

    def mul(f, g):
        F, G, mod, dom = f.unify_ANP(g)
        return f.new(F.mul(G).rem(mod), mod, dom)

    def pow(f, n):
        """Raise ``f`` to a non-negative power ``n``. """
        if not isinstance(n, int):
            raise TypeError("``int`` expected, got %s" % type(n))

        mod = f._mod
        F = f._rep

        if n < 0:
            F, n = F.invert(mod), -n

        # XXX: Need a pow_mod method for DMP
        return f.new(F.pow(n).rem(f._mod), mod, f.dom)

    def exquo(f, g):
        F, G, mod, dom = f.unify_ANP(g)
        return f.new(F.mul(G.invert(mod)).rem(mod), mod, dom)

    def div(f, g):
        return f.exquo(g), f.zero(f._mod, f.dom)

    def quo(f, g):
        return f.exquo(g)

    def rem(f, g):
        F, G, mod, dom = f.unify_ANP(g)
        s, h = F.half_gcdex(G)

        if h.is_one:
            return f.zero(mod, dom)
        else:
            raise NotInvertible("zero divisor")

    def LC(f):
        """Returns the leading coefficient of ``f``. """
        return f._rep.LC()

    def TC(f):
        """Returns the trailing coefficient of ``f``. """
        return f._rep.TC()

    @property
    def is_zero(f):
        """Returns ``True`` if ``f`` is a zero algebraic number. """
        return f._rep.is_zero

    @property
    def is_one(f):
        """Returns ``True`` if ``f`` is a unit algebraic number. """
        return f._rep.is_one

    @property
    def is_ground(f):
        """Returns ``True`` if ``f`` is an element of the ground domain. """
        return f._rep.is_ground

    def __pos__(f):
        return f

    def __neg__(f):
        return f.neg()

    def __add__(f, g):
        if isinstance(g, ANP):
            return f.add(g)
        try:
            g = f.dom.convert(g)
        except CoercionFailed:
            return NotImplemented
        else:
            return f.add_ground(g)

    def __radd__(f, g):
        return f.__add__(g)

    def __sub__(f, g):
        if isinstance(g, ANP):
            return f.sub(g)
        try:
            g = f.dom.convert(g)
        except CoercionFailed:
            return NotImplemented
        else:
            return f.sub_ground(g)

    def __rsub__(f, g):
        return (-f).__add__(g)

    def __mul__(f, g):
        if isinstance(g, ANP):
            return f.mul(g)
        try:
            g = f.dom.convert(g)
        except CoercionFailed:
            return NotImplemented
        else:
            return f.mul_ground(g)

    def __rmul__(f, g):
        return f.__mul__(g)

    def __pow__(f, n):
        return f.pow(n)

    def __divmod__(f, g):
        return f.div(g)

    def __mod__(f, g):
        return f.rem(g)

    def __truediv__(f, g):
        if isinstance(g, ANP):
            return f.quo(g)
        try:
            g = f.dom.convert(g)
        except CoercionFailed:
            return NotImplemented
        else:
            return f.quo_ground(g)

    def __eq__(f, g):
        try:
            F, G, _, _ = f.unify_ANP(g)
        except UnificationFailed:
            return NotImplemented
        return F == G

    def __ne__(f, g):
        try:
            F, G, _, _ = f.unify_ANP(g)
        except UnificationFailed:
            return NotImplemented
        return F != G

    def __lt__(f, g):
        F, G, _, _ = f.unify_ANP(g)
        return F < G

    def __le__(f, g):
        F, G, _, _ = f.unify_ANP(g)
        return F <= G

    def __gt__(f, g):
        F, G, _, _ = f.unify_ANP(g)
        return F > G

    def __ge__(f, g):
        F, G, _, _ = f.unify_ANP(g)
        return F >= G

    def __bool__(f):
        return bool(f._rep)
<|endoftext|>from sympy.core.symbol import Dummy
from sympy.ntheory import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import PolynomialRing
from sympy.polys.galoistools import (
    gf_gcd, gf_from_dict, gf_gcdex, gf_div, gf_lcm)
from sympy.polys.polyerrors import ModularGCDFailed

from mpmath import sqrt
import random


def _trivial_gcd(f, g):
    """
    Compute the GCD of two polynomials in trivial cases, i.e. when one
    or both polynomials are zero.
    """
    ring = f.ring

    if not (f or g):
        return ring.zero, ring.zero, ring.zero
    elif not f:
        if g.LC < ring.domain.zero:
            return -g, ring.zero, -ring.one
        else:
            return g, ring.zero, ring.one
    elif not g:
        if f.LC < ring.domain.zero:
            return -f, -ring.one, ring.zero
        else:
            return f, ring.one, ring.zero
    return None


def _gf_gcd(fp, gp, p):
    r"""
    Compute the GCD of two univariate polynomials in `\mathbb{Z}_p[x]`.
    """
    dom = fp.ring.domain

    while gp:
        rem = fp
        deg = gp.degree()
        lcinv = dom.invert(gp.LC, p)

        while True:
            degrem = rem.degree()
            if degrem < deg:
                break
            rem = (rem - gp.mul_monom((degrem - deg,)).mul_ground(lcinv * rem.LC)).trunc_ground(p)

        fp = gp
        gp = rem

    return fp.mul_ground(dom.invert(fp.LC, p)).trunc_ground(p)


def _degree_bound_univariate(f, g):
    r"""
    Compute an upper bound for the degree of the GCD of two univariate
    integer polynomials `f` and `g`.

    The function chooses a suitable prime `p` and computes the GCD of
    `f` and `g` in `\mathbb{Z}_p[x]`. The choice of `p` guarantees that
    the degree in `\mathbb{Z}_p[x]` is greater than or equal to the degree
    in `\mathbb{Z}[x]`.

    Parameters
    ==========

    f : PolyElement
        univariate integer polynomial
    g : PolyElement
        univariate integer polynomial

    """
    gamma = f.ring.domain.gcd(f.LC, g.LC)
    p = 1

    p = nextprime(p)
    while gamma % p == 0:
        p = nextprime(p)

    fp = f.trunc_ground(p)
    gp = g.trunc_ground(p)
    hp = _gf_gcd(fp, gp, p)
    deghp = hp.degree()
    return deghp


def _chinese_remainder_reconstruction_univariate(hp, hq, p, q):
    r"""
    Construct a polynomial `h_{pq}` in `\mathbb{Z}_{p q}[x]` such that

    .. math ::

        h_{pq} = h_p \; \mathrm{mod} \, p

        h_{pq} = h_q \; \mathrm{mod} \, q

    for relatively prime integers `p` and `q` and polynomials
    `h_p` and `h_q` in `\mathbb{Z}_p[x]` and `\mathbb{Z}_q[x]`
    respectively.

    The coefficients of the polynomial `h_{pq}` are computed with the
    Chinese Remainder Theorem. The symmetric representation in
    `\mathbb{Z}_p[x]`, `\mathbb{Z}_q[x]` and `\mathbb{Z}_{p q}[x]` is used.
    It is assumed that `h_p` and `h_q` have the same degree.

    Parameters
    ==========

    hp : PolyElement
        univariate integer polynomial with coefficients in `\mathbb{Z}_p`
    hq : PolyElement
        univariate integer polynomial with coefficients in `\mathbb{Z}_q`
    p : Integer
        modulus of `h_p`, relatively prime to `q`
    q : Integer
        modulus of `h_q`, relatively prime to `p`

    Examples
    ========

    >>> from sympy.polys.modulargcd import _chinese_remainder_reconstruction_univariate
    >>> from sympy.polys import ring, ZZ

    >>> R, x = ring("x", ZZ)
    >>> p = 3
    >>> q = 5

    >>> hp = -x**3 - 1
    >>> hq = 2*x**3 - 2*x**2 + x

    >>> hpq = _chinese_remainder_reconstruction_univariate(hp, hq, p, q)
    >>> hpq
    2*x**3 + 3*x**2 + 6*x + 5

    >>> hpq.trunc_ground(p) == hp
    True
    >>> hpq.trunc_ground(q) == hq
    True

    """
    n = hp.degree()
    x = hp.ring.gens[0]
    hpq = hp.ring.zero

    for i in range(n+1):
        hpq[(i,)] = crt([p, q], [hp.coeff(x**i), hq.coeff(x**i)], symmetric=True)[0]

    hpq.strip_zero()
    return hpq


def modgcd_univariate(f, g):
    r"""
    Computes the GCD of two polynomials in `\mathbb{Z}[x]` using a modular
    algorithm.

    The algorithm computes the GCD of two univariate integer polynomials
    `f` and `g` by computing the GCD in `\mathbb{Z}_p[x]` for suitable
    primes `p` and then reconstructing the coefficients with the Chinese
    Remainder Theorem. Trial division is only made for candidates which
    are very likely the desired GCD.

    Parameters
    ==========

    f : PolyElement
        univariate integer polynomial
    g : PolyElement
        univariate integer polynomial

    Returns
    =======

    h : PolyElement
        GCD of the polynomials `f` and `g`
    cff : PolyElement
        cofactor of `f`, i.e. `\frac{f}{h}`
    cfg : PolyElement
        cofactor of `g`, i.e. `\frac{g}{h}`

    Examples
    ========

    >>> from sympy.polys.modulargcd import modgcd_univariate
    >>> from sympy.polys import ring, ZZ

    >>> R, x = ring("x", ZZ)

    >>> f = x**5 - 1
    >>> g = x - 1

    >>> h, cff, cfg = modgcd_univariate(f, g)
    >>> h, cff, cfg
    (x - 1, x**4 + x**3 + x**2 + x + 1, 1)

    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    >>> f = 6*x**2 - 6
    >>> g = 2*x**2 + 4*x + 2

    >>> h, cff, cfg = modgcd_univariate(f, g)
    >>> h, cff, cfg
    (2*x + 2, 3*x - 3, x + 1)

    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    References
    ==========

    1. [Monagan00]_

    """
    assert f.ring == g.ring and f.ring.domain.is_ZZ

    result = _trivial_gcd(f, g)
    if result is not None:
        return result

    ring = f.ring

    cf, f = f.primitive()
    cg, g = g.primitive()
    ch = ring.domain.gcd(cf, cg)

    bound = _degree_bound_univariate(f, g)
    if bound == 0:
        return ring(ch), f.mul_ground(cf // ch), g.mul_ground(cg // ch)

    gamma = ring.domain.gcd(f.LC, g.LC)
    m = 1
    p = 1

    while True:
        p = nextprime(p)
        while gamma % p == 0:
            p = nextprime(p)

        fp = f.trunc_ground(p)
        gp = g.trunc_ground(p)
        hp = _gf_gcd(fp, gp, p)
        deghp = hp.degree()

        if deghp > bound:
            continue
        elif deghp < bound:
            m = 1
            bound = deghp
            continue

        hp = hp.mul_ground(gamma).trunc_ground(p)
        if m == 1:
            m = p
            hlastm = hp
            continue

        hm = _chinese_remainder_reconstruction_univariate(hp, hlastm, p, m)
        m *= p

        if not hm == hlastm:
            hlastm = hm
            continue

        h = hm.quo_ground(hm.content())
        fquo, frem = f.div(h)
        gquo, grem = g.div(h)
        if not frem and not grem:
            if h.LC < 0:
                ch = -ch
            h = h.mul_ground(ch)
            cff = fquo.mul_ground(cf // ch)
            cfg = gquo.mul_ground(cg // ch)
            return h, cff, cfg


def _primitive(f, p):
    r"""
    Compute the content and the primitive part of a polynomial in
    `\mathbb{Z}_p[x_0, \ldots, x_{k-2}, y] \cong \mathbb{Z}_p[y][x_0, \ldots, x_{k-2}]`.

    Parameters
    ==========

    f : PolyElement
        integer polynomial in `\mathbb{Z}_p[x0, \ldots, x{k-2}, y]`
    p : Integer
        modulus of `f`

    Returns
    =======

    contf : PolyElement
        integer polynomial in `\mathbb{Z}_p[y]`, content of `f`
    ppf : PolyElement
        primitive part of `f`, i.e. `\frac{f}{contf}`

    Examples
    ========

    >>> from sympy.polys.modulargcd import _primitive
    >>> from sympy.polys import ring, ZZ

    >>> R, x, y = ring("x, y", ZZ)
    >>> p = 3

    >>> f = x**2*y**2 + x**2*y - y**2 - y
    >>> _primitive(f, p)
    (y**2 + y, x**2 - 1)

    >>> R, x, y, z = ring("x, y, z", ZZ)

    >>> f = x*y*z - y**2*z**2
    >>> _primitive(f, p)
    (z, x*y - y**2*z)

    """
    ring = f.ring
    dom = ring.domain
    k = ring.ngens

    coeffs = {}
    for monom, coeff in f.iterterms():
        if monom[:-1] not in coeffs:
            coeffs[monom[:-1]] = {}
        coeffs[monom[:-1]][monom[-1]] = coeff

    cont = []
    for coeff in iter(coeffs.values()):
        cont = gf_gcd(cont, gf_from_dict(coeff, p, dom), p, dom)

    yring = ring.clone(symbols=ring.symbols[k-1])
    contf = yring.from_dense(cont).trunc_ground(p)

    return contf, f.quo(contf.set_ring(ring))


def _deg(f):
    r"""
    Compute the degree of a multivariate polynomial
    `f \in K[x_0, \ldots, x_{k-2}, y] \cong K[y][x_0, \ldots, x_{k-2}]`.

    Parameters
    ==========

    f : PolyElement
        polynomial in `K[x_0, \ldots, x_{k-2}, y]`

    Returns
    =======

    degf : Integer tuple
        degree of `f` in `x_0, \ldots, x_{k-2}`

    Examples
    ========

    >>> from sympy.polys.modulargcd import _deg
    >>> from sympy.polys import ring, ZZ

    >>> R, x, y = ring("x, y", ZZ)

    >>> f = x**2*y**2 + x**2*y - 1
    >>> _deg(f)
    (2,)

    >>> R, x, y, z = ring("x, y, z", ZZ)

    >>> f = x**2*y**2 + x**2*y - 1
    >>> _deg(f)
    (2, 2)

    >>> f = x*y*z - y**2*z**2
    >>> _deg(f)
    (1, 1)

    """
    k = f.ring.ngens
    degf = (0,) * (k-1)
    for monom in f.itermonoms():
        if monom[:-1] > degf:
            degf = monom[:-1]
    return degf


def _LC(f):
    r"""
    Compute the leading coefficient of a multivariate polynomial
    `f \in K[x_0, \ldots, x_{k-2}, y] \cong K[y][x_0, \ldots, x_{k-2}]`.

    Parameters
    ==========

    f : PolyElement
        polynomial in `K[x_0, \ldots, x_{k-2}, y]`

    Returns
    =======

    lcf : PolyElement
        polynomial in `K[y]`, leading coefficient of `f`

    Examples
    ========

    >>> from sympy.polys.modulargcd import _LC
    >>> from sympy.polys import ring, ZZ

    >>> R, x, y = ring("x, y", ZZ)

    >>> f = x**2*y**2 + x**2*y - 1
    >>> _LC(f)
    y**2 + y

    >>> R, x, y, z = ring("x, y, z", ZZ)

    >>> f = x**2*y**2 + x**2*y - 1
    >>> _LC(f)
    1

    >>> f = x*y*z - y**2*z**2
    >>> _LC(f)
    z

    """
    ring = f.ring
    k = ring.ngens
    yring = ring.clone(symbols=ring.symbols[k-1])
    y = yring.gens[0]
    degf = _deg(f)

    lcf = yring.zero
    for monom, coeff in f.iterterms():
        if monom[:-1] == degf:
            lcf += coeff*y**monom[-1]
    return lcf


def _swap(f, i):
    """
    Make the variable `x_i` the leading one in a multivariate polynomial `f`.
    """
    ring = f.ring
    fswap = ring.zero
    for monom, coeff in f.iterterms():
        monomswap = (monom[i],) + monom[:i] + monom[i+1:]
        fswap[monomswap] = coeff
    return fswap


def _degree_bound_bivariate(f, g):
    r"""
    Compute upper degree bounds for the GCD of two bivariate
    integer polynomials `f` and `g`.

    The GCD is viewed as a polynomial in `\mathbb{Z}[y][x]` and the
    function returns an upper bound for its degree and one for the degree
    of its content. This is done by choosing a suitable prime `p` and
    computing the GCD of the contents of `f \; \mathrm{mod} \, p` and
    `g \; \mathrm{mod} \, p`. The choice of `p` guarantees that the degree
    of the content in `\mathbb{Z}_p[y]` is greater than or equal to the
    degree in `\mathbb{Z}[y]`. To obtain the degree bound in the variable
    `x`, the polynomials are evaluated at `y = a` for a suitable
    `a \in \mathbb{Z}_p` and then their GCD in `\mathbb{Z}_p[x]` is
    computed. If no such `a` exists, i.e. the degree in `\mathbb{Z}_p[x]`
    is always smaller than the one in `\mathbb{Z}[y][x]`, then the bound is
    set to the minimum of the degrees of `f` and `g` in `x`.

    Parameters
    ==========

    f : PolyElement
        bivariate integer polynomial
    g : PolyElement
        bivariate integer polynomial

    Returns
    =======

    xbound : Integer
        upper bound for the degree of the GCD of the polynomials `f` and
        `g` in the variable `x`
    ycontbound : Integer
        upper bound for the degree of the content of the GCD of the
        polynomials `f` and `g` in the variable `y`

    References
    ==========

    1. [Monagan00]_

    """
    ring = f.ring

    gamma1 = ring.domain.gcd(f.LC, g.LC)
    gamma2 = ring.domain.gcd(_swap(f, 1).LC, _swap(g, 1).LC)
    badprimes = gamma1 * gamma2
    p = 1

    p = nextprime(p)
    while badprimes % p == 0:
        p = nextprime(p)

    fp = f.trunc_ground(p)
    gp = g.trunc_ground(p)
    contfp, fp = _primitive(fp, p)
    contgp, gp = _primitive(gp, p)
    conthp = _gf_gcd(contfp, contgp, p) # polynomial in Z_p[y]
    ycontbound = conthp.degree()

    # polynomial in Z_p[y]
    delta = _gf_gcd(_LC(fp), _LC(gp), p)

    for a in range(p):
        if not delta.evaluate(0, a) % p:
            continue
        fpa = fp.evaluate(1, a).trunc_ground(p)
        gpa = gp.evaluate(1, a).trunc_ground(p)
        hpa = _gf_gcd(fpa, gpa, p)
        xbound = hpa.degree()
        return xbound, ycontbound

    return min(fp.degree(), gp.degree()), ycontbound


def _chinese_remainder_reconstruction_multivariate(hp, hq, p, q):
    r"""
    Construct a polynomial `h_{pq}` in
    `\mathbb{Z}_{p q}[x_0, \ldots, x_{k-1}]` such that

    .. math ::

        h_{pq} = h_p \; \mathrm{mod} \, p

        h_{pq} = h_q \; \mathrm{mod} \, q

    for relatively prime integers `p` and `q` and polynomials
    `h_p` and `h_q` in `\mathbb{Z}_p[x_0, \ldots, x_{k-1}]` and
    `\mathbb{Z}_q[x_0, \ldots, x_{k-1}]` respectively.

    The coefficients of the polynomial `h_{pq}` are computed with the
    Chinese Remainder Theorem. The symmetric representation in
    `\mathbb{Z}_p[x_0, \ldots, x_{k-1}]`,
    `\mathbb{Z}_q[x_0, \ldots, x_{k-1}]` and
    `\mathbb{Z}_{p q}[x_0, \ldots, x_{k-1}]` is used.

    Parameters
    ==========

    hp : PolyElement
        multivariate integer polynomial with coefficients in `\mathbb{Z}_p`
    hq : PolyElement
        multivariate integer polynomial with coefficients in `\mathbb{Z}_q`
    p : Integer
        modulus of `h_p`, relatively prime to `q`
    q : Integer
        modulus of `h_q`, relatively prime to `p`

    Examples
    ========

    >>> from sympy.polys.modulargcd import _chinese_remainder_reconstruction_multivariate
    >>> from sympy.polys import ring, ZZ

    >>> R, x, y = ring("x, y", ZZ)
    >>> p = 3
    >>> q = 5

    >>> hp = x**3*y - x**2 - 1
    >>> hq = -x**3*y - 2*x*y**2 + 2

    >>> hpq = _chinese_remainder_reconstruction_multivariate(hp, hq, p, q)
    >>> hpq
    4*x**3*y + 5*x**2 + 3*x*y**2 + 2

    >>> hpq.trunc_ground(p) == hp
    True
    >>> hpq.trunc_ground(q) == hq
    True

    >>> R, x, y, z = ring("x, y, z", ZZ)
    >>> p = 6
    >>> q = 5

    >>> hp = 3*x**4 - y**3*z + z
    >>> hq = -2*x**4 + z

    >>> hpq = _chinese_remainder_reconstruction_multivariate(hp, hq, p, q)
    >>> hpq
    3*x**4 + 5*y**3*z + z

    >>> hpq.trunc_ground(p) == hp
    True
    >>> hpq.trunc_ground(q) == hq
    True

    """
    hpmonoms = set(hp.monoms())
    hqmonoms = set(hq.monoms())
    monoms = hpmonoms.intersection(hqmonoms)
    hpmonoms.difference_update(monoms)
    hqmonoms.difference_update(monoms)

    domain = hp.ring.domain
    zero = domain.zero

    hpq = hp.ring.zero

    if isinstance(hp.ring.domain, PolynomialRing):
        crt_ = _chinese_remainder_reconstruction_multivariate
    else:
        def crt_(cp, cq, p, q):
            return domain(crt([p, q], [cp, cq], symmetric=True)[0])

    for monom in monoms:
        hpq[monom] = crt_(hp[monom], hq[monom], p, q)
    for monom in hpmonoms:
        hpq[monom] = crt_(hp[monom], zero, p, q)
    for monom in hqmonoms:
        hpq[monom] = crt_(zero, hq[monom], p, q)

    return hpq


def _interpolate_multivariate(evalpoints, hpeval, ring, i, p, ground=False):
    r"""
    Reconstruct a polynomial `h_p` in `\mathbb{Z}_p[x_0, \ldots, x_{k-1}]`
    from a list of evaluation points in `\mathbb{Z}_p` and a list of
    polynomials in
    `\mathbb{Z}_p[x_0, \ldots, x_{i-1}, x_{i+1}, \ldots, x_{k-1}]`, which
    are the images of `h_p` evaluated in the variable `x_i`.

    It is also possible to reconstruct a parameter of the ground domain,
    i.e. if `h_p` is a polynomial over `\mathbb{Z}_p[x_0, \ldots, x_{k-1}]`.
    In this case, one has to set ``ground=True``.

    Parameters
    ==========

    evalpoints : list of Integer objects
        list of evaluation points in `\mathbb{Z}_p`
    hpeval : list of PolyElement objects
        list of polynomials in (resp. over)
        `\mathbb{Z}_p[x_0, \ldots, x_{i-1}, x_{i+1}, \ldots, x_{k-1}]`,
        images of `h_p` evaluated in the variable `x_i`
    ring : PolyRing
        `h_p` will be an element of this ring
    i : Integer
        index of the variable which has to be reconstructed
    p : Integer
        prime number, modulus of `h_p`
    ground : Boolean
        indicates whether `x_i` is in the ground domain, default is
        ``False``

    Returns
    =======

    hp : PolyElement
        interpolated polynomial in (resp. over)
        `\mathbb{Z}_p[x_0, \ldots, x_{k-1}]`

    """
    hp = ring.zero

    if ground:
        domain = ring.domain.domain
        y = ring.domain.gens[i]
    else:
        domain = ring.domain
        y = ring.gens[i]

    for a, hpa in zip(evalpoints, hpeval):
        numer = ring.one
        denom = domain.one
        for b in evalpoints:
            if b == a:
                continue

            numer *= y - b
            denom *= a - b

        denom = domain.invert(denom, p)
        coeff = numer.mul_ground(denom)
        hp += hpa.set_ring(ring) * coeff

    return hp.trunc_ground(p)


def modgcd_bivariate(f, g):
    r"""
    Computes the GCD of two polynomials in `\mathbb{Z}[x, y]` using a
    modular algorithm.

    The algorithm computes the GCD of two bivariate integer polynomials
    `f` and `g` by calculating the GCD in `\mathbb{Z}_p[x, y]` for
    suitable primes `p` and then reconstructing the coefficients with the
    Chinese Remainder Theorem. To compute the bivariate GCD over
    `\mathbb{Z}_p`, the polynomials `f \; \mathrm{mod} \, p` and
    `g \; \mathrm{mod} \, p` are evaluated at `y = a` for certain
    `a \in \mathbb{Z}_p` and then their univariate GCD in `\mathbb{Z}_p[x]`
    is computed. Interpolating those yields the bivariate GCD in
    `\mathbb{Z}_p[x, y]`. To verify the result in `\mathbb{Z}[x, y]`, trial
    division is done, but only for candidates which are very likely the
    desired GCD.

    Parameters
    ==========

    f : PolyElement
        bivariate integer polynomial
    g : PolyElement
        bivariate integer polynomial

    Returns
    =======

    h : PolyElement
        GCD of the polynomials `f` and `g`
    cff : PolyElement
        cofactor of `f`, i.e. `\frac{f}{h}`
    cfg : PolyElement
        cofactor of `g`, i.e. `\frac{g}{h}`

    Examples
    ========

    >>> from sympy.polys.modulargcd import modgcd_bivariate
    >>> from sympy.polys import ring, ZZ

    >>> R, x, y = ring("x, y", ZZ)

    >>> f = x**2 - y**2
    >>> g = x**2 + 2*x*y + y**2

    >>> h, cff, cfg = modgcd_bivariate(f, g)
    >>> h, cff, cfg
    (x + y, x - y, x + y)

    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    >>> f = x**2*y - x**2 - 4*y + 4
    >>> g = x + 2

    >>> h, cff, cfg = modgcd_bivariate(f, g)
    >>> h, cff, cfg
    (x + 2, x*y - x - 2*y + 2, 1)

    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    References
    ==========

    1. [Monagan00]_

    """
    assert f.ring == g.ring and f.ring.domain.is_ZZ

    result = _trivial_gcd(f, g)
    if result is not None:
        return result

    ring = f.ring

    cf, f = f.primitive()
    cg, g = g.primitive()
    ch = ring.domain.gcd(cf, cg)

    xbound, ycontbound = _degree_bound_bivariate(f, g)
    if xbound == ycontbound == 0:
        return ring(ch), f.mul_ground(cf // ch), g.mul_ground(cg // ch)

    fswap = _swap(f, 1)
    gswap = _swap(g, 1)
    degyf = fswap.degree()
    degyg = gswap.degree()

    ybound, xcontbound = _degree_bound_bivariate(fswap, gswap)
    if ybound == xcontbound == 0:
        return ring(ch), f.mul_ground(cf // ch), g.mul_ground(cg // ch)

    # TODO: to improve performance, choose the main variable here

    gamma1 = ring.domain.gcd(f.LC, g.LC)
    gamma2 = ring.domain.gcd(fswap.LC, gswap.LC)
    badprimes = gamma1 * gamma2
    m = 1
    p = 1

    while True:
        p = nextprime(p)
        while badprimes % p == 0:
            p = nextprime(p)

        fp = f.trunc_ground(p)
        gp = g.trunc_ground(p)
        contfp, fp = _primitive(fp, p)
        contgp, gp = _primitive(gp, p)
        conthp = _gf_gcd(contfp, contgp, p) # monic polynomial in Z_p[y]
        degconthp = conthp.degree()

        if degconthp > ycontbound:
            continue
        elif degconthp < ycontbound:
            m = 1
            ycontbound = degconthp
            continue

        # polynomial in Z_p[y]
        delta = _gf_gcd(_LC(fp), _LC(gp), p)

        degcontfp = contfp.degree()
        degcontgp = contgp.degree()
        degdelta = delta.degree()

        N = min(degyf - degcontfp, degyg - degcontgp,
            ybound - ycontbound + degdelta) + 1

        if p < N:
            continue

        n = 0
        evalpoints = []
        hpeval = []
        unlucky = False

        for a in range(p):
            deltaa = delta.evaluate(0, a)
            if not deltaa % p:
                continue

            fpa = fp.evaluate(1, a).trunc_ground(p)
            gpa = gp.evaluate(1, a).trunc_ground(p)
            hpa = _gf_gcd(fpa, gpa, p) # monic polynomial in Z_p[x]
            deghpa = hpa.degree()

            if deghpa > xbound:
                continue
            elif deghpa < xbound:
                m = 1
                xbound = deghpa
                unlucky = True
                break

            hpa = hpa.mul_ground(deltaa).trunc_ground(p)
            evalpoints.append(a)
            hpeval.append(hpa)
            n += 1

            if n == N:
                break

        if unlucky:
            continue
        if n < N:
            continue

        hp = _interpolate_multivariate(evalpoints, hpeval, ring, 1, p)

        hp = _primitive(hp, p)[1]
        hp = hp * conthp.set_ring(ring)
        degyhp = hp.degree(1)

        if degyhp > ybound:
            continue
        if degyhp < ybound:
            m = 1
            ybound = degyhp
            continue

        hp = hp.mul_ground(gamma1).trunc_ground(p)
        if m == 1:
            m = p
            hlastm = hp
            continue

        hm = _chinese_remainder_reconstruction_multivariate(hp, hlastm, p, m)
        m *= p

        if not hm == hlastm:
            hlastm = hm
            continue

        h = hm.quo_ground(hm.content())
        fquo, frem = f.div(h)
        gquo, grem = g.div(h)
        if not frem and not grem:
            if h.LC < 0:
                ch = -ch
            h = h.mul_ground(ch)
            cff = fquo.mul_ground(cf // ch)
            cfg = gquo.mul_ground(cg // ch)
            return h, cff, cfg


def _modgcd_multivariate_p(f, g, p, degbound, contbound):
    r"""
    Compute the GCD of two polynomials in
    `\mathbb{Z}_p[x_0, \ldots, x_{k-1}]`.

    The algorithm reduces the problem step by step by evaluating the
    polynomials `f` and `g` at `x_{k-1} = a` for suitable
    `a \in \mathbb{Z}_p` and then calls itself recursively to compute the GCD
    in `\mathbb{Z}_p[x_0, \ldots, x_{k-2}]`. If these recursive calls are
    successful for enough evaluation points, the GCD in `k` variables is
    interpolated, otherwise the algorithm returns ``None``. Every time a GCD
    or a content is computed, their degrees are compared with the bounds. If
    a degree greater then the bound is encountered, then the current call
    returns ``None`` and a new evaluation point has to be chosen. If at some
    point the degree is smaller, the correspondent bound is updated and the
    algorithm fails.

    Parameters
    ==========

    f : PolyElement
        multivariate integer polynomial with coefficients in `\mathbb{Z}_p`
    g : PolyElement
        multivariate integer polynomial with coefficients in `\mathbb{Z}_p`
    p : Integer
        prime number, modulus of `f` and `g`
    degbound : list of Integer objects
        ``degbound[i]`` is an upper bound for the degree of the GCD of `f`
        and `g` in the variable `x_i`
    contbound : list of Integer objects
        ``contbound[i]`` is an upper bound for the degree of the content of
        the GCD in `\mathbb{Z}_p[x_i][x_0, \ldots, x_{i-1}]`,
        ``contbound[0]`` is not used can therefore be chosen
        arbitrarily.

    Returns
    =======

    h : PolyElement
        GCD of the polynomials `f` and `g` or ``None``

    References
    ==========

    1. [Monagan00]_
    2. [Brown71]_

    """
    ring = f.ring
    k = ring.ngens

    if k == 1:
        h = _gf_gcd(f, g, p).trunc_ground(p)
        degh = h.degree()

        if degh > degbound[0]:
            return None
        if degh < degbound[0]:
            degbound[0] = degh
            raise ModularGCDFailed

        return h

    degyf = f.degree(k-1)
    degyg = g.degree(k-1)

    contf, f = _primitive(f, p)
    contg, g = _primitive(g, p)

    conth = _gf_gcd(contf, contg, p) # polynomial in Z_p[y]

    degcontf = contf.degree()
    degcontg = contg.degree()
    degconth = conth.degree()

    if degconth > contbound[k-1]:
        return None
    if degconth < contbound[k-1]:
        contbound[k-1] = degconth
        raise ModularGCDFailed

    lcf = _LC(f)
    lcg = _LC(g)

    delta = _gf_gcd(lcf, lcg, p) # polynomial in Z_p[y]

    evaltest = delta

    for i in range(k-1):
        evaltest *= _gf_gcd(_LC(_swap(f, i)), _LC(_swap(g, i)), p)

    degdelta = delta.degree()

    N = min(degyf - degcontf, degyg - degcontg,
            degbound[k-1] - contbound[k-1] + degdelta) + 1

    if p < N:
        return None

    n = 0
    d = 0
    evalpoints = []
    heval = []
    points = list(range(p))

    while points:
        a = random.sample(points, 1)[0]
        points.remove(a)

        if not evaltest.evaluate(0, a) % p:
            continue

        deltaa = delta.evaluate(0, a) % p

        fa = f.evaluate(k-1, a).trunc_ground(p)
        ga = g.evaluate(k-1, a).trunc_ground(p)

        # polynomials in Z_p[x_0, ..., x_{k-2}]
        ha = _modgcd_multivariate_p(fa, ga, p, degbound, contbound)

        if ha is None:
            d += 1
            if d > n:
                return None
            continue

        if ha.is_ground:
            h = conth.set_ring(ring).trunc_ground(p)
            return h

        ha = ha.mul_ground(deltaa).trunc_ground(p)

        evalpoints.append(a)
        heval.append(ha)
        n += 1

        if n == N:
            h = _interpolate_multivariate(evalpoints, heval, ring, k-1, p)

            h = _primitive(h, p)[1] * conth.set_ring(ring)
            degyh = h.degree(k-1)

            if degyh > degbound[k-1]:
                return None
            if degyh < degbound[k-1]:
                degbound[k-1] = degyh
                raise ModularGCDFailed

            return h

    return None


def modgcd_multivariate(f, g):
    r"""
    Compute the GCD of two polynomials in `\mathbb{Z}[x_0, \ldots, x_{k-1}]`
    using a modular algorithm.

    The algorithm computes the GCD of two multivariate integer polynomials
    `f` and `g` by calculating the GCD in
    `\mathbb{Z}_p[x_0, \ldots, x_{k-1}]` for suitable primes `p` and then
    reconstructing the coefficients with the Chinese Remainder Theorem. To
    compute the multivariate GCD over `\mathbb{Z}_p` the recursive
    subroutine :func:`_modgcd_multivariate_p` is used. To verify the result in
    `\mathbb{Z}[x_0, \ldots, x_{k-1}]`, trial division is done, but only for
    candidates which are very likely the desired GCD.

    Parameters
    ==========

    f : PolyElement
        multivariate integer polynomial
    g : PolyElement
        multivariate integer polynomial

    Returns
    =======

    h : PolyElement
        GCD of the polynomials `f` and `g`
    cff : PolyElement
        cofactor of `f`, i.e. `\frac{f}{h}`
    cfg : PolyElement
        cofactor of `g`, i.e. `\frac{g}{h}`

    Examples
    ========

    >>> from sympy.polys.modulargcd import modgcd_multivariate
    >>> from sympy.polys import ring, ZZ

    >>> R, x, y = ring("x, y", ZZ)

    >>> f = x**2 - y**2
    >>> g = x**2 + 2*x*y + y**2

    >>> h, cff, cfg = modgcd_multivariate(f, g)
    >>> h, cff, cfg
    (x + y, x - y, x + y)

    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    >>> R, x, y, z = ring("x, y, z", ZZ)

    >>> f = x*z**2 - y*z**2
    >>> g = x**2*z + z

    >>> h, cff, cfg = modgcd_multivariate(f, g)
    >>> h, cff, cfg
    (z, x*z - y*z, x**2 + 1)

    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    References
    ==========

    1. [Monagan00]_
    2. [Brown71]_

    See also
    ========

    _modgcd_multivariate_p

    """
    assert f.ring == g.ring and f.ring.domain.is_ZZ

    result = _trivial_gcd(f, g)
    if result is not None:
        return result

    ring = f.ring
    k = ring.ngens

    # divide out integer content
    cf, f = f.primitive()
    cg, g = g.primitive()
    ch = ring.domain.gcd(cf, cg)

    gamma = ring.domain.gcd(f.LC, g.LC)

    badprimes = ring.domain.one
    for i in range(k):
        badprimes *= ring.domain.gcd(_swap(f, i).LC, _swap(g, i).LC)

    degbound = [min(fdeg, gdeg) for fdeg, gdeg in zip(f.degrees(), g.degrees())]
    contbound = list(degbound)

    m = 1
    p = 1

    while True:
        p = nextprime(p)
        while badprimes % p == 0:
            p = nextprime(p)

        fp = f.trunc_ground(p)
        gp = g.trunc_ground(p)

        try:
            # monic GCD of fp, gp in Z_p[x_0, ..., x_{k-2}, y]
            hp = _modgcd_multivariate_p(fp, gp, p, degbound, contbound)
        except ModularGCDFailed:
            m = 1
            continue

        if hp is None:
            continue

        hp = hp.mul_ground(gamma).trunc_ground(p)
        if m == 1:
            m = p
            hlastm = hp
            continue

        hm = _chinese_remainder_reconstruction_multivariate(hp, hlastm, p, m)
        m *= p

        if not hm == hlastm:
            hlastm = hm
            continue

        h = hm.primitive()[1]
        fquo, frem = f.div(h)
        gquo, grem = g.div(h)
        if not frem and not grem:
            if h.LC < 0:
                ch = -ch
            h = h.mul_ground(ch)
            cff = fquo.mul_ground(cf // ch)
            cfg = gquo.mul_ground(cg // ch)
            return h, cff, cfg


def _gf_div(f, g, p):
    r"""
    Compute `\frac f g` modulo `p` for two univariate polynomials over
    `\mathbb Z_p`.
    """
    ring = f.ring
    densequo, denserem = gf_div(f.to_dense(), g.to_dense(), p, ring.domain)
    return ring.from_dense(densequo), ring.from_dense(denserem)


def _rational_function_reconstruction(c, p, m):
    r"""
    Reconstruct a rational function `\frac a b` in `\mathbb Z_p(t)` from

    .. math::

        c = \frac a b \; \mathrm{mod} \, m,

    where `c` and `m` are polynomials in `\mathbb Z_p[t]` and `m` has
    positive degree.

    The algorithm is based on the Euclidean Algorithm. In general, `m` is
    not irreducible, so it is possible that `b` is not invertible modulo
    `m`. In that case ``None`` is returned.

    Parameters
    ==========

    c : PolyElement
        univariate polynomial in `\mathbb Z[t]`
    p : Integer
        prime number
    m : PolyElement
        modulus, not necessarily irreducible

    Returns
    =======

    frac : FracElement
        either `\frac a b` in `\mathbb Z(t)` or ``None``

    References
    ==========

    1. [Hoeij04]_

    """
    ring = c.ring
    domain = ring.domain
    M = m.degree()
    N = M // 2
    D = M - N - 1

    r0, s0 = m, ring.zero
    r1, s1 = c, ring.one

    while r1.degree() > N:
        quo = _gf_div(r0, r1, p)[0]
        r0, r1 = r1, (r0 - quo*r1).trunc_ground(p)
        s0, s1 = s1, (s0 - quo*s1).trunc_ground(p)

    a, b = r1, s1
    if b.degree() > D or _gf_gcd(b, m, p) != 1:
        return None

    lc = b.LC
    if lc != 1:
        lcinv = domain.invert(lc, p)
        a = a.mul_ground(lcinv).trunc_ground(p)
        b = b.mul_ground(lcinv).trunc_ground(p)

    field = ring.to_field()

    return field(a) / field(b)


def _rational_reconstruction_func_coeffs(hm, p, m, ring, k):
    r"""
    Reconstruct every coefficient `c_h` of a polynomial `h` in
    `\mathbb Z_p(t_k)[t_1, \ldots, t_{k-1}][x, z]` from the corresponding
    coefficient `c_{h_m}` of a polynomial `h_m` in
    `\mathbb Z_p[t_1, \ldots, t_k][x, z] \cong \mathbb Z_p[t_k][t_1, \ldots, t_{k-1}][x, z]`
    such that

    .. math::

        c_{h_m} = c_h \; \mathrm{mod} \, m,

    where `m \in \mathbb Z_p[t]`.

    The reconstruction is based on the Euclidean Algorithm. In general, `m`
    is not irreducible, so it is possible that this fails for some
    coefficient. In that case ``None`` is returned.

    Parameters
    ==========

    hm : PolyElement
        polynomial in `\mathbb Z[t_1, \ldots, t_k][x, z]`
    p : Integer
        prime number, modulus of `\mathbb Z_p`
    m : PolyElement
        modulus, polynomial in `\mathbb Z[t]`, not necessarily irreducible
    ring : PolyRing
        `\mathbb Z(t_k)[t_1, \ldots, t_{k-1}][x, z]`, `h` will be an
        element of this ring
    k : Integer
        index of the parameter `t_k` which will be reconstructed

    Returns
    =======

    h : PolyElement
        reconstructed polynomial in
        `\mathbb Z(t_k)[t_1, \ldots, t_{k-1}][x, z]` or ``None``

    See also
    ========

    _rational_function_reconstruction

    """
    h = ring.zero

    for monom, coeff in hm.iterterms():
        if k == 0:
            coeffh = _rational_function_reconstruction(coeff, p, m)

            if not coeffh:
                return None

        else:
            coeffh = ring.domain.zero
            for mon, c in coeff.drop_to_ground(k).iterterms():
                ch = _rational_function_reconstruction(c, p, m)

                if not ch:
                    return None

                coeffh[mon] = ch

        h[monom] = coeffh

    return h


def _gf_gcdex(f, g, p):
    r"""
    Extended Euclidean Algorithm for two univariate polynomials over
    `\mathbb Z_p`.

    Returns polynomials `s, t` and `h`, such that `h` is the GCD of `f` and
    `g` and `sf + tg = h \; \mathrm{mod} \, p`.

    """
    ring = f.ring
    s, t, h = gf_gcdex(f.to_dense(), g.to_dense(), p, ring.domain)
    return ring.from_dense(s), ring.from_dense(t), ring.from_dense(h)


def _trunc(f, minpoly, p):
    r"""
    Compute the reduced representation of a polynomial `f` in
    `\mathbb Z_p[z] / (\check m_{\alpha}(z))[x]`

    Parameters
    ==========

    f : PolyElement
        polynomial in `\mathbb Z[x, z]`
    minpoly : PolyElement
        polynomial `\check m_{\alpha} \in \mathbb Z[z]`, not necessarily
        irreducible
    p : Integer
        prime number, modulus of `\mathbb Z_p`

    Returns
    =======

    ftrunc : PolyElement
        polynomial in `\mathbb Z[x, z]`, reduced modulo
        `\check m_{\alpha}(z)` and `p`

    """
    ring = f.ring
    minpoly = minpoly.set_ring(ring)
    p_ = ring.ground_new(p)

    return f.trunc_ground(p).rem([minpoly, p_]).trunc_ground(p)


def _euclidean_algorithm(f, g, minpoly, p):
    r"""
    Compute the monic GCD of two univariate polynomials in
    `\mathbb{Z}_p[z]/(\check m_{\alpha}(z))[x]` with the Euclidean
    Algorithm.

    In general, `\check m_{\alpha}(z)` is not irreducible, so it is possible
    that some leading coefficient is not invertible modulo
    `\check m_{\alpha}(z)`. In that case ``None`` is returned.

    Parameters
    ==========

    f, g : PolyElement
        polynomials in `\mathbb Z[x, z]`
    minpoly : PolyElement
        polynomial in `\mathbb Z[z]`, not necessarily irreducible
    p : Integer
        prime number, modulus of `\mathbb Z_p`

    Returns
    =======

    h : PolyElement
        GCD of `f` and `g` in `\mathbb Z[z, x]` or ``None``, coefficients
        are in `\left[ -\frac{p-1} 2, \frac{p-1} 2 \right]`

    """
    ring = f.ring

    f = _trunc(f, minpoly, p)
    g = _trunc(g, minpoly, p)

    while g:
        rem = f
        deg = g.degree(0) # degree in x
        lcinv, _, gcd = _gf_gcdex(ring.dmp_LC(g), minpoly, p)

        if not gcd == 1:
            return None

        while True:
            degrem = rem.degree(0) # degree in x
            if degrem < deg:
                break
            quo = (lcinv * ring.dmp_LC(rem)).set_ring(ring)
            rem = _trunc(rem - g.mul_monom((degrem - deg, 0))*quo, minpoly, p)

        f = g
        g = rem

    lcfinv = _gf_gcdex(ring.dmp_LC(f), minpoly, p)[0].set_ring(ring)

    return _trunc(f * lcfinv, minpoly, p)


def _trial_division(f, h, minpoly, p=None):
    r"""
    Check if `h` divides `f` in
    `\mathbb K[t_1, \ldots, t_k][z]/(m_{\alpha}(z))`, where `\mathbb K` is
    either `\mathbb Q` or `\mathbb Z_p`.

    This algorithm is based on pseudo division and does not use any
    fractions. By default `\mathbb K` is `\mathbb Q`, if a prime number `p`
    is given, `\mathbb Z_p` is chosen instead.

    Parameters
    ==========

    f, h : PolyElement
        polynomials in `\mathbb Z[t_1, \ldots, t_k][x, z]`
    minpoly : PolyElement
        polynomial `m_{\alpha}(z)` in `\mathbb Z[t_1, \ldots, t_k][z]`
    p : Integer or None
        if `p` is given, `\mathbb K` is set to `\mathbb Z_p` instead of
        `\mathbb Q`, default is ``None``

    Returns
    =======

    rem : PolyElement
        remainder of `\frac f h`

    References
    ==========

    .. [1] [Hoeij02]_

    """
    ring = f.ring

    zxring = ring.clone(symbols=(ring.symbols[1], ring.symbols[0]))

    minpoly = minpoly.set_ring(ring)

    rem = f

    degrem = rem.degree()
    degh = h.degree()
    degm = minpoly.degree(1)

    lch = _LC(h).set_ring(ring)
    lcm = minpoly.LC

    while rem and degrem >= degh:
        # polynomial in Z[t_1, ..., t_k][z]
        lcrem = _LC(rem).set_ring(ring)
        rem = rem*lch - h.mul_monom((degrem - degh, 0))*lcrem
        if p:
            rem = rem.trunc_ground(p)
        degrem = rem.degree(1)

        while rem and degrem >= degm:
            # polynomial in Z[t_1, ..., t_k][x]
            lcrem = _LC(rem.set_ring(zxring)).set_ring(ring)
            rem = rem.mul_ground(lcm) - minpoly.mul_monom((0, degrem - degm))*lcrem
            if p:
                rem = rem.trunc_ground(p)
            degrem = rem.degree(1)

        degrem = rem.degree()

    return rem


def _evaluate_ground(f, i, a):
    r"""
    Evaluate a polynomial `f` at `a` in the `i`-th variable of the ground
    domain.
    """
    ring = f.ring.clone(domain=f.ring.domain.ring.drop(i))
    fa = ring.zero

    for monom, coeff in f.iterterms():
        fa[monom] = coeff.evaluate(i, a)

    return fa


def _func_field_modgcd_p(f, g, minpoly, p):
    r"""
    Compute the GCD of two polynomials `f` and `g` in
    `\mathbb Z_p(t_1, \ldots, t_k)[z]/(\check m_\alpha(z))[x]`.

    The algorithm reduces the problem step by step by evaluating the
    polynomials `f` and `g` at `t_k = a` for suitable `a \in \mathbb Z_p`
    and then calls itself recursively to compute the GCD in
    `\mathbb Z_p(t_1, \ldots, t_{k-1})[z]/(\check m_\alpha(z))[x]`. If these
    recursive calls are successful, the GCD over `k` variables is
    interpolated, otherwise the algorithm returns ``None``. After
    interpolation, Rational Function Reconstruction is used to obtain the
    correct coefficients. If this fails, a new evaluation point has to be
    chosen, otherwise the desired polynomial is obtained by clearing
    denominators. The result is verified with a fraction free trial
    division.

    Parameters
    ==========

    f, g : PolyElement
        polynomials in `\mathbb Z[t_1, \ldots, t_k][x, z]`
    minpoly : PolyElement
        polynomial in `\mathbb Z[t_1, \ldots, t_k][z]`, not necessarily
        irreducible
    p : Integer
        prime number, modulus of `\mathbb Z_p`

    Returns
    =======

    h : PolyElement
        primitive associate in `\mathbb Z[t_1, \ldots, t_k][x, z]` of the
        GCD of the polynomials `f` and `g`  or ``None``, coefficients are
        in `\left[ -\frac{p-1} 2, \frac{p-1} 2 \right]`

    References
    ==========

    1. [Hoeij04]_

    """
    ring = f.ring
    domain = ring.domain # Z[t_1, ..., t_k]

    if isinstance(domain, PolynomialRing):
        k = domain.ngens
    else:
        return _euclidean_algorithm(f, g, minpoly, p)

    if k == 1:
        qdomain = domain.ring.to_field()
    else:
        qdomain = domain.ring.drop_to_ground(k - 1)
        qdomain = qdomain.clone(domain=qdomain.domain.ring.to_field())

    qring = ring.clone(domain=qdomain) # = Z(t_k)[t_1, ..., t_{k-1}][x, z]

    n = 1
    d = 1

    # polynomial in Z_p[t_1, ..., t_k][z]
    gamma = ring.dmp_LC(f) * ring.dmp_LC(g)
    # polynomial in Z_p[t_1, ..., t_k]
    delta = minpoly.LC

    evalpoints = []
    heval = []
    LMlist = []
    points = list(range(p))

    while points:
        a = random.sample(points, 1)[0]
        points.remove(a)

        if k == 1:
            test = delta.evaluate(k-1, a) % p == 0
        else:
            test = delta.evaluate(k-1, a).trunc_ground(p) == 0

        if test:
            continue

        gammaa = _evaluate_ground(gamma, k-1, a)
        minpolya = _evaluate_ground(minpoly, k-1, a)

        if gammaa.rem([minpolya, gammaa.ring(p)]) == 0:
            continue

        fa = _evaluate_ground(f, k-1, a)
        ga = _evaluate_ground(g, k-1, a)

        # polynomial in Z_p[x, t_1, ..., t_{k-1}, z]/(minpoly)
