        result = dmp_add(result, coeff, v, K)

    return result


def _rec_eval_in(g, a, v, i, j, K):
    """Recursive helper for :func:`dmp_eval_in`."""
    if i == j:
        return dmp_eval(g, a, v, K)

    v, i = v - 1, i + 1

    return dmp_strip([ _rec_eval_in(c, a, v, i, j, K) for c in g ], v)


def dmp_eval_in(f, a, j, u, K):
    """
    Evaluate a polynomial at ``x_j = a`` in ``K[X]`` using the Horner scheme.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = 2*x*y + 3*x + y + 2

    >>> R.dmp_eval_in(f, 2, 0)
    5*y + 8
    >>> R.dmp_eval_in(f, 2, 1)
    7*x + 4

    """
    if j < 0 or j > u:
        raise IndexError("0 <= j <= %s expected, got %s" % (u, j))

    return _rec_eval_in(f, a, u, 0, j, K)


def _rec_eval_tail(g, i, A, u, K):
    """Recursive helper for :func:`dmp_eval_tail`."""
    if i == u:
        return dup_eval(g, A[-1], K)
    else:
        h = [ _rec_eval_tail(c, i + 1, A, u, K) for c in g ]

        if i < u - len(A) + 1:
            return h
        else:
            return dup_eval(h, A[-u + i - 1], K)


def dmp_eval_tail(f, A, u, K):
    """
    Evaluate a polynomial at ``x_j = a_j, ...`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = 2*x*y + 3*x + y + 2

    >>> R.dmp_eval_tail(f, [2])
    7*x + 4
    >>> R.dmp_eval_tail(f, [2, 2])
    18

    """
    if not A:
        return f

    if dmp_zero_p(f, u):
        return dmp_zero(u - len(A))

    e = _rec_eval_tail(f, 0, A, u, K)

    if u == len(A) - 1:
        return e
    else:
        return dmp_strip(e, u - len(A))


def _rec_diff_eval(g, m, a, v, i, j, K):
    """Recursive helper for :func:`dmp_diff_eval`."""
    if i == j:
        return dmp_eval(dmp_diff(g, m, v, K), a, v, K)

    v, i = v - 1, i + 1

    return dmp_strip([ _rec_diff_eval(c, m, a, v, i, j, K) for c in g ], v)


def dmp_diff_eval_in(f, m, a, j, u, K):
    """
    Differentiate and evaluate a polynomial in ``x_j`` at ``a`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = x*y**2 + 2*x*y + 3*x + 2*y**2 + 3*y + 1

    >>> R.dmp_diff_eval_in(f, 1, 2, 0)
    y**2 + 2*y + 3
    >>> R.dmp_diff_eval_in(f, 1, 2, 1)
    6*x + 11

    """
    if j > u:
        raise IndexError("-%s <= j < %s expected, got %s" % (u, u, j))
    if not j:
        return dmp_eval(dmp_diff(f, m, u, K), a, u, K)

    return _rec_diff_eval(f, m, a, u, 0, j, K)


def dup_trunc(f, p, K):
    """
    Reduce a ``K[x]`` polynomial modulo a constant ``p`` in ``K``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_trunc(2*x**3 + 3*x**2 + 5*x + 7, ZZ(3))
    -x**3 - x + 1

    """
    if K.is_ZZ:
        g = []

        for c in f:
            c = c % p

            if c > p // 2:
                g.append(c - p)
            else:
                g.append(c)
    elif K.is_FiniteField:
        # XXX: python-flint's nmod does not support %
        pi = int(p)
        g = [ K(int(c) % pi) for c in f ]
    else:
        g = [ c % p for c in f ]

    return dup_strip(g)


def dmp_trunc(f, p, u, K):
    """
    Reduce a ``K[X]`` polynomial modulo a polynomial ``p`` in ``K[Y]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = 3*x**2*y + 8*x**2 + 5*x*y + 6*x + 2*y + 3
    >>> g = (y - 1).drop(x)

    >>> R.dmp_trunc(f, g)
    11*x**2 + 11*x + 5

    """
    return dmp_strip([ dmp_rem(c, p, u - 1, K) for c in f ], u)


def dmp_ground_trunc(f, p, u, K):
    """
    Reduce a ``K[X]`` polynomial modulo a constant ``p`` in ``K``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = 3*x**2*y + 8*x**2 + 5*x*y + 6*x + 2*y + 3

    >>> R.dmp_ground_trunc(f, ZZ(3))
    -x**2 - x*y - y

    """
    if not u:
        return dup_trunc(f, p, K)

    v = u - 1

    return dmp_strip([ dmp_ground_trunc(c, p, v, K) for c in f ], u)


def dup_monic(f, K):
    """
    Divide all coefficients by ``LC(f)`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x = ring("x", ZZ)
    >>> R.dup_monic(3*x**2 + 6*x + 9)
    x**2 + 2*x + 3

    >>> R, x = ring("x", QQ)
    >>> R.dup_monic(3*x**2 + 4*x + 2)
    x**2 + 4/3*x + 2/3

    """
    if not f:
        return f

    lc = dup_LC(f, K)

    if K.is_one(lc):
        return f
    else:
        return dup_exquo_ground(f, lc, K)


def dmp_ground_monic(f, u, K):
    """
    Divide all coefficients by ``LC(f)`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x,y = ring("x,y", ZZ)
    >>> f = 3*x**2*y + 6*x**2 + 3*x*y + 9*y + 3

    >>> R.dmp_ground_monic(f)
    x**2*y + 2*x**2 + x*y + 3*y + 1

    >>> R, x,y = ring("x,y", QQ)
    >>> f = 3*x**2*y + 8*x**2 + 5*x*y + 6*x + 2*y + 3

    >>> R.dmp_ground_monic(f)
    x**2*y + 8/3*x**2 + 5/3*x*y + 2*x + 2/3*y + 1

    """
    if not u:
        return dup_monic(f, K)

    if dmp_zero_p(f, u):
        return f

    lc = dmp_ground_LC(f, u, K)

    if K.is_one(lc):
        return f
    else:
        return dmp_exquo_ground(f, lc, u, K)


def dup_content(f, K):
    """
    Compute the GCD of coefficients of ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x = ring("x", ZZ)
    >>> f = 6*x**2 + 8*x + 12

    >>> R.dup_content(f)
    2

    >>> R, x = ring("x", QQ)
    >>> f = 6*x**2 + 8*x + 12

    >>> R.dup_content(f)
    2

    """
    from sympy.polys.domains import QQ

    if not f:
        return K.zero

    cont = K.zero

    if K == QQ:
        for c in f:
            cont = K.gcd(cont, c)
    else:
        for c in f:
            cont = K.gcd(cont, c)

            if K.is_one(cont):
                break

    return cont


def dmp_ground_content(f, u, K):
    """
    Compute the GCD of coefficients of ``f`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x,y = ring("x,y", ZZ)
    >>> f = 2*x*y + 6*x + 4*y + 12

    >>> R.dmp_ground_content(f)
    2

    >>> R, x,y = ring("x,y", QQ)
    >>> f = 2*x*y + 6*x + 4*y + 12

    >>> R.dmp_ground_content(f)
    2

    """
    from sympy.polys.domains import QQ

    if not u:
        return dup_content(f, K)

    if dmp_zero_p(f, u):
        return K.zero

    cont, v = K.zero, u - 1

    if K == QQ:
        for c in f:
            cont = K.gcd(cont, dmp_ground_content(c, v, K))
    else:
        for c in f:
            cont = K.gcd(cont, dmp_ground_content(c, v, K))

            if K.is_one(cont):
                break

    return cont


def dup_primitive(f, K):
    """
    Compute content and the primitive form of ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x = ring("x", ZZ)
    >>> f = 6*x**2 + 8*x + 12

    >>> R.dup_primitive(f)
    (2, 3*x**2 + 4*x + 6)

    >>> R, x = ring("x", QQ)
    >>> f = 6*x**2 + 8*x + 12

    >>> R.dup_primitive(f)
    (2, 3*x**2 + 4*x + 6)

    """
    if not f:
        return K.zero, f

    cont = dup_content(f, K)

    if K.is_one(cont):
        return cont, f
    else:
        return cont, dup_quo_ground(f, cont, K)


def dmp_ground_primitive(f, u, K):
    """
    Compute content and the primitive form of ``f`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x,y = ring("x,y", ZZ)
    >>> f = 2*x*y + 6*x + 4*y + 12

    >>> R.dmp_ground_primitive(f)
    (2, x*y + 3*x + 2*y + 6)

    >>> R, x,y = ring("x,y", QQ)
    >>> f = 2*x*y + 6*x + 4*y + 12

    >>> R.dmp_ground_primitive(f)
    (2, x*y + 3*x + 2*y + 6)

    """
    if not u:
        return dup_primitive(f, K)

    if dmp_zero_p(f, u):
        return K.zero, f

    cont = dmp_ground_content(f, u, K)

    if K.is_one(cont):
        return cont, f
    else:
        return cont, dmp_quo_ground(f, cont, u, K)


def dup_extract(f, g, K):
    """
    Extract common content from a pair of polynomials in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_extract(6*x**2 + 12*x + 18, 4*x**2 + 8*x + 12)
    (2, 3*x**2 + 6*x + 9, 2*x**2 + 4*x + 6)

    """
    fc = dup_content(f, K)
    gc = dup_content(g, K)

    gcd = K.gcd(fc, gc)

    if not K.is_one(gcd):
        f = dup_quo_ground(f, gcd, K)
        g = dup_quo_ground(g, gcd, K)

    return gcd, f, g


def dmp_ground_extract(f, g, u, K):
    """
    Extract common content from a pair of polynomials in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_ground_extract(6*x*y + 12*x + 18, 4*x*y + 8*x + 12)
    (2, 3*x*y + 6*x + 9, 2*x*y + 4*x + 6)

    """
    fc = dmp_ground_content(f, u, K)
    gc = dmp_ground_content(g, u, K)

    gcd = K.gcd(fc, gc)

    if not K.is_one(gcd):
        f = dmp_quo_ground(f, gcd, u, K)
        g = dmp_quo_ground(g, gcd, u, K)

    return gcd, f, g


def dup_real_imag(f, K):
    """
    Find ``f1`` and ``f2``, such that ``f(x+I*y) = f1(x,y) + f2(x,y)*I``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dup_real_imag(x**3 + x**2 + x + 1)
    (x**3 + x**2 - 3*x*y**2 + x - y**2 + 1, 3*x**2*y + 2*x*y - y**3 + y)

    >>> from sympy.abc import x, y, z
    >>> from sympy import I
    >>> (z**3 + z**2 + z + 1).subs(z, x+I*y).expand().collect(I)
    x**3 + x**2 - 3*x*y**2 + x - y**2 + I*(3*x**2*y + 2*x*y - y**3 + y) + 1

    """
    if not K.is_ZZ and not K.is_QQ:
        raise DomainError("computing real and imaginary parts is not supported over %s" % K)

    f1 = dmp_zero(1)
    f2 = dmp_zero(1)

    if not f:
        return f1, f2

    g = [[[K.one, K.zero]], [[K.one], []]]
    h = dmp_ground(f[0], 2)

    for c in f[1:]:
        h = dmp_mul(h, g, 2, K)
        h = dmp_add_term(h, dmp_ground(c, 1), 0, 2, K)

    H = dup_to_raw_dict(h)

    for k, h in H.items():
        m = k % 4

        if not m:
            f1 = dmp_add(f1, h, 1, K)
        elif m == 1:
            f2 = dmp_add(f2, h, 1, K)
        elif m == 2:
            f1 = dmp_sub(f1, h, 1, K)
        else:
            f2 = dmp_sub(f2, h, 1, K)

    return f1, f2


def dup_mirror(f, K):
    """
    Evaluate efficiently the composition ``f(-x)`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_mirror(x**3 + 2*x**2 - 4*x + 2)
    -x**3 + 2*x**2 + 4*x + 2

    """
    f = list(f)

    for i in range(len(f) - 2, -1, -2):
        f[i] = -f[i]

    return f


def dup_scale(f, a, K):
    """
    Evaluate efficiently composition ``f(a*x)`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_scale(x**2 - 2*x + 1, ZZ(2))
    4*x**2 - 4*x + 1

    """
    f, n, b = list(f), len(f) - 1, a

    for i in range(n - 1, -1, -1):
        f[i], b = b*f[i], b*a

    return f


def dup_shift(f, a, K):
    """
    Evaluate efficiently Taylor shift ``f(x + a)`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_shift(x**2 - 2*x + 1, ZZ(2))
    x**2 + 2*x + 1

    """
    f, n = list(f), len(f) - 1

    for i in range(n, 0, -1):
        for j in range(0, i):
            f[j + 1] += a*f[j]

    return f


def dmp_shift(f, a, u, K):
    """
    Evaluate efficiently Taylor shift ``f(X + A)`` in ``K[X]``.

    Examples
    ========

    >>> from sympy import symbols, ring, ZZ
    >>> x, y = symbols('x y')
    >>> R, _, _ = ring([x, y], ZZ)

    >>> p = x**2*y + 2*x*y + 3*x + 4*y + 5

    >>> R.dmp_shift(R(p), [ZZ(1), ZZ(2)])
    x**2*y + 2*x**2 + 4*x*y + 11*x + 7*y + 22

    >>> p.subs({x: x + 1, y: y + 2}).expand()
    x**2*y + 2*x**2 + 4*x*y + 11*x + 7*y + 22
    """
    if not u:
        return dup_shift(f, a[0], K)

    if dmp_zero_p(f, u):
        return f

    a0, a1 = a[0], a[1:]

    if any(a1):
        f = [ dmp_shift(c, a1, u-1, K) for c in f ]
    else:
        f = list(f)

    if a0:
        n = len(f) - 1

        for i in range(n, 0, -1):
            for j in range(0, i):
                afj = dmp_mul_ground(f[j], a0, u-1, K)
                f[j + 1] = dmp_add(f[j + 1], afj, u-1, K)

    return dmp_strip(f, u)


def dup_transform(f, p, q, K):
    """
    Evaluate functional transformation ``q**n * f(p/q)`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_transform(x**2 - 2*x + 1, x**2 + 1, x - 1)
    x**4 - 2*x**3 + 5*x**2 - 4*x + 4

    """
    if not f:
        return []

    n = len(f) - 1
    h, Q = [f[0]], [[K.one]]

    for i in range(0, n):
        Q.append(dup_mul(Q[-1], q, K))

    for c, q in zip(f[1:], Q[1:]):
        h = dup_mul(h, p, K)
        q = dup_mul_ground(q, c, K)
        h = dup_add(h, q, K)

    return h


def dup_compose(f, g, K):
    """
    Evaluate functional composition ``f(g)`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_compose(x**2 + x, x - 1)
    x**2 - x

    """
    if len(g) <= 1:
        return dup_strip([dup_eval(f, dup_LC(g, K), K)])

    if not f:
        return []

    h = [f[0]]

    for c in f[1:]:
        h = dup_mul(h, g, K)
        h = dup_add_term(h, c, 0, K)

    return h


def dmp_compose(f, g, u, K):
    """
    Evaluate functional composition ``f(g)`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_compose(x*y + 2*x + y, y)
    y**2 + 3*y

    """
    if not u:
        return dup_compose(f, g, K)

    if dmp_zero_p(f, u):
        return f

    h = [f[0]]

    for c in f[1:]:
        h = dmp_mul(h, g, u, K)
        h = dmp_add_term(h, c, 0, u, K)

    return h


def _dup_right_decompose(f, s, K):
    """Helper function for :func:`_dup_decompose`."""
    n = len(f) - 1
    lc = dup_LC(f, K)

    f = dup_to_raw_dict(f)
    g = { s: K.one }

    r = n // s

    for i in range(1, s):
        coeff = K.zero

        for j in range(0, i):
            if not n + j - i in f:
                continue

            if not s - j in g:
                continue

            fc, gc = f[n + j - i], g[s - j]
            coeff += (i - r*j)*fc*gc

        g[s - i] = K.quo(coeff, i*r*lc)

    return dup_from_raw_dict(g, K)


def _dup_left_decompose(f, h, K):
    """Helper function for :func:`_dup_decompose`."""
    g, i = {}, 0

    while f:
        q, r = dup_div(f, h, K)

        if dup_degree(r) > 0:
            return None
        else:
            g[i] = dup_LC(r, K)
            f, i = q, i + 1

    return dup_from_raw_dict(g, K)


def _dup_decompose(f, K):
    """Helper function for :func:`dup_decompose`."""
    df = len(f) - 1

    for s in range(2, df):
        if df % s != 0:
            continue

        h = _dup_right_decompose(f, s, K)

        if h is not None:
            g = _dup_left_decompose(f, h, K)

            if g is not None:
                return g, h

    return None


def dup_decompose(f, K):
    """
    Computes functional decomposition of ``f`` in ``K[x]``.

    Given a univariate polynomial ``f`` with coefficients in a field of
    characteristic zero, returns list ``[f_1, f_2, ..., f_n]``, where::

              f = f_1 o f_2 o ... f_n = f_1(f_2(... f_n))

    and ``f_2, ..., f_n`` are monic and homogeneous polynomials of at
    least second degree.

    Unlike factorization, complete functional decompositions of
    polynomials are not unique, consider examples:

    1. ``f o g = f(x + b) o (g - b)``
    2. ``x**n o x**m = x**m o x**n``
    3. ``T_n o T_m = T_m o T_n``

    where ``T_n`` and ``T_m`` are Chebyshev polynomials.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_decompose(x**4 - 2*x**3 + x**2)
    [x**2, x**2 - x]

    References
    ==========

    .. [1] [Kozen89]_

    """
    F = []

    while True:
        result = _dup_decompose(f, K)

        if result is not None:
            f, h = result
            F = [h] + F
        else:
            break

    return [f] + F


def dmp_alg_inject(f, u, K):
    """
    Convert polynomial from ``K(a)[X]`` to ``K[a,X]``.

    Examples
    ========

    >>> from sympy.polys.densetools import dmp_alg_inject
    >>> from sympy import QQ, sqrt

    >>> K = QQ.algebraic_field(sqrt(2))

    >>> p = [K.from_sympy(sqrt(2)), K.zero, K.one]
    >>> P, lev, dom = dmp_alg_inject(p, 0, K)
    >>> P
    [[1, 0, 0], [1]]
    >>> lev
    1
    >>> dom
    QQ

    """
    if K.is_GaussianRing or K.is_GaussianField:
        return _dmp_alg_inject_gaussian(f, u, K)
    elif K.is_Algebraic:
        return _dmp_alg_inject_alg(f, u, K)
    else:
        raise DomainError('computation can be done only in an algebraic domain')


def _dmp_alg_inject_gaussian(f, u, K):
    """Helper function for :func:`dmp_alg_inject`."""
    f, h = dmp_to_dict(f, u), {}

    for f_monom, g in f.items():
        x, y = g.x, g.y
        if x:
            h[(0,) + f_monom] = x
        if y:
            h[(1,) + f_monom] = y

    F = dmp_from_dict(h, u + 1, K.dom)

    return F, u + 1, K.dom


def _dmp_alg_inject_alg(f, u, K):
    """Helper function for :func:`dmp_alg_inject`."""
    f, h = dmp_to_dict(f, u), {}

    for f_monom, g in f.items():
        for g_monom, c in g.to_dict().items():
            h[g_monom + f_monom] = c

    F = dmp_from_dict(h, u + 1, K.dom)

    return F, u + 1, K.dom


def dmp_lift(f, u, K):
    """
    Convert algebraic coefficients to integers in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> from sympy import I

    >>> K = QQ.algebraic_field(I)
    >>> R, x = ring("x", K)

    >>> f = x**2 + K([QQ(1), QQ(0)])*x + K([QQ(2), QQ(0)])

    >>> R.dmp_lift(f)
    x**4 + x**2 + 4*x + 4

    """
    # Circular import. Probably dmp_lift should be moved to euclidtools
    from .euclidtools import dmp_resultant

    F, v, K2 = dmp_alg_inject(f, u, K)

    p_a = K.mod.to_list()
    P_A = dmp_include(p_a, list(range(1, v + 1)), 0, K2)

    return dmp_resultant(F, P_A, v, K2)


def dup_sign_variations(f, K):
    """
    Compute the number of sign variations of ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_sign_variations(x**4 - x**2 - x + 1)
    2

    """
    def is_negative_sympy(a):
        if not a:
            # XXX: requires zero equivalence testing in the domain
            return False
        else:
            # XXX: This is inefficient. It should not be necessary to use a
            # symbolic expression here at least for algebraic fields. If the
            # domain elements can be numerically evaluated to real values with
            # precision then this should work. We first need to rule out zero
            # elements though.
            return bool(K.to_sympy(a) < 0)

    # XXX: There should be a way to check for real numeric domains and
    # Domain.is_negative should be fixed to handle all real numeric domains.
    # It should not be necessary to special case all these different domains
    # in this otherwise generic function.
    if K.is_ZZ or K.is_QQ or K.is_RR:
        is_negative = K.is_negative
    elif K.is_AlgebraicField and K.ext.is_comparable:
        is_negative = is_negative_sympy
    elif ((K.is_PolynomialRing or K.is_FractionField) and len(K.symbols) == 1 and
          (K.dom.is_ZZ or K.dom.is_QQ or K.is_AlgebraicField) and
          K.symbols[0].is_transcendental and K.symbols[0].is_comparable):
        # We can handle a polynomial ring like QQ[E] if there is a single
        # transcendental generator because then zero equivalence is assured.
        is_negative = is_negative_sympy
    else:
        raise DomainError("sign variation counting not supported over %s" % K)

    prev, k = K.zero, 0

    for coeff in f:
        if is_negative(coeff*prev):
            k += 1

        if coeff:
            prev = coeff

    return k


def dup_clear_denoms(f, K0, K1=None, convert=False):
    """
    Clear denominators, i.e. transform ``K_0`` to ``K_1``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    >>> f = QQ(1,2)*x + QQ(1,3)

    >>> R.dup_clear_denoms(f, convert=False)
    (6, 3*x + 2)
    >>> R.dup_clear_denoms(f, convert=True)
    (6, 3*x + 2)

    """
    if K1 is None:
        if K0.has_assoc_Ring:
            K1 = K0.get_ring()
        else:
            K1 = K0

    common = K1.one

    for c in f:
        common = K1.lcm(common, K0.denom(c))

    if K1.is_one(common):
        if not convert:
            return common, f
        else:
            return common, dup_convert(f, K0, K1)

    # Use quo rather than exquo to handle inexact domains by discarding the
    # remainder.
    f = [K0.numer(c)*K1.quo(common, K0.denom(c)) for c in f]

    if not convert:
        return common, dup_convert(f, K1, K0)
    else:
        return common, f


def _rec_clear_denoms(g, v, K0, K1):
    """Recursive helper for :func:`dmp_clear_denoms`."""
    common = K1.one

    if not v:
        for c in g:
            common = K1.lcm(common, K0.denom(c))
    else:
        w = v - 1

        for c in g:
            common = K1.lcm(common, _rec_clear_denoms(c, w, K0, K1))

    return common


def dmp_clear_denoms(f, u, K0, K1=None, convert=False):
    """
    Clear denominators, i.e. transform ``K_0`` to ``K_1``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x,y = ring("x,y", QQ)

    >>> f = QQ(1,2)*x + QQ(1,3)*y + 1

    >>> R.dmp_clear_denoms(f, convert=False)
    (6, 3*x + 2*y + 6)
    >>> R.dmp_clear_denoms(f, convert=True)
    (6, 3*x + 2*y + 6)

    """
    if not u:
        return dup_clear_denoms(f, K0, K1, convert=convert)

    if K1 is None:
        if K0.has_assoc_Ring:
            K1 = K0.get_ring()
        else:
            K1 = K0

    common = _rec_clear_denoms(f, u, K0, K1)

    if not K1.is_one(common):
        f = dmp_mul_ground(f, common, u, K0)

    if not convert:
        return common, f
    else:
        return common, dmp_convert(f, u, K0, K1)


def dup_revert(f, n, K):
    """
    Compute ``f**(-1)`` mod ``x**n`` using Newton iteration.

    This function computes first ``2**n`` terms of a polynomial that
    is a result of inversion of a polynomial modulo ``x**n``. This is
    useful to efficiently compute series expansion of ``1/f``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    >>> f = -QQ(1,720)*x**6 + QQ(1,24)*x**4 - QQ(1,2)*x**2 + 1

    >>> R.dup_revert(f, 8)
    61/720*x**6 + 5/24*x**4 + 1/2*x**2 + 1

    """
    g = [K.revert(dup_TC(f, K))]
    h = [K.one, K.zero, K.zero]

    N = int(_ceil(_log2(n)))

    for i in range(1, N + 1):
        a = dup_mul_ground(g, K(2), K)
        b = dup_mul(f, dup_sqr(g, K), K)
        g = dup_rem(dup_sub(a, b, K), h, K)
        h = dup_lshift(h, dup_degree(h), K)

    return g


def dmp_revert(f, g, u, K):
    """
    Compute ``f**(-1)`` mod ``x**n`` using Newton iteration.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x,y = ring("x,y", QQ)

    """
    if not u:
        return dup_revert(f, g, K)
    else:
        raise MultivariatePolynomialError(f, g)
<|endoftext|>"""
This module contains functions for two multivariate resultants. These
are:

- Dixon's resultant.
- Macaulay's resultant.

Multivariate resultants are used to identify whether a multivariate
system has common roots. That is when the resultant is equal to zero.
"""
from math import prod

from sympy.core.mul import Mul
from sympy.matrices.dense import (Matrix, diag)
from sympy.polys.polytools import (Poly, degree_list, rem)
from sympy.simplify.simplify import simplify
from sympy.tensor.indexed import IndexedBase
from sympy.polys.monomials import itermonomials, monomial_deg
from sympy.polys.orderings import monomial_key
from sympy.polys.polytools import poly_from_expr, total_degree
from sympy.functions.combinatorial.factorials import binomial
from itertools import combinations_with_replacement
from sympy.utilities.exceptions import sympy_deprecation_warning

class DixonResultant():
    """
    A class for retrieving the Dixon's resultant of a multivariate
    system.

    Examples
    ========

    >>> from sympy import symbols

    >>> from sympy.polys.multivariate_resultants import DixonResultant
    >>> x, y = symbols('x, y')

    >>> p = x + y
    >>> q = x ** 2 + y ** 3
    >>> h = x ** 2 + y

    >>> dixon = DixonResultant(variables=[x, y], polynomials=[p, q, h])
    >>> poly = dixon.get_dixon_polynomial()
    >>> matrix = dixon.get_dixon_matrix(polynomial=poly)
    >>> matrix
    Matrix([
    [ 0,  0, -1,  0, -1],
    [ 0, -1,  0, -1,  0],
    [-1,  0,  1,  0,  0],
    [ 0, -1,  0,  0,  1],
    [-1,  0,  0,  1,  0]])
    >>> matrix.det()
    0

    See Also
    ========

    Notebook in examples: sympy/example/notebooks.

    References
    ==========

    .. [1] [Kapur1994]_
    .. [2] [Palancz08]_

    """

    def __init__(self, polynomials, variables):
        """
        A class that takes two lists, a list of polynomials and list of
        variables. Returns the Dixon matrix of the multivariate system.

        Parameters
        ----------
        polynomials : list of polynomials
            A  list of m n-degree polynomials
        variables: list
            A list of all n variables
        """
        self.polynomials = polynomials
        self.variables = variables

        self.n = len(self.variables)
        self.m = len(self.polynomials)

        a = IndexedBase("alpha")
        # A list of n alpha variables (the replacing variables)
        self.dummy_variables = [a[i] for i in range(self.n)]

        # A list of the d_max of each variable.
        self._max_degrees = [max(degree_list(poly)[i] for poly in self.polynomials)
            for i in range(self.n)]

    @property
    def max_degrees(self):
        sympy_deprecation_warning(
            """
            The max_degrees property of DixonResultant is deprecated.
            """,
            deprecated_since_version="1.5",
            active_deprecations_target="deprecated-dixonresultant-properties",
        )
        return self._max_degrees

    def get_dixon_polynomial(self):
        r"""
        Returns
        =======

        dixon_polynomial: polynomial
            Dixon's polynomial is calculated as:

            delta = Delta(A) / ((x_1 - a_1) ... (x_n - a_n)) where,

            A =  |p_1(x_1,... x_n), ..., p_n(x_1,... x_n)|
                 |p_1(a_1,... x_n), ..., p_n(a_1,... x_n)|
                 |...             , ...,              ...|
                 |p_1(a_1,... a_n), ..., p_n(a_1,... a_n)|
        """
        if self.m != (self.n + 1):
            raise ValueError('Method invalid for given combination.')

        # First row
        rows = [self.polynomials]

        temp = list(self.variables)

        for idx in range(self.n):
            temp[idx] = self.dummy_variables[idx]
            substitution = dict(zip(self.variables, temp))
            rows.append([f.subs(substitution) for f in self.polynomials])

        A = Matrix(rows)

        terms = zip(self.variables, self.dummy_variables)
        product_of_differences = Mul(*[a - b for a, b in terms])
        dixon_polynomial = (A.det() / product_of_differences).factor()

        return poly_from_expr(dixon_polynomial, self.dummy_variables)[0]

    def get_upper_degree(self):
        sympy_deprecation_warning(
            """
            The get_upper_degree() method of DixonResultant is deprecated. Use
            get_max_degrees() instead.
            """,
            deprecated_since_version="1.5",
            active_deprecations_target="deprecated-dixonresultant-properties"
        )
        list_of_products = [self.variables[i] ** self._max_degrees[i]
                            for i in range(self.n)]
        product = prod(list_of_products)
        product = Poly(product).monoms()

        return monomial_deg(*product)

    def get_max_degrees(self, polynomial):
        r"""
        Returns a list of the maximum degree of each variable appearing
        in the coefficients of the Dixon polynomial. The coefficients are
        viewed as polys in $x_1, x_2, \dots, x_n$.
        """
        deg_lists = [degree_list(Poly(poly, self.variables))
                     for poly in polynomial.coeffs()]

        max_degrees = [max(degs) for degs in zip(*deg_lists)]

        return max_degrees

    def get_dixon_matrix(self, polynomial):
        r"""
        Construct the Dixon matrix from the coefficients of polynomial
        \alpha. Each coefficient is viewed as a polynomial of x_1, ...,
        x_n.
        """

        max_degrees = self.get_max_degrees(polynomial)

        # list of column headers of the Dixon matrix.
        monomials = itermonomials(self.variables, max_degrees)
        monomials = sorted(monomials, reverse=True,
                           key=monomial_key('lex', self.variables))

        dixon_matrix = Matrix([[Poly(c, *self.variables).coeff_monomial(m)
                                for m in monomials]
                                for c in polynomial.coeffs()])

        # remove columns if needed
        if dixon_matrix.shape[0] != dixon_matrix.shape[1]:
            keep = [column for column in range(dixon_matrix.shape[-1])
                    if any(element != 0 for element
                        in dixon_matrix[:, column])]

            dixon_matrix = dixon_matrix[:, keep]

        return dixon_matrix

    def KSY_precondition(self, matrix):
        """
        Test for the validity of the Kapur-Saxena-Yang precondition.

        The precondition requires that the column corresponding to the
        monomial 1 = x_1 ^ 0 * x_2 ^ 0 * ... * x_n ^ 0 is not a linear
        combination of the remaining ones. In SymPy notation this is
        the last column. For the precondition to hold the last non-zero
        row of the rref matrix should be of the form [0, 0, ..., 1].
        """
        if matrix.is_zero_matrix:
            return False

        m, n = matrix.shape

        # simplify the matrix and keep only its non-zero rows
        matrix = simplify(matrix.rref()[0])
        rows = [i for i in range(m) if any(matrix[i, j] != 0 for j in range(n))]
        matrix = matrix[rows,:]

        condition = Matrix([[0]*(n-1) + [1]])

        if matrix[-1,:] == condition:
            return True
        else:
            return False

    def delete_zero_rows_and_columns(self, matrix):
        """Remove the zero rows and columns of the matrix."""
        rows = [
            i for i in range(matrix.rows) if not matrix.row(i).is_zero_matrix]
        cols = [
            j for j in range(matrix.cols) if not matrix.col(j).is_zero_matrix]

        return matrix[rows, cols]

    def product_leading_entries(self, matrix):
        """Calculate the product of the leading entries of the matrix."""
        res = 1
        for row in range(matrix.rows):
            for el in matrix.row(row):
                if el != 0:
                    res = res * el
                    break
        return res

    def get_KSY_Dixon_resultant(self, matrix):
        """Calculate the Kapur-Saxena-Yang approach to the Dixon Resultant."""
        matrix = self.delete_zero_rows_and_columns(matrix)
        _, U, _ = matrix.LUdecomposition()
        matrix = self.delete_zero_rows_and_columns(simplify(U))

        return self.product_leading_entries(matrix)

class MacaulayResultant():
    """
    A class for calculating the Macaulay resultant. Note that the
    polynomials must be homogenized and their coefficients must be
    given as symbols.

    Examples
    ========

    >>> from sympy import symbols

    >>> from sympy.polys.multivariate_resultants import MacaulayResultant
    >>> x, y, z = symbols('x, y, z')

    >>> a_0, a_1, a_2 = symbols('a_0, a_1, a_2')
    >>> b_0, b_1, b_2 = symbols('b_0, b_1, b_2')
    >>> c_0, c_1, c_2,c_3, c_4 = symbols('c_0, c_1, c_2, c_3, c_4')

    >>> f = a_0 * y -  a_1 * x + a_2 * z
    >>> g = b_1 * x ** 2 + b_0 * y ** 2 - b_2 * z ** 2
    >>> h = c_0 * y * z ** 2 - c_1 * x ** 3 + c_2 * x ** 2 * z - c_3 * x * z ** 2 + c_4 * z ** 3

    >>> mac = MacaulayResultant(polynomials=[f, g, h], variables=[x, y, z])
    >>> mac.monomial_set
    [x**4, x**3*y, x**3*z, x**2*y**2, x**2*y*z, x**2*z**2, x*y**3,
    x*y**2*z, x*y*z**2, x*z**3, y**4, y**3*z, y**2*z**2, y*z**3, z**4]
    >>> matrix = mac.get_matrix()
    >>> submatrix = mac.get_submatrix(matrix)
    >>> submatrix
    Matrix([
    [-a_1,  a_0,  a_2,    0],
    [   0, -a_1,    0,    0],
    [   0,    0, -a_1,    0],
    [   0,    0,    0, -a_1]])

    See Also
    ========

    Notebook in examples: sympy/example/notebooks.

    References
    ==========

    .. [1] [Bruce97]_
    .. [2] [Stiller96]_

    """
    def __init__(self, polynomials, variables):
        """
        Parameters
        ==========

        variables: list
            A list of all n variables
        polynomials : list of SymPy polynomials
            A  list of m n-degree polynomials
        """
        self.polynomials = polynomials
        self.variables = variables
        self.n = len(variables)

        # A list of the d_max of each polynomial.
        self.degrees = [total_degree(poly, *self.variables) for poly
                        in self.polynomials]

        self.degree_m = self._get_degree_m()
        self.monomials_size = self.get_size()

        # The set T of all possible monomials of degree degree_m
        self.monomial_set = self.get_monomials_of_certain_degree(self.degree_m)

    def _get_degree_m(self):
        r"""
        Returns
        =======

        degree_m: int
            The degree_m is calculated as  1 + \sum_1 ^ n (d_i - 1),
            where d_i is the degree of the i polynomial
        """
        return 1 + sum(d - 1 for d in self.degrees)

    def get_size(self):
        r"""
        Returns
        =======

        size: int
            The size of set T. Set T is the set of all possible
            monomials of the n variables for degree equal to the
            degree_m
        """
        return binomial(self.degree_m + self.n - 1, self.n - 1)

    def get_monomials_of_certain_degree(self, degree):
        """
        Returns
        =======

        monomials: list
            A list of monomials of a certain degree.
        """
        monomials = [Mul(*monomial) for monomial
                     in combinations_with_replacement(self.variables,
                                                      degree)]

        return sorted(monomials, reverse=True,
                      key=monomial_key('lex', self.variables))

    def get_row_coefficients(self):
        """
        Returns
        =======

        row_coefficients: list
            The row coefficients of Macaulay's matrix
        """
        row_coefficients = []
        divisible = []
        for i in range(self.n):
            if i == 0:
                degree = self.degree_m - self.degrees[i]
                monomial = self.get_monomials_of_certain_degree(degree)
                row_coefficients.append(monomial)
            else:
                divisible.append(self.variables[i - 1] **
                                 self.degrees[i - 1])
                degree = self.degree_m - self.degrees[i]
                poss_rows = self.get_monomials_of_certain_degree(degree)
                for div in divisible:
                    for p in poss_rows:
                        if rem(p, div) == 0:
                            poss_rows = [item for item in poss_rows
                                         if item != p]
                row_coefficients.append(poss_rows)
        return row_coefficients

    def get_matrix(self):
        """
        Returns
        =======

        macaulay_matrix: Matrix
            The Macaulay numerator matrix
        """
        rows = []
        row_coefficients = self.get_row_coefficients()
        for i in range(self.n):
            for multiplier in row_coefficients[i]:
                coefficients = []
                poly = Poly(self.polynomials[i] * multiplier,
                            *self.variables)

                for mono in self.monomial_set:
                    coefficients.append(poly.coeff_monomial(mono))
                rows.append(coefficients)

        macaulay_matrix = Matrix(rows)
        return macaulay_matrix

    def get_reduced_nonreduced(self):
        r"""
        Returns
        =======

        reduced: list
            A list of the reduced monomials
        non_reduced: list
            A list of the monomials that are not reduced

        Definition
        ==========

        A polynomial is said to be reduced in x_i, if its degree (the
        maximum degree of its monomials) in x_i is less than d_i. A
        polynomial that is reduced in all variables but one is said
        simply to be reduced.
        """
        divisible = []
        for m in self.monomial_set:
            temp = []
            for i, v in enumerate(self.variables):
                temp.append(bool(total_degree(m, v) >= self.degrees[i]))
            divisible.append(temp)
        reduced = [i for i, r in enumerate(divisible)
                   if sum(r) < self.n - 1]
        non_reduced = [i for i, r in enumerate(divisible)
                       if sum(r) >= self.n -1]

        return reduced, non_reduced

    def get_submatrix(self, matrix):
        r"""
        Returns
        =======

        macaulay_submatrix: Matrix
            The Macaulay denominator matrix. Columns that are non reduced are kept.
            The row which contains one of the a_{i}s is dropped. a_{i}s
            are the coefficients of x_i ^ {d_i}.
        """
        reduced, non_reduced = self.get_reduced_nonreduced()

        # if reduced == [], then det(matrix) should be 1
        if reduced == []:
            return diag([1])

        # reduced != []
        reduction_set = [v ** self.degrees[i] for i, v
                         in enumerate(self.variables)]

        ais = [self.polynomials[i].coeff(reduction_set[i])
               for i in range(self.n)]

        reduced_matrix = matrix[:, reduced]
        keep = []
        for row in range(reduced_matrix.rows):
            check = [ai in reduced_matrix[row, :] for ai in ais]
            if True not in check:
                keep.append(row)

        return matrix[keep, non_reduced]
<|endoftext|>"""High-level polynomials manipulation functions. """


from sympy.core import S, Basic, symbols, Dummy
from sympy.polys.polyerrors import (
    PolificationFailed, ComputationFailed,
    MultivariatePolynomialError, OptionError)
from sympy.polys.polyoptions import allowed_flags, build_options
from sympy.polys.polytools import poly_from_expr, Poly
from sympy.polys.specialpolys import (
    symmetric_poly, interpolating_poly)
from sympy.polys.rings import sring
from sympy.utilities import numbered_symbols, take, public

@public
def symmetrize(F, *gens, **args):
    r"""
    Rewrite a polynomial in terms of elementary symmetric polynomials.

    A symmetric polynomial is a multivariate polynomial that remains invariant
    under any variable permutation, i.e., if `f = f(x_1, x_2, \dots, x_n)`,
    then `f = f(x_{i_1}, x_{i_2}, \dots, x_{i_n})`, where
    `(i_1, i_2, \dots, i_n)` is a permutation of `(1, 2, \dots, n)` (an
    element of the group `S_n`).

    Returns a tuple of symmetric polynomials ``(f1, f2, ..., fn)`` such that
    ``f = f1 + f2 + ... + fn``.

    Examples
    ========

    >>> from sympy.polys.polyfuncs import symmetrize
    >>> from sympy.abc import x, y

    >>> symmetrize(x**2 + y**2)
    (-2*x*y + (x + y)**2, 0)

    >>> symmetrize(x**2 + y**2, formal=True)
    (s1**2 - 2*s2, 0, [(s1, x + y), (s2, x*y)])

    >>> symmetrize(x**2 - y**2)
    (-2*x*y + (x + y)**2, -2*y**2)

    >>> symmetrize(x**2 - y**2, formal=True)
    (s1**2 - 2*s2, -2*y**2, [(s1, x + y), (s2, x*y)])

    """
    allowed_flags(args, ['formal', 'symbols'])

    iterable = True

    if not hasattr(F, '__iter__'):
        iterable = False
        F = [F]

    R, F = sring(F, *gens, **args)
    gens = R.symbols

    opt = build_options(gens, args)
    symbols = opt.symbols
    symbols = [next(symbols) for i in range(len(gens))]

    result = []

    for f in F:
        p, r, m = f.symmetrize()
        result.append((p.as_expr(*symbols), r.as_expr(*gens)))

    polys = [(s, g.as_expr()) for s, (_, g) in zip(symbols, m)]

    if not opt.formal:
        for i, (sym, non_sym) in enumerate(result):
            result[i] = (sym.subs(polys), non_sym)

    if not iterable:
        result, = result

    if not opt.formal:
        return result
    else:
        if iterable:
            return result, polys
        else:
            return result + (polys,)


@public
def horner(f, *gens, **args):
    """
    Rewrite a polynomial in Horner form.

    Among other applications, evaluation of a polynomial at a point is optimal
    when it is applied using the Horner scheme ([1]).

    Examples
    ========

    >>> from sympy.polys.polyfuncs import horner
    >>> from sympy.abc import x, y, a, b, c, d, e

    >>> horner(9*x**4 + 8*x**3 + 7*x**2 + 6*x + 5)
    x*(x*(x*(9*x + 8) + 7) + 6) + 5

    >>> horner(a*x**4 + b*x**3 + c*x**2 + d*x + e)
    e + x*(d + x*(c + x*(a*x + b)))

    >>> f = 4*x**2*y**2 + 2*x**2*y + 2*x*y**2 + x*y

    >>> horner(f, wrt=x)
    x*(x*y*(4*y + 2) + y*(2*y + 1))

    >>> horner(f, wrt=y)
    y*(x*y*(4*x + 2) + x*(2*x + 1))

    References
    ==========
    [1] - https://en.wikipedia.org/wiki/Horner_scheme

    """
    allowed_flags(args, [])

    try:
        F, opt = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        return exc.expr

    form, gen = S.Zero, F.gen

    if F.is_univariate:
        for coeff in F.all_coeffs():
            form = form*gen + coeff
    else:
        F, gens = Poly(F, gen), gens[1:]

        for coeff in F.all_coeffs():
            form = form*gen + horner(coeff, *gens, **args)

    return form


@public
def interpolate(data, x):
    """
    Construct an interpolating polynomial for the data points
    evaluated at point x (which can be symbolic or numeric).

    Examples
    ========

    >>> from sympy.polys.polyfuncs import interpolate
    >>> from sympy.abc import a, b, x

    A list is interpreted as though it were paired with a range starting
    from 1:

    >>> interpolate([1, 4, 9, 16], x)
    x**2

    This can be made explicit by giving a list of coordinates:

    >>> interpolate([(1, 1), (2, 4), (3, 9)], x)
    x**2

    The (x, y) coordinates can also be given as keys and values of a
    dictionary (and the points need not be equispaced):

    >>> interpolate([(-1, 2), (1, 2), (2, 5)], x)
    x**2 + 1
    >>> interpolate({-1: 2, 1: 2, 2: 5}, x)
    x**2 + 1

    If the interpolation is going to be used only once then the
    value of interest can be passed instead of passing a symbol:

    >>> interpolate([1, 4, 9], 5)
    25

    Symbolic coordinates are also supported:

    >>> [(i,interpolate((a, b), i)) for i in range(1, 4)]
    [(1, a), (2, b), (3, -a + 2*b)]
    """
    n = len(data)

    if isinstance(data, dict):
        if x in data:
            return S(data[x])
        X, Y = list(zip(*data.items()))
    else:
        if isinstance(data[0], tuple):
            X, Y = list(zip(*data))
            if x in X:
                return S(Y[X.index(x)])
        else:
            if x in range(1, n + 1):
                return S(data[x - 1])
            Y = list(data)
            X = list(range(1, n + 1))

    try:
        return interpolating_poly(n, x, X, Y).expand()
    except ValueError:
        d = Dummy()
        return interpolating_poly(n, d, X, Y).expand().subs(d, x)


@public
def rational_interpolate(data, degnum, X=symbols('x')):
    """
    Returns a rational interpolation, where the data points are element of
    any integral domain.

    The first argument  contains the data (as a list of coordinates). The
    ``degnum`` argument is the degree in the numerator of the rational
    function. Setting it too high will decrease the maximal degree in the
    denominator for the same amount of data.

    Examples
    ========

    >>> from sympy.polys.polyfuncs import rational_interpolate

    >>> data = [(1, -210), (2, -35), (3, 105), (4, 231), (5, 350), (6, 465)]
    >>> rational_interpolate(data, 2)
    (105*x**2 - 525)/(x + 1)

    Values do not need to be integers:

    >>> from sympy import sympify
    >>> x = [1, 2, 3, 4, 5, 6]
    >>> y = sympify("[-1, 0, 2, 22/5, 7, 68/7]")
    >>> rational_interpolate(zip(x, y), 2)
    (3*x**2 - 7*x + 2)/(x + 1)

    The symbol for the variable can be changed if needed:
    >>> from sympy import symbols
    >>> z = symbols('z')
    >>> rational_interpolate(data, 2, X=z)
    (105*z**2 - 525)/(z + 1)

    References
    ==========

    .. [1] Algorithm is adapted from:
           http://axiom-wiki.newsynthesis.org/RationalInterpolation

    """
    from sympy.matrices.dense import ones

    xdata, ydata = list(zip(*data))

    k = len(xdata) - degnum - 1
    if k < 0:
        raise OptionError("Too few values for the required degree.")
    c = ones(degnum + k + 1, degnum + k + 2)
    for j in range(max(degnum, k)):
        for i in range(degnum + k + 1):
            c[i, j + 1] = c[i, j]*xdata[i]
    for j in range(k + 1):
        for i in range(degnum + k + 1):
            c[i, degnum + k + 1 - j] = -c[i, k - j]*ydata[i]
    r = c.nullspace()[0]
    return (sum(r[i] * X**i for i in range(degnum + 1))
            / sum(r[i + degnum + 1] * X**i for i in range(k + 1)))


@public
def viete(f, roots=None, *gens, **args):
    """
    Generate Viete's formulas for ``f``.

    Examples
    ========

    >>> from sympy.polys.polyfuncs import viete
    >>> from sympy import symbols

    >>> x, a, b, c, r1, r2 = symbols('x,a:c,r1:3')

    >>> viete(a*x**2 + b*x + c, [r1, r2], x)
    [(r1 + r2, -b/a), (r1*r2, c/a)]

    """
    allowed_flags(args, [])

    if isinstance(roots, Basic):
        gens, roots = (roots,) + gens, None

    try:
        f, opt = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('viete', 1, exc)

    if f.is_multivariate:
        raise MultivariatePolynomialError(
            "multivariate polynomials are not allowed")

    n = f.degree()

    if n < 1:
        raise ValueError(
            "Cannot derive Viete's formulas for a constant polynomial")

    if roots is None:
        roots = numbered_symbols('r', start=1)

    roots = take(roots, n)

    if n != len(roots):
        raise ValueError("required %s roots, got %s" % (n, len(roots)))

    lc, coeffs = f.LC(), f.all_coeffs()
    result, sign = [], -1

    for i, coeff in enumerate(coeffs[1:]):
        poly = symmetric_poly(i + 1, roots)
        coeff = sign*(coeff/lc)
        result.append((poly, coeff))
        sign = -sign

    return result
<|endoftext|>"""Implementation of matrix FGLM Groebner basis conversion algorithm. """


from sympy.polys.monomials import monomial_mul, monomial_div

def matrix_fglm(F, ring, O_to):
    """
    Converts the reduced Groebner basis ``F`` of a zero-dimensional
    ideal w.r.t. ``O_from`` to a reduced Groebner basis
    w.r.t. ``O_to``.

    References
    ==========

    .. [1] J.C. Faugere, P. Gianni, D. Lazard, T. Mora (1994). Efficient
           Computation of Zero-dimensional Groebner Bases by Change of
           Ordering
    """
    domain = ring.domain
    ngens = ring.ngens

    ring_to = ring.clone(order=O_to)

    old_basis = _basis(F, ring)
    M = _representing_matrices(old_basis, F, ring)

    # V contains the normalforms (wrt O_from) of S
    S = [ring.zero_monom]
    V = [[domain.one] + [domain.zero] * (len(old_basis) - 1)]
    G = []

    L = [(i, 0) for i in range(ngens)]  # (i, j) corresponds to x_i * S[j]
    L.sort(key=lambda k_l: O_to(_incr_k(S[k_l[1]], k_l[0])), reverse=True)
    t = L.pop()

    P = _identity_matrix(len(old_basis), domain)

    while True:
        s = len(S)
        v = _matrix_mul(M[t[0]], V[t[1]])
        _lambda = _matrix_mul(P, v)

        if all(_lambda[i] == domain.zero for i in range(s, len(old_basis))):
            # there is a linear combination of v by V
            lt = ring.term_new(_incr_k(S[t[1]], t[0]), domain.one)
            rest = ring.from_dict({S[i]: _lambda[i] for i in range(s)})

            g = (lt - rest).set_ring(ring_to)
            if g:
                G.append(g)
        else:
            # v is linearly independent from V
            P = _update(s, _lambda, P)
            S.append(_incr_k(S[t[1]], t[0]))
            V.append(v)

            L.extend([(i, s) for i in range(ngens)])
            L = list(set(L))
            L.sort(key=lambda k_l: O_to(_incr_k(S[k_l[1]], k_l[0])), reverse=True)

        L = [(k, l) for (k, l) in L if all(monomial_div(_incr_k(S[l], k), g.LM) is None for g in G)]

        if not L:
            G = [ g.monic() for g in G ]
            return sorted(G, key=lambda g: O_to(g.LM), reverse=True)

        t = L.pop()


def _incr_k(m, k):
    return tuple(list(m[:k]) + [m[k] + 1] + list(m[k + 1:]))


def _identity_matrix(n, domain):
    M = [[domain.zero]*n for _ in range(n)]

    for i in range(n):
        M[i][i] = domain.one

    return M


def _matrix_mul(M, v):
    return [sum(row[i] * v[i] for i in range(len(v))) for row in M]


def _update(s, _lambda, P):
    """
    Update ``P`` such that for the updated `P'` `P' v = e_{s}`.
    """
    k = min(j for j in range(s, len(_lambda)) if _lambda[j] != 0)

    for r in range(len(_lambda)):
        if r != k:
            P[r] = [P[r][j] - (P[k][j] * _lambda[r]) / _lambda[k] for j in range(len(P[r]))]

    P[k] = [P[k][j] / _lambda[k] for j in range(len(P[k]))]
    P[k], P[s] = P[s], P[k]

    return P


def _representing_matrices(basis, G, ring):
    r"""
    Compute the matrices corresponding to the linear maps `m \mapsto
    x_i m` for all variables `x_i`.
    """
    domain = ring.domain
    u = ring.ngens-1

    def var(i):
        return tuple([0] * i + [1] + [0] * (u - i))

    def representing_matrix(m):
        M = [[domain.zero] * len(basis) for _ in range(len(basis))]

        for i, v in enumerate(basis):
            r = ring.term_new(monomial_mul(m, v), domain.one).rem(G)

            for monom, coeff in r.terms():
                j = basis.index(monom)
                M[j][i] = coeff

        return M

    return [representing_matrix(var(i)) for i in range(u + 1)]


def _basis(G, ring):
    r"""
    Computes a list of monomials which are not divisible by the leading
    monomials wrt to ``O`` of ``G``. These monomials are a basis of
    `K[X_1, \ldots, X_n]/(G)`.
    """
    order = ring.order

    leading_monomials = [g.LM for g in G]
    candidates = [ring.zero_monom]
    basis = []

    while candidates:
        t = candidates.pop()
        basis.append(t)

        new_candidates = [_incr_k(t, k) for k in range(ring.ngens)
            if all(monomial_div(_incr_k(t, k), lmg) is None
            for lmg in leading_monomials)]
        candidates.extend(new_candidates)
        candidates.sort(key=order, reverse=True)

    basis = list(set(basis))

    return sorted(basis, key=order)
<|endoftext|>"""Sparse rational function fields. """

from __future__ import annotations
from functools import reduce

from operator import add, mul, lt, le, gt, ge

from sympy.core.expr import Expr
from sympy.core.mod import Mod
from sympy.core.numbers import Exp1
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import CantSympify, sympify
from sympy.functions.elementary.exponential import ExpBase
from sympy.polys.domains.domain import Domain
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.domains.polynomialring import PolynomialRing
from sympy.polys.constructor import construct_domain
from sympy.polys.orderings import lex, MonomialOrder
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polyoptions import build_options
from sympy.polys.polyutils import _parallel_dict_from_expr
from sympy.polys.rings import PolyRing, PolyElement
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public
from sympy.utilities.iterables import is_sequence
from sympy.utilities.magic import pollute

@public
def field(symbols, domain, order=lex):
    """Construct new rational function field returning (field, x1, ..., xn). """
    _field = FracField(symbols, domain, order)
    return (_field,) + _field.gens

@public
def xfield(symbols, domain, order=lex):
    """Construct new rational function field returning (field, (x1, ..., xn)). """
    _field = FracField(symbols, domain, order)
    return (_field, _field.gens)

@public
def vfield(symbols, domain, order=lex):
    """Construct new rational function field and inject generators into global namespace. """
    _field = FracField(symbols, domain, order)
    pollute([ sym.name for sym in _field.symbols ], _field.gens)
    return _field

@public
def sfield(exprs, *symbols, **options):
    """Construct a field deriving generators and domain
    from options and input expressions.

    Parameters
    ==========

    exprs   : py:class:`~.Expr` or sequence of :py:class:`~.Expr` (sympifiable)

    symbols : sequence of :py:class:`~.Symbol`/:py:class:`~.Expr`

    options : keyword arguments understood by :py:class:`~.Options`

    Examples
    ========

    >>> from sympy import exp, log, symbols, sfield

    >>> x = symbols("x")
    >>> K, f = sfield((x*log(x) + 4*x**2)*exp(1/x + log(x)/3)/x**2)
    >>> K
    Rational function field in x, exp(1/x), log(x), x**(1/3) over ZZ with lex order
    >>> f
    (4*x**2*(exp(1/x)) + x*(exp(1/x))*(log(x)))/((x**(1/3))**5)
    """
    single = False
    if not is_sequence(exprs):
        exprs, single = [exprs], True

    exprs = list(map(sympify, exprs))
    opt = build_options(symbols, options)
    numdens = []
    for expr in exprs:
        numdens.extend(expr.as_numer_denom())
    reps, opt = _parallel_dict_from_expr(numdens, opt)

    if opt.domain is None:
        # NOTE: this is inefficient because construct_domain() automatically
        # performs conversion to the target domain. It shouldn't do this.
        coeffs = sum([list(rep.values()) for rep in reps], [])
        opt.domain, _ = construct_domain(coeffs, opt=opt)

    _field = FracField(opt.gens, opt.domain, opt.order)
    fracs = []
    for i in range(0, len(reps), 2):
        fracs.append(_field(tuple(reps[i:i+2])))

    if single:
        return (_field, fracs[0])
    else:
        return (_field, fracs)


class FracField(DefaultPrinting):
    """Multivariate distributed rational function field. """

    ring: PolyRing
    gens: tuple[FracElement, ...]
    symbols: tuple[Expr, ...]
    ngens: int
    domain: Domain
    order: MonomialOrder

    def __new__(cls, symbols, domain, order=lex):
        ring = PolyRing(symbols, domain, order)
        symbols = ring.symbols
        ngens = ring.ngens
        domain = ring.domain
        order = ring.order

        _hash_tuple = (cls.__name__, symbols, ngens, domain, order)

        obj = object.__new__(cls)
        obj._hash_tuple = _hash_tuple
        obj._hash = hash(_hash_tuple)
        obj.ring = ring
        obj.symbols = symbols
        obj.ngens = ngens
        obj.domain = domain
        obj.order = order

        obj.dtype = FracElement(obj, ring.zero).raw_new

        obj.zero = obj.dtype(ring.zero)
        obj.one = obj.dtype(ring.one)

        obj.gens = obj._gens()

        for symbol, generator in zip(obj.symbols, obj.gens):
            if isinstance(symbol, Symbol):
                name = symbol.name

                if not hasattr(obj, name):
                    setattr(obj, name, generator)

        return obj

    def _gens(self):
        """Return a list of polynomial generators. """
        return tuple([ self.dtype(gen) for gen in self.ring.gens ])

    def __getnewargs__(self):
        return (self.symbols, self.domain, self.order)

    def __hash__(self):
        return self._hash

    def index(self, gen):
        if self.is_element(gen):
            return self.ring.index(gen.to_poly())
        else:
            raise ValueError("expected a %s, got %s instead" % (self.dtype,gen))

    def __eq__(self, other):
        return isinstance(other, FracField) and \
            (self.symbols, self.ngens, self.domain, self.order) == \
            (other.symbols, other.ngens, other.domain, other.order)

    def __ne__(self, other):
        return not self == other

    def is_element(self, element):
        """True if ``element`` is an element of this field. False otherwise. """
        return isinstance(element, FracElement) and element.field == self

    def raw_new(self, numer, denom=None):
        return self.dtype(numer, denom)

    def new(self, numer, denom=None):
        if denom is None: denom = self.ring.one
        numer, denom = numer.cancel(denom)
        return self.raw_new(numer, denom)

    def domain_new(self, element):
        return self.domain.convert(element)

    def ground_new(self, element):
        try:
            return self.new(self.ring.ground_new(element))
        except CoercionFailed:
            domain = self.domain

            if not domain.is_Field and domain.has_assoc_Field:
                ring = self.ring
                ground_field = domain.get_field()
                element = ground_field.convert(element)
                numer = ring.ground_new(ground_field.numer(element))
                denom = ring.ground_new(ground_field.denom(element))
                return self.raw_new(numer, denom)
            else:
                raise

    def field_new(self, element):
        if isinstance(element, FracElement):
            if self == element.field:
                return element

            if isinstance(self.domain, FractionField) and \
                self.domain.field == element.field:
                return self.ground_new(element)
            elif isinstance(self.domain, PolynomialRing) and \
                self.domain.ring.to_field() == element.field:
                return self.ground_new(element)
            else:
                raise NotImplementedError("conversion")
        elif isinstance(element, PolyElement):
            denom, numer = element.clear_denoms()

            if isinstance(self.domain, PolynomialRing) and \
                numer.ring == self.domain.ring:
                numer = self.ring.ground_new(numer)
            elif isinstance(self.domain, FractionField) and \
                numer.ring == self.domain.field.to_ring():
                numer = self.ring.ground_new(numer)
            else:
                numer = numer.set_ring(self.ring)

            denom = self.ring.ground_new(denom)
            return self.raw_new(numer, denom)
        elif isinstance(element, tuple) and len(element) == 2:
            numer, denom = list(map(self.ring.ring_new, element))
            return self.new(numer, denom)
        elif isinstance(element, str):
            raise NotImplementedError("parsing")
        elif isinstance(element, Expr):
            return self.from_expr(element)
        else:
            return self.ground_new(element)

    __call__ = field_new

    def _rebuild_expr(self, expr, mapping):
        domain = self.domain
        powers = tuple((gen, gen.as_base_exp()) for gen in mapping.keys()
            if gen.is_Pow or isinstance(gen, ExpBase))

        def _rebuild(expr):
            generator = mapping.get(expr)

            if generator is not None:
                return generator
            elif expr.is_Add:
                return reduce(add, list(map(_rebuild, expr.args)))
            elif expr.is_Mul:
                return reduce(mul, list(map(_rebuild, expr.args)))
            elif expr.is_Pow or isinstance(expr, (ExpBase, Exp1)):
                b, e = expr.as_base_exp()
                # look for bg**eg whose integer power may be b**e
                for gen, (bg, eg) in powers:
                    if bg == b and Mod(e, eg) == 0:
                        return mapping.get(gen)**int(e/eg)
                if e.is_Integer and e is not S.One:
                    return _rebuild(b)**int(e)
            elif mapping.get(1/expr) is not None:
                return 1/mapping.get(1/expr)

            try:
                return domain.convert(expr)
            except CoercionFailed:
                if not domain.is_Field and domain.has_assoc_Field:
                    return domain.get_field().convert(expr)
                else:
                    raise

        return _rebuild(expr)

    def from_expr(self, expr):
        mapping = dict(list(zip(self.symbols, self.gens)))

        try:
            frac = self._rebuild_expr(sympify(expr), mapping)
        except CoercionFailed:
            raise ValueError("expected an expression convertible to a rational function in %s, got %s" % (self, expr))
        else:
            return self.field_new(frac)

    def to_domain(self):
        return FractionField(self)

    def to_ring(self):
        return PolyRing(self.symbols, self.domain, self.order)

class FracElement(DomainElement, DefaultPrinting, CantSympify):
    """Element of multivariate distributed rational function field. """

    def __init__(self, field, numer, denom=None):
        if denom is None:
            denom = field.ring.one
        elif not denom:
            raise ZeroDivisionError("zero denominator")

        self.field = field
        self.numer = numer
        self.denom = denom

    def raw_new(f, numer, denom=None):
        return f.__class__(f.field, numer, denom)

    def new(f, numer, denom):
        return f.raw_new(*numer.cancel(denom))

    def to_poly(f):
        if f.denom != 1:
            raise ValueError("f.denom should be 1")
        return f.numer

    def parent(self):
        return self.field.to_domain()

    def __getnewargs__(self):
        return (self.field, self.numer, self.denom)

    _hash = None

    def __hash__(self):
        _hash = self._hash
        if _hash is None:
            self._hash = _hash = hash((self.field, self.numer, self.denom))
        return _hash

    def copy(self):
        return self.raw_new(self.numer.copy(), self.denom.copy())

    def set_field(self, new_field):
        if self.field == new_field:
            return self
        else:
            new_ring = new_field.ring
            numer = self.numer.set_ring(new_ring)
            denom = self.denom.set_ring(new_ring)
            return new_field.new(numer, denom)

    def as_expr(self, *symbols):
        return self.numer.as_expr(*symbols)/self.denom.as_expr(*symbols)

    def __eq__(f, g):
        if isinstance(g, FracElement) and f.field == g.field:
            return f.numer == g.numer and f.denom == g.denom
        else:
            return f.numer == g and f.denom == f.field.ring.one

    def __ne__(f, g):
        return not f == g

    def __bool__(f):
        return bool(f.numer)

    def sort_key(self):
        return (self.denom.sort_key(), self.numer.sort_key())

    def _cmp(f1, f2, op):
        if f1.field.is_element(f2):
            return op(f1.sort_key(), f2.sort_key())
        else:
            return NotImplemented

    def __lt__(f1, f2):
        return f1._cmp(f2, lt)
    def __le__(f1, f2):
        return f1._cmp(f2, le)
    def __gt__(f1, f2):
        return f1._cmp(f2, gt)
    def __ge__(f1, f2):
        return f1._cmp(f2, ge)

    def __pos__(f):
        """Negate all coefficients in ``f``. """
        return f.raw_new(f.numer, f.denom)

    def __neg__(f):
        """Negate all coefficients in ``f``. """
        return f.raw_new(-f.numer, f.denom)

    def _extract_ground(self, element):
        domain = self.field.domain

        try:
            element = domain.convert(element)
        except CoercionFailed:
            if not domain.is_Field and domain.has_assoc_Field:
                ground_field = domain.get_field()

                try:
                    element = ground_field.convert(element)
                except CoercionFailed:
                    pass
                else:
                    return -1, ground_field.numer(element), ground_field.denom(element)

            return 0, None, None
        else:
            return 1, element, None

    def __add__(f, g):
        """Add rational functions ``f`` and ``g``. """
        field = f.field

        if not g:
            return f
        elif not f:
            return g
        elif field.is_element(g):
            if f.denom == g.denom:
                return f.new(f.numer + g.numer, f.denom)
            else:
                return f.new(f.numer*g.denom + f.denom*g.numer, f.denom*g.denom)
        elif field.ring.is_element(g):
            return f.new(f.numer + f.denom*g, f.denom)
        else:
            if isinstance(g, FracElement):
                if isinstance(field.domain, FractionField) and field.domain.field == g.field:
                    pass
                elif isinstance(g.field.domain, FractionField) and g.field.domain.field == field:
                    return g.__radd__(f)
                else:
                    return NotImplemented
            elif isinstance(g, PolyElement):
                if isinstance(field.domain, PolynomialRing) and field.domain.ring == g.ring:
                    pass
                else:
                    return g.__radd__(f)

        return f.__radd__(g)

    def __radd__(f, c):
        if f.field.ring.is_element(c):
            return f.new(f.numer + f.denom*c, f.denom)

        op, g_numer, g_denom = f._extract_ground(c)

        if op == 1:
            return f.new(f.numer + f.denom*g_numer, f.denom)
        elif not op:
            return NotImplemented
        else:
            return f.new(f.numer*g_denom + f.denom*g_numer, f.denom*g_denom)

    def __sub__(f, g):
        """Subtract rational functions ``f`` and ``g``. """
        field = f.field

        if not g:
            return f
        elif not f:
            return -g
        elif field.is_element(g):
            if f.denom == g.denom:
                return f.new(f.numer - g.numer, f.denom)
            else:
                return f.new(f.numer*g.denom - f.denom*g.numer, f.denom*g.denom)
        elif field.ring.is_element(g):
            return f.new(f.numer - f.denom*g, f.denom)
        else:
            if isinstance(g, FracElement):
                if isinstance(field.domain, FractionField) and field.domain.field == g.field:
                    pass
                elif isinstance(g.field.domain, FractionField) and g.field.domain.field == field:
                    return g.__rsub__(f)
                else:
                    return NotImplemented
            elif isinstance(g, PolyElement):
                if isinstance(field.domain, PolynomialRing) and field.domain.ring == g.ring:
                    pass
                else:
                    return g.__rsub__(f)

        op, g_numer, g_denom = f._extract_ground(g)

        if op == 1:
            return f.new(f.numer - f.denom*g_numer, f.denom)
        elif not op:
            return NotImplemented
        else:
            return f.new(f.numer*g_denom - f.denom*g_numer, f.denom*g_denom)

    def __rsub__(f, c):
        if f.field.ring.is_element(c):
            return f.new(-f.numer + f.denom*c, f.denom)

        op, g_numer, g_denom = f._extract_ground(c)

        if op == 1:
            return f.new(-f.numer + f.denom*g_numer, f.denom)
        elif not op:
            return NotImplemented
        else:
            return f.new(-f.numer*g_denom + f.denom*g_numer, f.denom*g_denom)

    def __mul__(f, g):
        """Multiply rational functions ``f`` and ``g``. """
        field = f.field

        if not f or not g:
            return field.zero
        elif field.is_element(g):
            return f.new(f.numer*g.numer, f.denom*g.denom)
        elif field.ring.is_element(g):
            return f.new(f.numer*g, f.denom)
        else:
            if isinstance(g, FracElement):
                if isinstance(field.domain, FractionField) and field.domain.field == g.field:
                    pass
                elif isinstance(g.field.domain, FractionField) and g.field.domain.field == field:
                    return g.__rmul__(f)
                else:
                    return NotImplemented
            elif isinstance(g, PolyElement):
                if isinstance(field.domain, PolynomialRing) and field.domain.ring == g.ring:
                    pass
                else:
                    return g.__rmul__(f)

        return f.__rmul__(g)

    def __rmul__(f, c):
        if f.field.ring.is_element(c):
            return f.new(f.numer*c, f.denom)

        op, g_numer, g_denom = f._extract_ground(c)

        if op == 1:
            return f.new(f.numer*g_numer, f.denom)
        elif not op:
            return NotImplemented
        else:
            return f.new(f.numer*g_numer, f.denom*g_denom)

    def __truediv__(f, g):
        """Computes quotient of fractions ``f`` and ``g``. """
        field = f.field

        if not g:
            raise ZeroDivisionError
        elif field.is_element(g):
            return f.new(f.numer*g.denom, f.denom*g.numer)
        elif field.ring.is_element(g):
            return f.new(f.numer, f.denom*g)
        else:
            if isinstance(g, FracElement):
                if isinstance(field.domain, FractionField) and field.domain.field == g.field:
                    pass
                elif isinstance(g.field.domain, FractionField) and g.field.domain.field == field:
                    return g.__rtruediv__(f)
                else:
                    return NotImplemented
            elif isinstance(g, PolyElement):
                if isinstance(field.domain, PolynomialRing) and field.domain.ring == g.ring:
                    pass
                else:
                    return g.__rtruediv__(f)

        op, g_numer, g_denom = f._extract_ground(g)

        if op == 1:
            return f.new(f.numer, f.denom*g_numer)
        elif not op:
            return NotImplemented
        else:
            return f.new(f.numer*g_denom, f.denom*g_numer)

    def __rtruediv__(f, c):
        if not f:
            raise ZeroDivisionError
        elif f.field.ring.is_element(c):
            return f.new(f.denom*c, f.numer)

        op, g_numer, g_denom = f._extract_ground(c)

        if op == 1:
            return f.new(f.denom*g_numer, f.numer)
        elif not op:
            return NotImplemented
        else:
            return f.new(f.denom*g_numer, f.numer*g_denom)

    def __pow__(f, n):
        """Raise ``f`` to a non-negative power ``n``. """
        if n >= 0:
            return f.raw_new(f.numer**n, f.denom**n)
        elif not f:
            raise ZeroDivisionError
        else:
            return f.raw_new(f.denom**-n, f.numer**-n)

    def diff(f, x):
        """Computes partial derivative in ``x``.

        Examples
        ========

        >>> from sympy.polys.fields import field
        >>> from sympy.polys.domains import ZZ

        >>> _, x, y, z = field("x,y,z", ZZ)
        >>> ((x**2 + y)/(z + 1)).diff(x)
        2*x/(z + 1)

        """
        x = x.to_poly()
        return f.new(f.numer.diff(x)*f.denom - f.numer*f.denom.diff(x), f.denom**2)

    def __call__(f, *values):
        if 0 < len(values) <= f.field.ngens:
            return f.evaluate(list(zip(f.field.gens, values)))
        else:
            raise ValueError("expected at least 1 and at most %s values, got %s" % (f.field.ngens, len(values)))

    def evaluate(f, x, a=None):
        if isinstance(x, list) and a is None:
            x = [ (X.to_poly(), a) for X, a in x ]
            numer, denom = f.numer.evaluate(x), f.denom.evaluate(x)
        else:
            x = x.to_poly()
            numer, denom = f.numer.evaluate(x, a), f.denom.evaluate(x, a)

        field = numer.ring.to_field()
        return field.new(numer, denom)

    def subs(f, x, a=None):
        if isinstance(x, list) and a is None:
            x = [ (X.to_poly(), a) for X, a in x ]
            numer, denom = f.numer.subs(x), f.denom.subs(x)
        else:
            x = x.to_poly()
            numer, denom = f.numer.subs(x, a), f.denom.subs(x, a)

        return f.new(numer, denom)

    def compose(f, x, a=None):
        raise NotImplementedError
<|endoftext|>"""Tools for constructing domains for expressions. """
from math import prod

from sympy.core import sympify
from sympy.core.evalf import pure_complex
from sympy.core.sorting import ordered
from sympy.polys.domains import ZZ, QQ, ZZ_I, QQ_I, EX
from sympy.polys.domains.complexfield import ComplexField
from sympy.polys.domains.realfield import RealField
from sympy.polys.polyoptions import build_options
from sympy.polys.polyutils import parallel_dict_from_basic
from sympy.utilities import public


def _construct_simple(coeffs, opt):
    """Handle simple domains, e.g.: ZZ, QQ, RR and algebraic domains. """
    rationals = floats = complexes = algebraics = False
    float_numbers = []

    if opt.extension is True:
        is_algebraic = lambda coeff: coeff.is_number and coeff.is_algebraic
    else:
        is_algebraic = lambda coeff: False

    for coeff in coeffs:
        if coeff.is_Rational:
            if not coeff.is_Integer:
                rationals = True
        elif coeff.is_Float:
            if algebraics:
                # there are both reals and algebraics -> EX
                return False
            else:
                floats = True
                float_numbers.append(coeff)
        else:
            is_complex = pure_complex(coeff)
            if is_complex:
                complexes = True
                x, y = is_complex
                if x.is_Rational and y.is_Rational:
                    if not (x.is_Integer and y.is_Integer):
                        rationals = True
                    continue
                else:
                    floats = True
                    if x.is_Float:
                        float_numbers.append(x)
                    if y.is_Float:
                        float_numbers.append(y)
            elif is_algebraic(coeff):
                if floats:
                    # there are both algebraics and reals -> EX
                    return False
                algebraics = True
            else:
                # this is a composite domain, e.g. ZZ[X], EX
                return None

    # Use the maximum precision of all coefficients for the RR or CC
    # precision
    max_prec = max(c._prec for c in float_numbers) if float_numbers else 53

    if algebraics:
        domain, result = _construct_algebraic(coeffs, opt)
    else:
        if floats and complexes:
            domain = ComplexField(prec=max_prec)
        elif floats:
            domain = RealField(prec=max_prec)
        elif rationals or opt.field:
            domain = QQ_I if complexes else QQ
        else:
            domain = ZZ_I if complexes else ZZ

        result = [domain.from_sympy(coeff) for coeff in coeffs]

    return domain, result


def _construct_algebraic(coeffs, opt):
    """We know that coefficients are algebraic so construct the extension. """
    from sympy.polys.numberfields import primitive_element

    exts = set()

    def build_trees(args):
        trees = []
        for a in args:
            if a.is_Rational:
                tree = ('Q', QQ.from_sympy(a))
            elif a.is_Add:
                tree = ('+', build_trees(a.args))
            elif a.is_Mul:
                tree = ('*', build_trees(a.args))
            else:
                tree = ('e', a)
                exts.add(a)
            trees.append(tree)
        return trees

    trees = build_trees(coeffs)
    exts = list(ordered(exts))

    g, span, H = primitive_element(exts, ex=True, polys=True)
    root = sum(s*ext for s, ext in zip(span, exts))

    domain, g = QQ.algebraic_field((g, root)), g.rep.to_list()

    exts_dom = [domain.dtype.from_list(h, g, QQ) for h in H]
    exts_map = dict(zip(exts, exts_dom))

    def convert_tree(tree):
        op, args = tree
        if op == 'Q':
            return domain.dtype.from_list([args], g, QQ)
        elif op == '+':
            return sum((convert_tree(a) for a in args), domain.zero)
        elif op == '*':
            return prod(convert_tree(a) for a in args)
        elif op == 'e':
            return exts_map[args]
        else:
            raise RuntimeError

    result = [convert_tree(tree) for tree in trees]

    return domain, result


def _construct_composite(coeffs, opt):
    """Handle composite domains, e.g.: ZZ[X], QQ[X], ZZ(X), QQ(X). """
    numers, denoms = [], []

    for coeff in coeffs:
        numer, denom = coeff.as_numer_denom()

        numers.append(numer)
        denoms.append(denom)

    polys, gens = parallel_dict_from_basic(numers + denoms)  # XXX: sorting
    if not gens:
        return None

    if opt.composite is None:
        if any(gen.is_number and gen.is_algebraic for gen in gens):
            return None # generators are number-like so lets better use EX

        all_symbols = set()

        for gen in gens:
            symbols = gen.free_symbols

            if all_symbols & symbols:
                return None # there could be algebraic relations between generators
            else:
                all_symbols |= symbols

    n = len(gens)
    k = len(polys)//2

    numers = polys[:k]
    denoms = polys[k:]

    if opt.field:
        fractions = True
    else:
        fractions, zeros = False, (0,)*n

        for denom in denoms:
            if len(denom) > 1 or zeros not in denom:
                fractions = True
                break

    coeffs = set()

    if not fractions:
        for numer, denom in zip(numers, denoms):
            denom = denom[zeros]

            for monom, coeff in numer.items():
                coeff /= denom
                coeffs.add(coeff)
                numer[monom] = coeff
    else:
        for numer, denom in zip(numers, denoms):
            coeffs.update(list(numer.values()))
            coeffs.update(list(denom.values()))

    rationals = floats = complexes = False
    float_numbers = []

    for coeff in coeffs:
        if coeff.is_Rational:
            if not coeff.is_Integer:
                rationals = True
        elif coeff.is_Float:
            floats = True
            float_numbers.append(coeff)
        else:
            is_complex = pure_complex(coeff)
            if is_complex is not None:
                complexes = True
                x, y = is_complex
                if x.is_Rational and y.is_Rational:
                    if not (x.is_Integer and y.is_Integer):
                        rationals = True
                else:
                    floats = True
                    if x.is_Float:
                        float_numbers.append(x)
                    if y.is_Float:
                        float_numbers.append(y)

    max_prec = max(c._prec for c in float_numbers) if float_numbers else 53

    if floats and complexes:
        ground = ComplexField(prec=max_prec)
    elif floats:
        ground = RealField(prec=max_prec)
    elif complexes:
        if rationals:
            ground = QQ_I
        else:
            ground = ZZ_I
    elif rationals:
        ground = QQ
    else:
        ground = ZZ

    result = []

    if not fractions:
        domain = ground.poly_ring(*gens)

        for numer in numers:
            for monom, coeff in numer.items():
                numer[monom] = ground.from_sympy(coeff)

            result.append(domain(numer))
    else:
        domain = ground.frac_field(*gens)

        for numer, denom in zip(numers, denoms):
            for monom, coeff in numer.items():
                numer[monom] = ground.from_sympy(coeff)

            for monom, coeff in denom.items():
                denom[monom] = ground.from_sympy(coeff)

            result.append(domain((numer, denom)))

    return domain, result


def _construct_expression(coeffs, opt):
    """The last resort case, i.e. use the expression domain. """
    domain, result = EX, []

    for coeff in coeffs:
        result.append(domain.from_sympy(coeff))

    return domain, result


@public
def construct_domain(obj, **args):
    """Construct a minimal domain for a list of expressions.

    Explanation
    ===========

    Given a list of normal SymPy expressions (of type :py:class:`~.Expr`)
    ``construct_domain`` will find a minimal :py:class:`~.Domain` that can
    represent those expressions. The expressions will be converted to elements
    of the domain and both the domain and the domain elements are returned.

    Parameters
    ==========

    obj: list or dict
        The expressions to build a domain for.

    **args: keyword arguments
        Options that affect the choice of domain.

    Returns
    =======

    (K, elements): Domain and list of domain elements
        The domain K that can represent the expressions and the list or dict
        of domain elements representing the same expressions as elements of K.

    Examples
    ========

    Given a list of :py:class:`~.Integer` ``construct_domain`` will return the
    domain :ref:`ZZ` and a list of integers as elements of :ref:`ZZ`.

    >>> from sympy import construct_domain, S
    >>> expressions = [S(2), S(3), S(4)]
    >>> K, elements = construct_domain(expressions)
    >>> K
    ZZ
    >>> elements
    [2, 3, 4]
    >>> type(elements[0])  # doctest: +SKIP
    <class 'int'>
    >>> type(expressions[0])
    <class 'sympy.core.numbers.Integer'>

    If there are any :py:class:`~.Rational` then :ref:`QQ` is returned
    instead.

    >>> construct_domain([S(1)/2, S(3)/4])
    (QQ, [1/2, 3/4])

    If there are symbols then a polynomial ring :ref:`K[x]` is returned.

    >>> from sympy import symbols
    >>> x, y = symbols('x, y')
    >>> construct_domain([2*x + 1, S(3)/4])
    (QQ[x], [2*x + 1, 3/4])
    >>> construct_domain([2*x + 1, y])
    (ZZ[x,y], [2*x + 1, y])

    If any symbols appear with negative powers then a rational function field
    :ref:`K(x)` will be returned.

    >>> construct_domain([y/x, x/(1 - y)])
    (ZZ(x,y), [y/x, -x/(y - 1)])

    Irrational algebraic numbers will result in the :ref:`EX` domain by
    default. The keyword argument ``extension=True`` leads to the construction
    of an algebraic number field :ref:`QQ(a)`.

    >>> from sympy import sqrt
    >>> construct_domain([sqrt(2)])
    (EX, [EX(sqrt(2))])
    >>> construct_domain([sqrt(2)], extension=True)  # doctest: +SKIP
    (QQ<sqrt(2)>, [ANP([1, 0], [1, 0, -2], QQ)])

    See also
    ========

    Domain
    Expr
    """
    opt = build_options(args)

    if hasattr(obj, '__iter__'):
        if isinstance(obj, dict):
            if not obj:
                monoms, coeffs = [], []
            else:
                monoms, coeffs = list(zip(*list(obj.items())))
        else:
            coeffs = obj
    else:
        coeffs = [obj]

    coeffs = list(map(sympify, coeffs))
    result = _construct_simple(coeffs, opt)

    if result is not None:
        if result is not False:
            domain, coeffs = result
        else:
            domain, coeffs = _construct_expression(coeffs, opt)
    else:
        if opt.composite is False:
            result = None
        else:
            result = _construct_composite(coeffs, opt)

        if result is not None:
            domain, coeffs = result
        else:
            domain, coeffs = _construct_expression(coeffs, opt)

    if hasattr(obj, '__iter__'):
        if isinstance(obj, dict):
            return domain, dict(list(zip(monoms, coeffs)))
        else:
            return domain, coeffs
    else:
        return domain, coeffs[0]
<|endoftext|>"""Definitions of monomial orderings. """

from __future__ import annotations

__all__ = ["lex", "grlex", "grevlex", "ilex", "igrlex", "igrevlex"]

from sympy.core import Symbol
from sympy.utilities.iterables import iterable

class MonomialOrder:
    """Base class for monomial orderings. """

    alias: str | None = None
    is_global: bool | None = None
    is_default = False

    def __repr__(self):
        return self.__class__.__name__ + "()"

    def __str__(self):
        return self.alias

    def __call__(self, monomial):
        raise NotImplementedError

    def __eq__(self, other):
        return self.__class__ == other.__class__

    def __hash__(self):
        return hash(self.__class__)

    def __ne__(self, other):
        return not (self == other)

class LexOrder(MonomialOrder):
    """Lexicographic order of monomials. """

    alias = 'lex'
    is_global = True
    is_default = True

    def __call__(self, monomial):
        return monomial

class GradedLexOrder(MonomialOrder):
    """Graded lexicographic order of monomials. """

    alias = 'grlex'
    is_global = True

    def __call__(self, monomial):
        return (sum(monomial), monomial)

class ReversedGradedLexOrder(MonomialOrder):
    """Reversed graded lexicographic order of monomials. """

    alias = 'grevlex'
    is_global = True

    def __call__(self, monomial):
        return (sum(monomial), tuple(reversed([-m for m in monomial])))

class ProductOrder(MonomialOrder):
    """
    A product order built from other monomial orders.

    Given (not necessarily total) orders O1, O2, ..., On, their product order
    P is defined as M1 > M2 iff there exists i such that O1(M1) = O2(M2),
    ..., Oi(M1) = Oi(M2), O{i+1}(M1) > O{i+1}(M2).

    Product orders are typically built from monomial orders on different sets
    of variables.

    ProductOrder is constructed by passing a list of pairs
    [(O1, L1), (O2, L2), ...] where Oi are MonomialOrders and Li are callables.
    Upon comparison, the Li are passed the total monomial, and should filter
    out the part of the monomial to pass to Oi.

    Examples
    ========

    We can use a lexicographic order on x_1, x_2 and also on
    y_1, y_2, y_3, and their product on {x_i, y_i} as follows:

    >>> from sympy.polys.orderings import lex, grlex, ProductOrder
    >>> P = ProductOrder(
    ...     (lex, lambda m: m[:2]), # lex order on x_1 and x_2 of monomial
    ...     (grlex, lambda m: m[2:]) # grlex on y_1, y_2, y_3
    ... )
    >>> P((2, 1, 1, 0, 0)) > P((1, 10, 0, 2, 0))
    True

    Here the exponent `2` of `x_1` in the first monomial
    (`x_1^2 x_2 y_1`) is bigger than the exponent `1` of `x_1` in the
    second monomial (`x_1 x_2^10 y_2^2`), so the first monomial is greater
    in the product ordering.

    >>> P((2, 1, 1, 0, 0)) < P((2, 1, 0, 2, 0))
    True

    Here the exponents of `x_1` and `x_2` agree, so the grlex order on
    `y_1, y_2, y_3` is used to decide the ordering. In this case the monomial
    `y_2^2` is ordered larger than `y_1`, since for the grlex order the degree
    of the monomial is most important.
    """

    def __init__(self, *args):
        self.args = args

    def __call__(self, monomial):
        return tuple(O(lamda(monomial)) for (O, lamda) in self.args)

    def __repr__(self):
        contents = [repr(x[0]) for x in self.args]
        return self.__class__.__name__ + '(' + ", ".join(contents) + ')'

    def __str__(self):
        contents = [str(x[0]) for x in self.args]
        return self.__class__.__name__ + '(' + ", ".join(contents) + ')'

    def __eq__(self, other):
        if not isinstance(other, ProductOrder):
            return False
        return self.args == other.args

    def __hash__(self):
        return hash((self.__class__, self.args))

    @property
    def is_global(self):
        if all(o.is_global is True for o, _ in self.args):
            return True
        if all(o.is_global is False for o, _ in self.args):
            return False
        return None

class InverseOrder(MonomialOrder):
    """
    The "inverse" of another monomial order.

    If O is any monomial order, we can construct another monomial order iO
    such that `A >_{iO} B` if and only if `B >_O A`. This is useful for
    constructing local orders.

    Note that many algorithms only work with *global* orders.

    For example, in the inverse lexicographic order on a single variable `x`,
    high powers of `x` count as small:

    >>> from sympy.polys.orderings import lex, InverseOrder
    >>> ilex = InverseOrder(lex)
    >>> ilex((5,)) < ilex((0,))
    True
    """

    def __init__(self, O):
        self.O = O

    def __str__(self):
        return "i" + str(self.O)

    def __call__(self, monomial):
        def inv(l):
            if iterable(l):
                return tuple(inv(x) for x in l)
            return -l
        return inv(self.O(monomial))

    @property
    def is_global(self):
        if self.O.is_global is True:
            return False
        if self.O.is_global is False:
            return True
        return None

    def __eq__(self, other):
        return isinstance(other, InverseOrder) and other.O == self.O

    def __hash__(self):
        return hash((self.__class__, self.O))

lex = LexOrder()
grlex = GradedLexOrder()
grevlex = ReversedGradedLexOrder()
ilex = InverseOrder(lex)
igrlex = InverseOrder(grlex)
igrevlex = InverseOrder(grevlex)

_monomial_key = {
    'lex': lex,
    'grlex': grlex,
    'grevlex': grevlex,
    'ilex': ilex,
    'igrlex': igrlex,
    'igrevlex': igrevlex
}

def monomial_key(order=None, gens=None):
    """
    Return a function defining admissible order on monomials.

    The result of a call to :func:`monomial_key` is a function which should
    be used as a key to :func:`sorted` built-in function, to provide order
    in a set of monomials of the same length.

    Currently supported monomial orderings are:

    1. lex       - lexicographic order (default)
    2. grlex     - graded lexicographic order
    3. grevlex   - reversed graded lexicographic order
    4. ilex, igrlex, igrevlex - the corresponding inverse orders

    If the ``order`` input argument is not a string but has ``__call__``
    attribute, then it will pass through with an assumption that the
    callable object defines an admissible order on monomials.

    If the ``gens`` input argument contains a list of generators, the
    resulting key function can be used to sort SymPy ``Expr`` objects.

    """
    if order is None:
        order = lex

    if isinstance(order, Symbol):
        order = str(order)

    if isinstance(order, str):
        try:
            order = _monomial_key[order]
        except KeyError:
            raise ValueError("supported monomial orderings are 'lex', 'grlex' and 'grevlex', got %r" % order)
    if hasattr(order, '__call__'):
        if gens is not None:
            def _order(expr):
                return order(expr.as_poly(*gens).degree_list())
            return _order
        return order
    else:
        raise ValueError("monomial ordering specification must be a string or a callable, got %s" % order)

class _ItemGetter:
    """Helper class to return a subsequence of values."""

    def __init__(self, seq):
        self.seq = tuple(seq)

    def __call__(self, m):
        return tuple(m[idx] for idx in self.seq)

    def __eq__(self, other):
        if not isinstance(other, _ItemGetter):
            return False
        return self.seq == other.seq

def build_product_order(arg, gens):
    """
    Build a monomial order on ``gens``.

    ``arg`` should be a tuple of iterables. The first element of each iterable
    should be a string or monomial order (will be passed to monomial_key),
    the others should be subsets of the generators. This function will build
    the corresponding product order.

    For example, build a product of two grlex orders:

    >>> from sympy.polys.orderings import build_product_order
    >>> from sympy.abc import x, y, z, t

    >>> O = build_product_order((("grlex", x, y), ("grlex", z, t)), [x, y, z, t])
    >>> O((1, 2, 3, 4))
    ((3, (1, 2)), (7, (3, 4)))

    """
    gens2idx = {}
    for i, g in enumerate(gens):
        gens2idx[g] = i
    order = []
    for expr in arg:
        name = expr[0]
        var = expr[1:]

        def makelambda(var):
            return _ItemGetter(gens2idx[g] for g in var)
        order.append((monomial_key(name), makelambda(var)))
    return ProductOrder(*order)
<|endoftext|>"""Power series evaluation and manipulation using sparse Polynomials

Implementing a new function
---------------------------

There are a few things to be kept in mind when adding a new function here::

    - The implementation should work on all possible input domains/rings.
      Special cases include the ``EX`` ring and a constant term in the series
      to be expanded. There can be two types of constant terms in the series:

        + A constant value or symbol.
        + A term of a multivariate series not involving the generator, with
          respect to which the series is to expanded.

      Strictly speaking, a generator of a ring should not be considered a
      constant. However, for series expansion both the cases need similar
      treatment (as the user does not care about inner details), i.e, use an
      addition formula to separate the constant part and the variable part (see
      rs_sin for reference).

    - All the algorithms used here are primarily designed to work for Taylor
      series (number of iterations in the algo equals the required order).
      Hence, it becomes tricky to get the series of the right order if a
      Puiseux series is input. Use rs_puiseux? in your function if your
      algorithm is not designed to handle fractional powers.

Extending rs_series
-------------------

To make a function work with rs_series you need to do two things::

    - Many sure it works with a constant term (as explained above).
    - If the series contains constant terms, you might need to extend its ring.
      You do so by adding the new terms to the rings as generators.
      ``PolyRing.compose`` and ``PolyRing.add_gens`` are two functions that do
      so and need to be called every time you expand a series containing a
      constant term.

Look at rs_sin and rs_series for further reference.

"""

from sympy.polys.domains import QQ, EX
from sympy.polys.rings import PolyElement, ring, sring
from sympy.polys.puiseux import PuiseuxPoly
from sympy.polys.polyerrors import DomainError
from sympy.polys.monomials import (monomial_min, monomial_mul, monomial_div,
                                   monomial_ldiv)
from mpmath.libmp.libintmath import ifac
from sympy.core import PoleError, Function, Expr
from sympy.core.numbers import Rational
from sympy.core.intfunc import igcd
from sympy.functions import (sin, cos, tan, atan, exp, atanh, asinh, tanh, log,
                             ceiling, sinh, cosh)
from sympy.utilities.misc import as_int
from mpmath.libmp.libintmath import giant_steps
import math


def _invert_monoms(p1):
    """
    Compute ``x**n * p1(1/x)`` for a univariate polynomial ``p1`` in ``x``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import _invert_monoms
    >>> R, x = ring('x', ZZ)
    >>> p = x**2 + 2*x + 3
    >>> _invert_monoms(p)
    3*x**2 + 2*x + 1

    See Also
    ========

    sympy.polys.densebasic.dup_reverse
    """
    terms = list(p1.items())
    terms.sort()
    deg = p1.degree()
    R = p1.ring
    p = R.zero
    cv = p1.listcoeffs()
    mv = p1.listmonoms()
    for mvi, cvi in zip(mv, cv):
        p[(deg - mvi[0],)] = cvi
    return p

def _giant_steps(target):
    """Return a list of precision steps for the Newton's method"""
    # We use ceil here because giant_steps cannot handle flint.fmpq
    res = giant_steps(2, math.ceil(target))
    if res[0] != 2:
        res = [2] + res
    return res

def rs_trunc(p1, x, prec):
    """
    Truncate the series in the ``x`` variable with precision ``prec``,
    that is, modulo ``O(x**prec)``

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_trunc
    >>> R, x = ring('x', QQ)
    >>> p = x**10 + x**5 + x + 1
    >>> rs_trunc(p, x, 12)
    x**10 + x**5 + x + 1
    >>> rs_trunc(p, x, 10)
    x**5 + x + 1
    """
    R = p1.ring
    p = {}
    i = R.gens.index(x)
    for exp1 in p1:
        if exp1[i] >= prec:
            continue
        p[exp1] = p1[exp1]
    return R(p)

def rs_is_puiseux(p, x):
    """
    Test if ``p`` is Puiseux series in ``x``.

    Raise an exception if it has a negative power in ``x``.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.puiseux import puiseux_ring
    >>> from sympy.polys.ring_series import rs_is_puiseux
    >>> R, x = puiseux_ring('x', QQ)
    >>> p = x**QQ(2,5) + x**QQ(2,3) + x
    >>> rs_is_puiseux(p, x)
    True
    """
    index = p.ring.gens.index(x)
    for k in p.itermonoms():
        if k[index] != int(k[index]):
            return True
        if k[index] < 0:
            raise ValueError('The series is not regular in %s' % x)
    return False

def rs_puiseux(f, p, x, prec):
    """
    Return the puiseux series for `f(p, x, prec)`.

    To be used when function ``f`` is implemented only for regular series.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.puiseux import puiseux_ring
    >>> from sympy.polys.ring_series import rs_puiseux, rs_exp
    >>> R, x = puiseux_ring('x', QQ)
    >>> p = x**QQ(2,5) + x**QQ(2,3) + x
    >>> rs_puiseux(rs_exp,p, x, 1)
    1 + x**(2/5) + x**(2/3) + 1/2*x**(4/5)
    """
    index = p.ring.gens.index(x)
    n = 1
    for k in p:
        power = k[index]
        if isinstance(power, Rational):
            num, den = power.as_numer_denom()
            n = int(n*den // igcd(n, den))
        elif power != int(power):
            den = power.denominator
            n = int(n*den // igcd(n, den))
    if n != 1:
        p1 = pow_xin(p, index, n)
        r = f(p1, x, prec*n)
        n1 = QQ(1, n)
        if isinstance(r, tuple):
            r = tuple([pow_xin(rx, index, n1) for rx in r])
        else:
            r = pow_xin(r, index, n1)
    else:
        r = f(p, x, prec)
    return r

def rs_puiseux2(f, p, q, x, prec):
    """
    Return the puiseux series for `f(p, q, x, prec)`.

    To be used when function ``f`` is implemented only for regular series.
    """
    index = p.ring.gens.index(x)
    n = 1
    for k in p:
        power = k[index]
        if isinstance(power, Rational):
            num, den = power.as_numer_denom()
            n = n*den // igcd(n, den)
        elif power != int(power):
            den = power.denominator
            n = n*den // igcd(n, den)
    if n != 1:
        p1 = pow_xin(p, index, n)
        r = f(p1, q, x, prec*n)
        n1 = QQ(1, n)
        r = pow_xin(r, index, n1)
    else:
        r = f(p, q, x, prec)
    return r

def rs_mul(p1, p2, x, prec):
    """
    Return the product of the given two series, modulo ``O(x**prec)``.

    ``x`` is the series variable or its position in the generators.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_mul
    >>> R, x = ring('x', QQ)
    >>> p1 = x**2 + 2*x + 1
    >>> p2 = x + 1
    >>> rs_mul(p1, p2, x, 3)
    3*x**2 + 3*x + 1
    """
    R = p1.ring
    p = {}
    if R.__class__ != p2.ring.__class__ or R != p2.ring:
        raise ValueError('p1 and p2 must have the same ring')
    iv = R.gens.index(x)
    if not isinstance(p2, (PolyElement, PuiseuxPoly)):
        raise ValueError('p2 must be a polynomial')
    if R == p2.ring:
        get = p.get
        items2 = p2.terms()
        items2.sort(key=lambda e: e[0][iv])
        if R.ngens == 1:
            for exp1, v1 in p1.iterterms():
                for exp2, v2 in items2:
                    exp = exp1[0] + exp2[0]
                    if exp < prec:
                        exp = (exp, )
                        p[exp] = get(exp, 0) + v1*v2
                    else:
                        break
        else:
            monomial_mul = R.monomial_mul
            for exp1, v1 in p1.iterterms():
                for exp2, v2 in items2:
                    if exp1[iv] + exp2[iv] < prec:
                        exp = monomial_mul(exp1, exp2)
                        p[exp] = get(exp, 0) + v1*v2
                    else:
                        break

    return R(p)

def rs_square(p1, x, prec):
    """
    Square the series modulo ``O(x**prec)``

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_square
    >>> R, x = ring('x', QQ)
    >>> p = x**2 + 2*x + 1
    >>> rs_square(p, x, 3)
    6*x**2 + 4*x + 1
    """
    R = p1.ring
    p = {}
    iv = R.gens.index(x)
    get = p.get
    items = p1.terms()
    items.sort(key=lambda e: e[0][iv])
    monomial_mul = R.monomial_mul
    for i in range(len(items)):
        exp1, v1 = items[i]
        for j in range(i):
            exp2, v2 = items[j]
            if exp1[iv] + exp2[iv] < prec:
                exp = monomial_mul(exp1, exp2)
                p[exp] = get(exp, 0) + v1*v2
            else:
                break
    p = {m: 2*v for m, v in p.items()}
    get = p.get
    for expv, v in p1.iterterms():
        if 2*expv[iv] < prec:
            e2 = monomial_mul(expv, expv)
            p[e2] = get(e2, 0) + v**2
    return R(p)

def rs_pow(p1, n, x, prec):
    """
    Return ``p1**n`` modulo ``O(x**prec)``

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_pow
    >>> R, x = ring('x', QQ)
    >>> p = x + 1
    >>> rs_pow(p, 4, x, 3)
    6*x**2 + 4*x + 1
    """
    R = p1.ring
    if isinstance(n, Rational):
        np = int(n.p)
        nq = int(n.q)
        if nq != 1:
            res = rs_nth_root(p1, nq, x, prec)
            if np != 1:
                res = rs_pow(res, np, x, prec)
        else:
            res = rs_pow(p1, np, x, prec)
        return res

    n = as_int(n)
    if n == 0:
        if p1:
            return R(1)
        else:
            raise ValueError('0**0 is undefined')
    if n < 0:
        p1 = rs_pow(p1, -n, x, prec)
        return rs_series_inversion(p1, x, prec)
    if n == 1:
        return rs_trunc(p1, x, prec)
    if n == 2:
        return rs_square(p1, x, prec)
    if n == 3:
        p2 = rs_square(p1, x, prec)
        return rs_mul(p1, p2, x, prec)
    p = R(1)
    while 1:
        if n & 1:
            p = rs_mul(p1, p, x, prec)
            n -= 1
            if not n:
                break
        p1 = rs_square(p1, x, prec)
        n = n // 2
    return p

def rs_subs(p, rules, x, prec):
    """
    Substitution with truncation according to the mapping in ``rules``.

    Return a series with precision ``prec`` in the generator ``x``

    Note that substitutions are not done one after the other

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_subs
    >>> R, x, y = ring('x, y', QQ)
    >>> p = x**2 + y**2
    >>> rs_subs(p, {x: x+ y, y: x+ 2*y}, x, 3)
    2*x**2 + 6*x*y + 5*y**2
    >>> (x + y)**2 + (x + 2*y)**2
    2*x**2 + 6*x*y + 5*y**2

    which differs from

    >>> rs_subs(rs_subs(p, {x: x+ y}, x, 3), {y: x+ 2*y}, x, 3)
    5*x**2 + 12*x*y + 8*y**2

    Parameters
    ----------
    p : :class:`~.PolyElement` Input series.
    rules : ``dict`` with substitution mappings.
    x : :class:`~.PolyElement` in which the series truncation is to be done.
    prec : :class:`~.Integer` order of the series after truncation.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_subs
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_subs(x**2+y**2, {y: (x+y)**2}, x, 3)
     6*x**2*y**2 + x**2 + 4*x*y**3 + y**4
    """
    R = p.ring
    ngens = R.ngens
    d = R(0)
    for i in range(ngens):
        d[(i, 1)] = R.gens[i]
    for var in rules:
        d[(R.index(var), 1)] = rules[var]
    p1 = R(0)
    p_keys = sorted(p.keys())
    for expv in p_keys:
        p2 = R(1)
        for i in range(ngens):
            power = expv[i]
            if power == 0:
                continue
            if (i, power) not in d:
                q, r = divmod(power, 2)
                if r == 0 and (i, q) in d:
                    d[(i, power)] = rs_square(d[(i, q)], x, prec)
                elif (i, power - 1) in d:
                    d[(i, power)] = rs_mul(d[(i, power - 1)], d[(i, 1)],
                                           x, prec)
                else:
                    d[(i, power)] = rs_pow(d[(i, 1)], power, x, prec)
            p2 = rs_mul(p2, d[(i, power)], x, prec)
        p1 += p2*p[expv]
    return p1

def _has_constant_term(p, x):
    """
    Check if ``p`` has a constant term in ``x``

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import _has_constant_term
    >>> R, x = ring('x', QQ)
    >>> p = x**2 + x + 1
    >>> _has_constant_term(p, x)
    True
    """
    R = p.ring
    iv = R.gens.index(x)
    zm = R.zero_monom
    a = [0]*R.ngens
    a[iv] = 1
    miv = tuple(a)
    return any(monomial_min(expv, miv) == zm for expv in p)

def _get_constant_term(p, x):
    """Return constant term in p with respect to x

    Note that it is not simply `p[R.zero_monom]` as there might be multiple
    generators in the ring R. We want the `x`-free term which can contain other
    generators.
    """
    R = p.ring
    i = R.gens.index(x)
    zm = R.zero_monom
    a = [0]*R.ngens
    a[i] = 1
    miv = tuple(a)
    c = 0
    for expv in p:
        if monomial_min(expv, miv) == zm:
            c += R({expv: p[expv]})
    return c

def _check_series_var(p, x, name):
    index = p.ring.gens.index(x)
    m = min(p, key=lambda k: k[index])[index]
    if m < 0:
        raise PoleError("Asymptotic expansion of %s around [oo] not "
                        "implemented." % name)
    return index, m

def _series_inversion1(p, x, prec):
    """
    Univariate series inversion ``1/p`` modulo ``O(x**prec)``.

    The Newton method is used.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import _series_inversion1
    >>> R, x = ring('x', QQ)
    >>> p = x + 1
    >>> _series_inversion1(p, x, 4)
    -x**3 + x**2 - x + 1
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(_series_inversion1, p, x, prec)
    R = p.ring
    zm = R.zero_monom
    c = p[zm]

    # giant_steps does not seem to work with PythonRational numbers with 1 as
    # denominator. This makes sure such a number is converted to integer.
    if prec == int(prec):
        prec = int(prec)

    if zm not in p:
        raise ValueError("No constant term in series")
    if _has_constant_term(p - c, x):
        raise ValueError("p cannot contain a constant term depending on "
                         "parameters")
    if not R.domain.is_unit(c):
        raise ValueError(f"Constant term {c} must be a unit in {R.domain}")

    one = R(1)
    if R.domain is EX:
        one = 1
    if c != one:
        p1 = R(1)/c
    else:
        p1 = R(1)
    for precx in _giant_steps(prec):
        t = 1 - rs_mul(p1, p, x, precx)
        p1 = p1 + rs_mul(p1, t, x, precx)
    return p1

def rs_series_inversion(p, x, prec):
    """
    Multivariate series inversion ``1/p`` modulo ``O(x**prec)``.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_series_inversion
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_series_inversion(1 + x*y**2, x, 4)
    -x**3*y**6 + x**2*y**4 - x*y**2 + 1
    >>> rs_series_inversion(1 + x*y**2, y, 4)
    -x*y**2 + 1
    >>> rs_series_inversion(x + x**2, x, 4)
    x**3 - x**2 + x - 1 + x**(-1)
    """
    R = p.ring
    if p == R.zero:
        raise ZeroDivisionError
    zm = R.zero_monom
    index = R.gens.index(x)
    m = min(p, key=lambda k: k[index])[index]
    if m:
        p = mul_xin(p, index, -m)
        prec = prec + m
    if zm not in p:
        raise NotImplementedError("No constant term in series")

    if _has_constant_term(p - p[zm], x):
        raise NotImplementedError("p - p[0] must not have a constant term in "
                                  "the series variables")
    r = _series_inversion1(p, x, prec)
    if m != 0:
        r = mul_xin(r, index, -m)
    return r

def _coefficient_t(p, t):
    r"""Coefficient of `x_i**j` in p, where ``t`` = (i, j)"""
    i, j = t
    R = p.ring
    expv1 = [0]*R.ngens
    expv1[i] = j
    expv1 = tuple(expv1)
    p1 = R(0)
    for expv in p:
        if expv[i] == j:
            p1[monomial_div(expv, expv1)] = p[expv]
    return p1

def rs_series_reversion(p, x, n, y):
    r"""
    Reversion of a series.

    ``p`` is a series with ``O(x**n)`` of the form $p = ax + f(x)$
    where $a$ is a number different from 0.

    $f(x) = \sum_{k=2}^{n-1} a_kx_k$

    Parameters
    ==========

      a_k : Can depend polynomially on other variables, not indicated.
      x : Variable with name x.
      y : Variable with name y.

    Returns
    =======

    Solve $p = y$, that is, given $ax + f(x) - y = 0$,
    find the solution $x = r(y)$ up to $O(y^n)$.

    Algorithm
    =========

    If $r_i$ is the solution at order $i$, then:
    $ar_i + f(r_i) - y = O\left(y^{i + 1}\right)$

    and if $r_{i + 1}$ is the solution at order $i + 1$, then:
    $ar_{i + 1} + f(r_{i + 1}) - y = O\left(y^{i + 2}\right)$

    We have, $r_{i + 1} = r_i + e$, such that,
    $ae + f(r_i) = O\left(y^{i + 2}\right)$
    or $e = -f(r_i)/a$

    So we use the recursion relation:
    $r_{i + 1} = r_i - f(r_i)/a$
    with the boundary condition: $r_1 = y$

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_series_reversion, rs_trunc
    >>> R, x, y, a, b = ring('x, y, a, b', QQ)
    >>> p = x - x**2 - 2*b*x**2 + 2*a*b*x**2
    >>> p1 = rs_series_reversion(p, x, 3, y); p1
    -2*y**2*a*b + 2*y**2*b + y**2 + y
    >>> rs_trunc(p.compose(x, p1), y, 3)
    y
    """
    if rs_is_puiseux(p, x):
        raise NotImplementedError
    R = p.ring
    nx = R.gens.index(x)
    y = R(y)
    ny = R.gens.index(y)
    if _has_constant_term(p, x):
        raise ValueError("p must not contain a constant term in the series "
                         "variable")
    a = _coefficient_t(p, (nx, 1))
    zm = R.zero_monom
    assert zm in a and len(a) == 1
    a = a[zm]
    r = y/a
    for i in range(2, n):
        sp = rs_subs(p, {x: r}, y, i + 1)
        sp = _coefficient_t(sp, (ny, i))*y**i
        r -= sp/a
    return r

def rs_series_from_list(p, c, x, prec, concur=1):
    """
    Return a series `sum c[n]*p**n` modulo `O(x**prec)`.

    It reduces the number of multiplications by summing concurrently.

    `ax = [1, p, p**2, .., p**(J - 1)]`
    `s = sum(c[i]*ax[i]` for i in `range(r, (r + 1)*J))*p**((K - 1)*J)`
    with `K >= (n + 1)/J`

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_series_from_list, rs_trunc
    >>> R, x = ring('x', QQ)
    >>> p = x**2 + x + 1
    >>> c = [1, 2, 3]
    >>> rs_series_from_list(p, c, x, 4)
    6*x**3 + 11*x**2 + 8*x + 6
    >>> rs_trunc(1 + 2*p + 3*p**2, x, 4)
    6*x**3 + 11*x**2 + 8*x + 6
    >>> pc = R.from_list(list(reversed(c)))
    >>> rs_trunc(pc.compose(x, p), x, 4)
    6*x**3 + 11*x**2 + 8*x + 6

    See Also
    ========

    sympy.polys.rings.PolyRing.compose

    """
    R = p.ring
    n = len(c)
    if not concur:
        q = R(1)
        s = c[0]*q
        for i in range(1, n):
            q = rs_mul(q, p, x, prec)
            s += c[i]*q
        return s
    J = int(math.sqrt(n) + 1)
    K, r = divmod(n, J)
    if r:
        K += 1
    ax = [R(1)]
    q = R(1)
    if len(p) < 20:
        for i in range(1, J):
            q = rs_mul(q, p, x, prec)
            ax.append(q)
    else:
        for i in range(1, J):
            if i % 2 == 0:
                q = rs_square(ax[i//2], x, prec)
            else:
                q = rs_mul(q, p, x, prec)
            ax.append(q)
    # optimize using rs_square
    pj = rs_mul(ax[-1], p, x, prec)
    b = R(1)
    s = R(0)
    for k in range(K - 1):
        r = J*k
        s1 = c[r]
        for j in range(1, J):
            s1 += c[r + j]*ax[j]
        s1 = rs_mul(s1, b, x, prec)
        s += s1
        b = rs_mul(b, pj, x, prec)
        if not b:
            break
    k = K - 1
    r = J*k
    if r < n:
        s1 = c[r]*R(1)
        for j in range(1, J):
            if r + j >= n:
                break
            s1 += c[r + j]*ax[j]
        s1 = rs_mul(s1, b, x, prec)
        s += s1
    return s

def rs_diff(p, x):
    """
    Return partial derivative of ``p`` with respect to ``x``.

    Parameters
    ==========

    x : :class:`~.PolyElement` with respect to which ``p`` is differentiated.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_diff
    >>> R, x, y = ring('x, y', QQ)
    >>> p = x + x**2*y**3
    >>> rs_diff(p, x)
    2*x*y**3 + 1
    """
    R = p.ring
    n = R.gens.index(x)
    p1 = {}
    mn = [0]*R.ngens
    mn[n] = 1
    mn = tuple(mn)
    for expv in p:
        if expv[n]:
            e = monomial_ldiv(expv, mn)
            p1[e] = R.domain_new(p[expv]*expv[n])
    return R(p1)

def rs_integrate(p, x):
    """
    Integrate ``p`` with respect to ``x``.

    Parameters
    ==========

    x : :class:`~.PolyElement` with respect to which ``p`` is integrated.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_integrate
    >>> R, x, y = ring('x, y', QQ)
    >>> p = x + x**2*y**3
    >>> rs_integrate(p, x)
    1/3*x**3*y**3 + 1/2*x**2
    """
    R = p.ring
    p1 = {}
    n = R.gens.index(x)
    mn = [0]*R.ngens
    mn[n] = 1
    mn = tuple(mn)

    for expv in p:
        e = monomial_mul(expv, mn)
        p1[e] = R.domain_new(p[expv]/(expv[n] + 1))
    return R(p1)

def rs_fun(p, f, *args):
    r"""
    Function of a multivariate series computed by substitution.

    The case with f method name is used to compute `rs\_tan` and `rs\_nth\_root`
    of a multivariate series:

        `rs\_fun(p, tan, iv, prec)`

        tan series is first computed for a dummy variable _x,
        i.e, `rs\_tan(\_x, iv, prec)`. Then we substitute _x with p to get the
        desired series

    Parameters
    ==========

    p : :class:`~.PolyElement` The multivariate series to be expanded.
    f : `ring\_series` function to be applied on `p`.
    args[-2] : :class:`~.PolyElement` with respect to which, the series is to be expanded.
    args[-1] : Required order of the expanded series.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_fun, _tan1
    >>> R, x, y = ring('x, y', QQ)
    >>> p = x + x*y + x**2*y + x**3*y**2
    >>> rs_fun(p, _tan1, x, 4)
    1/3*x**3*y**3 + 2*x**3*y**2 + x**3*y + 1/3*x**3 + x**2*y + x*y + x
    """
    _R = p.ring
    R1, _x = ring('_x', _R.domain)
    h = int(args[-1])
    args1 = args[:-2] + (_x, h)
    zm = _R.zero_monom
    # separate the constant term of the series
    # compute the univariate series f(_x, .., 'x', sum(nv))
    if zm in p:
        x1 = _x + p[zm]
        p1 = p - p[zm]
    else:
        x1 = _x
        p1 = p
    if isinstance(f, str):
        q = getattr(x1, f)(*args1)
    else:
        q = f(x1, *args1)
    a = sorted(q.items())
    c = [0]*h
    for x in a:
        c[x[0][0]] = x[1]
    p1 = rs_series_from_list(p1, c, args[-2], args[-1])
    return p1

def mul_xin(p, i, n):
    r"""
    Return `p*x_i**n`.

    `x\_i` is the ith variable in ``p``.
    """
    R = p.ring
    q = {}
    for k, v in p.terms():
        k1 = list(k)
        k1[i] += n
        q[tuple(k1)] = v
    return R(q)

def pow_xin(p, i, n):
    """
    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.puiseux import puiseux_ring
    >>> from sympy.polys.ring_series import pow_xin
    >>> R, x, y = puiseux_ring('x, y', QQ)
    >>> p = x**QQ(2,5) + x + x**QQ(2,3)
    >>> index = p.ring.gens.index(x)
    >>> pow_xin(p, index, 15)
    x**6 + x**10 + x**15
    """
    R = p.ring
    q = {}
    for k, v in p.terms():
        k1 = list(k)
        k1[i] *= n
        q[tuple(k1)] = v
    return R(q)

def _nth_root1(p, n, x, prec):
    """
    Univariate series expansion of the nth root of ``p``.

    The Newton method is used.
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux2(_nth_root1, p, n, x, prec)
    R = p.ring
    zm = R.zero_monom
    if zm not in p:
        raise NotImplementedError('No constant term in series')
    n = as_int(n)
    assert p[zm] == 1
    p1 = R(1)
    if p == 1:
        return p
    if n == 0:
        return R(1)
    if n == 1:
        return p
    if n < 0:
        n = -n
        sign = 1
    else:
        sign = 0
    for precx in _giant_steps(prec):
        tmp = rs_pow(p1, n + 1, x, precx)
        tmp = rs_mul(tmp, p, x, precx)
        p1 += p1/n - tmp/n
    if sign:
        return p1
    else:
        return _series_inversion1(p1, x, prec)

def rs_nth_root(p, n, x, prec):
    """
    Multivariate series expansion of the nth root of ``p``.

    Parameters
    ==========

    p : Expr
        The polynomial to computer the root of.
    n : integer
        The order of the root to be computed.
    x : :class:`~.PolyElement`
    prec : integer
        Order of the expanded series.

    Notes
    =====

    The result of this function is dependent on the ring over which the
    polynomial has been defined. If the answer involves a root of a constant,
    make sure that the polynomial is over a real field. It cannot yet handle
    roots of symbols.

    Examples
    ========

    >>> from sympy.polys.domains import QQ, RR
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_nth_root
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_nth_root(1 + x + x*y, -3, x, 3)
    2/9*x**2*y**2 + 4/9*x**2*y + 2/9*x**2 - 1/3*x*y - 1/3*x + 1
    >>> R, x, y = ring('x, y', RR)
    >>> rs_nth_root(3 + x + x*y, 3, x, 2)
    0.160249952256379*x*y + 0.160249952256379*x + 1.44224957030741
    """
    if n == 0:
        if p == 0:
            raise ValueError('0**0 expression')
        else:
            return p.ring(1)
    if n == 1:
        return rs_trunc(p, x, prec)
    R = p.ring
    index = R.gens.index(x)
    m = min(p, key=lambda k: k[index])[index]
    p = mul_xin(p, index, -m)
    prec -= m

    if _has_constant_term(p - 1, x):
        zm = R.zero_monom
        c = p[zm]
        if isinstance(c, PolyElement):
            try:
                c_expr = c.as_expr()
                const = R(c_expr**(QQ(1, n)))
            except ValueError:
                raise DomainError("The given series cannot be expanded in "
                    "this domain.")
        else:
            try:                              # RealElement doesn't support
                const = R(c**Rational(1, n))  # exponentiation with mpq object
            except ValueError:                # as exponent
                raise DomainError("The given series cannot be expanded in "
                    "this domain.")
        res = rs_nth_root(p/c, n, x, prec)*const
    else:
        res = _nth_root1(p, n, x, prec)
    if m:
        m = QQ(m) / n
        res = mul_xin(res, index, m)
    return res

def rs_log(p, x, prec):
    """
    The Logarithm of ``p`` modulo ``O(x**prec)``.

    Notes
    =====

    Truncation of ``integral dx p**-1*d p/dx`` is used.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.puiseux import puiseux_ring
    >>> from sympy.polys.ring_series import rs_log
    >>> R, x = puiseux_ring('x', QQ)
    >>> rs_log(1 + x, x, 8)
    x + -1/2*x**2 + 1/3*x**3 + -1/4*x**4 + 1/5*x**5 + -1/6*x**6 + 1/7*x**7
    >>> rs_log(x**QQ(3, 2) + 1, x, 5)
    x**(3/2) + -1/2*x**3 + 1/3*x**(9/2)
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_log, p, x, prec)
    R = p.ring
    if p == 1:
        return R.zero
    c = _get_constant_term(p, x)
    if c:
        const = 0
        if c == 1:
            pass
        try:
            c_expr = c.as_expr()
            const = R(log(c_expr))
        except ValueError:
            R = R.add_gens([log(c_expr)])
            p = p.set_ring(R)
            x = x.set_ring(R)
            c = c.set_ring(R)
            const = R(log(c_expr))

        dlog = p.diff(x)
        dlog = rs_mul(dlog, _series_inversion1(p, x, prec), x, prec - 1)
        return rs_integrate(dlog, x) + const
    else:
        raise NotImplementedError

def rs_LambertW(p, x, prec):
    """
    Calculate the series expansion of the principal branch of the Lambert W
    function.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_LambertW
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_LambertW(x + x*y, x, 3)
    -x**2*y**2 - 2*x**2*y - x**2 + x*y + x

    See Also
    ========

    LambertW
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_LambertW, p, x, prec)
    R = p.ring
    p1 = R(0)
    if _has_constant_term(p, x):
        raise NotImplementedError("Polynomial must not have constant term in "
                                  "the series variables")
    if x in R.gens:
        for precx in _giant_steps(prec):
            e = rs_exp(p1, x, precx)
            p2 = rs_mul(e, p1, x, precx) - p
            p3 = rs_mul(e, p1 + 1, x, precx)
            p3 = rs_series_inversion(p3, x, precx)
            tmp = rs_mul(p2, p3, x, precx)
            p1 -= tmp
        return p1
    else:
        raise NotImplementedError

def _exp1(p, x, prec):
    r"""Helper function for `rs\_exp`. """
    R = p.ring
    p1 = R(1)
    for precx in _giant_steps(prec):
        pt = p - rs_log(p1, x, precx)
        tmp = rs_mul(pt, p1, x, precx)
        p1 += tmp
    return p1

def rs_exp(p, x, prec):
    """
    Exponentiation of a series modulo ``O(x**prec)``

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_exp
    >>> R, x = ring('x', QQ)
    >>> rs_exp(x**2, x, 7)
    1/6*x**6 + 1/2*x**4 + x**2 + 1
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_exp, p, x, prec)
    R = p.ring
    c = _get_constant_term(p, x)
    if c:
        try:
            c_expr = c.as_expr()
            const = R(exp(c_expr))
        except ValueError:
            R = R.add_gens([exp(c_expr)])
            p = p.set_ring(R)
            x = x.set_ring(R)
            c = c.set_ring(R)
            const = R(exp(c_expr))

        p1 = p - c

    # Makes use of SymPy functions to evaluate the values of the cos/sin
    # of the constant term.
        return const*rs_exp(p1, x, prec)

    if len(p) > 20:
        return _exp1(p, x, prec)
    one = R(1)
    n = 1
    c = []
    for k in range(prec):
        c.append(one/n)
        k += 1
        n *= k

    r = rs_series_from_list(p, c, x, prec)
    return r

def _atan(p, iv, prec):
    """
    Expansion using formula.

    Faster on very small and univariate series.
    """
    R = p.ring
    mo = R(-1)
    c = [-mo]
    p2 = rs_square(p, iv, prec)
    for k in range(1, prec):
        c.append(mo**k/(2*k + 1))
    s = rs_series_from_list(p2, c, iv, prec)
    s = rs_mul(s, p, iv, prec)
    return s

def rs_atan(p, x, prec):
    """
    The arctangent of a series

    Return the series expansion of the atan of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_atan
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_atan(x + x*y, x, 4)
    -1/3*x**3*y**3 - x**3*y**2 - x**3*y - 1/3*x**3 + x*y + x

    See Also
    ========

    atan
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_atan, p, x, prec)
    R = p.ring
    const = 0
    c = _get_constant_term(p, x)
    if c:
        try:
            c_expr = c.as_expr()
            const = R(atan(c_expr))
        except ValueError:
            R = R.add_gens([atan(c_expr)])
            p = p.set_ring(R)
            x = x.set_ring(R)
            c = c.set_ring(R)
            const = R(atan(c_expr))

    # Instead of using a closed form formula, we differentiate atan(p) to get
    # `1/(1+p**2) * dp`, whose series expansion is much easier to calculate.
    # Finally we integrate to get back atan
    dp = p.diff(x)
    p1 = rs_square(p, x, prec) + R(1)
    p1 = rs_series_inversion(p1, x, prec - 1)
    p1 = rs_mul(dp, p1, x, prec - 1)
    return rs_integrate(p1, x) + const

def rs_asin(p, x, prec):
    """
    Arcsine of a series

    Return the series expansion of the asin of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_asin
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_asin(x, x, 8)
    5/112*x**7 + 3/40*x**5 + 1/6*x**3 + x

    See Also
    ========

    asin
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_asin, p, x, prec)
    if _has_constant_term(p, x):
        raise NotImplementedError("Polynomial must not have constant term in "
                                    "series variables")
    R = p.ring
    if x in R.gens:
        # get a good value
        if len(p) > 20:
            dp = rs_diff(p, x)
            p1 = 1 - rs_square(p, x, prec - 1)
            p1 = rs_nth_root(p1, -2, x, prec - 1)
            p1 = rs_mul(dp, p1, x, prec - 1)
            return rs_integrate(p1, x)
        one = R(1)
        c = [0, one, 0]
        for k in range(3, prec, 2):
            c.append((k - 2)**2*c[-2]/(k*(k - 1)))
            c.append(0)
        return rs_series_from_list(p, c, x, prec)

    else:
        raise NotImplementedError

def _tan1(p, x, prec):
    r"""
    Helper function of :func:`rs_tan`.

    Return the series expansion of tan of a univariate series using Newton's
    method. It takes advantage of the fact that series expansion of atan is
    easier than that of tan.

    Consider `f(x) = y - \arctan(x)`
    Let r be a root of f(x) found using Newton's method.
    Then `f(r) = 0`
    Or `y = \arctan(x)` where `x = \tan(y)` as required.
    """
    R = p.ring
    p1 = R(0)
    for precx in _giant_steps(prec):
        tmp = p - rs_atan(p1, x, precx)
        tmp = rs_mul(tmp, 1 + rs_square(p1, x, precx), x, precx)
        p1 += tmp
    return p1

def rs_tan(p, x, prec):
    """
    Tangent of a series.

    Return the series expansion of the tan of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_tan
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_tan(x + x*y, x, 4)
    1/3*x**3*y**3 + x**3*y**2 + x**3*y + 1/3*x**3 + x*y + x

   See Also
   ========

   _tan1, tan
   """
    if rs_is_puiseux(p, x):
        r = rs_puiseux(rs_tan, p, x, prec)
        return r
    R = p.ring
    const = 0
    c = _get_constant_term(p, x)
    if c:
        try:
            c_expr = c.as_expr()
            const = R(tan(c_expr))
        except ValueError:
            R = R.add_gens([tan(c_expr, )])
            p = p.set_ring(R)
            x = x.set_ring(R)
            c = c.set_ring(R)
            const = R(tan(c_expr))

        p1 = p - c

    # Makes use of SymPy functions to evaluate the values of the cos/sin
    # of the constant term.
        t2 = rs_tan(p1, x, prec)
        t = rs_series_inversion(1 - const*t2, x, prec)
        return rs_mul(const + t2, t, x, prec)

    if R.ngens == 1:
        return _tan1(p, x, prec)
    else:
        return rs_fun(p, rs_tan, x, prec)

def rs_cot(p, x, prec):
    """
    Cotangent of a series

    Return the series expansion of the cot of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_cot
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_cot(x, x, 6)
    -2/945*x**5 - 1/45*x**3 - 1/3*x + x**(-1)

    See Also
    ========

    cot
    """
    # It can not handle series like `p = x + x*y` where the coefficient of the
    # linear term in the series variable is symbolic.
    if rs_is_puiseux(p, x):
        r = rs_puiseux(rs_cot, p, x, prec)
        return r
    i, m = _check_series_var(p, x, 'cot')
    prec1 = int(prec + 2*m)
    c, s = rs_cos_sin(p, x, prec1)
    s = mul_xin(s, i, -m)
    s = rs_series_inversion(s, x, prec1)
    res = rs_mul(c, s, x, prec1)
    res = mul_xin(res, i, -m)
    res = rs_trunc(res, x, prec)
    return res

def rs_sin(p, x, prec):
    """
    Sine of a series

    Return the series expansion of the sin of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.puiseux import puiseux_ring
    >>> from sympy.polys.ring_series import rs_sin
    >>> R, x, y = puiseux_ring('x, y', QQ)
    >>> rs_sin(x + x*y, x, 4)
    x + x*y + -1/6*x**3 + -1/2*x**3*y + -1/2*x**3*y**2 + -1/6*x**3*y**3
    >>> rs_sin(x**QQ(3, 2) + x*y**QQ(7, 5), x, 4)
    x*y**(7/5) + x**(3/2) + -1/6*x**3*y**(21/5) + -1/2*x**(7/2)*y**(14/5)

    See Also
    ========

    sin
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_sin, p, x, prec)
    R = x.ring
    if not p:
        return R(0)
    c = _get_constant_term(p, x)
    if c:
        try:
            c_expr = c.as_expr()
            t1, t2 = R(sin(c_expr)), R(cos(c_expr))
        except ValueError:
            R = R.add_gens([sin(c_expr), cos(c_expr)])
            p = p.set_ring(R)
            x = x.set_ring(R)
            c = c.set_ring(R)
            t1, t2 = R(sin(c_expr)), R(cos(c_expr))

        p1 = p - c

    # Makes use of SymPy cos, sin functions to evaluate the values of the
    # cos/sin of the constant term.
        p_cos, p_sin = rs_cos_sin(p1, x, prec)
        return p_sin*t2 + p_cos*t1

    # Series is calculated in terms of tan as its evaluation is fast.
    if len(p) > 20 and R.ngens == 1:
        t = rs_tan(p/2, x, prec)
        t2 = rs_square(t, x, prec)
        p1 = rs_series_inversion(1 + t2, x, prec)
        return rs_mul(p1, 2*t, x, prec)
    one = R(1)
    n = 1
    c = [0]
    for k in range(2, prec + 2, 2):
        c.append(one/n)
        c.append(0)
        n *= -k*(k + 1)
    return rs_series_from_list(p, c, x, prec)

def rs_cos(p, x, prec):
    """
    Cosine of a series

    Return the series expansion of the cos of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.puiseux import puiseux_ring
    >>> from sympy.polys.ring_series import rs_cos
    >>> R, x, y = puiseux_ring('x, y', QQ)
    >>> rs_cos(x + x*y, x, 4)
    1 + -1/2*x**2 + -1*x**2*y + -1/2*x**2*y**2
    >>> rs_cos(x + x*y, x, 4)/x**QQ(7, 5)
    x**(-7/5) + -1/2*x**(3/5) + -1*x**(3/5)*y + -1/2*x**(3/5)*y**2

    See Also
    ========

    cos
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_cos, p, x, prec)
    R = p.ring
    c = _get_constant_term(p, x)
    if c:
        try:
            c_expr = c.as_expr()
            t1, t2 = R(sin(c_expr)), R(cos(c_expr))
        except ValueError:
            R = R.add_gens([sin(c_expr), cos(c_expr)])
            p = p.set_ring(R)
            x = x.set_ring(R)
            c = c.set_ring(R)
            t1, t2 = R(sin(c_expr)), R(cos(c_expr))

        p1 = p - c
        # Makes use of SymPy cos, sin functions to evaluate the values of the
        # cos/sin of the constant term.
        p_cos, p_sin = rs_cos_sin(p1, x, prec)
        return p_cos*t2 - p_sin*t1

    # Series is calculated in terms of tan as its evaluation is fast.
    if len(p) > 20 and R.ngens == 1:
        t = rs_tan(p/2, x, prec)
        t2 = rs_square(t, x, prec)
        p1 = rs_series_inversion(1+t2, x, prec)
        return rs_mul(p1, 1 - t2, x, prec)
    one = R(1)
    n = 1
    c = []
    for k in range(2, prec + 2, 2):
        c.append(one/n)
        c.append(0)
        n *= -k*(k - 1)
    return rs_series_from_list(p, c, x, prec)

def rs_cos_sin(p, x, prec):
    """
    Cosine and sine of a series

    Return the series expansion of the cosine and sine of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_cos_sin
    >>> R, x, y = ring('x, y', QQ)
    >>> c, s = rs_cos_sin(x + x*y, x, 4)
    >>> c
    -1/2*x**2*y**2 - x**2*y - 1/2*x**2 + 1
    >>> s
    -1/6*x**3*y**3 - 1/2*x**3*y**2 - 1/2*x**3*y - 1/6*x**3 + x*y + x

    See Also
    ========

    rs_cos, rs_sin
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_cos_sin, p, x, prec)
    R = p.ring
    if not p:
        return R(0), R(0)
    c = _get_constant_term(p, x)
    if c:
        try:
            c_expr = c.as_expr()
            t1, t2 = R(sin(c_expr)), R(cos(c_expr))
        except ValueError:
            R = R.add_gens([sin(c_expr), cos(c_expr)])
            p = p.set_ring(R)
            x = x.set_ring(R)
            c = c.set_ring(R)
            t1, t2 = R(sin(c_expr)), R(cos(c_expr))

        p1 = p - c
        p_cos, p_sin = rs_cos_sin(p1, x, prec)
        return p_cos*t2 - p_sin*t1, p_cos*t1 + p_sin*t2

    if len(p) > 20 and R.ngens == 1:
        t = rs_tan(p/2, x, prec)
        t2 = rs_square(t, x, prec)
        p1 = rs_series_inversion(1 + t2, x, prec)
        return (rs_mul(p1, 1 - t2, x, prec), rs_mul(p1, 2*t, x, prec))

    one = R(1)
    coeffs = []
    cn, sn = 1, 1
    for k in range(2, prec+2, 2):
        coeffs.extend([(one/cn, 0), (0, one/sn)])
        cn, sn = -cn*k*(k - 1), -sn*k*(k + 1)

    c, s = zip(*coeffs)
    return (rs_series_from_list(p, c, x, prec), rs_series_from_list(p, s, x, prec))

def _atanh(p, x, prec):
    """
    Expansion using formula

    Faster for very small and univariate series
    """
    R = p.ring
    one = R(1)
    c = [one]
    p2 = rs_square(p, x, prec)
    for k in range(1, prec):
        c.append(one/(2*k + 1))
    s = rs_series_from_list(p2, c, x, prec)
    s = rs_mul(s, p, x, prec)
    return s

def rs_atanh(p, x, prec):
    """
    Hyperbolic arctangent of a series

    Return the series expansion of the atanh of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_atanh
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_atanh(x + x*y, x, 4)
    1/3*x**3*y**3 + x**3*y**2 + x**3*y + 1/3*x**3 + x*y + x

    See Also
    ========

    atanh
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_atanh, p, x, prec)
    R = p.ring
    const = 0
    c = _get_constant_term(p, x)
    if c:
        try:
            c_expr = c.as_expr()
            const = R(atanh(c_expr))
        except ValueError:
            raise DomainError("The given series cannot be expanded in "
                "this domain.")

    # Instead of using a closed form formula, we differentiate atanh(p) to get
    # `1/(1-p**2) * dp`, whose series expansion is much easier to calculate.
    # Finally we integrate to get back atanh
    dp = rs_diff(p, x)
    p1 = - rs_square(p, x, prec) + 1
    p1 = rs_series_inversion(p1, x, prec - 1)
    p1 = rs_mul(dp, p1, x, prec - 1)
    return rs_integrate(p1, x) + const

def rs_asinh(p, x, prec):
    """
    Hyperbolic arcsine of a series

    Return the series expansion of the arcsinh of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_asinh
    >>> R, x = ring('x', QQ)
    >>> rs_asinh(x, x, 9)
    -5/112*x**7 + 3/40*x**5 - 1/6*x**3 + x

    See Also
    ========

    asinh
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_asinh, p, x, prec)
    R = p.ring
    const = 0
    c = _get_constant_term(p, x)
    if c:
        try:
            c_expr = c.as_expr()
            const = R(asinh(c_expr))
        except ValueError:
            raise DomainError("The given series cannot be expanded in "
                "this domain.")

    # Instead of using a closed form formula, we differentiate asinh(p) to get
    # `1/sqrt(1+p**2) * dp`, whose series expansion is much easier to calculate.
    # Finally we integrate to get back asinh
    dp = rs_diff(p, x)
    p_squared = rs_square(p, x, prec)
    denom = p_squared + R(1)
    p1 = rs_nth_root(denom, -2, x, prec - 1)
    p1 = rs_mul(dp, p1, x, prec - 1)
    return rs_integrate(p1, x) + const

def rs_sinh(p, x, prec):
    """
    Hyperbolic sine of a series

    Return the series expansion of the sinh of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_sinh
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_sinh(x + x*y, x, 4)
    1/6*x**3*y**3 + 1/2*x**3*y**2 + 1/2*x**3*y + 1/6*x**3 + x*y + x

    See Also
    ========

    sinh
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_sinh, p, x, prec)
    R = p.ring
    if not p:
        return R(0)
    c = _get_constant_term(p, x)
    if c:
        try:
            c_expr = c.as_expr()
            t1, t2 = R(sinh(c_expr)), R(cosh(c_expr))
        except ValueError:
            R = R.add_gens([sinh(c_expr), cosh(c_expr)])
            p = p.set_ring(R)
            x = x.set_ring(R)
            c = c.set_ring(R)
            t1, t2 = R(sinh(c_expr)), R(cosh(c_expr))

        p1 = p - c
        p_cosh, p_sinh = rs_cosh_sinh(p1, x, prec)
        return p_sinh * t2 + p_cosh * t1

    t = rs_exp(p, x, prec)
    t1 = rs_series_inversion(t, x, prec)
    return (t - t1)/2

def rs_cosh(p, x, prec):
    """
    Hyperbolic cosine of a series

    Return the series expansion of the cosh of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_cosh
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_cosh(x + x*y, x, 4)
    1/2*x**2*y**2 + x**2*y + 1/2*x**2 + 1

    See Also
    ========

    cosh
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_cosh, p, x, prec)
    R = p.ring
    if not p:
        return R(0)
    c = _get_constant_term(p, x)
    if c:
        try:
            c_expr = c.as_expr()
            t1, t2 = R(sinh(c_expr)), R(cosh(c_expr))
        except ValueError:
            R = R.add_gens([sinh(c_expr), cosh(c_expr)])
            p = p.set_ring(R)
            x = x.set_ring(R)
            c = c.set_ring(R)
            t1, t2 = R(sinh(c_expr)), R(cosh(c_expr))

        p1 = p - c
        p_cosh, p_sinh = rs_cosh_sinh(p1, x, prec)
        return p_cosh * t2 + p_sinh * t1

    t = rs_exp(p, x, prec)
    t1 = rs_series_inversion(t, x, prec)
    return (t + t1)/2

def rs_cosh_sinh(p, x, prec):
    """
    Hyperbolic cosine and sine of a series

    Return the series expansion of the hyperbolic cosine and sine of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_cosh_sinh
    >>> R, x, y = ring('x, y', QQ)
    >>> c, s = rs_cosh_sinh(x + x*y, x, 4)
    >>> c
    1/2*x**2*y**2 + x**2*y + 1/2*x**2 + 1
    >>> s
    1/6*x**3*y**3 + 1/2*x**3*y**2 + 1/2*x**3*y + 1/6*x**3 + x*y + x

    See Also
    ========

    rs_cosh, rs_sinh
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_cosh_sinh, p, x, prec)
    R = p.ring
    if not p:
        return R(0), R(0)
    c = _get_constant_term(p, x)
    if c:
        try:
            c_expr = c.as_expr()
            t1, t2 = R(sinh(c_expr)), R(cosh(c_expr))
        except ValueError:
            R = R.add_gens([sinh(c_expr), cosh(c_expr)])
            p = p.set_ring(R)
            x = x.set_ring(R)
            c = c.set_ring(R)
            t1, t2 = R(sinh(c_expr)), R(cosh(c_expr))

        p1 = p - c
        p_cosh, p_sinh = rs_cosh_sinh(p1, x, prec)
        return p_cosh * t2 + p_sinh * t1, p_sinh * t2 + p_cosh * t1

    t = rs_exp(p, x, prec)
    t1 = rs_series_inversion(t, x, prec)
    return (t + t1)/2, (t - t1)/2


def _tanh(p, x, prec):
    r"""
    Helper function of :func:`rs_tanh`

    Return the series expansion of tanh of a univariate series using Newton's
    method. It takes advantage of the fact that series expansion of atanh is
    easier than that of tanh.

    See Also
    ========

    _tanh
    """
    R = p.ring
    p1 = R(0)
    for precx in _giant_steps(prec):
        tmp = p - rs_atanh(p1, x, precx)
        tmp = rs_mul(tmp, 1 - rs_square(p1, x, prec), x, precx)
        p1 += tmp
    return p1

def rs_tanh(p, x, prec):
    """
    Hyperbolic tangent of a series

    Return the series expansion of the tanh of ``p``, about 0.

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_tanh
    >>> R, x, y = ring('x, y', QQ)
    >>> rs_tanh(x + x*y, x, 4)
    -1/3*x**3*y**3 - x**3*y**2 - x**3*y - 1/3*x**3 + x*y + x

    See Also
    ========

    tanh
    """
    if rs_is_puiseux(p, x):
        return rs_puiseux(rs_tanh, p, x, prec)
    R = p.ring
    const = 0
    c = _get_constant_term(p, x)
    if c:
        try:
            c_expr = c.as_expr()
            const = R(tanh(c_expr))
        except ValueError:
            R = R.add_gens([tanh(c_expr)])
            p = p.set_ring(R)
            x = x.set_ring(R)
            c = c.set_ring(R)
            const = R(tanh(c_expr))

        p1 = p - c
        t1 = rs_tanh(p1, x, prec)
        t = rs_series_inversion(1 + const*t1, x, prec)
        return rs_mul(const + t1, t, x, prec)

    if R.ngens == 1:
        return _tanh(p, x, prec)
    else:
        return rs_fun(p, _tanh, x, prec)

def rs_newton(p, x, prec):
    """
    Compute the truncated Newton sum of the polynomial ``p``

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_newton
    >>> R, x = ring('x', QQ)
    >>> p = x**2 - 2
    >>> rs_newton(p, x, 5)
    8*x**4 + 4*x**2 + 2
    """
    deg = p.degree()
    p1 = _invert_monoms(p)
    p2 = rs_series_inversion(p1, x, prec)
    p3 = rs_mul(p1.diff(x), p2, x, prec)
    res = deg - p3*x
    return res

def rs_hadamard_exp(p1, inverse=False):
    """
    Return ``sum f_i/i!*x**i`` from ``sum f_i*x**i``,
    where ``x`` is the first variable.

    If ``inverse=True`` return ``sum f_i*i!*x**i``

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_hadamard_exp
    >>> R, x = ring('x', QQ)
    >>> p = 1 + x + x**2 + x**3
    >>> rs_hadamard_exp(p)
    1/6*x**3 + 1/2*x**2 + x + 1
    """
    R = p1.ring
    if R.domain != QQ:
        raise NotImplementedError
    p = R.zero
    if not inverse:
        for exp1, v1 in p1.items():
            p[exp1] = v1/int(ifac(exp1[0]))
    else:
        for exp1, v1 in p1.items():
            p[exp1] = v1*int(ifac(exp1[0]))
    return p

def rs_compose_add(p1, p2):
    """
    compute the composed sum ``prod(p2(x - beta) for beta root of p1)``

    Examples
    ========

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.ring_series import rs_compose_add
    >>> R, x = ring('x', QQ)
    >>> f = x**2 - 2
    >>> g = x**2 - 3
    >>> rs_compose_add(f, g)
    x**4 - 10*x**2 + 1

    References
    ==========

    .. [1] A. Bostan, P. Flajolet, B. Salvy and E. Schost
           "Fast Computation with Two Algebraic Numbers",
           (2002) Research Report 4579, Institut
           National de Recherche en Informatique et en Automatique
    """
    R = p1.ring
    x = R.gens[0]
    prec = p1.degree()*p2.degree() + 1
    np1 = rs_newton(p1, x, prec)
    np1e = rs_hadamard_exp(np1)
    np2 = rs_newton(p2, x, prec)
    np2e = rs_hadamard_exp(np2)
    np3e = rs_mul(np1e, np2e, x, prec)
    np3 = rs_hadamard_exp(np3e, True)
    np3a = (np3[(0,)] - np3) / x
    q = rs_integrate(np3a, x)
    q = rs_exp(q, x, prec)
    q = _invert_monoms(q)
    q = q.primitive()[1]
    dp = p1.degree()*p2.degree() - q.degree()
    # `dp` is the multiplicity of the zeroes of the resultant;
    # these zeroes are missed in this computation so they are put here.
    # if p1 and p2 are monic irreducible polynomials,
    # there are zeroes in the resultant
    # if and only if p1 = p2 ; in fact in that case p1 and p2 have a
    # root in common, so gcd(p1, p2) != 1; being p1 and p2 irreducible
    # this means p1 = p2
    if dp:
        q = q*x**dp
    return q


_convert_func = {
        'sin': 'rs_sin',
        'cos': 'rs_cos',
        'exp': 'rs_exp',
        'tan': 'rs_tan',
        'log': 'rs_log',
        'atan': 'rs_atan',
        'sinh': 'rs_sinh',
        'cosh': 'rs_cosh',
        'tanh': 'rs_tanh'
        }

def rs_min_pow(expr, series_rs, a):
    """Find the minimum power of `a` in the series expansion of expr"""
    series = 0
    n = 2
    while series == 0:
        series = _rs_series(expr, series_rs, a, n)
        n *= 2
    R = series.ring
    a = R(a)
    i = R.gens.index(a)
    return min(series, key=lambda t: t[i])[i]


def _rs_series(expr, series_rs, a, prec):
    # TODO Use _parallel_dict_from_expr instead of sring as sring is
    # inefficient. For details, read the todo in sring.
    args = expr.args
    R = series_rs.ring

    # expr does not contain any function to be expanded
    if not any(arg.has(Function) for arg in args) and not expr.is_Function:
        return series_rs

    if not expr.has(a):
        return series_rs

    elif expr.is_Function:
        arg = args[0]
        if len(args) > 1:
            raise NotImplementedError
        R1, series = sring(arg, domain=QQ, expand=False, series=True)
        series_inner = _rs_series(arg, series, a, prec)

        # Why do we need to compose these three rings?
        #
        # We want to use a simple domain (like ``QQ`` or ``RR``) but they don't
        # support symbolic coefficients. We need a ring that for example lets
        # us have `sin(1)` and `cos(1)` as coefficients if we are expanding
        # `sin(x + 1)`. The ``EX`` domain allows all symbolic coefficients, but
        # that makes it very complex and hence slow.
        #
        # To solve this problem, we add only those symbolic elements as
        # generators to our ring, that we need. Here, series_inner might
        # involve terms like `sin(4)`, `exp(a)`, etc, which are not there in
        # R1 or R. Hence, we compose these three rings to create one that has
        # the generators of all three.
        R = R.compose(R1).compose(series_inner.ring)
        series_inner = series_inner.set_ring(R)
        series = eval(_convert_func[str(expr.func)])(series_inner,
            R(a), prec)
        return series

    elif expr.is_Mul:
        n = len(args)
        for arg in args:    # XXX Looks redundant
            if not arg.is_Number:
                R1, _ = sring(arg, expand=False, series=True)
                R = R.compose(R1)
        min_pows = list(map(rs_min_pow, args, [R(arg) for arg in args],
            [a]*len(args)))
        sum_pows = sum(min_pows)
        series = R(1)

        for i in range(n):
            _series = _rs_series(args[i], R(args[i]), a, ceiling(prec
                - sum_pows + min_pows[i]))
            R = R.compose(_series.ring)
            _series = _series.set_ring(R)
            series = series.set_ring(R)
            series *= _series
        series = rs_trunc(series, R(a), prec)
        return series

    elif expr.is_Add:
        n = len(args)
        series = R(0)
        for i in range(n):
            _series = _rs_series(args[i], R(args[i]), a, prec)
            R = R.compose(_series.ring)
            _series = _series.set_ring(R)
            series = series.set_ring(R)
            series += _series
        return series

    elif expr.is_Pow:
        R1, _ = sring(expr.base, domain=QQ, expand=False, series=True)
        R = R.compose(R1)
        series_inner = _rs_series(expr.base, R(expr.base), a, prec)
        return rs_pow(series_inner, expr.exp, series_inner.ring(a), prec)

    # The `is_constant` method is buggy hence we check it at the end.
    # See issue #9786 for details.
    elif isinstance(expr, Expr) and expr.is_constant():
        return sring(expr, domain=QQ, expand=False, series=True)[1]

    else:
        raise NotImplementedError

def rs_series(expr, a, prec):
    """Return the series expansion of an expression about 0.

    Parameters
    ==========

    expr : :class:`~.Expr`
    a : :class:`~.Symbol` with respect to which expr is to be expanded
    prec : order of the series expansion

    Currently supports multivariate Taylor series expansion. This is much
    faster that SymPy's series method as it uses sparse polynomial operations.

    It automatically creates the simplest ring required to represent the series
    expansion through repeated calls to sring.

    Examples
    ========

    >>> from sympy.polys.ring_series import rs_series
    >>> from sympy import sin, cos, exp, tan, symbols, QQ
    >>> a, b, c = symbols('a, b, c')
    >>> rs_series(sin(a) + exp(a), a, 5)
    1/24*a**4 + 1/2*a**2 + 2*a + 1
    >>> series = rs_series(tan(a + b)*cos(a + c), a, 2)
    >>> series.as_expr()
    -a*sin(c)*tan(b) + a*cos(c)*tan(b)**2 + a*cos(c) + cos(c)*tan(b)
    >>> series = rs_series(exp(a**QQ(1,3) + a**QQ(2, 5)), a, 1)
    >>> series.as_expr()
    a**(11/15) + a**(4/5)/2 + a**(2/5) + a**(2/3)/2 + a**(1/3) + 1

    """
    R, series = sring(expr, domain=QQ, expand=False, series=True)
    if a not in R.symbols:
        R = R.add_gens([a, ])
    series = series.set_ring(R)
    series = _rs_series(expr, series, a, prec)
    R = series.ring
    gen = R(a)
    prec_got = series.degree(gen) + 1

    if prec_got >= prec:
        return rs_trunc(series, gen, prec)
    else:
        # increase the requested number of terms to get the desired
        # number keep increasing (up to 9) until the received order
        # is different than the original order and then predict how
        # many additional terms are needed
        for more in range(1, 9):
            p1 = _rs_series(expr, series, a, prec=prec + more)
            gen = gen.set_ring(p1.ring)
            new_prec = p1.degree(gen) + 1
            if new_prec != prec_got:
                prec_do = ceiling(prec + (prec - prec_got)*more/(new_prec -
                    prec_got))
                p1 = _rs_series(expr, series, a, prec=prec_do)
                while p1.degree(gen) + 1 < prec:
                    p1 = _rs_series(expr, series, a, prec=prec_do)
                    gen = gen.set_ring(p1.ring)
                    prec_do *= 2
                break
            else:
                break
        else:
            raise ValueError('Could not calculate %s terms for %s'
                             % (str(prec), expr))
        return rs_trunc(p1, gen, prec)
<|endoftext|>"""Groebner bases algorithms. """


from sympy.core.symbol import Dummy
from sympy.polys.monomials import monomial_mul, monomial_lcm, monomial_divides, term_div
from sympy.polys.orderings import lex
from sympy.polys.polyerrors import DomainError
from sympy.polys.polyconfig import query

def groebner(seq, ring, method=None):
    """
    Computes Groebner basis for a set of polynomials in `K[X]`.

    Wrapper around the (default) improved Buchberger and the other algorithms
    for computing Groebner bases. The choice of algorithm can be changed via
    ``method`` argument or :func:`sympy.polys.polyconfig.setup`, where
    ``method`` can be either ``buchberger`` or ``f5b``.

    """
    if method is None:
        method = query('groebner')

    _groebner_methods = {
        'buchberger': _buchberger,
        'f5b': _f5b,
    }

    try:
        _groebner = _groebner_methods[method]
    except KeyError:
        raise ValueError("'%s' is not a valid Groebner bases algorithm (valid are 'buchberger' and 'f5b')" % method)

    domain, orig = ring.domain, None

    if not domain.is_Field or not domain.has_assoc_Field:
        try:
            orig, ring = ring, ring.clone(domain=domain.get_field())
        except DomainError:
            raise DomainError("Cannot compute a Groebner basis over %s" % domain)
        else:
            seq = [ s.set_ring(ring) for s in seq ]

    G = _groebner(seq, ring)

    if orig is not None:
        G = [ g.clear_denoms()[1].set_ring(orig) for g in G ]

    return G

def _buchberger(f, ring):
    """
    Computes Groebner basis for a set of polynomials in `K[X]`.

    Given a set of multivariate polynomials `F`, finds another
    set `G`, such that Ideal `F = Ideal G` and `G` is a reduced
    Groebner basis.

    The resulting basis is unique and has monic generators if the
    ground domains is a field. Otherwise the result is non-unique
    but Groebner bases over e.g. integers can be computed (if the
    input polynomials are monic).

    Groebner bases can be used to choose specific generators for a
    polynomial ideal. Because these bases are unique you can check
    for ideal equality by comparing the Groebner bases.  To see if
    one polynomial lies in an ideal, divide by the elements in the
    base and see if the remainder vanishes.

    They can also be used to solve systems of polynomial equations
    as,  by choosing lexicographic ordering,  you can eliminate one
    variable at a time, provided that the ideal is zero-dimensional
    (finite number of solutions).

    Notes
    =====

    Algorithm used: an improved version of Buchberger's algorithm
    as presented in T. Becker, V. Weispfenning, Groebner Bases: A
    Computational Approach to Commutative Algebra, Springer, 1993,
    page 232.

    References
    ==========

    .. [1] [Bose03]_
    .. [2] [Giovini91]_
    .. [3] [Ajwa95]_
    .. [4] [Cox97]_

    """
    order = ring.order

    monomial_mul = ring.monomial_mul
    monomial_div = ring.monomial_div
    monomial_lcm = ring.monomial_lcm

    def select(P):
        # normal selection strategy
        # select the pair with minimum LCM(LM(f), LM(g))
        pr = min(P, key=lambda pair: order(monomial_lcm(f[pair[0]].LM, f[pair[1]].LM)))
        return pr

    def normal(g, J):
        h = g.rem([ f[j] for j in J ])

        if not h:
            return None
        else:
            h = h.monic()

            if h not in I:
                I[h] = len(f)
                f.append(h)

            return h.LM, I[h]

    def update(G, B, ih):
        # update G using the set of critical pairs B and h
        # [BW] page 230
        h = f[ih]
        mh = h.LM

        # filter new pairs (h, g), g in G
        C = G.copy()
        D = set()

        while C:
            # select a pair (h, g) by popping an element from C
            ig = C.pop()
            g = f[ig]
            mg = g.LM
            LCMhg = monomial_lcm(mh, mg)

            def lcm_divides(ip):
                # LCM(LM(h), LM(p)) divides LCM(LM(h), LM(g))
                m = monomial_lcm(mh, f[ip].LM)
                return monomial_div(LCMhg, m)

            # HT(h) and HT(g) disjoint: mh*mg == LCMhg
            if monomial_mul(mh, mg) == LCMhg or (
                not any(lcm_divides(ipx) for ipx in C) and
                    not any(lcm_divides(pr[1]) for pr in D)):
                D.add((ih, ig))

        E = set()

        while D:
            # select h, g from D (h the same as above)
            ih, ig = D.pop()
            mg = f[ig].LM
            LCMhg = monomial_lcm(mh, mg)

            if not monomial_mul(mh, mg) == LCMhg:
                E.add((ih, ig))

        # filter old pairs
        B_new = set()

        while B:
            # select g1, g2 from B (-> CP)
            ig1, ig2 = B.pop()
            mg1 = f[ig1].LM
            mg2 = f[ig2].LM
            LCM12 = monomial_lcm(mg1, mg2)

            # if HT(h) does not divide lcm(HT(g1), HT(g2))
            if not monomial_div(LCM12, mh) or \
                monomial_lcm(mg1, mh) == LCM12 or \
                    monomial_lcm(mg2, mh) == LCM12:
                B_new.add((ig1, ig2))

        B_new |= E

        # filter polynomials
        G_new = set()

        while G:
            ig = G.pop()
            mg = f[ig].LM

            if not monomial_div(mg, mh):
                G_new.add(ig)

        G_new.add(ih)

        return G_new, B_new
        # end of update ################################

    if not f:
        return []

    # replace f with a reduced list of initial polynomials; see [BW] page 203
    f1 = f[:]

    while True:
        f = f1[:]
        f1 = []

        for i in range(len(f)):
            p = f[i]
            r = p.rem(f[:i])

            if r:
                f1.append(r.monic())

        if f == f1:
            break

    I = {}            # ip = I[p]; p = f[ip]
    F = set()         # set of indices of polynomials
    G = set()         # set of indices of intermediate would-be Groebner basis
    CP = set()        # set of pairs of indices of critical pairs

    for i, h in enumerate(f):
        I[h] = i
        F.add(i)

    #####################################
    # algorithm GROEBNERNEWS2 in [BW] page 232

    while F:
        # select p with minimum monomial according to the monomial ordering
        h = min([f[x] for x in F], key=lambda f: order(f.LM))
        ih = I[h]
        F.remove(ih)
        G, CP = update(G, CP, ih)

    # count the number of critical pairs which reduce to zero
    reductions_to_zero = 0

    while CP:
        ig1, ig2 = select(CP)
        CP.remove((ig1, ig2))

        h = spoly(f[ig1], f[ig2], ring)
        # ordering divisors is on average more efficient [Cox] page 111
        G1 = sorted(G, key=lambda g: order(f[g].LM))
        ht = normal(h, G1)

        if ht:
            G, CP = update(G, CP, ht[1])
        else:
            reductions_to_zero += 1

    ######################################
    # now G is a Groebner basis; reduce it
    Gr = set()

    for ig in G:
        ht = normal(f[ig], G - {ig})

        if ht:
            Gr.add(ht[1])

    Gr = [f[ig] for ig in Gr]

    # order according to the monomial ordering
    Gr = sorted(Gr, key=lambda f: order(f.LM), reverse=True)

    return Gr

def spoly(p1, p2, ring):
    """
    Compute LCM(LM(p1), LM(p2))/LM(p1)*p1 - LCM(LM(p1), LM(p2))/LM(p2)*p2
    This is the S-poly provided p1 and p2 are monic
    """
    LM1 = p1.LM
    LM2 = p2.LM
    LCM12 = ring.monomial_lcm(LM1, LM2)
    m1 = ring.monomial_div(LCM12, LM1)
    m2 = ring.monomial_div(LCM12, LM2)
    s1 = p1.mul_monom(m1)
    s2 = p2.mul_monom(m2)
    s = s1 - s2
    return s

# F5B

# convenience functions


def Sign(f):
    return f[0]


def Polyn(f):
    return f[1]


def Num(f):
    return f[2]


def sig(monomial, index):
    return (monomial, index)


def lbp(signature, polynomial, number):
    return (signature, polynomial, number)

# signature functions


def sig_cmp(u, v, order):
    """
    Compare two signatures by extending the term order to K[X]^n.

    u < v iff
        - the index of v is greater than the index of u
    or
        - the index of v is equal to the index of u and u[0] < v[0] w.r.t. order

    u > v otherwise
    """
    if u[1] > v[1]:
        return -1
    if u[1] == v[1]:
        #if u[0] == v[0]:
        #    return 0
        if order(u[0]) < order(v[0]):
            return -1
    return 1


def sig_key(s, order):
    """
    Key for comparing two signatures.

    s = (m, k), t = (n, l)

    s < t iff [k > l] or [k == l and m < n]
    s > t otherwise
    """
    return (-s[1], order(s[0]))


def sig_mult(s, m):
    """
    Multiply a signature by a monomial.

    The product of a signature (m, i) and a monomial n is defined as
    (m * t, i).
    """
    return sig(monomial_mul(s[0], m), s[1])

# labeled polynomial functions


def lbp_sub(f, g):
    """
    Subtract labeled polynomial g from f.

    The signature and number of the difference of f and g are signature
    and number of the maximum of f and g, w.r.t. lbp_cmp.
    """
    if sig_cmp(Sign(f), Sign(g), Polyn(f).ring.order) < 0:
        max_poly = g
    else:
        max_poly = f

    ret = Polyn(f) - Polyn(g)

    return lbp(Sign(max_poly), ret, Num(max_poly))


def lbp_mul_term(f, cx):
    """
    Multiply a labeled polynomial with a term.

    The product of a labeled polynomial (s, p, k) by a monomial is
    defined as (m * s, m * p, k).
    """
    return lbp(sig_mult(Sign(f), cx[0]), Polyn(f).mul_term(cx), Num(f))


def lbp_cmp(f, g):
    """
    Compare two labeled polynomials.

    f < g iff
        - Sign(f) < Sign(g)
    or
        - Sign(f) == Sign(g) and Num(f) > Num(g)

    f > g otherwise
    """
    if sig_cmp(Sign(f), Sign(g), Polyn(f).ring.order) == -1:
        return -1
    if Sign(f) == Sign(g):
        if Num(f) > Num(g):
            return -1
        #if Num(f) == Num(g):
        #    return 0
    return 1


def lbp_key(f):
    """
    Key for comparing two labeled polynomials.
    """
    return (sig_key(Sign(f), Polyn(f).ring.order), -Num(f))

# algorithm and helper functions


def critical_pair(f, g, ring):
    """
    Compute the critical pair corresponding to two labeled polynomials.

    A critical pair is a tuple (um, f, vm, g), where um and vm are
    terms such that um * f - vm * g is the S-polynomial of f and g (so,
    wlog assume um * f > vm * g).
    For performance sake, a critical pair is represented as a tuple
    (Sign(um * f), um, f, Sign(vm * g), vm, g), since um * f creates
    a new, relatively expensive object in memory, whereas Sign(um *
    f) and um are lightweight and f (in the tuple) is a reference to
    an already existing object in memory.
    """
    domain = ring.domain

    ltf = Polyn(f).LT
    ltg = Polyn(g).LT
    lt = (monomial_lcm(ltf[0], ltg[0]), domain.one)

    um = term_div(lt, ltf, domain)
    vm = term_div(lt, ltg, domain)

    # The full information is not needed (now), so only the product
    # with the leading term is considered:
    fr = lbp_mul_term(lbp(Sign(f), Polyn(f).leading_term(), Num(f)), um)
    gr = lbp_mul_term(lbp(Sign(g), Polyn(g).leading_term(), Num(g)), vm)

    # return in proper order, such that the S-polynomial is just
    # u_first * f_first - u_second * f_second:
    if lbp_cmp(fr, gr) == -1:
        return (Sign(gr), vm, g, Sign(fr), um, f)
    else:
        return (Sign(fr), um, f, Sign(gr), vm, g)


def cp_cmp(c, d):
    """
    Compare two critical pairs c and d.

    c < d iff
        - lbp(c[0], _, Num(c[2]) < lbp(d[0], _, Num(d[2])) (this
        corresponds to um_c * f_c and um_d * f_d)
    or
        - lbp(c[0], _, Num(c[2]) >< lbp(d[0], _, Num(d[2])) and
        lbp(c[3], _, Num(c[5])) < lbp(d[3], _, Num(d[5])) (this
        corresponds to vm_c * g_c and vm_d * g_d)

    c > d otherwise
    """
    zero = Polyn(c[2]).ring.zero

    c0 = lbp(c[0], zero, Num(c[2]))
    d0 = lbp(d[0], zero, Num(d[2]))

    r = lbp_cmp(c0, d0)

    if r == -1:
        return -1
    if r == 0:
        c1 = lbp(c[3], zero, Num(c[5]))
        d1 = lbp(d[3], zero, Num(d[5]))

        r = lbp_cmp(c1, d1)

        if r == -1:
            return -1
        #if r == 0:
        #    return 0
    return 1


def cp_key(c, ring):
    """
    Key for comparing critical pairs.
    """
    return (lbp_key(lbp(c[0], ring.zero, Num(c[2]))), lbp_key(lbp(c[3], ring.zero, Num(c[5]))))


def s_poly(cp):
    """
    Compute the S-polynomial of a critical pair.

    The S-polynomial of a critical pair cp is cp[1] * cp[2] - cp[4] * cp[5].
    """
    return lbp_sub(lbp_mul_term(cp[2], cp[1]), lbp_mul_term(cp[5], cp[4]))


def is_rewritable_or_comparable(sign, num, B):
    """
    Check if a labeled polynomial is redundant by checking if its
    signature and number imply rewritability or comparability.

    (sign, num) is comparable if there exists a labeled polynomial
    h in B, such that sign[1] (the index) is less than Sign(h)[1]
    and sign[0] is divisible by the leading monomial of h.

    (sign, num) is rewritable if there exists a labeled polynomial
    h in B, such thatsign[1] is equal to Sign(h)[1], num < Num(h)
    and sign[0] is divisible by Sign(h)[0].
    """
    for h in B:
        # comparable
        if sign[1] < Sign(h)[1]:
            if monomial_divides(Polyn(h).LM, sign[0]):
                return True

        # rewritable
        if sign[1] == Sign(h)[1]:
            if num < Num(h):
                if monomial_divides(Sign(h)[0], sign[0]):
                    return True
    return False


def f5_reduce(f, B):
    """
    F5-reduce a labeled polynomial f by B.

    Continuously searches for non-zero labeled polynomial h in B, such
    that the leading term lt_h of h divides the leading term lt_f of
    f and Sign(lt_h * h) < Sign(f). If such a labeled polynomial h is
    found, f gets replaced by f - lt_f / lt_h * h. If no such h can be
    found or f is 0, f is no further F5-reducible and f gets returned.

    A polynomial that is reducible in the usual sense need not be
    F5-reducible, e.g.:

    >>> from sympy.polys.groebnertools import lbp, sig, f5_reduce, Polyn
    >>> from sympy.polys import ring, QQ, lex

    >>> R, x,y,z = ring("x,y,z", QQ, lex)

    >>> f = lbp(sig((1, 1, 1), 4), x, 3)
    >>> g = lbp(sig((0, 0, 0), 2), x, 2)

    >>> Polyn(f).rem([Polyn(g)])
    0
    >>> f5_reduce(f, [g])
    (((1, 1, 1), 4), x, 3)

    """
    order = Polyn(f).ring.order
    domain = Polyn(f).ring.domain

    if not Polyn(f):
        return f

    while True:
        g = f

        for h in B:
            if Polyn(h):
                if monomial_divides(Polyn(h).LM, Polyn(f).LM):
                    t = term_div(Polyn(f).LT, Polyn(h).LT, domain)
                    if sig_cmp(sig_mult(Sign(h), t[0]), Sign(f), order) < 0:
                        # The following check need not be done and is in general slower than without.
                        #if not is_rewritable_or_comparable(Sign(gp), Num(gp), B):
                        hp = lbp_mul_term(h, t)
                        f = lbp_sub(f, hp)
                        break

        if g == f or not Polyn(f):
            return f


def _f5b(F, ring):
    """
    Computes a reduced Groebner basis for the ideal generated by F.

    f5b is an implementation of the F5B algorithm by Yao Sun and
    Dingkang Wang. Similarly to Buchberger's algorithm, the algorithm
    proceeds by computing critical pairs, computing the S-polynomial,
    reducing it and adjoining the reduced S-polynomial if it is not 0.

    Unlike Buchberger's algorithm, each polynomial contains additional
    information, namely a signature and a number. The signature
    specifies the path of computation (i.e. from which polynomial in
    the original basis was it derived and how), the number says when
    the polynomial was added to the basis.  With this information it
    is (often) possible to decide if an S-polynomial will reduce to
    0 and can be discarded.

    Optimizations include: Reducing the generators before computing
    a Groebner basis, removing redundant critical pairs when a new
    polynomial enters the basis and sorting the critical pairs and
    the current basis.

    Once a Groebner basis has been found, it gets reduced.

    References
    ==========

    .. [1] Yao Sun, Dingkang Wang: "A New Proof for the Correctness of F5
           (F5-Like) Algorithm", https://arxiv.org/abs/1004.0084 (specifically
           v4)

    .. [2] Thomas Becker, Volker Weispfenning, Groebner bases: A computational
           approach to commutative algebra, 1993, p. 203, 216
    """
    order = ring.order

    # reduce polynomials (like in Mario Pernici's implementation) (Becker, Weispfenning, p. 203)
    B = F
    while True:
        F = B
        B = []

        for i in range(len(F)):
            p = F[i]
            r = p.rem(F[:i])

            if r:
                B.append(r)

        if F == B:
            break

    # basis
    B = [lbp(sig(ring.zero_monom, i + 1), F[i], i + 1) for i in range(len(F))]
    B.sort(key=lambda f: order(Polyn(f).LM), reverse=True)

    # critical pairs
    CP = [critical_pair(B[i], B[j], ring) for i in range(len(B)) for j in range(i + 1, len(B))]
    CP.sort(key=lambda cp: cp_key(cp, ring), reverse=True)

    k = len(B)

    reductions_to_zero = 0

    while len(CP):
        cp = CP.pop()

        # discard redundant critical pairs:
        if is_rewritable_or_comparable(cp[0], Num(cp[2]), B):
            continue
        if is_rewritable_or_comparable(cp[3], Num(cp[5]), B):
            continue

        s = s_poly(cp)

        p = f5_reduce(s, B)

        p = lbp(Sign(p), Polyn(p).monic(), k + 1)

        if Polyn(p):
            # remove old critical pairs, that become redundant when adding p:
            indices = []
            for i, cp in enumerate(CP):
                if is_rewritable_or_comparable(cp[0], Num(cp[2]), [p]):
                    indices.append(i)
                elif is_rewritable_or_comparable(cp[3], Num(cp[5]), [p]):
                    indices.append(i)

            for i in reversed(indices):
                del CP[i]

            # only add new critical pairs that are not made redundant by p:
            for g in B:
                if Polyn(g):
                    cp = critical_pair(p, g, ring)
                    if is_rewritable_or_comparable(cp[0], Num(cp[2]), [p]):
                        continue
                    elif is_rewritable_or_comparable(cp[3], Num(cp[5]), [p]):
                        continue

                    CP.append(cp)

            # sort (other sorting methods/selection strategies were not as successful)
            CP.sort(key=lambda cp: cp_key(cp, ring), reverse=True)

            # insert p into B:
            m = Polyn(p).LM
            if order(m) <= order(Polyn(B[-1]).LM):
                B.append(p)
            else:
                for i, q in enumerate(B):
                    if order(m) > order(Polyn(q).LM):
                        B.insert(i, p)
                        break

            k += 1

            #print(len(B), len(CP), "%d critical pairs removed" % len(indices))
        else:
            reductions_to_zero += 1

    # reduce Groebner basis:
    H = [Polyn(g).monic() for g in B]
    H = red_groebner(H, ring)

    return sorted(H, key=lambda f: order(f.LM), reverse=True)


def red_groebner(G, ring):
    """
    Compute reduced Groebner basis, from BeckerWeispfenning93, p. 216

    Selects a subset of generators, that already generate the ideal
    and computes a reduced Groebner basis for them.
    """
    def reduction(P):
        """
        The actual reduction algorithm.
        """
        Q = []
        for i, p in enumerate(P):
            h = p.rem(P[:i] + P[i + 1:])
            if h:
                Q.append(h)

        return [p.monic() for p in Q]

    F = G
    H = []

    while F:
        f0 = F.pop()

        if not any(monomial_divides(f.LM, f0.LM) for f in F + H):
            H.append(f0)

    # Becker, Weispfenning, p. 217: H is Groebner basis of the ideal generated by G.
    return reduction(H)


def is_groebner(G, ring):
    """
    Check if G is a Groebner basis.
    """
    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            s = spoly(G[i], G[j], ring)
            s = s.rem(G)
            if s:
                return False

    return True


def is_minimal(G, ring):
    """
    Checks if G is a minimal Groebner basis.
    """
    order = ring.order
    domain = ring.domain

    G.sort(key=lambda g: order(g.LM))

    for i, g in enumerate(G):
        if g.LC != domain.one:
            return False

        for h in G[:i] + G[i + 1:]:
            if monomial_divides(h.LM, g.LM):
                return False

    return True


def is_reduced(G, ring):
    """
    Checks if G is a reduced Groebner basis.
    """
    order = ring.order
    domain = ring.domain

    G.sort(key=lambda g: order(g.LM))

    for i, g in enumerate(G):
        if g.LC != domain.one:
            return False

        for term in g.terms():
            for h in G[:i] + G[i + 1:]:
                if monomial_divides(h.LM, term[0]):
                    return False

    return True

def groebner_lcm(f, g):
    """
    Computes LCM of two polynomials using Groebner bases.

    The LCM is computed as the unique generator of the intersection
    of the two ideals generated by `f` and `g`. The approach is to
    compute a Groebner basis with respect to lexicographic ordering
    of `t*f` and `(1 - t)*g`, where `t` is an unrelated variable and
    then filtering out the solution that does not contain `t`.

    References
    ==========

    .. [1] [Cox97]_

    """
    if f.ring != g.ring:
        raise ValueError("Values should be equal")

    ring = f.ring
    domain = ring.domain

    if not f or not g:
        return ring.zero

    if len(f) <= 1 and len(g) <= 1:
        monom = monomial_lcm(f.LM, g.LM)
        coeff = domain.lcm(f.LC, g.LC)
        return ring.term_new(monom, coeff)

    fc, f = f.primitive()
    gc, g = g.primitive()

    lcm = domain.lcm(fc, gc)

    f_terms = [ ((1,) + monom, coeff) for monom, coeff in f.terms() ]
    g_terms = [ ((0,) + monom, coeff) for monom, coeff in g.terms() ] \
            + [ ((1,) + monom,-coeff) for monom, coeff in g.terms() ]

    t = Dummy("t")
    t_ring = ring.clone(symbols=(t,) + ring.symbols, order=lex)

    F = t_ring.from_terms(f_terms)
    G = t_ring.from_terms(g_terms)

    basis = groebner([F, G], t_ring)

    def is_independent(h, j):
        return not any(monom[j] for monom in h.monoms())

    H = [ h for h in basis if is_independent(h, 0) ]

    h_terms = [ (monom[1:], coeff*lcm) for monom, coeff in H[0].terms() ]
    h = ring.from_terms(h_terms)

    return h

def groebner_gcd(f, g):
    """Computes GCD of two polynomials using Groebner bases. """
    if f.ring != g.ring:
        raise ValueError("Values should be equal")
    domain = f.ring.domain

    if not domain.is_Field:
        fc, f = f.primitive()
        gc, g = g.primitive()
        gcd = domain.gcd(fc, gc)

    H = (f*g).quo([groebner_lcm(f, g)])

    if len(H) != 1:
        raise ValueError("Length should be 1")
    h = H[0]

    if not domain.is_Field:
        return gcd*h
    else:
        return h.monic()
<|endoftext|>"""Arithmetics for dense recursive polynomials in ``K[x]`` or ``K[X]``. """


from sympy.polys.densebasic import (
    dup_slice,
    dup_LC, dmp_LC,
    dup_degree, dmp_degree,
    dup_strip, dmp_strip,
    dmp_zero_p, dmp_zero,
    dmp_one_p, dmp_one,
    dmp_ground, dmp_zeros)
from sympy.polys.polyerrors import (ExactQuotientFailed, PolynomialDivisionFailed)

def dup_add_term(f, c, i, K):
    """
    Add ``c*x**i`` to ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_add_term(x**2 - 1, ZZ(2), 4)
    2*x**4 + x**2 - 1

    """
    if not c:
        return f

    n = len(f)
    m = n - i - 1

    if i == n - 1:
        return dup_strip([f[0] + c] + f[1:])
    else:
        if i >= n:
            return [c] + [K.zero]*(i - n) + f
        else:
            return f[:m] + [f[m] + c] + f[m + 1:]


def dmp_add_term(f, c, i, u, K):
    """
    Add ``c(x_2..x_u)*x_0**i`` to ``f`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_add_term(x*y + 1, 2, 2)
    2*x**2 + x*y + 1

    """
    if not u:
        return dup_add_term(f, c, i, K)

    v = u - 1

    if dmp_zero_p(c, v):
        return f

    n = len(f)
    m = n - i - 1

    if i == n - 1:
        return dmp_strip([dmp_add(f[0], c, v, K)] + f[1:], u)
    else:
        if i >= n:
            return [c] + dmp_zeros(i - n, v, K) + f
        else:
            return f[:m] + [dmp_add(f[m], c, v, K)] + f[m + 1:]


def dup_sub_term(f, c, i, K):
    """
    Subtract ``c*x**i`` from ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_sub_term(2*x**4 + x**2 - 1, ZZ(2), 4)
    x**2 - 1

    """
    if not c:
        return f

    n = len(f)
    m = n - i - 1

    if i == n - 1:
        return dup_strip([f[0] - c] + f[1:])
    else:
        if i >= n:
            return [-c] + [K.zero]*(i - n) + f
        else:
            return f[:m] + [f[m] - c] + f[m + 1:]


def dmp_sub_term(f, c, i, u, K):
    """
    Subtract ``c(x_2..x_u)*x_0**i`` from ``f`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_sub_term(2*x**2 + x*y + 1, 2, 2)
    x*y + 1

    """
    if not u:
        return dup_add_term(f, -c, i, K)

    v = u - 1

    if dmp_zero_p(c, v):
        return f

    n = len(f)
    m = n - i - 1

    if i == n - 1:
        return dmp_strip([dmp_sub(f[0], c, v, K)] + f[1:], u)
    else:
        if i >= n:
            return [dmp_neg(c, v, K)] + dmp_zeros(i - n, v, K) + f
        else:
            return f[:m] + [dmp_sub(f[m], c, v, K)] + f[m + 1:]


def dup_mul_term(f, c, i, K):
    """
    Multiply ``f`` by ``c*x**i`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_mul_term(x**2 - 1, ZZ(3), 2)
    3*x**4 - 3*x**2

    """
    if not c or not f:
        return []
    else:
        return [ cf * c for cf in f ] + [K.zero]*i


def dmp_mul_term(f, c, i, u, K):
    """
    Multiply ``f`` by ``c(x_2..x_u)*x_0**i`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_mul_term(x**2*y + x, 3*y, 2)
    3*x**4*y**2 + 3*x**3*y

    """
    if not u:
        return dup_mul_term(f, c, i, K)

    v = u - 1

    if dmp_zero_p(f, u):
        return f
    if dmp_zero_p(c, v):
        return dmp_zero(u)
    else:
        return [ dmp_mul(cf, c, v, K) for cf in f ] + dmp_zeros(i, v, K)


def dup_add_ground(f, c, K):
    """
    Add an element of the ground domain to ``f``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_add_ground(x**3 + 2*x**2 + 3*x + 4, ZZ(4))
    x**3 + 2*x**2 + 3*x + 8

    """
    return dup_add_term(f, c, 0, K)


def dmp_add_ground(f, c, u, K):
    """
    Add an element of the ground domain to ``f``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_add_ground(x**3 + 2*x**2 + 3*x + 4, ZZ(4))
    x**3 + 2*x**2 + 3*x + 8

    """
    return dmp_add_term(f, dmp_ground(c, u - 1), 0, u, K)


def dup_sub_ground(f, c, K):
    """
    Subtract an element of the ground domain from ``f``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_sub_ground(x**3 + 2*x**2 + 3*x + 4, ZZ(4))
    x**3 + 2*x**2 + 3*x

    """
    return dup_sub_term(f, c, 0, K)


def dmp_sub_ground(f, c, u, K):
    """
    Subtract an element of the ground domain from ``f``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_sub_ground(x**3 + 2*x**2 + 3*x + 4, ZZ(4))
    x**3 + 2*x**2 + 3*x

    """
    return dmp_sub_term(f, dmp_ground(c, u - 1), 0, u, K)


def dup_mul_ground(f, c, K):
    """
    Multiply ``f`` by a constant value in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_mul_ground(x**2 + 2*x - 1, ZZ(3))
    3*x**2 + 6*x - 3

    """
    if not c or not f:
        return []
    else:
        return [ cf * c for cf in f ]


def dmp_mul_ground(f, c, u, K):
    """
    Multiply ``f`` by a constant value in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_mul_ground(2*x + 2*y, ZZ(3))
    6*x + 6*y

    """
    if not u:
        return dup_mul_ground(f, c, K)

    v = u - 1

    return [ dmp_mul_ground(cf, c, v, K) for cf in f ]


def dup_quo_ground(f, c, K):
    """
    Quotient by a constant in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x = ring("x", ZZ)
    >>> R.dup_quo_ground(3*x**2 + 2, ZZ(2))
    x**2 + 1

    >>> R, x = ring("x", QQ)
    >>> R.dup_quo_ground(3*x**2 + 2, QQ(2))
    3/2*x**2 + 1

    """
    if not c:
        raise ZeroDivisionError('polynomial division')
    if not f:
        return f

    if K.is_Field:
        return [ K.quo(cf, c) for cf in f ]
    else:
        return [ cf // c for cf in f ]


def dmp_quo_ground(f, c, u, K):
    """
    Quotient by a constant in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ, QQ

    >>> R, x,y = ring("x,y", ZZ)
    >>> R.dmp_quo_ground(2*x**2*y + 3*x, ZZ(2))
    x**2*y + x

    >>> R, x,y = ring("x,y", QQ)
    >>> R.dmp_quo_ground(2*x**2*y + 3*x, QQ(2))
    x**2*y + 3/2*x

    """
    if not u:
        return dup_quo_ground(f, c, K)

    v = u - 1

    return [ dmp_quo_ground(cf, c, v, K) for cf in f ]


def dup_exquo_ground(f, c, K):
    """
    Exact quotient by a constant in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    >>> R.dup_exquo_ground(x**2 + 2, QQ(2))
    1/2*x**2 + 1

    """
    if not c:
        raise ZeroDivisionError('polynomial division')
    if not f:
        return f

    return [ K.exquo(cf, c) for cf in f ]


def dmp_exquo_ground(f, c, u, K):
    """
    Exact quotient by a constant in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x,y = ring("x,y", QQ)

    >>> R.dmp_exquo_ground(x**2*y + 2*x, QQ(2))
    1/2*x**2*y + x

    """
    if not u:
        return dup_exquo_ground(f, c, K)

    v = u - 1

    return [ dmp_exquo_ground(cf, c, v, K) for cf in f ]


def dup_lshift(f, n, K):
    """
    Efficiently multiply ``f`` by ``x**n`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_lshift(x**2 + 1, 2)
    x**4 + x**2

    """
    if not f:
        return f
    else:
        return f + [K.zero]*n


def dup_rshift(f, n, K):
    """
    Efficiently divide ``f`` by ``x**n`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_rshift(x**4 + x**2, 2)
    x**2 + 1
    >>> R.dup_rshift(x**4 + x**2 + 2, 2)
    x**2 + 1

    """
    return f[:-n]


def dup_abs(f, K):
    """
    Make all coefficients positive in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_abs(x**2 - 1)
    x**2 + 1

    """
    return [ K.abs(coeff) for coeff in f ]


def dmp_abs(f, u, K):
    """
    Make all coefficients positive in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_abs(x**2*y - x)
    x**2*y + x

    """
    if not u:
        return dup_abs(f, K)

    v = u - 1

    return [ dmp_abs(cf, v, K) for cf in f ]


def dup_neg(f, K):
    """
    Negate a polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_neg(x**2 - 1)
    -x**2 + 1

    """
    return [ -coeff for coeff in f ]


def dmp_neg(f, u, K):
    """
    Negate a polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_neg(x**2*y - x)
    -x**2*y + x

    """
    if not u:
        return dup_neg(f, K)

    v = u - 1

    return [ dmp_neg(cf, v, K) for cf in f ]


def dup_add(f, g, K):
    """
    Add dense polynomials in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_add(x**2 - 1, x - 2)
    x**2 + x - 3

    """
    if not f:
        return g
    if not g:
        return f

    df = dup_degree(f)
    dg = dup_degree(g)

    if df == dg:
        return dup_strip([ a + b for a, b in zip(f, g) ])
    else:
        k = abs(df - dg)

        if df > dg:
            h, f = f[:k], f[k:]
        else:
            h, g = g[:k], g[k:]

        return h + [ a + b for a, b in zip(f, g) ]


def dmp_add(f, g, u, K):
    """
    Add dense polynomials in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_add(x**2 + y, x**2*y + x)
    x**2*y + x**2 + x + y

    """
    if not u:
        return dup_add(f, g, K)

    df = dmp_degree(f, u)

    if df < 0:
        return g

    dg = dmp_degree(g, u)

    if dg < 0:
        return f

    v = u - 1

    if df == dg:
        return dmp_strip([ dmp_add(a, b, v, K) for a, b in zip(f, g) ], u)
    else:
        k = abs(df - dg)

        if df > dg:
            h, f = f[:k], f[k:]
        else:
            h, g = g[:k], g[k:]

        return h + [ dmp_add(a, b, v, K) for a, b in zip(f, g) ]


def dup_sub(f, g, K):
    """
    Subtract dense polynomials in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_sub(x**2 - 1, x - 2)
    x**2 - x + 1

    """
    if not f:
        return dup_neg(g, K)
    if not g:
        return f

    df = dup_degree(f)
    dg = dup_degree(g)

    if df == dg:
        return dup_strip([ a - b for a, b in zip(f, g) ])
    else:
        k = abs(df - dg)

        if df > dg:
            h, f = f[:k], f[k:]
        else:
            h, g = dup_neg(g[:k], K), g[k:]

        return h + [ a - b for a, b in zip(f, g) ]


def dmp_sub(f, g, u, K):
    """
    Subtract dense polynomials in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_sub(x**2 + y, x**2*y + x)
    -x**2*y + x**2 - x + y

    """
    if not u:
        return dup_sub(f, g, K)

    df = dmp_degree(f, u)

    if df < 0:
        return dmp_neg(g, u, K)

    dg = dmp_degree(g, u)

    if dg < 0:
        return f

    v = u - 1

    if df == dg:
        return dmp_strip([ dmp_sub(a, b, v, K) for a, b in zip(f, g) ], u)
    else:
        k = abs(df - dg)

        if df > dg:
            h, f = f[:k], f[k:]
        else:
            h, g = dmp_neg(g[:k], u, K), g[k:]

        return h + [ dmp_sub(a, b, v, K) for a, b in zip(f, g) ]


def dup_add_mul(f, g, h, K):
    """
    Returns ``f + g*h`` where ``f, g, h`` are in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_add_mul(x**2 - 1, x - 2, x + 2)
    2*x**2 - 5

    """
    return dup_add(f, dup_mul(g, h, K), K)


def dmp_add_mul(f, g, h, u, K):
    """
    Returns ``f + g*h`` where ``f, g, h`` are in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_add_mul(x**2 + y, x, x + 2)
    2*x**2 + 2*x + y

    """
    return dmp_add(f, dmp_mul(g, h, u, K), u, K)


def dup_sub_mul(f, g, h, K):
    """
    Returns ``f - g*h`` where ``f, g, h`` are in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_sub_mul(x**2 - 1, x - 2, x + 2)
    3

    """
    return dup_sub(f, dup_mul(g, h, K), K)


def dmp_sub_mul(f, g, h, u, K):
    """
    Returns ``f - g*h`` where ``f, g, h`` are in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_sub_mul(x**2 + y, x, x + 2)
    -2*x + y

    """
    return dmp_sub(f, dmp_mul(g, h, u, K), u, K)


def dup_mul(f, g, K):
    """
    Multiply dense polynomials in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_mul(x - 2, x + 2)
    x**2 - 4

    """
    if f == g:
        return dup_sqr(f, K)

    if not (f and g):
        return []

    df = dup_degree(f)
    dg = dup_degree(g)

    n = max(df, dg) + 1

    if n < 100 or not K.is_Exact:
        h = []

        for i in range(0, df + dg + 1):
            coeff = K.zero

            for j in range(max(0, i - dg), min(df, i) + 1):
                coeff += f[j]*g[i - j]

            h.append(coeff)

        return dup_strip(h)
    else:
        # Use Karatsuba's algorithm (divide and conquer), see e.g.:
        # Joris van der Hoeven, Relax But Don't Be Too Lazy,
        # J. Symbolic Computation, 11 (2002), section 3.1.1.
        n2 = n//2

        fl, gl = dup_slice(f, 0, n2, K), dup_slice(g, 0, n2, K)

        fh = dup_rshift(dup_slice(f, n2, n, K), n2, K)
        gh = dup_rshift(dup_slice(g, n2, n, K), n2, K)

        lo, hi = dup_mul(fl, gl, K), dup_mul(fh, gh, K)

        mid = dup_mul(dup_add(fl, fh, K), dup_add(gl, gh, K), K)
        mid = dup_sub(mid, dup_add(lo, hi, K), K)

        return dup_add(dup_add(lo, dup_lshift(mid, n2, K), K),
                       dup_lshift(hi, 2*n2, K), K)


def dmp_mul(f, g, u, K):
    """
    Multiply dense polynomials in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_mul(x*y + 1, x)
    x**2*y + x

    """
    if not u:
        return dup_mul(f, g, K)

    if f == g:
        return dmp_sqr(f, u, K)

    df = dmp_degree(f, u)

    if df < 0:
        return f

    dg = dmp_degree(g, u)

    if dg < 0:
        return g

    h, v = [], u - 1

    for i in range(0, df + dg + 1):
        coeff = dmp_zero(v)

        for j in range(max(0, i - dg), min(df, i) + 1):
            coeff = dmp_add(coeff, dmp_mul(f[j], g[i - j], v, K), v, K)

        h.append(coeff)

    return dmp_strip(h, u)


def dup_sqr(f, K):
    """
    Square dense polynomials in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_sqr(x**2 + 1)
    x**4 + 2*x**2 + 1

    """
    df, h = len(f) - 1, []

    for i in range(0, 2*df + 1):
        c = K.zero

        jmin = max(0, i - df)
        jmax = min(i, df)

        n = jmax - jmin + 1

        jmax = jmin + n // 2 - 1

        for j in range(jmin, jmax + 1):
            c += f[j]*f[i - j]

        c += c

        if n & 1:
            elem = f[jmax + 1]
            c += elem**2

        h.append(c)

    return dup_strip(h)


def dmp_sqr(f, u, K):
    """
    Square dense polynomials in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_sqr(x**2 + x*y + y**2)
    x**4 + 2*x**3*y + 3*x**2*y**2 + 2*x*y**3 + y**4

    """
    if not u:
        return dup_sqr(f, K)

    df = dmp_degree(f, u)

    if df < 0:
        return f

    h, v = [], u - 1

    for i in range(0, 2*df + 1):
        c = dmp_zero(v)

        jmin = max(0, i - df)
        jmax = min(i, df)

        n = jmax - jmin + 1

        jmax = jmin + n // 2 - 1

        for j in range(jmin, jmax + 1):
            c = dmp_add(c, dmp_mul(f[j], f[i - j], v, K), v, K)

        c = dmp_mul_ground(c, K(2), v, K)

        if n & 1:
            elem = dmp_sqr(f[jmax + 1], v, K)
            c = dmp_add(c, elem, v, K)

        h.append(c)

    return dmp_strip(h, u)


def dup_pow(f, n, K):
    """
    Raise ``f`` to the ``n``-th power in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_pow(x - 2, 3)
    x**3 - 6*x**2 + 12*x - 8

    """
    if not n:
        return [K.one]
    if n < 0:
        raise ValueError("Cannot raise polynomial to a negative power")
    if n == 1 or not f or f == [K.one]:
        return f

    g = [K.one]

    while True:
        n, m = n//2, n

        if m % 2:
            g = dup_mul(g, f, K)

            if not n:
                break

        f = dup_sqr(f, K)

    return g


def dmp_pow(f, n, u, K):
    """
    Raise ``f`` to the ``n``-th power in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_pow(x*y + 1, 3)
    x**3*y**3 + 3*x**2*y**2 + 3*x*y + 1

    """
    if not u:
        return dup_pow(f, n, K)

    if not n:
        return dmp_one(u, K)
    if n < 0:
        raise ValueError("Cannot raise polynomial to a negative power")
    if n == 1 or dmp_zero_p(f, u) or dmp_one_p(f, u, K):
        return f

    g = dmp_one(u, K)

    while True:
        n, m = n//2, n

        if m & 1:
            g = dmp_mul(g, f, u, K)

            if not n:
                break

        f = dmp_sqr(f, u, K)

    return g


def dup_pdiv(f, g, K):
    """
    Polynomial pseudo-division in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_pdiv(x**2 + 1, 2*x - 4)
    (2*x + 4, 20)

    """
    df = dup_degree(f)
    dg = dup_degree(g)

    q, r, dr = [], f, df

    if not g:
        raise ZeroDivisionError("polynomial division")
    elif df < dg:
        return q, r

    N = df - dg + 1
    lc_g = dup_LC(g, K)

    while True:
        lc_r = dup_LC(r, K)
        j, N = dr - dg, N - 1

        Q = dup_mul_ground(q, lc_g, K)
        q = dup_add_term(Q, lc_r, j, K)

        R = dup_mul_ground(r, lc_g, K)
        G = dup_mul_term(g, lc_r, j, K)
        r = dup_sub(R, G, K)

        _dr, dr = dr, dup_degree(r)

        if dr < dg:
            break
        elif not (dr < _dr):
            raise PolynomialDivisionFailed(f, g, K)

    c = lc_g**N

    q = dup_mul_ground(q, c, K)
    r = dup_mul_ground(r, c, K)

    return q, r


def dup_prem(f, g, K):
    """
    Polynomial pseudo-remainder in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_prem(x**2 + 1, 2*x - 4)
    20

    """
    df = dup_degree(f)
    dg = dup_degree(g)

    r, dr = f, df

    if not g:
        raise ZeroDivisionError("polynomial division")
    elif df < dg:
        return r

    N = df - dg + 1
    lc_g = dup_LC(g, K)

    while True:
        lc_r = dup_LC(r, K)
        j, N = dr - dg, N - 1

        R = dup_mul_ground(r, lc_g, K)
        G = dup_mul_term(g, lc_r, j, K)
        r = dup_sub(R, G, K)

        _dr, dr = dr, dup_degree(r)

        if dr < dg:
            break
        elif not (dr < _dr):
            raise PolynomialDivisionFailed(f, g, K)

    return dup_mul_ground(r, lc_g**N, K)


def dup_pquo(f, g, K):
    """
    Polynomial exact pseudo-quotient in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_pquo(x**2 - 1, 2*x - 2)
    2*x + 2

    >>> R.dup_pquo(x**2 + 1, 2*x - 4)
    2*x + 4

    """
    return dup_pdiv(f, g, K)[0]


def dup_pexquo(f, g, K):
    """
    Polynomial pseudo-quotient in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_pexquo(x**2 - 1, 2*x - 2)
    2*x + 2

    >>> R.dup_pexquo(x**2 + 1, 2*x - 4)
    Traceback (most recent call last):
    ...
    ExactQuotientFailed: [2, -4] does not divide [1, 0, 1]

    """
    q, r = dup_pdiv(f, g, K)

    if not r:
        return q
    else:
        raise ExactQuotientFailed(f, g)


def dmp_pdiv(f, g, u, K):
    """
    Polynomial pseudo-division in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_pdiv(x**2 + x*y, 2*x + 2)
    (2*x + 2*y - 2, -4*y + 4)

    """
    if not u:
        return dup_pdiv(f, g, K)

    df = dmp_degree(f, u)
    dg = dmp_degree(g, u)

    if dg < 0:
        raise ZeroDivisionError("polynomial division")

    q, r, dr = dmp_zero(u), f, df

    if df < dg:
        return q, r

    N = df - dg + 1
    lc_g = dmp_LC(g, K)

    while True:
        lc_r = dmp_LC(r, K)
        j, N = dr - dg, N - 1

        Q = dmp_mul_term(q, lc_g, 0, u, K)
        q = dmp_add_term(Q, lc_r, j, u, K)

        R = dmp_mul_term(r, lc_g, 0, u, K)
        G = dmp_mul_term(g, lc_r, j, u, K)
        r = dmp_sub(R, G, u, K)

        _dr, dr = dr, dmp_degree(r, u)

        if dr < dg:
            break
        elif not (dr < _dr):
            raise PolynomialDivisionFailed(f, g, K)

    c = dmp_pow(lc_g, N, u - 1, K)

    q = dmp_mul_term(q, c, 0, u, K)
    r = dmp_mul_term(r, c, 0, u, K)

    return q, r


def dmp_prem(f, g, u, K):
    """
    Polynomial pseudo-remainder in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_prem(x**2 + x*y, 2*x + 2)
    -4*y + 4

    """
    if not u:
        return dup_prem(f, g, K)

    df = dmp_degree(f, u)
    dg = dmp_degree(g, u)

    if dg < 0:
        raise ZeroDivisionError("polynomial division")

    r, dr = f, df

    if df < dg:
        return r

    N = df - dg + 1
    lc_g = dmp_LC(g, K)

    while True:
        lc_r = dmp_LC(r, K)
        j, N = dr - dg, N - 1

        R = dmp_mul_term(r, lc_g, 0, u, K)
        G = dmp_mul_term(g, lc_r, j, u, K)
        r = dmp_sub(R, G, u, K)

        _dr, dr = dr, dmp_degree(r, u)

        if dr < dg:
            break
        elif not (dr < _dr):
            raise PolynomialDivisionFailed(f, g, K)

    c = dmp_pow(lc_g, N, u - 1, K)

    return dmp_mul_term(r, c, 0, u, K)


def dmp_pquo(f, g, u, K):
    """
    Polynomial exact pseudo-quotient in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = x**2 + x*y
    >>> g = 2*x + 2*y
    >>> h = 2*x + 2

    >>> R.dmp_pquo(f, g)
    2*x

    >>> R.dmp_pquo(f, h)
    2*x + 2*y - 2

    """
    return dmp_pdiv(f, g, u, K)[0]


def dmp_pexquo(f, g, u, K):
    """
    Polynomial pseudo-quotient in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = x**2 + x*y
    >>> g = 2*x + 2*y
    >>> h = 2*x + 2

    >>> R.dmp_pexquo(f, g)
    2*x

    >>> R.dmp_pexquo(f, h)
    Traceback (most recent call last):
    ...
    ExactQuotientFailed: [[2], [2]] does not divide [[1], [1, 0], []]

    """
