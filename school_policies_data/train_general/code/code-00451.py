        ha = _func_field_modgcd_p(fa, ga, minpolya, p)

        if ha is None:
            d += 1
            if d > n:
                return None
            continue

        if ha == 1:
            return ha

        LM = [ha.degree()] + [0]*(k-1)
        if k > 1:
            for monom, coeff in ha.iterterms():
                if monom[0] == LM[0] and coeff.LM > tuple(LM[1:]):
                    LM[1:] = coeff.LM

        evalpoints_a = [a]
        heval_a = [ha]
        if k == 1:
            m = qring.domain.get_ring().one
        else:
            m = qring.domain.domain.get_ring().one

        t = m.ring.gens[0]

        for b, hb, LMhb in zip(evalpoints, heval, LMlist):
            if LMhb == LM:
                evalpoints_a.append(b)
                heval_a.append(hb)
                m *= (t - b)

        m = m.trunc_ground(p)
        evalpoints.append(a)
        heval.append(ha)
        LMlist.append(LM)
        n += 1

        # polynomial in Z_p[t_1, ..., t_k][x, z]
        h = _interpolate_multivariate(evalpoints_a, heval_a, ring, k-1, p, ground=True)

        # polynomial in Z_p(t_k)[t_1, ..., t_{k-1}][x, z]
        h = _rational_reconstruction_func_coeffs(h, p, m, qring, k-1)

        if h is None:
            continue

        if k == 1:
            dom = qring.domain.field
            den = dom.ring.one

            for coeff in h.itercoeffs():
                den = dom.ring.from_dense(gf_lcm(den.to_dense(), coeff.denom.to_dense(),
                        p, dom.domain))

        else:
            dom = qring.domain.domain.field
            den = dom.ring.one

            for coeff in h.itercoeffs():
                for c in coeff.itercoeffs():
                    den = dom.ring.from_dense(gf_lcm(den.to_dense(), c.denom.to_dense(),
                            p, dom.domain))

        den = qring.domain_new(den.trunc_ground(p))
        h = ring(h.mul_ground(den).as_expr()).trunc_ground(p)

        if not _trial_division(f, h, minpoly, p) and not _trial_division(g, h, minpoly, p):
            return h

    return None


def _integer_rational_reconstruction(c, m, domain):
    r"""
    Reconstruct a rational number `\frac a b` from

    .. math::

        c = \frac a b \; \mathrm{mod} \, m,

    where `c` and `m` are integers.

    The algorithm is based on the Euclidean Algorithm. In general, `m` is
    not a prime number, so it is possible that `b` is not invertible modulo
    `m`. In that case ``None`` is returned.

    Parameters
    ==========

    c : Integer
        `c = \frac a b \; \mathrm{mod} \, m`
    m : Integer
        modulus, not necessarily prime
    domain : IntegerRing
        `a, b, c` are elements of ``domain``

    Returns
    =======

    frac : Rational
        either `\frac a b` in `\mathbb Q` or ``None``

    References
    ==========

    1. [Wang81]_

    """
    if c < 0:
        c += m

    r0, s0 = m, domain.zero
    r1, s1 = c, domain.one

    bound = sqrt(m / 2) # still correct if replaced by ZZ.sqrt(m // 2) ?

    while int(r1) >= bound:
        quo = r0 // r1
        r0, r1 = r1, r0 - quo*r1
        s0, s1 = s1, s0 - quo*s1

    if abs(int(s1)) >= bound:
        return None

    if s1 < 0:
        a, b = -r1, -s1
    elif s1 > 0:
        a, b = r1, s1
    else:
        return None

    field = domain.get_field()

    return field(a) / field(b)


def _rational_reconstruction_int_coeffs(hm, m, ring):
    r"""
    Reconstruct every rational coefficient `c_h` of a polynomial `h` in
    `\mathbb Q[t_1, \ldots, t_k][x, z]` from the corresponding integer
    coefficient `c_{h_m}` of a polynomial `h_m` in
    `\mathbb Z[t_1, \ldots, t_k][x, z]` such that

    .. math::

        c_{h_m} = c_h \; \mathrm{mod} \, m,

    where `m \in \mathbb Z`.

    The reconstruction is based on the Euclidean Algorithm. In general,
    `m` is not a prime number, so it is possible that this fails for some
    coefficient. In that case ``None`` is returned.

    Parameters
    ==========

    hm : PolyElement
        polynomial in `\mathbb Z[t_1, \ldots, t_k][x, z]`
    m : Integer
        modulus, not necessarily prime
    ring : PolyRing
        `\mathbb Q[t_1, \ldots, t_k][x, z]`, `h` will be an element of this
        ring

    Returns
    =======

    h : PolyElement
        reconstructed polynomial in `\mathbb Q[t_1, \ldots, t_k][x, z]` or
        ``None``

    See also
    ========

    _integer_rational_reconstruction

    """
    h = ring.zero

    if isinstance(ring.domain, PolynomialRing):
        reconstruction = _rational_reconstruction_int_coeffs
        domain = ring.domain.ring
    else:
        reconstruction = _integer_rational_reconstruction
        domain = hm.ring.domain

    for monom, coeff in hm.iterterms():
        coeffh = reconstruction(coeff, m, domain)

        if not coeffh:
            return None

        h[monom] = coeffh

    return h


def _func_field_modgcd_m(f, g, minpoly):
    r"""
    Compute the GCD of two polynomials in
    `\mathbb Q(t_1, \ldots, t_k)[z]/(m_{\alpha}(z))[x]` using a modular
    algorithm.

    The algorithm computes the GCD of two polynomials `f` and `g` by
    calculating the GCD in
    `\mathbb Z_p(t_1, \ldots, t_k)[z] / (\check m_{\alpha}(z))[x]` for
    suitable primes `p` and the primitive associate `\check m_{\alpha}(z)`
    of `m_{\alpha}(z)`. Then the coefficients are reconstructed with the
    Chinese Remainder Theorem and Rational Reconstruction. To compute the
    GCD over `\mathbb Z_p(t_1, \ldots, t_k)[z] / (\check m_{\alpha})[x]`,
    the recursive subroutine ``_func_field_modgcd_p`` is used. To verify the
    result in `\mathbb Q(t_1, \ldots, t_k)[z] / (m_{\alpha}(z))[x]`, a
    fraction free trial division is used.

    Parameters
    ==========

    f, g : PolyElement
        polynomials in `\mathbb Z[t_1, \ldots, t_k][x, z]`
    minpoly : PolyElement
        irreducible polynomial in `\mathbb Z[t_1, \ldots, t_k][z]`

    Returns
    =======

    h : PolyElement
        the primitive associate in `\mathbb Z[t_1, \ldots, t_k][x, z]` of
        the GCD of `f` and `g`

    Examples
    ========

    >>> from sympy.polys.modulargcd import _func_field_modgcd_m
    >>> from sympy.polys import ring, ZZ

    >>> R, x, z = ring('x, z', ZZ)
    >>> minpoly = (z**2 - 2).drop(0)

    >>> f = x**2 + 2*x*z + 2
    >>> g = x + z
    >>> _func_field_modgcd_m(f, g, minpoly)
    x + z

    >>> D, t = ring('t', ZZ)
    >>> R, x, z = ring('x, z', D)
    >>> minpoly = (z**2-3).drop(0)

    >>> f = x**2 + (t + 1)*x*z + 3*t
    >>> g = x*z + 3*t
    >>> _func_field_modgcd_m(f, g, minpoly)
    x + t*z

    References
    ==========

    1. [Hoeij04]_

    See also
    ========

    _func_field_modgcd_p

    """
    ring = f.ring
    domain = ring.domain

    if isinstance(domain, PolynomialRing):
        k = domain.ngens
        QQdomain = domain.ring.clone(domain=domain.domain.get_field())
        QQring = ring.clone(domain=QQdomain)
    else:
        k = 0
        QQring = ring.clone(domain=ring.domain.get_field())

    cf, f = f.primitive()
    cg, g = g.primitive()

    # polynomial in Z[t_1, ..., t_k][z]
    gamma = ring.dmp_LC(f) * ring.dmp_LC(g)
    # polynomial in Z[t_1, ..., t_k]
    delta = minpoly.LC

    p = 1
    primes = []
    hplist = []
    LMlist = []

    while True:
        p = nextprime(p)

        if gamma.trunc_ground(p) == 0:
            continue

        if k == 0:
            test = (delta % p == 0)
        else:
            test = (delta.trunc_ground(p) == 0)

        if test:
            continue

        fp = f.trunc_ground(p)
        gp = g.trunc_ground(p)
        minpolyp = minpoly.trunc_ground(p)

        hp = _func_field_modgcd_p(fp, gp, minpolyp, p)

        if hp is None:
            continue

        if hp == 1:
            return ring.one

        LM = [hp.degree()] + [0]*k
        if k > 0:
            for monom, coeff in hp.iterterms():
                if monom[0] == LM[0] and coeff.LM > tuple(LM[1:]):
                    LM[1:] = coeff.LM

        hm = hp
        m = p

        for q, hq, LMhq in zip(primes, hplist, LMlist):
            if LMhq == LM:
                hm = _chinese_remainder_reconstruction_multivariate(hq, hm, q, m)
                m *= q

        primes.append(p)
        hplist.append(hp)
        LMlist.append(LM)

        hm = _rational_reconstruction_int_coeffs(hm, m, QQring)

        if hm is None:
            continue

        if k == 0:
            h = hm.clear_denoms()[1]
        else:
            den = domain.domain.one
            for coeff in hm.itercoeffs():
                den = domain.domain.lcm(den, coeff.clear_denoms()[0])
            h = hm.mul_ground(den)

        # convert back to Z[t_1, ..., t_k][x, z] from Q[t_1, ..., t_k][x, z]
        h = h.set_ring(ring)
        h = h.primitive()[1]

        if not (_trial_division(f.mul_ground(cf), h, minpoly) or
            _trial_division(g.mul_ground(cg), h, minpoly)):
            return h


def _to_ZZ_poly(f, ring):
    r"""
    Compute an associate of a polynomial
    `f \in \mathbb Q(\alpha)[x_0, \ldots, x_{n-1}]` in
    `\mathbb Z[x_1, \ldots, x_{n-1}][z] / (\check m_{\alpha}(z))[x_0]`,
    where `\check m_{\alpha}(z) \in \mathbb Z[z]` is the primitive associate
    of the minimal polynomial `m_{\alpha}(z)` of `\alpha` over
    `\mathbb Q`.

    Parameters
    ==========

    f : PolyElement
        polynomial in `\mathbb Q(\alpha)[x_0, \ldots, x_{n-1}]`
    ring : PolyRing
        `\mathbb Z[x_1, \ldots, x_{n-1}][x_0, z]`

    Returns
    =======

    f_ : PolyElement
        associate of `f` in
        `\mathbb Z[x_1, \ldots, x_{n-1}][x_0, z]`

    """
    f_ = ring.zero

    if isinstance(ring.domain, PolynomialRing):
        domain = ring.domain.domain
    else:
        domain = ring.domain

    den = domain.one

    for coeff in f.itercoeffs():
        for c in coeff.to_list():
            if c:
                den = domain.lcm(den, c.denominator)

    for monom, coeff in f.iterterms():
        coeff = coeff.to_list()
        m = ring.domain.one
        if isinstance(ring.domain, PolynomialRing):
            m = m.mul_monom(monom[1:])
        n = len(coeff)

        for i in range(n):
            if coeff[i]:
                c = domain.convert(coeff[i] * den) * m

                if (monom[0], n-i-1) not in f_:
                    f_[(monom[0], n-i-1)] = c
                else:
                    f_[(monom[0], n-i-1)] += c

    return f_


def _to_ANP_poly(f, ring):
    r"""
    Convert a polynomial
    `f \in \mathbb Z[x_1, \ldots, x_{n-1}][z]/(\check m_{\alpha}(z))[x_0]`
    to a polynomial in `\mathbb Q(\alpha)[x_0, \ldots, x_{n-1}]`,
    where `\check m_{\alpha}(z) \in \mathbb Z[z]` is the primitive associate
    of the minimal polynomial `m_{\alpha}(z)` of `\alpha` over
    `\mathbb Q`.

    Parameters
    ==========

    f : PolyElement
        polynomial in `\mathbb Z[x_1, \ldots, x_{n-1}][x_0, z]`
    ring : PolyRing
        `\mathbb Q(\alpha)[x_0, \ldots, x_{n-1}]`

    Returns
    =======

    f_ : PolyElement
        polynomial in `\mathbb Q(\alpha)[x_0, \ldots, x_{n-1}]`

    """
    domain = ring.domain
    f_ = ring.zero

    if isinstance(f.ring.domain, PolynomialRing):
        for monom, coeff in f.iterterms():
            for mon, coef in coeff.iterterms():
                m = (monom[0],) + mon
                c = domain([domain.domain(coef)] + [0]*monom[1])

                if m not in f_:
                    f_[m] = c
                else:
                    f_[m] += c

    else:
        for monom, coeff in f.iterterms():
            m = (monom[0],)
            c = domain([domain.domain(coeff)] + [0]*monom[1])

            if m not in f_:
                f_[m] = c
            else:
                f_[m] += c

    return f_


def _minpoly_from_dense(minpoly, ring):
    r"""
    Change representation of the minimal polynomial from ``DMP`` to
    ``PolyElement`` for a given ring.
    """
    minpoly_ = ring.zero

    for monom, coeff in minpoly.terms():
        minpoly_[monom] = ring.domain(coeff)

    return minpoly_


def _primitive_in_x0(f):
    r"""
    Compute the content in `x_0` and the primitive part of a polynomial `f`
    in
    `\mathbb Q(\alpha)[x_0, x_1, \ldots, x_{n-1}] \cong \mathbb Q(\alpha)[x_1, \ldots, x_{n-1}][x_0]`.
    """
    fring = f.ring
    ring = fring.drop_to_ground(*range(1, fring.ngens))
    dom = ring.domain.ring
    f_ = ring(f.as_expr())
    cont = dom.zero

    for coeff in f_.itercoeffs():
        cont = func_field_modgcd(cont, coeff)[0]
        if cont == dom.one:
            return cont, f

    return cont, f.quo(cont.set_ring(fring))


# TODO: add support for algebraic function fields
def func_field_modgcd(f, g):
    r"""
    Compute the GCD of two polynomials `f` and `g` in
    `\mathbb Q(\alpha)[x_0, \ldots, x_{n-1}]` using a modular algorithm.

    The algorithm first computes the primitive associate
    `\check m_{\alpha}(z)` of the minimal polynomial `m_{\alpha}` in
    `\mathbb{Z}[z]` and the primitive associates of `f` and `g` in
    `\mathbb{Z}[x_1, \ldots, x_{n-1}][z]/(\check m_{\alpha})[x_0]`. Then it
    computes the GCD in
    `\mathbb Q(x_1, \ldots, x_{n-1})[z]/(m_{\alpha}(z))[x_0]`.
    This is done by calculating the GCD in
    `\mathbb{Z}_p(x_1, \ldots, x_{n-1})[z]/(\check m_{\alpha}(z))[x_0]` for
    suitable primes `p` and then reconstructing the coefficients with the
    Chinese Remainder Theorem and Rational Reconstruction. The GCD over
    `\mathbb{Z}_p(x_1, \ldots, x_{n-1})[z]/(\check m_{\alpha}(z))[x_0]` is
    computed with a recursive subroutine, which evaluates the polynomials at
    `x_{n-1} = a` for suitable evaluation points `a \in \mathbb Z_p` and
    then calls itself recursively until the ground domain does no longer
    contain any parameters. For
    `\mathbb{Z}_p[z]/(\check m_{\alpha}(z))[x_0]` the Euclidean Algorithm is
    used. The results of those recursive calls are then interpolated and
    Rational Function Reconstruction is used to obtain the correct
    coefficients. The results, both in
    `\mathbb Q(x_1, \ldots, x_{n-1})[z]/(m_{\alpha}(z))[x_0]` and
    `\mathbb{Z}_p(x_1, \ldots, x_{n-1})[z]/(\check m_{\alpha}(z))[x_0]`, are
    verified by a fraction free trial division.

    Apart from the above GCD computation some GCDs in
    `\mathbb Q(\alpha)[x_1, \ldots, x_{n-1}]` have to be calculated,
    because treating the polynomials as univariate ones can result in
    a spurious content of the GCD. For this ``func_field_modgcd`` is
    called recursively.

    Parameters
    ==========

    f, g : PolyElement
        polynomials in `\mathbb Q(\alpha)[x_0, \ldots, x_{n-1}]`

    Returns
    =======

    h : PolyElement
        monic GCD of the polynomials `f` and `g`
    cff : PolyElement
        cofactor of `f`, i.e. `\frac f h`
    cfg : PolyElement
        cofactor of `g`, i.e. `\frac g h`

    Examples
    ========

    >>> from sympy.polys.modulargcd import func_field_modgcd
    >>> from sympy.polys import AlgebraicField, QQ, ring
    >>> from sympy import sqrt

    >>> A = AlgebraicField(QQ, sqrt(2))
    >>> R, x = ring('x', A)

    >>> f = x**2 - 2
    >>> g = x + sqrt(2)

    >>> h, cff, cfg = func_field_modgcd(f, g)

    >>> h == x + sqrt(2)
    True
    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    >>> R, x, y = ring('x, y', A)

    >>> f = x**2 + 2*sqrt(2)*x*y + 2*y**2
    >>> g = x + sqrt(2)*y

    >>> h, cff, cfg = func_field_modgcd(f, g)

    >>> h == x + sqrt(2)*y
    True
    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    >>> f = x + sqrt(2)*y
    >>> g = x + y

    >>> h, cff, cfg = func_field_modgcd(f, g)

    >>> h == R.one
    True
    >>> cff * h == f
    True
    >>> cfg * h == g
    True

    References
    ==========

    1. [Hoeij04]_

    """
    ring = f.ring
    domain = ring.domain
    n = ring.ngens

    assert ring == g.ring and domain.is_Algebraic

    result = _trivial_gcd(f, g)
    if result is not None:
        return result

    z = Dummy('z')

    ZZring = ring.clone(symbols=ring.symbols + (z,), domain=domain.domain.get_ring())

    if n == 1:
        f_ = _to_ZZ_poly(f, ZZring)
        g_ = _to_ZZ_poly(g, ZZring)
        minpoly = ZZring.drop(0).from_dense(domain.mod.to_list())

        h = _func_field_modgcd_m(f_, g_, minpoly)
        h = _to_ANP_poly(h, ring)

    else:
        # contx0f in Q(a)[x_1, ..., x_{n-1}], f in Q(a)[x_0, ..., x_{n-1}]
        contx0f, f = _primitive_in_x0(f)
        contx0g, g = _primitive_in_x0(g)
        contx0h = func_field_modgcd(contx0f, contx0g)[0]

        ZZring_ = ZZring.drop_to_ground(*range(1, n))

        f_ = _to_ZZ_poly(f, ZZring_)
        g_ = _to_ZZ_poly(g, ZZring_)
        minpoly = _minpoly_from_dense(domain.mod, ZZring_.drop(0))

        h = _func_field_modgcd_m(f_, g_, minpoly)
        h = _to_ANP_poly(h, ring)

        contx0h_, h = _primitive_in_x0(h)
        h *= contx0h.set_ring(ring)
        f *= contx0f.set_ring(ring)
        g *= contx0g.set_ring(ring)

    h = h.quo_ground(h.LC)

    return h, f.quo(h), g.quo(h)
<|endoftext|>"""Tools for manipulation of rational expressions. """


from sympy.core import Basic, Add, sympify
from sympy.core.exprtools import gcd_terms
from sympy.utilities import public
from sympy.utilities.iterables import iterable


@public
def together(expr, deep=False, fraction=True):
    """
    Denest and combine rational expressions using symbolic methods.

    This function takes an expression or a container of expressions
    and puts it (them) together by denesting and combining rational
    subexpressions. No heroic measures are taken to minimize degree
    of the resulting numerator and denominator. To obtain completely
    reduced expression use :func:`~.cancel`. However, :func:`~.together`
    can preserve as much as possible of the structure of the input
    expression in the output (no expansion is performed).

    A wide variety of objects can be put together including lists,
    tuples, sets, relational objects, integrals and others. It is
    also possible to transform interior of function applications,
    by setting ``deep`` flag to ``True``.

    By definition, :func:`~.together` is a complement to :func:`~.apart`,
    so ``apart(together(expr))`` should return expr unchanged. Note
    however, that :func:`~.together` uses only symbolic methods, so
    it might be necessary to use :func:`~.cancel` to perform algebraic
    simplification and minimize degree of the numerator and denominator.

    Examples
    ========

    >>> from sympy import together, exp
    >>> from sympy.abc import x, y, z

    >>> together(1/x + 1/y)
    (x + y)/(x*y)
    >>> together(1/x + 1/y + 1/z)
    (x*y + x*z + y*z)/(x*y*z)

    >>> together(1/(x*y) + 1/y**2)
    (x + y)/(x*y**2)

    >>> together(1/(1 + 1/x) + 1/(1 + 1/y))
    (x*(y + 1) + y*(x + 1))/((x + 1)*(y + 1))

    >>> together(exp(1/x + 1/y))
    exp(1/y + 1/x)
    >>> together(exp(1/x + 1/y), deep=True)
    exp((x + y)/(x*y))

    >>> together(1/exp(x) + 1/(x*exp(x)))
    (x + 1)*exp(-x)/x

    >>> together(1/exp(2*x) + 1/(x*exp(3*x)))
    (x*exp(x) + 1)*exp(-3*x)/x

    """
    def _together(expr):
        if isinstance(expr, Basic):
            if expr.is_Atom or (expr.is_Function and not deep):
                return expr
            elif expr.is_Add:
                return gcd_terms(list(map(_together, Add.make_args(expr))), fraction=fraction)
            elif expr.is_Pow:
                base = _together(expr.base)

                if deep:
                    exp = _together(expr.exp)
                else:
                    exp = expr.exp

                return expr.func(base, exp)
            else:
                return expr.func(*[ _together(arg) for arg in expr.args ])
        elif iterable(expr):
            return expr.__class__([ _together(ex) for ex in expr ])

        return expr

    return _together(sympify(expr))
<|endoftext|>"""Euclidean algorithms, GCDs, LCMs and polynomial remainder sequences. """


from sympy.polys.densearith import (
    dup_sub_mul,
    dup_neg, dmp_neg,
    dmp_add,
    dmp_sub,
    dup_mul, dmp_mul,
    dmp_pow,
    dup_div, dmp_div,
    dup_rem,
    dup_quo, dmp_quo,
    dup_prem, dmp_prem,
    dup_mul_ground, dmp_mul_ground,
    dmp_mul_term,
    dup_quo_ground, dmp_quo_ground,
    dup_max_norm, dmp_max_norm)
from sympy.polys.densebasic import (
    dup_strip, dmp_raise,
    dmp_zero, dmp_one, dmp_ground,
    dmp_one_p, dmp_zero_p,
    dmp_zeros,
    dup_degree, dmp_degree, dmp_degree_in,
    dup_LC, dmp_LC, dmp_ground_LC,
    dmp_multi_deflate, dmp_inflate,
    dup_convert, dmp_convert,
    dmp_apply_pairs)
from sympy.polys.densetools import (
    dup_clear_denoms, dmp_clear_denoms,
    dup_diff, dmp_diff,
    dup_eval, dmp_eval, dmp_eval_in,
    dup_trunc, dmp_ground_trunc,
    dup_monic, dmp_ground_monic,
    dup_primitive, dmp_ground_primitive,
    dup_extract, dmp_ground_extract)
from sympy.polys.galoistools import (
    gf_int, gf_crt)
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import (
    MultivariatePolynomialError,
    HeuristicGCDFailed,
    HomomorphismFailed,
    NotInvertible,
    DomainError)




def dup_half_gcdex(f, g, K):
    """
    Half extended Euclidean algorithm in `F[x]`.

    Returns ``(s, h)`` such that ``h = gcd(f, g)`` and ``s*f = h (mod g)``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    >>> f = x**4 - 2*x**3 - 6*x**2 + 12*x + 15
    >>> g = x**3 + x**2 - 4*x - 4

    >>> R.dup_half_gcdex(f, g)
    (-1/5*x + 3/5, x + 1)

    """
    if not K.is_Field:
        raise DomainError("Cannot compute half extended GCD over %s" % K)

    a, b = [K.one], []

    while g:
        q, r = dup_div(f, g, K)
        f, g = g, r
        a, b = b, dup_sub_mul(a, q, b, K)

    a = dup_quo_ground(a, dup_LC(f, K), K)
    f = dup_monic(f, K)

    return a, f


def dmp_half_gcdex(f, g, u, K):
    """
    Half extended Euclidean algorithm in `F[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    """
    if not u:
        return dup_half_gcdex(f, g, K)
    else:
        raise MultivariatePolynomialError(f, g)


def dup_gcdex(f, g, K):
    """
    Extended Euclidean algorithm in `F[x]`.

    Returns ``(s, t, h)`` such that ``h = gcd(f, g)`` and ``s*f + t*g = h``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    >>> f = x**4 - 2*x**3 - 6*x**2 + 12*x + 15
    >>> g = x**3 + x**2 - 4*x - 4

    >>> R.dup_gcdex(f, g)
    (-1/5*x + 3/5, 1/5*x**2 - 6/5*x + 2, x + 1)

    """
    s, h = dup_half_gcdex(f, g, K)

    F = dup_sub_mul(h, s, f, K)
    t = dup_quo(F, g, K)

    return s, t, h


def dmp_gcdex(f, g, u, K):
    """
    Extended Euclidean algorithm in `F[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    """
    if not u:
        return dup_gcdex(f, g, K)
    else:
        raise MultivariatePolynomialError(f, g)


def dup_invert(f, g, K):
    """
    Compute multiplicative inverse of `f` modulo `g` in `F[x]`.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    >>> f = x**2 - 1
    >>> g = 2*x - 1
    >>> h = x - 1

    >>> R.dup_invert(f, g)
    -4/3

    >>> R.dup_invert(f, h)
    Traceback (most recent call last):
    ...
    NotInvertible: zero divisor

    """
    s, h = dup_half_gcdex(f, g, K)

    if h == [K.one]:
        return dup_rem(s, g, K)
    else:
        raise NotInvertible("zero divisor")


def dmp_invert(f, g, u, K):
    """
    Compute multiplicative inverse of `f` modulo `g` in `F[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    """
    if not u:
        return dup_invert(f, g, K)
    else:
        raise MultivariatePolynomialError(f, g)


def dup_euclidean_prs(f, g, K):
    """
    Euclidean polynomial remainder sequence (PRS) in `K[x]`.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    >>> f = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    >>> g = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    >>> prs = R.dup_euclidean_prs(f, g)

    >>> prs[0]
    x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    >>> prs[1]
    3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21
    >>> prs[2]
    -5/9*x**4 + 1/9*x**2 - 1/3
    >>> prs[3]
    -117/25*x**2 - 9*x + 441/25
    >>> prs[4]
    233150/19773*x - 102500/6591
    >>> prs[5]
    -1288744821/543589225

    """
    prs = [f, g]
    h = dup_rem(f, g, K)

    while h:
        prs.append(h)
        f, g = g, h
        h = dup_rem(f, g, K)

    return prs


def dmp_euclidean_prs(f, g, u, K):
    """
    Euclidean polynomial remainder sequence (PRS) in `K[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    """
    if not u:
        return dup_euclidean_prs(f, g, K)
    else:
        raise MultivariatePolynomialError(f, g)


def dup_primitive_prs(f, g, K):
    """
    Primitive polynomial remainder sequence (PRS) in `K[x]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> f = x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    >>> g = 3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21

    >>> prs = R.dup_primitive_prs(f, g)

    >>> prs[0]
    x**8 + x**6 - 3*x**4 - 3*x**3 + 8*x**2 + 2*x - 5
    >>> prs[1]
    3*x**6 + 5*x**4 - 4*x**2 - 9*x + 21
    >>> prs[2]
    -5*x**4 + x**2 - 3
    >>> prs[3]
    13*x**2 + 25*x - 49
    >>> prs[4]
    4663*x - 6150
    >>> prs[5]
    1

    """
    prs = [f, g]
    _, h = dup_primitive(dup_prem(f, g, K), K)

    while h:
        prs.append(h)
        f, g = g, h
        _, h = dup_primitive(dup_prem(f, g, K), K)

    return prs


def dmp_primitive_prs(f, g, u, K):
    """
    Primitive polynomial remainder sequence (PRS) in `K[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    """
    if not u:
        return dup_primitive_prs(f, g, K)
    else:
        raise MultivariatePolynomialError(f, g)


def dup_inner_subresultants(f, g, K):
    """
    Subresultant PRS algorithm in `K[x]`.

    Computes the subresultant polynomial remainder sequence (PRS)
    and the non-zero scalar subresultants of `f` and `g`.
    By [1] Thm. 3, these are the constants '-c' (- to optimize
    computation of sign).
    The first subdeterminant is set to 1 by convention to match
    the polynomial and the scalar subdeterminants.
    If 'deg(f) < deg(g)', the subresultants of '(g,f)' are computed.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_inner_subresultants(x**2 + 1, x**2 - 1)
    ([x**2 + 1, x**2 - 1, -2], [1, 1, 4])

    References
    ==========

    .. [1] W.S. Brown, The Subresultant PRS Algorithm.
           ACM Transaction of Mathematical Software 4 (1978) 237-249

    """
    n = dup_degree(f)
    m = dup_degree(g)

    if n < m:
        f, g = g, f
        n, m = m, n

    if not f:
        return [], []

    if not g:
        return [f], [K.one]

    R = [f, g]
    d = n - m

    b = (-K.one)**(d + 1)

    h = dup_prem(f, g, K)
    h = dup_mul_ground(h, b, K)

    lc = dup_LC(g, K)
    c = lc**d

    # Conventional first scalar subdeterminant is 1
    S = [K.one, c]
    c = -c

    while h:
        k = dup_degree(h)
        R.append(h)

        f, g, m, d = g, h, k, m - k

        b = -lc * c**d

        h = dup_prem(f, g, K)
        h = dup_quo_ground(h, b, K)

        lc = dup_LC(g, K)

        if d > 1:        # abnormal case
            q = c**(d - 1)
            c = K.quo((-lc)**d, q)
        else:
            c = -lc

        S.append(-c)

    return R, S


def dup_subresultants(f, g, K):
    """
    Computes subresultant PRS of two polynomials in `K[x]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_subresultants(x**2 + 1, x**2 - 1)
    [x**2 + 1, x**2 - 1, -2]

    """
    return dup_inner_subresultants(f, g, K)[0]


def dup_prs_resultant(f, g, K):
    """
    Resultant algorithm in `K[x]` using subresultant PRS.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_prs_resultant(x**2 + 1, x**2 - 1)
    (4, [x**2 + 1, x**2 - 1, -2])

    """
    if not f or not g:
        return (K.zero, [])

    R, S = dup_inner_subresultants(f, g, K)

    if dup_degree(R[-1]) > 0:
        return (K.zero, R)

    return S[-1], R


def dup_resultant(f, g, K, includePRS=False):
    """
    Computes resultant of two polynomials in `K[x]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_resultant(x**2 + 1, x**2 - 1)
    4

    """
    if includePRS:
        return dup_prs_resultant(f, g, K)
    return dup_prs_resultant(f, g, K)[0]


def dmp_inner_subresultants(f, g, u, K):
    """
    Subresultant PRS algorithm in `K[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = 3*x**2*y - y**3 - 4
    >>> g = x**2 + x*y**3 - 9

    >>> a = 3*x*y**4 + y**3 - 27*y + 4
    >>> b = -3*y**10 - 12*y**7 + y**6 - 54*y**4 + 8*y**3 + 729*y**2 - 216*y + 16

    >>> prs = [f, g, a, b]
    >>> sres = [[1], [1], [3, 0, 0, 0, 0], [-3, 0, 0, -12, 1, 0, -54, 8, 729, -216, 16]]

    >>> R.dmp_inner_subresultants(f, g) == (prs, sres)
    True

    """
    if not u:
        return dup_inner_subresultants(f, g, K)

    n = dmp_degree(f, u)
    m = dmp_degree(g, u)

    if n < m:
        f, g = g, f
        n, m = m, n

    if dmp_zero_p(f, u):
        return [], []

    v = u - 1
    if dmp_zero_p(g, u):
        return [f], [dmp_ground(K.one, v)]

    R = [f, g]
    d = n - m

    b = dmp_pow(dmp_ground(-K.one, v), d + 1, v, K)

    h = dmp_prem(f, g, u, K)
    h = dmp_mul_term(h, b, 0, u, K)

    lc = dmp_LC(g, K)
    c = dmp_pow(lc, d, v, K)

    S = [dmp_ground(K.one, v), c]
    c = dmp_neg(c, v, K)

    while not dmp_zero_p(h, u):
        k = dmp_degree(h, u)
        R.append(h)

        f, g, m, d = g, h, k, m - k

        b = dmp_mul(dmp_neg(lc, v, K),
                    dmp_pow(c, d, v, K), v, K)

        h = dmp_prem(f, g, u, K)
        h = [ dmp_quo(ch, b, v, K) for ch in h ]

        lc = dmp_LC(g, K)

        if d > 1:
            p = dmp_pow(dmp_neg(lc, v, K), d, v, K)
            q = dmp_pow(c, d - 1, v, K)
            c = dmp_quo(p, q, v, K)
        else:
            c = dmp_neg(lc, v, K)

        S.append(dmp_neg(c, v, K))

    return R, S


def dmp_subresultants(f, g, u, K):
    """
    Computes subresultant PRS of two polynomials in `K[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = 3*x**2*y - y**3 - 4
    >>> g = x**2 + x*y**3 - 9

    >>> a = 3*x*y**4 + y**3 - 27*y + 4
    >>> b = -3*y**10 - 12*y**7 + y**6 - 54*y**4 + 8*y**3 + 729*y**2 - 216*y + 16

    >>> R.dmp_subresultants(f, g) == [f, g, a, b]
    True

    """
    return dmp_inner_subresultants(f, g, u, K)[0]


def dmp_prs_resultant(f, g, u, K):
    """
    Resultant algorithm in `K[X]` using subresultant PRS.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = 3*x**2*y - y**3 - 4
    >>> g = x**2 + x*y**3 - 9

    >>> a = 3*x*y**4 + y**3 - 27*y + 4
    >>> b = -3*y**10 - 12*y**7 + y**6 - 54*y**4 + 8*y**3 + 729*y**2 - 216*y + 16

    >>> res, prs = R.dmp_prs_resultant(f, g)

    >>> res == b             # resultant has n-1 variables
    False
    >>> res == b.drop(x)
    True
    >>> prs == [f, g, a, b]
    True

    """
    if not u:
        return dup_prs_resultant(f, g, K)

    if dmp_zero_p(f, u) or dmp_zero_p(g, u):
        return (dmp_zero(u - 1), [])

    R, S = dmp_inner_subresultants(f, g, u, K)

    if dmp_degree(R[-1], u) > 0:
        return (dmp_zero(u - 1), R)

    return S[-1], R


def dmp_zz_modular_resultant(f, g, p, u, K):
    """
    Compute resultant of `f` and `g` modulo a prime `p`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = x + y + 2
    >>> g = 2*x*y + x + 3

    >>> R.dmp_zz_modular_resultant(f, g, 5)
    -2*y**2 + 1

    """
    if not u:
        return gf_int(dup_prs_resultant(f, g, K)[0] % p, p)

    v = u - 1

    n = dmp_degree(f, u)
    m = dmp_degree(g, u)

    N = dmp_degree_in(f, 1, u)
    M = dmp_degree_in(g, 1, u)

    B = n*M + m*N

    D, a = [K.one], -K.one
    r = dmp_zero(v)

    while dup_degree(D) <= B:
        while True:
            a += K.one

            if a == p:
                raise HomomorphismFailed('no luck')

            F = dmp_eval_in(f, gf_int(a, p), 1, u, K)

            if dmp_degree(F, v) == n:
                G = dmp_eval_in(g, gf_int(a, p), 1, u, K)

                if dmp_degree(G, v) == m:
                    break

        R = dmp_zz_modular_resultant(F, G, p, v, K)
        e = dmp_eval(r, a, v, K)

        if not v:
            R = dup_strip([R])
            e = dup_strip([e])
        else:
            R = [R]
            e = [e]

        d = K.invert(dup_eval(D, a, K), p)
        d = dup_mul_ground(D, d, K)
        d = dmp_raise(d, v, 0, K)

        c = dmp_mul(d, dmp_sub(R, e, v, K), v, K)
        r = dmp_add(r, c, v, K)

        r = dmp_ground_trunc(r, p, v, K)

        D = dup_mul(D, [K.one, -a], K)
        D = dup_trunc(D, p, K)

    return r


def _collins_crt(r, R, P, p, K):
    """Wrapper of CRT for Collins's resultant algorithm. """
    return gf_int(gf_crt([r, R], [P, p], K), P*p)


def dmp_zz_collins_resultant(f, g, u, K):
    """
    Collins's modular resultant algorithm in `Z[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = x + y + 2
    >>> g = 2*x*y + x + 3

    >>> R.dmp_zz_collins_resultant(f, g)
    -2*y**2 - 5*y + 1

    """

    n = dmp_degree(f, u)
    m = dmp_degree(g, u)

    if n < 0 or m < 0:
        return dmp_zero(u - 1)

    A = dmp_max_norm(f, u, K)
    B = dmp_max_norm(g, u, K)

    a = dmp_ground_LC(f, u, K)
    b = dmp_ground_LC(g, u, K)

    v = u - 1

    B = K(2)*K.factorial(K(n + m))*A**m*B**n
    r, p, P = dmp_zero(v), K.one, K.one

    from sympy.ntheory import nextprime

    while P <= B:
        p = K(nextprime(p))

        while not (a % p) or not (b % p):
            p = K(nextprime(p))

        F = dmp_ground_trunc(f, p, u, K)
        G = dmp_ground_trunc(g, p, u, K)

        try:
            R = dmp_zz_modular_resultant(F, G, p, u, K)
        except HomomorphismFailed:
            continue

        if K.is_one(P):
            r = R
        else:
            r = dmp_apply_pairs(r, R, _collins_crt, (P, p, K), v, K)

        P *= p

    return r


def dmp_qq_collins_resultant(f, g, u, K0):
    """
    Collins's modular resultant algorithm in `Q[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x,y = ring("x,y", QQ)

    >>> f = QQ(1,2)*x + y + QQ(2,3)
    >>> g = 2*x*y + x + 3

    >>> R.dmp_qq_collins_resultant(f, g)
    -2*y**2 - 7/3*y + 5/6

    """
    n = dmp_degree(f, u)
    m = dmp_degree(g, u)

    if n < 0 or m < 0:
        return dmp_zero(u - 1)

    K1 = K0.get_ring()

    cf, f = dmp_clear_denoms(f, u, K0, K1)
    cg, g = dmp_clear_denoms(g, u, K0, K1)

    f = dmp_convert(f, u, K0, K1)
    g = dmp_convert(g, u, K0, K1)

    r = dmp_zz_collins_resultant(f, g, u, K1)
    r = dmp_convert(r, u - 1, K1, K0)

    c = K0.convert(cf**m * cg**n, K1)

    return dmp_quo_ground(r, c, u - 1, K0)


def dmp_resultant(f, g, u, K, includePRS=False):
    """
    Computes resultant of two polynomials in `K[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = 3*x**2*y - y**3 - 4
    >>> g = x**2 + x*y**3 - 9

    >>> R.dmp_resultant(f, g)
    -3*y**10 - 12*y**7 + y**6 - 54*y**4 + 8*y**3 + 729*y**2 - 216*y + 16

    """
    if not u:
        return dup_resultant(f, g, K, includePRS=includePRS)

    if includePRS:
        return dmp_prs_resultant(f, g, u, K)

    if K.is_Field:
        if K.is_QQ and query('USE_COLLINS_RESULTANT'):
            return dmp_qq_collins_resultant(f, g, u, K)
    else:
        if K.is_ZZ and query('USE_COLLINS_RESULTANT'):
            return dmp_zz_collins_resultant(f, g, u, K)

    return dmp_prs_resultant(f, g, u, K)[0]


def dup_discriminant(f, K):
    """
    Computes discriminant of a polynomial in `K[x]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_discriminant(x**2 + 2*x + 3)
    -8

    """
    d = dup_degree(f)

    if d <= 0:
        return K.zero
    else:
        s = (-1)**((d*(d - 1)) // 2)
        c = dup_LC(f, K)

        r = dup_resultant(f, dup_diff(f, 1, K), K)

        return K.quo(r, c*K(s))


def dmp_discriminant(f, u, K):
    """
    Computes discriminant of a polynomial in `K[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y,z,t = ring("x,y,z,t", ZZ)

    >>> R.dmp_discriminant(x**2*y + x*z + t)
    -4*y*t + z**2

    """
    if not u:
        return dup_discriminant(f, K)

    d, v = dmp_degree(f, u), u - 1

    if d <= 0:
        return dmp_zero(v)
    else:
        s = (-1)**((d*(d - 1)) // 2)
        c = dmp_LC(f, K)

        r = dmp_resultant(f, dmp_diff(f, 1, u, K), u, K)
        c = dmp_mul_ground(c, K(s), v, K)

        return dmp_quo(r, c, v, K)


def _dup_rr_trivial_gcd(f, g, K):
    """Handle trivial cases in GCD algorithm over a ring. """
    if not (f or g):
        return [], [], []
    elif not f:
        if K.is_nonnegative(dup_LC(g, K)):
            return g, [], [K.one]
        else:
            return dup_neg(g, K), [], [-K.one]
    elif not g:
        if K.is_nonnegative(dup_LC(f, K)):
            return f, [K.one], []
        else:
            return dup_neg(f, K), [-K.one], []

    return None


def _dup_ff_trivial_gcd(f, g, K):
    """Handle trivial cases in GCD algorithm over a field. """
    if not (f or g):
        return [], [], []
    elif not f:
        return dup_monic(g, K), [], [dup_LC(g, K)]
    elif not g:
        return dup_monic(f, K), [dup_LC(f, K)], []
    else:
        return None


def _dmp_rr_trivial_gcd(f, g, u, K):
    """Handle trivial cases in GCD algorithm over a ring. """
    zero_f = dmp_zero_p(f, u)
    zero_g = dmp_zero_p(g, u)
    if_contain_one = dmp_one_p(f, u, K) or dmp_one_p(g, u, K)

    if zero_f and zero_g:
        return tuple(dmp_zeros(3, u, K))
    elif zero_f:
        if K.is_nonnegative(dmp_ground_LC(g, u, K)):
            return g, dmp_zero(u), dmp_one(u, K)
        else:
            return dmp_neg(g, u, K), dmp_zero(u), dmp_ground(-K.one, u)
    elif zero_g:
        if K.is_nonnegative(dmp_ground_LC(f, u, K)):
            return f, dmp_one(u, K), dmp_zero(u)
        else:
            return dmp_neg(f, u, K), dmp_ground(-K.one, u), dmp_zero(u)
    elif if_contain_one:
        return dmp_one(u, K), f, g
    elif query('USE_SIMPLIFY_GCD'):
        return _dmp_simplify_gcd(f, g, u, K)
    else:
        return None


def _dmp_ff_trivial_gcd(f, g, u, K):
    """Handle trivial cases in GCD algorithm over a field. """
    zero_f = dmp_zero_p(f, u)
    zero_g = dmp_zero_p(g, u)

    if zero_f and zero_g:
        return tuple(dmp_zeros(3, u, K))
    elif zero_f:
        return (dmp_ground_monic(g, u, K),
                dmp_zero(u),
                dmp_ground(dmp_ground_LC(g, u, K), u))
    elif zero_g:
        return (dmp_ground_monic(f, u, K),
                dmp_ground(dmp_ground_LC(f, u, K), u),
                dmp_zero(u))
    elif query('USE_SIMPLIFY_GCD'):
        return _dmp_simplify_gcd(f, g, u, K)
    else:
        return None


def _dmp_simplify_gcd(f, g, u, K):
    """Try to eliminate `x_0` from GCD computation in `K[X]`. """
    df = dmp_degree(f, u)
    dg = dmp_degree(g, u)

    if df > 0 and dg > 0:
        return None

    if not (df or dg):
        F = dmp_LC(f, K)
        G = dmp_LC(g, K)
    else:
        if not df:
            F = dmp_LC(f, K)
            G = dmp_content(g, u, K)
        else:
            F = dmp_content(f, u, K)
            G = dmp_LC(g, K)

    v = u - 1
    h = dmp_gcd(F, G, v, K)

    cff = [ dmp_quo(cf, h, v, K) for cf in f ]
    cfg = [ dmp_quo(cg, h, v, K) for cg in g ]

    return [h], cff, cfg


def dup_rr_prs_gcd(f, g, K):
    """
    Computes polynomial GCD using subresultants over a ring.

    Returns ``(h, cff, cfg)`` such that ``a = gcd(f, g)``, ``cff = quo(f, h)``,
    and ``cfg = quo(g, h)``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_rr_prs_gcd(x**2 - 1, x**2 - 3*x + 2)
    (x - 1, x + 1, x - 2)

    """
    result = _dup_rr_trivial_gcd(f, g, K)

    if result is not None:
        return result

    fc, F = dup_primitive(f, K)
    gc, G = dup_primitive(g, K)

    c = K.gcd(fc, gc)

    h = dup_subresultants(F, G, K)[-1]
    _, h = dup_primitive(h, K)

    c *= K.canonical_unit(dup_LC(h, K))

    h = dup_mul_ground(h, c, K)

    cff = dup_quo(f, h, K)
    cfg = dup_quo(g, h, K)

    return h, cff, cfg


def dup_ff_prs_gcd(f, g, K):
    """
    Computes polynomial GCD using subresultants over a field.

    Returns ``(h, cff, cfg)`` such that ``a = gcd(f, g)``, ``cff = quo(f, h)``,
    and ``cfg = quo(g, h)``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    >>> R.dup_ff_prs_gcd(x**2 - 1, x**2 - 3*x + 2)
    (x - 1, x + 1, x - 2)

    """
    result = _dup_ff_trivial_gcd(f, g, K)

    if result is not None:
        return result

    h = dup_subresultants(f, g, K)[-1]
    h = dup_monic(h, K)

    cff = dup_quo(f, h, K)
    cfg = dup_quo(g, h, K)

    return h, cff, cfg


def dmp_rr_prs_gcd(f, g, u, K):
    """
    Computes polynomial GCD using subresultants over a ring.

    Returns ``(h, cff, cfg)`` such that ``a = gcd(f, g)``, ``cff = quo(f, h)``,
    and ``cfg = quo(g, h)``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y, = ring("x,y", ZZ)

    >>> f = x**2 + 2*x*y + y**2
    >>> g = x**2 + x*y

    >>> R.dmp_rr_prs_gcd(f, g)
    (x + y, x + y, x)

    """
    if not u:
        return dup_rr_prs_gcd(f, g, K)

    result = _dmp_rr_trivial_gcd(f, g, u, K)

    if result is not None:
        return result

    fc, F = dmp_primitive(f, u, K)
    gc, G = dmp_primitive(g, u, K)

    h = dmp_subresultants(F, G, u, K)[-1]
    c, _, _ = dmp_rr_prs_gcd(fc, gc, u - 1, K)

    _, h = dmp_primitive(h, u, K)
    h = dmp_mul_term(h, c, 0, u, K)

    unit = K.canonical_unit(dmp_ground_LC(h, u, K))

    if unit != K.one:
        h = dmp_mul_ground(h, unit, u, K)

    cff = dmp_quo(f, h, u, K)
    cfg = dmp_quo(g, h, u, K)

    return h, cff, cfg


def dmp_ff_prs_gcd(f, g, u, K):
    """
    Computes polynomial GCD using subresultants over a field.

    Returns ``(h, cff, cfg)`` such that ``a = gcd(f, g)``, ``cff = quo(f, h)``,
    and ``cfg = quo(g, h)``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x,y, = ring("x,y", QQ)

    >>> f = QQ(1,2)*x**2 + x*y + QQ(1,2)*y**2
    >>> g = x**2 + x*y

    >>> R.dmp_ff_prs_gcd(f, g)
    (x + y, 1/2*x + 1/2*y, x)

    """
    if not u:
        return dup_ff_prs_gcd(f, g, K)

    result = _dmp_ff_trivial_gcd(f, g, u, K)

    if result is not None:
        return result

    fc, F = dmp_primitive(f, u, K)
    gc, G = dmp_primitive(g, u, K)

    h = dmp_subresultants(F, G, u, K)[-1]
    c, _, _ = dmp_ff_prs_gcd(fc, gc, u - 1, K)

    _, h = dmp_primitive(h, u, K)
    h = dmp_mul_term(h, c, 0, u, K)
    h = dmp_ground_monic(h, u, K)

    cff = dmp_quo(f, h, u, K)
    cfg = dmp_quo(g, h, u, K)

    return h, cff, cfg

HEU_GCD_MAX = 6


def _dup_zz_gcd_interpolate(h, x, K):
    """Interpolate polynomial GCD from integer GCD. """
    f = []

    while h:
        g = h % x

        if g > x // 2:
            g -= x

        f.insert(0, g)
        h = (h - g) // x

    return f


def dup_zz_heu_gcd(f, g, K):
    """
    Heuristic polynomial GCD in `Z[x]`.

    Given univariate polynomials `f` and `g` in `Z[x]`, returns
    their GCD and cofactors, i.e. polynomials ``h``, ``cff`` and ``cfg``
    such that::

          h = gcd(f, g), cff = quo(f, h) and cfg = quo(g, h)

    The algorithm is purely heuristic which means it may fail to compute
    the GCD. This will be signaled by raising an exception. In this case
    you will need to switch to another GCD method.

    The algorithm computes the polynomial GCD by evaluating polynomials
    f and g at certain points and computing (fast) integer GCD of those
    evaluations. The polynomial GCD is recovered from the integer image
    by interpolation.  The final step is to verify if the result is the
    correct GCD. This gives cofactors as a side effect.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_zz_heu_gcd(x**2 - 1, x**2 - 3*x + 2)
    (x - 1, x + 1, x - 2)

    References
    ==========

    .. [1] [Liao95]_

    """
    result = _dup_rr_trivial_gcd(f, g, K)

    if result is not None:
        return result

    df = dup_degree(f)
    dg = dup_degree(g)

    gcd, f, g = dup_extract(f, g, K)

    if df == 0 or dg == 0:
        return [gcd], f, g

    f_norm = dup_max_norm(f, K)
    g_norm = dup_max_norm(g, K)

    B = K(2*min(f_norm, g_norm) + 29)

    x = max(min(B, 99*K.sqrt(B)),
            2*min(f_norm // abs(dup_LC(f, K)),
                  g_norm // abs(dup_LC(g, K))) + 4)

    for i in range(0, HEU_GCD_MAX):
        ff = dup_eval(f, x, K)
        gg = dup_eval(g, x, K)

        if ff and gg:
            h = K.gcd(ff, gg)

            cff = ff // h
            cfg = gg // h

            h = _dup_zz_gcd_interpolate(h, x, K)
            h = dup_primitive(h, K)[1]

            cff_, r = dup_div(f, h, K)

            if not r:
                cfg_, r = dup_div(g, h, K)

                if not r:
                    h = dup_mul_ground(h, gcd, K)
                    return h, cff_, cfg_

            cff = _dup_zz_gcd_interpolate(cff, x, K)

            h, r = dup_div(f, cff, K)

            if not r:
                cfg_, r = dup_div(g, h, K)

                if not r:
                    h = dup_mul_ground(h, gcd, K)
                    return h, cff, cfg_

            cfg = _dup_zz_gcd_interpolate(cfg, x, K)

            h, r = dup_div(g, cfg, K)

            if not r:
                cff_, r = dup_div(f, h, K)

                if not r:
                    h = dup_mul_ground(h, gcd, K)
                    return h, cff_, cfg

        x = 73794*x * K.sqrt(K.sqrt(x)) // 27011

    raise HeuristicGCDFailed('no luck')


def _dmp_zz_gcd_interpolate(h, x, v, K):
    """Interpolate polynomial GCD from integer GCD. """
    f = []

    while not dmp_zero_p(h, v):
        g = dmp_ground_trunc(h, x, v, K)
        f.insert(0, g)

        h = dmp_sub(h, g, v, K)
        h = dmp_quo_ground(h, x, v, K)

    if K.is_negative(dmp_ground_LC(f, v + 1, K)):
        return dmp_neg(f, v + 1, K)
    else:
        return f


def dmp_zz_heu_gcd(f, g, u, K):
    """
    Heuristic polynomial GCD in `Z[X]`.

    Given univariate polynomials `f` and `g` in `Z[X]`, returns
    their GCD and cofactors, i.e. polynomials ``h``, ``cff`` and ``cfg``
    such that::

          h = gcd(f, g), cff = quo(f, h) and cfg = quo(g, h)

    The algorithm is purely heuristic which means it may fail to compute
    the GCD. This will be signaled by raising an exception. In this case
    you will need to switch to another GCD method.

    The algorithm computes the polynomial GCD by evaluating polynomials
    f and g at certain points and computing (fast) integer GCD of those
    evaluations. The polynomial GCD is recovered from the integer image
    by interpolation. The evaluation process reduces f and g variable by
    variable into a large integer.  The final step is to verify if the
    interpolated polynomial is the correct GCD. This gives cofactors of
    the input polynomials as a side effect.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y, = ring("x,y", ZZ)

    >>> f = x**2 + 2*x*y + y**2
    >>> g = x**2 + x*y

    >>> R.dmp_zz_heu_gcd(f, g)
    (x + y, x + y, x)

    References
    ==========

    .. [1] [Liao95]_

    """
    if not u:
        return dup_zz_heu_gcd(f, g, K)

    result = _dmp_rr_trivial_gcd(f, g, u, K)

    if result is not None:
        return result

    gcd, f, g = dmp_ground_extract(f, g, u, K)

    f_norm = dmp_max_norm(f, u, K)
    g_norm = dmp_max_norm(g, u, K)

    B = K(2*min(f_norm, g_norm) + 29)

    x = max(min(B, 99*K.sqrt(B)),
            2*min(f_norm // abs(dmp_ground_LC(f, u, K)),
                  g_norm // abs(dmp_ground_LC(g, u, K))) + 4)

    for i in range(0, HEU_GCD_MAX):
        ff = dmp_eval(f, x, u, K)
        gg = dmp_eval(g, x, u, K)

        v = u - 1

        if not (dmp_zero_p(ff, v) or dmp_zero_p(gg, v)):
            h, cff, cfg = dmp_zz_heu_gcd(ff, gg, v, K)

            h = _dmp_zz_gcd_interpolate(h, x, v, K)
            h = dmp_ground_primitive(h, u, K)[1]

            cff_, r = dmp_div(f, h, u, K)

            if dmp_zero_p(r, u):
                cfg_, r = dmp_div(g, h, u, K)

                if dmp_zero_p(r, u):
                    h = dmp_mul_ground(h, gcd, u, K)
                    return h, cff_, cfg_

            cff = _dmp_zz_gcd_interpolate(cff, x, v, K)

            h, r = dmp_div(f, cff, u, K)

            if dmp_zero_p(r, u):
                cfg_, r = dmp_div(g, h, u, K)

                if dmp_zero_p(r, u):
                    h = dmp_mul_ground(h, gcd, u, K)
                    return h, cff, cfg_

            cfg = _dmp_zz_gcd_interpolate(cfg, x, v, K)

            h, r = dmp_div(g, cfg, u, K)

            if dmp_zero_p(r, u):
                cff_, r = dmp_div(f, h, u, K)

                if dmp_zero_p(r, u):
                    h = dmp_mul_ground(h, gcd, u, K)
                    return h, cff_, cfg

        x = 73794*x * K.sqrt(K.sqrt(x)) // 27011

    raise HeuristicGCDFailed('no luck')


def dup_qq_heu_gcd(f, g, K0):
    """
    Heuristic polynomial GCD in `Q[x]`.

    Returns ``(h, cff, cfg)`` such that ``a = gcd(f, g)``,
    ``cff = quo(f, h)``, and ``cfg = quo(g, h)``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    >>> f = QQ(1,2)*x**2 + QQ(7,4)*x + QQ(3,2)
    >>> g = QQ(1,2)*x**2 + x

    >>> R.dup_qq_heu_gcd(f, g)
    (x + 2, 1/2*x + 3/4, 1/2*x)

    """
    result = _dup_ff_trivial_gcd(f, g, K0)

    if result is not None:
        return result

    K1 = K0.get_ring()

    cf, f = dup_clear_denoms(f, K0, K1)
    cg, g = dup_clear_denoms(g, K0, K1)

    f = dup_convert(f, K0, K1)
    g = dup_convert(g, K0, K1)

    h, cff, cfg = dup_zz_heu_gcd(f, g, K1)

    h = dup_convert(h, K1, K0)

    c = dup_LC(h, K0)
    h = dup_monic(h, K0)

    cff = dup_convert(cff, K1, K0)
    cfg = dup_convert(cfg, K1, K0)

    cff = dup_mul_ground(cff, K0.quo(c, cf), K0)
    cfg = dup_mul_ground(cfg, K0.quo(c, cg), K0)

    return h, cff, cfg


def dmp_qq_heu_gcd(f, g, u, K0):
    """
    Heuristic polynomial GCD in `Q[X]`.

    Returns ``(h, cff, cfg)`` such that ``a = gcd(f, g)``,
    ``cff = quo(f, h)``, and ``cfg = quo(g, h)``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x,y, = ring("x,y", QQ)

    >>> f = QQ(1,4)*x**2 + x*y + y**2
    >>> g = QQ(1,2)*x**2 + x*y

    >>> R.dmp_qq_heu_gcd(f, g)
    (x + 2*y, 1/4*x + 1/2*y, 1/2*x)

    """
    result = _dmp_ff_trivial_gcd(f, g, u, K0)

    if result is not None:
        return result

    K1 = K0.get_ring()

    cf, f = dmp_clear_denoms(f, u, K0, K1)
    cg, g = dmp_clear_denoms(g, u, K0, K1)

    f = dmp_convert(f, u, K0, K1)
    g = dmp_convert(g, u, K0, K1)

    h, cff, cfg = dmp_zz_heu_gcd(f, g, u, K1)

    h = dmp_convert(h, u, K1, K0)

    c = dmp_ground_LC(h, u, K0)
    h = dmp_ground_monic(h, u, K0)

    cff = dmp_convert(cff, u, K1, K0)
    cfg = dmp_convert(cfg, u, K1, K0)

    cff = dmp_mul_ground(cff, K0.quo(c, cf), u, K0)
    cfg = dmp_mul_ground(cfg, K0.quo(c, cg), u, K0)

    return h, cff, cfg


def dup_inner_gcd(f, g, K):
    """
    Computes polynomial GCD and cofactors of `f` and `g` in `K[x]`.

    Returns ``(h, cff, cfg)`` such that ``a = gcd(f, g)``,
    ``cff = quo(f, h)``, and ``cfg = quo(g, h)``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_inner_gcd(x**2 - 1, x**2 - 3*x + 2)
    (x - 1, x + 1, x - 2)

    """
    # XXX: This used to check for K.is_Exact but leads to awkward results when
    # the domain is something like RR[z] e.g.:
    #
    # >>> g, p, q = Poly(1, x).cancel(Poly(51.05*x*y - 1.0, x))
    # >>> g
    # 1.0
    # >>> p
    # Poly(17592186044421.0, x, domain='RR[y]')
    # >>> q
    # Poly(898081097567692.0*y*x - 17592186044421.0, x, domain='RR[y]'))
    #
    # Maybe it would be better to flatten into multivariate polynomials first.
    if K.is_RR or K.is_CC:
        try:
            exact = K.get_exact()
        except DomainError:
            return [K.one], f, g

        f = dup_convert(f, K, exact)
        g = dup_convert(g, K, exact)

        h, cff, cfg = dup_inner_gcd(f, g, exact)

        h = dup_convert(h, exact, K)
        cff = dup_convert(cff, exact, K)
        cfg = dup_convert(cfg, exact, K)

        return h, cff, cfg
    elif K.is_Field:
        if K.is_QQ and query('USE_HEU_GCD'):
            try:
                return dup_qq_heu_gcd(f, g, K)
            except HeuristicGCDFailed:
                pass

        return dup_ff_prs_gcd(f, g, K)
    else:
        if K.is_ZZ and query('USE_HEU_GCD'):
            try:
                return dup_zz_heu_gcd(f, g, K)
            except HeuristicGCDFailed:
                pass

        return dup_rr_prs_gcd(f, g, K)


def _dmp_inner_gcd(f, g, u, K):
    """Helper function for `dmp_inner_gcd()`. """
    if not K.is_Exact:
        try:
            exact = K.get_exact()
        except DomainError:
            return dmp_one(u, K), f, g

        f = dmp_convert(f, u, K, exact)
        g = dmp_convert(g, u, K, exact)

        h, cff, cfg = _dmp_inner_gcd(f, g, u, exact)

        h = dmp_convert(h, u, exact, K)
        cff = dmp_convert(cff, u, exact, K)
        cfg = dmp_convert(cfg, u, exact, K)

        return h, cff, cfg
    elif K.is_Field:
        if K.is_QQ and query('USE_HEU_GCD'):
            try:
                return dmp_qq_heu_gcd(f, g, u, K)
            except HeuristicGCDFailed:
                pass

        return dmp_ff_prs_gcd(f, g, u, K)
    else:
        if K.is_ZZ and query('USE_HEU_GCD'):
            try:
                return dmp_zz_heu_gcd(f, g, u, K)
            except HeuristicGCDFailed:
                pass

        return dmp_rr_prs_gcd(f, g, u, K)


def dmp_inner_gcd(f, g, u, K):
    """
    Computes polynomial GCD and cofactors of `f` and `g` in `K[X]`.

    Returns ``(h, cff, cfg)`` such that ``a = gcd(f, g)``,
    ``cff = quo(f, h)``, and ``cfg = quo(g, h)``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y, = ring("x,y", ZZ)

    >>> f = x**2 + 2*x*y + y**2
    >>> g = x**2 + x*y

    >>> R.dmp_inner_gcd(f, g)
    (x + y, x + y, x)

    """
    if not u:
        return dup_inner_gcd(f, g, K)

    J, (f, g) = dmp_multi_deflate((f, g), u, K)
    h, cff, cfg = _dmp_inner_gcd(f, g, u, K)

    return (dmp_inflate(h, J, u, K),
            dmp_inflate(cff, J, u, K),
            dmp_inflate(cfg, J, u, K))


def dup_gcd(f, g, K):
    """
    Computes polynomial GCD of `f` and `g` in `K[x]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_gcd(x**2 - 1, x**2 - 3*x + 2)
    x - 1

    """
    return dup_inner_gcd(f, g, K)[0]


def dmp_gcd(f, g, u, K):
    """
    Computes polynomial GCD of `f` and `g` in `K[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y, = ring("x,y", ZZ)

    >>> f = x**2 + 2*x*y + y**2
    >>> g = x**2 + x*y

    >>> R.dmp_gcd(f, g)
    x + y

    """
    return dmp_inner_gcd(f, g, u, K)[0]


def dup_rr_lcm(f, g, K):
    """
    Computes polynomial LCM over a ring in `K[x]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_rr_lcm(x**2 - 1, x**2 - 3*x + 2)
    x**3 - 2*x**2 - x + 2

    """
    if not f or not g:
        return dmp_zero(0)

    fc, f = dup_primitive(f, K)
    gc, g = dup_primitive(g, K)

    c = K.lcm(fc, gc)

    h = dup_quo(dup_mul(f, g, K),
                dup_gcd(f, g, K), K)

    u = K.canonical_unit(dup_LC(h, K))

    return dup_mul_ground(h, c*u, K)


def dup_ff_lcm(f, g, K):
    """
    Computes polynomial LCM over a field in `K[x]`.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    >>> f = QQ(1,2)*x**2 + QQ(7,4)*x + QQ(3,2)
    >>> g = QQ(1,2)*x**2 + x

    >>> R.dup_ff_lcm(f, g)
    x**3 + 7/2*x**2 + 3*x

    """
    h = dup_quo(dup_mul(f, g, K),
                dup_gcd(f, g, K), K)

    return dup_monic(h, K)


def dup_lcm(f, g, K):
    """
    Computes polynomial LCM of `f` and `g` in `K[x]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_lcm(x**2 - 1, x**2 - 3*x + 2)
    x**3 - 2*x**2 - x + 2

    """
    if K.is_Field:
        return dup_ff_lcm(f, g, K)
    else:
        return dup_rr_lcm(f, g, K)


def dmp_rr_lcm(f, g, u, K):
    """
    Computes polynomial LCM over a ring in `K[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y, = ring("x,y", ZZ)

    >>> f = x**2 + 2*x*y + y**2
    >>> g = x**2 + x*y

    >>> R.dmp_rr_lcm(f, g)
    x**3 + 2*x**2*y + x*y**2

    """
    fc, f = dmp_ground_primitive(f, u, K)
    gc, g = dmp_ground_primitive(g, u, K)

    c = K.lcm(fc, gc)

    h = dmp_quo(dmp_mul(f, g, u, K),
                dmp_gcd(f, g, u, K), u, K)

    return dmp_mul_ground(h, c, u, K)


def dmp_ff_lcm(f, g, u, K):
    """
    Computes polynomial LCM over a field in `K[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x,y, = ring("x,y", QQ)

    >>> f = QQ(1,4)*x**2 + x*y + y**2
    >>> g = QQ(1,2)*x**2 + x*y

    >>> R.dmp_ff_lcm(f, g)
    x**3 + 4*x**2*y + 4*x*y**2

    """
    h = dmp_quo(dmp_mul(f, g, u, K),
                dmp_gcd(f, g, u, K), u, K)

    return dmp_ground_monic(h, u, K)


def dmp_lcm(f, g, u, K):
    """
    Computes polynomial LCM of `f` and `g` in `K[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y, = ring("x,y", ZZ)

    >>> f = x**2 + 2*x*y + y**2
    >>> g = x**2 + x*y

    >>> R.dmp_lcm(f, g)
    x**3 + 2*x**2*y + x*y**2

    """
    if not u:
        return dup_lcm(f, g, K)

    if K.is_Field:
        return dmp_ff_lcm(f, g, u, K)
    else:
        return dmp_rr_lcm(f, g, u, K)


def dmp_content(f, u, K):
    """
    Returns GCD of multivariate coefficients.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y, = ring("x,y", ZZ)

    >>> R.dmp_content(2*x*y + 6*x + 4*y + 12)
    2*y + 6

    """
    cont, v = dmp_LC(f, K), u - 1

    if dmp_zero_p(f, u):
        return cont

    for c in f[1:]:
        cont = dmp_gcd(cont, c, v, K)

        if dmp_one_p(cont, v, K):
            break

    if K.is_negative(dmp_ground_LC(cont, v, K)):
        return dmp_neg(cont, v, K)
    else:
        return cont


def dmp_primitive(f, u, K):
    """
    Returns multivariate content and a primitive polynomial.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y, = ring("x,y", ZZ)

    >>> R.dmp_primitive(2*x*y + 6*x + 4*y + 12)
    (2*y + 6, x + 2)

    """
    cont, v = dmp_content(f, u, K), u - 1

    if dmp_zero_p(f, u) or dmp_one_p(cont, v, K):
        return cont, f
    else:
        return cont, [ dmp_quo(c, cont, v, K) for c in f ]


def dup_cancel(f, g, K, include=True):
    """
    Cancel common factors in a rational function `f/g`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_cancel(2*x**2 - 2, x**2 - 2*x + 1)
    (2*x + 2, x - 1)

    """
    return dmp_cancel(f, g, 0, K, include=include)


def dmp_cancel(f, g, u, K, include=True):
    """
    Cancel common factors in a rational function `f/g`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_cancel(2*x**2 - 2, x**2 - 2*x + 1)
    (2*x + 2, x - 1)

    """
    K0 = None

    if K.is_Field and K.has_assoc_Ring:
        K0, K = K, K.get_ring()

        cq, f = dmp_clear_denoms(f, u, K0, K, convert=True)
        cp, g = dmp_clear_denoms(g, u, K0, K, convert=True)
    else:
        cp, cq = K.one, K.one

    _, p, q = dmp_inner_gcd(f, g, u, K)

    if K0 is not None:
        _, cp, cq = K.cofactors(cp, cq)

        p = dmp_convert(p, u, K, K0)
        q = dmp_convert(q, u, K, K0)

        K = K0

    p_neg = K.is_negative(dmp_ground_LC(p, u, K))
    q_neg = K.is_negative(dmp_ground_LC(q, u, K))

    if p_neg and q_neg:
        p, q = dmp_neg(p, u, K), dmp_neg(q, u, K)
    elif p_neg:
        cp, p = -cp, dmp_neg(p, u, K)
    elif q_neg:
        cp, q = -cp, dmp_neg(q, u, K)

    if not include:
        return cp, cq, p, q

    p = dmp_mul_ground(p, cp, u, K)
    q = dmp_mul_ground(q, cq, u, K)

    return p, q
<|endoftext|>"""Efficient functions for generating orthogonal polynomials."""
from sympy.core.symbol import Dummy
from sympy.polys.densearith import (dup_mul, dup_mul_ground,
    dup_lshift, dup_sub, dup_add, dup_sub_term, dup_sub_ground, dup_sqr)
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polytools import named_poly
from sympy.utilities import public

def dup_jacobi(n, a, b, K):
    """Low-level implementation of Jacobi polynomials."""
    if n < 1:
        return [K.one]
    m2, m1 = [K.one], [(a+b)/K(2) + K.one, (a-b)/K(2)]
    for i in range(2, n+1):
        den = K(i)*(a + b + i)*(a + b + K(2)*i - K(2))
        f0 = (a + b + K(2)*i - K.one) * (a*a - b*b) / (K(2)*den)
        f1 = (a + b + K(2)*i - K.one) * (a + b + K(2)*i - K(2)) * (a + b + K(2)*i) / (K(2)*den)
        f2 = (a + i - K.one)*(b + i - K.one)*(a + b + K(2)*i) / den
        p0 = dup_mul_ground(m1, f0, K)
        p1 = dup_mul_ground(dup_lshift(m1, 1, K), f1, K)
        p2 = dup_mul_ground(m2, f2, K)
        m2, m1 = m1, dup_sub(dup_add(p0, p1, K), p2, K)
    return m1

@public
def jacobi_poly(n, a, b, x=None, polys=False):
    r"""Generates the Jacobi polynomial `P_n^{(a,b)}(x)`.

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    a
        Lower limit of minimal domain for the list of coefficients.
    b
        Upper limit of minimal domain for the list of coefficients.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
    return named_poly(n, dup_jacobi, None, "Jacobi polynomial", (x, a, b), polys)

def dup_gegenbauer(n, a, K):
    """Low-level implementation of Gegenbauer polynomials."""
    if n < 1:
        return [K.one]
    m2, m1 = [K.one], [K(2)*a, K.zero]
    for i in range(2, n+1):
        p1 = dup_mul_ground(dup_lshift(m1, 1, K), K(2)*(a-K.one)/K(i) + K(2), K)
        p2 = dup_mul_ground(m2, K(2)*(a-K.one)/K(i) + K.one, K)
        m2, m1 = m1, dup_sub(p1, p2, K)
    return m1

def gegenbauer_poly(n, a, x=None, polys=False):
    r"""Generates the Gegenbauer polynomial `C_n^{(a)}(x)`.

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    a
        Decides minimal domain for the list of coefficients.
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
    return named_poly(n, dup_gegenbauer, None, "Gegenbauer polynomial", (x, a), polys)

def dup_chebyshevt(n, K):
    """Low-level implementation of Chebyshev polynomials of the first kind."""
    if n < 1:
        return [K.one]
    # When n is small, it is faster to directly calculate the recurrence relation.
    if n < 64: # The threshold serves as a heuristic
        return _dup_chebyshevt_rec(n, K)
    return _dup_chebyshevt_prod(n, K)

def _dup_chebyshevt_rec(n, K):
    r""" Chebyshev polynomials of the first kind using recurrence.

    Explanation
    ===========

    Chebyshev polynomials of the first kind are defined by the recurrence
    relation:

    .. math::
        T_0(x) &= 1\\
        T_1(x) &= x\\
        T_n(x) &= 2xT_{n-1}(x) - T_{n-2}(x)

    This function calculates the Chebyshev polynomial of the first kind using
    the above recurrence relation.

    Parameters
    ==========

    n : int
        n is a nonnegative integer.
    K : domain

    """
    m2, m1 = [K.one], [K.one, K.zero]
    for _ in range(n - 1):
        m2, m1 = m1, dup_sub(dup_mul_ground(dup_lshift(m1, 1, K), K(2), K), m2, K)
    return m1

def _dup_chebyshevt_prod(n, K):
    r""" Chebyshev polynomials of the first kind using recursive products.

    Explanation
    ===========

    Computes Chebyshev polynomials of the first kind using

    .. math::
        T_{2n}(x) &= 2T_n^2(x) - 1\\
        T_{2n+1}(x) &= 2T_{n+1}(x)T_n(x) - x

    This is faster than ``_dup_chebyshevt_rec`` for large ``n``.

    Parameters
    ==========

    n : int
        n is a nonnegative integer.
    K : domain

    """
    m2, m1 = [K.one, K.zero], [K(2), K.zero, -K.one]
    for i in bin(n)[3:]:
        c = dup_sub_term(dup_mul_ground(dup_mul(m1, m2, K), K(2), K), K.one, 1, K)
        if  i  == '1':
            m2, m1 = c, dup_sub_ground(dup_mul_ground(dup_sqr(m1, K), K(2), K), K.one, K)
        else:
            m2, m1 = dup_sub_ground(dup_mul_ground(dup_sqr(m2, K), K(2), K), K.one, K), c
    return m2

def dup_chebyshevu(n, K):
    """Low-level implementation of Chebyshev polynomials of the second kind."""
    if n < 1:
        return [K.one]
    m2, m1 = [K.one], [K(2), K.zero]
    for i in range(2, n+1):
        m2, m1 = m1, dup_sub(dup_mul_ground(dup_lshift(m1, 1, K), K(2), K), m2, K)
    return m1

@public
def chebyshevt_poly(n, x=None, polys=False):
    r"""Generates the Chebyshev polynomial of the first kind `T_n(x)`.

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
    return named_poly(n, dup_chebyshevt, ZZ,
            "Chebyshev polynomial of the first kind", (x,), polys)

@public
def chebyshevu_poly(n, x=None, polys=False):
    r"""Generates the Chebyshev polynomial of the second kind `U_n(x)`.

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
    return named_poly(n, dup_chebyshevu, ZZ,
            "Chebyshev polynomial of the second kind", (x,), polys)

def dup_hermite(n, K):
    """Low-level implementation of Hermite polynomials."""
    if n < 1:
        return [K.one]
    m2, m1 = [K.one], [K(2), K.zero]
    for i in range(2, n+1):
        a = dup_lshift(m1, 1, K)
        b = dup_mul_ground(m2, K(i-1), K)
        m2, m1 = m1, dup_mul_ground(dup_sub(a, b, K), K(2), K)
    return m1

def dup_hermite_prob(n, K):
    """Low-level implementation of probabilist's Hermite polynomials."""
    if n < 1:
        return [K.one]
    m2, m1 = [K.one], [K.one, K.zero]
    for i in range(2, n+1):
        a = dup_lshift(m1, 1, K)
        b = dup_mul_ground(m2, K(i-1), K)
        m2, m1 = m1, dup_sub(a, b, K)
    return m1

@public
def hermite_poly(n, x=None, polys=False):
    r"""Generates the Hermite polynomial `H_n(x)`.

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
    return named_poly(n, dup_hermite, ZZ, "Hermite polynomial", (x,), polys)

@public
def hermite_prob_poly(n, x=None, polys=False):
    r"""Generates the probabilist's Hermite polynomial `He_n(x)`.

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
    return named_poly(n, dup_hermite_prob, ZZ,
            "probabilist's Hermite polynomial", (x,), polys)

def dup_legendre(n, K):
    """Low-level implementation of Legendre polynomials."""
    if n < 1:
        return [K.one]
    m2, m1 = [K.one], [K.one, K.zero]
    for i in range(2, n+1):
        a = dup_mul_ground(dup_lshift(m1, 1, K), K(2*i-1, i), K)
        b = dup_mul_ground(m2, K(i-1, i), K)
        m2, m1 = m1, dup_sub(a, b, K)
    return m1

@public
def legendre_poly(n, x=None, polys=False):
    r"""Generates the Legendre polynomial `P_n(x)`.

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
    return named_poly(n, dup_legendre, QQ, "Legendre polynomial", (x,), polys)

def dup_laguerre(n, alpha, K):
    """Low-level implementation of Laguerre polynomials."""
    m2, m1 = [K.zero], [K.one]
    for i in range(1, n+1):
        a = dup_mul(m1, [-K.one/K(i), (alpha-K.one)/K(i) + K(2)], K)
        b = dup_mul_ground(m2, (alpha-K.one)/K(i) + K.one, K)
        m2, m1 = m1, dup_sub(a, b, K)
    return m1

@public
def laguerre_poly(n, x=None, alpha=0, polys=False):
    r"""Generates the Laguerre polynomial `L_n^{(\alpha)}(x)`.

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    alpha : optional
        Decides minimal domain for the list of coefficients.
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
    return named_poly(n, dup_laguerre, None, "Laguerre polynomial", (x, alpha), polys)

def dup_spherical_bessel_fn(n, K):
    """Low-level implementation of fn(n, x)."""
    if n < 1:
        return [K.one, K.zero]
    m2, m1 = [K.one], [K.one, K.zero]
    for i in range(2, n+1):
        m2, m1 = m1, dup_sub(dup_mul_ground(dup_lshift(m1, 1, K), K(2*i-1), K), m2, K)
    return dup_lshift(m1, 1, K)

def dup_spherical_bessel_fn_minus(n, K):
    """Low-level implementation of fn(-n, x)."""
    m2, m1 = [K.one, K.zero], [K.zero]
    for i in range(2, n+1):
        m2, m1 = m1, dup_sub(dup_mul_ground(dup_lshift(m1, 1, K), K(3-2*i), K), m2, K)
    return m1

def spherical_bessel_fn(n, x=None, polys=False):
    """
    Coefficients for the spherical Bessel functions.

    These are only needed in the jn() function.

    The coefficients are calculated from:

    fn(0, z) = 1/z
    fn(1, z) = 1/z**2
    fn(n-1, z) + fn(n+1, z) == (2*n+1)/z * fn(n, z)

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.

    Examples
    ========

    >>> from sympy.polys.orthopolys import spherical_bessel_fn as fn
    >>> from sympy import Symbol
    >>> z = Symbol("z")
    >>> fn(1, z)
    z**(-2)
    >>> fn(2, z)
    -1/z + 3/z**3
    >>> fn(3, z)
    -6/z**2 + 15/z**4
    >>> fn(4, z)
    1/z - 45/z**3 + 105/z**5

    """
    if x is None:
        x = Dummy("x")
    f = dup_spherical_bessel_fn_minus if n < 0 else dup_spherical_bessel_fn
    return named_poly(abs(n), f, ZZ, "", (QQ(1)/x,), polys)
<|endoftext|>"""Implementation of RootOf class and related tools. """



from sympy.core.basic import Basic
from sympy.core import (S, Expr, Integer, Float, I, oo, Add, Lambda,
    symbols, sympify, Rational, Dummy)
from sympy.core.cache import cacheit
from sympy.core.relational import is_le
from sympy.core.sorting import ordered
from sympy.polys.domains import QQ
from sympy.polys.polyerrors import (
    MultivariatePolynomialError,
    GeneratorsNeeded,
    PolynomialError,
    DomainError)
from sympy.polys.polyfuncs import symmetrize, viete
from sympy.polys.polyroots import (
    roots_linear, roots_quadratic, roots_binomial,
    preprocess_roots, roots)
from sympy.polys.polytools import Poly, PurePoly, factor
from sympy.polys.rationaltools import together
from sympy.polys.rootisolation import (
    dup_isolate_complex_roots_sqf,
    dup_isolate_real_roots_sqf)
from sympy.utilities import lambdify, public, sift, numbered_symbols

from mpmath import mpf, mpc, findroot, workprec
from mpmath.libmp.libmpf import dps_to_prec, prec_to_dps
from sympy.multipledispatch import dispatch
from itertools import chain


__all__ = ['CRootOf']



class _pure_key_dict:
    """A minimal dictionary that makes sure that the key is a
    univariate PurePoly instance.

    Examples
    ========

    Only the following actions are guaranteed:

    >>> from sympy.polys.rootoftools import _pure_key_dict
    >>> from sympy import PurePoly
    >>> from sympy.abc import x, y

    1) creation

    >>> P = _pure_key_dict()

    2) assignment for a PurePoly or univariate polynomial

    >>> P[x] = 1
    >>> P[PurePoly(x - y, x)] = 2

    3) retrieval based on PurePoly key comparison (use this
       instead of the get method)

    >>> P[y]
    1

    4) KeyError when trying to retrieve a nonexisting key

    >>> P[y + 1]
    Traceback (most recent call last):
    ...
    KeyError: PurePoly(y + 1, y, domain='ZZ')

    5) ability to query with ``in``

    >>> x + 1 in P
    False

    NOTE: this is a *not* a dictionary. It is a very basic object
    for internal use that makes sure to always address its cache
    via PurePoly instances. It does not, for example, implement
    ``get`` or ``setdefault``.
    """
    def __init__(self):
        self._dict = {}

    def __getitem__(self, k):
        if not isinstance(k, PurePoly):
            if not (isinstance(k, Expr) and len(k.free_symbols) == 1):
                raise KeyError
            k = PurePoly(k, expand=False)
        return self._dict[k]

    def __setitem__(self, k, v):
        if not isinstance(k, PurePoly):
            if not (isinstance(k, Expr) and len(k.free_symbols) == 1):
                raise ValueError('expecting univariate expression')
            k = PurePoly(k, expand=False)
        self._dict[k] = v

    def __contains__(self, k):
        try:
            self[k]
            return True
        except KeyError:
            return False

_reals_cache = _pure_key_dict()
_complexes_cache = _pure_key_dict()


def _pure_factors(poly):
    _, factors = poly.factor_list()
    return [(PurePoly(f, expand=False), m) for f, m in factors]


def _imag_count_of_factor(f):
    """Return the number of imaginary roots for irreducible
    univariate polynomial ``f``.
    """
    terms = [(i, j) for (i,), j in f.terms()]
    if any(i % 2 for i, j in terms):
        return 0
    # update signs
    even = [(i, I**i*j) for i, j in terms]
    even = Poly.from_dict(dict(even), Dummy('x'))
    return int(even.count_roots(-oo, oo))


@public
def rootof(f, x, index=None, radicals=True, expand=True):
    """An indexed root of a univariate polynomial.

    Returns either a :obj:`ComplexRootOf` object or an explicit
    expression involving radicals.

    Parameters
    ==========

    f : Expr
        Univariate polynomial.
    x : Symbol, optional
        Generator for ``f``.
    index : int or Integer
    radicals : bool
               Return a radical expression if possible.
    expand : bool
             Expand ``f``.
    """
    return CRootOf(f, x, index=index, radicals=radicals, expand=expand)


@public
class RootOf(Expr):
    """Represents a root of a univariate polynomial.

    Base class for roots of different kinds of polynomials.
    Only complex roots are currently supported.
    """

    __slots__ = ('poly',)

    def __new__(cls, f, x, index=None, radicals=True, expand=True):
        """Construct a new ``CRootOf`` object for ``k``-th root of ``f``."""
        return rootof(f, x, index=index, radicals=radicals, expand=expand)

@public
class ComplexRootOf(RootOf):
    """Represents an indexed complex root of a polynomial.

    Roots of a univariate polynomial separated into disjoint
    real or complex intervals and indexed in a fixed order:

    * real roots come first and are sorted in increasing order;
    * complex roots come next and are sorted primarily by increasing
      real part, secondarily by increasing imaginary part.

    Currently only rational coefficients are allowed.
    Can be imported as ``CRootOf``. To avoid confusion, the
    generator must be a Symbol.


    Examples
    ========

    >>> from sympy import CRootOf, rootof
    >>> from sympy.abc import x

    CRootOf is a way to reference a particular root of a
    polynomial. If there is a rational root, it will be returned:

    >>> CRootOf.clear_cache()  # for doctest reproducibility
    >>> CRootOf(x**2 - 4, 0)
    -2

    Whether roots involving radicals are returned or not
    depends on whether the ``radicals`` flag is true (which is
    set to True with rootof):

    >>> CRootOf(x**2 - 3, 0)
    CRootOf(x**2 - 3, 0)
    >>> CRootOf(x**2 - 3, 0, radicals=True)
    -sqrt(3)
    >>> rootof(x**2 - 3, 0)
    -sqrt(3)

    The following cannot be expressed in terms of radicals:

    >>> r = rootof(4*x**5 + 16*x**3 + 12*x**2 + 7, 0); r
    CRootOf(4*x**5 + 16*x**3 + 12*x**2 + 7, 0)

    The root bounds can be seen, however, and they are used by the
    evaluation methods to get numerical approximations for the root.

    >>> interval = r._get_interval(); interval
    (-1, 0)
    >>> r.evalf(2)
    -0.98

    The evalf method refines the width of the root bounds until it
    guarantees that any decimal approximation within those bounds
    will satisfy the desired precision. It then stores the refined
    interval so subsequent requests at or below the requested
    precision will not have to recompute the root bounds and will
    return very quickly.

    Before evaluation above, the interval was

    >>> interval
    (-1, 0)

    After evaluation it is now

    >>> r._get_interval() # doctest: +SKIP
    (-165/169, -206/211)

    To reset all intervals for a given polynomial, the :meth:`_reset` method
    can be called from any CRootOf instance of the polynomial:

    >>> r._reset()
    >>> r._get_interval()
    (-1, 0)

    The :meth:`eval_approx` method will also find the root to a given
    precision but the interval is not modified unless the search
    for the root fails to converge within the root bounds. And
    the secant method is used to find the root. (The ``evalf``
    method uses bisection and will always update the interval.)

    >>> r.eval_approx(2)
    -0.98

    The interval needed to be slightly updated to find that root:

    >>> r._get_interval()
    (-1, -1/2)

    The ``evalf_rational`` will compute a rational approximation
    of the root to the desired accuracy or precision.

    >>> r.eval_rational(n=2)
    -69629/71318

    >>> t = CRootOf(x**3 + 10*x + 1, 1)
    >>> t.eval_rational(1e-1)
    15/256 - 805*I/256
    >>> t.eval_rational(1e-1, 1e-4)
    3275/65536 - 414645*I/131072
    >>> t.eval_rational(1e-4, 1e-4)
    6545/131072 - 414645*I/131072
    >>> t.eval_rational(n=2)
    104755/2097152 - 6634255*I/2097152

    Notes
    =====

    Although a PurePoly can be constructed from a non-symbol generator
    RootOf instances of non-symbols are disallowed to avoid confusion
    over what root is being represented.

    >>> from sympy import exp, PurePoly
    >>> PurePoly(x) == PurePoly(exp(x))
    True
    >>> CRootOf(x - 1, 0)
    1
    >>> CRootOf(exp(x) - 1, 0)  # would correspond to x == 0
    Traceback (most recent call last):
    ...
    sympy.polys.polyerrors.PolynomialError: generator must be a Symbol

    See Also
    ========

    eval_approx
    eval_rational

    """

    __slots__ = ('index',)
    is_complex = True
    is_number = True
    is_finite = True
    is_algebraic = True

    def __new__(cls, f, x, index=None, radicals=False, expand=True):
        """ Construct an indexed complex root of a polynomial.

        See ``rootof`` for the parameters.

        The default value of ``radicals`` is ``False`` to satisfy
        ``eval(srepr(expr) == expr``.
        """
        x = sympify(x)

        if index is None and x.is_Integer:
            x, index = None, x
        else:
            index = sympify(index)

        if index is not None and index.is_Integer:
            index = int(index)
        else:
            raise ValueError("expected an integer root index, got %s" % index)

        poly = PurePoly(f, x, greedy=False, expand=expand)

        if not poly.is_univariate:
            raise PolynomialError("only univariate polynomials are allowed")

        if not poly.gen.is_Symbol:
            # PurePoly(sin(x) + 1) == PurePoly(x + 1) but the roots of
            # x for each are not the same: issue 8617
            raise PolynomialError("generator must be a Symbol")

        degree = poly.degree()

        if degree <= 0:
            raise PolynomialError("Cannot construct CRootOf object for %s" % f)

        if index < -degree or index >= degree:
            raise IndexError("root index out of [%d, %d] range, got %d" %
                             (-degree, degree - 1, index))
        elif index < 0:
            index += degree

        dom = poly.get_domain()

        if not dom.is_Exact:
            poly = poly.to_exact()

        roots = cls._roots_trivial(poly, radicals)

        if roots is not None:
            return roots[index]

        coeff, poly = preprocess_roots(poly)
        dom = poly.get_domain()

        if not dom.is_ZZ:
            raise NotImplementedError("CRootOf is not supported over %s" % dom)

        root = cls._indexed_root(poly, index, lazy=True)
        return coeff * cls._postprocess_root(root, radicals)

    @classmethod
    def _new(cls, poly, index):
        """Construct new ``CRootOf`` object from raw data. """
        obj = Expr.__new__(cls)

        obj.poly = PurePoly(poly)
        obj.index = index

        try:
            _reals_cache[obj.poly] = _reals_cache[poly]
            _complexes_cache[obj.poly] = _complexes_cache[poly]
        except KeyError:
            pass

        return obj

    def _hashable_content(self):
        return (self.poly, self.index)

    @property
    def expr(self):
        return self.poly.as_expr()

    @property
    def args(self):
        return (self.expr, Integer(self.index))

    @property
    def free_symbols(self):
        # CRootOf currently only works with univariate expressions
        # whose poly attribute should be a PurePoly with no free
        # symbols
        return set()

    def _eval_is_real(self):
        """Return ``True`` if the root is real. """
        self._ensure_reals_init()
        return self.index < len(_reals_cache[self.poly])

    def _eval_is_imaginary(self):
        """Return ``True`` if the root is imaginary. """
        self._ensure_reals_init()
        if self.index >= len(_reals_cache[self.poly]):
            ivl = self._get_interval()
            return ivl.ax*ivl.bx <= 0  # all others are on one side or the other
        return False  # XXX is this necessary?

    @classmethod
    def real_roots(cls, poly, radicals=True):
        """Get real roots of a polynomial. """
        return cls._get_roots("_real_roots", poly, radicals)

    @classmethod
    def all_roots(cls, poly, radicals=True):
        """Get real and complex roots of a polynomial. """
        return cls._get_roots("_all_roots", poly, radicals)

    @classmethod
    def _get_reals_sqf(cls, currentfactor, use_cache=True):
        """Get real root isolating intervals for a square-free factor."""
        if use_cache and currentfactor in _reals_cache:
            real_part = _reals_cache[currentfactor]
        else:
            _reals_cache[currentfactor] = real_part = \
                dup_isolate_real_roots_sqf(
                    currentfactor.rep.to_list(), currentfactor.rep.dom, blackbox=True)

        return real_part

    @classmethod
    def _get_complexes_sqf(cls, currentfactor, use_cache=True):
        """Get complex root isolating intervals for a square-free factor."""
        if use_cache and currentfactor in _complexes_cache:
            complex_part = _complexes_cache[currentfactor]
        else:
            _complexes_cache[currentfactor] = complex_part = \
                dup_isolate_complex_roots_sqf(
                currentfactor.rep.to_list(), currentfactor.rep.dom, blackbox=True)
        return complex_part

    @classmethod
    def _get_reals(cls, factors, use_cache=True):
        """Compute real root isolating intervals for a list of factors. """
        reals = []

        for currentfactor, k in factors:
            try:
                if not use_cache:
                    raise KeyError
                r = _reals_cache[currentfactor]
                reals.extend([(i, currentfactor, k) for i in r])
            except KeyError:
                real_part = cls._get_reals_sqf(currentfactor, use_cache)
                new = [(root, currentfactor, k) for root in real_part]
                reals.extend(new)

        reals = cls._reals_sorted(reals)
        return reals

    @classmethod
    def _get_complexes(cls, factors, use_cache=True):
        """Compute complex root isolating intervals for a list of factors. """
        complexes = []

        for currentfactor, k in ordered(factors):
            try:
                if not use_cache:
                    raise KeyError
                c = _complexes_cache[currentfactor]
                complexes.extend([(i, currentfactor, k) for i in c])
            except KeyError:
                complex_part = cls._get_complexes_sqf(currentfactor, use_cache)
                new = [(root, currentfactor, k) for root in complex_part]
                complexes.extend(new)

        complexes = cls._complexes_sorted(complexes)
        return complexes

    @classmethod
    def _reals_sorted(cls, reals):
        """Make real isolating intervals disjoint and sort roots. """
        cache = {}

        for i, (u, f, k) in enumerate(reals):
            for j, (v, g, m) in enumerate(reals[i + 1:]):
                u, v = u.refine_disjoint(v)
                reals[i + j + 1] = (v, g, m)

            reals[i] = (u, f, k)

        reals = sorted(reals, key=lambda r: r[0].a)

        for root, currentfactor, _ in reals:
            if currentfactor in cache:
                cache[currentfactor].append(root)
            else:
                cache[currentfactor] = [root]

        for currentfactor, root in cache.items():
            _reals_cache[currentfactor] = root

        return reals

    @classmethod
    def _refine_imaginary(cls, complexes):
        sifted = sift(complexes, lambda c: c[1])
        complexes = []
        for f in ordered(sifted):
            nimag = _imag_count_of_factor(f)
            if nimag == 0:
                # refine until xbounds are neg or pos
                for u, f, k in sifted[f]:
                    while u.ax*u.bx <= 0:
                        u = u._inner_refine()
                    complexes.append((u, f, k))
            else:
                # refine until all but nimag xbounds are neg or pos
                potential_imag = list(range(len(sifted[f])))
                while True:
                    assert len(potential_imag) > 1
                    for i in list(potential_imag):
                        u, f, k = sifted[f][i]
                        if u.ax*u.bx > 0:
                            potential_imag.remove(i)
                        elif u.ax != u.bx:
                            u = u._inner_refine()
                            sifted[f][i] = u, f, k
                    if len(potential_imag) == nimag:
                        break
                complexes.extend(sifted[f])
        return complexes

    @classmethod
    def _refine_complexes(cls, complexes):
        """return complexes such that no bounding rectangles of non-conjugate
        roots would intersect. In addition, assure that neither ay nor by is
        0 to guarantee that non-real roots are distinct from real roots in
        terms of the y-bounds.
        """
        # get the intervals pairwise-disjoint.
        # If rectangles were drawn around the coordinates of the bounding
        # rectangles, no rectangles would intersect after this procedure.
        for i, (u, f, k) in enumerate(complexes):
            for j, (v, g, m) in enumerate(complexes[i + 1:]):
                u, v = u.refine_disjoint(v)
                complexes[i + j + 1] = (v, g, m)

            complexes[i] = (u, f, k)

        # refine until the x-bounds are unambiguously positive or negative
        # for non-imaginary roots
        complexes = cls._refine_imaginary(complexes)

        # make sure that all y bounds are off the real axis
        # and on the same side of the axis
        for i, (u, f, k) in enumerate(complexes):
            while u.ay*u.by <= 0:
                u = u.refine()
            complexes[i] = u, f, k
        return complexes

    @classmethod
    def _complexes_sorted(cls, complexes):
        """Make complex isolating intervals disjoint and sort roots. """
        complexes = cls._refine_complexes(complexes)
        # XXX don't sort until you are sure that it is compatible
        # with the indexing method but assert that the desired state
        # is not broken
        C, F = 0, 1  # location of ComplexInterval and factor
        fs = {i[F] for i in complexes}
        for i in range(1, len(complexes)):
            if complexes[i][F] != complexes[i - 1][F]:
                # if this fails the factors of a root were not
                # contiguous because a discontinuity should only
                # happen once
                fs.remove(complexes[i - 1][F])
        for i, cmplx in enumerate(complexes):
            # negative im part (conj=True) comes before
            # positive im part (conj=False)
            assert cmplx[C].conj is (i % 2 == 0)

        # update cache
        cache = {}
        # -- collate
        for root, currentfactor, _ in complexes:
            cache.setdefault(currentfactor, []).append(root)
        # -- store
        for currentfactor, root in cache.items():
            _complexes_cache[currentfactor] = root

        return complexes

    @classmethod
    def _reals_index(cls, reals, index):
        """
        Map initial real root index to an index in a factor where
        the root belongs.
        """
        i = 0

        for j, (_, currentfactor, k) in enumerate(reals):
            if index < i + k:
                poly, index = currentfactor, 0

                for _, currentfactor, _ in reals[:j]:
                    if currentfactor == poly:
                        index += 1

                return poly, index
            else:
                i += k

    @classmethod
    def _complexes_index(cls, complexes, index):
        """
        Map initial complex root index to an index in a factor where
        the root belongs.
        """
        i = 0
        for j, (_, currentfactor, k) in enumerate(complexes):
            if index < i + k:
                poly, index = currentfactor, 0

                for _, currentfactor, _ in complexes[:j]:
                    if currentfactor == poly:
                        index += 1

                index += len(_reals_cache[poly])

                return poly, index
            else:
                i += k

    @classmethod
    def _count_roots(cls, roots):
        """Count the number of real or complex roots with multiplicities."""
        return sum(k for _, _, k in roots)

    @classmethod
    def _indexed_root(cls, poly, index, lazy=False):
        """Get a root of a composite polynomial by index. """
        factors = _pure_factors(poly)

        # If the given poly is already irreducible, then the index does not
        # need to be adjusted, and we can postpone the heavy lifting of
        # computing and refining isolating intervals until that is needed.
        # Note, however, that `_pure_factors()` extracts a negative leading
        # coeff if present, so `factors[0][0]` may differ from `poly`, and
        # is the "normalized" version of `poly` that we must return.
        if lazy and len(factors) == 1 and factors[0][1] == 1:
            return factors[0][0], index

        reals = cls._get_reals(factors)
        reals_count = cls._count_roots(reals)

        if index < reals_count:
            return cls._reals_index(reals, index)
        else:
            complexes = cls._get_complexes(factors)
            return cls._complexes_index(complexes, index - reals_count)

    def _ensure_reals_init(self):
        """Ensure that our poly has entries in the reals cache. """
        if self.poly not in _reals_cache:
            self._indexed_root(self.poly, self.index)

    def _ensure_complexes_init(self):
        """Ensure that our poly has entries in the complexes cache. """
        if self.poly not in _complexes_cache:
            self._indexed_root(self.poly, self.index)

    @classmethod
    def _real_roots(cls, poly):
        """Get real roots of a composite polynomial. """
        factors = _pure_factors(poly)

        reals = cls._get_reals(factors)
        reals_count = cls._count_roots(reals)

        roots = []

        for index in range(0, reals_count):
            roots.append(cls._reals_index(reals, index))

        return roots

    def _reset(self):
        """
        Reset all intervals
        """
        self._all_roots(self.poly, use_cache=False)

    @classmethod
    def _all_roots(cls, poly, use_cache=True):
        """Get real and complex roots of a composite polynomial. """
        factors = _pure_factors(poly)

        reals = cls._get_reals(factors, use_cache=use_cache)
        reals_count = cls._count_roots(reals)

        roots = []

        for index in range(0, reals_count):
            roots.append(cls._reals_index(reals, index))

        complexes = cls._get_complexes(factors, use_cache=use_cache)
        complexes_count = cls._count_roots(complexes)

        for index in range(0, complexes_count):
            roots.append(cls._complexes_index(complexes, index))

        return roots

    @classmethod
    @cacheit
    def _roots_trivial(cls, poly, radicals):
        """Compute roots in linear, quadratic and binomial cases. """
        if poly.degree() == 1:
            return roots_linear(poly)

        if not radicals:
            return None

        if poly.degree() == 2:
            return roots_quadratic(poly)
        elif poly.length() == 2 and poly.TC():
            return roots_binomial(poly)
        else:
            return None

    @classmethod
    def _preprocess_roots(cls, poly):
        """Take heroic measures to make ``poly`` compatible with ``CRootOf``."""
        dom = poly.get_domain()

        if not dom.is_Exact:
            poly = poly.to_exact()

        coeff, poly = preprocess_roots(poly)
        dom = poly.get_domain()

        if not dom.is_ZZ:
            raise NotImplementedError(
                "sorted roots not supported over %s" % dom)

        return coeff, poly

    @classmethod
    def _postprocess_root(cls, root, radicals):
        """Return the root if it is trivial or a ``CRootOf`` object. """
        poly, index = root
        roots = cls._roots_trivial(poly, radicals)

        if roots is not None:
            return roots[index]
        else:
            return cls._new(poly, index)

    @classmethod
    def _get_roots(cls, method, poly, radicals):
        """Return postprocessed roots of specified kind. """
        if not poly.is_univariate:
            raise PolynomialError("only univariate polynomials are allowed")

        dom = poly.get_domain()

        # get rid of gen and it's free symbol
        d = Dummy()
        poly = poly.subs(poly.gen, d)
        x = symbols('x')
        # see what others are left and select x or a numbered x
        # that doesn't clash
        free_names = {str(i) for i in poly.free_symbols}
        for x in chain((symbols('x'),), numbered_symbols('x')):
            if x.name not in free_names:
                poly = poly.replace(d, x)
                break

        if dom.is_QQ or dom.is_ZZ:
            return cls._get_roots_qq(method, poly, radicals)
        elif dom.is_AlgebraicField or dom.is_ZZ_I or dom.is_QQ_I:
            return cls._get_roots_alg(method, poly, radicals)
        else:
            # XXX: not sure how to handle ZZ[x] which appears in some tests?
            # this makes the tests pass alright but has to be a better way?
            return cls._get_roots_qq(method, poly, radicals)


    @classmethod
    def _get_roots_qq(cls, method, poly, radicals):
        """Return postprocessed roots of specified kind
         for polynomials with rational coefficients. """
        coeff, poly = cls._preprocess_roots(poly)
        roots = []

        for root in getattr(cls, method)(poly):
            roots.append(coeff*cls._postprocess_root(root, radicals))

        return roots

    @classmethod
    def _get_roots_alg(cls, method, poly, radicals):
        """Return postprocessed roots of specified kind
         for polynomials with algebraic coefficients. It assumes
         the domain is already an algebraic field. First it
         finds the roots using _get_roots_qq, then uses the
         square-free factors to filter roots and get the correct
         multiplicity.
         """

        # Existing QQ code can find and sort the roots
        roots = cls._get_roots_qq(method, poly.lift(), radicals)

        subroots = {}
        for f, m in poly.sqf_list()[1]:
            if method == "_real_roots":
                roots_filt = f.which_real_roots(roots)
            elif method == "_all_roots":
                roots_filt = f.which_all_roots(roots)
            for r in roots_filt:
                subroots[r] = m

        roots_seen = set()
        roots_flat = []
        for r in roots:
            if r in subroots and r not in roots_seen:
                m = subroots[r]
                roots_flat.extend([r] * m)
                roots_seen.add(r)

        return roots_flat

    @classmethod
    def clear_cache(cls):
        """Reset cache for reals and complexes.

        The intervals used to approximate a root instance are updated
        as needed. When a request is made to see the intervals, the
        most current values are shown. `clear_cache` will reset all
        CRootOf instances back to their original state.

        See Also
        ========

        _reset
        """
        global _reals_cache, _complexes_cache
        _reals_cache = _pure_key_dict()
        _complexes_cache = _pure_key_dict()

    def _get_interval(self):
        """Internal function for retrieving isolation interval from cache. """
        self._ensure_reals_init()
        if self.is_real:
            return _reals_cache[self.poly][self.index]
        else:
            reals_count = len(_reals_cache[self.poly])
            self._ensure_complexes_init()
            return _complexes_cache[self.poly][self.index - reals_count]

    def _set_interval(self, interval):
        """Internal function for updating isolation interval in cache. """
        self._ensure_reals_init()
        if self.is_real:
            _reals_cache[self.poly][self.index] = interval
        else:
            reals_count = len(_reals_cache[self.poly])
            self._ensure_complexes_init()
            _complexes_cache[self.poly][self.index - reals_count] = interval

    def _eval_subs(self, old, new):
        # don't allow subs to change anything
        return self

    def _eval_conjugate(self):
        if self.is_real:
            return self
        expr, i = self.args
        return self.func(expr, i + (1 if self._get_interval().conj else -1))

    def eval_approx(self, n, return_mpmath=False):
        """Evaluate this complex root to the given precision.

        This uses secant method and root bounds are used to both
        generate an initial guess and to check that the root
        returned is valid. If ever the method converges outside the
        root bounds, the bounds will be made smaller and updated.
        """
        prec = dps_to_prec(n)
        with workprec(prec):
            g = self.poly.gen
            if not g.is_Symbol:
                d = Dummy('x')
                if self.is_imaginary:
                    d *= I
                func = lambdify(d, self.expr.subs(g, d))
            else:
                expr = self.expr
                if self.is_imaginary:
                    expr = self.expr.subs(g, I*g)
                func = lambdify(g, expr)

            interval = self._get_interval()
            while True:
                if self.is_real:
                    a = mpf(str(interval.a))
                    b = mpf(str(interval.b))
                    if a == b:
                        root = a
                        break
                    x0 = mpf(str(interval.center))
                    x1 = x0 + mpf(str(interval.dx))/4
                elif self.is_imaginary:
                    a = mpf(str(interval.ay))
                    b = mpf(str(interval.by))
                    if a == b:
                        root = mpc(mpf('0'), a)
                        break
                    x0 = mpf(str(interval.center[1]))
                    x1 = x0 + mpf(str(interval.dy))/4
                else:
                    ax = mpf(str(interval.ax))
                    bx = mpf(str(interval.bx))
                    ay = mpf(str(interval.ay))
                    by = mpf(str(interval.by))
                    if ax == bx and ay == by:
                        root = mpc(ax, ay)
                        break
                    x0 = mpc(*map(str, interval.center))
                    x1 = x0 + mpc(*map(str, (interval.dx, interval.dy)))/4
                try:
                    # without a tolerance, this will return when (to within
                    # the given precision) x_i == x_{i-1}
                    root = findroot(func, (x0, x1))
                    # If the (real or complex) root is not in the 'interval',
                    # then keep refining the interval. This happens if findroot
                    # accidentally finds a different root outside of this
                    # interval because our initial estimate 'x0' was not close
                    # enough. It is also possible that the secant method will
                    # get trapped by a max/min in the interval; the root
                    # verification by findroot will raise a ValueError in this
                    # case and the interval will then be tightened -- and
                    # eventually the root will be found.
                    #
                    # It is also possible that findroot will not have any
                    # successful iterations to process (in which case it
                    # will fail to initialize a variable that is tested
                    # after the iterations and raise an UnboundLocalError).
                    if self.is_real or self.is_imaginary:
                        if not bool(root.imag) == self.is_real and (
                                a <= root <= b):
                            if self.is_imaginary:
                                root = mpc(mpf('0'), root.real)
                            break
                    elif (ax <= root.real <= bx and ay <= root.imag <= by):
                        break
                except (UnboundLocalError, ValueError):
                    pass
                interval = interval.refine()

        # update the interval so we at least (for this precision or
        # less) don't have much work to do to recompute the root
        self._set_interval(interval)
        if return_mpmath:
            return root
        return (Float._new(root.real._mpf_, prec) +
            I*Float._new(root.imag._mpf_, prec))

    def _eval_evalf(self, prec, **kwargs):
        """Evaluate this complex root to the given precision."""
        # all kwargs are ignored
        return self.eval_rational(n=prec_to_dps(prec))._evalf(prec)

    def eval_rational(self, dx=None, dy=None, n=15):
        """
        Return a Rational approximation of ``self`` that has real
        and imaginary component approximations that are within ``dx``
        and ``dy`` of the true values, respectively. Alternatively,
        ``n`` digits of precision can be specified.

        The interval is refined with bisection and is sure to
        converge. The root bounds are updated when the refinement
        is complete so recalculation at the same or lesser precision
        will not have to repeat the refinement and should be much
        faster.

        The following example first obtains Rational approximation to
        1e-8 accuracy for all roots of the 4-th order Legendre
        polynomial. Since the roots are all less than 1, this will
        ensure the decimal representation of the approximation will be
        correct (including rounding) to 6 digits:

        >>> from sympy import legendre_poly, Symbol
        >>> x = Symbol("x")
        >>> p = legendre_poly(4, x, polys=True)
        >>> r = p.real_roots()[-1]
        >>> r.eval_rational(10**-8).n(6)
        0.861136

        It is not necessary to a two-step calculation, however: the
        decimal representation can be computed directly:

        >>> r.evalf(17)
        0.86113631159405258

        """
        dy = dy or dx
        if dx:
            rtol = None
            dx = dx if isinstance(dx, Rational) else Rational(str(dx))
            dy = dy if isinstance(dy, Rational) else Rational(str(dy))
        else:
            # 5 binary (or 2 decimal) digits are needed to ensure that
            # a given digit is correctly rounded
            # prec_to_dps(dps_to_prec(n) + 5) - n <= 2 (tested for
            # n in range(1000000)
            rtol = S(10)**-(n + 2)  # +2 for guard digits
        interval = self._get_interval()
        while True:
            if self.is_real:
                if rtol:
                    dx = abs(interval.center*rtol)
                interval = interval.refine_size(dx=dx)
                c = interval.center
                real = Rational(c)
                imag = S.Zero
                if not rtol or interval.dx < abs(c*rtol):
                    break
            elif self.is_imaginary:
                if rtol:
                    dy = abs(interval.center[1]*rtol)
                    dx = 1
                interval = interval.refine_size(dx=dx, dy=dy)
                c = interval.center[1]
                imag = Rational(c)
                real = S.Zero
                if not rtol or interval.dy < abs(c*rtol):
                    break
            else:
                if rtol:
                    dx = abs(interval.center[0]*rtol)
                    dy = abs(interval.center[1]*rtol)
                interval = interval.refine_size(dx, dy)
                c = interval.center
                real, imag = map(Rational, c)
                if not rtol or (
                        interval.dx < abs(c[0]*rtol) and
                        interval.dy < abs(c[1]*rtol)):
                    break

        # update the interval so we at least (for this precision or
        # less) don't have much work to do to recompute the root
        self._set_interval(interval)
        return real + I*imag


CRootOf = ComplexRootOf


@dispatch(ComplexRootOf, ComplexRootOf)
def _eval_is_eq(lhs, rhs): # noqa:F811
    # if we use is_eq to check here, we get infinite recursion
    return lhs == rhs


@dispatch(ComplexRootOf, Basic)  # type:ignore
def _eval_is_eq(lhs, rhs): # noqa:F811
    # CRootOf represents a Root, so if rhs is that root, it should set
    # the expression to zero *and* it should be in the interval of the
    # CRootOf instance. It must also be a number that agrees with the
    # is_real value of the CRootOf instance.
    if not rhs.is_number:
        return None
    if not rhs.is_finite:
        return False
    z = lhs.expr.subs(lhs.expr.free_symbols.pop(), rhs).is_zero
    if z is False:  # all roots will make z True but we don't know
        # whether this is the right root if z is True
        return False
    o = rhs.is_real, rhs.is_imaginary
    s = lhs.is_real, lhs.is_imaginary
    assert None not in s  # this is part of initial refinement
    if o != s and None not in o:
        return False
    re, im = rhs.as_real_imag()
    if lhs.is_real:
        if im:
            return False
        i = lhs._get_interval()
        a, b = [Rational(str(_)) for _ in (i.a, i.b)]
        return sympify(a <= rhs and rhs <= b)
    i = lhs._get_interval()
    r1, r2, i1, i2 = [Rational(str(j)) for j in (
        i.ax, i.bx, i.ay, i.by)]
    return is_le(r1, re) and is_le(re,r2) and is_le(i1,im) and is_le(im,i2)


@public
class RootSum(Expr):
    """Represents a sum of all roots of a univariate polynomial. """

    __slots__ = ('poly', 'fun', 'auto')

    def __new__(cls, expr, func=None, x=None, auto=True, quadratic=False):
        """Construct a new ``RootSum`` instance of roots of a polynomial."""
        coeff, poly = cls._transform(expr, x)

        if not poly.is_univariate:
            raise MultivariatePolynomialError(
                "only univariate polynomials are allowed")

        if func is None:
            func = Lambda(poly.gen, poly.gen)
        else:
            is_func = getattr(func, 'is_Function', False)

            if is_func and 1 in func.nargs:
                if not isinstance(func, Lambda):
                    func = Lambda(poly.gen, func(poly.gen))
            else:
                raise ValueError(
                    "expected a univariate function, got %s" % func)

        var, expr = func.variables[0], func.expr

        if coeff is not S.One:
            expr = expr.subs(var, coeff*var)

        deg = poly.degree()

        if not expr.has(var):
            return deg*expr

        if expr.is_Add:
            add_const, expr = expr.as_independent(var)
        else:
            add_const = S.Zero

        if expr.is_Mul:
            mul_const, expr = expr.as_independent(var)
        else:
            mul_const = S.One

        func = Lambda(var, expr)

        rational = cls._is_func_rational(poly, func)
        factors, terms = _pure_factors(poly), []

        for poly, k in factors:
            if poly.is_linear:
                term = func(roots_linear(poly)[0])
            elif quadratic and poly.is_quadratic:
                term = sum(map(func, roots_quadratic(poly)))
            else:
                if not rational or not auto:
                    term = cls._new(poly, func, auto)
                else:
                    term = cls._rational_case(poly, func)

            terms.append(k*term)

        return mul_const*Add(*terms) + deg*add_const

    @classmethod
    def _new(cls, poly, func, auto=True):
        """Construct new raw ``RootSum`` instance. """
        obj = Expr.__new__(cls)

        obj.poly = poly
        obj.fun = func
        obj.auto = auto

        return obj

    @classmethod
    def new(cls, poly, func, auto=True):
        """Construct new ``RootSum`` instance. """
        if not func.expr.has(*func.variables):
            return func.expr

        rational = cls._is_func_rational(poly, func)

        if not rational or not auto:
            return cls._new(poly, func, auto)
        else:
            return cls._rational_case(poly, func)

    @classmethod
    def _transform(cls, expr, x):
        """Transform an expression to a polynomial. """
        poly = PurePoly(expr, x, greedy=False)
        return preprocess_roots(poly)

    @classmethod
    def _is_func_rational(cls, poly, func):
        """Check if a lambda is a rational function. """
        var, expr = func.variables[0], func.expr
        return expr.is_rational_function(var)

    @classmethod
    def _rational_case(cls, poly, func):
        """Handle the rational function case. """
        roots = symbols('r:%d' % poly.degree())
        var, expr = func.variables[0], func.expr

        f = sum(expr.subs(var, r) for r in roots)
        p, q = together(f).as_numer_denom()

        domain = QQ[roots]

        p = p.expand()
        q = q.expand()

        try:
            p = Poly(p, domain=domain, expand=False)
        except GeneratorsNeeded:
            p, p_coeff = None, (p,)
        else:
            p_monom, p_coeff = zip(*p.terms())

        try:
            q = Poly(q, domain=domain, expand=False)
        except GeneratorsNeeded:
            q, q_coeff = None, (q,)
        else:
            q_monom, q_coeff = zip(*q.terms())

        coeffs, mapping = symmetrize(p_coeff + q_coeff, formal=True)
        formulas, values = viete(poly, roots), []

        for (sym, _), (_, val) in zip(mapping, formulas):
            values.append((sym, val))

        for i, (coeff, _) in enumerate(coeffs):
            coeffs[i] = coeff.subs(values)

        n = len(p_coeff)

        p_coeff = coeffs[:n]
        q_coeff = coeffs[n:]

        if p is not None:
            p = Poly(dict(zip(p_monom, p_coeff)), *p.gens).as_expr()
        else:
            (p,) = p_coeff

        if q is not None:
            q = Poly(dict(zip(q_monom, q_coeff)), *q.gens).as_expr()
        else:
            (q,) = q_coeff

        return factor(p/q)

    def _hashable_content(self):
        return (self.poly, self.fun)

    @property
    def expr(self):
        return self.poly.as_expr()

    @property
    def args(self):
        return (self.expr, self.fun, self.poly.gen)

    @property
    def free_symbols(self):
        return self.poly.free_symbols | self.fun.free_symbols

    @property
    def is_commutative(self):
        return True

    def doit(self, **hints):
        if not hints.get('roots', True):
            return self

        _roots = roots(self.poly, multiple=True)

        if len(_roots) < self.poly.degree():
            return self
        else:
            return Add(*[self.fun(r) for r in _roots])

    def _eval_evalf(self, prec):
        try:
            _roots = self.poly.nroots(n=prec_to_dps(prec))
        except (DomainError, PolynomialError):
            return self
        else:
            return Add(*[self.fun(r) for r in _roots])

    def _eval_derivative(self, x):
        var, expr = self.fun.args
        func = Lambda(var, expr.diff(x))
        return self.new(self.poly, func, self.auto)
<|endoftext|>"""Real and complex root isolation and refinement algorithms. """


from sympy.polys.densearith import (
    dup_neg, dup_rshift, dup_rem,
    dup_l2_norm_squared)
from sympy.polys.densebasic import (
    dup_LC, dup_TC, dup_degree,
    dup_strip, dup_reverse,
    dup_convert,
    dup_terms_gcd)
from sympy.polys.densetools import (
    dup_clear_denoms,
    dup_mirror, dup_scale, dup_shift,
    dup_transform,
    dup_diff,
    dup_eval, dmp_eval_in,
    dup_sign_variations,
    dup_real_imag)
from sympy.polys.euclidtools import (
    dup_discriminant)
from sympy.polys.factortools import (
    dup_factor_list)
from sympy.polys.polyerrors import (
    RefinementFailed,
    DomainError,
    PolynomialError)
from sympy.polys.sqfreetools import (
    dup_sqf_part, dup_sqf_list)


def dup_sturm(f, K):
    """
    Computes the Sturm sequence of ``f`` in ``F[x]``.

    Given a univariate, square-free polynomial ``f(x)`` returns the
    associated Sturm sequence ``f_0(x), ..., f_n(x)`` defined by::

       f_0(x), f_1(x) = f(x), f'(x)
       f_n = -rem(f_{n-2}(x), f_{n-1}(x))

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> R, x = ring("x", QQ)

    >>> R.dup_sturm(x**3 - 2*x**2 + x - 3)
    [x**3 - 2*x**2 + x - 3, 3*x**2 - 4*x + 1, 2/9*x + 25/9, -2079/4]

    References
    ==========

    .. [1] [Davenport88]_

    """
    if not K.is_Field:
        raise DomainError("Cannot compute Sturm sequence over %s" % K)

    f = dup_sqf_part(f, K)

    sturm = [f, dup_diff(f, 1, K)]

    while sturm[-1]:
        s = dup_rem(sturm[-2], sturm[-1], K)
        sturm.append(dup_neg(s, K))

    return sturm[:-1]

def dup_root_upper_bound(f, K):
    """Compute the LMQ upper bound for the positive roots of `f`;
       LMQ (Local Max Quadratic) was developed by Akritas-Strzebonski-Vigklas.

    References
    ==========
    .. [1] Alkiviadis G. Akritas: "Linear and Quadratic Complexity Bounds on the
        Values of the Positive Roots of Polynomials"
        Journal of Universal Computer Science, Vol. 15, No. 3, 523-537, 2009.
    """
    n, P = len(f), []
    t = n * [K.one]
    if dup_LC(f, K) < 0:
        f = dup_neg(f, K)
    f = list(reversed(f))

    for i in range(0, n):
        if f[i] >= 0:
            continue

        a, QL = K.log(-f[i], 2), []

        for j in range(i + 1, n):

            if f[j] <= 0:
                continue

            q = t[j] + a - K.log(f[j], 2)
            QL.append([q // (j - i), j])

        if not QL:
            continue

        q = min(QL)

        t[q[1]] = t[q[1]] + 1

        P.append(q[0])

    if not P:
        return None
    else:
        return K.get_field()(2)**(max(P) + 1)

def dup_root_lower_bound(f, K):
    """Compute the LMQ lower bound for the positive roots of `f`;
       LMQ (Local Max Quadratic) was developed by Akritas-Strzebonski-Vigklas.

       References
       ==========
       .. [1] Alkiviadis G. Akritas: "Linear and Quadratic Complexity Bounds on the
              Values of the Positive Roots of Polynomials"
              Journal of Universal Computer Science, Vol. 15, No. 3, 523-537, 2009.
    """
    bound = dup_root_upper_bound(dup_reverse(f), K)

    if bound is not None:
        return 1/bound
    else:
        return None

def dup_cauchy_upper_bound(f, K):
    """
    Compute the Cauchy upper bound on the absolute value of all roots of f,
    real or complex.

    References
    ==========
    .. [1] https://en.wikipedia.org/wiki/Geometrical_properties_of_polynomial_roots#Lagrange's_and_Cauchy's_bounds
    """
    n = dup_degree(f)
    if n < 1:
        raise PolynomialError('Polynomial has no roots.')

    if K.is_ZZ:
        L = K.get_field()
        f, K = dup_convert(f, K, L), L
    elif not K.is_QQ or K.is_RR or K.is_CC:
        # We need to compute absolute value, and we are not supporting cases
        # where this would take us outside the domain (or its quotient field).
        raise DomainError('Cauchy bound not supported over %s' % K)
    else:
        f = f[:]

    while K.is_zero(f[-1]):
        f.pop()
    if len(f) == 1:
        # Monomial. All roots are zero.
        return K.zero

    lc = f[0]
    return K.one + max(abs(n / lc) for n in f[1:])

def dup_cauchy_lower_bound(f, K):
    """Compute the Cauchy lower bound on the absolute value of all non-zero
       roots of f, real or complex."""
    g = dup_reverse(f)
    if len(g) < 2:
        raise PolynomialError('Polynomial has no non-zero roots.')
    if K.is_ZZ:
        K = K.get_field()
    b = dup_cauchy_upper_bound(g, K)
    return K.one / b

def dup_mignotte_sep_bound_squared(f, K):
    """
    Return the square of the Mignotte lower bound on separation between
    distinct roots of f. The square is returned so that the bound lies in
    K or its quotient field.

    References
    ==========

    .. [1] Mignotte, Maurice. "Some useful bounds." Computer algebra.
        Springer, Vienna, 1982. 259-263.
        https://people.dm.unipi.it/gianni/AC-EAG/Mignotte.pdf
    """
    n = dup_degree(f)
    if n < 2:
        raise PolynomialError('Polynomials of degree < 2 have no distinct roots.')

    if K.is_ZZ:
        L = K.get_field()
        f, K = dup_convert(f, K, L), L
    elif not K.is_QQ or K.is_RR or K.is_CC:
        # We need to compute absolute value, and we are not supporting cases
        # where this would take us outside the domain (or its quotient field).
        raise DomainError('Mignotte bound not supported over %s' % K)

    D = dup_discriminant(f, K)
    l2sq = dup_l2_norm_squared(f, K)
    return K(3)*K.abs(D) / ( K(n)**(n+1) * l2sq**(n-1) )

def _mobius_from_interval(I, field):
    """Convert an open interval to a Mobius transform. """
    s, t = I

    a, c = field.numer(s), field.denom(s)
    b, d = field.numer(t), field.denom(t)

    return a, b, c, d

def _mobius_to_interval(M, field):
    """Convert a Mobius transform to an open interval. """
    a, b, c, d = M

    s, t = field(a, c), field(b, d)

    if s <= t:
        return (s, t)
    else:
        return (t, s)

def dup_step_refine_real_root(f, M, K, fast=False):
    """One step of positive real root refinement algorithm. """
    a, b, c, d = M

    if a == b and c == d:
        return f, (a, b, c, d)

    A = dup_root_lower_bound(f, K)

    if A is not None:
        A = K(int(A))
    else:
        A = K.zero

    if fast and A > 16:
        f = dup_scale(f, A, K)
        a, c, A = A*a, A*c, K.one

    if A >= K.one:
        f = dup_shift(f, A, K)
        b, d = A*a + b, A*c + d

        if not dup_eval(f, K.zero, K):
            return f, (b, b, d, d)

    f, g = dup_shift(f, K.one, K), f

    a1, b1, c1, d1 = a, a + b, c, c + d

    if not dup_eval(f, K.zero, K):
        return f, (b1, b1, d1, d1)

    k = dup_sign_variations(f, K)

    if k == 1:
        a, b, c, d = a1, b1, c1, d1
    else:
        f = dup_shift(dup_reverse(g), K.one, K)

        if not dup_eval(f, K.zero, K):
            f = dup_rshift(f, 1, K)

        a, b, c, d = b, a + b, d, c + d

    return f, (a, b, c, d)

def dup_inner_refine_real_root(f, M, K, eps=None, steps=None, disjoint=None, fast=False, mobius=False):
    """Refine a positive root of `f` given a Mobius transform or an interval. """
    F = K.get_field()

    if len(M) == 2:
        a, b, c, d = _mobius_from_interval(M, F)
    else:
        a, b, c, d = M

    while not c:
        f, (a, b, c, d) = dup_step_refine_real_root(f, (a, b, c,
            d), K, fast=fast)

    if eps is not None and steps is not None:
        for i in range(0, steps):
            if abs(F(a, c) - F(b, d)) >= eps:
                f, (a, b, c, d) = dup_step_refine_real_root(f, (a, b, c, d), K, fast=fast)
            else:
                break
    else:
        if eps is not None:
            while abs(F(a, c) - F(b, d)) >= eps:
                f, (a, b, c, d) = dup_step_refine_real_root(f, (a, b, c, d), K, fast=fast)

        if steps is not None:
            for i in range(0, steps):
                f, (a, b, c, d) = dup_step_refine_real_root(f, (a, b, c, d), K, fast=fast)

    if disjoint is not None:
        while True:
            u, v = _mobius_to_interval((a, b, c, d), F)

            if v <= disjoint or disjoint <= u:
                break
            else:
                f, (a, b, c, d) = dup_step_refine_real_root(f, (a, b, c, d), K, fast=fast)

    if not mobius:
        return _mobius_to_interval((a, b, c, d), F)
    else:
        return f, (a, b, c, d)

def dup_outer_refine_real_root(f, s, t, K, eps=None, steps=None, disjoint=None, fast=False):
    """Refine a positive root of `f` given an interval `(s, t)`. """
    a, b, c, d = _mobius_from_interval((s, t), K.get_field())

    f = dup_transform(f, dup_strip([a, b]),
                         dup_strip([c, d]), K)

    if dup_sign_variations(f, K) != 1:
        raise RefinementFailed("there should be exactly one root in (%s, %s) interval" % (s, t))

    return dup_inner_refine_real_root(f, (a, b, c, d), K, eps=eps, steps=steps, disjoint=disjoint, fast=fast)

def dup_refine_real_root(f, s, t, K, eps=None, steps=None, disjoint=None, fast=False):
    """Refine real root's approximating interval to the given precision. """
    if K.is_QQ:
        (_, f), K = dup_clear_denoms(f, K, convert=True), K.get_ring()
    elif not K.is_ZZ:
        raise DomainError("real root refinement not supported over %s" % K)

    if s == t:
        return (s, t)

    if s > t:
        s, t = t, s

    negative = False

    if s < 0:
        if t <= 0:
            f, s, t, negative = dup_mirror(f, K), -t, -s, True
        else:
            raise ValueError("Cannot refine a real root in (%s, %s)" % (s, t))

    if negative and disjoint is not None:
        if disjoint < 0:
            disjoint = -disjoint
        else:
            disjoint = None

    s, t = dup_outer_refine_real_root(
        f, s, t, K, eps=eps, steps=steps, disjoint=disjoint, fast=fast)

    if negative:
        return (-t, -s)
    else:
        return ( s, t)

def dup_inner_isolate_real_roots(f, K, eps=None, fast=False):
    """Internal function for isolation positive roots up to given precision.

       References
       ==========
           1. Alkiviadis G. Akritas and Adam W. Strzebonski: A Comparative Study of Two Real Root
           Isolation Methods . Nonlinear Analysis: Modelling and Control, Vol. 10, No. 4, 297-304, 2005.
           2. Alkiviadis G. Akritas, Adam W. Strzebonski and Panagiotis S. Vigklas: Improving the
           Performance of the Continued Fractions Method Using new Bounds of Positive Roots. Nonlinear
           Analysis: Modelling and Control, Vol. 13, No. 3, 265-279, 2008.
    """
    a, b, c, d = K.one, K.zero, K.zero, K.one

    k = dup_sign_variations(f, K)

    if k == 0:
        return []
    if k == 1:
        roots = [dup_inner_refine_real_root(
            f, (a, b, c, d), K, eps=eps, fast=fast, mobius=True)]
    else:
        roots, stack = [], [(a, b, c, d, f, k)]

        while stack:
            a, b, c, d, f, k = stack.pop()

            A = dup_root_lower_bound(f, K)

            if A is not None:
                A = K(int(A))
            else:
                A = K.zero

            if fast and A > 16:
                f = dup_scale(f, A, K)
                a, c, A = A*a, A*c, K.one

            if A >= K.one:
                f = dup_shift(f, A, K)
                b, d = A*a + b, A*c + d

                if not dup_TC(f, K):
                    roots.append((f, (b, b, d, d)))
                    f = dup_rshift(f, 1, K)

                k = dup_sign_variations(f, K)

                if k == 0:
                    continue
                if k == 1:
                    roots.append(dup_inner_refine_real_root(
                        f, (a, b, c, d), K, eps=eps, fast=fast, mobius=True))
                    continue

            f1 = dup_shift(f, K.one, K)

            a1, b1, c1, d1, r = a, a + b, c, c + d, 0

            if not dup_TC(f1, K):
                roots.append((f1, (b1, b1, d1, d1)))
                f1, r = dup_rshift(f1, 1, K), 1

            k1 = dup_sign_variations(f1, K)
            k2 = k - k1 - r

            a2, b2, c2, d2 = b, a + b, d, c + d

            if k2 > 1:
                f2 = dup_shift(dup_reverse(f), K.one, K)

                if not dup_TC(f2, K):
                    f2 = dup_rshift(f2, 1, K)

                k2 = dup_sign_variations(f2, K)
            else:
                f2 = None

            if k1 < k2:
                a1, a2, b1, b2 = a2, a1, b2, b1
                c1, c2, d1, d2 = c2, c1, d2, d1
                f1, f2, k1, k2 = f2, f1, k2, k1

            if not k1:
                continue

            if f1 is None:
                f1 = dup_shift(dup_reverse(f), K.one, K)

                if not dup_TC(f1, K):
                    f1 = dup_rshift(f1, 1, K)

            if k1 == 1:
                roots.append(dup_inner_refine_real_root(
                    f1, (a1, b1, c1, d1), K, eps=eps, fast=fast, mobius=True))
            else:
                stack.append((a1, b1, c1, d1, f1, k1))

            if not k2:
                continue

            if f2 is None:
                f2 = dup_shift(dup_reverse(f), K.one, K)

                if not dup_TC(f2, K):
                    f2 = dup_rshift(f2, 1, K)

            if k2 == 1:
                roots.append(dup_inner_refine_real_root(
                    f2, (a2, b2, c2, d2), K, eps=eps, fast=fast, mobius=True))
            else:
                stack.append((a2, b2, c2, d2, f2, k2))

    return roots

def _discard_if_outside_interval(f, M, inf, sup, K, negative, fast, mobius):
    """Discard an isolating interval if outside ``(inf, sup)``. """
    F = K.get_field()

    while True:
        u, v = _mobius_to_interval(M, F)

        if negative:
            u, v = -v, -u

        if (inf is None or u >= inf) and (sup is None or v <= sup):
            if not mobius:
                return u, v
            else:
                return f, M
        elif (sup is not None and u > sup) or (inf is not None and v < inf):
            return None
        else:
            f, M = dup_step_refine_real_root(f, M, K, fast=fast)

def dup_inner_isolate_positive_roots(f, K, eps=None, inf=None, sup=None, fast=False, mobius=False):
    """Iteratively compute disjoint positive root isolation intervals. """
    if sup is not None and sup < 0:
        return []

    roots = dup_inner_isolate_real_roots(f, K, eps=eps, fast=fast)

    F, results = K.get_field(), []

    if inf is not None or sup is not None:
        for f, M in roots:
            result = _discard_if_outside_interval(f, M, inf, sup, K, False, fast, mobius)

            if result is not None:
                results.append(result)
    elif not mobius:
        results.extend(_mobius_to_interval(M, F) for _, M in roots)
    else:
        results = roots

    return results

def dup_inner_isolate_negative_roots(f, K, inf=None, sup=None, eps=None, fast=False, mobius=False):
    """Iteratively compute disjoint negative root isolation intervals. """
    if inf is not None and inf >= 0:
        return []

    roots = dup_inner_isolate_real_roots(dup_mirror(f, K), K, eps=eps, fast=fast)

    F, results = K.get_field(), []

    if inf is not None or sup is not None:
        for f, M in roots:
            result = _discard_if_outside_interval(f, M, inf, sup, K, True, fast, mobius)

            if result is not None:
                results.append(result)
    elif not mobius:
        for f, M in roots:
            u, v = _mobius_to_interval(M, F)
            results.append((-v, -u))
    else:
        results = roots

    return results

def _isolate_zero(f, K, inf, sup, basis=False, sqf=False):
    """Handle special case of CF algorithm when ``f`` is homogeneous. """
    j, f = dup_terms_gcd(f, K)

    if j > 0:
        F = K.get_field()

        if (inf is None or inf <= 0) and (sup is None or 0 <= sup):
            if not sqf:
                if not basis:
                    return [((F.zero, F.zero), j)], f
                else:
                    return [((F.zero, F.zero), j, [K.one, K.zero])], f
            else:
                return [(F.zero, F.zero)], f

    return [], f

def dup_isolate_real_roots_sqf(f, K, eps=None, inf=None, sup=None, fast=False, blackbox=False):
    """Isolate real roots of a square-free polynomial using the Vincent-Akritas-Strzebonski (VAS) CF approach.

       References
       ==========
       .. [1] Alkiviadis G. Akritas and Adam W. Strzebonski: A Comparative
              Study of Two Real Root Isolation Methods. Nonlinear Analysis:
              Modelling and Control, Vol. 10, No. 4, 297-304, 2005.
       .. [2] Alkiviadis G. Akritas, Adam W. Strzebonski and Panagiotis S.
              Vigklas: Improving the Performance of the Continued Fractions
              Method Using New Bounds of Positive Roots. Nonlinear Analysis:
              Modelling and Control, Vol. 13, No. 3, 265-279, 2008.

    """
    if K.is_QQ:
        (_, f), K = dup_clear_denoms(f, K, convert=True), K.get_ring()
    elif not K.is_ZZ:
        raise DomainError("isolation of real roots not supported over %s" % K)

    if dup_degree(f) <= 0:
        return []

    I_zero, f = _isolate_zero(f, K, inf, sup, basis=False, sqf=True)

    I_neg = dup_inner_isolate_negative_roots(f, K, eps=eps, inf=inf, sup=sup, fast=fast)
    I_pos = dup_inner_isolate_positive_roots(f, K, eps=eps, inf=inf, sup=sup, fast=fast)

    roots = sorted(I_neg + I_zero + I_pos)

    if not blackbox:
        return roots
    else:
        return [ RealInterval((a, b), f, K) for (a, b) in roots ]

def dup_isolate_real_roots(f, K, eps=None, inf=None, sup=None, basis=False, fast=False):
    """Isolate real roots using Vincent-Akritas-Strzebonski (VAS) continued fractions approach.

       References
       ==========

       .. [1] Alkiviadis G. Akritas and Adam W. Strzebonski: A Comparative
              Study of Two Real Root Isolation Methods. Nonlinear Analysis:
              Modelling and Control, Vol. 10, No. 4, 297-304, 2005.
       .. [2] Alkiviadis G. Akritas, Adam W. Strzebonski and Panagiotis S.
              Vigklas: Improving the Performance of the Continued Fractions
              Method Using New Bounds of Positive Roots.
              Nonlinear Analysis: Modelling and Control, Vol. 13, No. 3, 265-279, 2008.

    """
    if K.is_QQ:
        (_, f), K = dup_clear_denoms(f, K, convert=True), K.get_ring()
    elif not K.is_ZZ:
        raise DomainError("isolation of real roots not supported over %s" % K)

    if dup_degree(f) <= 0:
        return []

    I_zero, f = _isolate_zero(f, K, inf, sup, basis=basis, sqf=False)

    _, factors = dup_sqf_list(f, K)

    if len(factors) == 1:
        ((f, k),) = factors

        I_neg = dup_inner_isolate_negative_roots(f, K, eps=eps, inf=inf, sup=sup, fast=fast)
        I_pos = dup_inner_isolate_positive_roots(f, K, eps=eps, inf=inf, sup=sup, fast=fast)

        I_neg = [ ((u, v), k) for u, v in I_neg ]
        I_pos = [ ((u, v), k) for u, v in I_pos ]
    else:
        I_neg, I_pos = _real_isolate_and_disjoin(factors, K,
            eps=eps, inf=inf, sup=sup, basis=basis, fast=fast)

    return sorted(I_neg + I_zero + I_pos)

def dup_isolate_real_roots_list(polys, K, eps=None, inf=None, sup=None, strict=False, basis=False, fast=False):
    """Isolate real roots of a list of polynomial using Vincent-Akritas-Strzebonski (VAS) CF approach.

       References
       ==========

       .. [1] Alkiviadis G. Akritas and Adam W. Strzebonski: A Comparative
              Study of Two Real Root Isolation Methods. Nonlinear Analysis:
              Modelling and Control, Vol. 10, No. 4, 297-304, 2005.
       .. [2] Alkiviadis G. Akritas, Adam W. Strzebonski and Panagiotis S.
              Vigklas: Improving the Performance of the Continued Fractions
              Method Using New Bounds of Positive Roots.
              Nonlinear Analysis: Modelling and Control, Vol. 13, No. 3, 265-279, 2008.

    """
    if K.is_QQ:
        K, F, polys = K.get_ring(), K, polys[:]

        for i, p in enumerate(polys):
            polys[i] = dup_clear_denoms(p, F, K, convert=True)[1]
    elif not K.is_ZZ:
        raise DomainError("isolation of real roots not supported over %s" % K)

    zeros, factors_dict = False, {}

    if (inf is None or inf <= 0) and (sup is None or 0 <= sup):
        zeros, zero_indices = True, {}

    for i, p in enumerate(polys):
        j, p = dup_terms_gcd(p, K)

        if zeros and j > 0:
            zero_indices[i] = j

        for f, k in dup_factor_list(p, K)[1]:
            f = tuple(f)

            if f not in factors_dict:
                factors_dict[f] = {i: k}
            else:
                factors_dict[f][i] = k

    factors_list = [(list(f), indices) for f, indices in factors_dict.items()]
    I_neg, I_pos = _real_isolate_and_disjoin(factors_list, K, eps=eps,
        inf=inf, sup=sup, strict=strict, basis=basis, fast=fast)

    F = K.get_field()

    if not zeros or not zero_indices:
        I_zero = []
    else:
        if not basis:
            I_zero = [((F.zero, F.zero), zero_indices)]
        else:
            I_zero = [((F.zero, F.zero), zero_indices, [K.one, K.zero])]

    return sorted(I_neg + I_zero + I_pos)

def _disjoint_p(M, N, strict=False):
    """Check if Mobius transforms define disjoint intervals. """
    a1, b1, c1, d1 = M
    a2, b2, c2, d2 = N

    a1d1, b1c1 = a1*d1, b1*c1
    a2d2, b2c2 = a2*d2, b2*c2

    if a1d1 == b1c1 and a2d2 == b2c2:
        return True

    if a1d1 > b1c1:
        a1, c1, b1, d1 = b1, d1, a1, c1

    if a2d2 > b2c2:
        a2, c2, b2, d2 = b2, d2, a2, c2

    if not strict:
        return a2*d1 >= c2*b1 or b2*c1 <= d2*a1
    else:
        return a2*d1 > c2*b1 or b2*c1 < d2*a1

def _real_isolate_and_disjoin(factors, K, eps=None, inf=None, sup=None, strict=False, basis=False, fast=False):
    """Isolate real roots of a list of polynomials and disjoin intervals. """
    I_pos, I_neg = [], []

    for i, (f, k) in enumerate(factors):
        for F, M in dup_inner_isolate_positive_roots(f, K, eps=eps, inf=inf, sup=sup, fast=fast, mobius=True):
            I_pos.append((F, M, k, f))

        for G, N in dup_inner_isolate_negative_roots(f, K, eps=eps, inf=inf, sup=sup, fast=fast, mobius=True):
            I_neg.append((G, N, k, f))

    for i, (f, M, k, F) in enumerate(I_pos):
        for j, (g, N, m, G) in enumerate(I_pos[i + 1:]):
            while not _disjoint_p(M, N, strict=strict):
                f, M = dup_inner_refine_real_root(f, M, K, steps=1, fast=fast, mobius=True)
                g, N = dup_inner_refine_real_root(g, N, K, steps=1, fast=fast, mobius=True)

            I_pos[i + j + 1] = (g, N, m, G)

        I_pos[i] = (f, M, k, F)

    for i, (f, M, k, F) in enumerate(I_neg):
        for j, (g, N, m, G) in enumerate(I_neg[i + 1:]):
            while not _disjoint_p(M, N, strict=strict):
                f, M = dup_inner_refine_real_root(f, M, K, steps=1, fast=fast, mobius=True)
                g, N = dup_inner_refine_real_root(g, N, K, steps=1, fast=fast, mobius=True)

            I_neg[i + j + 1] = (g, N, m, G)

        I_neg[i] = (f, M, k, F)

    if strict:
        for i, (f, M, k, F) in enumerate(I_neg):
            if not M[0]:
                while not M[0]:
                    f, M = dup_inner_refine_real_root(f, M, K, steps=1, fast=fast, mobius=True)

                I_neg[i] = (f, M, k, F)
                break

        for j, (g, N, m, G) in enumerate(I_pos):
            if not N[0]:
                while not N[0]:
                    g, N = dup_inner_refine_real_root(g, N, K, steps=1, fast=fast, mobius=True)

                I_pos[j] = (g, N, m, G)
                break

    field = K.get_field()

    I_neg = [ (_mobius_to_interval(M, field), k, f) for (_, M, k, f) in I_neg ]
    I_pos = [ (_mobius_to_interval(M, field), k, f) for (_, M, k, f) in I_pos ]

    I_neg = [((-v, -u), k, f) for ((u, v), k, f) in I_neg]

    if not basis:
        I_neg = [((u, v), k) for ((u, v), k, _) in I_neg]
        I_pos = [((u, v), k) for ((u, v), k, _) in I_pos]

    return I_neg, I_pos

def dup_count_real_roots(f, K, inf=None, sup=None):
    """Returns the number of distinct real roots of ``f`` in ``[inf, sup]``. """
    if dup_degree(f) <= 0:
        return 0

    if not K.is_Field:
        R, K = K, K.get_field()
        f = dup_convert(f, R, K)

    sturm = dup_sturm(f, K)

    if inf is None:
        signs_inf = dup_sign_variations([ dup_LC(s, K)*(-1)**dup_degree(s) for s in sturm ], K)
    else:
        signs_inf = dup_sign_variations([ dup_eval(s, inf, K) for s in sturm ], K)

    if sup is None:
        signs_sup = dup_sign_variations([ dup_LC(s, K) for s in sturm ], K)
    else:
        signs_sup = dup_sign_variations([ dup_eval(s, sup, K) for s in sturm ], K)

    count = abs(signs_inf - signs_sup)

    if inf is not None and not dup_eval(f, inf, K):
        count += 1

    return count

OO = 'OO'  # Origin of (re, im) coordinate system

Q1 = 'Q1'  # Quadrant #1 (++): re > 0 and im > 0
Q2 = 'Q2'  # Quadrant #2 (-+): re < 0 and im > 0
Q3 = 'Q3'  # Quadrant #3 (--): re < 0 and im < 0
Q4 = 'Q4'  # Quadrant #4 (+-): re > 0 and im < 0

A1 = 'A1'  # Axis #1 (+0): re > 0 and im = 0
A2 = 'A2'  # Axis #2 (0+): re = 0 and im > 0
A3 = 'A3'  # Axis #3 (-0): re < 0 and im = 0
A4 = 'A4'  # Axis #4 (0-): re = 0 and im < 0

_rules_simple = {
    # Q --> Q (same) => no change
    (Q1, Q1): 0,
    (Q2, Q2): 0,
    (Q3, Q3): 0,
    (Q4, Q4): 0,

    # A -- CCW --> Q => +1/4 (CCW)
    (A1, Q1): 1,
    (A2, Q2): 1,
    (A3, Q3): 1,
    (A4, Q4): 1,

    # A --  CW --> Q => -1/4 (CCW)
    (A1, Q4): 2,
    (A2, Q1): 2,
    (A3, Q2): 2,
    (A4, Q3): 2,

    # Q -- CCW --> A => +1/4 (CCW)
    (Q1, A2): 3,
    (Q2, A3): 3,
    (Q3, A4): 3,
    (Q4, A1): 3,

    # Q --  CW --> A => -1/4 (CCW)
    (Q1, A1): 4,
    (Q2, A2): 4,
    (Q3, A3): 4,
    (Q4, A4): 4,

    # Q -- CCW --> Q => +1/2 (CCW)
    (Q1, Q2): +5,
    (Q2, Q3): +5,
    (Q3, Q4): +5,
    (Q4, Q1): +5,

    # Q --  CW --> Q => -1/2 (CW)
    (Q1, Q4): -5,
    (Q2, Q1): -5,
    (Q3, Q2): -5,
    (Q4, Q3): -5,
}

_rules_ambiguous = {
    # A -- CCW --> Q => { +1/4 (CCW), -9/4 (CW) }
    (A1, OO, Q1): -1,
    (A2, OO, Q2): -1,
    (A3, OO, Q3): -1,
    (A4, OO, Q4): -1,

    # A --  CW --> Q => { -1/4 (CCW), +7/4 (CW) }
    (A1, OO, Q4): -2,
    (A2, OO, Q1): -2,
    (A3, OO, Q2): -2,
    (A4, OO, Q3): -2,

    # Q -- CCW --> A => { +1/4 (CCW), -9/4 (CW) }
    (Q1, OO, A2): -3,
    (Q2, OO, A3): -3,
    (Q3, OO, A4): -3,
    (Q4, OO, A1): -3,

    # Q --  CW --> A => { -1/4 (CCW), +7/4 (CW) }
    (Q1, OO, A1): -4,
    (Q2, OO, A2): -4,
    (Q3, OO, A3): -4,
    (Q4, OO, A4): -4,

    # A --  OO --> A => { +1 (CCW), -1 (CW) }
    (A1, A3): 7,
    (A2, A4): 7,
    (A3, A1): 7,
    (A4, A2): 7,

    (A1, OO, A3): 7,
    (A2, OO, A4): 7,
    (A3, OO, A1): 7,
    (A4, OO, A2): 7,

    # Q -- DIA --> Q => { +1 (CCW), -1 (CW) }
    (Q1, Q3): 8,
    (Q2, Q4): 8,
    (Q3, Q1): 8,
    (Q4, Q2): 8,

    (Q1, OO, Q3): 8,
    (Q2, OO, Q4): 8,
    (Q3, OO, Q1): 8,
    (Q4, OO, Q2): 8,

    # A --- R ---> A => { +1/2 (CCW), -3/2 (CW) }
    (A1, A2): 9,
    (A2, A3): 9,
    (A3, A4): 9,
    (A4, A1): 9,

    (A1, OO, A2): 9,
    (A2, OO, A3): 9,
    (A3, OO, A4): 9,
    (A4, OO, A1): 9,

    # A --- L ---> A => { +3/2 (CCW), -1/2 (CW) }
    (A1, A4): 10,
    (A2, A1): 10,
    (A3, A2): 10,
    (A4, A3): 10,

    (A1, OO, A4): 10,
    (A2, OO, A1): 10,
    (A3, OO, A2): 10,
    (A4, OO, A3): 10,

    # Q --- 1 ---> A => { +3/4 (CCW), -5/4 (CW) }
    (Q1, A3): 11,
    (Q2, A4): 11,
    (Q3, A1): 11,
    (Q4, A2): 11,

    (Q1, OO, A3): 11,
    (Q2, OO, A4): 11,
    (Q3, OO, A1): 11,
    (Q4, OO, A2): 11,

    # Q --- 2 ---> A => { +5/4 (CCW), -3/4 (CW) }
    (Q1, A4): 12,
    (Q2, A1): 12,
    (Q3, A2): 12,
    (Q4, A3): 12,

    (Q1, OO, A4): 12,
    (Q2, OO, A1): 12,
    (Q3, OO, A2): 12,
    (Q4, OO, A3): 12,

    # A --- 1 ---> Q => { +5/4 (CCW), -3/4 (CW) }
    (A1, Q3): 13,
    (A2, Q4): 13,
    (A3, Q1): 13,
    (A4, Q2): 13,

    (A1, OO, Q3): 13,
    (A2, OO, Q4): 13,
    (A3, OO, Q1): 13,
    (A4, OO, Q2): 13,

    # A --- 2 ---> Q => { +3/4 (CCW), -5/4 (CW) }
    (A1, Q2): 14,
    (A2, Q3): 14,
    (A3, Q4): 14,
    (A4, Q1): 14,

    (A1, OO, Q2): 14,
    (A2, OO, Q3): 14,
    (A3, OO, Q4): 14,
    (A4, OO, Q1): 14,

    # Q --> OO --> Q => { +1/2 (CCW), -3/2 (CW) }
    (Q1, OO, Q2): 15,
    (Q2, OO, Q3): 15,
    (Q3, OO, Q4): 15,
    (Q4, OO, Q1): 15,

    # Q --> OO --> Q => { +3/2 (CCW), -1/2 (CW) }
    (Q1, OO, Q4): 16,
    (Q2, OO, Q1): 16,
    (Q3, OO, Q2): 16,
    (Q4, OO, Q3): 16,

    # A --> OO --> A => { +2 (CCW), 0 (CW) }
    (A1, OO, A1): 17,
    (A2, OO, A2): 17,
    (A3, OO, A3): 17,
    (A4, OO, A4): 17,

    # Q --> OO --> Q => { +2 (CCW), 0 (CW) }
    (Q1, OO, Q1): 18,
    (Q2, OO, Q2): 18,
    (Q3, OO, Q3): 18,
    (Q4, OO, Q4): 18,
}

_values = {
    0: [( 0, 1)],
    1: [(+1, 4)],
    2: [(-1, 4)],
    3: [(+1, 4)],
    4: [(-1, 4)],
    -1: [(+9, 4), (+1, 4)],
    -2: [(+7, 4), (-1, 4)],
    -3: [(+9, 4), (+1, 4)],
    -4: [(+7, 4), (-1, 4)],
    +5: [(+1, 2)],
    -5: [(-1, 2)],
    7: [(+1, 1), (-1, 1)],
    8: [(+1, 1), (-1, 1)],
    9: [(+1, 2), (-3, 2)],
    10: [(+3, 2), (-1, 2)],
    11: [(+3, 4), (-5, 4)],
    12: [(+5, 4), (-3, 4)],
    13: [(+5, 4), (-3, 4)],
    14: [(+3, 4), (-5, 4)],
    15: [(+1, 2), (-3, 2)],
    16: [(+3, 2), (-1, 2)],
    17: [(+2, 1), ( 0, 1)],
    18: [(+2, 1), ( 0, 1)],
}

def _classify_point(re, im):
    """Return the half-axis (or origin) on which (re, im) point is located. """
    if not re and not im:
        return OO

    if not re:
        if im > 0:
            return A2
        else:
            return A4
    elif not im:
        if re > 0:
            return A1
        else:
            return A3

def _intervals_to_quadrants(intervals, f1, f2, s, t, F):
    """Generate a sequence of extended quadrants from a list of critical points. """
    if not intervals:
        return []

    Q = []

    if not f1:
        (a, b), _, _ = intervals[0]

        if a == b == s:
            if len(intervals) == 1:
                if dup_eval(f2, t, F) > 0:
                    return [OO, A2]
                else:
                    return [OO, A4]
            else:
                (a, _), _, _ = intervals[1]

                if dup_eval(f2, (s + a)/2, F) > 0:
                    Q.extend([OO, A2])
                    f2_sgn = +1
                else:
                    Q.extend([OO, A4])
                    f2_sgn = -1

                intervals = intervals[1:]
        else:
            if dup_eval(f2, s, F) > 0:
                Q.append(A2)
                f2_sgn = +1
            else:
                Q.append(A4)
                f2_sgn = -1

        for (a, _), indices, _ in intervals:
            Q.append(OO)

            if indices[1] % 2 == 1:
                f2_sgn = -f2_sgn

            if a != t:
                if f2_sgn > 0:
                    Q.append(A2)
                else:
                    Q.append(A4)

        return Q

    if not f2:
        (a, b), _, _ = intervals[0]

        if a == b == s:
            if len(intervals) == 1:
                if dup_eval(f1, t, F) > 0:
                    return [OO, A1]
                else:
                    return [OO, A3]
            else:
                (a, _), _, _ = intervals[1]

                if dup_eval(f1, (s + a)/2, F) > 0:
                    Q.extend([OO, A1])
                    f1_sgn = +1
                else:
                    Q.extend([OO, A3])
                    f1_sgn = -1

                intervals = intervals[1:]
        else:
            if dup_eval(f1, s, F) > 0:
                Q.append(A1)
                f1_sgn = +1
            else:
                Q.append(A3)
                f1_sgn = -1

        for (a, _), indices, _ in intervals:
            Q.append(OO)

            if indices[0] % 2 == 1:
                f1_sgn = -f1_sgn

            if a != t:
                if f1_sgn > 0:
                    Q.append(A1)
                else:
                    Q.append(A3)

        return Q

    re = dup_eval(f1, s, F)
    im = dup_eval(f2, s, F)

    if not re or not im:
        Q.append(_classify_point(re, im))

        if len(intervals) == 1:
            re = dup_eval(f1, t, F)
            im = dup_eval(f2, t, F)
        else:
            (a, _), _, _ = intervals[1]

            re = dup_eval(f1, (s + a)/2, F)
            im = dup_eval(f2, (s + a)/2, F)

        intervals = intervals[1:]

    if re > 0:
        f1_sgn = +1
    else:
        f1_sgn = -1

    if im > 0:
        f2_sgn = +1
    else:
        f2_sgn = -1

    sgn = {
        (+1, +1): Q1,
        (-1, +1): Q2,
        (-1, -1): Q3,
        (+1, -1): Q4,
    }

    Q.append(sgn[(f1_sgn, f2_sgn)])

    for (a, b), indices, _ in intervals:
        if a == b:
            re = dup_eval(f1, a, F)
            im = dup_eval(f2, a, F)

            cls = _classify_point(re, im)

            if cls is not None:
                Q.append(cls)

        if 0 in indices:
            if indices[0] % 2 == 1:
                f1_sgn = -f1_sgn

        if 1 in indices:
            if indices[1] % 2 == 1:
                f2_sgn = -f2_sgn

        if not (a == b and b == t):
            Q.append(sgn[(f1_sgn, f2_sgn)])

    return Q

def _traverse_quadrants(Q_L1, Q_L2, Q_L3, Q_L4, exclude=None):
    """Transform sequences of quadrants to a sequence of rules. """
    if exclude is True:
        edges = [1, 1, 0, 0]

        corners = {
            (0, 1): 1,
            (1, 2): 1,
            (2, 3): 0,
            (3, 0): 1,
        }
    else:
        edges = [0, 0, 0, 0]

        corners = {
            (0, 1): 0,
            (1, 2): 0,
            (2, 3): 0,
            (3, 0): 0,
        }

    if exclude is not None and exclude is not True:
        exclude = set(exclude)

        for i, edge in enumerate(['S', 'E', 'N', 'W']):
            if edge in exclude:
                edges[i] = 1

        for i, corner in enumerate(['SW', 'SE', 'NE', 'NW']):
            if corner in exclude:
                corners[((i - 1) % 4, i)] = 1

    QQ, rules = [Q_L1, Q_L2, Q_L3, Q_L4], []

    for i, Q in enumerate(QQ):
        if not Q:
            continue

        if Q[-1] == OO:
            Q = Q[:-1]

        if Q[0] == OO:
            j, Q = (i - 1) % 4, Q[1:]
            qq = (QQ[j][-2], OO, Q[0])

            if qq in _rules_ambiguous:
                rules.append((_rules_ambiguous[qq], corners[(j, i)]))
            else:
                raise NotImplementedError("3 element rule (corner): " + str(qq))

        q1, k = Q[0], 1

        while k < len(Q):
            q2, k = Q[k], k + 1

            if q2 != OO:
                qq = (q1, q2)

                if qq in _rules_simple:
                    rules.append((_rules_simple[qq], 0))
                elif qq in _rules_ambiguous:
                    rules.append((_rules_ambiguous[qq], edges[i]))
                else:
                    raise NotImplementedError("2 element rule (inside): " + str(qq))
            else:
                qq, k = (q1, q2, Q[k]), k + 1

                if qq in _rules_ambiguous:
                    rules.append((_rules_ambiguous[qq], edges[i]))
                else:
                    raise NotImplementedError("3 element rule (edge): " + str(qq))

            q1 = qq[-1]

    return rules

def _reverse_intervals(intervals):
    """Reverse intervals for traversal from right to left and from top to bottom. """
    return [ ((b, a), indices, f) for (a, b), indices, f in reversed(intervals) ]

def _winding_number(T, field):
    """Compute the winding number of the input polynomial, i.e. the number of roots. """
    return int(sum(field(*_values[t][i]) for t, i in T) / field(2))

def dup_count_complex_roots(f, K, inf=None, sup=None, exclude=None):
    """Count all roots in [u + v*I, s + t*I] rectangle using Collins-Krandick algorithm. """
    if not K.is_ZZ and not K.is_QQ:
        raise DomainError("complex root counting is not supported over %s" % K)

    if K.is_ZZ:
        R, F = K, K.get_field()
    else:
        R, F = K.get_ring(), K

    f = dup_convert(f, K, F)

    if inf is None or sup is None:
        _, lc = dup_degree(f), abs(dup_LC(f, F))
        B = 2*max(F.quo(abs(c), lc) for c in f)

    if inf is None:
        (u, v) = (-B, -B)
    else:
        (u, v) = inf

    if sup is None:
        (s, t) = (+B, +B)
    else:
        (s, t) = sup

    f1, f2 = dup_real_imag(f, F)

    f1L1F = dmp_eval_in(f1, v, 1, 1, F)
    f2L1F = dmp_eval_in(f2, v, 1, 1, F)

    _, f1L1R = dup_clear_denoms(f1L1F, F, R, convert=True)
    _, f2L1R = dup_clear_denoms(f2L1F, F, R, convert=True)

    f1L2F = dmp_eval_in(f1, s, 0, 1, F)
    f2L2F = dmp_eval_in(f2, s, 0, 1, F)

    _, f1L2R = dup_clear_denoms(f1L2F, F, R, convert=True)
    _, f2L2R = dup_clear_denoms(f2L2F, F, R, convert=True)

    f1L3F = dmp_eval_in(f1, t, 1, 1, F)
    f2L3F = dmp_eval_in(f2, t, 1, 1, F)

    _, f1L3R = dup_clear_denoms(f1L3F, F, R, convert=True)
    _, f2L3R = dup_clear_denoms(f2L3F, F, R, convert=True)

    f1L4F = dmp_eval_in(f1, u, 0, 1, F)
    f2L4F = dmp_eval_in(f2, u, 0, 1, F)

    _, f1L4R = dup_clear_denoms(f1L4F, F, R, convert=True)
    _, f2L4R = dup_clear_denoms(f2L4F, F, R, convert=True)

    S_L1 = [f1L1R, f2L1R]
    S_L2 = [f1L2R, f2L2R]
    S_L3 = [f1L3R, f2L3R]
    S_L4 = [f1L4R, f2L4R]

    I_L1 = dup_isolate_real_roots_list(S_L1, R, inf=u, sup=s, fast=True, basis=True, strict=True)
    I_L2 = dup_isolate_real_roots_list(S_L2, R, inf=v, sup=t, fast=True, basis=True, strict=True)
    I_L3 = dup_isolate_real_roots_list(S_L3, R, inf=u, sup=s, fast=True, basis=True, strict=True)
    I_L4 = dup_isolate_real_roots_list(S_L4, R, inf=v, sup=t, fast=True, basis=True, strict=True)

    I_L3 = _reverse_intervals(I_L3)
    I_L4 = _reverse_intervals(I_L4)

    Q_L1 = _intervals_to_quadrants(I_L1, f1L1F, f2L1F, u, s, F)
    Q_L2 = _intervals_to_quadrants(I_L2, f1L2F, f2L2F, v, t, F)
    Q_L3 = _intervals_to_quadrants(I_L3, f1L3F, f2L3F, s, u, F)
    Q_L4 = _intervals_to_quadrants(I_L4, f1L4F, f2L4F, t, v, F)

    T = _traverse_quadrants(Q_L1, Q_L2, Q_L3, Q_L4, exclude=exclude)

    return _winding_number(T, F)

def _vertical_bisection(N, a, b, I, Q, F1, F2, f1, f2, F):
    """Vertical bisection step in Collins-Krandick root isolation algorithm. """
    (u, v), (s, t) = a, b

    I_L1, I_L2, I_L3, I_L4 = I
    Q_L1, Q_L2, Q_L3, Q_L4 = Q

    f1L1F, f1L2F, f1L3F, f1L4F = F1
    f2L1F, f2L2F, f2L3F, f2L4F = F2

    x = (u + s) / 2

    f1V = dmp_eval_in(f1, x, 0, 1, F)
    f2V = dmp_eval_in(f2, x, 0, 1, F)

    I_V = dup_isolate_real_roots_list([f1V, f2V], F, inf=v, sup=t, fast=True, strict=True, basis=True)

    I_L1_L, I_L1_R = [], []
    I_L2_L, I_L2_R = I_V, I_L2
    I_L3_L, I_L3_R = [], []
    I_L4_L, I_L4_R = I_L4, _reverse_intervals(I_V)

    for I in I_L1:
        (a, b), indices, h = I

        if a == b:
            if a == x:
                I_L1_L.append(I)
                I_L1_R.append(I)
            elif a < x:
                I_L1_L.append(I)
            else:
                I_L1_R.append(I)
        else:
            if b <= x:
                I_L1_L.append(I)
            elif a >= x:
                I_L1_R.append(I)
            else:
                a, b = dup_refine_real_root(h, a, b, F.get_ring(), disjoint=x, fast=True)

                if b <= x:
                    I_L1_L.append(((a, b), indices, h))
                if a >= x:
                    I_L1_R.append(((a, b), indices, h))

    for I in I_L3:
        (b, a), indices, h = I

        if a == b:
            if a == x:
                I_L3_L.append(I)
                I_L3_R.append(I)
            elif a < x:
                I_L3_L.append(I)
            else:
                I_L3_R.append(I)
        else:
            if b <= x:
                I_L3_L.append(I)
            elif a >= x:
                I_L3_R.append(I)
            else:
                a, b = dup_refine_real_root(h, a, b, F.get_ring(), disjoint=x, fast=True)

                if b <= x:
                    I_L3_L.append(((b, a), indices, h))
                if a >= x:
                    I_L3_R.append(((b, a), indices, h))

    Q_L1_L = _intervals_to_quadrants(I_L1_L, f1L1F, f2L1F, u, x, F)
    Q_L2_L = _intervals_to_quadrants(I_L2_L, f1V, f2V, v, t, F)
    Q_L3_L = _intervals_to_quadrants(I_L3_L, f1L3F, f2L3F, x, u, F)
    Q_L4_L = Q_L4

    Q_L1_R = _intervals_to_quadrants(I_L1_R, f1L1F, f2L1F, x, s, F)
    Q_L2_R = Q_L2
    Q_L3_R = _intervals_to_quadrants(I_L3_R, f1L3F, f2L3F, s, x, F)
    Q_L4_R = _intervals_to_quadrants(I_L4_R, f1V, f2V, t, v, F)

    T_L = _traverse_quadrants(Q_L1_L, Q_L2_L, Q_L3_L, Q_L4_L, exclude=True)
    T_R = _traverse_quadrants(Q_L1_R, Q_L2_R, Q_L3_R, Q_L4_R, exclude=True)

    N_L = _winding_number(T_L, F)
    N_R = _winding_number(T_R, F)

    I_L = (I_L1_L, I_L2_L, I_L3_L, I_L4_L)
    Q_L = (Q_L1_L, Q_L2_L, Q_L3_L, Q_L4_L)

    I_R = (I_L1_R, I_L2_R, I_L3_R, I_L4_R)
    Q_R = (Q_L1_R, Q_L2_R, Q_L3_R, Q_L4_R)

    F1_L = (f1L1F, f1V, f1L3F, f1L4F)
    F2_L = (f2L1F, f2V, f2L3F, f2L4F)

    F1_R = (f1L1F, f1L2F, f1L3F, f1V)
    F2_R = (f2L1F, f2L2F, f2L3F, f2V)

    a, b = (u, v), (x, t)
    c, d = (x, v), (s, t)

    D_L = (N_L, a, b, I_L, Q_L, F1_L, F2_L)
    D_R = (N_R, c, d, I_R, Q_R, F1_R, F2_R)

    return D_L, D_R

def _horizontal_bisection(N, a, b, I, Q, F1, F2, f1, f2, F):
    """Horizontal bisection step in Collins-Krandick root isolation algorithm. """
    (u, v), (s, t) = a, b

    I_L1, I_L2, I_L3, I_L4 = I
    Q_L1, Q_L2, Q_L3, Q_L4 = Q

    f1L1F, f1L2F, f1L3F, f1L4F = F1
    f2L1F, f2L2F, f2L3F, f2L4F = F2

    y = (v + t) / 2

    f1H = dmp_eval_in(f1, y, 1, 1, F)
    f2H = dmp_eval_in(f2, y, 1, 1, F)

    I_H = dup_isolate_real_roots_list([f1H, f2H], F, inf=u, sup=s, fast=True, strict=True, basis=True)

    I_L1_B, I_L1_U = I_L1, I_H
    I_L2_B, I_L2_U = [], []
    I_L3_B, I_L3_U = _reverse_intervals(I_H), I_L3
    I_L4_B, I_L4_U = [], []

    for I in I_L2:
        (a, b), indices, h = I

        if a == b:
            if a == y:
                I_L2_B.append(I)
                I_L2_U.append(I)
            elif a < y:
                I_L2_B.append(I)
            else:
                I_L2_U.append(I)
        else:
            if b <= y:
                I_L2_B.append(I)
            elif a >= y:
                I_L2_U.append(I)
            else:
                a, b = dup_refine_real_root(h, a, b, F.get_ring(), disjoint=y, fast=True)

                if b <= y:
                    I_L2_B.append(((a, b), indices, h))
                if a >= y:
                    I_L2_U.append(((a, b), indices, h))

    for I in I_L4:
        (b, a), indices, h = I

        if a == b:
            if a == y:
                I_L4_B.append(I)
                I_L4_U.append(I)
            elif a < y:
                I_L4_B.append(I)
            else:
                I_L4_U.append(I)
        else:
            if b <= y:
                I_L4_B.append(I)
            elif a >= y:
                I_L4_U.append(I)
            else:
                a, b = dup_refine_real_root(h, a, b, F.get_ring(), disjoint=y, fast=True)

                if b <= y:
                    I_L4_B.append(((b, a), indices, h))
                if a >= y:
                    I_L4_U.append(((b, a), indices, h))

    Q_L1_B = Q_L1
    Q_L2_B = _intervals_to_quadrants(I_L2_B, f1L2F, f2L2F, v, y, F)
    Q_L3_B = _intervals_to_quadrants(I_L3_B, f1H, f2H, s, u, F)
    Q_L4_B = _intervals_to_quadrants(I_L4_B, f1L4F, f2L4F, y, v, F)

    Q_L1_U = _intervals_to_quadrants(I_L1_U, f1H, f2H, u, s, F)
    Q_L2_U = _intervals_to_quadrants(I_L2_U, f1L2F, f2L2F, y, t, F)
    Q_L3_U = Q_L3
    Q_L4_U = _intervals_to_quadrants(I_L4_U, f1L4F, f2L4F, t, y, F)

    T_B = _traverse_quadrants(Q_L1_B, Q_L2_B, Q_L3_B, Q_L4_B, exclude=True)
    T_U = _traverse_quadrants(Q_L1_U, Q_L2_U, Q_L3_U, Q_L4_U, exclude=True)

    N_B = _winding_number(T_B, F)
    N_U = _winding_number(T_U, F)

    I_B = (I_L1_B, I_L2_B, I_L3_B, I_L4_B)
    Q_B = (Q_L1_B, Q_L2_B, Q_L3_B, Q_L4_B)

    I_U = (I_L1_U, I_L2_U, I_L3_U, I_L4_U)
    Q_U = (Q_L1_U, Q_L2_U, Q_L3_U, Q_L4_U)

    F1_B = (f1L1F, f1L2F, f1H, f1L4F)
    F2_B = (f2L1F, f2L2F, f2H, f2L4F)

    F1_U = (f1H, f1L2F, f1L3F, f1L4F)
    F2_U = (f2H, f2L2F, f2L3F, f2L4F)

    a, b = (u, v), (s, y)
    c, d = (u, y), (s, t)

    D_B = (N_B, a, b, I_B, Q_B, F1_B, F2_B)
    D_U = (N_U, c, d, I_U, Q_U, F1_U, F2_U)

    return D_B, D_U

def _depth_first_select(rectangles):
    """Find a rectangle of minimum area for bisection. """
    min_area, j = None, None

    for i, (_, (u, v), (s, t), _, _, _, _) in enumerate(rectangles):
        area = (s - u)*(t - v)

        if min_area is None or area < min_area:
            min_area, j = area, i

    return rectangles.pop(j)

def _rectangle_small_p(a, b, eps):
    """Return ``True`` if the given rectangle is small enough. """
    (u, v), (s, t) = a, b

    if eps is not None:
        return s - u < eps and t - v < eps
    else:
        return True

def dup_isolate_complex_roots_sqf(f, K, eps=None, inf=None, sup=None, blackbox=False):
    """Isolate complex roots of a square-free polynomial using Collins-Krandick algorithm. """
    if not K.is_ZZ and not K.is_QQ:
        raise DomainError("isolation of complex roots is not supported over %s" % K)

    if dup_degree(f) <= 0:
        return []

    if K.is_ZZ:
        F = K.get_field()
    else:
        F = K

    f = dup_convert(f, K, F)

    lc = abs(dup_LC(f, F))
    B = 2*max(F.quo(abs(c), lc) for c in f)

    (u, v), (s, t) = (-B, F.zero), (B, B)

    if inf is not None:
        u = inf

    if sup is not None:
        s = sup

    if v < 0 or t <= v or s <= u:
        raise ValueError("not a valid complex isolation rectangle")

    f1, f2 = dup_real_imag(f, F)

    f1L1 = dmp_eval_in(f1, v, 1, 1, F)
    f2L1 = dmp_eval_in(f2, v, 1, 1, F)

    f1L2 = dmp_eval_in(f1, s, 0, 1, F)
    f2L2 = dmp_eval_in(f2, s, 0, 1, F)

    f1L3 = dmp_eval_in(f1, t, 1, 1, F)
    f2L3 = dmp_eval_in(f2, t, 1, 1, F)

    f1L4 = dmp_eval_in(f1, u, 0, 1, F)
    f2L4 = dmp_eval_in(f2, u, 0, 1, F)

    S_L1 = [f1L1, f2L1]
    S_L2 = [f1L2, f2L2]
    S_L3 = [f1L3, f2L3]
    S_L4 = [f1L4, f2L4]

    I_L1 = dup_isolate_real_roots_list(S_L1, F, inf=u, sup=s, fast=True, strict=True, basis=True)
    I_L2 = dup_isolate_real_roots_list(S_L2, F, inf=v, sup=t, fast=True, strict=True, basis=True)
    I_L3 = dup_isolate_real_roots_list(S_L3, F, inf=u, sup=s, fast=True, strict=True, basis=True)
    I_L4 = dup_isolate_real_roots_list(S_L4, F, inf=v, sup=t, fast=True, strict=True, basis=True)

    I_L3 = _reverse_intervals(I_L3)
    I_L4 = _reverse_intervals(I_L4)

    Q_L1 = _intervals_to_quadrants(I_L1, f1L1, f2L1, u, s, F)
    Q_L2 = _intervals_to_quadrants(I_L2, f1L2, f2L2, v, t, F)
    Q_L3 = _intervals_to_quadrants(I_L3, f1L3, f2L3, s, u, F)
    Q_L4 = _intervals_to_quadrants(I_L4, f1L4, f2L4, t, v, F)

    T = _traverse_quadrants(Q_L1, Q_L2, Q_L3, Q_L4)
    N = _winding_number(T, F)

    if not N:
        return []

    I = (I_L1, I_L2, I_L3, I_L4)
    Q = (Q_L1, Q_L2, Q_L3, Q_L4)

    F1 = (f1L1, f1L2, f1L3, f1L4)
    F2 = (f2L1, f2L2, f2L3, f2L4)

    rectangles, roots = [(N, (u, v), (s, t), I, Q, F1, F2)], []

    while rectangles:
        N, (u, v), (s, t), I, Q, F1, F2 = _depth_first_select(rectangles)

        if s - u > t - v:
            D_L, D_R = _vertical_bisection(N, (u, v), (s, t), I, Q, F1, F2, f1, f2, F)

            N_L, a, b, I_L, Q_L, F1_L, F2_L = D_L
            N_R, c, d, I_R, Q_R, F1_R, F2_R = D_R

            if N_L >= 1:
                if N_L == 1 and _rectangle_small_p(a, b, eps):
                    roots.append(ComplexInterval(a, b, I_L, Q_L, F1_L, F2_L, f1, f2, F))
                else:
                    rectangles.append(D_L)

            if N_R >= 1:
                if N_R == 1 and _rectangle_small_p(c, d, eps):
                    roots.append(ComplexInterval(c, d, I_R, Q_R, F1_R, F2_R, f1, f2, F))
                else:
                    rectangles.append(D_R)
        else:
            D_B, D_U = _horizontal_bisection(N, (u, v), (s, t), I, Q, F1, F2, f1, f2, F)

            N_B, a, b, I_B, Q_B, F1_B, F2_B = D_B
            N_U, c, d, I_U, Q_U, F1_U, F2_U = D_U

            if N_B >= 1:
                if N_B == 1 and _rectangle_small_p(a, b, eps):
                    roots.append(ComplexInterval(
                        a, b, I_B, Q_B, F1_B, F2_B, f1, f2, F))
                else:
                    rectangles.append(D_B)

            if N_U >= 1:
                if N_U == 1 and _rectangle_small_p(c, d, eps):
                    roots.append(ComplexInterval(
                        c, d, I_U, Q_U, F1_U, F2_U, f1, f2, F))
                else:
                    rectangles.append(D_U)

    _roots, roots = sorted(roots, key=lambda r: (r.ax, r.ay)), []

    for root in _roots:
        roots.extend([root.conjugate(), root])

    if blackbox:
        return roots
    else:
        return [ r.as_tuple() for r in roots ]

def dup_isolate_all_roots_sqf(f, K, eps=None, inf=None, sup=None, fast=False, blackbox=False):
    """Isolate real and complex roots of a square-free polynomial ``f``. """
    return (
        dup_isolate_real_roots_sqf( f, K, eps=eps, inf=inf, sup=sup, fast=fast, blackbox=blackbox),
        dup_isolate_complex_roots_sqf(f, K, eps=eps, inf=inf, sup=sup, blackbox=blackbox))

def dup_isolate_all_roots(f, K, eps=None, inf=None, sup=None, fast=False):
    """Isolate real and complex roots of a non-square-free polynomial ``f``. """
    if not K.is_ZZ and not K.is_QQ:
        raise DomainError("isolation of real and complex roots is not supported over %s" % K)

    _, factors = dup_sqf_list(f, K)

    if len(factors) == 1:
        ((f, k),) = factors

        real_part, complex_part = dup_isolate_all_roots_sqf(
            f, K, eps=eps, inf=inf, sup=sup, fast=fast)

        real_part = [ ((a, b), k) for (a, b) in real_part ]
        complex_part = [ ((a, b), k) for (a, b) in complex_part ]

        return real_part, complex_part
    else:
        raise NotImplementedError( "only trivial square-free polynomials are supported")

class RealInterval:
    """A fully qualified representation of a real isolation interval. """

    def __init__(self, data, f, dom):
        """Initialize new real interval with complete information. """
        if len(data) == 2:
            s, t = data

            self.neg = False

            if s < 0:
                if t <= 0:
                    f, s, t, self.neg = dup_mirror(f, dom), -t, -s, True
                else:
                    raise ValueError("Cannot refine a real root in (%s, %s)" % (s, t))

            a, b, c, d = _mobius_from_interval((s, t), dom.get_field())

            f = dup_transform(f, dup_strip([a, b]),
                                 dup_strip([c, d]), dom)

            self.mobius = a, b, c, d
        else:
            self.mobius = data[:-1]
            self.neg = data[-1]

        self.f, self.dom = f, dom

    @property
    def func(self):
        return RealInterval

    @property
    def args(self):
        i = self
        return (i.mobius + (i.neg,), i.f, i.dom)

    def __eq__(self, other):
        if type(other) is not type(self):
            return False
        return self.args == other.args

    @property
    def a(self):
        """Return the position of the left end. """
        field = self.dom.get_field()
        a, b, c, d = self.mobius

        if not self.neg:
            if a*d < b*c:
                return field(a, c)
            return field(b, d)
        else:
            if a*d > b*c:
                return -field(a, c)
            return -field(b, d)

    @property
    def b(self):
        """Return the position of the right end. """
        was = self.neg
        self.neg = not was
        rv = -self.a
        self.neg = was
        return rv

    @property
    def dx(self):
        """Return width of the real isolating interval. """
        return self.b - self.a

    @property
    def center(self):
        """Return the center of the real isolating interval. """
        return (self.a + self.b)/2

    @property
    def max_denom(self):
        """Return the largest denominator occurring in either endpoint. """
        return max(self.a.denominator, self.b.denominator)

    def as_tuple(self):
        """Return tuple representation of real isolating interval. """
        return (self.a, self.b)

    def __repr__(self):
        return "(%s, %s)" % (self.a, self.b)

    def __contains__(self, item):
        """
        Say whether a complex number belongs to this real interval.

        Parameters
        ==========

        item : pair (re, im) or number re
            Either a pair giving the real and imaginary parts of the number,
            or else a real number.

        """
        if isinstance(item, tuple):
            re, im = item
        else:
            re, im = item, 0
        return im == 0 and self.a <= re <= self.b

    def is_disjoint(self, other):
        """Return ``True`` if two isolation intervals are disjoint. """
        if isinstance(other, RealInterval):
            return (self.b < other.a or other.b < self.a)
        assert isinstance(other, ComplexInterval)
        return (self.b < other.ax or other.bx < self.a
            or other.ay*other.by > 0)

    def _inner_refine(self):
        """Internal one step real root refinement procedure. """
        if self.mobius is None:
            return self

        f, mobius = dup_inner_refine_real_root(
            self.f, self.mobius, self.dom, steps=1, mobius=True)

        return RealInterval(mobius + (self.neg,), f, self.dom)

    def refine_disjoint(self, other):
        """Refine an isolating interval until it is disjoint with another one. """
        expr = self
        while not expr.is_disjoint(other):
            expr, other = expr._inner_refine(), other._inner_refine()

        return expr, other

    def refine_size(self, dx):
        """Refine an isolating interval until it is of sufficiently small size. """
        expr = self
        while not (expr.dx < dx):
            expr = expr._inner_refine()

        return expr

    def refine_step(self, steps=1):
        """Perform several steps of real root refinement algorithm. """
        expr = self
        for _ in range(steps):
            expr = expr._inner_refine()

        return expr

    def refine(self):
        """Perform one step of real root refinement algorithm. """
        return self._inner_refine()


class ComplexInterval:
    """A fully qualified representation of a complex isolation interval.
    The printed form is shown as (ax, bx) x (ay, by) where (ax, ay)
    and (bx, by) are the coordinates of the southwest and northeast
    corners of the interval's rectangle, respectively.

    Examples
    ========

    >>> from sympy import CRootOf, S
    >>> from sympy.abc import x
    >>> CRootOf.clear_cache()  # for doctest reproducibility
    >>> root = CRootOf(x**10 - 2*x + 3, 9)
    >>> i = root._get_interval(); i
    (3/64, 3/32) x (9/8, 75/64)

    The real part of the root lies within the range [0, 3/4] while
    the imaginary part lies within the range [9/8, 3/2]:

    >>> root.n(3)
    0.0766 + 1.14*I

    The width of the ranges in the x and y directions on the complex
    plane are:

    >>> i.dx, i.dy
    (3/64, 3/64)

    The center of the range is

    >>> i.center
    (9/128, 147/128)

    The northeast coordinate of the rectangle bounding the root in the
    complex plane is given by attribute b and the x and y components
    are accessed by bx and by:

    >>> i.b, i.bx, i.by
    ((3/32, 75/64), 3/32, 75/64)

    The southwest coordinate is similarly given by i.a

    >>> i.a, i.ax, i.ay
    ((3/64, 9/8), 3/64, 9/8)

    Although the interval prints to show only the real and imaginary
    range of the root, all the information of the underlying root
    is contained as properties of the interval.

    For example, an interval with a nonpositive imaginary range is
    considered to be the conjugate. Since the y values of y are in the
    range [0, 1/4] it is not the conjugate:

    >>> i.conj
    False

    The conjugate's interval is

    >>> ic = i.conjugate(); ic
    (3/64, 3/32) x (-75/64, -9/8)

        NOTE: the values printed still represent the x and y range
        in which the root -- conjugate, in this case -- is located,
        but the underlying a and b values of a root and its conjugate
        are the same:

        >>> assert i.a == ic.a and i.b == ic.b

        What changes are the reported coordinates of the bounding rectangle:

        >>> (i.ax, i.ay), (i.bx, i.by)
        ((3/64, 9/8), (3/32, 75/64))
        >>> (ic.ax, ic.ay), (ic.bx, ic.by)
        ((3/64, -75/64), (3/32, -9/8))

    The interval can be refined once:

    >>> i  # for reference, this is the current interval
    (3/64, 3/32) x (9/8, 75/64)

    >>> i.refine()
    (3/64, 3/32) x (9/8, 147/128)

    Several refinement steps can be taken:

    >>> i.refine_step(2)  # 2 steps
    (9/128, 3/32) x (9/8, 147/128)

    It is also possible to refine to a given tolerance:

    >>> tol = min(i.dx, i.dy)/2
    >>> i.refine_size(tol)
    (9/128, 21/256) x (9/8, 291/256)

    A disjoint interval is one whose bounding rectangle does not
    overlap with another. An interval, necessarily, is not disjoint with
    itself, but any interval is disjoint with a conjugate since the
    conjugate rectangle will always be in the lower half of the complex
    plane and the non-conjugate in the upper half:

    >>> i.is_disjoint(i), i.is_disjoint(i.conjugate())
    (False, True)

    The following interval j is not disjoint from i:

    >>> close = CRootOf(x**10 - 2*x + 300/S(101), 9)
    >>> j = close._get_interval(); j
    (75/1616, 75/808) x (225/202, 1875/1616)
    >>> i.is_disjoint(j)
    False

    The two can be made disjoint, however:

    >>> newi, newj = i.refine_disjoint(j)
    >>> newi
    (39/512, 159/2048) x (2325/2048, 4653/4096)
    >>> newj
    (3975/51712, 2025/25856) x (29325/25856, 117375/103424)

    Even though the real ranges overlap, the imaginary do not, so
    the roots have been resolved as distinct. Intervals are disjoint
    when either the real or imaginary component of the intervals is
    distinct. In the case above, the real components have not been
    resolved (so we do not know, yet, which root has the smaller real
    part) but the imaginary part of ``close`` is larger than ``root``:

    >>> close.n(3)
    0.0771 + 1.13*I
    >>> root.n(3)
    0.0766 + 1.14*I
    """

    def __init__(self, a, b, I, Q, F1, F2, f1, f2, dom, conj=False):
        """Initialize new complex interval with complete information. """
        # a and b are the SW and NE corner of the bounding interval,
        # (ax, ay) and (bx, by), respectively, for the NON-CONJUGATE
        # root (the one with the positive imaginary part); when working
        # with the conjugate, the a and b value are still non-negative
        # but the ay, by are reversed and have oppositite sign
        self.a, self.b = a, b
        self.I, self.Q = I, Q

        self.f1, self.F1 = f1, F1
        self.f2, self.F2 = f2, F2

        self.dom = dom
        self.conj = conj

    @property
    def func(self):
        return ComplexInterval

    @property
    def args(self):
        i = self
        return (i.a, i.b, i.I, i.Q, i.F1, i.F2, i.f1, i.f2, i.dom, i.conj)

    def __eq__(self, other):
        if type(other) is not type(self):
            return False
        return self.args == other.args

    @property
    def ax(self):
        """Return ``x`` coordinate of south-western corner. """
        return self.a[0]

    @property
    def ay(self):
        """Return ``y`` coordinate of south-western corner. """
        if not self.conj:
            return self.a[1]
        else:
            return -self.b[1]

    @property
    def bx(self):
        """Return ``x`` coordinate of north-eastern corner. """
        return self.b[0]

    @property
    def by(self):
        """Return ``y`` coordinate of north-eastern corner. """
        if not self.conj:
            return self.b[1]
        else:
            return -self.a[1]

    @property
    def dx(self):
        """Return width of the complex isolating interval. """
        return self.b[0] - self.a[0]

    @property
    def dy(self):
        """Return height of the complex isolating interval. """
        return self.b[1] - self.a[1]

    @property
    def center(self):
        """Return the center of the complex isolating interval. """
        return ((self.ax + self.bx)/2, (self.ay + self.by)/2)

    @property
    def max_denom(self):
        """Return the largest denominator occurring in either endpoint. """
        return max(self.ax.denominator, self.bx.denominator,
                   self.ay.denominator, self.by.denominator)

    def as_tuple(self):
        """Return tuple representation of the complex isolating
        interval's SW and NE corners, respectively. """
        return ((self.ax, self.ay), (self.bx, self.by))

    def __repr__(self):
        return "(%s, %s) x (%s, %s)" % (self.ax, self.bx, self.ay, self.by)

    def conjugate(self):
        """This complex interval really is located in lower half-plane. """
        return ComplexInterval(self.a, self.b, self.I, self.Q,
            self.F1, self.F2, self.f1, self.f2, self.dom, conj=True)

    def __contains__(self, item):
        """
        Say whether a complex number belongs to this complex rectangular
        region.

        Parameters
        ==========

        item : pair (re, im) or number re
            Either a pair giving the real and imaginary parts of the number,
            or else a real number.

        """
        if isinstance(item, tuple):
            re, im = item
        else:
            re, im = item, 0
        return self.ax <= re <= self.bx and self.ay <= im <= self.by

    def is_disjoint(self, other):
        """Return ``True`` if two isolation intervals are disjoint. """
        if isinstance(other, RealInterval):
            return other.is_disjoint(self)
        if self.conj != other.conj:  # above and below real axis
            return True
        re_distinct = (self.bx < other.ax or other.bx < self.ax)
        if re_distinct:
            return True
        im_distinct = (self.by < other.ay or other.by < self.ay)
        return im_distinct

    def _inner_refine(self):
        """Internal one step complex root refinement procedure. """
        (u, v), (s, t) = self.a, self.b

        I, Q = self.I, self.Q

        f1, F1 = self.f1, self.F1
        f2, F2 = self.f2, self.F2

        dom = self.dom

        if s - u > t - v:
            D_L, D_R = _vertical_bisection(1, (u, v), (s, t), I, Q, F1, F2, f1, f2, dom)

            if D_L[0] == 1:
                _, a, b, I, Q, F1, F2 = D_L
            else:
                _, a, b, I, Q, F1, F2 = D_R
        else:
            D_B, D_U = _horizontal_bisection(1, (u, v), (s, t), I, Q, F1, F2, f1, f2, dom)

            if D_B[0] == 1:
                _, a, b, I, Q, F1, F2 = D_B
            else:
                _, a, b, I, Q, F1, F2 = D_U

        return ComplexInterval(a, b, I, Q, F1, F2, f1, f2, dom, self.conj)

    def refine_disjoint(self, other):
        """Refine an isolating interval until it is disjoint with another one. """
        expr = self
        while not expr.is_disjoint(other):
            expr, other = expr._inner_refine(), other._inner_refine()

        return expr, other

    def refine_size(self, dx, dy=None):
        """Refine an isolating interval until it is of sufficiently small size. """
        if dy is None:
            dy = dx
        expr = self
        while not (expr.dx < dx and expr.dy < dy):
            expr = expr._inner_refine()

        return expr

    def refine_step(self, steps=1):
        """Perform several steps of complex root refinement algorithm. """
        expr = self
        for _ in range(steps):
            expr = expr._inner_refine()

        return expr

    def refine(self):
        """Perform one step of complex root refinement algorithm. """
        return self._inner_refine()
<|endoftext|>"""
This module contains functions for the computation
of Euclidean, (generalized) Sturmian, (modified) subresultant
polynomial remainder sequences (prs's) of two polynomials;
included are also three functions for the computation of the
resultant of two polynomials.

Except for the function res_z(), which computes the resultant
of two polynomials, the pseudo-remainder function prem()
of sympy is _not_ used by any of the functions in the module.

Instead of prem() we use the function

rem_z().

Included is also the function quo_z().

An explanation of why we avoid prem() can be found in the
references stated in the docstring of rem_z().

1. Theoretical background:
==========================
Consider the polynomials f, g in Z[x] of degrees deg(f) = n and
deg(g) = m with n >= m.

Definition 1:
=============
The sign sequence of a polynomial remainder sequence (prs) is the
sequence of signs of the leading coefficients of its polynomials.

Sign sequences can be computed with the function:

sign_seq(poly_seq, x)

Definition 2:
=============
A polynomial remainder sequence (prs) is called complete if the
degree difference between any two consecutive polynomials is 1;
otherwise, it called incomplete.

It is understood that f, g belong to the sequences mentioned in
the two definitions above.

1A. Euclidean and subresultant prs's:
=====================================
The subresultant prs of f, g is a sequence of polynomials in Z[x]
analogous to the Euclidean prs, the sequence obtained by applying
on f, g Euclid's algorithm for polynomial greatest common divisors
(gcd) in Q[x].

The subresultant prs differs from the Euclidean prs in that the
coefficients of each polynomial in the former sequence are determinants
--- also referred to as subresultants --- of appropriately selected
sub-matrices of sylvester1(f, g, x), Sylvester's matrix of 1840 of
dimensions (n + m) * (n + m).

Recall that the determinant of sylvester1(f, g, x) itself is
called the resultant of f, g and serves as a criterion of whether
the two polynomials have common roots or not.

In SymPy the resultant is computed with the function
resultant(f, g, x). This function does _not_ evaluate the
determinant of sylvester(f, g, x, 1); instead, it returns
the last member of the subresultant prs of f, g, multiplied
(if needed) by an appropriate power of -1; see the caveat below.

In this module we use three functions to compute the
resultant of f, g:
a) res(f, g, x) computes the resultant by evaluating
the determinant of sylvester(f, g, x, 1);
b) res_q(f, g, x) computes the resultant recursively, by
performing polynomial divisions in Q[x] with the function rem();
c) res_z(f, g, x) computes the resultant recursively, by
performing polynomial divisions in Z[x] with the function prem().

Caveat: If Df = degree(f, x) and Dg = degree(g, x), then:

resultant(f, g, x) = (-1)**(Df*Dg) * resultant(g, f, x).

For complete prs's the sign sequence of the Euclidean prs of f, g
is identical to the sign sequence of the subresultant prs of f, g
and the coefficients of one sequence  are easily computed from the
coefficients of the  other.

For incomplete prs's the polynomials in the subresultant prs, generally
differ in sign from those of the Euclidean prs, and --- unlike the
case of complete prs's --- it is not at all obvious how to compute
the coefficients of one sequence from the coefficients of the  other.

1B. Sturmian and modified subresultant prs's:
=============================================
For the same polynomials f, g in Z[x] mentioned above, their ``modified''
subresultant prs is a sequence of polynomials similar to the Sturmian
prs, the sequence obtained by applying in Q[x] Sturm's algorithm on f, g.

The two sequences differ in that the coefficients of each polynomial
in the modified subresultant prs are the determinants --- also referred
to as modified subresultants --- of appropriately selected  sub-matrices
of sylvester2(f, g, x), Sylvester's matrix of 1853 of dimensions 2n x 2n.

The determinant of sylvester2 itself is called the modified resultant
of f, g and it also can serve as a criterion of whether the two
polynomials have common roots or not.

For complete prs's the  sign sequence of the Sturmian prs of f, g is
identical to the sign sequence of the modified subresultant prs of
f, g and the coefficients of one sequence  are easily computed from
the coefficients of the other.

For incomplete prs's the polynomials in the modified subresultant prs,
generally differ in sign from those of the Sturmian prs, and --- unlike
the case of complete prs's --- it is not at all obvious how to compute
the coefficients of one sequence from the coefficients of the  other.

As Sylvester pointed out, the coefficients of the polynomial remainders
obtained as (modified) subresultants are the smallest possible without
introducing rationals and without computing (integer) greatest common
divisors.

1C. On terminology:
===================
Whence the terminology? Well generalized Sturmian prs's are
``modifications'' of Euclidean prs's; the hint came from the title
of the Pell-Gordon paper of 1917.

In the literature one also encounters the name ``non signed'' and
``signed'' prs for Euclidean and Sturmian prs respectively.

Likewise ``non signed'' and ``signed'' subresultant prs for
subresultant and modified subresultant prs respectively.

2. Functions in the module:
===========================
No function utilizes SymPy's function prem().

2A. Matrices:
=============
The functions sylvester(f, g, x, method=1) and
sylvester(f, g, x, method=2) compute either Sylvester matrix.
They can be used to compute (modified) subresultant prs's by
direct determinant evaluation.

The function bezout(f, g, x, method='prs') provides a matrix of
smaller dimensions than either Sylvester matrix. It is the function
of choice for computing (modified) subresultant prs's by direct
determinant evaluation.

sylvester(f, g, x, method=1)
sylvester(f, g, x, method=2)
bezout(f, g, x, method='prs')

The following identity holds:

bezout(f, g, x, method='prs') =
backward_eye(deg(f))*bezout(f, g, x, method='bz')*backward_eye(deg(f))

2B. Subresultant and modified subresultant prs's by
===================================================
determinant evaluations:
=======================
We use the Sylvester matrices of 1840 and 1853 to
compute, respectively, subresultant and modified
subresultant polynomial remainder sequences. However,
for large matrices this approach takes a lot of time.

Instead of utilizing the Sylvester matrices, we can
employ the Bezout matrix which is of smaller dimensions.

subresultants_sylv(f, g, x)
modified_subresultants_sylv(f, g, x)
subresultants_bezout(f, g, x)
modified_subresultants_bezout(f, g, x)

2C. Subresultant prs's by ONE determinant evaluation:
=====================================================
All three functions in this section evaluate one determinant
per remainder polynomial; this is the determinant of an
appropriately selected sub-matrix of sylvester1(f, g, x),
Sylvester's matrix of 1840.

To compute the remainder polynomials the function
subresultants_rem(f, g, x) employs rem(f, g, x).
By contrast, the other two functions implement Van Vleck's ideas
of 1900 and compute the remainder polynomials by trinagularizing
sylvester2(f, g, x), Sylvester's matrix of 1853.


subresultants_rem(f, g, x)
subresultants_vv(f, g, x)
subresultants_vv_2(f, g, x).

2E. Euclidean, Sturmian prs's in Q[x]:
======================================
euclid_q(f, g, x)
sturm_q(f, g, x)

2F. Euclidean, Sturmian and (modified) subresultant prs's P-G:
==============================================================
All functions in this section are based on the Pell-Gordon (P-G)
theorem of 1917.
Computations are done in Q[x], employing the function rem(f, g, x)
for the computation of the remainder polynomials.

euclid_pg(f, g, x)
sturm pg(f, g, x)
subresultants_pg(f, g, x)
modified_subresultants_pg(f, g, x)

2G. Euclidean, Sturmian and (modified) subresultant prs's A-M-V:
================================================================
All functions in this section are based on the Akritas-Malaschonok-
Vigklas (A-M-V) theorem of 2015.
Computations are done in Z[x], employing the function rem_z(f, g, x)
for the computation of the remainder polynomials.

euclid_amv(f, g, x)
sturm_amv(f, g, x)
subresultants_amv(f, g, x)
modified_subresultants_amv(f, g, x)

2Ga. Exception:
===============
subresultants_amv_q(f, g, x)

This function employs rem(f, g, x) for the computation of
the remainder polynomials, despite the fact that it implements
the A-M-V Theorem.

It is included in our module in order to show that theorems P-G
and A-M-V can be implemented utilizing either the function
rem(f, g, x) or the function rem_z(f, g, x).

For clearly historical reasons --- since the Collins-Brown-Traub
coefficients-reduction factor beta_i was not available in 1917 ---
we have implemented the Pell-Gordon theorem with the function
rem(f, g, x) and the A-M-V Theorem  with the function rem_z(f, g, x).

2H. Resultants:
===============
res(f, g, x)
res_q(f, g, x)
res_z(f, g, x)
"""


from sympy.concrete.summations import summation
from sympy.core.function import expand
from sympy.core.numbers import nan
from sympy.core.singleton import S
from sympy.core.symbol import Dummy as var
from sympy.functions.elementary.complexes import Abs, sign
from sympy.functions.elementary.integers import floor
from sympy.matrices.dense import eye, Matrix, zeros
from sympy.printing.pretty.pretty import pretty_print as pprint
from sympy.simplify.simplify import simplify
from sympy.polys.domains import QQ
from sympy.polys.polytools import degree, LC, Poly, pquo, quo, prem, rem
from sympy.polys.polyerrors import PolynomialError


def sylvester(f, g, x, method = 1):
    '''
      The input polynomials f, g are in Z[x] or in Q[x]. Let m = degree(f, x),
      n = degree(g, x) and mx = max(m, n).

      a. If method = 1 (default), computes sylvester1, Sylvester's matrix of 1840
          of dimension (m + n) x (m + n). The determinants of properly chosen
          submatrices of this matrix (a.k.a. subresultants) can be
          used to compute the coefficients of the Euclidean PRS of f, g.

      b. If method = 2, computes sylvester2, Sylvester's matrix of 1853
          of dimension (2*mx) x (2*mx). The determinants of properly chosen
          submatrices of this matrix (a.k.a. ``modified'' subresultants) can be
          used to compute the coefficients of the Sturmian PRS of f, g.

      Applications of these Matrices can be found in the references below.
      Especially, for applications of sylvester2, see the first reference!!

      References
      ==========
      1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``On a Theorem
      by Van Vleck Regarding Sturm Sequences. Serdica Journal of Computing,
      Vol. 7, No 4, 101-134, 2013.

      2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Sturm Sequences
      and Modified Subresultant Polynomial Remainder Sequences.''
      Serdica Journal of Computing, Vol. 8, No 1, 29-46, 2014.

    '''
    # obtain degrees of polys
    m, n = degree( Poly(f, x), x), degree( Poly(g, x), x)

    # Special cases:
    # A:: case m = n < 0 (i.e. both polys are 0)
    if m == n and n < 0:
        return Matrix([])

    # B:: case m = n = 0  (i.e. both polys are constants)
    if m == n and n == 0:
        return Matrix([])

    # C:: m == 0 and n < 0 or m < 0 and n == 0
    # (i.e. one poly is constant and the other is 0)
    if m == 0 and n < 0:
        return Matrix([])
    elif m < 0 and n == 0:
        return Matrix([])

    # D:: m >= 1 and n < 0 or m < 0 and n >=1
    # (i.e. one poly is of degree >=1 and the other is 0)
    if m >= 1 and n < 0:
        return Matrix([0])
    elif m < 0 and n >= 1:
        return Matrix([0])

    fp = Poly(f, x).all_coeffs()
    gp = Poly(g, x).all_coeffs()

    # Sylvester's matrix of 1840 (default; a.k.a. sylvester1)
    if method <= 1:
        M = zeros(m + n)
        k = 0
        for i in range(n):
            j = k
            for coeff in fp:
                M[i, j] = coeff
                j = j + 1
            k = k + 1
        k = 0
        for i in range(n, m + n):
            j = k
            for coeff in gp:
                M[i, j] = coeff
                j = j + 1
            k = k + 1
        return M

    # Sylvester's matrix of 1853 (a.k.a sylvester2)
    else:
        if len(fp) < len(gp):
            h = []
            for i in range(len(gp) - len(fp)):
                h.append(0)
            fp[ : 0] = h
        else:
            h = []
            for i in range(len(fp) - len(gp)):
                h.append(0)
            gp[ : 0] = h
        mx = max(m, n)
        dim = 2*mx
        M = zeros( dim )
        k = 0
        for i in range( mx ):
            j = k
            for coeff in fp:
                M[2*i, j] = coeff
                j = j + 1
            j = k
            for coeff in gp:
                M[2*i + 1, j] = coeff
                j = j + 1
            k = k + 1
        return M

def process_matrix_output(poly_seq, x):
    """
    poly_seq is a polynomial remainder sequence computed either by
    (modified_)subresultants_bezout or by (modified_)subresultants_sylv.

    This function removes from poly_seq all zero polynomials as well
    as all those whose degree is equal to the degree of a preceding
    polynomial in poly_seq, as we scan it from left to right.

    """
    L = poly_seq[:]  # get a copy of the input sequence
    d = degree(L[1], x)
    i = 2
    while i < len(L):
        d_i = degree(L[i], x)
        if d_i < 0:          # zero poly
            L.remove(L[i])
            i = i - 1
        if d == d_i:         # poly degree equals degree of previous poly
            L.remove(L[i])
            i = i - 1
        if d_i >= 0:
            d = d_i
        i = i + 1

    return L

def subresultants_sylv(f, g, x):
    """
    The input polynomials f, g are in Z[x] or in Q[x]. It is assumed
    that deg(f) >= deg(g).

    Computes the subresultant polynomial remainder sequence (prs)
    of f, g by evaluating determinants of appropriately selected
    submatrices of sylvester(f, g, x, 1). The dimensions of the
    latter are (deg(f) + deg(g)) x (deg(f) + deg(g)).

    Each coefficient is computed by evaluating the determinant of the
    corresponding submatrix of sylvester(f, g, x, 1).

    If the subresultant prs is complete, then the output coincides
    with the Euclidean sequence of the polynomials f, g.

    References:
    ===========
    1. G.M.Diaz-Toca,L.Gonzalez-Vega: Various New Expressions for Subresultants
    and Their Applications. Appl. Algebra in Engin., Communic. and Comp.,
    Vol. 15, 233-266, 2004.

    """

    # make sure neither f nor g is 0
    if f == 0 or g == 0:
        return [f, g]

    n = degF = degree(f, x)
    m = degG = degree(g, x)

    # make sure proper degrees
    if n == 0 and m == 0:
        return [f, g]
    if n < m:
        n, m, degF, degG, f, g = m, n, degG, degF, g, f
    if n > 0 and m == 0:
        return [f, g]

    SR_L = [f, g]      # subresultant list

    # form matrix sylvester(f, g, x, 1)
    S = sylvester(f, g, x, 1)

    # pick appropriate submatrices of S
    # and form subresultant polys
    j = m - 1

    while j > 0:
        Sp = S[:, :]  # copy of S
        # delete last j rows of coeffs of g
        for ind in range(m + n - j, m + n):
            Sp.row_del(m + n - j)
        # delete last j rows of coeffs of f
        for ind in range(m - j, m):
            Sp.row_del(m - j)

        # evaluate determinants and form coefficients list
        coeff_L, k, l = [], Sp.rows, 0
        while l <= j:
            coeff_L.append(Sp[:, 0:k].det())
            Sp.col_swap(k - 1, k + l)
            l += 1

        # form poly and append to SP_L
        SR_L.append(Poly(coeff_L, x).as_expr())
        j -= 1

    # j = 0
    SR_L.append(S.det())

    return process_matrix_output(SR_L, x)

def modified_subresultants_sylv(f, g, x):
    """
    The input polynomials f, g are in Z[x] or in Q[x]. It is assumed
    that deg(f) >= deg(g).

    Computes the modified subresultant polynomial remainder sequence (prs)
    of f, g by evaluating determinants of appropriately selected
    submatrices of sylvester(f, g, x, 2). The dimensions of the
    latter are (2*deg(f)) x (2*deg(f)).

    Each coefficient is computed by evaluating the determinant of the
    corresponding submatrix of sylvester(f, g, x, 2).

    If the modified subresultant prs is complete, then the output coincides
    with the Sturmian sequence of the polynomials f, g.

    References:
    ===========
    1. A. G. Akritas,G.I. Malaschonok and P.S. Vigklas:
    Sturm Sequences and Modified Subresultant Polynomial Remainder
    Sequences. Serdica Journal of Computing, Vol. 8, No 1, 29--46, 2014.

    """

    # make sure neither f nor g is 0
    if f == 0 or g == 0:
        return [f, g]

    n = degF = degree(f, x)
    m = degG = degree(g, x)

    # make sure proper degrees
    if n == 0 and m == 0:
        return [f, g]
    if n < m:
        n, m, degF, degG, f, g = m, n, degG, degF, g, f
    if n > 0 and m == 0:
        return [f, g]

    SR_L = [f, g]      # modified subresultant list

    # form matrix sylvester(f, g, x, 2)
    S = sylvester(f, g, x, 2)

    # pick appropriate submatrices of S
    # and form modified subresultant polys
    j = m - 1

    while j > 0:
        # delete last 2*j rows of pairs of coeffs of f, g
        Sp = S[0:2*n - 2*j, :]  # copy of first 2*n - 2*j rows of S

        # evaluate determinants and form coefficients list
        coeff_L, k, l = [], Sp.rows, 0
        while l <= j:
            coeff_L.append(Sp[:, 0:k].det())
            Sp.col_swap(k - 1, k + l)
            l += 1

        # form poly and append to SP_L
        SR_L.append(Poly(coeff_L, x).as_expr())
        j -= 1

    # j = 0
    SR_L.append(S.det())

    return process_matrix_output(SR_L, x)

def res(f, g, x):
    """
    The input polynomials f, g are in Z[x] or in Q[x].

    The output is the resultant of f, g computed by evaluating
    the determinant of the matrix sylvester(f, g, x, 1).

    References:
    ===========
    1. J. S. Cohen: Computer Algebra and Symbolic Computation
     - Mathematical Methods. A. K. Peters, 2003.

    """
    if f == 0 or g == 0:
        raise PolynomialError("The resultant of %s and %s is not defined" % (f, g))
    else:
        return sylvester(f, g, x, 1).det()

def res_q(f, g, x):
    """
    The input polynomials f, g are in Z[x] or in Q[x].

    The output is the resultant of f, g computed recursively
    by polynomial divisions in Q[x], using the function rem.
    See Cohen's book p. 281.

    References:
    ===========
    1. J. S. Cohen: Computer Algebra and Symbolic Computation
     - Mathematical Methods. A. K. Peters, 2003.
    """
    m = degree(f, x)
    n = degree(g, x)
    if m < n:
        return (-1)**(m*n) * res_q(g, f, x)
    elif n == 0:  # g is a constant
        return g**m
    else:
        r = rem(f, g, x)
        if r == 0:
            return 0
        else:
            s = degree(r, x)
            l = LC(g, x)
            return (-1)**(m*n) * l**(m-s)*res_q(g, r, x)

def res_z(f, g, x):
    """
    The input polynomials f, g are in Z[x] or in Q[x].

    The output is the resultant of f, g computed recursively
    by polynomial divisions in Z[x], using the function prem().
    See Cohen's book p. 283.

    References:
    ===========
    1. J. S. Cohen: Computer Algebra and Symbolic Computation
     - Mathematical Methods. A. K. Peters, 2003.
    """
    m = degree(f, x)
    n = degree(g, x)
    if m < n:
        return (-1)**(m*n) * res_z(g, f, x)
    elif n == 0:  # g is a constant
        return g**m
    else:
        r = prem(f, g, x)
        if r == 0:
            return 0
        else:
            delta = m - n + 1
            w = (-1)**(m*n) * res_z(g, r, x)
            s = degree(r, x)
            l = LC(g, x)
            k = delta * n - m + s
            return quo(w, l**k, x)

def sign_seq(poly_seq, x):
    """
    Given a sequence of polynomials poly_seq, it returns
    the sequence of signs of the leading coefficients of
    the polynomials in poly_seq.

    """
    return [sign(LC(poly_seq[i], x)) for i in range(len(poly_seq))]

def bezout(p, q, x, method='bz'):
    """
    The input polynomials p, q are in Z[x] or in Q[x]. Let
    mx = max(degree(p, x), degree(q, x)).

