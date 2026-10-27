    >>> pm1.ring
    ZZ[x]

    >>> v1*pm1
    PolyMatrix([
    [ x**2, -x],
    [-x**2,  x]], ring=QQ[x])

    >>> pm2 = PolyMatrix([[Poly(x**2, x, domain='QQ'), Poly(0, x, domain='QQ'), Poly(1, x, domain='QQ'), \
            Poly(x**3, x, domain='QQ'), Poly(0, x, domain='QQ'), Poly(-x**3, x, domain='QQ')]])
    >>> v2 = PolyMatrix([1, 0, 0, 0, 0, 0], x)
    >>> v2.ring
    QQ[x]
    >>> pm2*v2
    PolyMatrix([[x**2]], ring=QQ[x])

    """

    def __new__(cls, *args, ring=None):

        if not args:
            # PolyMatrix(ring=QQ[x])
            if ring is None:
                raise TypeError("The ring needs to be specified for an empty PolyMatrix")
            rows, cols, items, gens = 0, 0, [], ()
        elif isinstance(args[0], list):
            elements, gens = args[0], args[1:]
            if not elements:
                # PolyMatrix([])
                rows, cols, items = 0, 0, []
            elif isinstance(elements[0], (list, tuple)):
                # PolyMatrix([[1, 2]], x)
                rows, cols = len(elements), len(elements[0])
                items = [e for row in elements for e in row]
            else:
                # PolyMatrix([1, 2], x)
                rows, cols = len(elements), 1
                items = elements
        elif [type(a) for a in args[:3]] == [int, int, list]:
            # PolyMatrix(2, 2, [1, 2, 3, 4], x)
            rows, cols, items, gens = args[0], args[1], args[2], args[3:]
        elif [type(a) for a in args[:3]] == [int, int, type(lambda: 0)]:
            # PolyMatrix(2, 2, lambda i, j: i+j, x)
            rows, cols, func, gens = args[0], args[1], args[2], args[3:]
            items = [func(i, j) for i in range(rows) for j in range(cols)]
        else:
            raise TypeError("Invalid arguments")

        # PolyMatrix([[1]], x, y) vs PolyMatrix([[1]], (x, y))
        if len(gens) == 1 and isinstance(gens[0], tuple):
            gens = gens[0]
            # gens is now a tuple (x, y)

        return cls.from_list(rows, cols, items, gens, ring)

    @classmethod
    def from_list(cls, rows, cols, items, gens, ring):

        # items can be Expr, Poly, or a mix of Expr and Poly
        items = [_sympify(item) for item in items]
        if items and all(isinstance(item, Poly) for item in items):
            polys = True
        else:
            polys = False

        # Identify the ring for the polys
        if ring is not None:
            # Parse a domain string like 'QQ[x]'
            if isinstance(ring, str):
                ring = Poly(0, Dummy(), domain=ring).domain
        elif polys:
            p = items[0]
            for p2 in items[1:]:
                p, _ = p.unify(p2)
            ring = p.domain[p.gens]
        else:
            items, info = parallel_poly_from_expr(items, gens, field=True)
            ring = info['domain'][info['gens']]
            polys = True

        # Efficiently convert when all elements are Poly
        if polys:
            p_ring = Poly(0, ring.symbols, domain=ring.domain)
            to_ring = ring.ring.from_list
            convert_poly = lambda p: to_ring(p.unify(p_ring)[0].rep.to_list())
            elements = [convert_poly(p) for p in items]
        else:
            convert_expr = ring.from_sympy
            elements = [convert_expr(e.as_expr()) for e in items]

        # Convert to domain elements and construct DomainMatrix
        elements_lol = [[elements[i*cols + j] for j in range(cols)] for i in range(rows)]
        dm = DomainMatrix(elements_lol, (rows, cols), ring)
        return cls.from_dm(dm)

    @classmethod
    def from_dm(cls, dm):
        obj = super().__new__(cls)
        dm = dm.to_sparse()
        R = dm.domain
        obj._dm = dm
        obj.ring = R
        obj.domain = R.domain
        obj.gens = R.symbols
        return obj

    def to_Matrix(self):
        return self._dm.to_Matrix()

    @classmethod
    def from_Matrix(cls, other, *gens, ring=None):
        return cls(*other.shape, other.flat(), *gens, ring=ring)

    def set_gens(self, gens):
        return self.from_Matrix(self.to_Matrix(), gens)

    def __repr__(self):
        if self.rows * self.cols:
            return 'Poly' + repr(self.to_Matrix())[:-1] + f', ring={self.ring})'
        else:
            return f'PolyMatrix({self.rows}, {self.cols}, [], ring={self.ring})'

    @property
    def shape(self):
        return self._dm.shape

    @property
    def rows(self):
        return self.shape[0]

    @property
    def cols(self):
        return self.shape[1]

    def __len__(self):
        return self.rows * self.cols

    def __getitem__(self, key):

        def to_poly(v):
            ground = self._dm.domain.domain
            gens = self._dm.domain.symbols
            return Poly(v.to_dict(), gens, domain=ground)

        dm = self._dm

        if isinstance(key, slice):
            items = dm.flat()[key]
            return [to_poly(item) for item in items]
        elif isinstance(key, int):
            i, j = divmod(key, self.cols)
            e = dm[i,j]
            return to_poly(e.element)

        i, j = key
        if isinstance(i, int) and isinstance(j, int):
            return to_poly(dm[i, j].element)
        else:
            return self.from_dm(dm[i, j])

    def __eq__(self, other):
        if not isinstance(self, type(other)):
            return NotImplemented
        return self._dm == other._dm

    def __add__(self, other):
        if isinstance(other, type(self)):
            return self.from_dm(self._dm + other._dm)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, type(self)):
            return self.from_dm(self._dm - other._dm)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, type(self)):
            return self.from_dm(self._dm * other._dm)
        elif isinstance(other, int):
            other = _sympify(other)
        if isinstance(other, Expr):
            Kx = self.ring
            try:
                other_ds = DomainScalar(Kx.from_sympy(other), Kx)
            except (CoercionFailed, ValueError):
                other_ds = DomainScalar.from_sympy(other)
            return self.from_dm(self._dm * other_ds)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, int):
            other = _sympify(other)
        if isinstance(other, Expr):
            other_ds = DomainScalar.from_sympy(other)
            return self.from_dm(other_ds * self._dm)
        return NotImplemented

    def __truediv__(self, other):

        if isinstance(other, Poly):
            other = other.as_expr()
        elif isinstance(other, int):
            other = _sympify(other)
        if not isinstance(other, Expr):
            return NotImplemented

        other = self.domain.from_sympy(other)
        inverse = self.ring.convert_from(1/other, self.domain)
        inverse = DomainScalar(inverse, self.ring)
        dm = self._dm * inverse
        return self.from_dm(dm)

    def __neg__(self):
        return self.from_dm(-self._dm)

    def transpose(self):
        return self.from_dm(self._dm.transpose())

    def row_join(self, other):
        dm = DomainMatrix.hstack(self._dm, other._dm)
        return self.from_dm(dm)

    def col_join(self, other):
        dm = DomainMatrix.vstack(self._dm, other._dm)
        return self.from_dm(dm)

    def applyfunc(self, func):
        M = self.to_Matrix().applyfunc(func)
        return self.from_Matrix(M, self.gens)

    @classmethod
    def eye(cls, n, gens):
        return cls.from_dm(DomainMatrix.eye(n, QQ[gens]))

    @classmethod
    def zeros(cls, m, n, gens):
        return cls.from_dm(DomainMatrix.zeros((m, n), QQ[gens]))

    def rref(self, simplify='ignore', normalize_last='ignore'):
        # If this is K[x] then computes RREF in ground field K.
        if not (self.domain.is_Field and all(p.is_ground for p in self)):
            raise ValueError("PolyMatrix rref is only for ground field elements")
        dm = self._dm
        dm_ground = dm.convert_to(dm.domain.domain)
        dm_rref, pivots = dm_ground.rref()
        dm_rref = dm_rref.convert_to(dm.domain)
        return self.from_dm(dm_rref), pivots

    def nullspace(self):
        # If this is K[x] then computes nullspace in ground field K.
        if not (self.domain.is_Field and all(p.is_ground for p in self)):
            raise ValueError("PolyMatrix nullspace is only for ground field elements")
        dm = self._dm
        K, Kx = self.domain, self.ring
        dm_null_rows = dm.convert_to(K).nullspace(divide_last=True).convert_to(Kx)
        dm_null = dm_null_rows.transpose()
        dm_basis = [dm_null[:,i] for i in range(dm_null.shape[1])]
        return [self.from_dm(dmvec) for dmvec in dm_basis]

    def rank(self):
        return self.cols - len(self.nullspace())

MutablePolyMatrix = PolyMatrix = MutablePolyDenseMatrix
<|endoftext|>"""Polynomial factorization routines in characteristic zero. """

from sympy.external.gmpy import GROUND_TYPES

from sympy.core.random import _randint

from sympy.polys.galoistools import (
    gf_from_int_poly, gf_to_int_poly,
    gf_lshift, gf_add_mul, gf_mul,
    gf_div, gf_rem,
    gf_gcdex,
    gf_sqf_p,
    gf_factor_sqf, gf_factor)

from sympy.polys.densebasic import (
    dup_LC, dmp_LC, dmp_ground_LC,
    dup_TC,
    dup_convert, dmp_convert,
    dup_degree, dmp_degree,
    dmp_degree_in, dmp_degree_list,
    dmp_from_dict,
    dmp_zero_p,
    dmp_one,
    dmp_nest, dmp_raise,
    dup_strip,
    dmp_ground,
    dup_inflate,
    dmp_exclude, dmp_include,
    dmp_inject, dmp_eject,
    dup_terms_gcd, dmp_terms_gcd)

from sympy.polys.densearith import (
    dup_neg, dmp_neg,
    dup_add, dmp_add,
    dup_sub, dmp_sub,
    dup_mul, dmp_mul,
    dup_sqr,
    dmp_pow,
    dup_div, dmp_div,
    dup_quo, dmp_quo,
    dmp_expand,
    dmp_add_mul,
    dup_sub_mul, dmp_sub_mul,
    dup_lshift,
    dup_max_norm, dmp_max_norm,
    dup_l1_norm,
    dup_mul_ground, dmp_mul_ground,
    dup_quo_ground, dmp_quo_ground)

from sympy.polys.densetools import (
    dup_clear_denoms, dmp_clear_denoms,
    dup_trunc, dmp_ground_trunc,
    dup_content,
    dup_monic, dmp_ground_monic,
    dup_primitive, dmp_ground_primitive,
    dmp_eval_tail,
    dmp_eval_in, dmp_diff_eval_in,
    dup_shift, dmp_shift, dup_mirror)

from sympy.polys.euclidtools import (
    dmp_primitive,
    dup_inner_gcd, dmp_inner_gcd)

from sympy.polys.sqfreetools import (
    dup_sqf_p,
    dup_sqf_norm, dmp_sqf_norm,
    dup_sqf_part, dmp_sqf_part,
    _dup_check_degrees, _dmp_check_degrees,
    )

from sympy.polys.polyutils import _sort_factors
from sympy.polys.polyconfig import query

from sympy.polys.polyerrors import (
    ExtraneousFactors, DomainError, CoercionFailed, EvaluationFailed)

from sympy.utilities import subsets

from math import ceil as _ceil, log as _log, log2 as _log2


if GROUND_TYPES == 'flint':
    from flint import fmpz_poly
else:
    fmpz_poly = None


def dup_trial_division(f, factors, K):
    """
    Determine multiplicities of factors for a univariate polynomial
    using trial division.

    An error will be raised if any factor does not divide ``f``.
    """
    result = []

    for factor in factors:
        k = 0

        while True:
            q, r = dup_div(f, factor, K)

            if not r:
                f, k = q, k + 1
            else:
                break

        if k == 0:
            raise RuntimeError("trial division failed")

        result.append((factor, k))

    return _sort_factors(result)


def dmp_trial_division(f, factors, u, K):
    """
    Determine multiplicities of factors for a multivariate polynomial
    using trial division.

    An error will be raised if any factor does not divide ``f``.
    """
    result = []

    for factor in factors:
        k = 0

        while True:
            q, r = dmp_div(f, factor, u, K)

            if dmp_zero_p(r, u):
                f, k = q, k + 1
            else:
                break

        if k == 0:
            raise RuntimeError("trial division failed")

        result.append((factor, k))

    return _sort_factors(result)


def dup_zz_mignotte_bound(f, K):
    """
    The Knuth-Cohen variant of Mignotte bound for
    univariate polynomials in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> f = x**3 + 14*x**2 + 56*x + 64
    >>> R.dup_zz_mignotte_bound(f)
    152

    By checking ``factor(f)`` we can see that max coeff is 8

    Also consider a case that ``f`` is irreducible for example
    ``f = 2*x**2 + 3*x + 4``. To avoid a bug for these cases, we return the
    bound plus the max coefficient of ``f``

    >>> f = 2*x**2 + 3*x + 4
    >>> R.dup_zz_mignotte_bound(f)
    6

    Lastly, to see the difference between the new and the old Mignotte bound
    consider the irreducible polynomial:

    >>> f = 87*x**7 + 4*x**6 + 80*x**5 + 17*x**4 + 9*x**3 + 12*x**2 + 49*x + 26
    >>> R.dup_zz_mignotte_bound(f)
    744

    The new Mignotte bound is 744 whereas the old one (SymPy 1.5.1) is 1937664.


    References
    ==========

    ..[1] [Abbott13]_

    """
    from sympy.functions.combinatorial.factorials import binomial
    d = dup_degree(f)
    delta = _ceil(d / 2)
    delta2 = _ceil(delta / 2)

    # euclidean-norm
    eucl_norm = K.sqrt( sum( cf**2 for cf in f ) )

    # biggest values of binomial coefficients (p. 538 of reference)
    t1 = binomial(delta - 1, delta2)
    t2 = binomial(delta - 1, delta2 - 1)

    lc = K.abs(dup_LC(f, K))   # leading coefficient
    bound = t1 * eucl_norm + t2 * lc   # (p. 538 of reference)
    bound += dup_max_norm(f, K) # add max coeff for irreducible polys
    bound = _ceil(bound / 2) * 2   # round up to even integer

    return bound

def dmp_zz_mignotte_bound(f, u, K):
    """Mignotte bound for multivariate polynomials in `K[X]`. """
    a = dmp_max_norm(f, u, K)
    b = abs(dmp_ground_LC(f, u, K))
    n = sum(dmp_degree_list(f, u))

    return K.sqrt(K(n + 1))*2**n*a*b


def dup_zz_hensel_step(m, f, g, h, s, t, K):
    """
    One step in Hensel lifting in `Z[x]`.

    Given positive integer `m` and `Z[x]` polynomials `f`, `g`, `h`, `s`
    and `t` such that::

        f = g*h (mod m)
        s*g + t*h = 1 (mod m)

        lc(f) is not a zero divisor (mod m)
        lc(h) = 1

        deg(f) = deg(g) + deg(h)
        deg(s) < deg(h)
        deg(t) < deg(g)

    returns polynomials `G`, `H`, `S` and `T`, such that::

        f = G*H (mod m**2)
        S*G + T*H = 1 (mod m**2)

    References
    ==========

    .. [1] [Gathen99]_

    """
    M = m**2

    e = dup_sub_mul(f, g, h, K)
    e = dup_trunc(e, M, K)

    q, r = dup_div(dup_mul(s, e, K), h, K)

    q = dup_trunc(q, M, K)
    r = dup_trunc(r, M, K)

    u = dup_add(dup_mul(t, e, K), dup_mul(q, g, K), K)
    G = dup_trunc(dup_add(g, u, K), M, K)
    H = dup_trunc(dup_add(h, r, K), M, K)

    u = dup_add(dup_mul(s, G, K), dup_mul(t, H, K), K)
    b = dup_trunc(dup_sub(u, [K.one], K), M, K)

    c, d = dup_div(dup_mul(s, b, K), H, K)

    c = dup_trunc(c, M, K)
    d = dup_trunc(d, M, K)

    u = dup_add(dup_mul(t, b, K), dup_mul(c, G, K), K)
    S = dup_trunc(dup_sub(s, d, K), M, K)
    T = dup_trunc(dup_sub(t, u, K), M, K)

    return G, H, S, T


def dup_zz_hensel_lift(p, f, f_list, l, K):
    r"""
    Multifactor Hensel lifting in `Z[x]`.

    Given a prime `p`, polynomial `f` over `Z[x]` such that `lc(f)`
    is a unit modulo `p`, monic pair-wise coprime polynomials `f_i`
    over `Z[x]` satisfying::

        f = lc(f) f_1 ... f_r (mod p)

    and a positive integer `l`, returns a list of monic polynomials
    `F_1,\ F_2,\ \dots,\ F_r` satisfying::

       f = lc(f) F_1 ... F_r (mod p**l)

       F_i = f_i (mod p), i = 1..r

    References
    ==========

    .. [1] [Gathen99]_

    """
    r = len(f_list)
    lc = dup_LC(f, K)

    if r == 1:
        F = dup_mul_ground(f, K.gcdex(lc, p**l)[0], K)
        return [ dup_trunc(F, p**l, K) ]

    m = p
    k = r // 2
    d = int(_ceil(_log2(l)))

    g = gf_from_int_poly([lc], p)

    for f_i in f_list[:k]:
        g = gf_mul(g, gf_from_int_poly(f_i, p), p, K)

    h = gf_from_int_poly(f_list[k], p)

    for f_i in f_list[k + 1:]:
        h = gf_mul(h, gf_from_int_poly(f_i, p), p, K)

    s, t, _ = gf_gcdex(g, h, p, K)

    g = gf_to_int_poly(g, p)
    h = gf_to_int_poly(h, p)
    s = gf_to_int_poly(s, p)
    t = gf_to_int_poly(t, p)

    for _ in range(1, d + 1):
        (g, h, s, t), m = dup_zz_hensel_step(m, f, g, h, s, t, K), m**2

    return dup_zz_hensel_lift(p, g, f_list[:k], l, K) \
        + dup_zz_hensel_lift(p, h, f_list[k:], l, K)

def _test_pl(fc, q, pl):
    if q > pl // 2:
        q = q - pl
    if not q:
        return True
    return fc % q == 0

def dup_zz_zassenhaus(f, K):
    """Factor primitive square-free polynomials in `Z[x]`. """
    n = dup_degree(f)

    if n == 1:
        return [f]

    from sympy.ntheory import isprime

    fc = f[-1]
    A = dup_max_norm(f, K)
    b = dup_LC(f, K)
    B = int(abs(K.sqrt(K(n + 1))*2**n*A*b))
    C = int((n + 1)**(2*n)*A**(2*n - 1))
    gamma = int(_ceil(2*_log2(C)))
    bound = int(2*gamma*_log(gamma))
    a = []
    # choose a prime number `p` such that `f` be square free in Z_p
    # if there are many factors in Z_p, choose among a few different `p`
    # the one with fewer factors
    for px in range(3, bound + 1):
        if not isprime(px) or b % px == 0:
            continue

        px = K.convert(px)

        F = gf_from_int_poly(f, px)

        if not gf_sqf_p(F, px, K):
            continue
        fsqfx = gf_factor_sqf(F, px, K)[1]
        a.append((px, fsqfx))
        if len(fsqfx) < 15 or len(a) > 4:
            break
    p, fsqf = min(a, key=lambda x: len(x[1]))

    l = int(_ceil(_log(2*B + 1, p)))

    modular = [gf_to_int_poly(ff, p) for ff in fsqf]

    g = dup_zz_hensel_lift(p, f, modular, l, K)

    sorted_T = range(len(g))
    T = set(sorted_T)
    factors, s = [], 1
    pl = p**l

    while 2*s <= len(T):
        for S in subsets(sorted_T, s):
            # lift the constant coefficient of the product `G` of the factors
            # in the subset `S`; if it is does not divide `fc`, `G` does
            # not divide the input polynomial

            if b == 1:
                q = 1
                for i in S:
                    q = q*g[i][-1]
                q = q % pl
                if not _test_pl(fc, q, pl):
                    continue
            else:
                G = [b]
                for i in S:
                    G = dup_mul(G, g[i], K)
                G = dup_trunc(G, pl, K)
                G = dup_primitive(G, K)[1]
                q = G[-1]
                if q and fc % q != 0:
                    continue

            H = [b]
            S = set(S)
            T_S = T - S

            if b == 1:
                G = [b]
                for i in S:
                    G = dup_mul(G, g[i], K)
                G = dup_trunc(G, pl, K)

            for i in T_S:
                H = dup_mul(H, g[i], K)

            H = dup_trunc(H, pl, K)

            G_norm = dup_l1_norm(G, K)
            H_norm = dup_l1_norm(H, K)

            if G_norm*H_norm <= B:
                T = T_S
                sorted_T = [i for i in sorted_T if i not in S]

                G = dup_primitive(G, K)[1]
                f = dup_primitive(H, K)[1]

                factors.append(G)
                b = dup_LC(f, K)

                break
        else:
            s += 1

    return factors + [f]


def dup_zz_irreducible_p(f, K):
    """Test irreducibility using Eisenstein's criterion. """
    lc = dup_LC(f, K)
    tc = dup_TC(f, K)

    e_fc = dup_content(f[1:], K)

    if e_fc:
        from sympy.ntheory import factorint
        e_ff = factorint(int(e_fc))

        for p in e_ff.keys():
            if (lc % p) and (tc % p**2):
                return True


def dup_cyclotomic_p(f, K, irreducible=False):
    """
    Efficiently test if ``f`` is a cyclotomic polynomial.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> f = x**16 + x**14 - x**10 + x**8 - x**6 + x**2 + 1
    >>> R.dup_cyclotomic_p(f)
    False

    >>> g = x**16 + x**14 - x**10 - x**8 - x**6 + x**2 + 1
    >>> R.dup_cyclotomic_p(g)
    True

    References
    ==========

    Bradford, Russell J., and James H. Davenport. "Effective tests for
    cyclotomic polynomials." In International Symposium on Symbolic and
    Algebraic Computation, pp. 244-251. Springer, Berlin, Heidelberg, 1988.

    """
    if K.is_QQ:
        try:
            K0, K = K, K.get_ring()
            f = dup_convert(f, K0, K)
        except CoercionFailed:
            return False
    elif not K.is_ZZ:
        return False

    lc = dup_LC(f, K)
    tc = dup_TC(f, K)

    if lc != 1 or (tc != -1 and tc != 1):
        return False

    if not irreducible:
        coeff, factors = dup_factor_list(f, K)

        if coeff != K.one or factors != [(f, 1)]:
            return False

    n = dup_degree(f)
    g, h = [], []

    for i in range(n, -1, -2):
        g.insert(0, f[i])

    for i in range(n - 1, -1, -2):
        h.insert(0, f[i])

    g = dup_sqr(dup_strip(g), K)
    h = dup_sqr(dup_strip(h), K)

    F = dup_sub(g, dup_lshift(h, 1, K), K)

    if K.is_negative(dup_LC(F, K)):
        F = dup_neg(F, K)

    if F == f:
        return True

    g = dup_mirror(f, K)

    if K.is_negative(dup_LC(g, K)):
        g = dup_neg(g, K)

    if F == g and dup_cyclotomic_p(g, K):
        return True

    G = dup_sqf_part(F, K)

    if dup_sqr(G, K) == F and dup_cyclotomic_p(G, K):
        return True

    return False


def dup_zz_cyclotomic_poly(n, K):
    """Efficiently generate n-th cyclotomic polynomial. """
    from sympy.ntheory import factorint
    h = [K.one, -K.one]

    for p, k in factorint(n).items():
        h = dup_quo(dup_inflate(h, p, K), h, K)
        h = dup_inflate(h, p**(k - 1), K)

    return h


def _dup_cyclotomic_decompose(n, K):
    from sympy.ntheory import factorint

    H = [[K.one, -K.one]]

    for p, k in factorint(n).items():
        Q = [ dup_quo(dup_inflate(h, p, K), h, K) for h in H ]
        H.extend(Q)

        for i in range(1, k):
            Q = [ dup_inflate(q, p, K) for q in Q ]
            H.extend(Q)

    return H


def dup_zz_cyclotomic_factor(f, K):
    """
    Efficiently factor polynomials `x**n - 1` and `x**n + 1` in `Z[x]`.

    Given a univariate polynomial `f` in `Z[x]` returns a list of factors
    of `f`, provided that `f` is in the form `x**n - 1` or `x**n + 1` for
    `n >= 1`. Otherwise returns None.

    Factorization is performed using cyclotomic decomposition of `f`,
    which makes this method much faster that any other direct factorization
    approach (e.g. Zassenhaus's).

    References
    ==========

    .. [1] [Weisstein09]_

    """
    lc_f, tc_f = dup_LC(f, K), dup_TC(f, K)

    if dup_degree(f) <= 0:
        return None

    if lc_f != 1 or tc_f not in [-1, 1]:
        return None

    if any(bool(cf) for cf in f[1:-1]):
        return None

    n = dup_degree(f)
    F = _dup_cyclotomic_decompose(n, K)

    if not K.is_one(tc_f):
        return F
    else:
        H = []

        for h in _dup_cyclotomic_decompose(2*n, K):
            if h not in F:
                H.append(h)

        return H


def dup_zz_factor_sqf(f, K):
    """Factor square-free (non-primitive) polynomials in `Z[x]`. """
    cont, g = dup_primitive(f, K)

    n = dup_degree(g)

    if dup_LC(g, K) < 0:
        cont, g = -cont, dup_neg(g, K)

    if n <= 0:
        return cont, []
    elif n == 1:
        return cont, [g]

    if query('USE_IRREDUCIBLE_IN_FACTOR'):
        if dup_zz_irreducible_p(g, K):
            return cont, [g]

    factors = None

    if query('USE_CYCLOTOMIC_FACTOR'):
        factors = dup_zz_cyclotomic_factor(g, K)

    if factors is None:
        factors = dup_zz_zassenhaus(g, K)

    return cont, _sort_factors(factors, multiple=False)


def dup_zz_factor(f, K):
    """
    Factor (non square-free) polynomials in `Z[x]`.

    Given a univariate polynomial `f` in `Z[x]` computes its complete
    factorization `f_1, ..., f_n` into irreducibles over integers::

                f = content(f) f_1**k_1 ... f_n**k_n

    The factorization is computed by reducing the input polynomial
    into a primitive square-free polynomial and factoring it using
    Zassenhaus algorithm. Trial division is used to recover the
    multiplicities of factors.

    The result is returned as a tuple consisting of::

              (content(f), [(f_1, k_1), ..., (f_n, k_n))

    Examples
    ========

    Consider the polynomial `f = 2*x**4 - 2`::

        >>> from sympy.polys import ring, ZZ
        >>> R, x = ring("x", ZZ)

        >>> R.dup_zz_factor(2*x**4 - 2)
        (2, [(x - 1, 1), (x + 1, 1), (x**2 + 1, 1)])

    In result we got the following factorization::

                 f = 2 (x - 1) (x + 1) (x**2 + 1)

    Note that this is a complete factorization over integers,
    however over Gaussian integers we can factor the last term.

    By default, polynomials `x**n - 1` and `x**n + 1` are factored
    using cyclotomic decomposition to speedup computations. To
    disable this behaviour set cyclotomic=False.

    References
    ==========

    .. [1] [Gathen99]_

    """
    if GROUND_TYPES == 'flint':
        f_flint = fmpz_poly(f[::-1])
        cont, factors = f_flint.factor()
        factors = [(fac.coeffs()[::-1], exp) for fac, exp in factors]
        return cont, _sort_factors(factors)

    cont, g = dup_primitive(f, K)

    n = dup_degree(g)

    if dup_LC(g, K) < 0:
        cont, g = -cont, dup_neg(g, K)

    if n <= 0:
        return cont, []
    elif n == 1:
        return cont, [(g, 1)]

    if query('USE_IRREDUCIBLE_IN_FACTOR'):
        if dup_zz_irreducible_p(g, K):
            return cont, [(g, 1)]

    g = dup_sqf_part(g, K)
    H = None

    if query('USE_CYCLOTOMIC_FACTOR'):
        H = dup_zz_cyclotomic_factor(g, K)

    if H is None:
        H = dup_zz_zassenhaus(g, K)

    factors = dup_trial_division(f, H, K)

    _dup_check_degrees(f, factors)

    return cont, factors


def dmp_zz_wang_non_divisors(E, cs, ct, K):
    """Wang/EEZ: Compute a set of valid divisors.  """
    result = [ cs*ct ]

    for q in E:
        q = abs(q)

        for r in reversed(result):
            while r != 1:
                r = K.gcd(r, q)
                q = q // r

            if K.is_one(q):
                return None

        result.append(q)

    return result[1:]


def dmp_zz_wang_test_points(f, T, ct, A, u, K):
    """Wang/EEZ: Test evaluation points for suitability. """
    if not dmp_eval_tail(dmp_LC(f, K), A, u - 1, K):
        raise EvaluationFailed('no luck')

    g = dmp_eval_tail(f, A, u, K)

    if not dup_sqf_p(g, K):
        raise EvaluationFailed('no luck')

    c, h = dup_primitive(g, K)

    if K.is_negative(dup_LC(h, K)):
        c, h = -c, dup_neg(h, K)

    v = u - 1

    E = [ dmp_eval_tail(t, A, v, K) for t, _ in T ]
    D = dmp_zz_wang_non_divisors(E, c, ct, K)

    if D is not None:
        return c, h, E
    else:
        raise EvaluationFailed('no luck')


def dmp_zz_wang_lead_coeffs(f, T, cs, E, H, A, u, K):
    """Wang/EEZ: Compute correct leading coefficients. """
    C, J, v = [], [0]*len(E), u - 1

    for h in H:
        c = dmp_one(v, K)
        d = dup_LC(h, K)*cs

        for i in reversed(range(len(E))):
            k, e, (t, _) = 0, E[i], T[i]

            while not (d % e):
                d, k = d//e, k + 1

            if k != 0:
                c, J[i] = dmp_mul(c, dmp_pow(t, k, v, K), v, K), 1

        C.append(c)

    if not all(J):
        raise ExtraneousFactors  # pragma: no cover

    CC, HH = [], []

    for c, h in zip(C, H):
        d = dmp_eval_tail(c, A, v, K)
        lc = dup_LC(h, K)

        if K.is_one(cs):
            cc = lc//d
        else:
            g = K.gcd(lc, d)
            d, cc = d//g, lc//g
            h, cs = dup_mul_ground(h, d, K), cs//d

        c = dmp_mul_ground(c, cc, v, K)

        CC.append(c)
        HH.append(h)

    if K.is_one(cs):
        return f, HH, CC

    CCC, HHH = [], []

    for c, h in zip(CC, HH):
        CCC.append(dmp_mul_ground(c, cs, v, K))
        HHH.append(dmp_mul_ground(h, cs, 0, K))

    f = dmp_mul_ground(f, cs**(len(H) - 1), u, K)

    return f, HHH, CCC


def dup_zz_diophantine(F, m, p, K):
    """Wang/EEZ: Solve univariate Diophantine equations. """
    if len(F) == 2:
        a, b = F

        f = gf_from_int_poly(a, p)
        g = gf_from_int_poly(b, p)

        s, t, G = gf_gcdex(g, f, p, K)

        s = gf_lshift(s, m, K)
        t = gf_lshift(t, m, K)

        q, s = gf_div(s, f, p, K)

        t = gf_add_mul(t, q, g, p, K)

        s = gf_to_int_poly(s, p)
        t = gf_to_int_poly(t, p)

        result = [s, t]
    else:
        G = [F[-1]]

        for f in reversed(F[1:-1]):
            G.insert(0, dup_mul(f, G[0], K))

        S, T = [], [[1]]

        for f, g in zip(F, G):
            t, s = dmp_zz_diophantine([g, f], T[-1], [], 0, p, 1, K)
            T.append(t)
            S.append(s)

        result, S = [], S + [T[-1]]

        for s, f in zip(S, F):
            s = gf_from_int_poly(s, p)
            f = gf_from_int_poly(f, p)

            r = gf_rem(gf_lshift(s, m, K), f, p, K)
            s = gf_to_int_poly(r, p)

            result.append(s)

    return result


def dmp_zz_diophantine(F, c, A, d, p, u, K):
    """Wang/EEZ: Solve multivariate Diophantine equations. """
    if not A:
        S = [ [] for _ in F ]
        n = dup_degree(c)

        for i, coeff in enumerate(c):
            if not coeff:
                continue

            T = dup_zz_diophantine(F, n - i, p, K)

            for j, (s, t) in enumerate(zip(S, T)):
                t = dup_mul_ground(t, coeff, K)
                S[j] = dup_trunc(dup_add(s, t, K), p, K)
    else:
        n = len(A)
        e = dmp_expand(F, u, K)

        a, A = A[-1], A[:-1]
        B, G = [], []

        for f in F:
            B.append(dmp_quo(e, f, u, K))
            G.append(dmp_eval_in(f, a, n, u, K))

        C = dmp_eval_in(c, a, n, u, K)

        v = u - 1

        S = dmp_zz_diophantine(G, C, A, d, p, v, K)
        S = [ dmp_raise(s, 1, v, K) for s in S ]

        for s, b in zip(S, B):
            c = dmp_sub_mul(c, s, b, u, K)

        c = dmp_ground_trunc(c, p, u, K)

        m = dmp_nest([K.one, -a], n, K)
        M = dmp_one(n, K)

        for k in range(0, d):
            if dmp_zero_p(c, u):
                break

            M = dmp_mul(M, m, u, K)
            C = dmp_diff_eval_in(c, k + 1, a, n, u, K)

            if not dmp_zero_p(C, v):
                C = dmp_quo_ground(C, K.factorial(K(k) + 1), v, K)
                T = dmp_zz_diophantine(G, C, A, d, p, v, K)

                for i, t in enumerate(T):
                    T[i] = dmp_mul(dmp_raise(t, 1, v, K), M, u, K)

                for i, (s, t) in enumerate(zip(S, T)):
                    S[i] = dmp_add(s, t, u, K)

                for t, b in zip(T, B):
                    c = dmp_sub_mul(c, t, b, u, K)

                c = dmp_ground_trunc(c, p, u, K)

        S = [ dmp_ground_trunc(s, p, u, K) for s in S ]

    return S


def dmp_zz_wang_hensel_lifting(f, H, LC, A, p, u, K):
    """Wang/EEZ: Parallel Hensel lifting algorithm. """
    S, n, v = [f], len(A), u - 1

    H = list(H)

    for i, a in enumerate(reversed(A[1:])):
        s = dmp_eval_in(S[0], a, n - i, u - i, K)
        S.insert(0, dmp_ground_trunc(s, p, v - i, K))

    d = max(dmp_degree_list(f, u)[1:])

    for j, s, a in zip(range(2, n + 2), S, A):
        G, w = list(H), j - 1

        I, J = A[:j - 2], A[j - 1:]

        for i, (h, lc) in enumerate(zip(H, LC)):
            lc = dmp_ground_trunc(dmp_eval_tail(lc, J, v, K), p, w - 1, K)
            H[i] = [lc] + dmp_raise(h[1:], 1, w - 1, K)

        m = dmp_nest([K.one, -a], w, K)
        M = dmp_one(w, K)

        c = dmp_sub(s, dmp_expand(H, w, K), w, K)

        dj = dmp_degree_in(s, w, w)

        for k in range(0, dj):
            if dmp_zero_p(c, w):
                break

            M = dmp_mul(M, m, w, K)
            C = dmp_diff_eval_in(c, k + 1, a, w, w, K)

            if not dmp_zero_p(C, w - 1):
                C = dmp_quo_ground(C, K.factorial(K(k) + 1), w - 1, K)
                T = dmp_zz_diophantine(G, C, I, d, p, w - 1, K)

                for i, (h, t) in enumerate(zip(H, T)):
                    h = dmp_add_mul(h, dmp_raise(t, 1, w - 1, K), M, w, K)
                    H[i] = dmp_ground_trunc(h, p, w, K)

                h = dmp_sub(s, dmp_expand(H, w, K), w, K)
                c = dmp_ground_trunc(h, p, w, K)

    if dmp_expand(H, u, K) != f:
        raise ExtraneousFactors  # pragma: no cover
    else:
        return H


def dmp_zz_wang(f, u, K, mod=None, seed=None):
    r"""
    Factor primitive square-free polynomials in `Z[X]`.

    Given a multivariate polynomial `f` in `Z[x_1,...,x_n]`, which is
    primitive and square-free in `x_1`, computes factorization of `f` into
    irreducibles over integers.

    The procedure is based on Wang's Enhanced Extended Zassenhaus
    algorithm. The algorithm works by viewing `f` as a univariate polynomial
    in `Z[x_2,...,x_n][x_1]`, for which an evaluation mapping is computed::

                      x_2 -> a_2, ..., x_n -> a_n

    where `a_i`, for `i = 2, \dots, n`, are carefully chosen integers.  The
    mapping is used to transform `f` into a univariate polynomial in `Z[x_1]`,
    which can be factored efficiently using Zassenhaus algorithm. The last
    step is to lift univariate factors to obtain true multivariate
    factors. For this purpose a parallel Hensel lifting procedure is used.

    The parameter ``seed`` is passed to _randint and can be used to seed randint
    (when an integer) or (for testing purposes) can be a sequence of numbers.

    References
    ==========

    .. [1] [Wang78]_
    .. [2] [Geddes92]_

    """
    from sympy.ntheory import nextprime

    randint = _randint(seed)

    ct, T = dmp_zz_factor(dmp_LC(f, K), u - 1, K)

    b = dmp_zz_mignotte_bound(f, u, K)
    p = K(nextprime(b))

    if mod is None:
        if u == 1:
            mod = 2
        else:
            mod = 1

    history, configs, A, r = set(), [], [K.zero]*u, None

    try:
        cs, s, E = dmp_zz_wang_test_points(f, T, ct, A, u, K)

        _, H = dup_zz_factor_sqf(s, K)

        r = len(H)

        if r == 1:
            return [f]

        configs = [(s, cs, E, H, A)]
    except EvaluationFailed:
        pass

    eez_num_configs = query('EEZ_NUMBER_OF_CONFIGS')
    eez_num_tries = query('EEZ_NUMBER_OF_TRIES')
    eez_mod_step = query('EEZ_MODULUS_STEP')

    while len(configs) < eez_num_configs:
        for _ in range(eez_num_tries):
            A = [ K(randint(-mod, mod)) for _ in range(u) ]

            if tuple(A) not in history:
                history.add(tuple(A))
            else:
                continue

            try:
                cs, s, E = dmp_zz_wang_test_points(f, T, ct, A, u, K)
            except EvaluationFailed:
                continue

            _, H = dup_zz_factor_sqf(s, K)

            rr = len(H)

            if r is not None:
                if rr != r:  # pragma: no cover
                    if rr < r:
                        configs, r = [], rr
                    else:
                        continue
            else:
                r = rr

            if r == 1:
                return [f]

            configs.append((s, cs, E, H, A))

            if len(configs) == eez_num_configs:
                break
        else:
            mod += eez_mod_step

    s_norm, s_arg, i = None, 0, 0

    for s, _, _, _, _ in configs:
        _s_norm = dup_max_norm(s, K)

        if s_norm is not None:
            if _s_norm < s_norm:
                s_norm = _s_norm
                s_arg = i
        else:
            s_norm = _s_norm

        i += 1

    _, cs, E, H, A = configs[s_arg]
    orig_f = f

    try:
        f, H, LC = dmp_zz_wang_lead_coeffs(f, T, cs, E, H, A, u, K)
        factors = dmp_zz_wang_hensel_lifting(f, H, LC, A, p, u, K)
    except ExtraneousFactors:  # pragma: no cover
        if query('EEZ_RESTART_IF_NEEDED'):
            return dmp_zz_wang(orig_f, u, K, mod + 1)
        else:
            raise ExtraneousFactors(
                "we need to restart algorithm with better parameters")

    result = []

    for f in factors:
        _, f = dmp_ground_primitive(f, u, K)

        if K.is_negative(dmp_ground_LC(f, u, K)):
            f = dmp_neg(f, u, K)

        result.append(f)

    return result


def dmp_zz_factor(f, u, K):
    r"""
    Factor (non square-free) polynomials in `Z[X]`.

    Given a multivariate polynomial `f` in `Z[x]` computes its complete
    factorization `f_1, \dots, f_n` into irreducibles over integers::

                 f = content(f) f_1**k_1 ... f_n**k_n

    The factorization is computed by reducing the input polynomial
    into a primitive square-free polynomial and factoring it using
    Enhanced Extended Zassenhaus (EEZ) algorithm. Trial division
    is used to recover the multiplicities of factors.

    The result is returned as a tuple consisting of::

             (content(f), [(f_1, k_1), ..., (f_n, k_n))

    Consider polynomial `f = 2*(x**2 - y**2)`::

        >>> from sympy.polys import ring, ZZ
        >>> R, x,y = ring("x,y", ZZ)

        >>> R.dmp_zz_factor(2*x**2 - 2*y**2)
        (2, [(x - y, 1), (x + y, 1)])

    In result we got the following factorization::

                    f = 2 (x - y) (x + y)

    References
    ==========

    .. [1] [Gathen99]_

    """
    if not u:
        return dup_zz_factor(f, K)

    if dmp_zero_p(f, u):
        return K.zero, []

    cont, g = dmp_ground_primitive(f, u, K)

    if dmp_ground_LC(g, u, K) < 0:
        cont, g = -cont, dmp_neg(g, u, K)

    if all(d <= 0 for d in dmp_degree_list(g, u)):
        return cont, []

    G, g = dmp_primitive(g, u, K)

    factors = []

    if dmp_degree(g, u) > 0:
        g = dmp_sqf_part(g, u, K)
        H = dmp_zz_wang(g, u, K)
        factors = dmp_trial_division(f, H, u, K)

    for g, k in dmp_zz_factor(G, u - 1, K)[1]:
        factors.insert(0, ([g], k))

    _dmp_check_degrees(f, u, factors)

    return cont, _sort_factors(factors)


def dup_qq_i_factor(f, K0):
    """Factor univariate polynomials into irreducibles in `QQ_I[x]`. """
    # Factor in QQ<I>
    K1 = K0.as_AlgebraicField()
    f = dup_convert(f, K0, K1)
    coeff, factors = dup_factor_list(f, K1)
    factors = [(dup_convert(fac, K1, K0), i) for fac, i in factors]
    coeff = K0.convert(coeff, K1)
    return coeff, factors


def dup_zz_i_factor(f, K0):
    """Factor univariate polynomials into irreducibles in `ZZ_I[x]`. """
    # First factor in QQ_I
    K1 = K0.get_field()
    f = dup_convert(f, K0, K1)
    coeff, factors = dup_qq_i_factor(f, K1)

    new_factors = []
    for fac, i in factors:
        # Extract content
        fac_denom, fac_num = dup_clear_denoms(fac, K1)
        fac_num_ZZ_I = dup_convert(fac_num, K1, K0)
        content, fac_prim = dmp_ground_primitive(fac_num_ZZ_I, 0, K0)

        coeff = (coeff * content ** i) // fac_denom ** i
        new_factors.append((fac_prim, i))

    factors = new_factors
    coeff = K0.convert(coeff, K1)
    return coeff, factors


def dmp_qq_i_factor(f, u, K0):
    """Factor multivariate polynomials into irreducibles in `QQ_I[X]`. """
    # Factor in QQ<I>
    K1 = K0.as_AlgebraicField()
    f = dmp_convert(f, u, K0, K1)
    coeff, factors = dmp_factor_list(f, u, K1)
    factors = [(dmp_convert(fac, u, K1, K0), i) for fac, i in factors]
    coeff = K0.convert(coeff, K1)
    return coeff, factors


def dmp_zz_i_factor(f, u, K0):
    """Factor multivariate polynomials into irreducibles in `ZZ_I[X]`. """
    # First factor in QQ_I
    K1 = K0.get_field()
    f = dmp_convert(f, u, K0, K1)
    coeff, factors = dmp_qq_i_factor(f, u, K1)

    new_factors = []
    for fac, i in factors:
        # Extract content
        fac_denom, fac_num = dmp_clear_denoms(fac, u, K1)
        fac_num_ZZ_I = dmp_convert(fac_num, u, K1, K0)
        content, fac_prim = dmp_ground_primitive(fac_num_ZZ_I, u, K0)

        coeff = (coeff * content ** i) // fac_denom ** i
        new_factors.append((fac_prim, i))

    factors = new_factors
    coeff = K0.convert(coeff, K1)
    return coeff, factors


def dup_ext_factor(f, K):
    r"""Factor univariate polynomials over algebraic number fields.

    The domain `K` must be an algebraic number field `k(a)` (see :ref:`QQ(a)`).

    Examples
    ========

    First define the algebraic number field `K = \mathbb{Q}(\sqrt{2})`:

    >>> from sympy import QQ, sqrt
    >>> from sympy.polys.factortools import dup_ext_factor
    >>> K = QQ.algebraic_field(sqrt(2))

    We can now factorise the polynomial `x^2 - 2` over `K`:

    >>> p = [K(1), K(0), K(-2)] # x^2 - 2
    >>> p1 = [K(1), -K.unit]    # x - sqrt(2)
    >>> p2 = [K(1), +K.unit]    # x + sqrt(2)
    >>> dup_ext_factor(p, K) == (K.one, [(p1, 1), (p2, 1)])
    True

    Usually this would be done at a higher level:

    >>> from sympy import factor
    >>> from sympy.abc import x
    >>> factor(x**2 - 2, extension=sqrt(2))
    (x - sqrt(2))*(x + sqrt(2))

    Explanation
    ===========

    Uses Trager's algorithm. In particular this function is algorithm
    ``alg_factor`` from [Trager76]_.

    If `f` is a polynomial in `k(a)[x]` then its norm `g(x)` is a polynomial in
    `k[x]`. If `g(x)` is square-free and has irreducible factors `g_1(x)`,
    `g_2(x)`, `\cdots` then the irreducible factors of `f` in `k(a)[x]` are
    given by `f_i(x) = \gcd(f(x), g_i(x))` where the GCD is computed in
    `k(a)[x]`.

    The first step in Trager's algorithm is to find an integer shift `s` so
    that `f(x-sa)` has square-free norm. Then the norm is factorized in `k[x]`
    and the GCD of (shifted) `f` with each factor gives the shifted factors of
    `f`. At the end the shift is undone to recover the unshifted factors of `f`
    in `k(a)[x]`.

    The algorithm reduces the problem of factorization in `k(a)[x]` to
    factorization in `k[x]` with the main additional steps being to compute the
    norm (a resultant calculation in `k[x,y]`) and some polynomial GCDs in
    `k(a)[x]`.

    In practice in SymPy the base field `k` will be the rationals :ref:`QQ` and
    this function factorizes a polynomial with coefficients in an algebraic
    number field  like `\mathbb{Q}(\sqrt{2})`.

    See Also
    ========

    dmp_ext_factor:
        Analogous function for multivariate polynomials over ``k(a)``.
    dup_sqf_norm:
        Subroutine ``sqfr_norm`` also from [Trager76]_.
    sympy.polys.polytools.factor:
        The high-level function that ultimately uses this function as needed.
    """
    n, lc = dup_degree(f), dup_LC(f, K)

    f = dup_monic(f, K)

    if n <= 0:
        return lc, []
    if n == 1:
        return lc, [(f, 1)]

    f, F = dup_sqf_part(f, K), f
    s, g, r = dup_sqf_norm(f, K)

    factors = dup_factor_list_include(r, K.dom)

    if len(factors) == 1:
        return lc, [(f, n//dup_degree(f))]

    H = s*K.unit

    for i, (factor, _) in enumerate(factors):
        h = dup_convert(factor, K.dom, K)
        h, _, g = dup_inner_gcd(h, g, K)
        h = dup_shift(h, H, K)
        factors[i] = h

    factors = dup_trial_division(F, factors, K)

    _dup_check_degrees(F, factors)

    return lc, factors


def dmp_ext_factor(f, u, K):
    r"""Factor multivariate polynomials over algebraic number fields.

    The domain `K` must be an algebraic number field `k(a)` (see :ref:`QQ(a)`).

    Examples
    ========

    First define the algebraic number field `K = \mathbb{Q}(\sqrt{2})`:

    >>> from sympy import QQ, sqrt
    >>> from sympy.polys.factortools import dmp_ext_factor
    >>> K = QQ.algebraic_field(sqrt(2))

    We can now factorise the polynomial `x^2 y^2 - 2` over `K`:

    >>> p = [[K(1),K(0),K(0)], [], [K(-2)]] # x**2*y**2 - 2
    >>> p1 = [[K(1),K(0)], [-K.unit]]       # x*y - sqrt(2)
    >>> p2 = [[K(1),K(0)], [+K.unit]]       # x*y + sqrt(2)
    >>> dmp_ext_factor(p, 1, K) == (K.one, [(p1, 1), (p2, 1)])
    True

    Usually this would be done at a higher level:

    >>> from sympy import factor
    >>> from sympy.abc import x, y
    >>> factor(x**2*y**2 - 2, extension=sqrt(2))
    (x*y - sqrt(2))*(x*y + sqrt(2))

    Explanation
    ===========

    This is Trager's algorithm for multivariate polynomials. In particular this
    function is algorithm ``alg_factor`` from [Trager76]_.

    See :func:`dup_ext_factor` for explanation.

    See Also
    ========

    dup_ext_factor:
        Analogous function for univariate polynomials over ``k(a)``.
    dmp_sqf_norm:
        Multivariate version of subroutine ``sqfr_norm`` also from [Trager76]_.
    sympy.polys.polytools.factor:
        The high-level function that ultimately uses this function as needed.
    """
    if not u:
        return dup_ext_factor(f, K)

    lc = dmp_ground_LC(f, u, K)
    f = dmp_ground_monic(f, u, K)

    if all(d <= 0 for d in dmp_degree_list(f, u)):
        return lc, []

    f, F = dmp_sqf_part(f, u, K), f
    s, g, r = dmp_sqf_norm(f, u, K)

    factors = dmp_factor_list_include(r, u, K.dom)

    if len(factors) == 1:
        factors = [f]
    else:
        for i, (factor, _) in enumerate(factors):
            h = dmp_convert(factor, u, K.dom, K)
            h, _, g = dmp_inner_gcd(h, g, u, K)
            a = [si*K.unit for si in s]
            h = dmp_shift(h, a, u, K)
            factors[i] = h

    result = dmp_trial_division(F, factors, u, K)

    _dmp_check_degrees(F, u, result)

    return lc, result


def dup_gf_factor(f, K):
    """Factor univariate polynomials over finite fields. """
    f = dup_convert(f, K, K.dom)

    coeff, factors = gf_factor(f, K.mod, K.dom)

    for i, (f, k) in enumerate(factors):
        factors[i] = (dup_convert(f, K.dom, K), k)

    return K.convert(coeff, K.dom), factors


def dmp_gf_factor(f, u, K):
    """Factor multivariate polynomials over finite fields. """
    raise NotImplementedError('multivariate polynomials over finite fields')


def dup_factor_list(f, K0):
    """Factor univariate polynomials into irreducibles in `K[x]`. """
    j, f = dup_terms_gcd(f, K0)
    cont, f = dup_primitive(f, K0)

    if K0.is_FiniteField:
        coeff, factors = dup_gf_factor(f, K0)
    elif K0.is_Algebraic:
        coeff, factors = dup_ext_factor(f, K0)
    elif K0.is_GaussianRing:
        coeff, factors = dup_zz_i_factor(f, K0)
    elif K0.is_GaussianField:
        coeff, factors = dup_qq_i_factor(f, K0)
    else:
        if not K0.is_Exact:
            K0_inexact, K0 = K0, K0.get_exact()
            f = dup_convert(f, K0_inexact, K0)
        else:
            K0_inexact = None

        if K0.is_Field:
            K = K0.get_ring()

            denom, f = dup_clear_denoms(f, K0, K)
            f = dup_convert(f, K0, K)
        else:
            K = K0

        if K.is_ZZ:
            coeff, factors = dup_zz_factor(f, K)
        elif K.is_Poly:
            f, u = dmp_inject(f, 0, K)

            coeff, factors = dmp_factor_list(f, u, K.dom)

            for i, (f, k) in enumerate(factors):
                factors[i] = (dmp_eject(f, u, K), k)

            coeff = K.convert(coeff, K.dom)
        else:  # pragma: no cover
            raise DomainError('factorization not supported over %s' % K0)

        if K0.is_Field:
            for i, (f, k) in enumerate(factors):
                factors[i] = (dup_convert(f, K, K0), k)

            coeff = K0.convert(coeff, K)
            coeff = K0.quo(coeff, denom)

            if K0_inexact:
                for i, (f, k) in enumerate(factors):
                    max_norm = dup_max_norm(f, K0)
                    f = dup_quo_ground(f, max_norm, K0)
                    f = dup_convert(f, K0, K0_inexact)
                    factors[i] = (f, k)
                    coeff = K0.mul(coeff, K0.pow(max_norm, k))

                coeff = K0_inexact.convert(coeff, K0)
                K0 = K0_inexact

    if j:
        factors.insert(0, ([K0.one, K0.zero], j))

    return coeff*cont, _sort_factors(factors)


def dup_factor_list_include(f, K):
    """Factor univariate polynomials into irreducibles in `K[x]`. """
    coeff, factors = dup_factor_list(f, K)

    if not factors:
        return [(dup_strip([coeff]), 1)]
    else:
        g = dup_mul_ground(factors[0][0], coeff, K)
        return [(g, factors[0][1])] + factors[1:]


def dmp_factor_list(f, u, K0):
    """Factor multivariate polynomials into irreducibles in `K[X]`. """
    if not u:
        return dup_factor_list(f, K0)

    J, f = dmp_terms_gcd(f, u, K0)
    cont, f = dmp_ground_primitive(f, u, K0)

    if K0.is_FiniteField:  # pragma: no cover
        coeff, factors = dmp_gf_factor(f, u, K0)
    elif K0.is_Algebraic:
        coeff, factors = dmp_ext_factor(f, u, K0)
    elif K0.is_GaussianRing:
        coeff, factors = dmp_zz_i_factor(f, u, K0)
    elif K0.is_GaussianField:
        coeff, factors = dmp_qq_i_factor(f, u, K0)
    else:
        if not K0.is_Exact:
            K0_inexact, K0 = K0, K0.get_exact()
            f = dmp_convert(f, u, K0_inexact, K0)
        else:
            K0_inexact = None

        if K0.is_Field:
            K = K0.get_ring()

            denom, f = dmp_clear_denoms(f, u, K0, K)
            f = dmp_convert(f, u, K0, K)
        else:
            K = K0

        if K.is_ZZ:
            levels, f, v = dmp_exclude(f, u, K)
            coeff, factors = dmp_zz_factor(f, v, K)

            for i, (f, k) in enumerate(factors):
                factors[i] = (dmp_include(f, levels, v, K), k)
        elif K.is_Poly:
            f, v = dmp_inject(f, u, K)

            coeff, factors = dmp_factor_list(f, v, K.dom)

            for i, (f, k) in enumerate(factors):
                factors[i] = (dmp_eject(f, v, K), k)

            coeff = K.convert(coeff, K.dom)
        else:  # pragma: no cover
            raise DomainError('factorization not supported over %s' % K0)

        if K0.is_Field:
            for i, (f, k) in enumerate(factors):
                factors[i] = (dmp_convert(f, u, K, K0), k)

            coeff = K0.convert(coeff, K)
            coeff = K0.quo(coeff, denom)

            if K0_inexact:
                for i, (f, k) in enumerate(factors):
                    max_norm = dmp_max_norm(f, u, K0)
                    f = dmp_quo_ground(f, max_norm, u, K0)
                    f = dmp_convert(f, u, K0, K0_inexact)
                    factors[i] = (f, k)
                    coeff = K0.mul(coeff, K0.pow(max_norm, k))

                coeff = K0_inexact.convert(coeff, K0)
                K0 = K0_inexact

    for i, j in enumerate(reversed(J)):
        if not j:
            continue

        term = {(0,)*(u - i) + (1,) + (0,)*i: K0.one}
        factors.insert(0, (dmp_from_dict(term, u, K0), j))

    return coeff*cont, _sort_factors(factors)


def dmp_factor_list_include(f, u, K):
    """Factor multivariate polynomials into irreducibles in `K[X]`. """
    if not u:
        return dup_factor_list_include(f, K)

    coeff, factors = dmp_factor_list(f, u, K)

    if not factors:
        return [(dmp_ground(coeff, u), 1)]
    else:
        g = dmp_mul_ground(factors[0][0], coeff, u, K)
        return [(g, factors[0][1])] + factors[1:]


def dup_irreducible_p(f, K):
    """
    Returns ``True`` if a univariate polynomial ``f`` has no factors
    over its domain.
    """
    return dmp_irreducible_p(f, 0, K)


def dmp_irreducible_p(f, u, K):
    """
    Returns ``True`` if a multivariate polynomial ``f`` has no factors
    over its domain.
    """
    _, factors = dmp_factor_list(f, u, K)

    if not factors:
        return True
    elif len(factors) > 1:
        return False
    else:
        _, k = factors[0]
        return k == 1
<|endoftext|>"""Polynomial manipulation algorithms and algebraic objects. """

__all__ = [
    'Poly', 'PurePoly', 'poly_from_expr', 'parallel_poly_from_expr', 'degree',
    'total_degree', 'degree_list', 'LC', 'LM', 'LT', 'pdiv', 'prem', 'pquo',
    'pexquo', 'div', 'rem', 'quo', 'exquo', 'half_gcdex', 'gcdex', 'invert',
    'subresultants', 'resultant', 'discriminant', 'cofactors', 'gcd_list',
    'gcd', 'lcm_list', 'lcm', 'terms_gcd', 'trunc', 'monic', 'content',
    'primitive', 'compose', 'decompose', 'sturm', 'gff_list', 'gff',
    'sqf_norm', 'sqf_part', 'sqf_list', 'sqf', 'factor_list', 'factor',
    'intervals', 'refine_root', 'count_roots', 'all_roots', 'real_roots',
    'nroots', 'ground_roots', 'nth_power_roots_poly', 'cancel', 'reduced',
    'groebner', 'is_zero_dimensional', 'GroebnerBasis', 'poly',

    'symmetrize', 'horner', 'interpolate', 'rational_interpolate', 'viete',

    'together',

    'BasePolynomialError', 'ExactQuotientFailed', 'PolynomialDivisionFailed',
    'OperationNotSupported', 'HeuristicGCDFailed', 'HomomorphismFailed',
    'IsomorphismFailed', 'ExtraneousFactors', 'EvaluationFailed',
    'RefinementFailed', 'CoercionFailed', 'NotInvertible', 'NotReversible',
    'NotAlgebraic', 'DomainError', 'PolynomialError', 'UnificationFailed',
    'GeneratorsError', 'GeneratorsNeeded', 'ComputationFailed',
    'UnivariatePolynomialError', 'MultivariatePolynomialError',
    'PolificationFailed', 'OptionError', 'FlagError',

    'minpoly', 'minimal_polynomial', 'primitive_element', 'field_isomorphism',
    'to_number_field', 'isolate', 'round_two', 'prime_decomp',
    'prime_valuation', 'galois_group',

    'itermonomials', 'Monomial',

    'lex', 'grlex', 'grevlex', 'ilex', 'igrlex', 'igrevlex',

    'CRootOf', 'rootof', 'RootOf', 'ComplexRootOf', 'RootSum',

    'roots',

    'Domain', 'FiniteField', 'IntegerRing', 'RationalField', 'RealField',
    'ComplexField', 'PythonFiniteField', 'GMPYFiniteField',
    'PythonIntegerRing', 'GMPYIntegerRing', 'PythonRational',
    'GMPYRationalField', 'AlgebraicField', 'PolynomialRing', 'FractionField',
    'ExpressionDomain', 'FF_python', 'FF_gmpy', 'ZZ_python', 'ZZ_gmpy',
    'QQ_python', 'QQ_gmpy', 'GF', 'FF', 'ZZ', 'QQ', 'ZZ_I', 'QQ_I', 'RR',
    'CC', 'EX', 'EXRAW',

    'construct_domain',

    'swinnerton_dyer_poly', 'cyclotomic_poly', 'symmetric_poly',
    'random_poly', 'interpolating_poly',

    'jacobi_poly', 'chebyshevt_poly', 'chebyshevu_poly', 'hermite_poly',
    'hermite_prob_poly', 'legendre_poly', 'laguerre_poly',

    'bernoulli_poly', 'bernoulli_c_poly', 'genocchi_poly', 'euler_poly',
    'andre_poly',

    'apart', 'apart_list', 'assemble_partfrac_list',

    'Options',

    'ring', 'xring', 'vring', 'sring',

    'field', 'xfield', 'vfield', 'sfield'
]

from .polytools import (Poly, PurePoly, poly_from_expr,
        parallel_poly_from_expr, degree, total_degree, degree_list, LC, LM,
        LT, pdiv, prem, pquo, pexquo, div, rem, quo, exquo, half_gcdex, gcdex,
        invert, subresultants, resultant, discriminant, cofactors, gcd_list,
        gcd, lcm_list, lcm, terms_gcd, trunc, monic, content, primitive,
        compose, decompose, sturm, gff_list, gff, sqf_norm, sqf_part,
        sqf_list, sqf, factor_list, factor, intervals, refine_root,
        count_roots, all_roots, real_roots, nroots, ground_roots,
        nth_power_roots_poly, cancel, reduced, groebner, is_zero_dimensional,
        GroebnerBasis, poly)

from .polyfuncs import (symmetrize, horner, interpolate,
        rational_interpolate, viete)

from .rationaltools import together

from .polyerrors import (BasePolynomialError, ExactQuotientFailed,
        PolynomialDivisionFailed, OperationNotSupported, HeuristicGCDFailed,
        HomomorphismFailed, IsomorphismFailed, ExtraneousFactors,
        EvaluationFailed, RefinementFailed, CoercionFailed, NotInvertible,
        NotReversible, NotAlgebraic, DomainError, PolynomialError,
        UnificationFailed, GeneratorsError, GeneratorsNeeded,
        ComputationFailed, UnivariatePolynomialError,
        MultivariatePolynomialError, PolificationFailed, OptionError,
        FlagError)

from .numberfields import (minpoly, minimal_polynomial, primitive_element,
        field_isomorphism, to_number_field, isolate, round_two, prime_decomp,
        prime_valuation, galois_group)

from .monomials import itermonomials, Monomial

from .orderings import lex, grlex, grevlex, ilex, igrlex, igrevlex

from .rootoftools import CRootOf, rootof, RootOf, ComplexRootOf, RootSum

from .polyroots import roots

from .domains import (Domain, FiniteField, IntegerRing, RationalField,
        RealField, ComplexField, PythonFiniteField, GMPYFiniteField,
        PythonIntegerRing, GMPYIntegerRing, PythonRational, GMPYRationalField,
        AlgebraicField, PolynomialRing, FractionField, ExpressionDomain,
        FF_python, FF_gmpy, ZZ_python, ZZ_gmpy, QQ_python, QQ_gmpy, GF, FF,
        ZZ, QQ, ZZ_I, QQ_I, RR, CC, EX, EXRAW)

from .constructor import construct_domain

from .specialpolys import (swinnerton_dyer_poly, cyclotomic_poly,
        symmetric_poly, random_poly, interpolating_poly)

from .orthopolys import (jacobi_poly, chebyshevt_poly, chebyshevu_poly,
        hermite_poly, hermite_prob_poly, legendre_poly, laguerre_poly)

from .appellseqs import (bernoulli_poly, bernoulli_c_poly, genocchi_poly,
        euler_poly, andre_poly)

from .partfrac import apart, apart_list, assemble_partfrac_list

from .polyoptions import Options

from .rings import ring, xring, vring, sring

from .fields import field, xfield, vfield, sfield
<|endoftext|>"""Low-level linear systems solver. """


from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import connected_components

from sympy.core.sympify import sympify
from sympy.core.numbers import Integer, Rational
from sympy.matrices.dense import MutableDenseMatrix
from sympy.polys.domains import ZZ, QQ

from sympy.polys.domains import EX
from sympy.polys.rings import sring
from sympy.polys.polyerrors import NotInvertible
from sympy.polys.domainmatrix import DomainMatrix


class PolyNonlinearError(Exception):
    """Raised by solve_lin_sys for nonlinear equations"""
    pass


class RawMatrix(MutableDenseMatrix):
    """
    .. deprecated:: 1.9

       This class fundamentally is broken by design. Use ``DomainMatrix`` if
       you want a matrix over the polys domains or ``Matrix`` for a matrix
       with ``Expr`` elements. The ``RawMatrix`` class will be removed/broken
       in future in order to reestablish the invariant that the elements of a
       Matrix should be of type ``Expr``.

    """
    _sympify = staticmethod(lambda x, *args, **kwargs: x)

    def __init__(self, *args, **kwargs):
        sympy_deprecation_warning(
            """
            The RawMatrix class is deprecated. Use either DomainMatrix or
            Matrix instead.
            """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-rawmatrix",
        )

        domain = ZZ
        for i in range(self.rows):
            for j in range(self.cols):
                val = self[i,j]
                if getattr(val, 'is_Poly', False):
                    K = val.domain[val.gens]
                    val_sympy = val.as_expr()
                elif hasattr(val, 'parent'):
                    K = val.parent()
                    val_sympy = K.to_sympy(val)
                elif isinstance(val, (int, Integer)):
                    K = ZZ
                    val_sympy = sympify(val)
                elif isinstance(val, Rational):
                    K = QQ
                    val_sympy = val
                else:
                    for K in ZZ, QQ:
                        if K.of_type(val):
                            val_sympy = K.to_sympy(val)
                            break
                    else:
                        raise TypeError
                domain = domain.unify(K)
                self[i,j] = val_sympy
        self.ring = domain


def eqs_to_matrix(eqs_coeffs, eqs_rhs, gens, domain):
    """Get matrix from linear equations in dict format.

    Explanation
    ===========

    Get the matrix representation of a system of linear equations represented
    as dicts with low-level DomainElement coefficients. This is an
    *internal* function that is used by solve_lin_sys.

    Parameters
    ==========

    eqs_coeffs: list[dict[Symbol, DomainElement]]
        The left hand sides of the equations as dicts mapping from symbols to
        coefficients where the coefficients are instances of
        DomainElement.
    eqs_rhs: list[DomainElements]
        The right hand sides of the equations as instances of
        DomainElement.
    gens: list[Symbol]
        The unknowns in the system of equations.
    domain: Domain
        The domain for coefficients of both lhs and rhs.

    Returns
    =======

    The augmented matrix representation of the system as a DomainMatrix.

    Examples
    ========

    >>> from sympy import symbols, ZZ
    >>> from sympy.polys.solvers import eqs_to_matrix
    >>> x, y = symbols('x, y')
    >>> eqs_coeff = [{x:ZZ(1), y:ZZ(1)}, {x:ZZ(1), y:ZZ(-1)}]
    >>> eqs_rhs = [ZZ(0), ZZ(-1)]
    >>> eqs_to_matrix(eqs_coeff, eqs_rhs, [x, y], ZZ)
    DomainMatrix([[1, 1, 0], [1, -1, 1]], (2, 3), ZZ)

    See also
    ========

    solve_lin_sys: Uses :func:`~eqs_to_matrix` internally
    """
    sym2index = {x: n for n, x in enumerate(gens)}
    nrows = len(eqs_coeffs)
    ncols = len(gens) + 1
    rows = [[domain.zero] * ncols for _ in range(nrows)]
    for row, eq_coeff, eq_rhs in zip(rows, eqs_coeffs, eqs_rhs):
        for sym, coeff in eq_coeff.items():
            row[sym2index[sym]] = domain.convert(coeff)
        row[-1] = -domain.convert(eq_rhs)

    return DomainMatrix(rows, (nrows, ncols), domain)


def sympy_eqs_to_ring(eqs, symbols):
    """Convert a system of equations from Expr to a PolyRing

    Explanation
    ===========

    High-level functions like ``solve`` expect Expr as inputs but can use
    ``solve_lin_sys`` internally. This function converts equations from
    ``Expr`` to the low-level poly types used by the ``solve_lin_sys``
    function.

    Parameters
    ==========

    eqs: List of Expr
        A list of equations as Expr instances
    symbols: List of Symbol
        A list of the symbols that are the unknowns in the system of
        equations.

    Returns
    =======

    Tuple[List[PolyElement], Ring]: The equations as PolyElement instances
    and the ring of polynomials within which each equation is represented.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.polys.solvers import sympy_eqs_to_ring
    >>> a, x, y = symbols('a, x, y')
    >>> eqs = [x-y, x+a*y]
    >>> eqs_ring, ring = sympy_eqs_to_ring(eqs, [x, y])
    >>> eqs_ring
    [x - y, x + a*y]
    >>> type(eqs_ring[0])
    <class 'sympy.polys.rings.PolyElement'>
    >>> ring
    ZZ(a)[x,y]

    With the equations in this form they can be passed to ``solve_lin_sys``:

    >>> from sympy.polys.solvers import solve_lin_sys
    >>> solve_lin_sys(eqs_ring, ring)
    {y: 0, x: 0}
    """
    try:
        K, eqs_K = sring(eqs, symbols, field=True, extension=True)
    except NotInvertible:
        # https://github.com/sympy/sympy/issues/18874
        K, eqs_K = sring(eqs, symbols, domain=EX)
    return eqs_K, K.to_domain()


def solve_lin_sys(eqs, ring, _raw=True):
    """Solve a system of linear equations from a PolynomialRing

    Explanation
    ===========

    Solves a system of linear equations given as PolyElement instances of a
    PolynomialRing. The basic arithmetic is carried out using instance of
    DomainElement which is more efficient than :class:`~sympy.core.expr.Expr`
    for the most common inputs.

    While this is a public function it is intended primarily for internal use
    so its interface is not necessarily convenient. Users are suggested to use
    the :func:`sympy.solvers.solveset.linsolve` function (which uses this
    function internally) instead.

    Parameters
    ==========

    eqs: list[PolyElement]
        The linear equations to be solved as elements of a
        PolynomialRing (assumed equal to zero).
    ring: PolynomialRing
        The polynomial ring from which eqs are drawn. The generators of this
        ring are the unknowns to be solved for and the domain of the ring is
        the domain of the coefficients of the system of equations.
    _raw: bool
        If *_raw* is False, the keys and values in the returned dictionary
        will be of type Expr (and the unit of the field will be removed from
        the keys) otherwise the low-level polys types will be returned, e.g.
        PolyElement: PythonRational.

    Returns
    =======

    ``None`` if the system has no solution.

    dict[Symbol, Expr] if _raw=False

    dict[Symbol, DomainElement] if _raw=True.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.polys.solvers import solve_lin_sys, sympy_eqs_to_ring
    >>> x, y = symbols('x, y')
    >>> eqs = [x - y, x + y - 2]
    >>> eqs_ring, ring = sympy_eqs_to_ring(eqs, [x, y])
    >>> solve_lin_sys(eqs_ring, ring)
    {y: 1, x: 1}

    Passing ``_raw=False`` returns the same result except that the keys are
    ``Expr`` rather than low-level poly types.

    >>> solve_lin_sys(eqs_ring, ring, _raw=False)
    {x: 1, y: 1}

    See also
    ========

    sympy_eqs_to_ring: prepares the inputs to ``solve_lin_sys``.
    linsolve: ``linsolve`` uses ``solve_lin_sys`` internally.
    sympy.solvers.solvers.solve: ``solve`` uses ``solve_lin_sys`` internally.
    """
    as_expr = not _raw

    assert ring.domain.is_Field

    eqs_dict = [dict(eq) for eq in eqs]

    one_monom = ring.one.monoms()[0]
    zero = ring.domain.zero

    eqs_rhs = []
    eqs_coeffs = []
    for eq_dict in eqs_dict:
        eq_rhs = eq_dict.pop(one_monom, zero)
        eq_coeffs = {}
        for monom, coeff in eq_dict.items():
            if sum(monom) != 1:
                msg = "Nonlinear term encountered in solve_lin_sys"
                raise PolyNonlinearError(msg)
            eq_coeffs[ring.gens[monom.index(1)]] = coeff
        if not eq_coeffs:
            if not eq_rhs:
                continue
            else:
                return None
        eqs_rhs.append(eq_rhs)
        eqs_coeffs.append(eq_coeffs)

    result = _solve_lin_sys(eqs_coeffs, eqs_rhs, ring)

    if result is not None and as_expr:

        def to_sympy(x):
            as_expr = getattr(x, 'as_expr', None)
            if as_expr:
                return as_expr()
            else:
                return ring.domain.to_sympy(x)

        tresult = {to_sympy(sym): to_sympy(val) for sym, val in result.items()}

        # Remove 1.0x
        result = {}
        for k, v in tresult.items():
            if k.is_Mul:
                c, s = k.as_coeff_Mul()
                result[s] = v/c
            else:
                result[k] = v

    return result


def _solve_lin_sys(eqs_coeffs, eqs_rhs, ring):
    """Solve a linear system from dict of PolynomialRing coefficients

    Explanation
    ===========

    This is an **internal** function used by :func:`solve_lin_sys` after the
    equations have been preprocessed. The role of this function is to split
    the system into connected components and pass those to
    :func:`_solve_lin_sys_component`.

    Examples
    ========

    Setup a system for $x-y=0$ and $x+y=2$ and solve:

    >>> from sympy import symbols, sring
    >>> from sympy.polys.solvers import _solve_lin_sys
    >>> x, y = symbols('x, y')
    >>> R, (xr, yr) = sring([x, y], [x, y])
    >>> eqs = [{xr:R.one, yr:-R.one}, {xr:R.one, yr:R.one}]
    >>> eqs_rhs = [R.zero, -2*R.one]
    >>> _solve_lin_sys(eqs, eqs_rhs, R)
    {y: 1, x: 1}

    See also
    ========

    solve_lin_sys: This function is used internally by :func:`solve_lin_sys`.
    """
    V = ring.gens
    E = []
    for eq_coeffs in eqs_coeffs:
        syms = list(eq_coeffs)
        E.extend(zip(syms[:-1], syms[1:]))
    G = V, E

    components = connected_components(G)

    sym2comp = {}
    for n, component in enumerate(components):
        for sym in component:
            sym2comp[sym] = n

    subsystems = [([], []) for _ in range(len(components))]
    for eq_coeff, eq_rhs in zip(eqs_coeffs, eqs_rhs):
        sym = next(iter(eq_coeff), None)
        sub_coeff, sub_rhs = subsystems[sym2comp[sym]]
        sub_coeff.append(eq_coeff)
        sub_rhs.append(eq_rhs)

    sol = {}
    for subsystem in subsystems:
        subsol = _solve_lin_sys_component(subsystem[0], subsystem[1], ring)
        if subsol is None:
            return None
        sol.update(subsol)

    return sol


def _solve_lin_sys_component(eqs_coeffs, eqs_rhs, ring):
    """Solve a linear system from dict of PolynomialRing coefficients

    Explanation
    ===========

    This is an **internal** function used by :func:`solve_lin_sys` after the
    equations have been preprocessed. After :func:`_solve_lin_sys` splits the
    system into connected components this function is called for each
    component. The system of equations is solved using Gauss-Jordan
    elimination with division followed by back-substitution.

    Examples
    ========

    Setup a system for $x-y=0$ and $x+y=2$ and solve:

    >>> from sympy import symbols, sring
    >>> from sympy.polys.solvers import _solve_lin_sys_component
    >>> x, y = symbols('x, y')
    >>> R, (xr, yr) = sring([x, y], [x, y])
    >>> eqs = [{xr:R.one, yr:-R.one}, {xr:R.one, yr:R.one}]
    >>> eqs_rhs = [R.zero, -2*R.one]
    >>> _solve_lin_sys_component(eqs, eqs_rhs, R)
    {y: 1, x: 1}

    See also
    ========

    solve_lin_sys: This function is used internally by :func:`solve_lin_sys`.
    """

    # transform from equations to matrix form
    matrix = eqs_to_matrix(eqs_coeffs, eqs_rhs, ring.gens, ring.domain)

    # convert to a field for rref
    if not matrix.domain.is_Field:
        matrix = matrix.to_field()

    # solve by row-reduction
    echelon, pivots = matrix.rref()

    # construct the returnable form of the solutions
    keys = ring.gens

    if pivots and pivots[-1] == len(keys):
        return None

    if len(pivots) == len(keys):
        sol = []
        for s in [row[-1] for row in echelon.rep.to_ddm()]:
            a = s
            sol.append(a)
        sols = dict(zip(keys, sol))
    else:
        sols = {}
        g = ring.gens
        # Extract ground domain coefficients and convert to the ring:
        if hasattr(ring, 'ring'):
            convert = ring.ring.ground_new
        else:
            convert = ring.ground_new
        echelon = echelon.rep.to_ddm()
        vals_set = {v for row in echelon for v in row}
        vals_map = {v: convert(v) for v in vals_set}
        echelon = [[vals_map[eij] for eij in ei] for ei in echelon]
        for i, p in enumerate(pivots):
            v = echelon[i][-1] - sum(echelon[i][j]*g[j] for j in range(p+1, len(g)) if echelon[i][j])
            sols[keys[p]] = v

    return sols
<|endoftext|>"""Algorithms for computing symbolic roots of polynomials. """


import math
from functools import reduce

from sympy.core import S, I, pi
from sympy.core.exprtools import factor_terms
from sympy.core.function import _mexpand
from sympy.core.logic import fuzzy_not
from sympy.core.mul import expand_2arg, Mul
from sympy.core.intfunc import igcd
from sympy.core.numbers import Rational, comp
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol, symbols
from sympy.core.sympify import sympify
from sympy.functions import exp, im, cos, acos, Piecewise
from sympy.functions.elementary.miscellaneous import root, sqrt
from sympy.ntheory import divisors, isprime, nextprime
from sympy.polys.domains import EX
from sympy.polys.polyerrors import (PolynomialError, GeneratorsNeeded,
    DomainError, UnsolvableFactorError)
from sympy.polys.polyquinticconst import PolyQuintic
from sympy.polys.polytools import Poly, cancel, factor, gcd_list, discriminant
from sympy.polys.rationaltools import together
from sympy.polys.specialpolys import cyclotomic_poly
from sympy.utilities import public
from sympy.utilities.misc import filldedent



z = Symbol('z')  # importing from abc cause O to be lost as clashing symbol


def roots_linear(f):
    """Returns a list of roots of a linear polynomial."""
    r = -f.nth(0)/f.nth(1)
    dom = f.get_domain()

    if not dom.is_Numerical:
        if dom.is_Composite:
            r = factor(r)
        else:
            from sympy.simplify.simplify import simplify
            r = simplify(r)

    return [r]


def roots_quadratic(f):
    """Returns a list of roots of a quadratic polynomial. If the domain is ZZ
    then the roots will be sorted with negatives coming before positives.
    The ordering will be the same for any numerical coefficients as long as
    the assumptions tested are correct, otherwise the ordering will not be
    sorted (but will be canonical).
    """

    a, b, c = f.all_coeffs()
    dom = f.get_domain()

    def _sqrt(d):
        # remove squares from square root since both will be represented
        # in the results; a similar thing is happening in roots() but
        # must be duplicated here because not all quadratics are binomials
        co = []
        other = []
        for di in Mul.make_args(d):
            if di.is_Pow and di.exp.is_Integer and di.exp % 2 == 0:
                co.append(Pow(di.base, di.exp//2))
            else:
                other.append(di)
        if co:
            d = Mul(*other)
            co = Mul(*co)
            return co*sqrt(d)
        return sqrt(d)

    def _simplify(expr):
        if dom.is_Composite:
            return factor(expr)
        else:
            from sympy.simplify.simplify import simplify
            return simplify(expr)

    if c is S.Zero:
        r0, r1 = S.Zero, -b/a

        if not dom.is_Numerical:
            r1 = _simplify(r1)
        elif r1.is_negative:
            r0, r1 = r1, r0
    elif b is S.Zero:
        r = -c/a
        if not dom.is_Numerical:
            r = _simplify(r)

        R = _sqrt(r)
        r0 = -R
        r1 = R
    else:
        d = b**2 - 4*a*c
        A = 2*a
        B = -b/A

        if not dom.is_Numerical:
            d = _simplify(d)
            B = _simplify(B)

        D = factor_terms(_sqrt(d)/A)
        r0 = B - D
        r1 = B + D
        if a.is_negative:
            r0, r1 = r1, r0
        elif not dom.is_Numerical:
            r0, r1 = [expand_2arg(i) for i in (r0, r1)]

    return [r0, r1]


def roots_cubic(f, trig=False):
    """Returns a list of roots of a cubic polynomial.

    References
    ==========
    [1] https://en.wikipedia.org/wiki/Cubic_function, General formula for roots,
    (accessed November 17, 2014).
    """
    if trig:
        a, b, c, d = f.all_coeffs()
        p = (3*a*c - b**2)/(3*a**2)
        q = (2*b**3 - 9*a*b*c + 27*a**2*d)/(27*a**3)
        D = 18*a*b*c*d - 4*b**3*d + b**2*c**2 - 4*a*c**3 - 27*a**2*d**2
        if (D > 0) == True:
            rv = []
            for k in range(3):
                rv.append(2*sqrt(-p/3)*cos(acos(q/p*sqrt(-3/p)*Rational(3, 2))/3 - k*pi*Rational(2, 3)))
            return [i - b/3/a for i in rv]

    # a*x**3 + b*x**2 + c*x + d -> x**3 + a*x**2 + b*x + c
    _, a, b, c = f.monic().all_coeffs()

    if c is S.Zero:
        x1, x2 = roots([1, a, b], multiple=True)
        return [x1, S.Zero, x2]

    # x**3 + a*x**2 + b*x + c -> u**3 + p*u + q
    p = b - a**2/3
    q = c - a*b/3 + 2*a**3/27

    pon3 = p/3
    aon3 = a/3

    u1 = None
    if p is S.Zero:
        if q is S.Zero:
            return [-aon3]*3
        u1 = -root(q, 3) if q.is_positive else root(-q, 3)
    elif q is S.Zero:
        y1, y2 = roots([1, 0, p], multiple=True)
        return [tmp - aon3 for tmp in [y1, S.Zero, y2]]
    elif q.is_real and q.is_negative:
        u1 = -root(-q/2 + sqrt(q**2/4 + pon3**3), 3)

    coeff = I*sqrt(3)/2
    if u1 is None:
        u1 = S.One
        u2 = Rational(-1, 2) + coeff
        u3 = Rational(-1, 2) - coeff
        b, c, d = a, b, c  # a, b, c, d = S.One, a, b, c
        D0 = b**2 - 3*c  # b**2 - 3*a*c
        D1 = 2*b**3 - 9*b*c + 27*d  # 2*b**3 - 9*a*b*c + 27*a**2*d
        C = root((D1 + sqrt(D1**2 - 4*D0**3))/2, 3)
        return [-(b + uk*C + D0/C/uk)/3 for uk in [u1, u2, u3]]  # -(b + uk*C + D0/C/uk)/3/a

    u2 = u1*(Rational(-1, 2) + coeff)
    u3 = u1*(Rational(-1, 2) - coeff)

    if p is S.Zero:
        return [u1 - aon3, u2 - aon3, u3 - aon3]

    soln = [
        -u1 + pon3/u1 - aon3,
        -u2 + pon3/u2 - aon3,
        -u3 + pon3/u3 - aon3
    ]

    return soln

def _roots_quartic_euler(p, q, r, a):
    """
    Descartes-Euler solution of the quartic equation

    Parameters
    ==========

    p, q, r: coefficients of ``x**4 + p*x**2 + q*x + r``
    a: shift of the roots

    Notes
    =====

    This is a helper function for ``roots_quartic``.

    Look for solutions of the form ::

      ``x1 = sqrt(R) - sqrt(A + B*sqrt(R))``
      ``x2 = -sqrt(R) - sqrt(A - B*sqrt(R))``
      ``x3 = -sqrt(R) + sqrt(A - B*sqrt(R))``
      ``x4 = sqrt(R) + sqrt(A + B*sqrt(R))``

    To satisfy the quartic equation one must have
    ``p = -2*(R + A); q = -4*B*R; r = (R - A)**2 - B**2*R``
    so that ``R`` must satisfy the Descartes-Euler resolvent equation
    ``64*R**3 + 32*p*R**2 + (4*p**2 - 16*r)*R - q**2 = 0``

    If the resolvent does not have a rational solution, return None;
    in that case it is likely that the Ferrari method gives a simpler
    solution.

    Examples
    ========

    >>> from sympy import S
    >>> from sympy.polys.polyroots import _roots_quartic_euler
    >>> p, q, r = -S(64)/5, -S(512)/125, -S(1024)/3125
    >>> _roots_quartic_euler(p, q, r, S(0))[0]
    -sqrt(32*sqrt(5)/125 + 16/5) + 4*sqrt(5)/5
    """
    # solve the resolvent equation
    x = Dummy('x')
    eq = 64*x**3 + 32*p*x**2 + (4*p**2 - 16*r)*x - q**2
    xsols = list(roots(Poly(eq, x), cubics=False).keys())
    xsols = [sol for sol in xsols if sol.is_rational and sol.is_nonzero]
    if not xsols:
        return None
    R = max(xsols)
    c1 = sqrt(R)
    B = -q*c1/(4*R)
    A = -R - p/2
    c2 = sqrt(A + B)
    c3 = sqrt(A - B)
    return [c1 - c2 - a, -c1 - c3 - a, -c1 + c3 - a, c1 + c2 - a]


def roots_quartic(f):
    r"""
    Returns a list of roots of a quartic polynomial.

    There are many references for solving quartic expressions available [1-5].
    This reviewer has found that many of them require one to select from among
    2 or more possible sets of solutions and that some solutions work when one
    is searching for real roots but do not work when searching for complex roots
    (though this is not always stated clearly). The following routine has been
    tested and found to be correct for 0, 2 or 4 complex roots.

    The quasisymmetric case solution [6] looks for quartics that have the form
    `x**4 + A*x**3 + B*x**2 + C*x + D = 0` where `(C/A)**2 = D`.

    Although no general solution that is always applicable for all
    coefficients is known to this reviewer, certain conditions are tested
    to determine the simplest 4 expressions that can be returned:

      1) `f = c + a*(a**2/8 - b/2) == 0`
      2) `g = d - a*(a*(3*a**2/256 - b/16) + c/4) = 0`
      3) if `f != 0` and `g != 0` and `p = -d + a*c/4 - b**2/12` then
        a) `p == 0`
        b) `p != 0`

    Examples
    ========

        >>> from sympy import Poly
        >>> from sympy.polys.polyroots import roots_quartic

        >>> r = roots_quartic(Poly('x**4-6*x**3+17*x**2-26*x+20'))

        >>> # 4 complex roots: 1+-I*sqrt(3), 2+-I
        >>> sorted(str(tmp.evalf(n=2)) for tmp in r)
        ['1.0 + 1.7*I', '1.0 - 1.7*I', '2.0 + 1.0*I', '2.0 - 1.0*I']

    References
    ==========

    1. http://mathforum.org/dr.math/faq/faq.cubic.equations.html
    2. https://en.wikipedia.org/wiki/Quartic_function#Summary_of_Ferrari.27s_method
    3. https://planetmath.org/encyclopedia/GaloisTheoreticDerivationOfTheQuarticFormula.html
    4. https://people.bath.ac.uk/masjhd/JHD-CA.pdf
    5. http://www.albmath.org/files/Math_5713.pdf
    6. https://web.archive.org/web/20171002081448/http://www.statemaster.com/encyclopedia/Quartic-equation
    7. https://eqworld.ipmnet.ru/en/solutions/ae/ae0108.pdf
    """
    _, a, b, c, d = f.monic().all_coeffs()

    if not d:
        return [S.Zero] + roots([1, a, b, c], multiple=True)
    elif (c/a)**2 == d:
        x, m = f.gen, c/a

        g = Poly(x**2 + a*x + b - 2*m, x)

        z1, z2 = roots_quadratic(g)

        h1 = Poly(x**2 - z1*x + m, x)
        h2 = Poly(x**2 - z2*x + m, x)

        r1 = roots_quadratic(h1)
        r2 = roots_quadratic(h2)

        return r1 + r2
    else:
        a2 = a**2
        e = b - 3*a2/8
        f = _mexpand(c + a*(a2/8 - b/2))
        aon4 = a/4
        g = _mexpand(d - aon4*(a*(3*a2/64 - b/4) + c))

        if f.is_zero:
            y1, y2 = [sqrt(tmp) for tmp in
                      roots([1, e, g], multiple=True)]
            return [tmp - aon4 for tmp in [-y1, -y2, y1, y2]]
        if g.is_zero:
            y = [S.Zero] + roots([1, 0, e, f], multiple=True)
            return [tmp - aon4 for tmp in y]
        else:
            # Descartes-Euler method, see [7]
            sols = _roots_quartic_euler(e, f, g, aon4)
            if sols:
                return sols
            # Ferrari method, see [1, 2]
            p = -e**2/12 - g
            q = -e**3/108 + e*g/3 - f**2/8
            TH = Rational(1, 3)

            def _ans(y):
                w = sqrt(e + 2*y)
                arg1 = 3*e + 2*y
                arg2 = 2*f/w
                ans = []
                for s in [-1, 1]:
                    root = sqrt(-(arg1 + s*arg2))
                    for t in [-1, 1]:
                        ans.append((s*w - t*root)/2 - aon4)
                return ans

            # whether a Piecewise is returned or not
            # depends on knowing p, so try to put
            # in a simple form
            p = _mexpand(p)


            # p == 0 case
            y1 = e*Rational(-5, 6) - q**TH
            if p.is_zero:
                return _ans(y1)

            # if p != 0 then u below is not 0
            root = sqrt(q**2/4 + p**3/27)
            r = -q/2 + root  # or -q/2 - root
            u = r**TH  # primary root of solve(x**3 - r, x)
            y2 = e*Rational(-5, 6) + u - p/u/3
            if fuzzy_not(p.is_zero):
                return _ans(y2)

            # sort it out once they know the values of the coefficients
            return [Piecewise((a1, Eq(p, 0)), (a2, True))
                    for a1, a2 in zip(_ans(y1), _ans(y2))]


def roots_binomial(f):
    """Returns a list of roots of a binomial polynomial. If the domain is ZZ
    then the roots will be sorted with negatives coming before positives.
    The ordering will be the same for any numerical coefficients as long as
    the assumptions tested are correct, otherwise the ordering will not be
    sorted (but will be canonical).
    """
    n = f.degree()

    a, b = f.nth(n), f.nth(0)
    base = -cancel(b/a)
    alpha = root(base, n)

    if alpha.is_number:
        alpha = alpha.expand(complex=True)

    # define some parameters that will allow us to order the roots.
    # If the domain is ZZ this is guaranteed to return roots sorted
    # with reals before non-real roots and non-real sorted according
    # to real part and imaginary part, e.g. -1, 1, -1 + I, 2 - I
    neg = base.is_negative
    even = n % 2 == 0
    if neg:
        if even == True and (base + 1).is_positive:
            big = True
        else:
            big = False

    # get the indices in the right order so the computed
    # roots will be sorted when the domain is ZZ
    ks = []
    imax = n//2
    if even:
        ks.append(imax)
        imax -= 1
    if not neg:
        ks.append(0)
    for i in range(imax, 0, -1):
        if neg:
            ks.extend([i, -i])
        else:
            ks.extend([-i, i])
    if neg:
        ks.append(0)
        if big:
            for i in range(0, len(ks), 2):
                pair = ks[i: i + 2]
                pair = list(reversed(pair))

    # compute the roots
    roots, d = [], 2*I*pi/n
    for k in ks:
        zeta = exp(k*d).expand(complex=True)
        roots.append((alpha*zeta).expand(power_base=False))

    return roots


def _inv_totient_estimate(m):
    """
    Find ``(L, U)`` such that ``L <= phi^-1(m) <= U``.

    Examples
    ========

    >>> from sympy.polys.polyroots import _inv_totient_estimate

    >>> _inv_totient_estimate(192)
    (192, 840)
    >>> _inv_totient_estimate(400)
    (400, 1750)

    """
    primes = [ d + 1 for d in divisors(m) if isprime(d + 1) ]

    a, b = 1, 1

    for p in primes:
        a *= p
        b *= p - 1

    L = m
    U = int(math.ceil(m*(float(a)/b)))

    P = p = 2
    primes = []

    while P <= U:
        p = nextprime(p)
        primes.append(p)
        P *= p

    P //= p
    b = 1

    for p in primes[:-1]:
        b *= p - 1

    U = int(math.ceil(m*(float(P)/b)))

    return L, U


def roots_cyclotomic(f, factor=False):
    """Compute roots of cyclotomic polynomials. """
    L, U = _inv_totient_estimate(f.degree())

    for n in range(L, U + 1):
        g = cyclotomic_poly(n, f.gen, polys=True)

        if f.expr == g.expr:
            break
    else:  # pragma: no cover
        raise RuntimeError("failed to find index of a cyclotomic polynomial")

    roots = []

    if not factor:
        # get the indices in the right order so the computed
        # roots will be sorted
        h = n//2
        ks = [i for i in range(1, n + 1) if igcd(i, n) == 1]
        ks.sort(key=lambda x: (x, -1) if x <= h else (abs(x - n), 1))
        d = 2*I*pi/n
        for k in reversed(ks):
            roots.append(exp(k*d).expand(complex=True))
    else:
        g = Poly(f, extension=root(-1, n))

        for h, _ in ordered(g.factor_list()[1]):
            roots.append(-h.TC())

    return roots


def roots_quintic(f):
    """
    Calculate exact roots of a solvable irreducible quintic with rational coefficients.
    Return an empty list if the quintic is reducible or not solvable.
    """
    result = []

    coeff_5, coeff_4, p_, q_, r_, s_ = f.all_coeffs()

    if not all(coeff.is_Rational for coeff in (coeff_5, coeff_4, p_, q_, r_, s_)):
        return result

    if coeff_5 != 1:
        f = Poly(f / coeff_5)
        _, coeff_4, p_, q_, r_, s_ = f.all_coeffs()

    # Cancel coeff_4 to form x^5 + px^3 + qx^2 + rx + s
    if coeff_4:
        p = p_ - 2*coeff_4*coeff_4/5
        q = q_ - 3*coeff_4*p_/5 + 4*coeff_4**3/25
        r = r_ - 2*coeff_4*q_/5 + 3*coeff_4**2*p_/25 - 3*coeff_4**4/125
        s = s_ - coeff_4*r_/5 + coeff_4**2*q_/25 - coeff_4**3*p_/125 + 4*coeff_4**5/3125
        x = f.gen
        f = Poly(x**5 + p*x**3 + q*x**2 + r*x + s)
    else:
        p, q, r, s = p_, q_, r_, s_

    quintic = PolyQuintic(f)

    # Eqn standardized. Algo for solving starts here
    if not f.is_irreducible:
        return result
    f20 = quintic.f20
    # Check if f20 has linear factors over domain Z
    if f20.is_irreducible:
        return result
    # Now, we know that f is solvable
    for _factor in f20.factor_list()[1]:
        if _factor[0].is_linear:
            theta = _factor[0].root(0)
            break
    d = discriminant(f)
    delta = sqrt(d)
    # zeta = a fifth root of unity
    zeta1, zeta2, zeta3, zeta4 = quintic.zeta
    T = quintic.T(theta, d)
    tol = S(1e-10)
    alpha = T[1] + T[2]*delta
    alpha_bar = T[1] - T[2]*delta
    beta = T[3] + T[4]*delta
    beta_bar = T[3] - T[4]*delta

    disc = alpha**2 - 4*beta
    disc_bar = alpha_bar**2 - 4*beta_bar

    l0 = quintic.l0(theta)
    Stwo = S(2)
    l1 = _quintic_simplify((-alpha + sqrt(disc)) / Stwo)
    l4 = _quintic_simplify((-alpha - sqrt(disc)) / Stwo)

    l2 = _quintic_simplify((-alpha_bar + sqrt(disc_bar)) / Stwo)
    l3 = _quintic_simplify((-alpha_bar - sqrt(disc_bar)) / Stwo)

    order = quintic.order(theta, d)
    test = (order*delta.n()) - ( (l1.n() - l4.n())*(l2.n() - l3.n()) )
    # Comparing floats
    if not comp(test, 0, tol):
        l2, l3 = l3, l2

    # Now we have correct order of l's
    R1 = l0 + l1*zeta1 + l2*zeta2 + l3*zeta3 + l4*zeta4
    R2 = l0 + l3*zeta1 + l1*zeta2 + l4*zeta3 + l2*zeta4
    R3 = l0 + l2*zeta1 + l4*zeta2 + l1*zeta3 + l3*zeta4
    R4 = l0 + l4*zeta1 + l3*zeta2 + l2*zeta3 + l1*zeta4

    Res = [None, [None]*5, [None]*5, [None]*5, [None]*5]
    Res_n = [None, [None]*5, [None]*5, [None]*5, [None]*5]

    # Simplifying improves performance a lot for exact expressions
    R1 = _quintic_simplify(R1)
    R2 = _quintic_simplify(R2)
    R3 = _quintic_simplify(R3)
    R4 = _quintic_simplify(R4)

    # hard-coded results for [factor(i) for i in _vsolve(x**5 - a - I*b, x)]
    x0 = z**(S(1)/5)
    x1 = sqrt(2)
    x2 = sqrt(5)
    x3 = sqrt(5 - x2)
    x4 = I*x2
    x5 = x4 + I
    x6 = I*x0/4
    x7 = x1*sqrt(x2 + 5)
    sol = [x0, -x6*(x1*x3 - x5), x6*(x1*x3 + x5), -x6*(x4 + x7 - I), x6*(-x4 + x7 + I)]

    R1 = R1.as_real_imag()
    R2 = R2.as_real_imag()
    R3 = R3.as_real_imag()
    R4 = R4.as_real_imag()

    for i, s in enumerate(sol):
        Res[1][i] = _quintic_simplify(s.xreplace({z: R1[0] + I*R1[1]}))
        Res[2][i] = _quintic_simplify(s.xreplace({z: R2[0] + I*R2[1]}))
        Res[3][i] = _quintic_simplify(s.xreplace({z: R3[0] + I*R3[1]}))
        Res[4][i] = _quintic_simplify(s.xreplace({z: R4[0] + I*R4[1]}))

    for i in range(1, 5):
        for j in range(5):
            Res_n[i][j] = Res[i][j].n()
            Res[i][j] = _quintic_simplify(Res[i][j])
    r1 = Res[1][0]
    r1_n = Res_n[1][0]

    for i in range(5):
        if comp(im(r1_n*Res_n[4][i]), 0, tol):
            r4 = Res[4][i]
            break

    # Now we have various Res values. Each will be a list of five
    # values. We have to pick one r value from those five for each Res
    u, v = quintic.uv(theta, d)
    testplus = (u + v*delta*sqrt(5)).n()
    testminus = (u - v*delta*sqrt(5)).n()

    # Evaluated numbers suffixed with _n
    # We will use evaluated numbers for calculation. Much faster.
    r4_n = r4.n()
    r2 = r3 = None

    for i in range(5):
        r2temp_n = Res_n[2][i]
        for j in range(5):
            # Again storing away the exact number and using
            # evaluated numbers in computations
            r3temp_n = Res_n[3][j]
            if (comp((r1_n*r2temp_n**2 + r4_n*r3temp_n**2 - testplus).n(), 0, tol) and
                comp((r3temp_n*r1_n**2 + r2temp_n*r4_n**2 - testminus).n(), 0, tol)):
                r2 = Res[2][i]
                r3 = Res[3][j]
                break
        if r2 is not None:
            break
    else:
        return []  # fall back to normal solve

    # Now, we have r's so we can get roots
    x1 = (r1 + r2 + r3 + r4)/5
    x2 = (r1*zeta4 + r2*zeta3 + r3*zeta2 + r4*zeta1)/5
    x3 = (r1*zeta3 + r2*zeta1 + r3*zeta4 + r4*zeta2)/5
    x4 = (r1*zeta2 + r2*zeta4 + r3*zeta1 + r4*zeta3)/5
    x5 = (r1*zeta1 + r2*zeta2 + r3*zeta3 + r4*zeta4)/5
    result = [x1, x2, x3, x4, x5]

    # Now check if solutions are distinct

    saw = set()
    for r in result:
        r = r.n(2)
        if r in saw:
            # Roots were identical. Abort, return []
            # and fall back to usual solve
            return []
        saw.add(r)

    # Restore to original equation where coeff_4 is nonzero
    if coeff_4:
        result = [x - coeff_4 / 5 for x in result]
    return result


def _quintic_simplify(expr):
    from sympy.simplify.simplify import powsimp
    expr = powsimp(expr)
    expr = cancel(expr)
    return together(expr)


def _integer_basis(poly):
    """Compute coefficient basis for a polynomial over integers.

    Returns the integer ``div`` such that substituting ``x = div*y``
    ``p(x) = m*q(y)`` where the coefficients of ``q`` are smaller
    than those of ``p``.

    For example ``x**5 + 512*x + 1024 = 0``
    with ``div = 4`` becomes ``y**5 + 2*y + 1 = 0``

    Returns the integer ``div`` or ``None`` if there is no possible scaling.

    Examples
    ========

    >>> from sympy.polys import Poly
    >>> from sympy.abc import x
    >>> from sympy.polys.polyroots import _integer_basis
    >>> p = Poly(x**5 + 512*x + 1024, x, domain='ZZ')
    >>> _integer_basis(p)
    4
    """
    monoms, coeffs = list(zip(*poly.terms()))

    monoms, = list(zip(*monoms))
    coeffs = list(map(abs, coeffs))

    if coeffs[0] < coeffs[-1]:
        coeffs = list(reversed(coeffs))
        n = monoms[0]
        monoms = [n - i for i in reversed(monoms)]
    else:
        return None

    monoms = monoms[:-1]
    coeffs = coeffs[:-1]

    # Special case for two-term polynominals
    if len(monoms) == 1:
        r = Pow(coeffs[0], S.One/monoms[0])
        if r.is_Integer:
            return int(r)
        else:
            return None

    divs = reversed(divisors(gcd_list(coeffs))[1:])

    try:
        div = next(divs)
    except StopIteration:
        return None

    while True:
        for monom, coeff in zip(monoms, coeffs):
            if coeff % div**monom != 0:
                try:
                    div = next(divs)
                except StopIteration:
                    return None
                else:
                    break
        else:
            return div


def preprocess_roots(poly):
    """Try to get rid of symbolic coefficients from ``poly``. """
    coeff = S.One

    poly_func = poly.func
    try:
        _, poly = poly.clear_denoms(convert=True)
    except DomainError:
        return coeff, poly

    poly = poly.primitive()[1]
    poly = poly.retract()

    # TODO: This is fragile. Figure out how to make this independent of construct_domain().
    if poly.get_domain().is_Poly and all(c.is_term for c in poly.rep.coeffs()):
        poly = poly.inject()

        strips = list(zip(*poly.monoms()))
        gens = list(poly.gens[1:])

        base, strips = strips[0], strips[1:]

        for gen, strip in zip(list(gens), strips):
            reverse = False

            if strip[0] < strip[-1]:
                strip = reversed(strip)
                reverse = True

            ratio = None

            for a, b in zip(base, strip):
                if not a and not b:
                    continue
                elif not a or not b:
                    break
                elif b % a != 0:
                    break
                else:
                    _ratio = b // a

                    if ratio is None:
                        ratio = _ratio
                    elif ratio != _ratio:
                        break
            else:
                if reverse:
                    ratio = -ratio

                poly = poly.eval(gen, 1)
                coeff *= gen**(-ratio)
                gens.remove(gen)

        if gens:
            poly = poly.eject(*gens)

    if poly.is_univariate and poly.get_domain().is_ZZ:
        basis = _integer_basis(poly)

        if basis is not None:
            n = poly.degree()

            def func(k, coeff):
                return coeff//basis**(n - k[0])

            poly = poly.termwise(func)
            coeff *= basis

    if not isinstance(poly, poly_func):
        poly = poly_func(poly)
    return coeff, poly


@public
def roots(f, *gens,
        auto=True,
        cubics=True,
        trig=False,
        quartics=True,
        quintics=False,
        multiple=False,
        filter=None,
        predicate=None,
        strict=False,
        **flags):
    """
    Computes symbolic roots of a univariate polynomial.

    Given a univariate polynomial f with symbolic coefficients (or
    a list of the polynomial's coefficients), returns a dictionary
    with its roots and their multiplicities.

    Only roots expressible via radicals will be returned.  To get
    a complete set of roots use RootOf class or numerical methods
    instead. By default cubic and quartic formulas are used in
    the algorithm. To disable them because of unreadable output
    set ``cubics=False`` or ``quartics=False`` respectively. If cubic
    roots are real but are expressed in terms of complex numbers
    (casus irreducibilis [1]) the ``trig`` flag can be set to True to
    have the solutions returned in terms of cosine and inverse cosine
    functions.

    To get roots from a specific domain set the ``filter`` flag with
    one of the following specifiers: Z, Q, R, I, C. By default all
    roots are returned (this is equivalent to setting ``filter='C'``).

    By default a dictionary is returned giving a compact result in
    case of multiple roots.  However to get a list containing all
    those roots set the ``multiple`` flag to True; the list will
    have identical roots appearing next to each other in the result.
    (For a given Poly, the all_roots method will give the roots in
    sorted numerical order.)

    If the ``strict`` flag is True, ``UnsolvableFactorError`` will be
    raised if the roots found are known to be incomplete (because
    some roots are not expressible in radicals).

    Examples
    ========

    >>> from sympy import Poly, roots, degree
    >>> from sympy.abc import x, y

    >>> roots(x**2 - 1, x)
    {-1: 1, 1: 1}

    >>> p = Poly(x**2-1, x)
    >>> roots(p)
    {-1: 1, 1: 1}

    >>> p = Poly(x**2-y, x, y)

    >>> roots(Poly(p, x))
    {-sqrt(y): 1, sqrt(y): 1}

    >>> roots(x**2 - y, x)
    {-sqrt(y): 1, sqrt(y): 1}

    >>> roots([1, 0, -1])
    {-1: 1, 1: 1}

    ``roots`` will only return roots expressible in radicals. If
    the given polynomial has some or all of its roots inexpressible in
    radicals, the result of ``roots`` will be incomplete or empty
    respectively.

    Example where result is incomplete:

    >>> roots((x-1)*(x**5-x+1), x)
    {1: 1}

    In this case, the polynomial has an unsolvable quintic factor
    whose roots cannot be expressed by radicals. The polynomial has a
    rational root (due to the factor `(x-1)`), which is returned since
    ``roots`` always finds all rational roots.

    Example where result is empty:

    >>> roots(x**7-3*x**2+1, x)
    {}

    Here, the polynomial has no roots expressible in radicals, so
    ``roots`` returns an empty dictionary.

    The result produced by ``roots`` is complete if and only if the
    sum of the multiplicity of each root is equal to the degree of
    the polynomial. If strict=True, UnsolvableFactorError will be
    raised if the result is incomplete.

    The result can be be checked for completeness as follows:

    >>> f = x**3-2*x**2+1
    >>> sum(roots(f, x).values()) == degree(f, x)
    True
    >>> f = (x-1)*(x**5-x+1)
    >>> sum(roots(f, x).values()) == degree(f, x)
    False


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Cubic_equation#Trigonometric_and_hyperbolic_solutions

    """
    from sympy.polys.polytools import to_rational_coeffs
    flags = dict(flags)

    if isinstance(f, list):
        if gens:
            raise ValueError('redundant generators given')

        x = Dummy('x')

        poly, i = {}, len(f) - 1

        for coeff in f:
            poly[i], i = sympify(coeff), i - 1

        f = Poly(poly, x, field=True)
    else:
        try:
            F = Poly(f, *gens, **flags)
            if not isinstance(f, Poly) and not F.gen.is_Symbol:
                raise PolynomialError("generator must be a Symbol")
            f = F
        except GeneratorsNeeded:
            if multiple:
                return []
            else:
                return {}
        else:
            n = f.degree()
            if f.length() == 2 and n > 2:
                # check for foo**n in constant if dep is c*gen**m
                con, dep = f.as_expr().as_independent(*f.gens)
                fcon = -(-con).factor()
                if fcon != con:
                    con = fcon
                    bases = []
                    for i in Mul.make_args(con):
                        if i.is_Pow:
                            b, e = i.as_base_exp()
                            if e.is_Integer and b.is_Add:
                                bases.append((b, Dummy(positive=True)))
                    if bases:
                        rv = roots(Poly((dep + con).xreplace(dict(bases)),
                            *f.gens), *F.gens,
                            auto=auto,
                            cubics=cubics,
                            trig=trig,
                            quartics=quartics,
                            quintics=quintics,
                            multiple=multiple,
                            filter=filter,
                            predicate=predicate,
                            **flags)
                        return {factor_terms(k.xreplace(
                            {v: k for k, v in bases})
                        ): v for k, v in rv.items()}

        if f.is_multivariate:
            raise PolynomialError('multivariate polynomials are not supported')

    def _update_dict(result, zeros, currentroot, k):
        if currentroot == S.Zero:
            if S.Zero in zeros:
                zeros[S.Zero] += k
            else:
                zeros[S.Zero] = k
        if currentroot in result:
            result[currentroot] += k
        else:
            result[currentroot] = k

    def _try_decompose(f):
        """Find roots using functional decomposition. """
        factors, roots = f.decompose(), []

        for currentroot in _try_heuristics(factors[0]):
            roots.append(currentroot)

        for currentfactor in factors[1:]:
            previous, roots = list(roots), []

            for currentroot in previous:
                g = currentfactor - Poly(currentroot, f.gen)

                for currentroot in _try_heuristics(g):
                    roots.append(currentroot)

        return roots

    def _try_heuristics(f):
        """Find roots using formulas and some tricks. """
        if f.is_ground:
            return []
        if f.is_monomial:
            return [S.Zero]*f.degree()

        if f.length() == 2:
            if f.degree() == 1:
                return list(map(cancel, roots_linear(f)))
            else:
                return roots_binomial(f)

        result = []

        for i in [-1, 1]:
            if not f.eval(i):
                f = f.quo(Poly(f.gen - i, f.gen))
                result.append(i)
                break

        n = f.degree()

        if n == 1:
            result += list(map(cancel, roots_linear(f)))
        elif n == 2:
            result += list(map(cancel, roots_quadratic(f)))
        elif f.is_cyclotomic:
            result += roots_cyclotomic(f)
        elif n == 3 and cubics:
            result += roots_cubic(f, trig=trig)
        elif n == 4 and quartics:
            result += roots_quartic(f)
        elif n == 5 and quintics:
            result += roots_quintic(f)

        return result

    # Convert the generators to symbols
    dumgens = symbols('x:%d' % len(f.gens), cls=Dummy)
    f = f.per(f.rep, dumgens)

    (k,), f = f.terms_gcd()

    if not k:
        zeros = {}
    else:
        zeros = {S.Zero: k}

    coeff, f = preprocess_roots(f)

    if auto and f.get_domain().is_Ring:
        f = f.to_field()

    # Use EX instead of ZZ_I or QQ_I
    if f.get_domain().is_QQ_I:
        f = f.per(f.rep.convert(EX))

    rescale_x = None
    translate_x = None

    result = {}

    if not f.is_ground:
        dom = f.get_domain()
        if not dom.is_Exact and dom.is_Numerical:
            for r in f.nroots():
                _update_dict(result, zeros, r, 1)
        elif f.degree() == 1:
            _update_dict(result, zeros, roots_linear(f)[0], 1)
        elif f.length() == 2:
            roots_fun = roots_quadratic if f.degree() == 2 else roots_binomial
            for r in roots_fun(f):
                _update_dict(result, zeros, r, 1)
        else:
            _, factors = Poly(f.as_expr()).factor_list()
            if len(factors) == 1 and f.degree() == 2:
                for r in roots_quadratic(f):
                    _update_dict(result, zeros, r, 1)
            else:
                if len(factors) == 1 and factors[0][1] == 1:
                    if f.get_domain().is_EX:
                        res = to_rational_coeffs(f)
                        if res:
                            if res[0] is None:
                                translate_x, f = res[2:]
                            else:
                                rescale_x, f = res[1], res[-1]
                            result = roots(f)
                            if not result:
                                for currentroot in _try_decompose(f):
                                    _update_dict(result, zeros, currentroot, 1)
                        else:
                            for r in _try_heuristics(f):
                                _update_dict(result, zeros, r, 1)
                    else:
                        for currentroot in _try_decompose(f):
                            _update_dict(result, zeros, currentroot, 1)
                else:
                    for currentfactor, k in factors:
                        for r in _try_heuristics(Poly(currentfactor, f.gen, field=True)):
                            _update_dict(result, zeros, r, k)

    if coeff is not S.One:
        _result, result, = result, {}

        for currentroot, k in _result.items():
            result[coeff*currentroot] = k

    if filter not in [None, 'C']:
        handlers = {
            'Z': lambda r: r.is_Integer,
            'Q': lambda r: r.is_Rational,
            'R': lambda r: all(a.is_real for a in r.as_numer_denom()),
            'I': lambda r: r.is_imaginary,
        }

        try:
            query = handlers[filter]
        except KeyError:
            raise ValueError("Invalid filter: %s" % filter)

        for zero in dict(result).keys():
            if not query(zero):
                del result[zero]

    if predicate is not None:
        for zero in dict(result).keys():
            if not predicate(zero):
                del result[zero]
    if rescale_x:
        result1 = {}
        for k, v in result.items():
            result1[k*rescale_x] = v
        result = result1
    if translate_x:
        result1 = {}
        for k, v in result.items():
            result1[k + translate_x] = v
        result = result1

    # adding zero roots after non-trivial roots have been translated
    result.update(zeros)

    if strict and sum(result.values()) < f.degree():
        raise UnsolvableFactorError(filldedent('''
            Strict mode: some factors cannot be solved in radicals, so
            a complete list of solutions cannot be returned. Call
            roots with strict=False to get solutions expressible in
            radicals (if there are any).
            '''))

    if not multiple:
        return result
    else:
        zeros = []

        for zero in ordered(result):
            zeros.extend([zero]*result[zero])

        return zeros


def root_factors(f, *gens, filter=None, **args):
    """
    Returns all factors of a univariate polynomial.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.polys.polyroots import root_factors

    >>> root_factors(x**2 - y, x)
    [x - sqrt(y), x + sqrt(y)]

    """
    args = dict(args)

    F = Poly(f, *gens, **args)

    if not F.is_Poly:
        return [f]

    if F.is_multivariate:
        raise ValueError('multivariate polynomials are not supported')

    x = F.gens[0]

    zeros = roots(F, filter=filter)

    if not zeros:
        factors = [F]
    else:
        factors, N = [], 0

        for r, n in ordered(zeros.items()):
            factors, N = factors + [Poly(x - r, x)]*n, N + n

        if N < F.degree():
            G = reduce(lambda p, q: p*q, factors)
            factors.append(F.quo(G))

    if not isinstance(f, Poly):
        factors = [ f.as_expr() for f in factors ]

    return factors
<|endoftext|>"""Tools and arithmetics for monomials of distributed polynomials. """


from itertools import combinations_with_replacement, product
from textwrap import dedent

from sympy.core.cache import cacheit
from sympy.core import Mul, S, Tuple, sympify
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import PicklableWithSlots, dict_from_expr
from sympy.utilities import public
from sympy.utilities.iterables import is_sequence, iterable

@public
def itermonomials(variables, max_degrees, min_degrees=None):
    r"""
    ``max_degrees`` and ``min_degrees`` are either both integers or both lists.
    Unless otherwise specified, ``min_degrees`` is either ``0`` or
    ``[0, ..., 0]``.

    A generator of all monomials ``monom`` is returned, such that
    either
    ``min_degree <= total_degree(monom) <= max_degree``,
    or
    ``min_degrees[i] <= degree_list(monom)[i] <= max_degrees[i]``,
    for all ``i``.

    Case I. ``max_degrees`` and ``min_degrees`` are both integers
    =============================================================

    Given a set of variables $V$ and a min_degree $N$ and a max_degree $M$
    generate a set of monomials of degree less than or equal to $N$ and greater
    than or equal to $M$. The total number of monomials in commutative
    variables is huge and is given by the following formula if $M = 0$:

        .. math::
            \frac{(\#V + N)!}{\#V! N!}

    For example if we would like to generate a dense polynomial of
    a total degree $N = 50$ and $M = 0$, which is the worst case, in 5
    variables, assuming that exponents and all of coefficients are 32-bit long
    and stored in an array we would need almost 80 GiB of memory! Fortunately
    most polynomials, that we will encounter, are sparse.

    Consider monomials in commutative variables $x$ and $y$
    and non-commutative variables $a$ and $b$::

        >>> from sympy import symbols
        >>> from sympy.polys.monomials import itermonomials
        >>> from sympy.polys.orderings import monomial_key
        >>> from sympy.abc import x, y

        >>> sorted(itermonomials([x, y], 2), key=monomial_key('grlex', [y, x]))
        [1, x, y, x**2, x*y, y**2]

        >>> sorted(itermonomials([x, y], 3), key=monomial_key('grlex', [y, x]))
        [1, x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3]

        >>> a, b = symbols('a, b', commutative=False)
        >>> set(itermonomials([a, b, x], 2))
        {1, a, a**2, b, b**2, x, x**2, a*b, b*a, x*a, x*b}

        >>> sorted(itermonomials([x, y], 2, 1), key=monomial_key('grlex', [y, x]))
        [x, y, x**2, x*y, y**2]

    Case II. ``max_degrees`` and ``min_degrees`` are both lists
    ===========================================================

    If ``max_degrees = [d_1, ..., d_n]`` and
    ``min_degrees = [e_1, ..., e_n]``, the number of monomials generated
    is:

    .. math::
        (d_1 - e_1 + 1) (d_2 - e_2 + 1) \cdots (d_n - e_n + 1)

    Let us generate all monomials ``monom`` in variables $x$ and $y$
    such that ``[1, 2][i] <= degree_list(monom)[i] <= [2, 4][i]``,
    ``i = 0, 1`` ::

        >>> from sympy import symbols
        >>> from sympy.polys.monomials import itermonomials
        >>> from sympy.polys.orderings import monomial_key
        >>> from sympy.abc import x, y

        >>> sorted(itermonomials([x, y], [2, 4], [1, 2]), reverse=True, key=monomial_key('lex', [x, y]))
        [x**2*y**4, x**2*y**3, x**2*y**2, x*y**4, x*y**3, x*y**2]
    """
    if is_sequence(max_degrees):
        n = len(variables)
        if len(max_degrees) != n:
            raise ValueError('Argument sizes do not match')
        if min_degrees is None:
            min_degrees = [0]*n
        elif not is_sequence(min_degrees):
            raise ValueError('min_degrees is not a list')
        else:
            if len(min_degrees) != n:
                raise ValueError('Argument sizes do not match')
            if any(i < 0 for i in min_degrees):
                raise ValueError("min_degrees cannot contain negative numbers")
        if any(min_degrees[i] > max_degrees[i] for i in range(n)):
            raise ValueError('min_degrees[i] must be <= max_degrees[i] for all i')
        power_lists = []
        for var, min_d, max_d in zip(variables, min_degrees, max_degrees):
            power_lists.append([var**i for i in range(min_d, max_d + 1)])
        for powers in product(*power_lists):
            yield Mul(*powers)
    else:
        max_degree = max_degrees
        if max_degree < 0:
            raise ValueError("max_degrees cannot be negative")
        if min_degrees is None:
            min_degree = 0
        else:
            if min_degrees < 0:
                raise ValueError("min_degrees cannot be negative")
            min_degree = min_degrees
        if min_degree > max_degree:
            return
        if not variables or max_degree == 0:
            yield S.One
            return
        # Force to list in case of passed tuple or other incompatible collection
        variables = list(variables) + [S.One]
        if all(variable.is_commutative for variable in variables):
            it = combinations_with_replacement(variables, max_degree)
        else:
            it = product(variables, repeat=max_degree)
        monomials_set = set()
        d = max_degree - min_degree
        for item in it:
            count = 0
            for variable in item:
                if variable == 1:
                    count += 1
                    if d < count:
                        break
            else:
                monomials_set.add(Mul(*item))
        yield from monomials_set

def monomial_count(V, N):
    r"""
    Computes the number of monomials.

    The number of monomials is given by the following formula:

    .. math::

        \frac{(\#V + N)!}{\#V! N!}

    where `N` is a total degree and `V` is a set of variables.

    Examples
    ========

    >>> from sympy.polys.monomials import itermonomials, monomial_count
    >>> from sympy.polys.orderings import monomial_key
    >>> from sympy.abc import x, y

    >>> monomial_count(2, 2)
    6

    >>> M = list(itermonomials([x, y], 2))

    >>> sorted(M, key=monomial_key('grlex', [y, x]))
    [1, x, y, x**2, x*y, y**2]
    >>> len(M)
    6

    """
    from sympy.functions.combinatorial.factorials import factorial
    return factorial(V + N) / factorial(V) / factorial(N)

def monomial_mul(A, B):
    """
    Multiplication of tuples representing monomials.

    Examples
    ========

    Lets multiply `x**3*y**4*z` with `x*y**2`::

        >>> from sympy.polys.monomials import monomial_mul

        >>> monomial_mul((3, 4, 1), (1, 2, 0))
        (4, 6, 1)

    which gives `x**4*y**5*z`.

    """
    return tuple([ a + b for a, b in zip(A, B) ])

def monomial_div(A, B):
    """
    Division of tuples representing monomials.

    Examples
    ========

    Lets divide `x**3*y**4*z` by `x*y**2`::

        >>> from sympy.polys.monomials import monomial_div

        >>> monomial_div((3, 4, 1), (1, 2, 0))
        (2, 2, 1)

    which gives `x**2*y**2*z`. However::

        >>> monomial_div((3, 4, 1), (1, 2, 2)) is None
        True

    `x*y**2*z**2` does not divide `x**3*y**4*z`.

    """
    C = monomial_ldiv(A, B)

    if all(c >= 0 for c in C):
        return tuple(C)
    else:
        return None

def monomial_ldiv(A, B):
    """
    Division of tuples representing monomials.

    Examples
    ========

    Lets divide `x**3*y**4*z` by `x*y**2`::

        >>> from sympy.polys.monomials import monomial_ldiv

        >>> monomial_ldiv((3, 4, 1), (1, 2, 0))
        (2, 2, 1)

    which gives `x**2*y**2*z`.

        >>> monomial_ldiv((3, 4, 1), (1, 2, 2))
        (2, 2, -1)

    which gives `x**2*y**2*z**-1`.

    """
    return tuple([ a - b for a, b in zip(A, B) ])

def monomial_pow(A, n):
    """Return the n-th pow of the monomial. """
    return tuple([ a*n for a in A ])

def monomial_gcd(A, B):
    """
    Greatest common divisor of tuples representing monomials.

    Examples
    ========

    Lets compute GCD of `x*y**4*z` and `x**3*y**2`::

        >>> from sympy.polys.monomials import monomial_gcd

        >>> monomial_gcd((1, 4, 1), (3, 2, 0))
        (1, 2, 0)

    which gives `x*y**2`.

    """
    return tuple([ min(a, b) for a, b in zip(A, B) ])

def monomial_lcm(A, B):
    """
    Least common multiple of tuples representing monomials.

    Examples
    ========

    Lets compute LCM of `x*y**4*z` and `x**3*y**2`::

        >>> from sympy.polys.monomials import monomial_lcm

        >>> monomial_lcm((1, 4, 1), (3, 2, 0))
        (3, 4, 1)

    which gives `x**3*y**4*z`.

    """
    return tuple([ max(a, b) for a, b in zip(A, B) ])

def monomial_divides(A, B):
    """
    Does there exist a monomial X such that XA == B?

    Examples
    ========

    >>> from sympy.polys.monomials import monomial_divides
    >>> monomial_divides((1, 2), (3, 4))
    True
    >>> monomial_divides((1, 2), (0, 2))
    False
    """
    return all(a <= b for a, b in zip(A, B))

def monomial_max(*monoms):
    """
    Returns maximal degree for each variable in a set of monomials.

    Examples
    ========

    Consider monomials `x**3*y**4*z**5`, `y**5*z` and `x**6*y**3*z**9`.
    We wish to find out what is the maximal degree for each of `x`, `y`
    and `z` variables::

        >>> from sympy.polys.monomials import monomial_max

        >>> monomial_max((3,4,5), (0,5,1), (6,3,9))
        (6, 5, 9)

    """
    M = list(monoms[0])

    for N in monoms[1:]:
        for i, n in enumerate(N):
            M[i] = max(M[i], n)

    return tuple(M)

def monomial_min(*monoms):
    """
    Returns minimal degree for each variable in a set of monomials.

    Examples
    ========

    Consider monomials `x**3*y**4*z**5`, `y**5*z` and `x**6*y**3*z**9`.
    We wish to find out what is the minimal degree for each of `x`, `y`
    and `z` variables::

        >>> from sympy.polys.monomials import monomial_min

        >>> monomial_min((3,4,5), (0,5,1), (6,3,9))
        (0, 3, 1)

    """
    M = list(monoms[0])

    for N in monoms[1:]:
        for i, n in enumerate(N):
            M[i] = min(M[i], n)

    return tuple(M)

def monomial_deg(M):
    """
    Returns the total degree of a monomial.

    Examples
    ========

    The total degree of `xy^2` is 3:

    >>> from sympy.polys.monomials import monomial_deg
    >>> monomial_deg((1, 2))
    3
    """
    return sum(M)

def term_div(a, b, domain):
    """Division of two terms in over a ring/field. """
    a_lm, a_lc = a
    b_lm, b_lc = b

    monom = monomial_div(a_lm, b_lm)

    if domain.is_Field:
        if monom is not None:
            return monom, domain.quo(a_lc, b_lc)
        else:
            return None
    else:
        if not (monom is None or a_lc % b_lc):
            return monom, domain.quo(a_lc, b_lc)
        else:
            return None

class MonomialOps:
    """Code generator of fast monomial arithmetic functions. """

    @cacheit
    def __new__(cls, ngens):
        obj = super().__new__(cls)
        obj.ngens = ngens
        return obj

    def __getnewargs__(self):
        return (self.ngens,)

    def _build(self, code, name):
        ns = {}
        exec(code, ns)
        return ns[name]

    def _vars(self, name):
        return [ "%s%s" % (name, i) for i in range(self.ngens) ]

    @cacheit
    def mul(self):
        name = "monomial_mul"
        template = dedent("""\
        def %(name)s(A, B):
            (%(A)s,) = A
            (%(B)s,) = B
            return (%(AB)s,)
        """)
        A = self._vars("a")
        B = self._vars("b")
        AB = [ "%s + %s" % (a, b) for a, b in zip(A, B) ]
        code = template % {"name": name, "A": ", ".join(A), "B": ", ".join(B), "AB": ", ".join(AB)}
        return self._build(code, name)

    @cacheit
    def pow(self):
        name = "monomial_pow"
        template = dedent("""\
        def %(name)s(A, k):
            (%(A)s,) = A
            return (%(Ak)s,)
        """)
        A = self._vars("a")
        Ak = [ "%s*k" % a for a in A ]
        code = template % {"name": name, "A": ", ".join(A), "Ak": ", ".join(Ak)}
        return self._build(code, name)

    @cacheit
    def mulpow(self):
        name = "monomial_mulpow"
        template = dedent("""\
        def %(name)s(A, B, k):
            (%(A)s,) = A
            (%(B)s,) = B
            return (%(ABk)s,)
        """)
        A = self._vars("a")
        B = self._vars("b")
        ABk = [ "%s + %s*k" % (a, b) for a, b in zip(A, B) ]
        code = template % {"name": name, "A": ", ".join(A), "B": ", ".join(B), "ABk": ", ".join(ABk)}
        return self._build(code, name)

    @cacheit
    def ldiv(self):
        name = "monomial_ldiv"
        template = dedent("""\
        def %(name)s(A, B):
            (%(A)s,) = A
            (%(B)s,) = B
            return (%(AB)s,)
        """)
        A = self._vars("a")
        B = self._vars("b")
        AB = [ "%s - %s" % (a, b) for a, b in zip(A, B) ]
        code = template % {"name": name, "A": ", ".join(A), "B": ", ".join(B), "AB": ", ".join(AB)}
        return self._build(code, name)

    @cacheit
    def div(self):
        name = "monomial_div"
        template = dedent("""\
        def %(name)s(A, B):
            (%(A)s,) = A
            (%(B)s,) = B
            %(RAB)s
            return (%(R)s,)
        """)
        A = self._vars("a")
        B = self._vars("b")
        RAB = [ "r%(i)s = a%(i)s - b%(i)s\n    if r%(i)s < 0: return None" % {"i": i} for i in range(self.ngens) ]
        R = self._vars("r")
        code = template % {"name": name, "A": ", ".join(A), "B": ", ".join(B), "RAB": "\n    ".join(RAB), "R": ", ".join(R)}
        return self._build(code, name)

    @cacheit
    def lcm(self):
        name = "monomial_lcm"
        template = dedent("""\
        def %(name)s(A, B):
            (%(A)s,) = A
            (%(B)s,) = B
            return (%(AB)s,)
        """)
        A = self._vars("a")
        B = self._vars("b")
        AB = [ "%s if %s >= %s else %s" % (a, a, b, b) for a, b in zip(A, B) ]
        code = template % {"name": name, "A": ", ".join(A), "B": ", ".join(B), "AB": ", ".join(AB)}
        return self._build(code, name)

    @cacheit
    def gcd(self):
        name = "monomial_gcd"
        template = dedent("""\
        def %(name)s(A, B):
            (%(A)s,) = A
            (%(B)s,) = B
            return (%(AB)s,)
        """)
        A = self._vars("a")
        B = self._vars("b")
        AB = [ "%s if %s <= %s else %s" % (a, a, b, b) for a, b in zip(A, B) ]
        code = template % {"name": name, "A": ", ".join(A), "B": ", ".join(B), "AB": ", ".join(AB)}
        return self._build(code, name)

@public
class Monomial(PicklableWithSlots):
    """Class representing a monomial, i.e. a product of powers. """

    __slots__ = ('exponents', 'gens')

    def __init__(self, monom, gens=None):
        if not iterable(monom):
            rep, gens = dict_from_expr(sympify(monom), gens=gens)
            if len(rep) == 1 and list(rep.values())[0] == 1:
                monom = list(rep.keys())[0]
            else:
                raise ValueError("Expected a monomial got {}".format(monom))

        self.exponents = tuple(map(int, monom))
        self.gens = gens

    def rebuild(self, exponents, gens=None):
        return self.__class__(exponents, gens or self.gens)

    def __len__(self):
        return len(self.exponents)

    def __iter__(self):
        return iter(self.exponents)

    def __getitem__(self, item):
        return self.exponents[item]

    def __hash__(self):
        return hash((self.__class__.__name__, self.exponents, self.gens))

    def __str__(self):
        if self.gens:
            return "*".join([ "%s**%s" % (gen, exp) for gen, exp in zip(self.gens, self.exponents) ])
        else:
            return "%s(%s)" % (self.__class__.__name__, self.exponents)

    def as_expr(self, *gens):
        """Convert a monomial instance to a SymPy expression. """
        gens = gens or self.gens

        if not gens:
            raise ValueError(
                "Cannot convert %s to an expression without generators" % self)

        return Mul(*[ gen**exp for gen, exp in zip(gens, self.exponents) ])

    def __eq__(self, other):
        if isinstance(other, Monomial):
            exponents = other.exponents
        elif isinstance(other, (tuple, Tuple)):
            exponents = other
        else:
            return False

        return self.exponents == exponents

    def __ne__(self, other):
        return not self == other

    def __mul__(self, other):
        if isinstance(other, Monomial):
            exponents = other.exponents
        elif isinstance(other, (tuple, Tuple)):
            exponents = other
        else:
            raise NotImplementedError

        return self.rebuild(monomial_mul(self.exponents, exponents))

    def __truediv__(self, other):
        if isinstance(other, Monomial):
            exponents = other.exponents
        elif isinstance(other, (tuple, Tuple)):
            exponents = other
        else:
            raise NotImplementedError

        result = monomial_div(self.exponents, exponents)

        if result is not None:
            return self.rebuild(result)
        else:
            raise ExactQuotientFailed(self, Monomial(other))

    __floordiv__ = __truediv__

    def __pow__(self, other):
        n = int(other)
        if n < 0:
            raise ValueError("a non-negative integer expected, got %s" % other)
        return self.rebuild(monomial_pow(self.exponents, n))

    def gcd(self, other):
        """Greatest common divisor of monomials. """
        if isinstance(other, Monomial):
            exponents = other.exponents
        elif isinstance(other, (tuple, Tuple)):
            exponents = other
        else:
            raise TypeError(
                "an instance of Monomial class expected, got %s" % other)

        return self.rebuild(monomial_gcd(self.exponents, exponents))

    def lcm(self, other):
        """Least common multiple of monomials. """
        if isinstance(other, Monomial):
            exponents = other.exponents
        elif isinstance(other, (tuple, Tuple)):
            exponents = other
        else:
            raise TypeError(
                "an instance of Monomial class expected, got %s" % other)

        return self.rebuild(monomial_lcm(self.exponents, exponents))
<|endoftext|>"""Definitions of common exceptions for `polys` module. """


from sympy.utilities import public

@public
class BasePolynomialError(Exception):
    """Base class for polynomial related exceptions. """

    def new(self, *args):
        raise NotImplementedError("abstract base class")

@public
class ExactQuotientFailed(BasePolynomialError):

    def __init__(self, f, g, dom=None):
        self.f, self.g, self.dom = f, g, dom

    def __str__(self):  # pragma: no cover
        from sympy.printing.str import sstr

        if self.dom is None:
            return "%s does not divide %s" % (sstr(self.g), sstr(self.f))
        else:
            return "%s does not divide %s in %s" % (sstr(self.g), sstr(self.f), sstr(self.dom))

    def new(self, f, g):
        return self.__class__(f, g, self.dom)

@public
class PolynomialDivisionFailed(BasePolynomialError):

    def __init__(self, f, g, domain):
        self.f = f
        self.g = g
        self.domain = domain

    def __str__(self):
        if self.domain.is_EX:
            msg = "You may want to use a different simplification algorithm. Note " \
                  "that in general it's not possible to guarantee to detect zero "  \
                  "in this domain."
        elif not self.domain.is_Exact:
            msg = "Your working precision or tolerance of computations may be set " \
                  "improperly. Adjust those parameters of the coefficient domain "  \
                  "and try again."
        else:
            msg = "Zero detection is guaranteed in this coefficient domain. This "  \
                  "may indicate a bug in SymPy or the domain is user defined and "  \
                  "doesn't implement zero detection properly."

        return "couldn't reduce degree in a polynomial division algorithm when "    \
               "dividing %s by %s. This can happen when it's not possible to "      \
               "detect zero in the coefficient domain. The domain of computation "  \
               "is %s. %s" % (self.f, self.g, self.domain, msg)

@public
class OperationNotSupported(BasePolynomialError):

    def __init__(self, poly, func):
        self.poly = poly
        self.func = func

    def __str__(self):  # pragma: no cover
        return "`%s` operation not supported by %s representation" % (self.func, self.poly.rep.__class__.__name__)

@public
class HeuristicGCDFailed(BasePolynomialError):
    pass

class ModularGCDFailed(BasePolynomialError):
    pass

@public
class HomomorphismFailed(BasePolynomialError):
    pass

@public
class IsomorphismFailed(BasePolynomialError):
    pass

@public
class ExtraneousFactors(BasePolynomialError):
    pass

@public
class EvaluationFailed(BasePolynomialError):
    pass

@public
class RefinementFailed(BasePolynomialError):
    pass

@public
class CoercionFailed(BasePolynomialError):
    pass

@public
class NotInvertible(BasePolynomialError):
    pass

@public
class NotReversible(BasePolynomialError):
    pass

@public
class NotAlgebraic(BasePolynomialError):
    pass

@public
class DomainError(BasePolynomialError):
    pass

@public
class PolynomialError(BasePolynomialError):
    pass

@public
class UnificationFailed(BasePolynomialError):
    pass

@public
class UnsolvableFactorError(BasePolynomialError):
    """Raised if ``roots`` is called with strict=True and a polynomial
     having a factor whose solutions are not expressible in radicals
     is encountered."""

@public
class GeneratorsError(BasePolynomialError):
    pass

@public
class GeneratorsNeeded(GeneratorsError):
    pass

@public
class ComputationFailed(BasePolynomialError):

    def __init__(self, func, nargs, exc):
        self.func = func
        self.nargs = nargs
        self.exc = exc

    def __str__(self):
        return "%s(%s) failed without generators" % (self.func, ', '.join(map(str, self.exc.exprs[:self.nargs])))

@public
class UnivariatePolynomialError(PolynomialError):
    pass

@public
class MultivariatePolynomialError(PolynomialError):
    pass

@public
class PolificationFailed(PolynomialError):

    def __init__(self, opt, origs, exprs, seq=False):
        if not seq:
            self.orig = origs
            self.expr = exprs
            self.origs = [origs]
            self.exprs = [exprs]
        else:
            self.origs = origs
            self.exprs = exprs

        self.opt = opt
        self.seq = seq

    def __str__(self):  # pragma: no cover
        if not self.seq:
            return "Cannot construct a polynomial from %s" % str(self.orig)
        else:
            return "Cannot construct polynomials from %s" % ', '.join(map(str, self.origs))

@public
class OptionError(BasePolynomialError):
    pass

@public
class FlagError(OptionError):
    pass
<|endoftext|>"""Square-free decomposition algorithms and related tools. """


from sympy.polys.densearith import (
    dup_neg, dmp_neg,
    dup_sub, dmp_sub,
    dup_mul, dmp_mul,
    dup_quo, dmp_quo,
    dup_mul_ground, dmp_mul_ground)
from sympy.polys.densebasic import (
    dup_strip,
    dup_LC, dmp_ground_LC,
    dmp_zero_p,
    dmp_ground,
    dup_degree, dmp_degree, dmp_degree_in, dmp_degree_list,
    dmp_raise, dmp_inject,
    dup_convert)
from sympy.polys.densetools import (
    dup_diff, dmp_diff, dmp_diff_in,
    dup_shift, dmp_shift,
    dup_monic, dmp_ground_monic,
    dup_primitive, dmp_ground_primitive)
from sympy.polys.euclidtools import (
    dup_inner_gcd, dmp_inner_gcd,
    dup_gcd, dmp_gcd,
    dmp_resultant, dmp_primitive)
from sympy.polys.galoistools import (
    gf_sqf_list, gf_sqf_part)
from sympy.polys.polyerrors import (
    MultivariatePolynomialError,
    DomainError)


def _dup_check_degrees(f, result):
    """Sanity check the degrees of a computed factorization in K[x]."""
    deg = sum(k * dup_degree(fac) for (fac, k) in result)
    assert deg == dup_degree(f)


def _dmp_check_degrees(f, u, result):
    """Sanity check the degrees of a computed factorization in K[X]."""
    degs = [0] * (u + 1)
    for fac, k in result:
        degs_fac = dmp_degree_list(fac, u)
        degs = [d1 + k * d2 for d1, d2 in zip(degs, degs_fac)]
    assert tuple(degs) == dmp_degree_list(f, u)


def dup_sqf_p(f, K):
    """
    Return ``True`` if ``f`` is a square-free polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_sqf_p(x**2 - 2*x + 1)
    False
    >>> R.dup_sqf_p(x**2 - 1)
    True

    """
    if not f:
        return True
    else:
        return not dup_degree(dup_gcd(f, dup_diff(f, 1, K), K))


def dmp_sqf_p(f, u, K):
    """
    Return ``True`` if ``f`` is a square-free polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_sqf_p(x**2 + 2*x*y + y**2)
    False
    >>> R.dmp_sqf_p(x**2 + y**2)
    True

    """
    if dmp_zero_p(f, u):
        return True

    for i in range(u+1):

        fp = dmp_diff_in(f, 1, i, u, K)

        if dmp_zero_p(fp, u):
            continue

        gcd = dmp_gcd(f, fp, u, K)

        if dmp_degree_in(gcd, i, u) != 0:
            return False

    return True


def dup_sqf_norm(f, K):
    r"""
    Find a shift of `f` in `K[x]` that has square-free norm.

    The domain `K` must be an algebraic number field `k(a)` (see :ref:`QQ(a)`).

    Returns `(s,g,r)`, such that `g(x)=f(x-sa)`, `r(x)=\text{Norm}(g(x))` and
    `r` is a square-free polynomial over `k`.

    Examples
    ========

    We first create the algebraic number field `K=k(a)=\mathbb{Q}(\sqrt{3})`
    and rings `K[x]` and `k[x]`:

    >>> from sympy.polys import ring, QQ
    >>> from sympy import sqrt

    >>> K = QQ.algebraic_field(sqrt(3))
    >>> R, x = ring("x", K)
    >>> _, X = ring("x", QQ)

    We can now find a square free norm for a shift of `f`:

    >>> f = x**2 - 1
    >>> s, g, r = R.dup_sqf_norm(f)

    The choice of shift `s` is arbitrary and the particular values returned for
    `g` and `r` are determined by `s`.

    >>> s == 1
    True
    >>> g == x**2 - 2*sqrt(3)*x + 2
    True
    >>> r == X**4 - 8*X**2 + 4
    True

    The invariants are:

    >>> g == f.shift(-s*K.unit)
    True
    >>> g.norm() == r
    True
    >>> r.is_squarefree
    True

    Explanation
    ===========

    This is part of Trager's algorithm for factorizing polynomials over
    algebraic number fields. In particular this function is algorithm
    ``sqfr_norm`` from [Trager76]_.

    See Also
    ========

    dmp_sqf_norm:
        Analogous function for multivariate polynomials over ``k(a)``.
    dmp_norm:
        Computes the norm of `f` directly without any shift.
    dup_ext_factor:
        Function implementing Trager's algorithm that uses this.
    sympy.polys.polytools.sqf_norm:
        High-level interface for using this function.
    """
    if not K.is_Algebraic:
        raise DomainError("ground domain must be algebraic")

    s, g = 0, dmp_raise(K.mod.to_list(), 1, 0, K.dom)

    while True:
        h, _ = dmp_inject(f, 0, K, front=True)
        r = dmp_resultant(g, h, 1, K.dom)

        if dup_sqf_p(r, K.dom):
            break
        else:
            f, s = dup_shift(f, -K.unit, K), s + 1

    return s, f, r


def _dmp_sqf_norm_shifts(f, u, K):
    """Generate a sequence of candidate shifts for dmp_sqf_norm."""
    #
    # We want to find a minimal shift if possible because shifting high degree
    # variables can be expensive e.g. x**10 -> (x + 1)**10. We try a few easy
    # cases first before the final infinite loop that is guaranteed to give
    # only finitely many bad shifts (see Trager76 for proof of this in the
    # univariate case).
    #

    # First the trivial shift [0, 0, ...]
    n = u + 1
    s0 = [0] * n
    yield s0, f

    # Shift in multiples of the generator of the extension field K
    a = K.unit

    # Variables of degree > 0 ordered by increasing degree
    d = dmp_degree_list(f, u)
    var_indices = [i for di, i in sorted(zip(d, range(u+1))) if di > 0]

    # Now try [1, 0, 0, ...], [0, 1, 0, ...]
    for i in var_indices:
        s1 = s0.copy()
        s1[i] = 1
        a1 = [-a*s1i for s1i in s1]
        f1 = dmp_shift(f, a1, u, K)
        yield s1, f1

    # Now try [1, 1, 1, ...], [2, 2, 2, ...]
    j = 0
    while True:
        j += 1
        sj = [j] * n
        aj = [-a*j] * n
        fj = dmp_shift(f, aj, u, K)
        yield sj, fj


def dmp_sqf_norm(f, u, K):
    r"""
    Find a shift of ``f`` in ``K[X]`` that has square-free norm.

    The domain `K` must be an algebraic number field `k(a)` (see :ref:`QQ(a)`).

    Returns `(s,g,r)`, such that `g(x_1,x_2,\cdots)=f(x_1-s_1 a, x_2 - s_2 a,
    \cdots)`, `r(x)=\text{Norm}(g(x))` and `r` is a square-free polynomial over
    `k`.

    Examples
    ========

    We first create the algebraic number field `K=k(a)=\mathbb{Q}(i)` and rings
    `K[x,y]` and `k[x,y]`:

    >>> from sympy.polys import ring, QQ
    >>> from sympy import I

    >>> K = QQ.algebraic_field(I)
    >>> R, x, y = ring("x,y", K)
    >>> _, X, Y = ring("x,y", QQ)

    We can now find a square free norm for a shift of `f`:

    >>> f = x*y + y**2
    >>> s, g, r = R.dmp_sqf_norm(f)

    The choice of shifts ``s`` is arbitrary and the particular values returned
    for ``g`` and ``r`` are determined by ``s``.

    >>> s
    [0, 1]
    >>> g == x*y - I*x + y**2 - 2*I*y - 1
    True
    >>> r == X**2*Y**2 + X**2 + 2*X*Y**3 + 2*X*Y + Y**4 + 2*Y**2 + 1
    True

    The required invariants are:

    >>> g == f.shift_list([-si*K.unit for si in s])
    True
    >>> g.norm() == r
    True
    >>> r.is_squarefree
    True

    Explanation
    ===========

    This is part of Trager's algorithm for factorizing polynomials over
    algebraic number fields. In particular this function is a multivariate
    generalization of algorithm ``sqfr_norm`` from [Trager76]_.

    See Also
    ========

    dup_sqf_norm:
        Analogous function for univariate polynomials over ``k(a)``.
    dmp_norm:
        Computes the norm of `f` directly without any shift.
    dmp_ext_factor:
        Function implementing Trager's algorithm that uses this.
    sympy.polys.polytools.sqf_norm:
        High-level interface for using this function.
    """
    if not u:
        s, g, r = dup_sqf_norm(f, K)
        return [s], g, r

    if not K.is_Algebraic:
        raise DomainError("ground domain must be algebraic")

    g = dmp_raise(K.mod.to_list(), u + 1, 0, K.dom)

    for s, f in _dmp_sqf_norm_shifts(f, u, K):

        h, _ = dmp_inject(f, u, K, front=True)
        r = dmp_resultant(g, h, u + 1, K.dom)

        if dmp_sqf_p(r, u, K.dom):
            break

    return s, f, r


def dmp_norm(f, u, K):
    r"""
    Norm of ``f`` in ``K[X]``, often not square-free.

    The domain `K` must be an algebraic number field `k(a)` (see :ref:`QQ(a)`).

    Examples
    ========

    We first define the algebraic number field `K = k(a) = \mathbb{Q}(\sqrt{2})`:

    >>> from sympy import QQ, sqrt
    >>> from sympy.polys.sqfreetools import dmp_norm
    >>> k = QQ
    >>> K = k.algebraic_field(sqrt(2))

    We can now compute the norm of a polynomial `p` in `K[x,y]`:

    >>> p = [[K(1)], [K(1),K.unit]]                  # x + y + sqrt(2)
    >>> N = [[k(1)], [k(2),k(0)], [k(1),k(0),k(-2)]] # x**2 + 2*x*y + y**2 - 2
    >>> dmp_norm(p, 1, K) == N
    True

    In higher level functions that is:

    >>> from sympy import expand, roots, minpoly
    >>> from sympy.abc import x, y
    >>> from math import prod
    >>> a = sqrt(2)
    >>> e = (x + y + a)
    >>> e.as_poly([x, y], extension=a).norm()
    Poly(x**2 + 2*x*y + y**2 - 2, x, y, domain='QQ')

    This is equal to the product of the expressions `x + y + a_i` where the
    `a_i` are the conjugates of `a`:

    >>> pa = minpoly(a)
    >>> pa
    _x**2 - 2
    >>> rs = roots(pa, multiple=True)
    >>> rs
    [sqrt(2), -sqrt(2)]
    >>> n = prod(e.subs(a, r) for r in rs)
    >>> n
    (x + y - sqrt(2))*(x + y + sqrt(2))
    >>> expand(n)
    x**2 + 2*x*y + y**2 - 2

    Explanation
    ===========

    Given an algebraic number field `K = k(a)` any element `b` of `K` can be
    represented as polynomial function `b=g(a)` where `g` is in `k[x]`. If the
    minimal polynomial of `a` over `k` is `p_a` then the roots `a_1`, `a_2`,
    `\cdots` of `p_a(x)` are the conjugates of `a`. The norm of `b` is the
    product `g(a1) \times g(a2) \times \cdots` and is an element of `k`.

    As in [Trager76]_ we extend this norm to multivariate polynomials over `K`.
    If `b(x)` is a polynomial in `k(a)[X]` then we can think of `b` as being
    alternately a function `g_X(a)` where `g_X` is an element of `k[X][y]` i.e.
    a polynomial function with coefficients that are elements of `k[X]`. Then
    the norm of `b` is the product `g_X(a1) \times g_X(a2) \times \cdots` and
    will be an element of `k[X]`.

    See Also
    ========

    dmp_sqf_norm:
        Compute a shift of `f` so that the `\text{Norm}(f)` is square-free.
    sympy.polys.polytools.Poly.norm:
        Higher-level function that calls this.
    """
    if not K.is_Algebraic:
        raise DomainError("ground domain must be algebraic")

    g = dmp_raise(K.mod.to_list(), u + 1, 0, K.dom)
    h, _ = dmp_inject(f, u, K, front=True)

    return dmp_resultant(g, h, u + 1, K.dom)


def dup_gf_sqf_part(f, K):
    """Compute square-free part of ``f`` in ``GF(p)[x]``. """
    f = dup_convert(f, K, K.dom)
    g = gf_sqf_part(f, K.mod, K.dom)
    return dup_convert(g, K.dom, K)


def dmp_gf_sqf_part(f, u, K):
    """Compute square-free part of ``f`` in ``GF(p)[X]``. """
    raise NotImplementedError('multivariate polynomials over finite fields')


def dup_sqf_part(f, K):
    """
    Returns square-free part of a polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_sqf_part(x**3 - 3*x - 2)
    x**2 - x - 2

    See Also
    ========

    sympy.polys.polytools.Poly.sqf_part
    """
    if K.is_FiniteField:
        return dup_gf_sqf_part(f, K)

    if not f:
        return f

    if K.is_negative(dup_LC(f, K)):
        f = dup_neg(f, K)

    gcd = dup_gcd(f, dup_diff(f, 1, K), K)
    sqf = dup_quo(f, gcd, K)

    if K.is_Field:
        return dup_monic(sqf, K)
    else:
        return dup_primitive(sqf, K)[1]


def dmp_sqf_part(f, u, K):
    """
    Returns square-free part of a polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_sqf_part(x**3 + 2*x**2*y + x*y**2)
    x**2 + x*y

    """
    if not u:
        return dup_sqf_part(f, K)

    if K.is_FiniteField:
        return dmp_gf_sqf_part(f, u, K)

    if dmp_zero_p(f, u):
        return f

    if K.is_negative(dmp_ground_LC(f, u, K)):
        f = dmp_neg(f, u, K)

    gcd = f
    for i in range(u+1):
        gcd = dmp_gcd(gcd, dmp_diff_in(f, 1, i, u, K), u, K)
    sqf = dmp_quo(f, gcd, u, K)

    if K.is_Field:
        return dmp_ground_monic(sqf, u, K)
    else:
        return dmp_ground_primitive(sqf, u, K)[1]


def dup_gf_sqf_list(f, K, all=False):
    """Compute square-free decomposition of ``f`` in ``GF(p)[x]``. """
    f_orig = f

    f = dup_convert(f, K, K.dom)

    coeff, factors = gf_sqf_list(f, K.mod, K.dom, all=all)

    for i, (f, k) in enumerate(factors):
        factors[i] = (dup_convert(f, K.dom, K), k)

    _dup_check_degrees(f_orig, factors)

    return K.convert(coeff, K.dom), factors


def dmp_gf_sqf_list(f, u, K, all=False):
    """Compute square-free decomposition of ``f`` in ``GF(p)[X]``. """
    raise NotImplementedError('multivariate polynomials over finite fields')


def dup_sqf_list(f, K, all=False):
    """
    Return square-free decomposition of a polynomial in ``K[x]``.

    Uses Yun's algorithm from [Yun76]_.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> f = 2*x**5 + 16*x**4 + 50*x**3 + 76*x**2 + 56*x + 16

    >>> R.dup_sqf_list(f)
    (2, [(x + 1, 2), (x + 2, 3)])
    >>> R.dup_sqf_list(f, all=True)
    (2, [(1, 1), (x + 1, 2), (x + 2, 3)])

    See Also
    ========

    dmp_sqf_list:
        Corresponding function for multivariate polynomials.
    sympy.polys.polytools.sqf_list:
        High-level function for square-free factorization of expressions.
    sympy.polys.polytools.Poly.sqf_list:
        Analogous method on :class:`~.Poly`.

    References
    ==========

    [Yun76]_
    """
    if K.is_FiniteField:
        return dup_gf_sqf_list(f, K, all=all)

    f_orig = f

    if K.is_Field:
        coeff = dup_LC(f, K)
        f = dup_monic(f, K)
    else:
        coeff, f = dup_primitive(f, K)

        if K.is_negative(dup_LC(f, K)):
            f = dup_neg(f, K)
            coeff = -coeff

    if dup_degree(f) <= 0:
        return coeff, []

    result, i = [], 1

    h = dup_diff(f, 1, K)
    g, p, q = dup_inner_gcd(f, h, K)

    while True:
        d = dup_diff(p, 1, K)
        h = dup_sub(q, d, K)

        if not h:
            result.append((p, i))
            break

        g, p, q = dup_inner_gcd(p, h, K)

        if all or dup_degree(g) > 0:
            result.append((g, i))

        i += 1

    _dup_check_degrees(f_orig, result)

    return coeff, result


def dup_sqf_list_include(f, K, all=False):
    """
    Return square-free decomposition of a polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> f = 2*x**5 + 16*x**4 + 50*x**3 + 76*x**2 + 56*x + 16

    >>> R.dup_sqf_list_include(f)
    [(2, 1), (x + 1, 2), (x + 2, 3)]
    >>> R.dup_sqf_list_include(f, all=True)
    [(2, 1), (x + 1, 2), (x + 2, 3)]

    """
    coeff, factors = dup_sqf_list(f, K, all=all)

    if factors and factors[0][1] == 1:
        g = dup_mul_ground(factors[0][0], coeff, K)
        return [(g, 1)] + factors[1:]
    else:
        g = dup_strip([coeff])
        return [(g, 1)] + factors


def dmp_sqf_list(f, u, K, all=False):
    """
    Return square-free decomposition of a polynomial in `K[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = x**5 + 2*x**4*y + x**3*y**2

    >>> R.dmp_sqf_list(f)
    (1, [(x + y, 2), (x, 3)])
    >>> R.dmp_sqf_list(f, all=True)
    (1, [(1, 1), (x + y, 2), (x, 3)])

    Explanation
    ===========

    Uses Yun's algorithm for univariate polynomials from [Yun76]_ recursively.
    The multivariate polynomial is treated as a univariate polynomial in its
    leading variable. Then Yun's algorithm computes the square-free
    factorization of the primitive and the content is factored recursively.

    It would be better to use a dedicated algorithm for multivariate
    polynomials instead.

    See Also
    ========

    dup_sqf_list:
        Corresponding function for univariate polynomials.
    sympy.polys.polytools.sqf_list:
        High-level function for square-free factorization of expressions.
    sympy.polys.polytools.Poly.sqf_list:
        Analogous method on :class:`~.Poly`.
    """
    if not u:
        return dup_sqf_list(f, K, all=all)

    if K.is_FiniteField:
        return dmp_gf_sqf_list(f, u, K, all=all)

    f_orig = f

    if K.is_Field:
        coeff = dmp_ground_LC(f, u, K)
        f = dmp_ground_monic(f, u, K)
    else:
        coeff, f = dmp_ground_primitive(f, u, K)

        if K.is_negative(dmp_ground_LC(f, u, K)):
            f = dmp_neg(f, u, K)
            coeff = -coeff

    deg = dmp_degree(f, u)
    if deg < 0:
        return coeff, []

    # Yun's algorithm requires the polynomial to be primitive as a univariate
    # polynomial in its main variable.
    content, f = dmp_primitive(f, u, K)

    result = {}

    if deg != 0:

        h = dmp_diff(f, 1, u, K)
        g, p, q = dmp_inner_gcd(f, h, u, K)

        i = 1

        while True:
            d = dmp_diff(p, 1, u, K)
            h = dmp_sub(q, d, u, K)

            if dmp_zero_p(h, u):
                result[i] = p
                break

            g, p, q = dmp_inner_gcd(p, h, u, K)

            if all or dmp_degree(g, u) > 0:
                result[i] = g

            i += 1

    coeff_content, result_content = dmp_sqf_list(content, u-1, K, all=all)

    coeff *= coeff_content

    # Combine factors of the content and primitive part that have the same
    # multiplicity to produce a list in ascending order of multiplicity.
    for fac, i in result_content:
        fac = [fac]
        if i in result:
            result[i] = dmp_mul(result[i], fac, u, K)
        else:
            result[i] = fac

    result = [(result[i], i) for i in sorted(result)]

    _dmp_check_degrees(f_orig, u, result)

    return coeff, result


def dmp_sqf_list_include(f, u, K, all=False):
    """
    Return square-free decomposition of a polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = x**5 + 2*x**4*y + x**3*y**2

    >>> R.dmp_sqf_list_include(f)
    [(1, 1), (x + y, 2), (x, 3)]
    >>> R.dmp_sqf_list_include(f, all=True)
    [(1, 1), (x + y, 2), (x, 3)]

    """
    if not u:
        return dup_sqf_list_include(f, K, all=all)

    coeff, factors = dmp_sqf_list(f, u, K, all=all)

    if factors and factors[0][1] == 1:
        g = dmp_mul_ground(factors[0][0], coeff, u, K)
        return [(g, 1)] + factors[1:]
    else:
        g = dmp_ground(coeff, u)
        return [(g, 1)] + factors


def dup_gff_list(f, K):
    """
    Compute greatest factorial factorization of ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_gff_list(x**5 + 2*x**4 - x**3 - 2*x**2)
    [(x, 1), (x + 2, 4)]

    """
    if not f:
        raise ValueError("greatest factorial factorization doesn't exist for a zero polynomial")

    f = dup_monic(f, K)

    if not dup_degree(f):
        return []
    else:
        g = dup_gcd(f, dup_shift(f, K.one, K), K)
        H = dup_gff_list(g, K)

        for i, (h, k) in enumerate(H):
            g = dup_mul(g, dup_shift(h, -K(k), K), K)
            H[i] = (h, k + 1)

        f = dup_quo(f, g, K)

        if not dup_degree(f):
            return H
        else:
            return [(f, 1)] + H


def dmp_gff_list(f, u, K):
    """
    Compute greatest factorial factorization of ``f`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    """
    if not u:
        return dup_gff_list(f, K)
    else:
        raise MultivariatePolynomialError(f)
<|endoftext|>"""Useful utilities for higher level polynomial classes. """

from __future__ import annotations

from sympy.external.gmpy import GROUND_TYPES

from sympy.core import (S, Add, Mul, Pow, Eq, Expr,
    expand_mul, expand_multinomial)
from sympy.core.exprtools import decompose_power, decompose_power_rat
from sympy.core.numbers import _illegal
from sympy.polys.polyerrors import PolynomialError, GeneratorsError
from sympy.polys.polyoptions import build_options

import re


_gens_order = {
    'a': 301, 'b': 302, 'c': 303, 'd': 304,
    'e': 305, 'f': 306, 'g': 307, 'h': 308,
    'i': 309, 'j': 310, 'k': 311, 'l': 312,
    'm': 313, 'n': 314, 'o': 315, 'p': 216,
    'q': 217, 'r': 218, 's': 219, 't': 220,
    'u': 221, 'v': 222, 'w': 223, 'x': 124,
    'y': 125, 'z': 126,
}

_max_order = 1000
_re_gen = re.compile(r"^(.*?)(\d*)$", re.MULTILINE)


def _nsort(roots, separated=False):
    """Sort the numerical roots putting the real roots first, then sorting
    according to real and imaginary parts. If ``separated`` is True, then
    the real and imaginary roots will be returned in two lists, respectively.

    This routine tries to avoid issue 6137 by separating the roots into real
    and imaginary parts before evaluation. In addition, the sorting will raise
    an error if any computation cannot be done with precision.
    """
    if not all(r.is_number for r in roots):
        raise NotImplementedError
    if not len(roots):
        return [] if not separated else ([], [])
    # see issue 6137:
    # get the real part of the evaluated real and imaginary parts of each root
    key = [[i.n(2).as_real_imag()[0] for i in r.as_real_imag()] for r in roots]
    # make sure the parts were computed with precision
    if len(roots) > 1 and any(i._prec == 1 for k in key for i in k):
        raise NotImplementedError("could not compute root with precision")
    # insert a key to indicate if the root has an imaginary part
    key = [(1 if i else 0, r, i) for r, i in key]
    key = sorted(zip(key, roots))
    # return the real and imaginary roots separately if desired
    if separated:
        r = []
        i = []
        for (im, _, _), v in key:
            if im:
                i.append(v)
            else:
                r.append(v)
        return r, i
    _, roots = zip(*key)
    return list(roots)


def _sort_gens(gens, **args):
    """Sort generators in a reasonably intelligent way. """
    opt = build_options(args)

    gens_order, wrt = {}, None

    if opt is not None:
        gens_order, wrt = {}, opt.wrt

        for i, gen in enumerate(opt.sort):
            gens_order[gen] = i + 1

    def order_key(gen):
        gen = str(gen)

        if wrt is not None:
            try:
                return (-len(wrt) + wrt.index(gen), gen, 0)
            except ValueError:
                pass

        name, index = _re_gen.match(gen).groups()

        if index:
            index = int(index)
        else:
            index = 0

        try:
            return ( gens_order[name], name, index)
        except KeyError:
            pass

        try:
            return (_gens_order[name], name, index)
        except KeyError:
            pass

        return (_max_order, name, index)

    try:
        gens = sorted(gens, key=order_key)
    except TypeError:  # pragma: no cover
        pass

    return tuple(gens)


def _unify_gens(f_gens, g_gens):
    """Unify generators in a reasonably intelligent way. """
    f_gens = list(f_gens)
    g_gens = list(g_gens)

    if f_gens == g_gens:
        return tuple(f_gens)

    gens, common, k = [], [], 0

    for gen in f_gens:
        if gen in g_gens:
            common.append(gen)

    for i, gen in enumerate(g_gens):
        if gen in common:
            g_gens[i], k = common[k], k + 1

    for gen in common:
        i = f_gens.index(gen)

        gens.extend(f_gens[:i])
        f_gens = f_gens[i + 1:]

        i = g_gens.index(gen)

        gens.extend(g_gens[:i])
        g_gens = g_gens[i + 1:]

        gens.append(gen)

    gens.extend(f_gens)
    gens.extend(g_gens)

    return tuple(gens)


def _analyze_gens(gens):
    """Support for passing generators as `*gens` and `[gens]`. """
    if len(gens) == 1 and hasattr(gens[0], '__iter__'):
        return tuple(gens[0])
    else:
        return tuple(gens)


def _sort_factors(factors, **args):
    """Sort low-level factors in increasing 'complexity' order. """

    # XXX: GF(p) does not support comparisons so we need a key function to sort
    # the factors if python-flint is being used. A better solution might be to
    # add a sort key method to each domain.
    def order_key(factor):
        if isinstance(factor, _GF_types):
            return int(factor)
        elif isinstance(factor, list):
            return [order_key(f) for f in factor]
        else:
            return factor

    def order_if_multiple_key(factor):
        (f, n) = factor
        return (len(f), n, order_key(f))

    def order_no_multiple_key(f):
        return (len(f), order_key(f))

    if args.get('multiple', True):
        return sorted(factors, key=order_if_multiple_key)
    else:
        return sorted(factors, key=order_no_multiple_key)


illegal_types = [type(obj) for obj in _illegal]
finf = [float(i) for i in _illegal[1:3]]


def _not_a_coeff(expr):
    """Do not treat NaN and infinities as valid polynomial coefficients. """
    if type(expr) in illegal_types or expr in finf:
        return True
    if isinstance(expr, float) and float(expr) != expr:
        return True  # nan
    return  # could be


def _parallel_dict_from_expr_if_gens(exprs, opt):
    """Transform expressions into a multinomial form given generators. """
    k, indices = len(opt.gens), {}

    for i, g in enumerate(opt.gens):
        indices[g] = i

    polys = []

    for expr in exprs:
        poly = {}

        if expr.is_Equality:
            expr = expr.lhs - expr.rhs

        for term in Add.make_args(expr):
            coeff, monom = [], [0]*k

            for factor in Mul.make_args(term):
                if not _not_a_coeff(factor) and factor.is_Number:
                    coeff.append(factor)
                else:
                    try:
                        if opt.series is False:
                            base, exp = decompose_power(factor)

                            if exp < 0:
                                exp, base = -exp, Pow(base, -S.One)
                        else:
                            base, exp = decompose_power_rat(factor)

                        monom[indices[base]] = exp
                    except KeyError:
                        if not factor.has_free(*opt.gens):
                            coeff.append(factor)
                        else:
                            raise PolynomialError("%s contains an element of "
                                                  "the set of generators." % factor)

            monom = tuple(monom)

            if monom in poly:
                poly[monom] += Mul(*coeff)
            else:
                poly[monom] = Mul(*coeff)

        polys.append(poly)

    return polys, opt.gens


def _parallel_dict_from_expr_no_gens(exprs, opt):
    """Transform expressions into a multinomial form and figure out generators. """
    if opt.domain is not None:
        def _is_coeff(factor):
            return factor in opt.domain
    elif opt.extension is True:
        def _is_coeff(factor):
            return factor.is_algebraic
    elif opt.greedy is not False:
        def _is_coeff(factor):
            return factor is S.ImaginaryUnit
    else:
        def _is_coeff(factor):
            return factor.is_number

    gens, reprs = set(), []

    for expr in exprs:
        terms = []

        if expr.is_Equality:
            expr = expr.lhs - expr.rhs

        for term in Add.make_args(expr):
            coeff, elements = [], {}

            for factor in Mul.make_args(term):
                if not _not_a_coeff(factor) and (factor.is_Number or _is_coeff(factor)):
                    coeff.append(factor)
                else:
                    if opt.series is False:
                        base, exp = decompose_power(factor)

                        if exp < 0:
                            exp, base = -exp, Pow(base, -S.One)
                    else:
                        base, exp = decompose_power_rat(factor)

                    elements[base] = elements.setdefault(base, 0) + exp
                    gens.add(base)

            terms.append((coeff, elements))

        reprs.append(terms)

    gens = _sort_gens(gens, opt=opt)
    k, indices = len(gens), {}

    for i, g in enumerate(gens):
        indices[g] = i

    polys = []

    for terms in reprs:
        poly = {}

        for coeff, term in terms:
            monom = [0]*k

            for base, exp in term.items():
                monom[indices[base]] = exp

            monom = tuple(monom)

            if monom in poly:
                poly[monom] += Mul(*coeff)
            else:
                poly[monom] = Mul(*coeff)

        polys.append(poly)

    return polys, tuple(gens)


def _dict_from_expr_if_gens(expr, opt):
    """Transform an expression into a multinomial form given generators. """
    (poly,), gens = _parallel_dict_from_expr_if_gens((expr,), opt)
    return poly, gens


def _dict_from_expr_no_gens(expr, opt):
    """Transform an expression into a multinomial form and figure out generators. """
    (poly,), gens = _parallel_dict_from_expr_no_gens((expr,), opt)
    return poly, gens


def parallel_dict_from_expr(exprs, **args):
    """Transform expressions into a multinomial form. """
    reps, opt = _parallel_dict_from_expr(exprs, build_options(args))
    return reps, opt.gens


def _parallel_dict_from_expr(exprs, opt):
    """Transform expressions into a multinomial form. """
    if opt.expand is not False:
        exprs = [ expr.expand() for expr in exprs ]

    if any(expr.is_commutative is False for expr in exprs):
        raise PolynomialError('non-commutative expressions are not supported')

    if opt.gens:
        reps, gens = _parallel_dict_from_expr_if_gens(exprs, opt)
    else:
        reps, gens = _parallel_dict_from_expr_no_gens(exprs, opt)

    return reps, opt.clone({'gens': gens})


def dict_from_expr(expr, **args):
    """Transform an expression into a multinomial form. """
    rep, opt = _dict_from_expr(expr, build_options(args))
    return rep, opt.gens


def _dict_from_expr(expr, opt):
    """Transform an expression into a multinomial form. """
    if expr.is_commutative is False:
        raise PolynomialError('non-commutative expressions are not supported')

    def _is_expandable_pow(expr):
        return (expr.is_Pow and expr.exp.is_positive and expr.exp.is_Integer
                and expr.base.is_Add)

    if opt.expand is not False:
        if not isinstance(expr, (Expr, Eq)):
            raise PolynomialError('expression must be of type Expr')
        expr = expr.expand()
        # TODO: Integrate this into expand() itself
        while any(_is_expandable_pow(i) or i.is_Mul and
            any(_is_expandable_pow(j) for j in i.args) for i in
                Add.make_args(expr)):

            expr = expand_multinomial(expr)
        while any(i.is_Mul and any(j.is_Add for j in i.args) for i in Add.make_args(expr)):
            expr = expand_mul(expr)

    if opt.gens:
        rep, gens = _dict_from_expr_if_gens(expr, opt)
    else:
        rep, gens = _dict_from_expr_no_gens(expr, opt)

    return rep, opt.clone({'gens': gens})


def expr_from_dict(rep, *gens):
    """Convert a multinomial form into an expression. """
    result = []

    for monom, coeff in rep.items():
        term = [coeff]
        for g, m in zip(gens, monom):
            if m:
                term.append(Pow(g, m))

        result.append(Mul(*term))

    return Add(*result)

parallel_dict_from_basic = parallel_dict_from_expr
dict_from_basic = dict_from_expr
basic_from_dict = expr_from_dict


def _dict_reorder(rep, gens, new_gens):
    """Reorder levels using dict representation. """
    gens = list(gens)

    monoms = rep.keys()
    coeffs = rep.values()

    new_monoms = [ [] for _ in range(len(rep)) ]
    used_indices = set()

    for gen in new_gens:
        try:
            j = gens.index(gen)
            used_indices.add(j)

            for M, new_M in zip(monoms, new_monoms):
                new_M.append(M[j])
        except ValueError:
            for new_M in new_monoms:
                new_M.append(0)

    for i, _ in enumerate(gens):
        if i not in used_indices:
            for monom in monoms:
                if monom[i]:
                    raise GeneratorsError("unable to drop generators")

    return map(tuple, new_monoms), coeffs


class PicklableWithSlots:
    """
    Mixin class that allows to pickle objects with ``__slots__``.

    Examples
    ========

    First define a class that mixes :class:`PicklableWithSlots` in::

        >>> from sympy.polys.polyutils import PicklableWithSlots
        >>> class Some(PicklableWithSlots):
        ...     __slots__ = ('foo', 'bar')
        ...
        ...     def __init__(self, foo, bar):
        ...         self.foo = foo
        ...         self.bar = bar

    To make :mod:`pickle` happy in doctest we have to use these hacks::

        >>> import builtins
        >>> builtins.Some = Some
        >>> from sympy.polys import polyutils
        >>> polyutils.Some = Some

    Next lets see if we can create an instance, pickle it and unpickle::

        >>> some = Some('abc', 10)
        >>> some.foo, some.bar
        ('abc', 10)

        >>> from pickle import dumps, loads
        >>> some2 = loads(dumps(some))

        >>> some2.foo, some2.bar
        ('abc', 10)

    """

    __slots__ = ()

    def __getstate__(self, cls=None):
        if cls is None:
            # This is the case for the instance that gets pickled
            cls = self.__class__

        d = {}

        # Get all data that should be stored from super classes
        for c in cls.__bases__:
            # XXX: Python 3.11 defines object.__getstate__ and it does not
            # accept any arguments so we need to make sure not to call it with
            # an argument here. To be compatible with Python < 3.11 we need to
            # be careful not to assume that c or object has a __getstate__
            # method though.
            getstate = getattr(c, "__getstate__", None)
            objstate = getattr(object, "__getstate__", None)
            if getstate is not None and getstate is not objstate:
                d.update(getstate(self, c))

        # Get all information that should be stored from cls and return the dict
        for name in cls.__slots__:
            if hasattr(self, name):
                d[name] = getattr(self, name)

        return d

    def __setstate__(self, d):
        # All values that were pickled are now assigned to a fresh instance
        for name, value in d.items():
            setattr(self, name, value)


class IntegerPowerable:
    r"""
    Mixin class for classes that define a `__mul__` method, and want to be
    raised to integer powers in the natural way that follows. Implements
    powering via binary expansion, for efficiency.

    By default, only integer powers $\geq 2$ are supported. To support the
    first, zeroth, or negative powers, override the corresponding methods,
    `_first_power`, `_zeroth_power`, `_negative_power`, below.
    """

    def __pow__(self, e, modulo=None):
        if e < 2:
            try:
                if e == 1:
                    return self._first_power()
                elif e == 0:
                    return self._zeroth_power()
                else:
                    return self._negative_power(e, modulo=modulo)
            except NotImplementedError:
                return NotImplemented
        else:
            bits = [int(d) for d in reversed(bin(e)[2:])]
            n = len(bits)
            p = self
            first = True
            for i in range(n):
                if bits[i]:
                    if first:
                        r = p
                        first = False
                    else:
                        r *= p
                        if modulo is not None:
                            r %= modulo
                if i < n - 1:
                    p *= p
                    if modulo is not None:
                        p %= modulo
            return r

    def _negative_power(self, e, modulo=None):
        """
        Compute inverse of self, then raise that to the abs(e) power.
        For example, if the class has an `inv()` method,
            return self.inv() ** abs(e) % modulo
        """
        raise NotImplementedError

    def _zeroth_power(self):
        """Return unity element of algebraic struct to which self belongs."""
        raise NotImplementedError

    def _first_power(self):
        """Return a copy of self."""
        raise NotImplementedError


_GF_types: tuple[type, ...]


if GROUND_TYPES == 'flint':
    import flint
    _GF_types = (flint.nmod, flint.fmpz_mod)
else:
    from sympy.polys.domains.modularinteger import ModularInteger
    flint = None
    _GF_types = (ModularInteger,)
<|endoftext|>"""Basic tools for dense recursive polynomials in ``K[x]`` or ``K[X]``. """


from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key

import random


ninf = float('-inf')


def poly_LC(f, K):
    """
    Return leading coefficient of ``f``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import poly_LC

    >>> poly_LC([], ZZ)
    0
    >>> poly_LC([ZZ(1), ZZ(2), ZZ(3)], ZZ)
    1

    """
    if not f:
        return K.zero
    else:
        return f[0]


def poly_TC(f, K):
    """
    Return trailing coefficient of ``f``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import poly_TC

    >>> poly_TC([], ZZ)
    0
    >>> poly_TC([ZZ(1), ZZ(2), ZZ(3)], ZZ)
    3

    """
    if not f:
        return K.zero
    else:
        return f[-1]

dup_LC = dmp_LC = poly_LC
dup_TC = dmp_TC = poly_TC


def dmp_ground_LC(f, u, K):
    """
    Return the ground leading coefficient.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_ground_LC

    >>> f = ZZ.map([[[1], [2, 3]]])

    >>> dmp_ground_LC(f, 2, ZZ)
    1

    """
    while u:
        f = dmp_LC(f, K)
        u -= 1

    return dup_LC(f, K)


def dmp_ground_TC(f, u, K):
    """
    Return the ground trailing coefficient.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_ground_TC

    >>> f = ZZ.map([[[1], [2, 3]]])

    >>> dmp_ground_TC(f, 2, ZZ)
    3

    """
    while u:
        f = dmp_TC(f, K)
        u -= 1

    return dup_TC(f, K)


def dmp_true_LT(f, u, K):
    """
    Return the leading term ``c * x_1**n_1 ... x_k**n_k``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_true_LT

    >>> f = ZZ.map([[4], [2, 0], [3, 0, 0]])

    >>> dmp_true_LT(f, 1, ZZ)
    ((2, 0), 4)

    """
    monom = []

    while u:
        monom.append(len(f) - 1)
        f, u = f[0], u - 1

    if not f:
        monom.append(0)
    else:
        monom.append(len(f) - 1)

    return tuple(monom), dup_LC(f, K)


def dup_degree(f):
    """
    Return the leading degree of ``f`` in ``K[x]``.

    Note that the degree of 0 is negative infinity (``float('-inf')``).

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_degree

    >>> f = ZZ.map([1, 2, 0, 3])

    >>> dup_degree(f)
    3

    """
    if not f:
        return ninf
    return len(f) - 1


def dmp_degree(f, u):
    """
    Return the leading degree of ``f`` in ``x_0`` in ``K[X]``.

    Note that the degree of 0 is negative infinity (``float('-inf')``).

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_degree

    >>> dmp_degree([[[]]], 2)
    -inf

    >>> f = ZZ.map([[2], [1, 2, 3]])

    >>> dmp_degree(f, 1)
    1

    """
    if dmp_zero_p(f, u):
        return ninf
    else:
        return len(f) - 1


def _rec_degree_in(g, v, i, j):
    """Recursive helper function for :func:`dmp_degree_in`."""
    if i == j:
        return dmp_degree(g, v)

    v, i = v - 1, i + 1

    return max(_rec_degree_in(c, v, i, j) for c in g)


def dmp_degree_in(f, j, u):
    """
    Return the leading degree of ``f`` in ``x_j`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_degree_in

    >>> f = ZZ.map([[2], [1, 2, 3]])

    >>> dmp_degree_in(f, 0, 1)
    1
    >>> dmp_degree_in(f, 1, 1)
    2

    """
    if not j:
        return dmp_degree(f, u)
    if j < 0 or j > u:
        raise IndexError("0 <= j <= %s expected, got %s" % (u, j))

    return _rec_degree_in(f, u, 0, j)


def _rec_degree_list(g, v, i, degs):
    """Recursive helper for :func:`dmp_degree_list`."""
    degs[i] = max(degs[i], dmp_degree(g, v))

    if v > 0:
        v, i = v - 1, i + 1

        for c in g:
            _rec_degree_list(c, v, i, degs)


def dmp_degree_list(f, u):
    """
    Return a list of degrees of ``f`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_degree_list

    >>> f = ZZ.map([[1], [1, 2, 3]])

    >>> dmp_degree_list(f, 1)
    (1, 2)

    """
    degs = [ninf]*(u + 1)
    _rec_degree_list(f, u, 0, degs)
    return tuple(degs)


def dup_strip(f):
    """
    Remove leading zeros from ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys.densebasic import dup_strip

    >>> dup_strip([0, 0, 1, 2, 3, 0])
    [1, 2, 3, 0]

    """
    if not f or f[0]:
        return f

    i = 0

    for cf in f:
        if cf:
            break
        else:
            i += 1

    return f[i:]


def dmp_strip(f, u):
    """
    Remove leading zeros from ``f`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.densebasic import dmp_strip

    >>> dmp_strip([[], [0, 1, 2], [1]], 1)
    [[0, 1, 2], [1]]

    """
    if not u:
        return dup_strip(f)

    if dmp_zero_p(f, u):
        return f

    i, v = 0, u - 1

    for c in f:
        if not dmp_zero_p(c, v):
            break
        else:
            i += 1

    if i == len(f):
        return dmp_zero(u)
    else:
        return f[i:]


def _rec_validate(f, g, i, K):
    """Recursive helper for :func:`dmp_validate`."""
    if not isinstance(g, list):
        if K is not None and not K.of_type(g):
            raise TypeError("%s in %s in not of type %s" % (g, f, K.dtype))

        return {i - 1}
    elif not g:
        return {i}
    else:
        levels = set()

        for c in g:
            levels |= _rec_validate(f, c, i + 1, K)

        return levels


def _rec_strip(g, v):
    """Recursive helper for :func:`_rec_strip`."""
    if not v:
        return dup_strip(g)

    w = v - 1

    return dmp_strip([ _rec_strip(c, w) for c in g ], v)


def dmp_validate(f, K=None):
    """
    Return the number of levels in ``f`` and recursively strip it.

    Examples
    ========

    >>> from sympy.polys.densebasic import dmp_validate

    >>> dmp_validate([[], [0, 1, 2], [1]])
    ([[1, 2], [1]], 1)

    >>> dmp_validate([[1], 1])
    Traceback (most recent call last):
    ...
    ValueError: invalid data structure for a multivariate polynomial

    """
    levels = _rec_validate(f, f, 0, K)

    u = levels.pop()

    if not levels:
        return _rec_strip(f, u), u
    else:
        raise ValueError(
            "invalid data structure for a multivariate polynomial")


def dup_reverse(f):
    """
    Compute ``x**n * f(1/x)``, i.e.: reverse ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_reverse

    >>> f = ZZ.map([1, 2, 3, 0])

    >>> dup_reverse(f)
    [3, 2, 1]

    """
    return dup_strip(list(reversed(f)))


def dup_copy(f):
    """
    Create a new copy of a polynomial ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_copy

    >>> f = ZZ.map([1, 2, 3, 0])

    >>> dup_copy([1, 2, 3, 0])
    [1, 2, 3, 0]

    """
    return list(f)


def dmp_copy(f, u):
    """
    Create a new copy of a polynomial ``f`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_copy

    >>> f = ZZ.map([[1], [1, 2]])

    >>> dmp_copy(f, 1)
    [[1], [1, 2]]

    """
    if not u:
        return list(f)

    v = u - 1

    return [ dmp_copy(c, v) for c in f ]


def dup_to_tuple(f):
    """
    Convert `f` into a tuple.

    This is needed for hashing. This is similar to dup_copy().

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_copy

    >>> f = ZZ.map([1, 2, 3, 0])

    >>> dup_copy([1, 2, 3, 0])
    [1, 2, 3, 0]

    """
    return tuple(f)


def dmp_to_tuple(f, u):
    """
    Convert `f` into a nested tuple of tuples.

    This is needed for hashing.  This is similar to dmp_copy().

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_to_tuple

    >>> f = ZZ.map([[1], [1, 2]])

    >>> dmp_to_tuple(f, 1)
    ((1,), (1, 2))

    """
    if not u:
        return tuple(f)
    v = u - 1

    return tuple(dmp_to_tuple(c, v) for c in f)


def dup_normal(f, K):
    """
    Normalize univariate polynomial in the given domain.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_normal

    >>> dup_normal([0, 1, 2, 3], ZZ)
    [1, 2, 3]

    """
    return dup_strip([ K.normal(c) for c in f ])


def dmp_normal(f, u, K):
    """
    Normalize a multivariate polynomial in the given domain.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_normal

    >>> dmp_normal([[], [0, 1, 2]], 1, ZZ)
    [[1, 2]]

    """
    if not u:
        return dup_normal(f, K)

    v = u - 1

    return dmp_strip([ dmp_normal(c, v, K) for c in f ], u)


def dup_convert(f, K0, K1):
    """
    Convert the ground domain of ``f`` from ``K0`` to ``K1``.

    Examples
    ========

    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_convert

    >>> R, x = ring("x", ZZ)

    >>> dup_convert([R(1), R(2)], R.to_domain(), ZZ)
    [1, 2]
    >>> dup_convert([ZZ(1), ZZ(2)], ZZ, R.to_domain())
    [1, 2]

    """
    if K0 is not None and K0 == K1:
        return f
    else:
        return dup_strip([ K1.convert(c, K0) for c in f ])


def dmp_convert(f, u, K0, K1):
    """
    Convert the ground domain of ``f`` from ``K0`` to ``K1``.

    Examples
    ========

    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_convert

    >>> R, x = ring("x", ZZ)

    >>> dmp_convert([[R(1)], [R(2)]], 1, R.to_domain(), ZZ)
    [[1], [2]]
    >>> dmp_convert([[ZZ(1)], [ZZ(2)]], 1, ZZ, R.to_domain())
    [[1], [2]]

    """
    if not u:
        return dup_convert(f, K0, K1)
    if K0 is not None and K0 == K1:
        return f

    v = u - 1

    return dmp_strip([ dmp_convert(c, v, K0, K1) for c in f ], u)


def dup_from_sympy(f, K):
    """
    Convert the ground domain of ``f`` from SymPy to ``K``.

    Examples
    ========

    >>> from sympy import S
    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_from_sympy

    >>> dup_from_sympy([S(1), S(2)], ZZ) == [ZZ(1), ZZ(2)]
    True

    """
    return dup_strip([ K.from_sympy(c) for c in f ])


def dmp_from_sympy(f, u, K):
    """
    Convert the ground domain of ``f`` from SymPy to ``K``.

    Examples
    ========

    >>> from sympy import S
    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_from_sympy

    >>> dmp_from_sympy([[S(1)], [S(2)]], 1, ZZ) == [[ZZ(1)], [ZZ(2)]]
    True

    """
    if not u:
        return dup_from_sympy(f, K)

    v = u - 1

    return dmp_strip([ dmp_from_sympy(c, v, K) for c in f ], u)


def dup_nth(f, n, K):
    """
    Return the ``n``-th coefficient of ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_nth

    >>> f = ZZ.map([1, 2, 3])

    >>> dup_nth(f, 0, ZZ)
    3
    >>> dup_nth(f, 4, ZZ)
    0

    """
    if n < 0:
        raise IndexError("'n' must be non-negative, got %i" % n)
    elif n >= len(f):
        return K.zero
    else:
        return f[dup_degree(f) - n]


def dmp_nth(f, n, u, K):
    """
    Return the ``n``-th coefficient of ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_nth

    >>> f = ZZ.map([[1], [2], [3]])

    >>> dmp_nth(f, 0, 1, ZZ)
    [3]
    >>> dmp_nth(f, 4, 1, ZZ)
    []

    """
    if n < 0:
        raise IndexError("'n' must be non-negative, got %i" % n)
    elif n >= len(f):
        return dmp_zero(u - 1)
    else:
        return f[dmp_degree(f, u) - n]


def dmp_ground_nth(f, N, u, K):
    """
    Return the ground ``n``-th coefficient of ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_ground_nth

    >>> f = ZZ.map([[1], [2, 3]])

    >>> dmp_ground_nth(f, (0, 1), 1, ZZ)
    2

    """
    v = u

    for n in N:
        if n < 0:
            raise IndexError("`n` must be non-negative, got %i" % n)
        elif n >= len(f):
            return K.zero
        else:
            d = dmp_degree(f, v)
            if d == ninf:
                d = -1
            f, v = f[d - n], v - 1

    return f


def dmp_zero_p(f, u):
    """
    Return ``True`` if ``f`` is zero in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.densebasic import dmp_zero_p

    >>> dmp_zero_p([[[[[]]]]], 4)
    True
    >>> dmp_zero_p([[[[[1]]]]], 4)
    False

    """
    while u:
        if len(f) != 1:
            return False

        f = f[0]
        u -= 1

    return not f


def dmp_zero(u):
    """
    Return a multivariate zero.

    Examples
    ========

    >>> from sympy.polys.densebasic import dmp_zero

    >>> dmp_zero(4)
    [[[[[]]]]]

    """
    r = []

    for i in range(u):
        r = [r]

    return r


def dmp_one_p(f, u, K):
    """
    Return ``True`` if ``f`` is one in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_one_p

    >>> dmp_one_p([[[ZZ(1)]]], 2, ZZ)
    True

    """
    return dmp_ground_p(f, K.one, u)


def dmp_one(u, K):
    """
    Return a multivariate one over ``K``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_one

    >>> dmp_one(2, ZZ)
    [[[1]]]

    """
    return dmp_ground(K.one, u)


def dmp_ground_p(f, c, u):
    """
    Return True if ``f`` is constant in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.densebasic import dmp_ground_p

    >>> dmp_ground_p([[[3]]], 3, 2)
    True
    >>> dmp_ground_p([[[4]]], None, 2)
    True

    """
    if c is not None and not c:
        return dmp_zero_p(f, u)

    while u:
        if len(f) != 1:
            return False
        f = f[0]
        u -= 1

    if c is None:
        return len(f) <= 1
    else:
        return f == [c]


def dmp_ground(c, u):
    """
    Return a multivariate constant.

    Examples
    ========

    >>> from sympy.polys.densebasic import dmp_ground

    >>> dmp_ground(3, 5)
    [[[[[[3]]]]]]
    >>> dmp_ground(1, -1)
    1

    """
    if not c:
        return dmp_zero(u)

    for i in range(u + 1):
        c = [c]

    return c


def dmp_zeros(n, u, K):
    """
    Return a list of multivariate zeros.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_zeros

    >>> dmp_zeros(3, 2, ZZ)
    [[[[]]], [[[]]], [[[]]]]
    >>> dmp_zeros(3, -1, ZZ)
    [0, 0, 0]

    """
    if not n:
        return []

    if u < 0:
        return [K.zero]*n
    else:
        return [ dmp_zero(u) for i in range(n) ]


def dmp_grounds(c, n, u):
    """
    Return a list of multivariate constants.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_grounds

    >>> dmp_grounds(ZZ(4), 3, 2)
    [[[[4]]], [[[4]]], [[[4]]]]
    >>> dmp_grounds(ZZ(4), 3, -1)
    [4, 4, 4]

    """
    if not n:
        return []

    if u < 0:
        return [c]*n
    else:
        return [ dmp_ground(c, u) for i in range(n) ]


def dmp_negative_p(f, u, K):
    """
    Return ``True`` if ``LC(f)`` is negative.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_negative_p

    >>> dmp_negative_p([[ZZ(1)], [-ZZ(1)]], 1, ZZ)
    False
    >>> dmp_negative_p([[-ZZ(1)], [ZZ(1)]], 1, ZZ)
    True

    """
    return K.is_negative(dmp_ground_LC(f, u, K))


def dmp_positive_p(f, u, K):
    """
    Return ``True`` if ``LC(f)`` is positive.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_positive_p

    >>> dmp_positive_p([[ZZ(1)], [-ZZ(1)]], 1, ZZ)
    True
    >>> dmp_positive_p([[-ZZ(1)], [ZZ(1)]], 1, ZZ)
    False

    """
    return K.is_positive(dmp_ground_LC(f, u, K))


def dup_from_dict(f, K):
    """
    Create a ``K[x]`` polynomial from a ``dict``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_from_dict

    >>> dup_from_dict({(0,): ZZ(7), (2,): ZZ(5), (4,): ZZ(1)}, ZZ)
    [1, 0, 5, 0, 7]
    >>> dup_from_dict({}, ZZ)
    []

    """
    if not f:
        return []

    n, h = max(f.keys()), []

    if isinstance(n, int):
        for k in range(n, -1, -1):
            h.append(f.get(k, K.zero))
    else:
        (n,) = n

        for k in range(n, -1, -1):
            h.append(f.get((k,), K.zero))

    return dup_strip(h)


def dup_from_raw_dict(f, K):
    """
    Create a ``K[x]`` polynomial from a raw ``dict``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_from_raw_dict

    >>> dup_from_raw_dict({0: ZZ(7), 2: ZZ(5), 4: ZZ(1)}, ZZ)
    [1, 0, 5, 0, 7]

    """
    if not f:
        return []

    n, h = max(f.keys()), []

    for k in range(n, -1, -1):
        h.append(f.get(k, K.zero))

    return dup_strip(h)


def dmp_from_dict(f, u, K):
    """
    Create a ``K[X]`` polynomial from a ``dict``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_from_dict

    >>> dmp_from_dict({(0, 0): ZZ(3), (0, 1): ZZ(2), (2, 1): ZZ(1)}, 1, ZZ)
    [[1, 0], [], [2, 3]]
    >>> dmp_from_dict({}, 0, ZZ)
    []

    """
    if not u:
        return dup_from_dict(f, K)
    if not f:
        return dmp_zero(u)

    coeffs = {}

    for monom, coeff in f.items():
        head, tail = monom[0], monom[1:]

        if head in coeffs:
            coeffs[head][tail] = coeff
        else:
            coeffs[head] = { tail: coeff }

    n, v, h = max(coeffs.keys()), u - 1, []

    for k in range(n, -1, -1):
        coeff = coeffs.get(k)

        if coeff is not None:
            h.append(dmp_from_dict(coeff, v, K))
        else:
            h.append(dmp_zero(v))

    return dmp_strip(h, u)


def dup_to_dict(f, K=None, zero=False):
    """
    Convert ``K[x]`` polynomial to a ``dict``.

    Examples
    ========

    >>> from sympy.polys.densebasic import dup_to_dict

    >>> dup_to_dict([1, 0, 5, 0, 7])
    {(0,): 7, (2,): 5, (4,): 1}
    >>> dup_to_dict([])
    {}

    """
    if not f and zero:
        return {(0,): K.zero}

    n, result = len(f) - 1, {}

    for k in range(0, n + 1):
        if f[n - k]:
            result[(k,)] = f[n - k]

    return result


def dup_to_raw_dict(f, K=None, zero=False):
    """
    Convert a ``K[x]`` polynomial to a raw ``dict``.

    Examples
    ========

    >>> from sympy.polys.densebasic import dup_to_raw_dict

    >>> dup_to_raw_dict([1, 0, 5, 0, 7])
    {0: 7, 2: 5, 4: 1}

    """
    if not f and zero:
        return {0: K.zero}

    n, result = len(f) - 1, {}

    for k in range(0, n + 1):
        if f[n - k]:
            result[k] = f[n - k]

    return result


def dmp_to_dict(f, u, K=None, zero=False):
    """
    Convert a ``K[X]`` polynomial to a ``dict````.

    Examples
    ========

    >>> from sympy.polys.densebasic import dmp_to_dict

    >>> dmp_to_dict([[1, 0], [], [2, 3]], 1)
    {(0, 0): 3, (0, 1): 2, (2, 1): 1}
    >>> dmp_to_dict([], 0)
    {}

    """
    if not u:
        return dup_to_dict(f, K, zero=zero)

    if dmp_zero_p(f, u) and zero:
        return {(0,)*(u + 1): K.zero}

    n, v, result = dmp_degree(f, u), u - 1, {}

    if n == ninf:
        n = -1

    for k in range(0, n + 1):
        h = dmp_to_dict(f[n - k], v)

        for exp, coeff in h.items():
            result[(k,) + exp] = coeff

    return result


def dmp_swap(f, i, j, u, K):
    """
    Transform ``K[..x_i..x_j..]`` to ``K[..x_j..x_i..]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_swap

    >>> f = ZZ.map([[[2], [1, 0]], []])

    >>> dmp_swap(f, 0, 1, 2, ZZ)
    [[[2], []], [[1, 0], []]]
    >>> dmp_swap(f, 1, 2, 2, ZZ)
    [[[1], [2, 0]], [[]]]
    >>> dmp_swap(f, 0, 2, 2, ZZ)
    [[[1, 0]], [[2, 0], []]]

    """
    if i < 0 or j < 0 or i > u or j > u:
        raise IndexError("0 <= i < j <= %s expected" % u)
    elif i == j:
        return f

    F, H = dmp_to_dict(f, u), {}

    for exp, coeff in F.items():
        H[exp[:i] + (exp[j],) +
          exp[i + 1:j] +
          (exp[i],) + exp[j + 1:]] = coeff

    return dmp_from_dict(H, u, K)


def dmp_permute(f, P, u, K):
    """
    Return a polynomial in ``K[x_{P(1)},..,x_{P(n)}]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_permute

    >>> f = ZZ.map([[[2], [1, 0]], []])

    >>> dmp_permute(f, [1, 0, 2], 2, ZZ)
    [[[2], []], [[1, 0], []]]
    >>> dmp_permute(f, [1, 2, 0], 2, ZZ)
    [[[1], []], [[2, 0], []]]

    """
    F, H = dmp_to_dict(f, u), {}

    for exp, coeff in F.items():
        new_exp = [0]*len(exp)

        for e, p in zip(exp, P):
            new_exp[p] = e

        H[tuple(new_exp)] = coeff

    return dmp_from_dict(H, u, K)


def dmp_nest(f, l, K):
    """
    Return a multivariate value nested ``l``-levels.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_nest

    >>> dmp_nest([[ZZ(1)]], 2, ZZ)
    [[[[1]]]]

    """
    if not isinstance(f, list):
        return dmp_ground(f, l)

    for i in range(l):
        f = [f]

    return f


def dmp_raise(f, l, u, K):
    """
    Return a multivariate polynomial raised ``l``-levels.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_raise

    >>> f = ZZ.map([[], [1, 2]])

    >>> dmp_raise(f, 2, 1, ZZ)
    [[[[]]], [[[1]], [[2]]]]

    """
    if not l:
        return f

    if not u:
        if not f:
            return dmp_zero(l)

        k = l - 1

        return [ dmp_ground(c, k) for c in f ]

    v = u - 1

    return [ dmp_raise(c, l, v, K) for c in f ]


def dup_deflate(f, K):
    """
    Map ``x**m`` to ``y`` in a polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_deflate

    >>> f = ZZ.map([1, 0, 0, 1, 0, 0, 1])

    >>> dup_deflate(f, ZZ)
    (3, [1, 1, 1])

    """
    if dup_degree(f) <= 0:
        return 1, f

    g = 0

    for i in range(len(f)):
        if not f[-i - 1]:
            continue

        g = igcd(g, i)

        if g == 1:
            return 1, f

    return g, f[::g]


def dmp_deflate(f, u, K):
    """
    Map ``x_i**m_i`` to ``y_i`` in a polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_deflate

    >>> f = ZZ.map([[1, 0, 0, 2], [], [3, 0, 0, 4]])

    >>> dmp_deflate(f, 1, ZZ)
    ((2, 3), [[1, 2], [3, 4]])

    """
    if dmp_zero_p(f, u):
        return (1,)*(u + 1), f

    F = dmp_to_dict(f, u)
    B = [0]*(u + 1)

    for M in F.keys():
        for i, m in enumerate(M):
            B[i] = igcd(B[i], m)

    for i, b in enumerate(B):
        if not b:
            B[i] = 1

    B = tuple(B)

    if all(b == 1 for b in B):
        return B, f

    H = {}

    for A, coeff in F.items():
        N = [ a // b for a, b in zip(A, B) ]
        H[tuple(N)] = coeff

    return B, dmp_from_dict(H, u, K)


def dup_multi_deflate(polys, K):
    """
    Map ``x**m`` to ``y`` in a set of polynomials in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_multi_deflate

    >>> f = ZZ.map([1, 0, 2, 0, 3])
    >>> g = ZZ.map([4, 0, 0])

    >>> dup_multi_deflate((f, g), ZZ)
    (2, ([1, 2, 3], [4, 0]))

    """
    G = 0

    for p in polys:
        if dup_degree(p) <= 0:
            return 1, polys

        g = 0

        for i in range(len(p)):
            if not p[-i - 1]:
                continue

            g = igcd(g, i)

            if g == 1:
                return 1, polys

        G = igcd(G, g)

    return G, tuple([ p[::G] for p in polys ])


def dmp_multi_deflate(polys, u, K):
    """
    Map ``x_i**m_i`` to ``y_i`` in a set of polynomials in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_multi_deflate

    >>> f = ZZ.map([[1, 0, 0, 2], [], [3, 0, 0, 4]])
    >>> g = ZZ.map([[1, 0, 2], [], [3, 0, 4]])

    >>> dmp_multi_deflate((f, g), 1, ZZ)
    ((2, 1), ([[1, 0, 0, 2], [3, 0, 0, 4]], [[1, 0, 2], [3, 0, 4]]))

    """
    if not u:
        M, H = dup_multi_deflate(polys, K)
        return (M,), H

    F, B = [], [0]*(u + 1)

    for p in polys:
        f = dmp_to_dict(p, u)

        if not dmp_zero_p(p, u):
            for M in f.keys():
                for i, m in enumerate(M):
                    B[i] = igcd(B[i], m)

        F.append(f)

    for i, b in enumerate(B):
        if not b:
            B[i] = 1

    B = tuple(B)

    if all(b == 1 for b in B):
        return B, polys

    H = []

    for f in F:
        h = {}

        for A, coeff in f.items():
            N = [ a // b for a, b in zip(A, B) ]
            h[tuple(N)] = coeff

        H.append(dmp_from_dict(h, u, K))

    return B, tuple(H)


def dup_inflate(f, m, K):
    """
    Map ``y`` to ``x**m`` in a polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_inflate

    >>> f = ZZ.map([1, 1, 1])

    >>> dup_inflate(f, 3, ZZ)
    [1, 0, 0, 1, 0, 0, 1]

    """
    if m <= 0:
        raise IndexError("'m' must be positive, got %s" % m)
    if m == 1 or not f:
        return f

    result = [f[0]]

    for coeff in f[1:]:
        result.extend([K.zero]*(m - 1))
        result.append(coeff)

    return result


def _rec_inflate(g, M, v, i, K):
    """Recursive helper for :func:`dmp_inflate`."""
    if not v:
        return dup_inflate(g, M[i], K)
    if M[i] <= 0:
        raise IndexError("all M[i] must be positive, got %s" % M[i])

    w, j = v - 1, i + 1

    g = [ _rec_inflate(c, M, w, j, K) for c in g ]

    result = [g[0]]

    for coeff in g[1:]:
        for _ in range(1, M[i]):
            result.append(dmp_zero(w))

        result.append(coeff)

    return result


def dmp_inflate(f, M, u, K):
    """
    Map ``y_i`` to ``x_i**k_i`` in a polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_inflate

    >>> f = ZZ.map([[1, 2], [3, 4]])

    >>> dmp_inflate(f, (2, 3), 1, ZZ)
    [[1, 0, 0, 2], [], [3, 0, 0, 4]]

    """
    if not u:
        return dup_inflate(f, M[0], K)

    if all(m == 1 for m in M):
        return f
    else:
        return _rec_inflate(f, M, u, 0, K)


def dmp_exclude(f, u, K):
    """
    Exclude useless levels from ``f``.

    Return the levels excluded, the new excluded ``f``, and the new ``u``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_exclude

    >>> f = ZZ.map([[[1]], [[1], [2]]])

    >>> dmp_exclude(f, 2, ZZ)
    ([2], [[1], [1, 2]], 1)

    """
    if not u or dmp_ground_p(f, None, u):
        return [], f, u

    J, F = [], dmp_to_dict(f, u)

    for j in range(0, u + 1):
        for monom in F.keys():
            if monom[j]:
                break
        else:
            J.append(j)

    if not J:
        return [], f, u

    f = {}

    for monom, coeff in F.items():
        monom = list(monom)

        for j in reversed(J):
            del monom[j]

        f[tuple(monom)] = coeff

    u -= len(J)

    return J, dmp_from_dict(f, u, K), u


def dmp_include(f, J, u, K):
    """
    Include useless levels in ``f``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_include

    >>> f = ZZ.map([[1], [1, 2]])

    >>> dmp_include(f, [2], 1, ZZ)
    [[[1]], [[1], [2]]]

    """
    if not J:
        return f

    F, f = dmp_to_dict(f, u), {}

    for monom, coeff in F.items():
        monom = list(monom)

        for j in J:
            monom.insert(j, 0)

        f[tuple(monom)] = coeff

    u += len(J)

    return dmp_from_dict(f, u, K)


def dmp_inject(f, u, K, front=False):
    """
    Convert ``f`` from ``K[X][Y]`` to ``K[X,Y]``.

    Examples
    ========

    >>> from sympy.polys.rings import ring
    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_inject

    >>> R, x,y = ring("x,y", ZZ)

    >>> dmp_inject([R(1), x + 2], 0, R.to_domain())
    ([[[1]], [[1], [2]]], 2)
    >>> dmp_inject([R(1), x + 2], 0, R.to_domain(), front=True)
    ([[[1]], [[1, 2]]], 2)

    """
    f, h = dmp_to_dict(f, u), {}

    v = K.ngens - 1

    for f_monom, g in f.items():
        g = g.to_dict()

        for g_monom, c in g.items():
            if front:
                h[g_monom + f_monom] = c
            else:
                h[f_monom + g_monom] = c

    w = u + v + 1

    return dmp_from_dict(h, w, K.dom), w


def dmp_eject(f, u, K, front=False):
    """
    Convert ``f`` from ``K[X,Y]`` to ``K[X][Y]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_eject

    >>> dmp_eject([[[1]], [[1], [2]]], 2, ZZ['x', 'y'])
    [1, x + 2]

    """
    f, h = dmp_to_dict(f, u), {}

    n = K.ngens
    v = u - K.ngens + 1

    for monom, c in f.items():
        if front:
            g_monom, f_monom = monom[:n], monom[n:]
        else:
            g_monom, f_monom = monom[-n:], monom[:-n]

        if f_monom in h:
            h[f_monom][g_monom] = c
        else:
            h[f_monom] = {g_monom: c}

    for monom, c in h.items():
        h[monom] = K(c)

    return dmp_from_dict(h, v - 1, K)


def dup_terms_gcd(f, K):
    """
    Remove GCD of terms from ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dup_terms_gcd

    >>> f = ZZ.map([1, 0, 1, 0, 0])

    >>> dup_terms_gcd(f, ZZ)
    (2, [1, 0, 1])

    """
    if dup_TC(f, K) or not f:
        return 0, f

    i = 0

    for c in reversed(f):
        if not c:
            i += 1
        else:
            break

    return i, f[:-i]


def dmp_terms_gcd(f, u, K):
    """
    Remove GCD of terms from ``f`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_terms_gcd

    >>> f = ZZ.map([[1, 0], [1, 0, 0], [], []])

    >>> dmp_terms_gcd(f, 1, ZZ)
    ((2, 1), [[1], [1, 0]])

    """
