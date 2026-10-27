
    discrete_log

    References
    ==========

    .. [1] "Handbook of applied cryptography", Menezes, A. J., Van, O. P. C., &
        Vanstone, S. A. (1997).
    """
    a %= n
    b %= n

    if order is None:
        order = n_order(b, n)
    randint = _randint(rseed)

    for i in range(retries):
        aa = randint(1, order - 1)
        ba = randint(1, order - 1)
        xa = pow(b, aa, n) * pow(a, ba, n) % n

        c = xa % 3
        if c == 0:
            xb = a * xa % n
            ab = aa
            bb = (ba + 1) % order
        elif c == 1:
            xb = xa * xa % n
            ab = (aa + aa) % order
            bb = (ba + ba) % order
        else:
            xb = b * xa % n
            ab = (aa + 1) % order
            bb = ba

        for j in range(order):
            c = xa % 3
            if c == 0:
                xa = a * xa % n
                ba = (ba + 1) % order
            elif c == 1:
                xa = xa * xa % n
                aa = (aa + aa) % order
                ba = (ba + ba) % order
            else:
                xa = b * xa % n
                aa = (aa + 1) % order

            c = xb % 3
            if c == 0:
                xb = a * xb % n
                bb = (bb + 1) % order
            elif c == 1:
                xb = xb * xb % n
                ab = (ab + ab) % order
                bb = (bb + bb) % order
            else:
                xb = b * xb % n
                ab = (ab + 1) % order

            c = xb % 3
            if c == 0:
                xb = a * xb % n
                bb = (bb + 1) % order
            elif c == 1:
                xb = xb * xb % n
                ab = (ab + ab) % order
                bb = (bb + bb) % order
            else:
                xb = b * xb % n
                ab = (ab + 1) % order

            if xa == xb:
                r = (ba - bb) % order
                try:
                    e = invert(r, order) * (ab - aa) % order
                    if (pow(b, e, n) - a) % n == 0:
                        return e
                except ZeroDivisionError:
                    pass
                break
    raise ValueError("Pollard's Rho failed to find logarithm")


def _discrete_log_is_smooth(n: int, factorbase: list):
    """Try to factor n with respect to a given factorbase.
    Upon success a list of exponents with respect to the factorbase is returned.
    Otherwise None."""
    factors = [0]*len(factorbase)
    for i, p in enumerate(factorbase):
        while n % p == 0: # divide by p as many times as possible
            factors[i] += 1
            n = n // p
    if n != 1:
        return None # the number factors if at the end nothing is left
    return factors


def _discrete_log_index_calculus(n, a, b, order, rseed=None):
    """
    Index Calculus algorithm for computing the discrete logarithm of ``a`` to
    the base ``b`` modulo ``n``.

    The group order must be given and prime. It is not suitable for small orders
    and the algorithm might fail to find a solution in such situations.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _discrete_log_index_calculus
    >>> _discrete_log_index_calculus(24570203447, 23859756228, 2, 12285101723)
    4519867240

    See Also
    ========

    discrete_log

    References
    ==========

    .. [1] "Handbook of applied cryptography", Menezes, A. J., Van, O. P. C., &
        Vanstone, S. A. (1997).
    """
    randint = _randint(rseed)
    from math import sqrt, exp, log
    a %= n
    b %= n
    # assert isprime(order), "The order of the base must be prime."
    # First choose a heuristic the bound B for the factorbase.
    # We have added an extra term to the asymptotic value which
    # is closer to the theoretical optimum for n up to 2^70.
    B = int(exp(0.5 * sqrt( log(n) * log(log(n)) )*( 1 + 1/log(log(n)) )))
    max = 5 * B * B  # expected number of tries to find a relation
    factorbase = list(primerange(B)) # compute the factorbase
    lf = len(factorbase) # length of the factorbase
    ordermo = order-1
    abx = a
    for x in range(order):
        if abx == 1:
            return (order - x) % order
        relationa = _discrete_log_is_smooth(abx, factorbase)
        if relationa:
            relationa = [r % order for r in relationa] + [x]
            break
        abx = abx * b % n # abx = a*pow(b, x, n) % n

    else:
        raise ValueError("Index Calculus failed")

    relations = [None] * lf
    k = 1  # number of relations found
    kk = 0
    while k < 3 * lf and kk < max:  # find relations for all primes in our factor base
        x = randint(1,ordermo)
        relation = _discrete_log_is_smooth(pow(b,x,n), factorbase)
        if relation is None:
            kk += 1
            continue
        k += 1
        kk = 0
        relation += [ x ]
        index = lf  # determine the index of the first nonzero entry
        for i in range(lf):
            ri = relation[i] % order
            if ri> 0 and relations[i] is not None:  # make this entry zero if we can
                for j in range(lf+1):
                    relation[j] = (relation[j] - ri*relations[i][j]) % order
            else:
                relation[i] = ri
            if relation[i] > 0 and index == lf:  # is this the index of the first nonzero entry?
                index = i
        if index == lf or relations[index] is not None:  # the relation contains no new information
            continue
        # the relation contains new information
        rinv = pow(relation[index],-1,order)  # normalize the first nonzero entry
        for j in range(index,lf+1):
            relation[j] = rinv * relation[j] % order
        relations[index] = relation
        for i in range(lf):  # subtract the new relation from the one for a
            if relationa[i] > 0 and relations[i] is not None:
                rbi = relationa[i]
                for j in range(lf+1):
                    relationa[j] = (relationa[j] - rbi*relations[i][j]) % order
            if relationa[i] > 0:  # the index of the first nonzero entry
                break  # we do not need to reduce further at this point
        else:  # all unknowns are gone
            #print(f"Success after {k} relations out of {lf}")
            x = (order -relationa[lf]) % order
            if pow(b,x,n) == a:
                return x
            raise ValueError("Index Calculus failed")
    raise ValueError("Index Calculus failed")


def _discrete_log_pohlig_hellman(n, a, b, order=None, order_factors=None):
    """
    Pohlig-Hellman algorithm for computing the discrete logarithm of ``a`` to
    the base ``b`` modulo ``n``.

    In order to compute the discrete logarithm, the algorithm takes advantage
    of the factorization of the group order. It is more efficient when the
    group order factors into many small primes.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _discrete_log_pohlig_hellman
    >>> _discrete_log_pohlig_hellman(251, 210, 71)
    197

    See Also
    ========

    discrete_log

    References
    ==========

    .. [1] "Handbook of applied cryptography", Menezes, A. J., Van, O. P. C., &
        Vanstone, S. A. (1997).
    """
    from .modular import crt
    a %= n
    b %= n

    if order is None:
        order = n_order(b, n)
    if order_factors is None:
        order_factors = factorint(order)
    l = [0] * len(order_factors)

    for i, (pi, ri) in enumerate(order_factors.items()):
        for j in range(ri):
            aj = pow(a * pow(b, -l[i], n), order // pi**(j + 1), n)
            bj = pow(b, order // pi, n)
            cj = discrete_log(n, aj, bj, pi, True)
            l[i] += cj * pi**j

    d, _ = crt([pi**ri for pi, ri in order_factors.items()], l)
    return d


def discrete_log(n, a, b, order=None, prime_order=None):
    """
    Compute the discrete logarithm of ``a`` to the base ``b`` modulo ``n``.

    This is a recursive function to reduce the discrete logarithm problem in
    cyclic groups of composite order to the problem in cyclic groups of prime
    order.

    It employs different algorithms depending on the problem (subgroup order
    size, prime order or not):

        * Trial multiplication
        * Baby-step giant-step
        * Pollard's Rho
        * Index Calculus
        * Pohlig-Hellman

    Examples
    ========

    >>> from sympy.ntheory import discrete_log
    >>> discrete_log(41, 15, 7)
    3

    References
    ==========

    .. [1] https://mathworld.wolfram.com/DiscreteLogarithm.html
    .. [2] "Handbook of applied cryptography", Menezes, A. J., Van, O. P. C., &
        Vanstone, S. A. (1997).

    """
    from math import sqrt, log
    n, a, b = as_int(n), as_int(a), as_int(b)

    if n < 1:
        raise ValueError("n should be positive")
    if n == 1:
        return 0

    if order is None:
        # Compute the order and its factoring in one pass
        # order = totient(n), factors = factorint(order)
        factors = {}
        for px, kx in factorint(n).items():
            if kx > 1:
                if px in factors:
                    factors[px] += kx - 1
                else:
                    factors[px] = kx - 1
            for py, ky in factorint(px - 1).items():
                if py in factors:
                    factors[py] += ky
                else:
                    factors[py] = ky
        order = 1
        for px, kx in factors.items():
            order *= px**kx
        # Now the `order` is the order of the group and factors = factorint(order)
        # The order of `b` divides the order of the group.
        order_factors = {}
        for p, e in factors.items():
            i = 0
            for _ in range(e):
                if pow(b, order // p, n) == 1:
                    order //= p
                    i += 1
                else:
                    break
            if i < e:
                order_factors[p] = e - i

    if prime_order is None:
        prime_order = isprime(order)

    if order < 1000:
        return _discrete_log_trial_mul(n, a, b, order)
    elif prime_order:
        # Shanks and Pollard rho are O(sqrt(order)) while index calculus is O(exp(2*sqrt(log(n)log(log(n)))))
        # we compare the expected running times to determine the algorithm which is expected to be faster
        if 4*sqrt(log(n)*log(log(n))) < log(order) - 10:  # the number 10 was determined experimental
            return _discrete_log_index_calculus(n, a, b, order)
        elif order < 1000000000000:
            # Shanks seems typically faster, but uses O(sqrt(order)) memory
            return _discrete_log_shanks_steps(n, a, b, order)
        return _discrete_log_pollard_rho(n, a, b, order)

    return _discrete_log_pohlig_hellman(n, a, b, order, order_factors)



def quadratic_congruence(a, b, c, n):
    r"""
    Find the solutions to `a x^2 + b x + c \equiv 0 \pmod{n}`.

    Parameters
    ==========

    a : int
    b : int
    c : int
    n : int
        A positive integer.

    Returns
    =======

    list[int] :
        A sorted list of solutions. If no solution exists, ``[]``.

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import quadratic_congruence
    >>> quadratic_congruence(2, 5, 3, 7) # 2x^2 + 5x + 3 = 0 (mod 7)
    [2, 6]
    >>> quadratic_congruence(8, 6, 4, 15) # No solution
    []

    See Also
    ========

    polynomial_congruence : Solve the polynomial congruence

    """
    a = as_int(a)
    b = as_int(b)
    c = as_int(c)
    n = as_int(n)
    if n <= 1:
        raise ValueError("n should be an integer greater than 1")
    a %= n
    b %= n
    c %= n

    if a == 0:
        return linear_congruence(b, -c, n)
    if n == 2:
        # assert a == 1
        roots = []
        if c == 0:
            roots.append(0)
        if (b + c) % 2:
            roots.append(1)
        return roots
    if gcd(2*a, n) == 1:
        inv_a = invert(a, n)
        b *= inv_a
        c *= inv_a
        if b % 2:
            b += n
        b >>= 1
        return sorted((i - b) % n for i in sqrt_mod_iter(b**2 - c, n))
    res = set()
    for i in sqrt_mod_iter(b**2 - 4*a*c, 4*a*n):
        q, rem = divmod(i - b, 2*a)
        if rem == 0:
            res.add(q % n)

    return sorted(res)


def _valid_expr(expr):
    """
    return coefficients of expr if it is a univariate polynomial
    with integer coefficients else raise a ValueError.
    """

    if not expr.is_polynomial():
        raise ValueError("The expression should be a polynomial")
    polynomial = Poly(expr)
    if not polynomial.is_univariate:
        raise ValueError("The expression should be univariate")
    if not polynomial.domain == ZZ:
        raise ValueError("The expression should should have integer coefficients")
    return polynomial.all_coeffs()


def polynomial_congruence(expr, m):
    """
    Find the solutions to a polynomial congruence equation modulo m.

    Parameters
    ==========

    expr : integer coefficient polynomial
    m : positive integer

    Examples
    ========

    >>> from sympy.ntheory import polynomial_congruence
    >>> from sympy.abc import x
    >>> expr = x**6 - 2*x**5 -35
    >>> polynomial_congruence(expr, 6125)
    [3257]

    See Also
    ========

    sympy.polys.galoistools.gf_csolve : low level solving routine used by this routine

    """
    coefficients = _valid_expr(expr)
    coefficients = [num % m for num in coefficients]
    rank = len(coefficients)
    if rank == 3:
        return quadratic_congruence(*coefficients, m)
    if rank == 2:
        return quadratic_congruence(0, *coefficients, m)
    if coefficients[0] == 1 and 1 + coefficients[-1] == sum(coefficients):
        return nthroot_mod(-coefficients[-1], rank - 1, m, True)
    return gf_csolve(coefficients, m)


def binomial_mod(n, m, k):
    """Compute ``binomial(n, m) % k``.

    Explanation
    ===========

    Returns ``binomial(n, m) % k`` using a generalization of Lucas'
    Theorem for prime powers given by Granville [1]_, in conjunction with
    the Chinese Remainder Theorem.  The residue for each prime power
    is calculated in time O(log^2(n) + q^4*log(n)log(p) + q^4*p*log^3(p)).

    Parameters
    ==========

    n : an integer
    m : an integer
    k : a positive integer

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import binomial_mod
    >>> binomial_mod(10, 2, 6)  # binomial(10, 2) = 45
    3
    >>> binomial_mod(17, 9, 10)  # binomial(17, 9) = 24310
    0

    References
    ==========

    .. [1] Binomial coefficients modulo prime powers, Andrew Granville,
        Available: https://web.archive.org/web/20170202003812/http://www.dms.umontreal.ca/~andrew/PDF/BinCoeff.pdf
    """
    if k < 1: raise ValueError('k is required to be positive')
    # We decompose q into a product of prime powers and apply
    # the generalization of Lucas' Theorem given by Granville
    # to obtain binomial(n, k) mod p^e, and then use the Chinese
    # Remainder Theorem to obtain the result mod q
    if n < 0 or m < 0 or m > n: return 0
    factorisation = factorint(k)
    residues = [_binomial_mod_prime_power(n, m, p, e) for p, e in factorisation.items()]
    return crt([p**pw for p, pw in factorisation.items()], residues, check=False)[0]


def _binomial_mod_prime_power(n, m, p, q):
    """Compute ``binomial(n, m) % p**q`` for a prime ``p``.

    Parameters
    ==========

    n : positive integer
    m : a nonnegative integer
    p : a prime
    q : a positive integer (the prime exponent)

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _binomial_mod_prime_power
    >>> _binomial_mod_prime_power(10, 2, 3, 2)  # binomial(10, 2) = 45
    0
    >>> _binomial_mod_prime_power(17, 9, 2, 4)  # binomial(17, 9) = 24310
    6

    References
    ==========

    .. [1] Binomial coefficients modulo prime powers, Andrew Granville,
        Available: https://web.archive.org/web/20170202003812/http://www.dms.umontreal.ca/~andrew/PDF/BinCoeff.pdf
    """
    # Function/variable naming within this function follows Ref.[1]
    # n!_p will be used to denote the product of integers <= n not divisible by
    # p, with binomial(n, m)_p the same as binomial(n, m), but defined using
    # n!_p in place of n!
    modulo = pow(p, q)

    def up_factorial(u):
        """Compute (u*p)!_p modulo p^q."""
        r = q // 2
        fac = prod = 1
        if r == 1 and p == 2 or 2*r + 1 in (p, p*p):
            if q % 2 == 1: r += 1
            modulo, div = pow(p, 2*r), pow(p, 2*r - q)
        else:
            modulo, div = pow(p, 2*r + 1), pow(p, (2*r + 1) - q)
        for j in range(1, r + 1):
            for mul in range((j - 1)*p + 1, j*p):  # ignore jp itself
                fac *= mul
                fac %= modulo
            bj_ = bj(u, j, r)
            prod *= pow(fac, bj_, modulo)
            prod %= modulo
        if p == 2:
            sm = u // 2
            for j in range(1, r + 1): sm += j//2 * bj(u, j, r)
            if sm % 2 == 1: prod *= -1
        prod %= modulo//div
        return prod % modulo

    def bj(u, j, r):
        """Compute the exponent of (j*p)!_p in the calculation of (u*p)!_p."""
        prod = u
        for i in range(1, r + 1):
            if i != j: prod *= u*u - i*i
        for i in range(1, r + 1):
            if i != j: prod //= j*j - i*i
        return prod // j

    def up_plus_v_binom(u, v):
        """Compute binomial(u*p + v, v)_p modulo p^q."""
        prod = 1
        div = invert(factorial(v), modulo)
        for j in range(1, q):
            b = div
            for v_ in range(j*p + 1, j*p + v + 1):
                b *= v_
                b %= modulo
            aj = u
            for i in range(1, q):
                if i != j: aj *= u - i
            for i in range(1, q):
                if i != j: aj //= j - i
            aj //= j
            prod *= pow(b, aj, modulo)
            prod %= modulo
        return prod

    @recurrence_memo([1])
    def factorial(v, prev):
        """Compute v! modulo p^q."""
        return v*prev[-1] % modulo

    def factorial_p(n):
        """Compute n!_p modulo p^q."""
        u, v = divmod(n, p)
        return (factorial(v) * up_factorial(u) * up_plus_v_binom(u, v)) % modulo

    prod = 1
    Nj, Mj, Rj = n, m, n - m
    # e0 will be the p-adic valuation of binomial(n, m) at p
    e0 = carry = eq_1 = j = 0
    while Nj:
        numerator = factorial_p(Nj % modulo)
        denominator = factorial_p(Mj % modulo) * factorial_p(Rj % modulo) % modulo
        Nj, (Mj, mj), (Rj, rj) = Nj//p, divmod(Mj, p), divmod(Rj, p)
        carry = (mj + rj + carry) // p
        e0 += carry
        if j >= q - 1: eq_1 += carry
        prod *= numerator * invert(denominator, modulo)
        prod %= modulo
        j += 1

    mul = pow(1 if p == 2 and q >= 3 else -1, eq_1, modulo)
    return (pow(p, e0, modulo) * mul * prod) % modulo
<|endoftext|>from sympy.utilities.misc import as_int


def binomial_coefficients(n):
    """Return a dictionary containing pairs :math:`{(k1,k2) : C_kn}` where
    :math:`C_kn` are binomial coefficients and :math:`n=k1+k2`.

    Examples
    ========

    >>> from sympy.ntheory import binomial_coefficients
    >>> binomial_coefficients(9)
    {(0, 9): 1, (1, 8): 9, (2, 7): 36, (3, 6): 84,
     (4, 5): 126, (5, 4): 126, (6, 3): 84, (7, 2): 36, (8, 1): 9, (9, 0): 1}

    See Also
    ========

    binomial_coefficients_list, multinomial_coefficients
    """
    n = as_int(n)
    d = {(0, n): 1, (n, 0): 1}
    a = 1
    for k in range(1, n//2 + 1):
        a = (a * (n - k + 1))//k
        d[k, n - k] = d[n - k, k] = a
    return d


def binomial_coefficients_list(n):
    """ Return a list of binomial coefficients as rows of the Pascal's
    triangle.

    Examples
    ========

    >>> from sympy.ntheory import binomial_coefficients_list
    >>> binomial_coefficients_list(9)
    [1, 9, 36, 84, 126, 126, 84, 36, 9, 1]

    See Also
    ========

    binomial_coefficients, multinomial_coefficients
    """
    n = as_int(n)
    d = [1] * (n + 1)
    a = 1
    for k in range(1, n//2 + 1):
        a = (a * (n - k + 1))//k
        d[k] = d[n - k] = a
    return d


def multinomial_coefficients(m, n):
    r"""Return a dictionary containing pairs ``{(k1,k2,..,km) : C_kn}``
    where ``C_kn`` are multinomial coefficients such that
    ``n=k1+k2+..+km``.

    Examples
    ========

    >>> from sympy.ntheory import multinomial_coefficients
    >>> multinomial_coefficients(2, 5) # indirect doctest
    {(0, 5): 1, (1, 4): 5, (2, 3): 10, (3, 2): 10, (4, 1): 5, (5, 0): 1}

    Notes
    =====

    The algorithm is based on the following result:

    .. math::
        \binom{n}{k_1, \ldots, k_m} =
        \frac{k_1 + 1}{n - k_1} \sum_{i=2}^m \binom{n}{k_1 + 1, \ldots, k_i - 1, \ldots}

    Code contributed to Sage by Yann Laigle-Chapuy, copied with permission
    of the author.

    See Also
    ========

    binomial_coefficients_list, binomial_coefficients
    """
    m = as_int(m)
    n = as_int(n)
    if not m:
        if n:
            return {}
        return {(): 1}
    if m == 2:
        return binomial_coefficients(n)
    if m >= 2*n and n > 1:
        return dict(multinomial_coefficients_iterator(m, n))
    t = [n] + [0] * (m - 1)
    r = {tuple(t): 1}
    if n:
        j = 0  # j will be the leftmost nonzero position
    else:
        j = m
    # enumerate tuples in co-lex order
    while j < m - 1:
        # compute next tuple
        tj = t[j]
        if j:
            t[j] = 0
            t[0] = tj
        if tj > 1:
            t[j + 1] += 1
            j = 0
            start = 1
            v = 0
        else:
            j += 1
            start = j + 1
            v = r[tuple(t)]
            t[j] += 1
        # compute the value
        # NB: the initialization of v was done above
        for k in range(start, m):
            if t[k]:
                t[k] -= 1
                v += r[tuple(t)]
                t[k] += 1
        t[0] -= 1
        r[tuple(t)] = (v * tj) // (n - t[0])
    return r


def multinomial_coefficients_iterator(m, n, _tuple=tuple):
    """multinomial coefficient iterator

    This routine has been optimized for `m` large with respect to `n` by taking
    advantage of the fact that when the monomial tuples `t` are stripped of
    zeros, their coefficient is the same as that of the monomial tuples from
    ``multinomial_coefficients(n, n)``. Therefore, the latter coefficients are
    precomputed to save memory and time.

    >>> from sympy.ntheory.multinomial import multinomial_coefficients
    >>> m53, m33 = multinomial_coefficients(5,3), multinomial_coefficients(3,3)
    >>> m53[(0,0,0,1,2)] == m53[(0,0,1,0,2)] == m53[(1,0,2,0,0)] == m33[(0,1,2)]
    True

    Examples
    ========

    >>> from sympy.ntheory.multinomial import multinomial_coefficients_iterator
    >>> it = multinomial_coefficients_iterator(20,3)
    >>> next(it)
    ((3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 1)
    """
    m = as_int(m)
    n = as_int(n)
    if m < 2*n or n == 1:
        mc = multinomial_coefficients(m, n)
        yield from mc.items()
    else:
        mc = multinomial_coefficients(n, n)
        mc1 = {}
        for k, v in mc.items():
            mc1[_tuple(filter(None, k))] = v
        mc = mc1

        t = [n] + [0] * (m - 1)
        t1 = _tuple(t)
        b = _tuple(filter(None, t1))
        yield (t1, mc[b])
        if n:
            j = 0  # j will be the leftmost nonzero position
        else:
            j = m
        # enumerate tuples in co-lex order
        while j < m - 1:
            # compute next tuple
            tj = t[j]
            if j:
                t[j] = 0
                t[0] = tj
            if tj > 1:
                t[j + 1] += 1
                j = 0
            else:
                j += 1
                t[j] += 1

            t[0] -= 1
            t1 = _tuple(t)
            b = _tuple(filter(None, t1))
            yield (t1, mc[b])
<|endoftext|>from collections import defaultdict

from sympy.utilities.iterables import multiset, is_palindromic as _palindromic
from sympy.utilities.misc import as_int


def digits(n, b=10, digits=None):
    """
    Return a list of the digits of ``n`` in base ``b``. The first
    element in the list is ``b`` (or ``-b`` if ``n`` is negative).

    Examples
    ========

    >>> from sympy.ntheory.digits import digits
    >>> digits(35)
    [10, 3, 5]

    If the number is negative, the negative sign will be placed on the
    base (which is the first element in the returned list):

    >>> digits(-35)
    [-10, 3, 5]

    Bases other than 10 (and greater than 1) can be selected with ``b``:

    >>> digits(27, b=2)
    [2, 1, 1, 0, 1, 1]

    Use the ``digits`` keyword if a certain number of digits is desired:

    >>> digits(35, digits=4)
    [10, 0, 0, 3, 5]

    Parameters
    ==========

    n: integer
        The number whose digits are returned.

    b: integer
        The base in which digits are computed.

    digits: integer (or None for all digits)
        The number of digits to be returned (padded with zeros, if
        necessary).

    See Also
    ========
    sympy.core.intfunc.num_digits, count_digits
    """

    b = as_int(b)
    n = as_int(n)
    if b < 2:
        raise ValueError("b must be greater than 1")
    else:
        x, y = abs(n), []
        while x >= b:
            x, r = divmod(x, b)
            y.append(r)
        y.append(x)
        y.append(-b if n < 0 else b)
        y.reverse()
        ndig = len(y) - 1
        if digits is not None:
            if ndig > digits:
                raise ValueError(
                    "For %s, at least %s digits are needed." % (n, ndig))
            elif ndig < digits:
                y[1:1] = [0]*(digits - ndig)
        return y


def count_digits(n, b=10):
    """
    Return a dictionary whose keys are the digits of ``n`` in the
    given base, ``b``, with keys indicating the digits appearing in the
    number and values indicating how many times that digit appeared.

    Examples
    ========

    >>> from sympy.ntheory import count_digits

    >>> count_digits(1111339)
    {1: 4, 3: 2, 9: 1}

    The digits returned are always represented in base-10
    but the number itself can be entered in any format that is
    understood by Python; the base of the number can also be
    given if it is different than 10:

    >>> n = 0xFA; n
    250
    >>> count_digits(_)
    {0: 1, 2: 1, 5: 1}
    >>> count_digits(n, 16)
    {10: 1, 15: 1}

    The default dictionary will return a 0 for any digit that did
    not appear in the number. For example, which digits appear 7
    times in ``77!``:

    >>> from sympy import factorial
    >>> c77 = count_digits(factorial(77))
    >>> [i for i in range(10) if c77[i] == 7]
    [1, 3, 7, 9]

    See Also
    ========
    sympy.core.intfunc.num_digits, digits
    """
    rv = defaultdict(int, multiset(digits(n, b)).items())
    rv.pop(b) if b in rv else rv.pop(-b)  # b or -b is there
    return rv


def is_palindromic(n, b=10):
    """return True if ``n`` is the same when read from left to right
    or right to left in the given base, ``b``.

    Examples
    ========

    >>> from sympy.ntheory import is_palindromic

    >>> all(is_palindromic(i) for i in (-11, 1, 22, 121))
    True

    The second argument allows you to test numbers in other
    bases. For example, 88 is palindromic in base-10 but not
    in base-8:

    >>> is_palindromic(88, 8)
    False

    On the other hand, a number can be palindromic in base-8 but
    not in base-10:

    >>> 0o121, is_palindromic(0o121)
    (81, False)

    Or it might be palindromic in both bases:

    >>> oct(121), is_palindromic(121, 8) and is_palindromic(121)
    ('0o171', True)

    """
    return _palindromic(digits(n, b), 1)
<|endoftext|>'''
This implementation is a heavily modified fixed point implementation of
BBP_formula for calculating the nth position of pi. The original hosted
at: https://web.archive.org/web/20151116045029/http://en.literateprograms.org/Pi_with_the_BBP_formula_(Python)

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sub-license, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Modifications:

1.Once the nth digit and desired number of digits is selected, the
number of digits of working precision is calculated to ensure that
the hexadecimal digits returned are accurate. This is calculated as

    int(math.log(start + prec)/math.log(16) + prec + 3)
    ---------------------------------------   --------
                      /                          /
    number of hex digits           additional digits

This was checked by the following code which completed without
errors (and dig are the digits included in the test_bbp.py file):

    for i in range(0,1000):
     for j in range(1,1000):
      a, b = pi_hex_digits(i, j), dig[i:i+j]
      if a != b:
        print('%s\n%s'%(a,b))

Deceasing the additional digits by 1 generated errors, so '3' is
the smallest additional precision needed to calculate the above
loop without errors. The following trailing 10 digits were also
checked to be accurate (and the times were slightly faster with
some of the constant modifications that were made):

    >> from time import time
    >> t=time();pi_hex_digits(10**2-10 + 1, 10), time()-t
    ('e90c6cc0ac', 0.0)
    >> t=time();pi_hex_digits(10**4-10 + 1, 10), time()-t
    ('26aab49ec6', 0.17100000381469727)
    >> t=time();pi_hex_digits(10**5-10 + 1, 10), time()-t
    ('a22673c1a5', 4.7109999656677246)
    >> t=time();pi_hex_digits(10**6-10 + 1, 10), time()-t
    ('9ffd342362', 59.985999822616577)
    >> t=time();pi_hex_digits(10**7-10 + 1, 10), time()-t
    ('c1a42e06a1', 689.51800012588501)

2. The while loop to evaluate whether the series has converged quits
when the addition amount `dt` has dropped to zero.

3. the formatting string to convert the decimal to hexadecimal is
calculated for the given precision.

4. pi_hex_digits(n) changed to have coefficient to the formula in an
array (perhaps just a matter of preference).

'''

from sympy.utilities.misc import as_int


def _series(j, n, prec=14):

    # Left sum from the bbp algorithm
    s = 0
    D = _dn(n, prec)
    D4 = 4 * D
    d = j
    for k in range(n + 1):
        s += (pow(16, n - k, d) << D4) // d
        d += 8

    # Right sum iterates to infinity for full precision, but we
    # stop at the point where one iteration is beyond the precision
    # specified.

    t = 0
    k = n + 1
    e = D4 - 4 # 4*(D + n - k)
    d = 8 * k + j
    while True:
        dt = (1 << e) // d
        if not dt:
            break
        t += dt
        # k += 1
        e -= 4
        d += 8
    total = s + t

    return total


def pi_hex_digits(n, prec=14):
    """Returns a string containing ``prec`` (default 14) digits
    starting at the nth digit of pi in hex. Counting of digits
    starts at 0 and the decimal is not counted, so for n = 0 the
    returned value starts with 3; n = 1 corresponds to the first
    digit past the decimal point (which in hex is 2).

    Parameters
    ==========

    n : non-negative integer
    prec : non-negative integer. default = 14

    Returns
    =======

    str : Returns a string containing ``prec`` digits
          starting at the nth digit of pi in hex.
          If ``prec`` = 0, returns empty string.

    Raises
    ======

    ValueError
        If ``n`` < 0 or ``prec`` < 0.
        Or ``n`` or ``prec`` is not an integer.

    Examples
    ========

    >>> from sympy.ntheory.bbp_pi import pi_hex_digits
    >>> pi_hex_digits(0)
    '3243f6a8885a30'
    >>> pi_hex_digits(0, 3)
    '324'

    These are consistent with the following results

    >>> import math
    >>> hex(int(math.pi * 2**((14-1)*4)))
    '0x3243f6a8885a30'
    >>> hex(int(math.pi * 2**((3-1)*4)))
    '0x324'

    References
    ==========

    .. [1] http://www.numberworld.org/digits/Pi/
    """
    n, prec = as_int(n), as_int(prec)
    if n < 0:
        raise ValueError('n cannot be negative')
    if prec < 0:
        raise ValueError('prec cannot be negative')
    if prec == 0:
        return ''

    # main of implementation arrays holding formulae coefficients
    n -= 1
    a = [4, 2, 1, 1]
    j = [1, 4, 5, 6]

    #formulae
    D = _dn(n, prec)
    x = + (a[0]*_series(j[0], n, prec)
         - a[1]*_series(j[1], n, prec)
         - a[2]*_series(j[2], n, prec)
         - a[3]*_series(j[3], n, prec)) & (16**D - 1)

    s = ("%0" + "%ix" % prec) % (x // 16**(D - prec))
    return s


def _dn(n, prec):
    # controller for n dependence on precision
    # n = starting digit index
    # prec = the number of total digits to compute
    n += 1  # because we subtract 1 for _series

    # assert int(math.log(n + prec)/math.log(16)) ==\
    #  ((n + prec).bit_length() - 1) // 4
    return ((n + prec).bit_length() - 1) // 4 + prec + 3
<|endoftext|>from math import prod

from sympy.external.gmpy import gcd, gcdext
from sympy.ntheory.primetest import isprime
from sympy.polys.domains import ZZ
from sympy.polys.galoistools import gf_crt, gf_crt1, gf_crt2
from sympy.utilities.misc import as_int


def symmetric_residue(a, m):
    """Return the residual mod m such that it is within half of the modulus.

    >>> from sympy.ntheory.modular import symmetric_residue
    >>> symmetric_residue(1, 6)
    1
    >>> symmetric_residue(4, 6)
    -2
    """
    if a <= m // 2:
        return a
    return a - m


def crt(m, v, symmetric=False, check=True):
    r"""Chinese Remainder Theorem.

    The moduli in m are assumed to be pairwise coprime.  The output
    is then an integer f, such that f = v_i mod m_i for each pair out
    of v and m. If ``symmetric`` is False a positive integer will be
    returned, else \|f\| will be less than or equal to the LCM of the
    moduli, and thus f may be negative.

    If the moduli are not co-prime the correct result will be returned
    if/when the test of the result is found to be incorrect. This result
    will be None if there is no solution.

    The keyword ``check`` can be set to False if it is known that the moduli
    are coprime.

    Examples
    ========

    As an example consider a set of residues ``U = [49, 76, 65]``
    and a set of moduli ``M = [99, 97, 95]``. Then we have::

       >>> from sympy.ntheory.modular import crt

       >>> crt([99, 97, 95], [49, 76, 65])
       (639985, 912285)

    This is the correct result because::

       >>> [639985 % m for m in [99, 97, 95]]
       [49, 76, 65]

    If the moduli are not co-prime, you may receive an incorrect result
    if you use ``check=False``:

       >>> crt([12, 6, 17], [3, 4, 2], check=False)
       (954, 1224)
       >>> [954 % m for m in [12, 6, 17]]
       [6, 0, 2]
       >>> crt([12, 6, 17], [3, 4, 2]) is None
       True
       >>> crt([3, 6], [2, 5])
       (5, 6)

    Note: the order of gf_crt's arguments is reversed relative to crt,
    and that solve_congruence takes residue, modulus pairs.

    Programmer's note: rather than checking that all pairs of moduli share
    no GCD (an O(n**2) test) and rather than factoring all moduli and seeing
    that there is no factor in common, a check that the result gives the
    indicated residuals is performed -- an O(n) operation.

    See Also
    ========

    solve_congruence
    sympy.polys.galoistools.gf_crt : low level crt routine used by this routine
    """
    if check:
        m = list(map(as_int, m))
        v = list(map(as_int, v))

    result = gf_crt(v, m, ZZ)
    mm = prod(m)

    if check:
        if not all(v % m == result % m for v, m in zip(v, m)):
            result = solve_congruence(*list(zip(v, m)),
                    check=False, symmetric=symmetric)
            if result is None:
                return result
            result, mm = result

    if symmetric:
        return int(symmetric_residue(result, mm)), int(mm)
    return int(result), int(mm)


def crt1(m):
    """First part of Chinese Remainder Theorem, for multiple application.

    Examples
    ========

    >>> from sympy.ntheory.modular import crt, crt1, crt2
    >>> m = [99, 97, 95]
    >>> v = [49, 76, 65]

    The following two codes have the same result.

    >>> crt(m, v)
    (639985, 912285)

    >>> mm, e, s = crt1(m)
    >>> crt2(m, v, mm, e, s)
    (639985, 912285)

    However, it is faster when we want to fix ``m`` and
    compute for multiple ``v``, i.e. the following cases:

    >>> mm, e, s = crt1(m)
    >>> vs = [[52, 21, 37], [19, 46, 76]]
    >>> for v in vs:
    ...     print(crt2(m, v, mm, e, s))
    (397042, 912285)
    (803206, 912285)

    See Also
    ========

    sympy.polys.galoistools.gf_crt1 : low level crt routine used by this routine
    sympy.ntheory.modular.crt
    sympy.ntheory.modular.crt2

    """

    return gf_crt1(m, ZZ)


def crt2(m, v, mm, e, s, symmetric=False):
    """Second part of Chinese Remainder Theorem, for multiple application.

    See ``crt1`` for usage.

    Examples
    ========

    >>> from sympy.ntheory.modular import crt1, crt2
    >>> mm, e, s = crt1([18, 42, 6])
    >>> crt2([18, 42, 6], [0, 0, 0], mm, e, s)
    (0, 4536)

    See Also
    ========

    sympy.polys.galoistools.gf_crt2 : low level crt routine used by this routine
    sympy.ntheory.modular.crt
    sympy.ntheory.modular.crt1

    """

    result = gf_crt2(v, m, mm, e, s, ZZ)

    if symmetric:
        return int(symmetric_residue(result, mm)), int(mm)
    return int(result), int(mm)


def solve_congruence(*remainder_modulus_pairs, **hint):
    """Compute the integer ``n`` that has the residual ``ai`` when it is
    divided by ``mi`` where the ``ai`` and ``mi`` are given as pairs to
    this function: ((a1, m1), (a2, m2), ...). If there is no solution,
    return None. Otherwise return ``n`` and its modulus.

    The ``mi`` values need not be co-prime. If it is known that the moduli are
    not co-prime then the hint ``check`` can be set to False (default=True) and
    the check for a quicker solution via crt() (valid when the moduli are
    co-prime) will be skipped.

    If the hint ``symmetric`` is True (default is False), the value of ``n``
    will be within 1/2 of the modulus, possibly negative.

    Examples
    ========

    >>> from sympy.ntheory.modular import solve_congruence

    What number is 2 mod 3, 3 mod 5 and 2 mod 7?

    >>> solve_congruence((2, 3), (3, 5), (2, 7))
    (23, 105)
    >>> [23 % m for m in [3, 5, 7]]
    [2, 3, 2]

    If you prefer to work with all remainder in one list and
    all moduli in another, send the arguments like this:

    >>> solve_congruence(*zip((2, 3, 2), (3, 5, 7)))
    (23, 105)

    The moduli need not be co-prime; in this case there may or
    may not be a solution:

    >>> solve_congruence((2, 3), (4, 6)) is None
    True

    >>> solve_congruence((2, 3), (5, 6))
    (5, 6)

    The symmetric flag will make the result be within 1/2 of the modulus:

    >>> solve_congruence((2, 3), (5, 6), symmetric=True)
    (-1, 6)

    See Also
    ========

    crt : high level routine implementing the Chinese Remainder Theorem

    """
    def combine(c1, c2):
        """Return the tuple (a, m) which satisfies the requirement
        that n = a + i*m satisfy n = a1 + j*m1 and n = a2 = k*m2.

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Method_of_successive_substitution
        """
        a1, m1 = c1
        a2, m2 = c2
        a, b, c = m1, a2 - a1, m2
        g = gcd(a, b, c)
        a, b, c = [i//g for i in [a, b, c]]
        if a != 1:
            g, inv_a, _ = gcdext(a, c)
            if g != 1:
                return None
            b *= inv_a
        a, m = a1 + m1*b, m1*c
        return a, m

    rm = remainder_modulus_pairs
    symmetric = hint.get('symmetric', False)

    if hint.get('check', True):
        rm = [(as_int(r), as_int(m)) for r, m in rm]

        # ignore redundant pairs but raise an error otherwise; also
        # make sure that a unique set of bases is sent to gf_crt if
        # they are all prime.
        #
        # The routine will work out less-trivial violations and
        # return None, e.g. for the pairs (1,3) and (14,42) there
        # is no answer because 14 mod 42 (having a gcd of 14) implies
        # (14/2) mod (42/2), (14/7) mod (42/7) and (14/14) mod (42/14)
        # which, being 0 mod 3, is inconsistent with 1 mod 3. But to
        # preprocess the input beyond checking of another pair with 42
        # or 3 as the modulus (for this example) is not necessary.
        uniq = {}
        for r, m in rm:
            r %= m
            if m in uniq:
                if r != uniq[m]:
                    return None
                continue
            uniq[m] = r
        rm = [(r, m) for m, r in uniq.items()]
        del uniq

        # if the moduli are co-prime, the crt will be significantly faster;
        # checking all pairs for being co-prime gets to be slow but a prime
        # test is a good trade-off
        if all(isprime(m) for r, m in rm):
            r, m = list(zip(*rm))
            return crt(m, r, symmetric=symmetric, check=False)

    rv = (0, 1)
    for rmi in rm:
        rv = combine(rv, rmi)
        if rv is None:
            break
        n, m = rv
        n = n % m
    else:
        if symmetric:
            return symmetric_residue(n, m), m
        return n, m
<|endoftext|>"""
Generating and counting primes.

"""

from bisect import bisect, bisect_left
from itertools import count
# Using arrays for sieving instead of lists greatly reduces
# memory consumption
from array import array as _array

from sympy.core.random import randint
from sympy.external.gmpy import sqrt
from .primetest import isprime
from sympy.utilities.decorator import deprecated
from sympy.utilities.misc import as_int


def _as_int_ceiling(a):
    """ Wrapping ceiling in as_int will raise an error if there was a problem
        determining whether the expression was exactly an integer or not."""
    from sympy.functions.elementary.integers import ceiling
    return as_int(ceiling(a))


class Sieve:
    """A list of prime numbers, implemented as a dynamically
    growing sieve of Eratosthenes. When a lookup is requested involving
    an odd number that has not been sieved, the sieve is automatically
    extended up to that number. Implementation details limit the number of
    primes to ``2^32-1``.

    Examples
    ========

    >>> from sympy import sieve
    >>> sieve._reset() # this line for doctest only
    >>> 25 in sieve
    False
    >>> sieve._list
    array('L', [2, 3, 5, 7, 11, 13, 17, 19, 23])
    """

    # data shared (and updated) by all Sieve instances
    def __init__(self, sieve_interval=1_000_000):
        """ Initial parameters for the Sieve class.

        Parameters
        ==========

        sieve_interval (int): Amount of memory to be used

        Raises
        ======

        ValueError
            If ``sieve_interval`` is not positive.

        """
        self._n = 6
        self._list = _array('L', [2, 3, 5, 7, 11, 13]) # primes
        self._tlist = _array('L', [0, 1, 1, 2, 2, 4]) # totient
        self._mlist = _array('i', [0, 1, -1, -1, 0, -1]) # mobius
        if sieve_interval <= 0:
            raise ValueError("sieve_interval should be a positive integer")
        self.sieve_interval = sieve_interval
        assert all(len(i) == self._n for i in (self._list, self._tlist, self._mlist))

    def __repr__(self):
        return ("<%s sieve (%i): %i, %i, %i, ... %i, %i\n"
             "%s sieve (%i): %i, %i, %i, ... %i, %i\n"
             "%s sieve (%i): %i, %i, %i, ... %i, %i>") % (
             'prime', len(self._list),
                 self._list[0], self._list[1], self._list[2],
                 self._list[-2], self._list[-1],
             'totient', len(self._tlist),
                 self._tlist[0], self._tlist[1],
                 self._tlist[2], self._tlist[-2], self._tlist[-1],
             'mobius', len(self._mlist),
                 self._mlist[0], self._mlist[1],
                 self._mlist[2], self._mlist[-2], self._mlist[-1])

    def _reset(self, prime=None, totient=None, mobius=None):
        """Reset all caches (default). To reset one or more set the
            desired keyword to True."""
        if all(i is None for i in (prime, totient, mobius)):
            prime = totient = mobius = True
        if prime:
            self._list = self._list[:self._n]
        if totient:
            self._tlist = self._tlist[:self._n]
        if mobius:
            self._mlist = self._mlist[:self._n]

    def extend(self, n):
        """Grow the sieve to cover all primes <= n.

        Examples
        ========

        >>> from sympy import sieve
        >>> sieve._reset() # this line for doctest only
        >>> sieve.extend(30)
        >>> sieve[10] == 29
        True
        """
        n = int(n)
        # `num` is even at any point in the function.
        # This satisfies the condition required by `self._primerange`.
        num = self._list[-1] + 1
        if n < num:
            return
        num2 = num**2
        while num2 <= n:
            self._list += _array('L', self._primerange(num, num2))
            num, num2 = num2, num2**2
        # Merge the sieves
        self._list += _array('L', self._primerange(num, n + 1))

    def _primerange(self, a, b):
        """ Generate all prime numbers in the range (a, b).

        Parameters
        ==========

        a, b : positive integers assuming the following conditions
                * a is an even number
                * 2 < self._list[-1] < a < b < nextprime(self._list[-1])**2

        Yields
        ======

        p (int): prime numbers such that ``a < p < b``

        Examples
        ========

        >>> from sympy.ntheory.generate import Sieve
        >>> s = Sieve()
        >>> s._list[-1]
        13
        >>> list(s._primerange(18, 31))
        [19, 23, 29]

        """
        if b % 2:
            b -= 1
        while a < b:
            block_size = min(self.sieve_interval, (b - a) // 2)
            # Create the list such that block[x] iff (a + 2x + 1) is prime.
            # Note that even numbers are not considered here.
            block = [True] * block_size
            for p in self._list[1:bisect(self._list, sqrt(a + 2 * block_size + 1))]:
                for t in range((-(a + 1 + p) // 2) % p, block_size, p):
                    block[t] = False
            for idx, p in enumerate(block):
                if p:
                    yield a + 2 * idx + 1
            a += 2 * block_size

    def extend_to_no(self, i):
        """Extend to include the ith prime number.

        Parameters
        ==========

        i : integer

        Examples
        ========

        >>> from sympy import sieve
        >>> sieve._reset() # this line for doctest only
        >>> sieve.extend_to_no(9)
        >>> sieve._list
        array('L', [2, 3, 5, 7, 11, 13, 17, 19, 23])

        Notes
        =====

        The list is extended by 50% if it is too short, so it is
        likely that it will be longer than requested.
        """
        i = as_int(i)
        while len(self._list) < i:
            self.extend(int(self._list[-1] * 1.5))

    def primerange(self, a, b=None):
        """Generate all prime numbers in the range [2, a) or [a, b).

        Examples
        ========

        >>> from sympy import sieve, prime

        All primes less than 19:

        >>> print([i for i in sieve.primerange(19)])
        [2, 3, 5, 7, 11, 13, 17]

        All primes greater than or equal to 7 and less than 19:

        >>> print([i for i in sieve.primerange(7, 19)])
        [7, 11, 13, 17]

        All primes through the 10th prime

        >>> list(sieve.primerange(prime(10) + 1))
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        """
        if b is None:
            b = _as_int_ceiling(a)
            a = 2
        else:
            a = max(2, _as_int_ceiling(a))
            b = _as_int_ceiling(b)
        if a >= b:
            return
        self.extend(b)
        yield from self._list[bisect_left(self._list, a):
                              bisect_left(self._list, b)]

    def totientrange(self, a, b):
        """Generate all totient numbers for the range [a, b).

        Examples
        ========

        >>> from sympy import sieve
        >>> print([i for i in sieve.totientrange(7, 18)])
        [6, 4, 6, 4, 10, 4, 12, 6, 8, 8, 16]
        """
        a = max(1, _as_int_ceiling(a))
        b = _as_int_ceiling(b)
        n = len(self._tlist)
        if a >= b:
            return
        elif b <= n:
            for i in range(a, b):
                yield self._tlist[i]
        else:
            self._tlist += _array('L', range(n, b))
            for i in range(1, n):
                ti = self._tlist[i]
                if ti == i - 1:
                    startindex = (n + i - 1) // i * i
                    for j in range(startindex, b, i):
                        self._tlist[j] -= self._tlist[j] // i
                if i >= a:
                    yield ti

            for i in range(n, b):
                ti = self._tlist[i]
                if ti == i:
                    for j in range(i, b, i):
                        self._tlist[j] -= self._tlist[j] // i
                if i >= a:
                    yield self._tlist[i]

    def mobiusrange(self, a, b):
        """Generate all mobius numbers for the range [a, b).

        Parameters
        ==========

        a : integer
            First number in range

        b : integer
            First number outside of range

        Examples
        ========

        >>> from sympy import sieve
        >>> print([i for i in sieve.mobiusrange(7, 18)])
        [-1, 0, 0, 1, -1, 0, -1, 1, 1, 0, -1]
        """
        a = max(1, _as_int_ceiling(a))
        b = _as_int_ceiling(b)
        n = len(self._mlist)
        if a >= b:
            return
        elif b <= n:
            for i in range(a, b):
                yield self._mlist[i]
        else:
            self._mlist += _array('i', [0]*(b - n))
            for i in range(1, n):
                mi = self._mlist[i]
                startindex = (n + i - 1) // i * i
                for j in range(startindex, b, i):
                    self._mlist[j] -= mi
                if i >= a:
                    yield mi

            for i in range(n, b):
                mi = self._mlist[i]
                for j in range(2 * i, b, i):
                    self._mlist[j] -= mi
                if i >= a:
                    yield mi

    def search(self, n):
        """Return the indices i, j of the primes that bound n.

        If n is prime then i == j.

        Although n can be an expression, if ceiling cannot convert
        it to an integer then an n error will be raised.

        Examples
        ========

        >>> from sympy import sieve
        >>> sieve.search(25)
        (9, 10)
        >>> sieve.search(23)
        (9, 9)
        """
        test = _as_int_ceiling(n)
        n = as_int(n)
        if n < 2:
            raise ValueError("n should be >= 2 but got: %s" % n)
        if n > self._list[-1]:
            self.extend(n)
        b = bisect(self._list, n)
        if self._list[b - 1] == test:
            return b, b
        else:
            return b, b + 1

    def __contains__(self, n):
        try:
            n = as_int(n)
            assert n >= 2
        except (ValueError, AssertionError):
            return False
        if n % 2 == 0:
            return n == 2
        a, b = self.search(n)
        return a == b

    def __iter__(self):
        for n in count(1):
            yield self[n]

    def __getitem__(self, n):
        """Return the nth prime number"""
        if isinstance(n, slice):
            self.extend_to_no(n.stop)
            start = n.start if n.start is not None else 0
            if start < 1:
                # sieve[:5] would be empty (starting at -1), let's
                # just be explicit and raise.
                raise IndexError("Sieve indices start at 1.")
            return self._list[start - 1:n.stop - 1:n.step]
        else:
            if n < 1:
                # offset is one, so forbid explicit access to sieve[0]
                # (would surprisingly return the last one).
                raise IndexError("Sieve indices start at 1.")
            n = as_int(n)
            self.extend_to_no(n)
            return self._list[n - 1]

# Generate a global object for repeated use in trial division etc
sieve = Sieve()

def prime(nth):
    r"""
    Return the nth prime number, where primes are indexed starting from 1:
    prime(1) = 2, prime(2) = 3, etc.

    Parameters
    ==========

    nth : int
        The position of the prime number to return (must be a positive integer).

    Returns
    =======

    int
        The nth prime number.

    Examples
    ========

    >>> from sympy import prime
    >>> prime(10)
    29
    >>> prime(1)
    2
    >>> prime(100000)
    1299709

    See Also
    ========

    sympy.ntheory.primetest.isprime : Test if a number is prime.
    primerange : Generate all primes in a given range.
    primepi : Return the number of primes less than or equal to a given number.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Prime_number_theorem
    .. [2] https://en.wikipedia.org/wiki/Logarithmic_integral_function
    .. [3] https://en.wikipedia.org/wiki/Skewes%27_number
    """
    n = as_int(nth)
    if n < 1:
        raise ValueError("nth must be a positive integer; prime(1) == 2")

    # Check if n is within the sieve range
    if n <= len(sieve._list):
        return sieve[n]

    from sympy.functions.elementary.exponential import log
    from sympy.functions.special.error_functions import li

    if n < 1000:
        # Extend sieve up to 8*n as this is empirically sufficient
        sieve.extend(8 * n)
        return sieve[n]

    a = 2
    # Estimate an upper bound for the nth prime using the prime number theorem
    b = int(n * (log(n).evalf() + log(log(n)).evalf()))

    # Binary search for the least m such that li(m) > n
    while a < b:
        mid = (a + b) >> 1
        if li(mid).evalf() > n:
            b = mid
        else:
            a = mid + 1

    return nextprime(a - 1, n - _primepi(a - 1))


@deprecated("""\
The `sympy.ntheory.generate.primepi` has been moved to `sympy.functions.combinatorial.numbers.primepi`.""",
deprecated_since_version="1.13",
active_deprecations_target='deprecated-ntheory-symbolic-functions')
def primepi(n):
    r""" Represents the prime counting function pi(n) = the number
        of prime numbers less than or equal to n.

        .. deprecated:: 1.13

            The ``primepi`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.primepi`
            instead. See its documentation for more information. See
            :ref:`deprecated-ntheory-symbolic-functions` for details.

        Algorithm Description:

        In sieve method, we remove all multiples of prime p
        except p itself.

        Let phi(i,j) be the number of integers 2 <= k <= i
        which remain after sieving from primes less than
        or equal to j.
        Clearly, pi(n) = phi(n, sqrt(n))

        If j is not a prime,
        phi(i,j) = phi(i, j - 1)

        if j is a prime,
        We remove all numbers(except j) whose
        smallest prime factor is j.

        Let $x= j \times a$ be such a number, where $2 \le a \le i / j$
        Now, after sieving from primes $\le j - 1$,
        a must remain
        (because x, and hence a has no prime factor $\le j - 1$)
        Clearly, there are phi(i / j, j - 1) such a
        which remain on sieving from primes $\le j - 1$

        Now, if a is a prime less than equal to j - 1,
        $x= j \times a$ has smallest prime factor = a, and
        has already been removed(by sieving from a).
        So, we do not need to remove it again.
        (Note: there will be pi(j - 1) such x)

        Thus, number of x, that will be removed are:
        phi(i / j, j - 1) - phi(j - 1, j - 1)
        (Note that pi(j - 1) = phi(j - 1, j - 1))

        $\Rightarrow$ phi(i,j) = phi(i, j - 1) - phi(i / j, j - 1) + phi(j - 1, j - 1)

        So,following recursion is used and implemented as dp:

        phi(a, b) = phi(a, b - 1), if b is not a prime
        phi(a, b) = phi(a, b-1)-phi(a / b, b-1) + phi(b-1, b-1), if b is prime

        Clearly a is always of the form floor(n / k),
        which can take at most $2\sqrt{n}$ values.
        Two arrays arr1,arr2 are maintained
        arr1[i] = phi(i, j),
        arr2[i] = phi(n // i, j)

        Finally the answer is arr2[1]

        Examples
        ========

        >>> from sympy import primepi, prime, prevprime, isprime
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
        primerange : Generate all primes in a given range
        prime : Return the nth prime
    """
    from sympy.functions.combinatorial.numbers import primepi as func_primepi
    return func_primepi(n)


def _primepi(n:int) -> int:
    r""" Represents the prime counting function pi(n) = the number
    of prime numbers less than or equal to n.

    Explanation
    ===========

    In sieve method, we remove all multiples of prime p
    except p itself.

    Let phi(i,j) be the number of integers 2 <= k <= i
    which remain after sieving from primes less than
    or equal to j.
    Clearly, pi(n) = phi(n, sqrt(n))

    If j is not a prime,
    phi(i,j) = phi(i, j - 1)

    if j is a prime,
    We remove all numbers(except j) whose
    smallest prime factor is j.

    Let $x= j \times a$ be such a number, where $2 \le a \le i / j$
    Now, after sieving from primes $\le j - 1$,
    a must remain
    (because x, and hence a has no prime factor $\le j - 1$)
    Clearly, there are phi(i / j, j - 1) such a
    which remain on sieving from primes $\le j - 1$

    Now, if a is a prime less than equal to j - 1,
    $x= j \times a$ has smallest prime factor = a, and
    has already been removed(by sieving from a).
    So, we do not need to remove it again.
    (Note: there will be pi(j - 1) such x)

    Thus, number of x, that will be removed are:
    phi(i / j, j - 1) - phi(j - 1, j - 1)
    (Note that pi(j - 1) = phi(j - 1, j - 1))

    $\Rightarrow$ phi(i,j) = phi(i, j - 1) - phi(i / j, j - 1) + phi(j - 1, j - 1)

    So,following recursion is used and implemented as dp:

    phi(a, b) = phi(a, b - 1), if b is not a prime
    phi(a, b) = phi(a, b-1)-phi(a / b, b-1) + phi(b-1, b-1), if b is prime

    Clearly a is always of the form floor(n / k),
    which can take at most $2\sqrt{n}$ values.
    Two arrays arr1,arr2 are maintained
    arr1[i] = phi(i, j),
    arr2[i] = phi(n // i, j)

    Finally the answer is arr2[1]

    Parameters
    ==========

    n : int

    """
    if n < 2:
        return 0
    if n <= sieve._list[-1]:
        return sieve.search(n)[0]
    lim = sqrt(n)
    arr1 = [(i + 1) >> 1 for i in range(lim + 1)]
    arr2 = [0] + [(n//i + 1) >> 1 for i in range(1, lim + 1)]
    skip = [False] * (lim + 1)
    for i in range(3, lim + 1, 2):
        # Presently, arr1[k]=phi(k,i - 1),
        # arr2[k] = phi(n // k,i - 1) # not all k's do this
        if skip[i]:
            # skip if i is a composite number
            continue
        p = arr1[i - 1]
        for j in range(i, lim + 1, i):
            skip[j] = True
        # update arr2
        # phi(n/j, i) = phi(n/j, i-1) - phi(n/(i*j), i-1) + phi(i-1, i-1)
        for j in range(1, min(n // (i * i), lim) + 1, 2):
            # No need for arr2[j] in j such that skip[j] is True to
            # compute the final required arr2[1].
            if skip[j]:
                continue
            st = i * j
            if st <= lim:
                arr2[j] -= arr2[st] - p
            else:
                arr2[j] -= arr1[n // st] - p
        # update arr1
        # phi(j, i) = phi(j, i-1) - phi(j/i, i-1) + phi(i-1, i-1)
        # where the range below i**2 is fixed and
        # does not need to be calculated.
        for j in range(lim, min(lim, i*i - 1), -1):
            arr1[j] -= arr1[j // i] - p
    return arr2[1]


def nextprime(n, ith=1):
    """ Return the ith prime greater than n.

        Parameters
        ==========

        n : integer
        ith : positive integer

        Returns
        =======

        int : Return the ith prime greater than n

        Raises
        ======

        ValueError
            If ``ith <= 0``.
            If ``n`` or ``ith`` is not an integer.

        Notes
        =====

        Potential primes are located at 6*j +/- 1. This
        property is used during searching.

        >>> from sympy import nextprime
        >>> [(i, nextprime(i)) for i in range(10, 15)]
        [(10, 11), (11, 13), (12, 13), (13, 17), (14, 17)]
        >>> nextprime(2, ith=2) # the 2nd prime after 2
        5

        See Also
        ========

        prevprime : Return the largest prime smaller than n
        primerange : Generate all primes in a given range

    """
    n = int(n)
    i = as_int(ith)
    if i <= 0:
        raise ValueError("ith should be positive")
    if n < 2:
        n = 2
        i -= 1
    if n <= sieve._list[-2]:
        l, _ = sieve.search(n)
        if l + i - 1 < len(sieve._list):
            return sieve._list[l + i - 1]
        n = sieve._list[-1]
        i += l - len(sieve._list)
    nn = 6*(n//6)
    if nn == n:
        n += 1
        if isprime(n):
            i -= 1
            if not i:
                return n
        n += 4
    elif n - nn == 5:
        n += 2
        if isprime(n):
            i -= 1
            if not i:
                return n
        n += 4
    else:
        n = nn + 5
    while 1:
        if isprime(n):
            i -= 1
            if not i:
                return n
        n += 2
        if isprime(n):
            i -= 1
            if not i:
                return n
        n += 4


def prevprime(n):
    """ Return the largest prime smaller than n.

        Notes
        =====

        Potential primes are located at 6*j +/- 1. This
        property is used during searching.

        >>> from sympy import prevprime
        >>> [(i, prevprime(i)) for i in range(10, 15)]
        [(10, 7), (11, 7), (12, 11), (13, 11), (14, 13)]

        See Also
        ========

        nextprime : Return the ith prime greater than n
        primerange : Generates all primes in a given range
    """
    n = _as_int_ceiling(n)
    if n < 3:
        raise ValueError("no preceding primes")
    if n < 8:
        return {3: 2, 4: 3, 5: 3, 6: 5, 7: 5}[n]
    if n <= sieve._list[-1]:
        l, u = sieve.search(n)
        if l == u:
            return sieve[l-1]
        else:
            return sieve[l]
    nn = 6*(n//6)
    if n - nn <= 1:
        n = nn - 1
        if isprime(n):
            return n
        n -= 4
    else:
        n = nn + 1
    while 1:
        if isprime(n):
            return n
        n -= 2
        if isprime(n):
            return n
        n -= 4


def primerange(a, b=None):
    """ Generate a list of all prime numbers in the range [2, a),
        or [a, b).

        If the range exists in the default sieve, the values will
        be returned from there; otherwise values will be returned
        but will not modify the sieve.

        Examples
        ========

        >>> from sympy import primerange, prime

        All primes less than 19:

        >>> list(primerange(19))
        [2, 3, 5, 7, 11, 13, 17]

        All primes greater than or equal to 7 and less than 19:

        >>> list(primerange(7, 19))
        [7, 11, 13, 17]

        All primes through the 10th prime

        >>> list(primerange(prime(10) + 1))
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        The Sieve method, primerange, is generally faster but it will
        occupy more memory as the sieve stores values. The default
        instance of Sieve, named sieve, can be used:

        >>> from sympy import sieve
        >>> list(sieve.primerange(1, 30))
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        Notes
        =====

        Some famous conjectures about the occurrence of primes in a given
        range are [1]:

        - Twin primes: though often not, the following will give 2 primes
                    an infinite number of times:
                        primerange(6*n - 1, 6*n + 2)
        - Legendre's: the following always yields at least one prime
                        primerange(n**2, (n+1)**2+1)
        - Bertrand's (proven): there is always a prime in the range
                        primerange(n, 2*n)
        - Brocard's: there are at least four primes in the range
                        primerange(prime(n)**2, prime(n+1)**2)

        The average gap between primes is log(n) [2]; the gap between
        primes can be arbitrarily large since sequences of composite
        numbers are arbitrarily large, e.g. the numbers in the sequence
        n! + 2, n! + 3 ... n! + n are all composite.

        See Also
        ========

        prime : Return the nth prime
        nextprime : Return the ith prime greater than n
        prevprime : Return the largest prime smaller than n
        randprime : Returns a random prime in a given range
        primorial : Returns the product of primes based on condition
        Sieve.primerange : return range from already computed primes
                           or extend the sieve to contain the requested
                           range.

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Prime_number
        .. [2] https://primes.utm.edu/notes/gaps.html
    """
    if b is None:
        a, b = 2, a
    if a >= b:
        return
    # If we already have the range, return it.
    largest_known_prime = sieve._list[-1]
    if b <= largest_known_prime:
        yield from sieve.primerange(a, b)
        return
    # If we know some of it, return it.
    if a <= largest_known_prime:
        yield from sieve._list[bisect_left(sieve._list, a):]
        a = largest_known_prime + 1
    elif a % 2:
        a -= 1
    tail = min(b, (largest_known_prime)**2)
    if a < tail:
        yield from sieve._primerange(a, tail)
        a = tail
    if b <= a:
        return
    # otherwise compute, without storing, the desired range.
    while 1:
        a = nextprime(a)
        if a < b:
            yield a
        else:
            return


def randprime(a, b):
    """ Return a random prime number in the range [a, b).

        Bertrand's postulate assures that
        randprime(a, 2*a) will always succeed for a > 1.

        Note that due to implementation difficulties,
        the prime numbers chosen are not uniformly random.
        For example, there are two primes in the range [112, 128),
        ``113`` and ``127``, but ``randprime(112, 128)`` returns ``127``
        with a probability of 15/17.

        Examples
        ========

        >>> from sympy import randprime, isprime
        >>> randprime(1, 30) #doctest: +SKIP
        13
        >>> isprime(randprime(1, 30))
        True

        See Also
        ========

        primerange : Generate all primes in a given range

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Bertrand's_postulate

    """
    if a >= b:
        return
    a, b = map(int, (a, b))
    n = randint(a - 1, b)
    p = nextprime(n)
    if p >= b:
        p = prevprime(b)
    if p < a:
        raise ValueError("no primes exist in the specified range")
    return p


def primorial(n, nth=True):
    """
    Returns the product of the first n primes (default) or
    the primes less than or equal to n (when ``nth=False``).

    Examples
    ========

    >>> from sympy.ntheory.generate import primorial, primerange
    >>> from sympy import factorint, Mul, primefactors, sqrt
    >>> primorial(4) # the first 4 primes are 2, 3, 5, 7
    210
    >>> primorial(4, nth=False) # primes <= 4 are 2 and 3
    6
    >>> primorial(1)
    2
    >>> primorial(1, nth=False)
    1
    >>> primorial(sqrt(101), nth=False)
    210

    One can argue that the primes are infinite since if you take
    a set of primes and multiply them together (e.g. the primorial) and
    then add or subtract 1, the result cannot be divided by any of the
    original factors, hence either 1 or more new primes must divide this
    product of primes.

    In this case, the number itself is a new prime:

    >>> factorint(primorial(4) + 1)
    {211: 1}

    In this case two new primes are the factors:

    >>> factorint(primorial(4) - 1)
    {11: 1, 19: 1}

    Here, some primes smaller and larger than the primes multiplied together
    are obtained:

    >>> p = list(primerange(10, 20))
    >>> sorted(set(primefactors(Mul(*p) + 1)).difference(set(p)))
    [2, 5, 31, 149]

    See Also
    ========

    primerange : Generate all primes in a given range

    """
    if nth:
        n = as_int(n)
    else:
        n = int(n)
    if n < 1:
        raise ValueError("primorial argument must be >= 1")
    p = 1
    if nth:
        for i in range(1, n + 1):
            p *= prime(i)
    else:
        for i in primerange(2, n + 1):
            p *= i
    return p


def cycle_length(f, x0, nmax=None, values=False):
    """For a given iterated sequence, return a generator that gives
    the length of the iterated cycle (lambda) and the length of terms
    before the cycle begins (mu); if ``values`` is True then the
    terms of the sequence will be returned instead. The sequence is
    started with value ``x0``.

    Note: more than the first lambda + mu terms may be returned and this
    is the cost of cycle detection with Brent's method; there are, however,
    generally less terms calculated than would have been calculated if the
    proper ending point were determined, e.g. by using Floyd's method.

    >>> from sympy.ntheory.generate import cycle_length

    This will yield successive values of i <-- func(i):

        >>> def gen(func, i):
        ...     while 1:
        ...         yield i
        ...         i = func(i)
        ...

    A function is defined:

        >>> func = lambda i: (i**2 + 1) % 51

    and given a seed of 4 and the mu and lambda terms calculated:

        >>> next(cycle_length(func, 4))
        (6, 3)

    We can see what is meant by looking at the output:

        >>> iter = cycle_length(func, 4, values=True)
        >>> list(iter)
        [4, 17, 35, 2, 5, 26, 14, 44, 50, 2, 5, 26, 14]

    There are 6 repeating values after the first 3.

    If a sequence is suspected of being longer than you might wish, ``nmax``
    can be used to exit early (and mu will be returned as None):

        >>> next(cycle_length(func, 4, nmax = 4))
        (4, None)
        >>> list(cycle_length(func, 4, nmax = 4, values=True))
        [4, 17, 35, 2]

    Code modified from:
        https://en.wikipedia.org/wiki/Cycle_detection.
    """

    nmax = int(nmax or 0)

    # main phase: search successive powers of two
    power = lam = 1
    tortoise, hare = x0, f(x0)  # f(x0) is the element/node next to x0.
    i = 1
    if values:
        yield tortoise
    while tortoise != hare and (not nmax or i < nmax):
        i += 1
        if power == lam:   # time to start a new power of two?
            tortoise = hare
            power *= 2
            lam = 0
        if values:
            yield hare
        hare = f(hare)
        lam += 1
    if nmax and i == nmax:
        if values:
            return
        else:
            yield nmax, None
            return
    if not values:
        # Find the position of the first repetition of length lambda
        mu = 0
        tortoise = hare = x0
        for i in range(lam):
            hare = f(hare)
        while tortoise != hare:
            tortoise = f(tortoise)
            hare = f(hare)
            mu += 1
        yield lam, mu


def composite(nth):
    """ Return the nth composite number, with the composite numbers indexed as
        composite(1) = 4, composite(2) = 6, etc....

        Examples
        ========

        >>> from sympy import composite
        >>> composite(36)
        52
        >>> composite(1)
        4
        >>> composite(17737)
        20000

        See Also
        ========

        sympy.ntheory.primetest.isprime : Test if n is prime
        primerange : Generate all primes in a given range
        primepi : Return the number of primes less than or equal to n
        prime : Return the nth prime
        compositepi : Return the number of positive composite numbers less than or equal to n
    """
    n = as_int(nth)
    if n < 1:
        raise ValueError("nth must be a positive integer; composite(1) == 4")
    composite_arr = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18]
    if n <= 10:
        return composite_arr[n - 1]

    a, b = 4, sieve._list[-1]
    if n <= b - _primepi(b) - 1:
        while a < b - 1:
            mid = (a + b) >> 1
            if mid - _primepi(mid) - 1 > n:
                b = mid
            else:
                a = mid
        if isprime(a):
            a -= 1
        return a

    from sympy.functions.elementary.exponential import log
    from sympy.functions.special.error_functions import li
    a = 4 # Lower bound for binary search
    b = int(n*(log(n) + log(log(n)))) # Upper bound for the search.

    while a < b:
        mid = (a + b) >> 1
        if mid - li(mid) - 1 > n:
            b = mid
        else:
            a = mid + 1

    n_composites = a - _primepi(a) - 1
    while n_composites > n:
        if not isprime(a):
            n_composites -= 1
        a -= 1
    if isprime(a):
        a -= 1
    return a


def compositepi(n):
    """ Return the number of positive composite numbers less than or equal to n.
        The first positive composite is 4, i.e. compositepi(4) = 1.

        Examples
        ========

        >>> from sympy import compositepi
        >>> compositepi(25)
        15
        >>> compositepi(1000)
        831

        See Also
        ========

        sympy.ntheory.primetest.isprime : Test if n is prime
        primerange : Generate all primes in a given range
        prime : Return the nth prime
        primepi : Return the number of primes less than or equal to n
        composite : Return the nth composite number
    """
    n = int(n)
    if n < 4:
        return 0
    return n - _primepi(n) - 1
<|endoftext|>"""
Primality testing

"""

from itertools import count

from sympy.core.sympify import sympify
from sympy.external.gmpy import (gmpy as _gmpy, gcd, jacobi,
                                 is_square as gmpy_is_square,
                                 bit_scan1, is_fermat_prp, is_euler_prp,
                                 is_selfridge_prp, is_strong_selfridge_prp,
                                 is_strong_bpsw_prp)
from sympy.external.ntheory import _lucas_sequence
from sympy.utilities.misc import as_int, filldedent

# Note: This list should be updated whenever new Mersenne primes are found.
# Refer: https://www.mersenne.org/
MERSENNE_PRIME_EXPONENTS = (2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203,
 2281, 3217, 4253, 4423, 9689, 9941, 11213, 19937, 21701, 23209, 44497, 86243, 110503, 132049,
 216091, 756839, 859433, 1257787, 1398269, 2976221, 3021377, 6972593, 13466917, 20996011, 24036583,
 25964951, 30402457, 32582657, 37156667, 42643801, 43112609, 57885161, 74207281, 77232917, 82589933,
 136279841)


def is_fermat_pseudoprime(n, a):
    r"""Returns True if ``n`` is prime or is an odd composite integer that
    is coprime to ``a`` and satisfy the modular arithmetic congruence relation:

    .. math ::
        a^{n-1} \equiv 1 \pmod{n}

    (where mod refers to the modulo operation).

    Parameters
    ==========

    n : Integer
        ``n`` is a positive integer.
    a : Integer
        ``a`` is a positive integer.
        ``a`` and ``n`` should be relatively prime.

    Returns
    =======

    bool : If ``n`` is prime, it always returns ``True``.
           The composite number that returns ``True`` is called an Fermat pseudoprime.

    Examples
    ========

    >>> from sympy.ntheory.primetest import is_fermat_pseudoprime
    >>> from sympy.ntheory.factor_ import isprime
    >>> for n in range(1, 1000):
    ...     if is_fermat_pseudoprime(n, 2) and not isprime(n):
    ...         print(n)
    341
    561
    645

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Fermat_pseudoprime
    """
    n, a = as_int(n), as_int(a)
    if a == 1:
        return n == 2 or bool(n % 2)
    return is_fermat_prp(n, a)


def is_euler_pseudoprime(n, a):
    r"""Returns True if ``n`` is prime or is an odd composite integer that
    is coprime to ``a`` and satisfy the modular arithmetic congruence relation:

    .. math ::
        a^{(n-1)/2} \equiv \pm 1 \pmod{n}

    (where mod refers to the modulo operation).

    Parameters
    ==========

    n : Integer
        ``n`` is a positive integer.
    a : Integer
        ``a`` is a positive integer.
        ``a`` and ``n`` should be relatively prime.

    Returns
    =======

    bool : If ``n`` is prime, it always returns ``True``.
           The composite number that returns ``True`` is called an Euler pseudoprime.

    Examples
    ========

    >>> from sympy.ntheory.primetest import is_euler_pseudoprime
    >>> from sympy.ntheory.factor_ import isprime
    >>> for n in range(1, 1000):
    ...     if is_euler_pseudoprime(n, 2) and not isprime(n):
    ...         print(n)
    341
    561

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euler_pseudoprime
    """
    n, a = as_int(n), as_int(a)
    if a < 1:
        raise ValueError("a should be an integer greater than 0")
    if n < 1:
        raise ValueError("n should be an integer greater than 0")
    if n == 1:
        return False
    if a == 1:
        return n == 2 or bool(n % 2)  # (prime or odd composite)
    if n % 2 == 0:
        return n == 2
    if gcd(n, a) != 1:
        raise ValueError("The two numbers should be relatively prime")
    return pow(a, (n - 1) // 2, n) in [1, n - 1]


def is_euler_jacobi_pseudoprime(n, a):
    r"""Returns True if ``n`` is prime or is an odd composite integer that
    is coprime to ``a`` and satisfy the modular arithmetic congruence relation:

    .. math ::
        a^{(n-1)/2} \equiv \left(\frac{a}{n}\right) \pmod{n}

    (where mod refers to the modulo operation).

    Parameters
    ==========

    n : Integer
        ``n`` is a positive integer.
    a : Integer
        ``a`` is a positive integer.
        ``a`` and ``n`` should be relatively prime.

    Returns
    =======

    bool : If ``n`` is prime, it always returns ``True``.
           The composite number that returns ``True`` is called an Euler-Jacobi pseudoprime.

    Examples
    ========

    >>> from sympy.ntheory.primetest import is_euler_jacobi_pseudoprime
    >>> from sympy.ntheory.factor_ import isprime
    >>> for n in range(1, 1000):
    ...     if is_euler_jacobi_pseudoprime(n, 2) and not isprime(n):
    ...         print(n)
    561

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euler%E2%80%93Jacobi_pseudoprime
    """
    n, a = as_int(n), as_int(a)
    if a == 1:
        return n == 2 or bool(n % 2)
    return is_euler_prp(n, a)


def is_square(n, prep=True):
    """Return True if n == a * a for some integer a, else False.
    If n is suspected of *not* being a square then this is a
    quick method of confirming that it is not.

    Examples
    ========

    >>> from sympy.ntheory.primetest import is_square
    >>> is_square(25)
    True
    >>> is_square(2)
    False

    References
    ==========

    .. [1]  https://mersenneforum.org/showpost.php?p=110896

    See Also
    ========
    sympy.core.intfunc.isqrt
    """
    if prep:
        n = as_int(n)
        if n < 0:
            return False
        if n in (0, 1):
            return True
    return gmpy_is_square(n)


def _test(n, base, s, t):
    """Miller-Rabin strong pseudoprime test for one base.
    Return False if n is definitely composite, True if n is
    probably prime, with a probability greater than 3/4.

    """
    # do the Fermat test
    b = pow(base, t, n)
    if b == 1 or b == n - 1:
        return True
    for _ in range(s - 1):
        b = pow(b, 2, n)
        if b == n - 1:
            return True
        # see I. Niven et al. "An Introduction to Theory of Numbers", page 78
        if b == 1:
            return False
    return False


def mr(n, bases):
    """Perform a Miller-Rabin strong pseudoprime test on n using a
    given list of bases/witnesses.

    References
    ==========

    .. [1] Richard Crandall & Carl Pomerance (2005), "Prime Numbers:
           A Computational Perspective", Springer, 2nd edition, 135-138

    A list of thresholds and the bases they require are here:
    https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test#Deterministic_variants

    Examples
    ========

    >>> from sympy.ntheory.primetest import mr
    >>> mr(1373651, [2, 3])
    False
    >>> mr(479001599, [31, 73])
    True

    """
    from sympy.polys.domains import ZZ

    n = as_int(n)
    if n < 2 or (n > 2 and n % 2 == 0):
        return False
    # remove powers of 2 from n-1 (= t * 2**s)
    s = bit_scan1(n - 1)
    t = n >> s
    for base in bases:
        # Bases >= n are wrapped, bases < 2 are invalid
        if base >= n:
            base %= n
        if base >= 2:
            base = ZZ(base)
            if not _test(n, base, s, t):
                return False
    return True


def _lucas_extrastrong_params(n):
    """Calculates the "extra strong" parameters (D, P, Q) for n.

    Parameters
    ==========

    n : int
        positive odd integer

    Returns
    =======

    D, P, Q: "extra strong" parameters.
             ``(0, 0, 0)`` if we find a nontrivial divisor of ``n``.

    Examples
    ========

    >>> from sympy.ntheory.primetest import _lucas_extrastrong_params
    >>> _lucas_extrastrong_params(101)
    (12, 4, 1)
    >>> _lucas_extrastrong_params(15)
    (0, 0, 0)

    References
    ==========
    .. [1] OEIS A217719: Extra Strong Lucas Pseudoprimes
           https://oeis.org/A217719
    .. [2] https://en.wikipedia.org/wiki/Lucas_pseudoprime

    """
    for P in count(3):
        D = P**2 - 4
        j = jacobi(D, n)
        if j == -1:
            return (D, P, 1)
        elif j == 0 and D % n:
            return (0, 0, 0)


def is_lucas_prp(n):
    """Standard Lucas compositeness test with Selfridge parameters.  Returns
    False if n is definitely composite, and True if n is a Lucas probable
    prime.

    This is typically used in combination with the Miller-Rabin test.

    References
    ==========
    .. [1] Robert Baillie, Samuel S. Wagstaff, Lucas Pseudoprimes,
           Math. Comp. Vol 35, Number 152 (1980), pp. 1391-1417,
           https://doi.org/10.1090%2FS0025-5718-1980-0583518-6
           http://mpqs.free.fr/LucasPseudoprimes.pdf
    .. [2] OEIS A217120: Lucas Pseudoprimes
           https://oeis.org/A217120
    .. [3] https://en.wikipedia.org/wiki/Lucas_pseudoprime

    Examples
    ========

    >>> from sympy.ntheory.primetest import isprime, is_lucas_prp
    >>> for i in range(10000):
    ...     if is_lucas_prp(i) and not isprime(i):
    ...         print(i)
    323
    377
    1159
    1829
    3827
    5459
    5777
    9071
    9179
    """
    n = as_int(n)
    if n < 2:
        return False
    return is_selfridge_prp(n)


def is_strong_lucas_prp(n):
    """Strong Lucas compositeness test with Selfridge parameters.  Returns
    False if n is definitely composite, and True if n is a strong Lucas
    probable prime.

    This is often used in combination with the Miller-Rabin test, and
    in particular, when combined with M-R base 2 creates the strong BPSW test.

    References
    ==========
    .. [1] Robert Baillie, Samuel S. Wagstaff, Lucas Pseudoprimes,
           Math. Comp. Vol 35, Number 152 (1980), pp. 1391-1417,
           https://doi.org/10.1090%2FS0025-5718-1980-0583518-6
           http://mpqs.free.fr/LucasPseudoprimes.pdf
    .. [2] OEIS A217255: Strong Lucas Pseudoprimes
           https://oeis.org/A217255
    .. [3] https://en.wikipedia.org/wiki/Lucas_pseudoprime
    .. [4] https://en.wikipedia.org/wiki/Baillie-PSW_primality_test

    Examples
    ========

    >>> from sympy.ntheory.primetest import isprime, is_strong_lucas_prp
    >>> for i in range(20000):
    ...     if is_strong_lucas_prp(i) and not isprime(i):
    ...        print(i)
    5459
    5777
    10877
    16109
    18971
    """
    n = as_int(n)
    if n < 2:
        return False
    return is_strong_selfridge_prp(n)


def is_extra_strong_lucas_prp(n):
    """Extra Strong Lucas compositeness test.  Returns False if n is
    definitely composite, and True if n is an "extra strong" Lucas probable
    prime.

    The parameters are selected using P = 3, Q = 1, then incrementing P until
    (D|n) == -1.  The test itself is as defined in [1]_, from the
    Mo and Jones preprint.  The parameter selection and test are the same as
    used in OEIS A217719, Perl's Math::Prime::Util, and the Lucas pseudoprime
    page on Wikipedia.

    It is 20-50% faster than the strong test.

    Because of the different parameters selected, there is no relationship
    between the strong Lucas pseudoprimes and extra strong Lucas pseudoprimes.
    In particular, one is not a subset of the other.

    References
    ==========
    .. [1] Jon Grantham, Frobenius Pseudoprimes,
           Math. Comp. Vol 70, Number 234 (2001), pp. 873-891,
           https://doi.org/10.1090%2FS0025-5718-00-01197-2
    .. [2] OEIS A217719: Extra Strong Lucas Pseudoprimes
           https://oeis.org/A217719
    .. [3] https://en.wikipedia.org/wiki/Lucas_pseudoprime

    Examples
    ========

    >>> from sympy.ntheory.primetest import isprime, is_extra_strong_lucas_prp
    >>> for i in range(20000):
    ...     if is_extra_strong_lucas_prp(i) and not isprime(i):
    ...        print(i)
    989
    3239
    5777
    10877
    """
    # Implementation notes:
    #   1) the parameters differ from Thomas R. Nicely's.  His parameter
    #      selection leads to pseudoprimes that overlap M-R tests, and
    #      contradict Baillie and Wagstaff's suggestion of (D|n) = -1.
    #   2) The MathWorld page as of June 2013 specifies Q=-1.  The Lucas
    #      sequence must have Q=1.  See Grantham theorem 2.3, any of the
    #      references on the MathWorld page, or run it and see Q=-1 is wrong.
    n = as_int(n)
    if n == 2:
        return True
    if n < 2 or (n % 2) == 0:
        return False
    if gmpy_is_square(n):
        return False

    D, P, Q = _lucas_extrastrong_params(n)
    if D == 0:
        return False

    # remove powers of 2 from n+1 (= k * 2**s)
    s = bit_scan1(n + 1)
    k = (n + 1) >> s

    U, V, _ = _lucas_sequence(n, P, Q, k)

    if U == 0 and (V == 2 or V == n - 2):
        return True
    for _ in range(1, s):
        if V == 0:
            return True
        V = (V*V - 2) % n
    return False


def proth_test(n):
    r""" Test if the Proth number `n = k2^m + 1` is prime. where k is a positive odd number and `2^m > k`.

    Parameters
    ==========

    n : Integer
        ``n`` is Proth number

    Returns
    =======

    bool : If ``True``, then ``n`` is the Proth prime

    Raises
    ======

    ValueError
        If ``n`` is not Proth number.

    Examples
    ========

    >>> from sympy.ntheory.primetest import proth_test
    >>> proth_test(41)
    True
    >>> proth_test(57)
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Proth_prime

    """
    n = as_int(n)
    if n < 3:
        raise ValueError("n is not Proth number")
    m = bit_scan1(n - 1)
    k = n >> m
    if m < k.bit_length():
        raise ValueError("n is not Proth number")
    if n % 3 == 0:
        return n == 3
    if k % 3: # n % 12 == 5
        return pow(3, n >> 1, n) == n - 1
    # If `n` is a square number, then `jacobi(a, n) = 1` for any `a`
    if gmpy_is_square(n):
        return False
    # `a` may be chosen at random.
    # In any case, we want to find `a` such that `jacobi(a, n) = -1`.
    for a in range(5, n):
        j = jacobi(a, n)
        if j == -1:
            return pow(a, n >> 1, n) == n - 1
        if j == 0:
            return False


def _lucas_lehmer_primality_test(p):
    r""" Test if the Mersenne number `M_p = 2^p-1` is prime.

    Parameters
    ==========

    p : int
        ``p`` is an odd prime number

    Returns
    =======

    bool : If ``True``, then `M_p` is the Mersenne prime

    Examples
    ========

    >>> from sympy.ntheory.primetest import _lucas_lehmer_primality_test
    >>> _lucas_lehmer_primality_test(5) # 2**5 - 1 = 31 is prime
    True
    >>> _lucas_lehmer_primality_test(11) # 2**11 - 1 = 2047 is not prime
    False

    See Also
    ========

    is_mersenne_prime

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lucas%E2%80%93Lehmer_primality_test

    """
    v = 4
    m = 2**p - 1
    for _ in range(p - 2):
        v = pow(v, 2, m) - 2
    return v == 0


def is_mersenne_prime(n):
    """Returns True if  ``n`` is a Mersenne prime, else False.

    A Mersenne prime is a prime number having the form `2^i - 1`.

    Examples
    ========

    >>> from sympy.ntheory.factor_ import is_mersenne_prime
    >>> is_mersenne_prime(6)
    False
    >>> is_mersenne_prime(127)
    True

    References
    ==========

    .. [1] https://mathworld.wolfram.com/MersennePrime.html

    """
    n = as_int(n)
    if n < 1:
        return False
    if n & (n + 1):
        # n is not Mersenne number
        return False
    p = n.bit_length()
    if p in MERSENNE_PRIME_EXPONENTS:
        return True
    if p < 65_000_000 or not isprime(p):
        # According to GIMPS, verification was completed on September 19, 2023 for p less than 65 million.
        # https://www.mersenne.org/report_milestones/
        # If p is composite number, then n=2**p-1 is composite number.
        return False
    result = _lucas_lehmer_primality_test(p)
    if result:
        raise ValueError(filldedent('''
            This Mersenne Prime, 2^%s - 1, should
            be added to SymPy's known values.''' % p))
    return result


_MR_BASES_32 = [15591, 2018, 166, 7429, 8064, 16045, 10503, 4399, 1949, 1295,
                2776, 3620, 560, 3128, 5212, 2657, 2300, 2021, 4652, 1471,
                9336, 4018, 2398, 20462, 10277, 8028, 2213, 6219, 620, 3763,
                4852, 5012, 3185, 1333, 6227,5298, 1074, 2391, 5113, 7061,
                803, 1269, 3875, 422, 751, 580, 4729, 10239, 746, 2951, 556,
                2206, 3778, 481, 1522, 3476, 481, 2487, 3266, 5633, 488, 3373,
                6441, 3344, 17, 15105, 1490, 4154, 2036, 1882, 1813, 467,
                3307, 14042, 6371, 658, 1005, 903, 737, 1887, 7447, 1888,
                2848, 1784, 7559, 3400, 951, 13969, 4304, 177, 41, 19875,
                3110, 13221, 8726, 571, 7043, 6943, 1199, 352, 6435, 165,
                1169, 3315, 978, 233, 3003, 2562, 2994, 10587, 10030, 2377,
                1902, 5354, 4447, 1555, 263, 27027, 2283, 305, 669, 1912, 601,
                6186, 429, 1930, 14873, 1784, 1661, 524, 3577, 236, 2360,
                6146, 2850, 55637, 1753, 4178, 8466, 222, 2579, 2743, 2031,
                2226, 2276, 374, 2132, 813, 23788, 1610, 4422, 5159, 1725,
                3597, 3366, 14336, 579, 165, 1375, 10018, 12616, 9816, 1371,
                536, 1867, 10864, 857, 2206, 5788, 434, 8085, 17618, 727,
                3639, 1595, 4944, 2129, 2029, 8195, 8344, 6232, 9183, 8126,
                1870, 3296, 7455, 8947, 25017, 541, 19115, 368, 566, 5674,
                411, 522, 1027, 8215, 2050, 6544, 10049, 614, 774, 2333, 3007,
                35201, 4706, 1152, 1785, 1028, 1540, 3743, 493, 4474, 2521,
                26845, 8354, 864, 18915, 5465, 2447, 42, 4511, 1660, 166,
                1249, 6259, 2553, 304, 272, 7286, 73, 6554, 899, 2816, 5197,
                13330, 7054, 2818, 3199, 811, 922, 350, 7514, 4452, 3449,
                2663, 4708, 418, 1621, 1171, 3471, 88, 11345, 412, 1559, 194]


def isprime(n):
    """
    Test if n is a prime number (True) or not (False). For n < 2^64 the
    answer is definitive; larger n values have a small probability of actually
    being pseudoprimes.

    Negative numbers (e.g. -2) are not considered prime.

    The first step is looking for trivial factors, which if found enables
    a quick return.  Next, if the sieve is large enough, use bisection search
    on the sieve.  For small numbers, a set of deterministic Miller-Rabin
    tests are performed with bases that are known to have no counterexamples
    in their range.  Finally if the number is larger than 2^64, a strong
    BPSW test is performed.  While this is a probable prime test and we
    believe counterexamples exist, there are no known counterexamples.

    Examples
    ========

    >>> from sympy.ntheory import isprime
    >>> isprime(13)
    True
    >>> isprime(15)
    False

    Notes
    =====

    This routine is intended only for integer input, not numerical
    expressions which may represent numbers. Floats are also
    rejected as input because they represent numbers of limited
    precision. While it is tempting to permit 7.0 to represent an
    integer there are errors that may "pass silently" if this is
    allowed:

    >>> from sympy import Float, S
    >>> int(1e3) == 1e3 == 10**3
    True
    >>> int(1e23) == 1e23
    True
    >>> int(1e23) == 10**23
    False

    >>> near_int = 1 + S(1)/10**19
    >>> near_int == int(near_int)
    False
    >>> n = Float(near_int, 10)  # truncated by precision
    >>> n % 1 == 0
    True
    >>> n = Float(near_int, 20)
    >>> n % 1 == 0
    False

    See Also
    ========

    sympy.ntheory.generate.primerange : Generates all primes in a given range
    sympy.functions.combinatorial.numbers.primepi : Return the number of primes less than or equal to n
    sympy.ntheory.generate.prime : Return the nth prime

    References
    ==========
    .. [1] https://en.wikipedia.org/wiki/Strong_pseudoprime
    .. [2] Robert Baillie, Samuel S. Wagstaff, Lucas Pseudoprimes,
           Math. Comp. Vol 35, Number 152 (1980), pp. 1391-1417,
           https://doi.org/10.1090%2FS0025-5718-1980-0583518-6
           http://mpqs.free.fr/LucasPseudoprimes.pdf
    .. [3] https://en.wikipedia.org/wiki/Baillie-PSW_primality_test
    """
    n = as_int(n)

    # Step 1, do quick composite testing via trial division.  The individual
    # modulo tests benchmark faster than one or two primorial igcds for me.
    # The point here is just to speedily handle small numbers and many
    # composites.  Step 2 only requires that n <= 2 get handled here.
    if n in [2, 3, 5]:
        return True
    if n < 2 or (n % 2) == 0 or (n % 3) == 0 or (n % 5) == 0:
        return False
    if n < 49:
        return True
    if (n %  7) == 0 or (n % 11) == 0 or (n % 13) == 0 or (n % 17) == 0 or \
       (n % 19) == 0 or (n % 23) == 0 or (n % 29) == 0 or (n % 31) == 0 or \
       (n % 37) == 0 or (n % 41) == 0 or (n % 43) == 0 or (n % 47) == 0:
        return False
    if n < 2809:
        return True
    if n < 65077:
        # There are only five Euler pseudoprimes with a least prime factor greater than 47
        return pow(2, n >> 1, n) in [1, n - 1] and n not in [8321, 31621, 42799, 49141, 49981]

    # bisection search on the sieve if the sieve is large enough
    from sympy.ntheory.generate import sieve as s
    if n <= s._list[-1]:
        l, u = s.search(n)
        return l == u
    from sympy.ntheory.factor_ import factor_cache
    if (ret := factor_cache.get(n)) is not None:
        return ret == n

    # If we have GMPY2, skip straight to step 3 and do a strong BPSW test.
    # This should be a bit faster than our step 2, and for large values will
    # be a lot faster than our step 3 (C+GMP vs. Python).
    if _gmpy is not None:
        return is_strong_bpsw_prp(n)


    # Step 2: deterministic Miller-Rabin testing for numbers < 2^64.  See:
    #    https://miller-rabin.appspot.com/
    # for lists.  We have made sure the M-R routine will successfully handle
    # bases larger than n, so we can use the minimal set.
    # In September 2015 deterministic numbers were extended to over 2^81.
    #    https://arxiv.org/pdf/1509.00864.pdf
    #    https://oeis.org/A014233
    if n < 341531:
        return mr(n, [9345883071009581737])
    if n < 4296595241:
        # Michal Forisek and Jakub Jancina,
        # Fast Primality Testing for Integers That Fit into a Machine Word
        # https://ceur-ws.org/Vol-1326/020-Forisek.pdf
        h = ((n >> 16) ^ n) * 0x45d9f3b
        h = ((h >> 16) ^ h) * 0x45d9f3b
        h = ((h >> 16) ^ h) & 255
        return mr(n, [_MR_BASES_32[h]])
    if n < 350269456337:
        return mr(n, [4230279247111683200, 14694767155120705706, 16641139526367750375])
    if n < 55245642489451:
        return mr(n, [2, 141889084524735, 1199124725622454117, 11096072698276303650])
    if n < 7999252175582851:
        return mr(n, [2, 4130806001517, 149795463772692060, 186635894390467037, 3967304179347715805])
    if n < 585226005592931977:
        return mr(n, [2, 123635709730000, 9233062284813009, 43835965440333360, 761179012939631437, 1263739024124850375])
    if n < 18446744073709551616:
        return mr(n, [2, 325, 9375, 28178, 450775, 9780504, 1795265022])
    if n < 318665857834031151167461:
        return mr(n, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37])
    if n < 3317044064679887385961981:
        return mr(n, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41])

    # We could do this instead at any point:
    #if n < 18446744073709551616:
    #   return mr(n, [2]) and is_extra_strong_lucas_prp(n)

    # Here are tests that are safe for MR routines that don't understand
    # large bases.
    #if n < 9080191:
    #    return mr(n, [31, 73])
    #if n < 19471033:
    #    return mr(n, [2, 299417])
    #if n < 38010307:
    #    return mr(n, [2, 9332593])
    #if n < 316349281:
    #    return mr(n, [11000544, 31481107])
    #if n < 4759123141:
    #    return mr(n, [2, 7, 61])
    #if n < 105936894253:
    #    return mr(n, [2, 1005905886, 1340600841])
    #if n < 31858317218647:
    #    return mr(n, [2, 642735, 553174392, 3046413974])
    #if n < 3071837692357849:
    #    return mr(n, [2, 75088, 642735, 203659041, 3613982119])
    #if n < 18446744073709551616:
    #    return mr(n, [2, 325, 9375, 28178, 450775, 9780504, 1795265022])

    # Step 3: BPSW.
    #
    #  Time for isprime(10**2000 + 4561), no gmpy or gmpy2 installed
    #     44.0s   old isprime using 46 bases
    #      5.3s   strong BPSW + one random base
    #      4.3s   extra strong BPSW + one random base
    #      4.1s   strong BPSW
    #      3.2s   extra strong BPSW

    # Classic BPSW from page 1401 of the paper.  See alternate ideas below.
    return is_strong_bpsw_prp(n)

    # Using extra strong test, which is somewhat faster
    #return mr(n, [2]) and is_extra_strong_lucas_prp(n)

    # Add a random M-R base
    #import random
    #return mr(n, [2, random.randint(3, n-1)]) and is_strong_lucas_prp(n)


def is_gaussian_prime(num):
    r"""Test if num is a Gaussian prime number.

    References
    ==========

    .. [1] https://oeis.org/wiki/Gaussian_primes
    """

    num = sympify(num)
    a, b = num.as_real_imag()
    a = as_int(a, strict=False)
    b = as_int(b, strict=False)
    if a == 0:
        b = abs(b)
        return isprime(b) and b % 4 == 3
    elif b == 0:
        a = abs(a)
        return isprime(a) and a % 4 == 3
    return isprime(a**2 + b**2)
<|endoftext|>from mpmath.libmp import (fzero, from_int, from_rational,
    fone, fhalf, bitcount, to_int, mpf_mul, mpf_div, mpf_sub,
    mpf_add, mpf_sqrt, mpf_pi, mpf_cosh_sinh, mpf_cos, mpf_sin)
from .residue_ntheory import _sqrt_mod_prime_power, is_quad_residue
from sympy.utilities.decorator import deprecated
from sympy.utilities.memoization import recurrence_memo

import math
from itertools import count

def _pre():
    maxn = 10**5
    global _factor, _totient
    _factor = [0]*maxn
    _totient = [1]*maxn
    lim = int(maxn**0.5) + 5
    for i in range(2, lim):
        if _factor[i] == 0:
            for j in range(i*i, maxn, i):
                if _factor[j] == 0:
                    _factor[j] = i
    for i in range(2, maxn):
        if _factor[i] == 0:
            _factor[i] = i
            _totient[i] = i-1
            continue
        x = _factor[i]
        y = i//x
        if y % x == 0:
            _totient[i] = _totient[y]*x
        else:
            _totient[i] = _totient[y]*(x - 1)

def _a(n, k, prec):
    """ Compute the inner sum in HRR formula [1]_

    References
    ==========

    .. [1] https://msp.org/pjm/1956/6-1/pjm-v6-n1-p18-p.pdf

    """
    if k == 1:
        return fone

    k1 = k
    e = 0
    p = _factor[k]
    while k1 % p == 0:
        k1 //= p
        e += 1
    k2 = k//k1 # k2 = p^e
    v = 1 - 24*n
    pi = mpf_pi(prec)

    if k1 == 1:
        # k  = p^e
        if p == 2:
            mod = 8*k
            v = mod + v % mod
            v = (v*pow(9, k - 1, mod)) % mod
            m = _sqrt_mod_prime_power(v, 2, e + 3)[0]
            arg = mpf_div(mpf_mul(
                from_int(4*m), pi, prec), from_int(mod), prec)
            return mpf_mul(mpf_mul(
                from_int((-1)**e*(2 - (m % 4))),
                mpf_sqrt(from_int(k), prec), prec),
                mpf_sin(arg, prec), prec)
        if p == 3:
            mod = 3*k
            v = mod + v % mod
            if e > 1:
                v = (v*pow(64, k//3 - 1, mod)) % mod
            m = _sqrt_mod_prime_power(v, 3, e + 1)[0]
            arg = mpf_div(mpf_mul(from_int(4*m), pi, prec),
                from_int(mod), prec)
            return mpf_mul(mpf_mul(
                from_int(2*(-1)**(e + 1)*(3 - 2*(m % 3))),
                mpf_sqrt(from_int(k//3), prec), prec),
                mpf_sin(arg, prec), prec)
        v = k + v % k
        jacobi3 = -1 if k % 12 in [5, 7] else 1
        if v % p == 0:
            if e == 1:
                return mpf_mul(
                    from_int(jacobi3),
                    mpf_sqrt(from_int(k), prec), prec)
            return fzero
        if not is_quad_residue(v, p):
            return fzero
        _phi = p**(e - 1)*(p - 1)
        v = (v*pow(576, _phi - 1, k))
        m = _sqrt_mod_prime_power(v, p, e)[0]
        arg = mpf_div(
            mpf_mul(from_int(4*m), pi, prec),
            from_int(k), prec)
        return mpf_mul(mpf_mul(
            from_int(2*jacobi3),
            mpf_sqrt(from_int(k), prec), prec),
            mpf_cos(arg, prec), prec)

    if p != 2 or e >= 3:
        d1, d2 = math.gcd(k1, 24), math.gcd(k2, 24)
        e = 24//(d1*d2)
        n1 = ((d2*e*n + (k2**2 - 1)//d1)*
            pow(e*k2*k2*d2, _totient[k1] - 1, k1)) % k1
        n2 = ((d1*e*n + (k1**2 - 1)//d2)*
            pow(e*k1*k1*d1, _totient[k2] - 1, k2)) % k2
        return mpf_mul(_a(n1, k1, prec), _a(n2, k2, prec), prec)
    if e == 2:
        n1 = ((8*n + 5)*pow(128, _totient[k1] - 1, k1)) % k1
        n2 = (4 + ((n - 2 - (k1**2 - 1)//8)*(k1**2)) % 4) % 4
        return mpf_mul(mpf_mul(
            from_int(-1),
            _a(n1, k1, prec), prec),
            _a(n2, k2, prec))
    n1 = ((8*n + 1)*pow(32, _totient[k1] - 1, k1)) % k1
    n2 = (2 + (n - (k1**2 - 1)//8) % 2) % 2
    return mpf_mul(_a(n1, k1, prec), _a(n2, k2, prec), prec)

def _d(n, j, prec, sq23pi, sqrt8):
    """
    Compute the sinh term in the outer sum of the HRR formula.
    The constants sqrt(2/3*pi) and sqrt(8) must be precomputed.
    """
    j = from_int(j)
    pi = mpf_pi(prec)
    a = mpf_div(sq23pi, j, prec)
    b = mpf_sub(from_int(n), from_rational(1, 24, prec), prec)
    c = mpf_sqrt(b, prec)
    ch, sh = mpf_cosh_sinh(mpf_mul(a, c), prec)
    D = mpf_div(
        mpf_sqrt(j, prec),
        mpf_mul(mpf_mul(sqrt8, b), pi), prec)
    E = mpf_sub(mpf_mul(a, ch), mpf_div(sh, c, prec), prec)
    return mpf_mul(D, E)


@recurrence_memo([1, 1])
def _partition_rec(n: int, prev) -> int:
    """ Calculate the partition function P(n)

    Parameters
    ==========

    n : int
        nonnegative integer

    """
    v = 0
    penta = 0 # pentagonal number: 1, 5, 12, ...
    for i in count():
        penta += 3*i + 1
        np = n - penta
        if np < 0:
            break
        s = prev[np]
        np -= i + 1
        # np = n - gp where gp = generalized pentagonal: 2, 7, 15, ...
        if 0 <= np:
            s += prev[np]
        v += -s if i % 2 else s
    return v


def _partition(n: int) -> int:
    """ Calculate the partition function P(n)

    Parameters
    ==========

    n : int

    """
    if n < 0:
        return 0
    if (n <= 200_000 and n - _partition_rec.cache_length() < 70 or
            _partition_rec.cache_length() == 2 and n < 14_400):
        # There will be 2*10**5 elements created here
        # and n elements created by partition, so in case we
        # are going to be working with small n, we just
        # use partition to calculate (and cache) the values
        # since lookup is used there while summation, using
        # _factor and _totient, will be used below. But we
        # only do so if n is relatively close to the length
        # of the cache since doing 1 calculation here is about
        # the same as adding 70 elements to the cache. In addition,
        # the startup here costs about the same as calculating the first
        # 14,400 values via partition, so we delay startup here unless n
        # is smaller than that.
        return _partition_rec(n)
    if '_factor' not in globals():
        _pre()
    # Estimate number of bits in p(n). This formula could be tidied
    pbits = int((
        math.pi*(2*n/3.)**0.5 -
        math.log(4*n))/math.log(10) + 1) * \
        math.log2(10)
    prec = p = int(pbits*1.1 + 100)

    # find the number of terms needed so rounded sum will be accurate
    # using Rademacher's bound M(n, N) for the remainder after a partial
    # sum of N terms (https://arxiv.org/pdf/1205.5991.pdf, (1.8))
    c1 = 44*math.pi**2/(225*math.sqrt(3))
    c2 = math.pi*math.sqrt(2)/75
    c3 = math.pi*math.sqrt(2/3)
    def _M(n, N):
        sqrt = math.sqrt
        return c1/sqrt(N) + c2*sqrt(N/(n - 1))*math.sinh(c3*sqrt(n)/N)
    big = max(9, math.ceil(n**0.5))  # should be too large (for n > 65, ceil should work)
    assert _M(n, big) < 0.5  # else double big until too large
    while big > 40 and _M(n, big) < 0.5:
        big //= 2
    small = big
    big = small*2
    while big - small > 1:
        N = (big + small)//2
        if (er := _M(n, N)) < 0.5:
            big = N
        elif er >= 0.5:
            small = N
    M = big  # done with function M; now have value

    # sanity check for expected size of answer
    if M > 10**5:  # i.e. M > maxn
        raise ValueError("Input too big")  # i.e. n > 149832547102

    # calculate it
    s = fzero
    sq23pi = mpf_mul(mpf_sqrt(from_rational(2, 3, p), p), mpf_pi(p), p)
    sqrt8 = mpf_sqrt(from_int(8), p)
    for q in range(1, M):
        a = _a(n, q, p)
        d = _d(n, q, p, sq23pi, sqrt8)
        s = mpf_add(s, mpf_mul(a, d), prec)
        # On average, the terms decrease rapidly in magnitude.
        # Dynamically reducing the precision greatly improves
        # performance.
        p = bitcount(abs(to_int(d))) + 50
    return int(to_int(mpf_add(s, fhalf, prec)))


@deprecated("""\
The `sympy.ntheory.partitions_.npartitions` has been moved to `sympy.functions.combinatorial.numbers.partition`.""",
deprecated_since_version="1.13",
active_deprecations_target='deprecated-ntheory-symbolic-functions')
def npartitions(n, verbose=False):
    """
    Calculate the partition function P(n), i.e. the number of ways that
    n can be written as a sum of positive integers.

    .. deprecated:: 1.13

        The ``npartitions`` function is deprecated. Use :class:`sympy.functions.combinatorial.numbers.partition`
        instead. See its documentation for more information. See
        :ref:`deprecated-ntheory-symbolic-functions` for details.

    P(n) is computed using the Hardy-Ramanujan-Rademacher formula [1]_.


    The correctness of this implementation has been tested through $10^{10}$.

    Examples
    ========

    >>> from sympy.functions.combinatorial.numbers import partition
    >>> partition(25)
    1958

    References
    ==========

    .. [1] https://mathworld.wolfram.com/PartitionFunctionP.html

    """
    from sympy.functions.combinatorial.numbers import partition as func_partition
    return func_partition(n)
<|endoftext|>"""
Number theory module (primes, etc)
"""

from .generate import nextprime, prevprime, prime, primepi, primerange, \
    randprime, Sieve, sieve, primorial, cycle_length, composite, compositepi
from .primetest import isprime, is_gaussian_prime, is_mersenne_prime
from .factor_ import divisors, proper_divisors, factorint, multiplicity, \
    multiplicity_in_factorial, perfect_power, factor_cache, pollard_pm1, \
    pollard_rho, primefactors, totient, \
    divisor_count, proper_divisor_count, divisor_sigma, factorrat, \
    reduced_totient, primenu, primeomega, mersenne_prime_exponent, \
    is_perfect, is_abundant, is_deficient, is_amicable, is_carmichael, \
    abundance, dra, drm

from .partitions_ import npartitions
from .residue_ntheory import is_primitive_root, is_quad_residue, \
    legendre_symbol, jacobi_symbol, n_order, sqrt_mod, quadratic_residues, \
    primitive_root, nthroot_mod, is_nthpow_residue, sqrt_mod_iter, mobius, \
    discrete_log, quadratic_congruence, polynomial_congruence
from .multinomial import binomial_coefficients, binomial_coefficients_list, \
    multinomial_coefficients
from .continued_fraction import continued_fraction_periodic, \
    continued_fraction_iterator, continued_fraction_reduce, \
    continued_fraction_convergents, continued_fraction
from .digits import count_digits, digits, is_palindromic
from .egyptian_fraction import egyptian_fraction
from .ecm import ecm
from .qs import qs, qs_factor
__all__ = [
    'nextprime', 'prevprime', 'prime', 'primepi', 'primerange', 'randprime',
    'Sieve', 'sieve', 'primorial', 'cycle_length', 'composite', 'compositepi',

    'isprime', 'is_gaussian_prime', 'is_mersenne_prime',


    'divisors', 'proper_divisors', 'factorint', 'multiplicity', 'perfect_power',
    'pollard_pm1', 'factor_cache', 'pollard_rho', 'primefactors', 'totient',
    'divisor_count', 'proper_divisor_count', 'divisor_sigma', 'factorrat',
    'reduced_totient', 'primenu', 'primeomega', 'mersenne_prime_exponent',
    'is_perfect', 'is_abundant', 'is_deficient', 'is_amicable',
    'is_carmichael', 'abundance', 'dra', 'drm', 'multiplicity_in_factorial',

    'npartitions',

    'is_primitive_root', 'is_quad_residue', 'legendre_symbol',
    'jacobi_symbol', 'n_order', 'sqrt_mod', 'quadratic_residues',
    'primitive_root', 'nthroot_mod', 'is_nthpow_residue', 'sqrt_mod_iter',
    'mobius', 'discrete_log', 'quadratic_congruence', 'polynomial_congruence',

    'binomial_coefficients', 'binomial_coefficients_list',
    'multinomial_coefficients',

    'continued_fraction_periodic', 'continued_fraction_iterator',
    'continued_fraction_reduce', 'continued_fraction_convergents',
    'continued_fraction',

    'digits',
    'count_digits',
    'is_palindromic',

    'egyptian_fraction',

    'ecm',

    'qs', 'qs_factor',
]
<|endoftext|>from __future__ import annotations
import itertools
from sympy.core.exprtools import factor_terms
from sympy.core.numbers import Integer, Rational
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import _sympify
from sympy.utilities.misc import as_int


def continued_fraction(a) -> list:
    """Return the continued fraction representation of a Rational or
    quadratic irrational.

    Examples
    ========

    >>> from sympy.ntheory.continued_fraction import continued_fraction
    >>> from sympy import sqrt
    >>> continued_fraction((1 + 2*sqrt(3))/5)
    [0, 1, [8, 3, 34, 3]]

    See Also
    ========
    continued_fraction_periodic, continued_fraction_reduce, continued_fraction_convergents
    """
    e = _sympify(a)
    if all(i.is_Rational for i in e.atoms()):
        if e.is_Integer:
            return continued_fraction_periodic(e, 1, 0)
        elif e.is_Rational:
            return continued_fraction_periodic(e.p, e.q, 0)
        elif e.is_Pow and e.exp is S.Half and e.base.is_Integer:
            return continued_fraction_periodic(0, 1, e.base)
        elif e.is_Mul and len(e.args) == 2 and (
                e.args[0].is_Rational and
                e.args[1].is_Pow and
                e.args[1].base.is_Integer and
                e.args[1].exp is S.Half):
            a, b = e.args
            return continued_fraction_periodic(0, a.q, b.base, a.p)
        else:
            # this should not have to work very hard- no
            # simplification, cancel, etc... which should be
            # done by the user.  e.g. This is a fancy 1 but
            # the user should simplify it first:
            # sqrt(2)*(1 + sqrt(2))/(sqrt(2) + 2)
            p, d = e.expand().as_numer_denom()
            if d.is_Integer:
                if p.is_Rational:
                    return continued_fraction_periodic(p, d)
                # look for a + b*c
                # with c = sqrt(s)
                if p.is_Add and len(p.args) == 2:
                    a, bc = p.args
                else:
                    a = S.Zero
                    bc = p
                if a.is_Integer:
                    b = S.NaN
                    if bc.is_Mul and len(bc.args) == 2:
                        b, c = bc.args
                    elif bc.is_Pow:
                        b = Integer(1)
                        c = bc
                    if b.is_Integer and (
                            c.is_Pow and c.exp is S.Half and
                            c.base.is_Integer):
                        # (a + b*sqrt(c))/d
                        c = c.base
                        return continued_fraction_periodic(a, d, c, b)
    raise ValueError(
        'expecting a rational or quadratic irrational, not %s' % e)


def continued_fraction_periodic(p, q, d=0, s=1) -> list:
    r"""
    Find the periodic continued fraction expansion of a quadratic irrational.

    Compute the continued fraction expansion of a rational or a
    quadratic irrational number, i.e. `\frac{p + s\sqrt{d}}{q}`, where
    `p`, `q \ne 0` and `d \ge 0` are integers.

    Returns the continued fraction representation (canonical form) as
    a list of integers, optionally ending (for quadratic irrationals)
    with list of integers representing the repeating digits.

    Parameters
    ==========

    p : int
        the rational part of the number's numerator
    q : int
        the denominator of the number
    d : int, optional
        the irrational part (discriminator) of the number's numerator
    s : int, optional
        the coefficient of the irrational part

    Examples
    ========

    >>> from sympy.ntheory.continued_fraction import continued_fraction_periodic
    >>> continued_fraction_periodic(3, 2, 7)
    [2, [1, 4, 1, 1]]

    Golden ratio has the simplest continued fraction expansion:

    >>> continued_fraction_periodic(1, 2, 5)
    [[1]]

    If the discriminator is zero or a perfect square then the number will be a
    rational number:

    >>> continued_fraction_periodic(4, 3, 0)
    [1, 3]
    >>> continued_fraction_periodic(4, 3, 49)
    [3, 1, 2]

    See Also
    ========

    continued_fraction_iterator, continued_fraction_reduce

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Periodic_continued_fraction
    .. [2] K. Rosen. Elementary Number theory and its applications.
           Addison-Wesley, 3 Sub edition, pages 379-381, January 1992.

    """
    from sympy.functions import sqrt, floor

    p, q, d, s = list(map(as_int, [p, q, d, s]))

    if d < 0:
        raise ValueError("expected non-negative for `d` but got %s" % d)

    if q == 0:
        raise ValueError("The denominator cannot be 0.")

    if not s:
        d = 0

    # check for rational case
    sd = sqrt(d)
    if sd.is_Integer:
        return list(continued_fraction_iterator(Rational(p + s*sd, q)))

    # irrational case with sd != Integer
    if q < 0:
        p, q, s = -p, -q, -s

    n = (p + s*sd)/q
    if n < 0:
        w = floor(-n)
        f = -n - w
        one_f = continued_fraction(1 - f)  # 1-f < 1 so cf is [0 ... [...]]
        one_f[0] -= w + 1
        return one_f

    d *= s**2
    sd *= s

    if (d - p**2)%q:
        d *= q**2
        sd *= q
        p *= q
        q *= q

    terms: list[int] = []
    pq = {}

    while (p, q) not in pq:
        pq[(p, q)] = len(terms)
        terms.append((p + sd)//q)
        p = terms[-1]*q - p
        q = (d - p**2)//q

    i = pq[(p, q)]
    return terms[:i] + [terms[i:]]  # type: ignore


def continued_fraction_reduce(cf):
    """
    Reduce a continued fraction to a rational or quadratic irrational.

    Compute the rational or quadratic irrational number from its
    terminating or periodic continued fraction expansion.  The
    continued fraction expansion (cf) should be supplied as a
    terminating iterator supplying the terms of the expansion.  For
    terminating continued fractions, this is equivalent to
    ``list(continued_fraction_convergents(cf))[-1]``, only a little more
    efficient.  If the expansion has a repeating part, a list of the
    repeating terms should be returned as the last element from the
    iterator.  This is the format returned by
    continued_fraction_periodic.

    For quadratic irrationals, returns the largest solution found,
    which is generally the one sought, if the fraction is in canonical
    form (all terms positive except possibly the first).

    Examples
    ========

    >>> from sympy.ntheory.continued_fraction import continued_fraction_reduce
    >>> continued_fraction_reduce([1, 2, 3, 4, 5])
    225/157
    >>> continued_fraction_reduce([-2, 1, 9, 7, 1, 2])
    -256/233
    >>> continued_fraction_reduce([2, 1, 2, 1, 1, 4, 1, 1, 6, 1, 1, 8]).n(10)
    2.718281835
    >>> continued_fraction_reduce([1, 4, 2, [3, 1]])
    (sqrt(21) + 287)/238
    >>> continued_fraction_reduce([[1]])
    (1 + sqrt(5))/2
    >>> from sympy.ntheory.continued_fraction import continued_fraction_periodic
    >>> continued_fraction_reduce(continued_fraction_periodic(8, 5, 13))
    (sqrt(13) + 8)/5

    See Also
    ========

    continued_fraction_periodic

    """
    from sympy.solvers import solve

    period = []
    x = Dummy('x')

    def untillist(cf):
        for nxt in cf:
            if isinstance(nxt, list):
                period.extend(nxt)
                yield x
                break
            yield nxt

    a = S.Zero
    for a in continued_fraction_convergents(untillist(cf)):
        pass

    if period:
        y = Dummy('y')
        solns = solve(continued_fraction_reduce(period + [y]) - y, y)
        solns.sort()
        pure = solns[-1]
        rv = a.subs(x, pure).radsimp()
    else:
        rv = a
    if rv.is_Add:
        rv = factor_terms(rv)
        if rv.is_Mul and rv.args[0] == -1:
            rv = rv.func(*rv.args)
    return rv


def continued_fraction_iterator(x):
    """
    Return continued fraction expansion of x as iterator.

    Examples
    ========

    >>> from sympy import Rational, pi
    >>> from sympy.ntheory.continued_fraction import continued_fraction_iterator

    >>> list(continued_fraction_iterator(Rational(3, 8)))
    [0, 2, 1, 2]
    >>> list(continued_fraction_iterator(Rational(-3, 8)))
    [-1, 1, 1, 1, 2]

    >>> for i, v in enumerate(continued_fraction_iterator(pi)):
    ...     if i > 7:
    ...         break
    ...     print(v)
    3
    7
    15
    1
    292
    1
    1
    1

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Continued_fraction

    """
    from sympy.functions import floor
    while True:
        i = floor(x)
        yield i
        x -= i
        if not x:
            break
        x = 1/x


def continued_fraction_convergents(cf):
    """
    Return an iterator over the convergents of a continued fraction (cf).

    The parameter should be in either of the following to forms:
    - A list of partial quotients, possibly with the last element being a list
    of repeating partial quotients, such as might be returned by
    continued_fraction and continued_fraction_periodic.
    - An iterable returning successive partial quotients of the continued
    fraction, such as might be returned by continued_fraction_iterator.

    In computing the convergents, the continued fraction need not be strictly
    in canonical form (all integers, all but the first positive).
    Rational and negative elements may be present in the expansion.

    Examples
    ========

    >>> from sympy.core import pi
    >>> from sympy import S
    >>> from sympy.ntheory.continued_fraction import \
            continued_fraction_convergents, continued_fraction_iterator

    >>> list(continued_fraction_convergents([0, 2, 1, 2]))
    [0, 1/2, 1/3, 3/8]

    >>> list(continued_fraction_convergents([1, S('1/2'), -7, S('1/4')]))
    [1, 3, 19/5, 7]

    >>> it = continued_fraction_convergents(continued_fraction_iterator(pi))
    >>> for n in range(7):
    ...     print(next(it))
    3
    22/7
    333/106
    355/113
    103993/33102
    104348/33215
    208341/66317

    >>> it = continued_fraction_convergents([1, [1, 2]])  # sqrt(3)
    >>> for n in range(7):
    ...     print(next(it))
    1
    2
    5/3
    7/4
    19/11
    26/15
    71/41

    See Also
    ========

    continued_fraction_iterator, continued_fraction, continued_fraction_periodic

    """
    if isinstance(cf, list) and isinstance(cf[-1], list):
        cf = itertools.chain(cf[:-1], itertools.cycle(cf[-1]))
    p_2, q_2 = S.Zero, S.One
    p_1, q_1 = S.One, S.Zero
    for a in cf:
        p, q = a*p_1 + p_2, a*q_1 + q_2
        p_2, q_2 = p_1, q_1
        p_1, q_1 = p, q
        yield p/q
<|endoftext|>from sympy.core.numbers import oo
from sympy.core.symbol import symbols
from sympy.polys.domains import FiniteField, QQ, RationalField, FF
from sympy.polys.polytools import Poly
from sympy.solvers.solvers import solve
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import as_int
from .factor_ import divisors
from .residue_ntheory import polynomial_congruence


class EllipticCurve:
    """
    Create the following Elliptic Curve over domain.

    `y^{2} + a_{1} x y + a_{3} y = x^{3} + a_{2} x^{2} + a_{4} x + a_{6}`

    The default domain is ``QQ``. If no coefficient ``a1``, ``a2``, ``a3``,
    is given then it creates a curve with the following form:

    `y^{2} = x^{3} + a_{4} x + a_{6}`

    Examples
    ========

    References
    ==========

    .. [1] J. Silverman "A Friendly Introduction to Number Theory" Third Edition
    .. [2] https://mathworld.wolfram.com/EllipticDiscriminant.html
    .. [3] G. Hardy, E. Wright "An Introduction to the Theory of Numbers" Sixth Edition

    """

    def __init__(self, a4, a6, a1=0, a2=0, a3=0, modulus=0):
        if modulus == 0:
            domain = QQ
        else:
            domain = FF(modulus)
        a1, a2, a3, a4, a6 = map(domain.convert, (a1, a2, a3, a4, a6))
        self._domain = domain
        self.modulus = modulus
        # Calculate discriminant
        b2 = a1**2 + 4 * a2
        b4 = 2 * a4 + a1 * a3
        b6 = a3**2 + 4 * a6
        b8 = a1**2 * a6 + 4 * a2 * a6 - a1 * a3 * a4 + a2 * a3**2 - a4**2
        self._b2, self._b4, self._b6, self._b8 = b2, b4, b6, b8
        self._discrim = -b2**2 * b8 - 8 * b4**3 - 27 * b6**2 + 9 * b2 * b4 * b6
        self._a1 = a1
        self._a2 = a2
        self._a3 = a3
        self._a4 = a4
        self._a6 = a6
        x, y, z = symbols('x y z')
        self.x, self.y, self.z = x, y, z
        self._poly = Poly(y**2*z + a1*x*y*z + a3*y*z**2 - x**3 - a2*x**2*z - a4*x*z**2 - a6*z**3, domain=domain)
        if isinstance(self._domain, FiniteField):
            self._rank = 0
        elif isinstance(self._domain, RationalField):
            self._rank = None

    def __call__(self, x, y, z=1):
        return EllipticCurvePoint(x, y, z, self)

    def __contains__(self, point):
        if is_sequence(point):
            if len(point) == 2:
                z1 = 1
            else:
                z1 = point[2]
            x1, y1 = point[:2]
        elif isinstance(point, EllipticCurvePoint):
            x1, y1, z1 = point.x, point.y, point.z
        else:
            raise ValueError('Invalid point.')
        if self.characteristic == 0 and z1 == 0:
            return True
        return self._poly.subs({self.x: x1, self.y: y1, self.z: z1}) == 0

    def __repr__(self):
        return self._poly.__repr__()

    def minimal(self):
        """
        Return minimal Weierstrass equation.

        Examples
        ========

        >>> from sympy.ntheory.elliptic_curve import EllipticCurve

        >>> e1 = EllipticCurve(-10, -20, 0, -1, 1)
        >>> e1.minimal()
        Poly(-x**3 + 13392*x*z**2 + y**2*z + 1080432*z**3, x, y, z, domain='QQ')

        """
        char = self.characteristic
        if char == 2:
            return self
        if char == 3:
            return EllipticCurve(self._b4/2, self._b6/4, a2=self._b2/4, modulus=self.modulus)
        c4 = self._b2**2 - 24*self._b4
        c6 = -self._b2**3 + 36*self._b2*self._b4 - 216*self._b6
        return EllipticCurve(-27*c4, -54*c6, modulus=self.modulus)

    def points(self):
        """
        Return points of curve over Finite Field.

        Examples
        ========

        >>> from sympy.ntheory.elliptic_curve import EllipticCurve
        >>> e2 = EllipticCurve(1, 1, 1, 1, 1, modulus=5)
        >>> e2.points()
        {(0, 2), (1, 4), (2, 0), (2, 2), (3, 0), (3, 1), (4, 0)}

        """

        char = self.characteristic
        all_pt = set()
        if char >= 1:
            for i in range(char):
                congruence_eq = self._poly.subs({self.x: i, self.z: 1}).expr
                sol = polynomial_congruence(congruence_eq, char)
                all_pt.update((i, num) for num in sol)
            return all_pt
        else:
            raise ValueError("Infinitely many points")

    def points_x(self, x):
        """Returns points on the curve for the given x-coordinate."""
        pt = []
        if self._domain == QQ:
            for y in solve(self._poly.subs(self.x, x)):
                pt.append((x, y))
        else:
            congruence_eq = self._poly.subs({self.x: x, self.z: 1}).expr
            for y in polynomial_congruence(congruence_eq, self.characteristic):
                pt.append((x, y))
        return pt

    def torsion_points(self):
        """
        Return torsion points of curve over Rational number.

        Return point objects those are finite order.
        According to Nagell-Lutz theorem, torsion point p(x, y)
        x and y are integers, either y = 0 or y**2 is divisor
        of discriminent. According to Mazur's theorem, there are
        at most 15 points in torsion collection.

        Examples
        ========

        >>> from sympy.ntheory.elliptic_curve import EllipticCurve
        >>> e2 = EllipticCurve(-43, 166)
        >>> sorted(e2.torsion_points())
        [(-5, -16), (-5, 16), O, (3, -8), (3, 8), (11, -32), (11, 32)]

        """
        if self.characteristic > 0:
            raise ValueError("No torsion point for Finite Field.")
        l = [EllipticCurvePoint.point_at_infinity(self)]
        for xx in solve(self._poly.subs({self.y: 0, self.z: 1})):
            if xx.is_rational:
                l.append(self(xx, 0))
        for i in divisors(self.discriminant, generator=True):
            j = int(i**.5)
            if j**2 == i:
                for xx in solve(self._poly.subs({self.y: j, self.z: 1})):
                    if not xx.is_rational:
                        continue
                    p = self(xx, j)
                    if p.order() != oo:
                        l.extend([p, -p])
        return l

    @property
    def characteristic(self):
        """
        Return domain characteristic.

        Examples
        ========

        >>> from sympy.ntheory.elliptic_curve import EllipticCurve
        >>> e2 = EllipticCurve(-43, 166)
        >>> e2.characteristic
        0

        """
        return self._domain.characteristic()

    @property
    def discriminant(self):
        """
        Return curve discriminant.

        Examples
        ========

        >>> from sympy.ntheory.elliptic_curve import EllipticCurve
        >>> e2 = EllipticCurve(0, 17)
        >>> e2.discriminant
        -124848

        """
        return int(self._discrim)

    @property
    def is_singular(self):
        """
        Return True if curve discriminant is equal to zero.
        """
        return self.discriminant == 0

    @property
    def j_invariant(self):
        """
        Return curve j-invariant.

        Examples
        ========

        >>> from sympy.ntheory.elliptic_curve import EllipticCurve
        >>> e1 = EllipticCurve(-2, 0, 0, 1, 1)
        >>> e1.j_invariant
        1404928/389

        """
        c4 = self._b2**2 - 24*self._b4
        return self._domain.to_sympy(c4**3 / self._discrim)

    @property
    def order(self):
        """
        Number of points in Finite field.

        Examples
        ========

        >>> from sympy.ntheory.elliptic_curve import EllipticCurve
        >>> e2 = EllipticCurve(1, 0, modulus=19)
        >>> e2.order
        19

        """
        if self.characteristic == 0:
            raise NotImplementedError("Still not implemented")
        return len(self.points())

    @property
    def rank(self):
        """
        Number of independent points of infinite order.

        For Finite field, it must be 0.
        """
        if self._rank is not None:
            return self._rank
        raise NotImplementedError("Still not implemented")


class EllipticCurvePoint:
    """
    Point of Elliptic Curve

    Examples
    ========

    >>> from sympy.ntheory.elliptic_curve import EllipticCurve
    >>> e1 = EllipticCurve(-17, 16)
    >>> p1 = e1(0, -4, 1)
    >>> p2 = e1(1, 0)
    >>> p1 + p2
    (15, -56)
    >>> e3 = EllipticCurve(-1, 9)
    >>> e3(1, -3) * 3
    (664/169, 17811/2197)
    >>> (e3(1, -3) * 3).order()
    oo
    >>> e2 = EllipticCurve(-2, 0, 0, 1, 1)
    >>> p = e2(-1,1)
    >>> q = e2(0, -1)
    >>> p+q
    (4, 8)
    >>> p-q
    (1, 0)
    >>> 3*p-5*q
    (328/361, -2800/6859)
    """

    @staticmethod
    def point_at_infinity(curve):
        return EllipticCurvePoint(0, 1, 0, curve)

    def __init__(self, x, y, z, curve):
        dom = curve._domain.convert
        self.x = dom(x)
        self.y = dom(y)
        self.z = dom(z)
        self._curve = curve
        self._domain = self._curve._domain
        if not self._curve.__contains__(self):
            raise ValueError("The curve does not contain this point")

    def __add__(self, p):
        if self.z == 0:
            return p
        if p.z == 0:
            return self
        x1, y1 = self.x/self.z, self.y/self.z
        x2, y2 = p.x/p.z, p.y/p.z
        a1 = self._curve._a1
        a2 = self._curve._a2
        a3 = self._curve._a3
        a4 = self._curve._a4
        a6 = self._curve._a6
        if x1 != x2:
            slope = (y1 - y2) / (x1 - x2)
            yint = (y1 * x2 - y2 * x1) / (x2 - x1)
        else:
            if (y1 + y2) == 0:
                return self.point_at_infinity(self._curve)
            slope = (3 * x1**2 + 2*a2*x1 + a4 - a1*y1) / (a1 * x1 + a3 + 2 * y1)
            yint = (-x1**3 + a4*x1 + 2*a6 - a3*y1) / (a1*x1 + a3 + 2*y1)
        x3 = slope**2 + a1*slope - a2 - x1 - x2
        y3 = -(slope + a1) * x3 - yint - a3
        return self._curve(x3, y3, 1)

    def __lt__(self, other):
        return (self.x, self.y, self.z) < (other.x, other.y, other.z)

    def __mul__(self, n):
        n = as_int(n)
        r = self.point_at_infinity(self._curve)
        if n == 0:
            return r
        if n < 0:
            return -self * -n
        p = self
        while n:
            if n & 1:
                r = r + p
            n >>= 1
            p = p + p
        return r

    def __rmul__(self, n):
        return self * n

    def __neg__(self):
        return EllipticCurvePoint(self.x, -self.y - self._curve._a1*self.x - self._curve._a3, self.z, self._curve)

    def __repr__(self):
        if self.z == 0:
            return 'O'
        dom = self._curve._domain
        try:
            return '({}, {})'.format(dom.to_sympy(self.x), dom.to_sympy(self.y))
        except TypeError:
            pass
        return '({}, {})'.format(self.x, self.y)

    def __sub__(self, other):
        return self.__add__(-other)

    def order(self):
        """
        Return point order n where nP = 0.

        """
        if self.z == 0:
            return 1
        if self.y == 0:  # P = -P
            return 2
        p = self * 2
        if p.y == -self.y:  # 2P = -P
            return 3
        i = 2
        if self._domain != QQ:
            while int(p.x) == p.x and int(p.y) == p.y:
                p = self + p
                i += 1
                if p.z == 0:
                    return i
            return oo
        while p.x.numerator == p.x and p.y.numerator == p.y:
            p = self + p
            i += 1
            if i > 12:
                return oo
            if p.z == 0:
                return i
        return oo
<|endoftext|>from math import exp, log
from sympy.core.random import _randint
from sympy.external.gmpy import bit_scan1, gcd, invert, sqrt as isqrt
from sympy.ntheory.factor_ import _perfect_power
from sympy.ntheory.primetest import isprime
from sympy.ntheory.residue_ntheory import _sqrt_mod_prime_power


class SievePolynomial:
    def __init__(self, a, b, N):
        """This class denotes the sieve polynomial.
        Provide methods to compute `(a*x + b)**2 - N` and
        `a*x + b` when given `x`.

        Parameters
        ==========

        a : parameter of the sieve polynomial
        b : parameter of the sieve polynomial
        N : number to be factored

        """
        self.a = a
        self.b = b
        self.a2 = a**2
        self.ab = 2*a*b
        self.b2 = b**2 - N

    def eval_u(self, x):
        return self.a*x + self.b

    def eval_v(self, x):
        return (self.a2*x + self.ab)*x + self.b2


class FactorBaseElem:
    """This class stores an element of the `factor_base`.
    """
    def __init__(self, prime, tmem_p, log_p):
        """
        Initialization of factor_base_elem.

        Parameters
        ==========

        prime : prime number of the factor_base
        tmem_p : Integer square root of x**2 = n mod prime
        log_p : Compute Natural Logarithm of the prime
        """
        self.prime = prime
        self.tmem_p = tmem_p
        self.log_p = log_p
        # `soln1` and `soln2` are solutions to
        # the equation `(a*x + b)**2 - N = 0 (mod p)`.
        self.soln1 = None
        self.soln2 = None
        self.b_ainv = None


def _generate_factor_base(prime_bound, n):
    """Generate `factor_base` for Quadratic Sieve. The `factor_base`
    consists of all the points whose ``legendre_symbol(n, p) == 1``
    and ``p < num_primes``. Along with the prime `factor_base` also stores
    natural logarithm of prime and the residue n modulo p.
    It also returns the of primes numbers in the `factor_base` which are
    close to 1000 and 5000.

    Parameters
    ==========

    prime_bound : upper prime bound of the factor_base
    n : integer to be factored
    """
    from sympy.ntheory.generate import sieve
    factor_base = []
    idx_1000, idx_5000 = None, None
    for prime in sieve.primerange(1, prime_bound):
        if pow(n, (prime - 1) // 2, prime) == 1:
            if prime > 1000 and idx_1000 is None:
                idx_1000 = len(factor_base) - 1
            if prime > 5000 and idx_5000 is None:
                idx_5000 = len(factor_base) - 1
            residue = _sqrt_mod_prime_power(n, prime, 1)[0]
            log_p = round(log(prime)*2**10)
            factor_base.append(FactorBaseElem(prime, residue, log_p))
    return idx_1000, idx_5000, factor_base


def _generate_polynomial(N, M, factor_base, idx_1000, idx_5000, randint):
    """ Generate sieve polynomials indefinitely.
    Information such as `soln1` in the `factor_base` associated with
    the polynomial is modified in place.

    Parameters
    ==========

    N : Number to be factored
    M : sieve interval
    factor_base : factor_base primes
    idx_1000 : index of prime number in the factor_base near 1000
    idx_5000 : index of prime number in the factor_base near to 5000
    randint : A callable that takes two integers (a, b) and returns a random integer
              n such that a <= n <= b, similar to `random.randint`.
    """
    approx_val = log(2*N)/2 - log(M)
    start = idx_1000 or 0
    end = idx_5000 or (len(factor_base) - 1)
    while True:
        # Choose `a` that is close to `sqrt(2*N) / M`
        best_a, best_q, best_ratio = None, None, None
        for _ in range(50):
            a = 1
            q = []
            while log(a) < approx_val:
                rand_p = 0
                while(rand_p == 0 or rand_p in q):
                    rand_p = randint(start, end)
                p = factor_base[rand_p].prime
                a *= p
                q.append(rand_p)
            ratio = exp(log(a) - approx_val)
            if best_ratio is None or abs(ratio - 1) < abs(best_ratio - 1):
                best_q = q
                best_a = a
                best_ratio = ratio

        # Set `b` using the Chinese remainder theorem
        a = best_a
        q = best_q
        B = []
        for val in q:
            q_l = factor_base[val].prime
            gamma = factor_base[val].tmem_p * invert(a // q_l, q_l) % q_l
            if 2*gamma > q_l:
                gamma = q_l - gamma
            B.append(a//q_l*gamma)
        b = sum(B)
        g = SievePolynomial(a, b, N)
        for fb in factor_base:
            if a % fb.prime == 0:
                fb.soln1 = None
                continue
            a_inv = invert(a, fb.prime)
            fb.b_ainv = [2*b_elem*a_inv % fb.prime for b_elem in B]
            fb.soln1 = (a_inv*(fb.tmem_p - b)) % fb.prime
            fb.soln2 = (a_inv*(-fb.tmem_p - b)) % fb.prime
        yield g

        # Update `b` with Gray code
        for i in range(1, 2**(len(B)-1)):
            v = bit_scan1(i)
            neg_pow = 2*((i >> (v + 1)) % 2) - 1
            b = g.b + 2*neg_pow*B[v]
            a = g.a
            g = SievePolynomial(a, b, N)
            for fb in factor_base:
                if fb.soln1 is None:
                    continue
                fb.soln1 = (fb.soln1 - neg_pow*fb.b_ainv[v]) % fb.prime
                fb.soln2 = (fb.soln2 - neg_pow*fb.b_ainv[v]) % fb.prime
            yield g


def _gen_sieve_array(M, factor_base):
    """Sieve Stage of the Quadratic Sieve. For every prime in the factor_base
    that does not divide the coefficient `a` we add log_p over the sieve_array
    such that ``-M <= soln1 + i*p <=  M`` and ``-M <= soln2 + i*p <=  M`` where `i`
    is an integer. When p = 2 then log_p is only added using
    ``-M <= soln1 + i*p <=  M``.

    Parameters
    ==========

    M : sieve interval
    factor_base : factor_base primes
    """
    sieve_array = [0]*(2*M + 1)
    for factor in factor_base:
        if factor.soln1 is None: #The prime does not divides a
            continue
        for idx in range((M + factor.soln1) % factor.prime, 2*M, factor.prime):
            sieve_array[idx] += factor.log_p
        if factor.prime == 2:
            continue
        #if prime is 2 then sieve only with soln_1_p
        for idx in range((M + factor.soln2) % factor.prime, 2*M, factor.prime):
            sieve_array[idx] += factor.log_p
    return sieve_array


def _check_smoothness(num, factor_base):
    r""" Check if `num` is smooth with respect to the given `factor_base`
    and compute its factorization vector.

    Parameters
    ==========

    num : integer whose smootheness is to be checked
    factor_base : factor_base primes
    """
    if num < 0:
        num *= -1
        vec = 1
    else:
        vec = 0
    for i, fb in enumerate(factor_base, 1):
        if num % fb.prime:
            continue
        e = 1
        num //= fb.prime
        while num % fb.prime == 0:
            e += 1
            num //= fb.prime
        if e % 2:
            vec += 1 << i
    return vec, num


def _trial_division_stage(N, M, factor_base, sieve_array, sieve_poly, partial_relations, ERROR_TERM):
    """Trial division stage. Here we trial divide the values generetated
    by sieve_poly in the sieve interval and if it is a smooth number then
    it is stored in `smooth_relations`. Moreover, if we find two partial relations
    with same large prime then they are combined to form a smooth relation.
    First we iterate over sieve array and look for values which are greater
    than accumulated_val, as these values have a high chance of being smooth
    number. Then using these values we find smooth relations.
    In general, let ``t**2 = u*p modN`` and ``r**2 = v*p modN`` be two partial relations
    with the same large prime p. Then they can be combined ``(t*r/p)**2 = u*v modN``
    to form a smooth relation.

    Parameters
    ==========

    N : Number to be factored
    M : sieve interval
    factor_base : factor_base primes
    sieve_array : stores log_p values
    sieve_poly : polynomial from which we find smooth relations
    partial_relations : stores partial relations with one large prime
    ERROR_TERM : error term for accumulated_val
    """
    accumulated_val = (log(M) + log(N)/2 - ERROR_TERM) * 2**10
    smooth_relations = []
    proper_factor = set()
    partial_relation_upper_bound = 128*factor_base[-1].prime
    for x, val in enumerate(sieve_array, -M):
        if val < accumulated_val:
            continue
        v = sieve_poly.eval_v(x)
        vec, num = _check_smoothness(v, factor_base)
        if num == 1:
            smooth_relations.append((sieve_poly.eval_u(x), v, vec))
        elif num < partial_relation_upper_bound and isprime(num):
            if N % num == 0:
                proper_factor.add(num)
                continue
            u = sieve_poly.eval_u(x)
            if num in partial_relations:
                u_prev, v_prev, vec_prev = partial_relations.pop(num)
                u = u*u_prev*invert(num, N) % N
                v = v*v_prev // num**2
                vec ^= vec_prev
                smooth_relations.append((u, v, vec))
            else:
                partial_relations[num] = (u, v, vec)
    return smooth_relations, proper_factor


def _find_factor(N, smooth_relations, col):
    """ Finds proper factor of N using fast gaussian reduction for modulo 2 matrix.

    Parameters
    ==========

    N : Number to be factored
    smooth_relations : Smooth relations vectors matrix
    col : Number of columns in the matrix

    Reference
    ==========

    .. [1] A fast algorithm for gaussian elimination over GF(2) and
    its implementation on the GAPP. Cetin K.Koc, Sarath N.Arachchige
    """
    matrix = [s_relation[2] for s_relation in smooth_relations]
    row = len(matrix)
    mark = [False] * row
    for pos in range(col):
        m = 1 << pos
        for i in range(row):
            if p := matrix[i] & m:
                add_col = p ^ matrix[i]
                matrix[i] = m
                mark[i] = True
                for j in range(i + 1, row):
                    if matrix[j] & m:
                        matrix[j] ^= add_col
                break

    for m, mat, rel in zip(mark, matrix, smooth_relations):
        if m:
            continue
        u, v = rel[0], rel[1]
        for m1, mat1, rel1 in zip(mark, matrix, smooth_relations):
            if m1 and mat & mat1:
                u *= rel1[0]
                v *= rel1[1]
        # assert is_square(v)
        v = isqrt(v)
        if 1 < (g := gcd(u - v, N)) < N:
            yield g


def qs(N, prime_bound, M, ERROR_TERM=25, seed=1234):
    """Performs factorization using Self-Initializing Quadratic Sieve.
    In SIQS, let N be a number to be factored, and this N should not be a
    perfect power. If we find two integers such that ``X**2 = Y**2 modN`` and
    ``X != +-Y modN``, then `gcd(X + Y, N)` will reveal a proper factor of N.
    In order to find these integers X and Y we try to find relations of form
    t**2 = u modN where u is a product of small primes. If we have enough of
    these relations then we can form ``(t1*t2...ti)**2 = u1*u2...ui modN`` such that
    the right hand side is a square, thus we found a relation of ``X**2 = Y**2 modN``.

    Here, several optimizations are done like using multiple polynomials for
    sieving, fast changing between polynomials and using partial relations.
    The use of partial relations can speeds up the factoring by 2 times.

    Parameters
    ==========

    N : Number to be Factored
    prime_bound : upper bound for primes in the factor base
    M : Sieve Interval
    ERROR_TERM : Error term for checking smoothness
    seed : seed of random number generator

    Returns
    =======

    set(int) : A set of factors of N without considering multiplicity.
               Returns ``{N}`` if factorization fails.

    Examples
    ========

    >>> from sympy.ntheory import qs
    >>> qs(25645121643901801, 2000, 10000)
    {5394769, 4753701529}
    >>> qs(9804659461513846513, 2000, 10000)
    {4641991, 2112166839943}

    See Also
    ========

    qs_factor

    References
    ==========

    .. [1] https://pdfs.semanticscholar.org/5c52/8a975c1405bd35c65993abf5a4edb667c1db.pdf
    .. [2] https://www.rieselprime.de/ziki/Self-initializing_quadratic_sieve
    """
    return set(qs_factor(N, prime_bound, M, ERROR_TERM, seed))


def qs_factor(N, prime_bound, M, ERROR_TERM=25, seed=1234):
    """ Performs factorization using Self-Initializing Quadratic Sieve.

    Parameters
    ==========

    N : Number to be Factored
    prime_bound : upper bound for primes in the factor base
    M : Sieve Interval
    ERROR_TERM : Error term for checking smoothness
    seed : seed of random number generator

    Returns
    =======

    dict[int, int] : Factors of N.
                     Returns ``{N: 1}`` if factorization fails.
                     Note that the key is not always a prime number.

    Examples
    ========

    >>> from sympy.ntheory import qs_factor
    >>> qs_factor(1009 * 100003, 2000, 10000)
    {1009: 1, 100003: 1}

    See Also
    ========

    qs

    """
    if N < 2:
        raise ValueError("N should be greater than 1")
    factors = {}
    smooth_relations = []
    partial_relations = {}
    # Eliminate the possibility of even numbers,
    # prime numbers, and perfect powers.
    if N % 2 == 0:
        e = 1
        N //= 2
        while N % 2 == 0:
            N //= 2
            e += 1
        factors[2] = e
    if isprime(N):
        factors[N] = 1
        return factors
    if result := _perfect_power(N, 3):
        n, e = result
        factors[n] = e
        return factors
    N_copy = N
    randint = _randint(seed)
    idx_1000, idx_5000, factor_base = _generate_factor_base(prime_bound, N)
    threshold = len(factor_base) * 105//100
    for g in _generate_polynomial(N, M, factor_base, idx_1000, idx_5000, randint):
        sieve_array = _gen_sieve_array(M, factor_base)
        s_rel, p_f = _trial_division_stage(N, M, factor_base, sieve_array, g, partial_relations, ERROR_TERM)
        smooth_relations += s_rel
        for p in p_f:
            if N_copy % p:
                continue
            e = 1
            N_copy //= p
            while N_copy % p == 0:
                N_copy //= p
                e += 1
            factors[p] = e
        if threshold <= len(smooth_relations):
            break

    for factor in _find_factor(N, smooth_relations, len(factor_base) + 1):
        if N_copy % factor == 0:
            e = 1
            N_copy //= factor
            while N_copy % factor == 0:
                N_copy //= factor
                e += 1
            factors[factor] = e
            if N_copy == 1 or isprime(N_copy):
                break
    if N_copy != 1:
        factors[N_copy] = 1
    return factors
<|endoftext|>""" Common Exceptions for `holonomic` module. """

class BaseHolonomicError(Exception):

    def new(self, *args):
        raise NotImplementedError("abstract base class")

class NotPowerSeriesError(BaseHolonomicError):

    def __init__(self, holonomic, x0):
        self.holonomic = holonomic
        self.x0 = x0

    def __str__(self):
        s = 'A Power Series does not exists for '
        s += str(self.holonomic)
        s += ' about %s.' %self.x0
        return s

class NotHolonomicError(BaseHolonomicError):

    def __init__(self, m):
        self.m = m

    def __str__(self):
        return self.m

class SingularityError(BaseHolonomicError):

    def __init__(self, holonomic, x0):
        self.holonomic = holonomic
        self.x0 = x0

    def __str__(self):
        s = str(self.holonomic)
        s += ' has a singularity at %s.' %self.x0
        return s

class NotHyperSeriesError(BaseHolonomicError):

    def __init__(self, holonomic, x0):
        self.holonomic = holonomic
        self.x0 = x0

    def __str__(self):
        s = 'Power series expansion of '
        s += str(self.holonomic)
        s += ' about %s is not hypergeometric' %self.x0
        return s
<|endoftext|>"""
This module implements Holonomic Functions and
various operations on them.
"""

from sympy.core import Add, Mul, Pow
from sympy.core.numbers import (NaN, Infinity, NegativeInfinity, Float, I, pi,
        equal_valued, int_valued)
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial, factorial, rf
from sympy.functions.elementary.exponential import exp_polar, exp, log
from sympy.functions.elementary.hyperbolic import (cosh, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, sinc)
from sympy.functions.special.error_functions import (Ci, Shi, Si, erf, erfc, erfi)
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper, meijerg
from sympy.integrals import meijerint
from sympy.matrices import Matrix
from sympy.polys.rings import PolyElement
from sympy.polys.fields import FracElement
from sympy.polys.domains import QQ, RR
from sympy.polys.polyclasses import DMF
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly
from sympy.polys.matrices import DomainMatrix
from sympy.printing import sstr
from sympy.series.limits import limit
from sympy.series.order import Order
from sympy.simplify.hyperexpand import hyperexpand
from sympy.simplify.simplify import nsimplify
from sympy.solvers.solvers import solve

from .recurrence import HolonomicSequence, RecurrenceOperator, RecurrenceOperators
from .holonomicerrors import (NotPowerSeriesError, NotHyperSeriesError,
    SingularityError, NotHolonomicError)


def _find_nonzero_solution(r, homosys):
    ones = lambda shape: DomainMatrix.ones(shape, r.domain)
    particular, nullspace = r._solve(homosys)
    nullity = nullspace.shape[0]
    nullpart = ones((1, nullity)) * nullspace
    sol = (particular + nullpart).transpose()
    return sol



def DifferentialOperators(base, generator):
    r"""
    This function is used to create annihilators using ``Dx``.

    Explanation
    ===========

    Returns an Algebra of Differential Operators also called Weyl Algebra
    and the operator for differentiation i.e. the ``Dx`` operator.

    Parameters
    ==========

    base:
        Base polynomial ring for the algebra.
        The base polynomial ring is the ring of polynomials in :math:`x` that
        will appear as coefficients in the operators.
    generator:
        Generator of the algebra which can
        be either a noncommutative ``Symbol`` or a string. e.g. "Dx" or "D".

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.abc import x
    >>> from sympy.holonomic.holonomic import DifferentialOperators
    >>> R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    >>> R
    Univariate Differential Operator Algebra in intermediate Dx over the base ring ZZ[x]
    >>> Dx*x
    (1) + (x)*Dx
    """

    ring = DifferentialOperatorAlgebra(base, generator)
    return (ring, ring.derivative_operator)


class DifferentialOperatorAlgebra:
    r"""
    An Ore Algebra is a set of noncommutative polynomials in the
    intermediate ``Dx`` and coefficients in a base polynomial ring :math:`A`.
    It follows the commutation rule:

    .. math ::
       Dxa = \sigma(a)Dx + \delta(a)

    for :math:`a \subset A`.

    Where :math:`\sigma: A \Rightarrow A` is an endomorphism and :math:`\delta: A \rightarrow A`
    is a skew-derivation i.e. :math:`\delta(ab) = \delta(a) b + \sigma(a) \delta(b)`.

    If one takes the sigma as identity map and delta as the standard derivation
    then it becomes the algebra of Differential Operators also called
    a Weyl Algebra i.e. an algebra whose elements are Differential Operators.

    This class represents a Weyl Algebra and serves as the parent ring for
    Differential Operators.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy import symbols
    >>> from sympy.holonomic.holonomic import DifferentialOperators
    >>> x = symbols('x')
    >>> R, Dx = DifferentialOperators(ZZ.old_poly_ring(x), 'Dx')
    >>> R
    Univariate Differential Operator Algebra in intermediate Dx over the base ring
    ZZ[x]

    See Also
    ========

    DifferentialOperator
    """

    def __init__(self, base, generator):
        # the base polynomial ring for the algebra
        self.base = base
        # the operator representing differentiation i.e. `Dx`
        self.derivative_operator = DifferentialOperator(
            [base.zero, base.one], self)

        if generator is None:
            self.gen_symbol = Symbol('Dx', commutative=False)
        else:
            if isinstance(generator, str):
                self.gen_symbol = Symbol(generator, commutative=False)
            elif isinstance(generator, Symbol):
                self.gen_symbol = generator

    def __str__(self):
        string = 'Univariate Differential Operator Algebra in intermediate '\
            + sstr(self.gen_symbol) + ' over the base ring ' + \
            (self.base).__str__()

        return string

    __repr__ = __str__

    def __eq__(self, other):
        return self.base == other.base and \
               self.gen_symbol == other.gen_symbol


class DifferentialOperator:
    """
    Differential Operators are elements of Weyl Algebra. The Operators
    are defined by a list of polynomials in the base ring and the
    parent ring of the Operator i.e. the algebra it belongs to.

    Explanation
    ===========

    Takes a list of polynomials for each power of ``Dx`` and the
    parent ring which must be an instance of DifferentialOperatorAlgebra.

    A Differential Operator can be created easily using
    the operator ``Dx``. See examples below.

    Examples
    ========

    >>> from sympy.holonomic.holonomic import DifferentialOperator, DifferentialOperators
    >>> from sympy import ZZ
    >>> from sympy import symbols
    >>> x = symbols('x')
    >>> R, Dx = DifferentialOperators(ZZ.old_poly_ring(x),'Dx')

    >>> DifferentialOperator([0, 1, x**2], R)
    (1)*Dx + (x**2)*Dx**2

    >>> (x*Dx*x + 1 - Dx**2)**2
    (2*x**2 + 2*x + 1) + (4*x**3 + 2*x**2 - 4)*Dx + (x**4 - 6*x - 2)*Dx**2 + (-2*x**2)*Dx**3 + (1)*Dx**4

    See Also
    ========

    DifferentialOperatorAlgebra
    """

    _op_priority = 20

    def __init__(self, list_of_poly, parent):
        """
        Parameters
        ==========

        list_of_poly:
            List of polynomials belonging to the base ring of the algebra.
        parent:
            Parent algebra of the operator.
        """

        # the parent ring for this operator
        # must be an DifferentialOperatorAlgebra object
        self.parent = parent
        base = self.parent.base
        self.x = base.gens[0] if isinstance(base.gens[0], Symbol) else base.gens[0][0]
        # sequence of polynomials in x for each power of Dx
        # the list should not have trailing zeroes
        # represents the operator
        # convert the expressions into ring elements using from_sympy
        for i, j in enumerate(list_of_poly):
            if not isinstance(j, base.dtype):
                list_of_poly[i] = base.from_sympy(sympify(j))
            else:
                list_of_poly[i] = base.from_sympy(base.to_sympy(j))

        self.listofpoly = list_of_poly
        # highest power of `Dx`
        self.order = len(self.listofpoly) - 1

    def __mul__(self, other):
        """
        Multiplies two DifferentialOperator and returns another
        DifferentialOperator instance using the commutation rule
        Dx*a = a*Dx + a'
        """

        listofself = self.listofpoly
        if isinstance(other, DifferentialOperator):
            listofother = other.listofpoly
        elif isinstance(other, self.parent.base.dtype):
            listofother = [other]
        else:
            listofother = [self.parent.base.from_sympy(sympify(other))]

        # multiplies a polynomial `b` with a list of polynomials
        def _mul_dmp_diffop(b, listofother):
            if isinstance(listofother, list):
                return [i * b for i in listofother]
            return [b * listofother]

        sol = _mul_dmp_diffop(listofself[0], listofother)

        # compute Dx^i * b
        def _mul_Dxi_b(b):
            sol1 = [self.parent.base.zero]
            sol2 = []

            if isinstance(b, list):
                for i in b:
                    sol1.append(i)
                    sol2.append(i.diff())
            else:
                sol1.append(self.parent.base.from_sympy(b))
                sol2.append(self.parent.base.from_sympy(b).diff())

            return _add_lists(sol1, sol2)

        for i in range(1, len(listofself)):
            # find Dx^i * b in ith iteration
            listofother = _mul_Dxi_b(listofother)
            # solution = solution + listofself[i] * (Dx^i * b)
            sol = _add_lists(sol, _mul_dmp_diffop(listofself[i], listofother))

        return DifferentialOperator(sol, self.parent)

    def __rmul__(self, other):
        if not isinstance(other, DifferentialOperator):

            if not isinstance(other, self.parent.base.dtype):
                other = (self.parent.base).from_sympy(sympify(other))

            sol = [other * j for j in self.listofpoly]
            return DifferentialOperator(sol, self.parent)

    def __add__(self, other):
        if isinstance(other, DifferentialOperator):

            sol = _add_lists(self.listofpoly, other.listofpoly)
            return DifferentialOperator(sol, self.parent)

        list_self = self.listofpoly
        if not isinstance(other, self.parent.base.dtype):
            list_other = [((self.parent).base).from_sympy(sympify(other))]
        else:
            list_other = [other]
        sol = [list_self[0] + list_other[0]] + list_self[1:]
        return DifferentialOperator(sol, self.parent)

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-1) * self + other

    def __neg__(self):
        return -1 * self

    def __truediv__(self, other):
        return self * (S.One / other)

    def __pow__(self, n):
        if n == 1:
            return self
        result = DifferentialOperator([self.parent.base.one], self.parent)
        if n == 0:
            return result
        # if self is `Dx`
        if self.listofpoly == self.parent.derivative_operator.listofpoly:
            sol = [self.parent.base.zero]*n + [self.parent.base.one]
            return DifferentialOperator(sol, self.parent)
        x = self
        while True:
            if n % 2:
                result *= x
            n >>= 1
            if not n:
                break
            x *= x
        return result

    def __str__(self):
        listofpoly = self.listofpoly
        print_str = ''

        for i, j in enumerate(listofpoly):
            if j == self.parent.base.zero:
                continue

            j = self.parent.base.to_sympy(j)

            if i == 0:
                print_str += '(' + sstr(j) + ')'
                continue

            if print_str:
                print_str += ' + '

            if i == 1:
                print_str += '(' + sstr(j) + ')*%s' %(self.parent.gen_symbol)
                continue

            print_str += '(' + sstr(j) + ')' + '*%s**' %(self.parent.gen_symbol) + sstr(i)

        return print_str

    __repr__ = __str__

    def __eq__(self, other):
        if isinstance(other, DifferentialOperator):
            return self.listofpoly == other.listofpoly and \
                   self.parent == other.parent
        return self.listofpoly[0] == other and \
            all(i is self.parent.base.zero for i in self.listofpoly[1:])

    def is_singular(self, x0):
        """
        Checks if the differential equation is singular at x0.
        """

        base = self.parent.base
        return x0 in roots(base.to_sympy(self.listofpoly[-1]), self.x)


class HolonomicFunction:
    r"""
    A Holonomic Function is a solution to a linear homogeneous ordinary
    differential equation with polynomial coefficients. This differential
    equation can also be represented by an annihilator i.e. a Differential
    Operator ``L`` such that :math:`L.f = 0`. For uniqueness of these functions,
    initial conditions can also be provided along with the annihilator.

    Explanation
    ===========

    Holonomic functions have closure properties and thus forms a ring.
    Given two Holonomic Functions f and g, their sum, product,
    integral and derivative is also a Holonomic Function.

    For ordinary points initial condition should be a vector of values of
    the derivatives i.e. :math:`[y(x_0), y'(x_0), y''(x_0) ... ]`.

    For regular singular points initial conditions can also be provided in this
    format:
    :math:`{s0: [C_0, C_1, ...], s1: [C^1_0, C^1_1, ...], ...}`
    where s0, s1, ... are the roots of indicial equation and vectors
    :math:`[C_0, C_1, ...], [C^0_0, C^0_1, ...], ...` are the corresponding initial
    terms of the associated power series. See Examples below.

    Examples
    ========

    >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
    >>> from sympy import QQ
    >>> from sympy import symbols, S
    >>> x = symbols('x')
    >>> R, Dx = DifferentialOperators(QQ.old_poly_ring(x),'Dx')

    >>> p = HolonomicFunction(Dx - 1, x, 0, [1])  # e^x
    >>> q = HolonomicFunction(Dx**2 + 1, x, 0, [0, 1])  # sin(x)

    >>> p + q  # annihilator of e^x + sin(x)
    HolonomicFunction((-1) + (1)*Dx + (-1)*Dx**2 + (1)*Dx**3, x, 0, [1, 2, 1])

    >>> p * q  # annihilator of e^x * sin(x)
    HolonomicFunction((2) + (-2)*Dx + (1)*Dx**2, x, 0, [0, 1])

    An example of initial conditions for regular singular points,
    the indicial equation has only one root `1/2`.

    >>> HolonomicFunction(-S(1)/2 + x*Dx, x, 0, {S(1)/2: [1]})
    HolonomicFunction((-1/2) + (x)*Dx, x, 0, {1/2: [1]})

    >>> HolonomicFunction(-S(1)/2 + x*Dx, x, 0, {S(1)/2: [1]}).to_expr()
    sqrt(x)

    To plot a Holonomic Function, one can use `.evalf()` for numerical
    computation. Here's an example on `sin(x)**2/x` using numpy and matplotlib.

    >>> import sympy.holonomic # doctest: +SKIP
    >>> from sympy import var, sin # doctest: +SKIP
    >>> import matplotlib.pyplot as plt # doctest: +SKIP
    >>> import numpy as np # doctest: +SKIP
    >>> var("x") # doctest: +SKIP
    >>> r = np.linspace(1, 5, 100) # doctest: +SKIP
    >>> y = sympy.holonomic.expr_to_holonomic(sin(x)**2/x, x0=1).evalf(r) # doctest: +SKIP
    >>> plt.plot(r, y, label="holonomic function") # doctest: +SKIP
    >>> plt.show() # doctest: +SKIP

    """

    _op_priority = 20

    def __init__(self, annihilator, x, x0=0, y0=None):
        """

        Parameters
        ==========

        annihilator:
            Annihilator of the Holonomic Function, represented by a
            `DifferentialOperator` object.
        x:
            Variable of the function.
        x0:
            The point at which initial conditions are stored.
            Generally an integer.
        y0:
            The initial condition. The proper format for the initial condition
            is described in class docstring. To make the function unique,
            length of the vector `y0` should be equal to or greater than the
            order of differential equation.
        """

        # initial condition
        self.y0 = y0
        # the point for initial conditions, default is zero.
        self.x0 = x0
        # differential operator L such that L.f = 0
        self.annihilator = annihilator
        self.x = x

    def __str__(self):
        if self._have_init_cond():
            str_sol = 'HolonomicFunction(%s, %s, %s, %s)' % (str(self.annihilator),\
                sstr(self.x), sstr(self.x0), sstr(self.y0))
        else:
            str_sol = 'HolonomicFunction(%s, %s)' % (str(self.annihilator),\
                sstr(self.x))

        return str_sol

    __repr__ = __str__

    def unify(self, other):
        """
        Unifies the base polynomial ring of a given two Holonomic
        Functions.
        """

        R1 = self.annihilator.parent.base
        R2 = other.annihilator.parent.base

        dom1 = R1.dom
        dom2 = R2.dom

        if R1 == R2:
            return (self, other)

        R = (dom1.unify(dom2)).old_poly_ring(self.x)

        newparent, _ = DifferentialOperators(R, str(self.annihilator.parent.gen_symbol))

        sol1 = [R1.to_sympy(i) for i in self.annihilator.listofpoly]
        sol2 = [R2.to_sympy(i) for i in other.annihilator.listofpoly]

        sol1 = DifferentialOperator(sol1, newparent)
        sol2 = DifferentialOperator(sol2, newparent)

        sol1 = HolonomicFunction(sol1, self.x, self.x0, self.y0)
        sol2 = HolonomicFunction(sol2, other.x, other.x0, other.y0)

        return (sol1, sol2)

    def is_singularics(self):
        """
        Returns True if the function have singular initial condition
        in the dictionary format.

        Returns False if the function have ordinary initial condition
        in the list format.

        Returns None for all other cases.
        """

        if isinstance(self.y0, dict):
            return True
        elif isinstance(self.y0, list):
            return False

    def _have_init_cond(self):
        """
        Checks if the function have initial condition.
        """
        return bool(self.y0)

    def _singularics_to_ord(self):
        """
        Converts a singular initial condition to ordinary if possible.
        """
        a = list(self.y0)[0]
        b = self.y0[a]

        if len(self.y0) == 1 and a == int(a) and a > 0:
            a = int(a)
            y0 = [S.Zero] * a
            y0 += [j * factorial(a + i) for i, j in enumerate(b)]

            return HolonomicFunction(self.annihilator, self.x, self.x0, y0)

    def __add__(self, other):
        # if the ground domains are different
        if self.annihilator.parent.base != other.annihilator.parent.base:
            a, b = self.unify(other)
            return a + b

        deg1 = self.annihilator.order
        deg2 = other.annihilator.order
        dim = max(deg1, deg2)
        R = self.annihilator.parent.base
        K = R.get_field()

        rowsself = [self.annihilator]
        rowsother = [other.annihilator]
        gen = self.annihilator.parent.derivative_operator

        # constructing annihilators up to order dim
        for i in range(dim - deg1):
            diff1 = (gen * rowsself[-1])
            rowsself.append(diff1)

        for i in range(dim - deg2):
            diff2 = (gen * rowsother[-1])
            rowsother.append(diff2)

        row = rowsself + rowsother

        # constructing the matrix of the ansatz
        r = []

        for expr in row:
            p = []
            for i in range(dim + 1):
                if i >= len(expr.listofpoly):
                    p.append(K.zero)
                else:
                    p.append(K.new(expr.listofpoly[i].to_list()))
            r.append(p)

        # solving the linear system using gauss jordan solver
        r = DomainMatrix(r, (len(row), dim+1), K).transpose()
        homosys = DomainMatrix.zeros((dim+1, 1), K)
        sol = _find_nonzero_solution(r, homosys)

        # if a solution is not obtained then increasing the order by 1 in each
        # iteration
        while sol.is_zero_matrix:
            dim += 1

            diff1 = (gen * rowsself[-1])
            rowsself.append(diff1)

            diff2 = (gen * rowsother[-1])
            rowsother.append(diff2)

            row = rowsself + rowsother
            r = []

            for expr in row:
                p = []
                for i in range(dim + 1):
                    if i >= len(expr.listofpoly):
                        p.append(K.zero)
                    else:
                        p.append(K.new(expr.listofpoly[i].to_list()))
                r.append(p)

            # solving the linear system using gauss jordan solver
            r = DomainMatrix(r, (len(row), dim+1), K).transpose()
            homosys = DomainMatrix.zeros((dim+1, 1), K)
            sol = _find_nonzero_solution(r, homosys)

        # taking only the coefficients needed to multiply with `self`
        # can be also be done the other way by taking R.H.S and multiplying with
        # `other`
        sol = sol.flat()[:dim + 1 - deg1]
        sol1 = _normalize(sol, self.annihilator.parent)
        # annihilator of the solution
        sol = sol1 * (self.annihilator)
        sol = _normalize(sol.listofpoly, self.annihilator.parent, negative=False)

        if not (self._have_init_cond() and other._have_init_cond()):
            return HolonomicFunction(sol, self.x)

        # both the functions have ordinary initial conditions
        if self.is_singularics() == False and other.is_singularics() == False:

            # directly add the corresponding value
            if self.x0 == other.x0:
                # try to extended the initial conditions
                # using the annihilator
                y1 = _extend_y0(self, sol.order)
                y2 = _extend_y0(other, sol.order)
                y0 = [a + b for a, b in zip(y1, y2)]
                return HolonomicFunction(sol, self.x, self.x0, y0)

            # change the initial conditions to a same point
            selfat0 = self.annihilator.is_singular(0)
            otherat0 = other.annihilator.is_singular(0)
            if self.x0 == 0 and not selfat0 and not otherat0:
                return self + other.change_ics(0)
            if other.x0 == 0 and not selfat0 and not otherat0:
                return self.change_ics(0) + other

            selfatx0 = self.annihilator.is_singular(self.x0)
            otheratx0 = other.annihilator.is_singular(self.x0)
            if not selfatx0 and not otheratx0:
                return self + other.change_ics(self.x0)
            return self.change_ics(other.x0) + other

        if self.x0 != other.x0:
            return HolonomicFunction(sol, self.x)

        # if the functions have singular_ics
        y1 = None
        y2 = None

        if self.is_singularics() == False and other.is_singularics() == True:
            # convert the ordinary initial condition to singular.
            _y0 = [j / factorial(i) for i, j in enumerate(self.y0)]
            y1 = {S.Zero: _y0}
            y2 = other.y0
        elif self.is_singularics() == True and other.is_singularics() == False:
            _y0 = [j / factorial(i) for i, j in enumerate(other.y0)]
            y1 = self.y0
            y2 = {S.Zero: _y0}
        elif self.is_singularics() == True and other.is_singularics() == True:
            y1 = self.y0
            y2 = other.y0

        # computing singular initial condition for the result
        # taking union of the series terms of both functions
        y0 = {}
        for i in y1:
            # add corresponding initial terms if the power
            # on `x` is same
            if i in y2:
                y0[i] = [a + b for a, b in zip(y1[i], y2[i])]
            else:
                y0[i] = y1[i]
        for i in y2:
            if i not in y1:
                y0[i] = y2[i]
        return HolonomicFunction(sol, self.x, self.x0, y0)

    def integrate(self, limits, initcond=False):
        """
        Integrates the given holonomic function.

        Examples
        ========

        >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
        >>> from sympy import QQ
        >>> from sympy import symbols
        >>> x = symbols('x')
        >>> R, Dx = DifferentialOperators(QQ.old_poly_ring(x),'Dx')
        >>> HolonomicFunction(Dx - 1, x, 0, [1]).integrate((x, 0, x))  # e^x - 1
        HolonomicFunction((-1)*Dx + (1)*Dx**2, x, 0, [0, 1])
        >>> HolonomicFunction(Dx**2 + 1, x, 0, [1, 0]).integrate((x, 0, x))
        HolonomicFunction((1)*Dx + (1)*Dx**3, x, 0, [0, 1, 0])
        """

        # to get the annihilator, just multiply by Dx from right
        D = self.annihilator.parent.derivative_operator

        # if the function have initial conditions of the series format
        if self.is_singularics() == True:

            r = self._singularics_to_ord()
            if r:
                return r.integrate(limits, initcond=initcond)

            # computing singular initial condition for the function
            # produced after integration.
            y0 = {}
            for i in self.y0:
                c = self.y0[i]
                c2 = []
                for j, cj in enumerate(c):
                    if cj == 0:
                        c2.append(S.Zero)

                    # if power on `x` is -1, the integration becomes log(x)
                    # TODO: Implement this case
                    elif i + j + 1 == 0:
                        raise NotImplementedError("logarithmic terms in the series are not supported")
                    else:
                        c2.append(cj / S(i + j + 1))
                y0[i + 1] = c2

            if hasattr(limits, "__iter__"):
                raise NotImplementedError("Definite integration for singular initial conditions")

            return HolonomicFunction(self.annihilator * D, self.x, self.x0, y0)

        # if no initial conditions are available for the function
        if not self._have_init_cond():
            if initcond:
                return HolonomicFunction(self.annihilator * D, self.x, self.x0, [S.Zero])
            return HolonomicFunction(self.annihilator * D, self.x)

        # definite integral
        # initial conditions for the answer will be stored at point `a`,
        # where `a` is the lower limit of the integrand
        if hasattr(limits, "__iter__"):

            if len(limits) == 3 and limits[0] == self.x:
                x0 = self.x0
                a = limits[1]
                b = limits[2]
                definite = True

        else:
            definite = False

        y0 = [S.Zero]
        y0 += self.y0

        indefinite_integral = HolonomicFunction(self.annihilator * D, self.x, self.x0, y0)

        if not definite:
            return indefinite_integral

        # use evalf to get the values at `a`
        if x0 != a:
            try:
                indefinite_expr = indefinite_integral.to_expr()
            except (NotHyperSeriesError, NotPowerSeriesError):
                indefinite_expr = None

            if indefinite_expr:
                lower = indefinite_expr.subs(self.x, a)
                if isinstance(lower, NaN):
                    lower = indefinite_expr.limit(self.x, a)
            else:
                lower = indefinite_integral.evalf(a)

            if b == self.x:
                y0[0] = y0[0] - lower
                return HolonomicFunction(self.annihilator * D, self.x, x0, y0)

            elif S(b).is_Number:
                if indefinite_expr:
                    upper = indefinite_expr.subs(self.x, b)
                    if isinstance(upper, NaN):
                        upper = indefinite_expr.limit(self.x, b)
                else:
                    upper = indefinite_integral.evalf(b)

                return upper - lower


        # if the upper limit is `x`, the answer will be a function
        if b == self.x:
            return HolonomicFunction(self.annihilator * D, self.x, a, y0)

        # if the upper limits is a Number, a numerical value will be returned
        elif S(b).is_Number:
            try:
                s = HolonomicFunction(self.annihilator * D, self.x, a,\
                    y0).to_expr()
                indefinite = s.subs(self.x, b)
                if not isinstance(indefinite, NaN):
                    return indefinite
                else:
                    return s.limit(self.x, b)
            except (NotHyperSeriesError, NotPowerSeriesError):
                return HolonomicFunction(self.annihilator * D, self.x, a, y0).evalf(b)

        return HolonomicFunction(self.annihilator * D, self.x)

    def diff(self, *args, **kwargs):
        r"""
        Differentiation of the given Holonomic function.

        Examples
        ========

        >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
        >>> from sympy import ZZ
        >>> from sympy import symbols
        >>> x = symbols('x')
        >>> R, Dx = DifferentialOperators(ZZ.old_poly_ring(x),'Dx')
        >>> HolonomicFunction(Dx**2 + 1, x, 0, [0, 1]).diff().to_expr()
        cos(x)
        >>> HolonomicFunction(Dx - 2, x, 0, [1]).diff().to_expr()
        2*exp(2*x)

        See Also
        ========

        integrate
        """
        kwargs.setdefault('evaluate', True)
        if args:
            if args[0] != self.x:
                return S.Zero
            elif len(args) == 2:
                sol = self
                for i in range(args[1]):
                    sol = sol.diff(args[0])
                return sol

        ann = self.annihilator

        # if the function is constant.
        if ann.listofpoly[0] == ann.parent.base.zero and ann.order == 1:
            return S.Zero

        # if the coefficient of y in the differential equation is zero.
        # a shifting is done to compute the answer in this case.
        elif ann.listofpoly[0] == ann.parent.base.zero:

            sol = DifferentialOperator(ann.listofpoly[1:], ann.parent)

            if self._have_init_cond():
                # if ordinary initial condition
                if self.is_singularics() == False:
                    return HolonomicFunction(sol, self.x, self.x0, self.y0[1:])
                # TODO: support for singular initial condition
                return HolonomicFunction(sol, self.x)
            else:
                return HolonomicFunction(sol, self.x)

        # the general algorithm
        R = ann.parent.base
        K = R.get_field()

        seq_dmf = [K.new(i.to_list()) for i in ann.listofpoly]

        # -y = a1*y'/a0 + a2*y''/a0 ... + an*y^n/a0
        rhs = [i / seq_dmf[0] for i in seq_dmf[1:]]
        rhs.insert(0, K.zero)

        # differentiate both lhs and rhs
        sol = _derivate_diff_eq(rhs, K)

        # add the term y' in lhs to rhs
        sol = _add_lists(sol, [K.zero, K.one])

        sol = _normalize(sol[1:], self.annihilator.parent, negative=False)

        if not self._have_init_cond() or self.is_singularics() == True:
            return HolonomicFunction(sol, self.x)

        y0 = _extend_y0(self, sol.order + 1)[1:]
        return HolonomicFunction(sol, self.x, self.x0, y0)

    def __eq__(self, other):
        if self.annihilator != other.annihilator or self.x != other.x:
            return False
        if self._have_init_cond() and other._have_init_cond():
            return self.x0 == other.x0 and self.y0 == other.y0
        return True

    def __mul__(self, other):
        ann_self = self.annihilator

        if not isinstance(other, HolonomicFunction):
            other = sympify(other)

            if other.has(self.x):
                raise NotImplementedError(" Can't multiply a HolonomicFunction and expressions/functions.")

            if not self._have_init_cond():
                return self
            y0 = _extend_y0(self, ann_self.order)
            y1 = [(Poly.new(j, self.x) * other).rep for j in y0]
            return HolonomicFunction(ann_self, self.x, self.x0, y1)

        if self.annihilator.parent.base != other.annihilator.parent.base:
            a, b = self.unify(other)
            return a * b

        ann_other = other.annihilator

        a = ann_self.order
        b = ann_other.order

        R = ann_self.parent.base
        K = R.get_field()

        list_self = [K.new(j.to_list()) for j in ann_self.listofpoly]
        list_other = [K.new(j.to_list()) for j in ann_other.listofpoly]

        # will be used to reduce the degree
        self_red = [-list_self[i] / list_self[a] for i in range(a)]

        other_red = [-list_other[i] / list_other[b] for i in range(b)]

        # coeff_mull[i][j] is the coefficient of Dx^i(f).Dx^j(g)
        coeff_mul = [[K.zero for i in range(b + 1)] for j in range(a + 1)]
        coeff_mul[0][0] = K.one

        # making the ansatz
        lin_sys_elements = [[coeff_mul[i][j] for i in range(a) for j in range(b)]]
        lin_sys = DomainMatrix(lin_sys_elements, (1, a*b), K).transpose()

        homo_sys = DomainMatrix.zeros((a*b, 1), K)

        sol = _find_nonzero_solution(lin_sys, homo_sys)

        # until a non trivial solution is found
        while sol.is_zero_matrix:

            # updating the coefficients Dx^i(f).Dx^j(g) for next degree
            for i in range(a - 1, -1, -1):
                for j in range(b - 1, -1, -1):
                    coeff_mul[i][j + 1] += coeff_mul[i][j]
                    coeff_mul[i + 1][j] += coeff_mul[i][j]
                    if isinstance(coeff_mul[i][j], K.dtype):
                        coeff_mul[i][j] = DMFdiff(coeff_mul[i][j], K)
                    else:
                        coeff_mul[i][j] = coeff_mul[i][j].diff(self.x)

            # reduce the terms to lower power using annihilators of f, g
            for i in range(a + 1):
                if coeff_mul[i][b].is_zero:
                    continue
                for j in range(b):
                    coeff_mul[i][j] += other_red[j] * coeff_mul[i][b]
                coeff_mul[i][b] = K.zero

            # not d2 + 1, as that is already covered in previous loop
            for j in range(b):
                if coeff_mul[a][j] == 0:
                    continue
                for i in range(a):
                    coeff_mul[i][j] += self_red[i] * coeff_mul[a][j]
                coeff_mul[a][j] = K.zero

            lin_sys_elements.append([coeff_mul[i][j] for i in range(a) for j in range(b)])
            lin_sys = DomainMatrix(lin_sys_elements, (len(lin_sys_elements), a*b), K).transpose()

            sol = _find_nonzero_solution(lin_sys, homo_sys)

        sol_ann = _normalize(sol.flat(), self.annihilator.parent, negative=False)

        if not (self._have_init_cond() and other._have_init_cond()):
            return HolonomicFunction(sol_ann, self.x)

        if self.is_singularics() == False and other.is_singularics() == False:

            # if both the conditions are at same point
            if self.x0 == other.x0:

                # try to find more initial conditions
                y0_self = _extend_y0(self, sol_ann.order)
                y0_other = _extend_y0(other, sol_ann.order)
                # h(x0) = f(x0) * g(x0)
                y0 = [y0_self[0] * y0_other[0]]

                # coefficient of Dx^j(f)*Dx^i(g) in Dx^i(fg)
                for i in range(1, min(len(y0_self), len(y0_other))):
                    coeff = [[0 for i in range(i + 1)] for j in range(i + 1)]
                    for j in range(i + 1):
                        for k in range(i + 1):
                            if j + k == i:
                                coeff[j][k] = binomial(i, j)

                    sol = 0
                    for j in range(i + 1):
                        for k in range(i + 1):
                            sol += coeff[j][k]* y0_self[j] * y0_other[k]

                    y0.append(sol)

                return HolonomicFunction(sol_ann, self.x, self.x0, y0)

            # if the points are different, consider one
            selfat0 = self.annihilator.is_singular(0)
            otherat0 = other.annihilator.is_singular(0)

            if self.x0 == 0 and not selfat0 and not otherat0:
                return self * other.change_ics(0)
            if other.x0 == 0 and not selfat0 and not otherat0:
                return self.change_ics(0) * other

            selfatx0 = self.annihilator.is_singular(self.x0)
            otheratx0 = other.annihilator.is_singular(self.x0)
            if not selfatx0 and not otheratx0:
                return self * other.change_ics(self.x0)
            return self.change_ics(other.x0) * other

        if self.x0 != other.x0:
            return HolonomicFunction(sol_ann, self.x)

        # if the functions have singular_ics
        y1 = None
        y2 = None

        if self.is_singularics() == False and other.is_singularics() == True:
            _y0 = [j / factorial(i) for i, j in enumerate(self.y0)]
            y1 = {S.Zero: _y0}
            y2 = other.y0
        elif self.is_singularics() == True and other.is_singularics() == False:
            _y0 = [j / factorial(i) for i, j in enumerate(other.y0)]
            y1 = self.y0
            y2 = {S.Zero: _y0}
        elif self.is_singularics() == True and other.is_singularics() == True:
            y1 = self.y0
            y2 = other.y0

        y0 = {}
        # multiply every possible pair of the series terms
        for i in y1:
            for j in y2:
                k = min(len(y1[i]), len(y2[j]))
                c = [sum((y1[i][b] * y2[j][a - b] for b in range(a + 1)),
                         start=S.Zero) for a in range(k)]
                if not i + j in y0:
                    y0[i + j] = c
                else:
                    y0[i + j] = [a + b for a, b in zip(c, y0[i + j])]
        return HolonomicFunction(sol_ann, self.x, self.x0, y0)

    __rmul__ = __mul__

    def __sub__(self, other):
        return self + other * -1

    def __rsub__(self, other):
        return self * -1 + other

    def __neg__(self):
        return -1 * self

    def __truediv__(self, other):
        return self * (S.One / other)

    def __pow__(self, n):
        if self.annihilator.order <= 1:
            ann = self.annihilator
            parent = ann.parent

            if self.y0 is None:
                y0 = None
            else:
                y0 = [list(self.y0)[0] ** n]

            p0 = ann.listofpoly[0]
            p1 = ann.listofpoly[1]

            p0 = (Poly.new(p0, self.x) * n).rep

            sol = [parent.base.to_sympy(i) for i in [p0, p1]]
            dd = DifferentialOperator(sol, parent)
            return HolonomicFunction(dd, self.x, self.x0, y0)
        if n < 0:
            raise NotHolonomicError("Negative Power on a Holonomic Function")
        Dx = self.annihilator.parent.derivative_operator
        result = HolonomicFunction(Dx, self.x, S.Zero, [S.One])
        if n == 0:
            return result
        x = self
        while True:
            if n % 2:
                result *= x
            n >>= 1
            if not n:
                break
            x *= x
        return result

    def degree(self):
        """
        Returns the highest power of `x` in the annihilator.
        """
        return max(i.degree() for i in self.annihilator.listofpoly)

    def composition(self, expr, *args, **kwargs):
        """
        Returns function after composition of a holonomic
        function with an algebraic function. The method cannot compute
        initial conditions for the result by itself, so they can be also be
        provided.

        Examples
        ========

        >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
        >>> from sympy import QQ
        >>> from sympy import symbols
        >>> x = symbols('x')
        >>> R, Dx = DifferentialOperators(QQ.old_poly_ring(x),'Dx')
        >>> HolonomicFunction(Dx - 1, x).composition(x**2, 0, [1])  # e^(x**2)
        HolonomicFunction((-2*x) + (1)*Dx, x, 0, [1])
        >>> HolonomicFunction(Dx**2 + 1, x).composition(x**2 - 1, 1, [1, 0])
        HolonomicFunction((4*x**3) + (-1)*Dx + (x)*Dx**2, x, 1, [1, 0])

        See Also
        ========

        from_hyper
        """

        R = self.annihilator.parent
        a = self.annihilator.order
        diff = expr.diff(self.x)
        listofpoly = self.annihilator.listofpoly

        for i, j in enumerate(listofpoly):
            if isinstance(j, self.annihilator.parent.base.dtype):
                listofpoly[i] = self.annihilator.parent.base.to_sympy(j)

        r = listofpoly[a].subs({self.x:expr})
        subs = [-listofpoly[i].subs({self.x:expr}) / r for i in range (a)]
        coeffs = [S.Zero for i in range(a)]  # coeffs[i] == coeff of (D^i f)(a) in D^k (f(a))
        coeffs[0] = S.One
        system = [coeffs]
        homogeneous = Matrix([[S.Zero for i in range(a)]]).transpose()
        while True:
            coeffs_next = [p.diff(self.x) for p in coeffs]
            for i in range(a - 1):
                coeffs_next[i + 1] += (coeffs[i] * diff)
            for i in range(a):
                coeffs_next[i] += (coeffs[-1] * subs[i] * diff)
            coeffs = coeffs_next
            # check for linear relations
            system.append(coeffs)
            sol, taus = (Matrix(system).transpose()
                ).gauss_jordan_solve(homogeneous)
            if sol.is_zero_matrix is not True:
                break

        tau = list(taus)[0]
        sol = sol.subs(tau, 1)
        sol = _normalize(sol[0:], R, negative=False)

        # if initial conditions are given for the resulting function
        if args:
            return HolonomicFunction(sol, self.x, args[0], args[1])
        return HolonomicFunction(sol, self.x)

    def to_sequence(self, lb=True):
        r"""
        Finds recurrence relation for the coefficients in the series expansion
        of the function about :math:`x_0`, where :math:`x_0` is the point at
        which the initial condition is stored.

        Explanation
        ===========

        If the point :math:`x_0` is ordinary, solution of the form :math:`[(R, n_0)]`
        is returned. Where :math:`R` is the recurrence relation and :math:`n_0` is the
        smallest ``n`` for which the recurrence holds true.

        If the point :math:`x_0` is regular singular, a list of solutions in
        the format :math:`(R, p, n_0)` is returned, i.e. `[(R, p, n_0), ... ]`.
        Each tuple in this vector represents a recurrence relation :math:`R`
        associated with a root of the indicial equation ``p``. Conditions of
        a different format can also be provided in this case, see the
        docstring of HolonomicFunction class.

        If it's not possible to numerically compute a initial condition,
        it is returned as a symbol :math:`C_j`, denoting the coefficient of
        :math:`(x - x_0)^j` in the power series about :math:`x_0`.

        Examples
        ========

        >>> from sympy.holonomic.holonomic import HolonomicFunction, DifferentialOperators
        >>> from sympy import QQ
        >>> from sympy import symbols, S
        >>> x = symbols('x')
        >>> R, Dx = DifferentialOperators(QQ.old_poly_ring(x),'Dx')
        >>> HolonomicFunction(Dx - 1, x, 0, [1]).to_sequence()
        [(HolonomicSequence((-1) + (n + 1)Sn, n), u(0) = 1, 0)]
        >>> HolonomicFunction((1 + x)*Dx**2 + Dx, x, 0, [0, 1]).to_sequence()
        [(HolonomicSequence((n**2) + (n**2 + n)Sn, n), u(0) = 0, u(1) = 1, u(2) = -1/2, 2)]
        >>> HolonomicFunction(-S(1)/2 + x*Dx, x, 0, {S(1)/2: [1]}).to_sequence()
        [(HolonomicSequence((n), n), u(0) = 1, 1/2, 1)]

        See Also
        ========

        HolonomicFunction.series

        References
        ==========

        .. [1] https://hal.inria.fr/inria-00070025/document
        .. [2] https://www3.risc.jku.at/publications/download/risc_2244/DIPLFORM.pdf

        """

        if self.x0 != 0:
            return self.shift_x(self.x0).to_sequence()

        # check whether a power series exists if the point is singular
        if self.annihilator.is_singular(self.x0):
            return self._frobenius(lb=lb)

        dict1 = {}
        n = Symbol('n', integer=True)
        dom = self.annihilator.parent.base.dom
        R, _ = RecurrenceOperators(dom.old_poly_ring(n), 'Sn')

        # substituting each term of the form `x^k Dx^j` in the
        # annihilator, according to the formula below:
        # x^k Dx^j = Sum(rf(n + 1 - k, j) * a(n + j - k) * x^n, (n, k, oo))
        # for explanation see [2].
        for i, j in enumerate(self.annihilator.listofpoly):

            listofdmp = j.all_coeffs()
            degree = len(listofdmp) - 1

            for k in range(degree + 1):
                coeff = listofdmp[degree - k]

                if coeff == 0:
                    continue

                if (i - k, k) in dict1:
                    dict1[(i - k, k)] += (dom.to_sympy(coeff) * rf(n - k + 1, i))
                else:
                    dict1[(i - k, k)] = (dom.to_sympy(coeff) * rf(n - k + 1, i))


        sol = []
        keylist = [i[0] for i in dict1]
        lower = min(keylist)
        upper = max(keylist)
        degree = self.degree()

        # the recurrence relation holds for all values of
        # n greater than smallest_n, i.e. n >= smallest_n
        smallest_n = lower + degree
        dummys = {}
        eqs = []
        unknowns = []

        # an appropriate shift of the recurrence
        for j in range(lower, upper + 1):
            if j in keylist:
                temp = sum((v.subs(n, n - lower)
                           for k, v in dict1.items() if k[0] == j),
                           start=S.Zero)
                sol.append(temp)
            else:
                sol.append(S.Zero)

        # the recurrence relation
        sol = RecurrenceOperator(sol, R)

        # computing the initial conditions for recurrence
        order = sol.order
        all_roots = roots(R.base.to_sympy(sol.listofpoly[-1]), n, filter='Z')
        all_roots = all_roots.keys()

        if all_roots:
            max_root = max(all_roots) + 1
            smallest_n = max(max_root, smallest_n)
        order += smallest_n

        y0 = _extend_y0(self, order)
        # u(n) = y^n(0)/factorial(n)
        u0 = [j / factorial(i) for i, j in enumerate(y0)]

        # if sufficient conditions can't be computed then
        # try to use the series method i.e.
        # equate the coefficients of x^k in the equation formed by
        # substituting the series in differential equation, to zero.
        if len(u0) < order:

            for i in range(degree):
                eq = S.Zero

                for j in dict1:

                    if i + j[0] < 0:
                        dummys[i + j[0]] = S.Zero

                    elif i + j[0] < len(u0):
                        dummys[i + j[0]] = u0[i + j[0]]

                    elif not i + j[0] in dummys:
                        dummys[i + j[0]] = Symbol('C_%s' %(i + j[0]))
                        unknowns.append(dummys[i + j[0]])

                    if j[1] <= i:
                        eq += dict1[j].subs(n, i) * dummys[i + j[0]]

                eqs.append(eq)

            # solve the system of equations formed
            soleqs = solve(eqs, *unknowns)

            if isinstance(soleqs, dict):

                for i in range(len(u0), order):

                    if i not in dummys:
                        dummys[i] = Symbol('C_%s' %i)

                    if dummys[i] in soleqs:
                        u0.append(soleqs[dummys[i]])

                    else:
                        u0.append(dummys[i])

                if lb:
                    return [(HolonomicSequence(sol, u0), smallest_n)]
                return [HolonomicSequence(sol, u0)]

            for i in range(len(u0), order):

                if i not in dummys:
                    dummys[i] = Symbol('C_%s' %i)

                s = False
                for j in soleqs:
                    if dummys[i] in j:
                        u0.append(j[dummys[i]])
                        s = True
                if not s:
                    u0.append(dummys[i])

        if lb:
            return [(HolonomicSequence(sol, u0), smallest_n)]

        return [HolonomicSequence(sol, u0)]

    def _frobenius(self, lb=True):
        # compute the roots of indicial equation
        indicialroots = self._indicial()

        reals = []
        compl = []
        for i in ordered(indicialroots.keys()):
            if i.is_real:
                reals.extend([i] * indicialroots[i])
            else:
                a, b = i.as_real_imag()
                compl.extend([(i, a, b)] * indicialroots[i])

        # sort the roots for a fixed ordering of solution
        compl.sort(key=lambda x : x[1])
        compl.sort(key=lambda x : x[2])
        reals.sort()

        # grouping the roots, roots differ by an integer are put in the same group.
