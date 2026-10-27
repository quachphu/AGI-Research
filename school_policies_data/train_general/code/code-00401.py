    integers, then return a parametric representation for ``x`` and ``y``
    else return (None, None).

    Here ``x`` and ``y`` are functions of ``t``.
    """
    from sympy.simplify.simplify import clear_coefficients

    if x.is_number and not x.is_Integer:
        return DiophantineSolutionSet([x, y], parameters=params)

    if y.is_number and not y.is_Integer:
        return DiophantineSolutionSet([x, y], parameters=params)

    m, n = symbols("m, n", integer=True)
    c, p = (m*x + n*y).as_content_primitive()
    if a % c.q:
        return DiophantineSolutionSet([x, y], parameters=params)

    # clear_coefficients(mx + b, R)[1] -> (R - b)/m
    eq = clear_coefficients(x, m)[1] - clear_coefficients(y, n)[1]
    junk, eq = eq.as_content_primitive()

    return _diop_solve(eq, params=params)


def diop_ternary_quadratic(eq, parameterize=False):
    """
    Solves the general quadratic ternary form,
    `ax^2 + by^2 + cz^2 + fxy + gyz + hxz = 0`.

    Returns a tuple `(x, y, z)` which is a base solution for the above
    equation. If there are no solutions, `(None, None, None)` is returned.

    Usage
    =====

    ``diop_ternary_quadratic(eq)``: Return a tuple containing a basic solution
    to ``eq``.

    Details
    =======

    ``eq`` should be an homogeneous expression of degree two in three variables
    and it is assumed to be zero.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.solvers.diophantine.diophantine import diop_ternary_quadratic
    >>> diop_ternary_quadratic(x**2 + 3*y**2 - z**2)
    (1, 0, 1)
    >>> diop_ternary_quadratic(4*x**2 + 5*y**2 - z**2)
    (1, 0, 2)
    >>> diop_ternary_quadratic(45*x**2 - 7*y**2 - 8*x*y - z**2)
    (28, 45, 105)
    >>> diop_ternary_quadratic(x**2 - 49*y**2 - z**2 + 13*z*y -8*x*y)
    (9, 1, 5)
    """
    var, coeff, diop_type = classify_diop(eq, _dict=False)

    if diop_type in (
            HomogeneousTernaryQuadratic.name,
            HomogeneousTernaryQuadraticNormal.name):
        sol = _diop_ternary_quadratic(var, coeff)
        if len(sol) > 0:
            x_0, y_0, z_0 = list(sol)[0]
        else:
            x_0, y_0, z_0 = None, None, None

        if parameterize:
            return _parametrize_ternary_quadratic(
                (x_0, y_0, z_0), var, coeff)
        return x_0, y_0, z_0


def _diop_ternary_quadratic(_var, coeff):
    eq = sum(i*coeff[i] for i in coeff)
    if HomogeneousTernaryQuadratic(eq).matches():
        return HomogeneousTernaryQuadratic(eq, free_symbols=_var).solve()
    elif HomogeneousTernaryQuadraticNormal(eq).matches():
        return HomogeneousTernaryQuadraticNormal(eq, free_symbols=_var).solve()


def transformation_to_normal(eq):
    """
    Returns the transformation Matrix that converts a general ternary
    quadratic equation ``eq`` (`ax^2 + by^2 + cz^2 + dxy + eyz + fxz`)
    to a form without cross terms: `ax^2 + by^2 + cz^2 = 0`. This is
    not used in solving ternary quadratics; it is only implemented for
    the sake of completeness.
    """
    var, coeff, diop_type = classify_diop(eq, _dict=False)

    if diop_type in (
            "homogeneous_ternary_quadratic",
            "homogeneous_ternary_quadratic_normal"):
        return _transformation_to_normal(var, coeff)


def _transformation_to_normal(var, coeff):

    _var = list(var)  # copy
    x, y, z = var

    if not any(coeff[i**2] for i in var):
        # https://math.stackexchange.com/questions/448051/transform-quadratic-ternary-form-to-normal-form/448065#448065
        a = coeff[x*y]
        b = coeff[y*z]
        c = coeff[x*z]
        swap = False
        if not a:  # b can't be 0 or else there aren't 3 vars
            swap = True
            a, b = b, a
        T = Matrix(((1, 1, -b/a), (1, -1, -c/a), (0, 0, 1)))
        if swap:
            T.row_swap(0, 1)
            T.col_swap(0, 1)
        return T

    if coeff[x**2] == 0:
        # If the coefficient of x is zero change the variables
        if coeff[y**2] == 0:
            _var[0], _var[2] = var[2], var[0]
            T = _transformation_to_normal(_var, coeff)
            T.row_swap(0, 2)
            T.col_swap(0, 2)
            return T

        _var[0], _var[1] = var[1], var[0]
        T = _transformation_to_normal(_var, coeff)
        T.row_swap(0, 1)
        T.col_swap(0, 1)
        return T

    # Apply the transformation x --> X - (B*Y + C*Z)/(2*A)
    if coeff[x*y] != 0 or coeff[x*z] != 0:
        A = coeff[x**2]
        B = coeff[x*y]
        C = coeff[x*z]
        D = coeff[y**2]
        E = coeff[y*z]
        F = coeff[z**2]

        _coeff = {}

        _coeff[x**2] = 4*A**2
        _coeff[y**2] = 4*A*D - B**2
        _coeff[z**2] = 4*A*F - C**2
        _coeff[y*z] = 4*A*E - 2*B*C
        _coeff[x*y] = 0
        _coeff[x*z] = 0

        T_0 = _transformation_to_normal(_var, _coeff)
        return Matrix(3, 3, [1, S(-B)/(2*A), S(-C)/(2*A), 0, 1, 0, 0, 0, 1])*T_0

    elif coeff[y*z] != 0:
        if coeff[y**2] == 0:
            if coeff[z**2] == 0:
                # Equations of the form A*x**2 + E*yz = 0.
                # Apply transformation y -> Y + Z ans z -> Y - Z
                return Matrix(3, 3, [1, 0, 0, 0, 1, 1, 0, 1, -1])

            # Ax**2 + E*y*z + F*z**2  = 0
            _var[0], _var[2] = var[2], var[0]
            T = _transformation_to_normal(_var, coeff)
            T.row_swap(0, 2)
            T.col_swap(0, 2)
            return T

        # A*x**2 + D*y**2 + E*y*z + F*z**2 = 0, F may be zero
        _var[0], _var[1] = var[1], var[0]
        T = _transformation_to_normal(_var, coeff)
        T.row_swap(0, 1)
        T.col_swap(0, 1)
        return T

    return Matrix.eye(3)


def parametrize_ternary_quadratic(eq):
    """
    Returns the parametrized general solution for the ternary quadratic
    equation ``eq`` which has the form
    `ax^2 + by^2 + cz^2 + fxy + gyz + hxz = 0`.

    Examples
    ========

    >>> from sympy import Tuple, ordered
    >>> from sympy.abc import x, y, z
    >>> from sympy.solvers.diophantine.diophantine import parametrize_ternary_quadratic

    The parametrized solution may be returned with three parameters:

    >>> parametrize_ternary_quadratic(2*x**2 + y**2 - 2*z**2)
    (p**2 - 2*q**2, -2*p**2 + 4*p*q - 4*p*r - 4*q**2, p**2 - 4*p*q + 2*q**2 - 4*q*r)

    There might also be only two parameters:

    >>> parametrize_ternary_quadratic(4*x**2 + 2*y**2 - 3*z**2)
    (2*p**2 - 3*q**2, -4*p**2 + 12*p*q - 6*q**2, 4*p**2 - 8*p*q + 6*q**2)

    Notes
    =====

    Consider ``p`` and ``q`` in the previous 2-parameter
    solution and observe that more than one solution can be represented
    by a given pair of parameters. If `p` and ``q`` are not coprime, this is
    trivially true since the common factor will also be a common factor of the
    solution values. But it may also be true even when ``p`` and
    ``q`` are coprime:

    >>> sol = Tuple(*_)
    >>> p, q = ordered(sol.free_symbols)
    >>> sol.subs([(p, 3), (q, 2)])
    (6, 12, 12)
    >>> sol.subs([(q, 1), (p, 1)])
    (-1, 2, 2)
    >>> sol.subs([(q, 0), (p, 1)])
    (2, -4, 4)
    >>> sol.subs([(q, 1), (p, 0)])
    (-3, -6, 6)

    Except for sign and a common factor, these are equivalent to
    the solution of (1, 2, 2).

    References
    ==========

    .. [1] The algorithmic resolution of Diophantine equations, Nigel P. Smart,
           London Mathematical Society Student Texts 41, Cambridge University
           Press, Cambridge, 1998.

    """
    var, coeff, diop_type = classify_diop(eq, _dict=False)

    if diop_type in (
            "homogeneous_ternary_quadratic",
            "homogeneous_ternary_quadratic_normal"):
        x_0, y_0, z_0 = list(_diop_ternary_quadratic(var, coeff))[0]
        return _parametrize_ternary_quadratic(
            (x_0, y_0, z_0), var, coeff)


def _parametrize_ternary_quadratic(solution, _var, coeff):
    # called for a*x**2 + b*y**2 + c*z**2 + d*x*y + e*y*z + f*x*z = 0
    assert 1 not in coeff

    x_0, y_0, z_0 = solution

    v = list(_var)  # copy

    if x_0 is None:
        return (None, None, None)

    if solution.count(0) >= 2:
        # if there are 2 zeros the equation reduces
        # to k*X**2 == 0 where X is x, y, or z so X must
        # be zero, too. So there is only the trivial
        # solution.
        return (None, None, None)

    if x_0 == 0:
        v[0], v[1] = v[1], v[0]
        y_p, x_p, z_p = _parametrize_ternary_quadratic(
            (y_0, x_0, z_0), v, coeff)
        return x_p, y_p, z_p

    x, y, z = v
    r, p, q = symbols("r, p, q", integer=True)

    eq = sum(k*v for k, v in coeff.items())
    eq_1 = _mexpand(eq.subs(zip(
        (x, y, z), (r*x_0, r*y_0 + p, r*z_0 + q))))
    A, B = eq_1.as_independent(r, as_Add=True)


    x = A*x_0
    y = (A*y_0 - _mexpand(B/r*p))
    z = (A*z_0 - _mexpand(B/r*q))

    return _remove_gcd(x, y, z)


def diop_ternary_quadratic_normal(eq, parameterize=False):
    """
    Solves the quadratic ternary diophantine equation,
    `ax^2 + by^2 + cz^2 = 0`.

    Explanation
    ===========

    Here the coefficients `a`, `b`, and `c` should be non zero. Otherwise the
    equation will be a quadratic binary or univariate equation. If solvable,
    returns a tuple `(x, y, z)` that satisfies the given equation. If the
    equation does not have integer solutions, `(None, None, None)` is returned.

    Usage
    =====

    ``diop_ternary_quadratic_normal(eq)``: where ``eq`` is an equation of the form
    `ax^2 + by^2 + cz^2 = 0`.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.solvers.diophantine.diophantine import diop_ternary_quadratic_normal
    >>> diop_ternary_quadratic_normal(x**2 + 3*y**2 - z**2)
    (1, 0, 1)
    >>> diop_ternary_quadratic_normal(4*x**2 + 5*y**2 - z**2)
    (1, 0, 2)
    >>> diop_ternary_quadratic_normal(34*x**2 - 3*y**2 - 301*z**2)
    (4, 9, 1)
    """
    var, coeff, diop_type = classify_diop(eq, _dict=False)
    if diop_type == HomogeneousTernaryQuadraticNormal.name:
        sol = _diop_ternary_quadratic_normal(var, coeff)
        if len(sol) > 0:
            x_0, y_0, z_0 = list(sol)[0]
        else:
            x_0, y_0, z_0 = None, None, None
        if parameterize:
            return _parametrize_ternary_quadratic(
                (x_0, y_0, z_0), var, coeff)
        return x_0, y_0, z_0


def _diop_ternary_quadratic_normal(var, coeff):
    eq = sum(i * coeff[i] for i in coeff)
    return HomogeneousTernaryQuadraticNormal(eq, free_symbols=var).solve()


def sqf_normal(a, b, c, steps=False):
    """
    Return `a', b', c'`, the coefficients of the square-free normal
    form of `ax^2 + by^2 + cz^2 = 0`, where `a', b', c'` are pairwise
    prime.  If `steps` is True then also return three tuples:
    `sq`, `sqf`, and `(a', b', c')` where `sq` contains the square
    factors of `a`, `b` and `c` after removing the `gcd(a, b, c)`;
    `sqf` contains the values of `a`, `b` and `c` after removing
    both the `gcd(a, b, c)` and the square factors.

    The solutions for `ax^2 + by^2 + cz^2 = 0` can be
    recovered from the solutions of `a'x^2 + b'y^2 + c'z^2 = 0`.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import sqf_normal
    >>> sqf_normal(2 * 3**2 * 5, 2 * 5 * 11, 2 * 7**2 * 11)
    (11, 1, 5)
    >>> sqf_normal(2 * 3**2 * 5, 2 * 5 * 11, 2 * 7**2 * 11, True)
    ((3, 1, 7), (5, 55, 11), (11, 1, 5))

    References
    ==========

    .. [1] Legendre's Theorem, Legrange's Descent,
           https://public.csusm.edu/aitken_html/notes/legendre.pdf


    See Also
    ========

    reconstruct()
    """
    ABC = _remove_gcd(a, b, c)
    sq = tuple(square_factor(i) for i in ABC)
    sqf = A, B, C = tuple([i//j**2 for i,j in zip(ABC, sq)])
    pc = igcd(A, B)
    A /= pc
    B /= pc
    pa = igcd(B, C)
    B /= pa
    C /= pa
    pb = igcd(A, C)
    A /= pb
    B /= pb

    A *= pa
    B *= pb
    C *= pc

    if steps:
        return (sq, sqf, (A, B, C))
    else:
        return A, B, C


def square_factor(a):
    r"""
    Returns an integer `c` s.t. `a = c^2k, \ c,k \in Z`. Here `k` is square
    free. `a` can be given as an integer or a dictionary of factors.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import square_factor
    >>> square_factor(24)
    2
    >>> square_factor(-36*3)
    6
    >>> square_factor(1)
    1
    >>> square_factor({3: 2, 2: 1, -1: 1})  # -18
    3

    See Also
    ========
    sympy.ntheory.factor_.core
    """
    f = a if isinstance(a, dict) else factorint(a)
    return Mul(*[p**(e//2) for p, e in f.items()])


def reconstruct(A, B, z):
    """
    Reconstruct the `z` value of an equivalent solution of `ax^2 + by^2 + cz^2`
    from the `z` value of a solution of the square-free normal form of the
    equation, `a'*x^2 + b'*y^2 + c'*z^2`, where `a'`, `b'` and `c'` are square
    free and `gcd(a', b', c') == 1`.
    """
    f = factorint(igcd(A, B))
    for p, e in f.items():
        if e != 1:
            raise ValueError('a and b should be square-free')
        z *= p
    return z


def ldescent(A, B):
    """
    Return a non-trivial solution to `w^2 = Ax^2 + By^2` using
    Lagrange's method; return None if there is no such solution.

    Parameters
    ==========

    A : Integer
    B : Integer
        non-zero integer

    Returns
    =======

    (int, int, int) | None : a tuple `(w_0, x_0, y_0)` which is a solution to the above equation.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import ldescent
    >>> ldescent(1, 1) # w^2 = x^2 + y^2
    (1, 1, 0)
    >>> ldescent(4, -7) # w^2 = 4x^2 - 7y^2
    (2, -1, 0)

    This means that `x = -1, y = 0` and `w = 2` is a solution to the equation
    `w^2 = 4x^2 - 7y^2`

    >>> ldescent(5, -1) # w^2 = 5x^2 - y^2
    (2, 1, -1)

    References
    ==========

    .. [1] The algorithmic resolution of Diophantine equations, Nigel P. Smart,
           London Mathematical Society Student Texts 41, Cambridge University
           Press, Cambridge, 1998.
    .. [2] Cremona, J. E., Rusin, D. (2003). Efficient Solution of Rational Conics.
           Mathematics of Computation, 72(243), 1417-1441.
           https://doi.org/10.1090/S0025-5718-02-01480-1
    """
    if A == 0 or B == 0:
        raise ValueError("A and B must be non-zero integers")
    if abs(A) > abs(B):
        w, y, x = ldescent(B, A)
        return w, x, y
    if A == 1:
        return (1, 1, 0)
    if B == 1:
        return (1, 0, 1)
    if B == -1:  # and A == -1
        return

    r = sqrt_mod(A, B)
    if r is None:
        return
    Q = (r**2 - A) // B
    if Q == 0:
        return r, -1, 0
    for i in divisors(Q):
        d, _exact = integer_nthroot(abs(Q) // i, 2)
        if _exact:
            B_0 = sign(Q)*i
            W, X, Y = ldescent(A, B_0)
            return _remove_gcd(-A*X + r*W, r*X - W, Y*B_0*d)


def descent(A, B):
    """
    Returns a non-trivial solution, (x, y, z), to `x^2 = Ay^2 + Bz^2`
    using Lagrange's descent method with lattice-reduction. `A` and `B`
    are assumed to be valid for such a solution to exist.

    This is faster than the normal Lagrange's descent algorithm because
    the Gaussian reduction is used.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import descent
    >>> descent(3, 1) # x**2 = 3*y**2 + z**2
    (1, 0, 1)

    `(x, y, z) = (1, 0, 1)` is a solution to the above equation.

    >>> descent(41, -113)
    (-16, -3, 1)

    References
    ==========

    .. [1] Cremona, J. E., Rusin, D. (2003). Efficient Solution of Rational Conics.
           Mathematics of Computation, 72(243), 1417-1441.
           https://doi.org/10.1090/S0025-5718-02-01480-1
    """
    if abs(A) > abs(B):
        x, y, z = descent(B, A)
        return x, z, y

    if B == 1:
        return (1, 0, 1)
    if A == 1:
        return (1, 1, 0)
    if B == -A:
        return (0, 1, 1)
    if B == A:
        x, z, y = descent(-1, A)
        return (A*y, z, x)

    w = sqrt_mod(A, B)
    x_0, z_0 = gaussian_reduce(w, A, B)

    t = (x_0**2 - A*z_0**2) // B
    t_2 = square_factor(t)
    t_1 = t // t_2**2

    x_1, z_1, y_1 = descent(A, t_1)

    return _remove_gcd(x_0*x_1 + A*z_0*z_1, z_0*x_1 + x_0*z_1, t_1*t_2*y_1)


def gaussian_reduce(w:int, a:int, b:int) -> tuple[int, int]:
    r"""
    Returns a reduced solution `(x, z)` to the congruence
    `X^2 - aZ^2 \equiv 0 \pmod{b}` so that `x^2 + |a|z^2` is as small as possible.
    Here ``w`` is a solution of the congruence `x^2 \equiv a \pmod{b}`.

    This function is intended to be used only for ``descent()``.

    Explanation
    ===========

    The Gaussian reduction can find the shortest vector for any norm.
    So we define the special norm for the vectors `u = (u_1, u_2)` and `v = (v_1, v_2)` as follows.

    .. math ::
        u \cdot v := (wu_1 + bu_2)(wv_1 + bv_2) + |a|u_1v_1

    Note that, given the mapping `f: (u_1, u_2) \to (wu_1 + bu_2, u_1)`,
    `f((u_1,u_2))` is the solution to `X^2 - aZ^2 \equiv 0 \pmod{b}`.
    In other words, finding the shortest vector in this norm will yield a solution with smaller `X^2 + |a|Z^2`.
    The algorithm starts from basis vectors `(0, 1)` and `(1, 0)`
    (corresponding to solutions `(b, 0)` and `(w, 1)`, respectively) and finds the shortest vector.
    The shortest vector does not necessarily correspond to the smallest solution,
    but since ``descent()`` only wants the smallest possible solution, it is sufficient.

    Parameters
    ==========

    w : int
        ``w`` s.t. `w^2 \equiv a \pmod{b}`
    a : int
        square-free nonzero integer
    b : int
        square-free nonzero integer

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import gaussian_reduce
    >>> from sympy.ntheory.residue_ntheory import sqrt_mod
    >>> a, b = 19, 101
    >>> gaussian_reduce(sqrt_mod(a, b), a, b) # 1**2 - 19*(-4)**2 = -303
    (1, -4)
    >>> a, b = 11, 14
    >>> x, z = gaussian_reduce(sqrt_mod(a, b), a, b)
    >>> (x**2 - a*z**2) % b == 0
    True

    It does not always return the smallest solution.

    >>> a, b = 6, 95
    >>> min_x, min_z = 1, 4
    >>> x, z = gaussian_reduce(sqrt_mod(a, b), a, b)
    >>> (x**2 - a*z**2) % b == 0 and (min_x**2 - a*min_z**2) % b == 0
    True
    >>> min_x**2 + abs(a)*min_z**2 < x**2 + abs(a)*z**2
    True

    References
    ==========

    .. [1] Gaussian lattice Reduction [online]. Available:
           https://web.archive.org/web/20201021115213/http://home.ie.cuhk.edu.hk/~wkshum/wordpress/?p=404
    .. [2] Cremona, J. E., Rusin, D. (2003). Efficient Solution of Rational Conics.
           Mathematics of Computation, 72(243), 1417-1441.
           https://doi.org/10.1090/S0025-5718-02-01480-1
    """
    a = abs(a)
    def _dot(u, v):
        return u[0]*v[0] + a*u[1]*v[1]

    u = (b, 0)
    v = (w, 1) if b*w >= 0 else (-w, -1)
    # i.e., _dot(u, v) >= 0

    if b**2 < w**2 + a:
        u, v = v, u
    # i.e., norm(u) >= norm(v), where norm(u) := sqrt(_dot(u, u))

    while _dot(u, u) > (dv := _dot(v, v)):
        k = _dot(u, v) // dv
        u, v = v, (u[0] - k*v[0], u[1] - k*v[1])
    c = (v[0] - u[0], v[1] - u[1])
    if _dot(c, c) <= _dot(u, u) <= 2*_dot(u, v):
        return c
    return u


def holzer(x, y, z, a, b, c):
    r"""
    Simplify the solution `(x, y, z)` of the equation
    `ax^2 + by^2 = cz^2` with `a, b, c > 0` and `z^2 \geq \mid ab \mid` to
    a new reduced solution `(x', y', z')` such that `z'^2 \leq \mid ab \mid`.

    The algorithm is an interpretation of Mordell's reduction as described
    on page 8 of Cremona and Rusin's paper [1]_ and the work of Mordell in
    reference [2]_.

    References
    ==========

    .. [1] Cremona, J. E., Rusin, D. (2003). Efficient Solution of Rational Conics.
           Mathematics of Computation, 72(243), 1417-1441.
           https://doi.org/10.1090/S0025-5718-02-01480-1
    .. [2] Diophantine Equations, L. J. Mordell, page 48.

    """

    if _odd(c):
        k = 2*c
    else:
        k = c//2

    small = a*b*c
    step = 0
    while True:
        t1, t2, t3 = a*x**2, b*y**2, c*z**2
        # check that it's a solution
        if t1 + t2 != t3:
            if step == 0:
                raise ValueError('bad starting solution')
            break
        x_0, y_0, z_0 = x, y, z
        if max(t1, t2, t3) <= small:
            # Holzer condition
            break

        uv = u, v = base_solution_linear(k, y_0, -x_0)
        if None in uv:
            break

        p, q = -(a*u*x_0 + b*v*y_0), c*z_0
        r = Rational(p, q)
        if _even(c):
            w = _nint_or_floor(p, q)
            assert abs(w - r) <= S.Half
        else:
            w = p//q  # floor
            if _odd(a*u + b*v + c*w):
                w += 1
            assert abs(w - r) <= S.One

        A = (a*u**2 + b*v**2 + c*w**2)
        B = (a*u*x_0 + b*v*y_0 + c*w*z_0)
        x = Rational(x_0*A - 2*u*B, k)
        y = Rational(y_0*A - 2*v*B, k)
        z = Rational(z_0*A - 2*w*B, k)
        assert all(i.is_Integer for i in (x, y, z))
        step += 1

    return tuple([int(i) for i in (x_0, y_0, z_0)])


def diop_general_pythagorean(eq, param=symbols("m", integer=True)):
    """
    Solves the general pythagorean equation,
    `a_{1}^2x_{1}^2 + a_{2}^2x_{2}^2 + . . . + a_{n}^2x_{n}^2 - a_{n + 1}^2x_{n + 1}^2 = 0`.

    Returns a tuple which contains a parametrized solution to the equation,
    sorted in the same order as the input variables.

    Usage
    =====

    ``diop_general_pythagorean(eq, param)``: where ``eq`` is a general
    pythagorean equation which is assumed to be zero and ``param`` is the base
    parameter used to construct other parameters by subscripting.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import diop_general_pythagorean
    >>> from sympy.abc import a, b, c, d, e
    >>> diop_general_pythagorean(a**2 + b**2 + c**2 - d**2)
    (m1**2 + m2**2 - m3**2, 2*m1*m3, 2*m2*m3, m1**2 + m2**2 + m3**2)
    >>> diop_general_pythagorean(9*a**2 - 4*b**2 + 16*c**2 + 25*d**2 + e**2)
    (10*m1**2  + 10*m2**2  + 10*m3**2 - 10*m4**2, 15*m1**2  + 15*m2**2  + 15*m3**2  + 15*m4**2, 15*m1*m4, 12*m2*m4, 60*m3*m4)
    """
    var, coeff, diop_type  = classify_diop(eq, _dict=False)

    if diop_type == GeneralPythagorean.name:
        if param is None:
            params = None
        else:
            params = symbols('%s1:%i' % (param, len(var)), integer=True)
        return list(GeneralPythagorean(eq).solve(parameters=params))[0]


def diop_general_sum_of_squares(eq, limit=1):
    r"""
    Solves the equation `x_{1}^2 + x_{2}^2 + . . . + x_{n}^2 - k = 0`.

    Returns at most ``limit`` number of solutions.

    Usage
    =====

    ``general_sum_of_squares(eq, limit)`` : Here ``eq`` is an expression which
    is assumed to be zero. Also, ``eq`` should be in the form,
    `x_{1}^2 + x_{2}^2 + . . . + x_{n}^2 - k = 0`.

    Details
    =======

    When `n = 3` if `k = 4^a(8m + 7)` for some `a, m \in Z` then there will be
    no solutions. Refer to [1]_ for more details.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import diop_general_sum_of_squares
    >>> from sympy.abc import a, b, c, d, e
    >>> diop_general_sum_of_squares(a**2 + b**2 + c**2 + d**2 + e**2 - 2345)
    {(15, 22, 22, 24, 24)}

    Reference
    =========

    .. [1] Representing an integer as a sum of three squares, [online],
        Available:
        https://proofwiki.org/wiki/Integer_as_Sum_of_Three_Squares
    """
    var, coeff, diop_type = classify_diop(eq, _dict=False)

    if diop_type == GeneralSumOfSquares.name:
        return set(GeneralSumOfSquares(eq).solve(limit=limit))


def diop_general_sum_of_even_powers(eq, limit=1):
    """
    Solves the equation `x_{1}^e + x_{2}^e + . . . + x_{n}^e - k = 0`
    where `e` is an even, integer power.

    Returns at most ``limit`` number of solutions.

    Usage
    =====

    ``general_sum_of_even_powers(eq, limit)`` : Here ``eq`` is an expression which
    is assumed to be zero. Also, ``eq`` should be in the form,
    `x_{1}^e + x_{2}^e + . . . + x_{n}^e - k = 0`.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import diop_general_sum_of_even_powers
    >>> from sympy.abc import a, b
    >>> diop_general_sum_of_even_powers(a**4 + b**4 - (2**4 + 3**4))
    {(2, 3)}

    See Also
    ========

    power_representation
    """
    var, coeff, diop_type = classify_diop(eq, _dict=False)

    if diop_type == GeneralSumOfEvenPowers.name:
        return set(GeneralSumOfEvenPowers(eq).solve(limit=limit))


## Functions below this comment can be more suitably grouped under
## an Additive number theory module rather than the Diophantine
## equation module.


def partition(n, k=None, zeros=False):
    """
    Returns a generator that can be used to generate partitions of an integer
    `n`.

    Explanation
    ===========

    A partition of `n` is a set of positive integers which add up to `n`. For
    example, partitions of 3 are 3, 1 + 2, 1 + 1 + 1. A partition is returned
    as a tuple. If ``k`` equals None, then all possible partitions are returned
    irrespective of their size, otherwise only the partitions of size ``k`` are
    returned. If the ``zero`` parameter is set to True then a suitable
    number of zeros are added at the end of every partition of size less than
    ``k``.

    ``zero`` parameter is considered only if ``k`` is not None. When the
    partitions are over, the last `next()` call throws the ``StopIteration``
    exception, so this function should always be used inside a try - except
    block.

    Details
    =======

    ``partition(n, k)``: Here ``n`` is a positive integer and ``k`` is the size
    of the partition which is also positive integer.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import partition
    >>> f = partition(5)
    >>> next(f)
    (1, 1, 1, 1, 1)
    >>> next(f)
    (1, 1, 1, 2)
    >>> g = partition(5, 3)
    >>> next(g)
    (1, 1, 3)
    >>> next(g)
    (1, 2, 2)
    >>> g = partition(5, 3, zeros=True)
    >>> next(g)
    (0, 0, 5)

    """
    if not zeros or k is None:
        for i in ordered_partitions(n, k):
            yield tuple(i)
    else:
        for m in range(1, k + 1):
            for i in ordered_partitions(n, m):
                i = tuple(i)
                yield (0,)*(k - len(i)) + i


def prime_as_sum_of_two_squares(p):
    """
    Represent a prime `p` as a unique sum of two squares; this can
    only be done if the prime is congruent to 1 mod 4.

    Parameters
    ==========

    p : Integer
        A prime that is congruent to 1 mod 4

    Returns
    =======

    (int, int) | None : Pair of positive integers ``(x, y)`` satisfying ``x**2 + y**2 = p``.
                        None if ``p`` is not congruent to 1 mod 4.

    Raises
    ======

    ValueError
        If ``p`` is not prime number

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import prime_as_sum_of_two_squares
    >>> prime_as_sum_of_two_squares(7)  # can't be done
    >>> prime_as_sum_of_two_squares(5)
    (1, 2)

    Reference
    =========

    .. [1] Representing a number as a sum of four squares, [online],
           Available: https://schorn.ch/lagrange.html

    See Also
    ========

    sum_of_squares

    """
    p = as_int(p)
    if p % 4 != 1:
        return
    if not isprime(p):
        raise ValueError("p should be a prime number")

    if p % 8 == 5:
        # Legendre symbol (2/p) == -1 if p % 8 in [3, 5]
        b = 2
    elif p % 12 == 5:
        # Legendre symbol (3/p) == -1 if p % 12 in [5, 7]
        b = 3
    elif p % 5 in [2, 3]:
        # Legendre symbol (5/p) == -1 if p % 5 in [2, 3]
        b = 5
    else:
        b = 7
        while jacobi(b, p) == 1:
            b = nextprime(b)

    b = pow(b, p >> 2, p)
    a = p
    while b**2 > p:
        a, b = b, a % b
    return (int(a % b), int(b))  # convert from long


def sum_of_three_squares(n):
    r"""
    Returns a 3-tuple $(a, b, c)$ such that $a^2 + b^2 + c^2 = n$ and
    $a, b, c \geq 0$.

    Returns None if $n = 4^a(8m + 7)$ for some `a, m \in \mathbb{Z}`. See
    [1]_ for more details.

    Parameters
    ==========

    n : Integer
        non-negative integer

    Returns
    =======

    (int, int, int) | None : 3-tuple non-negative integers ``(a, b, c)`` satisfying ``a**2 + b**2 + c**2 = n``.
                             a,b,c are sorted in ascending order. ``None`` if no such ``(a,b,c)``.

    Raises
    ======

    ValueError
        If ``n`` is a negative integer

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import sum_of_three_squares
    >>> sum_of_three_squares(44542)
    (18, 37, 207)

    References
    ==========

    .. [1] Representing a number as a sum of three squares, [online],
        Available: https://schorn.ch/lagrange.html

    See Also
    ========

    power_representation :
        ``sum_of_three_squares(n)`` is one of the solutions output by ``power_representation(n, 2, 3, zeros=True)``

    """
    # https://math.stackexchange.com/questions/483101/rabin-and-shallit-algorithm/651425#651425
    # discusses these numbers (except for 1, 2, 3) as the exceptions of H&L's conjecture that
    # Every sufficiently large number n is either a square or the sum of a prime and a square.
    special = {1: (0, 0, 1), 2: (0, 1, 1), 3: (1, 1, 1), 10: (0, 1, 3), 34: (3, 3, 4),
               58: (0, 3, 7), 85: (0, 6, 7), 130: (0, 3, 11), 214: (3, 6, 13), 226: (8, 9, 9),
               370: (8, 9, 15), 526: (6, 7, 21), 706: (15, 15, 16), 730: (0, 1, 27),
               1414: (6, 17, 33), 1906: (13, 21, 36), 2986: (21, 32, 39), 9634: (56, 57, 57)}
    n = as_int(n)
    if n < 0:
        raise ValueError("n should be a non-negative integer")
    if n == 0:
        return (0, 0, 0)
    n, v = remove(n, 4)
    v = 1 << v
    if n % 8 == 7:
        return
    if n in special:
        return tuple([v*i for i in special[n]])

    s, _exact = integer_nthroot(n, 2)
    if _exact:
        return (0, 0, v*s)
    if n % 8 == 3:
        if not s % 2:
            s -= 1
        for x in range(s, -1, -2):
            N = (n - x**2) // 2
            if isprime(N):
                # n % 8 == 3 and x % 2 == 1 => N % 4 == 1
                y, z = prime_as_sum_of_two_squares(N)
                return tuple(sorted([v*x, v*(y + z), v*abs(y - z)]))
        # We will never reach this point because there must be a solution.
        assert False

    # assert n % 4 in [1, 2]
    if not((n % 2) ^ (s % 2)):
        s -= 1
    for x in range(s, -1, -2):
        N = n - x**2
        if isprime(N):
            # assert N % 4 == 1
            y, z = prime_as_sum_of_two_squares(N)
            return tuple(sorted([v*x, v*y, v*z]))
    # We will never reach this point because there must be a solution.
    assert False


def sum_of_four_squares(n):
    r"""
    Returns a 4-tuple `(a, b, c, d)` such that `a^2 + b^2 + c^2 + d^2 = n`.
    Here `a, b, c, d \geq 0`.

    Parameters
    ==========

    n : Integer
        non-negative integer

    Returns
    =======

    (int, int, int, int) : 4-tuple non-negative integers ``(a, b, c, d)`` satisfying ``a**2 + b**2 + c**2 + d**2 = n``.
                           a,b,c,d are sorted in ascending order.

    Raises
    ======

    ValueError
        If ``n`` is a negative integer

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import sum_of_four_squares
    >>> sum_of_four_squares(3456)
    (8, 8, 32, 48)
    >>> sum_of_four_squares(1294585930293)
    (0, 1234, 2161, 1137796)

    References
    ==========

    .. [1] Representing a number as a sum of four squares, [online],
        Available: https://schorn.ch/lagrange.html

    See Also
    ========

    power_representation :
        ``sum_of_four_squares(n)`` is one of the solutions output by ``power_representation(n, 2, 4, zeros=True)``

    """
    n = as_int(n)
    if n < 0:
        raise ValueError("n should be a non-negative integer")
    if n == 0:
        return (0, 0, 0, 0)
    # remove factors of 4 since a solution in terms of 3 squares is
    # going to be returned; this is also done in sum_of_three_squares,
    # but it needs to be done here to select d
    n, v = remove(n, 4)
    v = 1 << v
    if n % 8 == 7:
        d = 2
        n = n - 4
    elif n % 8 in (2, 6):
        d = 1
        n = n - 1
    else:
        d = 0
    x, y, z = sum_of_three_squares(n)  # sorted
    return tuple(sorted([v*d, v*x, v*y, v*z]))


def power_representation(n, p, k, zeros=False):
    r"""
    Returns a generator for finding k-tuples of integers,
    `(n_{1}, n_{2}, . . . n_{k})`, such that
    `n = n_{1}^p + n_{2}^p + . . . n_{k}^p`.

    Usage
    =====

    ``power_representation(n, p, k, zeros)``: Represent non-negative number
    ``n`` as a sum of ``k`` ``p``\ th powers. If ``zeros`` is true, then the
    solutions is allowed to contain zeros.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import power_representation

    Represent 1729 as a sum of two cubes:

    >>> f = power_representation(1729, 3, 2)
    >>> next(f)
    (9, 10)
    >>> next(f)
    (1, 12)

    If the flag `zeros` is True, the solution may contain tuples with
    zeros; any such solutions will be generated after the solutions
    without zeros:

    >>> list(power_representation(125, 2, 3, zeros=True))
    [(5, 6, 8), (3, 4, 10), (0, 5, 10), (0, 2, 11)]

    For even `p` the `permute_sign` function can be used to get all
    signed values:

    >>> from sympy.utilities.iterables import permute_signs
    >>> list(permute_signs((1, 12)))
    [(1, 12), (-1, 12), (1, -12), (-1, -12)]

    All possible signed permutations can also be obtained:

    >>> from sympy.utilities.iterables import signed_permutations
    >>> list(signed_permutations((1, 12)))
    [(1, 12), (-1, 12), (1, -12), (-1, -12), (12, 1), (-12, 1), (12, -1), (-12, -1)]
    """
    n, p, k = [as_int(i) for i in (n, p, k)]

    if n < 0:
        if p % 2:
            for t in power_representation(-n, p, k, zeros):
                yield tuple(-i for i in t)
        return

    if p < 1 or k < 1:
        raise ValueError(filldedent('''
    Expecting positive integers for `(p, k)`, but got `(%s, %s)`'''
    % (p, k)))

    if n == 0:
        if zeros:
            yield (0,)*k
        return

    if k == 1:
        if p == 1:
            yield (n,)
        elif n == 1:
            yield (1,)
        else:
            be = perfect_power(n)
            if be:
                b, e = be
                d, r = divmod(e, p)
                if not r:
                    yield (b**d,)
        return

    if p == 1:
        yield from partition(n, k, zeros=zeros)
        return

    if p == 2:
        if k == 3:
            n, v = remove(n, 4)
            if v:
                v = 1 << v
                for t in power_representation(n, p, k, zeros):
                    yield tuple(i*v for i in t)
                return
        feasible = _can_do_sum_of_squares(n, k)
        if not feasible:
            return
        if not zeros:
            if n > 33 and k >= 5 and k <= n and n - k in (
                13, 10, 7, 5, 4, 2, 1):
                '''Todd G. Will, "When Is n^2 a Sum of k Squares?", [online].
                Available: https://www.maa.org/sites/default/files/Will-MMz-201037918.pdf'''
                return
            # quick tests since feasibility includes the possibility of 0
            if k == 4 and (n in (1, 3, 5, 9, 11, 17, 29, 41) or remove(n, 4)[0] in (2, 6, 14)):
                # A000534
                return
            if k == 3 and n in (1, 2, 5, 10, 13, 25, 37, 58, 85, 130):  # or n = some number >= 5*10**10
                # A051952
                return
        if feasible is not True:  # it's prime and k == 2
            yield prime_as_sum_of_two_squares(n)
            return

    if k == 2 and p > 2:
        be = perfect_power(n)
        if be and be[1] % p == 0:
            return  # Fermat: a**n + b**n = c**n has no solution for n > 2

    if n >= k:
        a = integer_nthroot(n - (k - 1), p)[0]
        for t in pow_rep_recursive(a, k, n, [], p):
            yield tuple(reversed(t))

    if zeros:
        a = integer_nthroot(n, p)[0]
        for i in range(1, k):
            for t in pow_rep_recursive(a, i, n, [], p):
                yield tuple(reversed(t + (0,)*(k - i)))


sum_of_powers = power_representation


def pow_rep_recursive(n_i, k, n_remaining, terms, p):
    # Invalid arguments
    if n_i <= 0 or k <= 0:
        return

    # No solutions may exist
    if n_remaining < k:
        return
    if k * pow(n_i, p) < n_remaining:
        return

    if k == 0 and n_remaining == 0:
        yield tuple(terms)

    elif k == 1:
        # next_term^p must equal to n_remaining
        next_term, exact = integer_nthroot(n_remaining, p)
        if exact and next_term <= n_i:
            yield tuple(terms + [next_term])
        return

    else:
        # TODO: Fall back to diop_DN when k = 2
        if n_i >= 1 and k > 0:
            for next_term in range(1, n_i + 1):
                residual = n_remaining - pow(next_term, p)
                if residual < 0:
                    break
                yield from pow_rep_recursive(next_term, k - 1, residual, terms + [next_term], p)


def sum_of_squares(n, k, zeros=False):
    """Return a generator that yields the k-tuples of nonnegative
    values, the squares of which sum to n. If zeros is False (default)
    then the solution will not contain zeros. The nonnegative
    elements of a tuple are sorted.

    * If k == 1 and n is square, (n,) is returned.

    * If k == 2 then n can only be written as a sum of squares if
      every prime in the factorization of n that has the form
      4*k + 3 has an even multiplicity. If n is prime then
      it can only be written as a sum of two squares if it is
      in the form 4*k + 1.

    * if k == 3 then n can be written as a sum of squares if it does
      not have the form 4**m*(8*k + 7).

    * all integers can be written as the sum of 4 squares.

    * if k > 4 then n can be partitioned and each partition can
      be written as a sum of 4 squares; if n is not evenly divisible
      by 4 then n can be written as a sum of squares only if the
      an additional partition can be written as sum of squares.
      For example, if k = 6 then n is partitioned into two parts,
      the first being written as a sum of 4 squares and the second
      being written as a sum of 2 squares -- which can only be
      done if the condition above for k = 2 can be met, so this will
      automatically reject certain partitions of n.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import sum_of_squares
    >>> list(sum_of_squares(25, 2))
    [(3, 4)]
    >>> list(sum_of_squares(25, 2, True))
    [(3, 4), (0, 5)]
    >>> list(sum_of_squares(25, 4))
    [(1, 2, 2, 4)]

    See Also
    ========

    sympy.utilities.iterables.signed_permutations
    """
    yield from power_representation(n, 2, k, zeros)


def _can_do_sum_of_squares(n, k):
    """Return True if n can be written as the sum of k squares,
    False if it cannot, or 1 if ``k == 2`` and ``n`` is prime (in which
    case it *can* be written as a sum of two squares). A False
    is returned only if it cannot be written as ``k``-squares, even
    if 0s are allowed.
    """
    if k < 1:
        return False
    if n < 0:
        return False
    if n == 0:
        return True
    if k == 1:
        return is_square(n)
    if k == 2:
        if n in (1, 2):
            return True
        if isprime(n):
            if n % 4 == 1:
                return 1  # signal that it was prime
            return False
        # n is a composite number
        # we can proceed iff no prime factor in the form 4*k + 3
        # has an odd multiplicity
        return all(p % 4 !=3 or m % 2 == 0 for p, m in factorint(n).items())
    if k == 3:
        return remove(n, 4)[0] % 8 != 7
    # every number can be written as a sum of 4 squares; for k > 4 partitions
    # can be 0
    return True
<|endoftext|>""" SymPy interface to Unification engine

See sympy.unify for module level docstring
See sympy.unify.core for algorithmic docstring """

from sympy.core import Basic, Add, Mul, Pow
from sympy.core.operations import AssocOp, LatticeOp
from sympy.matrices import MatAdd, MatMul, MatrixExpr
from sympy.sets.sets import Union, Intersection, FiniteSet
from sympy.unify.core import Compound, Variable, CondVariable
from sympy.unify import core

basic_new_legal = [MatrixExpr]
eval_false_legal = [AssocOp, Pow, FiniteSet]
illegal = [LatticeOp]

def sympy_associative(op):
    assoc_ops = (AssocOp, MatAdd, MatMul, Union, Intersection, FiniteSet)
    return any(issubclass(op, aop) for aop in assoc_ops)

def sympy_commutative(op):
    comm_ops = (Add, MatAdd, Union, Intersection, FiniteSet)
    return any(issubclass(op, cop) for cop in comm_ops)

def is_associative(x):
    return isinstance(x, Compound) and sympy_associative(x.op)

def is_commutative(x):
    if not isinstance(x, Compound):
        return False
    if sympy_commutative(x.op):
        return True
    if issubclass(x.op, Mul):
        return all(construct(arg).is_commutative for arg in x.args)

def mk_matchtype(typ):
    def matchtype(x):
        return (isinstance(x, typ) or
                isinstance(x, Compound) and issubclass(x.op, typ))
    return matchtype

def deconstruct(s, variables=()):
    """ Turn a SymPy object into a Compound """
    if s in variables:
        return Variable(s)
    if isinstance(s, (Variable, CondVariable)):
        return s
    if not isinstance(s, Basic) or s.is_Atom:
        return s
    return Compound(s.__class__,
                    tuple(deconstruct(arg, variables) for arg in s.args))

def construct(t):
    """ Turn a Compound into a SymPy object """
    if isinstance(t, (Variable, CondVariable)):
        return t.arg
    if not isinstance(t, Compound):
        return t
    if any(issubclass(t.op, cls) for cls in eval_false_legal):
        return t.op(*map(construct, t.args), evaluate=False)
    elif any(issubclass(t.op, cls) for cls in basic_new_legal):
        return Basic.__new__(t.op, *map(construct, t.args))
    else:
        return t.op(*map(construct, t.args))

def rebuild(s):
    """ Rebuild a SymPy expression.

    This removes harm caused by Expr-Rules interactions.
    """
    return construct(deconstruct(s))

def unify(x, y, s=None, variables=(), **kwargs):
    """ Structural unification of two expressions/patterns.

    Examples
    ========

    >>> from sympy.unify.usympy import unify
    >>> from sympy import Basic, S
    >>> from sympy.abc import x, y, z, p, q

    >>> next(unify(Basic(S(1), S(2)), Basic(S(1), x), variables=[x]))
    {x: 2}

    >>> expr = 2*x + y + z
    >>> pattern = 2*p + q
    >>> next(unify(expr, pattern, {}, variables=(p, q)))
    {p: x, q: y + z}

    Unification supports commutative and associative matching

    >>> expr = x + y + z
    >>> pattern = p + q
    >>> len(list(unify(expr, pattern, {}, variables=(p, q))))
    12

    Symbols not indicated to be variables are treated as literal,
    else they are wild-like and match anything in a sub-expression.

    >>> expr = x*y*z + 3
    >>> pattern = x*y + 3
    >>> next(unify(expr, pattern, {}, variables=[x, y]))
    {x: y, y: x*z}

    The x and y of the pattern above were in a Mul and matched factors
    in the Mul of expr. Here, a single symbol matches an entire term:

    >>> expr = x*y + 3
    >>> pattern = p + 3
    >>> next(unify(expr, pattern, {}, variables=[p]))
    {p: x*y}

    """
    decons = lambda x: deconstruct(x, variables)
    s = s or {}
    s = {decons(k): decons(v) for k, v in s.items()}

    ds = core.unify(decons(x), decons(y), s,
                                     is_associative=is_associative,
                                     is_commutative=is_commutative,
                                     **kwargs)
    for d in ds:
        yield {construct(k): construct(v) for k, v in d.items()}
<|endoftext|>""" Functions to support rewriting of SymPy expressions """

from sympy.core.expr import Expr
from sympy.assumptions import ask
from sympy.strategies.tools import subs
from sympy.unify.usympy import rebuild, unify

def rewriterule(source, target, variables=(), condition=None, assume=None):
    """ Rewrite rule.

    Transform expressions that match source into expressions that match target
    treating all ``variables`` as wilds.

    Examples
    ========

    >>> from sympy.abc import w, x, y, z
    >>> from sympy.unify.rewrite import rewriterule
    >>> from sympy import default_sort_key
    >>> rl = rewriterule(x + y, x**y, [x, y])
    >>> sorted(rl(z + 3), key=default_sort_key)
    [3**z, z**3]

    Use ``condition`` to specify additional requirements.  Inputs are taken in
    the same order as is found in variables.

    >>> rl = rewriterule(x + y, x**y, [x, y], lambda x, y: x.is_integer)
    >>> list(rl(z + 3))
    [3**z]

    Use ``assume`` to specify additional requirements using new assumptions.

    >>> from sympy.assumptions import Q
    >>> rl = rewriterule(x + y, x**y, [x, y], assume=Q.integer(x))
    >>> list(rl(z + 3))
    [3**z]

    Assumptions for the local context are provided at rule runtime

    >>> list(rl(w + z, Q.integer(z)))
    [z**w]
    """

    def rewrite_rl(expr, assumptions=True):
        for match in unify(source, expr, {}, variables=variables):
            if (condition and
                not condition(*[match.get(var, var) for var in variables])):
                continue
            if (assume and not ask(assume.xreplace(match), assumptions)):
                continue
            expr2 = subs(match)(target)
            if isinstance(expr2, Expr):
                expr2 = rebuild(expr2)
            yield expr2
    return rewrite_rl
<|endoftext|>""" Generic Unification algorithm for expression trees with lists of children

This implementation is a direct translation of

Artificial Intelligence: A Modern Approach by Stuart Russel and Peter Norvig
Second edition, section 9.2, page 276

It is modified in the following ways:

1.  We allow associative and commutative Compound expressions. This results in
    combinatorial blowup.
2.  We explore the tree lazily.
3.  We provide generic interfaces to symbolic algebra libraries in Python.

A more traditional version can be found here
http://aima.cs.berkeley.edu/python/logic.html
"""

from sympy.utilities.iterables import kbins

class Compound:
    """ A little class to represent an interior node in the tree

    This is analogous to SymPy.Basic for non-Atoms
    """
    def __init__(self, op, args):
        self.op = op
        self.args = args

    def __eq__(self, other):
        return (type(self) is type(other) and self.op == other.op and
                self.args == other.args)

    def __hash__(self):
        return hash((type(self), self.op, self.args))

    def __str__(self):
        return "%s[%s]" % (str(self.op), ', '.join(map(str, self.args)))

class Variable:
    """ A Wild token """
    def __init__(self, arg):
        self.arg = arg

    def __eq__(self, other):
        return type(self) is type(other) and self.arg == other.arg

    def __hash__(self):
        return hash((type(self), self.arg))

    def __str__(self):
        return "Variable(%s)" % str(self.arg)

class CondVariable:
    """ A wild token that matches conditionally.

    arg   - a wild token.
    valid - an additional constraining function on a match.
    """
    def __init__(self, arg, valid):
        self.arg = arg
        self.valid = valid

    def __eq__(self, other):
        return (type(self) is type(other) and
                self.arg == other.arg and
                self.valid == other.valid)

    def __hash__(self):
        return hash((type(self), self.arg, self.valid))

    def __str__(self):
        return "CondVariable(%s)" % str(self.arg)

def unify(x, y, s=None, **fns):
    """ Unify two expressions.

    Parameters
    ==========

        x, y - expression trees containing leaves, Compounds and Variables.
        s    - a mapping of variables to subtrees.

    Returns
    =======

        lazy sequence of mappings {Variable: subtree}

    Examples
    ========

    >>> from sympy.unify.core import unify, Compound, Variable
    >>> expr    = Compound("Add", ("x", "y"))
    >>> pattern = Compound("Add", ("x", Variable("a")))
    >>> next(unify(expr, pattern, {}))
    {Variable(a): 'y'}
    """
    s = s or {}

    if x == y:
        yield s
    elif isinstance(x, (Variable, CondVariable)):
        yield from unify_var(x, y, s, **fns)
    elif isinstance(y, (Variable, CondVariable)):
        yield from unify_var(y, x, s, **fns)
    elif isinstance(x, Compound) and isinstance(y, Compound):
        is_commutative = fns.get('is_commutative', lambda x: False)
        is_associative = fns.get('is_associative', lambda x: False)
        for sop in unify(x.op, y.op, s, **fns):
            if is_associative(x) and is_associative(y):
                a, b = (x, y) if len(x.args) < len(y.args) else (y, x)
                if is_commutative(x) and is_commutative(y):
                    combs = allcombinations(a.args, b.args, 'commutative')
                else:
                    combs = allcombinations(a.args, b.args, 'associative')
                for aaargs, bbargs in combs:
                    aa = [unpack(Compound(a.op, arg)) for arg in aaargs]
                    bb = [unpack(Compound(b.op, arg)) for arg in bbargs]
                    yield from unify(aa, bb, sop, **fns)
            elif len(x.args) == len(y.args):
                yield from unify(x.args, y.args, sop, **fns)

    elif is_args(x) and is_args(y) and len(x) == len(y):
        if len(x) == 0:
            yield s
        else:
            for shead in unify(x[0], y[0], s, **fns):
                yield from unify(x[1:], y[1:], shead, **fns)

def unify_var(var, x, s, **fns):
    if var in s:
        yield from unify(s[var], x, s, **fns)
    elif occur_check(var, x):
        pass
    elif isinstance(var, CondVariable) and var.valid(x):
        yield assoc(s, var, x)
    elif isinstance(var, Variable):
        yield assoc(s, var, x)

def occur_check(var, x):
    """ var occurs in subtree owned by x? """
    if var == x:
        return True
    elif isinstance(x, Compound):
        return occur_check(var, x.args)
    elif is_args(x):
        if any(occur_check(var, xi) for xi in x): return True
    return False

def assoc(d, key, val):
    """ Return copy of d with key associated to val """
    d = d.copy()
    d[key] = val
    return d

def is_args(x):
    """ Is x a traditional iterable? """
    return type(x) in (tuple, list, set)

def unpack(x):
    if isinstance(x, Compound) and len(x.args) == 1:
        return x.args[0]
    else:
        return x

def allcombinations(A, B, ordered):
    """
    Restructure A and B to have the same number of elements.

    Parameters
    ==========

    ordered must be either 'commutative' or 'associative'.

    A and B can be rearranged so that the larger of the two lists is
    reorganized into smaller sublists.

    Examples
    ========

    >>> from sympy.unify.core import allcombinations
    >>> for x in allcombinations((1, 2, 3), (5, 6), 'associative'): print(x)
    (((1,), (2, 3)), ((5,), (6,)))
    (((1, 2), (3,)), ((5,), (6,)))

    >>> for x in allcombinations((1, 2, 3), (5, 6), 'commutative'): print(x)
        (((1,), (2, 3)), ((5,), (6,)))
        (((1, 2), (3,)), ((5,), (6,)))
        (((1,), (3, 2)), ((5,), (6,)))
        (((1, 3), (2,)), ((5,), (6,)))
        (((2,), (1, 3)), ((5,), (6,)))
        (((2, 1), (3,)), ((5,), (6,)))
        (((2,), (3, 1)), ((5,), (6,)))
        (((2, 3), (1,)), ((5,), (6,)))
        (((3,), (1, 2)), ((5,), (6,)))
        (((3, 1), (2,)), ((5,), (6,)))
        (((3,), (2, 1)), ((5,), (6,)))
        (((3, 2), (1,)), ((5,), (6,)))
    """

    if ordered == "commutative":
        ordered = 11
    if ordered == "associative":
        ordered = None
    sm, bg = (A, B) if len(A) < len(B) else (B, A)
    for part in kbins(list(range(len(bg))), len(sm), ordered=ordered):
        if bg == B:
            yield tuple((a,) for a in A), partition(B, part)
        else:
            yield partition(A, part), tuple((b,) for b in B)

def partition(it, part):
    """ Partition a tuple/list into pieces defined by indices.

    Examples
    ========

    >>> from sympy.unify.core import partition
    >>> partition((10, 20, 30, 40), [[0, 1, 2], [3]])
    ((10, 20, 30), (40,))
    """
    return type(it)([index(it, ind) for ind in part])

def index(it, ind):
    """ Fancy indexing into an indexable iterable (tuple, list).

    Examples
    ========

    >>> from sympy.unify.core import index
    >>> index([10, 20, 30], (1, 2, 0))
    [20, 30, 10]
    """
    return type(it)([it[i] for i in ind])
<|endoftext|>from sympy.tensor import Indexed
from sympy.core.containers import Tuple
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.integrals.integrals import Integral


class IndexedIntegral(Integral):
    """
    Experimental class to test integration by indexed variables.

    Usage is analogue to ``Integral``, it simply adds awareness of
    integration over indices.

    Contraction of non-identical index symbols referring to the same
    ``IndexedBase`` is not yet supported.

    Examples
    ========

    >>> from sympy.sandbox.indexed_integrals import IndexedIntegral
    >>> from sympy import IndexedBase, symbols
    >>> A = IndexedBase('A')
    >>> i, j = symbols('i j', integer=True)
    >>> ii = IndexedIntegral(A[i], A[i])
    >>> ii
    Integral(_A[i], _A[i])
    >>> ii.doit()
    A[i]**2/2

    If the indices are different, indexed objects are considered to be
    different variables:

    >>> i2 = IndexedIntegral(A[j], A[i])
    >>> i2
    Integral(A[j], _A[i])
    >>> i2.doit()
    A[i]*A[j]
    """

    def __new__(cls, function, *limits, **assumptions):
        repl, limits = IndexedIntegral._indexed_process_limits(limits)
        function = sympify(function)
        function = function.xreplace(repl)
        obj = Integral.__new__(cls, function, *limits, **assumptions)
        obj._indexed_repl = repl
        obj._indexed_reverse_repl = {val: key for key, val in repl.items()}
        return obj

    def doit(self):
        res = super().doit()
        return res.xreplace(self._indexed_reverse_repl)

    @staticmethod
    def _indexed_process_limits(limits):
        repl = {}
        newlimits = []
        for i in limits:
            if isinstance(i, (tuple, list, Tuple)):
                v = i[0]
                vrest = i[1:]
            else:
                v = i
                vrest = ()
            if isinstance(v, Indexed):
                if v not in repl:
                    r = Dummy(str(v))
                    repl[v] = r
                newlimits.append((r,)+vrest)
            else:
                newlimits.append(i)
        return repl, newlimits
<|endoftext|>"""
This file contains some classical ciphers and routines
implementing a linear-feedback shift register (LFSR)
and the Diffie-Hellman key exchange.

.. warning::

   This module is intended for educational purposes only. Do not use the
   functions in this module for real cryptographic applications. If you wish
   to encrypt real data, we recommend using something like the `cryptography
   <https://cryptography.io/en/latest/>`_ module.

"""

from string import whitespace, ascii_uppercase as uppercase, printable
from functools import reduce
import string
import warnings

from itertools import cycle

from sympy.external.gmpy import GROUND_TYPES
from sympy.core import Symbol
from sympy.core.numbers import Rational
from sympy.core.random import _randrange, _randint
from sympy.external.gmpy import gcd, invert
from sympy.functions.combinatorial.numbers import (totient as _euler,
                                                   reduced_totient as _carmichael)
from sympy.matrices import Matrix
from sympy.ntheory import isprime, primitive_root, factorint
from sympy.ntheory.generate import nextprime
from sympy.ntheory.modular import crt
from sympy.polys.domains import FF
from sympy.polys.polytools import Poly
from sympy.utilities.misc import as_int, filldedent, translate
from sympy.utilities.iterables import uniq, multiset
from sympy.utilities.decorator import doctest_depends_on


if GROUND_TYPES == 'flint':
    __doctest_skip__ = ['lfsr_sequence']


class NonInvertibleCipherWarning(RuntimeWarning):
    """A warning raised if the cipher is not invertible."""
    def __init__(self, msg):
        self.fullMessage = msg

    def __str__(self):
        return '\n\t' + self.fullMessage

    def warn(self, stacklevel=3):
        warnings.warn(self, stacklevel=stacklevel)


def AZ(s=None):
    """Return the letters of ``s`` in uppercase. In case more than
    one string is passed, each of them will be processed and a list
    of upper case strings will be returned.

    Examples
    ========

    >>> from sympy.crypto.crypto import AZ
    >>> AZ('Hello, world!')
    'HELLOWORLD'
    >>> AZ('Hello, world!'.split())
    ['HELLO', 'WORLD']

    See Also
    ========

    check_and_join

    """
    if not s:
        return uppercase
    t = isinstance(s, str)
    if t:
        s = [s]
    rv = [check_and_join(i.upper().split(), uppercase, filter=True)
        for i in s]
    if t:
        return rv[0]
    return rv

bifid5 = AZ().replace('J', '')
bifid6 = AZ() + string.digits
bifid10 = printable


def padded_key(key, symbols):
    """Return a string of the distinct characters of ``symbols`` with
    those of ``key`` appearing first. A ValueError is raised if
    a) there are duplicate characters in ``symbols`` or
    b) there are characters in ``key`` that are  not in ``symbols``.

    Examples
    ========

    >>> from sympy.crypto.crypto import padded_key
    >>> padded_key('PUPPY', 'OPQRSTUVWXY')
    'PUYOQRSTVWX'
    >>> padded_key('RSA', 'ARTIST')
    Traceback (most recent call last):
    ...
    ValueError: duplicate characters in symbols: T

    """
    syms = list(uniq(symbols))
    if len(syms) != len(symbols):
        extra = ''.join(sorted({
            i for i in symbols if symbols.count(i) > 1}))
        raise ValueError('duplicate characters in symbols: %s' % extra)
    extra = set(key) - set(syms)
    if extra:
        raise ValueError(
            'characters in key but not symbols: %s' % ''.join(
            sorted(extra)))
    key0 = ''.join(list(uniq(key)))
    # remove from syms characters in key0
    return key0 + translate(''.join(syms), None, key0)


def check_and_join(phrase, symbols=None, filter=None):
    """
    Joins characters of ``phrase`` and if ``symbols`` is given, raises
    an error if any character in ``phrase`` is not in ``symbols``.

    Parameters
    ==========

    phrase
        String or list of strings to be returned as a string.

    symbols
        Iterable of characters allowed in ``phrase``.

        If ``symbols`` is ``None``, no checking is performed.

    Examples
    ========

    >>> from sympy.crypto.crypto import check_and_join
    >>> check_and_join('a phrase')
    'a phrase'
    >>> check_and_join('a phrase'.upper().split())
    'APHRASE'
    >>> check_and_join('a phrase!'.upper().split(), 'ARE', filter=True)
    'ARAE'
    >>> check_and_join('a phrase!'.upper().split(), 'ARE')
    Traceback (most recent call last):
    ...
    ValueError: characters in phrase but not symbols: "!HPS"

    """
    rv = ''.join(''.join(phrase))
    if symbols is not None:
        symbols = check_and_join(symbols)
        missing = ''.join(sorted(set(rv) - set(symbols)))
        if missing:
            if not filter:
                raise ValueError(
                    'characters in phrase but not symbols: "%s"' % missing)
            rv = translate(rv, None, missing)
    return rv


def _prep(msg, key, alp, default=None):
    if not alp:
        if not default:
            alp = AZ()
            msg = AZ(msg)
            key = AZ(key)
        else:
            alp = default
    else:
        alp = ''.join(alp)
    key = check_and_join(key, alp, filter=True)
    msg = check_and_join(msg, alp, filter=True)
    return msg, key, alp


def cycle_list(k, n):
    """
    Returns the elements of the list ``range(n)`` shifted to the
    left by ``k`` (so the list starts with ``k`` (mod ``n``)).

    Examples
    ========

    >>> from sympy.crypto.crypto import cycle_list
    >>> cycle_list(3, 10)
    [3, 4, 5, 6, 7, 8, 9, 0, 1, 2]

    """
    k = k % n
    return list(range(k, n)) + list(range(k))


######## shift cipher examples ############


def encipher_shift(msg, key, symbols=None):
    """
    Performs shift cipher encryption on plaintext msg, and returns the
    ciphertext.

    Parameters
    ==========

    key : int
        The secret key.

    msg : str
        Plaintext of upper-case letters.

    Returns
    =======

    str
        Ciphertext of upper-case letters.

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_shift, decipher_shift
    >>> msg = "GONAVYBEATARMY"
    >>> ct = encipher_shift(msg, 1); ct
    'HPOBWZCFBUBSNZ'

    To decipher the shifted text, change the sign of the key:

    >>> encipher_shift(ct, -1)
    'GONAVYBEATARMY'

    There is also a convenience function that does this with the
    original key:

    >>> decipher_shift(ct, 1)
    'GONAVYBEATARMY'

    Notes
    =====

    ALGORITHM:

        STEPS:
            0. Number the letters of the alphabet from 0, ..., N
            1. Compute from the string ``msg`` a list ``L1`` of
               corresponding integers.
            2. Compute from the list ``L1`` a new list ``L2``, given by
               adding ``(k mod 26)`` to each element in ``L1``.
            3. Compute from the list ``L2`` a string ``ct`` of
               corresponding letters.

    The shift cipher is also called the Caesar cipher, after
    Julius Caesar, who, according to Suetonius, used it with a
    shift of three to protect messages of military significance.
    Caesar's nephew Augustus reportedly used a similar cipher, but
    with a right shift of 1.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Caesar_cipher
    .. [2] https://mathworld.wolfram.com/CaesarsMethod.html

    See Also
    ========

    decipher_shift

    """
    msg, _, A = _prep(msg, '', symbols)
    shift = len(A) - key % len(A)
    key = A[shift:] + A[:shift]
    return translate(msg, key, A)


def decipher_shift(msg, key, symbols=None):
    """
    Return the text by shifting the characters of ``msg`` to the
    left by the amount given by ``key``.

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_shift, decipher_shift
    >>> msg = "GONAVYBEATARMY"
    >>> ct = encipher_shift(msg, 1); ct
    'HPOBWZCFBUBSNZ'

    To decipher the shifted text, change the sign of the key:

    >>> encipher_shift(ct, -1)
    'GONAVYBEATARMY'

    Or use this function with the original key:

    >>> decipher_shift(ct, 1)
    'GONAVYBEATARMY'

    """
    return encipher_shift(msg, -key, symbols)

def encipher_rot13(msg, symbols=None):
    """
    Performs the ROT13 encryption on a given plaintext ``msg``.

    Explanation
    ===========

    ROT13 is a substitution cipher which substitutes each letter
    in the plaintext message for the letter furthest away from it
    in the English alphabet.

    Equivalently, it is just a Caeser (shift) cipher with a shift
    key of 13 (midway point of the alphabet).

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/ROT13

    See Also
    ========

    decipher_rot13
    encipher_shift

    """
    return encipher_shift(msg, 13, symbols)

def decipher_rot13(msg, symbols=None):
    """
    Performs the ROT13 decryption on a given plaintext ``msg``.

    Explanation
    ============

    ``decipher_rot13`` is equivalent to ``encipher_rot13`` as both
    ``decipher_shift`` with a key of 13 and ``encipher_shift`` key with a
    key of 13 will return the same results. Nonetheless,
    ``decipher_rot13`` has nonetheless been explicitly defined here for
    consistency.

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_rot13, decipher_rot13
    >>> msg = 'GONAVYBEATARMY'
    >>> ciphertext = encipher_rot13(msg);ciphertext
    'TBANILORNGNEZL'
    >>> decipher_rot13(ciphertext)
    'GONAVYBEATARMY'
    >>> encipher_rot13(msg) == decipher_rot13(msg)
    True
    >>> msg == decipher_rot13(ciphertext)
    True

    """
    return decipher_shift(msg, 13, symbols)

######## affine cipher examples ############


def encipher_affine(msg, key, symbols=None, _inverse=False):
    r"""
    Performs the affine cipher encryption on plaintext ``msg``, and
    returns the ciphertext.

    Explanation
    ===========

    Encryption is based on the map `x \rightarrow ax+b` (mod `N`)
    where ``N`` is the number of characters in the alphabet.
    Decryption is based on the map `x \rightarrow cx+d` (mod `N`),
    where `c = a^{-1}` (mod `N`) and `d = -a^{-1}b` (mod `N`).
    In particular, for the map to be invertible, we need
    `\mathrm{gcd}(a, N) = 1` and an error will be raised if this is
    not true.

    Parameters
    ==========

    msg : str
        Characters that appear in ``symbols``.

    a, b : int, int
        A pair integers, with ``gcd(a, N) = 1`` (the secret key).

    symbols
        String of characters (default = uppercase letters).

        When no symbols are given, ``msg`` is converted to upper case
        letters and all other characters are ignored.

    Returns
    =======

    ct
        String of characters (the ciphertext message)

    Notes
    =====

    ALGORITHM:

        STEPS:
            0. Number the letters of the alphabet from 0, ..., N
            1. Compute from the string ``msg`` a list ``L1`` of
               corresponding integers.
            2. Compute from the list ``L1`` a new list ``L2``, given by
               replacing ``x`` by ``a*x + b (mod N)``, for each element
               ``x`` in ``L1``.
            3. Compute from the list ``L2`` a string ``ct`` of
               corresponding letters.

    This is a straightforward generalization of the shift cipher with
    the added complexity of requiring 2 characters to be deciphered in
    order to recover the key.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Affine_cipher

    See Also
    ========

    decipher_affine

    """
    msg, _, A = _prep(msg, '', symbols)
    N = len(A)
    a, b = key
    assert gcd(a, N) == 1
    if _inverse:
        c = invert(a, N)
        d = -b*c
        a, b = c, d
    B = ''.join([A[(a*i + b) % N] for i in range(N)])
    return translate(msg, A, B)


def decipher_affine(msg, key, symbols=None):
    r"""
    Return the deciphered text that was made from the mapping,
    `x \rightarrow ax+b` (mod `N`), where ``N`` is the
    number of characters in the alphabet. Deciphering is done by
    reciphering with a new key: `x \rightarrow cx+d` (mod `N`),
    where `c = a^{-1}` (mod `N`) and `d = -a^{-1}b` (mod `N`).

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_affine, decipher_affine
    >>> msg = "GO NAVY BEAT ARMY"
    >>> key = (3, 1)
    >>> encipher_affine(msg, key)
    'TROBMVENBGBALV'
    >>> decipher_affine(_, key)
    'GONAVYBEATARMY'

    See Also
    ========

    encipher_affine

    """
    return encipher_affine(msg, key, symbols, _inverse=True)


def encipher_atbash(msg, symbols=None):
    r"""
    Enciphers a given ``msg`` into its Atbash ciphertext and returns it.

    Explanation
    ===========

    Atbash is a substitution cipher originally used to encrypt the Hebrew
    alphabet. Atbash works on the principle of mapping each alphabet to its
    reverse / counterpart (i.e. a would map to z, b to y etc.)

    Atbash is functionally equivalent to the affine cipher with ``a = 25``
    and ``b = 25``

    See Also
    ========

    decipher_atbash

    """
    return encipher_affine(msg, (25, 25), symbols)


def decipher_atbash(msg, symbols=None):
    r"""
    Deciphers a given ``msg`` using Atbash cipher and returns it.

    Explanation
    ===========

    ``decipher_atbash`` is functionally equivalent to ``encipher_atbash``.
    However, it has still been added as a separate function to maintain
    consistency.

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_atbash, decipher_atbash
    >>> msg = 'GONAVYBEATARMY'
    >>> encipher_atbash(msg)
    'TLMZEBYVZGZINB'
    >>> decipher_atbash(msg)
    'TLMZEBYVZGZINB'
    >>> encipher_atbash(msg) == decipher_atbash(msg)
    True
    >>> msg == encipher_atbash(encipher_atbash(msg))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Atbash

    See Also
    ========

    encipher_atbash

    """
    return decipher_affine(msg, (25, 25), symbols)

#################### substitution cipher ###########################


def encipher_substitution(msg, old, new=None):
    r"""
    Returns the ciphertext obtained by replacing each character that
    appears in ``old`` with the corresponding character in ``new``.
    If ``old`` is a mapping, then new is ignored and the replacements
    defined by ``old`` are used.

    Explanation
    ===========

    This is a more general than the affine cipher in that the key can
    only be recovered by determining the mapping for each symbol.
    Though in practice, once a few symbols are recognized the mappings
    for other characters can be quickly guessed.

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_substitution, AZ
    >>> old = 'OEYAG'
    >>> new = '034^6'
    >>> msg = AZ("go navy! beat army!")
    >>> ct = encipher_substitution(msg, old, new); ct
    '60N^V4B3^T^RM4'

    To decrypt a substitution, reverse the last two arguments:

    >>> encipher_substitution(ct, new, old)
    'GONAVYBEATARMY'

    In the special case where ``old`` and ``new`` are a permutation of
    order 2 (representing a transposition of characters) their order
    is immaterial:

    >>> old = 'NAVY'
    >>> new = 'ANYV'
    >>> encipher = lambda x: encipher_substitution(x, old, new)
    >>> encipher('NAVY')
    'ANYV'
    >>> encipher(_)
    'NAVY'

    The substitution cipher, in general, is a method
    whereby "units" (not necessarily single characters) of plaintext
    are replaced with ciphertext according to a regular system.

    >>> ords = dict(zip('abc', ['\\%i' % ord(i) for i in 'abc']))
    >>> print(encipher_substitution('abc', ords))
    \97\98\99

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Substitution_cipher

    """
    return translate(msg, old, new)


######################################################################
#################### Vigenere cipher examples ########################
######################################################################

def encipher_vigenere(msg, key, symbols=None):
    """
    Performs the Vigenere cipher encryption on plaintext ``msg``, and
    returns the ciphertext.

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_vigenere, AZ
    >>> key = "encrypt"
    >>> msg = "meet me on monday"
    >>> encipher_vigenere(msg, key)
    'QRGKKTHRZQEBPR'

    Section 1 of the Kryptos sculpture at the CIA headquarters
    uses this cipher and also changes the order of the
    alphabet [2]_. Here is the first line of that section of
    the sculpture:

    >>> from sympy.crypto.crypto import decipher_vigenere, padded_key
    >>> alp = padded_key('KRYPTOS', AZ())
    >>> key = 'PALIMPSEST'
    >>> msg = 'EMUFPHZLRFAXYUSDJKZLDKRNSHGNFIVJ'
    >>> decipher_vigenere(msg, key, alp)
    'BETWEENSUBTLESHADINGANDTHEABSENC'

    Explanation
    ===========

    The Vigenere cipher is named after Blaise de Vigenere, a sixteenth
    century diplomat and cryptographer, by a historical accident.
    Vigenere actually invented a different and more complicated cipher.
    The so-called *Vigenere cipher* was actually invented
    by Giovan Batista Belaso in 1553.

    This cipher was used in the 1800's, for example, during the American
    Civil War. The Confederacy used a brass cipher disk to implement the
    Vigenere cipher (now on display in the NSA Museum in Fort
    Meade) [1]_.

    The Vigenere cipher is a generalization of the shift cipher.
    Whereas the shift cipher shifts each letter by the same amount
    (that amount being the key of the shift cipher) the Vigenere
    cipher shifts a letter by an amount determined by the key (which is
    a word or phrase known only to the sender and receiver).

    For example, if the key was a single letter, such as "C", then the
    so-called Vigenere cipher is actually a shift cipher with a
    shift of `2` (since "C" is the 2nd letter of the alphabet, if
    you start counting at `0`). If the key was a word with two
    letters, such as "CA", then the so-called Vigenere cipher will
    shift letters in even positions by `2` and letters in odd positions
    are left alone (shifted by `0`, since "A" is the 0th letter, if
    you start counting at `0`).


    ALGORITHM:

        INPUT:

            ``msg``: string of characters that appear in ``symbols``
            (the plaintext)

            ``key``: a string of characters that appear in ``symbols``
            (the secret key)

            ``symbols``: a string of letters defining the alphabet


        OUTPUT:

            ``ct``: string of characters (the ciphertext message)

        STEPS:
            0. Number the letters of the alphabet from 0, ..., N
            1. Compute from the string ``key`` a list ``L1`` of
               corresponding integers. Let ``n1 = len(L1)``.
            2. Compute from the string ``msg`` a list ``L2`` of
               corresponding integers. Let ``n2 = len(L2)``.
            3. Break ``L2`` up sequentially into sublists of size
               ``n1``; the last sublist may be smaller than ``n1``
            4. For each of these sublists ``L`` of ``L2``, compute a
               new list ``C`` given by ``C[i] = L[i] + L1[i] (mod N)``
               to the ``i``-th element in the sublist, for each ``i``.
            5. Assemble these lists ``C`` by concatenation into a new
               list of length ``n2``.
            6. Compute from the new list a string ``ct`` of
               corresponding letters.

    Once it is known that the key is, say, `n` characters long,
    frequency analysis can be applied to every `n`-th letter of
    the ciphertext to determine the plaintext. This method is
    called *Kasiski examination* (although it was first discovered
    by Babbage). If they key is as long as the message and is
    comprised of randomly selected characters -- a one-time pad -- the
    message is theoretically unbreakable.

    The cipher Vigenere actually discovered is an "auto-key" cipher
    described as follows.

    ALGORITHM:

        INPUT:

          ``key``: a string of letters (the secret key)

          ``msg``: string of letters (the plaintext message)

        OUTPUT:

          ``ct``: string of upper-case letters (the ciphertext message)

        STEPS:
            0. Number the letters of the alphabet from 0, ..., N
            1. Compute from the string ``msg`` a list ``L2`` of
               corresponding integers. Let ``n2 = len(L2)``.
            2. Let ``n1`` be the length of the key. Append to the
               string ``key`` the first ``n2 - n1`` characters of
               the plaintext message. Compute from this string (also of
               length ``n2``) a list ``L1`` of integers corresponding
               to the letter numbers in the first step.
            3. Compute a new list ``C`` given by
               ``C[i] = L1[i] + L2[i] (mod N)``.
            4. Compute from the new list a string ``ct`` of letters
               corresponding to the new integers.

    To decipher the auto-key ciphertext, the key is used to decipher
    the first ``n1`` characters and then those characters become the
    key to  decipher the next ``n1`` characters, etc...:

    >>> m = AZ('go navy, beat army! yes you can'); m
    'GONAVYBEATARMYYESYOUCAN'
    >>> key = AZ('gold bug'); n1 = len(key); n2 = len(m)
    >>> auto_key = key + m[:n2 - n1]; auto_key
    'GOLDBUGGONAVYBEATARMYYE'
    >>> ct = encipher_vigenere(m, auto_key); ct
    'MCYDWSHKOGAMKZCELYFGAYR'
    >>> n1 = len(key)
    >>> pt = []
    >>> while ct:
    ...     part, ct = ct[:n1], ct[n1:]
    ...     pt.append(decipher_vigenere(part, key))
    ...     key = pt[-1]
    ...
    >>> ''.join(pt) == m
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Vigenere_cipher
    .. [2] https://web.archive.org/web/20071116100808/https://filebox.vt.edu/users/batman/kryptos.html
       (short URL: https://goo.gl/ijr22d)

    """
    msg, key, A = _prep(msg, key, symbols)
    map = {c: i for i, c in enumerate(A)}
    key = [map[c] for c in key]
    N = len(map)
    k = len(key)
    rv = []
    for i, m in enumerate(msg):
        rv.append(A[(map[m] + key[i % k]) % N])
    rv = ''.join(rv)
    return rv


def decipher_vigenere(msg, key, symbols=None):
    """
    Decode using the Vigenere cipher.

    Examples
    ========

    >>> from sympy.crypto.crypto import decipher_vigenere
    >>> key = "encrypt"
    >>> ct = "QRGK kt HRZQE BPR"
    >>> decipher_vigenere(ct, key)
    'MEETMEONMONDAY'

    """
    msg, key, A = _prep(msg, key, symbols)
    map = {c: i for i, c in enumerate(A)}
    N = len(A)   # normally, 26
    K = [map[c] for c in key]
    n = len(K)
    C = [map[c] for c in msg]
    rv = ''.join([A[(-K[i % n] + c) % N] for i, c in enumerate(C)])
    return rv


#################### Hill cipher  ########################


def encipher_hill(msg, key, symbols=None, pad="Q"):
    r"""
    Return the Hill cipher encryption of ``msg``.

    Explanation
    ===========

    The Hill cipher [1]_, invented by Lester S. Hill in the 1920's [2]_,
    was the first polygraphic cipher in which it was practical
    (though barely) to operate on more than three symbols at once.
    The following discussion assumes an elementary knowledge of
    matrices.

    First, each letter is first encoded as a number starting with 0.
    Suppose your message `msg` consists of `n` capital letters, with no
    spaces. This may be regarded an `n`-tuple M of elements of
    `Z_{26}` (if the letters are those of the English alphabet). A key
    in the Hill cipher is a `k x k` matrix `K`, all of whose entries
    are in `Z_{26}`, such that the matrix `K` is invertible (i.e., the
    linear transformation `K: Z_{N}^k \rightarrow Z_{N}^k`
    is one-to-one).


    Parameters
    ==========

    msg
        Plaintext message of `n` upper-case letters.

    key
        A `k \times k` invertible matrix `K`, all of whose entries are
        in `Z_{26}` (or whatever number of symbols are being used).

    pad
        Character (default "Q") to use to make length of text be a
        multiple of ``k``.

    Returns
    =======

    ct
        Ciphertext of upper-case letters.

    Notes
    =====

    ALGORITHM:

        STEPS:
            0. Number the letters of the alphabet from 0, ..., N
            1. Compute from the string ``msg`` a list ``L`` of
               corresponding integers. Let ``n = len(L)``.
            2. Break the list ``L`` up into ``t = ceiling(n/k)``
               sublists ``L_1``, ..., ``L_t`` of size ``k`` (with
               the last list "padded" to ensure its size is
               ``k``).
            3. Compute new list ``C_1``, ..., ``C_t`` given by
               ``C[i] = K*L_i`` (arithmetic is done mod N), for each
               ``i``.
            4. Concatenate these into a list ``C = C_1 + ... + C_t``.
            5. Compute from ``C`` a string ``ct`` of corresponding
               letters. This has length ``k*t``.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hill_cipher
    .. [2] Lester S. Hill, Cryptography in an Algebraic Alphabet,
       The American Mathematical Monthly Vol.36, June-July 1929,
       pp.306-312.

    See Also
    ========

    decipher_hill

    """
    assert key.is_square
    assert len(pad) == 1
    msg, pad, A = _prep(msg, pad, symbols)
    map = {c: i for i, c in enumerate(A)}
    P = [map[c] for c in msg]
    N = len(A)
    k = key.cols
    n = len(P)
    m, r = divmod(n, k)
    if r:
        P = P + [map[pad]]*(k - r)
        m += 1
    rv = ''.join([A[c % N] for j in range(m) for c in
        list(key*Matrix(k, 1, [P[i]
        for i in range(k*j, k*(j + 1))]))])
    return rv


def decipher_hill(msg, key, symbols=None):
    """
    Deciphering is the same as enciphering but using the inverse of the
    key matrix.

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_hill, decipher_hill
    >>> from sympy import Matrix

    >>> key = Matrix([[1, 2], [3, 5]])
    >>> encipher_hill("meet me on monday", key)
    'UEQDUEODOCTCWQ'
    >>> decipher_hill(_, key)
    'MEETMEONMONDAY'

    When the length of the plaintext (stripped of invalid characters)
    is not a multiple of the key dimension, extra characters will
    appear at the end of the enciphered and deciphered text. In order to
    decipher the text, those characters must be included in the text to
    be deciphered. In the following, the key has a dimension of 4 but
    the text is 2 short of being a multiple of 4 so two characters will
    be added.

    >>> key = Matrix([[1, 1, 1, 2], [0, 1, 1, 0],
    ...               [2, 2, 3, 4], [1, 1, 0, 1]])
    >>> msg = "ST"
    >>> encipher_hill(msg, key)
    'HJEB'
    >>> decipher_hill(_, key)
    'STQQ'
    >>> encipher_hill(msg, key, pad="Z")
    'ISPK'
    >>> decipher_hill(_, key)
    'STZZ'

    If the last two characters of the ciphertext were ignored in
    either case, the wrong plaintext would be recovered:

    >>> decipher_hill("HD", key)
    'ORMV'
    >>> decipher_hill("IS", key)
    'UIKY'

    See Also
    ========

    encipher_hill

    """
    assert key.is_square
    msg, _, A = _prep(msg, '', symbols)
    map = {c: i for i, c in enumerate(A)}
    C = [map[c] for c in msg]
    N = len(A)
    k = key.cols
    n = len(C)
    m, r = divmod(n, k)
    if r:
        C = C + [0]*(k - r)
        m += 1
    key_inv = key.inv_mod(N)
    rv = ''.join([A[p % N] for j in range(m) for p in
        list(key_inv*Matrix(
        k, 1, [C[i] for i in range(k*j, k*(j + 1))]))])
    return rv


#################### Bifid cipher  ########################


def encipher_bifid(msg, key, symbols=None):
    r"""
    Performs the Bifid cipher encryption on plaintext ``msg``, and
    returns the ciphertext.

    This is the version of the Bifid cipher that uses an `n \times n`
    Polybius square.

    Parameters
    ==========

    msg
        Plaintext string.

    key
        Short string for key.

        Duplicate characters are ignored and then it is padded with the
        characters in ``symbols`` that were not in the short key.

    symbols
        `n \times n` characters defining the alphabet.

        (default is string.printable)

    Returns
    =======

    ciphertext
        Ciphertext using Bifid5 cipher without spaces.

    See Also
    ========

    decipher_bifid, encipher_bifid5, encipher_bifid6

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bifid_cipher

    """
    msg, key, A = _prep(msg, key, symbols, bifid10)
    long_key = ''.join(uniq(key)) or A

    n = len(A)**.5
    if n != int(n):
        raise ValueError(
            'Length of alphabet (%s) is not a square number.' % len(A))
    N = int(n)
    if len(long_key) < N**2:
        long_key = list(long_key) + [x for x in A if x not in long_key]

    # the fractionalization
    row_col = {ch: divmod(i, N) for i, ch in enumerate(long_key)}
    r, c = zip(*[row_col[x] for x in msg])
    rc = r + c
    ch = {i: ch for ch, i in row_col.items()}
    rv = ''.join(ch[i] for i in zip(rc[::2], rc[1::2]))
    return rv


def decipher_bifid(msg, key, symbols=None):
    r"""
    Performs the Bifid cipher decryption on ciphertext ``msg``, and
    returns the plaintext.

    This is the version of the Bifid cipher that uses the `n \times n`
    Polybius square.

    Parameters
    ==========

    msg
        Ciphertext string.

    key
        Short string for key.

        Duplicate characters are ignored and then it is padded with the
        characters in symbols that were not in the short key.

    symbols
        `n \times n` characters defining the alphabet.

        (default=string.printable, a `10 \times 10` matrix)

    Returns
    =======

    deciphered
        Deciphered text.

    Examples
    ========

    >>> from sympy.crypto.crypto import (
    ...     encipher_bifid, decipher_bifid, AZ)

    Do an encryption using the bifid5 alphabet:

    >>> alp = AZ().replace('J', '')
    >>> ct = AZ("meet me on monday!")
    >>> key = AZ("gold bug")
    >>> encipher_bifid(ct, key, alp)
    'IEILHHFSTSFQYE'

    When entering the text or ciphertext, spaces are ignored so it
    can be formatted as desired. Re-entering the ciphertext from the
    preceding, putting 4 characters per line and padding with an extra
    J, does not cause problems for the deciphering:

    >>> decipher_bifid('''
    ... IEILH
    ... HFSTS
    ... FQYEJ''', key, alp)
    'MEETMEONMONDAY'

    When no alphabet is given, all 100 printable characters will be
    used:

    >>> key = ''
    >>> encipher_bifid('hello world!', key)
    'bmtwmg-bIo*w'
    >>> decipher_bifid(_, key)
    'hello world!'

    If the key is changed, a different encryption is obtained:

    >>> key = 'gold bug'
    >>> encipher_bifid('hello world!', 'gold_bug')
    'hg2sfuei7t}w'

    And if the key used to decrypt the message is not exact, the
    original text will not be perfectly obtained:

    >>> decipher_bifid(_, 'gold pug')
    'heldo~wor6d!'

    """
    msg, _, A = _prep(msg, '', symbols, bifid10)
    long_key = ''.join(uniq(key)) or A

    n = len(A)**.5
    if n != int(n):
        raise ValueError(
            'Length of alphabet (%s) is not a square number.' % len(A))
    N = int(n)
    if len(long_key) < N**2:
        long_key = list(long_key) + [x for x in A if x not in long_key]

    # the reverse fractionalization
    row_col = {
        ch: divmod(i, N) for i, ch in enumerate(long_key)}
    rc = [i for c in msg for i in row_col[c]]
    n = len(msg)
    rc = zip(*(rc[:n], rc[n:]))
    ch = {i: ch for ch, i in row_col.items()}
    rv = ''.join(ch[i] for i in rc)
    return rv


def bifid_square(key):
    """Return characters of ``key`` arranged in a square.

    Examples
    ========

    >>> from sympy.crypto.crypto import (
    ...    bifid_square, AZ, padded_key, bifid5)
    >>> bifid_square(AZ().replace('J', ''))
    Matrix([
    [A, B, C, D, E],
    [F, G, H, I, K],
    [L, M, N, O, P],
    [Q, R, S, T, U],
    [V, W, X, Y, Z]])

    >>> bifid_square(padded_key(AZ('gold bug!'), bifid5))
    Matrix([
    [G, O, L, D, B],
    [U, A, C, E, F],
    [H, I, K, M, N],
    [P, Q, R, S, T],
    [V, W, X, Y, Z]])

    See Also
    ========

    padded_key

    """
    A = ''.join(uniq(''.join(key)))
    n = len(A)**.5
    if n != int(n):
        raise ValueError(
            'Length of alphabet (%s) is not a square number.' % len(A))
    n = int(n)
    f = lambda i, j: Symbol(A[n*i + j])
    rv = Matrix(n, n, f)
    return rv


def encipher_bifid5(msg, key):
    r"""
    Performs the Bifid cipher encryption on plaintext ``msg``, and
    returns the ciphertext.

    Explanation
    ===========

    This is the version of the Bifid cipher that uses the `5 \times 5`
    Polybius square. The letter "J" is ignored so it must be replaced
    with something else (traditionally an "I") before encryption.

    ALGORITHM: (5x5 case)

        STEPS:
            0. Create the `5 \times 5` Polybius square ``S`` associated
               to ``key`` as follows:

                a) moving from left-to-right, top-to-bottom,
                   place the letters of the key into a `5 \times 5`
                   matrix,
                b) if the key has less than 25 letters, add the
                   letters of the alphabet not in the key until the
                   `5 \times 5` square is filled.

            1. Create a list ``P`` of pairs of numbers which are the
               coordinates in the Polybius square of the letters in
               ``msg``.
            2. Let ``L1`` be the list of all first coordinates of ``P``
               (length of ``L1 = n``), let ``L2`` be the list of all
               second coordinates of ``P`` (so the length of ``L2``
               is also ``n``).
            3. Let ``L`` be the concatenation of ``L1`` and ``L2``
               (length ``L = 2*n``), except that consecutive numbers
               are paired ``(L[2*i], L[2*i + 1])``. You can regard
               ``L`` as a list of pairs of length ``n``.
            4. Let ``C`` be the list of all letters which are of the
               form ``S[i, j]``, for all ``(i, j)`` in ``L``. As a
               string, this is the ciphertext of ``msg``.

    Parameters
    ==========

    msg : str
        Plaintext string.

        Converted to upper case and filtered of anything but all letters
        except J.

    key
        Short string for key; non-alphabetic letters, J and duplicated
        characters are ignored and then, if the length is less than 25
        characters, it is padded with other letters of the alphabet
        (in alphabetical order).

    Returns
    =======

    ct
        Ciphertext (all caps, no spaces).

    Examples
    ========

    >>> from sympy.crypto.crypto import (
    ...     encipher_bifid5, decipher_bifid5)

    "J" will be omitted unless it is replaced with something else:

    >>> round_trip = lambda m, k: \
    ...     decipher_bifid5(encipher_bifid5(m, k), k)
    >>> key = 'a'
    >>> msg = "JOSIE"
    >>> round_trip(msg, key)
    'OSIE'
    >>> round_trip(msg.replace("J", "I"), key)
    'IOSIE'
    >>> j = "QIQ"
    >>> round_trip(msg.replace("J", j), key).replace(j, "J")
    'JOSIE'


    Notes
    =====

    The Bifid cipher was invented around 1901 by Felix Delastelle.
    It is a *fractional substitution* cipher, where letters are
    replaced by pairs of symbols from a smaller alphabet. The
    cipher uses a `5 \times 5` square filled with some ordering of the
    alphabet, except that "J" is replaced with "I" (this is a so-called
    Polybius square; there is a `6 \times 6` analog if you add back in
    "J" and also append onto the usual 26 letter alphabet, the digits
    0, 1, ..., 9).
    According to Helen Gaines' book *Cryptanalysis*, this type of cipher
    was used in the field by the German Army during World War I.

    See Also
    ========

    decipher_bifid5, encipher_bifid

    """
    msg, key, _ = _prep(msg.upper(), key.upper(), None, bifid5)
    key = padded_key(key, bifid5)
    return encipher_bifid(msg, '', key)


def decipher_bifid5(msg, key):
    r"""
    Return the Bifid cipher decryption of ``msg``.

    Explanation
    ===========

    This is the version of the Bifid cipher that uses the `5 \times 5`
    Polybius square; the letter "J" is ignored unless a ``key`` of
    length 25 is used.

    Parameters
    ==========

    msg
        Ciphertext string.

    key
        Short string for key; duplicated characters are ignored and if
        the length is less then 25 characters, it will be padded with
        other letters from the alphabet omitting "J".
        Non-alphabetic characters are ignored.

    Returns
    =======

    plaintext
        Plaintext from Bifid5 cipher (all caps, no spaces).

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_bifid5, decipher_bifid5
    >>> key = "gold bug"
    >>> encipher_bifid5('meet me on friday', key)
    'IEILEHFSTSFXEE'
    >>> encipher_bifid5('meet me on monday', key)
    'IEILHHFSTSFQYE'
    >>> decipher_bifid5(_, key)
    'MEETMEONMONDAY'

    """
    msg, key, _ = _prep(msg.upper(), key.upper(), None, bifid5)
    key = padded_key(key, bifid5)
    return decipher_bifid(msg, '', key)


def bifid5_square(key=None):
    r"""
    5x5 Polybius square.

    Produce the Polybius square for the `5 \times 5` Bifid cipher.

    Examples
    ========

    >>> from sympy.crypto.crypto import bifid5_square
    >>> bifid5_square("gold bug")
    Matrix([
    [G, O, L, D, B],
    [U, A, C, E, F],
    [H, I, K, M, N],
    [P, Q, R, S, T],
    [V, W, X, Y, Z]])

    """
    if not key:
        key = bifid5
    else:
        _, key, _ = _prep('', key.upper(), None, bifid5)
        key = padded_key(key, bifid5)
    return bifid_square(key)


def encipher_bifid6(msg, key):
    r"""
    Performs the Bifid cipher encryption on plaintext ``msg``, and
    returns the ciphertext.

    This is the version of the Bifid cipher that uses the `6 \times 6`
    Polybius square.

    Parameters
    ==========

    msg
        Plaintext string (digits okay).

    key
        Short string for key (digits okay).

        If ``key`` is less than 36 characters long, the square will be
        filled with letters A through Z and digits 0 through 9.

    Returns
    =======

    ciphertext
        Ciphertext from Bifid cipher (all caps, no spaces).

    See Also
    ========

    decipher_bifid6, encipher_bifid

    """
    msg, key, _ = _prep(msg.upper(), key.upper(), None, bifid6)
    key = padded_key(key, bifid6)
    return encipher_bifid(msg, '', key)


def decipher_bifid6(msg, key):
    r"""
    Performs the Bifid cipher decryption on ciphertext ``msg``, and
    returns the plaintext.

    This is the version of the Bifid cipher that uses the `6 \times 6`
    Polybius square.

    Parameters
    ==========

    msg
        Ciphertext string (digits okay); converted to upper case

    key
        Short string for key (digits okay).

        If ``key`` is less than 36 characters long, the square will be
        filled with letters A through Z and digits 0 through 9.
        All letters are converted to uppercase.

    Returns
    =======

    plaintext
        Plaintext from Bifid cipher (all caps, no spaces).

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_bifid6, decipher_bifid6
    >>> key = "gold bug"
    >>> encipher_bifid6('meet me on monday at 8am', key)
    'KFKLJJHF5MMMKTFRGPL'
    >>> decipher_bifid6(_, key)
    'MEETMEONMONDAYAT8AM'

    """
    msg, key, _ = _prep(msg.upper(), key.upper(), None, bifid6)
    key = padded_key(key, bifid6)
    return decipher_bifid(msg, '', key)


def bifid6_square(key=None):
    r"""
    6x6 Polybius square.

    Produces the Polybius square for the `6 \times 6` Bifid cipher.
    Assumes alphabet of symbols is "A", ..., "Z", "0", ..., "9".

    Examples
    ========

    >>> from sympy.crypto.crypto import bifid6_square
    >>> key = "gold bug"
    >>> bifid6_square(key)
    Matrix([
    [G, O, L, D, B, U],
    [A, C, E, F, H, I],
    [J, K, M, N, P, Q],
    [R, S, T, V, W, X],
    [Y, Z, 0, 1, 2, 3],
    [4, 5, 6, 7, 8, 9]])

    """
    if not key:
        key = bifid6
    else:
        _, key, _ = _prep('', key.upper(), None, bifid6)
        key = padded_key(key, bifid6)
    return bifid_square(key)


#################### RSA  #############################

def _decipher_rsa_crt(i, d, factors):
    """Decipher RSA using chinese remainder theorem from the information
    of the relatively-prime factors of the modulus.

    Parameters
    ==========

    i : integer
        Ciphertext

    d : integer
        The exponent component.

    factors : list of relatively-prime integers
        The integers given must be coprime and the product must equal
        the modulus component of the original RSA key.

    Examples
    ========

    How to decrypt RSA with CRT:

    >>> from sympy.crypto.crypto import rsa_public_key, rsa_private_key
    >>> primes = [61, 53]
    >>> e = 17
    >>> args = primes + [e]
    >>> puk = rsa_public_key(*args)
    >>> prk = rsa_private_key(*args)

    >>> from sympy.crypto.crypto import encipher_rsa, _decipher_rsa_crt
    >>> msg = 65
    >>> crt_primes = primes
    >>> encrypted = encipher_rsa(msg, puk)
    >>> decrypted = _decipher_rsa_crt(encrypted, prk[1], primes)
    >>> decrypted
    65
    """
    moduluses = [pow(i, d, p) for p in factors]

    result = crt(factors, moduluses)
    if not result:
        raise ValueError("CRT failed")
    return result[0]


def _rsa_key(*args, public=True, private=True, totient='Euler', index=None, multipower=None):
    r"""A private subroutine to generate RSA key

    Parameters
    ==========

    public, private : bool, optional
        Flag to generate either a public key, a private key.

    totient : 'Euler' or 'Carmichael'
        Different notation used for totient.

    multipower : bool, optional
        Flag to bypass warning for multipower RSA.
    """

    if len(args) < 2:
        return False

    if totient not in ('Euler', 'Carmichael'):
        raise ValueError(
            "The argument totient={} should either be " \
            "'Euler', 'Carmichalel'." \
            .format(totient))

    if totient == 'Euler':
        _totient = _euler
    else:
        _totient = _carmichael

    if index is not None:
        index = as_int(index)
        if totient != 'Carmichael':
            raise ValueError(
                "Setting the 'index' keyword argument requires totient"
                "notation to be specified as 'Carmichael'.")

    primes, e = args[:-1], args[-1]

    if not all(isprime(p) for p in primes):
        new_primes = []
        for i in primes:
            new_primes.extend(factorint(i, multiple=True))
        primes = new_primes

    n = reduce(lambda i, j: i*j, primes)

    tally = multiset(primes)
    if all(v == 1 for v in tally.values()):
        phi = int(_totient(tally))

    else:
        if not multipower:
            NonInvertibleCipherWarning(
                'Non-distinctive primes found in the factors {}. '
                'The cipher may not be decryptable for some numbers '
                'in the complete residue system Z[{}], but the cipher '
                'can still be valid if you restrict the domain to be '
                'the reduced residue system Z*[{}]. You can pass '
                'the flag multipower=True if you want to suppress this '
                'warning.'
                .format(primes, n, n)
                # stacklevel=4 because most users will call a function that
                # calls this function
                ).warn(stacklevel=4)
        phi = int(_totient(tally))

    if gcd(e, phi) == 1:
        if public and not private:
            if isinstance(index, int):
                e = e % phi
                e += index * phi
            return n, e

        if private and not public:
            d = invert(e, phi)
            if isinstance(index, int):
                d += index * phi
            return n, d

    return False


def rsa_public_key(*args, **kwargs):
    r"""Return the RSA *public key* pair, `(n, e)`

    Parameters
    ==========

    args : naturals
        If specified as `p, q, e` where `p` and `q` are distinct primes
        and `e` is a desired public exponent of the RSA, `n = p q` and
        `e` will be verified against the totient
        `\phi(n)` (Euler totient) or `\lambda(n)` (Carmichael totient)
        to be `\gcd(e, \phi(n)) = 1` or `\gcd(e, \lambda(n)) = 1`.

        If specified as `p_1, p_2, \dots, p_n, e` where
        `p_1, p_2, \dots, p_n` are specified as primes,
        and `e` is specified as a desired public exponent of the RSA,
        it will be able to form a multi-prime RSA, which is a more
        generalized form of the popular 2-prime RSA.

        It can also be possible to form a single-prime RSA by specifying
        the argument as `p, e`, which can be considered a trivial case
        of a multiprime RSA.

        Furthermore, it can be possible to form a multi-power RSA by
        specifying two or more pairs of the primes to be same.
        However, unlike the two-distinct prime RSA or multi-prime
        RSA, not every numbers in the complete residue system
        (`\mathbb{Z}_n`) will be decryptable since the mapping
        `\mathbb{Z}_{n} \rightarrow \mathbb{Z}_{n}`
        will not be bijective.
        (Only except for the trivial case when
        `e = 1`
        or more generally,

        .. math::
            e \in \left \{ 1 + k \lambda(n)
            \mid k \in \mathbb{Z} \land k \geq 0 \right \}

        when RSA reduces to the identity.)
        However, the RSA can still be decryptable for the numbers in the
        reduced residue system (`\mathbb{Z}_n^{\times}`), since the
        mapping
        `\mathbb{Z}_{n}^{\times} \rightarrow \mathbb{Z}_{n}^{\times}`
        can still be bijective.

        If you pass a non-prime integer to the arguments
        `p_1, p_2, \dots, p_n`, the particular number will be
        prime-factored and it will become either a multi-prime RSA or a
        multi-power RSA in its canonical form, depending on whether the
        product equals its radical or not.
        `p_1 p_2 \dots p_n = \text{rad}(p_1 p_2 \dots p_n)`

    totient : bool, optional
        If ``'Euler'``, it uses Euler's totient `\phi(n)` which is
        :meth:`sympy.functions.combinatorial.numbers.totient` in SymPy.

        If ``'Carmichael'``, it uses Carmichael's totient `\lambda(n)`
        which is :meth:`sympy.functions.combinatorial.numbers.reduced_totient` in SymPy.

        Unlike private key generation, this is a trivial keyword for
        public key generation because
        `\gcd(e, \phi(n)) = 1 \iff \gcd(e, \lambda(n)) = 1`.

    index : nonnegative integer, optional
        Returns an arbitrary solution of a RSA public key at the index
        specified at `0, 1, 2, \dots`. This parameter needs to be
        specified along with ``totient='Carmichael'``.

        Similarly to the non-uniquenss of a RSA private key as described
        in the ``index`` parameter documentation in
        :meth:`rsa_private_key`, RSA public key is also not unique and
        there is an infinite number of RSA public exponents which
        can behave in the same manner.

        From any given RSA public exponent `e`, there are can be an
        another RSA public exponent `e + k \lambda(n)` where `k` is an
        integer, `\lambda` is a Carmichael's totient function.

        However, considering only the positive cases, there can be
        a principal solution of a RSA public exponent `e_0` in
        `0 < e_0 < \lambda(n)`, and all the other solutions
        can be canonicalzed in a form of `e_0 + k \lambda(n)`.

        ``index`` specifies the `k` notation to yield any possible value
        an RSA public key can have.

        An example of computing any arbitrary RSA public key:

        >>> from sympy.crypto.crypto import rsa_public_key
        >>> rsa_public_key(61, 53, 17, totient='Carmichael', index=0)
        (3233, 17)
        >>> rsa_public_key(61, 53, 17, totient='Carmichael', index=1)
        (3233, 797)
        >>> rsa_public_key(61, 53, 17, totient='Carmichael', index=2)
        (3233, 1577)

    multipower : bool, optional
        Any pair of non-distinct primes found in the RSA specification
        will restrict the domain of the cryptosystem, as noted in the
        explanation of the parameter ``args``.

        SymPy RSA key generator may give a warning before dispatching it
        as a multi-power RSA, however, you can disable the warning if
        you pass ``True`` to this keyword.

    Returns
    =======

    (n, e) : int, int
        `n` is a product of any arbitrary number of primes given as
        the argument.

        `e` is relatively prime (coprime) to the Euler totient
        `\phi(n)`.

    False
        Returned if less than two arguments are given, or `e` is
        not relatively prime to the modulus.

    Examples
    ========

    >>> from sympy.crypto.crypto import rsa_public_key

    A public key of a two-prime RSA:

    >>> p, q, e = 3, 5, 7
    >>> rsa_public_key(p, q, e)
    (15, 7)
    >>> rsa_public_key(p, q, 30)
    False

    A public key of a multiprime RSA:

    >>> primes = [2, 3, 5, 7, 11, 13]
    >>> e = 7
    >>> args = primes + [e]
    >>> rsa_public_key(*args)
    (30030, 7)

    Notes
    =====

    Although the RSA can be generalized over any modulus `n`, using
    two large primes had became the most popular specification because a
    product of two large primes is usually the hardest to factor
    relatively to the digits of `n` can have.

    However, it may need further understanding of the time complexities
    of each prime-factoring algorithms to verify the claim.

    See Also
    ========

    rsa_private_key
    encipher_rsa
    decipher_rsa

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/RSA_%28cryptosystem%29

    .. [2] https://cacr.uwaterloo.ca/techreports/2006/cacr2006-16.pdf

    .. [3] https://link.springer.com/content/pdf/10.1007/BFb0055738.pdf

    .. [4] https://www.itiis.org/digital-library/manuscript/1381
    """
    return _rsa_key(*args, public=True, private=False, **kwargs)


def rsa_private_key(*args, **kwargs):
    r"""Return the RSA *private key* pair, `(n, d)`

    Parameters
    ==========

    args : naturals
        The keyword is identical to the ``args`` in
        :meth:`rsa_public_key`.

    totient : bool, optional
        If ``'Euler'``, it uses Euler's totient convention `\phi(n)`
        which is :meth:`sympy.functions.combinatorial.numbers.totient` in SymPy.

        If ``'Carmichael'``, it uses Carmichael's totient convention
        `\lambda(n)` which is
        :meth:`sympy.functions.combinatorial.numbers.reduced_totient` in SymPy.

        There can be some output differences for private key generation
        as examples below.

        Example using Euler's totient:

        >>> from sympy.crypto.crypto import rsa_private_key
        >>> rsa_private_key(61, 53, 17, totient='Euler')
        (3233, 2753)

        Example using Carmichael's totient:

        >>> from sympy.crypto.crypto import rsa_private_key
        >>> rsa_private_key(61, 53, 17, totient='Carmichael')
        (3233, 413)

    index : nonnegative integer, optional
        Returns an arbitrary solution of a RSA private key at the index
        specified at `0, 1, 2, \dots`. This parameter needs to be
        specified along with ``totient='Carmichael'``.

        RSA private exponent is a non-unique solution of
        `e d \mod \lambda(n) = 1` and it is possible in any form of
        `d + k \lambda(n)`, where `d` is an another
        already-computed private exponent, and `\lambda` is a
        Carmichael's totient function, and `k` is any integer.

        However, considering only the positive cases, there can be
        a principal solution of a RSA private exponent `d_0` in
        `0 < d_0 < \lambda(n)`, and all the other solutions
        can be canonicalzed in a form of `d_0 + k \lambda(n)`.

        ``index`` specifies the `k` notation to yield any possible value
        an RSA private key can have.

        An example of computing any arbitrary RSA private key:

        >>> from sympy.crypto.crypto import rsa_private_key
        >>> rsa_private_key(61, 53, 17, totient='Carmichael', index=0)
        (3233, 413)
        >>> rsa_private_key(61, 53, 17, totient='Carmichael', index=1)
        (3233, 1193)
        >>> rsa_private_key(61, 53, 17, totient='Carmichael', index=2)
        (3233, 1973)

    multipower : bool, optional
        The keyword is identical to the ``multipower`` in
        :meth:`rsa_public_key`.

    Returns
    =======

    (n, d) : int, int
        `n` is a product of any arbitrary number of primes given as
        the argument.

        `d` is the inverse of `e` (mod `\phi(n)`) where `e` is the
        exponent given, and `\phi` is a Euler totient.

    False
        Returned if less than two arguments are given, or `e` is
        not relatively prime to the totient of the modulus.

    Examples
    ========

    >>> from sympy.crypto.crypto import rsa_private_key

    A private key of a two-prime RSA:

    >>> p, q, e = 3, 5, 7
    >>> rsa_private_key(p, q, e)
    (15, 7)
    >>> rsa_private_key(p, q, 30)
    False

    A private key of a multiprime RSA:

    >>> primes = [2, 3, 5, 7, 11, 13]
    >>> e = 7
    >>> args = primes + [e]
    >>> rsa_private_key(*args)
    (30030, 823)

    See Also
    ========

    rsa_public_key
    encipher_rsa
    decipher_rsa

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/RSA_%28cryptosystem%29

    .. [2] https://cacr.uwaterloo.ca/techreports/2006/cacr2006-16.pdf

    .. [3] https://link.springer.com/content/pdf/10.1007/BFb0055738.pdf

    .. [4] https://www.itiis.org/digital-library/manuscript/1381
    """
    return _rsa_key(*args, public=False, private=True, **kwargs)


def _encipher_decipher_rsa(i, key, factors=None):
    n, d = key
    if not factors:
        return pow(i, d, n)

    def _is_coprime_set(l):
        is_coprime_set = True
        for i in range(len(l)):
            for j in range(i+1, len(l)):
                if gcd(l[i], l[j]) != 1:
                    is_coprime_set = False
                    break
        return is_coprime_set

    prod = reduce(lambda i, j: i*j, factors)
    if prod == n and _is_coprime_set(factors):
        return _decipher_rsa_crt(i, d, factors)
    return _encipher_decipher_rsa(i, key, factors=None)


def encipher_rsa(i, key, factors=None):
    r"""Encrypt the plaintext with RSA.

    Parameters
    ==========

    i : integer
        The plaintext to be encrypted for.

    key : (n, e) where n, e are integers
        `n` is the modulus of the key and `e` is the exponent of the
        key. The encryption is computed by `i^e \bmod n`.

        The key can either be a public key or a private key, however,
        the message encrypted by a public key can only be decrypted by
        a private key, and vice versa, as RSA is an asymmetric
        cryptography system.

    factors : list of coprime integers
        This is identical to the keyword ``factors`` in
        :meth:`decipher_rsa`.

    Notes
    =====

    Some specifications may make the RSA not cryptographically
    meaningful.

    For example, `0`, `1` will remain always same after taking any
    number of exponentiation, thus, should be avoided.

    Furthermore, if `i^e < n`, `i` may easily be figured out by taking
    `e` th root.

    And also, specifying the exponent as `1` or in more generalized form
    as `1 + k \lambda(n)` where `k` is an nonnegative integer,
    `\lambda` is a carmichael totient, the RSA becomes an identity
    mapping.

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_rsa
    >>> from sympy.crypto.crypto import rsa_public_key, rsa_private_key

    Public Key Encryption:

    >>> p, q, e = 3, 5, 7
    >>> puk = rsa_public_key(p, q, e)
    >>> msg = 12
    >>> encipher_rsa(msg, puk)
    3

    Private Key Encryption:

    >>> p, q, e = 3, 5, 7
    >>> prk = rsa_private_key(p, q, e)
    >>> msg = 12
    >>> encipher_rsa(msg, prk)
    3

    Encryption using chinese remainder theorem:

    >>> encipher_rsa(msg, prk, factors=[p, q])
    3
    """
    return _encipher_decipher_rsa(i, key, factors=factors)


def decipher_rsa(i, key, factors=None):
    r"""Decrypt the ciphertext with RSA.

    Parameters
    ==========

    i : integer
        The ciphertext to be decrypted for.

    key : (n, d) where n, d are integers
        `n` is the modulus of the key and `d` is the exponent of the
        key. The decryption is computed by `i^d \bmod n`.

        The key can either be a public key or a private key, however,
        the message encrypted by a public key can only be decrypted by
        a private key, and vice versa, as RSA is an asymmetric
        cryptography system.

    factors : list of coprime integers
        As the modulus `n` created from RSA key generation is composed
        of arbitrary prime factors
        `n = {p_1}^{k_1}{p_2}^{k_2}\dots{p_n}^{k_n}` where
        `p_1, p_2, \dots, p_n` are distinct primes and
        `k_1, k_2, \dots, k_n` are positive integers, chinese remainder
        theorem can be used to compute `i^d \bmod n` from the
        fragmented modulo operations like

        .. math::
            i^d \bmod {p_1}^{k_1}, i^d \bmod {p_2}^{k_2}, \dots,
            i^d \bmod {p_n}^{k_n}

        or like

        .. math::
            i^d \bmod {p_1}^{k_1}{p_2}^{k_2},
            i^d \bmod {p_3}^{k_3}, \dots ,
            i^d \bmod {p_n}^{k_n}

        as long as every moduli does not share any common divisor each
        other.

        The raw primes used in generating the RSA key pair can be a good
        option.

        Note that the speed advantage of using this is only viable for
        very large cases (Like 2048-bit RSA keys) since the
        overhead of using pure Python implementation of
        :meth:`sympy.ntheory.modular.crt` may overcompensate the
        theoretical speed advantage.

    Notes
    =====

    See the ``Notes`` section in the documentation of
    :meth:`encipher_rsa`

    Examples
    ========

    >>> from sympy.crypto.crypto import decipher_rsa, encipher_rsa
    >>> from sympy.crypto.crypto import rsa_public_key, rsa_private_key

    Public Key Encryption and Decryption:

    >>> p, q, e = 3, 5, 7
    >>> prk = rsa_private_key(p, q, e)
    >>> puk = rsa_public_key(p, q, e)
    >>> msg = 12
    >>> new_msg = encipher_rsa(msg, prk)
    >>> new_msg
    3
    >>> decipher_rsa(new_msg, puk)
    12

    Private Key Encryption and Decryption:

    >>> p, q, e = 3, 5, 7
    >>> prk = rsa_private_key(p, q, e)
    >>> puk = rsa_public_key(p, q, e)
    >>> msg = 12
    >>> new_msg = encipher_rsa(msg, puk)
    >>> new_msg
    3
    >>> decipher_rsa(new_msg, prk)
    12

    Decryption using chinese remainder theorem:

    >>> decipher_rsa(new_msg, prk, factors=[p, q])
    12

    See Also
    ========

    encipher_rsa
    """
    return _encipher_decipher_rsa(i, key, factors=factors)


#################### kid krypto (kid RSA) #############################


def kid_rsa_public_key(a, b, A, B):
    r"""
    Kid RSA is a version of RSA useful to teach grade school children
    since it does not involve exponentiation.

    Explanation
    ===========

    Alice wants to talk to Bob. Bob generates keys as follows.
    Key generation:

    * Select positive integers `a, b, A, B` at random.
    * Compute `M = a b - 1`, `e = A M + a`, `d = B M + b`,
      `n = (e d - 1)//M`.
    * The *public key* is `(n, e)`. Bob sends these to Alice.
    * The *private key* is `(n, d)`, which Bob keeps secret.

    Encryption: If `p` is the plaintext message then the
    ciphertext is `c = p e \pmod n`.

    Decryption: If `c` is the ciphertext message then the
    plaintext is `p = c d \pmod n`.

    Examples
    ========

    >>> from sympy.crypto.crypto import kid_rsa_public_key
    >>> a, b, A, B = 3, 4, 5, 6
    >>> kid_rsa_public_key(a, b, A, B)
    (369, 58)

    """
    M = a*b - 1
    e = A*M + a
    d = B*M + b
    n = (e*d - 1)//M
    return n, e


def kid_rsa_private_key(a, b, A, B):
    """
    Compute `M = a b - 1`, `e = A M + a`, `d = B M + b`,
    `n = (e d - 1) / M`. The *private key* is `d`, which Bob
    keeps secret.

    Examples
    ========

    >>> from sympy.crypto.crypto import kid_rsa_private_key
    >>> a, b, A, B = 3, 4, 5, 6
    >>> kid_rsa_private_key(a, b, A, B)
    (369, 70)

    """
    M = a*b - 1
    e = A*M + a
    d = B*M + b
    n = (e*d - 1)//M
    return n, d


def encipher_kid_rsa(msg, key):
    """
    Here ``msg`` is the plaintext and ``key`` is the public key.

    Examples
    ========

    >>> from sympy.crypto.crypto import (
    ...     encipher_kid_rsa, kid_rsa_public_key)
    >>> msg = 200
    >>> a, b, A, B = 3, 4, 5, 6
    >>> key = kid_rsa_public_key(a, b, A, B)
    >>> encipher_kid_rsa(msg, key)
    161

    """
    n, e = key
    return (msg*e) % n


def decipher_kid_rsa(msg, key):
    """
    Here ``msg`` is the plaintext and ``key`` is the private key.

    Examples
    ========

    >>> from sympy.crypto.crypto import (
    ...     kid_rsa_public_key, kid_rsa_private_key,
    ...     decipher_kid_rsa, encipher_kid_rsa)
    >>> a, b, A, B = 3, 4, 5, 6
    >>> d = kid_rsa_private_key(a, b, A, B)
    >>> msg = 200
    >>> pub = kid_rsa_public_key(a, b, A, B)
    >>> pri = kid_rsa_private_key(a, b, A, B)
    >>> ct = encipher_kid_rsa(msg, pub)
    >>> decipher_kid_rsa(ct, pri)
    200

    """
    n, d = key
    return (msg*d) % n


#################### Morse Code ######################################

morse_char = {
    ".-": "A", "-...": "B",
    "-.-.": "C", "-..": "D",
    ".": "E", "..-.": "F",
    "--.": "G", "....": "H",
    "..": "I", ".---": "J",
    "-.-": "K", ".-..": "L",
    "--": "M", "-.": "N",
    "---": "O", ".--.": "P",
    "--.-": "Q", ".-.": "R",
    "...": "S", "-": "T",
    "..-": "U", "...-": "V",
    ".--": "W", "-..-": "X",
    "-.--": "Y", "--..": "Z",
    "-----": "0", ".----": "1",
    "..---": "2", "...--": "3",
    "....-": "4", ".....": "5",
    "-....": "6", "--...": "7",
    "---..": "8", "----.": "9",
    ".-.-.-": ".", "--..--": ",",
    "---...": ":", "-.-.-.": ";",
    "..--..": "?", "-....-": "-",
    "..--.-": "_", "-.--.": "(",
    "-.--.-": ")", ".----.": "'",
    "-...-": "=", ".-.-.": "+",
    "-..-.": "/", ".--.-.": "@",
    "...-..-": "$", "-.-.--": "!"}
char_morse = {v: k for k, v in morse_char.items()}


def encode_morse(msg, sep='|', mapping=None):
    """
    Encodes a plaintext into popular Morse Code with letters
    separated by ``sep`` and words by a double ``sep``.

    Examples
    ========

    >>> from sympy.crypto.crypto import encode_morse
    >>> msg = 'ATTACK RIGHT FLANK'
    >>> encode_morse(msg)
    '.-|-|-|.-|-.-.|-.-||.-.|..|--.|....|-||..-.|.-..|.-|-.|-.-'

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Morse_code

    """

    mapping = mapping or char_morse
    assert sep not in mapping
    word_sep = 2*sep
    mapping[" "] = word_sep
    suffix = msg and msg[-1] in whitespace

    # normalize whitespace
    msg = (' ' if word_sep else '').join(msg.split())
    # omit unmapped chars
    chars = set(''.join(msg.split()))
    ok = set(mapping.keys())
    msg = translate(msg, None, ''.join(chars - ok))

    morsestring = []
    words = msg.split()
    for word in words:
        morseword = []
        for letter in word:
            morseletter = mapping[letter]
            morseword.append(morseletter)

        word = sep.join(morseword)
        morsestring.append(word)

    return word_sep.join(morsestring) + (word_sep if suffix else '')


def decode_morse(msg, sep='|', mapping=None):
    """
    Decodes a Morse Code with letters separated by ``sep``
    (default is '|') and words by `word_sep` (default is '||)
    into plaintext.

    Examples
    ========

    >>> from sympy.crypto.crypto import decode_morse
    >>> mc = '--|---|...-|.||.|.-|...|-'
    >>> decode_morse(mc)
    'MOVE EAST'

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Morse_code

    """

    mapping = mapping or morse_char
    word_sep = 2*sep
    characterstring = []
    words = msg.strip(word_sep).split(word_sep)
    for word in words:
        letters = word.split(sep)
        chars = [mapping[c] for c in letters]
        word = ''.join(chars)
        characterstring.append(word)
    rv = " ".join(characterstring)
    return rv


#################### LFSRs  ##########################################


@doctest_depends_on(ground_types=['python', 'gmpy'])
def lfsr_sequence(key, fill, n):
    r"""
    This function creates an LFSR sequence.

    Parameters
    ==========

    key : list
        A list of finite field elements, `[c_0, c_1, \ldots, c_k].`

    fill : list
        The list of the initial terms of the LFSR sequence,
        `[x_0, x_1, \ldots, x_k].`

    n
        Number of terms of the sequence that the function returns.

    Returns
    =======

    L
        The LFSR sequence defined by
        `x_{n+1} = c_k x_n + \ldots + c_0 x_{n-k}`, for
        `n \leq k`.

    Notes
    =====

    S. Golomb [G]_ gives a list of three statistical properties a
    sequence of numbers `a = \{a_n\}_{n=1}^\infty`,
    `a_n \in \{0,1\}`, should display to be considered
    "random". Define the autocorrelation of `a` to be

    .. math::

        C(k) = C(k,a) = \lim_{N\rightarrow \infty} {1\over N}\sum_{n=1}^N (-1)^{a_n + a_{n+k}}.

    In the case where `a` is periodic with period
    `P` then this reduces to

    .. math::

        C(k) = {1\over P}\sum_{n=1}^P (-1)^{a_n + a_{n+k}}.

    Assume `a` is periodic with period `P`.

    - balance:

      .. math::

        \left|\sum_{n=1}^P(-1)^{a_n}\right| \leq 1.

    - low autocorrelation:

       .. math::

         C(k) = \left\{ \begin{array}{cc} 1,& k = 0,\\ \epsilon, & k \ne 0. \end{array} \right.

      (For sequences satisfying these first two properties, it is known
      that `\epsilon = -1/P` must hold.)

    - proportional runs property: In each period, half the runs have
      length `1`, one-fourth have length `2`, etc.
      Moreover, there are as many runs of `1`'s as there are of
      `0`'s.

    Examples
    ========

    >>> from sympy.crypto.crypto import lfsr_sequence
    >>> from sympy.polys.domains import FF
    >>> F = FF(2)
    >>> fill = [F(1), F(1), F(0), F(1)]
    >>> key = [F(1), F(0), F(0), F(1)]
    >>> lfsr_sequence(key, fill, 10)
    [1 mod 2, 1 mod 2, 0 mod 2, 1 mod 2, 0 mod 2,
    1 mod 2, 1 mod 2, 0 mod 2, 0 mod 2, 1 mod 2]

    References
    ==========

    .. [G] Solomon Golomb, Shift register sequences, Aegean Park Press,
       Laguna Hills, Ca, 1967

    """
    if not isinstance(key, list):
        raise TypeError("key must be a list")
    if not isinstance(fill, list):
        raise TypeError("fill must be a list")
    p = key[0].modulus()
    F = FF(p)
    s = fill
    k = len(fill)
    L = []
    for i in range(n):
        s0 = s[:]
        L.append(s[0])
        s = s[1:k]
        x = sum(int(key[i]*s0[i]) for i in range(k))
        s.append(F(x))
    return L       # use [int(x) for x in L] for int version


def lfsr_autocorrelation(L, P, k):
    """
    This function computes the LFSR autocorrelation function.

    Parameters
    ==========

    L
        A periodic sequence of elements of `GF(2)`.
        L must have length larger than P.

    P
        The period of L.

    k : int
        An integer `k` (`0 < k < P`).

    Returns
    =======

    autocorrelation
        The k-th value of the autocorrelation of the LFSR L.

    Examples
    ========

    >>> from sympy.crypto.crypto import (
    ...     lfsr_sequence, lfsr_autocorrelation)
    >>> from sympy.polys.domains import FF
    >>> F = FF(2)
    >>> fill = [F(1), F(1), F(0), F(1)]
    >>> key = [F(1), F(0), F(0), F(1)]
    >>> s = lfsr_sequence(key, fill, 20)
    >>> lfsr_autocorrelation(s, 15, 7)
    -1/15
    >>> lfsr_autocorrelation(s, 15, 0)
    1

    """
    if not isinstance(L, list):
        raise TypeError("L (=%s) must be a list" % L)
    P = int(P)
    k = int(k)
    L0 = L[:P]     # slices makes a copy
    L1 = L0 + L0[:k]
    L2 = [(-1)**(int(L1[i]) + int(L1[i + k])) for i in range(P)]
    tot = sum(L2)
    return Rational(tot, P)


def lfsr_connection_polynomial(s):
    """
    This function computes the LFSR connection polynomial.

    Parameters
    ==========

    s
        A sequence of elements of even length, with entries in a finite
        field.

    Returns
    =======

    C(x)
        The connection polynomial of a minimal LFSR yielding s.

        This implements the algorithm in section 3 of J. L. Massey's
        article [M]_.

    Examples
    ========

    >>> from sympy.crypto.crypto import (
    ...     lfsr_sequence, lfsr_connection_polynomial)
    >>> from sympy.polys.domains import FF
    >>> F = FF(2)
    >>> fill = [F(1), F(1), F(0), F(1)]
    >>> key = [F(1), F(0), F(0), F(1)]
    >>> s = lfsr_sequence(key, fill, 20)
    >>> lfsr_connection_polynomial(s)
    x**4 + x + 1
    >>> fill = [F(1), F(0), F(0), F(1)]
    >>> key = [F(1), F(1), F(0), F(1)]
    >>> s = lfsr_sequence(key, fill, 20)
    >>> lfsr_connection_polynomial(s)
    x**3 + 1
    >>> fill = [F(1), F(0), F(1)]
    >>> key = [F(1), F(1), F(0)]
    >>> s = lfsr_sequence(key, fill, 20)
    >>> lfsr_connection_polynomial(s)
    x**3 + x**2 + 1
    >>> fill = [F(1), F(0), F(1)]
    >>> key = [F(1), F(0), F(1)]
    >>> s = lfsr_sequence(key, fill, 20)
    >>> lfsr_connection_polynomial(s)
    x**3 + x + 1

    References
    ==========

    .. [M] James L. Massey, "Shift-Register Synthesis and BCH Decoding."
        IEEE Trans. on Information Theory, vol. 15(1), pp. 122-127,
        Jan 1969.

    """
    # Initialization:
    p = s[0].modulus()
    x = Symbol("x")
    C = 1*x**0
    B = 1*x**0
    m = 1
    b = 1*x**0
    L = 0
    N = 0
    while N < len(s):
        if L > 0:
            dC = Poly(C).degree()
            r = min(L + 1, dC + 1)
            coeffsC = [C.subs(x, 0)] + [C.coeff(x**i)
                for i in range(1, dC + 1)]
            d = (int(s[N]) + sum(coeffsC[i]*int(s[N - i])
                for i in range(1, r))) % p
        if L == 0:
            d = int(s[N])*x**0
        if d == 0:
            m += 1
            N += 1
        if d > 0:
            if 2*L > N:
                C = (C - d*((b**(p - 2)) % p)*x**m*B).expand()
                m += 1
                N += 1
            else:
                T = C
                C = (C - d*((b**(p - 2)) % p)*x**m*B).expand()
                L = N + 1 - L
                m = 1
                b = d
                B = T
                N += 1
    dC = Poly(C).degree()
    coeffsC = [C.subs(x, 0)] + [C.coeff(x**i) for i in range(1, dC + 1)]
    return sum(coeffsC[i] % p*x**i for i in range(dC + 1)
        if coeffsC[i] is not None)


#################### ElGamal  #############################


def elgamal_private_key(digit=10, seed=None):
    r"""
    Return three number tuple as private key.

    Explanation
    ===========

    Elgamal encryption is based on the mathematical problem
    called the Discrete Logarithm Problem (DLP). For example,

    `a^{b} \equiv c \pmod p`

    In general, if ``a`` and ``b`` are known, ``ct`` is easily
    calculated. If ``b`` is unknown, it is hard to use
    ``a`` and ``ct`` to get ``b``.

    Parameters
    ==========

    digit : int
        Minimum number of binary digits for key.

    Returns
    =======

    tuple : (p, r, d)
        p = prime number.

        r = primitive root.

        d = random number.

    Notes
    =====

    For testing purposes, the ``seed`` parameter may be set to control
    the output of this routine. See sympy.core.random._randrange.

    Examples
    ========

    >>> from sympy.crypto.crypto import elgamal_private_key
    >>> from sympy.ntheory import is_primitive_root, isprime
    >>> a, b, _ = elgamal_private_key()
    >>> isprime(a)
    True
    >>> is_primitive_root(b, a)
    True

    """
    randrange = _randrange(seed)
    p = nextprime(2**digit)
    return p, primitive_root(p), randrange(2, p)


def elgamal_public_key(key):
    r"""
    Return three number tuple as public key.

    Parameters
    ==========

    key : (p, r, e)
        Tuple generated by ``elgamal_private_key``.

    Returns
    =======

    tuple : (p, r, e)
        `e = r**d \bmod p`

        `d` is a random number in private key.

    Examples
    ========

    >>> from sympy.crypto.crypto import elgamal_public_key
    >>> elgamal_public_key((1031, 14, 636))
    (1031, 14, 212)

    """
    p, r, e = key
    return p, r, pow(r, e, p)


def encipher_elgamal(i, key, seed=None):
    r"""
    Encrypt message with public key.

    Explanation
    ===========

    ``i`` is a plaintext message expressed as an integer.
    ``key`` is public key (p, r, e). In order to encrypt
    a message, a random number ``a`` in ``range(2, p)``
    is generated and the encrypted message is returned as
    `c_{1}` and `c_{2}` where:

    `c_{1} \equiv r^{a} \pmod p`

    `c_{2} \equiv m e^{a} \pmod p`

    Parameters
    ==========

    msg
        int of encoded message.

    key
        Public key.

    Returns
    =======

    tuple : (c1, c2)
        Encipher into two number.

    Notes
    =====

    For testing purposes, the ``seed`` parameter may be set to control
    the output of this routine. See sympy.core.random._randrange.

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_elgamal, elgamal_private_key, elgamal_public_key
    >>> pri = elgamal_private_key(5, seed=[3]); pri
    (37, 2, 3)
    >>> pub = elgamal_public_key(pri); pub
    (37, 2, 8)
    >>> msg = 36
    >>> encipher_elgamal(msg, pub, seed=[3])
    (8, 6)

    """
    p, r, e = key
    if i < 0 or i >= p:
        raise ValueError(
            'Message (%s) should be in range(%s)' % (i, p))
    randrange = _randrange(seed)
    a = randrange(2, p)
    return pow(r, a, p), i*pow(e, a, p) % p


def decipher_elgamal(msg, key):
    r"""
    Decrypt message with private key.

    `msg = (c_{1}, c_{2})`

    `key = (p, r, d)`

    According to extended Eucliden theorem,
    `u c_{1}^{d} + p n = 1`

    `u \equiv 1/{{c_{1}}^d} \pmod p`

    `u c_{2} \equiv \frac{1}{c_{1}^d} c_{2} \equiv \frac{1}{r^{ad}} c_{2} \pmod p`

    `\frac{1}{r^{ad}} m e^a \equiv \frac{1}{r^{ad}} m {r^{d a}} \equiv m \pmod p`

    Examples
    ========

    >>> from sympy.crypto.crypto import decipher_elgamal
    >>> from sympy.crypto.crypto import encipher_elgamal
    >>> from sympy.crypto.crypto import elgamal_private_key
    >>> from sympy.crypto.crypto import elgamal_public_key

    >>> pri = elgamal_private_key(5, seed=[3])
    >>> pub = elgamal_public_key(pri); pub
    (37, 2, 8)
    >>> msg = 17
    >>> decipher_elgamal(encipher_elgamal(msg, pub), pri) == msg
    True

    """
    p, _, d = key
    c1, c2 = msg
    u = pow(c1, -d, p)
    return u * c2 % p


################ Diffie-Hellman Key Exchange  #########################

def dh_private_key(digit=10, seed=None):
    r"""
    Return three integer tuple as private key.

    Explanation
    ===========

    Diffie-Hellman key exchange is based on the mathematical problem
    called the Discrete Logarithm Problem (see ElGamal).

    Diffie-Hellman key exchange is divided into the following steps:

    *   Alice and Bob agree on a base that consist of a prime ``p``
        and a primitive root of ``p`` called ``g``
    *   Alice choses a number ``a`` and Bob choses a number ``b`` where
        ``a`` and ``b`` are random numbers in range `[2, p)`. These are
        their private keys.
    *   Alice then publicly sends Bob `g^{a} \pmod p` while Bob sends
        Alice `g^{b} \pmod p`
    *   They both raise the received value to their secretly chosen
        number (``a`` or ``b``) and now have both as their shared key
        `g^{ab} \pmod p`

    Parameters
    ==========

    digit
        Minimum number of binary digits required in key.

    Returns
    =======

    tuple : (p, g, a)
        p = prime number.

        g = primitive root of p.

        a = random number from 2 through p - 1.

    Notes
    =====

    For testing purposes, the ``seed`` parameter may be set to control
    the output of this routine. See sympy.core.random._randrange.

    Examples
    ========

    >>> from sympy.crypto.crypto import dh_private_key
    >>> from sympy.ntheory import isprime, is_primitive_root
    >>> p, g, _ = dh_private_key()
    >>> isprime(p)
    True
    >>> is_primitive_root(g, p)
    True
    >>> p, g, _ = dh_private_key(5)
    >>> isprime(p)
    True
    >>> is_primitive_root(g, p)
    True

    """
    p = nextprime(2**digit)
    g = primitive_root(p)
    randrange = _randrange(seed)
    a = randrange(2, p)
    return p, g, a


def dh_public_key(key):
    r"""
    Return three number tuple as public key.

    This is the tuple that Alice sends to Bob.

    Parameters
    ==========

    key : (p, g, a)
        A tuple generated by ``dh_private_key``.

    Returns
    =======

    tuple : int, int, int
        A tuple of `(p, g, g^a \mod p)` with `p`, `g` and `a` given as
        parameters.s

    Examples
    ========

    >>> from sympy.crypto.crypto import dh_private_key, dh_public_key
    >>> p, g, a = dh_private_key();
    >>> _p, _g, x = dh_public_key((p, g, a))
    >>> p == _p and g == _g
    True
    >>> x == pow(g, a, p)
    True

    """
    p, g, a = key
    return p, g, pow(g, a, p)


def dh_shared_key(key, b):
    """
    Return an integer that is the shared key.

    This is what Bob and Alice can both calculate using the public
    keys they received from each other and their private keys.

    Parameters
    ==========

    key : (p, g, x)
        Tuple `(p, g, x)` generated by ``dh_public_key``.

    b
        Random number in the range of `2` to `p - 1`
        (Chosen by second key exchange member (Bob)).

    Returns
    =======

    int
        A shared key.

    Examples
    ========

    >>> from sympy.crypto.crypto import (
    ...     dh_private_key, dh_public_key, dh_shared_key)
    >>> prk = dh_private_key();
    >>> p, g, x = dh_public_key(prk);
    >>> sk = dh_shared_key((p, g, x), 1000)
    >>> sk == pow(x, 1000, p)
    True

    """
    p, _, x = key
    if 1 >= b or b >= p:
        raise ValueError(filldedent('''
            Value of b should be greater 1 and less
            than prime %s.''' % p))

    return pow(x, b, p)


################ Goldwasser-Micali Encryption  #########################


def _legendre(a, p):
    """
    Returns the legendre symbol of a and p
    assuming that p is a prime.

    i.e. 1 if a is a quadratic residue mod p
        -1 if a is not a quadratic residue mod p
         0 if a is divisible by p

    Parameters
    ==========

    a : int
        The number to test.

    p : prime
        The prime to test ``a`` against.

    Returns
    =======

    int
        Legendre symbol (a / p).

    """
    sig = pow(a, (p - 1)//2, p)
    if sig == 1:
        return 1
    elif sig == 0:
        return 0
    else:
        return -1


def _random_coprime_stream(n, seed=None):
    randrange = _randrange(seed)
    while True:
        y = randrange(n)
        if gcd(y, n) == 1:
            yield y


def gm_private_key(p, q, a=None):
    r"""
    Check if ``p`` and ``q`` can be used as private keys for
    the Goldwasser-Micali encryption. The method works
    roughly as follows.

    Explanation
    ===========

    #. Pick two large primes $p$ and $q$.
    #. Call their product $N$.
    #. Given a message as an integer $i$, write $i$ in its bit representation $b_0, \dots, b_n$.
    #. For each $k$,

     if $b_k = 0$:
        let $a_k$ be a random square
        (quadratic residue) modulo $p q$
        such that ``jacobi_symbol(a, p*q) = 1``
     if $b_k = 1$:
        let $a_k$ be a random non-square
        (non-quadratic residue) modulo $p q$
        such that ``jacobi_symbol(a, p*q) = 1``

    returns $\left[a_1, a_2, \dots\right]$

    $b_k$ can be recovered by checking whether or not
    $a_k$ is a residue. And from the $b_k$'s, the message
    can be reconstructed.

    The idea is that, while ``jacobi_symbol(a, p*q)``
    can be easily computed (and when it is equal to $-1$ will
    tell you that $a$ is not a square mod $p q$), quadratic
    residuosity modulo a composite number is hard to compute
    without knowing its factorization.

    Moreover, approximately half the numbers coprime to $p q$ have
    :func:`~.jacobi_symbol` equal to $1$ . And among those, approximately half
    are residues and approximately half are not. This maximizes the
    entropy of the code.

    Parameters
    ==========

    p, q, a
        Initialization variables.

    Returns
    =======

    tuple : (p, q)
        The input value ``p`` and ``q``.

    Raises
    ======

    ValueError
        If ``p`` and ``q`` are not distinct odd primes.

    """
    if p == q:
        raise ValueError("expected distinct primes, "
                         "got two copies of %i" % p)
    elif not isprime(p) or not isprime(q):
        raise ValueError("first two arguments must be prime, "
                         "got %i of %i" % (p, q))
    elif p == 2 or q == 2:
        raise ValueError("first two arguments must not be even, "
                         "got %i of %i" % (p, q))
    return p, q


def gm_public_key(p, q, a=None, seed=None):
    """
    Compute public keys for ``p`` and ``q``.
    Note that in Goldwasser-Micali Encryption,
    public keys are randomly selected.

    Parameters
    ==========

    p, q, a : int, int, int
        Initialization variables.

    Returns
    =======

    tuple : (a, N)
        ``a`` is the input ``a`` if it is not ``None`` otherwise
        some random integer coprime to ``p`` and ``q``.

        ``N`` is the product of ``p`` and ``q``.

    """

    p, q = gm_private_key(p, q)
    N = p * q

    if a is None:
        randrange = _randrange(seed)
        while True:
            a = randrange(N)
            if _legendre(a, p) == _legendre(a, q) == -1:
                break
    else:
        if _legendre(a, p) != -1 or _legendre(a, q) != -1:
            return False
    return (a, N)


def encipher_gm(i, key, seed=None):
    """
    Encrypt integer 'i' using public_key 'key'
    Note that gm uses random encryption.

    Parameters
    ==========

    i : int
        The message to encrypt.

    key : (a, N)
        The public key.

    Returns
    =======

    list : list of int
        The randomized encrypted message.

    """
    if i < 0:
        raise ValueError(
            "message must be a non-negative "
            "integer: got %d instead" % i)
    a, N = key
    bits = []
    while i > 0:
        bits.append(i % 2)
        i //= 2

    gen = _random_coprime_stream(N, seed)
    rev = reversed(bits)
    encode = lambda b: next(gen)**2*pow(a, b) % N
    return [ encode(b) for b in rev ]



def decipher_gm(message, key):
    """
    Decrypt message 'message' using public_key 'key'.

    Parameters
    ==========

    message : list of int
        The randomized encrypted message.

    key : (p, q)
        The private key.

    Returns
    =======

    int
        The encrypted message.

    """
    p, q = key
    res = lambda m, p: _legendre(m, p) > 0
    bits = [res(m, p) * res(m, q) for m in message]
    m = 0
    for b in bits:
        m <<= 1
        m += not b
    return m



########### RailFence Cipher #############

def encipher_railfence(message,rails):
    """
    Performs Railfence Encryption on plaintext and returns ciphertext

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_railfence
    >>> message = "hello world"
    >>> encipher_railfence(message,3)
    'horel ollwd'

    Parameters
    ==========

    message : string, the message to encrypt.
    rails : int, the number of rails.

    Returns
    =======

    The Encrypted string message.

    References
    ==========
    .. [1] https://en.wikipedia.org/wiki/Rail_fence_cipher

    """
    r = list(range(rails))
    p = cycle(r + r[-2:0:-1])
    return ''.join(sorted(message, key=lambda i: next(p)))


def decipher_railfence(ciphertext,rails):
    """
    Decrypt the message using the given rails

    Examples
    ========

    >>> from sympy.crypto.crypto import decipher_railfence
    >>> decipher_railfence("horel ollwd",3)
    'hello world'

    Parameters
    ==========

    message : string, the message to encrypt.
    rails : int, the number of rails.

    Returns
    =======

    The Decrypted string message.

    """
    r = list(range(rails))
    p = cycle(r + r[-2:0:-1])

    idx = sorted(range(len(ciphertext)), key=lambda i: next(p))
    res = [''] * len(ciphertext)
    for i, c in zip(idx, ciphertext):
        res[i] = c
    return ''.join(res)


################ Blum-Goldwasser cryptosystem  #########################

def bg_private_key(p, q):
    """
    Check if p and q can be used as private keys for
    the Blum-Goldwasser cryptosystem.

    Explanation
    ===========

    The three necessary checks for p and q to pass
    so that they can be used as private keys:

        1. p and q must both be prime
        2. p and q must be distinct
        3. p and q must be congruent to 3 mod 4

    Parameters
    ==========

    p, q
        The keys to be checked.

    Returns
    =======

    p, q
        Input values.

    Raises
    ======

    ValueError
        If p and q do not pass the above conditions.

    """

    if not isprime(p) or not isprime(q):
        raise ValueError("the two arguments must be prime, "
                         "got %i and %i" %(p, q))
    elif p == q:
        raise ValueError("the two arguments must be distinct, "
                         "got two copies of %i. " %p)
    elif (p - 3) % 4 != 0 or (q - 3) % 4 != 0:
        raise ValueError("the two arguments must be congruent to 3 mod 4, "
                         "got %i and %i" %(p, q))
    return p, q

def bg_public_key(p, q):
    """
    Calculates public keys from private keys.

    Explanation
    ===========

    The function first checks the validity of
    private keys passed as arguments and
    then returns their product.

    Parameters
    ==========

    p, q
        The private keys.

    Returns
    =======

    N
        The public key.

    """
    p, q = bg_private_key(p, q)
    N = p * q
    return N

def encipher_bg(i, key, seed=None):
    """
    Encrypts the message using public key and seed.

    Explanation
    ===========

    ALGORITHM:
        1. Encodes i as a string of L bits, m.
        2. Select a random element r, where 1 < r < key, and computes
           x = r^2 mod key.
        3. Use BBS pseudo-random number generator to generate L random bits, b,
        using the initial seed as x.
        4. Encrypted message, c_i = m_i XOR b_i, 1 <= i <= L.
        5. x_L = x^(2^L) mod key.
        6. Return (c, x_L)

    Parameters
    ==========

    i
        Message, a non-negative integer

    key
        The public key

    Returns
    =======

    Tuple
        (encrypted_message, x_L)

    Raises
    ======

    ValueError
        If i is negative.

    """

    if i < 0:
        raise ValueError(
            "message must be a non-negative "
            "integer: got %d instead" % i)

    enc_msg = []
    while i > 0:
        enc_msg.append(i % 2)
        i //= 2
    enc_msg.reverse()
    L = len(enc_msg)

    r = _randint(seed)(2, key - 1)
    x = r**2 % key
    x_L = pow(int(x), int(2**L), int(key))

    rand_bits = []
    for _ in range(L):
        rand_bits.append(x % 2)
        x = x**2 % key

    encrypt_msg = [m ^ b for (m, b) in zip(enc_msg, rand_bits)]

    return (encrypt_msg, x_L)

def decipher_bg(message, key):
    """
    Decrypts the message using private keys.

    Explanation
    ===========

    ALGORITHM:
        1. Let, c be the encrypted message, y the second number received,
        and p and q be the private keys.
        2. Compute, r_p = y^((p+1)/4 ^ L) mod p and
        r_q = y^((q+1)/4 ^ L) mod q.
        3. Compute x_0 = (q(q^-1 mod p)r_p + p(p^-1 mod q)r_q) mod N.
        4. From, recompute the bits using the BBS generator, as in the
        encryption algorithm.
        5. Compute original message by XORing c and b.

    Parameters
    ==========

    message
        Tuple of encrypted message and a non-negative integer.

    key
        Tuple of private keys.

    Returns
    =======

    orig_msg
        The original message

    """

    p, q = key
    encrypt_msg, y = message
    public_key = p * q
    L = len(encrypt_msg)
    p_t = ((p + 1)/4)**L
    q_t = ((q + 1)/4)**L
    r_p = pow(int(y), int(p_t), int(p))
    r_q = pow(int(y), int(q_t), int(q))

    x = (q * invert(q, p) * r_p + p * invert(p, q) * r_q) % public_key

    orig_bits = []
    for _ in range(L):
        orig_bits.append(x % 2)
        x = x**2 % public_key

    orig_msg = 0
    for (m, b) in zip(encrypt_msg, orig_bits):
        orig_msg = orig_msg * 2
        orig_msg += (m ^ b)

    return orig_msg
<|endoftext|>from sympy.core import S, Basic, Dict, Symbol, Tuple, sympify
from sympy.core.symbol import Str
from sympy.sets import Set, FiniteSet, EmptySet
from sympy.utilities.iterables import iterable


class Class(Set):
    r"""
    The base class for any kind of class in the set-theoretic sense.

    Explanation
    ===========

    In axiomatic set theories, everything is a class.  A class which
    can be a member of another class is a set.  A class which is not a
    member of another class is a proper class.  The class `\{1, 2\}`
    is a set; the class of all sets is a proper class.

    This class is essentially a synonym for :class:`sympy.core.Set`.
    The goal of this class is to assure easier migration to the
    eventual proper implementation of set theory.
    """
    is_proper = False


class Object(Symbol):
    """
    The base class for any kind of object in an abstract category.

    Explanation
    ===========

    While technically any instance of :class:`~.Basic` will do, this
    class is the recommended way to create abstract objects in
    abstract categories.
    """


class Morphism(Basic):
    """
    The base class for any morphism in an abstract category.

    Explanation
    ===========

    In abstract categories, a morphism is an arrow between two
    category objects.  The object where the arrow starts is called the
    domain, while the object where the arrow ends is called the
    codomain.

    Two morphisms between the same pair of objects are considered to
    be the same morphisms.  To distinguish between morphisms between
    the same objects use :class:`NamedMorphism`.

    It is prohibited to instantiate this class.  Use one of the
    derived classes instead.

    See Also
    ========

    IdentityMorphism, NamedMorphism, CompositeMorphism
    """
    def __new__(cls, domain, codomain):
        raise(NotImplementedError(
            "Cannot instantiate Morphism.  Use derived classes instead."))

    @property
    def domain(self):
        """
        Returns the domain of the morphism.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> A = Object("A")
        >>> B = Object("B")
        >>> f = NamedMorphism(A, B, "f")
        >>> f.domain
        Object("A")

        """
        return self.args[0]

    @property
    def codomain(self):
        """
        Returns the codomain of the morphism.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> A = Object("A")
        >>> B = Object("B")
        >>> f = NamedMorphism(A, B, "f")
        >>> f.codomain
        Object("B")

        """
        return self.args[1]

    def compose(self, other):
        r"""
        Composes self with the supplied morphism.

        The order of elements in the composition is the usual order,
        i.e., to construct `g\circ f` use ``g.compose(f)``.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> g * f
        CompositeMorphism((NamedMorphism(Object("A"), Object("B"), "f"),
        NamedMorphism(Object("B"), Object("C"), "g")))
        >>> (g * f).domain
        Object("A")
        >>> (g * f).codomain
        Object("C")

        """
        return CompositeMorphism(other, self)

    def __mul__(self, other):
        r"""
        Composes self with the supplied morphism.

        The semantics of this operation is given by the following
        equation: ``g * f == g.compose(f)`` for composable morphisms
        ``g`` and ``f``.

        See Also
        ========

        compose
        """
        return self.compose(other)


class IdentityMorphism(Morphism):
    """
    Represents an identity morphism.

    Explanation
    ===========

    An identity morphism is a morphism with equal domain and codomain,
    which acts as an identity with respect to composition.

    Examples
    ========

    >>> from sympy.categories import Object, NamedMorphism, IdentityMorphism
    >>> A = Object("A")
    >>> B = Object("B")
    >>> f = NamedMorphism(A, B, "f")
    >>> id_A = IdentityMorphism(A)
    >>> id_B = IdentityMorphism(B)
    >>> f * id_A == f
    True
    >>> id_B * f == f
    True

    See Also
    ========

    Morphism
    """
    def __new__(cls, domain):
        return Basic.__new__(cls, domain)

    @property
    def codomain(self):
        return self.domain


class NamedMorphism(Morphism):
    """
    Represents a morphism which has a name.

    Explanation
    ===========

    Names are used to distinguish between morphisms which have the
    same domain and codomain: two named morphisms are equal if they
    have the same domains, codomains, and names.

    Examples
    ========

    >>> from sympy.categories import Object, NamedMorphism
    >>> A = Object("A")
    >>> B = Object("B")
    >>> f = NamedMorphism(A, B, "f")
    >>> f
    NamedMorphism(Object("A"), Object("B"), "f")
    >>> f.name
    'f'

    See Also
    ========

    Morphism
    """
    def __new__(cls, domain, codomain, name):
        if not name:
            raise ValueError("Empty morphism names not allowed.")

        if not isinstance(name, Str):
            name = Str(name)

        return Basic.__new__(cls, domain, codomain, name)

    @property
    def name(self):
        """
        Returns the name of the morphism.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> A = Object("A")
        >>> B = Object("B")
        >>> f = NamedMorphism(A, B, "f")
        >>> f.name
        'f'

        """
        return self.args[2].name


class CompositeMorphism(Morphism):
    r"""
    Represents a morphism which is a composition of other morphisms.

    Explanation
    ===========

    Two composite morphisms are equal if the morphisms they were
    obtained from (components) are the same and were listed in the
    same order.

    The arguments to the constructor for this class should be listed
    in diagram order: to obtain the composition `g\circ f` from the
    instances of :class:`Morphism` ``g`` and ``f`` use
    ``CompositeMorphism(f, g)``.

    Examples
    ========

    >>> from sympy.categories import Object, NamedMorphism, CompositeMorphism
    >>> A = Object("A")
    >>> B = Object("B")
    >>> C = Object("C")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> g * f
    CompositeMorphism((NamedMorphism(Object("A"), Object("B"), "f"),
    NamedMorphism(Object("B"), Object("C"), "g")))
    >>> CompositeMorphism(f, g) == g * f
    True

    """
    @staticmethod
    def _add_morphism(t, morphism):
        """
        Intelligently adds ``morphism`` to tuple ``t``.

        Explanation
        ===========

        If ``morphism`` is a composite morphism, its components are
        added to the tuple.  If ``morphism`` is an identity, nothing
        is added to the tuple.

        No composability checks are performed.
        """
        if isinstance(morphism, CompositeMorphism):
            # ``morphism`` is a composite morphism; we have to
            # denest its components.
            return t + morphism.components
        elif isinstance(morphism, IdentityMorphism):
            # ``morphism`` is an identity.  Nothing happens.
            return t
        else:
            return t + Tuple(morphism)

    def __new__(cls, *components):
        if components and not isinstance(components[0], Morphism):
            # Maybe the user has explicitly supplied a list of
            # morphisms.
            return CompositeMorphism.__new__(cls, *components[0])

        normalised_components = Tuple()

        for current, following in zip(components, components[1:]):
            if not isinstance(current, Morphism) or \
                    not isinstance(following, Morphism):
                raise TypeError("All components must be morphisms.")

            if current.codomain != following.domain:
                raise ValueError("Uncomposable morphisms.")

            normalised_components = CompositeMorphism._add_morphism(
                normalised_components, current)

        # We haven't added the last morphism to the list of normalised
        # components.  Add it now.
        normalised_components = CompositeMorphism._add_morphism(
            normalised_components, components[-1])

        if not normalised_components:
            # If ``normalised_components`` is empty, only identities
            # were supplied.  Since they all were composable, they are
            # all the same identities.
            return components[0]
        elif len(normalised_components) == 1:
            # No sense to construct a whole CompositeMorphism.
            return normalised_components[0]

        return Basic.__new__(cls, normalised_components)

    @property
    def components(self):
        """
        Returns the components of this composite morphism.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> (g * f).components
        (NamedMorphism(Object("A"), Object("B"), "f"),
        NamedMorphism(Object("B"), Object("C"), "g"))

        """
        return self.args[0]

    @property
    def domain(self):
        """
        Returns the domain of this composite morphism.

        The domain of the composite morphism is the domain of its
        first component.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> (g * f).domain
        Object("A")

        """
        return self.components[0].domain

    @property
    def codomain(self):
        """
        Returns the codomain of this composite morphism.

        The codomain of the composite morphism is the codomain of its
        last component.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> (g * f).codomain
        Object("C")

        """
        return self.components[-1].codomain

    def flatten(self, new_name):
        """
        Forgets the composite structure of this morphism.

        Explanation
        ===========

        If ``new_name`` is not empty, returns a :class:`NamedMorphism`
        with the supplied name, otherwise returns a :class:`Morphism`.
        In both cases the domain of the new morphism is the domain of
        this composite morphism and the codomain of the new morphism
        is the codomain of this composite morphism.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> (g * f).flatten("h")
        NamedMorphism(Object("A"), Object("C"), "h")

        """
        return NamedMorphism(self.domain, self.codomain, new_name)


class Category(Basic):
    r"""
    An (abstract) category.

    Explanation
    ===========

    A category [JoyOfCats] is a quadruple `\mbox{K} = (O, \hom, id,
    \circ)` consisting of

    * a (set-theoretical) class `O`, whose members are called
      `K`-objects,

    * for each pair `(A, B)` of `K`-objects, a set `\hom(A, B)` whose
      members are called `K`-morphisms from `A` to `B`,

    * for a each `K`-object `A`, a morphism `id:A\rightarrow A`,
      called the `K`-identity of `A`,

    * a composition law `\circ` associating with every `K`-morphisms
      `f:A\rightarrow B` and `g:B\rightarrow C` a `K`-morphism `g\circ
      f:A\rightarrow C`, called the composite of `f` and `g`.

    Composition is associative, `K`-identities are identities with
    respect to composition, and the sets `\hom(A, B)` are pairwise
    disjoint.

    This class knows nothing about its objects and morphisms.
    Concrete cases of (abstract) categories should be implemented as
    classes derived from this one.

    Certain instances of :class:`Diagram` can be asserted to be
    commutative in a :class:`Category` by supplying the argument
    ``commutative_diagrams`` in the constructor.

    Examples
    ========

    >>> from sympy.categories import Object, NamedMorphism, Diagram, Category
    >>> from sympy import FiniteSet
    >>> A = Object("A")
    >>> B = Object("B")
    >>> C = Object("C")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> d = Diagram([f, g])
    >>> K = Category("K", commutative_diagrams=[d])
    >>> K.commutative_diagrams == FiniteSet(d)
    True

    See Also
    ========

    Diagram
    """
    def __new__(cls, name, objects=EmptySet, commutative_diagrams=EmptySet):
        if not name:
            raise ValueError("A Category cannot have an empty name.")

        if not isinstance(name, Str):
            name = Str(name)

        if not isinstance(objects, Class):
            objects = Class(objects)

        new_category = Basic.__new__(cls, name, objects,
                                     FiniteSet(*commutative_diagrams))
        return new_category

    @property
    def name(self):
        """
        Returns the name of this category.

        Examples
        ========

        >>> from sympy.categories import Category
        >>> K = Category("K")
        >>> K.name
        'K'

        """
        return self.args[0].name

    @property
    def objects(self):
        """
        Returns the class of objects of this category.

        Examples
        ========

        >>> from sympy.categories import Object, Category
        >>> from sympy import FiniteSet
        >>> A = Object("A")
        >>> B = Object("B")
        >>> K = Category("K", FiniteSet(A, B))
        >>> K.objects
        Class({Object("A"), Object("B")})

        """
        return self.args[1]

    @property
    def commutative_diagrams(self):
        """
        Returns the :class:`~.FiniteSet` of diagrams which are known to
        be commutative in this category.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism, Diagram, Category
        >>> from sympy import FiniteSet
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> d = Diagram([f, g])
        >>> K = Category("K", commutative_diagrams=[d])
        >>> K.commutative_diagrams == FiniteSet(d)
        True

        """
        return self.args[2]

    def hom(self, A, B):
        raise NotImplementedError(
            "hom-sets are not implemented in Category.")

    def all_morphisms(self):
        raise NotImplementedError(
            "Obtaining the class of morphisms is not implemented in Category.")


class Diagram(Basic):
    r"""
    Represents a diagram in a certain category.

    Explanation
    ===========

    Informally, a diagram is a collection of objects of a category and
    certain morphisms between them.  A diagram is still a monoid with
    respect to morphism composition; i.e., identity morphisms, as well
    as all composites of morphisms included in the diagram belong to
    the diagram.  For a more formal approach to this notion see
    [Pare1970].

    The components of composite morphisms are also added to the
    diagram.  No properties are assigned to such morphisms by default.

    A commutative diagram is often accompanied by a statement of the
    following kind: "if such morphisms with such properties exist,
    then such morphisms which such properties exist and the diagram is
    commutative".  To represent this, an instance of :class:`Diagram`
    includes a collection of morphisms which are the premises and
    another collection of conclusions.  ``premises`` and
    ``conclusions`` associate morphisms belonging to the corresponding
    categories with the :class:`~.FiniteSet`'s of their properties.

    The set of properties of a composite morphism is the intersection
    of the sets of properties of its components.  The domain and
    codomain of a conclusion morphism should be among the domains and
    codomains of the morphisms listed as the premises of a diagram.

    No checks are carried out of whether the supplied object and
    morphisms do belong to one and the same category.

    Examples
    ========

    >>> from sympy.categories import Object, NamedMorphism, Diagram
    >>> from sympy import pprint, default_sort_key
    >>> A = Object("A")
    >>> B = Object("B")
    >>> C = Object("C")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> d = Diagram([f, g])
    >>> premises_keys = sorted(d.premises.keys(), key=default_sort_key)
    >>> pprint(premises_keys, use_unicode=False)
    [g*f:A-->C, id:A-->A, id:B-->B, id:C-->C, f:A-->B, g:B-->C]
    >>> pprint(d.premises, use_unicode=False)
    {g*f:A-->C: EmptySet, id:A-->A: EmptySet, id:B-->B: EmptySet,
     id:C-->C: EmptySet, f:A-->B: EmptySet, g:B-->C: EmptySet}
    >>> d = Diagram([f, g], {g * f: "unique"})
    >>> pprint(d.conclusions,use_unicode=False)
    {g*f:A-->C: {unique}}

    References
    ==========

    [Pare1970] B. Pareigis: Categories and functors.  Academic Press, 1970.

    """
    @staticmethod
    def _set_dict_union(dictionary, key, value):
        """
        If ``key`` is in ``dictionary``, set the new value of ``key``
        to be the union between the old value and ``value``.
        Otherwise, set the value of ``key`` to ``value.

        Returns ``True`` if the key already was in the dictionary and
        ``False`` otherwise.
        """
        if key in dictionary:
            dictionary[key] = dictionary[key] | value
            return True
        else:
            dictionary[key] = value
            return False

    @staticmethod
    def _add_morphism_closure(morphisms, morphism, props, add_identities=True,
                              recurse_composites=True):
        """
        Adds a morphism and its attributes to the supplied dictionary
        ``morphisms``.  If ``add_identities`` is True, also adds the
        identity morphisms for the domain and the codomain of
        ``morphism``.
        """
        if not Diagram._set_dict_union(morphisms, morphism, props):
            # We have just added a new morphism.

            if isinstance(morphism, IdentityMorphism):
                if props:
                    # Properties for identity morphisms don't really
                    # make sense, because very much is known about
                    # identity morphisms already, so much that they
                    # are trivial.  Having properties for identity
                    # morphisms would only be confusing.
                    raise ValueError(
                        "Instances of IdentityMorphism cannot have properties.")
                return

            if add_identities:
                empty = EmptySet

                id_dom = IdentityMorphism(morphism.domain)
                id_cod = IdentityMorphism(morphism.codomain)

                Diagram._set_dict_union(morphisms, id_dom, empty)
                Diagram._set_dict_union(morphisms, id_cod, empty)

            for existing_morphism, existing_props in list(morphisms.items()):
                new_props = existing_props & props
                if morphism.domain == existing_morphism.codomain:
                    left = morphism * existing_morphism
                    Diagram._set_dict_union(morphisms, left, new_props)
                if morphism.codomain == existing_morphism.domain:
                    right = existing_morphism * morphism
                    Diagram._set_dict_union(morphisms, right, new_props)

            if isinstance(morphism, CompositeMorphism) and recurse_composites:
                # This is a composite morphism, add its components as
                # well.
                empty = EmptySet
                for component in morphism.components:
                    Diagram._add_morphism_closure(morphisms, component, empty,
                                                  add_identities)

    def __new__(cls, *args):
        """
        Construct a new instance of Diagram.

        Explanation
        ===========

        If no arguments are supplied, an empty diagram is created.

        If at least an argument is supplied, ``args[0]`` is
        interpreted as the premises of the diagram.  If ``args[0]`` is
        a list, it is interpreted as a list of :class:`Morphism`'s, in
        which each :class:`Morphism` has an empty set of properties.
        If ``args[0]`` is a Python dictionary or a :class:`Dict`, it
        is interpreted as a dictionary associating to some
        :class:`Morphism`'s some properties.

        If at least two arguments are supplied ``args[1]`` is
        interpreted as the conclusions of the diagram.  The type of
        ``args[1]`` is interpreted in exactly the same way as the type
        of ``args[0]``.  If only one argument is supplied, the diagram
        has no conclusions.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> from sympy.categories import IdentityMorphism, Diagram
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> d = Diagram([f, g])
        >>> IdentityMorphism(A) in d.premises.keys()
        True
        >>> g * f in d.premises.keys()
        True
        >>> d = Diagram([f, g], {g * f: "unique"})
        >>> d.conclusions[g * f]
        {unique}

        """
        premises = {}
        conclusions = {}

        # Here we will keep track of the objects which appear in the
        # premises.
        objects = EmptySet

        if len(args) >= 1:
            # We've got some premises in the arguments.
            premises_arg = args[0]

            if isinstance(premises_arg, list):
                # The user has supplied a list of morphisms, none of
                # which have any attributes.
                empty = EmptySet

                for morphism in premises_arg:
                    objects |= FiniteSet(morphism.domain, morphism.codomain)
                    Diagram._add_morphism_closure(premises, morphism, empty)
            elif isinstance(premises_arg, (dict, Dict)):
                # The user has supplied a dictionary of morphisms and
                # their properties.
                for morphism, props in premises_arg.items():
                    objects |= FiniteSet(morphism.domain, morphism.codomain)
                    Diagram._add_morphism_closure(
                        premises, morphism, FiniteSet(*props) if iterable(props) else FiniteSet(props))

        if len(args) >= 2:
            # We also have some conclusions.
            conclusions_arg = args[1]

            if isinstance(conclusions_arg, list):
                # The user has supplied a list of morphisms, none of
                # which have any attributes.
                empty = EmptySet

                for morphism in conclusions_arg:
                    # Check that no new objects appear in conclusions.
                    if ((sympify(objects.contains(morphism.domain)) is S.true) and
                        (sympify(objects.contains(morphism.codomain)) is S.true)):
                        # No need to add identities and recurse
                        # composites this time.
                        Diagram._add_morphism_closure(
                            conclusions, morphism, empty, add_identities=False,
                            recurse_composites=False)
            elif isinstance(conclusions_arg, (dict, Dict)):
                # The user has supplied a dictionary of morphisms and
                # their properties.
                for morphism, props in conclusions_arg.items():
                    # Check that no new objects appear in conclusions.
                    if (morphism.domain in objects) and \
                       (morphism.codomain in objects):
                        # No need to add identities and recurse
                        # composites this time.
                        Diagram._add_morphism_closure(
                            conclusions, morphism, FiniteSet(*props) if iterable(props) else FiniteSet(props),
                            add_identities=False, recurse_composites=False)

        return Basic.__new__(cls, Dict(premises), Dict(conclusions), objects)

    @property
    def premises(self):
        """
        Returns the premises of this diagram.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> from sympy.categories import IdentityMorphism, Diagram
        >>> from sympy import pretty
        >>> A = Object("A")
        >>> B = Object("B")
        >>> f = NamedMorphism(A, B, "f")
        >>> id_A = IdentityMorphism(A)
        >>> id_B = IdentityMorphism(B)
        >>> d = Diagram([f])
        >>> print(pretty(d.premises, use_unicode=False))
        {id:A-->A: EmptySet, id:B-->B: EmptySet, f:A-->B: EmptySet}

        """
        return self.args[0]

    @property
    def conclusions(self):
        """
        Returns the conclusions of this diagram.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> from sympy.categories import IdentityMorphism, Diagram
        >>> from sympy import FiniteSet
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> d = Diagram([f, g])
        >>> IdentityMorphism(A) in d.premises.keys()
        True
        >>> g * f in d.premises.keys()
        True
        >>> d = Diagram([f, g], {g * f: "unique"})
        >>> d.conclusions[g * f] == FiniteSet("unique")
        True

        """
        return self.args[1]

    @property
    def objects(self):
        """
        Returns the :class:`~.FiniteSet` of objects that appear in this
        diagram.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism, Diagram
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> d = Diagram([f, g])
        >>> d.objects
        {Object("A"), Object("B"), Object("C")}

        """
        return self.args[2]

    def hom(self, A, B):
        """
        Returns a 2-tuple of sets of morphisms between objects ``A`` and
        ``B``: one set of morphisms listed as premises, and the other set
        of morphisms listed as conclusions.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism, Diagram
        >>> from sympy import pretty
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> d = Diagram([f, g], {g * f: "unique"})
        >>> print(pretty(d.hom(A, C), use_unicode=False))
        ({g*f:A-->C}, {g*f:A-->C})

        See Also
        ========
        Object, Morphism
        """
        premises = EmptySet
        conclusions = EmptySet

        for morphism in self.premises.keys():
            if (morphism.domain == A) and (morphism.codomain == B):
                premises |= FiniteSet(morphism)
        for morphism in self.conclusions.keys():
            if (morphism.domain == A) and (morphism.codomain == B):
                conclusions |= FiniteSet(morphism)

        return (premises, conclusions)

    def is_subdiagram(self, diagram):
        """
        Checks whether ``diagram`` is a subdiagram of ``self``.
        Diagram `D'` is a subdiagram of `D` if all premises
        (conclusions) of `D'` are contained in the premises
        (conclusions) of `D`.  The morphisms contained
        both in `D'` and `D` should have the same properties for `D'`
        to be a subdiagram of `D`.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism, Diagram
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> d = Diagram([f, g], {g * f: "unique"})
        >>> d1 = Diagram([f])
        >>> d.is_subdiagram(d1)
        True
        >>> d1.is_subdiagram(d)
        False
        """
        premises = all((m in self.premises) and
                       (diagram.premises[m] == self.premises[m])
                       for m in diagram.premises)
        if not premises:
            return False

        conclusions = all((m in self.conclusions) and
                          (diagram.conclusions[m] == self.conclusions[m])
                          for m in diagram.conclusions)

        # Premises is surely ``True`` here.
        return conclusions

    def subdiagram_from_objects(self, objects):
        """
        If ``objects`` is a subset of the objects of ``self``, returns
        a diagram which has as premises all those premises of ``self``
        which have a domains and codomains in ``objects``, likewise
        for conclusions.  Properties are preserved.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism, Diagram
        >>> from sympy import FiniteSet
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> d = Diagram([f, g], {f: "unique", g*f: "veryunique"})
        >>> d1 = d.subdiagram_from_objects(FiniteSet(A, B))
        >>> d1 == Diagram([f], {f: "unique"})
        True
        """
        if not objects.is_subset(self.objects):
            raise ValueError(
                "Supplied objects should all belong to the diagram.")

        new_premises = {}
        for morphism, props in self.premises.items():
            if ((sympify(objects.contains(morphism.domain)) is S.true) and
                (sympify(objects.contains(morphism.codomain)) is S.true)):
                new_premises[morphism] = props

        new_conclusions = {}
        for morphism, props in self.conclusions.items():
            if ((sympify(objects.contains(morphism.domain)) is S.true) and
                (sympify(objects.contains(morphism.codomain)) is S.true)):
                new_conclusions[morphism] = props

        return Diagram(new_premises, new_conclusions)
<|endoftext|>r"""
This module contains the functionality to arrange the nodes of a
diagram on an abstract grid, and then to produce a graphical
representation of the grid.

The currently supported back-ends are Xy-pic [Xypic].

Layout Algorithm
================

This section provides an overview of the algorithms implemented in
:class:`DiagramGrid` to lay out diagrams.

The first step of the algorithm is the removal composite and identity
morphisms which do not have properties in the supplied diagram.  The
premises and conclusions of the diagram are then merged.

The generic layout algorithm begins with the construction of the
"skeleton" of the diagram.  The skeleton is an undirected graph which
has the objects of the diagram as vertices and has an (undirected)
edge between each pair of objects between which there exist morphisms.
The direction of the morphisms does not matter at this stage.  The
skeleton also includes an edge between each pair of vertices `A` and
`C` such that there exists an object `B` which is connected via
a morphism to `A`, and via a morphism to `C`.

The skeleton constructed in this way has the property that every
object is a vertex of a triangle formed by three edges of the
skeleton.  This property lies at the base of the generic layout
algorithm.

After the skeleton has been constructed, the algorithm lists all
triangles which can be formed.  Note that some triangles will not have
all edges corresponding to morphisms which will actually be drawn.
Triangles which have only one edge or less which will actually be
drawn are immediately discarded.

The list of triangles is sorted according to the number of edges which
correspond to morphisms, then the triangle with the least number of such
edges is selected.  One of such edges is picked and the corresponding
objects are placed horizontally, on a grid.  This edge is recorded to
be in the fringe.  The algorithm then finds a "welding" of a triangle
to the fringe.  A welding is an edge in the fringe where a triangle
could be attached.  If the algorithm succeeds in finding such a
welding, it adds to the grid that vertex of the triangle which was not
yet included in any edge in the fringe and records the two new edges in
the fringe.  This process continues iteratively until all objects of
the diagram has been placed or until no more weldings can be found.

An edge is only removed from the fringe when a welding to this edge
has been found, and there is no room around this edge to place
another vertex.

When no more weldings can be found, but there are still triangles
left, the algorithm searches for a possibility of attaching one of the
remaining triangles to the existing structure by a vertex.  If such a
possibility is found, the corresponding edge of the found triangle is
placed in the found space and the iterative process of welding
triangles restarts.

When logical groups are supplied, each of these groups is laid out
independently.  Then a diagram is constructed in which groups are
objects and any two logical groups between which there exist morphisms
are connected via a morphism.  This diagram is laid out.  Finally,
the grid which includes all objects of the initial diagram is
constructed by replacing the cells which contain logical groups with
the corresponding laid out grids, and by correspondingly expanding the
rows and columns.

The sequential layout algorithm begins by constructing the
underlying undirected graph defined by the morphisms obtained after
simplifying premises and conclusions and merging them (see above).
The vertex with the minimal degree is then picked up and depth-first
search is started from it.  All objects which are located at distance
`n` from the root in the depth-first search tree, are positioned in
the `n`-th column of the resulting grid.  The sequential layout will
therefore attempt to lay the objects out along a line.

References
==========

.. [Xypic] https://xy-pic.sourceforge.net/

"""
from sympy.categories import (CompositeMorphism, IdentityMorphism,
                              NamedMorphism, Diagram)
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on

from itertools import chain


__doctest_requires__ = {('preview_diagram',): 'pyglet'}


class _GrowableGrid:
    """
    Holds a growable grid of objects.

    Explanation
    ===========

    It is possible to append or prepend a row or a column to the grid
    using the corresponding methods.  Prepending rows or columns has
    the effect of changing the coordinates of the already existing
    elements.

    This class currently represents a naive implementation of the
    functionality with little attempt at optimisation.
    """
    def __init__(self, width, height):
        self._width = width
        self._height = height

        self._array = [[None for j in range(width)] for i in range(height)]

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def __getitem__(self, i_j):
        """
        Returns the element located at in the i-th line and j-th
        column.
        """
        i, j = i_j
        return self._array[i][j]

    def __setitem__(self, i_j, newvalue):
        """
        Sets the element located at in the i-th line and j-th
        column.
        """
        i, j = i_j
        self._array[i][j] = newvalue

    def append_row(self):
        """
        Appends an empty row to the grid.
        """
        self._height += 1
        self._array.append([None for j in range(self._width)])

    def append_column(self):
        """
        Appends an empty column to the grid.
        """
        self._width += 1
        for i in range(self._height):
            self._array[i].append(None)

    def prepend_row(self):
        """
        Prepends the grid with an empty row.
        """
        self._height += 1
        self._array.insert(0, [None for j in range(self._width)])

    def prepend_column(self):
        """
        Prepends the grid with an empty column.
        """
        self._width += 1
        for i in range(self._height):
            self._array[i].insert(0, None)


class DiagramGrid:
    r"""
    Constructs and holds the fitting of the diagram into a grid.

    Explanation
    ===========

    The mission of this class is to analyse the structure of the
    supplied diagram and to place its objects on a grid such that,
    when the objects and the morphisms are actually drawn, the diagram
    would be "readable", in the sense that there will not be many
    intersections of moprhisms.  This class does not perform any
    actual drawing.  It does strive nevertheless to offer sufficient
    metadata to draw a diagram.

    Consider the following simple diagram.

    >>> from sympy.categories import Object, NamedMorphism
    >>> from sympy.categories import Diagram, DiagramGrid
    >>> from sympy import pprint
    >>> A = Object("A")
    >>> B = Object("B")
    >>> C = Object("C")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> diagram = Diagram([f, g])

    The simplest way to have a diagram laid out is the following:

    >>> grid = DiagramGrid(diagram)
    >>> (grid.width, grid.height)
    (2, 2)
    >>> pprint(grid)
    A  B
    <BLANKLINE>
       C

    Sometimes one sees the diagram as consisting of logical groups.
    One can advise ``DiagramGrid`` as to such groups by employing the
    ``groups`` keyword argument.

    Consider the following diagram:

    >>> D = Object("D")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> h = NamedMorphism(D, A, "h")
    >>> k = NamedMorphism(D, B, "k")
    >>> diagram = Diagram([f, g, h, k])

    Lay it out with generic layout:

    >>> grid = DiagramGrid(diagram)
    >>> pprint(grid)
    A  B  D
    <BLANKLINE>
       C

    Now, we can group the objects `A` and `D` to have them near one
    another:

    >>> grid = DiagramGrid(diagram, groups=[[A, D], B, C])
    >>> pprint(grid)
    B     C
    <BLANKLINE>
    A  D

    Note how the positioning of the other objects changes.

    Further indications can be supplied to the constructor of
    :class:`DiagramGrid` using keyword arguments.  The currently
    supported hints are explained in the following paragraphs.

    :class:`DiagramGrid` does not automatically guess which layout
    would suit the supplied diagram better.  Consider, for example,
    the following linear diagram:

    >>> E = Object("E")
    >>> f = NamedMorphism(A, B, "f")
    >>> g = NamedMorphism(B, C, "g")
    >>> h = NamedMorphism(C, D, "h")
    >>> i = NamedMorphism(D, E, "i")
    >>> diagram = Diagram([f, g, h, i])

    When laid out with the generic layout, it does not get to look
    linear:

    >>> grid = DiagramGrid(diagram)
    >>> pprint(grid)
    A  B
    <BLANKLINE>
       C  D
    <BLANKLINE>
          E

    To get it laid out in a line, use ``layout="sequential"``:

    >>> grid = DiagramGrid(diagram, layout="sequential")
    >>> pprint(grid)
    A  B  C  D  E

    One may sometimes need to transpose the resulting layout.  While
    this can always be done by hand, :class:`DiagramGrid` provides a
    hint for that purpose:

    >>> grid = DiagramGrid(diagram, layout="sequential", transpose=True)
    >>> pprint(grid)
    A
    <BLANKLINE>
    B
    <BLANKLINE>
    C
    <BLANKLINE>
    D
    <BLANKLINE>
    E

    Separate hints can also be provided for each group.  For an
    example, refer to ``tests/test_drawing.py``, and see the different
    ways in which the five lemma [FiveLemma] can be laid out.

    See Also
    ========

    Diagram

    References
    ==========

    .. [FiveLemma] https://en.wikipedia.org/wiki/Five_lemma
    """
    @staticmethod
    def _simplify_morphisms(morphisms):
        """
        Given a dictionary mapping morphisms to their properties,
        returns a new dictionary in which there are no morphisms which
        do not have properties, and which are compositions of other
        morphisms included in the dictionary.  Identities are dropped
        as well.
        """
        newmorphisms = {}
        for morphism, props in morphisms.items():
            if isinstance(morphism, CompositeMorphism) and not props:
                continue
            elif isinstance(morphism, IdentityMorphism):
                continue
            else:
                newmorphisms[morphism] = props
        return newmorphisms

    @staticmethod
    def _merge_premises_conclusions(premises, conclusions):
        """
        Given two dictionaries of morphisms and their properties,
        produces a single dictionary which includes elements from both
        dictionaries.  If a morphism has some properties in premises
        and also in conclusions, the properties in conclusions take
        priority.
        """
        return dict(chain(premises.items(), conclusions.items()))

    @staticmethod
    def _juxtapose_edges(edge1, edge2):
        """
        If ``edge1`` and ``edge2`` have precisely one common endpoint,
        returns an edge which would form a triangle with ``edge1`` and
        ``edge2``.

        If ``edge1`` and ``edge2`` do not have a common endpoint,
        returns ``None``.

        If ``edge1`` and ``edge`` are the same edge, returns ``None``.
        """
        intersection = edge1 & edge2
        if len(intersection) != 1:
            # The edges either have no common points or are equal.
            return None

        # The edges have a common endpoint.  Extract the different
        # endpoints and set up the new edge.
        return (edge1 - intersection) | (edge2 - intersection)

    @staticmethod
    def _add_edge_append(dictionary, edge, elem):
        """
        If ``edge`` is not in ``dictionary``, adds ``edge`` to the
        dictionary and sets its value to ``[elem]``.  Otherwise
        appends ``elem`` to the value of existing entry.

        Note that edges are undirected, thus `(A, B) = (B, A)`.
        """
        if edge in dictionary:
            dictionary[edge].append(elem)
        else:
            dictionary[edge] = [elem]

    @staticmethod
    def _build_skeleton(morphisms):
        """
        Creates a dictionary which maps edges to corresponding
        morphisms.  Thus for a morphism `f:A\rightarrow B`, the edge
        `(A, B)` will be associated with `f`.  This function also adds
        to the list those edges which are formed by juxtaposition of
        two edges already in the list.  These new edges are not
        associated with any morphism and are only added to assure that
        the diagram can be decomposed into triangles.
        """
        edges = {}
        # Create edges for morphisms.
        for morphism in morphisms:
            DiagramGrid._add_edge_append(
                edges, frozenset([morphism.domain, morphism.codomain]), morphism)

        # Create new edges by juxtaposing existing edges.
        edges1 = dict(edges)
        for w in edges1:
            for v in edges1:
                wv = DiagramGrid._juxtapose_edges(w, v)
                if wv and wv not in edges:
                    edges[wv] = []

        return edges

    @staticmethod
    def _list_triangles(edges):
        """
        Builds the set of triangles formed by the supplied edges.  The
        triangles are arbitrary and need not be commutative.  A
        triangle is a set that contains all three of its sides.
        """
        triangles = set()

        for w in edges:
            for v in edges:
                wv = DiagramGrid._juxtapose_edges(w, v)
                if wv and wv in edges:
                    triangles.add(frozenset([w, v, wv]))

        return triangles

    @staticmethod
    def _drop_redundant_triangles(triangles, skeleton):
        """
        Returns a list which contains only those triangles who have
        morphisms associated with at least two edges.
        """
        return [tri for tri in triangles
                if len([e for e in tri if skeleton[e]]) >= 2]

    @staticmethod
    def _morphism_length(morphism):
        """
        Returns the length of a morphism.  The length of a morphism is
        the number of components it consists of.  A non-composite
        morphism is of length 1.
        """
        if isinstance(morphism, CompositeMorphism):
            return len(morphism.components)
        else:
            return 1

    @staticmethod
    def _compute_triangle_min_sizes(triangles, edges):
        r"""
        Returns a dictionary mapping triangles to their minimal sizes.
        The minimal size of a triangle is the sum of maximal lengths
        of morphisms associated to the sides of the triangle.  The
        length of a morphism is the number of components it consists
        of.  A non-composite morphism is of length 1.

        Sorting triangles by this metric attempts to address two
        aspects of layout.  For triangles with only simple morphisms
        in the edge, this assures that triangles with all three edges
        visible will get typeset after triangles with less visible
        edges, which sometimes minimizes the necessity in diagonal
        arrows.  For triangles with composite morphisms in the edges,
        this assures that objects connected with shorter morphisms
        will be laid out first, resulting the visual proximity of
        those objects which are connected by shorter morphisms.
        """
        triangle_sizes = {}
        for triangle in triangles:
            size = 0
            for e in triangle:
                morphisms = edges[e]
                if morphisms:
                    size += max(DiagramGrid._morphism_length(m)
                                for m in morphisms)
            triangle_sizes[triangle] = size
        return triangle_sizes

    @staticmethod
    def _triangle_objects(triangle):
        """
        Given a triangle, returns the objects included in it.
        """
        # A triangle is a frozenset of three two-element frozensets
        # (the edges).  This chains the three edges together and
        # creates a frozenset from the iterator, thus producing a
        # frozenset of objects of the triangle.
        return frozenset(chain(*tuple(triangle)))

    @staticmethod
    def _other_vertex(triangle, edge):
        """
        Given a triangle and an edge of it, returns the vertex which
        opposes the edge.
        """
        # This gets the set of objects of the triangle and then
        # subtracts the set of objects employed in ``edge`` to get the
        # vertex opposite to ``edge``.
        return list(DiagramGrid._triangle_objects(triangle) - set(edge))[0]

    @staticmethod
    def _empty_point(pt, grid):
        """
        Checks if the cell at coordinates ``pt`` is either empty or
        out of the bounds of the grid.
        """
        if (pt[0] < 0) or (pt[1] < 0) or \
           (pt[0] >= grid.height) or (pt[1] >= grid.width):
            return True
        return grid[pt] is None

    @staticmethod
    def _put_object(coords, obj, grid, fringe):
        """
        Places an object at the coordinate ``cords`` in ``grid``,
        growing the grid and updating ``fringe``, if necessary.
        Returns (0, 0) if no row or column has been prepended, (1, 0)
        if a row was prepended, (0, 1) if a column was prepended and
        (1, 1) if both a column and a row were prepended.
        """
        (i, j) = coords
        offset = (0, 0)
        if i == -1:
            grid.prepend_row()
            i = 0
            offset = (1, 0)
            for k in range(len(fringe)):
                ((i1, j1), (i2, j2)) = fringe[k]
                fringe[k] = ((i1 + 1, j1), (i2 + 1, j2))
        elif i == grid.height:
            grid.append_row()

        if j == -1:
            j = 0
            offset = (offset[0], 1)
            grid.prepend_column()
            for k in range(len(fringe)):
                ((i1, j1), (i2, j2)) = fringe[k]
                fringe[k] = ((i1, j1 + 1), (i2, j2 + 1))
        elif j == grid.width:
            grid.append_column()

        grid[i, j] = obj
        return offset

    @staticmethod
    def _choose_target_cell(pt1, pt2, edge, obj, skeleton, grid):
        """
        Given two points, ``pt1`` and ``pt2``, and the welding edge
        ``edge``, chooses one of the two points to place the opposing
        vertex ``obj`` of the triangle.  If neither of this points
        fits, returns ``None``.
        """
        pt1_empty = DiagramGrid._empty_point(pt1, grid)
        pt2_empty = DiagramGrid._empty_point(pt2, grid)

        if pt1_empty and pt2_empty:
            # Both cells are empty.  Of these two, choose that cell
            # which will assure that a visible edge of the triangle
            # will be drawn perpendicularly to the current welding
            # edge.

            A = grid[edge[0]]

            if skeleton.get(frozenset([A, obj])):
                return pt1
            else:
                return pt2
        if pt1_empty:
            return pt1
        elif pt2_empty:
            return pt2
        else:
            return None

    @staticmethod
    def _find_triangle_to_weld(triangles, fringe, grid):
        """
        Finds, if possible, a triangle and an edge in the ``fringe`` to
        which the triangle could be attached.  Returns the tuple
        containing the triangle and the index of the corresponding
        edge in the ``fringe``.

        This function relies on the fact that objects are unique in
        the diagram.
        """
        for triangle in triangles:
            for (a, b) in fringe:
                if frozenset([grid[a], grid[b]]) in triangle:
                    return (triangle, (a, b))
        return None

    @staticmethod
    def _weld_triangle(tri, welding_edge, fringe, grid, skeleton):
        """
        If possible, welds the triangle ``tri`` to ``fringe`` and
        returns ``False``.  If this method encounters a degenerate
        situation in the fringe and corrects it such that a restart of
        the search is required, it returns ``True`` (which means that
        a restart in finding triangle weldings is required).

        A degenerate situation is a situation when an edge listed in
        the fringe does not belong to the visual boundary of the
        diagram.
        """
        a, b = welding_edge
        target_cell = None

        obj = DiagramGrid._other_vertex(tri, (grid[a], grid[b]))

        # We now have a triangle and an edge where it can be welded to
        # the fringe.  Decide where to place the other vertex of the
        # triangle and check for degenerate situations en route.

        if (abs(a[0] - b[0]) == 1) and (abs(a[1] - b[1]) == 1):
            # A diagonal edge.
            target_cell = (a[0], b[1])
            if grid[target_cell]:
                # That cell is already occupied.
                target_cell = (b[0], a[1])

                if grid[target_cell]:
                    # Degenerate situation, this edge is not
                    # on the actual fringe.  Correct the
                    # fringe and go on.
                    fringe.remove((a, b))
                    return True
        elif a[0] == b[0]:
            # A horizontal edge.  We first attempt to build the
            # triangle in the downward direction.

            down_left = a[0] + 1, a[1]
            down_right = a[0] + 1, b[1]

            target_cell = DiagramGrid._choose_target_cell(
                down_left, down_right, (a, b), obj, skeleton, grid)

            if not target_cell:
                # No room below this edge.  Check above.
                up_left = a[0] - 1, a[1]
                up_right = a[0] - 1, b[1]

                target_cell = DiagramGrid._choose_target_cell(
                    up_left, up_right, (a, b), obj, skeleton, grid)

                if not target_cell:
                    # This edge is not in the fringe, remove it
                    # and restart.
                    fringe.remove((a, b))
                    return True
        elif a[1] == b[1]:
            # A vertical edge.  We will attempt to place the other
            # vertex of the triangle to the right of this edge.
            right_up = a[0], a[1] + 1
            right_down = b[0], a[1] + 1

            target_cell = DiagramGrid._choose_target_cell(
                right_up, right_down, (a, b), obj, skeleton, grid)

            if not target_cell:
                # No room to the left.  See what's to the right.
                left_up = a[0], a[1] - 1
                left_down = b[0], a[1] - 1

                target_cell = DiagramGrid._choose_target_cell(
                    left_up, left_down, (a, b), obj, skeleton, grid)

                if not target_cell:
                    # This edge is not in the fringe, remove it
                    # and restart.
                    fringe.remove((a, b))
                    return True

        # We now know where to place the other vertex of the
        # triangle.
        offset = DiagramGrid._put_object(target_cell, obj, grid, fringe)

        # Take care of the displacement of coordinates if a row or
        # a column was prepended.
        target_cell = (target_cell[0] + offset[0],
                       target_cell[1] + offset[1])
        a = (a[0] + offset[0], a[1] + offset[1])
        b = (b[0] + offset[0], b[1] + offset[1])

