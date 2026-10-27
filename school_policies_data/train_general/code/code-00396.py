    >>> from sympy.solvers.inequalities import solve_poly_inequalities
    >>> from sympy.abc import x
    >>> solve_poly_inequalities(((
    ... Poly(x**2 - 3), ">"), (
    ... Poly(-x**2 + 1), ">")))
    Union(Interval.open(-oo, -sqrt(3)), Interval.open(-1, 1), Interval.open(sqrt(3), oo))
    """
    return Union(*[s for p in polys for s in solve_poly_inequality(*p)])


def solve_rational_inequalities(eqs):
    """Solve a system of rational inequalities with rational coefficients.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy import solve_rational_inequalities, Poly

    >>> solve_rational_inequalities([[
    ... ((Poly(-x + 1), Poly(1, x)), '>='),
    ... ((Poly(-x + 1), Poly(1, x)), '<=')]])
    {1}

    >>> solve_rational_inequalities([[
    ... ((Poly(x), Poly(1, x)), '!='),
    ... ((Poly(-x + 1), Poly(1, x)), '>=')]])
    Union(Interval.open(-oo, 0), Interval.Lopen(0, 1))

    See Also
    ========
    solve_poly_inequality
    """
    result = S.EmptySet

    for _eqs in eqs:
        if not _eqs:
            continue

        global_intervals = [Interval(S.NegativeInfinity, S.Infinity)]

        for (numer, denom), rel in _eqs:
            numer_intervals = solve_poly_inequality(numer*denom, rel)
            denom_intervals = solve_poly_inequality(denom, '==')

            intervals = []

            for numer_interval, global_interval in itertools.product(
                    numer_intervals, global_intervals):
                interval = numer_interval.intersect(global_interval)

                if interval is not S.EmptySet:
                    intervals.append(interval)

            global_intervals = intervals

            intervals = []

            for global_interval in global_intervals:
                for denom_interval in denom_intervals:
                    global_interval -= denom_interval

                if global_interval is not S.EmptySet:
                    intervals.append(global_interval)

            global_intervals = intervals

            if not global_intervals:
                break

        for interval in global_intervals:
            result = result.union(interval)

    return result


def reduce_rational_inequalities(exprs, gen, relational=True):
    """Reduce a system of rational inequalities with rational coefficients.

    Examples
    ========

    >>> from sympy import Symbol
    >>> from sympy.solvers.inequalities import reduce_rational_inequalities

    >>> x = Symbol('x', real=True)

    >>> reduce_rational_inequalities([[x**2 <= 0]], x)
    Eq(x, 0)

    >>> reduce_rational_inequalities([[x + 2 > 0]], x)
    -2 < x
    >>> reduce_rational_inequalities([[(x + 2, ">")]], x)
    -2 < x
    >>> reduce_rational_inequalities([[x + 2]], x)
    Eq(x, -2)

    This function find the non-infinite solution set so if the unknown symbol
    is declared as extended real rather than real then the result may include
    finiteness conditions:

    >>> y = Symbol('y', extended_real=True)
    >>> reduce_rational_inequalities([[y + 2 > 0]], y)
    (-2 < y) & (y < oo)
    """
    exact = True
    eqs = []
    solution = S.EmptySet  # add pieces for each group
    for _exprs in exprs:
        if not _exprs:
            continue
        _eqs = []
        _sol = S.Reals
        for expr in _exprs:
            if isinstance(expr, tuple):
                expr, rel = expr
            else:
                if expr.is_Relational:
                    expr, rel = expr.lhs - expr.rhs, expr.rel_op
                else:
                    rel = '=='

            if expr is S.true:
                numer, denom, rel = S.Zero, S.One, '=='
            elif expr is S.false:
                numer, denom, rel = S.One, S.One, '=='
            else:
                numer, denom = expr.together().as_numer_denom()

            try:
                (numer, denom), opt = parallel_poly_from_expr(
                    (numer, denom), gen)
            except PolynomialError:
                raise PolynomialError(filldedent('''
                    only polynomials and rational functions are
                    supported in this context.
                    '''))

            if not opt.domain.is_Exact:
                numer, denom, exact = numer.to_exact(), denom.to_exact(), False

            domain = opt.domain.get_exact()

            if not (domain.is_ZZ or domain.is_QQ):
                expr = numer/denom
                expr = Relational(expr, 0, rel)
                _sol &= solve_univariate_inequality(expr, gen, relational=False)
            else:
                _eqs.append(((numer, denom), rel))

        if _eqs:
            _sol &= solve_rational_inequalities([_eqs])
            exclude = solve_rational_inequalities([[((d, d.one), '==')
                for i in eqs for ((n, d), _) in i if d.has(gen)]])
            _sol -= exclude

        solution |= _sol

    if not exact and solution:
        solution = solution.evalf()

    if relational:
        solution = solution.as_relational(gen)

    return solution


def reduce_abs_inequality(expr, rel, gen):
    """Reduce an inequality with nested absolute values.

    Examples
    ========

    >>> from sympy import reduce_abs_inequality, Abs, Symbol
    >>> x = Symbol('x', real=True)

    >>> reduce_abs_inequality(Abs(x - 5) - 3, '<', x)
    (2 < x) & (x < 8)

    >>> reduce_abs_inequality(Abs(x + 2)*3 - 13, '<', x)
    (-19/3 < x) & (x < 7/3)

    See Also
    ========

    reduce_abs_inequalities
    """
    if gen.is_extended_real is False:
        raise TypeError(filldedent('''
            Cannot solve inequalities with absolute values containing
            non-real variables.
            '''))

    def _bottom_up_scan(expr):
        exprs = []

        if expr.is_Add or expr.is_Mul:
            op = expr.func

            for arg in expr.args:
                _exprs = _bottom_up_scan(arg)

                if not exprs:
                    exprs = _exprs
                else:
                    exprs = [(op(expr, _expr), conds + _conds) for (expr, conds), (_expr, _conds) in
                            itertools.product(exprs, _exprs)]
        elif expr.is_Pow:
            n = expr.exp
            if not n.is_Integer:
                raise ValueError("Only Integer Powers are allowed on Abs.")

            exprs.extend((expr**n, conds) for expr, conds in _bottom_up_scan(expr.base))
        elif isinstance(expr, Abs):
            _exprs = _bottom_up_scan(expr.args[0])

            for expr, conds in _exprs:
                exprs.append(( expr, conds + [Ge(expr, 0)]))
                exprs.append((-expr, conds + [Lt(expr, 0)]))
        else:
            exprs = [(expr, [])]

        return exprs

    mapping = {'<': '>', '<=': '>='}
    inequalities = []

    for expr, conds in _bottom_up_scan(expr):
        if rel not in mapping.keys():
            expr = Relational( expr, 0, rel)
        else:
            expr = Relational(-expr, 0, mapping[rel])

        inequalities.append([expr] + conds)

    return reduce_rational_inequalities(inequalities, gen)


def reduce_abs_inequalities(exprs, gen):
    """Reduce a system of inequalities with nested absolute values.

    Examples
    ========

    >>> from sympy import reduce_abs_inequalities, Abs, Symbol
    >>> x = Symbol('x', extended_real=True)

    >>> reduce_abs_inequalities([(Abs(3*x - 5) - 7, '<'),
    ... (Abs(x + 25) - 13, '>')], x)
    (-2/3 < x) & (x < 4) & (((-oo < x) & (x < -38)) | ((-12 < x) & (x < oo)))

    >>> reduce_abs_inequalities([(Abs(x - 4) + Abs(3*x - 5) - 7, '<')], x)
    (1/2 < x) & (x < 4)

    See Also
    ========

    reduce_abs_inequality
    """
    return And(*[ reduce_abs_inequality(expr, rel, gen)
        for expr, rel in exprs ])


def solve_univariate_inequality(expr, gen, relational=True, domain=S.Reals, continuous=False):
    """Solves a real univariate inequality.

    Parameters
    ==========

    expr : Relational
        The target inequality
    gen : Symbol
        The variable for which the inequality is solved
    relational : bool
        A Relational type output is expected or not
    domain : Set
        The domain over which the equation is solved
    continuous: bool
        True if expr is known to be continuous over the given domain
        (and so continuous_domain() does not need to be called on it)

    Raises
    ======

    NotImplementedError
        The solution of the inequality cannot be determined due to limitation
        in :func:`sympy.solvers.solveset.solvify`.

    Notes
    =====

    Currently, we cannot solve all the inequalities due to limitations in
    :func:`sympy.solvers.solveset.solvify`. Also, the solution returned for trigonometric inequalities
    are restricted in its periodic interval.

    See Also
    ========

    sympy.solvers.solveset.solvify: solver returning solveset solutions with solve's output API

    Examples
    ========

    >>> from sympy import solve_univariate_inequality, Symbol, sin, Interval, S
    >>> x = Symbol('x')

    >>> solve_univariate_inequality(x**2 >= 4, x)
    ((2 <= x) & (x < oo)) | ((-oo < x) & (x <= -2))

    >>> solve_univariate_inequality(x**2 >= 4, x, relational=False)
    Union(Interval(-oo, -2), Interval(2, oo))

    >>> domain = Interval(0, S.Infinity)
    >>> solve_univariate_inequality(x**2 >= 4, x, False, domain)
    Interval(2, oo)

    >>> solve_univariate_inequality(sin(x) > 0, x, relational=False)
    Interval.open(0, pi)

    """
    from sympy.solvers.solvers import denoms

    if domain.is_subset(S.Reals) is False:
        raise NotImplementedError(filldedent('''
        Inequalities in the complex domain are
        not supported. Try the real domain by
        setting domain=S.Reals'''))
    elif domain is not S.Reals:
        rv = solve_univariate_inequality(
        expr, gen, relational=False, continuous=continuous).intersection(domain)
        if relational:
            rv = rv.as_relational(gen)
        return rv
    else:
        pass  # continue with attempt to solve in Real domain

    # This keeps the function independent of the assumptions about `gen`.
    # `solveset` makes sure this function is called only when the domain is
    # real.
    _gen = gen
    _domain = domain
    if gen.is_extended_real is False:
        rv = S.EmptySet
        return rv if not relational else rv.as_relational(_gen)
    elif gen.is_extended_real is None:
        gen = Dummy('gen', extended_real=True)
        try:
            expr = expr.xreplace({_gen: gen})
        except TypeError:
            raise TypeError(filldedent('''
                When gen is real, the relational has a complex part
                which leads to an invalid comparison like I < 0.
                '''))

    rv = None

    if expr is S.true:
        rv = domain

    elif expr is S.false:
        rv = S.EmptySet

    else:
        e = expr.lhs - expr.rhs
        period = periodicity(e, gen)
        if period == S.Zero:
            e = expand_mul(e)
            const = expr.func(e, 0)
            if const is S.true:
                rv = domain
            elif const is S.false:
                rv = S.EmptySet
        elif period is not None:
            frange = function_range(e, gen, domain)

            rel = expr.rel_op
            if rel in ('<', '<='):
                if expr.func(frange.sup, 0):
                    rv = domain
                elif not expr.func(frange.inf, 0):
                    rv = S.EmptySet

            elif rel in ('>', '>='):
                if expr.func(frange.inf, 0):
                    rv = domain
                elif not expr.func(frange.sup, 0):
                    rv = S.EmptySet

            inf, sup = domain.inf, domain.sup
            if sup - inf is S.Infinity:
                domain = Interval(0, period, False, True).intersect(_domain)
                _domain = domain

        if rv is None:
            n, d = e.as_numer_denom()
            try:
                if gen not in n.free_symbols and len(e.free_symbols) > 1:
                    raise ValueError
                # this might raise ValueError on its own
                # or it might give None...
                solns = solvify(e, gen, domain)
                if solns is None:
                    # in which case we raise ValueError
                    raise ValueError
            except (ValueError, NotImplementedError):
                # replace gen with generic x since it's
                # univariate anyway
                raise NotImplementedError(filldedent('''
                    The inequality, %s, cannot be solved using
                    solve_univariate_inequality.
                    ''' % expr.subs(gen, Symbol('x'))))

            expanded_e = expand_mul(e)
            def valid(x):
                # this is used to see if gen=x satisfies the
                # relational by substituting it into the
                # expanded form and testing against 0, e.g.
                # if expr = x*(x + 1) < 2 then e = x*(x + 1) - 2
                # and expanded_e = x**2 + x - 2; the test is
                # whether a given value of x satisfies
                # x**2 + x - 2 < 0
                #
                # expanded_e, expr and gen used from enclosing scope
                v = expanded_e.subs(gen, expand_mul(x))
                try:
                    r = expr.func(v, 0)
                except TypeError:
                    r = S.false
                if r in (S.true, S.false):
                    return r
                if v.is_extended_real is False:
                    return S.false
                else:
                    v = v.n(2)
                    if v.is_comparable:
                        return expr.func(v, 0)
                    # not comparable or couldn't be evaluated
                    raise NotImplementedError(
                        'relationship did not evaluate: %s' % r)

            singularities = []
            for d in denoms(expr, gen):
                singularities.extend(solvify(d, gen, domain))
            if not continuous:
                domain = continuous_domain(expanded_e, gen, domain)

            include_x = '=' in expr.rel_op and expr.rel_op != '!='

            try:
                discontinuities = set(domain.boundary -
                    FiniteSet(domain.inf, domain.sup))
                # remove points that are not between inf and sup of domain
                critical_points = FiniteSet(*(solns + singularities + list(
                    discontinuities))).intersection(
                    Interval(domain.inf, domain.sup,
                    domain.inf not in domain, domain.sup not in domain))
                if all(r.is_number for r in critical_points):
                    reals = _nsort(critical_points, separated=True)[0]
                else:
                    sifted = sift(critical_points, lambda x: x.is_extended_real)
                    if sifted[None]:
                        # there were some roots that weren't known
                        # to be real
                        raise NotImplementedError
                    try:
                        reals = sifted[True]
                        if len(reals) > 1:
                            reals = sorted(reals)
                    except TypeError:
                        raise NotImplementedError
            except NotImplementedError:
                raise NotImplementedError('sorting of these roots is not supported')

            # If expr contains imaginary coefficients, only take real
            # values of x for which the imaginary part is 0
            make_real = S.Reals
            if (coeffI := expanded_e.coeff(S.ImaginaryUnit)) != S.Zero:
                check = True
                im_sol = FiniteSet()
                try:
                    a = solveset(coeffI, gen, domain)
                    if not isinstance(a, Interval):
                        for z in a:
                            if z not in singularities and valid(z) and z.is_extended_real:
                                im_sol += FiniteSet(z)
                    else:
                        start, end = a.inf, a.sup
                        for z in _nsort(critical_points + FiniteSet(end)):
                            valid_start = valid(start)
                            if start != end:
                                valid_z = valid(z)
                                pt = _pt(start, z)
                                if pt not in singularities and pt.is_extended_real and valid(pt):
                                    if valid_start and valid_z:
                                        im_sol += Interval(start, z)
                                    elif valid_start:
                                        im_sol += Interval.Ropen(start, z)
                                    elif valid_z:
                                        im_sol += Interval.Lopen(start, z)
                                    else:
                                        im_sol += Interval.open(start, z)
                            start = z
                        for s in singularities:
                            im_sol -= FiniteSet(s)
                except (TypeError):
                    im_sol = S.Reals
                    check = False

                if im_sol is S.EmptySet:
                    raise ValueError(filldedent('''
                        %s contains imaginary parts which cannot be
                        made 0 for any value of %s satisfying the
                        inequality, leading to relations like I < 0.
                        '''  % (expr.subs(gen, _gen), _gen)))

                make_real = make_real.intersect(im_sol)

            sol_sets = [S.EmptySet]

            start = domain.inf
            if start in domain and valid(start) and start.is_finite:
                sol_sets.append(FiniteSet(start))

            for x in reals:
                end = x

                if valid(_pt(start, end)):
                    sol_sets.append(Interval(start, end, True, True))

                if x in singularities:
                    singularities.remove(x)
                else:
                    if x in discontinuities:
                        discontinuities.remove(x)
                        _valid = valid(x)
                    else:  # it's a solution
                        _valid = include_x
                    if _valid:
                        sol_sets.append(FiniteSet(x))

                start = end

            end = domain.sup
            if end in domain and valid(end) and end.is_finite:
                sol_sets.append(FiniteSet(end))

            if valid(_pt(start, end)):
                sol_sets.append(Interval.open(start, end))

            if coeffI != S.Zero and check:
                rv = (make_real).intersect(_domain)
            else:
                rv = Intersection(
                    (Union(*sol_sets)), make_real, _domain).subs(gen, _gen)

    return rv if not relational else rv.as_relational(_gen)


def _pt(start, end):
    """Return a point between start and end"""
    if not start.is_infinite and not end.is_infinite:
        pt = (start + end)/2
    elif start.is_infinite and end.is_infinite:
        pt = S.Zero
    else:
        if (start.is_infinite and start.is_extended_positive is None or
                end.is_infinite and end.is_extended_positive is None):
            raise ValueError('cannot proceed with unsigned infinite values')
        if (end.is_infinite and end.is_extended_negative or
                start.is_infinite and start.is_extended_positive):
            start, end = end, start
        # if possible, use a multiple of self which has
        # better behavior when checking assumptions than
        # an expression obtained by adding or subtracting 1
        if end.is_infinite:
            if start.is_extended_positive:
                pt = start*2
            elif start.is_extended_negative:
                pt = start*S.Half
            else:
                pt = start + 1
        elif start.is_infinite:
            if end.is_extended_positive:
                pt = end*S.Half
            elif end.is_extended_negative:
                pt = end*2
            else:
                pt = end - 1
    return pt


def _solve_inequality(ie, s, linear=False):
    """Return the inequality with s isolated on the left, if possible.
    If the relationship is non-linear, a solution involving And or Or
    may be returned. False or True are returned if the relationship
    is never True or always True, respectively.

    If `linear` is True (default is False) an `s`-dependent expression
    will be isolated on the left, if possible
    but it will not be solved for `s` unless the expression is linear
    in `s`. Furthermore, only "safe" operations which do not change the
    sense of the relationship are applied: no division by an unsigned
    value is attempted unless the relationship involves Eq or Ne and
    no division by a value not known to be nonzero is ever attempted.

    Examples
    ========

    >>> from sympy import Eq, Symbol
    >>> from sympy.solvers.inequalities import _solve_inequality as f
    >>> from sympy.abc import x, y

    For linear expressions, the symbol can be isolated:

    >>> f(x - 2 < 0, x)
    x < 2
    >>> f(-x - 6 < x, x)
    x > -3

    Sometimes nonlinear relationships will be False

    >>> f(x**2 + 4 < 0, x)
    False

    Or they may involve more than one region of values:

    >>> f(x**2 - 4 < 0, x)
    (-2 < x) & (x < 2)

    To restrict the solution to a relational, set linear=True
    and only the x-dependent portion will be isolated on the left:

    >>> f(x**2 - 4 < 0, x, linear=True)
    x**2 < 4

    Division of only nonzero quantities is allowed, so x cannot
    be isolated by dividing by y:

    >>> y.is_nonzero is None  # it is unknown whether it is 0 or not
    True
    >>> f(x*y < 1, x)
    x*y < 1

    And while an equality (or inequality) still holds after dividing by a
    non-zero quantity

    >>> nz = Symbol('nz', nonzero=True)
    >>> f(Eq(x*nz, 1), x)
    Eq(x, 1/nz)

    the sign must be known for other inequalities involving > or <:

    >>> f(x*nz <= 1, x)
    nz*x <= 1
    >>> p = Symbol('p', positive=True)
    >>> f(x*p <= 1, x)
    x <= 1/p

    When there are denominators in the original expression that
    are removed by expansion, conditions for them will be returned
    as part of the result:

    >>> f(x < x*(2/x - 1), x)
    (x < 1) & Ne(x, 0)
    """
    from sympy.solvers.solvers import denoms
    if s not in ie.free_symbols:
        return ie
    if ie.rhs == s:
        ie = ie.reversed
    if ie.lhs == s and s not in ie.rhs.free_symbols:
        return ie

    def classify(ie, s, i):
        # return True or False if ie evaluates when substituting s with
        # i else None (if unevaluated) or NaN (when there is an error
        # in evaluating)
        try:
            v = ie.subs(s, i)
            if v is S.NaN:
                return v
            elif v not in (True, False):
                return
            return v
        except TypeError:
            return S.NaN

    rv = None
    oo = S.Infinity
    expr = ie.lhs - ie.rhs
    try:
        p = Poly(expr, s)
        if p.degree() == 0:
            rv = ie.func(p.as_expr(), 0)
        elif not linear and p.degree() > 1:
            # handle in except clause
            raise NotImplementedError
    except (PolynomialError, NotImplementedError):
        if not linear:
            try:
                rv = reduce_rational_inequalities([[ie]], s)
            except PolynomialError:
                rv = solve_univariate_inequality(ie, s)
            # remove restrictions wrt +/-oo that may have been
            # applied when using sets to simplify the relationship
            okoo = classify(ie, s, oo)
            if okoo is S.true and classify(rv, s, oo) is S.false:
                rv = rv.subs(s < oo, True)
            oknoo = classify(ie, s, -oo)
            if (oknoo is S.true and
                    classify(rv, s, -oo) is S.false):
                rv = rv.subs(-oo < s, True)
                rv = rv.subs(s > -oo, True)
            if rv is S.true:
                rv = (s <= oo) if okoo is S.true else (s < oo)
                if oknoo is not S.true:
                    rv = And(-oo < s, rv)
        else:
            p = Poly(expr)

    conds = []
    if rv is None:
        e = p.as_expr()  # this is in expanded form
        # Do a safe inversion of e, moving non-s terms
        # to the rhs and dividing by a nonzero factor if
        # the relational is Eq/Ne; for other relationals
        # the sign must also be positive or negative
        rhs = 0
        b, ax = e.as_independent(s, as_Add=True)
        e -= b
        rhs -= b
        ef = factor_terms(e)
        a, e = ef.as_independent(s, as_Add=False)
        if (a.is_zero != False or  # don't divide by potential 0
                a.is_negative ==
                a.is_positive is None and  # if sign is not known then
                ie.rel_op not in ('!=', '==')): # reject if not Eq/Ne
            e = ef
            a = S.One
        rhs /= a
        if a.is_positive:
            rv = ie.func(e, rhs)
        else:
            rv = ie.reversed.func(e, rhs)

        # return conditions under which the value is
        # valid, too.
        beginning_denoms = denoms(ie.lhs) | denoms(ie.rhs)
        current_denoms = denoms(rv)
        for d in beginning_denoms - current_denoms:
            c = _solve_inequality(Eq(d, 0), s, linear=linear)
            if isinstance(c, Eq) and c.lhs == s:
                if classify(rv, s, c.rhs) is S.true:
                    # rv is permitting this value but it shouldn't
                    conds.append(~c)
        for i in (-oo, oo):
            if (classify(rv, s, i) is S.true and
                    classify(ie, s, i) is not S.true):
                conds.append(s < i if i is oo else i < s)

    conds.append(rv)
    return And(*conds)


def _reduce_inequalities(inequalities, symbols):
    # helper for reduce_inequalities

    poly_part, abs_part = {}, {}
    other = []

    for inequality in inequalities:

        expr, rel = inequality.lhs, inequality.rel_op  # rhs is 0

        # check for gens using atoms which is more strict than free_symbols to
        # guard against EX domain which won't be handled by
        # reduce_rational_inequalities
        gens = expr.atoms(Symbol)

        if len(gens) == 1:
            gen = gens.pop()
        else:
            common = expr.free_symbols & symbols
            if len(common) == 1:
                gen = common.pop()
                other.append(_solve_inequality(Relational(expr, 0, rel), gen))
                continue
            else:
                raise NotImplementedError(filldedent('''
                    inequality has more than one symbol of interest.
                    '''))

        if expr.is_polynomial(gen):
            poly_part.setdefault(gen, []).append((expr, rel))
        else:
            components = expr.find(lambda u:
                u.has(gen) and (
                u.is_Function or u.is_Pow and not u.exp.is_Integer))
            if components and all(isinstance(i, Abs) for i in components):
                abs_part.setdefault(gen, []).append((expr, rel))
            else:
                other.append(_solve_inequality(Relational(expr, 0, rel), gen))

    poly_reduced = [reduce_rational_inequalities([exprs], gen) for gen, exprs in poly_part.items()]
    abs_reduced = [reduce_abs_inequalities(exprs, gen) for gen, exprs in abs_part.items()]

    return And(*(poly_reduced + abs_reduced + other))


def reduce_inequalities(inequalities, symbols=[]):
    """Reduce a system of inequalities with rational coefficients.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy import reduce_inequalities

    >>> reduce_inequalities(0 <= x + 3, [])
    (-3 <= x) & (x < oo)

    >>> reduce_inequalities(0 <= x + y*2 - 1, [x])
    (x < oo) & (x >= 1 - 2*y)
    """
    if not iterable(inequalities):
        inequalities = [inequalities]
    inequalities = [sympify(i) for i in inequalities]

    gens = set().union(*[i.free_symbols for i in inequalities])

    if not iterable(symbols):
        symbols = [symbols]
    symbols = (set(symbols) or gens) & gens
    if any(i.is_extended_real is False for i in symbols):
        raise TypeError(filldedent('''
            inequalities cannot contain symbols that are not real.
            '''))

    # make vanilla symbol real
    recast = {i: Dummy(i.name, extended_real=True)
        for i in gens if i.is_extended_real is None}
    inequalities = [i.xreplace(recast) for i in inequalities]
    symbols = {i.xreplace(recast) for i in symbols}

    # prefilter
    keep = []
    for i in inequalities:
        if isinstance(i, Relational):
            i = i.func(i.lhs.as_expr() - i.rhs.as_expr(), 0)
        elif i not in (True, False):
            i = Eq(i, 0)
        if i == True:
            continue
        elif i == False:
            return S.false
        if i.lhs.is_number:
            raise NotImplementedError(
                "could not determine truth value of %s" % i)
        keep.append(i)
    inequalities = keep
    del keep

    # solve system
    rv = _reduce_inequalities(inequalities, symbols)

    # restore original symbols and return
    return rv.xreplace({v: k for k, v in recast.items()})
<|endoftext|>from sympy.core import (Function, Pow, sympify, Expr)
from sympy.core.relational import Relational
from sympy.core.singleton import S
from sympy.polys import Poly, decompose
from sympy.utilities.misc import func_name
from sympy.functions.elementary.miscellaneous import Min, Max


def decompogen(f, symbol):
    """
    Computes General functional decomposition of ``f``.
    Given an expression ``f``, returns a list ``[f_1, f_2, ..., f_n]``,
    where::
              f = f_1 o f_2 o ... f_n = f_1(f_2(... f_n))

    Note: This is a General decomposition function. It also decomposes
    Polynomials. For only Polynomial decomposition see ``decompose`` in polys.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy import decompogen, sqrt, sin, cos
    >>> decompogen(sin(cos(x)), x)
    [sin(x), cos(x)]
    >>> decompogen(sin(x)**2 + sin(x) + 1, x)
    [x**2 + x + 1, sin(x)]
    >>> decompogen(sqrt(6*x**2 - 5), x)
    [sqrt(x), 6*x**2 - 5]
    >>> decompogen(sin(sqrt(cos(x**2 + 1))), x)
    [sin(x), sqrt(x), cos(x), x**2 + 1]
    >>> decompogen(x**4 + 2*x**3 - x - 1, x)
    [x**2 - x - 1, x**2 + x]

    """
    f = sympify(f)
    if not isinstance(f, Expr) or isinstance(f, Relational):
        raise TypeError('expecting Expr but got: `%s`' % func_name(f))
    if symbol not in f.free_symbols:
        return [f]


    # ===== Simple Functions ===== #
    if isinstance(f, (Function, Pow)):
        if f.is_Pow and f.base == S.Exp1:
            arg = f.exp
        else:
            arg = f.args[0]
        if arg == symbol:
            return [f]
        return [f.subs(arg, symbol)] + decompogen(arg, symbol)

    # ===== Min/Max Functions ===== #
    if isinstance(f, (Min, Max)):
        args = list(f.args)
        d0 = None
        for i, a in enumerate(args):
            if not a.has_free(symbol):
                continue
            d = decompogen(a, symbol)
            if len(d) == 1:
                d = [symbol] + d
            if d0 is None:
                d0 = d[1:]
            elif d[1:] != d0:
                # decomposition is not the same for each arg:
                # mark as having no decomposition
                d = [symbol]
                break
            args[i] = d[0]
        if d[0] == symbol:
            return [f]
        return [f.func(*args)] + d0

    # ===== Convert to Polynomial ===== #
    fp = Poly(f)
    gens = list(filter(lambda x: symbol in x.free_symbols, fp.gens))

    if len(gens) == 1 and gens[0] != symbol:
        f1 = f.subs(gens[0], symbol)
        f2 = gens[0]
        return [f1] + decompogen(f2, symbol)

    # ===== Polynomial decompose() ====== #
    try:
        return decompose(f)
    except ValueError:
        return [f]


def compogen(g_s, symbol):
    """
    Returns the composition of functions.
    Given a list of functions ``g_s``, returns their composition ``f``,
    where:
        f = g_1 o g_2 o .. o g_n

    Note: This is a General composition function. It also composes Polynomials.
    For only Polynomial composition see ``compose`` in polys.

    Examples
    ========

    >>> from sympy.solvers.decompogen import compogen
    >>> from sympy.abc import x
    >>> from sympy import sqrt, sin, cos
    >>> compogen([sin(x), cos(x)], x)
    sin(cos(x))
    >>> compogen([x**2 + x + 1, sin(x)], x)
    sin(x)**2 + sin(x) + 1
    >>> compogen([sqrt(x), 6*x**2 - 5], x)
    sqrt(6*x**2 - 5)
    >>> compogen([sin(x), sqrt(x), cos(x), x**2 + 1], x)
    sin(sqrt(cos(x**2 + 1)))
    >>> compogen([x**2 - x - 1, x**2 + x], x)
    -x**2 - x + (x**2 + x)**2 - 1
    """
    if len(g_s) == 1:
        return g_s[0]

    foo = g_s[0].subs(symbol, g_s[1])

    if len(g_s) == 2:
        return foo

    return compogen([foo] + g_s[2:], symbol)
<|endoftext|>"""Solvers of systems of polynomial equations. """

from __future__ import annotations

from typing import Any
from collections.abc import Sequence, Iterable

import itertools

from sympy import Dummy
from sympy.core import S
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.sorting import default_sort_key
from sympy.logic.boolalg import Boolean
from sympy.polys import Poly, groebner, roots
from sympy.polys.domains import ZZ
from sympy.polys.polyoptions import build_options
from sympy.polys.polytools import parallel_poly_from_expr, sqf_part
from sympy.polys.polyerrors import (
    ComputationFailed,
    PolificationFailed,
    CoercionFailed,
    GeneratorsNeeded,
    DomainError
)
from sympy.simplify import rcollect
from sympy.utilities import postfixes
from sympy.utilities.iterables import cartes
from sympy.utilities.misc import filldedent
from sympy.logic.boolalg import Or, And
from sympy.core.relational import Eq


class SolveFailed(Exception):
    """Raised when solver's conditions were not met. """


def solve_poly_system(seq, *gens, strict=False, **args):
    """
    Return a list of solutions for the system of polynomial equations
    or else None.

    Parameters
    ==========

    seq: a list/tuple/set
        Listing all the equations that are needed to be solved
    gens: generators
        generators of the equations in seq for which we want the
        solutions
    strict: a boolean (default is False)
        if strict is True, NotImplementedError will be raised if
        the solution is known to be incomplete (which can occur if
        not all solutions are expressible in radicals)
    args: Keyword arguments
        Special options for solving the equations.


    Returns
    =======

    List[Tuple]
        a list of tuples with elements being solutions for the
        symbols in the order they were passed as gens
    None
        None is returned when the computed basis contains only the ground.

    Examples
    ========

    >>> from sympy import solve_poly_system
    >>> from sympy.abc import x, y

    >>> solve_poly_system([x*y - 2*y, 2*y**2 - x**2], x, y)
    [(0, 0), (2, -sqrt(2)), (2, sqrt(2))]

    >>> solve_poly_system([x**5 - x + y**3, y**2 - 1], x, y, strict=True)
    Traceback (most recent call last):
    ...
    UnsolvableFactorError

    """
    try:
        polys, opt = parallel_poly_from_expr(seq, *gens, **args)
    except PolificationFailed as exc:
        raise ComputationFailed('solve_poly_system', len(seq), exc)

    if len(polys) == len(opt.gens) == 2:
        f, g = polys

        if all(i <= 2 for i in f.degree_list() + g.degree_list()):
            try:
                return solve_biquadratic(f, g, opt)
            except SolveFailed:
                pass

    return solve_generic(polys, opt, strict=strict)


def solve_biquadratic(f, g, opt):
    """Solve a system of two bivariate quadratic polynomial equations.

    Parameters
    ==========

    f: a single Expr or Poly
        First equation
    g: a single Expr or Poly
        Second Equation
    opt: an Options object
        For specifying keyword arguments and generators

    Returns
    =======

    List[Tuple]
        a list of tuples with elements being solutions for the
        symbols in the order they were passed as gens
    None
        None is returned when the computed basis contains only the ground.

    Examples
    ========

    >>> from sympy import Options, Poly
    >>> from sympy.abc import x, y
    >>> from sympy.solvers.polysys import solve_biquadratic
    >>> NewOption = Options((x, y), {'domain': 'ZZ'})

    >>> a = Poly(y**2 - 4 + x, y, x, domain='ZZ')
    >>> b = Poly(y*2 + 3*x - 7, y, x, domain='ZZ')
    >>> solve_biquadratic(a, b, NewOption)
    [(1/3, 3), (41/27, 11/9)]

    >>> a = Poly(y + x**2 - 3, y, x, domain='ZZ')
    >>> b = Poly(-y + x - 4, y, x, domain='ZZ')
    >>> solve_biquadratic(a, b, NewOption)
    [(7/2 - sqrt(29)/2, -sqrt(29)/2 - 1/2), (sqrt(29)/2 + 7/2, -1/2 + \
      sqrt(29)/2)]
    """
    G = groebner([f, g])

    if len(G) == 1 and G[0].is_ground:
        return None

    if len(G) != 2:
        raise SolveFailed

    x, y = opt.gens
    p, q = G
    if not p.gcd(q).is_ground:
        # not 0-dimensional
        raise SolveFailed

    p = Poly(p, x, expand=False)
    p_roots = [rcollect(expr, y) for expr in roots(p).keys()]

    q = q.ltrim(-1)
    q_roots = list(roots(q).keys())

    solutions = [(p_root.subs(y, q_root), q_root) for q_root, p_root in
                 itertools.product(q_roots, p_roots)]

    return sorted(solutions, key=default_sort_key)


def solve_generic(polys, opt, strict=False):
    """
    Solve a generic system of polynomial equations.

    Returns all possible solutions over C[x_1, x_2, ..., x_m] of a
    set F = { f_1, f_2, ..., f_n } of polynomial equations, using
    Groebner basis approach. For now only zero-dimensional systems
    are supported, which means F can have at most a finite number
    of solutions. If the basis contains only the ground, None is
    returned.

    The algorithm works by the fact that, supposing G is the basis
    of F with respect to an elimination order (here lexicographic
    order is used), G and F generate the same ideal, they have the
    same set of solutions. By the elimination property, if G is a
    reduced, zero-dimensional Groebner basis, then there exists an
    univariate polynomial in G (in its last variable). This can be
    solved by computing its roots. Substituting all computed roots
    for the last (eliminated) variable in other elements of G, new
    polynomial system is generated. Applying the above procedure
    recursively, a finite number of solutions can be found.

    The ability of finding all solutions by this procedure depends
    on the root finding algorithms. If no solutions were found, it
    means only that roots() failed, but the system is solvable. To
    overcome this difficulty use numerical algorithms instead.

    Parameters
    ==========

    polys: a list/tuple/set
        Listing all the polynomial equations that are needed to be solved
    opt: an Options object
        For specifying keyword arguments and generators
    strict: a boolean
        If strict is True, NotImplementedError will be raised if the solution
        is known to be incomplete

    Returns
    =======

    List[Tuple]
        a list of tuples with elements being solutions for the
        symbols in the order they were passed as gens
    None
        None is returned when the computed basis contains only the ground.

    References
    ==========

    .. [Buchberger01] B. Buchberger, Groebner Bases: A Short
    Introduction for Systems Theorists, In: R. Moreno-Diaz,
    B. Buchberger, J.L. Freire, Proceedings of EUROCAST'01,
    February, 2001

    .. [Cox97] D. Cox, J. Little, D. O'Shea, Ideals, Varieties
    and Algorithms, Springer, Second Edition, 1997, pp. 112

    Raises
    ========

    NotImplementedError
        If the system is not zero-dimensional (does not have a finite
        number of solutions)

    UnsolvableFactorError
        If ``strict`` is True and not all solution components are
        expressible in radicals

    Examples
    ========

    >>> from sympy import Poly, Options
    >>> from sympy.solvers.polysys import solve_generic
    >>> from sympy.abc import x, y
    >>> NewOption = Options((x, y), {'domain': 'ZZ'})

    >>> a = Poly(x - y + 5, x, y, domain='ZZ')
    >>> b = Poly(x + y - 3, x, y, domain='ZZ')
    >>> solve_generic([a, b], NewOption)
    [(-1, 4)]

    >>> a = Poly(x - 2*y + 5, x, y, domain='ZZ')
    >>> b = Poly(2*x - y - 3, x, y, domain='ZZ')
    >>> solve_generic([a, b], NewOption)
    [(11/3, 13/3)]

    >>> a = Poly(x**2 + y, x, y, domain='ZZ')
    >>> b = Poly(x + y*4, x, y, domain='ZZ')
    >>> solve_generic([a, b], NewOption)
    [(0, 0), (1/4, -1/16)]

    >>> a = Poly(x**5 - x + y**3, x, y, domain='ZZ')
    >>> b = Poly(y**2 - 1, x, y, domain='ZZ')
    >>> solve_generic([a, b], NewOption, strict=True)
    Traceback (most recent call last):
    ...
    UnsolvableFactorError

    """
    def _is_univariate(f):
        """Returns True if 'f' is univariate in its last variable. """
        for monom in f.monoms():
            if any(monom[:-1]):
                return False

        return True

    def _subs_root(f, gen, zero):
        """Replace generator with a root so that the result is nice. """
        p = f.as_expr({gen: zero})

        if f.degree(gen) >= 2:
            p = p.expand(deep=False)

        return p

    def _solve_reduced_system(system, gens, entry=False):
        """Recursively solves reduced polynomial systems. """
        if len(system) == len(gens) == 1:
            # the below line will produce UnsolvableFactorError if
            # strict=True and the solution from `roots` is incomplete
            zeros = list(roots(system[0], gens[-1], strict=strict).keys())
            return [(zero,) for zero in zeros]

        basis = groebner(system, gens, polys=True)

        if len(basis) == 1 and basis[0].is_ground:
            if not entry:
                return []
            else:
                return None

        univariate = list(filter(_is_univariate, basis))

        if len(basis) < len(gens):
            raise NotImplementedError(filldedent('''
                only zero-dimensional systems supported
                (finite number of solutions)
                '''))

        if len(univariate) == 1:
            f = univariate.pop()
        else:
            raise NotImplementedError(filldedent('''
                only zero-dimensional systems supported
                (finite number of solutions)
                '''))

        gens = f.gens
        gen = gens[-1]

        # the below line will produce UnsolvableFactorError if
        # strict=True and the solution from `roots` is incomplete
        zeros = list(roots(f.ltrim(gen), strict=strict).keys())

        if not zeros:
            return []

        if len(basis) == 1:
            return [(zero,) for zero in zeros]

        solutions = []

        for zero in zeros:
            new_system = []
            new_gens = gens[:-1]

            for b in basis[:-1]:
                eq = _subs_root(b, gen, zero)

                if eq is not S.Zero:
                    new_system.append(eq)

            for solution in _solve_reduced_system(new_system, new_gens):
                solutions.append(solution + (zero,))

        if solutions and len(solutions[0]) != len(gens):
            raise NotImplementedError(filldedent('''
                only zero-dimensional systems supported
                (finite number of solutions)
                '''))
        return solutions

    try:
        result = _solve_reduced_system(polys, opt.gens, entry=True)
    except CoercionFailed:
        raise NotImplementedError

    if result is not None:
        return sorted(result, key=default_sort_key)


def solve_triangulated(polys, *gens, **args):
    """
    Solve a polynomial system using Gianni-Kalkbrenner algorithm.

    The algorithm proceeds by computing one Groebner basis in the ground
    domain and then by iteratively computing polynomial factorizations in
    appropriately constructed algebraic extensions of the ground domain.

    Parameters
    ==========

    polys: a list/tuple/set
        Listing all the equations that are needed to be solved
    gens: generators
        generators of the equations in polys for which we want the
        solutions
    args: Keyword arguments
        Special options for solving the equations

    Returns
    =======

    List[Tuple]
        A List of tuples. Solutions for symbols that satisfy the
        equations listed in polys

    Examples
    ========

    >>> from sympy import solve_triangulated
    >>> from sympy.abc import x, y, z

    >>> F = [x**2 + y + z - 1, x + y**2 + z - 1, x + y + z**2 - 1]

    >>> solve_triangulated(F, x, y, z)
    [(0, 0, 1), (0, 1, 0), (1, 0, 0)]

    Using extension for algebraic solutions.

    >>> solve_triangulated(F, x, y, z, extension=True) #doctest: +NORMALIZE_WHITESPACE
    [(0, 0, 1), (0, 1, 0), (1, 0, 0),
     (CRootOf(x**2 + 2*x - 1, 0), CRootOf(x**2 + 2*x - 1, 0), CRootOf(x**2 + 2*x - 1, 0)),
     (CRootOf(x**2 + 2*x - 1, 1), CRootOf(x**2 + 2*x - 1, 1), CRootOf(x**2 + 2*x - 1, 1))]

    References
    ==========

    1. Patrizia Gianni, Teo Mora, Algebraic Solution of System of
    Polynomial Equations using Groebner Bases, AAECC-5 on Applied Algebra,
    Algebraic Algorithms and Error-Correcting Codes, LNCS 356 247--257, 1989

    """
    opt = build_options(gens, args)

    G = groebner(polys, gens, polys=True)
    G = list(reversed(G))

    extension = opt.get('extension', False)
    if extension:
        def _solve_univariate(f):
            return [r for r, _ in f.all_roots(multiple=False, radicals=False)]
    else:
        domain = opt.get('domain')

        if domain is not None:
            for i, g in enumerate(G):
                G[i] = g.set_domain(domain)

        def _solve_univariate(f):
            return list(f.ground_roots().keys())

    f, G = G[0].ltrim(-1), G[1:]
    dom = f.get_domain()

    zeros = _solve_univariate(f)

    if extension:
        solutions = {((zero,), dom.algebraic_field(zero)) for zero in zeros}
    else:
        solutions = {((zero,), dom) for zero in zeros}

    var_seq = reversed(gens[:-1])
    vars_seq = postfixes(gens[1:])

    for var, vars in zip(var_seq, vars_seq):
        _solutions = set()

        for values, dom in solutions:
            H, mapping = [], list(zip(vars, values))

            for g in G:
                _vars = (var,) + vars

                if g.has_only_gens(*_vars) and g.degree(var) != 0:
                    if extension:
                        g = g.set_domain(g.domain.unify(dom))
                    h = g.ltrim(var).eval(dict(mapping))

                    if g.degree(var) == h.degree():
                        H.append(h)

            p = min(H, key=lambda h: h.degree())
            zeros = _solve_univariate(p)

            for zero in zeros:
                if not (zero in dom):
                    dom_zero = dom.algebraic_field(zero)
                else:
                    dom_zero = dom

                _solutions.add(((zero,) + values, dom_zero))

        solutions = _solutions
    return sorted((s for s, _ in solutions), key=default_sort_key)


def factor_system(eqs: Sequence[Expr | complex], gens: Sequence[Expr] = (), **kwargs: Any) -> list[list[Expr]]:
    """
    Factorizes a system of polynomial equations into
    irreducible subsystems.

    Parameters
    ==========

    eqs : list
        List of expressions to be factored.
        Each expression is assumed to be equal to zero.

    gens : list, optional
        Generator(s) of the polynomial ring.
        If not provided, all free symbols will be used.

    **kwargs : dict, optional
        Same optional arguments taken by ``factor``

    Returns
    =======

    list[list[Expr]]
        A list of lists of expressions, where each sublist represents
        an irreducible subsystem. When solved, each subsystem gives
        one component of the solution. Only generic solutions are
        returned (cases not requiring parameters to be zero).

    Examples
    ========

    >>> from sympy.solvers.polysys import factor_system, factor_system_cond
    >>> from sympy.abc import x, y, a, b, c

    A simple system with multiple solutions:

    >>> factor_system([x**2 - 1, y - 1])
    [[x + 1, y - 1], [x - 1, y - 1]]

    A system with no solution:

    >>> factor_system([x, 1])
    []

    A system where any value of the symbol(s) is a solution:

    >>> factor_system([x - x, (x + 1)**2 - (x**2 + 2*x + 1)])
    [[]]

    A system with no generic solution:

    >>> factor_system([a*x*(x-1), b*y, c], [x, y])
    []

    If c is added to the unknowns then the system has a generic solution:

    >>> factor_system([a*x*(x-1), b*y, c], [x, y, c])
    [[x - 1, y, c], [x, y, c]]

    Alternatively :func:`factor_system_cond` can be used to get degenerate
    cases as well:

    >>> factor_system_cond([a*x*(x-1), b*y, c], [x, y])
    [[x - 1, y, c], [x, y, c], [x - 1, b, c], [x, b, c], [y, a, c], [a, b, c]]

    Each of the above cases is only satisfiable in the degenerate case `c = 0`.

    The solution set of the original system represented
    by eqs is the union of the solution sets of the
    factorized systems.

    An empty list [] means no generic solution exists.
    A list containing an empty list [[]] means any value of
    the symbol(s) is a solution.

    See Also
    ========

    factor_system_cond : Returns both generic and degenerate solutions
    factor_system_bool : Returns a Boolean combination representing all solutions
    sympy.polys.polytools.factor : Factors a polynomial into irreducible factors
                                   over the rational numbers
    """

    systems = _factor_system_poly_from_expr(eqs, gens, **kwargs)
    systems_generic = [sys for sys in systems if not _is_degenerate(sys)]
    systems_expr = [[p.as_expr() for p in system] for system in systems_generic]
    return systems_expr


def _is_degenerate(system: list[Poly]) -> bool:
    """Helper function to check if a system is degenerate"""
    return any(p.is_ground for p in system)


def factor_system_bool(eqs: Sequence[Expr | complex], gens: Sequence[Expr] = (), **kwargs: Any) -> Boolean:
    """
    Factorizes a system of polynomial equations into irreducible DNF.

    The system of expressions(eqs) is taken and a Boolean combination
    of equations is returned that represents the same solution set.
    The result is in disjunctive normal form (OR of ANDs).

    Parameters
    ==========

    eqs : list
       List of expressions to be factored.
       Each expression is assumed to be equal to zero.

    gens : list, optional
       Generator(s) of the polynomial ring.
       If not provided, all free symbols will be used.

    **kwargs : dict, optional
       Optional keyword arguments


    Returns
    =======

    Boolean:
       A Boolean combination of equations. The result is typically in
       the form of a conjunction (AND) of a disjunctive normal form
       with additional conditions.

    Examples
    ========

    >>> from sympy.solvers.polysys import factor_system_bool
    >>> from sympy.abc import x, y, a, b, c
    >>> factor_system_bool([x**2 - 1])
    Eq(x - 1, 0) | Eq(x + 1, 0)

    >>> factor_system_bool([x**2 - 1, y - 1])
    (Eq(x - 1, 0) & Eq(y - 1, 0)) | (Eq(x + 1, 0) & Eq(y - 1, 0))

    >>> eqs = [a * (x - 1), b]
    >>> factor_system_bool([a*(x - 1), b])
    (Eq(a, 0) & Eq(b, 0)) | (Eq(b, 0) & Eq(x - 1, 0))

    >>> factor_system_bool([a*x**2 - a, b*(x + 1), c], [x])
    (Eq(c, 0) & Eq(x + 1, 0)) | (Eq(a, 0) & Eq(b, 0) & Eq(c, 0)) | (Eq(b, 0) & Eq(c, 0) & Eq(x - 1, 0))

    >>> factor_system_bool([x**2 + 2*x + 1 - (x + 1)**2])
    True

    The result is logically equivalent to the system of equations
    i.e. eqs. The function returns ``True`` when all values of
    the symbol(s) is a solution and ``False`` when the system
    cannot be solved.

    See Also
    ========

    factor_system : Returns factors and solvability condition separately
    factor_system_cond : Returns both factors and conditions

    """

    systems = factor_system_cond(eqs, gens, **kwargs)
    return Or(*[And(*[Eq(eq, 0) for eq in sys]) for sys in systems])


def factor_system_cond(eqs: Sequence[Expr | complex], gens: Sequence[Expr] = (), **kwargs: Any) -> list[list[Expr]]:
    """
    Factorizes a polynomial system into irreducible components and returns
    both generic and degenerate solutions.

    Parameters
    ==========

    eqs : list
        List of expressions to be factored.
        Each expression is assumed to be equal to zero.

    gens : list, optional
        Generator(s) of the polynomial ring.
        If not provided, all free symbols will be used.

    **kwargs : dict, optional
        Optional keyword arguments.

    Returns
    =======

    list[list[Expr]]
        A list of lists of expressions, where each sublist represents
        an irreducible subsystem. Includes both generic solutions and
        degenerate cases requiring equality conditions on parameters.

    Examples
    ========

    >>> from sympy.solvers.polysys import factor_system_cond
    >>> from sympy.abc import x, y, a, b, c

    >>> factor_system_cond([x**2 - 4, a*y, b], [x, y])
    [[x + 2, y, b], [x - 2, y, b], [x + 2, a, b], [x - 2, a, b]]

    >>> factor_system_cond([a*x*(x-1), b*y, c], [x, y])
    [[x - 1, y, c], [x, y, c], [x - 1, b, c], [x, b, c], [y, a, c], [a, b, c]]

    An empty list [] means no solution exists.
    A list containing an empty list [[]] means any value of
    the symbol(s) is a solution.

    See Also
    ========

    factor_system : Returns only generic solutions
    factor_system_bool : Returns a Boolean combination representing all solutions
    sympy.polys.polytools.factor : Factors a polynomial into irreducible factors
                                   over the rational numbers
    """
    systems_poly = _factor_system_poly_from_expr(eqs, gens, **kwargs)
    systems = [[p.as_expr() for p in system] for system in systems_poly]
    return systems


def _factor_system_poly_from_expr(
        eqs: Sequence[Expr | complex], gens: Sequence[Expr], **kwargs: Any
) -> list[list[Poly]]:
    """
    Convert expressions to polynomials and factor the system.

    Takes a sequence of expressions, converts them to
    polynomials, and factors the resulting system. Handles both regular
    polynomial systems and purely numerical cases.
    """
    try:
        polys, opts = parallel_poly_from_expr(eqs, *gens, **kwargs)
        only_numbers = False
    except (GeneratorsNeeded, PolificationFailed):
        _u = Dummy('u')
        polys, opts = parallel_poly_from_expr(eqs, [_u], **kwargs)
        assert opts['domain'].is_Numerical
        only_numbers = True

    if only_numbers:
        return [[]] if all(p == 0 for p in polys) else []

    return factor_system_poly(polys)


def factor_system_poly(polys: list[Poly]) -> list[list[Poly]]:
    """
    Factors a system of polynomial equations into irreducible subsystems

    Core implementation that works directly with Poly instances.

    Parameters
    ==========

    polys : list[Poly]
        A list of Poly instances to be factored.

    Returns
    =======

    list[list[Poly]]
        A list of lists of polynomials, where each sublist represents
        an irreducible component of the solution. Includes both
        generic and degenerate cases.

    Examples
    ========

    >>> from sympy import symbols, Poly, ZZ
    >>> from sympy.solvers.polysys import factor_system_poly
    >>> a, b, c, x = symbols('a b c x')
    >>> p1 = Poly((a - 1)*(x - 2), x, domain=ZZ[a,b,c])
    >>> p2 = Poly((b - 3)*(x - 2), x, domain=ZZ[a,b,c])
    >>> p3 = Poly(c, x, domain=ZZ[a,b,c])

    The equation to be solved for x is ``x - 2 = 0`` provided either
    of the two conditions on the parameters ``a`` and ``b`` is nonzero
    and the constant parameter ``c`` should be zero.

    >>> sys1, sys2 = factor_system_poly([p1, p2, p3])
    >>> sys1
    [Poly(x - 2, x, domain='ZZ[a,b,c]'),
     Poly(c, x, domain='ZZ[a,b,c]')]
    >>> sys2
    [Poly(a - 1, x, domain='ZZ[a,b,c]'),
     Poly(b - 3, x, domain='ZZ[a,b,c]'),
     Poly(c, x, domain='ZZ[a,b,c]')]

     An empty list [] when returned means no solution exists.
     Whereas a list containing an empty list [[]] means any value is a solution.

    See Also
    ========

    factor_system : Returns only generic solutions
    factor_system_bool : Returns a Boolean combination representing the solutions
    factor_system_cond : Returns both generic and degenerate solutions
    sympy.polys.polytools.factor : Factors a polynomial into irreducible factors
                                   over the rational numbers
    """
    if not all(isinstance(poly, Poly) for poly in polys):
        raise TypeError("polys should be a list of Poly instances")
    if not polys:
        return [[]]

    base_domain = polys[0].domain
    base_gens = polys[0].gens
    if not all(poly.domain == base_domain and poly.gens == base_gens for poly in polys[1:]):
        raise DomainError("All polynomials must have the same domain and generators")

    factor_sets = []
    for poly in polys:
        constant, factors_mult = poly.factor_list()

        if constant.is_zero is True:
            continue
        elif constant.is_zero is False:
            if not factors_mult:
                return []
            factor_sets.append([f for f, _ in factors_mult])
        else:
            constant = sqf_part(factor_terms(constant).as_coeff_Mul()[1])
            constp = Poly(constant, base_gens, domain=base_domain)
            factors = [f for f, _ in factors_mult]
            factors.append(constp)
            factor_sets.append(factors)

    if not factor_sets:
        return [[]]

    result = _factor_sets(factor_sets)
    return _sort_systems(result)


def _factor_sets_slow(eqs: list[list]) -> set[frozenset]:
    """
    Helper to find the minimal set of factorised subsystems that is
    equivalent to the original system.

    The result is in DNF.
    """
    if not eqs:
        return {frozenset()}
    systems_set = {frozenset(sys) for sys in cartes(*eqs)}
    return {s1 for s1 in systems_set if not any(s1 > s2 for s2 in systems_set)}


def _factor_sets(eqs: list[list]) -> set[frozenset]:
    """
    Helper that builds factor combinations.
    """
    if not eqs:
        return {frozenset()}

    current_set = min(eqs, key=len)
    other_sets = [s for s in eqs if s is not current_set]

    stack = [(factor, [s for s in other_sets if factor not in s], {factor})
             for factor in current_set]

    result = set()

    while stack:
        factor, remaining_sets, current_solution = stack.pop()

        if not remaining_sets:
            result.add(frozenset(current_solution))
            continue

        next_set = min(remaining_sets, key=len)
        next_remaining = [s for s in remaining_sets if s is not next_set]

        for next_factor in next_set:
            valid_remaining = [s for s in next_remaining if next_factor not in s]
            new_solution = current_solution | {next_factor}
            stack.append((next_factor, valid_remaining, new_solution))

    return {s1 for s1 in result if not any(s1 > s2 for s2 in result)}


def _sort_systems(systems: Iterable[Iterable[Poly]]) -> list[list[Poly]]:
    """Sorts a list of lists of polynomials"""
    systems_list = [sorted(s, key=_poly_sort_key, reverse=True) for s in systems]
    return sorted(systems_list, key=_sys_sort_key, reverse=True)


def _poly_sort_key(poly):
    """Sort key for polynomials"""
    if poly.domain.is_FF:
        poly = poly.set_domain(ZZ)
    return poly.degree_list(), poly.rep.to_list()


def _sys_sort_key(sys):
    """Sort key for lists of polynomials"""
    return list(zip(*map(_poly_sort_key, sys)))
<|endoftext|>from sympy.core.add import Add
from sympy.core.exprtools import factor_terms
from sympy.core.function import expand_log, _mexpand
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.miscellaneous import root
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly, factor
from sympy.simplify.simplify import separatevars
from sympy.simplify.radsimp import collect
from sympy.simplify.simplify import powsimp
from sympy.solvers.solvers import solve, _invert
from sympy.utilities.iterables import uniq


def _filtered_gens(poly, symbol):
    """process the generators of ``poly``, returning the set of generators that
    have ``symbol``.  If there are two generators that are inverses of each other,
    prefer the one that has no denominator.

    Examples
    ========

    >>> from sympy.solvers.bivariate import _filtered_gens
    >>> from sympy import Poly, exp
    >>> from sympy.abc import x
    >>> _filtered_gens(Poly(x + 1/x + exp(x)), x)
    {x, exp(x)}

    """
    # TODO it would be good to pick the smallest divisible power
    # instead of the base for something like x**4 + x**2 -->
    # return x**2 not x
    gens = {g for g in poly.gens if symbol in g.free_symbols}
    for g in list(gens):
        ag = 1/g
        if g in gens and ag in gens:
            if ag.as_numer_denom()[1] is not S.One:
                g = ag
            gens.remove(g)
    return gens


def _mostfunc(lhs, func, X=None):
    """Returns the term in lhs which contains the most of the
    func-type things e.g. log(log(x)) wins over log(x) if both terms appear.

    ``func`` can be a function (exp, log, etc...) or any other SymPy object,
    like Pow.

    If ``X`` is not ``None``, then the function returns the term composed with the
    most ``func`` having the specified variable.

    Examples
    ========

    >>> from sympy.solvers.bivariate import _mostfunc
    >>> from sympy import exp
    >>> from sympy.abc import x, y
    >>> _mostfunc(exp(x) + exp(exp(x) + 2), exp)
    exp(exp(x) + 2)
    >>> _mostfunc(exp(x) + exp(exp(y) + 2), exp)
    exp(exp(y) + 2)
    >>> _mostfunc(exp(x) + exp(exp(y) + 2), exp, x)
    exp(x)
    >>> _mostfunc(x, exp, x) is None
    True
    >>> _mostfunc(exp(x) + exp(x*y), exp, x)
    exp(x)
    """
    fterms = [tmp for tmp in lhs.atoms(func) if (not X or
        X.is_Symbol and X in tmp.free_symbols or
        not X.is_Symbol and tmp.has(X))]
    if len(fterms) == 1:
        return fterms[0]
    elif fterms:
        return max(list(ordered(fterms)), key=lambda x: x.count(func))
    return None


def _linab(arg, symbol):
    """Return ``a, b, X`` assuming ``arg`` can be written as ``a*X + b``
    where ``X`` is a symbol-dependent factor and ``a`` and ``b`` are
    independent of ``symbol``.

    Examples
    ========

    >>> from sympy.solvers.bivariate import _linab
    >>> from sympy.abc import x, y
    >>> from sympy import exp, S
    >>> _linab(S(2), x)
    (2, 0, 1)
    >>> _linab(2*x, x)
    (2, 0, x)
    >>> _linab(y + y*x + 2*x, x)
    (y + 2, y, x)
    >>> _linab(3 + 2*exp(x), x)
    (2, 3, exp(x))
    """
    arg = factor_terms(arg.expand())
    ind, dep = arg.as_independent(symbol)
    if arg.is_Mul and dep.is_Add:
        a, b, x = _linab(dep, symbol)
        return ind*a, ind*b, x
    if not arg.is_Add:
        b = 0
        a, x = ind, dep
    else:
        b = ind
        a, x = separatevars(dep).as_independent(symbol, as_Add=False)
    if x.could_extract_minus_sign():
        a = -a
        x = -x
    return a, b, x


def _lambert(eq, x):
    """
    Given an expression assumed to be in the form
        ``F(X, a..f) = a*log(b*X + c) + d*X + f = 0``
    where X = g(x) and x = g^-1(X), return the Lambert solution,
        ``x = g^-1(-c/b + (a/d)*W(d/(a*b)*exp(c*d/a/b)*exp(-f/a)))``.
    """
    eq = _mexpand(expand_log(eq))
    mainlog = _mostfunc(eq, log, x)
    if not mainlog:
        return []  # violated assumptions
    other = eq.subs(mainlog, 0)
    if isinstance(-other, log):
        eq = (eq - other).subs(mainlog, mainlog.args[0])
        mainlog = mainlog.args[0]
        if not isinstance(mainlog, log):
            return []  # violated assumptions
        other = -(-other).args[0]
        eq += other
    if x not in other.free_symbols:
        return [] # violated assumptions
    d, f, X2 = _linab(other, x)
    logterm = collect(eq - other, mainlog)
    a = logterm.as_coefficient(mainlog)
    if a is None or x in a.free_symbols:
        return []  # violated assumptions
    logarg = mainlog.args[0]
    b, c, X1 = _linab(logarg, x)
    if X1 != X2:
        return []  # violated assumptions

    # invert the generator X1 so we have x(u)
    u = Dummy('rhs')
    xusolns = solve(X1 - u, x)

    # There are infinitely many branches for LambertW
    # but only branches for k = -1 and 0 might be real. The k = 0
    # branch is real and the k = -1 branch is real if the LambertW argument
    # in in range [-1/e, 0]. Since `solve` does not return infinite
    # solutions we will only include the -1 branch if it tests as real.
    # Otherwise, inclusion of any LambertW in the solution indicates to
    #  the user that there are imaginary solutions corresponding to
    # different k values.
    lambert_real_branches = [-1, 0]
    sol = []

    # solution of the given Lambert equation is like
    # sol = -c/b + (a/d)*LambertW(arg, k),
    # where arg = d/(a*b)*exp((c*d-b*f)/a/b) and k in lambert_real_branches.
    # Instead of considering the single arg, `d/(a*b)*exp((c*d-b*f)/a/b)`,
    # the individual `p` roots obtained when writing `exp((c*d-b*f)/a/b)`
    # as `exp(A/p) = exp(A)**(1/p)`, where `p` is an Integer, are used.

    # calculating args for LambertW
    num, den = ((c*d-b*f)/a/b).as_numer_denom()
    p, den = den.as_coeff_Mul()
    e = exp(num/den)
    t = Dummy('t')
    args = [d/(a*b)*t for t in roots(t**p - e, t).keys()]

    # calculating solutions from args
    for arg in args:
        for k in lambert_real_branches:
            w = LambertW(arg, k)
            if k and not w.is_real:
                continue
            rhs = -c/b + (a/d)*w

            sol.extend(xu.subs(u, rhs) for xu in xusolns)
    return sol


def _solve_lambert(f, symbol, gens):
    """Return solution to ``f`` if it is a Lambert-type expression
    else raise NotImplementedError.

    For ``f(X, a..f) = a*log(b*X + c) + d*X - f = 0`` the solution
    for ``X`` is ``X = -c/b + (a/d)*W(d/(a*b)*exp(c*d/a/b)*exp(f/a))``.
    There are a variety of forms for `f(X, a..f)` as enumerated below:

    1a1)
      if B**B = R for R not in [0, 1] (since those cases would already
      be solved before getting here) then log of both sides gives
      log(B) + log(log(B)) = log(log(R)) and
      X = log(B), a = 1, b = 1, c = 0, d = 1, f = log(log(R))
    1a2)
      if B*(b*log(B) + c)**a = R then log of both sides gives
      log(B) + a*log(b*log(B) + c) = log(R) and
      X = log(B), d=1, f=log(R)
    1b)
      if a*log(b*B + c) + d*B = R and
      X = B, f = R
    2a)
      if (b*B + c)*exp(d*B + g) = R then log of both sides gives
      log(b*B + c) + d*B + g = log(R) and
      X = B, a = 1, f = log(R) - g
    2b)
      if g*exp(d*B + h) - b*B = c then the log form is
      log(g) + d*B + h - log(b*B + c) = 0 and
      X = B, a = -1, f = -h - log(g)
    3)
      if d*p**(a*B + g) - b*B = c then the log form is
      log(d) + (a*B + g)*log(p) - log(b*B + c) = 0 and
      X = B, a = -1, d = a*log(p), f = -log(d) - g*log(p)
    """

    def _solve_even_degree_expr(expr, t, symbol):
        """Return the unique solutions of equations derived from
        ``expr`` by replacing ``t`` with ``+/- symbol``.

        Parameters
        ==========

        expr : Expr
            The expression which includes a dummy variable t to be
            replaced with +symbol and -symbol.

        symbol : Symbol
            The symbol for which a solution is being sought.

        Returns
        =======

        List of unique solution of the two equations generated by
        replacing ``t`` with positive and negative ``symbol``.

        Notes
        =====

        If ``expr = 2*log(t) + x/2` then solutions for
        ``2*log(x) + x/2 = 0`` and ``2*log(-x) + x/2 = 0`` are
        returned by this function. Though this may seem
        counter-intuitive, one must note that the ``expr`` being
        solved here has been derived from a different expression. For
        an expression like ``eq = x**2*g(x) = 1``, if we take the
        log of both sides we obtain ``log(x**2) + log(g(x)) = 0``. If
        x is positive then this simplifies to
        ``2*log(x) + log(g(x)) = 0``; the Lambert-solving routines will
        return solutions for this, but we must also consider the
        solutions for  ``2*log(-x) + log(g(x))`` since those must also
        be a solution of ``eq`` which has the same value when the ``x``
        in ``x**2`` is negated. If `g(x)` does not have even powers of
        symbol then we do not want to replace the ``x`` there with
        ``-x``. So the role of the ``t`` in the expression received by
        this function is to mark where ``+/-x`` should be inserted
        before obtaining the Lambert solutions.

        """
        nlhs, plhs = [
            expr.xreplace({t: sgn*symbol}) for sgn in (-1, 1)]
        sols = _solve_lambert(nlhs, symbol, gens)
        if plhs != nlhs:
            sols.extend(_solve_lambert(plhs, symbol, gens))
        # uniq is needed for a case like
        # 2*log(t) - log(-z**2) + log(z + log(x) + log(z))
        # where substituting t with +/-x gives all the same solution;
        # uniq, rather than list(set()), is used to maintain canonical
        # order
        return list(uniq(sols))

    nrhs, lhs = f.as_independent(symbol, as_Add=True)
    rhs = -nrhs

    lamcheck = [tmp for tmp in gens
                if (tmp.func in [exp, log] or
                (tmp.is_Pow and symbol in tmp.exp.free_symbols))]
    if not lamcheck:
        raise NotImplementedError()

    if lhs.is_Add or lhs.is_Mul:
        # replacing all even_degrees of symbol with dummy variable t
        # since these will need special handling; non-Add/Mul do not
        # need this handling
        t = Dummy('t', **symbol.assumptions0)
        lhs = lhs.replace(
            lambda i:  # find symbol**even
                i.is_Pow and i.base == symbol and i.exp.is_even,
            lambda i:  # replace t**even
                t**i.exp)

        if lhs.is_Add and lhs.has(t):
            t_indep = lhs.subs(t, 0)
            t_term = lhs - t_indep
            _rhs = rhs - t_indep
            if not t_term.is_Add and _rhs and not (
                    t_term.has(S.ComplexInfinity, S.NaN)):
                eq = expand_log(log(t_term) - log(_rhs))
                return _solve_even_degree_expr(eq, t, symbol)
        elif lhs.is_Mul and rhs:
            # this needs to happen whether t is present or not
            lhs = expand_log(log(lhs), force=True)
            rhs = log(rhs)
            if lhs.has(t) and lhs.is_Add:
                # it expanded from Mul to Add
                eq = lhs - rhs
                return _solve_even_degree_expr(eq, t, symbol)

        # restore symbol in lhs
        lhs = lhs.xreplace({t: symbol})

    lhs = powsimp(factor(lhs, deep=True))

    # make sure we have inverted as completely as possible
    r = Dummy()
    i, lhs = _invert(lhs - r, symbol)
    rhs = i.xreplace({r: rhs})

    # For the first forms:
    #
    # 1a1) B**B = R will arrive here as B*log(B) = log(R)
    #      lhs is Mul so take log of both sides:
    #        log(B) + log(log(B)) = log(log(R))
    # 1a2) B*(b*log(B) + c)**a = R will arrive unchanged so
    #      lhs is Mul, so take log of both sides:
    #        log(B) + a*log(b*log(B) + c) = log(R)
    # 1b) d*log(a*B + b) + c*B = R will arrive unchanged so
    #      lhs is Add, so isolate c*B and expand log of both sides:
    #        log(c) + log(B) = log(R - d*log(a*B + b))

    soln = []
    if not soln:
        mainlog = _mostfunc(lhs, log, symbol)
        if mainlog:
            if lhs.is_Mul and rhs != 0:
                soln = _lambert(log(lhs) - log(rhs), symbol)
            elif lhs.is_Add:
                other = lhs.subs(mainlog, 0)
                if other and not other.is_Add and [
                        tmp for tmp in other.atoms(Pow)
                        if symbol in tmp.free_symbols]:
                    if not rhs:
                        diff = log(other) - log(other - lhs)
                    else:
                        diff = log(lhs - other) - log(rhs - other)
                    soln = _lambert(expand_log(diff), symbol)
                else:
                    #it's ready to go
                    soln = _lambert(lhs - rhs, symbol)

    # For the next forms,
    #
    #     collect on main exp
    #     2a) (b*B + c)*exp(d*B + g) = R
    #         lhs is mul, so take log of both sides:
    #           log(b*B + c) + d*B = log(R) - g
    #     2b) g*exp(d*B + h) - b*B = R
    #         lhs is add, so add b*B to both sides,
    #         take the log of both sides and rearrange to give
    #           log(R + b*B) - d*B = log(g) + h

    if not soln:
        mainexp = _mostfunc(lhs, exp, symbol)
        if mainexp:
            lhs = collect(lhs, mainexp)
            if lhs.is_Mul and rhs != 0:
                soln = _lambert(expand_log(log(lhs) - log(rhs)), symbol)
            elif lhs.is_Add:
                # move all but mainexp-containing term to rhs
                other = lhs.subs(mainexp, 0)
                mainterm = lhs - other
                rhs = rhs - other
                if (mainterm.could_extract_minus_sign() and
                    rhs.could_extract_minus_sign()):
                    mainterm *= -1
                    rhs *= -1
                diff = log(mainterm) - log(rhs)
                soln = _lambert(expand_log(diff), symbol)

    # For the last form:
    #
    #  3) d*p**(a*B + g) - b*B = c
    #     collect on main pow, add b*B to both sides,
    #     take log of both sides and rearrange to give
    #       a*B*log(p) - log(b*B + c) = -log(d) - g*log(p)
    if not soln:
        mainpow = _mostfunc(lhs, Pow, symbol)
        if mainpow and symbol in mainpow.exp.free_symbols:
            lhs = collect(lhs, mainpow)
            if lhs.is_Mul and rhs != 0:
                # b*B = 0
                soln = _lambert(expand_log(log(lhs) - log(rhs)), symbol)
            elif lhs.is_Add:
                # move all but mainpow-containing term to rhs
                other = lhs.subs(mainpow, 0)
                mainterm = lhs - other
                rhs = rhs - other
                diff = log(mainterm) - log(rhs)
                soln = _lambert(expand_log(diff), symbol)

    if not soln:
        raise NotImplementedError('%s does not appear to have a solution in '
            'terms of LambertW' % f)

    return list(ordered(soln))


def bivariate_type(f, x, y, *, first=True):
    """Given an expression, f, 3 tests will be done to see what type
    of composite bivariate it might be, options for u(x, y) are::

        x*y
        x+y
        x*y+x
        x*y+y

    If it matches one of these types, ``u(x, y)``, ``P(u)`` and dummy
    variable ``u`` will be returned. Solving ``P(u)`` for ``u`` and
    equating the solutions to ``u(x, y)`` and then solving for ``x`` or
    ``y`` is equivalent to solving the original expression for ``x`` or
    ``y``. If ``x`` and ``y`` represent two functions in the same
    variable, e.g. ``x = g(t)`` and ``y = h(t)``, then if ``u(x, y) - p``
    can be solved for ``t`` then these represent the solutions to
    ``P(u) = 0`` when ``p`` are the solutions of ``P(u) = 0``.

    Only positive values of ``u`` are considered.

    Examples
    ========

    >>> from sympy import solve
    >>> from sympy.solvers.bivariate import bivariate_type
    >>> from sympy.abc import x, y
    >>> eq = (x**2 - 3).subs(x, x + y)
    >>> bivariate_type(eq, x, y)
    (x + y, _u**2 - 3, _u)
    >>> uxy, pu, u = _
    >>> usol = solve(pu, u); usol
    [sqrt(3)]
    >>> [solve(uxy - s) for s in solve(pu, u)]
    [[{x: -y + sqrt(3)}]]
    >>> all(eq.subs(s).equals(0) for sol in _ for s in sol)
    True

    """

    u = Dummy('u', positive=True)

    if first:
        p = Poly(f, x, y)
        f = p.as_expr()
        _x = Dummy()
        _y = Dummy()
        rv = bivariate_type(Poly(f.subs({x: _x, y: _y}), _x, _y), _x, _y, first=False)
        if rv:
            reps = {_x: x, _y: y}
            return rv[0].xreplace(reps), rv[1].xreplace(reps), rv[2]
        return

    p = f
    f = p.as_expr()

    # f(x*y)
    args = Add.make_args(p.as_expr())
    new = []
    for a in args:
        a = _mexpand(a.subs(x, u/y))
        free = a.free_symbols
        if x in free or y in free:
            break
        new.append(a)
    else:
        return x*y, Add(*new), u

    def ok(f, v, c):
        new = _mexpand(f.subs(v, c))
        free = new.free_symbols
        return None if (x in free or y in free) else new

    # f(a*x + b*y)
    new = []
    d = p.degree(x)
    if p.degree(y) == d:
        a = root(p.coeff_monomial(x**d), d)
        b = root(p.coeff_monomial(y**d), d)
        new = ok(f, x, (u - b*y)/a)
        if new is not None:
            return a*x + b*y, new, u

    # f(a*x*y + b*y)
    new = []
    d = p.degree(x)
    if p.degree(y) == d:
        for itry in range(2):
            a = root(p.coeff_monomial(x**d*y**d), d)
            b = root(p.coeff_monomial(y**d), d)
            new = ok(f, x, (u - b*y)/a/y)
            if new is not None:
                return a*x*y + b*y, new, u
            x, y = y, x
<|endoftext|>r"""
This module is intended for solving recurrences or, in other words,
difference equations. Currently supported are linear, inhomogeneous
equations with polynomial or rational coefficients.

The solutions are obtained among polynomials, rational functions,
hypergeometric terms, or combinations of hypergeometric term which
are pairwise dissimilar.

``rsolve_X`` functions were meant as a low level interface
for ``rsolve`` which would use Mathematica's syntax.

Given a recurrence relation:

    .. math:: a_{k}(n) y(n+k) + a_{k-1}(n) y(n+k-1) +
              ... + a_{0}(n) y(n) = f(n)

where `k > 0` and `a_{i}(n)` are polynomials in `n`. To use
``rsolve_X`` we need to put all coefficients in to a list ``L`` of
`k+1` elements the following way:

    ``L = [a_{0}(n), ..., a_{k-1}(n), a_{k}(n)]``

where ``L[i]``, for `i=0, \ldots, k`, maps to
`a_{i}(n) y(n+i)` (`y(n+i)` is implicit).

For example if we would like to compute `m`-th Bernoulli polynomial
up to a constant (example was taken from rsolve_poly docstring),
then we would use `b(n+1) - b(n) = m n^{m-1}` recurrence, which
has solution `b(n) = B_m + C`.

Then ``L = [-1, 1]`` and `f(n) = m n^(m-1)` and finally for `m=4`:

>>> from sympy import Symbol, bernoulli, rsolve_poly
>>> n = Symbol('n', integer=True)

>>> rsolve_poly([-1, 1], 4*n**3, n)
C0 + n**4 - 2*n**3 + n**2

>>> bernoulli(4, n)
n**4 - 2*n**3 + n**2 - 1/30

For the sake of completeness, `f(n)` can be:

    [1] a polynomial               -> rsolve_poly
    [2] a rational function        -> rsolve_ratio
    [3] a hypergeometric function  -> rsolve_hyper
"""
from collections import defaultdict

from sympy.concrete import product
from sympy.core.singleton import S
from sympy.core.numbers import Rational, I
from sympy.core.symbol import Symbol, Wild, Dummy
from sympy.core.relational import Equality
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import sympify

from sympy.simplify import simplify, hypersimp, hypersimilar  # type: ignore
from sympy.solvers import solve, solve_undetermined_coeffs
from sympy.polys import Poly, quo, gcd, lcm, roots, resultant
from sympy.functions import binomial, factorial, FallingFactorial, RisingFactorial
from sympy.matrices import Matrix, casoratian
from sympy.utilities.iterables import numbered_symbols


def rsolve_poly(coeffs, f, n, shift=0, **hints):
    r"""
    Given linear recurrence operator `\operatorname{L}` of order
    `k` with polynomial coefficients and inhomogeneous equation
    `\operatorname{L} y = f`, where `f` is a polynomial, we seek for
    all polynomial solutions over field `K` of characteristic zero.

    The algorithm performs two basic steps:

        (1) Compute degree `N` of the general polynomial solution.
        (2) Find all polynomials of degree `N` or less
            of `\operatorname{L} y = f`.

    There are two methods for computing the polynomial solutions.
    If the degree bound is relatively small, i.e. it's smaller than
    or equal to the order of the recurrence, then naive method of
    undetermined coefficients is being used. This gives a system
    of algebraic equations with `N+1` unknowns.

    In the other case, the algorithm performs transformation of the
    initial equation to an equivalent one for which the system of
    algebraic equations has only `r` indeterminates. This method is
    quite sophisticated (in comparison with the naive one) and was
    invented together by Abramov, Bronstein and Petkovsek.

    It is possible to generalize the algorithm implemented here to
    the case of linear q-difference and differential equations.

    Lets say that we would like to compute `m`-th Bernoulli polynomial
    up to a constant. For this we can use `b(n+1) - b(n) = m n^{m-1}`
    recurrence, which has solution `b(n) = B_m + C`. For example:

    >>> from sympy import Symbol, rsolve_poly
    >>> n = Symbol('n', integer=True)

    >>> rsolve_poly([-1, 1], 4*n**3, n)
    C0 + n**4 - 2*n**3 + n**2

    References
    ==========

    .. [1] S. A. Abramov, M. Bronstein and M. Petkovsek, On polynomial
           solutions of linear operator equations, in: T. Levelt, ed.,
           Proc. ISSAC '95, ACM Press, New York, 1995, 290-296.

    .. [2] M. Petkovsek, Hypergeometric solutions of linear recurrences
           with polynomial coefficients, J. Symbolic Computation,
           14 (1992), 243-264.

    .. [3] M. Petkovsek, H. S. Wilf, D. Zeilberger, A = B, 1996.

    """
    f = sympify(f)

    if not f.is_polynomial(n):
        return None

    homogeneous = f.is_zero

    r = len(coeffs) - 1

    coeffs = [Poly(coeff, n) for coeff in coeffs]

    polys = [Poly(0, n)]*(r + 1)
    terms = [(S.Zero, S.NegativeInfinity)]*(r + 1)

    for i in range(r + 1):
        for j in range(i, r + 1):
            polys[i] += coeffs[j]*(binomial(j, i).as_poly(n))

        if not polys[i].is_zero:
            (exp,), coeff = polys[i].LT()
            terms[i] = (coeff, exp)

    d = b = terms[0][1]

    for i in range(1, r + 1):
        if terms[i][1] > d:
            d = terms[i][1]

        if terms[i][1] - i > b:
            b = terms[i][1] - i

    d, b = int(d), int(b)

    x = Dummy('x')

    degree_poly = S.Zero

    for i in range(r + 1):
        if terms[i][1] - i == b:
            degree_poly += terms[i][0]*FallingFactorial(x, i)

    nni_roots = list(roots(degree_poly, x, filter='Z',
        predicate=lambda r: r >= 0).keys())

    if nni_roots:
        N = [max(nni_roots)]
    else:
        N = []

    if homogeneous:
        N += [-b - 1]
    else:
        N += [f.as_poly(n).degree() - b, -b - 1]

    N = int(max(N))

    if N < 0:
        if homogeneous:
            if hints.get('symbols', False):
                return (S.Zero, [])
            else:
                return S.Zero
        else:
            return None

    if N <= r:
        C = []
        y = E = S.Zero

        for i in range(N + 1):
            C.append(Symbol('C' + str(i + shift)))
            y += C[i] * n**i

        for i in range(r + 1):
            E += coeffs[i].as_expr()*y.subs(n, n + i)

        solutions = solve_undetermined_coeffs(E - f, C, n)

        if solutions is not None:
            _C = C
            C = [c for c in C if (c not in solutions)]
            result = y.subs(solutions)
        else:
            return None  # TBD
    else:
        A = r
        U = N + A + b + 1

        nni_roots = list(roots(polys[r], filter='Z',
            predicate=lambda r: r >= 0).keys())

        if nni_roots != []:
            a = max(nni_roots) + 1
        else:
            a = S.Zero

        def _zero_vector(k):
            return [S.Zero] * k

        def _one_vector(k):
            return [S.One] * k

        def _delta(p, k):
            B = S.One
            D = p.subs(n, a + k)

            for i in range(1, k + 1):
                B *= Rational(i - k - 1, i)
                D += B * p.subs(n, a + k - i)

            return D

        alpha = {}

        for i in range(-A, d + 1):
            I = _one_vector(d + 1)

            for k in range(1, d + 1):
                I[k] = I[k - 1] * (x + i - k + 1)/k

            alpha[i] = S.Zero

            for j in range(A + 1):
                for k in range(d + 1):
                    B = binomial(k, i + j)
                    D = _delta(polys[j].as_expr(), k)

                    alpha[i] += I[k]*B*D

        V = Matrix(U, A, lambda i, j: int(i == j))

        if homogeneous:
            for i in range(A, U):
                v = _zero_vector(A)

                for k in range(1, A + b + 1):
                    if i - k < 0:
                        break

                    B = alpha[k - A].subs(x, i - k)

                    for j in range(A):
                        v[j] += B * V[i - k, j]

                denom = alpha[-A].subs(x, i)

                for j in range(A):
                    V[i, j] = -v[j] / denom
        else:
            G = _zero_vector(U)

            for i in range(A, U):
                v = _zero_vector(A)
                g = S.Zero

                for k in range(1, A + b + 1):
                    if i - k < 0:
                        break

                    B = alpha[k - A].subs(x, i - k)

                    for j in range(A):
                        v[j] += B * V[i - k, j]

                    g += B * G[i - k]

                denom = alpha[-A].subs(x, i)

                for j in range(A):
                    V[i, j] = -v[j] / denom

                G[i] = (_delta(f, i - A) - g) / denom

        P, Q = _one_vector(U), _zero_vector(A)

        for i in range(1, U):
            P[i] = (P[i - 1] * (n - a - i + 1)/i).expand()

        for i in range(A):
            Q[i] = Add(*[(v*p).expand() for v, p in zip(V[:, i], P)])

        if not homogeneous:
            h = Add(*[(g*p).expand() for g, p in zip(G, P)])

        C = [Symbol('C' + str(i + shift)) for i in range(A)]

        g = lambda i: Add(*[c*_delta(q, i) for c, q in zip(C, Q)])

        if homogeneous:
            E = [g(i) for i in range(N + 1, U)]
        else:
            E = [g(i) + _delta(h, i) for i in range(N + 1, U)]

        if E != []:
            solutions = solve(E, *C)

            if not solutions:
                if homogeneous:
                    if hints.get('symbols', False):
                        return (S.Zero, [])
                    else:
                        return S.Zero
                else:
                    return None
        else:
            solutions = {}

        if homogeneous:
            result = S.Zero
        else:
            result = h

        _C = C[:]
        for c, q in list(zip(C, Q)):
            if c in solutions:
                s = solutions[c]*q
                C.remove(c)
            else:
                s = c*q

            result += s.expand()

    if C != _C:
        # renumber so they are contiguous
        result = result.xreplace(dict(zip(C, _C)))
        C = _C[:len(C)]

    if hints.get('symbols', False):
        return (result, C)
    else:
        return result


def rsolve_ratio(coeffs, f, n, **hints):
    r"""
    Given linear recurrence operator `\operatorname{L}` of order `k`
    with polynomial coefficients and inhomogeneous equation
    `\operatorname{L} y = f`, where `f` is a polynomial, we seek
    for all rational solutions over field `K` of characteristic zero.

    This procedure accepts only polynomials, however if you are
    interested in solving recurrence with rational coefficients
    then use ``rsolve`` which will pre-process the given equation
    and run this procedure with polynomial arguments.

    The algorithm performs two basic steps:

        (1) Compute polynomial `v(n)` which can be used as universal
            denominator of any rational solution of equation
            `\operatorname{L} y = f`.

        (2) Construct new linear difference equation by substitution
            `y(n) = u(n)/v(n)` and solve it for `u(n)` finding all its
            polynomial solutions. Return ``None`` if none were found.

    The algorithm implemented here is a revised version of the original
    Abramov's algorithm, developed in 1989. The new approach is much
    simpler to implement and has better overall efficiency. This
    method can be easily adapted to the q-difference equations case.

    Besides finding rational solutions alone, this functions is
    an important part of Hyper algorithm where it is used to find
    a particular solution for the inhomogeneous part of a recurrence.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.solvers.recurr import rsolve_ratio
    >>> rsolve_ratio([-2*x**3 + x**2 + 2*x - 1, 2*x**3 + x**2 - 6*x,
    ... - 2*x**3 - 11*x**2 - 18*x - 9, 2*x**3 + 13*x**2 + 22*x + 8], 0, x)
    C0*(2*x - 3)/(2*(x**2 - 1))

    References
    ==========

    .. [1] S. A. Abramov, Rational solutions of linear difference
           and q-difference equations with polynomial coefficients,
           in: T. Levelt, ed., Proc. ISSAC '95, ACM Press, New York,
           1995, 285-289

    See Also
    ========

    rsolve_hyper
    """
    f = sympify(f)

    if not f.is_polynomial(n):
        return None

    coeffs = list(map(sympify, coeffs))

    r = len(coeffs) - 1

    A, B = coeffs[r], coeffs[0]
    A = A.subs(n, n - r).expand()

    h = Dummy('h')

    res = resultant(A, B.subs(n, n + h), n)

    if not res.is_polynomial(h):
        p, q = res.as_numer_denom()
        res = quo(p, q, h)

    nni_roots = list(roots(res, h, filter='Z',
        predicate=lambda r: r >= 0).keys())

    if not nni_roots:
        return rsolve_poly(coeffs, f, n, **hints)
    else:
        C, numers = S.One, [S.Zero]*(r + 1)

        for i in range(int(max(nni_roots)), -1, -1):
            d = gcd(A, B.subs(n, n + i), n)

            A = quo(A, d, n)
            B = quo(B, d.subs(n, n - i), n)

            C *= Mul(*[d.subs(n, n - j) for j in range(i + 1)])

        denoms = [C.subs(n, n + i) for i in range(r + 1)]

        for i in range(r + 1):
            g = gcd(coeffs[i], denoms[i], n)

            numers[i] = quo(coeffs[i], g, n)
            denoms[i] = quo(denoms[i], g, n)

        for i in range(r + 1):
            numers[i] *= Mul(*(denoms[:i] + denoms[i + 1:]))

        result = rsolve_poly(numers, f * Mul(*denoms), n, **hints)

        if result is not None:
            if hints.get('symbols', False):
                return (simplify(result[0] / C), result[1])
            else:
                return simplify(result / C)
        else:
            return None


def rsolve_hyper(coeffs, f, n, **hints):
    r"""
    Given linear recurrence operator `\operatorname{L}` of order `k`
    with polynomial coefficients and inhomogeneous equation
    `\operatorname{L} y = f` we seek for all hypergeometric solutions
    over field `K` of characteristic zero.

    The inhomogeneous part can be either hypergeometric or a sum
    of a fixed number of pairwise dissimilar hypergeometric terms.

    The algorithm performs three basic steps:

        (1) Group together similar hypergeometric terms in the
            inhomogeneous part of `\operatorname{L} y = f`, and find
            particular solution using Abramov's algorithm.

        (2) Compute generating set of `\operatorname{L}` and find basis
            in it, so that all solutions are linearly independent.

        (3) Form final solution with the number of arbitrary
            constants equal to dimension of basis of `\operatorname{L}`.

    Term `a(n)` is hypergeometric if it is annihilated by first order
    linear difference equations with polynomial coefficients or, in
    simpler words, if consecutive term ratio is a rational function.

    The output of this procedure is a linear combination of fixed
    number of hypergeometric terms. However the underlying method
    can generate larger class of solutions - D'Alembertian terms.

    Note also that this method not only computes the kernel of the
    inhomogeneous equation, but also reduces in to a basis so that
    solutions generated by this procedure are linearly independent

    Examples
    ========

    >>> from sympy.solvers import rsolve_hyper
    >>> from sympy.abc import x

    >>> rsolve_hyper([-1, -1, 1], 0, x)
    C0*(1/2 - sqrt(5)/2)**x + C1*(1/2 + sqrt(5)/2)**x

    >>> rsolve_hyper([-1, 1], 1 + x, x)
    C0 + x*(x + 1)/2

    References
    ==========

    .. [1] M. Petkovsek, Hypergeometric solutions of linear recurrences
           with polynomial coefficients, J. Symbolic Computation,
           14 (1992), 243-264.

    .. [2] M. Petkovsek, H. S. Wilf, D. Zeilberger, A = B, 1996.
    """
    coeffs = list(map(sympify, coeffs))

    f = sympify(f)

    r, kernel, symbols = len(coeffs) - 1, [], set()

    if not f.is_zero:
        if f.is_Add:
            similar = {}

            for g in f.expand().args:
                if not g.is_hypergeometric(n):
                    return None

                for h in similar.keys():
                    if hypersimilar(g, h, n):
                        similar[h] += g
                        break
                else:
                    similar[g] = S.Zero

            inhomogeneous = [g + h for g, h in similar.items()]
        elif f.is_hypergeometric(n):
            inhomogeneous = [f]
        else:
            return None

        for i, g in enumerate(inhomogeneous):
            coeff, polys = S.One, coeffs[:]
            denoms = [S.One]*(r + 1)

            s = hypersimp(g, n)

            for j in range(1, r + 1):
                coeff *= s.subs(n, n + j - 1)

                p, q = coeff.as_numer_denom()

                polys[j] *= p
                denoms[j] = q

            for j in range(r + 1):
                polys[j] *= Mul(*(denoms[:j] + denoms[j + 1:]))

            # FIXME: The call to rsolve_ratio below should suffice (rsolve_poly
            # call can be removed) but the XFAIL test_rsolve_ratio_missed must
            # be fixed first.
            R = rsolve_ratio(polys, Mul(*denoms), n, symbols=True)
            if R is not None:
                R, syms = R
                if syms:
                    R = R.subs(zip(syms, [0]*len(syms)))
            else:
                R = rsolve_poly(polys, Mul(*denoms), n)

            if R:
                inhomogeneous[i] *= R
            else:
                return None

            result = Add(*inhomogeneous)
            result = simplify(result)
    else:
        result = S.Zero

    Z = Dummy('Z')

    p, q = coeffs[0], coeffs[r].subs(n, n - r + 1)

    p_factors = list(roots(p, n).keys())
    q_factors = list(roots(q, n).keys())

    factors = [(S.One, S.One)]

    for p in p_factors:
        for q in q_factors:
            if p.is_integer and q.is_integer and p <= q:
                continue
            else:
                factors += [(n - p, n - q)]

    p = [(n - p, S.One) for p in p_factors]
    q = [(S.One, n - q) for q in q_factors]

    factors = p + factors + q

    for A, B in factors:
        polys, degrees = [], []
        D = A*B.subs(n, n + r - 1)

        for i in range(r + 1):
            a = Mul(*[A.subs(n, n + j) for j in range(i)])
            b = Mul(*[B.subs(n, n + j) for j in range(i, r)])

            poly = quo(coeffs[i]*a*b, D, n)
            polys.append(poly.as_poly(n))

            if not poly.is_zero:
                degrees.append(polys[i].degree())

        if degrees:
            d, poly = max(degrees), S.Zero
        else:
            return None

        for i in range(r + 1):
            coeff = polys[i].nth(d)

            if coeff is not S.Zero:
                poly += coeff * Z**i

        for z in roots(poly, Z).keys():
            if z.is_zero:
                continue

            recurr_coeffs = [polys[i].as_expr()*z**i for i in range(r + 1)]
            if d == 0 and 0 != Add(*[recurr_coeffs[j]*j for j in range(1, r + 1)]):
                # faster inline check (than calling rsolve_poly) for a
                # constant solution to a constant coefficient recurrence.
                sol = [Symbol("C" + str(len(symbols)))]
            else:
                sol, syms = rsolve_poly(recurr_coeffs, 0, n, len(symbols), symbols=True)
                sol = sol.collect(syms)
                sol = [sol.coeff(s) for s in syms]

            for C in sol:
                ratio = z * A * C.subs(n, n + 1) / B / C
                ratio = simplify(ratio)
                # If there is a nonnegative root in the denominator of the ratio,
                # this indicates that the term y(n_root) is zero, and one should
                # start the product with the term y(n_root + 1).
                n0 = 0
                for n_root in roots(ratio.as_numer_denom()[1], n).keys():
                    if n_root.has(I):
                        return None
                    elif (n0 < (n_root + 1)) == True:
                        n0 = n_root + 1
                K = product(ratio, (n, n0, n - 1))
                if K.has(factorial, FallingFactorial, RisingFactorial):
                    K = simplify(K)

                if casoratian(kernel + [K], n, zero=False) != 0:
                    kernel.append(K)

    kernel.sort(key=default_sort_key)
    sk = list(zip(numbered_symbols('C'), kernel))

    for C, ker in sk:
        result += C * ker

    if hints.get('symbols', False):
        # XXX: This returns the symbols in a non-deterministic order
        symbols |= {s for s, k in sk}
        return (result, list(symbols))
    else:
        return result


def rsolve(f, y, init=None):
    r"""
    Solve univariate recurrence with rational coefficients.

    Given `k`-th order linear recurrence `\operatorname{L} y = f`,
    or equivalently:

    .. math:: a_{k}(n) y(n+k) + a_{k-1}(n) y(n+k-1) +
              \cdots + a_{0}(n) y(n) = f(n)

    where `a_{i}(n)`, for `i=0, \ldots, k`, are polynomials or rational
    functions in `n`, and `f` is a hypergeometric function or a sum
    of a fixed number of pairwise dissimilar hypergeometric terms in
    `n`, finds all solutions or returns ``None``, if none were found.

    Initial conditions can be given as a dictionary in two forms:

        (1) ``{  n_0  : v_0,   n_1  : v_1, ...,   n_m  : v_m}``
        (2) ``{y(n_0) : v_0, y(n_1) : v_1, ..., y(n_m) : v_m}``

    or as a list ``L`` of values:

        ``L = [v_0, v_1, ..., v_m]``

    where ``L[i] = v_i``, for `i=0, \ldots, m`, maps to `y(n_i)`.

    Examples
    ========

    Lets consider the following recurrence:

    .. math:: (n - 1) y(n + 2) - (n^2 + 3 n - 2) y(n + 1) +
              2 n (n + 1) y(n) = 0

    >>> from sympy import Function, rsolve
    >>> from sympy.abc import n
    >>> y = Function('y')

    >>> f = (n - 1)*y(n + 2) - (n**2 + 3*n - 2)*y(n + 1) + 2*n*(n + 1)*y(n)

    >>> rsolve(f, y(n))
    2**n*C0 + C1*factorial(n)

    >>> rsolve(f, y(n), {y(0):0, y(1):3})
    3*2**n - 3*factorial(n)

    See Also
    ========

    rsolve_poly, rsolve_ratio, rsolve_hyper

    """
    if isinstance(f, Equality):
        f = f.lhs - f.rhs

    n = y.args[0]
    k = Wild('k', exclude=(n,))

    # Preprocess user input to allow things like
    # y(n) + a*(y(n + 1) + y(n - 1))/2
    f = f.expand().collect(y.func(Wild('m', integer=True)))

    h_part = defaultdict(list)
    i_part = []
    for g in Add.make_args(f):
        coeff, dep = g.as_coeff_mul(y.func)
        if not dep:
            i_part.append(coeff)
            continue
        for h in dep:
            if h.is_Function and h.func == y.func:
                result = h.args[0].match(n + k)
                if result is not None:
                    h_part[int(result[k])].append(coeff)
                    continue
            raise ValueError(
                "'%s(%s + k)' expected, got '%s'" % (y.func, n, h))
    for k in h_part:
        h_part[k] = Add(*h_part[k])
    h_part.default_factory = lambda: 0
    i_part = Add(*i_part)

    for k, coeff in h_part.items():
        h_part[k] = simplify(coeff)

    common = S.One

    if not i_part.is_zero and not i_part.is_hypergeometric(n) and \
       not (i_part.is_Add and all((x.is_hypergeometric(n) for x in i_part.expand().args))):
        raise ValueError("The independent term should be a sum of hypergeometric functions, got '%s'" % i_part)

    for coeff in h_part.values():
        if coeff.is_rational_function(n):
            if not coeff.is_polynomial(n):
                common = lcm(common, coeff.as_numer_denom()[1], n)
        else:
            raise ValueError(
                "Polynomial or rational function expected, got '%s'" % coeff)

    i_numer, i_denom = i_part.as_numer_denom()

    if i_denom.is_polynomial(n):
        common = lcm(common, i_denom, n)

    if common is not S.One:
        for k, coeff in h_part.items():
            numer, denom = coeff.as_numer_denom()
            h_part[k] = numer*quo(common, denom, n)

        i_part = i_numer*quo(common, i_denom, n)

    K_min = min(h_part.keys())

    if K_min < 0:
        K = abs(K_min)

        H_part = defaultdict(lambda: S.Zero)
        i_part = i_part.subs(n, n + K).expand()
        common = common.subs(n, n + K).expand()

        for k, coeff in h_part.items():
            H_part[k + K] = coeff.subs(n, n + K).expand()
    else:
        H_part = h_part

    K_max = max(H_part.keys())
    coeffs = [H_part[i] for i in range(K_max + 1)]

    result = rsolve_hyper(coeffs, -i_part, n, symbols=True)

    if result is None:
        return None

    solution, symbols = result

    if init in ({}, []):
        init = None

    if symbols and init is not None:
        if isinstance(init, list):
            init = {i: init[i] for i in range(len(init))}

        equations = []

        for k, v in init.items():
            try:
                i = int(k)
            except TypeError:
                if k.is_Function and k.func == y.func:
                    i = int(k.args[0])
                else:
                    raise ValueError("Integer or term expected, got '%s'" % k)

            eq = solution.subs(n, i) - v
            if eq.has(S.NaN):
                eq = solution.limit(n, i) - v
            equations.append(eq)

        result = solve(equations, *symbols)

        if not result:
            return None
        else:
            solution = solution.subs(result)

    return solution
<|endoftext|>"""Utility functions for classifying and solving
ordinary and partial differential equations.

Contains
========
_preprocess
ode_order
_desolve

"""
from sympy.core import Pow
from sympy.core.function import Derivative, AppliedUndef
from sympy.core.relational import Equality
from sympy.core.symbol import Wild

def _preprocess(expr, func=None, hint='_Integral'):
    """Prepare expr for solving by making sure that differentiation
    is done so that only func remains in unevaluated derivatives and
    (if hint does not end with _Integral) that doit is applied to all
    other derivatives. If hint is None, do not do any differentiation.
    (Currently this may cause some simple differential equations to
    fail.)

    In case func is None, an attempt will be made to autodetect the
    function to be solved for.

    >>> from sympy.solvers.deutils import _preprocess
    >>> from sympy import Derivative, Function
    >>> from sympy.abc import x, y, z
    >>> f, g = map(Function, 'fg')

    If f(x)**p == 0 and p>0 then we can solve for f(x)=0
    >>> _preprocess((f(x).diff(x)-4)**5, f(x))
    (Derivative(f(x), x) - 4, f(x))

    Apply doit to derivatives that contain more than the function
    of interest:

    >>> _preprocess(Derivative(f(x) + x, x))
    (Derivative(f(x), x) + 1, f(x))

    Do others if the differentiation variable(s) intersect with those
    of the function of interest or contain the function of interest:

    >>> _preprocess(Derivative(g(x), y, z), f(y))
    (0, f(y))
    >>> _preprocess(Derivative(f(y), z), f(y))
    (0, f(y))

    Do others if the hint does not end in '_Integral' (the default
    assumes that it does):

    >>> _preprocess(Derivative(g(x), y), f(x))
    (Derivative(g(x), y), f(x))
    >>> _preprocess(Derivative(f(x), y), f(x), hint='')
    (0, f(x))

    Do not do any derivatives if hint is None:

    >>> eq = Derivative(f(x) + 1, x) + Derivative(f(x), y)
    >>> _preprocess(eq, f(x), hint=None)
    (Derivative(f(x) + 1, x) + Derivative(f(x), y), f(x))

    If it's not clear what the function of interest is, it must be given:

    >>> eq = Derivative(f(x) + g(x), x)
    >>> _preprocess(eq, g(x))
    (Derivative(f(x), x) + Derivative(g(x), x), g(x))
    >>> try: _preprocess(eq)
    ... except ValueError: print("A ValueError was raised.")
    A ValueError was raised.

    """
    if isinstance(expr, Pow):
        # if f(x)**p=0 then f(x)=0 (p>0)
        if (expr.exp).is_positive:
            expr = expr.base
    derivs = expr.atoms(Derivative)
    if not func:
        funcs = set().union(*[d.atoms(AppliedUndef) for d in derivs])
        if len(funcs) != 1:
            raise ValueError('The function cannot be '
                'automatically detected for %s.' % expr)
        func = funcs.pop()
    fvars = set(func.args)
    if hint is None:
        return expr, func
    reps = [(d, d.doit()) for d in derivs if not hint.endswith('_Integral') or
            d.has(func) or set(d.variables) & fvars]
    eq = expr.subs(reps)
    return eq, func


def ode_order(expr, func):
    """
    Returns the order of a given differential
    equation with respect to func.

    This function is implemented recursively.

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.solvers.deutils import ode_order
    >>> from sympy.abc import x
    >>> f, g = map(Function, ['f', 'g'])
    >>> ode_order(f(x).diff(x, 2) + f(x).diff(x)**2 +
    ... f(x).diff(x), f(x))
    2
    >>> ode_order(f(x).diff(x, 2) + g(x).diff(x, 3), f(x))
    2
    >>> ode_order(f(x).diff(x, 2) + g(x).diff(x, 3), g(x))
    3

    """
    a = Wild('a', exclude=[func])
    if expr.match(a):
        return 0

    if isinstance(expr, Derivative):
        if expr.args[0] == func:
            return len(expr.variables)
        else:
            args = expr.args[0].args
            rv = len(expr.variables)
            if args:
                rv += max(ode_order(_, func) for _ in args)
            return rv
    else:
        return max(ode_order(_, func) for _ in expr.args) if expr.args else 0


def _desolve(eq, func=None, hint="default", ics=None, simplify=True, *, prep=True, **kwargs):
    """This is a helper function to dsolve and pdsolve in the ode
    and pde modules.

    If the hint provided to the function is "default", then a dict with
    the following keys are returned

    'func'    - It provides the function for which the differential equation
                has to be solved. This is useful when the expression has
                more than one function in it.

    'default' - The default key as returned by classifier functions in ode
                and pde.py

    'hint'    - The hint given by the user for which the differential equation
                is to be solved. If the hint given by the user is 'default',
                then the value of 'hint' and 'default' is the same.

    'order'   - The order of the function as returned by ode_order

    'match'   - It returns the match as given by the classifier functions, for
                the default hint.

    If the hint provided to the function is not "default" and is not in
    ('all', 'all_Integral', 'best'), then a dict with the above mentioned keys
    is returned along with the keys which are returned when dict in
    classify_ode or classify_pde is set True

    If the hint given is in ('all', 'all_Integral', 'best'), then this function
    returns a nested dict, with the keys, being the set of classified hints
    returned by classifier functions, and the values being the dict of form
    as mentioned above.

    Key 'eq' is a common key to all the above mentioned hints which returns an
    expression if eq given by user is an Equality.

    See Also
    ========
    classify_ode(ode.py)
    classify_pde(pde.py)
    """
    if isinstance(eq, Equality):
        eq = eq.lhs - eq.rhs

    # preprocess the equation and find func if not given
    if prep or func is None:
        eq, func = _preprocess(eq, func)
        prep = False

    # type is an argument passed by the solve functions in ode and pde.py
    # that identifies whether the function caller is an ordinary
    # or partial differential equation. Accordingly corresponding
    # changes are made in the function.
    type = kwargs.get('type', None)
    xi = kwargs.get('xi')
    eta = kwargs.get('eta')
    x0 = kwargs.get('x0', 0)
    terms = kwargs.get('n')

    if type == 'ode':
        from sympy.solvers.ode import classify_ode, allhints
        classifier = classify_ode
        string = 'ODE '
        dummy = ''

    elif type == 'pde':
        from sympy.solvers.pde import classify_pde, allhints
        classifier = classify_pde
        string = 'PDE '
        dummy = 'p'

    # Magic that should only be used internally.  Prevents classify_ode from
    # being called more than it needs to be by passing its results through
    # recursive calls.
    if kwargs.get('classify', True):
        hints = classifier(eq, func, dict=True, ics=ics, xi=xi, eta=eta,
        n=terms, x0=x0, hint=hint, prep=prep)

    else:
        # Here is what all this means:
        #
        # hint:    The hint method given to _desolve() by the user.
        # hints:   The dictionary of hints that match the DE, along with other
        #          information (including the internal pass-through magic).
        # default: The default hint to return, the first hint from allhints
        #          that matches the hint; obtained from classify_ode().
        # match:   Dictionary containing the match dictionary for each hint
        #          (the parts of the DE for solving).  When going through the
        #          hints in "all", this holds the match string for the current
        #          hint.
        # order:   The order of the DE, as determined by ode_order().
        hints = kwargs.get('hint',
                           {'default': hint,
                            hint: kwargs['match'],
                            'order': kwargs['order']})
    if not hints['default']:
        # classify_ode will set hints['default'] to None if no hints match.
        if hint not in allhints and hint != 'default':
            raise ValueError("Hint not recognized: " + hint)
        elif hint not in hints['ordered_hints'] and hint != 'default':
            raise ValueError(string + str(eq) + " does not match hint " + hint)
        # If dsolve can't solve the purely algebraic equation then dsolve will raise
        # ValueError
        elif hints['order'] == 0:
            raise ValueError(
                str(eq) + " is not a solvable differential equation in " + str(func))
        else:
            raise NotImplementedError(dummy + "solve" + ": Cannot solve " + str(eq))
    if hint == 'default':
        return _desolve(eq, func, ics=ics, hint=hints['default'], simplify=simplify,
                      prep=prep, x0=x0, classify=False, order=hints['order'],
                      match=hints[hints['default']], xi=xi, eta=eta, n=terms, type=type)
    elif hint in ('all', 'all_Integral', 'best'):
        retdict = {}
        gethints = set(hints) - {'order', 'default', 'ordered_hints'}
        if hint == 'all_Integral':
            for i in hints:
                if i.endswith('_Integral'):
                    gethints.remove(i.removesuffix('_Integral'))
            # special cases
            for k in ["1st_homogeneous_coeff_best", "1st_power_series",
                "lie_group", "2nd_power_series_ordinary", "2nd_power_series_regular"]:
                if k in gethints:
                    gethints.remove(k)
        for i in gethints:
            sol = _desolve(eq, func, ics=ics, hint=i, x0=x0, simplify=simplify, prep=prep,
                classify=False, n=terms, order=hints['order'], match=hints[i], type=type)
            retdict[i] = sol
        retdict['all'] = True
        retdict['eq'] = eq
        return retdict
    elif hint not in allhints:  # and hint not in ('default', 'ordered_hints'):
        raise ValueError("Hint not recognized: " + hint)
    elif hint not in hints:
        raise ValueError(string + str(eq) + " does not match hint " + hint)
    else:
        # Key added to identify the hint needed to solve the equation
        hints['hint'] = hint
    hints.update({'func': func, 'eq': eq})
    return hints
<|endoftext|>"""A module for solving all kinds of equations.

    Examples
    ========

    >>> from sympy.solvers import solve
    >>> from sympy.abc import x
    >>> solve(((x + 1)**5).expand(), x)
    [-1]
"""
from sympy.core.assumptions import check_assumptions, failing_assumptions

from .solvers import solve, solve_linear_system, solve_linear_system_LU, \
    solve_undetermined_coeffs, nsolve, solve_linear, checksol, \
    det_quick, inv_quick

from sympy.solvers.diophantine.diophantine import diophantine

from .recurr import rsolve, rsolve_poly, rsolve_ratio, rsolve_hyper

from .ode import checkodesol, classify_ode, dsolve, \
    homogeneous_order

from .polysys import solve_poly_system, solve_triangulated, factor_system

from .pde import pde_separate, pde_separate_add, pde_separate_mul, \
    pdsolve, classify_pde, checkpdesol

from .deutils import ode_order

from .inequalities import reduce_inequalities, reduce_abs_inequality, \
    reduce_abs_inequalities, solve_poly_inequality, solve_rational_inequalities, solve_univariate_inequality

from .decompogen import decompogen

from .solveset import solveset, linsolve, linear_eq_to_matrix, nonlinsolve, substitution

from .simplex import lpmin, lpmax, linprog

# This is here instead of sympy/sets/__init__.py to avoid circular import issues
from ..core.singleton import S
Complexes = S.Complexes

__all__ = [
    'solve', 'solve_linear_system', 'solve_linear_system_LU',
    'solve_undetermined_coeffs', 'nsolve', 'solve_linear', 'checksol',
    'det_quick', 'inv_quick', 'check_assumptions', 'failing_assumptions',

    'diophantine',

    'rsolve', 'rsolve_poly', 'rsolve_ratio', 'rsolve_hyper',

    'checkodesol', 'classify_ode', 'dsolve', 'homogeneous_order',

    'solve_poly_system', 'solve_triangulated', 'factor_system',

    'pde_separate', 'pde_separate_add', 'pde_separate_mul', 'pdsolve',
    'classify_pde', 'checkpdesol',

    'ode_order',

    'reduce_inequalities', 'reduce_abs_inequality', 'reduce_abs_inequalities',
    'solve_poly_inequality', 'solve_rational_inequalities',
    'solve_univariate_inequality',

    'decompogen',

    'solveset', 'linsolve', 'linear_eq_to_matrix', 'nonlinsolve',
    'substitution',

    # This is here instead of sympy/sets/__init__.py to avoid circular import issues
    'Complexes',

    'lpmin', 'lpmax', 'linprog'
]
<|endoftext|>"""
This module contain solvers for all kinds of equations:

    - algebraic or transcendental, use solve()

    - recurrence, use rsolve()

    - differential, use dsolve()

    - nonlinear (numerically), use nsolve()
      (you will need a good starting point)

"""
from __future__ import annotations

from sympy.core import (S, Add, Symbol, Dummy, Expr, Mul)
from sympy.core.assumptions import check_assumptions
from sympy.core.exprtools import factor_terms
from sympy.core.function import (expand_mul, expand_log, Derivative,
                                 AppliedUndef, UndefinedFunction, nfloat,
                                 Function, expand_power_exp, _mexpand, expand,
                                 expand_func)
from sympy.core.logic import fuzzy_not, fuzzy_and
from sympy.core.numbers import Float, Rational, _illegal
from sympy.core.intfunc import integer_log, ilcm
from sympy.core.power import Pow
from sympy.core.relational import Eq, Ne
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.sympify import sympify, _sympify
from sympy.core.traversal import preorder_traversal
from sympy.logic.boolalg import And, BooleanAtom

from sympy.functions import (log, exp, LambertW, cos, sin, tan, acos, asin, atan,
                             Abs, re, im, arg, sqrt, atan2)
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.piecewise import piecewise_fold, Piecewise
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.integrals.integrals import Integral
from sympy.ntheory.factor_ import divisors
from sympy.simplify import (simplify, collect, powsimp, posify,  # type: ignore
    powdenest, nsimplify, denom, logcombine, sqrtdenest, fraction,
    separatevars)
from sympy.simplify.sqrtdenest import sqrt_depth
from sympy.simplify.fu import TR1, TR2i, TR10, TR11
from sympy.strategies.rl import rebuild
from sympy.matrices.exceptions import NonInvertibleMatrixError
from sympy.matrices import Matrix, zeros
from sympy.polys import roots, cancel, factor, Poly
from sympy.polys.solvers import sympy_eqs_to_ring, solve_lin_sys
from sympy.polys.polyerrors import GeneratorsNeeded, PolynomialError
from sympy.polys.polytools import gcd
from sympy.utilities.lambdify import lambdify
from sympy.utilities.misc import filldedent, debugf
from sympy.utilities.iterables import (connected_components,
    generate_bell, uniq, iterable, is_sequence, subsets, flatten, sift)
from sympy.utilities.decorator import conserve_mpmath_dps

from mpmath import findroot

from sympy.solvers.polysys import solve_poly_system

from types import GeneratorType
from collections import defaultdict
from itertools import combinations, product

import warnings


def recast_to_symbols(eqs, symbols):
    """
    Return (e, s, d) where e and s are versions of *eqs* and
    *symbols* in which any non-Symbol objects in *symbols* have
    been replaced with generic Dummy symbols and d is a dictionary
    that can be used to restore the original expressions.

    Examples
    ========

    >>> from sympy.solvers.solvers import recast_to_symbols
    >>> from sympy import symbols, Function
    >>> x, y = symbols('x y')
    >>> fx = Function('f')(x)
    >>> eqs, syms = [fx + 1, x, y], [fx, y]
    >>> e, s, d = recast_to_symbols(eqs, syms); (e, s, d)
    ([_X0 + 1, x, y], [_X0, y], {_X0: f(x)})

    The original equations and symbols can be restored using d:

    >>> assert [i.xreplace(d) for i in eqs] == eqs
    >>> assert [d.get(i, i) for i in s] == syms

    """
    if not iterable(eqs) and iterable(symbols):
        raise ValueError('Both eqs and symbols must be iterable')
    orig = list(symbols)
    symbols = list(ordered(symbols))
    swap_sym = {}
    i = 0
    for s in symbols:
        if not isinstance(s, Symbol) and s not in swap_sym:
            swap_sym[s] = Dummy('X%d' % i)
            i += 1
    new_f = []
    for i in eqs:
        isubs = getattr(i, 'subs', None)
        if isubs is not None:
            new_f.append(isubs(swap_sym))
        else:
            new_f.append(i)
    restore = {v: k for k, v in swap_sym.items()}
    return new_f, [swap_sym.get(i, i) for i in orig], restore


def _ispow(e):
    """Return True if e is a Pow or is exp."""
    return isinstance(e, Expr) and (e.is_Pow or isinstance(e, exp))


def _simple_dens(f, symbols):
    # when checking if a denominator is zero, we can just check the
    # base of powers with nonzero exponents since if the base is zero
    # the power will be zero, too. To keep it simple and fast, we
    # limit simplification to exponents that are Numbers
    dens = set()
    for d in denoms(f, symbols):
        if d.is_Pow and d.exp.is_Number:
            if d.exp.is_zero:
                continue  # foo**0 is never 0
            d = d.base
        dens.add(d)
    return dens


def denoms(eq, *symbols):
    """
    Return (recursively) set of all denominators that appear in *eq*
    that contain any symbol in *symbols*; if *symbols* are not
    provided then all denominators will be returned.

    Examples
    ========

    >>> from sympy.solvers.solvers import denoms
    >>> from sympy.abc import x, y, z

    >>> denoms(x/y)
    {y}

    >>> denoms(x/(y*z))
    {y, z}

    >>> denoms(3/x + y/z)
    {x, z}

    >>> denoms(x/2 + y/z)
    {2, z}

    If *symbols* are provided then only denominators containing
    those symbols will be returned:

    >>> denoms(1/x + 1/y + 1/z, y, z)
    {y, z}

    """

    pot = preorder_traversal(eq)
    dens = set()
    for p in pot:
        # Here p might be Tuple or Relational
        # Expr subtrees (e.g. lhs and rhs) will be traversed after by pot
        if not isinstance(p, Expr):
            continue
        den = denom(p)
        if den is S.One:
            continue
        dens.update(Mul.make_args(den))
    if not symbols:
        return dens
    elif len(symbols) == 1:
        if iterable(symbols[0]):
            symbols = symbols[0]
    return {d for d in dens if any(s in d.free_symbols for s in symbols)}


def checksol(f, symbol, sol=None, **flags):
    """
    Checks whether sol is a solution of equation f == 0.

    Explanation
    ===========

    Input can be either a single symbol and corresponding value
    or a dictionary of symbols and values. When given as a dictionary
    and flag ``simplify=True``, the values in the dictionary will be
    simplified. *f* can be a single equation or an iterable of equations.
    A solution must satisfy all equations in *f* to be considered valid;
    if a solution does not satisfy any equation, False is returned; if one or
    more checks are inconclusive (and none are False) then None is returned.

    Examples
    ========

    >>> from sympy import checksol, symbols
    >>> x, y = symbols('x,y')
    >>> checksol(x**4 - 1, x, 1)
    True
    >>> checksol(x**4 - 1, x, 0)
    False
    >>> checksol(x**2 + y**2 - 5**2, {x: 3, y: 4})
    True

    To check if an expression is zero using ``checksol()``, pass it
    as *f* and send an empty dictionary for *symbol*:

    >>> checksol(x**2 + x - x*(x + 1), {})
    True

    None is returned if ``checksol()`` could not conclude.

    flags:
        'numerical=True (default)'
           do a fast numerical check if ``f`` has only one symbol.
        'minimal=True (default is False)'
           a very fast, minimal testing.
        'warn=True (default is False)'
           show a warning if checksol() could not conclude.
        'simplify=True (default)'
           simplify solution before substituting into function and
           simplify the function before trying specific simplifications
        'force=True (default is False)'
           make positive all symbols without assumptions regarding sign.

    """
    from sympy.physics.units import Unit

    minimal = flags.get('minimal', False)

    if sol is not None:
        sol = {symbol: sol}
    elif isinstance(symbol, dict):
        sol = symbol
    else:
        msg = 'Expecting (sym, val) or ({sym: val}, None) but got (%s, %s)'
        raise ValueError(msg % (symbol, sol))

    if iterable(f):
        if not f:
            raise ValueError('no functions to check')
        return fuzzy_and(checksol(fi, sol, **flags) for fi in f)

    f = _sympify(f)

    if f.is_number:
        return f.is_zero

    if isinstance(f, Poly):
        f = f.as_expr()
    elif isinstance(f, (Eq, Ne)):
        if f.rhs in (S.true, S.false):
            f = f.reversed
        B, E = f.args
        if isinstance(B, BooleanAtom):
            f = f.subs(sol)
            if not f.is_Boolean:
                return
        elif isinstance(f, Eq):
            f = Add(f.lhs, -f.rhs, evaluate=False)

    if isinstance(f, BooleanAtom):
        return bool(f)
    elif not f.is_Relational and not f:
        return True

    illegal = set(_illegal)
    if any(sympify(v).atoms() & illegal for k, v in sol.items()):
        return False

    attempt = -1
    numerical = flags.get('numerical', True)
    while 1:
        attempt += 1
        if attempt == 0:
            val = f.subs(sol)
            if isinstance(val, Mul):
                val = val.as_independent(Unit)[0]
            if val.atoms() & illegal:
                return False
        elif attempt == 1:
            if not val.is_number:
                if not val.is_constant(*list(sol.keys()), simplify=not minimal):
                    return False
                # there are free symbols -- simple expansion might work
                _, val = val.as_content_primitive()
                val = _mexpand(val.as_numer_denom()[0], recursive=True)
        elif attempt == 2:
            if minimal:
                return
            if flags.get('simplify', True):
                for k in sol:
                    sol[k] = simplify(sol[k])
            # start over without the failed expanded form, possibly
            # with a simplified solution
            val = simplify(f.subs(sol))
            if flags.get('force', True):
                val, reps = posify(val)
                # expansion may work now, so try again and check
                exval = _mexpand(val, recursive=True)
                if exval.is_number:
                    # we can decide now
                    val = exval
        else:
            # if there are no radicals and no functions then this can't be
            # zero anymore -- can it?
            pot = preorder_traversal(expand_mul(val))
            seen = set()
            saw_pow_func = False
            for p in pot:
                if p in seen:
                    continue
                seen.add(p)
                if p.is_Pow and not p.exp.is_Integer:
                    saw_pow_func = True
                elif p.is_Function:
                    saw_pow_func = True
                elif isinstance(p, UndefinedFunction):
                    saw_pow_func = True
                if saw_pow_func:
                    break
            if saw_pow_func is False:
                return False
            if flags.get('force', True):
                # don't do a zero check with the positive assumptions in place
                val = val.subs(reps)
            nz = fuzzy_not(val.is_zero)
            if nz is not None:
                # issue 5673: nz may be True even when False
                # so these are just hacks to keep a false positive
                # from being returned

                # HACK 1: LambertW (issue 5673)
                if val.is_number and val.has(LambertW):
                    # don't eval this to verify solution since if we got here,
                    # numerical must be False
                    return None

                # add other HACKs here if necessary, otherwise we assume
                # the nz value is correct
                return not nz
            break
        if val.is_Rational:
            return val == 0
        if numerical and val.is_number:
            return (abs(val.n(18).n(12, chop=True)) < 1e-9) is S.true

    if flags.get('warn', False):
        warnings.warn("\n\tWarning: could not verify solution %s." % sol)
    # returns None if it can't conclude
    # TODO: improve solution testing


def solve(f, *symbols, **flags):
    r"""
    Algebraically solves equations and systems of equations.

    Explanation
    ===========

    Currently supported:
        - polynomial
        - transcendental
        - piecewise combinations of the above
        - systems of linear and polynomial equations
        - systems containing relational expressions
        - systems implied by undetermined coefficients

    Examples
    ========

    The default output varies according to the input and might
    be a list (possibly empty), a dictionary, a list of
    dictionaries or tuples, or an expression involving relationals.
    For specifics regarding different forms of output that may appear, see :ref:`solve_output`.
    Let it suffice here to say that to obtain a uniform output from
    `solve` use ``dict=True`` or ``set=True`` (see below).

        >>> from sympy import solve, Poly, Eq, Matrix, Symbol
        >>> from sympy.abc import x, y, z, a, b

    The expressions that are passed can be Expr, Equality, or Poly
    classes (or lists of the same); a Matrix is considered to be a
    list of all the elements of the matrix:

        >>> solve(x - 3, x)
        [3]
        >>> solve(Eq(x, 3), x)
        [3]
        >>> solve(Poly(x - 3), x)
        [3]
        >>> solve(Matrix([[x, x + y]]), x, y) == solve([x, x + y], x, y)
        True

    If no symbols are indicated to be of interest and the equation is
    univariate, a list of values is returned; otherwise, the keys in
    a dictionary will indicate which (of all the variables used in
    the expression(s)) variables and solutions were found:

        >>> solve(x**2 - 4)
        [-2, 2]
        >>> solve((x - a)*(y - b))
        [{a: x}, {b: y}]
        >>> solve([x - 3, y - 1])
        {x: 3, y: 1}
        >>> solve([x - 3, y**2 - 1])
        [{x: 3, y: -1}, {x: 3, y: 1}]

    If you pass symbols for which solutions are sought, the output will vary
    depending on the number of symbols you passed, whether you are passing
    a list of expressions or not, and whether a linear system was solved.
    Uniform output is attained by using ``dict=True`` or ``set=True``.

        >>> #### *** feel free to skip to the stars below *** ####
        >>> from sympy import TableForm
        >>> h = [None, ';|;'.join(['e', 's', 'solve(e, s)', 'solve(e, s, dict=True)',
        ... 'solve(e, s, set=True)']).split(';')]
        >>> t = []
        >>> for e, s in [
        ...         (x - y, y),
        ...         (x - y, [x, y]),
        ...         (x**2 - y, [x, y]),
        ...         ([x - 3, y -1], [x, y]),
        ...         ]:
        ...     how = [{}, dict(dict=True), dict(set=True)]
        ...     res = [solve(e, s, **f) for f in how]
        ...     t.append([e, '|', s, '|'] + [res[0], '|', res[1], '|', res[2]])
        ...
        >>> # ******************************************************* #
        >>> TableForm(t, headings=h, alignments="<")
        e              | s      | solve(e, s)  | solve(e, s, dict=True) | solve(e, s, set=True)
        ---------------------------------------------------------------------------------------
        x - y          | y      | [x]          | [{y: x}]               | ([y], {(x,)})
        x - y          | [x, y] | [(y, y)]     | [{x: y}]               | ([x, y], {(y, y)})
        x**2 - y       | [x, y] | [(x, x**2)]  | [{y: x**2}]            | ([x, y], {(x, x**2)})
        [x - 3, y - 1] | [x, y] | {x: 3, y: 1} | [{x: 3, y: 1}]         | ([x, y], {(3, 1)})

        * If any equation does not depend on the symbol(s) given, it will be
          eliminated from the equation set and an answer may be given
          implicitly in terms of variables that were not of interest:

            >>> solve([x - y, y - 3], x)
            {x: y}

    When you pass all but one of the free symbols, an attempt
    is made to find a single solution based on the method of
    undetermined coefficients. If it succeeds, a dictionary of values
    is returned. If you want an algebraic solutions for one
    or more of the symbols, pass the expression to be solved in a list:

        >>> e = a*x + b - 2*x - 3
        >>> solve(e, [a, b])
        {a: 2, b: 3}
        >>> solve([e], [a, b])
        {a: -b/x + (2*x + 3)/x}

    When there is no solution for any given symbol which will make all
    expressions zero, the empty list is returned (or an empty set in
    the tuple when ``set=True``):

        >>> from sympy import sqrt
        >>> solve(3, x)
        []
        >>> solve(x - 3, y)
        []
        >>> solve(sqrt(x) + 1, x, set=True)
        ([x], set())

    When an object other than a Symbol is given as a symbol, it is
    isolated algebraically and an implicit solution may be obtained.
    This is mostly provided as a convenience to save you from replacing
    the object with a Symbol and solving for that Symbol. It will only
    work if the specified object can be replaced with a Symbol using the
    subs method:

        >>> from sympy import exp, Function
        >>> f = Function('f')

        >>> solve(f(x) - x, f(x))
        [x]
        >>> solve(f(x).diff(x) - f(x) - x, f(x).diff(x))
        [x + f(x)]
        >>> solve(f(x).diff(x) - f(x) - x, f(x))
        [-x + Derivative(f(x), x)]
        >>> solve(x + exp(x)**2, exp(x), set=True)
        ([exp(x)], {(-sqrt(-x),), (sqrt(-x),)})

        >>> from sympy import Indexed, IndexedBase, Tuple
        >>> A = IndexedBase('A')
        >>> eqs = Tuple(A[1] + A[2] - 3, A[1] - A[2] + 1)
        >>> solve(eqs, eqs.atoms(Indexed))
        {A[1]: 1, A[2]: 2}

        * To solve for a function within a derivative, use :func:`~.dsolve`.

    To solve for a symbol implicitly, use implicit=True:

        >>> solve(x + exp(x), x)
        [-LambertW(1)]
        >>> solve(x + exp(x), x, implicit=True)
        [-exp(x)]

    It is possible to solve for anything in an expression that can be
    replaced with a symbol using :obj:`~sympy.core.basic.Basic.subs`:

        >>> solve(x + 2 + sqrt(3), x + 2)
        [-sqrt(3)]
        >>> solve((x + 2 + sqrt(3), x + 4 + y), y, x + 2)
        {y: -2 + sqrt(3), x + 2: -sqrt(3)}

        * Nothing heroic is done in this implicit solving so you may end up
          with a symbol still in the solution:

            >>> eqs = (x*y + 3*y + sqrt(3), x + 4 + y)
            >>> solve(eqs, y, x + 2)
            {y: -sqrt(3)/(x + 3), x + 2: -2*x/(x + 3) - 6/(x + 3) + sqrt(3)/(x + 3)}
            >>> solve(eqs, y*x, x)
            {x: -y - 4, x*y: -3*y - sqrt(3)}

        * If you attempt to solve for a number, remember that the number
          you have obtained does not necessarily mean that the value is
          equivalent to the expression obtained:

            >>> solve(sqrt(2) - 1, 1)
            [sqrt(2)]
            >>> solve(x - y + 1, 1)  # /!\ -1 is targeted, too
            [x/(y - 1)]
            >>> [_.subs(z, -1) for _ in solve((x - y + 1).subs(-1, z), 1)]
            [-x + y]

    **Additional Examples**

    ``solve()`` with check=True (default) will run through the symbol tags to
    eliminate unwanted solutions. If no assumptions are included, all possible
    solutions will be returned:

        >>> x = Symbol("x")
        >>> solve(x**2 - 1)
        [-1, 1]

    By setting the ``positive`` flag, only one solution will be returned:

        >>> pos = Symbol("pos", positive=True)
        >>> solve(pos**2 - 1)
        [1]

    When the solutions are checked, those that make any denominator zero
    are automatically excluded. If you do not want to exclude such solutions,
    then use the check=False option:

        >>> from sympy import sin, limit
        >>> solve(sin(x)/x)  # 0 is excluded
        [pi]

    If ``check=False``, then a solution to the numerator being zero is found
    but the value of $x = 0$ is a spurious solution since $\sin(x)/x$ has the well
    known limit (without discontinuity) of 1 at $x = 0$:

        >>> solve(sin(x)/x, check=False)
        [0, pi]

    In the following case, however, the limit exists and is equal to the
    value of $x = 0$ that is excluded when check=True:

        >>> eq = x**2*(1/x - z**2/x)
        >>> solve(eq, x)
        []
        >>> solve(eq, x, check=False)
        [0]
        >>> limit(eq, x, 0, '-')
        0
        >>> limit(eq, x, 0, '+')
        0

    **Solving Relationships**

    When one or more expressions passed to ``solve`` is a relational,
    a relational result is returned (and the ``dict`` and ``set`` flags
    are ignored):

        >>> solve(x < 3)
        (-oo < x) & (x < 3)
        >>> solve([x < 3, x**2 > 4], x)
        ((-oo < x) & (x < -2)) | ((2 < x) & (x < 3))
        >>> solve([x + y - 3, x > 3], x)
        (3 < x) & (x < oo) & Eq(x, 3 - y)

    Although checking of assumptions on symbols in relationals
    is not done, setting assumptions will affect how certain
    relationals might automatically simplify:

        >>> solve(x**2 > 4)
        ((-oo < x) & (x < -2)) | ((2 < x) & (x < oo))

        >>> r = Symbol('r', real=True)
        >>> solve(r**2 > 4)
        (2 < r) | (r < -2)

    There is currently no algorithm in SymPy that allows you to use
    relationships to resolve more than one variable. So the following
    does not determine that ``q < 0`` (and trying to solve for ``r``
    and ``q`` will raise an error):

        >>> from sympy import symbols
        >>> r, q = symbols('r, q', real=True)
        >>> solve([r + q - 3, r > 3], r)
        (3 < r) & Eq(r, 3 - q)

    You can directly call the routine that ``solve`` calls
    when it encounters a relational: :func:`~.reduce_inequalities`.
    It treats Expr like Equality.

        >>> from sympy import reduce_inequalities
        >>> reduce_inequalities([x**2 - 4])
        Eq(x, -2) | Eq(x, 2)

    If each relationship contains only one symbol of interest,
    the expressions can be processed for multiple symbols:

        >>> reduce_inequalities([0 <= x  - 1, y < 3], [x, y])
        (-oo < y) & (1 <= x) & (x < oo) & (y < 3)

    But an error is raised if any relationship has more than one
    symbol of interest:

        >>> reduce_inequalities([0 <= x*y  - 1, y < 3], [x, y])
        Traceback (most recent call last):
        ...
        NotImplementedError:
        inequality has more than one symbol of interest.

    **Disabling High-Order Explicit Solutions**

    When solving polynomial expressions, you might not want explicit solutions
    (which can be quite long). If the expression is univariate, ``CRootOf``
    instances will be returned instead:

        >>> solve(x**3 - x + 1)
        [-1/((-1/2 - sqrt(3)*I/2)*(3*sqrt(69)/2 + 27/2)**(1/3)) -
        (-1/2 - sqrt(3)*I/2)*(3*sqrt(69)/2 + 27/2)**(1/3)/3,
        -(-1/2 + sqrt(3)*I/2)*(3*sqrt(69)/2 + 27/2)**(1/3)/3 -
        1/((-1/2 + sqrt(3)*I/2)*(3*sqrt(69)/2 + 27/2)**(1/3)),
        -(3*sqrt(69)/2 + 27/2)**(1/3)/3 -
        1/(3*sqrt(69)/2 + 27/2)**(1/3)]
        >>> solve(x**3 - x + 1, cubics=False)
        [CRootOf(x**3 - x + 1, 0),
         CRootOf(x**3 - x + 1, 1),
         CRootOf(x**3 - x + 1, 2)]

    If the expression is multivariate, no solution might be returned:

        >>> solve(x**3 - x + a, x, cubics=False)
        []

    Sometimes solutions will be obtained even when a flag is False because the
    expression could be factored. In the following example, the equation can
    be factored as the product of a linear and a quadratic factor so explicit
    solutions (which did not require solving a cubic expression) are obtained:

        >>> eq = x**3 + 3*x**2 + x - 1
        >>> solve(eq, cubics=False)
        [-1, -1 + sqrt(2), -sqrt(2) - 1]

    **Solving Equations Involving Radicals**

    Because of SymPy's use of the principle root, some solutions
    to radical equations will be missed unless check=False:

        >>> from sympy import root
        >>> eq = root(x**3 - 3*x**2, 3) + 1 - x
        >>> solve(eq)
        []
        >>> solve(eq, check=False)
        [1/3]

    In the above example, there is only a single solution to the
    equation. Other expressions will yield spurious roots which
    must be checked manually; roots which give a negative argument
    to odd-powered radicals will also need special checking:

        >>> from sympy import real_root, S
        >>> eq = root(x, 3) - root(x, 5) + S(1)/7
        >>> solve(eq)  # this gives 2 solutions but misses a 3rd
        [CRootOf(7*x**5 - 7*x**3 + 1, 1)**15,
        CRootOf(7*x**5 - 7*x**3 + 1, 2)**15]
        >>> sol = solve(eq, check=False)
        >>> [abs(eq.subs(x,i).n(2)) for i in sol]
        [0.48, 0.e-110, 0.e-110, 0.052, 0.052]

    The first solution is negative so ``real_root`` must be used to see that it
    satisfies the expression:

        >>> abs(real_root(eq.subs(x, sol[0])).n(2))
        0.e-110

    If the roots of the equation are not real then more care will be
    necessary to find the roots, especially for higher order equations.
    Consider the following expression:

        >>> expr = root(x, 3) - root(x, 5)

    We will construct a known value for this expression at x = 3 by selecting
    the 1-th root for each radical:

        >>> expr1 = root(x, 3, 1) - root(x, 5, 1)
        >>> v = expr1.subs(x, -3)

    The ``solve`` function is unable to find any exact roots to this equation:

        >>> eq = Eq(expr, v); eq1 = Eq(expr1, v)
        >>> solve(eq, check=False), solve(eq1, check=False)
        ([], [])

    The function ``unrad``, however, can be used to get a form of the equation
    for which numerical roots can be found:

        >>> from sympy.solvers.solvers import unrad
        >>> from sympy import nroots
        >>> e, (p, cov) = unrad(eq)
        >>> pvals = nroots(e)
        >>> inversion = solve(cov, x)[0]
        >>> xvals = [inversion.subs(p, i) for i in pvals]

    Although ``eq`` or ``eq1`` could have been used to find ``xvals``, the
    solution can only be verified with ``expr1``:

        >>> z = expr - v
        >>> [xi.n(chop=1e-9) for xi in xvals if abs(z.subs(x, xi).n()) < 1e-9]
        []
        >>> z1 = expr1 - v
        >>> [xi.n(chop=1e-9) for xi in xvals if abs(z1.subs(x, xi).n()) < 1e-9]
        [-3.0]

    Parameters
    ==========

    f :
        - a single Expr or Poly that must be zero
        - an Equality
        - a Relational expression
        - a Boolean
        - iterable of one or more of the above

    symbols : (object(s) to solve for) specified as
        - none given (other non-numeric objects will be used)
        - single symbol
        - denested list of symbols
          (e.g., ``solve(f, x, y)``)
        - ordered iterable of symbols
          (e.g., ``solve(f, [x, y])``)

    flags :
        dict=True (default is False)
            Return list (perhaps empty) of solution mappings.
        set=True (default is False)
            Return list of symbols and set of tuple(s) of solution(s).
        exclude=[] (default)
            Do not try to solve for any of the free symbols in exclude;
            if expressions are given, the free symbols in them will
            be extracted automatically.
        check=True (default)
            If False, do not do any testing of solutions. This can be
            useful if you want to include solutions that make any
            denominator zero.
        numerical=True (default)
            Do a fast numerical check if *f* has only one symbol.
        minimal=True (default is False)
            A very fast, minimal testing.
        warn=True (default is False)
            Show a warning if ``checksol()`` could not conclude.
        simplify=True (default)
            Simplify all but polynomials of order 3 or greater before
            returning them and (if check is not False) use the
            general simplify function on the solutions and the
            expression obtained when they are substituted into the
            function which should be zero.
        force=True (default is False)
            Make positive all symbols without assumptions regarding sign.
        rational=True (default)
            Recast Floats as Rational; if this option is not used, the
            system containing Floats may fail to solve because of issues
            with polys. If rational=None, Floats will be recast as
            rationals but the answer will be recast as Floats. If the
            flag is False then nothing will be done to the Floats.
        manual=True (default is False)
            Do not use the polys/matrix method to solve a system of
            equations, solve them one at a time as you might "manually."
        implicit=True (default is False)
            Allows ``solve`` to return a solution for a pattern in terms of
            other functions that contain that pattern; this is only
            needed if the pattern is inside of some invertible function
            like cos, exp, etc.
        particular=True (default is False)
            Instructs ``solve`` to try to find a particular solution to
            a linear system with as many zeros as possible; this is very
            expensive.
        quick=True (default is False; ``particular`` must be True)
            Selects a fast heuristic to find a solution with many zeros
            whereas a value of False uses the very slow method guaranteed
            to find the largest number of zeros possible.
        cubics=True (default)
            Return explicit solutions when cubic expressions are encountered.
            When False, quartics and quintics are disabled, too.
        quartics=True (default)
            Return explicit solutions when quartic expressions are encountered.
            When False, quintics are disabled, too.
        quintics=True (default)
            Return explicit solutions (if possible) when quintic expressions
            are encountered.

    See Also
    ========

    rsolve: For solving recurrence relationships
    sympy.solvers.ode.dsolve: For solving differential equations

    """
    from .inequalities import reduce_inequalities

    # checking/recording flags
    ###########################################################################

    # set solver types explicitly; as soon as one is False
    # all the rest will be False
    hints = ('cubics', 'quartics', 'quintics')
    default = True
    for k in hints:
        default = flags.setdefault(k, bool(flags.get(k, default)))

    # allow solution to contain symbol if True:
    implicit = flags.get('implicit', False)

    # record desire to see warnings
    warn = flags.get('warn', False)

    # this flag will be needed for quick exits below, so record
    # now -- but don't record `dict` yet since it might change
    as_set = flags.get('set', False)

    # keeping track of how f was passed
    bare_f = not iterable(f)

    # check flag usage for particular/quick which should only be used
    # with systems of equations
    if flags.get('quick', None) is not None:
        if not flags.get('particular', None):
            raise ValueError('when using `quick`, `particular` should be True')
    if flags.get('particular', False) and bare_f:
        raise ValueError(filldedent("""
            The 'particular/quick' flag is usually used with systems of
            equations. Either pass your equation in a list or
            consider using a solver like `diophantine` if you are
            looking for a solution in integers."""))

    # sympify everything, creating list of expressions and list of symbols
    ###########################################################################

    def _sympified_list(w):
        return list(map(sympify, w if iterable(w) else [w]))
    f, symbols = (_sympified_list(w) for w in [f, symbols])

    # preprocess symbol(s)
    ###########################################################################

    ordered_symbols = None  # were the symbols in a well defined order?
    if not symbols:
        # get symbols from equations
        symbols = set().union(*[fi.free_symbols for fi in f])
        if len(symbols) < len(f):
            for fi in f:
                pot = preorder_traversal(fi)
                for p in pot:
                    if isinstance(p, AppliedUndef):
                        if not as_set:
                            flags['dict'] = True  # better show symbols
                        symbols.add(p)
                        pot.skip()  # don't go any deeper
        ordered_symbols = False
        symbols = list(ordered(symbols))  # to make it canonical
    else:
        if len(symbols) == 1 and iterable(symbols[0]):
            symbols = symbols[0]
        ordered_symbols = symbols and is_sequence(symbols,
                        include=GeneratorType)
        _symbols = list(uniq(symbols))
        if len(_symbols) != len(symbols):
            ordered_symbols = False
            symbols = list(ordered(symbols))
        else:
            symbols = _symbols

    # check for duplicates
    if len(symbols) != len(set(symbols)):
        raise ValueError('duplicate symbols given')
    # remove those not of interest
    exclude = flags.pop('exclude', set())
    if exclude:
        if isinstance(exclude, Expr):
            exclude = [exclude]
        exclude = set().union(*[e.free_symbols for e in sympify(exclude)])
        symbols = [s for s in symbols if s not in exclude]

    # preprocess equation(s)
    ###########################################################################

    # automatically ignore True values
    if isinstance(f, list):
        f = [s for s in f if s is not S.true]

    # handle canonicalization of equation types
    for i, fi in enumerate(f):
        if isinstance(fi, (Eq, Ne)):
            if 'ImmutableDenseMatrix' in [type(a).__name__ for a in fi.args]:
                fi = fi.lhs - fi.rhs
            else:
                L, R = fi.args
                if isinstance(R, BooleanAtom):
                    L, R = R, L
                if isinstance(L, BooleanAtom):
                    if isinstance(fi, Ne):
                        L = ~L
                    if R.is_Relational:
                        fi = ~R if L is S.false else R
                    elif R.is_Symbol:
                        return L
                    elif R.is_Boolean and (~R).is_Symbol:
                        return ~L
                    else:
                        raise NotImplementedError(filldedent('''
                            Unanticipated argument of Eq when other arg
                            is True or False.
                        '''))
                elif isinstance(fi, Eq):
                    fi = Add(fi.lhs, -fi.rhs, evaluate=False)
            f[i] = fi

        # *** dispatch and handle as a system of relationals
        # **************************************************
        if fi.is_Relational:
            if len(symbols) != 1:
                raise ValueError("can only solve for one symbol at a time")
            if warn and symbols[0].assumptions0:
                warnings.warn(filldedent("""
                    \tWarning: assumptions about variable '%s' are
                    not handled currently.""" % symbols[0]))
            return reduce_inequalities(f, symbols=symbols)

        # convert Poly to expression
        if isinstance(fi, Poly):
            f[i] = fi.as_expr()

        # rewrite hyperbolics in terms of exp if they have symbols of
        # interest
        f[i] = f[i].replace(lambda w: isinstance(w, HyperbolicFunction) and \
            w.has_free(*symbols), lambda w: w.rewrite(exp))

        # if we have a Matrix, we need to iterate over its elements again
        if f[i].is_Matrix:
            try:
                f[i] = f[i].as_explicit()
            except ValueError:
                raise ValueError(
                    "solve cannot handle matrices with symbolic shape."
                )
            bare_f = False
            f.extend(list(f[i]))
            f[i] = S.Zero

        # if we can split it into real and imaginary parts then do so
        freei = f[i].free_symbols
        if freei and all(s.is_extended_real or s.is_imaginary for s in freei):
            fr, fi = f[i].as_real_imag()
            # accept as long as new re, im, arg or atan2 are not introduced
            had = f[i].atoms(re, im, arg, atan2)
            if fr and fi and fr != fi and not any(
                    i.atoms(re, im, arg, atan2) - had for i in (fr, fi)):
                if bare_f:
                    bare_f = False
                f[i: i + 1] = [fr, fi]

    # real/imag handling -----------------------------
    if any(isinstance(fi, (bool, BooleanAtom)) for fi in f):
        if as_set:
            return [], set()
        return []

    for i, fi in enumerate(f):
        # Abs
        while True:
            was = fi
            fi = fi.replace(Abs, lambda arg:
                separatevars(Abs(arg)).rewrite(Piecewise) if arg.has(*symbols)
                else Abs(arg))
            if was == fi:
                break

        for e in fi.find(Abs):
            if e.has(*symbols):
                raise NotImplementedError('solving %s when the argument '
                    'is not real or imaginary.' % e)

        # arg
        fi = fi.replace(arg, lambda a: arg(a).rewrite(atan2).rewrite(atan))

        # save changes
        f[i] = fi

    # see if re(s) or im(s) appear
    freim = [fi for fi in f if fi.has(re, im)]
    if freim:
        irf = []
        for s in symbols:
            if s.is_real or s.is_imaginary:
                continue  # neither re(x) nor im(x) will appear
            # if re(s) or im(s) appear, the auxiliary equation must be present
            if any(fi.has(re(s), im(s)) for fi in freim):
                irf.append((s, re(s) + S.ImaginaryUnit*im(s)))
        if irf:
            for s, rhs in irf:
                f = [fi.xreplace({s: rhs}) for fi in f] + [s - rhs]
                symbols.extend([re(s), im(s)])
            if bare_f:
                bare_f = False
            flags['dict'] = True
    # end of real/imag handling  -----------------------------

    # we can solve for non-symbol entities by replacing them with Dummy symbols
    f, symbols, swap_sym = recast_to_symbols(f, symbols)
    # this set of symbols (perhaps recast) is needed below
    symset = set(symbols)

    # get rid of equations that have no symbols of interest; we don't
    # try to solve them because the user didn't ask and they might be
    # hard to solve; this means that solutions may be given in terms
    # of the eliminated equations e.g. solve((x-y, y-3), x) -> {x: y}
    newf = []
    for fi in f:
        # let the solver handle equations that..
        # - have no symbols but are expressions
        # - have symbols of interest
        # - have no symbols of interest but are constant
        # but when an expression is not constant and has no symbols of
        # interest, it can't change what we obtain for a solution from
        # the remaining equations so we don't include it; and if it's
        # zero it can be removed and if it's not zero, there is no
        # solution for the equation set as a whole
        #
        # The reason for doing this filtering is to allow an answer
        # to be obtained to queries like solve((x - y, y), x); without
        # this mod the return value is []
        ok = False
        if fi.free_symbols & symset:
            ok = True
        else:
            if fi.is_number:
                if fi.is_Number:
                    if fi.is_zero:
                        continue
                    return []
                ok = True
            else:
                if fi.is_constant():
                    ok = True
        if ok:
            newf.append(fi)
    if not newf:
        if as_set:
            return symbols, set()
        return []
    f = newf
    del newf

    # mask off any Object that we aren't going to invert: Derivative,
    # Integral, etc... so that solving for anything that they contain will
    # give an implicit solution
    seen = set()
    non_inverts = set()
    for fi in f:
        pot = preorder_traversal(fi)
        for p in pot:
            if not isinstance(p, Expr) or isinstance(p, Piecewise):
                pass
            elif (isinstance(p, bool) or
                    not p.args or
                    p in symset or
                    p.is_Add or p.is_Mul or
                    p.is_Pow and not implicit or
                    p.is_Function and not implicit) and p.func not in (re, im):
                continue
            elif p not in seen:
                seen.add(p)
                if p.free_symbols & symset:
                    non_inverts.add(p)
                else:
                    continue
            pot.skip()
    del seen
    non_inverts = dict(list(zip(non_inverts, [Dummy() for _ in non_inverts])))
    f = [fi.subs(non_inverts) for fi in f]

    # Both xreplace and subs are needed below: xreplace to force substitution
    # inside Derivative, subs to handle non-straightforward substitutions
    non_inverts = [(v, k.xreplace(swap_sym).subs(swap_sym)) for k, v in non_inverts.items()]

    # rationalize Floats
    floats = False
    if flags.get('rational', True) is not False:
        for i, fi in enumerate(f):
            if fi.has(Float):
                floats = True
                f[i] = nsimplify(fi, rational=True)

    # capture any denominators before rewriting since
    # they may disappear after the rewrite, e.g. issue 14779
    flags['_denominators'] = _simple_dens(f[0], symbols)

    # Any embedded piecewise functions need to be brought out to the
    # top level so that the appropriate strategy gets selected.
    # However, this is necessary only if one of the piecewise
    # functions depends on one of the symbols we are solving for.
    def _has_piecewise(e):
        if e.is_Piecewise:
            return e.has(*symbols)
        return any(_has_piecewise(a) for a in e.args)
    for i, fi in enumerate(f):
        if _has_piecewise(fi):
            f[i] = piecewise_fold(fi)

    # expand angles of sums; in general, expand_trig will allow
    # more roots to be found but this is not a great solultion
    # to not returning a parametric solution, otherwise
    # many values can be returned that have a simple
    # relationship between values
    targs = {t for fi in f for t in fi.atoms(TrigonometricFunction)}
    if len(targs) > 1:
        add, other = sift(targs, lambda x: x.args[0].is_Add, binary=True)
        add, other = [[i for i in l if i.has_free(*symbols)] for l in (add, other)]
        trep = {}
        for t in add:
            a = t.args[0]
            ind, dep = a.as_independent(*symbols)
            if dep in symbols or -dep in symbols:
                # don't let expansion expand wrt anything in ind
                n = Dummy() if not ind.is_Number else ind
                trep[t] = TR10(t.func(dep + n)).xreplace({n: ind})
        if other and len(other) <= 2:
            base = gcd(*[i.args[0] for i in other]) if len(other) > 1 else other[0].args[0]
            for i in other:
                trep[i] = TR11(i, base)
        f = [fi.xreplace(trep) for fi in f]

    #
    # try to get a solution
    ###########################################################################
    if bare_f:
        solution = None
        if len(symbols) != 1:
            solution = _solve_undetermined(f[0], symbols, flags)
        if not solution:
            solution = _solve(f[0], *symbols, **flags)
    else:
        linear, solution = _solve_system(f, symbols, **flags)
    assert type(solution) is list
    assert not solution or type(solution[0]) is dict, solution
    #
    # postprocessing
    ###########################################################################
    # capture as_dict flag now (as_set already captured)
    as_dict = flags.get('dict', False)

    # define how solution will get unpacked
    tuple_format = lambda s: [tuple([i.get(x, x) for x in symbols]) for i in s]
    if as_dict or as_set:
        unpack = None
    elif bare_f:
        if len(symbols) == 1:
            unpack = lambda s: [i[symbols[0]] for i in s]
        elif len(solution) == 1 and len(solution[0]) == len(symbols):
            # undetermined linear coeffs solution
            unpack = lambda s: s[0]
        elif ordered_symbols:
            unpack = tuple_format
        else:
            unpack = lambda s: s
    else:
        if solution:
            if linear and len(solution) == 1:
                # if you want the tuple solution for the linear
                # case, use `set=True`
                unpack = lambda s: s[0]
            elif ordered_symbols:
                unpack = tuple_format
            else:
                unpack = lambda s: s
        else:
            unpack = None

    # Restore masked-off objects
    if non_inverts and type(solution) is list:
        solution = [{k: v.subs(non_inverts) for k, v in s.items()}
            for s in solution]

    # Restore original "symbols" if a dictionary is returned.
    # This is not necessary for
    #   - the single univariate equation case
    #     since the symbol will have been removed from the solution;
    #   - the nonlinear poly_system since that only supports zero-dimensional
    #     systems and those results come back as a list
    #
    # ** unless there were Derivatives with the symbols, but those were handled
    #    above.
    if swap_sym:
        symbols = [swap_sym.get(k, k) for k in symbols]
        for i, sol in enumerate(solution):
            solution[i] = {swap_sym.get(k, k): v.subs(swap_sym)
                      for k, v in sol.items()}

    # Get assumptions about symbols, to filter solutions.
    # Note that if assumptions about a solution can't be verified, it is still
    # returned.
    check = flags.get('check', True)

    # restore floats
    if floats and solution and flags.get('rational', None) is None:
        solution = nfloat(solution, exponent=False)
        # nfloat might reveal more duplicates
        solution = _remove_duplicate_solutions(solution)

    if check and solution:  # assumption checking
        warn = flags.get('warn', False)
        got_None = []  # solutions for which one or more symbols gave None
        no_False = []  # solutions for which no symbols gave False
        for sol in solution:
            v = fuzzy_and(check_assumptions(val, **symb.assumptions0)
                          for symb, val in sol.items())
            if v is False:
                continue
            no_False.append(sol)
            if v is None:
                got_None.append(sol)

        solution = no_False
        if warn and got_None:
            warnings.warn(filldedent("""
                \tWarning: assumptions concerning following solution(s)
                cannot be checked:""" + '\n\t' +
                ', '.join(str(s) for s in got_None)))

    #
    # done
    ###########################################################################

    if not solution:
        if as_set:
            return symbols, set()
        return []

    # make orderings canonical for list of dictionaries
    if not as_set:  # for set, no point in ordering
        solution = [{k: s[k] for k in ordered(s)} for s in solution]
        solution.sort(key=default_sort_key)

    if not (as_set or as_dict):
        return unpack(solution)

    if as_dict:
        return solution

    # set output: (symbols, {t1, t2, ...}) from list of dictionaries;
    # include all symbols for those that like a verbose solution
    # and to resolve any differences in dictionary keys.
    #
    # The set results can easily be used to make a verbose dict as
    #   k, v = solve(eqs, syms, set=True)
    #   sol = [dict(zip(k,i)) for i in v]
    #
    if ordered_symbols:
        k = symbols  # keep preferred order
    else:
        # just unify the symbols for which solutions were found
        k = list(ordered(set(flatten(tuple(i.keys()) for i in solution))))
    return k, {tuple([s.get(ki, ki) for ki in k]) for s in solution}


def _solve_undetermined(g, symbols, flags):
    """solve helper to return a list with one dict (solution) else None

    A direct call to solve_undetermined_coeffs is more flexible and
    can return both multiple solutions and handle more than one independent
    variable. Here, we have to be more cautious to keep from solving
    something that does not look like an undetermined coeffs system --
    to minimize the surprise factor since singularities that cancel are not
    prohibited in solve_undetermined_coeffs.
    """
    if g.free_symbols - set(symbols):
        sol = solve_undetermined_coeffs(g, symbols, **dict(flags, dict=True, set=None))
        if len(sol) == 1:
            return sol


def _solve(f, *symbols, **flags):
    """Return a checked solution for *f* in terms of one or more of the
    symbols in the form of a list of dictionaries.

    If no method is implemented to solve the equation, a NotImplementedError
    will be raised. In the case that conversion of an expression to a Poly
    gives None a ValueError will be raised.
    """

    not_impl_msg = "No algorithms are implemented to solve equation %s"

    if len(symbols) != 1:
        # look for solutions for desired symbols that are independent
        # of symbols already solved for, e.g. if we solve for x = y
        # then no symbol having x in its solution will be returned.

        # First solve for linear symbols (since that is easier and limits
        # solution size) and then proceed with symbols appearing
        # in a non-linear fashion. Ideally, if one is solving a single
        # expression for several symbols, they would have to be
        # appear in factors of an expression, but we do not here
        # attempt factorization.  XXX perhaps handling a Mul
        # should come first in this routine whether there is
        # one or several symbols.
        nonlin_s = []
        got_s = set()
        rhs_s = set()
        result = []
        for s in symbols:
            xi, v = solve_linear(f, symbols=[s])
            if xi == s:
                # no need to check but we should simplify if desired
                if flags.get('simplify', True):
                    v = simplify(v)
                vfree = v.free_symbols
                if vfree & got_s:
                    # was linear, but has redundant relationship
                    # e.g. x - y = 0 has y == x is redundant for x == y
                    # so ignore
                    continue
                rhs_s |= vfree
                got_s.add(xi)
                result.append({xi: v})
            elif xi:  # there might be a non-linear solution if xi is not 0
                nonlin_s.append(s)
        if not nonlin_s:
            return result
        for s in nonlin_s:
            try:
                soln = _solve(f, s, **flags)
                for sol in soln:
                    if sol[s].free_symbols & got_s:
                        # depends on previously solved symbols: ignore
                        continue
                    got_s.add(s)
                    result.append(sol)
            except NotImplementedError:
                continue
        if got_s:
            return result
        else:
            raise NotImplementedError(not_impl_msg % f)

    # solve f for a single variable

    symbol = symbols[0]

    # expand binomials only if it has the unknown symbol
    f = f.replace(lambda e: isinstance(e, binomial) and e.has(symbol),
        lambda e: expand_func(e))

    # checking will be done unless it is turned off before making a
    # recursive call; the variables `checkdens` and `check` are
    # captured here (for reference below) in case flag value changes
    flags['check'] = checkdens = check = flags.pop('check', True)

    # build up solutions if f is a Mul
    if f.is_Mul:
        result = set()
        for m in f.args:
            if m in {S.NegativeInfinity, S.ComplexInfinity, S.Infinity}:
                result = set()
                break
            soln = _vsolve(m, symbol, **flags)
            result.update(set(soln))
        result = [{symbol: v} for v in result]
        if check:
            # all solutions have been checked but now we must
            # check that the solutions do not set denominators
            # in any factor to zero
            dens = flags.get('_denominators', _simple_dens(f, symbols))
            result = [s for s in result if
                not any(checksol(den, s, **flags) for den in
                        dens)]
        # set flags for quick exit at end; solutions for each
        # factor were already checked and simplified
        check = False
        flags['simplify'] = False

    elif f.is_Piecewise:
        result = set()
        if any(e.is_zero for e, c in f.args):
            f = f.simplify()  # failure imminent w/o help

        cond = neg = True
        for expr, cnd in f.args:
            # the explicit condition for this expr is the current cond
            # and none of the previous conditions
            cond = And(neg, cnd)
            neg = And(neg, ~cond)

            if expr.is_zero and cond.simplify() != False:
                raise NotImplementedError(filldedent('''
                    An expression is already zero when %s.
                    This means that in this *region* the solution
                    is zero but solve can only represent discrete,
                    not interval, solutions. If this is a spurious
                    interval it might be resolved with simplification
                    of the Piecewise conditions.''' % cond))
            candidates = _vsolve(expr, symbol, **flags)

            for candidate in candidates:
                if candidate in result:
                    # an unconditional value was already there
                    continue
                try:
                    v = cond.subs(symbol, candidate)
                    _eval_simplify = getattr(v, '_eval_simplify', None)
                    if _eval_simplify is not None:
                        # unconditionally take the simplification of v
                        v = _eval_simplify(ratio=2, measure=lambda x: 1)
                except TypeError:
                    # incompatible type with condition(s)
                    continue
                if v == False:
                    continue
                if v == True:
                    result.add(candidate)
                else:
                    result.add(Piecewise(
                        (candidate, v),
                        (S.NaN, True)))
        # solutions already checked and simplified
        # ****************************************
        return [{symbol: r} for r in result]
    else:
        # first see if it really depends on symbol and whether there
        # is only a linear solution
        f_num, sol = solve_linear(f, symbols=symbols)
        if f_num.is_zero or sol is S.NaN:
            return []
        elif f_num.is_Symbol:
            # no need to check but simplify if desired
            if flags.get('simplify', True):
                sol = simplify(sol)
            return [{f_num: sol}]

        poly = None
        # check for a single Add generator
        if not f_num.is_Add:
            add_args = [i for i in f_num.atoms(Add)
                if symbol in i.free_symbols]
            if len(add_args) == 1:
                gen = add_args[0]
                spart = gen.as_independent(symbol)[1].as_base_exp()[0]
                if spart == symbol:
                    try:
                        poly = Poly(f_num, spart)
                    except PolynomialError:
                        pass

        result = False  # no solution was obtained
        msg = ''  # there is no failure message

        # Poly is generally robust enough to convert anything to
        # a polynomial and tell us the different generators that it
        # contains, so we will inspect the generators identified by
        # polys to figure out what to do.

        # try to identify a single generator that will allow us to solve this
        # as a polynomial, followed (perhaps) by a change of variables if the
        # generator is not a symbol

        try:
            if poly is None:
                poly = Poly(f_num)
            if poly is None:
                raise ValueError('could not convert %s to Poly' % f_num)
        except GeneratorsNeeded:
            simplified_f = simplify(f_num)
            if simplified_f != f_num:
                return _solve(simplified_f, symbol, **flags)
            raise ValueError('expression appears to be a constant')

        gens = [g for g in poly.gens if g.has(symbol)]

        def _as_base_q(x):
            """Return (b**e, q) for x = b**(p*e/q) where p/q is the leading
            Rational of the exponent of x, e.g. exp(-2*x/3) -> (exp(x), 3)
            """
            b, e = x.as_base_exp()
            if e.is_Rational:
                return b, e.q
            if not e.is_Mul:
                return x, 1
            c, ee = e.as_coeff_Mul()
            if c.is_Rational and c is not S.One:  # c could be a Float
                return b**ee, c.q
            return x, 1

        if len(gens) > 1:
            # If there is more than one generator, it could be that the
            # generators have the same base but different powers, e.g.
            #   >>> Poly(exp(x) + 1/exp(x))
            #   Poly(exp(-x) + exp(x), exp(-x), exp(x), domain='ZZ')
            #
            # If unrad was not disabled then there should be no rational
            # exponents appearing as in
            #   >>> Poly(sqrt(x) + sqrt(sqrt(x)))
            #   Poly(sqrt(x) + x**(1/4), sqrt(x), x**(1/4), domain='ZZ')

            bases, qs = list(zip(*[_as_base_q(g) for g in gens]))
            bases = set(bases)

            if len(bases) > 1 or not all(q == 1 for q in qs):
                funcs = {b for b in bases if b.is_Function}

                trig = {_ for _ in funcs if
                    isinstance(_, TrigonometricFunction)}
                other = funcs - trig
                if not other and len(funcs.intersection(trig)) > 1:
                    newf = None
                    if f_num.is_Add and len(f_num.args) == 2:
                        # check for sin(x)**p = cos(x)**p
                        _args = f_num.args
                        t = a, b = [i.atoms(Function).intersection(
                            trig) for i in _args]
                        if all(len(i) == 1 for i in t):
                            a, b = [i.pop() for i in t]
                            if isinstance(a, cos):
                                a, b = b, a
                                _args = _args[::-1]
                            if isinstance(a, sin) and isinstance(b, cos
                                    ) and a.args[0] == b.args[0]:
                                # sin(x) + cos(x) = 0 -> tan(x) + 1 = 0
                                newf, _d = (TR2i(_args[0]/_args[1]) + 1
                                    ).as_numer_denom()
                                if not _d.is_Number:
                                    newf = None
                    if newf is None:
                        newf = TR1(f_num).rewrite(tan)
                    if newf != f_num:
                        # don't check the rewritten form --check
                        # solutions in the un-rewritten form below
                        flags['check'] = False
                        result = _solve(newf, symbol, **flags)
                        flags['check'] = check

                # just a simple case - see if replacement of single function
                # clears all symbol-dependent functions, e.g.
                # log(x) - log(log(x) - 1) - 3 can be solved even though it has
                # two generators.

                if result is False and funcs:
                    funcs = list(ordered(funcs))  # put shallowest function first
                    f1 = funcs[0]
                    t = Dummy('t')
                    # perform the substitution
                    ftry = f_num.subs(f1, t)

                    # if no Functions left, we can proceed with usual solve
                    if not ftry.has(symbol):
                        cv_sols = _solve(ftry, t, **flags)
                        cv_inv = list(ordered(_vsolve(t - f1, symbol, **flags)))[0]
                        result = [{symbol: cv_inv.subs(sol)} for sol in cv_sols]

                if result is False:
                    msg = 'multiple generators %s' % gens

            else:
                # e.g. case where gens are exp(x), exp(-x)
                u = bases.pop()
                t = Dummy('t')
                inv = _vsolve(u - t, symbol, **flags)
                if isinstance(u, (Pow, exp)):
                    # this will be resolved by factor in _tsolve but we might
                    # as well try a simple expansion here to get things in
                    # order so something like the following will work now without
                    # having to factor:
                    #
                    # >>> eq = (exp(I*(-x-2))+exp(I*(x+2)))
                    # >>> eq.subs(exp(x),y)  # fails
                    # exp(I*(-x - 2)) + exp(I*(x + 2))
                    # >>> eq.expand().subs(exp(x),y)  # works
                    # y**I*exp(2*I) + y**(-I)*exp(-2*I)
                    def _expand(p):
                        b, e = p.as_base_exp()
                        e = expand_mul(e)
                        return expand_power_exp(b**e)
                    ftry = f_num.replace(
                        lambda w: w.is_Pow or isinstance(w, exp),
                        _expand).subs(u, t)
                    if not ftry.has(symbol):
                        soln = _solve(ftry, t, **flags)
                        result = [{symbol: i.subs(s)} for i in inv for s in soln]

        elif len(gens) == 1:

            # There is only one generator that we are interested in, but
            # there may have been more than one generator identified by
            # polys (e.g. for symbols other than the one we are interested
            # in) so recast the poly in terms of our generator of interest.
            # Also use composite=True with f_num since Poly won't update
            # poly as documented in issue 8810.

            poly = Poly(f_num, gens[0], composite=True)

            # if we aren't on the tsolve-pass, use roots
            if not flags.pop('tsolve', False):
                soln = None
                deg = poly.degree()
                flags['tsolve'] = True
                hints = ('cubics', 'quartics', 'quintics')
                solvers = {h: flags.get(h) for h in hints}
                soln = roots(poly, **solvers)
                if sum(soln.values()) < deg:
                    # e.g. roots(32*x**5 + 400*x**4 + 2032*x**3 +
                    #            5000*x**2 + 6250*x + 3189) -> {}
                    # so all_roots is used and RootOf instances are
                    # returned *unless* the system is multivariate
                    # or high-order EX domain.
                    try:
                        soln = poly.all_roots()
                    except NotImplementedError:
                        if not flags.get('incomplete', True):
                                raise NotImplementedError(
                                filldedent('''
    Neither high-order multivariate polynomials
    nor sorting of EX-domain polynomials is supported.
    If you want to see any results, pass keyword incomplete=True to
    solve; to see numerical values of roots
    for univariate expressions, use nroots.
    '''))
                        else:
                            pass
                else:
                    soln = list(soln.keys())

                if soln is not None:
                    u = poly.gen
                    if u != symbol:
                        try:
                            t = Dummy('t')
                            inv = _vsolve(u - t, symbol, **flags)
                            soln = {i.subs(t, s) for i in inv for s in soln}
                        except NotImplementedError:
                            # perhaps _tsolve can handle f_num
                            soln = None
                    else:
                        check = False  # only dens need to be checked
                    if soln is not None:
                        if len(soln) > 2:
                            # if the flag wasn't set then unset it since high-order
                            # results are quite long. Perhaps one could base this
                            # decision on a certain critical length of the
                            # roots. In addition, wester test M2 has an expression
                            # whose roots can be shown to be real with the
                            # unsimplified form of the solution whereas only one of
                            # the simplified forms appears to be real.
                            flags['simplify'] = flags.get('simplify', False)
                if soln is not None:
                    result = [{symbol: v} for v in soln]

    # fallback if above fails
    # -----------------------
    if result is False:
        # try unrad
        if flags.pop('_unrad', True):
            try:
                u = unrad(f_num, symbol)
            except (ValueError, NotImplementedError):
                u = False
            if u:
                eq, cov = u
                if cov:
                    isym, ieq = cov
                    inv = _vsolve(ieq, symbol, **flags)[0]
                    rv = {inv.subs(xi) for xi in _solve(eq, isym, **flags)}
                else:
                    try:
                        rv = set(_vsolve(eq, symbol, **flags))
                    except NotImplementedError:
                        rv = None
                if rv is not None:
                    result = [{symbol: v} for v in rv]
                    # if the flag wasn't set then unset it since unrad results
                    # can be quite long or of very high order
                    flags['simplify'] = flags.get('simplify', False)
            else:
                pass  # for coverage

    # try _tsolve
    if result is False:
        flags.pop('tsolve', None)  # allow tsolve to be used on next pass
        try:
            soln = _tsolve(f_num, symbol, **flags)
            if soln is not None:
                result = [{symbol: v} for v in soln]
        except PolynomialError:
            pass
    # ----------- end of fallback ----------------------------

    if result is False:
        raise NotImplementedError('\n'.join([msg, not_impl_msg % f]))

    result = _remove_duplicate_solutions(result)

    if flags.get('simplify', True):
        result = [{k: d[k].simplify() for k in d} for d in result]
        # Simplification might reveal more duplicates
        result = _remove_duplicate_solutions(result)
        # we just simplified the solution so we now set the flag to
        # False so the simplification doesn't happen again in checksol()
        flags['simplify'] = False

    if checkdens:
        # reject any result that makes any denom. affirmatively 0;
        # if in doubt, keep it
        dens = _simple_dens(f, symbols)
        result = [r for r in result if
                  not any(checksol(d, r, **flags)
                          for d in dens)]
    if check:
        # keep only results if the check is not False
        result = [r for r in result if
                  checksol(f_num, r, **flags) is not False]
    return result


def _remove_duplicate_solutions(solutions: list[dict[Expr, Expr]]
                                ) -> list[dict[Expr, Expr]]:
    """Remove duplicates from a list of dicts"""
    solutions_set = set()
    solutions_new = []

    for sol in solutions:
        solset = frozenset(sol.items())
        if solset not in solutions_set:
            solutions_new.append(sol)
            solutions_set.add(solset)

    return solutions_new


def _solve_system(exprs, symbols, **flags):
    """return ``(linear, solution)`` where ``linear`` is True
    if the system was linear, else False; ``solution``
    is a list of dictionaries giving solutions for the symbols
    """
    if not exprs:
        return False, []

    if flags.pop('_split', True):
        # Split the system into connected components
        V = exprs
        symsset = set(symbols)
        exprsyms = {e: e.free_symbols & symsset for e in exprs}
        E = []
        sym_indices = {sym: i for i, sym in enumerate(symbols)}
        for n, e1 in enumerate(exprs):
            for e2 in exprs[:n]:
                # Equations are connected if they share a symbol
                if exprsyms[e1] & exprsyms[e2]:
                    E.append((e1, e2))
        G = V, E
        subexprs = connected_components(G)
        if len(subexprs) > 1:
            subsols = []
            linear = True
            for subexpr in subexprs:
                subsyms = set()
                for e in subexpr:
                    subsyms |= exprsyms[e]
                subsyms = sorted(subsyms, key = lambda x: sym_indices[x])
                flags['_split'] = False  # skip split step
                _linear, subsol = _solve_system(subexpr, subsyms, **flags)
                if linear:
                    linear = linear and _linear
                if not isinstance(subsol, list):
                    subsol = [subsol]
                subsols.append(subsol)
            # Full solution is cartesian product of subsystems
            sols = []
            for soldicts in product(*subsols):
                sols.append(dict(item for sd in soldicts
                    for item in sd.items()))
            return linear, sols

    polys = []
    dens = set()
    failed = []
    result = []
    solved_syms = []
    linear = True
    manual = flags.get('manual', False)
    checkdens = check = flags.get('check', True)

    for j, g in enumerate(exprs):
        dens.update(_simple_dens(g, symbols))
        i, d = _invert(g, *symbols)
        if d in symbols:
            if linear:
                linear = solve_linear(g, 0, [d])[0] == d
        g = d - i
        g = g.as_numer_denom()[0]
        if manual:
            failed.append(g)
            continue

        poly = g.as_poly(*symbols, extension=True)

        if poly is not None:
            polys.append(poly)
        else:
            failed.append(g)

    if polys:
        if all(p.is_linear for p in polys):
            n, m = len(polys), len(symbols)
            matrix = zeros(n, m + 1)

            for i, poly in enumerate(polys):
                for monom, coeff in poly.terms():
                    try:
                        j = monom.index(1)
                        matrix[i, j] = coeff
                    except ValueError:
                        matrix[i, m] = -coeff

            # returns a dictionary ({symbols: values}) or None
            if flags.pop('particular', False):
                result = minsolve_linear_system(matrix, *symbols, **flags)
            else:
                result = solve_linear_system(matrix, *symbols, **flags)
            result = [result] if result else []
            if failed:
                if result:
                    solved_syms = list(result[0].keys())  # there is only one result dict
                else:
                    solved_syms = []
            # linear doesn't change
        else:
            linear = False
            if len(symbols) > len(polys):

                free = set().union(*[p.free_symbols for p in polys])
                free = list(ordered(free.intersection(symbols)))
                got_s = set()
                result = []
                for syms in subsets(free, min(len(free), len(polys))):
                    try:
                        # returns [], None or list of tuples
                        res = solve_poly_system(polys, *syms)
                        if res:
                            for r in set(res):
                                skip = False
                                for r1 in r:
                                    if got_s and any(ss in r1.free_symbols
                                           for ss in got_s):
                                        # sol depends on previously
                                        # solved symbols: discard it
                                        skip = True
                                if not skip:
                                    got_s.update(syms)
                                    result.append(dict(list(zip(syms, r))))
                    except NotImplementedError:
                        pass
                if got_s:
                    solved_syms = list(got_s)
                else:
                    failed.extend([g.as_expr() for g in polys])
            else:
                try:
                    result = solve_poly_system(polys, *symbols)
                    if result:
                        solved_syms = symbols
                        result = [dict(list(zip(solved_syms, r))) for r in set(result)]
                except NotImplementedError:
                    failed.extend([g.as_expr() for g in polys])
                    solved_syms = []

    # convert None or [] to [{}]
    result = result or [{}]

    if failed:
        linear = False
        # For each failed equation, see if we can solve for one of the
        # remaining symbols from that equation. If so, we update the
        # solution set and continue with the next failed equation,
        # repeating until we are done or we get an equation that can't
        # be solved.
        def _ok_syms(e, sort=False):
            rv = e.free_symbols & legal

            # Solve first for symbols that have lower degree in the equation.
            # Ideally we want to solve firstly for symbols that appear linearly
            # with rational coefficients e.g. if e = x*y + z then we should
            # solve for z first.
            def key(sym):
                ep = e.as_poly(sym)
                if ep is None:
                    complexity = (S.Infinity, S.Infinity, S.Infinity)
                else:
                    coeff_syms = ep.LC().free_symbols
                    complexity = (ep.degree(), len(coeff_syms & rv), len(coeff_syms))
                return complexity + (default_sort_key(sym),)

            if sort:
                rv = sorted(rv, key=key)
            return rv

        legal = set(symbols)  # what we are interested in
        # sort so equation with the fewest potential symbols is first
        u = Dummy()  # used in solution checking
        for eq in ordered(failed, lambda _: len(_ok_syms(_))):
            newresult = []
            bad_results = []
            hit = False
            for r in result:
                got_s = set()
                # update eq with everything that is known so far
                eq2 = eq.subs(r)
                # if check is True then we see if it satisfies this
                # equation, otherwise we just accept it
                if check and r:
                    b = checksol(u, u, eq2, minimal=True)
                    if b is not None:
                        # this solution is sufficient to know whether
                        # it is valid or not so we either accept or
                        # reject it, then continue
                        if b:
                            newresult.append(r)
                        else:
                            bad_results.append(r)
                        continue
                # search for a symbol amongst those available that
                # can be solved for
                ok_syms = _ok_syms(eq2, sort=True)
                if not ok_syms:
                    if r:
                        newresult.append(r)
                    break  # skip as it's independent of desired symbols
                for s in ok_syms:
                    try:
                        soln = _vsolve(eq2, s, **flags)
                    except NotImplementedError:
                        continue
                    # put each solution in r and append the now-expanded
                    # result in the new result list; use copy since the
                    # solution for s is being added in-place
                    for sol in soln:
                        if got_s and any(ss in sol.free_symbols for ss in got_s):
                            # sol depends on previously solved symbols: discard it
                            continue
                        rnew = r.copy()
                        for k, v in r.items():
                            rnew[k] = v.subs(s, sol)
                        # and add this new solution
                        rnew[s] = sol
                        # check that it is independent of previous solutions
                        iset = set(rnew.items())
                        for i in newresult:
                            if len(i) < len(iset):
                                # update i with what is known
                                i_items_updated = {(k, v.xreplace(rnew)) for k, v in i.items()}
                                if not i_items_updated - iset:
                                    # this is a superset of a known solution that
                                    # is smaller
                                    break
                        else:
                            # keep it
                            newresult.append(rnew)
                    hit = True
                    got_s.add(s)
                if not hit:
                    raise NotImplementedError('could not solve %s' % eq2)
            else:
                result = newresult
                for b in bad_results:
                    if b in result:
                        result.remove(b)

    if not result:
        return False, []

    # rely on linear/polynomial system solvers to simplify
    # XXX the following tests show that the expressions
    # returned are not the same as they would be if simplify
    # were applied to this:
    #   sympy/solvers/ode/tests/test_systems/test__classify_linear_system
    #   sympy/solvers/tests/test_solvers/test_issue_4886
    # so the docs should be updated to reflect that or else
    # the following should be `bool(failed) or not linear`
    default_simplify = bool(failed)
    if flags.get('simplify', default_simplify):
        for r in result:
            for k in r:
                r[k] = simplify(r[k])
        flags['simplify'] = False  # don't need to do so in checksol now

    if checkdens:
        result = [r for r in result
            if not any(checksol(d, r, **flags) for d in dens)]

    if check and not linear:
        result = [r for r in result
            if not any(checksol(e, r, **flags) is False for e in exprs)]

    result = [r for r in result if r]
    return linear, result


def solve_linear(lhs, rhs=0, symbols=[], exclude=[]):
    r"""
    Return a tuple derived from ``f = lhs - rhs`` that is one of
    the following: ``(0, 1)``, ``(0, 0)``, ``(symbol, solution)``, ``(n, d)``.

    Explanation
    ===========

    ``(0, 1)`` meaning that ``f`` is independent of the symbols in *symbols*
    that are not in *exclude*.

    ``(0, 0)`` meaning that there is no solution to the equation amongst the
    symbols given. If the first element of the tuple is not zero, then the
    function is guaranteed to be dependent on a symbol in *symbols*.

    ``(symbol, solution)`` where symbol appears linearly in the numerator of
    ``f``, is in *symbols* (if given), and is not in *exclude* (if given). No
    simplification is done to ``f`` other than a ``mul=True`` expansion, so the
    solution will correspond strictly to a unique solution.

    ``(n, d)`` where ``n`` and ``d`` are the numerator and denominator of ``f``
    when the numerator was not linear in any symbol of interest; ``n`` will
    never be a symbol unless a solution for that symbol was found (in which case
    the second element is the solution, not the denominator).

    Examples
    ========

    >>> from sympy import cancel, Pow

    ``f`` is independent of the symbols in *symbols* that are not in
    *exclude*:

    >>> from sympy import cos, sin, solve_linear
    >>> from sympy.abc import x, y, z
    >>> eq = y*cos(x)**2 + y*sin(x)**2 - y  # = y*(1 - 1) = 0
    >>> solve_linear(eq)
    (0, 1)
    >>> eq = cos(x)**2 + sin(x)**2  # = 1
    >>> solve_linear(eq)
    (0, 1)
    >>> solve_linear(x, exclude=[x])
    (0, 1)

    The variable ``x`` appears as a linear variable in each of the
    following:

    >>> solve_linear(x + y**2)
    (x, -y**2)
    >>> solve_linear(1/x - y**2)
    (x, y**(-2))

    When not linear in ``x`` or ``y`` then the numerator and denominator are
    returned:

    >>> solve_linear(x**2/y**2 - 3)
    (x**2 - 3*y**2, y**2)

    If the numerator of the expression is a symbol, then ``(0, 0)`` is
    returned if the solution for that symbol would have set any
    denominator to 0:

    >>> eq = 1/(1/x - 2)
    >>> eq.as_numer_denom()
    (x, 1 - 2*x)
    >>> solve_linear(eq)
    (0, 0)

    But automatic rewriting may cause a symbol in the denominator to
    appear in the numerator so a solution will be returned:

    >>> (1/x)**-1
    x
    >>> solve_linear((1/x)**-1)
    (x, 0)

    Use an unevaluated expression to avoid this:

    >>> solve_linear(Pow(1/x, -1, evaluate=False))
    (0, 0)

    If ``x`` is allowed to cancel in the following expression, then it
    appears to be linear in ``x``, but this sort of cancellation is not
    done by ``solve_linear`` so the solution will always satisfy the
    original expression without causing a division by zero error.

    >>> eq = x**2*(1/x - z**2/x)
    >>> solve_linear(cancel(eq))
    (x, 0)
    >>> solve_linear(eq)
    (x**2*(1 - z**2), x)

    A list of symbols for which a solution is desired may be given:

    >>> solve_linear(x + y + z, symbols=[y])
    (y, -x - z)

    A list of symbols to ignore may also be given:

    >>> solve_linear(x + y + z, exclude=[x])
    (y, -x - z)

    (A solution for ``y`` is obtained because it is the first variable
    from the canonically sorted list of symbols that had a linear
    solution.)

    """
    if isinstance(lhs, Eq):
        if rhs:
            raise ValueError(filldedent('''
            If lhs is an Equality, rhs must be 0 but was %s''' % rhs))
        rhs = lhs.rhs
        lhs = lhs.lhs
    dens = None
    eq = lhs - rhs
    n, d = eq.as_numer_denom()
    if not n:
        return S.Zero, S.One

    free = n.free_symbols
    if not symbols:
        symbols = free
    else:
        bad = [s for s in symbols if not s.is_Symbol]
        if bad:
            if len(bad) == 1:
                bad = bad[0]
            if len(symbols) == 1:
                eg = 'solve(%s, %s)' % (eq, symbols[0])
            else:
                eg = 'solve(%s, *%s)' % (eq, list(symbols))
            raise ValueError(filldedent('''
                solve_linear only handles symbols, not %s. To isolate
                non-symbols use solve, e.g. >>> %s <<<.
                             ''' % (bad, eg)))
