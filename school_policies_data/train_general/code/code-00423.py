        a Rational result, or is greater than 1) then the evalf value will be
        used to return True or False.

        """
        from sympy.simplify.simplify import nsimplify, simplify
        from sympy.solvers.solvers import solve
        from sympy.polys.polyerrors import NotAlgebraic
        from sympy.polys.numberfields import minimal_polynomial

        other = sympify(other)

        if not isinstance(other, Expr):
            return False

        if self == other:
            return True

        # they aren't the same so see if we can make the difference 0;
        # don't worry about doing simplification steps one at a time
        # because if the expression ever goes to 0 then the subsequent
        # simplification steps that are done will be very fast.
        diff = factor_terms(simplify(self - other), radical=True)

        if not diff:
            return True

        if not diff.has(Add, Mod):
            # if there is no expanding to be done after simplifying
            # then this can't be a zero
            return False

        factors = diff.as_coeff_mul()[1]
        if len(factors) > 1:  # avoid infinity recursion
            fac_zero = [fac.equals(0) for fac in factors]
            if None not in fac_zero:  # every part can be decided
                return any(fac_zero)

        constant = diff.is_constant(simplify=False, failing_number=True)

        if constant is False:
            return False

        if not diff.is_number:
            if constant is None:
                # e.g. unless the right simplification is done, a symbolic
                # zero is possible (see expression of issue 6829: without
                # simplification constant will be None).
                return

        if constant is True:
            # this gives a number whether there are free symbols or not
            ndiff = diff._random()
            # is_comparable will work whether the result is real
            # or complex; it could be None, however.
            if ndiff and ndiff.is_comparable:
                return False

        # sometimes we can use a simplified result to give a clue as to
        # what the expression should be; if the expression is *not* zero
        # then we should have been able to compute that and so now
        # we can just consider the cases where the approximation appears
        # to be zero -- we try to prove it via minimal_polynomial.
        #
        # removed
        # ns = nsimplify(diff)
        # if diff.is_number and (not ns or ns == diff):
        #
        # The thought was that if it nsimplifies to 0 that's a sure sign
        # to try the following to prove it; or if it changed but wasn't
        # zero that might be a sign that it's not going to be easy to
        # prove. But tests seem to be working without that logic.
        #
        if diff.is_number:
            # try to prove via self-consistency
            surds = [s for s in diff.atoms(Pow) if s.args[0].is_Integer]
            # it seems to work better to try big ones first
            surds.sort(key=lambda x: -x.args[0])
            for s in surds:
                try:
                    # simplify is False here -- this expression has already
                    # been identified as being hard to identify as zero;
                    # we will handle the checking ourselves using nsimplify
                    # to see if we are in the right ballpark or not and if so
                    # *then* the simplification will be attempted.
                    sol = solve(diff, s, simplify=False)
                    if sol:
                        if s in sol:
                            # the self-consistent result is present
                            return True
                        if all(si.is_Integer for si in sol):
                            # perfect powers are removed at instantiation
                            # so surd s cannot be an integer
                            return False
                        if all(i.is_algebraic is False for i in sol):
                            # a surd is algebraic
                            return False
                        if any(si in surds for si in sol):
                            # it wasn't equal to s but it is in surds
                            # and different surds are not equal
                            return False
                        if any(nsimplify(s - si) == 0 and
                                simplify(s - si) == 0 for si in sol):
                            return True
                        if s.is_real:
                            if any(nsimplify(si, [s]) == s and simplify(si) == s
                                    for si in sol):
                                return True
                except NotImplementedError:
                    pass

            # try to prove with minimal_polynomial but know when
            # *not* to use this or else it can take a long time. e.g. issue 8354
            if True:  # change True to condition that assures non-hang
                try:
                    mp = minimal_polynomial(diff)
                    if mp.is_Symbol:
                        return True
                    return False
                except (NotAlgebraic, NotImplementedError):
                    pass

        # diff has not simplified to zero; constant is either None, True
        # or the number with significance (is_comparable) that was randomly
        # calculated twice as the same value.
        if constant not in (True, None) and constant != 0:
            return False

        if failing_expression:
            return diff
        return None

    def _eval_is_extended_positive_negative(self, positive):
        from sympy.polys.numberfields import minimal_polynomial
        from sympy.polys.polyerrors import NotAlgebraic
        if self.is_number:
            # check to see that we can get a value
            try:
                n2 = self._eval_evalf(2)
            # XXX: This shouldn't be caught here
            # Catches ValueError: hypsum() failed to converge to the requested
            # 34 bits of accuracy
            except ValueError:
                return None
            if n2 is None:
                return None
            if getattr(n2, '_prec', 1) == 1:  # no significance
                return None
            if n2 is S.NaN:
                return None

            f = self.evalf(2)
            if f.is_Float:
                match = f, S.Zero
            else:
                match = pure_complex(f)
            if match is None:
                return False
            r, i = match
            if not (i.is_Number and r.is_Number):
                return False
            if r._prec != 1 and i._prec != 1:
                return bool(not i and ((r > 0) if positive else (r < 0)))
            elif r._prec == 1 and (not i or i._prec == 1) and \
                    self._eval_is_algebraic() and not self.has(Function):
                try:
                    if minimal_polynomial(self).is_Symbol:
                        return False
                except (NotAlgebraic, NotImplementedError):
                    pass

    def _eval_is_extended_positive(self):
        return self._eval_is_extended_positive_negative(positive=True)

    def _eval_is_extended_negative(self):
        return self._eval_is_extended_positive_negative(positive=False)

    def _eval_interval(self, x, a, b):
        """
        Returns evaluation over an interval.  For most functions this is:

        self.subs(x, b) - self.subs(x, a),

        possibly using limit() if NaN is returned from subs, or if
        singularities are found between a and b.

        If b or a is None, it only evaluates -self.subs(x, a) or self.subs(b, x),
        respectively.

        """
        from sympy.calculus.accumulationbounds import AccumBounds
        from sympy.functions.elementary.exponential import log
        from sympy.series.limits import limit, Limit
        from sympy.sets.sets import Interval
        from sympy.solvers.solveset import solveset

        if (a is None and b is None):
            raise ValueError('Both interval ends cannot be None.')

        def _eval_endpoint(left):
            c = a if left else b
            if c is None:
                return S.Zero
            else:
                C = self.subs(x, c)
                if C.has(S.NaN, S.Infinity, S.NegativeInfinity,
                         S.ComplexInfinity, AccumBounds):
                    if (a < b) != False:
                        C = limit(self, x, c, "+" if left else "-")
                    else:
                        C = limit(self, x, c, "-" if left else "+")

                    if isinstance(C, Limit):
                        raise NotImplementedError("Could not compute limit")
            return C

        if a == b:
            return S.Zero

        A = _eval_endpoint(left=True)
        if A is S.NaN:
            return A

        B = _eval_endpoint(left=False)

        if (a and b) is None:
            return B - A

        value = B - A

        if a.is_comparable and b.is_comparable:
            if a < b:
                domain = Interval(a, b)
            else:
                domain = Interval(b, a)
            # check the singularities of self within the interval
            # if singularities is a ConditionSet (not iterable), catch the exception and pass
            singularities = solveset(self.cancel().as_numer_denom()[1], x,
                domain=domain)
            for logterm in self.atoms(log):
                singularities = singularities | solveset(logterm.args[0], x,
                    domain=domain)
            try:
                for s in singularities:
                    if value is S.NaN:
                        # no need to keep adding, it will stay NaN
                        break
                    if not s.is_comparable:
                        continue
                    if (a < s) == (s < b) == True:
                        value += -limit(self, x, s, "+") + limit(self, x, s, "-")
                    elif (b < s) == (s < a) == True:
                        value += limit(self, x, s, "+") - limit(self, x, s, "-")
            except TypeError:
                pass

        return value

    def _eval_power(self, expt) -> Expr | None:
        # subclass to compute self**other for cases when
        # other is not NaN, 0, or 1
        return None

    def _eval_conjugate(self):
        if self.is_extended_real:
            return self
        elif self.is_imaginary:
            return -self

    def conjugate(self):
        """Returns the complex conjugate of 'self'."""
        from sympy.functions.elementary.complexes import conjugate as c
        return c(self)

    def dir(self, x, cdir):
        if self.is_zero:
            return S.Zero
        from sympy.functions.elementary.exponential import log
        minexp = S.Zero
        arg = self
        while arg:
            minexp += S.One
            arg = arg.diff(x)
            coeff = arg.subs(x, 0)
            if coeff is S.NaN:
                coeff = arg.limit(x, 0)
            if coeff is S.ComplexInfinity:
                try:
                    coeff, _ = arg.leadterm(x)
                    if coeff.has(log(x)):
                        raise ValueError()
                except ValueError:
                    coeff = arg.limit(x, 0)
            if coeff != S.Zero:
                break
        return coeff*cdir**minexp

    def _eval_transpose(self):
        from sympy.functions.elementary.complexes import conjugate
        if self.is_commutative:
            return self
        elif self.is_hermitian:
            return conjugate(self)
        elif self.is_antihermitian:
            return -conjugate(self)

    def transpose(self):
        from sympy.functions.elementary.complexes import transpose
        return transpose(self)

    def _eval_adjoint(self):
        from sympy.functions.elementary.complexes import conjugate, transpose
        if self.is_hermitian:
            return self
        elif self.is_antihermitian:
            return -self
        obj = self._eval_conjugate()
        if obj is not None:
            return transpose(obj)
        obj = self._eval_transpose()
        if obj is not None:
            return conjugate(obj)

    def adjoint(self):
        from sympy.functions.elementary.complexes import adjoint
        return adjoint(self)

    @classmethod
    def _parse_order(cls, order):
        """Parse and configure the ordering of terms. """
        from sympy.polys.orderings import monomial_key

        startswith = getattr(order, "startswith", None)
        if startswith is None:
            reverse = False
        else:
            reverse = startswith('rev-')
            if reverse:
                order = order[4:]

        monom_key = monomial_key(order)

        def neg(monom):
            return tuple([neg(m) if isinstance(m, tuple) else -m for m in monom])

        def key(term):
            _, ((re, im), monom, ncpart) = term

            monom = neg(monom_key(monom))
            ncpart = tuple([e.sort_key(order=order) for e in ncpart])
            coeff = ((bool(im), im), (re, im))

            return monom, ncpart, coeff

        return key, reverse

    def as_ordered_factors(self, order=None):
        """Return list of ordered factors (if Mul) else [self]."""
        return [self]

    def as_poly(self, *gens, **args):
        """Converts ``self`` to a polynomial or returns ``None``.

        Explanation
        ===========

        >>> from sympy import sin
        >>> from sympy.abc import x, y

        >>> print((x**2 + x*y).as_poly())
        Poly(x**2 + x*y, x, y, domain='ZZ')

        >>> print((x**2 + x*y).as_poly(x, y))
        Poly(x**2 + x*y, x, y, domain='ZZ')

        >>> print((x**2 + sin(y)).as_poly(x, y))
        None

        """
        from sympy.polys.polyerrors import PolynomialError, GeneratorsNeeded
        from sympy.polys.polytools import Poly

        try:
            poly = Poly(self, *gens, **args)

            if not poly.is_Poly:
                return None
            else:
                return poly
        except (PolynomialError, GeneratorsNeeded):
            # PolynomialError is caught for e.g. exp(x).as_poly(x)
            # GeneratorsNeeded is caught for e.g. S(2).as_poly()
            return None

    def as_ordered_terms(self, order=None, data=False):
        """
        Transform an expression to an ordered list of terms.

        Examples
        ========

        >>> from sympy import sin, cos
        >>> from sympy.abc import x

        >>> (sin(x)**2*cos(x) + sin(x)**2 + 1).as_ordered_terms()
        [sin(x)**2*cos(x), sin(x)**2, 1]

        """

        from .numbers import Number, NumberSymbol

        if order is None and self.is_Add:
            # Spot the special case of Add(Number, Mul(Number, expr)) with the
            # first number positive and the second number negative
            key = lambda x:not isinstance(x, (Number, NumberSymbol))
            add_args = sorted(Add.make_args(self), key=key)
            if (len(add_args) == 2
                and isinstance(add_args[0], (Number, NumberSymbol))
                and isinstance(add_args[1], Mul)):
                mul_args = sorted(Mul.make_args(add_args[1]), key=key)
                if (len(mul_args) == 2
                    and isinstance(mul_args[0], Number)
                    and add_args[0].is_positive
                    and mul_args[0].is_negative):
                    return add_args

        key, reverse = self._parse_order(order)
        terms, gens = self.as_terms()

        if not any(term.is_Order for term, _ in terms):
            ordered = sorted(terms, key=key, reverse=reverse)
        else:
            _terms, _order = [], []

            for term, repr in terms:
                if not term.is_Order:
                    _terms.append((term, repr))
                else:
                    _order.append((term, repr))

            ordered = sorted(_terms, key=key, reverse=True) \
                + sorted(_order, key=key, reverse=True)

        if data:
            return ordered, gens
        else:
            return [term for term, _ in ordered]

    def as_terms(self):
        """Transform an expression to a list of terms. """
        from .exprtools import decompose_power

        gens, terms = set(), []

        for term in Add.make_args(self):
            coeff, _term = term.as_coeff_Mul()

            coeff = complex(coeff)
            cpart, ncpart = {}, []

            if _term is not S.One:
                for factor in Mul.make_args(_term):
                    if factor.is_number:
                        try:
                            coeff *= complex(factor)
                        except (TypeError, ValueError):
                            pass
                        else:
                            continue

                    if factor.is_commutative:
                        base, exp = decompose_power(factor)

                        cpart[base] = exp
                        gens.add(base)
                    else:
                        ncpart.append(factor)

            coeff = coeff.real, coeff.imag
            ncpart = tuple(ncpart)

            terms.append((term, (coeff, cpart, ncpart)))

        gens = sorted(gens, key=default_sort_key)

        k, indices = len(gens), {}

        for i, g in enumerate(gens):
            indices[g] = i

        result = []

        for term, (coeff, cpart, ncpart) in terms:
            monom = [0]*k

            for base, exp in cpart.items():
                monom[indices[base]] = exp

            result.append((term, (coeff, tuple(monom), ncpart)))

        return result, gens

    def removeO(self) -> Expr:
        """Removes the additive O(..) symbol if there is one"""
        return self

    def getO(self) -> Expr | None:
        """Returns the additive O(..) symbol if there is one, else None."""
        return None

    def getn(self):
        """
        Returns the order of the expression.

        Explanation
        ===========

        The order is determined either from the O(...) term. If there
        is no O(...) term, it returns None.

        Examples
        ========

        >>> from sympy import O
        >>> from sympy.abc import x
        >>> (1 + x + O(x**2)).getn()
        2
        >>> (1 + x).getn()

        """
        o = self.getO()
        if o is None:
            return None
        elif o.is_Order:
            o = o.expr
            if o is S.One:
                return S.Zero
            if o.is_Symbol:
                return S.One
            if o.is_Pow:
                return o.args[1]
            if o.is_Mul:  # x**n*log(x)**n or x**n/log(x)**n
                for oi in o.args:
                    if oi.is_Symbol:
                        return S.One
                    if oi.is_Pow:
                        from .symbol import Dummy, Symbol
                        syms = oi.atoms(Symbol)
                        if len(syms) == 1:
                            x = syms.pop()
                            oi = oi.subs(x, Dummy('x', positive=True))
                            if oi.base.is_Symbol and oi.exp.is_Rational:
                                return abs(oi.exp)

        raise NotImplementedError('not sure of order of %s' % o)

    def count_ops(self, visual=False):
        from .function import count_ops
        return count_ops(self, visual)

    def args_cnc(self, cset=False, warn=True, split_1=True):
        """Return [commutative factors, non-commutative factors] of self.

        Explanation
        ===========

        self is treated as a Mul and the ordering of the factors is maintained.
        If ``cset`` is True the commutative factors will be returned in a set.
        If there were repeated factors (as may happen with an unevaluated Mul)
        then an error will be raised unless it is explicitly suppressed by
        setting ``warn`` to False.

        Note: -1 is always separated from a Number unless split_1 is False.

        Examples
        ========

        >>> from sympy import symbols, oo
        >>> A, B = symbols('A B', commutative=0)
        >>> x, y = symbols('x y')
        >>> (-2*x*y).args_cnc()
        [[-1, 2, x, y], []]
        >>> (-2.5*x).args_cnc()
        [[-1, 2.5, x], []]
        >>> (-2*x*A*B*y).args_cnc()
        [[-1, 2, x, y], [A, B]]
        >>> (-2*x*A*B*y).args_cnc(split_1=False)
        [[-2, x, y], [A, B]]
        >>> (-2*x*y).args_cnc(cset=True)
        [{-1, 2, x, y}, []]

        The arg is always treated as a Mul:

        >>> (-2 + x + A).args_cnc()
        [[], [x - 2 + A]]
        >>> (-oo).args_cnc() # -oo is a singleton
        [[-1, oo], []]
        """
        args = list(Mul.make_args(self))

        for i, mi in enumerate(args):
            if not mi.is_commutative:
                c = args[:i]
                nc = args[i:]
                break
        else:
            c = args
            nc = []

        if c and split_1 and (
            c[0].is_Number and
            c[0].is_extended_negative and
                c[0] is not S.NegativeOne):
            c[:1] = [S.NegativeOne, -c[0]]

        if cset:
            clen = len(c)
            c = set(c)
            if clen and warn and len(c) != clen:
                raise ValueError('repeated commutative arguments: %s' %
                                 [ci for ci in c if list(self.args).count(ci) > 1])
        return [c, nc]

    def coeff(self, x: Expr, n=1, right=False, _first=True):
        """
        Returns the coefficient from the term(s) containing ``x**n``. If ``n``
        is zero then all terms independent of ``x`` will be returned.

        Explanation
        ===========

        When ``x`` is noncommutative, the coefficient to the left (default) or
        right of ``x`` can be returned. The keyword 'right' is ignored when
        ``x`` is commutative.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.abc import x, y, z

        You can select terms that have an explicit negative in front of them:

        >>> (-x + 2*y).coeff(-1)
        x
        >>> (x - 2*y).coeff(-1)
        2*y

        You can select terms with no Rational coefficient:

        >>> (x + 2*y).coeff(1)
        x
        >>> (3 + 2*x + 4*x**2).coeff(1)
        0

        You can select terms independent of x by making n=0; in this case
        expr.as_independent(x)[0] is returned (and 0 will be returned instead
        of None):

        >>> (3 + 2*x + 4*x**2).coeff(x, 0)
        3
        >>> eq = ((x + 1)**3).expand() + 1
        >>> eq
        x**3 + 3*x**2 + 3*x + 2
        >>> [eq.coeff(x, i) for i in reversed(range(4))]
        [1, 3, 3, 2]
        >>> eq -= 2
        >>> [eq.coeff(x, i) for i in reversed(range(4))]
        [1, 3, 3, 0]

        You can select terms that have a numerical term in front of them:

        >>> (-x - 2*y).coeff(2)
        -y
        >>> from sympy import sqrt
        >>> (x + sqrt(2)*x).coeff(sqrt(2))
        x

        The matching is exact:

        >>> (3 + 2*x + 4*x**2).coeff(x)
        2
        >>> (3 + 2*x + 4*x**2).coeff(x**2)
        4
        >>> (3 + 2*x + 4*x**2).coeff(x**3)
        0
        >>> (z*(x + y)**2).coeff((x + y)**2)
        z
        >>> (z*(x + y)**2).coeff(x + y)
        0

        In addition, no factoring is done, so 1 + z*(1 + y) is not obtained
        from the following:

        >>> (x + z*(x + x*y)).coeff(x)
        1

        If such factoring is desired, factor_terms can be used first:

        >>> from sympy import factor_terms
        >>> factor_terms(x + z*(x + x*y)).coeff(x)
        z*(y + 1) + 1

        >>> n, m, o = symbols('n m o', commutative=False)
        >>> n.coeff(n)
        1
        >>> (3*n).coeff(n)
        3
        >>> (n*m + m*n*m).coeff(n) # = (1 + m)*n*m
        1 + m
        >>> (n*m + m*n*m).coeff(n, right=True) # = (1 + m)*n*m
        m

        If there is more than one possible coefficient 0 is returned:

        >>> (n*m + m*n).coeff(n)
        0

        If there is only one possible coefficient, it is returned:

        >>> (n*m + x*m*n).coeff(m*n)
        x
        >>> (n*m + x*m*n).coeff(m*n, right=1)
        1

        See Also
        ========

        as_coefficient: separate the expression into a coefficient and factor
        as_coeff_Add: separate the additive constant from an expression
        as_coeff_Mul: separate the multiplicative constant from an expression
        as_independent: separate x-dependent terms/factors from others
        sympy.polys.polytools.Poly.coeff_monomial: efficiently find the single coefficient of a monomial in Poly
        sympy.polys.polytools.Poly.nth: like coeff_monomial but powers of monomial terms are used
        """
        x = sympify(x)
        if not isinstance(x, Basic):
            return S.Zero

        n = as_int(n)

        if not x:
            return S.Zero

        if x == self:
            if n == 1:
                return S.One
            return S.Zero

        co2: list[Expr]

        if x is S.One:
            co2 = [a for a in Add.make_args(self) if a.as_coeff_Mul()[0] is S.One]
            if not co2:
                return S.Zero
            return Add(*co2)

        if n == 0:
            if x.is_Add and self.is_Add:
                c = self.coeff(x, right=right)
                if not c:
                    return S.Zero
                if not right:
                    return self - Add(*[a*x for a in Add.make_args(c)])
                return self - Add(*[x*a for a in Add.make_args(c)])
            return self.as_independent(x, as_Add=True)[0]

        # continue with the full method, looking for this power of x:
        x = x**n

        def incommon(l1, l2):
            if not l1 or not l2:
                return []
            n = min(len(l1), len(l2))
            for i in range(n):
                if l1[i] != l2[i]:
                    return l1[:i]
            return l1[:]

        def find(l, sub, first=True):
            """ Find where list sub appears in list l. When ``first`` is True
            the first occurrence from the left is returned, else the last
            occurrence is returned. Return None if sub is not in l.

            Examples
            ========

            >> l = range(5)*2
            >> find(l, [2, 3])
            2
            >> find(l, [2, 3], first=0)
            7
            >> find(l, [2, 4])
            None

            """
            if not sub or not l or len(sub) > len(l):
                return None
            n = len(sub)
            if not first:
                l.reverse()
                sub.reverse()
            for i in range(len(l) - n + 1):
                if all(l[i + j] == sub[j] for j in range(n)):
                    break
            else:
                i = None
            if not first:
                l.reverse()
                sub.reverse()
            if i is not None and not first:
                i = len(l) - (i + n)
            return i

        co2 = []
        co: list[tuple[set[Expr], list[Expr]]] = []
        args = Add.make_args(self)
        self_c = self.is_commutative
        x_c = x.is_commutative
        if self_c and not x_c:
            return S.Zero
        if _first and self.is_Add and not self_c and not x_c:
            # get the part that depends on x exactly
            xargs = Mul.make_args(x)
            d = Add(*[i for i in Add.make_args(self.as_independent(x)[1])
                if all(xi in Mul.make_args(i) for xi in xargs)])
            rv = d.coeff(x, right=right, _first=False)
            if not rv.is_Add or not right:
                return rv
            c_part, nc_part = zip(*[i.args_cnc() for i in rv.args])
            if has_variety(c_part):
                return rv
            return Add(*[Mul._from_args(i) for i in nc_part])

        one_c = self_c or x_c
        xargs, nx = x.args_cnc(cset=True, warn=bool(not x_c))
        # find the parts that pass the commutative terms
        for a in args:
            margs, nc = a.args_cnc(cset=True, warn=bool(not self_c))
            if nc is None:
                nc = []
            if len(xargs) > len(margs):
                continue
            resid = margs.difference(xargs)
            if len(resid) + len(xargs) == len(margs):
                if one_c:
                    co2.append(Mul(*(list(resid) + nc)))
                else:
                    co.append((resid, nc))
        if one_c:
            if co2 == []:
                return S.Zero
            elif co2:
                return Add(*co2)
        else:  # both nc
            # now check the non-comm parts
            if not co:
                return S.Zero
            if all(n == co[0][1] for r, n in co):
                ii = find(co[0][1], nx, right)
                if ii is not None:
                    if not right:
                        return Mul(Add(*[Mul(*r) for r, c in co]), Mul(*co[0][1][:ii]))
                    else:
                        return Mul(*co[0][1][ii + len(nx):])
            beg = reduce(incommon, (n[1] for n in co))
            if beg:
                ii = find(beg, nx, right)
                if ii is not None:
                    if not right:
                        gcdc = co[0][0]
                        for i in range(1, len(co)):
                            gcdc = gcdc.intersection(co[i][0])
                            if not gcdc:
                                break
                        return Mul(*(list(gcdc) + beg[:ii]))
                    else:
                        m = ii + len(nx)
                        return Add(*[Mul(*(list(r) + n[m:])) for r, n in co])
            end = list(reversed(
                reduce(incommon, (list(reversed(n[1])) for n in co))))
            if end:
                ii = find(end, nx, right)
                if ii is not None:
                    if not right:
                        return Add(*[Mul(*(list(r) + n[:-len(end) + ii])) for r, n in co])
                    else:
                        return Mul(*end[ii + len(nx):])
            # look for single match
            hit = None
            for i, (r, n) in enumerate(co):
                ii = find(n, nx, right)
                if ii is not None:
                    if not hit:
                        hit = ii, r, n
                    else:
                        break
            else:
                if hit:
                    ii, r, n = hit
                    if not right:
                        return Mul(*(list(r) + n[:ii]))
                    else:
                        return Mul(*n[ii + len(nx):])

            return S.Zero

    def as_expr(self, *gens):
        """
        Convert a polynomial to a SymPy expression.

        Examples
        ========

        >>> from sympy import sin
        >>> from sympy.abc import x, y

        >>> f = (x**2 + x*y).as_poly(x, y)
        >>> f.as_expr()
        x**2 + x*y

        >>> sin(x).as_expr()
        sin(x)

        """
        return self

    def as_coefficient(self, expr: Expr) -> Expr | None:
        """
        Extracts symbolic coefficient at the given expression. In
        other words, this functions separates 'self' into the product
        of 'expr' and 'expr'-free coefficient. If such separation
        is not possible it will return None.

        Examples
        ========

        >>> from sympy import E, pi, sin, I, Poly
        >>> from sympy.abc import x

        >>> E.as_coefficient(E)
        1
        >>> (2*E).as_coefficient(E)
        2
        >>> (2*sin(E)*E).as_coefficient(E)

        Two terms have E in them so a sum is returned. (If one were
        desiring the coefficient of the term exactly matching E then
        the constant from the returned expression could be selected.
        Or, for greater precision, a method of Poly can be used to
        indicate the desired term from which the coefficient is
        desired.)

        >>> (2*E + x*E).as_coefficient(E)
        x + 2
        >>> _.args[0]  # just want the exact match
        2
        >>> p = Poly(2*E + x*E); p
        Poly(x*E + 2*E, x, E, domain='ZZ')
        >>> p.coeff_monomial(E)
        2
        >>> p.nth(0, 1)
        2

        Since the following cannot be written as a product containing
        E as a factor, None is returned. (If the coefficient ``2*x`` is
        desired then the ``coeff`` method should be used.)

        >>> (2*E*x + x).as_coefficient(E)
        >>> (2*E*x + x).coeff(E)
        2*x

        >>> (E*(x + 1) + x).as_coefficient(E)

        >>> (2*pi*I).as_coefficient(pi*I)
        2
        >>> (2*I).as_coefficient(pi*I)

        See Also
        ========

        coeff: return sum of terms have a given factor
        as_coeff_Add: separate the additive constant from an expression
        as_coeff_Mul: separate the multiplicative constant from an expression
        as_independent: separate x-dependent terms/factors from others
        sympy.polys.polytools.Poly.coeff_monomial: efficiently find the single coefficient of a monomial in Poly
        sympy.polys.polytools.Poly.nth: like coeff_monomial but powers of monomial terms are used


        """

        r = self.extract_multiplicatively(expr)
        if r and not r.has(expr):
            return r
        else:
            return None

    def as_independent(self, *deps, **hint) -> tuple[Expr, Expr]:
        """
        A mostly naive separation of a Mul or Add into arguments that are not
        are dependent on deps. To obtain as complete a separation of variables
        as possible, use a separation method first, e.g.:

        * separatevars() to change Mul, Add and Pow (including exp) into Mul
        * .expand(mul=True) to change Add or Mul into Add
        * .expand(log=True) to change log expr into an Add

        The only non-naive thing that is done here is to respect noncommutative
        ordering of variables and to always return (0, 0) for `self` of zero
        regardless of hints.

        For nonzero `self`, the returned tuple (i, d) has the
        following interpretation:

        * i will has no variable that appears in deps
        * d will either have terms that contain variables that are in deps, or
          be equal to 0 (when self is an Add) or 1 (when self is a Mul)
        * if self is an Add then self = i + d
        * if self is a Mul then self = i*d
        * otherwise (self, S.One) or (S.One, self) is returned.

        To force the expression to be treated as an Add, use the hint as_Add=True

        Examples
        ========

        -- self is an Add

        >>> from sympy import sin, cos, exp
        >>> from sympy.abc import x, y, z

        >>> (x + x*y).as_independent(x)
        (0, x*y + x)
        >>> (x + x*y).as_independent(y)
        (x, x*y)
        >>> (2*x*sin(x) + y + x + z).as_independent(x)
        (y + z, 2*x*sin(x) + x)
        >>> (2*x*sin(x) + y + x + z).as_independent(x, y)
        (z, 2*x*sin(x) + x + y)

        -- self is a Mul

        >>> (x*sin(x)*cos(y)).as_independent(x)
        (cos(y), x*sin(x))

        non-commutative terms cannot always be separated out when self is a Mul

        >>> from sympy import symbols
        >>> n1, n2, n3 = symbols('n1 n2 n3', commutative=False)
        >>> (n1 + n1*n2).as_independent(n2)
        (n1, n1*n2)
        >>> (n2*n1 + n1*n2).as_independent(n2)
        (0, n1*n2 + n2*n1)
        >>> (n1*n2*n3).as_independent(n1)
        (1, n1*n2*n3)
        >>> (n1*n2*n3).as_independent(n2)
        (n1, n2*n3)
        >>> ((x-n1)*(x-y)).as_independent(x)
        (1, (x - y)*(x - n1))

        -- self is anything else:

        >>> (sin(x)).as_independent(x)
        (1, sin(x))
        >>> (sin(x)).as_independent(y)
        (sin(x), 1)
        >>> exp(x+y).as_independent(x)
        (1, exp(x + y))

        -- force self to be treated as an Add:

        >>> (3*x).as_independent(x, as_Add=True)
        (0, 3*x)

        -- force self to be treated as a Mul:

        >>> (3+x).as_independent(x, as_Add=False)
        (1, x + 3)
        >>> (-3+x).as_independent(x, as_Add=False)
        (1, x - 3)

        Note how the below differs from the above in making the
        constant on the dep term positive.

        >>> (y*(-3+x)).as_independent(x)
        (y, x - 3)

        -- use .as_independent() for true independence testing instead
           of .has(). The former considers only symbols in the free
           symbols while the latter considers all symbols

        >>> from sympy import Integral
        >>> I = Integral(x, (x, 1, 2))
        >>> I.has(x)
        True
        >>> x in I.free_symbols
        False
        >>> I.as_independent(x) == (I, 1)
        True
        >>> (I + x).as_independent(x) == (I, x)
        True

        Note: when trying to get independent terms, a separation method
        might need to be used first. In this case, it is important to keep
        track of what you send to this routine so you know how to interpret
        the returned values

        >>> from sympy import separatevars, log
        >>> separatevars(exp(x+y)).as_independent(x)
        (exp(y), exp(x))
        >>> (x + x*y).as_independent(y)
        (x, x*y)
        >>> separatevars(x + x*y).as_independent(y)
        (x, y + 1)
        >>> (x*(1 + y)).as_independent(y)
        (x, y + 1)
        >>> (x*(1 + y)).expand(mul=True).as_independent(y)
        (x, x*y)
        >>> a, b=symbols('a b', positive=True)
        >>> (log(a*b).expand(log=True)).as_independent(b)
        (log(a), log(b))

        See Also
        ========

        separatevars
        expand_log
        sympy.core.add.Add.as_two_terms
        sympy.core.mul.Mul.as_two_terms
        as_coeff_mul
        """
        from .symbol import Symbol
        from .add import _unevaluated_Add
        from .mul import _unevaluated_Mul

        if self is S.Zero:
            return (self, self)

        func = self.func
        want: type[Add] | type[Mul]
        if hint.get('as_Add', isinstance(self, Add) ):
            want = Add
        else:
            want = Mul

        # sift out deps into symbolic and other and ignore
        # all symbols but those that are in the free symbols
        sym = set()
        other = []
        for d in deps:
            if isinstance(d, Symbol):  # Symbol.is_Symbol is True
                sym.add(d)
            else:
                other.append(d)

        def has(e):
            """return the standard has() if there are no literal symbols, else
            check to see that symbol-deps are in the free symbols."""
            has_other = e.has(*other)
            if not sym:
                return has_other
            return has_other or e.has(*(e.free_symbols & sym))

        if (want is not func or
                func is not Add and func is not Mul):
            if has(self):
                return (want.identity, self)
            else:
                return (self, want.identity)
        else:
            if func is Add:
                args = list(self.args)
            else:
                args, nc = self.args_cnc()

        d = sift(args, has)
        depend = d[True]
        indep = d[False]
        if func is Add:  # all terms were treated as commutative
            return (Add(*indep), _unevaluated_Add(*depend))
        else:  # handle noncommutative by stopping at first dependent term
            for i, n in enumerate(nc):
                if has(n):
                    depend.extend(nc[i:])
                    break
                indep.append(n)
            return Mul(*indep), _unevaluated_Mul(*depend)

    def as_real_imag(self, deep=True, **hints) -> tuple[Expr, Expr]:
        """Performs complex expansion on 'self' and returns a tuple
           containing collected both real and imaginary parts. This
           method cannot be confused with re() and im() functions,
           which does not perform complex expansion at evaluation.

           However it is possible to expand both re() and im()
           functions and get exactly the same results as with
           a single call to this function.

           >>> from sympy import symbols, I

           >>> x, y = symbols('x,y', real=True)

           >>> (x + y*I).as_real_imag()
           (x, y)

           >>> from sympy.abc import z, w

           >>> (z + w*I).as_real_imag()
           (re(z) - im(w), re(w) + im(z))

        """
        if hints.get('ignore') == self:
            return None  # type: ignore
        else:
            from sympy.functions.elementary.complexes import im, re
            return (re(self), im(self))

    def as_powers_dict(self):
        """Return self as a dictionary of factors with each factor being
        treated as a power. The keys are the bases of the factors and the
        values, the corresponding exponents. The resulting dictionary should
        be used with caution if the expression is a Mul and contains non-
        commutative factors since the order that they appeared will be lost in
        the dictionary.

        See Also
        ========
        as_ordered_factors: An alternative for noncommutative applications,
                            returning an ordered list of factors.
        args_cnc: Similar to as_ordered_factors, but guarantees separation
                  of commutative and noncommutative factors.
        """
        d = defaultdict(int)
        d.update([self.as_base_exp()])
        return d

    def as_coefficients_dict(self, *syms):
        """Return a dictionary mapping terms to their Rational coefficient.
        Since the dictionary is a defaultdict, inquiries about terms which
        were not present will return a coefficient of 0.

        If symbols ``syms`` are provided, any multiplicative terms
        independent of them will be considered a coefficient and a
        regular dictionary of syms-dependent generators as keys and
        their corresponding coefficients as values will be returned.

        Examples
        ========

        >>> from sympy.abc import a, x, y
        >>> (3*x + a*x + 4).as_coefficients_dict()
        {1: 4, x: 3, a*x: 1}
        >>> _[a]
        0
        >>> (3*a*x).as_coefficients_dict()
        {a*x: 3}
        >>> (3*a*x).as_coefficients_dict(x)
        {x: 3*a}
        >>> (3*a*x).as_coefficients_dict(y)
        {1: 3*a*x}

        """
        d = defaultdict(list)
        if not syms:
            for ai in Add.make_args(self):
                c, m = ai.as_coeff_Mul()
                d[m].append(c)
            for k, v in d.items():
                if len(v) == 1:
                    d[k] = v[0]
                else:
                    d[k] = Add(*v)
        else:
            ind, dep = self.as_independent(*syms, as_Add=True)
            for i in Add.make_args(dep):
                if i.is_Mul:
                    c, x = i.as_coeff_mul(*syms)
                    if c is S.One:
                        d[i].append(c)
                    else:
                        d[i._new_rawargs(*x)].append(c)
                elif i:
                    d[i].append(S.One)
            d = {k: Add(*d[k]) for k in d}
            if ind is not S.Zero:
                d.update({S.One: ind})
        di = defaultdict(int)
        di.update(d)
        return di

    def as_base_exp(self) -> tuple[Expr, Expr]:
        # a -> b ** e
        return self, S.One

    def as_coeff_mul(self, *deps, **kwargs) -> tuple[Expr, tuple[Expr, ...]]:
        """Return the tuple (c, args) where self is written as a Mul, ``m``.

        c should be a Rational multiplied by any factors of the Mul that are
        independent of deps.

        args should be a tuple of all other factors of m; args is empty
        if self is a Number or if self is independent of deps (when given).

        This should be used when you do not know if self is a Mul or not but
        you want to treat self as a Mul or if you want to process the
        individual arguments of the tail of self as a Mul.

        - if you know self is a Mul and want only the head, use self.args[0];
        - if you do not want to process the arguments of the tail but need the
          tail then use self.as_two_terms() which gives the head and tail;
        - if you want to split self into an independent and dependent parts
          use ``self.as_independent(*deps)``

        >>> from sympy import S
        >>> from sympy.abc import x, y
        >>> (S(3)).as_coeff_mul()
        (3, ())
        >>> (3*x*y).as_coeff_mul()
        (3, (x, y))
        >>> (3*x*y).as_coeff_mul(x)
        (3*y, (x,))
        >>> (3*y).as_coeff_mul(x)
        (3*y, ())
        """
        if deps:
            if not self.has(*deps):
                return self, ()
        return S.One, (self,)

    def as_coeff_add(self, *deps) -> tuple[Expr, tuple[Expr, ...]]:
        """Return the tuple (c, args) where self is written as an Add, ``a``.

        c should be a Rational added to any terms of the Add that are
        independent of deps.

        args should be a tuple of all other terms of ``a``; args is empty
        if self is a Number or if self is independent of deps (when given).

        This should be used when you do not know if self is an Add or not but
        you want to treat self as an Add or if you want to process the
        individual arguments of the tail of self as an Add.

        - if you know self is an Add and want only the head, use self.args[0];
        - if you do not want to process the arguments of the tail but need the
          tail then use self.as_two_terms() which gives the head and tail.
        - if you want to split self into an independent and dependent parts
          use ``self.as_independent(*deps)``

        >>> from sympy import S
        >>> from sympy.abc import x, y
        >>> (S(3)).as_coeff_add()
        (3, ())
        >>> (3 + x).as_coeff_add()
        (3, (x,))
        >>> (3 + x + y).as_coeff_add(x)
        (y + 3, (x,))
        >>> (3 + y).as_coeff_add(x)
        (y + 3, ())

        """
        if deps:
            if not self.has_free(*deps):
                return self, ()
        return S.Zero, (self,)

    def primitive(self) -> tuple[Number, Expr]:
        """Return the positive Rational that can be extracted non-recursively
        from every term of self (i.e., self is treated like an Add). This is
        like the as_coeff_Mul() method but primitive always extracts a positive
        Rational (never a negative or a Float).

        Examples
        ========

        >>> from sympy.abc import x
        >>> (3*(x + 1)**2).primitive()
        (3, (x + 1)**2)
        >>> a = (6*x + 2); a.primitive()
        (2, 3*x + 1)
        >>> b = (x/2 + 3); b.primitive()
        (1/2, x + 6)
        >>> (a*b).primitive() == (1, a*b)
        True
        """
        if not self:
            return S.One, S.Zero
        c, r = self.as_coeff_Mul(rational=True)
        if c.is_negative:
            c, r = -c, -r
        return c, r

    def as_content_primitive(self, radical=False, clear=True):
        """This method should recursively remove a Rational from all arguments
        and return that (content) and the new self (primitive). The content
        should always be positive and ``Mul(*foo.as_content_primitive()) == foo``.
        The primitive need not be in canonical form and should try to preserve
        the underlying structure if possible (i.e. expand_mul should not be
        applied to self).

        Examples
        ========

        >>> from sympy import sqrt
        >>> from sympy.abc import x, y, z

        >>> eq = 2 + 2*x + 2*y*(3 + 3*y)

        The as_content_primitive function is recursive and retains structure:

        >>> eq.as_content_primitive()
        (2, x + 3*y*(y + 1) + 1)

        Integer powers will have Rationals extracted from the base:

        >>> ((2 + 6*x)**2).as_content_primitive()
        (4, (3*x + 1)**2)
        >>> ((2 + 6*x)**(2*y)).as_content_primitive()
        (1, (2*(3*x + 1))**(2*y))

        Terms may end up joining once their as_content_primitives are added:

        >>> ((5*(x*(1 + y)) + 2*x*(3 + 3*y))).as_content_primitive()
        (11, x*(y + 1))
        >>> ((3*(x*(1 + y)) + 2*x*(3 + 3*y))).as_content_primitive()
        (9, x*(y + 1))
        >>> ((3*(z*(1 + y)) + 2.0*x*(3 + 3*y))).as_content_primitive()
        (1, 6.0*x*(y + 1) + 3*z*(y + 1))
        >>> ((5*(x*(1 + y)) + 2*x*(3 + 3*y))**2).as_content_primitive()
        (121, x**2*(y + 1)**2)
        >>> ((x*(1 + y) + 0.4*x*(3 + 3*y))**2).as_content_primitive()
        (1, 4.84*x**2*(y + 1)**2)

        Radical content can also be factored out of the primitive:

        >>> (2*sqrt(2) + 4*sqrt(10)).as_content_primitive(radical=True)
        (2, sqrt(2)*(1 + 2*sqrt(5)))

        If clear=False (default is True) then content will not be removed
        from an Add if it can be distributed to leave one or more
        terms with integer coefficients.

        >>> (x/2 + y).as_content_primitive()
        (1/2, x + 2*y)
        >>> (x/2 + y).as_content_primitive(clear=False)
        (1, x/2 + y)
        """
        return S.One, self

    def as_numer_denom(self) -> tuple[Expr, Expr]:
        """Return the numerator and the denominator of an expression.

        expression -> a/b -> a, b

        This is just a stub that should be defined by
        an object's class methods to get anything else.

        See Also
        ========

        normal: return ``a/b`` instead of ``(a, b)``

        """
        return self, S.One

    def normal(self):
        """Return the expression as a fraction.

        expression -> a/b

        See Also
        ========

        as_numer_denom: return ``(a, b)`` instead of ``a/b``

        """
        from .mul import _unevaluated_Mul
        n, d = self.as_numer_denom()
        if d is S.One:
            return n
        if d.is_Number:
            return _unevaluated_Mul(n, 1/d)
        else:
            return n/d

    def extract_multiplicatively(self, c: Expr) -> Expr | None:
        """Return None if it's not possible to make self in the form
           c * something in a nice way, i.e. preserving the properties
           of arguments of self.

           Examples
           ========

           >>> from sympy import symbols, Rational

           >>> x, y = symbols('x,y', real=True)

           >>> ((x*y)**3).extract_multiplicatively(x**2 * y)
           x*y**2

           >>> ((x*y)**3).extract_multiplicatively(x**4 * y)

           >>> (2*x).extract_multiplicatively(2)
           x

           >>> (2*x).extract_multiplicatively(3)

           >>> (Rational(1, 2)*x).extract_multiplicatively(3)
           x/6

        """
        from sympy.functions.elementary.exponential import exp
        from .add import _unevaluated_Add
        c = sympify(c)
        if self is S.NaN:
            return None
        if c is S.One:
            return self
        elif c == self:
            return S.One

        if c.is_Add:
            cc, pc = c.primitive()
            if cc is not S.One:
                c = Mul(cc, pc, evaluate=False)

        if c.is_Mul:
            a, b = c.as_two_terms() # type: ignore
            x = self.extract_multiplicatively(a)
            if x is not None:
                return x.extract_multiplicatively(b)
            else:
                return x

        quotient = self / c
        if self.is_Number:
            if self is S.Infinity:
                if c.is_positive:
                    return S.Infinity
            elif self is S.NegativeInfinity:
                if c.is_negative:
                    return S.Infinity
                elif c.is_positive:
                    return S.NegativeInfinity
            elif self is S.ComplexInfinity:
                if not c.is_zero:
                    return S.ComplexInfinity
            elif self.is_Integer:
                if not quotient.is_Integer:
                    return None
                elif self.is_positive and quotient.is_negative:
                    return None
                else:
                    return quotient
            elif self.is_Rational:
                if not quotient.is_Rational:
                    return None
                elif self.is_positive and quotient.is_negative:
                    return None
                else:
                    return quotient
            elif self.is_Float:
                if not quotient.is_Float:
                    return None
                elif self.is_positive and quotient.is_negative:
                    return None
                else:
                    return quotient
        elif self.is_NumberSymbol or self.is_Symbol or self is S.ImaginaryUnit:
            if quotient.is_Mul and len(quotient.args) == 2:
                if quotient.args[0].is_Integer and quotient.args[0].is_positive and quotient.args[1] == self:
                    return quotient
            elif quotient.is_Integer and c.is_Number:
                return quotient
        elif self.is_Add:
            cs, ps = self.primitive()
            # assert cs >= 1
            if c.is_Number and c is not S.NegativeOne:
                # assert c != 1 (handled at top)
                if cs is not S.One:
                    if c.is_negative:
                        xc = cs.extract_multiplicatively(-c)
                        if xc is not None:
                            xc = -xc
                    else:
                        xc = cs.extract_multiplicatively(c)
                    if xc is not None:
                        return xc*ps  # rely on 2-arg Mul to restore Add
                return None # |c| != 1 can only be extracted from cs
            if c == ps:
                return cs
            # check args of ps
            newargs = []
            arg: Expr
            for arg in ps.args: # type: ignore
                newarg = arg.extract_multiplicatively(c)
                if newarg is None:
                    return None # all or nothing
                newargs.append(newarg)
            if cs is not S.One:
                args = [cs*t for t in newargs]
                # args may be in different order
                return _unevaluated_Add(*args)
            else:
                return Add._from_args(newargs)
        elif self.is_Mul:
            args: list[Expr] = list(self.args) # type: ignore
            for i, arg in enumerate(args):
                newarg = arg.extract_multiplicatively(c)
                if newarg is not None:
                    args[i] = newarg
                    return Mul(*args)
        elif self.is_Pow or isinstance(self, exp):
            sb, se = self.as_base_exp()
            cb, ce = c.as_base_exp()
            if cb == sb:
                new_exp = se.extract_additively(ce)
                if new_exp is not None:
                    return Pow(sb, new_exp)
            elif c == sb:
                new_exp = se.extract_additively(1)
                if new_exp is not None:
                    return Pow(sb, new_exp)

        return None

    def extract_additively(self, c):
        """Return self - c if it's possible to subtract c from self and
        make all matching coefficients move towards zero, else return None.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> e = 2*x + 3
        >>> e.extract_additively(x + 1)
        x + 2
        >>> e.extract_additively(3*x)
        >>> e.extract_additively(4)
        >>> (y*(x + 1)).extract_additively(x + 1)
        >>> ((x + 1)*(x + 2*y + 1) + 3).extract_additively(x + 1)
        (x + 1)*(x + 2*y) + 3

        See Also
        ========
        extract_multiplicatively
        coeff
        as_coefficient

        """

        c = sympify(c)
        if self is S.NaN:
            return None
        if c.is_zero:
            return self
        elif c == self:
            return S.Zero
        elif self == S.Zero:
            return None

        if self.is_Number:
            if not c.is_Number:
                return None
            co = self
            diff = co - c
            # XXX should we match types? i.e should 3 - .1 succeed?
            if (co > 0 and diff >= 0 and diff < co or
                    co < 0 and diff <= 0 and diff > co):
                return diff
            return None

        if c.is_Number:
            co, t = self.as_coeff_Add()
            xa = co.extract_additively(c)
            if xa is None:
                return None
            return xa + t

        # handle the args[0].is_Number case separately
        # since we will have trouble looking for the coeff of
        # a number.
        if c.is_Add and c.args[0].is_Number:
            # whole term as a term factor
            co = self.coeff(c)
            xa0 = (co.extract_additively(1) or 0)*c
            if xa0:
                diff = self - co*c
                return (xa0 + (diff.extract_additively(c) or diff)) or None
            # term-wise
            h, t = c.as_coeff_Add()
            sh, st = self.as_coeff_Add()
            xa = sh.extract_additively(h)
            if xa is None:
                return None
            xa2 = st.extract_additively(t)
            if xa2 is None:
                return None
            return xa + xa2

        # whole term as a term factor
        co, diff = _corem(self, c)
        xa0 = (co.extract_additively(1) or 0)*c
        if xa0:
            return (xa0 + (diff.extract_additively(c) or diff)) or None
        # term-wise
        coeffs = []
        for a in Add.make_args(c):
            ac, at = a.as_coeff_Mul()
            co = self.coeff(at)
            if not co:
                return None
            coc, cot = co.as_coeff_Add()
            xa = coc.extract_additively(ac)
            if xa is None:
                return None
            self -= co*at
            coeffs.append((cot + xa)*at)
        coeffs.append(self)
        return Add(*coeffs)

    @property
    def expr_free_symbols(self):
        """
        Like ``free_symbols``, but returns the free symbols only if
        they are contained in an expression node.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> (x + y).expr_free_symbols # doctest: +SKIP
        {x, y}

        If the expression is contained in a non-expression object, do not return
        the free symbols. Compare:

        >>> from sympy import Tuple
        >>> t = Tuple(x + y)
        >>> t.expr_free_symbols # doctest: +SKIP
        set()
        >>> t.free_symbols
        {x, y}
        """
        sympy_deprecation_warning("""
        The expr_free_symbols property is deprecated. Use free_symbols to get
        the free symbols of an expression.
        """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-expr-free-symbols")
        return {j for i in self.args for j in i.expr_free_symbols}

    def could_extract_minus_sign(self) -> bool:
        """Return True if self has -1 as a leading factor or has
        more literal negative signs than positive signs in a sum,
        otherwise False.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> e = x - y
        >>> {i.could_extract_minus_sign() for i in (e, -e)}
        {False, True}

        Though the ``y - x`` is considered like ``-(x - y)``, since it
        is in a product without a leading factor of -1, the result is
        false below:

        >>> (x*(y - x)).could_extract_minus_sign()
        False

        To put something in canonical form wrt to sign, use `signsimp`:

        >>> from sympy import signsimp
        >>> signsimp(x*(y - x))
        -x*(x - y)
        >>> _.could_extract_minus_sign()
        True
        """
        return False

    def extract_branch_factor(self, allow_half=False):
        """
        Try to write self as ``exp_polar(2*pi*I*n)*z`` in a nice way.
        Return (z, n).

        >>> from sympy import exp_polar, I, pi
        >>> from sympy.abc import x, y
        >>> exp_polar(I*pi).extract_branch_factor()
        (exp_polar(I*pi), 0)
        >>> exp_polar(2*I*pi).extract_branch_factor()
        (1, 1)
        >>> exp_polar(-pi*I).extract_branch_factor()
        (exp_polar(I*pi), -1)
        >>> exp_polar(3*pi*I + x).extract_branch_factor()
        (exp_polar(x + I*pi), 1)
        >>> (y*exp_polar(-5*pi*I)*exp_polar(3*pi*I + 2*pi*x)).extract_branch_factor()
        (y*exp_polar(2*pi*x), -1)
        >>> exp_polar(-I*pi/2).extract_branch_factor()
        (exp_polar(-I*pi/2), 0)

        If allow_half is True, also extract exp_polar(I*pi):

        >>> exp_polar(I*pi).extract_branch_factor(allow_half=True)
        (1, 1/2)
        >>> exp_polar(2*I*pi).extract_branch_factor(allow_half=True)
        (1, 1)
        >>> exp_polar(3*I*pi).extract_branch_factor(allow_half=True)
        (1, 3/2)
        >>> exp_polar(-I*pi).extract_branch_factor(allow_half=True)
        (1, -1/2)
        """
        from sympy.functions.elementary.exponential import exp_polar
        from sympy.functions.elementary.integers import ceiling

        n = S.Zero
        res = S.One
        args = Mul.make_args(self)
        exps = []
        for arg in args:
            if isinstance(arg, exp_polar):
                exps += [arg.exp]
            else:
                res *= arg
        piimult = S.Zero
        extras = []

        ipi = S.Pi*S.ImaginaryUnit
        while exps:
            exp = exps.pop()
            if exp.is_Add:
                exps += exp.args
                continue
            if exp.is_Mul:
                coeff = exp.as_coefficient(ipi)
                if coeff is not None:
                    piimult += coeff
                    continue
            extras += [exp]
        if piimult.is_number:
            coeff = piimult
            tail = ()
        else:
            coeff, tail = piimult.as_coeff_add(*piimult.free_symbols)
        # round down to nearest multiple of 2
        branchfact = ceiling(coeff/2 - S.Half)*2
        n += branchfact/2
        c = coeff - branchfact
        if allow_half:
            nc = c.extract_additively(1)
            if nc is not None:
                n += S.Half
                c = nc
        newexp = ipi*Add(*((c, ) + tail)) + Add(*extras)
        if newexp != 0:
            res *= exp_polar(newexp)
        return res, n

    def is_polynomial(self, *syms):
        r"""
        Return True if self is a polynomial in syms and False otherwise.

        This checks if self is an exact polynomial in syms.  This function
        returns False for expressions that are "polynomials" with symbolic
        exponents.  Thus, you should be able to apply polynomial algorithms to
        expressions for which this returns True, and Poly(expr, \*syms) should
        work if and only if expr.is_polynomial(\*syms) returns True. The
        polynomial does not have to be in expanded form.  If no symbols are
        given, all free symbols in the expression will be used.

        This is not part of the assumptions system.  You cannot do
        Symbol('z', polynomial=True).

        Examples
        ========

        >>> from sympy import Symbol, Function
        >>> x = Symbol('x')
        >>> ((x**2 + 1)**4).is_polynomial(x)
        True
        >>> ((x**2 + 1)**4).is_polynomial()
        True
        >>> (2**x + 1).is_polynomial(x)
        False
        >>> (2**x + 1).is_polynomial(2**x)
        True
        >>> f = Function('f')
        >>> (f(x) + 1).is_polynomial(x)
        False
        >>> (f(x) + 1).is_polynomial(f(x))
        True
        >>> (1/f(x) + 1).is_polynomial(f(x))
        False

        >>> n = Symbol('n', nonnegative=True, integer=True)
        >>> (x**n + 1).is_polynomial(x)
        False

        This function does not attempt any nontrivial simplifications that may
        result in an expression that does not appear to be a polynomial to
        become one.

        >>> from sympy import sqrt, factor, cancel
        >>> y = Symbol('y', positive=True)
        >>> a = sqrt(y**2 + 2*y + 1)
        >>> a.is_polynomial(y)
        False
        >>> factor(a)
        y + 1
        >>> factor(a).is_polynomial(y)
        True

        >>> b = (y**2 + 2*y + 1)/(y + 1)
        >>> b.is_polynomial(y)
        False
        >>> cancel(b)
        y + 1
        >>> cancel(b).is_polynomial(y)
        True

        See also .is_rational_function()

        """
        if syms:
            syms = set(map(sympify, syms))
        else:
            syms = self.free_symbols
            if not syms:
                return True

        return self._eval_is_polynomial(syms)

    def _eval_is_polynomial(self, syms) -> bool | None:
        if self in syms:
            return True
        if not self.has_free(*syms):
            # constant polynomial
            return True
        # subclasses should return True or False
        return None

    def is_rational_function(self, *syms):
        """
        Test whether function is a ratio of two polynomials in the given
        symbols, syms. When syms is not given, all free symbols will be used.
        The rational function does not have to be in expanded or in any kind of
        canonical form.

        This function returns False for expressions that are "rational
        functions" with symbolic exponents.  Thus, you should be able to call
        .as_numer_denom() and apply polynomial algorithms to the result for
        expressions for which this returns True.

        This is not part of the assumptions system.  You cannot do
        Symbol('z', rational_function=True).

        Examples
        ========

        >>> from sympy import Symbol, sin
        >>> from sympy.abc import x, y

        >>> (x/y).is_rational_function()
        True

        >>> (x**2).is_rational_function()
        True

        >>> (x/sin(y)).is_rational_function(y)
        False

        >>> n = Symbol('n', integer=True)
        >>> (x**n + 1).is_rational_function(x)
        False

        This function does not attempt any nontrivial simplifications that may
        result in an expression that does not appear to be a rational function
        to become one.

        >>> from sympy import sqrt, factor
        >>> y = Symbol('y', positive=True)
        >>> a = sqrt(y**2 + 2*y + 1)/y
        >>> a.is_rational_function(y)
        False
        >>> factor(a)
        (y + 1)/y
        >>> factor(a).is_rational_function(y)
        True

        See also is_algebraic_expr().

        """
        if syms:
            syms = set(map(sympify, syms))
        else:
            syms = self.free_symbols
            if not syms:
                return self not in _illegal

        return self._eval_is_rational_function(syms)

    def _eval_is_rational_function(self, syms) -> bool | None:
        if self in syms:
            return True
        if not self.has_xfree(syms):
            return True
        # subclasses should return True or False
        return None

    def is_meromorphic(self, x, a):
        """
        This tests whether an expression is meromorphic as
        a function of the given symbol ``x`` at the point ``a``.

        This method is intended as a quick test that will return
        None if no decision can be made without simplification or
        more detailed analysis.

        Examples
        ========

        >>> from sympy import zoo, log, sin, sqrt
        >>> from sympy.abc import x

        >>> f = 1/x**2 + 1 - 2*x**3
        >>> f.is_meromorphic(x, 0)
        True
        >>> f.is_meromorphic(x, 1)
        True
        >>> f.is_meromorphic(x, zoo)
        True

        >>> g = x**log(3)
        >>> g.is_meromorphic(x, 0)
        False
        >>> g.is_meromorphic(x, 1)
        True
        >>> g.is_meromorphic(x, zoo)
        False

        >>> h = sin(1/x)*x**2
        >>> h.is_meromorphic(x, 0)
        False
        >>> h.is_meromorphic(x, 1)
        True
        >>> h.is_meromorphic(x, zoo)
        True

        Multivalued functions are considered meromorphic when their
        branches are meromorphic. Thus most functions are meromorphic
        everywhere except at essential singularities and branch points.
        In particular, they will be meromorphic also on branch cuts
        except at their endpoints.

        >>> log(x).is_meromorphic(x, -1)
        True
        >>> log(x).is_meromorphic(x, 0)
        False
        >>> sqrt(x).is_meromorphic(x, -1)
        True
        >>> sqrt(x).is_meromorphic(x, 0)
        False

        """
        if not x.is_symbol:
            raise TypeError("{} should be of symbol type".format(x))
        a = sympify(a)

        return self._eval_is_meromorphic(x, a)

    def _eval_is_meromorphic(self, x, a) -> bool | None:
        if self == x:
            return True
        if not self.has_free(x):
            return True
        # subclasses should return True or False
        return None

    def is_algebraic_expr(self, *syms):
        """
        This tests whether a given expression is algebraic or not, in the
        given symbols, syms. When syms is not given, all free symbols
        will be used. The rational function does not have to be in expanded
        or in any kind of canonical form.

        This function returns False for expressions that are "algebraic
        expressions" with symbolic exponents. This is a simple extension to the
        is_rational_function, including rational exponentiation.

        Examples
        ========

        >>> from sympy import Symbol, sqrt
        >>> x = Symbol('x', real=True)
        >>> sqrt(1 + x).is_rational_function()
        False
        >>> sqrt(1 + x).is_algebraic_expr()
        True

        This function does not attempt any nontrivial simplifications that may
        result in an expression that does not appear to be an algebraic
        expression to become one.

        >>> from sympy import exp, factor
        >>> a = sqrt(exp(x)**2 + 2*exp(x) + 1)/(exp(x) + 1)
        >>> a.is_algebraic_expr(x)
        False
        >>> factor(a).is_algebraic_expr()
        True

        See Also
        ========

        is_rational_function

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Algebraic_expression

        """
        if syms:
            syms = set(map(sympify, syms))
        else:
            syms = self.free_symbols
            if not syms:
                return True

        return self._eval_is_algebraic_expr(syms)

    def _eval_is_algebraic_expr(self, syms) -> bool | None:
        if self in syms:
            return True
        if not self.has_free(*syms):
            return True
        # subclasses should return True or False
        return None

    ###################################################################################
    ##################### SERIES, LEADING TERM, LIMIT, ORDER METHODS ##################
    ###################################################################################

    def series(self, x=None, x0=0, n=6, dir="+", logx=None, cdir=0):
        """
        Series expansion of "self" around ``x = x0`` yielding either terms of
        the series one by one (the lazy series given when n=None), else
        all the terms at once when n != None.

        Returns the series expansion of "self" around the point ``x = x0``
        with respect to ``x`` up to ``O((x - x0)**n, x, x0)`` (default n is 6).

        If ``x=None`` and ``self`` is univariate, the univariate symbol will
        be supplied, otherwise an error will be raised.

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
            The value used to represent the order in terms of ``x**n``,
            up to which the series is to be expanded.

        dir : String, optional
              The series-expansion can be bi-directional. If ``dir="+"``,
              then (x->x0+). If ``dir="-", then (x->x0-). For infinite
              ``x0`` (``oo`` or ``-oo``), the ``dir`` argument is determined
              from the direction of the infinity (i.e., ``dir="-"`` for
              ``oo``).

        logx : optional
               It is used to replace any log(x) in the returned series with a
               symbolic value rather than evaluating the actual value.

        cdir : optional
               It stands for complex direction, and indicates the direction
               from which the expansion needs to be evaluated.

        Examples
        ========

        >>> from sympy import cos, exp, tan
        >>> from sympy.abc import x, y
        >>> cos(x).series()
        1 - x**2/2 + x**4/24 + O(x**6)
        >>> cos(x).series(n=4)
        1 - x**2/2 + O(x**4)
        >>> cos(x).series(x, x0=1, n=2)
        cos(1) - (x - 1)*sin(1) + O((x - 1)**2, (x, 1))
        >>> e = cos(x + exp(y))
        >>> e.series(y, n=2)
        cos(x + 1) - y*sin(x + 1) + O(y**2)
        >>> e.series(x, n=2)
        cos(exp(y)) - x*sin(exp(y)) + O(x**2)

        If ``n=None`` then a generator of the series terms will be returned.

        >>> term=cos(x).series(n=None)
        >>> [next(term) for i in range(2)]
        [1, -x**2/2]

        For ``dir=+`` (default) the series is calculated from the right and
        for ``dir=-`` the series from the left. For smooth functions this
        flag will not alter the results.

        >>> abs(x).series(dir="+")
        x
        >>> abs(x).series(dir="-")
        -x
        >>> f = tan(x)
        >>> f.series(x, 2, 6, "+")
        tan(2) + (1 + tan(2)**2)*(x - 2) + (x - 2)**2*(tan(2)**3 + tan(2)) +
        (x - 2)**3*(1/3 + 4*tan(2)**2/3 + tan(2)**4) + (x - 2)**4*(tan(2)**5 +
        5*tan(2)**3/3 + 2*tan(2)/3) + (x - 2)**5*(2/15 + 17*tan(2)**2/15 +
        2*tan(2)**4 + tan(2)**6) + O((x - 2)**6, (x, 2))

        >>> f.series(x, 2, 3, "-")
        tan(2) + (2 - x)*(-tan(2)**2 - 1) + (2 - x)**2*(tan(2)**3 + tan(2))
        + O((x - 2)**3, (x, 2))

        For rational expressions this method may return original expression without the Order term.
        >>> (1/x).series(x, n=8)
        1/x

        Returns
        =======

        Expr : Expression
            Series expansion of the expression about x0

        Raises
        ======

        TypeError
            If "n" and "x0" are infinity objects

        PoleError
            If "x0" is an infinity object

        """
        if x is None:
            syms = self.free_symbols
            if not syms:
                return self
            elif len(syms) > 1:
                raise ValueError('x must be given for multivariate functions.')
            x = syms.pop()

        from .symbol import Dummy, Symbol
        if isinstance(x, Symbol):
            dep = x in self.free_symbols
        else:
            d = Dummy()
            dep = d in self.xreplace({x: d}).free_symbols
        if not dep:
            if n is None:
                return (s for s in [self])
            else:
                return self

        if len(dir) != 1 or dir not in '+-':
            raise ValueError("Dir must be '+' or '-'")

        if n is not None:
            n = int(n)
            if n < 0:
                raise ValueError("Number of terms should be nonnegative")

        x0 = sympify(x0)
        cdir = sympify(cdir)
        from sympy.functions.elementary.complexes import im, sign

        if not cdir.is_zero:
            if cdir.is_real:
                dir = '+' if cdir.is_positive else '-'
            else:
                dir = '+' if im(cdir).is_positive else '-'
        else:
            if x0 and x0.is_infinite:
                cdir = sign(x0).simplify()
            elif str(dir) == "+":
                cdir = S.One
            elif str(dir) == "-":
                cdir = S.NegativeOne
            elif cdir == S.Zero:
                cdir = S.One

        cdir = cdir/abs(cdir)

        if x0 and x0.is_infinite:
            from .function import PoleError
            try:
                s = self.subs(x, cdir/x).series(x, n=n, dir='+', cdir=1)
                if n is None:
                    return (si.subs(x, cdir/x) for si in s)
                return s.subs(x, cdir/x)
            except PoleError:
                s = self.subs(x, cdir*x).aseries(x, n=n)
                return s.subs(x, cdir*x)

        # use rep to shift origin to x0 and change sign (if dir is negative)
        # and undo the process with rep2
        if x0 or cdir != 1:
            s = self.subs({x: x0 + cdir*x}).series(x, x0=0, n=n, dir='+', logx=logx, cdir=1)
            if n is None:  # lseries...
                return (si.subs({x: x/cdir - x0/cdir}) for si in s)
            return s.subs({x: x/cdir - x0/cdir})

        # from here on it's x0=0 and dir='+' handling

        if x.is_positive is x.is_negative is None or x.is_Symbol is not True:
            # replace x with an x that has a positive assumption
            xpos = Dummy('x', positive=True)
            rv = self.subs(x, xpos).series(xpos, x0, n, dir, logx=logx, cdir=cdir)
            if n is None:
                return (s.subs(xpos, x) for s in rv)
            else:
                return rv.subs(xpos, x)

        from sympy.series.order import Order
        if n is not None:  # nseries handling
            s1 = self._eval_nseries(x, n=n, logx=logx, cdir=cdir)
            o = s1.getO() or S.Zero
            if o:
                # make sure the requested order is returned
                ngot = o.getn()
                if ngot > n:
                    # leave o in its current form (e.g. with x*log(x)) so
                    # it eats terms properly, then replace it below
                    if n != 0:
                        s1 += o.subs(x, x**Rational(n, ngot))
                    else:
                        s1 += Order(1, x)
                elif ngot < n:
                    # increase the requested number of terms to get the desired
                    # number keep increasing (up to 9) until the received order
                    # is different than the original order and then predict how
                    # many additional terms are needed
                    from sympy.functions.elementary.integers import ceiling
                    for more in range(1, 9):
                        s1 = self._eval_nseries(x, n=n + more, logx=logx, cdir=cdir)
                        newn = s1.getn()
                        if newn != ngot:
                            ndo = n + ceiling((n - ngot)*more/(newn - ngot))
                            s1 = self._eval_nseries(x, n=ndo, logx=logx, cdir=cdir)
                            while s1.getn() < n:
                                s1 = self._eval_nseries(x, n=ndo, logx=logx, cdir=cdir)
                                ndo += 1
                            break
                    else:
                        raise ValueError('Could not calculate %s terms for %s'
                                         % (str(n), self))
                    s1 += Order(x**n, x)
                o = s1.getO()
                s1 = s1.removeO()
            elif s1.has(Order):
                # asymptotic expansion
                return s1
            else:
                o = Order(x**n, x)
                s1done = s1.doit()
                try:
                    if (s1done + o).removeO() == s1done:
                        o = S.Zero
                except NotImplementedError:
                    return s1

            try:
                from sympy.simplify.radsimp import collect
                return collect(s1, x) + o
            except NotImplementedError:
                return s1 + o

        else:  # lseries handling
            def yield_lseries(s):
                """Return terms of lseries one at a time."""
                for si in s:
                    if not si.is_Add:
                        yield si
                        continue
                    # yield terms 1 at a time if possible
                    # by increasing order until all the
                    # terms have been returned
                    yielded = 0
                    o = Order(si, x)*x
                    ndid = 0
                    ndo = len(si.args)
                    while 1:
                        do = (si - yielded + o).removeO()
                        o *= x
                        if not do or do.is_Order:
                            continue
                        if do.is_Add:
                            ndid += len(do.args)
                        else:
                            ndid += 1
                        yield do
                        if ndid == ndo:
                            break
                        yielded += do

            return yield_lseries(self.removeO()._eval_lseries(x, logx=logx, cdir=cdir))

    def aseries(self, x=None, n=6, bound=0, hir=False):
        """Asymptotic Series expansion of self.
        This is equivalent to ``self.series(x, oo, n)``.

        Parameters
        ==========

        self : Expression
               The expression whose series is to be expanded.

        x : Symbol
            It is the variable of the expression to be calculated.

        n : Value
            The value used to represent the order in terms of ``x**n``,
            up to which the series is to be expanded.

        hir : Boolean
              Set this parameter to be True to produce hierarchical series.
              It stops the recursion at an early level and may provide nicer
              and more useful results.

        bound : Value, Integer
                Use the ``bound`` parameter to give limit on rewriting
                coefficients in its normalised form.

        Examples
        ========

        >>> from sympy import sin, exp
        >>> from sympy.abc import x

        >>> e = sin(1/x + exp(-x)) - sin(1/x)

        >>> e.aseries(x)
        (1/(24*x**4) - 1/(2*x**2) + 1 + O(x**(-6), (x, oo)))*exp(-x)

        >>> e.aseries(x, n=3, hir=True)
        -exp(-2*x)*sin(1/x)/2 + exp(-x)*cos(1/x) + O(exp(-3*x), (x, oo))

        >>> e = exp(exp(x)/(1 - 1/x))

        >>> e.aseries(x)
        exp(exp(x)/(1 - 1/x))

        >>> e.aseries(x, bound=3) # doctest: +SKIP
        exp(exp(x)/x**2)*exp(exp(x)/x)*exp(-exp(x) + exp(x)/(1 - 1/x) - exp(x)/x - exp(x)/x**2)*exp(exp(x))

        For rational expressions this method may return original expression without the Order term.
        >>> (1/x).aseries(x, n=8)
        1/x

        Returns
        =======

        Expr
            Asymptotic series expansion of the expression.

        Notes
        =====

        This algorithm is directly induced from the limit computational algorithm provided by Gruntz.
        It majorly uses the mrv and rewrite sub-routines. The overall idea of this algorithm is first
        to look for the most rapidly varying subexpression w of a given expression f and then expands f
        in a series in w. Then same thing is recursively done on the leading coefficient
        till we get constant coefficients.

        If the most rapidly varying subexpression of a given expression f is f itself,
        the algorithm tries to find a normalised representation of the mrv set and rewrites f
        using this normalised representation.

        If the expansion contains an order term, it will be either ``O(x ** (-n))`` or ``O(w ** (-n))``
        where ``w`` belongs to the most rapidly varying expression of ``self``.

        References
        ==========

        .. [1] Gruntz, Dominik. A new algorithm for computing asymptotic series.
               In: Proc. 1993 Int. Symp. Symbolic and Algebraic Computation. 1993.
               pp. 239-244.
        .. [2] Gruntz thesis - p90
        .. [3] https://en.wikipedia.org/wiki/Asymptotic_expansion

        See Also
        ========

        Expr.aseries: See the docstring of this function for complete details of this wrapper.
        """

        from .symbol import Dummy

        if x.is_positive is x.is_negative is None:
            xpos = Dummy('x', positive=True)
            return self.subs(x, xpos).aseries(xpos, n, bound, hir).subs(xpos, x)

        from .function import PoleError
        from sympy.series.gruntz import mrv, rewrite

        try:
            om, exps = mrv(self, x)
        except PoleError:
            return self

        # We move one level up by replacing `x` by `exp(x)`, and then
        # computing the asymptotic series for f(exp(x)). Then asymptotic series
        # can be obtained by moving one-step back, by replacing x by ln(x).

        from sympy.functions.elementary.exponential import exp, log
        from sympy.series.order import Order

        if x in om:
            s = self.subs(x, exp(x)).aseries(x, n, bound, hir).subs(x, log(x))
            if s.getO():
                return s + Order(1/x**n, (x, S.Infinity))
            return s

        k = Dummy('k', positive=True)
        # f is rewritten in terms of omega
        func, logw = rewrite(exps, om, x, k)

        if self in om:
            if bound <= 0:
                return self
            s = (self.exp).aseries(x, n, bound=bound)
            s = s.func(*[t.removeO() for t in s.args])
            try:
                res = exp(s.subs(x, 1/x).as_leading_term(x).subs(x, 1/x))
            except PoleError:
                res = self

            func = exp(self.args[0] - res.args[0]) / k
            logw = log(1/res)

        s = func.series(k, 0, n)
        from sympy.core.function import expand_mul
        s = expand_mul(s)
        # Hierarchical series
        if hir:
            return s.subs(k, exp(logw))

        o = s.getO()
        terms = sorted(Add.make_args(s.removeO()), key=lambda i: int(i.as_coeff_exponent(k)[1]))
        s = S.Zero
        has_ord = False

        # Then we recursively expand these coefficients one by one into
        # their asymptotic series in terms of their most rapidly varying subexpressions.
        for t in terms:
            coeff, expo = t.as_coeff_exponent(k)
            if coeff.has(x):
                # Recursive step
                snew = coeff.aseries(x, n, bound=bound-1)
                if has_ord and snew.getO():
                    break
                elif snew.getO():
                    has_ord = True
                s += (snew * k**expo)
            else:
                s += t

        if not o or has_ord:
            return s.subs(k, exp(logw))
        return (s + o).subs(k, exp(logw))


    def taylor_term(self, n, x, *previous_terms):
        """General method for the taylor term.

        This method is slow, because it differentiates n-times. Subclasses can
        redefine it to make it faster by using the "previous_terms".
        """
        from .symbol import Dummy
        from sympy.functions.combinatorial.factorials import factorial

        x = sympify(x)
        _x = Dummy('x')
        return self.subs(x, _x).diff(_x, n).subs(_x, x).subs(x, 0) * x**n / factorial(n)

    def lseries(self, x=None, x0=0, dir='+', logx=None, cdir=0):
        """
        Wrapper for series yielding an iterator of the terms of the series.

        Note: an infinite series will yield an infinite iterator. The following,
        for exaxmple, will never terminate. It will just keep printing terms
        of the sin(x) series::

          for term in sin(x).lseries(x):
              print term

        The advantage of lseries() over nseries() is that many times you are
        just interested in the next term in the series (i.e. the first term for
        example), but you do not know how many you should ask for in nseries()
        using the "n" parameter.

        See also nseries().
        """
        return self.series(x, x0, n=None, dir=dir, logx=logx, cdir=cdir)

    def _eval_lseries(self, x, logx=None, cdir=0):
        # default implementation of lseries is using nseries(), and adaptively
        # increasing the "n". As you can see, it is not very efficient, because
        # we are calculating the series over and over again. Subclasses should
        # override this method and implement much more efficient yielding of
        # terms.
        n = 0
        series = self._eval_nseries(x, n=n, logx=logx, cdir=cdir)

        while series.is_Order:
            n += 1
            series = self._eval_nseries(x, n=n, logx=logx, cdir=cdir)

        e = series.removeO()
        yield e
        if e is S.Zero:
            return

        while 1:
            while 1:
                n += 1
                series = self._eval_nseries(x, n=n, logx=logx, cdir=cdir).removeO()
                if e != series:
                    break
                if (series - self).cancel() is S.Zero:
                    return
            yield series - e
            e = series

    def nseries(self, x=None, x0=0, n=6, dir='+', logx=None, cdir=0):
        """
        Wrapper to _eval_nseries if assumptions allow, else to series.

        If x is given, x0 is 0, dir='+', and self has x, then _eval_nseries is
        called. This calculates "n" terms in the innermost expressions and
        then builds up the final series just by "cross-multiplying" everything
        out.

        The optional ``logx`` parameter can be used to replace any log(x) in the
        returned series with a symbolic value to avoid evaluating log(x) at 0. A
        symbol to use in place of log(x) should be provided.

        Advantage -- it's fast, because we do not have to determine how many
        terms we need to calculate in advance.

        Disadvantage -- you may end up with less terms than you may have
        expected, but the O(x**n) term appended will always be correct and
        so the result, though perhaps shorter, will also be correct.

        If any of those assumptions is not met, this is treated like a
        wrapper to series which will try harder to return the correct
        number of terms.

        See also lseries().

        Examples
        ========

        >>> from sympy import sin, log, Symbol
        >>> from sympy.abc import x, y
        >>> sin(x).nseries(x, 0, 6)
        x - x**3/6 + x**5/120 + O(x**6)
        >>> log(x+1).nseries(x, 0, 5)
        x - x**2/2 + x**3/3 - x**4/4 + O(x**5)

        Handling of the ``logx`` parameter --- in the following example the
        expansion fails since ``sin`` does not have an asymptotic expansion
        at -oo (the limit of log(x) as x approaches 0):

        >>> e = sin(log(x))
        >>> e.nseries(x, 0, 6)
        Traceback (most recent call last):
        ...
        PoleError: ...
        ...
        >>> logx = Symbol('logx')
        >>> e.nseries(x, 0, 6, logx=logx)
        sin(logx)

        In the following example, the expansion works but only returns self
        unless the ``logx`` parameter is used:

        >>> e = x**y
        >>> e.nseries(x, 0, 2)
        x**y
        >>> e.nseries(x, 0, 2, logx=logx)
        exp(logx*y)

        """
        if x and x not in self.free_symbols:
            return self
        if x is None or x0 or dir != '+':  # {see XPOS above} or (x.is_positive == x.is_negative == None):
            return self.series(x, x0, n, dir, cdir=cdir)
        else:
            return self._eval_nseries(x, n=n, logx=logx, cdir=cdir)

    def _eval_nseries(self, x, n, logx, cdir):
        """
        Return terms of series for self up to O(x**n) at x=0
        from the positive direction.

        This is a method that should be overridden in subclasses. Users should
        never call this method directly (use .nseries() instead), so you do not
        have to write docstrings for _eval_nseries().
        """
        raise NotImplementedError(filldedent("""
                     The _eval_nseries method should be added to
                     %s to give terms up to O(x**n) at x=0
                     from the positive direction so it is available when
                     nseries calls it.""" % self.func)
                     )

    def limit(self, x, xlim, dir='+'):
        """ Compute limit x->xlim.
        """
        from sympy.series.limits import limit
        return limit(self, x, xlim, dir)

    @cacheit
    def as_leading_term(self, *symbols, logx=None, cdir=0):
        """
        Returns the leading (nonzero) term of the series expansion of self.

        The _eval_as_leading_term routines are used to do this, and they must
        always return a non-zero value.

        Examples
        ========

        >>> from sympy.abc import x
        >>> (1 + x + x**2).as_leading_term(x)
        1
        >>> (1/x**2 + x + x**2).as_leading_term(x)
        x**(-2)

        """
        if len(symbols) > 1:
            c = self
            for x in symbols:
                c = c.as_leading_term(x, logx=logx, cdir=cdir)
            return c
        elif not symbols:
            return self
        x = sympify(symbols[0])
        cdir = sympify(cdir)
        if not x.is_symbol:
            raise ValueError('expecting a Symbol but got %s' % x)
        if x not in self.free_symbols:
            return self
        obj = self._eval_as_leading_term(x, logx=logx, cdir=cdir)
        if obj is not None:
            from sympy.simplify.powsimp import powsimp
            return powsimp(obj, deep=True, combine='exp')
        raise NotImplementedError('as_leading_term(%s, %s)' % (self, x))

    def _eval_as_leading_term(self, x, logx, cdir):
        return self

    def as_coeff_exponent(self, x) -> tuple[Expr, Expr]:
        """ ``c*x**e -> c,e`` where x can be any symbolic expression.
        """
        from sympy.simplify.radsimp import collect
        s = collect(self, x)
        c, p = s.as_coeff_mul(x)
        if len(p) == 1:
            b, e = p[0].as_base_exp()
            if b == x:
                return c, e
        return s, S.Zero

    def leadterm(self, x, logx=None, cdir=0):
        """
        Returns the leading term a*x**b as a tuple (a, b).

        Examples
        ========

        >>> from sympy.abc import x
        >>> (1+x+x**2).leadterm(x)
        (1, 0)
        >>> (1/x**2+x+x**2).leadterm(x)
        (1, -2)

        """
        from .symbol import Dummy
        from sympy.functions.elementary.exponential import log
        l = self.as_leading_term(x, logx=logx, cdir=cdir)
        d = Dummy('logx')
        if l.has(log(x)):
            l = l.subs(log(x), d)
        c, e = l.as_coeff_exponent(x)
        if x in c.free_symbols:
            raise ValueError(filldedent("""
                cannot compute leadterm(%s, %s). The coefficient
                should have been free of %s but got %s""" % (self, x, x, c)))
        c = c.subs(d, log(x))
        return c, e

    def as_coeff_Mul(self, rational: bool = False) -> tuple['Number', Expr]:
        """Efficiently extract the coefficient of a product."""
        return S.One, self

    def as_coeff_Add(self, rational=False) -> tuple['Number', Expr]:
        """Efficiently extract the coefficient of a summation."""
        return S.Zero, self

    def fps(self, x=None, x0=0, dir=1, hyper=True, order=4, rational=True,
            full=False):
        """
        Compute formal power power series of self.

        See the docstring of the :func:`fps` function in sympy.series.formal for
        more information.
        """
        from sympy.series.formal import fps

        return fps(self, x, x0, dir, hyper, order, rational, full)

    def fourier_series(self, limits=None):
        """Compute fourier sine/cosine series of self.

        See the docstring of the :func:`fourier_series` in sympy.series.fourier
        for more information.
        """
        from sympy.series.fourier import fourier_series

        return fourier_series(self, limits)

    ###################################################################################
    ##################### DERIVATIVE, INTEGRAL, FUNCTIONAL METHODS ####################
    ###################################################################################

    def diff(self, *symbols, **assumptions):
        assumptions.setdefault("evaluate", True)
        return _derivative_dispatch(self, *symbols, **assumptions)

    ###########################################################################
    ###################### EXPRESSION EXPANSION METHODS #######################
    ###########################################################################

    # Relevant subclasses should override _eval_expand_hint() methods.  See
    # the docstring of expand() for more info.

    def _eval_expand_complex(self, **hints):
        real, imag = self.as_real_imag(**hints)
        return real + S.ImaginaryUnit*imag

    @staticmethod
    def _expand_hint(expr, hint, deep=True, **hints):
        """
        Helper for ``expand()``.  Recursively calls ``expr._eval_expand_hint()``.

        Returns ``(expr, hit)``, where expr is the (possibly) expanded
        ``expr`` and ``hit`` is ``True`` if ``expr`` was truly expanded and
        ``False`` otherwise.
        """
        hit = False
        # XXX: Hack to support non-Basic args
        #              |
        #              V
        if deep and getattr(expr, 'args', ()) and not expr.is_Atom:
            sargs = []
            for arg in expr.args:
                arg, arghit = Expr._expand_hint(arg, hint, **hints)
                hit |= arghit
                sargs.append(arg)

            if hit:
                expr = expr.func(*sargs)

        if hasattr(expr, hint):
            newexpr = getattr(expr, hint)(**hints)
            if newexpr != expr:
                return (newexpr, True)

        return (expr, hit)

    @cacheit
    def expand(self, deep=True, modulus=None, power_base=True, power_exp=True,
            mul=True, log=True, multinomial=True, basic=True, **hints):
        """
        Expand an expression using hints.

        See the docstring of the expand() function in sympy.core.function for
        more information.

        """
        from sympy.simplify.radsimp import fraction

        hints.update(power_base=power_base, power_exp=power_exp, mul=mul,
           log=log, multinomial=multinomial, basic=basic)

        expr = self
        # default matches fraction's default
        _fraction = lambda x: fraction(x, hints.get('exact', False))
        if hints.pop('frac', False):
            n, d = [a.expand(deep=deep, modulus=modulus, **hints)
                    for a in _fraction(self)]
            return n/d
        elif hints.pop('denom', False):
            n, d = _fraction(self)
            return n/d.expand(deep=deep, modulus=modulus, **hints)
        elif hints.pop('numer', False):
            n, d = _fraction(self)
            return n.expand(deep=deep, modulus=modulus, **hints)/d

        # Although the hints are sorted here, an earlier hint may get applied
        # at a given node in the expression tree before another because of how
        # the hints are applied.  e.g. expand(log(x*(y + z))) -> log(x*y +
        # x*z) because while applying log at the top level, log and mul are
        # applied at the deeper level in the tree so that when the log at the
        # upper level gets applied, the mul has already been applied at the
        # lower level.

        # Additionally, because hints are only applied once, the expression
        # may not be expanded all the way.   For example, if mul is applied
        # before multinomial, x*(x + 1)**2 won't be expanded all the way.  For
        # now, we just use a special case to make multinomial run before mul,
        # so that at least polynomials will be expanded all the way.  In the
        # future, smarter heuristics should be applied.
        # TODO: Smarter heuristics

        def _expand_hint_key(hint):
            """Make multinomial come before mul"""
            if hint == 'mul':
                return 'mulz'
            return hint

        for hint in sorted(hints.keys(), key=_expand_hint_key):
            use_hint = hints[hint]
            if use_hint:
                hint = '_eval_expand_' + hint
                expr, hit = Expr._expand_hint(expr, hint, deep=deep, **hints)

        while True:
            was = expr
            if hints.get('multinomial', False):
                expr, _ = Expr._expand_hint(
                    expr, '_eval_expand_multinomial', deep=deep, **hints)
            if hints.get('mul', False):
                expr, _ = Expr._expand_hint(
                    expr, '_eval_expand_mul', deep=deep, **hints)
            if hints.get('log', False):
                expr, _ = Expr._expand_hint(
                    expr, '_eval_expand_log', deep=deep, **hints)
            if expr == was:
                break

        if modulus is not None:
            modulus = sympify(modulus)

            if not modulus.is_Integer or modulus <= 0:
                raise ValueError(
                    "modulus must be a positive integer, got %s" % modulus)

            terms = []

            for term in Add.make_args(expr):
                coeff, tail = term.as_coeff_Mul(rational=True)

                coeff %= modulus

                if coeff:
                    terms.append(coeff*tail)

            expr = Add(*terms)

        return expr

    ###########################################################################
    ################### GLOBAL ACTION VERB WRAPPER METHODS ####################
    ###########################################################################

    def integrate(self, *args, **kwargs):
        """See the integrate function in sympy.integrals"""
        from sympy.integrals.integrals import integrate
        return integrate(self, *args, **kwargs)

    def nsimplify(self, constants=(), tolerance=None, full=False):
        """See the nsimplify function in sympy.simplify"""
        from sympy.simplify.simplify import nsimplify
        return nsimplify(self, constants, tolerance, full)

    def separate(self, deep=False, force=False):
        """See the separate function in sympy.simplify"""
        from .function import expand_power_base
        return expand_power_base(self, deep=deep, force=force)

    def collect(self, syms, func=None, evaluate=True, exact=False, distribute_order_term=True):
        """See the collect function in sympy.simplify"""
        from sympy.simplify.radsimp import collect
        return collect(self, syms, func, evaluate, exact, distribute_order_term)

    def together(self, *args, **kwargs):
        """See the together function in sympy.polys"""
        from sympy.polys.rationaltools import together
        return together(self, *args, **kwargs)

    def apart(self, x=None, **args):
        """See the apart function in sympy.polys"""
        from sympy.polys.partfrac import apart
        return apart(self, x, **args)

    def ratsimp(self):
        """See the ratsimp function in sympy.simplify"""
        from sympy.simplify.ratsimp import ratsimp
        return ratsimp(self)

    def trigsimp(self, **args):
        """See the trigsimp function in sympy.simplify"""
        from sympy.simplify.trigsimp import trigsimp
        return trigsimp(self, **args)

    def radsimp(self, **kwargs):
        """See the radsimp function in sympy.simplify"""
        from sympy.simplify.radsimp import radsimp
        return radsimp(self, **kwargs)

    def powsimp(self, *args, **kwargs):
        """See the powsimp function in sympy.simplify"""
        from sympy.simplify.powsimp import powsimp
        return powsimp(self, *args, **kwargs)

    def combsimp(self):
        """See the combsimp function in sympy.simplify"""
        from sympy.simplify.combsimp import combsimp
        return combsimp(self)

    def gammasimp(self):
        """See the gammasimp function in sympy.simplify"""
        from sympy.simplify.gammasimp import gammasimp
        return gammasimp(self)

    def factor(self, *gens, **args):
        """See the factor() function in sympy.polys.polytools"""
        from sympy.polys.polytools import factor
        return factor(self, *gens, **args)

    def cancel(self, *gens, **args):
        """See the cancel function in sympy.polys"""
        from sympy.polys.polytools import cancel
        return cancel(self, *gens, **args)

    def invert(self, g, *gens, **args):
        """Return the multiplicative inverse of ``self`` mod ``g``
        where ``self`` (and ``g``) may be symbolic expressions).

        See Also
        ========
        sympy.core.intfunc.mod_inverse, sympy.polys.polytools.invert
        """
        if self.is_number and getattr(g, 'is_number', True):
            return mod_inverse(self, g)
        from sympy.polys.polytools import invert
        return invert(self, g, *gens, **args)

    def round(self, n=None):
        """Return x rounded to the given decimal place.

        If a complex number would results, apply round to the real
        and imaginary components of the number.

        Examples
        ========

        >>> from sympy import pi, E, I, S, Number
        >>> pi.round()
        3
        >>> pi.round(2)
        3.14
        >>> (2*pi + E*I).round()
        6 + 3*I

        The round method has a chopping effect:

        >>> (2*pi + I/10).round()
        6
        >>> (pi/10 + 2*I).round()
        2*I
        >>> (pi/10 + E*I).round(2)
        0.31 + 2.72*I

        Notes
        =====

        The Python ``round`` function uses the SymPy ``round`` method so it
        will always return a SymPy number (not a Python float or int):

        >>> isinstance(round(S(123), -2), Number)
        True
        """
        x = self

        if not x.is_number:
            raise TypeError("Cannot round symbolic expression")
        if not x.is_Atom:
            if not pure_complex(x.n(2), or_real=True):
                raise TypeError(
                    'Expected a number but got %s:' % func_name(x))
        elif x in _illegal:
            return x
        if not (xr := x.is_extended_real):
            r, i = x.as_real_imag()
            if xr is False:
                return r.round(n) + S.ImaginaryUnit*i.round(n)
            if i.equals(0):
                return r.round(n)
        if not x:
            return S.Zero if n is None else x

        p = as_int(n or 0)

        if x.is_Integer:
            return Integer(round(int(x), p))

        digits_to_decimal = _mag(x)  # _mag(12) = 2, _mag(.012) = -1
        allow = digits_to_decimal + p
        precs = [f._prec for f in x.atoms(Float)]
        dps = prec_to_dps(max(precs)) if precs else None
        if dps is None:
            # assume everything is exact so use the Python
            # float default or whatever was requested
            dps = max(15, allow)
        else:
            allow = min(allow, dps)
        # this will shift all digits to right of decimal
        # and give us dps to work with as an int
        shift = -digits_to_decimal + dps
        extra = 1  # how far we look past known digits
        # NOTE
        # mpmath will calculate the binary representation to
        # an arbitrary number of digits but we must base our
        # answer on a finite number of those digits, e.g.
        # .575 2589569785738035/2**52 in binary.
        # mpmath shows us that the first 18 digits are
        #     >>> Float(.575).n(18)
        #     0.574999999999999956
        # The default precision is 15 digits and if we ask
        # for 15 we get
        #     >>> Float(.575).n(15)
        #     0.575000000000000
        # mpmath handles rounding at the 15th digit. But we
        # need to be careful since the user might be asking
        # for rounding at the last digit and our semantics
        # are to round toward the even final digit when there
        # is a tie. So the extra digit will be used to make
        # that decision. In this case, the value is the same
        # to 15 digits:
        #     >>> Float(.575).n(16)
        #     0.5750000000000000
        # Now converting this to the 15 known digits gives
        #     575000000000000.0
        # which rounds to integer
        #    5750000000000000
        # And now we can round to the desired digt, e.g. at
        # the second from the left and we get
        #    5800000000000000
        # and rescaling that gives
        #    0.58
        # as the final result.
        # If the value is made slightly less than 0.575 we might
        # still obtain the same value:
        #    >>> Float(.575-1e-16).n(16)*10**15
        #    574999999999999.8
        # What 15 digits best represents the known digits (which are
        # to the left of the decimal? 5750000000000000, the same as
        # before. The only way we will round down (in this case) is
        # if we declared that we had more than 15 digits of precision.
        # For example, if we use 16 digits of precision, the integer
        # we deal with is
        #    >>> Float(.575-1e-16).n(17)*10**16
        #    5749999999999998.4
        # and this now rounds to 5749999999999998 and (if we round to
        # the 2nd digit from the left) we get 5700000000000000.
        #
        xf = x.n(dps + extra)*Pow(10, shift)
        if xf.is_Number and xf._prec == 1:  # xf.is_Add will raise below
            # is x == 0?
            if x.equals(0):
                return Float(0)
            raise ValueError('not computing with precision')
        xi = Integer(xf)
        # use the last digit to select the value of xi
        # nearest to x before rounding at the desired digit
        sign = 1 if x > 0 else -1
        dif2 = sign*(xf - xi).n(extra)
        if dif2 < 0:
            raise NotImplementedError(
                'not expecting int(x) to round away from 0')
        if dif2 > .5:
            xi += sign  # round away from 0
        elif dif2 == .5:
            xi += sign if xi%2 else -sign  # round toward even
        # shift p to the new position
        ip = p - shift
        # let Python handle the int rounding then rescale
        xr = round(xi.p, ip)
        # restore scale
        rv = Rational(xr, Pow(10, shift))
        # return Float or Integer
        if rv.is_Integer:
            if n is None:  # the single-arg case
                return rv
            # use str or else it won't be a float
            return Float(str(rv), dps)  # keep same precision
        else:
            if not allow and rv > self:
                allow += 1
            return Float(rv, allow)

    __round__ = round

    def _eval_derivative_matrix_lines(self, x):
        from sympy.matrices.expressions.matexpr import _LeftRightArgs
        return [_LeftRightArgs([S.One, S.One], higher=self._eval_derivative(x))]


class AtomicExpr(Atom, Expr):
    """
    A parent class for object which are both atoms and Exprs.

    For example: Symbol, Number, Rational, Integer, ...
    But not: Add, Mul, Pow, ...
    """
    is_number = False
    is_Atom = True

    __slots__ = ()

    def _eval_derivative(self, s):
        if self == s:
            return S.One
        return S.Zero

    def _eval_derivative_n_times(self, s, n):
        from .containers import Tuple
        from sympy.matrices.expressions.matexpr import MatrixExpr
        from sympy.matrices.matrixbase import MatrixBase
        if isinstance(s, (MatrixBase, Tuple, Iterable, MatrixExpr)):
            return super()._eval_derivative_n_times(s, n)
        from .relational import Eq
        from sympy.functions.elementary.piecewise import Piecewise
        if self == s:
            return Piecewise((self, Eq(n, 0)), (1, Eq(n, 1)), (0, True))
        else:
            return Piecewise((self, Eq(n, 0)), (0, True))

    def _eval_is_polynomial(self, syms):
        return True

    def _eval_is_rational_function(self, syms):
        return self not in _illegal

    def _eval_is_meromorphic(self, x, a):
        from sympy.calculus.accumulationbounds import AccumBounds
        return (not self.is_Number or self.is_finite) and not isinstance(self, AccumBounds)

    def _eval_is_algebraic_expr(self, syms):
        return True

    def _eval_nseries(self, x, n, logx, cdir=0):
        return self

    @property
    def expr_free_symbols(self):
        sympy_deprecation_warning("""
        The expr_free_symbols property is deprecated. Use free_symbols to get
        the free symbols of an expression.
        """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-expr-free-symbols")
        return {self}


def _mag(x):
    r"""Return integer $i$ such that $0.1 \le x/10^i < 1$

    Examples
    ========

    >>> from sympy.core.expr import _mag
    >>> from sympy import Float
    >>> _mag(Float(.1))
    0
    >>> _mag(Float(.01))
    -1
    >>> _mag(Float(1234))
    4
    """
    from math import log10, ceil, log
    xpos = abs(x.n())
    if not xpos:
        return S.Zero
    try:
        mag_first_dig = int(ceil(log10(xpos)))
    except (ValueError, OverflowError):
        mag_first_dig = int(ceil(Float(mpf_log(xpos._mpf_, 53))/log(10)))
    # check that we aren't off by 1
    if (xpos/S(10)**mag_first_dig) >= 1:
        assert 1 <= (xpos/S(10)**mag_first_dig) < 10
        mag_first_dig += 1
    return mag_first_dig


class UnevaluatedExpr(Expr):
    """
    Expression that is not evaluated unless released.

    Examples
    ========

    >>> from sympy import UnevaluatedExpr
    >>> from sympy.abc import x
    >>> x*(1/x)
    1
    >>> x*UnevaluatedExpr(1/x)
    x*1/x

    """

    def __new__(cls, arg, **kwargs):
        arg = _sympify(arg)
        obj = Expr.__new__(cls, arg, **kwargs)
        return obj

    def doit(self, **hints):
        if hints.get("deep", True):
            return self.args[0].doit(**hints)
        else:
            return self.args[0]



def unchanged(func, *args):
    """Return True if `func` applied to the `args` is unchanged.
    Can be used instead of `assert foo == foo`.

    Examples
    ========

    >>> from sympy import Piecewise, cos, pi
    >>> from sympy.core.expr import unchanged
    >>> from sympy.abc import x

    >>> unchanged(cos, 1)  # instead of assert cos(1) == cos(1)
    True

    >>> unchanged(cos, pi)
    False

    Comparison of args uses the builtin capabilities of the object's
    arguments to test for equality so args can be defined loosely. Here,
    the ExprCondPair arguments of Piecewise compare as equal to the
    tuples that can be used to create the Piecewise:

    >>> unchanged(Piecewise, (x, x > 1), (0, True))
    True
    """
    f = func(*args)
    return f.func == func and f.args == args


class ExprBuilder:
    def __init__(self, op, args=None, validator=None, check=True):
        if not hasattr(op, "__call__"):
            raise TypeError("op {} needs to be callable".format(op))
        self.op = op
        if args is None:
            self.args = []
        else:
            self.args = args
        self.validator = validator
        if (validator is not None) and check:
            self.validate()

    @staticmethod
    def _build_args(args):
        return [i.build() if isinstance(i, ExprBuilder) else i for i in args]

    def validate(self):
        if self.validator is None:
            return
        args = self._build_args(self.args)
        self.validator(*args)

    def build(self, check=True):
        args = self._build_args(self.args)
        if self.validator and check:
            self.validator(*args)
        return self.op(*args)

    def append_argument(self, arg, check=True):
        self.args.append(arg)
        if self.validator and check:
            self.validate(*self.args)

    def __getitem__(self, item):
        if item == 0:
            return self.op
        else:
            return self.args[item-1]

    def __repr__(self):
        return str(self.build())

    def search_element(self, elem):
        for i, arg in enumerate(self.args):
            if isinstance(arg, ExprBuilder):
                ret = arg.search_index(elem)
                if ret is not None:
                    return (i,) + ret
            elif id(arg) == id(elem):
                return (i,)
        return None


from .mul import Mul
from .add import Add
from .power import Pow
from .function import Function, _derivative_dispatch
from .mod import Mod
from .exprtools import factor_terms
from .numbers import Float, Integer, Rational, _illegal, int_valued
<|endoftext|>"""
SymPy core decorators.

The purpose of this module is to expose decorators without any other
dependencies, so that they can be easily imported anywhere in sympy/core.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from functools import wraps
from .sympify import SympifyError, sympify


if TYPE_CHECKING:
    from typing import Callable, TypeVar, Union
    T1 = TypeVar('T1')
    T2 = TypeVar('T2')
    T3 = TypeVar('T3')


def _sympifyit(arg, retval=None) -> Callable[[Callable[[T1, T2], T3]], Callable[[T1, T2], T3]]:
    """
    decorator to smartly _sympify function arguments

    Explanation
    ===========

    @_sympifyit('other', NotImplemented)
    def add(self, other):
        ...

    In add, other can be thought of as already being a SymPy object.

    If it is not, the code is likely to catch an exception, then other will
    be explicitly _sympified, and the whole code restarted.

    if _sympify(arg) fails, NotImplemented will be returned

    See also
    ========

    __sympifyit
    """
    def deco(func):
        return __sympifyit(func, arg, retval)

    return deco


def __sympifyit(func, arg, retval=None):
    """Decorator to _sympify `arg` argument for function `func`.

       Do not use directly -- use _sympifyit instead.
    """

    # we support f(a,b) only
    if not func.__code__.co_argcount:
        raise LookupError("func not found")
    # only b is _sympified
    assert func.__code__.co_varnames[1] == arg
    if retval is None:
        @wraps(func)
        def __sympifyit_wrapper(a, b):
            return func(a, sympify(b, strict=True))

    else:
        @wraps(func)
        def __sympifyit_wrapper(a, b):
            try:
                # If an external class has _op_priority, it knows how to deal
                # with SymPy objects. Otherwise, it must be converted.
                if not hasattr(b, '_op_priority'):
                    b = sympify(b, strict=True)
                return func(a, b)
            except SympifyError:
                return retval

    return __sympifyit_wrapper


def call_highest_priority(method_name: str
    ) -> Callable[[Callable[[T1, T2], T3]], Callable[[T1, T2], T3]]:
    """A decorator for binary special methods to handle _op_priority.

    Explanation
    ===========

    Binary special methods in Expr and its subclasses use a special attribute
    '_op_priority' to determine whose special method will be called to
    handle the operation. In general, the object having the highest value of
    '_op_priority' will handle the operation. Expr and subclasses that define
    custom binary special methods (__mul__, etc.) should decorate those
    methods with this decorator to add the priority logic.

    The ``method_name`` argument is the name of the method of the other class
    that will be called.  Use this decorator in the following manner::

        # Call other.__rmul__ if other._op_priority > self._op_priority
        @call_highest_priority('__rmul__')
        def __mul__(self, other):
            ...

        # Call other.__mul__ if other._op_priority > self._op_priority
        @call_highest_priority('__mul__')
        def __rmul__(self, other):
        ...
    """
    def priority_decorator(func: Callable[[T1, T2], T3]) -> Callable[[T1, T2], T3]:
        @wraps(func)
        def binary_op_wrapper(self: T1, other: T2) -> T3:
            if hasattr(other, '_op_priority'):
                if other._op_priority > self._op_priority:  # type: ignore
                    f: Union[Callable[[T1], T3], None] = getattr(other, method_name, None)
                    if f is not None:
                        return f(self)
            return func(self, other)
        return binary_op_wrapper
    return priority_decorator


def sympify_method_args(cls: type[T1]) -> type[T1]:
    '''Decorator for a class with methods that sympify arguments.

    Explanation
    ===========

    The sympify_method_args decorator is to be used with the sympify_return
    decorator for automatic sympification of method arguments. This is
    intended for the common idiom of writing a class like :

    Examples
    ========

    >>> from sympy import Basic, SympifyError, S
    >>> from sympy.core.sympify import _sympify

    >>> class MyTuple(Basic):
    ...     def __add__(self, other):
    ...         try:
    ...             other = _sympify(other)
    ...         except SympifyError:
    ...             return NotImplemented
    ...         if not isinstance(other, MyTuple):
    ...             return NotImplemented
    ...         return MyTuple(*(self.args + other.args))

    >>> MyTuple(S(1), S(2)) + MyTuple(S(3), S(4))
    MyTuple(1, 2, 3, 4)

    In the above it is important that we return NotImplemented when other is
    not sympifiable and also when the sympified result is not of the expected
    type. This allows the MyTuple class to be used cooperatively with other
    classes that overload __add__ and want to do something else in combination
    with instance of Tuple.

    Using this decorator the above can be written as

    >>> from sympy.core.decorators import sympify_method_args, sympify_return

    >>> @sympify_method_args
    ... class MyTuple(Basic):
    ...     @sympify_return([('other', 'MyTuple')], NotImplemented)
    ...     def __add__(self, other):
    ...          return MyTuple(*(self.args + other.args))

    >>> MyTuple(S(1), S(2)) + MyTuple(S(3), S(4))
    MyTuple(1, 2, 3, 4)

    The idea here is that the decorators take care of the boiler-plate code
    for making this happen in each method that potentially needs to accept
    unsympified arguments. Then the body of e.g. the __add__ method can be
    written without needing to worry about calling _sympify or checking the
    type of the resulting object.

    The parameters for sympify_return are a list of tuples of the form
    (parameter_name, expected_type) and the value to return (e.g.
    NotImplemented). The expected_type parameter can be a type e.g. Tuple or a
    string 'Tuple'. Using a string is useful for specifying a Type within its
    class body (as in the above example).

    Notes: Currently sympify_return only works for methods that take a single
    argument (not including self). Specifying an expected_type as a string
    only works for the class in which the method is defined.
    '''
    # Extract the wrapped methods from each of the wrapper objects created by
    # the sympify_return decorator. Doing this here allows us to provide the
    # cls argument which is used for forward string referencing.
    for attrname, obj in cls.__dict__.items():
        if isinstance(obj, _SympifyWrapper):
            setattr(cls, attrname, obj.make_wrapped(cls))
    return cls


def sympify_return(*args):
    '''Function/method decorator to sympify arguments automatically

    See the docstring of sympify_method_args for explanation.
    '''
    # Store a wrapper object for the decorated method
    def wrapper(func: Callable[[T1, T2], T3]) -> Callable[[T1, T2], T3]:
        return _SympifyWrapper(func, args)  # type: ignore
    return wrapper


class _SympifyWrapper:
    '''Internal class used by sympify_return and sympify_method_args'''

    def __init__(self, func, args):
        self.func = func
        self.args = args

    def make_wrapped(self, cls):
        func = self.func
        parameters, retval = self.args

        # XXX: Handle more than one parameter?
        [(parameter, expectedcls)] = parameters

        # Handle forward references to the current class using strings
        if expectedcls == cls.__name__:
            expectedcls = cls

        # Raise RuntimeError since this is a failure at import time and should
        # not be recoverable.
        nargs = func.__code__.co_argcount
        # we support f(a, b) only
        if nargs != 2:
            raise RuntimeError('sympify_return can only be used with 2 argument functions')
        # only b is _sympified
        if func.__code__.co_varnames[1] != parameter:
            raise RuntimeError('parameter name mismatch "%s" in %s' %
                    (parameter, func.__name__))

        @wraps(func)
        def _func(self, other):
            # XXX: The check for _op_priority here should be removed. It is
            # needed to stop mutable matrices from being sympified to
            # immutable matrices which breaks things in quantum...
            if not hasattr(other, '_op_priority'):
                try:
                    other = sympify(other, strict=True)
                except SympifyError:
                    return retval
            if not isinstance(other, expectedcls):
                return retval
            return func(self, other)

        return _func
<|endoftext|>from __future__ import annotations


from .assumptions import StdFactKB, _assume_defined
from .basic import Basic, Atom
from .cache import cacheit
from .containers import Tuple
from .expr import Expr, AtomicExpr
from .function import AppliedUndef, FunctionClass
from .kind import NumberKind, UndefinedKind
from .logic import fuzzy_bool
from .singleton import S
from .sorting import ordered
from .sympify import sympify
from sympy.logic.boolalg import Boolean
from sympy.utilities.iterables import sift, is_sequence
from sympy.utilities.misc import filldedent

import string
import re as _re
import random
from itertools import product
from typing import Any


class Str(Atom):
    """
    Represents string in SymPy.

    Explanation
    ===========

    Previously, ``Symbol`` was used where string is needed in ``args`` of SymPy
    objects, e.g. denoting the name of the instance. However, since ``Symbol``
    represents mathematical scalar, this class should be used instead.

    """
    __slots__ = ('name',)

    def __new__(cls, name, **kwargs):
        if not isinstance(name, str):
            raise TypeError("name should be a string, not %s" % repr(type(name)))
        obj = Expr.__new__(cls, **kwargs)
        obj.name = name
        return obj

    def __getnewargs__(self):
        return (self.name,)

    def _hashable_content(self):
        return (self.name,)


def _filter_assumptions(kwargs):
    """Split the given dict into assumptions and non-assumptions.
    Keys are taken as assumptions if they correspond to an
    entry in ``_assume_defined``.
    """
    assumptions, nonassumptions = map(dict, sift(kwargs.items(),
        lambda i: i[0] in _assume_defined,
        binary=True))
    Symbol._sanitize(assumptions)
    return assumptions, nonassumptions

def _symbol(s, matching_symbol=None, **assumptions):
    """Return s if s is a Symbol, else if s is a string, return either
    the matching_symbol if the names are the same or else a new symbol
    with the same assumptions as the matching symbol (or the
    assumptions as provided).

    Examples
    ========

    >>> from sympy import Symbol
    >>> from sympy.core.symbol import _symbol
    >>> _symbol('y')
    y
    >>> _.is_real is None
    True
    >>> _symbol('y', real=True).is_real
    True

    >>> x = Symbol('x')
    >>> _symbol(x, real=True)
    x
    >>> _.is_real is None  # ignore attribute if s is a Symbol
    True

    Below, the variable sym has the name 'foo':

    >>> sym = Symbol('foo', real=True)

    Since 'x' is not the same as sym's name, a new symbol is created:

    >>> _symbol('x', sym).name
    'x'

    It will acquire any assumptions give:

    >>> _symbol('x', sym, real=False).is_real
    False

    Since 'foo' is the same as sym's name, sym is returned

    >>> _symbol('foo', sym)
    foo

    Any assumptions given are ignored:

    >>> _symbol('foo', sym, real=False).is_real
    True

    NB: the symbol here may not be the same as a symbol with the same
    name defined elsewhere as a result of different assumptions.

    See Also
    ========

    sympy.core.symbol.Symbol

    """
    if isinstance(s, str):
        if matching_symbol and matching_symbol.name == s:
            return matching_symbol
        return Symbol(s, **assumptions)
    elif isinstance(s, Symbol):
        return s
    else:
        raise ValueError('symbol must be string for symbol name or Symbol')

def uniquely_named_symbol(xname, exprs=(), compare=str, modify=None, **assumptions):
    """
    Return a symbol whose name is derivated from *xname* but is unique
    from any other symbols in *exprs*.

    *xname* and symbol names in *exprs* are passed to *compare* to be
    converted to comparable forms. If ``compare(xname)`` is not unique,
    it is recursively passed to *modify* until unique name is acquired.

    Parameters
    ==========

    xname : str or Symbol
        Base name for the new symbol.

    exprs : Expr or iterable of Expr
        Expressions whose symbols are compared to *xname*.

    compare : function
        Unary function which transforms *xname* and symbol names from
        *exprs* to comparable form.

    modify : function
        Unary function which modifies the string. Default is appending
        the number, or increasing the number if exists.

    Examples
    ========

    By default, a number is appended to *xname* to generate unique name.
    If the number already exists, it is recursively increased.

    >>> from sympy.core.symbol import uniquely_named_symbol, Symbol
    >>> uniquely_named_symbol('x', Symbol('x'))
    x0
    >>> uniquely_named_symbol('x', (Symbol('x'), Symbol('x0')))
    x1
    >>> uniquely_named_symbol('x0', (Symbol('x1'), Symbol('x0')))
    x2

    Name generation can be controlled by passing *modify* parameter.

    >>> from sympy.abc import x
    >>> uniquely_named_symbol('x', x, modify=lambda s: 2*s)
    xx

    """
    def numbered_string_incr(s, start=0):
        if not s:
            return str(start)
        i = len(s) - 1
        while i != -1:
            if not s[i].isdigit():
                break
            i -= 1
        n = str(int(s[i + 1:] or start - 1) + 1)
        return s[:i + 1] + n

    default = None
    if is_sequence(xname):
        xname, default = xname
    x = compare(xname)
    if not exprs:
        return _symbol(x, default, **assumptions)
    if not is_sequence(exprs):
        exprs = [exprs]
    names = set().union(
        [i.name for e in exprs for i in e.atoms(Symbol)] +
        [i.func.name for e in exprs for i in e.atoms(AppliedUndef)])
    if modify is None:
        modify = numbered_string_incr
    while any(x == compare(s) for s in names):
        x = modify(x)
    return _symbol(x, default, **assumptions)
_uniquely_named_symbol = uniquely_named_symbol


# XXX: We need type: ignore below because Expr and Boolean are incompatible as
# superclasses. Really Symbol should not be a subclass of Boolean.


class Symbol(AtomicExpr, Boolean): # type: ignore
    """
    Symbol class is used to create symbolic variables.

    Explanation
    ===========

    Symbolic variables are placeholders for mathematical symbols that can represent numbers, constants, or any other mathematical entities and can be used in mathematical expressions and to perform symbolic computations.

    Assumptions:

    commutative = True
    positive = True
    real = True
    imaginary = True
    complex = True
    complete list of more assumptions- :ref:`predicates`

    You can override the default assumptions in the constructor.

    Examples
    ========

    >>> from sympy import Symbol
    >>> x = Symbol("x", positive=True)
    >>> x.is_positive
    True
    >>> x.is_negative
    False

    passing in greek letters:

    >>> from sympy import Symbol
    >>> alpha = Symbol('alpha')
    >>> alpha #doctest: +SKIP
    α

    Trailing digits are automatically treated like subscripts of what precedes them in the name.
    General format to add subscript to a symbol :
    ``<var_name> = Symbol('<symbol_name>_<subscript>')``

    >>> from sympy import Symbol
    >>> alpha_i = Symbol('alpha_i')
    >>> alpha_i #doctest: +SKIP
    αᵢ

    Parameters
    ==========

    AtomicExpr: variable name
    Boolean: Assumption with a boolean value(True or False)
    """

    is_comparable = False

    __slots__ = ('name', '_assumptions_orig', '_assumptions0')

    name: str

    is_Symbol = True
    is_symbol = True

    @property
    def kind(self):
        if self.is_commutative:
            return NumberKind
        return UndefinedKind

    @property
    def _diff_wrt(self):
        """Allow derivatives wrt Symbols.

        Examples
        ========

        >>> from sympy import Symbol
        >>> x = Symbol('x')
        >>> x._diff_wrt
        True
        """
        return True

    @staticmethod
    def _sanitize(assumptions, obj=None):
        """Remove None, convert values to bool, check commutativity *in place*.
        """

        # be strict about commutativity: cannot be None
        is_commutative = fuzzy_bool(assumptions.get('commutative', True))
        if is_commutative is None:
            whose = '%s ' % obj.__name__ if obj else ''
            raise ValueError(
                '%scommutativity must be True or False.' % whose)

        # sanitize other assumptions so 1 -> True and 0 -> False
        for key in list(assumptions.keys()):
            v = assumptions[key]
            if v is None:
                assumptions.pop(key)
                continue
            assumptions[key] = bool(v)

    def _merge(self, assumptions):
        base = self.assumptions0
        for k in set(assumptions) & set(base):
            if assumptions[k] != base[k]:
                raise ValueError(filldedent('''
                    non-matching assumptions for %s: existing value
                    is %s and new value is %s''' % (
                    k, base[k], assumptions[k])))
        base.update(assumptions)
        return base

    def __new__(cls, name, **assumptions):
        """Symbols are identified by name and assumptions::

        >>> from sympy import Symbol
        >>> Symbol("x") == Symbol("x")
        True
        >>> Symbol("x", real=True) == Symbol("x", real=False)
        False

        """
        cls._sanitize(assumptions, cls)
        return Symbol.__xnew_cached_(cls, name, **assumptions)


    @staticmethod
    @cacheit
    def _canonical_assumptions(**assumptions):
        # This is retained purely so that srepr can include commutative=True if
        # that was explicitly specified but not if it was not. Ideally srepr
        # should not distinguish these cases because the symbols otherwise
        # compare equal and are considered equivalent.
        #
        # See https://github.com/sympy/sympy/issues/8873
        #
        assumptions_orig = assumptions.copy()

        # The only assumption that is assumed by default is commutative=True:
        assumptions.setdefault('commutative', True)

        assumptions_kb = StdFactKB(assumptions)
        assumptions0 = dict(assumptions_kb)

        return assumptions_kb, assumptions_orig, assumptions0

    @staticmethod
    def __xnew__(cls, name, **assumptions):  # never cached (e.g. dummy)
        if not isinstance(name, str):
            raise TypeError("name should be a string, not %s" % repr(type(name)))


        obj = Expr.__new__(cls)
        obj.name = name

        assumptions_kb, assumptions_orig, assumptions0 = Symbol._canonical_assumptions(**assumptions)

        obj._assumptions = assumptions_kb
        obj._assumptions_orig = assumptions_orig
        obj._assumptions0 = tuple(sorted(assumptions0.items()))

        # The three assumptions dicts are all a little different:
        #
        #   >>> from sympy import Symbol
        #   >>> x = Symbol('x', finite=True)
        #   >>> x.is_positive  # query an assumption
        #   >>> x._assumptions
        #   {'finite': True, 'infinite': False, 'commutative': True, 'positive': None}
        #   >>> x._assumptions0
        #   {'finite': True, 'infinite': False, 'commutative': True}
        #   >>> x._assumptions_orig
        #   {'finite': True}
        #
        # Two symbols with the same name are equal if their _assumptions0 are
        # the same. Arguably it should be _assumptions_orig that is being
        # compared because that is more transparent to the user (it is
        # what was passed to the constructor modulo changes made by _sanitize).

        return obj

    @staticmethod
    @cacheit
    def __xnew_cached_(cls, name, **assumptions):  # symbols are always cached
        return Symbol.__xnew__(cls, name, **assumptions)

    def __getnewargs_ex__(self):
        return ((self.name,), self._assumptions_orig)

    # NOTE: __setstate__ is not needed for pickles created by __getnewargs_ex__
    # but was used before Symbol was changed to use __getnewargs_ex__ in v1.9.
    # Pickles created in previous SymPy versions will still need __setstate__
    # so that they can be unpickled in SymPy > v1.9.

    def __setstate__(self, state):
        for name, value in state.items():
            setattr(self, name, value)

    def _hashable_content(self):
        return (self.name,) + self._assumptions0

    def _eval_subs(self, old, new):
        if old.is_Pow:
            from sympy.core.power import Pow
            return Pow(self, S.One, evaluate=False)._eval_subs(old, new)

    def _eval_refine(self, assumptions):
        return self

    @property
    def assumptions0(self):
        return dict(self._assumptions0)

    @cacheit
    def sort_key(self, order=None):
        return self.class_key(), (1, (self.name,)), S.One.sort_key(), S.One

    def as_dummy(self):
        # only put commutativity in explicitly if it is False
        return Dummy(self.name) if self.is_commutative is not False \
            else Dummy(self.name, commutative=self.is_commutative)

    def as_real_imag(self, deep=True, **hints):
        if hints.get('ignore') == self:
            return None
        else:
            from sympy.functions.elementary.complexes import im, re
            return (re(self), im(self))

    def is_constant(self, *wrt, **flags):
        if not wrt:
            return False
        return self not in wrt

    @property
    def free_symbols(self):
        return {self}

    binary_symbols = free_symbols  # in this case, not always

    def as_set(self):
        return S.UniversalSet


class Dummy(Symbol):
    """Dummy symbols are each unique, even if they have the same name:

    Examples
    ========

    >>> from sympy import Dummy
    >>> Dummy("x") == Dummy("x")
    False

    If a name is not supplied then a string value of an internal count will be
    used. This is useful when a temporary variable is needed and the name
    of the variable used in the expression is not important.

    >>> Dummy() #doctest: +SKIP
    _Dummy_10

    """

    # In the rare event that a Dummy object needs to be recreated, both the
    # `name` and `dummy_index` should be passed.  This is used by `srepr` for
    # example:
    # >>> d1 = Dummy()
    # >>> d2 = eval(srepr(d1))
    # >>> d2 == d1
    # True
    #
    # If a new session is started between `srepr` and `eval`, there is a very
    # small chance that `d2` will be equal to a previously-created Dummy.

    _count = 0
    _prng = random.Random()
    _base_dummy_index = _prng.randint(10**6, 9*10**6)

    __slots__ = ('dummy_index',)

    is_Dummy = True

    def __new__(cls, name=None, dummy_index=None, **assumptions):
        if dummy_index is not None:
            assert name is not None, "If you specify a dummy_index, you must also provide a name"

        if name is None:
            name = "Dummy_" + str(Dummy._count)

        if dummy_index is None:
            dummy_index = Dummy._base_dummy_index + Dummy._count
            Dummy._count += 1

        cls._sanitize(assumptions, cls)
        obj = Symbol.__xnew__(cls, name, **assumptions)

        obj.dummy_index = dummy_index

        return obj

    def __getnewargs_ex__(self):
        return ((self.name, self.dummy_index), self._assumptions_orig)

    @cacheit
    def sort_key(self, order=None):
        return self.class_key(), (
            2, (self.name, self.dummy_index)), S.One.sort_key(), S.One

    def _hashable_content(self):
        return Symbol._hashable_content(self) + (self.dummy_index,)


class Wild(Symbol):
    """
    A Wild symbol matches anything, or anything
    without whatever is explicitly excluded.

    Parameters
    ==========

    name : str
        Name of the Wild instance.

    exclude : iterable, optional
        Instances in ``exclude`` will not be matched.

    properties : iterable of functions, optional
        Functions, each taking an expressions as input
        and returns a ``bool``. All functions in ``properties``
        need to return ``True`` in order for the Wild instance
        to match the expression.

    Examples
    ========

    >>> from sympy import Wild, WildFunction, cos, pi
    >>> from sympy.abc import x, y, z
    >>> a = Wild('a')
    >>> x.match(a)
    {a_: x}
    >>> pi.match(a)
    {a_: pi}
    >>> (3*x**2).match(a*x)
    {a_: 3*x}
    >>> cos(x).match(a)
    {a_: cos(x)}
    >>> b = Wild('b', exclude=[x])
    >>> (3*x**2).match(b*x)
    >>> b.match(a)
    {a_: b_}
    >>> A = WildFunction('A')
    >>> A.match(a)
    {a_: A_}

    Tips
    ====

    When using Wild, be sure to use the exclude
    keyword to make the pattern more precise.
    Without the exclude pattern, you may get matches
    that are technically correct, but not what you
    wanted. For example, using the above without
    exclude:

    >>> from sympy import symbols
    >>> a, b = symbols('a b', cls=Wild)
    >>> (2 + 3*y).match(a*x + b*y)
    {a_: 2/x, b_: 3}

    This is technically correct, because
    (2/x)*x + 3*y == 2 + 3*y, but you probably
    wanted it to not match at all. The issue is that
    you really did not want a and b to include x and y,
    and the exclude parameter lets you specify exactly
    this.  With the exclude parameter, the pattern will
    not match.

    >>> a = Wild('a', exclude=[x, y])
    >>> b = Wild('b', exclude=[x, y])
    >>> (2 + 3*y).match(a*x + b*y)

    Exclude also helps remove ambiguity from matches.

    >>> E = 2*x**3*y*z
    >>> a, b = symbols('a b', cls=Wild)
    >>> E.match(a*b)
    {a_: 2*y*z, b_: x**3}
    >>> a = Wild('a', exclude=[x, y])
    >>> E.match(a*b)
    {a_: z, b_: 2*x**3*y}
    >>> a = Wild('a', exclude=[x, y, z])
    >>> E.match(a*b)
    {a_: 2, b_: x**3*y*z}

    Wild also accepts a ``properties`` parameter:

    >>> a = Wild('a', properties=[lambda k: k.is_Integer])
    >>> E.match(a*b)
    {a_: 2, b_: x**3*y*z}

    """
    is_Wild = True

    __slots__ = ('exclude', 'properties')

    def __new__(cls, name, exclude=(), properties=(), **assumptions):
        exclude = tuple([sympify(x) for x in exclude])
        properties = tuple(properties)
        cls._sanitize(assumptions, cls)
        return Wild.__xnew__(cls, name, exclude, properties, **assumptions)

    def __getnewargs__(self):
        return (self.name, self.exclude, self.properties)

    @staticmethod
    @cacheit
    def __xnew__(cls, name, exclude, properties, **assumptions):
        obj = Symbol.__xnew__(cls, name, **assumptions)
        obj.exclude = exclude
        obj.properties = properties
        return obj

    def _hashable_content(self):
        return super()._hashable_content() + (self.exclude, self.properties)

    # TODO add check against another Wild
    def matches(self, expr, repl_dict=None, old=False):
        if any(expr.has(x) for x in self.exclude):
            return None
        if not all(f(expr) for f in self.properties):
            return None
        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()
        repl_dict[self] = expr
        return repl_dict


_range = _re.compile('([0-9]*:[0-9]+|[a-zA-Z]?:[a-zA-Z])')


def symbols(names, *, cls=Symbol, **args) -> Any:
    r"""
    Transform strings into instances of :class:`Symbol` class.

    :func:`symbols` function returns a sequence of symbols with names taken
    from ``names`` argument, which can be a comma or whitespace delimited
    string, or a sequence of strings::

        >>> from sympy import symbols, Function

        >>> x, y, z = symbols('x,y,z')
        >>> a, b, c = symbols('a b c')

    The type of output is dependent on the properties of input arguments::

        >>> symbols('x')
        x
        >>> symbols('x,')
        (x,)
        >>> symbols('x,y')
        (x, y)
        >>> symbols(('a', 'b', 'c'))
        (a, b, c)
        >>> symbols(['a', 'b', 'c'])
        [a, b, c]
        >>> symbols({'a', 'b', 'c'})
        {a, b, c}

    If an iterable container is needed for a single symbol, set the ``seq``
    argument to ``True`` or terminate the symbol name with a comma::

        >>> symbols('x', seq=True)
        (x,)

    To reduce typing, range syntax is supported to create indexed symbols.
    Ranges are indicated by a colon and the type of range is determined by
    the character to the right of the colon. If the character is a digit
    then all contiguous digits to the left are taken as the nonnegative
    starting value (or 0 if there is no digit left of the colon) and all
    contiguous digits to the right are taken as 1 greater than the ending
    value::

        >>> symbols('x:10')
        (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9)

        >>> symbols('x5:10')
        (x5, x6, x7, x8, x9)
        >>> symbols('x5(:2)')
        (x50, x51)

        >>> symbols('x5:10,y:5')
        (x5, x6, x7, x8, x9, y0, y1, y2, y3, y4)

        >>> symbols(('x5:10', 'y:5'))
        ((x5, x6, x7, x8, x9), (y0, y1, y2, y3, y4))

    If the character to the right of the colon is a letter, then the single
    letter to the left (or 'a' if there is none) is taken as the start
    and all characters in the lexicographic range *through* the letter to
    the right are used as the range::

        >>> symbols('x:z')
        (x, y, z)
        >>> symbols('x:c')  # null range
        ()
        >>> symbols('x(:c)')
        (xa, xb, xc)

        >>> symbols(':c')
        (a, b, c)

        >>> symbols('a:d, x:z')
        (a, b, c, d, x, y, z)

        >>> symbols(('a:d', 'x:z'))
        ((a, b, c, d), (x, y, z))

    Multiple ranges are supported; contiguous numerical ranges should be
    separated by parentheses to disambiguate the ending number of one
    range from the starting number of the next::

        >>> symbols('x:2(1:3)')
        (x01, x02, x11, x12)
        >>> symbols(':3:2')  # parsing is from left to right
        (00, 01, 10, 11, 20, 21)

    Only one pair of parentheses surrounding ranges are removed, so to
    include parentheses around ranges, double them. And to include spaces,
    commas, or colons, escape them with a backslash::

        >>> symbols('x((a:b))')
        (x(a), x(b))
        >>> symbols(r'x(:1\,:2)')  # or r'x((:1)\,(:2))'
        (x(0,0), x(0,1))

    All newly created symbols have assumptions set according to ``args``::

        >>> a = symbols('a', integer=True)
        >>> a.is_integer
        True

        >>> x, y, z = symbols('x,y,z', real=True)
        >>> x.is_real and y.is_real and z.is_real
        True

    Despite its name, :func:`symbols` can create symbol-like objects like
    instances of Function or Wild classes. To achieve this, set ``cls``
    keyword argument to the desired type::

        >>> symbols('f,g,h', cls=Function)
        (f, g, h)

        >>> type(_[0])
        <class 'sympy.core.function.UndefinedFunction'>

    """
    result = []

    if isinstance(names, str):
        marker = 0
        splitters = r'\,', r'\:', r'\ '
        literals: list[tuple[str, str]] = []
        for splitter in splitters:
            if splitter in names:
                while chr(marker) in names:
                    marker += 1
                lit_char = chr(marker)
                marker += 1
                names = names.replace(splitter, lit_char)
                literals.append((lit_char, splitter[1:]))
        def literal(s):
            if literals:
                for c, l in literals:
                    s = s.replace(c, l)
            return s

        names = names.strip()
        as_seq = names.endswith(',')
        if as_seq:
            names = names[:-1].rstrip()
        if not names:
            raise ValueError('no symbols given')

        # split on commas
        names = [n.strip() for n in names.split(',')]
        if not all(n for n in names):
            raise ValueError('missing symbol between commas')
        # split on spaces
        for i in range(len(names) - 1, -1, -1):
            names[i: i + 1] = names[i].split()

        seq = args.pop('seq', as_seq)

        for name in names:
            if not name:
                raise ValueError('missing symbol')

            if ':' not in name:
                symbol = cls(literal(name), **args)
                result.append(symbol)
                continue

            split: list[str] = _range.split(name)
            split_list: list[list[str]] = []
            # remove 1 layer of bounding parentheses around ranges
            for i in range(len(split) - 1):
                if i and ':' in split[i] and split[i] != ':' and \
                        split[i - 1].endswith('(') and \
                        split[i + 1].startswith(')'):
                    split[i - 1] = split[i - 1][:-1]
                    split[i + 1] = split[i + 1][1:]
            for s in split:
                if ':' in s:
                    if s.endswith(':'):
                        raise ValueError('missing end range')
                    a, b = s.split(':')
                    if b[-1] in string.digits:
                        a_i = 0 if not a else int(a)
                        b_i = int(b)
                        split_list.append([str(c) for c in range(a_i, b_i)])
                    else:
                        a = a or 'a'
                        split_list.append([string.ascii_letters[c] for c in range(
                            string.ascii_letters.index(a),
                            string.ascii_letters.index(b) + 1)])  # inclusive
                    if not split_list[-1]:
                        break
                else:
                    split_list.append([s])
            else:
                seq = True
                if len(split_list) == 1:
                    names = split_list[0]
                else:
                    names = [''.join(s) for s in product(*split_list)]
                if literals:
                    result.extend([cls(literal(s), **args) for s in names])
                else:
                    result.extend([cls(s, **args) for s in names])

        if not seq and len(result) <= 1:
            if not result:
                return ()
            return result[0]

        return tuple(result)
    else:
        for name in names:
            result.append(symbols(name, cls=cls, **args))

        return type(names)(result)


def var(names, **args):
    """
    Create symbols and inject them into the global namespace.

    Explanation
    ===========

    This calls :func:`symbols` with the same arguments and puts the results
    into the *global* namespace. It's recommended not to use :func:`var` in
    library code, where :func:`symbols` has to be used::

    Examples
    ========

    >>> from sympy import var

    >>> var('x')
    x
    >>> x # noqa: F821
    x

    >>> var('a,ab,abc')
    (a, ab, abc)
    >>> abc # noqa: F821
    abc

    >>> var('x,y', real=True)
    (x, y)
    >>> x.is_real and y.is_real # noqa: F821
    True

    See :func:`symbols` documentation for more details on what kinds of
    arguments can be passed to :func:`var`.

    """
    def traverse(symbols, frame):
        """Recursively inject symbols to the global namespace. """
        for symbol in symbols:
            if isinstance(symbol, Basic):
                frame.f_globals[symbol.name] = symbol
            elif isinstance(symbol, FunctionClass):
                frame.f_globals[symbol.__name__] = symbol
            else:
                traverse(symbol, frame)

    from inspect import currentframe
    frame = currentframe().f_back

    try:
        syms = symbols(names, **args)

        if syms is not None:
            if isinstance(syms, Basic):
                frame.f_globals[syms.name] = syms
            elif isinstance(syms, FunctionClass):
                frame.f_globals[syms.__name__] = syms
            else:
                traverse(syms, frame)
    finally:
        del frame  # break cyclic dependencies as stated in inspect docs

    return syms

def disambiguate(*iter):
    """
    Return a Tuple containing the passed expressions with symbols
    that appear the same when printed replaced with numerically
    subscripted symbols, and all Dummy symbols replaced with Symbols.

    Parameters
    ==========

    iter: list of symbols or expressions.

    Examples
    ========

    >>> from sympy.core.symbol import disambiguate
    >>> from sympy import Dummy, Symbol, Tuple
    >>> from sympy.abc import y

    >>> tup = Symbol('_x'), Dummy('x'), Dummy('x')
    >>> disambiguate(*tup)
    (x_2, x, x_1)

    >>> eqs = Tuple(Symbol('x')/y, Dummy('x')/y)
    >>> disambiguate(*eqs)
    (x_1/y, x/y)

    >>> ix = Symbol('x', integer=True)
    >>> vx = Symbol('x')
    >>> disambiguate(vx + ix)
    (x + x_1,)

    To make your own mapping of symbols to use, pass only the free symbols
    of the expressions and create a dictionary:

    >>> free = eqs.free_symbols
    >>> mapping = dict(zip(free, disambiguate(*free)))
    >>> eqs.xreplace(mapping)
    (x_1/y, x/y)

    """
    new_iter = Tuple(*iter)
    key = lambda x:tuple(sorted(x.assumptions0.items()))
    syms = ordered(new_iter.free_symbols, keys=key)
    mapping = {}
    for s in syms:
        mapping.setdefault(str(s).lstrip('_'), []).append(s)
    reps = {}
    for k in mapping:
        # the first or only symbol doesn't get subscripted but make
        # sure that it's a Symbol, not a Dummy
        mapk0 = Symbol("%s" % (k), **mapping[k][0].assumptions0)
        if mapping[k][0] != mapk0:
            reps[mapping[k][0]] = mapk0
        # the others get subscripts (and are made into Symbols)
        skip = 0
        for i in range(1, len(mapping[k])):
            while True:
                name = "%s_%i" % (k, i + skip)
                if name not in mapping:
                    break
                skip += 1
            ki = mapping[k][i]
            reps[ki] = Symbol(name, **ki.assumptions0)
    return new_iter.xreplace(reps)
<|endoftext|>from __future__ import annotations
from typing import Callable, TYPE_CHECKING
from itertools import product

from .sympify import _sympify
from .cache import cacheit
from .singleton import S
from .expr import Expr
from .evalf import PrecisionExhausted
from .function import (expand_complex, expand_multinomial,
    expand_mul, _mexpand, PoleError)
from .logic import fuzzy_bool, fuzzy_not, fuzzy_and, fuzzy_or
from .parameters import global_parameters
from .relational import is_gt, is_lt
from .kind import NumberKind, UndefinedKind
from sympy.utilities.iterables import sift
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.misc import as_int
from sympy.multipledispatch import Dispatcher


class Pow(Expr):
    """
    Defines the expression x**y as "x raised to a power y"

    .. deprecated:: 1.7

       Using arguments that aren't subclasses of :class:`~.Expr` in core
       operators (:class:`~.Mul`, :class:`~.Add`, and :class:`~.Pow`) is
       deprecated. See :ref:`non-expr-args-deprecated` for details.

    Singleton definitions involving (0, 1, -1, oo, -oo, I, -I):

    +--------------+---------+-----------------------------------------------+
    | expr         | value   | reason                                        |
    +==============+=========+===============================================+
    | z**0         | 1       | Although arguments over 0**0 exist, see [2].  |
    +--------------+---------+-----------------------------------------------+
    | z**1         | z       |                                               |
    +--------------+---------+-----------------------------------------------+
    | (-oo)**(-1)  | 0       |                                               |
    +--------------+---------+-----------------------------------------------+
    | (-1)**-1     | -1      |                                               |
    +--------------+---------+-----------------------------------------------+
    | S.Zero**-1   | zoo     | This is not strictly true, as 0**-1 may be    |
    |              |         | undefined, but is convenient in some contexts |
    |              |         | where the base is assumed to be positive.     |
    +--------------+---------+-----------------------------------------------+
    | 1**-1        | 1       |                                               |
    +--------------+---------+-----------------------------------------------+
    | oo**-1       | 0       |                                               |
    +--------------+---------+-----------------------------------------------+
    | 0**oo        | 0       | Because for all complex numbers z near        |
    |              |         | 0, z**oo -> 0.                                |
    +--------------+---------+-----------------------------------------------+
    | 0**-oo       | zoo     | This is not strictly true, as 0**oo may be    |
    |              |         | oscillating between positive and negative     |
    |              |         | values or rotating in the complex plane.      |
    |              |         | It is convenient, however, when the base      |
    |              |         | is positive.                                  |
    +--------------+---------+-----------------------------------------------+
    | 1**oo        | nan     | Because there are various cases where         |
    | 1**-oo       |         | lim(x(t),t)=1, lim(y(t),t)=oo (or -oo),       |
    |              |         | but lim( x(t)**y(t), t) != 1.  See [3].       |
    +--------------+---------+-----------------------------------------------+
    | b**zoo       | nan     | Because b**z has no limit as z -> zoo         |
    +--------------+---------+-----------------------------------------------+
    | (-1)**oo     | nan     | Because of oscillations in the limit.         |
    | (-1)**(-oo)  |         |                                               |
    +--------------+---------+-----------------------------------------------+
    | oo**oo       | oo      |                                               |
    +--------------+---------+-----------------------------------------------+
    | oo**-oo      | 0       |                                               |
    +--------------+---------+-----------------------------------------------+
    | (-oo)**oo    | nan     |                                               |
    | (-oo)**-oo   |         |                                               |
    +--------------+---------+-----------------------------------------------+
    | oo**I        | nan     | oo**e could probably be best thought of as    |
    | (-oo)**I     |         | the limit of x**e for real x as x tends to    |
    |              |         | oo. If e is I, then the limit does not exist  |
    |              |         | and nan is used to indicate that.             |
    +--------------+---------+-----------------------------------------------+
    | oo**(1+I)    | zoo     | If the real part of e is positive, then the   |
    | (-oo)**(1+I) |         | limit of abs(x**e) is oo. So the limit value  |
    |              |         | is zoo.                                       |
    +--------------+---------+-----------------------------------------------+
    | oo**(-1+I)   | 0       | If the real part of e is negative, then the   |
    | -oo**(-1+I)  |         | limit is 0.                                   |
    +--------------+---------+-----------------------------------------------+

    Because symbolic computations are more flexible than floating point
    calculations and we prefer to never return an incorrect answer,
    we choose not to conform to all IEEE 754 conventions.  This helps
    us avoid extra test-case code in the calculation of limits.

    See Also
    ========

    sympy.core.numbers.Infinity
    sympy.core.numbers.NegativeInfinity
    sympy.core.numbers.NaN

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Exponentiation
    .. [2] https://en.wikipedia.org/wiki/Zero_to_the_power_of_zero
    .. [3] https://en.wikipedia.org/wiki/Indeterminate_forms

    """
    is_Pow = True

    __slots__ = ('is_commutative',)

    if TYPE_CHECKING:

        @property
        def args(self) -> tuple[Expr, Expr]:
            ...

    @property
    def base(self) -> Expr:
        return self.args[0]

    @property
    def exp(self) -> Expr:
        return self.args[1]

    @property
    def kind(self):
        if self.exp.kind is NumberKind:
            return self.base.kind
        else:
            return UndefinedKind

    @cacheit
    def __new__(cls, b: Expr | complex, e: Expr | complex, evaluate=None) -> Expr: # type: ignore
        if evaluate is None:
            evaluate = global_parameters.evaluate

        base = _sympify(b)
        exp = _sympify(e)

        # XXX: This can be removed when non-Expr args are disallowed rather
        # than deprecated.
        from .relational import Relational
        if isinstance(base, Relational) or isinstance(exp, Relational):
            raise TypeError('Relational cannot be used in Pow')

        # XXX: This should raise TypeError once deprecation period is over:
        for arg in [base, exp]:
            if not isinstance(arg, Expr):
                sympy_deprecation_warning(
                    f"""
    Using non-Expr arguments in Pow is deprecated (in this case, one of the
    arguments is of type {type(arg).__name__!r}).

    If you really did intend to construct a power with this base, use the **
    operator instead.""",
                    deprecated_since_version="1.7",
                    active_deprecations_target="non-expr-args-deprecated",
                    stacklevel=4,
                )

        if evaluate:
            if exp is S.ComplexInfinity:
                return S.NaN
            if exp is S.Infinity:
                if is_gt(base, S.One):
                    return S.Infinity
                if is_gt(base, S.NegativeOne) and is_lt(base, S.One):
                    return S.Zero
                if is_lt(base, S.NegativeOne):
                    if base.is_finite:
                        return S.ComplexInfinity
                    if base.is_finite is False:
                        return S.NaN
            if exp is S.Zero:
                return S.One
            elif exp is S.One:
                return base
            elif exp == -1 and not base:
                return S.ComplexInfinity
            elif exp.__class__.__name__ == "AccumulationBounds":
                if base == S.Exp1:
                    from sympy.calculus.accumulationbounds import AccumBounds
                    return AccumBounds(Pow(base, exp.min), Pow(base, exp.max))
            # autosimplification if base is a number and exp odd/even
            # if base is Number then the base will end up positive; we
            # do not do this with arbitrary expressions since symbolic
            # cancellation might occur as in (x - 1)/(1 - x) -> -1. If
            # we returned Piecewise((-1, Ne(x, 1))) for such cases then
            # we could do this...but we don't
            elif (exp.is_Symbol and exp.is_integer or exp.is_Integer
                    ) and (base.is_number and base.is_Mul or base.is_Number
                    ) and base.could_extract_minus_sign():
                if exp.is_even:
                    base = -base
                elif exp.is_odd:
                    return -Pow(-base, exp)
            if S.NaN in (base, exp):  # XXX S.NaN**x -> S.NaN under assumption that x != 0
                return S.NaN
            elif base is S.One:
                if abs(exp).is_infinite:
                    return S.NaN
                return S.One
            else:
                # recognize base as E
                from sympy.functions.elementary.exponential import exp_polar
                if not exp.is_Atom and base is not S.Exp1 and not isinstance(base, exp_polar):
                    from .exprtools import factor_terms
                    from sympy.functions.elementary.exponential import log
                    from sympy.simplify.radsimp import fraction
                    c, ex = factor_terms(exp, sign=False).as_coeff_Mul()
                    num, den = fraction(ex)
                    if isinstance(den, log) and den.args[0] == base:
                        return S.Exp1**(c*num)
                    elif den.is_Add:
                        from sympy.functions.elementary.complexes import sign, im
                        s = sign(im(base))
                        if s.is_Number and s and den == \
                                log(-factor_terms(base, sign=False)) + s*S.ImaginaryUnit*S.Pi:
                            return S.Exp1**(c*num)

                obj = base._eval_power(exp)
                if obj is not None:
                    return obj
        obj = Expr.__new__(cls, base, exp)
        obj = cls._exec_constructor_postprocessors(obj)
        if not isinstance(obj, Pow):
            return obj
        obj.is_commutative = (base.is_commutative and exp.is_commutative)
        return obj

    def inverse(self, argindex=1):
        if self.base == S.Exp1:
            from sympy.functions.elementary.exponential import log
            return log
        return None

    @classmethod
    def class_key(cls):
        return 3, 2, cls.__name__

    def _eval_refine(self, assumptions):
        from sympy.assumptions.ask import ask, Q
        b, e = self.as_base_exp()
        if ask(Q.integer(e), assumptions) and b.could_extract_minus_sign():
            if ask(Q.even(e), assumptions):
                return Pow(-b, e)
            elif ask(Q.odd(e), assumptions):
                return -Pow(-b, e)

    def _eval_power(self, expt):
        b, e = self.as_base_exp()
        if b is S.NaN:
            return (b**e)**expt  # let __new__ handle it

        s = None
        if expt.is_integer:
            s = 1
        elif b.is_polar:  # e.g. exp_polar, besselj, var('p', polar=True)...
            s = 1
        elif e.is_extended_real is not None:
            from sympy.functions.elementary.complexes import arg, im, re, sign
            from sympy.functions.elementary.exponential import exp, log
            from sympy.functions.elementary.integers import floor
            # helper functions ===========================
            def _half(e):
                """Return True if the exponent has a literal 2 as the
                denominator, else None."""
                if getattr(e, 'q', None) == 2:
                    return True
                n, d = e.as_numer_denom()
                if n.is_integer and d == 2:
                    return True
            def _n2(e):
                """Return ``e`` evaluated to a Number with 2 significant
                digits, else None."""
                try:
                    rv = e.evalf(2, strict=True)
                    if rv.is_Number:
                        return rv
                except PrecisionExhausted:
                    pass
            # ===================================================
            if e.is_extended_real:
                # we need _half(expt) with constant floor or
                # floor(S.Half - e*arg(b)/2/pi) == 0


                # handle -1 as special case
                if e == -1:
                    # floor arg. is 1/2 + arg(b)/2/pi
                    if _half(expt):
                        if b.is_negative is True:
                            return S.NegativeOne**expt*Pow(-b, e*expt)
                        elif b.is_negative is False:  # XXX ok if im(b) != 0?
                            return Pow(b, -expt)
                elif e.is_even:
                    if b.is_extended_real:
                        b = abs(b)
                    if b.is_imaginary:
                        b = abs(im(b))*S.ImaginaryUnit

                if (abs(e) < 1) == True or e == 1:
                    s = 1  # floor = 0
                elif b.is_extended_nonnegative:
                    s = 1  # floor = 0
                elif re(b).is_extended_nonnegative and (abs(e) < 2) == True:
                    s = 1  # floor = 0
                elif _half(expt):
                    s = exp(2*S.Pi*S.ImaginaryUnit*expt*floor(
                        S.Half - e*arg(b)/(2*S.Pi)))
                    if s.is_extended_real and _n2(sign(s) - s) == 0:
                        s = sign(s)
                    else:
                        s = None
            else:
                # e.is_extended_real is False requires:
                #     _half(expt) with constant floor or
                #     floor(S.Half - im(e*log(b))/2/pi) == 0
                try:
                    s = exp(2*S.ImaginaryUnit*S.Pi*expt*
                        floor(S.Half - im(e*log(b))/2/S.Pi))
                    # be careful to test that s is -1 or 1 b/c sign(I) == I:
                    # so check that s is real
                    if s.is_extended_real and _n2(sign(s) - s) == 0:
                        s = sign(s)
                    else:
                        s = None
                except PrecisionExhausted:
                    s = None

        if s is not None:
            return s*Pow(b, e*expt)

    def _eval_Mod(self, q):
        r"""A dispatched function to compute `b^e \bmod q`, dispatched
        by ``Mod``.

        Notes
        =====

        Algorithms:

        1. For unevaluated integer power, use built-in ``pow`` function
        with 3 arguments, if powers are not too large wrt base.

        2. For very large powers, use totient reduction if $e \ge \log(m)$.
        Bound on m, is for safe factorization memory wise i.e. $m^{1/4}$.
        For pollard-rho to be faster than built-in pow $\log(e) > m^{1/4}$
        check is added.

        3. For any unevaluated power found in `b` or `e`, the step 2
        will be recursed down to the base and the exponent
        such that the $b \bmod q$ becomes the new base and
        $\phi(q) + e \bmod \phi(q)$ becomes the new exponent, and then
        the computation for the reduced expression can be done.
        """

        base, exp = self.base, self.exp

        if exp.is_integer and exp.is_positive:
            if q.is_integer and base % q == 0:
                return S.Zero

            from sympy.functions.combinatorial.numbers import totient

            if base.is_Integer and exp.is_Integer and q.is_Integer:
                b, e, m = int(base), int(exp), int(q)
                mb = m.bit_length()
                if mb <= 80 and e >= mb and e.bit_length()**4 >= m:
                    phi = int(totient(m))
                    return Integer(pow(b, phi + e%phi, m))
                return Integer(pow(b, e, m))

            from .mod import Mod

            if isinstance(base, Pow) and base.is_integer and base.is_number:
                base = Mod(base, q)
                return Mod(Pow(base, exp, evaluate=False), q)

            if isinstance(exp, Pow) and exp.is_integer and exp.is_number:
                bit_length = int(q).bit_length()
                # XXX Mod-Pow actually attempts to do a hanging evaluation
                # if this dispatched function returns None.
                # May need some fixes in the dispatcher itself.
                if bit_length <= 80:
                    phi = totient(q)
                    exp = phi + Mod(exp, phi)
                    return Mod(Pow(base, exp, evaluate=False), q)

    def _eval_is_even(self):
        if self.exp.is_integer and self.exp.is_positive:
            return self.base.is_even

    def _eval_is_negative(self):
        ext_neg = Pow._eval_is_extended_negative(self)
        if ext_neg is True:
            return self.is_finite
        return ext_neg

    def _eval_is_extended_positive(self):
        if self.base == self.exp:
            if self.base.is_extended_nonnegative:
                return True
        elif self.base.is_positive:
            if self.exp.is_real:
                return True
        elif self.base.is_extended_negative:
            if self.exp.is_even:
                return True
            if self.exp.is_odd:
                return False
        elif self.base.is_zero:
            if self.exp.is_extended_real:
                return self.exp.is_zero
        elif self.base.is_extended_nonpositive:
            if self.exp.is_odd:
                return False
        elif self.base.is_imaginary:
            if self.exp.is_integer:
                m = self.exp % 4
                if m.is_zero:
                    return True
                if m.is_integer and m.is_zero is False:
                    return False
            if self.exp.is_imaginary:
                from sympy.functions.elementary.exponential import log
                return log(self.base).is_imaginary

    def _eval_is_extended_negative(self):
        if self.exp is S.Half:
            if self.base.is_complex or self.base.is_extended_real:
                return False
        if self.base.is_extended_negative:
            if self.exp.is_odd and self.base.is_finite:
                return True
            if self.exp.is_even:
                return False
        elif self.base.is_extended_positive:
            if self.exp.is_extended_real:
                return False
        elif self.base.is_zero:
            if self.exp.is_extended_real:
                return False
        elif self.base.is_extended_nonnegative:
            if self.exp.is_extended_nonnegative:
                return False
        elif self.base.is_extended_nonpositive:
            if self.exp.is_even:
                return False
        elif self.base.is_extended_real:
            if self.exp.is_even:
                return False

    def _eval_is_zero(self):
        if self.base.is_zero:
            if self.exp.is_extended_positive:
                return True
            elif self.exp.is_extended_nonpositive:
                return False
        elif self.base == S.Exp1:
            return self.exp is S.NegativeInfinity
        elif self.base.is_zero is False:
            if self.base.is_finite and self.exp.is_finite:
                return False
            elif self.exp.is_negative:
                return self.base.is_infinite
            elif self.exp.is_nonnegative:
                return False
            elif self.exp.is_infinite and self.exp.is_extended_real:
                if (1 - abs(self.base)).is_extended_positive:
                    return self.exp.is_extended_positive
                elif (1 - abs(self.base)).is_extended_negative:
                    return self.exp.is_extended_negative
        elif self.base.is_finite and self.exp.is_negative:
            # when self.base.is_zero is None
            return False

    def _eval_is_integer(self):
        b, e = self.args
        if b.is_rational:
            if b.is_integer is False and e.is_positive:
                return False  # rat**nonneg
        if b.is_integer and e.is_integer:
            if b is S.NegativeOne:
                return True
            if e.is_nonnegative or e.is_positive:
                return True
        if b.is_integer and e.is_negative and (e.is_finite or e.is_integer):
            if fuzzy_not((b - 1).is_zero) and fuzzy_not((b + 1).is_zero):
                return False
        if b.is_Number and e.is_Number:
            check = self.func(*self.args)
            return check.is_Integer
        if e.is_negative and b.is_positive and (b - 1).is_positive:
            return False
        if e.is_negative and b.is_negative and (b + 1).is_negative:
            return False

    def _eval_is_extended_real(self):
        if self.base is S.Exp1:
            if self.exp.is_extended_real:
                return True
            elif self.exp.is_imaginary:
                return (2*S.ImaginaryUnit*self.exp/S.Pi).is_even

        from sympy.functions.elementary.exponential import log, exp
        real_b = self.base.is_extended_real
        if real_b is None:
            if self.base.func == exp and self.base.exp.is_imaginary:
                return self.exp.is_imaginary
            if self.base.func == Pow and self.base.base is S.Exp1 and self.base.exp.is_imaginary:
                return self.exp.is_imaginary
            return
        real_e = self.exp.is_extended_real
        if real_e is None:
            return
        if real_b and real_e:
            if self.base.is_extended_positive:
                return True
            elif self.base.is_extended_nonnegative and self.exp.is_extended_nonnegative:
                return True
            elif self.exp.is_integer and self.base.is_extended_nonzero:
                return True
            elif self.exp.is_integer and self.exp.is_nonnegative:
                return True
            elif self.base.is_extended_negative:
                if self.exp.is_Rational:
                    return False
        if real_e and self.exp.is_extended_negative and self.base.is_zero is False:
            return Pow(self.base, -self.exp).is_extended_real
        im_b = self.base.is_imaginary
        im_e = self.exp.is_imaginary
        if im_b:
            if self.exp.is_integer:
                if self.exp.is_even:
                    return True
                elif self.exp.is_odd:
                    return False
            elif im_e and log(self.base).is_imaginary:
                return True
            elif self.exp.is_Add:
                c, a = self.exp.as_coeff_Add()
                if c and c.is_Integer:
                    return Mul(
                        self.base**c, self.base**a, evaluate=False).is_extended_real
            elif self.base in (-S.ImaginaryUnit, S.ImaginaryUnit):
                if (self.exp/2).is_integer is False:
                    return False
        if real_b and im_e:
            if self.base is S.NegativeOne:
                return True
            c = self.exp.coeff(S.ImaginaryUnit)
            if c:
                if self.base.is_rational and c.is_rational:
                    if self.base.is_nonzero and (self.base - 1).is_nonzero and c.is_nonzero:
                        return False
                ok = (c*log(self.base)/S.Pi).is_integer
                if ok is not None:
                    return ok

        if real_b is False and real_e: # we already know it's not imag
            if isinstance(self.exp, Rational) and self.exp.p == 1:
                return False
            from sympy.functions.elementary.complexes import arg
            i = arg(self.base)*self.exp/S.Pi
            if i.is_complex: # finite
                return i.is_integer

    def _eval_is_complex(self):

        if self.base == S.Exp1:
            return fuzzy_or([self.exp.is_complex, self.exp.is_extended_negative])

        if all(a.is_complex for a in self.args) and self._eval_is_finite():
            return True

    def _eval_is_imaginary(self):
        if self.base.is_commutative is False:
            return False

        if self.base.is_imaginary:
            if self.exp.is_integer:
                odd = self.exp.is_odd
                if odd is not None:
                    return odd
                return

        if self.base == S.Exp1:
            f = 2 * self.exp / (S.Pi*S.ImaginaryUnit)
            # exp(pi*integer) = 1 or -1, so not imaginary
            if f.is_even:
                return False
            # exp(pi*integer + pi/2) = I or -I, so it is imaginary
            if f.is_odd:
                return True
            return None

        if self.exp.is_imaginary:
            from sympy.functions.elementary.exponential import log
            imlog = log(self.base).is_imaginary
            if imlog is not None:
                return False  # I**i -> real; (2*I)**i -> complex ==> not imaginary

        if self.base.is_extended_real and self.exp.is_extended_real:
            if self.base.is_positive:
                return False
            else:
                rat = self.exp.is_rational
                if not rat:
                    return rat
                if self.exp.is_integer:
                    return False
                else:
                    half = (2*self.exp).is_integer
                    if half:
                        return self.base.is_negative
                    return half

        if self.base.is_extended_real is False:  # we already know it's not imag
            from sympy.functions.elementary.complexes import arg
            i = arg(self.base)*self.exp/S.Pi
            isodd = (2*i).is_odd
            if isodd is not None:
                return isodd

    def _eval_is_odd(self):
        if self.exp.is_integer:
            if self.exp.is_positive:
                return self.base.is_odd
            elif self.exp.is_nonnegative and self.base.is_odd:
                return True
            elif self.base is S.NegativeOne:
                return True

    def _eval_is_finite(self):
        if self.exp.is_negative:
            if self.base.is_zero:
                return False
            if self.base.is_infinite or self.base.is_nonzero:
                return True
        c1 = self.base.is_finite
        if c1 is None:
            return
        c2 = self.exp.is_finite
        if c2 is None:
            return
        if c1 and c2:
            if self.exp.is_nonnegative or fuzzy_not(self.base.is_zero):
                return True

    def _eval_is_prime(self):
        '''
        An integer raised to the n(>=2)-th power cannot be a prime.
        '''
        if self.base.is_integer and self.exp.is_integer and (self.exp - 1).is_positive:
            return False

    def _eval_is_composite(self):
        """
        A power is composite if both base and exponent are greater than 1
        """
        if (self.base.is_integer and self.exp.is_integer and
            ((self.base - 1).is_positive and (self.exp - 1).is_positive or
            (self.base + 1).is_negative and self.exp.is_positive and self.exp.is_even)):
            return True

    def _eval_is_polar(self):
        return self.base.is_polar

    def _eval_subs(self, old, new):
        from sympy.calculus.accumulationbounds import AccumBounds

        if isinstance(self.exp, AccumBounds):
            b = self.base.subs(old, new)
            e = self.exp.subs(old, new)
            if isinstance(e, AccumBounds):
                return e.__rpow__(b)
            return self.func(b, e)

        from sympy.functions.elementary.exponential import exp, log

        def _check(ct1, ct2, old):
            """Return (bool, pow, remainder_pow) where, if bool is True, then the
            exponent of Pow `old` will combine with `pow` so the substitution
            is valid, otherwise bool will be False.

            For noncommutative objects, `pow` will be an integer, and a factor
            `Pow(old.base, remainder_pow)` needs to be included. If there is
            no such factor, None is returned. For commutative objects,
            remainder_pow is always None.

            cti are the coefficient and terms of an exponent of self or old
            In this _eval_subs routine a change like (b**(2*x)).subs(b**x, y)
            will give y**2 since (b**x)**2 == b**(2*x); if that equality does
            not hold then the substitution should not occur so `bool` will be
            False.

            """
            coeff1, terms1 = ct1
            coeff2, terms2 = ct2
            if terms1 == terms2:
                if old.is_commutative:
                    # Allow fractional powers for commutative objects
                    pow = coeff1/coeff2
                    try:
                        as_int(pow, strict=False)
                        combines = True
                    except ValueError:
                        b, e = old.as_base_exp()
                        # These conditions ensure that (b**e)**f == b**(e*f) for any f
                        combines = b.is_positive and e.is_real or b.is_nonnegative and e.is_nonnegative

                    return combines, pow, None
                else:
                    # With noncommutative symbols, substitute only integer powers
                    if not isinstance(terms1, tuple):
                        terms1 = (terms1,)
                    if not all(term.is_integer for term in terms1):
                        return False, None, None

                    try:
                        # Round pow toward zero
                        pow, remainder = divmod(as_int(coeff1), as_int(coeff2))
                        if pow < 0 and remainder != 0:
                            pow += 1
                            remainder -= as_int(coeff2)

                        if remainder == 0:
                            remainder_pow = None
                        else:
                            remainder_pow = Mul(remainder, *terms1)

                        return True, pow, remainder_pow
                    except ValueError:
                        # Can't substitute
                        pass

            return False, None, None

        if old == self.base or (old == exp and self.base == S.Exp1):
            if new.is_Function and isinstance(new, Callable):
                return new(self.exp._subs(old, new))
            else:
                return new**self.exp._subs(old, new)

        # issue 10829: (4**x - 3*y + 2).subs(2**x, y) -> y**2 - 3*y + 2
        if isinstance(old, self.func) and self.exp == old.exp:
            l = log(self.base, old.base)
            if l.is_Number:
                return Pow(new, l)

        if isinstance(old, self.func) and self.base == old.base:
            if self.exp.is_Add is False:
                ct1 = self.exp.as_independent(Symbol, as_Add=False)
                ct2 = old.exp.as_independent(Symbol, as_Add=False)
                ok, pow, remainder_pow = _check(ct1, ct2, old)
                if ok:
                    # issue 5180: (x**(6*y)).subs(x**(3*y),z)->z**2
                    result = self.func(new, pow)
                    if remainder_pow is not None:
                        result = Mul(result, Pow(old.base, remainder_pow))
                    return result
            else:  # b**(6*x + a).subs(b**(3*x), y) -> y**2 * b**a
                # exp(exp(x) + exp(x**2)).subs(exp(exp(x)), w) -> w * exp(exp(x**2))
                oarg = old.exp
                new_l = []
                o_al = []
                ct2 = oarg.as_coeff_mul()
                for a in self.exp.args:
                    newa = a._subs(old, new)
                    ct1 = newa.as_coeff_mul()
                    ok, pow, remainder_pow = _check(ct1, ct2, old)
                    if ok:
                        new_l.append(new**pow)
                        if remainder_pow is not None:
                            o_al.append(remainder_pow)
                        continue
                    elif not old.is_commutative and not newa.is_integer:
                        # If any term in the exponent is non-integer,
                        # we do not do any substitutions in the noncommutative case
                        return
                    o_al.append(newa)
                if new_l:
                    expo = Add(*o_al)
                    new_l.append(Pow(self.base, expo, evaluate=False) if expo != 1 else self.base)
                    return Mul(*new_l)

        if (isinstance(old, exp) or (old.is_Pow and old.base is S.Exp1)) and self.exp.is_extended_real and self.base.is_positive:
            ct1 = old.exp.as_independent(Symbol, as_Add=False)
            ct2 = (self.exp*log(self.base)).as_independent(
                Symbol, as_Add=False)
            ok, pow, remainder_pow = _check(ct1, ct2, old)
            if ok:
                result = self.func(new, pow)  # (2**x).subs(exp(x*log(2)), z) -> z
                if remainder_pow is not None:
                    result = Mul(result, Pow(old.base, remainder_pow))
                return result

    def as_base_exp(self):
        """Return base and exp of self.

        Explanation
        ===========

        If base a Rational less than 1, then return 1/Rational, -exp.
        If this extra processing is not needed, the base and exp
        properties will give the raw arguments.

        Examples
        ========

        >>> from sympy import Pow, S
        >>> p = Pow(S.Half, 2, evaluate=False)
        >>> p.as_base_exp()
        (2, -2)
        >>> p.args
        (1/2, 2)
        >>> p.base, p.exp
        (1/2, 2)

        """
        b, e = self.args
        if b.is_Rational and b.p == 1 and b.q != 1:
            return Integer(b.q), -e
        return b, e

    def _eval_adjoint(self):
        from sympy.functions.elementary.complexes import adjoint
        i, p = self.exp.is_integer, self.base.is_positive
        if i:
            return adjoint(self.base)**self.exp
        if p:
            return self.base**adjoint(self.exp)
        if i is False and p is False:
            expanded = expand_complex(self)
            if expanded != self:
                return adjoint(expanded)

    def _eval_conjugate(self):
        from sympy.functions.elementary.complexes import conjugate as c
        i, p = self.exp.is_integer, self.base.is_positive
        if i:
            return c(self.base)**self.exp
        if p:
            return self.base**c(self.exp)
        if i is False and p is False:
            expanded = expand_complex(self)
            if expanded != self:
                return c(expanded)
        if self.is_extended_real:
            return self

    def _eval_transpose(self):
        from sympy.functions.elementary.complexes import transpose
        if self.base == S.Exp1:
            return self.func(S.Exp1, self.exp.transpose())
        i, p = self.exp.is_integer, (self.base.is_complex or self.base.is_infinite)
        if p:
            return self.base**self.exp
        if i:
            return transpose(self.base)**self.exp
        if i is False and p is False:
            expanded = expand_complex(self)
            if expanded != self:
                return transpose(expanded)

    def _eval_expand_power_exp(self, **hints):
        """a**(n + m) -> a**n*a**m"""
        b = self.base
        e = self.exp
        if b == S.Exp1:
            from sympy.concrete.summations import Sum
            if isinstance(e, Sum) and e.is_commutative:
                from sympy.concrete.products import Product
                return Product(self.func(b, e.function), *e.limits)
        if e.is_Add and (hints.get('force', False) or
                b.is_zero is False or e._all_nonneg_or_nonppos()):
            if e.is_commutative:
                return Mul(*[self.func(b, x) for x in e.args])
            if b.is_commutative:
                c, nc = sift(e.args, lambda x: x.is_commutative, binary=True)
                if c:
                    return Mul(*[self.func(b, x) for x in c]
                        )*b**Add._from_args(nc)
        return self

    def _eval_expand_power_base(self, **hints):
        """(a*b)**n -> a**n * b**n"""
        force = hints.get('force', False)

        b = self.base
        e = self.exp
        if not b.is_Mul:
            return self

        cargs, nc = b.args_cnc(split_1=False)

        # expand each term - this is top-level-only
        # expansion but we have to watch out for things
        # that don't have an _eval_expand method
        if nc:
            nc = [i._eval_expand_power_base(**hints)
                if hasattr(i, '_eval_expand_power_base') else i
                for i in nc]

            if e.is_Integer:
                if e.is_positive:
                    rv = Mul(*nc*e)
                else:
                    rv = Mul(*[i**-1 for i in nc[::-1]]*-e)
                if cargs:
                    rv *= Mul(*cargs)**e
                return rv

            if not cargs:
                return self.func(Mul(*nc), e, evaluate=False)

            nc = [Mul(*nc)]

        # sift the commutative bases
        other, maybe_real = sift(cargs, lambda x: x.is_extended_real is False,
            binary=True)
        def pred(x):
            if x is S.ImaginaryUnit:
                return S.ImaginaryUnit
            polar = x.is_polar
            if polar:
                return True
            if polar is None:
                return fuzzy_bool(x.is_extended_nonnegative)
        sifted = sift(maybe_real, pred)
        nonneg = sifted[True]
        other += sifted[None]
        neg = sifted[False]
        imag = sifted[S.ImaginaryUnit]
        if imag:
            I = S.ImaginaryUnit
            i = len(imag) % 4
            if i == 0:
                pass
            elif i == 1:
                other.append(I)
            elif i == 2:
                if neg:
                    nonn = -neg.pop()
                    if nonn is not S.One:
                        nonneg.append(nonn)
                else:
                    neg.append(S.NegativeOne)
            else:
                if neg:
                    nonn = -neg.pop()
                    if nonn is not S.One:
                        nonneg.append(nonn)
                else:
                    neg.append(S.NegativeOne)
                other.append(I)
            del imag

        # bring out the bases that can be separated from the base

        if force or e.is_integer:
            # treat all commutatives the same and put nc in other
            cargs = nonneg + neg + other
            other = nc
        else:
            # this is just like what is happening automatically, except
            # that now we are doing it for an arbitrary exponent for which
            # no automatic expansion is done

            assert not e.is_Integer

            # handle negatives by making them all positive and putting
            # the residual -1 in other
            if len(neg) > 1:
                o = S.One
                if not other and neg[0].is_Number:
                    o *= neg.pop(0)
                if len(neg) % 2:
                    o = -o
                for n in neg:
                    nonneg.append(-n)
                if o is not S.One:
                    other.append(o)
            elif neg and other:
                if neg[0].is_Number and neg[0] is not S.NegativeOne:
                    other.append(S.NegativeOne)
                    nonneg.append(-neg[0])
                else:
                    other.extend(neg)
            else:
                other.extend(neg)
            del neg

            cargs = nonneg
            other += nc

        rv = S.One
        if cargs:
            if e.is_Rational:
                npow, cargs = sift(cargs, lambda x: x.is_Pow and
                    x.exp.is_Rational and x.base.is_number,
                    binary=True)
                rv = Mul(*[self.func(b.func(*b.args), e) for b in npow])
            rv *= Mul(*[self.func(b, e, evaluate=False) for b in cargs])
        if other:
            rv *= self.func(Mul(*other), e, evaluate=False)
        return rv

    def _eval_expand_multinomial(self, **hints):
        """(a + b + ..)**n -> a**n + n*a**(n-1)*b + .., n is nonzero integer"""

        base, exp = self.args
        result = self

        if exp.is_Rational and exp.p > 0 and base.is_Add:
            if not exp.is_Integer:
                n = Integer(exp.p // exp.q)

                if not n:
                    return result
                else:
                    radical, result = self.func(base, exp - n), []

                    expanded_base_n = self.func(base, n)
                    if expanded_base_n.is_Pow:
                        expanded_base_n = \
                            expanded_base_n._eval_expand_multinomial()
                    for term in Add.make_args(expanded_base_n):
                        result.append(term*radical)

                    return Add(*result)

            n = int(exp)

            if base.is_commutative:
                order_terms, other_terms = [], []

                for b in base.args:
                    if b.is_Order:
                        order_terms.append(b)
                    else:
                        other_terms.append(b)

                if order_terms:
                    # (f(x) + O(x^n))^m -> f(x)^m + m*f(x)^{m-1} *O(x^n)
                    f = Add(*other_terms)
                    o = Add(*order_terms)

                    if n == 2:
                        return expand_multinomial(f**n, deep=False) + n*f*o
                    else:
                        g = expand_multinomial(f**(n - 1), deep=False)
                        return expand_mul(f*g, deep=False) + n*g*o

                if base.is_number:
                    # Efficiently expand expressions of the form (a + b*I)**n
                    # where 'a' and 'b' are real numbers and 'n' is integer.
                    a, b = base.as_real_imag()

                    if a.is_Rational and b.is_Rational:
                        if not a.is_Integer:
                            if not b.is_Integer:
                                k = self.func(a.q * b.q, n)
                                a, b = a.p*b.q, a.q*b.p
                            else:
                                k = self.func(a.q, n)
                                a, b = a.p, a.q*b
                        elif not b.is_Integer:
                            k = self.func(b.q, n)
                            a, b = a*b.q, b.p
                        else:
                            k = 1

                        a, b, c, d = int(a), int(b), 1, 0

                        while n:
                            if n & 1:
                                c, d = a*c - b*d, b*c + a*d
                                n -= 1
                            a, b = a*a - b*b, 2*a*b
                            n //= 2

                        I = S.ImaginaryUnit

                        if k == 1:
                            return c + I*d
                        else:
                            return Integer(c)/k + I*d/k

                p = other_terms
                # (x + y)**3 -> x**3 + 3*x**2*y + 3*x*y**2 + y**3
                # in this particular example:
                # p = [x,y]; n = 3
                # so now it's easy to get the correct result -- we get the
                # coefficients first:
                from sympy.ntheory.multinomial import multinomial_coefficients
                from sympy.polys.polyutils import basic_from_dict
                expansion_dict = multinomial_coefficients(len(p), n)
                # in our example: {(3, 0): 1, (1, 2): 3, (0, 3): 1, (2, 1): 3}
                # and now construct the expression.
