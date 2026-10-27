                        v = Array(v[0])
                        count = 1
                    else:
                        v, count = v
                        v = Array(v)
                else:
                    v, count = v
                if count == 0:
                    continue
                variable_count.append(Tuple(v, count))
                continue

            v = sympify(v)
            if isinstance(v, Integer):
                if i == 0:
                    raise ValueError("First variable cannot be a number: %i" % v)
                count = v
                prev, prevcount = variable_count[-1]
                if prevcount != 1:
                    raise TypeError("tuple {} followed by number {}".format((prev, prevcount), v))
                if count == 0:
                    variable_count.pop()
                else:
                    variable_count[-1] = Tuple(prev, count)
            else:
                count = 1
                variable_count.append(Tuple(v, count))

        # light evaluation of contiguous, identical
        # items: (x, 1), (x, 1) -> (x, 2)
        merged = []
        for t in variable_count:
            v, c = t
            if c.is_negative:
                raise ValueError(
                    'order of differentiation must be nonnegative')
            if merged and merged[-1][0] == v:
                c += merged[-1][1]
                if not c:
                    merged.pop()
                else:
                    merged[-1] = Tuple(v, c)
            else:
                merged.append(t)
        variable_count = merged

        # sanity check of variables of differentation; we waited
        # until the counts were computed since some variables may
        # have been removed because the count was 0
        for v, c in variable_count:
            # v must have _diff_wrt True
            if not v._diff_wrt:
                __ = ''  # filler to make error message neater
                raise ValueError(filldedent('''
                    Can't calculate derivative wrt %s.%s''' % (v,
                    __)))

        # We make a special case for 0th derivative, because there is no
        # good way to unambiguously print this.
        if len(variable_count) == 0:
            return expr

        evaluate = kwargs.get('evaluate', False)

        if evaluate:
            if isinstance(expr, Derivative):
                expr = expr.canonical
            variable_count = [
                (v.canonical if isinstance(v, Derivative) else v, c)
                for v, c in variable_count]

            # Look for a quick exit if there are symbols that don't appear in
            # expression at all. Note, this cannot check non-symbols like
            # Derivatives as those can be created by intermediate
            # derivatives.
            zero = False
            free = expr.free_symbols
            from sympy.matrices.expressions.matexpr import MatrixExpr

            for v, c in variable_count:
                vfree = v.free_symbols
                if c.is_positive and vfree:
                    if isinstance(v, AppliedUndef):
                        # these match exactly since
                        # x.diff(f(x)) == g(x).diff(f(x)) == 0
                        # and are not created by differentiation
                        D = Dummy()
                        if not expr.xreplace({v: D}).has(D):
                            zero = True
                            break
                    elif isinstance(v, MatrixExpr):
                        zero = False
                        break
                    elif isinstance(v, Symbol) and v not in free:
                        zero = True
                        break
                    else:
                        if not free & vfree:
                            # e.g. v is IndexedBase or Matrix
                            zero = True
                            break
            if zero:
                return cls._get_zero_with_shape_like(expr)

            # make the order of symbols canonical
            #TODO: check if assumption of discontinuous derivatives exist
            variable_count = cls._sort_variable_count(variable_count)

        # denest
        if isinstance(expr, Derivative):
            variable_count = list(expr.variable_count) + variable_count
            expr = expr.expr
            return _derivative_dispatch(expr, *variable_count, **kwargs)

        # we return here if evaluate is False or if there is no
        # _eval_derivative method
        if not evaluate or not hasattr(expr, '_eval_derivative'):
            # return an unevaluated Derivative
            if evaluate and variable_count == [(expr, 1)] and expr.is_scalar:
                # special hack providing evaluation for classes
                # that have defined is_scalar=True but have no
                # _eval_derivative defined
                return S.One
            return Expr.__new__(cls, expr, *variable_count)

        # evaluate the derivative by calling _eval_derivative method
        # of expr for each variable
        # -------------------------------------------------------------
        nderivs = 0  # how many derivatives were performed
        unhandled = []
        from sympy.matrices.matrixbase import MatrixBase
        for i, (v, count) in enumerate(variable_count):

            old_expr = expr
            old_v = None

            is_symbol = v.is_symbol or isinstance(v,
                (Iterable, Tuple, MatrixBase, NDimArray))

            if not is_symbol:
                old_v = v
                v = Dummy('xi')
                expr = expr.xreplace({old_v: v})
                # Derivatives and UndefinedFunctions are independent
                # of all others
                clashing = not (isinstance(old_v, (Derivative, AppliedUndef)))
                if v not in expr.free_symbols and not clashing:
                    return expr.diff(v)  # expr's version of 0
                if not old_v.is_scalar and not hasattr(
                        old_v, '_eval_derivative'):
                    # special hack providing evaluation for classes
                    # that have defined is_scalar=True but have no
                    # _eval_derivative defined
                    expr *= old_v.diff(old_v)

            obj = cls._dispatch_eval_derivative_n_times(expr, v, count)
            if obj is not None and obj.is_zero:
                return obj

            nderivs += count

            if old_v is not None:
                if obj is not None:
                    # remove the dummy that was used
                    obj = obj.subs(v, old_v)
                # restore expr
                expr = old_expr

            if obj is None:
                # we've already checked for quick-exit conditions
                # that give 0 so the remaining variables
                # are contained in the expression but the expression
                # did not compute a derivative so we stop taking
                # derivatives
                unhandled = variable_count[i:]
                break

            expr = obj

        # what we have so far can be made canonical
        expr = expr.replace(
            lambda x: isinstance(x, Derivative),
            lambda x: x.canonical)

        if unhandled:
            if isinstance(expr, Derivative):
                unhandled = list(expr.variable_count) + unhandled
                expr = expr.expr
            expr = Expr.__new__(cls, expr, *unhandled)

        if (nderivs > 1) == True and kwargs.get('simplify', True):
            from .exprtools import factor_terms
            from sympy.simplify.simplify import signsimp
            expr = factor_terms(signsimp(expr))
        return expr

    @property
    def canonical(cls):
        return cls.func(cls.expr,
            *Derivative._sort_variable_count(cls.variable_count))

    @classmethod
    def _sort_variable_count(cls, vc):
        """
        Sort (variable, count) pairs into canonical order while
        retaining order of variables that do not commute during
        differentiation:

        * symbols and functions commute with each other
        * derivatives commute with each other
        * a derivative does not commute with anything it contains
        * any other object is not allowed to commute if it has
          free symbols in common with another object

        Examples
        ========

        >>> from sympy import Derivative, Function, symbols
        >>> vsort = Derivative._sort_variable_count
        >>> x, y, z = symbols('x y z')
        >>> f, g, h = symbols('f g h', cls=Function)

        Contiguous items are collapsed into one pair:

        >>> vsort([(x, 1), (x, 1)])
        [(x, 2)]
        >>> vsort([(y, 1), (f(x), 1), (y, 1), (f(x), 1)])
        [(y, 2), (f(x), 2)]

        Ordering is canonical.

        >>> def vsort0(*v):
        ...     # docstring helper to
        ...     # change vi -> (vi, 0), sort, and return vi vals
        ...     return [i[0] for i in vsort([(i, 0) for i in v])]

        >>> vsort0(y, x)
        [x, y]
        >>> vsort0(g(y), g(x), f(y))
        [f(y), g(x), g(y)]

        Symbols are sorted as far to the left as possible but never
        move to the left of a derivative having the same symbol in
        its variables; the same applies to AppliedUndef which are
        always sorted after Symbols:

        >>> dfx = f(x).diff(x)
        >>> assert vsort0(dfx, y) == [y, dfx]
        >>> assert vsort0(dfx, x) == [dfx, x]
        """
        if not vc:
            return []
        vc = list(vc)
        if len(vc) == 1:
            return [Tuple(*vc[0])]
        V = list(range(len(vc)))
        E = []
        v = lambda i: vc[i][0]
        D = Dummy()
        def _block(d, v, wrt=False):
            # return True if v should not come before d else False
            if d == v:
                return wrt
            if d.is_Symbol:
                return False
            if isinstance(d, Derivative):
                # a derivative blocks if any of it's variables contain
                # v; the wrt flag will return True for an exact match
                # and will cause an AppliedUndef to block if v is in
                # the arguments
                if any(_block(k, v, wrt=True)
                        for k in d._wrt_variables):
                    return True
                return False
            if not wrt and isinstance(d, AppliedUndef):
                return False
            if v.is_Symbol:
                return v in d.free_symbols
            if isinstance(v, AppliedUndef):
                return _block(d.xreplace({v: D}), D)
            return d.free_symbols & v.free_symbols
        for i in range(len(vc)):
            for j in range(i):
                if _block(v(j), v(i)):
                    E.append((j,i))
        # this is the default ordering to use in case of ties
        O = dict(zip(ordered(uniq([i for i, c in vc])), range(len(vc))))
        ix = topological_sort((V, E), key=lambda i: O[v(i)])
        # merge counts of contiguously identical items
        merged = []
        for v, c in [vc[i] for i in ix]:
            if merged and merged[-1][0] == v:
                merged[-1][1] += c
            else:
                merged.append([v, c])
        return [Tuple(*i) for i in merged]

    def _eval_is_commutative(self):
        return self.expr.is_commutative

    def _eval_derivative(self, v):
        # If v (the variable of differentiation) is not in
        # self.variables, we might be able to take the derivative.
        if v not in self._wrt_variables:
            dedv = self.expr.diff(v)
            if isinstance(dedv, Derivative):
                return dedv.func(dedv.expr, *(self.variable_count + dedv.variable_count))
            # dedv (d(self.expr)/dv) could have simplified things such that the
            # derivative wrt things in self.variables can now be done. Thus,
            # we set evaluate=True to see if there are any other derivatives
            # that can be done. The most common case is when dedv is a simple
            # number so that the derivative wrt anything else will vanish.
            return self.func(dedv, *self.variables, evaluate=True)
        # In this case v was in self.variables so the derivative wrt v has
        # already been attempted and was not computed, either because it
        # couldn't be or evaluate=False originally.
        variable_count = list(self.variable_count)
        variable_count.append((v, 1))
        return self.func(self.expr, *variable_count, evaluate=False)

    def doit(self, **hints):
        expr = self.expr
        if hints.get('deep', True):
            expr = expr.doit(**hints)
        hints['evaluate'] = True
        rv = self.func(expr, *self.variable_count, **hints)
        if rv!= self and rv.has(Derivative):
            rv =  rv.doit(**hints)
        return rv

    @_sympifyit('z0', NotImplementedError)
    def doit_numerically(self, z0):
        """
        Evaluate the derivative at z numerically.

        When we can represent derivatives at a point, this should be folded
        into the normal evalf. For now, we need a special method.
        """
        if len(self.free_symbols) != 1 or len(self.variables) != 1:
            raise NotImplementedError('partials and higher order derivatives')
        z = list(self.free_symbols)[0]

        def eval(x):
            f0 = self.expr.subs(z, Expr._from_mpmath(x, prec=mpmath.mp.prec))
            f0 = f0.evalf(prec_to_dps(mpmath.mp.prec))
            return f0._to_mpmath(mpmath.mp.prec)
        return Expr._from_mpmath(mpmath.diff(eval,
                                             z0._to_mpmath(mpmath.mp.prec)),
                                 mpmath.mp.prec)

    @property
    def expr(self):
        return self._args[0]

    @property
    def _wrt_variables(self):
        # return the variables of differentiation without
        # respect to the type of count (int or symbolic)
        return [i[0] for i in self.variable_count]

    @property
    def variables(self):
        # TODO: deprecate?  YES, make this 'enumerated_variables' and
        #       name _wrt_variables as variables
        # TODO: support for `d^n`?
        rv = []
        for v, count in self.variable_count:
            if not count.is_Integer:
                raise TypeError(filldedent('''
                Cannot give expansion for symbolic count. If you just
                want a list of all variables of differentiation, use
                _wrt_variables.'''))
            rv.extend([v]*count)
        return tuple(rv)

    @property
    def variable_count(self):
        return self._args[1:]

    @property
    def derivative_count(self):
        return sum([count for _, count in self.variable_count], 0)

    @property
    def free_symbols(self):
        ret = self.expr.free_symbols
        # Add symbolic counts to free_symbols
        for _, count in self.variable_count:
            ret.update(count.free_symbols)
        return ret

    @property
    def kind(self):
        return self.args[0].kind

    def _eval_subs(self, old, new):
        # The substitution (old, new) cannot be done inside
        # Derivative(expr, vars) for a variety of reasons
        # as handled below.
        if old in self._wrt_variables:
            # first handle the counts
            expr = self.func(self.expr, *[(v, c.subs(old, new))
                for v, c in self.variable_count])
            if expr != self:
                return expr._eval_subs(old, new)
            # quick exit case
            if not getattr(new, '_diff_wrt', False):
                # case (0): new is not a valid variable of
                # differentiation
                if isinstance(old, Symbol):
                    # don't introduce a new symbol if the old will do
                    return Subs(self, old, new)
                else:
                    xi = Dummy('xi')
                    return Subs(self.xreplace({old: xi}), xi, new)

        # If both are Derivatives with the same expr, check if old is
        # equivalent to self or if old is a subderivative of self.
        if old.is_Derivative and old.expr == self.expr:
            if self.canonical == old.canonical:
                return new

            # collections.Counter doesn't have __le__
            def _subset(a, b):
                return all((a[i] <= b[i]) == True for i in a)

            old_vars = Counter(dict(reversed(old.variable_count)))
            self_vars = Counter(dict(reversed(self.variable_count)))
            if _subset(old_vars, self_vars):
                return _derivative_dispatch(new, *(self_vars - old_vars).items()).canonical

        args = list(self.args)
        newargs = [x._subs(old, new) for x in args]
        if args[0] == old:
            # complete replacement of self.expr
            # we already checked that the new is valid so we know
            # it won't be a problem should it appear in variables
            return _derivative_dispatch(*newargs)

        if newargs[0] != args[0]:
            # case (1) can't change expr by introducing something that is in
            # the _wrt_variables if it was already in the expr
            # e.g.
            # for Derivative(f(x, g(y)), y), x cannot be replaced with
            # anything that has y in it; for f(g(x), g(y)).diff(g(y))
            # g(x) cannot be replaced with anything that has g(y)
            syms = {vi: Dummy() for vi in self._wrt_variables
                if not vi.is_Symbol}
            wrt = {syms.get(vi, vi) for vi in self._wrt_variables}
            forbidden = args[0].xreplace(syms).free_symbols & wrt
            nfree = new.xreplace(syms).free_symbols
            ofree = old.xreplace(syms).free_symbols
            if (nfree - ofree) & forbidden:
                return Subs(self, old, new)

        viter = ((i, j) for ((i, _), (j, _)) in zip(newargs[1:], args[1:]))
        if any(i != j for i, j in viter):  # a wrt-variable change
            # case (2) can't change vars by introducing a variable
            # that is contained in expr, e.g.
            # for Derivative(f(z, g(h(x), y)), y), y cannot be changed to
            # x, h(x), or g(h(x), y)
            for a in _atomic(self.expr, recursive=True):
                for i in range(1, len(newargs)):
                    vi, _ = newargs[i]
                    if a == vi and vi != args[i][0]:
                        return Subs(self, old, new)
            # more arg-wise checks
            vc = newargs[1:]
            oldv = self._wrt_variables
            newe = self.expr
            subs = []
            for i, (vi, ci) in enumerate(vc):
                if not vi._diff_wrt:
                    # case (3) invalid differentiation expression so
                    # create a replacement dummy
                    xi = Dummy('xi_%i' % i)
                    # replace the old valid variable with the dummy
                    # in the expression
                    newe = newe.xreplace({oldv[i]: xi})
                    # and replace the bad variable with the dummy
                    vc[i] = (xi, ci)
                    # and record the dummy with the new (invalid)
                    # differentiation expression
                    subs.append((xi, vi))

            if subs:
                # handle any residual substitution in the expression
                newe = newe._subs(old, new)
                # return the Subs-wrapped derivative
                return Subs(Derivative(newe, *vc), *zip(*subs))

        # everything was ok
        return _derivative_dispatch(*newargs)

    def _eval_lseries(self, x, logx, cdir=0):
        dx = self.variables
        for term in self.expr.lseries(x, logx=logx, cdir=cdir):
            yield self.func(term, *dx)

    def _eval_nseries(self, x, n, logx, cdir=0):
        arg = self.expr.nseries(x, n=n, logx=logx)
        o = arg.getO()
        dx = self.variables
        rv = [self.func(a, *dx) for a in Add.make_args(arg.removeO())]
        if o:
            rv.append(o/x)
        return Add(*rv)

    def _eval_as_leading_term(self, x, logx, cdir):
        series_gen = self.expr.lseries(x)
        d = S.Zero
        for leading_term in series_gen:
            d = diff(leading_term, *self.variables)
            if d != 0:
                break
        return d

    def as_finite_difference(self, points=1, x0=None, wrt=None):
        """ Expresses a Derivative instance as a finite difference.

        Parameters
        ==========

        points : sequence or coefficient, optional
            If sequence: discrete values (length >= order+1) of the
            independent variable used for generating the finite
            difference weights.
            If it is a coefficient, it will be used as the step-size
            for generating an equidistant sequence of length order+1
            centered around ``x0``. Default: 1 (step-size 1)

        x0 : number or Symbol, optional
            the value of the independent variable (``wrt``) at which the
            derivative is to be approximated. Default: same as ``wrt``.

        wrt : Symbol, optional
            "with respect to" the variable for which the (partial)
            derivative is to be approximated for. If not provided it
            is required that the derivative is ordinary. Default: ``None``.


        Examples
        ========

        >>> from sympy import symbols, Function, exp, sqrt, Symbol
        >>> x, h = symbols('x h')
        >>> f = Function('f')
        >>> f(x).diff(x).as_finite_difference()
        -f(x - 1/2) + f(x + 1/2)

        The default step size and number of points are 1 and
        ``order + 1`` respectively. We can change the step size by
        passing a symbol as a parameter:

        >>> f(x).diff(x).as_finite_difference(h)
        -f(-h/2 + x)/h + f(h/2 + x)/h

        We can also specify the discretized values to be used in a
        sequence:

        >>> f(x).diff(x).as_finite_difference([x, x+h, x+2*h])
        -3*f(x)/(2*h) + 2*f(h + x)/h - f(2*h + x)/(2*h)

        The algorithm is not restricted to use equidistant spacing, nor
        do we need to make the approximation around ``x0``, but we can get
        an expression estimating the derivative at an offset:

        >>> e, sq2 = exp(1), sqrt(2)
        >>> xl = [x-h, x+h, x+e*h]
        >>> f(x).diff(x, 1).as_finite_difference(xl, x+h*sq2)  # doctest: +ELLIPSIS
        2*h*((h + sqrt(2)*h)/(2*h) - (-sqrt(2)*h + h)/(2*h))*f(E*h + x)/...

        To approximate ``Derivative`` around ``x0`` using a non-equidistant
        spacing step, the algorithm supports assignment of undefined
        functions to ``points``:

        >>> dx = Function('dx')
        >>> f(x).diff(x).as_finite_difference(points=dx(x), x0=x-h)
        -f(-h + x - dx(-h + x)/2)/dx(-h + x) + f(-h + x + dx(-h + x)/2)/dx(-h + x)

        Partial derivatives are also supported:

        >>> y = Symbol('y')
        >>> d2fdxdy=f(x,y).diff(x,y)
        >>> d2fdxdy.as_finite_difference(wrt=x)
        -Derivative(f(x - 1/2, y), y) + Derivative(f(x + 1/2, y), y)

        We can apply ``as_finite_difference`` to ``Derivative`` instances in
        compound expressions using ``replace``:

        >>> (1 + 42**f(x).diff(x)).replace(lambda arg: arg.is_Derivative,
        ...     lambda arg: arg.as_finite_difference())
        42**(-f(x - 1/2) + f(x + 1/2)) + 1


        See also
        ========

        sympy.calculus.finite_diff.apply_finite_diff
        sympy.calculus.finite_diff.differentiate_finite
        sympy.calculus.finite_diff.finite_diff_weights

        """
        from sympy.calculus.finite_diff import _as_finite_diff
        return _as_finite_diff(self, points, x0, wrt)

    @classmethod
    def _get_zero_with_shape_like(cls, expr):
        return S.Zero

    @classmethod
    def _dispatch_eval_derivative_n_times(cls, expr, v, count):
        # Evaluate the derivative `n` times.  If
        # `_eval_derivative_n_times` is not overridden by the current
        # object, the default in `Basic` will call a loop over
        # `_eval_derivative`:
        return expr._eval_derivative_n_times(v, count)


def _derivative_dispatch(expr, *variables, **kwargs):
    from sympy.matrices.matrixbase import MatrixBase
    from sympy.matrices.expressions.matexpr import MatrixExpr
    from sympy.tensor.array import NDimArray
    array_types = (MatrixBase, MatrixExpr, NDimArray, list, tuple, Tuple)
    if isinstance(expr, array_types) or any(isinstance(i[0], array_types) if isinstance(i, (tuple, list, Tuple)) else isinstance(i, array_types) for i in variables):
        from sympy.tensor.array.array_derivatives import ArrayDerivative
        return ArrayDerivative(expr, *variables, **kwargs)
    return Derivative(expr, *variables, **kwargs)


class Lambda(Expr):
    """
    Lambda(x, expr) represents a lambda function similar to Python's
    'lambda x: expr'. A function of several variables is written as
    Lambda((x, y, ...), expr).

    Examples
    ========

    A simple example:

    >>> from sympy import Lambda
    >>> from sympy.abc import x
    >>> f = Lambda(x, x**2)
    >>> f(4)
    16

    For multivariate functions, use:

    >>> from sympy.abc import y, z, t
    >>> f2 = Lambda((x, y, z, t), x + y**z + t**z)
    >>> f2(1, 2, 3, 4)
    73

    It is also possible to unpack tuple arguments:

    >>> f = Lambda(((x, y), z), x + y + z)
    >>> f((1, 2), 3)
    6

    A handy shortcut for lots of arguments:

    >>> p = x, y, z
    >>> f = Lambda(p, x + y*z)
    >>> f(*p)
    x + y*z

    """
    is_Function = True

    def __new__(cls, signature, expr) -> Lambda:
        if iterable(signature) and not isinstance(signature, (tuple, Tuple)):
            sympy_deprecation_warning(
                """
                Using a non-tuple iterable as the first argument to Lambda
                is deprecated. Use Lambda(tuple(args), expr) instead.
                """,
                deprecated_since_version="1.5",
                active_deprecations_target="deprecated-non-tuple-lambda",
            )
            signature = tuple(signature)
        _sig = signature if iterable(signature) else (signature,)
        sig: Tuple = sympify(_sig) # type: ignore
        cls._check_signature(sig)

        if len(sig) == 1 and sig[0] == expr:
            return S.IdentityFunction

        return Expr.__new__(cls, sig, sympify(expr))

    @classmethod
    def _check_signature(cls, sig):
        syms = set()

        def rcheck(args):
            for a in args:
                if a.is_symbol:
                    if a in syms:
                        raise BadSignatureError("Duplicate symbol %s" % a)
                    syms.add(a)
                elif isinstance(a, Tuple):
                    rcheck(a)
                else:
                    raise BadSignatureError("Lambda signature should be only tuples"
                        " and symbols, not %s" % a)

        if not isinstance(sig, Tuple):
            raise BadSignatureError("Lambda signature should be a tuple not %s" % sig)
        # Recurse through the signature:
        rcheck(sig)

    @property
    def signature(self):
        """The expected form of the arguments to be unpacked into variables"""
        return self._args[0]

    @property
    def expr(self):
        """The return value of the function"""
        return self._args[1]

    @property
    def variables(self):
        """The variables used in the internal representation of the function"""
        def _variables(args):
            if isinstance(args, Tuple):
                for arg in args:
                    yield from _variables(arg)
            else:
                yield args
        return tuple(_variables(self.signature))

    @property
    def nargs(self):
        from sympy.sets.sets import FiniteSet
        return FiniteSet(len(self.signature))

    bound_symbols = variables

    @property
    def free_symbols(self):
        return self.expr.free_symbols - set(self.variables)

    def __call__(self, *args):
        n = len(args)
        if n not in self.nargs:  # Lambda only ever has 1 value in nargs
            # XXX: exception message must be in exactly this format to
            # make it work with NumPy's functions like vectorize(). See,
            # for example, https://github.com/numpy/numpy/issues/1697.
            # The ideal solution would be just to attach metadata to
            # the exception and change NumPy to take advantage of this.
            ## XXX does this apply to Lambda? If not, remove this comment.
            temp = ('%(name)s takes exactly %(args)s '
                   'argument%(plural)s (%(given)s given)')
            raise BadArgumentsError(temp % {
                'name': self,
                'args': list(self.nargs)[0],
                'plural': 's'*(list(self.nargs)[0] != 1),
                'given': n})

        d = self._match_signature(self.signature, args)

        return self.expr.xreplace(d)

    def _match_signature(self, sig, args):

        symargmap = {}

        def rmatch(pars, args):
            for par, arg in zip(pars, args):
                if par.is_symbol:
                    symargmap[par] = arg
                elif isinstance(par, Tuple):
                    if not isinstance(arg, (tuple, Tuple)) or len(args) != len(pars):
                        raise BadArgumentsError("Can't match %s and %s" % (args, pars))
                    rmatch(par, arg)

        rmatch(sig, args)

        return symargmap

    @property
    def is_identity(self):
        """Return ``True`` if this ``Lambda`` is an identity function. """
        return self.signature == self.expr

    def _eval_evalf(self, prec):
        return self.func(self.args[0], self.args[1].evalf(n=prec_to_dps(prec)))


class Subs(Expr):
    """
    Represents unevaluated substitutions of an expression.

    ``Subs(expr, x, x0)`` represents the expression resulting
    from substituting x with x0 in expr.

    Parameters
    ==========

    expr : Expr
        An expression.

    x : tuple, variable
        A variable or list of distinct variables.

    x0 : tuple or list of tuples
        A point or list of evaluation points
        corresponding to those variables.

    Examples
    ========

    >>> from sympy import Subs, Function, sin, cos
    >>> from sympy.abc import x, y, z
    >>> f = Function('f')

    Subs are created when a particular substitution cannot be made. The
    x in the derivative cannot be replaced with 0 because 0 is not a
    valid variables of differentiation:

    >>> f(x).diff(x).subs(x, 0)
    Subs(Derivative(f(x), x), x, 0)

    Once f is known, the derivative and evaluation at 0 can be done:

    >>> _.subs(f, sin).doit() == sin(x).diff(x).subs(x, 0) == cos(0)
    True

    Subs can also be created directly with one or more variables:

    >>> Subs(f(x)*sin(y) + z, (x, y), (0, 1))
    Subs(z + f(x)*sin(y), (x, y), (0, 1))
    >>> _.doit()
    z + f(0)*sin(1)

    Notes
    =====

    ``Subs`` objects are generally useful to represent unevaluated derivatives
    calculated at a point.

    The variables may be expressions, but they are subjected to the limitations
    of subs(), so it is usually a good practice to use only symbols for
    variables, since in that case there can be no ambiguity.

    There's no automatic expansion - use the method .doit() to effect all
    possible substitutions of the object and also of objects inside the
    expression.

    When evaluating derivatives at a point that is not a symbol, a Subs object
    is returned. One is also able to calculate derivatives of Subs objects - in
    this case the expression is always expanded (for the unevaluated form, use
    Derivative()).

    In order to allow expressions to combine before doit is done, a
    representation of the Subs expression is used internally to make
    expressions that are superficially different compare the same:

    >>> a, b = Subs(x, x, 0), Subs(y, y, 0)
    >>> a + b
    2*Subs(x, x, 0)

    This can lead to unexpected consequences when using methods
    like `has` that are cached:

    >>> s = Subs(x, x, 0)
    >>> s.has(x), s.has(y)
    (True, False)
    >>> ss = s.subs(x, y)
    >>> ss.has(x), ss.has(y)
    (True, False)
    >>> s, ss
    (Subs(x, x, 0), Subs(y, y, 0))
    """
    def __new__(cls, expr, variables, point, **assumptions):
        if not is_sequence(variables, Tuple):
            variables = [variables]
        variables = Tuple(*variables)

        if has_dups(variables):
            repeated = [str(v) for v, i in Counter(variables).items() if i > 1]
            __ = ', '.join(repeated)
            raise ValueError(filldedent('''
                The following expressions appear more than once: %s
                ''' % __))

        point = Tuple(*(point if is_sequence(point, Tuple) else [point]))

        if len(point) != len(variables):
            raise ValueError('Number of point values must be the same as '
                             'the number of variables.')

        if not point:
            return sympify(expr)

        # denest
        if isinstance(expr, Subs):
            variables = expr.variables + variables
            point = expr.point + point
            expr = expr.expr
        else:
            expr = sympify(expr)

        # use symbols with names equal to the point value (with prepended _)
        # to give a variable-independent expression
        pre = "_"
        pts = sorted(set(point), key=default_sort_key)
        from sympy.printing.str import StrPrinter
        class CustomStrPrinter(StrPrinter):
            def _print_Dummy(self, expr):
                return str(expr) + str(expr.dummy_index)
        def mystr(expr, **settings):
            p = CustomStrPrinter(settings)
            return p.doprint(expr)
        while 1:
            s_pts = {p: Symbol(pre + mystr(p)) for p in pts}
            reps = [(v, s_pts[p])
                for v, p in zip(variables, point)]
            # if any underscore-prepended symbol is already a free symbol
            # and is a variable with a different point value, then there
            # is a clash, e.g. _0 clashes in Subs(_0 + _1, (_0, _1), (1, 0))
            # because the new symbol that would be created is _1 but _1
            # is already mapped to 0 so __0 and __1 are used for the new
            # symbols
            if any(r in expr.free_symbols and
                   r in variables and
                   Symbol(pre + mystr(point[variables.index(r)])) != r
                   for _, r in reps):
                pre += "_"
                continue
            break

        obj = Expr.__new__(cls, expr, Tuple(*variables), point)
        obj._expr = expr.xreplace(dict(reps))
        return obj

    def _eval_is_commutative(self):
        return self.expr.is_commutative

    def doit(self, **hints):
        e, v, p = self.args

        # remove self mappings
        for i, (vi, pi) in enumerate(zip(v, p)):
            if vi == pi:
                v = v[:i] + v[i + 1:]
                p = p[:i] + p[i + 1:]
        if not v:
            return self.expr

        if isinstance(e, Derivative):
            # apply functions first, e.g. f -> cos
            undone = []
            for i, vi in enumerate(v):
                if isinstance(vi, FunctionClass):
                    e = e.subs(vi, p[i])
                else:
                    undone.append((vi, p[i]))
            if not isinstance(e, Derivative):
                e = e.doit()
            if isinstance(e, Derivative):
                # do Subs that aren't related to differentiation
                undone2 = []
                D = Dummy()
                arg = e.args[0]
                for vi, pi in undone:
                    if D not in e.xreplace({vi: D}).free_symbols:
                        if arg.has(vi):
                            e = e.subs(vi, pi)
                    else:
                        undone2.append((vi, pi))
                undone = undone2
                # differentiate wrt variables that are present
                wrt = []
                D = Dummy()
                expr = e.expr
                free = expr.free_symbols
                for vi, ci in e.variable_count:
                    if isinstance(vi, Symbol) and vi in free:
                        expr = expr.diff((vi, ci))
                    elif D in expr.subs(vi, D).free_symbols:
                        expr = expr.diff((vi, ci))
                    else:
                        wrt.append((vi, ci))
                # inject remaining subs
                rv = expr.subs(undone)
                # do remaining differentiation *in order given*
                for vc in wrt:
                    rv = rv.diff(vc)
            else:
                # inject remaining subs
                rv = e.subs(undone)
        else:
            rv = e.doit(**hints).subs(list(zip(v, p)))

        if hints.get('deep', True) and rv != self:
            rv = rv.doit(**hints)
        return rv

    def evalf(self, prec=None, **options):
        return self.doit().evalf(prec, **options)

    n = evalf  # type:ignore

    @property
    def variables(self):
        """The variables to be evaluated"""
        return self._args[1]

    bound_symbols = variables

    @property
    def expr(self):
        """The expression on which the substitution operates"""
        return self._args[0]

    @property
    def point(self):
        """The values for which the variables are to be substituted"""
        return self._args[2]

    @property
    def free_symbols(self):
        return (self.expr.free_symbols - set(self.variables) |
            set(self.point.free_symbols))

    @property
    def expr_free_symbols(self):
        sympy_deprecation_warning("""
        The expr_free_symbols property is deprecated. Use free_symbols to get
        the free symbols of an expression.
        """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-expr-free-symbols")
        # Don't show the warning twice from the recursive call
        with ignore_warnings(SymPyDeprecationWarning):
            return (self.expr.expr_free_symbols - set(self.variables) |
                    set(self.point.expr_free_symbols))

    def __eq__(self, other):
        if not isinstance(other, Subs):
            return False
        return self._hashable_content() == other._hashable_content()

    def __ne__(self, other):
        return not(self == other)

    def __hash__(self):
        return super().__hash__()

    def _hashable_content(self):
        return (self._expr.xreplace(self.canonical_variables),
            ) + tuple(ordered([(v, p) for v, p in
            zip(self.variables, self.point) if not self.expr.has(v)]))

    def _eval_subs(self, old, new):
        # Subs doit will do the variables in order; the semantics
        # of subs for Subs is have the following invariant for
        # Subs object foo:
        #    foo.doit().subs(reps) == foo.subs(reps).doit()
        pt = list(self.point)
        if old in self.variables:
            if _atomic(new) == {new} and not any(
                    i.has(new) for i in self.args):
                # the substitution is neutral
                return self.xreplace({old: new})
            # any occurrence of old before this point will get
            # handled by replacements from here on
            i = self.variables.index(old)
            for j in range(i, len(self.variables)):
                pt[j] = pt[j]._subs(old, new)
            return self.func(self.expr, self.variables, pt)
        v = [i._subs(old, new) for i in self.variables]
        if v != list(self.variables):
            return self.func(self.expr, self.variables + (old,), pt + [new])
        expr = self.expr._subs(old, new)
        pt = [i._subs(old, new) for i in self.point]
        return self.func(expr, v, pt)

    def _eval_derivative(self, s):
        # Apply the chain rule of the derivative on the substitution variables:
        f = self.expr
        vp = V, P = self.variables, self.point
        val = Add.fromiter(p.diff(s)*Subs(f.diff(v), *vp).doit()
            for v, p in zip(V, P))

        # these are all the free symbols in the expr
        efree = f.free_symbols
        # some symbols like IndexedBase include themselves and args
        # as free symbols
        compound = {i for i in efree if len(i.free_symbols) > 1}
        # hide them and see what independent free symbols remain
        dums = {Dummy() for i in compound}
        masked = f.xreplace(dict(zip(compound, dums)))
        ifree = masked.free_symbols - dums
        # include the compound symbols
        free = ifree | compound
        # remove the variables already handled
        free -= set(V)
        # add back any free symbols of remaining compound symbols
        free |= {i for j in free & compound for i in j.free_symbols}
        # if symbols of s are in free then there is more to do
        if free & s.free_symbols:
            val += Subs(f.diff(s), self.variables, self.point).doit()
        return val

    def _eval_nseries(self, x, n, logx, cdir=0):
        if x in self.point:
            # x is the variable being substituted into
            apos = self.point.index(x)
            other = self.variables[apos]
        else:
            other = x
        arg = self.expr.nseries(other, n=n, logx=logx)
        o = arg.getO()
        terms = Add.make_args(arg.removeO())
        rv = Add(*[self.func(a, *self.args[1:]) for a in terms])
        if o:
            rv += o.subs(other, x)
        return rv

    def _eval_as_leading_term(self, x, logx, cdir):
        if x in self.point:
            ipos = self.point.index(x)
            xvar = self.variables[ipos]
            return self.expr.as_leading_term(xvar)
        if x in self.variables:
            # if `x` is a dummy variable, it means it won't exist after the
            # substitution has been performed:
            return self
        # The variable is independent of the substitution:
        return self.expr.as_leading_term(x)


def diff(f, *symbols, **kwargs):
    """
    Differentiate f with respect to symbols.

    Explanation
    ===========

    This is just a wrapper to unify .diff() and the Derivative class; its
    interface is similar to that of integrate().  You can use the same
    shortcuts for multiple variables as with Derivative.  For example,
    diff(f(x), x, x, x) and diff(f(x), x, 3) both return the third derivative
    of f(x).

    You can pass evaluate=False to get an unevaluated Derivative class.  Note
    that if there are 0 symbols (such as diff(f(x), x, 0), then the result will
    be the function (the zeroth derivative), even if evaluate=False.

    Examples
    ========

    >>> from sympy import sin, cos, Function, diff
    >>> from sympy.abc import x, y
    >>> f = Function('f')

    >>> diff(sin(x), x)
    cos(x)
    >>> diff(f(x), x, x, x)
    Derivative(f(x), (x, 3))
    >>> diff(f(x), x, 3)
    Derivative(f(x), (x, 3))
    >>> diff(sin(x)*cos(y), x, 2, y, 2)
    sin(x)*cos(y)

    >>> type(diff(sin(x), x))
    cos
    >>> type(diff(sin(x), x, evaluate=False))
    <class 'sympy.core.function.Derivative'>
    >>> type(diff(sin(x), x, 0))
    sin
    >>> type(diff(sin(x), x, 0, evaluate=False))
    sin

    >>> diff(sin(x))
    cos(x)
    >>> diff(sin(x*y))
    Traceback (most recent call last):
    ...
    ValueError: specify differentiation variables to differentiate sin(x*y)

    Note that ``diff(sin(x))`` syntax is meant only for convenience
    in interactive sessions and should be avoided in library code.

    References
    ==========

    .. [1] https://reference.wolfram.com/legacy/v5_2/Built-inFunctions/AlgebraicComputation/Calculus/D.html

    See Also
    ========

    Derivative
    idiff: computes the derivative implicitly

    """
    if hasattr(f, 'diff'):
        return f.diff(*symbols, **kwargs)
    kwargs.setdefault('evaluate', True)
    return _derivative_dispatch(f, *symbols, **kwargs)


def expand(e, deep=True, modulus=None, power_base=True, power_exp=True,
        mul=True, log=True, multinomial=True, basic=True, **hints):
    r"""
    Expand an expression using methods given as hints.

    Explanation
    ===========

    Hints evaluated unless explicitly set to False are:  ``basic``, ``log``,
    ``multinomial``, ``mul``, ``power_base``, and ``power_exp`` The following
    hints are supported but not applied unless set to True:  ``complex``,
    ``func``, and ``trig``.  In addition, the following meta-hints are
    supported by some or all of the other hints:  ``frac``, ``numer``,
    ``denom``, ``modulus``, and ``force``.  ``deep`` is supported by all
    hints.  Additionally, subclasses of Expr may define their own hints or
    meta-hints.

    The ``basic`` hint is used for any special rewriting of an object that
    should be done automatically (along with the other hints like ``mul``)
    when expand is called. This is a catch-all hint to handle any sort of
    expansion that may not be described by the existing hint names. To use
    this hint an object should override the ``_eval_expand_basic`` method.
    Objects may also define their own expand methods, which are not run by
    default.  See the API section below.

    If ``deep`` is set to ``True`` (the default), things like arguments of
    functions are recursively expanded.  Use ``deep=False`` to only expand on
    the top level.

    If the ``force`` hint is used, assumptions about variables will be ignored
    in making the expansion.

    Hints
    =====

    These hints are run by default

    mul
    ---

    Distributes multiplication over addition:

    >>> from sympy import cos, exp, sin
    >>> from sympy.abc import x, y, z
    >>> (y*(x + z)).expand(mul=True)
    x*y + y*z

    multinomial
    -----------

    Expand (x + y + ...)**n where n is a positive integer.

    >>> ((x + y + z)**2).expand(multinomial=True)
    x**2 + 2*x*y + 2*x*z + y**2 + 2*y*z + z**2

    power_exp
    ---------

    Expand addition in exponents into multiplied bases.

    >>> exp(x + y).expand(power_exp=True)
    exp(x)*exp(y)
    >>> (2**(x + y)).expand(power_exp=True)
    2**x*2**y

    power_base
    ----------

    Split powers of multiplied bases.

    This only happens by default if assumptions allow, or if the
    ``force`` meta-hint is used:

    >>> ((x*y)**z).expand(power_base=True)
    (x*y)**z
    >>> ((x*y)**z).expand(power_base=True, force=True)
    x**z*y**z
    >>> ((2*y)**z).expand(power_base=True)
    2**z*y**z

    Note that in some cases where this expansion always holds, SymPy performs
    it automatically:

    >>> (x*y)**2
    x**2*y**2

    log
    ---

    Pull out power of an argument as a coefficient and split logs products
    into sums of logs.

    Note that these only work if the arguments of the log function have the
    proper assumptions--the arguments must be positive and the exponents must
    be real--or else the ``force`` hint must be True:

    >>> from sympy import log, symbols
    >>> log(x**2*y).expand(log=True)
    log(x**2*y)
    >>> log(x**2*y).expand(log=True, force=True)
    2*log(x) + log(y)
    >>> x, y = symbols('x,y', positive=True)
    >>> log(x**2*y).expand(log=True)
    2*log(x) + log(y)

    basic
    -----

    This hint is intended primarily as a way for custom subclasses to enable
    expansion by default.

    These hints are not run by default:

    complex
    -------

    Split an expression into real and imaginary parts.

    >>> x, y = symbols('x,y')
    >>> (x + y).expand(complex=True)
    re(x) + re(y) + I*im(x) + I*im(y)
    >>> cos(x).expand(complex=True)
    -I*sin(re(x))*sinh(im(x)) + cos(re(x))*cosh(im(x))

    Note that this is just a wrapper around ``as_real_imag()``.  Most objects
    that wish to redefine ``_eval_expand_complex()`` should consider
    redefining ``as_real_imag()`` instead.

    func
    ----

    Expand other functions.

    >>> from sympy import gamma
    >>> gamma(x + 1).expand(func=True)
    x*gamma(x)

    trig
    ----

    Do trigonometric expansions.

    >>> cos(x + y).expand(trig=True)
    -sin(x)*sin(y) + cos(x)*cos(y)
    >>> sin(2*x).expand(trig=True)
    2*sin(x)*cos(x)

    Note that the forms of ``sin(n*x)`` and ``cos(n*x)`` in terms of ``sin(x)``
    and ``cos(x)`` are not unique, due to the identity `\sin^2(x) + \cos^2(x)
    = 1`.  The current implementation uses the form obtained from Chebyshev
    polynomials, but this may change.  See `this MathWorld article
    <https://mathworld.wolfram.com/Multiple-AngleFormulas.html>`_ for more
    information.

    Notes
    =====

    - You can shut off unwanted methods::

        >>> (exp(x + y)*(x + y)).expand()
        x*exp(x)*exp(y) + y*exp(x)*exp(y)
        >>> (exp(x + y)*(x + y)).expand(power_exp=False)
        x*exp(x + y) + y*exp(x + y)
        >>> (exp(x + y)*(x + y)).expand(mul=False)
        (x + y)*exp(x)*exp(y)

    - Use deep=False to only expand on the top level::

        >>> exp(x + exp(x + y)).expand()
        exp(x)*exp(exp(x)*exp(y))
        >>> exp(x + exp(x + y)).expand(deep=False)
        exp(x)*exp(exp(x + y))

    - Hints are applied in an arbitrary, but consistent order (in the current
      implementation, they are applied in alphabetical order, except
      multinomial comes before mul, but this may change).  Because of this,
      some hints may prevent expansion by other hints if they are applied
      first. For example, ``mul`` may distribute multiplications and prevent
      ``log`` and ``power_base`` from expanding them. Also, if ``mul`` is
      applied before ``multinomial`, the expression might not be fully
      distributed. The solution is to use the various ``expand_hint`` helper
      functions or to use ``hint=False`` to this function to finely control
      which hints are applied. Here are some examples::

        >>> from sympy import expand, expand_mul, expand_power_base
        >>> x, y, z = symbols('x,y,z', positive=True)

        >>> expand(log(x*(y + z)))
        log(x) + log(y + z)

      Here, we see that ``log`` was applied before ``mul``.  To get the mul
      expanded form, either of the following will work::

        >>> expand_mul(log(x*(y + z)))
        log(x*y + x*z)
        >>> expand(log(x*(y + z)), log=False)
        log(x*y + x*z)

      A similar thing can happen with the ``power_base`` hint::

        >>> expand((x*(y + z))**x)
        (x*y + x*z)**x

      To get the ``power_base`` expanded form, either of the following will
      work::

        >>> expand((x*(y + z))**x, mul=False)
        x**x*(y + z)**x
        >>> expand_power_base((x*(y + z))**x)
        x**x*(y + z)**x

        >>> expand((x + y)*y/x)
        y + y**2/x

      The parts of a rational expression can be targeted::

        >>> expand((x + y)*y/x/(x + 1), frac=True)
        (x*y + y**2)/(x**2 + x)
        >>> expand((x + y)*y/x/(x + 1), numer=True)
        (x*y + y**2)/(x*(x + 1))
        >>> expand((x + y)*y/x/(x + 1), denom=True)
        y*(x + y)/(x**2 + x)

    - The ``modulus`` meta-hint can be used to reduce the coefficients of an
      expression post-expansion::

        >>> expand((3*x + 1)**2)
        9*x**2 + 6*x + 1
        >>> expand((3*x + 1)**2, modulus=5)
        4*x**2 + x + 1

    - Either ``expand()`` the function or ``.expand()`` the method can be
      used.  Both are equivalent::

        >>> expand((x + 1)**2)
        x**2 + 2*x + 1
        >>> ((x + 1)**2).expand()
        x**2 + 2*x + 1

    API
    ===

    Objects can define their own expand hints by defining
    ``_eval_expand_hint()``.  The function should take the form::

        def _eval_expand_hint(self, **hints):
            # Only apply the method to the top-level expression
            ...

    See also the example below.  Objects should define ``_eval_expand_hint()``
    methods only if ``hint`` applies to that specific object.  The generic
    ``_eval_expand_hint()`` method defined in Expr will handle the no-op case.

    Each hint should be responsible for expanding that hint only.
    Furthermore, the expansion should be applied to the top-level expression
    only.  ``expand()`` takes care of the recursion that happens when
    ``deep=True``.

    You should only call ``_eval_expand_hint()`` methods directly if you are
    100% sure that the object has the method, as otherwise you are liable to
    get unexpected ``AttributeError``s.  Note, again, that you do not need to
    recursively apply the hint to args of your object: this is handled
    automatically by ``expand()``.  ``_eval_expand_hint()`` should
    generally not be used at all outside of an ``_eval_expand_hint()`` method.
    If you want to apply a specific expansion from within another method, use
    the public ``expand()`` function, method, or ``expand_hint()`` functions.

    In order for expand to work, objects must be rebuildable by their args,
    i.e., ``obj.func(*obj.args) == obj`` must hold.

    Expand methods are passed ``**hints`` so that expand hints may use
    'metahints'--hints that control how different expand methods are applied.
    For example, the ``force=True`` hint described above that causes
    ``expand(log=True)`` to ignore assumptions is such a metahint.  The
    ``deep`` meta-hint is handled exclusively by ``expand()`` and is not
    passed to ``_eval_expand_hint()`` methods.

    Note that expansion hints should generally be methods that perform some
    kind of 'expansion'.  For hints that simply rewrite an expression, use the
    .rewrite() API.

    Examples
    ========

    >>> from sympy import Expr, sympify
    >>> class MyClass(Expr):
    ...     def __new__(cls, *args):
    ...         args = sympify(args)
    ...         return Expr.__new__(cls, *args)
    ...
    ...     def _eval_expand_double(self, *, force=False, **hints):
    ...         '''
    ...         Doubles the args of MyClass.
    ...
    ...         If there more than four args, doubling is not performed,
    ...         unless force=True is also used (False by default).
    ...         '''
    ...         if not force and len(self.args) > 4:
    ...             return self
    ...         return self.func(*(self.args + self.args))
    ...
    >>> a = MyClass(1, 2, MyClass(3, 4))
    >>> a
    MyClass(1, 2, MyClass(3, 4))
    >>> a.expand(double=True)
    MyClass(1, 2, MyClass(3, 4, 3, 4), 1, 2, MyClass(3, 4, 3, 4))
    >>> a.expand(double=True, deep=False)
    MyClass(1, 2, MyClass(3, 4), 1, 2, MyClass(3, 4))

    >>> b = MyClass(1, 2, 3, 4, 5)
    >>> b.expand(double=True)
    MyClass(1, 2, 3, 4, 5)
    >>> b.expand(double=True, force=True)
    MyClass(1, 2, 3, 4, 5, 1, 2, 3, 4, 5)

    See Also
    ========

    expand_log, expand_mul, expand_multinomial, expand_complex, expand_trig,
    expand_power_base, expand_power_exp, expand_func, sympy.simplify.hyperexpand.hyperexpand

    """
    # don't modify this; modify the Expr.expand method
    hints['power_base'] = power_base
    hints['power_exp'] = power_exp
    hints['mul'] = mul
    hints['log'] = log
    hints['multinomial'] = multinomial
    hints['basic'] = basic
    return sympify(e).expand(deep=deep, modulus=modulus, **hints)

# This is a special application of two hints

def _mexpand(expr, recursive=False):
    # expand multinomials and then expand products; this may not always
    # be sufficient to give a fully expanded expression (see
    # test_issue_8247_8354 in test_arit)
    if expr is None:
        return
    was = None
    while was != expr:
        was, expr = expr, expand_mul(expand_multinomial(expr))
        if not recursive:
            break
    return expr


# These are simple wrappers around single hints.


def expand_mul(expr, deep=True):
    """
    Wrapper around expand that only uses the mul hint.  See the expand
    docstring for more information.

    Examples
    ========

    >>> from sympy import symbols, expand_mul, exp, log
    >>> x, y = symbols('x,y', positive=True)
    >>> expand_mul(exp(x+y)*(x+y)*log(x*y**2))
    x*exp(x + y)*log(x*y**2) + y*exp(x + y)*log(x*y**2)

    """
    return sympify(expr).expand(deep=deep, mul=True, power_exp=False,
    power_base=False, basic=False, multinomial=False, log=False)


def expand_multinomial(expr, deep=True):
    """
    Wrapper around expand that only uses the multinomial hint.  See the expand
    docstring for more information.

    Examples
    ========

    >>> from sympy import symbols, expand_multinomial, exp
    >>> x, y = symbols('x y', positive=True)
    >>> expand_multinomial((x + exp(x + 1))**2)
    x**2 + 2*x*exp(x + 1) + exp(2*x + 2)

    """
    return sympify(expr).expand(deep=deep, mul=False, power_exp=False,
    power_base=False, basic=False, multinomial=True, log=False)


def expand_log(expr, deep=True, force=False, factor=False):
    """
    Wrapper around expand that only uses the log hint.  See the expand
    docstring for more information.

    Examples
    ========

    >>> from sympy import symbols, expand_log, exp, log
    >>> x, y = symbols('x,y', positive=True)
    >>> expand_log(exp(x+y)*(x+y)*log(x*y**2))
    (x + y)*(log(x) + 2*log(y))*exp(x + y)

    """
    from sympy.functions.elementary.exponential import log
    from sympy.simplify.radsimp import fraction
    if factor is False:
        def _handleMul(x):
            # look for the simple case of expanded log(b**a)/log(b) -> a in args
            n, d = fraction(x)
            n = [i for i in n.atoms(log) if i.args[0].is_Integer]
            d = [i for i in d.atoms(log) if i.args[0].is_Integer]
            if len(n) == 1 and len(d) == 1:
                n = n[0]
                d = d[0]
                from sympy import multiplicity
                m = multiplicity(d.args[0], n.args[0])
                if m:
                    r = m + log(n.args[0]//d.args[0]**m)/d
                    x = x.subs(n, d*r)
            x1 = expand_mul(expand_log(x, deep=deep, force=force, factor=True))
            if x1.count(log) <= x.count(log):
                return x1
            return x

        expr = expr.replace(
        lambda x: x.is_Mul and all(any(isinstance(i, log) and i.args[0].is_Rational
        for i in Mul.make_args(j)) for j in x.as_numer_denom()),
        _handleMul)

    return sympify(expr).expand(deep=deep, log=True, mul=False,
        power_exp=False, power_base=False, multinomial=False,
        basic=False, force=force, factor=factor)


def expand_func(expr, deep=True):
    """
    Wrapper around expand that only uses the func hint.  See the expand
    docstring for more information.

    Examples
    ========

    >>> from sympy import expand_func, gamma
    >>> from sympy.abc import x
    >>> expand_func(gamma(x + 2))
    x*(x + 1)*gamma(x)

    """
    return sympify(expr).expand(deep=deep, func=True, basic=False,
    log=False, mul=False, power_exp=False, power_base=False, multinomial=False)


def expand_trig(expr, deep=True):
    """
    Wrapper around expand that only uses the trig hint.  See the expand
    docstring for more information.

    Examples
    ========

    >>> from sympy import expand_trig, sin
    >>> from sympy.abc import x, y
    >>> expand_trig(sin(x+y)*(x+y))
    (x + y)*(sin(x)*cos(y) + sin(y)*cos(x))

    """
    return sympify(expr).expand(deep=deep, trig=True, basic=False,
    log=False, mul=False, power_exp=False, power_base=False, multinomial=False)


def expand_complex(expr, deep=True):
    """
    Wrapper around expand that only uses the complex hint.  See the expand
    docstring for more information.

    Examples
    ========

    >>> from sympy import expand_complex, exp, sqrt, I
    >>> from sympy.abc import z
    >>> expand_complex(exp(z))
    I*exp(re(z))*sin(im(z)) + exp(re(z))*cos(im(z))
    >>> expand_complex(sqrt(I))
    sqrt(2)/2 + sqrt(2)*I/2

    See Also
    ========

    sympy.core.expr.Expr.as_real_imag
    """
    return sympify(expr).expand(deep=deep, complex=True, basic=False,
    log=False, mul=False, power_exp=False, power_base=False, multinomial=False)


def expand_power_base(expr, deep=True, force=False):
    """
    Wrapper around expand that only uses the power_base hint.

    A wrapper to expand(power_base=True) which separates a power with a base
    that is a Mul into a product of powers, without performing any other
    expansions, provided that assumptions about the power's base and exponent
    allow.

    deep=False (default is True) will only apply to the top-level expression.

    force=True (default is False) will cause the expansion to ignore
    assumptions about the base and exponent. When False, the expansion will
    only happen if the base is non-negative or the exponent is an integer.

    >>> from sympy.abc import x, y, z
    >>> from sympy import expand_power_base, sin, cos, exp, Symbol

    >>> (x*y)**2
    x**2*y**2

    >>> (2*x)**y
    (2*x)**y
    >>> expand_power_base(_)
    2**y*x**y

    >>> expand_power_base((x*y)**z)
    (x*y)**z
    >>> expand_power_base((x*y)**z, force=True)
    x**z*y**z
    >>> expand_power_base(sin((x*y)**z), deep=False)
    sin((x*y)**z)
    >>> expand_power_base(sin((x*y)**z), force=True)
    sin(x**z*y**z)

    >>> expand_power_base((2*sin(x))**y + (2*cos(x))**y)
    2**y*sin(x)**y + 2**y*cos(x)**y

    >>> expand_power_base((2*exp(y))**x)
    2**x*exp(y)**x

    >>> expand_power_base((2*cos(x))**y)
    2**y*cos(x)**y

    Notice that sums are left untouched. If this is not the desired behavior,
    apply full ``expand()`` to the expression:

    >>> expand_power_base(((x+y)*z)**2)
    z**2*(x + y)**2
    >>> (((x+y)*z)**2).expand()
    x**2*z**2 + 2*x*y*z**2 + y**2*z**2

    >>> expand_power_base((2*y)**(1+z))
    2**(z + 1)*y**(z + 1)
    >>> ((2*y)**(1+z)).expand()
    2*2**z*y**(z + 1)

    The power that is unexpanded can be expanded safely when
    ``y != 0``, otherwise different values might be obtained for the expression:

    >>> prev = _

    If we indicate that ``y`` is positive but then replace it with
    a value of 0 after expansion, the expression becomes 0:

    >>> p = Symbol('p', positive=True)
    >>> prev.subs(y, p).expand().subs(p, 0)
    0

    But if ``z = -1`` the expression would not be zero:

    >>> prev.subs(y, 0).subs(z, -1)
    1

    See Also
    ========

    expand

    """
    return sympify(expr).expand(deep=deep, log=False, mul=False,
        power_exp=False, power_base=True, multinomial=False,
        basic=False, force=force)


def expand_power_exp(expr, deep=True):
    """
    Wrapper around expand that only uses the power_exp hint.

    See the expand docstring for more information.

    Examples
    ========

    >>> from sympy import expand_power_exp, Symbol
    >>> from sympy.abc import x, y
    >>> expand_power_exp(3**(y + 2))
    9*3**y
    >>> expand_power_exp(x**(y + 2))
    x**(y + 2)

    If ``x = 0`` the value of the expression depends on the
    value of ``y``; if the expression were expanded the result
    would be 0. So expansion is only done if ``x != 0``:

    >>> expand_power_exp(Symbol('x', zero=False)**(y + 2))
    x**2*x**y
    """
    return sympify(expr).expand(deep=deep, complex=False, basic=False,
    log=False, mul=False, power_exp=True, power_base=False, multinomial=False)


def count_ops(expr, visual=False):
    """
    Return a representation (integer or expression) of the operations in expr.

    Parameters
    ==========

    expr : Expr
        If expr is an iterable, the sum of the op counts of the
        items will be returned.

    visual : bool, optional
        If ``False`` (default) then the sum of the coefficients of the
        visual expression will be returned.
        If ``True`` then the number of each type of operation is shown
        with the core class types (or their virtual equivalent) multiplied by the
        number of times they occur.

    Examples
    ========

    >>> from sympy.abc import a, b, x, y
    >>> from sympy import sin, count_ops

    Although there is not a SUB object, minus signs are interpreted as
    either negations or subtractions:

    >>> (x - y).count_ops(visual=True)
    SUB
    >>> (-x).count_ops(visual=True)
    NEG

    Here, there are two Adds and a Pow:

    >>> (1 + a + b**2).count_ops(visual=True)
    2*ADD + POW

    In the following, an Add, Mul, Pow and two functions:

    >>> (sin(x)*x + sin(x)**2).count_ops(visual=True)
    ADD + MUL + POW + 2*SIN

    for a total of 5:

    >>> (sin(x)*x + sin(x)**2).count_ops(visual=False)
    5

    Note that "what you type" is not always what you get. The expression
    1/x/y is translated by sympy into 1/(x*y) so it gives a DIV and MUL rather
    than two DIVs:

    >>> (1/x/y).count_ops(visual=True)
    DIV + MUL

    The visual option can be used to demonstrate the difference in
    operations for expressions in different forms. Here, the Horner
    representation is compared with the expanded form of a polynomial:

    >>> eq=x*(1 + x*(2 + x*(3 + x)))
    >>> count_ops(eq.expand(), visual=True) - count_ops(eq, visual=True)
    -MUL + 3*POW

    The count_ops function also handles iterables:

    >>> count_ops([x, sin(x), None, True, x + 2], visual=False)
    2
    >>> count_ops([x, sin(x), None, True, x + 2], visual=True)
    ADD + SIN
    >>> count_ops({x: sin(x), x + 2: y + 1}, visual=True)
    2*ADD + SIN

    """
    from .relational import Relational
    from sympy.concrete.summations import Sum
    from sympy.integrals.integrals import Integral
    from sympy.logic.boolalg import BooleanFunction
    from sympy.simplify.radsimp import fraction

    expr = sympify(expr)
    if isinstance(expr, Expr) and not expr.is_Relational:

        ops = []
        args = [expr]
        NEG = Symbol('NEG')
        DIV = Symbol('DIV')
        SUB = Symbol('SUB')
        ADD = Symbol('ADD')
        EXP = Symbol('EXP')
        while args:
            a = args.pop()

            # if the following fails because the object is
            # not Basic type, then the object should be fixed
            # since it is the intention that all args of Basic
            # should themselves be Basic
            if a.is_Rational:
                #-1/3 = NEG + DIV
                if a is not S.One:
                    if a.p < 0:
                        ops.append(NEG)
                    if a.q != 1:
                        ops.append(DIV)
                    continue
            elif a.is_Mul or a.is_MatMul:
                if _coeff_isneg(a):
                    ops.append(NEG)
                    if a.args[0] is S.NegativeOne:
                        a = a.as_two_terms()[1]
                    else:
                        a = -a
                n, d = fraction(a)
                if n.is_Integer:
                    ops.append(DIV)
                    if n < 0:
                        ops.append(NEG)
                    args.append(d)
                    continue  # won't be -Mul but could be Add
                elif d is not S.One:
                    if not d.is_Integer:
                        args.append(d)
                    ops.append(DIV)
                    args.append(n)
                    continue  # could be -Mul
            elif a.is_Add or a.is_MatAdd:
                aargs = list(a.args)
                negs = 0
                for i, ai in enumerate(aargs):
                    if _coeff_isneg(ai):
                        negs += 1
                        args.append(-ai)
                        if i > 0:
                            ops.append(SUB)
                    else:
                        args.append(ai)
                        if i > 0:
                            ops.append(ADD)
                if negs == len(aargs):  # -x - y = NEG + SUB
                    ops.append(NEG)
                elif _coeff_isneg(aargs[0]):  # -x + y = SUB, but already recorded ADD
                    ops.append(SUB - ADD)
                continue
            if a.is_Pow and a.exp is S.NegativeOne:
                ops.append(DIV)
                args.append(a.base)  # won't be -Mul but could be Add
                continue
            if a == S.Exp1:
                ops.append(EXP)
                continue
            if a.is_Pow and a.base == S.Exp1:
                ops.append(EXP)
                args.append(a.exp)
                continue
            if a.is_Mul or isinstance(a, LatticeOp):
                o = Symbol(a.func.__name__.upper())
                # count the args
                ops.append(o*(len(a.args) - 1))
            elif a.args and (
                    a.is_Pow or a.is_Function or isinstance(a, (Derivative, Integral, Sum))):
                # if it's not in the list above we don't
                # consider a.func something to count, e.g.
                # Tuple, MatrixSymbol, etc...
                if isinstance(a.func, UndefinedFunction):
                    o = Symbol("FUNC_" + a.func.__name__.upper())
                else:
                    o = Symbol(a.func.__name__.upper())
                ops.append(o)

            if not a.is_Symbol:
                args.extend(a.args)

    elif isinstance(expr, Dict):
        ops = [count_ops(k, visual=visual) +
               count_ops(v, visual=visual) for k, v in expr.items()]
    elif iterable(expr):
        ops = [count_ops(i, visual=visual) for i in expr]
    elif isinstance(expr, (Relational, BooleanFunction)):
        ops = []
        for arg in expr.args:
            ops.append(count_ops(arg, visual=True))
        o = Symbol(func_name(expr, short=True).upper())
        ops.append(o)
    elif not isinstance(expr, Basic):
        ops = []
    else:  # it's Basic not isinstance(expr, Expr):
        if not isinstance(expr, Basic):
            raise TypeError("Invalid type of expr")
        else:
            ops = []
            args = [expr]
            while args:
                a = args.pop()

                if a.args:
                    o = Symbol(type(a).__name__.upper())
                    if a.is_Boolean:
                        ops.append(o*(len(a.args)-1))
                    else:
                        ops.append(o)
                    args.extend(a.args)

    if not ops:
        if visual:
            return S.Zero
        return 0

    ops = Add(*ops)

    if visual:
        return ops

    if ops.is_Number:
        return int(ops)

    return sum(int((a.args or [1])[0]) for a in Add.make_args(ops))


def nfloat(expr, n=15, exponent=False, dkeys=False):
    """Make all Rationals in expr Floats except those in exponents
    (unless the exponents flag is set to True) and those in undefined
    functions. When processing dictionaries, do not modify the keys
    unless ``dkeys=True``.

    Examples
    ========

    >>> from sympy import nfloat, cos, pi, sqrt
    >>> from sympy.abc import x, y
    >>> nfloat(x**4 + x/2 + cos(pi/3) + 1 + sqrt(y))
    x**4 + 0.5*x + sqrt(y) + 1.5
    >>> nfloat(x**4 + sqrt(y), exponent=True)
    x**4.0 + y**0.5

    Container types are not modified:

    >>> type(nfloat((1, 2))) is tuple
    True
    """
    from sympy.matrices.matrixbase import MatrixBase

    kw = {"n": n, "exponent": exponent, "dkeys": dkeys}

    if isinstance(expr, MatrixBase):
        return expr.applyfunc(lambda e: nfloat(e, **kw))

    # handling of iterable containers
    if iterable(expr, exclude=str):
        if isinstance(expr, (dict, Dict)):
            if dkeys:
                args = [tuple((nfloat(i, **kw) for i in a))
                    for a in expr.items()]
            else:
                args = [(k, nfloat(v, **kw)) for k, v in expr.items()]
            if isinstance(expr, dict):
                return type(expr)(args)
            else:
                return expr.func(*args)
        elif isinstance(expr, Basic):
            return expr.func(*[nfloat(a, **kw) for a in expr.args])
        return type(expr)([nfloat(a, **kw) for a in expr])

    rv = sympify(expr)

    if rv.is_Number:
        return Float(rv, n)
    elif rv.is_number:
        # evalf doesn't always set the precision
        rv = rv.n(n)
        if rv.is_Number:
            rv = Float(rv.n(n), n)
        else:
            pass  # pure_complex(rv) is likely True
        return rv
    elif rv.is_Atom:
        return rv
    elif rv.is_Relational:
        args_nfloat = (nfloat(arg, **kw) for arg in rv.args)
        return rv.func(*args_nfloat)


    # watch out for RootOf instances that don't like to have
    # their exponents replaced with Dummies and also sometimes have
    # problems with evaluating at low precision (issue 6393)
    from sympy.polys.rootoftools import RootOf
    rv = rv.xreplace({ro: ro.n(n) for ro in rv.atoms(RootOf)})

    from .power import Pow
    if not exponent:
        reps = [(p, Pow(p.base, Dummy())) for p in rv.atoms(Pow)]
        rv = rv.xreplace(dict(reps))
    rv = rv.n(n)
    if not exponent:
        rv = rv.xreplace({d.exp: p.exp for p, d in reps})
    else:
        # Pow._eval_evalf special cases Integer exponents so if
        # exponent is suppose to be handled we have to do so here
        rv = rv.xreplace(Transform(
            lambda x: Pow(x.base, Float(x.exp, n)),
            lambda x: x.is_Pow and x.exp.is_Integer))

    return rv.xreplace(Transform(
        lambda x: x.func(*nfloat(x.args, n, exponent)),
        lambda x: isinstance(x, Function) and not isinstance(x, AppliedUndef)))


from .symbol import Dummy, Symbol
<|endoftext|>"""
Provides functionality for multidimensional usage of scalar-functions.

Read the vectorize docstring for more details.
"""

from functools import wraps


def apply_on_element(f, args, kwargs, n):
    """
    Returns a structure with the same dimension as the specified argument,
    where each basic element is replaced by the function f applied on it. All
    other arguments stay the same.
    """
    # Get the specified argument.
    if isinstance(n, int):
        structure = args[n]
        is_arg = True
    elif isinstance(n, str):
        structure = kwargs[n]
        is_arg = False

    # Define reduced function that is only dependent on the specified argument.
    def f_reduced(x):
        if hasattr(x, "__iter__"):
            return list(map(f_reduced, x))
        else:
            if is_arg:
                args[n] = x
            else:
                kwargs[n] = x
            return f(*args, **kwargs)

    # f_reduced will call itself recursively so that in the end f is applied to
    # all basic elements.
    return list(map(f_reduced, structure))


def iter_copy(structure):
    """
    Returns a copy of an iterable object (also copying all embedded iterables).
    """
    return [iter_copy(i) if hasattr(i, "__iter__") else i for i in structure]


def structure_copy(structure):
    """
    Returns a copy of the given structure (numpy-array, list, iterable, ..).
    """
    if hasattr(structure, "copy"):
        return structure.copy()
    return iter_copy(structure)


class vectorize:
    """
    Generalizes a function taking scalars to accept multidimensional arguments.

    Examples
    ========

    >>> from sympy import vectorize, diff, sin, symbols, Function
    >>> x, y, z = symbols('x y z')
    >>> f, g, h = list(map(Function, 'fgh'))

    >>> @vectorize(0)
    ... def vsin(x):
    ...     return sin(x)

    >>> vsin([1, x, y])
    [sin(1), sin(x), sin(y)]

    >>> @vectorize(0, 1)
    ... def vdiff(f, y):
    ...     return diff(f, y)

    >>> vdiff([f(x, y, z), g(x, y, z), h(x, y, z)], [x, y, z])
    [[Derivative(f(x, y, z), x), Derivative(f(x, y, z), y), Derivative(f(x, y, z), z)], [Derivative(g(x, y, z), x), Derivative(g(x, y, z), y), Derivative(g(x, y, z), z)], [Derivative(h(x, y, z), x), Derivative(h(x, y, z), y), Derivative(h(x, y, z), z)]]
    """
    def __init__(self, *mdargs):
        """
        The given numbers and strings characterize the arguments that will be
        treated as data structures, where the decorated function will be applied
        to every single element.
        If no argument is given, everything is treated multidimensional.
        """
        for a in mdargs:
            if not isinstance(a, (int, str)):
                raise TypeError("a is of invalid type")
        self.mdargs = mdargs

    def __call__(self, f):
        """
        Returns a wrapper for the one-dimensional function that can handle
        multidimensional arguments.
        """
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Get arguments that should be treated multidimensional
            if self.mdargs:
                mdargs = self.mdargs
            else:
                mdargs = range(len(args)) + kwargs.keys()

            arglength = len(args)

            for n in mdargs:
                if isinstance(n, int):
                    if n >= arglength:
                        continue
                    entry = args[n]
                    is_arg = True
                elif isinstance(n, str):
                    try:
                        entry = kwargs[n]
                    except KeyError:
                        continue
                    is_arg = False
                if hasattr(entry, "__iter__"):
                    # Create now a copy of the given array and manipulate then
                    # the entries directly.
                    if is_arg:
                        args = list(args)
                        args[n] = structure_copy(entry)
                    else:
                        kwargs[n] = structure_copy(entry)
                    result = apply_on_element(wrapper, args, kwargs, n)
                    return result
            return f(*args, **kwargs)
        return wrapper
<|endoftext|>""" Caching facility for SymPy """
from importlib import import_module
from typing import Callable

class _cache(list):
    """ List of cached functions """

    def print_cache(self):
        """print cache info"""

        for item in self:
            name = item.__name__
            myfunc = item
            while hasattr(myfunc, '__wrapped__'):
                if hasattr(myfunc, 'cache_info'):
                    info = myfunc.cache_info()
                    break
                else:
                    myfunc = myfunc.__wrapped__
            else:
                info = None

            print(name, info)

    def clear_cache(self):
        """clear cache content"""
        for item in self:
            myfunc = item
            while hasattr(myfunc, '__wrapped__'):
                if hasattr(myfunc, 'cache_clear'):
                    myfunc.cache_clear()
                    break
                else:
                    myfunc = myfunc.__wrapped__


# global cache registry:
CACHE = _cache()
# make clear and print methods available
print_cache = CACHE.print_cache
clear_cache = CACHE.clear_cache

from functools import lru_cache, wraps

def __cacheit(maxsize):
    """caching decorator.

        important: the result of cached function must be *immutable*


        Examples
        ========

        >>> from sympy import cacheit
        >>> @cacheit
        ... def f(a, b):
        ...    return a+b

        >>> @cacheit
        ... def f(a, b): # noqa: F811
        ...    return [a, b] # <-- WRONG, returns mutable object

        to force cacheit to check returned results mutability and consistency,
        set environment variable SYMPY_USE_CACHE to 'debug'
    """
    def func_wrapper(func):
        cfunc = lru_cache(maxsize, typed=True)(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                retval = cfunc(*args, **kwargs)
            except TypeError as e:
                if not e.args or not e.args[0].startswith('unhashable type:'):
                    raise
                retval = func(*args, **kwargs)
            return retval

        wrapper.cache_info = cfunc.cache_info
        wrapper.cache_clear = cfunc.cache_clear

        CACHE.append(wrapper)
        return wrapper

    return func_wrapper
########################################


def __cacheit_nocache(func):
    return func


def __cacheit_debug(maxsize):
    """cacheit + code to check cache consistency"""
    def func_wrapper(func):
        cfunc = __cacheit(maxsize)(func)

        @wraps(func)
        def wrapper(*args, **kw_args):
            # always call function itself and compare it with cached version
            r1 = func(*args, **kw_args)
            r2 = cfunc(*args, **kw_args)

            # try to see if the result is immutable
            #
            # this works because:
            #
            # hash([1,2,3])         -> raise TypeError
            # hash({'a':1, 'b':2})  -> raise TypeError
            # hash((1,[2,3]))       -> raise TypeError
            #
            # hash((1,2,3))         -> just computes the hash
            hash(r1), hash(r2)

            # also see if returned values are the same
            if r1 != r2:
                raise RuntimeError("Returned values are not the same")
            return r1
        return wrapper
    return func_wrapper


def _getenv(key, default=None):
    from os import getenv
    return getenv(key, default)

# SYMPY_USE_CACHE=yes/no/debug
USE_CACHE = _getenv('SYMPY_USE_CACHE', 'yes').lower()
# SYMPY_CACHE_SIZE=some_integer/None
# special cases :
#  SYMPY_CACHE_SIZE=0    -> No caching
#  SYMPY_CACHE_SIZE=None -> Unbounded caching
scs = _getenv('SYMPY_CACHE_SIZE', '1000')
if scs.lower() == 'none':
    SYMPY_CACHE_SIZE = None
else:
    try:
        SYMPY_CACHE_SIZE = int(scs)
    except ValueError:
        raise RuntimeError(
            'SYMPY_CACHE_SIZE must be a valid integer or None. ' + \
            'Got: %s' % SYMPY_CACHE_SIZE)

if USE_CACHE == 'no':
    cacheit = __cacheit_nocache
elif USE_CACHE == 'yes':
    cacheit = __cacheit(SYMPY_CACHE_SIZE)
elif USE_CACHE == 'debug':
    cacheit = __cacheit_debug(SYMPY_CACHE_SIZE)   # a lot slower
else:
    raise RuntimeError(
        'unrecognized value for SYMPY_USE_CACHE: %s' % USE_CACHE)


def cached_property(func):
    '''Decorator to cache property method'''
    attrname = '__' + func.__name__
    _cached_property_sentinel = object()
    def propfunc(self):
        val = getattr(self, attrname, _cached_property_sentinel)
        if val is _cached_property_sentinel:
            val = func(self)
            setattr(self, attrname, val)
        return val
    return property(propfunc)


def lazy_function(module : str, name : str) -> Callable:
    """Create a lazy proxy for a function in a module.

    The module containing the function is not imported until the function is used.

    """
    func = None

    def _get_function():
        nonlocal func
        if func is None:
            func = getattr(import_module(module), name)
        return func

    # The metaclass is needed so that help() shows the docstring
    class LazyFunctionMeta(type):
        @property
        def __doc__(self):
            docstring = _get_function().__doc__
            docstring += f"\n\nNote: this is a {self.__class__.__name__} wrapper of '{module}.{name}'"
            return docstring

    class LazyFunction(metaclass=LazyFunctionMeta):
        def __call__(self, *args, **kwargs):
            # inline get of function for performance gh-23832
            nonlocal func
            if func is None:
                func = getattr(import_module(module), name)
            return func(*args, **kwargs)

        @property
        def __doc__(self):
            docstring = _get_function().__doc__
            docstring += f"\n\nNote: this is a {self.__class__.__name__} wrapper of '{module}.{name}'"
            return docstring

        def __str__(self):
            return _get_function().__str__()

        def __repr__(self):
            return f"<{__class__.__name__} object at 0x{id(self):x}>: wrapping '{module}.{name}'"

    return LazyFunction()
<|endoftext|>"""Tools for manipulating of large commutative expressions. """

from __future__ import annotations

from .add import Add
from .mul import Mul, _keep_coeff
from .power import Pow
from .basic import Basic
from .expr import Expr
from .function import expand_power_exp
from .sympify import sympify
from .numbers import Rational, Integer, Number, I, equal_valued
from .singleton import S
from .sorting import default_sort_key, ordered
from .symbol import Dummy
from .traversal import preorder_traversal
from .coreerrors import NonCommutativeExpression
from .containers import Tuple, Dict
from sympy.external.gmpy import SYMPY_INTS
from sympy.utilities.iterables import (common_prefix, common_suffix,
        variations, iterable, is_sequence)

from collections import defaultdict


_eps = Dummy(positive=True)


def _isnumber(i):
    return isinstance(i, (SYMPY_INTS, float)) or i.is_Number


def _monotonic_sign(self):
    """Return the value closest to 0 that ``self`` may have if all symbols
    are signed and the result is uniformly the same sign for all values of symbols.
    If a symbol is only signed but not known to be an
    integer or the result is 0 then a symbol representative of the sign of self
    will be returned. Otherwise, None is returned if a) the sign could be positive
    or negative or b) self is not in one of the following forms:

    - L(x, y, ...) + A: a function linear in all symbols x, y, ... with an
      additive constant; if A is zero then the function can be a monomial whose
      sign is monotonic over the range of the variables, e.g. (x + 1)**3 if x is
      nonnegative.
    - A/L(x, y, ...) + B: the inverse of a function linear in all symbols x, y, ...
      that does not have a sign change from positive to negative for any set
      of values for the variables.
    - M(x, y, ...) + A: a monomial M whose factors are all signed and a constant, A.
    - A/M(x, y, ...) + B: the inverse of a monomial and constants A and B.
    - P(x): a univariate polynomial

    Examples
    ========

    >>> from sympy.core.exprtools import _monotonic_sign as F
    >>> from sympy import Dummy
    >>> nn = Dummy(integer=True, nonnegative=True)
    >>> p = Dummy(integer=True, positive=True)
    >>> p2 = Dummy(integer=True, positive=True)
    >>> F(nn + 1)
    1
    >>> F(p - 1)
    _nneg
    >>> F(nn*p + 1)
    1
    >>> F(p2*p + 1)
    2
    >>> F(nn - 1)  # could be negative, zero or positive
    """
    if not self.is_extended_real:
        return

    if (-self).is_Symbol:
        rv = _monotonic_sign(-self)
        return rv if rv is None else -rv

    if not self.is_Add and self.as_numer_denom()[1].is_number:
        s = self
        if s.is_prime:
            if s.is_odd:
                return Integer(3)
            else:
                return Integer(2)
        elif s.is_composite:
            if s.is_odd:
                return Integer(9)
            else:
                return Integer(4)
        elif s.is_positive:
            if s.is_even:
                if s.is_prime is False:
                    return Integer(4)
                else:
                    return Integer(2)
            elif s.is_integer:
                return S.One
            else:
                return _eps
        elif s.is_extended_negative:
            if s.is_even:
                return Integer(-2)
            elif s.is_integer:
                return S.NegativeOne
            else:
                return -_eps
        if s.is_zero or s.is_extended_nonpositive or s.is_extended_nonnegative:
            return S.Zero
        return None

    # univariate polynomial
    free = self.free_symbols
    if len(free) == 1:
        if self.is_polynomial():
            from sympy.polys.polytools import real_roots
            from sympy.polys.polyroots import roots
            from sympy.polys.polyerrors import PolynomialError
            x = free.pop()
            x0 = _monotonic_sign(x)
            if x0 in (_eps, -_eps):
                x0 = S.Zero
            if x0 is not None:
                d = self.diff(x)
                if d.is_number:
                    currentroots = []
                else:
                    try:
                        currentroots = real_roots(d)
                    except (PolynomialError, NotImplementedError):
                        currentroots = [r for r in roots(d, x) if r.is_extended_real]
                y = self.subs(x, x0)
                if x.is_nonnegative and all(
                        (r - x0).is_nonpositive for r in currentroots):
                    if y.is_nonnegative and d.is_positive:
                        if y:
                            return y if y.is_positive else Dummy('pos', positive=True)
                        else:
                            return Dummy('nneg', nonnegative=True)
                    if y.is_nonpositive and d.is_negative:
                        if y:
                            return y if y.is_negative else Dummy('neg', negative=True)
                        else:
                            return Dummy('npos', nonpositive=True)
                elif x.is_nonpositive and all(
                        (r - x0).is_nonnegative for r in currentroots):
                    if y.is_nonnegative and d.is_negative:
                        if y:
                            return Dummy('pos', positive=True)
                        else:
                            return Dummy('nneg', nonnegative=True)
                    if y.is_nonpositive and d.is_positive:
                        if y:
                            return Dummy('neg', negative=True)
                        else:
                            return Dummy('npos', nonpositive=True)
        else:
            n, d = self.as_numer_denom()
            den = None
            if n.is_number:
                den = _monotonic_sign(d)
            elif not d.is_number:
                if _monotonic_sign(n) is not None:
                    den = _monotonic_sign(d)
            if den is not None and (den.is_positive or den.is_negative):
                v = n*den
                if v.is_positive:
                    return Dummy('pos', positive=True)
                elif v.is_nonnegative:
                    return Dummy('nneg', nonnegative=True)
                elif v.is_negative:
                    return Dummy('neg', negative=True)
                elif v.is_nonpositive:
                    return Dummy('npos', nonpositive=True)
        return None

    # multivariate
    c, a = self.as_coeff_Add()
    v = None
    if not a.is_polynomial():
        # F/A or A/F where A is a number and F is a signed, rational monomial
        n, d = a.as_numer_denom()
        if not (n.is_number or d.is_number):
            return
        if (
                a.is_Mul or a.is_Pow) and \
                a.is_rational and \
                all(p.exp.is_Integer for p in a.atoms(Pow) if p.is_Pow) and \
                (a.is_positive or a.is_negative):
            v = S.One
            for ai in Mul.make_args(a):
                if ai.is_number:
                    v *= ai
                    continue
                reps = {}
                for x in ai.free_symbols:
                    reps[x] = _monotonic_sign(x)
                    if reps[x] is None:
                        return
                v *= ai.subs(reps)
    elif c:
        # signed linear expression
        if not any(p for p in a.atoms(Pow) if not p.is_number) and (a.is_nonpositive or a.is_nonnegative):
            free = list(a.free_symbols)
            p = {}
            for i in free:
                v = _monotonic_sign(i)
                if v is None:
                    return
                p[i] = v or (_eps if i.is_nonnegative else -_eps)
            v = a.xreplace(p)
    if v is not None:
        rv = v + c
        if v.is_nonnegative and rv.is_positive:
            return rv.subs(_eps, 0)
        if v.is_nonpositive and rv.is_negative:
            return rv.subs(_eps, 0)


def decompose_power(expr: Expr) -> tuple[Expr, int]:
    """
    Decompose power into symbolic base and integer exponent.

    Examples
    ========

    >>> from sympy.core.exprtools import decompose_power
    >>> from sympy.abc import x, y
    >>> from sympy import exp

    >>> decompose_power(x)
    (x, 1)
    >>> decompose_power(x**2)
    (x, 2)
    >>> decompose_power(exp(2*y/3))
    (exp(y/3), 2)

    """
    base, exp = expr.as_base_exp()

    if exp.is_Number:
        if exp.is_Rational:
            if not exp.is_Integer:
                base = Pow(base, Rational(1, exp.q))  # type: ignore
            e = exp.p  # type: ignore
        else:
            base, e = expr, 1
    else:
        exp, tail = exp.as_coeff_Mul(rational=True)

        if exp is S.NegativeOne:
            base, e = Pow(base, tail), -1
        elif exp is not S.One:
            # todo: after dropping python 3.7 support, use overload and Literal
            #  in as_coeff_Mul to make exp Rational, and remove these 2 ignores
            tail = _keep_coeff(Rational(1, exp.q), tail)  # type: ignore
            base, e = Pow(base, tail), exp.p  # type: ignore
        else:
            base, e = expr, 1

    return base, e


def decompose_power_rat(expr: Expr) -> tuple[Expr, Rational]:
    """
    Decompose power into symbolic base and rational exponent;
    if the exponent is not a Rational, then separate only the
    integer coefficient.

    Examples
    ========

    >>> from sympy.core.exprtools import decompose_power_rat
    >>> from sympy.abc import x
    >>> from sympy import sqrt, exp

    >>> decompose_power_rat(sqrt(x))
    (x, 1/2)
    >>> decompose_power_rat(exp(-3*x/2))
    (exp(x/2), -3)

    """
    base, exp = expr.as_base_exp()
    if not exp.is_Rational:
        base, exp_i = decompose_power(expr)
        exp = Integer(exp_i)
    return base, exp # type: ignore


class Factors:
    """Efficient representation of ``f_1*f_2*...*f_n``."""

    __slots__ = ('factors', 'gens')

    def __init__(self, factors=None):  # Factors
        """Initialize Factors from dict or expr.

        Examples
        ========

        >>> from sympy.core.exprtools import Factors
        >>> from sympy.abc import x
        >>> from sympy import I
        >>> e = 2*x**3
        >>> Factors(e)
        Factors({2: 1, x: 3})
        >>> Factors(e.as_powers_dict())
        Factors({2: 1, x: 3})
        >>> f = _
        >>> f.factors  # underlying dictionary
        {2: 1, x: 3}
        >>> f.gens  # base of each factor
        frozenset({2, x})
        >>> Factors(0)
        Factors({0: 1})
        >>> Factors(I)
        Factors({I: 1})

        Notes
        =====

        Although a dictionary can be passed, only minimal checking is
        performed: powers of -1 and I are made canonical.

        """
        if isinstance(factors, (SYMPY_INTS, float)):
            factors = S(factors)
        if isinstance(factors, Factors):
            factors = factors.factors.copy()
        elif factors in (None, S.One):
            factors = {}
        elif factors is S.Zero or factors == 0:
            factors = {S.Zero: S.One}
        elif isinstance(factors, Number):
            n = factors
            factors = {}
            if n < 0:
                factors[S.NegativeOne] = S.One
                n = -n
            if n is not S.One:
                if n.is_Float or n.is_Integer or n is S.Infinity:
                    factors[n] = S.One
                elif n.is_Rational:
                    # since we're processing Numbers, the denominator is
                    # stored with a negative exponent; all other factors
                    # are left .
                    if n.p != 1:
                        factors[Integer(n.p)] = S.One
                    factors[Integer(n.q)] = S.NegativeOne
                else:
                    raise ValueError('Expected Float|Rational|Integer, not %s' % n)
        elif isinstance(factors, Basic) and not factors.args:
            factors = {factors: S.One}
        elif isinstance(factors, Expr):
            c, nc = factors.args_cnc()
            i = c.count(I)
            for _ in range(i):
                c.remove(I)
            factors = dict(Mul._from_args(c).as_powers_dict())
            # Handle all rational Coefficients
            for f in list(factors.keys()):
                if isinstance(f, Rational) and not isinstance(f, Integer):
                    p, q = Integer(f.p), Integer(f.q)
                    factors[p] = (factors[p] if p in factors else S.Zero) + factors[f]
                    factors[q] = (factors[q] if q in factors else S.Zero) - factors[f]
                    factors.pop(f)
            if i:
                factors[I] = factors.get(I, S.Zero) + i
            if nc:
                factors[Mul(*nc, evaluate=False)] = S.One
        else:
            factors = factors.copy()  # /!\ should be dict-like

            # tidy up -/+1 and I exponents if Rational

            handle = [k for k in factors if k is I or k in (-1, 1)]
            if handle:
                i1 = S.One
                for k in handle:
                    if not _isnumber(factors[k]):
                        continue
                    i1 *= k**factors.pop(k)
                if i1 is not S.One:
                    for a in i1.args if i1.is_Mul else [i1]:  # at worst, -1.0*I*(-1)**e
                        if a is S.NegativeOne:
                            factors[a] = S.One
                        elif a is I:
                            factors[I] = S.One
                        elif a.is_Pow:
                            factors[a.base] = factors.get(a.base, S.Zero) + a.exp
                        elif equal_valued(a, 1):
                            factors[a] = S.One
                        elif equal_valued(a, -1):
                            factors[-a] = S.One
                            factors[S.NegativeOne] = S.One
                        else:
                            raise ValueError('unexpected factor in i1: %s' % a)

        self.factors = factors
        keys = getattr(factors, 'keys', None)
        if keys is None:
            raise TypeError('expecting Expr or dictionary')
        self.gens = frozenset(keys())

    def __hash__(self):  # Factors
        keys = tuple(ordered(self.factors.keys()))
        values = [self.factors[k] for k in keys]
        return hash((keys, values))

    def __repr__(self):  # Factors
        return "Factors({%s})" % ', '.join(
            ['%s: %s' % (k, v) for k, v in ordered(self.factors.items())])

    @property
    def is_zero(self):  # Factors
        """
        >>> from sympy.core.exprtools import Factors
        >>> Factors(0).is_zero
        True
        """
        f = self.factors
        return len(f) == 1 and S.Zero in f

    @property
    def is_one(self):  # Factors
        """
        >>> from sympy.core.exprtools import Factors
        >>> Factors(1).is_one
        True
        """
        return not self.factors

    def as_expr(self):  # Factors
        """Return the underlying expression.

        Examples
        ========

        >>> from sympy.core.exprtools import Factors
        >>> from sympy.abc import x, y
        >>> Factors((x*y**2).as_powers_dict()).as_expr()
        x*y**2

        """

        args = []
        for factor, exp in self.factors.items():
            if exp != 1:
                if isinstance(exp, Integer):
                    b, e = factor.as_base_exp()
                    e = _keep_coeff(exp, e)
                    args.append(b**e)
                else:
                    args.append(factor**exp)
            else:
                args.append(factor)
        return Mul(*args)

    def mul(self, other):  # Factors
        """Return Factors of ``self * other``.

        Examples
        ========

        >>> from sympy.core.exprtools import Factors
        >>> from sympy.abc import x, y, z
        >>> a = Factors((x*y**2).as_powers_dict())
        >>> b = Factors((x*y/z).as_powers_dict())
        >>> a.mul(b)
        Factors({x: 2, y: 3, z: -1})
        >>> a*b
        Factors({x: 2, y: 3, z: -1})
        """
        if not isinstance(other, Factors):
            other = Factors(other)
        if any(f.is_zero for f in (self, other)):
            return Factors(S.Zero)
        factors = dict(self.factors)

        for factor, exp in other.factors.items():
            if factor in factors:
                exp = factors[factor] + exp

                if not exp:
                    del factors[factor]
                    continue

            factors[factor] = exp

        return Factors(factors)

    def normal(self, other):
        """Return ``self`` and ``other`` with ``gcd`` removed from each.
        The only differences between this and method ``div`` is that this
        is 1) optimized for the case when there are few factors in common and
        2) this does not raise an error if ``other`` is zero.

        See Also
        ========
        div

        """
        if not isinstance(other, Factors):
            other = Factors(other)
            if other.is_zero:
                return (Factors(), Factors(S.Zero))
            if self.is_zero:
                return (Factors(S.Zero), Factors())

        self_factors = dict(self.factors)
        other_factors = dict(other.factors)

        for factor, self_exp in self.factors.items():
            try:
                other_exp = other.factors[factor]
            except KeyError:
                continue

            exp = self_exp - other_exp

            if not exp:
                del self_factors[factor]
                del other_factors[factor]
            elif _isnumber(exp):
                if exp > 0:
                    self_factors[factor] = exp
                    del other_factors[factor]
                else:
                    del self_factors[factor]
                    other_factors[factor] = -exp
            else:
                r = self_exp.extract_additively(other_exp)
                if r is not None:
                    if r:
                        self_factors[factor] = r
                        del other_factors[factor]
                    else:  # should be handled already
                        del self_factors[factor]
                        del other_factors[factor]
                else:
                    sc, sa = self_exp.as_coeff_Add()
                    if sc:
                        oc, oa = other_exp.as_coeff_Add()
                        diff = sc - oc
                        if diff > 0:
                            self_factors[factor] -= oc
                            other_exp = oa
                        elif diff < 0:
                            self_factors[factor] -= sc
                            other_factors[factor] -= sc
                            other_exp = oa - diff
                        else:
                            self_factors[factor] = sa
                            other_exp = oa
                    if other_exp:
                        other_factors[factor] = other_exp
                    else:
                        del other_factors[factor]

        return Factors(self_factors), Factors(other_factors)

    def div(self, other):  # Factors
        """Return ``self`` and ``other`` with ``gcd`` removed from each.
        This is optimized for the case when there are many factors in common.

        Examples
        ========

        >>> from sympy.core.exprtools import Factors
        >>> from sympy.abc import x, y, z
        >>> from sympy import S

        >>> a = Factors((x*y**2).as_powers_dict())
        >>> a.div(a)
        (Factors({}), Factors({}))
        >>> a.div(x*z)
        (Factors({y: 2}), Factors({z: 1}))

        The ``/`` operator only gives ``quo``:

        >>> a/x
        Factors({y: 2})

        Factors treats its factors as though they are all in the numerator, so
        if you violate this assumption the results will be correct but will
        not strictly correspond to the numerator and denominator of the ratio:

        >>> a.div(x/z)
        (Factors({y: 2}), Factors({z: -1}))

        Factors is also naive about bases: it does not attempt any denesting
        of Rational-base terms, for example the following does not become
        2**(2*x)/2.

        >>> Factors(2**(2*x + 2)).div(S(8))
        (Factors({2: 2*x + 2}), Factors({8: 1}))

        factor_terms can clean up such Rational-bases powers:

        >>> from sympy import factor_terms
        >>> n, d = Factors(2**(2*x + 2)).div(S(8))
        >>> n.as_expr()/d.as_expr()
        2**(2*x + 2)/8
        >>> factor_terms(_)
        2**(2*x)/2

        """
        quo, rem = dict(self.factors), {}

        if not isinstance(other, Factors):
            other = Factors(other)
            if other.is_zero:
                raise ZeroDivisionError
            if self.is_zero:
                return (Factors(S.Zero), Factors())

        for factor, exp in other.factors.items():
            if factor in quo:
                d = quo[factor] - exp
                if _isnumber(d):
                    if d <= 0:
                        del quo[factor]

                    if d >= 0:
                        if d:
                            quo[factor] = d

                        continue

                    exp = -d

                else:
                    r = quo[factor].extract_additively(exp)
                    if r is not None:
                        if r:
                            quo[factor] = r
                        else:  # should be handled already
                            del quo[factor]
                    else:
                        other_exp = exp
                        sc, sa = quo[factor].as_coeff_Add()
                        if sc:
                            oc, oa = other_exp.as_coeff_Add()
                            diff = sc - oc
                            if diff > 0:
                                quo[factor] -= oc
                                other_exp = oa
                            elif diff < 0:
                                quo[factor] -= sc
                                other_exp = oa - diff
                            else:
                                quo[factor] = sa
                                other_exp = oa
                        if other_exp:
                            rem[factor] = other_exp
                        else:
                            assert factor not in rem
                    continue

            rem[factor] = exp

        return Factors(quo), Factors(rem)

    def quo(self, other):  # Factors
        """Return numerator Factor of ``self / other``.

        Examples
        ========

        >>> from sympy.core.exprtools import Factors
        >>> from sympy.abc import x, y, z
        >>> a = Factors((x*y**2).as_powers_dict())
        >>> b = Factors((x*y/z).as_powers_dict())
        >>> a.quo(b)  # same as a/b
        Factors({y: 1})
        """
        return self.div(other)[0]

    def rem(self, other):  # Factors
        """Return denominator Factors of ``self / other``.

        Examples
        ========

        >>> from sympy.core.exprtools import Factors
        >>> from sympy.abc import x, y, z
        >>> a = Factors((x*y**2).as_powers_dict())
        >>> b = Factors((x*y/z).as_powers_dict())
        >>> a.rem(b)
        Factors({z: -1})
        >>> a.rem(a)
        Factors({})
        """
        return self.div(other)[1]

    def pow(self, other):  # Factors
        """Return self raised to a non-negative integer power.

        Examples
        ========

        >>> from sympy.core.exprtools import Factors
        >>> from sympy.abc import x, y
        >>> a = Factors((x*y**2).as_powers_dict())
        >>> a**2
        Factors({x: 2, y: 4})

        """
        if isinstance(other, Factors):
            other = other.as_expr()
            if other.is_Integer:
                other = int(other)
        if isinstance(other, SYMPY_INTS) and other >= 0:
            factors = {}

            if other:
                for factor, exp in self.factors.items():
                    factors[factor] = exp*other

            return Factors(factors)
        else:
            raise ValueError("expected non-negative integer, got %s" % other)

    def gcd(self, other):  # Factors
        """Return Factors of ``gcd(self, other)``. The keys are
        the intersection of factors with the minimum exponent for
        each factor.

        Examples
        ========

        >>> from sympy.core.exprtools import Factors
        >>> from sympy.abc import x, y, z
        >>> a = Factors((x*y**2).as_powers_dict())
        >>> b = Factors((x*y/z).as_powers_dict())
        >>> a.gcd(b)
        Factors({x: 1, y: 1})
        """
        if not isinstance(other, Factors):
            other = Factors(other)
            if other.is_zero:
                return Factors(self.factors)

        factors = {}

        for factor, exp in self.factors.items():
            factor, exp = sympify(factor), sympify(exp)
            if factor in other.factors:
                lt = (exp - other.factors[factor]).is_negative
                if lt == True:
                    factors[factor] = exp
                elif lt == False:
                    factors[factor] = other.factors[factor]

        return Factors(factors)

    def lcm(self, other):  # Factors
        """Return Factors of ``lcm(self, other)`` which are
        the union of factors with the maximum exponent for
        each factor.

        Examples
        ========

        >>> from sympy.core.exprtools import Factors
        >>> from sympy.abc import x, y, z
        >>> a = Factors((x*y**2).as_powers_dict())
        >>> b = Factors((x*y/z).as_powers_dict())
        >>> a.lcm(b)
        Factors({x: 1, y: 2, z: -1})
        """
        if not isinstance(other, Factors):
            other = Factors(other)
            if any(f.is_zero for f in (self, other)):
                return Factors(S.Zero)

        factors = dict(self.factors)

        for factor, exp in other.factors.items():
            if factor in factors:
                exp = max(exp, factors[factor])

            factors[factor] = exp

        return Factors(factors)

    def __mul__(self, other):  # Factors
        return self.mul(other)

    def __divmod__(self, other):  # Factors
        return self.div(other)

    def __truediv__(self, other):  # Factors
        return self.quo(other)

    def __mod__(self, other):  # Factors
        return self.rem(other)

    def __pow__(self, other):  # Factors
        return self.pow(other)

    def __eq__(self, other):  # Factors
        if not isinstance(other, Factors):
            other = Factors(other)
        return self.factors == other.factors

    def __ne__(self, other):  # Factors
        return not self == other


class Term:
    """Efficient representation of ``coeff*(numer/denom)``. """

    __slots__ = ('coeff', 'numer', 'denom')

    def __init__(self, term, numer=None, denom=None):  # Term
        if numer is None and denom is None:
            if not term.is_commutative:
                raise NonCommutativeExpression(
                    'commutative expression expected')

            coeff, factors = term.as_coeff_mul()
            numer, denom = defaultdict(int), defaultdict(int)

            for factor in factors:
                base, exp = decompose_power(factor)

                if base.is_Add:
                    cont, base = base.primitive()
                    coeff *= cont**exp

                if exp > 0:
                    numer[base] += exp
                else:
                    denom[base] += -exp

            numer = Factors(numer)
            denom = Factors(denom)
        else:
            coeff = term

            if numer is None:
                numer = Factors()

            if denom is None:
                denom = Factors()

        self.coeff = coeff
        self.numer = numer
        self.denom = denom

    def __hash__(self):  # Term
        return hash((self.coeff, self.numer, self.denom))

    def __repr__(self):  # Term
        return "Term(%s, %s, %s)" % (self.coeff, self.numer, self.denom)

    def as_expr(self):  # Term
        return self.coeff*(self.numer.as_expr()/self.denom.as_expr())

    def mul(self, other):  # Term
        coeff = self.coeff*other.coeff
        numer = self.numer.mul(other.numer)
        denom = self.denom.mul(other.denom)

        numer, denom = numer.normal(denom)

        return Term(coeff, numer, denom)

    def inv(self):  # Term
        return Term(1/self.coeff, self.denom, self.numer)

    def quo(self, other):  # Term
        return self.mul(other.inv())

    def pow(self, other):  # Term
        if other < 0:
            return self.inv().pow(-other)
        else:
            return Term(self.coeff ** other,
                        self.numer.pow(other),
                        self.denom.pow(other))

    def gcd(self, other):  # Term
        return Term(self.coeff.gcd(other.coeff),
                    self.numer.gcd(other.numer),
                    self.denom.gcd(other.denom))

    def lcm(self, other):  # Term
        return Term(self.coeff.lcm(other.coeff),
                    self.numer.lcm(other.numer),
                    self.denom.lcm(other.denom))

    def __mul__(self, other):  # Term
        if isinstance(other, Term):
            return self.mul(other)
        else:
            return NotImplemented

    def __truediv__(self, other):  # Term
        if isinstance(other, Term):
            return self.quo(other)
        else:
            return NotImplemented

    def __pow__(self, other):  # Term
        if isinstance(other, SYMPY_INTS):
            return self.pow(other)
        else:
            return NotImplemented

    def __eq__(self, other):  # Term
        return (self.coeff == other.coeff and
                self.numer == other.numer and
                self.denom == other.denom)

    def __ne__(self, other):  # Term
        return not self == other


def _gcd_terms(terms, isprimitive=False, fraction=True):
    """Helper function for :func:`gcd_terms`.

    Parameters
    ==========

    isprimitive : boolean, optional
        If ``isprimitive`` is True then the call to primitive
        for an Add will be skipped. This is useful when the
        content has already been extracted.

    fraction : boolean, optional
        If ``fraction`` is True then the expression will appear over a common
        denominator, the lcm of all term denominators.
    """

    if isinstance(terms, Basic) and not isinstance(terms, Tuple):
        terms = Add.make_args(terms)

    terms = list(map(Term, [t for t in terms if t]))

    # there is some simplification that may happen if we leave this
    # here rather than duplicate it before the mapping of Term onto
    # the terms
    if len(terms) == 0:
        return S.Zero, S.Zero, S.One

    if len(terms) == 1:
        cont = terms[0].coeff
        numer = terms[0].numer.as_expr()
        denom = terms[0].denom.as_expr()

    else:
        cont = terms[0]
        for term in terms[1:]:
            cont = cont.gcd(term)

        for i, term in enumerate(terms):
            terms[i] = term.quo(cont)

        if fraction:
            denom = terms[0].denom

            for term in terms[1:]:
                denom = denom.lcm(term.denom)

            numers = []
            for term in terms:
                numer = term.numer.mul(denom.quo(term.denom))
                numers.append(term.coeff*numer.as_expr())
        else:
            numers = [t.as_expr() for t in terms]
            denom = Term(S.One).numer

        cont = cont.as_expr()
        numer = Add(*numers)
        denom = denom.as_expr()

    if not isprimitive and numer.is_Add:
        _cont, numer = numer.primitive()
        cont *= _cont

    return cont, numer, denom


def gcd_terms(terms, isprimitive=False, clear=True, fraction=True):
    """Compute the GCD of ``terms`` and put them together.

    Parameters
    ==========

    terms : Expr
        Can be an expression or a non-Basic sequence of expressions
        which will be handled as though they are terms from a sum.

    isprimitive : bool, optional
        If ``isprimitive`` is True the _gcd_terms will not run the primitive
        method on the terms.

    clear : bool, optional
        It controls the removal of integers from the denominator of an Add
        expression. When True (default), all numerical denominator will be cleared;
        when False the denominators will be cleared only if all terms had numerical
        denominators other than 1.

    fraction : bool, optional
        When True (default), will put the expression over a common
        denominator.

    Examples
    ========

    >>> from sympy import gcd_terms
    >>> from sympy.abc import x, y

    >>> gcd_terms((x + 1)**2*y + (x + 1)*y**2)
    y*(x + 1)*(x + y + 1)
    >>> gcd_terms(x/2 + 1)
    (x + 2)/2
    >>> gcd_terms(x/2 + 1, clear=False)
    x/2 + 1
    >>> gcd_terms(x/2 + y/2, clear=False)
    (x + y)/2
    >>> gcd_terms(x/2 + 1/x)
    (x**2 + 2)/(2*x)
    >>> gcd_terms(x/2 + 1/x, fraction=False)
    (x + 2/x)/2
    >>> gcd_terms(x/2 + 1/x, fraction=False, clear=False)
    x/2 + 1/x

    >>> gcd_terms(x/2/y + 1/x/y)
    (x**2 + 2)/(2*x*y)
    >>> gcd_terms(x/2/y + 1/x/y, clear=False)
    (x**2/2 + 1)/(x*y)
    >>> gcd_terms(x/2/y + 1/x/y, clear=False, fraction=False)
    (x/2 + 1/x)/y

    The ``clear`` flag was ignored in this case because the returned
    expression was a rational expression, not a simple sum.

    See Also
    ========

    factor_terms, sympy.polys.polytools.terms_gcd

    """
    def mask(terms):
        """replace nc portions of each term with a unique Dummy symbols
        and return the replacements to restore them"""
        args = [(a, []) if a.is_commutative else a.args_cnc() for a in terms]
        reps = []
        for i, (c, nc) in enumerate(args):
            if nc:
                nc = Mul(*nc)
                d = Dummy()
                reps.append((d, nc))
                c.append(d)
                args[i] = Mul(*c)
            else:
                args[i] = c
        return args, dict(reps)

    isadd = isinstance(terms, Add)
    addlike = isadd or not isinstance(terms, Basic) and \
        is_sequence(terms, include=set) and \
        not isinstance(terms, Dict)

    if addlike:
        if isadd:  # i.e. an Add
            terms = list(terms.args)
        else:
            terms = sympify(terms)
        terms, reps = mask(terms)
        cont, numer, denom = _gcd_terms(terms, isprimitive, fraction)
        numer = numer.xreplace(reps)
        coeff, factors = cont.as_coeff_Mul()
        if not clear:
            c, _coeff = coeff.as_coeff_Mul()
            if not c.is_Integer and not clear and numer.is_Add:
                n, d = c.as_numer_denom()
                _numer = numer/d
                if any(a.as_coeff_Mul()[0].is_Integer
                        for a in _numer.args):
                    numer = _numer
                    coeff = n*_coeff
        return _keep_coeff(coeff, factors*numer/denom, clear=clear)

    if not isinstance(terms, Basic):
        return terms

    if terms.is_Atom:
        return terms

    if terms.is_Mul:
        c, args = terms.as_coeff_mul()
        return _keep_coeff(c, Mul(*[gcd_terms(i, isprimitive, clear, fraction)
            for i in args]), clear=clear)

    def handle(a):
        # don't treat internal args like terms of an Add
        if not isinstance(a, Expr):
            if isinstance(a, Basic):
                if not a.args:
                    return a
                return a.func(*[handle(i) for i in a.args])
            return type(a)([handle(i) for i in a])
        return gcd_terms(a, isprimitive, clear, fraction)

    if isinstance(terms, Dict):
        return Dict(*[(k, handle(v)) for k, v in terms.args])
    return terms.func(*[handle(i) for i in terms.args])


def _factor_sum_int(expr, **kwargs):
    """Return Sum or Integral object with factors that are not
    in the wrt variables removed. In cases where there are additive
    terms in the function of the object that are independent, the
    object will be separated into two objects.

    Examples
    ========

    >>> from sympy import Sum, factor_terms
    >>> from sympy.abc import x, y
    >>> factor_terms(Sum(x + y, (x, 1, 3)))
    y*Sum(1, (x, 1, 3)) + Sum(x, (x, 1, 3))
    >>> factor_terms(Sum(x*y, (x, 1, 3)))
    y*Sum(x, (x, 1, 3))

    Notes
    =====

    If a function in the summand or integrand is replaced
    with a symbol, then this simplification should not be
    done or else an incorrect result will be obtained when
    the symbol is replaced with an expression that depends
    on the variables of summation/integration:

    >>> eq = Sum(y, (x, 1, 3))
    >>> factor_terms(eq).subs(y, x).doit()
    3*x
    >>> eq.subs(y, x).doit()
    6
    """
    result = expr.function
    if result == 0:
        return S.Zero
    limits = expr.limits

    # get the wrt variables
    wrt = {i.args[0] for i in limits}

    # factor out any common terms that are independent of wrt
    f = factor_terms(result, **kwargs)
    i, d = f.as_independent(*wrt)
    if isinstance(f, Add):
        return i * expr.func(1, *limits) + expr.func(d, *limits)
    else:
        return i * expr.func(d, *limits)


def factor_terms(expr: Expr | complex, radical=False, clear=False, fraction=False, sign=True) -> Expr:
    """Remove common factors from terms in all arguments without
    changing the underlying structure of the expr. No expansion or
    simplification (and no processing of non-commutatives) is performed.

    Parameters
    ==========

    radical: bool, optional
        If radical=True then a radical common to all terms will be factored
        out of any Add sub-expressions of the expr.

    clear : bool, optional
        If clear=False (default) then coefficients will not be separated
        from a single Add if they can be distributed to leave one or more
        terms with integer coefficients.

    fraction : bool, optional
        If fraction=True (default is False) then a common denominator will be
        constructed for the expression.

    sign : bool, optional
        If sign=True (default) then even if the only factor in common is a -1,
        it will be factored out of the expression.

    Examples
    ========

    >>> from sympy import factor_terms, Symbol
    >>> from sympy.abc import x, y
    >>> factor_terms(x + x*(2 + 4*y)**3)
    x*(8*(2*y + 1)**3 + 1)
    >>> A = Symbol('A', commutative=False)
    >>> factor_terms(x*A + x*A + x*y*A)
    x*(y*A + 2*A)

    When ``clear`` is False, a rational will only be factored out of an
    Add expression if all terms of the Add have coefficients that are
    fractions:

    >>> factor_terms(x/2 + 1, clear=False)
    x/2 + 1
    >>> factor_terms(x/2 + 1, clear=True)
    (x + 2)/2

    If a -1 is all that can be factored out, to *not* factor it out, the
    flag ``sign`` must be False:

    >>> factor_terms(-x - y)
    -(x + y)
    >>> factor_terms(-x - y, sign=False)
    -x - y
    >>> factor_terms(-2*x - 2*y, sign=False)
    -2*(x + y)

    See Also
    ========

    gcd_terms, sympy.polys.polytools.terms_gcd

    """
    def do(expr):
        from sympy.concrete.summations import Sum
        from sympy.integrals.integrals import Integral
        is_iterable = iterable(expr)

        if not isinstance(expr, Basic) or expr.is_Atom:
            if is_iterable:
                return type(expr)([do(i) for i in expr])
            return expr

        if expr.is_Pow or expr.is_Function or \
                is_iterable or not hasattr(expr, 'args_cnc'):
            args = expr.args
            newargs = tuple([do(i) for i in args])
            if newargs == args:
                return expr
            return expr.func(*newargs)

        if isinstance(expr, (Sum, Integral)):
            return _factor_sum_int(expr,
                radical=radical, clear=clear,
                fraction=fraction, sign=sign)

        cont, p = expr.as_content_primitive(radical=radical, clear=clear)
        if p.is_Add:
            list_args = [do(a) for a in Add.make_args(p)]
            # get a common negative (if there) which gcd_terms does not remove
            if not any(a.as_coeff_Mul()[0].extract_multiplicatively(-1) is None
                       for a in list_args):
                cont = -cont
                list_args = [-a for a in list_args]
            # watch out for exp(-(x+2)) which gcd_terms will change to exp(-x-2)
            special = {}
            for i, a in enumerate(list_args):
                b, e = a.as_base_exp()
                if e.is_Mul and e != Mul(*e.args):
                    list_args[i] = Dummy()
                    special[list_args[i]] = a
            # rebuild p not worrying about the order which gcd_terms will fix
            p = Add._from_args(list_args)
            p = gcd_terms(p,
                isprimitive=True,
                clear=clear,
                fraction=fraction).xreplace(special)
        elif p.args:
            p = p.func(
                *[do(a) for a in p.args])
        rv = _keep_coeff(cont, p, clear=clear, sign=sign)
        return rv
    expr2 = sympify(expr)
    return do(expr2)


def _mask_nc(eq, name=None):
    """
    Return ``eq`` with non-commutative objects replaced with Dummy
    symbols. A dictionary that can be used to restore the original
    values is returned: if it is None, the expression is noncommutative
    and cannot be made commutative. The third value returned is a list
    of any non-commutative symbols that appear in the returned equation.

    Explanation
    ===========

    All non-commutative objects other than Symbols are replaced with
    a non-commutative Symbol. Identical objects will be identified
    by identical symbols.

    If there is only 1 non-commutative object in an expression it will
    be replaced with a commutative symbol. Otherwise, the non-commutative
    entities are retained and the calling routine should handle
    replacements in this case since some care must be taken to keep
    track of the ordering of symbols when they occur within Muls.

    Parameters
    ==========

    name : str
        ``name``, if given, is the name that will be used with numbered Dummy
        variables that will replace the non-commutative objects and is mainly
        used for doctesting purposes.

    Examples
    ========

    >>> from sympy.physics.secondquant import Commutator, NO, F, Fd
    >>> from sympy import symbols
    >>> from sympy.core.exprtools import _mask_nc
    >>> from sympy.abc import x, y
    >>> A, B, C = symbols('A,B,C', commutative=False)

    One nc-symbol:

    >>> _mask_nc(A**2 - x**2, 'd')
    (_d0**2 - x**2, {_d0: A}, [])

    Multiple nc-symbols:

    >>> _mask_nc(A**2 - B**2, 'd')
    (A**2 - B**2, {}, [A, B])

    An nc-object with nc-symbols but no others outside of it:

    >>> _mask_nc(1 + x*Commutator(A, B), 'd')
    (_d0*x + 1, {_d0: Commutator(A, B)}, [])
    >>> _mask_nc(NO(Fd(x)*F(y)), 'd')
    (_d0, {_d0: NO(CreateFermion(x)*AnnihilateFermion(y))}, [])

    Multiple nc-objects:

    >>> eq = x*Commutator(A, B) + x*Commutator(A, C)*Commutator(A, B)
    >>> _mask_nc(eq, 'd')
    (x*_d0 + x*_d1*_d0, {_d0: Commutator(A, B), _d1: Commutator(A, C)}, [_d0, _d1])

    Multiple nc-objects and nc-symbols:

    >>> eq = A*Commutator(A, B) + B*Commutator(A, C)
    >>> _mask_nc(eq, 'd')
    (A*_d0 + B*_d1, {_d0: Commutator(A, B), _d1: Commutator(A, C)}, [_d0, _d1, A, B])

    """
    name = name or 'mask'
    # Make Dummy() append sequential numbers to the name

    def numbered_names():
        i = 0
        while True:
            yield name + str(i)
            i += 1

    names = numbered_names()

    def Dummy(*args, **kwargs):
        from .symbol import Dummy
        return Dummy(next(names), *args, **kwargs)

    expr = eq
    if expr.is_commutative:
        return eq, {}, []

    # identify nc-objects; symbols and other
    rep = []
    nc_obj = set()
    nc_syms = set()
    pot = preorder_traversal(expr, keys=default_sort_key)
    for a in pot:
        if any(a == r[0] for r in rep):
            pot.skip()
        elif not a.is_commutative:
            if a.is_symbol:
                nc_syms.add(a)
                pot.skip()
            elif not (a.is_Add or a.is_Mul or a.is_Pow):
                nc_obj.add(a)
                pot.skip()

    # If there is only one nc symbol or object, it can be factored regularly
    # but polys is going to complain, so replace it with a Dummy.
    if len(nc_obj) == 1 and not nc_syms:
        rep.append((nc_obj.pop(), Dummy()))
    elif len(nc_syms) == 1 and not nc_obj:
        rep.append((nc_syms.pop(), Dummy()))

    # Any remaining nc-objects will be replaced with an nc-Dummy and
    # identified as an nc-Symbol to watch out for
    nc_obj = sorted(nc_obj, key=default_sort_key)
    for n in nc_obj:
        nc = Dummy(commutative=False)
        rep.append((n, nc))
        nc_syms.add(nc)
    expr = expr.subs(rep)

    nc_syms = list(nc_syms)
    nc_syms.sort(key=default_sort_key)
    return expr, {v: k for k, v in rep}, nc_syms


def factor_nc(expr):
    """Return the factored form of ``expr`` while handling non-commutative
    expressions.

    Examples
    ========

    >>> from sympy import factor_nc, Symbol
    >>> from sympy.abc import x
    >>> A = Symbol('A', commutative=False)
    >>> B = Symbol('B', commutative=False)
    >>> factor_nc((x**2 + 2*A*x + A**2).expand())
    (x + A)**2
    >>> factor_nc(((x + A)*(x + B)).expand())
    (x + A)*(x + B)
    """
    expr = sympify(expr)
    if not isinstance(expr, Expr) or not expr.args:
        return expr
    if not expr.is_Add:
        return expr.func(*[factor_nc(a) for a in expr.args])
    expr = expr.func(*[expand_power_exp(i) for i in expr.args])

    from sympy.polys.polytools import gcd, factor
    expr, rep, nc_symbols = _mask_nc(expr)

    if rep:
        return factor(expr).subs(rep)
    else:
        args = [a.args_cnc() for a in Add.make_args(expr)]
        c = g = l = r = S.One
        hit = False
        # find any commutative gcd term
        for i, a in enumerate(args):
            if i == 0:
                c = Mul._from_args(a[0])
            elif a[0]:
                c = gcd(c, Mul._from_args(a[0]))
            else:
                c = S.One
        if c is not S.One:
            hit = True
            c, g = c.as_coeff_Mul()
            if g is not S.One:
                for i, (cc, _) in enumerate(args):
                    cc = list(Mul.make_args(Mul._from_args(list(cc))/g))
                    args[i][0] = cc
            for i, (cc, _) in enumerate(args):
                if cc:
                    cc[0] = cc[0]/c
                else:
                    cc = [1/c]
                args[i][0] = cc
        # find any noncommutative common prefix
        for i, a in enumerate(args):
            if i == 0:
                n = a[1][:]
            else:
                n = common_prefix(n, a[1])
            if not n:
                # is there a power that can be extracted?
                if not args[0][1]:
                    break
                b, e = args[0][1][0].as_base_exp()
                ok = False
                if e.is_Integer:
                    for t in args:
                        if not t[1]:
                            break
                        bt, et = t[1][0].as_base_exp()
                        if et.is_Integer and bt == b:
                            e = min(e, et)
                        else:
                            break
                    else:
                        ok = hit = True
                        l = b**e
                        il = b**-e
                        for _ in args:
                            _[1][0] = il*_[1][0]
                        break
                if not ok:
                    break
        else:
            hit = True
            lenn = len(n)
            l = Mul(*n)
            for _ in args:
                _[1] = _[1][lenn:]
        # find any noncommutative common suffix
        for i, a in enumerate(args):
            if i == 0:
                n = a[1][:]
            else:
                n = common_suffix(n, a[1])
            if not n:
                # is there a power that can be extracted?
                if not args[0][1]:
                    break
                b, e = args[0][1][-1].as_base_exp()
                ok = False
                if e.is_Integer:
                    for t in args:
                        if not t[1]:
                            break
                        bt, et = t[1][-1].as_base_exp()
                        if et.is_Integer and bt == b:
                            e = min(e, et)
                        else:
                            break
                    else:
                        ok = hit = True
                        r = b**e
                        il = b**-e
                        for _ in args:
                            _[1][-1] = _[1][-1]*il
                        break
                if not ok:
                    break
        else:
            hit = True
            lenn = len(n)
            r = Mul(*n)
            for _ in args:
                _[1] = _[1][:len(_[1]) - lenn]
        if hit:
            mid = Add(*[Mul(*cc)*Mul(*nc) for cc, nc in args])
        else:
            mid = expr

        from sympy.simplify.powsimp import powsimp

        # sort the symbols so the Dummys would appear in the same
        # order as the original symbols, otherwise you may introduce
        # a factor of -1, e.g. A**2 - B**2) -- {A:y, B:x} --> y**2 - x**2
        # and the former factors into two terms, (A - B)*(A + B) while the
        # latter factors into 3 terms, (-1)*(x - y)*(x + y)
        rep1 = [(n, Dummy()) for n in sorted(nc_symbols, key=default_sort_key)]
        unrep1 = [(v, k) for k, v in rep1]
        unrep1.reverse()
        new_mid, r2, _ = _mask_nc(mid.subs(rep1))
        new_mid = powsimp(factor(new_mid))

        new_mid = new_mid.subs(r2).subs(unrep1)

        if new_mid.is_Pow:
            return _keep_coeff(c, g*l*new_mid*r)

        if new_mid.is_Mul:
            def _pemexpand(expr):
                "Expand with the minimal set of hints necessary to check the result."
                return expr.expand(deep=True, mul=True, power_exp=True,
                    power_base=False, basic=False, multinomial=True, log=False)
            # XXX TODO there should be a way to inspect what order the terms
            # must be in and just select the plausible ordering without
            # checking permutations
            cfac = []
            ncfac = []
            for f in new_mid.args:
                if f.is_commutative:
                    cfac.append(f)
                else:
                    b, e = f.as_base_exp()
                    if e.is_Integer:
                        ncfac.extend([b]*e)
                    else:
                        ncfac.append(f)
            pre_mid = g*Mul(*cfac)*l
            target = _pemexpand(expr/c)
            for s in variations(ncfac, len(ncfac)):
                ok = pre_mid*Mul(*s)*r
                if _pemexpand(ok) == target:
                    return _keep_coeff(c, ok)

        # mid was an Add that didn't factor successfully
        return _keep_coeff(c, g*l*mid*r)
<|endoftext|>from .add import Add
from .exprtools import gcd_terms
from .function import DefinedFunction
from .kind import NumberKind
from .logic import fuzzy_and, fuzzy_not
from .mul import Mul
from .numbers import equal_valued
from .relational import is_le, is_lt, is_ge, is_gt
from .singleton import S


class Mod(DefinedFunction):
    """Represents a modulo operation on symbolic expressions.

    Parameters
    ==========

    p : Expr
        Dividend.

    q : Expr
        Divisor.

    Notes
    =====

    The convention used is the same as Python's: the remainder always has the
    same sign as the divisor.

    Many objects can be evaluated modulo ``n`` much faster than they can be
    evaluated directly (or at all).  For this, ``evaluate=False`` is
    necessary to prevent eager evaluation:

    >>> from sympy import binomial, factorial, Mod, Pow
    >>> Mod(Pow(2, 10**16, evaluate=False), 97)
    61
    >>> Mod(factorial(10**9, evaluate=False), 10**9 + 9)
    712524808
    >>> Mod(binomial(10**18, 10**12, evaluate=False), (10**5 + 3)**2)
    3744312326

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> x**2 % y
    Mod(x**2, y)
    >>> _.subs({x: 5, y: 6})
    1

    """

    kind = NumberKind

    @classmethod
    def eval(cls, p, q):
        def number_eval(p, q):
            """Try to return p % q if both are numbers or +/-p is known
            to be less than or equal q.
            """

            if q.is_zero:
                raise ZeroDivisionError("Modulo by zero")
            if p is S.NaN or q is S.NaN or p.is_finite is False or q.is_finite is False:
                return S.NaN
            if p is S.Zero or p in (q, -q) or (p.is_integer and q == 1):
                return S.Zero

            if q.is_Number:
                if p.is_Number:
                    return p%q
                if q == 2:
                    if p.is_even:
                        return S.Zero
                    elif p.is_odd:
                        return S.One

            if hasattr(p, '_eval_Mod'):
                rv = getattr(p, '_eval_Mod')(q)
                if rv is not None:
                    return rv

            # by ratio
            r = p/q
            if r.is_integer:
                return S.Zero
            try:
                d = int(r)
            except TypeError:
                pass
            else:
                if isinstance(d, int):
                    rv = p - d*q
                    if (rv*q < 0) == True:
                        rv += q
                    return rv

            # by difference
            # -2|q| < p < 2|q|
            if q.is_positive:
                comp1, comp2 = is_le, is_lt
            elif q.is_negative:
                comp1, comp2 = is_ge, is_gt
            else:
                return
            ls = -2*q
            r = p - q
            for _ in range(4):
                if not comp1(ls, p):
                    return
                if comp2(r, ls):
                    return p - ls
                ls += q

        rv = number_eval(p, q)
        if rv is not None:
            return rv

        # denest
        if isinstance(p, cls):
            qinner = p.args[1]
            if qinner % q == 0:
                return cls(p.args[0], q)
            elif (qinner*(q - qinner)).is_nonnegative:
                # |qinner| < |q| and have same sign
                return p
        elif isinstance(-p, cls):
            qinner = (-p).args[1]
            if qinner % q == 0:
                return cls(-(-p).args[0], q)
            elif (qinner*(q + qinner)).is_nonpositive:
                # |qinner| < |q| and have different sign
                return p
        elif isinstance(p, Add):
            # separating into modulus and non modulus
            both_l = non_mod_l, mod_l = [], []
            for arg in p.args:
                both_l[isinstance(arg, cls)].append(arg)
            # if q same for all
            if mod_l and all(inner.args[1] == q for inner in mod_l):
                net = Add(*non_mod_l) + Add(*[i.args[0] for i in mod_l])
                return cls(net, q)

        elif isinstance(p, Mul):
            # separating into modulus and non modulus
            both_l = non_mod_l, mod_l = [], []
            for arg in p.args:
                both_l[isinstance(arg, cls)].append(arg)

            if mod_l and all(inner.args[1] == q for inner in mod_l) and all(t.is_integer for t in p.args) and q.is_integer:
                # finding distributive term
                non_mod_l = [cls(x, q) for x in non_mod_l]
                mod = []
                non_mod = []
                for j in non_mod_l:
                    if isinstance(j, cls):
                        mod.append(j.args[0])
                    else:
                        non_mod.append(j)
                prod_mod = Mul(*mod)
                prod_non_mod = Mul(*non_mod)
                prod_mod1 = Mul(*[i.args[0] for i in mod_l])
                net = prod_mod1*prod_mod
                return prod_non_mod*cls(net, q)

            if q.is_Integer and q is not S.One:
                if all(t.is_integer for t in p.args):
                    non_mod_l = [i % q if i.is_Integer else i for i in p.args]
                    if any(iq is S.Zero for iq in non_mod_l):
                        return S.Zero

            p = Mul(*(non_mod_l + mod_l))

        # XXX other possibilities?

        from sympy.polys.polyerrors import PolynomialError
        from sympy.polys.polytools import gcd

        # extract gcd; any further simplification should be done by the user
        try:
            G = gcd(p, q)
            if not equal_valued(G, 1):
                p, q = [gcd_terms(i/G, clear=False, fraction=False)
                        for i in (p, q)]
        except PolynomialError:  # issue 21373
            G = S.One
        pwas, qwas = p, q

        # simplify terms
        # (x + y + 2) % x -> Mod(y + 2, x)
        if p.is_Add:
            args = []
            for i in p.args:
                a = cls(i, q)
                if a.count(cls) > i.count(cls):
                    args.append(i)
                else:
                    args.append(a)
            if args != list(p.args):
                p = Add(*args)

        else:
            # handle coefficients if they are not Rational
            # since those are not handled by factor_terms
            # e.g. Mod(.6*x, .3*y) -> 0.3*Mod(2*x, y)
            cp, p = p.as_coeff_Mul()
            cq, q = q.as_coeff_Mul()
            ok = False
            if not cp.is_Rational or not cq.is_Rational:
                r = cp % cq
                if equal_valued(r, 0):
                    G *= cq
                    p *= int(cp/cq)
                    ok = True
            if not ok:
                p = cp*p
                q = cq*q

        # simple -1 extraction
        if p.could_extract_minus_sign() and q.could_extract_minus_sign():
            G, p, q = [-i for i in (G, p, q)]

        # check again to see if p and q can now be handled as numbers
        rv = number_eval(p, q)
        if rv is not None:
            return rv*G

        # put 1.0 from G on inside
        if G.is_Float and equal_valued(G, 1):
            p *= G
            return cls(p, q, evaluate=False)
        elif G.is_Mul and G.args[0].is_Float and equal_valued(G.args[0], 1):
            p = G.args[0]*p
            G = Mul._from_args(G.args[1:])
        return G*cls(p, q, evaluate=(p, q) != (pwas, qwas))

    def _eval_is_integer(self):
        p, q = self.args
        if fuzzy_and([p.is_integer, q.is_integer, fuzzy_not(q.is_zero)]):
            return True

    def _eval_is_nonnegative(self):
        if self.args[1].is_positive:
            return True

    def _eval_is_nonpositive(self):
        if self.args[1].is_negative:
            return True

    def _eval_rewrite_as_floor(self, a, b, **kwargs):
        from sympy.functions.elementary.integers import floor
        return a - b*floor(a/b)

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.functions.elementary.integers import floor
        return self.rewrite(floor)._eval_as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.functions.elementary.integers import floor
        return self.rewrite(floor)._eval_nseries(x, n, logx=logx, cdir=cdir)
<|endoftext|>from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar

from collections import defaultdict
from functools import reduce
from itertools import product
import operator

from .sympify import sympify
from .basic import Basic, _args_sortkey
from .singleton import S
from .operations import AssocOp, AssocOpDispatcher
from .cache import cacheit
from .intfunc import integer_nthroot, trailing
from .logic import fuzzy_not, _fuzzy_group
from .expr import Expr
from .parameters import global_parameters
from .kind import KindDispatcher
from .traversal import bottom_up
from sympy.utilities.iterables import sift


# internal marker to indicate:
#   "there are still non-commutative objects -- don't forget to process them"
class NC_Marker:
    is_Order = False
    is_Mul = False
    is_Number = False
    is_Poly = False

    is_commutative = False


def _mulsort(args):
    # in-place sorting of args
    args.sort(key=_args_sortkey)


def _unevaluated_Mul(*args):
    """Return a well-formed unevaluated Mul: Numbers are collected and
    put in slot 0, any arguments that are Muls will be flattened, and args
    are sorted. Use this when args have changed but you still want to return
    an unevaluated Mul.

    Examples
    ========

    >>> from sympy.core.mul import _unevaluated_Mul as uMul
    >>> from sympy import S, sqrt, Mul
    >>> from sympy.abc import x
    >>> a = uMul(*[S(3.0), x, S(2)])
    >>> a.args[0]
    6.00000000000000
    >>> a.args[1]
    x

    Two unevaluated Muls with the same arguments will
    always compare as equal during testing:

    >>> m = uMul(sqrt(2), sqrt(3))
    >>> m == uMul(sqrt(3), sqrt(2))
    True
    >>> u = Mul(sqrt(3), sqrt(2), evaluate=False)
    >>> m == uMul(u)
    True
    >>> m == Mul(*m.args)
    False

    """
    cargs = []
    ncargs = []
    args = list(args)
    co = S.One
    for a in args:
        if a.is_Mul:
            a_c, a_nc = a.args_cnc()
            args.extend(a_c)  # grow args
            ncargs.extend(a_nc)
        elif a.is_Number:
            co *= a
        elif a.is_commutative:
            cargs.append(a)
        else:
            ncargs.append(a)
    _mulsort(cargs)
    if co is not S.One:
        cargs.insert(0, co)
    return Mul._from_args(cargs+ncargs)


class Mul(Expr, AssocOp):
    """
    Expression representing multiplication operation for algebraic field.

    .. deprecated:: 1.7

       Using arguments that aren't subclasses of :class:`~.Expr` in core
       operators (:class:`~.Mul`, :class:`~.Add`, and :class:`~.Pow`) is
       deprecated. See :ref:`non-expr-args-deprecated` for details.

    Every argument of ``Mul()`` must be ``Expr``. Infix operator ``*``
    on most scalar objects in SymPy calls this class.

    Another use of ``Mul()`` is to represent the structure of abstract
    multiplication so that its arguments can be substituted to return
    different class. Refer to examples section for this.

    ``Mul()`` evaluates the argument unless ``evaluate=False`` is passed.
    The evaluation logic includes:

    1. Flattening
        ``Mul(x, Mul(y, z))`` -> ``Mul(x, y, z)``

    2. Identity removing
        ``Mul(x, 1, y)`` -> ``Mul(x, y)``

    3. Exponent collecting by ``.as_base_exp()``
        ``Mul(x, x**2)`` -> ``Pow(x, 3)``

    4. Term sorting
        ``Mul(y, x, 2)`` -> ``Mul(2, x, y)``

    Since multiplication can be vector space operation, arguments may
    have the different :obj:`sympy.core.kind.Kind()`. Kind of the
    resulting object is automatically inferred.

    Examples
    ========

    >>> from sympy import Mul
    >>> from sympy.abc import x, y
    >>> Mul(x, 1)
    x
    >>> Mul(x, x)
    x**2

    If ``evaluate=False`` is passed, result is not evaluated.

    >>> Mul(1, 2, evaluate=False)
    1*2
    >>> Mul(x, x, evaluate=False)
    x*x

    ``Mul()`` also represents the general structure of multiplication
    operation.

    >>> from sympy import MatrixSymbol
    >>> A = MatrixSymbol('A', 2,2)
    >>> expr = Mul(x,y).subs({y:A})
    >>> expr
    x*A
    >>> type(expr)
    <class 'sympy.matrices.expressions.matmul.MatMul'>

    See Also
    ========

    MatMul

    """
    __slots__ = ()

    is_Mul = True

    _args_type = Expr
    _kind_dispatcher = KindDispatcher("Mul_kind_dispatcher", commutative=True)

    identity: ClassVar[Expr]

    @property
    def kind(self):
        arg_kinds = (a.kind for a in self.args)
        return self._kind_dispatcher(*arg_kinds)

    if TYPE_CHECKING:

        def __new__(cls, *args: Expr | complex, evaluate: bool=True) -> Expr: # type: ignore
            ...

        @property
        def args(self) -> tuple[Expr, ...]:
            ...

    def could_extract_minus_sign(self):
        if self == (-self):
            return False  # e.g. zoo*x == -zoo*x
        c = self.args[0]
        return c.is_Number and c.is_extended_negative

    def __neg__(self):
        c, args = self.as_coeff_mul()
        if args[0] is not S.ComplexInfinity:
            c = -c
        if c is not S.One:
            if args[0].is_Number:
                args = list(args)
                if c is S.NegativeOne:
                    args[0] = -args[0]
                else:
                    args[0] *= c
            else:
                args = (c,) + args
        return self._from_args(args, self.is_commutative)

    @classmethod
    def flatten(cls, seq):
        """Return commutative, noncommutative and order arguments by
        combining related terms.

        Notes
        =====
            * In an expression like ``a*b*c``, Python process this through SymPy
              as ``Mul(Mul(a, b), c)``. This can have undesirable consequences.

              -  Sometimes terms are not combined as one would like:
                 {c.f. https://github.com/sympy/sympy/issues/4596}

                >>> from sympy import Mul, sqrt
                >>> from sympy.abc import x, y, z
                >>> 2*(x + 1) # this is the 2-arg Mul behavior
                2*x + 2
                >>> y*(x + 1)*2
                2*y*(x + 1)
                >>> 2*(x + 1)*y # 2-arg result will be obtained first
                y*(2*x + 2)
                >>> Mul(2, x + 1, y) # all 3 args simultaneously processed
                2*y*(x + 1)
                >>> 2*((x + 1)*y) # parentheses can control this behavior
                2*y*(x + 1)

                Powers with compound bases may not find a single base to
                combine with unless all arguments are processed at once.
                Post-processing may be necessary in such cases.
                {c.f. https://github.com/sympy/sympy/issues/5728}

                >>> a = sqrt(x*sqrt(y))
                >>> a**3
                (x*sqrt(y))**(3/2)
                >>> Mul(a,a,a)
                (x*sqrt(y))**(3/2)
                >>> a*a*a
                x*sqrt(y)*sqrt(x*sqrt(y))
                >>> _.subs(a.base, z).subs(z, a.base)
                (x*sqrt(y))**(3/2)

              -  If more than two terms are being multiplied then all the
                 previous terms will be re-processed for each new argument.
                 So if each of ``a``, ``b`` and ``c`` were :class:`Mul`
                 expression, then ``a*b*c`` (or building up the product
                 with ``*=``) will process all the arguments of ``a`` and
                 ``b`` twice: once when ``a*b`` is computed and again when
                 ``c`` is multiplied.

                 Using ``Mul(a, b, c)`` will process all arguments once.

            * The results of Mul are cached according to arguments, so flatten
              will only be called once for ``Mul(a, b, c)``. If you can
              structure a calculation so the arguments are most likely to be
              repeats then this can save time in computing the answer. For
              example, say you had a Mul, M, that you wished to divide by ``d[i]``
              and multiply by ``n[i]`` and you suspect there are many repeats
              in ``n``. It would be better to compute ``M*n[i]/d[i]`` rather
              than ``M/d[i]*n[i]`` since every time n[i] is a repeat, the
              product, ``M*n[i]`` will be returned without flattening -- the
              cached value will be returned. If you divide by the ``d[i]``
              first (and those are more unique than the ``n[i]``) then that will
              create a new Mul, ``M/d[i]`` the args of which will be traversed
              again when it is multiplied by ``n[i]``.

              {c.f. https://github.com/sympy/sympy/issues/5706}

              This consideration is moot if the cache is turned off.

            NB
            --
              The validity of the above notes depends on the implementation
              details of Mul and flatten which may change at any time. Therefore,
              you should only consider them when your code is highly performance
              sensitive.

              Removal of 1 from the sequence is already handled by AssocOp.__new__.
        """

        from sympy.calculus.accumulationbounds import AccumBounds
        from sympy.matrices.expressions import MatrixExpr
        rv = None
        if len(seq) == 2:
            a, b = seq
            if b.is_Rational:
                a, b = b, a
                seq = [a, b]
            assert a is not S.One
            if a.is_Rational and not a.is_zero:
                r, b = b.as_coeff_Mul()
                if b.is_Add:
                    if r is not S.One:  # 2-arg hack
                        # leave the Mul as a Mul?
                        ar = a*r
                        if ar is S.One:
                            arb = b
                        else:
                            arb = cls(a*r, b, evaluate=False)
                        rv = [arb], [], None
                    elif global_parameters.distribute and b.is_commutative:
                        newb = Add(*[_keep_coeff(a, bi) for bi in b.args])
                        rv = [newb], [], None
            if rv:
                return rv

        # apply associativity, separate commutative part of seq
        c_part = []         # out: commutative factors
        nc_part = []        # out: non-commutative factors

        nc_seq = []

        coeff = S.One       # standalone term
                            # e.g. 3 * ...

        c_powers = []       # (base,exp)      n
                            # e.g. (x,n) for x

        num_exp = []        # (num-base, exp)           y
                            # e.g.  (3, y)  for  ... * 3  * ...

        neg1e = S.Zero      # exponent on -1 extracted from Number-based Pow and I

        pnum_rat = {}       # (num-base, Rat-exp)          1/2
                            # e.g.  (3, 1/2)  for  ... * 3     * ...

        order_symbols = None

        # --- PART 1 ---
        #
        # "collect powers and coeff":
        #
        # o coeff
        # o c_powers
        # o num_exp
        # o neg1e
        # o pnum_rat
        #
        # NOTE: this is optimized for all-objects-are-commutative case
        for o in seq:
            # O(x)
            if o.is_Order:
                o, order_symbols = o.as_expr_variables(order_symbols)

            # Mul([...])
            if o.is_Mul:
                if o.is_commutative:
                    seq.extend(o.args)    # XXX zerocopy?

                else:
                    # NCMul can have commutative parts as well
                    for q in o.args:
                        if q.is_commutative:
                            seq.append(q)
                        else:
                            nc_seq.append(q)

                    # append non-commutative marker, so we don't forget to
                    # process scheduled non-commutative objects
                    seq.append(NC_Marker)

                continue

            # 3
            elif o.is_Number:
                if o is S.NaN or coeff is S.ComplexInfinity and o.is_zero:
                    # we know for sure the result will be nan
                    return [S.NaN], [], None
                elif coeff.is_Number or isinstance(coeff, AccumBounds):  # it could be zoo
                    coeff *= o
                    if coeff is S.NaN:
                        # we know for sure the result will be nan
                        return [S.NaN], [], None
                continue

            elif isinstance(o, AccumBounds):
                coeff = o.__mul__(coeff)
                continue

            elif o is S.ComplexInfinity:
                if not coeff:
                    # 0 * zoo = NaN
                    return [S.NaN], [], None
                coeff = S.ComplexInfinity
                continue

            elif not coeff and isinstance(o, Add) and any(
                    _ in (S.NegativeInfinity, S.ComplexInfinity, S.Infinity)
                    for __ in o.args for _ in Mul.make_args(__)):
                # e.g 0 * (x + oo) = NaN but not
                # 0 * (1 + Integral(x, (x, 0, oo))) which is
                # treated like 0 * x -> 0
                return [S.NaN], [], None

            elif o is S.ImaginaryUnit:
                neg1e += S.Half
                continue

            elif o.is_commutative:
                #      e
                # o = b
                b, e = o.as_base_exp()

                #  y
                # 3
                if o.is_Pow:
                    if b.is_Number:

                        # get all the factors with numeric base so they can be
                        # combined below, but don't combine negatives unless
                        # the exponent is an integer
                        if e.is_Rational:
                            if e.is_Integer:
                                coeff *= Pow(b, e)  # it is an unevaluated power
                                continue
                            elif e.is_negative:    # also a sign of an unevaluated power
                                seq.append(Pow(b, e))
                                continue
                            elif b.is_negative:
                                neg1e += e
                                b = -b
                            if b is not S.One:
                                pnum_rat.setdefault(b, []).append(e)
                            continue
                        elif b.is_positive or e.is_integer:
                            num_exp.append((b, e))
                            continue

                c_powers.append((b, e))

            # NON-COMMUTATIVE
            # TODO: Make non-commutative exponents not combine automatically
            else:
                if o is not NC_Marker:
                    nc_seq.append(o)

                # process nc_seq (if any)
                while nc_seq:
                    o = nc_seq.pop(0)
                    if not nc_part:
                        nc_part.append(o)
                        continue

                    #                             b    c       b+c
                    # try to combine last terms: a  * a   ->  a
                    o1 = nc_part.pop()
                    b1, e1 = o1.as_base_exp()
                    b2, e2 = o.as_base_exp()
                    new_exp = e1 + e2
                    # Only allow powers to combine if the new exponent is
                    # not an Add. This allow things like a**2*b**3 == a**5
                    # if a.is_commutative == False, but prohibits
                    # a**x*a**y and x**a*x**b from combining (x,y commute).
                    if b1 == b2 and (not new_exp.is_Add):
                        o12 = b1 ** new_exp

                        # now o12 could be a commutative object
                        if o12.is_commutative:
                            seq.append(o12)
                            continue
                        else:
                            nc_seq.insert(0, o12)

                    else:
                        nc_part.extend([o1, o])

        # We do want a combined exponent if it would not be an Add, such as
        #  y    2y     3y
        # x  * x   -> x
        # We determine if two exponents have the same term by using
        # as_coeff_Mul.
        #
        # Unfortunately, this isn't smart enough to consider combining into
        # exponents that might already be adds, so things like:
        #  z - y    y
        # x      * x  will be left alone.  This is because checking every possible
        # combination can slow things down.

        # gather exponents of common bases...
        def _gather(c_powers):
            common_b = {}  # b:e
            for b, e in c_powers:
                co = e.as_coeff_Mul()
                common_b.setdefault(b, {}).setdefault(
                    co[1], []).append(co[0])
            for b, d in common_b.items():
                for di, li in d.items():
                    d[di] = Add(*li)
            new_c_powers = []
            for b, e in common_b.items():
                new_c_powers.extend([(b, c*t) for t, c in e.items()])
            return new_c_powers

        # in c_powers
        c_powers = _gather(c_powers)

        # and in num_exp
        num_exp = _gather(num_exp)

        # --- PART 2 ---
        #
        # o process collected powers  (x**0 -> 1; x**1 -> x; otherwise Pow)
        # o combine collected powers  (2**x * 3**x -> 6**x)
        #   with numeric base

        # ................................
        # now we have:
        # - coeff:
        # - c_powers:    (b, e)
        # - num_exp:     (2, e)
        # - pnum_rat:    {(1/3, [1/3, 2/3, 1/4])}

        #  0             1
        # x  -> 1       x  -> x

        # this should only need to run twice; if it fails because
        # it needs to be run more times, perhaps this should be
        # changed to a "while True" loop -- the only reason it
        # isn't such now is to allow a less-than-perfect result to
        # be obtained rather than raising an error or entering an
        # infinite loop
        for i in range(2):
            new_c_powers = []
            changed = False
            for b, e in c_powers:
                if e.is_zero:
                    # canceling out infinities yields NaN
                    if (b.is_Add or b.is_Mul) and any(infty in b.args
                        for infty in (S.ComplexInfinity, S.Infinity,
                                      S.NegativeInfinity)):
                        return [S.NaN], [], None
                    continue
                if e is S.One:
                    if b.is_Number:
                        coeff *= b
                        continue
                    p = b
                if e is not S.One:
                    p = Pow(b, e)
                    # check to make sure that the base doesn't change
                    # after exponentiation; to allow for unevaluated
                    # Pow, we only do so if b is not already a Pow
                    if p.is_Pow and not b.is_Pow:
                        bi = b
                        b, e = p.as_base_exp()
                        if b != bi:
                            changed = True
                c_part.append(p)
                new_c_powers.append((b, e))
            # there might have been a change, but unless the base
            # matches some other base, there is nothing to do
            if changed and len({
                    b for b, e in new_c_powers}) != len(new_c_powers):
                # start over again
                c_part = []
                c_powers = _gather(new_c_powers)
            else:
                break

        #  x    x     x
        # 2  * 3  -> 6
        inv_exp_dict = {}   # exp:Mul(num-bases)     x    x
                            # e.g.  x:6  for  ... * 2  * 3  * ...
        for b, e in num_exp:
            inv_exp_dict.setdefault(e, []).append(b)
        for e, b in inv_exp_dict.items():
            inv_exp_dict[e] = cls(*b)
        c_part.extend([Pow(b, e) for e, b in inv_exp_dict.items() if e])

        # b, e -> e' = sum(e), b
        # {(1/5, [1/3]), (1/2, [1/12, 1/4]} -> {(1/3, [1/5, 1/2])}
        comb_e = {}
        for b, e in pnum_rat.items():
            comb_e.setdefault(Add(*e), []).append(b)
        del pnum_rat
        # process them, reducing exponents to values less than 1
        # and updating coeff if necessary else adding them to
        # num_rat for further processing
        num_rat = []
        for e, b in comb_e.items():
            b = cls(*b)
            if e.q == 1:
                coeff *= Pow(b, e)
                continue
            if e.p > e.q:
                e_i, ep = divmod(e.p, e.q)
                coeff *= Pow(b, e_i)
                e = Rational(ep, e.q)
            num_rat.append((b, e))
        del comb_e

        # extract gcd of bases in num_rat
        # 2**(1/3)*6**(1/4) -> 2**(1/3+1/4)*3**(1/4)
        pnew = defaultdict(list)
        i = 0  # steps through num_rat which may grow
        while i < len(num_rat):
            bi, ei = num_rat[i]
            if bi == 1:
                i += 1
                continue
            grow = []
            for j in range(i + 1, len(num_rat)):
                bj, ej = num_rat[j]
                g = bi.gcd(bj)
                if g is not S.One:
                    # 4**r1*6**r2 -> 2**(r1+r2)  *  2**r1 *  3**r2
                    # this might have a gcd with something else
                    e = ei + ej
                    if e.q == 1:
                        coeff *= Pow(g, e)
                    else:
                        if e.p > e.q:
                            e_i, ep = divmod(e.p, e.q)  # change e in place
                            coeff *= Pow(g, e_i)
                            e = Rational(ep, e.q)
                        grow.append((g, e))
                    # update the jth item
                    num_rat[j] = (bj/g, ej)
                    # update bi that we are checking with
                    bi = bi/g
                    if bi is S.One:
                        break
            if bi is not S.One:
                obj = Pow(bi, ei)
                if obj.is_Number:
                    coeff *= obj
                else:
                    # changes like sqrt(12) -> 2*sqrt(3)
                    for obj in Mul.make_args(obj):
                        if obj.is_Number:
                            coeff *= obj
                        else:
                            assert obj.is_Pow
                            bi, ei = obj.args
                            pnew[ei].append(bi)

            num_rat.extend(grow)
            i += 1

        # combine bases of the new powers
        for e, b in pnew.items():
            pnew[e] = cls(*b)

        # handle -1 and I
        if neg1e:
            # treat I as (-1)**(1/2) and compute -1's total exponent
            p, q =  neg1e.as_numer_denom()
            # if the integer part is odd, extract -1
            n, p = divmod(p, q)
            if n % 2:
                coeff = -coeff
            # if it's a multiple of 1/2 extract I
            if q == 2:
                c_part.append(S.ImaginaryUnit)
            elif p:
                # see if there is any positive base this power of
                # -1 can join
                neg1e = Rational(p, q)
                for e, b in pnew.items():
                    if e == neg1e and b.is_positive:
                        pnew[e] = -b
                        break
                else:
                    # keep it separate; we've already evaluated it as
                    # much as possible so evaluate=False
                    c_part.append(Pow(S.NegativeOne, neg1e, evaluate=False))

        # add all the pnew powers
        c_part.extend([Pow(b, e) for e, b in pnew.items()])

        # oo, -oo
        if coeff in (S.Infinity, S.NegativeInfinity):
            def _handle_for_oo(c_part, coeff_sign):
                new_c_part = []
                for t in c_part:
                    if t.is_extended_positive:
                        continue
                    if t.is_extended_negative:
                        coeff_sign *= -1
                        continue
                    new_c_part.append(t)
                return new_c_part, coeff_sign
            c_part, coeff_sign = _handle_for_oo(c_part, 1)
            nc_part, coeff_sign = _handle_for_oo(nc_part, coeff_sign)
            coeff *= coeff_sign

        # zoo
        if coeff is S.ComplexInfinity:
            # zoo might be
            #   infinite_real + bounded_im
            #   bounded_real + infinite_im
            #   infinite_real + infinite_im
            # and non-zero real or imaginary will not change that status.
            c_part = [c for c in c_part if not (fuzzy_not(c.is_zero) and
                                                c.is_extended_real is not None)]
            nc_part = [c for c in nc_part if not (fuzzy_not(c.is_zero) and
                                                  c.is_extended_real is not None)]

        # 0
        elif coeff.is_zero:
            # we know for sure the result will be 0 except the multiplicand
            # is infinity or a matrix
            if any(isinstance(c, MatrixExpr) for c in nc_part):
                return [coeff], nc_part, order_symbols
            if any(c.is_finite == False for c in c_part):
                return [S.NaN], [], order_symbols
            return [coeff], [], order_symbols

        # check for straggling Numbers that were produced
        _new = []
        for i in c_part:
            if i.is_Number:
                coeff *= i
            else:
                _new.append(i)
        c_part = _new

        # order commutative part canonically
        _mulsort(c_part)

        # current code expects coeff to be always in slot-0
        if coeff is not S.One:
            c_part.insert(0, coeff)

        # we are done
        if (global_parameters.distribute and not nc_part and len(c_part) == 2 and
                c_part[0].is_Number and c_part[0].is_finite and c_part[1].is_Add):
            # 2*(1+a) -> 2 + 2 * a
            coeff = c_part[0]
            c_part = [Add(*[coeff*f for f in c_part[1].args])]

        return c_part, nc_part, order_symbols

    def _eval_power(self, expt):

        # don't break up NC terms: (A*B)**3 != A**3*B**3, it is A*B*A*B*A*B
        cargs, nc = self.args_cnc(split_1=False)

        if expt.is_Integer:
            return Mul(*[Pow(b, expt, evaluate=False) for b in cargs]) * \
                Pow(Mul._from_args(nc), expt, evaluate=False)
        if expt.is_Rational and expt.q == 2:
            if self.is_imaginary:
                a = self.as_real_imag()[1]
                if a.is_Rational:
                    n, d = abs(a/2).as_numer_denom()
                    n, t = integer_nthroot(n, 2)
                    if t:
                        d, t = integer_nthroot(d, 2)
                        if t:
                            from sympy.functions.elementary.complexes import sign
                            r = sympify(n)/d
                            return _unevaluated_Mul(r**expt.p, (1 + sign(a)*S.ImaginaryUnit)**expt.p)

        p = Pow(self, expt, evaluate=False)

        if expt.is_Rational or expt.is_Float:
            return p._eval_expand_power_base()

        return p

    @classmethod
    def class_key(cls):
        return 3, 0, cls.__name__

    def _eval_evalf(self, prec):
        c, m = self.as_coeff_Mul()
        if c is S.NegativeOne:
            if m.is_Mul:
                rv = -AssocOp._eval_evalf(m, prec)
            else:
                mnew = m._eval_evalf(prec)
                if mnew is not None:
                    m = mnew
                rv = -m
        else:
            rv = AssocOp._eval_evalf(self, prec)
        if rv.is_number:
            return rv.expand()
        return rv

    @property
    def _mpc_(self):
        """
        Convert self to an mpmath mpc if possible
        """
        from .numbers import Float
        im_part, imag_unit = self.as_coeff_Mul()
        if imag_unit is not S.ImaginaryUnit:
            # ValueError may seem more reasonable but since it's a @property,
            # we need to use AttributeError to keep from confusing things like
            # hasattr.
            raise AttributeError("Cannot convert Mul to mpc. Must be of the form Number*I")

        return (Float(0)._mpf_, Float(im_part)._mpf_)

    @cacheit
    def as_two_terms(self):
        """Return head and tail of self.

        This is the most efficient way to get the head and tail of an
        expression.

        - if you want only the head, use self.args[0];
        - if you want to process the arguments of the tail then use
          self.as_coef_mul() which gives the head and a tuple containing
          the arguments of the tail when treated as a Mul.
        - if you want the coefficient when self is treated as an Add
          then use self.as_coeff_add()[0]

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> (3*x*y).as_two_terms()
        (3, x*y)
        """
        args = self.args

        if len(args) == 1:
            return S.One, self
        elif len(args) == 2:
            return args

        else:
            return args[0], self._new_rawargs(*args[1:])

    @cacheit
    def as_coeff_mul(self, *deps, rational=True, **kwargs):
        if deps:
            l1, l2 = sift(self.args, lambda x: x.has(*deps), binary=True)
            return self._new_rawargs(*l2), tuple(l1)
        args = self.args
        if args[0].is_Number:
            if not rational or args[0].is_Rational:
                return args[0], args[1:]
            elif args[0].is_extended_negative:
                return S.NegativeOne, (-args[0],) + args[1:]
        return S.One, args

    def as_coeff_Mul(self, rational=False):
        """
        Efficiently extract the coefficient of a product.
        """
        coeff, args = self.args[0], self.args[1:]

        if coeff.is_Number:
            if not rational or coeff.is_Rational:
                if len(args) == 1:
                    return coeff, args[0]
                else:
                    return coeff, self._new_rawargs(*args)
            elif coeff.is_extended_negative:
                return S.NegativeOne, self._new_rawargs(*((-coeff,) + args))
        return S.One, self

    def as_real_imag(self, deep=True, **hints):
        from sympy.functions.elementary.complexes import Abs, im, re
        other = []
        coeffr = []
        coeffi = []
        addterms = S.One
        for a in self.args:
            r, i = a.as_real_imag()
            if i.is_zero:
                coeffr.append(r)
            elif r.is_zero:
                coeffi.append(i*S.ImaginaryUnit)
            elif a.is_commutative:
                aconj = a.conjugate() if other else None
                # search for complex conjugate pairs:
                for i, x in enumerate(other):
                    if x == aconj:
                        coeffr.append(Abs(x)**2)
                        del other[i]
                        break
                else:
                    if a.is_Add:
                        addterms *= a
                    else:
                        other.append(a)
            else:
                other.append(a)
        m = self.func(*other)
        if hints.get('ignore') == m:
            return
        if len(coeffi) % 2:
            imco = im(coeffi.pop(0))
            # all other pairs make a real factor; they will be
            # put into reco below
        else:
            imco = S.Zero
        reco = self.func(*(coeffr + coeffi))
        r, i = (reco*re(m), reco*im(m))
        if addterms == 1:
            if m == 1:
                if imco.is_zero:
                    return (reco, S.Zero)
                else:
                    return (S.Zero, reco*imco)
            if imco is S.Zero:
                return (r, i)
            return (-imco*i, imco*r)
        from .function import expand_mul
        addre, addim = expand_mul(addterms, deep=False).as_real_imag()
        if imco is S.Zero:
            return (r*addre - i*addim, i*addre + r*addim)
        else:
            r, i = -imco*i, imco*r
            return (r*addre - i*addim, r*addim + i*addre)

    @staticmethod
    def _expandsums(sums):
        """
        Helper function for _eval_expand_mul.

        sums must be a list of instances of Basic.
        """

        L = len(sums)
        if L == 1:
            return sums[0].args
        terms = []
        left = Mul._expandsums(sums[:L//2])
        right = Mul._expandsums(sums[L//2:])

        terms = [Mul(a, b) for a in left for b in right]
        added = Add(*terms)
        return Add.make_args(added)  # it may have collapsed down to one term

    def _eval_expand_mul(self, **hints):
        from sympy.simplify.radsimp import fraction

        # Handle things like 1/(x*(x + 1)), which are automatically converted
        # to 1/x*1/(x + 1)
        expr = self
        # default matches fraction's default
        n, d = fraction(expr, hints.get('exact', False))
        if d.is_Mul:
            n, d = [i._eval_expand_mul(**hints) if i.is_Mul else i
                for i in (n, d)]
        expr = n/d
        if not expr.is_Mul:
            return expr

        plain, sums, rewrite = [], [], False
        for factor in expr.args:
            if factor.is_Add:
                sums.append(factor)
                rewrite = True
            else:
                if factor.is_commutative:
                    plain.append(factor)
                else:
                    sums.append(Basic(factor))  # Wrapper

        if not rewrite:
            return expr
        else:
            plain = self.func(*plain)
            if sums:
                deep = hints.get("deep", False)
                terms = self.func._expandsums(sums)
                args = []
                for term in terms:
                    t = self.func(plain, term)
                    if t.is_Mul and any(a.is_Add for a in t.args) and deep:
                        t = t._eval_expand_mul()
                    args.append(t)
                return Add(*args)
            else:
                return plain

    @cacheit
    def _eval_derivative(self, s):
        args = list(self.args)
        terms = []
        for i in range(len(args)):
            d = args[i].diff(s)
            if d:
                # Note: reduce is used in step of Mul as Mul is unable to
                # handle subtypes and operation priority:
                terms.append(reduce(lambda x, y: x*y, (args[:i] + [d] + args[i + 1:]), S.One))
        return Add.fromiter(terms)

    @cacheit
    def _eval_derivative_n_times(self, s, n):
        from .function import AppliedUndef
        from .symbol import Symbol, symbols, Dummy
        if not isinstance(s, (AppliedUndef, Symbol)):
            # other types of s may not be well behaved, e.g.
            # (cos(x)*sin(y)).diff([[x, y, z]])
            return super()._eval_derivative_n_times(s, n)
        from .numbers import Integer
        args = self.args
        m = len(args)
        if isinstance(n, (int, Integer)):
            # https://en.wikipedia.org/wiki/General_Leibniz_rule#More_than_two_factors
            terms = []
            from sympy.ntheory.multinomial import multinomial_coefficients_iterator
            for kvals, c in multinomial_coefficients_iterator(m, n):
                p = Mul(*[arg.diff((s, k)) for k, arg in zip(kvals, args)])
                terms.append(c * p)
            return Add(*terms)
        from sympy.concrete.summations import Sum
        from sympy.functions.combinatorial.factorials import factorial
        from sympy.functions.elementary.miscellaneous import Max
        kvals = symbols("k1:%i" % m, cls=Dummy)
        klast = n - sum(kvals)
        nfact = factorial(n)
        e, l = (# better to use the multinomial?
            nfact/prod(map(factorial, kvals))/factorial(klast)*\
            Mul(*[args[t].diff((s, kvals[t])) for t in range(m-1)])*\
            args[-1].diff((s, Max(0, klast))),
            [(k, 0, n) for k in kvals])
        return Sum(e, *l)

    def _eval_difference_delta(self, n, step):
        from sympy.series.limitseq import difference_delta as dd
        arg0 = self.args[0]
        rest = Mul(*self.args[1:])
        return (arg0.subs(n, n + step) * dd(rest, n, step) + dd(arg0, n, step) *
                rest)

    def _matches_simple(self, expr, repl_dict):
        # handle (w*3).matches('x*5') -> {w: x*5/3}
        coeff, terms = self.as_coeff_Mul()
        terms = Mul.make_args(terms)
        if len(terms) == 1:
            newexpr = self.__class__._combine_inverse(expr, coeff)
            return terms[0].matches(newexpr, repl_dict)
        return

    def matches(self, expr, repl_dict=None, old=False):
        expr = sympify(expr)
        if self.is_commutative and expr.is_commutative:
            return self._matches_commutative(expr, repl_dict, old)
        elif self.is_commutative is not expr.is_commutative:
            return None

        # Proceed only if both both expressions are non-commutative
        c1, nc1 = self.args_cnc()
        c2, nc2 = expr.args_cnc()
        c1, c2 = [c or [1] for c in [c1, c2]]

        # TODO: Should these be self.func?
        comm_mul_self = Mul(*c1)
        comm_mul_expr = Mul(*c2)

        repl_dict = comm_mul_self.matches(comm_mul_expr, repl_dict, old)

        # If the commutative arguments didn't match and aren't equal, then
        # then the expression as a whole doesn't match
        if not repl_dict and c1 != c2:
            return None

        # Now match the non-commutative arguments, expanding powers to
        # multiplications
        nc1 = Mul._matches_expand_pows(nc1)
        nc2 = Mul._matches_expand_pows(nc2)

        repl_dict = Mul._matches_noncomm(nc1, nc2, repl_dict)

        return repl_dict or None

    @staticmethod
    def _matches_expand_pows(arg_list):
        new_args = []
        for arg in arg_list:
            if arg.is_Pow and arg.exp > 0:
                new_args.extend([arg.base] * arg.exp)
            else:
                new_args.append(arg)
        return new_args

    @staticmethod
    def _matches_noncomm(nodes, targets, repl_dict=None):
        """Non-commutative multiplication matcher.

        `nodes` is a list of symbols within the matcher multiplication
        expression, while `targets` is a list of arguments in the
        multiplication expression being matched against.
        """
        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        # List of possible future states to be considered
        agenda = []
        # The current matching state, storing index in nodes and targets
        state = (0, 0)
        node_ind, target_ind = state
        # Mapping between wildcard indices and the index ranges they match
        wildcard_dict = {}

        while target_ind < len(targets) and node_ind < len(nodes):
            node = nodes[node_ind]

            if node.is_Wild:
                Mul._matches_add_wildcard(wildcard_dict, state)

            states_matches = Mul._matches_new_states(wildcard_dict, state,
                                                     nodes, targets)
            if states_matches:
                new_states, new_matches = states_matches
                agenda.extend(new_states)
                if new_matches:
                    for match in new_matches:
                        repl_dict[match] = new_matches[match]
            if not agenda:
                return None
            else:
                state = agenda.pop()
                node_ind, target_ind = state

        return repl_dict

    @staticmethod
    def _matches_add_wildcard(dictionary, state):
        node_ind, target_ind = state
        if node_ind in dictionary:
            begin, end = dictionary[node_ind]
            dictionary[node_ind] = (begin, target_ind)
        else:
            dictionary[node_ind] = (target_ind, target_ind)

    @staticmethod
    def _matches_new_states(dictionary, state, nodes, targets):
        node_ind, target_ind = state
        node = nodes[node_ind]
        target = targets[target_ind]

        # Don't advance at all if we've exhausted the targets but not the nodes
        if target_ind >= len(targets) - 1 and node_ind < len(nodes) - 1:
            return None

        if node.is_Wild:
            match_attempt = Mul._matches_match_wilds(dictionary, node_ind,
                                                     nodes, targets)
            if match_attempt:
                # If the same node has been matched before, don't return
                # anything if the current match is diverging from the previous
                # match
                other_node_inds = Mul._matches_get_other_nodes(dictionary,
                                                               nodes, node_ind)
                for ind in other_node_inds:
                    other_begin, other_end = dictionary[ind]
                    curr_begin, curr_end = dictionary[node_ind]

                    other_targets = targets[other_begin:other_end + 1]
                    current_targets = targets[curr_begin:curr_end + 1]

                    for curr, other in zip(current_targets, other_targets):
                        if curr != other:
                            return None

                # A wildcard node can match more than one target, so only the
                # target index is advanced
                new_state = [(node_ind, target_ind + 1)]
                # Only move on to the next node if there is one
                if node_ind < len(nodes) - 1:
                    new_state.append((node_ind + 1, target_ind + 1))
                return new_state, match_attempt
        else:
            # If we're not at a wildcard, then make sure we haven't exhausted
            # nodes but not targets, since in this case one node can only match
            # one target
            if node_ind >= len(nodes) - 1 and target_ind < len(targets) - 1:
                return None

            match_attempt = node.matches(target)

            if match_attempt:
                return [(node_ind + 1, target_ind + 1)], match_attempt
            elif node == target:
                return [(node_ind + 1, target_ind + 1)], None
            else:
                return None

    @staticmethod
    def _matches_match_wilds(dictionary, wildcard_ind, nodes, targets):
        """Determine matches of a wildcard with sub-expression in `target`."""
        wildcard = nodes[wildcard_ind]
        begin, end = dictionary[wildcard_ind]
        terms = targets[begin:end + 1]
        # TODO: Should this be self.func?
        mult = Mul(*terms) if len(terms) > 1 else terms[0]
        return wildcard.matches(mult)

    @staticmethod
    def _matches_get_other_nodes(dictionary, nodes, node_ind):
        """Find other wildcards that may have already been matched."""
        ind_node = nodes[node_ind]
        return [ind for ind in dictionary if nodes[ind] == ind_node]

    @staticmethod
    def _combine_inverse(lhs, rhs):
        """
        Returns lhs/rhs, but treats arguments like symbols, so things
        like oo/oo return 1 (instead of a nan) and ``I`` behaves like
        a symbol instead of sqrt(-1).
        """
        from sympy.simplify.simplify import signsimp
        from .symbol import Dummy
        if lhs == rhs:
            return S.One

        def check(l, r):
            if l.is_Float and r.is_comparable:
                # if both objects are added to 0 they will share the same "normalization"
                # and are more likely to compare the same. Since Add(foo, 0) will not allow
                # the 0 to pass, we use __add__ directly.
                return l.__add__(0) == r.evalf().__add__(0)
            return False
        if check(lhs, rhs) or check(rhs, lhs):
            return S.One
        if any(i.is_Pow or i.is_Mul for i in (lhs, rhs)):
            # gruntz and limit wants a literal I to not combine
            # with a power of -1
            d = Dummy('I')
            _i = {S.ImaginaryUnit: d}
            i_ = {d: S.ImaginaryUnit}
            a = lhs.xreplace(_i).as_powers_dict()
            b = rhs.xreplace(_i).as_powers_dict()
            blen = len(b)
            for bi in tuple(b.keys()):
                if bi in a:
                    a[bi] -= b.pop(bi)
                    if not a[bi]:
                        a.pop(bi)
            if len(b) != blen:
                lhs = Mul(*[k**v for k, v in a.items()]).xreplace(i_)
                rhs = Mul(*[k**v for k, v in b.items()]).xreplace(i_)
        rv = lhs/rhs
        srv = signsimp(rv)
        return srv if srv.is_Number else rv

    def as_powers_dict(self):
        d = defaultdict(int)
        for term in self.args:
            for b, e in term.as_powers_dict().items():
                d[b] += e
        return d

    def as_numer_denom(self):
        # don't use _from_args to rebuild the numerators and denominators
        # as the order is not guaranteed to be the same once they have
        # been separated from each other
        numers, denoms = list(zip(*[f.as_numer_denom() for f in self.args]))
        return self.func(*numers), self.func(*denoms)

    def as_base_exp(self):
        e1 = None
        bases = []
        nc = 0
        for m in self.args:
            b, e = m.as_base_exp()
            if not b.is_commutative:
                nc += 1
            if e1 is None:
                e1 = e
            elif e != e1 or nc > 1 or not e.is_Integer:
                return self, S.One
            bases.append(b)
        return self.func(*bases), e1

    def _eval_is_polynomial(self, syms):
        return all(term._eval_is_polynomial(syms) for term in self.args)

    def _eval_is_rational_function(self, syms):
        return all(term._eval_is_rational_function(syms) for term in self.args)

    def _eval_is_meromorphic(self, x, a):
        return _fuzzy_group((arg.is_meromorphic(x, a) for arg in self.args),
                            quick_exit=True)

    def _eval_is_algebraic_expr(self, syms):
        return all(term._eval_is_algebraic_expr(syms) for term in self.args)

    _eval_is_commutative = lambda self: _fuzzy_group(
        a.is_commutative for a in self.args)

    def _eval_is_complex(self):
        comp = _fuzzy_group(a.is_complex for a in self.args)
        if comp is False:
            if any(a.is_infinite for a in self.args):
                if any(a.is_zero is not False for a in self.args):
                    return None
                return False
        return comp

    def _eval_is_zero_infinite_helper(self):
        #
        # Helper used by _eval_is_zero and _eval_is_infinite.
        #
        # Three-valued logic is tricky so let us reason this carefully. It
        # would be nice to say that we just check is_zero/is_infinite in all
        # args but we need to be careful about the case that one arg is zero
        # and another is infinite like Mul(0, oo) or more importantly a case
        # where it is not known if the arguments are zero or infinite like
        # Mul(y, 1/x). If either y or x could be zero then there is a
        # *possibility* that we have Mul(0, oo) which should give None for both
        # is_zero and is_infinite.
        #
        # We keep track of whether we have seen a zero or infinity but we also
        # need to keep track of whether we have *possibly* seen one which
        # would be indicated by None.
        #
        # For each argument there is the possibility that is_zero might give
        # True, False or None and likewise that is_infinite might give True,
        # False or None, giving 9 combinations. The True cases for is_zero and
        # is_infinite are mutually exclusive though so there are 3 main cases:
        #
        # - is_zero = True
        # - is_infinite = True
        # - is_zero and is_infinite are both either False or None
        #
        # At the end seen_zero and seen_infinite can be any of 9 combinations
        # of True/False/None. Unless one is False though we cannot return
        # anything except None:
        #
        # - is_zero=True needs seen_zero=True and seen_infinite=False
        # - is_zero=False needs seen_zero=False
        # - is_infinite=True needs seen_infinite=True and seen_zero=False
        # - is_infinite=False needs seen_infinite=False
        # - anything else gives both is_zero=None and is_infinite=None
        #
        # The loop only sets the flags to True or None and never back to False.
        # Hence as soon as neither flag is False we exit early returning None.
        # In particular as soon as we encounter a single arg that has
        # is_zero=is_infinite=None we exit. This is a common case since it is
        # the default assumptions for a Symbol and also the case for most
        # expressions containing such a symbol. The early exit gives a big
        # speedup for something like Mul(*symbols('x:1000')).is_zero.
        #
        seen_zero = seen_infinite = False

        for a in self.args:
            if a.is_zero:
                if seen_infinite is not False:
                    return None, None
                seen_zero = True
            elif a.is_infinite:
                if seen_zero is not False:
                    return None, None
                seen_infinite = True
            else:
                if seen_zero is False and a.is_zero is None:
                    if seen_infinite is not False:
                        return None, None
                    seen_zero = None
                if seen_infinite is False and a.is_infinite is None:
                    if seen_zero is not False:
                        return None, None
                    seen_infinite = None

        return seen_zero, seen_infinite

    def _eval_is_zero(self):
        # True iff any arg is zero and no arg is infinite but need to handle
        # three valued logic carefully.
        seen_zero, seen_infinite = self._eval_is_zero_infinite_helper()

        if seen_zero is False:
            return False
        elif seen_zero is True and seen_infinite is False:
            return True
        else:
            return None

    def _eval_is_infinite(self):
        # True iff any arg is infinite and no arg is zero but need to handle
        # three valued logic carefully.
        seen_zero, seen_infinite = self._eval_is_zero_infinite_helper()

        if seen_infinite is True and seen_zero is False:
            return True
        elif seen_infinite is False:
            return False
        else:
            return None

    # We do not need to implement _eval_is_finite because the assumptions
    # system can infer it from finite = not infinite.

    def _eval_is_rational(self):
        r = _fuzzy_group((a.is_rational for a in self.args), quick_exit=True)
        if r:
            return r
        elif r is False:
            # All args except one are rational
            if all(a.is_zero is False for a in self.args):
                return False

    def _eval_is_algebraic(self):
        r = _fuzzy_group((a.is_algebraic for a in self.args), quick_exit=True)
        if r:
            return r
        elif r is False:
            # All args except one are algebraic
            if all(a.is_zero is False for a in self.args):
                return False

    # without involving odd/even checks this code would suffice:
    #_eval_is_integer = lambda self: _fuzzy_group(
    #    (a.is_integer for a in self.args), quick_exit=True)
    def _eval_is_integer(self):
        is_rational = self._eval_is_rational()
        if is_rational is False:
            return False

        numerators = []
        denominators = []
        unknown = False
        for a in self.args:
            hit = False
            if a.is_integer:
                if abs(a) is not S.One:
                    numerators.append(a)
            elif a.is_Rational:
                n, d = a.as_numer_denom()
                if abs(n) is not S.One:
                    numerators.append(n)
                if d is not S.One:
                    denominators.append(d)
            elif a.is_Pow:
                b, e = a.as_base_exp()
                if not b.is_integer or not e.is_integer:
                    hit = unknown = True
                if e.is_negative:
                    denominators.append(2 if a is S.Half else
                        Pow(a, S.NegativeOne))
                elif not hit:
                    # int b and pos int e: a = b**e is integer
                    assert not e.is_positive
                    # for rational self and e equal to zero: a = b**e is 1
                    assert not e.is_zero
                    return # sign of e unknown -> self.is_integer unknown
                else:
                    # x**2, 2**x, or x**y with x and y int-unknown -> unknown
                    return
            else:
                return

        if not denominators and not unknown:
            return True

        allodd = lambda x: all(i.is_odd for i in x)
        alleven = lambda x: all(i.is_even for i in x)
        anyeven = lambda x: any(i.is_even for i in x)

        from .relational import is_gt
        if not numerators and denominators and all(
                is_gt(_, S.One) for _ in denominators):
            return False
        elif unknown:
            return
        elif allodd(numerators) and anyeven(denominators):
            return False
        elif anyeven(numerators) and denominators == [2]:
            return True
        elif alleven(numerators) and allodd(denominators
                ) and (Mul(*denominators, evaluate=False) - 1
                ).is_positive:
            return False
        if len(denominators) == 1:
            d = denominators[0]
            if d.is_Integer and d.is_even:
                # if minimal power of 2 in num vs den is not
                # negative then we have an integer
                if (Add(*[i.as_base_exp()[1] for i in
                        numerators if i.is_even]) - trailing(d.p)
                        ).is_nonnegative:
                    return True
        if len(numerators) == 1:
            n = numerators[0]
            if n.is_Integer and n.is_even:
                # if minimal power of 2 in den vs num is positive
                # then we have have a non-integer
                if (Add(*[i.as_base_exp()[1] for i in
                        denominators if i.is_even]) - trailing(n.p)
                        ).is_positive:
                    return False

    def _eval_is_polar(self):
        has_polar = any(arg.is_polar for arg in self.args)
        return has_polar and \
            all(arg.is_polar or arg.is_positive for arg in self.args)

    def _eval_is_extended_real(self):
        return self._eval_real_imag(True)

    def _eval_real_imag(self, real):
        zero = False
        t_not_re_im = None

        for t in self.args:
            if (t.is_complex or t.is_infinite) is False and t.is_extended_real is False:
                return False
            elif t.is_imaginary:  # I
                real = not real
            elif t.is_extended_real:  # 2
                if not zero:
                    z = t.is_zero
                    if not z and zero is False:
                        zero = z
                    elif z:
                        if all(a.is_finite for a in self.args):
                            return True
                        return
            elif t.is_extended_real is False:
                # symbolic or literal like `2 + I` or symbolic imaginary
                if t_not_re_im:
                    return  # complex terms might cancel
                t_not_re_im = t
            elif t.is_imaginary is False:  # symbolic like `2` or `2 + I`
                if t_not_re_im:
