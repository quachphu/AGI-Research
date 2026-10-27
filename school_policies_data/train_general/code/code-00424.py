                return basic_from_dict(expansion_dict, *p)
            else:
                if n == 2:
                    return Add(*[f*g for f in base.args for g in base.args])
                else:
                    multi = (base**(n - 1))._eval_expand_multinomial()
                    if multi.is_Add:
                        return Add(*[f*g for f in base.args
                            for g in multi.args])
                    else:
                        # XXX can this ever happen if base was an Add?
                        return Add(*[f*multi for f in base.args])
        elif (exp.is_Rational and exp.p < 0 and base.is_Add and
                abs(exp.p) > exp.q):
            return 1 / self.func(base, -exp)._eval_expand_multinomial()
        elif exp.is_Add and base.is_Number and (hints.get('force', False) or
                base.is_zero is False or exp._all_nonneg_or_nonppos()):
            #  a + b      a  b
            #  n      --> n  n, where n, a, b are Numbers
            # XXX should be in expand_power_exp?
            coeff, tail = [], []
            for term in exp.args:
                if term.is_Number:
                    coeff.append(self.func(base, term))
                else:
                    tail.append(term)
            return Mul(*(coeff + [self.func(base, Add._from_args(tail))]))
        else:
            return result

    def as_real_imag(self, deep=True, **hints):
        if self.exp.is_Integer:
            from sympy.polys.polytools import poly

            exp = self.exp
            re_e, im_e = self.base.as_real_imag(deep=deep)
            if not im_e:
                return self, S.Zero
            a, b = symbols('a b', cls=Dummy)
            if exp >= 0:
                if re_e.is_Number and im_e.is_Number:
                    # We can be more efficient in this case
                    expr = expand_multinomial(self.base**exp)
                    if expr != self:
                        return expr.as_real_imag()

                expr = poly(
                    (a + b)**exp)  # a = re, b = im; expr = (a + b*I)**exp
            else:
                mag = re_e**2 + im_e**2
                re_e, im_e = re_e/mag, -im_e/mag
                if re_e.is_Number and im_e.is_Number:
                    # We can be more efficient in this case
                    expr = expand_multinomial((re_e + im_e*S.ImaginaryUnit)**-exp)
                    if expr != self:
                        return expr.as_real_imag()

                expr = poly((a + b)**-exp)

            # Terms with even b powers will be real
            r = [i for i in expr.terms() if not i[0][1] % 2]
            re_part = Add(*[cc*a**aa*b**bb for (aa, bb), cc in r])
            # Terms with odd b powers will be imaginary
            r = [i for i in expr.terms() if i[0][1] % 4 == 1]
            im_part1 = Add(*[cc*a**aa*b**bb for (aa, bb), cc in r])
            r = [i for i in expr.terms() if i[0][1] % 4 == 3]
            im_part3 = Add(*[cc*a**aa*b**bb for (aa, bb), cc in r])

            return (re_part.subs({a: re_e, b: S.ImaginaryUnit*im_e}),
            im_part1.subs({a: re_e, b: im_e}) + im_part3.subs({a: re_e, b: -im_e}))

        from sympy.functions.elementary.trigonometric import atan2, cos, sin

        if self.exp.is_Rational:
            re_e, im_e = self.base.as_real_imag(deep=deep)

            if im_e.is_zero and self.exp is S.Half:
                if re_e.is_extended_nonnegative:
                    return self, S.Zero
                if re_e.is_extended_nonpositive:
                    return S.Zero, (-self.base)**self.exp

            # XXX: This is not totally correct since for x**(p/q) with
            #      x being imaginary there are actually q roots, but
            #      only a single one is returned from here.
            r = self.func(self.func(re_e, 2) + self.func(im_e, 2), S.Half)

            t = atan2(im_e, re_e)

            rp, tp = self.func(r, self.exp), t*self.exp

            return rp*cos(tp), rp*sin(tp)
        elif self.base is S.Exp1:
            from sympy.functions.elementary.exponential import exp
            re_e, im_e = self.exp.as_real_imag()
            if deep:
                re_e = re_e.expand(deep, **hints)
                im_e = im_e.expand(deep, **hints)
            c, s = cos(im_e), sin(im_e)
            return exp(re_e)*c, exp(re_e)*s
        else:
            from sympy.functions.elementary.complexes import im, re
            if deep:
                hints['complex'] = False

                expanded = self.expand(deep, **hints)
                if hints.get('ignore') == expanded:
                    return None
                else:
                    return (re(expanded), im(expanded))
            else:
                return re(self), im(self)

    def _eval_derivative(self, s):
        from sympy.functions.elementary.exponential import log
        dbase = self.base.diff(s)
        dexp = self.exp.diff(s)
        return self * (dexp * log(self.base) + dbase * self.exp/self.base)

    def _eval_evalf(self, prec):
        base, exp = self.as_base_exp()
        if base == S.Exp1:
            # Use mpmath function associated to class "exp":
            from sympy.functions.elementary.exponential import exp as exp_function
            return exp_function(self.exp, evaluate=False)._eval_evalf(prec)
        base = base._evalf(prec)
        if not exp.is_Integer:
            exp = exp._evalf(prec)
        if exp.is_negative and base.is_number and base.is_extended_real is False:
            base = base.conjugate() / (base * base.conjugate())._evalf(prec)
            exp = -exp
            return self.func(base, exp).expand()
        return self.func(base, exp)

    def _eval_is_polynomial(self, syms):
        if self.exp.has(*syms):
            return False

        if self.base.has(*syms):
            return bool(self.base._eval_is_polynomial(syms) and
                self.exp.is_Integer and (self.exp >= 0))
        else:
            return True

    def _eval_is_rational(self):
        # The evaluation of self.func below can be very expensive in the case
        # of integer**integer if the exponent is large.  We should try to exit
        # before that if possible:
        if (self.exp.is_integer and self.base.is_rational
                and fuzzy_not(fuzzy_and([self.exp.is_negative, self.base.is_zero]))):
            return True
        p = self.func(*self.as_base_exp())  # in case it's unevaluated
        if not p.is_Pow:
            return p.is_rational
        b, e = p.as_base_exp()
        if e.is_Rational and b.is_Rational:
            # we didn't check that e is not an Integer
            # because Rational**Integer autosimplifies
            return False
        if e.is_integer:
            if b.is_rational:
                if fuzzy_not(b.is_zero) or e.is_nonnegative:
                    return True
                if b == e:  # always rational, even for 0**0
                    return True
            elif b.is_irrational:
                return e.is_zero
        if b is S.Exp1:
            if e.is_rational and e.is_nonzero:
                return False

    def _eval_is_algebraic(self):
        def _is_one(expr):
            try:
                return (expr - 1).is_zero
            except ValueError:
                # when the operation is not allowed
                return False

        if self.base.is_zero or _is_one(self.base):
            return True
        elif self.base is S.Exp1:
            s = self.func(*self.args)
            if s.func == self.func:
                if self.exp.is_nonzero:
                    if self.exp.is_algebraic:
                        return False
                    elif (self.exp/S.Pi).is_rational:
                        return False
                    elif (self.exp/(S.ImaginaryUnit*S.Pi)).is_rational:
                        return True
            else:
                return s.is_algebraic
        elif self.exp.is_rational:
            if self.base.is_algebraic is False:
                return self.exp.is_zero
            if self.base.is_zero is False:
                if self.exp.is_nonzero:
                    return self.base.is_algebraic
                elif self.base.is_algebraic:
                    return True
            if self.exp.is_positive:
                return self.base.is_algebraic
        elif self.base.is_algebraic and self.exp.is_algebraic:
            if ((fuzzy_not(self.base.is_zero)
                and fuzzy_not(_is_one(self.base)))
                or self.base.is_integer is False
                or self.base.is_irrational):
                return self.exp.is_rational

    def _eval_is_rational_function(self, syms):
        if self.exp.has(*syms):
            return False

        if self.base.has(*syms):
            return self.base._eval_is_rational_function(syms) and \
                self.exp.is_Integer
        else:
            return True

    def _eval_is_meromorphic(self, x, a):
        # f**g is meromorphic if g is an integer and f is meromorphic.
        # E**(log(f)*g) is meromorphic if log(f)*g is meromorphic
        # and finite.
        base_merom = self.base._eval_is_meromorphic(x, a)
        exp_integer = self.exp.is_Integer
        if exp_integer:
            return base_merom

        exp_merom = self.exp._eval_is_meromorphic(x, a)
        if base_merom is False:
            # f**g = E**(log(f)*g) may be meromorphic if the
            # singularities of log(f) and g cancel each other,
            # for example, if g = 1/log(f). Hence,
            return False if exp_merom else None
        elif base_merom is None:
            return None

        b = self.base.subs(x, a)
        # b is extended complex as base is meromorphic.
        # log(base) is finite and meromorphic when b != 0, zoo.
        b_zero = b.is_zero
        if b_zero:
            log_defined = False
        else:
            log_defined = fuzzy_and((b.is_finite, fuzzy_not(b_zero)))

        if log_defined is False: # zero or pole of base
            return exp_integer  # False or None
        elif log_defined is None:
            return None

        if not exp_merom:
            return exp_merom  # False or None

        return self.exp.subs(x, a).is_finite

    def _eval_is_algebraic_expr(self, syms):
        if self.exp.has(*syms):
            return False

        if self.base.has(*syms):
            return self.base._eval_is_algebraic_expr(syms) and \
                self.exp.is_Rational
        else:
            return True

    def _eval_rewrite_as_exp(self, base, expo, **kwargs):
        from sympy.functions.elementary.exponential import exp, log

        if base.is_zero or base.has(exp) or expo.has(exp):
            return base**expo

        evaluate = expo.has(Symbol)

        if base.has(Symbol):
            # delay evaluation if expo is non symbolic
            # (as exp(x*log(5)) automatically reduces to x**5)
            if global_parameters.exp_is_pow:
                return Pow(S.Exp1, log(base)*expo, evaluate=evaluate)
            else:
                return exp(log(base)*expo, evaluate=evaluate)

        else:
            from sympy.functions.elementary.complexes import arg, Abs
            return exp((log(Abs(base)) + S.ImaginaryUnit*arg(base))*expo)

    def as_numer_denom(self):
        if not self.is_commutative:
            return self, S.One
        base, exp = self.as_base_exp()
        n, d = base.as_numer_denom()
        # this should be the same as ExpBase.as_numer_denom wrt
        # exponent handling
        neg_exp = exp.is_negative
        if exp.is_Mul and not neg_exp and not exp.is_positive:
            neg_exp = exp.could_extract_minus_sign()
        int_exp = exp.is_integer
        # the denominator cannot be separated from the numerator if
        # its sign is unknown unless the exponent is an integer, e.g.
        # sqrt(a/b) != sqrt(a)/sqrt(b) when a=1 and b=-1. But if the
        # denominator is negative the numerator and denominator can
        # be negated and the denominator (now positive) separated.
        if not (d.is_extended_real or int_exp):
            n = base
            d = S.One
        dnonpos = d.is_nonpositive
        if dnonpos:
            n, d = -n, -d
        elif dnonpos is None and not int_exp:
            n = base
            d = S.One
        if neg_exp:
            n, d = d, n
            exp = -exp
        if exp.is_infinite:
            if n is S.One and d is not S.One:
                return n, self.func(d, exp)
            if n is not S.One and d is S.One:
                return self.func(n, exp), d
        return self.func(n, exp), self.func(d, exp)

    def matches(self, expr, repl_dict=None, old=False):
        expr = _sympify(expr)
        if repl_dict is None:
            repl_dict = {}

        # special case, pattern = 1 and expr.exp can match to 0
        if expr is S.One:
            d = self.exp.matches(S.Zero, repl_dict)
            if d is not None:
                return d

        # make sure the expression to be matched is an Expr
        if not isinstance(expr, Expr):
            return None

        b, e = expr.as_base_exp()

        # special case number
        sb, se = self.as_base_exp()
        if sb.is_Symbol and se.is_Integer and expr:
            if e.is_rational:
                return sb.matches(b**(e/se), repl_dict)
            return sb.matches(expr**(1/se), repl_dict)

        d = repl_dict.copy()
        d = self.base.matches(b, d)
        if d is None:
            return None

        d = self.exp.xreplace(d).matches(e, d)
        if d is None:
            return Expr.matches(self, expr, repl_dict)
        return d

    def _eval_nseries(self, x, n, logx, cdir=0):
        # NOTE! This function is an important part of the gruntz algorithm
        #       for computing limits. It has to return a generalized power
        #       series with coefficients in C(log, log(x)). In more detail:
        # It has to return an expression
        #     c_0*x**e_0 + c_1*x**e_1 + ... (finitely many terms)
        # where e_i are numbers (not necessarily integers) and c_i are
        # expressions involving only numbers, the log function, and log(x).
        # The series expansion of b**e is computed as follows:
        # 1) We express b as f*(1 + g) where f is the leading term of b.
        #    g has order O(x**d) where d is strictly positive.
        # 2) Then b**e = (f**e)*((1 + g)**e).
        #    (1 + g)**e is computed using binomial series.
        from sympy.functions.elementary.exponential import exp, log
        from sympy.series.limits import limit
        from sympy.series.order import Order
        from sympy.core.sympify import sympify
        if self.base is S.Exp1:
            e_series = self.exp.nseries(x, n=n, logx=logx)
            if e_series.is_Order:
                return 1 + e_series
            e0 = limit(e_series.removeO(), x, 0)
            if e0 is S.NegativeInfinity:
                return Order(x**n, x)
            if e0 is S.Infinity:
                return self
            t = e_series - e0
            exp_series = term = exp(e0)
            # series of exp(e0 + t) in t
            for i in range(1, n):
                term *= t/i
                term = term.nseries(x, n=n, logx=logx)
                exp_series += term
            exp_series += Order(t**n, x)
            from sympy.simplify.powsimp import powsimp
            return powsimp(exp_series, deep=True, combine='exp')
        from sympy.simplify.powsimp import powdenest
        from .numbers import _illegal
        self = powdenest(self, force=True).trigsimp()
        b, e = self.as_base_exp()

        if e.has(*_illegal):
            raise PoleError()

        if e.has(x):
            return exp(e*log(b))._eval_nseries(x, n=n, logx=logx, cdir=cdir)

        if logx is not None and b.has(log):
            from .symbol import Wild
            c, ex = symbols('c, ex', cls=Wild, exclude=[x])
            b = b.replace(log(c*x**ex), log(c) + ex*logx)
            self = b**e

        b = b.removeO()
        try:
            from sympy.functions.special.gamma_functions import polygamma
            if b.has(polygamma, S.EulerGamma) and logx is not None:
                raise ValueError()
            _, m = b.leadterm(x)
        except (ValueError, NotImplementedError, PoleError):
            b = b._eval_nseries(x, n=max(2, n), logx=logx, cdir=cdir).removeO()
            if b.has(S.NaN, S.ComplexInfinity):
                raise NotImplementedError()
            _, m = b.leadterm(x)

        if e.has(log):
            from sympy.simplify.simplify import logcombine
            e = logcombine(e).cancel()

        if not (m.is_zero or e.is_number and e.is_real):
            if self == self._eval_as_leading_term(x, logx=logx, cdir=cdir):
                res = exp(e*log(b))._eval_nseries(x, n=n, logx=logx, cdir=cdir)
                if res == exp(e*log(b)):
                    return self
                return res

        f = b.as_leading_term(x, logx=logx)
        g = (_mexpand(b) - f).cancel()
        g = g/f
        if not m.is_number:
            raise NotImplementedError()
        maxpow = n - m*e
        if maxpow.has(Symbol):
            maxpow = sympify(n)

        if maxpow.is_negative:
            return Order(x**(m*e), x)

        if g.is_zero:
            r = f**e
            if r != self:
                r += Order(x**n, x)
            return r

        def coeff_exp(term, x):
            coeff, exp = S.One, S.Zero
            for factor in Mul.make_args(term):
                if factor.has(x):
                    base, exp = factor.as_base_exp()
                    if base != x:
                        try:
                            return term.leadterm(x)
                        except ValueError:
                            return term, S.Zero
                else:
                    coeff *= factor
            return coeff, exp

        def mul(d1, d2):
            res = {}
            for e1, e2 in product(d1, d2):
                ex = e1 + e2
                if ex < maxpow:
                    res[ex] = res.get(ex, S.Zero) + d1[e1]*d2[e2]
            return res

        try:
            c, d = g.leadterm(x, logx=logx)
        except (ValueError, NotImplementedError):
            if limit(g/x**maxpow, x, 0) == 0:
                # g has higher order zero
                return f**e + e*f**e*g  # first term of binomial series
            else:
                raise NotImplementedError()
        if c.is_Float and d == S.Zero:
            # Convert floats like 0.5 to exact SymPy numbers like S.Half, to
            # prevent rounding errors which can induce wrong values of d leading
            # to a NotImplementedError being returned from the block below.
            g = g.replace(lambda x: x.is_Float, lambda x: Rational(x))
            _, d = g.leadterm(x, logx=logx)
        if not d.is_positive:
            g = g.simplify()
            if g.is_zero:
                return f**e
            _, d = g.leadterm(x, logx=logx)
            if not d.is_positive:
                g = ((b - f)/f).expand()
                _, d = g.leadterm(x, logx=logx)
                if not d.is_positive:
                    raise NotImplementedError()

        from sympy.functions.elementary.integers import ceiling
        gpoly = g._eval_nseries(x, n=ceiling(maxpow), logx=logx, cdir=cdir).removeO()
        gterms = {}

        for term in Add.make_args(gpoly):
            co1, e1 = coeff_exp(term, x)
            gterms[e1] = gterms.get(e1, S.Zero) + co1

        k = S.One
        terms = {S.Zero: S.One}
        tk = gterms

        from sympy.functions.combinatorial.factorials import factorial, ff

        while (k*d - maxpow).is_negative:
            coeff = ff(e, k)/factorial(k)
            for ex in tk:
                terms[ex] = terms.get(ex, S.Zero) + coeff*tk[ex]
            tk = mul(tk, gterms)
            k += S.One

        from sympy.functions.elementary.complexes import im

        if not e.is_integer and m.is_zero and f.is_negative:
            ndir = (b - f).dir(x, cdir)
            if im(ndir).is_negative:
                inco, inex = coeff_exp(f**e*(-1)**(-2*e), x)
            elif im(ndir).is_zero:
                inco, inex = coeff_exp(exp(e*log(b)).as_leading_term(x, logx=logx, cdir=cdir), x)
            else:
                inco, inex = coeff_exp(f**e, x)
        else:
            inco, inex = coeff_exp(f**e, x)
        res = S.Zero

        for e1 in terms:
            ex = e1 + inex
            res += terms[e1]*inco*x**(ex)

        if not (e.is_integer and e.is_positive and (e*d - n).is_nonpositive and
                res == _mexpand(self)):
            try:
                res += Order(x**n, x)
            except NotImplementedError:
                return exp(e*log(b))._eval_nseries(x, n=n, logx=logx, cdir=cdir)
        return res

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.functions.elementary.exponential import exp, log
        e = self.exp
        b = self.base
        if self.base is S.Exp1:
            arg = e.as_leading_term(x, logx=logx)
            arg0 = arg.subs(x, 0)
            if arg0 is S.NaN:
                arg0 = arg.limit(x, 0)
            if arg0.is_infinite is False:
                return S.Exp1**arg0
            raise PoleError("Cannot expand %s around 0" % (self))
        elif e.has(x):
            lt = exp(e * log(b))
            return lt.as_leading_term(x, logx=logx, cdir=cdir)
        else:
            from sympy.functions.elementary.complexes import im
            try:
                f = b.as_leading_term(x, logx=logx, cdir=cdir)
            except PoleError:
                return self
            if not e.is_integer and f.is_negative and not f.has(x):
                ndir = (b - f).dir(x, cdir)
                if im(ndir).is_negative:
                    # Normally, f**e would evaluate to exp(e*log(f)) but on branch cuts
                    # an other value is expected through the following computation
                    # exp(e*(log(f) - 2*pi*I)) == f**e*exp(-2*e*pi*I) == f**e*(-1)**(-2*e).
                    return self.func(f, e) * (-1)**(-2*e)
                elif im(ndir).is_zero:
                    log_leadterm = log(b)._eval_as_leading_term(x, logx=logx, cdir=cdir)
                    if log_leadterm.is_infinite is False:
                        return exp(e*log_leadterm)
            return self.func(f, e)

    @cacheit
    def _taylor_term(self, n, x, *previous_terms): # of (1 + x)**e
        from sympy.functions.combinatorial.factorials import binomial
        return binomial(self.exp, n) * self.func(x, n)

    def taylor_term(self, n, x, *previous_terms):
        if self.base is not S.Exp1:
            return super().taylor_term(n, x, *previous_terms)
        if n < 0:
            return S.Zero
        if n == 0:
            return S.One
        from .sympify import sympify
        x = sympify(x)
        if previous_terms:
            p = previous_terms[-1]
            if p is not None:
                return p * x / n
        from sympy.functions.combinatorial.factorials import factorial
        return x**n/factorial(n)

    def _eval_rewrite_as_sin(self, base, exp, **hints):
        if self.base is S.Exp1:
            from sympy.functions.elementary.trigonometric import sin
            return sin(S.ImaginaryUnit*self.exp + S.Pi/2) - S.ImaginaryUnit*sin(S.ImaginaryUnit*self.exp)

    def _eval_rewrite_as_cos(self, base, exp, **hints):
        if self.base is S.Exp1:
            from sympy.functions.elementary.trigonometric import cos
            return cos(S.ImaginaryUnit*self.exp) + S.ImaginaryUnit*cos(S.ImaginaryUnit*self.exp + S.Pi/2)

    def _eval_rewrite_as_tanh(self, base, exp, **hints):
        if self.base is S.Exp1:
            from sympy.functions.elementary.hyperbolic import tanh
            return (1 + tanh(self.exp/2))/(1 - tanh(self.exp/2))

    def _eval_rewrite_as_sqrt(self, base, exp, **kwargs):
        from sympy.functions.elementary.trigonometric import sin, cos
        if base is not S.Exp1:
            return None
        if exp.is_Mul:
            coeff = exp.coeff(S.Pi * S.ImaginaryUnit)
            if coeff and coeff.is_number:
                cosine, sine = cos(S.Pi*coeff), sin(S.Pi*coeff)
                if not isinstance(cosine, cos) and not isinstance (sine, sin):
                    return cosine + S.ImaginaryUnit*sine

    def as_content_primitive(self, radical=False, clear=True):
        """Return the tuple (R, self/R) where R is the positive Rational
        extracted from self.

        Examples
        ========

        >>> from sympy import sqrt
        >>> sqrt(4 + 4*sqrt(2)).as_content_primitive()
        (2, sqrt(1 + sqrt(2)))
        >>> sqrt(3 + 3*sqrt(2)).as_content_primitive()
        (1, sqrt(3)*sqrt(1 + sqrt(2)))

        >>> from sympy import expand_power_base, powsimp, Mul
        >>> from sympy.abc import x, y

        >>> ((2*x + 2)**2).as_content_primitive()
        (4, (x + 1)**2)
        >>> (4**((1 + y)/2)).as_content_primitive()
        (2, 4**(y/2))
        >>> (3**((1 + y)/2)).as_content_primitive()
        (1, 3**((y + 1)/2))
        >>> (3**((5 + y)/2)).as_content_primitive()
        (9, 3**((y + 1)/2))
        >>> eq = 3**(2 + 2*x)
        >>> powsimp(eq) == eq
        True
        >>> eq.as_content_primitive()
        (9, 3**(2*x))
        >>> powsimp(Mul(*_))
        3**(2*x + 2)

        >>> eq = (2 + 2*x)**y
        >>> s = expand_power_base(eq); s.is_Mul, s
        (False, (2*x + 2)**y)
        >>> eq.as_content_primitive()
        (1, (2*(x + 1))**y)
        >>> s = expand_power_base(_[1]); s.is_Mul, s
        (True, 2**y*(x + 1)**y)

        See docstring of Expr.as_content_primitive for more examples.
        """

        b, e = self.as_base_exp()
        b = _keep_coeff(*b.as_content_primitive(radical=radical, clear=clear))
        ce, pe = e.as_content_primitive(radical=radical, clear=clear)
        if b.is_Rational:
            #e
            #= ce*pe
            #= ce*(h + t)
            #= ce*h + ce*t
            #=> self
            #= b**(ce*h)*b**(ce*t)
            #= b**(cehp/cehq)*b**(ce*t)
            #= b**(iceh + r/cehq)*b**(ce*t)
            #= b**(iceh)*b**(r/cehq)*b**(ce*t)
            #= b**(iceh)*b**(ce*t + r/cehq)
            h, t = pe.as_coeff_Add()
            if h.is_Rational and b != S.Zero:
                ceh = ce*h
                c = self.func(b, ceh)
                r = S.Zero
                if not c.is_Rational:
                    iceh, r = divmod(ceh.p, ceh.q)
                    c = self.func(b, iceh)
                return c, self.func(b, _keep_coeff(ce, t + r/ce/ceh.q))
        e = _keep_coeff(ce, pe)
        # b**e = (h*t)**e = h**e*t**e = c*m*t**e
        if e.is_Rational and b.is_Mul:
            h, t = b.as_content_primitive(radical=radical, clear=clear)  # h is positive
            c, m = self.func(h, e).as_coeff_Mul()  # so c is positive
            m, me = m.as_base_exp()
            if m is S.One or me == e:  # probably always true
                # return the following, not return c, m*Pow(t, e)
                # which would change Pow into Mul; we let SymPy
                # decide what to do by using the unevaluated Mul, e.g
                # should it stay as sqrt(2 + 2*sqrt(5)) or become
                # sqrt(2)*sqrt(1 + sqrt(5))
                return c, self.func(_keep_coeff(m, t), e)
        return S.One, self.func(b, e)

    def is_constant(self, *wrt, **flags):
        expr = self
        if flags.get('simplify', True):
            expr = expr.simplify()
        b, e = expr.as_base_exp()
        bz = b.equals(0)
        if bz:  # recalculate with assumptions in case it's unevaluated
            new = b**e
            if new != expr:
                return new.is_constant()
        econ = e.is_constant(*wrt)
        bcon = b.is_constant(*wrt)
        if bcon:
            if econ:
                return True
            bz = b.equals(0)
            if bz is False:
                return False
        elif bcon is None:
            return None

        return e.equals(0)

    def _eval_difference_delta(self, n, step):
        b, e = self.args
        if e.has(n) and not b.has(n):
            new_e = e.subs(n, n + step)
            return (b**(new_e - e) - 1) * self

power = Dispatcher('power')
power.add((object, object), Pow)

from .add import Add
from .numbers import Integer, Rational
from .mul import Mul, _keep_coeff
from .symbol import Symbol, Dummy, symbols
<|endoftext|>"""Singleton mechanism"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .core import Registry
from .sympify import sympify


if TYPE_CHECKING:
    from sympy.core.numbers import (
        Zero as _Zero,
        One as _One,
        NegativeOne as _NegativeOne,
        Half as _Half,
        Infinity as _Infinity,
        NegativeInfinity as _NegativeInfinity,
        ComplexInfinity as _ComplexInfinity,
        NaN as _NaN,
    )


class SingletonRegistry(Registry):
    """
    The registry for the singleton classes (accessible as ``S``).

    Explanation
    ===========

    This class serves as two separate things.

    The first thing it is is the ``SingletonRegistry``. Several classes in
    SymPy appear so often that they are singletonized, that is, using some
    metaprogramming they are made so that they can only be instantiated once
    (see the :class:`sympy.core.singleton.Singleton` class for details). For
    instance, every time you create ``Integer(0)``, this will return the same
    instance, :class:`sympy.core.numbers.Zero`. All singleton instances are
    attributes of the ``S`` object, so ``Integer(0)`` can also be accessed as
    ``S.Zero``.

    Singletonization offers two advantages: it saves memory, and it allows
    fast comparison. It saves memory because no matter how many times the
    singletonized objects appear in expressions in memory, they all point to
    the same single instance in memory. The fast comparison comes from the
    fact that you can use ``is`` to compare exact instances in Python
    (usually, you need to use ``==`` to compare things). ``is`` compares
    objects by memory address, and is very fast.

    Examples
    ========

    >>> from sympy import S, Integer
    >>> a = Integer(0)
    >>> a is S.Zero
    True

    For the most part, the fact that certain objects are singletonized is an
    implementation detail that users should not need to worry about. In SymPy
    library code, ``is`` comparison is often used for performance purposes
    The primary advantage of ``S`` for end users is the convenient access to
    certain instances that are otherwise difficult to type, like ``S.Half``
    (instead of ``Rational(1, 2)``).

    When using ``is`` comparison, make sure the argument is sympified. For
    instance,

    >>> x = 0
    >>> x is S.Zero
    False

    This problem is not an issue when using ``==``, which is recommended for
    most use-cases:

    >>> 0 == S.Zero
    True

    The second thing ``S`` is is a shortcut for
    :func:`sympy.core.sympify.sympify`. :func:`sympy.core.sympify.sympify` is
    the function that converts Python objects such as ``int(1)`` into SymPy
    objects such as ``Integer(1)``. It also converts the string form of an
    expression into a SymPy expression, like ``sympify("x**2")`` ->
    ``Symbol("x")**2``. ``S(1)`` is the same thing as ``sympify(1)``
    (basically, ``S.__call__`` has been defined to call ``sympify``).

    This is for convenience, since ``S`` is a single letter. It's mostly
    useful for defining rational numbers. Consider an expression like ``x +
    1/2``. If you enter this directly in Python, it will evaluate the ``1/2``
    and give ``0.5``, because both arguments are ints (see also
    :ref:`tutorial-gotchas-final-notes`). However, in SymPy, you usually want
    the quotient of two integers to give an exact rational number. The way
    Python's evaluation works, at least one side of an operator needs to be a
    SymPy object for the SymPy evaluation to take over. You could write this
    as ``x + Rational(1, 2)``, but this is a lot more typing. A shorter
    version is ``x + S(1)/2``. Since ``S(1)`` returns ``Integer(1)``, the
    division will return a ``Rational`` type, since it will call
    ``Integer.__truediv__``, which knows how to return a ``Rational``.

    """
    __slots__ = ()

    Zero: _Zero
    One: _One
    NegativeOne: _NegativeOne
    Half: _Half
    Infinity: _Infinity
    NegativeInfinity: _NegativeInfinity
    ComplexInfinity: _ComplexInfinity
    NaN: _NaN

    # Also allow things like S(5)
    __call__ = staticmethod(sympify)

    def __init__(self):
        self._classes_to_install = {}
        # Dict of classes that have been registered, but that have not have been
        # installed as an attribute of this SingletonRegistry.
        # Installation automatically happens at the first attempt to access the
        # attribute.
        # The purpose of this is to allow registration during class
        # initialization during import, but not trigger object creation until
        # actual use (which should not happen until after all imports are
        # finished).

    def register(self, cls):
        # Make sure a duplicate class overwrites the old one
        if hasattr(self, cls.__name__):
            delattr(self, cls.__name__)
        self._classes_to_install[cls.__name__] = cls

    def __getattr__(self, name):
        """Python calls __getattr__ if no attribute of that name was installed
        yet.

        Explanation
        ===========

        This __getattr__ checks whether a class with the requested name was
        already registered but not installed; if no, raises an AttributeError.
        Otherwise, retrieves the class, calculates its singleton value, installs
        it as an attribute of the given name, and unregisters the class."""
        if name not in self._classes_to_install:
            raise AttributeError(
                "Attribute '%s' was not installed on SymPy registry %s" % (
                name, self))
        class_to_install = self._classes_to_install[name]
        value_to_install = class_to_install()
        self.__setattr__(name, value_to_install)
        del self._classes_to_install[name]
        return value_to_install

    def __repr__(self):
        return "S"

S = SingletonRegistry()


class Singleton(type):
    """
    Metaclass for singleton classes.

    Explanation
    ===========

    A singleton class has only one instance which is returned every time the
    class is instantiated. Additionally, this instance can be accessed through
    the global registry object ``S`` as ``S.<class_name>``.

    Examples
    ========

        >>> from sympy import S, Basic
        >>> from sympy.core.singleton import Singleton
        >>> class MySingleton(Basic, metaclass=Singleton):
        ...     pass
        >>> Basic() is Basic()
        False
        >>> MySingleton() is MySingleton()
        True
        >>> S.MySingleton is MySingleton()
        True

    Notes
    =====

    Instance creation is delayed until the first time the value is accessed.
    (SymPy versions before 1.0 would create the instance during class
    creation time, which would be prone to import cycles.)
    """
    def __init__(cls, *args, **kwargs):
        cls._instance = obj = Basic.__new__(cls)
        cls.__new__ = lambda cls: obj
        cls.__getnewargs__ = lambda obj: ()
        cls.__getstate__ = lambda obj: None
        S.register(cls)


# Delayed to avoid cyclic import
from .basic import Basic
<|endoftext|>"""Definitions of common exceptions for :mod:`sympy.core` module. """

from typing import Callable


class BaseCoreError(Exception):
    """Base class for core related exceptions. """


class NonCommutativeExpression(BaseCoreError):
    """Raised when expression didn't have commutative property. """


class LazyExceptionMessage:
    """Wrapper class that lets you specify an expensive to compute
    error message that is only evaluated if the error is rendered."""
    callback: Callable[[], str]

    def __init__(self, callback: Callable[[], str]):
        self.callback = callback

    def __str__(self):
        return self.callback()
<|endoftext|>"""Logic expressions handling

NOTE
----

at present this is mainly needed for facts.py, feel free however to improve
this stuff for general purpose.
"""

from __future__ import annotations
from typing import Optional

# Type of a fuzzy bool
FuzzyBool = Optional[bool]


def _torf(args):
    """Return True if all args are True, False if they
    are all False, else None.

    >>> from sympy.core.logic import _torf
    >>> _torf((True, True))
    True
    >>> _torf((False, False))
    False
    >>> _torf((True, False))
    """
    sawT = sawF = False
    for a in args:
        if a is True:
            if sawF:
                return
            sawT = True
        elif a is False:
            if sawT:
                return
            sawF = True
        else:
            return
    return sawT


def _fuzzy_group(args, quick_exit=False):
    """Return True if all args are True, None if there is any None else False
    unless ``quick_exit`` is True (then return None as soon as a second False
    is seen.

     ``_fuzzy_group`` is like ``fuzzy_and`` except that it is more
    conservative in returning a False, waiting to make sure that all
    arguments are True or False and returning None if any arguments are
    None. It also has the capability of permiting only a single False and
    returning None if more than one is seen. For example, the presence of a
    single transcendental amongst rationals would indicate that the group is
    no longer rational; but a second transcendental in the group would make the
    determination impossible.


    Examples
    ========

    >>> from sympy.core.logic import _fuzzy_group

    By default, multiple Falses mean the group is broken:

    >>> _fuzzy_group([False, False, True])
    False

    If multiple Falses mean the group status is unknown then set
    `quick_exit` to True so None can be returned when the 2nd False is seen:

    >>> _fuzzy_group([False, False, True], quick_exit=True)

    But if only a single False is seen then the group is known to
    be broken:

    >>> _fuzzy_group([False, True, True], quick_exit=True)
    False

    """
    saw_other = False
    for a in args:
        if a is True:
            continue
        if a is None:
            return
        if quick_exit and saw_other:
            return
        saw_other = True
    return not saw_other


def fuzzy_bool(x):
    """Return True, False or None according to x.

    Whereas bool(x) returns True or False, fuzzy_bool allows
    for the None value and non-false values (which become None), too.

    Examples
    ========

    >>> from sympy.core.logic import fuzzy_bool
    >>> from sympy.abc import x
    >>> fuzzy_bool(x), fuzzy_bool(None)
    (None, None)
    >>> bool(x), bool(None)
    (True, False)

    """
    if x is None:
        return None
    if x in (True, False):
        return bool(x)


def fuzzy_and(args):
    """Return True (all True), False (any False) or None.

    Examples
    ========

    >>> from sympy.core.logic import fuzzy_and
    >>> from sympy import Dummy

    If you had a list of objects to test the commutivity of
    and you want the fuzzy_and logic applied, passing an
    iterator will allow the commutativity to only be computed
    as many times as necessary. With this list, False can be
    returned after analyzing the first symbol:

    >>> syms = [Dummy(commutative=False), Dummy()]
    >>> fuzzy_and(s.is_commutative for s in syms)
    False

    That False would require less work than if a list of pre-computed
    items was sent:

    >>> fuzzy_and([s.is_commutative for s in syms])
    False
    """

    rv = True
    for ai in args:
        ai = fuzzy_bool(ai)
        if ai is False:
            return False
        if rv:  # this will stop updating if a None is ever trapped
            rv = ai
    return rv


def fuzzy_not(v):
    """
    Not in fuzzy logic

    Return None if `v` is None else `not v`.

    Examples
    ========

    >>> from sympy.core.logic import fuzzy_not
    >>> fuzzy_not(True)
    False
    >>> fuzzy_not(None)
    >>> fuzzy_not(False)
    True

    """
    if v is None:
        return v
    else:
        return not v


def fuzzy_or(args):
    """
    Or in fuzzy logic. Returns True (any True), False (all False), or None

    See the docstrings of fuzzy_and and fuzzy_not for more info.  fuzzy_or is
    related to the two by the standard De Morgan's law.

    >>> from sympy.core.logic import fuzzy_or
    >>> fuzzy_or([True, False])
    True
    >>> fuzzy_or([True, None])
    True
    >>> fuzzy_or([False, False])
    False
    >>> print(fuzzy_or([False, None]))
    None

    """
    rv = False
    for ai in args:
        ai = fuzzy_bool(ai)
        if ai is True:
            return True
        if rv is False:  # this will stop updating if a None is ever trapped
            rv = ai
    return rv


def fuzzy_xor(args):
    """Return None if any element of args is not True or False, else
    True (if there are an odd number of True elements), else False."""
    t = 0
    for a in args:
        ai = fuzzy_bool(a)
        if ai:
            t += 1
        elif ai is None:
            return
    return t % 2 == 1


def fuzzy_nand(args):
    """Return False if all args are True, True if they are all False,
    else None."""
    return fuzzy_not(fuzzy_and(args))


class Logic:
    """Logical expression"""
    # {} 'op' -> LogicClass
    op_2class: dict[str, type[Logic]] = {}

    def __new__(cls, *args):
        obj = object.__new__(cls)
        obj.args = args
        return obj

    def __getnewargs__(self):
        return self.args

    def __hash__(self):
        return hash((type(self).__name__,) + tuple(self.args))

    def __eq__(a, b):
        if not isinstance(b, type(a)):
            return False
        else:
            return a.args == b.args

    def __ne__(a, b):
        if not isinstance(b, type(a)):
            return True
        else:
            return a.args != b.args

    def __lt__(self, other):
        if self.__cmp__(other) == -1:
            return True
        return False

    def __cmp__(self, other):
        if type(self) is not type(other):
            a = str(type(self))
            b = str(type(other))
        else:
            a = self.args
            b = other.args
        return (a > b) - (a < b)

    def __str__(self):
        return '%s(%s)' % (self.__class__.__name__,
                           ', '.join(str(a) for a in self.args))

    __repr__ = __str__

    @staticmethod
    def fromstring(text):
        """Logic from string with space around & and | but none after !.

           e.g.

           !a & b | c
        """
        lexpr = None  # current logical expression
        schedop = None  # scheduled operation
        for term in text.split():
            # operation symbol
            if term in '&|':
                if schedop is not None:
                    raise ValueError(
                        'double op forbidden: "%s %s"' % (term, schedop))
                if lexpr is None:
                    raise ValueError(
                        '%s cannot be in the beginning of expression' % term)
                schedop = term
                continue
            if '&' in term or '|' in term:
                raise ValueError('& and | must have space around them')
            if term[0] == '!':
                if len(term) == 1:
                    raise ValueError('do not include space after "!"')
                term = Not(term[1:])

            # already scheduled operation, e.g. '&'
            if schedop:
                lexpr = Logic.op_2class[schedop](lexpr, term)
                schedop = None
                continue

            # this should be atom
            if lexpr is not None:
                raise ValueError(
                    'missing op between "%s" and "%s"' % (lexpr, term))

            lexpr = term

        # let's check that we ended up in correct state
        if schedop is not None:
            raise ValueError('premature end-of-expression in "%s"' % text)
        if lexpr is None:
            raise ValueError('"%s" is empty' % text)

        # everything looks good now
        return lexpr


class AndOr_Base(Logic):

    def __new__(cls, *args):
        bargs = []
        for a in args:
            if a == cls.op_x_notx:
                return a
            elif a == (not cls.op_x_notx):
                continue    # skip this argument
            bargs.append(a)

        args = sorted(set(cls.flatten(bargs)), key=hash)

        for a in args:
            if Not(a) in args:
                return cls.op_x_notx

        if len(args) == 1:
            return args.pop()
        elif len(args) == 0:
            return not cls.op_x_notx

        return Logic.__new__(cls, *args)

    @classmethod
    def flatten(cls, args):
        # quick-n-dirty flattening for And and Or
        args_queue = list(args)
        res = []

        while True:
            try:
                arg = args_queue.pop(0)
            except IndexError:
                break
            if isinstance(arg, Logic):
                if isinstance(arg, cls):
                    args_queue.extend(arg.args)
                    continue
            res.append(arg)

        args = tuple(res)
        return args


class And(AndOr_Base):
    op_x_notx = False

    def _eval_propagate_not(self):
        # !(a&b&c ...) == !a | !b | !c ...
        return Or(*[Not(a) for a in self.args])

    # (a|b|...) & c == (a&c) | (b&c) | ...
    def expand(self):

        # first locate Or
        for i, arg in enumerate(self.args):
            if isinstance(arg, Or):
                arest = self.args[:i] + self.args[i + 1:]

                orterms = [And(*(arest + (a,))) for a in arg.args]
                for j in range(len(orterms)):
                    if isinstance(orterms[j], Logic):
                        orterms[j] = orterms[j].expand()

                res = Or(*orterms)
                return res

        return self


class Or(AndOr_Base):
    op_x_notx = True

    def _eval_propagate_not(self):
        # !(a|b|c ...) == !a & !b & !c ...
        return And(*[Not(a) for a in self.args])


class Not(Logic):

    def __new__(cls, arg):
        if isinstance(arg, str):
            return Logic.__new__(cls, arg)

        elif isinstance(arg, bool):
            return not arg
        elif isinstance(arg, Not):
            return arg.args[0]

        elif isinstance(arg, Logic):
            # XXX this is a hack to expand right from the beginning
            arg = arg._eval_propagate_not()
            return arg

        else:
            raise ValueError('Not: unknown argument %r' % (arg,))

    @property
    def arg(self):
        return self.args[0]


Logic.op_2class['&'] = And
Logic.op_2class['|'] = Or
Logic.op_2class['!'] = Not
<|endoftext|>r"""This is rule-based deduction system for SymPy

The whole thing is split into two parts

 - rules compilation and preparation of tables
 - runtime inference

For rule-based inference engines, the classical work is RETE algorithm [1],
[2] Although we are not implementing it in full (or even significantly)
it's still worth a read to understand the underlying ideas.

In short, every rule in a system of rules is one of two forms:

 - atom                     -> ...      (alpha rule)
 - And(atom1, atom2, ...)   -> ...      (beta rule)


The major complexity is in efficient beta-rules processing and usually for an
expert system a lot of effort goes into code that operates on beta-rules.


Here we take minimalistic approach to get something usable first.

 - (preparation)    of alpha- and beta- networks, everything except
 - (runtime)        FactRules.deduce_all_facts

             _____________________________________
            ( Kirr: I've never thought that doing )
            ( logic stuff is that difficult...    )
             -------------------------------------
                    o   ^__^
                     o  (oo)\_______
                        (__)\       )\/\
                            ||----w |
                            ||     ||


Some references on the topic
----------------------------

[1] https://en.wikipedia.org/wiki/Rete_algorithm
[2] http://reports-archive.adm.cs.cmu.edu/anon/1995/CMU-CS-95-113.pdf

https://en.wikipedia.org/wiki/Propositional_formula
https://en.wikipedia.org/wiki/Inference_rule
https://en.wikipedia.org/wiki/List_of_rules_of_inference
"""

from collections import defaultdict
from typing import Iterator

from .logic import Logic, And, Or, Not


def _base_fact(atom):
    """Return the literal fact of an atom.

    Effectively, this merely strips the Not around a fact.
    """
    if isinstance(atom, Not):
        return atom.arg
    else:
        return atom


def _as_pair(atom):
    if isinstance(atom, Not):
        return (atom.arg, False)
    else:
        return (atom, True)

# XXX this prepares forward-chaining rules for alpha-network


def transitive_closure(implications):
    """
    Computes the transitive closure of a list of implications

    Uses Warshall's algorithm, as described at
    http://www.cs.hope.edu/~cusack/Notes/Notes/DiscreteMath/Warshall.pdf.
    """
    full_implications = set(implications)
    literals = set().union(*map(set, full_implications))

    for k in literals:
        for i in literals:
            if (i, k) in full_implications:
                for j in literals:
                    if (k, j) in full_implications:
                        full_implications.add((i, j))

    return full_implications


def deduce_alpha_implications(implications):
    """deduce all implications

       Description by example
       ----------------------

       given set of logic rules:

         a -> b
         b -> c

       we deduce all possible rules:

         a -> b, c
         b -> c


       implications: [] of (a,b)
       return:       {} of a -> set([b, c, ...])
    """
    implications = implications + [(Not(j), Not(i)) for (i, j) in implications]
    res = defaultdict(set)
    full_implications = transitive_closure(implications)
    for a, b in full_implications:
        if a == b:
            continue    # skip a->a cyclic input

        res[a].add(b)

    # Clean up tautologies and check consistency
    for a, impl in res.items():
        impl.discard(a)
        na = Not(a)
        if na in impl:
            raise ValueError(
                'implications are inconsistent: %s -> %s %s' % (a, na, impl))

    return res


def apply_beta_to_alpha_route(alpha_implications, beta_rules):
    """apply additional beta-rules (And conditions) to already-built
    alpha implication tables

       TODO: write about

       - static extension of alpha-chains
       - attaching refs to beta-nodes to alpha chains


       e.g.

       alpha_implications:

       a  ->  [b, !c, d]
       b  ->  [d]
       ...


       beta_rules:

       &(b,d) -> e


       then we'll extend a's rule to the following

       a  ->  [b, !c, d, e]
    """
    x_impl = {}
    for x in alpha_implications.keys():
        x_impl[x] = (set(alpha_implications[x]), [])
    for bcond, bimpl in beta_rules:
        for bk in bcond.args:
            if bk in x_impl:
                continue
            x_impl[bk] = (set(), [])

    # static extensions to alpha rules:
    # A: x -> a,b   B: &(a,b) -> c  ==>  A: x -> a,b,c
    seen_static_extension = True
    while seen_static_extension:
        seen_static_extension = False

        for bcond, bimpl in beta_rules:
            if not isinstance(bcond, And):
                raise TypeError("Cond is not And")
            bargs = set(bcond.args)
            for x, (ximpls, bb) in x_impl.items():
                x_all = ximpls | {x}
                # A: ... -> a   B: &(...) -> a  is non-informative
                if bimpl not in x_all and bargs.issubset(x_all):
                    ximpls.add(bimpl)

                    # we introduced new implication - now we have to restore
                    # completeness of the whole set.
                    bimpl_impl = x_impl.get(bimpl)
                    if bimpl_impl is not None:
                        ximpls |= bimpl_impl[0]
                    seen_static_extension = True

    # attach beta-nodes which can be possibly triggered by an alpha-chain
    for bidx, (bcond, bimpl) in enumerate(beta_rules):
        bargs = set(bcond.args)
        for x, (ximpls, bb) in x_impl.items():
            x_all = ximpls | {x}
            # A: ... -> a   B: &(...) -> a      (non-informative)
            if bimpl in x_all:
                continue
            # A: x -> a...  B: &(!a,...) -> ... (will never trigger)
            # A: x -> a...  B: &(...) -> !a     (will never trigger)
            if any(Not(xi) in bargs or Not(xi) == bimpl for xi in x_all):
                continue

            if bargs & x_all:
                bb.append(bidx)

    return x_impl


def rules_2prereq(rules):
    """build prerequisites table from rules

       Description by example
       ----------------------

       given set of logic rules:

         a -> b, c
         b -> c

       we build prerequisites (from what points something can be deduced):

         b <- a
         c <- a, b

       rules:   {} of a -> [b, c, ...]
       return:  {} of c <- [a, b, ...]

       Note however, that this prerequisites may be *not* enough to prove a
       fact. An example is 'a -> b' rule, where prereq(a) is b, and prereq(b)
       is a. That's because a=T -> b=T, and b=F -> a=F, but a=F -> b=?
    """
    prereq = defaultdict(set)
    for (a, _), impl in rules.items():
        if isinstance(a, Not):
            a = a.args[0]
        for (i, _) in impl:
            if isinstance(i, Not):
                i = i.args[0]
            prereq[i].add(a)
    return prereq

################
# RULES PROVER #
################


class TautologyDetected(Exception):
    """(internal) Prover uses it for reporting detected tautology"""
    pass


class Prover:
    """ai - prover of logic rules

       given a set of initial rules, Prover tries to prove all possible rules
       which follow from given premises.

       As a result proved_rules are always either in one of two forms: alpha or
       beta:

       Alpha rules
       -----------

       This are rules of the form::

         a -> b & c & d & ...


       Beta rules
       ----------

       This are rules of the form::

         &(a,b,...) -> c & d & ...


       i.e. beta rules are join conditions that say that something follows when
       *several* facts are true at the same time.
    """

    def __init__(self):
        self.proved_rules = []
        self._rules_seen = set()

    def split_alpha_beta(self):
        """split proved rules into alpha and beta chains"""
        rules_alpha = []    # a      -> b
        rules_beta = []     # &(...) -> b
        for a, b in self.proved_rules:
            if isinstance(a, And):
                rules_beta.append((a, b))
            else:
                rules_alpha.append((a, b))
        return rules_alpha, rules_beta

    @property
    def rules_alpha(self):
        return self.split_alpha_beta()[0]

    @property
    def rules_beta(self):
        return self.split_alpha_beta()[1]

    def process_rule(self, a, b):
        """process a -> b rule"""   # TODO write more?
        if (not a) or isinstance(b, bool):
            return
        if isinstance(a, bool):
            return
        if (a, b) in self._rules_seen:
            return
        else:
            self._rules_seen.add((a, b))

        # this is the core of processing
        try:
            self._process_rule(a, b)
        except TautologyDetected:
            pass

    def _process_rule(self, a, b):
        # right part first

        # a -> b & c    -->  a -> b  ;  a -> c
        # (?) FIXME this is only correct when b & c != null !

        if isinstance(b, And):
            sorted_bargs = sorted(b.args, key=str)
            for barg in sorted_bargs:
                self.process_rule(a, barg)

        # a -> b | c    -->  !b & !c -> !a
        #               -->   a & !b -> c
        #               -->   a & !c -> b
        elif isinstance(b, Or):
            sorted_bargs = sorted(b.args, key=str)
            # detect tautology first
            if not isinstance(a, Logic):    # Atom
                # tautology:  a -> a|c|...
                if a in sorted_bargs:
                    raise TautologyDetected(a, b, 'a -> a|c|...')
            self.process_rule(And(*[Not(barg) for barg in b.args]), Not(a))

            for bidx in range(len(sorted_bargs)):
                barg = sorted_bargs[bidx]
                brest = sorted_bargs[:bidx] + sorted_bargs[bidx + 1:]
                self.process_rule(And(a, Not(barg)), Or(*brest))

        # left part

        # a & b -> c    -->  IRREDUCIBLE CASE -- WE STORE IT AS IS
        #                    (this will be the basis of beta-network)
        elif isinstance(a, And):
            sorted_aargs = sorted(a.args, key=str)
            if b in sorted_aargs:
                raise TautologyDetected(a, b, 'a & b -> a')
            self.proved_rules.append((a, b))
            # XXX NOTE at present we ignore  !c -> !a | !b

        elif isinstance(a, Or):
            sorted_aargs = sorted(a.args, key=str)
            if b in sorted_aargs:
                raise TautologyDetected(a, b, 'a | b -> a')
            for aarg in sorted_aargs:
                self.process_rule(aarg, b)

        else:
            # both `a` and `b` are atoms
            self.proved_rules.append((a, b))             # a  -> b
            self.proved_rules.append((Not(b), Not(a)))   # !b -> !a

########################################


class FactRules:
    """Rules that describe how to deduce facts in logic space

       When defined, these rules allow implications to quickly be determined
       for a set of facts. For this precomputed deduction tables are used.
       see `deduce_all_facts`   (forward-chaining)

       Also it is possible to gather prerequisites for a fact, which is tried
       to be proven.    (backward-chaining)


       Definition Syntax
       -----------------

       a -> b       -- a=T -> b=T  (and automatically b=F -> a=F)
       a -> !b      -- a=T -> b=F
       a == b       -- a -> b & b -> a
       a -> b & c   -- a=T -> b=T & c=T
       # TODO b | c


       Internals
       ---------

       .full_implications[k, v]: all the implications of fact k=v
       .beta_triggers[k, v]: beta rules that might be triggered when k=v
       .prereq  -- {} k <- [] of k's prerequisites

       .defined_facts -- set of defined fact names
    """

    def __init__(self, rules):
        """Compile rules into internal lookup tables"""

        if isinstance(rules, str):
            rules = rules.splitlines()

        # --- parse and process rules ---
        P = Prover()

        for rule in rules:
            # XXX `a` is hardcoded to be always atom
            a, op, b = rule.split(None, 2)

            a = Logic.fromstring(a)
            b = Logic.fromstring(b)

            if op == '->':
                P.process_rule(a, b)
            elif op == '==':
                P.process_rule(a, b)
                P.process_rule(b, a)
            else:
                raise ValueError('unknown op %r' % op)

        # --- build deduction networks ---
        self.beta_rules = []
        for bcond, bimpl in P.rules_beta:
            self.beta_rules.append(
                ({_as_pair(a) for a in bcond.args}, _as_pair(bimpl)))

        # deduce alpha implications
        impl_a = deduce_alpha_implications(P.rules_alpha)

        # now:
        # - apply beta rules to alpha chains  (static extension), and
        # - further associate beta rules to alpha chain (for inference
        # at runtime)
        impl_ab = apply_beta_to_alpha_route(impl_a, P.rules_beta)

        # extract defined fact names
        self.defined_facts = {_base_fact(k) for k in impl_ab.keys()}

        # build rels (forward chains)
        full_implications = defaultdict(set)
        beta_triggers = defaultdict(set)
        for k, (impl, betaidxs) in impl_ab.items():
            full_implications[_as_pair(k)] = {_as_pair(i) for i in impl}
            beta_triggers[_as_pair(k)] = betaidxs

        self.full_implications = full_implications
        self.beta_triggers = beta_triggers

        # build prereq (backward chains)
        prereq = defaultdict(set)
        rel_prereq = rules_2prereq(full_implications)
        for k, pitems in rel_prereq.items():
            prereq[k] |= pitems
        self.prereq = prereq

    def _to_python(self) -> str:
        """ Generate a string with plain python representation of the instance """
        return '\n'.join(self.print_rules())

    @classmethod
    def _from_python(cls, data : dict):
        """ Generate an instance from the plain python representation """
        self = cls('')
        for key in ['full_implications', 'beta_triggers', 'prereq']:
            d=defaultdict(set)
            d.update(data[key])
            setattr(self, key, d)
        self.beta_rules = data['beta_rules']
        self.defined_facts = set(data['defined_facts'])

        return self

    def _defined_facts_lines(self):
        yield 'defined_facts = ['
        for fact in sorted(self.defined_facts):
            yield f'    {fact!r},'
        yield '] # defined_facts'

    def _full_implications_lines(self):
        yield 'full_implications = dict( ['
        for fact in sorted(self.defined_facts):
            for value in (True, False):
                yield f'    # Implications of {fact} = {value}:'
                yield f'    (({fact!r}, {value!r}), set( ('
                implications = self.full_implications[(fact, value)]
                for implied in sorted(implications):
                    yield f'        {implied!r},'
                yield '       ) ),'
                yield '     ),'
        yield ' ] ) # full_implications'

    def _prereq_lines(self):
        yield 'prereq = {'
        yield ''
        for fact in sorted(self.prereq):
            yield f'    # facts that could determine the value of {fact}'
            yield f'    {fact!r}: {{'
            for pfact in sorted(self.prereq[fact]):
                yield f'        {pfact!r},'
            yield '    },'
            yield ''
        yield '} # prereq'

    def _beta_rules_lines(self):
        reverse_implications = defaultdict(list)
        for n, (pre, implied) in enumerate(self.beta_rules):
            reverse_implications[implied].append((pre, n))

        yield '# Note: the order of the beta rules is used in the beta_triggers'
        yield 'beta_rules = ['
        yield ''
        m = 0
        indices = {}
        for implied in sorted(reverse_implications):
            fact, value = implied
            yield f'    # Rules implying {fact} = {value}'
            for pre, n in reverse_implications[implied]:
                indices[n] = m
                m += 1
                setstr = ", ".join(map(str, sorted(pre)))
                yield f'    ({{{setstr}}},'
                yield f'        {implied!r}),'
            yield ''
        yield '] # beta_rules'

        yield 'beta_triggers = {'
        for query in sorted(self.beta_triggers):
            fact, value = query
            triggers = [indices[n] for n in self.beta_triggers[query]]
            yield f'    {query!r}: {triggers!r},'
        yield '} # beta_triggers'

    def print_rules(self) -> Iterator[str]:
        """ Returns a generator with lines to represent the facts and rules """
        yield from self._defined_facts_lines()
        yield ''
        yield ''
        yield from self._full_implications_lines()
        yield ''
        yield ''
        yield from self._prereq_lines()
        yield ''
        yield ''
        yield from self._beta_rules_lines()
        yield ''
        yield ''
        yield "generated_assumptions = {'defined_facts': defined_facts, 'full_implications': full_implications,"
        yield "               'prereq': prereq, 'beta_rules': beta_rules, 'beta_triggers': beta_triggers}"


class InconsistentAssumptions(ValueError):
    def __str__(self):
        kb, fact, value = self.args
        return "%s, %s=%s" % (kb, fact, value)


class FactKB(dict):
    """
    A simple propositional knowledge base relying on compiled inference rules.
    """
    def __str__(self):
        return '{\n%s}' % ',\n'.join(
            ["\t%s: %s" % i for i in sorted(self.items())])

    def __init__(self, rules):
        self.rules = rules

    def _tell(self, k, v):
        """Add fact k=v to the knowledge base.

        Returns True if the KB has actually been updated, False otherwise.
        """
        if k in self and self[k] is not None:
            if self[k] == v:
                return False
            else:
                raise InconsistentAssumptions(self, k, v)
        else:
            self[k] = v
            return True

    # *********************************************
    # * This is the workhorse, so keep it *fast*. *
    # *********************************************
    def deduce_all_facts(self, facts):
        """
        Update the KB with all the implications of a list of facts.

        Facts can be specified as a dictionary or as a list of (key, value)
        pairs.
        """
        # keep frequently used attributes locally, so we'll avoid extra
        # attribute access overhead
        full_implications = self.rules.full_implications
        beta_triggers = self.rules.beta_triggers
        beta_rules = self.rules.beta_rules

        if isinstance(facts, dict):
            facts = facts.items()

        while facts:
            beta_maytrigger = set()

            # --- alpha chains ---
            for k, v in facts:
                if not self._tell(k, v) or v is None:
                    continue

                # lookup routing tables
                for key, value in full_implications[k, v]:
                    self._tell(key, value)

                beta_maytrigger.update(beta_triggers[k, v])

            # --- beta chains ---
            facts = []
            for bidx in beta_maytrigger:
                bcond, bimpl = beta_rules[bidx]
                if all(self.get(k) is v for k, v in bcond):
                    facts.append(bimpl)
<|endoftext|>"""Base class for all the objects in SymPy"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Iterable
from itertools import zip_longest
from functools import cmp_to_key
from typing import TYPE_CHECKING, overload

from .assumptions import _prepare_class_assumptions
from .cache import cacheit
from .sympify import _sympify, sympify, SympifyError, _external_converter
from .sorting import ordered
from .kind import Kind, UndefinedKind
from ._print_helpers import Printable

from sympy.utilities.decorator import deprecated
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable, numbered_symbols
from sympy.utilities.misc import filldedent, func_name


if TYPE_CHECKING:
    from typing import ClassVar, TypeVar, Any
    from typing_extensions import Self
    from .assumptions import StdFactKB
    from .symbol import Symbol

    Tbasic = TypeVar("Tbasic", bound='Basic')


def as_Basic(expr):
    """Return expr as a Basic instance using strict sympify
    or raise a TypeError; this is just a wrapper to _sympify,
    raising a TypeError instead of a SympifyError."""
    try:
        return _sympify(expr)
    except SympifyError:
        raise TypeError(
            'Argument must be a Basic object, not `%s`' % func_name(
            expr))


# Key for sorting commutative args in canonical order
# by name. This is used for canonical ordering of the
# args for Add and Mul *if* the names of both classes
# being compared appear here. Some things in this list
# are not spelled the same as their name so they do not,
# in effect, appear here. See Basic.compare.
ordering_of_classes = [
    # singleton numbers
    'Zero', 'One', 'Half', 'Infinity', 'NaN', 'NegativeOne', 'NegativeInfinity',
    # numbers
    'Integer', 'Rational', 'Float',
    # singleton symbols
    'Exp1', 'Pi', 'ImaginaryUnit',
    # symbols
    'Symbol', 'Wild',
    # arithmetic operations
    'Pow', 'Mul', 'Add',
    # function values
    'Derivative', 'Integral',
    # defined singleton functions
    'Abs', 'Sign', 'Sqrt',
    'Floor', 'Ceiling',
    'Re', 'Im', 'Arg',
    'Conjugate',
    'Exp', 'Log',
    'Sin', 'Cos', 'Tan', 'Cot', 'ASin', 'ACos', 'ATan', 'ACot',
    'Sinh', 'Cosh', 'Tanh', 'Coth', 'ASinh', 'ACosh', 'ATanh', 'ACoth',
    'RisingFactorial', 'FallingFactorial',
    'factorial', 'binomial',
    'Gamma', 'LowerGamma', 'UpperGamma', 'PolyGamma',
    'Erf',
    # special polynomials
    'Chebyshev', 'Chebyshev2',
    # undefined functions
    'Function', 'WildFunction',
    # anonymous functions
    'Lambda',
    # Landau O symbol
    'Order',
    # relational operations
    'Equality', 'Unequality', 'StrictGreaterThan', 'StrictLessThan',
    'GreaterThan', 'LessThan',
]

def _cmp_name(x: type, y: type) -> int:
    """return -1, 0, 1 if the name of x is before that of y.
    A string comparison is done if either name does not appear
    in `ordering_of_classes`. This is the helper for
    ``Basic.compare``

    Examples
    ========

    >>> from sympy import cos, tan, sin
    >>> from sympy.core import basic
    >>> save = basic.ordering_of_classes
    >>> basic.ordering_of_classes = ()
    >>> basic._cmp_name(cos, tan)
    -1
    >>> basic.ordering_of_classes = ["tan", "sin", "cos"]
    >>> basic._cmp_name(cos, tan)
    1
    >>> basic._cmp_name(sin, cos)
    -1
    >>> basic.ordering_of_classes = save

    """
    n1 = x.__name__
    n2 = y.__name__
    if n1 == n2:
        return 0

    # If the other object is not a Basic subclass, then we are not equal to it.
    if not issubclass(y, Basic):
        return -1

    UNKNOWN = len(ordering_of_classes) + 1
    try:
        i1 = ordering_of_classes.index(n1)
    except ValueError:
        i1 = UNKNOWN
    try:
        i2 = ordering_of_classes.index(n2)
    except ValueError:
        i2 = UNKNOWN
    if i1 == UNKNOWN and i2 == UNKNOWN:
        return (n1 > n2) - (n1 < n2)
    return (i1 > i2) - (i1 < i2)



@cacheit
def _get_postprocessors(clsname, arg_type):
    # Since only Add, Mul, Pow can be clsname, this cache
    # is not quadratic.
    postprocessors = set()
    mappings = _get_postprocessors_for_type(arg_type)
    for mapping in mappings:
        f = mapping.get(clsname, None)
        if f is not None:
            postprocessors.update(f)
    return postprocessors

@cacheit
def _get_postprocessors_for_type(arg_type):
    return tuple(
        Basic._constructor_postprocessor_mapping[cls]
        for cls in arg_type.mro()
        if cls in Basic._constructor_postprocessor_mapping
    )


class Basic(Printable):
    """
    Base class for all SymPy objects.

    Notes and conventions
    =====================

    1) Always use ``.args``, when accessing parameters of some instance:

    >>> from sympy import cot
    >>> from sympy.abc import x, y

    >>> cot(x).args
    (x,)

    >>> cot(x).args[0]
    x

    >>> (x*y).args
    (x, y)

    >>> (x*y).args[1]
    y


    2) Never use internal methods or variables (the ones prefixed with ``_``):

    >>> cot(x)._args    # do not use this, use cot(x).args instead
    (x,)


    3)  By "SymPy object" we mean something that can be returned by
        ``sympify``.  But not all objects one encounters using SymPy are
        subclasses of Basic.  For example, mutable objects are not:

        >>> from sympy import Basic, Matrix, sympify
        >>> A = Matrix([[1, 2], [3, 4]]).as_mutable()
        >>> isinstance(A, Basic)
        False

        >>> B = sympify(A)
        >>> isinstance(B, Basic)
        True
    """
    __slots__ = ('_mhash',              # hash value
                 '_args',               # arguments
                 '_assumptions'
                )

    _args: tuple[Basic, ...]
    _mhash: int | None

    @property
    def __sympy__(self):
        return True

    def __init_subclass__(cls):
        # Initialize the default_assumptions FactKB and also any assumptions
        # property methods. This method will only be called for subclasses of
        # Basic but not for Basic itself so we call
        # _prepare_class_assumptions(Basic) below the class definition.
        super().__init_subclass__()
        _prepare_class_assumptions(cls)

    # To be overridden with True in the appropriate subclasses
    is_number = False
    is_Atom = False
    is_Symbol = False
    is_symbol = False
    is_Indexed = False
    is_Dummy = False
    is_Wild = False
    is_Function = False
    is_Add = False
    is_Mul = False
    is_Pow = False
    is_Number = False
    is_Float = False
    is_Rational = False
    is_Integer = False
    is_NumberSymbol = False
    is_Order = False
    is_Derivative = False
    is_Piecewise = False
    is_Poly = False
    is_AlgebraicNumber = False
    is_Relational = False
    is_Equality = False
    is_Boolean = False
    is_Not = False
    is_Matrix = False
    is_Vector = False
    is_Point = False
    is_MatAdd = False
    is_MatMul = False

    default_assumptions: ClassVar[StdFactKB]

    is_composite: bool | None
    is_noninteger: bool | None
    is_extended_positive: bool | None
    is_negative: bool | None
    is_complex: bool | None
    is_extended_nonpositive: bool | None
    is_integer: bool | None
    is_positive: bool | None
    is_rational: bool | None
    is_extended_nonnegative: bool | None
    is_infinite: bool | None
    is_antihermitian: bool | None
    is_extended_negative: bool | None
    is_extended_real: bool | None
    is_finite: bool | None
    is_polar: bool | None
    is_imaginary: bool | None
    is_transcendental: bool | None
    is_extended_nonzero: bool | None
    is_nonzero: bool | None
    is_odd: bool | None
    is_algebraic: bool | None
    is_prime: bool | None
    is_commutative: bool | None
    is_nonnegative: bool | None
    is_nonpositive: bool | None
    is_hermitian: bool | None
    is_irrational: bool | None
    is_real: bool | None
    is_zero: bool | None
    is_even: bool | None

    kind: Kind = UndefinedKind

    def __new__(cls, *args):
        obj = object.__new__(cls)
        obj._assumptions = cls.default_assumptions
        obj._mhash = None  # will be set by __hash__ method.

        obj._args = args  # all items in args must be Basic objects
        return obj

    def copy(self):
        return self.func(*self.args)

    def __getnewargs__(self):
        return self.args

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        for name, value in state.items():
            setattr(self, name, value)

    def __reduce_ex__(self, protocol):
        if protocol < 2:
            msg = "Only pickle protocol 2 or higher is supported by SymPy"
            raise NotImplementedError(msg)
        return super().__reduce_ex__(protocol)

    def __hash__(self) -> int:
        # hash cannot be cached using cache_it because infinite recurrence
        # occurs as hash is needed for setting cache dictionary keys
        h = self._mhash
        if h is None:
            h = hash((type(self).__name__,) + self._hashable_content())
            self._mhash = h
        return h

    def _hashable_content(self):
        """Return a tuple of information about self that can be used to
        compute the hash. If a class defines additional attributes,
        like ``name`` in Symbol, then this method should be updated
        accordingly to return such relevant attributes.

        Defining more than _hashable_content is necessary if __eq__ has
        been defined by a class. See note about this in Basic.__eq__."""
        return self._args

    @property
    def assumptions0(self):
        """
        Return object `type` assumptions.

        For example:

          Symbol('x', real=True)
          Symbol('x', integer=True)

        are different objects. In other words, besides Python type (Symbol in
        this case), the initial assumptions are also forming their typeinfo.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.abc import x
        >>> x.assumptions0
        {'commutative': True}
        >>> x = Symbol("x", positive=True)
        >>> x.assumptions0
        {'commutative': True, 'complex': True, 'extended_negative': False,
         'extended_nonnegative': True, 'extended_nonpositive': False,
         'extended_nonzero': True, 'extended_positive': True, 'extended_real':
         True, 'finite': True, 'hermitian': True, 'imaginary': False,
         'infinite': False, 'negative': False, 'nonnegative': True,
         'nonpositive': False, 'nonzero': True, 'positive': True, 'real':
         True, 'zero': False}
        """
        return {}

    def compare(self, other):
        """
        Return -1, 0, 1 if the object is less than, equal,
        or greater than other in a canonical sense.
        Non-Basic are always greater than Basic.
        If both names of the classes being compared appear
        in the `ordering_of_classes` then the ordering will
        depend on the appearance of the names there.
        If either does not appear in that list, then the
        comparison is based on the class name.
        If the names are the same then a comparison is made
        on the length of the hashable content.
        Items of the equal-lengthed contents are then
        successively compared using the same rules. If there
        is never a difference then 0 is returned.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> x.compare(y)
        -1
        >>> x.compare(x)
        0
        >>> y.compare(x)
        1

        """
        # all redefinitions of __cmp__ method should start with the
        # following lines:
        if self is other:
            return 0
        n1 = self.__class__
        n2 = other.__class__
        c = _cmp_name(n1, n2)
        if c:
            return c
        #
        st = self._hashable_content()
        ot = other._hashable_content()
        len_st = len(st)
        len_ot = len(ot)
        c = (len_st > len_ot) - (len_st < len_ot)
        if c:
            return c
        for l, r in zip(st, ot):
            if isinstance(l, Basic):
                c = l.compare(r)
            elif isinstance(l, frozenset):
                l = Basic(*l) if isinstance(l, frozenset) else l
                r = Basic(*r) if isinstance(r, frozenset) else r
                c = l.compare(r)
            else:
                c = (l > r) - (l < r)
            if c:
                return c
        return 0

    @classmethod
    def fromiter(cls, args, **assumptions):
        """
        Create a new object from an iterable.

        This is a convenience function that allows one to create objects from
        any iterable, without having to convert to a list or tuple first.

        Examples
        ========

        >>> from sympy import Tuple
        >>> Tuple.fromiter(i for i in range(5))
        (0, 1, 2, 3, 4)

        """
        return cls(*tuple(args), **assumptions)

    @classmethod
    def class_key(cls) -> tuple[int, int, str]:
        """Nice order of classes."""
        return 5, 0, cls.__name__

    @cacheit
    def sort_key(self, order=None):
        """
        Return a sort key.

        Examples
        ========

        >>> from sympy import S, I

        >>> sorted([S(1)/2, I, -I], key=lambda x: x.sort_key())
        [1/2, -I, I]

        >>> S("[x, 1/x, 1/x**2, x**2, x**(1/2), x**(1/4), x**(3/2)]")
        [x, 1/x, x**(-2), x**2, sqrt(x), x**(1/4), x**(3/2)]
        >>> sorted(_, key=lambda x: x.sort_key())
        [x**(-2), 1/x, x**(1/4), sqrt(x), x, x**(3/2), x**2]

        """

        # XXX: remove this when issue 5169 is fixed
        def inner_key(arg):
            if isinstance(arg, Basic):
                return arg.sort_key(order)
            else:
                return arg

        args = self._sorted_args
        args = len(args), tuple([inner_key(arg) for arg in args])
        return self.class_key(), args, S.One.sort_key(), S.One

    def _do_eq_sympify(self, other):
        """Returns a boolean indicating whether a == b when either a
        or b is not a Basic. This is only done for types that were either
        added to `converter` by a 3rd party or when the object has `_sympy_`
        defined. This essentially reuses the code in `_sympify` that is
        specific for this use case. Non-user defined types that are meant
        to work with SymPy should be handled directly in the __eq__ methods
        of the `Basic` classes it could equate to and not be converted. Note
        that after conversion, `==`  is used again since it is not
        necessarily clear whether `self` or `other`'s __eq__ method needs
        to be used."""
        for superclass in type(other).__mro__:
            conv = _external_converter.get(superclass)
            if conv is not None:
                return self == conv(other)
        if hasattr(other, '_sympy_'):
            return self == other._sympy_()
        return NotImplemented

    def __eq__(self, other):
        """Return a boolean indicating whether a == b on the basis of
        their symbolic trees.

        This is the same as a.compare(b) == 0 but faster.

        Notes
        =====

        If a class that overrides __eq__() needs to retain the
        implementation of __hash__() from a parent class, the
        interpreter must be told this explicitly by setting
        __hash__ : Callable[[object], int] = <ParentClass>.__hash__.
        Otherwise the inheritance of __hash__() will be blocked,
        just as if __hash__ had been explicitly set to None.

        References
        ==========

        from https://docs.python.org/dev/reference/datamodel.html#object.__hash__
        """
        if self is other:
            return True

        if not isinstance(other, Basic):
            return self._do_eq_sympify(other)

        # check for pure number expr
        if  not (self.is_Number and other.is_Number) and (
                type(self) != type(other)):
            return False
        a, b = self._hashable_content(), other._hashable_content()
        if a != b:
            return False
        # check number *in* an expression
        for a, b in zip(a, b):
            if not isinstance(a, Basic):
                continue
            if a.is_Number and type(a) != type(b):
                return False
        return True

    def __ne__(self, other):
        """``a != b``  -> Compare two symbolic trees and see whether they are different

        this is the same as:

        ``a.compare(b) != 0``

        but faster
        """
        return not self == other

    def dummy_eq(self, other, symbol=None):
        """
        Compare two expressions and handle dummy symbols.

        Examples
        ========

        >>> from sympy import Dummy
        >>> from sympy.abc import x, y

        >>> u = Dummy('u')

        >>> (u**2 + 1).dummy_eq(x**2 + 1)
        True
        >>> (u**2 + 1) == (x**2 + 1)
        False

        >>> (u**2 + y).dummy_eq(x**2 + y, x)
        True
        >>> (u**2 + y).dummy_eq(x**2 + y, y)
        False

        """
        s = self.as_dummy()
        o = _sympify(other)
        o = o.as_dummy()

        dummy_symbols = [i for i in s.free_symbols if i.is_Dummy]

        if len(dummy_symbols) == 1:
            dummy = dummy_symbols.pop()
        else:
            return s == o

        if symbol is None:
            symbols = o.free_symbols

            if len(symbols) == 1:
                symbol = symbols.pop()
            else:
                return s == o

        tmp = dummy.__class__()

        return s.xreplace({dummy: tmp}) == o.xreplace({symbol: tmp})

    @overload
    def atoms(self) -> set[Basic]: ...
    @overload
    def atoms(self, *types: Tbasic | type[Tbasic]) -> set[Tbasic]: ...

    def atoms(self, *types: Tbasic | type[Tbasic]) -> set[Basic] | set[Tbasic]:
        """Returns the atoms that form the current object.

        By default, only objects that are truly atomic and cannot
        be divided into smaller pieces are returned: symbols, numbers,
        and number symbols like I and pi. It is possible to request
        atoms of any type, however, as demonstrated below.

        Examples
        ========

        >>> from sympy import I, pi, sin
        >>> from sympy.abc import x, y
        >>> (1 + x + 2*sin(y + I*pi)).atoms()
        {1, 2, I, pi, x, y}

        If one or more types are given, the results will contain only
        those types of atoms.

        >>> from sympy import Number, NumberSymbol, Symbol
        >>> (1 + x + 2*sin(y + I*pi)).atoms(Symbol)
        {x, y}

        >>> (1 + x + 2*sin(y + I*pi)).atoms(Number)
        {1, 2}

        >>> (1 + x + 2*sin(y + I*pi)).atoms(Number, NumberSymbol)
        {1, 2, pi}

        >>> (1 + x + 2*sin(y + I*pi)).atoms(Number, NumberSymbol, I)
        {1, 2, I, pi}

        Note that I (imaginary unit) and zoo (complex infinity) are special
        types of number symbols and are not part of the NumberSymbol class.

        The type can be given implicitly, too:

        >>> (1 + x + 2*sin(y + I*pi)).atoms(x) # x is a Symbol
        {x, y}

        Be careful to check your assumptions when using the implicit option
        since ``S(1).is_Integer = True`` but ``type(S(1))`` is ``One``, a special type
        of SymPy atom, while ``type(S(2))`` is type ``Integer`` and will find all
        integers in an expression:

        >>> from sympy import S
        >>> (1 + x + 2*sin(y + I*pi)).atoms(S(1))
        {1}

        >>> (1 + x + 2*sin(y + I*pi)).atoms(S(2))
        {1, 2}

        Finally, arguments to atoms() can select more than atomic atoms: any
        SymPy type (loaded in core/__init__.py) can be listed as an argument
        and those types of "atoms" as found in scanning the arguments of the
        expression recursively:

        >>> from sympy import Function, Mul
        >>> from sympy.core.function import AppliedUndef
        >>> f = Function('f')
        >>> (1 + f(x) + 2*sin(y + I*pi)).atoms(Function)
        {f(x), sin(y + I*pi)}
        >>> (1 + f(x) + 2*sin(y + I*pi)).atoms(AppliedUndef)
        {f(x)}

        >>> (1 + x + 2*sin(y + I*pi)).atoms(Mul)
        {I*pi, 2*sin(y + I*pi)}

        """
        nodes = _preorder_traversal(self)
        if types:
            types2 = tuple([t if isinstance(t, type) else type(t) for t in types])
            return {node for node in nodes if isinstance(node, types2)}
        else:
            return {node for node in nodes if not node.args}

    @property
    def free_symbols(self) -> set[Basic]:
        """Return from the atoms of self those which are free symbols.

        Not all free symbols are ``Symbol`` (see examples)

        For most expressions, all symbols are free symbols. For some classes
        this is not true. e.g. Integrals use Symbols for the dummy variables
        which are bound variables, so Integral has a method to return all
        symbols except those. Derivative keeps track of symbols with respect
        to which it will perform a derivative; those are
        bound variables, too, so it has its own free_symbols method.

        Any other method that uses bound variables should implement a
        free_symbols method.

        Examples
        ========

        >>> from sympy import Derivative, Integral, IndexedBase
        >>> from sympy.abc import x, y, n
        >>> (x + 1).free_symbols
        {x}
        >>> Integral(x, y).free_symbols
        {x, y}

        Not all free symbols are actually symbols:

        >>> IndexedBase('F')[0].free_symbols
        {F, F[0]}

        The symbols of differentiation are not included unless they
        appear in the expression being differentiated.

        >>> Derivative(x + y, y).free_symbols
        {x, y}
        >>> Derivative(x, y).free_symbols
        {x}
        >>> Derivative(x, (y, n)).free_symbols
        {n, x}

        If you want to know if a symbol is in the variables of the
        Derivative you can do so as follows:

        >>> Derivative(x, y).has_free(y)
        True
        """
        empty: set[Basic] = set()
        return empty.union(*(a.free_symbols for a in self.args))

    @property
    def expr_free_symbols(self):
        sympy_deprecation_warning("""
        The expr_free_symbols property is deprecated. Use free_symbols to get
        the free symbols of an expression.
        """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-expr-free-symbols")
        return set()

    def as_dummy(self) -> "Self":
        """Return the expression with any objects having structurally
        bound symbols replaced with unique, canonical symbols within
        the object in which they appear and having only the default
        assumption for commutativity being True. When applied to a
        symbol a new symbol having only the same commutativity will be
        returned.

        Examples
        ========

        >>> from sympy import Integral, Symbol
        >>> from sympy.abc import x
        >>> r = Symbol('r', real=True)
        >>> Integral(r, (r, x)).as_dummy()
        Integral(_0, (_0, x))
        >>> _.variables[0].is_real is None
        True
        >>> r.as_dummy()
        _r

        Notes
        =====

        Any object that has structurally bound variables should have
        a property, ``bound_symbols`` that returns those symbols
        appearing in the object.
        """
        from .symbol import Dummy, Symbol
        def can(x):
            # mask free that shadow bound
            free = x.free_symbols
            bound = set(x.bound_symbols)
            d = {i: Dummy() for i in bound & free}
            x = x.subs(d)
            # replace bound with canonical names
            x = x.xreplace(x.canonical_variables)
            # return after undoing masking
            return x.xreplace({v: k for k, v in d.items()})
        if not self.has(Symbol):
            return self
        return self.replace(
            lambda x: hasattr(x, 'bound_symbols'),
            can,
            simultaneous=False) # type:ignore

    @property
    def canonical_variables(self) -> dict[Basic, Symbol]:
        """Return a dictionary mapping any variable defined in
        ``self.bound_symbols`` to Symbols that do not clash
        with any free symbols in the expression.

        Examples
        ========

        >>> from sympy import Lambda
        >>> from sympy.abc import x
        >>> Lambda(x, 2*x).canonical_variables
        {x: _0}
        """
        bound: list[Basic] | None = getattr(self, 'bound_symbols', None)
        if bound is None:
            return {}
        dums = numbered_symbols('_')
        reps = {}
        # watch out for free symbol that are not in bound symbols;
        # those that are in bound symbols are about to get changed

        # XXX: free_symbols only returns particular kinds of expressions that
        # generally have a .name attribute. There is not a proper class/type
        # that represents this.
        names = {i.name for i in self.free_symbols - set(bound)} # type: ignore
        for b in bound:
            d = next(dums)
            if b.is_Symbol:
                while d.name in names:
                    d = next(dums)
            reps[b] = d
        return reps

    def rcall(self, *args):
        """Apply on the argument recursively through the expression tree.

        This method is used to simulate a common abuse of notation for
        operators. For instance, in SymPy the following will not work:

        ``(x+Lambda(y, 2*y))(z) == x+2*z``,

        however, you can use:

        >>> from sympy import Lambda
        >>> from sympy.abc import x, y, z
        >>> (x + Lambda(y, 2*y)).rcall(z)
        x + 2*z
        """
        if callable(self):
            return self(*args)
        elif self.args:
            newargs = [sub.rcall(*args) for sub in self.args]
            return self.func(*newargs)
        else:
            return self

    def is_hypergeometric(self, k):
        from sympy.simplify.simplify import hypersimp
        from sympy.functions.elementary.piecewise import Piecewise
        if self.has(Piecewise):
            return None
        return hypersimp(self, k) is not None

    @property
    def is_comparable(self):
        """Return True if self can be computed to a real number
        (or already is a real number) with precision, else False.

        Examples
        ========

        >>> from sympy import exp_polar, pi, I
        >>> (I*exp_polar(I*pi/2)).is_comparable
        True
        >>> (I*exp_polar(I*pi*2)).is_comparable
        False

        A False result does not mean that `self` cannot be rewritten
        into a form that would be comparable. For example, the
        difference computed below is zero but without simplification
        it does not evaluate to a zero with precision:

        >>> e = 2**pi*(1 + 2**pi)
        >>> dif = e - e.expand()
        >>> dif.is_comparable
        False
        >>> dif.n(2)._prec
        1

        """
        return self._eval_is_comparable()

    def _eval_is_comparable(self) -> bool:
        # Expr.is_comparable overrides this
        return False

    @property
    def func(self):
        """
        The top-level function in an expression.

        The following should hold for all objects::

            >> x == x.func(*x.args)

        Examples
        ========

        >>> from sympy.abc import x
        >>> a = 2*x
        >>> a.func
        <class 'sympy.core.mul.Mul'>
        >>> a.args
        (2, x)
        >>> a.func(*a.args)
        2*x
        >>> a == a.func(*a.args)
        True

        """
        return self.__class__

    @property
    def args(self) -> tuple[Basic, ...]:
        """Returns a tuple of arguments of 'self'.

        Examples
        ========

        >>> from sympy import cot
        >>> from sympy.abc import x, y

        >>> cot(x).args
        (x,)

        >>> cot(x).args[0]
        x

        >>> (x*y).args
        (x, y)

        >>> (x*y).args[1]
        y

        Notes
        =====

        Never use self._args, always use self.args.
        Only use _args in __new__ when creating a new function.
        Do not override .args() from Basic (so that it is easy to
        change the interface in the future if needed).
        """
        return self._args

    @property
    def _sorted_args(self):
        """
        The same as ``args``.  Derived classes which do not fix an
        order on their arguments should override this method to
        produce the sorted representation.
        """
        return self.args

    def as_content_primitive(self, radical=False, clear=True):
        """A stub to allow Basic args (like Tuple) to be skipped when computing
        the content and primitive components of an expression.

        See Also
        ========

        sympy.core.expr.Expr.as_content_primitive
        """
        return S.One, self

    @overload
    def subs(self, arg1: Mapping[Basic | complex, Basic | complex], arg2: None=None, **kwargs: Any) -> Basic: ...
    @overload
    def subs(self, arg1: Iterable[tuple[Basic | complex, Basic | complex]], arg2: None=None, **kwargs: Any) -> Basic: ...
    @overload
    def subs(self, arg1: Basic | complex, arg2: Basic | complex, **kwargs: Any) -> Basic: ...

    def subs(self, arg1: Mapping[Basic | complex, Basic | complex]
            | Iterable[tuple[Basic | complex, Basic | complex]] | Basic | complex,
             arg2: Basic | complex | None = None, **kwargs: Any) -> Basic:
        """
        Substitutes old for new in an expression after sympifying args.

        `args` is either:
          - two arguments, e.g. foo.subs(old, new)
          - one iterable argument, e.g. foo.subs(iterable). The iterable may be
             o an iterable container with (old, new) pairs. In this case the
               replacements are processed in the order given with successive
               patterns possibly affecting replacements already made.
             o a dict or set whose key/value items correspond to old/new pairs.
               In this case the old/new pairs will be sorted by op count and in
               case of a tie, by number of args and the default_sort_key. The
               resulting sorted list is then processed as an iterable container
               (see previous).

        If the keyword ``simultaneous`` is True, the subexpressions will not be
        evaluated until all the substitutions have been made.

        Examples
        ========

        >>> from sympy import pi, exp, limit, oo
        >>> from sympy.abc import x, y
        >>> (1 + x*y).subs(x, pi)
        pi*y + 1
        >>> (1 + x*y).subs({x:pi, y:2})
        1 + 2*pi
        >>> (1 + x*y).subs([(x, pi), (y, 2)])
        1 + 2*pi
        >>> reps = [(y, x**2), (x, 2)]
        >>> (x + y).subs(reps)
        6
        >>> (x + y).subs(reversed(reps))
        x**2 + 2

        >>> (x**2 + x**4).subs(x**2, y)
        y**2 + y

        To replace only the x**2 but not the x**4, use xreplace:

        >>> (x**2 + x**4).xreplace({x**2: y})
        x**4 + y

        To delay evaluation until all substitutions have been made,
        set the keyword ``simultaneous`` to True:

        >>> (x/y).subs([(x, 0), (y, 0)])
        0
        >>> (x/y).subs([(x, 0), (y, 0)], simultaneous=True)
        nan

        This has the added feature of not allowing subsequent substitutions
        to affect those already made:

        >>> ((x + y)/y).subs({x + y: y, y: x + y})
        1
        >>> ((x + y)/y).subs({x + y: y, y: x + y}, simultaneous=True)
        y/(x + y)

        In order to obtain a canonical result, unordered iterables are
        sorted by count_op length, number of arguments and by the
        default_sort_key to break any ties. All other iterables are left
        unsorted.

        >>> from sympy import sqrt, sin, cos
        >>> from sympy.abc import a, b, c, d, e

        >>> A = (sqrt(sin(2*x)), a)
        >>> B = (sin(2*x), b)
        >>> C = (cos(2*x), c)
        >>> D = (x, d)
        >>> E = (exp(x), e)

        >>> expr = sqrt(sin(2*x))*sin(exp(x)*x)*cos(2*x) + sin(2*x)

        >>> expr.subs(dict([A, B, C, D, E]))
        a*c*sin(d*e) + b

        The resulting expression represents a literal replacement of the
        old arguments with the new arguments. This may not reflect the
        limiting behavior of the expression:

        >>> (x**3 - 3*x).subs({x: oo})
        nan

        >>> limit(x**3 - 3*x, x, oo)
        oo

        If the substitution will be followed by numerical
        evaluation, it is better to pass the substitution to
        evalf as

        >>> (1/x).evalf(subs={x: 3.0}, n=21)
        0.333333333333333333333

        rather than

        >>> (1/x).subs({x: 3.0}).evalf(21)
        0.333333333333333314830

        as the former will ensure that the desired level of precision is
        obtained.

        See Also
        ========
        replace: replacement capable of doing wildcard-like matching,
                 parsing of match, and conditional replacements
        xreplace: exact node replacement in expr tree; also capable of
                  using matching rules
        sympy.core.evalf.EvalfMixin.evalf: calculates the given formula to a desired level of precision

        """
        from .containers import Dict
        from .symbol import Dummy, Symbol
        from .numbers import _illegal

        items: Iterable[tuple[Basic | complex, Basic | complex]]

        unordered = False
        if arg2 is None:

            if isinstance(arg1, set):
                items = arg1
                unordered = True
            elif isinstance(arg1, (Dict, Mapping)):
                unordered = True
                items = arg1.items() # type: ignore
            elif not iterable(arg1):
                raise ValueError(filldedent("""
                   When a single argument is passed to subs
                   it should be a dictionary of old: new pairs or an iterable
                   of (old, new) tuples."""))
            else:
                items = arg1 # type: ignore
        else:
            items = [(arg1, arg2)] # type: ignore

        def sympify_old(old) -> Basic:
            if isinstance(old, str):
                # Use Symbol rather than parse_expr for old
                return Symbol(old)
            elif isinstance(old, type):
                # Allow a type e.g. Function('f') or sin
                return sympify(old, strict=False)
            else:
                return sympify(old, strict=True)

        def sympify_new(new) -> Basic:
            if isinstance(new, (str, type)):
                # Allow a type or parse a string input
                return sympify(new, strict=False)
            else:
                return sympify(new, strict=True)

        sequence = [(sympify_old(s1), sympify_new(s2)) for s1, s2 in items]

        # skip if there is no change
        sequence = [(s1, s2) for s1, s2 in sequence if not _aresame(s1, s2)]

        simultaneous = kwargs.pop('simultaneous', False)

        if unordered:
            from .sorting import _nodes, default_sort_key
            sequence_dict = dict(sequence)
            # order so more complex items are first and items
            # of identical complexity are ordered so
            # f(x) < f(y) < x < y
            # \___ 2 __/    \_1_/  <- number of nodes
            #
            # For more complex ordering use an unordered sequence.
            k = list(ordered(sequence_dict, default=False, keys=(
                lambda x: -_nodes(x),
                default_sort_key,
                )))
            sequence = [(k, sequence_dict[k]) for k in k]
            # do infinities first
            if not simultaneous:
                redo = [i for i, seq in enumerate(sequence) if seq[1] in _illegal]
                for i in reversed(redo):
                    sequence.insert(0, sequence.pop(i))

        if simultaneous:  # XXX should this be the default for dict subs?
            reps = {}
            rv = self
            kwargs['hack2'] = True
            m = Dummy('subs_m')
            for old, new in sequence:
                com = new.is_commutative
                if com is None:
                    com = True
                d = Dummy('subs_d', commutative=com)
                # using d*m so Subs will be used on dummy variables
                # in things like Derivative(f(x, y), x) in which x
                # is both free and bound
                rv = rv._subs(old, d*m, **kwargs)
                if not isinstance(rv, Basic):
                    break
                reps[d] = new
            reps[m] = S.One  # get rid of m
            return rv.xreplace(reps)
        else:
            rv = self
            for old, new in sequence:
                rv = rv._subs(old, new, **kwargs)
                if not isinstance(rv, Basic):
                    break
            return rv

    @cacheit
    def _subs(self, old, new, **hints):
        """Substitutes an expression old -> new.

        If self is not equal to old then _eval_subs is called.
        If _eval_subs does not want to make any special replacement
        then a None is received which indicates that the fallback
        should be applied wherein a search for replacements is made
        amongst the arguments of self.

        >>> from sympy import Add
        >>> from sympy.abc import x, y, z

        Examples
        ========

        Add's _eval_subs knows how to target x + y in the following
        so it makes the change:

        >>> (x + y + z).subs(x + y, 1)
        z + 1

        Add's _eval_subs does not need to know how to find x + y in
        the following:

        >>> Add._eval_subs(z*(x + y) + 3, x + y, 1) is None
        True

        The returned None will cause the fallback routine to traverse the args and
        pass the z*(x + y) arg to Mul where the change will take place and the
        substitution will succeed:

        >>> (z*(x + y) + 3).subs(x + y, 1)
        z + 3

        ** Developers Notes **

        An _eval_subs routine for a class should be written if:

            1) any arguments are not instances of Basic (e.g. bool, tuple);

            2) some arguments should not be targeted (as in integration
               variables);

            3) if there is something other than a literal replacement
               that should be attempted (as in Piecewise where the condition
               may be updated without doing a replacement).

        If it is overridden, here are some special cases that might arise:

            1) If it turns out that no special change was made and all
               the original sub-arguments should be checked for
               replacements then None should be returned.

            2) If it is necessary to do substitutions on a portion of
               the expression then _subs should be called. _subs will
               handle the case of any sub-expression being equal to old
               (which usually would not be the case) while its fallback
               will handle the recursion into the sub-arguments. For
               example, after Add's _eval_subs removes some matching terms
               it must process the remaining terms so it calls _subs
               on each of the un-matched terms and then adds them
               onto the terms previously obtained.

           3) If the initial expression should remain unchanged then
              the original expression should be returned. (Whenever an
              expression is returned, modified or not, no further
              substitution of old -> new is attempted.) Sum's _eval_subs
              routine uses this strategy when a substitution is attempted
              on any of its summation variables.
        """

        def fallback(self, old, new):
            """
            Try to replace old with new in any of self's arguments.
            """
            hit = False
            args = list(self.args)
            for i, arg in enumerate(args):
                if not hasattr(arg, '_eval_subs'):
                    continue
                arg = arg._subs(old, new, **hints)
                if not _aresame(arg, args[i]):
                    hit = True
                    args[i] = arg
            if hit:
                rv = self.func(*args)
                hack2 = hints.get('hack2', False)
                if hack2 and self.is_Mul and not rv.is_Mul:  # 2-arg hack
                    coeff = S.One
                    nonnumber = []
                    for i in args:
                        if i.is_Number:
                            coeff *= i
                        else:
                            nonnumber.append(i)
                    nonnumber = self.func(*nonnumber)
                    if coeff is S.One:
                        return nonnumber
                    else:
                        return self.func(coeff, nonnumber, evaluate=False)
                return rv
            return self

        if _aresame(self, old):
            return new

        rv = self._eval_subs(old, new)
        if rv is None:
            rv = fallback(self, old, new)
        return rv

    def _eval_subs(self, old, new) -> Basic | None:
        """Override this stub if you want to do anything more than
        attempt a replacement of old with new in the arguments of self.

        See also
        ========

        _subs
        """
        return None

    def xreplace(self, rule):
        """
        Replace occurrences of objects within the expression.

        Parameters
        ==========

        rule : dict-like
            Expresses a replacement rule

        Returns
        =======

        xreplace : the result of the replacement

        Examples
        ========

        >>> from sympy import symbols, pi, exp
        >>> x, y, z = symbols('x y z')
        >>> (1 + x*y).xreplace({x: pi})
        pi*y + 1
        >>> (1 + x*y).xreplace({x: pi, y: 2})
        1 + 2*pi

        Replacements occur only if an entire node in the expression tree is
        matched:

        >>> (x*y + z).xreplace({x*y: pi})
        z + pi
        >>> (x*y*z).xreplace({x*y: pi})
        x*y*z
        >>> (2*x).xreplace({2*x: y, x: z})
        y
        >>> (2*2*x).xreplace({2*x: y, x: z})
        4*z
        >>> (x + y + 2).xreplace({x + y: 2})
        x + y + 2
        >>> (x + 2 + exp(x + 2)).xreplace({x + 2: y})
        x + exp(y) + 2

        xreplace does not differentiate between free and bound symbols. In the
        following, subs(x, y) would not change x since it is a bound symbol,
        but xreplace does:

        >>> from sympy import Integral
        >>> Integral(x, (x, 1, 2*x)).xreplace({x: y})
        Integral(y, (y, 1, 2*y))

        Trying to replace x with an expression raises an error:

        >>> Integral(x, (x, 1, 2*x)).xreplace({x: 2*y}) # doctest: +SKIP
        ValueError: Invalid limits given: ((2*y, 1, 4*y),)

        See Also
        ========
        replace: replacement capable of doing wildcard-like matching,
                 parsing of match, and conditional replacements
        subs: substitution of subexpressions as defined by the objects
              themselves.

        """
        value, _ = self._xreplace(rule)
        return value

    def _xreplace(self, rule):
        """
        Helper for xreplace. Tracks whether a replacement actually occurred.
        """
        if self in rule:
            return rule[self], True
        elif rule:
            args = []
            changed = False
            for a in self.args:
                _xreplace = getattr(a, '_xreplace', None)
                if _xreplace is not None:
                    a_xr = _xreplace(rule)
                    args.append(a_xr[0])
                    changed |= a_xr[1]
                else:
                    args.append(a)
            args = tuple(args)
            if changed:
                return self.func(*args), True
        return self, False

    @cacheit
    def has(self, *patterns):
        """
        Test whether any subexpression matches any of the patterns.

        Examples
        ========

        >>> from sympy import sin
        >>> from sympy.abc import x, y, z
        >>> (x**2 + sin(x*y)).has(z)
        False
        >>> (x**2 + sin(x*y)).has(x, y, z)
        True
        >>> x.has(x)
        True

        Note ``has`` is a structural algorithm with no knowledge of
        mathematics. Consider the following half-open interval:

        >>> from sympy import Interval
        >>> i = Interval.Lopen(0, 5); i
        Interval.Lopen(0, 5)
        >>> i.args
        (0, 5, True, False)
        >>> i.has(4)  # there is no "4" in the arguments
        False
        >>> i.has(0)  # there *is* a "0" in the arguments
        True

        Instead, use ``contains`` to determine whether a number is in the
        interval or not:

        >>> i.contains(4)
        True
        >>> i.contains(0)
        False


        Note that ``expr.has(*patterns)`` is exactly equivalent to
        ``any(expr.has(p) for p in patterns)``. In particular, ``False`` is
        returned when the list of patterns is empty.

        >>> x.has()
        False

        """
        return self._has(iterargs, *patterns)

    def has_xfree(self, s: set[Basic]):
        """Return True if self has any of the patterns in s as a
        free argument, else False. This is like `Basic.has_free`
        but this will only report exact argument matches.

        Examples
        ========

        >>> from sympy import Function
        >>> from sympy.abc import x, y
        >>> f = Function('f')
        >>> f(x).has_xfree({f})
        False
        >>> f(x).has_xfree({f(x)})
        True
        >>> f(x + 1).has_xfree({x})
        True
        >>> f(x + 1).has_xfree({x + 1})
        True
        >>> f(x + y + 1).has_xfree({x + 1})
        False
        """
        # protect O(1) containment check by requiring:
        if type(s) is not set:
            raise TypeError('expecting set argument')
        return any(a in s for a in iterfreeargs(self))

    @cacheit
    def has_free(self, *patterns):
        """Return True if self has object(s) ``x`` as a free expression
        else False.

        Examples
        ========

        >>> from sympy import Integral, Function
        >>> from sympy.abc import x, y
        >>> f = Function('f')
        >>> g = Function('g')
        >>> expr = Integral(f(x), (f(x), 1, g(y)))
        >>> expr.free_symbols
        {y}
        >>> expr.has_free(g(y))
        True
        >>> expr.has_free(*(x, f(x)))
        False

        This works for subexpressions and types, too:

        >>> expr.has_free(g)
        True
        >>> (x + y + 1).has_free(y + 1)
        True
        """
        if not patterns:
            return False
        p0 = patterns[0]
        if len(patterns) == 1 and iterable(p0) and not isinstance(p0, Basic):
            # Basic can contain iterables (though not non-Basic, ideally)
            # but don't encourage mixed passing patterns
            raise TypeError(filldedent('''
                Expecting 1 or more Basic args, not a single
                non-Basic iterable. Don't forget to unpack
                iterables: `eq.has_free(*patterns)`'''))
        # try quick test first
        s = set(patterns)
        rv = self.has_xfree(s)
        if rv:
            return rv
        # now try matching through slower _has
        return self._has(iterfreeargs, *patterns)

    def _has(self, iterargs, *patterns):
        # separate out types and unhashable objects
        type_set = set()  # only types
        p_set = set()  # hashable non-types
        for p in patterns:
            if isinstance(p, type) and issubclass(p, Basic):
                type_set.add(p)
                continue
            if not isinstance(p, Basic):
                try:
                    p = _sympify(p)
                except SympifyError:
                    continue  # Basic won't have this in it
            p_set.add(p)  # fails if object defines __eq__ but
                          # doesn't define __hash__
        types = tuple(type_set)   #
        for i in iterargs(self):  #
            if i in p_set:        # <--- here, too
                return True
            if isinstance(i, types):
                return True

        # use matcher if defined, e.g. operations defines
        # matcher that checks for exact subset containment,
        # (x + y + 1).has(x + 1) -> True
        for i in p_set - type_set:  # types don't have matchers
            if not hasattr(i, '_has_matcher'):
                continue
            match = i._has_matcher()
            if any(match(arg) for arg in iterargs(self)):
                return True

        # no success
        return False

    def replace(self, query, value, map=False, simultaneous=True, exact=None) -> Basic:
        """
        Replace matching subexpressions of ``self`` with ``value``.

        If ``map = True`` then also return the mapping {old: new} where ``old``
        was a sub-expression found with query and ``new`` is the replacement
        value for it. If the expression itself does not match the query, then
        the returned value will be ``self.xreplace(map)`` otherwise it should
        be ``self.subs(ordered(map.items()))``.

        Traverses an expression tree and performs replacement of matching
        subexpressions from the bottom to the top of the tree. The default
        approach is to do the replacement in a simultaneous fashion so
        changes made are targeted only once. If this is not desired or causes
        problems, ``simultaneous`` can be set to False.

        In addition, if an expression containing more than one Wild symbol
        is being used to match subexpressions and the ``exact`` flag is None
        it will be set to True so the match will only succeed if all non-zero
        values are received for each Wild that appears in the match pattern.
        Setting this to False accepts a match of 0; while setting it True
        accepts all matches that have a 0 in them. See example below for
        cautions.

        The list of possible combinations of queries and replacement values
        is listed below:

        Examples
        ========

        Initial setup

        >>> from sympy import log, sin, cos, tan, Wild, Mul, Add
        >>> from sympy.abc import x, y
        >>> f = log(sin(x)) + tan(sin(x**2))

        1.1. type -> type
            obj.replace(type, newtype)

            When object of type ``type`` is found, replace it with the
            result of passing its argument(s) to ``newtype``.

            >>> f.replace(sin, cos)
            log(cos(x)) + tan(cos(x**2))
            >>> sin(x).replace(sin, cos, map=True)
            (cos(x), {sin(x): cos(x)})
            >>> (x*y).replace(Mul, Add)
            x + y

        1.2. type -> func
            obj.replace(type, func)

            When object of type ``type`` is found, apply ``func`` to its
            argument(s). ``func`` must be written to handle the number
            of arguments of ``type``.

            >>> f.replace(sin, lambda arg: sin(2*arg))
            log(sin(2*x)) + tan(sin(2*x**2))
            >>> (x*y).replace(Mul, lambda *args: sin(2*Mul(*args)))
            sin(2*x*y)

        2.1. pattern -> expr
            obj.replace(pattern(wild), expr(wild))

            Replace subexpressions matching ``pattern`` with the expression
            written in terms of the Wild symbols in ``pattern``.

            >>> a, b = map(Wild, 'ab')
            >>> f.replace(sin(a), tan(a))
            log(tan(x)) + tan(tan(x**2))
            >>> f.replace(sin(a), tan(a/2))
            log(tan(x/2)) + tan(tan(x**2/2))
            >>> f.replace(sin(a), a)
            log(x) + tan(x**2)
            >>> (x*y).replace(a*x, a)
            y

            Matching is exact by default when more than one Wild symbol
            is used: matching fails unless the match gives non-zero
            values for all Wild symbols:

            >>> (2*x + y).replace(a*x + b, b - a)
            y - 2
            >>> (2*x).replace(a*x + b, b - a)
            2*x

            When set to False, the results may be non-intuitive:

            >>> (2*x).replace(a*x + b, b - a, exact=False)
            2/x

        2.2. pattern -> func
            obj.replace(pattern(wild), lambda wild: expr(wild))

            All behavior is the same as in 2.1 but now a function in terms of
            pattern variables is used rather than an expression:

            >>> f.replace(sin(a), lambda a: sin(2*a))
            log(sin(2*x)) + tan(sin(2*x**2))

        3.1. func -> func
            obj.replace(filter, func)

            Replace subexpression ``e`` with ``func(e)`` if ``filter(e)``
            is True.

            >>> g = 2*sin(x**3)
            >>> g.replace(lambda expr: expr.is_Number, lambda expr: expr**2)
            4*sin(x**9)

        The expression itself is also targeted by the query but is done in
        such a fashion that changes are not made twice.

            >>> e = x*(x*y + 1)
            >>> e.replace(lambda x: x.is_Mul, lambda x: 2*x)
            2*x*(2*x*y + 1)

        When matching a single symbol, `exact` will default to True, but
        this may or may not be the behavior that is desired:

        Here, we want `exact=False`:

        >>> from sympy import Function
        >>> f = Function('f')
        >>> e = f(1) + f(0)
        >>> q = f(a), lambda a: f(a + 1)
        >>> e.replace(*q, exact=False)
        f(1) + f(2)
        >>> e.replace(*q, exact=True)
        f(0) + f(2)

        But here, the nature of matching makes selecting
        the right setting tricky:

        >>> e = x**(1 + y)
        >>> (x**(1 + y)).replace(x**(1 + a), lambda a: x**-a, exact=False)
        x
        >>> (x**(1 + y)).replace(x**(1 + a), lambda a: x**-a, exact=True)
        x**(-x - y + 1)
        >>> (x**y).replace(x**(1 + a), lambda a: x**-a, exact=False)
        x
        >>> (x**y).replace(x**(1 + a), lambda a: x**-a, exact=True)
        x**(1 - y)

        It is probably better to use a different form of the query
        that describes the target expression more precisely:

        >>> (1 + x**(1 + y)).replace(
        ... lambda x: x.is_Pow and x.exp.is_Add and x.exp.args[0] == 1,
        ... lambda x: x.base**(1 - (x.exp - 1)))
        ...
        x**(1 - y) + 1

        See Also
        ========

        subs: substitution of subexpressions as defined by the objects
              themselves.
        xreplace: exact node replacement in expr tree; also capable of
                  using matching rules

        """

        try:
            query = _sympify(query)
        except SympifyError:
            pass
        try:
            value = _sympify(value)
        except SympifyError:
            pass
        if isinstance(query, type):
            _query = lambda expr: isinstance(expr, query)

            if isinstance(value, type):
                _value = lambda expr, result: value(*expr.args)
            elif callable(value):
                _value = lambda expr, result: value(*expr.args)
            else:
                raise TypeError(
                    "given a type, replace() expects another "
                    "type or a callable")
        elif isinstance(query, Basic):
            _query = lambda expr: expr.match(query)
            if exact is None:
                from .symbol import Wild
                exact = (len(query.atoms(Wild)) > 1)

            if isinstance(value, Basic):
                if exact:
                    _value = lambda expr, result: (value.subs(result)
                        if all(result.values()) else expr)
                else:
                    _value = lambda expr, result: value.subs(result)
            elif callable(value):
                # match dictionary keys get the trailing underscore stripped
                # from them and are then passed as keywords to the callable;
                # if ``exact`` is True, only accept match if there are no null
                # values amongst those matched.
                if exact:
                    _value = lambda expr, result: (value(**
                        {str(k)[:-1]: v for k, v in result.items()})
                        if all(val for val in result.values()) else expr)
                else:
                    _value = lambda expr, result: value(**
                        {str(k)[:-1]: v for k, v in result.items()})
            else:
                raise TypeError(
                    "given an expression, replace() expects "
                    "another expression or a callable")
        elif callable(query):
            _query = query

            if callable(value):
                _value = lambda expr, result: value(expr)
            else:
                raise TypeError(
                    "given a callable, replace() expects "
                    "another callable")
        else:
            raise TypeError(
                "first argument to replace() must be a "
                "type, an expression or a callable")

        def walk(rv, F):
            """Apply ``F`` to args and then to result.
            """
            args = getattr(rv, 'args', None)
            if args is not None:
                if args:
                    newargs = tuple([walk(a, F) for a in args])
                    if args != newargs:
                        rv = rv.func(*newargs)
                        if simultaneous:
                            # if rv is something that was already
                            # matched (that was changed) then skip
                            # applying F again
                            for i, e in enumerate(args):
                                if rv == e and e != newargs[i]:
                                    return rv
                rv = F(rv)
            return rv

        mapping = {}  # changes that took place

        def rec_replace(expr):
            result = _query(expr)
            if result or result == {}:
                v = _value(expr, result)
                if v is not None and v != expr:
                    if map:
                        mapping[expr] = v
                    expr = v
            return expr

        rv = walk(self, rec_replace)
        return (rv, mapping) if map else rv # type: ignore

    def find(self, query, group=False):
        """Find all subexpressions matching a query."""
        query = _make_find_query(query)
        results = list(filter(query, _preorder_traversal(self)))

        if not group:
            return set(results)
        return dict(Counter(results))

    def count(self, query):
        """Count the number of matching subexpressions."""
        query = _make_find_query(query)
        return sum(bool(query(sub)) for sub in _preorder_traversal(self))

    def matches(self, expr, repl_dict=None, old=False):
        """
        Helper method for match() that looks for a match between Wild symbols
        in self and expressions in expr.

        Examples
        ========

        >>> from sympy import symbols, Wild, Basic
        >>> a, b, c = symbols('a b c')
        >>> x = Wild('x')
        >>> Basic(a + x, x).matches(Basic(a + b, c)) is None
        True
        >>> Basic(a + x, x).matches(Basic(a + b + c, b + c))
        {x_: b + c}
        """
        expr = sympify(expr)
        if not isinstance(expr, self.__class__):
            return None

        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        if self == expr:
            return repl_dict

        if len(self.args) != len(expr.args):
            return None

        d = repl_dict  # already a copy
        for arg, other_arg in zip(self.args, expr.args):
            if arg == other_arg:
                continue
            if arg.is_Relational:
                try:
                    d = arg.xreplace(d).matches(other_arg, d, old=old)
                except TypeError: # Should be InvalidComparisonError when introduced
                    d = None
            else:
                    d = arg.xreplace(d).matches(other_arg, d, old=old)
            if d is None:
                return None
        return d

    def match(self, pattern, old=False):
        """
        Pattern matching.

        Wild symbols match all.

        Return ``None`` when expression (self) does not match with pattern.
        Otherwise return a dictionary such that::

          pattern.xreplace(self.match(pattern)) == self

        Examples
        ========

        >>> from sympy import Wild, Sum
        >>> from sympy.abc import x, y
        >>> p = Wild("p")
        >>> q = Wild("q")
        >>> r = Wild("r")
        >>> e = (x+y)**(x+y)
        >>> e.match(p**p)
        {p_: x + y}
        >>> e.match(p**q)
        {p_: x + y, q_: x + y}
        >>> e = (2*x)**2
        >>> e.match(p*q**r)
        {p_: 4, q_: x, r_: 2}
        >>> (p*q**r).xreplace(e.match(p*q**r))
        4*x**2

        Since match is purely structural expressions that are equivalent up to
        bound symbols will not match:

        >>> print(Sum(x, (x, 1, 2)).match(Sum(y, (y, 1, p))))
        None

        An expression with bound symbols can be matched if the pattern uses
        a distinct ``Wild`` for each bound symbol:

        >>> Sum(x, (x, 1, 2)).match(Sum(q, (q, 1, p)))
        {p_: 2, q_: x}

        The ``old`` flag will give the old-style pattern matching where
        expressions and patterns are essentially solved to give the match. Both
        of the following give None unless ``old=True``:

        >>> (x - 2).match(p - x, old=True)
        {p_: 2*x - 2}
        >>> (2/x).match(p*x, old=True)
        {p_: 2/x**2}

        See Also
        ========

        matches: pattern.matches(expr) is the same as expr.match(pattern)
        xreplace: exact structural replacement
        replace: structural replacement with pattern matching
        Wild: symbolic placeholders for expressions in pattern matching
        """
        pattern = sympify(pattern)
        return pattern.matches(self, old=old)

    def count_ops(self, visual=False):
        """Wrapper for count_ops that returns the operation count."""
        from .function import count_ops
        return count_ops(self, visual)

    def doit(self, **hints):
        """Evaluate objects that are not evaluated by default like limits,
        integrals, sums and products. All objects of this kind will be
        evaluated recursively, unless some species were excluded via 'hints'
        or unless the 'deep' hint was set to 'False'.

        >>> from sympy import Integral
        >>> from sympy.abc import x

        >>> 2*Integral(x, x)
        2*Integral(x, x)

        >>> (2*Integral(x, x)).doit()
        x**2

        >>> (2*Integral(x, x)).doit(deep=False)
        2*Integral(x, x)

        """
        if hints.get('deep', True):
            terms = [term.doit(**hints) if isinstance(term, Basic) else term
                                         for term in self.args]
            return self.func(*terms)
        else:
            return self

    def simplify(self, **kwargs) -> Basic:
        """See the simplify function in sympy.simplify"""
        from sympy.simplify.simplify import simplify
        return simplify(self, **kwargs)

    def refine(self, assumption=True):
        """See the refine function in sympy.assumptions"""
        from sympy.assumptions.refine import refine
        return refine(self, assumption)

    def _eval_derivative_n_times(self, s, n):
        # This is the default evaluator for derivatives (as called by `diff`
        # and `Derivative`), it will attempt a loop to derive the expression
        # `n` times by calling the corresponding `_eval_derivative` method,
        # while leaving the derivative unevaluated if `n` is symbolic.  This
        # method should be overridden if the object has a closed form for its
        # symbolic n-th derivative.
        from .numbers import Integer
        if isinstance(n, (int, Integer)):
            obj = self
            for i in range(n):
                prev = obj
                obj = obj._eval_derivative(s)
                if obj is None:
                    return None
                elif obj == prev:
                    break
            return obj
        else:
            return None

    def rewrite(self, *args, deep=True, **hints):
        """
        Rewrite *self* using a defined rule.

        Rewriting transforms an expression to another, which is mathematically
        equivalent but structurally different. For example you can rewrite
        trigonometric functions as complex exponentials or combinatorial
        functions as gamma function.

        This method takes a *pattern* and a *rule* as positional arguments.
        *pattern* is optional parameter which defines the types of expressions
        that will be transformed. If it is not passed, all possible expressions
        will be rewritten. *rule* defines how the expression will be rewritten.

        Parameters
        ==========

        args : Expr
            A *rule*, or *pattern* and *rule*.
            - *pattern* is a type or an iterable of types.
            - *rule* can be any object.

        deep : bool, optional
            If ``True``, subexpressions are recursively transformed. Default is
            ``True``.

        Examples
        ========

        If *pattern* is unspecified, all possible expressions are transformed.

        >>> from sympy import cos, sin, exp, I
        >>> from sympy.abc import x
        >>> expr = cos(x) + I*sin(x)
        >>> expr.rewrite(exp)
        exp(I*x)

        Pattern can be a type or an iterable of types.

        >>> expr.rewrite(sin, exp)
        exp(I*x)/2 + cos(x) - exp(-I*x)/2
        >>> expr.rewrite([cos,], exp)
        exp(I*x)/2 + I*sin(x) + exp(-I*x)/2
        >>> expr.rewrite([cos, sin], exp)
        exp(I*x)

        Rewriting behavior can be implemented by defining ``_eval_rewrite()``
        method.

        >>> from sympy import Expr, sqrt, pi
        >>> class MySin(Expr):
        ...     def _eval_rewrite(self, rule, args, **hints):
        ...         x, = args
        ...         if rule == cos:
        ...             return cos(pi/2 - x, evaluate=False)
        ...         if rule == sqrt:
        ...             return sqrt(1 - cos(x)**2)
        >>> MySin(MySin(x)).rewrite(cos)
        cos(-cos(-x + pi/2) + pi/2)
        >>> MySin(x).rewrite(sqrt)
        sqrt(1 - cos(x)**2)

        Defining ``_eval_rewrite_as_[...]()`` method is supported for backwards
        compatibility reason. This may be removed in the future and using it is
        discouraged.

        >>> class MySin(Expr):
        ...     def _eval_rewrite_as_cos(self, *args, **hints):
        ...         x, = args
        ...         return cos(pi/2 - x, evaluate=False)
        >>> MySin(x).rewrite(cos)
        cos(-x + pi/2)

        """
        if not args:
            return self

        hints.update(deep=deep)

        pattern = args[:-1]
        rule = args[-1]

        # Special case: map `abs` to `Abs`
        if rule is abs:
            from sympy.functions.elementary.complexes import Abs
            rule = Abs

        # support old design by _eval_rewrite_as_[...] method
        if isinstance(rule, str):
            method = "_eval_rewrite_as_%s" % rule
        elif hasattr(rule, "__name__"):
            # rule is class or function
            clsname = rule.__name__
            method = "_eval_rewrite_as_%s" % clsname
        else:
            # rule is instance
            clsname = rule.__class__.__name__
            method = "_eval_rewrite_as_%s" % clsname

        if pattern:
            if iterable(pattern[0]):
                pattern = pattern[0]
            pattern = tuple(p for p in pattern if self.has(p))
            if not pattern:
                return self
        # hereafter, empty pattern is interpreted as all pattern.

        return self._rewrite(pattern, rule, method, **hints)

    def _rewrite(self, pattern, rule, method, **hints):
        deep = hints.pop('deep', True)
        if deep:
            args = [a._rewrite(pattern, rule, method, **hints)
                    for a in self.args]
        else:
            args = self.args
        if not pattern or any(isinstance(self, p) for p in pattern):
            meth = getattr(self, method, None)
            if meth is not None:
                rewritten = meth(*args, **hints)
            else:
                rewritten = self._eval_rewrite(rule, args, **hints)
            if rewritten is not None:
                return rewritten
        if not args:
            return self
        return self.func(*args)

    def _eval_rewrite(self, rule, args, **hints):
        return None

    _constructor_postprocessor_mapping = {}  # type: ignore

    @classmethod
    def _exec_constructor_postprocessors(cls, obj):
        # WARNING: This API is experimental.

        # This is an experimental API that introduces constructor
        # postprosessors for SymPy Core elements. If an argument of a SymPy
        # expression has a `_constructor_postprocessor_mapping` attribute, it will
        # be interpreted as a dictionary containing lists of postprocessing
        # functions for matching expression node names.

        clsname = obj.__class__.__name__
        postprocessors = {f for i in obj.args
                            for f in _get_postprocessors(clsname, type(i))}
        for f in postprocessors:
            obj = f(obj)

        return obj

    def _sage_(self):
        """
        Convert *self* to a symbolic expression of SageMath.

        This version of the method is merely a placeholder.
        """
        old_method = self._sage_
        from sage.interfaces.sympy import sympy_init # type: ignore
        sympy_init()  # may monkey-patch _sage_ method into self's class or superclasses
        if old_method == self._sage_:
            raise NotImplementedError('conversion to SageMath is not implemented')
        else:
            # call the freshly monkey-patched method
            return self._sage_()

    def could_extract_minus_sign(self) -> bool:
        return False  # see Expr.could_extract_minus_sign

    def is_same(a, b, approx=None):
        """Return True if a and b are structurally the same, else False.
        If `approx` is supplied, it will be used to test whether two
        numbers are the same or not. By default, only numbers of the
        same type will compare equal, so S.Half != Float(0.5).

        Examples
        ========

        In SymPy (unlike Python) two numbers do not compare the same if they are
        not of the same type:

        >>> from sympy import S
        >>> 2.0 == S(2)
        False
        >>> 0.5 == S.Half
        False

        By supplying a function with which to compare two numbers, such
        differences can be ignored. e.g. `equal_valued` will return True
        for decimal numbers having a denominator that is a power of 2,
        regardless of precision.

        >>> from sympy import Float
        >>> from sympy.core.numbers import equal_valued
        >>> (S.Half/4).is_same(Float(0.125, 1), equal_valued)
        True
        >>> Float(1, 2).is_same(Float(1, 10), equal_valued)
        True

        But decimals without a power of 2 denominator will compare
        as not being the same.

        >>> Float(0.1, 9).is_same(Float(0.1, 10), equal_valued)
        False

        But arbitrary differences can be ignored by supplying a function
        to test the equivalence of two numbers:

        >>> import math
        >>> Float(0.1, 9).is_same(Float(0.1, 10), math.isclose)
        True

        Other objects might compare the same even though types are not the
        same. This routine will only return True if two expressions are
        identical in terms of class types.

        >>> from sympy import eye, Basic
        >>> eye(1) == S(eye(1))  # mutable vs immutable
        True
        >>> Basic.is_same(eye(1), S(eye(1)))
        False

        """
        from .numbers import Number
        from .traversal import postorder_traversal as pot
        for t in zip_longest(pot(a), pot(b)):
            if None in t:
                return False
            a, b = t
            if isinstance(a, Number):
                if not isinstance(b, Number):
                    return False
                if approx:
                    return approx(a, b)
            if not (a == b and a.__class__ == b.__class__):
                return False
        return True

_aresame = Basic.is_same  # for sake of others importing this

# key used by Mul and Add to make canonical args
_args_sortkey = cmp_to_key(Basic.compare)

# For all Basic subclasses _prepare_class_assumptions is called by
# Basic.__init_subclass__ but that method is not called for Basic itself so we
# call the function here instead.
_prepare_class_assumptions(Basic)


class Atom(Basic):
    """
    A parent class for atomic things. An atom is an expression with no subexpressions.

    Examples
    ========

    Symbol, Number, Rational, Integer, ...
    But not: Add, Mul, Pow, ...
    """

    is_Atom = True

    __slots__ = ()

    def matches(self, expr, repl_dict=None, old=False):
        if self == expr:
            if repl_dict is None:
                return {}
            return repl_dict.copy()

    def xreplace(self, rule, hack2=False):
        return rule.get(self, self)

    def doit(self, **hints):
        return self

    @classmethod
    def class_key(cls):
        return 2, 0, cls.__name__

    @cacheit
    def sort_key(self, order=None):
        return self.class_key(), (1, (str(self),)), S.One.sort_key(), S.One

    def _eval_simplify(self, **kwargs):
        return self

    @property
    def _sorted_args(self):
        # this is here as a safeguard against accidentally using _sorted_args
        # on Atoms -- they cannot be rebuilt as atom.func(*atom._sorted_args)
        # since there are no args. So the calling routine should be checking
        # to see that this property is not called for Atoms.
        raise AttributeError('Atoms have no args. It might be necessary'
        ' to make a check for Atoms in the calling code.')


def _atomic(e, recursive=False):
    """Return atom-like quantities as far as substitution is
    concerned: Derivatives, Functions and Symbols. Do not
    return any 'atoms' that are inside such quantities unless
    they also appear outside, too, unless `recursive` is True.

    Examples
    ========

    >>> from sympy import Derivative, Function, cos
    >>> from sympy.abc import x, y
    >>> from sympy.core.basic import _atomic
    >>> f = Function('f')
    >>> _atomic(x + y)
    {x, y}
    >>> _atomic(x + f(y))
    {x, f(y)}
    >>> _atomic(Derivative(f(x), x) + cos(x) + y)
    {y, cos(x), Derivative(f(x), x)}

    """
    pot = _preorder_traversal(e)
    seen = set()
    if isinstance(e, Basic):
        free = getattr(e, "free_symbols", None)
        if free is None:
            return {e}
    else:
        return set()
    from .symbol import Symbol
    from .function import Derivative, Function
    atoms = set()
    for p in pot:
        if p in seen:
            pot.skip()
            continue
        seen.add(p)
        if isinstance(p, Symbol) and p in free:
            atoms.add(p)
        elif isinstance(p, (Derivative, Function)):
            if not recursive:
                pot.skip()
            atoms.add(p)
    return atoms


def _make_find_query(query):
    """Convert the argument of Basic.find() into a callable"""
    try:
        query = _sympify(query)
    except SympifyError:
        pass
    if isinstance(query, type):
        return lambda expr: isinstance(expr, query)
    elif isinstance(query, Basic):
        return lambda expr: expr.match(query) is not None
    return query

# Delayed to avoid cyclic import
from .singleton import S
from .traversal import (preorder_traversal as _preorder_traversal,
   iterargs, iterfreeargs)

preorder_traversal = deprecated(
    """
    Using preorder_traversal from the sympy.core.basic submodule is
    deprecated.

    Instead, use preorder_traversal from the top-level sympy namespace, like

        sympy.preorder_traversal
    """,
    deprecated_since_version="1.10",
    active_deprecations_target="deprecated-traversal-functions-moved",
)(_preorder_traversal)
<|endoftext|>"""Core module. Provides the basic operations needed in sympy.
"""

from .sympify import sympify, SympifyError
from .cache import cacheit
from .assumptions import assumptions, check_assumptions, failing_assumptions, common_assumptions
from .basic import Basic, Atom
from .singleton import S
from .expr import Expr, AtomicExpr, UnevaluatedExpr
from .symbol import Symbol, Wild, Dummy, symbols, var
from .numbers import Number, Float, Rational, Integer, NumberSymbol, \
    RealNumber, igcd, ilcm, seterr, E, I, nan, oo, pi, zoo, \
    AlgebraicNumber, comp, mod_inverse
from .power import Pow
from .intfunc import integer_nthroot, integer_log, num_digits, trailing
from .mul import Mul, prod
from .add import Add
from .mod import Mod
from .relational import ( Rel, Eq, Ne, Lt, Le, Gt, Ge,
    Equality, GreaterThan, LessThan, Unequality, StrictGreaterThan,
    StrictLessThan )
from .multidimensional import vectorize
from .function import Lambda, WildFunction, Derivative, diff, FunctionClass, \
    Function, Subs, expand, PoleError, count_ops, \
    expand_mul, expand_log, expand_func, \
    expand_trig, expand_complex, expand_multinomial, nfloat, \
    expand_power_base, expand_power_exp, arity
from .evalf import PrecisionExhausted, N
from .containers import Tuple, Dict
from .exprtools import gcd_terms, factor_terms, factor_nc
from .parameters import evaluate
from .kind import UndefinedKind, NumberKind, BooleanKind
from .traversal import preorder_traversal, bottom_up, use, postorder_traversal
from .sorting import default_sort_key, ordered

# expose singletons
Catalan = S.Catalan
EulerGamma = S.EulerGamma
GoldenRatio = S.GoldenRatio
TribonacciConstant = S.TribonacciConstant

__all__ = [
    'sympify', 'SympifyError',

    'cacheit',

    'assumptions', 'check_assumptions', 'failing_assumptions',
    'common_assumptions',

    'Basic', 'Atom',

    'S',

    'Expr', 'AtomicExpr', 'UnevaluatedExpr',

    'Symbol', 'Wild', 'Dummy', 'symbols', 'var',

    'Number', 'Float', 'Rational', 'Integer', 'NumberSymbol', 'RealNumber',
    'igcd', 'ilcm', 'seterr', 'E', 'I', 'nan', 'oo', 'pi', 'zoo',
    'AlgebraicNumber', 'comp', 'mod_inverse',

    'Pow',

    'integer_nthroot', 'integer_log', 'num_digits', 'trailing',

    'Mul', 'prod',

    'Add',

    'Mod',

    'Rel', 'Eq', 'Ne', 'Lt', 'Le', 'Gt', 'Ge', 'Equality', 'GreaterThan',
    'LessThan', 'Unequality', 'StrictGreaterThan', 'StrictLessThan',

    'vectorize',

    'Lambda', 'WildFunction', 'Derivative', 'diff', 'FunctionClass',
    'Function', 'Subs', 'expand', 'PoleError', 'count_ops', 'expand_mul',
    'expand_log', 'expand_func', 'expand_trig', 'expand_complex',
    'expand_multinomial', 'nfloat', 'expand_power_base', 'expand_power_exp',
    'arity',

    'PrecisionExhausted', 'N',

    'evalf', # The module?

    'Tuple', 'Dict',

    'gcd_terms', 'factor_terms', 'factor_nc',

    'evaluate',

    'Catalan',
    'EulerGamma',
    'GoldenRatio',
    'TribonacciConstant',

    'UndefinedKind', 'NumberKind', 'BooleanKind',

    'preorder_traversal', 'bottom_up', 'use', 'postorder_traversal',

    'default_sort_key', 'ordered',
]
<|endoftext|>"""
The routines here were removed from numbers.py, power.py,
digits.py and factor_.py so they could be imported into core
without raising circular import errors.

Although the name 'intfunc' was chosen to represent functions that
work with integers, it can also be thought of as containing
internal/core functions that are needed by the classes of the core.
"""

import math
import sys
from functools import lru_cache

from .sympify import sympify
from .singleton import S
from sympy.external.gmpy import (gcd as number_gcd, lcm as number_lcm, sqrt,
                                 iroot, bit_scan1, gcdext)
from sympy.utilities.misc import as_int, filldedent


def num_digits(n, base=10):
    """Return the number of digits needed to express n in give base.

    Examples
    ========

    >>> from sympy.core.intfunc import num_digits
    >>> num_digits(10)
    2
    >>> num_digits(10, 2)  # 1010 -> 4 digits
    4
    >>> num_digits(-100, 16)  # -64 -> 2 digits
    2


    Parameters
    ==========

    n: integer
        The number whose digits are counted.

    b: integer
        The base in which digits are computed.

    See Also
    ========
    sympy.ntheory.digits.digits, sympy.ntheory.digits.count_digits
    """
    if base < 0:
        raise ValueError('base must be int greater than 1')
    if not n:
        return 1
    e, t = integer_log(abs(n), base)
    return 1 + e


def integer_log(n, b):
    r"""
    Returns ``(e, bool)`` where e is the largest nonnegative integer
    such that :math:`|n| \geq |b^e|` and ``bool`` is True if $n = b^e$.

    Examples
    ========

    >>> from sympy import integer_log
    >>> integer_log(125, 5)
    (3, True)
    >>> integer_log(17, 9)
    (1, False)

    If the base is positive and the number negative the
    return value will always be the same except for 2:

    >>> integer_log(-4, 2)
    (2, False)
    >>> integer_log(-16, 4)
    (0, False)

    When the base is negative, the returned value
    will only be True if the parity of the exponent is
    correct for the sign of the base:

    >>> integer_log(4, -2)
    (2, True)
    >>> integer_log(8, -2)
    (3, False)
    >>> integer_log(-8, -2)
    (3, True)
    >>> integer_log(-4, -2)
    (2, False)

    See Also
    ========
    integer_nthroot
    sympy.ntheory.primetest.is_square
    sympy.ntheory.factor_.multiplicity
    sympy.ntheory.factor_.perfect_power
    """
    n = as_int(n)
    b = as_int(b)

    if b < 0:
        e, t = integer_log(abs(n), -b)
        # (-2)**3 == -8
        # (-2)**2 = 4
        t = t and e % 2 == (n < 0)
        return e, t
    if b <= 1:
        raise ValueError('base must be 2 or more')
    if n < 0:
        if b != 2:
            return 0, False
        e, t = integer_log(-n, b)
        return e, False
    if n == 0:
        raise ValueError('n cannot be 0')

    if n < b:
        return 0, n == 1
    if b == 2:
        e = n.bit_length() - 1
        return e, trailing(n) == e
    t = trailing(b)
    if 2**t == b:
        e = int(n.bit_length() - 1)//t
        n_ = 1 << (t*e)
        return e, n_ == n

    d = math.floor(math.log10(n) / math.log10(b))
    n_ = b ** d
    while n_ <= n:  # this will iterate 0, 1 or 2 times
        d += 1
        n_ *= b
    return d - (n_ > n), (n_ == n or n_//b == n)


def trailing(n):
    """Count the number of trailing zero digits in the binary
    representation of n, i.e. determine the largest power of 2
    that divides n.

    Examples
    ========

    >>> from sympy import trailing
    >>> trailing(128)
    7
    >>> trailing(63)
    0

    See Also
    ========
    sympy.ntheory.factor_.multiplicity

    """
    if not n:
        return 0
    return bit_scan1(int(n))


@lru_cache(1024)
def igcd(*args):
    """Computes nonnegative integer greatest common divisor.

    Explanation
    ===========

    The algorithm is based on the well known Euclid's algorithm [1]_. To
    improve speed, ``igcd()`` has its own caching mechanism.
    If you do not need the cache mechanism, using ``sympy.external.gmpy.gcd``.

    Examples
    ========

    >>> from sympy import igcd
    >>> igcd(2, 4)
    2
    >>> igcd(5, 10, 15)
    5

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euclidean_algorithm

    """
    if len(args) < 2:
        raise TypeError("igcd() takes at least 2 arguments (%s given)" % len(args))
    return int(number_gcd(*map(as_int, args)))


igcd2 = math.gcd


def igcd_lehmer(a, b):
    r"""Computes greatest common divisor of two integers.

    Explanation
    ===========

    Euclid's algorithm for the computation of the greatest
    common divisor ``gcd(a, b)``  of two (positive) integers
    $a$ and $b$ is based on the division identity
    $$ a = q \times b + r$$,
    where the quotient  $q$  and the remainder  $r$  are integers
    and  $0 \le r < b$. Then each common divisor of  $a$  and  $b$
    divides  $r$, and it follows that  ``gcd(a, b) == gcd(b, r)``.
    The algorithm works by constructing the sequence
    r0, r1, r2, ..., where  r0 = a, r1 = b,  and each  rn
    is the remainder from the division of the two preceding
    elements.

    In Python, ``q = a // b``  and  ``r = a % b``  are obtained by the
    floor division and the remainder operations, respectively.
    These are the most expensive arithmetic operations, especially
    for large  a  and  b.

    Lehmer's algorithm [1]_ is based on the observation that the quotients
    ``qn = r(n-1) // rn``  are in general small integers even
    when  a  and  b  are very large. Hence the quotients can be
    usually determined from a relatively small number of most
    significant bits.

    The efficiency of the algorithm is further enhanced by not
    computing each long remainder in Euclid's sequence. The remainders
    are linear combinations of  a  and  b  with integer coefficients
    derived from the quotients. The coefficients can be computed
    as far as the quotients can be determined from the chosen
    most significant parts of  a  and  b. Only then a new pair of
    consecutive remainders is computed and the algorithm starts
    anew with this pair.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lehmer%27s_GCD_algorithm

    """
    a, b = abs(as_int(a)), abs(as_int(b))
    if a < b:
        a, b = b, a

    # The algorithm works by using one or two digit division
    # whenever possible. The outer loop will replace the
    # pair (a, b) with a pair of shorter consecutive elements
    # of the Euclidean gcd sequence until a and b
    # fit into two Python (long) int digits.
    nbits = 2 * sys.int_info.bits_per_digit

    while a.bit_length() > nbits and b != 0:
        # Quotients are mostly small integers that can
        # be determined from most significant bits.
        n = a.bit_length() - nbits
        x, y = int(a >> n), int(b >> n)  # most significant bits

        # Elements of the Euclidean gcd sequence are linear
        # combinations of a and b with integer coefficients.
        # Compute the coefficients of consecutive pairs
        #     a' = A*a + B*b, b' = C*a + D*b
        # using small integer arithmetic as far as possible.
        A, B, C, D = 1, 0, 0, 1  # initial values

        while True:
            # The coefficients alternate in sign while looping.
            # The inner loop combines two steps to keep track
            # of the signs.

            # At this point we have
            #   A > 0, B <= 0, C <= 0, D > 0,
            #   x' = x + B <= x < x" = x + A,
            #   y' = y + C <= y < y" = y + D,
            # and
            #   x'*N <= a' < x"*N, y'*N <= b' < y"*N,
            # where N = 2**n.

            # Now, if y' > 0, and x"//y' and x'//y" agree,
            # then their common value is equal to  q = a'//b'.
            # In addition,
            #   x'%y" = x' - q*y" < x" - q*y' = x"%y',
            # and
            #   (x'%y")*N < a'%b' < (x"%y')*N.

            # On the other hand, we also have  x//y == q,
            # and therefore
            #   x'%y" = x + B - q*(y + D) = x%y + B',
            #   x"%y' = x + A - q*(y + C) = x%y + A',
            # where
            #    B' = B - q*D < 0, A' = A - q*C > 0.

            if y + C <= 0:
                break
            q = (x + A) // (y + C)

            # Now  x'//y" <= q, and equality holds if
            #   x' - q*y" = (x - q*y) + (B - q*D) >= 0.
            # This is a minor optimization to avoid division.
            x_qy, B_qD = x - q * y, B - q * D
            if x_qy + B_qD < 0:
                break

            # Next step in the Euclidean sequence.
            x, y = y, x_qy
            A, B, C, D = C, D, A - q * C, B_qD

            # At this point the signs of the coefficients
            # change and their roles are interchanged.
            #   A <= 0, B > 0, C > 0, D < 0,
            #   x' = x + A <= x < x" = x + B,
            #   y' = y + D < y < y" = y + C.

            if y + D <= 0:
                break
            q = (x + B) // (y + D)
            x_qy, A_qC = x - q * y, A - q * C
            if x_qy + A_qC < 0:
                break

            x, y = y, x_qy
            A, B, C, D = C, D, A_qC, B - q * D
            # Now the conditions on top of the loop
            # are again satisfied.
            #   A > 0, B < 0, C < 0, D > 0.

        if B == 0:
            # This can only happen when y == 0 in the beginning
            # and the inner loop does nothing.
            # Long division is forced.
            a, b = b, a % b
            continue

        # Compute new long arguments using the coefficients.
        a, b = A * a + B * b, C * a + D * b

    # Small divisors. Finish with the standard algorithm.
    while b:
        a, b = b, a % b

    return a


def ilcm(*args):
    """Computes integer least common multiple.

    Examples
    ========

    >>> from sympy import ilcm
    >>> ilcm(5, 10)
    10
    >>> ilcm(7, 3)
    21
    >>> ilcm(5, 10, 15)
    30

    """
    if len(args) < 2:
        raise TypeError("ilcm() takes at least 2 arguments (%s given)" % len(args))
    return int(number_lcm(*map(as_int, args)))


def igcdex(a, b):
    """Returns x, y, g such that g = x*a + y*b = gcd(a, b).

    Examples
    ========

    >>> from sympy.core.intfunc import igcdex
    >>> igcdex(2, 3)
    (-1, 1, 1)
    >>> igcdex(10, 12)
    (-1, 1, 2)

    >>> x, y, g = igcdex(100, 2004)
    >>> x, y, g
    (-20, 1, 4)
    >>> x*100 + y*2004
    4

    """
    g, x, y = gcdext(int(a), int(b))
    return x, y, g


def mod_inverse(a, m):
    r"""
    Return the number $c$ such that, $a \times c = 1 \pmod{m}$
    where $c$ has the same sign as $m$. If no such value exists,
    a ValueError is raised.

    Examples
    ========

    >>> from sympy import mod_inverse, S

    Suppose we wish to find multiplicative inverse $x$ of
    3 modulo 11. This is the same as finding $x$ such
    that $3x = 1 \pmod{11}$. One value of x that satisfies
    this congruence is 4. Because $3 \times 4 = 12$ and $12 = 1 \pmod{11}$.
    This is the value returned by ``mod_inverse``:

    >>> mod_inverse(3, 11)
    4
    >>> mod_inverse(-3, 11)
    7

    When there is a common factor between the numerators of
    `a` and `m` the inverse does not exist:

    >>> mod_inverse(2, 4)
    Traceback (most recent call last):
    ...
    ValueError: inverse of 2 mod 4 does not exist

    >>> mod_inverse(S(2)/7, S(5)/2)
    7/2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Modular_multiplicative_inverse
    .. [2] https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
    """
    c = None
    try:
        a, m = as_int(a), as_int(m)
        if m != 1 and m != -1:
            x, _, g = igcdex(a, m)
            if g == 1:
                c = x % m
    except ValueError:
        a, m = sympify(a), sympify(m)
        if not (a.is_number and m.is_number):
            raise TypeError(
                filldedent(
                    """
                Expected numbers for arguments; symbolic `mod_inverse`
                is not implemented
                but symbolic expressions can be handled with the
                similar function,
                sympy.polys.polytools.invert"""
                )
            )
        big = m > 1
        if big not in (S.true, S.false):
            raise ValueError("m > 1 did not evaluate; try to simplify %s" % m)
        elif big:
            c = 1 / a
    if c is None:
        raise ValueError("inverse of %s (mod %s) does not exist" % (a, m))
    return c


def isqrt(n):
    r""" Return the largest integer less than or equal to `\sqrt{n}`.

    Parameters
    ==========

    n : non-negative integer

    Returns
    =======

    int : `\left\lfloor\sqrt{n}\right\rfloor`

    Raises
    ======

    ValueError
        If n is negative.
    TypeError
        If n is of a type that cannot be compared to ``int``.
        Therefore, a TypeError is raised for ``str``, but not for ``float``.

    Examples
    ========

    >>> from sympy.core.intfunc import isqrt
    >>> isqrt(0)
    0
    >>> isqrt(9)
    3
    >>> isqrt(10)
    3
    >>> isqrt("30")
    Traceback (most recent call last):
        ...
    TypeError: '<' not supported between instances of 'str' and 'int'
    >>> from sympy.core.numbers import Rational
    >>> isqrt(Rational(-1, 2))
    Traceback (most recent call last):
        ...
    ValueError: n must be nonnegative

    """
    if n < 0:
        raise ValueError("n must be nonnegative")
    return int(sqrt(int(n)))


def integer_nthroot(y, n):
    """
    Return a tuple containing x = floor(y**(1/n))
    and a boolean indicating whether the result is exact (that is,
    whether x**n == y).

    Examples
    ========

    >>> from sympy import integer_nthroot
    >>> integer_nthroot(16, 2)
    (4, True)
    >>> integer_nthroot(26, 2)
    (5, False)

    To simply determine if a number is a perfect square, the is_square
    function should be used:

    >>> from sympy.ntheory.primetest import is_square
    >>> is_square(26)
    False

    See Also
    ========
    sympy.ntheory.primetest.is_square
    integer_log
    """
    x, b = iroot(as_int(y), as_int(n))
    return int(x), b
<|endoftext|>"""Thread-safe global parameters"""

from .cache import clear_cache
from contextlib import contextmanager
from threading import local

class _global_parameters(local):
    """
    Thread-local global parameters.

    Explanation
    ===========

    This class generates thread-local container for SymPy's global parameters.
    Every global parameters must be passed as keyword argument when generating
    its instance.
    A variable, `global_parameters` is provided as default instance for this class.

    WARNING! Although the global parameters are thread-local, SymPy's cache is not
    by now.
    This may lead to undesired result in multi-threading operations.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.core.cache import clear_cache
    >>> from sympy.core.parameters import global_parameters as gp

    >>> gp.evaluate
    True
    >>> x+x
    2*x

    >>> log = []
    >>> def f():
    ...     clear_cache()
    ...     gp.evaluate = False
    ...     log.append(x+x)
    ...     clear_cache()
    >>> import threading
    >>> thread = threading.Thread(target=f)
    >>> thread.start()
    >>> thread.join()

    >>> print(log)
    [x + x]

    >>> gp.evaluate
    True
    >>> x+x
    2*x

    References
    ==========

    .. [1] https://docs.python.org/3/library/threading.html

    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __setattr__(self, name, value):
        if getattr(self, name) != value:
            clear_cache()
        return super().__setattr__(name, value)

global_parameters = _global_parameters(evaluate=True, distribute=True, exp_is_pow=False)

class evaluate:
    """ Control automatic evaluation

    Explanation
    ===========

    This context manager controls whether or not all SymPy functions evaluate
    by default.

    Note that much of SymPy expects evaluated expressions.  This functionality
    is experimental and is unlikely to function as intended on large
    expressions.

    Examples
    ========

    >>> from sympy import evaluate
    >>> from sympy.abc import x
    >>> print(x + x)
    2*x
    >>> with evaluate(False):
    ...     print(x + x)
    x + x
    """
    def __init__(self, x):
        self.x = x
        self.old = []

    def __enter__(self):
        self.old.append(global_parameters.evaluate)
        global_parameters.evaluate = self.x

    def __exit__(self, exc_type, exc_val, exc_tb):
        global_parameters.evaluate = self.old.pop()

@contextmanager
def distribute(x):
    """ Control automatic distribution of Number over Add

    Explanation
    ===========

    This context manager controls whether or not Mul distribute Number over
    Add. Plan is to avoid distributing Number over Add in all of sympy. Once
    that is done, this contextmanager will be removed.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.core.parameters import distribute
    >>> print(2*(x + 1))
    2*x + 2
    >>> with distribute(False):
    ...     print(2*(x + 1))
    2*(x + 1)
    """

    old = global_parameters.distribute

    try:
        global_parameters.distribute = x
        yield
    finally:
        global_parameters.distribute = old


@contextmanager
def _exp_is_pow(x):
    """
    Control whether `e^x` should be represented as ``exp(x)`` or a ``Pow(E, x)``.

    Examples
    ========

    >>> from sympy import exp
    >>> from sympy.abc import x
    >>> from sympy.core.parameters import _exp_is_pow
    >>> with _exp_is_pow(True): print(type(exp(x)))
    <class 'sympy.core.power.Pow'>
    >>> with _exp_is_pow(False): print(type(exp(x)))
    exp
    """
    old = global_parameters.exp_is_pow

    clear_cache()
    try:
        global_parameters.exp_is_pow = x
        yield
    finally:
        clear_cache()
        global_parameters.exp_is_pow = old
<|endoftext|>""" The core's core. """
from __future__ import annotations


class Registry:
    """
    Base class for registry objects.

    Registries map a name to an object using attribute notation. Registry
    classes behave singletonically: all their instances share the same state,
    which is stored in the class object.

    All subclasses should set `__slots__ = ()`.
    """
    __slots__ = ()

    def __setattr__(self, name, obj):
        setattr(self.__class__, name, obj)

    def __delattr__(self, name):
        delattr(self.__class__, name)
<|endoftext|>from __future__ import annotations

from typing import overload

import numbers
import decimal
import fractions
import math

from .containers import Tuple
from .sympify import (SympifyError, _sympy_converter, sympify, _convert_numpy_types,
              _sympify, _is_numpy_instance)
from .singleton import S, Singleton
from .basic import Basic
from .expr import Expr, AtomicExpr
from .evalf import pure_complex
from .cache import cacheit, clear_cache
from .decorators import _sympifyit
from .intfunc import num_digits, igcd, ilcm, mod_inverse, integer_nthroot
from .logic import fuzzy_not
from .kind import NumberKind
from .sorting import ordered
from sympy.external.gmpy import SYMPY_INTS, gmpy, flint
from sympy.multipledispatch import dispatch
import mpmath
import mpmath.libmp as mlib
from mpmath.libmp import bitcount, round_nearest as rnd
from mpmath.libmp.backend import MPZ
from mpmath.libmp import mpf_pow, mpf_pi, mpf_e, phi_fixed
from mpmath.ctx_mp_python import mpnumeric
from mpmath.libmp.libmpf import (
    finf as _mpf_inf, fninf as _mpf_ninf,
    fnan as _mpf_nan, fzero, _normalize as mpf_normalize,
    prec_to_dps, dps_to_prec)
from sympy.utilities.misc import debug
from sympy.utilities.exceptions import sympy_deprecation_warning
from .parameters import global_parameters

_LOG2 = math.log(2)


def comp(z1, z2, tol=None):
    r"""Return a bool indicating whether the error between z1 and z2
    is $\le$ ``tol``.

    Examples
    ========

    If ``tol`` is ``None`` then ``True`` will be returned if
    :math:`|z1 - z2|\times 10^p \le 5` where $p$ is minimum value of the
    decimal precision of each value.

    >>> from sympy import comp, pi
    >>> pi4 = pi.n(4); pi4
    3.142
    >>> comp(_, 3.142)
    True
    >>> comp(pi4, 3.141)
    False
    >>> comp(pi4, 3.143)
    False

    A comparison of strings will be made
    if ``z1`` is a Number and ``z2`` is a string or ``tol`` is ''.

    >>> comp(pi4, 3.1415)
    True
    >>> comp(pi4, 3.1415, '')
    False

    When ``tol`` is provided and $z2$ is non-zero and
    :math:`|z1| > 1` the error is normalized by :math:`|z1|`:

    >>> abs(pi4 - 3.14)/pi4
    0.000509791731426756
    >>> comp(pi4, 3.14, .001)  # difference less than 0.1%
    True
    >>> comp(pi4, 3.14, .0005)  # difference less than 0.1%
    False

    When :math:`|z1| \le 1` the absolute error is used:

    >>> 1/pi4
    0.3183
    >>> abs(1/pi4 - 0.3183)/(1/pi4)
    3.07371499106316e-5
    >>> abs(1/pi4 - 0.3183)
    9.78393554684764e-6
    >>> comp(1/pi4, 0.3183, 1e-5)
    True

    To see if the absolute error between ``z1`` and ``z2`` is less
    than or equal to ``tol``, call this as ``comp(z1 - z2, 0, tol)``
    or ``comp(z1 - z2, tol=tol)``:

    >>> abs(pi4 - 3.14)
    0.00160156249999988
    >>> comp(pi4 - 3.14, 0, .002)
    True
    >>> comp(pi4 - 3.14, 0, .001)
    False
    """
    if isinstance(z2, str):
        if not pure_complex(z1, or_real=True):
            raise ValueError('when z2 is a str z1 must be a Number')
        return str(z1) == z2
    if not z1:
        z1, z2 = z2, z1
    if not z1:
        return True
    if not tol:
        a, b = z1, z2
        if tol == '':
            return str(a) == str(b)
        if tol is None:
            a, b = sympify(a), sympify(b)
            if not all(i.is_number for i in (a, b)):
                raise ValueError('expecting 2 numbers')
            fa = a.atoms(Float)
            fb = b.atoms(Float)
            if not fa and not fb:
                # no floats -- compare exactly
                return a == b
            # get a to be pure_complex
            for _ in range(2):
                ca = pure_complex(a, or_real=True)
                if not ca:
                    if fa:
                        a = a.n(prec_to_dps(min(i._prec for i in fa)))
                        ca = pure_complex(a, or_real=True)
                        break
                    else:
                        fa, fb = fb, fa
                        a, b = b, a
            cb = pure_complex(b)
            if not cb and fb:
                b = b.n(prec_to_dps(min(i._prec for i in fb)))
                cb = pure_complex(b, or_real=True)
            if ca and cb and (ca[1] or cb[1]):
                return all(comp(i, j) for i, j in zip(ca, cb))
            tol = 10**prec_to_dps(min(a._prec, getattr(b, '_prec', a._prec)))
            return int(abs(a - b)*tol) <= 5
    diff = abs(z1 - z2)
    az1 = abs(z1)
    if z2 and az1 > 1:
        return diff/az1 <= tol
    else:
        return diff <= tol


def mpf_norm(mpf, prec):
    """Return the mpf tuple normalized appropriately for the indicated
    precision after doing a check to see if zero should be returned or
    not when the mantissa is 0. ``mpf_normlize`` always assumes that this
    is zero, but it may not be since the mantissa for mpf's values "+inf",
    "-inf" and "nan" have a mantissa of zero, too.

    Note: this is not intended to validate a given mpf tuple, so sending
    mpf tuples that were not created by mpmath may produce bad results. This
    is only a wrapper to ``mpf_normalize`` which provides the check for non-
    zero mpfs that have a 0 for the mantissa.
    """
    sign, man, expt, bc = mpf
    if not man:
        # hack for mpf_normalize which does not do this;
        # it assumes that if man is zero the result is 0
        # (see issue 6639)
        if not bc:
            return fzero
        else:
            # don't change anything; this should already
            # be a well formed mpf tuple
            return mpf

    # Necessary if mpmath is using the gmpy backend
    from mpmath.libmp.backend import MPZ
    rv = mpf_normalize(sign, MPZ(man), expt, bc, prec, rnd)
    return rv

# TODO: we should use the warnings module
_errdict = {"divide": False}


def seterr(divide=False):
    """
    Should SymPy raise an exception on 0/0 or return a nan?

    divide == True .... raise an exception
    divide == False ... return nan
    """
    if _errdict["divide"] != divide:
        clear_cache()
        _errdict["divide"] = divide


def _as_integer_ratio(p):
    neg_pow, man, expt, _ = getattr(p, '_mpf_', mpmath.mpf(p)._mpf_)
    p = [1, -1][neg_pow % 2]*man
    if expt < 0:
        q = 2**-expt
    else:
        q = 1
        p *= 2**expt
    return int(p), int(q)


def _decimal_to_Rational_prec(dec):
    """Convert an ordinary decimal instance to a Rational."""
    if not dec.is_finite():
        raise TypeError("dec must be finite, got %s." % dec)
    s, d, e = dec.as_tuple()
    prec = len(d)
    if e >= 0:  # it's an integer
        rv = Integer(int(dec))
    else:
        s = (-1)**s
        d = sum(di*10**i for i, di in enumerate(reversed(d)))
        rv = Rational(s*d, 10**-e)
    return rv, prec

_dig = str.maketrans(dict.fromkeys('1234567890'))

def _literal_float(s):
    """return True if s is space-trimmed number literal else False

    Python allows underscore as digit separators: there must be a
    digit on each side. So neither a leading underscore nor a
    double underscore are valid as part of a number. A number does
    not have to precede the decimal point, but there must be a
    digit before the optional "e" or "E" that begins the signs
    exponent of the number which must be an integer, perhaps with
    underscore separators.

    SymPy allows space as a separator; if the calling routine replaces
    them with underscores then the same semantics will be enforced
    for them as for underscores: there can only be 1 *between* digits.

    We don't check for error from float(s) because we don't know
    whether s is malicious or not. A regex for this could maybe
    be written but will it be understood by most who read it?
    """
    # mantissa and exponent
    parts = s.split('e')
    if len(parts) > 2:
        return False
    if len(parts) == 2:
        m, e = parts
        if e.startswith(tuple('+-')):
            e = e[1:]
        if not e:
            return False
    else:
        m, e = s, '1'
    # integer and fraction of mantissa
    parts = m.split('.')
    if len(parts) > 2:
        return False
    elif len(parts) == 2:
        i, f = parts
    else:
        i, f = m, '1'
    if not i and not f:
        return False
    if i and i[0] in '+-':
        i = i[1:]
    if not i:  # -.3e4 -> -0.3e4
        i = '1'
    f = f or '1'
    # check that all groups contain only digits and are not null
    for n in (i, f, e):
        for g in n.split('_'):
            if not g or g.translate(_dig):
                return False
    return True

# (a,b) -> gcd(a,b)

# TODO caching with decorator, but not to degrade performance


class Number(AtomicExpr):
    """Represents atomic numbers in SymPy.

    Explanation
    ===========

    Floating point numbers are represented by the Float class.
    Rational numbers (of any size) are represented by the Rational class.
    Integer numbers (of any size) are represented by the Integer class.
    Float and Rational are subclasses of Number; Integer is a subclass
    of Rational.

    For example, ``2/3`` is represented as ``Rational(2, 3)`` which is
    a different object from the floating point number obtained with
    Python division ``2/3``. Even for numbers that are exactly
    represented in binary, there is a difference between how two forms,
    such as ``Rational(1, 2)`` and ``Float(0.5)``, are used in SymPy.
    The rational form is to be preferred in symbolic computations.

    Other kinds of numbers, such as algebraic numbers ``sqrt(2)`` or
    complex numbers ``3 + 4*I``, are not instances of Number class as
    they are not atomic.

    See Also
    ========

    Float, Integer, Rational
    """
    is_commutative = True
    is_number = True
    is_Number = True

    __slots__ = ()

    # Used to make max(x._prec, y._prec) return x._prec when only x is a float
    _prec = -1

    kind = NumberKind

    def __new__(cls, *obj):
        if len(obj) == 1:
            obj = obj[0]

        if isinstance(obj, Number):
            return obj
        if isinstance(obj, SYMPY_INTS):
            return Integer(obj)
        if isinstance(obj, tuple) and len(obj) == 2:
            return Rational(*obj)
        if isinstance(obj, (float, mpmath.mpf, decimal.Decimal)):
            return Float(obj)
        if isinstance(obj, str):
            _obj = obj.lower()  # float('INF') == float('inf')
            if _obj == 'nan':
                return S.NaN
            elif _obj == 'inf':
                return S.Infinity
            elif _obj == '+inf':
                return S.Infinity
            elif _obj == '-inf':
                return S.NegativeInfinity
            val = sympify(obj)
            if isinstance(val, Number):
                return val
            else:
                raise ValueError('String "%s" does not denote a Number' % obj)
        msg = "expected str|int|long|float|Decimal|Number object but got %r"
        raise TypeError(msg % type(obj).__name__)

    def could_extract_minus_sign(self):
        return bool(self.is_extended_negative)

    def invert(self, other, *gens, **args):
        from sympy.polys.polytools import invert
        if getattr(other, 'is_number', True):
            return mod_inverse(self, other)
        return invert(self, other, *gens, **args)

    def __divmod__(self, other):
        from sympy.functions.elementary.complexes import sign

        try:
            other = Number(other)
            if self.is_infinite or S.NaN in (self, other):
                return (S.NaN, S.NaN)
        except TypeError:
            return NotImplemented
        if not other:
            raise ZeroDivisionError('modulo by zero')
        if self.is_Integer and other.is_Integer:
            return Tuple(*divmod(self.p, other.p))
        elif isinstance(other, Float):
            rat = self/Rational(other)
        else:
            rat = self/other
        if other.is_finite:
            w = int(rat) if rat >= 0 else int(rat) - 1
            r = self - other*w
            if r == Float(other):
                w += 1
                r = 0
            if isinstance(self, Float) or isinstance(other, Float):
                r = Float(r)  # in case w or r is 0
        else:
            w = 0 if not self or (sign(self) == sign(other)) else -1
            r = other if w else self
        return Tuple(w, r)

    def __rdivmod__(self, other):
        try:
            other = Number(other)
        except TypeError:
            return NotImplemented
        return divmod(other, self)

    def _as_mpf_val(self, prec):
        """Evaluation of mpf tuple accurate to at least prec bits."""
        raise NotImplementedError('%s needs ._as_mpf_val() method' %
            (self.__class__.__name__))

    def _eval_evalf(self, prec):
        return Float._new(self._as_mpf_val(prec), prec)

    def _as_mpf_op(self, prec):
        prec = max(prec, self._prec)
        return self._as_mpf_val(prec), prec

    def __float__(self):
        return mlib.to_float(self._as_mpf_val(53))

    def floor(self):
        raise NotImplementedError('%s needs .floor() method' %
            (self.__class__.__name__))

    def ceiling(self):
        raise NotImplementedError('%s needs .ceiling() method' %
            (self.__class__.__name__))

    def __floor__(self):
        return self.floor()

    def __ceil__(self):
        return self.ceiling()

    def _eval_conjugate(self):
        return self

    def _eval_order(self, *symbols):
        from sympy.series.order import Order
        # Order(5, x, y) -> Order(1,x,y)
        return Order(S.One, *symbols)

    def _eval_subs(self, old, new):
        if old == -self:
            return -new
        return self  # there is no other possibility

    @classmethod
    def class_key(cls):
        return 1, 0, 'Number'

    @cacheit
    def sort_key(self, order=None):
        return self.class_key(), (0, ()), (), self

    def __neg__(self) -> Number:
        raise NotImplementedError

    @overload
    def __add__(self, other: Number | int | float) -> Number: ...
    @overload
    def __add__(self, other: Expr) -> Expr: ...

    @_sympifyit('other', NotImplemented)
    def __add__(self, other) -> Expr:
        if isinstance(other, Number) and global_parameters.evaluate:
            if other is S.NaN:
                return S.NaN
            elif other is S.Infinity:
                return S.Infinity
            elif other is S.NegativeInfinity:
                return S.NegativeInfinity
        return AtomicExpr.__add__(self, other)

    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other is S.NaN:
                return S.NaN
            elif other is S.Infinity:
                return S.NegativeInfinity
            elif other is S.NegativeInfinity:
                return S.Infinity
        return AtomicExpr.__sub__(self, other)

    @_sympifyit('other', NotImplemented)
    def __mul__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other is S.NaN:
                return S.NaN
            elif other is S.Infinity:
                if self.is_zero:
                    return S.NaN
                elif self.is_positive:
                    return S.Infinity
                else:
                    return S.NegativeInfinity
            elif other is S.NegativeInfinity:
                if self.is_zero:
                    return S.NaN
                elif self.is_positive:
                    return S.NegativeInfinity
                else:
                    return S.Infinity
        elif isinstance(other, Tuple):
            return NotImplemented
        return AtomicExpr.__mul__(self, other)

    @_sympifyit('other', NotImplemented)
    def __truediv__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other is S.NaN:
                return S.NaN
            elif other in (S.Infinity, S.NegativeInfinity):
                return S.Zero
        return AtomicExpr.__truediv__(self, other)

    def __eq__(self, other):
        raise NotImplementedError('%s needs .__eq__() method' %
            (self.__class__.__name__))

    def __ne__(self, other):
        raise NotImplementedError('%s needs .__ne__() method' %
            (self.__class__.__name__))

    def __lt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s < %s" % (self, other))
        raise NotImplementedError('%s needs .__lt__() method' %
            (self.__class__.__name__))

    def __le__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s <= %s" % (self, other))
        raise NotImplementedError('%s needs .__le__() method' %
            (self.__class__.__name__))

    def __gt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s > %s" % (self, other))
        return _sympify(other).__lt__(self)

    def __ge__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            raise TypeError("Invalid comparison %s >= %s" % (self, other))
        return _sympify(other).__le__(self)

    def __hash__(self):
        return super().__hash__()

    def is_constant(self, *wrt, **flags):
        return True

    def as_coeff_mul(self, *deps, rational=True, **kwargs):
        # a -> c*t
        if self.is_Rational or not rational:
            return self, ()
        elif self.is_negative:
            return S.NegativeOne, (-self,)
        return S.One, (self,)

    def as_coeff_add(self, *deps):
        # a -> c + t
        if self.is_Rational:
            return self, ()
        return S.Zero, (self,)

    def as_coeff_Mul(self, rational=False):
        """Efficiently extract the coefficient of a product."""
        if not rational:
            return self, S.One
        return S.One, self

    def as_coeff_Add(self, rational=False):
        """Efficiently extract the coefficient of a summation."""
        if not rational:
            return self, S.Zero
        return S.Zero, self

    def gcd(self, other):
        """Compute GCD of `self` and `other`. """
        from sympy.polys.polytools import gcd
        return gcd(self, other)

    def lcm(self, other):
        """Compute LCM of `self` and `other`. """
        from sympy.polys.polytools import lcm
        return lcm(self, other)

    def cofactors(self, other):
        """Compute GCD and cofactors of `self` and `other`. """
        from sympy.polys.polytools import cofactors
        return cofactors(self, other)


class Float(Number):
    """Represent a floating-point number of arbitrary precision.

    Examples
    ========

    >>> from sympy import Float
    >>> Float(3.5)
    3.50000000000000
    >>> Float(3)
    3.00000000000000

    Creating Floats from strings (and Python ``int`` and ``long``
    types) will give a minimum precision of 15 digits, but the
    precision will automatically increase to capture all digits
    entered.

    >>> Float(1)
    1.00000000000000
    >>> Float(10**20)
    100000000000000000000.
    >>> Float('1e20')
    100000000000000000000.

    However, *floating-point* numbers (Python ``float`` types) retain
    only 15 digits of precision:

    >>> Float(1e20)
    1.00000000000000e+20
    >>> Float(1.23456789123456789)
    1.23456789123457

    It may be preferable to enter high-precision decimal numbers
    as strings:

    >>> Float('1.23456789123456789')
    1.23456789123456789

    The desired number of digits can also be specified:

    >>> Float('1e-3', 3)
    0.00100
    >>> Float(100, 4)
    100.0

    Float can automatically count significant figures if a null string
    is sent for the precision; spaces or underscores are also allowed. (Auto-
    counting is only allowed for strings, ints and longs).

    >>> Float('123 456 789.123_456', '')
    123456789.123456
    >>> Float('12e-3', '')
    0.012
    >>> Float(3, '')
    3.

    If a number is written in scientific notation, only the digits before the
    exponent are considered significant if a decimal appears, otherwise the
    "e" signifies only how to move the decimal:

    >>> Float('60.e2', '')  # 2 digits significant
    6.0e+3
    >>> Float('60e2', '')  # 4 digits significant
    6000.
    >>> Float('600e-2', '')  # 3 digits significant
    6.00

    Notes
    =====

    Floats are inexact by their nature unless their value is a binary-exact
    value.

    >>> approx, exact = Float(.1, 1), Float(.125, 1)

    For calculation purposes, evalf needs to be able to change the precision
    but this will not increase the accuracy of the inexact value. The
    following is the most accurate 5-digit approximation of a value of 0.1
    that had only 1 digit of precision:

    >>> approx.evalf(5)
    0.099609

    By contrast, 0.125 is exact in binary (as it is in base 10) and so it
    can be passed to Float or evalf to obtain an arbitrary precision with
    matching accuracy:

    >>> Float(exact, 5)
    0.12500
    >>> exact.evalf(20)
    0.12500000000000000000

    Trying to make a high-precision Float from a float is not disallowed,
    but one must keep in mind that the *underlying float* (not the apparent
    decimal value) is being obtained with high precision. For example, 0.3
    does not have a finite binary representation. The closest rational is
    the fraction 5404319552844595/2**54. So if you try to obtain a Float of
    0.3 to 20 digits of precision you will not see the same thing as 0.3
    followed by 19 zeros:

    >>> Float(0.3, 20)
    0.29999999999999998890

    If you want a 20-digit value of the decimal 0.3 (not the floating point
    approximation of 0.3) you should send the 0.3 as a string. The underlying
    representation is still binary but a higher precision than Python's float
    is used:

    >>> Float('0.3', 20)
    0.30000000000000000000

    Although you can increase the precision of an existing Float using Float
    it will not increase the accuracy -- the underlying value is not changed:

    >>> def show(f): # binary rep of Float
    ...     from sympy import Mul, Pow
    ...     s, m, e, b = f._mpf_
    ...     v = Mul(int(m), Pow(2, int(e), evaluate=False), evaluate=False)
    ...     print('%s at prec=%s' % (v, f._prec))
    ...
    >>> t = Float('0.3', 3)
    >>> show(t)
    4915/2**14 at prec=13
    >>> show(Float(t, 20)) # higher prec, not higher accuracy
    4915/2**14 at prec=70
    >>> show(Float(t, 2)) # lower prec
    307/2**10 at prec=10

    The same thing happens when evalf is used on a Float:

    >>> show(t.evalf(20))
    4915/2**14 at prec=70
    >>> show(t.evalf(2))
    307/2**10 at prec=10

    Finally, Floats can be instantiated with an mpf tuple (n, c, p) to
    produce the number (-1)**n*c*2**p:

    >>> n, c, p = 1, 5, 0
    >>> (-1)**n*c*2**p
    -5
    >>> Float((1, 5, 0))
    -5.00000000000000

    An actual mpf tuple also contains the number of bits in c as the last
    element of the tuple:

    >>> _._mpf_
    (1, 5, 0, 3)

    This is not needed for instantiation and is not the same thing as the
    precision. The mpf tuple and the precision are two separate quantities
    that Float tracks.

    In SymPy, a Float is a number that can be computed with arbitrary
    precision. Although floating point 'inf' and 'nan' are not such
    numbers, Float can create these numbers:

    >>> Float('-inf')
    -oo
    >>> _.is_Float
    False

    Zero in Float only has a single value. Values are not separate for
    positive and negative zeroes.
    """
    __slots__ = ('_mpf_', '_prec')

    _mpf_: tuple[int, int, int, int]

    # A Float, though rational in form, does not behave like
    # a rational in all Python expressions so we deal with
    # exceptions (where we want to deal with the rational
    # form of the Float as a rational) at the source rather
    # than assigning a mathematically loaded category of 'rational'
    is_rational = None
    is_irrational = None
    is_number = True

    is_real = True
    is_extended_real = True

    is_Float = True

    _remove_non_digits = str.maketrans(dict.fromkeys("-+_."))

    def __new__(cls, num, dps=None, precision=None):
        if dps is not None and precision is not None:
            raise ValueError('Both decimal and binary precision supplied. '
                             'Supply only one. ')

        if isinstance(num, str):
            _num = num = num.strip()  # Python ignores leading and trailing space
            num = num.replace(' ', '_').lower()  # Float treats spaces as digit sep; E -> e
            if num.startswith('.') and len(num) > 1:
                num = '0' + num
            elif num.startswith('-.') and len(num) > 2:
                num = '-0.' + num[2:]
            elif num in ('inf', '+inf'):
                return S.Infinity
            elif num == '-inf':
                return S.NegativeInfinity
            elif num == 'nan':
                return S.NaN
            elif not _literal_float(num):
                raise ValueError('string-float not recognized: %s' % _num)
        elif isinstance(num, float) and num == 0:
            num = '0'
        elif isinstance(num, float) and num == float('inf'):
            return S.Infinity
        elif isinstance(num, float) and num == float('-inf'):
            return S.NegativeInfinity
        elif isinstance(num, float) and math.isnan(num):
            return S.NaN
        elif isinstance(num, (SYMPY_INTS, Integer)):
            num = str(num)
        elif num is S.Infinity:
            return num
        elif num is S.NegativeInfinity:
            return num
        elif num is S.NaN:
            return num
        elif _is_numpy_instance(num):  # support for numpy datatypes
            num = _convert_numpy_types(num)
        elif isinstance(num, mpmath.mpf):
            if precision is None:
                if dps is None:
                    precision = num.context.prec
            num = num._mpf_

        if dps is None and precision is None:
            dps = 15
            if isinstance(num, Float):
                return num
            if isinstance(num, str):
                try:
                    Num = decimal.Decimal(num)
                except decimal.InvalidOperation:
                    pass
                else:
                    isint = '.' not in num
                    num, dps = _decimal_to_Rational_prec(Num)
                    if num.is_Integer and isint:
                        # 12e3 is shorthand for int, not float;
                        # 12.e3 would be the float version
                        dps = max(dps, num_digits(num))
                    dps = max(15, dps)
                    precision = dps_to_prec(dps)
        elif precision == '' and dps is None or precision is None and dps == '':
            if not isinstance(num, str):
                raise ValueError('The null string can only be used when '
                'the number to Float is passed as a string or an integer.')
            try:
                Num = decimal.Decimal(num)
            except decimal.InvalidOperation:
                raise ValueError('string-float not recognized by Decimal: %s' % num)
            else:
                isint = '.' not in num
                num, dps = _decimal_to_Rational_prec(Num)
                if num.is_Integer and isint:
                    # without dec, e-notation is short for int
                    dps = max(dps, num_digits(num))
                    precision = dps_to_prec(dps)

        # decimal precision(dps) is set and maybe binary precision(precision)
        # as well.From here on binary precision is used to compute the Float.
        # Hence, if supplied use binary precision else translate from decimal
        # precision.

        if precision is None or precision == '':
            precision = dps_to_prec(dps)

        precision = int(precision)

        if isinstance(num, float):
            _mpf_ = mlib.from_float(num, precision, rnd)
        elif isinstance(num, str):
            _mpf_ = mlib.from_str(num, precision, rnd)
        elif isinstance(num, decimal.Decimal):
            if num.is_finite():
                _mpf_ = mlib.from_str(str(num), precision, rnd)
            elif num.is_nan():
                return S.NaN
            elif num.is_infinite():
                if num > 0:
                    return S.Infinity
                return S.NegativeInfinity
            else:
                raise ValueError("unexpected decimal value %s" % str(num))
        elif isinstance(num, tuple) and len(num) in (3, 4):
            if isinstance(num[1], str):
                # it's a hexadecimal (coming from a pickled object)
                num = list(num)
                # If we're loading an object pickled in Python 2 into
                # Python 3, we may need to strip a tailing 'L' because
                # of a shim for int on Python 3, see issue #13470.
                # Strip leading '0x' - gmpy2 only documents such inputs
                # with base prefix as valid when the 2nd argument (base) is 0.
                # When mpmath uses Sage as the backend, however, it
                # ends up including '0x' when preparing the picklable tuple.
                # See issue #19690.
                num[1] = num[1].removeprefix('0x').removesuffix('L')
                # Now we can assume that it is in standard form
                num[1] = MPZ(num[1], 16)
                _mpf_ = tuple(num)
            else:
                if len(num) == 4:
                    # handle normalization hack
                    return Float._new(num, precision)
                else:
                    if not all((
                            num[0] in (0, 1),
                            num[1] >= 0,
                            all(type(i) in (int, int) for i in num)
                            )):
                        raise ValueError('malformed mpf: %s' % (num,))
                    # don't compute number or else it may
                    # over/underflow
                    return Float._new(
                        (num[0], num[1], num[2], bitcount(num[1])),
                        precision)
        elif isinstance(num, (Number, NumberSymbol)):
            _mpf_ = num._as_mpf_val(precision)
        else:
            _mpf_ = mpmath.mpf(num, prec=precision)._mpf_

        return cls._new(_mpf_, precision, zero=False)

    @classmethod
    def _new(cls, _mpf_, _prec, zero=True):
        # special cases
        if zero and _mpf_ == fzero:
            return S.Zero  # Float(0) -> 0.0; Float._new((0,0,0,0)) -> 0
        elif _mpf_ == _mpf_nan:
            return S.NaN
        elif _mpf_ == _mpf_inf:
            return S.Infinity
        elif _mpf_ == _mpf_ninf:
            return S.NegativeInfinity

        obj = Expr.__new__(cls)
        obj._mpf_ = mpf_norm(_mpf_, _prec)
        obj._prec = _prec
        return obj

    def __getnewargs_ex__(self):
        sign, man, exp, bc = self._mpf_
        arg = (sign, hex(man)[2:], exp, bc)
        kwargs = {'precision': self._prec}
        return ((arg,), kwargs)

    def _hashable_content(self):
        return (self._mpf_, self._prec)

    def floor(self):
        return Integer(int(mlib.to_int(
            mlib.mpf_floor(self._mpf_, self._prec))))

    def ceiling(self):
        return Integer(int(mlib.to_int(
            mlib.mpf_ceil(self._mpf_, self._prec))))

    def __floor__(self):
        return self.floor()

    def __ceil__(self):
        return self.ceiling()

    @property
    def num(self):
        return mpmath.mpf(self._mpf_)

    def _as_mpf_val(self, prec):
        rv = mpf_norm(self._mpf_, prec)
        if rv != self._mpf_ and self._prec == prec:
            debug(self._mpf_, rv)
        return rv

    def _as_mpf_op(self, prec):
        return self._mpf_, max(prec, self._prec)

    def _eval_is_finite(self):
        if self._mpf_ in (_mpf_inf, _mpf_ninf):
            return False
        return True

    def _eval_is_infinite(self):
        if self._mpf_ in (_mpf_inf, _mpf_ninf):
            return True
        return False

    def _eval_is_integer(self):
        if self._mpf_ == fzero:
            return True
        if not int_valued(self):
            return False

    def _eval_is_negative(self):
        if self._mpf_ in (_mpf_ninf, _mpf_inf):
            return False
        return self.num < 0

    def _eval_is_positive(self):
        if self._mpf_ in (_mpf_ninf, _mpf_inf):
            return False
        return self.num > 0

    def _eval_is_extended_negative(self):
        if self._mpf_ == _mpf_ninf:
            return True
        if self._mpf_ == _mpf_inf:
            return False
        return self.num < 0

    def _eval_is_extended_positive(self):
        if self._mpf_ == _mpf_inf:
            return True
        if self._mpf_ == _mpf_ninf:
            return False
        return self.num > 0

    def _eval_is_zero(self):
        return self._mpf_ == fzero

    def __bool__(self):
        return self._mpf_ != fzero

    def __neg__(self):
        if not self:
            return self
        return Float._new(mlib.mpf_neg(self._mpf_), self._prec)

    @_sympifyit('other', NotImplemented)
    def __add__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            rhs, prec = other._as_mpf_op(self._prec)
            return Float._new(mlib.mpf_add(self._mpf_, rhs, prec, rnd), prec)
        return Number.__add__(self, other)

    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            rhs, prec = other._as_mpf_op(self._prec)
            return Float._new(mlib.mpf_sub(self._mpf_, rhs, prec, rnd), prec)
        return Number.__sub__(self, other)

    @_sympifyit('other', NotImplemented)
    def __mul__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            rhs, prec = other._as_mpf_op(self._prec)
            return Float._new(mlib.mpf_mul(self._mpf_, rhs, prec, rnd), prec)
        return Number.__mul__(self, other)

    @_sympifyit('other', NotImplemented)
    def __truediv__(self, other):
        if isinstance(other, Number) and other != 0 and global_parameters.evaluate:
            rhs, prec = other._as_mpf_op(self._prec)
