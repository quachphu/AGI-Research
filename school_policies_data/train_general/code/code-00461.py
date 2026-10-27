            return pcls
        return pcls.rewrite(sqrt, **kwargs)

    def _eval_rewrite_as_sec(self, arg, **kwargs):
        return 1/sec(arg)

    def _eval_rewrite_as_csc(self, arg, **kwargs):
        return 1/sec(arg).rewrite(csc, **kwargs)

    def _eval_rewrite_as_besselj(self, arg, **kwargs):
        from sympy.functions.special.bessel import besselj
        return Piecewise(
                (sqrt(pi*arg/2)*besselj(-S.Half, arg), Ne(arg, 0)),
                (1, True)
            )

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
        from sympy.functions.elementary.hyperbolic import cosh, sinh
        re, im = self._as_real_imag(deep=deep, **hints)
        return (cos(re)*cosh(im), -sin(re)*sinh(im))

    def _eval_expand_trig(self, **hints):
        from sympy.functions.special.polynomials import chebyshevt
        arg = self.args[0]
        x = None
        if arg.is_Add:  # TODO: Do this more efficiently for more than two terms
            x, y = arg.as_two_terms()
            sx = sin(x, evaluate=False)._eval_expand_trig()
            sy = sin(y, evaluate=False)._eval_expand_trig()
            cx = cos(x, evaluate=False)._eval_expand_trig()
            cy = cos(y, evaluate=False)._eval_expand_trig()
            return cx*cy - sx*sy
        elif arg.is_Mul:
            coeff, terms = arg.as_coeff_Mul(rational=True)
            if coeff.is_Integer:
                return chebyshevt(coeff, cos(terms))
        return cos(arg)

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.calculus.accumulationbounds import AccumBounds
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        n = (x0 + pi/2)/pi
        if n.is_integer:
            lt = (arg - n*pi + pi/2).as_leading_term(x)
            return (S.NegativeOne**n)*lt
        if x0 is S.ComplexInfinity:
            x0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if x0 in [S.Infinity, S.NegativeInfinity]:
            return AccumBounds(-1, 1)
        return self.func(x0) if x0.is_finite else self

    def _eval_is_extended_real(self):
        if self.args[0].is_extended_real:
            return True

    def _eval_is_finite(self):
        arg = self.args[0]

        if arg.is_extended_real:
            return True

    def _eval_is_complex(self):
        if self.args[0].is_extended_real \
            or self.args[0].is_complex:
            return True

    def _eval_is_zero(self):
        rest, pi_mult = _peeloff_pi(self.args[0])
        if rest.is_zero and pi_mult:
            return (pi_mult - S.Half).is_integer


class tan(TrigonometricFunction):
    """
    The tangent function.

    Returns the tangent of x (measured in radians).

    Explanation
    ===========

    See :class:`sin` for notes about automatic evaluation.

    Examples
    ========

    >>> from sympy import tan, pi
    >>> from sympy.abc import x
    >>> tan(x**2).diff(x)
    2*x*(tan(x**2)**2 + 1)
    >>> tan(1).diff(x)
    0
    >>> tan(pi/8).expand()
    -1 + sqrt(2)

    See Also
    ========

    sin, csc, cos, sec, cot
    asin, acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.14
    .. [3] https://functions.wolfram.com/ElementaryFunctions/Tan

    """

    def period(self, symbol=None):
        return self._period(pi, symbol)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return S.One + self**2
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return atan

    @classmethod
    def eval(cls, arg):
        from sympy.calculus.accumulationbounds import AccumBounds
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg.is_zero:
                return S.Zero
            elif arg in (S.Infinity, S.NegativeInfinity):
                return AccumBounds(S.NegativeInfinity, S.Infinity)

        if arg is S.ComplexInfinity:
            return S.NaN

        if isinstance(arg, AccumBounds):
            min, max = arg.min, arg.max
            d = floor(min/pi)
            if min is not S.NegativeInfinity:
                min = min - d*pi
            if max is not S.Infinity:
                max = max - d*pi
            from sympy.sets.sets import FiniteSet
            if AccumBounds(min, max).intersection(FiniteSet(pi/2, pi*Rational(3, 2))):
                return AccumBounds(S.NegativeInfinity, S.Infinity)
            else:
                return AccumBounds(tan(min), tan(max))

        if arg.could_extract_minus_sign():
            return -cls(-arg)

        i_coeff = _imaginary_unit_as_coefficient(arg)
        if i_coeff is not None:
            from sympy.functions.elementary.hyperbolic import tanh
            return S.ImaginaryUnit*tanh(i_coeff)

        pi_coeff = _pi_coeff(arg, 2)
        if pi_coeff is not None:
            if pi_coeff.is_integer:
                return S.Zero

            if not pi_coeff.is_Rational:
                narg = pi_coeff*pi
                if narg != arg:
                    return cls(narg)
                return None

            if pi_coeff.is_Rational:
                q = pi_coeff.q
                p = pi_coeff.p % q
                # ensure simplified results are returned for n*pi/5, n*pi/10
                table10 = {
                    1: sqrt(1 - 2*sqrt(5)/5),
                    2: sqrt(5 - 2*sqrt(5)),
                    3: sqrt(1 + 2*sqrt(5)/5),
                    4: sqrt(5 + 2*sqrt(5))
                    }
                if q in (5, 10):
                    n = 10*p/q
                    if n > 5:
                        n = 10 - n
                        return -table10[n]
                    else:
                        return table10[n]
                if not pi_coeff.q % 2:
                    narg = pi_coeff*pi*2
                    cresult, sresult = cos(narg), cos(narg - pi/2)
                    if not isinstance(cresult, cos) \
                            and not isinstance(sresult, cos):
                        if sresult == 0:
                            return S.ComplexInfinity
                        return 1/sresult - cresult/sresult

                table2 = _table2()
                if q in table2:
                    a, b = table2[q]
                    nvala, nvalb = cls(p*pi/a), cls(p*pi/b)
                    if None in (nvala, nvalb):
                        return None
                    return (nvala - nvalb)/(1 + nvala*nvalb)
                narg = ((pi_coeff + S.Half) % 1 - S.Half)*pi
                # see cos() to specify which expressions should  be
                # expanded automatically in terms of radicals
                cresult, sresult = cos(narg), cos(narg - pi/2)
                if not isinstance(cresult, cos) \
                        and not isinstance(sresult, cos):
                    if cresult == 0:
                        return S.ComplexInfinity
                    return (sresult/cresult)
                if narg != arg:
                    return cls(narg)

        if arg.is_Add:
            x, m = _peeloff_pi(arg)
            if m:
                tanm = tan(m*pi)
                if tanm is S.ComplexInfinity:
                    return -cot(x)
                else: # tanm == 0
                    return tan(x)

        if arg.is_zero:
            return S.Zero

        if isinstance(arg, atan):
            return arg.args[0]

        if isinstance(arg, atan2):
            y, x = arg.args
            return y/x

        if isinstance(arg, asin):
            x = arg.args[0]
            return x/sqrt(1 - x**2)

        if isinstance(arg, acos):
            x = arg.args[0]
            return sqrt(1 - x**2)/x

        if isinstance(arg, acot):
            x = arg.args[0]
            return 1/x

        if isinstance(arg, acsc):
            x = arg.args[0]
            return 1/(sqrt(1 - 1/x**2)*x)

        if isinstance(arg, asec):
            x = arg.args[0]
            return sqrt(1 - 1/x**2)*x

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)

            a, b = ((n - 1)//2), 2**(n + 1)

            B = bernoulli(n + 1)
            F = factorial(n + 1)

            return S.NegativeOne**a*b*(b - 1)*B/F*x**n

    def _eval_nseries(self, x, n, logx, cdir=0):
        i = self.args[0].limit(x, 0)*2/pi
        if i and i.is_Integer:
            return self.rewrite(cos)._eval_nseries(x, n=n, logx=logx)
        return super()._eval_nseries(x, n=n, logx=logx)

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        if isinstance(arg, log):
            I = S.ImaginaryUnit
            x = arg.args[0]
            return I*(x**-I - x**I)/(x**-I + x**I)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
        re, im = self._as_real_imag(deep=deep, **hints)
        if im:
            from sympy.functions.elementary.hyperbolic import cosh, sinh
            denom = cos(2*re) + cosh(2*im)
            return (sin(2*re)/denom, sinh(2*im)/denom)
        else:
            return (self.func(re), S.Zero)

    def _eval_expand_trig(self, **hints):
        arg = self.args[0]
        x = None
        if arg.is_Add:
            n = len(arg.args)
            TX = []
            for x in arg.args:
                tx = tan(x, evaluate=False)._eval_expand_trig()
                TX.append(tx)

            Yg = numbered_symbols('Y')
            Y = [ next(Yg) for i in range(n) ]

            p = [0, 0]
            for i in range(n + 1):
                p[1 - i % 2] += symmetric_poly(i, Y)*(-1)**((i % 4)//2)
            return (p[0]/p[1]).subs(list(zip(Y, TX)))

        elif arg.is_Mul:
            coeff, terms = arg.as_coeff_Mul(rational=True)
            if coeff.is_Integer and coeff > 1:
                I = S.ImaginaryUnit
                z = Symbol('dummy', real=True)
                P = ((1 + I*z)**coeff).expand()
                return (im(P)/re(P)).subs([(z, tan(terms))])
        return tan(arg)

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        I = S.ImaginaryUnit
        from sympy.functions.elementary.hyperbolic import HyperbolicFunction
        if isinstance(arg, (TrigonometricFunction, HyperbolicFunction)):
            arg = arg.func(arg.args[0]).rewrite(exp)
        neg_exp, pos_exp = exp(-arg*I), exp(arg*I)
        return I*(neg_exp - pos_exp)/(neg_exp + pos_exp)

    def _eval_rewrite_as_sin(self, x, **kwargs):
        return 2*sin(x)**2/sin(2*x)

    def _eval_rewrite_as_cos(self, x, **kwargs):
        return cos(x - pi/2, evaluate=False)/cos(x)

    def _eval_rewrite_as_sincos(self, arg, **kwargs):
        return sin(arg)/cos(arg)

    def _eval_rewrite_as_cot(self, arg, **kwargs):
        return 1/cot(arg)

    def _eval_rewrite_as_sec(self, arg, **kwargs):
        sin_in_sec_form = sin(arg).rewrite(sec, **kwargs)
        cos_in_sec_form = cos(arg).rewrite(sec, **kwargs)
        return sin_in_sec_form/cos_in_sec_form

    def _eval_rewrite_as_csc(self, arg, **kwargs):
        sin_in_csc_form = sin(arg).rewrite(csc, **kwargs)
        cos_in_csc_form = cos(arg).rewrite(csc, **kwargs)
        return sin_in_csc_form/cos_in_csc_form

    def _eval_rewrite_as_pow(self, arg, **kwargs):
        y = self.rewrite(cos, **kwargs).rewrite(pow, **kwargs)
        if y.has(cos):
            return None
        return y

    def _eval_rewrite_as_sqrt(self, arg, **kwargs):
        y = self.rewrite(cos, **kwargs).rewrite(sqrt, **kwargs)
        if y.has(cos):
            return None
        return y

    def _eval_rewrite_as_besselj(self, arg, **kwargs):
        from sympy.functions.special.bessel import besselj
        return besselj(S.Half, arg)/besselj(-S.Half, arg)

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.calculus.accumulationbounds import AccumBounds
        from sympy.functions.elementary.complexes import re
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        n = 2*x0/pi
        if n.is_integer:
            lt = (arg - n*pi/2).as_leading_term(x)
            return lt if n.is_even else -1/lt
        if x0 is S.ComplexInfinity:
            x0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if x0 in (S.Infinity, S.NegativeInfinity):
            return AccumBounds(S.NegativeInfinity, S.Infinity)
        return self.func(x0) if x0.is_finite else self

    def _eval_is_extended_real(self):
        # FIXME: currently tan(pi/2) return zoo
        return self.args[0].is_extended_real

    def _eval_is_real(self):
        arg = self.args[0]
        if arg.is_real and (arg/pi - S.Half).is_integer is False:
            return True

    def _eval_is_finite(self):
        arg = self.args[0]

        if arg.is_real and (arg/pi - S.Half).is_integer is False:
            return True

        if arg.is_imaginary:
            return True

    def _eval_is_zero(self):
        rest, pi_mult = _peeloff_pi(self.args[0])
        if rest.is_zero:
            return pi_mult.is_integer

    def _eval_is_complex(self):
        arg = self.args[0]

        if arg.is_real and (arg/pi - S.Half).is_integer is False:
            return True


class cot(TrigonometricFunction):
    """
    The cotangent function.

    Returns the cotangent of x (measured in radians).

    Explanation
    ===========

    See :class:`sin` for notes about automatic evaluation.

    Examples
    ========

    >>> from sympy import cot, pi
    >>> from sympy.abc import x
    >>> cot(x**2).diff(x)
    2*x*(-cot(x**2)**2 - 1)
    >>> cot(1).diff(x)
    0
    >>> cot(pi/12)
    sqrt(3) + 2

    See Also
    ========

    sin, csc, cos, sec, tan
    asin, acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.14
    .. [3] https://functions.wolfram.com/ElementaryFunctions/Cot

    """

    def period(self, symbol=None):
        return self._period(pi, symbol)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return S.NegativeOne - self**2
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return acot

    @classmethod
    def eval(cls, arg):
        from sympy.calculus.accumulationbounds import AccumBounds
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            if arg.is_zero:
                return S.ComplexInfinity
            elif arg in (S.Infinity, S.NegativeInfinity):
                return AccumBounds(S.NegativeInfinity, S.Infinity)

        if arg is S.ComplexInfinity:
            return S.NaN

        if isinstance(arg, AccumBounds):
            return -tan(arg + pi/2)

        if arg.could_extract_minus_sign():
            return -cls(-arg)

        i_coeff = _imaginary_unit_as_coefficient(arg)
        if i_coeff is not None:
            from sympy.functions.elementary.hyperbolic import coth
            return -S.ImaginaryUnit*coth(i_coeff)

        pi_coeff = _pi_coeff(arg, 2)
        if pi_coeff is not None:
            if pi_coeff.is_integer:
                return S.ComplexInfinity

            if not pi_coeff.is_Rational:
                narg = pi_coeff*pi
                if narg != arg:
                    return cls(narg)
                return None

            if pi_coeff.is_Rational:
                if pi_coeff.q in (5, 10):
                    return tan(pi/2 - arg)
                if pi_coeff.q > 2 and not pi_coeff.q % 2:
                    narg = pi_coeff*pi*2
                    cresult, sresult = cos(narg), cos(narg - pi/2)
                    if not isinstance(cresult, cos) \
                            and not isinstance(sresult, cos):
                        return 1/sresult + cresult/sresult
                q = pi_coeff.q
                p = pi_coeff.p % q
                table2 = _table2()
                if q in table2:
                    a, b = table2[q]
                    nvala, nvalb = cls(p*pi/a), cls(p*pi/b)
                    if None in (nvala, nvalb):
                        return None
                    return (1 + nvala*nvalb)/(nvalb - nvala)
                narg = (((pi_coeff + S.Half) % 1) - S.Half)*pi
                # see cos() to specify which expressions should be
                # expanded automatically in terms of radicals
                cresult, sresult = cos(narg), cos(narg - pi/2)
                if not isinstance(cresult, cos) \
                        and not isinstance(sresult, cos):
                    if sresult == 0:
                        return S.ComplexInfinity
                    return cresult/sresult
                if narg != arg:
                    return cls(narg)

        if arg.is_Add:
            x, m = _peeloff_pi(arg)
            if m:
                cotm = cot(m*pi)
                if cotm is S.ComplexInfinity:
                    return cot(x)
                else: # cotm == 0
                    return -tan(x)

        if arg.is_zero:
            return S.ComplexInfinity

        if isinstance(arg, acot):
            return arg.args[0]

        if isinstance(arg, atan):
            x = arg.args[0]
            return 1/x

        if isinstance(arg, atan2):
            y, x = arg.args
            return x/y

        if isinstance(arg, asin):
            x = arg.args[0]
            return sqrt(1 - x**2)/x

        if isinstance(arg, acos):
            x = arg.args[0]
            return x/sqrt(1 - x**2)

        if isinstance(arg, acsc):
            x = arg.args[0]
            return sqrt(1 - 1/x**2)*x

        if isinstance(arg, asec):
            x = arg.args[0]
            return 1/(sqrt(1 - 1/x**2)*x)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return 1/sympify(x)
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)

            B = bernoulli(n + 1)
            F = factorial(n + 1)

            return S.NegativeOne**((n + 1)//2)*2**(n + 1)*B/F*x**n

    def _eval_nseries(self, x, n, logx, cdir=0):
        i = self.args[0].limit(x, 0)/pi
        if i and i.is_Integer:
            return self.rewrite(cos)._eval_nseries(x, n=n, logx=logx)
        return self.rewrite(tan)._eval_nseries(x, n=n, logx=logx)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
        re, im = self._as_real_imag(deep=deep, **hints)
        if im:
            from sympy.functions.elementary.hyperbolic import cosh, sinh
            denom = cos(2*re) - cosh(2*im)
            return (-sin(2*re)/denom, sinh(2*im)/denom)
        else:
            return (self.func(re), S.Zero)

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        from sympy.functions.elementary.hyperbolic import HyperbolicFunction
        I = S.ImaginaryUnit
        if isinstance(arg, (TrigonometricFunction, HyperbolicFunction)):
            arg = arg.func(arg.args[0]).rewrite(exp, **kwargs)
        neg_exp, pos_exp = exp(-arg*I), exp(arg*I)
        return I*(pos_exp + neg_exp)/(pos_exp - neg_exp)

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        if isinstance(arg, log):
            I = S.ImaginaryUnit
            x = arg.args[0]
            return -I*(x**-I + x**I)/(x**-I - x**I)

    def _eval_rewrite_as_sin(self, x, **kwargs):
        return sin(2*x)/(2*(sin(x)**2))

    def _eval_rewrite_as_cos(self, x, **kwargs):
        return cos(x)/cos(x - pi/2, evaluate=False)

    def _eval_rewrite_as_sincos(self, arg, **kwargs):
        return cos(arg)/sin(arg)

    def _eval_rewrite_as_tan(self, arg, **kwargs):
        return 1/tan(arg)

    def _eval_rewrite_as_sec(self, arg, **kwargs):
        cos_in_sec_form = cos(arg).rewrite(sec, **kwargs)
        sin_in_sec_form = sin(arg).rewrite(sec, **kwargs)
        return cos_in_sec_form/sin_in_sec_form

    def _eval_rewrite_as_csc(self, arg, **kwargs):
        cos_in_csc_form = cos(arg).rewrite(csc, **kwargs)
        sin_in_csc_form = sin(arg).rewrite(csc, **kwargs)
        return cos_in_csc_form/sin_in_csc_form

    def _eval_rewrite_as_pow(self, arg, **kwargs):
        y = self.rewrite(cos, **kwargs).rewrite(pow, **kwargs)
        if y.has(cos):
            return None
        return y

    def _eval_rewrite_as_sqrt(self, arg, **kwargs):
        y = self.rewrite(cos, **kwargs).rewrite(sqrt, **kwargs)
        if y.has(cos):
            return None
        return y

    def _eval_rewrite_as_besselj(self, arg, **kwargs):
        from sympy.functions.special.bessel import besselj
        return besselj(-S.Half, arg)/besselj(S.Half, arg)

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.calculus.accumulationbounds import AccumBounds
        from sympy.functions.elementary.complexes import re
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        n = 2*x0/pi
        if n.is_integer:
            lt = (arg - n*pi/2).as_leading_term(x)
            return 1/lt if n.is_even else -lt
        if x0 is S.ComplexInfinity:
            x0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if x0 in (S.Infinity, S.NegativeInfinity):
            return AccumBounds(S.NegativeInfinity, S.Infinity)
        return self.func(x0) if x0.is_finite else self

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real

    def _eval_expand_trig(self, **hints):
        arg = self.args[0]
        x = None
        if arg.is_Add:
            n = len(arg.args)
            CX = []
            for x in arg.args:
                cx = cot(x, evaluate=False)._eval_expand_trig()
                CX.append(cx)

            Yg = numbered_symbols('Y')
            Y = [ next(Yg) for i in range(n) ]

            p = [0, 0]
            for i in range(n, -1, -1):
                p[(n - i) % 2] += symmetric_poly(i, Y)*(-1)**(((n - i) % 4)//2)
            return (p[0]/p[1]).subs(list(zip(Y, CX)))
        elif arg.is_Mul:
            coeff, terms = arg.as_coeff_Mul(rational=True)
            if coeff.is_Integer and coeff > 1:
                I = S.ImaginaryUnit
                z = Symbol('dummy', real=True)
                P = ((z + I)**coeff).expand()
                return (re(P)/im(P)).subs([(z, cot(terms))])
        return cot(arg)  # XXX sec and csc return 1/cos and 1/sin

    def _eval_is_finite(self):
        arg = self.args[0]
        if arg.is_real and (arg/pi).is_integer is False:
            return True
        if arg.is_imaginary:
            return True

    def _eval_is_real(self):
        arg = self.args[0]
        if arg.is_real and (arg/pi).is_integer is False:
            return True

    def _eval_is_complex(self):
        arg = self.args[0]
        if arg.is_real and (arg/pi).is_integer is False:
            return True

    def _eval_is_zero(self):
        rest, pimult = _peeloff_pi(self.args[0])
        if pimult and rest.is_zero:
            return (pimult - S.Half).is_integer

    def _eval_subs(self, old, new):
        arg = self.args[0]
        argnew = arg.subs(old, new)
        if arg != argnew and (argnew/pi).is_integer:
            return S.ComplexInfinity
        return cot(argnew)


class ReciprocalTrigonometricFunction(TrigonometricFunction):
    """Base class for reciprocal functions of trigonometric functions. """

    _reciprocal_of = None       # mandatory, to be defined in subclass
    _singularities = (S.ComplexInfinity,)

    # _is_even and _is_odd are used for correct evaluation of csc(-x), sec(-x)
    # TODO refactor into TrigonometricFunction common parts of
    # trigonometric functions eval() like even/odd, func(x+2*k*pi), etc.

    # optional, to be defined in subclasses:
    _is_even: FuzzyBool = None
    _is_odd: FuzzyBool = None

    @classmethod
    def eval(cls, arg):
        if arg.could_extract_minus_sign():
            if cls._is_even:
                return cls(-arg)
            if cls._is_odd:
                return -cls(-arg)

        pi_coeff = _pi_coeff(arg)
        if (pi_coeff is not None
            and not (2*pi_coeff).is_integer
            and pi_coeff.is_Rational):
                q = pi_coeff.q
                p = pi_coeff.p % (2*q)
                if p > q:
                    narg = (pi_coeff - 1)*pi
                    return -cls(narg)
                if 2*p > q:
                    narg = (1 - pi_coeff)*pi
                    if cls._is_odd:
                        return cls(narg)
                    elif cls._is_even:
                        return -cls(narg)

        if hasattr(arg, 'inverse') and arg.inverse() == cls:
            return arg.args[0]

        t = cls._reciprocal_of.eval(arg)
        if t is None:
            return t
        elif any(isinstance(i, cos) for i in (t, -t)):
            return (1/t).rewrite(sec)
        elif any(isinstance(i, sin) for i in (t, -t)):
            return (1/t).rewrite(csc)
        else:
            return 1/t

    def _call_reciprocal(self, method_name, *args, **kwargs):
        # Calls method_name on _reciprocal_of
        o = self._reciprocal_of(self.args[0])
        return getattr(o, method_name)(*args, **kwargs)

    def _calculate_reciprocal(self, method_name, *args, **kwargs):
        # If calling method_name on _reciprocal_of returns a value != None
        # then return the reciprocal of that value
        t = self._call_reciprocal(method_name, *args, **kwargs)
        return 1/t if t is not None else t

    def _rewrite_reciprocal(self, method_name, arg):
        # Special handling for rewrite functions. If reciprocal rewrite returns
        # unmodified expression, then return None
        t = self._call_reciprocal(method_name, arg)
        if t is not None and t != self._reciprocal_of(arg):
            return 1/t

    def _period(self, symbol):
        f = expand_mul(self.args[0])
        return self._reciprocal_of(f).period(symbol)

    def fdiff(self, argindex=1):
        return -self._calculate_reciprocal("fdiff", argindex)/self**2

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_exp", arg)

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_Pow", arg)

    def _eval_rewrite_as_sin(self, arg, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_sin", arg)

    def _eval_rewrite_as_cos(self, arg, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_cos", arg)

    def _eval_rewrite_as_tan(self, arg, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_tan", arg)

    def _eval_rewrite_as_pow(self, arg, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_pow", arg)

    def _eval_rewrite_as_sqrt(self, arg, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_sqrt", arg)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
        return (1/self._reciprocal_of(self.args[0])).as_real_imag(deep,
                                                                  **hints)

    def _eval_expand_trig(self, **hints):
        return self._calculate_reciprocal("_eval_expand_trig", **hints)

    def _eval_is_extended_real(self):
        return self._reciprocal_of(self.args[0])._eval_is_extended_real()

    def _eval_as_leading_term(self, x, logx, cdir):
        return (1/self._reciprocal_of(self.args[0]))._eval_as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_is_finite(self):
        return (1/self._reciprocal_of(self.args[0])).is_finite

    def _eval_nseries(self, x, n, logx, cdir=0):
        return (1/self._reciprocal_of(self.args[0]))._eval_nseries(x, n, logx)


class sec(ReciprocalTrigonometricFunction):
    """
    The secant function.

    Returns the secant of x (measured in radians).

    Explanation
    ===========

    See :class:`sin` for notes about automatic evaluation.

    Examples
    ========

    >>> from sympy import sec
    >>> from sympy.abc import x
    >>> sec(x**2).diff(x)
    2*x*tan(x**2)*sec(x**2)
    >>> sec(1).diff(x)
    0

    See Also
    ========

    sin, csc, cos, tan, cot
    asin, acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.14
    .. [3] https://functions.wolfram.com/ElementaryFunctions/Sec

    """

    _reciprocal_of = cos
    _is_even = True

    def period(self, symbol=None):
        return self._period(symbol)

    def _eval_rewrite_as_cot(self, arg, **kwargs):
        cot_half_sq = cot(arg/2)**2
        return (cot_half_sq + 1)/(cot_half_sq - 1)

    def _eval_rewrite_as_cos(self, arg, **kwargs):
        return (1/cos(arg))

    def _eval_rewrite_as_sincos(self, arg, **kwargs):
        return sin(arg)/(cos(arg)*sin(arg))

    def _eval_rewrite_as_sin(self, arg, **kwargs):
        return (1/cos(arg).rewrite(sin, **kwargs))

    def _eval_rewrite_as_tan(self, arg, **kwargs):
        return (1/cos(arg).rewrite(tan, **kwargs))

    def _eval_rewrite_as_csc(self, arg, **kwargs):
        return csc(pi/2 - arg, evaluate=False)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return tan(self.args[0])*sec(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_besselj(self, arg, **kwargs):
        from sympy.functions.special.bessel import besselj
        return Piecewise(
                (1/(sqrt(pi*arg)/(sqrt(2))*besselj(-S.Half, arg)), Ne(arg, 0)),
                (1, True)
            )

    def _eval_is_complex(self):
        arg = self.args[0]

        if arg.is_complex and (arg/pi - S.Half).is_integer is False:
            return True

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        # Reference Formula:
        # https://functions.wolfram.com/ElementaryFunctions/Sec/06/01/02/01/
        if n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)
            k = n//2
            return S.NegativeOne**k*euler(2*k)/factorial(2*k)*x**(2*k)

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.calculus.accumulationbounds import AccumBounds
        from sympy.functions.elementary.complexes import re
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        n = (x0 + pi/2)/pi
        if n.is_integer:
            lt = (arg - n*pi + pi/2).as_leading_term(x)
            return (S.NegativeOne**n)/lt
        if x0 is S.ComplexInfinity:
            x0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if x0 in (S.Infinity, S.NegativeInfinity):
            return AccumBounds(S.NegativeInfinity, S.Infinity)
        return self.func(x0) if x0.is_finite else self


class csc(ReciprocalTrigonometricFunction):
    """
    The cosecant function.

    Returns the cosecant of x (measured in radians).

    Explanation
    ===========

    See :func:`sin` for notes about automatic evaluation.

    Examples
    ========

    >>> from sympy import csc
    >>> from sympy.abc import x
    >>> csc(x**2).diff(x)
    -2*x*cot(x**2)*csc(x**2)
    >>> csc(1).diff(x)
    0

    See Also
    ========

    sin, cos, sec, tan, cot
    asin, acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.14
    .. [3] https://functions.wolfram.com/ElementaryFunctions/Csc

    """

    _reciprocal_of = sin
    _is_odd = True

    def period(self, symbol=None):
        return self._period(symbol)

    def _eval_rewrite_as_sin(self, arg, **kwargs):
        return (1/sin(arg))

    def _eval_rewrite_as_sincos(self, arg, **kwargs):
        return cos(arg)/(sin(arg)*cos(arg))

    def _eval_rewrite_as_cot(self, arg, **kwargs):
        cot_half = cot(arg/2)
        return (1 + cot_half**2)/(2*cot_half)

    def _eval_rewrite_as_cos(self, arg, **kwargs):
        return 1/sin(arg).rewrite(cos, **kwargs)

    def _eval_rewrite_as_sec(self, arg, **kwargs):
        return sec(pi/2 - arg, evaluate=False)

    def _eval_rewrite_as_tan(self, arg, **kwargs):
        return (1/sin(arg).rewrite(tan, **kwargs))

    def _eval_rewrite_as_besselj(self, arg, **kwargs):
        from sympy.functions.special.bessel import besselj
        return sqrt(2/pi)*(1/(sqrt(arg)*besselj(S.Half, arg)))

    def fdiff(self, argindex=1):
        if argindex == 1:
            return -cot(self.args[0])*csc(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_is_complex(self):
        arg = self.args[0]
        if arg.is_real and (arg/pi).is_integer is False:
            return True

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return 1/sympify(x)
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            k = n//2 + 1
            return (S.NegativeOne**(k - 1)*2*(2**(2*k - 1) - 1)*
                    bernoulli(2*k)*x**(2*k - 1)/factorial(2*k))

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.calculus.accumulationbounds import AccumBounds
        from sympy.functions.elementary.complexes import re
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        n = x0/pi
        if n.is_integer:
            lt = (arg - n*pi).as_leading_term(x)
            return (S.NegativeOne**n)/lt
        if x0 is S.ComplexInfinity:
            x0 = arg.limit(x, 0, dir='-' if re(cdir).is_negative else '+')
        if x0 in (S.Infinity, S.NegativeInfinity):
            return AccumBounds(S.NegativeInfinity, S.Infinity)
        return self.func(x0) if x0.is_finite else self


class sinc(DefinedFunction):
    r"""
    Represents an unnormalized sinc function:

    .. math::

        \operatorname{sinc}(x) =
        \begin{cases}
          \frac{\sin x}{x} & \qquad x \neq 0 \\
          1 & \qquad x = 0
        \end{cases}

    Examples
    ========

    >>> from sympy import sinc, oo, jn
    >>> from sympy.abc import x
    >>> sinc(x)
    sinc(x)

    * Automated Evaluation

    >>> sinc(0)
    1
    >>> sinc(oo)
    0

    * Differentiation

    >>> sinc(x).diff()
    cos(x)/x - sin(x)/x**2

    * Series Expansion

    >>> sinc(x).series()
    1 - x**2/6 + x**4/120 + O(x**6)

    * As zero'th order spherical Bessel Function

    >>> sinc(x).rewrite(jn)
    jn(0, x)

    See also
    ========

    sin

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Sinc_function

    """
    _singularities = (S.ComplexInfinity,)

    def fdiff(self, argindex=1):
        x = self.args[0]
        if argindex == 1:
            # We would like to return the Piecewise here, but Piecewise.diff
            # currently can't handle removable singularities, meaning things
            # like sinc(x).diff(x, 2) give the wrong answer at x = 0. See
            # https://github.com/sympy/sympy/issues/11402.
            #
            # return Piecewise(((x*cos(x) - sin(x))/x**2, Ne(x, S.Zero)), (S.Zero, S.true))
            return cos(x)/x - sin(x)/x**2
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        if arg.is_zero:
            return S.One
        if arg.is_Number:
            if arg in [S.Infinity, S.NegativeInfinity]:
                return S.Zero
            elif arg is S.NaN:
                return S.NaN

        if arg is S.ComplexInfinity:
            return S.NaN

        if arg.could_extract_minus_sign():
            return cls(-arg)

        pi_coeff = _pi_coeff(arg)
        if pi_coeff is not None:
            if pi_coeff.is_integer:
                if fuzzy_not(arg.is_zero):
                    return S.Zero
            elif (2*pi_coeff).is_integer:
                return S.NegativeOne**(pi_coeff - S.Half)/arg

    def _eval_nseries(self, x, n, logx, cdir=0):
        x = self.args[0]
        return (sin(x)/x)._eval_nseries(x, n, logx)

    def _eval_rewrite_as_jn(self, arg, **kwargs):
        from sympy.functions.special.bessel import jn
        return jn(0, arg)

    def _eval_rewrite_as_sin(self, arg, **kwargs):
        return Piecewise((sin(arg)/arg, Ne(arg, S.Zero)), (S.One, S.true))

    def _eval_is_zero(self):
        if self.args[0].is_infinite:
            return True
        rest, pi_mult = _peeloff_pi(self.args[0])
        if rest.is_zero:
            return fuzzy_and([pi_mult.is_integer, pi_mult.is_nonzero])
        if rest.is_Number and pi_mult.is_integer:
            return False

    def _eval_is_real(self):
        if self.args[0].is_extended_real or self.args[0].is_imaginary:
            return True

    _eval_is_finite = _eval_is_real


###############################################################################
########################### TRIGONOMETRIC INVERSES ############################
###############################################################################


class InverseTrigonometricFunction(DefinedFunction):
    """Base class for inverse trigonometric functions."""
    _singularities: tuple[Expr, ...] = (S.One, S.NegativeOne, S.Zero, S.ComplexInfinity)

    @staticmethod
    @cacheit
    def _asin_table():
        # Only keys with could_extract_minus_sign() == False
        # are actually needed.
        return {
            sqrt(3)/2: pi/3,
            sqrt(2)/2: pi/4,
            1/sqrt(2): pi/4,
            sqrt((5 - sqrt(5))/8): pi/5,
            sqrt(2)*sqrt(5 - sqrt(5))/4: pi/5,
            sqrt((5 + sqrt(5))/8): pi*Rational(2, 5),
            sqrt(2)*sqrt(5 + sqrt(5))/4: pi*Rational(2, 5),
            S.Half: pi/6,
            sqrt(2 - sqrt(2))/2: pi/8,
            sqrt(S.Half - sqrt(2)/4): pi/8,
            sqrt(2 + sqrt(2))/2: pi*Rational(3, 8),
            sqrt(S.Half + sqrt(2)/4): pi*Rational(3, 8),
            (sqrt(5) - 1)/4: pi/10,
            (1 - sqrt(5))/4: -pi/10,
            (sqrt(5) + 1)/4: pi*Rational(3, 10),
            sqrt(6)/4 - sqrt(2)/4: pi/12,
            -sqrt(6)/4 + sqrt(2)/4: -pi/12,
            (sqrt(3) - 1)/sqrt(8): pi/12,
            (1 - sqrt(3))/sqrt(8): -pi/12,
            sqrt(6)/4 + sqrt(2)/4: pi*Rational(5, 12),
            (1 + sqrt(3))/sqrt(8): pi*Rational(5, 12)
        }


    @staticmethod
    @cacheit
    def _atan_table():
        # Only keys with could_extract_minus_sign() == False
        # are actually needed.
        return {
            sqrt(3)/3: pi/6,
            1/sqrt(3): pi/6,
            sqrt(3): pi/3,
            sqrt(2) - 1: pi/8,
            1 - sqrt(2): -pi/8,
            1 + sqrt(2): pi*Rational(3, 8),
            sqrt(5 - 2*sqrt(5)): pi/5,
            sqrt(5 + 2*sqrt(5)): pi*Rational(2, 5),
            sqrt(1 - 2*sqrt(5)/5): pi/10,
            sqrt(1 + 2*sqrt(5)/5): pi*Rational(3, 10),
            2 - sqrt(3): pi/12,
            -2 + sqrt(3): -pi/12,
            2 + sqrt(3): pi*Rational(5, 12)
        }

    @staticmethod
    @cacheit
    def _acsc_table():
        # Keys for which could_extract_minus_sign()
        # will obviously return True are omitted.
        return {
            2*sqrt(3)/3: pi/3,
            sqrt(2): pi/4,
            sqrt(2 + 2*sqrt(5)/5): pi/5,
            1/sqrt(Rational(5, 8) - sqrt(5)/8): pi/5,
            sqrt(2 - 2*sqrt(5)/5): pi*Rational(2, 5),
            1/sqrt(Rational(5, 8) + sqrt(5)/8): pi*Rational(2, 5),
            2: pi/6,
            sqrt(4 + 2*sqrt(2)): pi/8,
            2/sqrt(2 - sqrt(2)): pi/8,
            sqrt(4 - 2*sqrt(2)): pi*Rational(3, 8),
            2/sqrt(2 + sqrt(2)): pi*Rational(3, 8),
            1 + sqrt(5): pi/10,
            sqrt(5) - 1: pi*Rational(3, 10),
            -(sqrt(5) - 1): pi*Rational(-3, 10),
            sqrt(6) + sqrt(2): pi/12,
            sqrt(6) - sqrt(2): pi*Rational(5, 12),
            -(sqrt(6) - sqrt(2)): pi*Rational(-5, 12)
        }


class asin(InverseTrigonometricFunction):
    r"""
    The inverse sine function.

    Returns the arcsine of x in radians.

    Explanation
    ===========

    ``asin(x)`` will evaluate automatically in the cases
    $x \in \{\infty, -\infty, 0, 1, -1\}$ and for some instances when the
    result is a rational multiple of $\pi$ (see the ``eval`` class method).

    A purely imaginary argument will lead to an asinh expression.

    Examples
    ========

    >>> from sympy import asin, oo
    >>> asin(1)
    pi/2
    >>> asin(-1)
    -pi/2
    >>> asin(-oo)
    oo*I
    >>> asin(oo)
    -oo*I

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    acsc, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.23
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcSin

    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 1/sqrt(1 - self.args[0]**2)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_is_rational(self):
        s = self.func(*self.args)
        if s.func == self.func:
            if s.args[0].is_rational:
                return False
        else:
            return s.is_rational

    def _eval_is_positive(self):
        return self._eval_is_extended_real() and self.args[0].is_positive

    def _eval_is_negative(self):
        return self._eval_is_extended_real() and self.args[0].is_negative

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.NegativeInfinity*S.ImaginaryUnit
            elif arg is S.NegativeInfinity:
                return S.Infinity*S.ImaginaryUnit
            elif arg.is_zero:
                return S.Zero
            elif arg is S.One:
                return pi/2
            elif arg is S.NegativeOne:
                return -pi/2

        if arg is S.ComplexInfinity:
            return S.ComplexInfinity

        if arg.could_extract_minus_sign():
            return -cls(-arg)

        if arg.is_number:
            asin_table = cls._asin_table()
            if arg in asin_table:
                return asin_table[arg]

        i_coeff = _imaginary_unit_as_coefficient(arg)
        if i_coeff is not None:
            from sympy.functions.elementary.hyperbolic import asinh
            return S.ImaginaryUnit*asinh(i_coeff)

        if arg.is_zero:
            return S.Zero

        if isinstance(arg, sin):
            ang = arg.args[0]
            if ang.is_comparable:
                ang %= 2*pi # restrict to [0,2*pi)
                if ang > pi: # restrict to (-pi,pi]
                    ang = pi - ang

                # restrict to [-pi/2,pi/2]
                if ang > pi/2:
                    ang = pi - ang
                if ang < -pi/2:
                    ang = -pi - ang

                return ang

        if isinstance(arg, cos): # acos(x) + asin(x) = pi/2
            ang = arg.args[0]
            if ang.is_comparable:
                return pi/2 - acos(arg)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) >= 2 and n > 2:
                p = previous_terms[-2]
                return p*(n - 2)**2/(n*(n - 1))*x**2
            else:
                k = (n - 1) // 2
                R = RisingFactorial(S.Half, k)
                F = factorial(k)
                return R/F*x**n/n

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        if x0 is S.NaN:
            return self.func(arg.as_leading_term(x))
        if x0.is_zero:
            return arg.as_leading_term(x)

        # Handling branch points
        if x0 in (-S.One, S.One, S.ComplexInfinity):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        # Handling points lying on branch cuts (-oo, -1) U (1, oo)
        if (1 - x0**2).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if x0.is_negative:
                    return -pi - self.func(x0)
            elif im(ndir).is_positive:
                if x0.is_positive:
                    return pi - self.func(x0)
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):  # asin
        from sympy.series.order import O
        arg0 = self.args[0].subs(x, 0)
        # Handling branch points
        if arg0 is S.One:
            t = Dummy('t', positive=True)
            ser = asin(S.One - t**2).rewrite(log).nseries(t, 0, 2*n)
            arg1 = S.One - self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            if not g.is_meromorphic(x, 0):   # cannot be expanded
                return O(1) if n == 0 else pi/2 + O(sqrt(x))
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO()*sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        if arg0 is S.NegativeOne:
            t = Dummy('t', positive=True)
            ser = asin(S.NegativeOne + t**2).rewrite(log).nseries(t, 0, 2*n)
            arg1 = S.One + self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            if not g.is_meromorphic(x, 0):   # cannot be expanded
                return O(1) if n == 0 else -pi/2 + O(sqrt(x))
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO()*sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        res = super()._eval_nseries(x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res
        # Handling points lying on branch cuts (-oo, -1) U (1, oo)
        if (1 - arg0**2).is_negative:
            ndir = self.args[0].dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if arg0.is_negative:
                    return -pi - res
            elif im(ndir).is_positive:
                if arg0.is_positive:
                    return pi - res
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def _eval_rewrite_as_acos(self, x, **kwargs):
        return pi/2 - acos(x)

    def _eval_rewrite_as_atan(self, x, **kwargs):
        return 2*atan(x/(1 + sqrt(1 - x**2)))

    def _eval_rewrite_as_log(self, x, **kwargs):
        return -S.ImaginaryUnit*log(S.ImaginaryUnit*x + sqrt(1 - x**2))

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_acot(self, arg, **kwargs):
        return 2*acot((1 + sqrt(1 - arg**2))/arg)

    def _eval_rewrite_as_asec(self, arg, **kwargs):
        return pi/2 - asec(1/arg)

    def _eval_rewrite_as_acsc(self, arg, **kwargs):
        return acsc(1/arg)

    def _eval_is_extended_real(self):
        x = self.args[0]
        return x.is_extended_real and (1 - abs(x)).is_nonnegative

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return sin


class acos(InverseTrigonometricFunction):
    r"""
    The inverse cosine function.

    Explanation
    ===========

    Returns the arc cosine of x (measured in radians).

    ``acos(x)`` will evaluate automatically in the cases
    $x \in \{\infty, -\infty, 0, 1, -1\}$ and for some instances when
    the result is a rational multiple of $\pi$ (see the eval class method).

    ``acos(zoo)`` evaluates to ``zoo``
    (see note in :class:`sympy.functions.elementary.trigonometric.asec`)

    A purely imaginary argument will be rewritten to asinh.

    Examples
    ========

    >>> from sympy import acos, oo
    >>> acos(1)
    0
    >>> acos(0)
    pi/2
    >>> acos(oo)
    oo*I

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    asin, acsc, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.23
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcCos

    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return -1/sqrt(1 - self.args[0]**2)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_is_rational(self):
        s = self.func(*self.args)
        if s.func == self.func:
            if s.args[0].is_rational:
                return False
        else:
            return s.is_rational

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity*S.ImaginaryUnit
            elif arg is S.NegativeInfinity:
                return S.NegativeInfinity*S.ImaginaryUnit
            elif arg.is_zero:
                return pi/2
            elif arg is S.One:
                return S.Zero
            elif arg is S.NegativeOne:
                return pi

        if arg is S.ComplexInfinity:
            return S.ComplexInfinity

        if arg.is_number:
            asin_table = cls._asin_table()
            if arg in asin_table:
                return pi/2 - asin_table[arg]
            elif -arg in asin_table:
                return pi/2 + asin_table[-arg]

        i_coeff = _imaginary_unit_as_coefficient(arg)
        if i_coeff is not None:
            return pi/2 - asin(arg)

        if arg.is_Mul and len(arg.args) == 2 and arg.args[0] == -1:
            narg = arg.args[1]
            minus = True
        else:
            narg = arg
            minus = False

        if isinstance(narg, cos):
            # acos(cos(x)) = x or acos(-cos(x)) = pi - x
            ang = narg.args[0]
            if ang.is_comparable:
                if minus:
                    ang = pi - ang
                ang %= 2*pi # restrict to [0,2*pi)
                if ang > pi: # restrict to [0,pi]
                    ang = 2*pi - ang
                return ang

        if isinstance(narg, sin): # acos(x) + asin(x) = pi/2
            ang = narg.args[0]
            if ang.is_comparable:
                if minus:
                    return pi/2 + asin(narg)
                return pi/2 - asin(narg)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return pi/2
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) >= 2 and n > 2:
                p = previous_terms[-2]
                return p*(n - 2)**2/(n*(n - 1))*x**2
            else:
                k = (n - 1) // 2
                R = RisingFactorial(S.Half, k)
                F = factorial(k)
                return -R/F*x**n/n

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        if x0 is S.NaN:
            return self.func(arg.as_leading_term(x))
        # Handling branch points
        if x0 == 1:
            return sqrt(2)*sqrt((S.One - arg).as_leading_term(x))
        if x0 in (-S.One, S.ComplexInfinity):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        # Handling points lying on branch cuts (-oo, -1) U (1, oo)
        if (1 - x0**2).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if x0.is_negative:
                    return 2*pi - self.func(x0)
            elif im(ndir).is_positive:
                if x0.is_positive:
                    return -self.func(x0)
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        return self.func(x0)

    def _eval_is_extended_real(self):
        x = self.args[0]
        return x.is_extended_real and (1 - abs(x)).is_nonnegative

    def _eval_is_nonnegative(self):
        return self._eval_is_extended_real()

    def _eval_nseries(self, x, n, logx, cdir=0):  # acos
        from sympy.series.order import O
        arg0 = self.args[0].subs(x, 0)
        # Handling branch points
        if arg0 is S.One:
            t = Dummy('t', positive=True)
            ser = acos(S.One - t**2).rewrite(log).nseries(t, 0, 2*n)
            arg1 = S.One - self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            if not g.is_meromorphic(x, 0):   # cannot be expanded
                return O(1) if n == 0 else O(sqrt(x))
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO()*sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        if arg0 is S.NegativeOne:
            t = Dummy('t', positive=True)
            ser = acos(S.NegativeOne + t**2).rewrite(log).nseries(t, 0, 2*n)
            arg1 = S.One + self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            if not g.is_meromorphic(x, 0):   # cannot be expanded
                return O(1) if n == 0 else pi + O(sqrt(x))
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO()*sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        res = super()._eval_nseries(x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res
        # Handling points lying on branch cuts (-oo, -1) U (1, oo)
        if (1 - arg0**2).is_negative:
            ndir = self.args[0].dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if arg0.is_negative:
                    return 2*pi - res
            elif im(ndir).is_positive:
                if arg0.is_positive:
                    return -res
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def _eval_rewrite_as_log(self, x, **kwargs):
        return pi/2 + S.ImaginaryUnit*\
            log(S.ImaginaryUnit*x + sqrt(1 - x**2))

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_asin(self, x, **kwargs):
        return pi/2 - asin(x)

    def _eval_rewrite_as_atan(self, x, **kwargs):
        return atan(sqrt(1 - x**2)/x) + (pi/2)*(1 - x*sqrt(1/x**2))

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return cos

    def _eval_rewrite_as_acot(self, arg, **kwargs):
        return pi/2 - 2*acot((1 + sqrt(1 - arg**2))/arg)

    def _eval_rewrite_as_asec(self, arg, **kwargs):
        return asec(1/arg)

    def _eval_rewrite_as_acsc(self, arg, **kwargs):
        return pi/2 - acsc(1/arg)

    def _eval_conjugate(self):
        z = self.args[0]
        r = self.func(self.args[0].conjugate())
        if z.is_extended_real is False:
            return r
        elif z.is_extended_real and (z + 1).is_nonnegative and (z - 1).is_nonpositive:
            return r


class atan(InverseTrigonometricFunction):
    r"""
    The inverse tangent function.

    Returns the arc tangent of x (measured in radians).

    Explanation
    ===========

    ``atan(x)`` will evaluate automatically in the cases
    $x \in \{\infty, -\infty, 0, 1, -1\}$ and for some instances when the
    result is a rational multiple of $\pi$ (see the eval class method).

    Examples
    ========

    >>> from sympy import atan, oo
    >>> atan(0)
    0
    >>> atan(1)
    pi/4
    >>> atan(oo)
    pi/2

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    asin, acsc, acos, asec, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.23
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcTan

    """

    args: tuple[Expr]

    _singularities = (S.ImaginaryUnit, -S.ImaginaryUnit)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 1/(1 + self.args[0]**2)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_is_rational(self):
        s = self.func(*self.args)
        if s.func == self.func:
            if s.args[0].is_rational:
                return False
        else:
            return s.is_rational

    def _eval_is_positive(self):
        return self.args[0].is_extended_positive

    def _eval_is_nonnegative(self):
        return self.args[0].is_extended_nonnegative

    def _eval_is_zero(self):
        return self.args[0].is_zero

    def _eval_is_real(self):
        return self.args[0].is_extended_real

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return pi/2
            elif arg is S.NegativeInfinity:
                return -pi/2
            elif arg.is_zero:
                return S.Zero
            elif arg is S.One:
                return pi/4
            elif arg is S.NegativeOne:
                return -pi/4

        if arg is S.ComplexInfinity:
            from sympy.calculus.accumulationbounds import AccumBounds
            return AccumBounds(-pi/2, pi/2)

        if arg.could_extract_minus_sign():
            return -cls(-arg)

        if arg.is_number:
            atan_table = cls._atan_table()
            if arg in atan_table:
                return atan_table[arg]

        i_coeff = _imaginary_unit_as_coefficient(arg)
        if i_coeff is not None:
            from sympy.functions.elementary.hyperbolic import atanh
            return S.ImaginaryUnit*atanh(i_coeff)

        if arg.is_zero:
            return S.Zero

        if isinstance(arg, tan):
            ang = arg.args[0]
            if ang.is_comparable:
                ang %= pi # restrict to [0,pi)
                if ang > pi/2: # restrict to [-pi/2,pi/2]
                    ang -= pi

                return ang

        if isinstance(arg, cot): # atan(x) + acot(x) = pi/2
            ang = arg.args[0]
            if ang.is_comparable:
                ang = pi/2 - acot(arg)
                if ang > pi/2: # restrict to [-pi/2,pi/2]
                    ang -= pi
                return ang

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            return S.NegativeOne**((n - 1)//2)*x**n/n

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        if x0 is S.NaN:
            return self.func(arg.as_leading_term(x))
        if x0.is_zero:
            return arg.as_leading_term(x)
        # Handling branch points
        if x0 in (-S.ImaginaryUnit, S.ImaginaryUnit, S.ComplexInfinity):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        # Handling points lying on branch cuts (-I*oo, -I) U (I, I*oo)
        if (1 + x0**2).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if re(ndir).is_negative:
                if im(x0).is_positive:
                    return self.func(x0) - pi
            elif re(ndir).is_positive:
                if im(x0).is_negative:
                    return self.func(x0) + pi
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):  # atan
        arg0 = self.args[0].subs(x, 0)

        # Handling branch points
        if arg0 in (S.ImaginaryUnit, S.NegativeOne*S.ImaginaryUnit):
            return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)

        res = super()._eval_nseries(x, n=n, logx=logx)
        ndir = self.args[0].dir(x, cdir if cdir else 1)
        if arg0 is S.ComplexInfinity:
            if re(ndir) > 0:
                return res - pi
            return res
        # Handling points lying on branch cuts (-I*oo, -I) U (I, I*oo)
        if (1 + arg0**2).is_negative:
            if re(ndir).is_negative:
                if im(arg0).is_positive:
                    return res - pi
            elif re(ndir).is_positive:
                if im(arg0).is_negative:
                    return res + pi
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def _eval_rewrite_as_log(self, x, **kwargs):
        return S.ImaginaryUnit/2*(log(S.One - S.ImaginaryUnit*x)
            - log(S.One + S.ImaginaryUnit*x))

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_aseries(self, n, args0, x, logx):
        if args0[0] in [S.Infinity, S.NegativeInfinity]:
            return (pi/2 - atan(1/self.args[0]))._eval_nseries(x, n, logx)
        else:
            return super()._eval_aseries(n, args0, x, logx)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return tan

    def _eval_rewrite_as_asin(self, arg, **kwargs):
        return sqrt(arg**2)/arg*(pi/2 - asin(1/sqrt(1 + arg**2)))

    def _eval_rewrite_as_acos(self, arg, **kwargs):
        return sqrt(arg**2)/arg*acos(1/sqrt(1 + arg**2))

    def _eval_rewrite_as_acot(self, arg, **kwargs):
        return acot(1/arg)

    def _eval_rewrite_as_asec(self, arg, **kwargs):
        return sqrt(arg**2)/arg*asec(sqrt(1 + arg**2))

    def _eval_rewrite_as_acsc(self, arg, **kwargs):
        return sqrt(arg**2)/arg*(pi/2 - acsc(sqrt(1 + arg**2)))


class acot(InverseTrigonometricFunction):
    r"""
    The inverse cotangent function.

    Returns the arc cotangent of x (measured in radians).

    Explanation
    ===========

    ``acot(x)`` will evaluate automatically in the cases
    $x \in \{\infty, -\infty, \tilde{\infty}, 0, 1, -1\}$
    and for some instances when the result is a rational multiple of $\pi$
    (see the eval class method).

    A purely imaginary argument will lead to an ``acoth`` expression.

    ``acot(x)`` has a branch cut along $(-i, i)$, hence it is discontinuous
    at 0. Its range for real $x$ is $(-\frac{\pi}{2}, \frac{\pi}{2}]$.

    Examples
    ========

    >>> from sympy import acot, sqrt
    >>> acot(0)
    pi/2
    >>> acot(1)
    pi/4
    >>> acot(sqrt(3) - 2)
    -5*pi/12

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    asin, acsc, acos, asec, atan, atan2

    References
    ==========

    .. [1] https://dlmf.nist.gov/4.23
    .. [2] https://functions.wolfram.com/ElementaryFunctions/ArcCot

    """
    _singularities = (S.ImaginaryUnit, -S.ImaginaryUnit)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return -1/(1 + self.args[0]**2)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_is_rational(self):
        s = self.func(*self.args)
        if s.func == self.func:
            if s.args[0].is_rational:
                return False
        else:
            return s.is_rational

    def _eval_is_positive(self):
        return self.args[0].is_nonnegative

    def _eval_is_negative(self):
        return self.args[0].is_negative

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Zero
            elif arg is S.NegativeInfinity:
                return S.Zero
            elif arg.is_zero:
                return pi/ 2
            elif arg is S.One:
                return pi/4
            elif arg is S.NegativeOne:
                return -pi/4

        if arg is S.ComplexInfinity:
            return S.Zero

        if arg.could_extract_minus_sign():
            return -cls(-arg)

        if arg.is_number:
            atan_table = cls._atan_table()
            if arg in atan_table:
                ang = pi/2 - atan_table[arg]
                if ang > pi/2: # restrict to (-pi/2,pi/2]
                    ang -= pi
                return ang

        i_coeff = _imaginary_unit_as_coefficient(arg)
        if i_coeff is not None:
            from sympy.functions.elementary.hyperbolic import acoth
            return -S.ImaginaryUnit*acoth(i_coeff)

        if arg.is_zero:
            return pi*S.Half

        if isinstance(arg, cot):
            ang = arg.args[0]
            if ang.is_comparable:
                ang %= pi # restrict to [0,pi)
                if ang > pi/2: # restrict to (-pi/2,pi/2]
                    ang -= pi
                return ang

        if isinstance(arg, tan): # atan(x) + acot(x) = pi/2
            ang = arg.args[0]
            if ang.is_comparable:
                ang = pi/2 - atan(arg)
                if ang > pi/2: # restrict to (-pi/2,pi/2]
                    ang -= pi
                return ang

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return pi/2  # FIX THIS
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            return S.NegativeOne**((n + 1)//2)*x**n/n

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        if x0 is S.NaN:
            return self.func(arg.as_leading_term(x))
        if x0 is S.ComplexInfinity:
            return (1/arg).as_leading_term(x)
        # Handling branch points
        if x0 in (-S.ImaginaryUnit, S.ImaginaryUnit, S.Zero):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        # Handling points lying on branch cuts [-I, I]
        if x0.is_imaginary and (1 + x0**2).is_positive:
            ndir = arg.dir(x, cdir if cdir else 1)
            if re(ndir).is_positive:
                if im(x0).is_positive:
                    return self.func(x0) + pi
            elif re(ndir).is_negative:
                if im(x0).is_negative:
                    return self.func(x0) - pi
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):  # acot
        arg0 = self.args[0].subs(x, 0)

        # Handling branch points
        if arg0 in (S.ImaginaryUnit, S.NegativeOne*S.ImaginaryUnit):
            return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)

        res = super()._eval_nseries(x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res
        ndir = self.args[0].dir(x, cdir if cdir else 1)
        if arg0.is_zero:
            if re(ndir) < 0:
                return res - pi
            return res
        # Handling points lying on branch cuts [-I, I]
        if arg0.is_imaginary and (1 + arg0**2).is_positive:
            if re(ndir).is_positive:
                if im(arg0).is_positive:
                    return res + pi
            elif re(ndir).is_negative:
                if im(arg0).is_negative:
                    return res - pi
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def _eval_aseries(self, n, args0, x, logx):
        if args0[0] in [S.Infinity, S.NegativeInfinity]:
            return atan(1/self.args[0])._eval_nseries(x, n, logx)
        else:
            return super()._eval_aseries(n, args0, x, logx)

    def _eval_rewrite_as_log(self, x, **kwargs):
        return S.ImaginaryUnit/2*(log(1 - S.ImaginaryUnit/x)
            - log(1 + S.ImaginaryUnit/x))

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return cot

    def _eval_rewrite_as_asin(self, arg, **kwargs):
        return (arg*sqrt(1/arg**2)*
                (pi/2 - asin(sqrt(-arg**2)/sqrt(-arg**2 - 1))))

    def _eval_rewrite_as_acos(self, arg, **kwargs):
        return arg*sqrt(1/arg**2)*acos(sqrt(-arg**2)/sqrt(-arg**2 - 1))

    def _eval_rewrite_as_atan(self, arg, **kwargs):
        return atan(1/arg)

    def _eval_rewrite_as_asec(self, arg, **kwargs):
        return arg*sqrt(1/arg**2)*asec(sqrt((1 + arg**2)/arg**2))

    def _eval_rewrite_as_acsc(self, arg, **kwargs):
        return arg*sqrt(1/arg**2)*(pi/2 - acsc(sqrt((1 + arg**2)/arg**2)))


class asec(InverseTrigonometricFunction):
    r"""
    The inverse secant function.

    Returns the arc secant of x (measured in radians).

    Explanation
    ===========

    ``asec(x)`` will evaluate automatically in the cases
    $x \in \{\infty, -\infty, 0, 1, -1\}$ and for some instances when the
    result is a rational multiple of $\pi$ (see the eval class method).

    ``asec(x)`` has branch cut in the interval $[-1, 1]$. For complex arguments,
    it can be defined [4]_ as

    .. math::
        \operatorname{sec^{-1}}(z) = -i\frac{\log\left(\sqrt{1 - z^2} + 1\right)}{z}

    At ``x = 0``, for positive branch cut, the limit evaluates to ``zoo``. For
    negative branch cut, the limit

    .. math::
        \lim_{z \to 0}-i\frac{\log\left(-\sqrt{1 - z^2} + 1\right)}{z}

    simplifies to :math:`-i\log\left(z/2 + O\left(z^3\right)\right)` which
    ultimately evaluates to ``zoo``.

    As ``acos(x) = asec(1/x)``, a similar argument can be given for
    ``acos(x)``.

    Examples
    ========

    >>> from sympy import asec, oo
    >>> asec(1)
    0
    >>> asec(-1)
    pi
    >>> asec(0)
    zoo
    >>> asec(-oo)
    pi/2

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    asin, acsc, acos, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.23
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcSec
    .. [4] https://reference.wolfram.com/language/ref/ArcSec.html

    """

    @classmethod
    def eval(cls, arg):
        if arg.is_zero:
            return S.ComplexInfinity
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.One:
                return S.Zero
            elif arg is S.NegativeOne:
                return pi
        if arg in [S.Infinity, S.NegativeInfinity, S.ComplexInfinity]:
            return pi/2

        if arg.is_number:
            acsc_table = cls._acsc_table()
            if arg in acsc_table:
                return pi/2 - acsc_table[arg]
            elif -arg in acsc_table:
                return pi/2 + acsc_table[-arg]

        if arg.is_infinite:
            return pi/2

        if arg.is_Mul and len(arg.args) == 2 and arg.args[0] == -1:
            narg = arg.args[1]
            minus = True
        else:
            narg = arg
            minus = False

        if isinstance(narg, sec):
            # asec(sec(x)) = x or asec(-sec(x)) = pi - x
            ang = narg.args[0]
            if ang.is_comparable:
                if minus:
                    ang = pi - ang
                ang %= 2*pi # restrict to [0,2*pi)
                if ang > pi: # restrict to [0,pi]
                    ang = 2*pi - ang
                return ang

        if isinstance(narg, csc): # asec(x) + acsc(x) = pi/2
            ang = narg.args[0]
            if ang.is_comparable:
                if minus:
                    pi/2 + acsc(narg)
                return pi/2 - acsc(narg)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 1/(self.args[0]**2*sqrt(1 - 1/self.args[0]**2))
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return sec

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return S.ImaginaryUnit*log(2 / x)
        elif n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 2 and n > 2:
                p = previous_terms[-2]
                return p * ((n - 1)*(n-2)) * x**2/(4 * (n//2)**2)
            else:
                k = n // 2
                R = RisingFactorial(S.Half, k) *  n
                F = factorial(k) * n // 2 * n // 2
                return -S.ImaginaryUnit * R / F * x**n / 4

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        if x0 is S.NaN:
            return self.func(arg.as_leading_term(x))
        # Handling branch points
        if x0 == 1:
            return sqrt(2)*sqrt((arg - S.One).as_leading_term(x))
        if x0 in (-S.One, S.Zero):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        # Handling points lying on branch cuts (-1, 1)
        if x0.is_real and (1 - x0**2).is_positive:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if x0.is_positive:
                    return -self.func(x0)
            elif im(ndir).is_positive:
                if x0.is_negative:
                    return 2*pi - self.func(x0)
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):  # asec
        from sympy.series.order import O
        arg0 = self.args[0].subs(x, 0)
        # Handling branch points
        if arg0 is S.One:
            t = Dummy('t', positive=True)
            ser = asec(S.One + t**2).rewrite(log).nseries(t, 0, 2*n)
            arg1 = S.NegativeOne + self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO()*sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        if arg0 is S.NegativeOne:
            t = Dummy('t', positive=True)
            ser = asec(S.NegativeOne - t**2).rewrite(log).nseries(t, 0, 2*n)
            arg1 = S.NegativeOne - self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO()*sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        res = super()._eval_nseries(x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res
        # Handling points lying on branch cuts (-1, 1)
        if arg0.is_real and (1 - arg0**2).is_positive:
            ndir = self.args[0].dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if arg0.is_positive:
                    return -res
            elif im(ndir).is_positive:
                if arg0.is_negative:
                    return 2*pi - res
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def _eval_is_extended_real(self):
        x = self.args[0]
        if x.is_extended_real is False:
            return False
        return fuzzy_or(((x - 1).is_nonnegative, (-x - 1).is_nonnegative))

    def _eval_rewrite_as_log(self, arg, **kwargs):
        return pi/2 + S.ImaginaryUnit*log(S.ImaginaryUnit/arg + sqrt(1 - 1/arg**2))

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_asin(self, arg, **kwargs):
        return pi/2 - asin(1/arg)

    def _eval_rewrite_as_acos(self, arg, **kwargs):
        return acos(1/arg)

    def _eval_rewrite_as_atan(self, x, **kwargs):
        sx2x = sqrt(x**2)/x
        return pi/2*(1 - sx2x) + sx2x*atan(sqrt(x**2 - 1))

    def _eval_rewrite_as_acot(self, x, **kwargs):
        sx2x = sqrt(x**2)/x
        return pi/2*(1 - sx2x) + sx2x*acot(1/sqrt(x**2 - 1))

    def _eval_rewrite_as_acsc(self, arg, **kwargs):
        return pi/2 - acsc(arg)


class acsc(InverseTrigonometricFunction):
    r"""
    The inverse cosecant function.

    Returns the arc cosecant of x (measured in radians).

    Explanation
    ===========

    ``acsc(x)`` will evaluate automatically in the cases
    $x \in \{\infty, -\infty, 0, 1, -1\}$` and for some instances when the
    result is a rational multiple of $\pi$ (see the ``eval`` class method).

    Examples
    ========

    >>> from sympy import acsc, oo
    >>> acsc(1)
    pi/2
    >>> acsc(-1)
    -pi/2
    >>> acsc(oo)
    0
    >>> acsc(-oo) == acsc(oo)
    True
    >>> acsc(0)
    zoo

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    asin, acos, asec, atan, acot, atan2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
    .. [2] https://dlmf.nist.gov/4.23
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcCsc

    """

    @classmethod
    def eval(cls, arg):
        if arg.is_zero:
            return S.ComplexInfinity
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.One:
                return pi/2
            elif arg is S.NegativeOne:
                return -pi/2
        if arg in [S.Infinity, S.NegativeInfinity, S.ComplexInfinity]:
            return S.Zero

        if arg.could_extract_minus_sign():
            return -cls(-arg)

        if arg.is_infinite:
            return S.Zero

        if arg.is_number:
            acsc_table = cls._acsc_table()
            if arg in acsc_table:
                return acsc_table[arg]

        if isinstance(arg, csc):
            ang = arg.args[0]
            if ang.is_comparable:
                ang %= 2*pi # restrict to [0,2*pi)
                if ang > pi: # restrict to (-pi,pi]
                    ang = pi - ang

                # restrict to [-pi/2,pi/2]
                if ang > pi/2:
                    ang = pi - ang
                if ang < -pi/2:
                    ang = -pi - ang

                return ang

        if isinstance(arg, sec): # asec(x) + acsc(x) = pi/2
            ang = arg.args[0]
            if ang.is_comparable:
                return pi/2 - asec(arg)

    def fdiff(self, argindex=1):
        if argindex == 1:
            return -1/(self.args[0]**2*sqrt(1 - 1/self.args[0]**2))
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return csc

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return pi/2 - S.ImaginaryUnit*log(2) + S.ImaginaryUnit*log(x)
        elif n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 2 and n > 2:
                p = previous_terms[-2]
                return p * ((n - 1)*(n-2)) * x**2/(4 * (n//2)**2)
            else:
                k = n // 2
                R = RisingFactorial(S.Half, k) *  n
                F = factorial(k) * n // 2 * n // 2
                return S.ImaginaryUnit * R / F * x**n / 4

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        if x0 is S.NaN:
            return self.func(arg.as_leading_term(x))
        # Handling branch points
        if x0 in (-S.One, S.One, S.Zero):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        if x0 is S.ComplexInfinity:
            return (1/arg).as_leading_term(x)
        # Handling points lying on branch cuts (-1, 1)
        if x0.is_real and (1 - x0**2).is_positive:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if x0.is_positive:
                    return pi - self.func(x0)
            elif im(ndir).is_positive:
                if x0.is_negative:
                    return -pi - self.func(x0)
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir).expand()
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):  # acsc
        from sympy.series.order import O
        arg0 = self.args[0].subs(x, 0)
        # Handling branch points
        if arg0 is S.One:
            t = Dummy('t', positive=True)
            ser = acsc(S.One + t**2).rewrite(log).nseries(t, 0, 2*n)
            arg1 = S.NegativeOne + self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO()*sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        if arg0 is S.NegativeOne:
            t = Dummy('t', positive=True)
            ser = acsc(S.NegativeOne - t**2).rewrite(log).nseries(t, 0, 2*n)
            arg1 = S.NegativeOne - self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO()*sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        res = super()._eval_nseries(x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res
        # Handling points lying on branch cuts (-1, 1)
        if arg0.is_real and (1 - arg0**2).is_positive:
            ndir = self.args[0].dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if arg0.is_positive:
                    return pi - res
            elif im(ndir).is_positive:
                if arg0.is_negative:
                    return -pi - res
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def _eval_rewrite_as_log(self, arg, **kwargs):
        return -S.ImaginaryUnit*log(S.ImaginaryUnit/arg + sqrt(1 - 1/arg**2))

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_asin(self, arg, **kwargs):
        return asin(1/arg)

    def _eval_rewrite_as_acos(self, arg, **kwargs):
        return pi/2 - acos(1/arg)

    def _eval_rewrite_as_atan(self, x, **kwargs):
        return sqrt(x**2)/x*(pi/2 - atan(sqrt(x**2 - 1)))

    def _eval_rewrite_as_acot(self, arg, **kwargs):
        return sqrt(arg**2)/arg*(pi/2 - acot(1/sqrt(arg**2 - 1)))

    def _eval_rewrite_as_asec(self, arg, **kwargs):
        return pi/2 - asec(arg)


class atan2(InverseTrigonometricFunction):
    r"""
    The function ``atan2(y, x)`` computes `\operatorname{atan}(y/x)` taking
    two arguments `y` and `x`.  Signs of both `y` and `x` are considered to
    determine the appropriate quadrant of `\operatorname{atan}(y/x)`.
    The range is `(-\pi, \pi]`. The complete definition reads as follows:

    .. math::

        \operatorname{atan2}(y, x) =
        \begin{cases}
          \arctan\left(\frac y x\right) & \qquad x > 0 \\
          \arctan\left(\frac y x\right) + \pi& \qquad y \ge 0, x < 0 \\
          \arctan\left(\frac y x\right) - \pi& \qquad y < 0, x < 0 \\
          +\frac{\pi}{2} & \qquad y > 0, x = 0 \\
          -\frac{\pi}{2} & \qquad y < 0, x = 0 \\
          \text{undefined} & \qquad y = 0, x = 0
        \end{cases}

    Attention: Note the role reversal of both arguments. The `y`-coordinate
    is the first argument and the `x`-coordinate the second.

    If either `x` or `y` is complex:

    .. math::

        \operatorname{atan2}(y, x) =
            -i\log\left(\frac{x + iy}{\sqrt{x^2 + y^2}}\right)

    Examples
    ========

    Going counter-clock wise around the origin we find the
    following angles:

    >>> from sympy import atan2
    >>> atan2(0, 1)
    0
    >>> atan2(1, 1)
    pi/4
    >>> atan2(1, 0)
    pi/2
    >>> atan2(1, -1)
    3*pi/4
    >>> atan2(0, -1)
    pi
    >>> atan2(-1, -1)
    -3*pi/4
    >>> atan2(-1, 0)
    -pi/2
    >>> atan2(-1, 1)
    -pi/4

    which are all correct. Compare this to the results of the ordinary
    `\operatorname{atan}` function for the point `(x, y) = (-1, 1)`

    >>> from sympy import atan, S
    >>> atan(S(1)/-1)
    -pi/4
    >>> atan2(1, -1)
    3*pi/4

    where only the `\operatorname{atan2}` function returns what we expect.
    We can differentiate the function with respect to both arguments:

    >>> from sympy import diff
    >>> from sympy.abc import x, y
    >>> diff(atan2(y, x), x)
    -y/(x**2 + y**2)

    >>> diff(atan2(y, x), y)
    x/(x**2 + y**2)

    We can express the `\operatorname{atan2}` function in terms of
    complex logarithms:

    >>> from sympy import log
    >>> atan2(y, x).rewrite(log)
    -I*log((x + I*y)/sqrt(x**2 + y**2))

    and in terms of `\operatorname(atan)`:

    >>> from sympy import atan
    >>> atan2(y, x).rewrite(atan)
    Piecewise((2*atan(y/(x + sqrt(x**2 + y**2))), Ne(y, 0)), (pi, re(x) < 0), (0, Ne(x, 0)), (nan, True))

    but note that this form is undefined on the negative real axis.

    See Also
    ========

    sin, csc, cos, sec, tan, cot
    asin, acsc, acos, asec, atan, acot

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
    .. [2] https://en.wikipedia.org/wiki/Atan2
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcTan2

    """

    @classmethod
    def eval(cls, y, x):
        from sympy.functions.special.delta_functions import Heaviside
        if x is S.NegativeInfinity:
            if y.is_zero:
                # Special case y = 0 because we define Heaviside(0) = 1/2
                return pi
            return 2*pi*(Heaviside(re(y))) - pi
        elif x is S.Infinity:
            return S.Zero
        elif x.is_imaginary and y.is_imaginary and x.is_number and y.is_number:
            x = im(x)
            y = im(y)

        if x.is_extended_real and y.is_extended_real:
            if x.is_positive:
                return atan(y/x)
            elif x.is_negative:
                if y.is_negative:
                    return atan(y/x) - pi
                elif y.is_nonnegative:
                    return atan(y/x) + pi
            elif x.is_zero:
                if y.is_positive:
                    return pi/2
                elif y.is_negative:
                    return -pi/2
                elif y.is_zero:
                    return S.NaN
        if y.is_zero:
            if x.is_extended_nonzero:
                return pi*(S.One - Heaviside(x))
            if x.is_number:
                return Piecewise((pi, re(x) < 0),
                                 (0, Ne(x, 0)),
                                 (S.NaN, True))
        if x.is_number and y.is_number:
            return -S.ImaginaryUnit*log(
                (x + S.ImaginaryUnit*y)/sqrt(x**2 + y**2))

    def _eval_rewrite_as_log(self, y, x, **kwargs):
        return -S.ImaginaryUnit*log((x + S.ImaginaryUnit*y)/sqrt(x**2 + y**2))

    def _eval_rewrite_as_atan(self, y, x, **kwargs):
        return Piecewise((2*atan(y/(x + sqrt(x**2 + y**2))), Ne(y, 0)),
                         (pi, re(x) < 0),
                         (0, Ne(x, 0)),
                         (S.NaN, True))

    def _eval_rewrite_as_arg(self, y, x, **kwargs):
        if x.is_extended_real and y.is_extended_real:
            return arg_f(x + y*S.ImaginaryUnit)
        n = x + S.ImaginaryUnit*y
        d = x**2 + y**2
        return arg_f(n/sqrt(d)) - S.ImaginaryUnit*log(abs(n)/sqrt(abs(d)))

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real and self.args[1].is_extended_real

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate(), self.args[1].conjugate())

    def fdiff(self, argindex):
        y, x = self.args
        if argindex == 1:
            # Diff wrt y
            return x/(x**2 + y**2)
        elif argindex == 2:
            # Diff wrt x
            return -y/(x**2 + y**2)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_evalf(self, prec):
        y, x = self.args
        if x.is_extended_real and y.is_extended_real:
            return super()._eval_evalf(prec)
<|endoftext|>r"""A module for special angle formulas for trigonometric functions

TODO
====

This module should be developed in the future to contain direct square root
representation of

.. math
    F(\frac{n}{m} \pi)

for every

- $m \in \{ 3, 5, 17, 257, 65537 \}$
- $n \in \mathbb{N}$, $0 \le n < m$
- $F \in \{\sin, \cos, \tan, \csc, \sec, \cot\}$

Without multi-step rewrites
(e.g. $\tan \to \cos/\sin \to \cos/\sqrt \to \ sqrt$)
or using chebyshev identities
(e.g. $\cos \to \cos + \cos^2 + \cdots \to \sqrt{} + \sqrt{}^2 + \cdots $),
which are trivial to implement in sympy,
and had used to give overly complicated expressions.

The reference can be found below, if anyone may need help implementing them.

References
==========

.. [*] Gottlieb, Christian. (1999). The Simple and straightforward construction
   of the regular 257-gon. The Mathematical Intelligencer. 21. 31-37.
   10.1007/BF03024829.
.. [*] https://resources.wolframcloud.com/FunctionRepository/resources/Cos2PiOverFermatPrime
"""
from __future__ import annotations
from typing import Callable
from functools import reduce
from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.intfunc import igcdex
from sympy.core.numbers import Integer
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core.cache import cacheit


def migcdex(*x: int) -> tuple[tuple[int, ...], int]:
    r"""Compute extended gcd for multiple integers.

    Explanation
    ===========

    Given the integers $x_1, \cdots, x_n$ and
    an extended gcd for multiple arguments are defined as a solution
    $(y_1, \cdots, y_n), g$ for the diophantine equation
    $x_1 y_1 + \cdots + x_n y_n = g$ such that
    $g = \gcd(x_1, \cdots, x_n)$.

    Examples
    ========

    >>> from sympy.functions.elementary._trigonometric_special import migcdex
    >>> migcdex()
    ((), 0)
    >>> migcdex(4)
    ((1,), 4)
    >>> migcdex(4, 6)
    ((-1, 1), 2)
    >>> migcdex(6, 10, 15)
    ((1, 1, -1), 1)
    """
    if not x:
        return (), 0

    if len(x) == 1:
        return (1,), x[0]

    if len(x) == 2:
        u, v, h = igcdex(x[0], x[1])
        return (u, v), h

    y, g = migcdex(*x[1:])
    u, v, h = igcdex(x[0], g)
    return (u, *(v * i for i in y)), h


def ipartfrac(*denoms: int) -> tuple[int, ...]:
    r"""Compute the partial fraction decomposition.

    Explanation
    ===========

    Given a rational number $\frac{1}{q_1 \cdots q_n}$ where all
    $q_1, \cdots, q_n$ are pairwise coprime,

    A partial fraction decomposition is defined as

    .. math::
        \frac{1}{q_1 \cdots q_n} = \frac{p_1}{q_1} + \cdots + \frac{p_n}{q_n}

    And it can be derived from solving the following diophantine equation for
    the $p_1, \cdots, p_n$

    .. math::
        1 = p_1 \prod_{i \ne 1}q_i + \cdots + p_n \prod_{i \ne n}q_i

    Where $q_1, \cdots, q_n$ being pairwise coprime implies
    $\gcd(\prod_{i \ne 1}q_i, \cdots, \prod_{i \ne n}q_i) = 1$,
    which guarantees the existence of the solution.

    It is sufficient to compute partial fraction decomposition only
    for numerator $1$ because partial fraction decomposition for any
    $\frac{n}{q_1 \cdots q_n}$ can be easily computed by multiplying
    the result by $n$ afterwards.

    Parameters
    ==========

    denoms : int
        The pairwise coprime integer denominators $q_i$ which defines the
        rational number $\frac{1}{q_1 \cdots q_n}$

    Returns
    =======

    tuple[int, ...]
        The list of numerators which semantically corresponds to $p_i$ of the
        partial fraction decomposition
        $\frac{1}{q_1 \cdots q_n} = \frac{p_1}{q_1} + \cdots + \frac{p_n}{q_n}$

    Examples
    ========

    >>> from sympy import Rational, Mul
    >>> from sympy.functions.elementary._trigonometric_special import ipartfrac

    >>> denoms = 2, 3, 5
    >>> numers = ipartfrac(2, 3, 5)
    >>> numers
    (1, 7, -14)

    >>> Rational(1, Mul(*denoms))
    1/30
    >>> out = 0
    >>> for n, d in zip(numers, denoms):
    ...    out += Rational(n, d)
    >>> out
    1/30
    """
    if not denoms:
        return ()

    def mul(x: int, y: int) -> int:
        return x * y

    denom = reduce(mul, denoms)
    a = [denom // x for x in denoms]
    h, _ = migcdex(*a)
    return h


def fermat_coords(n: int) -> list[int] | None:
    """If n can be factored in terms of Fermat primes with
    multiplicity of each being 1, return those primes, else
    None
    """
    primes = []
    for p in [3, 5, 17, 257, 65537]:
        quotient, remainder = divmod(n, p)
        if remainder == 0:
            n = quotient
            primes.append(p)
            if n == 1:
                return primes
    return None


@cacheit
def cos_3() -> Expr:
    r"""Computes $\cos \frac{\pi}{3}$ in square roots"""
    return S.Half


@cacheit
def cos_5() -> Expr:
    r"""Computes $\cos \frac{\pi}{5}$ in square roots"""
    return (sqrt(5) + 1) / 4


@cacheit
def cos_17() -> Expr:
    r"""Computes $\cos \frac{\pi}{17}$ in square roots"""
    return sqrt(
        (15 + sqrt(17)) / 32 + sqrt(2) * (sqrt(17 - sqrt(17)) +
        sqrt(sqrt(2) * (-8 * sqrt(17 + sqrt(17)) - (1 - sqrt(17))
        * sqrt(17 - sqrt(17))) + 6 * sqrt(17) + 34)) / 32)


@cacheit
def cos_257() -> Expr:
    r"""Computes $\cos \frac{\pi}{257}$ in square roots

    References
    ==========

    .. [*] https://math.stackexchange.com/questions/516142/how-does-cos2-pi-257-look-like-in-real-radicals
    .. [*] https://r-knott.surrey.ac.uk/Fibonacci/simpleTrig.html
    """
    def f1(a: Expr, b: Expr) -> tuple[Expr, Expr]:
        return (a + sqrt(a**2 + b)) / 2, (a - sqrt(a**2 + b)) / 2

    def f2(a: Expr, b: Expr) -> Expr:
        return (a - sqrt(a**2 + b))/2

    t1, t2 = f1(S.NegativeOne, Integer(256))
    z1, z3 = f1(t1, Integer(64))
    z2, z4 = f1(t2, Integer(64))
    y1, y5 = f1(z1, 4*(5 + t1 + 2*z1))
    y6, y2 = f1(z2, 4*(5 + t2 + 2*z2))
    y3, y7 = f1(z3, 4*(5 + t1 + 2*z3))
    y8, y4 = f1(z4, 4*(5 + t2 + 2*z4))
    x1, x9 = f1(y1, -4*(t1 + y1 + y3 + 2*y6))
    x2, x10 = f1(y2, -4*(t2 + y2 + y4 + 2*y7))
    x3, x11 = f1(y3, -4*(t1 + y3 + y5 + 2*y8))
    x4, x12 = f1(y4, -4*(t2 + y4 + y6 + 2*y1))
    x5, x13 = f1(y5, -4*(t1 + y5 + y7 + 2*y2))
    x6, x14 = f1(y6, -4*(t2 + y6 + y8 + 2*y3))
    x15, x7 = f1(y7, -4*(t1 + y7 + y1 + 2*y4))
    x8, x16 = f1(y8, -4*(t2 + y8 + y2 + 2*y5))
    v1 = f2(x1, -4*(x1 + x2 + x3 + x6))
    v2 = f2(x2, -4*(x2 + x3 + x4 + x7))
    v3 = f2(x8, -4*(x8 + x9 + x10 + x13))
    v4 = f2(x9, -4*(x9 + x10 + x11 + x14))
    v5 = f2(x10, -4*(x10 + x11 + x12 + x15))
    v6 = f2(x16, -4*(x16 + x1 + x2 + x5))
    u1 = -f2(-v1, -4*(v2 + v3))
    u2 = -f2(-v4, -4*(v5 + v6))
    w1 = -2*f2(-u1, -4*u2)
    return sqrt(sqrt(2)*sqrt(w1 + 4)/8 + S.Half)


def cos_table() -> dict[int, Callable[[], Expr]]:
    r"""Lazily evaluated table for $\cos \frac{\pi}{n}$ in square roots for
    $n \in \{3, 5, 17, 257, 65537\}$.

    Notes
    =====

    65537 is the only other known Fermat prime and it is nearly impossible to
    build in the current SymPy due to performance issues.

    References
    ==========

    https://r-knott.surrey.ac.uk/Fibonacci/simpleTrig.html
    """
    return {
        3: cos_3,
        5: cos_5,
        17: cos_17,
        257: cos_257
    }
<|endoftext|>from sympy.core import S, sympify, cacheit
from sympy.core.add import Add
from sympy.core.function import DefinedFunction, ArgumentIndexError
from sympy.core.logic import fuzzy_or, fuzzy_and, fuzzy_not, FuzzyBool
from sympy.core.numbers import I, pi, Rational
from sympy.core.symbol import Dummy
from sympy.functions.combinatorial.factorials import (binomial, factorial,
                                                      RisingFactorial)
from sympy.functions.combinatorial.numbers import bernoulli, euler, nC
from sympy.functions.elementary.complexes import Abs, im, re
from sympy.functions.elementary.exponential import exp, log, match_real_imag
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (
    acos, acot, asin, atan, cos, cot, csc, sec, sin, tan,
    _imaginary_unit_as_coefficient)
from sympy.polys.specialpolys import symmetric_poly


def _rewrite_hyperbolics_as_exp(expr):
    return expr.xreplace({h: h.rewrite(exp)
        for h in expr.atoms(HyperbolicFunction)})


@cacheit
def _acosh_table():
    return {
        I: log(I*(1 + sqrt(2))),
        -I: log(-I*(1 + sqrt(2))),
        S.Half: pi/3,
        Rational(-1, 2): pi*Rational(2, 3),
        sqrt(2)/2: pi/4,
        -sqrt(2)/2: pi*Rational(3, 4),
        1/sqrt(2): pi/4,
        -1/sqrt(2): pi*Rational(3, 4),
        sqrt(3)/2: pi/6,
        -sqrt(3)/2: pi*Rational(5, 6),
        (sqrt(3) - 1)/sqrt(2**3): pi*Rational(5, 12),
        -(sqrt(3) - 1)/sqrt(2**3): pi*Rational(7, 12),
        sqrt(2 + sqrt(2))/2: pi/8,
        -sqrt(2 + sqrt(2))/2: pi*Rational(7, 8),
        sqrt(2 - sqrt(2))/2: pi*Rational(3, 8),
        -sqrt(2 - sqrt(2))/2: pi*Rational(5, 8),
        (1 + sqrt(3))/(2*sqrt(2)): pi/12,
        -(1 + sqrt(3))/(2*sqrt(2)): pi*Rational(11, 12),
        (sqrt(5) + 1)/4: pi/5,
        -(sqrt(5) + 1)/4: pi*Rational(4, 5)
    }


@cacheit
def _acsch_table():
    return {
            I: -pi / 2,
            I*(sqrt(2) + sqrt(6)): -pi / 12,
            I*(1 + sqrt(5)): -pi / 10,
            I*2 / sqrt(2 - sqrt(2)): -pi / 8,
            I*2: -pi / 6,
            I*sqrt(2 + 2/sqrt(5)): -pi / 5,
            I*sqrt(2): -pi / 4,
            I*(sqrt(5)-1): -3*pi / 10,
            I*2 / sqrt(3): -pi / 3,
            I*2 / sqrt(2 + sqrt(2)): -3*pi / 8,
            I*sqrt(2 - 2/sqrt(5)): -2*pi / 5,
            I*(sqrt(6) - sqrt(2)): -5*pi / 12,
            S(2): -I*log((1+sqrt(5))/2),
        }


@cacheit
def _asech_table():
        return {
            I: - (pi*I / 2) + log(1 + sqrt(2)),
            -I: (pi*I / 2) + log(1 + sqrt(2)),
            (sqrt(6) - sqrt(2)): pi / 12,
            (sqrt(2) - sqrt(6)): 11*pi / 12,
            sqrt(2 - 2/sqrt(5)): pi / 10,
            -sqrt(2 - 2/sqrt(5)): 9*pi / 10,
            2 / sqrt(2 + sqrt(2)): pi / 8,
            -2 / sqrt(2 + sqrt(2)): 7*pi / 8,
            2 / sqrt(3): pi / 6,
            -2 / sqrt(3): 5*pi / 6,
            (sqrt(5) - 1): pi / 5,
            (1 - sqrt(5)): 4*pi / 5,
            sqrt(2): pi / 4,
            -sqrt(2): 3*pi / 4,
            sqrt(2 + 2/sqrt(5)): 3*pi / 10,
            -sqrt(2 + 2/sqrt(5)): 7*pi / 10,
            S(2): pi / 3,
            -S(2): 2*pi / 3,
            sqrt(2*(2 + sqrt(2))): 3*pi / 8,
            -sqrt(2*(2 + sqrt(2))): 5*pi / 8,
            (1 + sqrt(5)): 2*pi / 5,
            (-1 - sqrt(5)): 3*pi / 5,
            (sqrt(6) + sqrt(2)): 5*pi / 12,
            (-sqrt(6) - sqrt(2)): 7*pi / 12,
            I*S.Infinity: -pi*I / 2,
            I*S.NegativeInfinity: pi*I / 2,
        }

###############################################################################
########################### HYPERBOLIC FUNCTIONS ##############################
###############################################################################


class HyperbolicFunction(DefinedFunction):
    """
    Base class for hyperbolic functions.

    See Also
    ========

    sinh, cosh, tanh, coth
    """

    unbranched = True


def _peeloff_ipi(arg):
    r"""
    Split ARG into two parts, a "rest" and a multiple of $I\pi$.
    This assumes ARG to be an ``Add``.
    The multiple of $I\pi$ returned in the second position is always a ``Rational``.

    Examples
    ========

    >>> from sympy.functions.elementary.hyperbolic import _peeloff_ipi as peel
    >>> from sympy import pi, I
    >>> from sympy.abc import x, y
    >>> peel(x + I*pi/2)
    (x, 1/2)
    >>> peel(x + I*2*pi/3 + I*pi*y)
    (x + I*pi*y + I*pi/6, 1/2)
    """
    ipi = pi*I
    for a in Add.make_args(arg):
        if a == ipi:
            K = S.One
            break
        elif a.is_Mul:
            K, p = a.as_two_terms()
            if p == ipi and K.is_Rational:
                break
    else:
        return arg, S.Zero

    m1 = (K % S.Half)
    m2 = K - m1
    return arg - m2*ipi, m2


class sinh(HyperbolicFunction):
    r"""
    ``sinh(x)`` is the hyperbolic sine of ``x``.

    The hyperbolic sine function is $\frac{e^x - e^{-x}}{2}$.

    Examples
    ========

    >>> from sympy import sinh
    >>> from sympy.abc import x
    >>> sinh(x)
    sinh(x)

    See Also
    ========

    cosh, tanh, asinh
    """

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return cosh(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return asinh

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.NegativeInfinity
            elif arg.is_zero:
                return S.Zero
            elif arg.is_negative:
                return -cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return S.NaN

            i_coeff = _imaginary_unit_as_coefficient(arg)

            if i_coeff is not None:
                return I * sin(i_coeff)
            else:
                if arg.could_extract_minus_sign():
                    return -cls(-arg)

            if arg.is_Add:
                x, m = _peeloff_ipi(arg)
                if m:
                    m = m*pi*I
                    return sinh(m)*cosh(x) + cosh(m)*sinh(x)

            if arg.is_zero:
                return S.Zero

            if arg.func == asinh:
                return arg.args[0]

            if arg.func == acosh:
                x = arg.args[0]
                return sqrt(x - 1) * sqrt(x + 1)

            if arg.func == atanh:
                x = arg.args[0]
                return x/sqrt(1 - x**2)

            if arg.func == acoth:
                x = arg.args[0]
                return 1/(sqrt(x - 1) * sqrt(x + 1))

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        """
        Returns the next term in the Taylor series expansion.
        """
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)

            if len(previous_terms) > 2:
                p = previous_terms[-2]
                return p * x**2 / (n*(n - 1))
            else:
                return x**(n) / factorial(n)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
        """
        Returns this function as a complex coordinate.
        """
        if self.args[0].is_extended_real:
            if deep:
                hints['complex'] = False
                return (self.expand(deep, **hints), S.Zero)
            else:
                return (self, S.Zero)
        if deep:
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            re, im = self.args[0].as_real_imag()
        return (sinh(re)*cos(im), cosh(re)*sin(im))

    def _eval_expand_complex(self, deep=True, **hints):
        re_part, im_part = self.as_real_imag(deep=deep, **hints)
        return re_part + im_part*I

    def _eval_expand_trig(self, deep=True, **hints):
        if deep:
            arg = self.args[0].expand(deep, **hints)
        else:
            arg = self.args[0]
        x = None
        if arg.is_Add: # TODO, implement more if deep stuff here
            x, y = arg.as_two_terms()
        else:
            coeff, terms = arg.as_coeff_Mul(rational=True)
            if coeff is not S.One and coeff.is_Integer and terms is not S.One:
                x = terms
                y = (coeff - 1)*x
        if x is not None:
            return (sinh(x)*cosh(y) + sinh(y)*cosh(x)).expand(trig=True)
        return sinh(arg)

    def _eval_rewrite_as_tractable(self, arg, limitvar=None, **kwargs):
        return (exp(arg) - exp(-arg)) / 2

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        return (exp(arg) - exp(-arg)) / 2

    def _eval_rewrite_as_sin(self, arg, **kwargs):
        return -I * sin(I * arg)

    def _eval_rewrite_as_csc(self, arg, **kwargs):
        return -I / csc(I * arg)

    def _eval_rewrite_as_cosh(self, arg, **kwargs):
        return -I*cosh(arg + pi*I/2)

    def _eval_rewrite_as_tanh(self, arg, **kwargs):
        tanh_half = tanh(S.Half*arg)
        return 2*tanh_half/(1 - tanh_half**2)

    def _eval_rewrite_as_coth(self, arg, **kwargs):
        coth_half = coth(S.Half*arg)
        return 2*coth_half/(coth_half**2 - 1)

    def _eval_rewrite_as_csch(self, arg, **kwargs):
        return 1 / csch(arg)

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if cdir.is_negative else '+')
        if arg0.is_zero:
            return arg
        elif arg0.is_finite:
            return self.func(arg0)
        else:
            return self

    def _eval_is_real(self):
        arg = self.args[0]
        if arg.is_real:
            return True

        # if `im` is of the form n*pi
        # else, check if it is a number
        re, im = arg.as_real_imag()
        return (im%pi).is_zero

    def _eval_is_extended_real(self):
        if self.args[0].is_extended_real:
            return True

    def _eval_is_positive(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_positive

    def _eval_is_negative(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_negative

    def _eval_is_finite(self):
        arg = self.args[0]
        return arg.is_finite

    def _eval_is_zero(self):
        rest, ipi_mult = _peeloff_ipi(self.args[0])
        if rest.is_zero:
            return ipi_mult.is_integer


class cosh(HyperbolicFunction):
    r"""
    ``cosh(x)`` is the hyperbolic cosine of ``x``.

    The hyperbolic cosine function is $\frac{e^x + e^{-x}}{2}$.

    Examples
    ========

    >>> from sympy import cosh
    >>> from sympy.abc import x
    >>> cosh(x)
    cosh(x)

    See Also
    ========

    sinh, tanh, acosh
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return sinh(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        from sympy.functions.elementary.trigonometric import cos
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.Infinity
            elif arg.is_zero:
                return S.One
            elif arg.is_negative:
                return cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return S.NaN

            i_coeff = _imaginary_unit_as_coefficient(arg)

            if i_coeff is not None:
                return cos(i_coeff)
            else:
                if arg.could_extract_minus_sign():
                    return cls(-arg)

            if arg.is_Add:
                x, m = _peeloff_ipi(arg)
                if m:
                    m = m*pi*I
                    return cosh(m)*cosh(x) + sinh(m)*sinh(x)

            if arg.is_zero:
                return S.One

            if arg.func == asinh:
                return sqrt(1 + arg.args[0]**2)

            if arg.func == acosh:
                return arg.args[0]

            if arg.func == atanh:
                return 1/sqrt(1 - arg.args[0]**2)

            if arg.func == acoth:
                x = arg.args[0]
                return x/(sqrt(x - 1) * sqrt(x + 1))

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)

            if len(previous_terms) > 2:
                p = previous_terms[-2]
                return p * x**2 / (n*(n - 1))
            else:
                return x**(n)/factorial(n)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
        if self.args[0].is_extended_real:
            if deep:
                hints['complex'] = False
                return (self.expand(deep, **hints), S.Zero)
            else:
                return (self, S.Zero)
        if deep:
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            re, im = self.args[0].as_real_imag()

        return (cosh(re)*cos(im), sinh(re)*sin(im))

    def _eval_expand_complex(self, deep=True, **hints):
        re_part, im_part = self.as_real_imag(deep=deep, **hints)
        return re_part + im_part*I

    def _eval_expand_trig(self, deep=True, **hints):
        if deep:
            arg = self.args[0].expand(deep, **hints)
        else:
            arg = self.args[0]
        x = None
        if arg.is_Add: # TODO, implement more if deep stuff here
            x, y = arg.as_two_terms()
        else:
            coeff, terms = arg.as_coeff_Mul(rational=True)
            if coeff is not S.One and coeff.is_Integer and terms is not S.One:
                x = terms
                y = (coeff - 1)*x
        if x is not None:
            return (cosh(x)*cosh(y) + sinh(x)*sinh(y)).expand(trig=True)
        return cosh(arg)

    def _eval_rewrite_as_tractable(self, arg, limitvar=None, **kwargs):
        return (exp(arg) + exp(-arg)) / 2

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        return (exp(arg) + exp(-arg)) / 2

    def _eval_rewrite_as_cos(self, arg, **kwargs):
        return cos(I * arg, evaluate=False)

    def _eval_rewrite_as_sec(self, arg, **kwargs):
        return 1 / sec(I * arg, evaluate=False)

    def _eval_rewrite_as_sinh(self, arg, **kwargs):
        return -I*sinh(arg + pi*I/2, evaluate=False)

    def _eval_rewrite_as_tanh(self, arg, **kwargs):
        tanh_half = tanh(S.Half*arg)**2
        return (1 + tanh_half)/(1 - tanh_half)

    def _eval_rewrite_as_coth(self, arg, **kwargs):
        coth_half = coth(S.Half*arg)**2
        return (coth_half + 1)/(coth_half - 1)

    def _eval_rewrite_as_sech(self, arg, **kwargs):
        return 1 / sech(arg)

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0].as_leading_term(x, logx=logx, cdir=cdir)
        arg0 = arg.subs(x, 0)

        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0, dir='-' if cdir.is_negative else '+')
        if arg0.is_zero:
            return S.One
        elif arg0.is_finite:
            return self.func(arg0)
        else:
            return self

    def _eval_is_real(self):
        arg = self.args[0]

        # `cosh(x)` is real for real OR purely imaginary `x`
        if arg.is_real or arg.is_imaginary:
            return True

        # cosh(a+ib) = cos(b)*cosh(a) + i*sin(b)*sinh(a)
        # the imaginary part can be an expression like n*pi
        # if not, check if the imaginary part is a number
        re, im = arg.as_real_imag()
        return (im%pi).is_zero

    def _eval_is_positive(self):
        # cosh(x+I*y) = cos(y)*cosh(x) + I*sin(y)*sinh(x)
        # cosh(z) is positive iff it is real and the real part is positive.
        # So we need sin(y)*sinh(x) = 0 which gives x=0 or y=n*pi
        # Case 1 (y=n*pi): cosh(z) = (-1)**n * cosh(x) -> positive for n even
        # Case 2 (x=0): cosh(z) = cos(y) -> positive when cos(y) is positive
        z = self.args[0]

        x, y = z.as_real_imag()
        ymod = y % (2*pi)

        yzero = ymod.is_zero
        # shortcut if ymod is zero
        if yzero:
            return True

        xzero = x.is_zero
        # shortcut x is not zero
        if xzero is False:
            return yzero

        return fuzzy_or([
                # Case 1:
                yzero,
                # Case 2:
                fuzzy_and([
                    xzero,
                    fuzzy_or([ymod < pi/2, ymod > 3*pi/2])
                ])
            ])


    def _eval_is_nonnegative(self):
        z = self.args[0]

        x, y = z.as_real_imag()
        ymod = y % (2*pi)

        yzero = ymod.is_zero
        # shortcut if ymod is zero
        if yzero:
            return True

        xzero = x.is_zero
        # shortcut x is not zero
        if xzero is False:
            return yzero

        return fuzzy_or([
                # Case 1:
                yzero,
                # Case 2:
                fuzzy_and([
                    xzero,
                    fuzzy_or([ymod <= pi/2, ymod >= 3*pi/2])
                ])
            ])

    def _eval_is_finite(self):
        arg = self.args[0]
        return arg.is_finite

    def _eval_is_zero(self):
        rest, ipi_mult = _peeloff_ipi(self.args[0])
        if ipi_mult and rest.is_zero:
            return (ipi_mult - S.Half).is_integer


class tanh(HyperbolicFunction):
    r"""
    ``tanh(x)`` is the hyperbolic tangent of ``x``.

    The hyperbolic tangent function is $\frac{\sinh(x)}{\cosh(x)}$.

    Examples
    ========

    >>> from sympy import tanh
    >>> from sympy.abc import x
    >>> tanh(x)
    tanh(x)

    See Also
    ========

    sinh, cosh, atanh
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return S.One - tanh(self.args[0])**2
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return atanh

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.One
            elif arg is S.NegativeInfinity:
                return S.NegativeOne
            elif arg.is_zero:
                return S.Zero
            elif arg.is_negative:
                return -cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return S.NaN

            i_coeff = _imaginary_unit_as_coefficient(arg)

            if i_coeff is not None:
                if i_coeff.could_extract_minus_sign():
                    return -I * tan(-i_coeff)
                return I * tan(i_coeff)
            else:
                if arg.could_extract_minus_sign():
                    return -cls(-arg)

            if arg.is_Add:
                x, m = _peeloff_ipi(arg)
                if m:
                    tanhm = tanh(m*pi*I)
                    if tanhm is S.ComplexInfinity:
                        return coth(x)
                    else: # tanhm == 0
                        return tanh(x)

            if arg.is_zero:
                return S.Zero

            if arg.func == asinh:
                x = arg.args[0]
                return x/sqrt(1 + x**2)

            if arg.func == acosh:
                x = arg.args[0]
                return sqrt(x - 1) * sqrt(x + 1) / x

            if arg.func == atanh:
                return arg.args[0]

            if arg.func == acoth:
                return 1/arg.args[0]

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)

            a = 2**(n + 1)

            B = bernoulli(n + 1)
            F = factorial(n + 1)

            return a*(a - 1) * B/F * x**n

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
        if self.args[0].is_extended_real:
            if deep:
                hints['complex'] = False
                return (self.expand(deep, **hints), S.Zero)
            else:
                return (self, S.Zero)
        if deep:
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            re, im = self.args[0].as_real_imag()
        denom = sinh(re)**2 + cos(im)**2
        return (sinh(re)*cosh(re)/denom, sin(im)*cos(im)/denom)

    def _eval_expand_trig(self, **hints):
        arg = self.args[0]
        if arg.is_Add:
            n = len(arg.args)
            TX = [tanh(x, evaluate=False)._eval_expand_trig()
                for x in arg.args]
            p = [0, 0]  # [den, num]
            for i in range(n + 1):
                p[i % 2] += symmetric_poly(i, TX)
            return p[1]/p[0]
        elif arg.is_Mul:
            coeff, terms = arg.as_coeff_Mul()
            if coeff.is_Integer and coeff > 1:
                T = tanh(terms)
                n = [nC(range(coeff), k)*T**k for k in range(1, coeff + 1, 2)]
                d = [nC(range(coeff), k)*T**k for k in range(0, coeff + 1, 2)]
                return Add(*n)/Add(*d)
        return tanh(arg)

    def _eval_rewrite_as_tractable(self, arg, limitvar=None, **kwargs):
        neg_exp, pos_exp = exp(-arg), exp(arg)
        return (pos_exp - neg_exp)/(pos_exp + neg_exp)

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        neg_exp, pos_exp = exp(-arg), exp(arg)
        return (pos_exp - neg_exp)/(pos_exp + neg_exp)

    def _eval_rewrite_as_tan(self, arg, **kwargs):
        return -I * tan(I * arg, evaluate=False)

    def _eval_rewrite_as_cot(self, arg, **kwargs):
        return -I / cot(I * arg, evaluate=False)

    def _eval_rewrite_as_sinh(self, arg, **kwargs):
        return I*sinh(arg)/sinh(pi*I/2 - arg, evaluate=False)

    def _eval_rewrite_as_cosh(self, arg, **kwargs):
        return I*cosh(pi*I/2 - arg, evaluate=False)/cosh(arg)

    def _eval_rewrite_as_coth(self, arg, **kwargs):
        return 1/coth(arg)

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.series.order import Order
        arg = self.args[0].as_leading_term(x)

        if x in arg.free_symbols and Order(1, x).contains(arg):
            return arg
        else:
            return self.func(arg)

    def _eval_is_real(self):
        arg = self.args[0]
        if arg.is_real:
            return True

        re, im = arg.as_real_imag()

        # if denom = 0, tanh(arg) = zoo
        if re == 0 and im % pi == pi/2:
            return None

        # check if im is of the form n*pi/2 to make sin(2*im) = 0
        # if not, im could be a number, return False in that case
        return (im % (pi/2)).is_zero

    def _eval_is_extended_real(self):
        if self.args[0].is_extended_real:
            return True

    def _eval_is_positive(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_positive

    def _eval_is_negative(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_negative

    def _eval_is_finite(self):
        arg = self.args[0]

        re, im = arg.as_real_imag()
        denom = cos(im)**2 + sinh(re)**2
        if denom == 0:
            return False
        elif denom.is_number:
            return True
        if arg.is_extended_real:
            return True

    def _eval_is_zero(self):
        arg = self.args[0]
        if arg.is_zero:
            return True


class coth(HyperbolicFunction):
    r"""
    ``coth(x)`` is the hyperbolic cotangent of ``x``.

    The hyperbolic cotangent function is $\frac{\cosh(x)}{\sinh(x)}$.

    Examples
    ========

    >>> from sympy import coth
    >>> from sympy.abc import x
    >>> coth(x)
    coth(x)

    See Also
    ========

    sinh, cosh, acoth
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return -1/sinh(self.args[0])**2
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return acoth

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.One
            elif arg is S.NegativeInfinity:
                return S.NegativeOne
            elif arg.is_zero:
                return S.ComplexInfinity
            elif arg.is_negative:
                return -cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return S.NaN

            i_coeff = _imaginary_unit_as_coefficient(arg)

            if i_coeff is not None:
                if i_coeff.could_extract_minus_sign():
                    return I * cot(-i_coeff)
                return -I * cot(i_coeff)
            else:
                if arg.could_extract_minus_sign():
                    return -cls(-arg)

            if arg.is_Add:
                x, m = _peeloff_ipi(arg)
                if m:
                    cothm = coth(m*pi*I)
                    if cothm is S.ComplexInfinity:
                        return coth(x)
                    else: # cothm == 0
                        return tanh(x)

            if arg.is_zero:
                return S.ComplexInfinity

            if arg.func == asinh:
                x = arg.args[0]
                return sqrt(1 + x**2)/x

            if arg.func == acosh:
                x = arg.args[0]
                return x/(sqrt(x - 1) * sqrt(x + 1))

            if arg.func == atanh:
                return 1/arg.args[0]

            if arg.func == acoth:
                return arg.args[0]

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return 1 / sympify(x)
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)

            B = bernoulli(n + 1)
            F = factorial(n + 1)

            return 2**(n + 1) * B/F * x**n

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def as_real_imag(self, deep=True, **hints):
        from sympy.functions.elementary.trigonometric import (cos, sin)
        if self.args[0].is_extended_real:
            if deep:
                hints['complex'] = False
                return (self.expand(deep, **hints), S.Zero)
            else:
                return (self, S.Zero)
        if deep:
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            re, im = self.args[0].as_real_imag()
        denom = sinh(re)**2 + sin(im)**2
        return (sinh(re)*cosh(re)/denom, -sin(im)*cos(im)/denom)

    def _eval_rewrite_as_tractable(self, arg, limitvar=None, **kwargs):
        neg_exp, pos_exp = exp(-arg), exp(arg)
        return (pos_exp + neg_exp)/(pos_exp - neg_exp)

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        neg_exp, pos_exp = exp(-arg), exp(arg)
        return (pos_exp + neg_exp)/(pos_exp - neg_exp)

    def _eval_rewrite_as_sinh(self, arg, **kwargs):
        return -I*sinh(pi*I/2 - arg, evaluate=False)/sinh(arg)

    def _eval_rewrite_as_cosh(self, arg, **kwargs):
        return -I*cosh(arg)/cosh(pi*I/2 - arg, evaluate=False)

    def _eval_rewrite_as_tanh(self, arg, **kwargs):
        return 1/tanh(arg)

    def _eval_is_positive(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_positive

    def _eval_is_negative(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_negative

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.series.order import Order
        arg = self.args[0].as_leading_term(x)

        if x in arg.free_symbols and Order(1, x).contains(arg):
            return 1/arg
        else:
            return self.func(arg)

    def _eval_expand_trig(self, **hints):
        arg = self.args[0]
        if arg.is_Add:
            CX = [coth(x, evaluate=False)._eval_expand_trig() for x in arg.args]
            p = [[], []]
            n = len(arg.args)
            for i in range(n, -1, -1):
                p[(n - i) % 2].append(symmetric_poly(i, CX))
            return Add(*p[0])/Add(*p[1])
        elif arg.is_Mul:
            coeff, x = arg.as_coeff_Mul(rational=True)
            if coeff.is_Integer and coeff > 1:
                c = coth(x, evaluate=False)
                p = [[], []]
                for i in range(coeff, -1, -1):
                    p[(coeff - i) % 2].append(binomial(coeff, i)*c**i)
                return Add(*p[0])/Add(*p[1])
        return coth(arg)


class ReciprocalHyperbolicFunction(HyperbolicFunction):
    """Base class for reciprocal functions of hyperbolic functions. """

    #To be defined in class
    _reciprocal_of = None
    _is_even: FuzzyBool = None
    _is_odd: FuzzyBool = None

    @classmethod
    def eval(cls, arg):
        if arg.could_extract_minus_sign():
            if cls._is_even:
                return cls(-arg)
            if cls._is_odd:
                return -cls(-arg)

        t = cls._reciprocal_of.eval(arg)
        if hasattr(arg, 'inverse') and arg.inverse() == cls:
            return arg.args[0]
        return 1/t if t is not None else t

    def _call_reciprocal(self, method_name, *args, **kwargs):
        # Calls method_name on _reciprocal_of
        o = self._reciprocal_of(self.args[0])
        return getattr(o, method_name)(*args, **kwargs)

    def _calculate_reciprocal(self, method_name, *args, **kwargs):
        # If calling method_name on _reciprocal_of returns a value != None
        # then return the reciprocal of that value
        t = self._call_reciprocal(method_name, *args, **kwargs)
        return 1/t if t is not None else t

    def _rewrite_reciprocal(self, method_name, arg):
        # Special handling for rewrite functions. If reciprocal rewrite returns
        # unmodified expression, then return None
        t = self._call_reciprocal(method_name, arg)
        if t is not None and t != self._reciprocal_of(arg):
            return 1/t

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_exp", arg)

    def _eval_rewrite_as_tractable(self, arg, limitvar=None, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_tractable", arg)

    def _eval_rewrite_as_tanh(self, arg, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_tanh", arg)

    def _eval_rewrite_as_coth(self, arg, **kwargs):
        return self._rewrite_reciprocal("_eval_rewrite_as_coth", arg)

    def as_real_imag(self, deep = True, **hints):
        return (1 / self._reciprocal_of(self.args[0])).as_real_imag(deep, **hints)

    def _eval_conjugate(self):
        return self.func(self.args[0].conjugate())

    def _eval_expand_complex(self, deep=True, **hints):
        re_part, im_part = self.as_real_imag(deep=True, **hints)
        return re_part + I*im_part

    def _eval_expand_trig(self, **hints):
        return self._calculate_reciprocal("_eval_expand_trig", **hints)

    def _eval_as_leading_term(self, x, logx, cdir):
        return (1/self._reciprocal_of(self.args[0]))._eval_as_leading_term(x, logx=logx, cdir=cdir)

    def _eval_is_extended_real(self):
        return self._reciprocal_of(self.args[0]).is_extended_real

    def _eval_is_finite(self):
        return (1/self._reciprocal_of(self.args[0])).is_finite


class csch(ReciprocalHyperbolicFunction):
    r"""
    ``csch(x)`` is the hyperbolic cosecant of ``x``.

    The hyperbolic cosecant function is $\frac{2}{e^x - e^{-x}}$

    Examples
    ========

    >>> from sympy import csch
    >>> from sympy.abc import x
    >>> csch(x)
    csch(x)

    See Also
    ========

    sinh, cosh, tanh, sech, asinh, acosh
    """

    _reciprocal_of = sinh
    _is_odd = True

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function
        """
        if argindex == 1:
            return -coth(self.args[0]) * csch(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        """
        Returns the next term in the Taylor series expansion
        """
        if n == 0:
            return 1/sympify(x)
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)

            B = bernoulli(n + 1)
            F = factorial(n + 1)

            return 2 * (1 - 2**n) * B/F * x**n

    def _eval_rewrite_as_sin(self, arg, **kwargs):
        return I / sin(I * arg, evaluate=False)

    def _eval_rewrite_as_csc(self, arg, **kwargs):
        return I * csc(I * arg, evaluate=False)

    def _eval_rewrite_as_cosh(self, arg, **kwargs):
        return I / cosh(arg + I * pi / 2, evaluate=False)

    def _eval_rewrite_as_sinh(self, arg, **kwargs):
        return 1 / sinh(arg)

    def _eval_is_positive(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_positive

    def _eval_is_negative(self):
        if self.args[0].is_extended_real:
            return self.args[0].is_negative


class sech(ReciprocalHyperbolicFunction):
    r"""
    ``sech(x)`` is the hyperbolic secant of ``x``.

    The hyperbolic secant function is $\frac{2}{e^x + e^{-x}}$

    Examples
    ========

    >>> from sympy import sech
    >>> from sympy.abc import x
    >>> sech(x)
    sech(x)

    See Also
    ========

    sinh, cosh, tanh, coth, csch, asinh, acosh
    """

    _reciprocal_of = cosh
    _is_even = True

    def fdiff(self, argindex=1):
        if argindex == 1:
            return - tanh(self.args[0])*sech(self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)
            return euler(n) / factorial(n) * x**(n)

    def _eval_rewrite_as_cos(self, arg, **kwargs):
        return 1 / cos(I * arg, evaluate=False)

    def _eval_rewrite_as_sec(self, arg, **kwargs):
        return sec(I * arg, evaluate=False)

    def _eval_rewrite_as_sinh(self, arg, **kwargs):
        return I / sinh(arg + I * pi /2, evaluate=False)

    def _eval_rewrite_as_cosh(self, arg, **kwargs):
        return 1 / cosh(arg)

    def _eval_is_positive(self):
        if self.args[0].is_extended_real:
            return True


###############################################################################
############################# HYPERBOLIC INVERSES #############################
###############################################################################

class InverseHyperbolicFunction(DefinedFunction):
    """Base class for inverse hyperbolic functions."""

    pass


class asinh(InverseHyperbolicFunction):
    """
    ``asinh(x)`` is the inverse hyperbolic sine of ``x``.

    The inverse hyperbolic sine function.

    Examples
    ========

    >>> from sympy import asinh
    >>> from sympy.abc import x
    >>> asinh(x).diff(x)
    1/sqrt(x**2 + 1)
    >>> asinh(1)
    log(1 + sqrt(2))

    See Also
    ========

    acosh, atanh, sinh
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 1/sqrt(self.args[0]**2 + 1)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.NegativeInfinity
            elif arg.is_zero:
                return S.Zero
            elif arg is S.One:
                return log(sqrt(2) + 1)
            elif arg is S.NegativeOne:
                return log(sqrt(2) - 1)
            elif arg.is_negative:
                return -cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return S.ComplexInfinity

            if arg.is_zero:
                return S.Zero

            i_coeff = _imaginary_unit_as_coefficient(arg)

            if i_coeff is not None:
                return I * asin(i_coeff)
            else:
                if arg.could_extract_minus_sign():
                    return -cls(-arg)

        if isinstance(arg, sinh) and arg.args[0].is_number:
            z = arg.args[0]
            if z.is_real:
                return z
            r, i = match_real_imag(z)
            if r is not None and i is not None:
                f = floor((i + pi/2)/pi)
                m = z - I*pi*f
                even = f.is_even
                if even is True:
                    return m
                elif even is False:
                    return -m

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) >= 2 and n > 2:
                p = previous_terms[-2]
                return -p * (n - 2)**2/(n*(n - 1)) * x**2
            else:
                k = (n - 1) // 2
                R = RisingFactorial(S.Half, k)
                F = factorial(k)
                return S.NegativeOne**k * R / F * x**n / n

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        if x0.is_zero:
            return arg.as_leading_term(x)

        if x0 is S.NaN:
            expr = self.func(arg.as_leading_term(x))
            if expr.is_finite:
                return expr
            else:
                return self

        # Handling branch points
        if x0 in (-I, I, S.ComplexInfinity):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        # Handling points lying on branch cuts (-I*oo, -I) U (I, I*oo)
        if (1 + x0**2).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if re(ndir).is_positive:
                if im(x0).is_negative:
                    return -self.func(x0) - I*pi
            elif re(ndir).is_negative:
                if im(x0).is_positive:
                    return -self.func(x0) + I*pi
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):  # asinh
        arg = self.args[0]
        arg0 = arg.subs(x, 0)

        # Handling branch points
        if arg0 in (I, -I):
            return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)

        res = super()._eval_nseries(x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res

        # Handling points lying on branch cuts (-I*oo, -I) U (I, I*oo)
        if (1 + arg0**2).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if re(ndir).is_positive:
                if im(arg0).is_negative:
                    return -res - I*pi
            elif re(ndir).is_negative:
                if im(arg0).is_positive:
                    return -res + I*pi
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def _eval_rewrite_as_log(self, x, **kwargs):
        return log(x + sqrt(x**2 + 1))

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_atanh(self, x, **kwargs):
        return atanh(x/sqrt(1 + x**2))

    def _eval_rewrite_as_acosh(self, x, **kwargs):
        ix = I*x
        return I*(sqrt(1 - ix)/sqrt(ix - 1) * acosh(ix) - pi/2)

    def _eval_rewrite_as_asin(self, x, **kwargs):
        return -I * asin(I * x, evaluate=False)

    def _eval_rewrite_as_acos(self, x, **kwargs):
        return I * acos(I * x, evaluate=False) - I*pi/2

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return sinh

    def _eval_is_zero(self):
        return self.args[0].is_zero

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real

    def _eval_is_finite(self):
        return self.args[0].is_finite


class acosh(InverseHyperbolicFunction):
    """
    ``acosh(x)`` is the inverse hyperbolic cosine of ``x``.

    The inverse hyperbolic cosine function.

    Examples
    ========

    >>> from sympy import acosh
    >>> from sympy.abc import x
    >>> acosh(x).diff(x)
    1/(sqrt(x - 1)*sqrt(x + 1))
    >>> acosh(1)
    0

    See Also
    ========

    asinh, atanh, cosh
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            arg = self.args[0]
            return 1/(sqrt(arg - 1)*sqrt(arg + 1))
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.Infinity
            elif arg.is_zero:
                return pi*I / 2
            elif arg is S.One:
                return S.Zero
            elif arg is S.NegativeOne:
                return pi*I

        if arg.is_number:
            cst_table = _acosh_table()

            if arg in cst_table:
                if arg.is_extended_real:
                    return cst_table[arg]*I
                return cst_table[arg]

        if arg is S.ComplexInfinity:
            return S.ComplexInfinity
        if arg == I*S.Infinity:
            return S.Infinity + I*pi/2
        if arg == -I*S.Infinity:
            return S.Infinity - I*pi/2

        if arg.is_zero:
            return pi*I*S.Half

        if isinstance(arg, cosh) and arg.args[0].is_number:
            z = arg.args[0]
            if z.is_real:
                return Abs(z)
            r, i = match_real_imag(z)
            if r is not None and i is not None:
                f = floor(i/pi)
                m = z - I*pi*f
                even = f.is_even
                if even is True:
                    if r.is_nonnegative:
                        return m
                    elif r.is_negative:
                        return -m
                elif even is False:
                    m -= I*pi
                    if r.is_nonpositive:
                        return -m
                    elif r.is_positive:
                        return m

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return I*pi/2
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) >= 2 and n > 2:
                p = previous_terms[-2]
                return p * (n - 2)**2/(n*(n - 1)) * x**2
            else:
                k = (n - 1) // 2
                R = RisingFactorial(S.Half, k)
                F = factorial(k)
                return -R / F * I * x**n / n

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        # Handling branch points
        if x0 in (-S.One, S.Zero, S.One, S.ComplexInfinity):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)

        if x0 is S.NaN:
            expr = self.func(arg.as_leading_term(x))
            if expr.is_finite:
                return expr
            else:
                return self

        # Handling points lying on branch cuts (-oo, 1)
        if (x0 - 1).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if (x0 + 1).is_negative:
                    return self.func(x0) - 2*I*pi
                return -self.func(x0)
            elif not im(ndir).is_positive:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):  # acosh
        arg = self.args[0]
        arg0 = arg.subs(x, 0)

        # Handling branch points
        if arg0 in (S.One, S.NegativeOne):
            return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)

        res = super()._eval_nseries(x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res

        # Handling points lying on branch cuts (-oo, 1)
        if (arg0 - 1).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if (arg0 + 1).is_negative:
                    return res - 2*I*pi
                return -res
            elif not im(ndir).is_positive:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def _eval_rewrite_as_log(self, x, **kwargs):
        return log(x + sqrt(x + 1) * sqrt(x - 1))

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_acos(self, x, **kwargs):
        return sqrt(x - 1)/sqrt(1 - x) * acos(x)

    def _eval_rewrite_as_asin(self, x, **kwargs):
        return sqrt(x - 1)/sqrt(1 - x) * (pi/2 - asin(x))

    def _eval_rewrite_as_asinh(self, x, **kwargs):
        return sqrt(x - 1)/sqrt(1 - x) * (pi/2 + I*asinh(I*x, evaluate=False))

    def _eval_rewrite_as_atanh(self, x, **kwargs):
        sxm1 = sqrt(x - 1)
        s1mx = sqrt(1 - x)
        sx2m1 = sqrt(x**2 - 1)
        return (pi/2*sxm1/s1mx*(1 - x * sqrt(1/x**2)) +
                sxm1*sqrt(x + 1)/sx2m1 * atanh(sx2m1/x))

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return cosh

    def _eval_is_zero(self):
        if (self.args[0] - 1).is_zero:
            return True

    def _eval_is_extended_real(self):
        return fuzzy_and([self.args[0].is_extended_real, (self.args[0] - 1).is_extended_nonnegative])

    def _eval_is_finite(self):
        return self.args[0].is_finite


class atanh(InverseHyperbolicFunction):
    """
    ``atanh(x)`` is the inverse hyperbolic tangent of ``x``.

    The inverse hyperbolic tangent function.

    Examples
    ========

    >>> from sympy import atanh
    >>> from sympy.abc import x
    >>> atanh(x).diff(x)
    1/(1 - x**2)

    See Also
    ========

    asinh, acosh, tanh
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 1/(1 - self.args[0]**2)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg.is_zero:
                return S.Zero
            elif arg is S.One:
                return S.Infinity
            elif arg is S.NegativeOne:
                return S.NegativeInfinity
            elif arg is S.Infinity:
                return -I * atan(arg)
            elif arg is S.NegativeInfinity:
                return I * atan(-arg)
            elif arg.is_negative:
                return -cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                from sympy.calculus.accumulationbounds import AccumBounds
                return I*AccumBounds(-pi/2, pi/2)

            i_coeff = _imaginary_unit_as_coefficient(arg)

            if i_coeff is not None:
                return I * atan(i_coeff)
            else:
                if arg.could_extract_minus_sign():
                    return -cls(-arg)

        if arg.is_zero:
            return S.Zero

        if isinstance(arg, tanh) and arg.args[0].is_number:
            z = arg.args[0]
            if z.is_real:
                return z
            r, i = match_real_imag(z)
            if r is not None and i is not None:
                f = floor(2*i/pi)
                even = f.is_even
                m = z - I*f*pi/2
                if even is True:
                    return m
                elif even is False:
                    return m - I*pi/2

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            return x**n / n

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        if x0.is_zero:
            return arg.as_leading_term(x)
        if x0 is S.NaN:
            expr = self.func(arg.as_leading_term(x))
            if expr.is_finite:
                return expr
            else:
                return self

        # Handling branch points
        if x0 in (-S.One, S.One, S.ComplexInfinity):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        # Handling points lying on branch cuts (-oo, -1] U [1, oo)
        if (1 - x0**2).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if x0.is_negative:
                    return self.func(x0) - I*pi
            elif im(ndir).is_positive:
                if x0.is_positive:
                    return self.func(x0) + I*pi
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):  # atanh
        arg = self.args[0]
        arg0 = arg.subs(x, 0)

        # Handling branch points
        if arg0 in (S.One, S.NegativeOne):
            return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)

        res = super()._eval_nseries(x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res

        # Handling points lying on branch cuts (-oo, -1] U [1, oo)
        if (1 - arg0**2).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if arg0.is_negative:
                    return res - I*pi
            elif im(ndir).is_positive:
                if arg0.is_positive:
                    return res + I*pi
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def _eval_rewrite_as_log(self, x, **kwargs):
        return (log(1 + x) - log(1 - x)) / 2

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_asinh(self, x, **kwargs):
        f = sqrt(1/(x**2 - 1))
        return (pi*x/(2*sqrt(-x**2)) -
                sqrt(-x)*sqrt(1 - x**2)/sqrt(x)*f*asinh(f))

    def _eval_is_zero(self):
        if self.args[0].is_zero:
            return True

    def _eval_is_extended_real(self):
        return fuzzy_and([self.args[0].is_extended_real, (1 - self.args[0]).is_nonnegative, (self.args[0] + 1).is_nonnegative])

    def _eval_is_finite(self):
        return fuzzy_not(fuzzy_or([(self.args[0] - 1).is_zero, (self.args[0] + 1).is_zero]))

    def _eval_is_imaginary(self):
        return self.args[0].is_imaginary

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return tanh


class acoth(InverseHyperbolicFunction):
    """
    ``acoth(x)`` is the inverse hyperbolic cotangent of ``x``.

    The inverse hyperbolic cotangent function.

    Examples
    ========

    >>> from sympy import acoth
    >>> from sympy.abc import x
    >>> acoth(x).diff(x)
    1/(1 - x**2)

    See Also
    ========

    asinh, acosh, coth
    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            return 1/(1 - self.args[0]**2)
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Zero
            elif arg is S.NegativeInfinity:
                return S.Zero
            elif arg.is_zero:
                return pi*I / 2
            elif arg is S.One:
                return S.Infinity
            elif arg is S.NegativeOne:
                return S.NegativeInfinity
            elif arg.is_negative:
                return -cls(-arg)
        else:
            if arg is S.ComplexInfinity:
                return S.Zero

            i_coeff = _imaginary_unit_as_coefficient(arg)

            if i_coeff is not None:
                return -I * acot(i_coeff)
            else:
                if arg.could_extract_minus_sign():
                    return -cls(-arg)

        if arg.is_zero:
            return pi*I*S.Half

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return -I*pi/2
        elif n < 0 or n % 2 == 0:
            return S.Zero
        else:
            x = sympify(x)
            return x**n / n

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        if x0 is S.ComplexInfinity:
            return (1/arg).as_leading_term(x)
        if x0 is S.NaN:
            expr = self.func(arg.as_leading_term(x))
            if expr.is_finite:
                return expr
            else:
                return self

        # Handling branch points
        if x0 in (-S.One, S.One, S.Zero):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        # Handling points lying on branch cuts [-1, 1]
        if x0.is_real and (1 - x0**2).is_positive:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if x0.is_positive:
                    return self.func(x0) + I*pi
            elif im(ndir).is_positive:
                if x0.is_negative:
                    return self.func(x0) - I*pi
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):  # acoth
        arg = self.args[0]
        arg0 = arg.subs(x, 0)

        # Handling branch points
        if arg0 in (S.One, S.NegativeOne):
            return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)

        res = super()._eval_nseries(x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res

        # Handling points lying on branch cuts [-1, 1]
        if arg0.is_real and (1 - arg0**2).is_positive:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_negative:
                if arg0.is_positive:
                    return res + I*pi
            elif im(ndir).is_positive:
                if arg0.is_negative:
                    return res - I*pi
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def _eval_rewrite_as_log(self, x, **kwargs):
        return (log(1 + 1/x) - log(1 - 1/x)) / 2

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_atanh(self, x, **kwargs):
        return atanh(1/x)

    def _eval_rewrite_as_asinh(self, x, **kwargs):
        return (pi*I/2*(sqrt((x - 1)/x)*sqrt(x/(x - 1)) - sqrt(1 + 1/x)*sqrt(x/(x + 1))) +
                x*sqrt(1/x**2)*asinh(sqrt(1/(x**2 - 1))))

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return coth

    def _eval_is_extended_real(self):
        return fuzzy_and([self.args[0].is_extended_real, fuzzy_or([(self.args[0] - 1).is_extended_nonnegative, (self.args[0] + 1).is_extended_nonpositive])])

    def _eval_is_finite(self):
        return fuzzy_not(fuzzy_or([(self.args[0] - 1).is_zero, (self.args[0] + 1).is_zero]))


class asech(InverseHyperbolicFunction):
    """
    ``asech(x)`` is the inverse hyperbolic secant of ``x``.

    The inverse hyperbolic secant function.

    Examples
    ========

    >>> from sympy import asech, sqrt, S
    >>> from sympy.abc import x
    >>> asech(x).diff(x)
    -1/(x*sqrt(1 - x**2))
    >>> asech(1).diff(x)
    0
    >>> asech(1)
    0
    >>> asech(S(2))
    I*pi/3
    >>> asech(-sqrt(2))
    3*I*pi/4
    >>> asech((sqrt(6) - sqrt(2)))
    I*pi/12

    See Also
    ========

    asinh, atanh, cosh, acoth

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hyperbolic_function
    .. [2] https://dlmf.nist.gov/4.37
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcSech/

    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            z = self.args[0]
            return -1/(z*sqrt(1 - z**2))
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return pi*I / 2
            elif arg is S.NegativeInfinity:
                return pi*I / 2
            elif arg.is_zero:
                return S.Infinity
            elif arg is S.One:
                return S.Zero
            elif arg is S.NegativeOne:
                return pi*I

        if arg.is_number:
            cst_table = _asech_table()

            if arg in cst_table:
                if arg.is_extended_real:
                    return cst_table[arg]*I
                return cst_table[arg]

        if arg is S.ComplexInfinity:
            from sympy.calculus.accumulationbounds import AccumBounds
            return I*AccumBounds(-pi/2, pi/2)

        if arg.is_zero:
            return S.Infinity

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return log(2 / x)
        elif n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 2 and n > 2:
                p = previous_terms[-2]
                return p * ((n - 1)*(n-2)) * x**2/(4 * (n//2)**2)
            else:
                k = n // 2
                R = RisingFactorial(S.Half, k) * n
                F = factorial(k) * n // 2 * n // 2
                return -1 * R / F * x**n / 4

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        # Handling branch points
        if x0 in (-S.One, S.Zero, S.One, S.ComplexInfinity):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)

        if x0 is S.NaN:
            expr = self.func(arg.as_leading_term(x))
            if expr.is_finite:
                return expr
            else:
                return self

        # Handling points lying on branch cuts (-oo, 0] U (1, oo)
        if x0.is_negative or (1 - x0).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_positive:
                if x0.is_positive or (x0 + 1).is_negative:
                    return -self.func(x0)
                return self.func(x0) - 2*I*pi
            elif not im(ndir).is_negative:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):  # asech
        from sympy.series.order import O
        arg = self.args[0]
        arg0 = arg.subs(x, 0)

        # Handling branch points
        if arg0 is S.One:
            t = Dummy('t', positive=True)
            ser = asech(S.One - t**2).rewrite(log).nseries(t, 0, 2*n)
            arg1 = S.One - self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            if not g.is_meromorphic(x, 0):   # cannot be expanded
                return O(1) if n == 0 else O(sqrt(x))
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO()*sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        if arg0 is S.NegativeOne:
            t = Dummy('t', positive=True)
            ser = asech(S.NegativeOne + t**2).rewrite(log).nseries(t, 0, 2*n)
            arg1 = S.One + self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            if not g.is_meromorphic(x, 0):   # cannot be expanded
                return O(1) if n == 0 else I*pi + O(sqrt(x))
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO()*sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        res = super()._eval_nseries(x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res

        # Handling points lying on branch cuts (-oo, 0] U (1, oo)
        if arg0.is_negative or (1 - arg0).is_negative:
            ndir = arg.dir(x, cdir if cdir else 1)
            if im(ndir).is_positive:
                if arg0.is_positive or (arg0 + 1).is_negative:
                    return -res
                return res - 2*I*pi
            elif not im(ndir).is_negative:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return sech

    def _eval_rewrite_as_log(self, arg, **kwargs):
        return log(1/arg + sqrt(1/arg - 1) * sqrt(1/arg + 1))

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_acosh(self, arg, **kwargs):
        return acosh(1/arg)

    def _eval_rewrite_as_asinh(self, arg, **kwargs):
        return sqrt(1/arg - 1)/sqrt(1 - 1/arg)*(I*asinh(I/arg, evaluate=False)
                                                + pi*S.Half)

    def _eval_rewrite_as_atanh(self, x, **kwargs):
        return (I*pi*(1 - sqrt(x)*sqrt(1/x) - I/2*sqrt(-x)/sqrt(x) - I/2*sqrt(x**2)/sqrt(-x**2))
                + sqrt(1/(x + 1))*sqrt(x + 1)*atanh(sqrt(1 - x**2)))

    def _eval_rewrite_as_acsch(self, x, **kwargs):
        return sqrt(1/x - 1)/sqrt(1 - 1/x)*(pi/2 - I*acsch(I*x, evaluate=False))

    def _eval_is_extended_real(self):
        return fuzzy_and([self.args[0].is_extended_real, self.args[0].is_nonnegative, (1 - self.args[0]).is_nonnegative])

    def _eval_is_finite(self):
        return fuzzy_not(self.args[0].is_zero)


class acsch(InverseHyperbolicFunction):
    """
    ``acsch(x)`` is the inverse hyperbolic cosecant of ``x``.

    The inverse hyperbolic cosecant function.

    Examples
    ========

    >>> from sympy import acsch, sqrt, I
    >>> from sympy.abc import x
    >>> acsch(x).diff(x)
    -1/(x**2*sqrt(1 + x**(-2)))
    >>> acsch(1).diff(x)
    0
    >>> acsch(1)
    log(1 + sqrt(2))
    >>> acsch(I)
    -I*pi/2
    >>> acsch(-2*I)
    I*pi/6
    >>> acsch(I*(sqrt(6) - sqrt(2)))
    -5*I*pi/12

    See Also
    ========

    asinh

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hyperbolic_function
    .. [2] https://dlmf.nist.gov/4.37
    .. [3] https://functions.wolfram.com/ElementaryFunctions/ArcCsch/

    """

    def fdiff(self, argindex=1):
        if argindex == 1:
            z = self.args[0]
            return -1/(z**2*sqrt(1 + 1/z**2))
        else:
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, arg):
        if arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg is S.Infinity:
                return S.Zero
            elif arg is S.NegativeInfinity:
                return S.Zero
            elif arg.is_zero:
                return S.ComplexInfinity
            elif arg is S.One:
                return log(1 + sqrt(2))
            elif arg is S.NegativeOne:
                return - log(1 + sqrt(2))

        if arg.is_number:
            cst_table = _acsch_table()

            if arg in cst_table:
                return cst_table[arg]*I

        if arg is S.ComplexInfinity:
            return S.Zero

        if arg.is_infinite:
            return S.Zero

        if arg.is_zero:
            return S.ComplexInfinity

        if arg.could_extract_minus_sign():
            return -cls(-arg)

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        if n == 0:
            return log(2 / x)
        elif n < 0 or n % 2 == 1:
            return S.Zero
        else:
            x = sympify(x)
            if len(previous_terms) > 2 and n > 2:
                p = previous_terms[-2]
                return -p * ((n - 1)*(n-2)) * x**2/(4 * (n//2)**2)
            else:
                k = n // 2
                R = RisingFactorial(S.Half, k) *  n
                F = factorial(k) * n // 2 * n // 2
                return S.NegativeOne**(k +1) * R / F * x**n / 4

    def _eval_as_leading_term(self, x, logx, cdir):
        arg = self.args[0]
        x0 = arg.subs(x, 0).cancel()
        # Handling branch points
        if x0 in (-I, I, S.Zero):
            return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)

        if x0 is S.NaN:
            expr = self.func(arg.as_leading_term(x))
            if expr.is_finite:
                return expr
            else:
                return self

        if x0 is S.ComplexInfinity:
            return (1/arg).as_leading_term(x)
        # Handling points lying on branch cuts (-I, I)
        if x0.is_imaginary and (1 + x0**2).is_positive:
            ndir = arg.dir(x, cdir if cdir else 1)
            if re(ndir).is_positive:
                if im(x0).is_positive:
                    return -self.func(x0) - I*pi
            elif re(ndir).is_negative:
                if im(x0).is_negative:
                    return -self.func(x0) + I*pi
            else:
                return self.rewrite(log)._eval_as_leading_term(x, logx=logx, cdir=cdir)
        return self.func(x0)

    def _eval_nseries(self, x, n, logx, cdir=0):  # acsch
        from sympy.series.order import O
        arg = self.args[0]
        arg0 = arg.subs(x, 0)

        # Handling branch points
        if arg0 is I:
            t = Dummy('t', positive=True)
            ser = acsch(I + t**2).rewrite(log).nseries(t, 0, 2*n)
            arg1 = -I + self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            if not g.is_meromorphic(x, 0):   # cannot be expanded
                return O(1) if n == 0 else -I*pi/2 + O(sqrt(x))
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO()*sqrt(f)).expand()
            res = ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)
            return res

        if arg0 == S.NegativeOne*I:
            t = Dummy('t', positive=True)
            ser = acsch(-I + t**2).rewrite(log).nseries(t, 0, 2*n)
            arg1 = I + self.args[0]
            f = arg1.as_leading_term(x)
            g = (arg1 - f)/ f
            if not g.is_meromorphic(x, 0):   # cannot be expanded
                return O(1) if n == 0 else I*pi/2 + O(sqrt(x))
            res1 = sqrt(S.One + g)._eval_nseries(x, n=n, logx=logx)
            res = (res1.removeO()*sqrt(f)).expand()
            return ser.removeO().subs(t, res).expand().powsimp() + O(x**n, x)

        res = super()._eval_nseries(x, n=n, logx=logx)
        if arg0 is S.ComplexInfinity:
            return res

        # Handling points lying on branch cuts (-I, I)
        if arg0.is_imaginary and (1 + arg0**2).is_positive:
            ndir = self.args[0].dir(x, cdir if cdir else 1)
            if re(ndir).is_positive:
                if im(arg0).is_positive:
                    return -res - I*pi
            elif re(ndir).is_negative:
                if im(arg0).is_negative:
                    return -res + I*pi
            else:
                return self.rewrite(log)._eval_nseries(x, n, logx=logx, cdir=cdir)
        return res

    def inverse(self, argindex=1):
        """
        Returns the inverse of this function.
        """
        return csch

    def _eval_rewrite_as_log(self, arg, **kwargs):
        return log(1/arg + sqrt(1/arg**2 + 1))

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    def _eval_rewrite_as_asinh(self, arg, **kwargs):
        return asinh(1/arg)

    def _eval_rewrite_as_acosh(self, arg, **kwargs):
        return I*(sqrt(1 - I/arg)/sqrt(I/arg - 1)*
                                acosh(I/arg, evaluate=False) - pi*S.Half)

    def _eval_rewrite_as_atanh(self, arg, **kwargs):
        arg2 = arg**2
        arg2p1 = arg2 + 1
        return sqrt(-arg2)/arg*(pi*S.Half -
                                sqrt(-arg2p1**2)/arg2p1*atanh(sqrt(arg2p1)))

    def _eval_is_zero(self):
        return self.args[0].is_infinite

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_real

    def _eval_is_finite(self):
        return fuzzy_not(self.args[0].is_zero)
<|endoftext|>from __future__ import annotations
from itertools import product

from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import (DefinedFunction, ArgumentIndexError, expand_log,
    expand_mul, FunctionClass, PoleError, expand_multinomial, expand_complex)
from sympy.core.logic import fuzzy_and, fuzzy_not, fuzzy_or
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, Rational, pi, I
from sympy.core.parameters import global_parameters
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Wild, Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import arg, unpolarify, im, re, Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.ntheory import multiplicity, perfect_power
from sympy.ntheory.factor_ import factorint

# NOTE IMPORTANT
# The series expansion code in this file is an important part of the gruntz
# algorithm for determining limits. _eval_nseries has to return a generalized
# power series with coefficients in C(log(x), log).
# In more detail, the result of _eval_nseries(self, x, n) must be
#   c_0*x**e_0 + ... (finitely many terms)
# where e_i are numbers (not necessarily integers) and c_i involve only
# numbers, the function log, and log(x). [This also means it must not contain
# log(x(1+p)), this *has* to be expanded to log(x)+log(1+p) if x.is_positive and
# p.is_positive.]


class ExpBase(DefinedFunction):

    unbranched = True
    _singularities = (S.ComplexInfinity,)

    @property
    def kind(self):
        return self.exp.kind

    def inverse(self, argindex=1):
        """
        Returns the inverse function of ``exp(x)``.
        """
        return log

    def as_numer_denom(self):
        """
        Returns this with a positive exponent as a 2-tuple (a fraction).

        Examples
        ========

        >>> from sympy import exp
        >>> from sympy.abc import x
        >>> exp(-x).as_numer_denom()
        (1, exp(x))
        >>> exp(x).as_numer_denom()
        (exp(x), 1)
        """
        # this should be the same as Pow.as_numer_denom wrt
        # exponent handling
        if not self.is_commutative:
            return self, S.One
        exp = self.exp
        neg_exp = exp.is_negative
        if not neg_exp and not (-exp).is_negative:
            neg_exp = exp.could_extract_minus_sign()
        if neg_exp:
            return S.One, self.func(-exp)
        return self, S.One

    @property
    def exp(self):
        """
        Returns the exponent of the function.
        """
        return self.args[0]

    def as_base_exp(self):
        """
        Returns the 2-tuple (base, exponent).
        """
        return self.func(1), Mul(*self.args)

    def _eval_adjoint(self):
        return self.func(self.exp.adjoint())

    def _eval_conjugate(self):
        return self.func(self.exp.conjugate())

    def _eval_transpose(self):
        return self.func(self.exp.transpose())

    def _eval_is_finite(self):
        arg = self.exp
        if arg.is_infinite:
            if arg.is_extended_negative:
                return True
            if arg.is_extended_positive:
                return False
        if arg.is_finite:
            return True

    def _eval_is_rational(self):
        s = self.func(*self.args)
        if s.func == self.func:
            z = s.exp.is_zero
            if z:
                return True
            elif s.exp.is_rational and fuzzy_not(z):
                return False
        else:
            return s.is_rational

    def _eval_is_zero(self):
        return self.exp is S.NegativeInfinity

    def _eval_power(self, other):
        """exp(arg)**e -> exp(arg*e) if assumptions allow it.
        """
        b, e = self.as_base_exp()
        return Pow._eval_power(Pow(b, e, evaluate=False), other)

    def _eval_expand_power_exp(self, **hints):
        from sympy.concrete.products import Product
        from sympy.concrete.summations import Sum
        arg = self.args[0]
        if arg.is_Add and arg.is_commutative:
            return Mul.fromiter(self.func(x) for x in arg.args)
        elif isinstance(arg, Sum) and arg.is_commutative:
            return Product(self.func(arg.function), *arg.limits)
        return self.func(arg)


class exp_polar(ExpBase):
    r"""
    Represent a *polar number* (see g-function Sphinx documentation).

    Explanation
    ===========

    ``exp_polar`` represents the function
    `Exp: \mathbb{C} \rightarrow \mathcal{S}`, sending the complex number
    `z = a + bi` to the polar number `r = exp(a), \theta = b`. It is one of
    the main functions to construct polar numbers.

    Examples
    ========

    >>> from sympy import exp_polar, pi, I, exp

    The main difference is that polar numbers do not "wrap around" at `2 \pi`:

    >>> exp(2*pi*I)
    1
    >>> exp_polar(2*pi*I)
    exp_polar(2*I*pi)

    apart from that they behave mostly like classical complex numbers:

    >>> exp_polar(2)*exp_polar(3)
    exp_polar(5)

    See Also
    ========

    sympy.simplify.powsimp.powsimp
    polar_lift
    periodic_argument
    principal_branch
    """

    is_polar = True
    is_comparable = False  # cannot be evalf'd

    def _eval_Abs(self):   # Abs is never a polar number
        return exp(re(self.args[0]))

    def _eval_evalf(self, prec):
        """ Careful! any evalf of polar numbers is flaky """
        i = im(self.args[0])
        try:
            bad = (i <= -pi or i > pi)
        except TypeError:
            bad = True
        if bad:
            return self  # cannot evalf for this argument
        res = exp(self.args[0])._eval_evalf(prec)
        if i > 0 and im(res) < 0:
            # i ~ pi, but exp(I*i) evaluated to argument slightly bigger than pi
            return re(res)
        return res

    def _eval_power(self, other):
        return self.func(self.args[0]*other)

    def _eval_is_extended_real(self):
        if self.args[0].is_extended_real:
            return True

    def as_base_exp(self):
        # XXX exp_polar(0) is special!
        if self.args[0] == 0:
            return self, S.One
        return ExpBase.as_base_exp(self)


class ExpMeta(FunctionClass):
    def __instancecheck__(cls, instance):
        if exp in instance.__class__.__mro__:
            return True
        return isinstance(instance, Pow) and instance.base is S.Exp1


class exp(ExpBase, metaclass=ExpMeta):
    """
    The exponential function, :math:`e^x`.

    Examples
    ========

    >>> from sympy import exp, I, pi
    >>> from sympy.abc import x
    >>> exp(x)
    exp(x)
    >>> exp(x).diff(x)
    exp(x)
    >>> exp(I*pi)
    -1

    Parameters
    ==========

    arg : Expr

    See Also
    ========

    log
    """

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return self
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_refine(self, assumptions):
        from sympy.assumptions import ask, Q
        arg = self.args[0]
        if arg.is_Mul:
            Ioo = I*S.Infinity
            if arg in [Ioo, -Ioo]:
                return S.NaN

            coeff = arg.as_coefficient(pi*I)
            if coeff:
                if ask(Q.integer(2*coeff)):
                    if ask(Q.even(coeff)):
                        return S.One
                    elif ask(Q.odd(coeff)):
                        return S.NegativeOne
                    elif ask(Q.even(coeff + S.Half)):
                        return -I
                    elif ask(Q.odd(coeff + S.Half)):
                        return I

    @classmethod
    def eval(cls, arg):
        from sympy.calculus import AccumBounds
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.sets.setexpr import SetExpr
        from sympy.simplify.simplify import logcombine
        if isinstance(arg, MatrixBase):
            return arg.exp()
        elif global_parameters.exp_is_pow:
            return Pow(S.Exp1, arg)
        elif arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg.is_zero:
                return S.One
            elif arg is S.One:
                return S.Exp1
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.Zero
        elif arg is S.ComplexInfinity:
            return S.NaN
        elif isinstance(arg, log):
            return arg.args[0]
        elif isinstance(arg, AccumBounds):
            return AccumBounds(exp(arg.min), exp(arg.max))
        elif isinstance(arg, SetExpr):
            return arg._eval_func(cls)
        elif arg.is_Mul:
            coeff = arg.as_coefficient(pi*I)
            if coeff:
                if (2*coeff).is_integer:
                    if coeff.is_even:
                        return S.One
                    elif coeff.is_odd:
                        return S.NegativeOne
                    elif (coeff + S.Half).is_even:
                        return -I
                    elif (coeff + S.Half).is_odd:
                        return I
                elif coeff.is_Rational:
                    ncoeff = coeff % 2 # restrict to [0, 2pi)
                    if ncoeff > 1: # restrict to (-pi, pi]
                        ncoeff -= 2
                    if ncoeff != coeff:
                        return cls(ncoeff*pi*I)

            # Warning: code in risch.py will be very sensitive to changes
            # in this (see DifferentialExtension).

            # look for a single log factor

            coeff, terms = arg.as_coeff_Mul()

            # but it can't be multiplied by oo
            if coeff in [S.NegativeInfinity, S.Infinity]:
                if terms.is_number:
                    if coeff is S.NegativeInfinity:
                        terms = -terms
                    if re(terms).is_zero and terms is not S.Zero:
                        return S.NaN
                    if re(terms).is_positive and im(terms) is not S.Zero:
                        return S.ComplexInfinity
                    if re(terms).is_negative:
                        return S.Zero
                return None

            coeffs, log_term = [coeff], None
            for term in Mul.make_args(terms):
                term_ = logcombine(term)
                if isinstance(term_, log):
                    if log_term is None:
                        log_term = term_.args[0]
                    else:
                        return None
                elif term.is_comparable:
                    coeffs.append(term)
                else:
                    return None

            return log_term**Mul(*coeffs) if log_term else None

        elif arg.is_Add:
            out = []
            add = []
            argchanged = False
            for a in arg.args:
                if a is S.One:
                    add.append(a)
                    continue
                newa = cls(a)
                if isinstance(newa, cls):
                    if newa.args[0] != a:
                        add.append(newa.args[0])
                        argchanged = True
                    else:
                        add.append(a)
                else:
                    out.append(newa)
            if out or argchanged:
                return Mul(*out)*cls(Add(*add), evaluate=False)

        if arg.is_zero:
            return S.One

    @property
    def base(self):
        """
        Returns the base of the exponential function.
        """
        return S.Exp1

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        """
        Calculates the next term in the Taylor series expansion.
        """
        if n < 0:
            return S.Zero
        if n == 0:
            return S.One
        x = sympify(x)
        if previous_terms:
            p = previous_terms[-1]
            if p is not None:
                return p * x / n
        return x**n/factorial(n)

    def as_real_imag(self, deep=True, **hints):
        """
        Returns this function as a 2-tuple representing a complex number.

        Examples
        ========

        >>> from sympy import exp, I
        >>> from sympy.abc import x
        >>> exp(x).as_real_imag()
        (exp(re(x))*cos(im(x)), exp(re(x))*sin(im(x)))
        >>> exp(1).as_real_imag()
        (E, 0)
        >>> exp(I).as_real_imag()
        (cos(1), sin(1))
        >>> exp(1+I).as_real_imag()
        (E*cos(1), E*sin(1))

        See Also
        ========

        sympy.functions.elementary.complexes.re
        sympy.functions.elementary.complexes.im
        """
        from sympy.functions.elementary.trigonometric import cos, sin
        re, im = self.args[0].as_real_imag()
        if deep:
            re = re.expand(deep, **hints)
            im = im.expand(deep, **hints)
        cos, sin = cos(im), sin(im)
        return (exp(re)*cos, exp(re)*sin)

    def _eval_subs(self, old, new):
        # keep processing of power-like args centralized in Pow
        if old.is_Pow:  # handle (exp(3*log(x))).subs(x**2, z) -> z**(3/2)
            old = exp(old.exp*log(old.base))
        elif old is S.Exp1 and new.is_Function:
            old = exp
        if isinstance(old, exp) or old is S.Exp1:
            f = lambda a: Pow(*a.as_base_exp(), evaluate=False) if (
                a.is_Pow or isinstance(a, exp)) else a
            return Pow._eval_subs(f(self), f(old), new)

        if old is exp and not new.is_Function:
            return new**self.exp._subs(old, new)
        return super()._eval_subs(old, new)

    def _eval_is_extended_real(self):
        if self.args[0].is_extended_real:
            return True
        elif self.args[0].is_imaginary:
            arg2 = -S(2) * I * self.args[0] / pi
            return arg2.is_even

    def _eval_is_complex(self):
        def complex_extended_negative(arg):
            yield arg.is_complex
            yield arg.is_extended_negative
        return fuzzy_or(complex_extended_negative(self.args[0]))

    def _eval_is_algebraic(self):
        if (self.exp / pi / I).is_rational:
            return True
        if fuzzy_not(self.exp.is_zero):
            if self.exp.is_algebraic:
                return False
            elif (self.exp / pi).is_rational:
                return False

    def _eval_is_extended_positive(self):
        if self.exp.is_extended_real:
            return self.args[0] is not S.NegativeInfinity
        elif self.exp.is_imaginary:
            arg2 = -I * self.args[0] / pi
            return arg2.is_even

    def _eval_nseries(self, x, n, logx, cdir=0):
        # NOTE Please see the comment at the beginning of this file, labelled
        #      IMPORTANT.
        from sympy.functions.elementary.complexes import sign
        from sympy.functions.elementary.integers import ceiling
        from sympy.series.limits import limit
        from sympy.series.order import Order
        from sympy.simplify.powsimp import powsimp
        arg = self.exp
        arg_series = arg._eval_nseries(x, n=n, logx=logx)
        if arg_series.is_Order:
            return 1 + arg_series
        arg0 = limit(arg_series.removeO(), x, 0)
        if arg0 is S.NegativeInfinity:
            return Order(x**n, x)
        if arg0 is S.Infinity:
            return self
        if arg0.is_infinite:
            raise PoleError("Cannot expand %s around 0" % (self))
        # checking for indecisiveness/ sign terms in arg0
        if any(isinstance(arg, sign) for arg in arg0.args):
            return self
        t = Dummy("t")
        nterms = n
        try:
            cf = Order(arg.as_leading_term(x, logx=logx), x).getn()
        except (NotImplementedError, PoleError):
            cf = 0
        if cf and cf > 0:
            nterms = ceiling(n/cf)
        exp_series = exp(t)._taylor(t, nterms)
        r = exp(arg0)*exp_series.subs(t, arg_series - arg0)
        rep = {logx: log(x)} if logx is not None else {}
        if r.subs(rep) == self:
            return r
        if cf and cf > 1:
            r += Order((arg_series - arg0)**n, x)/x**((cf-1)*n)
        else:
            r += Order((arg_series - arg0)**n, x)
        r = r.expand()
        r = powsimp(r, deep=True, combine='exp')
        # powsimp may introduce unexpanded (-1)**Rational; see PR #17201
        simplerat = lambda x: x.is_Rational and x.q in [3, 4, 6]
        w = Wild('w', properties=[simplerat])
        r = r.replace(S.NegativeOne**w, expand_complex(S.NegativeOne**w))
        return r

    def _taylor(self, x, n):
        l = []
        g = None
        for i in range(n):
            g = self.taylor_term(i, self.args[0], g)
            g = g.nseries(x, n=n)
            l.append(g.removeO())
        return Add(*l)

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.calculus.util import AccumBounds
        arg = self.args[0].cancel().as_leading_term(x, logx=logx)
        arg0 = arg.subs(x, 0)
        if arg is S.NaN:
            return S.NaN
        if isinstance(arg0, AccumBounds):
            # This check addresses a corner case involving AccumBounds.
            # if isinstance(arg, AccumBounds) is True, then arg0 can either be 0,
            # AccumBounds(-oo, 0) or AccumBounds(-oo, oo).
            # Check out function: test_issue_18473() in test_exponential.py and
            # test_limits.py for more information.
            if re(cdir) < S.Zero:
                return exp(-arg0)
            return exp(arg0)
        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0)
        if arg0.is_infinite is False:
            return exp(arg0)
        raise PoleError("Cannot expand %s around 0" % (self))

    def _eval_rewrite_as_sin(self, arg, **kwargs):
        from sympy.functions.elementary.trigonometric import sin
        return sin(I*arg + pi/2) - I*sin(I*arg)

    def _eval_rewrite_as_cos(self, arg, **kwargs):
        from sympy.functions.elementary.trigonometric import cos
        return cos(I*arg) + I*cos(I*arg + pi/2)

    def _eval_rewrite_as_tanh(self, arg, **kwargs):
        from sympy.functions.elementary.hyperbolic import tanh
        return (1 + tanh(arg/2))/(1 - tanh(arg/2))

    def _eval_rewrite_as_sqrt(self, arg, **kwargs):
        from sympy.functions.elementary.trigonometric import sin, cos
        if arg.is_Mul:
            coeff = arg.coeff(pi*I)
            if coeff and coeff.is_number:
                cosine, sine = cos(pi*coeff), sin(pi*coeff)
                if not isinstance(cosine, cos) and not isinstance (sine, sin):
                    return cosine + I*sine

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        if arg.is_Mul:
            logs = [a for a in arg.args if isinstance(a, log) and len(a.args) == 1]
            if logs:
                return Pow(logs[0].args[0], arg.coeff(logs[0]))


def match_real_imag(expr):
    r"""
    Try to match expr with $a + Ib$ for real $a$ and $b$.

    ``match_real_imag`` returns a tuple containing the real and imaginary
    parts of expr or ``(None, None)`` if direct matching is not possible. Contrary
    to :func:`~.re`, :func:`~.im``, and ``as_real_imag()``, this helper will not force things
    by returning expressions themselves containing ``re()`` or ``im()`` and it
    does not expand its argument either.

    """
    r_, i_ = expr.as_independent(I, as_Add=True)
    if i_ == 0 and r_.is_real:
        return (r_, i_)
    i_ = i_.as_coefficient(I)
    if i_ and i_.is_real and r_.is_real:
        return (r_, i_)
    else:
        return (None, None) # simpler to check for than None


class log(DefinedFunction):
    r"""
    The natural logarithm function `\ln(x)` or `\log(x)`.

    Explanation
    ===========

    Logarithms are taken with the natural base, `e`. To get
    a logarithm of a different base ``b``, use ``log(x, b)``,
    which is essentially short-hand for ``log(x)/log(b)``.

    ``log`` represents the principal branch of the natural
    logarithm. As such it has a branch cut along the negative
    real axis and returns values having a complex argument in
    `(-\pi, \pi]`.

    Examples
    ========

    >>> from sympy import log, sqrt, S, I
    >>> log(8, 2)
    3
    >>> log(S(8)/3, 2)
    -log(3)/log(2) + 3
    >>> log(-1 + I*sqrt(3))
    log(2) + 2*I*pi/3

    See Also
    ========

    exp

    """

    args: tuple[Expr]

    _singularities = (S.Zero, S.ComplexInfinity)

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of the function.
        """
        if argindex == 1:
            return 1/self.args[0]
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        r"""
        Returns `e^x`, the inverse function of `\log(x)`.
        """
        return exp

    @classmethod
    def eval(cls, arg, base=None):
        from sympy.calculus import AccumBounds
        from sympy.sets.setexpr import SetExpr

        arg = sympify(arg)

        if base is not None:
            base = sympify(base)
            if base == 1:
                if arg == 1:
                    return S.NaN
                else:
                    return S.ComplexInfinity
            try:
                # handle extraction of powers of the base now
                # or else expand_log in Mul would have to handle this
                n = multiplicity(base, arg)
                if n:
                    return n + log(arg / base**n) / log(base)
                else:
                    return log(arg)/log(base)
            except ValueError:
                pass
            if base is not S.Exp1:
                return cls(arg)/cls(base)
            else:
                return cls(arg)

        if arg.is_Number:
            if arg.is_zero:
                return S.ComplexInfinity
            elif arg is S.One:
                return S.Zero
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.Infinity
            elif arg is S.NaN:
                return S.NaN
            elif arg.is_Rational and arg.p == 1:
                return -cls(arg.q)

        if arg.is_Pow and arg.base is S.Exp1 and arg.exp.is_extended_real:
            return arg.exp
        if isinstance(arg, exp) and arg.exp.is_extended_real:
            return arg.exp
        elif isinstance(arg, exp) and arg.exp.is_number:
            r_, i_ = match_real_imag(arg.exp)
            if i_ and i_.is_comparable:
                i_ %= 2*pi
                if i_ > pi:
                    i_ -= 2*pi
                return r_ + expand_mul(i_ * I, deep=False)
        elif isinstance(arg, exp_polar):
            return unpolarify(arg.exp)
        elif isinstance(arg, AccumBounds):
            if arg.min.is_positive:
                return AccumBounds(log(arg.min), log(arg.max))
            elif arg.min.is_zero:
                return AccumBounds(S.NegativeInfinity, log(arg.max))
            else:
                return S.NaN
        elif isinstance(arg, SetExpr):
            return arg._eval_func(cls)

        if arg.is_number:
            if arg.is_negative:
                return pi * I + cls(-arg)
            elif arg is S.ComplexInfinity:
                return S.ComplexInfinity
            elif arg is S.Exp1:
                return S.One

        if arg.is_zero:
            return S.ComplexInfinity

        # don't autoexpand Pow or Mul (see the issue 3351):
        if not arg.is_Add:
            coeff = arg.as_coefficient(I)

            if coeff is not None:
                if coeff is S.Infinity:
                    return S.Infinity
                elif coeff is S.NegativeInfinity:
                    return S.Infinity
                elif coeff.is_Rational:
                    if coeff.is_nonnegative:
                        return pi * I * S.Half + cls(coeff)
                    else:
                        return -pi * I * S.Half + cls(-coeff)

        if arg.is_number and arg.is_algebraic:
            # Match arg = coeff*(r_ + i_*I) with coeff>0, r_ and i_ real.
            coeff, arg_ = arg.as_independent(I, as_Add=False)
            if coeff.is_negative:
                coeff *= -1
                arg_ *= -1
            arg_ = expand_mul(arg_, deep=False)
            r_, i_ = arg_.as_independent(I, as_Add=True)
            i_ = i_.as_coefficient(I)
            if coeff.is_real and i_ and i_.is_real and r_.is_real:
                if r_.is_zero:
                    if i_.is_positive:
                        return pi * I * S.Half + cls(coeff * i_)
                    elif i_.is_negative:
                        return -pi * I * S.Half + cls(coeff * -i_)
                else:
                    from sympy.simplify import ratsimp
                    # Check for arguments involving rational multiples of pi
                    t = (i_/r_).cancel()
                    t1 = (-t).cancel()
                    atan_table = _log_atan_table()
                    if t in atan_table:
                        modulus = ratsimp(coeff * Abs(arg_))
                        if r_.is_positive:
                            return cls(modulus) + I * atan_table[t]
                        else:
                            return cls(modulus) + I * (atan_table[t] - pi)
                    elif t1 in atan_table:
                        modulus = ratsimp(coeff * Abs(arg_))
                        if r_.is_positive:
                            return cls(modulus) + I * (-atan_table[t1])
                        else:
                            return cls(modulus) + I * (pi - atan_table[t1])

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):  # of log(1+x)
        r"""
        Returns the next term in the Taylor series expansion of `\log(1+x)`.
        """
        from sympy.simplify.powsimp import powsimp
        if n < 0:
            return S.Zero
        x = sympify(x)
        if n == 0:
            return x
        if previous_terms:
            p = previous_terms[-1]
            if p is not None:
                return powsimp((-n) * p * x / (n + 1), deep=True, combine='exp')
        return (1 - 2*(n % 2)) * x**(n + 1)/(n + 1)

    def _eval_expand_log(self, deep=True, **hints):
        from sympy.concrete import Sum, Product
        force = hints.get('force', False)
        factor = hints.get('factor', False)
        if (len(self.args) == 2):
            return expand_log(self.func(*self.args), deep=deep, force=force)
        arg = self.args[0]
        if arg.is_Integer:
            # remove perfect powers
            p = perfect_power(arg)
            logarg = None
            coeff = 1
            if p is not False:
                arg, coeff = p
                logarg = self.func(arg)
            # expand as product of its prime factors if factor=True
            if factor:
                p = factorint(arg)
                if arg not in p.keys():
                    logarg = sum(n*log(val) for val, n in p.items())
            if logarg is not None:
                return coeff*logarg
        elif arg.is_Rational:
            return log(arg.p) - log(arg.q)
        elif arg.is_Mul:
            expr = []
            nonpos = []
            for x in arg.args:
                if force or x.is_positive or x.is_polar:
                    a = self.func(x)
                    if isinstance(a, log):
                        expr.append(self.func(x)._eval_expand_log(**hints))
                    else:
                        expr.append(a)
                elif x.is_negative:
                    a = self.func(-x)
                    expr.append(a)
                    nonpos.append(S.NegativeOne)
                else:
                    nonpos.append(x)
            return Add(*expr) + log(Mul(*nonpos))
        elif arg.is_Pow or isinstance(arg, exp):
            if force or (arg.exp.is_extended_real and (arg.base.is_positive or ((arg.exp+1)
                .is_positive and (arg.exp-1).is_nonpositive))) or arg.base.is_polar:
                b = arg.base
                e = arg.exp
                a = self.func(b)
                if isinstance(a, log):
                    return unpolarify(e) * a._eval_expand_log(**hints)
                else:
                    return unpolarify(e) * a
        elif isinstance(arg, Product):
            if force or arg.function.is_positive:
                return Sum(log(arg.function), *arg.limits)

        return self.func(arg)

    def _eval_simplify(self, **kwargs):
        from sympy.simplify.simplify import expand_log, simplify, inversecombine
        if len(self.args) == 2:  # it's unevaluated
            return simplify(self.func(*self.args), **kwargs)

        expr = self.func(simplify(self.args[0], **kwargs))
        if kwargs['inverse']:
            expr = inversecombine(expr)
        expr = expand_log(expr, deep=True)
        return min([expr, self], key=kwargs['measure'])

    def as_real_imag(self, deep=True, **hints):
        """
        Returns this function as a complex coordinate.

        Examples
        ========

        >>> from sympy import I, log
        >>> from sympy.abc import x
        >>> log(x).as_real_imag()
        (log(Abs(x)), arg(x))
        >>> log(I).as_real_imag()
        (0, pi/2)
        >>> log(1 + I).as_real_imag()
        (log(sqrt(2)), pi/4)
        >>> log(I*x).as_real_imag()
        (log(Abs(x)), arg(I*x))

        """
        sarg = self.args[0]
        if deep:
            sarg = self.args[0].expand(deep, **hints)
        sarg_abs = Abs(sarg)
        if sarg_abs == sarg:
            return self, S.Zero
        sarg_arg = arg(sarg)
        if hints.get('log', False):  # Expand the log
            hints['complex'] = False
            return (log(sarg_abs).expand(deep, **hints), sarg_arg)
        else:
            return log(sarg_abs), sarg_arg

    def _eval_is_rational(self):
        s = self.func(*self.args)
        if s.func == self.func:
            if (self.args[0] - 1).is_zero:
                return True
            if s.args[0].is_rational and fuzzy_not((self.args[0] - 1).is_zero):
                return False
        else:
            return s.is_rational

    def _eval_is_algebraic(self):
        s = self.func(*self.args)
        if s.func == self.func:
            if (self.args[0] - 1).is_zero:
                return True
            elif fuzzy_not((self.args[0] - 1).is_zero):
                if self.args[0].is_algebraic:
                    return False
        else:
            return s.is_algebraic

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_positive

    def _eval_is_complex(self):
        z = self.args[0]
        return fuzzy_and([z.is_complex, fuzzy_not(z.is_zero)])

    def _eval_is_finite(self):
        arg = self.args[0]
        if arg.is_zero:
            return False
        return arg.is_finite

    def _eval_is_extended_positive(self):
        return (self.args[0] - 1).is_extended_positive

    def _eval_is_zero(self):
        return (self.args[0] - 1).is_zero

    def _eval_is_extended_nonnegative(self):
        return (self.args[0] - 1).is_extended_nonnegative

    def _eval_nseries(self, x, n, logx, cdir=0):
        # NOTE Please see the comment at the beginning of this file, labelled
        #      IMPORTANT.
        from sympy.series.order import Order
        from sympy.simplify.simplify import logcombine
        from sympy.core.symbol import Dummy

        if self.args[0] == x:
            return log(x) if logx is None else logx
        arg = self.args[0]
        t = Dummy('t', positive=True)
        if cdir == 0:
            cdir = 1
        z = arg.subs(x, cdir*t)

        k, l = Wild("k"), Wild("l")
        r = z.match(k*t**l)
        if r is not None:
            k, l = r[k], r[l]
            if l != 0 and not l.has(t) and not k.has(t):
                r = l*log(x) if logx is None else l*logx
                r += log(k) - l*log(cdir) # XXX true regardless of assumptions?
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

        # TODO new and probably slow
        try:
            a, b = z.leadterm(t, logx=logx, cdir=1)
        except (ValueError, NotImplementedError, PoleError):
            s = z._eval_nseries(t, n=n, logx=logx, cdir=1)
            while s.is_Order:
                n += 1
                s = z._eval_nseries(t, n=n, logx=logx, cdir=1)
            try:
                a, b = s.removeO().leadterm(t, cdir=1)
            except ValueError:
                a, b = s.removeO().as_leading_term(t, cdir=1), S.Zero

        p = (z/(a*t**b) - 1).cancel()._eval_nseries(t, n=n, logx=logx, cdir=1)
        if p.has(exp):
            p = logcombine(p)
        if isinstance(p, Order):
            n = p.getn()
        _, d = coeff_exp(p, t)
        logx = log(x) if logx is None else logx

        if not d.is_positive:
            res = log(a) - b*log(cdir) + b*logx
            _res = res
            logflags = {"deep": True, "log": True, "mul": False, "power_exp": False,
                "power_base": False, "multinomial": False, "basic": False, "force": True,
                "factor": False}
            expr = self.expand(**logflags)
            if (not a.could_extract_minus_sign() and
                logx.could_extract_minus_sign()):
                _res = _res.subs(-logx, -log(x)).expand(**logflags)
            else:
                _res = _res.subs(logx, log(x)).expand(**logflags)
            if _res == expr:
                return res
            return res + Order(x**n, x)

        def mul(d1, d2):
            res = {}
            for e1, e2 in product(d1, d2):
                ex = e1 + e2
                if ex < n:
                    res[ex] = res.get(ex, S.Zero) + d1[e1]*d2[e2]
            return res

        pterms = {}

