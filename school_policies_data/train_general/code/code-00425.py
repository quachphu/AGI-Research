            return Float._new(mlib.mpf_div(self._mpf_, rhs, prec, rnd), prec)
        return Number.__truediv__(self, other)

    @_sympifyit('other', NotImplemented)
    def __mod__(self, other):
        if isinstance(other, Rational) and other.q != 1 and global_parameters.evaluate:
            # calculate mod with Rationals, *then* round the result
            return Float(Rational.__mod__(Rational(self), other),
                         precision=self._prec)
        if isinstance(other, Float) and global_parameters.evaluate:
            r = self/other
            if int_valued(r):
                return Float(0, precision=max(self._prec, other._prec))
        if isinstance(other, Number) and global_parameters.evaluate:
            rhs, prec = other._as_mpf_op(self._prec)
            return Float._new(mlib.mpf_mod(self._mpf_, rhs, prec, rnd), prec)
        return Number.__mod__(self, other)

    @_sympifyit('other', NotImplemented)
    def __rmod__(self, other):
        if isinstance(other, Float) and global_parameters.evaluate:
            return other.__mod__(self)
        if isinstance(other, Number) and global_parameters.evaluate:
            rhs, prec = other._as_mpf_op(self._prec)
            return Float._new(mlib.mpf_mod(rhs, self._mpf_, prec, rnd), prec)
        return Number.__rmod__(self, other)

    def _eval_power(self, expt):
        """
        expt is symbolic object but not equal to 0, 1

        (-p)**r -> exp(r*log(-p)) -> exp(r*(log(p) + I*Pi)) ->
                  -> p**r*(sin(Pi*r) + cos(Pi*r)*I)
        """
        if equal_valued(self, 0):
            if expt.is_extended_positive:
                return self
            if expt.is_extended_negative:
                return S.ComplexInfinity
        if isinstance(expt, Number):
            if isinstance(expt, Integer):
                prec = self._prec
                return Float._new(
                    mlib.mpf_pow_int(self._mpf_, expt.p, prec, rnd), prec)
            elif isinstance(expt, Rational) and \
                    expt.p == 1 and expt.q % 2 and self.is_negative:
                return Pow(S.NegativeOne, expt, evaluate=False)*(
                    -self)._eval_power(expt)
            expt, prec = expt._as_mpf_op(self._prec)
            mpfself = self._mpf_
            try:
                y = mpf_pow(mpfself, expt, prec, rnd)
                return Float._new(y, prec)
            except mlib.ComplexResult:
                re, im = mlib.mpc_pow(
                    (mpfself, fzero), (expt, fzero), prec, rnd)
                return Float._new(re, prec) + \
                    Float._new(im, prec)*S.ImaginaryUnit

    def __abs__(self):
        return Float._new(mlib.mpf_abs(self._mpf_), self._prec)

    def __int__(self):
        if self._mpf_ == fzero:
            return 0
        return int(mlib.to_int(self._mpf_))  # uses round_fast = round_down

    def __eq__(self, other):
        if isinstance(other, float):
            other = Float(other)
        return Basic.__eq__(self, other)

    def __ne__(self, other):
        eq = self.__eq__(other)
        if eq is NotImplemented:
            return eq
        else:
            return not eq

    def __hash__(self):
        float_val = float(self)
        if not math.isinf(float_val):
            return hash(float_val)
        return Basic.__hash__(self)

    def _Frel(self, other, op):
        try:
            other = _sympify(other)
        except SympifyError:
            return NotImplemented
        if other.is_Rational:
            # test self*other.q <?> other.p without losing precision
            '''
            >>> f = Float(.1,2)
            >>> i = 1234567890
            >>> (f*i)._mpf_
            (0, 471, 18, 9)
            >>> mlib.mpf_mul(f._mpf_, mlib.from_int(i))
            (0, 505555550955, -12, 39)
            '''
            smpf = mlib.mpf_mul(self._mpf_, mlib.from_int(other.q))
            ompf = mlib.from_int(other.p)
            return _sympify(bool(op(smpf, ompf)))
        elif other.is_Float:
            return _sympify(bool(
                        op(self._mpf_, other._mpf_)))
        elif other.is_comparable and other not in (
                S.Infinity, S.NegativeInfinity):
            other = other.evalf(prec_to_dps(self._prec))
            if other._prec > 1:
                if other.is_Number:
                    return _sympify(bool(
                        op(self._mpf_, other._as_mpf_val(self._prec))))

    def __gt__(self, other):
        if isinstance(other, NumberSymbol):
            return other.__lt__(self)
        rv = self._Frel(other, mlib.mpf_gt)
        if rv is None:
            return Expr.__gt__(self, other)
        return rv

    def __ge__(self, other):
        if isinstance(other, NumberSymbol):
            return other.__le__(self)
        rv = self._Frel(other, mlib.mpf_ge)
        if rv is None:
            return Expr.__ge__(self, other)
        return rv

    def __lt__(self, other):
        if isinstance(other, NumberSymbol):
            return other.__gt__(self)
        rv = self._Frel(other, mlib.mpf_lt)
        if rv is None:
            return Expr.__lt__(self, other)
        return rv

    def __le__(self, other):
        if isinstance(other, NumberSymbol):
            return other.__ge__(self)
        rv = self._Frel(other, mlib.mpf_le)
        if rv is None:
            return Expr.__le__(self, other)
        return rv

    def epsilon_eq(self, other, epsilon="1e-15"):
        return abs(self - other) < Float(epsilon)

    def __format__(self, format_spec):
        return format(decimal.Decimal(str(self)), format_spec)


# Add sympify converters
_sympy_converter[float] = _sympy_converter[decimal.Decimal] = Float

# this is here to work nicely in Sage
RealNumber = Float


class Rational(Number):
    """Represents rational numbers (p/q) of any size.

    Examples
    ========

    >>> from sympy import Rational, nsimplify, S, pi
    >>> Rational(1, 2)
    1/2

    Rational is unprejudiced in accepting input. If a float is passed, the
    underlying value of the binary representation will be returned:

    >>> Rational(.5)
    1/2
    >>> Rational(.2)
    3602879701896397/18014398509481984

    If the simpler representation of the float is desired then consider
    limiting the denominator to the desired value or convert the float to
    a string (which is roughly equivalent to limiting the denominator to
    10**12):

    >>> Rational(str(.2))
    1/5
    >>> Rational(.2).limit_denominator(10**12)
    1/5

    An arbitrarily precise Rational is obtained when a string literal is
    passed:

    >>> Rational("1.23")
    123/100
    >>> Rational('1e-2')
    1/100
    >>> Rational(".1")
    1/10
    >>> Rational('1e-2/3.2')
    1/320

    The conversion of other types of strings can be handled by
    the sympify() function, and conversion of floats to expressions
    or simple fractions can be handled with nsimplify:

    >>> S('.[3]')  # repeating digits in brackets
    1/3
    >>> S('3**2/10')  # general expressions
    9/10
    >>> nsimplify(.3)  # numbers that have a simple form
    3/10

    But if the input does not reduce to a literal Rational, an error will
    be raised:

    >>> Rational(pi)
    Traceback (most recent call last):
    ...
    TypeError: invalid input: pi


    Low-level
    ---------

    Access numerator and denominator as .p and .q:

    >>> r = Rational(3, 4)
    >>> r
    3/4
    >>> r.p
    3
    >>> r.q
    4

    Note that p and q return integers (not SymPy Integers) so some care
    is needed when using them in expressions:

    >>> r.p/r.q
    0.75

    See Also
    ========
    sympy.core.sympify.sympify, sympy.simplify.simplify.nsimplify
    """
    is_real = True
    is_integer = False
    is_rational = True
    is_number = True

    __slots__ = ('p', 'q')

    p: int
    q: int

    is_Rational = True

    @cacheit
    def __new__(cls, p, q=None, gcd=None):
        if q is None:
            if isinstance(p, Rational):
                return p

            if isinstance(p, SYMPY_INTS):
                pass
            else:
                if isinstance(p, (float, Float)):
                    return Rational(*_as_integer_ratio(p))

                if not isinstance(p, str):
                    try:
                        p = sympify(p)
                    except (SympifyError, SyntaxError):
                        pass  # error will raise below
                else:
                    if p.count('/') > 1:
                        raise TypeError('invalid input: %s' % p)
                    p = p.replace(' ', '')
                    pq = p.rsplit('/', 1)
                    if len(pq) == 2:
                        p, q = pq
                        fp = fractions.Fraction(p)
                        fq = fractions.Fraction(q)
                        p = fp/fq
                    try:
                        p = fractions.Fraction(p)
                    except ValueError:
                        pass  # error will raise below
                    else:
                        return cls._new(p.numerator, p.denominator, 1)

                if not isinstance(p, Rational):
                    raise TypeError('invalid input: %s' % p)

            q = 1

        Q = 1

        if not isinstance(p, SYMPY_INTS):
            p = Rational(p)
            Q *= p.q
            p = p.p
        else:
            p = int(p)

        if not isinstance(q, SYMPY_INTS):
            q = Rational(q)
            p *= q.q
            Q *= q.p
        else:
            Q *= int(q)
        q = Q

        if gcd is not None:
            sympy_deprecation_warning(
                "gcd is deprecated in Rational, use nsimplify instead",
                deprecated_since_version="1.11",
                active_deprecations_target="deprecated-rational-gcd",
                stacklevel=4,
            )
            return cls._new(p, q, gcd)

        # p and q are now ints
        return cls._new(p, q)

    @classmethod
    def _new(cls, p, q, gcd=None):
        if q == 0:
            if p == 0:
                if _errdict["divide"]:
                    raise ValueError("Indeterminate 0/0")
                else:
                    return S.NaN
            return S.ComplexInfinity

        if q < 0:
            q = -q
            p = -p

        if gcd is None:
            gcd = igcd(abs(p), q)

        if gcd > 1:
            p //= gcd
            q //= gcd

        return cls.from_coprime_ints(p, q)

    @classmethod
    def from_coprime_ints(cls, p: int, q: int) -> Rational:
        """Create a Rational from a pair of coprime integers.

        Both ``p`` and ``q`` should be strictly of type ``int``.

        The caller should ensure that ``gcd(p,q) == 1`` and ``q > 0``.

        This may be more efficient than ``Rational(p, q)``. The validity of the
        arguments may or may not be checked so it should not be relied upon to
        pass unvalidated or invalid arguments to this function.
        """
        if q == 1:
            return Integer(p)
        if p == 1 and q == 2:
            return S.Half

        obj = Expr.__new__(cls)
        obj.p = p
        obj.q = q
        return obj

    def limit_denominator(self, max_denominator=1000000):
        """Closest Rational to self with denominator at most max_denominator.

        Examples
        ========

        >>> from sympy import Rational
        >>> Rational('3.141592653589793').limit_denominator(10)
        22/7
        >>> Rational('3.141592653589793').limit_denominator(100)
        311/99

        """
        f = fractions.Fraction(self.p, self.q)
        return Rational(f.limit_denominator(fractions.Fraction(int(max_denominator))))

    def __getnewargs__(self):
        return (self.p, self.q)

    def _hashable_content(self):
        return (self.p, self.q)

    def _eval_is_positive(self):
        return self.p > 0

    def _eval_is_zero(self):
        return self.p == 0

    def __neg__(self):
        return Rational(-self.p, self.q)

    @_sympifyit('other', NotImplemented)
    def __add__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, Integer):
                return Rational._new(self.p + self.q*other.p, self.q, 1)
            elif isinstance(other, Rational):
                #TODO: this can probably be optimized more
                return Rational(self.p*other.q + self.q*other.p, self.q*other.q)
            elif isinstance(other, Float):
                return other + self
            else:
                return Number.__add__(self, other)
        return Number.__add__(self, other)
    __radd__ = __add__

    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, Integer):
                return Rational._new(self.p - self.q*other.p, self.q, 1)
            elif isinstance(other, Rational):
                return Rational(self.p*other.q - self.q*other.p, self.q*other.q)
            elif isinstance(other, Float):
                return -other + self
            else:
                return Number.__sub__(self, other)
        return Number.__sub__(self, other)
    @_sympifyit('other', NotImplemented)
    def __rsub__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, Integer):
                return Rational._new(self.q*other.p - self.p, self.q, 1)
            elif isinstance(other, Rational):
                return Rational(self.q*other.p - self.p*other.q, self.q*other.q)
            elif isinstance(other, Float):
                return -self + other
            else:
                return Number.__rsub__(self, other)
        return Number.__rsub__(self, other)
    @_sympifyit('other', NotImplemented)
    def __mul__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, Integer):
                return Rational._new(self.p*other.p, self.q, igcd(other.p, self.q))
            elif isinstance(other, Rational):
                return Rational._new(self.p*other.p, self.q*other.q, igcd(self.p, other.q)*igcd(self.q, other.p))
            elif isinstance(other, Float):
                return other*self
            else:
                return Number.__mul__(self, other)
        return Number.__mul__(self, other)
    __rmul__ = __mul__

    @_sympifyit('other', NotImplemented)
    def __truediv__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, Integer):
                if self.p and other.p == S.Zero:
                    return S.ComplexInfinity
                else:
                    return Rational._new(self.p, self.q*other.p, igcd(self.p, other.p))
            elif isinstance(other, Rational):
                return Rational._new(self.p*other.q, self.q*other.p, igcd(self.p, other.p)*igcd(self.q, other.q))
            elif isinstance(other, Float):
                return self*(1/other)
            else:
                return Number.__truediv__(self, other)
        return Number.__truediv__(self, other)
    @_sympifyit('other', NotImplemented)
    def __rtruediv__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, Integer):
                return Rational._new(other.p*self.q, self.p, igcd(self.p, other.p))
            elif isinstance(other, Rational):
                return Rational._new(other.p*self.q, other.q*self.p, igcd(self.p, other.p)*igcd(self.q, other.q))
            elif isinstance(other, Float):
                return other*(1/self)
            else:
                return Number.__rtruediv__(self, other)
        return Number.__rtruediv__(self, other)

    @_sympifyit('other', NotImplemented)
    def __mod__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, Rational):
                n = (self.p*other.q) // (other.p*self.q)
                return Rational(self.p*other.q - n*other.p*self.q, self.q*other.q)
            if isinstance(other, Float):
                # calculate mod with Rationals, *then* round the answer
                return Float(self.__mod__(Rational(other)),
                             precision=other._prec)
            return Number.__mod__(self, other)
        return Number.__mod__(self, other)

    @_sympifyit('other', NotImplemented)
    def __rmod__(self, other):
        if isinstance(other, Rational):
            return Rational.__mod__(other, self)
        return Number.__rmod__(self, other)

    def _eval_power(self, expt):
        if isinstance(expt, Number):
            if isinstance(expt, Float):
                return self._eval_evalf(expt._prec)**expt
            if expt.is_extended_negative:
                # (3/4)**-2 -> (4/3)**2
                ne = -expt
                if (ne is S.One):
                    return Rational(self.q, self.p)
                if self.is_negative:
                    return S.NegativeOne**expt*Rational(self.q, -self.p)**ne
                else:
                    return Rational(self.q, self.p)**ne
            if expt is S.Infinity:  # -oo already caught by test for negative
                if self.p > self.q:
                    # (3/2)**oo -> oo
                    return S.Infinity
                if self.p < -self.q:
                    # (-3/2)**oo -> oo + I*oo
                    return S.Infinity + S.Infinity*S.ImaginaryUnit
                return S.Zero
            if isinstance(expt, Integer):
                # (4/3)**2 -> 4**2 / 3**2
                return Rational._new(self.p**expt.p, self.q**expt.p, 1)
            if isinstance(expt, Rational):
                intpart = expt.p // expt.q
                if intpart:
                    intpart += 1
                    remfracpart = intpart*expt.q - expt.p
                    ratfracpart = Rational(remfracpart, expt.q)
                    if self.p != 1:
                        return Integer(self.p)**expt*Integer(self.q)**ratfracpart*Rational._new(1, self.q**intpart, 1)
                    return Integer(self.q)**ratfracpart*Rational._new(1, self.q**intpart, 1)
                else:
                    remfracpart = expt.q - expt.p
                    ratfracpart = Rational(remfracpart, expt.q)
                    if self.p != 1:
                        return Integer(self.p)**expt*Integer(self.q)**ratfracpart*Rational._new(1, self.q, 1)
                    return Integer(self.q)**ratfracpart*Rational._new(1, self.q, 1)

        if self.is_extended_negative and expt.is_even:
            return (-self)**expt

        return

    def _as_mpf_val(self, prec):
        return mlib.from_rational(self.p, self.q, prec, rnd)

    def _mpmath_(self, prec, rnd):
        return mpmath.make_mpf(mlib.from_rational(self.p, self.q, prec, rnd))

    def __abs__(self):
        return Rational(abs(self.p), self.q)

    def __int__(self):
        p, q = self.p, self.q
        if p < 0:
            return -int(-p//q)
        return int(p//q)

    def floor(self):
        return Integer(self.p // self.q)

    def ceiling(self):
        return -Integer(-self.p // self.q)

    def __floor__(self):
        return self.floor()

    def __ceil__(self):
        return self.ceiling()

    def __eq__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            return NotImplemented
        if not isinstance(other, Number):
            # S(0) == S.false is False
            # S(0) == False is True
            return False
        if other.is_NumberSymbol:
            if other.is_irrational:
                return False
            return other.__eq__(self)
        if other.is_Rational:
            # a Rational is always in reduced form so will never be 2/4
            # so we can just check equivalence of args
            return self.p == other.p and self.q == other.q
        return False

    def __ne__(self, other):
        return not self == other

    def _Rrel(self, other, attr):
        # if you want self < other, pass self, other, __gt__
        try:
            other = _sympify(other)
        except SympifyError:
            return NotImplemented
        if other.is_Number:
            op = None
            s, o = self, other
            if other.is_NumberSymbol:
                op = getattr(o, attr)
            elif other.is_Float:
                op = getattr(o, attr)
            elif other.is_Rational:
                s, o = Integer(s.p*o.q), Integer(s.q*o.p)
                op = getattr(o, attr)
            if op:
                return op(s)
            if o.is_number and o.is_extended_real:
                return Integer(s.p), s.q*o

    def __gt__(self, other):
        rv = self._Rrel(other, '__lt__')
        if rv is None:
            rv = self, other
        elif not isinstance(rv, tuple):
            return rv
        return Expr.__gt__(*rv)

    def __ge__(self, other):
        rv = self._Rrel(other, '__le__')
        if rv is None:
            rv = self, other
        elif not isinstance(rv, tuple):
            return rv
        return Expr.__ge__(*rv)

    def __lt__(self, other):
        rv = self._Rrel(other, '__gt__')
        if rv is None:
            rv = self, other
        elif not isinstance(rv, tuple):
            return rv
        return Expr.__lt__(*rv)

    def __le__(self, other):
        rv = self._Rrel(other, '__ge__')
        if rv is None:
            rv = self, other
        elif not isinstance(rv, tuple):
            return rv
        return Expr.__le__(*rv)

    def __hash__(self):
        return super().__hash__()

    def factors(self, limit=None, use_trial=True, use_rho=False,
                use_pm1=False, verbose=False, visual=False):
        """A wrapper to factorint which return factors of self that are
        smaller than limit (or cheap to compute). Special methods of
        factoring are disabled by default so that only trial division is used.
        """
        from sympy.ntheory.factor_ import factorrat

        return factorrat(self, limit=limit, use_trial=use_trial,
                      use_rho=use_rho, use_pm1=use_pm1,
                      verbose=verbose).copy()

    @property
    def numerator(self):
        return self.p

    @property
    def denominator(self):
        return self.q

    @_sympifyit('other', NotImplemented)
    def gcd(self, other):
        if isinstance(other, Rational):
            if other == S.Zero:
                return other
            return Rational(
                igcd(self.p, other.p),
                ilcm(self.q, other.q))
        return Number.gcd(self, other)

    @_sympifyit('other', NotImplemented)
    def lcm(self, other):
        if isinstance(other, Rational):
            return Rational(
                self.p // igcd(self.p, other.p) * other.p,
                igcd(self.q, other.q))
        return Number.lcm(self, other)

    def as_numer_denom(self):
        return Integer(self.p), Integer(self.q)

    def as_content_primitive(self, radical=False, clear=True):
        """Return the tuple (R, self/R) where R is the positive Rational
        extracted from self.

        Examples
        ========

        >>> from sympy import S
        >>> (S(-3)/2).as_content_primitive()
        (3/2, -1)

        See docstring of Expr.as_content_primitive for more examples.
        """

        if self:
            if self.is_positive:
                return self, S.One
            return -self, S.NegativeOne
        return S.One, self

    def as_coeff_Mul(self, rational=False):
        """Efficiently extract the coefficient of a product."""
        return self, S.One

    def as_coeff_Add(self, rational=False):
        """Efficiently extract the coefficient of a summation."""
        return self, S.Zero


class Integer(Rational):
    """Represents integer numbers of any size.

    Examples
    ========

    >>> from sympy import Integer
    >>> Integer(3)
    3

    If a float or a rational is passed to Integer, the fractional part
    will be discarded; the effect is of rounding toward zero.

    >>> Integer(3.8)
    3
    >>> Integer(-3.8)
    -3

    A string is acceptable input if it can be parsed as an integer:

    >>> Integer("9" * 20)
    99999999999999999999

    It is rarely needed to explicitly instantiate an Integer, because
    Python integers are automatically converted to Integer when they
    are used in SymPy expressions.
    """
    q = 1
    is_integer = True
    is_number = True

    is_Integer = True

    __slots__ = ()

    def _as_mpf_val(self, prec):
        return mlib.from_int(self.p, prec, rnd)

    def _mpmath_(self, prec, rnd):
        return mpmath.make_mpf(self._as_mpf_val(prec))

    @cacheit
    def __new__(cls, i):
        if isinstance(i, str):
            i = i.replace(' ', '')
        # whereas we cannot, in general, make a Rational from an
        # arbitrary expression, we can make an Integer unambiguously
        # (except when a non-integer expression happens to round to
        # an integer). So we proceed by taking int() of the input and
        # let the int routines determine whether the expression can
        # be made into an int or whether an error should be raised.
        try:
            ival = int(i)
        except TypeError:
            raise TypeError(
                "Argument of Integer should be of numeric type, got %s." % i)
        # We only work with well-behaved integer types. This converts, for
        # example, numpy.int32 instances.
        if ival == 1:
            return S.One
        if ival == -1:
            return S.NegativeOne
        if ival == 0:
            return S.Zero
        obj = Expr.__new__(cls)
        obj.p = ival
        return obj

    def __getnewargs__(self):
        return (self.p,)

    # Arithmetic operations are here for efficiency
    def __int__(self):
        return self.p

    def floor(self):
        return Integer(self.p)

    def ceiling(self):
        return Integer(self.p)

    def __floor__(self):
        return self.floor()

    def __ceil__(self):
        return self.ceiling()

    def __neg__(self):
        return Integer(-self.p)

    def __abs__(self):
        if self.p >= 0:
            return self
        else:
            return Integer(-self.p)

    def __divmod__(self, other):
        if isinstance(other, Integer) and global_parameters.evaluate:
            return Tuple(*(divmod(self.p, other.p)))
        else:
            return Number.__divmod__(self, other)

    def __rdivmod__(self, other):
        if isinstance(other, int) and global_parameters.evaluate:
            return Tuple(*(divmod(other, self.p)))
        else:
            try:
                other = Number(other)
            except TypeError:
                msg = "unsupported operand type(s) for divmod(): '%s' and '%s'"
                oname = type(other).__name__
                sname = type(self).__name__
                raise TypeError(msg % (oname, sname))
            return Number.__divmod__(other, self)

    # TODO make it decorator + bytecodehacks?
    def __add__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, int):
                return Integer(self.p + other)
            elif isinstance(other, Integer):
                return Integer(self.p + other.p)
            elif isinstance(other, Rational):
                return Rational._new(self.p*other.q + other.p, other.q, 1)
            return Rational.__add__(self, other)
        else:
            return Add(self, other)

    def __radd__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, int):
                return Integer(other + self.p)
            elif isinstance(other, Rational):
                return Rational._new(other.p + self.p*other.q, other.q, 1)
            return Rational.__radd__(self, other)
        return Rational.__radd__(self, other)

    def __sub__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, int):
                return Integer(self.p - other)
            elif isinstance(other, Integer):
                return Integer(self.p - other.p)
            elif isinstance(other, Rational):
                return Rational._new(self.p*other.q - other.p, other.q, 1)
            return Rational.__sub__(self, other)
        return Rational.__sub__(self, other)

    def __rsub__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, int):
                return Integer(other - self.p)
            elif isinstance(other, Rational):
                return Rational._new(other.p - self.p*other.q, other.q, 1)
            return Rational.__rsub__(self, other)
        return Rational.__rsub__(self, other)

    def __mul__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, int):
                return Integer(self.p*other)
            elif isinstance(other, Integer):
                return Integer(self.p*other.p)
            elif isinstance(other, Rational):
                return Rational._new(self.p*other.p, other.q, igcd(self.p, other.q))
            return Rational.__mul__(self, other)
        return Rational.__mul__(self, other)

    def __rmul__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, int):
                return Integer(other*self.p)
            elif isinstance(other, Rational):
                return Rational._new(other.p*self.p, other.q, igcd(self.p, other.q))
            return Rational.__rmul__(self, other)
        return Rational.__rmul__(self, other)

    def __mod__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, int):
                return Integer(self.p % other)
            elif isinstance(other, Integer):
                return Integer(self.p % other.p)
            return Rational.__mod__(self, other)
        return Rational.__mod__(self, other)

    def __rmod__(self, other):
        if global_parameters.evaluate:
            if isinstance(other, int):
                return Integer(other % self.p)
            elif isinstance(other, Integer):
                return Integer(other.p % self.p)
            return Rational.__rmod__(self, other)
        return Rational.__rmod__(self, other)

    def __eq__(self, other):
        if isinstance(other, int):
            return (self.p == other)
        elif isinstance(other, Integer):
            return (self.p == other.p)
        return Rational.__eq__(self, other)

    def __ne__(self, other):
        return not self == other

    def __gt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            return NotImplemented
        if other.is_Integer:
            return _sympify(self.p > other.p)
        return Rational.__gt__(self, other)

    def __lt__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            return NotImplemented
        if other.is_Integer:
            return _sympify(self.p < other.p)
        return Rational.__lt__(self, other)

    def __ge__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            return NotImplemented
        if other.is_Integer:
            return _sympify(self.p >= other.p)
        return Rational.__ge__(self, other)

    def __le__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            return NotImplemented
        if other.is_Integer:
            return _sympify(self.p <= other.p)
        return Rational.__le__(self, other)

    def __hash__(self):
        return hash(self.p)

    def __index__(self):
        return self.p

    ########################################

    def _eval_is_odd(self):
        return bool(self.p % 2)

    def _eval_power(self, expt):
        """
        Tries to do some simplifications on self**expt

        Returns None if no further simplifications can be done.

        Explanation
        ===========

        When exponent is a fraction (so we have for example a square root),
        we try to find a simpler representation by factoring the argument
        up to factors of 2**15, e.g.

          - sqrt(4) becomes 2
          - sqrt(-4) becomes 2*I
          - (2**(3+7)*3**(6+7))**Rational(1,7) becomes 6*18**(3/7)

        Further simplification would require a special call to factorint on
        the argument which is not done here for sake of speed.

        """
        from sympy.ntheory.factor_ import perfect_power

        if expt is S.Infinity:
            if self.p > S.One:
                return S.Infinity
            # cases -1, 0, 1 are done in their respective classes
            return S.Infinity + S.ImaginaryUnit*S.Infinity
        if expt is S.NegativeInfinity:
            return Rational._new(1, self, 1)**S.Infinity
        if not isinstance(expt, Number):
            # simplify when expt is even
            # (-2)**k --> 2**k
            if self.is_negative and expt.is_even:
                return (-self)**expt
        if isinstance(expt, Float):
            # Rational knows how to exponentiate by a Float
            return super()._eval_power(expt)
        if not isinstance(expt, Rational):
            return
        if expt is S.Half and self.is_negative:
            # we extract I for this special case since everyone is doing so
            return S.ImaginaryUnit*Pow(-self, expt)
        if expt.is_negative:
            # invert base and change sign on exponent
            ne = -expt
            if self.is_negative:
                return S.NegativeOne**expt*Rational._new(1, -self.p, 1)**ne
            else:
                return Rational._new(1, self.p, 1)**ne
        # see if base is a perfect root, sqrt(4) --> 2
        x, xexact = integer_nthroot(abs(self.p), expt.q)
        if xexact:
            # if it's a perfect root we've finished
            result = Integer(x**abs(expt.p))
            if self.is_negative:
                result *= S.NegativeOne**expt
            return result

        # The following is an algorithm where we collect perfect roots
        # from the factors of base.

        # if it's not an nth root, it still might be a perfect power
        b_pos = int(abs(self.p))
        p = perfect_power(b_pos)
        if p is not False:
            # XXX: Convert to int because perfect_power may return fmpz
            # Ideally that should be fixed in perfect_power though...
            dict = {int(p[0]): int(p[1])}
        else:
            dict = Integer(b_pos).factors(limit=2**15)

        # now process the dict of factors
        out_int = 1  # integer part
        out_rad = 1  # extracted radicals
        sqr_int = 1
        sqr_gcd = 0
        sqr_dict = {}
        for prime, exponent in dict.items():
            exponent *= expt.p
            # remove multiples of expt.q: (2**12)**(1/10) -> 2*(2**2)**(1/10)
            div_e, div_m = divmod(exponent, expt.q)
            if div_e > 0:
                out_int *= prime**div_e
            if div_m > 0:
                # see if the reduced exponent shares a gcd with e.q
                # (2**2)**(1/10) -> 2**(1/5)
                g = igcd(div_m, expt.q)
                if g != 1:
                    out_rad *= Pow(prime, Rational._new(div_m//g, expt.q//g, 1))
                else:
                    sqr_dict[prime] = div_m
        # identify gcd of remaining powers
        for p, ex in sqr_dict.items():
            if sqr_gcd == 0:
                sqr_gcd = ex
            else:
                sqr_gcd = igcd(sqr_gcd, ex)
                if sqr_gcd == 1:
                    break
        for k, v in sqr_dict.items():
            sqr_int *= k**(v//sqr_gcd)
        if sqr_int == b_pos and out_int == 1 and out_rad == 1:
            result = None
        else:
            result = out_int*out_rad*Pow(sqr_int, Rational(sqr_gcd, expt.q))
            if self.is_negative:
                result *= Pow(S.NegativeOne, expt)
        return result

    def _eval_is_prime(self):
        from sympy.ntheory.primetest import isprime

        return isprime(self)

    def _eval_is_composite(self):
        if self > 1:
            return fuzzy_not(self.is_prime)
        else:
            return False

    def as_numer_denom(self):
        return self, S.One

    @_sympifyit('other', NotImplemented)
    def __floordiv__(self, other):
        if not isinstance(other, Expr):
            return NotImplemented
        if isinstance(other, Integer):
            return Integer(self.p // other)
        return divmod(self, other)[0]

    def __rfloordiv__(self, other):
        return Integer(Integer(other).p // self.p)

    # These bitwise operations (__lshift__, __rlshift__, ..., __invert__) are defined
    # for Integer only and not for general SymPy expressions. This is to achieve
    # compatibility with the numbers.Integral ABC which only defines these operations
    # among instances of numbers.Integral. Therefore, these methods check explicitly for
    # integer types rather than using sympify because they should not accept arbitrary
    # symbolic expressions and there is no symbolic analogue of numbers.Integral's
    # bitwise operations.
    def __lshift__(self, other):
        if isinstance(other, (int, Integer, numbers.Integral)):
            return Integer(self.p << int(other))
        else:
            return NotImplemented

    def __rlshift__(self, other):
        if isinstance(other, (int, numbers.Integral)):
            return Integer(int(other) << self.p)
        else:
            return NotImplemented

    def __rshift__(self, other):
        if isinstance(other, (int, Integer, numbers.Integral)):
            return Integer(self.p >> int(other))
        else:
            return NotImplemented

    def __rrshift__(self, other):
        if isinstance(other, (int, numbers.Integral)):
            return Integer(int(other) >> self.p)
        else:
            return NotImplemented

    def __and__(self, other):
        if isinstance(other, (int, Integer, numbers.Integral)):
            return Integer(self.p & int(other))
        else:
            return NotImplemented

    def __rand__(self, other):
        if isinstance(other, (int, numbers.Integral)):
            return Integer(int(other) & self.p)
        else:
            return NotImplemented

    def __xor__(self, other):
        if isinstance(other, (int, Integer, numbers.Integral)):
            return Integer(self.p ^ int(other))
        else:
            return NotImplemented

    def __rxor__(self, other):
        if isinstance(other, (int, numbers.Integral)):
            return Integer(int(other) ^ self.p)
        else:
            return NotImplemented

    def __or__(self, other):
        if isinstance(other, (int, Integer, numbers.Integral)):
            return Integer(self.p | int(other))
        else:
            return NotImplemented

    def __ror__(self, other):
        if isinstance(other, (int, numbers.Integral)):
            return Integer(int(other) | self.p)
        else:
            return NotImplemented

    def __invert__(self):
        return Integer(~self.p)

# Add sympify converters
_sympy_converter[int] = Integer


class AlgebraicNumber(Expr):
    r"""
    Class for representing algebraic numbers in SymPy.

    Symbolically, an instance of this class represents an element
    $\alpha \in \mathbb{Q}(\theta) \hookrightarrow \mathbb{C}$. That is, the
    algebraic number $\alpha$ is represented as an element of a particular
    number field $\mathbb{Q}(\theta)$, with a particular embedding of this
    field into the complex numbers.

    Formally, the primitive element $\theta$ is given by two data points: (1)
    its minimal polynomial (which defines $\mathbb{Q}(\theta)$), and (2) a
    particular complex number that is a root of this polynomial (which defines
    the embedding $\mathbb{Q}(\theta) \hookrightarrow \mathbb{C}$). Finally,
    the algebraic number $\alpha$ which we represent is then given by the
    coefficients of a polynomial in $\theta$.
    """

    __slots__ = ('rep', 'root', 'alias', 'minpoly', '_own_minpoly')

    is_AlgebraicNumber = True
    is_algebraic = True
    is_number = True


    kind = NumberKind

    # Optional alias symbol is not free.
    # Actually, alias should be a Str, but some methods
    # expect that it be an instance of Expr.
    free_symbols: set[Basic] = set()

    def __new__(cls, expr, coeffs=None, alias=None, **args):
        r"""
        Construct a new algebraic number $\alpha$ belonging to a number field
        $k = \mathbb{Q}(\theta)$.

        There are four instance attributes to be determined:

        ===========  ============================================================================
        Attribute    Type/Meaning
        ===========  ============================================================================
        ``root``     :py:class:`~.Expr` for $\theta$ as a complex number
        ``minpoly``  :py:class:`~.Poly`, the minimal polynomial of $\theta$
        ``rep``      :py:class:`~sympy.polys.polyclasses.DMP` giving $\alpha$ as poly in $\theta$
        ``alias``    :py:class:`~.Symbol` for $\theta$, or ``None``
        ===========  ============================================================================

        See Parameters section for how they are determined.

        Parameters
        ==========

        expr : :py:class:`~.Expr`, or pair $(m, r)$
            There are three distinct modes of construction, depending on what
            is passed as *expr*.

            **(1)** *expr* is an :py:class:`~.AlgebraicNumber`:
            In this case we begin by copying all four instance attributes from
            *expr*. If *coeffs* were also given, we compose the two coeff
            polynomials (see below). If an *alias* was given, it overrides.

            **(2)** *expr* is any other type of :py:class:`~.Expr`:
            Then ``root`` will equal *expr*. Therefore it
            must express an algebraic quantity, and we will compute its
            ``minpoly``.

            **(3)** *expr* is an ordered pair $(m, r)$ giving the
            ``minpoly`` $m$, and a ``root`` $r$ thereof, which together
            define $\theta$. In this case $m$ may be either a univariate
            :py:class:`~.Poly` or any :py:class:`~.Expr` which represents the
            same, while $r$ must be some :py:class:`~.Expr` representing a
            complex number that is a root of $m$, including both explicit
            expressions in radicals, and instances of
            :py:class:`~.ComplexRootOf` or :py:class:`~.AlgebraicNumber`.

        coeffs : list, :py:class:`~.ANP`, None, optional (default=None)
            This defines ``rep``, giving the algebraic number $\alpha$ as a
            polynomial in $\theta$.

            If a list, the elements should be integers or rational numbers.
            If an :py:class:`~.ANP`, we take its coefficients (using its
            :py:meth:`~.ANP.to_list()` method). If ``None``, then the list of
            coefficients defaults to ``[1, 0]``, meaning that $\alpha = \theta$
            is the primitive element of the field.

            If *expr* was an :py:class:`~.AlgebraicNumber`, let $g(x)$ be its
            ``rep`` polynomial, and let $f(x)$ be the polynomial defined by
            *coeffs*. Then ``self.rep`` will represent the composition
            $(f \circ g)(x)$.

        alias : str, :py:class:`~.Symbol`, None, optional (default=None)
            This is a way to provide a name for the primitive element. We
            described several ways in which the *expr* argument can define the
            value of the primitive element, but none of these methods gave it
            a name. Here, for example, *alias* could be set as
            ``Symbol('theta')``, in order to make this symbol appear when
            $\alpha$ is printed, or rendered as a polynomial, using the
            :py:meth:`~.as_poly()` method.

        Examples
        ========

        Recall that we are constructing an algebraic number as a field element
        $\alpha \in \mathbb{Q}(\theta)$.

        >>> from sympy import AlgebraicNumber, sqrt, CRootOf, S
        >>> from sympy.abc import x

        Example (1): $\alpha = \theta = \sqrt{2}$

        >>> a1 = AlgebraicNumber(sqrt(2))
        >>> a1.minpoly_of_element().as_expr(x)
        x**2 - 2
        >>> a1.evalf(10)
        1.414213562

        Example (2): $\alpha = 3 \sqrt{2} - 5$, $\theta = \sqrt{2}$. We can
        either build on the last example:

        >>> a2 = AlgebraicNumber(a1, [3, -5])
        >>> a2.as_expr()
        -5 + 3*sqrt(2)

        or start from scratch:

        >>> a2 = AlgebraicNumber(sqrt(2), [3, -5])
        >>> a2.as_expr()
        -5 + 3*sqrt(2)

        Example (3): $\alpha = 6 \sqrt{2} - 11$, $\theta = \sqrt{2}$. Again we
        can build on the previous example, and we see that the coeff polys are
        composed:

        >>> a3 = AlgebraicNumber(a2, [2, -1])
        >>> a3.as_expr()
        -11 + 6*sqrt(2)

        reflecting the fact that $(2x - 1) \circ (3x - 5) = 6x - 11$.

        Example (4): $\alpha = \sqrt{2}$, $\theta = \sqrt{2} + \sqrt{3}$. The
        easiest way is to use the :py:func:`~.to_number_field()` function:

        >>> from sympy import to_number_field
        >>> a4 = to_number_field(sqrt(2), sqrt(2) + sqrt(3))
        >>> a4.minpoly_of_element().as_expr(x)
        x**2 - 2
        >>> a4.to_root()
        sqrt(2)
        >>> a4.primitive_element()
        sqrt(2) + sqrt(3)
        >>> a4.coeffs()
        [1/2, 0, -9/2, 0]

        but if you already knew the right coefficients, you could construct it
        directly:

        >>> a4 = AlgebraicNumber(sqrt(2) + sqrt(3), [S(1)/2, 0, S(-9)/2, 0])
        >>> a4.to_root()
        sqrt(2)
        >>> a4.primitive_element()
        sqrt(2) + sqrt(3)

        Example (5): Construct the Golden Ratio as an element of the 5th
        cyclotomic field, supposing we already know its coefficients. This time
        we introduce the alias $\zeta$ for the primitive element of the field:

        >>> from sympy import cyclotomic_poly
        >>> from sympy.abc import zeta
        >>> a5 = AlgebraicNumber(CRootOf(cyclotomic_poly(5), -1),
        ...                  [-1, -1, 0, 0], alias=zeta)
        >>> a5.as_poly().as_expr()
        -zeta**3 - zeta**2
        >>> a5.evalf()
        1.61803398874989

        (The index ``-1`` to ``CRootOf`` selects the complex root with the
        largest real and imaginary parts, which in this case is
        $\mathrm{e}^{2i\pi/5}$. See :py:class:`~.ComplexRootOf`.)

        Example (6): Building on the last example, construct the number
        $2 \phi \in \mathbb{Q}(\phi)$, where $\phi$ is the Golden Ratio:

        >>> from sympy.abc import phi
        >>> a6 = AlgebraicNumber(a5.to_root(), coeffs=[2, 0], alias=phi)
        >>> a6.as_poly().as_expr()
        2*phi
        >>> a6.primitive_element().evalf()
        1.61803398874989

        Note that we needed to use ``a5.to_root()``, since passing ``a5`` as
        the first argument would have constructed the number $2 \phi$ as an
        element of the field $\mathbb{Q}(\zeta)$:

        >>> a6_wrong = AlgebraicNumber(a5, coeffs=[2, 0])
        >>> a6_wrong.as_poly().as_expr()
        -2*zeta**3 - 2*zeta**2
        >>> a6_wrong.primitive_element().evalf()
        0.309016994374947 + 0.951056516295154*I

        """
        from sympy.polys.polyclasses import ANP, DMP
        from sympy.polys.numberfields import minimal_polynomial

        expr = sympify(expr)
        rep0 = None
        alias0 = None

        if isinstance(expr, (tuple, Tuple)):
            minpoly, root = expr

            if not minpoly.is_Poly:
                from sympy.polys.polytools import Poly
                minpoly = Poly(minpoly)
        elif expr.is_AlgebraicNumber:
            minpoly, root, rep0, alias0 = (expr.minpoly, expr.root,
                                           expr.rep, expr.alias)
        else:
            minpoly, root = minimal_polynomial(
                expr, args.get('gen'), polys=True), expr

        dom = minpoly.get_domain()

        if coeffs is not None:
            if not isinstance(coeffs, ANP):
                rep = DMP.from_sympy_list(sympify(coeffs), 0, dom)
                scoeffs = Tuple(*coeffs)
            else:
                rep = DMP.from_list(coeffs.to_list(), 0, dom)
                scoeffs = Tuple(*coeffs.to_list())

        else:
            rep = DMP.from_list([1, 0], 0, dom)
            scoeffs = Tuple(1, 0)

        if rep0 is not None:
            from sympy.polys.densetools import dup_compose
            c = dup_compose(rep.to_list(), rep0.to_list(), dom)
            rep = DMP.from_list(c, 0, dom)
            scoeffs = Tuple(*c)

        if rep.degree() >= minpoly.degree():
            rep = rep.rem(minpoly.rep)

        sargs = (root, scoeffs)

        alias = alias or alias0
        if alias is not None:
            from .symbol import Symbol
            if not isinstance(alias, Symbol):
                alias = Symbol(alias)
            sargs = sargs + (alias,)

        obj = Expr.__new__(cls, *sargs)

        obj.rep = rep
        obj.root = root
        obj.alias = alias
        obj.minpoly = minpoly

        obj._own_minpoly = None

        return obj

    def __hash__(self):
        return super().__hash__()

    def _eval_evalf(self, prec):
        return self.as_expr()._evalf(prec)

    @property
    def is_aliased(self):
        """Returns ``True`` if ``alias`` was set. """
        return self.alias is not None

    def as_poly(self, x=None):
        """Create a Poly instance from ``self``. """
        from sympy.polys.polytools import Poly, PurePoly
        if x is not None:
            return Poly.new(self.rep, x)
        else:
            if self.alias is not None:
                return Poly.new(self.rep, self.alias)
            else:
                from .symbol import Dummy
                return PurePoly.new(self.rep, Dummy('x'))

    def as_expr(self, x=None):
        """Create a Basic expression from ``self``. """
        return self.as_poly(x or self.root).as_expr().expand()

    def coeffs(self):
        """Returns all SymPy coefficients of an algebraic number. """
        return [ self.rep.dom.to_sympy(c) for c in self.rep.all_coeffs() ]

    def native_coeffs(self):
        """Returns all native coefficients of an algebraic number. """
        return self.rep.all_coeffs()

    def to_algebraic_integer(self):
        """Convert ``self`` to an algebraic integer. """
        from sympy.polys.polytools import Poly

        f = self.minpoly

        if f.LC() == 1:
            return self

        coeff = f.LC()**(f.degree() - 1)
        poly = f.compose(Poly(f.gen/f.LC()))

        minpoly = poly*coeff
        root = f.LC()*self.root

        return AlgebraicNumber((minpoly, root), self.coeffs())

    def _eval_simplify(self, **kwargs):
        from sympy.polys.rootoftools import CRootOf
        from sympy.polys import minpoly
        measure, ratio = kwargs['measure'], kwargs['ratio']
        for r in [r for r in self.minpoly.all_roots() if r.func != CRootOf]:
            if minpoly(self.root - r).is_Symbol:
                # use the matching root if it's simpler
                if measure(r) < ratio*measure(self.root):
                    return AlgebraicNumber(r)
        return self

    def field_element(self, coeffs):
        r"""
        Form another element of the same number field.

        Explanation
        ===========

        If we represent $\alpha \in \mathbb{Q}(\theta)$, form another element
        $\beta \in \mathbb{Q}(\theta)$ of the same number field.

        Parameters
        ==========

        coeffs : list, :py:class:`~.ANP`
            Like the *coeffs* arg to the class
            :py:meth:`constructor<.AlgebraicNumber.__new__>`, defines the
            new element as a polynomial in the primitive element.

            If a list, the elements should be integers or rational numbers.
            If an :py:class:`~.ANP`, we take its coefficients (using its
            :py:meth:`~.ANP.to_list()` method).

        Examples
        ========

        >>> from sympy import AlgebraicNumber, sqrt
        >>> a = AlgebraicNumber(sqrt(5), [-1, 1])
        >>> b = a.field_element([3, 2])
        >>> print(a)
        1 - sqrt(5)
        >>> print(b)
        2 + 3*sqrt(5)
        >>> print(b.primitive_element() == a.primitive_element())
        True

        See Also
        ========

        AlgebraicNumber
        """
        return AlgebraicNumber(
            (self.minpoly, self.root), coeffs=coeffs, alias=self.alias)

    @property
    def is_primitive_element(self):
        r"""
        Say whether this algebraic number $\alpha \in \mathbb{Q}(\theta)$ is
        equal to the primitive element $\theta$ for its field.
        """
        c = self.coeffs()
        # Second case occurs if self.minpoly is linear:
        return c == [1, 0] or c == [self.root]

    def primitive_element(self):
        r"""
        Get the primitive element $\theta$ for the number field
        $\mathbb{Q}(\theta)$ to which this algebraic number $\alpha$ belongs.

        Returns
        =======

        AlgebraicNumber

        """
        if self.is_primitive_element:
            return self
        return self.field_element([1, 0])

    def to_primitive_element(self, radicals=True):
        r"""
        Convert ``self`` to an :py:class:`~.AlgebraicNumber` instance that is
        equal to its own primitive element.

        Explanation
        ===========

        If we represent $\alpha \in \mathbb{Q}(\theta)$, $\alpha \neq \theta$,
        construct a new :py:class:`~.AlgebraicNumber` that represents
        $\alpha \in \mathbb{Q}(\alpha)$.

        Examples
        ========

        >>> from sympy import sqrt, to_number_field
        >>> from sympy.abc import x
        >>> a = to_number_field(sqrt(2), sqrt(2) + sqrt(3))

        The :py:class:`~.AlgebraicNumber` ``a`` represents the number
        $\sqrt{2}$ in the field $\mathbb{Q}(\sqrt{2} + \sqrt{3})$. Rendering
        ``a`` as a polynomial,

        >>> a.as_poly().as_expr(x)
        x**3/2 - 9*x/2

        reflects the fact that $\sqrt{2} = \theta^3/2 - 9 \theta/2$, where
        $\theta = \sqrt{2} + \sqrt{3}$.

        ``a`` is not equal to its own primitive element. Its minpoly

        >>> a.minpoly.as_poly().as_expr(x)
        x**4 - 10*x**2 + 1

        is that of $\theta$.

        Converting to a primitive element,

        >>> a_prim = a.to_primitive_element()
        >>> a_prim.minpoly.as_poly().as_expr(x)
        x**2 - 2

        we obtain an :py:class:`~.AlgebraicNumber` whose ``minpoly`` is that of
        the number itself.

        Parameters
        ==========

        radicals : boolean, optional (default=True)
            If ``True``, then we will try to return an
            :py:class:`~.AlgebraicNumber` whose ``root`` is an expression
            in radicals. If that is not possible (or if *radicals* is
            ``False``), ``root`` will be a :py:class:`~.ComplexRootOf`.

        Returns
        =======

        AlgebraicNumber

        See Also
        ========

        is_primitive_element

        """
        if self.is_primitive_element:
            return self
        m = self.minpoly_of_element()
        r = self.to_root(radicals=radicals)
        return AlgebraicNumber((m, r))

    def minpoly_of_element(self):
        r"""
        Compute the minimal polynomial for this algebraic number.

        Explanation
        ===========

        Recall that we represent an element $\alpha \in \mathbb{Q}(\theta)$.
        Our instance attribute ``self.minpoly`` is the minimal polynomial for
        our primitive element $\theta$. This method computes the minimal
        polynomial for $\alpha$.

        """
        if self._own_minpoly is None:
            if self.is_primitive_element:
                self._own_minpoly = self.minpoly
            else:
                from sympy.polys.numberfields.minpoly import minpoly
                theta = self.primitive_element()
                self._own_minpoly = minpoly(self.as_expr(theta), polys=True)
        return self._own_minpoly

    def to_root(self, radicals=True, minpoly=None):
        """
        Convert to an :py:class:`~.Expr` that is not an
        :py:class:`~.AlgebraicNumber`, specifically, either a
        :py:class:`~.ComplexRootOf`, or, optionally and where possible, an
        expression in radicals.

        Parameters
        ==========

        radicals : boolean, optional (default=True)
            If ``True``, then we will try to return the root as an expression
            in radicals. If that is not possible, we will return a
            :py:class:`~.ComplexRootOf`.

        minpoly : :py:class:`~.Poly`
            If the minimal polynomial for `self` has been pre-computed, it can
            be passed in order to save time.

        """
        if self.is_primitive_element and not isinstance(self.root, AlgebraicNumber):
            return self.root
        m = minpoly or self.minpoly_of_element()
        roots = m.all_roots(radicals=radicals)
        if len(roots) == 1:
            return roots[0]
        ex = self.as_expr()
        for b in roots:
            if m.same_root(b, ex):
                return b


class RationalConstant(Rational):
    """
    Abstract base class for rationals with specific behaviors

    Derived classes must define class attributes p and q and should probably all
    be singletons.
    """
    __slots__ = ()

    def __new__(cls):
        return AtomicExpr.__new__(cls)


class IntegerConstant(Integer):
    __slots__ = ()

    def __new__(cls):
        return AtomicExpr.__new__(cls)


class Zero(IntegerConstant, metaclass=Singleton):
    """The number zero.

    Zero is a singleton, and can be accessed by ``S.Zero``

    Examples
    ========

    >>> from sympy import S, Integer
    >>> Integer(0) is S.Zero
    True
    >>> 1/S.Zero
    zoo

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Zero
    """

    p = 0
    q = 1
    is_positive = False
    is_negative = False
    is_zero = True
    is_number = True
    is_comparable = True

    __slots__ = ()

    def __getnewargs__(self):
        return ()

    @staticmethod
    def __abs__():
        return S.Zero

    @staticmethod
    def __neg__():
        return S.Zero

    def _eval_power(self, expt):
        if expt.is_extended_positive:
            return self
        if expt.is_extended_negative:
            return S.ComplexInfinity
        if expt.is_extended_real is False:
            return S.NaN
        if expt.is_zero:
            return S.One

        # infinities are already handled with pos and neg
        # tests above; now throw away leading numbers on Mul
        # exponent since 0**-x = zoo**x even when x == 0
        coeff, terms = expt.as_coeff_Mul()
        if coeff.is_negative:
            return S.ComplexInfinity**terms
        if coeff is not S.One:  # there is a Number to discard
            return self**terms

    def _eval_order(self, *symbols):
        # Order(0,x) -> 0
        return self

    def __bool__(self):
        return False


class One(IntegerConstant, metaclass=Singleton):
    """The number one.

    One is a singleton, and can be accessed by ``S.One``.

    Examples
    ========

    >>> from sympy import S, Integer
    >>> Integer(1) is S.One
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/1_%28number%29
    """
    is_number = True
    is_positive = True

    p = 1
    q = 1

    __slots__ = ()

    def __getnewargs__(self):
        return ()

    @staticmethod
    def __abs__():
        return S.One

    @staticmethod
    def __neg__():
        return S.NegativeOne

    def _eval_power(self, expt):
        return self

    def _eval_order(self, *symbols):
        return

    @staticmethod
    def factors(limit=None, use_trial=True, use_rho=False, use_pm1=False,
                verbose=False, visual=False):
        if visual:
            return S.One
        else:
            return {}


class NegativeOne(IntegerConstant, metaclass=Singleton):
    """The number negative one.

    NegativeOne is a singleton, and can be accessed by ``S.NegativeOne``.

    Examples
    ========

    >>> from sympy import S, Integer
    >>> Integer(-1) is S.NegativeOne
    True

    See Also
    ========

    One

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/%E2%88%921_%28number%29

    """
    is_number = True

    p = -1
    q = 1

    __slots__ = ()

    def __getnewargs__(self):
        return ()

    @staticmethod
    def __abs__():
        return S.One

    @staticmethod
    def __neg__():
        return S.One

    def _eval_power(self, expt):
        if expt.is_odd:
            return S.NegativeOne
        if expt.is_even:
            return S.One
        if isinstance(expt, Number):
            if isinstance(expt, Float):
                return Float(-1.0)**expt
            if expt is S.NaN:
                return S.NaN
            if expt in (S.Infinity, S.NegativeInfinity):
                return S.NaN
            if expt is S.Half:
                return S.ImaginaryUnit
            if isinstance(expt, Rational):
                if expt.q == 2:
                    return S.ImaginaryUnit**Integer(expt.p)
                i, r = divmod(expt.p, expt.q)
                if i:
                    return self**i*self**Rational(r, expt.q)
        return


class Half(RationalConstant, metaclass=Singleton):
    """The rational number 1/2.

    Half is a singleton, and can be accessed by ``S.Half``.

    Examples
    ========

    >>> from sympy import S, Rational
    >>> Rational(1, 2) is S.Half
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/One_half
    """
    is_number = True

    p = 1
    q = 2

    __slots__ = ()

    def __getnewargs__(self):
        return ()

    @staticmethod
    def __abs__():
        return S.Half


class Infinity(Number, metaclass=Singleton):
    r"""Positive infinite quantity.

    Explanation
    ===========

    In real analysis the symbol `\infty` denotes an unbounded
    limit: `x\to\infty` means that `x` grows without bound.

    Infinity is often used not only to define a limit but as a value
    in the affinely extended real number system.  Points labeled `+\infty`
    and `-\infty` can be added to the topological space of the real numbers,
    producing the two-point compactification of the real numbers.  Adding
    algebraic properties to this gives us the extended real numbers.

    Infinity is a singleton, and can be accessed by ``S.Infinity``,
    or can be imported as ``oo``.

    Examples
    ========

    >>> from sympy import oo, exp, limit, Symbol
    >>> 1 + oo
    oo
    >>> 42/oo
    0
    >>> x = Symbol('x')
    >>> limit(exp(x), x, oo)
    oo

    See Also
    ========

    NegativeInfinity, NaN

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Infinity
    """

    is_commutative = True
    is_number = True
    is_complex = False
    is_extended_real = True
    is_infinite = True
    is_comparable = True
    is_extended_positive = True
    is_prime = False

    __slots__ = ()

    def __new__(cls):
        return AtomicExpr.__new__(cls)

    def _latex(self, printer):
        return r"\infty"

    def _eval_subs(self, old, new):
        if self == old:
            return new

    def _eval_evalf(self, prec=None):
        return Float('inf')

    def evalf(self, prec=None, **options):
        return self._eval_evalf(prec)

    @_sympifyit('other', NotImplemented)
    def __add__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other in (S.NegativeInfinity, S.NaN):
                return S.NaN
            return self
        return Number.__add__(self, other)
    __radd__ = __add__

    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other in (S.Infinity, S.NaN):
                return S.NaN
            return self
        return Number.__sub__(self, other)

    @_sympifyit('other', NotImplemented)
    def __rsub__(self, other):
        return (-self).__add__(other)

    @_sympifyit('other', NotImplemented)
    def __mul__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other.is_zero or other is S.NaN:
                return S.NaN
            if other.is_extended_positive:
                return self
            return S.NegativeInfinity
        return Number.__mul__(self, other)
    __rmul__ = __mul__

    @_sympifyit('other', NotImplemented)
    def __truediv__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other is S.Infinity or \
                other is S.NegativeInfinity or \
                    other is S.NaN:
                return S.NaN
            if other.is_extended_nonnegative:
                return self
            return S.NegativeInfinity
        return Number.__truediv__(self, other)

    def __abs__(self):
        return S.Infinity

    def __neg__(self):
        return S.NegativeInfinity

    def _eval_power(self, expt):
        """
        ``expt`` is symbolic object but not equal to 0 or 1.

        ================ ======= ==============================
        Expression       Result  Notes
        ================ ======= ==============================
        ``oo ** nan``    ``nan``
        ``oo ** -p``     ``0``   ``p`` is number, ``oo``
        ================ ======= ==============================

        See Also
        ========
        Pow
        NaN
        NegativeInfinity

        """
        if expt.is_extended_positive:
            return S.Infinity
        if expt.is_extended_negative:
            return S.Zero
        if expt is S.NaN:
            return S.NaN
        if expt is S.ComplexInfinity:
            return S.NaN
        if expt.is_extended_real is False and expt.is_number:
            from sympy.functions.elementary.complexes import re
            expt_real = re(expt)
            if expt_real.is_positive:
                return S.ComplexInfinity
            if expt_real.is_negative:
                return S.Zero
            if expt_real.is_zero:
                return S.NaN

            return self**expt.evalf()

    def _as_mpf_val(self, prec):
        return mlib.finf

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        return other is S.Infinity or other == float('inf')

    def __ne__(self, other):
        return other is not S.Infinity and other != float('inf')

    __gt__ = Expr.__gt__
    __ge__ = Expr.__ge__
    __lt__ = Expr.__lt__
    __le__ = Expr.__le__

    @_sympifyit('other', NotImplemented)
    def __mod__(self, other):
        if not isinstance(other, Expr):
            return NotImplemented
        return S.NaN

    __rmod__ = __mod__

    def floor(self):
        return self

    def ceiling(self):
        return self

oo = S.Infinity


class NegativeInfinity(Number, metaclass=Singleton):
    """Negative infinite quantity.

    NegativeInfinity is a singleton, and can be accessed
    by ``S.NegativeInfinity``.

    See Also
    ========

    Infinity
    """

    is_extended_real = True
    is_complex = False
    is_commutative = True
    is_infinite = True
    is_comparable = True
    is_extended_negative = True
    is_number = True
    is_prime = False

    __slots__ = ()

    def __new__(cls):
        return AtomicExpr.__new__(cls)

    def _latex(self, printer):
        return r"-\infty"

    def _eval_subs(self, old, new):
        if self == old:
            return new

    def _eval_evalf(self, prec=None):
        return Float('-inf')

    def evalf(self, prec=None, **options):
        return self._eval_evalf(prec)

    @_sympifyit('other', NotImplemented)
    def __add__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other in (S.Infinity, S.NaN):
                return S.NaN
            return self
        return Number.__add__(self, other)
    __radd__ = __add__

    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other in (S.NegativeInfinity, S.NaN):
                return S.NaN
            return self
        return Number.__sub__(self, other)

    @_sympifyit('other', NotImplemented)
    def __rsub__(self, other):
        return (-self).__add__(other)

    @_sympifyit('other', NotImplemented)
    def __mul__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other.is_zero or other is S.NaN:
                return S.NaN
            if other.is_extended_positive:
                return self
            return S.Infinity
        return Number.__mul__(self, other)
    __rmul__ = __mul__

    @_sympifyit('other', NotImplemented)
    def __truediv__(self, other):
        if isinstance(other, Number) and global_parameters.evaluate:
            if other is S.Infinity or \
                other is S.NegativeInfinity or \
                    other is S.NaN:
                return S.NaN
            if other.is_extended_nonnegative:
                return self
            return S.Infinity
        return Number.__truediv__(self, other)

    def __abs__(self):
        return S.Infinity

    def __neg__(self):
        return S.Infinity

    def _eval_power(self, expt):
        """
        ``expt`` is symbolic object but not equal to 0 or 1.

        ================ ======= ==============================
        Expression       Result  Notes
        ================ ======= ==============================
        ``(-oo) ** nan`` ``nan``
        ``(-oo) ** oo``  ``nan``
        ``(-oo) ** -oo`` ``nan``
        ``(-oo) ** e``   ``oo``  ``e`` is positive even integer
        ``(-oo) ** o``   ``-oo`` ``o`` is positive odd integer
        ================ ======= ==============================

        See Also
        ========

        Infinity
        Pow
        NaN

        """
        if expt.is_number:
            if expt is S.NaN or \
                expt is S.Infinity or \
                    expt is S.NegativeInfinity:
                return S.NaN

            if isinstance(expt, Integer) and expt.is_extended_positive:
                if expt.is_odd:
                    return S.NegativeInfinity
                else:
                    return S.Infinity

            inf_part = S.Infinity**expt
            s_part = S.NegativeOne**expt
            if inf_part == 0 and s_part.is_finite:
                return inf_part
            if (inf_part is S.ComplexInfinity and
                    s_part.is_finite and not s_part.is_zero):
                return S.ComplexInfinity
            return s_part*inf_part

    def _as_mpf_val(self, prec):
        return mlib.fninf

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        return other is S.NegativeInfinity or other == float('-inf')

    def __ne__(self, other):
        return other is not S.NegativeInfinity and other != float('-inf')

    __gt__ = Expr.__gt__
    __ge__ = Expr.__ge__
    __lt__ = Expr.__lt__
    __le__ = Expr.__le__

    @_sympifyit('other', NotImplemented)
    def __mod__(self, other):
        if not isinstance(other, Expr):
            return NotImplemented
        return S.NaN

    __rmod__ = __mod__

    def floor(self):
        return self

    def ceiling(self):
        return self

    def as_powers_dict(self):
        return {S.NegativeOne: 1, S.Infinity: 1}


class NaN(Number, metaclass=Singleton):
    """
    Not a Number.

    Explanation
    ===========

    This serves as a place holder for numeric values that are indeterminate.
    Most operations on NaN, produce another NaN.  Most indeterminate forms,
    such as ``0/0`` or ``oo - oo` produce NaN.  Two exceptions are ``0**0``
    and ``oo**0``, which all produce ``1`` (this is consistent with Python's
    float).

    NaN is loosely related to floating point nan, which is defined in the
    IEEE 754 floating point standard, and corresponds to the Python
    ``float('nan')``.  Differences are noted below.

    NaN is mathematically not equal to anything else, even NaN itself.  This
    explains the initially counter-intuitive results with ``Eq`` and ``==`` in
    the examples below.

    NaN is not comparable so inequalities raise a TypeError.  This is in
    contrast with floating point nan where all inequalities are false.

    NaN is a singleton, and can be accessed by ``S.NaN``, or can be imported
    as ``nan``.

    Examples
    ========

    >>> from sympy import nan, S, oo, Eq
    >>> nan is S.NaN
    True
    >>> oo - oo
    nan
    >>> nan + 1
    nan
    >>> Eq(nan, nan)   # mathematical equality
    False
    >>> nan == nan     # structural equality
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/NaN

    """
    is_commutative = True
    is_extended_real = None
    is_real = None
    is_rational = None
    is_algebraic = None
    is_transcendental = None
    is_integer = None
    is_comparable = False
    is_finite = None
    is_zero = None
    is_prime = None
    is_positive = None
    is_negative = None
    is_number = True

    __slots__ = ()

    def __new__(cls):
        return AtomicExpr.__new__(cls)

    def _latex(self, printer):
        return r"\text{NaN}"

    def __neg__(self):
        return self

    @_sympifyit('other', NotImplemented)
    def __add__(self, other):
        return self

    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        return self

    @_sympifyit('other', NotImplemented)
    def __mul__(self, other):
        return self

    @_sympifyit('other', NotImplemented)
    def __truediv__(self, other):
        return self

    def floor(self):
        return self

    def ceiling(self):
        return self

    def _as_mpf_val(self, prec):
        return _mpf_nan

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        # NaN is structurally equal to another NaN
        return other is S.NaN

    def __ne__(self, other):
        return other is not S.NaN

    # Expr will _sympify and raise TypeError
    __gt__ = Expr.__gt__
    __ge__ = Expr.__ge__
    __lt__ = Expr.__lt__
    __le__ = Expr.__le__

nan = S.NaN

@dispatch(NaN, Expr) # type:ignore
def _eval_is_eq(a, b): # noqa:F811
    return False


class ComplexInfinity(AtomicExpr, metaclass=Singleton):
    r"""Complex infinity.

    Explanation
    ===========

    In complex analysis the symbol `\tilde\infty`, called "complex
    infinity", represents a quantity with infinite magnitude, but
    undetermined complex phase.

    ComplexInfinity is a singleton, and can be accessed by
    ``S.ComplexInfinity``, or can be imported as ``zoo``.

    Examples
    ========

    >>> from sympy import zoo
    >>> zoo + 42
    zoo
    >>> 42/zoo
    0
    >>> zoo + zoo
    nan
    >>> zoo*zoo
    zoo

    See Also
    ========

    Infinity
    """

    is_commutative = True
    is_infinite = True
    is_number = True
    is_prime = False
    is_complex = False
    is_extended_real = False

    kind = NumberKind

    __slots__ = ()

    def __new__(cls):
        return AtomicExpr.__new__(cls)

    def _latex(self, printer):
        return r"\tilde{\infty}"

    @staticmethod
    def __abs__():
        return S.Infinity

    def floor(self):
        return self

    def ceiling(self):
        return self

    @staticmethod
    def __neg__():
        return S.ComplexInfinity

    def _eval_power(self, expt):
        if expt is S.ComplexInfinity:
            return S.NaN

        if isinstance(expt, Number):
            if expt.is_zero:
                return S.NaN
            else:
                if expt.is_positive:
                    return S.ComplexInfinity
                else:
                    return S.Zero


zoo = S.ComplexInfinity


class NumberSymbol(AtomicExpr):

    is_commutative = True
    is_finite = True
    is_number = True

    __slots__ = ()

    is_NumberSymbol = True

    kind = NumberKind

    def __new__(cls):
        return AtomicExpr.__new__(cls)

    def approximation(self, number_cls):
        """ Return an interval with number_cls endpoints
        that contains the value of NumberSymbol.
        If not implemented, then return None.
        """

    def _eval_evalf(self, prec):
        return Float._new(self._as_mpf_val(prec), prec)

    def __eq__(self, other):
        try:
            other = _sympify(other)
        except SympifyError:
            return NotImplemented
        if self is other:
            return True
        if other.is_Number and self.is_irrational:
            return False

        return False    # NumberSymbol != non-(Number|self)

    def __ne__(self, other):
        return not self == other

    def __le__(self, other):
        if self is other:
            return S.true
        return Expr.__le__(self, other)

    def __ge__(self, other):
        if self is other:
            return S.true
        return Expr.__ge__(self, other)

    def __int__(self):
        # subclass with appropriate return value
        raise NotImplementedError

    def __hash__(self):
        return super().__hash__()


class Exp1(NumberSymbol, metaclass=Singleton):
    r"""The `e` constant.

    Explanation
    ===========

    The transcendental number `e = 2.718281828\ldots` is the base of the
    natural logarithm and of the exponential function, `e = \exp(1)`.
    Sometimes called Euler's number or Napier's constant.

    Exp1 is a singleton, and can be accessed by ``S.Exp1``,
    or can be imported as ``E``.

    Examples
    ========

    >>> from sympy import exp, log, E
    >>> E is exp(1)
    True
    >>> log(E)
    1

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/E_%28mathematical_constant%29
    """

    is_real = True
    is_positive = True
    is_negative = False  # XXX Forces is_negative/is_nonnegative
    is_irrational = True
    is_number = True
    is_algebraic = False
    is_transcendental = True

    __slots__ = ()

    def _latex(self, printer):
        return r"e"

    @staticmethod
    def __abs__():
        return S.Exp1

    def __int__(self):
        return 2

    def _as_mpf_val(self, prec):
        return mpf_e(prec)

    def approximation_interval(self, number_cls):
        if issubclass(number_cls, Integer):
            return (Integer(2), Integer(3))
        elif issubclass(number_cls, Rational):
            pass

    def _eval_power(self, expt):
        if global_parameters.exp_is_pow:
            return self._eval_power_exp_is_pow(expt)
        else:
            from sympy.functions.elementary.exponential import exp
            return exp(expt)

    def _eval_power_exp_is_pow(self, arg):
        if arg.is_Number:
            if arg is oo:
                return oo
            elif arg == -oo:
                return S.Zero
        from sympy.functions.elementary.exponential import log
        if isinstance(arg, log):
            return arg.args[0]

        # don't autoexpand Pow or Mul (see the issue 3351):
        elif not arg.is_Add:
            Ioo = I*oo
            if arg in [Ioo, -Ioo]:
                return nan

            coeff = arg.coeff(pi*I)
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
                        return S.Exp1**(ncoeff*S.Pi*S.ImaginaryUnit)

            # Warning: code in risch.py will be very sensitive to changes
            # in this (see DifferentialExtension).

            # look for a single log factor

            coeff, terms = arg.as_coeff_Mul()

            # but it can't be multiplied by oo
            if coeff in (oo, -oo):
                return

            coeffs, log_term = [coeff], None
            for term in Mul.make_args(terms):
                if isinstance(term, log):
                    if log_term is None:
                        log_term = term.args[0]
                    else:
                        return
                elif term.is_comparable:
                    coeffs.append(term)
                else:
                    return

            return log_term**Mul(*coeffs) if log_term else None
        elif arg.is_Add:
            out = []
            add = []
            argchanged = False
            for a in arg.args:
                if a is S.One:
                    add.append(a)
                    continue
                newa = self**a
                if isinstance(newa, Pow) and newa.base is self:
                    if newa.exp != a:
                        add.append(newa.exp)
                        argchanged = True
                    else:
                        add.append(a)
                else:
                    out.append(newa)
            if out or argchanged:
                return Mul(*out)*Pow(self, Add(*add), evaluate=False)
        elif arg.is_Matrix:
            return arg.exp()

    def _eval_rewrite_as_sin(self, **kwargs):
        from sympy.functions.elementary.trigonometric import sin
        return sin(I + S.Pi/2) - I*sin(I)

    def _eval_rewrite_as_cos(self, **kwargs):
        from sympy.functions.elementary.trigonometric import cos
        return cos(I) + I*cos(I + S.Pi/2)

E = S.Exp1


class Pi(NumberSymbol, metaclass=Singleton):
    r"""The `\pi` constant.

    Explanation
    ===========

    The transcendental number `\pi = 3.141592654\ldots` represents the ratio
    of a circle's circumference to its diameter, the area of the unit circle,
    the half-period of trigonometric functions, and many other things
    in mathematics.

    Pi is a singleton, and can be accessed by ``S.Pi``, or can
    be imported as ``pi``.

    Examples
    ========

    >>> from sympy import S, pi, oo, sin, exp, integrate, Symbol
    >>> S.Pi
    pi
    >>> pi > 3
    True
    >>> pi.is_irrational
    True
    >>> x = Symbol('x')
    >>> sin(x + 2*pi)
    sin(x)
    >>> integrate(exp(-x**2), (x, -oo, oo))
    sqrt(pi)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Pi
    """

    is_real = True
    is_positive = True
    is_negative = False
    is_irrational = True
    is_number = True
    is_algebraic = False
    is_transcendental = True

    __slots__ = ()

    def _latex(self, printer):
        return r"\pi"

    @staticmethod
    def __abs__():
        return S.Pi

    def __int__(self):
        return 3

    def _as_mpf_val(self, prec):
        return mpf_pi(prec)

    def approximation_interval(self, number_cls):
        if issubclass(number_cls, Integer):
            return (Integer(3), Integer(4))
        elif issubclass(number_cls, Rational):
            return (Rational(223, 71, 1), Rational(22, 7, 1))

pi = S.Pi


class GoldenRatio(NumberSymbol, metaclass=Singleton):
    r"""The golden ratio, `\phi`.

    Explanation
    ===========

    `\phi = \frac{1 + \sqrt{5}}{2}` is an algebraic number.  Two quantities
    are in the golden ratio if their ratio is the same as the ratio of
    their sum to the larger of the two quantities, i.e. their maximum.

    GoldenRatio is a singleton, and can be accessed by ``S.GoldenRatio``.

    Examples
    ========

    >>> from sympy import S
    >>> S.GoldenRatio > 1
    True
    >>> S.GoldenRatio.expand(func=True)
    1/2 + sqrt(5)/2
    >>> S.GoldenRatio.is_irrational
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Golden_ratio
    """

    is_real = True
    is_positive = True
    is_negative = False
    is_irrational = True
    is_number = True
    is_algebraic = True
    is_transcendental = False

    __slots__ = ()

    def _latex(self, printer):
        return r"\phi"

    def __int__(self):
        return 1

    def _as_mpf_val(self, prec):
         # XXX track down why this has to be increased
        rv = mlib.from_man_exp(phi_fixed(prec + 10), -prec - 10)
        return mpf_norm(rv, prec)

    def _eval_expand_func(self, **hints):
        from sympy.functions.elementary.miscellaneous import sqrt
        return S.Half + S.Half*sqrt(5)

    def approximation_interval(self, number_cls):
        if issubclass(number_cls, Integer):
            return (S.One, Rational(2))
        elif issubclass(number_cls, Rational):
            pass

    _eval_rewrite_as_sqrt = _eval_expand_func


class TribonacciConstant(NumberSymbol, metaclass=Singleton):
    r"""The tribonacci constant.

    Explanation
    ===========

    The tribonacci numbers are like the Fibonacci numbers, but instead
    of starting with two predetermined terms, the sequence starts with
    three predetermined terms and each term afterwards is the sum of the
    preceding three terms.

    The tribonacci constant is the ratio toward which adjacent tribonacci
    numbers tend. It is a root of the polynomial `x^3 - x^2 - x - 1 = 0`,
    and also satisfies the equation `x + x^{-3} = 2`.

    TribonacciConstant is a singleton, and can be accessed
    by ``S.TribonacciConstant``.

    Examples
    ========

    >>> from sympy import S
    >>> S.TribonacciConstant > 1
    True
    >>> S.TribonacciConstant.expand(func=True)
    1/3 + (19 - 3*sqrt(33))**(1/3)/3 + (3*sqrt(33) + 19)**(1/3)/3
    >>> S.TribonacciConstant.is_irrational
    True
    >>> S.TribonacciConstant.n(20)
    1.8392867552141611326

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Generalizations_of_Fibonacci_numbers#Tribonacci_numbers
    """

    is_real = True
    is_positive = True
    is_negative = False
    is_irrational = True
    is_number = True
    is_algebraic = True
    is_transcendental = False

    __slots__ = ()

    def _latex(self, printer):
        return r"\text{TribonacciConstant}"

    def __int__(self):
        return 1

    def _as_mpf_val(self, prec):
        return self._eval_evalf(prec)._mpf_

    def _eval_evalf(self, prec):
        rv = self._eval_expand_func(function=True)._eval_evalf(prec + 4)
        return Float(rv, precision=prec)

    def _eval_expand_func(self, **hints):
        from sympy.functions.elementary.miscellaneous import cbrt, sqrt
        return (1 + cbrt(19 - 3*sqrt(33)) + cbrt(19 + 3*sqrt(33))) / 3

    def approximation_interval(self, number_cls):
        if issubclass(number_cls, Integer):
            return (S.One, Rational(2))
        elif issubclass(number_cls, Rational):
            pass

    _eval_rewrite_as_sqrt = _eval_expand_func


class EulerGamma(NumberSymbol, metaclass=Singleton):
    r"""The Euler-Mascheroni constant.

    Explanation
    ===========

    `\gamma = 0.5772157\ldots` (also called Euler's constant) is a mathematical
    constant recurring in analysis and number theory.  It is defined as the
    limiting difference between the harmonic series and the
    natural logarithm:

    .. math:: \gamma = \lim\limits_{n\to\infty}
              \left(\sum\limits_{k=1}^n\frac{1}{k} - \ln n\right)

    EulerGamma is a singleton, and can be accessed by ``S.EulerGamma``.

    Examples
    ========

    >>> from sympy import S
    >>> S.EulerGamma.is_irrational
    >>> S.EulerGamma > 0
    True
    >>> S.EulerGamma > 1
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euler%E2%80%93Mascheroni_constant
    """

    is_real = True
    is_positive = True
    is_negative = False
    is_irrational = None
    is_number = True

    __slots__ = ()

    def _latex(self, printer):
        return r"\gamma"

    def __int__(self):
        return 0

    def _as_mpf_val(self, prec):
         # XXX track down why this has to be increased
        v = mlib.libhyper.euler_fixed(prec + 10)
        rv = mlib.from_man_exp(v, -prec - 10)
        return mpf_norm(rv, prec)

    def approximation_interval(self, number_cls):
        if issubclass(number_cls, Integer):
            return (S.Zero, S.One)
        elif issubclass(number_cls, Rational):
            return (S.Half, Rational(3, 5, 1))


class Catalan(NumberSymbol, metaclass=Singleton):
    r"""Catalan's constant.

    Explanation
    ===========

    $G = 0.91596559\ldots$ is given by the infinite series

    .. math:: G = \sum_{k=0}^{\infty} \frac{(-1)^k}{(2k+1)^2}

    Catalan is a singleton, and can be accessed by ``S.Catalan``.

    Examples
    ========

    >>> from sympy import S
    >>> S.Catalan.is_irrational
    >>> S.Catalan > 0
    True
    >>> S.Catalan > 1
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Catalan%27s_constant
    """

    is_real = True
    is_positive = True
    is_negative = False
    is_irrational = None
    is_number = True

    __slots__ = ()

    def __int__(self):
        return 0

    def _as_mpf_val(self, prec):
        # XXX track down why this has to be increased
        v = mlib.catalan_fixed(prec + 10)
        rv = mlib.from_man_exp(v, -prec - 10)
        return mpf_norm(rv, prec)

    def approximation_interval(self, number_cls):
        if issubclass(number_cls, Integer):
            return (S.Zero, S.One)
        elif issubclass(number_cls, Rational):
            return (Rational(9, 10, 1), S.One)

    def _eval_rewrite_as_Sum(self, k_sym=None, symbols=None, **hints):
        if (k_sym is not None) or (symbols is not None):
            return self
        from .symbol import Dummy
        from sympy.concrete.summations import Sum
        k = Dummy('k', integer=True, nonnegative=True)
        return Sum(S.NegativeOne**k / (2*k+1)**2, (k, 0, S.Infinity))

    def _latex(self, printer):
        return "G"


class ImaginaryUnit(AtomicExpr, metaclass=Singleton):
    r"""The imaginary unit, `i = \sqrt{-1}`.

    I is a singleton, and can be accessed by ``S.I``, or can be
    imported as ``I``.

    Examples
    ========

    >>> from sympy import I, sqrt
    >>> sqrt(-1)
    I
    >>> I*I
    -1
    >>> 1/I
    -I

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Imaginary_unit
    """

    is_commutative = True
    is_imaginary = True
    is_finite = True
    is_number = True
    is_algebraic = True
    is_transcendental = False

    kind = NumberKind

    __slots__ = ()

    def _latex(self, printer):
        return printer._settings['imaginary_unit_latex']

    @staticmethod
    def __abs__():
        return S.One

    def _eval_evalf(self, prec):
        return self

    def _eval_conjugate(self):
        return -S.ImaginaryUnit

    def _eval_power(self, expt):
        """
        b is I = sqrt(-1)
        e is symbolic object but not equal to 0, 1

        I**r -> (-1)**(r/2) -> exp(r/2*Pi*I) -> sin(Pi*r/2) + cos(Pi*r/2)*I, r is decimal
        I**0 mod 4 -> 1
        I**1 mod 4 -> I
        I**2 mod 4 -> -1
        I**3 mod 4 -> -I
        """

        if isinstance(expt, Integer):
            expt = expt % 4
            if expt == 0:
                return S.One
            elif expt == 1:
                return S.ImaginaryUnit
            elif expt == 2:
                return S.NegativeOne
            elif expt == 3:
                return -S.ImaginaryUnit
        if isinstance(expt, Rational):
            i, r = divmod(expt, 2)
            rv = Pow(S.ImaginaryUnit, r, evaluate=False)
            if i % 2:
                return Mul(S.NegativeOne, rv, evaluate=False)
            return rv

    def as_base_exp(self):
        return S.NegativeOne, S.Half

    @property
    def _mpc_(self):
        return (Float(0)._mpf_, Float(1)._mpf_)


I = S.ImaginaryUnit


def int_valued(x):
    """return True only for a literal Number whose internal
    representation as a fraction has a denominator of 1,
    else False, i.e. integer, with no fractional part.
    """
    if isinstance(x, (SYMPY_INTS, int)):
        return True
    if type(x) is float:
        return x.is_integer()
    if isinstance(x, Integer):
        return True
    if isinstance(x, Float):
        # x = s*m*2**p; _mpf_ = s,m,e,p
        return x._mpf_[2] >= 0
    return False  # or add new types to recognize


def equal_valued(x, y):
    """Compare expressions treating plain floats as rationals.

    Examples
    ========

    >>> from sympy import S, symbols, Rational, Float
    >>> from sympy.core.numbers import equal_valued
    >>> equal_valued(1, 2)
    False
    >>> equal_valued(1, 1)
    True

    In SymPy expressions with Floats compare unequal to corresponding
    expressions with rationals:

    >>> x = symbols('x')
    >>> x**2 == x**2.0
    False

    However an individual Float compares equal to a Rational:

    >>> Rational(1, 2) == Float(0.5)
    False

    In a future version of SymPy this might change so that Rational and Float
    compare unequal. This function provides the behavior currently expected of
    ``==`` so that it could still be used if the behavior of ``==`` were to
    change in future.

    >>> equal_valued(1, 1.0) # Float vs Rational
    True
    >>> equal_valued(S(1).n(3), S(1).n(5)) # Floats of different precision
    True

    Explanation
    ===========

    In future SymPy versions Float and Rational might compare unequal and floats
    with different precisions might compare unequal. In that context a function
    is needed that can check if a number is equal to 1 or 0 etc. The idea is
    that instead of testing ``if x == 1:`` if we want to accept floats like
    ``1.0`` as well then the test can be written as ``if equal_valued(x, 1):``
    or ``if equal_valued(x, 2):``. Since this function is intended to be used
    in situations where one or both operands are expected to be concrete
    numbers like 1 or 0 the function does not recurse through the args of any
    compound expression to compare any nested floats.

    References
    ==========

    .. [1] https://github.com/sympy/sympy/pull/20033
    """
    x = _sympify(x)
    y = _sympify(y)

    # Handle everything except Float/Rational first
    if not x.is_Float and not y.is_Float:
        return x == y
    elif x.is_Float and y.is_Float:
        # Compare values without regard for precision
        return x._mpf_ == y._mpf_
    elif x.is_Float:
        x, y = y, x
    if not x.is_Rational:
        return False

    # Now y is Float and x is Rational. A simple approach at this point would
    # just be x == Rational(y) but if y has a large exponent creating a
    # Rational could be prohibitively expensive.

    sign, man, exp, _ = y._mpf_
    p, q = x.p, x.q

    if sign:
        man = -man

    if exp == 0:
        # y odd integer
        return q == 1 and man == p
    elif exp > 0:
        # y even integer
        if q != 1:
            return False
        if p.bit_length() != man.bit_length() + exp:
            return False
        return man << exp == p
    else:
        # y non-integer. Need p == man and q == 2**-exp
        if p != man:
            return False
        neg_exp = -exp
        if q.bit_length() - 1 != neg_exp:
            return False
        return (1 << neg_exp) == q


def all_close(expr1, expr2, rtol=1e-5, atol=1e-8):
    """Return True if expr1 and expr2 are numerically close.

    The expressions must have the same structure, but any Rational, Integer, or
    Float numbers they contain are compared approximately using rtol and atol.
    Any other parts of expressions are compared exactly. However, allowance is
    made to allow for the additive and multiplicative identities.

    Relative tolerance is measured with respect to expr2 so when used in
    testing expr2 should be the expected correct answer.

    Examples
    ========

    >>> from sympy import exp
    >>> from sympy.abc import x, y
    >>> from sympy.core.numbers import all_close
    >>> expr1 = 0.1*exp(x - y)
    >>> expr2 = exp(x - y)/10
    >>> expr1
    0.1*exp(x - y)
    >>> expr2
    exp(x - y)/10
    >>> expr1 == expr2
    False
    >>> all_close(expr1, expr2)
    True

    Identities are automatically supplied:

    >>> all_close(x, x + 1e-10)
    True
    >>> all_close(x, 1.0*x)
    True
    >>> all_close(x, 1.0*x + 1e-10)
    True

    """
    NUM_TYPES = (Rational, Float)

    def _all_close(obj1, obj2):
        if type(obj1) == type(obj2) and isinstance(obj1, (list, tuple)):
            if len(obj1) != len(obj2):
                return False
            return all(_all_close(e1, e2) for e1, e2 in zip(obj1, obj2))
        else:
            return _all_close_expr(_sympify(obj1), _sympify(obj2))

    def _all_close_expr(expr1, expr2):
        num1 = isinstance(expr1, NUM_TYPES)
        num2 = isinstance(expr2, NUM_TYPES)
        if num1 != num2:
            return False
        elif num1:
            return _close_num(expr1, expr2)
        if expr1.is_Add or expr1.is_Mul or expr2.is_Add or expr2.is_Mul:
            return _all_close_ac(expr1, expr2)
        if expr1.func != expr2.func or len(expr1.args) != len(expr2.args):
            return False
        args = zip(expr1.args, expr2.args)
        return all(_all_close_expr(a1, a2) for a1, a2 in args)

    def _close_num(num1, num2):
        return bool(abs(num1 - num2) <= atol + rtol*abs(num2))

    def _all_close_ac(expr1, expr2):
        # compare expressions with associative commutative operators for
        # approximate equality by seeing that all terms have equivalent
        # coefficients (which are always Rational or Float)
        if expr1.is_Mul or expr2.is_Mul:
            # as_coeff_mul automatically will supply coeff of 1
            c1, e1 = expr1.as_coeff_mul(rational=False)
            c2, e2 = expr2.as_coeff_mul(rational=False)
            if not _close_num(c1, c2):
                return False
            s1 = set(e1)
            s2 = set(e2)
            common = s1 & s2
            s1 -= common
            s2 -= common
            if not s1:
                return True
            if not any(i.has(Float) for j in (s1, s2) for i in j):
                return False
            # factors might not be matching, e.g.
            # x != x**1.0, exp(x) != exp(1.0*x), etc...
            s1 = [i.as_base_exp() for i in ordered(s1)]
            s2 = [i.as_base_exp() for i in ordered(s2)]
            unmatched = list(range(len(s1)))
            for be1 in s1:
                for i in unmatched:
                    be2 = s2[i]
                    if _all_close(be1, be2):
                        unmatched.remove(i)
                        break
                else:
                    return False
            return not(unmatched)
        assert expr1.is_Add or expr2.is_Add
        cd1 = expr1.as_coefficients_dict()
        cd2 = expr2.as_coefficients_dict()
        # this test will assure that the key of 1 is in
        # each dict and that they have equal values
        if not _close_num(cd1[1], cd2[1]):
            return False
        if len(cd1) != len(cd2):
            return False
        for k in list(cd1):
            if k in cd2:
                if not _close_num(cd1.pop(k), cd2.pop(k)):
                    return False
            # k (or a close version in cd2) might have
            # Floats in a factor of the term which will
            # be handled below
        else:
            if not cd1:
                return True
        for k1 in cd1:
            for k2 in cd2:
                if _all_close_expr(k1, k2):
                    # found a matching key
                    # XXX there could be a corner case where
                    # more than 1 might match and the numbers are
                    # such that one is better than the other
                    # that is not being considered here
                    if not _close_num(cd1[k1], cd2[k2]):
                        return False
                    break
            else:
                # no key matched
                return False
        return True

    return _all_close(expr1, expr2)


@dispatch(Tuple, Number) # type:ignore
def _eval_is_eq(self, other): # noqa: F811
    return False


def sympify_fractions(f):
    return Rational._new(f.numerator, f.denominator, 1)

_sympy_converter[fractions.Fraction] = sympify_fractions


if gmpy is not None:

    def sympify_mpz(x):
        return Integer(int(x))

    def sympify_mpq(x):
        return Rational(int(x.numerator), int(x.denominator))

    _sympy_converter[type(gmpy.mpz(1))] = sympify_mpz
    _sympy_converter[type(gmpy.mpq(1, 2))] = sympify_mpq


if flint is not None:

    def sympify_fmpz(x):
        return Integer(int(x))

    def sympify_fmpq(x):
        return Rational(int(x.numerator), int(x.denominator))

    _sympy_converter[type(flint.fmpz(1))] = sympify_fmpz
    _sympy_converter[type(flint.fmpq(1, 2))] = sympify_fmpq


def sympify_mpmath(x):
    return Expr._from_mpmath(x, x.context.prec)

_sympy_converter[mpnumeric] = sympify_mpmath


def sympify_complex(a):
    real, imag = list(map(sympify, (a.real, a.imag)))
    return real + S.ImaginaryUnit*imag

_sympy_converter[complex] = sympify_complex

from .power import Pow
from .mul import Mul
Mul.identity = One()
from .add import Add
Add.identity = Zero()


def _register_classes():
    numbers.Number.register(Number)
    numbers.Real.register(Float)
    numbers.Rational.register(Rational)
    numbers.Integral.register(Integer)

_register_classes()

_illegal = (S.NaN, S.Infinity, S.NegativeInfinity, S.ComplexInfinity)
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar
from collections import defaultdict
from functools import reduce
from operator import attrgetter
from .basic import _args_sortkey
from .parameters import global_parameters
from .logic import _fuzzy_group, fuzzy_or, fuzzy_not
from .singleton import S
from .operations import AssocOp, AssocOpDispatcher
from .cache import cacheit
from .intfunc import ilcm, igcd
from .expr import Expr
from .kind import UndefinedKind
from sympy.utilities.iterables import is_sequence, sift


if TYPE_CHECKING:
    from sympy.core.numbers import Number
    from sympy.series.order import Order


def _could_extract_minus_sign(expr):
    # assume expr is Add-like
    # We choose the one with less arguments with minus signs
    negative_args = sum(1 for i in expr.args
        if i.could_extract_minus_sign())
    positive_args = len(expr.args) - negative_args
    if positive_args > negative_args:
        return False
    elif positive_args < negative_args:
        return True
    # choose based on .sort_key() to prefer
    # x - 1 instead of 1 - x and
    # 3 - sqrt(2) instead of -3 + sqrt(2)
    return bool(expr.sort_key() < (-expr).sort_key())


def _addsort(args):
    # in-place sorting of args
    args.sort(key=_args_sortkey)


def _unevaluated_Add(*args):
    """Return a well-formed unevaluated Add: Numbers are collected and
    put in slot 0 and args are sorted. Use this when args have changed
    but you still want to return an unevaluated Add.

    Examples
    ========

    >>> from sympy.core.add import _unevaluated_Add as uAdd
    >>> from sympy import S, Add
    >>> from sympy.abc import x, y
    >>> a = uAdd(*[S(1.0), x, S(2)])
    >>> a.args[0]
    3.00000000000000
    >>> a.args[1]
    x

    Beyond the Number being in slot 0, there is no other assurance of
    order for the arguments since they are hash sorted. So, for testing
    purposes, output produced by this in some other function can only
    be tested against the output of this function or as one of several
    options:

    >>> opts = (Add(x, y, evaluate=False), Add(y, x, evaluate=False))
    >>> a = uAdd(x, y)
    >>> assert a in opts and a == uAdd(x, y)
    >>> uAdd(x + 1, x + 2)
    x + x + 3
    """
    args = list(args)
    newargs = []
    co = S.Zero
    while args:
        a = args.pop()
        if a.is_Add:
            # this will keep nesting from building up
            # so that x + (x + 1) -> x + x + 1 (3 args)
            args.extend(a.args)
        elif a.is_Number:
            co += a
        else:
            newargs.append(a)
    _addsort(newargs)
    if co:
        newargs.insert(0, co)
    return Add._from_args(newargs)


class Add(Expr, AssocOp):
    """
    Expression representing addition operation for algebraic group.

    .. deprecated:: 1.7

       Using arguments that aren't subclasses of :class:`~.Expr` in core
       operators (:class:`~.Mul`, :class:`~.Add`, and :class:`~.Pow`) is
       deprecated. See :ref:`non-expr-args-deprecated` for details.

    Every argument of ``Add()`` must be ``Expr``. Infix operator ``+``
    on most scalar objects in SymPy calls this class.

    Another use of ``Add()`` is to represent the structure of abstract
    addition so that its arguments can be substituted to return different
    class. Refer to examples section for this.

    ``Add()`` evaluates the argument unless ``evaluate=False`` is passed.
    The evaluation logic includes:

    1. Flattening
        ``Add(x, Add(y, z))`` -> ``Add(x, y, z)``

    2. Identity removing
        ``Add(x, 0, y)`` -> ``Add(x, y)``

    3. Coefficient collecting by ``.as_coeff_Mul()``
        ``Add(x, 2*x)`` -> ``Mul(3, x)``

    4. Term sorting
        ``Add(y, x, 2)`` -> ``Add(2, x, y)``

    If no argument is passed, identity element 0 is returned. If single
    element is passed, that element is returned.

    Note that ``Add(*args)`` is more efficient than ``sum(args)`` because
    it flattens the arguments. ``sum(a, b, c, ...)`` recursively adds the
    arguments as ``a + (b + (c + ...))``, which has quadratic complexity.
    On the other hand, ``Add(a, b, c, d)`` does not assume nested
    structure, making the complexity linear.

    Since addition is group operation, every argument should have the
    same :obj:`sympy.core.kind.Kind()`.

    Examples
    ========

    >>> from sympy import Add, I
    >>> from sympy.abc import x, y
    >>> Add(x, 1)
    x + 1
    >>> Add(x, x)
    2*x
    >>> 2*x**2 + 3*x + I*y + 2*y + 2*x/5 + 1.0*y + 1
    2*x**2 + 17*x/5 + 3.0*y + I*y + 1

    If ``evaluate=False`` is passed, result is not evaluated.

    >>> Add(1, 2, evaluate=False)
    1 + 2
    >>> Add(x, x, evaluate=False)
    x + x

    ``Add()`` also represents the general structure of addition operation.

    >>> from sympy import MatrixSymbol
    >>> A,B = MatrixSymbol('A', 2,2), MatrixSymbol('B', 2,2)
    >>> expr = Add(x,y).subs({x:A, y:B})
    >>> expr
    A + B
    >>> type(expr)
    <class 'sympy.matrices.expressions.matadd.MatAdd'>

    Note that the printers do not display in args order.

    >>> Add(x, 1)
    x + 1
    >>> Add(x, 1).args
    (1, x)

    See Also
    ========

    MatAdd

    """

    __slots__ = ()

    is_Add = True

    _args_type = Expr

    identity: ClassVar[Expr]

    if TYPE_CHECKING:

        def __new__(cls, *args: Expr | complex, evaluate: bool=True) -> Expr: # type: ignore
            ...

        @property
        def args(self) -> tuple[Expr, ...]:
            ...

    @classmethod
    def flatten(cls, seq: list[Expr]) -> tuple[list[Expr], list[Expr], None]:
        """
        Takes the sequence "seq" of nested Adds and returns a flatten list.

        Returns: (commutative_part, noncommutative_part, order_symbols)

        Applies associativity, all terms are commutable with respect to
        addition.

        NB: the removal of 0 is already handled by AssocOp.__new__

        See Also
        ========

        sympy.core.mul.Mul.flatten

        """
        from sympy.calculus.accumulationbounds import AccumBounds
        from sympy.matrices.expressions import MatrixExpr
        from sympy.tensor.tensor import TensExpr, TensAdd
        rv = None
        if len(seq) == 2:
            a, b = seq
            if b.is_Rational:
                a, b = b, a
            if a.is_Rational:
                if b.is_Mul:
                    rv = [a, b], [], None
            if rv:
                if all(s.is_commutative for s in rv[0]):
                    return rv
                return [], rv[0], None

        # term -> coeff
        # e.g. x**2 -> 5   for ... + 5*x**2 + ...
        terms: dict[Expr, Number] = {}

        # coefficient (Number or zoo) to always be in slot 0
        # e.g. 3 + ...
        coeff: Expr = S.Zero

        order_factors: list[Order] = []

        extra: list[MatrixExpr] = []

        for o in seq:

            # O(x)
            if o.is_Order:
                if o.expr.is_zero: # type: ignore
                    continue
                if any(o1.contains(o) for o1 in order_factors):
                    continue
                order_factors = [o1 for o1 in order_factors if not o.contains(o1)] # type: ignore
                order_factors = [o] + order_factors # type: ignore
                continue

            # 3 or NaN
            elif o.is_Number:
                if (o is S.NaN or coeff is S.ComplexInfinity and
                        o.is_finite is False) and not extra:
                    # we know for sure the result will be nan
                    return [S.NaN], [], None
                if coeff.is_Number or isinstance(coeff, AccumBounds):
                    coeff += o
                    if coeff is S.NaN and not extra:
                        # we know for sure the result will be nan
                        return [S.NaN], [], None
                continue

            elif isinstance(o, AccumBounds):
                coeff = o.__add__(coeff)
                continue

            elif isinstance(o, MatrixExpr):
                # can't add 0 to Matrix so make sure coeff is not 0
                extra.append(o)
                continue

            elif isinstance(o, TensExpr):
                coeff = TensAdd(o, coeff).doit(deep=False)
                continue

            elif o is S.ComplexInfinity:
                if coeff.is_finite is False and not extra:
                    # we know for sure the result will be nan
                    return [S.NaN], [], None
                coeff = S.ComplexInfinity
                continue

            # Add([...])
            elif o.is_Add:
                # NB: here we assume Add is always commutative
                o_args: tuple[Expr, ...] = o.args # type: ignore
                seq.extend(o_args)  # TODO zerocopy?
                continue

            # Mul([...])
            elif o.is_Mul:
                c, s = o.as_coeff_Mul()

            # check for unevaluated Pow, e.g. 2**3 or 2**(-1/2)
            elif o.is_Pow:
                b, e = o.as_base_exp()
                if b.is_Number and (e.is_Integer or
                                   (e.is_Rational and e.is_negative)):
                    seq.append(b**e)
                    continue
                c, s = S.One, o

            else:
                # everything else
                c = S.One
                s = o

            # now we have:
            # o = c*s, where
            #
            # c is a Number
            # s is an expression with number factor extracted
            # let's collect terms with the same s, so e.g.
            # 2*x**2 + 3*x**2  ->  5*x**2
            if s in terms:
                terms[s] += c
                if terms[s] is S.NaN and not extra:
                    # we know for sure the result will be nan
                    return [S.NaN], [], None
            else:
                terms[s] = c

        # now let's construct new args:
        # [2*x**2, x**3, 7*x**4, pi, ...]
        newseq = []
        noncommutative = False
        for s, c in terms.items():
            # 0*s
            if c.is_zero:
                continue
            # 1*s
            elif c is S.One:
                newseq.append(s)
            # c*s
            else:
                if s.is_Mul:
                    # Mul, already keeps its arguments in perfect order.
                    # so we can simply put c in slot0 and go the fast way.
                    #
                    # XXX: This breaks VectorMul unless it overrides
                    # _new_rawargs
                    cs = s._new_rawargs(*((c,) + s.args)) # type: ignore
                    newseq.append(cs)
                elif s.is_Add:
                    # we just re-create the unevaluated Mul
                    newseq.append(Mul(c, s, evaluate=False))
                else:
                    # alternatively we have to call all Mul's machinery (slow)
                    newseq.append(Mul(c, s))

            noncommutative = noncommutative or not s.is_commutative

        # oo, -oo
        if coeff is S.Infinity:
            newseq = [f for f in newseq if not (f.is_extended_nonnegative or f.is_real)]

        elif coeff is S.NegativeInfinity:
            newseq = [f for f in newseq if not (f.is_extended_nonpositive or f.is_real)]

        if coeff is S.ComplexInfinity:
            # zoo might be
            #   infinite_real + finite_im
            #   finite_real + infinite_im
            #   infinite_real + infinite_im
            # addition of a finite real or imaginary number won't be able to
            # change the zoo nature; adding an infinite qualtity would result
            # in a NaN condition if it had sign opposite of the infinite
            # portion of zoo, e.g., infinite_real - infinite_real.
            newseq = [c for c in newseq if not (c.is_finite and
                                                c.is_extended_real is not None)]

        # process O(x)
        if order_factors:
            newseq2 = []
            for t in newseq:
                # x + O(x) -> O(x)
                if not any(o.contains(t) for o in order_factors):
                    newseq2.append(t)
            newseq = newseq2 + order_factors # type: ignore
            # 1 + O(1) -> O(1)
            for o in order_factors:
                if o.contains(coeff):
                    coeff = S.Zero
                    break

        # order args canonically
        _addsort(newseq)

        # current code expects coeff to be first
        if coeff is not S.Zero:
            newseq.insert(0, coeff)

        if extra:
            newseq += extra
            noncommutative = True

        # we are done
        if noncommutative:
            return [], newseq, None
        else:
            return newseq, [], None

    @classmethod
    def class_key(cls):
        return 3, 1, cls.__name__

    @property
    def kind(self):
        k = attrgetter('kind')
        kinds = map(k, self.args)
        kinds = frozenset(kinds)
        if len(kinds) != 1:
            # Since addition is group operator, kind must be same.
            # We know that this is unexpected signature, so return this.
            result = UndefinedKind
        else:
            result, = kinds
        return result

    def could_extract_minus_sign(self):
        return _could_extract_minus_sign(self)

    @cacheit
    def as_coeff_add(self, *deps):
        """
        Returns a tuple (coeff, args) where self is treated as an Add and coeff
        is the Number term and args is a tuple of all other terms.

        Examples
        ========

        >>> from sympy.abc import x
        >>> (7 + 3*x).as_coeff_add()
        (7, (3*x,))
        >>> (7*x).as_coeff_add()
        (0, (7*x,))
        """
        if deps:
            l1, l2 = sift(self.args, lambda x: x.has_free(*deps), binary=True)
            return self._new_rawargs(*l2), tuple(l1)
        coeff, notrat = self.args[0].as_coeff_add()
        if coeff is not S.Zero:
            return coeff, notrat + self.args[1:]
        return S.Zero, self.args

    def as_coeff_Add(self, rational=False, deps=None) -> tuple[Number, Expr]:
        """
        Efficiently extract the coefficient of a summation.
        """
        coeff, args = self.args[0], self.args[1:]

        if coeff.is_Number and not rational or coeff.is_Rational:
            return coeff, self._new_rawargs(*args) # type: ignore
        return S.Zero, self

    # Note, we intentionally do not implement Add.as_coeff_mul().  Rather, we
    # let Expr.as_coeff_mul() just always return (S.One, self) for an Add.  See
    # issue 5524.

    def _eval_power(self, expt):
        from .evalf import pure_complex
        from .relational import is_eq
        if len(self.args) == 2 and any(_.is_infinite for _ in self.args):
            if expt.is_zero is False and is_eq(expt, S.One) is False:
                # looking for literal a + I*b
                a, b = self.args
                if a.coeff(S.ImaginaryUnit):
                    a, b = b, a
                ico = b.coeff(S.ImaginaryUnit)
                if ico and ico.is_extended_real and a.is_extended_real:
                    if expt.is_extended_negative:
                        return S.Zero
                    if expt.is_extended_positive:
                        return S.ComplexInfinity
            return
        if expt.is_Rational and self.is_number:
            ri = pure_complex(self)
            if ri:
                r, i = ri
                if expt.q == 2:
                    from sympy.functions.elementary.miscellaneous import sqrt
                    D = sqrt(r**2 + i**2)
                    if D.is_Rational:
                        from .exprtools import factor_terms
                        from sympy.functions.elementary.complexes import sign
                        from .function import expand_multinomial
                        # (r, i, D) is a Pythagorean triple
                        root = sqrt(factor_terms((D - r)/2))**expt.p
                        return root*expand_multinomial((
                            # principle value
                            (D + r)/abs(i) + sign(i)*S.ImaginaryUnit)**expt.p)
                elif expt == -1:
                    return _unevaluated_Mul(
                        r - i*S.ImaginaryUnit,
                        1/(r**2 + i**2))

    @cacheit
    def _eval_derivative(self, s):
        return self.func(*[a.diff(s) for a in self.args])

    def _eval_nseries(self, x, n, logx, cdir=0):
        terms = [t.nseries(x, n=n, logx=logx, cdir=cdir) for t in self.args]
        return self.func(*terms)

    def _matches_simple(self, expr, repl_dict):
        # handle (w+3).matches('x+5') -> {w: x+2}
        coeff, terms = self.as_coeff_add()
        if len(terms) == 1:
            return terms[0].matches(expr - coeff, repl_dict)
        return

    def matches(self, expr, repl_dict=None, old=False):
        return self._matches_commutative(expr, repl_dict, old)

    @staticmethod
    def _combine_inverse(lhs, rhs):
        """
        Returns lhs - rhs, but treats oo like a symbol so oo - oo
        returns 0, instead of a nan.
        """
        from sympy.simplify.simplify import signsimp
        inf = (S.Infinity, S.NegativeInfinity)
        if lhs.has(*inf) or rhs.has(*inf):
            from .symbol import Dummy
            oo = Dummy('oo')
            reps = {
                S.Infinity: oo,
                S.NegativeInfinity: -oo}
            ireps = {v: k for k, v in reps.items()}
            eq = lhs.xreplace(reps) - rhs.xreplace(reps)
            if eq.has(oo):
                eq = eq.replace(
                    lambda x: x.is_Pow and x.base is oo,
                    lambda x: x.base)
            rv = eq.xreplace(ireps)
        else:
            rv = lhs - rhs
        srv = signsimp(rv)
        return srv if srv.is_Number else rv

    @cacheit
    def as_two_terms(self):
        """Return head and tail of self.

        This is the most efficient way to get the head and tail of an
        expression.

        - if you want only the head, use self.args[0];
        - if you want to process the arguments of the tail then use
          self.as_coef_add() which gives the head and a tuple containing
          the arguments of the tail when treated as an Add.
        - if you want the coefficient when self is treated as a Mul
          then use self.as_coeff_mul()[0]

        >>> from sympy.abc import x, y
        >>> (3*x - 2*y + 5).as_two_terms()
        (5, 3*x - 2*y)
        """
        return self.args[0], self._new_rawargs(*self.args[1:])

    def as_numer_denom(self) -> tuple[Expr, Expr]:
        """
        Decomposes an expression to its numerator part and its
        denominator part.

        Examples
        ========

        >>> from sympy.abc import x, y, z
        >>> (x*y/z).as_numer_denom()
        (x*y, z)
        >>> (x*(y + 1)/y**7).as_numer_denom()
        (x*(y + 1), y**7)

        See Also
        ========

        sympy.core.expr.Expr.as_numer_denom
        """
        # clear rational denominator
        content, expr = self.primitive()
        if not isinstance(expr, Add):
            return Mul(content, expr, evaluate=False).as_numer_denom()
        ncon, dcon = content.as_numer_denom()

        # collect numerators and denominators of the terms
        nd = defaultdict(list)
        for f in expr.args:
            ni, di = f.as_numer_denom()
            nd[di].append(ni)

        # check for quick exit
        if len(nd) == 1:
            d, n = nd.popitem()
            return self.func(
                *[_keep_coeff(ncon, ni) for ni in n]), _keep_coeff(dcon, d)

        # sum up the terms having a common denominator
        nd2 = {d: self.func(*n) if len(n) > 1 else n[0] for d, n in nd.items()}

        # assemble single numerator and denominator
        denoms, numers = [list(i) for i in zip(*iter(nd2.items()))]
        n, d = self.func(*[Mul(*(denoms[:i] + [numers[i]] + denoms[i + 1:]))
                   for i in range(len(numers))]), Mul(*denoms)

        return _keep_coeff(ncon, n), _keep_coeff(dcon, d)

    def _eval_is_polynomial(self, syms):
        return all(term._eval_is_polynomial(syms) for term in self.args)

    def _eval_is_rational_function(self, syms):
        return all(term._eval_is_rational_function(syms) for term in self.args)

    def _eval_is_meromorphic(self, x, a):
        return _fuzzy_group((arg.is_meromorphic(x, a) for arg in self.args),
                            quick_exit=True)

    def _eval_is_algebraic_expr(self, syms):
        return all(term._eval_is_algebraic_expr(syms) for term in self.args)

    # assumption methods
    _eval_is_real = lambda self: _fuzzy_group(
        (a.is_real for a in self.args), quick_exit=True)
    _eval_is_extended_real = lambda self: _fuzzy_group(
        (a.is_extended_real for a in self.args), quick_exit=True)
    _eval_is_complex = lambda self: _fuzzy_group(
        (a.is_complex for a in self.args), quick_exit=True)
    _eval_is_antihermitian = lambda self: _fuzzy_group(
        (a.is_antihermitian for a in self.args), quick_exit=True)
    _eval_is_finite = lambda self: _fuzzy_group(
        (a.is_finite for a in self.args), quick_exit=True)
    _eval_is_hermitian = lambda self: _fuzzy_group(
        (a.is_hermitian for a in self.args), quick_exit=True)
    _eval_is_integer = lambda self: _fuzzy_group(
        (a.is_integer for a in self.args), quick_exit=True)
    _eval_is_rational = lambda self: _fuzzy_group(
        (a.is_rational for a in self.args), quick_exit=True)
    _eval_is_algebraic = lambda self: _fuzzy_group(
        (a.is_algebraic for a in self.args), quick_exit=True)
    _eval_is_commutative = lambda self: _fuzzy_group(
        a.is_commutative for a in self.args)

    def _eval_is_infinite(self):
        sawinf = False
        for a in self.args:
            ainf = a.is_infinite
            if ainf is None:
                return None
            elif ainf is True:
                # infinite+infinite might not be infinite
                if sawinf is True:
                    return None
                sawinf = True
        return sawinf

    def _eval_is_imaginary(self):
        nz = []
        im_I = []
        for a in self.args:
            if a.is_extended_real:
                if a.is_zero:
                    pass
                elif a.is_zero is False:
                    nz.append(a)
                else:
                    return
            elif a.is_imaginary:
                im_I.append(a*S.ImaginaryUnit)
            elif a.is_Mul and S.ImaginaryUnit in a.args:
                coeff, ai = a.as_coeff_mul(S.ImaginaryUnit)
                if ai == (S.ImaginaryUnit,) and coeff.is_extended_real:
                    im_I.append(-coeff)
                else:
                    return
            else:
                return
        b = self.func(*nz)
        if b != self:
            if b.is_zero:
                return fuzzy_not(self.func(*im_I).is_zero)
            elif b.is_zero is False:
                return False

    def _eval_is_zero(self):
        if self.is_commutative is False:
            # issue 10528: there is no way to know if a nc symbol
            # is zero or not
            return
        nz = []
        z = 0
        im_or_z = False
        im = 0
        for a in self.args:
            if a.is_extended_real:
                if a.is_zero:
                    z += 1
                elif a.is_zero is False:
                    nz.append(a)
                else:
                    return
            elif a.is_imaginary:
                im += 1
            elif a.is_Mul and S.ImaginaryUnit in a.args:
                coeff, ai = a.as_coeff_mul(S.ImaginaryUnit)
                if ai == (S.ImaginaryUnit,) and coeff.is_extended_real:
                    im_or_z = True
                else:
                    return
            else:
                return
        if z == len(self.args):
            return True
        if len(nz) in [0, len(self.args)]:
            return None
        b = self.func(*nz)
        if b.is_zero:
            if not im_or_z:
                if im == 0:
                    return True
                elif im == 1:
                    return False
        if b.is_zero is False:
            return False

    def _eval_is_odd(self):
        l = [f for f in self.args if not (f.is_even is True)]
        if not l:
            return False
        if l[0].is_odd:
            return self._new_rawargs(*l[1:]).is_even

    def _eval_is_irrational(self):
        for t in self.args:
            a = t.is_irrational
            if a:
                others = list(self.args)
                others.remove(t)
                if all(x.is_rational is True for x in others):
                    return True
                return None
            if a is None:
                return
        return False

    def _all_nonneg_or_nonppos(self):
        nn = np = 0
        for a in self.args:
            if a.is_nonnegative:
                if np:
                    return False
                nn = 1
            elif a.is_nonpositive:
                if nn:
                    return False
                np = 1
            else:
                break
        else:
            return True

    def _eval_is_extended_positive(self):
        if self.is_number:
            return super()._eval_is_extended_positive()
        c, a = self.as_coeff_Add()
        if not c.is_zero:
            from .exprtools import _monotonic_sign
            v = _monotonic_sign(a)
            if v is not None:
                s = v + c
                if s != self and s.is_extended_positive and a.is_extended_nonnegative:
                    return True
                if len(self.free_symbols) == 1:
                    v = _monotonic_sign(self)
                    if v is not None and v != self and v.is_extended_positive:
                        return True
        pos = nonneg = nonpos = unknown_sign = False
        saw_INF = set()
        args = [a for a in self.args if not a.is_zero]
        if not args:
            return False
        for a in args:
            ispos = a.is_extended_positive
            infinite = a.is_infinite
            if infinite:
                saw_INF.add(fuzzy_or((ispos, a.is_extended_nonnegative)))
                if True in saw_INF and False in saw_INF:
                    return
            if ispos:
                pos = True
                continue
            elif a.is_extended_nonnegative:
                nonneg = True
                continue
            elif a.is_extended_nonpositive:
                nonpos = True
                continue

            if infinite is None:
                return
            unknown_sign = True

        if saw_INF:
            if len(saw_INF) > 1:
                return
            return saw_INF.pop()
        elif unknown_sign:
            return
        elif not nonpos and not nonneg and pos:
            return True
        elif not nonpos and pos:
            return True
        elif not pos and not nonneg:
            return False

    def _eval_is_extended_nonnegative(self):
        if not self.is_number:
            c, a = self.as_coeff_Add()
            if not c.is_zero and a.is_extended_nonnegative:
                from .exprtools import _monotonic_sign
                v = _monotonic_sign(a)
                if v is not None:
                    s = v + c
                    if s != self and s.is_extended_nonnegative:
                        return True
                    if len(self.free_symbols) == 1:
                        v = _monotonic_sign(self)
                        if v is not None and v != self and v.is_extended_nonnegative:
                            return True

    def _eval_is_extended_nonpositive(self):
        if not self.is_number:
            c, a = self.as_coeff_Add()
            if not c.is_zero and a.is_extended_nonpositive:
                from .exprtools import _monotonic_sign
                v = _monotonic_sign(a)
                if v is not None:
                    s = v + c
                    if s != self and s.is_extended_nonpositive:
                        return True
                    if len(self.free_symbols) == 1:
                        v = _monotonic_sign(self)
                        if v is not None and v != self and v.is_extended_nonpositive:
                            return True

    def _eval_is_extended_negative(self):
        if self.is_number:
            return super()._eval_is_extended_negative()
        c, a = self.as_coeff_Add()
        if not c.is_zero:
            from .exprtools import _monotonic_sign
            v = _monotonic_sign(a)
            if v is not None:
                s = v + c
                if s != self and s.is_extended_negative and a.is_extended_nonpositive:
                    return True
                if len(self.free_symbols) == 1:
                    v = _monotonic_sign(self)
                    if v is not None and v != self and v.is_extended_negative:
                        return True
        neg = nonpos = nonneg = unknown_sign = False
        saw_INF = set()
        args = [a for a in self.args if not a.is_zero]
        if not args:
            return False
        for a in args:
            isneg = a.is_extended_negative
            infinite = a.is_infinite
            if infinite:
                saw_INF.add(fuzzy_or((isneg, a.is_extended_nonpositive)))
                if True in saw_INF and False in saw_INF:
                    return
            if isneg:
                neg = True
                continue
            elif a.is_extended_nonpositive:
                nonpos = True
                continue
            elif a.is_extended_nonnegative:
                nonneg = True
                continue

            if infinite is None:
                return
            unknown_sign = True

        if saw_INF:
            if len(saw_INF) > 1:
                return
            return saw_INF.pop()
        elif unknown_sign:
            return
        elif not nonneg and not nonpos and neg:
            return True
        elif not nonneg and neg:
            return True
        elif not neg and not nonpos:
            return False

    def _eval_subs(self, old, new):
        if not old.is_Add:
            if old is S.Infinity and -old in self.args:
                # foo - oo is foo + (-oo) internally
                return self.xreplace({-old: -new})
            return None

        coeff_self, terms_self = self.as_coeff_Add()
        coeff_old, terms_old = old.as_coeff_Add()

        if coeff_self.is_Rational and coeff_old.is_Rational:
            if terms_self == terms_old:   # (2 + a).subs( 3 + a, y) -> -1 + y
                return self.func(new, coeff_self, -coeff_old)
            if terms_self == -terms_old:  # (2 + a).subs(-3 - a, y) -> -1 - y
                return self.func(-new, coeff_self, coeff_old)

        if coeff_self.is_Rational and coeff_old.is_Rational \
                or coeff_self == coeff_old:
            args_old, args_self = self.func.make_args(
                terms_old), self.func.make_args(terms_self)
            if len(args_old) < len(args_self):  # (a+b+c).subs(b+c,x) -> a+x
                self_set = set(args_self)
                old_set = set(args_old)

                if old_set < self_set:
                    ret_set = self_set - old_set
                    return self.func(new, coeff_self, -coeff_old,
                               *[s._subs(old, new) for s in ret_set])

                args_old = self.func.make_args(
                    -terms_old)     # (a+b+c+d).subs(-b-c,x) -> a-x+d
                old_set = set(args_old)
                if old_set < self_set:
                    ret_set = self_set - old_set
                    return self.func(-new, coeff_self, coeff_old,
                               *[s._subs(old, new) for s in ret_set])

    def removeO(self):
        args = [a for a in self.args if not a.is_Order]
        return self._new_rawargs(*args)

    def getO(self):
        args = [a for a in self.args if a.is_Order]
        if args:
            return self._new_rawargs(*args)

    @cacheit
    def extract_leading_order(self, symbols, point=None):
        """
        Returns the leading term and its order.

        Examples
        ========

        >>> from sympy.abc import x
        >>> (x + 1 + 1/x**5).extract_leading_order(x)
        ((x**(-5), O(x**(-5))),)
        >>> (1 + x).extract_leading_order(x)
        ((1, O(1)),)
        >>> (x + x**2).extract_leading_order(x)
        ((x, O(x)),)

        """
        from sympy.series.order import Order
        lst = []
        symbols = list(symbols if is_sequence(symbols) else [symbols])
        if not point:
            point = [0]*len(symbols)
        seq = [(f, Order(f, *zip(symbols, point))) for f in self.args]
        for ef, of in seq:
            for e, o in lst:
                if o.contains(of) and o != of:
                    of = None
                    break
            if of is None:
                continue
            new_lst = [(ef, of)]
            for e, o in lst:
                if of.contains(o) and o != of:
                    continue
                new_lst.append((e, o))
            lst = new_lst
        return tuple(lst)

    def as_real_imag(self, deep=True, **hints):
        """
        Return a tuple representing a complex number.

        Examples
        ========

        >>> from sympy import I
        >>> (7 + 9*I).as_real_imag()
        (7, 9)
        >>> ((1 + I)/(1 - I)).as_real_imag()
        (0, 1)
        >>> ((1 + 2*I)*(1 + 3*I)).as_real_imag()
        (-5, 5)
        """
        sargs = self.args
        re_part, im_part = [], []
        for term in sargs:
            re, im = term.as_real_imag(deep=deep)
            re_part.append(re)
            im_part.append(im)
        return (self.func(*re_part), self.func(*im_part))

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.core.symbol import Dummy, Symbol
        from sympy.series.order import Order
        from sympy.functions.elementary.exponential import log
        from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold
        from .function import expand_mul

        o = self.getO()
        if o is None:
            o = Order(0)
        old = self.removeO()

        if old.has(Piecewise):
            old = piecewise_fold(old)

        # This expansion is the last part of expand_log. expand_log also calls
        # expand_mul with factor=True, which would be more expensive
        if any(isinstance(a, log) for a in self.args):
            logflags = {"deep": True, "log": True, "mul": False, "power_exp": False,
                "power_base": False, "multinomial": False, "basic": False, "force": False,
                "factor": False}
            old = old.expand(**logflags)
        expr = expand_mul(old)

        if not expr.is_Add:
            return expr.as_leading_term(x, logx=logx, cdir=cdir)

        infinite = [t for t in expr.args if t.is_infinite]

        _logx = Dummy('logx') if logx is None else logx
        leading_terms = [t.as_leading_term(x, logx=_logx, cdir=cdir) for t in expr.args]

        min, new_expr = Order(0), S.Zero

        try:
            for term in leading_terms:
                order = Order(term, x)
                if not min or order not in min:
                    min = order
                    new_expr = term
                elif min in order:
                    new_expr += term

        except TypeError:
            return expr

        if logx is None:
            new_expr = new_expr.subs(_logx, log(x))

        is_zero = new_expr.is_zero
        if is_zero is None:
            new_expr = new_expr.trigsimp().cancel()
            is_zero = new_expr.is_zero
        if is_zero is True:
            # simple leading term analysis gave us cancelled terms but we have to send
            # back a term, so compute the leading term (via series)
            try:
                n0 = min.getn()
            except NotImplementedError:
                n0 = S.One
            if n0.has(Symbol):
                n0 = S.One
            res = Order(1)
            incr = S.One
            while res.is_Order:
                res = old._eval_nseries(x, n=n0+incr, logx=logx, cdir=cdir).cancel().powsimp().trigsimp()
                incr *= 2
            return res.as_leading_term(x, logx=logx, cdir=cdir)

        elif new_expr is S.NaN:
            return old.func._from_args(infinite) + o

        else:
            return new_expr

    def _eval_adjoint(self):
        return self.func(*[t.adjoint() for t in self.args])

    def _eval_conjugate(self):
        return self.func(*[t.conjugate() for t in self.args])

    def _eval_transpose(self):
        return self.func(*[t.transpose() for t in self.args])

    def primitive(self):
        """
        Return ``(R, self/R)`` where ``R``` is the Rational GCD of ``self```.

        ``R`` is collected only from the leading coefficient of each term.

        Examples
        ========

        >>> from sympy.abc import x, y

        >>> (2*x + 4*y).primitive()
        (2, x + 2*y)

        >>> (2*x/3 + 4*y/9).primitive()
        (2/9, 3*x + 2*y)

        >>> (2*x/3 + 4.2*y).primitive()
        (1/3, 2*x + 12.6*y)

        No subprocessing of term factors is performed:

        >>> ((2 + 2*x)*x + 2).primitive()
        (1, x*(2*x + 2) + 2)

        Recursive processing can be done with the ``as_content_primitive()``
        method:

        >>> ((2 + 2*x)*x + 2).as_content_primitive()
        (2, x*(x + 1) + 1)

        See also: primitive() function in polytools.py

        """

        terms = []
        inf = False
        for a in self.args:
            c, m = a.as_coeff_Mul()
            if not c.is_Rational:
                c = S.One
                m = a
            inf = inf or m is S.ComplexInfinity
            terms.append((c.p, c.q, m))

        if not inf:
            ngcd = reduce(igcd, [t[0] for t in terms], 0)
            dlcm = reduce(ilcm, [t[1] for t in terms], 1)
        else:
            ngcd = reduce(igcd, [t[0] for t in terms if t[1]], 0)
            dlcm = reduce(ilcm, [t[1] for t in terms if t[1]], 1)

        if ngcd == dlcm == 1:
            return S.One, self
        if not inf:
            for i, (p, q, term) in enumerate(terms):
                terms[i] = _keep_coeff(Rational((p//ngcd)*(dlcm//q)), term)
        else:
            for i, (p, q, term) in enumerate(terms):
                if q:
                    terms[i] = _keep_coeff(Rational((p//ngcd)*(dlcm//q)), term)
                else:
                    terms[i] = _keep_coeff(Rational(p, q), term)

        # we don't need a complete re-flattening since no new terms will join
        # so we just use the same sort as is used in Add.flatten. When the
        # coefficient changes, the ordering of terms may change, e.g.
        #     (3*x, 6*y) -> (2*y, x)
        #
        # We do need to make sure that term[0] stays in position 0, however.
        #
        if terms[0].is_Number or terms[0] is S.ComplexInfinity:
            c = terms.pop(0)
        else:
            c = None
        _addsort(terms)
        if c:
            terms.insert(0, c)
        return Rational(ngcd, dlcm), self._new_rawargs(*terms)

    def as_content_primitive(self, radical=False, clear=True):
        """Return the tuple (R, self/R) where R is the positive Rational
        extracted from self. If radical is True (default is False) then
        common radicals will be removed and included as a factor of the
        primitive expression.

        Examples
        ========

        >>> from sympy import sqrt
        >>> (3 + 3*sqrt(2)).as_content_primitive()
        (3, 1 + sqrt(2))

        Radical content can also be factored out of the primitive:

        >>> (2*sqrt(2) + 4*sqrt(10)).as_content_primitive(radical=True)
        (2, sqrt(2)*(1 + 2*sqrt(5)))

        See docstring of Expr.as_content_primitive for more examples.
        """
        con, prim = self.func(*[_keep_coeff(*a.as_content_primitive(
            radical=radical, clear=clear)) for a in self.args]).primitive()
        if not clear and not con.is_Integer and prim.is_Add:
            con, d = con.as_numer_denom()
            _p = prim/d
            if any(a.as_coeff_Mul()[0].is_Integer for a in _p.args):
                prim = _p
            else:
                con /= d
        if radical and prim.is_Add:
            # look for common radicals that can be removed
            args = prim.args
            rads = []
            common_q = None
            for m in args:
                term_rads = defaultdict(list)
                for ai in Mul.make_args(m):
                    if ai.is_Pow:
                        b, e = ai.as_base_exp()
                        if e.is_Rational and b.is_Integer:
                            term_rads[e.q].append(abs(int(b))**e.p)
                if not term_rads:
                    break
                if common_q is None:
                    common_q = set(term_rads.keys())
                else:
                    common_q = common_q & set(term_rads.keys())
                    if not common_q:
                        break
                rads.append(term_rads)
            else:
                # process rads
                # keep only those in common_q
                for r in rads:
                    for q in list(r.keys()):
                        if q not in common_q:
                            r.pop(q)
                    for q in r:
                        r[q] = Mul(*r[q])
                # find the gcd of bases for each q
                G = []
                for q in common_q:
                    g = reduce(igcd, [r[q] for r in rads], 0)
                    if g != 1:
                        G.append(g**Rational(1, q))
                if G:
                    G = Mul(*G)
                    args = [ai/G for ai in args]
                    prim = G*prim.func(*args)

        return con, prim

    @property
    def _sorted_args(self):
        from .sorting import default_sort_key
        return tuple(sorted(self.args, key=default_sort_key))

    def _eval_difference_delta(self, n, step):
        from sympy.series.limitseq import difference_delta as dd
        return self.func(*[dd(a, n, step) for a in self.args])

    @property
    def _mpc_(self):
        """
        Convert self to an mpmath mpc if possible
        """
        from .numbers import Float
        re_part, rest = self.as_coeff_Add()
        im_part, imag_unit = rest.as_coeff_Mul()
        if not imag_unit == S.ImaginaryUnit:
            # ValueError may seem more reasonable but since it's a @property,
            # we need to use AttributeError to keep from confusing things like
            # hasattr.
            raise AttributeError("Cannot convert Add to mpc. Must be of the form Number + Number*I")

        return (Float(re_part)._mpf_, Float(im_part)._mpf_)

    def __neg__(self):
        if not global_parameters.distribute:
            return super().__neg__()
        return Mul(S.NegativeOne, self)

add = AssocOpDispatcher('add')

from .mul import Mul, _keep_coeff, _unevaluated_Mul
from .numbers import Rational
<|endoftext|>from collections import defaultdict

from .sympify import sympify, SympifyError
from sympy.utilities.iterables import iterable, uniq


__all__ = ['default_sort_key', 'ordered']


def default_sort_key(item, order=None):
    """Return a key that can be used for sorting.

    The key has the structure:

    (class_key, (len(args), args), exponent.sort_key(), coefficient)

    This key is supplied by the sort_key routine of Basic objects when
    ``item`` is a Basic object or an object (other than a string) that
    sympifies to a Basic object. Otherwise, this function produces the
    key.

    The ``order`` argument is passed along to the sort_key routine and is
    used to determine how the terms *within* an expression are ordered.
    (See examples below) ``order`` options are: 'lex', 'grlex', 'grevlex',
    and reversed values of the same (e.g. 'rev-lex'). The default order
    value is None (which translates to 'lex').

    Examples
    ========

    >>> from sympy import S, I, default_sort_key, sin, cos, sqrt
    >>> from sympy.core.function import UndefinedFunction
    >>> from sympy.abc import x

    The following are equivalent ways of getting the key for an object:

    >>> x.sort_key() == default_sort_key(x)
    True

    Here are some examples of the key that is produced:

    >>> default_sort_key(UndefinedFunction('f'))
    ((0, 0, 'UndefinedFunction'), (1, ('f',)), ((1, 0, 'Number'),
        (0, ()), (), 1), 1)
    >>> default_sort_key('1')
    ((0, 0, 'str'), (1, ('1',)), ((1, 0, 'Number'), (0, ()), (), 1), 1)
    >>> default_sort_key(S.One)
    ((1, 0, 'Number'), (0, ()), (), 1)
    >>> default_sort_key(2)
    ((1, 0, 'Number'), (0, ()), (), 2)

    While sort_key is a method only defined for SymPy objects,
    default_sort_key will accept anything as an argument so it is
    more robust as a sorting key. For the following, using key=
    lambda i: i.sort_key() would fail because 2 does not have a sort_key
    method; that's why default_sort_key is used. Note, that it also
    handles sympification of non-string items likes ints:

    >>> a = [2, I, -I]
    >>> sorted(a, key=default_sort_key)
    [2, -I, I]

    The returned key can be used anywhere that a key can be specified for
    a function, e.g. sort, min, max, etc...:

    >>> a.sort(key=default_sort_key); a[0]
    2
    >>> min(a, key=default_sort_key)
    2

    Notes
    =====

    The key returned is useful for getting items into a canonical order
    that will be the same across platforms. It is not directly useful for
    sorting lists of expressions:

    >>> a, b = x, 1/x

    Since ``a`` has only 1 term, its value of sort_key is unaffected by
    ``order``:

    >>> a.sort_key() == a.sort_key('rev-lex')
    True

    If ``a`` and ``b`` are combined then the key will differ because there
    are terms that can be ordered:

    >>> eq = a + b
    >>> eq.sort_key() == eq.sort_key('rev-lex')
    False
    >>> eq.as_ordered_terms()
    [x, 1/x]
    >>> eq.as_ordered_terms('rev-lex')
    [1/x, x]

    But since the keys for each of these terms are independent of ``order``'s
    value, they do not sort differently when they appear separately in a list:

    >>> sorted(eq.args, key=default_sort_key)
    [1/x, x]
    >>> sorted(eq.args, key=lambda i: default_sort_key(i, order='rev-lex'))
    [1/x, x]

    The order of terms obtained when using these keys is the order that would
    be obtained if those terms were *factors* in a product.

    Although it is useful for quickly putting expressions in canonical order,
    it does not sort expressions based on their complexity defined by the
    number of operations, power of variables and others:

    >>> sorted([sin(x)*cos(x), sin(x)], key=default_sort_key)
    [sin(x)*cos(x), sin(x)]
    >>> sorted([x, x**2, sqrt(x), x**3], key=default_sort_key)
    [sqrt(x), x, x**2, x**3]

    See Also
    ========

    ordered, sympy.core.expr.Expr.as_ordered_factors, sympy.core.expr.Expr.as_ordered_terms

    """
    from .basic import Basic
    from .singleton import S

    if isinstance(item, Basic):
        return item.sort_key(order=order)

    if iterable(item, exclude=str):
        if isinstance(item, dict):
            args = item.items()
            unordered = True
        elif isinstance(item, set):
            args = item
            unordered = True
        else:
            # e.g. tuple, list
            args = list(item)
            unordered = False

        args = [default_sort_key(arg, order=order) for arg in args]

        if unordered:
            # e.g. dict, set
            args = sorted(args)

        cls_index, args = 10, (len(args), tuple(args))
    else:
        if not isinstance(item, str):
            try:
                item = sympify(item, strict=True)
            except SympifyError:
                # e.g. lambda x: x
                pass
            else:
                if isinstance(item, Basic):
                    # e.g int -> Integer
                    return default_sort_key(item)
                # e.g. UndefinedFunction

        # e.g. str
        cls_index, args = 0, (1, (str(item),))

    return (cls_index, 0, item.__class__.__name__
            ), args, S.One.sort_key(), S.One


def _node_count(e):
    # this not only counts nodes, it affirms that the
    # args are Basic (i.e. have an args property). If
    # some object has a non-Basic arg, it needs to be
    # fixed since it is intended that all Basic args
    # are of Basic type (though this is not easy to enforce).
    if e.is_Float:
        return 0.5
    return 1 + sum(map(_node_count, e.args))


def _nodes(e):
    """
    A helper for ordered() which returns the node count of ``e`` which
    for Basic objects is the number of Basic nodes in the expression tree
    but for other objects is 1 (unless the object is an iterable or dict
    for which the sum of nodes is returned).
    """
    from .basic import Basic
    from .function import Derivative

    if isinstance(e, Basic):
        if isinstance(e, Derivative):
            return _nodes(e.expr) + sum(i[1] if i[1].is_Number else
                _nodes(i[1]) for i in e.variable_count)
        return _node_count(e)
    elif iterable(e):
        return 1 + sum(_nodes(ei) for ei in e)
    elif isinstance(e, dict):
        return 1 + sum(_nodes(k) + _nodes(v) for k, v in e.items())
    else:
        return 1


def ordered(seq, keys=None, default=True, warn=False):
    """Return an iterator of the seq where keys are used to break ties
    in a conservative fashion: if, after applying a key, there are no
    ties then no other keys will be computed.

    Two default keys will be applied if 1) keys are not provided or
    2) the given keys do not resolve all ties (but only if ``default``
    is True). The two keys are ``_nodes`` (which places smaller
    expressions before large) and ``default_sort_key`` which (if the
    ``sort_key`` for an object is defined properly) should resolve
    any ties. This strategy is similar to sorting done by
    ``Basic.compare``, but differs in that ``ordered`` never makes a
    decision based on an objects name.

    If ``warn`` is True then an error will be raised if there were no
    keys remaining to break ties. This can be used if it was expected that
    there should be no ties between items that are not identical.

    Examples
    ========

    >>> from sympy import ordered, count_ops
    >>> from sympy.abc import x, y

    The count_ops is not sufficient to break ties in this list and the first
    two items appear in their original order (i.e. the sorting is stable):

    >>> list(ordered([y + 2, x + 2, x**2 + y + 3],
    ...    count_ops, default=False, warn=False))
    ...
    [y + 2, x + 2, x**2 + y + 3]

    The default_sort_key allows the tie to be broken:

    >>> list(ordered([y + 2, x + 2, x**2 + y + 3]))
    ...
    [x + 2, y + 2, x**2 + y + 3]

    Here, sequences are sorted by length, then sum:

    >>> seq, keys = [[[1, 2, 1], [0, 3, 1], [1, 1, 3], [2], [1]], [
    ...    lambda x: len(x),
    ...    lambda x: sum(x)]]
    ...
    >>> list(ordered(seq, keys, default=False, warn=False))
    [[1], [2], [1, 2, 1], [0, 3, 1], [1, 1, 3]]

    If ``warn`` is True, an error will be raised if there were not
    enough keys to break ties:

    >>> list(ordered(seq, keys, default=False, warn=True))
    Traceback (most recent call last):
    ...
    ValueError: not enough keys to break ties


    Notes
    =====

    The decorated sort is one of the fastest ways to sort a sequence for
    which special item comparison is desired: the sequence is decorated,
    sorted on the basis of the decoration (e.g. making all letters lower
    case) and then undecorated. If one wants to break ties for items that
    have the same decorated value, a second key can be used. But if the
    second key is expensive to compute then it is inefficient to decorate
    all items with both keys: only those items having identical first key
    values need to be decorated. This function applies keys successively
    only when needed to break ties. By yielding an iterator, use of the
    tie-breaker is delayed as long as possible.

    This function is best used in cases when use of the first key is
    expected to be a good hashing function; if there are no unique hashes
    from application of a key, then that key should not have been used. The
    exception, however, is that even if there are many collisions, if the
    first group is small and one does not need to process all items in the
    list then time will not be wasted sorting what one was not interested
    in. For example, if one were looking for the minimum in a list and
    there were several criteria used to define the sort order, then this
    function would be good at returning that quickly if the first group
    of candidates is small relative to the number of items being processed.

    """

    d = defaultdict(list)
    if keys:
        if isinstance(keys, (list, tuple)):
            keys = list(keys)
            f = keys.pop(0)
        else:
            f = keys
            keys = []
        for a in seq:
            d[f(a)].append(a)
    else:
        if not default:
            raise ValueError('if default=False then keys must be provided')
        d[None].extend(seq)

    for k, value in sorted(d.items()):
        if len(value) > 1:
            if keys:
                value = ordered(value, keys, default, warn)
            elif default:
                value = ordered(value, (_nodes, default_sort_key,),
                               default=False, warn=warn)
            elif warn:
                u = list(uniq(value))
                if len(u) > 1:
                    raise ValueError(
                        'not enough keys to break ties: %s' % u)
        yield from value
<|endoftext|>"""
There are three types of functions implemented in SymPy:

    1) defined functions (in the sense that they can be evaluated) like
       exp or sin; they have a name and a body:
           f = exp
    2) undefined function which have a name but no body. Undefined
       functions can be defined using a Function class as follows:
           f = Function('f')
       (the result will be a Function instance)
    3) anonymous function (or lambda function) which have a body (defined
       with dummy variables) but have no name:
           f = Lambda(x, exp(x)*x)
           f = Lambda((x, y), exp(x)*y)
    The fourth type of functions are composites, like (sin + cos)(x); these work in
    SymPy core, but are not yet part of SymPy.

    Examples
    ========

    >>> import sympy
    >>> f = sympy.Function("f")
    >>> from sympy.abc import x
    >>> f(x)
    f(x)
    >>> print(sympy.srepr(f(x).func))
    Function('f')
    >>> f(x).args
    (x,)

"""

from __future__ import annotations

from typing import Any
from collections.abc import Iterable
import copyreg

from .add import Add
from .basic import Basic, _atomic
from .cache import cacheit
from .containers import Tuple, Dict
from .decorators import _sympifyit
from .evalf import pure_complex
from .expr import Expr, AtomicExpr
from .logic import fuzzy_and, fuzzy_or, fuzzy_not, FuzzyBool
from .mul import Mul
from .numbers import Rational, Float, Integer
from .operations import LatticeOp
from .parameters import global_parameters
from .rules import Transform
from .singleton import S
from .sympify import sympify, _sympify

from .sorting import default_sort_key, ordered
from sympy.utilities.exceptions import (sympy_deprecation_warning,
                                        SymPyDeprecationWarning, ignore_warnings)
from sympy.utilities.iterables import (has_dups, sift, iterable,
    is_sequence, uniq, topological_sort)
from sympy.utilities.lambdify import MPMATH_TRANSLATIONS
from sympy.utilities.misc import as_int, filldedent, func_name

import mpmath
from mpmath.libmp.libmpf import prec_to_dps

import inspect
from collections import Counter

def _coeff_isneg(a):
    """Return True if the leading Number is negative.

    Examples
    ========

    >>> from sympy.core.function import _coeff_isneg
    >>> from sympy import S, Symbol, oo, pi
    >>> _coeff_isneg(-3*pi)
    True
    >>> _coeff_isneg(S(3))
    False
    >>> _coeff_isneg(-oo)
    True
    >>> _coeff_isneg(Symbol('n', negative=True)) # coeff is 1
    False

    For matrix expressions:

    >>> from sympy import MatrixSymbol, sqrt
    >>> A = MatrixSymbol("A", 3, 3)
    >>> _coeff_isneg(-sqrt(2)*A)
    True
    >>> _coeff_isneg(sqrt(2)*A)
    False
    """

    if a.is_MatMul:
        a = a.args[0]
    if a.is_Mul:
        a = a.args[0]
    return a.is_Number and a.is_extended_negative


class PoleError(Exception):
    pass


class ArgumentIndexError(ValueError):
    def __str__(self):
        return ("Invalid operation with argument number %s for Function %s" %
               (self.args[1], self.args[0]))


class BadSignatureError(TypeError):
    '''Raised when a Lambda is created with an invalid signature'''
    pass


class BadArgumentsError(TypeError):
    '''Raised when a Lambda is called with an incorrect number of arguments'''
    pass


# Python 3 version that does not raise a Deprecation warning
def arity(cls):
    """Return the arity of the function if it is known, else None.

    Explanation
    ===========

    When default values are specified for some arguments, they are
    optional and the arity is reported as a tuple of possible values.

    Examples
    ========

    >>> from sympy import arity, log
    >>> arity(lambda x: x)
    1
    >>> arity(log)
    (1, 2)
    >>> arity(lambda *x: sum(x)) is None
    True
    """
    eval_ = getattr(cls, 'eval', cls)

    parameters = inspect.signature(eval_).parameters.items()
    if [p for _, p in parameters if p.kind == p.VAR_POSITIONAL]:
        return
    p_or_k = [p for _, p in parameters if p.kind == p.POSITIONAL_OR_KEYWORD]
    # how many have no default and how many have a default value
    no, yes = map(len, sift(p_or_k,
        lambda p:p.default == p.empty, binary=True))
    return no if not yes else tuple(range(no, no + yes + 1))

class FunctionClass(type):
    """
    Base class for function classes. FunctionClass is a subclass of type.

    Use Function('<function name>' [ , signature ]) to create
    undefined function classes.
    """
    _new = type.__new__

    def __init__(cls, *args, **kwargs):
        # honor kwarg value or class-defined value before using
        # the number of arguments in the eval function (if present)
        nargs = kwargs.pop('nargs', cls.__dict__.get('nargs', arity(cls)))
        if nargs is None and 'nargs' not in cls.__dict__:
            for supcls in cls.__mro__:
                if hasattr(supcls, '_nargs'):
                    nargs = supcls._nargs
                    break
                else:
                    continue

        # Canonicalize nargs here; change to set in nargs.
        if is_sequence(nargs):
            if not nargs:
                raise ValueError(filldedent('''
                    Incorrectly specified nargs as %s:
                    if there are no arguments, it should be
                    `nargs = 0`;
                    if there are any number of arguments,
                    it should be
                    `nargs = None`''' % str(nargs)))
            nargs = tuple(ordered(set(nargs)))
        elif nargs is not None:
            nargs = (as_int(nargs),)
        cls._nargs = nargs

        # When __init__ is called from UndefinedFunction it is called with
        # just one arg but when it is called from subclassing Function it is
        # called with the usual (name, bases, namespace) type() signature.
        if len(args) == 3:
            namespace = args[2]
            if 'eval' in namespace and not isinstance(namespace['eval'], classmethod):
                raise TypeError("eval on Function subclasses should be a class method (defined with @classmethod)")

    @property
    def __signature__(self):
        """
        Allow Python 3's inspect.signature to give a useful signature for
        Function subclasses.
        """
        # Python 3 only, but backports (like the one in IPython) still might
        # call this.
        try:
            from inspect import signature
        except ImportError:
            return None

        # TODO: Look at nargs
        return signature(self.eval)

    @property
    def free_symbols(self):
        return set()

    @property
    def xreplace(self):
        # Function needs args so we define a property that returns
        # a function that takes args...and then use that function
        # to return the right value
        return lambda rule, **_: rule.get(self, self)

    @property
    def nargs(self):
        """Return a set of the allowed number of arguments for the function.

        Examples
        ========

        >>> from sympy import Function
        >>> f = Function('f')

        If the function can take any number of arguments, the set of whole
        numbers is returned:

        >>> Function('f').nargs
        Naturals0

        If the function was initialized to accept one or more arguments, a
        corresponding set will be returned:

        >>> Function('f', nargs=1).nargs
        {1}
        >>> Function('f', nargs=(2, 1)).nargs
        {1, 2}

        The undefined function, after application, also has the nargs
        attribute; the actual number of arguments is always available by
        checking the ``args`` attribute:

        >>> f = Function('f')
        >>> f(1).nargs
        Naturals0
        >>> len(f(1).args)
        1
        """
        from sympy.sets.sets import FiniteSet
        # XXX it would be nice to handle this in __init__ but there are import
        # problems with trying to import FiniteSet there
        return FiniteSet(*self._nargs) if self._nargs else S.Naturals0

    def _valid_nargs(self, n : int) -> bool:
        """ Return True if the specified integer is a valid number of arguments

        The number of arguments n is guaranteed to be an integer and positive

        """
        if self._nargs:
            return n in self._nargs

        nargs = self.nargs
        return nargs is S.Naturals0 or n in nargs

    def __repr__(cls):
        return cls.__name__


class Application(Basic, metaclass=FunctionClass):
    """
    Base class for applied functions.

    Explanation
    ===========

    Instances of Application represent the result of applying an application of
    any type to any object.
    """

    is_Function = True

    @cacheit
    def __new__(cls, *args, **options):
        from sympy.sets.fancysets import Naturals0
        from sympy.sets.sets import FiniteSet

        args = list(map(sympify, args))
        evaluate = options.pop('evaluate', global_parameters.evaluate)
        # WildFunction (and anything else like it) may have nargs defined
        # and we throw that value away here
        options.pop('nargs', None)

        if options:
            raise ValueError("Unknown options: %s" % options)

        if evaluate:
            evaluated = cls.eval(*args)
            if evaluated is not None:
                return evaluated

        obj = super().__new__(cls, *args, **options)

        # make nargs uniform here
        sentinel = object()
        objnargs = getattr(obj, "nargs", sentinel)
        if objnargs is not sentinel:
            # things passing through here:
            #  - functions subclassed from Function (e.g. myfunc(1).nargs)
            #  - functions like cos(1).nargs
            #  - AppliedUndef with given nargs like Function('f', nargs=1)(1).nargs
            # Canonicalize nargs here
            if is_sequence(objnargs):
                nargs = tuple(ordered(set(objnargs)))
            elif objnargs is not None:
                nargs = (as_int(objnargs),)
            else:
                nargs = None
        else:
            # things passing through here:
            #  - WildFunction('f').nargs
            #  - AppliedUndef with no nargs like Function('f')(1).nargs
            nargs = obj._nargs  # note the underscore here
        # convert to FiniteSet
        obj.nargs = FiniteSet(*nargs) if nargs else Naturals0()
        return obj

    @classmethod
    def eval(cls, *args):
        """
        Returns a canonical form of cls applied to arguments args.

        Explanation
        ===========

        The ``eval()`` method is called when the class ``cls`` is about to be
        instantiated and it should return either some simplified instance
        (possible of some other class), or if the class ``cls`` should be
        unmodified, return None.

        Examples of ``eval()`` for the function "sign"

        .. code-block:: python

            @classmethod
            def eval(cls, arg):
                if arg is S.NaN:
                    return S.NaN
                if arg.is_zero: return S.Zero
                if arg.is_positive: return S.One
                if arg.is_negative: return S.NegativeOne
                if isinstance(arg, Mul):
                    coeff, terms = arg.as_coeff_Mul(rational=True)
                    if coeff is not S.One:
                        return cls(coeff) * cls(terms)

        """
        return

    @property
    def func(self):
        return self.__class__

    def _eval_subs(self, old, new):
        if (old.is_Function and new.is_Function and
            callable(old) and callable(new) and
            old == self.func and len(self.args) in new.nargs):
            return new(*[i._subs(old, new) for i in self.args])


class Function(Application, Expr):
    r"""
    Base class for applied mathematical functions.

    It also serves as a constructor for undefined function classes.

    See the :ref:`custom-functions` guide for details on how to subclass
    ``Function`` and what methods can be defined.

    Examples
    ========

    **Undefined Functions**

    To create an undefined function, pass a string of the function name to
    ``Function``.

    >>> from sympy import Function, Symbol
    >>> x = Symbol('x')
    >>> f = Function('f')
    >>> g = Function('g')(x)
    >>> f
    f
    >>> f(x)
    f(x)
    >>> g
    g(x)
    >>> f(x).diff(x)
    Derivative(f(x), x)
    >>> g.diff(x)
    Derivative(g(x), x)

    Assumptions can be passed to ``Function`` the same as with a
    :class:`~.Symbol`. Alternatively, you can use a ``Symbol`` with
    assumptions for the function name and the function will inherit the name
    and assumptions associated with the ``Symbol``:

    >>> f_real = Function('f', real=True)
    >>> f_real(x).is_real
    True
    >>> f_real_inherit = Function(Symbol('f', real=True))
    >>> f_real_inherit(x).is_real
    True

    Note that assumptions on a function are unrelated to the assumptions on
    the variables it is called on. If you want to add a relationship, subclass
    ``Function`` and define custom assumptions handler methods. See the
    :ref:`custom-functions-assumptions` section of the :ref:`custom-functions`
    guide for more details.

    **Custom Function Subclasses**

    The :ref:`custom-functions` guide has several
    :ref:`custom-functions-complete-examples` of how to subclass ``Function``
    to create a custom function.

    """

    @property
    def _diff_wrt(self):
        return False

    @cacheit
    def __new__(cls, *args, **options) -> type[AppliedUndef]:  # type: ignore
        # Handle calls like Function('f')
        if cls is Function:
            return UndefinedFunction(*args, **options)  # type: ignore
        else:
            return cls._new_(*args, **options)  # type: ignore

    @classmethod
    def _new_(cls, *args, **options) -> Expr:
        n = len(args)

        if not cls._valid_nargs(n):
            # XXX: exception message must be in exactly this format to
            # make it work with NumPy's functions like vectorize(). See,
            # for example, https://github.com/numpy/numpy/issues/1697.
            # The ideal solution would be just to attach metadata to
            # the exception and change NumPy to take advantage of this.
            temp = ('%(name)s takes %(qual)s %(args)s '
                   'argument%(plural)s (%(given)s given)')
            raise TypeError(temp % {
                'name': cls,
                'qual': 'exactly' if len(cls.nargs) == 1 else 'at least',
                'args': min(cls.nargs),
                'plural': 's'*(min(cls.nargs) != 1),
                'given': n})

        evaluate = options.get('evaluate', global_parameters.evaluate)
        result = super().__new__(cls, *args, **options)
        if evaluate and isinstance(result, cls) and result.args:
            _should_evalf = [cls._should_evalf(a) for a in result.args]
            pr2 = min(_should_evalf)
            if pr2 > 0:
                pr = max(_should_evalf)
                result = result.evalf(prec_to_dps(pr))

        return _sympify(result)

    @classmethod
    def _should_evalf(cls, arg):
        """
        Decide if the function should automatically evalf().

        Explanation
        ===========

        By default (in this implementation), this happens if (and only if) the
        ARG is a floating point number (including complex numbers).
        This function is used by __new__.

        Returns the precision to evalf to, or -1 if it should not evalf.
        """
        if arg.is_Float:
            return arg._prec
        if not arg.is_Add:
            return -1
        m = pure_complex(arg)
        if m is None:
            return -1
        # the elements of m are of type Number, so have a _prec
        return max(m[0]._prec, m[1]._prec)

    @classmethod
    def class_key(cls):
        from sympy.sets.fancysets import Naturals0
        funcs = {
            'exp': 10,
            'log': 11,
            'sin': 20,
            'cos': 21,
            'tan': 22,
            'cot': 23,
            'sinh': 30,
            'cosh': 31,
            'tanh': 32,
            'coth': 33,
            'conjugate': 40,
            're': 41,
            'im': 42,
            'arg': 43,
        }
        name = cls.__name__

        try:
            i = funcs[name]
        except KeyError:
            i = 0 if isinstance(cls.nargs, Naturals0) else 10000

        return 4, i, name

    def _eval_evalf(self, prec):

        def _get_mpmath_func(fname):
            """Lookup mpmath function based on name"""
            if isinstance(self, AppliedUndef):
                # Shouldn't lookup in mpmath but might have ._imp_
                return None

            if not hasattr(mpmath, fname):
                fname = MPMATH_TRANSLATIONS.get(fname, None)
                if fname is None:
                    return None
            return getattr(mpmath, fname)

        _eval_mpmath = getattr(self, '_eval_mpmath', None)
        if _eval_mpmath is None:
            func = _get_mpmath_func(self.func.__name__)
            args = self.args
        else:
            func, args = _eval_mpmath()

        # Fall-back evaluation
        if func is None:
            imp = getattr(self, '_imp_', None)
            if imp is None:
                return None
            try:
                return Float(imp(*[i.evalf(prec) for i in self.args]), prec)
            except (TypeError, ValueError):
                return None

        # Convert all args to mpf or mpc
        # Convert the arguments to *higher* precision than requested for the
        # final result.
        # XXX + 5 is a guess, it is similar to what is used in evalf.py. Should
        #     we be more intelligent about it?
        try:
            args = [arg._to_mpmath(prec + 5) for arg in args]
            def bad(m):
                from mpmath import mpf, mpc
                # the precision of an mpf value is the last element
                # if that is 1 (and m[1] is not 1 which would indicate a
                # power of 2), then the eval failed; so check that none of
                # the arguments failed to compute to a finite precision.
                # Note: An mpc value has two parts, the re and imag tuple;
                # check each of those parts, too. Anything else is allowed to
                # pass
                if isinstance(m, mpf):
                    m = m._mpf_
                    return m[1] !=1 and m[-1] == 1
                elif isinstance(m, mpc):
                    m, n = m._mpc_
                    return m[1] !=1 and m[-1] == 1 and \
                        n[1] !=1 and n[-1] == 1
                else:
                    return False
            if any(bad(a) for a in args):
                raise ValueError  # one or more args failed to compute with significance
        except ValueError:
            return

        with mpmath.workprec(prec):
            v = func(*args)

        return Expr._from_mpmath(v, prec)

    def _eval_derivative(self, s):
        # f(x).diff(s) -> x.diff(s) * f.fdiff(1)(s)
        i = 0
        l = []
        for a in self.args:
            i += 1
            da = a.diff(s)
            if da.is_zero:
                continue
            try:
                df = self.fdiff(i)
            except ArgumentIndexError:
                df = Function.fdiff(self, i)
            l.append(df * da)
        return Add(*l)

    def _eval_is_commutative(self):
        return fuzzy_and(a.is_commutative for a in self.args)

    def _eval_is_meromorphic(self, x, a):
        if not self.args:
            return True
        if any(arg.has(x) for arg in self.args[1:]):
            return False

        arg = self.args[0]
        if not arg._eval_is_meromorphic(x, a):
            return None

        return fuzzy_not(type(self).is_singular(arg.subs(x, a)))

    _singularities: FuzzyBool | tuple[Expr, ...] = None

    @classmethod
    def is_singular(cls, a):
        """
        Tests whether the argument is an essential singularity
        or a branch point, or the functions is non-holomorphic.
        """
        ss = cls._singularities
        if ss in (True, None, False):
            return ss

        return fuzzy_or(a.is_infinite if s is S.ComplexInfinity
                        else (a - s).is_zero for s in ss)

    def _eval_aseries(self, n, args0, x, logx):
        """
        Compute an asymptotic expansion around args0, in terms of self.args.
        This function is only used internally by _eval_nseries and should not
        be called directly; derived classes can overwrite this to implement
        asymptotic expansions.
        """
        raise PoleError(filldedent('''
            Asymptotic expansion of %s around %s is
            not implemented.''' % (type(self), args0)))

    def _eval_nseries(self, x, n, logx, cdir=0):
        """
        This function does compute series for multivariate functions,
        but the expansion is always in terms of *one* variable.

        Examples
        ========

        >>> from sympy import atan2
        >>> from sympy.abc import x, y
        >>> atan2(x, y).series(x, n=2)
        atan2(0, y) + x/y + O(x**2)
        >>> atan2(x, y).series(y, n=2)
        -y/x + atan2(x, 0) + O(y**2)

        This function also computes asymptotic expansions, if necessary
        and possible:

        >>> from sympy import loggamma
        >>> loggamma(1/x)._eval_nseries(x,0,None)
        -1/x - log(x)/x + log(x)/2 + O(1)

        """
        from .symbol import uniquely_named_symbol
        from sympy.series.order import Order
        from sympy.sets.sets import FiniteSet
        args = self.args
        args0 = [t.limit(x, 0) for t in args]
        if any(t.is_finite is False for t in args0):
            from .numbers import oo, zoo, nan
            a = [t.as_leading_term(x, logx=logx) for t in args]
            a0 = [t.limit(x, 0) for t in a]
            if any(t.has(oo, -oo, zoo, nan) for t in a0):
                return self._eval_aseries(n, args0, x, logx)
            # Careful: the argument goes to oo, but only logarithmically so. We
            # are supposed to do a power series expansion "around the
            # logarithmic term". e.g.
            #      f(1+x+log(x))
            #     -> f(1+logx) + x*f'(1+logx) + O(x**2)
            # where 'logx' is given in the argument
            a = [t._eval_nseries(x, n, logx) for t in args]
            z = [r - r0 for (r, r0) in zip(a, a0)]
            p = [Dummy() for _ in z]
            q = []
            v = None
            for ai, zi, pi in zip(a0, z, p):
                if zi.has(x):
                    if v is not None:
                        raise NotImplementedError
                    q.append(ai + pi)
                    v = pi
                else:
                    q.append(ai)
            e1 = self.func(*q)
            if v is None:
                return e1
            s = e1._eval_nseries(v, n, logx)
            o = s.getO()
            s = s.removeO()
            s = s.subs(v, zi).expand() + Order(o.expr.subs(v, zi), x)
            return s
        if (self.func.nargs is S.Naturals0
                or (self.func.nargs == FiniteSet(1) and args0[0])
                or any(c > 1 for c in self.func.nargs)):
            e = self
            e1 = e.expand()
            if e == e1:
                #for example when e = sin(x+1) or e = sin(cos(x))
                #let's try the general algorithm
                if len(e.args) == 1:
                    # issue 14411
                    e = e.func(e.args[0].cancel())
                term = e.subs(x, S.Zero)
                if term.is_finite is False or term is S.NaN:
                    raise PoleError("Cannot expand %s around 0" % (self))
                series = term
                fact = S.One

                _x = uniquely_named_symbol('xi', self)
                e = e.subs(x, _x)
                for i in range(1, n):
                    fact *= Rational(i)
                    e = e.diff(_x)
                    subs = e.subs(_x, S.Zero)
                    if subs is S.NaN:
                        # try to evaluate a limit if we have to
                        subs = e.limit(_x, S.Zero)
                    if subs.is_finite is False:
                        raise PoleError("Cannot expand %s around 0" % (self))
                    term = subs*(x**i)/fact
                    term = term.expand()
                    series += term
                return series + Order(x**n, x)
            return e1.nseries(x, n=n, logx=logx)
        arg = self.args[0]
        l = []
        g = None
        # try to predict a number of terms needed
        nterms = n + 2
        cf = Order(arg.as_leading_term(x), x).getn()
        if cf != 0:
            nterms = (n/cf).ceiling()
        for i in range(nterms):
            g = self.taylor_term(i, arg, g)
            g = g.nseries(x, n=n, logx=logx)
            l.append(g)
        return Add(*l) + Order(x**n, x)

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of the function.
        """
        if not (1 <= argindex <= len(self.args)):
            raise ArgumentIndexError(self, argindex)
        ix = argindex - 1
        A = self.args[ix]
        if A._diff_wrt:
            if len(self.args) == 1 or not A.is_Symbol:
                return _derivative_dispatch(self, A)
            for i, v in enumerate(self.args):
                if i != ix and A in v.free_symbols:
                    # it can't be in any other argument's free symbols
                    # issue 8510
                    break
            else:
                return _derivative_dispatch(self, A)

        # See issue 4624 and issue 4719, 5600 and 8510
        D = Dummy('xi_%i' % argindex, dummy_index=hash(A))
        args = self.args[:ix] + (D,) + self.args[ix + 1:]
        return Subs(Derivative(self.func(*args), D), D, A)

    def _eval_as_leading_term(self, x, logx, cdir):
        """Stub that should be overridden by new Functions to return
        the first non-zero term in a series if ever an x-dependent
        argument whose leading term vanishes as x -> 0 might be encountered.
        See, for example, cos._eval_as_leading_term.
        """
        from sympy.series.order import Order
        args = [a.as_leading_term(x, logx=logx) for a in self.args]
        o = Order(1, x)
        if any(x in a.free_symbols and o.contains(a) for a in args):
            # Whereas x and any finite number are contained in O(1, x),
            # expressions like 1/x are not. If any arg simplified to a
            # vanishing expression as x -> 0 (like x or x**2, but not
            # 3, 1/x, etc...) then the _eval_as_leading_term is needed
            # to supply the first non-zero term of the series,
            #
            # e.g. expression    leading term
            #      ----------    ------------
            #      cos(1/x)      cos(1/x)
            #      cos(cos(x))   cos(1)
            #      cos(x)        1        <- _eval_as_leading_term needed
            #      sin(x)        x        <- _eval_as_leading_term needed
            #
            raise NotImplementedError(
                '%s has no _eval_as_leading_term routine' % self.func)
        else:
            return self


class DefinedFunction(Function):
    """Base class for defined functions like ``sin``, ``cos``, ..."""

    @cacheit
    def __new__(cls, *args, **options) -> Expr:  # type: ignore
        return cls._new_(*args, **options)


class AppliedUndef(Function):
    """
    Base class for expressions resulting from the application of an undefined
    function.
    """

    is_number = False

    name: str

    def __new__(cls, *args, **options) -> Expr:  # type: ignore
        args = tuple(map(sympify, args))
        u = [a.name for a in args if isinstance(a, UndefinedFunction)]
        if u:
            raise TypeError('Invalid argument: expecting an expression, not UndefinedFunction%s: %s' % (
                's'*(len(u) > 1), ', '.join(u)))
        obj: Expr = super().__new__(cls, *args, **options)  # type: ignore
        return obj

    def _eval_as_leading_term(self, x, logx, cdir):
        return self

    @property
    def _diff_wrt(self):
        """
        Allow derivatives wrt to undefined functions.

        Examples
        ========

        >>> from sympy import Function, Symbol
        >>> f = Function('f')
        >>> x = Symbol('x')
        >>> f(x)._diff_wrt
        True
        >>> f(x).diff(x)
        Derivative(f(x), x)
        """
        return True


class UndefSageHelper:
    """
    Helper to facilitate Sage conversion.
    """
    def __get__(self, ins, typ):
        import sage.all as sage
        if ins is None:
            return lambda: sage.function(typ.__name__)
        else:
            args = [arg._sage_() for arg in ins.args]
            return lambda : sage.function(ins.__class__.__name__)(*args)

_undef_sage_helper = UndefSageHelper()


class UndefinedFunction(FunctionClass):
    """
    The (meta)class of undefined functions.
    """
    name: str
    _sage_: UndefSageHelper

    def __new__(mcl, name, bases=(AppliedUndef,), __dict__=None, **kwargs) -> type[AppliedUndef]:
        from .symbol import _filter_assumptions
        # Allow Function('f', real=True)
        # and/or Function(Symbol('f', real=True))
        assumptions, kwargs = _filter_assumptions(kwargs)
        if isinstance(name, Symbol):
            assumptions = name._merge(assumptions)
            name = name.name
        elif not isinstance(name, str):
            raise TypeError('expecting string or Symbol for name')
        else:
            commutative = assumptions.get('commutative', None)
            assumptions = Symbol(name, **assumptions).assumptions0
            if commutative is None:
                assumptions.pop('commutative')
        __dict__ = __dict__ or {}
        # put the `is_*` for into __dict__
        __dict__.update({'is_%s' % k: v for k, v in assumptions.items()})
        # You can add other attributes, although they do have to be hashable
        # (but seriously, if you want to add anything other than assumptions,
        # just subclass Function)
        __dict__.update(kwargs)
        # add back the sanitized assumptions without the is_ prefix
        kwargs.update(assumptions)
        # Save these for __eq__
        __dict__.update({'_kwargs': kwargs})
        # do this for pickling
        __dict__['__module__'] = None
        obj = super().__new__(mcl, name, bases, __dict__)  # type: ignore
        obj.name = name
        obj._sage_ = _undef_sage_helper
        return obj  # type: ignore

    def __instancecheck__(cls, instance):
        return cls in type(instance).__mro__

    _kwargs: dict[str, bool | None] = {}

    def __hash__(self):
        return hash((self.class_key(), frozenset(self._kwargs.items())))

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
            self.class_key() == other.class_key() and
            self._kwargs == other._kwargs)

    def __ne__(self, other):
        return not self == other

    @property
    def _diff_wrt(self):
        return False


# Using copyreg is the only way to make a dynamically generated instance of a
# metaclass picklable without using a custom pickler. It is not possible to
# define e.g. __reduce__ on the metaclass because obj.__reduce__ will retrieve
# the __reduce__ method for reducing instances of the type rather than for the
# type itself.
def _reduce_undef(f):
    return (_rebuild_undef, (f.name, f._kwargs))

def _rebuild_undef(name, kwargs):
    return Function(name, **kwargs)

copyreg.pickle(UndefinedFunction, _reduce_undef)


# XXX: The type: ignore on WildFunction is because mypy complains:
#
# sympy/core/function.py:939: error: Cannot determine type of 'sort_key' in
# base class 'Expr'
#
# Somehow this is because of the @cacheit decorator but it is not clear how to
# fix it.


class WildFunction(Function, AtomicExpr):  # type: ignore
    """
    A WildFunction function matches any function (with its arguments).

    Examples
    ========

    >>> from sympy import WildFunction, Function, cos
    >>> from sympy.abc import x, y
    >>> F = WildFunction('F')
    >>> f = Function('f')
    >>> F.nargs
    Naturals0
    >>> x.match(F)
    >>> F.match(F)
    {F_: F_}
    >>> f(x).match(F)
    {F_: f(x)}
    >>> cos(x).match(F)
    {F_: cos(x)}
    >>> f(x, y).match(F)
    {F_: f(x, y)}

    To match functions with a given number of arguments, set ``nargs`` to the
    desired value at instantiation:

    >>> F = WildFunction('F', nargs=2)
    >>> F.nargs
    {2}
    >>> f(x).match(F)
    >>> f(x, y).match(F)
    {F_: f(x, y)}

    To match functions with a range of arguments, set ``nargs`` to a tuple
    containing the desired number of arguments, e.g. if ``nargs = (1, 2)``
    then functions with 1 or 2 arguments will be matched.

    >>> F = WildFunction('F', nargs=(1, 2))
    >>> F.nargs
    {1, 2}
    >>> f(x).match(F)
    {F_: f(x)}
    >>> f(x, y).match(F)
    {F_: f(x, y)}
    >>> f(x, y, 1).match(F)

    """

    # XXX: What is this class attribute used for?
    include: set[Any] = set()

    def __init__(cls, name, **assumptions):
        from sympy.sets.sets import Set, FiniteSet
        cls.name = name
        nargs = assumptions.pop('nargs', S.Naturals0)
        if not isinstance(nargs, Set):
            # Canonicalize nargs here.  See also FunctionClass.
            if is_sequence(nargs):
                nargs = tuple(ordered(set(nargs)))
            elif nargs is not None:
                nargs = (as_int(nargs),)
            nargs = FiniteSet(*nargs)
        cls.nargs = nargs

    def matches(self, expr, repl_dict=None, old=False):
        if not isinstance(expr, (AppliedUndef, Function)):
            return None
        if len(expr.args) not in self.nargs:
            return None

        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        repl_dict[self] = expr
        return repl_dict


class Derivative(Expr):
    """
    Carries out differentiation of the given expression with respect to symbols.

    Examples
    ========

    >>> from sympy import Derivative, Function, symbols, Subs
    >>> from sympy.abc import x, y
    >>> f, g = symbols('f g', cls=Function)

    >>> Derivative(x**2, x, evaluate=True)
    2*x

    Denesting of derivatives retains the ordering of variables:

        >>> Derivative(Derivative(f(x, y), y), x)
        Derivative(f(x, y), y, x)

    Contiguously identical symbols are merged into a tuple giving
    the symbol and the count:

        >>> Derivative(f(x), x, x, y, x)
        Derivative(f(x), (x, 2), y, x)

    If the derivative cannot be performed, and evaluate is True, the
    order of the variables of differentiation will be made canonical:

        >>> Derivative(f(x, y), y, x, evaluate=True)
        Derivative(f(x, y), x, y)

    Derivatives with respect to undefined functions can be calculated:

        >>> Derivative(f(x)**2, f(x), evaluate=True)
        2*f(x)

    Such derivatives will show up when the chain rule is used to
    evaluate a derivative:

        >>> f(g(x)).diff(x)
        Derivative(f(g(x)), g(x))*Derivative(g(x), x)

    Substitution is used to represent derivatives of functions with
    arguments that are not symbols or functions:

        >>> f(2*x + 3).diff(x) == 2*Subs(f(y).diff(y), y, 2*x + 3)
        True

    Notes
    =====

    Simplification of high-order derivatives:

    Because there can be a significant amount of simplification that can be
    done when multiple differentiations are performed, results will be
    automatically simplified in a fairly conservative fashion unless the
    keyword ``simplify`` is set to False.

        >>> from sympy import sqrt, diff, Function, symbols
        >>> from sympy.abc import x, y, z
        >>> f, g = symbols('f,g', cls=Function)

        >>> e = sqrt((x + 1)**2 + x)
        >>> diff(e, (x, 5), simplify=False).count_ops()
        136
        >>> diff(e, (x, 5)).count_ops()
        30

    Ordering of variables:

    If evaluate is set to True and the expression cannot be evaluated, the
    list of differentiation symbols will be sorted, that is, the expression is
    assumed to have continuous derivatives up to the order asked.

    Derivative wrt non-Symbols:

    For the most part, one may not differentiate wrt non-symbols.
    For example, we do not allow differentiation wrt `x*y` because
    there are multiple ways of structurally defining where x*y appears
    in an expression: a very strict definition would make
    (x*y*z).diff(x*y) == 0. Derivatives wrt defined functions (like
    cos(x)) are not allowed, either:

        >>> (x*y*z).diff(x*y)
        Traceback (most recent call last):
        ...
        ValueError: Can't calculate derivative wrt x*y.

    To make it easier to work with variational calculus, however,
    derivatives wrt AppliedUndef and Derivatives are allowed.
    For example, in the Euler-Lagrange method one may write
    F(t, u, v) where u = f(t) and v = f'(t). These variables can be
    written explicitly as functions of time::

        >>> from sympy.abc import t
        >>> F = Function('F')
        >>> U = f(t)
        >>> V = U.diff(t)

    The derivative wrt f(t) can be obtained directly:

        >>> direct = F(t, U, V).diff(U)

    When differentiation wrt a non-Symbol is attempted, the non-Symbol
    is temporarily converted to a Symbol while the differentiation
    is performed and the same answer is obtained:

        >>> indirect = F(t, U, V).subs(U, x).diff(x).subs(x, U)
        >>> assert direct == indirect

    The implication of this non-symbol replacement is that all
    functions are treated as independent of other functions and the
    symbols are independent of the functions that contain them::

        >>> x.diff(f(x))
        0
        >>> g(x).diff(f(x))
        0

    It also means that derivatives are assumed to depend only
    on the variables of differentiation, not on anything contained
    within the expression being differentiated::

        >>> F = f(x)
        >>> Fx = F.diff(x)
        >>> Fx.diff(F)  # derivative depends on x, not F
        0
        >>> Fxx = Fx.diff(x)
        >>> Fxx.diff(Fx)  # derivative depends on x, not Fx
        0

    The last example can be made explicit by showing the replacement
    of Fx in Fxx with y:

        >>> Fxx.subs(Fx, y)
        Derivative(y, x)

        Since that in itself will evaluate to zero, differentiating
        wrt Fx will also be zero:

        >>> _.doit()
        0

    Replacing undefined functions with concrete expressions

    One must be careful to replace undefined functions with expressions
    that contain variables consistent with the function definition and
    the variables of differentiation or else insconsistent result will
    be obtained. Consider the following example:

    >>> eq = f(x)*g(y)
    >>> eq.subs(f(x), x*y).diff(x, y).doit()
    y*Derivative(g(y), y) + g(y)
    >>> eq.diff(x, y).subs(f(x), x*y).doit()
    y*Derivative(g(y), y)

    The results differ because `f(x)` was replaced with an expression
    that involved both variables of differentiation. In the abstract
    case, differentiation of `f(x)` by `y` is 0; in the concrete case,
    the presence of `y` made that derivative nonvanishing and produced
    the extra `g(y)` term.

    Defining differentiation for an object

    An object must define ._eval_derivative(symbol) method that returns
    the differentiation result. This function only needs to consider the
    non-trivial case where expr contains symbol and it should call the diff()
    method internally (not _eval_derivative); Derivative should be the only
    one to call _eval_derivative.

    Any class can allow derivatives to be taken with respect to
    itself (while indicating its scalar nature). See the
    docstring of Expr._diff_wrt.

    See Also
    ========
    _sort_variable_count
    """

    is_Derivative = True

    @property
    def _diff_wrt(self):
        """An expression may be differentiated wrt a Derivative if
        it is in elementary form.

        Examples
        ========

        >>> from sympy import Function, Derivative, cos
        >>> from sympy.abc import x
        >>> f = Function('f')

        >>> Derivative(f(x), x)._diff_wrt
        True
        >>> Derivative(cos(x), x)._diff_wrt
        False
        >>> Derivative(x + 1, x)._diff_wrt
        False

        A Derivative might be an unevaluated form of what will not be
        a valid variable of differentiation if evaluated. For example,

        >>> Derivative(f(f(x)), x).doit()
        Derivative(f(x), x)*Derivative(f(f(x)), f(x))

        Such an expression will present the same ambiguities as arise
        when dealing with any other product, like ``2*x``, so ``_diff_wrt``
        is False:

        >>> Derivative(f(f(x)), x)._diff_wrt
        False
        """
        return self.expr._diff_wrt and isinstance(self.doit(), Derivative)

    def __new__(cls, expr, *variables, **kwargs):
        expr = sympify(expr)
        if not isinstance(expr, Basic):
            raise TypeError(f"Cannot represent derivative of {type(expr)}")
        symbols_or_none = getattr(expr, "free_symbols", None)
        has_symbol_set = isinstance(symbols_or_none, set)

        if not has_symbol_set:
            raise ValueError(filldedent('''
                Since there are no variables in the expression %s,
                it cannot be differentiated.''' % expr))

        # determine value for variables if it wasn't given
        if not variables:
            variables = expr.free_symbols
            if len(variables) != 1:
                if expr.is_number:
                    return S.Zero
                if len(variables) == 0:
                    raise ValueError(filldedent('''
                        Since there are no variables in the expression,
                        the variable(s) of differentiation must be supplied
                        to differentiate %s''' % expr))
                else:
                    raise ValueError(filldedent('''
                        Since there is more than one variable in the
                        expression, the variable(s) of differentiation
                        must be supplied to differentiate %s''' % expr))

        # Split the list of variables into a list of the variables we are diff
        # wrt, where each element of the list has the form (s, count) where
        # s is the entity to diff wrt and count is the order of the
        # derivative.
        variable_count = []
        array_likes = (tuple, list, Tuple)

        from sympy.tensor.array import Array, NDimArray

        for i, v in enumerate(variables):
            if isinstance(v, UndefinedFunction):
                raise TypeError(
                    "cannot differentiate wrt "
                    "UndefinedFunction: %s" % v)

            if isinstance(v, array_likes):
                if len(v) == 0:
                    # Ignore empty tuples: Derivative(expr, ... , (), ... )
                    continue
                if isinstance(v[0], array_likes):
                    # Derive by array: Derivative(expr, ... , [[x, y, z]], ... )
                    if len(v) == 1:
